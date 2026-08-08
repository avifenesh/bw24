#!/usr/bin/env bash
# Final placement-policy battery on box2.
#
# Proves:
#   1. run-spec K=1..8 remains self-consistent on single-card and PP-2;
#   2. PP-2's naked serving default admits no spec while single-card still does;
#   3. MEMRA_SPEC_GATE=0 keeps the formerly fatal #87 PP-2 path live and clean;
#   4. the repository's full serve-smoke battery remains green.
set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"

LANE=research/specplace-20260808
TS=${SPECPLACE_GATE_TS:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${SPECPLACE_GATE_DIR:-$LANE/gates/$TS}
mkdir -p "$OUT"

DRIVER=$OUT/driver.log
SUMMARY=$OUT/SUMMARY.md
POINTS=$OUT/points.jsonl
PORT=${SPECPLACE_GATE_PORT:-8131}
ADDR=127.0.0.1:$PORT
BASE=http://$ADDR

Q9=${Q9:-/data/models/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf}
Q9_DRAFT=${Q9_DRAFT:-/data/models/draft-9b-owntrim-nvfp4head-q4blk.gguf}
TARGET_DIR=$(realpath -m "${CARGO_TARGET_DIR:-$ROOT/target}")
SERVER=$TARGET_DIR/release/memra-server
RUNSPEC=$TARGET_DIR/release/run-spec

FAILS=0
CLEANUP_PID=
TARGET_LINK_CREATED=0

stop_server() {
    local pid=${1:-}
    test -n "$pid" || return 0
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 60); do
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null || true
            return 0
        fi
        sleep 1
    done
    kill -9 "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}

# shellcheck disable=SC2329 # invoked through EXIT/INT/TERM traps
cleanup() {
    stop_server "$CLEANUP_PID"
    if ((TARGET_LINK_CREATED)); then
        rm -f "$ROOT/target"
    fi
}
trap cleanup EXIT INT TERM

exec > >(tee "$DRIVER") 2>&1

fail() {
    echo "FAIL: $*"
    FAILS=$((FAILS + 1))
}

pass() {
    echo "PASS: $*"
}

run_capture() {
    local log=$1
    shift
    echo "=== run: $log ==="
    printf 'command:'
    printf ' %q' "$@"
    printf '\n'
    set +e
    "$@" > "$log" 2>&1
    local rc=$?
    set -e
    cat "$log"
    return "$rc"
}

wait_up() {
    local pid=$1
    for _ in $(seq 1 240); do
        if curl -sf "$BASE/readyz" >/dev/null 2>&1; then
            return 0
        fi
        kill -0 "$pid" 2>/dev/null || return 1
        sleep 2
    done
    return 1
}

wait_worker_idle() {
    local label=$1
    local metrics idle
    for _ in $(seq 1 100); do
        metrics=$(curl -sf "$BASE/metrics" 2>/dev/null || true)
        if [[ -n "$metrics" ]] && python3 -c '
import json
import sys

metrics = json.load(sys.stdin)
raise SystemExit(0 if metrics.get("serve_idle_seconds", 0.0) >= 0.5 else 1)
' <<< "$metrics"; then
            idle=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["serve_idle_seconds"])' \
                <<< "$metrics")
            echo "$label worker quiesced (serve_idle_seconds=$idle)"
            return 0
        fi
        sleep 0.1
    done
    return 1
}

port_free() {
    ! ss -tln 2>/dev/null | grep -qE "[:.]${1}[[:space:]]"
}

launch_server() {
    local label=$1
    shift
    local log=$OUT/$label-server.log
    local -a command=(
        env
        -u CUDA_VISIBLE_DEVICES
        -u MEMRA_PP_STAGES
        -u MEMRA_PP_DEVICES
        -u MEMRA_PP_SHARD
        -u MEMRA_PP_STREAMS
        -u MEMRA_SERVE_SPEC
        -u MEMRA_SPEC_GATE
        -u MEMRA_SPEC_GATE_LOW
        -u MEMRA_SPEC_GATE_HIGH
        "$@"
        MEMRA_MODELS="q9=${Q9}+${Q9_DRAFT}"
        MEMRA_ADDR="$ADDR"
        MEMRA_CTX=4096
        MEMRA_SPEC_K=3
        "$SERVER"
    )

    if ! port_free "$PORT"; then
        fail "port $PORT already in use before $label"
        ss -tlnp | grep -E "[:.]${PORT}[[:space:]]" || true
        return 1
    fi

    echo "=== server: $label ==="
    printf 'command:'
    printf ' %q' "${command[@]}"
    printf '\n'
    "${command[@]}" > "$log" 2>&1 &
    CLEANUP_PID=$!
    if ! wait_up "$CLEANUP_PID"; then
        fail "$label did not become ready"
        tail -80 "$log" || true
        stop_server "$CLEANUP_PID"
        CLEANUP_PID=
        return 1
    fi
    if ! ss -tlnp 2>/dev/null | grep -E "[:.]${PORT}[[:space:]]" \
            | grep -q "pid=$CLEANUP_PID,"; then
        fail "$label port responder is not child pid $CLEANUP_PID"
        ss -tlnp | grep -E "[:.]${PORT}[[:space:]]" || true
        stop_server "$CLEANUP_PID"
        CLEANUP_PID=
        return 1
    fi
}

finish_server() {
    local label=${1:-server}
    if ! wait_worker_idle "$label"; then
        fail "$label worker did not report 0.5s idle before shutdown"
    fi
    stop_server "$CLEANUP_PID"
    CLEANUP_PID=
    sleep 3
}

load_point() {
    local label=$1 concurrency=$2 requests=$3 warmup=$4
    local log=$OUT/$label-load.log
    if ! run_capture "$log" python3 tools/load-serve.py \
            --base "$BASE" \
            --model q9 \
            --concurrency "$concurrency" \
            --requests "$requests" \
            --max-tokens 96 \
            --greedy \
            --warmup "$warmup" \
            --label "$label" \
            --out "$POINTS"; then
        fail "$label load command"
        return 1
    fi
}

check_point() {
    local label=$1 expected=$2
    if python3 - "$POINTS" "$label" "$expected" <<'PY'
import json
import sys

path, label, expected = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = [json.loads(line) for line in open(path) if line.strip()]
matches = [row for row in rows if row.get("label") == label]
assert len(matches) == 1, f"{label}: expected one row, got {len(matches)}"
row = matches[0]
assert row["n_ok"] == expected, f"{label}: n_ok={row['n_ok']} expected={expected}"
assert row["n_err"] == 0, f"{label}: n_err={row['n_err']}"
assert row["n_shed"] == 0, f"{label}: n_shed={row['n_shed']}"
print(
    f"{label}: {row['n_ok']}/{expected} ok, "
    f"{row['agg_tok_s']:.1f} aggregate tok/s"
)
PY
    then
        pass "$label completion"
    else
        fail "$label completion"
    fi
}

check_runspec() {
    local log=$1 label=$2
    local passes ks
    passes=$(grep -c "self-consistency: PASS" "$log" || true)
    ks=$(grep -cE '^\[generate_spec K=[1-8]\]' "$log" || true)
    if [[ "$passes" -eq 8 && "$ks" -eq 8 ]] \
            && grep -q "=== SELF-CONSISTENCY PASS ===" "$log" \
            && ! grep -q "self-consistency: FAIL" "$log"; then
        pass "$label run-spec K=1..8"
    else
        fail "$label run-spec (K rows=$ks, PASS rows=$passes)"
    fi
}

for artifact in "$Q9" "$Q9_DRAFT"; do
    test -f "$artifact" || {
        echo "FAIL: missing artifact $artifact"
        exit 1
    }
done

echo "=== specplace gate $TS ==="
echo "host=$(hostname) commit=$(git rev-parse HEAD)"
git status --short --untracked-files=no
stat -c '%n %s bytes' "$Q9" "$Q9_DRAFT"
sha256sum "$Q9" "$Q9_DRAFT" > "$OUT/artifact-sha256.txt"

export PATH="$HOME/.cargo/bin:$PATH"
echo "=== release build ==="
{
    cargo build --release -p memra-server
    cargo build --release -p memra-engine --bin run-spec
} > "$OUT/build.log" 2>&1
cat "$OUT/build.log"
sha256sum "$SERVER" "$RUNSPEC" > "$OUT/binary-sha256.txt"

# tools/serve-smoke.sh intentionally invokes target/release/memra-server. Let a
# remote shared target directory satisfy that path without duplicating the build.
if [[ "$TARGET_DIR" != "$ROOT/target" ]]; then
    if [[ -e "$ROOT/target" || -L "$ROOT/target" ]]; then
        if [[ "$(realpath -m "$ROOT/target")" != "$TARGET_DIR" ]]; then
            echo "FAIL: $ROOT/target exists but does not resolve to $TARGET_DIR"
            exit 1
        fi
    else
        ln -s "$TARGET_DIR" "$ROOT/target"
        TARGET_LINK_CREATED=1
    fi
fi

exec 9>/tmp/memra-gpu.lock
flock -w 14400 9 || {
    echo "FAIL: GPU lock timeout"
    exit 75
}
echo "GPU lock acquired $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=index,memory.used,temperature.gpu,clocks.sm,power.draw \
    --format=csv,noheader > "$OUT/gpu-pre.csv"
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name \
    --format=csv,noheader > "$OUT/gpu-processes-pre.csv" 2>&1 || true

echo "=== run-spec placement battery ==="
if run_capture "$OUT/runspec-single.log" \
        env -u MEMRA_PP_STAGES -u MEMRA_PP_DEVICES -u MEMRA_SPEC_K \
        CUDA_VISIBLE_DEVICES=0 \
        MEMRA_QWEN_DC=0 \
        MEMRA_MTP_DRAFT="$Q9_DRAFT" \
        MEMRA_SPEC_TEMP=0 \
        MEMRA_NGEN=64 \
        "$RUNSPEC" "$Q9" 55; then
    check_runspec "$OUT/runspec-single.log" single-card
else
    fail "single-card run-spec command"
fi

for devices in 1,0 0,1; do
    label=pp2-${devices/,/}
    log=$OUT/runspec-$label.log
    if run_capture "$log" \
            env -u CUDA_VISIBLE_DEVICES -u MEMRA_SPEC_K \
            MEMRA_PP_STAGES=2 \
            MEMRA_PP_DEVICES="$devices" \
            MEMRA_QWEN_DC=0 \
            MEMRA_MTP_DRAFT="$Q9_DRAFT" \
            MEMRA_SPEC_TEMP=0 \
            MEMRA_NGEN=64 \
            "$RUNSPEC" "$Q9" 55; then
        check_runspec "$log" "$label"
    else
        fail "$label run-spec command"
    fi
done

echo "=== naked policy liveness ==="
if launch_server policy-pp2-default \
        MEMRA_PP_STAGES=2 \
        MEMRA_PP_DEVICES=1,0; then
    load_point policy-pp2-default-c1 1 4 1 || true
    finish_server policy-pp2-default
    check_point policy-pp2-default-c1 4
    pp2_log=$OUT/policy-pp2-default-server.log
    if grep -q '\[spec-gate\] policy placement=pp2-cross-device LOW=0 HIGH=1 source=placement-default spec-admission=off' "$pp2_log"; then
        pass "PP-2 placement default logged LOW=0/HIGH=1"
    else
        fail "PP-2 placement default log"
    fi
    if grep -q '\[spec-acc\]' "$pp2_log"; then
        fail "PP-2 naked default ran spec"
    else
        pass "PP-2 naked default emitted no spec-acc lines"
    fi
    if grep -q '\[spec-gate\] admit batched' "$pp2_log"; then
        pass "PP-2 naked default admitted batched"
    else
        fail "PP-2 naked default never logged batched admission"
    fi
fi

if launch_server policy-single-default CUDA_VISIBLE_DEVICES=0; then
    load_point policy-single-default-c1 1 4 1 || true
    finish_server policy-single-default
    check_point policy-single-default-c1 4
    single_log=$OUT/policy-single-default-server.log
    if grep -q '\[spec-gate\] policy placement=single-or-non-pp2 LOW=2 HIGH=4 source=placement-default spec-admission=on' "$single_log"; then
        pass "single-card placement default logged LOW=2/HIGH=4"
    else
        fail "single-card placement default log"
    fi
    if grep -q '\[spec-acc\]' "$single_log"; then
        pass "single-card naked default ran spec"
    else
        fail "single-card naked default never ran spec"
    fi
fi

echo "=== #87 quick crash gate: formerly fatal dev10, forced spec ==="
if launch_server crash-pp2-forced-spec \
        MEMRA_PP_STAGES=2 \
        MEMRA_PP_DEVICES=1,0 \
        MEMRA_SPEC_GATE=0; then
    load_point crash-c2 2 8 1 || true
    load_point crash-c4 4 16 0 || true
    load_point crash-recovery-c1 1 4 0 || true
    finish_server crash-pp2-forced-spec
    check_point crash-c2 8
    check_point crash-c4 16
    check_point crash-recovery-c1 4
    crash_log=$OUT/crash-pp2-forced-spec-server.log
    if grep -q '\[spec-gate\] policy disabled by MEMRA_SPEC_GATE=0: always-spec' "$crash_log"; then
        pass "#87 gate forced the always-spec rollback"
    else
        fail "#87 gate rollback policy log"
    fi
    if grep -q '\[spec-acc\]' "$crash_log"; then
        pass "#87 gate exercised spec"
    else
        fail "#87 gate never ran spec"
    fi
    if grep -Eiq 'CUDA_ERROR_(ILLEGAL_ADDRESS|DEINITIALIZED)|#87 trap|argmax sentinel|spec pending flush failed|Xid 31|FAULT_(PDE|PTE)|abort' "$crash_log"; then
        fail "#87 gate saw a fatal/sentinel/teardown-error signature"
    else
        pass "#87 gate fatal/sentinel/teardown-error scan clean"
    fi
fi

echo "=== repository serve-smoke ==="
if port_free 8177; then
    if run_capture "$OUT/serve-smoke.log" \
            env -u MEMRA_PP_STAGES \
                -u MEMRA_PP_DEVICES \
                -u MEMRA_SERVE_SPEC \
                -u MEMRA_SPEC_GATE \
                -u MEMRA_SPEC_GATE_LOW \
                -u MEMRA_SPEC_GATE_HIGH \
                CUDA_VISIBLE_DEVICES=0 \
                tools/serve-smoke.sh "$Q9" "$Q9_DRAFT"; then
        pass "tools/serve-smoke.sh"
    else
        fail "tools/serve-smoke.sh"
    fi
    test ! -f /tmp/serve-smoke.log \
        || cp /tmp/serve-smoke.log "$OUT/serve-smoke-last-server.log"
else
    fail "serve-smoke port 8177 already in use"
fi

nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name \
    --format=csv,noheader > "$OUT/gpu-processes-post.csv" 2>&1 || true
nvidia-smi --query-gpu=index,memory.used,temperature.gpu,clocks.sm,power.draw \
    --format=csv,noheader > "$OUT/gpu-post.csv"
echo "GPU lock released $(date -u +%FT%TZ)"
flock -u 9

{
    echo "# specplace gate summary"
    echo
    echo "- Host: $(hostname)"
    echo "- Commit: $(git rev-parse HEAD)"
    echo "- Script-detected failures: $FAILS"
    echo
    echo "## run-spec"
    grep -H "=== SELF-CONSISTENCY PASS ===" "$OUT"/runspec-*.log \
        | sed "s|$OUT/||" || true
    echo
    echo "## policy"
    grep -H '\[spec-gate\] policy' "$OUT"/policy-*-server.log \
        "$OUT"/crash-pp2-forced-spec-server.log \
        | sed "s|$OUT/||" || true
    echo
    echo "## load points"
    if [[ -f "$POINTS" ]]; then
        python3 - "$POINTS" <<'PY'
import json
import sys

for line in open(sys.argv[1]):
    row = json.loads(line)
    print(
        f"- {row['label']}: ok={row['n_ok']} err={row['n_err']} "
        f"shed={row['n_shed']} agg={row['agg_tok_s']:.1f} tok/s"
    )
PY
    else
        echo "- no load-point file"
    fi
    echo
    echo "## serve-smoke"
    grep -E '^serve-smoke:|^  (ok|FAIL):|^== serve-smoke' "$OUT/serve-smoke.log" || true
} > "$SUMMARY"
cat "$SUMMARY"

echo "SPECPLACE_GATES_DONE failures=$FAILS"
exit "$FAILS"
