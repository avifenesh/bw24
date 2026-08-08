#!/usr/bin/env bash
# Current-train placement sweep on box2.
#
# q9: single-card S/N at c=1/2/4, N=3, to refresh #89's stale crossover.
# step35: PP-2 S/N at c=1/2/4, N=3, to price spec against the newly merged
# batched step35 denominator. PP-2 q9 is not re-measured: the existing N=3/5
# cells are large-margin and are cited in PROGRESS.md.
set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"

LANE=research/specplace-20260808
RAW=$LANE/raw
mkdir -p "$RAW"

TS=${SPECPLACE_TS:-$(date -u +%Y%m%dT%H%M%SZ)}
DRIVER=$RAW/driver-$TS.log
POINTS=$RAW/points-$TS.jsonl
SUMMARY=$RAW/summary-$TS.md
GPU=$RAW/gpu-$TS.csv
PORT=${SPECPLACE_PORT:-8127}
ADDR=127.0.0.1:$PORT
BASE=http://$ADDR
REPS=${SPECPLACE_REPS:-3}
CS=${SPECPLACE_CS:-"1 2 4"}
MAX_TOKENS=${SPECPLACE_MAX_TOKENS:-96}

Q9=${Q9:-/data/models/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf}
Q9_DRAFT=${Q9_DRAFT:-/data/models/draft-9b-owntrim-nvfp4head-q4blk.gguf}
STEP35=${STEP35:-/data/models/step37/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf}
STEP35_DRAFT=${STEP35_DRAFT:-/data/models/step37/step-3.7-flash/Step3.7-flash-mtp-Q8_0.gguf}
BIN=${CARGO_TARGET_DIR:-target}/release/memra-server

exec > >(tee "$DRIVER") 2>&1

for artifact in "$Q9" "$Q9_DRAFT" "$STEP35" "$STEP35_DRAFT"; do
    test -f "$artifact" || { echo "FAIL: missing artifact $artifact"; exit 1; }
done

echo "=== specplace measurement $TS ==="
echo "host=$(hostname) commit=$(git rev-parse HEAD)"
echo "status:"
git status --short
echo "models:"
stat -c '%n %s bytes' "$Q9" "$Q9_DRAFT" "$STEP35" "$STEP35_DRAFT"
echo "sha256 (artifact identity; computed before the GPU window):"
sha256sum "$Q9" "$Q9_DRAFT" "$STEP35" "$STEP35_DRAFT" \
    > "$RAW/artifact-sha256-$TS.txt"
cat "$RAW/artifact-sha256-$TS.txt"

echo "=== release build ==="
cargo build --release -p memra-server > "$RAW/build-$TS.log" 2>&1
cat "$RAW/build-$TS.log"
sha256sum "$BIN" > "$RAW/binary-sha256-$TS.txt"
cat "$RAW/binary-sha256-$TS.txt"

gpu_sample() {
    {
        printf '%s,' "$(date -u +%FT%TZ)"
        nvidia-smi --query-gpu=index,memory.used,temperature.gpu,clocks.sm,power.draw \
            --format=csv,noheader | paste -sd ';' -
    } >> "$GPU"
}

port_free() {
    ! ss -tln 2>/dev/null | grep -qE "[:.]${PORT}[[:space:]]"
}

wait_up() {
    local pid=$1
    for _ in $(seq 1 240); do
        if curl -sf "$BASE/readyz" >/dev/null 2>&1; then
            return 0
        fi
        kill -0 "$pid" 2>/dev/null || return 1
        sleep 5
    done
    return 1
}

stop_server() {
    local pid=$1
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 60); do
        kill -0 "$pid" 2>/dev/null || { wait "$pid" 2>/dev/null || true; return; }
        sleep 1
    done
    kill -9 "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}

serve_arm() {
    local model=$1 placement=$2 arm=$3 rep=$4
    local model_spec label server_log models
    local -a topology policy command

    label="${model}-${placement}-${arm}-r${rep}"
    server_log="$RAW/${label}-server.log"
    case "$model" in
        q9)
            models="q9=${Q9}+${Q9_DRAFT}"
            model_spec=q9
            ;;
        step35)
            models="step35=${STEP35}+${STEP35_DRAFT}"
            model_spec=step35
            ;;
        *)
            echo "FAIL: unknown model $model"
            return 1
            ;;
    esac

    case "$placement" in
        sc)
            topology=(CUDA_VISIBLE_DEVICES=0)
            ;;
        pp2)
            topology=(MEMRA_PP_STAGES=2 "MEMRA_PP_DEVICES=0,1")
            ;;
        *)
            echo "FAIL: unknown placement $placement"
            return 1
            ;;
    esac

    case "$arm" in
        S) policy=(MEMRA_SPEC_GATE=0) ;;
        N) policy=(MEMRA_SERVE_SPEC=0) ;;
        *)
            echo "FAIL: unknown arm $arm"
            return 1
            ;;
    esac

    port_free || {
        echo "FAIL: port $PORT is already listening before $label"
        ss -tlnp | grep -E "[:.]${PORT}[[:space:]]" || true
        return 1
    }

    command=(
        env
        -u CUDA_VISIBLE_DEVICES
        -u MEMRA_PP_STAGES
        -u MEMRA_PP_DEVICES
        -u MEMRA_SERVE_SPEC
        -u MEMRA_SPEC_GATE
        -u MEMRA_SPEC_GATE_LOW
        -u MEMRA_SPEC_GATE_HIGH
        "${topology[@]}"
        "${policy[@]}"
        MEMRA_MODELS="$models"
        MEMRA_ADDR="$ADDR"
        MEMRA_CTX=4096
        MEMRA_SPEC_K=3
        "$BIN"
    )

    echo "=== arm $label ==="
    printf 'command:'
    printf ' %q' "${command[@]}"
    printf '\n'
    "${command[@]}" > "$server_log" 2>&1 &
    local pid=$!
    cleanup_pid=$pid
    if ! wait_up "$pid"; then
        echo "FAIL: $label server did not become ready"
        tail -80 "$server_log" || true
        stop_server "$pid"
        cleanup_pid=
        return 1
    fi
    if ! ss -tlnp 2>/dev/null | grep -E "[:.]${PORT}[[:space:]]" | grep -q "pid=$pid,"; then
        echo "FAIL: $label port responder is not child pid $pid"
        ss -tlnp | grep -E "[:.]${PORT}[[:space:]]" || true
        stop_server "$pid"
        cleanup_pid=
        return 1
    fi

    for concurrency in $CS; do
        local load_log="$RAW/${label}-c${concurrency}.log"
        echo "--- $label c=$concurrency ---"
        set +e
        python3 tools/load-serve.py \
            --base "$BASE" \
            --model "$model_spec" \
            --concurrency "$concurrency" \
            --requests $((concurrency * 4)) \
            --max-tokens "$MAX_TOKENS" \
            --greedy \
            --warmup 1 \
            --label "${label}-c${concurrency}" \
            --out "$POINTS" \
            > "$load_log" 2>&1
        local rc=$?
        set -e
        cat "$load_log"
        gpu_sample
        if ((rc != 0)); then
            echo "FAIL: load arm $label c=$concurrency exited $rc"
            stop_server "$pid"
            cleanup_pid=
            return "$rc"
        fi
    done

    curl -sf "$BASE/metrics" > "$RAW/${label}-metrics.txt" 2>&1 || true
    stop_server "$pid"
    cleanup_pid=
    sleep 3

    local spec_lines
    spec_lines=$(grep -c '\[spec-acc\]' "$server_log" || true)
    echo "arm evidence: $label spec-acc lines=$spec_lines"
    if [[ "$arm" == S && "$spec_lines" -eq 0 ]]; then
        echo "FAIL: forced-spec arm $label never ran spec"
        return 1
    fi
    if [[ "$arm" == N && "$spec_lines" -ne 0 ]]; then
        echo "FAIL: forced-plain arm $label ran spec"
        return 1
    fi
}

run_matrix() {
    local model=$1 placement=$2
    for rep in $(seq 1 "$REPS"); do
        if ((rep % 2 == 1)); then
            serve_arm "$model" "$placement" S "$rep"
            serve_arm "$model" "$placement" N "$rep"
        else
            serve_arm "$model" "$placement" N "$rep"
            serve_arm "$model" "$placement" S "$rep"
        fi
    done
}

cleanup_pid=
trap 'test -z "$cleanup_pid" || stop_server "$cleanup_pid"' EXIT

exec 9>/tmp/memra-gpu.lock
flock -w 14400 9 || { echo "FAIL: GPU lock timeout"; exit 75; }
echo "GPU lock acquired $(date -u +%FT%TZ)"
gpu_sample
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name \
    --format=csv,noheader > "$RAW/gpu-processes-pre-$TS.csv" 2>&1 || true

run_matrix q9 sc
run_matrix step35 pp2

nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name \
    --format=csv,noheader > "$RAW/gpu-processes-post-$TS.csv" 2>&1 || true
gpu_sample
echo "GPU lock released $(date -u +%FT%TZ)"
flock -u 9

python3 "$LANE/analyze.py" "$POINTS" > "$SUMMARY"
cat "$SUMMARY"
echo "SPECPLACE_MEASURE_DONE"
