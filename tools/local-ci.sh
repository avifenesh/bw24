#!/usr/bin/env bash
# memra local CI — the real gate (GitHub CI is compile-only; the rig is the test machine).
#
#   tools/local-ci.sh                correctness stage only (~3 min)
#   tools/local-ci.sh --perf         correctness + full perf battery (~15 min)
#   tools/local-ci.sh --perf-quick   correctness + gemma-31B cells only (~6 min)
#
# Correctness stage: kernel-check, run-gen argmax gate, spec self-consistency,
# VERIFY-GATE logit maxdiff at depth — the standing exactness battery, one command.
#
# Perf stage: the cell battery from research/tune-data/perf-cells.json. Every spec cell
# records tok/s + ACCEPTANCE + tok/round — the drift class that silently cost the spec
# board 2026-07-13..15 (acceptance 1.000 -> 0.669 across ~40 green-gated commits).
# Rows append to research/tune-data/perf-ci.jsonl; each cell is verdicted against the
# rolling median of its last N rows: FAIL on >3% tok/s drop or >0.05 acceptance drop,
# WARN on >1.5%. A FAIL exits non-zero — treat it like a red test.
#
# Contributor machines: cells whose model file is absent are SKIPPED cleanly; the
# correctness stage runs wherever a GPU + at least one model exists. Set
# MEMRA_MODELS_DIR to your model root (default /data/ai-ml/hf-models).
#
# Window discipline (recorded per row, enforced where it can be): no other compute
# process on the GPU (co-resident engines spill experts and read 10x low), host load
# sane, power profile noted (pin it with gpu-full-power on|off — profiles pair fairly
# only against themselves).
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS="${MEMRA_MODELS_DIR:-/data/ai-ml/hf-models}"
MANIFEST=research/tune-data/perf-cells.json
OUT=research/tune-data/perf-ci.jsonl
MODE="${1:---correctness}"

command -v jq >/dev/null || { echo "local-ci: jq required"; exit 2; }
[ -x target/release/kernel-check ] || cargo build --release

# ---- window state ----
# allowed co-residents: embedding servers (tiny, CPU-bound; identified by --embedding in cmdline)
apps=""
while IFS=, read -r pid _name; do
    pid=$(echo "$pid" | tr -d ' '); [ -n "$pid" ] || continue
    if ! tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q -- "--embedding"; then
        apps+="$pid $(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-80)\n"
    fi
done < <(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null)
apps=$(printf "%b" "$apps")
if [ -n "$apps" ]; then
    echo "local-ci: WARNING — other GPU compute apps present (numbers not window-valid):"
    echo "$apps"
    WINDOW_CLEAN=false
else
    WINDOW_CLEAN=true
fi
# Per-cell recheck (2026-07-26): the entry-only check let a co-agent job that joined
# MID-battery silently poison later cells (26b-spec-d1736 read accept 0.656 in a battery
# whose entry was clean; 7 windowed re-runs read 0.846). Cells re-verify the window after
# their reps and retry once instead of recording a contended row as evidence.
window_free_now() {
    local n=0 pid
    while IFS=, read -r pid _; do
        pid=$(echo "$pid" | tr -d ' '); [ -n "$pid" ] || continue
        tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -qE -- "--embedding|llama-server" \
            || n=$((n+1))
    done < <(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null)
    [ "$n" -eq 0 ]
}
LOAD=$(awk '{print $1}' /proc/loadavg)
PROFILE=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null || echo unknown)

echo "== local-ci: correctness stage =="
out=$(target/release/kernel-check 2>&1 | tail -1)
echo "$out" | grep -q "ALL GREEN" || { echo "kernel-check FAIL"; exit 1; }
echo "kernel-check: GREEN"

# prime-gate (#46): batched-prime vs tokenwise first-token agreement on the mixed prompt
# set — near-tie flips report, structured divergence or non-determinism exits non-zero.
Q35="$MODELS/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"
if [ -f "$Q35" ]; then
    if ! target/release/prime-gate "$Q35" \
            --prompts-file research/prime-gate-coverage-20260802/prompts-mixed.txt \
            --steps 0 > /tmp/prime-gate-ci.log 2>&1; then
        echo "prime-gate FAIL (q35)"; tail -3 /tmp/prime-gate-ci.log; exit 1
    fi
    grep "prime-gate" /tmp/prime-gate-ci.log | tail -2
else
    echo "prime-gate: SKIP (no q35 model at $Q35)"
fi

G31="$MODELS/gemma4-31b-qat-gguf/gemma-4-31B_q4_0-it.gguf"
DEPTH=research/gemma4-bringup/depth-prompt-1736-ids.txt
if [ -f "$G31" ]; then
    out=$(MEMRA_NGEN=8 target/release/run-gen "$G31" 55 2>&1)
    echo "$out" | grep -q "MATCH" || { echo "run-gen argmax FAIL (31B)"; exit 1; }
    echo "run-gen argmax: MATCH (31B)"
    # shellcheck disable=SC2046
    out=$(MEMRA_VERIFY_GATE=7 target/release/gemma-gate "$G31" $(cat "$DEPTH") 2>&1)
    echo "$out" | grep -q "VERIFY-GATE K=7: PASS" || { echo "VERIFY-GATE FAIL (31B depth)"; exit 1; }
    echo "VERIFY-GATE K=7 depth: PASS (31B)"
    D31="$MODELS/gemma4-31b-tooluse-gguf/gemma-4-31B-it-Q4_0-MTP.gguf"
    if [ -f "$D31" ]; then
        # shellcheck disable=SC2046
        out=$(MEMRA_SPEC=6 MEMRA_DRAFT="$D31" MEMRA_NGEN=64 target/release/gemma-gate "$G31" \
            $(cat research/gemma4-bringup/e4b-chat-watercycle-ids.txt) 2>&1)
        echo "$out" | grep -qE "stream agreement 64/64" || { echo "spec self-consistency FAIL (31B)"; exit 1; }
        echo "spec self-consistency 64/64: PASS (31B)"
    fi
else
    echo "run-gen/VERIFY-GATE/spec: SKIP (no 31B model at $G31)"
fi

# gemma-4-12B (dense, MQA globals nkv=1 — the gqa=16 hd512 lane 31B never exercises).
G12="${MEMRA_G12_MODEL:-/data/ai-ml/models/gemma-4-12b-it-qat/gemma-4-12b-it-qat-q4_0.gguf}"
if [ -f "$G12" ]; then
    # shellcheck disable=SC2046
    out=$(MEMRA_NGEN=8 target/release/run-gen "$G12" $(cat "$DEPTH") 2>&1)
    echo "$out" | grep -q "MATCH" || { echo "run-gen argmax FAIL (12B depth)"; exit 1; }
    echo "run-gen argmax depth: MATCH (12B)"
    # shellcheck disable=SC2046
    out=$(MEMRA_VERIFY_GATE=7 target/release/gemma-gate "$G12" $(cat "$DEPTH") 2>&1)
    echo "$out" | grep -q "VERIFY-GATE K=7: PASS" || { echo "VERIFY-GATE FAIL (12B depth)"; exit 1; }
    echo "VERIFY-GATE K=7 depth: PASS (12B)"
else
    echo "12B run-gen/VERIFY-GATE: SKIP (no 12B model at $G12)"
fi
# BATCHED/SOLO DECODE EXACTNESS (serve-path phase 2, 2026-08-05). This battery guards the
# serving tick's numeric contract and it was rotting OUTSIDE the 5090 gate list — only
# validate-h100.sh ran it, so every sm_120 merge took it on trust. The law from the H100 lane:
# anything guarding a live lane belongs INSIDE the battery.
#
# What it pins here: (1) B=1 == decode_step_h, i.e. the MEMRA_SERVE_B1FAST fused solo path a
# c=1 serve request now rides; (2) B=N per-row logits bit-identical to isolated B=1 (the
# isolation contract); (3) device sampling greedy==host-argmax + sampled isolation + lean-logits.
#
# Strict runs on BOTH dtypes since lane/nvfp4-strict (2026-08-05). It used to be Q8_0-only:
# the equalizing env (MEMRA_MMVQ=0 MEMRA_NO_FUSE_NORMQ=1) was Q8/dp4a-shaped — the NVFP4
# gate+up/beta+alpha pair door (`matmul_pre_dual_noscale`) ignored MEMRA_MMVQ=0, so the oracle
# rode the fused MMVQ-family dual while the batched side fell to dp4a and strict FAILED on any
# NVFP4 model at pristine trees (train HEAD 70ce5a0f: gate1 maxdiff 1.639e-1 @ step 2 —
# research/servepath-p2-20260805/; q27 at 93420980: gate2 step-6 divergence —
# research/nvfp4-strict-20260805/repro.log). The engine fix pins that door under MMVQ=0 (the
# same FP-order law `q8_fused_params` always enforced for Q8_0), so a strict FAIL on NVFP4 is
# a REAL failure now. Q8_0 strict remains (it caught the pre-H3 B=1 deviation, maxdiff 1.591e-1).
DBG_NVFP4="${MEMRA_CI_DBG_NVFP4:-$MODELS/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf}"
DBG_Q8="${MEMRA_CI_DBG_Q8:-$MODELS/ornith-1.0-9b-gguf/ornith-1.0-9b-Q8_0.gguf}"
[ -x target/release/decode-batch-gate ] \
    || cargo build --release -p memra-engine --bin decode-batch-gate >/dev/null 2>&1
if [ -f "$DBG_NVFP4" ]; then
    out=$(target/release/decode-batch-gate "$DBG_NVFP4" --steps 32 --batch 8 --mode config 2>&1)
    echo "$out" | grep -q "ALL GREEN" \
        || { echo "$out" | tail -20; echo "decode-batch-gate FAIL (NVFP4 config B=8)"; exit 1; }
    echo "decode-batch-gate config B=8: ALL GREEN (9B NVFP4)"
    out=$(MEMRA_MMVQ=0 MEMRA_NO_FUSE_NORMQ=1 target/release/decode-batch-gate \
        "$DBG_NVFP4" --steps 32 --batch 4 --mode strict 2>&1)
    echo "$out" | grep -q "ALL GREEN" \
        || { echo "$out" | tail -20; echo "decode-batch-gate FAIL (NVFP4 strict B=4)"; exit 1; }
    echo "decode-batch-gate strict B=4 equalized: ALL GREEN (9B NVFP4)"
elif [ -n "${MEMRA_CI_DBG_NVFP4:-}" ]; then
    # an EXPLICIT override that does not resolve is an operator error, not a skip
    echo "decode-batch-gate: MEMRA_CI_DBG_NVFP4 set but not a file: $DBG_NVFP4"; exit 1
else
    echo "decode-batch-gate NVFP4: SKIP (no model at $DBG_NVFP4)"
fi
if [ -f "$DBG_Q8" ]; then
    out=$(MEMRA_Q8RP=1 target/release/decode-batch-gate "$DBG_Q8" \
        --steps 32 --batch 8 --mode config 2>&1)
    echo "$out" | grep -q "ALL GREEN" \
        || { echo "$out" | tail -20; echo "decode-batch-gate FAIL (Q8_0 config B=8)"; exit 1; }
    echo "decode-batch-gate config B=8: ALL GREEN (9B Q8_0)"
    out=$(MEMRA_Q8RP=1 MEMRA_MMVQ=0 MEMRA_NO_FUSE_NORMQ=1 target/release/decode-batch-gate \
        "$DBG_Q8" --steps 32 --batch 4 --mode strict 2>&1)
    echo "$out" | grep -q "ALL GREEN" \
        || { echo "$out" | tail -20; echo "decode-batch-gate FAIL (Q8_0 strict B=4)"; exit 1; }
    echo "decode-batch-gate strict B=4 equalized: ALL GREEN (9B Q8_0)"
elif [ -n "${MEMRA_CI_DBG_Q8:-}" ]; then
    echo "decode-batch-gate: MEMRA_CI_DBG_Q8 set but not a file: $DBG_Q8"; exit 1
else
    echo "decode-batch-gate Q8_0: SKIP (no model at $DBG_Q8)"
fi
# GRAPH-WARMUP STRESS (lane/graph-warmups, 2026-08-05): the pool-growth adversarial gate
# behind the MEMRA_GRAPH_WARMUPS=1 default. Large<->small session cycles + overlap arm force
# captures over freed async-pool blocks; every stream must be bit-identical to eager (the #68
# stale-baked-address class corrupts WITHOUT faulting) and the canary arm proves the
# comparator can catch injected graph-memory corruption. In-battery per the H100 lane law:
# gates outside the battery rot silently. MEMRA_CI_GWSTRESS=0 skips.
if [ "${MEMRA_CI_GWSTRESS:-1}" = "1" ] && [ -x tools/graph-warmup-stress-gate.sh ]; then
    tools/graph-warmup-stress-gate.sh || { echo "graph-warmup-stress FAIL"; exit 1; }
fi
echo "correctness stage: GREEN"

# normal-usage serving battery (2026-07-30): OpenAI surface, streaming, determinism,
# concurrency, lanes, spec==plain serving exactness. MEMRA_CI_SERVE=0 skips.
if [ "${MEMRA_CI_SERVE:-1}" = "1" ] && [ -x tools/serve-smoke.sh ]; then
    tools/serve-smoke.sh || { echo "serve-smoke FAIL"; exit 1; }
fi

[ "$MODE" = "--correctness" ] && exit 0

echo "== local-ci: perf stage ($MODE) =="
GIT_SHA=$(git rev-parse --short HEAD)
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
FAILS=0; WARNS=0

run_cell() {
    local id="$1" model="$2" mode="$3" prompt="$4" ngen="$5" k="$6" draft="$7" ranks="$8"
    local mp="$MODELS/$model"
    [ -f "$mp" ] || { echo "  $id: SKIP (no model)"; return 0; }
    local pfile; pfile=$(jq -r ".prompts[\"$prompt\"]" $MANIFEST)
    local best_toks="0" accept="" tokround="" cell_try
    for cell_try in 1 2; do
    best_toks="0"; accept=""; tokround=""
    for _rep in 1 2; do
        local out toks
        if [ "$mode" = "plain" ]; then
            # shellcheck disable=SC2046
            out=$(MEMRA_NGEN="$ngen" timeout 420 target/release/run-gen "$mp" $(cat "$pfile") 2>&1 || true)
            toks=$(echo "$out" | grep -oE "= [0-9.]+ tok/s" | tail -1 | grep -oE "[0-9.]+" || echo 0)
        else
            local envs=(MEMRA_SPEC_ONLY=1 "MEMRA_SPEC=$k" "MEMRA_DRAFT=$MODELS/$draft" "MEMRA_NGEN=$ngen")
            [ -n "$ranks" ] && [ "$ranks" != "null" ] && envs+=("MEMRA_GEMMA_DRAFT_RANKS=$ranks")
            # shellcheck disable=SC2046
            out=$(env "${envs[@]}" timeout 420 target/release/gemma-gate "$mp" $(cat "$pfile") 2>&1 || true)
            toks=$(echo "$out" | grep -oE "spec: [0-9.]+" | grep -oE "[0-9.]+" || echo 0)
            accept=$(echo "$out" | grep -oE "accept-rate=[0-9.]+" | grep -oE "[0-9.]+" | tail -1 || true)
            tokround=$(echo "$out" | grep -oE "tok/round=[0-9.]+" | grep -oE "[0-9.]+" | tail -1 || true)
        fi
        awk -v a="$toks" -v b="$best_toks" 'BEGIN{exit !(a>b)}' && best_toks="$toks"
    done
    if window_free_now; then break; fi
    if [ "$cell_try" = 1 ]; then
        echo "  $id: window went DIRTY mid-cell — waiting + retrying once"
        while ! window_free_now; do sleep 40; done
    else
        echo "  $id: DIRTY twice — recording with window_clean=false"
        WINDOW_CLEAN=false
    fi
    done
    [ "$best_toks" = "0" ] && { echo "  $id: FAIL (no reading)"; FAILS=$((FAILS+1)); return 0; }

    # rolling-median verdict from prior rows of this cell
    local base verdict="OK" note="" rows
    rows=$(grep "\"cell\":\"$id\"" "$OUT" 2>/dev/null || true)
    base=$(printf '%s\n' "$rows" | tail -"$(jq -r .gates.baseline_window $MANIFEST)" \
        | jq -s 'map(.toks) | sort | .[length/2|floor] // 0' 2>/dev/null)
    base=${base:-0}
    if awk -v b="$base" 'BEGIN{exit !(b>0)}'; then
        local drop
        drop=$(awk -v n="$best_toks" -v b="$base" 'BEGIN{printf "%.2f", (b-n)/b*100}')
        if awk -v d="$drop" -v t="$(jq -r .gates.cell_drop_fail_pct $MANIFEST)" 'BEGIN{exit !(d>t)}'; then
            verdict="FAIL"; FAILS=$((FAILS+1)); note="tok/s -$drop% vs median $base"
        elif awk -v d="$drop" -v t="$(jq -r .gates.cell_drop_warn_pct $MANIFEST)" 'BEGIN{exit !(d>t)}'; then
            verdict="WARN"; WARNS=$((WARNS+1)); note="tok/s -$drop% vs median $base"
        fi
        if [ -n "$accept" ]; then
            local abase
            abase=$(printf '%s\n' "$rows" | tail -5 \
                | jq -s 'map(.accept // empty) | sort | .[length/2|floor] // 0' 2>/dev/null)
            abase=${abase:-0}
            if awk -v a="$accept" -v b="$abase" -v t="$(jq -r .gates.accept_drop_fail $MANIFEST)" \
                 'BEGIN{exit !(b>0 && b-a>t)}'; then
                verdict="FAIL"; FAILS=$((FAILS+1)); note="$note; ACCEPTANCE $abase -> $accept"
            fi
        fi
    else
        note="first row (baseline seed)"
    fi
    printf '{"ts":"%s","git":"%s","cell":"%s","toks":%s%s%s,"profile":"%s","load":%s,"window_clean":%s}\n' \
        "$TS" "$GIT_SHA" "$id" "$best_toks" \
        "${accept:+,\"accept\":$accept}" "${tokround:+,\"tok_round\":$tokround}" \
        "$PROFILE" "$LOAD" "$WINDOW_CLEAN" >> "$OUT"
    echo "  $id: $best_toks tok/s${accept:+ accept=$accept} [$verdict]${note:+ — $note}"
}

while read -r cell; do
    id=$(echo "$cell" | jq -r .id)
    if [ "$MODE" = "--perf-quick" ] && [[ "$id" != 31b-* ]]; then continue; fi
    # MEMRA_CI_CELLS: extended-regex cell-id filter (e.g. "26b-|e4b-") — run a subset
    # without touching the manifest; verdicts/rows behave exactly like a full run.
    if [ -n "${MEMRA_CI_CELLS:-}" ] && ! echo "$id" | grep -qE "$MEMRA_CI_CELLS"; then continue; fi
    run_cell "$id" "$(echo "$cell" | jq -r .model)" "$(echo "$cell" | jq -r .mode)" \
             "$(echo "$cell" | jq -r .prompt)" "$(echo "$cell" | jq -r .ngen)" \
             "$(echo "$cell" | jq -r '.k // 0')" "$(echo "$cell" | jq -r '.draft // ""')" \
             "$(echo "$cell" | jq -r '.ranks // ""')"
done < <(jq -c '.cells[]' $MANIFEST)

echo "perf stage: $FAILS fail, $WARNS warn"
[ "$FAILS" -eq 0 ] || exit 1
