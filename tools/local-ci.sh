#!/usr/bin/env bash
# memra local CI — the real gate (GitHub CI is compile-only; the rig is the test machine).
#
#   tools/local-ci.sh                correctness stage only (~3 min)
#   tools/local-ci.sh --perf         correctness + full perf battery (~15 min)
#   tools/local-ci.sh --perf-quick   correctness + gemma-31B cells only (~6 min)
#
# Correctness stage: kernel-check, run-gen argmax gate, run-spec K=1..8 self-consistency,
# Gemma stream agreement, and VERIFY-GATE logit maxdiff at depth — the standing exactness
# battery, one command.
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
# MEMRA_CI_DIRTY_WAIT (default 600s): how long a cell waits for a co-resident GPU process
# to leave before recording its row as window_clean=false. Latched after the first cell
# that outwaits it — a permanently-co-resident process (an owner service holding an idle
# CUDA context) must not turn the perf stage into an unbounded hang.
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
Q35_DRAFT="$MODELS/qwen36-35b-moe/draft-35b-owntrim-nvfp4head-q4blk.gguf"
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

# The standing MTP exactness gate. A naked run-spec invocation sweeps K=1..8; explicitly clear
# single-K and alternate-mode env so a caller cannot silently narrow or change the gate.
# The Gemma-4 31B target below uses a separate assistant-drafter API, so its independent
# stream-agreement check remains on gemma-gate. MEMRA_CI_RUNSPEC=0 skips this sweep.
if [ "${MEMRA_CI_RUNSPEC:-1}" = "1" ]; then
    if [ -f "$Q35" ] && [ -f "$Q35_DRAFT" ]; then
        [ -x target/release/run-spec ] \
            || cargo build --release -p memra-engine --bin run-spec >/dev/null
        RUNSPEC_LOG=/tmp/local-ci-run-spec.log
        runspec_rc=0
        (
            unset MEMRA_PROMPT_DIR MEMRA_SPEC_K MEMRA_GEN_ONLY
            MEMRA_SPEC_TEMP=0 MEMRA_MTP_DRAFT="$Q35_DRAFT" MEMRA_NGEN=32 \
                MEMRA_PROMPT_FILE=tools/fast-gate/prompts/probe.txt \
                timeout 900 target/release/run-spec "$Q35"
        ) 2>&1 | tee "$RUNSPEC_LOG" >/dev/null || runspec_rc=$?
        runspec_passes=$(grep -c "self-consistency: PASS" "$RUNSPEC_LOG" || true)
        runspec_ks=$(grep -cE '^\[generate_spec K=[1-8]\]' "$RUNSPEC_LOG" || true)
        if [ "$runspec_rc" -ne 0 ] || [ "$runspec_passes" -ne 8 ] \
                || [ "$runspec_ks" -ne 8 ] \
                || ! grep -q "=== SELF-CONSISTENCY PASS ===" "$RUNSPEC_LOG"; then
            fail_detail=$(awk '
                /^\[generate_spec K=[0-9]+\]/ {
                    k = $0
                    sub(/^.*K=/, "", k)
                    sub(/\].*$/, "", k)
                }
                /self-consistency: FAIL/ { failed_k = k }
                /FIRST DIVERGENCE at index [0-9]+:/ && failed_k != "" {
                    pos = $0
                    sub(/^.*FIRST DIVERGENCE at index /, "", pos)
                    sub(/:.*/, "", pos)
                    print failed_k " " pos
                    exit
                }
            ' "$RUNSPEC_LOG")
            if [ -n "$fail_detail" ]; then
                echo "run-spec self-consistency FAIL (K=${fail_detail%% *}, FIRST DIVERGENCE at index ${fail_detail#* })"
            else
                echo "run-spec K=1..8 FAIL (exit $runspec_rc, $runspec_passes/8 per-K passes)"
            fi
            tail -12 "$RUNSPEC_LOG"
            exit 1
        fi
        echo "run-spec K=1..8 self-consistency: PASS (Qwen 35B, 8/8)"
    elif [ ! -f "$Q35" ]; then
        echo "run-spec K=1..8: SKIP (no q35 model at $Q35)"
    else
        echo "run-spec K=1..8: SKIP (no q35 draft at $Q35_DRAFT)"
    fi
else
    echo "run-spec K=1..8: SKIP (MEMRA_CI_RUNSPEC=0)"
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

# c=64 CONCURRENCY STRESS (lane/admit-oom, 2026-08-06): 64 staggered streaming clients on a
# 24GB card — the cell that was RED until the admission cost model charged the spec transient
# reserve and step-OOM learned to park instead of kill. In-battery per the H100 lane law
# (gates outside the battery rot silently); serving-density deliberately left it unwired while
# it was red, because wiring a known-red gate either blocks every merge or normalizes a red.
# Its own teeth: `tools/serve-stress-gate.sh --teeth` forces the reserve to 16MB and asserts
# the RED returns — run that whenever the admission math moves. MEMRA_CI_STRESS=0 skips.
if [ "${MEMRA_CI_STRESS:-1}" = "1" ] && [ -x tools/serve-stress-gate.sh ]; then
    tools/serve-stress-gate.sh || { echo "serve-stress FAIL"; exit 1; }
fi

# SERVED-SPEC ACCEPTANCE + LONG-TEXT ASSERTION (lane/accept-gate, 2026-08-06): the arm that
# closes a receipted blind spot in THIS battery. research/f8f4-flip-20260806 (merged c506317e)
# showed a kernel arm move served greedy text in 4 of 6 regime cells at temperature 0 and move
# spec acceptance up to -9.5pp while EVERY gate above stayed green — because (1) the token
# goldens stop at 20 tokens and both divergences landed at generated index 22 and 38, (2)
# `fast-gate --refresh-goldens` after such a change would silently re-pin the new arm, and (3)
# nothing here compared accepted-draft COUNTS, which are spec throughput, i.e. the product.
# Each arm was internally self-consistent and reproduced its own goldens, so self-consistency
# could never see it.
#
# This arm asserts, at the production serve config (real regime drafter attached, real serve K):
# exact (rounds, drafted, accepted) integers — temp 0 makes drafting deterministic — plus the
# full generated text sha256 to ngen=128, 6.4x past the golden window. In-battery per the H100
# lane law: gates outside the battery rot silently.
#
# Default arm = the smoke tier (ONE model, ONE cell: q27-p1, ~1 min incl. the 16G load) to keep
# the correctness stage near its ~3 min budget. The full 6-cell matrix (both NVFP4-reachable
# models x 3 prompt lengths) is `tools/accept-gate.sh --full`, and `--control` adds the
# second-boot determinism control. Its own teeth: `tools/accept-gate.sh --teeth` sets
# MEMRA_MMQ_F8F4=1 and REQUIRES the gate to fail — run that whenever the spec/draft or NVFP4
# prefill path moves. MEMRA_CI_ACCEPT=0 skips.
if [ "${MEMRA_CI_ACCEPT:-1}" = "1" ] && [ -x tools/accept-gate.sh ]; then
    tools/accept-gate.sh || { echo "accept-gate FAIL"; exit 1; }
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
        # The reps run UNDER THE SHARED GPU LOCK (2026-08-06, v0.71.0 release battery). The
        # window_free_now() recheck below samples only BETWEEN reps, so a neighbor lane that
        # starts and finishes inside a rep is invisible to it — that hole reported 10/10 cells
        # FAIL (-8.31%..-24.75%) at the v0.71.0 tag candidate while a concurrent Q8RP census
        # held run-gen on the same card, and every poisoned row still recorded
        # window_clean:true. Every other GPU consumer in this repo (fast-gate, the gate
        # scripts) already takes /tmp/gpu5090.lock; the perf stage — the one stage whose whole
        # output is a timing number — did not.
        if [ "$mode" = "plain" ]; then
            # shellcheck disable=SC2046
            out=$(flock -w "${MEMRA_CI_LOCK_WAIT:-7200}" "${MEMRA_CI_LOCK:-/tmp/gpu5090.lock}" \
                  env MEMRA_NGEN="$ngen" timeout 420 target/release/run-gen "$mp" $(cat "$pfile") 2>&1 || true)
            toks=$(echo "$out" | grep -oE "= [0-9.]+ tok/s" | tail -1 | grep -oE "[0-9.]+" || echo 0)
        else
            local envs=(MEMRA_SPEC_ONLY=1 "MEMRA_SPEC=$k" "MEMRA_DRAFT=$MODELS/$draft" "MEMRA_NGEN=$ngen")
            [ -n "$ranks" ] && [ "$ranks" != "null" ] && envs+=("MEMRA_GEMMA_DRAFT_RANKS=$ranks")
            # shellcheck disable=SC2046
            out=$(flock -w "${MEMRA_CI_LOCK_WAIT:-7200}" "${MEMRA_CI_LOCK:-/tmp/gpu5090.lock}" \
                  env "${envs[@]}" timeout 420 target/release/gemma-gate "$mp" $(cat "$pfile") 2>&1 || true)
            toks=$(echo "$out" | grep -oE "spec: [0-9.]+" | grep -oE "[0-9.]+" || echo 0)
            accept=$(echo "$out" | grep -oE "accept-rate=[0-9.]+" | grep -oE "[0-9.]+" | tail -1 || true)
            tokround=$(echo "$out" | grep -oE "tok/round=[0-9.]+" | grep -oE "[0-9.]+" | tail -1 || true)
        fi
        awk -v a="$toks" -v b="$best_toks" 'BEGIN{exit !(a>b)}' && best_toks="$toks"
    done
    if window_free_now; then break; fi
    if [ "$cell_try" = 1 ]; then
        # BOUNDED wait, LATCHED once (2026-08-07, lane/spec-gate). This loop used to be
        # `while ! window_free_now; do sleep 40; done` — unbounded, so a PERSISTENT
        # co-resident deadlocked the whole perf stage and made the honest fallback two
        # lines below (record with window_clean=false) unreachable. Hit for real: the
        # owner's hermes-gateway.service holds a 394 MiB idle CUDA context 24/7 on this
        # box, 0% GPU util, and is not a lane's job to kill — the battery sat in that
        # loop through 31b-plain-short and produced no rows at all. A gate that hangs
        # forever is worse than one that records an honestly-labeled row.
        #
        # Latched, because the wait is only worth paying for a TRANSIENT joiner: once one
        # cell has proven the co-resident outlasts the wait, every later cell skips
        # straight to the labeled retry instead of re-paying it (10 cells x 600 s of
        # pure sleeping is not a gate, it is a hang with progress output).
        if [ "${PERSISTENT_CORESIDENT:-0}" = 1 ]; then
            echo "  $id: window DIRTY, co-resident already known persistent — retrying, row will be window_clean=false"
        else
            local wait_left="${MEMRA_CI_DIRTY_WAIT:-600}"
            echo "  $id: window went DIRTY mid-cell — waiting up to ${wait_left}s + retrying once"
            while ! window_free_now && [ "$wait_left" -gt 0 ]; do
                sleep 20; wait_left=$((wait_left - 20))
            done
            if ! window_free_now; then
                PERSISTENT_CORESIDENT=1
                echo "  $id: co-resident did not leave in ${MEMRA_CI_DIRTY_WAIT:-600}s — treating it as persistent; rows from here are window_clean=false"
            fi
        fi
    else
        echo "  $id: DIRTY twice — recording with window_clean=false"
        WINDOW_CLEAN=false
    fi
    done
    [ "$best_toks" = "0" ] && { echo "  $id: FAIL (no reading)"; FAILS=$((FAILS+1)); return 0; }

    # Rolling-median verdict from prior rows of this cell.
    #
    # WHAT THIS VERDICT IS, EXACTLY (2026-08-06): a DRIFT TRIPWIRE, not evidence. The
    # denominator is a median of rows measured on earlier days, so a tok/s FAIL here is a
    # CROSS-DAY comparison — precisely the form the measurement law (research/benchmarks.md,
    # the H100 lane's law 1) forbids as proof, because clock/thermal/power state drifts under
    # both numerator and denominator. It answers "did something move?", never "did this commit
    # regress?".
    #
    # THE PROTOCOL WHEN IT GOES RED (do not skip to a conclusion either way):
    #   build the last-green commit's binary, then run the SAME cell interleaved A/B/A/B, N>=5
    #   each, in ONE thermal window under one exclusive lock hold, and compare only within
    #   that window. See research/v071-prep-20260806/battery-logs/perf-ab.sh for the harness.
    # v0.71.0 is the worked example: 10/10 cells "FAIL" at -8.31%..-24.75%, and the
    # interleaved A/B put the last-green baseline binary at 37.87 tok/s against the candidate's
    # 37.87 (+0.00%) — the drop was the machine's state, and zero code had regressed. A uniform
    # multi-cell drop with correctness green is that signature, not ten simultaneous
    # regressions.
    #
    # ACCEPTANCE drops are the exception: acceptance is a RATIO, clock-independent, and
    # invisible to every exactness gate by construction. Treat an acceptance FAIL as real.
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
if [ "$FAILS" -gt 0 ]; then
    cat <<'PERFRED'

  ^ A tok/s FAIL above is a DRIFT TRIPWIRE against a cross-day median, NOT a proven
    regression, and it is not by itself a merge/tag blocker. Settle it before concluding:
      1. build the last-green commit's binary for this cell,
      2. run that cell interleaved A/B/A/B, N>=5 each, ONE thermal window, one exclusive
         lock hold (harness: research/v071-prep-20260806/battery-logs/perf-ab.sh),
      3. compare medians WITHIN that window only.
    A uniform drop across many cells with correctness green points at machine state
    (power/thermal/profile) or a contended window, not at the diff. An ACCEPTANCE FAIL is
    different — acceptance is clock-independent, so treat it as real.
PERFRED
fi
[ "$FAILS" -eq 0 ] || exit 1
