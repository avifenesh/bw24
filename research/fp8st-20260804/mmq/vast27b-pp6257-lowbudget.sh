#!/usr/bin/env bash
# Supplementary leg: the 27B pp6257 mmq cell at a budget that FITS.
#
# WHY. The main battery's pp6257 mmq cell died with a captured
#   Error: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")
# at BUDGET=3072 (vast27b/RESULTS.jsonl). The 1 Hz sampler explains it without inference: the
# FLOOR arm alone peaks at 31488 of 32607 MiB at pp6257 on this 27 GB model, leaving ~1119 MiB,
# and the MEMRA_PP_FP8 stash duplicates every F8-origin projection ON TOP of the resident Q8_0.
# So 3072 MB of duplicate cannot fit at this context length, and the cell has a failure but no
# number. This leg gives it a number.
#
# BUDGET=768 is chosen from that measurement, not guessed: 1119 MiB of headroom minus room for
# allocator slack. At the measured ~355 MiB of e4m3 per 27B layer that is a prefix of ~2.2 of 64
# layers, i.e. ~3.4% coverage — LOWER than the pp512 leg's 8.7%, and the row is labeled with the
# coverage the ledger actually reports rather than this prediction.
#
# PROTOCOL. Same as the main battery so the rows are comparable: MEMRA_PP_ONLY=1
# MEMRA_PP_REPS=3 (in-process median), N=3 process reps, floor and mmq INTERLEAVED inside each
# rep so both share one clock/thermal regime. Its OWN floor column is re-measured here — the main
# battery's pp6257 floor was taken in a different thermal window, and cross-run comparison is
# invalid by the interleaving law, including for the denominator.
#
# The ledger print is on: a pp number for this arm is only evidence alongside a nonzero dispatch
# count, and at a budget this small the count is the whole point.
#
# GPU 0 only, flock'd — RUN THIS ONLY AFTER THE MAIN BATTERY HAS FINISHED (a queued flock would
# otherwise interleave between the battery's own reps and perturb its thermal regime mid-leg).
set -uo pipefail
cd /root/memra-fp8mmq
OUT=research/fp8st-20260804/mmq/vast27b-pp6257-lowbudget
mkdir -p "$OUT"
CKPT=/root/models/qwen36-27b-fp8
P6257=research/e2e/prompts/p3-agentic-long.txt
LOCK=/tmp/memra-bench.lock
BIN=/root/target-instr/release/run-gen   # the instrumented build — it prints the dispatch ledger
BUDGET=768
DLOG=$OUT/driver.log
log(){ echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$DLOG"; }
snap(){ nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader -i 0; }

nvidia-smi --query-gpu=timestamp,index,temperature.gpu,clocks.sm,power.draw,memory.used \
  --format=csv -l 1 -i 0 > "$OUT/gpu0-1hz.csv" 2>&1 &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

pp(){ # arm, rep, env...
  local arm=$1 rep=$2; shift 2
  local out=$OUT/pp6257-$arm-r$rep.log
  log "$arm rep$rep pre: $(snap)"
  flock "$LOCK" env CUDA_VISIBLE_DEVICES=0 MEMRA_PP_ONLY=1 MEMRA_PP_REPS=3 \
    MEMRA_PROMPT_FILE="$P6257" "$@" timeout 7200 "$BIN" "$CKPT" > "$out" 2>&1
  local rc=$?
  log "$arm rep$rep post: $(snap) | rc=$rc | $(grep -aoE 'pp-only MEDIAN: [0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s' "$out" | head -1) | $(grep -aoE 'fp8-mmq dispatches: .*' "$out" | head -1) | oom=$(grep -ac 'out of memory' "$out") | $(grep -ao 'Error: .*' "$out" | head -1)"
}

log "== 27B pp6257 SUPPLEMENTARY: floor vs mmq at BUDGET=${BUDGET}MB, N=3 interleaved =="
for r in 1 2 3; do
  pp floor $r MEMRA_PP_X=0
  pp mmq   $r MEMRA_FP8_MMQ=1 MEMRA_PP_FP8_BUDGET_MB=$BUDGET
done
log "SUPPLEMENTARY LEG DONE"
