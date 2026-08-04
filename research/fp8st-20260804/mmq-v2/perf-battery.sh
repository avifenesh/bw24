#!/usr/bin/env bash
# v2 slice 4 — pp perf battery for the v2 per-block FP8 MMQ prefill kernel (lane/fp8-mmq-v2).
#
# Same protocol as the v1 lane's perf-battery.sh (research/fp8st-20260804/mmq/perf-battery.sh):
# MEMRA_PP_ONLY=1 MEMRA_PP_REPS=3 for the in-process median, N=5 PROCESS reps per arm, arms
# INTERLEAVED within each rep so clock/thermal drift hits both arms equally. The floor number here
# is RE-MEASURED in this session — v1's recorded floor is never used as a cross-day denominator.
#
# Difference from v1's battery: only two arms. ARM A (MEMRA_FP8_FOLD, the lossy per-tensor scale
# fold) is not a shippable arm and is not what v2 competes with; the Q8_0 ARM B' floor is.
#
# Two lengths: pp512 and pp6257 (deep context). p4-16k.txt truncated to exactly 6257 tokens with
# the checkpoint's own tokenizer, same as v1's local deep leg.
#
# GPU 0 only, flock'd, params baked as literals (workflow args do not propagate).
set -uo pipefail
CK=${CK:-/data/ai-ml/hf-models/qwen3-1.7b-blk128fp8-synth}
BIN=${BIN:-./target/release/fp8_mmq_stream}
R=${R:-research/fp8st-20260804/mmq-v2}
LOCK=${LOCK:-/tmp/gpu5090.lock}
P6257=${P6257:-research/e2e/prompts/p4-16k.txt}
mkdir -p "$R/perf"
DLOG=$R/perf/driver.log
log(){ echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$DLOG"; }
snap(){ nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader -i 0; }

nvidia-smi --query-gpu=timestamp,index,temperature.gpu,clocks.sm,power.draw,memory.used \
  --format=csv -l 1 -i 0 > "$R/perf/gpu0-1hz.csv" 2>&1 &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

one(){ # $1 arm, $2 len, $3 rep, $4.. env
  local arm=$1 len=$2 rep=$3; shift 3
  local out=$R/perf/pp$len-$arm-r$rep.log
  log "$arm pp$len rep$rep pre: $(snap)"
  local pf=()
  [ "$len" = 6257 ] && pf=(MEMRA_PROMPT_FILE=$P6257)
  flock "$LOCK" env CUDA_VISIBLE_DEVICES=0 MEMRA_PP_ONLY=1 MEMRA_PP_REPS=3 \
    MEMRA_PP_TOKENS=$len "${pf[@]}" "$@" timeout 3600 "$BIN" "$CK" 1 > "$out" 2>&1
  local rc=$?
  log "$arm pp$len rep$rep post: $(snap) | rc=$rc | $(grep -aoE 'pp-only MEDIAN: [0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s' "$out" | head -1) | oom=$(grep -ac 'out of memory' "$out")"
}

LENS=${LENS:-"512 6257"}
for len in $LENS; do
  for r in 1 2 3 4 5; do
    one floor $len $r MEMRA_PP_X=0
    one mmq   $len $r MEMRA_FP8_MMQ=1 MEMRA_PP_FP8_BUDGET_MB=8192
  done
done
log "V2 PERF BATTERY DONE"
