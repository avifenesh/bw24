#!/usr/bin/env bash
# Slice 4 — pp perf battery for the per-block FP8 MMQ prefill kernel (lane/fp8-mmq).
#
# Protocol matches the ARM A battery it is compared against
# (research/fp8st-20260803/gemm-arm/gemm-arm-battery.sh): MEMRA_PP_ONLY=1 MEMRA_PP_REPS=3 for the
# in-process median, N=5 PROCESS reps per arm, arms INTERLEAVED within each rep so a clock or
# thermal drift hits both denominators equally. Cross-run / cross-day comparison is invalid — the
# floor number in this file is re-measured in the same session as the MMQ number, never quoted
# from another day.
#
# Two lengths: pp512 (the ARM A comparison point) and pp6257 (the deep-context point; the
# p3-agentic-long prompt, whose token count the 2026-08-02 board rows call d6257).
#
# GPU 0 only, flock'd, params baked as literals (workflow args do not propagate).
set -uo pipefail
CK=${CK:-/data/ai-ml/hf-models/qwen3-1.7b-blk128fp8-synth}
BIN=${BIN:-./target/release/fp8_mmq_stream}
R=${R:-research/fp8st-20260804/mmq}
LOCK=${LOCK:-/tmp/gpu5090.lock}
# Deep-context prompt. p3-agentic-long.txt is the file whose token count the 2026-08-02 board rows
# call d6257 — but that count is the 27B/35B tokenizer's. The 1.7B tokenizer yields only 5882 for
# it ("tokenizes to 5882 tokens, need 6257"), so the LOCAL deep leg reads p4-16k.txt and truncates
# to exactly 6257 tokens: same length, so the same prefill shape, with the checkpoint's own
# tokenizer. The 27B leg (vast27b-battery.sh) uses p3-agentic-long.txt directly, as ARM A did.
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
    one arma  $len $r MEMRA_FP8_FOLD=1 MEMRA_ST_E4M3=1
  done
done
log "PERF BATTERY DONE"
