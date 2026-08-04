#!/usr/bin/env bash
# gemm-arm-battery.sh — lane/fp8-gemm-arm on the vast 2x5090 box.
# Phase 1 (accuracy N=2 repro + attribution control):
#   floor-accuracy-r2       : default env (Q8_0 re-encode floor), NGEN=128
#   armA-accuracy-r2        : MEMRA_ST_E4M3=1 MEMRA_FP8_FOLD=1, NGEN=128
#   armA-nofold-control-r1  : MEMRA_ST_E4M3=1 only (block-128 carry, no fold) — attributes
#                             the token divergence to the FOLD vs the e4m3 config itself.
# Phase 2 (pp512 perf, N=5 process reps per arm, interleaved FLOOR,ARMA,...):
#   MEMRA_PP_ONLY=1 MEMRA_PP_REPS=3 (in-process median printed by run-gen).
# GPU 0 only, flock'd, params baked as literals.
set -uo pipefail
cd /root/memra-fp8gemm
OUT=research/fp8st-20260803/gemm-arm
mkdir -p "$OUT"
CKPT=/root/models/qwen36-27b-fp8
P512=research/e2e/prompts/pp512.txt
LOCK=/tmp/memra-bench.lock
DLOG=$OUT/battery-driver.log
log(){ echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$DLOG"; }
snap(){ nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader -i 0; }

nvidia-smi --query-gpu=timestamp,index,temperature.gpu,clocks.sm,power.draw,memory.used \
  --format=csv -l 1 -i 0 > "$OUT/battery-gpu0-1hz.csv" 2>&1 &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

log "== PHASE 1: accuracy repro + control =="
log "floor-accuracy-r2 pre: $(snap)"
flock "$LOCK" env CUDA_VISIBLE_DEVICES=0 MEMRA_NGEN=128 MEMRA_PROMPT_FILE=$P512 \
  timeout 3600 target/release/run-gen "$CKPT" > "$OUT/floor-accuracy-r2.log" 2>&1
log "floor-accuracy-r2 post: $(snap) | $(grep -aoE 'generated 128 tokens in [0-9.]+s = [0-9.]+ tok/s' "$OUT/floor-accuracy-r2.log" | head -1) | $(grep -a verify-prefill "$OUT/floor-accuracy-r2.log")"

log "armA-accuracy-r2 pre: $(snap)"
flock "$LOCK" env CUDA_VISIBLE_DEVICES=0 MEMRA_ST_E4M3=1 MEMRA_FP8_FOLD=1 MEMRA_NGEN=128 \
  MEMRA_PROMPT_FILE=$P512 timeout 7200 target/release/run-gen "$CKPT" \
  > "$OUT/armA-accuracy-r2.log" 2>&1
log "armA-accuracy-r2 post: $(snap) | $(grep -aoE 'generated 128 tokens in [0-9.]+s = [0-9.]+ tok/s' "$OUT/armA-accuracy-r2.log" | head -1) | $(grep -a verify-prefill "$OUT/armA-accuracy-r2.log")"

log "armA-nofold-control-r1 pre: $(snap)"
flock "$LOCK" env CUDA_VISIBLE_DEVICES=0 MEMRA_ST_E4M3=1 MEMRA_NGEN=128 \
  MEMRA_PROMPT_FILE=$P512 timeout 7200 target/release/run-gen "$CKPT" \
  > "$OUT/armA-nofold-control-r1.log" 2>&1
log "armA-nofold-control-r1 post: $(snap) | $(grep -aoE 'generated 128 tokens in [0-9.]+s = [0-9.]+ tok/s' "$OUT/armA-nofold-control-r1.log" | head -1) | $(grep -a verify-prefill "$OUT/armA-nofold-control-r1.log")"

log "== PHASE 2: pp512 battery, N=5 interleaved =="
for r in 1 2 3 4 5; do
  log "FLOOR pp rep $r pre: $(snap)"
  flock "$LOCK" env CUDA_VISIBLE_DEVICES=0 MEMRA_PP_ONLY=1 MEMRA_PP_REPS=3 \
    MEMRA_PROMPT_FILE=$P512 timeout 3600 target/release/run-gen "$CKPT" \
    > "$OUT/pp-floor-r$r.log" 2>&1
  log "FLOOR pp rep $r post: $(snap) | $(grep -aoE 'pp-only MEDIAN: [0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s' "$OUT/pp-floor-r$r.log" | head -1) | oom=$(grep -ac 'out of memory' "$OUT/pp-floor-r$r.log")"
  log "ARMA pp rep $r pre: $(snap)"
  flock "$LOCK" env CUDA_VISIBLE_DEVICES=0 MEMRA_ST_E4M3=1 MEMRA_FP8_FOLD=1 \
    MEMRA_PP_ONLY=1 MEMRA_PP_REPS=3 MEMRA_PROMPT_FILE=$P512 timeout 7200 \
    target/release/run-gen "$CKPT" > "$OUT/pp-armA-r$r.log" 2>&1
  log "ARMA pp rep $r post: $(snap) | $(grep -aoE 'pp-only MEDIAN: [0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s' "$OUT/pp-armA-r$r.log" | head -1) | oom=$(grep -ac 'out of memory' "$OUT/pp-armA-r$r.log")"
done
log "GEMM-ARM BATTERY DONE"
