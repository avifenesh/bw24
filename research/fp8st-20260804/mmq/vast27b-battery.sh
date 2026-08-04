#!/usr/bin/env bash
# Slice 4 (27B) — pp perf + exactness for the per-block FP8 MMQ prefill kernel on the vast 2x5090
# box, against the SAME 27B block-128 FP8 checkpoint ARM A was measured on
# (/root/models/qwen36-27b-fp8, Qwen3.6-27B, quantization_config fmt=e4m3 quant_method=fp8).
#
# Runs on the box, not locally: this is the only rig carrying a 27B-class block-128 grid.
#
# The ARM A row this compares against (research/fp8st-20260803/gemm-arm/armA-vs-floor.jsonl):
# floor pp512 median 4066.7, ARM A 4816.3, +18.4% — measured on THIS box with THIS protocol.
# Per the interleaving law those numbers are NOT quoted as this session's denominator: floor is
# re-measured here in the same session, and the ARM A column is re-measured too, so all three
# arms share one clock/thermal regime. The 2026-08-03 medians serve only as a cross-check that
# the box reproduces its own prior behaviour.
#
# Protocol: MEMRA_PP_ONLY=1 MEMRA_PP_REPS=3 (in-process median), N=5 process reps, arms
# INTERLEAVED floor,mmq,arma within each rep. GPU 0 only, flock'd, literals baked.
set -uo pipefail
cd /root/memra-fp8mmq
OUT=research/fp8st-20260804/mmq/vast27b
mkdir -p "$OUT"
CKPT=/root/models/qwen36-27b-fp8
P512=research/e2e/prompts/pp512.txt
P6257=research/e2e/prompts/p3-agentic-long.txt
LOCK=/tmp/memra-bench.lock
BIN=target/release/run-gen
# STASH BUDGET, measured not guessed. First attempt used 16384 MB and died with
#   Error: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")
# (accuracy-mmq.log, run 02:40-02:50Z). The 1 Hz sampler shows the FLOOR arm already peaking at
# 27488 MiB of this 5090's 32607 MiB => 5119 MiB headroom, and the stash duplicates every
# F8-origin projection's e4m3 bytes on top of the resident Q8_0. 3072 MB leaves ~2 GiB for
# activations/KV growth at pp6257. Budget covers a PREFIX of layers in load order, so this is
# partial coverage of the 27B by design — the pp delta it produces is a LOWER bound on the
# kernel's full-coverage effect, and the file says so.
BUDGET=${BUDGET:-3072}
DLOG=$OUT/driver.log
log(){ echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$DLOG"; }
snap(){ nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader -i 0; }

nvidia-smi --query-gpu=timestamp,index,temperature.gpu,clocks.sm,power.draw,memory.used \
  --format=csv -l 1 -i 0 > "$OUT/gpu0-1hz.csv" 2>&1 &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

pp(){ # arm, len, promptfile, rep, env...
  local arm=$1 len=$2 pf=$3 rep=$4; shift 4
  local out=$OUT/pp$len-$arm-r$rep.log
  log "$arm pp$len rep$rep pre: $(snap)"
  flock "$LOCK" env CUDA_VISIBLE_DEVICES=0 MEMRA_PP_ONLY=1 MEMRA_PP_REPS=3 \
    MEMRA_PROMPT_FILE="$pf" "$@" timeout 7200 "$BIN" "$CKPT" > "$out" 2>&1
  local rc=$?
  log "$arm pp$len rep$rep post: $(snap) | rc=$rc | $(grep -aoE 'pp-only MEDIAN: [0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s' "$out" | head -1) | oom=$(grep -ac 'out of memory' "$out")"
}

log "== 27B ACCURACY: 128-tok greedy stream, floor vs MMQ (same protocol as the ARM A row) =="
for arm in floor mmq; do
  case $arm in
    floor) E=(MEMRA_PP_X=0) ;;
    mmq)   E=(MEMRA_FP8_MMQ=1 MEMRA_PP_FP8_BUDGET_MB=$BUDGET) ;;
  esac
  log "$arm accuracy pre: $(snap)"
  flock "$LOCK" env CUDA_VISIBLE_DEVICES=0 MEMRA_NGEN=128 MEMRA_PROMPT_FILE=$P512 \
    "${E[@]}" timeout 7200 "$BIN" "$CKPT" > "$OUT/accuracy-$arm.log" 2>&1
  log "$arm accuracy post: $(snap) | rc=$? | $(grep -aoE 'generated 128 tokens in [0-9.]+s = [0-9.]+ tok/s' "$OUT/accuracy-$arm.log" | head -1) | $(grep -a verify-prefill "$OUT/accuracy-$arm.log" | head -1)"
done

log "== 27B PERF: pp512 + pp6257, N=5 interleaved floor,mmq,arma =="
for r in 1 2 3 4 5; do
  pp floor 512 $P512 $r MEMRA_PP_X=0
  pp mmq   512 $P512 $r MEMRA_FP8_MMQ=1 MEMRA_PP_FP8_BUDGET_MB=$BUDGET
  pp arma  512 $P512 $r MEMRA_FP8_FOLD=1 MEMRA_ST_E4M3=1
done
for r in 1 2 3 4 5; do
  pp floor 6257 $P6257 $r MEMRA_PP_X=0
  pp mmq   6257 $P6257 $r MEMRA_FP8_MMQ=1 MEMRA_PP_FP8_BUDGET_MB=$BUDGET
  pp arma  6257 $P6257 $r MEMRA_FP8_FOLD=1 MEMRA_ST_E4M3=1
done
log "27B BATTERY DONE"
