#!/usr/bin/env bash
# pro6000-prod: Battery A — anchor cells, q27 daily NVFP4+MTP artifact + Q8_0 prod artifact.
# Interleaved arms per rep (nv,q8 within each rep), N=5 process reps, medians read offline.
# GPU exclusive (single-tenant pod, no flock needed). Power cap fixed 600W (nvidia-smi -pl container-blocked).
set -u
cd /root/bw24
R=/root/receipts/anchor
mkdir -p "$R"
NV=/root/models/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
Q8=/root/models/Qwen3.6-27B-Q8_0.gguf
DRAFT=/root/models/draft-owntrim-nvfp4head-q4blk.gguf
PP512=research/e2e/prompts/pp512.txt
PLONG=research/e2e/prompts/p3-agentic-long.txt
P1=research/e2e/prompts/p1-code-short.txt
P2=research/e2e/prompts/p2-code-medium.txt

gpustate() { nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader; }
log() { echo "[$(date -u +%H:%M:%SZ)] $*" >> "$R/driver.log"; }

nvidia-smi --query-gpu=timestamp,power.draw,clocks.sm,temperature.gpu,utilization.gpu,memory.used --format=csv -l 1 > "$R/gpu-1hz.csv" 2>&1 &
SMPID=$!
trap 'kill $SMPID 2>/dev/null' EXIT

# ---- cell 1: plain tg128 @ d512, interleaved arms, N=5
for r in 1 2 3 4 5; do
  for arm in nv q8; do
    M=$NV; [ "$arm" = q8 ] && M=$Q8
    log "plain-d512 $arm r$r pre: $(gpustate)"
    MEMRA_NGEN=128 MEMRA_PROMPT_FILE=$PP512 timeout 600 target/release/run-gen "$M" > "$R/plain-d512-$arm-r$r.log" 2>&1
    log "plain-d512 $arm r$r post: $(gpustate) | $(grep -oE 'generated 128 tokens in [0-9.]+s = [0-9.]+ tok/s' "$R/plain-d512-$arm-r$r.log" | head -1) | $(grep -oE 'argmax=[0-9]+ +[a-z-]* *argmax=[0-9]+ +logit maxdiff=[0-9.e-]+ +(MATCH|MISMATCH)' "$R/plain-d512-$arm-r$r.log" | head -1)"
  done
done

# ---- cell 2: pp512 prefill-only, interleaved, N=5
for r in 1 2 3 4 5; do
  for arm in nv q8; do
    M=$NV; [ "$arm" = q8 ] && M=$Q8
    log "pp512 $arm r$r pre: $(gpustate)"
    MEMRA_PP_ONLY=1 MEMRA_PROMPT_FILE=$PP512 timeout 600 target/release/run-gen "$M" > "$R/pp512-$arm-r$r.log" 2>&1
    log "pp512 $arm r$r post: $(gpustate) | $(grep -oE 'pp-only MEDIAN: [0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s' "$R/pp512-$arm-r$r.log" | head -1)"
  done
done

# ---- cell 3: pp-long (p3 ~6257 tok) prefill-only, interleaved, N=5
for r in 1 2 3 4 5; do
  for arm in nv q8; do
    M=$NV; [ "$arm" = q8 ] && M=$Q8
    log "pplong $arm r$r pre: $(gpustate)"
    MEMRA_PP_ONLY=1 MEMRA_PROMPT_FILE=$PLONG timeout 900 target/release/run-gen "$M" > "$R/pplong-$arm-r$r.log" 2>&1
    log "pplong $arm r$r post: $(gpustate) | $(grep -oE 'pp-only MEDIAN: [0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s' "$R/pplong-$arm-r$r.log" | head -1)"
  done
done

# ---- cell 4: plain tg128 @ d-long (p3), interleaved, N=5
for r in 1 2 3 4 5; do
  for arm in nv q8; do
    M=$NV; [ "$arm" = q8 ] && M=$Q8
    log "plain-dlong $arm r$r pre: $(gpustate)"
    MEMRA_NGEN=128 MEMRA_PROMPT_FILE=$PLONG timeout 900 target/release/run-gen "$M" > "$R/plain-dlong-$arm-r$r.log" 2>&1
    log "plain-dlong $arm r$r post: $(gpustate) | $(grep -oE 'generated 128 tokens in [0-9.]+s = [0-9.]+ tok/s' "$R/plain-dlong-$arm-r$r.log" | head -1)"
  done
done

# ---- cell 5: spec K-sweep K=2..5 (pp512 continuation), interleaved, N=2
for r in 1 2; do
  for K in 2 3 4 5; do
    for arm in nv q8; do
      M=$NV; [ "$arm" = q8 ] && M=$Q8
      log "spec-k$K $arm r$r pre: $(gpustate)"
      MEMRA_SPEC_K=$K MEMRA_MTP_DRAFT=$DRAFT MEMRA_NGEN=128 MEMRA_PROMPT_FILE=$PP512 timeout 900 target/release/run-spec "$M" > "$R/spec-k$K-$arm-r$r.log" 2>&1
      log "spec-k$K $arm r$r post: $(gpustate) | $(grep -oE '\[generate_spec K=[0-9]+\] [0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s' "$R/spec-k$K-$arm-r$r.log" | head -1) | $(grep -oE 'acceptance: [0-9/]+ = [0-9.]+%' "$R/spec-k$K-$arm-r$r.log" | head -1) | $(grep -oE 'self-consistency: (PASS|FAIL)' "$R/spec-k$K-$arm-r$r.log" | head -1)"
    done
  done
done

# ---- cell 6: spec board classes p1/p2/p3 at K=3, interleaved, N=5
for r in 1 2 3 4 5; do
  for cls in p1 p2 p3; do
    P=$P1; [ "$cls" = p2 ] && P=$P2; [ "$cls" = p3 ] && P=$PLONG
    for arm in nv q8; do
      M=$NV; [ "$arm" = q8 ] && M=$Q8
      log "spec-$cls-k3 $arm r$r pre: $(gpustate)"
      MEMRA_SPEC_K=3 MEMRA_MTP_DRAFT=$DRAFT MEMRA_NGEN=128 MEMRA_PROMPT_FILE=$P timeout 900 target/release/run-spec "$M" > "$R/spec-$cls-k3-$arm-r$r.log" 2>&1
      log "spec-$cls-k3 $arm r$r post: $(gpustate) | $(grep -oE '\[generate_spec K=[0-9]+\] [0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s' "$R/spec-$cls-k3-$arm-r$r.log" | head -1) | $(grep -oE 'acceptance: [0-9/]+ = [0-9.]+%' "$R/spec-$cls-k3-$arm-r$r.log" | head -1) | $(grep -oE 'self-consistency: (PASS|FAIL)' "$R/spec-$cls-k3-$arm-r$r.log" | head -1)"
    done
  done
done

log "BATTERY-A DONE"
echo "BATTERY-A DONE"
