#!/usr/bin/env bash
# pro6000-dev: (1) exactness gates with winner MEMRA_NV_MR=1 forced (argmax + run-spec K=1..3),
# (2) spec K=3..6 A/B: auto vs NV_MR=1 vs NV_MR=1+MMVQ_BV=rp (the b8-tier r1-grid probe), N=2 interleaved.
set -u
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat
cd /root/bw24
R=/root/receipts-dev/exact-spec
mkdir -p "$R"
M=/root/models/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
DRAFT=/root/models/draft-owntrim-nvfp4head-q4blk.gguf
P512=research/e2e/prompts/pp512.txt
log() { echo "[$(date -u +%H:%M:%SZ)] $*" >> "$R/driver.log"; }
gpustate() { nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader; }
nvidia-smi --query-gpu=timestamp,power.draw,clocks.sm,temperature.gpu --format=csv -l 1 > "$R/gpu-1hz.csv" 2>&1 &
SMPID=$!
trap 'kill $SMPID 2>/dev/null' EXIT

# gate: argmax with winner forced
MEMRA_NV_MR=1 MEMRA_NGEN=32 MEMRA_PROMPT_FILE=$P512 timeout 600 target/release/run-gen $M > "$R/argmax-mr1.log" 2>&1
log "argmax-mr1 rc=$? | $(grep -oE '(MATCH|MISMATCH)' "$R/argmax-mr1.log" | tr '\n' ' ')"

# gate: run-spec K=1..3 with winner forced
for K in 1 2 3; do
  MEMRA_NV_MR=1 MEMRA_SPEC_K=$K MEMRA_MTP_DRAFT=$DRAFT MEMRA_NGEN=128 MEMRA_PROMPT_FILE=$P512 timeout 900 target/release/run-spec $M > "$R/spec-k$K-mr1.log" 2>&1
  log "spec-k$K-mr1 rc=$? | $(grep -oE '\[generate_spec K=[0-9]+\] [0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s' "$R/spec-k$K-mr1.log" | head -1) | $(grep -oE 'self-consistency: (PASS|FAIL)' "$R/spec-k$K-mr1.log" | head -1)"
done

# spec-tier A/B K=3..6, N=2 interleaved arms
for r in 1 2; do
  for K in 3 4 5 6; do
    for arm in auto mr1 mr1rp; do
      env=""
      [ "$arm" = mr1 ] && env="MEMRA_NV_MR=1"
      [ "$arm" = mr1rp ] && env="MEMRA_NV_MR=1 MEMRA_MMVQ_BV=rp"
      log "speck$K $arm r$r pre: $(gpustate)"
      eval "$env MEMRA_SPEC_K=$K MEMRA_MTP_DRAFT=$DRAFT MEMRA_NGEN=128 MEMRA_PROMPT_FILE=$P512 timeout 900 target/release/run-spec $M" > "$R/speck$K-$arm-r$r.log" 2>&1
      log "speck$K $arm r$r post rc=$?: $(gpustate) | $(grep -oE '\[generate_spec K=[0-9]+\] [0-9]+ tok in [0-9.]+s = [0-9.]+ tok/s' "$R/speck$K-$arm-r$r.log" | head -1) | $(grep -oE 'acceptance: [0-9/]+ = [0-9.]+%' "$R/speck$K-$arm-r$r.log" | head -1) | $(grep -oE 'self-consistency: (PASS|FAIL)' "$R/speck$K-$arm-r$r.log" | head -1)"
    done
  done
done
log "SWEEP_D_DONE"
echo SWEEP_D_DONE
