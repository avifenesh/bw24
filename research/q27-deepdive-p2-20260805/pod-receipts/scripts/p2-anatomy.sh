#!/bin/bash
# PHASE-2 item 1a: spec-round anatomy, BARE layer.
#  - spec-econ v(T) verify cost curve, T=1..9, both artifacts (the 188-SM vt-fixes picture)
#  - MEMRA_SPEC_PHASE=1 round decomposition at K=3..6, both artifacts, N=3 interleaved
set -u
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat
export PATH=$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH
cd /root/bw24
R=/root/receipts-p2
NV=/root/models/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
Q8=/root/models/Qwen3.6-27B-Q8_0.gguf
DRAFT=/root/models/draft-owntrim-nvfp4head-q4blk.gguf
P512=research/e2e/prompts/pp512.txt
log() { echo "[$(date -u +%H:%M:%SZ)] $*" >> "$R/logs/anatomy-driver.log"; }
gpustate() { nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader; }

# ---- spec-econ v(T): T=1..9, N=50 (+3 warmup), arms interleaved inside the binary
log "econ nv start: $(gpustate)"
MEMRA_ECON_N=50 MEMRA_ECON_TMAX=9 MEMRA_PROMPT_FILE=$P512 timeout 1800 \
  target/release/spec-econ $NV > "$R/logs/econ-nv.log" 2>&1
log "econ nv rc=$?: $(gpustate)"
MEMRA_ECON_N=50 MEMRA_ECON_TMAX=9 MEMRA_PROMPT_FILE=$P512 timeout 1800 \
  target/release/spec-econ $Q8 > "$R/logs/econ-q8.log" 2>&1
log "econ q8 rc=$?: $(gpustate)"

# ---- MEMRA_SPEC_PHASE round decomposition: K=3..6 x r=1..3, artifact order alternated per rep
for r in 1 2 3; do
  if [ $((r % 2)) -eq 1 ]; then ARTS="nv q8"; else ARTS="q8 nv"; fi
  for art in $ARTS; do
    for K in 3 4 5 6; do
      if [ "$art" = nv ]; then M=$NV; DR=""; else M=$Q8; DR=$DRAFT; fi
      MEMRA_MTP_DRAFT=$DR MEMRA_SPEC_PHASE=1 MEMRA_SPEC_STATS=1 MEMRA_SPEC_K=$K \
        MEMRA_NGEN=256 MEMRA_PROMPT_FILE=$P512 timeout 900 \
        target/release/run-spec $M > "$R/logs/phase-$art-K$K-r$r.log" 2>&1
      log "phase $art K=$K r$r rc=$? $(grep -oE 'spec-phase.*' "$R/logs/phase-$art-K$K-r$r.log" | tail -1)"
    done
  done
done
log "ANATOMY_DONE: $(gpustate)"
echo ANATOMY_DONE
