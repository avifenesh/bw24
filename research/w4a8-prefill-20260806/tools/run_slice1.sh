#!/bin/bash
# Slice 1 driver: wait for VRAM, take the box lock, lock clocks, run the interleaved A/B, unlock.
# Never leaves the GPU clock-locked (trap on EXIT).
set -u
W=/home/avifenesh/projects/wt-w4a8
MODEL=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
TAG=${1:-q27}
ROUNDS=${2:-3}
NEED_MIB=${NEED_MIB:-19000}

freemib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }

# 1) wait for the box: another lane's gemma-gate may hold ~21.6 GiB.
for i in $(seq 1 240); do
  f=$(freemib)
  if [ "$f" -ge "$NEED_MIB" ]; then echo "[vram] ${f} MiB free after ${i} polls"; break; fi
  sleep 30
done
f=$(freemib)
if [ "$f" -lt "$NEED_MIB" ]; then echo "[abort] only ${f} MiB free, need ${NEED_MIB}"; exit 2; fi

# 2) box lock + clock lock (unlock unconditionally on exit)
exec 9>/tmp/gpu5090.lock
flock -w 7200 9 || { echo "[abort] could not take /tmp/gpu5090.lock"; exit 3; }
cleanup(){ sudo -n nvidia-smi -rgc >/dev/null 2>&1; echo "[clocks] reset"; }
trap cleanup EXIT
sudo -n nvidia-smi -lgc 1860,1860 >/dev/null 2>&1 && echo "[clocks] locked 1860,1860" || echo "[clocks] LOCK FAILED — measurement invalid under locked-clock law"
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm --format=csv,noheader

# 3) the A/B
"$W/research/w4a8-prefill-20260806/tools/ab_f8f4.sh" "$MODEL" "$TAG" "$ROUNDS"
