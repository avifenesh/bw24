#!/usr/bin/env bash
# v0.72 tag-gate RE-BATTERY — R1: blocker-1 post-merge confirmation on box2.
# tickinv35 naked must PASS (bit-exact at every budget/split) AND tickinv35c canary must
# BREAK the assertion (teeth restored by 73c65c91, merged at d8363ccd; pre-merge receipts:
# research/v072-fix1-20260808/). Train tip 5ad87a63. One lock hold.
set -uo pipefail
cd ~/memra
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
export MEMRA_STEP37_GGUF=/data/models/step37/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf
export MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1
RAW=$HOME/v072rebat/raw; mkdir -p "$RAW"
TS=$(date -u +%Y%m%dT%H%M%SZ); LOG=$RAW/R1-tick-$TS.log
TARGS=(--label step35-tick --prompts research/chunk-invariance-20260805/prompt-pp6257.txt
       --budgets 0,1024,513,512,256,64 --splits 64,256,512 --seam MEMRA_PRIME_CALLLOCAL --steps 24)
{
echo "=== v072 REBATTERY R1 $TS commit=$(git rev-parse HEAD)"
(
  flock -w 14400 9 || { echo "LOCK TIMEOUT"; exit 75; }
  echo "lock acquired $(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

  echo; echo "########## R1a: tickinv35 (assert INVARIANT, naked) ##########"
  timeout 5400 tools/tick-invariance-gate.sh "${TARGS[@]}"
  echo "=== R1a tickinv35 rc=$?"

  echo; echo "########## R1b: tickinv35c (canary: MEMRA_PRIME_CALLLOCAL=1 must BREAK it) ##########"
  timeout 5400 tools/tick-invariance-gate.sh "${TARGS[@]}" --canary
  echo "=== R1b tickinv35c rc=$?"

  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  echo "lock released $(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
echo "=== R1 rc=$?"
echo "=== done $(date -u +%FT%TZ)"
} > "$LOG" 2>&1
echo "LOG=$LOG"
