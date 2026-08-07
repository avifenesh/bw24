#!/usr/bin/env bash
# lane/pp-prefill LEVER A battery 3 (post canary-seam fix 82b216b8): the canary must be RED again.
#   G2  chunkinv35 naked — INVARIANT (must not regress)
#   G2c canary — must now BREAK (both seam halves restored)
#   G2v expect-variant control — the legacy seam must reproduce the pinned divergence
set -uo pipefail
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
cd "$HOME/ppserve-memra"
M=$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf
RAW=$HOME/ppserve-raw; mkdir -p "$RAW"
TS=$(date -u +%Y%m%dT%H%M%SZ); LOG=$RAW/leverA-gates3-$TS.log
CIARGS=(--label step35-swa --prompts research/chunk-invariance-20260805/prompt-pp6257.txt
        --chunks 4096,513,512,256,64 --seam MEMRA_STEP35_SWA_TKV --steps 24)
{
echo "=== leverA battery 3 $TS commit=82b216b8 (rsync)"
(
  flock -w 14400 9 || { echo "LOCK TIMEOUT"; exit 75; }
  echo "lock acquired $(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

  echo; echo "########## G2: chunkinv35 naked (INVARIANT) ##########"
  MEMRA_STEP37_GGUF=$M MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 timeout 5400 \
    tools/chunk-invariance-gate.sh "${CIARGS[@]}"
  echo "G2 exit=$?"

  echo; echo "########## G2c: canary (must BREAK now) ##########"
  MEMRA_STEP37_GGUF=$M MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 timeout 5400 \
    tools/chunk-invariance-gate.sh "${CIARGS[@]}" --canary
  echo "G2c exit=$?"

  echo; echo "########## G2v: expect-variant control (legacy seam reproduces the divergence) ##########"
  MEMRA_STEP37_GGUF=$M MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 timeout 5400 \
    tools/chunk-invariance-gate.sh "${CIARGS[@]}" --expect-variant
  echo "G2v exit=$?"

  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  echo "lock released $(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
echo "=== battery3 rc=$?"
echo "=== done $(date -u +%FT%TZ)"
} > "$LOG" 2>&1
echo "LOG=$LOG"
