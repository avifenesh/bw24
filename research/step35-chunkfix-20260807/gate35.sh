#!/usr/bin/env bash
# lane/step35-chunkfix: the chunkinv35 gate (must be GREEN) + its canary (must have teeth).
# The gate is the whole point of the fix: pre-fix this returned CHUNK-DEPENDENT.
set -uo pipefail
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
cd "$HOME/step37/memra"
export MEMRA_STEP37_GGUF=$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf
export MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1
RAW=$HOME/step37/raw; mkdir -p "$RAW"
TS=$(date -u +%Y%m%dT%H%M%SZ); LOG=$RAW/gate35-$TS.log
ARGS=(--label step35-swa --prompts research/chunk-invariance-20260805/prompt-pp6257.txt
      --chunks 4096,513,512,256,64 --seam MEMRA_STEP35_SWA_TKV --steps 24)
{
echo "=== lane/step35-chunkfix chunkinv35 gate $TS  commit=$(cat BOX-COMMIT.txt)"
(
  flock -w 3600 9 || { echo "LOCK TIMEOUT"; exit 75; }
  echo "lock acquired $(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

  echo; echo "########## chunkinv35 (assert INVARIANT, naked — the fix's gate) ##########"
  timeout 5400 tools/chunk-invariance-gate.sh "${ARGS[@]}"
  echo "=== chunkinv35 rc=$?"

  echo; echo "########## chunkinv35c (canary: MEMRA_STEP35_SWA_TKV=1 must BREAK it) ##########"
  timeout 5400 tools/chunk-invariance-gate.sh "${ARGS[@]}" --canary
  echo "=== chunkinv35c rc=$?"

  echo; echo "########## legacy-seam control: --expect-variant (pre-fix predicate still reproduces) ##########"
  timeout 5400 tools/chunk-invariance-gate.sh "${ARGS[@]}" --expect-variant
  echo "=== expect-variant rc=$?"

  echo; echo "########## qwen chunkinv arm UNAFFECTED (default label/seam/prompts) ##########"
  timeout 3600 tools/chunk-invariance-gate.sh
  echo "=== qwen chunkinv rc=$?"

  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  echo "lock released $(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
echo "=== done $(date -u +%FT%TZ)"
} > "$LOG" 2>&1
echo "LOG=$LOG"
