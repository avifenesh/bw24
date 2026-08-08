#!/usr/bin/env bash
# lane/tick-seg: the tickinv35 gate (registered RED, must now be GREEN) + its canary (must have
# teeth) + chunkinv35 (axis 1 must NOT regress) + its canary + the qwen chunkinv arm (unaffected
# arch control on the box build).
set -uo pipefail
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
cd "$HOME/step37/memra"
export MEMRA_STEP37_GGUF=$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf
export MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1
RAW=$HOME/step37/raw; mkdir -p "$RAW"
TS=$(date -u +%Y%m%dT%H%M%SZ); LOG=$RAW/gate-tickseg-$TS.log
TARGS=(--label step35-tick --prompts research/chunk-invariance-20260805/prompt-pp6257.txt
       --budgets 0,1024,513,512,256,64 --splits 64,256,512 --seam MEMRA_PRIME_CALLLOCAL --steps 24)
CARGS=(--label step35-swa --prompts research/chunk-invariance-20260805/prompt-pp6257.txt
       --chunks 4096,513,512,256,64 --seam MEMRA_STEP35_SWA_TKV --steps 24)
{
echo "=== lane/tick-seg gate battery $TS  commit=$(cat BOX-COMMIT.txt)"
echo "=== build (bins this battery runs)"
cargo build --release --bin concat-prime-probe --bin run-gen --bin run-spec \
  --bin kernel-check --bin ppn-gate 2>&1 | tail -3
echo "BUILD_RC=${PIPESTATUS[0]}"
(
  flock -w 7200 9 || { echo "LOCK TIMEOUT"; exit 75; }
  echo "lock acquired $(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

  echo; echo "########## tickinv35 (assert INVARIANT, naked — the fix's gate; was RED) ##########"
  timeout 5400 tools/tick-invariance-gate.sh "${TARGS[@]}"
  echo "=== tickinv35 rc=$?"

  echo; echo "########## tickinv35c (canary: MEMRA_PRIME_CALLLOCAL=1 must BREAK it) ##########"
  timeout 5400 tools/tick-invariance-gate.sh "${TARGS[@]}" --canary
  echo "=== tickinv35c rc=$?"

  echo; echo "########## chunkinv35 (axis 1 — must NOT regress) ##########"
  timeout 5400 tools/chunk-invariance-gate.sh "${CARGS[@]}"
  echo "=== chunkinv35 rc=$?"

  echo; echo "########## chunkinv35c (axis-1 canary still has teeth) ##########"
  timeout 5400 tools/chunk-invariance-gate.sh "${CARGS[@]}" --canary
  echo "=== chunkinv35c rc=$?"

  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  echo "lock released $(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
echo "=== gate-tickseg rc=$?"
echo "=== done $(date -u +%FT%TZ)"
} > "$LOG" 2>&1
echo "LOG=$LOG"
