#!/usr/bin/env bash
# v0.72 pair-box battery — DRIVER B: the step35 gate family (both segmentation axes) +
# canaries (teeth must bite) + decode-batch config/strict on the box NVFP4 artifacts.
set -uo pipefail
cd ~/v072/memra
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
BIN=target/release
export MEMRA_STEP37_GGUF=$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf
export MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1
Q27=/scratch-models/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
Q9=/scratch-models/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
RAW=$HOME/v072/raw; mkdir -p "$RAW"
TS=$(date -u +%Y%m%dT%H%M%SZ); LOG=$RAW/gatesB-$TS.log
TARGS=(--label step35-tick --prompts research/chunk-invariance-20260805/prompt-pp6257.txt
       --budgets 0,1024,513,512,256,64 --splits 64,256,512 --seam MEMRA_PRIME_CALLLOCAL --steps 24)
CARGS=(--label step35-swa --prompts research/chunk-invariance-20260805/prompt-pp6257.txt
       --chunks 4096,513,512,256,64 --seam MEMRA_STEP35_SWA_TKV --steps 24)
{
echo "=== v072 battery DRIVER B $TS commit=$(cat BOX-COMMIT.txt)"
(
  flock -w 14400 9 || { echo "LOCK TIMEOUT"; exit 75; }
  echo "lock acquired $(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

  echo; echo "########## B1: tickinv35 (assert INVARIANT, naked) ##########"
  timeout 5400 tools/tick-invariance-gate.sh "${TARGS[@]}"
  echo "=== B1 tickinv35 rc=$?"

  echo; echo "########## B2: tickinv35c (canary: MEMRA_PRIME_CALLLOCAL=1 must BREAK it) ##########"
  timeout 5400 tools/tick-invariance-gate.sh "${TARGS[@]}" --canary
  echo "=== B2 tickinv35c rc=$?"

  echo; echo "########## B3: chunkinv35 (axis 1) ##########"
  timeout 5400 tools/chunk-invariance-gate.sh "${CARGS[@]}"
  echo "=== B3 chunkinv35 rc=$?"

  echo; echo "########## B4: chunkinv35c (axis-1 canary must have teeth) ##########"
  timeout 5400 tools/chunk-invariance-gate.sh "${CARGS[@]}" --canary
  echo "=== B4 chunkinv35c rc=$?"

  echo; echo "########## B5: decode-batch-gate q9 NVFP4 config B=8 ##########"
  env -u MEMRA_PP_STAGES -u MEMRA_PP_DEVICES timeout 2400 \
    $BIN/decode-batch-gate "$Q9" --steps 32 --batch 8 --mode config
  echo "=== B5 rc=$?"

  echo; echo "########## B6: decode-batch-gate q9 NVFP4 strict B=4 equalized ##########"
  env -u MEMRA_PP_STAGES -u MEMRA_PP_DEVICES MEMRA_MMVQ=0 MEMRA_NO_FUSE_NORMQ=1 timeout 2400 \
    $BIN/decode-batch-gate "$Q9" --steps 32 --batch 4 --mode strict
  echo "=== B6 rc=$?"

  echo; echo "########## B7: decode-batch-gate q27 NVFP4 config B=8 ##########"
  env -u MEMRA_PP_STAGES -u MEMRA_PP_DEVICES timeout 2400 \
    $BIN/decode-batch-gate "$Q27" --steps 32 --batch 8 --mode config
  echo "=== B7 rc=$?"

  echo; echo "########## B8: decode-batch-gate q27 NVFP4 strict B=4 equalized ##########"
  env -u MEMRA_PP_STAGES -u MEMRA_PP_DEVICES MEMRA_MMVQ=0 MEMRA_NO_FUSE_NORMQ=1 timeout 2400 \
    $BIN/decode-batch-gate "$Q27" --steps 32 --batch 4 --mode strict
  echo "=== B8 rc=$?"

  echo "NOTE: decode-batch Q8_0 arm SKIP — no Q8 main-model artifact on this box"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  echo "lock released $(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
echo "=== driverB rc=$?"
echo "=== done $(date -u +%FT%TZ)"
} > "$LOG" 2>&1
echo "LOG=$LOG"
