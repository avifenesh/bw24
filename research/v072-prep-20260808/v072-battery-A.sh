#!/usr/bin/env bash
# v0.72 pair-box battery — DRIVER A: exactness core (kernel-check, run-gen, ppn-gate, run-spec)
# Train 6afc4f65 (restructure/public-split), read-only validation. Receipts -> ~/v072/raw/.
set -uo pipefail
cd ~/v072/memra
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
BIN=target/release
STEP=$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf
STEPD=$HOME/step37/models/step-3.7-flash/Step3.7-flash-mtp-Q8_0.gguf
Q27=/scratch-models/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
Q9=/scratch-models/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
RAW=$HOME/v072/raw; mkdir -p "$RAW"
TS=$(date -u +%Y%m%dT%H%M%SZ); LOG=$RAW/exactA-$TS.log
# baseline-matched prompt (exact-tickseg E2/E4)
P="Write a short paragraph explaining how a CPU pipeline improves instruction throughput, and mention one hazard."
{
echo "=== v072 battery DRIVER A $TS commit=$(cat BOX-COMMIT.txt)"
nvidia-smi --query-gpu=index,name,memory.total,temperature.gpu --format=csv,noheader
(
  flock -w 14400 9 || { echo "LOCK TIMEOUT"; exit 75; }
  echo "lock acquired $(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

  echo; echo "########## A1: kernel-check model-backed step35 IQ4_XS ##########"
  timeout 3600 $BIN/kernel-check "$STEP"
  echo "A1 exit=$?"

  echo; echo "########## A2: kernel-check model-backed q27 NVFP4 ##########"
  timeout 3600 $BIN/kernel-check "$Q27"
  echo "A2 exit=$?"

  echo; echo "########## A3: run-gen argmax gate, step35 PP-2, 64 tok ##########"
  MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_NGEN=64 timeout 2400 \
    $BIN/run-gen "$STEP" --prompt "$P"
  echo "A3 exit=$?"

  echo; echo "########## A4: run-gen argmax gate, q27 single-card, naked ##########"
  MEMRA_NGEN=64 timeout 2400 $BIN/run-gen "$Q27" 55
  echo "A4 exit=$?"

  echo; echo "########## A5: ppn-gate stages=2 bit-identity, step35 ##########"
  MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 timeout 2400 $BIN/ppn-gate "$STEP" 2 8 16
  echo "A5 exit=$?"

  echo; echo "########## A6: ppn-gate stages=2 bit-identity, q27 ##########"
  MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 timeout 2400 $BIN/ppn-gate "$Q27" 2 8 16
  echo "A6 exit=$?"

  echo; echo "########## A7: run-spec K=1..8, step35+MTP drafter over PP-2 (baseline: 77.8% K=1, mtp-draft-PASS-20260806T215132Z) ##########"
  MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_MTP_DRAFT="$STEPD" MEMRA_NGEN=32 \
    MEMRA_PROMPT="$P" timeout 3600 $BIN/run-spec "$STEP"
  echo "A7 exit=$?"

  echo; echo "########## A8: run-spec K=1..8, q27 single-card (embedded MTP; baseline: 95.8% K=1, pp2-spec runspec-q27-doorshut) ##########"
  MEMRA_NGEN=48 timeout 3600 $BIN/run-spec "$Q27" 55
  echo "A8 exit=$?"

  echo; echo "########## A9: run-spec q9 door-shut NGEN=32 control (the pinned 82.4% K=1 receipt shape) ##########"
  MEMRA_NGEN=32 timeout 3600 $BIN/run-spec "$Q9" 55
  echo "A9 exit=$?"

  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  echo "lock released $(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
echo "=== driverA rc=$?"
echo "=== done $(date -u +%FT%TZ)"
} > "$LOG" 2>&1
echo "LOG=$LOG"
