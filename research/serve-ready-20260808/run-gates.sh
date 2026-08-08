#!/usr/bin/env bash
# SERVE-READY RECEIPT — gates on the serve binary/tree at the tip (ed1550f8 base):
#   G1 serve-smoke FULL over PP-2 with the Step trunk + MTP drafter (the deployment shape)
#   G2 run-gen argmax spot-check over PP-2 (prefill-vs-decode agreement)
# One flock window each; cards verified back before release.
set -uo pipefail
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
ROOT=$HOME/serve-receipt
REPO=$ROOT/memra
MODEL=$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf
DRAFT=$HOME/step37/models/step-3.7-flash/Step3.7-flash-mtp-Q8_0.gguf
RAW=$ROOT/raw
mkdir -p "$RAW"
cd "$REPO" || exit 1
TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG=$RAW/gates-$TS.log
{
echo "=== serve-ready receipt GATES $TS"
echo "commit=$(cat $ROOT/COMMIT.txt)"
echo "binary sha256=$(sha256sum target/release/memra-server | awk '{print $1}')"
(
  flock -w 10800 9 || { echo "LOCK TIMEOUT gates"; exit 75; }
  echo "lock acquired $(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=index,temperature.gpu,memory.used --format=csv,noheader

  echo; echo "########## G1: serve-smoke FULL, PP-2, step35 trunk+drafter ##########"
  MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 timeout 7200 \
    bash tools/serve-smoke.sh "$MODEL" "$DRAFT"
  echo "serve-smoke exit=$?"
  sleep 3
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

  echo; echo "########## G2: run-gen argmax spot-check, PP-2, 64 tokens ##########"
  MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_NGEN=64 timeout 2400 \
    ./target/release/run-gen "$MODEL" --prompt "Write a short paragraph explaining how a CPU pipeline improves instruction throughput, and mention one hazard."
  echo "run-gen exit=$?"

  sleep 3
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  echo "lock released $(date -u +%FT%TZ)"
) 9>/tmp/memra-gpu.lock
echo "=== gates rc=$?"
echo "=== done $(date -u +%FT%TZ)"
} > "$LOG" 2>&1
echo "log: $LOG"
tail -20 "$LOG"
