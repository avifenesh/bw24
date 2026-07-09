#!/usr/bin/env bash
# bw24 spec round at established ST configs: fresh run-spec process per prompt, idle-wait between.
# Usage: bw24-spec-round.sh <9b|27b> <logfile>
set -euo pipefail
MODE=$1; LOG=$2
DIR=/home/avifenesh/projects/bw24/research/e2e/prompts
NGEN=256
if [ "$MODE" = 27b ]; then
  MODELDIR=/data/ai-ml/hf-models/nvidia-qwen36-27b-nvfp4
  BW_ENV="BW24_SPEC_HPOST=1 BW24_SPEC_PMIN=0.4 BW24_SPEC_NV_W4=1 BW24_FRSPEC_TRIM=/data/ai-ml/hf-models/nvidia-qwen36-27b-nvfp4/frspec-corpus-32768.gguf"
  BK=3
else
  MODELDIR=/data/ai-ml/hf-models/qwen35-9b-nvfp4-st-modelopt
  BW_ENV="BW24_SPEC_PMIN=0.3 BW24_FRSPEC_TRIM=/data/ai-ml/hf-models/qwen35-9b-nvfp4-st-modelopt/frspec-9bst-modelopt-32768.gguf"
  BK=2
fi
wait_idle() {
  for i in $(seq 60); do
    CLK=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader | tr -d ' MHz')
    [ "$CLK" -lt 1000 ] && return 0
    sleep 2
  done
}
{
echo "### bw24 $MODE ST spec round (K=$BK, $BW_ENV) $(date -Is)"
for P in p1-code-short p2-code-medium p3-agentic-long; do
  wait_idle
  echo "=== bw24 $MODE ST $P (K=$BK) ==="
  echo "gpu-pre: $(nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader)"
  env $BW_ENV BW24_PROMPT="$(cat "$DIR/$P.txt")" \
    BW24_SPEC_K=$BK BW24_NGEN=$NGEN /home/avifenesh/projects/bw24/target/release/run-spec "$MODELDIR" 2>/dev/null \
    | grep -E "text prompt|generate\]|K=$BK\]" | head -4
  echo "gpu-post: $(nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader)"
done
} 2>&1 | tee "$LOG"
