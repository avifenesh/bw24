#!/usr/bin/env bash
# plainbatch battery: q9 GGUF + 9B ST x n=400/800/1600 x N=3 (probe.sh per cell).
# Params baked as literals (workflow-args-no-propagate).
set -uo pipefail
cd "$(dirname "$0")"
PROMPT="What is the capital of France? Answer in one short sentence."
Q9=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
ST=/data/ai-ml/hf-models/qwen35-9b-nvfp4-st-modelopt
for n in 400 800 1600; do
  for r in 1 2 3; do
    echo "=== q9 n=$n r$r ==="
    flock /tmp/gpu5090.lock ./probe.sh q9 "$Q9" 1 "$PROMPT" $n $r || echo "CELL FAILED q9 n=$n r$r"
  done
done
for n in 400 800 1600; do
  for r in 1 2 3; do
    echo "=== 9bst n=$n r$r ==="
    flock /tmp/gpu5090.lock ./probe.sh 9bst "$ST" 1 "$PROMPT" $n $r || echo "CELL FAILED 9bst n=$n r$r"
  done
done
echo "battery done"
