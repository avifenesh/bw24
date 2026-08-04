#!/usr/bin/env bash
# q9 long-window supplement: the France prompt EOSes at 209 tok, capping every q9 window
# at 208 — the short-window trap. Long-form prompt gives real 400/800/1600 windows.
set -uo pipefail
cd "$(dirname "$0")"
PROMPT="Write a detailed essay about the history of the Roman Empire, covering its founding, expansion, governance, and fall."
Q9=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
for n in 400 800 1600; do
  for r in 1 2 3; do
    echo "=== q9long n=$n r$r ==="
    flock /tmp/gpu5090.lock ./probe.sh q9long "$Q9" 1 "$PROMPT" $n $r || echo "CELL FAILED q9long n=$n r$r"
  done
done
echo "battery-q9long done"
