#!/usr/bin/env bash
# bw24 vs llama.cpp — prefill + decode tok/s, N medians, on the daily models.
# vLLM/SGLang require separate install (torch nightly cu130, immature sm_120) — TODO when scoped.
set -euo pipefail
export PATH=/usr/local/cuda-13.1/bin:$PATH
gpu-full-power on >/dev/null 2>&1 || true
LB=$HOME/projects/llama.cpp/build/bin/llama-bench
BW=/home/avifenesh/projects/bw24
N=${N:-5}
declare -A M=(
  [9b-q8]=/home/avifenesh/ai-ml/models/qwen3.5-9b-judge-q8_0.gguf
)
for name in "${!M[@]}"; do
  m=${M[$name]}
  echo "### $name : $m ###"
  echo "--- llama.cpp (pp64 prefill / tg32 decode) ---"
  $LB -m "$m" -p 64 -n 32 -ngl 99 -r "$N" 2>/dev/null | grep -E "pp64|tg32"
  echo "--- bw24 decode (FAST, $N runs) ---"
  for i in $(seq 1 $N); do
    BW24_FAST=1 BW24_NGEN=32 cargo run -q --release -p bw24-engine --bin run-gen -- "$m" 55 100 200 2>/dev/null | grep -oE "[0-9.]+ tok/s" | head -1
  done
done
