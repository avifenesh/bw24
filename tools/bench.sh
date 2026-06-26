#!/usr/bin/env bash
# bw24 vs llama.cpp vs vLLM vs SGLang — single-stream prefill+decode, N medians.
# SERIAL by design: each engine runs ALONE. vLLM/SGLang GPU-memory profiling RACES if other GPU
# procs allocate/free during init (that race killed the first vLLM run). NEVER run two engines
# (or a bench + bw24) concurrently. Build jobs capped via .cargo/config.toml (jobs=6).
set -euo pipefail
export PATH=/usr/local/cuda-13.1/bin:$PATH
gpu-full-power on >/dev/null 2>&1 || true
N=${N:-5}

# guard: warn if other compute procs hold GPU mem (vLLM/SGLang init will race)
others=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | awk -F, '$2+0 > 500' || true)
[ -n "$others" ] && { echo "WARN: other GPU compute procs hold >500MiB (vLLM/SGLang init may race):"; echo "$others"; }

M9=/home/avifenesh/ai-ml/models/qwen3.5-9b-judge-q8_0.gguf
LB=$HOME/projects/llama.cpp/build/bin/llama-bench

echo "### llama.cpp 9B Q8_0 (pp64 / tg32, N=$N) ###"
$LB -m "$M9" -p 64 -n 32 -ngl 99 -r "$N" 2>/dev/null | grep -E "pp64|tg32" || echo "llama-bench failed"

echo "### bw24 9B Q8_0 decode (FAST, $N runs) ###"
for i in $(seq 1 "$N"); do
  BW24_FAST=1 BW24_NGEN=32 cargo run -q --release -p bw24-engine --bin run-gen -- "$M9" 55 100 200 2>/dev/null \
    | grep -oE "[0-9.]+ tok/s" | head -1
done

# vLLM / SGLang: best-tuned commands live in COMPETITOR-SETUP.md (from the best-setup workflow).
# Run them SEPARATELY, only when the GPU is otherwise idle (serial). Not auto-run here -> avoids the race.
echo "### vLLM / SGLang: run via COMPETITOR-SETUP.md, GPU must be idle (serial) ###"
