#!/usr/bin/env bash
# bw24 EC2 head-to-head setup — g6e.12xl (4x L40S, sm_89 Ada, 48GB ea, 192GB total).
# PURPOSE: the vLLM vs SGLang vs llama.cpp three-way we've been blocked on (laptop venv broken,
#          Qwen3_5 multimodal-registry dead-end, vLLM never installed). Fresh box clears all of it.
# NOT for exact-arch tuning: bw24's kernels are sm_120a (mxf4nvf4/m16n8k32.s8) — they DO NOT run on
#          Ada. No bw24 here, no kernel_check here. Ada = FP8 tensor cores, NO native FP4.
# BRIDGE to laptop: llama.cpp runs the SAME f16 GGUF on both boxes -> ratio ties laptop<->L40S.
set -euo pipefail
log(){ printf '\n=== %s ===\n' "$*"; }

log "0. box facts"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv 2>/dev/null || true
nproc; free -g | head -2

log "1. system deps"
sudo apt-get update -y && sudo apt-get install -y build-essential cmake git python3-venv python3-pip libcurl4-openssl-dev || true

log "2. CUDA check (g6e DLAMI usually ships CUDA 12.x)"
nvcc --version 2>/dev/null || echo "no nvcc in PATH — try: export PATH=/usr/local/cuda/bin:\$PATH"
ls -d /usr/local/cuda* 2>/dev/null || true

# ---- engines ----
# vLLM + SGLang both want a CUDA-12 torch venv. Ada (sm_89) is fully supported by stock wheels.
log "3. python venv + engines (vLLM, SGLang)"
python3 -m venv ~/bench-venv
source ~/bench-venv/bin/activate
pip install -U pip wheel
# vLLM: stock wheel supports sm_89. FP8 (not FP4) on Ada.
pip install vllm 2>&1 | tail -5 || echo "vLLM install issue — see above"
# SGLang: the laptop blocker was sgl-kernel pkg-name + multimodal registry; fresh wheel should be clean.
pip install "sglang[all]" 2>&1 | tail -5 || echo "SGLang install issue — see above"
pip install huggingface_hub

log "4. llama.cpp (CUDA build, sm_89) — the BRIDGE engine"
cd ~
if [ ! -d llama.cpp ]; then git clone https://github.com/ggml-org/llama.cpp; fi
cd llama.cpp
# pin to laptop commit for an apples-to-apples bridge
git fetch --all 2>/dev/null || true
git checkout c57607016 2>/dev/null || echo "commit not fetched — using HEAD (note the drift)"
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 >/dev/null
cmake --build build --config Release -j --target llama-bench llama-cli 2>&1 | tail -3

log "5. models"
mkdir -p ~/models && cd ~/models
# HF safetensors (vLLM/SGLang native). Use HF_TOKEN env if the repo is gated.
# 9B HF bf16 (vLLM/SGLang) — adjust repo id to the actual one you used locally.
echo "Download HF models here, e.g.:"
echo "  huggingface-cli download Qwen/Qwen3.5-9B --local-dir ~/models/qwen35-9b-hf"
echo "  huggingface-cli download sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP --local-dir ~/models/qwen36-27b-hf"
# BRIDGE GGUF for llama.cpp on BOTH boxes (f16, runs on sm_89 + sm_120):
echo "  # scp the f16 gguf from laptop, or download a matching f16 gguf:"
echo "  # scp laptop:/data/ai-ml/hf-models/qwen35-9b-judge-f16.gguf ~/models/"

log "DONE. Next: run bench.sh after models land."
