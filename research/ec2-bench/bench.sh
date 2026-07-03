#!/usr/bin/env bash
# bw24 EC2 head-to-head — the THREE-WAY we've been blocked on.
# Runs vLLM, SGLang, llama.cpp on the SAME model/box; reports prefill (pp) + decode (tg) tok/s.
# Bridge: the llama.cpp numbers here + llama.cpp on the laptop give the laptop<->L40S ratio,
#         so bw24-vs-llama(laptop) composes with vLLM/SGLang-vs-llama(here) into one ranking.
#
# MEASUREMENT DISCIPLINE (same as laptop):
#  - warm runs only (discard first/cold).
#  - one GPU at a time for timing (L40S x4: pin CUDA_VISIBLE_DEVICES=0 for single-GPU parity vs laptop;
#    use TP=4 separately to see the multi-GPU story).
#  - same prompt lengths the laptop uses: pp512 (prefill), tg64/128 (decode), ctx sweep 128/512/2048.
set -uo pipefail
log(){ printf '\n############ %s ############\n' "$*"; }

# BENCH MODEL = gemma-4-31B NVFP4 (user's choice). 31B@NVFP4 ~16-18GB -> fits ONE L40S, no TP.
# L40S=Ada sm_89, NO FP4 tensor cores -> ALL 3 engines dequant NVFP4->bf16/fp16 = fair fight.
# llama DOES support NVFP4 (GGML_TYPE_NVFP4=40) + LLM_ARCH_GEMMA4 -> stays in. needs gemma4-NVFP4 GGUF.
GEMMA_HF=${GEMMA_HF:-~/models/gemma4-31b-nvfp4-hf}   # HF NVFP4 checkpoint for vLLM/SGLang
GEMMA_GGUF=${GEMMA_GGUF:-~/models/gemma4-31b-nvfp4.gguf}  # NVFP4 GGUF for llama.cpp
OUT=~/bench-results ; mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}   # single L40S; fits NVFP4 31B

source ~/bench-venv/bin/activate 2>/dev/null || true

# ---------- 1. llama.cpp (NVFP4 gemma4) ----------
log "llama.cpp pp512 + tg128 (single L40S) — gemma4 NVFP4"
~/llama.cpp/build/bin/llama-bench -m "$GEMMA_GGUF" -p 512 -n 128 -ngl 999 -r 3 2>&1 | tee "$OUT/llama.txt"

# ---------- 2. vLLM ----------
log "vLLM offline throughput (prefill+decode) — gemma4-31B NVFP4, single L40S"
# offline LLM() avoids the `serve` daemon reaping issue we hit on the laptop.
# WATCH: modelopt-NVFP4 kernels may be Blackwell-gated -> on Ada vLLM may error 'requires sm_100+'
# or fall to a Marlin/dequant path. If it errors, retry quantization='modelopt' / dtype auto, log it.
python3 - "$GEMMA_HF" <<'PY' 2>&1 | tee "$OUT/vllm_gemma.txt"
import sys, time
from vllm import LLM, SamplingParams
m = sys.argv[1]
# gemma4 is Gemma4ForConditionalGeneration (multimodal) — text-only inference is fine.
llm = LLM(model=m, gpu_memory_utilization=0.9, enforce_eager=False, max_model_len=4096)
pp = "word " * 512
t=time.time(); o=llm.generate([pp], SamplingParams(max_tokens=1)); print(f"VLLM prefill(512) 1tok in {time.time()-t:.3f}s")
sp = SamplingParams(max_tokens=128)
for _ in range(2): llm.generate(["Hello"], sp)  # warm
t=time.time(); o=llm.generate(["Hello"], sp); dt=time.time()-t
n=len(o[0].outputs[0].token_ids)
print(f"VLLM decode: {n} toks in {dt:.3f}s = {n/dt:.1f} tok/s")
PY

# ---------- 3. SGLang ----------
log "SGLang offline (Engine API) — gemma4-31B NVFP4, single L40S"
# laptop blocker was sgl-kernel pkg-name + Qwen3_5 multimodal registry. gemma4 is mainstream-supported.
# Same NVFP4-on-Ada watch as vLLM: may refuse FP4 kernels on sm_89.
python3 - "$GEMMA_HF" <<'PY' 2>&1 | tee "$OUT/sglang_gemma.txt"
import sys, time
try:
    import sglang as sgl
    m = sys.argv[1]
    eng = sgl.Engine(model_path=m)
    for _ in range(2): eng.generate("Hello", {"max_new_tokens": 128})  # warm
    t=time.time(); o=eng.generate("Hello", {"max_new_tokens":128}); dt=time.time()-t
    print(f"SGLANG decode ~128 toks in {dt:.3f}s = {128/dt:.1f} tok/s (verify tok count)")
except Exception as e:
    print("SGLANG FAILED:", repr(e))
PY

log "RESULTS in $OUT — copy back to laptop research/ec2-bench/results/"
echo "gemma4-31B NVFP4, single L40S (Ada=no FP4 cores, all engines dequant). engine-vs-engine, best-each."
echo "FALLBACK: if an engine refuses NVFP4 on sm_89, run it bf16/FP8 and note the precision in results."
