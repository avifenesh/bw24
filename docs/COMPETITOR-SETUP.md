# COMPETITOR-SETUP.md — peak single-stream setups on RTX 5090 Laptop

Beat-target reference for the bw24 benchmark. Every competitor is tuned to its **best** runnable single-stream config on this box (consumer Blackwell GB203, sm_120, 24 GB, 858 GB/s measured read wall, 82 SMs, thermal-bound). We beat them at their peak, not their defaults.

_Measured tok/s values are point-in-time records from the tuning session — both engines move. The current board lives in the README performance section and `research/tune-data/rig5090.jsonl`; this file documents configs (build flags, serve lines, model files), which change rarely._

Daily models: **Qwen3.5-9B** and **Qwen3.6-27B** (hybrid GDN: gated-deltanet linear-attn + periodic full-attn + MTP). The 35B-A3B MoE setups live in per-engine sections of the research dump; this is the 9B + 27B copy-paste reference.

Box prep (run once per session, before any engine):
```bash
export PATH=/usr/local/cuda-13.1/bin:$PATH
gpu-full-power on                 # /home/avifenesh/.local/bin/gpu-full-power
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader   # MUST be empty (serial!)
```
**SERIAL RULE:** never run two engines (or a bench + bw24) at once. vLLM/SGLang GPU-memory
profiling RACES any other allocator during init. One engine on the GPU at a time, period.

---

## 0. TL;DR — which format fits 24 GB, per engine, per model

| Model | llama.cpp (GGUF) | vLLM (HF safetensors) | SGLang (HF safetensors) |
|-------|------------------|------------------------|--------------------------|
| 9B    | NVFP4 5.7 GB ✅   | NVFP4 modelopt ~5.3 GB ✅ (or bf16 ~18 GB, tight) | FP8 ~9–10 GB ✅ |
| 27B   | NVFP4+Q4_K_M 16 GB ✅ | NVFP4 modelopt ~14–15 GB ✅ | **NVFP4 modelopt_fp4 ~14–15 GB only** — FP8 (~27 GB) does NOT fit ❌ |

Hard non-starters on 24 GB (all engines): 27B bf16 (~54 GB), 27B FP8 (~27 GB), 35B Q6_K_XL (~31 GB).

---

## 1. llama.cpp — PEAK (this is the primary beat-target; numbers MEASURED on this box)

Build: b9743 (c57607016), `arch = sm_120a` SASS, CUDA 13.1, compile flags `GGML_CUDA_FA=ON GGML_CUDA_GRAPHS=ON GGML_CUDA_FA_ALL_QUANTS=ON`, `FORCE_MMQ=OFF`. Bins: `/home/avifenesh/projects/llama.cpp/build/bin/`. **Set `GGML_CUDA_GRAPH_OPT=1` before every run.**

### 1a. llama-bench — RAW pp/tg (fair, kernel-speed apples-to-apples; CANNOT do MTP)
```bash
export GGML_CUDA_GRAPH_OPT=1
LB=/home/avifenesh/projects/llama.cpp/build/bin/llama-bench

# 9B NVFP4 (the winner format — beats Q8_0 decode by +50%)
$LB -m /data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf \
  -ngl 999 -fa 1 -ctk q8_0 -ctv q5_1 -p 2048 -n 128 -d 0,4096,8192 -r 5
#   MEASURED: pp512=6220 t/s, tg128=126.6 t/s (f16 KV); q8_0/q5_1 KV: 5961 pp / 123.9 tg; d8192: 5531 pp / 116.8 tg

# 27B NVFP4+Q4_K_M (14.7 GiB on disk; the file shows 16 GiB)
$LB -m /data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf \
  -ngl 999 -fa 1 -ctk q8_0 -ctv q5_1 -p 2048 -n 128 -d 0,4096,8192 -r 5
#   MEASURED: pp512=1980 t/s, tg128=42.1 t/s (raw, no MTP)
```

### 1b. llama-server — PEAK single-stream WITH MTP spec-decode (the real beat-target for decode)
```bash
export GGML_CUDA_GRAPH_OPT=1
LS=/home/avifenesh/projects/llama.cpp/build/bin/llama-server

# 27B NVFP4 PEAK — MTP gives +58% decode (MEASURED 42.1 -> 66.6 t/s, accept 0.762, mean acc len 3.29)
# GOTCHA: the "-mtp.gguf" file has NO embedded NextN head — you MUST pass the external draft (-md).
$LS -m /data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf \
  -md /data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/mtp-Qwen3.6-27B-NVFP4.gguf \
  -ngl 999 -fa on -ctk q8_0 -ctv q5_1 -c 65536 --parallel 1 \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.2 \
  --jinja --temp 0.6 --top-p 0.95 --top-k 20 --host 127.0.0.1 --port 8099
#   MEASURED: TG 66.6 t/s (MTP on) vs 42.1 raw; peak VRAM 19.5/24 GiB @ 8K ctx (draft adds ~3.4 GiB)

# 9B NVFP4 PEAK (already 126 t/s raw; no separate draft file on disk -> run raw, MTP not wired for 9B here)
$LS -m /data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf \
  -ngl 999 -fa on -ctk q8_0 -ctv q5_1 -c 65536 --parallel 1 \
  --jinja --temp 0.6 --top-p 0.95 --top-k 20 --host 127.0.0.1 --port 8099
```
- **Format/quant:** NVFP4 GGUF (native sm_120a FP4 tensor cores). Fastest + smallest. 9B = 5.7 GB, 27B = 16 GB.
- **Attention:** `-fa on` (llama.cpp's ggml FA kernel, sm_120a SASS — not FlashInfer/FA3/FA4).
- **KV quant:** `-ctk q8_0 -ctv q5_1` (~2% decode cost, lets 64k ctx fit). MTP draft ctx is f16.
- **Graphs:** ON (compiled-in), plus `GGML_CUDA_GRAPH_OPT=1`. No enforce-eager penalty.
- **Spec:** `--spec-type draft-mtp` (b9743 token; not old `mtp`). `--spec-draft-n-max 3` sweet spot.

### Fits-24 GB note (llama.cpp)
9B NVFP4 (5.7 GB): trivial. 27B NVFP4 (16 GB) + MTP draft (5.9 GB) peaks 19.5 GiB @ 8K ctx, fits with room at 64k with q8_0/q5_1 KV. 27B bf16 (~54 GB) and 35B Q6_K_XL (~31 GB) do not fit.

---

## 2. vLLM 0.23.0 — PEAK

venv: `/data/projects/bench-engines/vllm-venv` (torch 2.11+cu130, CUDA 13.0, sees cap 12,0). Native
sm_120 SASS in `_C.abi3.so`; `cutlass_scaled_mm_supports_fp4(120)=True` (native NVFP4 W4A4 present).

> **Must download a pre-quantized HF checkpoint** — vLLM cannot run the on-disk GGUF efficiently
> (slow dequant path, no qwen3_5 hybrid GGUF loader). 27B/35B bf16 cannot even be loaded to quantize
> on the fly (needs full bf16 in VRAM first). Use modelopt-NVFP4 safetensors with **bf16-restored MTP**.

```bash
source /data/projects/bench-engines/vllm-venv/bin/activate

# 27B PEAK — HF modelopt-NVFP4 + MTP n=3 + FP8 KV + CUDA graphs ON
VLLM_ATTENTION_BACKEND=FLASHINFER \
vllm serve sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP \
  --quantization modelopt \
  --language-model-only --trust-remote-code \
  --tensor-parallel-size 1 \
  --max-model-len 65536 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 8192 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.92 \
  --mamba-cache-mode align \
  --max-cudagraph-capture-size 32 \
  --enable-prefix-caching --enable-chunked-prefill \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'

# 9B PEAK — NVFP4 modelopt HF checkpoint (~5.3 GB, lots of headroom -> max-model-len 131072)
VLLM_ATTENTION_BACKEND=FLASHINFER \
vllm serve <qwen35-9b NVFP4 modelopt HF repo> \
  --quantization modelopt --language-model-only --trust-remote-code \
  --tensor-parallel-size 1 --max-model-len 131072 --max-num-seqs 1 \
  --max-num-batched-tokens 8192 --kv-cache-dtype fp8 --gpu-memory-utilization 0.92 \
  --mamba-cache-mode align --max-cudagraph-capture-size 32 \
  --enable-prefix-caching --enable-chunked-prefill --reasoning-parser qwen3 \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'
# Fallback 9B if no NVFP4 repo: local bf16 /data/ai-ml/hf-models/qwen35-9b-hf, --max-model-len ~24576
# (bf16 weights ~18 GB leave little KV room in 24 GB; tight but runs single-stream).
```
For the **honest no-spec decode number**, drop `--speculative-config`; re-add it for the tuned number.
- **Format:** HF modelopt-NVFP4 (W4A4, native sm_120 cutlass). Prefer `modelopt` over compressed-tensors
  (the latter is the SLOW Blackwell fallback AND usually dropped the MTP head → 0% accept).
- **Attention:** `VLLM_ATTENTION_BACKEND=FLASHINFER` (FlashInfer 0.6.12, supports 7.5≤cap≤12.1).
  NEVER FLASH_ATTN/FA3 on sm_120 (kernels are sm_80-only; + FA rejects fp8 KV). TRITON_ATTN is the fallback.
- **KV quant:** `--kv-cache-dtype fp8` (e4m3, +12% throughput; MUST pair with FlashInfer). NOT nvfp4 (sm_100 only).
- **Graphs:** ON (FULL_AND_PIECEWISE). NEVER `--enforce-eager` (~5–8× slower on sm_120).
  `--max-cudagraph-capture-size 32` avoids the Mamba-cache capture crash at low max-num-seqs.
- **Spec:** `{"method":"qwen3_5_mtp","num_speculative_tokens":3}` (auto-normalizes to mtp; n=3 sweet spot).

### Fits-24 GB note (vLLM)
9B NVFP4 ~5.3 GB ✅ (huge headroom). 27B NVFP4 ~14–15 GB + fp8 KV ✅ (cap `--gpu-memory-utilization 0.92`; subtract 1–3 GB if display/compositor attached). 27B FP8 ~27 GB ❌, 27B bf16 ~54 GB ❌. Keep `--max-num-seqs 1` for fair single-stream bench (vLLM peaks at 2 concurrent on the 5090, but we hold single-stream across engines).

---

## 3. SGLang 0.5.9 — PEAK

venv: `/data/projects/bench-engines/sglang-venv` (torch 2.9.1+cu128, CUDA 12.8 → is_sm120_supported()
AND is_blackwell() both True). qwen35 is hybrid GDN → **full-attention backend MUST be `triton`** on
sm_120 in 0.5.9 (trtllm_mha is sm_100-only here; FlashInfer/FA3/FA4 are rejected by the hybrid-GDN
assertion). GGUF does NOT load (no GDN/mamba conv_state tensor-name mapping) → HF safetensors only.

```bash
source /data/projects/bench-engines/sglang-venv/bin/activate

# 9B — FP8 HF safetensors (~9–10 GB, fits with room)
SGLANG_USE_CUTEDSL_GDN_DECODE=1 \
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-9B-FP8 \
  --tp-size 1 \
  --attention-backend triton \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.85 \
  --context-length 65536 \
  --cuda-graph-max-bs 8 \
  --chunked-prefill-size 4096 \
  --reasoning-parser qwen3 \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --max-running-requests 1 --host 127.0.0.1 --port 30000

# 27B — dense FP8 (~27 GB) does NOT fit 24 GB. Use NVFP4 (modelopt_fp4) ~14–15 GB:
SGLANG_USE_CUTEDSL_GDN_DECODE=1 \
python -m sglang.launch_server \
  --model-path <Qwen3.6-27B NVFP4 modelopt_fp4 HF repo> \
  --quantization modelopt_fp4 \
  --tp-size 1 \
  --attention-backend triton \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.80 \
  --context-length 32768 \
  --cuda-graph-max-bs 4 \
  --chunked-prefill-size 2048 \
  --reasoning-parser qwen3 \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --max-running-requests 1 --host 127.0.0.1 --port 30000
```
- **Format:** 9B = FP8 HF safetensors (reliable, official Qwen repos). 27B = **NVFP4 (modelopt_fp4) only**
  (FP8 27B exceeds 24 GB). GGUF/AWQ/GPTQ: skip (no hybrid loader / no published checkpoints / slower).
- **Attention:** `--attention-backend triton` MANDATORY (hybrid-GDN constraint on Blackwell).
- **KV quant:** `--kv-cache-dtype fp8_e4m3` (triton supports FP8 KV). Mamba SSM state dtype is separate.
- **Graphs:** ON. Small `--cuda-graph-max-bs` (4–8) for single-stream cuts capture memory / avoids OOM.
  Do NOT `--disable-cuda-graph`; do NOT `--enable-torch-compile`.
- **Spec:** `--speculative-algorithm NEXTN` (= MTP; auto-rewrites to EAGLE). `--speculative-eagle-topk 1`
  is REQUIRED on the triton/GDN path (topk>1 needs flashinfer/fa3 paging the constraint forbids).
  MTP head is embedded in the checkpoint — no separate draft path.

### Fits-24 GB note (SGLang)
9B FP8 ~9–10 GB ✅. **27B FP8 ~27 GB ❌ — must use NVFP4 modelopt_fp4 (~14–15 GB) ✅.** Trim `--mem-fraction-static` to 0.80 and `--cuda-graph-max-bs` to 4 if OOM at load with fp8 KV + long ctx.

---

## 4. ENGINE THAT CANNOT RUN A DAILY MODEL (flagged)

- **SGLang 27B-dense FP8:** genuinely cannot run on 24 GB — FP8 weights are ~27 GB > 24 GB. Mitigation
  used above: NVFP4 (`modelopt_fp4`) checkpoint instead. If no NVFP4 27B repo is available, SGLang
  cannot serve the 27B at all on this box.
- **SGLang GGUF (any daily model):** cannot load — GGUFModelLoader has no tensor-name mapping for the
  GatedDeltaNet/mamba conv1d/conv_state/A_log weights. Daily GGUFs (Q8_0/NVFP4/IQ4_XS/Q6_K) are
  llama.cpp/bw24-only. SGLang requires HF safetensors.
- **vLLM GGUF:** loads but is the slow dequant path with no qwen3_5 hybrid GGUF loader — not used; HF
  modelopt-NVFP4 required. 27B/35B bf16 cannot be loaded to quantize on-the-fly (won't fit to start).
- **vLLM/SGLang FlashAttention / FA3 / FA4:** prebuilt kernels are sm_80-only and do not run on sm_120;
  forcing them errors or PTX-JITs. Use FLASHINFER (vLLM) / triton (SGLang).

---

---

## 5. FAIR BENCHMARK PROTOCOL (identical across all 4 engines incl bw24)

Fair benchmark requires: **same prompt, same generation length, single-stream, with warmup, N=5 medians, gpu-full-power on**, and prefill/decode tok/s extracted identically. Driver: `tools/bench.sh`.

### 5.1 Invariants (held identical for all 4)
- **Hardware state:** `gpu-full-power on`; `--prio 3` where supported; GPU otherwise idle (serial).
- **Single stream:** parallelism = 1 everywhere (llama `--parallel 1`; vLLM `--max-num-seqs 1`;
  SGLang `--max-running-requests 1` + bench `--max-concurrency 1`; bw24 is inherently single-stream).
- **Prompt:** ONE fixed prompt of **P = 512 tokens** (prefill workload). Use the same tokenized prompt
  string for the server engines; for bw24 (token-id CLI) use a fixed 512-id list of the SAME content.
- **Generation length:** **N_GEN = 128** new tokens, **greedy / temp 0** (`ignore_eos`/`-n 128`), so
  acceptance and length are deterministic and the decode count is identical.
- **Context depth sweep:** `d = 0` for the headline; optionally `d = 4096, 8192` (llama-bench `-d`,
  servers via a primed conversation) for a long-context curve. Headline number is `d=0`.
- **Warmup:** 1 discarded run per engine (loads weights, JIT, captures CUDA graphs) before timed runs.
- **Repetitions:** **N = 5** timed runs; report the **median** (thermal-bound laptop drifts down under
  sustained load → median, not peak). `N` is overridable via `N=...` env.
- **KV quant:** matched where the engine supports it — llama `q8_0/q5_1`, vLLM/SGLang `fp8(_e4m3)`.
  bw24 currently uses its native KV (KV-quant kernels are task #2, not yet landed); note this in results.

### 5.2 What we measure and HOW (identical definition)
- **Prefill tok/s = P / prefill_time**, where `prefill_time` is wall-clock from prompt submission to the
  first generated token (TTFT minus queueing), for a `P=512` prompt.
- **Decode tok/s = (N_GEN − 1) / decode_time**, where `decode_time` is wall-clock from the first to the
  last generated token (excludes prefill), for `N_GEN=128` greedy tokens.
- **Two decode numbers per engine** that supports spec: **(a) no-spec** (apples-to-apples kernel speed)
  and **(b) MTP/NEXTN-on** (peak tuned). Report both; bw24 must be compared against the no-spec number
  for kernel fairness and against the spec number for the headline "beat them at peak".

Extraction per engine:
| Engine | Tool | Prefill source | Decode source |
|--------|------|----------------|---------------|
| llama.cpp | `llama-bench -p 512 -n 128 -r 5` | `pp512` t/s | `tg128` t/s (no-spec). MTP via `llama-server` `prompt eval`/`eval` lines (`tg` with `--spec-type draft-mtp`). |
| vLLM | `vllm bench` / Python `LLM.generate` w/ `SamplingParams(max_tokens=128, ignore_eos=True)` | `prompt_throughput` / TTFT-derived | `decode_throughput` (run once without, once with `--speculative-config`). |
| SGLang | `sglang.bench_one_batch_server --batch-size 1 --input-len 512 --output-len 128` (or `bench_serving --max-concurrency 1`) | reported input throughput / TTFT | reported output throughput (run with/without `--speculative-algorithm NEXTN`). |
| bw24 | `run-gen <model> <512 prompt ids>` w/ `BW24_NGEN=128` | see 5.3 (currently NOT timed separately) | the printed `... tok/s` line (decode-only; prefill done via decode_step loop, excluded from the timer). |

### 5.3 bw24 honesty notes (do not paper over)
- `run-gen` (`crates/bw24-engine/src/bin/run_gen.rs`) currently times **decode only** (`tok/s` line at
  L46): it loops `decode_step` over the prompt to warm the cache, then times `N_GEN` greedy steps. It
  does **not** print a separate prefill tok/s, and prefill is done as a decode_step loop (not a batched
  prefill GEMM). For a fair PREFILL comparison, bw24 needs a `run-gen` addition that times
  `forward_last(prompt)` (the batched prefill path) and prints `prefill P/dt tok/s`. Until that lands,
  bw24's prefill is reported as **"N/A (batched-prefill timing not yet in run-gen)"**, not faked.
- bw24 spec-decode (MTP) is roadmap task #9, not yet landed → bw24's decode number is **no-spec**, so it
  must be compared against the competitors' **no-spec** decode for fairness, and we explicitly note that
  the competitors' headline uses MTP that bw24 does not yet have.
- bw24 KV-quant (task #2) not landed → bw24 KV is native; competitors use q8_0/q5_1 or fp8. Note it.

### 5.4 Run order (serial)
1. `gpu-full-power on`; confirm GPU idle.
2. llama-bench (9B, 27B) → pp512 / tg128, N=5 median.
3. llama-server + MTP (27B) → decode-with-spec, N=5 median. Kill server.
4. bw24 run-gen (9B, 27B) → decode tok/s, N=5 median.
5. vLLM serve (9B, 27B) → no-spec then MTP decode, N=5 median. Kill server.
6. SGLang launch (9B, 27B) → no-spec then NEXTN decode, N=5 median. Kill server.
Each step fully releases the GPU before the next (`tools/bench.sh` enforces the serial guard).

---

## 6. EXPECTED RANKING — where bw24 must land to win

All targets are the MEASURED competitor numbers on this box (the bar to clear). Win = strictly faster
than the best competitor on that metric, single-stream, same protocol.

### 9B (decode, tok/s)
| Engine | Format | Decode no-spec | Decode peak (spec) |
|--------|--------|----------------|---------------------|
| llama.cpp | NVFP4 | **126.6** (the bar) | 126.6 (no 9B draft on disk) |
| vLLM | NVFP4 | ~70–90 (FlashInfer, est.) | ~85–100 (MTP n=3) |
| SGLang | FP8 | ~70–90 (triton, est.) | higher w/ NEXTN |
| **bw24 (now)** | Q8_0 dp4a | **59.6** | n/a |
- **9B target: beat 126.6 tok/s decode (llama.cpp NVFP4).** bw24 is at 59.6 (Q8_0); the gap is ~2.1×.
  Closing it needs the decode host-round-trip removal (#1, done) + NVFP4 GEMM + CUDA-graph decode (#4).
  Realistic intermediate win: beat vLLM/SGLang no-spec (~70–90) first, then llama.cpp.

### 27B (decode, tok/s)
| Engine | Format | Decode no-spec | Decode peak (spec) |
|--------|--------|----------------|---------------------|
| llama.cpp | NVFP4 | **42.1** (no-spec bar) | **66.6** (MTP, the peak bar) |
| vLLM | NVFP4 | est. lower than llama raw | ~85–100 (community 24 GB laptop, MTP) |
| SGLang | NVFP4 | est. | NEXTN |
| **bw24 (now)** | — | not yet benched at 27B decode | n/a |
- **27B no-spec target: beat 42.1 tok/s** (this is the apples-to-apples kernel bar — bw24's first win).
- **27B peak target: beat 66.6 tok/s (llama.cpp MTP)** and the ~85–100 community vLLM-MTP number.
  bw24 reaches this only after MTP spec-decode (#9) lands. Until then, the honest claim is "beat the
  no-spec decode (42.1)".

### Prefill (tok/s)
| Model | llama.cpp pp512 (bar) | bw24 |
|-------|------------------------|------|
| 9B | **6220** | N/A until batched-prefill timing in run-gen |
| 27B | **1980** | N/A until batched-prefill timing in run-gen |
- **Prefill targets: beat 6220 (9B) / 1980 (27B) pp512.** Requires the hand-written FA prefill (#3) +
  NVFP4 MMQ prefill path, and adding the prefill timer to run-gen (5.3). bw24's structural prefill
  advantage candidate: it fits the 35B MoE in 24 GB where llama.cpp full-offload OOMs (already proven,
  argmax=1178 match) — the "wins where they OOM" angle is a real differentiator even before raw tok/s.

### Where bw24 already wins (capability, not just speed)
- **35B-A3B MoE fits 24 GB (~4 GB peak via EDGE-1 selective expert staging)** where llama.cpp
  full-offload OOMs at 30.5 GB. That is a clean capability win to headline alongside the tok/s gaps.

### Headline bar to beat (single number per metric, 24 GB sm_120, single-stream)
- 9B decode: **126.6 tok/s** (llama.cpp NVFP4).
- 27B decode no-spec: **42.1 tok/s**; 27B decode peak: **66.6 tok/s** (llama.cpp MTP).
- 9B prefill: **6220 pp512**; 27B prefill: **1980 pp512** (llama.cpp NVFP4).

The driver `tools/bench.sh` produces the bw24-vs-llama.cpp side of this table automatically; vLLM/SGLang
are run separately (serial) via the section 2/3 commands and pasted into `research/benchmarks.md`.

## Gemma-4 pairing note (2026-07-15)

`llama-bench` `-fa auto` (the default) resolves flash-attention **off** for the gemma-4
GGUFs on this build and silently costs llama ~6-11% — always pass `-fa 1` when pairing
gemma cells (26B short reads 168 under auto vs 190 with `-fa 1`). KV-quant flags
(`-ctk q8_0 -ctv q8_0`) LOSE on gemma (26B short 174.7): llama's best gemma config is
plain `-fa 1` with f16 KV. Power state must be pinned per window (`gpu-full-power on|off`)
— both profiles pair fairly but are not comparable to each other.
