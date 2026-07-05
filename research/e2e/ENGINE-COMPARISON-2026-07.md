# 4-Engine E2E Comparison: bw24 vs llama.cpp vs vLLM vs SGLang — 2026-07-04

Extends the e2e protocol (`research/e2e/run-e2e.sh`) to vLLM and SGLang. Same three
prompts (`prompts/p1|p2|p3`, verbatim), temp 0, 256 gen tokens, single request,
1 warmup + 1 measured run per cell (N=1 — indicative, not a tuned sweep; the gaps
claimed below are 3-7x, far beyond run-to-run variance, but no sub-20% delta here
should be treated as real).

## THE ANSWER (what beats bw24 today, sized)

1. **Prefill / TTFT: vLLM and SGLang beat bw24 decisively, everywhere.**
   - 27B NVFP4, G7e: vLLM prefill **10.8-13.1k tok/s vs bw24 1.7-1.9k = 5.6-6.8x**.
     TTFT on the 6.3k-token agentic prompt: 0.58s vs 3.25s.
   - 9B, G7e: SGLang **20.0k**, vLLM 18.0-19.0k vs bw24 5.3-5.9k = **3.1-3.4x**
     (and that's bf16 engines out-prefilling a 4-bit bw24).
   - Local 5090: vLLM 27B prefill 4.6-4.9k vs bw24 ~0.79k = **~6x**.
   - Lever to adopt: **chunked-prefill scheduling + native-FP4 (cutlass) GEMM for the
     m-large regime**. vLLM's cutlass FP4 path prefilled 2.4x faster than its own
     marlin path (13.1k vs 5.5k) — the FP4 tensor-core GEMM is the prefill win,
     while staying on dequant-style matvec for decode.
2. **Decode at matched quant: bw24 still wins vs vLLM, but llama.cpp beats bw24 on G7e.**
   - 27B NVFP4 G7e: bw24 spec 102-128 tok/s vs vLLM best (marlin+MTP) 92-121.
     vLLM only closes the gap on the short prompt (121 vs 128).
   - BUT llama.cpp 27B spec-serve (f16 KV) does **130-159 tok/s — beats bw24 on all
     three prompts on G7e** (it did NOT on the local 5090). bw24's decode tuning is
     rig-specific to the laptop 5090; on the 1.8TB/s RTX PRO 6000 it leaves decode
     on the table. Same story at 9B plain: llama 182-186 vs bw24 plain 126-137.
   - vLLM CUDA-graph decode: worth ~3% at bs=1 on this hybrid-GDN model (local:
     26.8 -> 27.5 tok/s). Not a lever.
   - m=1 kernel note: on sm_120, vLLM's marlin W4A16 dequant path decodes 1.4x
     faster than its native cutlass FP4 GEMM (54.9 vs 39.5 tok/s) — native FP4
     GEMM has poor m=1 occupancy; matches community SM120 reports. bw24's MMVQ
     approach is already on the right side of this.
3. **Engine MTP spec works and is the same lever bw24 already has**: vLLM
   qwen3_5_mtp (k=3) took 27B decode 55 -> 92-121 tok/s (1.7-2.2x), acceptance
   comparable to bw24's. No new information beyond confirming spec is the
   dominant decode lever; bw24's implementation is still ahead of vLLM's.
4. **SGLang**: 9B bf16 ran (fastest prefill measured: 20.0k tok/s, decode 75-80).
   27B NVFP4: **blocked** — the text-only modelopt export
   (`vision_config: null` under `Qwen3_5ForConditionalGeneration`) crashes
   SGLang's loader (`Qwen3_5VisionConfig has no attribute hidden_size`), and
   `--language-only` requires a disaggregated encoder endpoint. vLLM tolerates
   the same checkpoint fine.

## Adoption candidates for the next SOTA rev, ranked

| candidate | evidence | size of prize |
|---|---|---|
| Chunked-prefill scheduling + FP4-GEMM prefill path | vLLM 27B 10.8-13.1k vs bw24 1.9k on same GPU/model/prompt | 5.6-6.8x prefill, TTFT 3.25s -> 0.6s |
| Rig-adaptive decode tuning (G7e regression) | llama 27B spec 130-159 vs bw24 102-128 on G7e; reversed on 5090 | 1.25-1.3x decode on big-BW GPUs |
| FlashInfer prefill/GEMM kernels | untested — flashinfer paths (vLLM auto-default) require JIT-on-first-use; blocked by no-JIT rule this run | unknown, likely ≥ cutlass path |
| CUDA-graph decode | +3% at bs=1 (GDN hybrid limits capture to decode-only) | negligible for bs=1 |

## Results — G7e box (RTX PRO 6000 Blackwell Server 96GB, sm_120, CUDA 13.2)

All four engines, same hardware, same model files or same-quant checkpoints.
p1 = 28 tok prompt, p2 = 1845, p3 = 6257. gen = 256 tok, temp 0, ignore_eos.

### 27B (Qwen3.6-27B) — 4-bit weights everywhere

| engine + config | p1 prefill / gen (tok/s) | p2 prefill / gen | p3 prefill / gen |
|---|---|---|---|
| bw24 spec K=3 (NVFP4+Q4_K_M GGUF, frspec trim, pmin .2) | 126 / **127.7** | 1715 / **118.9** | 1924 / **102.1** |
| llama.cpp spec-serve (same GGUFs, MTP draft n-max 3, f16 KV)¹ | 237 / **158.7** | 3781 / **154.9** | 4401 / **130.5**² |
| vLLM 0.24 modelopt NVFP4, cutlass linear, no spec³ | 255 / 39.5 | **13118** / 39.5 | **10823** / 41.5 |
| vLLM marlin linear (W4A16 dequant), no spec | 311 / 54.9 | 5539 / 55.4 | 5047 / 54.7 |
| vLLM cutlass + qwen3_5_mtp spec k=3⁴ | 276 / 93.0 | 12104 / 92.4 | 9836 / 87.9 |
| vLLM marlin + qwen3_5_mtp spec k=3⁴ | ~311 / **121.2** | ~5539 / 94.2 | ~5047 / 91.9 |
| SGLang 0.5.9 | blocked (vision-config loader crash on text-only export) | — | — |

TTFT (s), 27B: bw24 .215/1.08/3.25 · llama .118/.49/1.42 · vLLM-cutlass .110/.141/.578.

¹ The serve-script q8_0/q5_1 KV config is catastrophically broken on this build/box:
  p2 prefill 45 tok/s, p3 decode 4.3 tok/s (asymmetric-KV FA kernels absent —
  GGML_CUDA_FA_ALL_QUANTS not set; known 18x cliff). f16 KV used instead.
² p3 hits immediate EOS at temp 0 on llama; measured with ignore_eos.
³ vLLM no-JIT constraint: compilation mode 0 (no inductor), FULL_DECODE_ONLY CUDA
  graphs [1,2,4], flashinfer autotune off, flashinfer sampler off (its default path
  JIT-compiles via ninja; also community-reported 8.6x regression), linear backend
  forced to precompiled cutlass/marlin (auto picks FlashInferCutlassNvFp4 = JIT).
  fp8 KV + MTP: crashed (fp8-KV attention path also wants ninja JIT) — not measured.
⁴ Spec rows: decode measured by delta method ((256-1 tok)/(t_256 - t_1tok), warm)
  because SSE chunks carry multiple tokens under spec. Prefill from streaming TTFT
  of the non-spec run with the same backend (spec doesn't change prefill path).

### 9B (Qwen3.5-9B) — QUANT MISMATCH: vLLM/SGLang are bf16 (no NVFP4 9B HF on disk); bw24/llama are 4-bit GGUF. bf16 = 4x weight-read handicap at decode; treat decode columns accordingly.

| engine + config | p1 prefill / gen | p2 prefill / gen | p3 prefill / gen |
|---|---|---|---|
| bw24 spec K=2 (NVFP4 GGUF, pmin .3) | 338 / **234.6** | 5271 / **191.5** | 5886 / **173.1** |
| bw24 plain (same GGUF) | — / 137.4 | — / 126.0 | — / 125.9 |
| llama.cpp plain (same GGUF, f16 KV) | 737 / 185.7 | 11104 / 185.2 | 12446 / 182.1 |
| vLLM bf16 (CUDA graphs decode-only) | 615 / 74.6 | 18972 / 74.4 | 18037 / 74.8 |
| SGLang bf16 (triton attn — required for hybrid-GDN on Blackwell) | 953 / 79.8 | 18829 / 78.2 | **20027** / 74.9 |

TTFT (s), 9B: bw24 .080/.350/1.063 · llama .038/.166/.503 · vLLM .046/.097/.347 · SGLang .029/.098/.312.

## Results — local rig (RTX 5090 Laptop 24GB, sm_120, CUDA 13.1) — partial

vLLM measured before the study relocated to G7e (GPU-politeness: k-quant agent
owns the local GPU). bw24/llama numbers are the standing reference (free clocks).

| engine + config | p1 prefill / gen | p2 prefill / gen | p3 prefill / gen |
|---|---|---|---|
| bw24 9B spec (reference) | — / 180.5 | 2527 / 148.8 | 2674 / 124.1 |
| llama 9B plain (reference) | — / 123 | — / 122 | up to 5996 / 119 |
| vLLM 9B bf16 (eager; CUDA graphs = OOM at 24GB)⁵ | 389 / 40.7 | 4363 / 40.3 | 4378 / 40.0 |
| bw24 27B spec (reference) | — / 98.7 | 759 / 86.7 | 787 / 66.4 |
| llama 27B spec-serve (reference) | — / 87.7 | 2114 / 93.4 | — / 76.8 |
| vLLM 27B NVFP4 cutlass + decode CUDA graphs | 378 / 27.5 | 4929 / 27.3 | 4692 / 26.9 |
| vLLM 27B + MTP spec | OOM (0.85 and 0.95 util) | — | — |
| SGLang local | not run (relocated before install) | — | — |

⁵ vLLM's memory model doesn't fit 19GB checkpoints + graphs + KV on 24GB.
Local vLLM decode is crippled (27 tok/s on 27B) but its prefill still beats
bw24's local prefill ~6x — the prefill conclusion holds on both boxes.

## Caveats / provenance

- vLLM 0.24.0 (G7e venv ~/venvs/vllm-bench, cu130 torch 2.11) and
  0.22.1rc1.dev481 (local uv tool install). SGLang 0.5.9 + transformers 5.13
  (G7e venv ~/venvs/sglang-bench), sgl_kernel AOT + flashinfer_cubin prebuilt.
- G7e llama.cpp master 94875285e rebuilt with CMAKE_CUDA_ARCHITECTURES=120 (stock
  build lacked sm_120 → "no kernel image"). bw24 at ea9fcb4 (A2 expert-grouped prefill).
- 27B HF checkpoint: sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP (modelopt NVFP4,
  group 16, MTP head included; local copy qwen36-27b-text-nvfp4-mtp-hf +
  preprocessor configs copied from the 9B to satisfy the multimodal processor).
- Client-side TTFT/streaming measured over localhost HTTP; prefill tok/s =
  prompt_tokens / TTFT (includes fixed server overhead — p1's 28-token cells are
  overhead-dominated, compare p2/p3 for real prefill rates).
- bw24 prefill derived from run-spec "prime" times; bw24 has no server loop, so
  its cells exclude HTTP overhead (favors bw24 slightly on TTFT).
- No JIT/source builds were used for any engine (prebuilt wheels/cubins only);
  flashinfer JIT paths in vLLM (default linear backend, sampler, fp8-KV+MTP)
  are therefore unmeasured and could raise vLLM's numbers.
- Community SM120 context: vLLM forum report (Qwen3.5-397B MoE, 4x RTX PRO 6000)
  found cutlass grouped-GEMM MoE broken/garbage on SM120 and marlin fastest; our
  dense-27B run shows cutlass linear GEMM correct (sanity-checked output) but
  m=1-slow, matching the same underlying kernel-maturity story.

Raw data: `engine-comparison-2026-07.jsonl` (one record per cell).

## IMAGE 5 — 2026-07-05 (post base-path day: chunked prime, W4A8 default, deep-ctx FA pair, session KV, dual fuse)

Same exact prompts, N=3 medians, bw24 K=3 pmin0.15+frspec vs llama serve config (n-max 3, p-min 0.1, KV q8/q5, fa, graphs).

| cell | bw24 gen (spec) | llama gen (spec) | bw24/llama | bw24 pp | llama pp |
|---|---|---|---|---|---|
| 27B p1 (28 tok) | **95.8** | 87.2 | **1.10x** | — | 163 |
| 27B p2 (1845) | 87.3 | 92.4 | 0.94x | 1267 | 1958 |
| 27B p3 (6257) | **73.9** | 75.8 | 0.97x | 1259 | 2113 |
| 27B 32k (40488) | 33.9 | (untested llama ctx 16k script cap) | — | ~1030 | — |

Movement since image 4: p3 0.87x -> 0.97x (gap 12.9 -> 1.9 tok/s), p1 now WINS +10%. pp gap narrowed 2.6x -> 1.6x (W4A8-MMQ default + ARC B).
bw24-only capabilities now: session KV reuse (turn-start 42.6x at 40k history; llama has prompt-cache but not cross-request continuation in our serve config), exactness self-consistency contract, 278k-token 9B on 24GB.
llama still ahead: pp throughput 1.6x (their stream-K + fixup tail), p2/p3 gen by 1-5%.
OPEN to flip p2/p3: ARC A aligned-KV probe (in flight G7e), k-quant b4 tranche, draft-head cost.
