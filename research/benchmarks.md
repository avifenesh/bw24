# Benchmarks — bw24 vs llama.cpp on RTX 5090 Laptop (the beat-targets)

Goal (user): beat **vLLM + SGLang + llama.cpp** on **prefill, decode, AND overall**.
Box: RTX 5090 Laptop sm_120, gpu-full-power on. Model: Qwen3.5-9B Q8_0 (8.86 GiB, hybrid arch).

## CURRENT STATE (2026-07-03, 9B-NVFP4, INTERLEAVED A/B protocol — the only valid ratio protocol)

| metric | bw24 | llama.cpp | ratio |
|---|---|---|---|
| pp512 | 5049 | 5072 | **0.995x (parity band)** |
| tg128 graph @ctx128 | **110.5** | 106.3 | **1.04x — ABOVE** |
| MTP spec K=1..4 | exact (PASS) | — | 0.95x of plain @K=2-4 (needs draft-cost cut to profit) |

Protocol rule (measured 2026-07-03): sequential cross-session numbers LIE by up to 10% — llama
holds higher clocks when run alone/cold. Interleave the two engines in the same minute, N>=3 pairs,
both orders. Prior "llama 117.8 / 5451" baselines in this file are sequential-protocol numbers.

## Measurement protocol — window rules (2026-07-16, all lanes)

The rules below are the standing A/B protocol referenced by CONTRIBUTING.md. Each one was
paid for by a real false reading (the jsonl rows carry the archaeology):

1. **Interleave, same session, N≥2-3 medians, both orders.** Sequential cross-session ratios
   drift up to ~10% from clock/thermal state (rule above). A same-session-only number is not
   evidence.
2. **Pin the power state per window** (`gpu-full-power on|off`) and record it. Numbers from
   different power states never pair; a profile-run arm only compares against another
   profile-run arm.
3. **Serialize bench arms — one engine on the GPU at a time.** A co-resident llama-server
   (or any VRAM-heavy process) silently forces expert/weight spill to host and reads 10x low
   (measured: 26B/31B "collapse" to 17-21 tok/s that vanished once the server was stopped).
   `tools/local-ci.sh` refuses to run a battery in a dirty window for exactly this reason.
   The only allowed co-resident is a small embedding server.
4. **llama.cpp bars = llama's BEST config, per model** (owner ruling). For gemma that means
   `-fa 1` explicitly — llama-bench's `-fa auto` resolves OFF for gemma and produces
   fake-low bars. KV-quant flags LOSE for gemma-vs-f16 on this box; best config is `-fa 1`
   with f16 KV. Verify the resolved config in llama's own output, don't trust the flag.
5. **Validity-gate the window with a known cell.** After marathon benching the EC pulls boost
   (chassis skin-temp, not die) — re-run a known-good cell and check it lands on its rolling
   median before trusting any new number from that window.
6. **The standing cell battery is `tools/local-ci.sh --perf`** (cells in
   `research/tune-data/perf-cells.json`, rows in `research/tune-data/perf-ci.jsonl`).
   It records tok/s AND speculative acceptance/tokens-per-round per spec cell — acceptance
   drift is invisible to every exactness gate and must be tracked longitudinally.

## Baselines (measured 2026-06-26)

| Engine | prefill pp64 (tok/s) | decode tg32 (tok/s) | tool |
|---|---|---|---|
| **llama.cpp** | **2849 ± 271** | **81.8 ± 1.1** | llama-bench |
| bw24 Stage-A (f32 dequant) | (not measured, slow) | 26.1 | run-gen |
| bw24 Stage-B (int8 dp4a Q8_0 only) | — | 38.1 | run-gen |
| vLLM | TODO | TODO | (sm_120 maturity; python per-token overhead = our edge) |
| SGLang | TODO | TODO | TODO |

## Gap analysis (decode, the daily hot path)

- 847 GB/s ÷ 8.86 GB ≈ **95.6 tok/s** hardware ceiling (read weights once/token).
- llama.cpp 81.8 = **86% of ceiling** (mature MMQ/MMVQ).
- bw24 Stage-B 38.1 = **40% of ceiling**, **2.1x slower than llama.cpp**.

### Why bw24 is behind (the work to win):
1. **Only Q8_0 GEMMs use int8 dp4a.** Linear-attn projections (wqkv/ssm_*) + any non-Q8_0 still hit
   Stage-A f32 dequant (3.6x slower). → extend fast path to Q4_K/Q6_K/NVFP4.
2. **Host KV re-upload every step.** decode.rs round-trips K/V through host f32 each token →
   massive overhead. → keep KV resident on GPU (Stage-2 cache), fp16.
3. **MMVQ kernel = 1 block/output, 64 threads.** Under-occupied vs llama.cpp's tuned mmvq
   (ncols batching, warp-per-row, vectorized loads). → tune block/grid + vectorize.
4. **Per-op kernel launches, no CUDA graph.** 32 layers × ~10 kernels × per-token launch overhead.
   → CUDA-graph the decode step (researched in ARCHITECTURE.md §3.10).
5. **GDN/conv state round-trips host each step** (decode.rs dtoh/htod). → keep state resident.

### Edge vs vLLM/SGLang (to measure + exploit)
Native Rust runtime = no python per-token dispatch, no GC. On sm_120 single-stream decode, vLLM/SGLang
pay structural per-step overhead + immature sm_120 kernels. Our win path there is lower per-token CPU
overhead + the hybrid-arch KV advantage (only 8/32 layers grow KV). Must measure vLLM/SGLang on this box.

## Beat-target milestones
- [ ] decode: bw24 > 81.8 tok/s (beat llama.cpp) — needs items 1-5 above
- [ ] prefill: bw24 > 2849 tok/s — needs Stage-B MMQ prefill (int8 tiles) + batched
- [ ] overall throughput: continuous batching
- [ ] all of the above vs vLLM + SGLang too

## Update — fast GEMM all-dtypes + resident state (post-restart)
- 9B Q8_0 decode FAST (Q8_0 int8 dp4a + resident SSM state) = **52.9 tok/s** (decode==prefill exact).
- Q4_K + Q6_K int8 dp4a MMVQ landed + validated (rel 3e-3 vs oracle) -> 27B-Q4_K_M now fast too.
- KNOWN GAP: NVFP4 dequant not implemented (dequant.rs) -> 27B-NVFP4 file panics; needs NVFP4 CPU
  dequant + decode dot (fast-GEMM workflow deferred it, spec-only). Blocks NVFP4 daily models.
- Remaining decode levers: extend MMVQ occupancy, CUDA graph (320 launches/token), hand-FA (in v2 fix).

## NVFP4 global-scale gap (found 2026-06-26, blocks daily 9B/27B-NVFP4)
9B-NVFP4 forward = argmax 1543, llama.cpp = 268 (WRONG). Root cause: NVFP4 weights carry EXTRA
per-tensor F32 globals `<w>.scale` (e.g. ffn_gate.scale=1.01e-4) + `<w>.input_scale` (=0.143) that
multiply the block-scaled (ue4m3 per-16) dequant. bw24 ignores them -> every NVFP4 matmul off by a
global factor. NOT a dequant bug (block dequant matches ggml; ggml's to_float ALSO ignores globals —
this is vLLM-style compressed-tensors NVFP4 where scale applies in scaled-mm). FIX: load <w>.scale +
<w>.input_scale, apply y = (nvfp4_blockdequant(W) @ x) * w_scale  (input_scale for the W4A4 act path).
The isolated dequant validation had the SAME blind spot as ggml -> gate passed but forward wrong.
Q8_0/Q4_K/Q6_K/Q5_K/IQ models have NO global scale -> unaffected (9B-Q8_0 argmax 268 correct, 35B-MoE
1178 correct). Only NVFP4 tensors need the global-scale fix.

## Competitor arch-support reality (2026-06-26, VALIDATED by reading installed code)
- vLLM + SGLang INSTALLED (py3.12 venvs, /data/projects/bench-engines), torch 2.11+cu130 sees sm_120 (cap 12,0).
- BOTH SUPPORT qwen35 hybrid (corrected — I had wrongly guessed "not supported" from the arch name):
  * vLLM 0.23.0 registry.py:569 -> Qwen3_5ForConditionalGeneration + Qwen3_5MoeForConditionalGeneration (qwen3_5.py, shares Qwen3Next gated-deltanet).
  * SGLang 0.5.9 -> srt/models/qwen3_5.py + qwen3_5_mtp.py, Qwen3_5Config/Qwen3_5MoeConfig (extends Qwen3NextConfig + mamba2 utils).
- So the FULL 3-engine beat-target IS live on the DAILY models (9B/27B hybrid + 35B MoE). Must bench all 3.
- LESSON: validate by reading the actual registry, not by inferring from the config arch string.

## bw24 decode progression (9B Q8_0, gpu-full-power)
- Stage-A f32 dequant: 26 tok/s
- Stage-B int8 dp4a Q8_0: 38
- + resident GPU SSM state: 56
- + GPU repack (no host scatter): 59.6
- Remaining levers to 81.8+: CUDA graph (320 launches/tok -> 1 replay) = biggest; MMVQ occupancy; KV-quant bandwidth.
