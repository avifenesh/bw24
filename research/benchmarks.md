# Benchmarks — bw24 vs llama.cpp on RTX 5090 Laptop (the beat-targets)

Goal (user): beat **vLLM + SGLang + llama.cpp** on **prefill, decode, AND overall**.
Box: RTX 5090 Laptop sm_120, gpu-full-power on. Model: Qwen3.5-9B Q8_0 (8.86 GiB, hybrid arch).

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
