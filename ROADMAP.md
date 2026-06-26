# bw24 Build Roadmap — full scope, no shortcuts (week+ project)

Working method (per user): **A. research every component via workflow. B. implement via workflow** —
main thread orchestrates + reviews; agents write code. Don't inline-code everything (saves context).
Every "naive/Stage-A" placeholder is DEBT to be paid in full, not deferred. Hand-write the FA3/FA4
algorithmic improvements adapted to sm_120 (FA3/FA4 binaries don't run here: no wgmma/tcgen05).

## DONE (validated vs llama.cpp ground truth)
- GGUF v3 parser (all quant + NVFP4/IQ), ModelConfig arch-dispatch, CPU dequant oracle
- All Stage-1 kernels validated (rmsnorm/l2/rope/sdpa/conv1d/gated-deltanet/gated-rmsnorm)
- Dense (qwen3) + HYBRID (qwen35) forward: argmax-MATCH llama.cpp
- Resident-quant GEMM (no OOM); KV-cache decode + generation (decode==prefill exact)
- Stage-B int8 dp4a Q8_0 + resident GPU state: **decode 56 tok/s** (llama.cpp 81.8; gap 1.45x)

## REMAINING SCOPE (full, no shortcuts)

### Kernels (isolated .cu files — parallelizable across agents)
1. **Fast GEMM all dtypes**: MMVQ decode + MMQ prefill (int8 dp4a/mma) for Q4_K, Q6_K, NVFP4.
   Currently only Q8_0 is fast; everything else is Stage-A f32 (3.6x slow). Validate vs CPU oracle.
2. **NVFP4 native block-scale GEMM** (the 762-TFLOP path) — for the NVFP4 GGUFs (both daily models have them).
3. **Hand-written FlashAttention** for sm_120:
   - FA-2 base: mma.sync m16n8k16, online softmax, KV tiling (replace naive sdpa).
   - FA-3 improvements BY HAND: warp-specialization (producer/consumer), async cp.async pipeline,
     2-stage softmax/rescale overlap, ping-pong buffers — adapted to sm_120 mma.sync (NOT wgmma).
   - FA-4 improvements BY HAND: the newer softmax/exp schedule + tile refinements from the FA4 beta.
   - head_dim 256 (qwen35), GQA, partial RoPE already applied upstream, causal. Prefill + decode (fattn-vec style).
4. **CUDA graph capture** for the decode step (320 launches/token → 1 graph replay).
5. **KV quant kernels**: q8_0 K / q5_1 V (matches the daily serve script), asymmetric, fused into attention.

### Rust runtime features
6. **MoE forward** (qwen35moe, Qwen3.6-35B-A3B): router softmax top-8/256 + grouped expert GEMM
   (Marlin-style or MMQ grouped) + sigmoid-gated shared expert. Reuse hybrid mixers.
7. **Safetensors loader** (SAFETENSORS-DECISION.md): HF bf16/fp16, name map, transforms (A_log negate,
   norm +1, dim-reverse), config.json→ModelConfig, sharding. Add GpuTensor::Bf16 resident variant.
8. **Spilling**: tiered VRAM↔pinned-host↔mmap-disk for MoE experts + weights that don't fit; SLRU
   hot-expert cache; prefetch. (qwen36-35B Q6_K = 31GB spills.)
9. **Spec decode**: MTP (built-in NextN head — daily serve script uses --spec-type draft-mtp) + EAGLE3.1
   (drafts on disk). draft-n-max 3, p-min 0.2 like the serve script.

### Benchmark (the headline)
10. **Beat vLLM + SGLang + llama.cpp** on prefill + decode + overall, each at its BEST-tuned setup
    (copy user's serve scripts for flags: -fa on, KV q8_0/q5_1, MTP spec, CUDA graphs, full power, ctx 64k).
    N=5 medians, gpu-full-power on. research/benchmarks.md tracks it.

## Reference docs (all researched)
ARCHITECTURE.md, PHASE1-HYBRID.md, QUANT-GEMM-DECISION.md, SAFETENSORS-DECISION.md,
research/{sm120-empirical-capabilities, benchmarks, current-model-targets, claim-verification-report}.md

## Orchestration pattern for implementation workflows
- Agents write SELF-CONTAINED new .cu files + a documented launcher signature + a CPU/llama reference
  for the validation gate. Avoid parallel edits to shared lib.rs (main thread integrates launchers).
- Each kernel agent returns: the .cu source, the launcher signature, and the exact validation (vs
  dequant.rs CPU oracle or vs llama.cpp). Main thread integrates + runs the gate + reviews.
- Worktree isolation for agents that must touch shared files concurrently.
