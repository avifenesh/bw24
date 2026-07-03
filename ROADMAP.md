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

## UPDATE 2 (post-MoE)
- HYBRID FA shipped (FA-2 + FA-3/FA-4 improvements scoped to sm_120, validated 11/11 vs oracle, argmax 268).
- **MoE + EDGE-1 VALIDATED**: 35B-A3B argmax=1178 == llama.cpp; bw24 fits 24GB (~4GB peak) where
  llama.cpp full-offload OOMs (30.5GB). Selective per-token expert staging proven.
- Q4_K/Q6_K fast int8 dp4a landed. ALL daily targets run correct: 9B/27B hybrid + 35B MoE.
- **DISCIPLINE DEBT (must fix): new dtypes Q5_K/Q3_K/IQ4_XS/IQ3_S/NVFP4 added to dequant.rs+qmatvec.cu
  WITHOUT a CPU-oracle validation gate.** Daily NVFP4 (9B 3.45GB + 27B) + IQ4_XS/IQ3_S (35B 14.5GB)
  models DEPEND on these being correct. Validate each vs an independent reference (llama.cpp dequant
  or hand-derived) before trusting those daily models. THIS IS THE NEXT WORKFLOW.

## REMAINING SCOPE (full, no shortcuts)

### Kernels (isolated .cu files — parallelizable across agents)
1. **Fast GEMM ALL dtypes** — every quant the daily models actually use, no gaps:
   DONE: Q8_0, Q4_K, Q6_K. MISSING (verified in daily GGUFs): **Q5_K** (9B-NVFP4 44 tensors),
   **NVFP4** (9B 3.4GB + 27B 9.6GB — BIGGEST share, currently PANICS no dequant at all),
   **IQ3_S** (35B-MoE 9GB — biggest), **IQ4_XS** (35B 5.6GB), **Q3_K**. Also add CPU dequant-oracle
   for each (dequant.rs) so the validation gate works. Without NVFP4+IQ, the 27B-NVFP4 and 35B-MoE
   daily models cannot run at all. MMVQ decode + MMQ prefill for each.
2. **NVFP4 native block-scale GEMM** (762-TFLOP path) — biggest tensor share of both daily models.

### DECODE host-round-trip removal (HIGH PRIORITY — GPU is free, CPU work is stage-debt not by design)
0b. decode.rs linear_attn_decode/full_attn_decode still do per-step dtoh→host-scalar-loop→htod for:
    conv-ring assembly, q/k/v GDN repack (4096-wide per-element host loop!), q|gate split. These are
    pure overhead (GPU idle during host scatter). Move ALL to GPU kernels: a conv-state-assemble
    kernel, a qkv→GDN-layout repack kernel, a q|gate split kernel. Only ssm_state was made resident;
    finish the rest. This is likely the dominant decode cost now, bigger than the GEMM gap.
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

### Web-sweep ranked techniques (folded 2026-07-03 from the web-sweep agent output)
11. **DFlash-style spec decode integration** — vLLM's scheduler reserves `num_speculative_tokens+1`
    lookahead slots for DFlash drafts (vllm.md:60). For bw24: the MTP verify band already exists;
    the DFlash lever is block-diffusion drafting (parallel draft of K tokens in one pass) — evaluate
    AFTER MTP re-measure (run-spec) since acceptance-rate lift is the profitable axis (DECODE-GAP L6).
12. **TCQ KV quant** (per-token/channel quant): our KV is q8_0/q5_1 per-token already; TCQ's win is
    per-CHANNEL K scales at low bit. Only worth it below q8 K — pairs with KVQUANT-PLAN.md, not before.
13. **FR-Spec vocab trim** — draft head scores only the top-frequency vocab slice (~25% lm_head cost
    in draft). bw24 hook: striped-vocab MTP head (mtp-tail work) + q6_K lm_head is 1.07ms/tok — a
    frequency-sliced draft head skips most of it per draft token. MED value, needs MTP live first.
14. **NVFP4 tensor-split fix** (llama PR) — multi-GPU tensor-split only; DEAD for the single-GPU
    local rig, note for L40S/fleet mirrors if they ever go multi-card.
15. **ST-MoE prefetch** — layer-wise async expert prefetch on a dedicated stream (lmcache.md:99-100
    pattern + ST-MOE-PLAN.md). The lmcache async load/store stream template is the substrate; wire
    into moe_cache.rs SLRU when MoE spilling becomes the active target.

## Reference docs (all researched)
ARCHITECTURE.md, PHASE1-HYBRID.md, QUANT-GEMM-DECISION.md, SAFETENSORS-DECISION.md,
research/{sm120-empirical-capabilities, benchmarks, current-model-targets, claim-verification-report}.md

## Orchestration pattern for implementation workflows
- Agents write SELF-CONTAINED new .cu files + a documented launcher signature + a CPU/llama reference
  for the validation gate. Avoid parallel edits to shared lib.rs (main thread integrates launchers).
- Each kernel agent returns: the .cu source, the launcher signature, and the exact validation (vs
  dequant.rs CPU oracle or vs llama.cpp). Main thread integrates + runs the gate + reviews.
- Worktree isolation for agents that must touch shared files concurrently.
