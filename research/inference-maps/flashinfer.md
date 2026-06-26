# FlashInfer — Implementation Map for a single-stream sm_120 GGUF engine (bw24)

FlashInfer is the shared attention/GEMM kernel library underneath vLLM and SGLang. It
bundles two kernel families: (1) its own hand-written FlashAttention CUDA (`decode.cuh`,
`prefill.cuh`, paged scheduler) and (2) the **TensorRT-LLM `nv_internal` kernels** vendored
into `data/csrc/nv_internal/` (XQA, FMHA v2, Blackwell CUTLASS 3.x FMHA, deep_gemm, fused
MoE). The package ships **explicit `sm120` codegen** for the FP4/FP8/MXFP8 GEMMs and TRT-LLM
FMHA v2, plus XQA which checks for SM120 at runtime. This makes it the single best donor of
**portable attention math** for an sm_120 (RTX 5090, Blackwell consumer) engine.

Target machine for this map: **batch=1, single-stream, GGUF weights, sm_120 (CC 12.0)**.
The decisive arch facts (verified in `data/include/flashinfer/mma.cuh:30-49`):
- `m16n8k16` / `m16n8k8` FP16/BF16 `mma.sync` — enabled `>=750`/`>=800`. RUNS on sm_120.
- `m16n8k32` FP8 (e4m3/e5m2) `mma.sync` — enabled `>=890` AND CUDA `>=12.4`. RUNS on sm_120.
- `stmatrix m8n8x4` — gated `>=900` only. NOT on sm_120.
- `wgmma` (SM90) and `tcgen05` (SM100) cluster-wide MMA — NOT on sm_120.

So: **warp-level `mma.sync` + `cp.async`/TMA = portable. wgmma/tcgen05 = dead.**

---

## What bw24 could take from flashinfer (3-8 highest-value, most portable)

1. **On-the-fly (streaming) softmax decode loop** — `decode.cuh:217` `SingleDecodeWithKVCacheKernel`.
   A single query scans the KV cache in tiles, maintaining warp-local `(m=max, d=expsum, o=acc)`
   and rescaling on each tile. No matmul library, no paging needed. This is the canonical
   batch=1 decode kernel and maps 1:1 to a GGUF single-stream engine. Pair with the
   `cp_async` double-buffered K/V tile pipeline (`decode.cuh:308-320`). **Highest value, lowest risk.**

2. **FP8 KV-cache inline-dequant in the decode loop** — `mma.cuh:216-276`
   `mma_sync_m16n16k32_row_col_f8f8f32`. Q stays BF16/FP16; K,V stored e4m3/e5m2 with per-block
   FP32 scales, dequantized inside the QK / V-accumulate loop. bw24 already ships q8_0/q5_1 KV
   (commit 9ebf958) — FlashInfer's `m16n8k32` FP8 path is the tensor-core upgrade: ~1.5-2x on the
   KV-BW-bound decode. Native on sm_120 (guarded `>=890 && CUDA>=12.4`).

3. **NVFP4 (4-bit) KV cache + per-16-elt block scales** — `fp4_kv_quantization.cu` /
   `fp4_kv_dequantization.cu`. 2x further KV shrink vs FP8 (4 bits + FP8 block scale, packed
   2/byte). Dequant via DP4A/FMA at load, available on every modern arch. For a 24 GB card
   running large GGUF models this is the difference between fitting a long context or not. XQA
   gates NVFP4 KV explicitly to SM120 (`xqa.py:232`).

4. **Split-K decode for long context** — `scheduler.cuh:148-210`
   (`BatchDecodeWithPagedKVCacheWorkEstimationDispatched`). Uses
   `cudaOccupancyMaxActiveBlocksPerMultiprocessor` to decide when to chop one long sequence into
   KV chunks, each emitting a partial `(o, lse)`, then a merge pass. **This is the one
   batch>1-flavored idea that still helps batch=1**: at decode with seq_len > ~4-8K, a single CTA
   leaves the GPU idle; split-K turns the long scan into parallel chunks + a cheap LSE merge. Port
   only the occupancy-driven split decision + the merge math, drop the paging.

5. **LSE merge / streaming-softmax combine primitive** — the `(o_partial, lse_partial)` merge used
   by both decode split-K and prefill split-K (`cascade.cuh`, `fmhaReduction`). A standalone
   `merge_state(o_a, lse_a, o_b, lse_b)` is reusable for: split-K, chunked prefill, and sliding-
   window/sink attention. Pure warp-shuffle max/sum + FMA — trivially portable.

6. **Skip-softmax threshold + logits soft-cap + attention-sink accumulation** —
   `prefill.py:240-296`, `trtllm_fmha_kernel_launcher.cu:174-182`. Cheap scalar guards
   (tanh soft-cap pre-softmax; skip a KV block whose residual mass < threshold; prepend sink
   tokens directly into the accumulator). All branchy scalar ops, no special instruction. Drop-in
   correctness/perf features for any attention loop.

7. **GQA broadcast pattern from XQA** — `xqa.py:154-179`. For LLaMA-style GQA (32 Q heads / 8 KV
   heads) load each KV pair once per warp and broadcast to the 4 Q heads sharing it, instead of
   re-reading KV per Q head. ~1.3x on the KV-BW-bound decode. The *idea* ports cleanly even if
   FlashInfer's TMA-based XQA binary does not — implement the broadcast in the `decode.cuh` loop.

8. **(Optional) raw `m16n8k32` FP8 GEMM tiles for dequantized weights** — `mma.cuh:223-299`. If
   bw24 ever wants tensor-core FP8 matmul for projections without CUTLASS, these are the raw PTX
   wrappers that compile on sm_120. Manual tiling required (no collective), but no wgmma needed.

---

## DEAD for bw24 (do not port)

| Kernel / machinery | Why dead on single-stream sm_120 | source |
|---|---|---|
| Blackwell SM100/SM103 CUTLASS 3.x FMHA (warp-specialized, cluster MMA) | Uses **tcgen05** cluster-wide MMA + cluster launch; not present on sm_120. Separate code path, never auto-selected for sm_120. | `data/include/flashinfer/attention/blackwell/device/fmha.hpp:49-150`, `collective/sm100_fmha_gen_mainloop_warpspecialized.hpp` |
| Block-scaled FP8 dense GEMM (`fp8_blockscale_gemm`) | SM90-only; depends on **wgmma** + `sm90_wgmma`. Would need full BlockScaledTensorOp re-port to `mma.sync` + block-scale epilogue. | `jit/gemm/fp8_blockscale.py:10-35`; `nv_internal/.../fp8_blockscale_gemm/` |
| `stmatrix m8n8x4` store path | Gated `__CUDA_ARCH__ >= 900` only. | `mma.cuh:37-39` |
| Multi-request paged-KV server machinery (block tables, indptr, last-page-len, `block_valid_mask` for CUDA-graph determinism across requests) | All exists to batch many concurrent requests; batch=1 has one contiguous KV. Keep only the split-K math (idea #4), drop paging. | `batch_decode.cu:39-200`; `scheduler.cuh` |
| Gated Delta Rule / linear-attention decode (Mamba/SSM) | State-space, not softmax attention; no sm_120 codegen confirmed (relies on `gdn_kernels.gdn_decode_*` imports). Irrelevant to a softmax GGUF decoder. | `gdn_decode.py:56-106`; `gdn_kernels/` |
| Fused MoE grouped GEMM (CUTLASS `TmaWarpSpecializedGroupedGemmInput`) | SM90 collective; SM120 variants declared in codegen but kernel bodies not in package. Multi-expert routing is orthogonal to single-stream dense decode. | `jit/gemm/cutlass/generate_kernels.py:240-297`; `fused_moe/cutlass_backend/cutlass_fused_moe_instantiation.cu:20-69` |
| AMX paths | CPU-only; no GPU relevance. | n/a |

---

## Subsystem 1 — Decode attention

| Technique | How implemented (mechanism) | Kernel/layout/instruction | source file:line | sm120_fit |
|---|---|---|---|---|
| Single-request decode (non-paged KV) | One query scans full KV in tiles; per-tile QK with RoPE; on-the-fly softmax (`m`,`d`,`o` rescale); window_left mask, logits_soft_cap, alibi | KV NHD `[seq,kvH,D]` or HND; thread-local `vec_t<float,vec_size>`; `cp_async` double-buffered K/V smem; warp-local state | `decode.cuh:217` (kernel), `:308-320` (qk + cp_async load), `:214-368` | **RUNS.** Warp `m16n8k16/k32` + `cp.async`, no wgmma/tcgen05. The canonical batch=1 decode; port directly. Linear scan = latency ~O(seq_len). |
| Batch decode w/ paged KV (split-K) | Scheduler partitions KV on seq-len; if occupancy low → split_kv, each chunk a pseudo-request emitting partials, merge pass; `block_valid_mask` for CUDA-graph | Paged `[max_pages,page_sz,kvH,D]`; indptr `[B+1]`; last-page-len `[B]`; occupancy via `cudaOccupancyMaxActiveBlocksPerMultiprocessor`; block-per-chunk×kvH | `batch_decode.cu:39-81` (plan), `:83-200` (run); `scheduler.cuh:148-210` (split-K) | **RUNS.** Same warp mma; split reduction = atomics + FP32 workspace; std CUDA occupancy API. **Helps batch=1 only when seq_len long** (chop the scan). Port split decision + merge; drop paging. |
| FP8 (E4M3/E5M2) KV decode | Q stays FP16/BF16; K,V e4m3/e5m2 + per-block FP32 scale; inline dequant in QK & V-load; TC matmul | KV uint8/float8 in 16B granule; FP32 block scale (16/32 elts); `mma.sync m16n8k32` (2 accum, 4×A 128b, 2×B 128b) | `mma.cuh:216-276` (`mma_sync_m16n16k32_row_col_f8f8f32`); guard `mma.cuh:30-34`; `fp8_gemm_cutlass.cu` | **RUNS** (gated `>=890` & CUDA `>=12.4`, both true on sm_120). Confirmed sm120 modules exist (`group_gemm_fp8_groupwise_sm120.cu`, `mxfp8_gemm_cutlass_sm120.cu`). ~1.5-2x KV-BW-bound decode at batch=1. |
| XQA (cross/grouped-query) | Large q:kv head ratio; one KV pair broadcast to many Q heads per warp → kills KV re-reads; page-table + seq_len indirection; optional sinks, spec-decode (q_seq_len>1), NVFP4 KV | Q `[B,beam,qS,qH,D]`; KV pages NHD; page table `[B,maxBlk]`; semaphores for CTA work-dist; TMA load + warp mma | `xqa.py:154-179` (API); `xqa.py:240` (`only supported on SM90/100/120/121`) | **RUNS** (sm120 explicitly listed `xqa.py:240`; NVFP4 KV sm120-gated `:232`). Uses TMA (Blackwell). ~1.3x on GQA at batch=1. Port the broadcast *idea* into `decode.cuh` even without TMA binary. |
| TRT-LLM FMHA v2 decode (Blackwell) | Wraps TRT-LLM FMHA v2 (CUTLASS); packed/separate/paged KV; per-req & per-block quant scales; deterministic chunked softmax | Kv_block_array + mBlockOffsets; TMA descriptors; CUTLASS collective (m16n8 tile); softmax stats `[B,H,2]` | `trtllm_fmha_v2_binding.cu:39-50` (sm120 decls); `prefill.py` (`gen_trtllm_fmha_v2_sm120_module`) | **RUNS** (explicit `_sm120_nl_tiled` kernels e.g. `...bf16_64_128_S_q_k_v_192x128_sm120_nl_tiled`). Vendor kernel, ~1.1-1.2x vs hand FA2. Usable as a drop-in if you accept the TRT-LLM dependency. |
| Gated Delta Rule (GDN) decode | Linear-attn state update + gating: `q·kᵀ + decay·old → new`; beta gate; BF16 (TC) or FP32 (scalar) state | State `[B,HV,V,K]`/`[B,HV,K,V]`; A_log `[HV]`; tile_v∈{8,32}, vec∈{4,8} | `gdn_decode.py:56-106`; `gdn_kernels/` (no sm120 codegen confirmed) | **NEEDS-PORT / DEAD.** SSM, not softmax attention. No confirmed sm120 codegen. Does NOT help a softmax GGUF decoder — skip. |
| NVFP4 (4-bit) KV cache | K,V → 4 bits + per-16-elt FP8 block scale, packed 2/byte; dequant at load; works in XQA & batch_decode | KV uint8 `[...,D/2]`; scales FP8 `[...,D/16]`; DP4A/FMA dequant | `fp4_kv_quantization.cu`, `fp4_kv_dequantization.cu`; `xqa.py` (`k_sf_cache`,`v_sf_cache`) | **RUNS.** DP4A universal; XQA NVFP4 KV sm120-gated (`xqa.py:232`). 2x KV shrink — high value on 24 GB. Port the pack/unpack + block-scale dequant. |
| Cascade decode (KV reuse) | QK matmul once, reuse scores across queries to amortize K-cache BW | Same paged KV + QK intermediate workspace; tiled QK then sparse V gather | `cascade.py` (API); `attention/cascade.cuh` | **RUNS** (warp mma, no wgmma). Marginal at batch=1 (no cross-query reuse in single-token decode). The LSE-merge half (idea #5) is the reusable bit. |

---

## Subsystem 2 — Prefill / FMHA

| Technique | How implemented (mechanism) | Kernel/layout/instruction | source file:line | sm120_fit |
|---|---|---|---|---|
| TRT-LLM Gen FMHA (paged context) | Kernel-selection loop + CTA launch planning by problem shape; cubin loader of precompiled kernels per SM/dtype/head_dim; QK→softmax→PV split-K; block-scaled FP4/FP8 KV (per-128-elt scale) | Paged KV w/ block_tables; Q NHD/HND; split-K reduction kernels; 1 MB guard-pad workspace for softmax stats | `trtllm_fmha_kernel_launcher.cu:96-313` | **RUNS.** `m16n8k16/k32 mma.sync` for FP16/FP8 QK & PV (no tcgen05); TMA on sm_120. Block-scale needs per-block scale gather; MMA handles mixed-precision accum. |
| FMHA v2 (FA2 backend) split-K prefill | Classic FA2: iterate KV blocks, Q·Kᵀ (BF16/FP16), causal mask + sinks, softmax, P·Vᵀ; `fp16_qk_reduction` flag; 1×N/4×N/multi-CTA scheduling | Packed QKV / contig KV / paged (Q_PAGED_KV) / separate; stride addressing HND vs NHD; paged block-offset table | `fmha_v2_run.cu:45-200`; `fmha_v2/fused_multihead_attention_utils.h` | **RUNS.** `ldmatrix m8n8x4` (`>=750`) + `m16n8k8/k16 mma.sync` — no wgmma/tcgen05. Batch=1 prefill is BW-bound (O(BHN²) FLOPs but load-bound). Best portable prefill reference. |
| Blackwell SM100/SM103 CUTLASS 3.x FMHA | CUTLASS 3.x CollectiveBuilder MMA collectives + TMA; warp-specialized mainloop (TMA threads vs math threads); cluster launch; tcgen05 on SM100 w/ collective fallback | TMA desc global→smem w/ predicate mask; smem A/B per-stage + multi-CTA barrier; `kNumMathThreads`/`kNumTMAThreads` split | `attention/blackwell/device/fmha.hpp:49-150`; `collective/sm100_fmha_gen_mainloop_warpspecialized.hpp` | **NEEDS-PORT → effectively DEAD.** Uses **tcgen05/wgmma** cluster MMA + cluster launch, absent on sm_120. Re-port would mean replacing collective MMA with `m16n8k16/k32` tile loops + new thread/smem layout. Not auto-selected for sm_120. |
| FP4/FP8 block-scale GEMM (deep_gemm) | Per-128-elt (BLOCK_K=128) row/col scaling; wgmma (SM90+) for Q·K·scales then P·V; FP8 e4m3/e5m2 fallback to `m16n8k32 mma.sync` pre-SM90 (manual tiling) | Scales per 128-tok chunk row-major; TMA loads A+scales_a, B+scales_b; smem per-stage + shared scales; TMA/math barrier | `nv_internal/tensorrt_llm/deep_gemm/fp8_gemm_impl.cuh:64-150` | **RUNS (via m16n8k32) / NEEDS-PORT (no CUTLASS path).** Raw FP8 `mma.sync m16n8k32` works on sm_120 (`mma.cuh:223-299`); the wgmma CUTLASS path does not. Manual tiling required if not using SM90 collective. |
| Quantized softmax + sinks + soft-cap | Skip softmax when residual < threshold; sink tokens prepended into output accumulator; tanh logits soft-cap pre-softmax; return LSE for split-K merge | LSE `[B,H]` in workspace; sinks as separate prepended tensor; per-block threshold check after QK reduce | `trtllm_fmha_kernel_launcher.cu:174-182`; `prefill.py:240-296` | **RUNS.** All scalar/branch ops + warp-shuffle min/max reduce. No special instruction. Drop-in correctness + perf features. |
| Split-K prefill + parallel reduction | Each CTA does a KV chunk → partial P + LSE; CTA-per-chunk tiling; merge kernel: max-reduce LSE + weighted-sum P; 2-pass forward+merge | Output workspace `[Nq,H,Dvo]` per split; LSE workspace `[Nq,H,nsplit]`; merge via atomic/sequential reduce | `fmhaReduction.cu` (in launchers); `trtllm/fmha/fmhaReduction.h` | **RUNS.** Standard grid + FP32 atomic add (sm_120 OK). Merge pass is BW-bound. Same LSE-merge primitive as decode split-K (idea #5). |
| Paged KV w/ TMA | Block-table indirection seq_id→page_ids; TMA descriptor = base + block-offset strides; async-load whole page to smem, compute on tile | Page size (e.g. 16) fixed; block tables `[B,maxBlk]` int32; KV pool linear; HND/NHD strides | `trtllm/fmha/fmhaKernels.cuh:78-150`; `fmha_v2_run.cu:115-136` | **RUNS** (`cp.async`/TMA on `>=90`/sm_120; predicated masking). But paging is multi-request flexibility — **batch=1 uses contiguous KV**, strictly simpler. Skip paging; keep TMA tile-load idea. |

---

## Subsystem 3 — Quant GEMM (CUTLASS)

| Technique | How implemented (mechanism) | Kernel/layout/instruction | source file:line | sm120_fit |
|---|---|---|---|---|
| W4A16 / W4A4 NVFP4 CUTLASS GEMM (SM120) | `CutlassFp4GemmRunner<T, W4A4_NVFP4_NVFP4>` → `genericFp4GemmKernelLauncher`, CTA shape + 1×1×1 cluster; TMA block loads; warp-specialized, 1-SM constraint | TMA mainloop, FP4 block-scaled (UE8M0 per-128b) weight; tiles 128×{32,64,128,256}×128B, 256×128×128B; `TmaWarpSpecializedCooperativeSm120`/`PingpongSm120`; LinearCombination epilogue | `fp4_gemm_cutlass_sm120.cu:40-50`; `gemm/fp4_gemm_cutlass_template_sm120.h:113-147` | **RUNS** (native sm120 codegen, file confirmed). Warp `m16n8k16/k32` + TMA for FP4/UE8M0. batch=1 supported; occupancy ~60-70% at 128×128×128. Most useful GEMM donor for sm_120. |
| MXFP8×MXFP8 (block-scaled E4M3) GEMM (SM120) | `CutlassMxfp8GemmRunnerSm120` → `dispatchMXFP8xMXFP8GemmCTAShapeSm120` → `genericMxfp8GemmKernelLauncherSm120`; UE8M0 per-128b scales; cluster always 1×1×1 (no dynamic CGA unlike SM100) | E4M3 + UE8M0 scales; tiles 128×{32,64,128,256}×128B; TMA collective mainloop; cooperative/pingpong via swap_ab | `mxfp8_gemm_cutlass_sm120.cu:65-78`; `gemm/mxfp8_gemm_cutlass_template_sm120.h:44-89` | **RUNS** (native sm120). Block-scale CUTLASS MMA + E4M3, TMA 128b-aligned. batch_count>1 via loop unroll. ~65-75% occupancy at 128×128×128. |
| FP8 E4M3×E4M3 per-column-scale GEMM (generic SM90+) | `CutlassFp8GemmRunner` wraps `generic_mixed_gemm_kernelLauncher`; per-column scale_a/scale_b; SM90 `PtrArrayTmaWarpSpecializedCooperative`/`Pingpong`; batched via batch dim | Dense E4M3 row/col-major; per-col FP32 scales; CTA heuristic (default 128×128×64); clusters (1,1,1)…(2,2,1) on SM90 | `fp8_gemm_cutlass.cu:48-85`; `gemm/fp8_gemm_cutlass_template.h` | **NEEDS-PORT.** Targets SM90 TMA warp-spec. sm_120 needs `Mxf8f6f4TmaWarpSpecializedCooperativeSm120` schedule (`generate_kernels.py:638-644`); kernel body not in package. Scales FP32 (single global scale viable). |
| FP4/FP8 block-scale interleave layout | `block_scale_interleave` (`nvfp4_block_scale_interleave`) → `block_scale_interleave_sm100`; swizzles unswizzled scales into CUTLASS BlockScaledTensorOp layout post-quant; row/col pad (12× FP4) | UE8M0 block scales; 128-bit interleave; returns `[groups, padH, padW]` | `quantization/fp4_quantization.py` (`block_scale_interleave`); `gemm/cutlass_gemm_configs.h` (`CutlassTileConfigSM120`, 128B K-tile) | **RUNS.** UE8M0 native to sm120 MMA; 128b matches TMA unit. Needs JIT/prebuilt sm120 module (`gen_fp4_quantization_sm120_module`). Required preprocessing if you adopt block-scaled GEMM. |
| MoE grouped GEMM (CUTLASS TmaWarpSpecializedGroupedGemmInput) | `CutlassMoeFCRunner` → `tma_warp_specialized_generic_moe_gemm_kernelLauncher` (SM90 macro); per-expert M×K×N GEMM; problem-visitor scheduling; mixed E4M3×FP4; `is_mx_fpx` block-scale variant | Grouped descriptor `[m_i,n_i,k_i,lda,ldb,ldc]`; cute Shape cluster/tile; TmaWarpSpecializedCooperative (SM90); EpilogueOpDefault/Bias | `jit/gemm/cutlass/generate_kernels.py:240-297`; `fused_moe/cutlass_backend/cutlass_fused_moe_instantiation.cu:20-69` | **NEEDS-PORT → DEAD for bw24.** sm120 MoE variants declared (`generate_kernels.py:755-851`) but kernel bodies absent; cluster forced 1×1×1. MoE routing is orthogonal to single-stream dense decode. |
| Deep GEMM (prefill MoE, vectorized W4/W8) | `deep_gemm` E4M3/FP4/FP8-W × I4/U4/U8 path for prefill routing; per-channel/per-token scales; vLLM expert dispatch via `cutlass_kernel_selector.h` autotuner; dynamic M | Mixed E4M3/FP8/FP4 weight + E4M3 act, per-token(-per-expert) scales; dynamic-M problem visitor | `deep_gemm.py`; `fused_moe/trtllm_backend/trtllm_fused_moe_dev_kernel.cu` | **NEEDS-PORT.** Not sm120-tuned (SM80/SM90 focus); needs small-M (16/32) tiles + vector loads + custom persistent kernel. MoE/prefill-routing specific. |
| Block-scaled FP8 dense GEMM (`fp8_blockscale_gemm`, SM90) | TMA + MMA w/ block-scale epilogue fusion; per-128b UE8M0 scales; `-DENABLE_FP8_BLOCK_SCALE` (CUDA 12.8+); runner infers workspace + tactic | BlockScaledTensorOp UE8M0 per-128b; TMA weight load; MMA `sm90_fma`/`sm90_wgmma`; 128×128×128 CTA; LinCombBlockScaleFactor epilogue | `jit/gemm/fp8_blockscale.py:10-35`; `nv_internal/.../fp8_blockscale_gemm/` (SM90-only) | **DEAD.** Depends on **wgmma** (SM90). sm_120 would need a full re-port to `mma.sync m16n8k16` + block-scale epilogue; not exposed for sm_120. |

---

## sm_120 reality cross-check (from the source guards)

`data/include/flashinfer/mma.cuh:30-49` is the authoritative gate. For bw24 it means:
- **Take the warp-MMA attention + FP8/FP4 KV ideas** (decode loop, FP8 `m16n8k32`, NVFP4 KV, LSE
  merge, split-K-for-long-seq, soft-cap/sink/skip-softmax) — all compile and run on sm_120.
- **Lean on the shipped sm120 GEMM kernels** (`fp4_gemm_cutlass_sm120.cu`,
  `mxfp8_gemm_cutlass_sm120.cu`, `group_gemm_*_groupwise_sm120.cu`) as references — they prove the
  CUTLASS sm120 schedule names (`TmaWarpSpecialized{Cooperative,Pingpong}Sm120`, 1×1×1 cluster).
- **Ignore everything wgmma/tcgen05** (Blackwell SM100 FMHA, fp8_blockscale, SM90 fp8/MoE
  collectives) and all multi-request paging/server machinery.
