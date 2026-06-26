# sm_120 Empirical Capability Ledger — RTX 5090 Laptop

Two classes of fact, kept strictly separate:
- **HARD FACTS** = physical silicon properties. Immutable. A newer toolchain cannot change them.
  These drive all architecture decisions.
- **CURRENT STATE** = software/toolchain/runtime that we CAN change (install newer CUDA, CUTLASS,
  free up RAM, etc.). Never design a constraint around these — only around hard facts.

All measured on-device 2026-06-26 (CUDA 13.1 nvcc, driver 595) by compiling/running/assembling.

---

## HARD FACTS — silicon (immutable)

### Device properties (measured)

| Property | Value | Why it's hard |
|---|---|---|
| Compute capability | 12.0 (**sm_120**, consumer Blackwell GB203) | fixed in silicon |
| SMs | **82** (desktop 5090 = 170; laptop ≈ half) | fixed |
| Peak mem bandwidth | **896 GB/s** (256-bit GDDR7 @ 14001 MHz) | fixed bus width × clock |
| **Achieved read BW** | **829 GB/s = 92.5% of peak** (measured, -O3 float4 stream) | what kernels actually get |
| VRAM | 25.15 GB total (~24 GB usable) | fixed |
| L2 cache | **64 MB** (very large — exploit for KV/prefix locality) | fixed |
| smem/SM | 100 KB (99 KB opt-in per block) | fixed — tile budget |
| regs/SM | 65536 | fixed — occupancy budget |
| maxThreads/SM | 1536 | fixed |
| copy engines | 2 (compute/copy overlap, bidir) | fixed |
| clusterLaunch | supported | fixed |
| Power cap | ~150 W typical, **up to 175 W peak** | thermal/laptop hard limit |

### Tensor-core / ISA — what the silicon can execute

These are HARD: wgmma/tcgen05 are absent because those tensor-core *generations don't exist on
sm_120 silicon*. No ptxas/CUDA version can add them. The dtype MMAs that pass are the real ISA.

| Feature | sm_120 silicon | Evidence |
|---|---|---|
| FP16/BF16 `mma.sync.m16n8k16` | ✅ executes | ran on GPU |
| FP8 `mma.sync.m16n8k32` e4m3 + e5m2 | ✅ executes | ran on GPU |
| FP4 e2m1 block-scale `mma.sync.m16n8k64.kind::mxf4.block_scale.scale_vec::2X...ue8m0` | ✅ executes | assembled to cubin |
| `wgmma` (Hopper warpgroup MMA) | ❌ **absent** | ptxas: "not supported" — silicon lacks it |
| `tcgen05.mma` (datacenter 5th-gen TC + tmem) | ❌ **absent** | sm_100-only silicon feature |
| TMA `cp.async.bulk` | ✅ present | instruction accepted |

### Measured compute peaks (tensor core, this GPU)

| dtype | measured peak | ratio | crossover AI (vs 829 GB/s) |
|---|---|---|---|
| FP16/BF16 mma | **117 TFLOP/s** | 1.0x | ~141 FLOP/byte |
| FP8 e4m3 mma | **219 TFLOP/s** | 1.88x | ~264 FLOP/byte |
| FP4 e2m1 (block-scale) | ~**440 TFLOP/s** (est, ≈2x FP8; bench pending) | ~3.8x | ~530 FLOP/byte |

(Microbench = tight independent-mma loop, 2 accumulators, 82×4 blocks. Real GEMM hits ~70-85% of
this with good tiling. Sparsity 2:4 could ~2x again — to verify.)

### THE architecture-defining conclusions (from hard facts)

1. **sm_120 programming model = Ada-style warp-level `mma.sync` + Blackwell FP4/FP8 dtypes + TMA + clusters.
   NOT the Hopper/datacenter (wgmma/tcgen05/tmem) model.**
   - ❌ CUTLASS **sm_100** kernels and **FlashAttention-3** (both wgmma/tcgen05) WILL NOT RUN.
     → use CUTLASS **SM120 collectives** (warp-MMA + block-scale FP4) and FA-2-style `mma.sync` attention.
   - ✅ FP4 (nvfp4/mxfp4) hardware block-scale MMA present → headline weapon: ~3.8x FP16 compute AND 4x
     smaller weights → fits big models in 24GB AND moves 4x fewer bytes.

2. **Everything in decode is bandwidth-bound.** Decode arithmetic intensity ≈ 1-2 FLOP/byte; crossover
   to compute-bound is 141 (FP16) / 264 (FP8) / 530 (FP4) FLOP/byte. So single-stream decode speed is
   set ENTIRELY by bytes-moved-per-token, i.e. weight + KV quant. Low-bit wins by shrinking bytes, not FLOPs.
   - Beat-target anchor: 7B Q4 (~3.8 GB) → **~218 tok/s** single-stream ceiling (829/3.8).
   - Only large-batch prefill / many concurrent requests push into compute-bound, where FP4 TFLOPs matter.

3. **64 MB L2 is unusually large** for this class → prefix-cache / KV / hot-weight locality is a real lever
   competitors may underuse.

---

## CURRENT STATE — mutable (do NOT design constraints around these)

- **Toolchain:** nvcc 13.1 (also 12.8), driver 595 (CUDA 13.2 runtime), cuBLAS/cuBLASLt 13.2, CUB/CCCL,
  cuda_fp4/fp6/fp8 headers, cmake/ninja/gcc/rustc. **We can install newer CUDA (13.2/13.3+), CUTLASS,
  any Rust/C++ deps as research dictates.** CUTLASS not yet fetched.
- **No torch; Python 3.14.** Can install whatever runtime we choose — not a constraint on stack choice.
- **Free host RAM ~12-16 GB right now** (other LLM servers running) — TEMPORARY. Could be more/less.
  → spilling must query free RAM at runtime and size the host tier dynamically; never hardcode it;
  fall back to mmap'd disk when tight. (This is a *design requirement from variability*, not a fixed budget.)

### Toolchain-version-specific gotcha (current nvcc 13.1)

FP4 block-scale MMA compiles ONLY with `-gencode arch=compute_120a,code=sm_120a`.
The `-arch=sm_120a` shortcut silently misroutes through a `compute_120` (no `a`) PTX intermediate in
the full compile pipeline → ptxas rejects `.block_scale`/`.kind::mxf4`/`.scale_vec::2X`.
(May differ in a newer nvcc — re-check after any toolchain upgrade. Not a hardware limit.)
