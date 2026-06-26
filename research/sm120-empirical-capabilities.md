# sm_120 Empirical Capability Ledger — RTX 5090 Laptop

Ground-truthed by compiling/running/assembling on the actual GPU (2026-06-26, CUDA 13.1 nvcc, driver 595).
These are facts from this machine, not web claims.

## Device properties (measured)

| Property | Value |
|---|---|
| Name | NVIDIA GeForce RTX 5090 Laptop GPU |
| Compute capability | 12.0 (sm_120) |
| SMs | **82** (desktop 5090 = 170; laptop ≈ half) |
| Peak mem bandwidth | **896 GB/s** (256-bit GDDR7 @ 14001 MHz) |
| VRAM | 25.15 GB total (24463 MiB usable reported) |
| L2 cache | **64 MB** |
| smem/SM | 100 KB (99 KB opt-in per block) |
| regs/SM | 65536 |
| maxThreads/SM | 1536 |
| asyncEngines (copy) | 2 |
| clusterLaunch | supported (1) |
| cooperativeLaunch | supported (1) |
| Power cap (observed) | ~150 W |

**Decode is memory-bound.** Single-stream ceiling ≈ 896e9 / bytes_per_token.
- 7B Q4 (~3.8 GB resident) → ~235 tok/s theoretical max.
- This is the headline number to beat-target against (vs llama.cpp/vLLM on same chip).

## Tensor-core / ISA capability (compiled + assembled)

| Feature | sm_120 status | Evidence |
|---|---|---|
| FP16/BF16 `mma.sync.m16n8k16` | ✅ runs | launched OK |
| FP8 `mma.sync.m16n8k32` e4m3 | ✅ runs | launched OK |
| FP8 `mma.sync.m16n8k32` e5m2 | ✅ runs | launched OK |
| FP4 e2m1 block-scale `mma.sync.m16n8k64.kind::mxf4.block_scale.scale_vec::2X...ue8m0` | ✅ assembles to cubin | **only with `-gencode arch=compute_120a,code=sm_120a`** |
| `wgmma` (Hopper warpgroup MMA, sm_90a) | ❌ **rejected** | "wgmma.mma_async ... not supported" |
| `tcgen05.mma` (datacenter 5th-gen TC + tmem, sm_100) | ❌ **rejected** | "not supported on sm_120" (and sm_100a needs different form) |
| TMA `cp.async.bulk` | ✅ assembles | instruction accepted |
| `__nv_fp4_e2m1` host/device type (`cuda_fp4.h`) | ✅ works | cvt to float OK |

## THE architecture-defining conclusion

**sm_120 (consumer Blackwell GB20x) programming model = Ada-style warp-level `mma.sync`
+ Blackwell FP4/FP8 dtypes + TMA + clusters. It is NOT the Hopper/datacenter model.**

Therefore:
- ❌ CUTLASS **sm_100** kernels (tcgen05/tmem) will NOT run → must use CUTLASS **SM120 collectives** (warp-MMA + block-scale FP4). This is the path vLLM/SGLang nvfp4 kernels take.
- ❌ FlashAttention-3 (wgmma-based) will NOT run → need FA-2-style `mma.sync` attention, or FlashInfer's sm_120 path, or hand-rolled.
- ✅ FP4 (nvfp4/mxfp4) is our headline weapon — hardware block-scale MMA present. ~2x FP8 throughput, 4x FP16, fits big models in 24GB.
- ✅ FP8 fully available as a safe high-quality fast path.

## CRITICAL BUILD FLAG (the trap)

FP4 block-scale MMA compiles ONLY with:
```
nvcc -gencode arch=compute_120a,code=sm_120a ...
```
The shortcut `-arch=sm_120a` **silently misroutes** through a `compute_120` (no `a`) intermediate
PTX in the full (non-`-ptx`) compile pipeline, and ptxas then rejects:
`'mma with block scale' not supported on .target 'sm_120'`, `.kind::mxf4`, `.block_scale`, `.scale_vec::2X`.

Always use the explicit `-gencode arch=compute_120a,code=sm_120a` form for any FP4/block-scale TU.

## Toolchain

- nvcc 13.1 (V13.1.115), also CUDA 12.8 available
- driver 595.71.05 (CUDA 13.2 runtime), cuBLAS/cuBLASLt 13.2
- ships cuda_fp4.h / cuda_fp6.h / cuda_fp8.h, CUB/CCCL
- sm_120 / sm_120a / sm_121a all compile (121a = laptop-specific, untested for features)
- CUTLASS NOT installed standalone (must fetch)
