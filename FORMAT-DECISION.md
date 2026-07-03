# FORMAT-DECISION.md — what bw24 optimizes for (2026-07-04)

**User's criterion (verbatim intent):** the decision is NOT what we already built and NOT the
current daily models — models change daily, the RIG doesn't. Optimize for the fastest possible
inference on THIS rig (sm_120a, 24GB, RTX 5090 Laptop, this CPU), for ANY model that may become
the strongest local candidate — including a future spilled/REAP'd MoE running CPU+GPU. Multi-month
rework is acceptable; this is a long-running inference project, not a weekend build.

## Decision: RIG-NATIVE internal layout; GGUF and safetensors are both just IMPORT formats

Neither GGUF blocks nor HF/modelopt safetensors layouts are the optimization target. The target is
a **bw24-internal weight layout chosen per weight-class by what the sm_120a datapath wants**, with
BOTH file formats repacked into it at load (one-time cost, irrelevant at serve time).

Why not GGUF-first: GGUF block layouts were designed for llama's dp4a/MMQ kernels. The FASTEST
sm_120a path for 4-bit is the mxf4nvf4 block-scale tensor-core mma (762 TFLOPS), and its operand
layout is NOT GGUF's 36-byte block_nvfp4 — we already pay a repack+swizzle (cutlass path builds
"repacked B + swizzled SFB ALONGSIDE bytes" = dual VRAM copies on a 24GB card). NVIDIA's native
NVFP4 releases (modelopt, faster than community GGUF quants of the same models) land in ST form
first; a GGUF-internal engine converts them THROUGH a lossy-in-layout hop.

Why not ST-first: ST is a container, not a runtime layout either. modelopt packing is closer to
what the tensor core wants for PREFILL, but decode (matvec, bandwidth-bound) and CPU-side spilled
experts (AVX/AMX dot on the host CPU) each have their own optimal layouts. No file format is the
answer; the RIG's three datapaths are:

| datapath | optimal internal layout | today's state |
|---|---|---|
| prefill GEMM (tensor-core mxf4nvf4 / int8 mma) | operand-order + swizzled scale factors exactly as `mma.sync`/TMA consume them (modelopt-shaped for FP4) | GGUF blocks + per-load repack for cutlass; MMQ reads GGUF raw (llama-shaped, at parity) |
| decode matvec (HBM-bound) | coalesced per-warp block reads — current GGUF-raw layout is ALREADY at llama parity (42% DRAM, SOL) | GGUF raw — keep until a measured better layout exists |
| CPU-spilled experts (future MoE) | host-page-aligned, mmap-direct, dtype the CPU dots fast (int8/bf16 rows) | not built |

## Execution order (does NOT invalidate current work)

1. **Now — keep shipping on GGUF-raw kernels.** They are at/above llama parity and every gate is
   built around them. Nothing regresses.
2. **Loader unification (already 80% built):** `GpuTensor` is the abstraction; safetensors→repack
   at load already exists (modelopt NVFP4 → engine blocks, "no kernel change"). Formalize: every
   loader produces INTERNAL layout, tagged per weight-class; GGUF-raw is one internal layout among
   several, not the definition.
3. **Per-class migration only WITH measured wins** (the standing empirical rule): when a prefill
   FP4 kernel on modelopt-shaped operands beats the MMQ-on-GGUF floor interleaved, that class
   flips its internal layout and the GGUF loader gains the (already-written) repack for it. Decode
   stays GGUF-raw until a layout beats 42% DRAM.
4. **Spill tier (MoE arc)** designs against ST shards mmap'd directly (no conversion on the cold
   path), host-side layout chosen by CPU dot benchmarks — this is where ST-native pays off first.
5. **Dual support is the end state** (user: "probably at a later point should support both") —
   both as IMPORTS. The engine's identity is the rig-native layout, not either container.

## What this changes about ongoing optimization

- Tune-data records gain a `layout` field going forward (GGUF-raw vs repacked variants) so the
  autotuner learns layout as a dimension.
- New kernels are written against an OPERAND SPEC (what bytes in what order), not "the GGUF
  block" — provenance comments say which container maps to it and how.
- The FP4 prefill rebuild (PREFILL-GEMM-REBUILD.md) targets the tensor-core-native operand layout
  FIRST and treats the GGUF repack as the import step, not the other way around.
