# sm_120 Empirical Capability Ledger — RTX 5090 Laptop

Two classes of fact, kept strictly separate:
- **HARD FACTS** = physical silicon properties. Immutable. A newer toolchain cannot change them.
  These drive all architecture decisions.
- **CURRENT STATE** = software/toolchain/runtime that we CAN change (install newer CUDA, CUTLASS,
  free up RAM, etc.). Never design a constraint around these — only around hard facts.

All measured on-device 2026-06-26 (CUDA 13.1 nvcc, driver 595) by compiling/running/assembling.

> **THIS DOC IS THE CANONICAL MMA RATE TABLE for sm_120a** (§ *CANONICAL MMA RATE TABLE*, below —
> 12 forms, cyc/warp-MMA, plus the ptxas-verified list of forms that do NOT exist). Every inline-PTX
> MMA site in `crates/memra-engine/cu/` points here with a `rate-audited 2026-08-06` comment.
>
> **Before you write or change an inline MMA, check its rate here.** Two PTX spellings can compute
> the bit-identical product at 2x different issue rates, and *no correctness gate can see the
> difference* — that is how three live kernels shipped the slow `kind::f8f6f4` form for a month
> while this very file already had the measurement. Per-site verdicts (OPTIMAL / SWAP-AVAILABLE /
> DEAD-DOOR / NOT-APPLICABLE): `research/ptx-audit-20260806/AUDIT.md`.

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
| max SM clock | 3090 MHz | fixed |
| thermal target | 87 °C (asus nv_temp_target max) | the REAL sustained limit |

### Power: NOT a hard constraint (settings bug, patch pending)

The "150 W cap" is a **firmware-settings bug**, not silicon. `asus-armoury/attributes/nv_tgp.max_value`
is wrongly clamped to 150; spec is TGP 150 + dynamic_boost 25 = **175 W** (nvidia-smi `power.max_limit`).
User has sent a patch to the asus-linux maintainer raising the `nv_tgp` ceiling; tracking it.
**Therefore do NOT design around 150 W (or even 175 W) as a hard power-bound.** The real sustained
limit is THERMAL (87 °C target) + whatever the patch unlocks. Benchmark at full power:
`gpu-full-power on` (sets nv_tgp=150, boost=25, profile=performance) — without it the box sits at
`balanced`/boost=5 and measurements are throttled.

Microbench note: issue-rate peaks below barely move between balanced and full-power (short bursts
don't sustain to the power wall: 829→847 GB/s, FP8 219→233, FP4 762→761 TFLOP/s) so the ROOFLINE
NUMBERS ARE VALID regardless. Power/thermal only bites SUSTAINED real decode/prefill — measure those
at full power, and re-measure after the TGP patch lands.

### Tensor-core / ISA — what the silicon can execute

These are HARD: wgmma/tcgen05 are absent because those tensor-core *generations don't exist on
sm_120 silicon*. No ptxas/CUDA version can add them. The dtype MMAs that pass are the real ISA.

All re-tested with the CORRECT `-gencode arch=compute_120a,code=sm_120a` flag (the bare
`-arch=sm_120a` shortcut produced false-negatives on block-scale — see gotcha below).

| Feature | sm_120 silicon | Evidence |
|---|---|---|
| FP16/BF16 `mma.sync.m16n8k16` | ✅ executes | ran on GPU |
| FP8 `mma.sync.m16n8k32` e4m3 + e5m2 (plain) | ✅ executes | ran on GPU |
| **FP4 e2m1 block-scale** `mma.sync.m16n8k64.kind::mxf4.block_scale.scale_vec::2X..ue8m0` | ✅ **executes** | ran on GPU (correct flag) |
| **FP8/6/4 block-scale** `mma.sync.m16n8k32.kind::mxf8f6f4.block_scale..ue8m0` | ✅ **executes** | ran on GPU |
| NVFP4 unified `kind::mxf4nvf4.block_scale.scale_vec::4X..ue4m3` | ✅ **executes** | ran on GPU. The 2026-06-26 "⚠️ my PTX form rejected" was the *operand spelling*, not silicon — resolved: `mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3` (needs `.block_scale`, and `ue4m3` not `e4m3`). Live at `mmq_fp4.cu:194`; 619.2 TFLOP/s, row D1 below. (This row also had a missing table cell for six weeks — fixed 2026-08-06.) |
| `wgmma` (Hopper warpgroup MMA) | ❌ **absent** | ptxas rejects even w/ correct flag — silicon lacks it |
| `tcgen05.mma` (datacenter 5th-gen TC + tmem) | ❌ **absent** | ptxas rejects even w/ correct flag — sm_100-only silicon |
| TMA `cp.async.bulk` | ✅ present | instruction accepted |

### Measured compute peaks (tensor core, this GPU)

> **⚠️ SUPERSEDED as a rate reference by the CANONICAL MMA RATE TABLE below (2026-08-06).** This
> older block is kept for its roofline crossover-AI column and its history. Where the two disagree,
> **the canonical table wins**: it measures *cyc/warp-MMA per form* with an ILP control and a SASS
> instruction census, whereas this one reported ILP-dependent TFLOP/s with 2 accumulators.

| dtype (FP32 accumulate) | measured peak | ratio vs FP16 | crossover AI (vs 829 GB/s) |
|---|---|---|---|
| FP16/BF16 mma | **117 TFLOP/s** | 1.0x | ~141 FLOP/byte |
| FP8 e4m3 **plain** mma | **219 TFLOP/s** | 1.88x | ~264 FLOP/byte |
| FP8 **block-scale** (mxf8f6f4) | **381 TFLOP/s** | 3.26x | ~460 FLOP/byte |
| FP4 e2m1 **block-scale** (mxf4) | **762 TFLOP/s** | 6.52x | ~920 FLOP/byte |

(Microbench = tight independent-mma issue loop, 2 accumulators, 82×4 blocks — upper bound on
issue rate. Real GEMM hits ~70-85% with good tiling. Internally consistent: plain vs block FP8
have identical FLOP/instr yet block runs 1.74x faster → block-scale lifts the FP32-accumulate
throttle. KEY FINDING: the **block-scaled** path (mxf8f6f4/mxf4) IS a genuine compute win, NOT
just a bytes-saver. This refutes the "FP8≈FP16, FP4 no compute win" claim — that holds only for
the *plain* (non-block-scale) mma path. Sparsity 2:4 may ~2x again — to verify.)

> **2026-08-06 — THIS TABLE WAS RIGHT AND THREE LIVE KERNELS IGNORED IT FOR A MONTH.** The plain-vs-
> block-scale row above is the whole finding, and it was sitting here unused: `mmq_nvfp4_w4a8.cu`,
> `mmq_fp8_blk.cu` and the v3 gate's `mmq_q8_0_f32acc.cu` all issued **plain** `kind::f8f6f4` while
> their comments claimed the "381-TF class". Re-measured per-instruction on the real tiles: plain
> `kind::f8f6f4.m16n8k32` = **32.02 cyc/warp-MMA**, `kind::mxf8f6f4.block_scale.scale_vec::1X` with
> the **ue8m0 identity scale `0x7F7F7F7F`** = **16.06 cyc** for the *bit-identical* e4m3×e4m3
> product (1.994x, slightly wider than this table's 1.74x because that microbench is an issue-rate
> upper bound with different ILP). Yields: W4A8 tile **1.2153x** e2e prefill, per-block FP8 tile
> **1.0654x** e2e pp512, and the v3 gate's Q1 verdict **inverted** (+17.6pp → −8.8pp; its s32
> control was racing a 2x-slower f32 arm). Receipts: `research/w4a8-prefill-20260806/`,
> `research/rp-on-st-20260806/`. **Lesson: the identity scale makes the fast form a drop-in for any
> plain `kind::f8f6f4` site — if you write one, justify it against this row or use the block_scale
> form.** That lesson is what produced the canonical table below.

### ★ CANONICAL MMA RATE TABLE — every form the repo issues (2026-08-06)

**This is THE rate reference for sm_120a. Cite this table, not a comment, not a design doc.** It
exists because the plain-vs-block finding above was already measured a month earlier and three live
kernels still picked the slow form: a rate that lives only in prose gets re-picked wrong. Every
inline-PTX MMA site in `crates/memra-engine/cu/` now carries a `rate-audited 2026-08-06` pointer
here, and the per-site verdicts are in `research/ptx-audit-20260806/AUDIT.md`.

Method: `clock64()` around a tight loop of mutually-independent MMAs, **NACC swept 1..16** — flat
across NACC ⇒ the number is a pipe **ISSUE INTERVAL**, not a latency/ILP artifact. Converted to
delivered rate by full-GPU `cudaEvent` at the shipped 82 CTA × 256 thr shape. Locked clocks
1860/1860, `flock`'d, **3 reruns agreeing within 0.5%**, and **every arm's SASS MMA count
verified** with `cuobjdump -sass` (see the census caveat below — it caught two probe bugs).
Probe: `research/ptx-audit-20260806/tools/rate_audit.cu`. Raw: `logs/rate-audit-12form.log`.

Site line numbers are as of commit `9fd00b3f`+ (the pointer-comment commit shifted them).

| # | PTX form | cyc/warp-MMA | MACs/MMA | delivered | vs int8-k16 | repo sites |
|---|---|---|---|---|---|---|
| A1 | `m16n8k16.row.col.s32.s8.s8.s32` | 16.06 | 2048 | 155.2 TOP/s | 1.000x | `mmq_nvfp4_w4a8:219`, `mmq_iq_experts:157` — **both HOT, both at half the available int8 rate** |
| **A2** | **`m16n8k32.row.col.s32.s8.s8.s32`** | **16.06** | 4096 | **309.7 TOP/s** | **1.997x** | `mmq_q8_0:152`, `mmq_q45k:157`, `mmq_q4_0:164`, `qmatvec_gemm:168`, `mmq_q8_0_f32acc:157` |
| B1 | `m16n8k16.f32.bf16.bf16.f32` | 32.03 | 2048 | 77.7 TFLOP/s | 0.500x | `flash_attn:160`, `hybrid:1508`, `mma_tile.cuh:132` (dead file) |
| B2 | `m16n8k16.f32.f16.f16.f32` | 32.03 | 2048 | 77.8 TFLOP/s | 0.501x | `moe_f16_grouped:365`, `hybrid:1518` |
| **B3** | **`m16n8k16.f16.f16.f16.f16`** (f16 accum) | **16.10** | 2048 | **155.2 TFLOP/s** | **1.001x** | `flash_attn:988` |
| B4 | `m16n8k8.f32.tf32.tf32.f32` | 32.03 | 1024 | 38.9 TFLOP/s | 0.250x | — (slowest form on the rig) |
| C1 | `kind::f8f6f4` **plain**, e4m3×e4m3 | 32.03 | 4096 | 155.5 TFLOP/s | 1.002x | `mmq_nvfp4_f8f4:51` (uncalled), rollback seams `mmq_fp8_blk:251` / `mmq_nvfp4_w4a8:1094` / `mmq_q8_0_f32acc:195` |
| **C2** | **`kind::mxf8f6f4.block_scale.scale_vec::1X` ue8m0, e4m3×e4m3** | **16.06** | 4096 | **309.3 TFLOP/s** | **1.99x** | `mmq_fp8_blk:256`, `mmq_nvfp4_w4a8:1099`, `mmq_q8_0_f32acc:200` — **the defaults; the fast form** |
| C3 | `kind::f8f6f4` **plain**, e2m1×e4m3 | 32.03 | 4096 | 155.4 TFLOP/s | — | — |
| C4 | `kind::mxf8f6f4.block_scale.scale_vec::1X` ue8m0, e2m1×e4m3 | 16.06 | 4096 | 309.6 TFLOP/s | — | — |
| **D1** | **`m16n8k64.kind::mxf4nvf4.block_scale.scale_vec::4X` ue4m3** | **16.06** | 8192 | **619.2 TFLOP/s** | **3.99x** | `mmq_fp4:194`, `qmatvec_gemm:1243` |
| D2 | `m16n8k64.kind::mxf4.block_scale.scale_vec::2X` ue8m0 | 16.06 | 8192 | 619.1 TFLOP/s | 3.99x | — |

**Do not invert C1/C2.** Plain is the **slow** 32.03 form; `block_scale` is the **fast** 16.06 one,
and the live defaults are the fast ones. (An intermediate trace in the audit lane reported these
backwards, which would have "justified" reverting two correct fixes — see AUDIT.md §6.)

**Three mechanisms, in order of how much they cost the repo:**

1. **The int8 tensor pipe is K-FREE.** A1 and A2 both cost **16.06 cyc** — the *same* interval for
   *twice* the K depth. So `m16n8k32.s8` delivers **1.997x** the MAC rate for the identical product,
   and every k16 int8 site runs at **half** the available int8 rate. This is a second instance of
   the same bug class as the f8f4 find, in a different family.
2. **16-bit float with f32 accumulate is the slowest tensor path on this silicon.** B1 and B2 are
   both **32.03 cyc / ~77.7 TFLOP/s**, while B3 (f16 in, **f16 accumulate**) is **16.10 cyc /
   155.2 TFLOP/s** = exactly **2.0x**. This *measures* the f32-accumulate throttle the older block
   above could only infer from plain-vs-block FP8 — it taxes bf16 and f16 identically, so the
   operand format is free and the **accumulator** is what costs 2x. (It is also the rate half of
   why `MEMRA_FA_F16PV` is default-ON.)
3. **The KIND carries the FP8 cost, not the operand format.** C3/C4 track C1/C2 exactly, so an
   e2m1×e4m3 pair is no cheaper than e4m3×e4m3 — only `plain` vs `block_scale` moves the rate.
   Likewise D1 ≡ D2, so FP4 scale-vector granularity (4X ue4m3 vs 2X ue8m0) is free.

**ISA sibling oracle — what does NOT exist (verified, not assumed).** Every "could a deeper form
replace two shallower ones?" candidate was put to ptxas
(`research/ptx-audit-20260806/tools/isa_sibling_check.cu`, log `isa-sibling-check.log`). **All 7
REJECTED:** bf16 k32, f16 k32, bf16 `.block_scale`, s8 k64, f8f6f4-blocksc k64, f16-accum k32,
mxf4nvf4 k128. Also: **tf32 has no m16n8k16 shape** — ptxas says *"Illegal instruction types
specified for '_mma' with shape '.m16n8k16'"*; the ISA gives tf32 only `.m16n8k4`/`.m16n8k8`.
Consequences: the **k16→k32 int8 lift is the only depth lever the ISA offers**, and B1/B2 have **no
deeper form to escape to** — any bf16/f16 remedy must be an *accumulator* change (a numeric
decision), never a depth change.

**What a k16→k32 swap is actually worth (tile level, not instruction level).** 1.997x is the
instruction bound; a real MMQ tile also pays the scale fold, and the k16 form folds **twice** as
often (2 C tiles, 2 dA loads, 2 FMAs/element). Both inner loops replicated verbatim and measured:
**1.42x** full-GPU (0.713 → 0.502 ms, 82 CTA × 256 thr, 3 reruns bit-identical) = **~71% of the
bound**, the missing 29% being the doubled fold arity.
Probe `tools/k16_vs_k32_tileloop.cu`, log `logs/k16-vs-k32-tileloop.log`.

**SASS-census caveat — read before trusting any new rate probe.** An issue-rate probe is only valid
if the SASS really contains the MMA count you think it does. `cuobjdump -sass` caught **two** bugs
in the tile probe before any number was believed: (a) giving every accumulator the same A/B operands
let ptxas CSE all NACC copies *and* hoist them out of the loop (IMMA=1–2 for the whole kernel at
every NACC — the first "result" was pure fiction); (b) rotating the operand index by `4*i` **aliases
mod 32**, silently halving the NACC=16 arm. Also note ptxas unrolls the outer loop ~8x, so
per-instantiation `IMMA = NACC × mma_per_step × it_unroll`, floored at 8 — **only NACC ≥ 8 gives an
exact per-step count**, and low-NACC columns are ILP context only. Standing rule: **census the SASS
opcode count, and require it to equal the count you intended, before reading a cycle number.**
(SASS families on sm_120: `IMMA` int, `HMMA` 16-bit float/tf32, `QMMA` FP8/FP6/FP4-in-8b k32,
`OMMA` FP4 k64; a `.SF` suffix marks the scale-factor/block_scale variants.)

### THE architecture-defining conclusions (from hard facts)

1. **sm_120 programming model = Ada-style warp-level `mma.sync` + Blackwell FP4/FP8 dtypes + TMA + clusters.
   NOT the Hopper/datacenter (wgmma/tcgen05/tmem) model.**
   - ❌ CUTLASS **sm_100** kernels and **FlashAttention-3** (both wgmma/tcgen05) WILL NOT RUN.
     → use CUTLASS **SM120 collectives** (warp-MMA + block-scale FP4) and FA-2-style `mma.sync` attention.
   - ✅ FP4 (mxfp4) hardware block-scale MMA present → headline weapon: **6.5x FP16 compute** (762 TFLOP/s)
     AND 4x smaller weights → fits big models in 24GB AND moves 4x fewer bytes. Block-scaled FP8 = 381 (3.26x).

2. **Everything in DECODE is bandwidth-bound.** Decode arithmetic intensity ≈ 1-2 FLOP/byte; crossover
   to compute-bound is 141 (FP16) / 460 (block-FP8) / 920 (block-FP4) FLOP/byte. So single-stream decode
   speed is set ENTIRELY by bytes-moved-per-token (weight + KV quant). Low-bit wins decode by shrinking
   bytes, not FLOPs. BUT **PREFILL / large-batch is compute-bound** → there block-scaled FP4/FP8 is a real
   6.5x/3.3x compute lever (TTFT, batched throughput, and block-scaled attention QK/PV mainloops).
   - Beat-target anchor: 7B Q4 (~3.8 GB) → **~218 tok/s** single-stream ceiling (829/3.8).
   - Only large-batch prefill / many concurrent requests push into compute-bound, where FP4 TFLOPs matter.

3. **64 MB L2 is unusually large** for this class → prefix-cache / KV / hot-weight locality is a real lever
   competitors may underuse.

---

## CURRENT STATE — mutable (do NOT design constraints around these)

- **Toolchain:** nvcc 13.1 (also 12.8), driver 595 (CUDA 13.2 runtime), cuBLAS/cuBLASLt 13.2, CUB/CCCL,
  cuda_fp4/fp6/fp8 headers, cmake/ninja/gcc/rustc. **We can install newer CUDA (13.2/13.3+), CUTLASS,
  any Rust/C++ deps as research dictates.** CUTLASS not yet fetched.
- **MMQ dual-toolkit build (verified on-device):** system gcc is 15 (too new for CUDA 12.8 nvcc;
  gcc-15 headers break 12.8's compiler — `-allow-unsupported-compiler` passes the gate but still
  errors on `<type_traits>`). FIX: **CUDA 12.8 nvcc + `-ccbin /usr/bin/gcc-14`** produces valid
  sm_120 AND sm_120a cubins (gcc-14 is installed, within 12.8's supported range). So the dual-toolkit
  plan works: 13.1 nvcc for FP4/cuBLASLt TUs, 12.8+gcc-14 for MMQ TUs (which segfault under 13.1's -O3).
- **No torch; Python 3.14.** Can install whatever runtime we choose — not a constraint on stack choice.
- **Free host RAM ~12-16 GB right now** (other LLM servers running) — TEMPORARY. Could be more/less.
  → spilling must query free RAM at runtime and size the host tier dynamically; never hardcode it;
  fall back to mmap'd disk when tight. (This is a *design requirement from variability*, not a fixed budget.)

### Toolchain-version-specific gotcha (current nvcc 13.1)

FP4 block-scale MMA compiles ONLY with `-gencode arch=compute_120a,code=sm_120a`.
The `-arch=sm_120a` shortcut silently misroutes through a `compute_120` (no `a`) PTX intermediate in
the full compile pipeline → ptxas rejects `.block_scale`/`.kind::mxf4`/`.scale_vec::2X`.
(May differ in a newer nvcc — re-check after any toolchain upgrade. Not a hardware limit.)
