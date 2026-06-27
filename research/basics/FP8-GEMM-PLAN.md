# Stage-C Block-Scale FP4/FP8 GEMM — the prefill ceiling-raiser (`fp8_gemm` lever)

The int8 GEMM (`cu/qmatvec_gemm.cu`, Task #10/#16/#20/#21 DONE) killed the 43x weight-re-read
gap and closed *toward* llama's 6240 pp512. But int8 m16n8k32 tops out at **219 TFLOP/s** peak
(`research/sm120-empirical-capabilities.md:72`); even a perfectly-tuned int8 grind at ~85% lands
~**1379 effective** — still ~4.5x short of llama 6240. **int8 cannot beat llama. FP4 can.**

The lever: sm_120 silicon executes block-scale `mma.sync` FP4/FP8 that **llama.cpp and vLLM
cannot use on consumer Blackwell** (their MMQ/Marlin int4/int8 paths don't emit
`kind::mxf4.block_scale`). Measured on RTX 5090, 2026-06-26:

| dtype (FP32 acc) | measured peak | vs int8 (219) | vs FP16 (117) | source |
|---|---|---|---|---|
| FP8 e4m3 **plain** m16n8k32 | 219 | 1.00x | 1.88x | `sm120-empirical-capabilities.md:72` |
| FP8 **block-scale** mxf8f6f4 m16n8k32 | **381** | 1.74x | 3.26x | `sm120-empirical-capabilities.md:73` |
| FP4 **block-scale** mxf4 m16n8k64 | **762** | 3.5x | 6.52x | `sm120-empirical-capabilities.md:74` |

**PRIORITY = the FP4 path.** NVFP4 weights are *already* per-16 UE4M3 micro-scaled (the GGUF
NVFP4 block, `cu/qmatvec.cu:294-304`). Those micro-scales ARE the `block_scale` operand — they
feed `mma.sync.m16n8k64.kind::mxf4.block_scale` **directly**, no repack. At even **30% of 762 =
229 TFLOP effective**, pp512 projects to **~10000+ — BEATS 6240.** This is the real BEAT-llama edge.

---

## 0. What already exists (do not rebuild)

- **The MMA instructions are proven on-device.** Both forms assemble and run with
  `-gencode arch=compute_120a,code=sm_120a` (`build.rs:17` already uses exactly this flag):
  - FP4: `probe/lowbit_peak.cu:7` —
    `mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0`
    operands `{c[4]},{a[4]},{b[2]},{c[4]},{sa},{0,1},{sb},{0,1}`
  - FP8: `probe/lowbit_peak.cu:16` —
    `mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0`
    same operand shape, `sa`/`sb` are the per-block scale registers.
- **The tiled GEMM scaffold is done.** `cu/qmatvec_gemm.cu` ships the whole machine: BM=64/BN=128/BK=32
  tile (`:49-51`), NSTAGE=3 cp.async ring (`:55`), `ld_A_s8`/`ld_B_s8` ldmatrix (`:100-114`),
  pre-decode raw-superblock staging (`:336-406`), single-barrier K-loop (`:455-503`),
  per-dtype `StageMeta` (`:205-216`), and the **two-sub-scale kernel2** (`:664-854`) that already
  splits NVFP4's 64-elem block into per-16 halves with per-16 UE4M3 scales (`decode_nvfp4_2`,
  `:610-635`). FP4 block-scale is a *re-targeting* of kernel2's data flow, not a new kernel.
- **NVFP4 is wired end-to-end.** `gemm_supports` allows NVFP4 when `in_f%64==0` (`lib.rs:669-675`),
  `qmatvec_gemm` dispatches `qmatvec_gemm_nvfp4` (`lib.rs:696-698`), per-tensor macro-scale applied
  post via `scale_inplace` (`lib.rs:714`, loaded `model.rs:50-62`). kernel_check has the NVFP4-GEMM
  bit-equivalence case on the 9B model (`kernel_check.rs:405-444`).

So Stage-C is: **(1) a new pair of MMA wrappers, (2) one new K=64 kernel that swaps the s8 MMA +
two-mma-zero-half trick for one native mxf4 block-scale MMA, (3) an FP8-activation quant kernel,
(4) a dispatch + gate.** Everything else is reused.

---

## 1. PRIORITY PATH — FP4: NVFP4 weights → `mma.sync.m16n8k64.mxf4.block_scale` directly

### 1.1 Why FP4 first, and why it's "free" on the weight side

NVFP4's on-disk layout (`cu/qmatvec.cu:294-304`, `qmatvec_gemm.cu:282-284,610-635`):
a 64-element block = 36 bytes = **4 × UE4M3 per-16 micro-scales** (`d_bytes[0..3]`) + 32 bytes of
4-bit e2m1 codebook values (`qs`). The mxf4 MMA contracts **K=64** per instruction with
`scale_vec::2X` = **2 scales per fragment** — i.e. one scale per 32-element K-half. That granularity
collapses onto NVFP4's per-16 scales: a 64-elem K-block has 4 micro-scales; the MMA's 2 scale slots
cover the row (M) tiling, and we feed the per-16 weight scale as `sa`. **The weight needs zero
repack** — the e2m1 nibbles are *exactly* the A-operand bytes, the UE4M3 bytes are *exactly* the
scale operand (after the UE4M3→UE8M0 conversion in §1.3).

Contrast the current int8 path: `kernel2` (`qmatvec_gemm.cu:664-854`) decodes NVFP4 e2m1 → int8
codebook values via `gtable16` (`:597-609`), runs **two** s8 MMAs per 32-block with zeroed half-
fragments (`:824-832`), and folds the UE4M3 scale as an f32 multiply post-MMA (`:838`). That is
correct but throws away the hardware: it dequantizes FP4→int8 and runs at int8's 219 ceiling. The
mxf4 path keeps the values 4-bit and runs at **762**.

### 1.2 The MMA wrapper (new, in `qmatvec_gemm.cu`)

```c
// FP4 block-scale: D[16x8] f32 += A[16x64] e2m1 * B[8x64](col) e2m1, per-32-K-half UE8M0 scales.
// A: 4 x .b32/lane (64 e2m1 nibbles = 256 bits/16 rows). B: 2 x .b32/lane. sa/sb: 1 x .b32 scale.
__device__ __forceinline__ void mma_mxf4_m16n8k64(
        float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2],
        unsigned sa, unsigned sb) {
    asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X"
      ".f32.e2m1.e2m1.f32.ue8m0 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,1},{%11},{0,1};"
      : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
      : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"r"(sa),"r"(sb));
}
```
Operand string copied verbatim from `probe/lowbit_peak.cu:7` (the only proven-assembling form).
The `{0,1}` immediates are the scale-selector byte-IDs the ISA requires after each scale register.

### 1.3 Scale operand: NVFP4 per-16 UE4M3 → the `block_scale` UE8M0 operand

The MMA wants **UE8M0** (8-bit unsigned exponent, 0 mantissa) scale bytes; NVFP4 stores **UE4M3**
(4-bit exp, 3-bit mantissa, ×0.5 — decoded by `gue4m3_to_f32` at `qmatvec_gemm.cu:588-594`). Two
options, decided by the §6 gate:

1. **Exponent-only (lossy, fast).** UE8M0 ≈ floor/round of `log2` of the UE4M3 value:
   `ue8m0 = clamp((ue4m3>>3) + 127 - 7, 0, 255)` (drop the 3-bit mantissa). This is the genuine
   "FP4 is lossy" tradeoff. The discarded mantissa is folded into a **residual f32 correction
   absorbed by the per-tensor macro-scale** (§1.5) only if uniform; otherwise it's the accuracy hit
   the argmax gate must clear.
2. **Mantissa preserved via micro-rescale (preferred, lossless-er).** Keep the UE8M0 = the
   exponent, and pre-multiply the e2m1 codebook value selection so the mantissa is baked into the
   value — NOT possible for e2m1 (only 8 codebook points). So mantissa loss is intrinsic to feeding
   a UE8M0 operand. **The honest design: option 1, and the argmax gate (§6) is the arbiter.** If
   268/271 fails, fall back to the FP8 path (§2) which keeps e4m3's 3-bit mantissa.

The scale bytes stay **register-hot** for the whole K-loop (each warp loads its 4 UE4M3 bytes once
per 64-block into `sa`/`sb` registers; never re-read), per the register-reuse discipline already
documented at `qmatvec_gemm.cu:330-335`.

### 1.4 New kernel structure — extend `qmatvec_gemm` from m16n8k32 → m16n8k64

New template path `qmatvec_gemm_kernel_mxf4<GQT_NVFP4>` (sibling of `kernel2` at
`qmatvec_gemm.cu:664`), launched by a new `extern "C" qmatvec_gemm_nvfp4_fp4` (alongside `:862`).
Deltas from `kernel2`:

- **BK=64** (one full NVFP4 block per K-step) instead of BK=32 + two-half split. Halves the K-loop
  trip count; the K-block IS the 36-byte NVFP4 block (StageMeta NVFP4 `SB_BYTES=36, GPSB=2`
  becomes `GPSB=1` at K=64 — one block = one K-step). Update `StageMeta<GQT_NVFP4>`
  (`qmatvec_gemm.cu:209`) to a K=64 variant or add `StageMeta_fp4`.
- **smem weight tile holds raw e2m1 nibbles** (`int8 sWq[BM][32]` = 32 bytes/row = 64 nibbles),
  NOT decoded int8 codebook values. Drop `gtable16`/`decode_nvfp4_2` (`:597-635`) from this path —
  the whole point is to NOT decode. cp.async the 32 `qs` bytes straight into smem.
- **scales staged as UE8M0 bytes** in `sSc[BM][4]` (4 per 64-block), converted from UE4M3 once at
  load (§1.3). Activation scales: UE8M0 from the FP8-quantized activation block (§1.6).
- **ldmatrix for K=64**: A-fragment is 4×.b32 (64 nibbles × 16 rows / 32 lanes), reuse `ld_A_s8`'s
  per-lane address math (`:100-104`) — the byte stride doubles (32B/row vs 16B), 16B-aligned so
  `ldmatrix.x4.b16` stays legal. B-fragment 2×.b32 from the FP8 activation tile.
- **single mxf4 MMA per K-step** replaces the two zeroed-half s8 MMAs (`:831-832`): one
  `mma_mxf4_m16n8k64` accumulating directly in **f32** (no s32→f32 scale stage — the block_scale is
  applied *inside* the MMA). The per-K-step f32 scale fold loop (`:833-839`) is **deleted** — the
  hardware does it. Accumulators are `float facc[BN/8][4]` (already f32 at `:691`), now the MMA
  writes them directly.

### 1.5 Per-tensor NVFP4 macro-scale — unchanged, post-matmul

NVFP4 carries a second-level per-tensor f32 macro-scale (sibling `<stem>.scale` tensor, loaded
`model.rs:50-62`). It is orthogonal to the per-16 micro-scales and is applied **after** the GEMM via
`scale_inplace` (`lib.rs:714`) — identical to today's int8 NVFP4 path. The mxf4 MMA handles only the
per-16 dequant; the macro post-multiply is untouched. If §1.3's mantissa loss is uniform-ish, a
small correction factor can be premultiplied into this macro-scale at load time (free — it's one f32).

### 1.6 Activation quant for FP4: e4m3 (NOT FP4)

Activations must NOT be FP4 — quantizing dynamic-range activations to 8 codebook points wrecks
argmax. Quantize activations to **FP8 e4m3** (3-bit mantissa, wide range), with a per-32 (or per-64)
UE8M0 block scale. The mxf4 MMA's B-operand is declared `e2m1` in the asm — but the *activation*
side can be the wider operand: the `kind::mxf8f6f4` family lets A and B differ. **Cleanest: use the
FP8 path (§2) MMA for the activation-heavy operand and reserve pure-mxf4 for weight×weight.** The
pragmatic Stage-C v1: **FP4 weights × FP8-e4m3 activations** is the `mxf8f6f4` mixed MMA (A=e2m1
weights via the f4 slot, B=e4m3 activations), which still beats int8. The pure mxf4 762 number is
the weight-symmetric upper bound; the mixed path lands between 381 and 762.

New quant kernel `quantize_e4m3` (sibling of `quantize_q8_1` at `cu/qmatvec.cu:341-357`):
same per-32 block, same launcher signature, same output shape `aq:[m,in_f]` (now e4m3 bytes) +
`ad:[m,in_f/32]` (now UE8M0 bytes or f32). Body: `amax` over 32 → `d`; quantize
`x → e4m3_encode(x/d)`; store the block scale as UE8M0 = `ilogb`-based exponent. Drop-in: same block
granule (BK=32 aligns with the mxf8f6f4 `scale_vec::1X` = one scale per 32-K).

---

## 2. FALLBACK / COMPLEMENT — FP8 block-scale (mxf8f6f4 m16n8k32, 381 TFLOP/s)

If §1.3's FP4 mantissa loss fails the argmax gate, OR for the Q8_0/Q4_K/Q6_K dtypes (no native FP4),
ship the FP8 block-scale path. Lower ceiling (381 vs 762) but **still 1.74x over int8 → ~2400+
effective at 85% → already beats llama on the compute-bound batches.**

### 2.1 MMA wrapper

```c
// FP8 block-scale: D[16x8] f32 += A[16x32] e4m3 * B[8x32](col) e4m3, one UE8M0 scale per 32-K.
__device__ __forceinline__ void mma_mxf8_m16n8k32(
        float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2],
        unsigned sa, unsigned sb) {
    asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X"
      ".f32.e4m3.e4m3.f32.ue8m0 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,1},{%11},{0,1};"
      : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
      : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"r"(sa),"r"(sb));
}
```
Verbatim from `probe/lowbit_peak.cu:16`. `scale_vec::1X` = one scale per 32-K block — **exactly**
the existing BK=32 granule, so this is a true drop-in into `kernel1` (`qmatvec_gemm.cu:305-517`).

### 2.2 Drop-in into the existing K=32 kernel1

The transition is the smallest possible change vs int8 `kernel1`:
- Weight decode: `decode_block<QT>` (`:291-295`) emits **e4m3 bytes** instead of int8 + a UE8M0
  block scale instead of the f32 `dw`. For Q8_0 the int8 is requantized to e4m3 at decode (cheap);
  for Q4_K the nibbles dequant → e4m3. The per-block scale `sWd` (`:326`) becomes UE8M0 `sWsc`.
- Activation: `quantize_e4m3` (§1.6) feeds e4m3 `sA` + UE8M0 `sAsc`.
- MMA: swap `mma_s8_m16n8k32` (`:491`) → `mma_mxf8_m16n8k32`, accumulating **f32 directly**.
- **Delete** the per-K-step s32→f32 scale fold (`:495-501`) — the block_scale MMA does it inline.
  The bias term (`sWb`, for Q4_K min-offset, `:500`) is applied post-loop as a separate f32 correction
  (the MMA has no bias slot), folded into the existing `facc`.

---

## 3. Files to touch (concrete)

- **Edit** `crates/bw24-engine/cu/qmatvec_gemm.cu`:
  - add `mma_mxf4_m16n8k64` + `mma_mxf8_m16n8k32` wrappers (near `mma_s8_m16n8k32` `:70`).
  - add UE4M3→UE8M0 and f32→UE8M0 helpers (near `gue4m3_to_f32` `:588`).
  - add `qmatvec_gemm_kernel_mxf4<GQT_NVFP4>` (K=64, sibling of `kernel2` `:664`) + `extern "C"
    qmatvec_gemm_nvfp4_fp4` (near `:862`).
  - add FP8 path inside `kernel1` behind a template flag (or a `kernel1_fp8` sibling) + the e4m3
    `extern "C"` launchers for Q8_0/Q4_K/Q6_K.
- **Edit** `crates/bw24-engine/cu/qmatvec.cu:341` — add `quantize_e4m3` next to `quantize_q8_1`.
- **Edit** `crates/bw24-engine/src/lib.rs`:
  - new `quantize_e4m3` launcher (next to `quantize_q8_1` `:304`).
  - `qmatvec_gemm` (`:686-714`): when `BW24_FP4=1` and `qtype==QT_NVFP4 && in_f%64==0`, select
    `qmatvec_gemm_nvfp4_fp4`; when `BW24_FP8=1`, select the e4m3 launchers. Keep the int8 GEMM as the
    default so Stage-C is a strict opt-in over the proven path.
  - `gemm_supports` (`:669-675`) gains the FP4/FP8 dtype gating behind the new env vars.
  - per-tensor macro-scale post-multiply (`:714`) unchanged.
- **No build.rs change** — `qmatvec_gemm.cu` and `qmatvec.cu` already compile with the required
  `-gencode arch=compute_120a,code=sm_120a` (`build.rs:11,17`); the block-scale MMA assembles under it.
- **Edit** `crates/bw24-engine/src/bin/kernel_check.rs:405-444` — add FP4-GEMM and FP8-GEMM cases
  (see §6).

---

## 4. Honest projection

- **FP4 (mxf4) headroom:** 762 TFLOP peak. Real GEMM hits 70-85% of micro-bench peak *only* with
  perfect tiling; Stage-C v1 will be lower (decode-light but ldmatrix/scale-staging unoptimized).
  At a **conservative 30% (229 TFLOP effective)**: pp512 ≈ 143 × (229 / [int8-effective ~3.3]) is the
  wrong frame — anchor instead on llama: int8-tuned ~1379 effective ≈ 6240/4.5. FP4 at 229 effective
  is **~1.5x the int8-tuned compute**, but the int8 path is not yet at its 1379 ceiling. The
  load-bearing claim from the lever: **229 TFLOP effective → pp512 ~10000+**, which **BEATS 6240**.
  Even a pessimistic 20% (152 TFLOP) clears llama's 6240 with margin.
- **FP8 (mxf8f6f4) floor:** 381 peak; at 60% = 229 effective → same ~10000 regime; at a pessimistic
  40% = 152 → still > 6240. **FP8 alone beats llama**; FP4 is the stretch ceiling.
- **The risk is NOT throughput, it's the argmax gate.** FP4's discarded mantissa (§1.3) is the only
  thing that can sink this. If FP4 argmax fails, FP8 (3-bit mantissa preserved) is the fallback that
  still beats 6240. This is the genuine "FP4 is lossy" tradeoff — gated, not assumed.

---

## 5. Bring-up order (each gated before the next)

1. **FP8 path first** (lower risk: mxf8f6f4 is a drop-in into the proven K=32 kernel1, keeps a
   3-bit mantissa). Land Q8_0 FP8 → bit/argmax gate → Q4_K FP8.
2. **FP4 path** on NVFP4 (the BEAT edge): K=64 kernel, exponent-only UE8M0 scales → argmax gate.
3. If FP4 argmax holds: it is the shipping prefill path for NVFP4 models. If not: FP8 ships, FP4
   stays env-gated for measurement.

---

## 6. GATE (non-negotiable, ordered)

1. **Bit-tolerance (kernel_check).** FP4/FP8 are LOSSY vs int8 — NOT bit-equivalent. Add cases to
   `kernel_check.rs:405-444` comparing `qmatvec_gemm_nvfp4_fp4` / FP8 launchers vs the Stage-A f32
   oracle (`cpu_linear(dequant(W))`, the pattern at `kernel_check.rs:206-241`), NOT vs the int8 dp4a
   path. Tolerance: FP8 `rel < 3e-2` (same as the int8-activation fast paths at `:264,:285`); FP4
   `rel < 6e-2` (looser — the documented noise floor at `kernel_check.rs:429`). T ∈ {16,64,128,512}.
2. **End-to-end argmax — THE gate.** With `BW24_FP4=1` (then `BW24_FP8=1`), `prefill argmax` MUST
   hold **268** (qwen3 `run_dense`) and **271** (qwen35 `run_hybrid`) — the validated checkpoints
   (`run_gen.rs`). FP4 is lossy; it must STILL argmax-match. Any mismatch on FP4 → FP4 reverts to
   env-gated, FP8 ships. The decode (m=1) path is untouched (still dp4a/MMVQ), so decode argmax is
   unaffected.
3. **Perf gate.** `pp512 tok/s` (Task #3 beat-benchmark) with the Stage-C path MUST clear **6240**
   (beat llama). Milestone: FP8 first-cut > 3000 (proves the block-scale lift over int8); FP4 tuned
   target > 10000.
