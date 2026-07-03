# Is Parity the Ceiling for the 27B-Q4_K Prefill? — Read-Only Adversarial Re-Investigation

**Date:** 2026-06-28 · **Scope:** READ-ONLY analysis. No kernel edits, no GPU, no touching `qmatvec_gemm.cu`
(another agent is mid-edit + running ncu on it). Pure source reading + cross-citation.

**Question (binding-goal version):** Is parity-with-llama truly the ceiling for the 27B-Q4_K_M model's
prefill on this RTX 5090 laptop (sm_120), or is there a route to *beat* (not match) llama on the Q4_K
model too? And on decode?

---

## TL;DR — the verdict, sharpened by ground-truth evidence

The adversarial verdict that "kernel1 (Q4_K int8 W4A8) is capped at PARITY, and the only beat is the FP4
path" is **directionally right on the int8 ceiling but rests on two factual errors** that change the
strategy:

1. **FACTUAL ERROR #1 (the big one): llama DOES run the native FP4 762-TFLOP block-scale MMA on this laptop.**
   The memory plan repeatedly asserts "their FP4 = SM100 tcgen05/TMA → they fall to Marlin FP16, so the
   mxf4 path is bw24's unmatchable edge" (`prefill-gemm-rebuild-plan.md:13`, `no-magic-match-then-exceed.md:10`).
   **This is false.** llama's NVFP4/MXFP4 MMA is `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64`
   (`/data/projects/llama.cpp/ggml/src/ggml-cuda/mma.cuh:1145`) — a *warp-level* `mma.sync`, NOT wgmma/tcgen05 —
   gated on `BLACKWELL_MMA_AVAILABLE`, which is defined for `__CUDA_ARCH__ >= 1200` i.e. **sm_120**
   (`common.cuh:286-288`). It is dispatched whenever `blackwell_mma_available(cc)` is true on the device
   (`mmq.cu:125`, `mmq.cuh:3334`, `vec_dot_fp4_fp4_mma` at `mmq.cuh:996-1064`). **So FP4 is NOT a free
   bw24-only edge — llama already runs the exact same 762-TFLOP instruction.** The FP4 path is still where the
   *compute headroom* is, but beating llama there means out-*implementing* their FP4 MMQ, not running an
   instruction they can't.

2. **FACTUAL ERROR #2 (the nuance): neither "the 27B" nor "the 9B" is single-quant.** Tensor-type histograms
   read straight from the gguf headers:
   - **27B-Q4_K_M** (`Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf`): **192 NVFP4 + 273 Q4_K + 33 Q6_K** weight tensors
     (+737 F32 norms/biases).
   - **9B-NVFP4** (`Qwen3.5-9B-NVFP4-MTP-GGUF.gguf`): **114 NVFP4 + 50 Q8_0 + 44 Q5_K + 43 Q4_K + 1 Q6_K**.

   (Type-id 40 = `GGML_TYPE_NVFP4`, confirmed `ggml/include/ggml.h:430`.) So **the FP4-vs-int8 split exists
   *inside each model's* prefill.** The 27B is NOT "stuck entirely on int8" — its 192 NVFP4 tensors can hit the
   762-TFLOP path; only its 273 Q4_K + 33 Q6_K tensors are on the 219-TFLOP int8 ceiling. And the 9B is NOT
   "pure NVFP4" — 137 of its tensors are k-quant/Q8_0 on the int8 path.

**Bottom line on the question:**
- For the **Q4_K and Q6_K tensors specifically** (the 306-of-498 majority of the 27B's weight tensors):
  **yes, the int8 W4A8 m16n8k32 path is a hard 219-TFLOP compute ceiling, and llama runs the identical
  instruction at that same ceiling.** There is no requant trick that beats it without accuracy loss. So on
  *raw compute*, the Q4_K tensors top out at parity.
- **BUT parity-on-the-instruction is NOT parity-on-the-kernel.** llama wins its Q4_K prefill (the 581-vs-llama
  gap) not via a faster instruction but via **stream-k work partitioning + a fixup pass + a persistent
  nsm-block grid + a per-call-autotuned token tile** — three structural levers bw24 has NOT copied. Because
  bw24 compiles for *exactly* 82 SMs / 100 KB / one config, a laptop-specialized stream-k can in principle
  beat llama's *general* stream-k. **That is the genuine route to beat llama on the Q4_K tensors: not a better
  MMA, but a better scheduler around the same MMA, hand-fit to 82 SMs.**
- The **single largest beat lever for the 27B-Q4_K_M as a whole is still the FP4 path** — but applied to its
  **192 NVFP4 tensors**, where bw24's hand-rolled block-scale GEMM (831 measured) must be brought up to
  llama's FP4-MMQ efficiency *and past it* via laptop-exact tuning. This is the same conclusion the verdict
  reached, but for a *different reason* (llama is a real FP4 competitor here, not absent) and with a
  *different scope* (the 192 NVFP4 tensors of the 27B, not "switch to a 9B-NVFP4 model").

So: **the daily-model beat does NOT require switching to a different (pure-NVFP4) model.** The 27B-Q4_K_M can
beat llama if (a) its NVFP4 tensors win on the FP4 path via 82-SM-exact tuning, and (b) its Q4_K/Q6_K tensors
win on the int8 path via the stream-k/fixup/persistent-grid structure llama uses and bw24 lacks. Both are
"copy-then-tune-to-the-exact-config", on-thesis with `win-is-copy-then-tune-exact-case` and
`no-magic-match-then-exceed`.

---

## 1. Can the Q4_K weights be requantized/upcast to a higher-TFLOP MMA? (cost/benefit)

**Mechanically possible, but it does not beat int8 for the Q4_K tensors. Here is why, rigorously.**

### 1a. Q4_K → NVFP4 on-the-fly (target the 762-TFLOP block-scale MMA)

- The 762-TFLOP `mxf4nvf4.block_scale` MMA consumes **e2m1 (FP4)** operands with per-16 **e4m3** block scales
  (`mma.cuh:1145`; bw24's mirror `mma_mxf4_m16n8k64` at `qmatvec_gemm.cu:1097`). To feed Q4_K weights into it
  you must transcode each Q4_K weight from its native representation (a 6-bit super-scale + 6-bit min per
  32-sub-block, 4-bit quants) into e2m1 nibbles + e4m3 per-16 scales.
- **The accuracy cost is real and one-directional.** Q4_K's value grid is `d*(q - m)` with `q∈[0,15]` (16
  *uniform* levels per sub-block, offset by a learned min `m`). NVFP4's grid is `scale * e2m1` where e2m1 has
  **only 16 levels but they are non-uniform** (`{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}`) and **has no min/offset
  term**. Transcoding Q4_K→NVFP4 is a *re-quantization onto a different, offset-free grid*, so it loses the
  Q4_K min term and re-rounds onto the non-uniform e2m1 levels. This is strictly lossier than keeping Q4_K
  on its native int8-expanded path (where `decode_q4_k` reproduces the exact `d*(q-m)` integer values
  bit-for-bit, `qmatvec_gemm.cu:224`/`:343`). The whole reason these tensors were shipped as Q4_K and not
  NVFP4 by the model author is that Q4_K's offset grid fits their distribution better; forcing them onto
  e2m1 throws that away. **Expected: argmax drift; would fail the `kernel_check rel<1e-3` + argmax-MATCH
  gate** that every bw24 prefill lever must clear (`autonomous-work-state-and-direction.md:62`).
- Even ignoring accuracy: the transcode itself is not free. Done at load time it doubles resident weight VRAM
  for those tensors (the same VRAM concern the `STEP 2` note flags, `prefill-gemm-rebuild-plan.md:114`); done
  on-the-fly per call it adds a re-quant pass that, at the 273-Q4_K-tensor count, competes with the matmul it's
  trying to accelerate.

**Verdict on 1a: NOT a beat. It trades a guaranteed accuracy regression (argmax-gate failure) for a compute
ceiling lift that, for these tensors, is not realizable without losing correctness. Reject.**

### 1b. Q4_K → block-scaled FP8 (the 381-TFLOP `mxf8f6f4` path)

- sm_120 has `mma.sync.m16n8k32.kind::mxf8f6f4.block_scale` at **381 TFLOP** (measured,
  `sm120-empirical-capabilities.md` peaks table) — 1.74× the int8 219. Upcasting Q4_K's 4-bit quants to FP8
  e4m3 is **loss-free in the quant levels** (FP8 e4m3 trivially represents all 16 Q4_K integer levels, and the
  per-sub-block scale/min maps to the block-scale). So unlike 1a, this *can* be argmax-stable.
- **But the byte/AI math kills it for the W4A8 regime.** The 381-TFLOP figure is for FP8×FP8 (both operands
  4-bit-expanded-to-8-or-the-block-scale path). bw24's int8 path is already W4A8 (4-bit weight, int8
  activation) running the 219-TFLOP int8 MMA. To use the 381 path you'd expand the **weight** to FP8 (8-bit) —
  doubling weight smem traffic and weight VRAM — to gain 1.74× compute on a kernel the memory measurements show
  is **smem/L1-bound, not compute-bound** (`prefill-gemm-rebuild-plan.md:59`: "L1/smem throughput 70%,
  issue_active only 28%"). A compute-ceiling lift does nothing for a kernel stalled on smem bandwidth; doubling
  the weight bytes makes the actual bottleneck *worse*. The 381-TFLOP lever pays only once the kernel is
  compute-bound, which the Q4_K kernel is provably not.

**Verdict on 1b: argmax-safe but net-negative on this (smem-bound) kernel — more weight bytes for compute you
can't use. Reject for prefill. (Could be revisited only after the smem-bound is fixed, see §2.)**

### 1c. Conclusion for the question's part (a)

For the Q4_K/Q6_K tensors, **there is no requant/upcast that beats the native int8 W4A8 path** — every higher-
TFLOP target either fails the accuracy gate (FP4) or worsens the actual smem bottleneck (FP8). The int8
219-TFLOP MMA is the right primitive for these tensors. **The ceiling on the *instruction* is real and is
parity with llama (who runs the same instruction).**

---

## 2. What does llama's Q4_K MMQ do that bw24 (post Step 1) still doesn't?

llama uses MMQ for Q4_K prefill on sm_120 (not cuBLAS): `ggml_cuda_should_use_mmq` returns `true` for Q4_K
whenever `turing_mma_available(cc)` (`mmq.cuh:307-308`), which is true for sm_120 at *any* batch including
pp512. So the head-to-head is bw24 `qmatvec_gemm_q4_K` vs llama `mul_mat_q<GGML_TYPE_Q4_K>`.

### 2a. The instruction and the per-K register tile are ALREADY matched

- **Same MMA.** llama Q4_K decodes to the q8_1 int8 layout (`MMQ_MMA_TILE_X_K_Q8_1`, `mmq.cuh:254`) and runs
  `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` (`mma.cuh:946`). bw24's `mma_s8_m16n8k32`
  (`qmatvec_gemm.cu:138-140`) is byte-identical. No gap here.
- **A-fragment register residence across the token tile is ALREADY matched.** The earlier plan framing of "the
  #49 register-K-pipeline rewrite" as the missing lever is partly obsolete: llama's `vec_dot_q8_1_q8_1_mma`
  loads `tile_A A[ntx][MMQ_TILE_NE_K/QI8_1]` once per K-step then reuses it across the `mmq_x` token loop
  (`mmq.cuh:1399, 1426-1452`). bw24 *already does this* — `afrag[MFRAG][4]` is loaded once per K-step
  (`qmatvec_gemm.cu:625-628`) and reused across all 8 n-tiles (`:644-660`). **Both reload A per K-step from
  smem; both hold it register-resident across the token inner loop. This is NOT the gap.** (The fleet already
  proved the #49 A-load double-buffer regresses — `autonomous-work-state-and-direction.md:16`. Consistent.)
- **The B (activation) load trick is matched in spirit.** llama loads B with `load_generic` and comments
  "faster than load_ldmatrix" (`mmq.cuh:1433`); bw24 uses `ld_B_s8` ldmatrix.x2 (`qmatvec_gemm.cu:196-199,647`).
  Minor; not the dominant gap.

### 2b. The THREE structural levers bw24 has NOT copied (these ARE the gap)

These are the parts of llama's `mul_mat_q` that have no bw24 analogue, and they are *exactly* the kind of thing
an 82-SM-specialized kernel can beat the general version on:

**(i) Stream-K work partitioning** (`mmq.cuh:3528-3783`, the `mul_mat_q` kernel). Instead of a 2-D tile grid
(one CTA per output tile, bw24's model), llama launches a **persistent grid of `nsm` blocks** (one per SM when
tiles don't evenly fill the machine — `mmq.cuh:4001`: `block_nums_stream_k = ... ? ntiles_dst : nsm`). Each
block walks a *continuous* slice of the (tile × K-block) iteration space (`kbc`/`kbc_stop`, `mmq.cuh:3642-3650`),
so the **K dimension is split across blocks** and partial output tiles are reduced afterward. This eliminates
the "tail effect" / wave quantization where bw24's last partial wave of CTAs leaves most of the 82 SMs idle.
For the 27B's tall-skinny attention projections and the FFN shapes, the tile count rarely divides 82 evenly, so
bw24's plain tiling wastes a fractional wave on every layer. **bw24 has no stream-k.**

**(ii) The fixup reduction pass** (`mul_mat_q_stream_k_fixup`, `mmq.cuh:3787-3922`). Because stream-k splits a
tile's K across blocks, the last block to touch a tile writes its partial to a `tmp_fixup` buffer
(`mmq.cuh:3520-3521, 3779`), and a tiny second kernel sums the partials into `dst` (`:3920`). This is the
machinery that *makes* stream-k correct. **bw24 has no fixup pass** (it can't — it has no stream-k).

**(iii) Per-call token-tile autotune** (`mul_mat_q_case`, `mmq.cuh:4069-4082`). llama sweeps `mmq_x ∈ {8,16,…,128}`
*per matmul call* and picks the value that minimizes the number of token-tiles `ntiles_x` for that exact shape,
subject to the smem budget (`smpbo`). So a 512-token prefill of a 4096-wide FFN gets a different token tile
than a 512-token attention projection. **bw24 is fixed at `K1_BN=128`** (`qmatvec_gemm.cu:101`) for every Q4_K
shape — it cannot adapt the tile to the layer.

### 2c. Is llama's Q4_K actually at 40% SM, and can bw24 pass it via 82-SM specialization?

- The memory's ncu head-to-head (`prefill-gemm-rebuild-plan.md:84-90`) measured **llama `mul_mat_q` at 40.9%
  SM throughput, 255 regs/thread, 1 CTA/SM, 16.6% occupancy** vs bw24 Q4_K at 22% SM. The 40% is the *general*
  kernel's number on this GPU — with its general stream-k tuned for "all NVIDIA GPUs," and a heuristic
  (`tiles_efficiency_percent >= 90`, `mmq.cuh:4001`) that decides persistent-vs-tiled based on a generic 90%
  threshold, not on 82 SMs specifically.
- **The 82-SM-exact specialization headroom (above parity), concretely:**
  - **Compile-time `nsm = 82`.** llama reads `nsm` at runtime and computes `tiles_nwaves`/`tiles_efficiency`
    with runtime divides (`mmq.cuh:3999-4001`). bw24 can bake `82` as a `constexpr`, letting the compiler fold
    the wave/efficiency math and the kbc/kbc_stop fast-divides (`mmq.cuh:3642`) to constants — removing the
    runtime `init_fastdiv_values` indirection on every launch (`mmq.cuh:3968-3973`). **Est. +1-3%** (launch +
    integer-divide overhead removal; matters because prefill issues ~339 GEMM launches, the launch overhead the
    memory flagged in `autonomous-work-state-and-direction.md:132`).
  - **Compile-time single tile shape.** bw24 already knows the daily 27B's per-layer shapes at build time. It
    can pick the optimal `mmq_x` per *named* matmul (FFN-down vs QKV vs gate) as a `constexpr` template arg and
    drop llama's per-call autotune sweep *and* its branchy `need_check` bounds logic
    (`mmq.cuh:3976-3992` — the `nrows_x % mmq_y` runtime branch). A `need_check=false` specialization for shapes
    known to divide evenly removes a per-tile bounds compare in the hot loop. **Est. +2-5%** (the same class of
    win as the "tiny-out_f→dp4a" trace-found fix, `prefill-gemm-rebuild-plan.md:49-50`).
  - **Stream-K tuned to 82 SMs, not the 90% generic threshold.** llama's persistent-vs-tiled decision is a
    single generic constant. bw24 can tune the split point and the per-block K-slice granularity to 82 SMs +
    the actual per-layer tile counts, choosing the partition that leaves *zero* idle SMs for the daily shapes
    (vs llama accepting up to ~10% wave inefficiency). This is the largest of the three. **Est. +5-12% on the
    Q4_K/Q6_K tensors** (closing a meaningful slice of the 22%→40% SM gap by killing the wave-quantization
    bw24 currently eats with plain tiling, which is *why* bw24 sits at 22% SM — under-filled waves, not a slow
    inner loop).

**So: parity-on-the-instruction, but a credible +8-20% *above* llama on the Q4_K/Q6_K tensors is available from
stream-k + fixup + compile-time 82-SM/single-shape specialization** — levers llama structurally cannot match
because it must stay general. This is the honest "match-then-exceed" for these tensors. It is NOT a guaranteed
beat (the inner MMA is at parity, so the win is purely scheduling efficiency), but it is real headroom and it is
on-thesis. **It is also the hardest remaining prefill work** (stream-k + fixup is a substantial, multi-step,
probe-and-gate rewrite), which is why it has not been done — not because it's impossible.

---

## 3. Decode: any route to beat llama on the 27B besides batched-MTP?

### 3a. The k-quant dequant ALU is already at parity (this lever is spent)

The question asks whether bw24's q4_K/q6_K dequant instruction count is still above llama's `mul_mat_vec_q`.
**It is not — it was, and it was fixed.** Side-by-side:
- **q6_K:** bw24 `qmatvec_q6_K_mmvq` (`qmatvec.cu:722-780`) reads 4 ql + 4 qh bytes as `get_int_b2` 32-bit
  words and extracts with SIMD masks + `__vsubss4(vpack, 0x20202020)` + `dp4a`. llama
  `vec_dot_q6_K_q8_1_impl_mmvq` (`vecdotq.cuh:624-644`) does the identical `(vl>>4i)&0x0F0F0F0F` |
  `((vh>>4i)<<4)&0x30303030`, `__vsubss4(...,0x20202020)`, `dp4a`. **Instruction-for-instruction equivalent.**
  The memory confirms the on-silicon effect: q6_K DRAM% rose 19%→24-40% after vectorizing
  (`autonomous-work-state-and-direction.md:92`) — the ALU was freed, throughput is now memory-bound.
- **q4_K:** bw24 `qmatvec_q4_K_mmvq` (`qmatvec.cu:579-585`) does 8× `dp4a(wpack) + dp4a(0x01010101)`. llama
  `vec_dot_q4_K_q8_1_impl_vmmq` (`vecdotq.cuh:512-522`) does the same dot1 (weight·act) + dot2
  (`0x01010101`·act sum) structure. The `dp4a(0x01010101, …)` activation-sum recompute that looks redundant in
  bw24 is **also present in llama's vmmq** (`vecdotq.cuh:518`) — it is not a bw24-specific waste. Parity.
- **Launch structure matched too.** llama's m=1 k-quant decode uses `nwarps=2`, `rows_per_cuda_block=1`
  (`mmvq.cu:430-433, 459`) — i.e. 2 warps/CTA, one output row per warp. bw24's `BW24_MMVQ_ROWS` warp-per-row is
  the same structure.

So the decode dequant-ALU grind the memory was pursuing (`autonomous-work-state-and-direction.md:65-69`) **is
essentially complete; bw24 is at llama's instruction count.** No further ALU lever there.

### 3b. Decode is at the m=1 bandwidth/latency floor — no single-kernel structural beat

Every memory measurement converges: decode matvec is m=1 **latency/bandwidth-bound** (ncu: nvfp4 30-46% DRAM,
q6_K 19→24-40%, q5_K 31% — `autonomous-work-state-and-direction.md:68-69, 126`), the matvec runs at near-parity
(1.06×, `:113`), and the residual gap is non-matvec **glue** that runs serially between matvecs (norm/quant/attn),
most of which is already fused. At m=1, llama and bw24 read the same weight bytes per token; there is no kernel
that beats "read each weight row once and dp4a it" for a single token. **bw24 is at the structural floor here.**

### 3c. The ONLY remaining decode beat is amortization across concurrent tokens — i.e. the batched path

This is the same conclusion the memory reached (`autonomous-work-state-and-direction.md:118-127`). At m=1 there
is no beat; the daily endpoint serving 2-4 concurrent agents = m=2-4 batched decode of the *same* model (legit
single-model, not the rejected multi-model dodge). The batched-MTP work already built (`d1564ce`, partial-accept
replay `abef37d`) is the kernel foundation; the decisive win is the end-to-end batched *path* that runs the
~34%-of-step glue **once per forward instead of once per token**, amortizing it across the m concurrent
decodes. **There is no other route to a decode beat on the 27B.** Specifically:
- It is NOT a faster matvec (at floor).
- It is NOT lower dequant ALU (at parity, §3a).
- It is NOT more KV-quant (llama q8_0-KV ≈ f16-KV, no win to chase — `autonomous-work-state-and-direction.md:82`).
- It IS amortizing the per-forward glue across m>1 concurrent tokens. (This is the batched path the memory has
  scoped as "the big remaining decode win," not yet built end-to-end.)

---

## 4. SYNTHESIS — ranked levers, PARITY → BEAT, for the 27B-Q4_K_M

Each: mechanism · expected % · sm_120-implementability (mma.sync/ldmatrix/cp.async only — no wgmma/tcgen05/TMA)
· probe-gateable.

### PREFILL

| # | Lever | Mechanism | Expected | sm_120 impl | Probe-gate |
|---|-------|-----------|----------|-------------|-----------|
| **P1** | **FP4 path for the 192 NVFP4 tensors** | Bring bw24's hand-rolled `mma_mxf4_m16n8k64` block-scale GEMM (831 measured, occupancy-starved by the per-K repack) up to llama's FP4-MMQ efficiency, then *past* it via 82-SM tuning. This is the **single biggest beat lever for the 27B as a whole** — its 192 NVFP4 tensors are on the 762-TFLOP ceiling, 3.5× the int8 219. **Note: llama is a real competitor here (it runs the same block-scale MMA, §intro #1), so the win is implementation, not instruction.** | **Large** — the 762-vs-219 TFLOP headroom is the only super-parity *compute* lever in the model. Realizing even half of it on the NVFP4 tensors beats llama on that slice. | ✅ `mma.sync.kind::mxf4nvf4.block_scale` is native sm_120 (`mma.cuh:1145`, bw24 `:1097`). cp.async + ldmatrix only. | ✅ — `STEP 2/3` of the plan (kill per-K repack, widen-K) are already probe-scoped (`prefill-gemm-rebuild-plan.md:114-115`); maxrel=0 vs oracle already proven for the block-scale math. Needs a **W4A4 oracle** for the accuracy gate (its `maxdiff=1.0` is vs a W4A8 oracle — apples-to-oranges, `prefill-gemm-rebuild-plan.md:111`). |
| **P2** | **Stream-K + fixup pass on the int8 Q4_K/Q6_K kernel** | Copy llama's `mul_mat_q` stream-k (`mmq.cuh:3528-3783`) + `mul_mat_q_stream_k_fixup` (`:3787-3922`): persistent `nsm`-block grid walking continuous (tile×K) space, partials reduced by the fixup kernel. Kills the wave-quantization that pins bw24 at 22% SM (vs llama 40%) — the tile counts of the 27B's layers don't divide 82 evenly, so plain tiling wastes a fractional wave per layer. | **+5-12%** on the Q4_K/Q6_K tensors (closes a real slice of the 22%→40% SM gap). | ✅ Pure grid/scheduling + a tiny reduction kernel; no exotic instructions. The hard part is correctness (the fixup race-avoidance), not the ISA. | ✅ — bit-exact vs the current tiled kernel (fixup sums must equal the single-tile result); `kernel_check rel<1e-3` + argmax MATCH gates it. |
| **P3** | **Compile-time 82-SM + single-shape specialization** | Bake `nsm=82`, per-named-matmul `mmq_x`, and `need_check=false` as `constexpr` (vs llama's runtime autotune + runtime nsm + bounds branches, `mmq.cuh:3968-3992, 4069-4082`). Folds launch/divide overhead and a hot-loop bounds compare. This is the "general-vs-specialized" edge — llama *can't* do it (must stay general). | **+3-8%** combined (P3 stacks on P2; partly overlaps it). | ✅ Pure compile-time constants + template specialization. | ✅ — bit-exact (same math, fewer runtime branches). |
| P4 | Q4_K → FP8 block-scale (381 TFLOP) | Upcast Q4_K weight to FP8 e4m3, run `mxf8f6f4.block_scale`. | **Negative for now** — argmax-safe but doubles weight smem on an already-smem-bound kernel (§1b). | ✅ native sm_120 | Only after P2/P3 fix the smem bound; revisit then. |
| ✗ | Q4_K → NVFP4 requant | Transcode to e2m1+e4m3 for the 762 path. | **Reject** — loses Q4_K's min/offset grid → argmax-gate failure (§1a). | n/a | Fails the gate by construction. |

### DECODE

| # | Lever | Mechanism | Expected | sm_120 impl | Probe-gate |
|---|-------|-----------|----------|-------------|-----------|
| **D1** | **Batched concurrent-decode path (m=2-4)** | End-to-end batched forward (KV append × m, mixers × m, residual_norm_ffn × m, scheduler gathering 2-4 in-flight requests) so the ~34% per-forward glue runs once per forward, not once per token. The **only** decode beat route — m=1 is at the bandwidth floor (§3b), ALU at parity (§3a). | **Decisive** at the concurrent-endpoint workload (per-token glue amortized m-fold); per-kernel matvec win is modest (+11-14% measured, `prefill-gemm-rebuild-plan.md` / `autonomous-work-state-and-direction.md:122`) but the glue amortization is the throughput win. | ✅ Existing dp4a/MMVQ kernels + batched weight-resident matvec already built (`d1564ce`); no new ISA. Large multi-file *path* build. | ✅ — batched output must match m sequential m=1 decodes (bit-exact); argmax MATCH per stream. |
| ✗ | Faster m=1 matvec | tune occupancy/rows-in-flight | **Reject** — proven at m=1 SOL (ncu: nothing saturates, latency-bound; multi-row gained +0.7%). | n/a | n/a |
| ✗ | Lower dequant ALU | vectorize unpack | **Already done** — at llama's instruction count (§3a). | n/a | n/a |

---

## 5. The brutally honest bottom line

**Is parity the ceiling for the 27B-Q4_K model's prefill? — Split answer, and the split is the finding:**

- **For the instruction (the int8 m16n8k32 MMA on the Q4_K/Q6_K tensors): YES, parity is the compute ceiling,
  and llama runs the same instruction at the same 219-TFLOP ceiling.** No requant beats it without failing the
  accuracy gate (§1). On raw compute for these tensors, you can match but not out-compute llama.

- **For the kernel: NO, parity is not the ceiling.** llama's Q4_K win is *scheduling* (stream-k + fixup +
  persistent grid + per-call autotune), not a faster MMA. An 82-SM-exact, single-config bw24 can beat llama's
  *general* scheduler on those exact shapes — **est. +8-20% above llama on the Q4_K/Q6_K tensors** from P2+P3.
  This is genuine super-parity headroom, on-thesis, but it is the hardest remaining prefill work and is
  scheduling-bound, not instruction-bound, so the upside is bounded by how much of the 22%→40% SM gap is wave
  quantization (large) vs inner-loop inefficiency (small).

- **The single largest beat lever for the 27B-Q4_K_M as a whole is the FP4 path applied to its 192 NVFP4
  tensors (P1)** — the only place with super-parity *compute* headroom (762 vs 219 TFLOP). **This corrects the
  verdict's framing on two points:** (1) llama is NOT absent from the FP4 path — it runs the identical
  `mma.sync.kind::mxf4nvf4.block_scale` on sm_120 (`mma.cuh:1145`, `common.cuh:286`), so the FP4 beat is
  out-implementing llama's FP4 MMQ, not running an instruction they lack; and (2) **the beat does NOT require
  switching to a different "9B-NVFP4" model** — the daily 27B-Q4_K_M already has 192 NVFP4 tensors that route
  to the FP4 path, and both models are mixed-quant anyway (the 9B has 137 int8-path tensors too).

**So the corrected strategic statement:** the daily-model beat on the 27B-Q4_K_M prefill needs **both** (a) its
192 NVFP4 tensors winning on the FP4 path via 82-SM-exact tuning [P1, the compute beat, llama is a real
competitor], **and** (b) its 273 Q4_K + 33 Q6_K tensors winning on the int8 path via stream-k/fixup/persistent-
grid + compile-time specialization [P2+P3, the scheduling beat, llama structurally can't match the
specialization]. Decode beats only via the batched concurrent path [D1]; m=1 is at the floor and the ALU is at
parity. **There is no shortcut and no single lever** — it is the sum of P1+P2+P3 (prefill) and D1 (decode), each
copy-then-tune-to-the-exact-config, each probe-and-gate, consistent with `win-is-copy-then-tune-exact-case` and
`no-magic-match-then-exceed`.

---

## Evidence index (file:line)

**llama.cpp** (`/data/projects/llama.cpp/ggml/src/ggml-cuda/`):
- FP4 block-scale MMA is native sm_120 warp-level: `mma.cuh:1126-1154` (`mma_block_scaled_fp4`, the
  `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64` at `:1145`).
- `BLACKWELL_MMA_AVAILABLE` = `__CUDA_ARCH__ >= 1200` (sm_120): `common.cuh:286-288`;
  `blackwell_mma_available(cc)`: `common.cuh:360-363`.
- FP4 dispatched on sm_120: `mmq.cu:125` (`use_native_fp4 = blackwell_mma_available(cc) && …`),
  `mmq.cuh:3334`, `vec_dot_fp4_fp4_mma`: `mmq.cuh:996-1064`.
- Q4_K uses MMQ int8 q8_1 path: `should_use_mmq` true for turing+ at any batch `mmq.cuh:307-308`;
  Q4_K MMA tile = `MMQ_MMA_TILE_X_K_Q8_1` `mmq.cuh:254`; load `mmq.cuh:2093-2200`; int8 MMA
  `mma.cuh:946`; q8_1 mma dot (register-resident A) `mmq.cuh:1330-1452`.
- Stream-K: `mmq.cuh:3528-3783` (kernel), persistent `nsm` grid `mmq.cuh:4001`; fixup pass
  `mmq.cuh:3787-3922`; per-call token-tile autotune `mmq.cuh:4069-4082`; runtime nsm/fastdiv
  `mmq.cuh:3968-3973, 3999-4001`.
- Decode mmvq k-quant: `nwarps=2`, `rows_per_block=1` at m=1 `mmvq.cu:430-433, 459`; q4_K vmmq
  `vecdotq.cuh:505-527`; q6_K mmvq `vecdotq.cuh:624-644`.
- Type-id 40 = NVFP4: `ggml/include/ggml.h:430`.

**bw24** (`/home/avifenesh/projects/bw24/crates/bw24-engine/cu/`):
- int8 MMA `qmatvec_gemm.cu:138-140`; kernel1 main loop + A-frag residence `qmatvec_gemm.cu:596-661`;
  fixed `K1_BN=128` `:101`; `__launch_bounds__(256,2)` Q4_K `:687`; decode_q4_k `:224/:343`.
- FP4 block-scale MMA mirror `mma_mxf4_m16n8k64` `qmatvec_gemm.cu:1097-1101`; FP4 kernel
  `__launch_bounds__(128,4)` `:1406`.
- Decode k-quant matvec: q4_K `qmatvec.cu:546-592`, q5_K `:598-646`, q6_K `:722-780`.

**Model gguf histograms** (read from headers this session):
- 27B-Q4_K_M: 192 NVFP4 + 273 Q4_K + 33 Q6_K (+737 F32).
- 9B-NVFP4: 114 NVFP4 + 50 Q8_0 + 44 Q5_K + 43 Q4_K + 1 Q6_K (+410 F32 + 6 BF16).

**bw24 memory plans:**
- `prefill-gemm-rebuild-plan.md` (the binding roadmap; FP4-edge claim at `:13`, ncu head-to-head `:84-90`,
  smem-bound stall `:59`, STEP 2/3 probe scope `:114-115`, W4A4 accuracy gate `:111`).
- `autonomous-work-state-and-direction.md` (decode floor `:68-69,126`, batched path `:118-127`, dequant ALU
  done `:65-69,92`, fleet stream-k-not-tried `:16`, win-mechanism direction `:60`).
- `no-magic-match-then-exceed.md:10` and `win-is-copy-then-tune-exact-case` (the corrected thesis).
