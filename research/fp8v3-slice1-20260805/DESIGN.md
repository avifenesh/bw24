# FP8 v3 slice 1 — mantissa-extraction design + paper verdict

Lane `lane/fp8-v3-slice1`, 2026-08-06. Written by the orchestrator after four agent runs died
in API churn with zero artifacts; the analysis below supersedes the pod-measurement plan with a
paper verdict strong enough to re-scope the lane. Sources: `research/fp8v3-gate-20260805/VERDICT.md`
(Q1 GO: s32-vs-f32 accumulate +16.7pp m=512 / +23.1pp m=6257, weighted, on the FLOOR kernel;
extraction charged at zero), `cu/mmq_fp8_blk.cu` (the v2 kernel structure),
`research/prefill-gemm-20260806/VERDICT.md` (the issue-interval finding, landed AFTER the Q1
verdict and material to it).

## 1. The problem, restated with the new fact

v3's premise: swap the v2 kernel's f32 accumulator (`mma...f8f6f4...f32.e4m3`) for the floor's
s32 chain (`mma...s32.s8.s8.s32`), paying a per-128-block mantissa-extraction cost that Q1
charged at zero. Budget: +16.7/+23.1pp.

**New fact that reframes everything** (prefill-gemm phase 1, 2026-08-06): the s8 MMA's
tensor_cycles/tensor_insts = exactly 16.00 on sm_120a — the kernels in this family are
ISSUE-INTERVAL-bound, not pipe-utilization-bound, and the NVFP4 W4A8 kernel sits at 80.9% of
dense s8 peak. The measured +3.17% for deleting the entire f32 scale fold (vs 44.7% predicted
from pipe %) proves non-MMA work in these loops is nearly free — it hides inside the MMA issue
interval. Two consequences cut in OPPOSITE directions for v3:

- FOR v3: extraction work in the loader is overlappable — `long_scoreboard = 0.04` says loads
  are fully hidden today; adding ALU/LUT work to the copy path plausibly costs ~0, so the Q1
  "charged at zero" assumption may in fact be nearly TRUE.
- AGAINST v3: the Q1 +16.7/+23.1pp is an ISSUE-INTERVAL difference between the two MMA classes
  (the s8 arm executed MORE instructions and still won ~20% — only a faster per-MMA issue rate
  explains that). But v2 already recovered that gap STRUCTURALLY: v2 measures ~1.00x the s8
  floor at m=6257 despite riding the slower e4m3 MMA, because its once-per-128 fold beats the
  floor's per-32 fold. v3 = s8 MMA + v2's fold structure, so its ceiling over the FLOOR is not
  +20pp — it is (fold-structure edge) + (byte edge), i.e. the ~3% fold family (measured, retired)
  plus ≤6.25% loader bytes (and DRAM is at 8–10% utilization, so the byte edge likely converts
  to ~0 in-kernel). The +20pp exists only against v2's OWN f32 arm, and v2 already closed that
  gap by other means.

## 2. Mapping A — LUT requant (e4m3 → s8 against the block scale)

Construction: per 128-block, `s8_i = round(w_i / amax_blk * 127)`. As a LUT: e4m3 byte → s8
needs the block's amax exponent only (e4m3 is sign/exp/mantissa; division by a power-of-two
amax bucket is an exponent shift), so one 256-entry s8 table per amax-bucket, shared across
blocks with equal bucket — in practice a handful of tables in constant/LDS, one lookup per
element in the copy path (overlappable, per §1).

**Error model.** e4m3 = M × 2^E with M ∈ {8..15} normal (3 mantissa bits) or {0..7} denormal,
E ∈ [-9, 5]. An s8 grid against block amax carries 7 magnitude bits. A value 2^-k below the
block amax needs 3+k fixed-point bits to keep all e4m3 mantissa bits; s8 keeps them exactly for
k ≤ 4 and loses (k−4) low bits below that. Worst-case relative error for surviving (nonzero)
values: 2^-(3-(k-4)) ... saturating to total loss for k ≥ 11 (values >2^11 below amax flush
toward zero — e4m3 itself only spans 15 binades, and block-128 amax scaling concentrates mass
near amax; the mmq-v2 receipts measured the same widening argument at 32→128 and found the tail
population negligible). **Measured (host_proof.py, uniform-random e4m3 codes): rms_rel = 7.1e-1, worst = 1.0** — on
codes spanning the full 15-binade e4m3 range, the s8 grid destroys everything >2^4 below block
amax. Real weight blocks are amax-concentrated, so the production number would be far smaller —
but the uniform-random result is the honest worst case, and it is the SAME failure mode that
made ARM A's per-tensor fold non-shippable (stream divergence at pos 20). Mapping A is a
requant arm with a distribution-dependent error floor, needing its own host reference + NLL
gate, and it is NOT bit-preserving of e4m3 — a strictly worse exactness contract than the
shipped v2 kernel's zero-weight-loss copy path.

## 3. Mapping B — exact decomposition (refuted for 8-bit operands)

Exact integer form: w = (−1)^s · M · 2^E, M ∈ 0..15, E ∈ [−9, 5]. To accumulate a 128-block in
one s32 chain the products must share one binade: align M_i by 2^(E_i − E_min). Worst-case
in-block exponent spread = 14 binades → aligned integer width 4+14 = 18 bits. **s8 MMA operands
are 8-bit; an 18-bit aligned mantissa cannot ride s8×s8 MMA.** Clamping the spread to fit 8 bits
(shift-and-truncate below amax−2^4) is arithmetically IDENTICAL to mapping A's s8 grid — the
"exact" mapping collapses into the requant mapping the moment the operand width is fixed at 8.
Wider integer MMA (s16/s32 operands) does not exist in the sm_120a mma.sync menu at competitive
throughput. **Mapping B is not a distinct option; there is only mapping A.**
(host_proof.py demonstrates both: B reproduces f64 exactly with unbounded integers — proving
the decomposition logic — and collapses to A's error the moment operands clamp to 8 bits.)

## 4. Op-count and paper cost

v2 inner loop today (per 16B int4 chunk): 1 ld.global.v4 + 1 st.shared (pure copy — the header's
"no dequant, no LUT, no fold" contract). Mapping A adds per element: 1 LDS table read + byte
insert (or PRMT-class byte math) ≈ 2–3 ALU-class ops/element in the copy path. Against §1's
finding (non-MMA work hides in the 16-cycle MMA interval; fold removal = +3.17% TOTAL), the
extraction cost projects to **~0–2% — the Q1 zero-charge holds approximately**. The extraction
is NOT what kills v3.

## 5. Verdict — NO-GO for the v3 kernel build

What kills v3 is §1's second arm: the budget was misattributed. The +16.7/+23.1pp is the MMA-
class issue-interval gap, which v2 already neutralized structurally (v2 ≈ 1.00x floor). v3's
honest ceiling over the shipped state = fold-structure (+~3%, measured and retired by the
prefill lane on the adjacent kernel) + loader bytes (≤6.25% at 8–10% DRAM utilization → ~0
in-kernel) − extraction (~0–2%) ≈ **low single digits at best, with a NEW requant exactness
contract and its own NLL gate as the price**. The prefill-gemm lane's conclusion ("two lanes
converged on a 3% lever") stands confirmed by this analysis from the third direction.

Recommendation: **close the v3 idea.** The FP8 prefill story rests on the shipped exact
per-block MMQ (v2 at parity, native residency default, −430 MiB) — which is already the right
artifact. If anyone reopens this: the ONLY measurement that could revive it is a microbenched
issue-interval comparison of `f8f6f4.e4m3.f32` vs `s8.s32` MMA on sm_120a (the 16.00-method);
revive only if e4m3's interval is ≥2x s8's AND v2's fold advantage does not already absorb it.

## 6. host_proof.py

Committed alongside: numpy proof that (B) with unbounded integers reproduces f64 dots exactly
(decomposition logic sound), and measured rms_rel for (A) on 1000 random e4m3 blocks (the
number §2 predicts). Run: `python3 host_proof.py`.
