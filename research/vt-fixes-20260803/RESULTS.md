# vt-fixes — verify-tier fixes 1+2, implementation + receipts

Lane: `lane/vt-fixes` @ restructure/public-split (50bf95bb). RTX 5090 Laptop 24GB sm_120a.
Implements verify-tier-20260802 RESULTS.md fixes 1 and 2 (the merged diagnosis). All raw
logs under `logs/`; machine-readable rows in `vt-fixes-summary.jsonl`. Pre-fix binaries
(built at the branch point, before any lane change) preserved in `prefix-bin/` and used as
the interleaved A-arm.

## What landed (commit order)

1. `8db4c23f` — kernel-check T-row bit-identity arms: add_rms_norm_q8_1 + rms_norm_q8_1 at
   T=2/4/5/8 vs the verify's exact unfused chain (add_f32 -> rms_norm_decode(1024) ->
   quantize_q8_1). No new CUDA needed — the m=1 fused kernels are row-indexed
   (blockIdx.x=row), so a T-row launch IS the per-row m=1 program.
2. `4b791095` — kernel-check batched arms: silu_mul_scaled_q8_1 over flat n=T*n_ff (unit
   scales) vs silu_mul_f32 -> quantize_q8_1; gated_rmsnorm_q8_1 at nrows=num_v*T (T=1,5)
   vs gated_rmsnorm(128) -> quantize_q8_1. All mismatch=0.
3. `188a0083` — **fix 2 wiring**: decode_step_t_core_stream (the batched verify) now runs
   the m=1 decode's fused epilogue structure at nrows=T — attn-input norm emits q8_1
   directly (rms_norm_q8_1), post-attn residual+norm+quantize is ONE add_rms_norm_q8_1,
   SwiGLU emits act pre-quantized with NVFP4 macro-scales folded (silu_mul_scaled_q8_1 via
   the deferred-scale dual), GDN out-norm emits q8_1 (gated_rmsnorm_q8_1 T-wide). New
   engine entries `matmul_decode_exact_pre` / `matmul_decode_exact_dual_pre` mirror the
   decode-exact dispatch condition-for-condition, consuming a caller-quantized activation
   (quantize_q8_1 is deterministic -> identical kernel input bytes).
4. `13e78e0a` — **fix 2 cross-layer carry**: decode_step_h's launch-arc fusion ported to
   the verify loop — layer il's post-FFN residual add folds into layer il+1's
   add_rms_norm_q8_1 (take()-first pattern kept).
5. `1e6fb1f1` — **fix 1**: exact-width b5/b6/b7 batched twins (rpsc + rpr2w8 schedules).
   The T=5..7 tier rode MCOLS=8 kernels whose acc[2][8] is allocated at any m. Same
   template at MCOLS=m = identical per-(token,row) chain (columns c>=m never execute in
   either form) -> bit-identical. Dispatch remap in qmatvec_mmvq_batched, NVFP4 rp only,
   MEMRA_B567=0 rollback. kernel-check m=7 row added (bit-bad=0).
6. `425b0dc3` — **fix 1b**: exact-width dual gate+up twins for T=5..7
   (dual_b5/b6/b7_rpr2). Distinct cell from the killed verify-economics b8 dual (which was
   MCOLS=8 at m=5..8, flat): exact width keeps the b4-dual register shape. bit-bad=0/0.

## v(T) cost curve (spec-econ, N=50+3 warmups, ctx=2048, prose prime; pre = branch-point binary)

Boost-clock cells (1830-1890 MHz band logged in every probe log; same-session interleaved
pre/post). q27 = Qwen3.6-27B NVFP4+Q4_K_M MTP; q9 = Qwen3.5-9B NVFP4 MTP.

### q27 (decode_h ~21.3-21.5 ms)

| T | pre x | post-fix2 x | post-fix1+2 x | Δms (pre→final) |
|---|---|---|---|---|
| 1 | 1.041 | 1.005 | 0.999 | −0.9 |
| 2 | 1.116 | 1.065 | 1.059* | −1.2 |
| 3 | 1.167 | 1.115 | 1.117* | −1.1 |
| 4 | 1.257 | 1.201 | 1.205 | −1.0 |
| 5 | **1.569** | 1.518 | **1.465** | −2.1 |
| 6 | 1.677 | 1.630 | 1.571 | −2.1 |
| 7 | 1.806 | 1.750 | 1.693 | −2.3 |
| 8 | 1.948 | 1.888 | 1.893 | −1.0 |
| 9 | 4.513 | 4.416 | (unchanged — b16 untouched) | — |

### q9 (decode_h ~7.40 ms)

| T | pre x | post-fix2 x | post-fix1+2 x | Δms (pre→final) |
|---|---|---|---|---|
| 1 | 1.052 | 1.010 | — | −0.34 |
| 2 | 1.222 | 1.145 | — | −0.61 |
| 3 | 1.271 | 1.198 | — | −0.65 |
| 4 | 1.356 | 1.296 | 1.285 | −0.52 |
| 5 | **1.655** | 1.592 | **1.528** | −0.93 |
| 6 | 1.810 | 1.750 | 1.668 | −1.04 |
| 7 | 1.938 | 1.883 | 1.810 | −0.93 |
| 8 | 2.071 | 2.020 | 2.019 | −0.36 |

\* T=2/3 final numbers from the fix1a probe (b5-b7 singles don't fire below T=5; duals
b2-b4 unchanged) — see probe-*-post-s5/-fix1a logs.

The glue step-tax is gone where it was promised: T=1 verify premium collapses from
+4.1%/+5.2% to ~0 (q27 0.999x — noise-level). The T=4→5 cliff softens but is NOT fully
killed: q27 marginal T=4→5 cost 6.67 ms pre → 5.57 ms post (fix 1 exact-width removes the
register tax, but b8-class BW (§4 of the diagnosis, ~50% of peak) still binds — the "b8
pool to b4-class BW" item was priced under fix 3, not fixes 1+2).

## Fix-2 delta at K=3 (the +5% claim) — VERIFIED

run-spec e2e, 256 gen, interleaved pre/post per rep, N=3 each arm, same session. Medians;
all runs self-consistency PASS with acceptance identical pre/post (exactness receipts).

| cell | pre | post | Δ |
|---|---|---|---|
| q9 code K=3 | 271.8 | **282.4** | **+3.9%** |
| q9 prose K=3 | 204.8 | 213.4 | +4.2% |
| q27 code K=3 | 106.1 | 109.4 | +3.1% |
| q27 prose K=3 | 101.7 | 105.4 | +3.6% |

The diagnosis priced "fix 2 alone = +5% (code 275→290)" from the counterfactual glue
subtraction; measured is +3.9-4.2% on q9 (280-284 tok/s vs the 290 prediction). The gap:
the counterfactual removed ALL glue delta; the fused chain still pays the (cheaper) fused
launches, and verify-wait is 64-68% of round time so ~0.5 ms of the 0.7 ms/pass reaches
e2e. Close to claim, slightly under. Note baseline drift: this session's pre-fix K=3 code
is 271.8 (diagnosis session: 275) — same binary, cross-day clock drift, which is why every
comparison here is same-session interleaved.

## K=4 verdict (fixes 1+2, all commits live)

| cell | K=3 post | K=4 post | K=5 post | optimum |
|---|---|---|---|---|
| q9 code | **282.4** | 253.8 | 237.8 | **K=3 stays** |
| q9 prose | **213.4** | 193.8 | 179.6 | **K=3 stays** |
| q27 code | **109.4** | 107.5 | 100.8 | K=3 (K=4 gap 3.1%→1.7%) |
| q27 prose | **105.4** | 100.1 | 100.2 | K=3 |

**K=4 does NOT re-open at +7-9%.** The diagnosis's unlock pricing required v(T5) ≈ 1.19
(the FULL stack incl. fix 3's b8 BW push); fixes 1+2 alone were priced at K=4 *parity*
(q9 v(T5) 1.35 predicted). Measured v(T5) landed at 1.47-1.53 — the fix-1 exact-width
twins removed the register-shape tax (~0.9-2.3 ms at T=5-7) but not the b8-class bandwidth
gap, which fix 3 owns. Consistent with the diagnosis's own structure: the +7-9% cell was
conditional on the full stack. K=4 gap narrowed everywhere (q9 code 11.6%→10.1% behind
K=3; q27 code 3.7%→1.7%) — fix 3 (BW-push) + fix 4 (fa rows deep) remain the unlock path.
No default-K change is made or recommended from this lane (owner-visible decision).

## Battery status (lane end, this tree = 425b0dc3)

- kernel-check FULL naked: **ALL GREEN**, 0 KC-SKIP (logs/kernel-check-full-final.log).
  New arms in the battery: T-row add_rms_norm_q8_1/rms_norm_q8_1 (T=2/4/5/8),
  silu_mul_scaled_q8_1 batched, gated_rmsnorm_q8_1 (T=1,5), RP BATCHED m=7,
  DUAL m=5/6/7 rp.
- run-gen argmax x3 models (q9, q27, q35 chat): prefill+decode **MATCH**, zero
  MISMATCH-STRUCTURED (logs/rungen-*-final.log).
- run-spec K=1..8 self-consistency: **PASS 8/8 on q9 AND q27** at every commit point
  (logs/runspec-*-postfix*.log — 6 sweeps total across slices).
- fast-gate: full-scope run GREEN at slice 4 (kernel-check FULL 251s + 6 probes
  golden-identical); final full sweep vs the branch point in /tmp/vt-fastgate-final.log.
- Every e2e row in the K-sweep: self-consistency PASS, acceptance bit-identical pre/post.

## Thermal regime

All probe logs carry gpustate lines: sustained 1830-1890 MHz, 56-75 C. Medians are N=50
(probes) / N=3 interleaved same-session (e2e). No cross-day comparisons.
