# verify-tier premium diagnosis — RESULTS

Lane: `lane/verify-tier` @ tree 216419d6 (restructure/public-split). RTX 5090 Laptop 24GB
sm_120a. Lever #3 (perf-frontier-20260802 REPORT §3 row / §"3" fix list): name the kernels
that carry the batched-verify b-tier premium that pins spec K at 3. **DIAGNOSIS ONLY — no
kernel changes in this lane.**

Method: `spec-econ` fixed-position interleaved-arm probe (prime 2048 tokens, N=50 + 3
warmups, sync-bounded, rollback outside the timed region) for the cost curve; per-arm
`MEMRA_ECON_ONLY` nsys traces (N=15+3) for attribution; sudo ncu (BASE-CLOCK cells,
`--clock-control base` — never compared to boost numbers) for limiter analysis. Probe
thermal regime: sustained boost 1830–1890 MHz, 72–75 C, 165–170 W (gpustate lines in every
log). Attribution is exact loop-window accounting from the nsys sqlite timelines
(`attribute-loop.py`): the verify loop is the region from the first b-tier (or off-tier
`_mmvq_rp`) launch to trace end; per-pass = window/18; single-stream serialization verified
(kernel-interval sum/union = 1.0000–1.0002). No correction model.

Files: `attribution.md` (kern_sum view), `glue-attribution.md` + `glue-share.jsonl`
(loop-window class split), `cost-curve.jsonl`, `btier-grid-shares.json`,
`ncu/ncu-summary.{md,jsonl}` (regenerated with the ms/us unit fix — the q27-t4-q5k and
q9-t4-q6k exports report duration in ms), raw logs under `logs/` and `ncu/`.

## 1. Cost curve (probe medians, N=50, ctx=2048; T = verify columns = K+1)

| T | q27 verify ms | x decode | µs/extra-col | q9 verify ms | x decode | µs/extra-col |
|---|---|---|---|---|---|---|
| 1 | 21.681 | 1.030x | 0 | 7.682 | 1.047x | 0 |
| 2 | 23.297 | 1.107x | 1616 | 8.852 | 1.207x | 1170 |
| 3 | 24.400 | 1.160x | 1360 | 9.207 | 1.255x | 763 |
| 4 | 26.223 | 1.246x | 1514 | 9.793 | 1.335x | 704 |
| 5 | 32.654 | 1.552x | 2743 | 11.896 | 1.622x | 1054 |
| 6 | 34.894 | 1.658x | 2643 | 12.995 | 1.771x | 1063 |
| 7 | 37.464 | 1.780x | 2631 | 13.959 | 1.903x | 1046 |
| 8 | 40.308 | 1.916x | 2661 | 14.886 | 2.029x | 1029 |
| 9 | 94.853 | 4.508x | 9147 | 29.174 | 3.977x | 2687 |

decode T=1: q27 21.042 ms, q9 7.336 ms. Two cliffs, both tier switches, not smooth
scaling: **T=4→5** (b4-class twins hand off to `_b8_rpsc`; marginal col cost jumps
1823→6431 µs on q27) and **T=8→9** (no b16 twin for NVFP4/Q4_K/Q5_K; the whole pass falls
off-tier onto grid.y=m per-row MMVQ — 4.5x decode). Within a tier the marginal column is
cheap and *falling* per-column (b4 tier: 1360–1514 µs/col q27, 704–763 q9).

## 2. Premium split — glue vs matvec (loop-window deltas vs decode_h, ms/pass)

Classes: `matvec_b` = batched `qmatvec_*_bN` tier; `matvec_m1` = per-row/off-tier matvecs;
`fa_attn` = `fa_decode_* + append_kv`; `gdn` = linear-attention decode; `glue` = norms,
adds, activations, quantize_q8_1; `gap` = window wall − busy.

### q27 (decode_h step = 21.520 ms wall in-window)

| T | premium | matvec net (b − m1) | fa_attn | gdn | glue | gap | non-matvec share |
|---|---|---|---|---|---|---|---|
| 2 | +2.446 | +0.497 (20%) | +0.508 (21%) | +0.120 (5%) | +1.232 (50%) | +0.101 (4%) | **80%** |
| 3 | +3.920 | +1.604 (41%) | +0.744 (19%) | +0.242 (6%) | +1.353 (35%) | −0.006 | 60% |
| 4 | +5.808 | +2.943 (51%) | +1.091 (19%) | +0.340 (6%) | +1.432 (25%) | +0.019 | **50%** |
| 5 | +11.364 | +8.508 (75%) | +1.158 (10%) | +0.417 (4%) | +1.278 (11%) | +0.023 | 25% |
| 6 | +14.186 | +10.917 (77%) | +1.505 (11%) | +0.517 (4%) | +1.244 (9%) | +0.022 | 23% |
| 8 | +19.788 | +15.903 (80%) | +1.946 (10%) | +0.733 (4%) | +1.185 (6%) | +0.041 | 20% |
| 9 | +77.641 | +72.478 (93%) | +2.513 (3%) | +1.028 (1%) | +1.461 (2%) | +0.176 | 7% |

### q9 (decode_h step = 7.591 ms wall in-window)

| T | premium | matvec net | fa_attn | gdn | glue | gap | non-matvec share |
|---|---|---|---|---|---|---|---|
| 2 | +1.360 | +0.376 (28%) | +0.233 (17%) | +0.110 (8%) | +0.667 (49%) | −0.026 | **72%** |
| 3 | +1.760 | +0.691 (39%) | +0.300 (17%) | +0.147 (8%) | +0.702 (40%) | −0.081 | 61% |
| 4 | +2.471 | +1.164 (47%) | +0.464 (19%) | +0.175 (7%) | +0.742 (30%) | −0.073 | **53%** |
| 5 | +4.226 | +2.854 (68%) | +0.499 (12%) | +0.187 (4%) | +0.743 (18%) | −0.057 | 32% |
| 8 | +7.211 | +5.320 (74%) | +0.855 (12%) | +0.302 (4%) | +0.780 (11%) | −0.046 | 26% |
| 9 | +23.729 | +21.569 (91%) | +1.059 (4%) | +0.407 (2%) | +0.725 (3%) | −0.030 | 9% |

**The headline the survey missed: at the operating tier (T=4, K=3) the premium is only
half matvec.** Non-matvec carries 50% (q27) / 53% (q9); at T=2–3 it carries 60–80%. The
matvec share only dominates after the T=5 tier cliff.

**The glue premium is a step function, not T-scaled.** Batched verify switches the whole
epilogue off the fused decode path: decode runs `add_rms_norm_q8_1` (0.649 ms/step q27) +
`silu_mul_scaled_q8_1` + `gated_rmsnorm_q8_1` + `ssm_conv1d_fused_decode` +
`gdn_prep_decode`; verify replaces them with the unfused f32 chain — `quantize_q8_1`
(0.027→0.86 ms/pass), `rms_norm_f32` (0.08→0.68), `add_f32` (0.002→0.23), `l2_norm_f32`
(0→0.18), `silu_mul_f32`, `gated_rmsnorm_f32` (q27 numbers; q9 mirrors at ~40% scale,
plus a verify-only `scale_f32` 0.14–0.16). Total glue: 1.09→2.33 ms at T=2 and *flat
through T=8* (2.28–2.53). This ~1.3 ms (q27) / ~0.7 ms (q9) is a constant tax on every
verify pass at every T. Per-kernel tables: `glue-attribution.md`.

**The only truly T-scaled non-matvec carrier is attention:** `fa_decode_vec_q_rows_v4`
0.75→1.28→2.04 ms/pass at T=2/4/8 (q27) vs 0.30 for the decode deep kernel — the rows
variant never got the fa-decode-deep treatment. gdn scales weakly (`gdn_scan_s128`
0.28→0.79 at T=8). Launch-count gap is a non-issue (≤0.1 ms everywhere; ~2100
launches/pass q27).

## 3. Premium-carrier table (b-tier kernels, nsys kern_sum, ms/pass; full set in attribution.md)

q27: T=2–4 carried by `dual_b2/b4` (7.9–8.7, 33% of wall), `b4_rpr2w8` (7.6–7.9, 30%),
`b4_rp` (2.4–2.8, 10%), `q5_K_b4_r2` (1.1–1.2); T=5–8 collapses onto **`_b8_rpsc`**
(25.7→32.8, 79–81% of wall); T=9 off-tier `_mmvq_rp` 81.8 + `q5_K_mmvq` 9.2.
q9: T=2–4 spread over `dual_b4` (2.1–2.2), `b4_rpr2` (1.5), `q6_K_b4` (1.0), `q5_K_b4`
(0.9–1.0), `q4_K_b4` (0.9–1.0); T=5–8 `_b8_rpsc` (4.9→6.1) + k-quant b8 tail; T=9
off-tier 26.0 ms of m=1-class launches.

The T=4→5 cliff is *entirely* the b4→b8 tier switch: matvec_b jumps +5.57 ms of the
+5.56 ms wall cliff (q27), +1.69 of +1.76 (q9).

## 4. Gap-to-peak, top carriers (ncu BASE-CLOCK, time-weighted across grids by nsys ms/pass)

Peak reference: m=1 `dual_mr2_rp` sustains 92.9% DRAM / 467 GB/s (⇒ peak ≈ 503 GB/s at
base clocks).

| carrier | arm | ms/pass | wDRAM% | wSM% | wOcc% | verdict |
|---|---|---|---|---|---|---|
| `dual_b4_rpr2` | q27 T=4 | 8.74 | **84.6** | 48.0 | 55.9 | near-roofline; ≤10% headroom — leave alone |
| `b4_rpr2w8` | q27 T=4 | 7.89 | 59.1 | 35.8 | 64.4 | ~25 pp BW headroom |
| `b4_rp` | q27 T=4 | 2.84 | 27.2 | 17.5 | 45.3 | latency-bound small grids (incl. grid=12 tails) |
| `b8_rpsc` | q27 T=5 | 25.70 | **49.5** | 34.5 | 50.3 | biggest pool; see verdict below |
| `b8_rpsc` | q27 T=8 | 32.80 | 39.1 | 32.0 | 51.5 | degrades with T (columns thrash L2) |
| `b8_rpsc` | q9 T=5/T=8 | 4.87/6.08 | 51.9/42.7 | 40/37 | 53/54 | same shape |
| `dual_b4` / `q6_K_b4` | q9 T=4 | 2.23/1.06 | 76.2/93.5 | 49/63 | 54/56 | healthy |
| `q5_K_b4` (no r2) | q9 T=4 | 1.04 | 32.5 | 35.5 | 62.5 | worst b4-class cell |
| off-tier `_mmvq_rp` | T=9 | 81.8/17.6 | **18.5/17.6** | 79/80 | 78 | compute-bound (re-dequant per row) — the K=8 cliff mechanism |

**mmvq_b8 verdict: the survey's "30–35% of peak BW" is REFUTED as a kernel-level
characterization.** Time-weighted, `_b8_rpsc` runs at 49.5% (q27 T=5) / 51.9% (q9 T=5) of
peak DRAM, degrading to 39–43% at T=8. Only the low-end grid cells match the survey number
(grid=640 head-projection launches: 41% at T=5 → 32% at T=8; near-zero tiny grids). The
real story is still a large gap: 35–45 pp below the dual_b4/m=1 class (84.6–92.9%), and
the floor argument is sharper than any BW% — verify reads the *same weight bytes* as one
decode step, so perfect twins cost ≈ decode matvec time (18.5 ms q27) + T-scaled
activation traffic. `_b8_rpsc` at T=5 spends 27.0 ms (incl. q5_K_b8) against that ~19 ms
floor.

## 5. Acceptance receipts (q9, run-spec K=1..8 sweeps, single-run labeled, self-consistency PASS all K)

prose: K=3 207 tok/s (acc 49.4%, 2.48 tok/round) > K=4 185 (41.5%) > K=5 172 (34.9%);
K=8 76 tok/s (0.56x — the off-tier cliff). code: K=3 275 tok/s (72.8%, 3.19 tok/round) >
K=4 244 (60.2%, 3.41) > K=5 229 (51.0%, 3.55). verify-wait is 64–68% of round time at
K≤7. K=3 is the current optimum purely because v(T) grows faster than tok/round.

## 6. Ranked fix list (with exactness classification)

Exactness classes: **[trivial]** = per-(token,row) dot-product chain identity preserved by
construction (independent columns, same per-column accumulation order as m=1 — the house
dual-twin pattern); **[needs-care]** = a reduction/scheduling order changes and must be
pinned bit-exact (KAT bitdiff=0 precedent from the fa lane).

1. **Kill the T=5 tier cliff** — extend b4-class exact twins to b5–b8 (or rebuild
   `_b8_rpsc` to b4-class efficiency). The cliff is 100% matvec_b (+5.6 ms q27). On the b4
   trajectory (+1.3–1.8 ms/col within-tier) v(T5) ≈ 1.35 instead of 1.55 (q27), ≈ 1.46
   instead of 1.62 (q9). [trivial] if per-column chains are kept; rpsc's split-k reorder
   is [needs-care].
2. **Re-fuse the batched epilogue** — build bN twins of `add_rms_norm_q8_1`,
   `silu_mul_scaled_q8_1`, `gated_rmsnorm_q8_1`, the fused ssm-conv/gdn-prep decode path,
   and kill the verify-only `scale_f32`/`l2_norm_f32` unfused chain. Constant −1.3–1.4
   ms/pass (q27) / −0.7 (q9) at *every* T: v(T4) 1.246→~1.18 (q27), 1.335→~1.24 (q9).
   Elementwise/rowwise per-column ops: [trivial].
3. **BW-push the mid-tier b4 cells** — `b4_rpr2w8` 59%→80% class and the latency-bound
   `b4_rp` small grids (batch the grid≤256 tail launches); q9's twin-less `q5_K_b4`
   (32.5%) wants the `_r2` treatment its q27 sibling has (84.9%). Pool ≈ 2–3 ms/pass at
   T≤4. Scheduling-only: [trivial].
4. **fa rows deep-ladder** — `fa_decode_vec_q_rows_v4` is the only T-scaled non-matvec
   carrier (2.0 ms at T=8 q27 vs 0.3 decode); port the fa-decode-deep split. Softmax
   reduction order: [needs-care].
5. **b16 twins for NVFP4/Q4_K/Q5_K** — removes the T=9/K=8 off-tier catastrophe (4.5x/4.0x
   decode; off-tier kernel is compute-bound at 18% DRAM). Unblocks K=8 experiments, but
   measured K=8 acceptance (21–33%) caps its e2e value: do after 1–4. [trivial] pattern.

**Expected K=4–5 unlock** (counterfactual v(T) from the measured class splits, scaled
through the measured verify-wait share of 65–68%): fixes 1+2 give q9 v(T5) 1.622→1.35
(q27 1.552→1.29) — K=4 reaches *parity* with K=3 (code 274 vs 275 tok/s, prose 208 vs
207). Fix 2 alone lifts the still-optimal K=3 by +5% (code 275→290, prose 207→218). The
full stack (1–4, b8 pool to b4-class BW) gives v(T5) ≈ 1.19 and **re-opens K=4 at +7–9%
e2e over today's K=3** (code 296, prose 225), consistent with (and bounding under) the
+40% v=1.05 counterfactual from verify-economics §3 that the REPORT quotes as the K=4–5
ceiling. K=5 stays acceptance-limited on prose (34.9%); code (51%, 3.55 tok/round) is the
candidate scenario.

Survey cross-checks: "b4 trunk matvecs ≈ 78% of the T=4 verify pass" — CONFIRMED (81.9%
of probe wall / 78.6% of nsys window). "mmvq_b8 at 30–35% of peak" — REFUTED as stated,
see §4 (39–52% time-weighted; the gap is real but the mechanism is per-T degradation +
small-grid cells, not a uniform 30% ceiling).
