# verify-economics: the q27/q35 spec-verify price sheet, the payoff model, and the kernel verdicts (2026-08-02)

Lane `lane/verify-economics` (from `restructure/public-split` 5e37780e). Rig: RTX 5090 Laptop
24GB (sm_120a, 82 SMs), the deployment target. Every GPU-touching process under
`flock /tmp/gpu5090.lock` (two other lanes shared the rig during the session — every log
carries its own GPU bracket; all A/Bs are interleaved within-window by construction).
Artifacts: q27 = `/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf`
(the NVFP4 daily, 15.7GB, NVFP4 trunk + Q5_K embed/head) with the own-trim drafter
`draft-daily-owntrim-nvfp4head-q4blk.gguf` (the board spec-row config); q35 =
`/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` (Q8_0 trunk,
IQ3_S/IQ4_XS expert banks) with its own-trim drafter.

## Verdict up front

1. **The mission's premise is REFUTED on this tree/rig**: verify does NOT cost ~(K+1)
   decode-steps. The weight-shared multi-row dp4a verify kernel the hypothesis calls for
   ALREADY EXISTS (the `_b{2,4,8}` batched-MMVQ tier, shipped 2026-07-03, bit-identical per
   (token,row) by construction, kernel-check-pinned): q27 verify T=2/3/4 costs
   **1.13/1.19/1.30x ONE decode step** (probe, N=50). The remaining slope is structural, not
   a missing kernel.
2. **q27 spec pays 2.00x (prose) / 2.18x (code) at K=3 on this rig** — already past vLLM's
   1.9x H100 economics in ratio terms, from the opposite direction (slow decode makes
   accepted tokens valuable; our verify is 1.19-1.30x, not free).
3. Kernel work: **one arm ships** — the DUAL gate+up batched twin (one launch for the verify
   FFN pair at t=2..4, split-plane + GGUF layouts, bit-identical per (tensor,token,row);
   **-0.2..-0.3ms/verify pass at T=2..4, 9/9 interleaved probe pairs; +0.8% q27 spec e2e at
   K=3, 6/6 live pairs positive-or-flat**; `MEMRA_SPEC_DUAL_T=0` rollback). Refuted along
   the way: the b8 dual (flat vs the rpsc singles — killed), pfr2/ca b4 variants (flat/-25%),
   a blanket b8 r2-over-r2w8 rule (shape-mixed, ~1%, parked). The contract held everywhere:
   **kernel-check ALL GREEN (332 OK / 0 FAIL, DUAL-BATCHED bitwise 6/6 on both layouts),
   run-spec K=1..8 self-consistency PASS on every battery (72/72 baseline + 72/72 after +
   12 A/B cells), acceptance counts bit-identical before vs after**.

## 1. The price sheet — decode(T=1) vs verify(T), probe receipts

`spec-econ` (new probe, committed): fixed cache position (snapshot/rollback between every
measurement), arms interleaved round-robin per iteration (thermal drift spreads evenly),
sync-to-sync timing, N=50 + 3 warmups, real greedy continuation tokens. Medians:

| arm (ctx, N=50) | decode T=1 | vT1 | vT2 | vT3 | vT4 | vT5 | vT6 |
|---|---|---|---|---|---|---|---|
| q27 board-2048 | 23.29ms | 1.03x | **1.13x** | **1.19x** | **1.30x** | 1.58x | 1.71x |
| q27 code ctx=28 | 22.93ms | 1.05x | 1.15x | 1.24x | 1.34x | 1.63x | 1.76x |
| q35 board-2048 | 5.66ms | 1.00x | **1.27x** | **1.41x** | 1.66x | 1.95x | 2.18x |

- The q27 curve is **ctx-independent** (28 vs 2048 tokens: same shape) — the slope lives in
  the trunk matvec tier, not attention.
- **T=5 tier-entry cliff** (1.30 -> 1.58): the b4->b8 mcols boundary at m=5.
- **T=9 (K=8) falls off the tier entirely**: NVFP4/Q4_K/Q5_K have no b16 twin — verify
  drops to grid.y=m per-row MMVQ (weights re-read T times). Live effect: K=8 verify
  103-106ms/round, e2e **0.77x** (a real cliff, but K=8 is far off-optimum; noted, not fixed).
- q35's steeper slope is MoE-structural: expert weights are read per (token,slot) — the
  IQ-bank CSR arm dedups gate_up, `down` re-reads per token (both CSR-down variants were
  mechanism-refuted earlier: 16-group rows can't amortize dedup), plus F32 ssm_beta/alpha
  ride t per-column cuBLASLt m=1 calls (the decode-exact Float contract — cuBLASLt m=1 IS
  the T=1 program; batching it would change decode's FP order, out of contract).

## 2. K-sweeps + phase decomposition (run-spec, N=3 reps, NGEN=256, MEMRA_SPEC_PHASE=1)

72/72 self-consistency PASS. Acceptance bit-identical across reps (greedy-deterministic).
Medians (full tables: `parse-econ.py logs`):

**q27 prose (board-2048), plain 45.00 tok/s:**

| K | acc | spec tok/s | ratio | draft/rd | verify/rd | commit/rd |
|---|---|---|---|---|---|---|
| 1 | 85.5% | 68.8 | 1.53x | 0.9ms | 25.3ms | 0.4ms |
| 2 | 81.1% | 88.0 | 1.95x | 1.7 | 27.1 | 0.4 |
| **3** | 66.3% | **90.0** | **2.00x** | 2.5 | 29.5 | 0.5 |
| 4 | 60.2% | 84.9 | 1.89x | 3.2 | 35.3 | 0.5 |
| 8 | 36.7% | 34.4 | 0.77x | 6.7 | 106.0 | 0.6 |

**q27 code (p1-code-short), plain 46.59 tok/s:** K=3 best **102.1 tok/s = 2.18x** (acc 71.5%),
K=4 2.12x, K=8 0.79x.

**q35 prose, plain 172.95 tok/s:** K=2 best **212.7 = 1.23x** (acc 52.4%), K=3 1.16x, K>=4 <1x
(the own-trim head chains poorly past 2 on prose).

Phase decomposition: **verify is ~90% of round wall at every profitable K** (draft 3-11%,
commit ~1%), and the live verify ms/round equals the probe's c_v(K+1) within ~2% (K=3:
29.5 live vs 30.3 probe x its round mix) — no hidden live overhead; the probe curve IS the
live verify price.

## 3. The payoff model, and where vLLM's economics sit

`speedup(K) = (1 + a*K) / (v(K+1)*c1 + K*d + o) * c1_live` — the model tracks measured
within ~5% at K<=5 (parse-econ.py §3; K>=6 uses v(7..9)~v(6), K=8 excluded — the tier
fall-off isn't in the probe range). The counterfactual column re-prices OUR acceptance
under **vLLM-class verify economics (v = 1.05, verify rides the same tier as decode)**:

| q27 code | K=2 | K=3 | K=4 | K=5 |
|---|---|---|---|---|
| measured | 1.98x | **2.18x** | 2.12x | 1.98x |
| counterfactual v=1.05 | 2.29x | 2.67x | **3.05x** | 3.07x |

Reading: our verify premium costs ~0.5x at the optimum and — more importantly — **pins the
optimum at K=3**; with vLLM-class verify the same acceptance curve would push K to 4-5 at
~3x. That is the precise, quantified residual gap to "their verify rides its fastest GEMM
tier unpinned". On the H100 (the receipts that motivated this lane) the same structure holds
with a 2x faster decode denominator, which is why their q27 spec pays 1.08x there while the
5090's pays 2.0-2.2x: **spec payoff scales with decode slowness, and our verify tier scales
with T — both rigs sit on the same curve, at different points.**

## 4. Content-class effect (the Hy3-profile prediction, confirmed on q27)

Same model, same drafter, same battery, only the prompt class changes:

| K | prose acc | code acc | prose ratio | code ratio |
|---|---|---|---|---|
| 1 | 85.5% | **92.5%** | 1.53x | 1.61x |
| 3 | 66.3% | **71.5%** | 2.00x | **2.18x** |
| 4 | 60.2% | **67.5%** | 1.89x | 2.12x |

Code-class acceptance is +5-7pts at every K -> +0.18x at the K=3 optimum (2.00 -> 2.18).
Same direction as the Hy3 K=1 profile (code-gen 75.3% vs prose 43.8%), compressed by the
much stronger own-trim drafter (this drafter chains; Hy3's nextn=1 head could not).

## 5. Kernel work — what was tried, what survived

The mission's branch test "if weight reads are NOT already shared across rows" resolved to
**they ARE** (map: `matmul_decode_exact` -> `qmatvec_mmvq_batched` `_b{2,4,8}`, one weight
read serves m tokens, per-(token,row) chains bit-identical to the m=1 MMVQ program;
`fa_decode_rows` batches attention grid.z=row; MoE rides rows/CSR twins). So the lane
attacked the measured walls:

- **SHIPPED — dual gate+up batched twin** (`qmatvec_nvfp4_mmvq_dual_b2[_rp]` /
  `dual_b4_r2|rpr2`, `Engine::matmul_decode_exact_dual`, wired in `decode_step_t_core`'s
  Dense FFN arm; default ON, `MEMRA_SPEC_DUAL_T=0` rollback): ONE launch computes both FFN
  projections of a verify batch — blockIdx.y selects the tensor, per (tensor,token,row) the
  body is the single-launch template verbatim on the SAME layout -> bit-identical by
  construction, kernel-check DUAL-BATCHED **6/6 bitwise OK on both layouts** (GGUF +
  split-plane rp). Measured (all interleaved within-window): probe verify pass
  **-0.2..-0.3ms at T=2/3/4, 9/9 pairs negative** (control arms decode_h/verify_t1 flat);
  live q27 K=3 **+0.8% e2e (6/6 pairs, off med 90.54 vs on med 91.10 across both A/B
  batteries)**. The b8 dual measured FLAT vs the faster rpsc singles and was cut — the
  dual covers exactly the b2/b4 tiers (verify T=2..4 = the profitable K=1..3 window).
- **THE DEAD-ARM LESSON (evidence discipline)**: the FIRST dual A/B measured "FLAT" — and
  was VOID. The live q27 trunk carries the A6 split-plane (rp) layout; the first wrapper
  gated `rp: false` and silently never engaged (nsys caught it: the live verify runs
  `_b4_rpr2/_rpr2w8/_rp` twins). Every A/B since carries an ENGAGED receipt line
  (`MEMRA_DEBUG=1`) in the log. A flat A/B without an engagement receipt is not a verdict.
- **pfr2 / ca variants at b4** (never-auto'd; forced A/B x2 interleaved, ffn_gate,
  DRAM-cold copies=8): pfr2 flat-to-worse, ca -25%. Refuted; auto's pick stands for b4.
- **b8 variant pick re-sweep** (5 shapes x2 interleaved, GGUF layout,
  `msweep-q27-b8-r2-vs-auto-allshapes.log`): auto's unconditional r2w8 is wrong on THIS rig
  for 3 of 5 shapes (r2 wins ffn_gate -7%, attn_qkv -6%, attn_gate -20%) and right for 2
  (ffn_down, ssm_out). No clean rule at these margins (the b4 wave-crossing transplant
  scores 3/5); net pass effect ~1% on the off-optimum T=5-8 tiers; live dispatch rides the
  rp family anyway. Parked as receipts, not promoted.
- **Boost-vs-sustained clocks lesson (re-learned locally)**: single-process msweep cells
  disagree up to 30% with cross-process cells on this laptop rig (boost decay). Every
  verdict above is from within-window interleaved pairs only; the pass-level probe
  (10s+ sustained, arms interleaved per-iteration) is the price authority.

## 6. Where the verify cost actually goes (nsys attribution)

Single-arm probe runs (`MEMRA_ECON_ONLY`, 15+3 iters/arm, kernel sums in `logs/nsys-*csv`;
per-pass numbers below exclude the prime/setup kernels which are fixed-count in these logs):

- **q27 verify T=4 pass (~28.2ms)**: batched b4 trunk matvecs **~22.1ms = ~78%**
  (`_b4_rpr2` 10.7ms/144 launches + `_b4_rpr2w8` 8.5ms/176 + `_b4_rp` 2.9ms/176), FA rows +
  combine ~2.4ms (~8%), Q5_K lm-head b4 1.5ms, glue (silu/quantize/norm/add) the rest.
  Decode T=1 has the same structure (m=1 `mr2_rp`+`dual_mr2_rp` ~80% of the step) — the
  verify premium IS the b-tier's per-column premium, exactly where §1's msweep put it.
- **q35 verify T=3 pass (~8.0ms)**: expert kernels ~2.4ms (CSR gate_up 1.5 + down rows 0.8),
  Q8_0 trunk b4 tiers ~2.4ms (fused2_b4 1.3 + b4 0.8 + fused3_b4 0.3), router 1.2ms,
  FA rows+combine 0.8ms. The slope vs decode is spread across expert re-reads (down),
  the b4 premium, and the T-scaled glue — no single kernel to attack; the structural MoE
  story (per-token expert reads; CSR-down twice-refuted with mechanism) stands.

## 7. Contract verdict

- kernel-check: **ALL GREEN on the final tree — 332 OK / 0 FAIL / 2 SKIP**
  (`logs/kc-final.log`; the dual-carrying intermediate trees were green too:
  `logs/kc-dual.log`, `logs/kc-dual-rp.log` with 12/12 bitwise dual entries).
- run-spec K=1..8 self-consistency: **PASS 72/72 baseline** (`logs/sweep-*.log`),
  **PASS 72/72 on the shipped tree** (`logs/after-sweep-*.log`), PASS 12/12 on the live
  A/B cells (`logs/spec-q27-k3-dual*`); **acceptance counts bit-identical** across reps,
  across dual on/off, and across before/after — zero drift, as the dispatch-parity law
  requires. Same draws, same tokens.

## 8. Before/after e2e (the shipped dual, same 9-run battery re-run on the final tree)

N=3 medians per cell, plain-vs-spec interleaved within every process (run-spec runs the
plain oracle then spec in one load), before = `logs/sweep-*`, after = `logs/after-sweep-*`;
acceptance counts BIT-IDENTICAL at all 24 (class, K) cells — same draws, same tokens:

| cell | before | after | delta |
|---|---|---|---|
| q27 prose K=1 | 68.78 | 69.73 | +1.4% |
| q27 prose K=2 | 87.97 | 89.08 | +1.3% |
| q27 prose K=3 (opt) | 90.01 (2.00x) | **91.14 (2.00x)** | **+1.3%** |
| q27 code K=1 | 75.04 | 76.12 | +1.4% |
| q27 code K=2 | 92.36 | 93.76 | +1.5% |
| q27 code K=3 (opt) | 101.40 (2.18x) | **101.86 (2.18x)** | +0.5% |
| q27 K>=4 (dual off — T>=5) | — | — | flat (±0.5%) |
| q35 all K (dual unreachable — MoE, no Dense FFN) | — | — | ±1-2% window noise |

The dual's tiers are exactly K=1..3 (verify T=2..4); K>=4 flat confirms the m<=4 gate; q35
is untouched by construction (the call site sits in the Dense-FFN arm only — its drift here
is cross-battery window variance on a 3-lane shared rig, acceptance identical). The
published board spec rows (2026-07-18 re-pairing protocol) are a different harness; a
post-merge board re-pair would be the place to claim any number movement.

## Files

`run-econ.sh` (driver: econ/sweep/msweep/nsys phases), `parse-econ.py` (tables + payoff
model), `crates/memra-engine/src/bin/spec_econ.rs` (the probe), `logs/`: econ probes
(`econ-*.log` + `[econ-json]`), baseline sweeps (`sweep-*-r{1,2,3}.log`, spec-phase +
spec-stats in-log), after sweeps (`after-sweep-*-r{1,2,3}.log`), dual A/Bs (void-arm:
`econ-q27-dual-ab.log` + `spec-q27-k3-dual-{on,off}-r*`; engaged: `econ-q27-dualrp-ab.log`
+ `spec-q27-k3-dualrp-{on,off}-r*`), variant sweeps (`msweep-*.log`), kernel-check
(`kc-dual.log`, `kc-dual-rp.log`, `kc-final.log`), run-gen argmax (`rungen-q27-final.log`,
MATCH both gates), nsys (`nsys-*`). Every log carries GPU brackets (temp/power/clock/mem +
compute-apps).
