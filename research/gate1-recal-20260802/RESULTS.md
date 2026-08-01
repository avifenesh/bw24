# gate1-config per-rig recalibration — RTX 5090, 2026-08-02

Lane: `lane/gate1-recal` (base `restructure/public-split` @ 5cfad376). Rig: RTX 5090
Laptop (sm_120a, 24GB), every GPU run under `flock /tmp/gpu5090.lock`. Issue: #47.

## 1. Why

decode-batch-gate's config-mode gate1 rule ("FAIL iff ANY of 6 draws diverges before
step 3", calibrated on H100 2026-07-31, round 45) had two receipted pre-existing 5090
fails that gate2 — the actual isolation contract, at bit strength — PASSes:

- `research/batched-tick-inc3-20260801/gate1-seed-sweep.log` — 18-draw q9j sweep,
  15/18 perfect, strict-mode green; seed 0 diverges at step 1.
- `research/prime-gate-coverage-20260802/battery-decode-batch-q9j-BASE.log` — the same
  seed-0 step-1 signature bit-identical on the pristine base.

The round-45 rule assumed "observed tie flips start at step 6+" — an H100-only
observation.

## 2. Characterization: divergence-step distribution (unmodified binary)

18 draws per model (`MEMRA_GATE_SEED` in {0,6,12} x 6 internal draws), this tree,
`sweep-*.log`:

| rig / model | draws | first-divergence steps (seed) | early (<3) | worst 6-window early |
|---|---|---|---|---|
| H100, round-45 ledger (pre-change tree) | 6 | 7, 8, 15 | 0/6 | 0 |
| 5090 q9j (Qwen3.5-9B judge Q8_0, dense) | 18 | 1 (s0), 3 (s3), 19 (s9) | 1/18 | 1 (base 0) |
| 5090 q35 (Qwen3.6-35B-A3B IQ4_XS, MoE) | 18 | 6 (s0), 4 (s2), 6 (s3), 7 (s10), 28 (s11), 16 (s12), 0 (s16), 0 (s17) | 2/18 | 2 (base 12) |

New find: q35 seeds 16/17 flip at **step 0** (`sweep-q35-base12.log`) — on this rig the
near-tie dice reach the very first decode step. Every run's gate2 (B=4 vs isolated,
bit-checked) and gate3 stayed PASS.

**Dice proven, not assumed** (`strict-equalized-*.log`): the exact worst draws — q35
seeds 16 and 17, q9j seed 0 — are **bit-identical for all 32 steps** under the equalized
composition (`MEMRA_MMVQ=0 MEMRA_NO_FUSE_NORMQ=1 --mode strict`), gates 1/2/3 ALL GREEN.
The config-mode flips are the accepted FP-composition near-tie class, not plumbing. #47
confirmed with the strongest discriminator this gate owns.

## 3. The recalibrated rule (fraction-of-seeds)

**FAIL iff >= 4 of the 6 draws diverge before step 3.** Strict-mode bit-identity, gate2,
and gate3 are untouched — they remain the hard exactness floor.

Why this shape and not the alternatives:

- **Per-arch/per-rig step threshold** — refuted by the data: 5090 dice land at step 0
  (q35) and step 1 (q9j), so any per-draw threshold that passes legal dice passes
  step-0/1 plumbing too. The step dimension carries no rig-invariant signal per draw.
- **Fraction rule (chosen)** — the plumbing class (wrong token fed, KV misindexed)
  diverges at step 0-2 on EVERY draw, on every rig; observed dice reach at most 2 early
  draws per 6-window (q35 base 12). K=4 sits margin 2 above the observed dice maximum
  and margin 2 below the plumbing floor (6/6). The step<3 window keeps the round-45
  plumbing-class definition.
- **Informational-only without a rig-calibration marker** — rejected: it removes teeth
  exactly where plumbing risk is highest (first run on a new rig), and the fraction rule
  needs no marker because the every-draw signature is rig-invariant.

## 4. Verification (`verify-*.log`)

- Recalibrated gate ALL GREEN on both models x all three seed bases — including both
  receipted failure conditions (q9j `MEMRA_GATE_SEED=0`, q35 `MEMRA_GATE_SEED=12`).
- Strict-equalized re-probe post-edit (q9j seed 0, q35 seed 16): ALL GREEN — the edit
  did not touch the bit floor.
- **Canary keeps the teeth**: `MEMRA_GATE_CANARY=1` (test-only door: feed the batched
  lane one wrong token at step 1) FAILs on both models, 6/6 draws early, exit 1
  (`verify-canary-*.log`).

## 5. Battery (`battery-*.log`)

- decode-batch-gate default env, new rule: q9j ALL GREEN, q35 ALL GREEN (gates 1/2/3).
- kernel-check: untouched, ALL GREEN.

## 6. Files

`run-sweep.sh` + `sweep-{q9j,q35}-base{0,6,12}.log` (characterization),
`run-strict-probes.sh` + `strict-equalized-*.log` (dice proof),
`run-verify.sh` + `verify-*.log` (recalibrated rule + canary),
`run-battery.sh` + `battery-*.log` (ship battery).
