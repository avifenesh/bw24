# Hy3 measured-effect 100GB allocation

## Question

Can an exact-format, globally optimized mixture of per-projection Q8/NVFP4/IQ4_XS/Q4_K/IQ3_S/Q3_K/Q2 and whole-expert
pruning retain more quality at the same 100GB ceiling than the first traffic-ranked prune + joint
heal arm?

The first arm proves that joint expert/router healing helps, but its quantization tiers were frozen
before pruning and were based mainly on traffic. It cannot identify whether a particular layer,
expert, or gate/up/down projection deserves the next byte.

## Frozen private evidence

The map uses only the existing 24-request, six-stratum private calibration corpus:

- 19,060 teacher-forced tokens and all 79 MoE-layer inputs;
- exact runtime top-8 expert ids and combine weights;
- teacher MoE output targets;
- REAP contribution, traffic, functional uniqueness, and per-stratum mass per expert;
- domain-local low-confidence correct-token rescue mass;
- current-mask joint-heal holdout reconstruction receipts per layer.

Public capability, SWE, and Terminal results are excluded from mapping, allocation, and healing.

## Missing measurement and effect map

`tools/build_hy3_quant_sensitivity.py` applies the same quantizers used by the GGUF-oriented
artifact builder, dequantizes their exact bytes, and evaluates deterministic routed-token samples.
For every `(layer, expert, qtype)` it records:

- joint expert-output squared error and normalized MSE;
- gate-only, up-only, and down-only output error;
- exact encoded bytes and weight reconstruction error per projection;
- routed-token count, sampled router mass, baseline contribution energy, and sample scale.

The derived effects map also preserves per-layer/per-projection damage and aggregate equal-byte
win counts by projection. This makes Q3_K versus IQ3_S and NVFP4 versus Q4_K directly auditable
without inferring a winner from nominal bit width or a small list of top outliers.
It also reports top-1/top-10/top-100 error share, Herfindahl concentration, and effective error-cell
count per format and projection. A private objective dominated by one routed function is therefore
visible before treating its byte allocation as a generally useful model-quality policy.

The base map has 15,168 expert rows and four precision alternatives. The extension measures
IQ3_S, IQ4_XS, and Q4_K on the same frozen routed samples using the current pinned upstream ggml C
quantizers and private per-column activation importance. Eight disjoint layer shards are merged
only if calibration, source, measurement policy, and complete coverage match; the two full maps are
then joined only if their routed evidence is identical. Library SHA, upstream commit, sidecar hashes,
and sidecar shapes are carried through the plan, heal receipts, artifact manifest, and validator.

## Global allocator

`tools/build_hy3_smart_budget_plan.py` solves one global integer program. Each retained
gate/up/down projection independently chooses any exactly measured format. The seven-format extension
adds IQ3_S, IQ4_XS, and Q4_K beside Q8, NVFP4, Q3_K, and Q2; equal-size Q3_K/IQ3_S
and Q4_K/NVFP4 alternatives compete on measured error without a precision-order assumption. Pruning is a shared
whole-expert decision, so a pruned expert removes all three projections. Constraints enforce:

- logical size at or below 100,000,000,000 bytes;
- at least 96 surviving experts per layer and therefore well above runtime top-8;
- retention of every private per-stratum protected expert;
- exactly one precision for every retained projection.

The base objective is measured router-weighted output squared error scaled to the full routed-token
count. Optional multipliers protect REAP/domain importance, correct low-confidence rescue experts,
and layers where the current mask causes high teacher reconstruction error.

Before solving, the allocator subtracts each expert projection's lowest measured retained-format
error from every format choice and subtracts the three corresponding minima from that expert's
prune choice. Every feasible solution therefore loses the same constant and the exact optimum is
unchanged. The centered objective prevents a large unavoidable reconstruction floor in one
layer/expert/projection cell from consuming the MILP relative-gap tolerance and making the remaining
precision choices effectively arbitrary. The plan records the removed constant, centered scale,
centered objective, and reconstructed absolute objective.

The integer program is exact for bytes and pruning, but its mixed-precision objective is an additive
sum of gate-only, up-only, and down-only ablations. It does not claim that this sum is the exact joint
error of every mixed `(gate, up, down)` tuple: the SiLU gate and up projection interact
multiplicatively. Therefore an allocation is only an optimizer proposal. Promotion additionally
requires terminal requantization followed by the joint expert-output holdout gate, the full-corpus
routing audit, and matched capability evaluation. If a plan's predicted error-per-byte improvement
does not survive those joint gates, the next measurement extension is gate/up pairwise interaction
damage rather than another public-eval-tuned allocation.

## Three candidates

All candidates have the same byte ceiling, private calibration, exact quantizer measurements, and
joint rank-8 expert + F32 router/bias healing. Only the frozen objective weights differ:

| Arm | REAP/domain | low-confidence rescue | layer damage | Purpose |
|---|---:|---:|---:|---|
| `smart100_empirical` | 0 | 0 | 0 | Pure measured error-per-byte baseline. |
| `smart100_balanced` | 1 | 1 | 1 | Equal protection for function, hard-token rescue, and depth. |
| `smart100_rescue` | 0.5 | 2 | 1 | Test whether rare hard-token experts deserve disproportionate precision. |

The existing `prune100_joint_heal` remains the traffic-ranked control. A single
`smart100_iq3_iq4_q4_empirical` extension re-solves the pure measured-error objective with all seven
formats at the same exact 100GB ceiling. It is promoted only if its matched held-out evidence adds
to or improves the current Pareto frontier.

Healing uses a private-holdout monotonic selector. Every layer is trained and terminally requantized,
but a trained overlay is retained only when its frozen private holdout normalized MSE is strictly
lower than the unhealed quantized layer. A non-improving layer emits the exact source expert/router/
bias values whose one-pass artifact repack recreates the unhealed baseline. Receipts preserve both
the rejected trained metrics and the selected metrics; public evaluation is never used for this
decision. This prevents the systematic late-layer regressions observed in the first always-trained
healing pass from consuming capability budget.

## Allocation comparison receipt

`tools/summarize_hy3_smart_allocations.py` expands every plan into the canonical
`(layer, expert, projection) -> {Q8_0, NVFP4, IQ4_XS, Q4_K, IQ3_S, Q3_K, Q2_K,
PRUNED}` map before healing starts. It writes `allocation-comparison.json` beside the plans with
each plan and allocation hash, exact
per-layer tier counts, per-projection qtype counts, ranked `(gate, up, down)` format combinations,
uniform-versus-mixed expert counts, and pairwise qtype/prune transitions. Historical whole-expert v2
plans are accepted for comparison, but absent historical byte totals remain explicitly null rather
than being reconstructed.

For archived analysis, pass `--receipt RECEIPT --analysis-commit COMMIT`. The output and receipt
must both be new paths. The receipt binds the exact analysis commit, script SHA-256, ordered plan
paths and SHA-256 values, output path and SHA-256, and the private-only selection declaration. This
keeps the finalizer input reproducible and prevents a later automation pass from silently replacing
an allocation map.

The smart build uses `--require-distinct`: two objective recipes that produce the same allocation
are not separately healed or evaluated. This prevents benchmark noise from being mistaken for an
allocation effect and keeps every scored arm a genuinely different compression hypothesis.

`tools/summarize_hy3_plan_agreement.py` also writes a reproducible consensus map: pairwise pruning
overlap, exact quant/prune-state agreement, stable per-projection quant counts, and layer-level
disagreement. Its optional private retention-score overlay records routed weight mass and frequency
for every pruned/quantized state, preventing expert-count compression from being mistaken for
negligible functional traffic. This separates structural decisions that survive all private priors
from precision choices that still require held-out capability evaluation.

## Evaluation order

1. Validate plan coverage, distinct allocation hashes, exact bytes, source hashes, and zero routing
   to pruned ids.
2. Jointly heal each plan against the same teacher targets; require finite loss, no dead active
   experts, and immutable per-layer receipts.
3. Run the matched 115-question capability panel on all three in parallel with the existing control.
4. Promote only point-estimate Pareto leaders without task collapse to the practical panel.
5. Run the trusted 4,746-question capability suite and full SWE500/Terminal89 only after this map is
   resolved.

## Current layer signal

The first joint-heal receipts already show that uniform per-layer pruning is unsafe. Current-mask
prune damage correlates strongly with experts removed (`r=0.85`) and decreases with depth
(`r=-0.68`). Layers 14, 18, and 19 have holdout normalized MSE around 0.40 before healing, while
several later layers remain near 0.05 despite far fewer removals. Layer 79 is an exception: 96
experts can be removed with only 0.0024 normalized MSE. The allocator therefore uses measured
layer effects rather than a monotonic early/late rule.

## Frozen recipe audit

The final private allocation maps show that scalar objective multipliers are not enough to enforce
that intent. The base empirical, balanced, rescue, and seven-format Pareto recipes all prune the
maximum 96 of 192 experts in layers 14, 18, and 19. Their retained projections in those layers are
almost entirely Q2:

| Plan | Layer 14 non-Q2 projections | Layer 18 non-Q2 projections | Layer 19 non-Q2 projections |
|---|---:|---:|---:|
| base empirical | 0 | 0 | 1 Q3_K |
| balanced | 3 Q3_K | 0 | 0 |
| rescue | 4 Q3_K | 0 | 2 Q3_K |
| seven-format Pareto | 0 | 0 | 1 IQ3_S |

The same seven-format Pareto plan spends 397 Q8/Q4_K/IQ4_XS projection choices in layer 79 while
pruning only 14 experts there. This is consistent with the global private objective: one Q2 cell,
layer 79 expert 120 down-projection, contributes 91.86% of total Q2 error. The allocator correctly
protects that cell with Q8, but a single absolute-error hotspot can dominate the global byte trade
even though the layer-level prune holdout says earlier layers are more structurally fragile.

This rules out merely increasing the existing layer or rescue scalar as a clean next experiment.
A future private-only allocation should instead preregister a structural control such as a
layer-normalized objective, a cap on per-cell objective share, or explicit per-layer retention and
damage budgets. It must be frozen against a new private holdout before another public capability
screen; public task failures must not be used to choose those bounds.
