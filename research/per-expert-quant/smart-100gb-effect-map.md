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

## Allocation comparison receipt

`tools/summarize_hy3_smart_allocations.py` expands every plan into the canonical
`(layer, expert, projection) -> {Q8_0, NVFP4, IQ4_XS, Q4_K, IQ3_S, Q3_K, Q2_K,
PRUNED}` map before healing starts. It writes `allocation-comparison.json` beside the plans with
each plan and allocation hash, exact
per-layer tier counts, and pairwise qtype/prune transitions. Historical whole-expert v2 plans are
accepted for comparison, but absent historical byte totals remain explicitly null rather than
being reconstructed.

The smart build uses `--require-distinct`: two objective recipes that produce the same allocation
are not separately healed or evaluated. This prevents benchmark noise from being mistaken for an
allocation effect and keeps every scored arm a genuinely different compression hypothesis.

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
