# Confidence-conditioned expert quantization

## Research question

Can expert precision follow *difficulty* rather than raw traffic: aggressively quantize experts
used on easy/high-confidence tokens, while protecting experts whose contribution is important when
the full model is uncertain but still correct?

This is deliberately a static artifact-selection experiment first. Same-token dynamic precision is
circular: logits expose confidence only after the MoE experts have already executed. A runtime
policy would need either a pre-expert proxy or a second-pass recompute, plus duplicate weight
representations. Do not add that systems cost before the static allocation is shown to improve
quality at a matched byte budget.

## Phase A: forward-only static allocation

Use a frozen, domain-balanced calibration set that contains reference continuations and excludes
every public evaluation item. Run the strongest unpruned baseline in teacher-forcing mode and
record, for each target token:

- reference-token log probability, top-1 correctness, top-1/top-2 margin, and entropy;
- selected expert IDs and router weights for every MoE layer;
- the existing gate-weighted expert-output norm used by the REAP-style trace.

Only *low-confidence, full-model-correct* tokens contribute to the primary criticality score. This
avoids protecting experts merely because they participate when the model is confidently wrong. A
separate diagnostic retains the wrong-token population so this choice can be audited.

For token `t`, define a bounded uncertainty weight from the reference-token margin:

```text
u_t = 1[argmax(p_t) == y_t] * min(u_max, 1 / (margin_t + epsilon))
```

For expert `e`, accumulate confidence-conditioned contribution:

```text
C_e = sum_t u_t * router_weight(t,e) * normalized_output_norm(t,e)
```

Independently measure each expert's weight reconstruction error under Q2_K, NVFP4, and Q8_0 from
the source weights. The allocation cost for format `q` is:

```text
D_(e,q) = C_e * normalized_weight_error(e,q)
```

Choose one format per expert by minimizing total `D_(e,q)` under an exact artifact-byte budget.
Use a global multiple-choice knapsack across all layer/experts, not identical per-layer quotas.
Phase A does **not prune**: pruning and precision allocation must not be confounded again.

## Matched controls

At each selected byte budget compare:

1. confidence-conditioned global allocation;
2. traffic-only allocation using the same format counts;
3. quantization-error-only allocation using the same format counts;
4. at least three random assignments with the same per-format counts;
5. the existing uniform/mixed and plain-quant baselines.

Freeze calibration hashes, format counts, byte totals, tie-breaking, and the plan before any public
evaluation. First use the locked hourish screen. Only candidates passing its existing promotion
gates reach the Harbor SWE/Terminal directional panels and then full trusted suites.

## Phase B: causal refinement if Phase A wins

The contribution-times-error proxy is intentionally cheap. If it wins directionally, refine only
the boundary experts using forward perturbations: substitute one candidate quantized expert output
on sampled low-confidence-correct tokens and measure reference-token log-probability loss. This
tests whether an expert is causally protective rather than merely correlated with difficult tokens.

Do not attempt an all-expert exhaustive perturbation. Restrict refinement to experts near the
knapsack precision boundaries; retain the Phase-A assignment elsewhere.

## Deferred runtime escalation

Dynamic precision is a later systems experiment, not part of the first quality claim. Two valid
forms are:

- pre-expert proxy: router entropy/margin or previous-token uncertainty selects a resident precision;
- two-pass escalation: run the low-precision token, then recompute only when final-logit confidence
  is below a frozen threshold.

Both require duplicate or residual weight storage and a matched latency/quality study. Neither is
allowed to replace the static test until it beats the static artifact at the same logical-byte
budget.

## Prior-art anchors

- AWQ establishes activation-aware protection rather than weight-magnitude-only allocation:
  https://arxiv.org/abs/2306.00978
- MxMoE jointly uses expert sensitivity, activation dynamics, and hardware cost:
  https://arxiv.org/abs/2505.05799
- MoPEQ uses expert-level Hessian sensitivity rather than activation frequency alone:
  https://arxiv.org/abs/2509.02512
- GEMQ allocates precision globally and accounts for router distortion:
  https://openreview.net/forum?id=wAc718O8UM
- STEP uses token-aware, direct loss-impact expert scoring:
  https://openreview.net/forum?id=Ty1Dflkz2J
- Recent theory argues that infrequent experts can carry rare but critical features and need more,
  not less, precision: https://arxiv.org/abs/2604.06515

## Implementation gate

Do not start confidence calibration while a matched evaluation or artifact build is active. After
the two no-prune traffic ablations finish, implement only the minimum capture additions needed for
`u_t` and `C_e`, freeze a new non-public calibration lock, and project the first plan at the better
of the 127.086 GiB and 139.128 GiB no-prune budgets.
