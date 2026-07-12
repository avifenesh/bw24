# Hy3 exact-100GB prune and heal screen

## Question

Can a strong per-expert quantized Hy3 candidate be reduced to at most
`100,000,000,000` logical bytes by pruning whole routed experts, then recover useful capability by
healing both the router and the surviving experts before applying the final quantization layout?

This is a causal screen, not a full benchmark. Public evaluation examples are excluded from expert
selection and healing.

## Locked invariants

- The target is a hard decimal-byte ceiling. Report exact logical bytes and GiB.
- Start from the pinned high-precision source. A quantized artifact supplies only the promoted tier
  layout and the byte model; it is never the training source.
- Prune whole expert ids. The three projections of an expert live or die together, and original
  router ids remain stable.
- Select one survivor mask before healing. The unhealed, router-only, and joint-healed arms use the
  same mask and the same final quantization tiers.
- Quantization happens after healing.
- Keep at least 96 experts per MoE layer initially, comfortably above top-k 8. Tighten this floor
  only in a separately locked follow-up.
- Freeze source revision, private calibration revision, traces, teacher targets, score components,
  mask, training configuration, checkpoints, and final artifact manifests before evaluation.

## Candidate selection

Run the first screen on the strongest promoted compact candidate from the locked ten-arm expanded
screen. If joint healing is directionally positive, repeat only the winning recipe on the second
compact Pareto candidate.

The selection score is private and combines:

1. REAP-style expert saliency.
2. Router-weighted traffic across the private calibration strata.
3. A diversity term that protects experts whose outputs are not well represented by survivors.
4. A rare-domain floor so low-traffic specialists are not removed solely because they are cold.

The selector maximizes retained score subject to the per-layer survivor floor and the exact final
artifact byte ceiling. It must use tensor bytes from a source-verified manifest, not nominal bit
rates or a guessed prune percentage.

Capture the MoE input hidden states during the same sequential private calibration pass that writes
the weighted route trace. `BW24_MOE_INPUT_TRACE_DIR` is diagnostic-only and writes one f32 payload
per layer plus `index.jsonl`. Validate exact request/layer/token coverage, contiguous offsets,
finite values, file sizes, and SHA-256 hashes with `tools/validate_moe_input_trace.py` before scoring.
The REAP component uses the actual HyV3 combine coefficient: sigmoid router weights selected after
expert-bias correction, renormalized over top-k, then multiplied by the model router scaling factor.
Freeze the first composite before evaluation at 0.65 REAP, 0.10 total router-weight traffic,
0.15 activation-signature uniqueness, and 0.10 rare-stratum specialization. Protect the top two
router-mass experts per private stratum and layer as hard solver constraints. Score layers in eight
disjoint GPU shards and require `tools/merge_expert_score_shards.py` to prove exact, non-overlapping
1--79 coverage before the byte solver consumes the result.

## Matched arms

| Arm | Trainable parameters | Purpose |
|---|---|---|
| `prune100_unhealed` | none | Isolate the loss from the frozen survivor mask. |
| `prune100_router_repair` | router matrix and F32 selection bias | Measure cheap routing realignment against the frozen teacher MoE output. |
| `prune100_joint_heal` | router and surviving experts | Primary recovery arm; repair expert functions and routing together. |

Use precomputed teacher targets from the private corpus. Heal every pruned MoE block independently
against its frozen layer teacher target, and preserve per-layer train/holdout reconstruction metrics
in the receipts. The router repair minimizes survivor-model output reconstruction with a
source-router anchor. The F32 selection bias receives bounded load-feedback toward the original
survivor traffic; do not force uniform routing. Joint healing adds rank-8 LoRA updates to each
surviving gate/up/down expert and merges them before final quantization.

The final GGUF-oriented expert overlay must embed the healed router matrix and selection bias as
explicit F32 tensor overrides (`ffn_gate_inp.weight` and `exp_probs_b.bias`). Falling through to the
base artifact would silently restore the original router and invalidate both healed comparisons.

## Gates

Before public directional evaluation, every arm must pass:

1. Exact survivor-mask and tier-plan equality across arms.
2. Logical bytes `<= 100,000,000,000` and byte-identical staged artifact validation.
3. No routed pruned ids; finite router probabilities; at least top-k active experts per layer.
4. Short and long prompt health gates with no MTP, speculative decoding, or KV reuse.
5. Private held-out teacher divergence and routing-collapse report.

The pre-evaluation quality thresholds are frozen in
`100gb-heal-quality-gates.lock.json`. Both healed arms must retain at least half of their original
routing entropy, keep dead survivors and maximum expert load below the locked safety ceilings, and
the joint arm must improve mean held-out normalized MSE with at least half of layers improving.

Promote joint healing only if it beats the unhealed control directionally without a domain collapse.
Router-only is an ablation and is not promoted merely because it is cheaper. Only promoted Pareto
leaders proceed to practical SWE/Terminal triage and trusted full suites.

## Current byte implications

The non-expert logical component is approximately `24,999,514,624` bytes, so a 100GB artifact has
about `75,000,485,376` bytes available for routed-expert tensors.

- `mix_quant` starts at `150,249,820,672` logical bytes and must remove `50,249,820,672` expert
  bytes. A naive cold-tier estimate is roughly 47% of experts, but the exact selector decides.
- `traffic_q8_q2_no_prune` starts at `136,457,376,256` logical bytes and must remove
  `36,457,376,256` expert bytes, roughly 39% by the same rough count estimate.

These percentages are planning estimates only and must never become the frozen mask.
