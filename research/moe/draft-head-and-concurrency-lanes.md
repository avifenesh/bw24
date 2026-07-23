# Owner-selected lanes (2026-07-23): trained draft head + multi-request serving

## Lane 1 — trained draft head for Hy3 (EAGLE-class)

No existing Hy3 draft/EAGLE head on the Hub (searched 2026-07-23). The shipped layer-80 MTP
head ceilings at 77% gated acceptance (owner bar: 0.80; receipts in
`../per-expert-quant/local-5090-10toks-plan.md`). The gating framework (BW24_SPEC_PMIN,
K>2) is proven and waiting — a head drafting at 85-90% makes K=4 spec a 1.3-1.5x multiplier
on the whole stack.

Plan:
1. **Data**: self-distillation corpus — prompts through Hy3 itself (greedy, the serving
   regime), capturing per-token hidden state (pre-lm_head) + sampled token. The MoE input
   trace hooks already capture hidden states; extend to the final-norm state or reuse the
   layer-79 capture. Volume target: 50-100M tokens (EAGLE-3 recipes use ~68M).
2. **Architecture**: EAGLE-style single-layer autoregressive head over (hidden, token-embed)
   pairs, sized to Hy3's 4096 hidden. Reuse the existing MTP serving path in bw24 (the spec
   plumbing is arch-agnostic given a draft head that emits logits) — the head replaces the
   layer-80 weights, keeping the verify machinery untouched.
3. **Compute**: training runs on a rented research GPU (vast/G7e class per project rules —
   never the serving rig). Head is ~0.5-1B params: single-GPU trainable.
4. **Gate**: gated acceptance ≥0.80 at PMIN 0.7-0.85 AND net ≥1.15x vs plain at NGEN≥64,
   N=3 interleaved, before any default flip.

## Lane 3 — multi-request serving (expert-io amortization across streams)

Measured overlap (route trace, 300 trials/m): concurrent streams share routed experts at
**1.12x (m=2), 1.32x (m=4), 1.66x (m=8)** — per-token expert io and weight-decode cost
divide by these factors when execution is grouped by expert; GPU-side resident experts gain
additionally from batched matvec weight reuse. Projected aggregate at m=8: ~7-9 tok/s.

Build (in order):
1. **Recon**: prefill's `moe_ffn_grouped` (T>1 expert-grouped dispatch) and the verify pass's
   batched decode-like path — the two existing seams closest to lockstep multi-stream decode.
2. **Multi-stream state**: per-stream KV cache + GDN linear-attention state; block-diagonal
   attention for T=m lockstep steps (each stream attends only its own history).
3. **Scheduler**: m prompts advanced in lockstep; per-layer router batches m tokens; expert
   execution grouped by unique expert across streams (CPU companion already accepts
   arbitrary expert lists per call — one call per unique expert with m-row activation is the
   natural extension; the multi-token ABI experiment from 2026-07-21 was removed under
   winners-only but its receipt documents the grouping approach).
4. **Gate**: aggregate tok/s at m=2/4/8 vs m× single-stream baseline; per-stream latency
   reported alongside (aggregate wins must not hide unacceptable per-stream tails);
   correctness = each stream's output identical to its single-stream run.
