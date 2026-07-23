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

## Lane 3 seam recon (2026-07-23, full map in agent transcript; key facts)

- `Cache` is hard single-sequence (scalar `len`/`pos`, one recurrent state per layer) — but
  v1 lockstep AVOIDS the refactor: m independent `Cache` objects, one per stream.
- Attention stays per-stream (existing `full_attn_decode`/`linear_attn_decode` calls, T=1
  each) — no block-diagonal mask needed until attention batching becomes worth it
  (`fa_decode_rows` per-row key ranges are the seam when it does).
- `moe_ffn_grouped` (hybrid_forward.rs:2742) gathers/scatters by flat row index, stream-
  agnostic — reusable for the cross-stream MoE stage, GPU side.
- CPU companion ABI is one-row-per-call: within-step io amortization needs NO ABI change
  (stream 1's miss fills the shared RAM cache; siblings hit). Weight-decode compute
  amortization needs a multi-row ABI v3 — deferred to M3.

Increments (each battery-gated):
- **M1**: `decode_step_lockstep(streams)` — per-layer walk, per-stream mixers, per-stream
  sequential MoE (correctness first: each stream's tokens identical to its single-stream
  run; aggregate baseline measured).
- **M2**: cross-stream MoE stage — route all m rows, dispatch CPU experts stream-ordered so
  shared-cache reuse lands within the step; measure io amortization vs the 1.12/1.32/1.66x
  curve.
- **M3**: companion ABI v3 multi-row-per-expert + GPU grouped dispatch across streams.
- **M4**: m=4/8 scaling, serve loop, per-stream latency reporting.

## M1 first gate (2026-07-23, lockstep-m*.log)

`decode_step_lockstep` + `run_lockstep`: per-stream math is `decode_step_h`'s; the harness
gate requires all streams (same prompt) to emit identical tokens — PASS at m=1/2/4.

| m | aggregate | per stream | whole-run cache hit rate |
|---|---:|---:|---:|
| 1 | 4.44 | 4.44 | 58.6% |
| 2 | **6.10 (+37%)** | 3.05 | 67.4% |
| 4 | 2.52 (COLLAPSE) | 0.63 | 72.0% |

Cross-stream cache amortization works (hit rate climbs with m; m=2 beats the 1.12x overlap
prediction because GPU-side residency adjacency stacks on top). The m=4 collapse is a
per-step nonlinearity (step wall 328 ms -> 1590 ms), not RAM (RSS flat, zero swaps) —
prime suspect is VRAM-edge pressure from 4 streams' recurrent states allocated after the
0.90-frac expert slab sized itself; discriminator arms (m=4 at frac 0.85/0.80, m=3 at 0.90)
in flight.
