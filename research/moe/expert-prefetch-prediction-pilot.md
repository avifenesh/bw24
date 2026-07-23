# Prefetch-prediction pilot: cross-layer router application — POSITIVE (2026-07-23)

The lead's cheapest measurement, upgraded to zero-training: apply layer j's actual router
(`mlp.router.gate.weight` + `expert_bias`, sigmoid scoring) to layer k's captured router input
and compare the predicted top-8 with the routes the model actually took. Data: 173 decode
steps captured on the local 5090 Hy3 Layer-103.5 profile (`BW24_MOE_TRACE` +
`BW24_MOE_INPUT_TRACE_DIR`, capture log in
`../per-expert-quant/evidence/local-5090-next3-20260722/probe-capture.log`; routes copy at
`probe-capture-routes-20260723.trace`; hidden-state payloads 272 MB, session-scratch only).

Mean top-8 overlap (% of the true top-8 predicted), [argmax-in-true-set %]:

| k \ d | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| 8  | 63 [66] | 53 [65] | 45 [56] | 39 [73] |
| 16 | 64 [76] | 55 [64] | 51 [76] | 23 [13] |
| 24 | 49 [27] | 50 [72] | 41 [55] | 32 [46] |
| 32 | 48 [38] | 54 [66] | 51 [75] | 47 [83] |
| 40 | 60 [72] | 53 [71] | 48 [47] | 33 [55] |
| 48 | 62 [78] | 58 [87] | 53 [84] | 49 [86] |
| 56 | 54 [63] | 66 [80] | 66 [91] | 58 [87] |
| 64 | 75 [84] | 74 [99] | 70 [94] | 61 [98] |
| 72 | 83 [99] | 75 [100] | 59 [91] | — |

Verdict: clears the lead's confidence-gated top-1/top-2 bar in the deep half of the network
(k≥48: argmax-hit 84–100% at d=1–4; lead time d=2 ≈ 5 ms vs ~1 ms per predicted-expert read).
This is the first measured mechanism whose information is not already captured by the LRU
cache (history-based predictors: 28–40%, all flat in e2e).

Build plan (increment 1): engine computes d=1–2 lookahead scores (one 192x4096 matvec per
lookahead layer on the copy stream + async DtoH), confidence-gated top-1/2; a new companion
prefetch entry hands predicted-and-likely-missing projections to the parked IoPool, reading
into the RAM cache as lowest-priority insertions with separately-accounted traffic (the
lead's misprediction rules). Risk to A/B: prefetch DMA re-introduces concurrent-io-under-
compute at trickle volume — the fabric-interference regression was measured at full rate;
the arm must watch phase_compute, not just the hit rate.

## Increment-2 A/B: REJECT as-built (2026-07-23, arm-pf-* logs in local-5090-next3-20260722)

pf-off 4.74/4.75 vs pf-on 2.90/2.88 (depth=2, top=2, ungated). Mechanism autopsy from
counters: (1) no cross-token dedup — the same predicted experts re-prefetched every token,
~14k speculative projection reads per 32-token window (~37 GB, exceeding demand io);
(2) cold LRU-front insertion self-cancels at a full cache — demand fills evict speculative
entries before their layer arrives, so they are re-read in a loop; (3) no depth/margin gate —
the pilot's precision only justifies k>=48; (4) the concurrent-DMA compute tax reappears at
this volume (phase_compute 2.87→3.49 s). Predictor plumbing itself is sound (6083 submissions,
0 drops, decode thread unblocked).

Redesign targets before the next arm: a small prefetch ANNEX outside the LRU (speculative
entries never enter or evict the main cache; promoted on first demand hit), cross-token dedup
with an in-annex/in-flight check, deep-half + score-margin gating, top-1 only, and a byte
budget per token (~a quarter of demand io) so speculation can never dominate the bus.

## Increments 3-4 and lane CLOSURE (2026-07-23)

Annex redesign (speculation isolated from the main cache, cross-token dedup, deep-half gate,
E-core-pinned predictor) A/B'd twice more (pf3-*/pf4-* logs): still negative, and the
decode-window demand counters never moved (RAM 15736/10691 in every arm — zero annex
promotions despite 39 GB of completed speculative reads). Mechanism: a d=1-2 lookahead is
~2.6-5 ms of lead, but a queued 2-4 MB O_DIRECT read under load takes tens of ms — demand
beats every prefetch to its own expert, and the annex expires the late arrivals.

The closing measurement is a lead-time/precision scissors, from the capture itself:

| lead | non-resident precision |
|---|---|
| d=1-2 (2.6-5 ms — cannot beat read latency) | 55-75% (k>=64) |
| d=16 (42 ms — beats read latency) | 10-34% |
| d=24-32 (62-83 ms) | 10-17% |

Prediction-guided expert prefetch on this box is therefore closed: the window where the
router signal is strong is too short to move MB-scale weights, and the window long enough to
move them carries no signal. This holds on top of the invariant fabric tax (speculative DMA
inflates compute in every arm, 2.87 -> 3.4-12.6 s). The predictor/annex machinery stays in
the tree env-off (correct, gated, bit-identical) as scaffolding. Resurrection bar: hardware
with spare bus bandwidth relative to compute (desktop 5090 + faster DRAM/NVMe), KB-scale
fetch granularity (sub-expert/projection-fragment storage), or a fundamentally better
long-lead predictor (trained probe on deeper context, not the router cross-application).

Per the lead's own protocol: negative result recorded, lead dropped.
