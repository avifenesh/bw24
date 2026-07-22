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
