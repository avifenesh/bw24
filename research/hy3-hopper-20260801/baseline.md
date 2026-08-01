# Hy3 single-H100 spill-regime baseline — Mumbai, 2026-08-01

Protocol: board depth prompt (`research/gemma4-bringup/depth-prompt-1736.txt`, the H100
board's d1736 text; **1818 tokens** under Hy3's tokenizer — board-2048-class), greedy
`run-gen`, `MEMRA_NGEN=128`, fresh process per run, every run under
`flock /tmp/gpu-h100.lock` (shared box, other lane interleaving allowed between runs).
N=3 same-config repeats, same session, same thermal regime (steady-state box, GPU
otherwise idle between locks). Raw logs: `logs/baseline-r{1,2,3}.log`,
`logs/probe-n256.log`, battery driver `logs/battery.log`.

## Result (N=3)

| run | decode tok/s (128 tok, ST greedy) | argmax gate | logit maxdiff |
|---|---|---|---|
| r1 | 2.47 | `verify-prefill argmax=478 decode argmax=478 MATCH` | 1.888e0 |
| r2 | 2.48 | MATCH (argmax=478) | 1.888e0 |
| r3 | 2.50 | MATCH (argmax=478) | 1.888e0 |

**Median 2.48 tok/s, N=3, spread 1.2%.** Identical argmax and identical maxdiff across
runs — the decode path is run-to-run deterministic on sm_90a for this artifact.

Decode-window cache state (r1, representative): 6476 SLRU slots, hit-rate 14.2% at depth
(the 1818-token prefill floods the window), staged 3.9 GB/token H2D. Storage physical
reads ~0 (the 96 GiB artifact sits in the 249G page cache after first touch) — per the
staging rule this is NOT an EBS/NVMe fault-throughput number and is not reported as one.

Wall clock per run ~17.5 min, dominated by the spill-regime prefill of 1818 tokens
(every layer restages most of its expert bank). Prefill throughput is the single-GPU
pain point, not decode.

## Graph-door probe (single run, labeled)

`MEMRA_NGEN=256` (the `MEMRA_GEN_GRAPH` budget threshold): 256 tokens in 102.442s =
2.50 tok/s, argmax MATCH, exit 0. No crash at the graph budget on sm_90a; the log prints
no graph-capture line, so graph *engagement* under SLRU staging is not confirmed — carried
as a follow-up check for the spike, not claimed.

## What this number is for

This is the **1-GPU degenerate point** for the spike's $/Mtok table: a 295B-A21B mixed-tier
Hy3 that does NOT fit one 80GB H100 decodes at ~2.5 tok/s expert-staging-bound. It is the
floor the PP-2 replica must destroy (bank fully resident across 2x80GB removes the
1.9-3.9 GB/token staging term entirely). It is NOT a memra-vs-anyone comparison and NOT a
spill-storage benchmark.
