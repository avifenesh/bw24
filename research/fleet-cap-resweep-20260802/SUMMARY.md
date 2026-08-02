# Fleet cap re-sweep 8/12/16 + multi-tenant QoS probe — 8xH100 box, 2026-08-02

Phase-3 stale-verdict lever from `tools/box-aug2-mission.md` §2c: cap 8/replica was
calibrated on the v0.59 core; this binary (lane/m2-pp8 build, 2026-08-02) moved the tick.
Per the clock-drift law the **in-window cap-8 arm is the denominator** — the v0.60
reference row (1477.0 managed) is context only.

Setup: GPUs 5-7, 2 replicas/GPU (6 replicas), Qwen3.5-9B-Q8_0 (NVMe copy), proxy :8080.
Four passes (two sweep invocations of two passes each), caps interleaved pass-wise
(8→12→16 within every pass). Raw JSONL beside this file (`points.jsonl`, `greedy.jsonl`,
`qos-*.jsonl`, per-replica logs under `fleet-cap*-p*/` — second sweep overwrote the
per-replica log dirs for p1/p2 names; the points/qos JSONLs are append-only and hold all
four passes). Ran 17:29–17:42Z and 18:04–18:17Z, box otherwise idle for the fleet GPUs
5-7 (`gpu-state-pre-fleet.txt`; the second sweep overlapped the ppn soak on GPU 0).
Thermal regime: steady, mid-window.

## Throughput (agg tok/s, c=96 primary cell; zero 5xx anywhere in 4 passes)

Four passes total (two interleaved sweeps of pass 1/2 each, caps rotated 8→12→16 within
every pass — 12 fleet bring-ups). Medians of N=4:

| cap | c=96 median (N=4) | c=96 per-pass | c=48 median | qosbulk median |
|---|---|---|---|---|
| 8 (incumbent) | 1563.7 | 1423.1 / 1701.8 / 1588.5 / 1539.0 | 1568.2 | 1547.2 |
| 12 | 1697.8 | 1696.6 / 1638.3 / 1966.6 / 1699.1 | 1502.7 | 1571.2 |
| 16 | **1834.8** | 1781.2 / 1760.4 / 1888.5 / **2097.6** | 1541.8 | **1823.0** |

## Multi-tenant QoS probe (c=4 latency tenant under a concurrent c=96 bulk tenant, N=4)

| cap | tenant tok/s median | tenant p95 median | p95 per-pass |
|---|---|---|---|
| 8 | **227.3** | **4.79s** | 5.37 / 4.48 / 3.71 / 5.10 |
| 12 | 170.2 | 8.86s | 8.87 / 9.12 / 3.06 / 8.84 |
| 16 | 203.9 | 6.17s | 6.37 / 6.57 / 5.96 / 5.94 |

## Verdict (mission shape: beat same-window cap-8 at c=96, zero 5xx, both passes)

- **cap 16 wins the bulk-throughput cell decisively at N=4**: c=96 median 1834.8 vs
  cap-8's 1563.7 (**+17.3%**), ahead of cap 8 in every one of the four passes
  (min cap-16 pass 1760.4 > max cap-8 pass 1701.8), zero errors, p95 stable ~7s.
- **cap 12 beats cap 8 on median (+8.6%) but carries the worst tail**: c=96 p95 >10s in
  the first sweep and the QoS tenant's p95 ~8.9s median — dominated by cap 16 on both
  axes. NEGATIVE row, recorded.
- **QoS trade is real and priced**: cap 8→16 costs the latency tenant ~29% p95
  (4.79s→6.17s median) and ~10% tenant tok/s. Latency-class fleets keep cap 8;
  bulk-throughput fleets take cap 16.
- **Greedy anchor limitation**: all twelve greedy arms returned ok=6 err=0 tok=768, but
  `load-serve.py` records no output hash, so the mission's "greedy hash unchanged
  across caps" check is NOT computable from these receipts — token-count identity is
  the weaker anchor actually captured. Add a hash field to load-serve.py before the
  next cap sweep.

N=4 medians, one box-session, same binary, caps interleaved within pass; thermal regime
steady mid-window. This is the in-window re-sweep receipt; a board move still needs the
x5-interleaved board protocol.
