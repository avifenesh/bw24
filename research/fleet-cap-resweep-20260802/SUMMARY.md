# Fleet cap re-sweep 8/12/16 + multi-tenant QoS probe — 8xH100 box, 2026-08-02

Phase-3 stale-verdict lever from `tools/box-aug2-mission.md` §2c: cap 8/replica was
calibrated on the v0.59 core; this binary (lane/m2-pp8 build, 2026-08-02) moved the tick.
Per the clock-drift law the **in-window cap-8 arm is the denominator** — the v0.60
reference row (1477.0 managed) is context only.

Setup: GPUs 5-7, 2 replicas/GPU (6 replicas), Qwen3.5-9B-Q8_0 (NVMe copy), proxy :8080.
Two passes, caps interleaved pass-wise (8→12→16 per pass). Raw JSONL beside this file
(`points.jsonl`, `greedy.jsonl`, `qos-*.jsonl`, per-replica logs under `fleet-cap*-p*/`).
Ran 17:29–17:42Z, box otherwise idle (`gpu-state-pre-fleet.txt`). One 13-minute sweep;
thermal regime: steady, mid-window.

## Throughput (agg tok/s, c=96 primary cell; zero 5xx everywhere)

| cap | c=96 p1 | c=96 p2 | c=48 p1 | c=48 p2 | c=96 p95 lat p1/p2 |
|---|---|---|---|---|---|
| 8 (incumbent) | 1423.1 | 1701.8 | 1428.3 | 1712.9 | 8.73s / 7.30s |
| 12 | 1696.6 | 1638.3 | 1506.6 | 1486.0 | **10.05s / 10.61s** |
| 16 | **1781.2** | **1760.4** | 1530.1 | 1521.3 | 7.04s / 7.10s |

## Multi-tenant QoS probe (c=4 latency tenant under a concurrent c=96 bulk tenant)

| cap | lat-tenant p50 p1/p2 | lat-tenant p95 p1/p2 | tenant tok/s p1/p2 |
|---|---|---|---|
| 8 | 1.95s / 1.72s | **5.37s / 4.48s** | 219.0 / 234.6 |
| 12 | 1.92s / 1.86s | 8.87s / 9.12s | 169.0 / 163.6 |
| 16 | 1.91s / 1.86s | 6.37s / 6.57s | 202.5 / 193.4 |

## Verdict (per the mission's shape: beat same-window cap-8 in BOTH passes at c=96, zero 5xx)

- **cap 16 formally wins the throughput cell**: beats cap 8 at c=96 in both passes
  (+25.2% p1, +3.4% p2), zero errors, and its p95 stays in the 7s band.
- **BUT the pass spread is a confound**: cap8-p1 (1423) was the sweep's first loaded arm
  and is 16% below its own p2 (1702) — a cold-start/settle effect the 45s health wait did
  not fully absorb. Against the *warmed* cap-8 denominator (1702), cap 16 is +3.4% only.
  N=2 passes; publish as "cap 16 ≥ cap 8, +3.4% on the warmed pass" — not the p1 +25%.
- **cap 12 is dominated**: middling throughput and the worst tail everywhere
  (c=96 p95 >10s both passes; QoS-tenant p95 ~9s). NEGATIVE row, recorded.
- **QoS trade**: raising cap 8→16 costs the latency tenant ~40% p95 (4.5-5.4s → 6.4-6.6s)
  at roughly equal p50, and ~14% tenant throughput. If the fleet serves a
  latency-sensitive class, cap 8 remains the pick; cap 16 is the bulk-throughput pick.
- **Greedy anchor limitation**: all six greedy arms returned 6/6 ok, 768 tokens, but
  `load-serve.py`'s summary JSONL records no output hash, so the mission's
  "greedy hash unchanged across caps" check is NOT computable from these receipts —
  token-count identity is the weaker anchor actually captured. A hash field in
  load-serve.py is the follow-up before the next cap sweep.

Medians: single 13-min sweep, N=2 passes/cap; treat as the in-window re-sweep receipt,
not a board cell (board rule: interleaved ×5 with same-session denominators).
