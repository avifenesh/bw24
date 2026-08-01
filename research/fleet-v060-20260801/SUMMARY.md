# Fleet v0.60.0 validation — darklanes-bench, 2026-08-01 (~10:00Z)

Deployed the merged v0.60.0 tree (commit `36bc1d9f`, tag `v0.60.0` — batched-tick +
device sampling + lean logits) onto the real 6-replica fleet (pairs on GPUs 5/6/7,
cap 8, proxy :8080), replacing the v0.59-era binary (on-box `~/memra`, built Aug 1
00:56). Build on-box: 3m58s (`logs/build-tail.log`). Cutover was clean: old fleet
stopped via its own supervisor script (`~/darklane-serving-20260801/serve-fleet.sh
stop`), all pidfile pids dead, ports free, GPUs 5/6/7 drained to 0 MiB
(`cutover.log`), new fleet up healthy in ~30s.

## Greedy determinism (the cross-replica contract)

Hash = sha256(content)[:16] of the R3/R4 greedy probe (primes prompt, max_tokens 64,
temperature 0, seed 0).

- Old fleet, pre-teardown, all 6 replicas: `dbd1c98f9fed4efe` (`old-fleet-final-state.txt`)
- **v0.60.0, all 6 replicas, sequential AND all-6 concurrent: `dbd1c98f9fed4efe`**
  (`greedy-hash-v060.log`)
- Post-chaos-restart, all 6: `dbd1c98f9fed4efe` (`chaos-v060.log`)

The hash was allowed to change with the new binary; it did not — greedy output is
byte-identical across the upgrade, and all 6 replicas agree in every condition.
18/18 hash checks match.

## Load points through the proxy (:8080)

Two interleaved passes per point (N=2, warm box, ~5 min window; chaos c48 is a third
c48 sample at 1487). Prior = R4 receipts (v0.59-era binary, same harness, same
request counts). Raw: `v060-points.jsonl`, `v060-per-request.jsonl`,
`metrics-v060.log` (1 Hz /metrics).

| point | v0.59-era (R4) | v0.60.0 p1 | v0.60.0 p2 | delta | p50 / p95 | err |
|---|---:|---:|---:|---:|---|---:|
| c=24 (96 req) | — (no R4 managed point) | 1167.8 | 1162.5 | — | 2.60s / 2.75s | 0 |
| c=48 (192 req) | 1367.9 | 1471.8 | 1463.0 | **+7.0-7.6%** | 4.10s / 4.24-4.35s | 0 |
| c=96 (288 req) | 1378.6 | 1477.0 | 1473.1 | **+6.9-7.1%** | 8.12s / 8.39-8.49s | 0 |

- Managed v0.60.0 (~1470) now matches the v0.59-era **direct** number (1480): the
  ~7% admission-overhead gap closed at the fleet level.
- Queue behavior at c=96 reproduces R4 exactly: peak depth 49, enqueued 480 (2
  passes), wait p95 4.22s (= one service generation), **zero 429s, zero 5xx** —
  cap sheds surplus to the queue, throughput flat c48->c96.
- The isolated single-replica tick gains (305->655) do NOT translate 1:1 to fleet
  aggregate at cap 8: fleet throughput is admission/pipeline-bound at this cap, not
  tick-bound. The fleet-level lever for the batched tick is a cap re-sweep
  (cap 8 was calibrated on the v0.59 core — stale-verdict risk); left for the
  next box window, not measurable in this one.

## Chaos (round-4 machinery on the new binary)

SIGKILL replica :8090 (pidfile pid 419448) at 10:04:19.749, mid c=48 x 768 req
(`chaos-v060.log`, `chaos-metrics-v060.log`):

- proxy passive breaker: DOWN same second (10:04:19, RemoteDisconnected)
- supervisor restart: +2s (10:04:21, new pid 424916)
- proxy backend UP: +9s (10:04:28; page-cache-warm reload)
- errors: **8/768** = exactly the in-flight cap on the victim, all quoted 502s
  naming :8090
- aggregate across the kill: **1487.3 tok/s** — recovery invisible in throughput
  (vs 1387 in the R4 chaos run: +7.2%)
- post-restart greedy hash all 6: match. **Chaos verdict: PASS.**

## End state

6 replicas + proxy + supervisor healthy on the v0.60.0 binary
(`final-fleet-status.txt`), FLEET_RUN=~/darklane-fleet, SERVER_BIN now
`~/fleet-v060/target/release/memra-server` (recorded with sha256 of old + new
binaries in `build-info.txt`). Old fleet logs preserved on-box at
`~/darklane-fleet/logs-v059-final/`. GPUs 0-4 untouched; GPU 2 never touched.

Thermal regime: steady-state box, all points within a 5-minute warm window,
single day, same harness — cross-day comparisons vs R4 carry the usual
clock-drift caveat; the like-for-like R4 deltas above are same-box, same-harness,
same request counts.
