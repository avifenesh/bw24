# Fleet capacity re-sweep — replica scaling on the 8xH100 box (2026-08-02)

Phase-3 paid-window deliverable, lane/fleet-resweep. The darklanes product question:
how many concurrent serving replicas per box, what per-replica throughput at each
fleet size, and where per-GPU replica stacking stops paying on H100 80GB.

Setup: devices **4-7** (ran in parallel with the M2 lane on 0-3; GPU claims in
`~/receipts/gpu-assignment.txt`, released 20:38Z). Binary: the box-prebuilt lane/m2-pp8
`~/memra/target/release/memra-server` (sha256 `ad8604b3…`, `binary-sha256.txt`) — the
exact-16 decode-chunk-tier + fa-deep core, NOT the chunk-8-era v0.60 binary the prior
fleet receipts used. Model: Qwen3.5-9B-Q8_0 (NVMe copy, 16,659 MiB resident/replica).
Fleet supervisor `tools/serve-fleet.sh` (cap 8 proxy on :9080, replicas :9085+),
harness `tools/load-serve.py` (~200-tok prompt, max_tokens 128, temp 0.7 + per-request
seeds — the darklane-serving/dl-metering protocol). Scaling cells are **direct**
per-replica harnesses at matched per-replica concurrency (the R3.3
matched-saturation protocol, `research/darklane-serving-20260801/`); QoS cells go
through the admission proxy. **N=3 passes, fleet sizes interleaved pass-wise**
(1→4→8→12 within every pass, full teardown/bring-up per cell — 12 fleet bring-ups).
One thermal window, 20:21–20:38Z, steady mid-window (peak 61 C / 478 W, GPU 4;
others ≤58 C — `vram-1hz.csv`). Zero request errors in the entire campaign
(8,592 measured requests). Driver: `run.sh`; timeline: `driver.log`.

## 1. Single-replica baseline (device 4, N=3)

| c | agg tok/s median (per-pass) | p50 | p95 | p99 |
|---|---|---|---|---|
| 8 | **657.9** (654.1 / 657.9 / 662.0) | 1.554s | 1.662s | 1.675s |
| 16 | **801.3** (800.6 / 804.0 / 801.3) | 2.548s | 2.655s | 2.658s |
| 32 | **802.6** (797.1 / 802.6 / 815.1) | 5.105s | 5.214s | 5.218s |

The exact-16 decode chunk tier is live on this binary and moves the H100
single-replica saturation point: **c=8 → c=16, ~658 → ~801 tok/s (+21.8%)**, flat
c=16 → c=32 (802.6, pure queueing past 16). The SERVING.md H100 numbers
(654/657/659, chunk-8 era) are superseded on-box; the pending "chunk-16 fleet
effect" re-validation flagged there is answered below.

## 2. Replica scaling (per-replica c=8 and c=16, direct, N=3)

Aggregate = sum of per-replica direct harnesses within a pass; median of 3 passes.

| fleet | layout | sessions/GPU | agg tok/s med (per-pass) | per-replica | **per-GPU** | p50 | p99 |
|---|---|---|---|---|---|---|---|
| f1 c16 | 1 on dev4 | 16 | 801.3 | 801.3 | **801.3** | 2.548s | 2.658s |
| f4 c8/r | 1/GPU | 8 | 2632.4 (2615 / 2633 / 2632) | 658.1 | 658.1 | 1.558s | 1.677s |
| **f4 c16/r** | 1/GPU | 16 | **3218.6** (3211 / 3219 / 3223) | **804.7** | **804.7** | 2.544s | 2.696s |
| f8 c8/r | 2/GPU | 16 | 2091.9 (2054 / 2092 / 2193) | 261.5 | 522.9 | 3.879s | 4.489s |
| f8 c16/r | 2/GPU | 32 | 2549.6 (2501 / 2550 / 2680) | 318.7 | 637.4 | 6.329s | 7.564s |
| f12 c8/r | 3/GPU | 24 | 2110.9 (2015 / 2231 / 2111) | 175.9 | 527.7 | 5.821s | 6.393s |

Replica-balance spread (p2, per-replica agg): f4 654–662 (tight), f8 c8/r 237–288,
f8 c16/r 283–357, f12 175–194. Greedy probe hash `56b8502cfb8de57a` identical on
**75/75** probes — every replica, every fleet size, every pass, including 3-per-GPU
co-residency (`greedy-hashes.txt`; hash differs from the v0.60-era `dbd1c98f9fed4efe`
because the binary moved — allowed — but is internally consistent everywhere).

### Where stacking stops paying: at ONE replica per GPU, on this core

At every matched sessions-per-GPU count, one replica beats stacked replicas:

- 16 sessions/GPU: 1×c16 = **801–805** vs 2×c8 = 522.9 (**-35%** for stacking)
- 32 sessions/GPU: 1×c32 = **802.6** vs 2×c16 = 637.4 (**-21%**)
- 24 sessions/GPU: 3×c8 = 527.7 — flat vs 2×c8 (522.9); the third replica buys nothing.

**The R3 pair-packing verdict (2026-08-01: pairs +62%, 490/GPU vs 305) is STALE on
this binary** — another instance of the stale-verdict law. Pair-packing paid when a
replica saturated at c=8/~305 with a serial-latency-bound tick; the exact-16 tier
raised single-replica saturation to c=16/~801, and co-residency contention
(two processes' kernels inflating each other's latency) now costs more than the
second replica adds. Latency agrees: 2×c8 serves the same 16 sessions/GPU at p50
3.88s vs 2.55s for 1×c16 — worse throughput AND worse latency.

**Deployment answer:** on this core the H100 box config is **1 replica/GPU at
admission cap 16**. Measured: **3,218.6 tok/s on 4 GPUs (804.7/GPU)**; a 7-GPU
serving fleet (one GPU reserved) projects ~5,633 tok/s, full box ~6,437 — vs the
v0.60 receipt's 1,477 managed on 3 GPUs (492/GPU): **+63% per GPU**, from the
exact-16 tier plus un-stacking. Proxy CAP should move 8 → 16 for 1/GPU fleets
(consistent with the m2-pp8 in-window cap-16 finding, +17.3% at c=96 on 6 stacked
replicas — this sweep shows the bigger step is de-stacking, 611.6/GPU there vs
804.7/GPU here).

VRAM headroom (`vram-1hz.csv`, `vram-f*-up.txt`): 16,659 MiB/replica at bring-up;
under load 20.3 GiB/replica; f12 peak 61,017 MiB/GPU of 81,559 — a 4th replica/GPU
fits but there is no throughput case for stacking at all.

## 3. Multi-tenant QoS at 8 replicas (proxy :9080, cap 8, N=3)

Interactive tier = c=4, 24 req; batch tier = c=96, 288 req, started 5s earlier
(the fleet-cap-resweep probe shape; per-request tails in `perreq/f8-qos*.jsonl`).

| tenant / condition | agg tok/s med (per-pass) | p50 med (per-pass) | p95 med (per-pass) |
|---|---|---|---|
| interactive alone | 295.0 (280.2 / 295.0 / 296.3) | 1.733s (1.770 / 1.733 / 1.727) | 1.741s |
| interactive + bulk | 187.9 (181.1 / 187.9 / 199.5) | **1.778s** (1.846 / 1.737 / 1.778) | **7.33s** (7.33 / 7.60 / 6.43) |
| bulk (concurrent) | 1896.2 (1866.9 / 1896.2 / 2035.0) | 5.352s | 7.4–7.9s |

Proxy metrics (all 3 passes): **zero 429s, zero 5xx**, queue peak depth 36 of 256,
wait p95 3.57–3.94s ≈ one service generation (`metrics-f8-qos-p*.json`).

**Verdict: PARTIAL — p50 holds, the tail does not.** Interactive p50 held within
2.6% of alone (1.733 → 1.778s) in all three passes, matching the dl-metering
2026-08-02 receipt's headline (p50 held exactly there). But interactive p95
inflated 4.2x (1.74 → 7.33s) vs ~2x in the dl-metering battery — because the
mechanisms differ: dl-metering's SLO gate is **engine-side** (x-lane admission,
harvest shed at 429 when interactive step p99 nears 50ms; `lane/dl-metering`,
commit 73d3adf4, **not merged into this binary** — verified absent from the box
tree). Here the only QoS mechanism is the proxy's per-backend cap + one shared
FIFO: interactive requests queue behind bulk waves with no priority, so roughly
the slowest fifth of interactive requests eat a full bulk service generation
(the per-request tails are bimodal: ~1.7s uncontended, ~6.4–7.6s behind a wave).
Scaling 2-replica-era intuition to 8 replicas did not break p50 — the cap kept
replicas inside the exactness tier and sheds nothing — but tail-class QoS at
fleet scale needs either lane-priority in the proxy queue or the dl-metering
engine gate merged. Negative result recorded as such.

## 4. Negatives and caveats

- **f8/f12 stacking rows are negative results** and are the point: they price
  stacking out on this core.
- The interactive-alone baseline itself is proxy-routed across 8 replicas
  (295 tok/s at c=4; requests fan out, no batching benefit — expected).
- Single box, one model class (dense 9B Q8_0 — the fleet model). MoE/larger
  models change the VRAM story (16.7 GiB → single-replica-only regardless).
- Cross-day comparisons (v0.60 1477, R3 1480) carry the clock-drift caveat;
  the in-window cells here are same-session, same-binary, interleaved.
- `load-serve.py` still records no output hash (gap flagged in the m2-pp8
  sweep); the greedy anchor here is the separate 75-probe curl battery, which
  does hash content.
- Proxy c=96 through 8 stacked replicas (qosbulk ~1896) is not directly
  comparable to the m2-pp8 cap sweep (6 replicas, GPUs 5-7, different denominator).

## 5. Receipts

All raw on-box at `~/receipts/fleet-resweep/` (rsynced here byte-identical):
`points.jsonl` (126 load points), `perreq/` (8,592 per-request rows),
`greedy-hashes.txt` (75 probes), `driver.log` + `driver-outer.log` +
`logs/` (per-cell load logs), `fleet-f*-p*/logs/` (per-replica + proxy +
supervisor logs, all 12 bring-ups), `metrics-f8-qos-p*.json` (proxy metrics
after each QoS cell), `vram-1hz.csv` (1 Hz VRAM/temp/power, devices 4-7,
whole campaign), `vram-f*-p*-up.txt` (resident VRAM at each bring-up),
`gpu-state-{pre,post}.txt`, `binary-sha256.txt`, `run.sh` (exact driver,
params baked as literals).
