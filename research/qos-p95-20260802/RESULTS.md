# QoS p95 fix-and-verify — the engine-side x-lane SLO gate at fleet scale (2026-08-02/03)

lane/qos-p95, off restructure/public-split (dad7f178). The fleet-resweep finding
(`research/fleet-resweep-20260802/RESULTS.md` §3): at 8 replicas under a c=96 bulk
tenant, interactive p50 held (+2.6%) but p95 inflated **4.2x** (1.74 → 7.33s) — the
fleet binary's proxy is a lane-blind FIFO, and the engine-side x-lane SLO gate lived
only on unmerged `lane/dl-metering`. This lane extracts the QoS half of that gate
(commit 73d3adf4's base — the metering half stays darklane-side), ports it onto the
v0.67 serve surface, and re-runs the exact fleet-resweep QoS@8 scenario A/B.

Box: the 8xH100 (13.58.112.171), devices **4-7** (moe-speq held 0-3;
`~/receipts/gpu-assignment.txt`). Model: Qwen3.5-9B-Q8_0 (NVMe). Gate binary
`5b4bf744…`, pristine-parent binary `d9460808…` (built from dad7f178 exactly;
`binaries-sha256.txt`). Harness: `tools/load-serve.py` (+ dl-metering's
`--lane/--tenant/--retry-shed` flags, ported), fleet `tools/serve-fleet.sh`
(cap N proxy :9080, replicas 9085-9092), drivers `run.sh` / `run-slo.sh`
(params baked as literals). One thermal window 23:20–00:35Z, peak 56 C / 399 W
(`vram-1hz.csv`). Zero request errors in every cell of the campaign.

## 1. What the gate is (mechanism)

`lane/dl-metering`'s QoS machinery is **admission control + priority scheduling +
budgeted dark prefill**, engine-side — not queue reordering:

- Three classes via the `x-lane` header (absent = interactive; naked traffic takes the
  identical code path): `interactive` (protected), `judge` (prefill-shaped),
  `harvest` (decode-shaped bulk).
- **Admission gate**: interactive always admits (FIFO-waits past its cap, never shed);
  judge/harvest are admitted only while measured interactive decode-step p99 <
  `MEMRA_SHED_JUDGE`(1.00) / `MEMRA_SHED_HARVEST`(0.90) x `MEMRA_SLO_P99_MS`(50ms).
  Over budget = immediate shed (HTTP 429 + Retry-After) — dark work is never queued
  inside the engine ("the engine's own queue is where the tail dies"). A starvation
  sentinel sheds dark admits when interactive work exists but hasn't decoded within
  the SLO age (the estimator's blind spot).
- **The SLO sensor is engine truth**: the wall time of each batched decode tick that
  carried >= 1 interactive session, windowed p99 (30s).
- **Priority inside the tick**: interactive decode rows sort first into the batched
  chunks; interactive prefill runs at the full tick chunk before decode; dark-lane
  prefill runs AFTER decode, one chunk/tick, capped by both its lane budget (256 tok)
  and the tick's measured SLO headroom (the adaptive cap — the 282ms-p99 lesson).
- Extraction was clean: the QoS gate is 73d3adf4's parent's lane machinery
  (memra-lanes crate + worker/handler wiring); meter.rs, /usage, tenant identity and
  rid plumbing stayed behind. `/yield/metrics` (per-lane counters + step p50/p99)
  came along as the receipts endpoint.

Port surface: `crates/memra-lanes/` (new), `crates/memra-server/src/{main,worker}.rs`,
`tools/load-serve.py`. The prime-batch concat, prefix cache, VRAM-aware admission,
GS/spec paths of v0.67 all now carry lane predicates (interactive-only where the
dl-metering original gated them).

## 2. The A/B: fleet QoS@8, four conditions, N=3 interleaved

Exact fleet-resweep probe shape: f8 (2 replicas/GPU on 4-7), proxy :9080; interactive
c=4/24req alone, then bulk c=96/288req with interactive c=4/24req starting 5s in.
Conditions interleaved within each pass, full teardown/bring-up per cell (12
bring-ups): **off8** = no lane headers, CAP=8 (the fleet-resweep mechanism, the
before); **on8** = lanes on, CAP=8; **off16** = no lanes, CAP=16 (cap attribution
control); **on16** = lanes on, CAP=16. Bulk in "on" cells sends `x-lane: harvest
--retry-shed` (the real harvest-client shape); interactive sends
`x-lane: interactive`.

| cond | int alone p50/p95 | int contended p50 | int contended **p95** | bulk agg tok/s | combined agg |
|---|---|---|---|---|---|
| off8 (= fleet-resweep) | 1.842 / 1.957s | 1.846s | **7.150s** (6.37/7.21/7.15) | 2010.5 | 2194.8 |
| on8 | 1.757 / 1.803s | 1.756s | 6.463s (6.44/7.13/6.46) | 2174.0 | 2361.2 |
| off16 | 1.681 / 1.805s | 1.686s | 4.335s (4.34/4.19/5.07) | 2490.2 | 2709.7 |
| **on16** | 1.675 / 1.832s | 2.387s | **3.690s** (3.59/3.69/4.11) | 2213.8 | 2398.0 |

(t medians of 3 passes, per-pass values in parens; `analyze.py box/points.jsonl`.)

- **off8 reproduces the fleet-resweep failure**: p95 7.15s ≈ the original 7.33s,
  bimodal tails (~1.7s uncontended / 6.4-7.2s behind a bulk wave), p50 holds.
- **on8: the gate is starved, not broken.** With the proxy still capping at 8/replica,
  engine step p99 sits ~19-25ms — far under the 45ms shed line — so almost nothing
  sheds (4 total) and the queueing stays in the lane-blind proxy FIFO where the gate
  cannot see it. p95 6.46s: marginal. **The engine gate cannot fix a queue it never
  sees — the proxy cap must open wide enough to move contention into the engine.**
- **on16 is the fix shape**: proxy queue empties (peak depth 0, zero proxy 429s), the
  engine's harvest lane cap (8/replica) + SLO gate take over admission
  (engine-side sheds 804-835/pass, all retried and completed: 288/288 bulk OK every
  pass). Interactive contended p95 **7.15 → 3.69s (1.9x better)**, now 1.8-2.1x of
  alone — **the dl-metering-class ~2x tail restored** at 8 replicas.
- **Attribution split**: cap16 alone (off16) gets p95 to 4.34s (shorter proxy FIFO
  wait ≈ one service generation at depth ≤16); the lane gate takes it from 4.34 →
  3.69s and cuts the p99-class outliers (off16 max 5.07 vs on16 4.11 across passes).

## 3. The priced trade + the SLO dial

The gate's cost axes, measured:

- **Bulk throughput**: on16 2213.8 vs off16 2490.2 tok/s (-11.1%) — the shed-retry
  cycles + interactive-first scheduling are the QoS price. vs the off8 before
  (2010.5) it is still +10%. Combined aggregate on16 2398 vs off16 2710 (-11.5%),
  vs off8 2195 (+9.2%). **No regression vs the before; -11% vs the cap-16 ceiling.**
- **Interactive contended p50 rides up under the 50ms SLO**: 1.69 → 2.39s. Mechanism:
  the gate defends *step* p99 (50ms), so admitted-load equilibrium settles where
  interactive TPOT ≈ SLO x shed fraction — request latency ≈ 128 tok x ~18ms ≈ 2.3s.
  **`MEMRA_SLO_P99_MS` is the dial** (confirm sweep, N=3 each, `points-slo*.jsonl`):

| SLO (cap16, lanes on) | int cont p50 | int cont p95 | bulk agg | engine sheds/pass |
|---|---|---|---|---|
| 50ms (default) | 2.39-2.81s | 3.69-4.15s | ~2200 | ~810 |
| 35ms | 2.06s | 3.56s | 1592 (noisy: 797-2150) | ~2800 |
| 25ms | **1.637s** | **2.158s** | 818.1 | ~6200 |

  At 25ms the interactive tenant is fully protected — contended p50/p95
  (1.637/2.158s) statistically equal to alone (1.635/2.065s), i.e. the fleet-resweep
  p95 inflation goes 4.2x → **1.24x of alone** — with bulk paying 2490 → 818 (-67%).
  The knob spans "bulk-first" to "interactive-first"; 50ms default keeps ~89% of
  ceiling throughput with the ~2x tail.

## 4. Naked-path exactness + throughput (the isolation contract)

- **Greedy identity**: 101/102 fleet-campaign probes = `56b8502cfb8de57a` (the
  fleet-resweep anchor) across all 12 bring-ups, every replica, lanes on and off,
  interactive and harvest classes. Parent-vs-gate probes 6/6 identical. The single
  divergent probe (p2-off8 :9085, `dbd1c98f9fed4efe`) is **exactly the v0.60-era
  content hash** (`research/darklane-serving-20260801`), not noise: a known
  alternate greedy continuation class for this prompt. Not reproduced in x24
  same-binary probes, x12 co-resident-load probes, or any of the other 101 —
  logged as a 1/102 cold-bring-up flake with both known-good contents, gate-neutral
  (it appeared in an OFF cell).
- **Naked throughput, pristine parent (dad7f178) vs gate binary, interleaved x3,
  single replica c=16/64req, no lane headers**: parent 950.9/950.9/950.5 vs gate
  944.4/949.3/946.7 tok/s — **-0.3% median, inside the window**; p50/p95 equal
  (2.15/2.17s both). The earlier ad8604b3-vs-gate delta (-15%) was the box binary
  being the *older m2-pp8 lane* build, not the gate — the pristine-parent rebuild
  closes it. (`points-parent-ab.jsonl`, `greedy-parent-ab.txt`.)

## 5. Verdict

**MERGE-READY on the box evidence** — with the deployment note that the gate only
bites when engine admission (not the proxy) owns the queue: ship it with proxy
CAP=16 for 1-2 replica/GPU fleets (consistent with the fleet-resweep cap answer).
Naked path: byte-identical greedy, flat throughput, zero behavior change without
`x-lane` headers (flags doctrine: winners default, the gate is header-opt-in per
request, `MEMRA_SLO_P99_MS` tunes it). Remaining before a runtime-default claim per
CLAUDE.md: the 5090-rig battery (kernel-check / run-gen / run-spec) on merge day —
this lane's H100 receipts are serve-layer only and the diff touches no kernels.

## 6. Receipts

All raw on-box at `~/receipts/qos-p95/` (rsynced here byte-identical under `box/`):
`points.jsonl` (36 load points, 12 cells), `points-slo.jsonl` + `points-slo35.jsonl`
(SLO dial, 27 points), `points-parent-ab.jsonl` + `points-naked*.jsonl` (naked A/B),
`perreq/` (per-request tails, every cell), `greedy-hashes.txt` (102 probes) +
`greedy-repro-x24.txt` + `greedy-corepro-x12.txt` + `greedy-parent-ab.txt`,
`yield-*.json` (per-replica lane counters incl. engine shed counts),
`metrics-*.json` (proxy queue state per cell), `fleet-*/logs/` (replica + proxy +
supervisor logs, all 18 bring-ups), `vram-1hz.csv`, `binary-sha256.txt` +
`binaries-sha256.txt`, `driver*.log`, `run.sh` / `run-slo.sh` (exact drivers),
`analyze.py`.
