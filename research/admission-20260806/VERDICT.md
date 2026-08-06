# Admission-latency fix — VERDICT (2026-08-06, local 5090, lane/admission-latency)

## Mechanism (confirmed in code, then by receipt)

A request that arrived while a spec burst was in flight waited in the worker's mpsc
channel until the burst returned — the tick-top admission phase (worker.rs step 2) only
runs between ticks, and phase (a) gives each spec session a full `MEMRA_SPEC_BURST`
burst. Contended first-text therefore scaled with burst size (sse-cadence VERDICT:
0.57s at B32, 1.67s at B128). Two independent waits, fixed separately:

1. **Pre-admission wait** (the burst in flight when the request arrives):
   `PENDING_ADMITS` atomic gauge — the HTTP handler increments before sending
   `Cmd::Generate` (both send sites; decrement on send-failure), the worker decrements
   at pop (`handle_cmd`, saturating). The sse-cadence `on_commit` round-boundary hook
   now returns a continue-verdict; the worker's flush closure polls the gauge at every
   round boundary (including empty poll-only flushes — round-stream drains, zero-round
   folds) and `false` ends the burst exactly as if burst-count had been reached (same
   session-tail path; burst size is content-neutral per the spec-levers battery).
2. **Post-admission wait** (measured after fix 1 alone: 1.61s -> only 1.30s — the
   background session, at a lower index, ran its whole NEXT burst before the newcomer's
   prime ever flushed): COLD-FIRST spec ordering — phase (a) steps sessions that have
   emitted nothing this request (`generated.is_empty()`; pool resumes count as cold)
   before mid-generation peers. Stable sort = FIFO within cold/warm classes.

Rollback seam: `MEMRA_ADMIT_YIELD=0` disables BOTH pieces (full-burst holds +
index-order phase (a)) — the complete pre-lane behavior in one flag.

## Contended first-text (q27 nv+draft, K=3, one 512-tok bg request streaming, N=5, one hold, 54-56C)

| arm            | contended first text | runs (s)                        |
|----------------|----------------------|---------------------------------|
| B32  fix-on    | 0.123s               | 0.132 0.122 0.123 0.123 0.124   |
| B32  fix-off   | 0.541s               | 0.832 0.454 0.624 0.486 0.541   |
| B128 fix-on    | 0.152s               | 0.121 0.153 0.152 0.152 0.151   |
| B128 fix-off   | 1.601s               | 2.353 1.561 1.671 1.537 1.601   |

fix-off reproduces the sse-cadence classes (0.54s / 1.60s). fix-on: contended first-text
no longer scales with burst size — 0.12-0.15s at ANY burst, i.e. the SOLO first-text
class (0.124s). B128 contended = 10.5x better than fix-off.

## Solo TTFT (N=5 — no admit waiting => no behavior change)

| arm          | first text | chunks | gap p50 |
|--------------|-----------|--------|---------|
| B32  fix-on  | 0.124s    | 89     | 27.5ms  |
| B128 fix-on  | 0.124s    | 89     | 27.6ms  |
| B128 fix-off | 0.124s    | 89     | 27.6ms  |

Identical — the gauge is 0 for a solo stream; no round ever ends early.

## Throughput (load-serve 128tok temp0.7, alternating boots x2/arm, 2 passes/boot, err=0, 76-85C interleaved)

c=1 (solo): fix makes NO difference — gauge stays 0 mid-request.
- B128: on 100.7-101.4 vs off 100.8-101.4 agg tok/s — parity.
- B32: on 93.2-93.5 vs off 93.0-93.5 — parity.

c=8 (saturated, THE risk cell): the yield is not free here — it changes scheduling.
- B128: on 94.8-96.5 vs off 98.5-99.4 agg tok/s — **-3.4% agg**, interleaved and
  consistent across both rounds (not thermal drift; alternation clean).
- The same cell's latency shape: p50 **2.72-2.78s on vs 10.32-10.42s off (3.8x
  better)**; p95/max grow (20.4-21.8s vs 10.5s). Mechanism: fix-off runs fair
  lockstep — all 8 finish together at ~10.4s; fix-on serves newcomers first
  (closed-loop load = constant arrivals = constant round-boundary yields + cold-first
  priority), so median requests finish 3.8x sooner and the unlucky tail pays.
- B32 c=8 same shape, smaller: on 87.7-88.6 vs off 90.8-91.6 (-3.4%), p50 10.9 vs 11.4.

The -3.4% is the receipt-carried cost of newcomer-first scheduling under full
saturation. For the interactive tier (the pill: 1-2 concurrent) the contended cell IS
the felt path and the cost does not exist (c=1 parity). Pure-batch tiers that want
lockstep fairness back: `MEMRA_ADMIT_YIELD=0`.

## The +7% burst lever survives the fix (fix-on, this window)

- c=1: B128 101.1 vs B32 93.3 = **+8.4%**
- c=8: B128 95.7 vs B32 88.2 = **+8.5%**

## Exactness / gates (fix-on binary)

- Greedy streamed content byte-identity (522B, concat of every SSE delta):
  solo B128 on==off, solo B32 on==off, B32==B128, **contended B128 on==off**,
  solo==contended, and the contended BACKGROUND response on==off (the session whose
  bursts get chopped + reordered) — all cmp-clean.
- run-spec K=1..8 nv+draft BURST=128: rc=0, 9 PASS.
- serve-smoke: rc=0, 0 failed.
- decode-batch-gate 9B NVFP4: config B=8 ALL GREEN; strict B=4 equalized ALL GREEN.
- cargo test -p memra-server: 75 passed, 0 failed.

## Burst-default table (the flip decision)

| criterion            | B32 fix-on | B128 fix-on | B128 >= B32? |
|----------------------|-----------|-------------|--------------|
| solo TTFT            | 0.124s    | 0.124s      | TIE          |
| contended first-text | 0.123s    | 0.152s      | -29ms        |
| throughput c=1       | 93.3      | 101.1       | +8.4%        |
| throughput c=8       | 88.2      | 95.7        | +8.5%        |

HOLD, per the flip criterion (B128-with-fix >= B32 on solo TTFT, contended TTFT, AND
throughput): contended first-text misses — 0.152s vs 0.123s, 29ms worse at B128. The
miss is exactly one spec-round gap (the 27ms steady-state cadence): the newcomer now
waits at most one round boundary, and a B128 round costs the same as a B32 round, so
this is the yield-poll quantum, not burst scaling. Everything else favors B128
(+8.4-8.5% both regimes, solo TTFT tied), and the old flip-blocker (0.57s->1.67s
admission scaling) is GONE — flipping is now a 29ms-for-8.5% owner call instead of a
latency cliff. Receipt carries both numbers; default stays 32 until the owner takes it.

## Files

- Fix: crates/memra-server/src/worker.rs (gauge, verdict-closure, cold-first),
  crates/memra-server/src/main.rs (gauge raise at both send sites),
  crates/memra-engine/src/spec.rs (on_commit -> continue-verdict, loop exit).
- First receipt + iteration: first-result.log (1.61 -> 1.30 -> 0.149 progression).
- Full eval: run-full-eval.sh; raw logs/points-{contended,solo,thru}.jsonl,
  logs/ident-*.txt, logs/gate-*.log, logs/full-eval-driver.log.
