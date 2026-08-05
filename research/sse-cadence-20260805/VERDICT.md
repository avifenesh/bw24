# SSE cadence fix — VERDICT (2026-08-05, local 5090, lane/sse-cadence)

## The change

Streaming emission for spec-burst sessions moved from ONE `Event::Token` per
`MEMRA_SPEC_BURST` to one flush per spec-round commit.

- Engine seam: `generate_spec_inner2` gains an `on_commit: Option<&mut dyn FnMut(&[u32])>`
  hook — called after the prime's first token, after every round commit (step 4), and after
  a round-stream ring drain. Slices are disjoint, in order, concatenate to the returned
  vec. Emission timing only; token bytes, session state, exactness untouched.
- Worker: `step_session`'s spec arm passes a closure that mirrors the post-burst emission
  exactly — same detokenize-tail + `utf8_delta` cursor, same EOS-text-never-streamed rule,
  same disconnect-abort semantics (mid-burst send failure latches; abort fires post-burst,
  the pre-fix abort point). The post-burst send now covers only the remainder (held
  multi-byte UTF-8).
- Rollback seam: `MEMRA_SSE_PER_BURST=1` restores per-burst emission (documented in
  FLAGS.md). finish_reason/usage still ride `Event::Done`, unchanged.

## TTFT (q27 nv, K=3, stream greedy 256 tok, N=5 medians, one flock hold, 54-76C)

| arm            | first text | chunks/resp | gap p50 |
|----------------|-----------|-------------|---------|
| B32  fix-on    | 0.123s    | 89          | 27ms    |
| B32  fix-off   | 0.410s    | 8           | 299ms   |
| B128 fix-on    | 0.124s    | 89          | 28ms    |
| B128 fix-off   | 1.161s    | 2           | 1.43s   |

fix-off reproduces the spec-levers flip-blocker exactly (0.41s/1.15s). fix-on: burst size
no longer touches streaming cadence — first text ~0.12s, per-round chunks, at ANY burst.
That is 3.3x better than the old B32 default's felt TTFT, not just B128 rescued.

## Throughput no-regression (load-serve 128tok temp0.7, alternating boots x2/arm, err=0)

- c=1 B128: fix-on 100.0-100.3 agg tok/s vs fix-off 99.8-100.2 — parity.
- c=8 B128: fix-on 98.0-99.6 vs fix-off 97.9-98.8, interleaved in-hold — parity
  (differences within the visible thermal drift 79->84C).
- The +7% lever survives: B128 fix-on vs B32 fix-on = +7.7% c=1 (100.1 vs 92.9),
  +7.5% c=8 (98.4 vs 91.5), same window.

## Exactness / gates (fix-on binary)

- Greedy streamed content byte-identity (concat of every SSE delta, 522B):
  B128 on==off, B32 on==off, B32==B128 — cmp-clean.
- run-spec K=1..8 nv+draft BURST=128: rc=0, 9 PASS.
- serve-smoke: rc=0, 0 failed.
- serve-st-gate (default ckpt, all 4 items incl. the spec-on prefix gate): rc=0, 0 FAIL.
- cargo test -p memra-server: 72 passed, 0 failed (incl. the streaming-utf8 suite).

## Burst-default recommendation: HOLD B32 interactive default

The cadence fix removes the STREAMING blocker, but burst size still sets round-robin
admission latency: a new request joining a server with a burst in flight waits the whole
burst out. Measured (contended first-text, one 512-tok request in flight, N=3, fix-on):

| arm  | contended first text |
|------|----------------------|
| B32  | 0.57s                |
| B128 | 1.67s                |

1.67s felt TTFT for the second user is the same class of regression the old per-burst
emission caused for the first user — so B128 does NOT become the interactive default.
Standing posture unchanged: 32 = interactive default, 128 = documented throughput-tier
setting (c>=2 batch, judge/harvest, non-streaming API). What DID change: interactive
streams at ANY burst now open at ~0.12s instead of 0.41s, and throughput-tier configs
that also stream (harvest with progress UIs) no longer pay 1.15s/2-chunk cadence.

## Affinity interaction (#71)

Verified no interaction: the turn checkpoint is captured at prompt end inside
`generate_spec_inner2` (non-continuation primes only, before the round loop);
`spec_rewind_to_checkpoint` runs only at admit-time resume of a PARKED session
(worker.rs affinity probe), and sessions park only at retire. Rewind therefore happens
at turn boundaries, never mid-stream — the round-cadence flushes are strictly inside a
live burst and touch emission only (no checkpoint, cache, or committed-state writes).
Gate evidence: serve-st-gate + serve-smoke green with the fix on; worker fingerprint/
affinity unit tests all pass.

## Raw

logs/points-first.jsonl, logs/points-ttft-full.jsonl, logs/points-thru.jsonl,
logs/points-contention.jsonl, logs/ident-*.txt, logs/gate-*.log, per-boot server logs,
driver logs. Runners: run-first-result.sh, run-full-ttft.sh, run-thru.sh, run-ident.sh,
run-gates.sh, run-contention.sh. All on this box, shared-lock discipline (short holds).
