# graph-warmups 5090 re-mint + stress gate — VERDICT: default FLIPPED to 1

Lane: `lane/graph-warmups` (from restructure/public-split b1f7b84e). Rig: local RTX 5090
Laptop (82 SM, 24GB), shared with the fp8-blk128 lane via `flock /tmp/gpu5090.lock`.
Pod finding being validated: `research/graph-allocfree-20260805/logs/warmup-lever-N5.txt`
(MEMRA_GRAPH_WARMUPS=1 → recapture −38% q27 / −41% q9, decode +1.1%, gates PASS — but
default NOT flipped there because two models passing was evidence, not proof, against the
pool-growth hazard warmup 2 exists to absorb).

## 1. Re-mint: the win TRANSFERS to 82 SM

`logs/remint-warmups-N5.txt`. N=5 ADJACENT pairs per model, lock held across both arms of a
pair (the fp8 lane's competing ~4-min prefill burns would otherwise land between arms), arm
order alternating per rep (2,1/1,2/...). Probe = graph-allocfree-probe, --reps 5 medians
inside each invocation. Thermal: 55-56C at start, clocks 1590MHz reported per pair.

| quantity | q27 w=2 | q27 w=1 | Δ | q9 w=2 | q9 w=1 | Δ |
|---|---|---|---|---|---|---|
| recapture (ms) | 52.6 | 30.8 | **−41.4%** | 19.4 | 11.8 | **−39.2%** |
| capture+prime (ms) | 1130.9 | 1110.5 | −20.4 ms | 413.4 | 407.0 | −6.4 ms |
| decode (tok/s) | 46.56 | 47.06 | **+1.07%** | 132.40 | 133.87 | **+1.11%** |

Pairwise wins for w=1: 5/5 on recapture AND decode, both models. Distributions disjoint
(q27 recapture: w=1 max 31.7 < w=2 min 52.2; q9: 12.4 < 19.2). Same shape as the pod
(−38/−41% there): the lever is eager-step wall time, node-count- and SM-count-invariant.

## 2. The stress gate (the flip decision's real deliverable)

Mechanism first (read from `capture_graph` + the #68 receipts): warmup 1's allocations may
grow/map the async pool; warmup 2 re-walks the same alloc/free sequence over the freed
blocks so the captured third run bakes settled addresses. A wrong flip reproduces the #68
stale-baked-address class — which corrupts token streams WITHOUT a CUDA fault (fp8ship
2026-08-04), so the gate's arbiter is per-token BIT-IDENTITY vs eager decode_step, not
fault-freeness alone.

Why the structural guards make warmup 2 redundant (the theory the gate then attacked):
- in-body transients are captured as BALANCED in-graph alloc/free node pairs (census 1589
  ALLOC / 1589 FREE) — replays allocate for themselves, no baked transient pointers;
- every externally-referenced buffer is stable-pointer by design: fa_part_pool
  retires-on-grow and never frees (#68's fix), counters/mask/cache set in place;
- the draft-graph path rides capture_graph_retained (capture_keep holds every
  warmup+capture alloc alive for the session).

`graph-warmup-stress` (new bin) attacks exactly the residual hazard, warmups=1:
- **large→small**: 4096-budget session boots (pool GROWS — receipts: q27 reserved
  14.0→15.4GB, q9 4.7→5.4GB), generates through natural recaptures + one FORCED mid-stream
  recapture, retires (its GBs become freed pool blocks); a 96-budget session then captures
  OVER those freed blocks (reserved−used gap ≈1.4GB q27 / 0.8GB q9 at capture time).
- **small→large**: reverse order — the large session's warmup 1 must itself grow the pool.
- **overlap arm**: two LIVE graph sessions share the engine pools; the large boot grows
  fa_part_pool under the small session's baked pointers; the small session is dropped
  mid-flight and the survivor takes a forced recapture over its freed blocks + keeps
  generating (the F5-adjacent park/resume-shaped path — worker promotion rides the same
  graph_capture_segment).
- x10 cycles, both directions, every stream diffed token-by-token vs eager.

Verdict: **q9 10/10 cycles + overlap ALL GREEN; q27 10/10 cycles + overlap ALL GREEN**
(`logs/stress-q9-x10.txt`, `logs/stress-q27-x10.txt`). Zero mismatches, zero faults.

Teeth: `--canary` clobbers a graph-referenced buffer (token_d) mid-stream and the
comparator CAUGHT it at token 30 (`logs/stress-canary.txt`). A true cross-allocation alias
is not deterministically constructible from user code (cuMemAllocAsync exposes no placement
control), so the canary corrupts graph-read memory directly — it proves the comparator +
plumbing detect graph-memory corruption end-to-end; it changes the world, not the label
(the chunkinv-canary trap).

## 3. Flip + gate wiring

- Default flipped: `MEMRA_GRAPH_WARMUPS` unset → 1 (lib.rs `capture_graph`); `=2` is the
  rollback seam. FLAGS.md updated.
- `tools/graph-warmup-stress-gate.sh` = the runnable gate (stress x10 + canary arm).
  Wired into `tools/local-ci.sh` correctness stage (MEMRA_CI_GWSTRESS=0 skips) and
  `tools/fast-gate` as the `gwstress` cmd probe (3-cycle tier-1 variant; graph-path +
  DEFAULT rows) — the H100 lane law: gates outside the battery rot silently.

## 4. Post-flip battery (naked env = the new default under test)

`logs/gate-*.txt`: kernel-check ALL GREEN; graph-decode-gate 256-step bit-identity PASS
q9+q27; graph-session-gate PASS q9+q27; run-gen argmax MATCH q9+q27; run-spec K=1..8
self-consistency PASS (q27 NVFP4+MTP arm); serve-smoke PASS.

## Refutation bar (kept)

One reproduced stale-address divergence in graph-warmup-stress at warmups=1 = the flip is
REFUTED: revert the default to 2, keep the door and this gate, document the reproducing
config here.
