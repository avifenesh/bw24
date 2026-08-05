# q27 deep dive — PHASE 2 PRIORITY SPEC: close the serve-path lever gap (H1 + H3)

Status: SPEC ONLY — not implemented in phase 1. This is the binding priority for the phase-2
lane, ranked above the MTP/drafter work, because it is the single largest measured number in
the phase-1 verdict and it is pure recoverable overhead.

## The finding (phase-1 §4, RESULTS.md)

Serve c=1 misses BOTH phase-1 levers. Same board, same commit, same prompt:

| path | c=1 tok/s | vs serve |
|---|---|---|
| `run-gen` naked default (lever 1 + graph key=48 → graph arm at 128 tok) | 52.22 | serve is **−11.74%** |
| `run-gen`, graph door closed (pre-lever-2 default) | 50.28 | serve is −8.33% |
| `memra-server` c=1 | 46.09 | — |

Lever 2 *widened* the user-visible gap (−8.3% → −11.7%): the naked command now rides the CUDA-graph
arm at the 128-token shape, and the serve worker does not. For the dominant darklane shape —
single-tenant interactive — serving is 11.7% below what the same box already does naked. That is
larger than everything phase 1 landed combined (+4.82%).

## Root cause (confirmed, not inferred)

The serve worker routes B=1 through `decode_step_batch`, which:

- (a) has **no CUDA-graph door at all** — the worker runs its own tick loop; `MEMRA_GEN_GRAPH`
  lives in `generate_with`, which the worker never calls. Lever 2 (graph key 256→48) therefore
  never fires on the serve path.
- (b) dispatches the dense-FFN gate+up pair through `matmul_pre` at `b_n=1`, so **lever 1's fused
  arm in `matmul_pre_dual_noscale` never fires** on the serve path.

Receipt for (b): the serve-path A/B for `MEMRA_Q8_FFN_FUSE2` at c=1 is order-paired **+0.06%**
(pairs +0.20% / −0.25% / +0.23% — sign-flipping noise), against **+0.94% with 5/5 winning pairs**
in `run-gen`. That contrast is the proof the serve path bypasses the m=1 dispatch family.

## The two hypotheses (from the phase-1 named list, RESULTS.md §6)

### H1 — serve worker should ride the graph door at B=1

Route B=1 requests to the `generate_with`/graph-door path, or grow an equivalent CUDA-graph door
inside the worker tick loop. Recovers the lever-2 share of the gap (~+3.8% at 128-token shape on
the pod board, monotone up with generation length).

- Capture-legality: the door must keep the existing auto-close when MoE experts are on the SLRU
  cache path (capture-illegal) — same rule as `generate_with`.
- The worker's tick loop services cancellation/streaming between steps; a graph door must not
  regress stream latency or cancel responsiveness — measure p50/p99 latency alongside tok/s.

### H3 — serve B=1 should dispatch the m=1 kernel family

`b_n == 1` fast-path in `decode_step_batch` → the m=1 dispatch (`matmul_pre_dual_noscale` and
friends), so lever 1 (and every future m=1 lever) fires on the serve path automatically instead
of each one needing a batched twin. Recovers the lever-1 share (~+0.9%) and de-duplicates all
future m=1 wins.

## Gate battery for the phase-2 lane (non-negotiable)

1. **Stream identity**: serve B=1 token stream byte-identical before/after the routing change
   (greedy, fixed prompt, ≥128 tokens), plus the graph arm's existing bit-identity gates
   (`graph-decode-gate`, `graph-session-gate`) on the worker path.
2. **Serve c=1 A/B**: `memra-server` + `tools/load-serve.py`, N=5, arms interleaved, both
   orderings, server restarted per arm. Target: close toward the naked 52.22 (pod board) — the
   recoverable bound is +11.7%.
3. **No batched regression**: serve c=8 A/B must hold (the B=1 fast-path must not perturb the
   batched tick), serve-smoke + serve-st-gate 0-failed.
4. **Measurement harness rule** (phase-1 §5.3): size every serve claim on
   `memra-server` + `load-serve.py`, never on `decode-batch-bench` — it overstates batched cost
   by 35% (host argmax over n_vocab per row).

## Explicitly out of scope for this spec

- MTP/drafter phase-2 work (H2 residual sweep, H6 round-graph re-measure, H7 ssm_alpha/beta fold)
  — separately tracked; this spec is the priority ahead of them.
- Any change to the batched (c>1) dispatch — lever 3 was REFUTED at c=8 and its call site stays
  reverted.
