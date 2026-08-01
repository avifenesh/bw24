# fast-router: recovering the prefill the exactness contract paid for (2026-08-02)

Lane `lane/fast-router` (from `restructure/public-split`, a387fd7b). Rig: RTX 5090 Laptop
(24463 MiB), every GPU run under `flock /tmp/gpu5090.lock`, co-resident `llama-server
--embedding` untouched, servers killed by pid only. Builds `nice cargo build --release`.

## The problem

`MEMRA_ROUTER_PREFILL_EXACT` (default ON since lane/concat-prime-exact) routes the prefill
router (`ffn_gate_inp`, out_f=256) and the shexp sigmoid-dot (out_f=1) through decode's
m-invariant kernels. Correct — but `router_gemv_f32_w8` is a GEMV program at GEMM shape:
one 8-warp block per (expert, token) output, both operand rows re-streamed per output,
zero reuse. q35 board-2048 prefill paid −10.1% (3524 → 3167 on this session's interleave).

## The fix: a batch twin that is the same FP program

`router_gemv_f32_w8_batch`: one 256-thread block computes an 8x8 (expert x token) register
tile. Every per-row reduction chain is IDENTICAL to the w8 form — same tid-strided k order
(i = tid + j*256, one serial FFMA chain per output), same 5-step `__shfl_down_sync` tree,
same serial 8-partial warp-order fold — only WHERE operands come from changes (16 row-streams
feed 64 outputs instead of 2 feeding 1). m-invariance by construction: a row's chain never
sees m. **The fast form IS the exact form**; no new flag, same
`MEMRA_ROUTER_PREFILL_EXACT` contract, `MEMRA_ROUTER_BATCH=0` as a perf-only rollback seam.

## Exactness gate first (the whole point)

kernel-check gained a weight-oracle section: on the REAL q35 router weight, 32 m-points in
1..2048, every output bit-compared between plain w8 and the batch twin, plus m-invariance of
the twin against the plain form's m=2048 row prefixes. **mism=0 everywhere, first build** —
and it stayed 0 through every later kernel iteration. `kernel-check` ALL GREEN.

## Crossover (swept, not guessed) — `crossover-router-final.jsonl`

| t | plain us | batch us | speedup | cuBLASLt us (banned ref) |
|---|---|---|---|---|
| 4 | 3.8 | 5.6 | 0.68x | 5.3 |
| 8 | 6.5 | 6.0 | 1.09x | 5.4 |
| 16 | 12.1 | 6.5 | 1.88x | 5.9 |
| 512 | 386 | 119 | 3.24x | 20.1 |
| 2048 | 1605 | 453 | 3.54x | 68.9 |

`ROUTER_BATCH_MIN_T = 8`. Decode t=1 and spec verify t<8 keep the plain w8 form (bit-equal
either way — the crossover is pure perf). The cuBLASLt column is the m-DEPENDENT kernel the
contract bans: its k-split is exactly the reduction shape that broke serve isolation. Closing
the remaining 453-vs-69us would need a new shared reduction order for decode+prefill — a
numeric-config re-arbitration, not this lane.

Killed arms (bit-identity-green before dying, JSONL is the record): the 8x16 tile
(128 accumulators cost the occupancy its halved w-traffic needs — slower at every t,
`crossover-router-tiles.jsonl`) and the sigmoid-dot batch twin (out_f=1 is
launch-latency-bound, 0.62–0.89x at every prefill t, ~7us/layer at m=2048 —
`crossover-router.jsonl`). The shexp dot keeps its per-token form at every t.

## Prefill recovery — interleaved x5, same session, N=5 medians (`prefill-sweep.jsonl`)

| cell | exact0 (pre-fix cuBLASLt) | plain (fix as merged) | **batch (this lane)** |
|---|---|---|---|
| q35 board-2048 prefill tok/s | 3524.1 [3518–3539] | 3167.3 [3158–3178] | **3416.6** [3395–3428] |
| o35b pp512 prefill tok/s (resident) | 1079.7 [1077–1082] | 1049.4 [1042–1052] | **1074.7** [1068–1080] |

q35: 70% of the −357 tok/s regression recovered (−10.1% → −3.1% vs the banned kernel).
o35b: −0.5% vs exact0; the batch arm sits on the residency-cap lane's 1079.2 resident
baseline. Decode flat across arms (t=1 path untouched). run-gen argmax MATCH on all 30 runs.

## Found + fixed on the way: c=16 serve admission OOM (not this lane's kernels)

The serve gate first returned 13/16 and 15/16 with **zero mismatches** — the failures were
instant HTTP 400s (try1 silent, try2 status-only, try3 carries the full quoted body after
teaching `tools/check-batch-exact.py` to stop swallowing error strings and bodies):
`cache alloc failed: CUDA_ERROR_OUT_OF_MEMORY`. Resident-if-fits
(residency-cap merge) plans ~2GB reserve; sixteen 8192-ctx session caches want more, and
admission REJECTED instead of waiting. Fix in `worker.rs`: VRAM-aware admission wait — after
the first admit observes a model's per-session VRAM cost, further admissions require free ≥
2x that cost, otherwise the request waits in the existing never-rejected FIFO exactly like
the session-count cap. First session always admits; empty-active OOM still errors (real
capacity, quoted).

## Battery (shipping binary)

- kernel-check ALL GREEN (incl. the new bit-identity + m-invariance entries)
- run-gen argmax MATCH: q35 x15, o35b x15 (every sweep run)
- serve greedy c=1-vs-c=16, o35b, 16 x 96 tok: **16/16 PASS, x2 runs**
- run-spec K=1..8 self-consistency PASS (o35b + owntrim draft; acceptance counts identical
  to the concat lane's run — cross-lane routing identity)
- prime-batch-gate --carried o35b ALL GREEN

No published board numbers move (the README board carries decode/spec rows only; its values
predate the exactness fix).
