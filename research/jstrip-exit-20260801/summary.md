# j-strip early-exit in vivo — mmq_iq_experts ragged token tiles (swapab-probe residue)

Lane 4, darklanes-8x GPU 4 (H100 80GB, driver 595.71.05; 40C / 1830MHz at measurement end).
Branch: lane/jstrip-exit. Change: `crates/memra-engine/cu/mmq_iq_experts.cu` —
`vec_dot_mma` gains a `jexit` template bool + `j_max` arg; ragged tiles skip j0 strips whose
8-token block lies entirely past `j_max` (load-balanced: each warp's strips stride the whole
token axis). Call-site dispatch (`vdot` lambda): full tiles (`cnt == mmq_x`) take the
jexit=false instantiation — the guard folds away, full-tile codegen unchanged (the naked
runtime check cost +2.1% at m=128 in the swapab microbench, research/swapab-20260801/).

## Byte-identity contract

Skipped strips satisfy `j0+joff > j_max`; every column they feed is `j > j_max` — exactly the
writeback's discarded set (`if(j>j_max) continue;`). Boundary strips are kept. Verified in vivo:
base vs exit run-gen non-timing output IDENTICAL on both models (same prefill/decode argmax,
same logit maxdiff to the CPU oracle, identical text):
- q35: argmax 485/485, maxdiff 8.402e-1, MATCH (gate-q35-{base,exit}.log)
- g26: argmax 623/623, maxdiff 5.527e0 — the round-46 constant — MATCH (gate-g26-{base,exit}.log)

## Gates (GPU 4, exit build)

- kernel-check: **ALL GREEN** (kernel-check-exit.log). NC26 + RAGK pins OK. The dtype5
  IQ3_S/IQ4_XS and D.2 entries SKIP on this box (hardcoded 5090-rig model paths
  /home/avifenesh/... and /data/... do not exist here) — same skip set as rounds 44-47.
- run-gen argmax: q35 board-2048 **MATCH**; g26 depth-1736 **MATCH**.

## Perf (interleaved x3 pairs, each = MEMRA_PP_ONLY MEMRA_PP_REPS=5 in-process medians; raw perf-x3.log)

| cell | base medians (x3) | exit medians (x3) | delta (med/med) |
|---|---|---|---|
| q35 board-2048 prefill | 5413.2 / 5416.3 / 5394.2 | 5424.5 / 5423.5 / 5427.3 | **+0.21%** (5413.2 -> 5424.5) |
| g26 pp1736             | 10045.8 / 10053.2 / 10053.5 | 10442.0 / 10428.7 / 10439.2 | **+3.84%** (10053.2 -> 10439.2) |

Both deltas have non-overlapping base/exit ranges across the 3 interleaved pairs.

- q35: +0.2% — as predicted by the probe: the (4,252) gate/up form is long_scoreboard-bound
  and q35's expert-MMA share of prime is small; the mma-slot saving mostly hides under latency.
  Consistent-positive, not noise (ranges disjoint), but e2e-invisible — matches the probe's
  <=0.5% e2e ceiling arithmetic.
- g26 control: NOT flat — a real **+3.84%** pp win. The "must be flat" expectation missed that
  g26's 147-236-pair groups split into one FULL tile (codegen-unchanged false path — the actual
  regression control, which held) plus one RAGGED second tile (cnt 19-108, where the skip
  removes up to ceil8(19)/128 = 81% of that tile's mma), and g26's expert MMA is ~95% of its
  timed prime (round-45 nsys). The win is the residue paying exactly where mma-share is high,
  as the probe predicted for compute-bound forms.

## Files

- run-gen.base md5 09432d4a1767f3831c0a549db34fe32e; run-gen.exit md5 01d2645a906b1a6011fbf94ebbb0cab7
  (binaries kept on box only, ~/lane4/research/jstrip-exit-20260801/)
- build: MEMRA_CUDA_ARCH=90a, nvcc 13.1 (/usr/local/cuda-13.1), cargo release (build-exit.log)
- jstrip-perf.sh = the exact interleave script (baked literals)

Residue for the lane owner: the down (16,256) form (SM 59.9% short-scoreboard, same ~65-pair
groups) is inside these q35 numbers already; the g26 result shows the exit pays on
compute-bound forms. 5090 battery re-run required before any default claim (Blackwell is the
primary target; this lane is sm_90a evidence).
