# Verify megakernel — the last engine lever for the 31B spec-depth cell (2026-07-13)

> **CLOSED 2026-07-14 (falsification #7, jsonl row):** the persistent counter-barrier
> form was built bit-exact (FFN slab: entry-quantize -> gate|up -> gelu+q8 -> down, one
> co-resident launch) and LOST on both dense cells (31B depth −0.3%, E4B −2.3%,
> interleaved N=3): PDL glue launches already hide the boundaries it fuses, and it pays
> grid-wide barrier latency + worst-segment occupancy for them. Down's activation
> dependency is all-to-all, so the sentinel refinement cannot rescue the down segment.
> The extractable share of the launch-tail tax was the f2/f3 independent-pair tail-fill
> (+5%, shipped v0.33.0). Landmine for future co-resident kernels: size gridDim from
> THAT kernel's occupancy (a shared per-process cache deadlocked the b4 width at 100%
> GPU), and remember single-K gates cannot expose per-kernel-cache bugs.

## Why this, why now

The 31B spec depth cell (0.893x, THE open goal front) is verify-wall-bound: the MTP round
is 33.7 rounds/s x 29ms batched verify ≈ 0.98 of wall — drafts ride graph replay ~free.
The verify reads the whole 17.4GB q4_0 trunk once per round; byte floor = 20.5ms at the
847GB/s wall. The b-tier verify runs at ~71% of that floor, and SIX bit-exact kernel
variants have now been falsified against the plateau (jsonl rows, all N>=2 interleaved):

| variant | mechanism | result |
|---|---|---|
| r1 | one row/warp at batched m | −25% |
| ms | m-split across warp pairs | flat (regs stay 72) |
| sm | smem slab staging (+bank pad) | −11% |
| la | register load-ahead | flat (nvcc already schedules) |
| lb | __launch_bounds__(128,10) reg diet | −13.6% (spills) |
| ca | cp.async 3-stage split-plane ring (bit-exact, tail arm) | −2.7% (staging round-trip > latency hidden; matches NVFP4's −11.3% verdict) |

Conclusion: the plateau is STRUCTURAL to the kernel-per-op launch pattern — per-launch
drain tails and inter-launch gaps, not in-kernel load scheduling. ncu signature at depth:
DRAM 66.8%, No-Eligible 75%, achieved occupancy 51.3% (theoretical 66.7% at 64 regs).

Cell math: floor-perfect verify = 124 tok/s = 1.26x at CURRENT acceptance — the whole
0.893x -> 1.1x gap fits inside verify efficiency. The SOTA sweep's megakernel verdict
(research/SOTA-SWEEP-2026-07-13.md §2d) gives the honest band: **+10-24% over an
already-graphed engine** — exactly the 71% -> ~87% class needed — with the precondition
"do the cheap levers (2a/2b) first", which is now satisfied with measured verdicts
(weight prefetch, PDL, all six b-tier variants).

## Mechanism (why a persistent kernel attacks THIS plateau)

Each of the ~400 verify launches/round fully drains before the next starts: the tail
wave of a 70us matvec idles a growing fraction of the 82 SMs behind a kernel boundary.
A persistent schedule lets the next matvec's row-tiles start on freed SMs the moment
their INPUTS are ready (sentinel data-path flags, ~0.8us, vs ~1-2us x 400 kernel
boundaries + tails). The per-(row, column) dp4a program stays VERBATIM — FP order
per row identical, the parity law holds by construction.

## Design constraints (from the sweep's honest-negative list + house rules)

- STATIC compile-time schedule. No on-GPU interpreter (MPK-style interpreters lose).
- Sentinel data-path sync: a flag word per producer tile, spin on consumer entry
  (0.8us class, NOT counter barriers at 7.6us).
- DECODE/VERIFY-ONLY: prefill keeps the GEMM path; plain decode keeps kernel-per-op
  (its t=1 walk already runs 84% of floor — the plateau is a BATCHED-verify disease).
- Bit-exactness gates: kernel-check pins each fused stage vs the standalone kernel;
  VERIFY-GATE logit maxdiff 0.000e0 at depth (the E4B lesson: agreement-only gates lie);
  run-spec 256-gen self-consistency.
- Scope stage 1 = the 31B dense verify chain ONLY (61 uniform layers, no MoE): per layer
  qkv (fused3-class) -> rms/rope glue -> append -> fa rows -> wo -> ffn glue -> gate/up
  -> down. MoE (26B) and E4B come later if stage 1 pays.
- The fa stage keeps its split-K structure in-kernel (the one-partition law's ns_eff
  derivation ports directly; partials in smem/global as today).
- Rollback seam: whole arm behind BW24_VERIFY_MK=1 until it wins a valid window; the
  jsonl row is the record if it loses (falsification #7 would close the front for real).

## Expected magnitude, honestly

Launch/drain structure ~= the 29 - 20.5 = 8.5ms/round tax; recovering half of it =
~+17% on the depth cell (0.893 -> ~1.04); recovering most = ~+29% (-> ~1.15, CROSSES).
The sweep band (+10-24%) brackets the same range. If the persistent form ALSO enables
in-kernel next-stage weight streaming (the E4B wo-prefetch mechanism, now without kernel
boundaries in the way), the upper half of the band is reachable.

## Not part of this lane

Drafter/acceptance work is the owner's lane — this plan holds acceptance FIXED and moves
only the engine denominator.
