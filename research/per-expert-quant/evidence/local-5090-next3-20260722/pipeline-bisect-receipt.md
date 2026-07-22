# 2026-07-22 gate + bisect: io/compute pipeline chain — flat e2e, mechanism fully attributed

Target: local RTX 5090 laptop, Hy3 Layer-103.5 dual-NVMe profile, 45-token chat prompt, N=32
post-freeze greedy decode, cooled 55-56 °C starts, GPU/load contamination guards. Candidate
companion = async read pipeline + paired IQ3_S/Q4_K AVX-VNNI kernels + thread-local scratch
pooling (branch `hy3-10toks`); control = merged main companion
(`26303685576126a829933144be6af7dad6a6c19995b0b90421ca196d47c31621`).

## Measurement chain (all runs in this directory)

| arm | tok/s | phase_compute | exposed io (companion) | verdict |
|---|---|---:|---:|---|
| control ×5 (pairs + anchors + probe) | 4.55 4.50 4.77 4.46 4.48 → median 4.50 | 3.0–3.2 s | 2.85–2.94 s | clean band |
| candidate, passive waits ×4 | 2.86–3.05 → ~2.9 | **8.3–8.7 s** | 0.11–0.24 s | −36%: io fully hidden but compute tripled |
| candidate + io on E-cores ×2 | 2.97 2.98 | 8.6 s | 0.6 s | core placement irrelevant |
| candidate + `OMP_WAIT_POLICY=passive` | 3.00 | 8.4 s | 0.2 s | spin-policy-alone not the fix |
| candidate + `PIPELINE=0` (serial) ×2 | 4.63 4.09 | 2.98/3.52 s | n/a | kernels+pooling exonerated |
| candidate + ACTIVE spin + E-core io ×3 | 4.70 4.47 4.55 → median 4.55 | 2.72–2.99 s | 1.92–2.17 s | band recovered; **e2e flat vs control** |
| + engine core-0 reservation (7 threads) ×2 | 3.93 4.66 | — | — | no rescue; engine threads float |

Ruled out with direct measurement: core preemption (involuntary context switches 61k→3.8k in
the E-core arm with no speedup), OMP spin policy alone, package power collapse (P-core clocks
sag ≤10%, `freq-*.txt`), and DMA/memory interference (cache-hot compute bench unchanged beside
two full-rate O_DIRECT streams).

## Mechanism (confirmed)

1. The pipeline's consumer loop inserts cv-wait gaps between many small OMP regions. With
   default passive waits, workers sleep at each gap; every ready-batch region entry pays a
   futex + C-state cold wake, ~10k times per 32-token window → compute 3.0 → 8.4 s.
2. `OMP_WAIT_POLICY=ACTIVE` removes the wakes (compute 2.7 s, better than control), but the
   workers then spin between companion calls and starve the caller: Rust-side
   `exposed_wait` (5.7–6.1 s) exceeds companion `backend_wall` (5.0–5.1 s) — impossible
   without caller starvation. The 1.2 s the companion saves, the spinners take back.
3. Per-call overlap ceiling is also structural: a call's missing experts cannot overlap
   their own reads, so only ~0.8 s of the 2.9 s io hides under cached-expert compute.
   Hiding the rest needs cross-call prefetch (the owner's prediction lead) or fewer bytes
   (HBM residency donors).

## Decisions

- `BW24_CPU_EXPERT_PIPELINE` becomes **opt-in** (companion default off). Callers without the
  launcher's `OMP_WAIT_POLICY=ACTIVE` would silently hit the −36% passive regression.
- The Hy3 launcher opts in (`PIPELINE=1`, `ACTIVE`, io pool on E-cores 8–15): best measured
  median (4.55 vs 4.50), statistically flat, and it keeps the prefetch scaffold exercised.
- Paired IQ3_S/Q4_K kernels + scratch pooling ship unconditionally: bit-identical, micro
  −26.6%/−16.9%, e2e-neutral at the current io-bound balance (stage counters 2.7–3.0 vs
  3.0–3.2 s), zero flag surface.
- Next lever, in order: single-region-per-call restructure (waits inside the region — removes
  the wait-policy dependence and the caller starvation), then HBM residency donors
  (KV fp8/fp4 + `VRAM_FRAC`), then cross-call prefetch on the pipeline scaffold.

Correctness this session: post-freeze argmax `40129 == 40129` MATCH on every scored arm,
run-spec K=1..8 self-consistency PASS (candidate), `kernel-check` ALL GREEN, perf-ci quick
0 fail. Anomaly noted: `spin-arm-3` froze 9,050 CPU-routed experts vs 8,809 elsewhere
(profile-admit timing sensitivity) — its 4.55 slightly understates that arm.
Non-scoring contended runs quarantined under `premature-migration-window/`.

## Addendum — the shipped winner (same day, `v2-*`/`final-*`/`v4` logs)

Two earlier probe rounds (`pool-*`, `stage-pipe-stage`) ran a stale binary: the buffer-pool
commit did not compile (nested-class NSDMI made the deleter non-default-constructible) and
the build script's exit status went unchecked. Those rounds are void. After the fix:

| arm (serial io, pipeline off) | runs | median |
|---|---|---:|
| control (merged main companion) | 4.55 4.50 4.77 4.46 4.48 | 4.50 |
| + paired kernels + scratch pooling + RawBlockPool buffer recycling | 4.92 4.72 4.67 | **4.72** |
| + 2 MB-aligned blocks with `MADV_HUGEPAGE` | 4.84 4.80 | **4.82** (N=2) |

Decode-window compute: control 3.0–3.2 s → pool 2.63–2.99 s → THP 2.76–2.85 s; unchanged
io volume; argmax MATCH on every run. Net **+7% e2e** over the same-day control band and
+4.8% over the 2026-07-21 4.60 receipt, from CPU-side memory-system work only.

Stage-split instrumentation on the pipelined path localized its inflation inside the
worksharing compute loops of both cached and missing subsets (entry/wait/accumulation
innocent), with the serial path fast under the identical region structure — consistent with
concurrent-DMA fabric interference (snoop/invalidation traffic across the rotating buffer
space). The planned `IO_THREADS=1` trickle-DMA confirmation run was lost to a host
interruption and has not been obtained; the interference mechanism is the surviving
explanation, not a confirmed one. The pipeline scaffold (IoPool, per-expert readiness,
single-region wait) stays in the tree behind the opt-in flag for the cross-call prefetch
lane, which does not require full-rate concurrency.
