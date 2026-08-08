# Dynamic microchunk schedule for PP-2 prime

Branch: `lane/cx-dynamic-microchunk`
Base: `cbc8d76f`

## Mission

Replace the naked PP-2 prime pipeline's fixed equal-token microchunks with a
deterministic schedule that reduces exposed fill and drain work while compensating for
the increasing attention cost at later prefix positions.

This is scheduler-only. Chunk boundaries may not select arithmetic:

- `seq_end` remains request-level;
- `chunkinv35` and `tickinv35` plus both canaries remain the segmentation authority;
- the serial split remains the bit-identity oracle for the pipelined split;
- no kernel, tensor encoding, router, quantization, or layer dispatch may change.

`MEMRA_PRIME_CHUNK_SCHED=fixed` is the rollback. An explicit
`MEMRA_PRIME_CHUNK` remains authoritative and retains fixed-size semantics.

The requested `~/.lanectl/inbox/cx-microchunk.md` was absent at lane start. The
adjacent live coordination briefs reserve box2 for serving and direct new GPU work to
box1 when free. Box availability will be verified with both the shared lock and
`nvidia-smi` before every run.

## Increment 0 - current geometry and proposed schedule

### Current policy

`prime_chunk_tokens()` owns the naked internal geometry:

1. An explicit `MEMRA_PRIME_CHUNK` value is returned unchanged.
2. Otherwise the legacy cap is 4096 tokens.
3. A live PP-2 prime split with `T >= 256` uses
   `min(4096, max(128, ceil(T / 8)))`.
4. `prime_cache()` and `prime_cache_pp2_pipelined()` independently walk fixed
   `[start, min(start + chunk, T))` ranges, merging a final tail below
   `PRIME_MIN_T=16`.

The measured naked shapes are therefore approximately:

| prompt class | realized T | fixed geometry |
|---|---:|---|
| pp512 | 461 | 128, 128, 128, 77 |
| pp2048 | 1833 | 230 x 7, then 223 |
| pp4096 | 4096 | 512 x 8 |

The pipeprime sweep showed the opportunity but also the overhead limit: at pp4096,
16 fixed 256-token chunks reached 423.7 tok/s versus 417.5 tok/s for eight fixed
512-token chunks, only +1.5% despite twice the host walkers, boundary transfers, and
epilogues. The dynamic arm will therefore keep the fixed policy's chunk count rather
than buying smaller bubbles with more chunks.

### Bubble model

For chunk `i`, let `a_i` and `b_i` be its stage-0 and stage-1 times. Stage 1 cannot
start until stage 0 publishes chunk 1, exposing a fill term `a_1`. After stage 0
finishes the final chunk, stage 1 still owns the final drain term `b_N`. In the
balanced fixed case `a_i ~= b_i ~= tau`, the two edge terms are both the full
fixed-chunk stage time; the interior pairs are the work the pipeline can overlap.

Equal-token chunks also become progressively unequal in time because a later causal
attention chunk reads a longer prefix. That pushes late stage work onto the critical
path and makes the fixed final chunk an especially expensive drain.

The schedule therefore has two objectives:

- make chunk 1 short enough to publish the first stage boundary quickly;
- make the remaining chunks decrease in modeled execution time as the prefix grows,
  leaving a smaller final drain without increasing the number of chunks.

### Proposed generator

The dynamic generator is deterministic host integer math:

1. Generate the existing fixed ranges and retain their count `N`.
2. For `N < 3`, keep the fixed ranges.
3. Set the fill chunk to `max(64, ceil(fixed_chunk / 2))`. The 64-token lower
   bound is already covered by the chunk-invariance family and by pipeprime's pp512
   geometry sweep.
4. Model cumulative work at prefix length `L` as
   `W(L) = L^2 + 8*T*L`. The quadratic term captures growing causal-attention work;
   the `8*T` linear term keeps the model conservative for Step, whose prime wall is
   dominated by linear/MoE work rather than attention.
5. Place the remaining `N - 1` boundaries at equal increments of `W` between the
   fill boundary and `T`, using integer binary search. Convexity makes chunk 2 the
   largest middle chunk and then shrinks successive chunks toward the drain.
6. Clamp every remaining range to leave at least `PRIME_MIN_T` tokens per future
   chunk. The ranges must cover `[0,T)` exactly with no gaps, overlaps, or empty
   chunks.

Pre-registered example schedules:

| realized T | fixed | dynamic proposal |
|---:|---|---|
| 461 | 128,128,128,77 | 64,141,132,124 |
| 1833 | 230 x 7,223 | 115,269,260,252,244,237,231,225 |
| 4096 | 512 x 8 | 256,602,580,563,545,531,516,503 |

The seam defaults to `dynamic` only for naked auto PP-2 geometry. `fixed` restores
the existing ranges byte-for-byte, and any explicit `MEMRA_PRIME_CHUNK` continues to
request fixed ranges regardless of the schedule seam.

### Required proof

| Surface | Required verdict |
|---|---|
| pure schedule tests | exact coverage, same chunk count, fixed rollback shapes, short fill, shrinking tail |
| `chunkinv35` / `chunkinv35c` | invariant / canary teeth |
| `tickinv35` / `tickinv35c` | invariant / canary teeth |
| `ppsplit` | fixed serial versus dynamic pipeline bit-identical and both schedules live |
| `ppsplitc` | overlap liveness fails when only the pipeline arm is forced serial |
| model-backed acceptance | `kernel-check` green, `run-gen` argmax match, `run-spec` K=1..8 pass |
| performance | pp512/2048/4096 dynamic versus fixed, N=5 interleaved under one GPU-lock hold |

Raw logs will live under `research/microchunk-20260808/raw/`. Every reported median
will state N and the thermal/lock regime.

## Increment 1 - shared range generator and fixed rollback

The scheduler core is implemented without changing any forward arithmetic:

- `prime_chunk_ranges()` is now the single range authority for both the serial chunk
  loop and the PP-2 pipelined loop;
- `fixed_prime_chunk_ranges()` preserves the old tail-merge behavior byte-for-byte;
- the dynamic generator uses the pre-registered integer work model and retains the
  fixed schedule's chunk count;
- naked auto PP-2 geometry defaults to dynamic;
- `MEMRA_PRIME_CHUNK_SCHED=fixed` restores the measured equal-token ranges;
- any explicit `MEMRA_PRIME_CHUNK` remains fixed and authoritative.

No kernel, layer walk, cache update, boundary transport, epilogue, or dispatch
predicate changed. The pipeline function now receives the already-generated ranges
instead of reconstructing fixed ranges internally.

Local verification:

| Check | Result |
|---|---|
| targeted test build | PASS, CUDA 13.1, auto-detected sm_120a |
| fixed geometry tests | PASS: pp512/2048/4096 shapes match the prior policy |
| registered dynamic shapes | PASS: T=461/1833/4096 match Increment 0 |
| exhaustive pure invariants | PASS for every T=256..8192: exact cover, same count, no chunk below 16, short fill, non-increasing post-fill sizes |

Command: `cargo test -p memra-engine prime_chunk_schedule_tests --lib`.
