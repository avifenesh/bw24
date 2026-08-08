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

## Increment 2 - schedule-aware gates and benchmark driver

The existing PP schedule gate now makes the new comparison directly:

- REF: fixed ranges, unsplit whole-trunk walk;
- SERIAL: fixed ranges, serial PP stage split;
- PIPE: dynamic ranges, pipelined PP stage split.

For the naked `auto` arm, the gate prints both realized vectors and compares logits,
`h_seed`, the full hidden stack, and teacher-forced continuation logits bit-for-bit.
The explicit `513` stress arm remains fixed in all three schedules, preserving the old
pipeline regression row. Split and active-walker counters are checked against each
arm's realized range count. The canary retains dynamic ranges but forces only PIPE
back to `MEMRA_PRIME_PIPE=0`, so overlap liveness must fail while split liveness and
bits remain valid.

`concat-prime-probe` also has a new `ppschedperf` mode. Both arms keep the PP-2
pipeline live; it alternates fixed/dynamic order inside one sharded model load,
requires `chunks - 1` active-walker overlaps on every sample, and reports the realized
vectors with N=5 medians.

Standing surfaces updated:

- `tools/prime-split-gate.sh`;
- `tools/fast-gate/models.tsv` and `tools/fast-gate/map.tsv`;
- `docs/FLAGS.md` with `MEMRA_PRIME_CHUNK_SCHED`;
- `run-gates-box1.sh` for the full segmentation/acceptance battery;
- `run-perf-box1.sh` for pp512/2048/4096 under one lock hold.

Local verification:

| Check | Result |
|---|---|
| `cargo check -p memra-engine --bin concat-prime-probe` | PASS |
| both lane drivers + `prime-split-gate.sh`, `bash -n` | PASS |
| `git diff --check` | PASS |

Box1 was not free at this increment: PID 230699 (`target/release/memra-server`) held
29,322 MiB on GPU 0 and 1,036 MiB on GPU 1. No GPU work was queued.

## Increment 3 - box1 staging and fail-fast battery

Box1 later cleared to 0 MiB on both GPUs with `/tmp/memra-gpu.lock` free. The branch
was transferred without an origin push through a verified complete Git bundle into
`~/memra-cx-microchunk`.

The release build at `77040bf6` passed in 3m45s with CUDA 13.2 and auto-detected
sm_120a. The initial non-interactive invocation failed before compilation with
`cargo: command not found`; the successful retry called the installed stable
`cargo`/`rustc` binaries directly and did not invoke `rustup`.

Before launching the GPU battery, its driver was corrected to exit immediately after
the first red gate. This matches the lane stop rule: no later exactness or performance
work may continue after a failure.

## Increment 4 - box1 exactness and liveness receipt

The complete battery ran on box1 under one exclusive
`/tmp/memra-gpu.lock` hold from `2026-08-08T13:46:36Z` through
`2026-08-08T14:04:09Z`. Both GPUs were idle at the start (27-30 C, 0 MiB)
and returned to 34 C / 0 MiB at the end. The driver completed with `rc=0`.

| Gate | Result | Evidence |
|---|---|---|
| `kernel-check` | PASS | ALL GREEN on the available Step model-backed and synthetic sections |
| `ppsplit` | PASS | fixed unsplit, fixed serial, and dynamic pipeline bit-identical; all expected overlaps live |
| `ppsplitc` | PASS | forced serial replay retained exact bits but reduced overlap to zero |
| `chunkinv35` | PASS | 4096/513/512/256/64 all exact |
| `chunkinv35c` | PASS | legacy sequence-boundary seam produced the expected differences |
| `tickinv35` | PASS | budgets 0/1024/513/512/256/64 and split points 64/256/512 all exact |
| `tickinv35c` | PASS | legacy call-local seam produced the expected differences |
| `run-gen` | PASS | prefill/decode argmax 6776 matched; batched-prime/tokenwise argmax 6776 matched |
| `run-spec` | PASS | K=1..8 self-consistency identical to generate |

The dynamic `auto` geometry exercised by `ppsplit` at T=4883 was
`306,717,692,670,651,632,616,599`, versus fixed
`611,611,611,611,611,611,611,606`. Logits, final hidden state, seed hidden
state, and eight teacher-forced decode continuations had zero differences.
The dynamic pipeline reported eight split chunks and seven active-walker
overlaps. The explicit 513-token fixed rollback arm likewise had ten chunks,
nine overlaps, and zero differences.

The two `kernel-check` Qwen3.6-only sections were explicitly skipped because
that unrelated model was absent on box1; all available checks passed. Raw
build and gate logs are preserved under `raw/box1/build/` and
`raw/box1/gates/`.

## Increment 5 - interleaved performance verdict

The performance hold ran on box1 from `2026-08-08T14:18:30Z` through
`2026-08-08T14:23:22Z` under one exclusive GPU lock. The GPUs were idle and
cold at acquisition (32-33 C, 0 MiB), and the final snapshot was 41-44 C with
the benchmark allocation released. Each shape used one warmup followed by
N=5 fixed/dynamic pairs with alternating arm order inside one sharded model
load. Every measured arm retained the full expected overlap count.

| prompt | realized T | fixed median | dynamic median | median wall reduction | paired wins |
|---|---:|---:|---:|---:|---:|
| pp512 | 461 | 339.0 tok/s, 1.3600 s | 343.8 tok/s, 1.3408 s | 19.2 ms, +1.4% throughput | 4/5 |
| pp2048 | 1833 | 410.7 tok/s, 4.4629 s | 411.8 tok/s, 4.4510 s | 11.9 ms, +0.3% throughput | 5/5 |
| pp4096 | 4096 | 427.5 tok/s, 9.5804 s | 427.7 tok/s, 9.5769 s | 3.5 ms, effectively flat | 4/5 |

The bubble model predicts direction but overstates the available long-prompt
gain. Using first-plus-last chunk tokens only as a size proxy for exposed edge
work:

| realized T | fixed edge tokens | dynamic edge tokens | proxy reduction | dynamic peak middle chunk |
|---:|---:|---:|---:|---:|
| 461 | 128 + 77 = 205 | 64 + 124 = 188 | 8.3% | 141, +10.2% versus fixed cap |
| 1833 | 230 + 223 = 453 | 115 + 225 = 340 | 24.9% | 269, +17.0% |
| 4096 | 512 + 512 = 1024 | 256 + 503 = 759 | 25.9% | 602, +17.6% |

This is not a direct stage-time measurement. It explains the observed limit:
shortening the exposed edges requires larger middle chunks at fixed chunk
count, and the enlarged critical interior plus non-pipeline work absorbs most
of the modeled savings. At pp512 the legacy remainder had already made the
fixed drain short, so the dynamic schedule primarily buys faster fill and
gives some drain work back.

Verdict: retain the dynamic auto schedule and the fixed rollback seam. It is
bit-identical, live, and nonnegative across the registered prompt classes,
with a repeatable short-prompt gain. Do not claim a material pp4096 win or a
closure of the pipeline-versus-grouped gap: the long-prompt result is flat at
this N. A follow-on scheduler study should record per-stage, per-chunk timing
and solve against the measured critical path rather than further tuning a
token-count proxy.

These are pipeline-only scheduler results from this lane. They do not compose
or compare directly with the separately measured grouped-prefill result.
Exact prompts, hashes, per-repetition output, and the combined summary are
preserved under `raw/box1/perf/`.
