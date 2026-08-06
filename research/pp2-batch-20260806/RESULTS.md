# pp2-batch — batched decode over PP-2

Rig: **2x RTX PRO 6000 Blackwell Server Edition** (96 GB each, `18.195.123.14`, SPOT), sm_120a,
nvcc auto-resolved to CUDA 13.2. Date 2026-08-06. Branch `lane/pp2-batch`.
Models: `Qwen3.5-9B-NVFP4-MTP-GGUF.gguf` (32 layers + MTP head), `Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf`
(64 layers), both on the box's local NVMe (`/scratch-models`).
GPU windows held under `flock /tmp/memra-gpu.lock` (box shared with the step37-p2 lane).

## What this lane was for

Four decode paths (batch / dc / graph / spec) walked the full layer trunk with no pp awareness,
so under a sharded cross-device placement each one peer-read every remote stage's weights on
every step — measured 13.9-28x slower by the pp2-hardening lane, which made all four **fail
closed** behind `pp::refuse_unsplit_if_remote`. Step-3.7-Flash (105 GB) fits only across the
pair, so without a pp-aware batched path that SKU serves single-stream only.

This lane makes the **batched** path legitimately pp-aware. The refusal is lifted for it: the
stage-split dispatch now precedes the guard, and the guard survives to cover the residue
(`MEMRA_BATCH_PP=0`, `MEMRA_PP_STREAMS=0`, or a placement whose `PpNRt` fails to build).

## Exactness: batched PP-N adds ZERO deviation

`decode-batch-gate --mode pp` (new). Per arm it records a reference with the door OFF over the
same loaded weights, then replays the same input token sequence through the split and compares
**every f32 logit of every row of every step, bit by bit**.

| arm | config | widths | verdict |
|---|---|---|---|
| dev01 | stages=2, fence [0,16,32] | B=1/4/8 x3 reps | **0 differing bits** (7.9M / 31.8M / 63.6M logits) |
| dev10 | reversed placement | B=1/4/8 x3 reps | **0 differing bits** |
| singledev | stages=2, one card (seam only) | B=1/4/8 x3 reps | **0 differing bits** |
| split5 | fence [0,5,32] (uneven cut) | B=1/4/8 x3 reps | **0 differing bits** |
| N=4 | devices 0,0,1,1, fence [0,8,16,24,32] | B=1/4/8 x2 reps | **0 differing bits** |
| q27 | 64 layers, dev01 | B=1/4/8 x2 reps | **0 differing bits** |
| wide | B=12/16 under `MEMRA_DECODE_BATCH_CAP=16`, dev01 | B=12/16 x2 reps | **0 differing bits** |

Two things make this a localizer rather than a pass/fail coin flip:

- the **`unsplit@ppncache` arm** replays the unsplit walk over the SAME stage-owned caches, so
  cache placement is held constant and only the walk varies. Green on every config, so a red
  split arm would have pointed at the stage split and nothing else.
- the **`epilogue` arm** runs mixed per-row metas (even rows device-sampled greedy, odd rows
  host) and checks the lean `last_logits_dev` park through UVA from the primary context — the
  same read the server's retire path does.

The **shared-Engine scratch race** (the 2026-08-02 35% flake: `fa_part_pool` /
`argmax_partials` / `fa_vf16_scratch` are stable-pointer pools that are single-stream-safe by
design) did **not** fire in 3 replays per width per config. Per-stage `Engine` isolation holds
at batched widths, which is why reps are in the gate: one green replay is not evidence of
absence for a 35% flake.

The **exact-16 arm as first written was an invalid arm**, not a pp failure. Quoted:

```
thread 'main' panicked at crates/memra-engine/src/decode_batch.rs:474:9:
decode_step_batch: B=12 > cap 8 with no exact tier (Q8_0 m>8 needs the q8rp mirror's b16
class; m>16 crosses GEMM/dp4a numeric configs) — refused
```

It fired in the door-OFF reference, before any split ran: `decode_batch_exact16_ok` admits only
Q4_0/Q6_K/F8_E4M3/Q8_0+rp4, and both box models are NVFP4, so neither has an exact-16 tier to
admit. `decode_step_batch_ppn` duplicates the same assert, so the split does not bypass the
width policy. The replacement runs both sides under the `MEMRA_DECODE_BATCH_CAP=16` measurement
door, which tests strictly more for this lane: the m>=16 GEMM-tier kernel family crossing a
stage boundary.

## Standing battery (door shut — the split must not move single-device behavior)

`kernel-check` ALL GREEN · `decode-batch-gate` config gate1/2/3 PASS · strict
(`MEMRA_MMVQ=0 MEMRA_NO_FUSE_NORMQ=1`) gate1/2/3 PASS · `run-gen` **MATCH** (prefill
argmax=268, decode argmax=268) · `ppn-gate` serial + pipelined BIT-IDENTICAL.
Refusal residue: `MEMRA_BATCH_PP=0` over dev01 exits nonzero with the full refusal text.

## Serving over the split

`tools/serve-smoke.sh`, three arms, one binary:

| arm | config | failed checks |
|---|---|---|
| A | door shut, spec on (its own default) | **0** |
| B | **PP-2 dev01**, `MEMRA_SERVE_SPEC=0` | **0** |
| C | door shut, `MEMRA_SERVE_SPEC=0` (control) | **0** |

B minus C = **empty**: the split adds nothing. Arm B's server stdout carries the
`[pp] cross-device transport: stage0=dev0 stage1=dev1` banner and arm C's does not, so the
liveness check is proven able to read zero where zero is. Checks covered: `/models`, chat
non-stream, chat SSE stream, `/v1/completions`, greedy determinism (2 runs identical), 3
concurrent chats, long generation (>=100 tok).

### `MEMRA_SERVE_SPEC=0` is load-bearing, and finding out why is a result

q9 carries an embedded MTP head, so the server **self-specs by default** and every request
funnels through the spec verify trunk — still unsplit, still failing closed. The repro returns
HTTP 400:

```
step error: decode_step_t (spec verify): refused with the ppN door open across 2+ devices
```

with the server log confirming the split loaded. So **batched PP-2 serving is real today on the
non-spec path only.** That is a serving fact, not a gate artifact, and it is the top item on
what remains.

## Capacity: what batched PP-2 costs

q9, 64 steps, 512-token prompts, greedy, **N=5 rep-major interleaved in one lock hold on one
binary** (all four arms back to back inside each rep — cross-run comparisons are clock-drift
invalid). Thermal regime: 26-27C / 180 MHz idle at start, ~32-36C at 2325-2377 MHz through the
run, both cards at 0 MiB before and after. Medians of 5; `MEMRA_DECODE_BATCH_CAP=16` applied to
**all** arms equally (q9 has no exact-16 tier, so B=16 would otherwise refuse — the B=16 column
describes the door-open tier, not shipped behavior).

| arm | B=1 | B=4 | B=8 | B=16 |
|---|---|---|---|---|
| A door shut, single device | 208.5 | 489.3 | 654.0 | 269.4 |
| B split stages=2, single device (seam only) | 178.1 | 489.4 | 650.6 | 269.2 |
| C split dev01 (**the serving config**) | 177.5 | 487.0 | 646.9 | 265.5 |
| D split dev10 (placement symmetry) | 177.3 | 486.3 | 647.0 | 266.6 |
| **C/A** | 0.851x | **0.995x** | **0.989x** | **0.986x** |
| B/A (seam alone) | 0.854x | 1.000x | 0.995x | 0.999x |
| C/B (transport alone) | 0.997x | 0.995x | 0.994x | 0.986x |

**The answer the Step SKU needed: the `[B, n_embd]` boundary transfer does NOT bite at m>1.**
Batched PP-2 costs **0.5-1.5%** at B=4/8/16 — the same order as the eager serial path's ~0.4% —
and the transport is 0.986-0.997x of the seam, so almost all of the small loss is the seam, not
PCIe. Both placement orders agree within 0.3%. At B=8/16 the A and C ranges are disjoint
(max C 648.3 < min A 652.2), so the loss is real and small rather than noise.

Aggregate scaling under the split is intact: B=8 reaches 3.65x B=1's aggregate.

### The B=1 finding: -14.9%, and it was never a split cost

Arm B pays the identical 178 on ONE card, and the prior lane's `MEMRA_PP_SHARD=0` batched-body
B=1 was 178.5. Cause: the `b1_fast` guard includes `pp_cuts().is_none()`, so **opening the pp
door dropped every solo session off the m=1 fusion chain** (cross-layer add+norm+q8_1, fused
SwiGLU, lever 1's gate+up dual) and onto the batched m=1 walk. On a SKU that only serves with
the door open, that is a permanent 15% tax on exactly the request shape a 2-card box serves
interactively.

Fixed by giving the split its own B=1 path: each stage runs its range through
`decode_layers_eager(fence[s], fence[s+1])` — the same per-stage call `decode_step_h_ppn`
already makes, same engines, streams, boundary slots, stage-owned caches. Exactness bar for it
is bit-identity **to the eager split arm** (`decode_step_h`), not to the batched body, against
which it carries the accepted m=1 fusion FP gap by design — which is why the pp gate pins
`set_b1_fast(false)` for arms 1-3 and added **arm 4** to measure the bar that actually applies,
with a per-step `pos` equality assert so a double-advance cannot hide behind matching logits.

**Arm 4 verdict, all six configs (dev01, dev10, singledev, split5, N=4, q27):
3,973,120 f32 logits BIT-IDENTICAL, 0 differing bits.**

**Recovered, re-measured on the fixed binary** (same script shape, N=5 rep-major interleaved,
one lock hold; the `E` arm is N=3):

| arm | B=1 median | min-max | vs A |
|---|---|---|---|
| A door shut, single device | 208.4 | 207.7-208.9 | 1.000x |
| B split stages=2, single device | 205.6 | 205.4-205.8 | 0.987x |
| C split dev01 (**serving**) | **204.7** | 204.2-205.1 | **0.982x** |
| D split dev10 | 204.8 | 204.2-205.1 | 0.983x |
| E split dev01, `MEMRA_SERVE_B1FAST=0` (rollback control) | 177.4 | 177.3-177.4 | 0.851x |

**177.5 -> 204.7 = +15.3%**, and B=1 over the split now costs 1.8% instead of 14.9%. The `E`
control matters as much as the win: with the rollback seam off, the same binary in the same lock
hold reproduces the pre-fix 177.4 to within 0.1 tok/s, so "the fix worked" and "something else on
the box changed" are distinguishable observations. Every arm's range is disjoint from every
other's.

## Also fixed here (found while building the harness)

**Two paths allocated stage 1's KV on the wrong card.** `memra-server`'s worker (session cache,
prefix-cache restore, post-eviction retry) and `decode-batch-bench` built caches with
`Cache::new`, which homes everything on the primary device. Under an open cross-device door that
makes every remote stage peer-read its OWN KV every step — the same silent-PCIe class the
refusal exists to prevent, on the serving path itself. All now go through `pp::new_cache`, which
is `Cache::new` verbatim with the door shut, so single-device behavior is byte-unchanged. Left
unfixed, the lane's own perf receipt would have charged the split for a harness bug.

**`build.rs` hardcoded `/usr/local/cuda-13.1/bin/nvcc`.** This box has 13.2 / 13.0 / 12.9 /
12.8 and no 13.1 at all. Now resolved by version rank (newest wins) with `MEMRA_NVCC` and
`CUDA_HOME` respected first. First-hit-on-PATH was tried and rejected because it breaks a
working rig: `/usr/bin/nvcc` is 12.4 and emits
`nvcc fatal : Unsupported gpu architecture 'compute_120a'`.

**Two harness bugs in this lane's own scripts, caught and fixed rather than reported as
findings.** The liveness check read `/tmp/serve-smoke.log` after a later arm had overwritten it,
and its `||`/`&&` chain's precedence fired the failure branch on a run that WAS split-live.

## What remains for spec-over-PP2

1. **The verify trunk is the next stage split.** `decode_step_t` is the single funnel every
   verify forward reaches, and it is a batched T=K+1 forward — structurally the same problem
   this lane just solved, so `decode_batch_layers(lo..hi)` plus the boundary-slot pattern
   should port. The payload becomes `T x n_embd` with T = K+1; the grow-only slot sizing landed
   in this lane already handles that (`bf.len() < n`, with `rx` slicing to `n`).
2. **Two knobs cannot both be default on a 2-card SKU today.** With the pp door open, serving
   requires `MEMRA_SERVE_SPEC=0`. Whichever lands first, that combination needs a serve-smoke
   arm with spec ON over the split before the Step SKU can serve its intended config.
3. **`dc` and `graph` still fail closed** (`decode_step_dc`, and the capture paths transitively).
   They are not on the batched serving path, so they did not block this lane, but they are the
   remaining two of the original four.
4. **Draft-model placement is unexamined.** Spec over PP-2 has to decide where the draft lives
   (co-resident with stage 0, split too, or pinned) and whether its KV follows `pp::new_cache`.

## Receipts

- `logs/gates/` — the PRE-fix gate battery (incl. the invalid-arm panic verbatim), both
  `gpu-state` snapshots, the refusal text verbatim
- `logs/gates-postfix/` — the same battery re-run on the B=1-fix binary, with arm 4 on every
  config; `logs/POSTFIX.log` is the combined driver log (gates + B=1 perf + serve, one file)
- `logs/serve/` — the three serve-smoke arms, the fail-set diffs, the server stdout with the
  transport banner, the wide-width B=12/16 log
- `logs/perf/` — 20 bench logs (4 arms x 5 reps) + gpu pre/post
- `logs/perf-b1/` — the B=1 re-measure after the fix, incl. the `MEMRA_SERVE_B1FAST=0`
  rollback-seam control
- `logs/nvcc-resolve-local.txt` — the 5-arm local compile matrix behind the build.rs change
- scripts: `run-ppbatch-gates.sh`, `run-ppbatch-serve.sh`, `run-ppbatch-perf.sh`,
  `run-ppbatch-b1.sh`
