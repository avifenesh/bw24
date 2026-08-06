# pp2-spec — the spec-decode verify trunk over PP-2

**Lane**: `lane/pp2-spec`, off `origin/restructure/public-split` @ `a6601b8a`.
**Subject**: the LAST item on the Step-3.7-Flash serving bill — PP-2 serving required
`MEMRA_SERVE_SPEC=0`, because the verify forward failed closed under a sharded cross-device
placement. The predecessor lane (`research/pp2-batch-20260806`) found and named that:

```
step error: decode_step_t (spec verify): refused with the ppN door open across 2+ devices
```

**Rig**: 2x RTX PRO 6000 Blackwell Server Edition (96 GB each), sm_120a, CUDA 13.2, driver
595.71.05. SPOT box shared with the step37-p2 lane; GPU windows under `flock /tmp/memra-gpu.lock`.

**Status**: IN PROGRESS — code landed and compiling, box battery pending.

## What changed in the engine

`decode_step_t_core_stream` is the SINGLE funnel every verify forward reaches — `decode_step_t`,
`_h`, `_h_emb`, `_h_emb_dev`, `_core` all land there, as do both hot-loop call sites (the
round-stream burst at spec.rs:4009 and the main verify at spec.rs:4365). So the whole spec surface
is wired at one point, and the draft/accept/commit machinery is untouched.

1. **`verify_layers(e, x, lo, hi, pos_d, t, cache, ckpt, stream)`** — the per-layer verify subgraph
   EXTRACTED (not duplicated) from the funnel's inline loop and made range-scoped, mirroring
   `decode_batch_layers(lo..hi)` and `decode_layers_eager(lo, hi)`. The unsplit body now calls it
   with `(0, n_layers)`. There is deliberately no "split version" of the verify math: the funnel's
   per-layer dispatch mirroring (norm fusion, the `t>=3 || (t==2 && spec_m2())` batched-linear
   window, the fused-q8 FFN chain, decode-exact projections) is exactly what makes verify
   bit-identical to eager decode, and a copy would be free to drift from it.
2. **`decode_step_t_core_ppn`** — the stage-split T=K+1 verify, structured as
   `decode_step_batch_ppn` / `decode_step_h_ppn`: per-stage Engine (the 2026-08-02 shared-scratch
   race structure preserved — the verify path allocates MORE of that scratch than eager decode,
   FA at m=T plus the per-layer `GdnStash` retains), per-stage `pos_d`, embed on stage 0,
   `output_norm` + head on the last stage, `[T, n_embd]` boundary payload through the existing
   persistent grow-only slots. One `VerifyCkpt` threads all stages.
   - In `stream` mode each stage runs its own `pos_iota` over the SHARED device pos counter. The
     counter is read-only for the duration of the forward (the round's `inc`/`copy_add` happen
     outside it), so every stage derives the identical iota while keeping its own output buffer
     stream-local — the M2 pipelining law, which a single shared `pos_d` freed at fn return
     would break.
3. **The refusal survives for the residue** — `MEMRA_SPEC_PP=0`, `MEMRA_PP_STREAMS=0`, or a
   placement whose `PpNRt` fails to build. A config that would still walk the whole trunk on one
   stream refuses instead of regressing 28x.
4. **`pp::spec_pp_on()` / `MEMRA_SPEC_PP=0`** — the A/B rollback seam, read per verify call and
   never memoized, so the bit-identity gate can compare split vs unsplit in ONE process against
   the same loaded weights.
5. **Stage-owned KV on the spec path** — three `Cache::new` sites now route through
   `pp::new_cache`: `new_session` (spec.rs:2719, THE serving spec-session path),
   `generate_spec_inner2`'s `own_cache` (3067), and `replay_acceptance` (5275). Primary-homed KV
   under an open cross-device door makes every remote stage peer-read its OWN KV each round; the
   same wrong-card class was already fixed on the two batched serving paths (worker.rs 2483 /
   2837). Door shut, `new_cache` IS `Cache::new`, so single-device allocation is byte-unchanged.
   Left unfixed this would also have charged the split for a harness bug in this lane's own perf
   receipt — the trap the predecessor lane documented.

## The gate: `decode-batch-gate --mode ppspec`

Same method as `--mode pp` (door open BEFORE load, because weight sharding is a load-time
decision; reference = door-shut walk over the SAME sharded weights, whose peer reads are slow but
byte-exact), different forward. Per round it checks:

- **ALL T logit columns**, bit-by-bit — not just the last. Greedy accept argmaxes every column, so
  a bug that only perturbs interior columns still changes the accept walk.
- **the `h_seed` hidden** ([n_embd], last column pre/post-norm per `MEMRA_SPEC_HPOST`) — the
  drafter is re-seeded from it every round, so a wrong h_seed degrades acceptance without ever
  changing a verify logit.
- **`cache.pos` parity** at every round (asserted): verify advances position by T, and a stage
  that advanced it twice would otherwise show up only as slow drift.

Two arms (`split` xreps for the flake class, and `unsplit@ppncache` as the localizer — same cache
placement, `MEMRA_SPEC_PP=0` varying ONLY the walk). Both placement orders are two invocations,
not two arms: the primary device follows `MEMRA_PP_DEVICES[0]` and the door opens before load.

`--ts 2,5,9` = T=K+1 for K=1,4,8 — the same K range `run-spec` walks, and T=9 crosses the
`t>=3` batched-linear window.

## Results

Pending — box battery not yet run.

## Why this lane pushes with `MEMRA_SKIP_PERF_CI=1`

Same two structural reasons as the predecessor lane, unchanged:

- the local 5090 was occupied at push time (the owner's resident `llama-server`, 332 MiB —
  memra is the owner's default engine and llama is the fallback, so both stay up); a perf battery
  contending for that GPU produces clock-invalid numbers, and a contended run satisfies neither
  half of the interleaved-in-one-hold law;
- more fundamentally, **the local 5090 is one card and this lane's subject is a two-card stage
  split.** There is no 5090 measurement of spec-over-PP-2 to be had. The target rig for every
  claim here is the PRO 6000 pair.

The 5090 default-flip gate still applies to anything changing a single-card runtime default.
Nothing here does: the pp door is off by default, and with it shut `spec_pp_on()` is never
consulted and `pp::new_cache` is `Cache::new`.
