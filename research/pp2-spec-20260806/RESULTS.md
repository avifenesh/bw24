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

**Status**: refusal LIFTED, exactness battery ALL GREEN, cost receipt measured. The remaining
open item is a placement bug this lane FOUND but did not create (see §Draft-head placement).

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

Raw logs: `logs/gates/`, `logs/serve/`, `logs/perf/`, plus the localization series `logs/probe/`
(p1-p8) and `logs/postfix/` (v1-v5). Runner scripts are `run-ppspec-gates.sh`,
`run-ppspec-perf.sh`, `run-ppspec-drafthead.sh`. Box commit stamped in `BOX-COMMIT.txt`.

### The refusal is LIFTED

`decode_step_t` no longer refuses under a sharded cross-device placement: the verify trunk takes
its own stage split. Serve-smoke with spec ON over PP-2 passes and the **HTTP 400 is gone**
(`logs/serve/smoke-ppspec.log`); the `[spec-acc]` liveness lines in every spec-ON server log prove
spec actually ran rather than silently degrading to plain decode. The refusal still bites on the
residue it should: `MEMRA_SPEC_PP=0` under the same placement dies with the quoted
`refused with the ppN door open`, so the rollback seam cannot silently cost 28x.

### Bit-identity: 7/7 arms ALL GREEN

`decode-batch-gate --mode ppspec`, every arm `0 failing arm(s)`. Every logit column at T=2/5/9,
16 rounds x 3 reps (2 for the wider arms), plus the `h_seed` ledger and `cache.pos` parity —
**0 differing bits** throughout.

| arm | stages | fence | devices |
| --- | --- | --- | --- |
| q9 dev01 | 2 | [0, 16, 32] | 0,1 |
| q9 dev10 | 2 | [0, 16, 32] | 1,0 |
| q9 singledev | 2 | [0, 16, 32] | default(primary) |
| q9 split5 (uneven cut) | 2 | [0, 5, 32] | 0,1 |
| q9 PP-4 | 4 | [0, 8, 16, 24, 32] | 0,0,1,1 |
| q9 dev01 HPOST | 2 | [0, 16, 32] | 0,1 |
| q27 dev01 | 2 | [0, 32, 64] | 0,1 |

### Acceptance: IDENTICAL split vs door-shut, both placements

`run-spec` K=1..8, `SELF-CONSISTENCY PASS` on all 7 invocations, and the sharper check — the
acceptance counts are equal token-for-token across split and door-shut:

- q9 (`dev01` = `dev10` = `doorshut-dc0` = `doorshut`):
  `27/36  33/62  36/84  36/112  36/140  36/168  36/196  36/224`
- q27 (`dev01` = `doorshut-dc0` = `doorshut`):
  `23/24  31/36  36/42  35/48  40/60  39/66  43/77  37/80`

The `doorshut` vs `doorshut-dc0` pair also retires the assumption that `MEMRA_QWEN_DC=0` changes
the oracle: identical counts, so the `DC0` seam the split arms need is measurement-neutral.

### Standing battery (door shut — the split must not move single-device behavior)

`kernel-check` `ALL GREEN: kernels match CPU reference.`; `decode-batch-gate` config + strict
(`MEMRA_MMVQ=0 MEMRA_NO_FUSE_NORMQ=1`) both `ALL GREEN`; `run-gen` argmax `MATCH`; and the
predecessor lane's own `--mode pp` gate still `0 failing arm(s)` on this tree, so extracting
`verify_layers` out of the shared file did not move the batched split.

### Two bugs found and fixed

**1. Stream publication at the ppN body exit** (this lane's own bug, introduced and fixed here).
The first ppspec battery passed `dev01` and `dev10` but FAILED `singledev` and `dev00`, with
nondeterministic garbage (NaN, `3155.677`, `2.87e-5` against a reference `-2.0048926`) AND the
unsplit localizer arm failing too. Localized by probes p1-p8 (`logs/probe/`): `--mode pp`
singledev PASS (p3), `MEMRA_PP_STREAMS=0` PASS (p4), `dev01` PASS (p6), both-arms-unsplit PASS
(p7). Cause: unlike both predecessor ppN bodies, which return HOST values computed inside the
last-stage scope, this body returns **device slices**; the stage-stream guard dropped at return
and the caller then read those buffers on the primary stream with nothing ordering the two.
Cross-device placements were immune because the caller's first touch there is a driver-ordered
cross-device copy. Fixed with `PpNRt::publish_to(s, dst)` — record an event on the stage stream,
wait it on the caller's stream, no-op when they are the same stream. The caller's stream is
captured BEFORE any `rt.enter()`. Verified V1-V5 (`logs/postfix/`) all exit 0.

**2. Zero-emit burst underflowed the session-tail commit** (PRE-EXISTING, from `b4aea184`, not
this lane's split). At c=8 with spec ON, 31 of 32 requests returned HTTP 500 "worker closed
stream" on **every** spec arm — including the door-shut single-card denominator, which is what
identified it as unrelated to PP:

```
thread 'memra-gpu-worker' (50489) panicked at crates/memra-engine/src/spec.rs:5218:49:
range end index 18446744073709551615 out of range for slice of length 0
```

A burst that emits nothing leaves `out` empty while the tail unconditionally committed
`&out[..out.len() - 1]`, so `0 - 1` wrapped. Fixed with `out.len().saturating_sub(1)`.
Post-fix c=8 verification: **32/32 OK, 0 errors, 0 panic lines, 344.3 agg tok/s** (was
`n_ok 1 n_err 31 agg_tok_s 51.9`). All pre-fix c=8 spec numbers are therefore INVALID and were
re-measured on top of the fix; the pre-fix receipts are preserved on the box at
`~/receipts/pp2spec/perf-PREFIX-INVALID` rather than deleted. The c=1 numbers were unaffected
(0 errors on every arm).

### Cost: spec over the split is EXPENSIVE, and asymmetric by placement

c=1, N=5 interleaved rep-major with the arm order alternating per rep, greedy, steady thermals
(pre/post `nvidia-smi` in `logs/perf/gpu-pre.csv` / `gpu-post.csv`), one server per arm per rep,
`memra-server` + `tools/load-serve.py` per the servepath-p2 harness law:

| arm | config | agg tok/s | lat_p50 (s) |
| --- | --- | --- | --- |
| A | door shut, spec ON (one card) | 346.5 | 0.373 |
| C | pp2 dev01, spec OFF (predecessor's shipped config) | 223.4 | 0.573 |
| D | pp2 dev10, spec ON | 123.4 | 1.045 |
| B | pp2 dev01, spec ON | 17.2 | 7.515 |

So: spec over PP-2 is **2.8x slower than spec on one card in the good placement and 20x slower in
the bad one** — and in both placements it is slower than simply running the split with spec OFF.
The exactness work is done and the door is open, but on a model that FITS one card there is no
reason to prefer it. That is the honest verdict for the felt-latency story, and it is a
capacity-only feature until the placement bug below is fixed.

The B1FAST check comes out clean in the sense that matters: acceptance is bit-identical to the
door-shut arm at every K, so a solo spec session does not fall off the draft chain or lose
acceptance when the door opens. The loss is entirely data movement, not degraded speculation.

### Draft-head placement: the answer, and a bug this lane FOUND

Placement matters enormously — 7x between the two orders (B 17.2 vs D 123.4) — while
bit-identity and acceptance are IDENTICAL in both. So it is pure data movement, and there is a
named cause, checked against the artifact and the code rather than inferred:

1. `memra-server` always builds `Engine::new(0)` (`worker.rs:1079`): the serving primary is
   **always device 0**, whatever `MEMRA_PP_DEVICES` says. `decode-batch-gate` instead follows
   `MEMRA_PP_DEVICES[0]` (`decode_batch_gate.rs:121-127`) — which is exactly why the gate
   battery never saw this: in the gate, stage 0 IS the primary in both orders.
2. `output_norm` + the lm head upload through the LAST stage's engine (`hybrid.rs:881`) —
   correct for the trunk, since the last stage is what runs the head.
3. Every MTP/draft weight uploads through the plain primary engine (`hybrid.rs` ~1021,
   `load_t(e, ...)`) and every draft forward runs on `e` (`spec.rs:4276`, `5434`).
4. q9 ships **no** `blk.32.nextn.shared_head.weight` — the GGUF header carries only
   `eh_proj` / `enorm` / `hnorm` / `shared_head_norm` — so the draft head falls back to the
   TRUNK head at `spec.rs:781`, `mtp.shared_head_head.as_ref().unwrap_or(&self.output)`.

Therefore with `devices=0,1` the last stage is dev1, so `output` (`[4096, 248320]` NVFP4, ~508 MB)
lives on dev1 while the draft GEMV runs on dev0 — half a gigabyte of peer-read weight traffic
**per draft step**. With `devices=1,0` the last stage is dev0 = the primary, head and draft are
co-resident, and the traffic disappears. Arm C (spec OFF) is unaffected because the trunk runs the
head on the stage that owns it. The measured per-token times (B ~59 ms, D ~8 ms) are the right
order for that transfer.

`run-ppspec-drafthead.sh` decides this with no code change: `MEMRA_PP_SHARD=0` brings every weight
home to the primary while the stage streams still split, so E1 (`dev01` + `SHARD=0` + spec ON)
recovering arm D's speed confirms the mechanism, and E1 staying at ~17 refutes it and indicts the
split itself. E2 is the draft-free control; E3 re-measures arm D inside the same hold so E1 has an
interleaved denominator.

Note the shape of the real fix, which is NOT in this lane's diff: the draft should run on the
engine that owns the head it uses (or the head it uses should be resident where the draft runs).
Both are one-line placement changes in load/dispatch, but they are trunk-loader decisions with
their own gate obligations, so they are named here rather than smuggled in behind a spec change.

### What remains unsplit

`decode_step_dc` — `generate()`'s **default** Qwen route — plus the graph capture that wraps it.
That is the one remaining hole, and it is what forced `MEMRA_QWEN_DC=0` onto the split `run-spec`
arms: the refusal fired in the oracle before spec ran at all. Quoted:

```
decode_step_dc: refused with the ppN door open across 2+ devices — this path has no pp stage split
```

It fails closed, so it is safe, not silent. Serving does not hit it (the serving spec session goes
through the verify trunk, now split), but `generate()` does.

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
