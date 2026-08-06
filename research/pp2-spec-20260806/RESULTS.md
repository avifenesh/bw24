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

**Status**: refusal LIFTED and the exactness battery is ALL GREEN — the correctness deliverable is
done. **But the feature is NOT shippable for concurrent serving**: spec over PP-2 is 20x slow in
one placement and, in the other, provokes a sticky `CUDA_ERROR_ILLEGAL_ADDRESS` that kills the
worker's CUDA context under concurrency (100% of requests lost at c=4). That is this lane's bug,
not the predecessor's — the same placement with spec OFF is 96/96 clean and the fastest arm
measured here. See §A deterministic illegal address. Recommendation: keep `MEMRA_SERVE_SPEC=0`
for PP-2 serving until it is fixed.

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

N=5 interleaved rep-major with the arm order alternating per rep, greedy, steady thermals
(pre/post `nvidia-smi` in `logs/perf/gpu-pre.csv` / `gpu-post.csv` — 32->34 C dev0, 27->32 C dev1,
idle clocks both ends, so no thermal drift to account for), one server per arm per rep,
`memra-server` + `tools/load-serve.py` per the servepath-p2 harness law. Medians of 5; the
per-rep spread is remarkably tight (arm B is 17.1 in all five reps at both concurrencies, arm A
344.6-347.7), so these medians are not hiding variance:

| arm | config | c=1 agg tok/s | c=1 p50 (s) | c=8 agg tok/s | c=8 p50 (s) | errors |
| --- | --- | --- | --- | --- | --- | --- |
| A | door shut, spec ON (one card) | 346.5 | 0.373 | 345.2 | 2.826 | 0/180 |
| C | pp2 dev01, spec OFF | 223.7 | 0.572 | **872.9** | 1.171 | 0/180 |
| D | pp2 dev10, spec ON | 123.3 | 1.046 | 119.2 | 8.265 | **5/180** |
| B | pp2 dev01, spec ON | 17.1 | 7.536 | 17.1 | 56.839 | 0/180 |

Two things fall out, and the second was not what this lane went looking for:

1. **Spec over the split is not worth it on a model that fits one card.** 2.8x slower than
   one-card spec in the good placement, 20x in the bad one, and slower than the split with spec
   OFF in both. The exactness work is done and the door is open, but the felt-latency story does
   not pay here. (Arm D's 5/180 errors are the illegal-address bug below — and its c=8 rate
   badly understates it; at c=4 the same config loses 100% of requests.)
2. **Spec does not scale with concurrency at all, and the split without spec does.** Arm C goes
   223.7 -> 872.9 (3.9x) from c=1 to c=8, while arms A, B and D are FLAT (346.5->345.2,
   17.1->17.1, 123.3->119.2). Flat aggregate throughput under 8x the offered load means the spec
   path is serializing sessions rather than batching them. That is a serving-architecture finding
   about spec generally, not about PP — arm A is the door-shut single-card arm and it is just as
   flat — and it is the reason arm C, the predecessor's shipped spec-OFF config, is the fastest
   c=8 configuration measured anywhere in this lane by 2.5x over one-card spec. Worth its own
   lane; not fixable inside a PP change.

The B1FAST check comes out clean in the sense that matters: acceptance is bit-identical to the
door-shut arm at every K, so a solo spec session does not fall off the draft chain or lose
acceptance when the door opens. The c=1 loss is data movement, not degraded speculation.

### A deterministic illegal address on the reversed placement — OWNED, and worse than it looked

Arm D failed **exactly 1 of 32 requests at c=8 in all five reps** with:

```
step error: DriverError(CUDA_ERROR_ILLEGAL_ADDRESS, "an illegal memory access was encountered")
```

Deterministic 1/32 x 5/5 is a bug, not a flake. `run-ppspec-illegal.sh` (N=3, `logs/illegal/`)
settles ownership and finds the c=8 receipt was measuring the mild end of it:

| arm | config | c | ok | err | agg tok/s |
| --- | --- | --- | --- | --- | --- |
| F1 | dev10, spec **OFF** | 8 | 96/96 | **0** | 875.1 |
| F2 | dev10, spec ON, `SPEC_NOGRAPH=1` | 8 | 93/96 | 3 | 119.1 |
| F3 | dev10, spec ON (arm D in-hold) | 8 | 93/96 | 3 | 119.1 |
| F4 | dev10, spec ON | 2 | 9/24 | **15** | 84.1 |
| F4 | dev10, spec ON | 4 | 0/48 | **48** | **0.0** |

**Verdict on ownership: it is the SPEC path, not the predecessor's split.** F1 — the same reversed
placement, same stage split, spec OFF — is 96/96 clean and the fastest arm in the lane (875.1).
So the reversed placement itself is sound; adding spec breaks it. That makes it this lane's to
fix, and it is NOT attributable to `pp2-batch`.

**The draft graph is exonerated**: F2 (`MEMRA_SPEC_NOGRAPH=1`, eager draft) fails identically to
F3 (3/96 both, 119.1 both). The captured graph was the best structural suspect — it bakes launch
args across replays and disables the context's event tracking for the whole session
(`spec.rs:2937-2942`) — and it is not the cause.

**The failure rate is inverted in concurrency, which is the real finding.** c=8 loses 1 in 32,
but c=4 loses **everything** (0/48, wall 0.008s) and c=2 loses 15/24. A c=4 run finishing in 8 ms
with 48 errors is not a slow failure — the CUDA context is already dead when the requests arrive.
The server log shows the sequence, and the cause is quoted rather than inferred:

```
[worker] spec pool evicted (2) after alloc failure; retrying (evict-first learned)
[worker] spec session alloc failed (DriverError(CUDA_ERROR_ILLEGAL_ADDRESS, "an illegal memory
         access was encountered")); tokenwise path
```

then that second line repeats for **every subsequent request in the process** — 20 of them in
r1-F4 — including the whole c=4 phase that ran after the c=2 phase in the same server. So:
concurrent spec sessions on the reversed placement provoke an illegal access, and once it fires,
`CUDA_ERROR_ILLEGAL_ADDRESS` is **sticky for the CUDA context** — every later `new_session` inherits
it, the worker charitably reports "tokenwise path", and the process serves nothing but 400s until
restarted. The c=8 arm's benign-looking 1/32 was luck of ordering: it happened late in the run.

That also retro-explains why the step-4 receipt looked survivable and why c=1 is clean everywhere:
the trigger needs two live spec sessions. The eviction line points at the spec session pool
(`worker.rs:2733-2745`) as where it surfaces, but the first bad dereference is upstream of the
allocator's error report, so the mechanism is still **unlocalized to a kernel or buffer** — repro
in hand (`c=4, MEMRA_PP_DEVICES=1,0`, spec ON, 100% reproducible in 3/3 reps), no conclusion built
past the quotes.

**Serving consequence, stated plainly: PP-2 + spec is NOT safe to enable for concurrent serving
in either placement.** `dev01` is 20x slow, `dev10` is fast but dies under concurrency and takes
the process with it. The exactness work stands; the feature is not shippable until this is fixed.
The predecessor's spec-OFF split (F1/arm C, 872-875 tok/s at c=8) remains the configuration that
actually serves.

### Draft placement: the answer

**Placement matters enormously — 7x between the two orders (B 17.1 vs D 123.3), N=5, every rep
within 0.2 tok/s** — while bit-identity and acceptance are IDENTICAL in both. So the answer to the
brief's question is: yes, and it is the single largest factor in spec-over-PP2 cost. `dev10`
(drafter co-resident with the serving primary, which is also the last stage) is the configuration
to use; `dev01` is 7x worse. That much is a measured, tight, reproducible fact.

The MECHANISM is only partly established, and the rest of this section is the honest account of
that. A candidate cause was constructed from the code and the artifact:

1. `memra-server` always builds `Engine::new(0)` (`worker.rs:1079`): the serving primary is
   **always device 0**, whatever `MEMRA_PP_DEVICES` says. `decode-batch-gate` instead follows
   `MEMRA_PP_DEVICES[0]` (`decode_batch_gate.rs:121-127`) — which is why the gate battery never
   saw this: in the gate, stage 0 IS the primary in both orders. (This asymmetry between the
   serving harness and the gate harness is worth knowing on its own.)
2. `output_norm` + the lm head upload through the LAST stage's engine (`hybrid.rs:881`) —
   correct for the trunk, since the last stage is what runs the head.
3. Every MTP/draft weight uploads through the plain primary engine (`hybrid.rs` ~1021,
   `load_t(e, ...)`) and every draft forward runs on `e` (`spec.rs:4276`, `5434`).
4. q9 ships **no** `blk.32.nextn.shared_head.weight` — the GGUF header carries only
   `eh_proj` / `enorm` / `hnorm` / `shared_head_norm` — so the draft head falls back to the
   TRUNK head at `spec.rs:781`, `mtp.shared_head_head.as_ref().unwrap_or(&self.output)`.

All four are true. The inference drawn from them — that with `devices=0,1` the draft GEMV on dev0
peer-reads a `[4096, 248320]` NVFP4 head (~508 MB) from dev1 every draft step, matching the
measured ~59 vs ~8 ms/token — was plausible and is NOT what dominates. It got an isolation arm
instead of a writeup, and the arm knocked it down:

`run-ppspec-drafthead.sh`, N=3 interleaved, c=1, all arms 12/12 OK, zero errors:

| arm | config | agg tok/s | p50 (s) | all reps |
| --- | --- | --- | --- | --- |
| E3 | dev10 sharded, spec ON (arm D in-hold) | 123.3 | 1.049 | 123.4, 123.2, 123.3 |
| E1 | dev01 + `SHARD=0`, spec ON | 23.4 | 5.510 | 23.4, 23.4, 23.4 |
| E2 | dev01 + `SHARD=0`, spec OFF | 7.5 | 17.128 | 7.5, 7.5, 7.5 |

E1 confirms the placement (`weight home: dev0 (MEMRA_PP_SHARD=0 bring-up placement)`) and brings
the lm head home to the primary — and recovers almost nothing: 17.1 -> 23.4, still 5.3x short of
E3's 123.3.

**The honest reading: the remote-lm-head story is NOT the dominant term, and this arm does not
cleanly refute it either — it is confounded.** E2 is why. With `SHARD=0` and spec OFF the same
placement collapses to **7.5 tok/s, 30x below arm C's 223.7** at the identical stage split, because
`SHARD=0` puts every weight on dev0 while stage 1's kernels run on dev1, so stage 1 peer-reads its
whole layer range every token. That is a known-slow bring-up mode and E2's collapse is expected —
but it means E1 pays that same whole-trunk peer-read tax while removing only the head tax. E1's
17.1 -> 23.4 is therefore a FLOOR on the head effect, not a measurement of it, and the confounder
runs in the direction that hides what the arm was built to see. I designed the arm believing
`SHARD=0` isolated the head cleanly; E2 proves it does not, which is what the control was there
to catch.

What IS established: removing the head tax entirely still leaves 5.3x on the table, so the head
cannot be the whole story even at its most favourable reading. The 508 MB-per-draft-step
arithmetic was consistent with the data and is not sufficient to explain it.

What that leaves as the live explanation for B vs D is the primary-device asymmetry itself
(`memra-server` always builds `Engine::new(0)`, so with `devices=1,0` stage 1 IS the primary and
the draft, the embed, and the head all sit together on it, whereas with `0,1` the draft runs on
dev0 while stage 1's work — including the head — is remote). Deciding that needs an arm that moves
the DRAFT rather than the weights, i.e. a code change to run the draft on the head's engine, which
is a trunk-loader placement decision with its own gate obligations. Named here, not smuggled in
behind a spec change. The B/D asymmetry and its 7x remain a measured, reproducible fact regardless
of which term dominates.

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
