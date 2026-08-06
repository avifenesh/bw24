# docs-fit sweep — 2026-08-07 (lane/docs-fit)

Base: `origin/restructure/public-split` @ `9971e7f8`. Scope: make README + docs describe the
product as it now is, after v0.71.0 and the merged lane run (pp2-batch, pp2-spec, pp2-hardening,
serve-hardening, spec-scaling, accept-gate, step37-p2, chunkinv-flip, f8f4-flip, q8-argmax,
ptx-audit). No code touched, nothing built, no `PERF-*` marker block hand-edited;
`tools/update-perf-board.py --check` verified green before every commit and after the last one.

Method note, because it changed the outcome: the assignment's change list was **verified against
the tree rather than trusted**. That found one claimed item that does not exist, one that is not
on this base, and — after cross-checking an audit against the step37 receipts — **one factual
error I had myself written into an earlier commit in this lane**. Each is recorded below rather
than quietly corrected.

## Fixed (13)

| # | Surface | Drift | Fix | Commit |
|---|---|---|---|---|
| 1 | `README.md` | "Multi-GPU boxes serve as a replica fleet: **1,477 tok/s** managed on 3xH100" was the *only* multi-GPU serving shape described. PP-2 had shipped as a real serving path. | Both shapes described; PP-2 carries its `MEMRA_SERVE_SPEC=0` constraint inline, so nobody reads it as spec-capable. | `67e79016` |
| 2 | `README.md` | Exactness contract asserted chunk-invariance without qualification. `step35` has a receipted chunk-dependence defect. | Scoped to a **per-architecture** property, with the shipped arches still gated so, and a pointer to Known gaps. | `67e79016` |
| 3 | `README.md` | "One GPU per engine process — no tensor parallelism yet (pipeline-parallel seam merged, default off)" — describes a seam, not the shipped serving path. | Requirements/limits bullet rewritten: PP-2 real and gated for plain batched serving, opt-in via `MEMRA_PP_STAGES`/`MEMRA_PP_DEVICES`, spec off. | `67e79016` |
| 4 | `README.md` | "Use something else when ... you need tensor-parallel serving" conflated two different asks — TP throughput (go elsewhere) and >1-card capacity (supported now). | Carved apart in the Why-memra bullet and the limits bullet. | `67e79016` |
| 5 | `README.md` | Bring-up list and Known gaps predate step35 and the PP-2 spec verdict; Loaders bullet predates multi-shard GGUF. | Step-3.7-Flash added to bring-up; two Known-gaps entries added (step35 chunk-dependence with its closed form, spec-over-PP-2 not shippable); multi-shard GGUF + a PP-2 "What's inside" bullet. | `67e79016` |
| 6 | `docs/SERVING.md` | Opening asserted "memra's engine owns one GPU per process ... Multi-GPU serving is therefore a **replica fleet**", with TP as the only other shape. Structurally wrong post-PP-2. | Two shapes presented (replica fleet = throughput, PP-2 = capacity), TP as neither. New `## Pipeline-parallel (PP-2) serving` section: 7-config bit-identity battery, cost table, the −14.9% B=1 regression and its fix, the spec rationale, serve-smoke 0-failed, and the four fail-closed paths with the 28x/13.9x cliff. | `8c7d45fc` |
| 7 | `docs/SERVING.md` | Chunked-prefill section claimed invariance universally; fleet-tooling table had no systemd row though `deploy/systemd/memra-server.service` ships. | Added a "Scope: this is a per-architecture property" paragraph with the step35 closed form and receipts; added the systemd row. | `8c7d45fc` |
| 8 | `docs/TESTING.md` | **Zero** mentions of `ppn`/`pp`/`ppspec`. A whole merged gate family was undocumented. | New `## Multi-GPU (PP-N) exactness gates` section with exact invocations taken from each binary's own arg parsing, plus the three load-bearing properties: `--reps` defaults to 2 because the class was a 35% flake; the door must open BEFORE load because sharding is load-time; the two localizer arms. | `506faf8f` |
| 9 | `docs/TESTING.md` | `chunkinv` entry implied general coverage. | Bounded: coverage is per-architecture and prompt-length-bounded — the pinned probe prompts are short, which is exactly why they could not reach the step35 defect. | `506faf8f` |
| 10 | `docs/PERFORMANCE.md` | Rigs table had no 2x PRO 6000 row though PP-2 cells are measured there; Bring-up notes had no step35 entry. | Added both. Rig row labeled rented per the Rigs doctrine. | `974433e5` |
| 11 | `docs/PERFORMANCE.md` | **My own error in `974433e5`**: I wrote that step35 "boots **resident** over PP-2". The boot log says `101.07 GB experts + 3.92 GB trunk vs 100.88 GB free -> SLRU cache`. | Corrected to state the SKU is a **spill** path even on 2x96 GB, with the measured cache health (89.0% steady-state hit, 133.5 MB/decode-token vs a 2678 MB/token Stage-1 baseline = 20.1x less PCIe) and the caveat that the residency decision is **PP-blind in its numerator** — it sums every layer's expert bytes including the other card's and compares against one stage's free VRAM. Not fixed in code: residency selection is perf-affecting and belongs behind an A/B. | `5afe2da2` |
| 12 | `docs/PERFORMANCE.md` | Three lanes of PP-2 numbers had no home in a doc where every number belongs to exactly one listed rig. | New `### Pipeline-parallel (PP-2) — the capacity shape` subsection: batched split cost 0.995x/0.989x/0.986x at B=4/8/16, B=1 0.982x with its rollback control, transport 0.986–0.997x of seam, placement symmetry within 0.3%, B=8 = 3.65x B=1, spec-OFF c=8 dev10 875.1 tok/s 96/96 0 err, P2P 13.6–14.0x a host bounce. Three caveats so the cells cannot be misquoted (see Caveats below). | `5afe2da2` |
| 13 | `docs/FLAGS.md` | `MEMRA_BATCH_PP` and `MEMRA_SPEC_PP` — both **default ON** — were entirely absent. A default-ON seam missing from the catalog is a flags-doctrine violation. `MEMRA_SERVE_SPEC` did not mention the PP-2 requirement; `MEMRA_PRIME_CHUNK` claimed unqualified invariance. | Both rows added from `crates/memra-engine/src/pp.rs:283,294`; `MEMRA_SERVE_SPEC` gained the PP-2 requirement; `MEMRA_PRIME_CHUNK` scoped with the step35 closed form. | `f7364462` |

### FLAGS.md second pass (5 more, `8be6fbbd`)

Found by diffing `env::var("MEMRA_*")` read sites against the catalog rather than by reading prose.

- **`MEMRA_MOE_GDEC_GATE` is a phantom.** Listed as a live byte-identity oracle; **zero read
  sites** in the tree. The only trace is the comment at `hybrid_forward.rs:2510` describing what
  it *would* compare. Moved to the graveyard. Nothing is left uncovered — the identity is
  asserted by construction (slot-ordered `__fmaf_rn` chain == sequential `axpy_f32` chain) and
  `MEMRA_MOE_GDEC=0` is the live rollback arm — but a documented gate nobody can run reads as
  coverage, which is worse than no gate.
- **`MEMRA_PP_STAGES` "batch/dc/graph/spec warn-once" was wrong twice.** Batched and spec verify
  now take their own stage split (both default ON). The paths that *don't* split never warned:
  `warn_unwired_once` has exactly two call sites and both are gemma4-specific. What actually
  protects `decode_step_dc` and its graph wrapper is `refuse_unsplit_if_remote` failing closed.
  Replaced with a per-path coverage list. (`pp.rs`'s own header already carried this correction —
  the doc had not caught up.)
- **`MEMRA_PP_ALLOW_UNSPLIT_BATCH`: "wiring a real stage split is the open weeks-class item"** —
  true the day the door landed, superseded the same day by `MEMRA_BATCH_PP`/`MEMRA_SPEC_PP`.
- **`MEMRA_SERVE_B1FAST`: "Skipped for ... ppN cuts"** — no longer true, and the reason matters.
  Skipping the lever under an open pp door cost **−15.0% at B=1** (208.5 vs 177.3), provably not
  as a split cost since stages=2 on one card paid the same 177. The split path now applies it per
  stage. Documented alongside why the pp bit-identity gate pins `set_b1_fast(false)` — with it on,
  the B=1 reference and the split arm sit on opposite sides of the accepted 1.591e-1 decode-config
  FP gap and the arm reports a *fake* stage-split failure.
- **Two user-facing gaps added**: `MEMRA_MODELS_DIR` (the one knob for putting weights on another
  filesystem — undocumented despite being the download/lookup root) and `MEMRA_SPEC_TEMP` (the
  spec path's own sampling temperature, distinct from `MEMRA_TEMP`, gated on seeded
  *reproducibility* rather than token-identity because Leviathan/Chen guarantees distribution
  equality only).
- **Header now states coverage honestly**: ~421 `env::var` read sites vs ~380 listed, with the
  residue named by class (per-kernel A/B forcers, dump/trace probes, bench-bin inputs, `build.rs`
  nvcc tunables, the whole `MEMRA_DFLASH_*` block). Categories (a)–(c) — runtime params, machine
  config, rollback seams — are complete, which is the part a naked run or a rollback depends on.
  Explicitly: silence here is not evidence a seam is absent. That assumption is what kept the
  phantom gate listed.

### Cross-doc scoping (2 more, `9fcd96d1`)

- **`ARCHITECTURE-H100.md`** round 49 banked the M0 comms floor as "PP ~free, EP<=4, graphed a2a
  mandatory". Measured: "~free" holds at **N=2 serial only** (185.39 vs 185.83 baseline, inside
  the band, confirming M0's 0.3–0.5%/tick prediction). N=4 is 0.90x, N=8 is 0.89x — ~10% is the
  honest N>2 serial cost. And free *only* with per-stage placement: the `MEMRA_PP_SHARD=0`
  peer-read arms are a 3–4x cliff (55.5/42.8/38.7 tok/s at N=2/4/8). **Appended as round 57, not
  edited** — that ledger is append-only, so the round-49 line stands as written and the new entry
  is its scope. The 2026-08-06 PRO 6000 numbers are noted as cross-rig context and explicitly not
  promoted to H100 cells.
- **`docs/HY3-SPILL.md`** said "the PP-2 spike wires spec K=1 in and measures the verify-batch
  overhead phi" — future tense, and PP-2 shipped *without* spec. Verify does take its own stage
  split and is bit-identical (ppspec 7/7 green), but spec-over-PP-2 is not shippable under
  concurrency, so phi remains unpriced on a pair. Marked the resident-bank `S_est` as an estimate
  on this shape, not a measured PP-2 result. The acceptance profile itself needed no caveat —
  measured single-GPU.

## Deferred, with reason (4)

1. **`lane/spec-gate` / concurrency-gated spec scheduler — NOT ON THIS BASE.** The brief flagged
   it as "may merge while you work — check the train tip". Checked after a fresh fetch: tip is
   still `9971e7f8`, and `git branch -r --contains faba56cf` returns nothing. Related: the brief's
   `MEMRA_SPEC_GATE` **does not exist anywhere in the repo** — zero hits in `crates/`, `tools/`,
   or `research/`. Documenting a flag by name from a brief, without a read site, is exactly the
   failure mode that left `MEMRA_MOE_GDEC_GATE` listed for a month. The refutation that motivated
   the lane *is* on this base (`fe2b3740`, spec serializes at `worker.rs:1686`, obvious fix
   refuted by the m=16 exact-kernel ceiling); when the scheduler lands, its flag and its
   concurrency threshold need a FLAGS.md row and a SERVING.md paragraph.
2. **The brief's crash id "#87" has no in-repo trace.** Zero hits in any `.md`. Every reference
   in the docs I wrote cites `research/pp2-spec-20260806/` and commit `5882b753` instead. If #87
   is a tracker id, the docs should carry the link — owner call on whether to add it.
3. **step35 is deliberately absent from the generated supported-models table.** It has not
   cleared the deployment bar (best-vs-best e2e ≥1.1x on every prompt class). It gets an honest
   bring-up entry in README and PERFORMANCE.md instead. This follows the deployment-bar doctrine
   and needs no board regeneration — no published number moved, which is also why every
   `PERF-*` block is byte-unchanged.
4. **~90 category-(d)/(e) instrumentation vars stay uncatalogued**, now named by class in the
   FLAGS.md header rather than silently omitted. Cataloguing each per-kernel A/B forcer and dump
   probe would roughly double the file to document things that cannot change a default or a
   rollback. If the owner wants literal completeness, that is a separate mechanical pass.

## Owner calls

- **The residency check is PP-blind in its numerator.** It sums every layer's expert bytes —
  including layers resident on the *other* card — and compares against one stage's free VRAM. On
  the Step SKU that decides spill vs resident at `101.07 + 3.92 vs 100.88 GB`, i.e. a coin-flip
  margin, and it would wrongly spill a bank that fits per-stage on a wider split. Documented as
  a caveat; **not fixed**, because residency selection is perf-affecting and needs an A/B, not a
  docs commit. This is the highest-value item the sweep surfaced.
- **`Engine::new(0)` is unconditional in the serving worker**, regardless of `MEMRA_PP_DEVICES`.
  The serving primary is therefore always device 0. That asymmetry is the root of the
  dev01-vs-dev10 split, and dev10 is the placement that goes fatal at c=4 with spec on.
  Documented; a fix is a code lane.
- **Whether "#87" should appear in the docs** (see Deferred 2).

## Caveats deliberately written into the docs

So they cannot be lost by later summarizing:

- The **1.786x/1.905x pipelined figures are not serving throughput** — they come from a bench
  loop replaying a pre-recorded token stream with tokens in flight. Plain autoregressive serving
  cannot do that, because token N+1's input is token N's output. The pipelined arm is also still
  quarantined (same-device refused outright after a reproduced 35% co-located-stream race;
  cross-device record ~69/70 with an OPEN root cause) even though the same-device flake was
  refuted on the PRO 6000 silicon (20/20, p<0.001).
- **PP-2 is the capacity shape, never a scaling win.** The replica fleet is the throughput answer.
- **`MEMRA_SERVE_SPEC=0` is required for PP-2 serving**, with the reason stated every place the
  constraint appears.
