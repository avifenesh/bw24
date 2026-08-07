# docs-fit sweep — 2026-08-07 (lane/docs-fit)

Base: `origin/restructure/public-split` @ `9971e7f8`, **rebased mid-lane onto `d873240e`** when
`lane/spec-gate` merged (see Deferred 1, now resolved). Scope: make README + docs describe the
product as it now is, after v0.71.0 and the merged lane run (pp2-batch, pp2-spec, pp2-hardening,
serve-hardening, spec-scaling, accept-gate, step37-p2, chunkinv-flip, f8f4-flip, q8-argmax,
ptx-audit). No code touched, nothing built, no `PERF-*` marker block hand-edited;
`tools/update-perf-board.py --check` verified green before every commit and after the last one.

Method note, because it changed the outcome: the assignment's change list was **verified against
the tree rather than trusted**. That found one claimed item that does not exist, one that is not
on this base, and — after cross-checking an audit against the step37 receipts — **one factual
error I had myself written into an earlier commit in this lane**. Each is recorded below rather
than quietly corrected.

Four passes, because the first one read prose and the later ones read the artifact. Pass 1 worked
from the assignment's change list. Pass 2 read every gate wrapper's argument parsing and refusal
logic instead of its description. Pass 3 read the serve code paths behind each documented contract.
Pass 4 was forced by the base moving under the lane. Passes 2-4 found 19 further drifts, including
both inverted statements and one of the two over-claimed contracts — a doc audit that only reads
docs cannot find a sentence whose negation is the code.

Totals: **37 drifts fixed** (13 first-pass + 5 FLAGS.md + 2 cross-doc + 6 TESTING.md + 10 SERVING.md
including the code's own stale module header + 3 spec-gate), **3 still deferred with reason** (a
fourth resolved when its lane merged mid-sweep), **6 owner calls**.

Two of the fixes correct statements that were **inverted** rather than merely stale (accept-gate's
`--force`, the SSE error shape) and two more are **over-claimed contracts** — chunk-invariance and
concurrent-load isolation, both real properties asserted more broadly than their gates establish.
Those four are the ones that would have cost a reader real time; the rest are staleness.

## Fixed (13)

| # | Surface | Drift | Fix | Commit |
|---|---|---|---|---|
| 1 | `README.md` | "Multi-GPU boxes serve as a replica fleet: **1,477 tok/s** managed on 3xH100" was the *only* multi-GPU serving shape described. PP-2 had shipped as a real serving path. | Both shapes described; PP-2 carries its `MEMRA_SERVE_SPEC=0` constraint inline, so nobody reads it as spec-capable. | `4dc33ab1` |
| 2 | `README.md` | Exactness contract asserted chunk-invariance without qualification. `step35` has a receipted chunk-dependence defect. | Scoped to a **per-architecture** property, with the shipped arches still gated so, and a pointer to Known gaps. | `4dc33ab1` |
| 3 | `README.md` | "One GPU per engine process — no tensor parallelism yet (pipeline-parallel seam merged, default off)" — describes a seam, not the shipped serving path. | Requirements/limits bullet rewritten: PP-2 real and gated for plain batched serving, opt-in via `MEMRA_PP_STAGES`/`MEMRA_PP_DEVICES`, spec off. | `4dc33ab1` |
| 4 | `README.md` | "Use something else when ... you need tensor-parallel serving" conflated two different asks — TP throughput (go elsewhere) and >1-card capacity (supported now). | Carved apart in the Why-memra bullet and the limits bullet. | `4dc33ab1` |
| 5 | `README.md` | Bring-up list and Known gaps predate step35 and the PP-2 spec verdict; Loaders bullet predates multi-shard GGUF. | Step-3.7-Flash added to bring-up; two Known-gaps entries added (step35 chunk-dependence with its closed form, spec-over-PP-2 not shippable); multi-shard GGUF + a PP-2 "What's inside" bullet. | `4dc33ab1` |
| 6 | `docs/SERVING.md` | Opening asserted "memra's engine owns one GPU per process ... Multi-GPU serving is therefore a **replica fleet**", with TP as the only other shape. Structurally wrong post-PP-2. | Two shapes presented (replica fleet = throughput, PP-2 = capacity), TP as neither. New `## Pipeline-parallel (PP-2) serving` section: 7-config bit-identity battery, cost table, the −14.9% B=1 regression and its fix, the spec rationale, serve-smoke 0-failed, and the four fail-closed paths with the 28x/13.9x cliff. | `1e12ad37` |
| 7 | `docs/SERVING.md` | Chunked-prefill section claimed invariance universally; fleet-tooling table had no systemd row though `deploy/systemd/memra-server.service` ships. | Added a "Scope: this is a per-architecture property" paragraph with the step35 closed form and receipts; added the systemd row. | `1e12ad37` |
| 8 | `docs/TESTING.md` | **Zero** mentions of `ppn`/`pp`/`ppspec`. A whole merged gate family was undocumented. | New `## Multi-GPU (PP-N) exactness gates` section with exact invocations taken from each binary's own arg parsing, plus the three load-bearing properties: `--reps` defaults to 2 because the class was a 35% flake; the door must open BEFORE load because sharding is load-time; the two localizer arms. | `f8542a73` |
| 9 | `docs/TESTING.md` | `chunkinv` entry implied general coverage. | Bounded: coverage is per-architecture and prompt-length-bounded — the pinned probe prompts are short, which is exactly why they could not reach the step35 defect. | `f8542a73` |
| 10 | `docs/PERFORMANCE.md` | Rigs table had no 2x PRO 6000 row though PP-2 cells are measured there; Bring-up notes had no step35 entry. | Added both. Rig row labeled rented per the Rigs doctrine. | `313ea5d5` |
| 11 | `docs/PERFORMANCE.md` | **My own error in `313ea5d5`**: I wrote that step35 "boots **resident** over PP-2". The boot log says `101.07 GB experts + 3.92 GB trunk vs 100.88 GB free -> SLRU cache`. | Corrected to state the SKU is a **spill** path even on 2x96 GB, with the measured cache health (89.0% steady-state hit, 133.5 MB/decode-token vs a 2678 MB/token Stage-1 baseline = 20.1x less PCIe) and the caveat that the residency decision is **PP-blind in its numerator** — it sums every layer's expert bytes including the other card's and compares against one stage's free VRAM. Not fixed in code: residency selection is perf-affecting and belongs behind an A/B. | `78b35a86` |
| 12 | `docs/PERFORMANCE.md` | Three lanes of PP-2 numbers had no home in a doc where every number belongs to exactly one listed rig. | New `### Pipeline-parallel (PP-2) — the capacity shape` subsection: batched split cost 0.995x/0.989x/0.986x at B=4/8/16, B=1 0.982x with its rollback control, transport 0.986–0.997x of seam, placement symmetry within 0.3%, B=8 = 3.65x B=1, spec-OFF c=8 dev10 875.1 tok/s 96/96 0 err, P2P 13.6–14.0x a host bounce. Three caveats so the cells cannot be misquoted (see Caveats below). | `78b35a86` |
| 13 | `docs/FLAGS.md` | `MEMRA_BATCH_PP` and `MEMRA_SPEC_PP` — both **default ON** — were entirely absent. A default-ON seam missing from the catalog is a flags-doctrine violation. `MEMRA_SERVE_SPEC` did not mention the PP-2 requirement; `MEMRA_PRIME_CHUNK` claimed unqualified invariance. | Both rows added from `crates/memra-engine/src/pp.rs:283,294`; `MEMRA_SERVE_SPEC` gained the PP-2 requirement; `MEMRA_PRIME_CHUNK` scoped with the step35 closed form. | `a4e32e8b` |

### FLAGS.md second pass (5 more, `71f0dff2`)

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

### Cross-doc scoping (2 more, `4a9ae9a5`)

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

### Second audit pass — TESTING.md against the gate scripts (6 more, `59137f1e`)

Method: read every gate wrapper's argument parsing and refusal logic instead of its README prose.

- **A sentence had accept-gate's central law exactly inverted.** TESTING.md said the gate "refuses
  on a dirty `crates/` with no `--force`". `accept-gate.sh:120` says, verbatim,
  `There is deliberately no --force here.` The refusal is unconditional and that is the entire
  point of the gate — a `--force` door would let the accept battery certify a tree that is not the
  tree. Fixed, and the second `--pin` guard documented too: it also refuses if
  `MEMRA_MMQ_F8F4`, `MEMRA_MMQ_F8F4_PLAIN`, `MEMRA_MMQ_FP8BLK_PLAIN`, `MEMRA_FAST`, or
  `MEMRA_PRIME_F32CHUNK0` is set in the environment, because a pinned arm inherited from the shell
  is a FALSE GREEN with no visible cause.
- **`--tier 2` did not run what three documents said it ran.** `local-ci.sh` does **not** run the
  `run-spec` K=1..8 sweep, and `--tier 2` does not run perf — the `--perf` mention was removed
  from the tier row and the spec claim corrected to what the script does (one gemma-gate
  stream-agreement check). See the owner call below: this is a doctrine-vs-reality gap, not a
  wording nit.
- **`--probes` was undocumented as the only non-FALSE-GREEN path for a clean tree** and the only
  way to reach the `amargin`, `amarginc`, `e4b`, and `kat` probes at all.
- **`amargin`'s advertised `--window 24` is not what runs**: the wrapper passes 12. Documented as
  the effective value with the discrepancy named.
- **Missing gates added**: `pp2-gate` to the PP-N table, plus `serve-st-gate`, `apikeys-gate`, and
  `serve-stress-gate`; validate-h100's graph lane named explicitly; the chunk-invariance gate's
  flags listed.

### Third audit pass — SERVING.md against the serve code (10 more, `e9d530e4` + `1507490f`)

- **The compatibility contract described `MEMRA_COMPAT`-gated behavior as unconditional.** The
  sentence "mid-stream worker errors arrive as a final `data:` error chunk + `[DONE]`, never a
  named SSE event" holds only on the OpenAI-shape surface: both the terminator and the error shape
  are gated `if chat || openai_compat()` (`main.rs:1966, 2007`), and `openai_compat()`
  (`main.rs:624-632`) is true only for `MEMRA_COMPAT=openai` or unset-plus-`MEMRA_API_KEY`. On a
  native-default server a streaming `/v1/completions` emits a named `event: error` and
  `event: done` with **no `data: [DONE]`** — the exact opposite, and a silent hang for an SDK
  waiting on the sentinel. Added as a precondition naming the gate, both shapes, and the fact that
  the shipped unit sets `MEMRA_COMPAT=openai` so a *deployed* server does match the section.
  Highest-severity find of this pass: the failure mode is a client that hangs rather than errors.
- **A fourth 503 shape was undocumented and sits outside the retry contract** (`main.rs:1741`,
  `1819`): `cmd_tx.send()` failing yields `"worker unavailable"` / `server_error` with
  `code: null`, no `Retry-After`, no `retry-after-ms`, no `x-should-retry` (`error_response` passes
  `code: None`; the header attaches only on `is_client_error()`). Precisely the transient condition
  where retry is correct. Added as a taxonomy row + paragraph, documented **as-is** rather than as
  fixed — see the owner call.
- **The respawn paragraph gave neither the backoff nor the two exits.** Backoff is `2 * attempt`
  seconds (`worker.rs:3917`) = 2 s at the default max of 1, and **two** paths reach exit 70 with
  different `sd_notify` STATUS lines: budget exhausted (`worker unrecoverable; exiting`,
  `worker.rs:3910`) vs a respawn whose weight reload failed (`respawn load failed; exiting`,
  `worker.rs:3886`) — the second is not a panic and exits rather than looping. An operator reading
  `systemctl status` needs the distinction. Exit 70 named as sysexits `EX_SOFTWARE` against the
  exit-1 startup FATALs.
- **The GPU-fault probe had no interval.** `MEMRA_GPU_WATCH_S` default 60 (`health.rs:322-325`),
  and the code's own comment calls "checks every 60 s" a *published detection commitment* — so it
  is documented as a stated fact about the instrumentation, not a free knob.
- **The health body was listed as its `worker.*` fields only.** Real shape is
  `{status, models, worker:{...}}` plus top-level `detail` on a red (`health_payload`,
  `main.rs:1235-1253`), and `status` has different vocabularies per route:
  `ok`/`draining`/`unhealthy` on `/health`-`/livez`, `ready`/`not_ready` on `/readyz`.
- **No route table existed** — routes appeared only as prose made them relevant, so `GET /models`
  was invisible. All nine added (`main.rs:1093-1101`) with the bind default, and `/models` marked
  as **not** a `/v1/models` alias: it is the plain `{"data":[{"id":"<alias>"}]}` listing with no
  `context_length`/`architecture`/`pricing`, and it is what `serve-smoke` asserts.
- **The systemd unit's couplings were doc-invisible.** The unit's own comments are good, but a
  reader who copies and edits it had no doc-side statement of which values are tied to server-side
  defaults. Three break *silently*, i.e. only during a failure: `WatchdogSec=180` must exceed
  `MEMRA_HEALTH_STALL_S=120` (else systemd restarts a healthy server mid-prefill),
  `TimeoutStopSec=60` must exceed `MEMRA_DRAIN_S=30` (else a correct drain is SIGKILLed),
  `TimeoutStartSec=600` must exceed the ~120 s cold load. Added as a table, with
  `StartLimitIntervalSec/Burst=3600/4` and *why* systemd's 10 s/5 defaults **cannot trip here**
  (5 starts do not fit in 10 s at ~120 s each → infinite restart loop instead of a failed unit),
  the `RestartSec`/`RestartSteps`/`RestartMaxDelaySec` ramp + its systemd ≥ 254 requirement, and
  `OOMPolicy=kill` with its reason (default `stop` can leave a worker-less process that accepts
  connections and serves nothing). Plus `Type=notify`'s `READY=1` meaning, the CAP_SYSLOG /
  `kernel.dmesg_restrict` Xid-visibility door, and why the unit is deliberately not
  `ProtectSystem=strict`.
- **"the decode-batch gate battery (gate1-3, gate3c lean-vs-full)" conflated modes with gates.**
  `validate-h100.sh` runs `decode-batch-gate` **twice** (`--mode config --batch 8`; `--mode strict
  --batch 4` under `MEMRA_MMVQ=0 MEMRA_NO_FUSE_NORMQ=1`), each running gate1/gate2/gate3 — and
  `gate3c` is gate3's third **sub-check**, not a fourth gate. Gate3 prints ONE PASS/FAIL line
  covering (a) device-argmax == host-argmax, (b) sampled B=N == B=1, (c) lean-vs-full, so the
  sub-check names surface only on failure and a green line is the only evidence (c) ran. Also
  stated that `--mode pp`/`ppspec` SKIP gate1/2/3 by design (`decode_batch_gate.rs:154-171` returns
  before them) and that neither PP mode is wired into `validate-h100.sh`.
- **Rate-limit headers are documented `X-RateLimit-*` but emitted lowercase**
  (`main.rs:270-272`). Harmless per RFC 9110, but a client parsing into a case-sensitive dict must
  key on `x-ratelimit-*`. Noted inline rather than restyling.
- **The code's own module header described the pre-serve-hardening server** (`main.rs:8-13`): four
  endpoints of nine, no `/livez`, no `/readyz`, no `/yield/metrics`, and a `/health` body with no
  `worker` object. Rewritten to the real set, pointing at `router()` as the authority. The only
  non-`.md` edit in this lane, and comment-only.

### Fourth pass — the base moved: lane/spec-gate coverage (3 more, `f329e951`)

Base advanced `9971e7f8` → `d873240e` mid-lane. Rebased; the merge's own docs commit
(`446c5203`) covered FLAGS.md and SERVING.md, leaving three surfaces:

- **README described spec and batched serving as two listed features**, which is how the
  superseded guidance read ("run spec and bulk as separate server processes"). Rewritten as one
  gated process — spec admitted at low concurrency, live sessions demoted into the batched phase
  as load arrives, byte-exact for greedy — stated as the reader-facing property (the server
  tracks whichever tier wins at the current concurrency instead of picking at deploy time) with
  thresholds left in FLAGS.md.
- **README's concurrent-load isolation contract was unconditional** — "byte-identical whether it
  arrives alone or inside a full batch" — while SERVING.md had just scoped it to *equal depth*.
  Exactly the same class as the chunk-invariance drift found in pass 1: a real contract,
  over-claimed. Scoped with the receipt (768-token greedy vs staggered-depth batchmates diverges
  at byte 1347 on one run, 2379 on another — the byte *moving* is the proof the configuration is
  nondeterministic), named as pre-existing engine behavior with its mechanism (one `split_keys`
  across mixed-depth sessions in `fa_decode_batch_seqs_v4` + B-dependent tier selection), and
  added as a Known gap so the exposure is listed rather than merely qualified.
- **TESTING.md had no serve-path exactness section at all**, so `exactness.py` (5 arms, one server
  boot each) was invisible and its not-in-any-battery status unstated. Added with the two
  load-bearing facts: the pinned-shape method (`MEMRA_SPEC_DEMOTE_AT` forces the transition at
  B=1 with no load, because under load neither arrival timing nor batch composition is
  deterministic — generalized into a rule for any property under test inside a nondeterministic
  config) and the three arms that each once produced a false green.

## Deferred, with reason (4 → 3 live)

1. ~~**`lane/spec-gate` — NOT ON THIS BASE.**~~ **RESOLVED: it merged mid-lane and is now
   covered.** The brief flagged it as "may merge while you work — check the train tip", and it
   did: base moved `9971e7f8` → `d873240e` (`Merge lane/spec-gate: concurrency-gated spec ships
   as DEFAULT`). Rebased onto it; one FLAGS.md conflict, resolved by keeping the base's two new
   `MEMRA_SPEC_GATE*` rows and this lane's `MEMRA_SERVE_SPEC` row with its PP-2 addendum — both
   changes are needed and neither supersedes the other. The merge brought its own FLAGS.md and
   SERVING.md coverage, so what remained were the surfaces it did not reach, fixed in `f329e951`:
   README described "speculative serving" as a listed feature (reading as two deployment choices,
   which was literally the superseded guidance) rather than as one gated process; README's
   isolation contract was **unconditional** where SERVING.md had just scoped it to equal depth;
   and TESTING.md had no serve-path exactness section, leaving the mode-switch harness invisible.
   Note the sequencing lesson: had this lane trusted the brief and documented `MEMRA_SPEC_GATE`
   from its name, the row would have been wrong in detail (the flag ships with `_LOW`/`_HIGH`
   thresholds, a hysteresis band, and one-way demotion — none of which a name implies) and would
   have collided with the real one. Waiting for a read site was correct.
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
- **`local-ci.sh` does not run the `run-spec` K=1..8 sweep that three documents name as a standing
  merge gate.** CLAUDE.md, CONTRIBUTING.md, and TESTING.md all name `run-spec` K=1..8
  self-consistency as one of the three gates; `local-ci.sh --tier 2` runs a single gemma-gate
  stream-agreement check instead. The docs now say what the script does, so nobody reads a green
  battery as K=1..8 evidence — but the consequence has to be stated plainly: **"the battery ran" is
  not evidence K=1..8 passed; quote the `run-spec` log.** Closing the gap is a tools change (wire
  the sweep into tier 2, or rename the tier so the doctrine gate is visibly separate).
- **Four registered fast-gate probes are dispatched by no `map.tsv` row, including DEFAULT**:
  `amargin`, `amarginc`, `e4b`, `kat`. They are reachable only via an explicit `--probes` list, so
  a default `fast-gate` run silently skips them. Documented; wiring them is a tools change.
- **Give the bare 503 a `code` and a retry pair.** The `cmd_tx.send()` failure path is the one
  transient 503 with nothing for a client to branch on. `code: "overloaded"` + `Retry-After: 5` +
  `retry-after-ms: 5000` would put it inside the contract the rest of the taxonomy already keeps.
  Server change, not a docs change — documented as-is rather than described as fixed.

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
