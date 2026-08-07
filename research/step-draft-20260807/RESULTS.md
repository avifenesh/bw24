# lane/step-draft — the external MTP drafter on the SERVER path for step35

**Verdict: the attach needed no new spelling. The silence did.**

Step-3.7-Flash served through `memra-server` ran plain decode with no error, no warning, and no
log line — forgoing its entire felt-latency story invisibly. That is now impossible: a step35
model loaded without a drafter says so, a drafter path that cannot load refuses to start with the
loader's own error quoted, and the config that would boot into the #87 spec-over-PP2 bug refuses
before a CUDA context exists.

Commits: `97d0bd75`, `ef9a884b`, `d3585836`, `97edc284` on `lane/step-draft`.

**Gates: every arm green.** 13/13 ON BOX on the real 105 GB Step-3.7-Flash artifact over PP-2
(`raw/box-assert-20260807T000837Z.log`, FAILS=0), 11/11 local (`raw/loud-gates-5090-
20260807T001954Z.log`), 11/11 GPU-free preflight (`raw/preflight-gate-20260807T000622Z.log`),
`cargo test` 97 passed in `memra-server` + 77 in `memra-gguf`. **Two product bugs found by the
gates themselves** (§4), one of them a panic on the drafter path that made the refusal text
unreachable.

---

## 1. The attach convention: `+draft`, unchanged

The brief asked whether step35's external MTP head could ride the same `+draft` convention the
q9/q27 regime drafters use, "if possible, not a new one." It can, and it already did:

```
MEMRA_MODELS="step=/models/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf+/models/Step3.7-flash-mtp-Q8_0.gguf"
```

Nothing was added to the syntax. `+draft` already means "replace this model's MTP head with the
head in that file" (`docs/DRAFT-REGIME.md`), and `MtpHead::load_draft` already resolved step35's
per-layer draft geometry from the drafter file's own arrays — `Step35MtpGeom::resolve`, shipped by
the step37 lane in `d316162c`, which asserts `n_head` against both the `wq` out-features and the
`attn_gate` width and checks KV width against the trunk scalar.

### A correction to the brief's premise

The brief said "the server load path ignores `MEMRA_MTP_DRAFT`." It does not. The global env twin
is read inside `load_from_source_impl` (`crates/memra-engine/src/hybrid.rs:1277`), and the server
reaches it through `HybridModel::load` → `load_from_source` → `load_from_source_impl`. Both attach
spellings worked before this lane touched anything.

So the gap was never the plumbing. It was three things, all in the reporting:

1. a step35 model served **without** a drafter said nothing;
2. a drafter path that was wrong was only discovered **after** the trunk load, where on a busy
   card the operator instead read `CUDA_ERROR_OUT_OF_MEMORY` about the trunk;
3. the config that pairs a drafter with armed spec over sharded cross-device PP-2 — the #87
   quarantine regime — booted green and would have died on its second concurrent spec session.

Getting this right mattered for the shape of the fix: had the premise been taken at face value,
the lane would have added a second attach spelling to a path that already had one.

## 2. The loud-failure semantics as shipped

Four decisions, all from one pure function (`draft_verdict`, `crates/memra-server/src/worker.rs`)
so the message **text** is under test. The text is the entire detector for this defect class —
`kernel-check` is model-free, `run-gen` argmax MATCHes because plain decode is correct, and
`run-spec` is never reached because spec never engages. A warning nobody can act on is the same
defect as no warning, so the actionable attach spelling is asserted to be IN the line.

| situation | behavior |
| --- | --- |
| drafter attached (embedded or `+draft`) | quiet; `regime draft attached (<path>)` as before |
| **step35, no drafter** | `WARN`: names plain decode, explains `nextn=0` is expected for this arch, gives the exact `MEMRA_MODELS` attach string |
| non-step35, no drafter | quiet — a headless model genuinely has no head |
| `+draft` path missing / not a file | **FATAL at parse time**, before any GPU work |
| `+draft` path unloadable | **FATAL**, loader's error quoted (magic bytes and all), refuses rather than degrading |
| drafter + spec armed + sharded cross-device PP-2 | **FATAL before `Engine::new`**, cites #87, the receipts, the quoted `CUDA_ERROR_ILLEGAL_ADDRESS`, and the fix |

The no-noise half is a deliberate, tested constraint: `NoDrafterQuiet` exists so the warning does
not fire on every headless model the server has ever hosted, which is how a real warning gets
ignored.

### The #87 interaction, and why it refuses *early*

Step is a PP-2 model (105 GB against 96 GB cards), and PP-2 is exactly where spec is quarantined:
`dev01` is 20x slow and `dev10` provokes a `CUDA_ERROR_ILLEGAL_ADDRESS` that is **sticky for the
whole CUDA context** — measured `c=4 → 0/48` requests served, 100% reproducible 3/3
(`research/pp2-spec-20260806/RESULTS.md`, arm F4). Per the brief the quarantine holds; nothing here
enables spec over PP-2.

The first implementation refused at the load site, which was correct and ~20 minutes late: it
streamed 105 GB across two cards before announcing a verdict fixed at startup. A gate that takes
20 minutes to say "no" is a gate operators route around. `preflight_pp2_spec_refusal` now runs in
`worker::spawn` before the worker thread exists, because a `+draft` attach is a *promise* that a
drafter will be attached and both other terms are pure env reads (`serve_spec_enabled()`,
`pp_sharded_cross_device()`).

Both checks stay. The load-path verdict still owns the **embedded-head** case, where
`mtp.is_some()` is only knowable after the trunk is parsed, and the preflight explicitly falls
through when no `+draft` was given rather than waving that case past. Both emit the *same string
from the same function*, so the operator cannot get two differently-worded verdicts depending on
which check happened to fire.

**No collateral.** The refusal binds only where all three terms hold. `MEMRA_SERVE_SPEC=0` with a
drafter attached still boots and keeps the head loaded — that is what makes an operator's config
ready for when #87 lifts — and single-card configs are untouched, where spec is fully live. This
is tested in both directions (`the_quarantine_binds_only_where_all_three_conditions_hold`,
`the_87_refusal_lands_before_the_load_when_a_draft_was_attached`) and asserted against real booted
servers (arms I). A refusal one term too wide would take the whole 105 GB SKU offline, since PP-2
is the only placement it fits in at all.

## 3. Gate results

### `cargo test -p memra-server` — 97 passed, 0 failed

Five new tests. (`--lib` does not work here: `memra-server` is binary-only.)

### `run-preflight-gate.sh` — 11/11, GPU-FREE, on a fully contended card

`raw/preflight-gate-20260807T000622Z.log`. This gate needs no GPU and no artifact, which is the
point: the preflight decides before `Engine::new`, so the trunk path is never opened. It ran with
the 5090 at **22466/24463 MiB held by another lane** and was unaffected. A gate that only runs when
the box is idle is a gate that does not run.

- **arm H** — refuses; cites #87, the receipts, the quoted `CUDA_ERROR_ILLEGAL_ADDRESS`, and
  `MEMRA_SERVE_SPEC=0`; and **neither `Engine ready` nor `loading model` appears in the log.**
- **arms I** (`MEMRA_SERVE_SPEC=0`, `MEMRA_PP_STAGES=1`) — no refusal, and each is shown to have
  *reached the engine* rather than merely failed to print one. On this card they reach it and OOM
  on the trunk, which still proves it: a CUDA context at all means the config was passed through.

### `run-loud-gates.sh` — 11/11 on the 5090

`raw/loud-gates-5090-20260807T001954Z.log`. Arm A (`+draft` attaches, no spurious warning,
generation served), arm B (non-step35 stays quiet — the no-noise half), arm C (unloadable drafter
refuses, FATAL, rationale present, path quoted), arm D (missing drafter refuses at parse time,
cause names the DRAFTER not the trunk, refuses before any GPU work).

An earlier run on a card held at 22466/24463 MiB by another lane's `run-gen` recorded arms A-C as
SKIP. Both of §4's product bugs were found in the transition: arm D's on the contended card, arm
C's the moment the card freed and the arm could finally execute its real assertion.

### ON BOX — 13/13 on the real Step-3.7-Flash artifact, FAILS=0

`raw/box-armF-preflight-realartifact-20260807T000829Z.log`, 2x RTX PRO 6000 Blackwell Server
Edition, real 44 GB trunk shard + real 3.5 GB `Step3.7-flash-mtp-Q8_0.gguf`, `MEMRA_PP_STAGES=2
MEMRA_PP_DEVICES=0,1`:

```
[server] FATAL: worker init failed: step: REFUSING to start — a drafter is attached AND spec
serving is armed AND the ppN door is open across 2+ devices. Spec over a sharded cross-device PP
placement is QUARANTINED (#87, receipts research/pp2-spec-20260806) ...
```

Both cards sat at 78865 / 78353 MiB from another tenant throughout and were never touched. That
contention is the arm working, not a caveat: the whole claim is that this verdict costs no VRAM.

`run-box-assert.sh` then ran all three arms under `/tmp/memra-gpu.lock` —
`raw/box-assert-20260807T000837Z.log`, **13/13, FAILS=0**:

- **arm E** — step35 over PP-2 with NO drafter **WARNS**. THE arm that cannot run anywhere else:
  it needs a real `arch.is_step35()` off a real Step GGUF, and there is no Step artifact on the
  5090. `raw/box-armE-warn-20260807T000837Z.log`, on the real 45-layer trunk:

  ```
  [worker] WARN: step: step35: no MTP drafter attached — serving plain decode, no speculative
  decoding. This arch ships its MTP/NextN head in a SEPARATE GGUF, so the trunk's
  nextn_predict_layers=0 is expected and does NOT mean the model has no drafter. Attach with
  MEMRA_MODELS="step=/home/ubuntu/.../Step-3.7-flash-IQ4_XS-00001-of-00003.gguf+/path/to/
  Step3.7-flash-mtp-Q8_0.gguf" (the same '+draft' convention every regime drafter uses; ...)
  [worker]   loaded "step": 45 layers, eos=128007
  ```

  That is the silent defect, audible. Before this lane the same load printed only the second line.

- **arm F** — refuses, with every element asserted, and **before the 105 GB load** (`Engine ready`
  absent from the log).

- **arm G** — drafter attached + `MEMRA_SERVE_SPEC=0` **boots and serves** over PP-2: no spurious
  refusal, no spurious warning, and a real completion (`raw/box-armG-gen-20260807T000837Z.json`,
  32 tokens in 1.64 s, coherent). The quarantine is not collateral damage on an operator who
  attaches the head today to be ready for when #87 lifts.

State stamped in `~/STATE-stepdraft.md`.

### `run-spec` K=1..8 with the drafter — already PASS, deliberately not re-run

Per the brief's "check before re-running":
`research/step37-p2-20260806/raw/mtp-draft-PASS-20260806T215132Z.log` shows
`=== SELF-CONSISTENCY PASS ===` for K=1..8 with the step35 MTP drafter attached (acceptance 77.8%
at K=1 falling to 11.0% at K=8), with the per-layer geometry line
`[mtp-draft] step35 MTP geometry blk.45: n_head=96 n_head_kv=8 n_rot=128 rope_base=10000 swa=true
window=512`. The drafter itself gates green; nothing in this lane touched the draft math.

## 4. Two product bugs, both found by the gates

### (a) A nonexistent `+draft` path survived parse and failed only after the entire trunk load

On a shared card that load OOMs first, so the operator's FATAL read `CUDA_ERROR_OUT_OF_MEMORY`
about the **trunk** and never mentioned that the drafter path was wrong — a typo diagnosed as a
capacity problem. Reproduced against the pre-fix binary (`97d0bd75`, built in a temp worktree so
the receipt is real rather than remembered):
`raw/armD-PREFIX-missing-draft-survives-parse-20260807T001652Z.log` shows `Engine ready` and the
full `loading model` before the drafter is ever looked at. Fixed in `ef9a884b`:
`parse_models_config` now validates the drafter path the same way it always validated the model
path. Post-fix arm D is 3/3, including the assertion that the refusal precedes any GPU work.

### (b) A non-GGUF drafter file PANICKED, making the refusal text unreachable

Found by this lane's own arm C, on its first execution — the earlier attempt was on a contended
card and reported a trunk OOM without reaching the draft path. With the card free, arm C got a
verdict of `FAIL: C: refusal text missing the rationale`, and the log said why
(`raw/armC-refuse-baddraft-20260807T001704Z.log`):

```
thread 'memra-gpu-worker' panicked at crates/memra-gguf/src/lib.rs:238:5:
assertion `left == right` failed: bad GGUF magic: 0x73696874 in /tmp/step-draft-not-a-gguf-....gguf
[worker] PANIC in the GPU worker thread: ...
[worker] respawn attempt 1/1 in 2s (reloading weights)
[server] FATAL: worker init failed: worker died during init
```

`parse_one` is declared `io::Result` but `assert_eq!`'d on the magic and the version. A panic does
not cross a thread boundary usefully: the worker caught it, **burned a full respawn attempt
reloading every weight**, and told the operator `worker died during init` — while the carefully
worded drafter refusal, which names the offending path and says what to do, never ran, because
nothing ever came back as an `Err`. For a 105 GB PP-2 model that respawn is minutes of pointless
weight streaming to arrive at a worse message.

Both header checks now return `Err` with the observed bytes quoted alongside the expected value
(a wrong magic is usually a wrong *file*, and the magic is the fastest way to see what it actually
is). Post-fix, one line, no panic, no respawn:

```
[server] FATAL: worker init failed: draft m: bad GGUF magic: 0x73696874 (expected 0x46554747 =
"GGUF") in /tmp/step-draft-not-a-gguf-1834885.gguf — this file is not a GGUF (drafter path
"..." was requested via the MEMRA_MODELS '+draft' attach — refusing to start rather than
silently serving plain decode)
```

Regression test `a_non_gguf_file_is_an_error_not_a_panic` covers both the magic and the v2-version
arm. This bug was never step35-specific — it sat under **every** GGUF open in the engine, and it
took a gate that asserted on the *text* of a failure to surface it.

## 5. The gates' own bugs — four, three of them false-green

Recorded because every one of them is the exact shape this lane exists to remove: a check that
reports success without having checked.

1. **`ST=$(boot ...)` ran `boot` in a subshell**, so `$SPID` never reached the caller. No server
   was ever killed, and later arms health-checked the *previous* arm's still-running process.
2. **`kill "${SPID:-0}"`** — `kill 0` signals the entire **process group**, i.e. the script. It
   killed itself right after arm A, which is how bug 1 surfaced. An unset PID means "nothing to
   kill," never "kill everything."
3. **`$FAILS` incremented inside the `flock` subshell** could not reach the exit check outside it.
   The script would have exited **0 with failing arms**. Fixed with a temp file.
4. **Arm C "passed" its refusal assertions against a log whose only FATAL was
   `CUDA_ERROR_OUT_OF_MEMORY`** — a trunk load that never reached the draft path. Fixed with
   `not_incidental()`, and the verdict for an incidental OOM is now **SKIP** on arm C (nothing was
   tested) but **FAIL** on arm D. That asymmetry is the content: D asserts a *parse-time* refusal,
   so reaching a trunk GPU load at all means the check did not fire.

   This one paid for itself immediately. Had arm C stayed a false PASS, §4(b) — a panic on every
   GGUF open in the engine — would have shipped behind a green gate.

And one in the box script, caught before it ran: the first draft set `MEMRA_PP_STREAMS=0
MEMRA_PP_SHARD=0`, copied from the step37 lane's chunk-invariance scripts. **Both of those seams
bring every weight home to the primary, making `pp_sharded_cross_device()` false.** Arm F would
have passed its refusal assertions without ever entering the regime under test. Two flags that
read like "the placement that serves" silently invert the predicate the gate is built on.

## 6. What remains for spec-served Step

**Only #87.** The wiring is complete and gated; the block is entirely the quarantine.

- **Ready now:** `+draft` attaches the external step35 MTP head on the server path; the drafter
  gates green K=1..8 via the CLI; spec is fully live on any single-card-capable config; the silent
  class is closed on every path.
- **Blocked:** spec *served* for Step, because Step is 105 GB and PP-2 is the only placement it
  fits, and spec over sharded cross-device PP-2 is quarantined. `MEMRA_SERVE_SPEC=0` stands, and
  PP-2 serves plain decode at **875.1 tok/s at c=8, 96/96 clean** (arm F1) — the fastest arm in
  that lane, so plain-decode PP-2 serving is not a consolation.
- **When #87 lifts:** delete `DraftVerdict::RefuseSpecOverPp2` and its preflight. No other change
  — an operator who attached the head today is already configured, since the refusal keeps the
  drafter loaded rather than dropping it.

Ownership per that lane's own verdict is the **spec path**, not the stage split: arm F1 is the same
reversed placement with the same split and spec OFF, and it is 96/96 clean. The captured draft
graph is exonerated (F2 eager fails identically to F3).
