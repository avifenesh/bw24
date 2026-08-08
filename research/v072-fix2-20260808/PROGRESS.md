# v0.72 tag-blocker 2 — spec+PP-2 serving collapse (112.5 -> 17.5 agg tok/s)

Lane: lane/v072-blocker2, base a131e8c7. Perf-only regression, correctness intact
(battery #87 crash gate 212/212). Evidence base: lane/v072-battery,
research/v072-prep-20260808/ (on that branch, not this worktree).

## Known facts (from the pair-box battery)

- run-spec over PP-2: FAST (K=1 164.7 tok/s) — engine path fine
- spec-OFF PP-2 serve: FAST (223.2) — serving+PP-2 fine without spec
- door-shut single-card spec serve: FAST (547.3) — serving+spec fine without PP-2
- placement-independent (dev10 == dev01)
- ONLY serving-layer spec over PP-2 is slow. => serving-layer spec-round x PP interaction.

## Prime suspect

5f27c55c "fix(server): follow PP primary device" (cx-503b round 2):
`worker_device()` now returns the PP primary (first device in CUDA_VISIBLE_DEVICES
order); worker boot pins device=1 on dev10 placements. Hypotheses:

- H1: drafter/MTP head loads on a different device than the verify trunk's stage-0,
  so every spec round pays a cross-device hop for draft logits.
- H2: the worker thread's current-device context makes the spec round's host syncs
  cross-device (peer sync / context switch per round).

Related: leverb lane found the same merge's residency sizing is a ~3% pp regression
(sigmoid-router archs) — ab564179. Two regressions, one merge. Fix must keep the
correctness win (multi-tenant device follow): surgical repair, not revert.

## Plan

1. [x] PROGRESS.md committed (this file).
2. [x] Read 5f27c55c diff + worker.rs drafter attach/load path + spec round loop
       device context. Identify where the drafter loads relative to worker_device().
3. [x] Repro on box2 (q9 embedded MTP over PP-2). Collapse at BASE: 17.4/17.5.
4. [x] Device experiment: BASE vs FIX A/B, one lock hold — 17.4 -> 111.7/112.0/111.9.
5. [x] Fix committed (05ddfef2): worker primary follows the PP HEAD stage (the lm
       head's device), keeping the device-follow correctness win.
6. [x] Verify: spec+PP-2 c=1/c=2 112 class N=3, spec-off 221.7 unchanged, single-card
       543.5 unchanged, run-spec 8/8 PASS pinned acceptance, crash gate c=4 x50 clean.
7. [x] Receipts (raw logs + points jsonl + driver log) committed in raw/.

## Log

- 2026-08-08: lane start. Plan committed before any code reading (write-first).
- 2026-08-08: STATIC DIAGNOSIS (code + existing receipts, pre-box):

### The mechanism

1. `crates/memra-engine/src/hybrid.rs:1213` — `e_head = layer_engine(e, n_trunk,
   n_trunk-1)`: the lm head (`output` + `output_norm`) uploads through the LAST
   stage's engine, i.e. lives on the last stage's device under a sharded PP placement.
2. `crates/memra-engine/src/spec.rs` `mtp_head_forward_dev` op 12:
   `let head = mtp.shared_head_head.as_ref().unwrap_or(&self.output)` — qwen35-family
   drafters (q9 embedded MTP included) ship no own head, so EVERY draft token's head
   matmul reads the TRUNK lm head (the biggest tensor in the model). Op 11's fallback
   `shared_head_norm.unwrap_or(&self.output_norm)` reads last-stage bytes too.
3. The draft chain (and its graph capture) runs on the PRIMARY engine. Therefore:
   spec serving is fast iff primary device == LAST-stage device (head co-located);
   primary == stage-0 device puts a full lm-head peer read on every draft token.

### Why every existing receipt fits

| receipt | topology (primary vs head) | speed |
|---|---|---|
| lane binary serve, dev10 (`Engine::new(0)`, placement 1,0) | primary=0 == last stage | 112.5 FAST |
| lane-era note, dev01 (primary=0 == stage 0, head on dev1) | mismatch | ~20x SLOW (pp2spec PROGRESS "known non-blockers") |
| HEAD serve (5f27c55c: primary=PP_DEVICES[0] = stage 0 ALWAYS), dev10 AND dev01 | mismatch always | 17.5 both — the battery's "placement-independent" |
| run-spec engine E1, dev10 (`Engine::new(0)`) | primary=0 == last stage | 164.7 FAST |
| spec-OFF serve (no draft chain; head matmul runs ON the last stage via `el.matmul_decode_exact`) | insensitive | 223.2 == lane 223.3 |
| door-shut single card | no PP | 547.3 unchanged |

The merge flipped serving on dev10 from the validated fast topology (primary on the
head stage) to the slow one (primary on stage 0) — and made the slow topology universal.

### Fix shape (surgical, keeps the correctness win)

`worker_device()` follows the LAST device in MEMRA_PP_DEVICES (the head stage's
device), not the first. This:
- restores EXACTLY the device topology the 212/212 crash battery + 112.5 receipts
  validated on dev10 (primary=dev0, stage0 own-engine dev1, stage1 own-engine dev0);
- keeps 5f27c55c's win — the worker primary is a placement device, never an
  unrelated device 0 (the multi-tenant device-follow), invalid strings still refuse
  at boot, boot line still logs the device;
- should ALSO fix the old dev01 ~20x (primary lands on dev1 = head stage there).
Engine gate binaries (ppn-gate, decode-batch-*) keep primary=PP_DEVICES[0] — they
deliberately test the shared-engine stage-0 case and are not the serving surface.

### Box plan (box2 first)

- Repro at HEAD: q9 (embedded MTP) spec+PP-2 dev10 c=1 -> expect collapse class.
- Experiment: patched worker (primary=last) -> expect 112-class return, N=3.
- Controls: spec-off PP-2 unchanged; door-shut single-card unchanged; dev01 spec
  (expect fast NOW — differentiates fix from a plain revert); run-spec 8/8;
  #87 quick crash gate c=4 x50 clean.

## Box2 verification (driver box2-fix2-verify.sh, q9 @ /data/models, tree a131e8c7
## + spot-guard checkpoint aa2895b2 [engine-only seam edits, no worker.rs delta])

Interim receipts (points-*.jsonl, first 4 arms, single lock hold):

| arm | binary | placement | c | agg tok/s | prediction |
|---|---|---|---|---|---|
| base-dev10-spec-c1 | 5f27c55c worker (stage-0 primary) | 1,0 | 1 | **17.4** | collapse class 17.5 — REPRODUCED, digit-match |
| base-dev10-spec-c2 SPEC_GATE=0 | same | 1,0 | 2 | **17.5** | crash-gate shape that read 112.5 on the lane binary — REPRODUCED |
| base-dev10-specoff-c1 | same | 1,0 | 1 | **221.7** | ~223 control — spec-off unaffected, CONFIRMED |
| fix-dev10-spec-c1-r1 | HEAD-stage primary fix | 1,0 | 1 | **111.7** | 112 class RETURNED (r1 of N=3) |

Boot lines quoted: BASE dev10 "Engine ready (device=1, ...)" (stage-0 pin),
FIX dev10 "Engine ready (device=0, ...)" (head stage).

### FINAL verify table (driver rc=0, one lock hold, raw/ in this directory)

All rows: q9 embedded-MTP, greedy, max_tokens 96, warmup 1. FIX = the head-stage
worker_device patch (identical bytes to commit 05ddfef2's worker.rs); BASE = tree as
found (5f27c55c stage-0 pin). Single runs labeled single runs; headline cells N=3.

| arm | binary | placement | c | agg tok/s | verdict |
|---|---|---|---|---|---|
| base-dev10-spec-c1 | BASE | 1,0 | 1 | 17.4 | collapse REPRODUCED (battery E2: 17.5) |
| base-dev10-spec-c2 (GATE=0) | BASE | 1,0 | 2 | 17.5 | crash-gate shape collapse REPRODUCED (lane receipt: 112.5) |
| base-dev10-specoff-c1 | BASE | 1,0 | 1 | 221.7 | control clean (lane: 223.3) |
| fix-dev10-spec-c1 r1/r2/r3 | FIX | 1,0 | 1 | 111.7 / 112.0 / 111.9 | **112 class RETURNED, N=3, spread 0.3** |
| fix-dev10-spec-c2 (GATE=0) r1/r2/r3 | FIX | 1,0 | 2 | 111.8 / 111.9 / 111.9 | **matches the lane's 112.3 c=2 receipt, N=3** |
| fix-dev10-specoff-c1 | FIX | 1,0 | 1 | 221.7 | spec-off UNCHANGED (digit-match to BASE) |
| fix-doorshut-c4 (single card) | FIX | none | 4 | 543.5 | single-card unchanged (battery: 547.3; single run) |
| fix-dev01-spec-c1 | FIX | 0,1 | 1 | 111.0 | **THEORY DIFFERENTIATOR: the pre-merge ~20x-slow placement is now FAST** — a plain revert would NOT do this (single run) |
| fix-dev10-crash-c4 x50 (GATE=0) | FIX | 1,0 | 4 | 50/50 ok, 0 err, agg 111.6 | #87 quick crash gate CLEAN; fault-line grep count 0 |
| run-spec K=1..8 dev10 PP-2 (engine) | shared bin | 1,0 | — | K=1 161.5 tok/s | **SELF-CONSISTENCY PASS 8/8**, acceptance 27/36, 33/62, 36/84... — IDENTICAL to the pinned door-shut table |

Boot-line receipt across all arms: every BASE PP arm booted device=1 (stage 0 of 1,0);
every FIX dev10 arm booted device=0 (head stage); fix-dev01 booted device=1 (head stage
of 0,1); door-shut booted device=0 (no placement -> default). Exactly the topology rule.

### Root cause, one paragraph (CONFIRMED by the A/B)

The sharded PP loader homes `output_norm` + the lm head on the LAST stage's device
(hybrid.rs:1213). The serving spec round runs its draft chain on the PRIMARY engine, and
`mtp_head_forward_dev` op 12 falls back to `&self.output` for qwen35-family drafters —
one full lm-head read per draft token, plus op 11's `output_norm` fallback and the
round's UVA readbacks of last-stage verify buffers. 5f27c55c pinned the worker primary
to MEMRA_PP_DEVICES[0] = stage 0, making the head remote from the draft chain on every
placement order (the battery's "placement-independent" 17.5). The lane's fast receipts
were all topologies where primary == last-stage device (`Engine::new(0)` under 1,0). The
fix makes the worker primary follow the HEAD stage (last device): 17.4 -> 111.9 (N=3,
c=1 and c=2), spec-off/door-shut digit-unchanged, crash gate 50/50 clean, run-spec 8/8
PASS at pinned acceptance, and dev01 — slow since before the merge for the same reason,
recorded as a "~20x placement-scheduling question" in the pp2spec lane — is fixed to
111.0 by the same rule, which a revert to `Engine::new(0)` could not have done. The
correctness win of 5f27c55c is preserved: the primary is always a placement device
(multi-tenant device-follow), invalid strings still refuse at boot (now validated at
every position, not just [0]).

### GPU-free suites (local)

- `cargo test -p memra-server worker_device` — 2 passed (new semantics pinned: 1,0 -> 0,
  0,1 -> 1, every position validated).
- `cargo test -p memra-server -- --test-threads=1` — 115 passed, 0 failed.
- `cargo test -p memra-engine --lib` — 48 passed, 1 GPU-only ignored.
- PARALLEL-RUN FLAKE, PRE-EXISTING, quoted: the default (parallel) suite intermittently
  fails ONE health test (`liveness_failure_obeys_the_retry_contract` or
  `a_wedged_gpu_flips_health...`) with `left: 200, right: 503`. Counted x10 both arms:
  HEAD worker.rs 9/10 pass, a131e8c7 worker.rs 8/10 pass — present at BASE, unchanged by
  this fix, and it is the SAME transient the cx-503b lane recorded in its own
  verification ("One earlier parallel full-suite invocation failed the unrelated
  a_wedged_gpu... assertion... the serial 115-test suite passed"). Serial is 115/115
  deterministic. Not this lane's regression; flagged, not fixed here.

### Not addressed here (out of scope, already owned elsewhere)

- The leverb lane's ~3% pp residency regression from the SAME merge (dead slabs starve
  the SLRU) — separate mechanism (238beae0, engine residency sizing), receipts and the
  slab-arm recovery live in lane/pp-leverb (ab564179).
- Spec-ON vs spec-OFF on PP-2 (112 vs 222 at c=1) — the pp2spec lane's flagged
  placement-aware spec-gate tuning question, pre-existing, unchanged by this fix.
