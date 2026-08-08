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
2. [ ] Read 5f27c55c diff + worker.rs drafter attach/load path + spec round loop
       device context. Identify where the drafter loads relative to worker_device().
3. [ ] Repro on box2 (fastest: 9B+drafter over PP-2, memra at ~/memra @ a131e8c7,
       models at /data/models). Confirm collapse at HEAD, N=3.
4. [ ] Device experiment: force old device-0 behavior (env/patch) -> confirm ~112
       returns. Quote the receipt.
5. [ ] Fix: drafter + spec-round buffers follow the same primary-device rule (or the
       round loop sets the correct current device). Keep device-follow correctness.
6. [ ] Verify: spec+PP-2 c=1/c=2 ~112 class (N>=3), spec-off unchanged, single-card
       unchanged, run-spec K=1..8 PASS, #87 quick crash gate c=4 x50 clean.
7. [ ] Receipts (raw logs + summary) committed here.

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
FIX dev10 "Engine ready (device=0, ...)" expected (head stage). Remaining arms running:
F1r2/r3, F2 c=2 x3, spec-off, door-shut, dev01 differentiator, crash c=4 x50, run-spec.
