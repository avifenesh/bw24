# v0.72 tag-gate RE-BATTERY (lane/v072-rebattery)

Target: train `restructure/public-split` @ **5ad87a63** (the merged tip carrying both
tag-blocker fixes). The full battery ran green at 6afc4f65 (`research/v072-prep-20260808/`)
with two named blockers; this lane re-verifies only the fixed cells post-merge plus a
smoke regression layer, per the release rule "never tag a commit without the target-rig
battery."

Rig: box2 `13.59.112.147` — 2x RTX PRO 6000 Blackwell Server 96GB, CUDA 13.2, sm_120a
auto-detected. `~/memra` fetched to 5ad87a63 (pre-existing spot-guard wip checkpoint
preserved on box branch `keep/spot-guard-023823Z`), `cargo build --release` clean.
All GPU windows one `flock /tmp/memra-gpu.lock` hold per driver (R1 04:36:48–04:52:32Z,
R2 04:52:32–04:55:09Z, R3 04:55:09–05:00:32Z). Raw receipts: `raw/` (drivers committed
beside them). Box left clean: 0 MiB both cards, lock released.

## Verdicts

### R1 — blocker 1 (tickinv35c canary inert -> fixed by 73c65c91, merged d8363ccd)
raw/R1-tick-20260808T043648Z.log, rc=0

| gate | verdict |
|---|---|
| R1a tickinv35 naked | **PASS** — EXACT maxdiff 0.000e0 at budgets 1024/513/512/256/64 + sp64/sp256/sp512, streams identical |
| R1b tickinv35c canary | **PASS (teeth restored)** — canary BROKE the assertion as required; DIFFER on every arm |

R1b divergence table is digit-identical to the fix lane's pre-merge receipt
(research/v072-fix1-20260808: 1024→1.310e0, 513→1.115e0, 512/256/64→1.164e0@row512,
sp64→1.204e0, sp256→1.533e0, sp512→1.164e0; stream divergence step 6 except 513@step 3).
The post-merge tip reproduces the exact FA-view-tile-grid canary signature — the gate can
fail again.

### R2 — blocker 2 (spec+PP-2 serving 112.5 -> 17.5, fixed by 05ddfef2, merged d1abd0f3)
raw/R2-serve-20260808T045232Z.log + R2-points-*.jsonl, rc=0. q9 embedded MTP, greedy,
max_tokens 96, N=3 on every headline cell, all cells 0 err.

| cell | N | agg tok/s | expectation | verdict |
|---|---|---|---|---|
| dev10 naked c=1 (spec admitted, #89 LOW=2) | 3 | 111.2 / 111.4 / 111.5 | ~112 class | **PASS** |
| dev10 SPEC_GATE=0 c=2 (the crash-gate shape) | 3 | 111.4 / 111.6 / 111.5 | ~112 class | **PASS** |
| dev01 naked c=1 (the differentiator — pre-fix ~20x slow always) | 3 | 111.0 / 110.8 / 110.8 | ~111 | **PASS** — head-affinity confirmed at the merged tip |
| dev10 spec-OFF c=1 control | 1 | 221.9 | ~222 unchanged | PASS (fix-lane receipt 221.7) |
| door-shut single-card spec c=4 control | 1 | 543.3 | ~548 class unchanged | PASS (fix-lane receipt 543.5) |

Mechanism receipt in the server logs: dev10 boots `Engine ready (device=0 ...)` and dev01
boots `Engine ready (device=1 ...)` — the worker primary follows the PP HEAD stage, and
`[spec-acc]` lines are live throughout (spec engaged, not demoted).

### R3 — smoke layer (fast regression sweep at the tip)
raw/R3-smoke-20260808T045509Z.log, rc=0

| gate | verdict |
|---|---|
| R3a kernel-check model-backed step35 IQ4_XS | **ALL GREEN** |
| R3b kernel-check model-backed q27 NVFP4 | **ALL GREEN** |
| R3c run-gen argmax step35 PP-2 (64 tok) | **MATCH** — argmax 6776 prefill/decode/batched-prime/tokenwise, same token as the 6afc4f65 battery + tick-seg baselines |
| R3d run-gen argmax q27 single-card | **MATCH** — argmax 1178, same as battery A4 |
| R3e #87 crash gate dev10 SPEC_GATE=0 c=4 x50 | **50/50 ok, 0 err**, agg 111.5 (the fixed class, not 17.5), server log fault-grep 0 (no sentinel/illegal/panic/CUDA_ERROR) |
| R3f serve-smoke FULL battery (q9+draft + gemma4 arm staged) | **0 failed, 47 ok-checks** — cache-metering exact, spec==plain greedy, truncation matrix, affinity resume, gemma4 default-scheduler clean + thinking separation + zero panics |

Note R3e doubles as a perf receipt: the crash-gate shape that read 17.5 agg at 6afc4f65
now reads 111.5 at the tip — blocker 2's regression is gone in the exact cell that found it.

## Tag readiness (prepared, NOT tagged — orchestrator tags)

- [x] Both blocker fixes re-verified post-merge on the pair box (this file).
- [x] Smoke layer green (kernel-check x2, run-gen x2 topologies, #87 c=4 x50, serve-smoke full).
- [x] `tools/changelog.sh v0.71.0` runs clean; preview committed
      (`changelog-v072-preview.md`).
- [x] `python3 tools/update-perf-board.py --check` → "perf board is up to date" (rc=0).
- [x] Version bump per docs/RELEASING.md: workspace `Cargo.toml` 0.71.0 → 0.72.0
      (`[workspace.package].version` + all 8 pinned `[workspace.dependencies]`) +
      `Cargo.lock` refresh, committed on this lane (publish.yml refuses a
      tag/version mismatch). Docs sweep: no other current-version references need a bump
      (README/docs 0.71 mentions are historical receipts only).
- [ ] ORCHESTRATOR: merge lane/v072-rebattery to the train (the bump must ride the train
      before the tag).
- [ ] ORCHESTRATOR: `git tag v0.72.0 && git push origin v0.72.0` on the post-merge tip.
- [ ] ORCHESTRATOR: edit the drafted GitHub release notes if the headline needs context
      (draft is floor, not ceiling).

Thermal regime: all serving medians above are N=3 back-to-back on a cold-start box
(0 MiB idle before R1); single-run cells labeled N=1.
