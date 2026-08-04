# Stale-branch sweep — 2026-08-04

Repo hygiene review flagged by the v0.68 release: 8 stale candidates, evaluated against
`main` (v0.68.0, 57686050) and `restructure/public-split` (3b98ca63). Unmerged-commit
counts were identical vs both bases for all 8 branches, so evidence below cites main.

Method per branch: `git log --oneline main..BRANCH` + `git cherry` for patch-equivalence,
diffstat vs merge-base, supersession/abandonment evidence located on main (commits, jsonl
rows, docs, PR state) before any deletion. Archive tags taken for every branch with >5
unmerged commits, plus fa-fp8-pv-cpasync (unique kernel work).

## Verdict table

| Branch | Tip | Unmerged (main / public-split) | Class | Action |
|---|---|---|---|---|
| restructure/docs-scrub | b713fdec (Jul 31) | 1 / 1 | SUPERSEDED | deleted (local only) |
| kv-fp4 | f8cda266 (Jul 21) | 12 / 12 | ABANDONED-NEGATIVE | tagged `archive/kv-fp4`, deleted local + origin |
| draft-dispatch | b43c3f08 (Jul 17) | 2 / 2 | SUPERSEDED | deleted local + origin (PR #13 CLOSED) |
| sv-draft-head | d433c17c (Jul 12) | 1 / 1 | SUPERSEDED | deleted local + origin (PR #7 MERGED) |
| feat/per-expert-quant | a62e8784 (Jul 10) | 48 / 48 | SUPERSEDED (44/48 patch-equivalent) | KEPT — checked out in codex worktree; report only |
| lane/q6krp | 70cfdb22 (Jul 8) | 1 / 1 | ABANDONED-NEGATIVE (neutral) | tagged `archive/lane/q6krp`, deleted (local only) |
| lane/g7e | a38f4df6 (Jul 4) | 9 / 9 | SUPERSEDED | tagged `archive/lane/g7e`, deleted (local only) |
| fa-fp8-pv-cpasync | a0555f3d (Jun 28) | 1 / 1 | SUPERSEDED | tagged `archive/fa-fp8-pv-cpasync`, deleted (local only) |

No LIVE-VALUE branch found; nothing escalated as unbanked work.

## Per-branch evidence

### restructure/docs-scrub — SUPERSEDED, deleted
One commit: `docs: lane serving content moves to the private distribution` (scrubs lane
scheduling from README/ARCHITECTURE-H100 + deletes darkbatch-20260730 logs). The same
scrub was merged onto the public-split lineage as merge `4a289637` ("Merge docs-scrub:
lane serving content moves to the private distribution") — the merged tree drops "lane
scheduling"/bw24-lanes from the README (confirmed absent on main and public-split today)
while deliberately KEEPING the darkbatch raw logs (present in both trees; evidence-
discipline rule says raw runs stay). The branch tip's extra deletions of raw logs were
not taken, which is the current standing state — so the branch carries nothing wanted.
Its agent worktree (`.claude/worktrees/agent-a17b5b11482c5fc44`) was clean (0 dirty
files) and was removed. Local-only branch; no origin ref.

### kv-fp4 — ABANDONED-NEGATIVE, archived + deleted
12 commits: NVFP4 KV-cache format arm (kf4/vf4/kf4vf4 fatbins, kernel-check twins,
battery + deep-context sweeps). The lane's own final data commits record the negative:
`8a0d4d63 data: kf4 full battery — correctness green everywhere … default-flip NO at
board depths` and `be2e9373 data: kf4 deep-context verdict — loses all 36 deep cells
(pre-LUT arm); capacity-only feature`. Corroborating standing record on main:
`research/perf-frontier-20260802/REPORT.md` Appendix B (refuted/blocked) — "fp8 K-cache:
spec self-consistency FAIL … fp8 KV e2e flat-to-negative, reverted 2026-07-28" (the KV-
quant lever class closed), and rig5090.jsonl row 346 (2026-07-28 KV-format adoption
reverted). Per flags doctrine, a concluded-negative arm's flag and code die; the jsonl
rows + raw logs on the branch are preserved under `archive/kv-fp4` (pushed to origin).
Deleted locally and on origin. No PR referenced it.

### draft-dispatch — SUPERSEDED, deleted
2 commits: `BW24_MTP_DRAFT2` second resident draft head + per-request `pick_draft`
routing by (ctx len, sampled). PR #13 is CLOSED (not merged). Superseded by the draft
REGIME rollout on main (`docs/DRAFT-REGIME.md`, rig5090.jsonl 2026-07-18 "REGIME
ROLLOUT — every supported Qwen…" row 345): one regime draft file per model, zero flags,
plus per-model draft attachment via `MEMRA_MODELS name=/model.gguf+/draft.gguf`
(crates/memra-server/src/main.rs `parse_models_config`, worker.rs draft attach) — the
multi-head dispatch problem is now solved per-model at the server layer, and no
MTP_DRAFT2/dispatch symbol exists on main. Its payoff cell was also blocked by the
DraftGeom sampled regression (issue #12, now CLOSED via a different fix). Deleted
locally and on origin.

### sv-draft-head — SUPERSEDED, deleted
1 commit (local), 2 on origin. The feature itself (DraftGeom distilled-student draft
head) landed on main as `f1395e90` (same title) via merged PR #7. The origin tip's extra
review commit `19fcf82f` (defensive shape asserts at student-draft load) is present on
main in renamed form: `crates/memra-engine/src/hybrid.rs:713-723` carries the eh_proj
`2*n_embd` named assert with the same comment language ("named assert, not later as
garbage drafts"). Nothing unbanked. Deleted locally and on origin.

### feat/per-expert-quant — SUPERSEDED, KEPT (checked out)
48 commits, but `git cherry` shows 44/48 patch-equivalent to main — the lane landed via
`origin/feat/per-expert-quant-rebase` (PR #2 MERGED, 0 commits ahead of main). The 4
non-patch-equivalent commits (1db887b0 usage-tiered recipes, b7eed57a mmap prefetch
window, c7791ae7 disk extents for explicit I/O, 8c1e291f worker prefetch overlap) all
exist on main in renamed/evolved form: `tools/build_expert_tier_plan.py` +
`make_random_tier_control.py`, `MEMRA_MOE_PAGE_PREFETCH_WINDOW` (docs/FLAGS.md:171),
expert disk extents (`memra-engine/src/model.rs:1059 find_expert_disk`,
memra-gguf/src/source.rs), and the bounded-worker-prefetch spill backend
(`MEMRA_SPILL_IO=worker|direct`, spill_pread.rs). Branch is CHECKED OUT in the codex
worktree `~/.codex/worktrees/bw24-per-expert-quant` (clean, in sync with
origin/feat/per-expert-quant) — per sweep scope, branch NOT deleted; state reported.
Recommendation for the orchestrator: once the codex worktree is confirmed done, remove
the worktree and delete local + origin branch (content fully banked; no archive tag
needed beyond origin/feat/per-expert-quant-rebase which is merged).

### lane/q6krp — ABANDONED-NEGATIVE (neutral), archived + deleted
1 commit: q6_K lm_head split-plane repack, 10 bit-identical `_rp` twins behind
BW24_Q6K_RP=1. Verdict recorded on main in rig5090.jsonl:217 (2026-07-08 4-lane
ultracode merge row): "q6krp NEUTRAL (rp dispatched, 968us == 967us plain — q6_K is
instruction-ISSUE bound not addressing-bound; branch kept unmerged)", with the lesson
(decode instruction count is the lever, split-plane only pays when loads straggle)
banked in the same row. Per flags doctrine the jsonl row is the record, not dead code.
Kernel code preserved at `archive/lane/q6krp`. Local-only; deleted.

### lane/g7e — SUPERSEDED, archived + deleted
9 commits: the July-4/5 G7e RTX 6000 decode campaign (A2 expert-grouped prefill,
staging-elision triple, stage-2 grouped decode, FA split-32 default, T=1 rows routing,
shared-K negative). Supersession record on main: rig5090.jsonl row 70 (2026-07-05,
"landed G7e-lane work on main: T=1 rows routing + split-32 default + pmin 0.15 +
shared-K negative + NO_EVT defa…") plus the merged main lineage (expert-grouped/
grouped-GEMM merges, e.g. 245ca2ef/0e949b43). The branch's 12 g7e-rtx6000.jsonl rows are
a subset of main's 39-row file except one (fa-split-32-default row) whose content is
duplicated by main's row 11 (same ts/commit/change, minor wording drift) and by the
rig5090.jsonl landing row. G7e remote-tracking refs (`g7e/lane/g7e` etc.) belong to the
DEAD g7e host — unreachable for push deletes; harmless, prune with
`git remote remove g7e` if/when the remote is retired (owner call, not taken here).
Tagged `archive/lane/g7e`; local branch deleted.

### fa-fp8-pv-cpasync — SUPERSEDED, archived + deleted
1 commit (Jun 28, pre-resurrection era): cp.async double-buffered FA + native sm_120
fp8-PV mma. The mechanism landed independently and better on the modern tree: main's
flash_attn.cu carries fp8-e4m3 KV formats with native `cvt.f32.e4m3` decode
(flash_attn.cu:230-250) and the cp.async staging lineage (`cdb84481 FA cp.async 2-stage
ring on bf16 K/V tiles … bit-identical`, FA3 harness arc). The branch predates the
project-kill/resurrection boundary and its base (e192d380) is a month behind; the diff
does not apply to the current FA architecture. fp8-PV as a lossy-attention direction is
additionally closed by perf-frontier Appendix B (SageAttention3/FP4 attention: "lossy —
outside the exact verify lane by law"). Tagged `archive/fa-fp8-pv-cpasync`; local-only;
deleted.

## Tags created

- `archive/kv-fp4` -> f8cda266 (pushed to origin — 12 commits incl. raw battery logs)
- `archive/lane/g7e` -> a38f4df6 (local)
- `archive/lane/q6krp` -> 70cfdb22 (local)
- `archive/fa-fp8-pv-cpasync` -> a0555f3d (local)

## Worktree state after sweep

Removed: `/home/avifenesh/projects/bw24/.claude/worktrees/agent-a17b5b11482c5fc44`
(docs-scrub agent worktree, clean) + `git worktree prune`. `.claude/worktrees/` is now
empty. Remaining worktrees:

| Path | Branch |
|---|---|
| /home/avifenesh/projects/bw24 | main |
| /home/avifenesh/projects/bw24-unified | restructure/public-split |
| /home/avifenesh/.codex/worktrees/bw24-per-expert-quant | feat/per-expert-quant (kept, see above) |
| /home/avifenesh/projects/wt-fp8mmq2 | lane/fp8-mmq-v2 (active fp8 lanes, not in sweep scope) |
| /home/avifenesh/projects/wt-fp8ship | lane/fp8-ship (active fp8 lanes, not in sweep scope) |

## Origin deletions

`kv-fp4`, `draft-dispatch`, `sv-draft-head` deleted from origin (no open PRs — PR #13
CLOSED, PR #7 MERGED; no tags pointed at the tips before the archive tags above).
Remaining stale-looking origin refs (feat/per-expert-quant, feat/per-expert-quant-rebase,
and the wider origin/lane/* + codex/* population) were outside this sweep's 8-candidate
scope — a follow-up origin-side sweep is a separate task.
