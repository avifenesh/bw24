# bw24 — project instructions

## Branch isolation

Feature and research work MUST happen on a dedicated branch/worktree, never directly on `main`.
Preserve unrelated dirty work and stage only the intended lane.

## Hy3 spilling and quantization research

This lane owns two separate deliverables: (1) spill-path improvements for large expert banks, and
(2) a controlled four-arm quantization study. Do not trade correctness in one track for a result in
the other, and report spill performance separately from model-quality comparisons.

- `HostExps.layouts == None` is the uniform-layout fast-path contract. `Some(layouts)` makes each
  expert's `qtype`, `row_bytes`, `len`, and `offset` authoritative; use `expert_layout()` and
  `max_expert_bytes()` rather than projection-wide fields.
- Mixed layers run through metadata-aware staged, SLRU-cache, or grouped dispatch. Resident slab,
  pointer-table, pairs, dev, and grouped-decode fused kernels remain uniform-only until they group
  pointers by layout; never send mixed metadata through those kernels.
- A v2 tier plan MUST assign every retained expert projection to Q2_K, Q3_K, or NVFP4. Missing
  assignments are errors; never silently retain a BF16 expert. Q2_K remains on the generic staged
  f32-dequant kernel until the target-rig correctness and performance gates justify a fast path.
- A plan's pruned expert ids keep their original router positions. `active_experts()` masks them
  before top-k and their weights must be absent. Never dispatch, cache, or fabricate bytes for a
  masked id, and never let a fallback uniform slab bypass split expert overrides.
- The public Hy3 REAP50 checkpoint renumbers retained experts and publishes no original-id list.
  Recover the frozen mask only through `tools/recover_hy3_reap_mask.py`: require one-to-one router
  row matches, the locked nearest-match margin, and exact correction-bias confirmation. Scored
  artifacts always quantize the pinned BF16 source; never re-quantize the public MLX experts.
- The four scored arms are fixed in `research/per-expert-quant/arms.lock.json`: `plain_quant`
  (full bank, uniform NVFP4), `plain_reap_quant` (REAP50 mask, uniform NVFP4),
  `plain_reap_mix_quant` (REAP50 mask, 48 least-used Q2_K plus 48 NVFP4), and `mix_quant`
  (full bank, hottest 25% NVFP4, middle 50% Q3_K, coldest 25% Q2_K, zero-count pruned).
- BF16 Hy3 is source material only, never an evaluation arm. All four arms must share the same
  source revision, non-expert tensor encodings, REAP mask where applicable, prompt template,
  runtime commit, and evaluation settings.
- Rank per layer from non-public calibration traces and freeze trace/plan hashes before viewing
  public eval scores. Uniform plans must not consume calibration traces.
- Public eval runs require `ARTIFACT` and must retain its manifest/hash. Public benchmark data
  must never select experts, thresholds, tier fractions, or pruning decisions.
- Model loading, spill correctness, research measurements, artifact generation, and public evals
  run on the provisioned G7e research machine. Do not merge or tag this lane until its remote raw logs
  and four-arm eval report exist.
- The local RTX 5090 rig remains bw24's deployment and final performance target. Treat G7e results
  as research evidence, not a default-flip decision; re-run correctness, memory, and throughput gates
  on the 5090 before shipping any runtime default.
- Optimize expert serving as one storage-to-compute pipeline: mmap/zero-copy, local-NVMe access,
  pinned host memory, residency caching, asynchronous prefetch/overlap, PCIe transfer, and GPU
  kernels. Measure the stages together so a faster kernel cannot hide a data-movement regression.
- Keep durable model/artifact copies under `/data`, but stage byte-identical scored artifacts onto
  the G7e local NVMe (`/scratch`) for calibration, public evals, and spill benchmarks. Record the
  staged manifest hash; do not report persistent-EBS 4 KiB fault throughput as bw24 spill speed.

Why: a projection-wide dtype silently decodes some experts with the wrong block layout; routing a
pruned id dereferences nonexistent weights; and a G7e-only performance win may not transfer to the
5090's smaller HBM and different storage/PCIe balance.

## Perf board: README must stay current, every push

The tuning campaign lands new numbers several times a day (`research/tune-data/rig5090.jsonl` is
the append-only research log). The README's Performance section and `docs/perf-card.svg` are
**generated**, not hand-written — they come from `research/tune-data/current-board.json` via
`tools/update-perf-board.py`.

Rule: any commit that changes the *published* numbers (a board-moving merge — i.e. the numbers
that belong in the README table, not every raw jsonl row) MUST:

1. Update `research/tune-data/current-board.json` with the new values.
2. Run `python3 tools/update-perf-board.py` to regenerate README.md's perf tables and
   `docs/perf-card.svg`.
3. Commit the JSON + the regenerated README.md + SVG together, in the same commit as the
   number-moving change.

Never hand-edit the table rows or the date line inside the `<!-- PERF-*:START -->` /
`<!-- PERF-*:END -->` marker blocks in README.md — edit `current-board.json` and regenerate.
Prose around the tables (depth-behavior notes, mechanism writeups, "why it moved") stays
hand-written; only the tables + date sentence are mechanical.

A `pre-push` hook (`tools/hooks/pre-push`, wired via `git config core.hooksPath tools/hooks`)
runs `tools/update-perf-board.py --check` and refuses the push if the board and README have
drifted — treat a failure there as "regenerate and re-commit." **Never** bypass with `--no-verify`.

This does not cover the GitHub repo social-preview image (the OG thumbnail used for link
shares) — GitHub has no API for that field, it's a manual upload in Settings → Social preview,
and isn't worth automating at this update cadence.

## Correctness discipline

Same three gates as CONTRIBUTING.md: `kernel-check`, the `run-gen` argmax gate, and `run-spec`
K=1..8 self-consistency. A kernel change without before/after numbers measured per
`research/benchmarks.md` isn't done.

## Evidence discipline (measurement lanes)

- Raw sweep output is part of the deliverable: commit the per-run JSONL/log next to the summary
  row (`research/<lane>/`), never summary-only. A claim whose raw runs exist nowhere in the repo
  is not evidence.
- Never let a pipe swallow error output: `run-* 2>&1 | parser` loses the failure text. Always
  `tee` a raw log first, parse the log second.
- Failure causes are quoted, never inferred: "OOM" means a captured `out of memory` /
  `CUDA_ERROR_OUT_OF_MEMORY` line, with the concurrent-GPU state recorded (`nvidia-smi`
  compute-apps at failure time). A run that died without captured stderr is "died, cause
  unknown — repro needed", and no conclusion may be built on it.
- Every published median states its N and its thermal regime; single runs are labeled single
  runs.

## Releases: every board-moving or user-facing change

Tag it — `git tag vX.Y.Z && git push origin vX.Y.Z`. The `release` workflow compiles, drafts the
changelog from conventional commits (`tools/changelog.sh`), and publishes. Minor bump per
mechanism/board move, patch per fix/docs. Full process: `docs/RELEASING.md`. Commit prefixes feed
the changelog: `perf:`/`feat:`/`fix:`/`config:`/`docs:` are public; `data:`/`chore:`/`wip:`/`probe:`
are filtered as research-log noise — pick the prefix accordingly.

## CI is compile-only; the exactness battery is the real gate

GitHub runners have no GPU. `.github/workflows/ci.yml` catches build breaks (nvcc compiles fine
GPU-less). Before any merge or tag, the battery runs on the designated target GPU rig:
`kernel-check` ALL GREEN, `run-gen` argmax MATCH on affected models, `run-spec` K=1..8
self-consistency PASS. Never tag a commit without the target-rig battery.

## Flags doctrine

Winners are defaults — no flag needed to get the tuned path (naked commands = full speed).
Environment variables exist only for: runtime parameters (prompt/gen/spec knobs), machine-specific
config (VRAM budgets, KV formats, spill), rollback seams (`BW24_FAST=0` oracle path), diagnostics,
and explicitly-blocked experimental doors. Catalog: `docs/FLAGS.md`. When an experiment concludes
negative or flat, kill its flag and dispatch arm — the JSONL row is the record, not dead code.
