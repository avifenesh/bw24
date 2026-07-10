# bw24 — project instructions

## Branch isolation

Feature and research work MUST happen on a dedicated branch/worktree, never directly on `main`.
Preserve unrelated dirty work and stage only the intended lane.

## Per-expert mixed precision

- `HostExps.layouts == None` is the uniform-layout fast-path contract. `Some(layouts)` makes each
  expert's `qtype`, `row_bytes`, `len`, and `offset` authoritative; use `expert_layout()` and
  `max_expert_bytes()` rather than projection-wide fields.
- Mixed layers run through metadata-aware staged, SLRU-cache, or grouped dispatch. Resident slab,
  pointer-table, pairs, dev, and grouped-decode fused kernels remain uniform-only until they group
  pointers by layout; never send mixed metadata through those kernels.
- Sparse overlay manifests require `BW24_FULL_PREC=1`: selected experts come from the overlay,
  every unlisted expert must retain the base checkpoint's BF16 bytes, and
  `preserve_expert_encodings()` keeps the overlay authoritative even for an all-Q4_K control.
- This research lane is implementation/CPU-validation only on the current host. Model loading,
  GPU correctness gates, performance measurements, and public evals run on the provisioned target
  machine. Do not merge or tag mixed-expert work until its remote raw logs and eval report exist.

Why: a projection-wide dtype silently decodes some experts with the wrong block layout; a local
performance or quality claim would be fabricated because this host is not the experiment machine.

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
