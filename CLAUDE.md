# bw24 — project instructions

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
GPU-less). Before any merge or tag, the battery runs on the rig: `kernel-check` ALL GREEN,
`run-gen` argmax MATCH on affected models, `run-spec` K=1..8 self-consistency PASS. Never tag on
a commit whose battery didn't run here.

## Flags doctrine

Winners are defaults — no flag needed to get the tuned path (naked commands = full speed).
Environment variables exist only for: runtime parameters (prompt/gen/spec knobs), machine-specific
config (VRAM budgets, KV formats, spill), rollback seams (`BW24_FAST=0` oracle path), diagnostics,
and explicitly-blocked experimental doors. Catalog: `docs/FLAGS.md`. When an experiment concludes
negative or flat, kill its flag and dispatch arm — the JSONL row is the record, not dead code.
