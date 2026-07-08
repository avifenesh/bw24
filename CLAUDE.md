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
