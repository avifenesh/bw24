# Perf-board verdict for H3 (+8.33% q9 / +5.19% q27 serve c=1) — 2026-08-05

**Verdict: NO published board number moves. No current-board.json delta is needed.**

Receipt, checked on lane/v070-prep at 7ac05f54 + docs sweep:

1. `research/tune-data/current-board.json` top-level sections: `plain_decode`,
   `speculative`, `plain_decode_depth`, `samples`, `supported_models`, `h100_board`,
   `extra_card_rows`, plus metadata (`updated`, `rig`, `protocol`,
   `bold_ratio_threshold`). Grep for serve-path signatures: zero `c=1` cells, no 46.09,
   no 170.5, no serve tok/s of any kind. Every generated cell is a **bare-CLI**
   (`run-gen`/`run-spec`) or H100-board number.
2. The only place serve c=1 numbers are published is the **27B serving board** section of
   docs/PERFORMANCE.md — hand-written prose, deliberately OUTSIDE the `PERF-*` marker
   blocks (markers live at lines ~201/212/222/344; the serving board is at ~68). The
   PERFORMANCE.md header states the generated scope: "the full tracked boards … the
   README shows only a few representative samples". Serve numbers were never in the
   generated surface — confirmed also by the serve-path docs commit 32eeadb6, whose
   message records "Perf board untouched: it carries naked plain_decode/speculative
   surfaces only, no serve board, and update-perf-board.py --check reports up to date."
3. `python3 tools/update-perf-board.py --check` → "perf board is up to date" on this
   tree, after the docs sweep (rc=0).
4. H3's own receipts corroborate the direction of the non-move: H3 changes what the
   **serve** path dispatches at b_n==1; the bare-CLI `run-gen` denominators the board
   tracks are the *reference* H3 converged serve TO (serve c=1 123.7 → 134.1 vs run-gen
   134.8/134.5/134.0 on the same board — `research/servepath-p2-20260805/RESULTS.md`).
   Nothing upstream of `run-gen` changed.

## Tag-day follow-up (optional, receipts-gated, NOT a tag blocker)

The hand-written 27B serving board row "Spec decode K=3, nv, through the serve surface at
c=1 = 170.5" and the retired-gap prose are pre-H3 measurements on the 188-SM pod. If
tag-day battery time on that rig allows, re-measure serve c=1 (plain and spec) N=5 per
`research/benchmarks.md` and update the hand-written rows with the new rig-labeled
medians. That is a prose edit, not a current-board.json edit — no
`tools/update-perf-board.py` regeneration is involved unless a bare-CLI tracked cell also
moved (then: edit JSON → `python3 tools/update-perf-board.py` → commit JSON + README +
PERFORMANCE.md + SVGs together, per CLAUDE.md).
