# v0.71 prep — perf-board verdict (2026-08-06, lane/v071-prep @ a85135ae + prep commits)

**Verdict: NO board delta from the pile. `update-perf-board.py --check` GREEN on this tree.**

Checked, same shape as the v070 receipt:

- `research/tune-data/current-board.json` sections: `plain_decode`, `speculative`,
  `plain_decode_depth`, `samples`, `supported_models`, `h100_board`, `extra_card_rows` —
  all bare-CLI (`run-gen`) and H100 cells. Programmatic scan for serve-path fields
  (ttft / first_text / contended / admission / cadence): zero hits (one `sse` substring
  false-positive inside the word "classes" — inspected, not a serve cell).
- The pile's numbers are serve-path (SSE cadence, admission, spec-burst tiers), exactness
  (chunkinv, k27), or non-board deltas (block-128 +1.69% decode and warmups +1.1% are
  FP8-ST/graph-path cells not tracked in current-board.json; the tracked plain/spec cells
  did not move). The 27B serving board and the felt-latency numbers live in HAND-WRITTEN
  prose in docs/PERFORMANCE.md/SERVING.md, outside the PERF-* markers — updated in the
  docs-sweep commit of this lane, no regeneration required or permitted for them.
- Standing rule re-verified: no serve number is generated anywhere; if a tag-day
  re-measure moves a *bare-CLI* tracked cell, edit current-board.json + regenerate +
  commit together, per CLAUDE.md.
