# specplace - placement-aware speculative serving policy

Lane: `lane/cx-spec-placement`, train `13f5ddb8`. Rig: box2
(`ubuntu@13.59.112.147`), 2x RTX PRO 6000 Blackwell Server 96 GB. Primary policy
model: q9 (`Qwen3.5-9B-NVFP4-MTP-GGUF.gguf`). Step35 is a PP-2 spot-check because
the merged batched step35 arm changed its plain-decode denominator to the
81/96/116/129-130 aggregate tok/s class at c=1/2/4/8.

`~/.lanectl/inbox/cx-specplace.md` was absent at lane start. The lane registry
does contain `cx-specplace` on this worktree with status `running`, but no lane
note was available.

## Question and decision rule

The scheduler currently uses one concurrency policy everywhere:

```
admit spec while active + 1 <= LOW=2
demote live spec at active >= HIGH=4
```

That policy was calibrated on one 5090. PP-2 has a different execution shape:
the draft chain remains a serial queue while plain decode uses batched,
stage-split execution.

Decision rule fixed before new measurements:

1. If PP-2 spec OFF wins every q9 cell by a non-close margin, default PP-2 to
   never admit spec. Keep `MEMRA_SPEC_GATE=0` as the always-spec rollback seam
   and keep explicit LOW/HIGH overrides available.
2. If PP-2 spec wins only at c=1, use PP-2 defaults LOW=1/HIGH=2.
3. If any policy-boundary cell is close, do not flip a default; publish a
   verdict-class report.
4. Single-card keeps the existing gate unless the same-train re-sweep refutes
   its c=1/c=2 win.

For this lane, "close" means the N=3+ arm medians overlap by run spread or the
winner is under 5%. Existing PP-2 deltas are 2x-5x, so they are not close.

## q9 decision matrix

Aggregate output tok/s, greedy. `S` forces the pure spec path with
`MEMRA_SPEC_GATE=0`; `N` forces plain batched serving with
`MEMRA_SERVE_SPEC=0`.

| placement | c | S spec ON | N spec OFF | S/N | current verdict | receipt / status |
|---|---:|---:|---:|---:|---|---|
| PP-2 dev10 | 1 | 112.5 | 223.3 | 0.50x | OFF wins | `research/pp2spec-crash-20260807/PROGRESS.md`, perf section, N=3; corroborated post-v0.72 fix at 111.9 vs 221.7 |
| PP-2 dev10 | 2 | 112.3 | 340.3 | 0.33x | OFF wins | same receipt, N=5; spec corroborated post-v0.72 fix at 111.9, N=3 |
| PP-2 dev10 | 4 | 112.1 | 593.4 | 0.19x | OFF wins | same receipt, N=5; #87 post-fix crash arm remains 111.5-111.6 |
| single card | 1 | 251.9 | 138.5 | 1.82x | stale: ON wins | `research/spec-gate-20260806/RESULTS.md`, N=5 on the pre-current core; re-sweep required |
| single card | 2 | 250.8 | 221.5 | 1.13x | stale: ON wins | same receipt; re-sweep required because this is the narrow crossover cell |
| single card | 4 | 249.7 | 383.5 | 0.65x | stale: OFF wins | same receipt; re-sweep required |

The PP-2 cells are already filled by same-rig, interleaved receipts and are
large-margin results. They will be cited, not re-measured. The missing work is
the single-card c=1/2/4 ladder on the current train.

## Step35 PP-2 spot-check

The current-train plain-decode denominator is already receipted by
`research/step35-batch-20260808/` on box2, N=3:

| c | spec ON | spec OFF | status |
|---:|---:|---:|---|
| 1 | missing | 81.0 | measure S, cite N |
| 2 | missing | 96.3 | measure S, cite N |
| 4 | missing | 116.2 | measure S, cite N |

Step35 cannot be a single-card placement cell: the 105 GB trunk is a PP-2
capacity artifact. Its role here is to check that PP-2's policy verdict survives
the newly merged batched denominator on the launch architecture.

## Measurement protocol

- One exclusive `/tmp/memra-gpu.lock` hold per sweep.
- Current lane commit transferred without an origin push; clean release build on
  box2.
- q9 single-card: S/N, c=1/2/4, N=3+, rep-major interleaving, greedy,
  identical prompt and token budget.
- Step35 PP-2: forced S at c=1/2/4, N=3+, compared only to the already-current
  N=3 box2 receipt above. Re-measure N only if the inherited artifact/config
  cannot be reproduced exactly.
- Raw server logs, load JSONL, build provenance, GPU pre/post state, medians,
  run spread, and error counts land under `research/specplace-20260808/raw/`.
- No cross-day absolute comparison decides a close cell. Existing PP-2 cells
  are used only because their margins are far outside the close class and were
  repeated after the v0.72 head-affinity fix.

## Required gates after any policy change

- `run-spec` K=1..8 on single card and PP-2.
- `tools/serve-smoke.sh`.
- #87 quick crash gate on PP-2 with `MEMRA_SPEC_GATE=0` so the default-off policy
  cannot hide the formerly fatal path.
- Default-policy boot evidence: PP-2 admits no spec; single-card still admits
  spec at low concurrency.

## Log

- 2026-08-08: lane started at `13f5ddb8`; `CLAUDE.md` read; branch clean.
- 2026-08-08: prior PP-2, v0.72-fix, spec-gate, and step35-batch receipts
  extracted into the matrices above.
