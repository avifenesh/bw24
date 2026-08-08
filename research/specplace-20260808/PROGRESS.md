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
| single card | 1 | 374.8 | 224.5 | 1.67x | ON wins | current train `d2a52fe6`, N=3, ranges 371.2-375.9 vs 223.8-224.6 |
| single card | 2 | 374.5 | 347.5 | 1.08x | ON wins | current train, N=3, ranges 370.5-378.8 vs 346.2-348.2: no overlap |
| single card | 4 | 377.3 | 617.1 | 0.61x | OFF wins | current train, N=3, ranges 369.4-377.6 vs 612.2-617.9 |

The PP-2 cells are already filled by same-rig, interleaved receipts and are
large-margin results and were cited instead of re-measured. The current-train
single-card re-sweep satisfies the stale-verdict law: absolute throughput moved,
but the crossover remains between c=2 and c=4. Even the narrow c=2 cell clears
the 5% bar and has no run-range overlap.

## Step35 PP-2 spot-check

Both arms were re-measured because the inherited step35 plain receipt used a
sampled load while the policy surface is greedy:

| c | spec ON | spec OFF | S/N | verdict |
|---:|---:|---:|---:|---|
| 1 | 35.9 (35.8-35.9) | 85.7 (85.7-85.7) | 0.42x | OFF wins |
| 2 | 36.2 (36.2-36.3) | 101.6 (101.5-101.6) | 0.36x | OFF wins |
| 4 | 36.7 (36.7-36.7) | 121.7 (121.5-122.0) | 0.30x | OFF wins |

All cells are N=3 medians on current train `d2a52fe6`. Forced-spec logs contain
104 `[spec-acc]` lines per rep; forced-plain logs contain zero and show the new
`[step35-batch] first B>1` arm, so the comparison exercised the intended paths.

Step35 cannot be a single-card placement cell: the 105 GB trunk is a PP-2
capacity artifact. Its role here is to check that PP-2's policy verdict survives
the newly merged batched denominator on the launch architecture.

## Measurement verdict

**Decisive default flip: PP-2 defaults to spec OFF. Single-card keeps LOW=2,
HIGH=4.**

- q9 PP-2: OFF wins every inherited cell, including 2.0x at c=1.
- step35 PP-2 on the current batched core: OFF wins 2.39x at c=1, 2.80x at c=2,
  and 3.32x at c=4.
- No PP-2 cell is close under the pre-declared rule.
- Single-card still has a real spec win at c=1/c=2 and a decisive loss at c=4,
  so changing its policy would discard measured throughput.

Implementation target: placement-aware gate defaults. Sharded cross-device PP
uses LOW=0/HIGH=1 (never admit spec); single-card keeps LOW=2/HIGH=4. Explicit
`MEMRA_SPEC_GATE_LOW`/`_HIGH` values override those defaults, and
`MEMRA_SPEC_GATE=0` remains the rollback to always-spec on every placement.

## Measurement protocol

- One exclusive `/tmp/memra-gpu.lock` hold per sweep.
- Current lane commit transferred without an origin push; clean release build on
  box2.
- q9 single-card: S/N, c=1/2/4, N=3+, rep-major interleaving, greedy,
  identical prompt and token budget.
- Step35 PP-2: forced S/N at c=1/2/4, N=3, because the inherited N receipt was
  sampled rather than greedy.
- Raw server logs, load JSONL, build provenance, GPU pre/post state, medians,
  run spread, and error counts land under `research/specplace-20260808/raw/`.
- No cross-day absolute comparison decides a close cell. Existing PP-2 cells
  are used only because their margins are far outside the close class and were
  repeated after the v0.72 head-affinity fix.

Actual window: one lock hold, 2026-08-08 10:18:40Z-10:30:32Z. Both GPUs were at
0 MiB on entry and exit; `nvidia-smi` reported no compute processes at either
boundary. Load points: 36, with 0 errors and 0 shed requests. Temperatures stayed
31-46 C during the sweep. Artifact and binary hashes are in
`raw/{artifact,binary}-sha256-20260808T101200Z.txt`.

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
- 2026-08-08: current-train box2 sweep completed, 36/36 clean load points. The
  decision rule selects PP-2 default OFF and preserves the single-card gate.
