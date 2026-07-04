# tune-data — labeled optimization corpus (arm 3)

One JSONL record per optimization attempt on a specific rig. This is TRAINING DATA for a
future config→performance model (the autotuner arm): every record pairs a concrete change
(kernel, launch geometry, layout, dispatch policy) with its measured effect and its
accuracy-gate outcome, including NEGATIVES and structural-closure reasoning — the corpus
must never be survivorship-biased toward wins.

## Schema (one JSON object per line)
- `ts`            ISO timestamp
- `rig`           hardware id (e.g. `rtx5090-laptop-sm120a`, `l40s-sm89-ec2`)
- `commit`        repo state marker (informal; `reverted-uncommitted` for non-landed attempts)
- `change`        one-sentence description of the attempted change
- `kernel`        kernel/component touched
- `baseline`      object: metric name -> value BEFORE (same-session interleaved where possible)
- `result`        object: metric name -> value AFTER + gate outcomes
- `speedup_graph` headline ratio (null when not a single-ratio change)
- `clock_note`    measurement-protocol notes (clock lock, interleave, run count)
- `accuracy`      object: gate results / exactness constraints discovered
- `label`         `positive` | `negative` | `neutral` (negative = measured loss or gate break; neutral = measurement/infra/protocol records)
- `notes`         free-form: mechanism, lesson, next lever

## Rules
- Interleaved same-minute A/B only for cross-engine ratios (cross-session ratios lie ~10%).
- Real prompts for spec-decode verdicts (synthetic under-states acceptance ~20pts).
- Record REVERTED attempts with `label: negative` and the mechanism — they are the most
  valuable training rows (the model must learn what NOT to do and why).
- Per-rig files (all loaded together by `research/train/dataset.py`, which globs
  `*.jsonl`): `rig5090.jsonl` (RTX 5090 laptop, this box, 62), `g7e-rtx6000.jsonl`
  (RTX PRO 6000 Blackwell, 2), `l40s-sm89.jsonl` (4x L40S sm_89, 1; the
  bring-up row was normalized from the closed sm89 branch into this schema).
- `model-meta.jsonl` is NOT training data: it holds `corpus-meta` rows written by
  `train_gbm.py --append-meta` (stage-b eval snapshots) and is excluded from the
  glob / LOO folds.
