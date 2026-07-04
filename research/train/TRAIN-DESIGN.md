# TRAIN-DESIGN — the bw24 autotuner-prior training arm (arm 2)

The honest formulation of the thing arm 2 is trying to become, and the growth
path from 44 records to a model that earns its keep.

## Goal

Learn a prior `f(rig, component/kernel, change, baseline metrics) -> (speedup, gate-break risk)`
so a grind session (or a diffusion-autotuner search) can rank candidate changes
*before* paying to run them. Every optimization attempt — win, loss, and
measurement-only — is recorded in `research/tune-data/*.jsonl`; that corpus is
the training set. Negatives and reverts are first-class rows (README rule): the
model must learn what *not* to do and why, so the corpus is never
survivorship-biased.

## The honest constraint: ~65 records still train almost nothing

Current corpus (all rig files, `research/tune-data/*.jsonl`): **65 records**
across **3 rigs** — `rtx5090-laptop-sm120a` (62), `g7e-rtx6000-sm120-188sm` (2),
`l40s-sm89-ec2` (1) — labels `{positive: 34, neutral: 18, negative: 13}`,
headline speedup ratio present in only 34. Component mix is still dominated by
one cluster (`spec-decode` 33, `flash-attn` 9). This is a design *seed*, not a
training set:

- A neural regressor/classifier has far more parameters than we have rows; it
  would memorize, not generalize. Even a gradient-boosted tree wants ~10x the
  rows per feature we'd like to condition on.
- The label distribution is skewed (55% positive) and the 9 negatives are
  spread across kernels, so per-class support is ~3-9. No held-out test set is
  wider than 3-4 rows.
- Many rows are near-duplicates in *change text* but opposite in *outcome*: the
  stream-K MMQ win (row: vendor MMQ, +2.44x) and the stream-K MMQ **revert**
  (gate-break) sit next to each other. Text similarity alone cannot separate
  them — the separating signal is the measured gate result, which is the label
  we're trying to predict.

So the near-term deliverable is **retrieval, not learning**, and the honest
measured numbers below say exactly that.

## Growth path

### (a) NOW (<~200 records): retrieval baseline + precedent oracle — `baseline.py`

Embed each past record's `change + kernel + notes` (BGE-small on CPU if the
model is on-box, else TF-IDF; no network). A proposal is embedded from
`change + kernel` only, nearest-neighbour over the corpus returns:
the matched precedents (with their measured outcome and mechanism as the
explanation), plus a top-k-vote predicted label / median speedup / gate-break
risk.

**Leave-one-out results on the current 65 (both backends, CPU):**

| metric | BGE-small | TF-IDF | baseline to beat |
|---|---|---|---|
| label top-1 acc | 50.8% (95% CI 39-63) | 53.8% (95% CI 42-65) | 52.3% majority ("positive") |
| label top-3 vote | 52.3% | 61.5% | 52.3% |
| speedup-sign acc | 61.8% (N=34) | 58.8% (N=34) | 73.5% majority ("up") |
| leaky upper bound (notes in query) | 63.1% | — | — |

Read this honestly: **as a point predictor the retrieval baseline is at or below
the majority-class bar.** That is expected at N=44 and is *not* the reason this
tool exists. Its value is **precedent retrieval** — "have we tried this, and
what happened?":

- Query "stream-K split on the k-quant MMQ GEMM" and the #2 precedent is the
  exact record where stream-K reordered the f32 partial sums and **broke the
  run-gen argmax gate** — the oracle flags `negative` + gate-break risk. A human
  reads that note and does not re-run the experiment. That is the win, and the
  label-accuracy number does not capture it.
- Query "fuse gate+up FFN matvecs" and #1 is the dual-matvec fusion that landed
  +1.2% — the right neighbourhood, with the measured magnitude attached.

Why the aggregate metric lags the anecdote: within a kernel neighbourhood the
win and its revert are both present, so nearest-neighbour label transfer is a
coin-flip *there* — which is precisely why the session still reads the notes.
The point estimate (especially the median speedup) is the **weakest** output;
the ranked precedent list with mechanisms is the signal.

Ship criterion for (a): it is already useful as an oracle. Do **not** report its
label accuracy as a model result — report it as the floor the next stages beat.

### (b) IMPLEMENTED — gradient-boosted / regularized-linear head — `train_gbm.py`

Built and running now (not waiting for 200 records) so the loop is real and
rerunnable, and so the honest N=65 floor is on the scoreboard. `train_gbm.py`
trains two heads over `dataset.record_to_features` (structured) plus a few dense
text dims from the *same* embedding path `baseline.py` uses (BGE-small on the
colbert-2 venv, else TF-IDF → SVD):

- **label** (3-class positive/neutral/negative),
- **speedup-sign** (up/down/flat, over the 34 rows with a headline ratio).

Trainer choice is box-driven and honest: sklearn is importable here, so the
heads are `HistGradientBoostingClassifier` ("GBM") and an L2 `LogisticRegression`
("LogReg"); if sklearn is absent the trainer degrades to a self-contained numpy
multinomial logistic regression (structured-only). Everything (text vectorizer,
SVD, category vocab, model) is **refit per fold on train only** and both train
and test rows use the notes-free `query_text`, so there is no held-out leakage.

**Leakage caught:** `dataset.INPUT_FEATURES` also lists `n_result_metrics` and
`has_accuracy_gate`, but both are OUTCOME-derived (result{} is the AFTER
measurement, accuracy{} is the gate result) and are *not* known when a session
decides whether to run a change. Conditioning on them inflated label accuracy by
~20 pts — a false win. `train_gbm.py` deliberately EXCLUDES them; the kept
structured features are all knowable at proposal time.

**First measured LOO results (frozen folds, N=65, BGE backend — `eval.sh`):**

| model | LABEL top-1 (95% CI) | SPEEDUP-SIGN (95% CI, N=34) |
|---|---|---|
| majority-class | 52.3% [40, 64] | **73.5%** [57, 85] |
| retrieval-TF-IDF | 53.8% [42, 65] | 58.8% [42, 74] |
| retrieval-BGE | 50.8% [39, 63] | 61.8% [45, 76] |
| **stage-b GBM** | **69.2%** [57, 79] | 67.6% [51, 81] |
| stage-b LogReg | 66.2% [54, 76] | 58.8% [42, 74] |

Read this honestly:

- **On the 3-class label the GBM shows a real, non-leaky edge** (69.2% vs 52-54%
  for retrieval/majority; CI lower bound 57% sits above the majority *point*).
  But a structured-only ablation (no text) scores 66.2% label — so ~all of the
  edge is the `is_measurement`/`is_revert` row-type features, which are
  query-time-knowable yet partly *definitional* of the neutral/negative classes
  (README: neutral = measurement/infra rows; negative = losses/reverts). It is
  a fair signal retrieval ignores, but it is the *easy* axis (row type), not the
  hard positive-vs-negative discrimination the oracle actually needs. Text dims
  add only noise-level lift and flip with the backend (TF-IDF text pushes the
  GBM to 73.8% label / 70.6% sign) — that backend sensitivity is itself small-N
  instability, not signal.
- **On speedup-sign — the decision-relevant target — nothing beats the 73.5%
  majority-sign bar.** The GBM's 67.6% is below it; the structured-only GBM only
  reaches 73.5% by degenerating to "always up". So the sign head does **not**
  clear its bar.

**Verdict: stage-b does NOT yet earn its keep as a point predictor.** Per the
graduation rule below it needs BOTH — a label CI clear of the majority bar AND a
sign accuracy clearing the 73.5% majority-sign bar with a non-overlapping CI —
and the sign head fails. What it delivers today is the *comparable, rerunnable
loop* (`eval.sh`), the leakage discipline, and the honest floor. It graduates
when more headline-ratio rows (now 34) and more negatives/distinct kernels lift
the sign head clear of majority — the data levers below, not a fancier model.

### (c) 1000+ records: LoRA finetune (the autotuner prior proper)

At four figures, finetune a small instruct model into the prior. The proven
in-house pattern to *mirror* (not copy): a **balanced-class binary LoRA judge**
(the sxc recipe — Qwen3.5-9B, binary verdict, balanced classes — beat a prompted
27B). Applied here: a "will this change break an exactness/perf gate?" judge and
a speedup-sign judge, each trained on class-balanced samples (the 24/11/9 skew
must be rebalanced), evaluated on the *same* LOO folds so it is directly
comparable to (a) and (b). The free-text `change`/`notes`/`kernel` fields are the
model's natural input; the structured features become prompt scaffolding.

## Eval protocol — FIXED NOW so every generation is comparable

Defined once here and implemented in `dataset.leave_one_out` +
`baseline.loo_eval` + `train_gbm.loo_head` (all three iterate the identical
folds, so retrieval and the stage-b head are directly comparable):

- **Split:** leave-one-out over all labeled records (no held-out set is
  defensible at this N). Every model generation is scored on identical folds.
- **Metrics:**
  1. **label accuracy** (3-class positive/neutral/negative, top-1 and top-k vote),
  2. **speedup-sign accuracy** (up/down/flat vs 1.0, over rows with a headline
     ratio),
  reported with **Wilson 95% CIs** (they span ~25-30 pts at N=65 — quote them,
  never a bare number), against **majority-class** and **majority-sign**
  baselines as the bar to beat.
- **Leakage boundary** (`dataset.py`): past records are indexed with `notes`
  (`document_text`); a proposal is represented without them (`query_text`). LOO
  holds the held-out row to the query representation so the score reflects real
  query-time conditions. The "notes-in-query" number is printed separately as an
  explicit upper bound (63.1% vs the honest 50.8% for BGE) — it is not the headline.

## What moves the needle (data, not model)

Ranked by expected impact on the prior's usefulness:

1. **More negatives and more distinct kernels.** 13 negatives across ~8
   components is the binding constraint; the model can't learn a gate-break prior
   from 1-2 examples per kernel. Related: only 34/65 rows carry a headline ratio,
   which caps the speedup-sign head — that is the metric still stuck at/below
   majority.
2. **More rigs (partly done).** The corpus now spans 3 rigs
   (`rig5090` 62, `g7e-rtx6000` 2, `l40s-sm89` 1) and `rig` is a real
   multi-value conditioning feature — but the two new rigs contribute 3 rows, so
   cross-rig transfer is not yet measurable. Grow g7e/l40s (and add rigs) to make
   `rig` earn its place instead of being ~constant.
3. **Normalized baseline metrics.** Metric names are per-row free-form
   (`pp512_clock_locked`, `tg128_graph`, …). A small canonicalization pass
   (metric -> {prefill|decode} throughput, ctx length, model) would turn
   `baseline{}` into usable numeric features for stage (b).

## Files

- `dataset.py` — loader, validator, feature extraction, LOO splitter. Zero deps.
  Globs **all** rig files in `research/tune-data/*.jsonl` by default (excludes
  `model-meta.jsonl`). `python3 dataset.py` validates + prints a per-rig summary.
- `baseline.py` — retrieval baseline (BGE/TF-IDF) + oracle. CPU-only, GPU locked
  out at import (`CUDA_VISIBLE_DEVICES=""`).
- `train_gbm.py` — the stage-b head (§(b)): GBM + LogReg over structured + text
  features, scored on the same LOO folds, prints the comparison table. CPU-only.
  `--append-meta` writes a `corpus-meta` snapshot row to `model-meta.jsonl`.
- `eval.sh` — the one-command cadence: dataset validate → retrieval LOO →
  stage-b LOO + comparison table. CPU-only, offline, < 1 min at N=65. Run after
  every ~10 new records.
- `../tune-data/model-meta.jsonl` — model-quality snapshots over corpus growth
  (rig `corpus-meta`, label `neutral`); excluded from the training folds.
- `ORACLE.md` — the one-command grind-session query + the `eval.sh` cadence.
