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

## The honest constraint: 44 records train nothing

Current corpus (`rig5090.jsonl`, one rig): **44 records**, labels
`{positive: 24, neutral: 11, negative: 9}`, headline speedup ratio present in
only 23. Component mix is dominated by one cluster (`spec-decode` 22,
`flash-attn` 7). This is a design *seed*, not a training set:

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

**Leave-one-out results on the current 44 (both backends, CPU):**

| metric | BGE-small | TF-IDF | baseline to beat |
|---|---|---|---|
| label top-1 acc | 47.7% (95% CI 34-62) | 50.0% (95% CI 36-64) | 54.5% majority ("positive") |
| label top-3 vote | 45.5% | 59.1% | 54.5% |
| speedup-sign acc | 47.8% (N=23) | 52.2% (N=23) | 69.6% majority ("up") |
| leaky upper bound (notes in query) | 61.4% | — | — |

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

### (b) ~200+ records: gradient-boosted head on structured + text features

When per-class support reaches ~40+ negatives (currently 9) and components
diversify, train a small model on `dataset.record_to_features` (rig, component,
is_measurement, is_revert, metric counts, gate presence, text lengths) plus a
few dense dims from the BGE embedding. Two heads:

- classifier: `label` (or the more decision-relevant binary **gate-break: yes/no**),
- regressor: signed speedup.

Gradient-boosted trees (or a 1-2 layer MLP over frozen embeddings) — small
enough to not overfit ~200 rows, and they consume the structured baseline
metrics the retrieval baseline ignores. Graduate from (a) when the LOO label CI
stops overlapping the majority baseline and the regressor's sign accuracy clears
the 69.6% majority-sign bar.

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

Defined once here and implemented in `dataset.leave_one_out` + `baseline.loo_eval`:

- **Split:** leave-one-out over all labeled records (no held-out set is
  defensible at this N). Every model generation is scored on identical folds.
- **Metrics:**
  1. **label accuracy** (3-class positive/neutral/negative, top-1 and top-k vote),
  2. **speedup-sign accuracy** (up/down/flat vs 1.0, over rows with a headline
     ratio),
  reported with **Wilson 95% CIs** (they span ~25-30 pts at N=44 — quote them,
  never a bare number), against **majority-class** and **majority-sign**
  baselines as the bar to beat.
- **Leakage boundary** (`dataset.py`): past records are indexed with `notes`
  (`document_text`); a proposal is represented without them (`query_text`). LOO
  holds the held-out row to the query representation so the score reflects real
  query-time conditions. The "notes-in-query" number is printed separately as an
  explicit upper bound (61.4% vs the honest 47.7%) — it is not the headline.

## What moves the needle (data, not model)

Ranked by expected impact on the prior's usefulness:

1. **More negatives and more distinct kernels.** 9 negatives across ~8
   components is the binding constraint; the model can't learn a gate-break prior
   from 1-2 examples per kernel.
2. **A second rig.** `l40s-sm89.jsonl` lives on the sm89 branch; `rig` is already
   a feature. Once merged, cross-rig transfer becomes measurable and `rig`
   becomes a real conditioning variable instead of a constant.
3. **Normalized baseline metrics.** Metric names are per-row free-form
   (`pp512_clock_locked`, `tg128_graph`, …). A small canonicalization pass
   (metric -> {prefill|decode} throughput, ctx length, model) would turn
   `baseline{}` into usable numeric features for stage (b).

## Files

- `dataset.py` — loader, validator, feature extraction, LOO splitter. Zero deps.
  `python3 dataset.py` validates + summarizes.
- `baseline.py` — retrieval baseline (BGE/TF-IDF) + oracle. CPU-only, GPU locked
  out at import (`CUDA_VISIBLE_DEVICES=""`).
- `ORACLE.md` — the one-command grind-session query.
