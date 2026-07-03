# ORACLE — "have we tried this?" for a grind session

Before you spend a session on a kernel change, ask the corpus whether a past
attempt already settled it. One command:

```bash
# BGE embedding (best matches) — uses the on-box bge-small via the colbert-2 venv:
/home/avifenesh/projects/colbert-2/.venv/bin/python \
    research/train/baseline.py \
    --query "split k across CTAs with stream-K on the k-quant MMQ prefill GEMM" \
    --kernel "mul_mat_q_q45k stream_k" -k 5
```

No BGE / fully offline / any Python? Same command with the system interpreter
falls back to TF-IDF automatically (or force it):

```bash
python3 research/train/baseline.py --backend tfidf \
    --query "fuse gate and up FFN projections into one matvec launch" \
    --kernel "qmatvec_nvfp4_mmvq" -k 5
```

Both are **CPU-only** — the GPU is left alone for the kernel work
(`CUDA_VISIBLE_DEVICES=""` is forced inside the script).

## What you get

- **PROPOSAL** — your change, and the component it was bucketed into.
- **PREDICTION** — top-k-vote label, median speedup, and a **gate-break risk**
  count (how many of the nearest precedents were negatives / mention an
  argmax/exactness flip or gate FAIL).
- **PRECEDENTS** — the nearest past records with their measured label, speedup,
  and the `notes` mechanism.

## How to read it

- The **precedents and their notes are the signal.** That is where you see "we
  ported stream-K, it reordered f32 partial sums, and it broke the run-gen
  argmax gate" and decide not to re-run it.
- The **point prediction is weak** at this corpus size (label accuracy ~= the
  majority baseline; the median-speedup estimate mixes different changes in the
  neighbourhood). Treat `label`/`speedup` as a hint, `gate-break risk` as a
  yellow flag, and the precedent notes as the actual answer. See `TRAIN-DESIGN.md`
  for the measured numbers and why.
- A high similarity (#1 `sim` > ~0.8 on BGE) to a precedent with the same kernel
  is a strong "already answered" signal. Low top similarity (< ~0.3) means the
  corpus has no close precedent — you are in new territory, go run it.

## After you run the experiment

Append the result — win, loss, or measurement — as one JSONL line to
`research/tune-data/rig5090.jsonl` (schema + measurement rules in that dir's
`README.md`). Reverts and gate-breaks are the most valuable rows; record them
with `label: negative` and the mechanism. The oracle gets better the moment the
file grows — no rebuild step, `baseline.py` reads the corpus live.
