#!/usr/bin/env python3
"""train_gbm.py — the STAGE-B parametric head for the bw24 tune corpus.

Stage (b) of TRAIN-DESIGN.md: a small supervised model over dataset.py's
features — the structured block (rig, component, is_measurement, is_revert,
metric counts, gate presence, text lengths) PLUS a few dense text dims from the
same embedding path baseline.py uses (BGE-small on the colbert-2 venv, else
TF-IDF). Two heads:
  (a) label     : positive / neutral / negative  (3-class)
  (b) speedup-sign : up / down / flat vs 1.0      (over rows with a headline ratio)

It is scored ONLY through the frozen leave-one-out protocol in dataset.py
(ds.leave_one_out) — the SAME folds baseline.py uses — so this head is directly
comparable to the retrieval baselines it prints alongside. Everything (text
vectorizer, dimensionality reduction, category vocab, and the model) is refit
per fold on the training split only, so there is no held-out leakage. Both train
and test rows are represented by the notes-free query_text (dataset.query_text),
i.e. the model only ever sees what a grind session has BEFORE running the
experiment — retrieval's use of corpus-side notes is its structural advantage,
not a protocol difference.

Trainer choice (honest, box-driven): sklearn is importable here, so the two
heads are (default) sklearn HistGradientBoostingClassifier ("gbm") and an L2
LogisticRegression ("logreg"). If sklearn is absent the trainer degrades to a
self-contained numpy multinomial logistic regression on structured features
only — the point is a RUNNABLE, rerunnable loop, not a fancy model.

At N~50-65 the expectation (stated in TRAIN-DESIGN.md) is that this head does
NOT beat retrieval and sits near the majority-class floor; the deliverable is
the comparable, rerunnable loop, not a win. The printed table says so honestly.

Usage:
  python3 train_gbm.py                      # full table (BGE row if venv has it)
  python3 train_gbm.py --text-backend tfidf # force TF-IDF text dims
  python3 train_gbm.py --no-bge             # skip the BGE retrieval row (faster)
  # BGE dims + BGE retrieval row need sentence-transformers (colbert-2 venv):
  /home/avifenesh/projects/colbert-2/.venv/bin/python train_gbm.py
"""
from __future__ import annotations

# --- HARD GPU LOCKOUT: this arm is CPU-only; the GPU belongs to a kernel agent.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Be a good citizen on a rig shared with LLM servers: cap the BLAS/OpenMP thread
# pools BEFORE numpy/sklearn import them. Refitting ~200 tiny models per run
# otherwise oversubscribes every core (2000% CPU) for no wall-time gain.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import warnings
from collections import Counter

import numpy as np

warnings.filterwarnings("ignore")  # sklearn 1.9 deprecation chatter drowns the table

import dataset as ds
import baseline as bl

LABELS = ["positive", "neutral", "negative"]
SIGNS = ["up", "down", "flat"]

# Sklearn is the box's available trainer; probe it once here.
try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.decomposition import TruncatedSVD
    HAVE_SKLEARN = True
except Exception:  # noqa: BLE001
    HAVE_SKLEARN = False


# ---------------------------------------------------------------------------
# numpy multinomial logistic regression — the bare-box fallback (no sklearn)
# ---------------------------------------------------------------------------
class NumpyLogReg:
    """Softmax regression with L2, batch gradient descent. Deterministic, tiny,
    dependency-free — used only when sklearn is unavailable so the loop still
    runs. Not tuned; it exists to keep the trainer honest and runnable."""

    def __init__(self, l2=1.0, lr=0.5, iters=300):
        self.l2, self.lr, self.iters = l2, lr, iters
        self.classes_ = None
        self.W = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        self.classes_ = sorted(set(y))
        cidx = {c: i for i, c in enumerate(self.classes_)}
        Y = np.zeros((len(y), len(self.classes_)))
        for i, v in enumerate(y):
            Y[i, cidx[v]] = 1.0
        n, d = X.shape
        Xb = np.hstack([X, np.ones((n, 1))])  # bias column
        self.W = np.zeros((d + 1, len(self.classes_)))
        for _ in range(self.iters):
            logits = Xb @ self.W
            logits -= logits.max(axis=1, keepdims=True)
            P = np.exp(logits)
            P /= P.sum(axis=1, keepdims=True)
            grad = Xb.T @ (P - Y) / n
            grad[:-1] += self.l2 / n * self.W[:-1]  # L2 on weights, not bias
            self.W -= self.lr * grad
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        Xb = np.hstack([X, np.ones((len(X), 1))])
        return np.array([self.classes_[i] for i in (Xb @ self.W).argmax(axis=1)])


# ---------------------------------------------------------------------------
# Feature assembly (structured block + a few dense text dims), refit per fold
# ---------------------------------------------------------------------------
# Numeric structured features (standardized on train stats for the linear head;
# harmless for the tree head).
#
# LEAKAGE NOTE: dataset.INPUT_FEATURES also lists `n_result_metrics` and
# `has_accuracy_gate`, but both are OUTCOME-derived — result{} is the AFTER
# measurement and accuracy{} is the gate result, neither known when a grind
# session is deciding whether to run the change. Conditioning on them inflates
# label accuracy by ~20 pts (a false win). We deliberately EXCLUDE them so the
# head sees only what query_text sees: the proposal, not its result. The kept
# features are all knowable at proposal time (is_measurement/is_revert come from
# the change description; n_baseline_metrics is the pre-change state you already
# measured).
_NUM_KEYS = ("is_measurement", "is_revert", "n_baseline_metrics",
             "change_len", "kernel_len")


def _num_row(rec):
    f = ds.record_to_features(rec)
    return [float(f[k]) for k in _NUM_KEYS]


class FoldFeaturizer:
    """Fits category vocab + numeric standardizer + text vectorizer/SVD on the
    TRAIN split only, then transforms any record set into a dense matrix. One
    instance per LOO fold => no held-out leakage."""

    def __init__(self, embedder, text_dims):
        self.embedder = embedder          # None => structured-only
        self.text_dims = text_dims
        self.rig_vocab = []
        self.comp_vocab = []
        self.mu = None
        self.sd = None
        self.svd = None

    def fit(self, train):
        self.rig_vocab = sorted({r.rig for r in train})
        self.comp_vocab = sorted({r.component for r in train})
        num = np.array([_num_row(r) for r in train], dtype=np.float64)
        self.mu = num.mean(axis=0)
        self.sd = num.std(axis=0)
        self.sd[self.sd == 0] = 1.0
        if self.embedder is not None:
            docs = [ds.query_text(r) for r in train]
            if self.embedder.needs_refit:
                self.embedder.fit(docs)
            M = self.embedder.embed_docs(docs)
            if HAVE_SKLEARN and self.text_dims and M.shape[1] > self.text_dims:
                k = min(self.text_dims, M.shape[0] - 1, M.shape[1] - 1)
                k = max(1, k)
                self.svd = TruncatedSVD(n_components=k, random_state=0).fit(M)
        return self

    def _onehot(self, val, vocab):
        v = [0.0] * len(vocab)
        if val in vocab:
            v[vocab.index(val)] = 1.0
        return v

    def transform(self, recs):
        rows = []
        for r in recs:
            row = self._onehot(r.rig, self.rig_vocab) + self._onehot(r.component, self.comp_vocab)
            num = (np.array(_num_row(r)) - self.mu) / self.sd
            row += list(num)
            rows.append(row)
        X = np.array(rows, dtype=np.float64)
        if self.embedder is not None:
            T = self.embedder.embed_docs([ds.query_text(r) for r in recs])
            if self.svd is not None:
                T = self.svd.transform(T)
            X = np.hstack([X, T.astype(np.float64)])
        return X


def make_model(kind):
    if not HAVE_SKLEARN:
        return NumpyLogReg()
    if kind == "gbm":
        return HistGradientBoostingClassifier(
            max_depth=3, max_iter=120, learning_rate=0.1,
            l2_regularization=1.0, min_samples_leaf=3, random_state=0)
    return LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")


# ---------------------------------------------------------------------------
# Leave-one-out for a parametric head (label + sign), frozen folds
# ---------------------------------------------------------------------------
def loo_head(records, kind, target, embedder, text_dims):
    """LOO over `records` (ds.leave_one_out), refitting featurizer+model per
    fold. target='label' -> 3-class label; target='sign' -> speedup sign over
    rows that have a defined sign. Returns (hits, n)."""
    hits = 0
    n = 0
    for test, train in ds.leave_one_out(records):
        if target == "sign":
            if test.speedup_sign is None:
                continue
            train = [r for r in train if r.speedup_sign is not None]
            y = [r.speedup_sign for r in train]
            gold = test.speedup_sign
        else:
            y = [r.label for r in train]
            gold = test.label
        if len(set(y)) < 2:  # degenerate fold: predict the only class present
            pred = y[0]
        else:
            fz = FoldFeaturizer(embedder, text_dims).fit(train)
            Xtr = fz.transform(train)
            Xte = fz.transform([test])
            model = make_model(kind)
            model.fit(Xtr, y)
            pred = model.predict(Xte)[0]
        hits += int(pred == gold)
        n += 1
    return hits, n


# ---------------------------------------------------------------------------
# Retrieval baseline rows (reuse baseline.loo_eval on the same folds)
# ---------------------------------------------------------------------------
def retrieval_row(records, embedder):
    m = bl.loo_eval(records, embedder, k_vote=3, leaky=False)
    lh = round(m["label_top1_acc"] * m["n"])
    return {
        "label": (m["label_top1_acc"], bl.wilson_ci(lh, m["n"]), m["n"]),
        "sign": ((m["sign_acc"], m["sign_ci"], m["sign_evalN"])
                 if m["sign_acc"] is not None else None),
    }


def try_bge():
    """Return a BGEEmbedder or None (no sentence-transformers / model missing)."""
    try:
        path = bl.resolve_bge_model()
        if path is None:
            return None, "BGE model not found on box"
        return bl.BGEEmbedder(path), f"BGE: {path}"
    except Exception as e:  # noqa: BLE001  (import or load failure)
        return None, f"BGE unavailable ({type(e).__name__}: {e})"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _cell(acc, ci, n):
    if acc is None:
        return f"{'n/a':>26}"
    lo, hi = ci
    return f"{bl._fmt_pct(acc):>6} [{bl._fmt_pct(lo)},{bl._fmt_pct(hi)}] N={n:<3}"


def build_table(records, text_backend, text_dims, use_bge_row):
    recs = ds.labeled(records)
    n = len(recs)

    # --- majority baselines (from the corpus label / sign distributions) ---
    lab_counts = Counter(r.label for r in recs)
    maj_label = lab_counts.most_common(1)[0]
    sign_counts = Counter(r.speedup_sign for r in recs if r.speedup_sign is not None)
    maj_sign = sign_counts.most_common(1)[0]
    sign_n = sum(sign_counts.values())

    rows = []
    rows.append(("majority-class",
                 (maj_label[1] / n, bl.wilson_ci(maj_label[1], n), n),
                 (maj_sign[1] / sign_n, bl.wilson_ci(maj_sign[1], sign_n), sign_n)))

    # --- retrieval baselines (same folds) ---
    tfidf = bl.TfidfEmbedder()
    r_tfidf = retrieval_row(recs, tfidf)
    rows.append(("retrieval-TFIDF", r_tfidf["label"],
                 r_tfidf["sign"] if r_tfidf["sign"] else (None, None, 0)))

    bge_note = "skipped (--no-bge)"
    if use_bge_row:
        bge, bge_note = try_bge()
        if bge is not None:
            r_bge = retrieval_row(recs, bge)
            rows.append(("retrieval-BGE", r_bge["label"],
                         r_bge["sign"] if r_bge["sign"] else (None, None, 0)))
        else:
            rows.append(("retrieval-BGE", (None, None, 0), (None, None, 0)))

    # --- the stage-b heads (text dims via chosen backend) ---
    text_emb = None
    text_note = "structured-only (no text backend)"
    if text_backend != "none":
        if text_backend in ("auto", "bge"):
            b, _ = try_bge()
            if b is not None:
                text_emb, text_note = b, "text dims: BGE-small"
            elif text_backend == "bge":
                text_emb, text_note = None, "BGE requested but unavailable -> structured-only"
        if text_emb is None and text_backend in ("auto", "tfidf"):
            if HAVE_SKLEARN:
                text_emb, text_note = bl.TfidfEmbedder(), "text dims: TF-IDF -> SVD"
            else:
                text_note = "no sklearn/BGE -> structured-only"

    model_kind = "gbm" if HAVE_SKLEARN else "logreg(numpy)"
    for kind, name in ((model_kind, f"stage-b GBM ({'HistGBC' if HAVE_SKLEARN else 'numpy-logreg'})"),
                       ("logreg", "stage-b LogReg (L2)" if HAVE_SKLEARN else None)):
        if name is None:
            continue
        lh, ln = loo_head(recs, kind, "label", text_emb, text_dims)
        sh, sn = loo_head(recs, kind, "sign", text_emb, text_dims)
        rows.append((name,
                     (lh / ln, bl.wilson_ci(lh, ln), ln),
                     (sh / sn, bl.wilson_ci(sh, sn), sn) if sn else (None, None, 0)))

    return rows, n, text_note, bge_note, maj_label, maj_sign


def print_table(rows, n, text_note, bge_note, sklearn_ok):
    print("=" * 78)
    print("bw24 stage-b head vs baselines — leave-one-out (frozen folds, N=%d)" % n)
    print("=" * 78)
    print(f"trainer   : {'sklearn HistGBC + L2 LogReg' if sklearn_ok else 'numpy multinomial logreg (no sklearn)'}")
    print(f"text feats: {text_note}")
    print(f"bge row   : {bge_note}")
    print()
    print(f"{'model':<28}{'LABEL top-1 (95% CI)':>26}{'  ':>2}{'SIGN up/down/flat (95% CI)':>26}")
    print("-" * 82)
    for name, lab, sign in rows:
        la, lci, ln = lab
        sa, sci, sn = sign
        print(f"{name:<28}{_cell(la, lci, ln)}{'  '}{_cell(sa, sci, sn)}")
    print("-" * 82)
    print("Read honestly: at N~50-65 the stage-b head is NOT expected to beat")
    print("retrieval and sits near the majority-class floor (CIs span ~25-30 pts and")
    print("overlap every row). The deliverable is this comparable, rerunnable loop —")
    print("stage-b earns its keep only when the label CI clears the majority bar AND")
    print("the sign accuracy clears the majority-sign bar (see TRAIN-DESIGN.md (b)).")
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=ds.DEFAULT_CORPUS, help="corpus (dir/glob/file)")
    ap.add_argument("--text-backend", choices=["auto", "bge", "tfidf", "none"],
                    default="auto", help="text-feature embedding path for the head")
    ap.add_argument("--text-dims", type=int, default=24, help="SVD dims for text block")
    ap.add_argument("--no-bge", action="store_true", help="skip the retrieval-BGE row")
    ap.add_argument("--emit-record", action="store_true",
                    help="print a corpus-meta JSONL record with the eval table")
    ap.add_argument("--append-meta", action="store_true",
                    help="append the corpus-meta record to model-meta.jsonl "
                         "(implies --emit-record; excluded from training by dataset.py)")
    args = ap.parse_args()

    records = ds.load(args.file, validate=True)
    rows, n, text_note, bge_note, maj_label, maj_sign = build_table(
        records, args.text_backend, args.text_dims, use_bge_row=not args.no_bge)
    print_table(rows, n, text_note, bge_note, HAVE_SKLEARN)

    if args.emit_record or args.append_meta:
        import json
        import datetime
        table = {name: {"label_acc": (None if lab[0] is None else round(lab[0], 4)),
                        "label_ci": (None if lab[1] is None else [round(x, 4) for x in lab[1]]),
                        "sign_acc": (None if sign[0] is None else round(sign[0], 4)),
                        "sign_ci": (None if sign[1] is None else [round(x, 4) for x in sign[1]])}
                 for name, lab, sign in rows}
        rec = {
            "ts": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "rig": "corpus-meta",
            "commit": "stage-b eval snapshot",
            "change": f"stage-b LOO eval on consolidated corpus (N={n}); "
                      f"trainer={'sklearn HistGBC+LogReg' if HAVE_SKLEARN else 'numpy-logreg'}, {text_note}",
            "kernel": "research/train/train_gbm.py (stage-b head vs retrieval/majority)",
            "baseline": {"majority_label_acc": round(maj_label[1] / n, 4),
                         "majority_label": maj_label[0]},
            "result": table,
            "speedup_graph": None,
            "clock_note": "leave-one-out over labeled records; frozen folds (dataset.leave_one_out); CPU-only",
            "accuracy": {"protocol": "LOO, label top-1 + speedup-sign, Wilson 95% CI"},
            "label": "neutral",
            "notes": "corpus-meta row: tracks stage-b model quality over corpus growth. "
                     "At this N the stage-b head does not beat retrieval and overlaps the "
                     "majority floor (CIs ~25-30pt). Regenerate via research/train/eval.sh "
                     "after ~10 new records; stage-b graduates when its label CI clears the "
                     "majority bar and sign acc clears the majority-sign bar.",
        }
        line = json.dumps(rec)
        print("\nCORPUS-META RECORD (rig=corpus-meta; tracked in model-meta.jsonl,")
        print("excluded from training folds by dataset.py):")
        print(line)
        if args.append_meta:
            with open(ds.MODEL_META_FILE, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            print(f"\nappended to {ds.MODEL_META_FILE}")


if __name__ == "__main__":
    main()
