#!/usr/bin/env python3
"""baseline.py — the retrieval baseline + live oracle for the bw24 tune corpus.

Two jobs:
  1. EVAL   (default): leave-one-out over the labeled corpus, reporting label
     accuracy and speedup-sign accuracy with honest error bars (N~=45 -> wide).
     This is the arm-2 scoreboard every later model (GBM head, LoRA judge) is
     measured against, on identical folds (dataset.leave_one_out).
  2. ORACLE (--query): "proposed change X on kernel Y -> have we tried this?".
     Embeds the proposal, returns nearest past precedents with their measured
     outcome + mechanism as the explanation, and a predicted (label, speedup,
     gate-break risk). This is the immediately-useful grind-session tool.

Embedding backend (auto):
  - BGE-small (BAAI/bge-small-en-v1.5) via sentence-transformers if the model is
    on-box (reuses the llm-wiki setup). CPU-ONLY and CUDA_VISIBLE_DEVICES="" are
    forced at import — a kernel agent owns the GPU; this arm never touches it.
  - Else TF-IDF (scikit-learn), no network. Same interface, refit per fold.

Leakage discipline (see dataset.py): past records are INDEXED with notes
(document_text); a proposal is REPRESENTED without them (query_text). LOO uses
the query representation for the held-out row so the score reflects the oracle's
real operating conditions. A leaky "with-notes" number is also printed as an
explicit upper bound.

Usage:
  # eval (BGE if available, else TF-IDF):
  <colbert2-venv>/bin/python baseline.py
  python3 baseline.py --backend tfidf

  # oracle:
  <colbert2-venv>/bin/python baseline.py \
      --query "fuse the two FFN matvecs into one launch" --kernel qmatvec_nvfp4 -k 5
"""
from __future__ import annotations

# --- HARD GPU LOCKOUT: this arm is CPU-only; the GPU belongs to a kernel agent.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import glob
import math
import sys
from collections import Counter

import numpy as np

import dataset as ds

# BGE query-side instruction (asymmetric retrieval: queries get it, docs don't).
# Matches llm-wiki/tools/query_embed.py.
BGE_QPREFIX = "Represent this sentence for searching relevant passages: "
BGE_CANDIDATE_DIRS = (
    "/data/ai-ml/hf-models/models--BAAI--bge-small-en-v1.5",
    os.path.expanduser("~/ai-ml/hf-models/models--BAAI--bge-small-en-v1.5"),
)


# ---------------------------------------------------------------------------
# Embedders
# ---------------------------------------------------------------------------
def _l2norm(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


class BGEEmbedder:
    """BGE-small dense cosine. No fitting; caches per-text so per-fold LOO
    re-embedding is free."""

    name = "bge-small-en-v1.5 (cpu)"
    needs_refit = False

    def __init__(self, model_path: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_path, device="cpu")
        self._cache: dict = {}

    def _dim(self):
        for meth in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
            f = getattr(self.model, meth, None)
            if callable(f):
                return f()
        return int(self.model.encode(["x"], normalize_embeddings=True).shape[1])

    def _encode(self, texts, prefix=""):
        out = np.empty((len(texts), self._dim()), dtype=np.float32)
        todo, todo_ix = [], []
        for i, t in enumerate(texts):
            key = (prefix, t)
            if key in self._cache:
                out[i] = self._cache[key]
            else:
                todo.append(prefix + t)
                todo_ix.append((i, key))
        if todo:
            vecs = self.model.encode(todo, normalize_embeddings=True, batch_size=32,
                                     show_progress_bar=False).astype(np.float32)
            for (i, key), v in zip(todo_ix, vecs):
                out[i] = v
                self._cache[key] = v
        return out

    def fit(self, docs):  # no-op
        return self

    def embed_docs(self, docs):
        return self._encode(list(docs), prefix="")

    def embed_queries(self, queries):
        return self._encode(list(queries), prefix=BGE_QPREFIX)


class TfidfEmbedder:
    """TF-IDF char+word fallback. Fits on the fold's train docs (no held-out
    vocab leakage), transforms docs and queries into the same space."""

    name = "tfidf (sklearn, word 1-2gram)"
    needs_refit = True

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._Vec = TfidfVectorizer
        self.vec = None

    def fit(self, docs):
        self.vec = self._Vec(lowercase=True, ngram_range=(1, 2),
                             min_df=1, sublinear_tf=True)
        self.vec.fit(list(docs))
        return self

    def embed_docs(self, docs):
        return _l2norm(self.vec.transform(list(docs)).toarray().astype(np.float32))

    def embed_queries(self, queries):
        return _l2norm(self.vec.transform(list(queries)).toarray().astype(np.float32))


def resolve_bge_model():
    env = os.environ.get("BW24_BGE_MODEL")
    dirs = ([env] if env else []) + list(BGE_CANDIDATE_DIRS)
    for d in dirs:
        if not d:
            continue
        snaps = sorted(glob.glob(os.path.join(d, "snapshots", "*")))
        for s in snaps:
            if os.path.exists(os.path.join(s, "config.json")):
                return s
        if os.path.exists(os.path.join(d, "config.json")):
            return d
    return None


def make_embedder(backend: str):
    """backend in {auto, bge, tfidf} -> (embedder, note)."""
    if backend in ("auto", "bge"):
        path = resolve_bge_model()
        if path is not None:
            try:
                return BGEEmbedder(path), f"BGE model: {path}"
            except Exception as e:  # noqa: BLE001
                if backend == "bge":
                    print(f"error: BGE requested but load failed: {e}", file=sys.stderr)
                    sys.exit(2)
                note = f"BGE load failed ({e}); fell back to TF-IDF"
        else:
            if backend == "bge":
                print("error: BGE requested but model not found on box "
                      "(set BW24_BGE_MODEL).", file=sys.stderr)
                sys.exit(2)
            note = "BGE model not found on box; using TF-IDF"
        return TfidfEmbedder(), note
    return TfidfEmbedder(), "TF-IDF (forced)"


# ---------------------------------------------------------------------------
# Stats helpers (honest error bars for small N)
# ---------------------------------------------------------------------------
def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _fmt_pct(x):
    return f"{100 * x:.1f}%"


# ---------------------------------------------------------------------------
# Retrieval core
# ---------------------------------------------------------------------------
def _rank(embedder, doc_texts, query_text):
    """Return similarity-descending index order of doc_texts for one query.
    Assumes embedder already fit (for refit backends)."""
    dmat = embedder.embed_docs(doc_texts)
    qv = embedder.embed_queries([query_text])[0]
    sims = dmat @ qv
    order = np.argsort(-sims)
    return order, sims


def loo_eval(records, embedder, k_vote: int = 3, leaky: bool = False):
    """Leave-one-out. Returns a metrics dict."""
    n = len(records)
    label_top1_hits = 0
    label_vote_hits = 0
    sign_hits = 0
    sign_evalN = 0
    confusion = Counter()  # (actual, pred_top1)
    per_test = []

    for test, train in ds.leave_one_out(records):
        train_docs = [ds.document_text(r) for r in train]
        if embedder.needs_refit:
            embedder.fit(train_docs)
        q = ds.document_text(test) if leaky else ds.query_text(test)
        order, sims = _rank(embedder, train_docs, q)

        top1 = train[order[0]]
        pred_label = top1.label
        label_top1_hits += int(pred_label == test.label)
        confusion[(test.label, pred_label)] += 1

        vote = Counter(train[order[j]].label for j in range(min(k_vote, len(order))))
        pred_vote = vote.most_common(1)[0][0]
        label_vote_hits += int(pred_vote == test.label)

        # speedup sign: nearest neighbour that HAS a defined sign
        pred_sign = None
        for j in order:
            s = train[j].speedup_sign
            if s is not None:
                pred_sign = s
                break
        if test.speedup_sign is not None and pred_sign is not None:
            sign_evalN += 1
            sign_hits += int(pred_sign == test.speedup_sign)

        per_test.append((test, top1, pred_label, pred_sign))

    label_counts = Counter(r.label for r in records)
    maj_label = label_counts.most_common(1)[0]
    sign_counts = Counter(r.speedup_sign for r in records if r.speedup_sign is not None)
    maj_sign = sign_counts.most_common(1)[0] if sign_counts else ("n/a", 0)

    return {
        "n": n,
        "label_top1_acc": label_top1_hits / n,
        "label_top1_ci": wilson_ci(label_top1_hits, n),
        "label_vote_acc": label_vote_hits / n,
        "label_vote_k": k_vote,
        "label_majority": (maj_label[0], maj_label[1] / n),
        "sign_acc": (sign_hits / sign_evalN) if sign_evalN else None,
        "sign_ci": wilson_ci(sign_hits, sign_evalN) if sign_evalN else None,
        "sign_evalN": sign_evalN,
        "sign_majority": (maj_sign[0], (maj_sign[1] / sum(sign_counts.values())) if sign_counts else 0.0),
        "confusion": confusion,
        "label_counts": dict(label_counts),
    }


def print_eval(metrics, embedder_name, note):
    m = metrics
    print("=" * 72)
    print("bw24 tune-corpus retrieval baseline — leave-one-out eval")
    print("=" * 72)
    print(f"embedder : {embedder_name}")
    print(f"           {note}")
    print(f"records  : {m['n']}   labels={m['label_counts']}")
    print()
    lo, hi = m["label_top1_ci"]
    print("LABEL prediction (nearest precedent's verdict)")
    print(f"  top-1 accuracy      : {_fmt_pct(m['label_top1_acc'])}  "
          f"95% CI [{_fmt_pct(lo)}, {_fmt_pct(hi)}]")
    print(f"  top-{m['label_vote_k']} vote accuracy  : {_fmt_pct(m['label_vote_acc'])}")
    print(f"  majority-class base : {_fmt_pct(m['label_majority'][1])}  "
          f"(always predict '{m['label_majority'][0]}')")
    print()
    if m["sign_acc"] is not None:
        lo, hi = m["sign_ci"]
        print("SPEEDUP-SIGN prediction (up / down / flat vs 1.0)")
        print(f"  accuracy            : {_fmt_pct(m['sign_acc'])}  "
              f"95% CI [{_fmt_pct(lo)}, {_fmt_pct(hi)}]  (N={m['sign_evalN']} with headline ratio)")
        print(f"  majority-sign base  : {_fmt_pct(m['sign_majority'][1])}  "
              f"(always predict '{m['sign_majority'][0]}')")
        print()
    print("LABEL confusion (row=actual, col=top-1 predicted)")
    labels = ["positive", "neutral", "negative"]
    header = "           " + "".join(f"{l[:4]:>8}" for l in labels)
    print(header)
    for a in labels:
        row = "".join(f"{m['confusion'].get((a, p), 0):>8}" for p in labels)
        print(f"  {a:<9}{row}")
    print()
    print("CAVEAT: N=%d. The CIs above span ~25-30 points — treat these as a" % m["n"])
    print("        provisional floor, not a validated score. The eval protocol")
    print("        (LOO, these two metrics) is fixed so the GBM head (~200 rec)")
    print("        and LoRA judge (~1000 rec) are directly comparable to this.")


# ---------------------------------------------------------------------------
# Oracle mode
# ---------------------------------------------------------------------------
def oracle(records, embedder, change, kernel, k=5):
    doc_texts = [ds.document_text(r) for r in records]
    if embedder.needs_refit:
        embedder.fit(doc_texts)
    q = ds.proposal_query_text(change, kernel)
    order, sims = _rank(embedder, doc_texts, q)
    top = [(records[i], float(sims[i])) for i in order[:k]]

    print("=" * 72)
    print("PROPOSAL")
    print(f"  change: {change}")
    if kernel:
        print(f"  kernel: {kernel}")
    print(f"  component (inferred): {ds.component_of(kernel, change)}")
    print("=" * 72)

    votes = Counter(r.label for r, _ in top)
    pred_label = votes.most_common(1)[0][0]
    signed = [r.signed_speedup for r, _ in top if r.signed_speedup is not None]
    pred_speedup = float(np.median(signed)) if signed else None
    neg = sum(1 for r, _ in top if r.label == "negative")
    gate_flips = sum(1 for r, _ in top
                     if "flip" in (r.raw.get("notes", "") + str(r.raw.get("accuracy", ""))).lower()
                     or "fail" in str(r.raw.get("accuracy", "")).lower())

    print(f"PREDICTION (top-{k} vote)")
    print(f"  label        : {pred_label}   (votes {dict(votes)})")
    print(f"  speedup      : {('%.3f' % pred_speedup) if pred_speedup is not None else 'n/a (precedents are measurement rows)'}")
    print(f"  gate-break risk: {neg}/{k} precedents were negatives; "
          f"{gate_flips}/{k} mention an argmax/exactness flip or gate FAIL")
    print()
    print(f"PRECEDENTS (nearest {k})")
    for rank, (r, sc) in enumerate(top, 1):
        sg = r.signed_speedup
        sg_s = f"{sg:.3f}x" if sg is not None else "n/a"
        print(f"  #{rank} sim={sc:.3f}  [{r.label}] speedup={sg_s}  component={r.component}")
        print(f"       change: {r.change}")
        print(f"       kernel: {r.kernel[:100]}")
        notes = r.notes.strip().replace("\n", " ")
        if notes:
            print(f"       notes : {notes[:300]}{'…' if len(notes) > 300 else ''}")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=ds.DEFAULT_CORPUS, help="corpus JSONL path")
    ap.add_argument("--backend", choices=["auto", "bge", "tfidf"], default="auto")
    ap.add_argument("--query", help="oracle mode: proposed change description")
    ap.add_argument("--kernel", default="", help="oracle mode: kernel/component touched")
    ap.add_argument("-k", type=int, default=5, help="neighbours to show/vote (oracle)")
    ap.add_argument("--vote-k", type=int, default=3, help="top-k vote for eval label acc")
    ap.add_argument("--leaky", action="store_true",
                    help="eval with notes in the query (leaky upper bound)")
    args = ap.parse_args()

    records = ds.load(args.file, validate=True)
    embedder, note = make_embedder(args.backend)

    if args.query:
        oracle(records, embedder, args.query, args.kernel, k=args.k)
        return

    metrics = loo_eval(records, embedder, k_vote=args.vote_k, leaky=False)
    print_eval(metrics, embedder.name, note)
    if not embedder.needs_refit:
        leaky_m = loo_eval(records, embedder, k_vote=args.vote_k, leaky=True)
        print()
        print(f"(leaky upper bound, notes IN query) label top-1 = "
              f"{_fmt_pct(leaky_m['label_top1_acc'])} — the honest oracle number "
              f"is the {_fmt_pct(metrics['label_top1_acc'])} above.")


if __name__ == "__main__":
    main()
