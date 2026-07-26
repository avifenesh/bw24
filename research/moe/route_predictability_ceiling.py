#!/usr/bin/env python3
"""Measure the offline predictability ceiling of Hy3 expert routing from banked calibration
weight traces — the reopen-gate question for the prefetch-predictor lane.

Two prediction axes, each with the lead time that matters for NVMe prefetch:
  A. Cross-layer, within token: routes at layer L predict routes at layer L+d.
     Lead ~2.5 ms per layer of d. The old lane died at d>=16 with 10-34% precision
     from a naive predictor; this measures what a trained co-occurrence table gets.
  B. Cross-token, same layer: routes at token t predict routes at token t+lag.
     Lead = one full decode step (~200 ms) per lag — enough for any queued read.

Predictors:
  - freq: static per-layer top-8 by training routing mass (the no-signal baseline)
  - cooc: score_j = sum_{i in observed set} C[i][j] over the training co-occurrence table
  - persist (axis B only): predict exactly the previous token's top-8

Split: train on the first --train prompts, test on the rest (prompt-level holdout).
Metric: precision@8 = |predicted top-8 ∩ actual top-8| / 8, averaged over test positions,
plus the non-resident variant counting only experts OUTSIDE the static top-K "resident"
set per layer (--resident-k) — prefetch only pays on cache misses.
"""
import argparse, glob, os
from collections import defaultdict

import numpy as np

N_EXPERT = 192


def parse_runs(paths):
    """-> list of R[token, layer, expert] uint8 route tensors (token = position order)."""
    runs = []
    for path in paths:
        # Trace `t` is the forward index, not the position: sequential prefill writes one
        # line per position per layer, all stamped with the same t, in position-major order.
        # Recover the position as the per-layer occurrence count (decode lines then continue
        # the count seamlessly).
        per_tok = defaultdict(dict)
        seen = defaultdict(int)
        with open(path) as fh:
            for line in fh:
                layer_s, _t_s, pairs = line.split(" ", 2)
                layer = int(layer_s)
                pos = seen[layer]
                seen[layer] += 1
                per_tok[pos][layer] = [
                    int(p.split(":")[0]) for p in pairs.strip().split(",")
                ]
        toks = sorted(per_tok)
        n_layer = max(l for t in per_tok.values() for l in t) + 1
        R = np.zeros((len(toks), n_layer, N_EXPERT), dtype=np.uint8)
        for row, t in enumerate(toks):
            for l, ids in per_tok[t].items():
                R[row, l, ids] = 1
        runs.append(R)
    return runs


def top8(scores):
    """scores [n, 192] -> bool mask of top-8 per row."""
    idx = np.argpartition(scores, -8, axis=1)[:, -8:]
    mask = np.zeros_like(scores, dtype=bool)
    np.put_along_axis(mask, idx, True, axis=1)
    return mask


def prec(pred_mask, actual, miss_mask):
    hits = (pred_mask & (actual > 0)).sum()
    denom = 8 * len(actual)
    m = (actual > 0) & miss_mask
    miss_hits = (pred_mask & m).sum()
    return hits, denom, miss_hits, m.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--train", type=int, default=18)
    ap.add_argument("--resident-k", type=int, default=48)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.traces, "weights-*.trace")),
                   key=lambda p: int(p.split("-")[-1].split(".")[0]))
    runs = parse_runs(files)
    L = min(r.shape[1] for r in runs)
    runs = [r[:, :L, :] for r in runs]
    train, test = runs[: args.train], runs[args.train:]
    print(f"{len(runs)} runs ({sum(len(r) for r in runs)} positions), "
          f"{len(train)} train / {len(test)} test, {L} layers")

    Rtr = np.concatenate(train).astype(np.float32)
    freq = Rtr.sum(axis=0)                              # [L,192]
    freq_mask = top8(freq)                              # [L,192] bool
    order = np.argsort(freq, axis=1)
    resident = np.zeros_like(freq_mask)
    np.put_along_axis(resident, order[:, -args.resident_k:], True, axis=1)
    nonres = ~resident

    def report(name, h, d, mh, md):
        print(f"  {name}: precision@8 {100*h/max(d,1):.1f}%  |  "
              f"non-resident-only {100*mh/max(md,1):.1f}% ({int(md)} miss slots)")

    for d in (1, 2, 4, 8, 16, 32):
        cooc = np.einsum("nli,nlj->lij", Rtr[:, : L - d, :], Rtr[:, d:, :])  # [L-d,192,192]
        stats = {"freq": np.zeros(4), "cooc": np.zeros(4)}
        for r in test:
            Rte = r.astype(np.float32)
            for l in range(L - d):
                actual = Rte[:, l + d, :]
                fm = np.broadcast_to(freq_mask[l + d], actual.shape)
                stats["freq"] += prec(fm, actual, nonres[l + d])
                cm = top8(Rte[:, l, :] @ cooc[l])
                stats["cooc"] += prec(cm, actual, nonres[l + d])
        print(f"axis A d={d} (lead ~{d*2.5:.0f} ms):")
        for k in ("freq", "cooc"):
            report(k, *stats[k])

    for lag in (1, 2, 4):
        stats = {"freq": np.zeros(4), "persist": np.zeros(4)}
        for r in test:
            Rte = r.astype(np.float32)
            src, tgt = Rte[:-lag], Rte[lag:]
            for l in range(L):
                actual = tgt[:, l, :]
                fm = np.broadcast_to(freq_mask[l], actual.shape)
                stats["freq"] += prec(fm, actual, nonres[l])
                stats["persist"] += prec(src[:, l, :] > 0, actual, nonres[l])
        print(f"axis B lag={lag} tokens (lead ~{lag*200} ms at m=1):")
        for k in ("freq", "persist"):
            report(k, *stats[k])


if __name__ == "__main__":
    main()
