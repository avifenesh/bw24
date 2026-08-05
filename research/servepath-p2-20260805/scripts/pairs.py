#!/usr/bin/env python3
"""Order-paired A/B analysis over serve-points.jsonl.

Every claim in this lane is an ORDER-PAIRED delta: arms run interleaved within a rep and the
rep order alternates, so a monotone thermal drift cancels in the pair mean. A raw
median-vs-median comparison across a drifting set is NOT evidence (the phase-1 lever-4
lesson), so this prints per-pair deltas, the pair mean, and the win count.

Usage: pairs.py <points.jsonl> <armA-substr> <armB-substr> [--key agg_tok_s] [--group max_tokens]
"""
import json
import statistics
import sys


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("{"):
                rows.append(json.loads(line))
    return rows


def rep_of(label):
    # labels end in -r<N>
    tail = label.rsplit("-r", 1)
    return tail[1] if len(tail) == 2 else "?"


def main():
    path, a_sub, b_sub = sys.argv[1], sys.argv[2], sys.argv[3]
    key = "agg_tok_s"
    group = "max_tokens"
    args = sys.argv[4:]
    for i, x in enumerate(args):
        if x == "--key":
            key = args[i + 1]
        if x == "--group":
            group = args[i + 1]

    rows = load(path)
    groups = sorted({r[group] for r in rows if a_sub in r["label"] or b_sub in r["label"]})
    print("%-8s | %-28s | %8s %8s | %7s | %s" % (
        group, "per-pair delta %", a_sub[:8], b_sub[:8], "pair", "wins"))
    print("-" * 88)
    for g in groups:
        a = {rep_of(r["label"]): r for r in rows
             if r[group] == g and a_sub in r["label"]}
        b = {rep_of(r["label"]): r for r in rows
             if r[group] == g and b_sub in r["label"]}
        reps = sorted(set(a) & set(b))
        if not reps:
            continue
        deltas = []
        for rp in reps:
            av, bv = a[rp][key], b[rp][key]
            if not av:
                continue
            deltas.append((bv - av) / av * 100.0)
        if not deltas:
            continue
        am = statistics.median([a[r][key] for r in reps])
        bm = statistics.median([b[r][key] for r in reps])
        wins = sum(1 for d in deltas if d > 0)
        print("%-8s | %-28s | %8.2f %8.2f | %+6.2f%% | %d/%d" % (
            g, " ".join("%+.2f" % d for d in deltas), am, bm,
            statistics.mean(deltas), wins, len(deltas)))


if __name__ == "__main__":
    main()
