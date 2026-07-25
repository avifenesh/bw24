#!/usr/bin/env python3
"""Importance-guided demotion candidate: rank the served plan's non-Q2_K tail by ROUTED MASS
over the frozen private calibration corpus, and emit demote-set candidates at conservative
mass slices.

Why mass, not counts: the failed blanket demotion (tailq2k, code 8/14 vs 11/14) sent every
tail pair to Q2_K regardless of how much routing weight it carries. Aggregate routing weight
is the quality-relevant signal — an expert selected rarely but with high weight shapes the
output more than a frequent low-weight one. Ranking uses ONLY the private calibration traces
(rules: public eval data never selects experts or precision).

Inputs:
  --traces DIR    weights-*.trace files (`il t ex:w,ex:w,...` lines, BW24_MOE_WEIGHT_TRACE)
  --manifest F    served runtime manifest (plan assignments = qtype per (layer,proj,expert))
Outputs (next to --out prefix):
  <out>-mass.json          per-(layer,expert) mass table + coverage curve
  <out>-demote-p<N>.json   demote sets: tail pairs whose cumulative mass share of the TAIL
                           is below N% (N in --slices), format identical to
                           tail-q2k-demote-set.json ([[layer, expert], ...])
"""
import argparse, glob, json, os
from collections import defaultdict

SB = {"Q2_K": 84, "Q3_K": 110, "IQ3_S": 110, "IQ4_XS": 136, "Q4_K": 144, "Q8_0": 272}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--slices", default="10,25,50")
    args = ap.parse_args()

    mass = defaultdict(float)
    hits = defaultdict(int)
    files = sorted(glob.glob(os.path.join(args.traces, "weights-*.trace")))
    if not files:
        raise SystemExit("no weight traces found")
    for path in files:
        for line in open(path):
            layer_s, _t, pairs = line.split(" ", 2)
            layer = int(layer_s)
            for pair in pairs.strip().split(","):
                ex_s, w_s = pair.split(":")
                key = (layer, int(ex_s))
                mass[key] += float(w_s)
                hits[key] += 1
    print(f"traces: {len(files)} files, {len(mass)} routed (layer,expert) pairs")

    plan = json.load(open(args.manifest))["plan"]["assignments"]
    # A pair's tier = the max-bytes qtype across its three projections (demotion targets the
    # whole expert, mirroring the surgery tool's unit).
    pair_qtype = {}
    for a in plan:
        for e in a["experts"]:
            key = (a["layer"], e)
            q = a["qtype"]
            if key not in pair_qtype or SB[q] > SB[pair_qtype[key]]:
                pair_qtype[key] = q
    tail = [k for k, q in pair_qtype.items() if q != "Q2_K"]
    print(f"plan: {len(pair_qtype)} retained pairs, tail (non-Q2_K): {len(tail)}")

    # Rank tail ascending by calibration mass. Pairs never routed on calibration get mass 0
    # and sort first — exactly the ones calibration says the outputs never lean on.
    ranked = sorted(tail, key=lambda k: (mass.get(k, 0.0), hits.get(k, 0)))
    total_tail_mass = sum(mass.get(k, 0.0) for k in tail) or 1.0

    curve = []
    cum = 0.0
    for i, k in enumerate(ranked):
        cum += mass.get(k, 0.0)
        curve.append((i + 1, cum / total_tail_mass))

    json.dump(
        {
            "format": "bw24-tail-routed-mass-v1",
            "trace_files": len(files),
            "tail_pairs": len(tail),
            "total_tail_mass": total_tail_mass,
            "mass": {f"{l}:{e}": mass.get((l, e), 0.0) for (l, e) in tail},
            "hits": {f"{l}:{e}": hits.get((l, e), 0) for (l, e) in tail},
            "coverage_curve_every_100": curve[::100],
        },
        open(f"{args.out}-mass.json", "w"),
    )

    for pct in [int(x) for x in args.slices.split(",")]:
        cutoff = pct / 100.0
        chosen = [list(k) for i, k in enumerate(ranked) if curve[i][1] <= cutoff]
        json.dump(chosen, open(f"{args.out}-demote-p{pct}.json", "w"))
        bytes_frac = len(chosen) / max(len(pair_qtype), 1)
        print(
            f"slice p{pct}: demote {len(chosen)}/{len(tail)} tail pairs "
            f"({100*len(chosen)/len(tail):.0f}% of tail count) carrying <= {pct}% of tail mass; "
            f"~{100*bytes_frac:.1f}% of all retained pairs"
        )


if __name__ == "__main__":
    main()
