#!/usr/bin/env python3
"""Fuse the plain-arm routed-mass ranking with the frozen G7e big-corpus tier classes into a
byte-bounded tier-plan candidate. No pruning (owner directive: prune arms showed bad results).

Evidence fusion (conservative by construction):
  - promote to Q8_0 only pairs HOT in both sources (prior class Q8_0/Q4_K AND local top slice)
  - demote to Q2_K only pairs COLD in both sources (prior Q2_K-or-pruned AND local bottom slice)
  - every disagreement stays at the uniform Q3_K baseline
Per-layer byte bound: candidate bytes <= uniform Q3_K bytes, enforced via
  promotions <= floor(demotions * (B3-B2)/(B8-B3)) = floor(demotions * 26/162).

Inputs: --mass (rank_tail_by_routed_mass.py output), --prior (frozen G7e plan), --out plan path.
Ranking sources: local mass ranks per layer; prior classes from big-corpus traffic ranking
(192 requests / 103M routed assignments). Public eval data enters nowhere.
"""
import argparse, json
from collections import defaultdict

PRIOR_CLASS = {"Q8_0": 5, "Q4_K": 4, "IQ4_XS": 3, "IQ3_S": 2, "Q3_K": 1, "Q2_K": 0}
ROW_BYTES = {"Q8_0": 272, "Q3_K": 110, "Q2_K": 84}  # per 256 weights
N_EXPERT = 192


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mass", required=True)
    ap.add_argument("--prior", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hot-local-frac", type=float, default=0.10)
    ap.add_argument("--cold-local-frac", type=float, default=0.40)
    ap.add_argument("--max-demote-per-layer", type=int, default=64)
    args = ap.parse_args()

    m = json.load(open(args.mass))
    mass = {tuple(map(int, k.split(":"))): v for k, v in m["mass"].items()}

    prior = json.load(open(args.prior))
    prior_class = {}
    for a in prior["assignments"]:
        for e in a["experts"]:
            key = (a["layer"], e)
            c = PRIOR_CLASS[a["qtype"]]
            if key not in prior_class or c > prior_class[key]:
                prior_class[key] = c
    for layer_s, ids in prior.get("pruned_experts", {}).items():
        for e in ids:
            prior_class[(int(layer_s), e)] = -1  # big corpus called these coldest

    layers = sorted({l for (l, _e) in mass})
    tiers = {}
    stats = defaultdict(int)
    for layer in layers:
        ranked = sorted(range(N_EXPERT), key=lambda e: mass.get((layer, e), 0.0))
        n_cold = int(N_EXPERT * args.cold_local_frac)
        n_hot = int(N_EXPERT * args.hot_local_frac)
        cold_local = set(ranked[:n_cold])
        hot_local = set(ranked[-n_hot:])
        demote = [
            e for e in ranked
            if e in cold_local and prior_class.get((layer, e), 1) <= 0
        ][: args.max_demote_per_layer]
        budget = int(len(demote) * (ROW_BYTES["Q3_K"] - ROW_BYTES["Q2_K"])
                     / (ROW_BYTES["Q8_0"] - ROW_BYTES["Q3_K"]))
        promote = [
            e for e in reversed(ranked)
            if e in hot_local and prior_class.get((layer, e), 1) >= 4
        ][:budget]
        for e in range(N_EXPERT):
            tiers[(layer, e)] = ("Q2_K" if e in set(demote)
                                 else "Q8_0" if e in set(promote)
                                 else "Q3_K")
        stats["demoted"] += len(demote)
        stats["promoted"] += len(promote)

    groups = defaultdict(list)
    for (layer, e), q in tiers.items():
        groups[(layer, q)].append(e)
    plan = {
        "format": "bw24-expert-tier-plan-v2",
        "description": "importance-fused candidate: plain-arm routed mass x G7e big-corpus prior; "
                       "no pruning; byte-bounded to the uniform Q3_K baseline per layer",
        "recipe": "fused-mass-prior-q8-q3-q2-noprune",
        "model": {"expert_count": N_EXPERT, "expert_used_count": 8,
                  "moe_layers": layers, "original_expert_count": N_EXPERT},
        "pruned_experts": {},
        "assignments": [
            {"layer": layer, "experts": sorted(es), "projections": ["gate", "up", "down"],
             "qtype": q}
            for (layer, q), es in sorted(groups.items())
        ],
        "calibration": {
            "provenance": {
                "local_mass": {"traces": m["trace_files"], "corpus": "private-24"},
                "prior_plan": args.prior,
            }
        },
    }
    json.dump(plan, open(args.out, "w"), indent=1)
    total = len(tiers)
    b_uni = total * ROW_BYTES["Q3_K"]
    b_new = sum(ROW_BYTES[q] for q in tiers.values())
    print(f"pairs={total} demoted={stats['demoted']} promoted={stats['promoted']} "
          f"bytes vs uniform: {100 * b_new / b_uni:.2f}%")


if __name__ == "__main__":
    main()
