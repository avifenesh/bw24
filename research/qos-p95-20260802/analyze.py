#!/usr/bin/env python3
"""QoS p95 A/B analysis — summarize points.jsonl per condition (median of N=3 passes)."""
import json
import statistics
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "points.jsonl"
rows = [json.loads(l) for l in open(path) if l.strip()]

# label shape: p{pass}-{cond}-{int-alone|bulk|int-cont}
by = defaultdict(list)
for r in rows:
    lbl = r["label"]  # e.g. p1-off8-int-alone
    parts = lbl.split("-")
    p, cond, kind = parts[0], parts[1], "-".join(parts[2:])
    by[(cond, kind)].append((p, r))

def med(vals):
    return statistics.median(vals)

conds = ["off8", "on8", "off16", "on16"]
kinds = ["int-alone", "int-cont", "bulk"]
print(f"{'cond':7} {'kind':10} {'agg tok/s (per-pass)':34} {'p50 s (per-pass)':28} "
      f"{'p95 s (per-pass)':28} {'shed':>6} {'err':>4}")
for cond in conds:
    for kind in kinds:
        rs = sorted(by.get((cond, kind), []))
        if not rs:
            continue
        aggs = [r["agg_tok_s"] for _, r in rs]
        p50s = [r["lat_p50_s"] for _, r in rs]
        p95s = [r["lat_p95_s"] for _, r in rs]
        sheds = sum(r.get("n_shed", 0) for _, r in rs)
        errs = sum(r.get("n_err", 0) for _, r in rs)
        print(f"{cond:7} {kind:10} "
              f"{med(aggs):7.1f} ({'/'.join(f'{a:.0f}' for a in aggs)})".ljust(43) +
              f"{med(p50s):6.3f} ({'/'.join(f'{v:.3f}' for v in p50s)})".ljust(29) +
              f"{med(p95s):6.3f} ({'/'.join(f'{v:.3f}' for v in p95s)})".ljust(29) +
              f"{sheds:>6} {errs:>4}")

# combined-tenant aggregate throughput per condition (bulk + contended interactive, per pass)
print("\ncombined aggregate (bulk + int-cont), per pass then median:")
for cond in conds:
    bulk = dict(by.get((cond, "bulk"), []))
    cont = dict(by.get((cond, "int-cont"), []))
    tots = []
    for p in sorted(set(bulk) & set(cont)):
        tots.append(bulk[p]["agg_tok_s"] + cont[p]["agg_tok_s"])
    if tots:
        print(f"  {cond:7} {med(tots):8.1f}  ({'/'.join(f'{t:.0f}' for t in tots)})")
