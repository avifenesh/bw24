#!/usr/bin/env python3
"""Analyze fleet-resweep points.jsonl -> per-cell medians (N per cell)."""
import json, sys, statistics, re
from collections import defaultdict

rows = [json.loads(l) for l in open(sys.argv[1])]
cells = defaultdict(list)
for r in rows:
    lab = r["label"]
    # strip pass + replica suffix: f8-c8pr-p2-r9091 -> f8-c8pr ; f1-c16-p3 -> f1-c16
    key = re.sub(r"-p\d+(-r\d+)?$", "", lab)
    cells[key].append(r)

# fleet direct cells: aggregate per pass = sum over replicas within the pass
def agg_by_pass(rs):
    per_pass = defaultdict(list)
    for r in rs:
        m = re.search(r"-p(\d+)", r["label"])
        per_pass[int(m.group(1))].append(r)
    return per_pass

print(f"{'cell':28} {'N':>2} {'agg tok/s med':>13} {'per-pass aggs':>36} {'p50 med':>8} {'p99~max med':>11} {'errs':>4}")
for key in sorted(cells):
    rs = cells[key]
    pp = agg_by_pass(rs)
    aggs, p50s, p95s, maxs, errs = [], [], [], [], 0
    for p, prs in sorted(pp.items()):
        aggs.append(sum(r["agg_tok_s"] for r in prs))
        p50s.extend(r["lat_p50_s"] for r in prs)
        p95s.extend(r["lat_p95_s"] for r in prs)
        maxs.extend(r["lat_max_s"] for r in prs if r["lat_max_s"])
        errs += sum(r["n_err"] for r in prs)
    med = statistics.median(aggs)
    print(f"{key:28} {len(pp):>2} {med:>13.1f} {str([round(a,1) for a in aggs]):>36} "
          f"{statistics.median(p50s):>8.3f} {statistics.median(maxs):>11.3f} {errs:>4}")
