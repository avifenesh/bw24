#!/usr/bin/env python3
"""Summarize logs/points.jsonl by arm: median agg_tok_s / lat_p50 over all passes.

Label convention: <art>-<arm>-r<round>-p<pass>; the arm key is everything before -r<N>.
Prints per-arm N, median, min..max, and the delta vs a --base arm if given.
"""
import json, sys, argparse, statistics, re
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--points", default="logs/points.jsonl")
ap.add_argument("--base", default=None, help="arm key to diff against")
ap.add_argument("--prefix", default=None, help="only arms starting with this")
a = ap.parse_args()

arms = defaultdict(list)
for line in open(a.points):
    d = json.loads(line)
    m = re.match(r"(.+)-r\d+-p\d+$", d["label"])
    key = m.group(1) if m else d["label"]
    if a.prefix and not key.startswith(a.prefix):
        continue
    if d.get("n_err", 0) or d.get("n_shed", 0):
        print(f"WARN {d['label']}: n_err={d['n_err']} n_shed={d['n_shed']}", file=sys.stderr)
    arms[key].append((d["agg_tok_s"], d["lat_p50_s"]))

base_med = None
if a.base and a.base in arms:
    base_med = statistics.median(x[0] for x in arms[a.base])

for key in sorted(arms):
    toks = sorted(x[0] for x in arms[key])
    lats = sorted(x[1] for x in arms[key])
    med = statistics.median(toks)
    lmed = statistics.median(lats)
    delta = f"  {100*(med/base_med-1):+.1f}% vs {a.base}" if base_med and key != a.base else ""
    print(f"{key:24s} N={len(toks):2d}  med={med:7.1f} tok/s  [{toks[0]:.1f}..{toks[-1]:.1f}]  "
          f"p50={lmed:.3f}s{delta}")
