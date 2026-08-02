#!/usr/bin/env python3
"""Per-replica p50/p99 from perreq jsonl files: usage: analyze-perreq.py <dir>"""
import json, sys, glob, re, statistics
from collections import defaultdict

def pct(lats, p):
    if not lats: return None
    ls = sorted(lats)
    k = min(len(ls)-1, max(0, int(round(p/100*(len(ls)-1)))))
    return ls[k]

cells = defaultdict(list)   # cell key -> all latencies (ok rows)
errs = defaultdict(int)
for f in glob.glob(sys.argv[1] + "/*.jsonl"):
    for l in open(f):
        r = json.loads(l)
        key = re.sub(r"-p\d+(-r\d+)?$", "", r["label"])
        if r["ok"]:
            cells[key].append(r["latency_s"])
        else:
            errs[key] += 1

print(f"{'cell':28} {'n_ok':>5} {'p50':>7} {'p95':>7} {'p99':>7} {'errs':>4}")
for key in sorted(cells):
    ls = cells[key]
    print(f"{key:28} {len(ls):>5} {pct(ls,50):>7.3f} {pct(ls,95):>7.3f} {pct(ls,99):>7.3f} {errs[key]:>4}")
