#!/usr/bin/env python3
"""Summarize spec-serving matrix points + acceptance telemetry."""
import json, statistics, collections, re, glob

BASE = "/home/ubuntu/arc5/research/spec-serving-20260801"
pts = [json.loads(l) for l in open(f"{BASE}/points.jsonl")]
g = collections.defaultdict(list)
for p in pts:
    arm, _, c = p["label"].split("-")
    g[(arm, int(c[1:]))].append(p)

print(f'{"point":11s} {"agg_tok_s med":14s} {"n":3s} {"range":16s} {"p50_lat":9s} err')
for arm in ["plain", "spec"]:
    for c in [1, 2, 4, 8]:
        ok = [x for x in g[(arm, c)] if x["n_ok"] > 0]
        dead = [x for x in g[(arm, c)] if x["n_ok"] == 0]
        v = sorted(x["agg_tok_s"] for x in ok)
        l = statistics.median(x["lat_p50_s"] for x in ok)
        e = sum(x["n_err"] for x in g[(arm, c)])
        note = f"  ({len(dead)} dead point: {dead[0]['label']})" if dead else ""
        print(f"{arm}-c{c:<8d} {statistics.median(v):8.1f}     {len(ok)}  "
              f"{v[0]:6.1f}-{v[-1]:6.1f}   {l:6.2f}s   {e}{note}")
for c in [1, 2, 4, 8]:
    ps = statistics.median(x["agg_tok_s"] for x in g[("plain", c)] if x["n_ok"] > 0)
    ss = statistics.median(x["agg_tok_s"] for x in g[("spec", c)] if x["n_ok"] > 0)
    print(f"c={c}: spec/plain = {ss/ps:.2f}x")

d = a = n = 0
for f in glob.glob(f"{BASE}/server-spec-r*.log"):
    for l in open(f):
        m = re.search(r"burst=(\d+)/(\d+)", l)
        if m:
            a += int(m.group(1)); d += int(m.group(2)); n += 1
print(f"acceptance over all bursts: {a}/{d} = {a/d:.3f} (n_bursts={n})")
