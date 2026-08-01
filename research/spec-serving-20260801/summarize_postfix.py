#!/usr/bin/env python3
"""Summarize postfix A/B points: spec-pre vs spec-post vs plain-post + burst sweep."""
import json, statistics, collections, re, glob

BASE = "/home/ubuntu/arc5/research/spec-serving-20260801"
pts = [json.loads(l) for l in open(f"{BASE}/postfix-points.jsonl")]
g = collections.defaultdict(list)
for p in pts:
    arm, _, c = p["label"].split("-")
    g[(arm, int(c[1:]))].append(p)

def med(arm, c, field="agg_tok_s"):
    ok = [x for x in g[(arm, c)] if x["n_ok"] > 0]
    if not ok:
        return None
    return statistics.median(x[field] for x in ok)

print(f'{"point":16s} {"agg med":9s} {"n":2s} {"range":16s} {"p50_lat":8s} err')
for arm in ["specpre", "specpost", "plainpost", "burstpost8", "burstpost32", "burstpost128"]:
    for c in [1, 2, 4, 8]:
        ok = [x for x in g[(arm, c)] if x["n_ok"] > 0]
        if not ok:
            continue
        v = sorted(x["agg_tok_s"] for x in ok)
        l = statistics.median(x["lat_p50_s"] for x in ok)
        e = sum(x["n_err"] for x in g[(arm, c)])
        print(f"{arm}-c{c:<8d} {statistics.median(v):8.1f} {len(ok):2d} "
              f"{v[0]:6.1f}-{v[-1]:6.1f}   {l:6.2f}s  {e}")
print()
for c in [1, 4, 8]:
    pre, post = med("specpre", c), med("specpost", c)
    if pre and post:
        print(f"c={c}: spec post/pre = {post/pre:.3f}x  ({pre:.1f} -> {post:.1f})")
print()
for c in [1, 2, 4, 8]:
    ps, ss = med("plainpost", c), med("specpost", c)
    if ps and ss:
        print(f"c={c}: spec-post/plain-post = {ss/ps:.2f}x  ({ss:.1f} vs {ps:.1f})")
print()
for b in [8, 32, 128]:
    for c in [4, 8]:
        v = med(f"burstpost{b}", c)
        if v:
            print(f"burst={b} c={c}: {v:.1f} tok/s (post-fix)")

d = a = n = 0
for f in glob.glob(f"{BASE}/server-specpost-r*.log"):
    for l in open(f):
        m = re.search(r"burst=(\d+)/(\d+)", l)
        if m:
            a += int(m.group(1)); d += int(m.group(2)); n += 1
if d:
    print(f"\npost-fix acceptance: {a}/{d} = {a/d:.3f} (n_bursts={n})")
