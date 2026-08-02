#!/usr/bin/env python3
"""fa-decode-deep depth A/B summarizer: medians per (model, arm, depth) from depth-ab.jsonl,
old-vs-new table + vs-llama restatement + within-arm decay. N stated per cell."""
import json, statistics, sys, collections

path = sys.argv[1] if len(sys.argv) > 1 else "depth-ab.jsonl"
cells = collections.defaultdict(list)
for line in open(path):
    r = json.loads(line)
    if r.get("cell") != "fa-deep-depth-ab" or r.get("metric") != "tg128_toks":
        continue
    if r["value"] is None:
        continue
    cells[(r["model"], r["arm"], r["depth"])].append(float(r["value"]))

models = ["kat", "q35", "o35b"]
depths = [512, 2048, 4096, 6144]

def med(m, a, d):
    v = cells.get((m, a, d), [])
    return (statistics.median(v), len(v)) if v else (None, 0)

print("depth A/B (tg128 tok/s, medians; per-cell N in parens)")
hdr = "| model | arm | " + " | ".join(f"d{d}" for d in depths) + " | decay 512->6144 |"
print(hdr); print("|---" * (len(depths) + 3) + "|")
for m in models:
    for arm in ["old", "new", "llama"]:
        vals = [med(m, arm, d) for d in depths]
        if all(v[0] is None for v in vals):
            continue
        cellstr = " | ".join(f"{v:.1f} ({n})" if v is not None else "-" for v, n in vals)
        v0, v3 = vals[0][0], vals[-1][0]
        decay = f"{(v3 / v0 - 1) * 100:+.1f}%" if v0 and v3 else "-"
        print(f"| {m} | {arm} | {cellstr} | {decay} |")
    # ratio rows
    for num, den, name in [("new", "old", "new/old"), ("new", "llama", "new/llama"),
                           ("old", "llama", "old/llama")]:
        ratios = []
        for d in depths:
            a, _ = med(m, num, d); b, _ = med(m, den, d)
            ratios.append(f"{a/b:.3f}x" if a and b else "-")
        print(f"| {m} | {name} | " + " | ".join(ratios) + " | |")
print()
print("absolute depth cost (ms/token added 512->6144):")
for m in models:
    for arm in ["old", "new", "llama"]:
        v0, _ = med(m, arm, 512); v3, _ = med(m, arm, 6144)
        if v0 and v3:
            print(f"  {m}/{arm}: +{(1000/v3 - 1000/v0):.3f} ms")
