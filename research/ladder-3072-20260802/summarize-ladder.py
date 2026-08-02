#!/usr/bin/env python3
"""ladder-3072: medians per (model, arm, depth) from ladder-sweep.jsonl (tg128_toks,
non-quarantined only), plus best-arm and vs-llama columns."""
import json, sys, statistics as st
from collections import defaultdict

rows = defaultdict(list)
temps = defaultdict(list)
for line in open(sys.argv[1] if len(sys.argv) > 1 else
                 "/home/avifenesh/projects/wt-ladder-3072/research/ladder-3072-20260802/ladder-sweep.jsonl"):
    r = json.loads(line)
    if r.get("metric") != "tg128_toks" or r.get("value") in (None, "null"): continue
    if r.get("quarantined"): continue
    key = (r["model"], str(r["arm"]), r["depth"])
    rows[key].append(float(r["value"]))
    temps[key].append(r.get("temp_c"))

models = sorted({k[0] for k in rows})
depths = sorted({k[2] for k in rows})
arms = ["8", "32", "64", "llama"]
for m in models:
    print(f"\n=== {m} (tg128 tok/s, median (N) [reps]) ===")
    hdr = "arm      " + "".join(f"d{d:<12}" for d in depths)
    print(hdr)
    for a in arms:
        line = f"sp{a:<6} " if a != "llama" else "llama   "
        for d in depths:
            v = rows.get((m, a, d), [])
            if v: line += f"{st.median(v):7.2f} ({len(v)})  "
            else: line += "      -       "
        print(line)
    # ratios vs sp8 (current-at-depth<=3072) and vs llama
    for a in ["32", "64"]:
        line = f"sp{a}/sp8 "
        for d in depths:
            v, b = rows.get((m, a, d), []), rows.get((m, "8", d), [])
            line += f"{st.median(v)/st.median(b):7.3f}x     " if v and b else "      -       "
        print(line)
    line = "best/llama"
    for d in depths:
        l = rows.get((m, "llama", d), [])
        best = max((st.median(rows[(m, a, d)]) for a in ["8","32","64"] if rows.get((m,a,d))), default=None)
        line += f"{best/st.median(l):7.3f}x     " if l and best else "      -       "
    print(line)
    trange = [t for k, ts in temps.items() if k[0] == m and k[1] != "llama" for t in ts if t is not None]
    if trange: print(f"temps {min(trange)}-{max(trange)}C")
