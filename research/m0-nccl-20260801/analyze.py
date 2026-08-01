#!/usr/bin/env python3
"""Regenerate the m0-nccl summary tables from receipts/all.jsonl (+ cudarc rows)."""
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

here = Path(__file__).parent
rows = []
for f in [here / "receipts/all.jsonl", here / "receipts/cudarc_pp_0_3.jsonl"]:
    rows += [json.loads(l) for l in open(f)]

g = defaultdict(list)
for r in rows:
    if r["test"] == "pingpong":
        k = ("pp", r["impl"], f"{r['devA']}-{r['devB']}", r["size_bytes"])
        g[k].append((r["lat_us_oneway"], r["bw_GBps_oneway"]))
    elif r["test"] == "uni":
        k = ("uni", r["impl"], r["dir"], r["size_bytes"])
        g[k].append((r["per_iter_us"], r["bw_GBps"]))
    elif r["test"] == "bidir":
        k = ("bidir", r["impl"], f"{r['devA']}-{r['devB']}", r["size_bytes"])
        g[k].append((r["per_iter_us"], r["bw_GBps_aggregate"]))
    elif r["test"] == "alltoall":
        k = ("a2a", r["impl"], str(r["devs"]), r["size_bytes_per_peer"])
        g[k].append((r["per_a2a_us"], r["agg_bw_GBps"]))


def fmt(b):
    return f"{b//1024}KB" if b < 1 << 20 else f"{b//(1<<20)}MB"


out = sys.stdout
for test in ["pp", "uni", "bidir", "a2a"]:
    out.write(f"\n=== {test} ===\n")
    for k in sorted([k for k in g if k[0] == test], key=lambda k: (k[1], k[2], k[3])):
        lat = [v[0] for v in g[k]]
        bw = [v[1] for v in g[k]]
        out.write(
            f"{k[1]:12s} {k[2]:12s} {fmt(k[3]):>6s}  "
            f"lat/iter med={st.median(lat):9.3f}us min={min(lat):9.3f} max={max(lat):9.3f}  "
            f"bw med={st.median(bw):8.2f} GB/s\n"
        )
