#!/usr/bin/env python3
"""Boundary-cost medians from the interleaved x5 run-gen arms (receipts/m1-pp2/boundary)."""
import re, statistics as st
from pathlib import Path

here = Path(__file__).parent
bdir = here / "receipts/m1-pp2/boundary"
arms = {"naked": "naked-r3-p*.log", "pp2-samedev": "pp2-samedev-r3-p*.log",
        "pp2-dev01": "pp2-dev01-r3-p*.log"}
meds = {}
for arm, pat in arms.items():
    rates = []
    for f in sorted(bdir.glob(pat)):
        m = re.search(r"= ([\d.]+) tok/s \(Stage-B", f.read_text())
        if m: rates.append(float(m.group(1)))
    meds[arm] = (st.median(rates), rates)
    print(f"{arm:12s} median={st.median(rates):7.2f} tok/s  N={len(rates)}  runs={sorted(rates)}")
base = meds["naked"][0]
for arm in ("pp2-samedev", "pp2-dev01"):
    d = (base - meds[arm][0]) / base * 100
    tick_us = (1/meds[arm][0] - 1/base) * 1e6
    print(f"{arm}: per-tick boundary cost = {d:+.2f}% ({tick_us:+.1f} us/tick vs naked {1e3/base:.3f} ms/tick)")
