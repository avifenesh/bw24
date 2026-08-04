#!/usr/bin/env python3
"""Reduce the 27B pp battery's per-rep logs to per-arm medians + the RESULTS rows.

Reads the committed raw logs (pp<len>-<arm>-r<rep>.log) rather than the driver summary, so the
number in the table and the number in the receipts come from the same bytes. Each log carries its
own in-process median over MEMRA_PP_REPS=3; this takes the median ACROSS the N process reps of
those, and reports N explicitly (the repo rule: every published median states its N).

usage: summarize.py [dir]      (default: the directory holding this script)
"""
import re
import statistics
import sys
from pathlib import Path

d = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent)
pat = re.compile(r"pp-only MEDIAN: (\d+) tok in ([0-9.]+)s = ([0-9.]+) tok/s")

runs: dict[tuple[str, str], list[tuple[int, float]]] = {}
for f in sorted(d.glob("pp*-*-r*.log")):
    m = re.match(r"pp(\d+)-(\w+)-r(\d+)\.log", f.name)
    if not m:
        continue
    ln, arm, rep = m.group(1), m.group(2), int(m.group(3))
    hit = pat.search(f.read_text(errors="replace"))
    if not hit:
        print(f"WARN {f.name}: no median line (run incomplete or died)")
        continue
    runs.setdefault((ln, arm), []).append((rep, float(hit.group(3))))

for ln in sorted({k[0] for k in runs}, key=int):
    med = {}
    for arm in ("floor", "mmq", "arma"):
        v = runs.get((ln, arm))
        if not v:
            continue
        toks = [t for _, t in sorted(v)]
        med[arm] = (statistics.median(toks), len(toks), toks)
    if "floor" not in med:
        continue
    base = med["floor"][0]
    print(f"\n== pp{ln} ==")
    for arm, (t, n, toks) in med.items():
        label = "single run" if n == 1 else f"median of N={n}"
        print(f"  {arm:<6} {t:8.1f} tok/s  {t / base:6.4f}x floor   ({label}: {toks})")
