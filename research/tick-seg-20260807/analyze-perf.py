#!/usr/bin/env python3
"""lane/tick-seg: parse perf-tickseg raw log -> per-cell interleaved medians + delta.

Each rep prints 'ppprime MEDIAN: <T> tok in <t>s = <tok/s> tok/s (budget=...)'; arms alternate
AFTER/BEFORE inside one lock hold. N per arm = 5 medians (each itself median-of-3 timed reps).
Delta = (median_after - median_before) / median_before on the times (negative = AFTER faster).
"""
import re
import statistics
import sys

log = open(sys.argv[1], encoding="utf-8", errors="replace").read()
cells = re.split(r"#{10} CELL ", log)[1:]
print(f"{'cell':34} | {'arm':6} | N | median s | tok/s   | spread% | delta%")
for c in cells:
    header = c.split(" ", 3)
    name = c.split(":")[0].strip()
    arms = {"AFTER": [], "BEFORE": []}
    cur = None
    for line in c.splitlines():
        m = re.match(r"--- rep \d+ (AFTER|BEFORE)", line)
        if m:
            cur = m.group(1)
            continue
        m = re.search(r"ppprime MEDIAN: (\d+) tok in ([0-9.]+)s = ([0-9.]+) tok/s", line)
        if m and cur:
            arms[cur].append((float(m.group(2)), float(m.group(3))))
            cur = None
    if not arms["AFTER"] or not arms["BEFORE"]:
        continue
    med = {}
    for k, v in arms.items():
        ts = sorted(t for t, _ in v)
        med[k] = statistics.median(ts)
        spread = (max(ts) - min(ts)) / med[k] * 100
        toks = statistics.median(sorted(r for _, r in v))
        n = len(ts)
        delta = ""
        if k == "AFTER":
            pass
        med[k + "_str"] = (n, med[k], toks, spread)
    delta = (med["AFTER"] - med["BEFORE"]) / med["BEFORE"] * 100
    for k in ("AFTER", "BEFORE"):
        n, m_, toks, spread = med[k + "_str"]
        d = f"{delta:+.3f}" if k == "AFTER" else ""
        print(f"{name:34} | {k:6} | {n} | {m_:.4f}  | {toks:7.2f} | {spread:6.3f} | {d}")
