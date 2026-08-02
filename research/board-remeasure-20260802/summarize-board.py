#!/usr/bin/env python3
"""Summarize board-remeasure.jsonl: per cell (model x depth x engine) median, N, spread,
temp range; old-board comparison + moved-beyond-noise verdict.

Noise bound: per-cell relative spread (max-min)/median of the N=5 reps; a cell "moved" iff
|new_median - old| / old > max(own-arm spread, paired-arm spread) for that cell.
"""
import json, sys, statistics
from collections import defaultdict

R = "/home/avifenesh/projects/wt-board-remeasure/research/board-remeasure-20260802/board-remeasure.jsonl"
OLD = {  # published board values (current-board.json @ d14d7d8d)
    ("q9", 512, "memra"): 135.7, ("q9", 512, "llama"): 126.7,
    ("q27", 512, "memra"): 48.4, ("q27", 512, "llama"): 44.9,
    ("q35", 512, "memra"): 178.2, ("q35", 512, "llama"): 167.8,
    ("q9", 6257, "memra"): 127.2, ("q9", 6257, "llama"): 120.0,
    ("q27", 6257, "memra"): 45.9, ("q27", 6257, "llama"): 43.0,
    ("q35", 6257, "memra"): 163.7, ("q35", 6257, "llama"): 160.9,
}

vals = defaultdict(list)
temps = defaultdict(list)
gates = defaultdict(list)
for line in open(R):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    key = (r["model"], r["depth"], r["engine"])
    if r["metric"] == "tg128_toks" and r["value"] is not None:
        vals[key].append(float(r["value"]))
        temps[key].append(r.get("temp_c"))
    if r["metric"] == "argmax_match_lines":
        gates[key].append(int(r["value"]))

print(f"{'cell':28} {'N':>2} {'median':>8} {'min':>8} {'max':>8} {'spread%':>8} {'old':>7} {'delta%':>7} {'temps':>9}")
spreads = {}
for key in sorted(vals, key=lambda k: (k[1], k[0], k[2])):
    v = sorted(vals[key])
    med = statistics.median(v)
    spread = (v[-1] - v[0]) / med * 100
    spreads[key] = spread
    old = OLD.get(key)
    delta = (med - old) / old * 100 if old else float("nan")
    t = [x for x in temps[key] if x is not None]
    trange = f"{min(t)}-{max(t)}C" if t else "?"
    name = f"{key[0]} d{key[1]} {key[2]}"
    print(f"{name:28} {len(v):>2} {med:>8.1f} {v[0]:>8.1f} {v[-1]:>8.1f} {spread:>7.1f}% {old:>7} {delta:>+6.1f}% {trange:>9}")

print("\nverdicts (moved iff |delta| > max(memra spread, llama spread) for the pair):")
for model in ("q9", "q27", "q35"):
    for depth in (512, 6257):
        pair_noise = max(spreads.get((model, depth, "memra"), 0), spreads.get((model, depth, "llama"), 0))
        for eng in ("memra", "llama"):
            key = (model, depth, eng)
            if key not in vals:
                continue
            med = statistics.median(vals[key])
            old = OLD[key]
            delta = (med - old) / old * 100
            moved = abs(delta) > pair_noise
            print(f"  {model} d{depth} {eng:6}: old {old:>6} -> new {med:>7.1f} ({delta:+.1f}%), pair-noise {pair_noise:.1f}% -> {'MOVED' if moved else 'within noise'}")

print("\nargmax gate lines per memra cell (must be >=2 MATCH per run, 0 MISMATCH):")
for key in sorted(gates):
    print(f"  {key}: {gates[key]}")
