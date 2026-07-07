#!/usr/bin/env python3
"""Analyze a BW24_MOE_TRACE file: decode working set + step-to-step expert reuse.

Trace line format: "<layer> <t> <id,id,...>" — one line per (layer, forward call).
Decode steps are t==1 lines; a "step" is one full pass over all MoE layers.

Outputs the go/no-go numbers for resident-expert tiering:
  - per-layer unique experts touched across all decode steps (working set)
  - step-to-step reuse rate (fraction of (layer,expert) picks already picked at step-1)
  - cumulative-coverage curve (how fast the working set saturates)
  - global frequency skew (top-N% experts take what fraction of picks)
"""
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/m3-route-trace.txt"

# decode-only picks, grouped into steps by layer wrap-around
steps = []           # list of dict: layer -> set(expert ids)
cur = {}
prev_layer = -1
freq = defaultdict(int)          # (layer, expert) -> count
layer_ws = defaultdict(set)      # layer -> set of experts ever used (decode)

for line in open(path):
    parts = line.split()
    if len(parts) != 3:
        continue
    il, t = int(parts[0]), int(parts[1])
    if t != 1:
        continue
    ids = [int(x) for x in parts[2].split(",")]
    if il <= prev_layer and cur:
        steps.append(cur)
        cur = {}
    prev_layer = il
    cur[il] = set(ids)
    layer_ws[il].update(ids)
    for e in ids:
        freq[(il, e)] += 1
if cur:
    steps.append(cur)

n_steps = len(steps)
if n_steps < 2:
    sys.exit(f"need >=2 decode steps, got {n_steps}")

layers = sorted(layer_ws)
n_used = len(next(iter(steps[0].values())))

# step-to-step reuse
reuse_hits = reuse_total = 0
for i in range(1, n_steps):
    for il, sel in steps[i].items():
        prev = steps[i - 1].get(il, set())
        reuse_hits += len(sel & prev)
        reuse_total += len(sel)

# cumulative coverage curve
cum = defaultdict(set)
curve = []
for i, st in enumerate(steps):
    for il, sel in st.items():
        cum[il] |= sel
    curve.append(sum(len(v) for v in cum.values()))

total_ws = sum(len(v) for v in layer_ws.values())
picks = sorted(freq.values(), reverse=True)
tot_picks = sum(picks)


def topfrac(f):
    k = max(1, int(len(picks) * f))
    return sum(picks[:k]) / tot_picks


ws_sizes = sorted(len(layer_ws[il]) for il in layers)
print(f"decode steps: {n_steps}  layers: {len(layers)}  topk: {n_used}")
print(f"working set: {total_ws} (layer,expert) pairs "
      f"(per-layer min/med/max = {ws_sizes[0]}/{ws_sizes[len(ws_sizes)//2]}/{ws_sizes[-1]} of 64)")
print(f"step-to-step reuse: {reuse_hits}/{reuse_total} = {reuse_hits/reuse_total:.1%}")
print(f"freq skew: top 10% pairs take {topfrac(0.10):.1%} of picks, "
      f"top 25% take {topfrac(0.25):.1%}, top 50% take {topfrac(0.50):.1%}")
print("coverage curve (unique pairs after step k):")
marks = sorted({1, 2, 4, 8, 16, 32, n_steps} & set(range(1, n_steps + 1)))
for k in marks:
    print(f"  step {k:3d}: {curve[k-1]:5d} pairs  ({curve[k-1]/(len(layers)*64):.1%} of all experts)")
