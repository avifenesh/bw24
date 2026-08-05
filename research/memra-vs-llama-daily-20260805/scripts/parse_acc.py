#!/usr/bin/env python3
"""Parse memra server logs: per-request final spec acceptance, attributed to arms by
request order (driver order is fixed: warmup, then 4 arms x 3 cells)."""
import re, statistics, collections

RDIR = "/home/avifenesh/projects/bw24/research/memra-vs-llama-daily-20260805"
ORDER = ["warmup"] + [f"{arm}/{cell}"
         for arm in ("memra-t0.8", "memra-t0.8-lsampler", "memra-t1.0", "memra-greedy")
         for cell in ("short-agentic", "long-gen", "ctx4k")]

acc = collections.defaultdict(list)
for rep in range(1, 6):
    reqs = []   # list of final (num, den) per request
    cur = None
    for line in open(f"{RDIR}/logs/server-memra-r{rep}.log"):
        m = re.search(r"cum=(\d+)/(\d+)=", line)
        if not m:
            continue
        num, den = int(m.group(1)), int(m.group(2))
        if cur is not None and den < cur[1]:
            reqs.append(cur)           # denominator reset => new request
            cur = None
        cur = (num, den)
    if cur:
        reqs.append(cur)
    for i, (num, den) in enumerate(reqs):
        label = ORDER[i] if i < len(ORDER) else f"extra{i}"
        acc[label].append(num / den)

print(f"{'arm/cell':40s} {'N':>2s} {'acc_med':>8s} {'min':>6s} {'max':>6s}")
for label in ORDER:
    v = acc.get(label, [])
    if not v:
        continue
    print(f"{label:40s} {len(v):2d} {statistics.median(v):8.3f} {min(v):6.3f} {max(v):6.3f}")
