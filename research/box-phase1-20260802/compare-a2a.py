#!/usr/bin/env python3
"""a2a curve compare: box-aug2 (us-east-2 p5.48xlarge) vs Jul-31 Mumbai receipts.
Medians of per_a2a_us at 64 KiB/peer, N stated per row (evidence discipline)."""
import json, statistics as st, sys
from pathlib import Path

here = Path(__file__).parent
ref = here / "../m0-nccl-20260801/receipts"
new = here / "receipts/m0-a2a"

def med(f, size=65536):
    rows = [json.loads(l) for l in open(f)]
    vals = [r["per_a2a_us"] for r in rows if r.get("size_bytes_per_peer") == size]
    return st.median(vals), len(vals)

print(f"{'set':28s} {'jul31 (Mumbai)':>16s} {'aug2 (use2 box)':>16s} {'delta':>8s}")
for n, refname, newname in [(2, "nccl_a2a_n2", "a2a_n2"), (4, "nccl_a2a_n4", "a2a_n4"),
                            (8, "nccl_a2a_n8", "a2a_n8"),
                            (2, "nccl_ga2a_n2", "ga2a_n2"), (4, "nccl_ga2a_n4", "ga2a_n4"),
                            (8, "nccl_ga2a_n8", "ga2a_n8")]:
    r, rn = med(ref / f"{refname}.jsonl")
    m, mn = med(new / f"{newname}.jsonl")
    kind = "graph" if "ga2a" in refname else "eager"
    print(f"{kind}-nccl a2a n={n} @64KiB    {r:>10.2f}us N{rn} {m:>10.2f}us N{mn} {100*(m-r)/r:>+7.1f}%")
