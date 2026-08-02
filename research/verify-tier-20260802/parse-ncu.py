#!/usr/bin/env python3
"""ncu raw-CSV summarizer: per-(kernel, grid) medians of the limiter metrics.

Usage: parse-ncu.py <ncu-dir>
Reads every ncu-*-raw.csv (`ncu --import --csv --page raw`, one row per captured launch).
Groups by (report, kernel, grid) — a name-regex capture can span multiple shapes; the grid
disambiguates. Emits a markdown table + ncu-summary.jsonl.

Columns: duration us, DRAM % of peak, achieved DRAM GB/s, SM %, achieved/theoretical
occupancy, regs/thread, waves/SM, top-2 warp-stall reasons (cycles per issue-active).
BASE-CLOCK cells (--clock-control base): compare ncu cells to ncu cells only.
"""
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

d = Path(sys.argv[1] if len(sys.argv) > 1 else "ncu")

COLS = {
    "gpu__time_duration.avg": ("dur_us", 1),             # unit row: us
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": ("dram_pct", 1),
    "dram__bytes.sum.per_second": ("dram_gbs", 1),       # unit row: Gbyte/s
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": ("sm_pct", 1),
    "sm__warps_active.avg.pct_of_peak_sustained_active": ("occ_pct", 1),
    "sm__maximum_warps_per_active_cycle_pct": ("occ_theo_pct", 1),
    "launch__registers_per_thread": ("regs", 1),
    "launch__waves_per_multiprocessor": ("waves", 1),
    "launch__grid_size": ("grid", 1),
    "smsp__issue_active.avg.per_cycle_active": ("issue_per_cyc", 1),
}
STALL_RE = re.compile(r"smsp__average_warps_issue_stalled_(\w+)_per_issue_active\.ratio")

rows_out = []
for f in sorted(d.glob("ncu-*-raw.csv")):
    groups = defaultdict(lambda: defaultdict(list))
    with open(f) as fh:
        rd = csv.DictReader(fh)
        try:
            unit_row = next(rd)  # second header line holds units
        except StopIteration:
            continue
        # duration unit varies per export (us in most, ms in some) — normalize to us
        dur_scale = {"us": 1.0, "ms": 1000.0, "ns": 0.001}[
            unit_row.get("gpu__time_duration.avg", "us")]
        for r in rd:
            key = (r["Kernel Name"], r.get("launch__grid_size", "?"))
            for col, (name, sc) in COLS.items():
                v = r.get(col, "")
                if v not in ("", None):
                    try:
                        x = float(v.replace(",", "")) * sc
                        if name == "dur_us":
                            x *= dur_scale
                        groups[key][name].append(x)
                    except ValueError:
                        pass
            for col, v in r.items():
                m = STALL_RE.match(col or "")
                if m and v not in ("", None):
                    try:
                        groups[key]["stall_" + m.group(1)].append(float(v))
                    except ValueError:
                        pass
    for (kn, grid), mm in groups.items():
        med = {k: statistics.median(v) for k, v in mm.items()}
        st = sorted(((v, k[6:]) for k, v in med.items() if k.startswith("stall_")
                     and k not in ("stall_selected",)), reverse=True)[:2]
        out = {"report": f.stem.replace("-raw", "").replace("ncu-", ""),
               "kernel": kn, "grid": int(med.get("grid", 0)),
               "n": len(mm.get("dur_us", [])),
               "dur_us": round(med.get("dur_us", float("nan")), 1),
               "dram_pct": round(med.get("dram_pct", float("nan")), 1),
               "dram_gbs": round(med.get("dram_gbs", float("nan")), 0),
               "sm_pct": round(med.get("sm_pct", float("nan")), 1),
               "occ_pct": round(med.get("occ_pct", float("nan")), 1),
               "occ_theo_pct": round(med.get("occ_theo_pct", float("nan")), 1),
               "regs": int(med.get("regs", 0)),
               "waves": round(med.get("waves", float("nan")), 2),
               "top_stalls": [f"{n}={v:.1f}" for v, n in st]}
        rows_out.append(out)

rows_out.sort(key=lambda r: (r["report"], -r["dur_us"] * r["n"]))
print("| report | kernel | grid | N | dur us | DRAM% | GB/s | SM% | occ ach/theo | regs | waves | top stalls (warps/issue) |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for m in rows_out:
    print(f"| {m['report']} | {m['kernel']} | {m['grid']} | {m['n']} | {m['dur_us']} "
          f"| {m['dram_pct']} | {m['dram_gbs']:.0f} | {m['sm_pct']} "
          f"| {m['occ_pct']}/{m['occ_theo_pct']} | {m['regs']} | {m['waves']} "
          f"| {'; '.join(m['top_stalls'])} |")

with open(d / "ncu-summary.jsonl", "w") as fh:
    for m in rows_out:
        fh.write(json.dumps(m) + "\n")
