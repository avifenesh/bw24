#!/usr/bin/env python3
"""Parse an ab-<tag>.log (pp512 interleaved OFF/ON) into per-arm rep pools + the ratio.

Reports median over ALL reps of ALL rounds per arm, the [min..max] range, the spread, and whether
the two arms' ranges overlap (the w4a8 lane's readability criterion). Also carries the per-round
clock/temp so a power-capped round is visible.

Usage: parse_ab.py <ab-tag.log> [--jsonl out.jsonl] [--tag TAG]
"""
import json
import re
import statistics as st
import sys


def main():
    path = sys.argv[1]
    jsonl = sys.argv[sys.argv.index("--jsonl") + 1] if "--jsonl" in sys.argv else None
    tag = (sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv
           else path.split("ab-")[-1].replace(".log", ""))

    model = prompt = "?"
    arms = {"OFF": [], "ON": []}
    meta = {"OFF": [], "ON": []}
    rcs = {"OFF": [], "ON": []}
    cur = None
    for ln in open(path).read().splitlines():
        if ln.startswith("[model] "):
            model = ln.split(None, 1)[1]
        elif ln.startswith("[prompt] "):
            prompt = ln.split(None, 1)[1]
        m = re.match(r"=== round (\d+) arm (\w+) (\d+) %, (\d+), ([\d.]+) W, (\d+) MHz", ln)
        if m:
            cur = m.group(2)
            meta[cur].append({"round": int(m.group(1)), "util": int(m.group(3)),
                              "temp": int(m.group(4)), "power_w": float(m.group(5)),
                              "clock_mhz": int(m.group(6))})
            continue
        if ln.startswith("=== round") and "SKIPPED" in ln:
            print(f"SKIPPED ROUND: {ln}")
            cur = None
            continue
        if cur is None:
            continue
        m = re.match(r"pp-only rep \d+: [\d.]+s = ([\d.]+) tok/s", ln)
        if m:
            arms[cur].append(float(m.group(1)))
        m = re.match(r"\[rc=(\d+)\]", ln)
        if m:
            rcs[cur].append(int(m.group(1)))

    print(f"model={model}\nprompt={prompt}\n")
    res = {}
    for a in ("OFF", "ON"):
        v = arms[a]
        if not v:
            continue
        med, lo, hi = st.median(v), min(v), max(v)
        spread = (hi - lo) / med * 100
        cl = [x["clock_mhz"] for x in meta[a]]
        tp = [x["temp"] for x in meta[a]]
        print(f"ARM {a}: N={len(v)}  median {med:.1f}  [{lo:.1f} .. {hi:.1f}]  "
              f"spread {spread:.2f}%")
        print(f"  clocks {min(cl)}-{max(cl)} MHz  temp {min(tp)}-{max(tp)} C  rc={rcs[a]}")
        res[a] = {"n": len(v), "median": med, "min": lo, "max": hi, "spread_pct": spread,
                  "reps": v, "clock_min": min(cl), "clock_max": max(cl),
                  "temp_min": min(tp), "temp_max": max(tp), "rc": rcs[a]}

    if "OFF" in res and "ON" in res:
        ratio = res["ON"]["median"] / res["OFF"]["median"]
        overlap = not (res["ON"]["min"] > res["OFF"]["max"] or res["OFF"]["min"] > res["ON"]["max"])
        print(f"\nRATIO ON/OFF = {ratio:.4f}x   ranges {'OVERLAP' if overlap else 'DO NOT OVERLAP'}")
        if jsonl:
            with open(jsonl, "a") as f:
                f.write(json.dumps({"tag": tag, "kind": "pp512-ab", "model": model,
                                    "prompt": prompt, "ratio_on_over_off": ratio,
                                    "ranges_overlap": overlap, "arms": res}) + "\n")
            print(f"appended 1 row -> {jsonl}")


if __name__ == "__main__":
    main()
