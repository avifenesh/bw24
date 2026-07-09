#!/usr/bin/env python3
"""Delta table: MTP acceptance CEILING (bf16 full-prec) vs the NVFP4 quant hit.

Reads two JSONL files emitted by acceptance_battery.sh (rows: arm/prompt/k/run/acc_rate/per_slot/...)
and prints, per (prompt, K), the median acceptance of each arm and the delta. The delta (ceiling minus
quant) IS the quant hit on drafting — the MTP-heal protocol deliverable (HANDOVER "BW24 DUAL-SHAPE").
First file = ceiling (bf16), second = quant (NVFP4), unless --a-arm/--b-arm pin labels.

Usage:
  tools/acceptance_delta.py out-bf16.jsonl out-nvfp4.jsonl
  tools/acceptance_delta.py out-bf16.jsonl out-nvfp4.jsonl --json summary.json
"""
import argparse, json, statistics, sys
from collections import defaultdict

def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def median_by_key(rows, arm=None):
    # key = (prompt, k) -> list of acc_rate ; also collect per_slot lists
    acc = defaultdict(list)
    slots = defaultdict(list)
    consist = defaultdict(list)
    for r in rows:
        if arm is not None and r.get("arm") != arm:
            continue
        if r.get("acc_rate") is None:
            continue
        key = (r["prompt"], r["k"])
        acc[key].append(r["acc_rate"])
        if r.get("per_slot"):
            slots[key].append([s for s in r["per_slot"] if s is not None])
        if r.get("self_consistency"):
            consist[key].append(r["self_consistency"])
    med = {k: statistics.median(v) for k, v in acc.items()}
    # median per-slot position-wise (only across equal-length lists)
    slotmed = {}
    for k, lists in slots.items():
        if not lists:
            continue
        L = min(len(x) for x in lists)
        slotmed[k] = [statistics.median([x[i] for x in lists]) for i in range(L)]
    return med, slotmed, consist

def prompt_sort_key(p):
    # p1,p2,p3 then agentloop-tN
    if p.startswith("agentloop-t"):
        try:
            return (1, int(p.split("t")[-1]))
        except ValueError:
            return (1, 0)
    return (0, p)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ceiling", help="bf16 full-prec JSONL")
    ap.add_argument("quant", help="NVFP4 JSONL")
    ap.add_argument("--a-arm", default=None, help="filter arm label in ceiling file")
    ap.add_argument("--b-arm", default=None, help="filter arm label in quant file")
    ap.add_argument("--json", default=None, help="also write a machine-readable summary here")
    args = ap.parse_args()

    a_rows, b_rows = load(args.ceiling), load(args.quant)
    a_med, a_slot, a_con = median_by_key(a_rows, args.a_arm)
    b_med, b_slot, b_con = median_by_key(b_rows, args.b_arm)

    keys = sorted(set(a_med) | set(b_med), key=lambda k: (prompt_sort_key(k[0]), k[1]))

    print(f"# MTP acceptance: ceiling (bf16 full-prec) vs quant (NVFP4)  [median of N runs]")
    print(f"# ceiling={args.ceiling}  quant={args.quant}")
    print(f"{'prompt':<14} {'K':>2} {'bf16':>7} {'nvfp4':>7} {'delta':>7} {'hit%':>6} {'consist'}")
    print("-" * 60)
    summary = []
    for key in keys:
        p, k = key
        a = a_med.get(key)
        b = b_med.get(key)
        a_s = f"{a:.3f}" if a is not None else "   -  "
        b_s = f"{b:.3f}" if b is not None else "   -  "
        if a is not None and b is not None:
            delta = a - b
            hit = (delta / a * 100.0) if a > 0 else 0.0
            d_s = f"{delta:+.3f}"
            h_s = f"{hit:+.1f}"
        else:
            delta = hit = None
            d_s = h_s = "  -  "
        cons = ""
        cc = a_con.get(key, []) + b_con.get(key, [])
        if cc:
            cons = "FAIL" if "FAIL" in cc else "PASS"
        print(f"{p:<14} {k:>2} {a_s:>7} {b_s:>7} {d_s:>7} {h_s:>6} {cons}")
        summary.append({
            "prompt": p, "k": k, "ceiling_acc": a, "quant_acc": b,
            "delta": delta, "hit_pct": hit,
            "ceiling_per_slot": a_slot.get(key), "quant_per_slot": b_slot.get(key),
            "self_consistency": cons or None,
        })

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"ceiling_file": args.ceiling, "quant_file": args.quant,
                       "rows": summary}, f, indent=2)
        print(f"\n[wrote {args.json}]")

    if not keys:
        print("\nWARN: no comparable (prompt,K) keys — check arm labels / that both files have rows",
              file=sys.stderr)

if __name__ == "__main__":
    main()
