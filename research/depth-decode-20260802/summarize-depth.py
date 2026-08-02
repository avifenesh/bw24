#!/usr/bin/env python3
"""Summarize depth-sweep.jsonl / depth-accept.jsonl into per-cell medians.
Filters the 4 orphan rows from the aborted first launch (ts 2026-08-02T07:24:00Z,
kat/memra/d512 rep1 — the relaunched session re-measured that point)."""
import json, statistics, sys, collections

R = "/home/avifenesh/projects/wt-depth-decode/research/depth-decode-20260802"
ORPHAN_TS = "2026-08-02T07:24:00Z"

def load(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if not line: continue
        r = json.loads(line)
        if r.get("ts") == ORPHAN_TS: continue
        rows.append(r)
    return rows

def cell(rows, **kv):
    out = [r["value"] for r in rows
           if all(r.get(k) == v for k, v in kv.items()) and r["value"] is not None]
    return out

def med(v):
    return statistics.median(v) if v else None

def fmt(v):
    return f"{v:.1f}" if v is not None else "-"

sweep = load(f"{R}/depth-sweep.jsonl")
print("== depth-sweep: decode tok/s (tg128 @ depth), per-cell median (N per cell in brackets) ==")
print(f"{'model':6} {'depth':>6} {'memra':>16} {'llama':>16} {'ratio':>7}")
for m in ["kat", "q35", "o35b"]:
    base = {}
    for d in [512, 2048, 4096, 6144]:
        mem = cell(sweep, model=m, engine="memra", depth=d, metric="tg128_toks")
        lla = cell(sweep, model=m, engine="llama", depth=d, metric="tg128_toks")
        mm, lm = med(mem), med(lla)
        ratio = f"{mm/lm:.3f}x" if mm and lm else "-"
        print(f"{m:6} {d:>6} {fmt(mm):>10} [N={len(mem)}] {fmt(lm):>10} [N={len(lla)}] {ratio:>7}")
        if d == 512: base[m] = (mm, lm)
    mm0, lm0 = base.get(m, (None, None))
    for d in [2048, 4096, 6144]:
        mm = med(cell(sweep, model=m, engine="memra", depth=d, metric="tg128_toks"))
        lm = med(cell(sweep, model=m, engine="llama", depth=d, metric="tg128_toks"))
        if mm and lm and mm0 and lm0:
            print(f"   decay d512->d{d}: memra {100*(mm/mm0-1):+.1f}%  llama {100*(lm/lm0-1):+.1f}%")

try:
    acc = load(f"{R}/depth-accept.jsonl")
    print("\n== depth-accept: drafter acceptance %% @K=2 (greedy-deterministic; reps = determinism check) ==")
    print(f"{'model':6} {'depth':>6} {'acc%':>8} {'plain':>8} {'spec':>8} {'ratio':>7} {'consistency':>12}")
    for m in ["kat", "o35b"]:
        for d in [512, 2048, 4096, 6144]:
            a = cell(acc, model=m, depth=d, metric="acceptance_pct")
            p = med(cell(acc, model=m, depth=d, metric="plain_decode_toks"))
            s = med(cell(acc, model=m, depth=d, metric="spec_k2_decode_toks"))
            c = cell(acc, model=m, depth=d, metric="spec_consistency_pass")
            det = "det-OK" if len(set(a)) <= 1 else f"DRIFT {sorted(set(a))}"
            ratio = f"{s/p:.2f}x" if s and p else "-"
            print(f"{m:6} {d:>6} {fmt(med(a)):>8} {fmt(p):>8} {fmt(s):>8} {ratio:>7} "
                  f"pass={sum(1 for x in c if x>0)}/{len(c)} {det}")
except FileNotFoundError:
    pass
