#!/usr/bin/env python3
"""Summarize ornith-serve-20260801 receipts: serve points (median over reps) and
board cells (median over interleaved pairs). Prints markdown tables for SUMMARY.md."""
import json
import statistics
import sys
from collections import defaultdict

R = "/home/avifenesh/projects/wt-ornith-serve-bench/research/ornith-serve-20260801"


def load(path):
    rows = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except FileNotFoundError:
        pass
    return rows


def serve_table():
    pts = load(f"{R}/serve-points.jsonl")
    # label: <cfg>-c<N>-rep<r>
    g = defaultdict(list)
    for p in pts:
        cfg = p["label"].rsplit("-rep", 1)[0]
        g[cfg].append(p)
    print("| config | c | reqs | N | agg tok/s (median) | reps | p50 lat s | p95 lat s | errs |")
    print("|---|---|---|---|---|---|---|---|---|")
    order = sorted(g, key=lambda k: (k.rsplit("-c", 1)[0], int(k.rsplit("-c", 1)[1])))
    for cfg in order:
        rows = g[cfg]
        aggs = [r["agg_tok_s"] for r in rows]
        p50 = statistics.median(r["lat_p50_s"] for r in rows)
        p95 = statistics.median(r["lat_p95_s"] for r in rows)
        errs = sum(r["n_err"] for r in rows)
        name, c = cfg.rsplit("-c", 1)
        print(f"| {name} | {c} | {rows[0]['requests']} | {len(rows)} | "
              f"**{statistics.median(aggs):.1f}** | {', '.join(f'{a:.1f}' for a in aggs)} | "
              f"{p50:.2f} | {p95:.2f} | {errs} |")


def board_table():
    rows = load(f"{R}/board-cells.jsonl")
    g = defaultdict(list)
    for r in rows:
        g[(r["cell"], r["arm"], r["metric"])].append(float(r["toks"]))
    cells = sorted({c for c, _, _ in g})
    print("| cell | metric | memra median (N) | llama median (N) | ratio |")
    print("|---|---|---|---|---|")
    for cell in cells:
        for mm, lm in (("decode", "tg128"), ("prefill", "pp512")):
            mv = g.get((cell, "memra", mm), [])
            lv = g.get((cell, "llama", lm), [])
            if not mv or not lv:
                continue
            m, l = statistics.median(mv), statistics.median(lv)
            print(f"| {cell} | {mm}/{lm} | {m:.1f} (N={len(mv)}: {', '.join(f'{x:.1f}' for x in sorted(mv))}) "
                  f"| {l:.1f} (N={len(lv)}: {', '.join(f'{x:.1f}' for x in sorted(lv))}) | {m/l:.3f} |")
    # e2e proxy: 512 prefill + 128 decode wall from the cell medians
    print()
    print("| cell | e2e proxy memra s | llama s | e2e ratio (llama/memra wall) |")
    print("|---|---|---|---|")
    for cell in cells:
        try:
            mpp = statistics.median(g[(cell, "memra", "prefill")])
            mtg = statistics.median(g[(cell, "memra", "decode")])
            lpp = statistics.median(g[(cell, "llama", "pp512")])
            ltg = statistics.median(g[(cell, "llama", "tg128")])
        except (KeyError, statistics.StatisticsError):
            continue
        me = 512 / mpp + 128 / mtg
        le = 512 / lpp + 128 / ltg
        print(f"| {cell} | {me:.3f} | {le:.3f} | {le/me:.3f} |")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "serve"):
        print("## Serve points\n")
        serve_table()
        print()
    if which in ("all", "board"):
        print("## Board cells\n")
        board_table()
