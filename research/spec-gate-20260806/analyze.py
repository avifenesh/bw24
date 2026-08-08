#!/usr/bin/env python3
"""spec-gate: turn the raw load points into the lane's tables.

Medians across reps per (arm, concurrency). Same-rep arms only are ever compared (the H100-lane
law: cross-run and cross-day comparisons are clock-drift-invalid), which is why the harness is
rep-major with a rotating arm order — the median of N reps is the summary of N legitimate
same-rep comparisons, not a pooled cross-run average.

Usage: analyze.py <points.jsonl> [--metric agg_tok_s] [--csv]
"""
import argparse, json, statistics as st
from collections import defaultdict


def load(path):
    pts = defaultdict(list)
    for ln in open(path):
        ln = ln.strip()
        if not ln:
            continue
        p = json.loads(ln)
        lbl = p["label"]  # e.g. G-gated-c4-r2
        arm = lbl.rsplit("-c", 1)[0]
        rep = int(lbl.rsplit("-r", 1)[1])
        pts[(arm, p["concurrency"])].append((rep, p))
    return pts


def med(vals):
    vals = [v for v in vals if v is not None]
    return st.median(vals) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("points")
    ap.add_argument("--csv", action="store_true")
    a = ap.parse_args()
    pts = load(a.points)
    arms = sorted({k[0] for k in pts})
    cs = sorted({k[1] for k in pts})

    def cell(arm, c, field):
        rows = pts.get((arm, c), [])
        return med([p.get(field) for _, p in rows])

    def n_of(arm, c):
        return len(pts.get((arm, c), []))

    fields = [("agg_tok_s", "agg tok/s", "{:.1f}"),
              ("ttft_p50_s", "TTFT p50", "{:.3f}"),
              ("ttft_p95_s", "TTFT p95", "{:.3f}"),
              ("lat_p50_s", "per-stream p50", "{:.3f}"),
              ("lat_p95_s", "per-stream p95", "{:.3f}")]

    for f, name, fmt in fields:
        print(f"\n### {name}  (median of N reps)\n")
        print("| c | " + " | ".join(arms) + " | N |")
        print("|" + "---|" * (len(arms) + 2))
        for c in cs:
            vals = []
            for arm in arms:
                v = cell(arm, c, f)
                vals.append(fmt.format(v) if v is not None else "-")
            ns = {n_of(arm, c) for arm in arms}
            print(f"| {c} | " + " | ".join(vals) + f" | {min(ns)}-{max(ns)} |")

    # error/shed accounting — a table of medians hides a failing arm otherwise
    print("\n### run health (totals across reps)\n")
    print("| arm | c | n_ok | n_err | n_shed | errors |")
    print("|---|---|---|---|---|---|")
    for arm in arms:
        for c in cs:
            rows = pts.get((arm, c), [])
            if not rows:
                continue
            ok = sum(p["n_ok"] for _, p in rows)
            er = sum(p["n_err"] for _, p in rows)
            sh = sum(p["n_shed"] for _, p in rows)
            samples = [s for _, p in rows for s in p.get("errors_sample", [])]
            mark = "" if er == 0 else " **"
            print(f"| {arm} | {c} | {ok} | {er}{mark} | {sh} | "
                  f"{samples[0][:70] if samples else ''} |")

    # per-rep spread on the headline metric: a median is only honest with its spread stated
    print("\n### per-rep agg tok/s (spread check)\n")
    print("| arm | c | reps | min | median | max | spread |")
    print("|---|---|---|---|---|---|---|")
    for arm in arms:
        for c in cs:
            rows = sorted(pts.get((arm, c), []))
            if not rows:
                continue
            v = [p["agg_tok_s"] for _, p in rows]
            print(f"| {arm} | {c} | {len(v)} | {min(v):.1f} | {st.median(v):.1f} | {max(v):.1f} | "
                  f"{(max(v)/min(v) - 1) * 100:.1f}% |")


if __name__ == "__main__":
    main()
