#!/usr/bin/env python3
"""Aggregate runs.jsonl -> per (arm, cell) medians with N and spread."""
import json, statistics, sys, collections

RDIR = "/home/avifenesh/projects/bw24/research/memra-vs-llama-daily-20260805"
rows = [json.loads(l) for l in open(f"{RDIR}/runs.jsonl")]

cells = ["short-agentic", "long-gen", "ctx4k"]
arms = ["memra-t0.8", "memra-t0.8-lsampler", "memra-t1.0", "memra-greedy",
        "llama-default-t0.8", "llama-t1.0"]

g = collections.defaultdict(list)
for r in rows:
    if "error" in r:
        continue
    g[(r["arm"], r["cell"])].append(r)

def med(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None

def spread(vals):
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return None
    return (max(vals) - min(vals))

print(f"{'arm':22s} {'cell':14s} {'N':>2s} {'ttft_med':>8s} {'dec_med':>8s} {'dec_rng':>8s} "
      f"{'e2e_med':>8s} {'ntok_med':>8s} {'srv_dec':>8s} {'acc':>6s}")
for arm in arms:
    for cell in cells:
        rs = g[(arm, cell)]
        if not rs:
            continue
        ttft = med([r["ttft_s"] for r in rs])
        dec = med([r["decode_tok_s"] for r in rs])
        rng = spread([r["decode_tok_s"] for r in rs])
        e2e = med([r["e2e_tok_s"] for r in rs])
        ntok = med([r["completion_tokens"] for r in rs])
        srv = med([(r.get("server_timings") or {}).get("predicted_per_second") for r in rs])
        # llama acceptance: draft_n_accepted/draft_n
        accs = []
        for r in rs:
            st = r.get("server_timings") or {}
            if st.get("draft_n"):
                accs.append(st["draft_n_accepted"] / st["draft_n"])
        acc = med(accs)
        print(f"{arm:22s} {cell:14s} {len(rs):2d} "
              f"{ttft:8.3f} {dec if dec else 0:8.1f} {rng if rng else 0:8.1f} "
              f"{e2e if e2e else 0:8.1f} {ntok:8.0f} "
              f"{srv if srv else 0:8.1f} {acc if acc else 0:6.2f}")
