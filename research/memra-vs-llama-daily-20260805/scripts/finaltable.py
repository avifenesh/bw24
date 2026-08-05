#!/usr/bin/env python3
"""Final RESULTS table: per (arm, cell) medians, N, spread; prefill truth; evict counts."""
import json, statistics, collections, re, glob

RDIR = "/home/avifenesh/projects/bw24/research/memra-vs-llama-daily-20260805"
rows = [json.loads(l) for l in open(f"{RDIR}/runs.jsonl")]
g = collections.defaultdict(list)
for r in rows:
    if "error" not in r:
        g[(r["arm"], r["cell"])].append(r)

def med(v):
    v = [x for x in v if x is not None]
    return statistics.median(v) if v else None

print("== decode / e2e / ttft medians (N=5 each, interleaved) ==")
hdr = f"{'arm':22s} {'cell':14s} {'ttft':>6s} {'dec':>6s} {'dec-rng':>9s} {'e2e':>6s} {'ntok':>5s} {'llama-srv-dec':>13s}"
print(hdr)
for arm in ["memra-t0.8", "memra-t0.8-lsampler", "memra-t1.0", "memra-greedy",
            "llama-default-t0.8", "llama-t1.0"]:
    for cell in ["short-agentic", "long-gen", "ctx4k"]:
        rs = g[(arm, cell)]
        dec = [r["decode_tok_s"] for r in rs if r["decode_tok_s"]]
        srv = med([(r.get("server_timings") or {}).get("predicted_per_second") for r in rs])
        print(f"{arm:22s} {cell:14s} "
              f"{med([r['ttft_s'] for r in rs]):6.3f} "
              f"{med(dec) or 0:6.1f} "
              f"{(min(dec) if dec else 0):4.0f}-{(max(dec) if dec else 0):-4.0f} "
              f"{med([r['e2e_tok_s'] for r in rs]) or 0:6.1f} "
              f"{med([r['completion_tokens'] for r in rs]):5.0f} "
              f"{srv or 0:13.1f}")

print("\n== ctx4k prefill (cold, nonce-defeated caches) ==")
for arm in ["memra-t1.0", "memra-greedy", "llama-default-t0.8", "llama-t1.0"]:
    rs = g[(arm, "ctx4k")]
    ptoks = med([r["prompt_tokens"] for r in rs])
    ttft = med([r["ttft_s"] for r in rs])
    print(f"{arm:22s} prompt_tokens={ptoks:.0f} ttft={ttft:.3f}s -> eff prefill ~{ptoks/ttft:.0f} tok/s")
# llama server-truth prompt_per_second
pps = []
for cell in ["ctx4k"]:
    for arm in ["llama-default-t0.8", "llama-t1.0"]:
        for r in g[(arm, cell)]:
            st = r.get("server_timings") or {}
            if st.get("prompt_per_second"):
                pps.append(st["prompt_per_second"])
print(f"llama server-truth prompt_per_second @ctx4k: median {statistics.median(pps):.0f} tok/s")

print("\n== memra server-truth cross-check (usage.elapsed_s vs client e2e_s) ==")
deltas = []
for (arm, cell), rs in g.items():
    if not arm.startswith("memra"):
        continue
    for r in rs:
        if r.get("server_elapsed_s"):
            deltas.append(r["e2e_s"] - r["server_elapsed_s"])
print(f"client-minus-server wall delta: median {statistics.median(deltas)*1000:.0f} ms, "
      f"max {max(deltas)*1000:.0f} ms (N={len(deltas)})")

print("\n== memra spec-pool evict-retry per phase (the F5 signature) ==")
for f in sorted(glob.glob(f"{RDIR}/logs/server-memra-r*.log")):
    n = sum(1 for l in open(f) if "spec pool evicted" in l)
    print(f"{f.split('/')[-1]}: {n} evict-retries (13 requests/phase incl warmup)")

print("\n== llama draft acceptance (server truth) ==")
for arm in ["llama-default-t0.8", "llama-t1.0"]:
    for cell in ["short-agentic", "long-gen", "ctx4k"]:
        accs = []
        for r in g[(arm, cell)]:
            st = r.get("server_timings") or {}
            if st.get("draft_n"):
                accs.append(st["draft_n_accepted"] / st["draft_n"])
        if accs:
            print(f"{arm:22s} {cell:14s} acc median {statistics.median(accs):.3f}")
