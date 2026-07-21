#!/usr/bin/env python3
"""P0.B: simulate CPU-side expert cache hit rate vs size, from a real route trace.

Model: HBM residency = hottest complete experts (size-aware, 13.97 GB budget, mimics
profile-admit freeze). Remaining routed expert-projections stream through an LRU byte
cache (the companion's normal-RAM cache). Calibrated against the real 20 GiB anchor:
55.0% projection hit rate, ~0.87 GB/token NVMe fill.
"""
import json, sys
from collections import Counter, OrderedDict

TRACE = "/home/avifenesh/projects/bw24/research/per-expert-quant/evidence/local-5090-sota-20260719/window4-routes.trace"
MANIFEST = "/home/avifenesh/.local/share/bw24-models/hy3-layer103p5-dual-nvme/manifest.json"
HBM_BUDGET = 13.97 * 1024**3
N_USED = 8
N_EMBD, N_FF = 4096, 1536

# bytes per 256-weight superblock (Q8_0 given per 32-weight block scaled x8)
SB = {"Q2_K": 84, "Q3_K": 110, "IQ3_S": 110, "IQ4_XS": 136, "Q4_K": 144, "Q8_0": 34 * 8}

def row_bytes(qtype, in_features):
    return SB[qtype] * (in_features // 256)

def proj_bytes(qtype, proj):
    if proj == "down":
        return N_EMBD * row_bytes(qtype, N_FF)      # 4096 rows, 1536-wide
    return N_FF * row_bytes(qtype, N_EMBD)          # gate/up: 1536 rows, 4096-wide

def load_plan():
    m = json.load(open(MANIFEST))
    qt = {}  # (layer, expert, proj) -> qtype
    for a in m["plan"]["assignments"]:
        for p in a["projections"]:
            for e in a["experts"]:
                qt[(a["layer"], e, p)] = a["qtype"]
    return qt

def load_trace():
    """Return list of events; each decode token = list of (layer, [experts])."""
    prefill, decode = [], []
    current = {}
    for line in open(TRACE):
        layer_s, tok_s, routes_s = line.split()
        layer, ntok = int(layer_s), int(tok_s)
        routes = [int(x) for x in routes_s.split(",")]
        if ntok == 1:
            if current and (layer in current or layer < max(current)):
                decode.append(current); current = {}
            current[layer] = routes
        else:
            for t in range(ntok):
                prefill.append((layer, routes[t*N_USED:(t+1)*N_USED]))
    if current: decode.append(current)
    return prefill, decode

def main():
    qt = load_plan()
    prefill, decode = load_trace()
    print(f"trace: {len(prefill)} prefill layer-token events, {len(decode)} decode passes")

    # frequency across everything (warmup proxy) for residency admit
    freq = Counter()
    for layer, experts in prefill:
        for e in experts: freq[(layer, e)] += 1
    for p in decode:
        for layer, experts in p.items():
            for e in experts: freq[(layer, e)] += 1

    def expert_bytes(layer, e):
        total = 0
        for proj in ("gate", "up", "down"):
            q = qt.get((layer, e, proj))
            if q is None: return None  # pruned/unmanaged
            total += proj_bytes(q, proj)
        return total

    # admit hottest complete experts until HBM budget
    resident = set()
    used = 0
    for (layer, e), _ in freq.most_common():
        b = expert_bytes(layer, e)
        if b is None: continue
        if used + b > HBM_BUDGET: continue
        resident.add((layer, e)); used += b
    print(f"residency: {len(resident)} complete experts, {used/1024**3:.2f} GiB")

    # CPU-side stream: decode passes, non-resident experts, projection granularity
    stream = []
    per_tok_qtype_bytes = Counter()
    for p in decode:
        for layer, experts in p.items():
            for e in experts:
                if (layer, e) in resident: continue
                for proj in ("gate", "up", "down"):
                    q = qt.get((layer, e, proj))
                    if q is None: continue
                    b = proj_bytes(q, proj)
                    stream.append(((layer, e, proj), b))
                    per_tok_qtype_bytes[q] += b
    ntok = len(decode)
    cpu_inst = sum(1 for _ in stream) / 3 / ntok
    print(f"CPU-routed: {cpu_inst:.0f} expert-instances/token, "
          f"{sum(b for _, b in stream)/ntok/1024**3:.3f} GB demand/token")
    tot = sum(per_tok_qtype_bytes.values())
    print("CPU demand by qtype:", {q: f"{100*b/tot:.1f}%" for q, b in per_tok_qtype_bytes.most_common()})

    for cache_gib in (20, 24, 28, 32, 36):
        cap = cache_gib * 1024**3
        lru = OrderedDict()
        used = 0
        hits = misses = miss_bytes = 0
        for key, b in stream:
            if key in lru:
                lru.move_to_end(key); hits += 1
            else:
                misses += 1; miss_bytes += b
                lru[key] = b; used += b
                while used > cap:
                    _, ob = lru.popitem(last=False); used -= ob
        print(f"cache {cache_gib:2d} GiB: hit {100*hits/(hits+misses):5.1f}%  "
              f"miss {miss_bytes/ntok/1024**3:.3f} GB/token  "
              f"(anchor@20: 55.0% / 0.87 GB-fills per token incl warmup diffs)")

if __name__ == "__main__":
    main()
