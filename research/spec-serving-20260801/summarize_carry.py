#!/usr/bin/env python3
"""Summarize carry A/B points + burst flattening + trace + correctness byte-equality."""
import json, statistics, collections, re, glob

BASE = "/home/ubuntu/arc5/research/spec-serving-20260801"
pts = [json.loads(l) for l in open(f"{BASE}/carry-points.jsonl")]
g = collections.defaultdict(list)
for p in pts:
    arm, _, c = p["label"].split("-")
    g[(arm, int(c[1:]))].append(p)

def med(arm, c, field="agg_tok_s"):
    ok = [x for x in g[(arm, c)] if x["n_ok"] > 0]
    return statistics.median(x[field] for x in ok) if ok else None

print(f'{"point":18s} {"agg med":9s} {"n":2s} {"range":16s} {"p50_lat":8s} err')
for arm in ["speccarry", "specpre2", "plaincarry",
            "carryburst8", "carryburst32", "carryburst128"]:
    for c in [1, 2, 4, 8]:
        ok = [x for x in g[(arm, c)] if x["n_ok"] > 0]
        if not ok:
            continue
        v = sorted(x["agg_tok_s"] for x in ok)
        l = statistics.median(x["lat_p50_s"] for x in ok)
        e = sum(x["n_err"] for x in g[(arm, c)])
        print(f"{arm}-c{c:<8d} {statistics.median(v):8.1f} {len(ok):2d} "
              f"{v[0]:6.1f}-{v[-1]:6.1f}   {l:6.2f}s  {e}")
print()
for c in [1, 8]:
    pre, post = med("specpre2", c), med("speccarry", c)
    if pre and post:
        print(f"c={c}: spec carry/pre = {post/pre:.3f}x  ({pre:.1f} -> {post:.1f})")
print()
for c in [1, 2, 4]:
    ps, ss = med("plaincarry", c), med("speccarry", c)
    if ps and ss:
        print(f"c={c}: spec-carry/plain = {ss/ps:.2f}x  ({ss:.1f} vs {ps:.1f})")
print()
for b in [8, 32, 128]:
    for c in [4, 8]:
        v = med(f"carryburst{b}", c)
        if v:
            print(f"carry burst={b} c={c}: {v:.1f} tok/s")

d = a = n = 0
for f in glob.glob(f"{BASE}/server-speccarry-r*.log"):
    for l in open(f):
        m = re.search(r"burst=(\d+)/(\d+)", l)
        if m:
            a += int(m.group(1)); d += int(m.group(2)); n += 1
if d:
    print(f"\ncarry acceptance: {a}/{d} = {a/d:.3f} (n_bursts={n})")

print("\ncorrectness byte-equality (vs PRE-fix spec captures + plain):")
for p in ["p1", "p2"]:
    texts = {}
    for tag in ["spec", "specpost", "speccarry", "plain"]:
        try:
            texts[tag] = json.load(open(f"{BASE}/correctness-{tag}-{p}.json"))["choices"][0]["message"]["content"]
        except Exception:
            pass
    ref = texts.get("spec")
    for tag, t in texts.items():
        if tag == "spec" or ref is None:
            continue
        n_ = min(len(t), len(ref))
        print(f"  {p} {tag:10s} vs spec-pre: len={len(t)} common_prefix_exact={t[:n_]==ref[:n_]}")

print("\n[spec-setup] carry trace (continuation bursts):")
try:
    for l in open(f"{BASE}/server-carrytrace.log"):
        if "spec-setup" in l:
            print("  " + l.strip())
except Exception:
    pass
