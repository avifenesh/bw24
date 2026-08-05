#!/usr/bin/env python3
"""Gate (b) verdict computation: diff the pre-fix vs post-fix serve-repro captures.

Reads serve-repro-{prefix,fixed}.txt (produced by serve-repro.sh on the SAME tree, only the
spec.rs bonus-column stats rule differing) and prints, per arm:
  n      = usage.completion_tokens
  bang   = count of '!' in the completion text (the id-0 injection signature; token 0 in this
           tokenizer decodes to '!')
  byte-eq = pre-fix text == post-fix text

The two binding readings:
  1. every truncated arm goes bang>0 -> bang=0
  2. greedy / untruncated / top_k-only arms are byte-eq True  (the exactness contract:
     top_k-only stays byte-identical, greedy untouched)
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path):
    out, lab = {}, None
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith("==="):
            lab = line.strip("= ")
            continue
        if not line.strip():
            continue
        d = json.loads(line)
        out[lab] = (d["choices"][0]["text"], d["usage"]["completion_tokens"])
    return out


pre = load(os.path.join(HERE, "serve-repro-prefix.txt"))
fix = load(os.path.join(HERE, "serve-repro-fixed.txt"))

# arms that MUST be byte-identical across the fix (no truncation, or top_k-only)
EXACT = lambda k: ("greedy" in k) or ("untruncated" in k) or ("top_k:40 only" in k)

print(f"{'arm':42s} {'pre n/bang':>12s} {'fix n/bang':>12s}  byte-eq")
fails = []
for k in pre:
    pt, pn = pre[k]
    ft, fn = fix[k]
    pb, fb = pt.count("!"), ft.count("!")
    print(f"{k:42s} {pn:5d}/{pb:<6d} {fn:5d}/{fb:<6d}  {pt == ft}")
    if fb > 0:
        fails.append(f"{k}: post-fix still injects {fb} '!'")
    if EXACT(k) and pt != ft:
        fails.append(f"{k}: exactness contract broken (text changed across the fix)")
    if not EXACT(k) and pb == 0:
        fails.append(f"{k}: pre-fix arm did not reproduce the bug (gate went vacuous)")

print()
# post-fix cross-arm identity at a fixed seed: with the fix, every truncation shape that keeps
# the sampled id inside the same set must land on the same stream as untruncated at this seed.
for seed in (7, 999, 31337, 424242):
    base = fix.get("untruncated t0.8 seed=7", (None,))[0] if seed == 7 else None
    if base is None:
        continue
    for knob in ("top_k:40", "top_p:0.95", "min_p:0.05"):
        k = f"t0.8 seed={seed} {knob} only"
        if k in fix:
            print(f"post-fix {k}: byte-eq-to-untruncated={fix[k][0] == base}")

print()
if fails:
    print("FAIL:")
    for f in fails:
        print("  " + f)
    sys.exit(1)
print("=== serve-repro A/B GREEN: bug reproduced pre-fix on every truncated arm, "
      "zero id-0 injection post-fix, exact arms byte-identical ===")
