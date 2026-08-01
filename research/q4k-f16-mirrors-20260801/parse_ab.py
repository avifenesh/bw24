#!/usr/bin/env python3
# Parse q4k-f16 ab-logs: prime s -> prefill tok/s, plain tok/s, spec tok/s, acceptance.
import re, glob, os, json, statistics as st
from collections import defaultdict

D = os.path.expanduser("~/arc4/research/q4k-f16-mirrors-20260801/ab-logs")
rows = []
for p in sorted(glob.glob(D + "/*.log")):
    b = os.path.basename(p)[:-4]
    bin_, cls, run = b.split("-")
    t = open(p, errors="replace").read()
    ntok = re.search(r"-> (\d+) tokens", t)
    gen = re.search(r"\[generate\]\s+\d+ tok in [\d.]+s = ([\d.]+) tok/s.*prime ([\d.]+)s", t)
    spec = re.search(r"\[generate_spec K=\d+\] \d+ tok in [\d.]+s = ([\d.]+) tok/s.*prime ([\d.]+)s", t)
    acc = re.search(r"acceptance: \d+/\d+ = ([\d.]+)%", t)
    sc = "PASS" if "SELF-CONSISTENCY PASS" in t else "FAIL"
    ntok = int(ntok.group(1))
    prime_s = float(gen.group(2))
    row = dict(bin=bin_, cls=cls, run=run, ntok=ntok,
               prefill=round(ntok / prime_s, 1), prime_s=prime_s,
               plain=float(gen.group(1)), spec=float(spec.group(1)),
               acc=float(acc.group(1)), gate=sc)
    rows.append(row)
    print(json.dumps(row))
print()
agg = defaultdict(list)
for r in rows:
    agg[(r["bin"], r["cls"])].append(r)
print("%-5s %-11s %10s %10s %9s %7s" % ("bin", "class", "prefill_med", "plain_med", "spec_med", "acc_med"))
for k in sorted(agg):
    v = agg[k]
    print("%-5s %-11s %10.1f %10.2f %9.2f %7.1f" % (
        k[0], k[1], st.median(x["prefill"] for x in v), st.median(x["plain"] for x in v),
        st.median(x["spec"] for x in v), st.median(x["acc"] for x in v)))
