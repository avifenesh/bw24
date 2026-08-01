#!/usr/bin/env python3
# q27 shipping-config matrix verdict (2026-08-01, H100 GPU 3, board-2048, NGEN=512).
# e2e = 512 / (512/decode + 2048/prefill); prefill term uses that run's measured prime
# seconds directly (2048/prefill == prime_s). Emits per-run rows + N=3 median verdict.
import re, glob, os, json, statistics as st
from collections import defaultdict

D = os.path.dirname(os.path.abspath(__file__))
VLLM = 72.9
rows = []
for p in sorted(glob.glob(D + "/matrix-*-r?.log")):
    b = os.path.basename(p)[:-4].replace("matrix-", "")
    cfg, run = b.rsplit("-", 1)
    t = open(p, errors="replace").read()
    gen = re.search(r"\[generate\]\s+\d+ tok in [\d.]+s = ([\d.]+) tok/s.*prime ([\d.]+)s", t)
    spec = re.search(r"\[generate_spec K=\d+\] \d+ tok in [\d.]+s = ([\d.]+) tok/s.*prime ([\d.]+)s", t)
    acc = re.search(r"acceptance: \d+/\d+ = ([\d.]+)%", t)
    mir = re.search(r"\[q4kf16\] prefill fp16 mirrors built: (\d+) tensors \((\d+) MB\)", t)
    q8rp = re.search(r"\[q8rp\] split-plane decode mirrors built: (\d+) tensors", t)
    sc = "PASS" if "SELF-CONSISTENCY PASS" in t else "FAIL"
    plain, prime_p = float(gen.group(1)), float(gen.group(2))
    specd, prime_s = float(spec.group(1)), float(spec.group(2))
    row = dict(cfg=cfg, run=run, prime_s=prime_p,
               prefill=round(2048 / prime_p, 1), plain=plain, spec=specd,
               acc=float(acc.group(1)),
               e2e_plain=round(512 / (512 / plain + prime_p), 2),
               e2e_spec=round(512 / (512 / specd + prime_s), 2),
               f16_mirrors=f"{mir.group(1)}t/{mir.group(2)}MB",
               kqrp_tensors=int(q8rp.group(1)), gate=sc)
    rows.append(row)
    print(json.dumps(row))

print()
agg = defaultdict(list)
for r in rows:
    agg[r["cfg"]].append(r)
hdr = ("cfg", "prefill", "plain", "spec", "acc", "e2e_plain", "e2e_spec", "x_vllm_plain", "x_vllm_spec")
print("%-18s %8s %7s %7s %5s %9s %8s %12s %11s" % hdr)
best = None
for k in sorted(agg):
    v = agg[k]
    m = {f: st.median(x[f] for x in v) for f in
         ("prefill", "plain", "spec", "acc", "e2e_plain", "e2e_spec")}
    print("%-18s %8.1f %7.2f %7.2f %5.1f %9.2f %8.2f %11.3fx %10.3fx" % (
        k, m["prefill"], m["plain"], m["spec"], m["acc"], m["e2e_plain"], m["e2e_spec"],
        m["e2e_plain"] / VLLM, m["e2e_spec"] / VLLM))
    if best is None or (m["e2e_plain"], m["e2e_spec"]) > (best[1]["e2e_plain"], best[1]["e2e_spec"]):
        best = (k, m)
print("\nWINNER: %s  (e2e_plain %.2f = %.2fx vLLM %.1f; e2e_spec %.2f = %.2fx)  N=3 medians" % (
    best[0], best[1]["e2e_plain"], best[1]["e2e_plain"] / VLLM, VLLM,
    best[1]["e2e_spec"], best[1]["e2e_spec"] / VLLM))
