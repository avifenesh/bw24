#!/usr/bin/env python3
"""verify-tier parser: probe cost curves + nsys per-arm kernel attribution -> premium tables.

Usage: parse-tier.py <lane-dir>
Reads logs/probe-*.log ([econ-json]) and logs/nsys-*_cuda_gpu_kern_sum.csv.

Attribution model (mirrors verify-economics §6, generalized to every T):
  spec-econ with MEMRA_ECON_ONLY runs ONE arm N+3 times after (a) a 2048-token prime and
  (b) a t_max-step greedy continuation (rolled back). So in an arm's kern_sum:
    - prefill-only kernels (mul_mat_q*, fa_prefill*, quantize_mmq*, l2_norm_pp*,
      memra_f16_cvt*) are fixed-count -> excluded;
    - batched b-tier kernels (_b2*/_b4*/_b8*/dual_b2*/dual_b4*) run ONLY in the verify pass
      -> per-pass = total/ITER (ITER = N+3);
    - m=1-class kernels (mr2_rp, dual_mr2_rp, q5_K il/mr2) in a verify arm come ONLY from
      the t_max continuation -> excluded there; in the decode_h arm they run in both the
      continuation (t_max steps) and the loop (ITER steps) -> per-step = total/(ITER+t_max);
    - glue (rms_norm/add/silu/quantize_q8_1/gdn_*/fa_decode*/sigmoid/...) runs in prime +
      continuation + loop; per-pass glue is reported as the probe-wall residual, not
      per-kernel (their per-launch cost is ns-scale; the b-tier question doesn't need it).
Emits: cost-curve table + curve JSONL rows, per-arm b-tier kernel tables (ms/pass and
share of the arm's probe wall), and the decode_h m=1 reference table.
"""
import csv
import json
import re
import sys
from pathlib import Path

lane = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
logs = lane / "logs"

PREFILL = re.compile(r"mul_mat_q|fa_prefill|quantize_mmq|l2_norm_pp|memra_f16_cvt|CUDA memcpy|cvt_")
M1 = re.compile(r"_mmvq_mr2|_mmvq_dual_mr2|_mmvq_il$|_mmvq_mr2_il|_mmvq_rp$|q5_K_mmvq(_il)?$"
                r"|q4_K_mmvq$|q6_K_mmvq$|q8_0_mmvq$|e4m3_mmvq$|_dp4a(_rp)?$")
BATCH = re.compile(r"_mmvq(_dual)?_b\d+|_mmvq_b\d+")

def probe_curves():
    out = {}
    for f in sorted(logs.glob("probe-*.log")):
        for line in f.read_text().splitlines():
            if line.startswith("[econ-json] "):
                out[f.stem.replace("probe-", "").replace("-t9", "")] = json.loads(line[12:])
    return out

def kern_csv(f):
    rows = []
    with open(f) as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    return rows

probes = probe_curves()
print("## Cost curve — decode(T=1) vs verify(T=1..9), probe medians (N per json)\n")
curve_rows = []
for tag, j in probes.items():
    arms = j["arms"]
    d = arms["decode_h"]["med_ms"]
    print(f"### {tag} (ctx={j['pos']}, N={j['n']}), decode T=1 = {d:.3f}ms\n")
    print("| T | verify ms | x decode | premium ms | us/extra-column (vs T=1) | marginal us/col (T-1 -> T) |")
    print("|---|---|---|---|---|---|")
    prev = None
    v1 = arms.get("verify_t1", {}).get("med_ms", d)
    for t in range(1, 10):
        a = arms.get(f"verify_t{t}")
        if not a:
            continue
        v = a["med_ms"]
        prem = v - d
        percol = (v - v1) / (t - 1) * 1000 if t > 1 else 0.0
        marg = (v - prev) * 1000 if prev is not None else float("nan")
        print(f"| {t} | {v:.3f} | {v/d:.3f}x | {prem:+.3f} | {percol:.0f} | {marg:.0f} |")
        curve_rows.append({"model": tag, "T": t, "verify_ms": round(v, 4),
                           "decode_ms": round(d, 4), "ratio": round(v / d, 4),
                           "us_per_extra_col_vs_t1": round(percol, 1),
                           "marginal_us_col": round(marg, 1) if prev is not None else None,
                           "n": j["n"], "pos": j["pos"]})
        prev = v
    print()

with open(lane / "cost-curve.jsonl", "w") as fh:
    for r in curve_rows:
        fh.write(json.dumps(r) + "\n")

ITER_N = 15 + 3   # MEMRA_ECON_N=15 + 3 warmups in the nsys runs

print("## Per-arm kernel attribution (nsys kern_sum; per-pass = total/18)\n")
for f in sorted(logs.glob("nsys-*_cuda_gpu_kern_sum.csv")):
    m = re.match(r"nsys-(q\d+)-(decode_h|verify_t\d+)_cuda", f.name)
    if not m:
        continue
    tag, arm = m.group(1), m.group(2)
    j = probes.get(tag)
    wall = None
    if j:
        wall = j["arms"].get(arm if arm != "decode_h" else "decode_h", {}).get("med_ms")
    rows = kern_csv(f)
    t_max = 9 if arm == "decode_h" else int(arm.split("_t")[1])
    batch, m1, glue_total = [], [], 0.0
    for r in rows:
        name = r["Name"].split("<")[0].strip('"')
        ns = float(r["Total Time (ns)"])
        inst = int(r["Instances"])
        if PREFILL.search(name):
            continue
        if BATCH.search(name):
            batch.append((name, ns / ITER_N / 1e6, inst / ITER_N, float(r["Avg (ns)"]) / 1e3))
        elif M1.search(name):
            m1.append((name, ns, inst))
        else:
            glue_total += ns
    print(f"### {tag} {arm}" + (f" — probe wall {wall:.2f}ms/pass" if wall else "") + "\n")
    if arm == "decode_h":
        div = ITER_N + t_max
        print("| m=1 kernel | ms/step | launches/step | avg us |")
        print("|---|---|---|---|")
        for name, ns, inst in sorted(m1, key=lambda x: -x[1]):
            print(f"| {name} | {ns/div/1e6:.3f} | {inst/div:.0f} | {ns/inst/1e3:.1f} |")
    else:
        tot = sum(b[1] for b in batch)
        print("| b-tier kernel | ms/pass | share of wall | launches/pass | avg us/launch |")
        print("|---|---|---|---|---|")
        for name, ms, inst, avg in sorted(batch, key=lambda x: -x[1]):
            sh = f"{ms/wall*100:.1f}%" if wall else "-"
            print(f"| {name} | {ms:.3f} | {sh} | {inst:.0f} | {avg:.1f} |")
        if wall:
            print(f"| **sum b-tier** | **{tot:.3f}** | **{tot/wall*100:.1f}%** | | |")
        # m=1-class kernels in a verify arm come from (a) the t_max-step greedy continuation
        # (a decode-step program each) and (b) any per-pass residents — the dp4a class runs at
        # grid.y=m INSIDE the verify pass (no decode-exact batched twin at its call sites), and
        # at T=9 the WHOLE pass falls off the b-tier onto grid.y=9 per-row MMVQ. Correct with
        # the decode_h profile: per_pass = (total - t_max x decode-profile per-step) / ITER.
        dref = kern_csv(logs / f"nsys-{tag}-decode_h_cuda_gpu_kern_sum.csv")
        def dstep_of(name):
            return sum(float(r["Total Time (ns)"]) for r in dref
                       if r["Name"].split("<")[0].strip('"') == name) / (ITER_N + 9)
        corr = []
        for name, ns, inst in m1:
            pp = (ns - t_max * dstep_of(name)) / ITER_N / 1e6
            if pp > 0.05:
                corr.append((name, pp, ns / inst / 1e3))
        if corr:
            tot_corr = sum(c[1] for c in corr)
            print(f"\nm=1-class kernels WITH per-pass residency (continuation-corrected vs the "
                  f"decode_h profile): {tot_corr:.1f}ms/pass = "
                  f"{tot_corr/wall*100:.0f}% of wall" + ("  <- THE OFF-TIER PASS" if t_max == 9 else "") + "\n")
            print("| kernel | ms/pass (corrected) | avg us/launch |")
            print("|---|---|---|")
            for name, pp, avg in sorted(corr, key=lambda x: -x[1]):
                print(f"| {name} | {pp:.3f} | {avg:.1f} |")
        print()
print()
