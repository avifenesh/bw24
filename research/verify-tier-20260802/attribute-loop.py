#!/usr/bin/env python3
"""Exact loop-window attribution from the nsys sqlite timelines (no correction model).

The spec-econ MEMRA_ECON_ONLY run is: 2048-token prime -> t_max-step greedy continuation
(m=1 decode steps, mr2-class kernels) -> the arm N+3 times. In the GPU timeline the verify
loop is exactly the region from the first b-tier launch to the end of the trace (the
continuation's mr2 launches all precede it); decode_h's loop region starts at the first
mr2-class launch (continuation steps and loop steps are the same decode-step program).

Per-pass per-class cost = in-window per-kernel time / n_passes, classes:
  matvec_b     qmatvec_*_bN (the batched verify tier)
  matvec_m1    any other qmatvec_* inside the window (off-tier / per-pass residents)
  fa_attn      fa_decode_* + append_quantize_kv (T-scaled attention + KV append)
  gdn          gdn_* / ssm_* / qkv_to_gdn (linear-attention decode path)
  glue         everything else (norms, add, silu, quantize_q8_1, rope, gates, ...)
Gap = window wall - busy (union of kernel intervals; sum==union checked -> single stream).

Emits markdown + glue-share.jsonl. Usage: attribute-loop.py <lane-dir>
"""
import json
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

lane = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
logs = lane / "logs"

B_RE = re.compile(r"qmatvec_.*_b\d+")
# verify-loop window markers: a small-b b-tier launch (t2..t8), or the bare off-tier
# per-row NVFP4 kernel (t9 — the continuation only runs the mr2-suffixed decode twins,
# so the first bare _mmvq_rp launch is the first verify pass).
WIN_RE = re.compile(r"qmatvec_.*_b\d+|qmatvec_nvfp4_mmvq_rp$")
MV_RE = re.compile(r"qmatvec_")
FA_RE = re.compile(r"fa_decode|append_quantize_kv")
GDN_RE = re.compile(r"gdn_|ssm_|qkv_to_gdn")
PREFILL_RE = re.compile(r"mul_mat_q|quantize_mmq|fa_prefill|memra_f16_cvt|l2_norm_pp"
                        r"|embed_gather|f32_to_bf16")

def classify(name):
    if PREFILL_RE.search(name):
        return "prefill!"          # should not appear in-window; loud if it does
    if B_RE.search(name):
        return "matvec_b"
    if MV_RE.search(name):
        return "matvec_m1"
    if FA_RE.search(name):
        return "fa_attn"
    if GDN_RE.search(name):
        return "gdn"
    return "glue"

def load(db):
    con = sqlite3.connect(db)
    rows = con.execute(
        "SELECT k.start, k.end, s.value, k.gridX*k.gridY*k.gridZ, k.streamId "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName = s.id "
        "ORDER BY k.start").fetchall()
    con.close()
    return rows

def union_busy(iv):
    iv.sort()
    tot, cs, ce = 0, None, None
    for s, e in iv:
        if cs is None:
            cs, ce = s, e
        elif s <= ce:
            ce = max(ce, e)
        else:
            tot += ce - cs
            cs, ce = s, e
    if cs is not None:
        tot += ce - cs
    return tot

N_PASSES = 18   # MEMRA_ECON_N=15 + 3 warmups
T_CONT = 9      # decode_h continuation steps (TMAX)

results = {}
for db in sorted(logs.glob("nsys-*.sqlite")):
    m = re.match(r"nsys-(q\d+)-(decode_h|verify_t\d+)", db.stem)
    if not m:
        continue
    tag, arm = m.group(1), m.group(2)
    rows = load(db)
    if arm == "decode_h":
        # window: first mr2-class decode matvec -> end; 27 identical decode steps
        w0 = next(r[0] for r in rows if re.search(r"_mmvq_(dual_)?mr2|_mmvq_il$|q\d_K_mmvq$"
                                                  r"|_mmvq_mr2_il", r[2]))
        div = N_PASSES + T_CONT
    else:
        w0 = next(r[0] for r in rows if WIN_RE.search(r[2]))
        div = N_PASSES
    win = [r for r in rows if r[0] >= w0]
    wall = (win[-1][1] - w0) / 1e6
    busy = union_busy([(r[0], r[1]) for r in win]) / 1e6
    sumdur = sum(r[1] - r[0] for r in win) / 1e6
    per_kernel = defaultdict(lambda: [0.0, 0])
    per_kg = defaultdict(lambda: [0.0, 0])   # (kernel, grid) for ncu joins
    for s, e, name, grid, _ in win:
        per_kernel[name][0] += (e - s) / 1e6
        per_kernel[name][1] += 1
        per_kg[(name, grid)][0] += (e - s) / 1e6
        per_kg[(name, grid)][1] += 1
    cls = defaultdict(float)
    for name, (ms, n) in per_kernel.items():
        cls[classify(name)] += ms
    results[(tag, arm)] = dict(wall=wall, busy=busy, sumdur=sumdur, div=div,
                               cls=dict(cls), per_kernel=dict(per_kernel),
                               per_kg={f"{k}|{g}": v for (k, g), v in per_kg.items()})

CLS = ["matvec_b", "matvec_m1", "fa_attn", "gdn", "glue"]
out_rows = []
for tag in ("q27", "q9"):
    d = results[(tag, "decode_h")]
    ddiv = d["div"]
    dstep = {c: d["cls"].get(c, 0.0) / ddiv for c in CLS}
    dgap = (d["wall"] - d["busy"]) / ddiv
    dwall = d["wall"] / ddiv
    print(f"\n## {tag} — loop-window attribution (nsys timeline, ms/pass; "
          f"decode_h = {dwall:.3f} wall, overlap sum/union = {d['sumdur']/d['busy']:.4f})\n")
    hdr = "| arm | wall | " + " | ".join(CLS) + " | gap(idle+launch) |"
    print(hdr)
    print("|" + "---|" * (len(CLS) + 3))
    print(f"| decode_h | {dwall:.3f} | " +
          " | ".join(f"{dstep[c]:.3f}" for c in CLS) + f" | {dgap:.3f} |")
    for t in range(2, 10):
        key = (tag, f"verify_t{t}")
        if key not in results:
            continue
        v = results[key]
        vp = {c: v["cls"].get(c, 0.0) / v["div"] for c in CLS}
        vgap = (v["wall"] - v["busy"]) / v["div"]
        vwall = v["wall"] / v["div"]
        print(f"| verify_t{t} | {vwall:.3f} | " +
              " | ".join(f"{vp[c]:.3f}" for c in CLS) + f" | {vgap:.3f} |")
        prem = vwall - dwall
        deltas = {c: vp[c] - dstep[c] for c in CLS}
        deltas["gap"] = vgap - dgap
        out_rows.append({"model": tag, "T": t, "wall_pp_ms": round(vwall, 4),
                         "decode_wall_ms": round(dwall, 4), "premium_ms": round(prem, 4),
                         **{f"d_{c}_ms": round(x, 4) for c, x in deltas.items()},
                         **{f"pp_{c}_ms": round(vp[c], 4) for c in CLS},
                         "overlap_check": round(v["sumdur"] / v["busy"], 4)})
    # premium split table
    print(f"\n### {tag} premium split (delta vs decode_h step, ms and % of premium)\n")
    print("| T | premium | " + " | ".join(CLS) + " | gap |")
    print("|" + "---|" * (len(CLS) + 3))
    for r in out_rows:
        if r["model"] != tag:
            continue
        p = r["premium_ms"]
        cells = " | ".join(f"{r[f'd_{c}_ms']:+.3f} ({r[f'd_{c}_ms']/p*100:.0f}%)"
                           for c in CLS)
        print(f"| {r['T']} | {p:+.3f} | {cells} | {r['d_gap_ms']:+.3f} "
              f"({r['d_gap_ms']/p*100:.0f}%) |")

with open(lane / "glue-share.jsonl", "w") as fh:
    for r in out_rows:
        fh.write(json.dumps(r) + "\n")

# per-kernel detail for the fa/gdn/glue classes at T=2,4,8 (the non-matvec carriers)
for tag in ("q27", "q9"):
    d = results[(tag, "decode_h")]
    print(f"\n### {tag} non-matvec per-kernel detail (ms/pass, vs decode_h per-step)\n")
    print("| kernel | class | decode_h | " +
          " | ".join(f"t{t}" for t in (2, 4, 8) if (tag, f"verify_t{t}") in results) + " |")
    print("|---|---|---|---|---|---|")
    names = set()
    for t in (2, 4, 8):
        v = results.get((tag, f"verify_t{t}"))
        if v:
            names |= {n for n in v["per_kernel"]
                      if classify(n) in ("fa_attn", "gdn", "glue")}
    names |= {n for n in d["per_kernel"] if classify(n) in ("fa_attn", "gdn", "glue")}
    def tot_at(res, n):
        return res["per_kernel"].get(n, [0.0, 0])[0] / res["div"]
    for n in sorted(names, key=lambda n: -max(tot_at(results.get((tag, "verify_t4"),
                                                                 d), n), tot_at(d, n))):
        cells = []
        for t in (2, 4, 8):
            v = results.get((tag, f"verify_t{t}"))
            cells.append(f"{tot_at(v, n):.4f}" if v else "-")
        base = tot_at(d, n)
        if max([float(c) for c in cells if c != "-"] + [base]) < 0.01:
            continue
        print(f"| {n} | {classify(n)} | {base:.4f} | " + " | ".join(cells) + " |")

# dump per-(kernel,grid) time shares for the b-tier carriers (ncu gap-to-peak weighting)
kg_out = {}
for (tag, arm), v in results.items():
    if arm == "decode_h":
        continue
    for kg, (ms, n) in v["per_kg"].items():
        name, grid = kg.rsplit("|", 1)
        if B_RE.search(name) or (MV_RE.search(name) and arm == "verify_t9"):
            kg_out.setdefault(f"{tag}-{arm}", []).append(
                {"kernel": name, "grid": int(grid), "ms_pp": round(ms / v["div"], 4),
                 "launches_pp": round(n / v["div"], 2)})
with open(lane / "btier-grid-shares.json", "w") as fh:
    json.dump(kg_out, fh, indent=1)
print("\nwrote glue-share.jsonl + btier-grid-shares.json")
