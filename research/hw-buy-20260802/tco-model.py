#!/usr/bin/env python3
"""3-year TCO per million tokens served — hardware-buy research 2026-08-02.

Method (REPORT.md section 3):
- hy3 saturated output tok/s per box = 44.8 x aggregate_memory_TB/s (calibrated at the
  PP-2 H100 SXM point: 6.7 TB/s -> 300 tok/s, the or-provider revenue-model assumption;
  bracket +/-50%; superseded when the PP-2 spike measures it).
- Revenue: blended $1.10 gross per 1M output tokens at hy3 wedge prices ($0.105/M in,
  $0.44/M out, cache-read $0.026/M; 10:1 agentic in:out; bracket $0.70-1.49).
- Utilization: 30% base / 60% high. Energy: while-serving load 0.70xTDP, idle 0.08xTDP,
  host 250 W always-on; PUE 1.1 (self-host) unless colo. Electricity $0.12-0.25/kWh.
- Colo alternative charged at $/kW/month on nameplate (GPUs TDP + host).
- Resale at 3 years: fraction of PURCHASE price recovered (from section 6 resale arcs).

Prices are August-2026 street/executed with sources in REPORT.md section 2.
"""

HOURS_3Y = 3 * 8760
TOKS_PER_TBS = 300.0 / 6.7
REV = 1.10  # $/M output tokens, blended

# name, n_gpu, gpu_price, host_cost, bw_TBs_per_gpu, tdp_W, vram_GB, resale_frac, note
CAND = [
    ("2x RTX PRO 6000 (new $12.1k)",     2, 12100, 4000, 1.79, 600, 96, 0.55, "sm_120a PP-2"),
    ("2x RTX PRO 6000 (refurb $10k)",    2, 10000, 4000, 1.79, 600, 96, 0.55, "sm_120a PP-2"),
    ("2x RTX PRO 6000 Max-Q (new)",      2, 13250, 4000, 1.79, 300, 96, 0.55, "residential"),
    ("6x RTX 5090 used ($2.7k)",         6,  2700, 7000, 1.79, 575, 32, 0.60, "PP-6, EULA risk"),
    ("6x RTX 5090 new ($4.1k)",          6,  4100, 7000, 1.79, 575, 32, 0.45, "PP-6, EULA risk"),
    ("2x H100 NVL 94GB used ($26.25k)",  2, 26250, 5000, 3.90, 400, 94, 0.45, "sm_90a PP-2"),
    ("2x H200 NVL new ($31k)",           2, 31000, 5000, 4.80, 600, 141, 0.50, "sm_90a PP-2"),
    ("4x L40S used ($6.8k)",             4,  6800, 5000, 0.864, 350, 48, 0.45, "sm_89 port"),
    ("4x RTX 6000 Ada used ($6k)",       4,  6000, 5000, 0.960, 300, 48, 0.45, "sm_89 port"),
    ("8x H100 SXM used server (~$120k)", 8, 11500, 28000, 3.35, 700, 80, 0.45, "4 PP-2 replicas, NVLink"),
    ("8x A100 SXM4 used server (~$65k)", 8,  5529, 21000, 2.04, 400, 80, 0.25, "sm_80 port, no FP8/FP4"),
    ("8x B200 HGX new ($450k)",          8, 50000, 50000, 8.00, 1000, 192, 0.60, "allocation-gated"),
]

def row(name, n, gp, host, bw, tdp, vram, rf, note,
        util=0.30, kwh=0.18, pue=1.1, colo_kw_mo=None):
    capex = n * gp + host
    agg = n * bw
    sat_toks = agg * TOKS_PER_TBS
    mtok_3y = sat_toks * util * HOURS_3Y * 3600 / 1e6
    avg_kw = (n * tdp * (util * 0.70 + (1 - util) * 0.08) + 250) / 1000.0
    if colo_kw_mo:
        nameplate_kw = (n * tdp + 250) / 1000.0
        housing = colo_kw_mo * nameplate_kw * 36
        energy = 0.0  # colo price includes power
    else:
        housing = 0.0
        energy = avg_kw * pue * HOURS_3Y * kwh
    resale = n * gp * rf
    tco = capex - resale + energy + housing
    return dict(name=name, capex=capex, agg=agg, sat=sat_toks, mtok=mtok_3y,
                energy=energy + housing, resale=resale, tco=tco,
                dpm=tco / mtok_3y, rev=mtok_3y * REV,
                gross_hr_sat=sat_toks * 3600 / 1e6 * REV, note=note)

if __name__ == "__main__":
    import sys
    util = float(sys.argv[1]) if len(sys.argv) > 1 else 0.30
    kwh = float(sys.argv[2]) if len(sys.argv) > 2 else 0.18
    print(f"util={util:.0%} elec=${kwh}/kWh  (hy3 blended ${REV}/M-out)")
    print(f"{'box':34s} {'capex$':>7s} {'aggTB/s':>7s} {'sat t/s':>7s} {'$/hr sat':>8s} "
          f"{'3yMtok':>7s} {'engy$':>6s} {'resale$':>7s} {'TCO$':>7s} {'$/Mtok':>7s} {'3y rev$':>8s}")
    rows = []
    for c in CAND:
        r = row(*c, util=util, kwh=kwh)
        rows.append(r)
    for r in sorted(rows, key=lambda r: r['dpm']):
        print(f"{r['name']:34s} {r['capex']:7.0f} {r['agg']:7.2f} {r['sat']:7.0f} "
              f"{r['gross_hr_sat']:8.2f} {r['mtok']:7.0f} {r['energy']:6.0f} "
              f"{r['resale']:7.0f} {r['tco']:7.0f} {r['dpm']:7.2f} {r['rev']:8.0f}")

# ---- SKU-B: the 9-35B lane (what a 2-4x 5090 box actually sells) ----
# Throughput anchor: MEASURED Qwen3.6-35B-A3B IQ4_XS plain decode 178.2 tok/s on the
# 896 GB/s laptop rig (current-board.json 2026-08-02); desktop 5090 / RTX PRO 6000 are
# 1.79 TB/s -> 2.0x bandwidth -> ~356 ceiling; we use 300 tok/s/card (0.84x of scaling,
# MoE-efficiency held constant), bracket 180-500 (spec receipts reach 302 ON THE LAPTOP).
# H100: MEASURED 226 e2e single-stream (h100_board) -> 250/GPU assumed batched-mild.
# Revenue: qwen3.6-35b-a3b live floor $0.10/M in, $0.95/M out (endpoints API receipt,
# 2026-08-02, 9 providers); 5:1 in:out chat/agent mix -> $1.45/M-out bundle uncached;
# blended used here: $1.20/M-out (bracket 0.95-1.45).
SKU_B_REV = 1.20
SKU_B = [
    # name, n, gpu$, host$, tok/s per card, tdp, resale
    ("2x 5090 used (35B lane)",    2, 2700, 3000, 300, 575, 0.60),
    ("2x 5090 new (35B lane)",     2, 4100, 3000, 300, 575, 0.45),
    ("4x 5090 used (35B lane)",    4, 2700, 4000, 300, 575, 0.60),
    ("1x RTX PRO 6000 new (35B)",  1, 12100, 2500, 300, 600, 0.55),
    ("1x RTX PRO 6000 refurb",     1, 10000, 2500, 300, 600, 0.55),
    ("6x 5090 used (35B lane)",    6, 2700, 7000, 300, 575, 0.60),
    ("1x H100 NVL used (35B)",     1, 26250, 4000, 250, 400, 0.45),
]

def sku_b_row(name, n, gp, host, tps, tdp, rf, util=0.30, kwh=0.18, pue=1.1):
    capex = n * gp + host
    sat = n * tps
    mtok = sat * util * HOURS_3Y * 3600 / 1e6
    avg_kw = (n * tdp * (util * 0.70 + (1 - util) * 0.08) + 250) / 1000.0
    energy = avg_kw * pue * HOURS_3Y * kwh
    resale = n * gp * rf
    tco = capex - resale + energy
    return dict(name=name, capex=capex, sat=sat, mtok=mtok, energy=energy,
                resale=resale, tco=tco, dpm=tco / mtok, rev=mtok * SKU_B_REV,
                hr=sat * 3600 / 1e6 * SKU_B_REV)
