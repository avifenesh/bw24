#!/usr/bin/env python3
"""Per-SKU revenue + 3y TCO on 2-5x used RTX 5090 — sku-repick research 2026-08-02.

Extends research/hw-buy-20260802/tco-model.py (same capex/energy/resale method) with an
explicit prefill/decode time split, because the candidate set spans traffic shapes from
6:1 to 133:1 in:out (measured, OR 4-day rankings) — the hw-buy "bundled $/M-out"
shortcut is only honest at <=10:1.

Method (every number's basis):
- D_card = projected saturated output tok/s per DESKTOP 5090 (1.79 TB/s). Anchor: the
  MEASURED Qwen3.6-35B-A3B IQ4_XS 178.2 tok/s plain on the 896 GB/s laptop rig
  (research/tune-data/current-board.json, 2026-08-02) -> ~40% of peak bandwidth for
  A3B-class MoE -> 300 tok/s/card desktop projection (hw-buy method, bracket 180-500).
  Other MoE candidates: D_card = 0.40 * 1790 / active_GB_per_tok, active_GB/tok =
  active_params * bpw/8 at the honest quant. Dense (gemma-31b): 85% efficiency receipt,
  spec-decode 1.5x per board ratios. All projections superseded by bring-up measurement.
- PP-N box: saturated D_box = N * D_card (pipeline full; hop cost 7-11.5us << decode
  tick, research/m0-nccl-20260801). Single-stream ~= D_card.
- Prefill P_box = pmult * D_box. Default pmult=10 ("order of magnitude faster", the
  hw-buy/or-provider assumption). pmult=5 sensitivity shown: sm_120a prefill is our
  least-receipted stage and at 90:1+ shapes prefill IS the revenue engine.
- Time split: f = R_comp/(pmult + R_comp) where R_comp = R*(1-cache_hit).
  Y (out tok/s) = D_box*(1-f); X_computed = pmult*D_box*f; X_billed = X_comp/(1-hit).
- Cache-read billed at 25% of our input price (OR norm; poolside prices 10% — flagged).
- Wedge prices ~5-10% under the 2026-08-02 endpoint floors (receipts in raw/).
- TCO: used 5090 $2.7k/card (sold-comps, hw-buy), host $3k (n<=2) / $4k (n>=3), energy
  $0.18/kWh PUE 1.1, load 0.70*TDP / idle 0.08*TDP, host 250W, resale 60% at 3y.
"""

HOURS_3Y = 3 * 8760

# name, n_cards, D_card, R in:out (measured), wedge $/M in, $/M out, weights
CAND = [
    ("q3.6-35b-a3b 2 replicas [SUPPORTED]",  2, 300, 10.3, 0.095, 0.90,  "IQ4_XS 18GB/card"),
    ("q3.6-35b-a3b 4 replicas [SUPPORTED]",  4, 300, 10.3, 0.095, 0.90,  "IQ4_XS 18GB/card"),
    ("gemma-4-26b-a4b 2 repl [SUPPORTED]",   2, 280, 16.4, 0.065, 0.28,  "QAT Q4_0 ~15GB/card"),
    ("gemma-4-31b 2 repl spec [SUPPORTED]",  2, 130, 13.1, 0.085, 0.31,  "Q4_0 17GB/card"),
    ("step-3.7-flash PP-4",                  4, 122, 93.5, 0.190, 1.090, "UD-IQ4_XS 95.3GB"),
    ("minimax-m2.7 PP-4",                    4, 135, 39.0, 0.215, 0.860, "UD-IQ4_XS 108.4GB"),
    ("laguna-s-2.1 PP-4",                    4, 160, 133,  0.081, 0.162, "Q4_K_M 96GB / NVFP4 ~62GB"),
    ("laguna-xs-2.1 2 replicas",             2, 300, 124,  0.054, 0.108, "Q4_K_M 20.3GB/card"),
    ("qwen3.5-122b-a10b PP-2 [ARCH SUPP.]",  2, 135, 13.3, 0.245, 1.950, "UD-IQ4_XS 60.2GB"),
    ("gpt-oss-120b PP-3",                    3, 240, 6.0,  0.028, 0.160, "MXFP4 63.4GB"),
    ("nemotron-3-super PP-3",                3, 112, 31.5, 0.080, 0.380, "UD-IQ4_XS 64.5GB"),
    ("v4-flash PP-5 [5 CARDS]",              5, 104, 14.1, 0.080, 0.160, "UD-IQ4_XS 136.7GB"),
    ("mimo-v2.5 PP-5 [5-6 CARDS, tight]",    5, 90,  117,  0.100, 0.200, "UD-IQ4_XS 149.4GB"),
]

def econ(n, D_card, R, p_in, p_out, cache_hit=0.0, util=0.30, kwh=0.18,
         pue=1.1, pmult=10.0):
    D_box = n * D_card
    R_comp = R * (1 - cache_hit)
    f = R_comp / (pmult + R_comp)
    Y = D_box * (1 - f)
    X_comp = pmult * D_box * f
    X_billed = X_comp / (1 - cache_hit) if cache_hit < 1 else 0
    p_cache = 0.25 * p_in
    rev_s = (X_comp * p_in + (X_billed - X_comp) * p_cache + Y * p_out) / 1e6
    hr = rev_s * 3600
    capex = n * 2700 + (3000 if n <= 2 else 4000)
    avg_kw = (n * 575 * (util * 0.70 + (1 - util) * 0.08) + 250) / 1000.0
    energy = avg_kw * pue * HOURS_3Y * kwh
    resale = n * 2700 * 0.60
    tco = capex - resale + energy
    rev_3y = hr * util * HOURS_3Y
    return dict(hr=hr, Y=Y, Xb=X_billed, capex=capex, tco=tco, rev=rev_3y,
                net=rev_3y - tco)

if __name__ == "__main__":
    for label, ch, pm in [("cache 0%, P=10xD", 0.0, 10.0),
                          ("cache 70%, P=10xD", 0.70, 10.0),
                          ("cache 70%, P=5xD (prefill sensitivity)", 0.70, 5.0)]:
        print(f"\n=== {label} | util=30% | wedge ~5-10% under 2026-08-02 floors ===")
        print(f"{'SKU':38s} {'$/hr sat':>8s} {'out t/s':>7s} {'in t/s':>8s} "
              f"{'capex$':>7s} {'3yTCO$':>7s} {'3y rev$':>8s} {'3y net$':>8s}")
        for name, n, d, r, pi, po, note in CAND:
            e = econ(n, d, r, pi, po, cache_hit=ch, pmult=pm)
            print(f"{name:38s} {e['hr']:8.2f} {e['Y']:7.0f} {e['Xb']:8.0f} "
                  f"{e['capex']:7.0f} {e['tco']:7.0f} {e['rev']:8.0f} {e['net']:8.0f}")
