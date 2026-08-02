#!/usr/bin/env python3
"""Agentic tool-loop cacheability + Step-scale earnings math. All inputs cited in REPORT."""

# --- 1. Cacheable fraction of a K-turn agent tool loop ---
# Turn i sends prompt = B (system+tools header) + (i-1)*delta growth.
# Turn i (i>=2) cache-reads the entire turn (i-1) prompt as an exact prefix:
#   cached_i = B + (i-2)*delta ; turn 1 cold within a session but B hits cross-session.
def loop(B, delta, K, cross_session_B_hit=True):
    total_in = sum(B + (i-1)*delta for i in range(1, K+1))
    # per-session prefix reuse (turns 2..K)
    cached = sum(B + (i-2)*delta for i in range(2, K+1))
    if cross_session_B_hit:
        cached += B  # turn 1 hits the shared system+tools header via cross-request cache
    return total_in, cached, cached/total_in

for (B, delta, K) in [(2000,500,3),(2000,500,5),(2000,500,10),(3000,800,8),(1500,300,12)]:
    t,c,f = loop(B,delta,K)
    print(f"B={B} delta={delta} K={K:2d}: total_in={t:6d} cached={c:6d} frac={f:.1%}")

print()
# --- 2. The CORRECT frame: saturated prefill-bound replica (the section-3 multiplier) ---
# A saturated Step replica is prefill-bound (93.5:1 in:out): the binding constraint is prefill
# COMPUTE, not price. Cache does NOT lower a fixed stream's price -- it lets the SAME prefill
# compute serve 1/(1-h) as many billable input tokens, the extra ones billing at r*p_in on
# ~zero marginal compute. Input-revenue multiplier vs h=0, cache-read = r * p_in:
print("Saturated prefill-bound replica -- billable-input revenue multiplier vs no-cache:")
for r, label in [(0.25, "25% (hy3 / OR band)"), (0.20, "20% (Step)")]:
    row = []
    for h in [0.50, 0.70, 0.85, 0.90, 0.991]:
        mult = 1 + r*h/(1-h)
        row.append(f"h={h:.0%}:x{mult:.2f}")
    print(f"  cache-read {label:22s} | " + "  ".join(row))

print()
# --- 3. WRONG frame (documented so nobody re-derives it): fixed token VOLUME ---
# Holding token volume fixed makes cache look like it CUTS revenue (-61% at h=85%) because you
# bill 25% instead of 100% per cached token. That is the wrong model for a saturated endpoint:
# it ignores that cached tokens free up compute to serve MORE billable tokens. Kept as a warning.
def endpoint_rev_fixedvol(in_tok, out_tok, p_in, p_out, p_cache, h):
    return (in_tok*(1-h)*p_in + in_tok*h*p_cache + out_tok*p_out)/1e6
out_day = 29800e6/(93.5*0.20+1.15)   # one endpoint's daily out-tok at $29.8K/day floor pool
in_day  = 93.5*out_day
base = endpoint_rev_fixedvol(in_day, out_day, 0.19, 1.09, 0.045, 0.0)
print("WRONG frame (fixed token volume -- do NOT use for a saturated endpoint):")
for h in [0.0, 0.5, 0.73, 0.85]:
    rev = endpoint_rev_fixedvol(in_day, out_day, 0.19, 1.09, 0.045, h)
    print(f"  h={h:.0%}: ${rev/1e3:5.1f}K/day  (delta vs h=0: {(rev-base)/base:+.0%})  <- misleading")
