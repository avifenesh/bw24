#!/usr/bin/env python3
"""ornith-bar summarizer: 9B best-vs-best cell + o35b KQRP sweep medians from the jsonl rows."""
import json, statistics, sys, os

RD = os.path.dirname(os.path.abspath(__file__))

def load(fn):
    rows = []
    p = os.path.join(RD, fn)
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows

def med(vals):
    return statistics.median(vals) if vals else None

def fmt(v, nd=2):
    return "-" if v is None else f"{v:.{nd}f}"

def cell9b():
    rows = [r for r in load("o9b-cell.jsonl") if r["rep"] != 0]
    if not rows:
        print("no 9B cell rows"); return
    classes = ["p1-code-short", "p2-code-medium", "p3-agentic-long"]
    arms = sorted({r["arm"] for r in rows})
    def vals(arm, cls, metric):
        return [r["value"] for r in rows if r["arm"] == arm and r["class"] == cls and r["metric"] == metric]
    print("=== Ornith-9B best-vs-best (N=3 medians, interleaved same-session) ===")
    for cls in classes:
        marm = "memra-spec"
        larm = [a for a in arms if a.startswith("llama")][0] if len([a for a in arms if a.startswith("llama")]) == 1 else "llama-plain"
        m_dec = med(vals(marm, cls, "decode_toks"))
        m_plain = med(vals(marm, cls, "plain_decode_toks"))
        m_np = med(vals(marm, cls, "n_prompt"))
        m_prime = med(vals(marm, cls, "prime_spec_s"))
        m_pp = med(vals(marm, cls, "prefill_toks"))
        m_acc = med(vals(marm, cls, "accept_pct"))
        l_dec = med(vals(larm, cls, "decode_toks"))
        l_pp = med(vals(larm, cls, "prefill_toks"))
        l_ps = med(vals(larm, cls, "prefill_s"))
        m_e2e = [ps + 256.0/d for ps, d in zip(vals(marm, cls, "prime_spec_s"), vals(marm, cls, "decode_toks"))]
        l_e2e = [ps + 256.0/d for ps, d in zip(vals(larm, cls, "prefill_s"), vals(larm, cls, "decode_toks"))]
        me, le = med(m_e2e), med(l_e2e)
        print(f"\n[{cls}] memra=run-spec drafter K=3, llama={larm}")
        print(f"  decode : memra spec {fmt(m_dec)} (plain {fmt(m_plain)}, acc {fmt(m_acc,1)}%) vs llama {fmt(l_dec)}"
              f"  -> ratio {fmt(m_dec/l_dec if m_dec and l_dec else None)}x")
        print(f"  prefill: memra {fmt(m_pp,1)} ({fmt(m_np,0)} tok in {fmt(m_prime,3)}s) vs llama {fmt(l_pp,1)}"
              f"  -> ratio {fmt(m_pp/l_pp if m_pp and l_pp else None)}x")
        if me and le:
            print(f"  e2e    : memra {me:.3f}s vs llama {le:.3f}s (prime + 256/decode)"
                  f"  -> ratio {le/me:.2f}x  {'>= 1.1x BAR: PASS' if le/me >= 1.1 else '< 1.1x BAR: FAIL'}")
        sc = vals(marm, cls, "self_consistency")
        print(f"  memra self-consistency: {sum(1 for v in sc if v==1)}/{len(sc)} PASS; reps decode memra={sorted(vals(marm,cls,'decode_toks'))} llama={sorted(vals(larm,cls,'decode_toks'))}")

def kqrp():
    rows = load("kqrp-sweep.jsonl")
    if not rows:
        print("no kqrp rows"); return
    def vals(arm, metric):
        return [r["value"] for r in rows if r["arm"] == arm and r["metric"] == metric]
    print("\n=== Ornith-35B KQRP sweep (interleaved x5, board shape pp512/tg128) ===")
    for metric in ("decode_toks", "prefill_toks"):
        a, b = vals("off", metric), vals("on", metric)
        ma, mb = med(a), med(b)
        r = mb/ma if ma and mb else None
        print(f"  {metric:13s}: off {fmt(ma)} {sorted(a)} | on {fmt(mb)} {sorted(b)} | on/off {fmt(r,3)}x")
    ga, gb = vals("off", "argmax_match"), vals("on", "argmax_match")
    print(f"  argmax MATCH : off {sum(1 for v in ga if v>=1)}/{len(ga)} | on {sum(1 for v in gb if v>=1)}/{len(gb)}")
    print(f"  mirrors built: on {vals('on','mirrors_built')} | off {vals('off','mirrors_built')}")

if __name__ == "__main__":
    cell9b()
    kqrp()
