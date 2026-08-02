#!/usr/bin/env python3
"""agentworld bar-cell summarizer: plain-vs-llama + best-vs-best medians from aw-cell.jsonl.

Both cell families come out of the same interleaved session: the memra run-spec
invocation carries plain [generate] AND spec [generate_spec K] in-process; llama arm is
plain llama-completion (its per-class best on this NextN-less GGUF).
e2e convention = o9b-cell: prime/prompt-eval wall + 256/decode.
"""
import json, statistics, os

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

rows = load("aw-cell.jsonl")
classes = ["p1-code-short", "p2-code-medium", "p3-agentic-long"]

def vals(arm, cls, metric):
    return [r["value"] for r in rows if r["arm"] == arm and r["class"] == cls and r["metric"] == metric]

temps = [r["temp_c"] for r in rows]
print(f"=== AgentWorld-35B UD-Q4_K_M bar cells (N=3 medians, interleaved same-session; "
      f"temps {min(temps)}-{max(temps)} C) ===" if temps else "no rows")

for cls in classes:
    m_dec = med(vals("memra-spec", cls, "decode_toks"))
    m_plain = med(vals("memra-spec", cls, "plain_decode_toks"))
    m_np = med(vals("memra-spec", cls, "n_prompt"))
    m_prime = med(vals("memra-spec", cls, "prime_spec_s"))
    m_prime_p = med(vals("memra-spec", cls, "prime_plain_s"))
    m_pp = med(vals("memra-spec", cls, "prefill_toks"))
    m_acc = med(vals("memra-spec", cls, "accept_pct"))
    m_k = med(vals("memra-spec", cls, "spec_k"))
    l_dec = med(vals("llama-plain", cls, "decode_toks"))
    l_pp = med(vals("llama-plain", cls, "prefill_toks"))
    m_e2e_best = [ps + 256.0/d for ps, d in zip(vals("memra-spec", cls, "prime_spec_s"),
                                                vals("memra-spec", cls, "decode_toks"))]
    m_e2e_plain = [ps + 256.0/d for ps, d in zip(vals("memra-spec", cls, "prime_plain_s"),
                                                 vals("memra-spec", cls, "plain_decode_toks"))]
    l_e2e = [ps + 256.0/d for ps, d in zip(vals("llama-plain", cls, "prefill_s"),
                                           vals("llama-plain", cls, "decode_toks"))]
    mb, mp, le = med(m_e2e_best), med(m_e2e_plain), med(l_e2e)
    print(f"\n[{cls}] (prompt {fmt(m_np,0)} tok)")
    print(f"  PLAIN  decode : memra {fmt(m_plain)} vs llama {fmt(l_dec)}"
          f" -> {fmt(m_plain/l_dec if m_plain and l_dec else None)}x")
    print(f"  PLAIN  prefill: memra {fmt((m_np/m_prime_p) if m_np and m_prime_p else None,1)}"
          f" vs llama {fmt(l_pp,1)}"
          f" -> {fmt((m_np/m_prime_p)/l_pp if m_np and m_prime_p and l_pp else None)}x")
    if mp and le:
        print(f"  PLAIN  e2e    : memra {mp:.3f}s vs llama {le:.3f}s -> {le/mp:.2f}x")
    print(f"  BEST   decode : memra spec K={fmt(m_k,0)} {fmt(m_dec)} (acc {fmt(m_acc,1)}%)"
          f" vs llama plain {fmt(l_dec)} -> {fmt(m_dec/l_dec if m_dec and l_dec else None)}x")
    print(f"  BEST   prefill: memra {fmt(m_pp,1)} ({fmt(m_np,0)} tok in {fmt(m_prime,3)}s)"
          f" vs llama {fmt(l_pp,1)} -> {fmt(m_pp/l_pp if m_pp and l_pp else None)}x")
    if mb and le:
        verdict = ">= 1.1x BAR: PASS" if le/mb >= 1.1 else "< 1.1x BAR: FAIL"
        print(f"  BEST   e2e    : memra {mb:.3f}s vs llama {le:.3f}s (prime + 256/decode)"
              f" -> {le/mb:.2f}x  {verdict}")
    sc = vals("memra-spec", cls, "self_consistency")
    print(f"  memra self-consistency: {sum(1 for v in sc if v == 1)}/{len(sc)} PASS; "
          f"reps: memra spec={sorted(vals('memra-spec', cls, 'decode_toks'))} "
          f"plain={sorted(vals('memra-spec', cls, 'plain_decode_toks'))} "
          f"llama={sorted(vals('llama-plain', cls, 'decode_toks'))}")
