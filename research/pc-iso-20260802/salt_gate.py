#!/usr/bin/env python3
"""PC-ISO salt gate (lane/pc-iso, 2026-08-02): per-tenant prefix-cache isolation, e2e.

Server under test: bulk tier (MEMRA_SERVE_SPEC=0 — the tier the prefix cache serves),
prefix cache ON (MEMRA_PREFIX_CACHE_MB=1024 so LRU eviction cannot masquerade as
isolation). All generation deterministic greedy (temperature 0, seed 0). No tools —
the intersection gate (research/integrate-cache-20260802/intersection_gate.py, run
unmodified) already covers the tools x cache x NO-SALT surface; this gate covers the
namespace laws on top.

THE LAWS UNDER TEST (requirements a/b of the PC-ISO spec; vLLM cache_salt design):
  (a) same `cache_salt` + same prompt  -> the second request HITS
      (cached_tokens == prompt_tokens > 0);
  (b) different `cache_salt` + same prompt -> MISS in BOTH directions
      (tenant A's seeded entry is invisible to tenant B on prompt P, and tenant B's
      seeded entry is invisible to tenant A on prompt Q), while each tenant still hits
      its OWN entry (proving the misses are isolation, not a dead cache);
  (+) the no-salt default namespace ("") sees NEITHER tenant's entries and still
      caches for itself — single-tenant behavior preserved.

The `cached_tokens` billing field is the oracle CacheProbe (arXiv 2605.30613) used to
prove cross-account leakage on OpenRouter default-mode providers; after PC-ISO it may
only ever reflect the caller's own namespace's history — which is exactly what the
cross-miss rows assert.

Usage: salt_gate.py --base URL --model NAME --out DIR
"""

import argparse
import json
import sys
import time
import urllib.request

SYS_P = ("You are the routing desk agent for a logistics marketplace. Follow house "
         "style: answer in at most two short sentences, never speculate about customs "
         "delays, always quote transit times in business days, and prefer rail over "
         "road when both are available at the same price. If a request is ambiguous, "
         "ask exactly one clarifying question instead of guessing.")
Q_P = "How long does a pallet from Rotterdam to Milan usually take?"

SYS_Q = ("You are the returns desk agent for an electronics retailer. Follow house "
         "style: answer in at most two short sentences, cite the 30-day return window "
         "where relevant, never promise refunds before inspection, and escalate any "
         "battery-damage mention to the safety queue. If a request is ambiguous, ask "
         "exactly one clarifying question instead of guessing.")
Q_Q = "A customer wants to return a laptop bought five weeks ago. What do I tell them?"


def post(base, model, sys_text, user_text, salt, timeout=900):
    body = {"model": model,
            "messages": [{"role": "system", "content": sys_text},
                         {"role": "user", "content": user_text}],
            "max_tokens": 24, "temperature": 0, "seed": 0, "stream": False}
    if salt is not None:
        body["cache_salt"] = salt
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def usage(resp):
    u = resp["usage"]
    return u["prompt_tokens"], u["prompt_tokens_details"]["cached_tokens"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = []

    def row(gate, verdict, **kw):
        r = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "gate": gate,
             "verdict": verdict, **kw}
        rows.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)

    def step(tag, sys_text, user_text, salt, want):
        """want: 'cold' (cached == 0), 'full-hit' (cached == prompt > 0), 'hit' (> 0)."""
        resp = post(args.base, args.model, sys_text, user_text, salt)
        with open(f"{args.out}/salt-{tag}.json", "w") as f:
            json.dump({"salt": salt, "response": resp}, f, indent=2, ensure_ascii=False)
        p, cached = usage(resp)
        ok = {"cold": cached == 0,
              "full-hit": 0 < cached == p,
              "hit": cached > 0}[want] and p > 0
        row(tag, "PASS" if ok else "FAIL", salt=salt, want=want,
            prompt_tokens=p, cached_tokens=cached)
        return p, cached

    # ---- law (a): same salt + same prefix -> hit ----
    pa1, _ = step("A1-P-cold", SYS_P, Q_P, "tenant-a", "cold")
    pa2, _ = step("A2-P-same-salt-hit", SYS_P, Q_P, "tenant-a", "full-hit")

    # ---- law (b) direction A->B: A's entry invisible to B on the SAME prompt ----
    pb1, _ = step("B1-P-cross-salt-miss", SYS_P, Q_P, "tenant-b", "cold")
    step("B2-P-own-salt-hit", SYS_P, Q_P, "tenant-b", "full-hit")

    # ---- law (b) direction B->A: B seeds a fresh prompt; A must miss it ----
    step("B3-Q-cold", SYS_Q, Q_Q, "tenant-b", "cold")
    step("A3-Q-cross-salt-miss", SYS_Q, Q_Q, "tenant-a", "cold")
    step("A4-Q-own-salt-hit", SYS_Q, Q_Q, "tenant-a", "full-hit")

    # ---- default namespace: no salt sees neither tenant, still caches for itself ----
    step("N1-P-nosalt-miss", SYS_P, Q_P, None, "cold")
    step("N2-P-nosalt-hit", SYS_P, Q_P, None, "full-hit")

    # same prompt -> same worker-truth prompt count, salt or not (the salt must never
    # change tokenization/rendering — it is a cache key, not prompt bytes).
    row("P-prompt-count-stable", "PASS" if pa1 == pa2 == pb1 else "FAIL",
        a1=pa1, a2=pa2, b1=pb1)

    with open(f"{args.out}/salt-gates.jsonl", "a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    fails = [r for r in rows if r["verdict"] != "PASS"]
    print(f"[pc-iso salt gate] {len(rows) - len(fails)}/{len(rows)} rows PASS")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
