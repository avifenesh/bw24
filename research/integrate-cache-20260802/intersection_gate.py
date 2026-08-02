#!/usr/bin/env python3
"""integrate-cache INTERSECTION gate (2026-08-02): tools x prompt-cache on the MERGED binary.

Server under test: q35, bulk tier (MEMRA_SERVE_SPEC=0 — the tier the prefix cache serves;
spec sessions bypass it by policy), prefix cache ON (default MEMRA_PREFIX_CACHE_MB=256).
All generation deterministic greedy (temperature 0, seed 0).

THE LAW UNDER TEST: a TOOLS request is cacheable like any other request — the prefix cache
keys on the rendered prompt's token ids, and the tools block is just prompt tokens. Usage
carries ONE worker-truth prompt count (tools block included) plus the cached split; the
cache must not change prompt_tokens, and the cached bytes must not change the tool call.

Leg R (the gate as specified — "a TOOLS request repeated 3x"):
  the SAME get_weather tools request 3x.
    rep1: cold -> cached_tokens == 0, seeds the full prompt.
    rep2/rep3: FULL-prefix hits -> cached_tokens == prompt_tokens > 0.
  Every rep: finish_reason "tool_calls", get_weather{city~Paris}; prompt_tokens equals the
  tok-check count of the python-rendered prompt (worker-truth crosscheck, tools included);
  0 <= cached_tokens <= prompt_tokens. GATED identity: the parsed tool_calls + finish are
  byte-identical across reps. Full content identity (think prose included) is a REPORT
  row: cold-vs-hit prose can move at a near-tie on this tier independent of the tools
  surface — attribution probes (attribution_probe.py / run-attribution.sh) pin whether
  the class predates the merge. Law history: the first R-intersection FAIL rows in the
  JSONL are (a) a gate-script bug (tools field never sent — prompt_tokens 27 vs 330
  tok-check in the rows shows the plain render) and (b) this content-identity overreach.

Leg M (marketplace shape): same tools, same system line, THREE different user turns.
    A: miss (cached 0) — split-primes at the LCP against leg R's entry and/or seeds.
    B: hit on a shared boundary -> 0 < cached_tokens < prompt_tokens.
    C (a third distinct question) and B2 (=B re-sent): hits, cached_tokens > 0.
  Every request still parses a get_weather call and passes the tok-check crosscheck.

Usage: intersection_gate.py --base URL --model NAME --gguf PATH --tok-check PATH --out DIR
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "serve-tools-20260802"))
from render_prompt import render_prompt  # noqa: E402

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. Paris"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"],
                         "description": "Temperature unit (default celsius)"},
            },
            "required": ["city"],
        },
    },
}]

USER_Q = "What is the current weather in Paris right now? Use the tools available to you."
SYS_M = ("You are the weather desk agent for a travel marketplace. Always use the provided "
         "tools to fetch live conditions before answering, and answer concisely.")
# Leg M user turns: shared first sentence, divergence starts at a fresh sentence.
M_QA = "What is the current weather in Paris right now?\nUse the tools available to you."
M_QB = "What is the current weather in Paris right now?\nPlease call the provided tool."
M_QC = "What is the current weather in Paris right now?\nCheck it with the tool provided."


def post(base, body, timeout=900):
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def body_for(messages):
    return {"model": MODEL, "messages": messages, "tools": TOOLS, "max_tokens": 1024,
            "temperature": 0, "seed": 0, "stream": False}


def tok_count(tok_check, gguf, text):
    out = subprocess.run([tok_check, gguf, text], capture_output=True, text=True, check=True)
    for line in out.stdout.splitlines():
        if line.startswith("encode("):
            ids = line[line.index("= [") + 3:line.rindex("]")]
            return 0 if not ids.strip() else len(ids.split(","))
    raise RuntimeError(f"tok-check output unparsed: {out.stdout[:200]}")


def gen_key(resp):
    """generation identity: message content + tool_calls + finish reason (usage/timing out)."""
    ch = resp["choices"][0]
    return json.dumps({"content": ch["message"].get("content"),
                       "tool_calls": ch["message"].get("tool_calls"),
                       "finish": ch["finish_reason"]}, sort_keys=True)


def call_key(resp):
    """the specified identity: the parsed tool_calls + finish reason (think prose excluded)."""
    ch = resp["choices"][0]
    return json.dumps({"tool_calls": ch["message"].get("tool_calls"),
                       "finish": ch["finish_reason"]}, sort_keys=True)


def call_ok(resp):
    ch = resp["choices"][0]
    calls = ch["message"].get("tool_calls") or []
    if ch["finish_reason"] != "tool_calls" or len(calls) < 1:
        return False, calls
    if calls[0]["function"]["name"] != "get_weather":
        return False, calls
    argd = json.loads(calls[0]["function"]["arguments"])
    return (isinstance(argd, dict) and "paris" in str(argd.get("city", "")).lower()), calls


def usage_fields(resp):
    u = resp["usage"]
    return (u["prompt_tokens"], u["completion_tokens"], u["total_tokens"],
            u["prompt_tokens_details"]["cached_tokens"])


def main():
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tok-check", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    MODEL = args.model
    rows = []

    def row(gate, verdict, **kw):
        r = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "gate": gate,
             "verdict": verdict, **kw}
        rows.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)

    def checked(tag, messages, rep):
        resp = post(args.base, body_for(messages))
        with open(f"{args.out}/intersection-{tag}-rep{rep}.json", "w") as f:
            json.dump({"messages": messages, "response": resp}, f, indent=2,
                      ensure_ascii=False)
        ok_call, calls = call_ok(resp)
        p, c, t, cached = usage_fields(resp)
        tc = tok_count(args.tok_check, args.gguf, render_prompt(messages, TOOLS))
        ok_usage = (p == tc) and (t == p + c) and (0 <= cached <= p)
        return resp, ok_call, ok_usage, {"prompt_tokens": p, "tok_check": tc,
                                         "completion_tokens": c, "total_tokens": t,
                                         "cached_tokens": cached}

    # ---------------- Leg R: the same tools request repeated 3x ----------------
    msgs_r = [{"role": "user", "content": USER_Q}]
    reps = []
    for rep in (1, 2, 3):
        resp, ok_call, ok_usage, u = checked("legR", msgs_r, rep)
        reps.append((resp, ok_call, ok_usage, u))
        row(f"R-rep{rep}", "PASS" if ok_call and ok_usage else "FAIL", **u,
            finish=resp["choices"][0]["finish_reason"])
    u1, u2, u3 = reps[0][3], reps[1][3], reps[2][3]
    cold_ok = u1["cached_tokens"] == 0
    hit_ok = (u3["cached_tokens"] > 0 and u3["cached_tokens"] == u3["prompt_tokens"]
              and u2["cached_tokens"] == u2["prompt_tokens"])
    same_prompt_count = u1["prompt_tokens"] == u2["prompt_tokens"] == u3["prompt_tokens"]
    # THE SPECIFIED LAW: third request hits (cached > 0), the tool_call parses
    # IDENTICALLY, usage exact. Gated on the parsed call + finish identity.
    calls_eq = call_key(reps[0][0]) == call_key(reps[1][0]) == call_key(reps[2][0])
    row("R-intersection",
        "PASS" if cold_ok and hit_ok and same_prompt_count and calls_eq else "FAIL",
        rep1_cached=u1["cached_tokens"], rep2_cached=u2["cached_tokens"],
        rep3_cached=u3["cached_tokens"], prompt_tokens=u1["prompt_tokens"],
        cold_ok=cold_ok, third_hits=hit_ok, prompt_count_stable=same_prompt_count,
        tool_calls_identical=calls_eq)
    # REPORT (not gated): full content byte-identity cold-vs-hit. The full-hit decode
    # inherits the cache tier's cross-run law; a think-prose near-tie flip lands here.
    # Attribution receipts: attr-* rows (cold-vs-cold, raw-3x on merged AND pre-merge
    # lane binary — the class predates the merge if the lane binary reproduces it).
    byte_eq = gen_key(reps[0][0]) == gen_key(reps[1][0]) == gen_key(reps[2][0])
    hit_eq = gen_key(reps[1][0]) == gen_key(reps[2][0])
    row("R-content-identity", "PASS" if byte_eq else "REPORT-DIVERGED", gated=False,
        cold_vs_hit_identical=byte_eq, hit_vs_hit_identical=hit_eq)

    # ---------------- Leg M: shared system+tools prefix, distinct user turns ----------------
    def m_msgs(q):
        return [{"role": "system", "content": SYS_M}, {"role": "user", "content": q}]

    ra, oka_c, oka_u, ua = checked("legM-A", m_msgs(M_QA), 1)
    row("M-A-cold", "PASS" if oka_c and oka_u and ua["cached_tokens"] == 0 else "FAIL", **ua)
    rb, okb_c, okb_u, ub = checked("legM-B", m_msgs(M_QB), 1)
    row("M-B-hit", "PASS" if okb_c and okb_u
        and 0 < ub["cached_tokens"] < ub["prompt_tokens"] else "FAIL", **ub)
    rc, okc_c, okc_u, uc = checked("legM-C", m_msgs(M_QC), 1)
    row("M-C-hit", "PASS" if okc_c and okc_u and uc["cached_tokens"] > 0 else "FAIL", **uc)
    rb2, okb2_c, okb2_u, ub2 = checked("legM-B", m_msgs(M_QB), 2)
    eq_b = gen_key(rb) == gen_key(rb2)
    row("M-B2-hit-identical", "PASS" if okb2_c and okb2_u and ub2["cached_tokens"] > 0
        and eq_b else "FAIL", **ub2, generation_matches_B=eq_b)

    with open(f"{args.out}/intersection-gates.jsonl", "a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    gated = [r for r in rows if r.get("gated", True)]
    fails = [r for r in gated if r["verdict"] != "PASS"]
    print(f"[intersection] {len(gated) - len(fails)}/{len(gated)} gated rows PASS "
          f"({len(rows) - len(gated)} report-only rows)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
