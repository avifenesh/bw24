#!/usr/bin/env python3
"""agentic_load: prompt-cache TTFT + throughput harness (lane/prompt-cache, 2026-08-02).

Agentic-shaped marketplace workload: ONE shared ~1.9k-token system prompt, varying user
turns, c=8. Four waves of 8 distinct users per pass (32 requests). Wave semantics on a
cache-ON server: wave 1 = cold fill (seeds), wave 2 = LCP split learning, waves 3-4 =
steady-state hits. On a cache-OFF server every wave pays full prefill.

Measures per request (streaming): TTFT (POST -> first content chunk), wall, completion
tokens, usage.prompt_tokens / prompt_tokens_details.cached_tokens (worker truth).
Emits one JSONL row per request + per-wave and per-pass summaries.

modes:
  load  --label X : the c=8 wave workload
  audit           : the 3-request as-is demonstration (same sys re-sent = what is skipped?)
                    R1 [sys,u1] cold; R2 [sys,u2] shared-prefix NEW session;
                    R3 [sys,u1,assistant,u3] exact continuation (the only as-is hit class)
"""

import argparse
import json
import threading
import time
import urllib.request

WORDS = ("orchestrate the fleet with bounded budgets and deterministic decode paths "
         "every request is metered per token and cached prefixes bill at a discount "
         "route traffic by least outstanding admission and never exceed the session cap "
         "verify outputs byte for byte before publishing any performance claim").split()


def system_prompt(n_words=1400):
    out = ["You are the memra marketplace serving agent. Operating rules follow."]
    for w in range(n_words):
        out.append(WORDS[(w * 11 + 5) % len(WORDS)])
        if w % 17 == 16:
            out[-1] += "."
    out.append("Follow every rule above and answer each user concisely.")
    return " ".join(out)


USERS = [
    "Summarize your rules in one sentence.",
    "What is the admission policy? Answer briefly.",
    "Explain how cached prefixes are billed.",
    "List two invariants you must never violate.",
    "How do you verify outputs before publishing?",
    "Describe the session cap in one sentence.",
    "What does least outstanding routing mean?",
    "Why are decode paths deterministic here?",
    "Explain per token metering in two sentences.",
    "What discount applies to cached prefixes?",
    "How is fleet traffic bounded? Be brief.",
    "State one rule about performance claims.",
    "What happens when the session cap is hit?",
    "Summarize the metering pipeline briefly.",
    "How are budgets enforced per request?",
    "Name the two phases of serving a prompt.",
    "When is a prefix eligible for caching?",
    "What proves a cache hit was correct?",
    "Describe cold start behavior in one line.",
    "How should evictions be prioritized?",
    "What telemetry do you expose for billing?",
    "Explain wave scheduling in two sentences.",
    "What is the retry policy under pressure?",
    "How do you handle an unknown model name?",
    "State the rule for byte identical outputs.",
    "What bounds time to first token here?",
    "Describe one failure mode of stale cache.",
    "How are concurrent sessions isolated?",
    "What is measured before a claim ships?",
    "Summarize the discount ledger in a line.",
    "Why must budgets be explicit knobs?",
    "What ends a session normally?",
]


def stream_chat(base, model, messages, max_tokens, out, idx, timeout=900):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0, "seed": 0, "stream": True}
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    ttft = None
    n_chunks = 0
    usage = {}
    text = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                d = json.loads(payload)
                if "error" in d:
                    out[idx] = {"err": d["error"]}
                    return
                delta = (d.get("choices") or [{}])[0].get("delta", {}).get("content")
                if delta:
                    if ttft is None:
                        ttft = time.monotonic() - t0
                    n_chunks += 1
                    text.append(delta)
                if d.get("usage"):
                    usage = d["usage"]
    except Exception as e:  # noqa: BLE001 — quoted, not inferred
        out[idx] = {"err": f"{type(e).__name__}: {e}"}
        return
    wall = time.monotonic() - t0
    out[idx] = {
        "ttft_s": ttft, "wall_s": wall, "chunks": n_chunks,
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "text_head": "".join(text)[:60],
        "text_full": "".join(text),
    }


def pctl(v, q):
    if not v:
        return None
    v = sorted(v)
    i = round(q / 100 * (len(v) - 1))
    return v[min(i, len(v) - 1)]


def run_load(base, model, label, outpath):
    sysp = system_prompt()
    rows = []
    pass_t0 = time.monotonic()
    wave_summaries = []
    for wave in range(4):
        users = USERS[wave * 8:(wave + 1) * 8]
        msgs = [[{"role": "system", "content": sysp}, {"role": "user", "content": u}]
                for u in users]
        out = [None] * 8
        threads = [threading.Thread(target=stream_chat,
                                    args=(base, model, msgs[i], 96, out, i))
                   for i in range(8)]
        w0 = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wwall = time.monotonic() - w0
        errs = [o for o in out if o and "err" in o]
        oks = [o for o in out if o and "err" not in o]
        for i, o in enumerate(out):
            row = {"label": label, "wave": wave, "req": i, **(o or {"err": "no result"})}
            row.pop("text_full", None)  # keep the JSONL lean; text_head is the sample
            rows.append(row)
        gen = sum(o.get("completion_tokens") or 0 for o in oks)
        ptoks = sum(o.get("prompt_tokens") or 0 for o in oks)
        ctoks = sum(o.get("cached_tokens") or 0 for o in oks)
        ws = {"label": label, "wave": wave, "n_ok": len(oks), "n_err": len(errs),
              "wall_s": round(wwall, 3),
              "ttft_p50_s": pctl([o["ttft_s"] for o in oks if o["ttft_s"]], 50),
              "ttft_p95_s": pctl([o["ttft_s"] for o in oks if o["ttft_s"]], 95),
              "gen_toks": gen, "agg_tok_s": round(gen / wwall, 1) if wwall else None,
              "prompt_tokens": ptoks, "cached_tokens": ctoks,
              "cached_frac": round(ctoks / ptoks, 4) if ptoks else None,
              "errors": [e["err"] for e in errs][:4]}
        wave_summaries.append(ws)
        print(json.dumps(ws))
    pass_wall = time.monotonic() - pass_t0
    oks = [r for r in rows if "err" not in r]
    steady = [r for r in rows if "err" not in r and r["wave"] >= 2]
    ptoks = sum(r.get("prompt_tokens") or 0 for r in oks)
    ctoks = sum(r.get("cached_tokens") or 0 for r in oks)
    summary = {"label": label, "kind": "pass-summary", "pass_wall_s": round(pass_wall, 3),
               "n_ok": len(oks), "n_err": len(rows) - len(oks),
               "ttft_p50_all_s": pctl([r["ttft_s"] for r in oks if r["ttft_s"]], 50),
               "ttft_p50_steady_s": pctl([r["ttft_s"] for r in steady if r["ttft_s"]], 50),
               "ttft_p95_steady_s": pctl([r["ttft_s"] for r in steady if r["ttft_s"]], 95),
               "gen_toks": sum(r.get("completion_tokens") or 0 for r in oks),
               "agg_tok_s": round(sum(r.get("completion_tokens") or 0 for r in oks) / pass_wall, 1),
               "prompt_tokens": ptoks, "cached_tokens": ctoks,
               "cached_frac": round(ctoks / ptoks, 4) if ptoks else None}
    print(json.dumps(summary))
    if outpath:
        with open(outpath, "a") as f:
            for r in rows + wave_summaries + [summary]:
                f.write(json.dumps(r) + "\n")
    return 0 if summary["n_err"] == 0 else 1


def run_audit(base, model, label, outpath):
    sysp = system_prompt()
    rows = []

    def one(tag, messages):
        out = [None]
        stream_chat(base, model, messages, 96, out, 0)
        full = (out[0] or {}).pop("text_full", "") if out[0] else ""
        row = {"label": label, "req": tag, **(out[0] or {"err": "no result"})}
        rows.append(row)
        print(json.dumps(row))
        return row, full

    r1, r1_text = one("R1-cold", [{"role": "system", "content": sysp},
                                  {"role": "user", "content": USERS[0]}])
    one("R2-shared-prefix-new-session", [{"role": "system", "content": sysp},
                                         {"role": "user", "content": USERS[1]}])
    # third distinct-suffix session: cache-ON servers hit the learned prefix here;
    # as-is / cache-OFF servers pay full prefill again (the gap, quantified)
    one("R2b-shared-prefix-third-session", [{"role": "system", "content": sysp},
                                            {"role": "user", "content": USERS[3]}])
    if "err" not in r1 and r1_text:
        # exact continuation: full history including the assistant reply verbatim —
        # the ONLY hit class that exists without the cross-request prefix cache
        one("R3-exact-continuation",
            [{"role": "system", "content": sysp},
             {"role": "user", "content": USERS[0]},
             {"role": "assistant", "content": r1_text},
             {"role": "user", "content": USERS[2]}])
    if outpath:
        with open(outpath, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["load", "audit"])
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", default="m")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.mode == "load":
        raise SystemExit(run_load(args.base, args.model, args.label, args.out))
    raise SystemExit(run_audit(args.base, args.model, args.label, args.out))


if __name__ == "__main__":
    main()
