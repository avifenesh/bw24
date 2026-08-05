#!/usr/bin/env python3
"""API-key gate (lane/api-keys, 2026-08-05): multi-key tenant auth, live e2e.

Server under test: bulk tier (MEMRA_SERVE_SPEC=0 — the tier the prefix cache serves),
prefix cache ON (MEMRA_PREFIX_CACHE_MB=1024 so LRU eviction cannot masquerade as
isolation), MEMRA_API_KEYS=<keys.toml> with tenants acme (2 interactive keys), blue
(1 interactive key), bulk (1 batch-class key, rate_limit=2), dead (revoked pre-boot),
plus MEMRA_API_KEY=<single> (the back-compat daily driver). Isolation runs entirely on
INTERACTIVE-class keys so dark-lane shed/prefill behavior cannot confound the oracle.
All generation deterministic greedy (temperature 0, seed 0).

LAWS UNDER TEST:
  (1) 401 on missing/garbage key; 403 on a revoked key (distinct, actionable).
  (2) MEMRA_API_KEY still authenticates (tenant "default") — back-compat.
  (3) TWO-TENANT CACHE ISOLATION, proven by cache-hit behavior (the CacheProbe oracle):
      acme key1 seeds prompt P -> acme key2 HITS on P (same tenant, different key,
      cached_tokens == prompt_tokens > 0) -> blue MISSES on P (cached_tokens == 0)
      while blue still hits its OWN seeded prompt Q (proving the miss is isolation,
      not a dead cache) -> acme misses on Q (both directions).
  (4) cache_salt still sub-scopes WITHIN a tenant: acme@salt-x misses acme@no-salt's
      entry on P (namespaces differ), then hits itself on repeat.
  (5) Per-tenant rate-limit headers: the bulk key (rate_limit=2) reports
      x-ratelimit-limit=2, remaining=1 (this request holding one slot);
      an uncapped key reports the global lane cap.
  (6) Batch-class lane law: bulk with x-lane: interactive -> 403;
      default (no header) admits (harvest lane); judge admits (or 429-sheds).
  (7) Hot revoke: --revoke-key on acme key2 while the server runs -> 403 within the
      poll window; acme key1 unaffected.

Usage: apikey_gate.py --base URL --model NAME --out DIR
       --key-a1 K --key-a2 K --key-b K --key-bulk K --key-revoked K --single K
       [--revoke-cmd "..."] (run between phases 6 and 7)
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.error
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


def post(base, model, key, sys_text, user_text, salt=None, lane=None, timeout=900):
    body = {"model": model,
            "messages": [{"role": "system", "content": sys_text},
                         {"role": "user", "content": user_text}],
            "max_tokens": 24, "temperature": 0, "seed": 0, "stream": False}
    if salt is not None:
        body["cache_salt"] = salt
    headers = {"Content-Type": "application/json"}
    if key is not None:
        headers["Authorization"] = f"Bearer {key}"
    if lane is not None:
        headers["x-lane"] = lane
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read()), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}"), dict(e.headers)


def usage(resp):
    u = resp["usage"]
    return u["prompt_tokens"], u["prompt_tokens_details"]["cached_tokens"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--key-a1", required=True)
    ap.add_argument("--key-a2", required=True)
    ap.add_argument("--key-b", required=True)
    ap.add_argument("--key-bulk", required=True)
    ap.add_argument("--key-revoked", required=True)
    ap.add_argument("--single", required=True)
    ap.add_argument("--revoke-cmd", default=None,
                    help="shell command that revokes key-a2 (hot-reload phase)")
    args = ap.parse_args()
    rows = []
    fails = 0

    def row(gate, ok, **kw):
        nonlocal fails
        if not ok:
            fails += 1
        r = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "gate": gate,
             "verdict": "PASS" if ok else "FAIL", **kw}
        rows.append(r)
        print(f"  {'ok' if ok else 'FAIL'}: {gate} {kw}")

    # (1) auth refusals
    st, body, _ = post(args.base, args.model, None, SYS_P, Q_P)
    row("no-key-401", st == 401, status=st)
    st, body, _ = post(args.base, args.model, "mk-garbage-key", SYS_P, Q_P)
    row("bad-key-401", st == 401, status=st)
    st, body, _ = post(args.base, args.model, args.key_revoked, SYS_P, Q_P)
    row("revoked-key-403", st == 403, status=st,
        message=body.get("error", {}).get("message"))

    # (2) back-compat single key -> 200
    st, body, _ = post(args.base, args.model, args.single, SYS_P,
                       "Reply with the single word: ready.")
    row("single-key-200", st == 200, status=st)

    # (3) two-tenant isolation via cache-hit behavior
    st, r1, _ = post(args.base, args.model, args.key_a1, SYS_P, Q_P)
    p1, c1 = usage(r1)
    row("acme-k1-seed-P", st == 200 and c1 == 0, prompt=p1, cached=c1)
    st, r2, _ = post(args.base, args.model, args.key_a2, SYS_P, Q_P)
    p2, c2 = usage(r2)
    row("acme-k2-HIT-P-same-tenant", st == 200 and c2 == p2 and c2 > 0,
        prompt=p2, cached=c2)
    st, r3, _ = post(args.base, args.model, args.key_b, SYS_P, Q_P)
    p3, c3 = usage(r3)
    row("blue-MISS-P-cross-tenant", st == 200 and c3 == 0, prompt=p3, cached=c3)
    st, r4, _ = post(args.base, args.model, args.key_b, SYS_Q, Q_Q)
    p4, c4 = usage(r4)
    row("blue-seed-Q", st == 200 and c4 == 0, prompt=p4, cached=c4)
    st, r5, _ = post(args.base, args.model, args.key_b, SYS_Q, Q_Q)
    p5, c5 = usage(r5)
    row("blue-HIT-own-Q-cache-alive", st == 200 and c5 == p5 and c5 > 0,
        prompt=p5, cached=c5)
    st, r6, _ = post(args.base, args.model, args.key_a1, SYS_Q, Q_Q)
    p6, c6 = usage(r6)
    row("acme-MISS-Q-cross-tenant-reverse", st == 200 and c6 == 0,
        prompt=p6, cached=c6)

    # (4) cache_salt sub-scopes WITHIN a tenant
    st, r7, _ = post(args.base, args.model, args.key_a1, SYS_P, Q_P, salt="proj-x")
    p7, c7 = usage(r7)
    row("acme-salted-MISS-vs-unsalted", st == 200 and c7 == 0, prompt=p7, cached=c7)
    st, r8, _ = post(args.base, args.model, args.key_a1, SYS_P, Q_P, salt="proj-x")
    p8, c8 = usage(r8)
    row("acme-salted-HIT-itself", st == 200 and c8 == p8 and c8 > 0,
        prompt=p8, cached=c8)

    # (5) per-tenant rate-limit headers (key-bulk carries rate_limit=2 in the ring)
    st, _, h = post(args.base, args.model, args.key_bulk, SYS_Q,
                    "Reply with the single word: ready.")
    lim = int(h.get("x-ratelimit-limit", -1))
    rem = int(h.get("x-ratelimit-remaining", -1))
    row("tenant-rl-override-headers", st in (200, 429) and lim == 2 and rem == 1,
        status=st, limit=lim, remaining=rem)
    st, _, h = post(args.base, args.model, args.key_a1, SYS_P,
                    "Reply with the single word: ready.")
    lim_a = int(h.get("x-ratelimit-limit", -1))
    row("uncapped-key-global-cap", st == 200 and lim_a > 2, limit=lim_a)

    # (6) batch-class lane law (key-bulk is lane = "batch")
    st, body, _ = post(args.base, args.model, args.key_bulk, SYS_Q,
                       "Reply with the single word: ready.", lane="interactive")
    row("batch-key-interactive-403", st == 403, status=st,
        message=body.get("error", {}).get("message"))
    # judge lane is permitted for batch keys (may 429-shed under load; both are law-consistent)
    st, body, _ = post(args.base, args.model, args.key_bulk, SYS_Q,
                       "Reply with the single word: ready.", lane="judge")
    row("batch-key-judge-admits", st in (200, 429), status=st)

    # (7) hot revoke while the server runs
    if args.revoke_cmd:
        subprocess.run(args.revoke_cmd, shell=True, check=True)
        time.sleep(3)  # > the 2s keyring poll
        st, body, _ = post(args.base, args.model, args.key_a2, SYS_P,
                           "Reply with the single word: ready.")
        row("hot-revoke-403-within-poll", st == 403, status=st)
        st, _, _ = post(args.base, args.model, args.key_a1, SYS_P,
                        "Reply with the single word: ready.")
        row("sibling-key-survives-revoke", st == 200, status=st)

    with open(args.out + "/apikey-gates.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"apikey_gate: {fails} failed / {len(rows)} gates")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
