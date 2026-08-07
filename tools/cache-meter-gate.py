#!/usr/bin/env python3
"""cache-meter-gate.py — prefix-cache accounting exactness gate (lane/cache-metering).

Synthetic shared-prefix workload against a LIVE memra-server (boot it with
MEMRA_SERVE_SPEC=0 — spec sessions bypass the prefix cache by policy, so the
gate must run the batched bulk tier the cache serves; prefix cache at its
default budget). Asserts the accounting is EXACT, not merely plausible:

  workload: N sequential /v1/completions with prompt_ids = K shared prefix
  tokens + S unique suffix tokens (namespace salt A), then 1 request with the
  SAME K-prefix under salt B.

  learning sequence (docs/SERVING.md "Prompt caching"):
    A-req1  cold seed        -> usage cached_tokens == 0
    A-req2  miss + LCP split -> usage cached_tokens == 0
    A-req3..N hit at K       -> usage cached_tokens == K, prompt_tokens == K+S
    B-req1  cold (PC-ISO)    -> usage cached_tokens == 0 despite the shared K

  /metrics afterwards must carry the closed-form totals:
    prompt_tokens_in  == (N+1)*(K+S)
    cached_tokens_in  == (N-2)*K
    computed_tokens_in == prompt - cached
    cache_hit_token_ratio == cached/prompt (float-equal to 1e-9)
    prefix_cache_{hits,misses,inserts} == N-2, 3, 3
    prefix_cache_hit_tokens == (N-2)*K
    lcp_histogram: 2 probes in bucket [0] (A-req1, B-req1: best LCP 0) and
      N-1 probes in K's bucket — K=256 by default, i.e. inside the tick-seg
      [64,512) window (buckets 4..=6)
    tenants: {A: [(N)*(K+S), (N-2)*K], B: [(K+S), 0]}

  finally tools/cache_economics.py runs on the scrape and its
  revenue_multiplier must equal prompt/computed (factor 1.0) exactly.

Usage: cache-meter-gate.py BASE_URL MODEL [--n 5] [--k 256] [--suffix 16]
Exit 0 = every assertion held; first failure prints and exits 1.
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

FAILS = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global FAILS
    if ok:
        print(f"  ok: {name}")
    else:
        FAILS += 1
        print(f"  FAIL: {name}{' — ' + detail if detail else ''}")


def post(base: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{base}/v1/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as f:
        return json.load(f)


def scrape(base: str) -> dict:
    with urllib.request.urlopen(f"{base}/metrics", timeout=10) as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("model")
    ap.add_argument("--n", type=int, default=5, help="requests in namespace A (>=3)")
    ap.add_argument("--k", type=int, default=256,
                    help="shared prefix tokens (default 256: inside the tick-seg "
                         "[64,512) LCP window)")
    ap.add_argument("--suffix", type=int, default=16, help="unique suffix tokens")
    ap.add_argument("--raw-out", help="write per-request responses + the scrape here (JSONL)")
    args = ap.parse_args()
    n, k, s = args.n, args.k, args.suffix
    assert n >= 3 and k >= 64 and s >= 1

    # token ids: arbitrary but stable, well inside any vocab; suffixes disjoint per request.
    prefix = list(range(2000, 2000 + k))
    raw = open(args.raw_out, "w") if args.raw_out else None

    def run(salt: str, i: int) -> dict:
        body = {"model": args.model,
                "prompt_ids": prefix + list(range(5000 + i * s, 5000 + (i + 1) * s)),
                "max_tokens": 8, "temperature": 0, "cache_salt": salt}
        t0 = time.monotonic()
        r = post(args.base, body)
        # compat mode carries the OpenAI usage object; the native /v1/completions shape
        # carries the same worker truth as flat prompt_tokens/cached_tokens fields.
        u = r.get("usage") or {
            "prompt_tokens": r["prompt_tokens"],
            "prompt_tokens_details": {"cached_tokens": r["cached_tokens"]},
        }
        if raw:
            raw.write(json.dumps({"salt": salt, "i": i,
                                  "elapsed_s": round(time.monotonic() - t0, 4),
                                  "usage": u}) + "\n")
        return u

    # ---- per-request exactness (deliverable 1's receipt) ----
    expect = [0, 0] + [k] * (n - 2)          # namespace A learning sequence
    for i, want in enumerate(expect):
        u = run("meter-A", i)
        got = u["prompt_tokens_details"]["cached_tokens"]
        check(f"A-req{i + 1} cached_tokens == {want}", got == want, f"got {got}")
        check(f"A-req{i + 1} prompt_tokens == {k + s}",
              u["prompt_tokens"] == k + s, f"got {u['prompt_tokens']}")
    # PC-ISO composition: same K-prefix, different salt -> structurally cold.
    u = run("meter-B", n)
    got = u["prompt_tokens_details"]["cached_tokens"]
    check("B-req1 cached_tokens == 0 (cross-salt blindness)", got == 0, f"got {got}")

    # ---- aggregate /metrics exactness (deliverable 2's receipt) ----
    # publish-on-retire makes the scrape current as soon as the last retire lands;
    # tiny race between the client's Done and the worker's end-of-tick publish, so retry.
    total_p, total_c = (n + 1) * (k + s), (n - 2) * k
    m = {}
    for _ in range(20):
        m = scrape(args.base)
        if m.get("prompt_tokens_in") == total_p and m.get("cached_tokens_in") == total_c:
            break
        time.sleep(0.2)
    if raw:
        raw.write(json.dumps({"metrics": m}) + "\n")
        raw.close()
    check(f"prompt_tokens_in == {total_p}", m.get("prompt_tokens_in") == total_p,
          f"got {m.get('prompt_tokens_in')}")
    check(f"cached_tokens_in == {total_c}", m.get("cached_tokens_in") == total_c,
          f"got {m.get('cached_tokens_in')}")
    check("computed_tokens_in == prompt - cached",
          m.get("computed_tokens_in") == total_p - total_c,
          f"got {m.get('computed_tokens_in')}")
    ratio = m.get("cache_hit_token_ratio", -1)
    check("cache_hit_token_ratio matches arithmetic",
          abs(ratio - total_c / total_p) < 1e-9, f"got {ratio}")
    check("prefix_cache_hits == N-2", m.get("prefix_cache_hits") == n - 2,
          f"got {m.get('prefix_cache_hits')}")
    check("prefix_cache_misses == 3", m.get("prefix_cache_misses") == 3,
          f"got {m.get('prefix_cache_misses')}")
    check("prefix_cache_inserts == 3 (A seed + A split + B seed)",
          m.get("prefix_cache_inserts") == 3, f"got {m.get('prefix_cache_inserts')}")
    check(f"prefix_cache_hit_tokens == {total_c}",
          m.get("prefix_cache_hit_tokens") == total_c,
          f"got {m.get('prefix_cache_hit_tokens')}")

    h = m.get("lcp_histogram", {})
    edges, counts = h.get("edges", []), h.get("counts", [])
    kb = max(i for i, e in enumerate(edges) if k >= e) if edges else -1
    check("lcp_histogram: 6 probes total", sum(counts) == n + 1, f"got {sum(counts)}")
    check("lcp_histogram: 2 cold probes in bucket [0]",
          counts and counts[0] == 2, f"got {counts[:1]}")
    check(f"lcp_histogram: {n - 1} probes in K's bucket (edge {edges[kb] if kb >= 0 else '?'})",
          kb >= 0 and counts[kb] == n - 1, f"got {counts[kb] if kb >= 0 else None}")
    if 64 <= k < 512:
        window = sum(c for e, c in zip(edges, counts) if 64 <= e < 512)
        check("tick-seg [64,512) window carries the shared-prefix probes",
              window == n - 1, f"got {window}")

    tenants = m.get("tenants", {})
    ta, tb = tenants.get("meter-A", {}), tenants.get("meter-B", {})
    check("tenants[meter-A] split exact",
          ta.get("prompt_tokens_in") == n * (k + s)
          and ta.get("cached_tokens_in") == total_c, f"got {ta}")
    check("tenants[meter-B] split exact (0 cached)",
          tb.get("prompt_tokens_in") == k + s and tb.get("cached_tokens_in") == 0,
          f"got {tb}")

    # ---- the economics row crosschecks the same scrape (deliverable 3) ----
    econ = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("cache_economics.py")),
         "/dev/stdin"], input=json.dumps(m), capture_output=True, text=True)
    if econ.returncode != 0:
        check("cache_economics.py runs on the scrape", False, econ.stderr.strip())
    else:
        row = json.loads(econ.stdout)
        want_mult = round(total_p / (total_p - total_c), 4)
        check(f"economics revenue_multiplier == {want_mult} (factor 1.0)",
              row.get("revenue_multiplier") == want_mult,
              f"got {row.get('revenue_multiplier')}")

    print(f"cache-meter-gate: {FAILS} failed")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
