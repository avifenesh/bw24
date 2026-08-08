#!/usr/bin/env python3
"""perf_ab.py — metering-overhead A/B (lane/cache-metering, 2026-08-07).

Bar: the metering adds no measurable serve overhead (< 0.5% p95 request latency).

Design (the interleaving law — cross-run comparisons are clock-drift-invalid):
two servers resident on one GPU, BASE = the pre-lane binary (main @ e54dd2e6),
METER = the instrumented one. Both boot the same model with the same env
(compat, spec off, prefix cache default). 5 reps; each rep runs one 20-request
batch per arm, order alternating (AB, BA, AB, ...) so thermal/clock drift hits
both arms symmetrically. Workload = the metering HOT path: 256-token shared
prefix + unique suffix + 64 generated tokens, one salt per arm, so steady-state
requests are prefix-cache hits (agent-traffic shape) and every request crosses
admit-metering, LCP recording, and the publish-on-retire.

Output: per-rep p50/p95 per arm + pooled p95 delta, JSONL rows for every
request (raw receipts). Requests are sequential (c=1) — the idle server runs no
kernels, so residence is not contention.
"""
import json
import sys
import time
import urllib.request

BASE_URL = sys.argv[1]   # pre-lane binary
METER_URL = sys.argv[2]  # instrumented binary
OUT = sys.argv[3]        # raw JSONL
REPS, BATCH, K, S, GEN = 5, 20, 256, 16, 64

PREFIX = list(range(2000, 2000 + K))
uniq = [0]


def ask(url: str, salt: str) -> float:
    uniq[0] += 1
    body = {"model": "smoke",
            "prompt_ids": PREFIX + list(range(50000 + uniq[0] * S,
                                              50000 + (uniq[0] + 1) * S)),
            "max_tokens": GEN, "temperature": 0, "cache_salt": salt}
    req = urllib.request.Request(f"{url}/v1/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=300) as f:
        json.load(f)
    return time.monotonic() - t0


def pctl(xs, p):
    ys = sorted(xs)
    i = min(len(ys) - 1, max(0, round(p / 100 * (len(ys) - 1))))
    return ys[i]


def main():
    arms = {"base": BASE_URL, "meter": METER_URL}
    lat = {"base": [], "meter": []}
    raw = open(OUT, "w")
    # warmup: seed each arm's prefix cache (seed + split + first hit) — steady state after.
    for name, url in arms.items():
        for _ in range(3):
            ask(url, f"perf-{name}")
    for rep in range(REPS):
        order = ["base", "meter"] if rep % 2 == 0 else ["meter", "base"]
        rep_lat = {a: [] for a in arms}
        for arm in order:
            for _ in range(BATCH):
                dt = ask(arms[arm], f"perf-{arm}")
                rep_lat[arm].append(dt)
                lat[arm].append(dt)
                raw.write(json.dumps({"rep": rep, "arm": arm,
                                      "elapsed_s": round(dt, 5)}) + "\n")
        print(f"rep {rep} ({'->'.join(order)}): " + "  ".join(
            f"{a}: p50 {pctl(rep_lat[a], 50) * 1e3:.1f}ms "
            f"p95 {pctl(rep_lat[a], 95) * 1e3:.1f}ms" for a in arms))
    raw.close()
    b95, m95 = pctl(lat["base"], 95), pctl(lat["meter"], 95)
    b50, m50 = pctl(lat["base"], 50), pctl(lat["meter"], 50)
    d95 = (m95 - b95) / b95 * 100
    d50 = (m50 - b50) / b50 * 100
    verdict = "PASS" if d95 < 0.5 else "FAIL"
    print(f"pooled N={len(lat['base'])}/arm interleaved x{REPS}: "
          f"p50 base {b50 * 1e3:.1f}ms meter {m50 * 1e3:.1f}ms ({d50:+.2f}%), "
          f"p95 base {b95 * 1e3:.1f}ms meter {m95 * 1e3:.1f}ms ({d95:+.2f}%) "
          f"-> {verdict} (<0.5% p95 bar)")
    sys.exit(0 if d95 < 0.5 else 1)


if __name__ == "__main__":
    main()
