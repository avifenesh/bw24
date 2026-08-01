#!/usr/bin/env python3
"""load-serve: concurrent OpenAI-format load harness for memra-server / serve-proxy.

darklanes serving v1 (2026-08-01). Fires `--concurrency` simultaneous workers, each looping
POST /v1/chat/completions (streaming off, temperature 0 by default off — load prompts use
temperature 0.7 + per-request seed so batched decode exercises realistic divergent sequences;
pass --greedy for temperature=0 determinism checks). Prompt is ~200 tokens, max_tokens=128.

Measures per-request wall latency and completion_tokens from the server's usage block.
Reports aggregate output tok/s (sum completion_tokens / wall window), p50/p95 latency,
error count. Emits one JSON line per load point to --out (append), one line per request
with --per-request.

Usage:
  python3 load-serve.py --base http://127.0.0.1:8085 --concurrency 8 --requests 32 \
      --model qwen --out points.jsonl --label single-8085
"""

import argparse
import json
import statistics
import threading
import time
import urllib.error
import urllib.request

# ~200 tokens of prompt: a fixed instruction + repeated filler clause (repeats tokenize
# steadily, so the prompt length is stable across replicas and runs).
FILLER = ("The quick brown fox jumps over the lazy dog while the seasoned engineer "
          "measures throughput, latency, and saturation across every replica. ")
PROMPT = ("Summarize the operational state of a GPU serving cluster in exactly three "
          "sentences, then list four risks. Context follows. " + FILLER * 8)


def one_request(base, model, max_tokens, greedy, seed, timeout):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.0 if greedy else 0.7,
        "seed": seed,
        "stream": False,
    }
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.load(r)
        dt = time.monotonic() - t0
        usage = data.get("usage", {})
        return {
            "ok": True,
            "latency_s": dt,
            "completion_tokens": usage.get("completion_tokens", 0),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "finish_reason": data["choices"][0].get("finish_reason"),
            "text": data["choices"][0]["message"]["content"],
        }
    except Exception as e:
        detail = ""
        if isinstance(e, urllib.error.HTTPError):
            try:
                detail = e.read()[:300].decode(errors="replace")
            except Exception:
                pass
        return {"ok": False, "latency_s": time.monotonic() - t0,
                "error": f"{type(e).__name__}: {e} {detail}".strip()}


def run_point(base, model, concurrency, requests, max_tokens, greedy, timeout):
    results = []
    rlock = threading.Lock()
    idx = {"n": 0}

    def worker(wid):
        while True:
            with rlock:
                if idx["n"] >= requests:
                    return
                my = idx["n"]
                idx["n"] += 1
            res = one_request(base, model, max_tokens, greedy, seed=1000 + my, timeout=timeout)
            res["worker"] = wid
            res["req_index"] = my
            with rlock:
                results.append(res)

    t0 = time.monotonic()
    threads = [threading.Thread(target=worker, args=(w,)) for w in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.monotonic() - t0

    oks = [r for r in results if r["ok"]]
    errs = [r for r in results if not r["ok"]]
    lats = sorted(r["latency_s"] for r in oks)
    tot_completion = sum(r["completion_tokens"] for r in oks)

    def pct(p):
        if not lats:
            return None
        k = min(len(lats) - 1, max(0, int(round(p / 100 * (len(lats) - 1)))))
        return lats[k]

    return {
        "wall_s": wall,
        "n_ok": len(oks),
        "n_err": len(errs),
        "completion_tokens_total": tot_completion,
        "agg_tok_s": tot_completion / wall if wall > 0 else 0.0,
        "req_per_s": len(oks) / wall if wall > 0 else 0.0,
        "lat_p50_s": pct(50),
        "lat_p95_s": pct(95),
        "lat_mean_s": statistics.mean(lats) if lats else None,
        "lat_max_s": lats[-1] if lats else None,
        "errors_sample": [e["error"] for e in errs[:3]],
    }, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="server base URL")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--requests", type=int, default=None,
                    help="total requests (default 4x concurrency, min 8)")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--greedy", action="store_true", help="temperature=0")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=None, help="append summary JSONL here")
    ap.add_argument("--per-request", default=None, help="append per-request JSONL here")
    ap.add_argument("--warmup", type=int, default=1,
                    help="warmup requests before measuring (default 1)")
    args = ap.parse_args()

    requests = args.requests if args.requests is not None else max(8, 4 * args.concurrency)

    for _ in range(args.warmup):
        one_request(args.base, args.model, 16, True, seed=1, timeout=args.timeout)

    summary, results = run_point(args.base, args.model, args.concurrency, requests,
                                 args.max_tokens, args.greedy, args.timeout)
    point = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "label": args.label,
        "base": args.base,
        "concurrency": args.concurrency,
        "requests": requests,
        "max_tokens": args.max_tokens,
        "greedy": args.greedy,
        **summary,
    }
    print(json.dumps(point))
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(point) + "\n")
    if args.per_request:
        with open(args.per_request, "a") as f:
            for r in results:
                row = {k: v for k, v in r.items() if k != "text"}
                row["label"] = args.label
                row["concurrency"] = args.concurrency
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
