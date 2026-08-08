#!/usr/bin/env python3
"""Barrier-synchronized same-prefix TTFT receipt against a live memra server."""

import argparse
import concurrent.futures
import json
import math
import statistics
import threading
import time
import urllib.request


def one(base, model, prefix, suffix, salt, start, timeout):
    body = {
        "model": model,
        "prompt_ids": prefix + suffix,
        "max_ctx": len(prefix) + len(suffix) + 64,
        "max_tokens": 8,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_salt": salt,
    }
    req = urllib.request.Request(
        base + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    start.wait()
    t0 = time.monotonic()
    ttft = None
    usage = {}
    chunks = 0
    with urllib.request.urlopen(req, timeout=timeout) as response:
        for raw in response:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            if event.get("error"):
                raise RuntimeError(json.dumps(event["error"], sort_keys=True))
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                text = choice.get("text") or ""
                if text:
                    chunks += 1
                    if ttft is None:
                        ttft = time.monotonic() - t0
    if ttft is None:
        raise RuntimeError("stream completed without a non-empty text chunk")
    return {
        "ttft_s": round(ttft, 6),
        "wall_s": round(time.monotonic() - t0, 6),
        "chunks": chunks,
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
    }


def percentile(values, q):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q / 100 * len(ordered)) - 1)]


def burst(args, salt, n):
    prefix = list(range(2000, 2000 + args.k))
    start = threading.Barrier(n)
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
        futures = []
        for i in range(n):
            suffix = list(range(100_000 + i * args.suffix,
                                100_000 + (i + 1) * args.suffix))
            futures.append(pool.submit(
                one, args.base, args.model, prefix, suffix, salt, start, args.timeout))
        return [future.result() for future in futures]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--k", type=int, default=1024)
    ap.add_argument("--suffix", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--expect", choices=("cold", "dedup"), required=True)
    ap.add_argument("--warmup", action="store_true")
    args = ap.parse_args()
    if args.n < 2 or args.k < 64 or args.suffix < 1:
        ap.error("require n>=2, k>=64, suffix>=1")

    if args.warmup:
        warm = burst(args, f"{args.label}-warmup-{time.time_ns()}", 1)
        print(json.dumps({"label": args.label, "kind": "warmup", "row": warm[0]}))

    rows = burst(args, f"{args.label}-measured-{time.time_ns()}", args.n)
    cached = sorted(row["cached_tokens"] for row in rows)
    expected = ([0] * args.n if args.expect == "cold"
                else [0] + [args.k] * (args.n - 1))
    if cached != expected:
        raise SystemExit(f"{args.label}: cached_tokens {cached}, expected {expected}")
    if any(row["prompt_tokens"] != args.k + args.suffix for row in rows):
        raise SystemExit(f"{args.label}: prompt token count mismatch: {rows}")

    ttft = [row["ttft_s"] for row in rows]
    summary = {
        "label": args.label,
        "kind": "summary",
        "concurrency": args.n,
        "shared_prefix_tokens": args.k,
        "suffix_tokens": args.suffix,
        "expect": args.expect,
        "n": len(rows),
        "ttft_p50_s": round(statistics.median(ttft), 6),
        "ttft_p95_s": round(percentile(ttft, 95), 6),
        "ttft_min_s": round(min(ttft), 6),
        "ttft_max_s": round(max(ttft), 6),
        "ttft_mean_s": round(statistics.mean(ttft), 6),
        "cached_tokens": cached,
    }
    with open(args.out, "a", encoding="utf-8") as out:
        for i, row in enumerate(rows):
            out.write(json.dumps({
                "label": args.label,
                "kind": "request",
                "request": i,
                **row,
            }, sort_keys=True) + "\n")
        out.write(json.dumps(summary, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
