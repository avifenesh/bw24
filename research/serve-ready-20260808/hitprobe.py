#!/usr/bin/env python3
"""Cache-hit repeat TTFT probe: same prompt + SAME cache_salt, 1 prime + N measured hits.

The main probe (probe.py) deliberately makes every request cold (unique salt); this one is
the warm arm of the serve-ready receipt: after one priming request, repeats must be served
from the prefix cache and their TTFT is the cache-hit class. Sends Authorization from
MEMRA_API_KEY (the trial config runs keys-on)."""

import argparse
import datetime
import json
import os
import pathlib
import time
import urllib.request


def stream_request(args, salt, phase, index):
    prompt = pathlib.Path(args.prompt_file).read_text()
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_salt": salt,
    }
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("MEMRA_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(
        args.base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers=headers,
    )
    started = time.monotonic()
    first_visible = None
    usage = {}
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        for raw_line in response:
            line = raw_line.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            usage = event.get("usage") or usage
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                visible = (
                    (delta.get("content") or "")
                    + (delta.get("reasoning") or "")
                    + (delta.get("reasoning_content") or "")
                )
                if visible and first_visible is None:
                    first_visible = time.monotonic()
    ended = time.monotonic()
    if first_visible is None:
        raise RuntimeError(f"cache-hit {phase} {index}: no visible SSE delta")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    return {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "label": args.label,
        "shape": "cache-hit",
        "phase": phase,
        "index": index,
        "cache_salt": salt,
        "client_ttft_ms": (first_visible - started) * 1_000.0,
        "latency_ms": (ended - started) * 1_000.0,
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": cached,
        "completion_tokens": usage.get("completion_tokens"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", default="step35")
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    salt = f"{args.label}-cache-hit-fixed"
    rows = []
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as output:
        row = stream_request(args, salt, "prime", 0)
        output.write(json.dumps(row, sort_keys=True) + "\n")
        output.flush()
        print(json.dumps(row, sort_keys=True), flush=True)
        for index in range(args.requests):
            row = stream_request(args, salt, "hit", index)
            output.write(json.dumps(row, sort_keys=True) + "\n")
            output.flush()
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    ttfts = sorted(r["client_ttft_ms"] for r in rows)
    summary = {
        "label": args.label,
        "shape": "cache-hit",
        "n": len(rows),
        "cached_tokens": sorted({r["cached_tokens"] for r in rows}),
        "client_ttft_p50_ms": ttfts[len(ttfts) // 2],
        "client_ttft_min_ms": ttfts[0],
        "client_ttft_max_ms": ttfts[-1],
    }
    print(json.dumps({"summary": summary}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
