#!/usr/bin/env python3
"""Sequential streaming TTFT probe with cold prefix-cache namespaces."""

import argparse
import datetime
import json
import os
import pathlib
import time
import urllib.request

FILLER = (
    "The quick brown fox jumps over the lazy dog while the seasoned engineer "
    "measures throughput, latency, and saturation across every replica. "
)
SHORT_PROMPT = (
    "Summarize the operational state of a GPU serving cluster in exactly three "
    "sentences, then list four risks. Context follows. " + FILLER * 8
)


def upper_median(values):
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def stream_request(args, prompt, measured, index):
    phase = "measured" if measured else "warmup"
    cache_salt = f"{args.label}-{args.shape}-{phase}-{index}-{time.time_ns()}"
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_salt": cache_salt,
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
    request_id = None
    usage = {}
    finish_reason = None
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
            request_id = event.get("id") or request_id
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
                finish_reason = choice.get("finish_reason") or finish_reason

    ended = time.monotonic()
    if first_visible is None:
        raise RuntimeError(f"{args.shape} {phase} request {index}: no visible SSE delta")
    prompt_tokens = usage.get("prompt_tokens")
    if args.expect_prompt_tokens is not None and prompt_tokens != args.expect_prompt_tokens:
        raise RuntimeError(
            f"{args.shape} {phase} request {index}: expected "
            f"{args.expect_prompt_tokens} prompt tokens, got {prompt_tokens}"
        )
    if not request_id:
        raise RuntimeError(f"{args.shape} {phase} request {index}: no response id")

    return {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "label": args.label,
        "shape": args.shape,
        "measured": measured,
        "index": index,
        "id": request_id,
        "cache_salt": cache_salt,
        "client_ttft_ms": (first_visible - started) * 1_000.0,
        "latency_ms": (ended - started) * 1_000.0,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": finish_reason,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", default="step35")
    parser.add_argument("--shape", choices=("short", "4k"), required=True)
    parser.add_argument("--prompt-file")
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--expect-prompt-tokens", type=int)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    if args.shape == "short":
        prompt = SHORT_PROMPT
    else:
        if not args.prompt_file:
            parser.error("--prompt-file is required for --shape 4k")
        prompt = pathlib.Path(args.prompt_file).read_text()

    rows = []
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as output:
        for measured, count in ((False, args.warmup), (True, args.requests)):
            for index in range(count):
                row = stream_request(args, prompt, measured, index)
                output.write(json.dumps(row, sort_keys=True) + "\n")
                output.flush()
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)

    measured_rows = [row for row in rows if row["measured"]]
    summary = {
        "label": args.label,
        "shape": args.shape,
        "n": len(measured_rows),
        "prompt_tokens": sorted({row["prompt_tokens"] for row in measured_rows}),
        "client_ttft_p50_ms": upper_median(
            [row["client_ttft_ms"] for row in measured_rows]
        ),
        "client_ttft_min_ms": min(row["client_ttft_ms"] for row in measured_rows),
        "client_ttft_max_ms": max(row["client_ttft_ms"] for row in measured_rows),
    }
    print(json.dumps({"summary": summary}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
