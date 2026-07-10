#!/usr/bin/env python3
"""Submit a frozen prompt-id corpus to one bw24 control arm and record request outcomes."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--requests", type=Path, required=True)
    p.add_argument("--endpoint", default="http://127.0.0.1:8080/v1/completions")
    p.add_argument("--model", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--retries", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    records = [json.loads(line) for line in args.requests.read_text().splitlines() if line.strip()]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    total_tokens = 0
    with args.out.open("w") as out:
        for record in records:
            payload = json.dumps(
                {
                    "model": args.model,
                    "prompt_ids": record["prompt_ids"],
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "stream": False,
                    "max_ctx": len(record["prompt_ids"]) + 8,
                }
            ).encode()
            started = time.monotonic()
            error = None
            response = None
            for attempt in range(args.retries + 1):
                try:
                    request = urllib.request.Request(
                        args.endpoint, data=payload, headers={"Content-Type": "application/json"}
                    )
                    with urllib.request.urlopen(request, timeout=args.timeout) as reply:
                        response = json.loads(reply.read())
                    error = None
                    break
                except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    if attempt < args.retries:
                        time.sleep(2**attempt)
            result = {
                "ordinal": record["ordinal"],
                "stratum": record["stratum"],
                "source_id": record["source_id"],
                "prompt_tokens": record["prompt_tokens"],
                "elapsed_s": time.monotonic() - started,
                "ok": error is None,
                "error": error,
                "response": response,
            }
            out.write(json.dumps(result, sort_keys=True) + "\n")
            out.flush()
            if error is not None:
                raise RuntimeError(f"request {record['ordinal']} failed: {error}")
            completed += 1
            total_tokens += record["prompt_tokens"]
            print(
                f"[{completed}/{len(records)}] {record['stratum']} "
                f"tokens={record['prompt_tokens']} elapsed={result['elapsed_s']:.2f}s",
                flush=True,
            )
    print(f"captured {completed} requests / {total_tokens} prompt tokens for {args.model}")


if __name__ == "__main__":
    main()
