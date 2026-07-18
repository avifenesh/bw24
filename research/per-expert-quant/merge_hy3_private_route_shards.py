#!/usr/bin/env python3
"""Merge request-sharded private route captures back into frozen request order."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Any


LAYERS = tuple(range(1, 80))


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=pathlib.Path, required=True)
    parser.add_argument("--lane-root", type=pathlib.Path, required=True)
    parser.add_argument("--lanes", type=int, required=True)
    parser.add_argument("--output-trace", type=pathlib.Path, required=True)
    parser.add_argument("--output-results", type=pathlib.Path, required=True)
    parser.add_argument("--receipt", type=pathlib.Path, required=True)
    args = parser.parse_args()

    frozen = jsonl(args.requests)
    frozen_ordinals = [int(row["ordinal"]) for row in frozen]
    if len(set(frozen_ordinals)) != len(frozen_ordinals):
        raise SystemExit("frozen request ordinals are not unique")

    trace_blocks: dict[int, list[str]] = {}
    results: dict[int, dict[str, Any]] = {}
    inputs: list[dict[str, Any]] = []
    for lane in range(args.lanes):
        root = args.lane_root / f"lane{lane}"
        requests_path = root / "requests.jsonl"
        trace_path = root / "routes.trace"
        results_path = root / "results.jsonl"
        lane_requests = jsonl(requests_path)
        lines = trace_path.read_text().splitlines()
        if len(lines) != len(lane_requests) * len(LAYERS):
            raise SystemExit(
                f"lane {lane}: trace rows {len(lines)} != "
                f"requests {len(lane_requests)} * {len(LAYERS)}"
            )
        for index, request in enumerate(lane_requests):
            ordinal = int(request["ordinal"])
            prompt_tokens = int(request["prompt_tokens"])
            block = lines[index * len(LAYERS) : (index + 1) * len(LAYERS)]
            for expected_layer, line in zip(LAYERS, block, strict=True):
                layer_text, tokens_text, _packed = line.split(maxsplit=2)
                if int(layer_text) != expected_layer or int(tokens_text) != prompt_tokens:
                    raise SystemExit(
                        f"lane {lane} ordinal {ordinal}: malformed layer block"
                    )
            if ordinal in trace_blocks:
                raise SystemExit(f"duplicate trace ordinal {ordinal}")
            trace_blocks[ordinal] = block
        for result in jsonl(results_path):
            ordinal = int(result["ordinal"])
            if not result.get("ok"):
                raise SystemExit(f"lane {lane} ordinal {ordinal}: capture failed")
            if ordinal in results:
                raise SystemExit(f"duplicate result ordinal {ordinal}")
            results[ordinal] = result
        inputs.append(
            {
                "lane": lane,
                "requests_sha256": sha256(requests_path),
                "trace_sha256": sha256(trace_path),
                "results_sha256": sha256(results_path),
            }
        )

    if set(trace_blocks) != set(frozen_ordinals) or set(results) != set(frozen_ordinals):
        raise SystemExit("shards do not cover the frozen request ordinals exactly")
    args.output_trace.parent.mkdir(parents=True, exist_ok=True)
    args.output_trace.write_text(
        "".join(line + "\n" for ordinal in frozen_ordinals for line in trace_blocks[ordinal])
    )
    args.output_results.write_text(
        "".join(json.dumps(results[ordinal], sort_keys=True) + "\n" for ordinal in frozen_ordinals)
    )
    receipt = {
        "format": "bw24-hy3-private-route-shard-merge-v1",
        "public_capability_results_used": False,
        "requests": len(frozen),
        "prompt_tokens": sum(int(row["prompt_tokens"]) for row in frozen),
        "trace_rows": len(frozen) * len(LAYERS),
        "requests_sha256": sha256(args.requests),
        "output_trace_sha256": sha256(args.output_trace),
        "output_results_sha256": sha256(args.output_results),
        "inputs": inputs,
    }
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
