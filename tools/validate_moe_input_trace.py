#!/usr/bin/env python3
"""Validate and lock a BW24_MOE_INPUT_TRACE_DIR capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


FORMAT = "bw24-moe-input-trace-v1"
LOCK_FORMAT = "bw24-moe-input-trace-lock-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_layers(raw: str) -> list[int]:
    if "-" in raw:
        lo, hi = (int(value) for value in raw.split("-", 1))
        if lo > hi:
            raise ValueError("layer range is descending")
        return list(range(lo, hi + 1))
    layers = [int(value) for value in raw.split(",") if value]
    if not layers:
        raise ValueError("at least one layer is required")
    return layers


def load_requests(path: Path) -> list[dict[str, Any]]:
    requests = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not requests:
        raise ValueError("request lock is empty")
    seen: set[str] = set()
    for row in requests:
        trace_id = str(row["ordinal"])
        if trace_id in seen:
            raise ValueError(f"duplicate request ordinal {trace_id}")
        seen.add(trace_id)
        tokens = int(row["prompt_tokens"])
        if tokens <= 0 or ("prompt_ids" in row and tokens != len(row["prompt_ids"])):
            raise ValueError(f"request {trace_id} has inconsistent prompt token coverage")
    return requests


def verify_finite(path: Path, values: int) -> None:
    data = np.memmap(path, dtype="<f4", mode="r", shape=(values,))
    chunk = 1 << 24
    for start in range(0, values, chunk):
        if not np.isfinite(data[start : start + chunk]).all():
            raise ValueError(f"{path}: non-finite f32 value in chunk starting at {start}")


def validate(args: argparse.Namespace) -> dict[str, Any]:
    layers = parse_layers(args.layers)
    requests = load_requests(args.requests)
    index_path = args.trace_dir / "index.jsonl"
    rows = [json.loads(line) for line in index_path.read_text().splitlines() if line.strip()]
    expected_rows = sum(int(request["prompt_tokens"]) for request in requests) * len(layers)
    if len(rows) != expected_rows:
        raise ValueError(f"index rows={len(rows)}, expected {expected_rows}")

    files: dict[str, dict[str, Any]] = {}
    offsets: dict[str, int] = {}
    total_prompt_tokens = 0
    row_index = 0
    for request in requests:
        tokens = int(request["prompt_tokens"])
        total_prompt_tokens += tokens
        for position in range(tokens):
            token_rows = rows[row_index : row_index + len(layers)]
            row_index += len(layers)
            for expected_layer, row in zip(layers, token_rows, strict=True):
                if row.get("format") != FORMAT:
                    raise ValueError(f"row has unsupported format {row.get('format')!r}")
                layer = int(row["layer"])
                hidden = int(row["hidden_size"])
                row_tokens = int(row["tokens"])
                if layer != expected_layer or row_tokens != 1 or hidden != args.hidden_size:
                    raise ValueError(
                        f"request {request['ordinal']} position {position} layer row mismatch: got "
                        f"layer/tokens/hidden={layer}/{row_tokens}/{hidden}, expected "
                        f"{expected_layer}/1/{args.hidden_size}"
                    )
                file_name = str(row["file"])
                if file_name != f"layer-{layer:03}.f32":
                    raise ValueError(f"layer {layer}: unexpected payload file {file_name!r}")
                payload_bytes = int(row["payload_bytes"])
                expected_bytes = args.hidden_size * 4
                if payload_bytes != expected_bytes:
                    raise ValueError(
                        f"request {request['ordinal']} position {position} layer {layer}: "
                        f"payload bytes {payload_bytes}, expected {expected_bytes}"
                    )
                expected_offset = offsets.get(file_name, 0)
                offset = int(row["offset"])
                if offset != expected_offset:
                    raise ValueError(
                        f"{file_name}: non-contiguous offset {offset}, expected {expected_offset}"
                    )
                offsets[file_name] = offset + payload_bytes

    for file_name, expected_bytes in sorted(offsets.items()):
        path = args.trace_dir / file_name
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(f"{path}: bytes={actual_bytes}, expected {expected_bytes}")
        if not args.skip_finite:
            verify_finite(path, actual_bytes // 4)
        files[file_name] = {"bytes": actual_bytes, "sha256": sha256(path)}

    expected_layer_bytes = total_prompt_tokens * args.hidden_size * 4
    if any(spec["bytes"] != expected_layer_bytes for spec in files.values()):
        raise AssertionError("validated layer files do not have identical token coverage")
    return {
        "format": LOCK_FORMAT,
        "trace_format": FORMAT,
        "trace_dir": str(args.trace_dir.resolve()),
        "index": {"path": str(index_path.resolve()), "sha256": sha256(index_path)},
        "requests": {
            "path": str(args.requests.resolve()),
            "sha256": sha256(args.requests),
            "count": len(requests),
            "prompt_tokens": total_prompt_tokens,
        },
        "layers": layers,
        "hidden_size": args.hidden_size,
        "rows": len(rows),
        "bytes_per_layer": expected_layer_bytes,
        "payload_bytes": sum(spec["bytes"] for spec in files.values()),
        "finite_values_verified": not args.skip_finite,
        "files": files,
        "public_eval_data_used_for_selection": False,
    }


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="bw24-moe-input-trace-") as tmp:
        root = Path(tmp)
        requests = root / "requests.jsonl"
        requests.write_text("\n".join(json.dumps(row) for row in [
            {"ordinal": 0, "prompt_tokens": 2, "prompt_ids": [1, 2]},
            {"ordinal": 1, "prompt_tokens": 1, "prompt_ids": [3]},
        ]) + "\n")
        index_rows = []
        offsets = {1: 0, 2: 0}
        for request_tokens in (2, 1):
            for _ in range(request_tokens):
                for layer in (1, 2):
                    values = np.arange(3, dtype="<f4") + layer
                    path = root / f"layer-{layer:03}.f32"
                    with path.open("ab") as handle:
                        handle.write(values.tobytes())
                    index_rows.append({
                        "format": FORMAT,
                        "layer": layer,
                        "tokens": 1,
                        "hidden_size": 3,
                        "file": path.name,
                        "offset": offsets[layer],
                        "payload_bytes": values.nbytes,
                    })
                    offsets[layer] += values.nbytes
        (root / "index.jsonl").write_text(
            "\n".join(json.dumps(row) for row in index_rows) + "\n"
        )
        args = argparse.Namespace(
            trace_dir=root, requests=requests, layers="1-2", hidden_size=3, skip_finite=False,
        )
        lock = validate(args)
        assert lock["rows"] == 6 and lock["requests"]["prompt_tokens"] == 3
        assert lock["payload_bytes"] == 2 * 3 * 3 * 4
        assert lock["finite_values_verified"] is True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--layers", default="1-79")
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--skip-finite", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        print("MoE input trace validator self-test: PASS")
        return
    args = parse_args()
    lock = validate(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(
        f"wrote {args.out} sha256={sha256(args.out)} "
        f"requests={lock['requests']['count']} rows={lock['rows']} "
        f"payload_bytes={lock['payload_bytes']}"
    )


if __name__ == "__main__":
    main()
