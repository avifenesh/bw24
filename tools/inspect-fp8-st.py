#!/usr/bin/env python3
"""Classify FP8 safetensors weights using memra's loader contracts.

This reads safetensors headers only. It does not load model payloads or use a GPU.
The runtime residency census remains authoritative for payload and transform checks.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from collections import Counter
from pathlib import Path


MAX_HEADER_BYTES = 256 * 1024 * 1024


def read_header(path: Path) -> dict[str, dict]:
    with path.open("rb") as handle:
        raw_len = handle.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"{path}: truncated safetensors header length")
        header_len = struct.unpack("<Q", raw_len)[0]
        if header_len > MAX_HEADER_BYTES:
            raise ValueError(f"{path}: implausible header length {header_len}")
        raw_header = handle.read(header_len)
        if len(raw_header) != header_len:
            raise ValueError(f"{path}: truncated safetensors header")
    parsed = json.loads(raw_header)
    return {name: info for name, info in parsed.items() if name != "__metadata__"}


def tensor_headers(directory: Path) -> tuple[list[Path], dict[str, dict]]:
    shards = sorted(directory.glob("*.safetensors"))
    if not shards:
        raise ValueError(f"{directory}: no *.safetensors files")

    index_path = directory / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        referenced = {directory / name for name in index.get("weight_map", {}).values()}
        missing = sorted(path for path in referenced if not path.is_file())
        if missing:
            raise ValueError(
                "index references missing shards: " + ", ".join(str(path) for path in missing)
            )

    tensors: dict[str, dict] = {}
    for shard in shards:
        for name, info in read_header(shard).items():
            if name in tensors:
                raise ValueError(f"duplicate tensor header {name!r}")
            tensors[name] = info
    return shards, tensors


def product(values: list[int]) -> int:
    return math.prod(int(value) for value in values)


def classify_weight(name: str, info: dict, tensors: dict[str, dict]) -> tuple[str, str]:
    shape = [int(value) for value in info.get("shape", [])]
    if len(shape) != 2:
        return "unsupported", f"rank={len(shape)}"
    out_f, in_f = shape
    stem = name[: -len(".weight")]
    scale_name = next(
        (
            candidate
            for candidate in (f"{stem}.weight_scale", f"{stem}.weight_scale_inv")
            if candidate in tensors
        ),
        None,
    )
    if scale_name is None:
        return "unsupported", "missing weight_scale/weight_scale_inv sibling"

    scale = tensors[scale_name]
    scale_shape = [int(value) for value in scale.get("shape", [])]
    scale_dtype = scale.get("dtype")
    if scale_dtype not in {"F32", "BF16"}:
        return "unsupported", f"{scale_name} dtype={scale_dtype}"
    if out_f <= 0 or in_f % 32 != 0:
        return "unsupported", f"native residency needs out_f>0 and in_f%32=0, got {shape}"

    count = product(scale_shape)
    if count == 1:
        return "per-tensor", f"{scale_name} {scale_dtype} shape={scale_shape}"
    if count == out_f:
        return "per-row", f"{scale_name} {scale_dtype} shape={scale_shape}"
    block_shape = [(out_f + 127) // 128, (in_f + 127) // 128]
    if scale_shape == block_shape:
        return "block-128", f"{scale_name} {scale_dtype} shape={scale_shape}"
    return (
        "unsupported",
        f"{scale_name} {scale_dtype} shape={scale_shape}, expected scalar, "
        f"[{out_f},1], or {block_shape}",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--require-direct",
        action="store_true",
        help="fail unless every 2D F8_E4M3 weight is per-tensor or block-128",
    )
    args = parser.parse_args()

    directory = args.checkpoint.resolve()
    try:
        shards, tensors = tensor_headers(directory)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FP8-ST INSPECT FAIL: {exc}", file=sys.stderr)
        return 1

    classes: Counter[str] = Counter()
    class_bytes: Counter[str] = Counter()
    problems: list[tuple[str, str, str]] = []
    f8_weights = 0
    packed_scale_planes = 0
    unknown_f8_2d: list[str] = []

    for name, info in sorted(tensors.items()):
        shape = [int(value) for value in info.get("shape", [])]
        if info.get("dtype") != "F8_E4M3" or len(shape) != 2:
            continue
        if not name.endswith(".weight"):
            packed_weight = name[: -len("_scale")] if name.endswith(".weight_scale") else ""
            if packed_weight and tensors.get(packed_weight, {}).get("dtype") == "U8":
                packed_scale_planes += 1
            else:
                unknown_f8_2d.append(name)
            continue
        f8_weights += 1
        kind, detail = classify_weight(name, info, tensors)
        classes[kind] += 1
        class_bytes[kind] += product(shape)
        if kind in {"per-row", "unsupported"}:
            problems.append((name, kind, detail))

    print(f"checkpoint: {directory}")
    print(f"safetensors files: {len(shards)}")
    print(f"tensor headers: {len(tensors)}")
    print(f"2D F8_E4M3 weights: {f8_weights}")
    for kind in ("per-tensor", "block-128", "per-row", "unsupported"):
        mib = class_bytes[kind] / (1024 * 1024)
        print(f"  {kind:11s}: {classes[kind]:4d} tensors  {mib:10.3f} MiB")
    if packed_scale_planes:
        print(f"packed-U8 E4M3 scale planes: {packed_scale_planes}")
    if unknown_f8_2d:
        print(f"unclassified 2D F8_E4M3 tensors: {len(unknown_f8_2d)}")
        for name in unknown_f8_2d[:20]:
            print(f"  UNKNOWN-F8: {name}")
        if len(unknown_f8_2d) > 20:
            print(f"  ... {len(unknown_f8_2d) - 20} more unclassified F8 tensors")

    for name, kind, detail in problems[:20]:
        print(f"  {kind.upper()}: {name}: {detail}")
    if len(problems) > 20:
        print(f"  ... {len(problems) - 20} more non-direct tensors")

    direct = f8_weights > 0 and not problems and not unknown_f8_2d
    print(f"header-only direct-path verdict: {'PASS' if direct else 'FAIL'}")
    print(
        "runtime checks still required: finite positive scales, no E4M3 NaN codes, "
        "transform support, native residency census, and FP8-MMQ dispatch"
    )
    if args.require_direct and not direct:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
