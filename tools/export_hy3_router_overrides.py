#!/usr/bin/env python3
"""Export healed Hy3 router and selection-bias tensors as a bw24 F32 overlay blob."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
import torch


FORMAT = "bw24-tensor-overrides-v1"


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


def hf_names(layer: int) -> tuple[str, str]:
    return (
        f"model.layers.{layer}.mlp.router.gate.weight",
        f"model.layers.{layer}.mlp.expert_bias",
    )


def ggml_names(layer: int) -> tuple[str, str]:
    return (
        f"blk.{layer}.ffn_gate_inp.weight",
        f"blk.{layer}.exp_probs_b.bias",
    )


def export(args: argparse.Namespace) -> dict:
    layers = parse_layers(args.layers)
    index_path = args.overlay_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    args.blob.parent.mkdir(parents=True, exist_ok=True)
    tmp_blob = args.blob.with_name(args.blob.name + ".tmp")
    tensors: dict[str, dict] = {}
    sources: dict[str, dict] = {}
    offset = 0
    try:
        with tmp_blob.open("wb") as output:
            for layer in layers:
                hf_router, hf_bias = hf_names(layer)
                ggml_router, ggml_bias = ggml_names(layer)
                for hf_name, ggml_name, ndim in (
                    (hf_router, ggml_router, 2), (hf_bias, ggml_bias, 1),
                ):
                    shard_name = weight_map.get(hf_name)
                    if shard_name is None:
                        raise ValueError(f"overlay index is missing healed tensor {hf_name}")
                    with safe_open(
                        str(args.overlay_dir / shard_name), framework="pt", device="cpu"
                    ) as handle:
                        tensor = handle.get_tensor(hf_name)
                    if tensor.ndim != ndim:
                        raise ValueError(f"{hf_name}: expected {ndim} dimensions, got {tensor.ndim}")
                    values = tensor.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
                    if not np.isfinite(values).all():
                        raise ValueError(f"{hf_name}: non-finite value")
                    raw = values.tobytes(order="C")
                    output.write(raw)
                    tensors[ggml_name] = {
                        "source": hf_name,
                        "offset": offset,
                        "qtype": "F32",
                        "ne": list(reversed(values.shape)),
                        "bytes": len(raw),
                    }
                    sources[hf_name] = {
                        "shard": shard_name,
                        "shard_sha256": sha256(args.overlay_dir / shard_name),
                    }
                    offset += len(raw)
            output.flush()
            os.fsync(output.fileno())
        tmp_blob.replace(args.blob)
    finally:
        tmp_blob.unlink(missing_ok=True)

    receipt = {
        "format": FORMAT,
        "overlay_dir": str(args.overlay_dir.resolve()),
        "overlay_index": {
            "path": str(index_path.resolve()),
            "sha256": sha256(index_path),
        },
        "layers": layers,
        "blob": {
            "path": str(args.blob.resolve()),
            "bytes": args.blob.stat().st_size,
            "sha256": sha256(args.blob),
        },
        "tensors": dict(sorted(tensors.items())),
        "sources": dict(sorted(sources.items())),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="bw24-router-overrides-") as tmp:
        root = Path(tmp)
        overlay = root / "overlay"
        overlay.mkdir()
        shard = overlay / "layer-001.safetensors"
        router = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        bias = torch.tensor([0.25, -0.5, 0.75], dtype=torch.float32)
        hf_router, hf_bias = hf_names(1)
        save_file({hf_router: router, hf_bias: bias}, shard)
        (overlay / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {hf_router: shard.name, hf_bias: shard.name}
        }))
        args = argparse.Namespace(
            overlay_dir=overlay, layers="1", blob=root / "router.bin",
            receipt=root / "overrides.json",
        )
        receipt = export(args)
        assert receipt["format"] == FORMAT
        assert receipt["blob"]["bytes"] == (12 + 3) * 4
        router_record = receipt["tensors"]["blk.1.ffn_gate_inp.weight"]
        assert router_record["ne"] == [4, 3]
        raw = args.blob.read_bytes()
        got_router = np.frombuffer(
            raw[router_record["offset"]:router_record["offset"] + router_record["bytes"]],
            dtype="<f4",
        ).reshape(3, 4)
        assert np.array_equal(got_router, router.numpy())


def main() -> None:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        print("Hy3 router override export self-test: PASS")
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay-dir", type=Path, required=True)
    parser.add_argument("--layers", default="1-79")
    parser.add_argument("--blob", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = export(args)
    print(
        f"wrote {args.receipt} tensors={len(receipt['tensors'])} "
        f"bytes={receipt['blob']['bytes']} sha256={receipt['blob']['sha256']}"
    )


if __name__ == "__main__":
    main()
