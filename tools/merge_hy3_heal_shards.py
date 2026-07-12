#!/usr/bin/env python3
"""Validate Hy3 layer-heal receipts and create a sparse HF safetensors overlay index."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file
import torch


RECEIPT_FORMAT = "bw24-hy3-prune-heal-layer-v1"
LOCK_FORMAT = "bw24-hy3-prune-heal-overlay-v1"


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


def merge(args: argparse.Namespace) -> dict:
    layers = parse_layers(args.layers)
    receipts = []
    for layer in layers:
        path = args.receipt_dir / f"layer-{layer:03}.receipt.json"
        receipt = json.loads(path.read_text())
        if receipt.get("format") != RECEIPT_FORMAT or int(receipt["layer"]) != layer:
            raise ValueError(f"{path}: invalid receipt format or layer")
        if receipt.get("public_eval_data_used_for_healing") is not False:
            raise ValueError(f"{path}: healing provenance is not private")
        receipts.append((path, receipt))
    first = receipts[0][1]
    common = {
        key: first[key] for key in ("mode", "source_dir", "source_config_sha256", "source_index_sha256", "plan", "scores", "training")
    }
    weight_map: dict[str, str] = {}
    shard_receipts = []
    before = []
    after = []
    for receipt_path, receipt in receipts:
        if any(receipt.get(key) != value for key, value in common.items()):
            raise ValueError(f"{receipt_path}: common healing configuration differs")
        output = receipt["output"]
        shard = Path(output["path"])
        if shard.parent.resolve() != args.overlay_dir.resolve():
            raise ValueError(f"{receipt_path}: shard is outside the overlay directory")
        if shard.stat().st_size != int(output["bytes"]) or sha256(shard) != output["sha256"]:
            raise ValueError(f"{receipt_path}: output shard size or hash changed")
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
        if len(keys) != int(output["tensor_count"]):
            raise ValueError(f"{receipt_path}: tensor count changed")
        duplicate = set(keys) & weight_map.keys()
        if duplicate:
            raise ValueError(f"{receipt_path}: duplicate overlay tensors {sorted(duplicate)[:3]}")
        for key in keys:
            weight_map[key] = shard.name
        shard_receipts.append({
            "layer": receipt["layer"],
            "receipt_path": str(receipt_path.resolve()),
            "receipt_sha256": sha256(receipt_path),
            "shard": output,
        })
        before.append(float(receipt["before"]["normalized_mse"]))
        after.append(float(receipt["after"]["normalized_mse"]))

    source_dir = Path(first["source_dir"])
    source_config = source_dir / "config.json"
    if sha256(source_config) != first["source_config_sha256"]:
        raise ValueError("source config hash changed")
    args.overlay_dir.mkdir(parents=True, exist_ok=True)
    config_out = args.overlay_dir / "config.json"
    shutil.copy2(source_config, config_out)
    index = {
        "metadata": {
            "total_size": sum(int(receipt["output"]["bytes"]) for _, receipt in receipts),
            "format": LOCK_FORMAT,
            "mode": first["mode"],
        },
        "weight_map": dict(sorted(weight_map.items())),
    }
    index_path = args.overlay_dir / "model.safetensors.index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    lock = {
        "format": LOCK_FORMAT,
        "mode": first["mode"],
        "layers": layers,
        "plan": first["plan"],
        "scores": first["scores"],
        "training": first["training"],
        "source": {
            "directory": str(source_dir.resolve()),
            "config_sha256": first["source_config_sha256"],
            "index_sha256": first["source_index_sha256"],
        },
        "overlay": {
            "directory": str(args.overlay_dir.resolve()),
            "config_sha256": sha256(config_out),
            "index_sha256": sha256(index_path),
            "tensor_count": len(weight_map),
            "bytes": index["metadata"]["total_size"],
        },
        "quality": {
            "mean_before_normalized_mse": sum(before) / len(before),
            "mean_after_normalized_mse": sum(after) / len(after),
            "improved_layers": sum(post < pre for pre, post in zip(before, after, strict=True)),
        },
        "shards": shard_receipts,
        "public_eval_data_used_for_healing": False,
    }
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    args.lock.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    return lock


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="bw24-heal-merge-") as tmp:
        root = Path(tmp)
        source, overlay, receipt_dir = root / "source", root / "overlay", root / "receipts"
        source.mkdir(); overlay.mkdir(); receipt_dir.mkdir()
        (source / "config.json").write_text("{}\n")
        (source / "model.safetensors.index.json").write_text("{}\n")
        common = {
            "format": RECEIPT_FORMAT,
            "mode": "router",
            "source_dir": str(source),
            "source_config_sha256": sha256(source / "config.json"),
            "source_index_sha256": sha256(source / "model.safetensors.index.json"),
            "plan": {"sha256": "plan"},
            "scores": {"sha256": "scores"},
            "training": {"steps": 2},
            "public_eval_data_used_for_healing": False,
        }
        for layer in (1, 2):
            shard = overlay / f"layer-{layer:03}.safetensors"
            save_file({f"layer.{layer}.weight": torch.ones(2, 2)}, shard)
            receipt = dict(common)
            receipt.update({
                "layer": layer,
                "before": {"normalized_mse": 1.0},
                "after": {"normalized_mse": 0.5},
                "output": {
                    "path": str(shard), "bytes": shard.stat().st_size,
                    "sha256": sha256(shard), "tensor_count": 1,
                },
            })
            (receipt_dir / f"layer-{layer:03}.receipt.json").write_text(json.dumps(receipt))
        args = argparse.Namespace(
            receipt_dir=receipt_dir, overlay_dir=overlay, layers="1-2", lock=root / "lock.json"
        )
        lock = merge(args)
        assert lock["overlay"]["tensor_count"] == 2
        assert lock["quality"]["improved_layers"] == 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--overlay-dir", type=Path, required=True)
    parser.add_argument("--layers", default="1-79")
    parser.add_argument("--lock", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        print("Hy3 heal shard merge self-test: PASS")
        return
    args = parse_args()
    lock = merge(args)
    print(
        f"wrote {args.lock} sha256={sha256(args.lock)} "
        f"tensors={lock['overlay']['tensor_count']} bytes={lock['overlay']['bytes']}"
    )


if __name__ == "__main__":
    main()
