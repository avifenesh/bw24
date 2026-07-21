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


def load_verified_shard_hashes(
    lock_path: Path, overlay_dir: Path, index_path: Path, hash_file=sha256
) -> tuple[dict[str, str], dict[str, str]]:
    lock = json.loads(lock_path.read_text())
    if lock.get("format") != "bw24-hy3-prune-heal-overlay-v1":
        raise ValueError(f"unsupported overlay lock format in {lock_path}")
    resolved_overlay = overlay_dir.resolve()
    if Path(lock["overlay"]["directory"]).resolve() != resolved_overlay:
        raise ValueError("overlay lock directory does not match --overlay-dir")
    if lock["overlay"]["index_sha256"] != hash_file(index_path):
        raise ValueError("overlay index does not match verified overlay lock")

    shard_hashes: dict[str, str] = {}
    for item in lock["shards"]:
        shard = item["shard"]
        path = Path(shard["path"])
        if path.resolve().parent != resolved_overlay:
            raise ValueError(f"overlay lock shard is outside overlay directory: {path}")
        current = overlay_dir / path.name
        if current.resolve() != path.resolve() or current.stat().st_size != shard["bytes"]:
            raise ValueError(f"overlay lock shard metadata mismatch: {path.name}")
        if path.name in shard_hashes:
            raise ValueError(f"duplicate overlay lock shard: {path.name}")
        shard_hashes[path.name] = shard["sha256"]
    return shard_hashes, {
        "path": str(lock_path.resolve()),
        "sha256": hash_file(lock_path),
    }


def export(args: argparse.Namespace, hash_file=sha256) -> dict:
    layers = parse_layers(args.layers)
    index_path = args.overlay_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    args.blob.parent.mkdir(parents=True, exist_ok=True)
    tmp_blob = args.blob.with_name(args.blob.name + ".tmp")
    tensors: dict[str, dict] = {}
    sources: dict[str, dict] = {}
    overlay_lock = getattr(args, "overlay_lock", None)
    if overlay_lock is not None:
        shard_hashes, overlay_lock_record = load_verified_shard_hashes(
            overlay_lock, args.overlay_dir, index_path, hash_file=hash_file
        )
    else:
        shard_hashes, overlay_lock_record = {}, None
    used_shard_hashes: dict[str, str] = {}
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
                    shard_sha256 = used_shard_hashes.get(shard_name)
                    if shard_sha256 is None:
                        if overlay_lock_record is not None:
                            shard_sha256 = shard_hashes.get(shard_name)
                            if shard_sha256 is None:
                                raise ValueError(
                                    f"verified overlay lock is missing shard {shard_name}"
                                )
                        else:
                            shard_sha256 = hash_file(args.overlay_dir / shard_name)
                        used_shard_hashes[shard_name] = shard_sha256
                    tensors[ggml_name] = {
                        "source": hf_name,
                        "offset": offset,
                        "qtype": "F32",
                        "ne": list(reversed(values.shape)),
                        "bytes": len(raw),
                    }
                    sources[hf_name] = {
                        "shard": shard_name,
                        "shard_sha256": shard_sha256,
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
            "sha256": hash_file(index_path),
        },
        "layers": layers,
        "blob": {
            "path": str(args.blob.resolve()),
            "bytes": args.blob.stat().st_size,
            "sha256": hash_file(args.blob),
        },
        "tensors": dict(sorted(tensors.items())),
        "sources": dict(sorted(sources.items())),
    }
    if overlay_lock_record is not None:
        receipt["overlay_lock"] = overlay_lock_record
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
        index = overlay / "model.safetensors.index.json"
        args = argparse.Namespace(
            overlay_dir=overlay, layers="1", blob=root / "router.bin",
            receipt=root / "overrides.json", overlay_lock=None,
        )
        hash_counts: dict[Path, int] = {}

        def counted_sha256(path: Path) -> str:
            hash_counts[path] = hash_counts.get(path, 0) + 1
            return sha256(path)

        receipt = export(args, hash_file=counted_sha256)
        assert hash_counts[shard] == 1
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

        lock = root / "overlay.lock.json"
        lock.write_text(json.dumps({
            "format": "bw24-hy3-prune-heal-overlay-v1",
            "overlay": {
                "directory": str(overlay.resolve()),
                "index_sha256": sha256(index),
            },
            "shards": [{
                "layer": 1,
                "shard": {
                    "path": str(shard.resolve()),
                    "bytes": shard.stat().st_size,
                    "sha256": sha256(shard),
                },
            }],
        }))
        args.overlay_lock = lock
        args.blob = root / "router-from-lock.bin"
        args.receipt = root / "overrides-from-lock.json"
        hash_counts.clear()
        locked_receipt = export(args, hash_file=counted_sha256)
        assert hash_counts.get(shard, 0) == 0
        assert locked_receipt["overlay_lock"]["sha256"] == sha256(lock)
        assert locked_receipt["sources"][hf_router]["shard_sha256"] == sha256(shard)


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
    parser.add_argument(
        "--overlay-lock", type=Path,
        help="reuse shard hashes from an immediately verified heal overlay lock",
    )
    args = parser.parse_args()
    receipt = export(args)
    print(
        f"wrote {args.receipt} tensors={len(receipt['tensors'])} "
        f"bytes={receipt['blob']['bytes']} sha256={receipt['blob']['sha256']}"
    )


if __name__ == "__main__":
    main()
