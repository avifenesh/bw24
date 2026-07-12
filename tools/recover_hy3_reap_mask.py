#!/usr/bin/env python3
"""Recover original Hy3 expert ids from a renumbered MLX REAP checkpoint.

The public REAP50 checkpoint retains 96 of 192 experts per layer but does not
publish the original ids. Its router rows and correction biases are retained in
the same order as the expert tensors. Match the checkpoint's 8-bit MLX router
rows against the pinned BF16 source, then verify every match independently with
the unquantized correction bias.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from hy3_mlx_to_q4k import (
    SafeTensorDir,
    dequant_mlx_affine_rows,
    mlx_quant_params,
    read_numeric,
    sha256_file,
)


FORMAT = "bw24-hy3-reap-mask-v1"


def parse_layers(raw: str) -> list[int]:
    if "-" in raw:
        lo, hi = (int(value) for value in raw.split("-", 1))
        if lo > hi:
            raise ValueError("layer range is descending")
        return list(range(lo, hi + 1))
    return [int(value) for value in raw.split(",") if value]


def match_rows(base: np.ndarray, retained: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return nearest base row, its RMSE, and the second/first RMSE ratio."""
    if base.ndim != 2 or retained.ndim != 2 or base.shape[1] != retained.shape[1]:
        raise ValueError(f"router shape mismatch: base={base.shape}, retained={retained.shape}")
    base64 = np.asarray(base, dtype=np.float64)
    retained64 = np.asarray(retained, dtype=np.float64)
    distances = (
        np.sum(retained64 * retained64, axis=1)[:, None]
        + np.sum(base64 * base64, axis=1)[None, :]
        - 2.0 * (retained64 @ base64.T)
    )
    distances = np.maximum(distances, 0.0)
    order = np.argsort(distances, axis=1)
    best = order[:, 0]
    rows = np.arange(retained.shape[0])
    rmse = np.sqrt(distances[rows, best] / base.shape[1])
    second_rmse = np.sqrt(distances[rows, order[:, 1]] / base.shape[1])
    margin = np.divide(
        second_rmse,
        rmse,
        out=np.full_like(second_rmse, math.inf),
        where=rmse > 0,
    )
    return best, rmse, margin


def read_array(store: SafeTensorDir, name: str) -> np.ndarray:
    info, raw = store.raw(name)
    # Copy before another shard is opened or the store is closed; mmap memoryviews
    # otherwise remain exported.
    return np.array(read_numeric(info, raw).reshape(info.shape), copy=True)


def recover_layer(
    base: SafeTensorDir,
    reference: SafeTensorDir,
    reference_config: dict[str, Any],
    layer: int,
    min_margin: float,
    bias_atol: float,
) -> dict[str, Any]:
    router = f"model.layers.{layer}.mlp.router.gate"
    base_router = read_array(base, router + ".weight").astype(np.float32)
    packed = read_array(reference, router + ".weight")
    scales = read_array(reference, router + ".scales")
    biases = read_array(reference, router + ".biases")
    params = mlx_quant_params(reference_config, router)
    if params.get("mode", "affine") != "affine":
        raise ValueError(f"layer {layer}: expected affine MLX router quantization")
    retained_router = dequant_mlx_affine_rows(
        packed,
        scales,
        biases,
        int(params["bits"]),
        int(params["group_size"]),
    )
    retained, rmse, margin = match_rows(base_router, retained_router)
    retained_ids = [int(expert) for expert in retained]
    if len(set(retained_ids)) != len(retained_ids):
        raise ValueError(f"layer {layer}: nearest router-row matches are not one-to-one")
    if float(margin.min()) < min_margin:
        raise ValueError(
            f"layer {layer}: minimum nearest-row margin {float(margin.min()):.3f} "
            f"is below required {min_margin:.3f}"
        )

    base_bias = read_array(base, f"model.layers.{layer}.mlp.expert_bias").reshape(-1)
    retained_bias = read_array(
        reference, f"model.layers.{layer}.mlp.router.expert_bias"
    ).reshape(-1)
    bias_error = np.abs(retained_bias - base_bias[retained])
    if float(bias_error.max(initial=0.0)) > bias_atol:
        raise ValueError(
            f"layer {layer}: correction-bias confirmation failed; "
            f"max error {float(bias_error.max()):.3e} > {bias_atol:.3e}"
        )

    original_count = base_router.shape[0]
    pruned_ids = sorted(set(range(original_count)) - set(retained_ids))
    return {
        "retained_experts": retained_ids,
        "pruned_experts": pruned_ids,
        "match": {
            "router_rmse_max": float(rmse.max(initial=0.0)),
            "router_rmse_mean": float(rmse.mean()),
            "nearest_margin_min": float(margin.min(initial=math.inf)),
            "correction_bias_error_max": float(bias_error.max(initial=0.0)),
        },
    }


def recover(args: argparse.Namespace) -> dict[str, Any]:
    base_dir = args.base.resolve()
    reference_dir = args.reference.resolve()
    layers = parse_layers(args.layers)
    reference_config = json.loads((reference_dir / "config.json").read_text())
    base = SafeTensorDir(base_dir)
    reference = SafeTensorDir(reference_dir)
    try:
        layer_masks = {
            str(layer): recover_layer(
                base,
                reference,
                reference_config,
                layer,
                args.min_margin,
                args.bias_atol,
            )
            for layer in layers
        }
    finally:
        base.close()
        reference.close()

    retained_counts = {len(spec["retained_experts"]) for spec in layer_masks.values()}
    original_counts = {
        len(spec["retained_experts"]) + len(spec["pruned_experts"])
        for spec in layer_masks.values()
    }
    if len(retained_counts) != 1 or len(original_counts) != 1:
        raise ValueError("expert counts vary across recovered layers")
    return {
        "format": FORMAT,
        "method": "nearest MLX-8bit router row with exact correction-bias confirmation",
        "source": {
            "path": str(base_dir),
            "revision": args.base_revision,
            "index_sha256": sha256_file(base_dir / "model.safetensors.index.json"),
        },
        "reference": {
            "path": str(reference_dir),
            "revision": args.reference_revision,
            "index_sha256": sha256_file(reference_dir / "model.safetensors.index.json"),
        },
        "model": {
            "layers": layers,
            "original_expert_count": next(iter(original_counts)),
            "retained_expert_count": next(iter(retained_counts)),
        },
        "layers": layer_masks,
    }


def self_test() -> None:
    rng = np.random.default_rng(17)
    base = rng.normal(size=(12, 64)).astype(np.float32)
    expected = np.array([1, 3, 4, 8, 10, 11])
    retained = base[expected] + rng.normal(scale=1e-4, size=(6, 64)).astype(np.float32)
    matched, rmse, margin = match_rows(base, retained)
    assert np.array_equal(matched, expected)
    assert float(rmse.max()) < 2e-4
    assert float(margin.min()) > 100
    with tempfile.TemporaryDirectory(prefix="bw24-reap-mask-") as tmp:
        path = Path(tmp) / "mask.json"
        path.write_text(json.dumps({"retained": matched.tolist()}))
        assert json.loads(path.read_text())["retained"] == expected.tolist()
    print("Hy3 REAP mask recovery self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--base-revision", required=False)
    parser.add_argument("--reference-revision", required=False)
    parser.add_argument("--layers", default="1-79")
    parser.add_argument("--min-margin", type=float, default=10.0)
    parser.add_argument("--bias-atol", type=float, default=0.0)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.base is None or args.reference is None or args.out is None:
        parser.error("--base, --reference, and --out are required")
    result = recover(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        f"wrote {args.out}: {len(result['layers'])} layers, "
        f"{result['model']['retained_expert_count']}/"
        f"{result['model']['original_expert_count']} experts retained per layer"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
