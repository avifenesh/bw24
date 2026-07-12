#!/usr/bin/env python3
"""Summarize private Hy3 quantization sensitivity into an inspectable effects map."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


FORMAT = "bw24-hy3-quant-sensitivity-v1"
OUTPUT_FORMAT = "bw24-hy3-quant-effects-map-v1"
PROJECTIONS = ("gate", "up", "down")
QTYPES = ("Q8_0", "NVFP4", "Q3_K", "Q2_K")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weighted_summary(values: list[tuple[float, float]]) -> dict[str, float]:
    weight = sum(item[1] for item in values)
    total = sum(value * item_weight for value, item_weight in values)
    ordered = sorted(values)
    midpoint = weight / 2
    cumulative = 0.0
    median = 0.0
    for value, item_weight in ordered:
        cumulative += item_weight
        if cumulative >= midpoint:
            median = value
            break
    return {
        "weighted_mean": total / max(weight, 1e-30),
        "weighted_median": median,
        "maximum": max((value for value, _ in values), default=0.0),
        "weight": weight,
    }


def build_map(payload: dict[str, Any], top_n: int) -> dict[str, Any]:
    if payload.get("format") != FORMAT:
        raise ValueError(f"input format must be {FORMAT}")
    if payload["calibration"].get("public_eval_data_used_for_selection") is not False:
        raise ValueError("effects map requires private-only selection evidence")
    rows = payload["scores"]
    expected = len(payload["model"]["complete_moe_layers"]) * int(payload["model"]["expert_count"])
    if len(rows) != expected:
        raise ValueError(f"expected {expected} expert rows, got {len(rows)}")

    projection_values: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    layer_values: dict[tuple[int, str], list[tuple[float, float]]] = defaultdict(list)
    hotspots: list[dict[str, Any]] = []
    upgrades: list[dict[str, Any]] = []
    for row in rows:
        layer, expert = int(row["layer"]), int(row["expert"])
        scale = float(row["sample_scale"])
        for qtype in QTYPES:
            quant = row["quantization"][qtype]
            joint = quant["joint_output_error"]
            joint_weight = float(joint["baseline_energy"]) * scale
            layer_values[(layer, qtype)].append((float(joint["normalized_mse"]), joint_weight))
            for projection in PROJECTIONS:
                metric = quant["projection_output_error"][projection]
                weight = float(metric["baseline_energy"]) * scale
                value = float(metric["normalized_mse"])
                projection_values[(projection, qtype)].append((value, weight))
                hotspots.append({
                    "layer": layer, "expert": expert, "projection": projection,
                    "qtype": qtype, "normalized_mse": value,
                    "full_scaled_squared_error": float(metric["squared_error"]) * scale,
                    "routed_tokens": int(row["routed_tokens"]),
                })
        for projection in PROJECTIONS:
            for lower, higher in zip(reversed(QTYPES[1:]), reversed(QTYPES[:-1]), strict=True):
                low = row["quantization"][lower]
                high = row["quantization"][higher]
                low_error = float(low["projection_output_error"][projection]["squared_error"]) * scale
                high_error = float(high["projection_output_error"][projection]["squared_error"]) * scale
                low_bytes = int(low["projection_weight_error"][projection]["encoded_bytes"])
                high_bytes = int(high["projection_weight_error"][projection]["encoded_bytes"])
                extra_bytes = high_bytes - low_bytes
                reduction = low_error - high_error
                if extra_bytes > 0 and math.isfinite(reduction):
                    upgrades.append({
                        "layer": layer, "expert": expert, "projection": projection,
                        "from_qtype": lower, "to_qtype": higher,
                        "extra_bytes": extra_bytes, "error_reduction": reduction,
                        "error_reduction_per_gb": reduction * 1_000_000_000 / extra_bytes,
                    })

    layers: dict[str, Any] = {}
    for layer in payload["model"]["complete_moe_layers"]:
        layers[str(layer)] = {
            qtype: weighted_summary(layer_values[(int(layer), qtype)]) for qtype in QTYPES
        }
    projection_map = {
        projection: {
            qtype: weighted_summary(projection_values[(projection, qtype)]) for qtype in QTYPES
        } for projection in PROJECTIONS
    }
    hotspots.sort(key=lambda row: (-row["full_scaled_squared_error"], row["layer"], row["expert"]))
    upgrades.sort(key=lambda row: (-row["error_reduction_per_gb"], row["layer"], row["expert"]))
    return {
        "format": OUTPUT_FORMAT,
        "source": payload.get("source"),
        "measurement": payload["measurement"],
        "calibration": payload["calibration"],
        "coverage": {"expert_rows": len(rows), "layers": len(layers)},
        "projection_damage": projection_map,
        "layer_damage": layers,
        "top_sensitive_functions": hotspots[:top_n],
        "best_precision_upgrades": upgrades[:top_n],
        "public_eval_data_used_for_selection": False,
    }


def write_csv(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "layer", "qtype", "weighted_mean", "weighted_median", "maximum", "weight"
        ])
        writer.writeheader()
        for layer, qtypes in result["layer_damage"].items():
            for qtype, metrics in qtypes.items():
                writer.writerow({"layer": layer, "qtype": qtype, **metrics})


def self_test() -> None:
    rows = []
    for layer in (1, 2):
        for expert in (0, 1):
            quant = {}
            for rank, qtype in enumerate(QTYPES):
                error = 0.01 * (rank + 1) * (layer + expert)
                quant[qtype] = {
                    "joint_output_error": {"normalized_mse": error, "baseline_energy": 10.0},
                    "projection_output_error": {
                        p: {"normalized_mse": error, "baseline_energy": 10.0,
                            "squared_error": error * 10.0} for p in PROJECTIONS
                    },
                    "projection_weight_error": {
                        p: {"encoded_bytes": 400 - rank * 50} for p in PROJECTIONS
                    },
                }
            rows.append({"layer": layer, "expert": expert, "sample_scale": 2.0,
                         "routed_tokens": 8, "quantization": quant})
    payload = {
        "format": FORMAT,
        "model": {"complete_moe_layers": [1, 2], "expert_count": 2},
        "measurement": {}, "source": {},
        "calibration": {"public_eval_data_used_for_selection": False}, "scores": rows,
    }
    result = build_map(payload, 3)
    assert result["coverage"] == {"expert_rows": 4, "layers": 2}
    assert len(result["top_sensitive_functions"]) == 3
    assert len(result["best_precision_upgrades"]) == 3
    assert result["projection_damage"]["gate"]["Q8_0"]["weighted_mean"] \
        < result["projection_damage"]["gate"]["Q2_K"]["weighted_mean"]
    with tempfile.TemporaryDirectory() as tmp:
        write_csv(result, Path(tmp) / "layers.csv")


def main() -> None:
    if sys.argv[1:] == ["--self-test"]:
        self_test(); print("Hy3 quant effects map self-test: PASS"); return
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layer-csv", type=Path)
    parser.add_argument("--top-n", type=int, default=256)
    args = parser.parse_args()
    result = build_map(json.loads(args.input.read_text()), args.top_n)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.layer_csv:
        write_csv(result, args.layer_csv)
    print(f"wrote {args.out} sha256={sha256(args.out)}")


if __name__ == "__main__":
    main()
