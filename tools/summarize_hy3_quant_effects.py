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
DEFAULT_QTYPES = ("Q8_0", "NVFP4", "Q3_K", "Q2_K")


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
    qtypes = tuple(payload.get("measurement", {}).get("qtypes", ()))
    if not qtypes or len(set(qtypes)) != len(qtypes):
        raise ValueError("measurement.qtypes must contain distinct quantization types")
    expected = len(payload["model"]["complete_moe_layers"]) * int(payload["model"]["expert_count"])
    if len(rows) != expected:
        raise ValueError(f"expected {expected} expert rows, got {len(rows)}")

    projection_values: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    layer_values: dict[tuple[int, str], list[tuple[float, float]]] = defaultdict(list)
    layer_projection_values: dict[
        tuple[int, str, str], list[tuple[float, float]]
    ] = defaultdict(list)
    format_totals: dict[str, dict[str, float | int]] = {
        qtype: {"encoded_bytes": 0, "full_scaled_squared_error": 0.0}
        for qtype in qtypes
    }
    hotspots: list[dict[str, Any]] = []
    expert_hotspots: list[dict[str, Any]] = []
    upgrades: list[dict[str, Any]] = []
    equal_byte_swaps: list[dict[str, Any]] = []
    for row in rows:
        layer, expert = int(row["layer"]), int(row["expert"])
        scale = float(row["sample_scale"])
        missing = [qtype for qtype in qtypes if qtype not in row["quantization"]]
        if missing:
            raise ValueError(f"layer {layer} expert {expert} is missing qtypes {missing}")
        expert_entry: dict[str, Any] = {
            "layer": layer,
            "expert": expert,
            "routed_tokens": int(row["routed_tokens"]),
            "qtypes": {},
        }
        for qtype in qtypes:
            quant = row["quantization"][qtype]
            joint = quant["joint_output_error"]
            joint_weight = float(joint["baseline_energy"]) * scale
            layer_values[(layer, qtype)].append((float(joint["normalized_mse"]), joint_weight))
            qtype_entry: dict[str, Any] = {
                "joint_normalized_mse": float(joint["normalized_mse"]),
                "full_scaled_joint_squared_error": (
                    float(joint["normalized_mse"]) * joint_weight
                ),
                "projections": {},
            }
            for projection in PROJECTIONS:
                metric = quant["projection_output_error"][projection]
                weight = float(metric["baseline_energy"]) * scale
                value = float(metric["normalized_mse"])
                full_error = float(metric["squared_error"]) * scale
                encoded_bytes = int(
                    quant["projection_weight_error"][projection]["encoded_bytes"]
                )
                projection_values[(projection, qtype)].append((value, weight))
                layer_projection_values[(layer, projection, qtype)].append((value, weight))
                format_totals[qtype]["encoded_bytes"] += encoded_bytes
                format_totals[qtype]["full_scaled_squared_error"] += full_error
                qtype_entry["projections"][projection] = {
                    "normalized_mse": value,
                    "full_scaled_squared_error": full_error,
                    "encoded_bytes": encoded_bytes,
                }
                hotspots.append({
                    "layer": layer, "expert": expert, "projection": projection,
                    "qtype": qtype, "normalized_mse": value,
                    "full_scaled_squared_error": full_error,
                    "routed_tokens": int(row["routed_tokens"]),
                })
            expert_entry["qtypes"][qtype] = qtype_entry
        expert_entry["maximum_full_scaled_joint_squared_error"] = max(
            item["full_scaled_joint_squared_error"]
            for item in expert_entry["qtypes"].values()
        )
        expert_hotspots.append(expert_entry)
        for projection in PROJECTIONS:
            # Compare every strictly larger representation without assuming input precision order.
            for lower in qtypes:
                for higher in qtypes:
                    if lower == higher:
                        continue
                    low = row["quantization"][lower]
                    high = row["quantization"][higher]
                    low_error = (
                        float(low["projection_output_error"][projection]["squared_error"]) * scale
                    )
                    high_error = (
                        float(high["projection_output_error"][projection]["squared_error"]) * scale
                    )
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
            # Equal-byte formats are a pure quality decision.  Record each unordered pair once so
            # Q3_K vs IQ3_S and NVFP4 vs Q4_K remain visible in the effects evidence.
            for index, first in enumerate(qtypes):
                for second in qtypes[index + 1:]:
                    first_quant = row["quantization"][first]
                    second_quant = row["quantization"][second]
                    first_bytes = int(
                        first_quant["projection_weight_error"][projection]["encoded_bytes"]
                    )
                    second_bytes = int(
                        second_quant["projection_weight_error"][projection]["encoded_bytes"]
                    )
                    if first_bytes != second_bytes:
                        continue
                    first_error = (
                        float(first_quant["projection_output_error"][projection]["squared_error"])
                        * scale
                    )
                    second_error = (
                        float(second_quant["projection_output_error"][projection]["squared_error"])
                        * scale
                    )
                    if not (math.isfinite(first_error) and math.isfinite(second_error)):
                        continue
                    equal_byte_swaps.append({
                        "layer": layer,
                        "expert": expert,
                        "projection": projection,
                        "qtype_a": first,
                        "qtype_b": second,
                        "encoded_bytes": first_bytes,
                        "full_scaled_squared_error_a": first_error,
                        "full_scaled_squared_error_b": second_error,
                        "error_delta_a_minus_b": first_error - second_error,
                        "absolute_error_delta": abs(first_error - second_error),
                        "winner": (
                            first if first_error < second_error
                            else second if second_error < first_error
                            else "tie"
                        ),
                    })

    layers: dict[str, Any] = {}
    for layer in payload["model"]["complete_moe_layers"]:
        layers[str(layer)] = {
            qtype: weighted_summary(layer_values[(int(layer), qtype)]) for qtype in qtypes
        }
    projection_map = {
        projection: {
            qtype: weighted_summary(projection_values[(projection, qtype)]) for qtype in qtypes
        } for projection in PROJECTIONS
    }
    layer_projection_map = {
        str(layer): {
            projection: {
                qtype: weighted_summary(
                    layer_projection_values[(int(layer), projection, qtype)]
                )
                for qtype in qtypes
            }
            for projection in PROJECTIONS
        }
        for layer in payload["model"]["complete_moe_layers"]
    }
    equal_byte_pair_summary = []
    for index, first in enumerate(qtypes):
        for second in qtypes[index + 1:]:
            matches = [
                item for item in equal_byte_swaps
                if item["qtype_a"] == first and item["qtype_b"] == second
            ]
            if not matches:
                continue

            def summarize_matches(items: list[dict[str, Any]]) -> dict[str, Any]:
                error_a = sum(float(item["full_scaled_squared_error_a"]) for item in items)
                error_b = sum(float(item["full_scaled_squared_error_b"]) for item in items)
                wins_a = sum(item["winner"] == first for item in items)
                wins_b = sum(item["winner"] == second for item in items)
                ties = len(items) - wins_a - wins_b
                return {
                    "comparisons": len(items),
                    "wins_a": wins_a,
                    "wins_b": wins_b,
                    "ties": ties,
                    "total_encoded_bytes": sum(int(item["encoded_bytes"]) for item in items),
                    "total_full_scaled_squared_error_a": error_a,
                    "total_full_scaled_squared_error_b": error_b,
                    "error_delta_a_minus_b": error_a - error_b,
                    "lower_total_error_qtype": (
                        first if error_a < error_b
                        else second if error_b < error_a
                        else "tie"
                    ),
                }

            equal_byte_pair_summary.append({
                "qtype_a": first,
                "qtype_b": second,
                **summarize_matches(matches),
                "by_projection": {
                    projection: summarize_matches([
                        item for item in matches if item["projection"] == projection
                    ])
                    for projection in PROJECTIONS
                },
            })
    format_pairwise = []
    for index, first in enumerate(qtypes):
        for second in qtypes[index + 1:]:
            first_bytes = int(format_totals[first]["encoded_bytes"])
            second_bytes = int(format_totals[second]["encoded_bytes"])
            first_error = float(format_totals[first]["full_scaled_squared_error"])
            second_error = float(format_totals[second]["full_scaled_squared_error"])
            same_bytes = first_bytes == second_bytes
            smaller = None
            larger = None
            reduction_per_added_gb = None
            if not same_bytes:
                smaller, larger = (
                    (first, second) if first_bytes < second_bytes else (second, first)
                )
                smaller_error = float(format_totals[smaller]["full_scaled_squared_error"])
                larger_error = float(format_totals[larger]["full_scaled_squared_error"])
                added_bytes = int(format_totals[larger]["encoded_bytes"]) - int(
                    format_totals[smaller]["encoded_bytes"]
                )
                reduction_per_added_gb = (
                    (smaller_error - larger_error) * 1_000_000_000 / added_bytes
                )
            format_pairwise.append({
                "qtype_a": first,
                "qtype_b": second,
                "same_encoded_bytes": same_bytes,
                "total_encoded_bytes_a": first_bytes,
                "total_encoded_bytes_b": second_bytes,
                "total_full_scaled_squared_error_a": first_error,
                "total_full_scaled_squared_error_b": second_error,
                "error_delta_a_minus_b": first_error - second_error,
                "relative_error_gap": abs(first_error - second_error) / max(
                    first_error, second_error, 1e-30
                ),
                "lower_error_qtype": (
                    first if first_error < second_error
                    else second if second_error < first_error
                    else "tie"
                ),
                "smaller_qtype": smaller,
                "larger_qtype": larger,
                "error_reduction_per_added_gb": reduction_per_added_gb,
            })
    hotspots.sort(key=lambda row: (-row["full_scaled_squared_error"], row["layer"], row["expert"]))
    expert_hotspots.sort(key=lambda row: (
        -row["maximum_full_scaled_joint_squared_error"], row["layer"], row["expert"]
    ))
    upgrades.sort(key=lambda row: (-row["error_reduction_per_gb"], row["layer"], row["expert"]))
    equal_byte_swaps.sort(key=lambda row: (
        -row["absolute_error_delta"], row["layer"], row["expert"], row["projection"]
    ))
    return {
        "format": OUTPUT_FORMAT,
        "source": payload.get("source"),
        "measurement": payload["measurement"],
        "calibration": payload["calibration"],
        "coverage": {"expert_rows": len(rows), "layers": len(layers)},
        "projection_damage": projection_map,
        "layer_damage": layers,
        "layer_projection_damage": layer_projection_map,
        "format_totals": format_totals,
        "format_pairwise": format_pairwise,
        "equal_byte_pair_summary": equal_byte_pair_summary,
        "top_sensitive_experts": expert_hotspots[:top_n],
        "top_sensitive_functions": hotspots[:top_n],
        "best_precision_upgrades": upgrades[:top_n],
        "best_equal_byte_swaps": equal_byte_swaps[:top_n],
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


def write_layer_projection_csv(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "layer", "projection", "qtype", "weighted_mean", "weighted_median",
            "maximum", "weight",
        ])
        writer.writeheader()
        for layer, projections in result["layer_projection_damage"].items():
            for projection, qtypes in projections.items():
                for qtype, metrics in qtypes.items():
                    writer.writerow({
                        "layer": layer,
                        "projection": projection,
                        "qtype": qtype,
                        **metrics,
                    })


def self_test() -> None:
    qtypes = ("Q8_0", "NVFP4", "Q4_K", "IQ4_XS", "Q3_K", "IQ3_S", "Q2_K")
    encoded = {
        "Q8_0": 500,
        "NVFP4": 400,
        "Q4_K": 400,
        "IQ4_XS": 380,
        "Q3_K": 300,
        "IQ3_S": 300,
        "Q2_K": 200,
    }
    rows = []
    for layer in (1, 2):
        for expert in (0, 1):
            quant = {}
            for rank, qtype in enumerate(qtypes):
                error = 0.01 * (rank + 1) * (layer + expert)
                quant[qtype] = {
                    "joint_output_error": {"normalized_mse": error, "baseline_energy": 10.0},
                    "projection_output_error": {
                        p: {"normalized_mse": error, "baseline_energy": 10.0,
                            "squared_error": error * 10.0} for p in PROJECTIONS
                    },
                    "projection_weight_error": {
                        p: {"encoded_bytes": encoded[qtype]} for p in PROJECTIONS
                    },
                }
            rows.append({"layer": layer, "expert": expert, "sample_scale": 2.0,
                         "routed_tokens": 8, "quantization": quant})
    payload = {
        "format": FORMAT,
        "model": {"complete_moe_layers": [1, 2], "expert_count": 2},
        "measurement": {"qtypes": list(qtypes)}, "source": {},
        "calibration": {"public_eval_data_used_for_selection": False}, "scores": rows,
    }
    result = build_map(payload, 3)
    assert result["coverage"] == {"expert_rows": 4, "layers": 2}
    assert len(result["layer_projection_damage"]) == 2
    assert set(result["layer_projection_damage"]["1"]) == set(PROJECTIONS)
    assert len(result["top_sensitive_functions"]) == 3
    assert len(result["top_sensitive_experts"]) == 3
    assert len(result["best_precision_upgrades"]) == 3
    assert len(result["best_equal_byte_swaps"]) == 3
    assert all(item["encoded_bytes"] in (300, 400)
               for item in result["best_equal_byte_swaps"])
    pairwise = {(item["qtype_a"], item["qtype_b"]): item
                for item in result["format_pairwise"]}
    assert pairwise[("NVFP4", "Q4_K")]["same_encoded_bytes"] is True
    assert pairwise[("Q3_K", "IQ3_S")]["same_encoded_bytes"] is True
    assert pairwise[("Q4_K", "IQ4_XS")]["same_encoded_bytes"] is False
    equal_pairwise = {
        (item["qtype_a"], item["qtype_b"]): item
        for item in result["equal_byte_pair_summary"]
    }
    assert equal_pairwise[("NVFP4", "Q4_K")]["comparisons"] == 12
    assert equal_pairwise[("NVFP4", "Q4_K")]["wins_a"] == 12
    assert equal_pairwise[("Q3_K", "IQ3_S")]["comparisons"] == 12
    assert equal_pairwise[("Q3_K", "IQ3_S")]["wins_a"] == 12
    assert all(
        equal_pairwise[("Q3_K", "IQ3_S")]["by_projection"][projection]["comparisons"]
        == 4
        for projection in PROJECTIONS
    )
    assert result["projection_damage"]["gate"]["Q8_0"]["weighted_mean"] \
        < result["projection_damage"]["gate"]["Q2_K"]["weighted_mean"]
    with tempfile.TemporaryDirectory() as tmp:
        write_csv(result, Path(tmp) / "layers.csv")
        write_layer_projection_csv(result, Path(tmp) / "layer-projections.csv")


def main() -> None:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        print("Hy3 quant effects map self-test: PASS")
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layer-csv", type=Path)
    parser.add_argument("--layer-projection-csv", type=Path)
    parser.add_argument("--top-n", type=int, default=256)
    args = parser.parse_args()
    result = build_map(json.loads(args.input.read_text()), args.top_n)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.layer_csv:
        write_csv(result, args.layer_csv)
    if args.layer_projection_csv:
        write_layer_projection_csv(result, args.layer_projection_csv)
    print(f"wrote {args.out} sha256={sha256(args.out)}")


if __name__ == "__main__":
    main()
