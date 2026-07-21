#!/usr/bin/env python3
"""Build private-only per-layer structural constraints for Hy3 quant/prune allocation.

The survivor floor is ranked by residual post-heal output error.  The Q2 cap is ranked
independently by robust (weighted-median) measured Q2 output damage.  Public evaluation
results are neither accepted as inputs nor consulted by this policy generator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any


OUTPUT_FORMAT = "bw24-hy3-layer-constraints-v1"
PLAN_FORMAT = "bw24-expert-tier-plan-v2"
EFFECTS_FORMAT = "bw24-hy3-quant-effects-map-v1"
PROJECTIONS_PER_EXPERT = 3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def require_private(payload: dict[str, Any], *, label: str) -> None:
    direct = payload.get("public_eval_data_used_for_selection")
    calibration = payload.get("calibration")
    nested = calibration.get("public_eval_data_used_for_selection") \
        if isinstance(calibration, dict) else None
    if direct is not False and nested is not False:
        raise ValueError(f"{label} lacks a private-only selection attestation")
    if direct is True or nested is True:
        raise ValueError(f"{label} was derived from public evaluation data")


def finite_nonnegative(value: Any, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"invalid {label}: {value}")
    return result


def ranked_tiers(
    values: dict[int, float], critical_fraction: float, high_fraction: float
) -> tuple[dict[int, str], dict[int, int]]:
    if not values:
        raise ValueError("cannot rank an empty layer set")
    if not 0 <= critical_fraction <= 1 or not 0 <= high_fraction <= 1:
        raise ValueError("tier fractions must be in [0, 1]")
    if critical_fraction + high_fraction > 1:
        raise ValueError("critical and high fractions must sum to at most 1")
    ordered = sorted(values, key=lambda layer: (-values[layer], layer))
    critical_count = math.ceil(len(ordered) * critical_fraction)
    high_count = math.ceil(len(ordered) * high_fraction)
    if critical_count + high_count > len(ordered):
        high_count = len(ordered) - critical_count
    tiers: dict[int, str] = {}
    ranks: dict[int, int] = {}
    for rank, layer in enumerate(ordered, start=1):
        ranks[layer] = rank
        tiers[layer] = (
            "critical" if rank <= critical_count
            else "high" if rank <= critical_count + high_count
            else "baseline"
        )
    return tiers, ranks


def q2_counts(reference: dict[str, Any], layers: list[int]) -> dict[int, int]:
    counts = {layer: 0 for layer in layers}
    for row in reference.get("assignments", []):
        if not isinstance(row, dict) or row.get("qtype") != "Q2_K":
            continue
        layer = int(row["layer"])
        if layer not in counts:
            raise ValueError(f"Q2 assignment references unknown layer {layer}")
        experts = row.get("experts")
        projections = row.get("projections")
        if not isinstance(experts, list) or not isinstance(projections, list):
            raise ValueError(f"invalid Q2 assignment in layer {layer}")
        counts[layer] += len(experts) * len(projections)
    return counts


def build_constraints(args: argparse.Namespace) -> dict[str, Any]:
    reference = load_json(args.reference_plan)
    effects = load_json(args.effects_map)
    if reference.get("format") != PLAN_FORMAT:
        raise ValueError("unsupported reference plan format")
    if effects.get("format") != EFFECTS_FORMAT:
        raise ValueError("unsupported quant-effects format")
    require_private(reference, label="reference plan")
    require_private(effects, label="quant-effects map")

    model = reference.get("model", {})
    layers = [int(layer) for layer in model.get("moe_layers", [])]
    if not layers or len(set(layers)) != len(layers):
        raise ValueError("reference plan has invalid MoE layer coverage")
    expert_count = int(model.get("expert_count", 0))
    if expert_count <= 0:
        raise ValueError("reference plan has invalid expert_count")
    expected_layers = {str(layer) for layer in layers}
    if set(reference.get("layer_summary", {})) != expected_layers:
        raise ValueError("reference layer summary does not exactly cover MoE layers")
    layer_damage = effects.get("layer_damage")
    if not isinstance(layer_damage, dict) or set(layer_damage) != expected_layers:
        raise ValueError("quant-effects layer damage does not exactly cover MoE layers")

    post_heal_error: dict[int, float] = {}
    receipt_hashes: list[dict[str, Any]] = []
    for layer in layers:
        path = args.joint_receipts / f"layer-{layer:03}.receipt.json"
        receipt = load_json(path)
        if receipt.get("mode") != "joint" or int(receipt.get("layer", -1)) != layer:
            raise ValueError(f"invalid joint-heal receipt {path}")
        if receipt.get("public_eval_data_used_for_healing") is not False:
            raise ValueError(f"joint-heal receipt is not private-only: {path}")
        post_heal_error[layer] = finite_nonnegative(
            receipt.get("after", {}).get("normalized_mse"),
            label=f"layer {layer} post-heal normalized MSE",
        )
        receipt_hashes.append({
            "layer": layer,
            "path": str(path.resolve()),
            "sha256": sha256(path),
        })

    q2_damage = {
        layer: finite_nonnegative(
            layer_damage[str(layer)].get("Q2_K", {}).get("weighted_median"),
            label=f"layer {layer} Q2 weighted-median damage",
        )
        for layer in layers
    }
    survivor_tiers, survivor_ranks = ranked_tiers(
        post_heal_error, args.critical_fraction, args.high_fraction
    )
    q2_tiers, q2_ranks = ranked_tiers(
        q2_damage, args.critical_fraction, args.high_fraction
    )
    survivor_floors = {
        "critical": args.critical_min_survivors,
        "high": args.high_min_survivors,
        "baseline": args.baseline_min_survivors,
    }
    q2_caps = {
        "critical": args.critical_max_q2_projections,
        "high": args.high_max_q2_projections,
        "baseline": args.baseline_max_q2_projections,
    }
    max_projections = expert_count * PROJECTIONS_PER_EXPERT
    if not (
        0 < survivor_floors["baseline"] <= survivor_floors["high"]
        <= survivor_floors["critical"] <= expert_count
    ):
        raise ValueError("survivor floors must be ordered baseline <= high <= critical")
    if not (
        0 <= q2_caps["critical"] <= q2_caps["high"]
        <= q2_caps["baseline"] <= max_projections
    ):
        raise ValueError("Q2 caps must be ordered critical <= high <= baseline")

    current_q2 = q2_counts(reference, layers)
    rows: dict[str, dict[str, Any]] = {}
    for layer in layers:
        survivor_tier = survivor_tiers[layer]
        q2_tier = q2_tiers[layer]
        rows[str(layer)] = {
            "min_survivors": survivor_floors[survivor_tier],
            "max_q2_projections": q2_caps[q2_tier],
            "diagnostics": {
                "post_heal_normalized_mse": post_heal_error[layer],
                "post_heal_error_rank_descending": survivor_ranks[layer],
                "survivor_tier": survivor_tier,
                "q2_weighted_median_output_damage": q2_damage[layer],
                "q2_damage_rank_descending": q2_ranks[layer],
                "q2_tier": q2_tier,
                "reference_retained_experts": int(
                    reference["layer_summary"][str(layer)]["retained"]
                ),
                "reference_q2_projections": current_q2[layer],
            },
        }

    tier_counts = {
        axis: {
            tier: sum(value == tier for value in tiers.values())
            for tier in ("critical", "high", "baseline")
        }
        for axis, tiers in (("survivor", survivor_tiers), ("q2", q2_tiers))
    }
    return {
        "format": OUTPUT_FORMAT,
        "description": (
            "Preregistered private-only layer constraints: residual post-heal error "
            "sets survivor floors and robust Q2 output damage sets Q2 caps"
        ),
        "model": {
            "expert_count": expert_count,
            "moe_layers": layers,
            "projections_per_expert": PROJECTIONS_PER_EXPERT,
        },
        "policy": {
            "tier_order": ["critical", "high", "baseline"],
            "critical_fraction": args.critical_fraction,
            "high_fraction": args.high_fraction,
            "ranking_tie_break": "ascending_layer_id",
            "survivor_signal": "joint_heal_after_normalized_mse_descending",
            "survivor_floors": survivor_floors,
            "q2_signal": "Q2_K_layer_weighted_median_output_damage_descending",
            "q2_caps": q2_caps,
            "tier_counts": tier_counts,
            "public_capability_results_consulted": False,
        },
        "calibration": {
            "reference_plan": {
                "path": str(args.reference_plan.resolve()),
                "sha256": sha256(args.reference_plan),
            },
            "quant_effects_map": {
                "path": str(args.effects_map.resolve()),
                "sha256": sha256(args.effects_map),
            },
            "joint_heal_receipts": receipt_hashes,
            "public_eval_data_used_for_selection": False,
        },
        "layers": rows,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="bw24-layer-constraints-") as tmp:
        root = Path(tmp)
        layers = list(range(1, 9))
        reference = {
            "format": PLAN_FORMAT,
            "model": {"expert_count": 16, "moe_layers": layers},
            "calibration": {"public_eval_data_used_for_selection": False},
            "layer_summary": {
                str(layer): {"retained": 8, "pruned": 8} for layer in layers
            },
            "assignments": [
                {"layer": layer, "experts": list(range(layer % 4)),
                 "projections": ["gate"], "qtype": "Q2_K"}
                for layer in layers
            ],
        }
        effects = {
            "format": EFFECTS_FORMAT,
            "calibration": {"public_eval_data_used_for_selection": False},
            "public_eval_data_used_for_selection": False,
            "layer_damage": {
                str(layer): {"Q2_K": {"weighted_median": float(9 - layer)}}
                for layer in layers
            },
        }
        reference_path = root / "reference.json"
        effects_path = root / "effects.json"
        write_json(reference_path, reference)
        write_json(effects_path, effects)
        receipts = root / "receipts"
        receipts.mkdir()
        for layer in layers:
            write_json(receipts / f"layer-{layer:03}.receipt.json", {
                "format": "bw24-hy3-prune-heal-layer-v1",
                "mode": "joint",
                "layer": layer,
                "after": {"normalized_mse": float(layer)},
                "public_eval_data_used_for_healing": False,
            })
        args = argparse.Namespace(
            reference_plan=reference_path, effects_map=effects_path,
            joint_receipts=receipts, critical_fraction=0.25, high_fraction=0.25,
            critical_min_survivors=12, high_min_survivors=10,
            baseline_min_survivors=8, critical_max_q2_projections=16,
            high_max_q2_projections=24, baseline_max_q2_projections=48,
        )
        result = build_constraints(args)
        assert result["policy"]["tier_counts"]["survivor"] == {
            "critical": 2, "high": 2, "baseline": 4,
        }
        assert result["layers"]["8"]["min_survivors"] == 12
        assert result["layers"]["7"]["min_survivors"] == 12
        assert result["layers"]["1"]["max_q2_projections"] == 16
        assert result["layers"]["2"]["max_q2_projections"] == 16
        assert result["layers"]["1"]["diagnostics"]["reference_q2_projections"] == 1
        assert result["calibration"]["public_eval_data_used_for_selection"] is False
        effects["public_eval_data_used_for_selection"] = True
        write_json(effects_path, effects)
        try:
            build_constraints(args)
        except ValueError as error:
            assert "public evaluation" in str(error)
        else:
            raise AssertionError("public-eval-derived effects map was accepted")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-plan", type=Path, required=True)
    parser.add_argument("--effects-map", type=Path, required=True)
    parser.add_argument("--joint-receipts", type=Path, required=True)
    parser.add_argument("--critical-fraction", type=float, default=0.25)
    parser.add_argument("--high-fraction", type=float, default=0.25)
    parser.add_argument("--critical-min-survivors", type=int, default=128)
    parser.add_argument("--high-min-survivors", type=int, default=112)
    parser.add_argument("--baseline-min-survivors", type=int, default=96)
    parser.add_argument("--critical-max-q2-projections", type=int, default=192)
    parser.add_argument("--high-max-q2-projections", type=int, default=252)
    parser.add_argument("--baseline-max-q2-projections", type=int, default=576)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        print("Hy3 layer constraints self-test: PASS")
        return
    args = parse_args()
    result = build_constraints(args)
    write_json(args.out, result)
    print(f"wrote {args.out} sha256={sha256(args.out)} layers={len(result['layers'])}")


if __name__ == "__main__":
    main()
