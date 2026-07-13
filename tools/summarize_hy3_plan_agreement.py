#!/usr/bin/env python3
"""Summarize stable and disputed decisions across private Hy3 mixed-quant plans."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


PLAN_FORMAT = "bw24-expert-tier-plan-v2"
SCORE_FORMAT = "bw24-expert-retention-scores-v1"
OUTPUT_FORMAT = "bw24-hy3-smart-plan-agreement-v1"
PROJECTIONS = ("gate", "up", "down")
PRUNED = "PRUNE"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expand_plan(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("format") != PLAN_FORMAT:
        raise ValueError(f"{path} is not a {PLAN_FORMAT} plan")
    if payload.get("calibration", {}).get("public_eval_data_used_for_selection") is not False:
        raise ValueError(f"{path} is not bound to private-only selection evidence")
    model = payload.get("model", {})
    expert_count = int(model["expert_count"])
    layers = tuple(int(layer) for layer in model["moe_layers"])
    pruned = {
        (int(layer_text), int(expert))
        for layer_text, experts in payload["pruned_experts"].items()
        for expert in experts
    }
    states: dict[tuple[int, int, str], str] = {}
    for layer in layers:
        for expert in range(expert_count):
            if (layer, expert) in pruned:
                for projection in PROJECTIONS:
                    states[(layer, expert, projection)] = PRUNED
    for row in payload["assignments"]:
        layer, qtype = int(row["layer"]), str(row["qtype"])
        for expert in (int(value) for value in row["experts"]):
            if (layer, expert) in pruned:
                raise ValueError(f"{path} assigns pruned layer {layer} expert {expert}")
            for projection in row["projections"]:
                if projection not in PROJECTIONS:
                    raise ValueError(f"{path} has unknown projection {projection}")
                key = (layer, expert, projection)
                if key in states:
                    raise ValueError(f"{path} assigns {key} more than once")
                states[key] = qtype
    expected = {
        (layer, expert, projection)
        for layer in layers
        for expert in range(expert_count)
        for projection in PROJECTIONS
    }
    if set(states) != expected:
        missing = sorted(expected - set(states))[:5]
        extra = sorted(set(states) - expected)[:5]
        raise ValueError(f"{path} has incomplete state coverage missing={missing} extra={extra}")
    return {
        "name": path.stem,
        "path": path.resolve(),
        "sha256": sha256(path),
        "model": {"expert_count": expert_count, "moe_layers": layers},
        "states": states,
        "pruned": pruned,
    }


def build_summary(
    paths: list[Path], retention_scores: Path | None = None
) -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("at least two plans are required")
    plans = [expand_plan(path) for path in paths]
    reference = plans[0]["model"]
    if any(plan["model"] != reference for plan in plans[1:]):
        raise ValueError("plans do not describe the same MoE shape")
    layers = reference["moe_layers"]
    expert_count = reference["expert_count"]
    all_keys = tuple(plans[0]["states"])
    qtypes = sorted({state for plan in plans for state in plan["states"].values()} - {PRUNED})

    pairwise = []
    for left, right in itertools.combinations(plans, 2):
        left_pruned, right_pruned = left["pruned"], right["pruned"]
        jointly_retained = [
            key for key in all_keys
            if left["states"][key] != PRUNED and right["states"][key] != PRUNED
        ]
        pairwise.append({
            "left": left["name"],
            "right": right["name"],
            "prune_jaccard": len(left_pruned & right_pruned) / max(len(left_pruned | right_pruned), 1),
            "same_pruned_experts": len(left_pruned & right_pruned),
            "union_pruned_experts": len(left_pruned | right_pruned),
            "all_state_agreement_fraction": sum(
                left["states"][key] == right["states"][key] for key in all_keys
            ) / len(all_keys),
            "jointly_retained_qtype_agreement_fraction": sum(
                left["states"][key] == right["states"][key] for key in jointly_retained
            ) / max(len(jointly_retained), 1),
            "jointly_retained_projections": len(jointly_retained),
        })

    stable_pruned: dict[str, list[int]] = {}
    layer_rows: dict[str, Any] = {}
    stable_qtypes: Counter[str] = Counter()
    stable_states = 0
    variable_prune_experts = 0
    projection_stable: Counter[str] = Counter()
    projection_total: Counter[str] = Counter()
    projection_qtypes: dict[str, Counter[str]] = {
        projection: Counter() for projection in PROJECTIONS
    }
    for layer in layers:
        stable_pruned[str(layer)] = []
        layer_stable = 0
        layer_variable_prune = 0
        layer_qtypes: Counter[str] = Counter()
        for expert in range(expert_count):
            prune_states = {
                plan["states"][(layer, expert, PROJECTIONS[0])] == PRUNED for plan in plans
            }
            if len(prune_states) > 1:
                variable_prune_experts += 1
                layer_variable_prune += 1
            elif prune_states == {True}:
                stable_pruned[str(layer)].append(expert)
            for projection in PROJECTIONS:
                key = (layer, expert, projection)
                values = {plan["states"][key] for plan in plans}
                projection_total[projection] += 1
                if len(values) != 1:
                    continue
                state = next(iter(values))
                stable_states += 1
                layer_stable += 1
                projection_stable[projection] += 1
                if state != PRUNED:
                    stable_qtypes[state] += 1
                    layer_qtypes[state] += 1
                    projection_qtypes[projection][state] += 1
        layer_rows[str(layer)] = {
            "all_state_agreement_fraction": layer_stable / (expert_count * len(PROJECTIONS)),
            "stable_pruned_experts": len(stable_pruned[str(layer)]),
            "variable_prune_experts": layer_variable_prune,
            "stable_qtype_projection_counts": {
                qtype: layer_qtypes[qtype] for qtype in qtypes
            },
        }
    result = {
        "format": OUTPUT_FORMAT,
        "inputs": [
            {"name": plan["name"], "path": str(plan["path"]), "sha256": plan["sha256"]}
            for plan in plans
        ],
        "model": {
            "expert_count": expert_count,
            "moe_layers": list(layers),
            "projections": list(PROJECTIONS),
        },
        "pairwise": pairwise,
        "consensus": {
            "all_state_agreement_fraction": stable_states / len(all_keys),
            "stable_state_projections": stable_states,
            "total_state_projections": len(all_keys),
            "stable_pruned_experts": sum(len(values) for values in stable_pruned.values()),
            "stable_pruned_expert_ids_by_layer": stable_pruned,
            "variable_prune_experts": variable_prune_experts,
            "stable_retained_projection_qtype_counts": {
                qtype: stable_qtypes[qtype] for qtype in qtypes
            },
        },
        "projection_agreement": {
            projection: {
                "all_state_agreement_fraction": projection_stable[projection] / projection_total[projection],
                "stable_retained_qtype_counts": {
                    qtype: projection_qtypes[projection][qtype] for qtype in qtypes
                },
            }
            for projection in PROJECTIONS
        },
        "layers": layer_rows,
        "public_eval_data_used_for_selection": False,
    }
    if retention_scores is not None:
        attach_traffic_overlay(result, plans, retention_scores)
    return result


def attach_traffic_overlay(
    result: dict[str, Any], plans: list[dict[str, Any]], path: Path
) -> None:
    payload = json.loads(path.read_text())
    if payload.get("format") != SCORE_FORMAT:
        raise ValueError(f"{path} is not a {SCORE_FORMAT} file")
    if payload.get("calibration", {}).get("public_eval_data_used_for_selection") is not False:
        raise ValueError(f"{path} is not private-only retention evidence")
    layers = tuple(int(layer) for layer in result["model"]["moe_layers"])
    expert_count = int(result["model"]["expert_count"])
    expected = {(layer, expert) for layer in layers for expert in range(expert_count)}
    mass: dict[tuple[int, int], float] = {}
    frequency: dict[tuple[int, int], int] = {}
    for row in payload["scores"]:
        key = (int(row["layer"]), int(row["expert"]))
        if key in mass:
            raise ValueError(f"{path} repeats score row {key}")
        mass[key] = float(row["router_weight_mass"])
        frequency[key] = int(row["frequency"])
        if mass[key] < 0 or frequency[key] < 0:
            raise ValueError(f"{path} has negative traffic for {key}")
    if set(mass) != expected:
        raise ValueError(f"{path} does not cover the plan's complete expert bank")
    total_mass, total_frequency = sum(mass.values()), sum(frequency.values())
    if total_mass <= 0 or total_frequency <= 0:
        raise ValueError(f"{path} has zero total traffic")
    stable_pruned = {
        (int(layer), int(expert))
        for layer, experts in result["consensus"]["stable_pruned_expert_ids_by_layer"].items()
        for expert in experts
    }
    result["traffic_overlay"] = {
        "source": {"path": str(path.resolve()), "sha256": sha256(path)},
        "stable_pruned_router_weight_mass_fraction":
            sum(mass[key] for key in stable_pruned) / total_mass,
        "stable_pruned_frequency_fraction":
            sum(frequency[key] for key in stable_pruned) / total_frequency,
        "plans": {},
    }
    for plan in plans:
        state_mass: Counter[str] = Counter()
        state_frequency: Counter[str] = Counter()
        for (layer, expert, _projection), state in plan["states"].items():
            key = (layer, expert)
            state_mass[state] += mass[key] / len(PROJECTIONS)
            state_frequency[state] += frequency[key] / len(PROJECTIONS)
        result["traffic_overlay"]["plans"][plan["name"]] = {
            "pruned_router_weight_mass_fraction":
                sum(mass[key] for key in plan["pruned"]) / total_mass,
            "pruned_frequency_fraction":
                sum(frequency[key] for key in plan["pruned"]) / total_frequency,
            "state_router_weight_mass_fraction": {
                state: value / total_mass for state, value in sorted(state_mass.items())
            },
            "state_frequency_fraction": {
                state: value / total_frequency
                for state, value in sorted(state_frequency.items())
            },
        }
    for layer in layers:
        layer_mass = sum(mass[(layer, expert)] for expert in range(expert_count))
        layer_frequency = sum(frequency[(layer, expert)] for expert in range(expert_count))
        layer_stable = {
            (layer, int(expert))
            for expert in result["consensus"]["stable_pruned_expert_ids_by_layer"][str(layer)]
        }
        result["layers"][str(layer)].update({
            "stable_pruned_router_weight_mass_fraction":
                sum(mass[key] for key in layer_stable) / max(layer_mass, 1e-30),
            "stable_pruned_frequency_fraction":
                sum(frequency[key] for key in layer_stable) / max(layer_frequency, 1),
        })


def write_layer_csv(result: dict[str, Any], path: Path) -> None:
    qtypes = sorted(result["consensus"]["stable_retained_projection_qtype_counts"])
    fieldnames = [
        "layer", "all_state_agreement_fraction", "stable_pruned_experts",
        "variable_prune_experts",
    ]
    if "traffic_overlay" in result:
        fieldnames += [
            "stable_pruned_router_weight_mass_fraction",
            "stable_pruned_frequency_fraction",
        ]
    fieldnames += [f"stable_{qtype.lower()}_projections" for qtype in qtypes]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for layer, row in result["layers"].items():
            writer.writerow({
                "layer": layer,
                "all_state_agreement_fraction": row["all_state_agreement_fraction"],
                "stable_pruned_experts": row["stable_pruned_experts"],
                "variable_prune_experts": row["variable_prune_experts"],
                **({
                    "stable_pruned_router_weight_mass_fraction":
                        row["stable_pruned_router_weight_mass_fraction"],
                    "stable_pruned_frequency_fraction":
                        row["stable_pruned_frequency_fraction"],
                } if "traffic_overlay" in result else {}),
                **{
                    f"stable_{qtype.lower()}_projections":
                        row["stable_qtype_projection_counts"][qtype]
                    for qtype in qtypes
                },
            })


def self_test() -> None:
    def make_plan(
        path: Path,
        pruned: dict[int, list[int]],
        overrides: dict[tuple[int, int, str], str],
    ) -> None:
        assignments = []
        for layer in (1, 2):
            for expert in range(3):
                if expert in pruned.get(layer, []):
                    continue
                for projection in PROJECTIONS:
                    assignments.append({
                        "layer": layer,
                        "experts": [expert],
                        "projections": [projection],
                        "qtype": overrides.get((layer, expert, projection), "Q2_K"),
                    })
        path.write_text(json.dumps({
            "format": PLAN_FORMAT,
            "model": {"expert_count": 3, "moe_layers": [1, 2]},
            "calibration": {"public_eval_data_used_for_selection": False},
            "pruned_experts": {str(layer): values for layer, values in pruned.items()},
            "assignments": assignments,
        }))

    with tempfile.TemporaryDirectory(prefix="bw24-plan-agreement-") as tmp:
        root = Path(tmp)
        make_plan(root / "a.json", {1: [2], 2: []}, {(2, 1, "down"): "Q3_K"})
        make_plan(root / "b.json", {1: [2], 2: []}, {(2, 1, "down"): "Q8_0"})
        make_plan(root / "c.json", {1: [1], 2: []}, {(2, 1, "down"): "Q3_K"})
        scores = root / "scores.json"
        scores.write_text(json.dumps({
            "format": SCORE_FORMAT,
            "calibration": {"public_eval_data_used_for_selection": False},
            "scores": [
                {"layer": layer, "expert": expert,
                 "router_weight_mass": 1 + expert, "frequency": 2 + expert}
                for layer in (1, 2) for expert in range(3)
            ],
        }))
        result = build_summary(
            [root / "a.json", root / "b.json", root / "c.json"], scores
        )
        assert result["model"]["expert_count"] == 3
        assert result["consensus"]["stable_pruned_experts"] == 0
        assert result["consensus"]["variable_prune_experts"] == 2
        assert result["layers"]["2"]["all_state_agreement_fraction"] == 8 / 9
        assert result["traffic_overlay"]["stable_pruned_router_weight_mass_fraction"] == 0
        for plan in result["traffic_overlay"]["plans"].values():
            assert abs(sum(plan["state_router_weight_mass_fraction"].values()) - 1) < 1e-12
        assert len(result["pairwise"]) == 3
        csv_path = root / "layers.csv"
        write_layer_csv(result, csv_path)
        assert csv_path.read_text().splitlines()[0].startswith(
            "layer,all_state_agreement_fraction"
        )


def main() -> None:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        print("Hy3 smart plan agreement self-test: PASS")
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("plans", nargs="+", type=Path)
    parser.add_argument("--retention-scores", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layer-csv", type=Path)
    args = parser.parse_args()
    result = build_summary(args.plans, args.retention_scores)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.layer_csv:
        write_layer_csv(result, args.layer_csv)
    print(f"wrote {args.out} sha256={sha256(args.out)}")


if __name__ == "__main__":
    main()
