#!/usr/bin/env python3
"""Allocate Hy3 expert projection precision and whole-expert pruning under a byte budget.

The objective uses private routed-token output damage measured with the exact artifact quantizers.
Optional REAP/domain, low-confidence rescue, and layer-reconstruction terms protect specialized or
depth-sensitive functions.  Public evaluation data is rejected by provenance checks.
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

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array


PLAN_FORMAT = "bw24-expert-tier-plan-v2"
RETENTION_FORMAT = "bw24-expert-retention-scores-v1"
SENSITIVITY_FORMAT = "bw24-hy3-quant-sensitivity-v1"
LAYER_CONSTRAINTS_FORMAT = "bw24-hy3-layer-constraints-v1"
PROJECTIONS = ("gate", "up", "down")
QTYPES = {
    "Q8_0": (32, 34),
    "NVFP4": (64, 36),
    "IQ3_S": (256, 110),
    "IQ4_XS": (256, 136),
    "Q4_K": (256, 144),
    "Q3_K": (256, 110),
    "Q2_K": (256, 84),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def projection_bytes(projection: str, hidden: int, intermediate: int, qtype: str) -> int:
    rows, cols = (
        (intermediate, hidden) if projection in ("gate", "up") else (hidden, intermediate)
    )
    block, encoded = QTYPES[qtype]
    if cols % block:
        raise ValueError(f"{projection}/{qtype}: input width {cols} is not block aligned")
    return rows * (cols // block) * encoded


def percentile(values: dict[int, float]) -> dict[int, float]:
    ranked = sorted(values, key=lambda key: (values[key], key))
    if len(ranked) == 1:
        return {ranked[0]: 1.0}
    return {key: index / (len(ranked) - 1) for index, key in enumerate(ranked)}


def dominated_qtypes(
    projection_values: dict[str, float], projection_sizes: dict[str, int]
) -> dict[str, str]:
    """Return states that cannot improve any byte-constrained solution."""
    if projection_values.keys() != projection_sizes.keys():
        raise ValueError("projection values and sizes cover different qtypes")
    for qtype, value in projection_values.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid projection damage for {qtype}: {value}")
    dominated: dict[str, str] = {}
    for candidate, candidate_value in projection_values.items():
        candidate_size = projection_sizes[candidate]
        witnesses = [
            other
            for other, other_value in projection_values.items()
            if other != candidate
            and projection_sizes[other] <= candidate_size
            and other_value <= candidate_value
            and (
                projection_sizes[other] < candidate_size
                or other_value < candidate_value
            )
        ]
        if witnesses:
            dominated[candidate] = min(
                witnesses,
                key=lambda other: (
                    projection_sizes[other], projection_values[other], other
                ),
            )
    return dominated


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    retention = load_json(args.retention_scores)
    sensitivity = load_json(args.quant_sensitivity)
    reference = load_json(args.reference_plan)
    confidence = load_json(args.confidence_plan) if args.confidence_plan else None
    layer_constraints = load_json(args.layer_constraints) if args.layer_constraints else None
    if retention.get("format") != RETENTION_FORMAT:
        raise ValueError("unsupported retention score format")
    if sensitivity.get("format") != SENSITIVITY_FORMAT:
        raise ValueError("unsupported quant sensitivity format")
    if reference.get("format") != PLAN_FORMAT:
        raise ValueError("unsupported reference plan format")
    for payload, field in ((retention, "calibration"), (sensitivity, "calibration")):
        if payload[field].get("public_eval_data_used_for_selection") is not False:
            raise ValueError("all selection evidence must be private")
    if confidence is not None:
        if confidence.get("format") != PLAN_FORMAT:
            raise ValueError("unsupported confidence plan format")
        if confidence["calibration"].get("public_eval_data_used_for_selection") is not False:
            raise ValueError("confidence evidence must be private")
        confidence_rows = confidence.get("score_diagnostics", {}).get("experts")
        if not isinstance(confidence_rows, list):
            raise ValueError("confidence plan lacks full expert diagnostics")
    else:
        confidence_rows = []

    model = sensitivity["model"]
    qtypes = tuple(sensitivity.get("measurement", {}).get("qtypes", ()))
    if not qtypes or len(set(qtypes)) != len(qtypes) or any(qtype not in QTYPES for qtype in qtypes):
        raise ValueError(f"sensitivity qtypes must be distinct values drawn from {tuple(QTYPES)}")
    layers = [int(x) for x in model["moe_layers"]]
    expert_count = int(model["expert_count"])
    hidden, intermediate = int(model["hidden_size"]), int(model["intermediate_size"])
    experts = [(layer, expert) for layer in layers for expert in range(expert_count)]
    layer_policy: dict[int, dict[str, int]] = {
        layer: {
            "min_survivors": args.min_survivors_per_layer,
            "max_q2_projections": expert_count * len(PROJECTIONS),
        }
        for layer in layers
    }
    if layer_constraints is not None:
        if layer_constraints.get("format") != LAYER_CONSTRAINTS_FORMAT:
            raise ValueError("unsupported layer constraints format")
        calibration = layer_constraints.get("calibration", {})
        if calibration.get("public_eval_data_used_for_selection") is not False:
            raise ValueError("layer constraints must use private selection evidence")
        rows = layer_constraints.get("layers")
        if not isinstance(rows, dict) or set(rows) != {str(layer) for layer in layers}:
            raise ValueError("layer constraints must exactly cover model MoE layers")
        for layer in layers:
            row = rows[str(layer)]
            if not isinstance(row, dict):
                raise ValueError(f"invalid layer constraints row for layer {layer}")
            minimum = int(row.get("min_survivors", args.min_survivors_per_layer))
            max_q2 = int(row.get("max_q2_projections", expert_count * len(PROJECTIONS)))
            if not args.min_survivors_per_layer <= minimum <= expert_count:
                raise ValueError(f"invalid min_survivors for layer {layer}: {minimum}")
            if not 0 <= max_q2 <= expert_count * len(PROJECTIONS):
                raise ValueError(f"invalid max_q2_projections for layer {layer}: {max_q2}")
            layer_policy[layer] = {
                "min_survivors": minimum,
                "max_q2_projections": max_q2,
            }
    retention_rows = {
        (int(row["layer"]), int(row["expert"])): row for row in retention["scores"]
    }
    sensitivity_rows = {
        (int(row["layer"]), int(row["expert"])): row for row in sensitivity["scores"]
    }
    if any(set(row.get("quantization", {})) != set(qtypes) for row in sensitivity_rows.values()):
        raise ValueError("sensitivity rows do not exactly cover their declared qtypes")
    external = set(qtypes) & {"IQ3_S", "IQ4_XS", "Q4_K"}
    if external:
        sidecars = sensitivity.get("importance_sidecars", {})
        if set(sidecars) != {str(layer) for layer in layers}:
            raise ValueError(f"{sorted(external)} require one private importance sidecar per layer")
        provenance = sensitivity["measurement"].get("exact_quantizer_implementation", {})
        if not isinstance(provenance, dict) or any(qtype not in provenance for qtype in external):
            raise ValueError(f"{sorted(external)} lack exact quantizer provenance")
    confidence_scores = {
        (int(row["layer"]), int(row["expert"])): float(row["score"])
        for row in confidence_rows
    }
    if retention_rows.keys() != set(experts) or sensitivity_rows.keys() != set(experts):
        raise ValueError("retention or sensitivity coverage does not match the model")
    if confidence is not None and confidence_scores.keys() != set(experts):
        raise ValueError("confidence coverage does not match the model")

    layer_before: dict[int, float] = {}
    receipt_hashes = []
    if args.joint_receipts:
        for layer in layers:
            path = args.joint_receipts / f"layer-{layer:03}.receipt.json"
            receipt = load_json(path)
            if receipt.get("mode") != "joint" or int(receipt.get("layer", -1)) != layer:
                raise ValueError(f"invalid joint-heal receipt {path}")
            if receipt.get("public_eval_data_used_for_healing") is not False:
                raise ValueError("joint-heal receipt lacks private-data attestation")
            layer_before[layer] = float(receipt["before"]["normalized_mse"])
            receipt_hashes.append({"path": str(path.resolve()), "sha256": sha256(path)})
    else:
        layer_before = {layer: 0.0 for layer in layers}
    layer_percentile = percentile(layer_before)

    retention_values = np.asarray(
        [float(retention_rows[key]["retain_score"]) for key in experts], dtype=np.float64
    )
    retention_max = max(float(retention_values.max(initial=0.0)), 1e-30)
    importance: dict[tuple[int, int], float] = {}
    for key in experts:
        retain = float(retention_rows[key]["retain_score"]) / retention_max
        rescue = confidence_scores.get(key, 0.0)
        depth = layer_percentile[key[0]]
        importance[key] = (
            1.0 + args.retention_weight * retain
            + args.confidence_weight * rescue + args.layer_weight * depth
        )
        if importance[key] <= 0 or not math.isfinite(importance[key]):
            raise ValueError(f"invalid importance multiplier for {key}")

    fixed_bytes = int(reference["policy"]["fixed_non_expert_bytes"])
    expert_budget = args.target_logical_bytes - fixed_bytes
    if expert_budget <= 0:
        raise ValueError("target leaves no expert byte budget")

    # One prune variable per expert, followed by one variable per expert/projection/qtype.
    prune_index = {key: index for index, key in enumerate(experts)}
    quant_index: dict[tuple[int, int, str, str], int] = {}
    next_index = len(experts)
    for layer, expert in experts:
        for projection in PROJECTIONS:
            for qtype in qtypes:
                quant_index[(layer, expert, projection, qtype)] = next_index
                next_index += 1
    n_variables = next_index
    objective = np.zeros(n_variables, dtype=np.float64)
    bytes_vector = np.zeros(n_variables, dtype=np.float64)
    objective_offset = 0.0
    dominated_states: dict[tuple[int, int, str, str], str] = {}
    for key in experts:
        row = sensitivity_rows[key]
        first_qtype = qtypes[0]
        prune_damage = float(
            row["quantization"][first_qtype]["joint_output_error"]["baseline_energy"]
        ) * float(row["sample_scale"])
        prune_damage *= importance[key]
        projection_minima: dict[str, float] = {}
        for projection in PROJECTIONS:
            projection_values: dict[str, float] = {}
            projection_sizes: dict[str, int] = {}
            for qtype in qtypes:
                metric = row["quantization"][qtype]["projection_output_error"][projection]
                index = quant_index[(*key, projection, qtype)]
                projection_values[qtype] = (
                    float(metric["squared_error"]) * float(row["sample_scale"]) * importance[key]
                )
                objective[index] = projection_values[qtype]
                projection_sizes[qtype] = projection_bytes(
                    projection, hidden, intermediate, qtype
                )
                bytes_vector[index] = projection_sizes[qtype]
            dominated_states.update({
                (*key, projection, qtype): witness
                for qtype, witness in dominated_qtypes(
                    projection_values, projection_sizes
                ).items()
            })
            projection_minima[projection] = min(projection_values.values())

        # Each feasible expert contributes either one prune variable or one quant variable for
        # every projection.  Subtracting the corresponding per-projection minimum from both sides
        # therefore removes the same constant from every feasible solution.  This preserves the
        # exact argmin while preventing a large, unavoidable Q8 reconstruction floor in one cell
        # from swallowing HiGHS' relative-gap tolerance for the rest of the model.
        expert_offset = sum(projection_minima.values())
        centered_prune_damage = prune_damage - expert_offset
        tolerance = 1e-12 * max(abs(prune_damage), abs(expert_offset), 1.0)
        if centered_prune_damage < -tolerance:
            raise ValueError(
                f"best retained quant damage exceeds prune damage for {key}: "
                f"retained={expert_offset} prune={prune_damage}"
            )
        objective[prune_index[key]] = max(centered_prune_damage, 0.0)
        for projection in PROJECTIONS:
            minimum = projection_minima[projection]
            for qtype in qtypes:
                index = quant_index[(*key, projection, qtype)]
                objective[index] = max(objective[index] - minimum, 0.0)
        objective_offset += expert_offset
    scale = max(float(objective.max(initial=0.0)), 1e-30)
    objective /= scale

    row_ids: list[int] = []
    col_ids: list[int] = []
    data: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    row = 0
    # Every retained projection chooses exactly one qtype; the shared prune variable disables all 3.
    for key in experts:
        for projection in PROJECTIONS:
            row_ids.append(row); col_ids.append(prune_index[key]); data.append(1.0)
            for qtype in qtypes:
                row_ids.append(row); col_ids.append(quant_index[(*key, projection, qtype)])
                data.append(1.0)
            lower.append(1.0); upper.append(1.0); row += 1
    # Exact global byte ceiling.
    for index, value in enumerate(bytes_vector):
        if value:
            row_ids.append(row); col_ids.append(index); data.append(value)
    lower.append(-np.inf); upper.append(float(expert_budget)); row += 1
    # Per-layer structural constraints.  The optional private policy can raise the survivor floor
    # or cap Q2 assignments in fragile layers without changing the global byte ceiling.
    for layer in layers:
        for expert in range(expert_count):
            row_ids.append(row)
            col_ids.append(prune_index[(layer, expert)])
            data.append(1.0)
        lower.append(-np.inf)
        upper.append(float(expert_count - layer_policy[layer]["min_survivors"]))
        row += 1
        if "Q2_K" in qtypes:
            for expert in range(expert_count):
                for projection in PROJECTIONS:
                    row_ids.append(row)
                    col_ids.append(quant_index[(layer, expert, projection, "Q2_K")])
                    data.append(1.0)
            lower.append(-np.inf)
            upper.append(float(layer_policy[layer]["max_q2_projections"]))
            row += 1
    matrix = coo_array(
        (np.asarray(data), (np.asarray(row_ids), np.asarray(col_ids))),
        shape=(row, n_variables),
    ).tocsr()
    protected = {
        key for key in experts if bool(retention_rows[key].get("protected", False))
    }
    lb = np.zeros(n_variables)
    ub = np.ones(n_variables)
    for state in dominated_states:
        ub[quant_index[state]] = 0.0
    for key in protected:
        ub[prune_index[key]] = 0.0
    result = milp(
        c=objective,
        integrality=np.ones(n_variables, dtype=np.uint8),
        bounds=Bounds(lb, ub),
        constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
        options={
            "mip_rel_gap": float(args.mip_rel_gap),
            "time_limit": float(args.time_limit_seconds),
        },
    )
    if result.x is None or int(result.status) not in (0, 1):
        raise RuntimeError(f"smart budget solver failed: {result.status} {result.message}")
    rounded = np.rint(result.x)
    if float(np.max(np.abs(result.x - rounded), initial=0.0)) > 1e-6:
        raise RuntimeError("smart budget solver returned a non-integral incumbent")
    activity = matrix @ rounded
    if np.any(activity < np.asarray(lower) - 1e-5) or np.any(
        activity > np.asarray(upper) + 1e-5
    ):
        raise RuntimeError("smart budget solver incumbent violates frozen constraints")
    result.x = rounded
    selected_dominated = [
        state for state in dominated_states if result.x[quant_index[state]] >= 0.5
    ]
    if selected_dominated:
        raise RuntimeError(f"solver selected dominated quant states: {selected_dominated[:3]}")
    pruned = {key for key in experts if result.x[prune_index[key]] >= 0.5}
    assignments = []
    counts = {qtype: 0 for qtype in qtypes}
    selected_bytes = 0
    for layer in layers:
        for projection in PROJECTIONS:
            for qtype in qtypes:
                ids = [
                    expert for expert in range(expert_count)
                    if (layer, expert) not in pruned
                    and result.x[quant_index[(layer, expert, projection, qtype)]] >= 0.5
                ]
                if ids:
                    assignments.append({
                        "layer": layer, "experts": ids,
                        "projections": [projection], "qtype": qtype,
                    })
                    counts[qtype] += len(ids)
                    selected_bytes += len(ids) * projection_bytes(
                        projection, hidden, intermediate, qtype
                    )
    logical_bytes = fixed_bytes + selected_bytes
    if logical_bytes > args.target_logical_bytes:
        raise AssertionError("solver output violates byte ceiling")
    layer_summary = {}
    for layer in layers:
        layer_pruned = [expert for expert in range(expert_count) if (layer, expert) in pruned]
        layer_summary[str(layer)] = {
            "retained": expert_count - len(layer_pruned),
            "pruned": len(layer_pruned),
            "min_survivors": layer_policy[layer]["min_survivors"],
            "max_q2_projections": layer_policy[layer]["max_q2_projections"],
            "current_prune_damage_normalized_mse": layer_before[layer],
            "current_prune_damage_percentile": layer_percentile[layer],
        }
    return {
        "format": PLAN_FORMAT,
        "recipe": "measured-global-projection-budget",
        "description": "Global per-projection precision and whole-expert prune allocation from private exact-format output damage",
        "model": {
            "expert_count": expert_count,
            "original_expert_count": expert_count,
            "expert_used_count": int(model["top_k"]),
            "moe_layers": layers,
        },
        "policy": {
            "target_logical_bytes": args.target_logical_bytes,
            "fixed_non_expert_bytes": fixed_bytes,
            "expert_byte_budget": expert_budget,
            "result_logical_bytes": logical_bytes,
            "headroom_bytes": args.target_logical_bytes - logical_bytes,
            "min_survivors_per_layer": args.min_survivors_per_layer,
            "layer_constraints": {
                str(layer): layer_policy[layer] for layer in layers
            },
            "rank_metric": "measured_projection_output_error_per_byte",
            "importance_weights": {
                "retention": args.retention_weight,
                "confidence": args.confidence_weight,
                "layer": args.layer_weight,
            },
            "qtype_projection_counts": counts,
            "candidate_qtypes": list(qtypes),
            "solver": "scipy.optimize.milp/HiGHS",
            "solver_status": int(result.status),
            "solver_message": str(result.message),
            "mip_gap": float(getattr(result, "mip_gap", 0.0)),
            "mip_gap_target": args.mip_rel_gap,
            "optimal": int(result.status) == 0,
            "objective_centering": {
                "method": "per_expert_projection_minimum",
                "preserves_exact_argmin": True,
                "constant_offset": objective_offset,
                "centered_scale": scale,
            },
            "per_cell_dominance_pruning": {
                "condition": "other_bytes <= candidate_bytes and other_damage <= candidate_damage with at least one strict inequality",
                "preserves_exact_feasible_argmin": True,
                "disabled_states": len(dominated_states),
                "disabled_by_qtype": {
                    qtype: sum(state[3] == qtype for state in dominated_states)
                    for qtype in qtypes
                },
            },
        },
        "calibration": {
            "retention_scores": {"path": str(args.retention_scores.resolve()), "sha256": sha256(args.retention_scores)},
            "quant_sensitivity": {"path": str(args.quant_sensitivity.resolve()), "sha256": sha256(args.quant_sensitivity)},
            "confidence_plan": ({"path": str(args.confidence_plan.resolve()), "sha256": sha256(args.confidence_plan)} if args.confidence_plan else None),
            "joint_heal_receipts": receipt_hashes,
            "layer_constraints": (
                {"path": str(args.layer_constraints.resolve()), "sha256": sha256(args.layer_constraints)}
                if args.layer_constraints else None
            ),
            "public_eval_data_used_for_selection": False,
        },
        "reference_plan": {"path": str(args.reference_plan.resolve()), "sha256": sha256(args.reference_plan)},
        "selection": {
            "retained_experts": len(experts) - len(pruned),
            "pruned_experts": len(pruned),
            "protected_experts": len(protected),
            "estimated_objective": float(result.fun),
            "estimated_centered_output_damage": float(result.fun) * scale,
            "estimated_absolute_output_damage": float(result.fun) * scale + objective_offset,
        },
        "pruned_experts": {
            str(layer): [expert for expert in range(expert_count) if (layer, expert) in pruned]
            for layer in layers
        },
        "assignments": assignments,
        "layer_summary": layer_summary,
    }


def self_test() -> None:
    assert dominated_qtypes(
        {"IQ3_S": 1.0, "Q3_K": 2.0, "Q2_K": 3.0},
        {"IQ3_S": 110, "Q3_K": 110, "Q2_K": 84},
    ) == {"Q3_K": "IQ3_S"}
    with tempfile.TemporaryDirectory(prefix="bw24-smart-budget-") as tmp:
        root = Path(tmp); layers = [1, 2]; experts = range(3)
        qtypes = ["Q8_0", "NVFP4", "IQ3_S", "Q3_K", "Q2_K"]
        retention = {
            "format": RETENTION_FORMAT,
            "calibration": {"public_eval_data_used_for_selection": False},
            "scores": [
                {"layer": layer, "expert": expert, "retain_score": 1 + expert,
                 "protected": layer == 1 and expert == 2}
                for layer in layers for expert in experts
            ],
        }
        sensitivity_rows = []
        for layer in layers:
            for expert in experts:
                quant = {}
                for rank, qtype in enumerate(qtypes):
                    error = 0.01 * (rank + 1) * (expert + 1)
                    quant[qtype] = {
                        "joint_output_error": {"baseline_energy": 10.0},
                        "projection_output_error": {
                            p: {"squared_error": error} for p in PROJECTIONS
                        },
                    }
                sensitivity_rows.append({
                    "layer": layer, "expert": expert, "sample_scale": 1.0,
                    "quantization": quant,
                })
        sensitivity = {
            "format": SENSITIVITY_FORMAT,
            "model": {"expert_count": 3, "top_k": 1, "hidden_size": 256,
                      "intermediate_size": 256, "moe_layers": layers},
            "measurement": {
                "qtypes": qtypes,
                "exact_quantizer_implementation": {"IQ3_S": {"implementation": "test"}},
            },
            "calibration": {"public_eval_data_used_for_selection": False},
            "importance_sidecars": {str(layer): {"path": "test"} for layer in layers},
            "scores": sensitivity_rows,
        }
        confidence = {
            "format": PLAN_FORMAT,
            "calibration": {"public_eval_data_used_for_selection": False},
            "score_diagnostics": {"experts": [
                {"layer": layer, "expert": expert, "score": expert / 2}
                for layer in layers for expert in experts
            ]},
        }
        q2_projection = projection_bytes("gate", 256, 256, "Q2_K")
        reference = {"format": PLAN_FORMAT, "policy": {"fixed_non_expert_bytes": 100}}
        paths = {}
        for name, payload in {
            "retention": retention, "sensitivity": sensitivity,
            "confidence": confidence, "reference": reference,
        }.items():
            paths[name] = root / f"{name}.json"; paths[name].write_text(json.dumps(payload))
        args = argparse.Namespace(
            retention_scores=paths["retention"], quant_sensitivity=paths["sensitivity"],
            confidence_plan=paths["confidence"], reference_plan=paths["reference"],
            joint_receipts=None, target_logical_bytes=100 + 6 * 3 * q2_projection,
            min_survivors_per_layer=1, retention_weight=1.0, confidence_weight=1.0,
            layer_weight=0.0, time_limit_seconds=30,
            mip_rel_gap=1e-4, layer_constraints=None,
        )
        plan = build_plan(args)
        assert plan["policy"]["result_logical_bytes"] <= args.target_logical_bytes
        assert plan["selection"]["protected_experts"] == 1
        assert 2 not in plan["pruned_experts"]["1"]
        centering = plan["policy"]["objective_centering"]
        assert centering["method"] == "per_expert_projection_minimum"
        assert centering["preserves_exact_argmin"] is True
        assert centering["constant_offset"] > 0
        dominance = plan["policy"]["per_cell_dominance_pruning"]
        assert dominance["preserves_exact_feasible_argmin"] is True
        assert dominance["disabled_by_qtype"]["Q3_K"] == 18
        assert plan["policy"]["qtype_projection_counts"]["Q3_K"] == 0
        assert math.isclose(
            plan["selection"]["estimated_absolute_output_damage"],
            plan["selection"]["estimated_centered_output_damage"]
            + centering["constant_offset"],
        )
        layer_constraints = {
            "format": LAYER_CONSTRAINTS_FORMAT,
            "calibration": {"public_eval_data_used_for_selection": False},
            "layers": {
                "1": {"min_survivors": 3, "max_q2_projections": 0},
                "2": {"min_survivors": 1, "max_q2_projections": 9},
            },
        }
        paths["layer_constraints"] = root / "layer-constraints.json"
        paths["layer_constraints"].write_text(json.dumps(layer_constraints))
        args.layer_constraints = paths["layer_constraints"]
        constrained = build_plan(args)
        assert constrained["layer_summary"]["1"]["retained"] == 3
        assert constrained["layer_summary"]["1"]["max_q2_projections"] == 0
        assert not any(
            row["layer"] == 1 and row["qtype"] == "Q2_K"
            for row in constrained["assignments"]
        )
        layer_constraints["calibration"]["public_eval_data_used_for_selection"] = True
        paths["layer_constraints"].write_text(json.dumps(layer_constraints))
        try:
            build_plan(args)
        except ValueError as error:
            assert "private selection evidence" in str(error)
        else:
            raise AssertionError("public-derived layer constraints were accepted")
        from prepare_mixed_expert_repack import load_assignments
        path = root / "plan.json"; path.write_text(json.dumps(plan))
        _, expanded, pruned = load_assignments(path)
        assert len(expanded) == plan["selection"]["retained_experts"] * 3
        assert sum(len(x) for x in pruned.values()) == plan["selection"]["pruned_experts"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retention-scores", type=Path, required=True)
    parser.add_argument("--quant-sensitivity", type=Path, required=True)
    parser.add_argument("--confidence-plan", type=Path)
    parser.add_argument("--layer-constraints", type=Path)
    parser.add_argument("--joint-receipts", type=Path)
    parser.add_argument("--reference-plan", type=Path, required=True)
    parser.add_argument("--target-logical-bytes", type=int, default=100_000_000_000)
    parser.add_argument("--min-survivors-per-layer", type=int, default=96)
    parser.add_argument("--retention-weight", type=float, default=1.0)
    parser.add_argument("--confidence-weight", type=float, default=1.0)
    parser.add_argument("--layer-weight", type=float, default=1.0)
    parser.add_argument("--time-limit-seconds", type=int, default=1800)
    parser.add_argument("--mip-rel-gap", type=float, default=1e-4)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    if sys.argv[1:] == ["--self-test"]:
        self_test(); print("Hy3 smart budget plan self-test: PASS"); return
    args = parse_args()
    for name in ("retention_weight", "confidence_weight", "layer_weight"):
        if getattr(args, name) < 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be non-negative")
    if not 0 <= args.mip_rel_gap < 1:
        raise SystemExit("--mip-rel-gap must be in [0, 1)")
    plan = build_plan(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out} sha256={sha256(args.out)} logical_bytes={plan['policy']['result_logical_bytes']}")


if __name__ == "__main__":
    main()
