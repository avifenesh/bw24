#!/usr/bin/env python3
"""Recompute additive private damage for complete Hy3 prune/quant plans."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import tempfile
from pathlib import Path
from typing import Any


SENSITIVITY_FORMAT = "bw24-hy3-quant-sensitivity-v1"
PLAN_FORMAT = "bw24-expert-tier-plan-v2"
OUTPUT_FORMAT = "bw24-hy3-quant-plan-damage-v1"
RECEIPT_FORMAT = "bw24-hy3-quant-plan-damage-receipt-v1"
PROJECTIONS = ("gate", "up", "down")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def parse_plan_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("plan must be NAME=PATH")
    name, raw = value.split("=", 1)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise argparse.ArgumentTypeError(f"invalid plan name {name!r}")
    return name, Path(raw)


def parse_logical_bytes_spec(value: str) -> tuple[str, int]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("logical bytes must be NAME=BYTES")
    name, raw = value.split("=", 1)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise argparse.ArgumentTypeError(f"invalid plan name {name!r}")
    try:
        logical_bytes = int(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError("logical bytes must be an integer") from error
    if logical_bytes <= 0:
        raise argparse.ArgumentTypeError("logical bytes must be positive")
    return name, logical_bytes


def expand_plan(
    plan: dict[str, Any], layers: list[int], expert_count: int, qtypes: set[str]
) -> tuple[set[tuple[int, int]], dict[tuple[int, int, str], str]]:
    if plan.get("format") != PLAN_FORMAT:
        raise ValueError("unsupported plan format")
    if plan.get("calibration", {}).get("public_eval_data_used_for_selection") is not False:
        raise ValueError("plan does not attest private-only selection")
    expected_experts = {(layer, expert) for layer in layers for expert in range(expert_count)}
    raw_pruned = plan.get("pruned_experts")
    if raw_pruned is None or raw_pruned == {}:
        policy = plan.get("policy", {})
        # Frozen full-bank traffic plans encode the absence of pruning as a
        # policy invariant instead of materializing 79 empty lists.  Accept
        # that compact representation only when both no-prune fields agree.
        if (
            int(policy.get("fixed_prune_count", -1)) == 0
            and policy.get("prune_unused") is False
        ):
            raw_pruned = {str(layer): [] for layer in layers}
        else:
            raise ValueError(
                "plan omits or empties pruned_experts without a no-prune policy"
            )
    if set(raw_pruned) != {str(layer) for layer in layers}:
        raise ValueError("plan pruned_experts layer coverage mismatch")
    pruned = {
        (int(layer), int(expert))
        for layer, experts in raw_pruned.items()
        for expert in experts
    }
    if not pruned <= expected_experts:
        raise ValueError("plan prunes an out-of-range expert")
    assigned: dict[tuple[int, int, str], str] = {}
    for row in plan.get("assignments", []):
        layer = int(row["layer"])
        projections = tuple(row.get("projections", PROJECTIONS))
        qtype = str(row["qtype"])
        if layer not in layers or not projections or any(p not in PROJECTIONS for p in projections):
            raise ValueError("invalid assignment layer or projection")
        if qtype not in qtypes:
            raise ValueError(f"plan uses unmeasured qtype {qtype}")
        for expert in row["experts"]:
            key = (layer, int(expert))
            if key not in expected_experts or key in pruned:
                raise ValueError("assignment targets invalid or pruned expert")
            for projection in projections:
                cell = (*key, projection)
                if cell in assigned:
                    raise ValueError(f"duplicate assignment {cell}")
                assigned[cell] = qtype
    expected_cells = {
        (layer, expert, projection)
        for layer, expert in expected_experts - pruned
        for projection in PROJECTIONS
    }
    if assigned.keys() != expected_cells:
        missing = len(expected_cells - assigned.keys())
        extra = len(assigned.keys() - expected_cells)
        raise ValueError(f"assignment coverage mismatch: missing={missing} extra={extra}")
    return pruned, assigned


def score_plan(
    sensitivity: dict[str, Any], plan_path: Path,
    logical_bytes_override: int | None = None,
) -> dict[str, Any]:
    plan = load(plan_path)
    model = sensitivity["model"]
    layers = [int(layer) for layer in model["moe_layers"]]
    expert_count = int(model["expert_count"])
    qtypes = set(sensitivity["measurement"]["qtypes"])
    pruned, assigned = expand_plan(plan, layers, expert_count, qtypes)
    rows = {
        (int(row["layer"]), int(row["expert"])): row
        for row in sensitivity["scores"]
    }
    expected = {(layer, expert) for layer in layers for expert in range(expert_count)}
    if rows.keys() != expected:
        raise ValueError("sensitivity coverage mismatch")

    total = 0.0
    prune_damage = 0.0
    retained_damage = 0.0
    projection_damage = {projection: 0.0 for projection in PROJECTIONS}
    qtype_cells = {qtype: 0 for qtype in sorted(qtypes)}
    cells: list[dict[str, Any]] = []
    for key in sorted(rows):
        row = rows[key]
        scale = float(row["sample_scale"])
        if not math.isfinite(scale) or scale < 0:
            raise ValueError(f"invalid sample scale for {key}")
        if key in pruned:
            baselines = [
                float(row["quantization"][qtype]["joint_output_error"]["baseline_energy"])
                for qtype in qtypes
            ]
            if not all(math.isclose(value, baselines[0], rel_tol=1e-12, abs_tol=1e-12)
                       for value in baselines[1:]):
                raise ValueError(f"qtypes disagree on prune baseline for {key}")
            damage = baselines[0] * scale
            prune_damage += damage
            total += damage
            cells.append({"layer": key[0], "expert": key[1], "state": "PRUNED",
                          "damage": damage})
            continue
        for projection in PROJECTIONS:
            qtype = assigned[(*key, projection)]
            damage = float(
                row["quantization"][qtype]["projection_output_error"][projection][
                    "squared_error"
                ]
            ) * scale
            if not math.isfinite(damage) or damage < 0:
                raise ValueError(f"invalid damage for {(*key, projection, qtype)}")
            total += damage
            retained_damage += damage
            projection_damage[projection] += damage
            qtype_cells[qtype] += 1
            cells.append({"layer": key[0], "expert": key[1], "projection": projection,
                          "state": qtype, "damage": damage})
    cells.sort(key=lambda item: (-item["damage"], item["layer"], item["expert"],
                                item.get("projection", "")))
    declared_logical_bytes = plan.get("policy", {}).get("result_logical_bytes")
    if declared_logical_bytes is None:
        if logical_bytes_override is None:
            raise ValueError("plan lacks result_logical_bytes and no override was supplied")
        logical_bytes = logical_bytes_override
    else:
        logical_bytes = int(declared_logical_bytes)
        if logical_bytes_override is not None and logical_bytes_override != logical_bytes:
            raise ValueError("logical-bytes override differs from the plan declaration")
    if logical_bytes <= 0:
        raise ValueError("invalid logical model bytes")
    return {
        "path": str(plan_path.resolve()),
        "sha256": sha256(plan_path),
        "logical_bytes": logical_bytes,
        "retained_experts": len(expected) - len(pruned),
        "pruned_experts": len(pruned),
        "retained_projection_cells": len(assigned),
        "qtype_projection_cells": qtype_cells,
        "total_additive_damage": total,
        "prune_damage": prune_damage,
        "retained_quant_damage": retained_damage,
        "projection_quant_damage": projection_damage,
        "top_damage_cells": cells[:20],
    }


def build_output(
    sensitivity_path: Path,
    specs: list[tuple[str, Path]],
    logical_bytes_overrides: dict[str, int] | None = None,
) -> dict[str, Any]:
    sensitivity = load(sensitivity_path)
    if sensitivity.get("format") != SENSITIVITY_FORMAT:
        raise ValueError("unsupported sensitivity format")
    if sensitivity.get("calibration", {}).get("public_eval_data_used_for_selection") is not False:
        raise ValueError("sensitivity does not attest private-only selection")
    if len(specs) < 2 or len({name for name, _ in specs}) != len(specs):
        raise ValueError("at least two uniquely named plans are required")
    logical_bytes_overrides = logical_bytes_overrides or {}
    plan_names = {name for name, _ in specs}
    if not set(logical_bytes_overrides) <= plan_names:
        raise ValueError("logical-bytes override names are absent from the plan set")
    plans = {
        name: score_plan(sensitivity, path, logical_bytes_overrides.get(name))
        for name, path in specs
    }
    pairwise = {}
    names = [name for name, _ in specs]
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            a = plans[left]["total_additive_damage"]
            b = plans[right]["total_additive_damage"]
            pairwise[f"{left}__{right}"] = {
                "right_minus_left_damage": b - a,
                "right_minus_left_fraction": (b - a) / a if a else None,
                "right_minus_left_bytes": plans[right]["logical_bytes"]
                - plans[left]["logical_bytes"],
            }
    best = min(plans, key=lambda name: (plans[name]["total_additive_damage"],
                                        plans[name]["logical_bytes"], name))
    return {
        "format": OUTPUT_FORMAT,
        "sensitivity": {"path": str(sensitivity_path.resolve()),
                        "sha256": sha256(sensitivity_path)},
        "public_eval_data_used": False,
        "plans": plans,
        "pairwise": pairwise,
        "lowest_private_damage_plan": best,
    }


def write_receipt(
    path: Path, output: Path, sensitivity: Path, specs: list[tuple[str, Path]],
    analysis_commit: str, logical_bytes_overrides: dict[str, int] | None = None,
) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", analysis_commit):
        raise ValueError("analysis commit must be a full Git SHA")
    script = Path(__file__).resolve()
    payload = {
        "format": RECEIPT_FORMAT,
        "analysis_commit": analysis_commit,
        "public_eval_data_used": False,
        "sensitivity": {"path": str(sensitivity.resolve()), "sha256": sha256(sensitivity)},
        "plans": [{"name": name, "path": str(plan.resolve()), "sha256": sha256(plan)}
                  for name, plan in specs],
        "logical_bytes_overrides": dict(sorted((logical_bytes_overrides or {}).items())),
        "script": {"path": str(script), "sha256": sha256(script)},
        "output": {"path": str(output.resolve()), "sha256": sha256(output)},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="bw24-plan-damage-") as tmp:
        root = Path(tmp)
        quantization = {
            qtype: {
                "joint_output_error": {"baseline_energy": 10.0},
                "projection_output_error": {
                    projection: {"squared_error": error}
                    for projection in PROJECTIONS
                },
            }
            for qtype, error in (("Q8_0", 0.1), ("Q2_K", 1.0))
        }
        sensitivity = {
            "format": SENSITIVITY_FORMAT,
            "calibration": {"public_eval_data_used_for_selection": False},
            "model": {"moe_layers": [1], "expert_count": 2},
            "measurement": {"qtypes": ["Q8_0", "Q2_K"]},
            "scores": [
                {"layer": 1, "expert": expert, "sample_scale": 1.0,
                 "quantization": quantization}
                for expert in range(2)
            ],
        }
        sensitivity_path = root / "sensitivity.json"
        sensitivity_path.write_text(json.dumps(sensitivity))
        base = {
            "format": PLAN_FORMAT,
            "calibration": {"public_eval_data_used_for_selection": False},
            "policy": {"result_logical_bytes": 100},
        }
        pruned = dict(base)
        pruned.update({"pruned_experts": {"1": [1]}, "assignments": [
            {"layer": 1, "experts": [0], "qtype": "Q8_0"}
        ]})
        retained = dict(base)
        retained.update({"pruned_experts": {"1": []}, "assignments": [
            {"layer": 1, "experts": [0, 1], "qtype": "Q2_K"}
        ]})
        full_bank_policy = dict(base)
        full_bank_policy.update({
            "policy": {
                "result_logical_bytes": 100,
                "fixed_prune_count": 0,
                "prune_unused": False,
            },
            "assignments": [
                {"layer": 1, "experts": [0, 1], "qtype": "Q2_K"}
            ],
        })
        full_bank_empty_map = {
            **full_bank_policy,
            "pruned_experts": {},
        }
        paths = []
        for name, payload in (
            ("pruned", pruned),
            ("retained", retained),
            ("full_bank_policy", full_bank_policy),
            ("full_bank_empty_map", full_bank_empty_map),
        ):
            path = root / f"{name}.json"
            path.write_text(json.dumps(payload))
            paths.append((name, path))
        result = build_output(sensitivity_path, paths)
        assert math.isclose(result["plans"]["pruned"]["total_additive_damage"], 10.3)
        assert math.isclose(result["plans"]["retained"]["total_additive_damage"], 6.0)
        assert math.isclose(
            result["plans"]["full_bank_policy"]["total_additive_damage"], 6.0
        )
        assert result["plans"]["full_bank_policy"]["pruned_experts"] == 0
        assert math.isclose(
            result["plans"]["full_bank_empty_map"]["total_additive_damage"], 6.0
        )
        assert result["plans"]["full_bank_empty_map"]["pruned_experts"] == 0
        assert result["lowest_private_damage_plan"] == "full_bank_empty_map"
        legacy_full_bank = {
            **full_bank_policy,
            "policy": {
                "fixed_prune_count": 0,
                "prune_unused": False,
            },
        }
        legacy_path = root / "legacy-full-bank.json"
        legacy_path.write_text(json.dumps(legacy_full_bank))
        legacy_result = build_output(
            sensitivity_path,
            [("retained", paths[1][1]), ("legacy", legacy_path)],
            {"legacy": 101},
        )
        assert legacy_result["plans"]["legacy"]["logical_bytes"] == 101
        assert math.isclose(
            legacy_result["plans"]["legacy"]["total_additive_damage"], 6.0
        )
        try:
            score_plan(sensitivity, legacy_path)
        except ValueError as error:
            assert "no override" in str(error)
        else:
            raise AssertionError("accepted legacy plan without a logical-bytes override")
        try:
            score_plan(sensitivity, paths[1][1], 101)
        except ValueError as error:
            assert "differs" in str(error)
        else:
            raise AssertionError("accepted a conflicting logical-bytes override")
        invalid_path = root / "invalid-omitted-prune-map.json"
        invalid_path.write_text(json.dumps({
            **base,
            "assignments": [
                {"layer": 1, "experts": [0, 1], "qtype": "Q2_K"}
            ],
        }))
        try:
            score_plan(sensitivity, invalid_path)
        except ValueError as error:
            assert "omits or empties pruned_experts" in str(error)
        else:
            raise AssertionError("accepted omitted prune map without no-prune policy")
        output = root / "output.json"
        output.write_text(json.dumps(result))
        receipt = root / "receipt.json"
        write_receipt(receipt, output, sensitivity_path, paths, "a" * 40)
        assert load(receipt)["output"]["sha256"] == sha256(output)


def main() -> None:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        print("Hy3 quant plan damage self-test: PASS")
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensitivity", type=Path, required=True)
    parser.add_argument("--plan", action="append", type=parse_plan_spec, required=True)
    parser.add_argument(
        "--logical-bytes", action="append", type=parse_logical_bytes_spec, default=[]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--analysis-commit", required=True)
    args = parser.parse_args()
    for path in (args.output, args.receipt):
        if path.exists():
            raise SystemExit(f"refusing existing output {path}")
    logical_bytes_overrides = dict(args.logical_bytes)
    if len(logical_bytes_overrides) != len(args.logical_bytes):
        raise SystemExit("logical-bytes override names must be unique")
    result = build_output(args.sensitivity, args.plan, logical_bytes_overrides)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_receipt(
        args.receipt, args.output, args.sensitivity, args.plan,
        args.analysis_commit, logical_bytes_overrides,
    )
    print(f"wrote {args.output} sha256={sha256(args.output)} best={result['lowest_private_damage_plan']}")


if __name__ == "__main__":
    main()
