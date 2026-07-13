#!/usr/bin/env python3
"""Compare smart-budget plans at layer/expert/projection resolution."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


PLAN_FORMAT = "bw24-expert-tier-plan-v2"
OUTPUT_FORMAT = "bw24-hy3-smart-allocation-comparison-v1"
PROJECTIONS = ("gate", "up", "down")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_plan(path: Path) -> tuple[dict[str, Any], dict[tuple[int, int, str], str]]:
    plan = json.loads(path.read_text())
    if plan.get("format") != PLAN_FORMAT:
        raise ValueError(f"{path}: unsupported plan format")
    if plan.get("calibration", {}).get("public_eval_data_used_for_selection") is not False:
        raise ValueError(f"{path}: selection evidence is not private")
    layers = [int(layer) for layer in plan["model"]["moe_layers"]]
    expert_count = int(plan["model"]["expert_count"])
    pruned = {
        (int(layer), int(expert))
        for layer, experts in plan["pruned_experts"].items()
        for expert in experts
    }
    cells: dict[tuple[int, int, str], str] = {}
    for layer in layers:
        for expert in range(expert_count):
            if (layer, expert) in pruned:
                for projection in PROJECTIONS:
                    cells[(layer, expert, projection)] = "PRUNED"
    for assignment in plan["assignments"]:
        layer = int(assignment["layer"])
        qtype = str(assignment["qtype"])
        projections = assignment.get("projections", PROJECTIONS)
        for expert in assignment["experts"]:
            for projection_value in projections:
                projection = str(projection_value)
                if projection not in PROJECTIONS:
                    raise ValueError(f"{path}: unsupported projection {projection!r}")
                key = (layer, int(expert), projection)
                if key in cells:
                    raise ValueError(f"{path}: duplicate allocation for {key}")
                cells[key] = qtype
    expected = len(layers) * expert_count * len(PROJECTIONS)
    if len(cells) != expected:
        raise ValueError(f"{path}: allocation cells={len(cells)}, expected={expected}")
    return plan, cells


def allocation_hash(cells: dict[tuple[int, int, str], str]) -> str:
    canonical = [
        [layer, expert, projection, cells[(layer, expert, projection)]]
        for layer, expert, projection in sorted(cells)
    ]
    payload = json.dumps(canonical, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def summarize(paths: list[Path]) -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("at least two plans are required")
    loaded = {path.stem: (*load_plan(path), path) for path in paths}
    if len(loaded) != len(paths):
        raise ValueError("plan stems must be unique")
    reference_keys = None
    plans: dict[str, Any] = {}
    hashes: dict[str, list[str]] = {}
    for arm, (plan, cells, path) in sorted(loaded.items()):
        keys = set(cells)
        if reference_keys is None:
            reference_keys = keys
        elif keys != reference_keys:
            raise ValueError("plans do not cover the same model cells")
        digest = allocation_hash(cells)
        hashes.setdefault(digest, []).append(arm)
        expert_keys = {key[:2] for key in cells}
        pruned_keys = {
            key for key in expert_keys
            if all(cells[(*key, projection)] == "PRUNED" for projection in PROJECTIONS)
        }
        derived_qtype_counts = Counter(
            qtype for qtype in cells.values() if qtype != "PRUNED"
        )
        retained_keys = sorted(expert_keys - pruned_keys)
        projection_qtype_counts = {
            projection: dict(sorted(Counter(
                cells[(*key, projection)] for key in retained_keys
            ).items()))
            for projection in PROJECTIONS
        }
        expert_projection_combinations = Counter(
            tuple(cells[(*key, projection)] for projection in PROJECTIONS)
            for key in retained_keys
        )
        uniform_precision_experts = sum(
            len({cells[(*key, projection)] for projection in PROJECTIONS}) == 1
            for key in retained_keys
        )
        logical_model_bytes = plan["policy"].get("result_logical_bytes")
        headroom_bytes = plan["policy"].get("headroom_bytes")
        layer_counts: dict[str, dict[str, int]] = {}
        for (layer, _expert, _projection), qtype in cells.items():
            layer_counts.setdefault(str(layer), {})[qtype] = (
                layer_counts.setdefault(str(layer), {}).get(qtype, 0) + 1
            )
        plans[arm] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "allocation_sha256": digest,
            "logical_model_bytes": (
                int(logical_model_bytes) if logical_model_bytes is not None else None
            ),
            "headroom_bytes": int(headroom_bytes) if headroom_bytes is not None else None,
            "retained_experts": int(
                plan.get("selection", {}).get(
                    "retained_experts", len(expert_keys) - len(pruned_keys)
                )
            ),
            "pruned_experts": int(
                plan.get("selection", {}).get("pruned_experts", len(pruned_keys))
            ),
            "qtype_projection_counts": plan["policy"].get(
                "qtype_projection_counts", dict(sorted(derived_qtype_counts.items()))
            ),
            "projection_qtype_counts": projection_qtype_counts,
            "expert_projection_combinations": [
                {
                    "qtypes": dict(zip(PROJECTIONS, combination)),
                    "experts": count,
                }
                for combination, count in sorted(
                    expert_projection_combinations.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ],
            "uniform_precision_experts": uniform_precision_experts,
            "mixed_precision_experts": len(retained_keys) - uniform_precision_experts,
            "layer_cell_counts": layer_counts,
        }
    pairwise: dict[str, Any] = {}
    arms = sorted(loaded)
    for left_index, left in enumerate(arms):
        left_plan, left_cells, _ = loaded[left]
        left_pruned = {
            (int(layer), int(expert))
            for layer, experts in left_plan["pruned_experts"].items()
            for expert in experts
        }
        for right in arms[left_index + 1 :]:
            right_plan, right_cells, _ = loaded[right]
            right_pruned = {
                (int(layer), int(expert))
                for layer, experts in right_plan["pruned_experts"].items()
                for expert in experts
            }
            transitions = Counter(
                f"{left_cells[key]}->{right_cells[key]}"
                for key in sorted(left_cells)
                if left_cells[key] != right_cells[key]
            )
            changed = sum(transitions.values())
            pairwise[f"{left}__{right}"] = {
                "allocation_cells": len(left_cells),
                "changed_cells": changed,
                "identical_cells": len(left_cells) - changed,
                "retention_changed_experts": len(left_pruned ^ right_pruned),
                "shared_pruned_experts": len(left_pruned & right_pruned),
                "shared_retained_experts": len(
                    {key[:2] for key in left_cells} - (left_pruned | right_pruned)
                ),
                "transitions": dict(sorted(transitions.items())),
            }
    duplicates = [sorted(arms) for arms in hashes.values() if len(arms) > 1]
    return {
        "format": OUTPUT_FORMAT,
        "public_eval_data_used": False,
        "plans": plans,
        "pairwise": pairwise,
        "duplicate_allocations": sorted(duplicates),
    }


def self_test() -> None:
    def plan(qtype: str, prune: int | None) -> dict[str, Any]:
        pruned = [] if prune is None else [prune]
        retained = [expert for expert in range(3) if expert not in pruned]
        return {
            "format": PLAN_FORMAT,
            "model": {"moe_layers": [1], "expert_count": 3},
            "calibration": {"public_eval_data_used_for_selection": False},
            "policy": {
                "result_logical_bytes": 100,
                "headroom_bytes": 0,
                "qtype_projection_counts": {qtype: len(retained) * 3},
            },
            "selection": {"retained_experts": len(retained), "pruned_experts": len(pruned)},
            "pruned_experts": {"1": pruned},
            "assignments": [{
                "layer": 1, "experts": retained,
                "projections": list(PROJECTIONS), "qtype": qtype,
            }],
        }

    with tempfile.TemporaryDirectory(prefix="bw24-smart-allocation-") as tmp:
        root = Path(tmp)
        paths = []
        for name, payload in (
            ("a", plan("Q3_K", 0)),
            ("b", plan("Q2_K", 0)),
            ("c", plan("Q3_K", 1)),
        ):
            path = root / f"{name}.json"
            path.write_text(json.dumps(payload))
            paths.append(path)
        result = summarize(paths)
        assert result["duplicate_allocations"] == []
        assert result["plans"]["a"]["uniform_precision_experts"] == 2
        assert result["plans"]["a"]["mixed_precision_experts"] == 0
        assert result["plans"]["a"]["projection_qtype_counts"] == {
            projection: {"Q3_K": 2} for projection in PROJECTIONS
        }
        assert result["plans"]["a"]["expert_projection_combinations"] == [{
            "qtypes": {projection: "Q3_K" for projection in PROJECTIONS},
            "experts": 2,
        }]
        assert result["pairwise"]["a__b"]["changed_cells"] == 6
        assert result["pairwise"]["a__c"]["retention_changed_experts"] == 2
        duplicate = root / "duplicate.json"
        duplicate.write_text(json.dumps(plan("Q3_K", 0)))
        duplicate_result = summarize([paths[0], duplicate])
        assert duplicate_result["duplicate_allocations"] == [["a", "duplicate"]]
        legacy = root / "legacy.json"
        legacy_payload = plan("Q3_K", None)
        del legacy_payload["assignments"][0]["projections"]
        legacy_payload["policy"].pop("result_logical_bytes")
        legacy_payload["policy"].pop("headroom_bytes")
        legacy_payload["policy"].pop("qtype_projection_counts")
        legacy_payload.pop("selection")
        legacy.write_text(json.dumps(legacy_payload))
        legacy_result = summarize([paths[0], legacy])
        assert legacy_result["pairwise"]["a__legacy"]["changed_cells"] == 3
        assert legacy_result["plans"]["legacy"]["logical_model_bytes"] is None
        assert legacy_result["plans"]["legacy"]["retained_experts"] == 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("plans", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--require-distinct", action="store_true")
    return parser.parse_args()


def main() -> None:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        print("Hy3 smart allocation comparison self-test: PASS")
        return
    args = parse_args()
    result = summarize(args.plans)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out} sha256={sha256(args.out)}")
    if args.require_distinct and result["duplicate_allocations"]:
        raise SystemExit(f"duplicate candidate allocations: {result['duplicate_allocations']}")


if __name__ == "__main__":
    main()
