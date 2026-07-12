#!/usr/bin/env python3
"""Select whole experts under an exact logical-byte ceiling.

The base artifact manifest is authoritative for per-projection bytes and qtypes. A separate frozen
private score file supplies one non-negative retention score per routed expert. The resulting v2
tier plan preserves the base qtype of every retained projection and masks pruned expert ids without
renumbering them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array


PLAN_FORMAT = "bw24-expert-tier-plan-v2"
SCORE_FORMAT = "bw24-expert-retention-scores-v1"
TENSOR_RE = re.compile(
    r"^blk\.(?P<layer>\d+)\.ffn_(?P<projection>gate|up|down)_exps\."
    r"(?P<expert>\d+)\.weight$"
)
PROJECTIONS = ("gate", "up", "down")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_inventory(
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[tuple[int, int, str], dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text())
    tensors = manifest.get("tensors")
    if not isinstance(tensors, dict) or not tensors:
        raise ValueError("base manifest must contain a non-empty tensors object")
    inventory: dict[tuple[int, int, str], dict[str, Any]] = {}
    for name, spec in tensors.items():
        match = TENSOR_RE.match(name)
        if match is None:
            raise ValueError(f"unsupported expert tensor name {name!r}")
        key = (
            int(match.group("layer")),
            int(match.group("expert")),
            match.group("projection"),
        )
        if key in inventory:
            raise ValueError(f"duplicate expert tensor {key}")
        qtype = spec.get("qtype")
        size = int(spec.get("bytes", 0))
        if not isinstance(qtype, str) or not qtype or size <= 0:
            raise ValueError(f"tensor {name!r} has invalid qtype/bytes")
        inventory[key] = {"qtype": qtype, "bytes": size}

    manifest_bytes = int(manifest.get("artifact_bytes", 0))
    inventory_bytes = sum(spec["bytes"] for spec in inventory.values())
    if inventory_bytes != manifest_bytes:
        raise ValueError(
            f"manifest artifact_bytes={manifest_bytes} but tensor inventory sums to "
            f"{inventory_bytes}"
        )
    return manifest, inventory


def load_scores(
    path: Path, expected: set[tuple[int, int]],
) -> tuple[dict[str, Any], dict[tuple[int, int], float]]:
    payload = json.loads(path.read_text())
    if payload.get("format") != SCORE_FORMAT:
        raise ValueError(f"score format must be {SCORE_FORMAT!r}")
    calibration = payload.get("calibration")
    if not isinstance(calibration, dict) or calibration.get(
        "public_eval_data_used_for_selection"
    ) is not False:
        raise ValueError("score file must explicitly attest that public eval data was not used")
    rows = payload.get("scores")
    if not isinstance(rows, list):
        raise ValueError("scores must be a list")
    scores: dict[tuple[int, int], float] = {}
    for index, row in enumerate(rows):
        key = (int(row["layer"]), int(row["expert"]))
        score = float(row["retain_score"])
        if key in scores:
            raise ValueError(f"duplicate score for {key}")
        if not math.isfinite(score) or score < 0:
            raise ValueError(f"score row {index} is negative or non-finite")
        scores[key] = score
    missing, extra = expected - scores.keys(), scores.keys() - expected
    if missing or extra:
        raise ValueError(
            f"score coverage mismatch: missing={len(missing)} extra={len(extra)} "
            f"sample_missing={sorted(missing)[:5]}"
        )
    return payload, scores


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    if args.target_logical_bytes <= 0 or args.base_logical_bytes <= 0:
        raise ValueError("logical byte counts must be positive")
    if args.target_logical_bytes >= args.base_logical_bytes:
        raise ValueError("target logical bytes must be below the base logical bytes")
    if args.min_survivors_per_layer < args.top_k:
        raise ValueError("minimum survivors per layer must be at least top-k")

    manifest, inventory = load_inventory(args.base_manifest)
    experts = sorted({(layer, expert) for layer, expert, _ in inventory})
    layers = sorted({layer for layer, _ in experts})
    expert_ids = sorted({expert for _, expert in experts})
    if not layers or expert_ids != list(range(max(expert_ids) + 1)):
        raise ValueError("manifest expert ids must be contiguous from zero")
    expert_count = len(expert_ids)
    expected_tensors = {
        (layer, expert, projection)
        for layer in layers for expert in expert_ids for projection in PROJECTIONS
    }
    missing_tensors, extra_tensors = expected_tensors - inventory.keys(), inventory.keys() - expected_tensors
    if missing_tensors or extra_tensors:
        raise ValueError(
            f"tensor coverage mismatch: missing={len(missing_tensors)} extra={len(extra_tensors)}"
        )
    if args.min_survivors_per_layer > expert_count:
        raise ValueError("minimum survivors exceeds expert count")

    score_payload, scores = load_scores(args.scores, set(experts))
    score_model = score_payload.get("model")
    if not isinstance(score_model, dict):
        raise ValueError("score file model metadata is required")
    if int(score_model.get("expert_count", -1)) != expert_count:
        raise ValueError("score file expert_count does not match the manifest")
    if [int(layer) for layer in score_model.get("moe_layers", [])] != layers:
        raise ValueError("score file MoE layers do not match the manifest")
    expert_bytes = np.asarray([
        sum(inventory[(layer, expert, projection)]["bytes"] for projection in PROJECTIONS)
        for layer, expert in experts
    ], dtype=np.float64)
    retention = np.asarray([scores[key] for key in experts], dtype=np.float64)
    artifact_bytes = int(manifest["artifact_bytes"])
    fixed_bytes = args.base_logical_bytes - artifact_bytes
    if fixed_bytes < 0:
        raise ValueError("base logical bytes are smaller than artifact bytes")
    expert_budget = args.target_logical_bytes - fixed_bytes
    if expert_budget <= 0:
        raise ValueError("target leaves no bytes for routed experts")

    rows: list[int] = [0] * len(experts)
    cols: list[int] = list(range(len(experts)))
    data: list[float] = expert_bytes.tolist()
    lower = [-np.inf]
    upper = [float(expert_budget)]
    by_layer: dict[int, list[int]] = defaultdict(list)
    for index, (layer, _) in enumerate(experts):
        by_layer[layer].append(index)
    for row_index, layer in enumerate(layers, 1):
        for index in by_layer[layer]:
            rows.append(row_index)
            cols.append(index)
            data.append(1.0)
        lower.append(float(args.min_survivors_per_layer))
        upper.append(np.inf)
    matrix = coo_array(
        (np.asarray(data), (np.asarray(rows), np.asarray(cols))),
        shape=(1 + len(layers), len(experts)),
    ).tocsr()

    result = milp(
        c=-retention,
        integrality=np.ones(len(experts), dtype=np.uint8),
        bounds=Bounds(np.zeros(len(experts)), np.ones(len(experts))),
        constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
        options={"mip_rel_gap": 0.0, "time_limit": float(args.time_limit_seconds)},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"exact-byte selection failed: status={result.status} {result.message}")
    retained = {experts[index] for index, value in enumerate(result.x) if value >= 0.5}
    pruned = set(experts) - retained
    retained_artifact_bytes = int(sum(
        sum(inventory[(layer, expert, projection)]["bytes"] for projection in PROJECTIONS)
        for layer, expert in retained
    ))
    logical_bytes = fixed_bytes + retained_artifact_bytes
    if logical_bytes > args.target_logical_bytes:
        raise AssertionError("solver output violates logical byte ceiling")
    for layer in layers:
        if sum((layer, expert) in retained for expert in expert_ids) < args.min_survivors_per_layer:
            raise AssertionError(f"solver output violates layer {layer} survivor floor")

    assignments: list[dict[str, Any]] = []
    for layer in layers:
        for projection in PROJECTIONS:
            by_qtype: dict[str, list[int]] = defaultdict(list)
            for expert in expert_ids:
                if (layer, expert) in retained:
                    by_qtype[inventory[(layer, expert, projection)]["qtype"]].append(expert)
            for qtype in sorted(by_qtype):
                assignments.append({
                    "layer": layer,
                    "experts": by_qtype[qtype],
                    "projections": [projection],
                    "qtype": qtype,
                })

    pruned_experts = {
        str(layer): [expert for expert in expert_ids if (layer, expert) in pruned]
        for layer in layers
    }
    layer_summary: dict[str, Any] = {}
    for layer in layers:
        layer_retained = [(layer, expert) for expert in expert_ids if (layer, expert) in retained]
        qtype_counts: dict[str, int] = defaultdict(int)
        for key in layer_retained:
            qtypes = {inventory[(*key, projection)]["qtype"] for projection in PROJECTIONS}
            qtype_counts[next(iter(qtypes)) if len(qtypes) == 1 else "MIXED"] += 1
        layer_summary[str(layer)] = {
            "retained": len(layer_retained),
            "pruned": expert_count - len(layer_retained),
            "retained_score": sum(scores[key] for key in layer_retained),
            "qtype_expert_counts": dict(sorted(qtype_counts.items())),
        }

    plan = {
        "format": PLAN_FORMAT,
        "recipe": "exact-byte-score-prune",
        "description": "Whole-expert pruning maximizing frozen private retention score under an exact byte ceiling",
        "model": {
            "expert_count": expert_count,
            "original_expert_count": expert_count,
            "expert_used_count": args.top_k,
            "moe_layers": layers,
        },
        "policy": {
            "target_logical_bytes": args.target_logical_bytes,
            "base_logical_bytes": args.base_logical_bytes,
            "fixed_non_expert_bytes": fixed_bytes,
            "expert_byte_budget": expert_budget,
            "retained_artifact_bytes": retained_artifact_bytes,
            "result_logical_bytes": logical_bytes,
            "headroom_bytes": args.target_logical_bytes - logical_bytes,
            "min_survivors_per_layer": args.min_survivors_per_layer,
            "rank_metric": score_payload.get("rank_metric", "frozen_composite_retention_score"),
            "solver": "scipy.optimize.milp/HiGHS",
            "solver_status": int(result.status),
            "solver_message": str(result.message),
            "mip_gap": float(getattr(result, "mip_gap", 0.0)),
            "tie_break": "manifest order; score producer must provide deterministic composite scores",
        },
        "calibration": {
            "retention_scores": {
                "path": str(args.scores.resolve()),
                "sha256": sha256(args.scores),
                "canonical_sha256": canonical_sha256(score_payload),
            },
            "public_eval_data_used_for_selection": False,
        },
        "base_artifact": {
            "manifest_path": str(args.base_manifest.resolve()),
            "manifest_sha256": sha256(args.base_manifest),
            "manifest_canonical_sha256": canonical_sha256(manifest),
            "artifact_bytes": artifact_bytes,
        },
        "selection": {
            "retained_experts": len(retained),
            "pruned_experts": len(pruned),
            "retained_score": float(sum(scores[key] for key in retained)),
            "pruned_score": float(sum(scores[key] for key in pruned)),
        },
        "pruned_experts": pruned_experts,
        "assignments": assignments,
        "layer_summary": layer_summary,
    }
    return plan


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="bw24-exact-budget-plan-") as tmp:
        root = Path(tmp)
        manifest_path, scores_path, plan_path = (
            root / "manifest.json", root / "scores.json", root / "plan.json"
        )
        tensors: dict[str, Any] = {}
        for layer in (1, 2):
            for expert in range(4):
                for projection in PROJECTIONS:
                    tensors[f"blk.{layer}.ffn_{projection}_exps.{expert}.weight"] = {
                        "bytes": 10 if expert < 2 else 20,
                        "qtype": "Q2_K" if expert < 2 else "NVFP4",
                    }
        manifest = {"artifact_bytes": sum(v["bytes"] for v in tensors.values()), "tensors": tensors}
        manifest_path.write_text(json.dumps(manifest))
        scores = {
            "format": SCORE_FORMAT,
            "rank_metric": "self_test",
            "model": {"expert_count": 4, "moe_layers": [1, 2]},
            "calibration": {"public_eval_data_used_for_selection": False},
            "scores": [
                {"layer": layer, "expert": expert, "retain_score": 10 - expert - layer / 10}
                for layer in (1, 2) for expert in range(4)
            ],
        }
        scores_path.write_text(json.dumps(scores))
        args = argparse.Namespace(
            base_manifest=manifest_path,
            scores=scores_path,
            base_logical_bytes=400,
            target_logical_bytes=220,
            min_survivors_per_layer=2,
            top_k=1,
            time_limit_seconds=30,
        )
        plan = build_plan(args)
        plan_path.write_text(json.dumps(plan))
        assert plan["policy"]["result_logical_bytes"] <= 220
        assert all(row["retained"] >= 2 for row in plan["layer_summary"].values())
        assert plan["selection"]["retained_experts"] == 5
        assert plan["selection"]["pruned_experts"] == 3
        assert math.isclose(plan["selection"]["retained_score"], 45.3)
        assert math.isclose(plan["selection"]["pruned_score"], 21.5)
        from prepare_mixed_expert_repack import load_assignments

        _, expanded, pruned = load_assignments(plan_path)
        assert len(expanded) == 5 * 3
        assert sum(len(ids) for ids in pruned.values()) == 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--base-logical-bytes", type=int, required=True)
    parser.add_argument("--target-logical-bytes", type=int, default=100_000_000_000)
    parser.add_argument("--min-survivors-per-layer", type=int, default=96)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--time-limit-seconds", type=int, default=600)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        print("exact budget prune plan self-test: PASS")
        return
    args = parse_args()
    plan = build_plan(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(
        f"wrote {args.out} sha256={sha256(args.out)} "
        f"logical_bytes={plan['policy']['result_logical_bytes']} "
        f"pruned={plan['selection']['pruned_experts']}"
    )


if __name__ == "__main__":
    main()
