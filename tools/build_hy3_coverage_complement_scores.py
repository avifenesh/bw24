#!/usr/bin/env python3
"""Rank private-calibration experts that can complement a frozen Layer100 base.

The score deliberately contains no public capability result.  It asks which experts pruned by
the base carry concentrated tail traffic, distinct outputs, and high routed-output magnitude.
It emits the standard retention-score schema so the existing byte allocator can consume it.
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


FORMAT = "bw24-expert-retention-scores-v1"
PLAN_FORMAT = "bw24-expert-tier-plan-v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scaled(values: np.ndarray) -> np.ndarray:
    values = np.maximum(values.astype(np.float64), 0.0)
    maximum = float(values.max(initial=0.0))
    return np.zeros_like(values) if maximum == 0 else np.log1p(values) / math.log1p(maximum)


def base_pruned(payload: dict[str, Any]) -> set[tuple[int, int]]:
    if payload.get("format") != PLAN_FORMAT:
        raise ValueError("unsupported base plan format")
    return {
        (int(layer), int(expert))
        for layer, experts in payload.get("pruned_experts", {}).items()
        for expert in experts
    }


def build_scores(source: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    if source.get("format") != FORMAT:
        raise ValueError("unsupported source retention score format")
    calibration = source.get("calibration", {})
    if calibration.get("public_eval_data_used_for_selection") is not False:
        raise ValueError("coverage-complement construction requires private calibration")
    rows = source.get("scores")
    if not isinstance(rows, list) or not rows:
        raise ValueError("source retention scores are empty")
    pruned = base_pruned(base)
    keys = {(int(row["layer"]), int(row["expert"])) for row in rows}
    if not pruned <= keys:
        raise ValueError("base plan and retention score coverage disagree")

    by_layer: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_layer.setdefault(int(row["layer"]), []).append(row)
    output_rows: list[dict[str, Any]] = []
    for layer, layer_rows in sorted(by_layer.items()):
        layer_rows.sort(key=lambda row: int(row["expert"]))
        candidates = [row for row in layer_rows if (layer, int(row["expert"])) in pruned]
        if not candidates:
            output_rows.extend(dict(row, base_status="retained", protected=False) for row in layer_rows)
            continue
        strata = sorted({name for row in candidates for name in row["stratum_router_mass"]})
        mass = np.asarray(
            [[float(row["stratum_router_mass"].get(name, 0.0)) for name in strata]
             for row in candidates],
            dtype=np.float64,
        )
        relative = np.divide(
            mass,
            mass.max(axis=0, keepdims=True),
            out=np.zeros_like(mass),
            where=mass.max(axis=0, keepdims=True) > 0,
        )
        tail_width = min(2, len(strata))
        tail = np.sort(relative, axis=1)[:, -tail_width:].mean(axis=1)
        total = mass.sum(axis=1)
        specialization = np.divide(
            mass.max(axis=1, initial=0.0),
            total,
            out=np.zeros_like(total),
            where=total > 0,
        )
        reap = scaled(np.asarray([float(row.get("reap", 0.0)) for row in candidates]))
        uniqueness = scaled(
            np.asarray([float(row.get("diversity_uniqueness", 0.0)) for row in candidates])
        )
        raw = 0.45 * tail + 0.25 * reap + 0.20 * uniqueness + 0.10 * specialization
        composite = raw * (0.5 + 0.5 * specialization)
        candidate_index = {int(row["expert"]): index for index, row in enumerate(candidates)}
        for row in layer_rows:
            expert = int(row["expert"])
            result = dict(row)
            result["protected"] = False
            if expert in candidate_index:
                index = candidate_index[expert]
                result["retain_score"] = float(composite[index] + (len(layer_rows) - expert) * 1e-12)
                result["base_status"] = "pruned-restore-candidate"
                result["coverage_complement"] = {
                    "tail_top2_mean": float(tail[index]),
                    "specialization": float(specialization[index]),
                    "reap_scaled_among_base_pruned": float(reap[index]),
                    "uniqueness_scaled_among_base_pruned": float(uniqueness[index]),
                    "raw_score": float(raw[index]),
                }
            else:
                # Base constraints, not this score, preserve retained experts.  Keeping their
                # source score makes diagnostics readable without forcing them into the restore rank.
                result["base_status"] = "retained"
            output_rows.append(result)

    result = dict(source)
    result["rank_metric"] = "layer100_private_coverage_complement_v1"
    result["policy"] = {
        "formula": "(0.45*tail_top2_mean + 0.25*reap + 0.20*uniqueness + 0.10*specialization) * (0.5 + 0.5*specialization)",
        "normalization": "per-layer among Layer100-pruned experts only",
        "base_retained_selection": "frozen by allocator; not reranked",
        "public_capability_results_used": False,
    }
    result["base_plan"] = {
        "format": PLAN_FORMAT,
        "pruned_experts": len(pruned),
    }
    result["scores"] = output_rows
    return result


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="bw24-coverage-complement-") as tmp:
        root = Path(tmp)
        source = {
            "format": FORMAT,
            "calibration": {"public_eval_data_used_for_selection": False},
            "scores": [
                {
                    "layer": 1,
                    "expert": expert,
                    "retain_score": 1.0,
                    "protected": expert == 0,
                    "reap": reap,
                    "diversity_uniqueness": uniqueness,
                    "stratum_router_mass": mass,
                }
                for expert, reap, uniqueness, mass in (
                    (0, 1.0, 0.1, {"code": 10.0, "math": 10.0}),
                    (1, 4.0, 0.8, {"code": 9.0, "math": 0.1}),
                    (2, 1.0, 0.2, {"code": 2.0, "math": 2.0}),
                )
            ],
        }
        base = {
            "format": PLAN_FORMAT,
            "pruned_experts": {"1": [1, 2]},
            "assignments": [],
        }
        result = build_scores(source, base)
        by_expert = {row["expert"]: row for row in result["scores"]}
        assert by_expert[0]["base_status"] == "retained"
        assert by_expert[1]["retain_score"] > by_expert[2]["retain_score"]
        assert by_expert[1]["coverage_complement"]["specialization"] > 0.9
        source["calibration"]["public_eval_data_used_for_selection"] = True
        try:
            build_scores(source, base)
        except ValueError as error:
            assert "private calibration" in str(error)
        else:
            raise AssertionError("public-derived source was accepted")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retention-scores", type=Path, required=True)
    parser.add_argument("--base-plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        print("Hy3 coverage complement score self-test: PASS")
        return
    args = parse_args()
    source = json.loads(args.retention_scores.read_text())
    base = json.loads(args.base_plan.read_text())
    result = build_scores(source, base)
    result["base_plan"] = {
        "path": str(args.base_plan.resolve()),
        "sha256": sha256(args.base_plan),
        "format": PLAN_FORMAT,
        "pruned_experts": len(base_pruned(base)),
    }
    result["calibration"] = dict(result["calibration"])
    result["calibration"]["source_retention_scores"] = {
        "path": str(args.retention_scores.resolve()),
        "sha256": sha256(args.retention_scores),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out} sha256={sha256(args.out)}")


if __name__ == "__main__":
    main()
