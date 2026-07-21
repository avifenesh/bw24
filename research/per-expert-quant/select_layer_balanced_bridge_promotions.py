#!/usr/bin/env python3
"""Apply frozen directional gates to the 120/137 GB layer-balanced bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


FRONTIER_FORMAT = "bw24-cross-run-expanded-capability-frontier-v1"
LOCK_FORMAT = "bw24-layer-balanced-bridge-promotion-gates-v1"
OUTPUT_FORMAT = "bw24-layer-balanced-bridge-directional-promotion-v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(frontier: dict[str, Any], lock: dict[str, Any]) -> dict[str, Any]:
    if frontier.get("format") != FRONTIER_FORMAT or lock.get("format") != LOCK_FORMAT:
        raise ValueError("wrong frontier or bridge gate-lock format")
    if frontier["panel_lock"]["sha256"] != lock["panel_lock_sha256"]:
        raise ValueError("panel hashes differ")
    candidates = lock["candidate_anchors"]
    required = set(lock["fixed_practical_reference_arms"]) | set(candidates) | set(candidates.values())
    missing = required - frontier["arms"].keys()
    if missing:
        raise ValueError(f"frontier is missing {sorted(missing)}")
    decisions: dict[str, Any] = {}
    passed = []
    for arm, anchor_name in candidates.items():
        candidate = frontier["arms"][arm]
        anchor = frontier["arms"][anchor_name]
        deficits = {
            task: float(anchor["tasks"][task]["rate"]) - float(candidate["tasks"][task]["rate"])
            for task in sorted(anchor["tasks"])
        }
        checks = {
            "logical_byte_ceiling": int(candidate["logical_model_bytes"])
                <= int(lock["hard_logical_byte_ceiling_by_arm"][arm]),
            "global_point_estimate_pareto": arm in frontier["point_estimate_pareto"],
            "question_weighted_alignment": float(anchor["question_weighted"])
                - float(candidate["question_weighted"])
                <= float(lock["max_question_weighted_deficit_vs_anchor"]),
            "no_task_collapse": max(deficits.values())
                <= float(lock["max_single_task_rate_deficit_vs_anchor"]),
        }
        decisions[arm] = {
            "anchor_arm": anchor_name,
            "passed": all(checks.values()),
            "checks": checks,
            "task_rate_deficits_vs_anchor": deficits,
            "pairwise_vs_anchor": frontier["pairwise_comparisons"][arm][anchor_name],
        }
        if decisions[arm]["passed"]:
            passed.append(arm)
    passed.sort(key=lambda arm: (
        int(frontier["arms"][arm]["logical_model_bytes"]),
        -float(frontier["arms"][arm]["domain_macro"]), arm,
    ))
    selected = passed[: int(lock["max_practical_candidates"])]
    return {
        "format": OUTPUT_FORMAT,
        "directional_decisions": decisions,
        "qualified_candidates": passed,
        "selected_practical_candidates": selected,
        "practical_arms": [*lock["fixed_practical_reference_arms"], *selected],
        "trusted_full_max_arms": int(lock["trusted_full_max_arms"]),
    }


def self_test() -> None:
    tasks = {"a": {"rate": 0.6}, "b": {"rate": 0.5}}
    arms = {
        "plain": {"logical_model_bytes": 180, "domain_macro": .8, "question_weighted": .8, "tasks": tasks},
        "traffic": {"logical_model_bytes": 137, "domain_macro": .7, "question_weighted": .7, "tasks": tasks},
        "small": {"logical_model_bytes": 100, "domain_macro": .6, "question_weighted": .6, "tasks": tasks},
        "bridge120": {"logical_model_bytes": 120, "domain_macro": .65, "question_weighted": .6, "tasks": tasks},
        "bridge137": {"logical_model_bytes": 137, "domain_macro": .69, "question_weighted": .69, "tasks": tasks},
    }
    pairwise = {arm: {anchor: {"paired_wins": 1, "paired_losses": 0}
                      for anchor in arms} for arm in arms}
    frontier = {"format": FRONTIER_FORMAT, "panel_lock": {"sha256": "panel"},
                "arms": arms, "point_estimate_pareto": ["small", "bridge120", "traffic", "plain"],
                "pairwise_comparisons": pairwise}
    lock = {"format": LOCK_FORMAT, "panel_lock_sha256": "panel",
            "fixed_practical_reference_arms": ["plain", "traffic", "small"],
            "candidate_anchors": {"bridge120": "small", "bridge137": "traffic"},
            "hard_logical_byte_ceiling_by_arm": {"bridge120": 120, "bridge137": 137},
            "max_question_weighted_deficit_vs_anchor": .01,
            "max_single_task_rate_deficit_vs_anchor": .2,
            "max_practical_candidates": 2, "trusted_full_max_arms": 4}
    result = select(frontier, lock)
    assert result["selected_practical_candidates"] == ["bridge120"]
    assert not result["directional_decisions"]["bridge137"]["checks"]["global_point_estimate_pareto"]


def main() -> None:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        print("layer-balanced bridge promotion selector self-test: PASS")
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontier", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = select(json.loads(args.frontier.read_text()), json.loads(args.lock.read_text()))
    result["frontier"] = {"path": str(args.frontier.resolve()), "sha256": sha256(args.frontier)}
    result["gate_lock"] = {"path": str(args.lock.resolve()), "sha256": sha256(args.lock)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {args.output} practical={result['practical_arms']}")


if __name__ == "__main__":
    main()
