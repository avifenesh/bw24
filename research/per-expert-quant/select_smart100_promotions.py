#!/usr/bin/env python3
"""Apply frozen same-size alignment gates to measured smart-100GB candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any


FRONTIER_FORMAT = "bw24-cross-run-expanded-capability-frontier-v1"
LOCK_FORMAT = "bw24-smart100-promotion-gates-v1"
OUTPUT_FORMAT = "bw24-smart100-directional-promotion-v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(frontier: dict[str, Any], lock: dict[str, Any]) -> dict[str, Any]:
    if frontier.get("format") != FRONTIER_FORMAT or lock.get("format") != LOCK_FORMAT:
        raise ValueError("wrong frontier or gate-lock format")
    if frontier["panel_lock"]["sha256"] != lock["panel_lock_sha256"]:
        raise ValueError("panel hashes differ")
    required = {
        lock["quality_ceiling_arm"], lock["strong_compact_arm"],
        lock["same_size_control_arm"],
    }
    missing = required - frontier["arms"].keys()
    if missing:
        raise ValueError(f"frontier is missing {sorted(missing)}")
    available_candidates = [arm for arm in lock["candidate_arms"] if arm in frontier["arms"]]
    unavailable_candidates = [arm for arm in lock["candidate_arms"] if arm not in frontier["arms"]]
    if not available_candidates:
        raise ValueError("frontier contains no heal-eligible smart100 candidates")
    control_name = lock["same_size_control_arm"]
    control = frontier["arms"][control_name]
    decisions = {}
    passed = []
    for arm in available_candidates:
        candidate = frontier["arms"][arm]
        deficits = {
            task: float(control["tasks"][task]["rate"]) - float(candidate["tasks"][task]["rate"])
            for task in sorted(control["tasks"])
        }
        checks = {
            "logical_byte_ceiling": int(candidate["logical_model_bytes"]) <= int(lock["hard_logical_byte_ceiling"]),
            "global_point_estimate_pareto": arm in frontier["point_estimate_pareto"],
            "domain_macro_alignment": float(control["domain_macro"]) - float(candidate["domain_macro"])
                <= float(lock["max_domain_macro_deficit_vs_same_size_control"]),
            "question_weighted_alignment": float(control["question_weighted"]) - float(candidate["question_weighted"])
                <= float(lock["max_question_weighted_deficit_vs_same_size_control"]),
            "no_task_collapse": max(deficits.values())
                <= float(lock["max_single_task_rate_deficit_vs_same_size_control"]),
        }
        decisions[arm] = {
            "passed": all(checks.values()), "checks": checks,
            "task_rate_deficits_vs_same_size_control": deficits,
            "pairwise_vs_same_size_control": frontier["pairwise_comparisons"][arm][control_name],
        }
        if decisions[arm]["passed"]:
            passed.append(arm)
    passed.sort(key=lambda arm: (
        -float(frontier["arms"][arm]["domain_macro"]),
        -float(frontier["arms"][arm]["question_weighted"]),
        -int(frontier["pairwise_comparisons"][arm][control_name]["paired_wins"])
        + int(frontier["pairwise_comparisons"][arm][control_name]["paired_losses"]),
        int(frontier["arms"][arm]["logical_model_bytes"]), arm,
    ))
    selected = passed[: int(lock["max_practical_candidates"])]
    return {
        "format": OUTPUT_FORMAT,
        "same_size_control_arm": control_name,
        "directional_decisions": decisions,
        "unavailable_candidate_arms": unavailable_candidates,
        "qualified_candidates": passed,
        "selected_practical_candidates": selected,
        "practical_arms": [lock["quality_ceiling_arm"], lock["strong_compact_arm"], *selected],
        "trusted_full_max_arms": int(lock["trusted_full_max_arms"]),
    }


def self_test() -> None:
    tasks = {"a": {"rate": .6}, "b": {"rate": .5}}
    arms = {
        "plain": {"logical_model_bytes": 180, "domain_macro": .8, "question_weighted": .8, "tasks": tasks},
        "compact": {"logical_model_bytes": 137, "domain_macro": .7, "question_weighted": .7, "tasks": tasks},
        "control": {"logical_model_bytes": 100, "domain_macro": .6, "question_weighted": .6, "tasks": tasks},
        "good": {"logical_model_bytes": 99, "domain_macro": .61, "question_weighted": .6, "tasks": tasks},
        "bad": {"logical_model_bytes": 99, "domain_macro": .5, "question_weighted": .5, "tasks": tasks},
    }
    frontier = {"format": FRONTIER_FORMAT, "panel_lock": {"sha256": "panel"}, "arms": arms,
        "point_estimate_pareto": ["plain", "compact", "control", "good"],
        "pairwise_comparisons": {x: {"control": {"paired_wins": 2, "paired_losses": 1}} for x in arms}}
    lock = {"format": LOCK_FORMAT, "panel_lock_sha256": "panel", "quality_ceiling_arm": "plain",
        "strong_compact_arm": "compact", "same_size_control_arm": "control",
        "candidate_arms": ["good", "bad"], "hard_logical_byte_ceiling": 100,
        "max_domain_macro_deficit_vs_same_size_control": .02,
        "max_question_weighted_deficit_vs_same_size_control": .02,
        "max_single_task_rate_deficit_vs_same_size_control": .2,
        "max_practical_candidates": 2, "trusted_full_max_arms": 3}
    result = select(frontier, lock)
    assert result["selected_practical_candidates"] == ["good"]
    del frontier["arms"]["bad"]
    result = select(frontier, lock)
    assert result["selected_practical_candidates"] == ["good"]
    assert result["unavailable_candidate_arms"] == ["bad"]


def main() -> None:
    if sys.argv[1:] == ["--self-test"]:
        self_test(); print("smart100 promotion selector self-test: PASS"); return
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
        json.dump(result, handle, indent=2, sort_keys=True); handle.write("\n")
    print(f"wrote {args.output} practical={result['practical_arms']}")


if __name__ == "__main__":
    main()
