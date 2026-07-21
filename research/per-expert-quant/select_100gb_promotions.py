#!/usr/bin/env python3
"""Apply the frozen exact-100GB promotion gates to a strict cross-run frontier."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import tempfile
from typing import Any


LOCK_FORMAT = "bw24-100gb-promotion-gates-v1"
REPORT_FORMAT = "bw24-cross-run-expanded-capability-frontier-v1"
OUTPUT_FORMAT = "bw24-100gb-directional-promotion-v1"


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def task_deficits(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, float]:
    if set(candidate["tasks"]) != set(reference["tasks"]):
        raise ValueError("candidate and reference task sets differ")
    return {
        task: float(reference["tasks"][task]["rate"]) - float(candidate["tasks"][task]["rate"])
        for task in sorted(reference["tasks"])
    }


def directional_checks(
    arm: str, report: dict[str, Any], lock: dict[str, Any]
) -> dict[str, Any]:
    cfg = lock["directional"]
    candidate = report["arms"][arm]
    reference = report["arms"][cfg["reference_arm"]]
    deficits = task_deficits(candidate, reference)
    checks = {
        "logical_byte_ceiling": int(candidate["logical_model_bytes"])
        <= int(cfg["hard_logical_byte_ceiling"]),
        "global_point_estimate_pareto": (
            arm in report["point_estimate_pareto"]
            if cfg["require_global_point_estimate_pareto"]
            else True
        ),
        "domain_macro_alignment": float(reference["domain_macro"])
        - float(candidate["domain_macro"])
        <= float(cfg["max_domain_macro_deficit_vs_reference"]),
        "question_weighted_alignment": float(reference["question_weighted"])
        - float(candidate["question_weighted"])
        <= float(cfg["max_question_weighted_deficit_vs_reference"]),
        "no_reference_task_collapse": max(deficits.values())
        <= float(cfg["max_single_task_rate_deficit_vs_reference"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "task_rate_deficits_vs_reference": deficits,
    }


def healing_checks(
    arm: str, report: dict[str, Any], lock: dict[str, Any]
) -> dict[str, Any]:
    cfg = lock["healing"]
    if arm == cfg["router_ablation_arm"] and cfg["router_ablation_is_not_promoted"]:
        return {"passed": False, "checks": {"router_ablation_not_promoted": False}}
    if arm != cfg["joint_arm"]:
        return {"passed": True, "checks": {"healing_gate_not_applicable": True}}
    unhealed = cfg["unhealed_arm"]
    pair = report["pairwise_comparisons"][arm][unhealed]
    deficits = task_deficits(report["arms"][arm], report["arms"][unhealed])
    checks = {
        "positive_domain_macro_delta": (
            float(pair["domain_macro_delta"]) > 0
            if cfg["joint_requires_positive_domain_macro_delta"]
            else True
        ),
        "nonnegative_question_weighted_delta": (
            float(pair["question_weighted_delta"]) >= 0
            if cfg["joint_requires_nonnegative_question_weighted_delta"]
            else True
        ),
        "more_paired_wins_than_losses": (
            int(pair["paired_wins"]) > int(pair["paired_losses"])
            if cfg["joint_requires_more_paired_wins_than_losses"]
            else True
        ),
        "no_unhealed_task_collapse": max(deficits.values())
        <= float(cfg["max_single_task_rate_deficit_vs_unhealed"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "pairwise_vs_unhealed": pair,
        "task_rate_deficits_vs_unhealed": deficits,
    }


def select(report: dict[str, Any], lock: dict[str, Any]) -> dict[str, Any]:
    if report.get("format") != REPORT_FORMAT:
        raise ValueError("wrong cross-run report format")
    if lock.get("format") != LOCK_FORMAT:
        raise ValueError("wrong promotion lock format")
    if report["panel_lock"]["sha256"] != lock["panel_lock_sha256"]:
        raise ValueError("promotion lock and report panel hashes differ")
    required = {
        lock["directional"]["reference_arm"],
        lock["directional"]["quality_ceiling_arm"],
        lock["directional"]["strong_compact_arm"],
        *lock["directional"]["candidate_arms"],
    }
    missing = required - report["arms"].keys()
    if missing:
        raise ValueError(f"cross-run report is missing arms {sorted(missing)}")

    decisions = {}
    qualified = []
    for arm in lock["directional"]["candidate_arms"]:
        directional = directional_checks(arm, report, lock)
        healing = healing_checks(arm, report, lock)
        passed = directional["passed"] and healing["passed"]
        decisions[arm] = {
            "passed": passed,
            "directional": directional,
            "healing": healing,
        }
        if passed:
            qualified.append(arm)

    ranked = sorted(
        qualified,
        key=lambda arm: (
            -float(report["arms"][arm]["domain_macro"]),
            -float(report["arms"][arm]["question_weighted"]),
            int(report["arms"][arm]["logical_model_bytes"]),
            arm,
        ),
    )
    selected_100gb = ranked[: int(lock["practical"]["max_100gb_arms"])]
    practical = list(dict.fromkeys(lock["practical"]["fixed_reference_arms"] + selected_100gb))
    return {
        "format": OUTPUT_FORMAT,
        "directional_decisions": decisions,
        "qualified_100gb_arms": ranked,
        "selected_100gb_arms": selected_100gb,
        "practical_arms": practical,
        "trusted_full_arm_limit": int(lock["trusted_full"]["max_arms"]),
    }


def self_test() -> None:
    tasks = {"a": {"rate": 0.6}, "b": {"rate": 0.7}}
    arms = {
        "plain": {"logical_model_bytes": 180, "domain_macro": 0.8, "question_weighted": 0.8, "tasks": tasks},
        "compact": {"logical_model_bytes": 135, "domain_macro": 0.65, "question_weighted": 0.64, "tasks": tasks},
        "unhealed": {"logical_model_bytes": 100, "domain_macro": 0.61, "question_weighted": 0.60, "tasks": tasks},
        "router": {"logical_model_bytes": 100, "domain_macro": 0.62, "question_weighted": 0.61, "tasks": tasks},
        "joint": {"logical_model_bytes": 100, "domain_macro": 0.63, "question_weighted": 0.62, "tasks": tasks},
    }
    report = {
        "format": REPORT_FORMAT,
        "panel_lock": {"sha256": "panel"},
        "arms": arms,
        "point_estimate_pareto": ["plain", "compact", "joint"],
        "pairwise_comparisons": {
            "joint": {"unhealed": {"domain_macro_delta": 0.02, "question_weighted_delta": 0.02, "paired_wins": 3, "paired_losses": 1}},
            "unhealed": {"unhealed": {}},
            "router": {"unhealed": {}},
        },
    }
    lock = {
        "format": LOCK_FORMAT,
        "panel_lock_sha256": "panel",
        "directional": {
            "reference_arm": "compact", "quality_ceiling_arm": "plain", "strong_compact_arm": "compact",
            "candidate_arms": ["unhealed", "router", "joint"], "hard_logical_byte_ceiling": 100,
            "require_global_point_estimate_pareto": True, "max_domain_macro_deficit_vs_reference": 0.05,
            "max_question_weighted_deficit_vs_reference": 0.05, "max_single_task_rate_deficit_vs_reference": 0.20,
        },
        "healing": {
            "unhealed_arm": "unhealed", "router_ablation_arm": "router", "joint_arm": "joint",
            "joint_requires_positive_domain_macro_delta": True,
            "joint_requires_nonnegative_question_weighted_delta": True,
            "joint_requires_more_paired_wins_than_losses": True,
            "max_single_task_rate_deficit_vs_unhealed": 0.10,
            "router_ablation_is_not_promoted": True,
        },
        "practical": {"fixed_reference_arms": ["plain", "compact"], "max_100gb_arms": 1},
        "trusted_full": {"max_arms": 3},
    }
    result = select(report, lock)
    assert result["selected_100gb_arms"] == ["joint"]
    assert result["practical_arms"] == ["plain", "compact", "joint"]


def main() -> None:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        print("100GB promotion selector self-test: PASS")
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    parser.add_argument("--lock", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    lock = json.loads(args.lock.read_text())
    result = select(report, lock)
    result["report"] = {"path": str(args.report.resolve()), "sha256": sha256(args.report)}
    result["promotion_lock"] = {"path": str(args.lock.resolve()), "sha256": sha256(args.lock)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output} practical={result['practical_arms']}")


if __name__ == "__main__":
    main()
