#!/usr/bin/env python3
"""Apply the frozen SWE/Terminal directional gate to practical comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_comparison(path: Path, panel: str, baseline: str, candidate: str) -> dict[str, Any]:
    report = json.loads(path.read_text())
    if report.get("format") != "bw24-practical-comparison-v1":
        raise ValueError(f"wrong practical comparison format: {path}")
    if report.get("panel") != panel:
        raise ValueError(f"wrong panel in {path}")
    if report.get("baseline", {}).get("arm") != baseline:
        raise ValueError(f"wrong baseline in {path}")
    if report.get("candidate", {}).get("arm") != candidate:
        raise ValueError(f"wrong candidate in {path}")
    if report.get("n_tasks") != 12 or len(report.get("tasks", [])) != 12:
        raise ValueError(f"practical panel is not the frozen 12-task panel: {path}")
    return report


def select(
    promotion: dict[str, Any],
    gate_lock: dict[str, Any],
    comparison_root: Path,
    confirmation_lock: dict[str, Any] | None = None,
    frontier: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if promotion.get("format") != "bw24-100gb-directional-promotion-v1":
        raise ValueError("wrong directional promotion format")
    if gate_lock.get("format") != "bw24-100gb-promotion-gates-v1":
        raise ValueError("wrong promotion gate lock format")

    practical_cfg = gate_lock["practical"]
    trusted_cfg = gate_lock["trusted_full"]
    fixed = practical_cfg["fixed_reference_arms"]
    if len(fixed) != 2 or promotion.get("practical_arms", [])[:2] != fixed:
        raise ValueError("practical arms do not start with the frozen references")
    if len(promotion["practical_arms"]) > 2 + int(practical_cfg["max_100gb_arms"]):
        raise ValueError("too many 100GB practical candidates")

    preregistered_candidates = list(promotion["practical_arms"][2:])
    confirmatory_candidates: list[str] = []
    if confirmation_lock is not None:
        if confirmation_lock.get("format") != "bw24-surprise-practical-confirmation-v1":
            raise ValueError("wrong surprise-confirmation lock format")
        confirmatory_candidates = confirmation_lock.get("candidate_arms", [])
        if (
            not isinstance(confirmatory_candidates, list)
            or len(confirmatory_candidates)
            > int(confirmation_lock.get("max_confirmatory_arms", 0))
            or len(set(confirmatory_candidates)) != len(confirmatory_candidates)
        ):
            raise ValueError("invalid surprise-confirmation candidate arms")
        if set(confirmatory_candidates) & set(fixed):
            raise ValueError("a fixed reference cannot be a surprise-confirmation arm")
        if confirmation_lock.get("strong_compact_arm") != fixed[1]:
            raise ValueError("surprise confirmation uses the wrong compact reference")
        if float(confirmation_lock["max_solved_deficit_per_panel_vs_strong_compact"]) != float(
            practical_cfg["max_solved_deficit_per_panel_vs_strong_compact"]
        ):
            raise ValueError("surprise confirmation changed the frozen practical gate")
        if frontier is None or frontier.get("format") != "bw24-cross-run-expanded-capability-frontier-v1":
            raise ValueError("surprise confirmation requires the strict cross-run frontier")
        missing_frontier = set(preregistered_candidates + confirmatory_candidates) - set(
            frontier.get("arms", {})
        )
        if missing_frontier:
            raise ValueError(
                f"strict frontier is missing practical candidates {sorted(missing_frontier)}"
            )
        max_finalists = int(confirmation_lock.get("max_100gb_finalists", 0))
        if not 1 <= max_finalists <= int(trusted_cfg["max_arms"]) - len(fixed):
            raise ValueError("invalid surprise-confirmation finalist limit")

    practical_candidates = list(
        dict.fromkeys(preregistered_candidates + confirmatory_candidates)
    )

    strong = fixed[1]
    max_deficit = float(practical_cfg["max_solved_deficit_per_panel_vs_strong_compact"])
    decisions: dict[str, Any] = {}
    promoted_100gb: list[str] = []
    ranking: dict[str, tuple[float, float, float, float, int, str]] = {}
    for candidate in practical_candidates:
        panels: dict[str, Any] = {}
        passed = True
        for panel in ("swe", "terminal"):
            path = comparison_root / f"{strong}-vs-{candidate}.{panel}.json"
            report = load_comparison(path, panel, strong, candidate)
            strong_solved = sum(float(row["baseline_reward"]) for row in report["tasks"])
            candidate_solved = sum(float(row["candidate_reward"]) for row in report["tasks"])
            deficit = strong_solved - candidate_solved
            panel_passed = deficit <= max_deficit
            panels[panel] = {
                "passed": panel_passed,
                "strong_compact_solved": strong_solved,
                "candidate_solved": candidate_solved,
                "solved_deficit": deficit,
                "comparison": {"path": str(path.resolve()), "sha256": sha256(path)},
            }
            passed = passed and panel_passed
        decisions[candidate] = {"passed": passed, "panels": panels}
        if passed:
            promoted_100gb.append(candidate)

        total_solved = sum(float(row["candidate_solved"]) for row in panels.values())
        min_panel_solved = min(float(row["candidate_solved"]) for row in panels.values())
        capability = (frontier or {}).get("arms", {}).get(candidate, {})
        ranking[candidate] = (
            -total_solved,
            -min_panel_solved,
            -float(capability.get("domain_macro", 0.0)),
            -float(capability.get("question_weighted", 0.0)),
            int(capability.get("logical_model_bytes", 1 << 62)),
            candidate,
        )

    if confirmation_lock is not None:
        promoted_100gb.sort(key=ranking.__getitem__)
        promoted_100gb = promoted_100gb[
            : int(confirmation_lock["max_100gb_finalists"])
        ]

    trusted = list(dict.fromkeys(fixed + promoted_100gb))
    trusted = trusted[: int(trusted_cfg["max_arms"])]
    if trusted_cfg.get("required_reference_arm") not in trusted:
        raise ValueError("trusted-full selection omitted required reference")
    return {
        "format": "bw24-practical-promotion-v1",
        "directional_practical_arms": promotion["practical_arms"],
        "executed_practical_arms": fixed + practical_candidates,
        "preregistered_100gb_arms": preregistered_candidates,
        "confirmatory_100gb_arms": confirmatory_candidates,
        "decisions": decisions,
        "promoted_100gb_arms": promoted_100gb,
        "trusted_full_arms": trusted,
        "note": "Directional 12+12 practical screen; not full SWE-Bench or Terminal-Bench evidence.",
    }


def self_test() -> None:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="bw24-practical-select-") as tmp:
        root = Path(tmp)
        promotion = {
            "format": "bw24-100gb-directional-promotion-v1",
            "practical_arms": ["plain", "compact", "joint"],
        }
        lock = {
            "format": "bw24-100gb-promotion-gates-v1",
            "practical": {
                "fixed_reference_arms": ["plain", "compact"],
                "max_100gb_arms": 1,
                "max_solved_deficit_per_panel_vs_strong_compact": 1,
            },
            "trusted_full": {"max_arms": 3, "required_reference_arm": "plain"},
        }
        for panel, candidate_rewards in (("swe", [1] * 11 + [0]), ("terminal", [1] * 10 + [0, 0])):
            report = {
                "format": "bw24-practical-comparison-v1",
                "panel": panel,
                "n_tasks": 12,
                "baseline": {"arm": "compact"},
                "candidate": {"arm": "joint"},
                "tasks": [
                    {"baseline_reward": 1, "candidate_reward": value}
                    for value in candidate_rewards
                ],
            }
            (root / f"compact-vs-joint.{panel}.json").write_text(json.dumps(report))
        result = select(promotion, lock, root)
        assert result["promoted_100gb_arms"] == []
        assert result["trusted_full_arms"] == ["plain", "compact"]

        confirmation = {
            "format": "bw24-surprise-practical-confirmation-v1",
            "candidate_arms": ["router"],
            "max_confirmatory_arms": 1,
            "strong_compact_arm": "compact",
            "max_solved_deficit_per_panel_vs_strong_compact": 1,
            "max_100gb_finalists": 1,
        }
        frontier = {
            "format": "bw24-cross-run-expanded-capability-frontier-v1",
            "arms": {
                "joint": {"domain_macro": 0.6, "question_weighted": 0.6, "logical_model_bytes": 100},
                "router": {"domain_macro": 0.7, "question_weighted": 0.7, "logical_model_bytes": 100},
            },
        }
        for panel in ("swe", "terminal"):
            report = {
                "format": "bw24-practical-comparison-v1",
                "panel": panel,
                "n_tasks": 12,
                "baseline": {"arm": "compact"},
                "candidate": {"arm": "router"},
                "tasks": [
                    {"baseline_reward": 1, "candidate_reward": 1}
                    for _ in range(12)
                ],
            }
            (root / f"compact-vs-router.{panel}.json").write_text(json.dumps(report))
        result = select(promotion, lock, root, confirmation, frontier)
        assert result["confirmatory_100gb_arms"] == ["router"]
        assert result["promoted_100gb_arms"] == ["router"]
        assert result["trusted_full_arms"] == ["plain", "compact", "router"]
    print("practical promotion selector self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promotion", type=Path)
    parser.add_argument("--gate-lock", type=Path)
    parser.add_argument("--comparison-root", type=Path)
    parser.add_argument("--confirmation-lock", type=Path)
    parser.add_argument("--frontier", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not all((args.promotion, args.gate_lock, args.comparison_root, args.output)):
        parser.error("--promotion, --gate-lock, --comparison-root, and --output are required")
    result = select(
        json.loads(args.promotion.read_text()),
        json.loads(args.gate_lock.read_text()),
        args.comparison_root,
        json.loads(args.confirmation_lock.read_text()) if args.confirmation_lock else None,
        json.loads(args.frontier.read_text()) if args.frontier else None,
    )
    result["directional_promotion"] = {
        "path": str(args.promotion.resolve()), "sha256": sha256(args.promotion)
    }
    result["gate_lock"] = {
        "path": str(args.gate_lock.resolve()), "sha256": sha256(args.gate_lock)
    }
    if args.confirmation_lock:
        result["confirmation_lock"] = {
            "path": str(args.confirmation_lock.resolve()),
            "sha256": sha256(args.confirmation_lock),
        }
        result["frontier"] = {
            "path": str(args.frontier.resolve()), "sha256": sha256(args.frontier)
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {args.output} trusted_full={result['trusted_full_arms']}")


if __name__ == "__main__":
    main()
