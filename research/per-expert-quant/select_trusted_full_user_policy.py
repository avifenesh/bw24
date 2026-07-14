#!/usr/bin/env python3
"""Apply an explicit user policy after preserving the frozen practical verdict."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any


PRACTICAL_FORMAT = "bw24-practical-promotion-v1"
FRONTIER_FORMAT = "bw24-cross-run-expanded-capability-frontier-v1"
POLICY_FORMAT = "bw24-user-trusted-full-directional-policy-v1"
OUTPUT_FORMAT = "bw24-effective-trusted-full-selection-v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(
    practical: dict[str, Any], frontier: dict[str, Any], policy: dict[str, Any]
) -> dict[str, Any]:
    if practical.get("format") != PRACTICAL_FORMAT:
        raise ValueError("wrong practical-promotion format")
    if frontier.get("format") != FRONTIER_FORMAT:
        raise ValueError("wrong directional-frontier format")
    if policy.get("format") != POLICY_FORMAT:
        raise ValueError("wrong user-policy format")

    reference = policy["reference_arm"]
    candidate = policy["candidate_arm"]
    arms = list(practical.get("trusted_full_arms", []))
    if not arms or arms[0] != reference or len(arms) != len(set(arms)):
        raise ValueError("practical selection lost the required reference ordering")
    missing = {reference, candidate} - set(frontier.get("arms", {}))
    if missing:
        raise ValueError(f"directional frontier is missing {sorted(missing)}")

    reference_row = frontier["arms"][reference]
    candidate_row = frontier["arms"][candidate]
    expected_questions = int(policy["required_total_questions"])
    if int(reference_row["total_questions"]) != expected_questions:
        raise ValueError("reference directional panel is incomplete")
    if int(candidate_row["total_questions"]) != expected_questions:
        raise ValueError("candidate directional panel is incomplete")

    deficit = int(reference_row["total_correct"]) - int(candidate_row["total_correct"])
    checks = {
        "within_directional_deficit": deficit <= int(policy["max_total_correct_deficit"]),
        "logical_byte_ceiling": int(candidate_row["logical_model_bytes"])
        <= int(policy["hard_logical_byte_ceiling"]),
    }
    forced_into_trusted = all(checks.values()) and candidate not in arms
    if forced_into_trusted:
        if len(arms) >= int(policy["max_trusted_full_arms"]):
            raise ValueError("user-qualified candidate does not fit the trusted-full arm budget")
        arms.append(candidate)

    return {
        "format": OUTPUT_FORMAT,
        "trusted_full_arms": arms,
        "base_trusted_full_arms": practical["trusted_full_arms"],
        "decision": {
            "candidate_arm": candidate,
            "reference_arm": reference,
            "candidate_total_correct": int(candidate_row["total_correct"]),
            "reference_total_correct": int(reference_row["total_correct"]),
            "total_correct_deficit": deficit,
            "checks": checks,
            "qualified_by_user_policy": all(checks.values()),
            "forced_into_trusted_full": forced_into_trusted,
        },
        "note": (
            "Post-hoc user decision governs trusted-eval coverage only; it does not alter "
            "the model, artifact, frozen directional verdict, or frozen practical verdict."
        ),
    }


def self_test() -> None:
    practical = {
        "format": PRACTICAL_FORMAT,
        "trusted_full_arms": ["plain", "compact"],
    }
    policy = {
        "format": POLICY_FORMAT,
        "reference_arm": "plain",
        "candidate_arm": "small",
        "max_total_correct_deficit": 1,
        "required_total_questions": 115,
        "hard_logical_byte_ceiling": 100,
        "max_trusted_full_arms": 5,
    }

    def frontier(candidate_score: int, candidate_bytes: int = 100) -> dict[str, Any]:
        return {
            "format": FRONTIER_FORMAT,
            "arms": {
                "plain": {
                    "total_correct": 85,
                    "total_questions": 115,
                    "logical_model_bytes": 186,
                },
                "small": {
                    "total_correct": candidate_score,
                    "total_questions": 115,
                    "logical_model_bytes": candidate_bytes,
                },
            },
        }

    selected = select(practical, frontier(84), policy)
    assert selected["trusted_full_arms"] == ["plain", "compact", "small"]
    assert selected["decision"]["forced_into_trusted_full"]
    expanded = dict(
        practical,
        trusted_full_arms=["plain", "compact", "bridge120", "bridge137"],
    )
    selected = select(expanded, frontier(84), policy)
    assert selected["trusted_full_arms"] == [
        "plain", "compact", "bridge120", "bridge137", "small"
    ]
    assert selected["decision"]["forced_into_trusted_full"]
    assert not select(practical, frontier(83), policy)["decision"]["qualified_by_user_policy"]
    assert not select(practical, frontier(84, 101), policy)["decision"]["qualified_by_user_policy"]
    already = dict(practical, trusted_full_arms=["plain", "compact", "small"])
    assert not select(already, frontier(84), policy)["decision"]["forced_into_trusted_full"]

    with tempfile.TemporaryDirectory(prefix="bw24-user-trusted-policy-") as tmp:
        out = Path(tmp) / "selection.json"
        out.write_text(json.dumps(selected))
        assert json.loads(out.read_text())["format"] == OUTPUT_FORMAT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--practical-promotion", type=Path)
    parser.add_argument("--frontier", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("trusted-full user-policy selector self-test: PASS")
        return
    if not all((args.practical_promotion, args.frontier, args.policy, args.output)):
        parser.error("--practical-promotion, --frontier, --policy, and --output are required")

    result = select(
        json.loads(args.practical_promotion.read_text()),
        json.loads(args.frontier.read_text()),
        json.loads(args.policy.read_text()),
    )
    result["practical_promotion"] = {
        "path": str(args.practical_promotion.resolve()),
        "sha256": sha256(args.practical_promotion),
    }
    result["directional_frontier"] = {
        "path": str(args.frontier.resolve()),
        "sha256": sha256(args.frontier),
    }
    result["user_policy"] = {
        "path": str(args.policy.resolve()),
        "sha256": sha256(args.policy),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {args.output} trusted_full={result['trusted_full_arms']}")


if __name__ == "__main__":
    main()
