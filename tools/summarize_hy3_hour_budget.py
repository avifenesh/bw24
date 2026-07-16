#!/usr/bin/env python3
"""Build a hash-bound Hy3 prune/quant conclusion from bounded screens.

This is intentionally separate from ``summarize_hy3_quant_research.py``.  That
tool proves the old 4,746-document + full-agentic contract; this one proves the
resource-limited decision after that escalation was explicitly retired.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import tempfile
from datetime import datetime, timezone
from typing import Any


FORMAT = "bw24-hy3-hour-budget-conclusion-v1"
RECEIPT_FORMAT = "bw24-hy3-hour-budget-conclusion-receipt-v1"
HOURISH_FORMAT = "bw24-hourish-capability-screen-v1"
EXPANDED_FORMAT = "bw24-expanded-capability-screen-v1"
PRACTICAL_FORMAT = "bw24-practical-promotion-v1"
DAMAGE_FORMAT = "bw24-hy3-quant-plan-damage-v1"


def load(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(payload)
        temporary = pathlib.Path(handle.name)
    os.replace(temporary, path)


def atomic_text(path: pathlib.Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(value)
        temporary = pathlib.Path(handle.name)
    os.replace(temporary, path)


def require_format(value: dict[str, Any], expected: str, label: str) -> None:
    if value.get("format") != expected:
        raise ValueError(f"wrong {label} format: {value.get('format')!r}")


def arm(summary: dict[str, Any], name: str) -> dict[str, Any]:
    arms = summary.get("arms")
    if not isinstance(arms, dict) or not isinstance(arms.get(name), dict):
        raise ValueError(f"missing arm {name}")
    value = arms[name]
    for field in ("logical_model_bytes", "total_correct", "total_questions"):
        if not isinstance(value.get(field), (int, float)):
            raise ValueError(f"{name} lacks numeric {field}")
    if int(value["total_questions"]) <= 0:
        raise ValueError(f"{name} has no scored questions")
    return value


def public_row(name: str, value: dict[str, Any], source: str) -> dict[str, Any]:
    return {
        "arm": name,
        "source": source,
        "logical_model_bytes": int(value["logical_model_bytes"]),
        "total_correct": int(value["total_correct"]),
        "total_questions": int(value["total_questions"]),
        "question_weighted": float(value["question_weighted"]),
        "domain_macro": float(value["domain_macro"]),
        "tasks": value.get("tasks", {}),
    }


def ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        raise ValueError("ratio denominator must be positive")
    return numerator / denominator


def build(
    hourish: dict[str, Any],
    expanded: dict[str, Any],
    layer100: dict[str, Any],
    bridge: dict[str, Any],
    practical: dict[str, Any],
    damage: dict[str, Any],
    *,
    max_candidate_wall_seconds: int,
) -> dict[str, Any]:
    require_format(hourish, HOURISH_FORMAT, "hourish")
    require_format(expanded, EXPANDED_FORMAT, "expanded")
    require_format(layer100, EXPANDED_FORMAT, "Layer100")
    require_format(bridge, EXPANDED_FORMAT, "bridge")
    require_format(practical, PRACTICAL_FORMAT, "practical")
    require_format(damage, DAMAGE_FORMAT, "private damage")
    if max_candidate_wall_seconds <= 0 or max_candidate_wall_seconds > 3600:
        raise ValueError("candidate wall budget must be in 1..3600 seconds")

    plain = arm(expanded, "plain_quant")
    traffic = arm(expanded, "traffic_nvfp4_53_q2_139")
    compact = arm(layer100, "layer_balanced100")
    layer120 = arm(bridge, "layer_balanced120")
    layer137 = arm(bridge, "layer_balanced137")
    reap = arm(hourish, "plain_reap_quant")
    hourish_plain = arm(hourish, "plain_quant")

    promoted = practical.get("promoted_100gb_arms")
    decisions = practical.get("decisions")
    if promoted != ["layer_balanced100"]:
        raise ValueError("practical evidence did not uniquely promote Layer100")
    if not isinstance(decisions, dict) or not decisions.get("layer_balanced100", {}).get(
        "passed"
    ):
        raise ValueError("Layer100 practical gate did not pass")

    damage_plans = damage.get("plans")
    required_damage = {"traffic137", "layer100", "layer120", "layer137"}
    if not isinstance(damage_plans, dict) or not required_damage.issubset(damage_plans):
        raise ValueError("private damage evidence lacks the four registered plans")
    if damage.get("public_eval_data_used") is not False:
        raise ValueError("private damage evidence used public evaluation data")

    # The bridge candidates are strictly worse than an already-built candidate.
    if not (
        int(layer120["logical_model_bytes"]) > int(compact["logical_model_bytes"])
        and int(layer120["total_correct"]) < int(compact["total_correct"])
    ):
        raise ValueError("Layer120 is not dominated by Layer100 as expected")
    if not (
        int(layer137["logical_model_bytes"]) >= int(traffic["logical_model_bytes"]) - 2_000_000
        and int(layer137["total_correct"]) < int(traffic["total_correct"])
    ):
        raise ValueError("Layer137 is not dominated by Traffic137 as expected")

    quality_ceiling = public_row("plain_quant", plain, "expanded115")
    balanced = public_row("traffic_nvfp4_53_q2_139", traffic, "expanded115")
    hard_cap = public_row("layer_balanced100", compact, "expanded115")
    quality_ceiling["role"] = "quality_ceiling"
    balanced["role"] = "balanced_compact"
    hard_cap["role"] = "hard_100gb_efficiency_winner"

    for row in (balanced, hard_cap):
        row["size_fraction_vs_plain"] = ratio(
            row["logical_model_bytes"], quality_ceiling["logical_model_bytes"]
        )
        row["score_retention_vs_plain"] = ratio(
            row["question_weighted"], quality_ceiling["question_weighted"]
        )

    return {
        "format": FORMAT,
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "resource_contract": {
            "max_candidate_wall_seconds": max_candidate_wall_seconds,
            "multi_day_trusted_eval_retired": True,
            "full_agentic_escalation_retired": True,
            "partial_trusted4746_is_ranking_evidence": False,
            "future_candidate_policy": (
                "paired bounded capability screen only; stop at the wall budget and never "
                "heal or allocate from public scores"
            ),
        },
        "pareto_frontier": [quality_ceiling, balanced, hard_cap],
        "method_decision": {
            "best_quality": "plain_quant",
            "best_balanced_compression": "traffic_nvfp4_53_q2_139",
            "best_hard_100gb": "layer_balanced100",
            "overall_resource_limited_choice": "layer_balanced100",
            "why": (
                "Layer100 retains 71/115 versus Traffic137's 74/115 while removing "
                "37,459,869,696 logical bytes, and it passed the registered 12+12 "
                "practical compact-model gate."
            ),
            "architecture": (
                "Use quant-aware layer-balanced pruning with private calibration and healing "
                "when the 100GB cap is binding; otherwise use traffic-aware mixed precision "
                "without pruning."
            ),
        },
        "rejections": {
            "plain_reap_quant": {
                **public_row("plain_reap_quant", reap, "hourish56"),
                "plain_control_correct": int(hourish_plain["total_correct"]),
                "reason": "public REAP50 selection loses too much capability on Hy3",
            },
            "layer_balanced120": {
                **public_row("layer_balanced120", layer120, "expanded115"),
                "reason": "strictly dominated by the smaller, higher-scoring Layer100",
            },
            "layer_balanced137": {
                **public_row("layer_balanced137", layer137, "expanded115"),
                "reason": "same-size public score is below Traffic137",
            },
        },
        "private_only_corroboration": {
            "used_for_public_selection": False,
            "lowest_private_damage_plan": damage.get("lowest_private_damage_plan"),
            "plans": {
                name: {
                    "logical_bytes": int(damage_plans[name]["logical_bytes"]),
                    "total_additive_damage": float(
                        damage_plans[name]["total_additive_damage"]
                    ),
                }
                for name in sorted(required_damage)
            },
        },
        "practical": {
            "format": practical["format"],
            "executed_arms": practical.get("executed_practical_arms"),
            "promoted_100gb_arms": promoted,
            "layer_balanced100": decisions["layer_balanced100"],
            "note": practical.get("note"),
        },
    }


def markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Hy3 prune/quant conclusion under a one-hour candidate budget",
        "",
        "The multi-day trusted/full escalation is retired. Partial trusted4746 output is not "
        "ranking evidence. Every future candidate is capped at one wall-clock hour.",
        "",
        "| Role | Arm | Logical GB | Score | Retention vs plain |",
        "|---|---|---:|---:|---:|",
    ]
    for row in result["pareto_frontier"]:
        retention = row.get("score_retention_vs_plain", 1.0)
        lines.append(
            f"| {row['role']} | `{row['arm']}` | "
            f"{row['logical_model_bytes'] / 1e9:.3f} | "
            f"{row['total_correct']}/{row['total_questions']} | {retention:.1%} |"
        )
    decision = result["method_decision"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"**Resource-limited winner: `{decision['overall_resource_limited_choice']}`.**",
            "",
            decision["why"],
            "",
            decision["architecture"],
            "",
            "Public REAP50 is rejected for Hy3. Layer120 is dominated by Layer100, and "
            "Layer137 is dominated at essentially equal size by Traffic137.",
            "",
        ]
    )
    return "\n".join(lines)


def self_test() -> None:
    def scored(fmt: str, values: dict[str, tuple[int, int]]) -> dict[str, Any]:
        return {
            "format": fmt,
            "arms": {
                name: {
                    "logical_model_bytes": size,
                    "total_correct": correct,
                    "total_questions": 115 if name != "plain_reap_quant" else 56,
                    "question_weighted": correct / (115 if name != "plain_reap_quant" else 56),
                    "domain_macro": correct / (115 if name != "plain_reap_quant" else 56),
                    "tasks": {"test": {"n": 1, "successes": 1}},
                }
                for name, (size, correct) in values.items()
            },
        }

    hourish = scored(
        HOURISH_FORMAT,
        {"plain_quant": (186_000_000_000, 37), "plain_reap_quant": (105_000_000_000, 19)},
    )
    expanded = scored(
        EXPANDED_FORMAT,
        {"plain_quant": (186_000_000_000, 85), "traffic_nvfp4_53_q2_139": (137_459_000_000, 74)},
    )
    compact = scored(EXPANDED_FORMAT, {"layer_balanced100": (99_999_000_000, 71)})
    bridge = scored(
        EXPANDED_FORMAT,
        {"layer_balanced120": (119_999_000_000, 66), "layer_balanced137": (137_458_000_000, 70)},
    )
    practical = {
        "format": PRACTICAL_FORMAT,
        "promoted_100gb_arms": ["layer_balanced100"],
        "executed_practical_arms": ["plain_quant", "traffic_nvfp4_53_q2_139", "layer_balanced100"],
        "decisions": {"layer_balanced100": {"passed": True, "panels": {}}},
    }
    damage = {
        "format": DAMAGE_FORMAT,
        "public_eval_data_used": False,
        "lowest_private_damage_plan": "layer137",
        "plans": {
            name: {"logical_bytes": size, "total_additive_damage": float(index + 1)}
            for index, (name, size) in enumerate(
                (("traffic137", 137), ("layer100", 100), ("layer120", 120), ("layer137", 137))
            )
        },
    }
    result = build(hourish, expanded, compact, bridge, practical, damage, max_candidate_wall_seconds=3600)
    assert result["method_decision"]["overall_resource_limited_choice"] == "layer_balanced100"
    assert result["rejections"]["plain_reap_quant"]["total_correct"] == 19
    assert len(result["pareto_frontier"]) == 3
    try:
        build(hourish, expanded, compact, bridge, practical, damage, max_candidate_wall_seconds=3601)
    except ValueError as error:
        assert "1..3600" in str(error)
    else:
        raise AssertionError("oversized budget was accepted")
    print("hour-budget conclusion self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hourish", type=pathlib.Path)
    parser.add_argument("--expanded", type=pathlib.Path)
    parser.add_argument("--layer100", type=pathlib.Path)
    parser.add_argument("--bridge", type=pathlib.Path)
    parser.add_argument("--practical", type=pathlib.Path)
    parser.add_argument("--damage", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--markdown", type=pathlib.Path)
    parser.add_argument("--receipt", type=pathlib.Path)
    parser.add_argument("--max-candidate-wall-seconds", type=int, default=3600)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    required = ("hourish", "expanded", "layer100", "bridge", "practical", "damage", "output", "markdown", "receipt")
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error("missing required arguments: " + ", ".join(missing))

    inputs = {
        "hourish": args.hourish,
        "expanded": args.expanded,
        "layer100": args.layer100,
        "bridge": args.bridge,
        "practical": args.practical,
        "damage": args.damage,
    }
    values = {name: load(path) for name, path in inputs.items()}
    result = build(
        values["hourish"],
        values["expanded"],
        values["layer100"],
        values["bridge"],
        values["practical"],
        values["damage"],
        max_candidate_wall_seconds=args.max_candidate_wall_seconds,
    )
    atomic_json(args.output, result)
    atomic_text(args.markdown, markdown(result))
    receipt = {
        "format": RECEIPT_FORMAT,
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in inputs.items()
        },
        "outputs": {
            "json": {"path": str(args.output.resolve()), "sha256": sha256(args.output)},
            "markdown": {"path": str(args.markdown.resolve()), "sha256": sha256(args.markdown)},
        },
        "max_candidate_wall_seconds": args.max_candidate_wall_seconds,
    }
    atomic_json(args.receipt, receipt)
    print(f"wrote {args.output} and {args.receipt}")


if __name__ == "__main__":
    main()
