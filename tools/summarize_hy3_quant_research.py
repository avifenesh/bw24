#!/usr/bin/env python3
"""Synthesize the complete Hy3 prune/quant study into one hash-bound conclusion."""

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


OUTPUT_FORMAT = "bw24-hy3-quant-research-conclusion-v1"
RECEIPT_FORMAT = "bw24-hy3-quant-research-conclusion-receipt-v1"
PLAN_FORMAT = "bw24-expert-tier-plan-v2"
PLAN_ARMS = {
    "uncentered": "smart100_iq3_iq4_q4_empirical",
    "centered": "smart100_iq3_iq4_q4_centered",
    "pareto": "smart100_iq3_iq4_q4_pareto",
    "layer_balanced": "layer_balanced100",
}
PLAN_DAMAGE_KEYS = {
    "uncentered": "uncentered",
    "centered": "old_centered",
    "pareto": "pareto",
}
HEALING_ARMS = (
    "prune100_unhealed",
    "prune100_router_repair",
    "prune100_joint_heal",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def source(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def verify_receipt(path: Path) -> None:
    receipt = load(path)
    if receipt.get("format") != RECEIPT_FORMAT:
        raise ValueError("wrong conclusion receipt format")
    if not re.fullmatch(r"[0-9a-f]{40}", str(receipt.get("analysis_commit", ""))):
        raise ValueError("invalid conclusion analysis commit")
    if receipt.get("public_eval_data_used_for_allocation_or_healing") is not False:
        raise ValueError("conclusion receipt does not preserve private-only allocation")
    items = [*receipt.get("inputs", []), *receipt.get("outputs", []), receipt.get("script")]
    if not items or any(not isinstance(item, dict) for item in items):
        raise ValueError("conclusion receipt has malformed evidence entries")
    for item in items:
        target = Path(item["path"])
        if not target.is_file() or sha256(target) != item["sha256"]:
            raise ValueError(f"conclusion evidence mismatch: {target}")


def format_efficiency_summary(effects: dict[str, Any]) -> dict[str, Any]:
    expected = {"Q8_0", "NVFP4", "IQ4_XS", "Q4_K", "IQ3_S", "Q3_K", "Q2_K"}
    totals = effects["format_totals"]
    if set(totals) != expected:
        raise ValueError("format totals do not cover the seven-format study")
    rows: list[dict[str, Any]] = []
    for qtype, values in totals.items():
        encoded_bytes = int(values["encoded_bytes"])
        damage = float(values["full_scaled_squared_error"])
        if encoded_bytes <= 0 or not math.isfinite(damage) or damage < 0:
            raise ValueError(f"invalid format total for {qtype}")
        dominated_by = sorted(
            other
            for other, candidate in totals.items()
            if other != qtype
            and int(candidate["encoded_bytes"]) <= encoded_bytes
            and float(candidate["full_scaled_squared_error"]) <= damage
            and (
                int(candidate["encoded_bytes"]) < encoded_bytes
                or float(candidate["full_scaled_squared_error"]) < damage
            )
        )
        rows.append({
            "qtype": qtype,
            "encoded_bytes": encoded_bytes,
            "full_scaled_squared_error": damage,
            "point_estimate_pareto": not dominated_by,
            "dominated_by": dominated_by,
        })
    rows.sort(key=lambda row: (
        row["encoded_bytes"], row["full_scaled_squared_error"], row["qtype"]
    ))
    same_byte: list[dict[str, Any]] = []
    byte_groups: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        byte_groups.setdefault(row["encoded_bytes"], []).append(row)
    for encoded_bytes, group in sorted(byte_groups.items()):
        if len(group) < 2:
            continue
        winner = min(group, key=lambda row: (
            row["full_scaled_squared_error"], row["qtype"]
        ))
        for loser in sorted(group, key=lambda row: row["qtype"]):
            if loser is winner:
                continue
            loser_damage = float(loser["full_scaled_squared_error"])
            same_byte.append({
                "encoded_bytes": encoded_bytes,
                "winner": winner["qtype"],
                "loser": loser["qtype"],
                "damage_reduction": (
                    0.0 if loser_damage == 0 else
                    1.0 - float(winner["full_scaled_squared_error"]) / loser_damage
                ),
            })
    return {
        "formats": rows,
        "point_estimate_pareto": [
            row["qtype"] for row in rows if row["point_estimate_pareto"]
        ],
        "same_byte_winners": same_byte,
    }


def healing_ablation_summary(frontier: dict[str, Any]) -> dict[str, Any]:
    if frontier.get("format") != "bw24-cross-run-expanded-capability-frontier-v1":
        raise ValueError("wrong healing frontier format")
    arms = frontier.get("arms")
    if not isinstance(arms, dict) or any(name not in arms for name in HEALING_ARMS):
        raise ValueError("healing frontier does not contain the complete ablation")
    rows: dict[str, dict[str, Any]] = {}
    logical_bytes = set()
    total_questions = set()
    for name in HEALING_ARMS:
        arm = arms[name]
        tasks = arm.get("tasks")
        if not isinstance(tasks, dict) or not tasks:
            raise ValueError(f"{name} has no task evidence")
        task_successes: dict[str, int] = {}
        task_counts: dict[str, int] = {}
        for task, values in tasks.items():
            successes = int(values["successes"])
            count = int(values["n"])
            if count <= 0 or successes < 0 or successes > count:
                raise ValueError(f"{name}/{task} has invalid task counts")
            task_successes[str(task)] = successes
            task_counts[str(task)] = count
        size = int(arm["logical_model_bytes"])
        questions = int(arm["total_questions"])
        correct = int(arm["total_correct"])
        question_weighted = float(arm["question_weighted"])
        domain_macro = float(arm["domain_macro"])
        if (
            size <= 0 or questions <= 0 or correct < 0 or correct > questions
            or not math.isfinite(question_weighted) or not math.isfinite(domain_macro)
            or not 0 <= question_weighted <= 1 or not 0 <= domain_macro <= 1
            or sum(task_counts.values()) != questions
            or sum(task_successes.values()) != correct
        ):
            raise ValueError(f"{name} has invalid aggregate evidence")
        logical_bytes.add(size)
        total_questions.add(questions)
        rows[name] = {
            "logical_model_bytes": size,
            "total_correct": correct,
            "total_questions": questions,
            "question_weighted": question_weighted,
            "domain_macro": domain_macro,
            "task_successes": task_successes,
            "task_counts": task_counts,
        }
    if len(logical_bytes) != 1 or len(total_questions) != 1:
        raise ValueError("healing ablation is not matched for size and question count")
    task_shapes = {
        tuple(sorted(row["task_counts"].items())) for row in rows.values()
    }
    if len(task_shapes) != 1:
        raise ValueError("healing ablation task sets or counts differ")
    unhealed = rows["prune100_unhealed"]
    deltas: dict[str, dict[str, Any]] = {}
    for name in HEALING_ARMS[1:]:
        arm = rows[name]
        deltas[name] = {
            "total_correct_delta": arm["total_correct"] - unhealed["total_correct"],
            "question_weighted_delta": (
                arm["question_weighted"] - unhealed["question_weighted"]
            ),
            "domain_macro_delta": arm["domain_macro"] - unhealed["domain_macro"],
            "task_success_delta": {
                task: arm["task_successes"][task] - unhealed["task_successes"][task]
                for task in sorted(unhealed["task_successes"])
            },
        }
    random_names = sorted(name for name in arms if re.fullmatch(r"random_[0-9]+", name))
    random_control_variance = None
    if random_names:
        if len(random_names) != 3:
            raise ValueError("expected exactly three random controls")
        random_rows = [arms[name] for name in random_names]
        random_sizes = {int(row["logical_model_bytes"]) for row in random_rows}
        random_questions = {int(row["total_questions"]) for row in random_rows}
        if len(random_sizes) != 1 or random_questions != total_questions:
            raise ValueError("random controls are not size/question matched")
        random_control_variance = {"arms": random_names}
        for metric in ("question_weighted", "domain_macro"):
            values = [float(row[metric]) for row in random_rows]
            if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
                raise ValueError(f"invalid random-control {metric}")
            mean = math.fsum(values) / len(values)
            random_control_variance[metric] = {
                "values": values,
                "mean": mean,
                "population_stddev": math.sqrt(
                    math.fsum((value - mean) ** 2 for value in values) / len(values)
                ),
                "minimum": min(values),
                "maximum": max(values),
            }
        random_control_variance["matched_logical_model_bytes"] = next(iter(random_sizes))
    return {
        "matched_logical_model_bytes": next(iter(logical_bytes)),
        "total_questions": next(iter(total_questions)),
        "arms": rows,
        "deltas_vs_unhealed": deltas,
        "best_question_weighted_arm": max(
            HEALING_ARMS, key=lambda name: (rows[name]["question_weighted"], name)
        ),
        "best_domain_macro_arm": max(
            HEALING_ARMS, key=lambda name: (rows[name]["domain_macro"], name)
        ),
        "random_control_variance": random_control_variance,
    }


def directional_frontier_summary(frontier: dict[str, Any]) -> dict[str, Any]:
    if frontier.get("format") != "bw24-cross-run-expanded-capability-frontier-v1":
        raise ValueError("wrong directional frontier format")
    arms = frontier.get("arms")
    pareto = frontier.get("point_estimate_pareto")
    if not isinstance(arms, dict) or not arms or not isinstance(pareto, list):
        raise ValueError("directional frontier is incomplete")
    if any(name not in arms for name in pareto):
        raise ValueError("directional Pareto arm is absent from arm evidence")
    rows: list[dict[str, Any]] = []
    for name, arm in arms.items():
        size = int(arm["logical_model_bytes"])
        question_weighted = float(arm["question_weighted"])
        domain_macro = float(arm["domain_macro"])
        if (
            size <= 0 or not math.isfinite(question_weighted)
            or not math.isfinite(domain_macro)
            or not 0 <= question_weighted <= 1 or not 0 <= domain_macro <= 1
        ):
            raise ValueError(f"invalid directional arm {name}")
        rows.append({
            "arm": str(name),
            "logical_model_bytes": size,
            "question_weighted": question_weighted,
            "domain_macro": domain_macro,
            "point_estimate_pareto": name in pareto,
            "tasks": arm.get("tasks", {}),
        })
    rows.sort(key=lambda row: (
        row["logical_model_bytes"], -row["question_weighted"], row["arm"]
    ))
    return {"arms": rows, "point_estimate_pareto": pareto}


def plan_summary(
    name: str,
    path: Path,
    plan: dict[str, Any],
    damage: dict[str, Any],
    frontier: dict[str, Any],
) -> dict[str, Any]:
    if plan.get("format") != PLAN_FORMAT:
        raise ValueError(f"{name} has unsupported plan format")
    if plan.get("calibration", {}).get("public_eval_data_used_for_selection") is not False:
        raise ValueError(f"{name} does not attest private-only allocation")
    arm = PLAN_ARMS[name]
    logical_bytes = int(plan["policy"]["result_logical_bytes"])
    damage_key = PLAN_DAMAGE_KEYS.get(name)
    if damage_key is not None:
        plan_damage = damage["plans"][damage_key]
        if plan_damage["sha256"] != sha256(path):
            raise ValueError(f"{name} damage receipt binds a different plan")
        if int(plan_damage["logical_bytes"]) != logical_bytes:
            raise ValueError(f"{name} logical bytes differ between plan and damage report")
        private_damage = {
            "metric": "measured_additive_projection_output_damage",
            "source": "independent_plan_damage_receipt",
            "total": float(plan_damage["total_additive_damage"]),
            "centered_total": None,
            "normalized_objective": None,
            "prune": float(plan_damage["prune_damage"]),
            "retained_quant": float(plan_damage["retained_quant_damage"]),
            "projection_quant": plan_damage["projection_quant_damage"],
            "top_damage_cells": plan_damage["top_damage_cells"],
        }
    else:
        selection = plan["selection"]
        absolute = float(selection["estimated_absolute_output_damage"])
        centered = float(selection["estimated_centered_output_damage"])
        objective = float(selection["estimated_objective"])
        if not all(math.isfinite(value) and value >= 0 for value in (
            absolute, centered, objective
        )):
            raise ValueError(f"{name} has invalid private damage estimates")
        private_damage = {
            "metric": "measured_additive_projection_output_damage",
            "source": "optimal_plan_selection_estimate",
            "total": absolute,
            "centered_total": centered,
            "normalized_objective": objective,
            "prune": None,
            "retained_quant": None,
            "projection_quant": None,
            "top_damage_cells": None,
        }
    directional = frontier["arms"].get(arm)
    if directional is not None and int(directional["logical_model_bytes"]) != logical_bytes:
        raise ValueError(f"{name} directional artifact has different logical bytes")
    layers = plan["layer_summary"]
    layer_qtypes: dict[str, dict[str, int]] = {}
    projection_qtypes: dict[str, dict[str, int]] = {}
    for assignment in plan["assignments"]:
        layer = str(assignment["layer"])
        qtype = str(assignment["qtype"])
        count = len(assignment["experts"])
        for projection in assignment["projections"]:
            layer_counts = layer_qtypes.setdefault(layer, {})
            layer_counts[qtype] = layer_counts.get(qtype, 0) + count
            projection_counts = projection_qtypes.setdefault(projection, {})
            projection_counts[qtype] = projection_counts.get(qtype, 0) + count
    derived_qtypes: dict[str, int] = {}
    for counts in layer_qtypes.values():
        for qtype, count in counts.items():
            derived_qtypes[qtype] = derived_qtypes.get(qtype, 0) + count
    policy_qtypes = {
        str(qtype): int(count)
        for qtype, count in plan["policy"]["qtype_projection_counts"].items()
    }
    for qtype in policy_qtypes:
        derived_qtypes.setdefault(qtype, 0)
    if derived_qtypes != policy_qtypes:
        raise ValueError(f"{name} assignment counts do not match policy summary")
    return {
        "arm": arm,
        "plan": source(path),
        "recipe": plan["recipe"],
        "logical_model_bytes": logical_bytes,
        "logical_model_gib": logical_bytes / 2**30,
        "retained_experts": int(plan["selection"]["retained_experts"]),
        "pruned_experts": int(plan["selection"]["pruned_experts"]),
        "minimum_survivors_in_layer": min(int(row["retained"]) for row in layers.values()),
        "maximum_pruned_in_layer": max(int(row["pruned"]) for row in layers.values()),
        "qtype_projection_counts": plan["policy"]["qtype_projection_counts"],
        "projection_qtype_counts": projection_qtypes,
        "layer_qtype_projection_counts": layer_qtypes,
        "layer_retention": {
            layer: {"retained": int(row["retained"]), "pruned": int(row["pruned"])}
            for layer, row in layers.items()
        },
        "private_damage_metric": private_damage["metric"],
        "private_damage_source": private_damage["source"],
        "private_total_additive_damage": private_damage["total"],
        "private_centered_output_damage": private_damage["centered_total"],
        "private_normalized_objective": private_damage["normalized_objective"],
        "private_prune_damage": private_damage["prune"],
        "private_retained_quant_damage": private_damage["retained_quant"],
        "private_projection_quant_damage": private_damage["projection_quant"],
        "private_top_damage_cells": private_damage["top_damage_cells"],
        "directional": (
            {
                "domain_macro": float(directional["domain_macro"]),
                "question_weighted": float(directional["question_weighted"]),
                "tasks": directional.get("tasks", {}),
            }
            if directional is not None
            else None
        ),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    effects = load(args.effects)
    damage = load(args.damage)
    frontier = load(args.frontier)
    healing_frontier = load(args.healing_frontier)
    directional = load(args.directional_promotion)
    practical = load(args.practical_promotion)
    trusted = load(args.trusted_report)
    full = load(args.full_agentic)
    if effects.get("format") != "bw24-hy3-quant-effects-map-v1":
        raise ValueError("wrong effects format")
    if effects.get("public_eval_data_used_for_selection") is not False:
        raise ValueError("effects map is not private-only")
    if set(effects["measurement"]["qtypes"]) != {
        "Q8_0", "NVFP4", "IQ4_XS", "Q4_K", "IQ3_S", "Q3_K", "Q2_K"
    }:
        raise ValueError("effects map is not the seven-format study")
    expected_formats = (
        (damage, "bw24-hy3-quant-plan-damage-v1"),
        (frontier, "bw24-cross-run-expanded-capability-frontier-v1"),
        (directional, "bw24-smart100-directional-promotion-v1"),
        (practical, "bw24-practical-promotion-v1"),
        (trusted, "bw24-promoted-candidate-v1"),
        (full, "bw24-full-agentic-comparison-v1"),
    )
    for payload, expected in expected_formats:
        if payload.get("format") != expected:
            raise ValueError(f"expected {expected}, got {payload.get('format')}")
    if damage.get("public_eval_data_used") is not False:
        raise ValueError("private plan damage report used public eval")
    if damage.get("lowest_private_damage_plan") != "pareto":
        raise ValueError("Pareto-pruned plan is not the private-damage winner")
    trusted_counts = trusted.get("n_per_task")
    trusted_documents = int(trusted.get("documents_per_arm", 0))
    if (
        not isinstance(trusted_counts, dict)
        or not trusted_counts
        or any(not isinstance(value, int) or value <= 0 for value in trusted_counts.values())
        or sum(trusted_counts.values()) != trusted_documents
        or trusted_documents != 4746
    ):
        raise ValueError("trusted capability report is not the full 4,746-document suite")
    if full.get("baseline") != "plain_quant" or int(full.get("total_tasks", 0)) != 589:
        raise ValueError("full agentic report is not plain-vs-finalist SWE500+Terminal89")
    finalist = str(full["candidate"])
    if trusted["selection"]["selected_finalist"] != finalist:
        raise ValueError("trusted and full-agentic finalists differ")
    if finalist not in practical["trusted_full_arms"] or finalist not in frontier["arms"]:
        raise ValueError("finalist is absent from the promotion chain")

    plan_paths = {name: path for name, path in args.plan}
    if set(plan_paths) != set(PLAN_ARMS):
        raise ValueError(f"plans must be exactly {sorted(PLAN_ARMS)}")
    plans = {
        name: plan_summary(name, path, load(path), damage, frontier)
        for name, path in sorted(plan_paths.items())
    }
    baseline = frontier["arms"]["plain_quant"]
    winner = frontier["arms"][finalist]
    winner_method = next(
        (summary for summary in plans.values() if summary["arm"] == finalist), None
    )
    size_reduction = 1.0 - int(winner["logical_model_bytes"]) / int(
        baseline["logical_model_bytes"]
    )
    format_efficiency = format_efficiency_summary(effects)
    healing_ablation = healing_ablation_summary(healing_frontier)
    directional_frontier = directional_frontier_summary(frontier)
    result = {
        "format": OUTPUT_FORMAT,
        "analysis_commit": args.analysis_commit,
        "study_goal": (
            "smallest validated MoE prune/quant method retaining the most task performance"
        ),
        "data_separation": {
            "allocation_and_healing_used_public_eval_data": False,
            "public_eval_used_only_for_frozen_gate_promotion": True,
        },
        "recommended_method": {
            "arm": finalist,
            "logical_model_bytes": int(winner["logical_model_bytes"]),
            "logical_model_gib": int(winner["logical_model_bytes"]) / 2**30,
            "size_reduction_vs_plain_quant": size_reduction,
            "directional_domain_macro": float(winner["domain_macro"]),
            "directional_question_weighted": float(winner["question_weighted"]),
            "full_agentic_candidate_total_solved_delta": int(
                full["candidate_total_solved_delta"]
            ),
            "measured_global_plan": winner_method,
            "interpretation": (
                "Winner of the preregistered directional, practical, trusted full capability, "
                "and complete SWE/Terminal chain among tested candidates; uncertainty remains "
                "descriptive rather than an equivalence claim."
            ),
        },
        "seven_format_candidates": plans,
        "private_damage_comparison": {
            "legacy_three_plan_lowest_damage_plan": damage["lowest_private_damage_plan"],
            "all_candidate_point_estimate_lowest_damage_plan": min(
                plans,
                key=lambda name: plans[name]["private_total_additive_damage"],
            ),
            "pairwise": damage["pairwise"],
        },
        "format_effects": {
            "format_totals": effects["format_totals"],
            "format_pairwise": effects["format_pairwise"],
            "equal_byte_pair_summary": effects["equal_byte_pair_summary"],
            "projection_damage": effects["projection_damage"],
            "layer_damage": effects["layer_damage"],
            "layer_projection_damage": effects["layer_projection_damage"],
            "error_concentration": effects["error_concentration"],
            "top_sensitive_experts": effects["top_sensitive_experts"][:50],
            "top_sensitive_functions": effects["top_sensitive_functions"][:20],
            "best_precision_upgrades": effects["best_precision_upgrades"][:20],
        },
        "format_efficiency": format_efficiency,
        "healing_ablation": healing_ablation,
        "directional": {
            **directional_frontier,
            "promotion": directional,
        },
        "practical_promotion": practical,
        "trusted_full": {
            "documents_per_arm": trusted["documents_per_arm"],
            "arms": trusted["arms"],
            "paired_vs_baseline": trusted["paired_vs_baseline"],
            "selection": trusted["selection"],
        },
        "full_agentic": full,
    }
    return result


def markdown(result: dict[str, Any]) -> str:
    winner = result["recommended_method"]
    lines = [
        "# Hy3 MoE prune/quant research conclusion",
        "",
        f"Recommended arm: **`{winner['arm']}`**",
        "",
        f"- Logical size: {winner['logical_model_bytes']:,} bytes "
        f"({winner['logical_model_gib']:.3f} GiB)",
        f"- Size reduction versus plain quant: {winner['size_reduction_vs_plain_quant']:.2%}",
        f"- Directional domain macro: {winner['directional_domain_macro']:.2%}",
        f"- Directional question-weighted score: {winner['directional_question_weighted']:.2%}",
        f"- Full SWE/Terminal solved delta versus plain: "
        f"{winner['full_agentic_candidate_total_solved_delta']:+d}",
        "",
        "## Seven-format candidates",
        "",
        "| Plan | Size (GB) | Pruned experts | Minimum survivors/layer | Private damage |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, item in result["seven_format_candidates"].items():
        lines.append(
            f"| {name} | {item['logical_model_bytes']/1e9:.3f} | "
            f"{item['pruned_experts']:,} | {item['minimum_survivors_in_layer']} | "
            f"{item['private_total_additive_damage']:.8g} |"
        )
    measured_plan = winner["measured_global_plan"]
    if measured_plan is not None:
        counts = ", ".join(
            f"{qtype}={count:,}"
            for qtype, count in sorted(measured_plan["qtype_projection_counts"].items())
        )
        lines += [
            "",
            "## Recommended measured allocation",
            "",
            f"- Retained/pruned experts: {measured_plan['retained_experts']:,} / "
            f"{measured_plan['pruned_experts']:,}",
            f"- Projection cells by format: {counts}",
            f"- Private additive damage: "
            f"{measured_plan['private_total_additive_damage']:.8g}",
        ]
    lines += [
        "",
        "## Private format quality per byte",
        "",
        "| Format | Full-bank size (GB) | Scaled quant damage | Point-estimate Pareto |",
        "|---|---:|---:|:---:|",
    ]
    for item in result["format_efficiency"]["formats"]:
        lines.append(
            f"| {item['qtype']} | {item['encoded_bytes']/1e9:.3f} | "
            f"{item['full_scaled_squared_error']:.8g} | "
            f"{'yes' if item['point_estimate_pareto'] else 'no'} |"
        )
    same_byte_winners = result["format_efficiency"]["same_byte_winners"]
    if same_byte_winners:
        lines.append("")
    for item in same_byte_winners:
        lines.append(
            f"- At identical bytes, `{item['winner']}` reduces measured damage by "
            f"{item['damage_reduction']:.2%} versus `{item['loser']}`."
        )
    healing = result["healing_ablation"]
    lines += [
        "",
        "## Matched healing ablation",
        "",
        "| Arm | Correct | Question-weighted | Domain macro |",
        "|---|---:|---:|---:|",
    ]
    for name in HEALING_ARMS:
        row = healing["arms"][name]
        lines.append(
            f"| {name} | {row['total_correct']}/{row['total_questions']} | "
            f"{row['question_weighted']:.2%} | {row['domain_macro']:.2%} |"
        )
    joint = healing["deltas_vs_unhealed"]["prune100_joint_heal"]
    router = healing["deltas_vs_unhealed"]["prune100_router_repair"]
    lines += [
        "",
        f"- Joint heal changes total correct by {joint['total_correct_delta']:+d} and "
        f"domain macro by {joint['domain_macro_delta']:+.2%} versus unhealed.",
        f"- Router-only repair changes total correct by {router['total_correct_delta']:+d} and "
        f"domain macro by {router['domain_macro_delta']:+.2%} versus unhealed.",
    ]
    random_control = healing["random_control_variance"]
    if random_control is not None:
        lines.append(
            "- Three matched random controls span "
            f"{random_control['question_weighted']['minimum']:.2%}–"
            f"{random_control['question_weighted']['maximum']:.2%} question-weighted "
            f"(population SD {random_control['question_weighted']['population_stddev']:.2%})."
        )
    lines += [
        "",
        "## Frozen directional comparison",
        "",
        "| Arm | Size (GB) | Question-weighted | Domain macro | Pareto |",
        "|---|---:|---:|---:|:---:|",
    ]
    for row in result["directional"]["arms"]:
        lines.append(
            f"| {row['arm']} | {row['logical_model_bytes']/1e9:.3f} | "
            f"{row['question_weighted']:.2%} | {row['domain_macro']:.2%} | "
            f"{'yes' if row['point_estimate_pareto'] else 'no'} |"
        )
    lines += [
        "",
        "The allocation/healing map used only frozen private calibration. Public tasks were used "
        "only after artifact construction through preregistered promotion gates.",
        "",
        result["recommended_method"]["interpretation"],
        "",
    ]
    return "\n".join(lines)


def parse_plan(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("plan must be NAME=PATH")
    name, raw = value.split("=", 1)
    if name not in PLAN_ARMS:
        raise argparse.ArgumentTypeError(f"plan name must be one of {sorted(PLAN_ARMS)}")
    return name, Path(raw)


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="bw24-hy3-conclusion-") as tmp:
        root = Path(tmp)
        paths: dict[str, Path] = {}

        def write(name: str, payload: dict[str, Any]) -> Path:
            path = root / name
            path.write_text(json.dumps(payload))
            paths[name] = path
            return path

        layer_summary = {"1": {"retained": 100, "pruned": 92}}
        plans = {}
        for name, arm, size in (
            ("uncentered", PLAN_ARMS["uncentered"], 100),
            ("centered", PLAN_ARMS["centered"], 99),
            ("pareto", PLAN_ARMS["pareto"], 98),
            ("layer_balanced", PLAN_ARMS["layer_balanced"], 97),
        ):
            selection = {"retained_experts": 100, "pruned_experts": 92}
            if name == "layer_balanced":
                selection.update({
                    "estimated_absolute_output_damage": 97.0,
                    "estimated_centered_output_damage": 96.0,
                    "estimated_objective": 0.97,
                })
            path = write(f"{name}.json", {
                "format": PLAN_FORMAT, "recipe": "measured-global-projection-budget",
                "policy": {"result_logical_bytes": size, "qtype_projection_counts": {"Q2_K": 300}},
                "calibration": {"public_eval_data_used_for_selection": False},
                "selection": selection,
                "layer_summary": layer_summary,
                "assignments": [{"layer": 1, "experts": list(range(100)),
                                 "projections": ["gate", "up", "down"],
                                 "qtype": "Q2_K"}],
            })
            plans[name] = (arm, path, size)
        damage_plans = {
            PLAN_DAMAGE_KEYS[name]: {"sha256": sha256(path), "logical_bytes": size,
                   "total_additive_damage": float(size), "prune_damage": 1.0,
                   "retained_quant_damage": float(size-1),
                   "projection_quant_damage": {"gate": 1.0, "up": 2.0, "down": 3.0},
                   "top_damage_cells": []}
            for name, (_, path, size) in plans.items()
            if name in PLAN_DAMAGE_KEYS
        }
        format_totals = {
            "Q2_K": {"encoded_bytes": 10, "full_scaled_squared_error": 100.0},
            "Q3_K": {"encoded_bytes": 20, "full_scaled_squared_error": 50.0},
            "IQ3_S": {"encoded_bytes": 20, "full_scaled_squared_error": 40.0},
            "IQ4_XS": {"encoded_bytes": 25, "full_scaled_squared_error": 20.0},
            "NVFP4": {"encoded_bytes": 30, "full_scaled_squared_error": 15.0},
            "Q4_K": {"encoded_bytes": 30, "full_scaled_squared_error": 10.0},
            "Q8_0": {"encoded_bytes": 60, "full_scaled_squared_error": 1.0},
        }
        effects = write("effects.json", {
            "format": "bw24-hy3-quant-effects-map-v1", "public_eval_data_used_for_selection": False,
            "measurement": {"qtypes": ["Q8_0","NVFP4","IQ4_XS","Q4_K","IQ3_S","Q3_K","Q2_K"]},
            "format_totals": format_totals, "format_pairwise": [], "equal_byte_pair_summary": [],
            "projection_damage": {}, "layer_damage": [], "layer_projection_damage": [],
            "error_concentration": {}, "top_sensitive_experts": [],
            "top_sensitive_functions": [], "best_precision_upgrades": [],
        })
        damage = write("damage.json", {"format":"bw24-hy3-quant-plan-damage-v1",
            "public_eval_data_used":False,"lowest_private_damage_plan":"pareto",
            "plans":damage_plans,"pairwise":{}})
        arm_rows = {"plain_quant":{"logical_model_bytes":200,"domain_macro":.8,"question_weighted":.8},
                    PLAN_ARMS["uncentered"]:{"logical_model_bytes":100,"domain_macro":.7,"question_weighted":.7},
                    PLAN_ARMS["centered"]:{"logical_model_bytes":99,"domain_macro":.71,"question_weighted":.71},
                    PLAN_ARMS["pareto"]:{"logical_model_bytes":98,"domain_macro":.72,"question_weighted":.72},
                    PLAN_ARMS["layer_balanced"]:{"logical_model_bytes":97,"domain_macro":.73,"question_weighted":.73}}
        frontier = write("frontier.json", {"format":"bw24-cross-run-expanded-capability-frontier-v1",
            "arms":arm_rows,"point_estimate_pareto":["plain_quant",PLAN_ARMS["pareto"]]})
        healing_frontier = write("healing-frontier.json", {
            "format":"bw24-cross-run-expanded-capability-frontier-v1",
            "arms": {
                "prune100_unhealed": {
                    "logical_model_bytes": 97, "total_correct": 66, "total_questions": 115,
                    "question_weighted": 66/115, "domain_macro": .54,
                    "tasks": {"code": {"n": 32, "successes": 27},
                              "math": {"n": 56, "successes": 26},
                              "history": {"n": 10, "successes": 2},
                              "other": {"n": 17, "successes": 11}},
                },
                "prune100_router_repair": {
                    "logical_model_bytes": 97, "total_correct": 65, "total_questions": 115,
                    "question_weighted": 65/115, "domain_macro": .542,
                    "tasks": {"code": {"n": 32, "successes": 29},
                              "math": {"n": 56, "successes": 22},
                              "history": {"n": 10, "successes": 1},
                              "other": {"n": 17, "successes": 13}},
                },
                "prune100_joint_heal": {
                    "logical_model_bytes": 97, "total_correct": 68, "total_questions": 115,
                    "question_weighted": 68/115, "domain_macro": .592,
                    "tasks": {"code": {"n": 32, "successes": 28},
                              "math": {"n": 56, "successes": 25},
                              "history": {"n": 10, "successes": 4},
                              "other": {"n": 17, "successes": 11}},
                },
            },
        })
        directional = write("directional.json", {"format":"bw24-smart100-directional-promotion-v1",
            "practical_arms":["plain_quant","traffic_nvfp4_53_q2_139",
                              PLAN_ARMS["pareto"],PLAN_ARMS["layer_balanced"]]})
        practical = write("practical.json", {"format":"bw24-practical-promotion-v1",
            "trusted_full_arms":["plain_quant",PLAN_ARMS["pareto"],
                                 PLAN_ARMS["layer_balanced"]]})
        trusted = write("trusted.json", {"format":"bw24-promoted-candidate-v1",
            "n_per_task":{"synthetic_full_suite":4746},"documents_per_arm":4746,
            "baseline":"plain_quant","arms":{},
            "paired_vs_baseline":{},
            "selection":{"selected_finalist":PLAN_ARMS["layer_balanced"]}})
        full = write("full.json", {"format":"bw24-full-agentic-comparison-v1",
            "baseline":"plain_quant","candidate":PLAN_ARMS["layer_balanced"],"total_tasks":589,
            "candidate_total_solved_delta":1})
        args = argparse.Namespace(effects=effects, damage=damage, frontier=frontier,
            healing_frontier=healing_frontier,
            directional_promotion=directional, practical_promotion=practical,
            trusted_report=trusted, full_agentic=full,
            plan=[(name,path) for name,(_,path,_) in plans.items()], analysis_commit="a"*40)
        result = build(args)
        assert result["recommended_method"]["arm"] == PLAN_ARMS["layer_balanced"]
        assert abs(result["recommended_method"]["size_reduction_vs_plain_quant"] - .515) < 1e-12
        assert result["seven_format_candidates"]["layer_balanced"][
            "private_damage_source"
        ] == "optimal_plan_selection_estimate"
        assert result["format_efficiency"]["point_estimate_pareto"] == [
            "Q2_K", "IQ3_S", "IQ4_XS", "Q4_K", "Q8_0"
        ]
        same_byte = result["format_efficiency"]["same_byte_winners"]
        assert [
            (item["encoded_bytes"], item["winner"], item["loser"])
            for item in same_byte
        ] == [(20, "IQ3_S", "Q3_K"), (30, "Q4_K", "NVFP4")]
        assert math.isclose(same_byte[0]["damage_reduction"], .2)
        assert math.isclose(same_byte[1]["damage_reduction"], 1 / 3)
        assert result["healing_ablation"]["best_question_weighted_arm"] == (
            "prune100_joint_heal"
        )
        assert result["healing_ablation"]["deltas_vs_unhealed"][
            "prune100_router_repair"
        ]["total_correct_delta"] == -1
        assert result["directional"]["arms"][0]["arm"] == (
            PLAN_ARMS["layer_balanced"]
        )
        assert "Recommended arm" in markdown(result)
        assert "Private format quality per byte" in markdown(result)
        assert "Matched healing ablation" in markdown(result)
        assert "Frozen directional comparison" in markdown(result)
        output = write("conclusion.json", result)
        rendered = root / "conclusion.md"
        rendered.write_text(markdown(result))
        receipt = write("receipt.json", {
            "format": RECEIPT_FORMAT,
            "analysis_commit": "a" * 40,
            "public_eval_data_used_for_allocation_or_healing": False,
            "inputs": [source(effects), source(damage)],
            "outputs": [source(output), source(rendered)],
            "script": source(Path(__file__).resolve()),
        })
        verify_receipt(receipt)
        output.write_text("{}")
        try:
            verify_receipt(receipt)
        except ValueError as error:
            assert "evidence mismatch" in str(error)
        else:
            raise AssertionError("receipt verifier accepted mutated evidence")


def main() -> None:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        print("Hy3 quant research conclusion self-test: PASS")
        return
    if len(sys.argv) == 3 and sys.argv[1] == "--verify-receipt":
        verify_receipt(Path(sys.argv[2]))
        print("Hy3 quant research conclusion receipt: PASS")
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--effects", type=Path, required=True)
    parser.add_argument("--damage", type=Path, required=True)
    parser.add_argument("--frontier", type=Path, required=True)
    parser.add_argument("--healing-frontier", type=Path, required=True)
    parser.add_argument("--directional-promotion", type=Path, required=True)
    parser.add_argument("--practical-promotion", type=Path, required=True)
    parser.add_argument("--trusted-report", type=Path, required=True)
    parser.add_argument("--full-agentic", type=Path, required=True)
    parser.add_argument("--plan", action="append", type=parse_plan, required=True)
    parser.add_argument("--analysis-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.analysis_commit):
        raise SystemExit("analysis commit must be a full Git SHA")
    for path in (args.output, args.markdown, args.receipt):
        if path.exists():
            raise SystemExit(f"refusing existing output {path}")
    result = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.markdown.write_text(markdown(result))
    inputs = [args.effects,args.damage,args.frontier,args.healing_frontier,
              args.directional_promotion,
              args.practical_promotion,args.trusted_report,args.full_agentic,
              *(path for _,path in args.plan)]
    receipt = {
        "format": RECEIPT_FORMAT,
        "analysis_commit": args.analysis_commit,
        "public_eval_data_used_for_allocation_or_healing": False,
        "inputs": [source(path) for path in inputs],
        "outputs": [source(args.output), source(args.markdown)],
        "script": source(Path(__file__).resolve()),
    }
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output} winner={result['recommended_method']['arm']}")


if __name__ == "__main__":
    main()
