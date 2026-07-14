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
    "layer_balanced120": "layer_balanced120",
    "layer_balanced137": "layer_balanced137",
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
TRAFFIC_ARM = "traffic_nvfp4_53_q2_139"


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


def remap_evidence_path(
    path: Path, path_maps: list[tuple[Path, Path]]
) -> Path:
    resolved = path.resolve()
    for source_root, destination_root in sorted(
        path_maps, key=lambda item: len(item[0].parts), reverse=True
    ):
        try:
            relative = resolved.relative_to(source_root.resolve())
        except ValueError:
            continue
        return destination_root.resolve() / relative
    return resolved


def verify_receipt(
    path: Path, path_maps: list[tuple[Path, Path]] | None = None
) -> None:
    path_maps = path_maps or []
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
        target = remap_evidence_path(Path(item["path"]), path_maps)
        if not target.is_file() or sha256(target) != item["sha256"]:
            raise ValueError(f"conclusion evidence mismatch: {target}")


def parse_path_map(value: str) -> tuple[Path, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("path map must be SOURCE=DESTINATION")
    source_root, destination_root = value.split("=", 1)
    if not source_root.startswith("/") or not destination_root.startswith("/"):
        raise argparse.ArgumentTypeError("path map roots must be absolute")
    return Path(source_root), Path(destination_root)


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


def effect_map_summary(effects: dict[str, Any]) -> dict[str, Any]:
    qtypes = ("Q2_K", "IQ3_S", "IQ4_XS", "Q4_K", "Q8_0")

    def metrics(values: dict[str, Any], context: str) -> dict[str, float]:
        result = {
            "weighted_mean": float(values["weighted_mean"]),
            "maximum": float(values["maximum"]),
        }
        if any(not math.isfinite(value) or value < 0 for value in result.values()):
            raise ValueError(f"invalid effect-map metric for {context}")
        return result

    projection_damage = effects.get("projection_damage")
    if not isinstance(projection_damage, dict) or set(projection_damage) != {
        "gate", "up", "down"
    }:
        raise ValueError("projection effect map is incomplete")
    projections = []
    for projection in ("gate", "up", "down"):
        values = projection_damage[projection]
        if not isinstance(values, dict) or any(qtype not in values for qtype in qtypes):
            raise ValueError(f"projection effect map lacks formats for {projection}")
        projections.append({
            "projection": projection,
            "qtypes": {
                qtype: metrics(values[qtype], f"{projection}/{qtype}")
                for qtype in qtypes
            },
        })

    layer_damage = effects.get("layer_damage")
    layer_projection_damage = effects.get("layer_projection_damage")
    if not isinstance(layer_damage, dict) or not layer_damage:
        raise ValueError("layer effect map is incomplete")
    if not isinstance(layer_projection_damage, dict) or set(layer_projection_damage) != set(
        layer_damage
    ):
        raise ValueError("layer-projection effect map does not match layer map")
    q2_layers = []
    q2_layer_projections = []
    for layer, values in layer_damage.items():
        layer_number = int(layer)
        if "Q2_K" not in values:
            raise ValueError(f"layer {layer} lacks Q2_K effects")
        q2_layers.append({
            "layer": layer_number,
            **metrics(values["Q2_K"], f"layer {layer}/Q2_K"),
        })
        cells = layer_projection_damage[layer]
        if not isinstance(cells, dict) or set(cells) != {"gate", "up", "down"}:
            raise ValueError(f"layer {layer} projection effects are incomplete")
        for projection, projection_values in cells.items():
            if "Q2_K" not in projection_values:
                raise ValueError(f"layer {layer}/{projection} lacks Q2_K effects")
            q2_layer_projections.append({
                "layer": layer_number,
                "projection": projection,
                **metrics(
                    projection_values["Q2_K"],
                    f"layer {layer}/{projection}/Q2_K",
                ),
            })
    q2_layers.sort(key=lambda row: (-row["weighted_mean"], row["layer"]))
    q2_layer_projections.sort(key=lambda row: (
        -row["weighted_mean"], row["layer"], row["projection"]
    ))

    experts = effects.get("top_sensitive_experts")
    functions = effects.get("top_sensitive_functions")
    upgrades = effects.get("best_precision_upgrades")
    if not all(isinstance(items, list) and items for items in (experts, functions, upgrades)):
        raise ValueError("expert/function/upgrade effect map is incomplete")
    top_experts = []
    for item in experts[:10]:
        row = {
            "layer": int(item["layer"]),
            "expert": int(item["expert"]),
            "routed_tokens": int(item["routed_tokens"]),
            "maximum_full_scaled_joint_squared_error": float(
                item["maximum_full_scaled_joint_squared_error"]
            ),
        }
        if row["routed_tokens"] < 0 or not math.isfinite(
            row["maximum_full_scaled_joint_squared_error"]
        ) or row["maximum_full_scaled_joint_squared_error"] < 0:
            raise ValueError("invalid sensitive expert")
        top_experts.append(row)
    q2_functions = []
    for item in functions:
        if item.get("qtype") != "Q2_K":
            continue
        row = {
            "layer": int(item["layer"]),
            "expert": int(item["expert"]),
            "projection": str(item["projection"]),
            "routed_tokens": int(item["routed_tokens"]),
            "normalized_mse": float(item["normalized_mse"]),
            "full_scaled_squared_error": float(item["full_scaled_squared_error"]),
        }
        if row["projection"] not in {"gate", "up", "down"} or any(
            not math.isfinite(row[key]) or row[key] < 0
            for key in ("normalized_mse", "full_scaled_squared_error")
        ):
            raise ValueError("invalid sensitive function")
        q2_functions.append(row)
    q2_functions.sort(key=lambda row: (
        -row["full_scaled_squared_error"], row["layer"], row["expert"],
        row["projection"],
    ))
    if not q2_functions:
        raise ValueError("effect map has no Q2_K sensitive functions")

    top_upgrades = []
    for item in upgrades[:10]:
        row = {
            "layer": int(item["layer"]),
            "expert": int(item["expert"]),
            "projection": str(item["projection"]),
            "from_qtype": str(item["from_qtype"]),
            "to_qtype": str(item["to_qtype"]),
            "extra_bytes": int(item["extra_bytes"]),
            "error_reduction": float(item["error_reduction"]),
            "error_reduction_per_gb": float(item["error_reduction_per_gb"]),
        }
        if row["extra_bytes"] <= 0 or any(
            not math.isfinite(row[key]) or row[key] < 0
            for key in ("error_reduction", "error_reduction_per_gb")
        ):
            raise ValueError("invalid precision upgrade")
        top_upgrades.append(row)

    return {
        "projection_sensitivity": projections,
        "most_q2_sensitive_layers": q2_layers[:10],
        "most_q2_sensitive_layer_projections": q2_layer_projections[:10],
        "most_sensitive_experts": top_experts,
        "most_q2_sensitive_functions": q2_functions[:10],
        "best_precision_upgrades": top_upgrades,
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


def practical_comparison_paths(practical: dict[str, Any]) -> list[Path]:
    """Return the immutable comparison evidence referenced by the practical verdict."""
    decisions = practical.get("decisions")
    if not isinstance(decisions, dict) or not decisions:
        raise ValueError("practical promotion has no candidate decisions")
    paths: list[Path] = []
    for candidate, decision in decisions.items():
        panels = decision.get("panels") if isinstance(decision, dict) else None
        if not isinstance(panels, dict) or set(panels) != {"swe", "terminal"}:
            raise ValueError(f"practical decision for {candidate} lacks both panels")
        for panel in ("swe", "terminal"):
            evidence = panels[panel].get("comparison")
            if not isinstance(evidence, dict):
                raise ValueError(f"practical {candidate}/{panel} lacks comparison evidence")
            path = Path(str(evidence.get("path", "")))
            expected_sha = str(evidence.get("sha256", ""))
            if not path.is_file() or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
                raise ValueError(f"practical {candidate}/{panel} comparison is absent")
            if sha256(path) != expected_sha:
                raise ValueError(f"practical {candidate}/{panel} comparison hash differs")
            paths.append(path)
    unique = list(dict.fromkeys(path.resolve() for path in paths))
    if len(unique) != len(paths):
        raise ValueError("practical promotion repeats comparison evidence")
    return unique


def practical_screen_summary(practical: dict[str, Any]) -> dict[str, Any]:
    """Validate and expose the normalized SWE/Terminal promotion evidence."""
    if practical.get("format") != "bw24-practical-promotion-v1":
        raise ValueError("wrong practical-promotion format")
    executed = practical.get("executed_practical_arms")
    trusted = practical.get("trusted_full_arms")
    promoted = practical.get("promoted_100gb_arms")
    if (
        not isinstance(executed, list) or len(executed) < 3
        or not isinstance(trusted, list) or len(trusted) < 2
        or not isinstance(promoted, list)
        or executed[0] != "plain_quant"
        or trusted[:2] != executed[:2]
        or any(arm not in trusted for arm in promoted)
    ):
        raise ValueError("practical promotion arm sets are inconsistent")
    strong = str(executed[1])
    decisions = practical["decisions"]
    rows: dict[str, Any] = {}
    for candidate, decision in decisions.items():
        if candidate not in executed[2:]:
            raise ValueError(f"practical decision has unexecuted candidate {candidate}")
        panels: dict[str, Any] = {}
        panel_passes: list[bool] = []
        for panel in ("swe", "terminal"):
            verdict = decision["panels"][panel]
            comparison_path = Path(verdict["comparison"]["path"])
            report = load(comparison_path)
            if (
                report.get("format") != "bw24-practical-comparison-v1"
                or report.get("panel") != panel
                or int(report.get("n_tasks", 0)) != 12
                or len(report.get("tasks", [])) != 12
                or report.get("baseline", {}).get("arm") != strong
                or report.get("candidate", {}).get("arm") != candidate
            ):
                raise ValueError(f"practical comparison contract differs for {candidate}/{panel}")
            totals: dict[str, dict[str, Any]] = {}
            for side in ("baseline", "candidate"):
                reward_key = f"{side}_reward"
                raw_key = f"{side}_raw_verifier_reward"
                timeout_key = f"{side}_timed_out"
                normalized = 0.0
                raw = 0.0
                timeout_count = 0
                override_count = 0
                for task in report["tasks"]:
                    reward = float(task[reward_key])
                    raw_reward = float(task[raw_key])
                    timed_out = task[timeout_key]
                    if (
                        not isinstance(timed_out, bool)
                        or not math.isfinite(reward) or not 0 <= reward <= 1
                        or not math.isfinite(raw_reward) or not 0 <= raw_reward <= 1
                        or reward != (0.0 if timed_out else raw_reward)
                    ):
                        raise ValueError(
                            f"invalid normalized practical reward for {candidate}/{panel}/{side}"
                        )
                    normalized += reward
                    raw += raw_reward
                    timeout_count += int(timed_out)
                    override_count += int(timed_out and raw_reward != 0.0)
                aggregate = report[side]
                if (
                    not math.isclose(float(aggregate["mean_reward"]), normalized / 12)
                    or not math.isclose(
                        float(aggregate["raw_verifier_mean_reward"]), raw / 12
                    )
                    or int(aggregate["timeout_count"]) != timeout_count
                    or int(aggregate["timeout_reward_override_count"]) != override_count
                ):
                    raise ValueError(
                        f"practical aggregate differs for {candidate}/{panel}/{side}"
                    )
                totals[side] = {
                    "arm": aggregate["arm"],
                    "solved": normalized,
                    "raw_verifier_solved": raw,
                    "timeout_count": timeout_count,
                    "timeout_reward_override_count": override_count,
                    "logical_model_bytes": int(aggregate["logical_model_bytes"]),
                }
            deficit = totals["baseline"]["solved"] - totals["candidate"]["solved"]
            passed = bool(verdict["passed"])
            if (
                not math.isclose(float(verdict["strong_compact_solved"]), totals["baseline"]["solved"])
                or not math.isclose(float(verdict["candidate_solved"]), totals["candidate"]["solved"])
                or not math.isclose(float(verdict["solved_deficit"]), deficit)
                or passed != (deficit <= 1.0)
            ):
                raise ValueError(f"practical verdict differs for {candidate}/{panel}")
            wins = int(report["paired_wins"])
            losses = int(report["paired_losses"])
            ties = int(report["paired_ties"])
            sign_p = float(report["exact_sign_p"])
            if (
                min(wins, losses, ties) < 0 or wins + losses + ties != 12
                or not math.isfinite(sign_p) or not 0 <= sign_p <= 1
            ):
                raise ValueError(f"practical paired evidence differs for {candidate}/{panel}")
            panel_passes.append(passed)
            panels[panel] = {
                "passed": passed,
                "strong_compact": totals["baseline"],
                "candidate": totals["candidate"],
                "solved_deficit": deficit,
                "paired_wins": wins,
                "paired_losses": losses,
                "paired_ties": ties,
                "exact_sign_p": sign_p,
                "comparison": source(comparison_path),
            }
        passed = all(panel_passes)
        if bool(decision["passed"]) != passed or ((candidate in promoted) != passed):
            raise ValueError(f"practical aggregate verdict differs for {candidate}")
        rows[str(candidate)] = {"passed": passed, "panels": panels}
    return {
        "reference_arms": executed[:2],
        "strong_compact_arm": strong,
        "executed_arms": executed,
        "promoted_100gb_arms": promoted,
        "trusted_full_arms": trusted,
        "candidates": rows,
        "reward_policy": (
            "AgentTimeoutError scores zero; late verifier rewards are retained only as raw provenance."
        ),
    }


def trusted_full_summary(trusted: dict[str, Any]) -> dict[str, Any]:
    arms = trusted.get("arms")
    baseline = trusted.get("baseline")
    paired = trusted.get("paired_vs_baseline")
    if not isinstance(arms, dict) or baseline not in arms or not isinstance(paired, dict):
        raise ValueError("trusted capability report lacks arm evidence")
    if set(paired) != set(arms) - {baseline}:
        raise ValueError("trusted paired comparisons do not match trusted arms")
    rows = []
    for arm, values in arms.items():
        size = int(values["logical_model_bytes"])
        macro = float(values["macro"])
        tasks = values.get("tasks")
        if size <= 0 or not math.isfinite(macro) or not 0 <= macro <= 1:
            raise ValueError(f"invalid trusted capability aggregate for {arm}")
        if not isinstance(tasks, dict) or not tasks:
            raise ValueError(f"trusted capability tasks are absent for {arm}")
        task_counts = [int(task["n"]) for task in tasks.values()]
        task_successes = [int(task["successes"]) for task in tasks.values()]
        if (
            sum(task_counts) != 4746
            or any(n <= 0 or successes < 0 or successes > n
                   for n, successes in zip(task_counts, task_successes))
        ):
            raise ValueError(f"trusted capability task count differs for {arm}")
        pair = paired.get(arm)
        if pair is not None:
            interval = pair.get("bootstrap_ci95")
            if (
                not isinstance(interval, list) or len(interval) != 2
                or any(not math.isfinite(float(value)) for value in interval)
                or not math.isfinite(float(pair["macro_delta"]))
                or not math.isclose(float(pair["macro_delta"]), macro - float(arms[baseline]["macro"]))
                or min(int(pair["paired_wins"]), int(pair["paired_losses"])) < 0
                or not 0 <= float(pair["exact_sign_p"]) <= 1
            ):
                raise ValueError(f"trusted paired evidence differs for {arm}")
        rows.append({
            "arm": arm,
            "logical_model_bytes": size,
            "macro": macro,
            "tasks": tasks,
            "paired_vs_baseline": pair,
        })
    selected = str(trusted.get("selection", {}).get("selected_finalist", ""))
    if selected not in arms or trusted.get("selection", {}).get("full_eval_arms") != [
        baseline, selected
    ]:
        raise ValueError("trusted capability selection is inconsistent")
    rows.sort(key=lambda row: (-row["macro"], row["logical_model_bytes"], row["arm"]))
    return {"baseline": baseline, "arms": rows, "selected_finalist": selected}


def full_agentic_summary(full: dict[str, Any]) -> dict[str, Any]:
    baseline = str(full.get("baseline", ""))
    candidate = str(full.get("candidate", ""))
    panels = []
    total_delta = 0.0
    for panel, expected_n in (("swe", 500), ("terminal", 89)):
        report = full.get(panel)
        if (
            not isinstance(report, dict)
            or report.get("format") != "bw24-full-practical-comparison-v1"
            or int(report.get("n_tasks", 0)) != expected_n
            or report.get("baseline", {}).get("arm") != baseline
            or report.get("candidate", {}).get("arm") != candidate
            or len(report.get("tasks", [])) != expected_n
        ):
            raise ValueError(f"full agentic {panel} report is incomplete")
        for side in ("baseline", "candidate"):
            values = report[side]
            solved = float(values["solved"])
            raw = float(values["raw_verifier_solved"])
            timed_out = int(values["timed_out"])
            overrides = int(values["timeout_reward_overrides"])
            if (
                not math.isfinite(solved) or not 0 <= solved <= expected_n
                or not math.isfinite(raw) or not 0 <= raw <= expected_n
                or not 0 <= overrides <= timed_out <= expected_n
            ):
                raise ValueError(f"full agentic {panel}/{side} aggregate is invalid")
        delta = float(report["candidate_solved_delta"])
        if not math.isclose(delta, float(report["candidate"]["solved"]) - float(
            report["baseline"]["solved"]
        )):
            raise ValueError(f"full agentic {panel} solved delta differs")
        wins = int(report["paired_wins"])
        losses = int(report["paired_losses"])
        ties = int(report["paired_ties"])
        if min(wins, losses, ties) < 0 or wins + losses + ties != expected_n:
            raise ValueError(f"full agentic {panel} paired evidence differs")
        total_delta += delta
        panels.append({
            "panel": panel,
            "n_tasks": expected_n,
            "baseline": report["baseline"],
            "candidate": report["candidate"],
            "candidate_solved_delta": delta,
            "paired_wins": wins,
            "paired_losses": losses,
            "paired_ties": ties,
        })
    if not math.isclose(total_delta, float(full["candidate_total_solved_delta"])):
        raise ValueError("full agentic combined solved delta differs")
    return {
        "baseline": baseline,
        "candidate": candidate,
        "total_tasks": 589,
        "candidate_total_solved_delta": total_delta,
        "panels": panels,
        "reward_policy": (
            "AgentTimeoutError scores zero; late verifier rewards are retained only as raw provenance."
        ),
    }


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


def traffic_plan_summary(
    path: Path, plan: dict[str, Any], frontier: dict[str, Any]
) -> dict[str, Any]:
    if plan.get("format") != PLAN_FORMAT or plan.get("recipe") != "traffic-ladder":
        raise ValueError("strong compact reference has unsupported plan format or recipe")
    if plan.get("calibration", {}).get("public_eval_data_used_for_selection") is not False:
        raise ValueError("strong compact reference does not attest private-only allocation")
    policy = plan.get("policy", {})
    expected_tiers = {"Q8_0": 0, "NVFP4": 53, "Q2_K": 139}
    if (
        policy.get("fixed_tier_counts") != expected_tiers
        or int(policy.get("fixed_prune_count", -1)) != 0
        or policy.get("prune_unused") is not False
    ):
        raise ValueError("strong compact reference is not the frozen 53-NVFP4/139-Q2 plan")
    model = plan.get("model", {})
    layers = [int(layer) for layer in model.get("moe_layers", [])]
    expert_count = int(model.get("expert_count", 0))
    if len(layers) != 79 or len(set(layers)) != 79 or expert_count != 192:
        raise ValueError("strong compact reference has the wrong model shape")

    by_layer: dict[int, dict[str, set[int]]] = {
        layer: {"NVFP4": set(), "Q2_K": set()} for layer in layers
    }
    for assignment in plan.get("assignments", []):
        layer = int(assignment["layer"])
        qtype = str(assignment["qtype"])
        if layer not in by_layer or qtype not in by_layer[layer]:
            raise ValueError("strong compact reference has an unexpected assignment")
        experts = {int(expert) for expert in assignment["experts"]}
        if len(experts) != len(assignment["experts"]):
            raise ValueError("strong compact reference repeats an expert")
        if by_layer[layer][qtype] & experts:
            raise ValueError("strong compact reference repeats an assignment")
        by_layer[layer][qtype].update(experts)

    all_experts = set(range(expert_count))
    for layer, qtypes in by_layer.items():
        if len(qtypes["NVFP4"]) != 53 or len(qtypes["Q2_K"]) != 139:
            raise ValueError(f"strong compact layer {layer} has the wrong tier counts")
        if qtypes["NVFP4"] & qtypes["Q2_K"] or set.union(*qtypes.values()) != all_experts:
            raise ValueError(f"strong compact layer {layer} does not cover the expert bank")
        summary = plan.get("layer_summary", {}).get(str(layer), {})
        if (
            int(summary.get("nvfp4", -1)) != 53
            or int(summary.get("q2_k", -1)) != 139
            or int(summary.get("pruned", -1)) != 0
        ):
            raise ValueError(f"strong compact layer {layer} summary differs")

    directional = frontier.get("arms", {}).get(TRAFFIC_ARM)
    if directional is None:
        raise ValueError("strong compact reference is absent from the directional frontier")
    logical_bytes = int(directional["logical_model_bytes"])
    per_projection = {"NVFP4": 53 * len(layers), "Q2_K": 139 * len(layers)}
    layer_projection = {"NVFP4": 53 * 3, "Q2_K": 139 * 3}
    return {
        "arm": TRAFFIC_ARM,
        "plan": source(path),
        "recipe": plan["recipe"],
        "description": plan.get("description"),
        "logical_model_bytes": logical_bytes,
        "logical_model_gib": logical_bytes / 2**30,
        "retained_experts": expert_count * len(layers),
        "pruned_experts": 0,
        "minimum_survivors_in_layer": expert_count,
        "maximum_pruned_in_layer": 0,
        "qtype_projection_counts": {
            qtype: count * 3 for qtype, count in per_projection.items()
        },
        "projection_qtype_counts": {
            projection: dict(per_projection) for projection in ("gate", "up", "down")
        },
        "layer_qtype_projection_counts": {
            str(layer): dict(layer_projection) for layer in layers
        },
        "layer_retention": {
            str(layer): {"retained": expert_count, "pruned": 0} for layer in layers
        },
        "private_damage_metric": None,
        "private_damage_source": "not_in_seven_format_plan_damage_study",
        "private_total_additive_damage": None,
        "private_centered_output_damage": None,
        "private_normalized_objective": None,
        "private_prune_damage": None,
        "private_retained_quant_damage": None,
        "private_projection_quant_damage": None,
        "private_top_damage_cells": None,
        "directional": {
            "domain_macro": float(directional["domain_macro"]),
            "question_weighted": float(directional["question_weighted"]),
            "tasks": directional.get("tasks", {}),
        },
    }


def effect_alignment_summary(
    plan: dict[str, Any], effect_summary: dict[str, Any]
) -> dict[str, Any]:
    assignments: dict[tuple[int, int, str], str] = {}
    for item in plan["assignments"]:
        layer = int(item["layer"])
        qtype = str(item["qtype"])
        for expert in item["experts"]:
            for projection in item["projections"]:
                key = (layer, int(expert), str(projection))
                if key in assignments:
                    raise ValueError(f"duplicate plan assignment for {key}")
                assignments[key] = qtype

    def assigned_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result = []
        for row in rows:
            key = (int(row["layer"]), int(row["expert"]), str(row["projection"]))
            result.append({**row, "assigned_qtype": assignments.get(key, "PRUNED")})
        return result

    sensitive_functions = assigned_rows(
        effect_summary["most_q2_sensitive_functions"]
    )
    precision_upgrades = assigned_rows(effect_summary["best_precision_upgrades"])

    def qtype_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in rows:
            qtype = row["assigned_qtype"]
            counts[qtype] = counts.get(qtype, 0) + 1
        return dict(sorted(counts.items()))

    layer_summary = plan["layer_summary"]
    sensitive_layers = {
        int(row["layer"]) for row in effect_summary["most_q2_sensitive_layers"]
    }
    all_layers = {int(layer) for layer in layer_summary}
    if not sensitive_layers or not sensitive_layers < all_layers:
        raise ValueError("sensitive-layer set cannot be contrasted with remaining layers")
    sensitive_retained = [
        int(layer_summary[str(layer)]["retained"]) for layer in sensitive_layers
    ]
    other_retained = [
        int(layer_summary[str(layer)]["retained"])
        for layer in all_layers - sensitive_layers
    ]
    sensitive_pruned = [
        int(layer_summary[str(layer)]["pruned"]) for layer in sensitive_layers
    ]
    other_pruned = [
        int(layer_summary[str(layer)]["pruned"])
        for layer in all_layers - sensitive_layers
    ]
    return {
        "most_q2_sensitive_function_assignments": sensitive_functions,
        "most_q2_sensitive_function_qtype_counts": qtype_counts(sensitive_functions),
        "best_precision_upgrade_assignments": precision_upgrades,
        "best_precision_upgrade_qtype_counts": qtype_counts(precision_upgrades),
        "top_sensitive_layer_count": len(sensitive_layers),
        "top_sensitive_layers_mean_retained": (
            math.fsum(sensitive_retained) / len(sensitive_retained)
        ),
        "other_layers_mean_retained": math.fsum(other_retained) / len(other_retained),
        "top_sensitive_layers_mean_pruned": (
            math.fsum(sensitive_pruned) / len(sensitive_pruned)
        ),
        "other_layers_mean_pruned": math.fsum(other_pruned) / len(other_pruned),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    effects = load(args.effects)
    damage = load(args.damage)
    frontier = load(args.frontier)
    healing_frontier = load(args.healing_frontier)
    directional = load(args.directional_promotion)
    practical = load(args.practical_promotion)
    trusted_selection_path = getattr(args, "trusted_selection", None)
    trusted_selection = (
        load(trusted_selection_path) if trusted_selection_path is not None else practical
    )
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
        (practical, "bw24-practical-promotion-v1"),
        (trusted, "bw24-promoted-candidate-v1"),
        (full, "bw24-full-agentic-comparison-v1"),
    )
    for payload, expected in expected_formats:
        if payload.get("format") != expected:
            raise ValueError(f"expected {expected}, got {payload.get('format')}")
    if directional.get("format") not in {
        "bw24-smart100-directional-promotion-v1",
        "bw24-layer-balanced-bridge-directional-promotion-v1",
    }:
        raise ValueError("wrong directional-promotion format")
    if trusted_selection.get("format") not in {
        "bw24-practical-promotion-v1",
        "bw24-effective-trusted-full-selection-v1",
    }:
        raise ValueError("wrong effective trusted-selection format")
    if trusted_selection.get("format") == "bw24-effective-trusted-full-selection-v1":
        if trusted_selection.get("base_trusted_full_arms") != practical.get(
            "trusted_full_arms"
        ):
            raise ValueError("effective trusted selection rewrote the frozen practical verdict")
        if trusted_selection.get("practical_promotion", {}).get("sha256") != sha256(
            args.practical_promotion
        ):
            raise ValueError("effective trusted selection is not bound to practical evidence")
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
    if (
        finalist not in trusted_selection["trusted_full_arms"]
        or finalist not in frontier["arms"]
    ):
        raise ValueError("finalist is absent from the promotion chain")

    plan_paths = {name: path for name, path in args.plan}
    if set(plan_paths) != set(PLAN_ARMS):
        raise ValueError(f"plans must be exactly {sorted(PLAN_ARMS)}")
    plans = {
        name: plan_summary(name, path, load(path), damage, frontier)
        for name, path in sorted(plan_paths.items())
    }
    traffic_method = traffic_plan_summary(
        args.traffic_plan, load(args.traffic_plan), frontier
    )
    baseline = frontier["arms"]["plain_quant"]
    winner = frontier["arms"][finalist]
    method_candidates = [traffic_method, *plans.values()]
    winner_method = next(
        (summary for summary in method_candidates if summary["arm"] == finalist), None
    )
    size_reduction = 1.0 - int(winner["logical_model_bytes"]) / int(
        baseline["logical_model_bytes"]
    )
    format_efficiency = format_efficiency_summary(effects)
    effect_summary = effect_map_summary(effects)
    for name in ("layer_balanced", "layer_balanced120", "layer_balanced137"):
        plans[name]["effect_alignment"] = effect_alignment_summary(
            load(plan_paths[name]), effect_summary
        )
    healing_ablation = healing_ablation_summary(healing_frontier)
    directional_frontier = directional_frontier_summary(frontier)
    practical_screen = practical_screen_summary(practical)
    trusted_summary = trusted_full_summary(trusted)
    full_summary = full_agentic_summary(full)
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
                "Winner after the frozen directional and practical evidence, the separately "
                "recorded trusted-coverage decision, trusted full capability, and complete "
                "SWE/Terminal chain among tested candidates; uncertainty remains descriptive "
                "rather than an equivalence claim."
            ),
        },
        "strong_compact_reference": traffic_method,
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
        "effect_map_summary": effect_summary,
        "healing_ablation": healing_ablation,
        "directional": {
            **directional_frontier,
            "promotion": directional,
        },
        "practical_promotion": practical,
        "practical_screen": practical_screen,
        "effective_trusted_selection": trusted_selection,
        "trusted_full": {
            "documents_per_arm": trusted["documents_per_arm"],
            "arms": trusted["arms"],
            "paired_vs_baseline": trusted["paired_vs_baseline"],
            "selection": trusted["selection"],
            "summary": trusted_summary,
        },
        "full_agentic": full,
        "full_agentic_summary": full_summary,
    }
    return result


def markdown(result: dict[str, Any]) -> str:
    winner = result["recommended_method"]
    traffic = result["strong_compact_reference"]
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
        "## Strong compact reference",
        "",
        f"- Arm: `{traffic['arm']}`",
        f"- Logical size: {traffic['logical_model_bytes']/1e9:.3f} GB",
        "- Allocation per MoE layer: 53 NVFP4 experts, 139 Q2_K experts, zero pruned",
        "- Ranking: private router-selection traffic with ascending expert-id tie break",
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
        ]
        private_damage = measured_plan["private_total_additive_damage"]
        if private_damage is not None:
            lines.append(f"- Private additive damage: {private_damage:.8g}")
        lines += [
            "",
            "### Projection allocation",
            "",
            "| Projection | Q8_0 | NVFP4 | Q4_K | IQ4_XS | IQ3_S | Q3_K | Q2_K |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for projection in ("gate", "up", "down"):
            values = measured_plan["projection_qtype_counts"].get(projection, {})
            lines.append(
                f"| {projection} | {values.get('Q8_0', 0):,} | "
                f"{values.get('NVFP4', 0):,} | {values.get('Q4_K', 0):,} | "
                f"{values.get('IQ4_XS', 0):,} | "
                f"{values.get('IQ3_S', 0):,} | {values.get('Q3_K', 0):,} | "
                f"{values.get('Q2_K', 0):,} |"
            )
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
    effect_map = result["effect_map_summary"]
    alignment = result["seven_format_candidates"]["layer_balanced"][
        "effect_alignment"
    ]
    lines += [
        "",
        "## Private layer, expert, and function effect map",
        "",
        "### Projection sensitivity",
        "",
        "| Projection | Q2_K mean | IQ3_S mean | IQ4_XS mean | Q4_K mean |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in effect_map["projection_sensitivity"]:
        values = row["qtypes"]
        lines.append(
            f"| {row['projection']} | {values['Q2_K']['weighted_mean']:.6g} | "
            f"{values['IQ3_S']['weighted_mean']:.6g} | "
            f"{values['IQ4_XS']['weighted_mean']:.6g} | "
            f"{values['Q4_K']['weighted_mean']:.6g} |"
        )
    lines += [
        "",
        "### Most Q2-sensitive layers",
        "",
        "| Layer | Weighted mean | Maximum |",
        "|---:|---:|---:|",
    ]
    for row in effect_map["most_q2_sensitive_layers"]:
        lines.append(
            f"| {row['layer']} | {row['weighted_mean']:.6g} | {row['maximum']:.6g} |"
        )
    lines += [
        "",
        f"The top {alignment['top_sensitive_layer_count']} Q2-sensitive layers retain "
        f"{alignment['top_sensitive_layers_mean_retained']:.1f} experts on average, versus "
        f"{alignment['other_layers_mean_retained']:.1f} across the remaining layers.",
    ]
    if measured_plan is not None:
        lines += [
            "",
            "### Recommended allocation at the most Q2-sensitive layers",
            "",
            "| Layer | Retained | Pruned | Q8_0 | NVFP4 | Q4_K | IQ4_XS | IQ3_S | Q3_K | Q2_K |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in effect_map["most_q2_sensitive_layers"]:
            layer = str(row["layer"])
            retention = measured_plan["layer_retention"][layer]
            values = measured_plan["layer_qtype_projection_counts"][layer]
            lines.append(
                f"| {layer} | {retention['retained']:,} | {retention['pruned']:,} | "
                f"{values.get('Q8_0', 0):,} | {values.get('NVFP4', 0):,} | "
                f"{values.get('Q4_K', 0):,} | "
                f"{values.get('IQ4_XS', 0):,} | {values.get('IQ3_S', 0):,} | "
                f"{values.get('Q3_K', 0):,} | {values.get('Q2_K', 0):,} |"
            )
    lines += [
        "",
        "### Most Q2-sensitive layer/projection cells",
        "",
        "| Layer | Projection | Weighted mean | Maximum |",
        "|---:|---|---:|---:|",
    ]
    for row in effect_map["most_q2_sensitive_layer_projections"]:
        lines.append(
            f"| {row['layer']} | {row['projection']} | "
            f"{row['weighted_mean']:.6g} | {row['maximum']:.6g} |"
        )
    lines += [
        "",
        "### Most Q2-sensitive expert functions",
        "",
        "| Layer | Expert | Projection | Assigned format | Routed tokens | Normalized MSE | Scaled error |",
        "|---:|---:|---|---|---:|---:|---:|",
    ]
    for row in alignment["most_q2_sensitive_function_assignments"]:
        lines.append(
            f"| {row['layer']} | {row['expert']} | {row['projection']} | "
            f"{row['assigned_qtype']} | "
            f"{row['routed_tokens']:,} | {row['normalized_mse']:.6g} | "
            f"{row['full_scaled_squared_error']:.6g} |"
        )
    lines += [
        "",
        "### Best measured precision upgrades",
        "",
        "| Layer | Expert | Projection | Suggested upgrade | Assigned format | Extra bytes | Error reduction/GB |",
        "|---:|---:|---|---|---|---:|---:|",
    ]
    for row in alignment["best_precision_upgrade_assignments"]:
        lines.append(
            f"| {row['layer']} | {row['expert']} | {row['projection']} | "
            f"{row['from_qtype']} -> {row['to_qtype']} | {row['assigned_qtype']} | "
            f"{row['extra_bytes']:,} | "
            f"{row['error_reduction_per_gb']:.6g} |"
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
    practical = result["practical_screen"]
    lines += [
        "",
        "## Matched practical promotion gate",
        "",
        "| Candidate | Panel | Strong compact solved | Candidate solved | Deficit | "
        "Timeouts strong/candidate | Late-reward overrides strong/candidate | W/L/T | Passed |",
        "|---|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for candidate, decision in practical["candidates"].items():
        for panel in ("swe", "terminal"):
            row = decision["panels"][panel]
            strong = row["strong_compact"]
            candidate_values = row["candidate"]
            lines.append(
                f"| {candidate} | {panel} | {strong['solved']:.0f}/12 | "
                f"{candidate_values['solved']:.0f}/12 | {row['solved_deficit']:+.0f} | "
                f"{strong['timeout_count']}/{candidate_values['timeout_count']} | "
                f"{strong['timeout_reward_override_count']}/"
                f"{candidate_values['timeout_reward_override_count']} | "
                f"{row['paired_wins']}/{row['paired_losses']}/{row['paired_ties']} | "
                f"{'yes' if row['passed'] else 'no'} |"
            )
    lines += [
        "",
        f"Trusted-full arms selected by the practical gate: "
        f"{', '.join(f'`{arm}`' for arm in practical['trusted_full_arms'])}.",
        practical["reward_policy"],
    ]
    trusted = result["trusted_full"]["summary"]
    lines += [
        "",
        "## Trusted full capability (4,746 documents per arm)",
        "",
        "| Arm | Logical size (GB) | Macro | Delta vs plain | Paired W/L | "
        "95% paired-bootstrap CI | Exact sign p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in trusted["arms"]:
        pair = row["paired_vs_baseline"]
        if pair is None:
            delta = wins = interval = sign_p = "—"
        else:
            delta = f"{float(pair['macro_delta']):+.2%}"
            wins = f"{int(pair['paired_wins'])}/{int(pair['paired_losses'])}"
            interval = (
                f"[{float(pair['bootstrap_ci95'][0]):+.2%}, "
                f"{float(pair['bootstrap_ci95'][1]):+.2%}]"
            )
            sign_p = f"{float(pair['exact_sign_p']):.4f}"
        lines.append(
            f"| {row['arm']} | {row['logical_model_bytes']/1e9:.3f} | "
            f"{row['macro']:.2%} | {delta} | {wins} | {interval} | {sign_p} |"
        )
    lines += [
        "",
        f"Trusted capability selected **`{trusted['selected_finalist']}`** for the complete "
        "SWE/Terminal comparison.",
    ]
    full = result["full_agentic_summary"]
    lines += [
        "",
        "## Complete SWE-Bench Verified and Terminal-Bench 2",
        "",
        "| Panel | Tasks | Plain solved | Candidate solved | Delta | "
        "Timeouts plain/candidate | Late-reward overrides plain/candidate | W/L/T |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in full["panels"]:
        baseline_values = row["baseline"]
        candidate_values = row["candidate"]
        lines.append(
            f"| {row['panel']} | {row['n_tasks']} | {baseline_values['solved']:.0f} | "
            f"{candidate_values['solved']:.0f} | {row['candidate_solved_delta']:+.0f} | "
            f"{baseline_values['timed_out']}/{candidate_values['timed_out']} | "
            f"{baseline_values['timeout_reward_overrides']}/"
            f"{candidate_values['timeout_reward_overrides']} | "
            f"{row['paired_wins']}/{row['paired_losses']}/{row['paired_ties']} |"
        )
    lines += [
        "",
        f"Across all {full['total_tasks']} agentic tasks, `{full['candidate']}` changed solved "
        f"count by {full['candidate_total_solved_delta']:+.0f} versus `{full['baseline']}`.",
        full["reward_policy"],
    ]
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

        layer_summary = {
            "1": {"retained": 100, "pruned": 92},
            "2": {"retained": 90, "pruned": 102},
        }
        plans = {}
        for name, arm, size in (
            ("uncentered", PLAN_ARMS["uncentered"], 100),
            ("centered", PLAN_ARMS["centered"], 99),
            ("pareto", PLAN_ARMS["pareto"], 98),
            ("layer_balanced", PLAN_ARMS["layer_balanced"], 97),
            ("layer_balanced120", PLAN_ARMS["layer_balanced120"], 120),
            ("layer_balanced137", PLAN_ARMS["layer_balanced137"], 137),
        ):
            selection = {"retained_experts": 190, "pruned_experts": 194}
            if name.startswith("layer_balanced"):
                selection.update({
                    "estimated_absolute_output_damage": 97.0,
                    "estimated_centered_output_damage": 96.0,
                    "estimated_objective": 0.97,
                })
            path = write(f"{name}.json", {
                "format": PLAN_FORMAT, "recipe": "measured-global-projection-budget",
                "policy": {"result_logical_bytes": size, "qtype_projection_counts": {"Q2_K": 570}},
                "calibration": {"public_eval_data_used_for_selection": False},
                "selection": selection,
                "layer_summary": layer_summary,
                "assignments": [
                    {"layer": layer, "experts": list(range(layer_summary[str(layer)]["retained"])),
                     "projections": ["gate", "up", "down"], "qtype": "Q2_K"}
                    for layer in (1, 2)
                ],
            })
            plans[name] = (arm, path, size)
        traffic_layers = list(range(1, 80))
        traffic_plan = write("traffic.json", {
            "format": PLAN_FORMAT,
            "recipe": "traffic-ladder",
            "description": "Usage-ranked full bank: top 0 Q8_0, next 53 NVFP4, "
                           "next 139 Q2_K, coldest 0 pruned",
            "model": {"expert_count": 192, "moe_layers": traffic_layers},
            "policy": {
                "fixed_tier_counts": {"Q8_0": 0, "NVFP4": 53, "Q2_K": 139},
                "fixed_prune_count": 0,
                "prune_unused": False,
            },
            "calibration": {"public_eval_data_used_for_selection": False},
            "layer_summary": {
                str(layer): {"nvfp4": 53, "q2_k": 139, "pruned": 0}
                for layer in traffic_layers
            },
            "assignments": [
                assignment
                for layer in traffic_layers
                for assignment in (
                    {"layer": layer, "qtype": "NVFP4", "experts": list(range(53))},
                    {"layer": layer, "qtype": "Q2_K", "experts": list(range(53, 192))},
                )
            ],
        })
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
        effect_qtypes = ("Q2_K", "IQ3_S", "IQ4_XS", "Q4_K", "Q8_0")
        projection_damage = {
            projection: {
                qtype: {"weighted_mean": float(index + 1), "maximum": float(index + 2)}
                for index, qtype in enumerate(effect_qtypes)
            }
            for projection in ("gate", "up", "down")
        }
        layer_damage = {"1": {
            "Q2_K": {"weighted_mean": 2.0, "maximum": 3.0},
        }}
        layer_projection_damage = {"1": {
            projection: {"Q2_K": {"weighted_mean": 1.0, "maximum": 2.0}}
            for projection in ("gate", "up", "down")
        }}
        effects = write("effects.json", {
            "format": "bw24-hy3-quant-effects-map-v1", "public_eval_data_used_for_selection": False,
            "measurement": {"qtypes": ["Q8_0","NVFP4","IQ4_XS","Q4_K","IQ3_S","Q3_K","Q2_K"]},
            "format_totals": format_totals, "format_pairwise": [], "equal_byte_pair_summary": [],
            "projection_damage": projection_damage, "layer_damage": layer_damage,
            "layer_projection_damage": layer_projection_damage,
            "error_concentration": {}, "top_sensitive_experts": [{
                "layer": 1, "expert": 2, "routed_tokens": 3,
                "maximum_full_scaled_joint_squared_error": 4.0,
            }],
            "top_sensitive_functions": [{
                "layer": 1, "expert": 2, "projection": "down", "qtype": "Q2_K",
                "routed_tokens": 3, "normalized_mse": .1,
                "full_scaled_squared_error": 4.0,
            }],
            "best_precision_upgrades": [{
                "layer": 1, "expert": 2, "projection": "down",
                "from_qtype": "Q2_K", "to_qtype": "IQ3_S", "extra_bytes": 5,
                "error_reduction": 3.0, "error_reduction_per_gb": 6.0,
            }],
        })
        damage = write("damage.json", {"format":"bw24-hy3-quant-plan-damage-v1",
            "public_eval_data_used":False,"lowest_private_damage_plan":"pareto",
            "plans":damage_plans,"pairwise":{}})
        arm_rows = {"plain_quant":{"logical_model_bytes":200,"domain_macro":.8,"question_weighted":.8},
                    TRAFFIC_ARM:{"logical_model_bytes":137,"domain_macro":.75,"question_weighted":.74},
                    PLAN_ARMS["uncentered"]:{"logical_model_bytes":100,"domain_macro":.7,"question_weighted":.7},
                    PLAN_ARMS["centered"]:{"logical_model_bytes":99,"domain_macro":.71,"question_weighted":.71},
                    PLAN_ARMS["pareto"]:{"logical_model_bytes":98,"domain_macro":.72,"question_weighted":.72},
                    PLAN_ARMS["layer_balanced"]:{"logical_model_bytes":97,"domain_macro":.73,"question_weighted":.73},
                    PLAN_ARMS["layer_balanced120"]:{"logical_model_bytes":120,"domain_macro":.74,"question_weighted":.74},
                    PLAN_ARMS["layer_balanced137"]:{"logical_model_bytes":137,"domain_macro":.75,"question_weighted":.75}}
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
        directional = write("directional.json", {
            "format":"bw24-layer-balanced-bridge-directional-promotion-v1",
            "practical_arms":["plain_quant","traffic_nvfp4_53_q2_139",
                              PLAN_ARMS["pareto"],PLAN_ARMS["layer_balanced"]]})
        practical_comparisons = {}
        for panel in ("swe", "terminal"):
            tasks = []
            for index in range(12):
                baseline_reward = float(index < 8)
                candidate_reward = float(index < 7)
                tasks.append({
                    "task": f"{panel}-{index}",
                    "baseline_reward": baseline_reward,
                    "candidate_reward": candidate_reward,
                    "baseline_raw_verifier_reward": baseline_reward,
                    "candidate_raw_verifier_reward": candidate_reward,
                    "baseline_timed_out": False,
                    "candidate_timed_out": False,
                })
            practical_comparisons[panel] = write(f"practical-{panel}.json", {
                "format": "bw24-practical-comparison-v1", "panel": panel,
                "n_tasks": 12,
                "baseline": {
                    "arm": TRAFFIC_ARM, "logical_model_bytes": 137,
                    "mean_reward": 8/12, "raw_verifier_mean_reward": 8/12,
                    "timeout_count": 0, "timeout_reward_override_count": 0,
                },
                "candidate": {
                    "arm": PLAN_ARMS["layer_balanced"], "logical_model_bytes": 97,
                    "mean_reward": 7/12, "raw_verifier_mean_reward": 7/12,
                    "timeout_count": 0, "timeout_reward_override_count": 0,
                },
                "candidate_mean_delta": -1/12, "candidate_size_reduction": 40/137,
                "paired_wins": 0, "paired_losses": 1, "paired_ties": 11,
                "exact_sign_p": 1.0, "tasks": tasks,
            })
        practical = write("practical.json", {
            "format":"bw24-practical-promotion-v1",
            "directional_practical_arms": [
                "plain_quant", TRAFFIC_ARM, PLAN_ARMS["layer_balanced"]
            ],
            "executed_practical_arms": [
                "plain_quant", TRAFFIC_ARM, PLAN_ARMS["layer_balanced"]
            ],
            "preregistered_100gb_arms": [],
            "confirmatory_100gb_arms": [PLAN_ARMS["layer_balanced"]],
            "decisions": {PLAN_ARMS["layer_balanced"]: {
                "passed": True,
                "panels": {
                    panel: {
                        "passed": True,
                        "strong_compact_solved": 8.0,
                        "candidate_solved": 7.0,
                        "solved_deficit": 1.0,
                        "comparison": {
                            "path": str(path.resolve()), "sha256": sha256(path)
                        },
                    }
                    for panel, path in practical_comparisons.items()
                },
            }},
            "promoted_100gb_arms": [PLAN_ARMS["layer_balanced"]],
            "trusted_full_arms": [
                "plain_quant", TRAFFIC_ARM, PLAN_ARMS["layer_balanced"]
            ],
        })
        trusted_selection = write("trusted-selection.json", {
            "format":"bw24-effective-trusted-full-selection-v1",
            "trusted_full_arms":["plain_quant", TRAFFIC_ARM,
                                 PLAN_ARMS["layer_balanced"]],
            "base_trusted_full_arms":["plain_quant", TRAFFIC_ARM,
                                      PLAN_ARMS["layer_balanced"]],
            "practical_promotion":{"path":str(practical.resolve()),
                                   "sha256":sha256(practical)},
            "decision":{"candidate_arm":PLAN_ARMS["layer_balanced"],
                        "qualified_by_user_policy":False,
                        "forced_into_trusted_full":False}})
        trusted_arms = {
            "plain_quant": {
                "logical_model_bytes": 200, "macro": .8,
                "tasks": {"synthetic_full_suite": {
                    "n": 4746, "successes": 3797, "rate": 3797/4746,
                }},
            },
            TRAFFIC_ARM: {
                "logical_model_bytes": 137, "macro": .75,
                "tasks": {"synthetic_full_suite": {
                    "n": 4746, "successes": 3560, "rate": 3560/4746,
                }},
            },
            PLAN_ARMS["layer_balanced"]: {
                "logical_model_bytes": 97, "macro": .76,
                "tasks": {"synthetic_full_suite": {
                    "n": 4746, "successes": 3607, "rate": 3607/4746,
                }},
            },
        }
        trusted_pairs = {
            arm: {
                "macro_delta": values["macro"] - trusted_arms["plain_quant"]["macro"],
                "bootstrap_ci95": [-.06, .02], "paired_wins": 100,
                "paired_losses": 120, "exact_sign_p": .2,
            }
            for arm, values in trusted_arms.items() if arm != "plain_quant"
        }
        trusted = write("trusted.json", {"format":"bw24-promoted-candidate-v1",
            "n_per_task":{"synthetic_full_suite":4746},"documents_per_arm":4746,
            "baseline":"plain_quant","arms":trusted_arms,
            "paired_vs_baseline":trusted_pairs,
            "selection":{"selected_finalist":PLAN_ARMS["layer_balanced"],
                         "full_eval_arms":["plain_quant",PLAN_ARMS["layer_balanced"]]}})

        def full_panel(
            panel: str, n: int, delta: int,
            candidate: str = PLAN_ARMS["layer_balanced"],
        ) -> dict[str, Any]:
            baseline_solved = n // 4
            candidate_solved = baseline_solved + delta
            return {
                "format": "bw24-full-practical-comparison-v1", "panel": panel,
                "n_tasks": n,
                "baseline": {"arm": "plain_quant", "solved": baseline_solved,
                             "raw_verifier_solved": baseline_solved,
                             "timed_out": 0, "timeout_reward_overrides": 0},
                "candidate": {"arm": candidate,
                              "solved": candidate_solved,
                              "raw_verifier_solved": candidate_solved,
                              "timed_out": 0, "timeout_reward_overrides": 0},
                "candidate_solved_delta": delta,
                "paired_wins": max(delta, 0), "paired_losses": max(-delta, 0),
                "paired_ties": n - abs(delta), "tasks": [{} for _ in range(n)],
            }

        full = write("full.json", {"format":"bw24-full-agentic-comparison-v1",
            "baseline":"plain_quant","candidate":PLAN_ARMS["layer_balanced"],"total_tasks":589,
            "swe": full_panel("swe", 500, 1),
            "terminal": full_panel("terminal", 89, 0),
            "candidate_total_solved_delta":1})
        args = argparse.Namespace(effects=effects, damage=damage, frontier=frontier,
            healing_frontier=healing_frontier,
            directional_promotion=directional, practical_promotion=practical,
            trusted_selection=trusted_selection,
            trusted_report=trusted, full_agentic=full,
            traffic_plan=traffic_plan,
            plan=[(name,path) for name,(_,path,_) in plans.items()], analysis_commit="a"*40)
        result = build(args)
        assert result["recommended_method"]["arm"] == PLAN_ARMS["layer_balanced"]
        assert not result["effective_trusted_selection"]["decision"][
            "forced_into_trusted_full"
        ]
        assert abs(result["recommended_method"]["size_reduction_vs_plain_quant"] - .515) < 1e-12
        assert result["seven_format_candidates"]["layer_balanced"][
            "private_damage_source"
        ] == "optimal_plan_selection_estimate"
        assert result["strong_compact_reference"]["qtype_projection_counts"] == {
            "NVFP4": 53 * 79 * 3,
            "Q2_K": 139 * 79 * 3,
        }
        assert result["strong_compact_reference"]["pruned_experts"] == 0
        traffic_trusted = write("traffic-trusted.json", {
            "format":"bw24-promoted-candidate-v1",
            "n_per_task":{"synthetic_full_suite":4746},"documents_per_arm":4746,
            "baseline":"plain_quant","arms":trusted_arms,
            "paired_vs_baseline":trusted_pairs,
            "selection":{"selected_finalist":TRAFFIC_ARM,
                         "full_eval_arms":["plain_quant", TRAFFIC_ARM]}})
        traffic_full = write("traffic-full.json", {
            "format":"bw24-full-agentic-comparison-v1",
            "baseline":"plain_quant","candidate":TRAFFIC_ARM,"total_tasks":589,
            "swe": full_panel("swe", 500, 0, TRAFFIC_ARM),
            "terminal": full_panel("terminal", 89, 0, TRAFFIC_ARM),
            "candidate_total_solved_delta":0})
        traffic_selection = write("traffic-selection.json", {
            "format":"bw24-effective-trusted-full-selection-v1",
            "trusted_full_arms":["plain_quant", TRAFFIC_ARM,
                                 PLAN_ARMS["layer_balanced"]],
            "base_trusted_full_arms":["plain_quant", TRAFFIC_ARM,
                                      PLAN_ARMS["layer_balanced"]],
            "practical_promotion":{"path":str(practical.resolve()),
                                   "sha256":sha256(practical)},
            "decision":{"candidate_arm":TRAFFIC_ARM,
                        "qualified_by_user_policy":False,
                        "forced_into_trusted_full":False}})
        traffic_args = argparse.Namespace(
            **{
                **vars(args),
                "trusted_selection": traffic_selection,
                "trusted_report": traffic_trusted,
                "full_agentic": traffic_full,
            }
        )
        traffic_result = build(traffic_args)
        assert traffic_result["recommended_method"]["measured_global_plan"][
            "arm"
        ] == TRAFFIC_ARM
        assert "53 NVFP4 experts, 139 Q2_K experts" in markdown(traffic_result)
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
        rendered_markdown = markdown(result)
        assert "Recommended arm" in rendered_markdown
        assert "Private format quality per byte" in rendered_markdown
        assert "Private layer, expert, and function effect map" in rendered_markdown
        assert result["effect_map_summary"]["most_q2_sensitive_layers"][0][
            "layer"
        ] == 1
        assert "Matched healing ablation" in rendered_markdown
        assert "Frozen directional comparison" in rendered_markdown
        assert "Matched practical promotion gate" in rendered_markdown
        assert "Trusted full capability (4,746 documents per arm)" in rendered_markdown
        assert "Complete SWE-Bench Verified and Terminal-Bench 2" in rendered_markdown
        assert len(practical_comparison_paths(load(practical))) == 2
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
        mirror = root / "archive"
        for original in (effects, damage, output, rendered):
            archived = mirror / original.relative_to(root)
            archived.parent.mkdir(parents=True, exist_ok=True)
            archived.write_bytes(original.read_bytes())
        output.write_text("{}")
        try:
            verify_receipt(receipt)
        except ValueError as error:
            assert "evidence mismatch" in str(error)
        else:
            raise AssertionError("receipt verifier accepted mutated evidence")
        verify_receipt(receipt, [(root, mirror)])


def main() -> None:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        print("Hy3 quant research conclusion self-test: PASS")
        return
    if sys.argv[1:2] == ["--verify-receipt"]:
        verifier = argparse.ArgumentParser()
        verifier.add_argument("--verify-receipt", type=Path, required=True)
        verifier.add_argument("--path-map", action="append", type=parse_path_map,
                              default=[])
        verify_args = verifier.parse_args()
        verify_receipt(verify_args.verify_receipt, verify_args.path_map)
        print("Hy3 quant research conclusion receipt: PASS")
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--effects", type=Path, required=True)
    parser.add_argument("--damage", type=Path, required=True)
    parser.add_argument("--frontier", type=Path, required=True)
    parser.add_argument("--healing-frontier", type=Path, required=True)
    parser.add_argument("--directional-promotion", type=Path, required=True)
    parser.add_argument("--practical-promotion", type=Path, required=True)
    parser.add_argument("--trusted-selection", type=Path)
    parser.add_argument("--trusted-report", type=Path, required=True)
    parser.add_argument("--full-agentic", type=Path, required=True)
    parser.add_argument("--traffic-plan", type=Path, required=True)
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
    practical_comparisons = practical_comparison_paths(load(args.practical_promotion))
    inputs = [args.effects,args.damage,args.frontier,args.healing_frontier,
              args.directional_promotion,
              args.practical_promotion,
              *practical_comparisons,
              *([args.trusted_selection] if args.trusted_selection else []),
              args.trusted_report,args.full_agentic,args.traffic_plan,
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
