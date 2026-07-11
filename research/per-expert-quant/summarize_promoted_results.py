#!/usr/bin/env python3
"""Summarize matched higher-N candidate results with uncertainty and paired tests."""

from __future__ import annotations

import argparse
import json
import math
import random
import tempfile
from pathlib import Path
from typing import Any

from summarize_directional_results import candidate_specs, exactly_one, numeric


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = successes / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - half, center + half


def exact_sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def bootstrap_delta(
    baseline: dict[str, dict[str, float]],
    candidate: dict[str, dict[str, float]],
    iterations: int = 5000,
) -> tuple[float, float]:
    rng = random.Random(20260711)
    tasks = sorted(baseline)
    samples = []
    for _ in range(iterations):
        task_means = []
        for task in tasks:
            ids = sorted(baseline[task])
            draws = [ids[rng.randrange(len(ids))] for _ in ids]
            task_means.append(sum(candidate[task][key] - baseline[task][key] for key in draws) / len(draws))
        samples.append(sum(task_means) / len(task_means))
    samples.sort()
    return samples[int(0.025 * iterations)], samples[int(0.975 * iterations)]


def load_arm(
    out_root: Path,
    run_id: str,
    arm: str,
    specs: list[dict[str, str]],
    expected_n: int,
) -> dict[str, Any]:
    run_dir = out_root / arm / run_id
    manifest_path = run_dir / "artifact-manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"{arm}: missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    result_path = exactly_one(sorted(run_dir.rglob("results_*.json")), f"{arm} results")
    results = json.loads(result_path.read_text())
    tasks = {}
    values_by_task: dict[str, dict[str, float]] = {}
    for spec in specs:
        task = spec["result_task"]
        metric_key = f"{spec['metric']},{spec['filter']}"
        aggregate = results.get(spec["result_section"], {}).get(task, {})
        aggregate_value = numeric(aggregate.get(metric_key), f"{arm}/{task} aggregate")
        sample_path = exactly_one(sorted(run_dir.rglob(spec["sample_glob"])), f"{arm}/{task} samples")
        values: dict[str, float] = {}
        with sample_path.open() as handle:
            for line_number, line in enumerate(handle, 1):
                row = json.loads(line)
                if row.get("filter") != spec["filter"]:
                    continue
                value = numeric(row.get(spec["metric"]), f"{sample_path}:{line_number}")
                if value not in (0.0, 1.0):
                    raise ValueError(f"{arm}/{task}: expected binary metric, got {value}")
                doc_hash, target_hash = row.get("doc_hash"), row.get("target_hash")
                if not isinstance(doc_hash, str) or not isinstance(target_hash, str):
                    raise ValueError(f"{arm}/{task}: missing sample identity")
                identity = f"{doc_hash}:{target_hash}"
                if identity in values:
                    raise ValueError(f"{arm}/{task}: duplicate sample identity {identity}")
                values[identity] = value
        if len(values) != expected_n:
            raise ValueError(f"{arm}/{task}: expected N={expected_n}, found {len(values)}")
        successes = int(sum(values.values()))
        rate = successes / expected_n
        if not math.isclose(rate, aggregate_value, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{arm}/{task}: aggregate {aggregate_value} != sample mean {rate}")
        low, high = wilson(successes, expected_n)
        tasks[task] = {"successes": successes, "n": expected_n, "rate": rate, "ci95": [low, high]}
        values_by_task[task] = values
    macro = sum(task["rate"] for task in tasks.values()) / len(tasks)
    return {
        "artifact_bytes": int(manifest["artifact_bytes"]),
        "result_file": str(result_path),
        "tasks": tasks,
        "macro": macro,
        "values": values_by_task,
    }


def build_report(
    out_root: Path,
    run_id: str,
    arms: list[str],
    baseline: str,
    expected_n: int,
    lock: dict[str, Any],
) -> dict[str, Any]:
    specs = candidate_specs(lock)
    loaded = {arm: load_arm(out_root, run_id, arm, specs, expected_n) for arm in arms}
    for task in (spec["result_task"] for spec in specs):
        identities = {arm: set(data["values"][task]) for arm, data in loaded.items()}
        first = identities[arms[0]]
        if any(value != first for value in identities.values()):
            raise ValueError(f"{task}: sample identities differ across arms")
    paired = {}
    base_values = loaded[baseline]["values"]
    for arm in arms:
        if arm == baseline:
            continue
        candidate = loaded[arm]["values"]
        wins = losses = 0
        for task in base_values:
            for identity, base in base_values[task].items():
                value = candidate[task][identity]
                wins += value > base
                losses += value < base
        delta = loaded[arm]["macro"] - loaded[baseline]["macro"]
        low, high = bootstrap_delta(base_values, candidate)
        paired[arm] = {
            "macro_delta": delta,
            "bootstrap_ci95": [low, high],
            "paired_wins": wins,
            "paired_losses": losses,
            "exact_sign_p": exact_sign_p(wins, losses),
        }
    for data in loaded.values():
        data.pop("values")
    return {
        "format": "bw24-promoted-candidate-v1",
        "run_id": run_id,
        "n_per_task": expected_n,
        "baseline": baseline,
        "arms": loaded,
        "paired_vs_baseline": paired,
        "tasks": [{"task": spec["result_task"], "label": spec["label"]} for spec in specs],
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Promoted-arm matched evaluation",
        "",
        f"Run ID: `{report['run_id']}` · N={report['n_per_task']} per task · baseline `{report['baseline']}`",
        "",
        "| Arm | Expert overlay bytes | Macro accuracy | Delta vs baseline | Paired W/L | 95% paired-bootstrap CI | Exact sign p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, data in report["arms"].items():
        pair = report["paired_vs_baseline"].get(arm)
        if pair is None:
            cells = [arm, f"{data['artifact_bytes']:,}", f"{data['macro']:.1%}", "—", "—", "—", "—"]
        else:
            lo, hi = pair["bootstrap_ci95"]
            cells = [
                arm, f"{data['artifact_bytes']:,}", f"{data['macro']:.1%}",
                f"{pair['macro_delta']:+.1%}", f"{pair['paired_wins']}/{pair['paired_losses']}",
                f"[{lo:+.1%}, {hi:+.1%}]", f"{pair['exact_sign_p']:.4f}",
            ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Per-task accuracy (Wilson 95% CI)", ""]
    for task in report["tasks"]:
        lines += [f"### {task['label']}", "", "| Arm | Correct/N | Accuracy | Wilson 95% CI |", "|---|---:|---:|---:|"]
        for arm, data in report["arms"].items():
            value = data["tasks"][task["task"]]
            lines.append(
                f"| {arm} | {value['successes']}/{value['n']} | {value['rate']:.1%} | "
                f"[{value['ci95'][0]:.1%}, {value['ci95'][1]:.1%}] |"
            )
        lines.append("")
    lines.append("All comparisons use identical sample hashes across arms. The paired bootstrap is stratified by task.")
    return "\n".join(lines) + "\n"


def self_test(lock: dict[str, Any]) -> None:
    specs = candidate_specs(lock)
    with tempfile.TemporaryDirectory(prefix="bw24-promoted-summary-") as tmp:
        root = Path(tmp)
        arms = ["plain_quant", "candidate"]
        for arm_index, arm in enumerate(arms):
            run_dir = root / arm / "fixture"
            model_dir = run_dir / arm
            model_dir.mkdir(parents=True)
            (run_dir / "artifact-manifest.json").write_text(json.dumps({"artifact_bytes": 100 + arm_index}))
            results = {}
            for task_index, spec in enumerate(specs):
                rows = []
                for i in range(4):
                    value = float((i + task_index + arm_index) % 3 == 0)
                    rows.append({
                        "filter": spec["filter"], spec["metric"]: value,
                        "doc_hash": f"doc-{task_index}-{i}", "target_hash": f"target-{task_index}-{i}",
                    })
                results.setdefault(spec["result_section"], {})[spec["result_task"]] = {
                    f"{spec['metric']},{spec['filter']}": sum(row[spec["metric"]] for row in rows) / 4,
                }
                sample = model_dir / spec["sample_glob"].replace("*", "fixture")
                sample.write_text("".join(json.dumps(row) + "\n" for row in rows))
            (model_dir / "results_fixture.json").write_text(json.dumps(results))
        report = build_report(root, "fixture", arms, "plain_quant", 4, lock)
        assert report["n_per_task"] == 4
        assert report["paired_vs_baseline"]["candidate"]["paired_wins"] > 0
        assert "Wilson 95% CI" in markdown(report)
        print("promoted result summarizer self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--arms", help="comma-separated arm names")
    parser.add_argument("--baseline", default="plain_quant")
    parser.add_argument("--expected-n", type=int, default=50)
    parser.add_argument("--lock", type=Path, default=Path(__file__).with_name("suite.lock.json"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text())
    if args.self_test:
        self_test(lock)
        return 0
    if args.out_root is None or not args.run_id or not args.arms:
        parser.error("--out-root, --run-id, and --arms are required")
    arms = [arm for arm in args.arms.split(",") if arm]
    if args.baseline not in arms:
        parser.error("--baseline must be present in --arms")
    report = build_report(args.out_root, args.run_id, arms, args.baseline, args.expected_n, lock)
    out = args.out or args.out_root / "_runs" / args.run_id / "promoted-results.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown(report))
    out.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out} and {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
