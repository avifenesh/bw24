#!/usr/bin/env python3
"""Build a strict five-arm table for the N=1 directional candidate screen."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path
from typing import Any


ARMS = (
    "plain_quant",
    "plain_reap_quant",
    "plain_reap_mix_quant",
    "mix_quant",
    "mix_quant_prune25",
)


def candidate_specs(lock: dict[str, Any]) -> list[dict[str, str]]:
    tasks = lock["suites"]["candidate"]
    by_task: dict[str, list[dict[str, str]]] = {}
    for spec in lock["primary_metrics"]:
        by_task.setdefault(spec["result_task"], []).append(spec)
    specs = []
    for task in tasks:
        matches = by_task.get(task, [])
        if len(matches) != 1:
            raise ValueError(f"candidate task {task!r} has {len(matches)} primary metric specs")
        specs.append(matches[0])
    return specs


def numeric(value: Any, context: str) -> float:
    if not isinstance(value, (int, float, bool)):
        raise ValueError(f"{context}: expected numeric metric, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context}: metric is not finite")
    return result


def exactly_one(paths: list[Path], context: str) -> Path:
    if len(paths) != 1:
        rendered = ", ".join(str(path) for path in paths) or "none"
        raise ValueError(f"{context}: expected exactly one file, found {len(paths)} ({rendered})")
    return paths[0]


def load_arm(
    out_root: Path,
    run_id: str,
    arm: str,
    specs: list[dict[str, str]],
) -> dict[str, Any]:
    run_dir = out_root / arm / run_id
    if not run_dir.is_dir():
        raise ValueError(f"{arm}: missing run directory {run_dir}")

    manifest_path = run_dir / "artifact-manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"{arm}: missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    artifact_bytes = manifest.get("artifact_bytes")
    if isinstance(artifact_bytes, bool) or not isinstance(artifact_bytes, int) or artifact_bytes < 0:
        raise ValueError(f"{arm}: invalid artifact_bytes {artifact_bytes!r}")

    result_path = exactly_one(sorted(run_dir.rglob("results_*.json")), f"{arm} results")
    results = json.loads(result_path.read_text())
    task_rows: dict[str, Any] = {}
    for spec in specs:
        task = spec["result_task"]
        section = results.get(spec["result_section"], {})
        aggregate = section.get(task)
        if not isinstance(aggregate, dict):
            raise ValueError(f"{arm}/{task}: missing result section {spec['result_section']!r}")
        metric_key = f"{spec['metric']},{spec['filter']}"
        if metric_key not in aggregate:
            raise ValueError(f"{arm}/{task}: missing aggregate metric {metric_key!r}")
        aggregate_value = numeric(aggregate[metric_key], f"{arm}/{task} aggregate")

        sample_path = exactly_one(
            sorted(run_dir.rglob(spec["sample_glob"])),
            f"{arm}/{task} samples",
        )
        matching = []
        with sample_path.open() as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{sample_path}:{line_number}: invalid JSON") from exc
                if row.get("filter") == spec["filter"]:
                    matching.append(row)
        if len(matching) != 1:
            raise ValueError(
                f"{arm}/{task}: expected exactly one {spec['filter']!r} sample, "
                f"found {len(matching)} in {sample_path}"
            )
        sample = matching[0]
        sample_value = numeric(sample.get(spec["metric"]), f"{arm}/{task} sample")
        if not math.isclose(aggregate_value, sample_value, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"{arm}/{task}: N=1 aggregate {aggregate_value} differs from sample {sample_value}"
            )
        identity = (sample.get("doc_hash"), sample.get("target_hash"))
        if not all(isinstance(value, str) and value for value in identity):
            raise ValueError(f"{arm}/{task}: missing doc_hash or target_hash")
        task_rows[task] = {
            "value": aggregate_value,
            "doc_hash": identity[0],
            "target_hash": identity[1],
            "sample_file": str(sample_path),
        }

    return {
        "run_dir": str(run_dir),
        "result_file": str(result_path),
        "artifact_bytes": artifact_bytes,
        "tasks": task_rows,
    }


def build_report(out_root: Path, run_id: str, lock: dict[str, Any]) -> dict[str, Any]:
    specs = candidate_specs(lock)
    arms = {arm: load_arm(out_root, run_id, arm, specs) for arm in ARMS}
    for spec in specs:
        task = spec["result_task"]
        identities = {
            (arm_data["tasks"][task]["doc_hash"], arm_data["tasks"][task]["target_hash"])
            for arm_data in arms.values()
        }
        if len(identities) != 1:
            raise ValueError(f"{task}: sample identities differ across arms: {sorted(identities)!r}")
    return {
        "format": "bw24-directional-candidate-v1",
        "run_id": run_id,
        "n_per_task": 1,
        "confidence_intervals": False,
        "tasks": [
            {
                "label": spec["label"],
                "task": spec["result_task"],
                "metric": spec["metric"],
                "filter": spec["filter"],
            }
            for spec in specs
        ],
        "arms": arms,
    }


def markdown(report: dict[str, Any]) -> str:
    tasks = report["tasks"]
    lines = [
        "# Five-arm directional candidate results",
        "",
        f"Run ID: `{report['run_id']}`",
        "",
        "Each task has exactly N=1 matched sample. These are directional 0/1 checks; no confidence intervals are computed.",
        "",
        "| Arm | Artifact bytes | "
        + " | ".join(
            f"{task['label']} [{task['metric']}/{task['filter']}] (N=1)" for task in tasks
        )
        + " |",
        "|---|---:|" + "---:|" * len(tasks),
    ]
    for arm in ARMS:
        data = report["arms"][arm]
        cells = [arm, f"{data['artifact_bytes']:,}"]
        cells.extend(f"{data['tasks'][task['task']]['value']:.0f}" for task in tasks)
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "Artifact bytes are the exact expert-overlay payload recorded in each copied artifact manifest.",
    ]
    return "\n".join(lines) + "\n"


def write_fixture(root: Path, lock: dict[str, Any], run_id: str) -> None:
    specs = candidate_specs(lock)
    for arm_index, arm in enumerate(ARMS):
        run_dir = root / arm / run_id
        model_dir = run_dir / arm
        model_dir.mkdir(parents=True)
        (run_dir / "artifact-manifest.json").write_text(
            json.dumps({"artifact_bytes": 1000 + arm_index})
        )
        results: dict[str, dict[str, Any]] = {}
        for task_index, spec in enumerate(specs):
            value = float((arm_index + task_index) % 2)
            section = results.setdefault(spec["result_section"], {})
            section[spec["result_task"]] = {
                f"{spec['metric']},{spec['filter']}": value,
            }
            sample = {
                "filter": spec["filter"],
                spec["metric"]: value,
                "doc_hash": f"doc-{task_index}",
                "target_hash": f"target-{task_index}",
            }
            sample_path = model_dir / spec["sample_glob"].replace("*", "fixture")
            sample_path.write_text(json.dumps(sample) + "\n")
        (model_dir / "results_fixture.json").write_text(json.dumps(results))


def self_test(lock: dict[str, Any]) -> None:
    with tempfile.TemporaryDirectory(prefix="bw24-directional-summary-") as tmp:
        root = Path(tmp)
        run_id = "fixture"
        write_fixture(root, lock, run_id)
        report = build_report(root, run_id, lock)
        assert report["n_per_task"] == 1 and report["confidence_intervals"] is False
        assert report["arms"][ARMS[-1]]["artifact_bytes"] == 1004
        text = markdown(report)
        assert "no confidence intervals" in text and "(N=1)" in text

        specs = candidate_specs(lock)
        sample = root / ARMS[0] / run_id / ARMS[0] / specs[0]["sample_glob"].replace("*", "fixture")
        saved = sample.read_text()
        sample.unlink()
        try:
            build_report(root, run_id, lock)
        except ValueError as exc:
            assert "expected exactly one file, found 0" in str(exc)
        else:
            raise AssertionError("missing sample was accepted")
        sample.write_text(saved)

        duplicate = sample.with_name(sample.name.replace("fixture", "duplicate"))
        duplicate.write_text(saved)
        try:
            build_report(root, run_id, lock)
        except ValueError as exc:
            assert "expected exactly one file, found 2" in str(exc)
        else:
            raise AssertionError("duplicate sample was accepted")
        duplicate.unlink()

        sample.write_text(saved + saved)
        try:
            build_report(root, run_id, lock)
        except ValueError as exc:
            assert "expected exactly one" in str(exc) and "sample, found 2" in str(exc)
        else:
            raise AssertionError("duplicate sample row was accepted")
        sample.write_text(saved)

        other_sample = (
            root / ARMS[1] / run_id / ARMS[1] / specs[0]["sample_glob"].replace("*", "fixture")
        )
        row = json.loads(other_sample.read_text())
        row["doc_hash"] = "different-doc"
        other_sample.write_text(json.dumps(row) + "\n")
        try:
            build_report(root, run_id, lock)
        except ValueError as exc:
            assert "sample identities differ across arms" in str(exc)
        else:
            raise AssertionError("cross-arm sample mismatch was accepted")
        print("directional result summarizer self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--lock", type=Path, default=Path(__file__).with_name("suite.lock.json"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text())
    if args.self_test:
        self_test(lock)
        return 0
    if args.out_root is None or not args.run_id:
        parser.error("--out-root and --run-id are required")
    report = build_report(args.out_root, args.run_id, lock)
    out = args.out or args.out_root / "_runs" / args.run_id / "directional-results.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown(report))
    json_out = out.with_suffix(".json")
    json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out} and {json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
