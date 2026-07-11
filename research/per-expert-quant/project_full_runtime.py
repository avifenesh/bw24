#!/usr/bin/env python3
"""Project the pinned full candidate runtime from a completed bounded lm-eval run."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any


PROGRESS_RE = re.compile(
    r"Requesting API:.*?\|\s*(?P<count>\d+)/(?P<total>\d+)\s+"
    r"\[(?P<elapsed>[0-9:]+)<"
)
SELECTED_TASKS_RE = re.compile(r"Selected Tasks:\s*(?P<tasks>\[[^\n\r]+\])")


def elapsed_seconds(value: str) -> int:
    parts = [int(part) for part in value.split(":")]
    if not 1 <= len(parts) <= 3:
        raise ValueError(f"invalid elapsed time {value!r}")
    seconds = 0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def progress_snapshots(path: Path) -> tuple[int, dict[int, int]]:
    totals: set[int] = set()
    snapshots: dict[int, int] = {}
    for match in PROGRESS_RE.finditer(path.read_text(errors="replace")):
        count = int(match.group("count"))
        totals.add(int(match.group("total")))
        snapshots[count] = elapsed_seconds(match.group("elapsed"))
    if not snapshots:
        raise ValueError(f"no Requesting API progress found in {path}")
    if len(totals) != 1:
        raise ValueError(f"multiple progress totals found in {path}: {sorted(totals)}")
    return totals.pop(), snapshots


def selected_tasks(log_text: str) -> list[str] | None:
    match = SELECTED_TASKS_RE.search(log_text)
    if match is None:
        return None
    value = ast.literal_eval(match.group("tasks"))
    if not isinstance(value, list) or not all(isinstance(task, str) for task in value):
        raise ValueError(f"invalid Selected Tasks value: {value!r}")
    return value


def infer_limit(progress_total: int, tasks: list[str], full_counts: dict[str, int]) -> int:
    candidates = [
        limit for limit in range(1, max(int(full_counts[task]) for task in tasks) + 1)
        if sum(min(limit, int(full_counts[task])) for task in tasks) == progress_total
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"cannot uniquely infer bounded limit from progress total {progress_total}; "
            f"candidates={candidates[:10]}"
        )
    return candidates[0]


def build_report(run_dir: Path, lock_path: Path) -> dict[str, Any]:
    log_path = run_dir / "lm-eval.log"
    receipt_path = run_dir / "run-metadata.json"
    if not log_path.is_file() or not receipt_path.is_file():
        raise ValueError(f"run directory must contain lm-eval.log and run-metadata.json: {run_dir}")

    log_text = log_path.read_text(errors="replace")
    lock = json.loads(lock_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("suite") != "candidate":
        raise ValueError(f"expected candidate suite, got {receipt.get('suite')!r}")
    expected_order = lock["suites"]["candidate"]
    logged_tasks = selected_tasks(log_text)
    receipt_tasks = receipt.get("tasks")
    if receipt_tasks is not None and logged_tasks is not None and receipt_tasks != logged_tasks:
        raise ValueError(f"receipt and log task order differ: {receipt_tasks!r} != {logged_tasks!r}")
    tasks = receipt_tasks or logged_tasks
    if tasks != expected_order:
        raise ValueError(f"candidate task order differs from suite lock: {tasks!r}")
    full_counts = lock["eval_documents"]
    progress_total, snapshots = progress_snapshots(log_path)
    limit_raw = receipt.get("limit")
    if limit_raw is None:
        limit = infer_limit(progress_total, tasks, full_counts)
    elif isinstance(limit_raw, bool):
        raise ValueError(f"bounded integer limit required, got {limit_raw!r}")
    else:
        try:
            limit = int(limit_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"bounded integer limit required, got {limit_raw!r}") from exc
    if limit <= 0:
        raise ValueError(f"bounded integer limit required, got {limit}")

    measured_counts = {task: min(limit, int(full_counts[task])) for task in tasks}
    expected_total = sum(measured_counts.values())
    if progress_total != expected_total:
        raise ValueError(
            f"progress total {progress_total} differs from expected bounded total {expected_total}"
        )
    if expected_total not in snapshots:
        raise ValueError(
            f"run is incomplete: last progress count is {max(snapshots)}/{expected_total}"
        )

    rows = []
    cumulative = 0
    previous_elapsed = 0
    for task in tasks:
        measured = measured_counts[task]
        cumulative += measured
        if cumulative not in snapshots:
            raise ValueError(f"missing progress snapshot at task boundary {cumulative} for {task}")
        elapsed = snapshots[cumulative]
        task_seconds = elapsed - previous_elapsed
        if task_seconds <= 0:
            raise ValueError(f"non-positive measured duration for {task}: {task_seconds}")
        seconds_per_request = task_seconds / measured
        full_count = int(full_counts[task])
        projected_seconds = seconds_per_request * full_count
        rows.append(
            {
                "task": task,
                "measured_requests": measured,
                "measured_seconds": task_seconds,
                "seconds_per_request": seconds_per_request,
                "full_requests": full_count,
                "projected_full_seconds": projected_seconds,
            }
        )
        previous_elapsed = elapsed

    return {
        "format": "bw24-full-runtime-projection-v1",
        "source_run_dir": str(run_dir.resolve()),
        "source_log_sha256": sha256(log_path),
        "source_receipt_sha256": sha256(receipt_path),
        "source_limit": limit,
        "source_elapsed_seconds": snapshots[expected_total],
        "full_requests": sum(row["full_requests"] for row in rows),
        "projected_full_seconds": sum(row["projected_full_seconds"] for row in rows),
        "tasks": rows,
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Full candidate runtime projection",
        "",
        f"Source: `{report['source_run_dir']}` (LIMIT={report['source_limit']})",
        "",
        "| Task | Measured N | Seconds/request | Full N | Projected hours |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["tasks"]:
        lines.append(
            f"| {row['task']} | {row['measured_requests']} | "
            f"{row['seconds_per_request']:.2f} | {row['full_requests']} | "
            f"{row['projected_full_seconds'] / 3600:.2f} |"
        )
    lines.extend(
        [
            "",
            f"Total: **{report['full_requests']:,} requests**, "
            f"**{report['projected_full_seconds'] / 3600:.2f} hours per arm** at the measured settings.",
            "",
            "This is a wall-time projection from the bounded run, not a quality result. "
            "Recompute after changing concurrency or spill configuration.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_new(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x") as handle:
            handle.write(content)
    except FileExistsError as exc:
        raise ValueError(f"refusing to overwrite projection evidence: {path}") from exc


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="bw24-runtime-projection-") as tmp:
        root = Path(tmp)
        run_dir = root / "run"
        run_dir.mkdir()
        tasks = ["task_a", "task_b"]
        lock = {
            "suites": {"candidate": tasks},
            "eval_documents": {"task_a": 20, "task_b": 30},
        }
        lock_path = root / "suite.lock.json"
        lock_path.write_text(json.dumps(lock))
        (run_dir / "run-metadata.json").write_text(
            json.dumps({"suite": "candidate", "tasks": tasks, "limit": "10"})
        )
        lines = [
            f"Selected Tasks: {tasks!r}",
            "Requesting API:  50%|x| 10/20 [00:20<00:20, 2.00s/it]",
            "Requesting API: 100%|x| 20/20 [01:20<00:00, 6.00s/it]",
        ]
        (run_dir / "lm-eval.log").write_text("\r".join(lines))
        report = build_report(run_dir, lock_path)
        assert report["full_requests"] == 50
        assert report["tasks"][0]["projected_full_seconds"] == 40
        assert report["tasks"][1]["projected_full_seconds"] == 180
        assert report["projected_full_seconds"] == 220
        assert "0.06 hours per arm" in markdown(report)
        projection_path = root / "evidence" / "projection.json"
        write_new(projection_path, "evidence\n")
        try:
            write_new(projection_path, "replacement\n")
        except ValueError as exc:
            assert "refusing to overwrite" in str(exc)
        else:
            raise AssertionError("existing projection evidence was overwritten")
        (run_dir / "run-metadata.json").write_text(json.dumps({"suite": "candidate"}))
        legacy_report = build_report(run_dir, lock_path)
        assert legacy_report["source_limit"] == 10
        (run_dir / "lm-eval.log").write_text("\n".join(lines[:2]))
        try:
            build_report(run_dir, lock_path)
        except ValueError as exc:
            assert "incomplete" in str(exc)
        else:
            raise AssertionError("incomplete run was accepted")
    print("runtime projection self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--suite-lock",
        type=Path,
        default=Path(__file__).with_name("suite.lock.json"),
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.run_dir is None:
        parser.error("--run-dir is required")
    report = build_report(args.run_dir, args.suite_lock)
    rendered = markdown(report)
    if args.json_out:
        write_new(args.json_out, json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.markdown_out:
        write_new(args.markdown_out, rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
