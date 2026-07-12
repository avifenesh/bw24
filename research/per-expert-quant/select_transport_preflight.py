#!/usr/bin/env python3
"""Validate matched transport preflights and select the fastest safe setting."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tempfile
from pathlib import Path
from typing import Any

from compare_eval_generations import compare, load_samples


SPILL_KEYS = ("reads", "bytes", "errors", "short_reads", "fallbacks", "buffer_waits", "ring_full")
LOG_FAILURE_RE = re.compile(
    r"Traceback|\bRetrying\b|\bHTTP\s+[45][0-9]{2}\b|\bpanic(?:ked)?\b|\bfatal\b|"
    r"CUDA error|server error|error:",
    re.IGNORECASE,
)


def exactly_one(paths: list[Path], label: str) -> Path:
    if len(paths) != 1:
        raise ValueError(f"{label}: expected exactly one file, found {len(paths)}")
    return paths[0]


def positive_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}: expected numeric value, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label}: expected positive finite value, got {value!r}")
    return number


def integer_setting(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label}: invalid integer {value!r}")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: invalid integer {value!r}") from exc
    if number <= 0 or str(number) != str(value):
        raise ValueError(f"{label}: invalid positive integer {value!r}")
    return number


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_set_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def validate_preflight(
    run_dir: Path,
    baseline_dir: Path,
    expected_tasks: list[str],
    expected_limit: int,
    dimension: str,
) -> dict[str, Any]:
    receipt_path = exactly_one(sorted(run_dir.rglob("run-metadata.json")), f"{run_dir} receipt")
    log_path = exactly_one(sorted(run_dir.rglob("lm-eval.log")), f"{run_dir} evaluator log")
    exactly_one(sorted(run_dir.rglob("results_*.json")), f"{run_dir} result")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("suite") != "candidate" or receipt.get("tasks") != expected_tasks:
        raise ValueError(f"{run_dir}: suite or ordered tasks differ from the pinned candidate suite")
    if str(receipt.get("limit")) != str(expected_limit):
        raise ValueError(f"{run_dir}: expected limit={expected_limit}, got {receipt.get('limit')!r}")
    if receipt.get("max_gen_toks_override") != 256:
        raise ValueError(f"{run_dir}: expected max_gen_toks_override=256")
    elapsed = positive_number(receipt.get("elapsed_seconds"), f"{run_dir} elapsed_seconds")
    if (
        receipt.get("completed_successfully") is not True
        or receipt.get("evaluator_exit_code") != 0
        or receipt.get("tee_exit_code") != 0
        or not isinstance(receipt.get("started_utc"), str)
        or not isinstance(receipt.get("completed_utc"), str)
    ):
        raise ValueError(f"{run_dir}: receipt is not a successful timed completion")
    if (
        receipt.get("declared_spill_io") != "worker"
        or str(receipt.get("declared_spill_stats")) != "1"
        or str(receipt.get("declared_serve_spec")) != "0"
    ):
        raise ValueError(f"{run_dir}: transport declarations differ from worker/stats-on/spec-off")
    depth = integer_setting(receipt.get("declared_spill_pread_depth"), f"{run_dir} spill depth")
    concurrency = integer_setting(receipt.get("num_concurrent"), f"{run_dir} concurrency")
    if dimension == "spill_depth" and concurrency != 1:
        raise ValueError(f"{run_dir}: spill-depth preflight must use concurrency 1")
    setting = depth if dimension == "spill_depth" else concurrency

    spill = receipt.get("spill_delta")
    if not isinstance(spill, dict) or set(spill) != set(SPILL_KEYS):
        raise ValueError(f"{run_dir}: invalid spill delta")
    if any(isinstance(spill[key], bool) or not isinstance(spill[key], int) or spill[key] < 0 for key in SPILL_KEYS):
        raise ValueError(f"{run_dir}: non-monotonic spill delta")
    if spill["reads"] <= 0 or spill["bytes"] <= 0:
        raise ValueError(f"{run_dir}: no spill reads recorded")
    if spill["errors"] != 0 or spill["short_reads"] != 0:
        raise ValueError(f"{run_dir}: spill I/O failure recorded")

    samples = load_samples(run_dir)
    expected_samples = expected_limit * len(expected_tasks)
    if len(samples) != expected_samples:
        raise ValueError(f"{run_dir}: expected {expected_samples} samples, found {len(samples)}")
    compare(baseline_dir, run_dir, candidate_subset=True)

    log_failures = []
    server_log = Path(receipt.get("server_log") or "")
    for source in (log_path, server_log):
        if not source.is_file():
            raise ValueError(f"{run_dir}: missing declared log {source}")
        for line_number, line in enumerate(source.read_text(errors="replace").splitlines(), 1):
            if LOG_FAILURE_RE.search(line):
                log_failures.append(f"{source}:{line_number}: {line[:300]}")
                if len(log_failures) == 5:
                    break
    if log_failures:
        raise ValueError(f"{run_dir}: retry/error evidence found: {log_failures}")

    return {
        "run_dir": str(run_dir.resolve()),
        "setting": setting,
        "spill_depth": depth,
        "num_concurrent": concurrency,
        "elapsed_seconds": elapsed,
        "sample_count": len(samples),
        "spill_delta": spill,
        "receipt_sha256": sha256(receipt_path),
        "evaluator_log_sha256": sha256(log_path),
        "server_log_sha256": sha256(server_log),
        "sample_logs_sha256": file_set_sha256(sorted(run_dir.rglob("samples_*.jsonl"))),
    }


def build_report(
    baseline_dir: Path,
    candidates: list[tuple[str, Path]],
    lock_path: Path,
    expected_limit: int,
    dimension: str,
    safe_setting: int,
    min_improvement: float,
) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text())
    expected_tasks = lock["suites"]["candidate"]
    rows = []
    seen_settings: set[int] = set()
    for label, run_dir in candidates:
        try:
            row = validate_preflight(run_dir, baseline_dir, expected_tasks, expected_limit, dimension)
            setting = row["setting"]
            if setting in seen_settings:
                raise ValueError(f"duplicate {dimension} setting {setting}")
            seen_settings.add(setting)
            row.update({"label": label, "passed": True, "rejection": None})
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            row = {
                "label": label,
                "run_dir": str(run_dir.resolve()),
                "passed": False,
                "rejection": str(exc),
            }
        rows.append(row)
    passing = [row for row in rows if row["passed"]]
    if dimension == "num_concurrent" and len({row["spill_depth"] for row in passing}) != 1:
        raise ValueError(f"concurrency preflights use different spill depths; rows={rows}")
    safe = next((row for row in passing if row["setting"] == safe_setting), None)
    if safe is None:
        raise ValueError(f"safe {dimension}={safe_setting} did not pass; rows={rows}")
    fastest = min(passing, key=lambda row: row["elapsed_seconds"])
    improvement = 1.0 - fastest["elapsed_seconds"] / safe["elapsed_seconds"]
    selected = fastest if improvement >= min_improvement else safe
    return {
        "format": "bw24-transport-preflight-v1",
        "baseline_run_dir": str(baseline_dir.resolve()),
        "baseline_samples_sha256": file_set_sha256(sorted(baseline_dir.rglob("samples_*.jsonl"))),
        "dimension": dimension,
        "expected_limit": expected_limit,
        "expected_samples": expected_limit * len(expected_tasks),
        "safe_setting": safe_setting,
        "minimum_improvement": min_improvement,
        "fastest_passing_setting": fastest["setting"],
        "fastest_improvement_vs_safe": improvement,
        "selected_setting": selected["setting"],
        "selection_reason": (
            "fastest passing setting cleared minimum improvement"
            if selected is fastest and fastest is not safe
            else "retained safe setting; improvement threshold was not cleared"
        ),
        "candidates": rows,
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# {report['dimension']} preflight selection",
        "",
        "| Candidate | Setting | Elapsed seconds | Samples | Result |",
        "|---|---:|---:|---:|---|",
    ]
    for row in report["candidates"]:
        if row["passed"]:
            lines.append(
                f"| {row['label']} | {row['setting']} | {row['elapsed_seconds']:.3f} | "
                f"{row['sample_count']} | PASS |"
            )
        else:
            lines.append(f"| {row['label']} | - | - | - | REJECT: {row['rejection']} |")
    lines.extend(
        [
            "",
            f"Selected **{report['dimension']}={report['selected_setting']}**: "
            f"{report['selection_reason']}.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_new(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x") as handle:
            handle.write(content)
    except FileExistsError as exc:
        raise ValueError(f"refusing to overwrite preflight evidence: {path}") from exc


def fixture_run(root: Path, name: str, depth: int, elapsed: float, mismatch: bool = False) -> Path:
    run = root / name
    run.mkdir()
    server_log = run / "server.log"
    server_log.write_text("healthy\n")
    (run / "lm-eval.log").write_text("complete\n")
    (run / "results_fixture.json").write_text("{}\n")
    for task in ("task_a", "task_b"):
        row = {
            "doc_hash": task, "prompt_hash": "prompt", "target_hash": "target",
            "filter": "fixture", "resps": [["bad" if mismatch else "ok"]],
            "filtered_resps": ["ok"],
        }
        (run / f"samples_{task}_fixture.jsonl").write_text(json.dumps(row) + "\n")
    receipt = {
        "suite": "candidate", "tasks": ["task_a", "task_b"], "limit": "1",
        "max_gen_toks_override": 256, "elapsed_seconds": elapsed,
        "completed_successfully": True, "evaluator_exit_code": 0, "tee_exit_code": 0,
        "started_utc": "start", "completed_utc": "end", "declared_spill_io": "worker",
        "declared_spill_stats": "1", "declared_serve_spec": "0",
        "declared_spill_pread_depth": str(depth), "num_concurrent": 1,
        "spill_delta": {key: (100 if key in ("reads", "bytes") else 0) for key in SPILL_KEYS},
        "server_log": str(server_log),
    }
    (run / "run-metadata.json").write_text(json.dumps(receipt))
    return run


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="bw24-transport-preflight-") as tmp:
        root = Path(tmp)
        lock = root / "suite.lock.json"
        lock.write_text(json.dumps({"suites": {"candidate": ["task_a", "task_b"]}}))
        baseline = fixture_run(root, "baseline", 8, 100)
        depth8 = fixture_run(root, "depth8", 8, 100)
        depth16 = fixture_run(root, "depth16", 16, 80)
        depth32 = fixture_run(root, "depth32", 32, 70, mismatch=True)
        report = build_report(
            baseline, [("d8", depth8), ("d16", depth16), ("d32", depth32)],
            lock, 1, "spill_depth", 8, 0.05,
        )
        assert report["selected_setting"] == 16
        assert report["candidates"][2]["passed"] is False
        assert "generation mismatches" in report["candidates"][2]["rejection"]
        assert "Selected **spill_depth=16**" in markdown(report)
    print("transport preflight selector self-test: PASS")


def parse_candidate(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("candidate must be LABEL=RUN_DIR")
    return label, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", action="append", type=parse_candidate, default=[])
    parser.add_argument("--suite-lock", type=Path, default=Path(__file__).with_name("suite.lock.json"))
    parser.add_argument("--expected-limit", type=int)
    parser.add_argument("--dimension", choices=("spill_depth", "num_concurrent"))
    parser.add_argument("--safe-setting", type=int)
    parser.add_argument("--min-improvement", type=float, default=0.05)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    missing = [name for name in ("baseline", "expected_limit", "dimension", "safe_setting") if getattr(args, name) is None]
    if missing or not args.candidate:
        parser.error(f"missing required arguments: {', '.join(missing + ([] if args.candidate else ['candidate']))}")
    if args.expected_limit <= 0 or args.safe_setting <= 0 or not 0 <= args.min_improvement < 1:
        parser.error("expected limit/safe setting must be positive and min improvement must be in [0,1)")
    report = build_report(
        args.baseline, args.candidate, args.suite_lock, args.expected_limit,
        args.dimension, args.safe_setting, args.min_improvement,
    )
    rendered = markdown(report)
    if args.json_out:
        write_new(args.json_out, json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.markdown_out:
        write_new(args.markdown_out, rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
