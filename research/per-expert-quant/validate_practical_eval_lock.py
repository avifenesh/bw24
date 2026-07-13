#!/usr/bin/env python3
"""Validate the pinned SWE-bench Verified and Terminal-Bench directional subsets."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import tempfile
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any


SHA256_RE = re.compile(r"(?:sha256:)?[0-9a-f]{64}")
TERMINAL_FILE_DIRS = ("environment", "tests", "solution", "steps")
DEFAULT_IGNORES = ("__pycache__/*", "*.pyc", ".DS_Store", "*.swp", "*.swo", "*~")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_content_hash(task_dir: Path) -> str:
    require(not (task_dir / ".gitignore").exists(), f"root .gitignore requires Harbor pathspec: {task_dir}")
    files = [
        task_dir / name for name in ("task.toml", "instruction.md", "README.md")
        if (task_dir / name).is_file()
    ]
    for name in TERMINAL_FILE_DIRS:
        directory = task_dir / name
        if directory.is_dir():
            files.extend(path for path in directory.rglob("*") if path.is_file())
    filtered = []
    for path in files:
        relative = path.relative_to(task_dir).as_posix()
        if not any(fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch(path.name, pattern) for pattern in DEFAULT_IGNORES):
            filtered.append(path)
    outer = hashlib.sha256()
    for path in sorted(filtered, key=lambda item: item.relative_to(task_dir).as_posix()):
        relative = path.relative_to(task_dir).as_posix()
        outer.update(f"{relative}\0{file_sha256(path)}\n".encode())
    return outer.hexdigest()


def validate_structure(lock: dict[str, Any]) -> None:
    require(lock.get("format") == "bw24-practical-evals-v1", "wrong lock format")
    protocol = lock.get("protocol")
    require(isinstance(protocol, dict), "missing protocol")
    for key in (
        "same_agent_scaffold", "same_tool_and_turn_budget", "deterministic_generation",
        "container_isolation_required", "full_suite_only_after_directional_promotion",
    ):
        require(protocol.get(key) is True, f"protocol must require {key}")
    require(protocol.get("mtp_or_speculation") is False, "MTP/speculation must be disabled")
    require(protocol.get("initial_trials_per_task") == 1, "directional screen must use one trial")
    scaffold = protocol.get("agent_scaffold")
    require(isinstance(scaffold, dict), "missing practical agent scaffold")
    expected_scaffold = {
        "harbor_version": "0.18.0", "agent": "terminus-2",
        "model_name_template": "openai/{arm}", "api_base": "http://127.0.0.1:8080/v1",
        "temperature": 0, "parser_name": "json", "max_turns": 20,
        "proactive_summarization_threshold": 1024, "store_all_messages": True,
        "max_input_tokens": 5120, "max_output_tokens": 3072,
        "llm_call_max_tokens": 3072, "llm_call_timeout_seconds": 7200,
        "enable_summarize": True,
        "record_terminal_session": True,
        "agent_timeout_multiplier": 4.0,
        "n_concurrent_trials": 1, "n_attempts": 1, "max_retries": 0,
    }
    require(scaffold == expected_scaffold, "practical agent scaffold differs from the frozen protocol")

    swe = lock.get("swe_bench_verified")
    require(isinstance(swe, dict), "missing SWE-bench lock")
    swe_tasks = swe.get("tasks")
    require(isinstance(swe_tasks, list) and len(swe_tasks) == 12, "SWE subset must contain 12 tasks")
    require(len({row.get("instance_id") for row in swe_tasks}) == 12, "duplicate SWE instance")
    require(len({row.get("repo") for row in swe_tasks}) == 12, "SWE subset must cover 12 repositories")
    require(
        Counter(row.get("difficulty") for row in swe_tasks)
        == Counter({"<15 min fix": 4, "15 min - 1 hour": 4, "1-4 hours": 3, ">4 hours": 1}),
        "unexpected SWE difficulty strata",
    )
    require(re.fullmatch(r"[0-9a-f]{40}", str(swe.get("dataset_revision"))) is not None, "invalid SWE revision")
    require(SHA256_RE.fullmatch(str(swe.get("parquet_sha256"))) is not None, "invalid parquet hash")
    require(swe.get("harbor_dataset_revision") == 2, "SWE Harbor adapter revision must be 2")
    require(SHA256_RE.fullmatch(str(swe.get("harbor_dataset_digest"))) is not None, "invalid SWE Harbor dataset digest")
    for row in swe_tasks:
        require(re.fullmatch(r"[0-9a-f]{40}", str(row.get("base_commit"))) is not None, f"bad SWE base commit: {row}")
        require(row.get("harbor_task") == f"swe-bench/{row['instance_id']}", f"bad SWE Harbor task: {row}")
        require(SHA256_RE.fullmatch(str(row.get("harbor_digest"))) is not None, f"bad SWE Harbor digest: {row}")

    terminal = lock.get("terminal_bench_2")
    require(isinstance(terminal, dict), "missing Terminal-Bench lock")
    terminal_tasks = terminal.get("tasks")
    require(isinstance(terminal_tasks, list) and len(terminal_tasks) == 12, "Terminal subset must contain 12 tasks")
    require(len({row.get("name") for row in terminal_tasks}) == 12, "duplicate Terminal task")
    require(terminal.get("dataset_revision") == 1, "Terminal dataset revision must be 1")
    require(SHA256_RE.fullmatch(str(terminal.get("dataset_digest"))) is not None, "bad Terminal dataset digest")
    require(Counter(row.get("difficulty") for row in terminal_tasks) == Counter({"medium": 10, "hard": 2}), "unexpected Terminal difficulty mix")
    require(len({row.get("category") for row in terminal_tasks}) >= 9, "Terminal subset lacks category diversity")
    for row in terminal_tasks:
        require(SHA256_RE.fullmatch(str(row.get("digest"))) is not None, f"bad Terminal digest: {row}")
        require(row.get("gpus") == 0, f"Terminal directional task requires GPU: {row['name']}")
        require(isinstance(row.get("agent_timeout_sec"), int) and row["agent_timeout_sec"] <= 2400, f"unbounded Terminal timeout: {row['name']}")


def validate_swe_source(lock: dict[str, Any], parquet: Path) -> None:
    swe = lock["swe_bench_verified"]
    require(file_sha256(parquet) == swe["parquet_sha256"], "pinned SWE parquet hash differs")
    try:
        import duckdb  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ValueError("duckdb is required to validate SWE parquet rows") from exc
    connection = duckdb.connect()
    try:
        count = connection.execute("SELECT count(*) FROM read_parquet(?)", [str(parquet)]).fetchone()[0]
        require(count == swe["dataset_rows"], f"SWE row count differs: {count}")
        for expected in swe["tasks"]:
            rows = connection.execute(
                "SELECT repo, base_commit, difficulty FROM read_parquet(?) WHERE instance_id = ?",
                [str(parquet), expected["instance_id"]],
            ).fetchall()
            require(rows == [(expected["repo"], expected["base_commit"], expected["difficulty"])], f"SWE task differs: {expected['instance_id']}")
    finally:
        connection.close()


def validate_terminal_source(lock: dict[str, Any], root: Path) -> None:
    for expected in lock["terminal_bench_2"]["tasks"]:
        short_name = expected["name"].split("/", 1)[1]
        task_dir = root / short_name
        require(task_dir.is_dir(), f"missing Terminal task directory: {task_dir}")
        config = tomllib.loads((task_dir / "task.toml").read_text())
        require(config["task"]["name"] == expected["name"], f"Terminal task name differs: {short_name}")
        require(config["metadata"]["difficulty"] == expected["difficulty"], f"Terminal difficulty differs: {short_name}")
        require(config["metadata"]["category"] == expected["category"], f"Terminal category differs: {short_name}")
        require(int(config["agent"]["timeout_sec"]) == expected["agent_timeout_sec"], f"Terminal timeout differs: {short_name}")
        environment = config["environment"]
        for key in ("cpus", "memory_mb", "gpus", "docker_image"):
            require(environment[key] == expected[key], f"Terminal {key} differs: {short_name}")
        actual_digest = "sha256:" + task_content_hash(task_dir)
        require(actual_digest == expected["digest"], f"Terminal task digest differs: {short_name}")


def validate_swe_harbor_source(lock: dict[str, Any], root: Path) -> None:
    for expected in lock["swe_bench_verified"]["tasks"]:
        task_dir = root / expected["instance_id"]
        require(task_dir.is_dir(), f"missing SWE Harbor task directory: {task_dir}")
        config = tomllib.loads((task_dir / "task.toml").read_text())
        require(config["task"]["name"] == expected["harbor_task"], f"SWE Harbor task name differs: {expected['instance_id']}")
        actual_digest = "sha256:" + task_content_hash(task_dir)
        require(actual_digest == expected["harbor_digest"], f"SWE Harbor task digest differs: {expected['instance_id']}")


def self_test() -> None:
    source = Path(__file__).with_name("practical-evals.lock.json")
    if source.is_file():
        validate_structure(json.loads(source.read_text()))
    with tempfile.TemporaryDirectory(prefix="bw24-practical-lock-") as tmp:
        task = Path(tmp)
        (task / "task.toml").write_text("version = '1'\n")
        first = task_content_hash(task)
        (task / "instruction.md").write_text("do the task\n")
        second = task_content_hash(task)
        assert first != second and len(first) == 64 and len(second) == 64
    print("practical eval lock self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=Path(__file__).with_name("practical-evals.lock.json"))
    parser.add_argument("--swe-parquet", type=Path)
    parser.add_argument("--terminal-root", type=Path)
    parser.add_argument("--swe-harbor-root", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    lock = json.loads(args.lock.read_text())
    validate_structure(lock)
    if args.swe_parquet:
        validate_swe_source(lock, args.swe_parquet)
    if args.terminal_root:
        validate_terminal_source(lock, args.terminal_root)
    if args.swe_harbor_root:
        validate_swe_harbor_source(lock, args.swe_harbor_root)
    print("practical eval lock validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
