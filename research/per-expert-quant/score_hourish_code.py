#!/usr/bin/env python3
"""Score frozen HumanEval/MBPP samples inside a locked-down container."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import resource
import signal
import subprocess
import sys
import tempfile
from collections import defaultdict
from typing import Any


CPU_SECONDS = 5
WALL_SECONDS = 10
ADDRESS_SPACE_BYTES = 512 * 1024 * 1024
MAX_STDERR_BYTES = 4096


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
    resource.setrlimit(resource.RLIMIT_AS, (ADDRESS_SPACE_BYTES, ADDRESS_SPACE_BYTES))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def single_response(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if (
        not isinstance(value, list)
        or len(value) != 1
        or not isinstance(value[0], list)
        or len(value[0]) != 1
        or not isinstance(value[0][0], str)
    ):
        raise ValueError(f"expected exactly one {key} candidate")
    return value[0][0]


def candidate_from(row: dict[str, Any], task: str) -> str:
    if task == "mbpp_instruct":
        # The task's gen_prefix already opens a Python fence but lm-eval does not include that
        # prefix in `resps`. Its upstream filter prepends "```" without a newline, causing the
        # regex to consume a leading `def` as though it were a Markdown language tag. The code is
        # the raw response up to the model's first closing fence.
        raw = single_response(row, "resps")
        return raw.split("```", 1)[0].rstrip()
    filtered = row.get("filtered_resps")
    if (
        not isinstance(filtered, list)
        or len(filtered) != 1
        or not isinstance(filtered[0], list)
        or len(filtered[0]) != 1
        or not isinstance(filtered[0][0], str)
    ):
        raise ValueError("expected exactly one filtered candidate")
    return filtered[0][0]


def evaluate_candidate(candidate: str, target: str) -> dict[str, Any]:
    source = candidate + "\n" + target + "\n"
    with tempfile.TemporaryDirectory(prefix="bw24-code-") as tmp:
        script = pathlib.Path(tmp) / "candidate.py"
        script.write_text(source, encoding="utf-8")
        process = subprocess.Popen(
            [sys.executable, "-I", "-B", str(script)],
            cwd=tmp,
            env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
            preexec_fn=limits,
        )
        try:
            _, raw_stderr = process.communicate(timeout=WALL_SECONDS)
            stderr = raw_stderr[-MAX_STDERR_BYTES:].decode("utf-8", errors="replace")
            return {
                "passed": process.returncode == 0,
                "returncode": process.returncode,
                "timed_out": False,
                "stderr_tail": stderr,
            }
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            _, raw_stderr = process.communicate()
            stderr = raw_stderr[-MAX_STDERR_BYTES:].decode("utf-8", errors="replace")
            return {
                "passed": False,
                "returncode": None,
                "timed_out": True,
                "stderr_tail": stderr,
            }
        finally:
            # Kill descendants that outlived a normally exiting parent before the next sample.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def task_from_path(path: pathlib.Path) -> str:
    name = path.name
    for task in ("humaneval_instruct", "mbpp_instruct"):
        if name.startswith(f"samples_{task}_") and name.endswith(".jsonl"):
            return task
    raise ValueError(f"unrecognized code sample file: {path}")


def score(paths: list[pathlib.Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("no sample files")
    results = []
    seen = set()
    input_files = []
    for path in sorted(paths):
        task = task_from_path(path)
        input_files.append({"path": str(path), "sha256": sha256(path)})
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                row = json.loads(line)
                doc_id = row.get("doc_id")
                key = (task, doc_id)
                if key in seen:
                    raise ValueError(f"duplicate sample {key}")
                seen.add(key)
                target = row.get("target")
                if not isinstance(doc_id, int) or not isinstance(target, str):
                    raise ValueError(f"invalid sample identity at {path}:{line_number}")
                outcome = evaluate_candidate(candidate_from(row, task), target)
                results.append(
                    {
                        "task": task,
                        "doc_id": doc_id,
                        "doc_hash": row.get("doc_hash"),
                        "prompt_hash": row.get("prompt_hash"),
                        "target_hash": row.get("target_hash"),
                        **outcome,
                    }
                )
    by_task: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "total": 0})
    for result in results:
        summary = by_task[result["task"]]
        summary["total"] += 1
        summary["passed"] += int(result["passed"])
    return {
        "format": "bw24-hourish-code-score-v1",
        "limits": {
            "cpu_seconds": CPU_SECONDS,
            "wall_seconds": WALL_SECONDS,
            "address_space_bytes": ADDRESS_SPACE_BYTES,
        },
        "input_files": input_files,
        "by_task": dict(sorted(by_task.items())),
        "passed": sum(int(result["passed"]) for result in results),
        "total": len(results),
        "samples": results,
    }


def self_test() -> None:
    assert evaluate_candidate("def f(): return 3", "assert f() == 3")["passed"]
    assert not evaluate_candidate("def f(): return 4", "assert f() == 3")["passed"]
    assert not evaluate_candidate("while True: pass", "")["passed"]
    row = {"resps": [["def f():\n    return 3\n```\nignored"]]}
    assert candidate_from(row, "mbpp_instruct") == "def f():\n    return 3"
    print("hourish code scorer self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("samples", nargs="*", type=pathlib.Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    print(json.dumps(score(args.samples), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
