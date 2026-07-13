#!/usr/bin/env python3
"""Strictly validate a one-task Harbor pilot before practical fanout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_agent_timeout(exception: object) -> bool:
    return (
        isinstance(exception, dict)
        and exception.get("exception_type") == "AgentTimeoutError"
        and isinstance(exception.get("exception_message"), str)
        and exception["exception_message"].startswith("Agent execution timed out after ")
        and exception["exception_message"].endswith(" seconds")
        and "harbor.trial.errors.AgentTimeoutError" in exception.get("exception_traceback", "")
        and isinstance(exception.get("occurred_at"), str)
    )


def validate(
    run_dir: Path, lock_path: Path, arm: str, panel: str,
    expected_task_override: str | None = None,
) -> dict:
    lock = json.loads(lock_path.read_text())
    expected_task = expected_task_override or lock["protocol"]["pilot_tasks"][panel]
    panel_tasks = (
        {row["harbor_task"] for row in lock["swe_bench_verified"]["tasks"]}
        if panel == "swe"
        else {row["name"] for row in lock["terminal_bench_2"]["tasks"]}
    )
    require(expected_task in panel_tasks, "pilot task is outside the frozen panel")
    receipt_path = run_dir / "run-metadata.json"
    config_path = run_dir / "resolved-harbor-config.json"
    copied_lock = run_dir / "practical-evals.lock.json"
    for path in (receipt_path, config_path, copied_lock):
        require(path.is_file(), f"missing pilot evidence: {path}")
    receipt = json.loads(receipt_path.read_text())
    config = json.loads(config_path.read_text())
    require(receipt.get("completed_successfully") is True, "pilot runner did not complete")
    require(receipt.get("arm") == arm and receipt.get("panel") == panel, "pilot arm/panel differs")
    require(receipt.get("pilot_task") == expected_task, "pilot task differs from lock")
    require(receipt.get("lock_sha256") == sha256(lock_path) == sha256(copied_lock), "pilot lock differs")
    datasets = config.get("datasets")
    require(isinstance(datasets, list) and len(datasets) == 1, "pilot must use one dataset")
    require(datasets[0].get("task_names") == [expected_task], "pilot config must contain exactly its frozen task")
    job_dir = run_dir / "jobs" / receipt["run_id"]
    result = json.loads((job_dir / "result.json").read_text())
    stats = result.get("stats", {})
    require(result.get("n_total_trials") == 1, "pilot must contain one trial")
    trial_dirs = [path for path in job_dir.iterdir() if path.is_dir()]
    require(len(trial_dirs) == 1, "pilot must create one trial directory")
    trial = json.loads((trial_dirs[0] / "result.json").read_text())
    require(trial.get("task_name") == expected_task, "pilot trial task differs")
    agent = trial.get("agent_info", {})
    model = agent.get("model_info", {})
    require(
        agent.get("name") == "terminus-2"
        and model.get("name") == arm and model.get("provider") == "openai",
        "pilot trial agent/model differs",
    )
    exception = trial.get("exception_info")
    timed_out = is_agent_timeout(exception)
    require(exception is None or timed_out, "pilot trial has a non-timeout exception")
    reward = trial.get("verifier_result", {}).get("rewards", {}).get("reward")
    require(
        isinstance(reward, (int, float)) and not isinstance(reward, bool)
        and math.isfinite(float(reward)) and 0 <= float(reward) <= 1,
        "pilot verifier reward is invalid",
    )
    require(not timed_out or float(reward) == 0.0, "timed-out pilot must score zero")
    require(
        stats.get("n_completed_trials") == 1
        and stats.get("n_errored_trials") == int(timed_out)
        and stats.get("n_cancelled_trials") == 0
        and stats.get("n_running_trials", 0) == 0
        and stats.get("n_pending_trials", 0) == 0
        and stats.get("n_retries") == 0,
        "pilot completion/error accounting differs",
    )
    return {
        "arm": arm, "panel": panel, "task": expected_task,
        "reward": float(reward), "timed_out": timed_out,
    }


def self_test() -> None:
    lock_path = Path(__file__).with_name("practical-evals.lock.json")
    lock = json.loads(lock_path.read_text())
    arm = "pilot_arm"
    panel = "swe"
    task = lock["protocol"]["pilot_tasks"][panel]
    run_id = "pilot-self-test"
    with tempfile.TemporaryDirectory(prefix="bw24-practical-pilot-") as tmp:
        run_dir = Path(tmp)
        copied_lock = run_dir / "practical-evals.lock.json"
        copied_lock.write_bytes(lock_path.read_bytes())
        (run_dir / "run-metadata.json").write_text(json.dumps({
            "completed_successfully": True,
            "arm": arm,
            "panel": panel,
            "pilot_task": task,
            "lock_sha256": sha256(lock_path),
            "run_id": run_id,
        }))
        (run_dir / "resolved-harbor-config.json").write_text(json.dumps({
            "datasets": [{"task_names": [task]}],
        }))
        job_dir = run_dir / "jobs" / run_id
        trial_dir = job_dir / "trial"
        trial_dir.mkdir(parents=True)
        (job_dir / "result.json").write_text(json.dumps({
            "n_total_trials": 1,
            "stats": {
                "n_completed_trials": 1,
                "n_errored_trials": 0,
                "n_cancelled_trials": 0,
                "n_running_trials": 0,
                "n_pending_trials": 0,
                "n_retries": 0,
            },
        }))
        (trial_dir / "result.json").write_text(json.dumps({
            "task_name": task,
            "exception_info": None,
            "agent_info": {
                "name": "terminus-2",
                "model_info": {"name": arm, "provider": "openai"},
            },
            "verifier_result": {"rewards": {"reward": 1.0}},
        }))
        result = validate(run_dir, lock_path, arm, panel)
        assert result == {
            "arm": arm, "panel": panel, "task": task,
            "reward": 1.0, "timed_out": False,
        }
        trial = json.loads((trial_dir / "result.json").read_text())
        trial["exception_info"] = {
            "exception_type": "AgentTimeoutError",
            "exception_message": "Agent execution timed out after 60.0 seconds",
            "exception_traceback": "harbor.trial.errors.AgentTimeoutError: timeout",
            "occurred_at": "now",
        }
        trial["verifier_result"]["rewards"]["reward"] = 0.0
        (trial_dir / "result.json").write_text(json.dumps(trial))
        job_result = json.loads((job_dir / "result.json").read_text())
        job_result["stats"]["n_errored_trials"] = 1
        (job_dir / "result.json").write_text(json.dumps(job_result))
        timed_out = validate(run_dir, lock_path, arm, panel)
        assert timed_out["timed_out"] is True and timed_out["reward"] == 0.0
        trial["exception_info"]["exception_type"] = "RuntimeError"
        (trial_dir / "result.json").write_text(json.dumps(trial))
        try:
            validate(run_dir, lock_path, arm, panel)
        except ValueError as exc:
            assert "non-timeout exception" in str(exc)
        else:
            raise AssertionError("pilot validator accepted an errored trial")
    print("practical pilot self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--arm")
    parser.add_argument("--panel", choices=("swe", "terminal"))
    parser.add_argument("--expected-task")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not all((args.run_dir, args.lock, args.arm, args.panel)):
        parser.error("--run-dir, --lock, --arm, and --panel are required")
    result = validate(
        args.run_dir, args.lock, args.arm, args.panel,
        expected_task_override=args.expected_task,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
