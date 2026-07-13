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


def validate(run_dir: Path, lock_path: Path, arm: str, panel: str) -> dict:
    lock = json.loads(lock_path.read_text())
    expected_task = lock["protocol"]["pilot_tasks"][panel]
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
    require(
        stats.get("n_completed_trials") == 1
        and stats.get("n_errored_trials") == 0
        and stats.get("n_cancelled_trials") == 0
        and stats.get("n_running_trials", 0) == 0
        and stats.get("n_pending_trials", 0) == 0
        and stats.get("n_retries") == 0,
        "pilot has incomplete, errored, cancelled, or retried work",
    )
    trial_dirs = [path for path in job_dir.iterdir() if path.is_dir()]
    require(len(trial_dirs) == 1, "pilot must create one trial directory")
    trial = json.loads((trial_dirs[0] / "result.json").read_text())
    require(trial.get("task_name") == expected_task, "pilot trial task differs")
    require(trial.get("exception_info") is None, "pilot trial has an exception")
    reward = trial.get("verifier_result", {}).get("rewards", {}).get("reward")
    require(
        isinstance(reward, (int, float)) and not isinstance(reward, bool)
        and math.isfinite(float(reward)) and 0 <= float(reward) <= 1,
        "pilot verifier reward is invalid",
    )
    return {"arm": arm, "panel": panel, "task": expected_task, "reward": float(reward)}


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
            "verifier_result": {"rewards": {"reward": 1.0}},
        }))
        result = validate(run_dir, lock_path, arm, panel)
        assert result == {"arm": arm, "panel": panel, "task": task, "reward": 1.0}
        trial = json.loads((trial_dir / "result.json").read_text())
        trial["exception_info"] = {"type": "AgentTimeoutError"}
        (trial_dir / "result.json").write_text(json.dumps(trial))
        try:
            validate(run_dir, lock_path, arm, panel)
        except ValueError as exc:
            assert "exception" in str(exc)
        else:
            raise AssertionError("pilot validator accepted an errored trial")
    print("practical pilot self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--arm")
    parser.add_argument("--panel", choices=("swe", "terminal"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not all((args.run_dir, args.lock, args.arm, args.panel)):
        parser.error("--run-dir, --lock, --arm, and --panel are required")
    result = validate(args.run_dir, args.lock, args.arm, args.panel)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
