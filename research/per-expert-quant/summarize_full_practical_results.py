#!/usr/bin/env python3
"""Strictly validate and compare complete SWE/Terminal Harbor runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SPILL_KEYS = ("reads", "bytes", "errors", "short_reads", "fallbacks", "buffer_waits", "ring_full")
SHARED_KEYS = (
    "panel", "dataset", "dataset_name", "dataset_digest", "expected_tasks", "base_url",
    "harbor_version", "bw24_commit", "lock_sha256", "server_binary_sha256",
    "declared_spill_io", "declared_spill_pread_depth", "declared_spill_stats",
    "declared_serve_spec",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_run(run_dir: Path, panel: str, lock_path: Path) -> dict[str, Any]:
    receipt_path = run_dir / "run-metadata.json"
    trials_path = run_dir / "validated-trials.json"
    manifest_path = run_dir / "artifact-manifest.json"
    config_path = run_dir / "resolved-harbor-config.json"
    images_path = run_dir / "container-images.jsonl"
    lock_copy = run_dir / "practical-evals.lock.json"
    for path in (receipt_path, trials_path, manifest_path, config_path, images_path, lock_copy):
        require(path.is_file(), f"missing full practical evidence: {path}")
    receipt = json.loads(receipt_path.read_text())
    trials = json.loads(trials_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    config = json.loads(config_path.read_text())
    lock = json.loads(lock_path.read_text())
    require(receipt.get("format") == "bw24-full-practical-run-v1", f"wrong receipt: {run_dir}")
    require(receipt.get("panel") == panel, f"wrong panel: {run_dir}")
    expected = 500 if panel == "swe" else 89
    require(receipt.get("expected_tasks") == expected and len(trials) == expected, f"wrong task count: {run_dir}")
    require(
        receipt.get("completed_successfully") is True
        and receipt.get("harbor_exit_code") == 0 and receipt.get("tee_exit_code") == 0,
        f"unfinished full practical run: {run_dir}",
    )
    elapsed = receipt.get("elapsed_seconds")
    require(isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool) and elapsed > 0, f"invalid elapsed: {run_dir}")
    require(receipt.get("lock_sha256") == sha256(lock_copy) == sha256(lock_path), f"lock differs: {run_dir}")
    require(receipt.get("artifact_manifest_sha256") == sha256(manifest_path), f"manifest differs: {run_dir}")
    require(receipt.get("artifact_bytes") == manifest.get("artifact_bytes"), f"artifact bytes differ: {run_dir}")
    require(receipt.get("validated_trials_sha256") == sha256(trials_path), f"trials hash differs: {run_dir}")
    require(receipt.get("resolved_harbor_config_sha256") == sha256(config_path), f"config hash differs: {run_dir}")
    require(receipt.get("container_images_sha256") == sha256(images_path), f"images hash differs: {run_dir}")
    server_log = run_dir / "server.log"
    require(server_log.is_file() and receipt.get("server_log_sha256") == sha256(server_log), f"server log differs: {run_dir}")
    spill = receipt.get("spill_delta")
    require(isinstance(spill, dict) and set(spill) == set(SPILL_KEYS), f"invalid spill: {run_dir}")
    require(all(isinstance(spill[k], int) and spill[k] >= 0 for k in SPILL_KEYS), f"negative spill: {run_dir}")
    require(spill["reads"] > 0 and spill["bytes"] > 0, f"no spill activity: {run_dir}")
    require(spill["errors"] == 0 and spill["short_reads"] == 0, f"spill failure: {run_dir}")

    suite = lock["swe_bench_verified"] if panel == "swe" else lock["terminal_bench_2"]
    name = suite["harbor_dataset"] if panel == "swe" else suite["dataset"]
    digest = suite["harbor_dataset_digest"] if panel == "swe" else suite["dataset_digest"]
    require(receipt.get("dataset_name") == name and receipt.get("dataset_digest") == digest, f"dataset differs: {run_dir}")
    datasets = config.get("datasets")
    require(isinstance(datasets, list) and len(datasets) == 1, f"wrong Harbor datasets: {run_dir}")
    require(
        datasets[0].get("name") == name
        and (datasets[0].get("version") == digest or datasets[0].get("ref") == digest),
        f"resolved Harbor dataset differs: {run_dir}",
    )
    require(config.get("n_concurrent_trials") == 1, f"Harbor concurrency differs: {run_dir}")
    agents = config.get("agents")
    arm = receipt.get("arm")
    require(isinstance(agents, list) and len(agents) == 1, f"wrong agent config: {run_dir}")
    require(agents[0].get("name") == "terminus-2" and agents[0].get("model_name") == f"openai/{arm}", f"wrong model: {run_dir}")
    expected_kwargs = {
        "api_base": lock["protocol"]["agent_scaffold"]["api_base"], "temperature": 0,
        "max_turns": 20, "parser_name": "json", "proactive_summarization_threshold": 1024,
        "enable_summarize": True, "store_all_messages": True, "record_terminal_session": True,
        "model_info": {"max_input_tokens": 8192, "max_output_tokens": 512,
                       "input_cost_per_token": 0, "output_cost_per_token": 0},
        "llm_call_kwargs": {"max_tokens": 512},
    }
    require(agents[0].get("kwargs") == expected_kwargs, f"agent scaffold differs: {run_dir}")

    rewards: dict[str, float] = {}
    digests: dict[str, str] = {}
    for row in trials:
        task, digest_value, reward = row.get("task"), row.get("task_digest"), row.get("reward")
        require(isinstance(task, str) and task not in rewards, f"duplicate task: {run_dir}")
        require(isinstance(digest_value, str) and digest_value, f"missing task digest: {run_dir}")
        require(isinstance(reward, (int, float)) and not isinstance(reward, bool) and math.isfinite(reward), f"bad reward: {run_dir}")
        rewards[task] = float(reward)
        digests[task] = digest_value
    require(math.isclose(sum(rewards.values()), float(receipt.get("solved")), abs_tol=1e-12), f"solved aggregate differs: {run_dir}")
    return {
        "arm": arm, "artifact_bytes": receipt["artifact_bytes"], "elapsed_seconds": float(elapsed),
        "rewards": rewards, "task_digests": digests, "receipt": receipt,
    }


def compare(baseline_dir: Path, candidate_dir: Path, panel: str, lock: Path) -> dict[str, Any]:
    baseline = load_run(baseline_dir, panel, lock)
    candidate = load_run(candidate_dir, panel, lock)
    require(baseline["arm"] != candidate["arm"], "baseline and candidate are identical")
    for key in SHARED_KEYS:
        require(baseline["receipt"].get(key) == candidate["receipt"].get(key), f"receipts differ on {key}")
    require(baseline["task_digests"] == candidate["task_digests"], "task digests differ")
    require(baseline["rewards"].keys() == candidate["rewards"].keys(), "task sets differ")
    wins = losses = ties = 0
    tasks = []
    for task, base in baseline["rewards"].items():
        cand = candidate["rewards"][task]
        delta = cand - base
        wins += delta > 0
        losses += delta < 0
        ties += delta == 0
        tasks.append({"task": task, "baseline_reward": base, "candidate_reward": cand, "delta": delta})
    n = len(tasks)
    return {
        "format": "bw24-full-practical-comparison-v1", "panel": panel, "n_tasks": n,
        "baseline": {"arm": baseline["arm"], "solved": sum(baseline["rewards"].values()),
                     "rate": sum(baseline["rewards"].values()) / n,
                     "artifact_bytes": baseline["artifact_bytes"], "elapsed_seconds": baseline["elapsed_seconds"]},
        "candidate": {"arm": candidate["arm"], "solved": sum(candidate["rewards"].values()),
                      "rate": sum(candidate["rewards"].values()) / n,
                      "artifact_bytes": candidate["artifact_bytes"], "elapsed_seconds": candidate["elapsed_seconds"]},
        "candidate_solved_delta": sum(candidate["rewards"].values()) - sum(baseline["rewards"].values()),
        "paired_wins": wins, "paired_losses": losses, "paired_ties": ties, "tasks": tasks,
        "note": "Complete digest-pinned Harbor suite with one attempt per task.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--panel", choices=("swe", "terminal"), required=True)
    parser.add_argument("--lock", type=Path, default=Path(__file__).with_name("practical-evals.lock.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.baseline, args.candidate, args.panel, args.lock)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"wrote {args.output}: delta={report['candidate_solved_delta']:+g}")


if __name__ == "__main__":
    main()
