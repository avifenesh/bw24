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
FULL_MAX_TURNS = 100
SHARED_KEYS = (
    "panel", "dataset", "dataset_name", "dataset_digest", "base_url",
    "harbor_version", "bw24_commit", "lock_sha256", "server_binary_sha256",
    "declared_spill_io", "declared_spill_pread_depth", "declared_spill_stats",
    "declared_serve_spec", "declared_max_turns",
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


def load_run(run_dir: Path, panel: str, lock_path: Path, full_lock_path: Path) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text())
    full_lock = json.loads(full_lock_path.read_text())
    require(full_lock.get("format") == "bw24-full-practical-task-lock-v1", "wrong full task lock")
    require(full_lock.get("source_practical_lock_sha256") == sha256(lock_path), "full task lock binding differs")
    suite = lock["swe_bench_verified"] if panel == "swe" else lock["terminal_bench_2"]
    full_suite = full_lock[panel]
    name = suite["harbor_dataset"] if panel == "swe" else suite["dataset"]
    digest = suite["harbor_dataset_digest"] if panel == "swe" else suite["dataset_digest"]
    expected_map = {row["name"]: row["digest"] for row in full_suite["tasks"]}
    expected_total = 500 if panel == "swe" else 89
    require(len(expected_map) == expected_total, "full task lock count differs")

    receipt_paths = sorted(run_dir.glob("shards/*/run-metadata.json"))
    if not receipt_paths and (run_dir / "run-metadata.json").is_file():
        receipt_paths = [run_dir / "run-metadata.json"]
    require(bool(receipt_paths), f"no full practical receipts under {run_dir}")
    rewards: dict[str, float] = {}
    raw_verifier_rewards: dict[str, float] = {}
    digests: dict[str, str] = {}
    timeouts: dict[str, bool] = {}
    reference: dict[str, Any] | None = None
    artifact_bytes: int | None = None
    elapsed_total = 0.0
    for receipt_path in receipt_paths:
        shard = receipt_path.parent
        paths = {
            "trials": shard / "validated-trials.json", "manifest": shard / "artifact-manifest.json",
            "config": shard / "resolved-harbor-config.json", "images": shard / "container-images.jsonl",
            "lock": shard / "practical-evals.lock.json", "full_lock": shard / "full-practical-tasks.lock.json",
            "selected": shard / "selected-tasks.json", "server_log": shard / "server.log",
        }
        require(all(path.is_file() for path in paths.values()), f"missing full practical evidence: {shard}")
        receipt = json.loads(receipt_path.read_text())
        trials = json.loads(paths["trials"].read_text())
        manifest = json.loads(paths["manifest"].read_text())
        config = json.loads(paths["config"].read_text())
        selected = {row["name"]: row["digest"] for row in json.loads(paths["selected"].read_text())}
        require(receipt.get("format") == "bw24-full-practical-run-v1", f"wrong receipt: {shard}")
        require(receipt.get("panel") == panel, f"wrong panel: {shard}")
        require(receipt.get("expected_tasks") == len(selected) == len(trials), f"wrong shard count: {shard}")
        require(receipt.get("completed_successfully") is True and receipt.get("harbor_exit_code") == 0 and receipt.get("tee_exit_code") == 0, f"unfinished run: {shard}")
        elapsed = receipt.get("elapsed_seconds")
        require(isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool) and elapsed > 0, f"bad elapsed: {shard}")
        elapsed_total += float(elapsed)
        require(receipt.get("lock_sha256") == sha256(paths["lock"]) == sha256(lock_path), f"lock differs: {shard}")
        require(receipt.get("full_task_lock_sha256") == sha256(paths["full_lock"]) == sha256(full_lock_path), f"full lock differs: {shard}")
        require(receipt.get("selected_tasks_sha256") == sha256(paths["selected"]), f"selection differs: {shard}")
        require(receipt.get("artifact_manifest_sha256") == sha256(paths["manifest"]), f"manifest differs: {shard}")
        require(receipt.get("artifact_bytes") == manifest.get("artifact_bytes"), f"artifact bytes differ: {shard}")
        require(receipt.get("validated_trials_sha256") == sha256(paths["trials"]), f"trials hash differs: {shard}")
        require(receipt.get("resolved_harbor_config_sha256") == sha256(paths["config"]), f"config hash differs: {shard}")
        require(receipt.get("container_images_sha256") == sha256(paths["images"]), f"images hash differs: {shard}")
        require(receipt.get("server_log_sha256") == sha256(paths["server_log"]), f"server log differs: {shard}")
        spill = receipt.get("spill_delta")
        require(isinstance(spill, dict) and set(spill) == set(SPILL_KEYS), f"invalid spill: {shard}")
        require(all(isinstance(spill[k], int) and spill[k] >= 0 for k in SPILL_KEYS), f"negative spill: {shard}")
        require(spill["reads"] > 0 and spill["bytes"] > 0 and spill["errors"] == 0 and spill["short_reads"] == 0, f"spill failure: {shard}")
        require(receipt.get("dataset_name") == name and receipt.get("dataset_digest") == digest, f"dataset differs: {shard}")
        require(receipt.get("declared_max_turns") == FULL_MAX_TURNS, f"full turn budget differs: {shard}")
        datasets = config.get("datasets")
        require(isinstance(datasets, list) and len(datasets) == 1 and datasets[0].get("name") == name and (datasets[0].get("version") == digest or datasets[0].get("ref") == digest), f"Harbor dataset differs: {shard}")
        require(datasets[0].get("task_names") == list(selected), f"Harbor selected tasks differ: {shard}")
        require(config.get("n_concurrent_trials") == 1, f"Harbor concurrency differs: {shard}")
        require(config.get("agent_timeout_multiplier") == 4.0, f"Harbor agent timeout multiplier differs: {shard}")
        arm = receipt.get("arm")
        agents = config.get("agents")
        require(isinstance(agents, list) and len(agents) == 1 and agents[0].get("name") == "terminus-2" and agents[0].get("model_name") == f"openai/{arm}", f"wrong model: {shard}")
        scaffold = lock["protocol"]["agent_scaffold"]
        expected_kwargs = {
            "api_base": scaffold["api_base"],
            "temperature": scaffold["temperature"],
            "max_turns": FULL_MAX_TURNS,
            "parser_name": scaffold["parser_name"],
            "proactive_summarization_threshold": scaffold["proactive_summarization_threshold"],
            "enable_summarize": scaffold["enable_summarize"],
            "store_all_messages": scaffold["store_all_messages"],
            "record_terminal_session": scaffold["record_terminal_session"],
            "model_info": {
                "max_input_tokens": scaffold["max_input_tokens"],
                "max_output_tokens": scaffold["max_output_tokens"],
                "input_cost_per_token": 0,
                "output_cost_per_token": 0,
            },
            "llm_call_kwargs": {
                "max_tokens": scaffold["llm_call_max_tokens"],
                "timeout": scaffold["llm_call_timeout_seconds"],
            },
        }
        require(agents[0].get("kwargs") == expected_kwargs, f"agent scaffold differs: {shard}")
        if reference is None:
            reference = receipt
            artifact_bytes = receipt["artifact_bytes"]
        else:
            for key in SHARED_KEYS + ("arm", "artifact_manifest_sha256"):
                require(receipt.get(key) == reference.get(key), f"shards differ on {key}")
        observed: dict[str, str] = {}
        shard_solved = 0.0
        shard_timeouts = 0
        for row in trials:
            task, task_digest, reward = row.get("task"), row.get("task_digest"), row.get("reward")
            raw_verifier_reward = row.get("raw_verifier_reward")
            require(isinstance(task, str) and task not in rewards and task not in observed, f"duplicate task: {shard}")
            timed_out = row.get("timed_out")
            timeout_reward_overridden = row.get("timeout_reward_overridden")
            require(
                isinstance(task_digest, str) and task_digest
                and isinstance(reward, (int, float)) and not isinstance(reward, bool)
                and math.isfinite(reward)
                and isinstance(raw_verifier_reward, (int, float))
                and not isinstance(raw_verifier_reward, bool)
                and math.isfinite(raw_verifier_reward)
                and 0 <= float(raw_verifier_reward) <= 1
                and isinstance(timed_out, bool)
                and isinstance(timeout_reward_overridden, bool)
                and float(reward) == (0.0 if timed_out else float(raw_verifier_reward))
                and timeout_reward_overridden == (timed_out and float(raw_verifier_reward) != 0.0),
                f"bad trial: {shard}",
            )
            observed[task] = task_digest
            rewards[task] = float(reward)
            raw_verifier_rewards[task] = float(raw_verifier_reward)
            digests[task] = task_digest
            timeouts[task] = timed_out
            shard_solved += float(reward)
            shard_timeouts += int(timed_out)
        require(observed == selected, f"selected task/digest set differs: {shard}")
        require(math.isclose(shard_solved, float(receipt.get("solved")), abs_tol=1e-12), f"solved differs: {shard}")
        require(
            math.isclose(
                sum(float(row["raw_verifier_reward"]) for row in trials),
                float(receipt.get("raw_verifier_solved")), abs_tol=1e-12,
            ),
            f"raw verifier solved differs: {shard}",
        )
        require(
            receipt.get("timeout_reward_overrides")
            == sum(bool(row["timeout_reward_overridden"]) for row in trials),
            f"timeout reward override count differs: {shard}",
        )
        require(receipt.get("timed_out") == shard_timeouts, f"timeout count differs: {shard}")
    require(digests == expected_map, f"full task union differs: {run_dir}")
    assert reference is not None and artifact_bytes is not None
    return {
        "arm": reference["arm"], "artifact_bytes": artifact_bytes,
        "elapsed_seconds": elapsed_total, "rewards": rewards,
        "raw_verifier_rewards": raw_verifier_rewards,
        "timeouts": timeouts, "timeout_count": sum(timeouts.values()),
        "timeout_reward_override_count": sum(
            timeouts[task] and raw_verifier_rewards[task] != 0.0 for task in rewards
        ),
        "task_digests": digests, "receipt": reference,
    }


def compare(
    baseline_dir: Path, candidate_dir: Path, panel: str, lock: Path, full_task_lock: Path
) -> dict[str, Any]:
    baseline = load_run(baseline_dir, panel, lock, full_task_lock)
    candidate = load_run(candidate_dir, panel, lock, full_task_lock)
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
        tasks.append({
            "task": task, "baseline_reward": base, "candidate_reward": cand,
            "delta": delta, "baseline_timed_out": baseline["timeouts"][task],
            "candidate_timed_out": candidate["timeouts"][task],
            "baseline_raw_verifier_reward": baseline["raw_verifier_rewards"][task],
            "candidate_raw_verifier_reward": candidate["raw_verifier_rewards"][task],
        })
    n = len(tasks)
    return {
        "format": "bw24-full-practical-comparison-v1", "panel": panel, "n_tasks": n,
        "baseline": {"arm": baseline["arm"], "solved": sum(baseline["rewards"].values()),
                     "raw_verifier_solved": sum(baseline["raw_verifier_rewards"].values()),
                     "rate": sum(baseline["rewards"].values()) / n,
                     "artifact_bytes": baseline["artifact_bytes"], "elapsed_seconds": baseline["elapsed_seconds"],
                     "timed_out": baseline["timeout_count"],
                     "timeout_reward_overrides": baseline["timeout_reward_override_count"]},
        "candidate": {"arm": candidate["arm"], "solved": sum(candidate["rewards"].values()),
                      "raw_verifier_solved": sum(candidate["raw_verifier_rewards"].values()),
                      "rate": sum(candidate["rewards"].values()) / n,
                      "artifact_bytes": candidate["artifact_bytes"], "elapsed_seconds": candidate["elapsed_seconds"],
                      "timed_out": candidate["timeout_count"],
                      "timeout_reward_overrides": candidate["timeout_reward_override_count"]},
        "candidate_solved_delta": sum(candidate["rewards"].values()) - sum(baseline["rewards"].values()),
        "paired_wins": wins, "paired_losses": losses, "paired_ties": ties, "tasks": tasks,
        "note": (
            "Complete digest-pinned Harbor suite with one attempt per task. "
            "AgentTimeoutError tasks score zero; late verifier rewards remain recorded as raw provenance."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--panel", choices=("swe", "terminal"), required=True)
    parser.add_argument("--lock", type=Path, default=Path(__file__).with_name("practical-evals.lock.json"))
    parser.add_argument(
        "--full-task-lock", type=Path,
        default=Path(__file__).with_name("full-practical-tasks.lock.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.baseline, args.candidate, args.panel, args.lock, args.full_task_lock)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"wrote {args.output}: delta={report['candidate_solved_delta']:+g}")


if __name__ == "__main__":
    main()
