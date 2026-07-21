#!/usr/bin/env python3
"""Validate and compare matched Harbor practical-eval panels."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


DEFAULT_SHARED_MODEL_BYTES = 24_999_514_624
SPILL_KEYS = ("reads", "bytes", "errors", "short_reads", "fallbacks", "buffer_waits", "ring_full")
SHARED_RECEIPT_KEYS = (
    "panel", "dataset", "harbor_version", "bw24_commit",
    "server_binary_sha256", "declared_spill_io", "declared_spill_pread_depth",
    "declared_spill_stats", "declared_serve_spec", "lock_sha256",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def expected_tasks(lock: dict[str, Any], panel: str) -> dict[str, str]:
    if panel == "swe":
        return {row["harbor_task"]: row["harbor_digest"] for row in lock["swe_bench_verified"]["tasks"]}
    return {row["name"]: row["digest"] for row in lock["terminal_bench_2"]["tasks"]}


def normalized_loopback_api_base(value: object) -> str:
    require(isinstance(value, str), "practical API base is not a string")
    parsed = urlsplit(value)
    require(
        parsed.scheme == "http"
        and parsed.hostname == "127.0.0.1"
        and parsed.port is not None
        and 1 <= parsed.port <= 65535
        and parsed.path == "/v1"
        and not parsed.query
        and not parsed.fragment
        and parsed.username is None
        and parsed.password is None,
        f"practical API base is not an isolated loopback /v1 endpoint: {value!r}",
    )
    return "http://127.0.0.1:{port}/v1"


def normalized_harbor_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(config)
    normalized.pop("job_name", None)
    normalized.pop("jobs_dir", None)
    agents = normalized.get("agents")
    if isinstance(agents, list) and len(agents) == 1:
        agents[0]["model_name"] = "openai/{arm}"
        kwargs = agents[0].get("kwargs")
        if isinstance(kwargs, dict):
            kwargs["api_base"] = normalized_loopback_api_base(kwargs.get("api_base"))
    return normalized


def load_run(run_dir: Path, lock: dict[str, Any], panel: str) -> dict[str, Any]:
    receipt_path = run_dir / "run-metadata.json"
    config_path = run_dir / "resolved-harbor-config.json"
    manifest_path = run_dir / "artifact-manifest.json"
    lock_copy_path = run_dir / "practical-evals.lock.json"
    images_path = run_dir / "container-images.jsonl"
    for path in (receipt_path, config_path, manifest_path, lock_copy_path, images_path):
        require(path.is_file(), f"missing practical evidence: {path}")
    receipt = json.loads(receipt_path.read_text())
    config = json.loads(config_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    require(receipt.get("format") == "bw24-practical-run-v1", f"bad receipt format: {run_dir}")
    require(receipt.get("panel") == panel, f"wrong panel in {run_dir}")
    actual_api_base = receipt.get("base_url")
    normalized_loopback_api_base(actual_api_base)
    elapsed = receipt.get("elapsed_seconds")
    require(
        receipt.get("completed_successfully") is True
        and receipt.get("harbor_exit_code") == 0
        and receipt.get("tee_exit_code") == 0
        and isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool)
        and math.isfinite(float(elapsed)) and elapsed > 0,
        f"unfinished practical receipt: {run_dir}",
    )
    artifact_bytes = receipt.get("artifact_bytes")
    require(isinstance(artifact_bytes, int) and not isinstance(artifact_bytes, bool) and artifact_bytes > 0, f"invalid artifact bytes: {run_dir}")
    require(manifest.get("artifact_bytes") == artifact_bytes, f"artifact bytes differ from manifest: {run_dir}")
    require(receipt.get("artifact_manifest_sha256") == sha256(manifest_path), f"artifact manifest hash differs: {run_dir}")
    require(receipt.get("lock_sha256") == sha256(lock_copy_path), f"lock hash differs: {run_dir}")
    require(receipt.get("container_images_sha256") == sha256(images_path), f"container image snapshot hash differs: {run_dir}")
    server_log = Path(receipt.get("server_log") or "")
    require(server_log.is_file() and receipt.get("server_log_sha256") == sha256(server_log), f"server log hash differs: {run_dir}")
    spill = receipt.get("spill_delta")
    require(isinstance(spill, dict) and set(spill) == set(SPILL_KEYS), f"invalid spill delta: {run_dir}")
    require(all(isinstance(spill[key], int) and not isinstance(spill[key], bool) and spill[key] >= 0 for key in SPILL_KEYS), f"non-monotonic spill delta: {run_dir}")
    require(spill["reads"] > 0 and spill["bytes"] > 0, f"no practical spill reads: {run_dir}")
    require(spill["errors"] == 0 and spill["short_reads"] == 0, f"practical spill failure: {run_dir}")
    for key in SHARED_RECEIPT_KEYS:
        require(receipt.get(key) is not None, f"receipt missing {key}: {run_dir}")

    arm = receipt.get("arm")
    require(isinstance(arm, str) and arm, f"missing arm: {run_dir}")
    agents = config.get("agents")
    require(isinstance(agents, list) and len(agents) == 1, f"expected one Harbor agent: {run_dir}")
    agent = agents[0]
    require(agent.get("name") == "terminus-2" and agent.get("model_name") == f"openai/{arm}", f"wrong practical agent/model: {run_dir}")
    require(config.get("n_concurrent_trials") == 1, f"wrong Harbor concurrency: {run_dir}")
    require(config.get("agent_timeout_multiplier") == 4.0, f"wrong Harbor agent timeout multiplier: {run_dir}")
    scaffold = lock["protocol"]["agent_scaffold"]
    normalized_loopback_api_base(scaffold["api_base"])
    expected_kwargs = {
        "api_base": actual_api_base, "temperature": scaffold["temperature"],
        "max_turns": scaffold["max_turns"], "parser_name": scaffold["parser_name"],
        "proactive_summarization_threshold": scaffold["proactive_summarization_threshold"],
        "enable_summarize": scaffold["enable_summarize"],
        "store_all_messages": scaffold["store_all_messages"],
        "record_terminal_session": scaffold["record_terminal_session"],
        "model_info": {
            "max_input_tokens": scaffold["max_input_tokens"],
            "max_output_tokens": scaffold["max_output_tokens"],
            "input_cost_per_token": 0, "output_cost_per_token": 0,
        },
        "llm_call_kwargs": {
            "max_tokens": scaffold["llm_call_max_tokens"],
            "timeout": scaffold["llm_call_timeout_seconds"],
        },
    }
    require(agent.get("kwargs") == expected_kwargs, f"Harbor agent kwargs differ from lock: {run_dir}")

    expected = expected_tasks(lock, panel)
    datasets = config.get("datasets")
    require(isinstance(datasets, list) and len(datasets) == 1, f"expected one Harbor dataset: {run_dir}")
    task_names = datasets[0].get("task_names")
    require(isinstance(task_names, list) and len(task_names) == len(expected), f"wrong task count in Harbor config: {run_dir}")
    require(task_names == list(expected), f"Harbor task order differs from lock: {run_dir}")
    suite = lock["swe_bench_verified"] if panel == "swe" else lock["terminal_bench_2"]
    expected_dataset_name = suite["harbor_dataset"] if panel == "swe" else suite["dataset"]
    expected_dataset_ref = suite["harbor_dataset_digest"] if panel == "swe" else suite["dataset_digest"]
    require(datasets[0].get("name") == expected_dataset_name and datasets[0].get("ref") == expected_dataset_ref, f"Harbor dataset differs from lock: {run_dir}")

    job_dir = run_dir / "jobs" / receipt["run_id"]
    job_result_path = job_dir / "result.json"
    require(job_result_path.is_file(), f"missing Harbor job result: {job_result_path}")
    job_result = json.loads(job_result_path.read_text())
    stats = job_result.get("stats", {})
    require(job_result.get("n_total_trials") == len(expected), f"wrong total trials: {run_dir}")
    rewards: dict[str, float] = {}
    raw_verifier_rewards: dict[str, float] = {}
    timeouts: dict[str, bool] = {}
    trial_paths = sorted(path for path in job_dir.iterdir() if path.is_dir())
    require(len(trial_paths) == len(expected), f"wrong number of trial directories: {run_dir}")
    for trial_dir in trial_paths:
        result_path = trial_dir / "result.json"
        require(result_path.is_file(), f"missing trial result: {trial_dir}")
        result = json.loads(result_path.read_text())
        task_name = result.get("task_name")
        require(task_name in expected, f"unexpected practical task {task_name!r}: {run_dir}")
        require(task_name not in rewards, f"duplicate practical task {task_name}: {run_dir}")
        task_id = result.get("task_id", {})
        require(task_id.get("ref") == expected[task_name], f"task digest differs for {task_name}: {run_dir}")
        exception = result.get("exception_info")
        timed_out = is_agent_timeout(exception)
        require(
            exception is None or timed_out,
            f"non-timeout trial exception for {task_name}: {run_dir}",
        )
        agent_info = result.get("agent_info", {})
        require(agent_info.get("name") == "terminus-2", f"wrong trial agent for {task_name}: {run_dir}")
        trial_model = agent_info.get("model_info", {})
        require(
            trial_model.get("name") == arm and trial_model.get("provider") == "openai",
            f"wrong trial model for {task_name}: {run_dir}",
        )
        reward = result.get("verifier_result", {}).get("rewards", {}).get("reward")
        require(isinstance(reward, (int, float)) and not isinstance(reward, bool) and math.isfinite(float(reward)) and 0 <= reward <= 1, f"invalid reward for {task_name}: {run_dir}")
        require(isinstance(result.get("started_at"), str) and isinstance(result.get("finished_at"), str), f"missing trial timestamps for {task_name}: {run_dir}")
        raw_verifier_rewards[task_name] = float(reward)
        rewards[task_name] = 0.0 if timed_out else float(reward)
        timeouts[task_name] = timed_out
    require(rewards.keys() == expected.keys(), f"practical tasks differ from lock: {run_dir}")
    rewards = {task_name: rewards[task_name] for task_name in expected}
    raw_verifier_rewards = {task_name: raw_verifier_rewards[task_name] for task_name in expected}
    timeouts = {task_name: timeouts[task_name] for task_name in expected}
    timeout_count = sum(timeouts.values())
    timeout_reward_overrides = sum(
        timeouts[task_name] and raw_verifier_rewards[task_name] != 0.0
        for task_name in expected
    )
    require(
        stats.get("n_completed_trials") == len(expected)
        and stats.get("n_errored_trials") == timeout_count
        and stats.get("n_cancelled_trials") == 0
        and stats.get("n_running_trials", 0) == 0
        and stats.get("n_pending_trials", 0) == 0
        and stats.get("n_retries") == 0,
        f"Harbor job completion/error accounting differs: {run_dir}",
    )

    return {
        "arm": arm,
        "artifact_bytes": artifact_bytes,
        "logical_model_bytes": artifact_bytes + DEFAULT_SHARED_MODEL_BYTES,
        "receipt": receipt,
        "normalized_config": normalized_harbor_config(config),
        "rewards": rewards,
        "raw_verifier_rewards": raw_verifier_rewards,
        "timeouts": timeouts,
        "timeout_count": timeout_count,
        "timeout_reward_override_count": timeout_reward_overrides,
        "mean_reward": sum(rewards.values()) / len(rewards),
        "raw_verifier_mean_reward": sum(raw_verifier_rewards.values()) / len(raw_verifier_rewards),
        "elapsed_seconds": float(elapsed),
    }


def build_report(baseline_dir: Path, candidate_dir: Path, lock_path: Path, panel: str) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text())
    baseline = load_run(baseline_dir, lock, panel)
    candidate = load_run(candidate_dir, lock, panel)
    require(baseline["arm"] != candidate["arm"], "baseline and candidate arms must differ")
    require(baseline["normalized_config"] == candidate["normalized_config"], "Harbor configs differ across arms")
    for key in SHARED_RECEIPT_KEYS:
        require(baseline["receipt"][key] == candidate["receipt"][key], f"practical receipts differ on {key}")

    wins = losses = ties = 0
    tasks = []
    for task_name in baseline["rewards"]:
        base = baseline["rewards"][task_name]
        cand = candidate["rewards"][task_name]
        delta = cand - base
        if delta > 0:
            wins += 1
        elif delta < 0:
            losses += 1
        else:
            ties += 1
        tasks.append({
            "task": task_name, "baseline_reward": base, "candidate_reward": cand,
            "delta": delta, "baseline_timed_out": baseline["timeouts"][task_name],
            "candidate_timed_out": candidate["timeouts"][task_name],
            "baseline_raw_verifier_reward": baseline["raw_verifier_rewards"][task_name],
            "candidate_raw_verifier_reward": candidate["raw_verifier_rewards"][task_name],
        })
    size_reduction = 1.0 - candidate["logical_model_bytes"] / baseline["logical_model_bytes"]
    return {
        "format": "bw24-practical-comparison-v1",
        "panel": panel,
        "n_tasks": len(tasks),
        "baseline": {key: baseline[key] for key in (
            "arm", "artifact_bytes", "logical_model_bytes", "mean_reward",
            "raw_verifier_mean_reward", "elapsed_seconds", "timeout_count",
            "timeout_reward_override_count",
        )},
        "candidate": {key: candidate[key] for key in (
            "arm", "artifact_bytes", "logical_model_bytes", "mean_reward",
            "raw_verifier_mean_reward", "elapsed_seconds", "timeout_count",
            "timeout_reward_override_count",
        )},
        "candidate_mean_delta": candidate["mean_reward"] - baseline["mean_reward"],
        "candidate_size_reduction": size_reduction,
        "paired_wins": wins, "paired_losses": losses, "paired_ties": ties,
        "exact_sign_p": exact_sign_p(wins, losses),
        "tasks": tasks,
        "note": (
            "Directional matched panel; not evidence of full-benchmark equivalence. "
            "AgentTimeoutError tasks score zero; late verifier rewards remain recorded as raw provenance."
        ),
    }


def markdown(report: dict[str, Any]) -> str:
    base, cand = report["baseline"], report["candidate"]
    lines = [
        f"# Practical {report['panel']} comparison",
        "",
        "| Arm | Logical bytes | Mean reward | Timeouts | Wall hours |",
        "|---|---:|---:|---:|---:|",
        f"| {base['arm']} | {base['logical_model_bytes']:,} | {base['mean_reward']:.3f} | {base['timeout_count']} | {base['elapsed_seconds']/3600:.2f} |",
        f"| {cand['arm']} | {cand['logical_model_bytes']:,} | {cand['mean_reward']:.3f} | {cand['timeout_count']} | {cand['elapsed_seconds']/3600:.2f} |",
        "",
        f"Candidate delta: **{report['candidate_mean_delta']:+.3f}**; size reduction: **{report['candidate_size_reduction']:.2%}**; paired W/L/T: **{report['paired_wins']}/{report['paired_losses']}/{report['paired_ties']}**; exact sign p={report['exact_sign_p']:.4f}.",
        "",
        "| Task | Baseline | Candidate | Delta |",
        "|---|---:|---:|---:|",
    ]
    for row in report["tasks"]:
        lines.append(f"| {row['task']} | {row['baseline_reward']:.3f} | {row['candidate_reward']:.3f} | {row['delta']:+.3f} |")
    lines.extend(["", report["note"]])
    return "\n".join(lines) + "\n"


def write_new(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x") as handle:
            handle.write(content)
    except FileExistsError as exc:
        raise ValueError(f"refusing to overwrite practical comparison: {path}") from exc


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="bw24-practical-summary-") as tmp:
        root = Path(tmp)
        lock = {
            "protocol": {"agent_scaffold": {
                "api_base": "http://127.0.0.1:8080/v1", "temperature": 0, "max_turns": 2,
                "parser_name": "json", "proactive_summarization_threshold": 1,
                "enable_summarize": True, "store_all_messages": True,
                "record_terminal_session": True, "max_input_tokens": 8,
                "max_output_tokens": 2, "llm_call_max_tokens": 2,
                "llm_call_timeout_seconds": 7200,
            }},
            "terminal_bench_2": {"dataset": "terminal", "dataset_digest": "digest", "tasks": [
                {"name": "terminal-bench/a", "digest": "sha256:a"},
                {"name": "terminal-bench/b", "digest": "sha256:b"},
            ]}
        }
        lock_path = root / "lock.json"
        lock_path.write_text(json.dumps(lock))

        def make_run(
            arm: str, rewards: list[float], artifact_bytes: int,
            timed_out: set[str] | None = None, port: int = 8080,
        ) -> Path:
            timed_out = timed_out or set()
            run = root / arm
            job = run / "jobs" / "run"
            job.mkdir(parents=True)
            receipt = {
                "format": "bw24-practical-run-v1", "arm": arm, "panel": "terminal",
                "dataset": "terminal@digest", "base_url": f"http://127.0.0.1:{port}/v1",
                "harbor_version": "0.18.0", "bw24_commit": "commit",
                "server_binary_sha256": "server", "declared_spill_io": "worker",
                "declared_spill_pread_depth": "16", "declared_spill_stats": "1",
                "declared_serve_spec": "0", "lock_sha256": "lock", "run_id": "run",
                "artifact_bytes": artifact_bytes, "elapsed_seconds": 10.0,
                "completed_successfully": True, "harbor_exit_code": 0, "tee_exit_code": 0,
                "spill_delta": {key: (1 if key in ("reads", "bytes") else 0) for key in SPILL_KEYS},
            }
            manifest_path = run / "artifact-manifest.json"
            manifest_path.write_text(json.dumps({"artifact_bytes": artifact_bytes}))
            server_log = run / "server.log"
            server_log.write_text("server evidence\n")
            receipt["artifact_manifest_sha256"] = sha256(manifest_path)
            receipt["server_log"] = str(server_log)
            receipt["server_log_sha256"] = sha256(server_log)
            lock_copy = run / "practical-evals.lock.json"
            lock_copy.write_text(json.dumps(lock))
            receipt["lock_sha256"] = sha256(lock_copy)
            images = run / "container-images.jsonl"
            images.write_text("{}\n")
            receipt["container_images_sha256"] = sha256(images)
            (run / "run-metadata.json").write_text(json.dumps(receipt))
            config = {
                "n_concurrent_trials": 1,
                "agent_timeout_multiplier": 4.0,
                "agents": [{"name": "terminus-2", "model_name": f"openai/{arm}", "kwargs": {
                    "api_base": f"http://127.0.0.1:{port}/v1", "temperature": 0, "max_turns": 2,
                    "parser_name": "json", "proactive_summarization_threshold": 1,
                    "enable_summarize": True, "store_all_messages": True,
                    "record_terminal_session": True,
                    "model_info": {"max_input_tokens": 8, "max_output_tokens": 2,
                                   "input_cost_per_token": 0, "output_cost_per_token": 0},
                    "llm_call_kwargs": {"max_tokens": 2, "timeout": 7200},
                }}],
                "datasets": [{"name": "terminal", "ref": "digest", "task_names": [
                    "terminal-bench/a", "terminal-bench/b"
                ]}],
            }
            (run / "resolved-harbor-config.json").write_text(json.dumps(config))
            (job / "result.json").write_text(json.dumps({
                "n_total_trials": 2, "stats": {"n_completed_trials": 2, "n_errored_trials": len(timed_out),
                "n_cancelled_trials": 0, "n_retries": 0},
            }))
            for name, digest, reward in zip(("a", "b"), ("sha256:a", "sha256:b"), rewards):
                trial = job / name
                trial.mkdir()
                (trial / "result.json").write_text(json.dumps({
                    "task_name": f"terminal-bench/{name}", "task_id": {"ref": digest},
                    "exception_info": ({
                        "exception_type": "AgentTimeoutError",
                        "exception_message": "Agent execution timed out after 60.0 seconds",
                        "exception_traceback": "harbor.trial.errors.AgentTimeoutError: timeout",
                        "occurred_at": "now",
                    } if name in timed_out else None),
                    "agent_info": {"name": "terminus-2", "model_info": {
                        "name": arm, "provider": "openai",
                    }},
                    "verifier_result": {"rewards": {"reward": reward}},
                    "started_at": "start", "finished_at": "finish",
                }))
            return run

        baseline = make_run("plain", [1, 1], 100, {"a"}, port=8080)
        candidate = make_run("candidate", [1, 1], 80, port=8082)
        report = build_report(baseline, candidate, lock_path, "terminal")
        assert report["candidate_mean_delta"] == 0.5
        assert report["paired_wins"] == 1 and report["paired_losses"] == 0
        assert report["baseline"]["timeout_count"] == 1
        assert report["baseline"]["timeout_reward_override_count"] == 1
        assert report["baseline"]["raw_verifier_mean_reward"] == 1.0
        assert report["tasks"][0]["baseline_raw_verifier_reward"] == 1.0
        assert report["tasks"][0]["baseline_reward"] == 0.0
        assert "Directional matched panel" in markdown(report)
        candidate_receipt = candidate / "run-metadata.json"
        payload = json.loads(candidate_receipt.read_text())
        payload["base_url"] = "http://example.com/v1"
        candidate_receipt.write_text(json.dumps(payload))
        try:
            build_report(baseline, candidate, lock_path, "terminal")
        except ValueError as exc:
            assert "isolated loopback" in str(exc)
        else:
            raise AssertionError("non-loopback practical endpoint was accepted")
    print("practical result summarizer self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--panel", choices=("swe", "terminal"))
    parser.add_argument("--lock", type=Path, default=Path(__file__).with_name("practical-evals.lock.json"))
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.baseline is None or args.candidate is None or args.panel is None:
        parser.error("--baseline, --candidate, and --panel are required")
    report = build_report(args.baseline, args.candidate, args.lock, args.panel)
    rendered = markdown(report)
    if args.json_out:
        write_new(args.json_out, json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.markdown_out:
        write_new(args.markdown_out, rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
