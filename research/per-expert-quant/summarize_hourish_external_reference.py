#!/usr/bin/env python3
"""Strictly compare the pinned MLX REAP50 reference to a completed bw24 arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Any

import summarize_hourish_results as core


LOCK_SHA256 = "dd85692e943bd58a348c7cfbc271508160e53178af43cef976fcebd4fde4c131"
EXPECTED_BASE_URL = "http://127.0.0.1:8080/v1/completions"
EXPECTED_MODEL = "default_model"


def validate_external_receipt(
    path: pathlib.Path,
    arm: str,
    task: str,
    panel: dict[str, Any],
    manifest_sha: str,
    runtime_sha: str,
) -> dict[str, Any]:
    receipt = json.loads(path.read_text())
    required = {
        "arm": arm,
        "model": EXPECTED_MODEL,
        "runtime_kind": "external_openai",
        "tasks": [task],
        "shard_id": task,
        "samples": {task: panel["samples"][task]},
        "panel_lock_sha256": core.PANEL_SHA256,
        "predict_only": task == "humaneval_instruct",
        "max_gen_toks_override": panel["max_gen_toks"][task],
        "num_concurrent": 1,
        "declared_spill_io": None,
        "declared_spill_pread_depth": None,
        "declared_spill_stats": None,
        "declared_serve_spec": None,
        "artifact_identity_sha256": manifest_sha,
        "runtime_identity_sha256": runtime_sha,
        "completed_successfully": True,
        "evaluator_exit_code": 0,
        "tee_exit_code": 0,
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise ValueError(
                f"{arm}/{task}: receipt {key}={receipt.get(key)!r}, expected {expected!r}"
            )
    if receipt.get("base_url") != EXPECTED_BASE_URL:
        raise ValueError(f"{arm}/{task}: unexpected endpoint")
    elapsed = core.finite_number(receipt.get("elapsed_seconds"), f"{arm}/{task} elapsed")
    if elapsed <= 0:
        raise ValueError(f"{arm}/{task}: elapsed time must be positive")
    server_log = pathlib.Path(receipt.get("server_log") or "")
    if not server_log.is_file() or receipt.get("server_log_sha256") != core.sha256(server_log):
        raise ValueError(f"{arm}/{task}: invalid copied server log")
    if not isinstance(receipt.get("server_binary_sha256"), str) or not receipt["server_binary_sha256"]:
        raise ValueError(f"{arm}/{task}: missing external server binary hash")
    return receipt


def validate_external_result(
    result: dict[str, Any], arm: str, task: str, panel: dict[str, Any]
) -> None:
    expected_model_args = {
        "model": EXPECTED_MODEL,
        "base_url": EXPECTED_BASE_URL,
        "num_concurrent": 1,
        "max_retries": 3,
        "tokenized_requests": False,
        "tokenizer_backend": "none",
    }
    expected = {
        "model": "local-completions",
        "model_args": expected_model_args,
        "batch_size": "1",
        "gen_kwargs": {"max_gen_toks": panel["max_gen_toks"][task]},
        "random_seed": 0,
        "numpy_seed": 1234,
        "torch_seed": 1234,
        "fewshot_seed": 1234,
    }
    if result.get("model_name") != EXPECTED_MODEL or result.get("model_source") != "local-completions":
        raise ValueError(f"{arm}/{task}: wrong external model identity")
    for key, expected_value in expected.items():
        if result.get("config", {}).get(key) != expected_value:
            raise ValueError(f"{arm}/{task}: result config differs on {key}")
    if result.get("n-samples", {}).get(task, {}).get("effective") != panel["task_counts"][task]:
        raise ValueError(f"{arm}/{task}: wrong effective sample count")
    if set(result.get("task_hashes", {})) != {task} or set(result.get("versions", {})) != {task}:
        raise ValueError(f"{arm}/{task}: missing task provenance")
    if core.finite_number(
        result.get("total_evaluation_time_seconds"), f"{arm}/{task} evaluation time"
    ) <= 0:
        raise ValueError(f"{arm}/{task}: non-positive evaluation time")


def load_external_arm(
    out_root: pathlib.Path,
    run_id: str,
    arm: str,
    panel: dict[str, Any],
    lock: dict[str, Any],
) -> dict[str, Any]:
    run_dir = out_root / arm / run_id
    if not run_dir.is_dir():
        raise ValueError(f"{arm}: missing run directory {run_dir}")
    shard_dirs = {path.name: path for path in (run_dir / "shards").iterdir() if path.is_dir()}
    if set(shard_dirs) != set(core.TASKS):
        raise ValueError(f"{arm}: shard set differs from panel")

    manifest_payloads: set[bytes] = set()
    runtime_payloads: set[bytes] = set()
    suite_payloads: set[bytes] = set()
    shared_config = None
    task_hashes: dict[str, str] = {}
    result_by_task: dict[str, dict[str, Any]] = {}
    evidence: list[pathlib.Path] = []
    for task, shard_dir in shard_dirs.items():
        panel_copy = shard_dir / "panel.lock.json"
        suite_copy = shard_dir / "suite.lock.json"
        manifest_path = shard_dir / "artifact-manifest.json"
        runtime_path = shard_dir / "runtime-identity.json"
        metadata_path = shard_dir / "run-metadata.json"
        if core.sha256(panel_copy) != core.PANEL_SHA256:
            raise ValueError(f"{arm}/{task}: copied panel lock differs")
        manifest_payloads.add(manifest_path.read_bytes())
        runtime_payloads.add(runtime_path.read_bytes())
        suite_payloads.add(suite_copy.read_bytes())
        receipt = validate_external_receipt(
            metadata_path,
            arm,
            task,
            panel,
            core.sha256(manifest_path),
            core.sha256(runtime_path),
        )
        comparable = {
            key: receipt.get(key)
            for key in (
                "bw24_commit",
                "lm_eval_commit",
                "base_url",
                "num_concurrent",
                "runtime_kind",
                "server_binary_sha256",
                "runtime_identity_sha256",
            )
        }
        if shared_config is None:
            shared_config = comparable
        elif comparable != shared_config:
            raise ValueError(f"{arm}/{task}: shared external configuration differs")
        result_path = core.exactly_one(
            sorted(shard_dir.rglob("results_*.json")), f"{arm}/{task} result"
        )
        result = json.loads(result_path.read_text())
        validate_external_result(result, arm, task, panel)
        task_hashes[task] = result["task_hashes"][task]
        result_by_task[task] = result
        evidence.extend((panel_copy, suite_copy, manifest_path, runtime_path, metadata_path, result_path))

    if len(manifest_payloads) != 1 or len(runtime_payloads) != 1 or len(suite_payloads) != 1:
        raise ValueError(f"{arm}: identity payloads differ across shards")
    manifest = json.loads(manifest_payloads.pop())
    runtime = json.loads(runtime_payloads.pop())
    if (
        manifest.get("model_repo") != lock["model"]["repo"]
        or manifest.get("model_revision") != lock["model"]["revision"]
        or manifest.get("artifact_bytes") != lock["model"]["repo_storage_bytes"]
        or runtime.get("runtime_repo") != lock["runtime"]["repo"]
        or runtime.get("runtime_revision") != lock["runtime"]["revision"]
        or runtime.get("draft_model") not in (None, False)
    ):
        raise ValueError(f"{arm}: external artifact or runtime differs from the lock")

    tasks: dict[str, dict[str, Any]] = {}
    values: dict[str, dict[str, float]] = {}
    code_task, code_values, code_evidence, code_scorer = core.load_code_task(run_dir, arm, panel)
    tasks["humaneval_instruct"] = code_task
    values["humaneval_instruct"] = code_values
    evidence.extend(code_evidence)
    for task in core.TASKS:
        if task == "humaneval_instruct":
            continue
        task_result, task_values, sample_path = core.load_regular_task(
            run_dir, arm, task, result_by_task[task], panel
        )
        tasks[task] = task_result
        values[task] = task_values
        evidence.append(sample_path)
    total_correct = sum(row["successes"] for row in tasks.values())
    total_questions = sum(row["n"] for row in tasks.values())
    return {
        "run_dir": str(run_dir),
        "artifact_bytes": manifest["artifact_bytes"],
        "artifact_gib": manifest["artifact_bytes"] / (1 << 30),
        "tasks": tasks,
        "domain_macro": sum(row["rate"] for row in tasks.values()) / len(tasks),
        "total_correct": total_correct,
        "total_questions": total_questions,
        "question_weighted": total_correct / total_questions,
        "values": values,
        "task_hashes": task_hashes,
        "suite_lock_sha256": hashlib.sha256(suite_payloads.pop()).hexdigest(),
        "code_scorer": code_scorer,
        "shared_config": shared_config,
        "evidence": [
            {"path": str(path), "sha256": core.sha256(path)} for path in sorted(set(evidence))
        ],
    }


def build_report(
    baseline_root: pathlib.Path,
    baseline_run_id: str,
    baseline_arm: str,
    external_root: pathlib.Path,
    external_run_id: str,
    external_arm: str,
    panel_path: pathlib.Path,
    lock_path: pathlib.Path,
) -> dict[str, Any]:
    if core.sha256(panel_path) != core.PANEL_SHA256 or core.sha256(lock_path) != LOCK_SHA256:
        raise ValueError("analysis lock hash differs")
    panel = json.loads(panel_path.read_text())
    lock = json.loads(lock_path.read_text())
    baseline = core.load_arm(baseline_root, baseline_run_id, baseline_arm, panel)
    external = load_external_arm(external_root, external_run_id, external_arm, panel, lock)

    comparison = paired_comparison(baseline, external)
    for data in (baseline, external):
        data.pop("values")
    return {
        "format": "bw24-hourish-external-reference-comparison-v1",
        "purpose": "paired quality reference; excluded from NVIDIA artifact-size Pareto",
        "panel_lock_sha256": core.PANEL_SHA256,
        "external_lock_sha256": LOCK_SHA256,
        "baseline": {"arm": baseline_arm, "run_id": baseline_run_id, **baseline},
        "external": {"arm": external_arm, "run_id": external_run_id, **external},
        "comparison_to_baseline": comparison,
        "included_in_nvidia_size_pareto": False,
    }


def paired_comparison(
    baseline: dict[str, Any], external: dict[str, Any]
) -> dict[str, Any]:
    if external["task_hashes"] != baseline["task_hashes"]:
        raise ValueError("external task definitions differ from baseline")
    if external["suite_lock_sha256"] != baseline["suite_lock_sha256"]:
        raise ValueError("external suite lock differs from baseline")
    if external["code_scorer"]["tool_sha256"] != baseline["code_scorer"]["tool_sha256"]:
        raise ValueError("external code scorer tool differs from baseline")
    wins = losses = ties = 0
    for task in core.TASKS:
        if set(external["values"][task]) != set(baseline["values"][task]):
            raise ValueError(f"external paired identities differ for {task}")
        for identity, base_value in baseline["values"][task].items():
            delta = external["values"][task][identity] - base_value
            wins += delta > 0
            losses += delta < 0
            ties += delta == 0
    return {
        "domain_macro_delta": external["domain_macro"] - baseline["domain_macro"],
        "domain_macro_delta_bootstrap_ci95": core.bootstrap_domain_macro_delta(
            baseline["values"], external["values"]
        ),
        "question_weighted_delta": external["question_weighted"] - baseline["question_weighted"],
        "paired_wins": wins,
        "paired_losses": losses,
        "paired_ties": ties,
        "paired_exact_sign_p": core.exact_sign_p(wins, losses),
    }


def main() -> None:
    here = pathlib.Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=pathlib.Path)
    parser.add_argument("--baseline-run-id")
    parser.add_argument("--baseline-arm", default="plain_quant")
    parser.add_argument("--external-root", type=pathlib.Path)
    parser.add_argument("--external-run-id")
    parser.add_argument("--external-arm", default="mlx_reap50_reference")
    parser.add_argument("--panel-lock", type=pathlib.Path, default=here / "hourish-panel.lock.json")
    parser.add_argument(
        "--external-lock", type=pathlib.Path, default=here / "mlx-reap50-reference.lock.json"
    )
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        assert core.sha256(args.external_lock) == LOCK_SHA256
        assert core.exact_sign_p(3, 0) == 0.25
        task_hashes = {task: f"hash-{task}" for task in core.TASKS}
        baseline_values = {task: {f"sample-{task}": 0.0} for task in core.TASKS}
        external_values = {task: dict(values) for task, values in baseline_values.items()}
        external_values["humaneval_instruct"]["sample-humaneval_instruct"] = 1.0
        shared = {
            "task_hashes": task_hashes,
            "suite_lock_sha256": "suite",
            "code_scorer": {"tool_sha256": "scorer"},
        }
        comparison = paired_comparison(
            {
                **shared,
                "values": baseline_values,
                "domain_macro": 0.0,
                "question_weighted": 0.0,
            },
            {
                **shared,
                "values": external_values,
                "domain_macro": 0.25,
                "question_weighted": 0.25,
            },
        )
        assert comparison["paired_wins"] == 1
        assert comparison["paired_losses"] == 0
        assert comparison["paired_ties"] == 3
        assert comparison["domain_macro_delta"] == 0.25
        try:
            paired_comparison(
                {
                    **shared,
                    "values": baseline_values,
                    "domain_macro": 0.0,
                    "question_weighted": 0.0,
                },
                {
                    **shared,
                    "task_hashes": {**task_hashes, "hendrycks_math500": "wrong"},
                    "values": external_values,
                    "domain_macro": 0.25,
                    "question_weighted": 0.25,
                },
            )
        except ValueError as exc:
            assert "task definitions" in str(exc)
        else:
            raise AssertionError("task-hash mismatch was not rejected")
        print("hourish external reference summarizer self-test: PASS")
        return
    if not all((args.baseline_root, args.baseline_run_id, args.external_root, args.external_run_id)):
        parser.error("baseline and external roots/run IDs are required")
    report = build_report(
        args.baseline_root,
        args.baseline_run_id,
        args.baseline_arm,
        args.external_root,
        args.external_run_id,
        args.external_arm,
        args.panel_lock,
        args.external_lock,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite {args.output}")
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
