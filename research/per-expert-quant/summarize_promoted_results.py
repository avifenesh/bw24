#!/usr/bin/env python3
"""Summarize matched higher-N candidate results with uncertainty and paired tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any

from summarize_directional_results import candidate_specs, exactly_one, numeric
from summarize_hourish_results import MATH_SCORE_POLICY, MATH_SCORE_VERSIONS


# The promoted artifacts are expert overlays for the same frozen BW24 GGUF body.
# Keep this explicit so reports compare the finished logical model, not just the overlay.
DEFAULT_SHARED_MODEL_BYTES = 24_999_514_624
FULL_SHARED_RECEIPT_KEYS = (
    "suite",
    "base_url",
    "bw24_commit",
    "lm_eval_commit",
    "eval_timeout_s",
    "max_gen_toks_override",
    "num_concurrent",
    "server_binary_sha256",
    "platform",
    "nvidia_smi",
    "declared_spill_io",
    "declared_spill_pread_depth",
    "declared_spill_stats",
    "declared_serve_spec",
)
N50_SHARED_RECEIPT_KEYS = (
    "suite",
    "base_url",
    "bw24_commit",
    "lm_eval_commit",
    "eval_timeout_s",
    "max_gen_toks_override",
    "server_binary_sha256",
    "platform",
    "nvidia_smi",
)
# Full runs may shard one arm across several identical GPU/server lanes. Each receipt validates
# its own copied server log below, so the source path is intentionally lane-local rather than a
# within-arm equality key.
FULL_WITHIN_ARM_RECEIPT_KEYS = FULL_SHARED_RECEIPT_KEYS + (
    "arm",
    "model",
    "artifact_identity_sha256",
)
SPILL_COUNTER_KEYS = (
    "reads", "bytes", "errors", "short_reads", "fallbacks", "buffer_waits", "ring_full",
)
PROMOTED_MATH_TASK = "hendrycks_math500"
PROMOTED_MATH_SCORE_FORMAT = "bw24-promoted-math-score-v1"
PROMOTED_MATH_RECEIPT_FORMAT = "bw24-promoted-math-score-receipt-v1"
PROMOTED_MATH_SANDBOX = {
    "network": "none",
    "read_only_root": True,
    "capabilities": "all dropped",
    "no_new_privileges": True,
    "pids_limit": 32,
    "memory_bytes": 1024 * 1024 * 1024,
    "cpus": 1,
    "cpu_shares": 2,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evidence(paths: list[Path]) -> list[dict[str, Any]]:
    return [{"path": str(path), "sha256": sha256(path)} for path in paths]


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def frozen_suite_contract(lock: dict[str, Any]) -> dict[str, Any]:
    # eval_documents was added after the frozen directional/N=50 harness checkout solely to pin
    # later full-run counts. Every other suite field must remain byte-semantically identical.
    return {key: value for key, value in lock.items() if key != "eval_documents"}


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = successes / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - half, center + half


def exact_sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def bootstrap_delta(
    baseline: dict[str, dict[str, float]],
    candidate: dict[str, dict[str, float]],
    iterations: int = 5000,
) -> tuple[float, float]:
    rng = random.Random(20260711)
    tasks = sorted(baseline)
    samples = []
    for _ in range(iterations):
        task_means = []
        for task in tasks:
            ids = sorted(baseline[task])
            draws = [ids[rng.randrange(len(ids))] for _ in ids]
            task_means.append(sum(candidate[task][key] - baseline[task][key] for key in draws) / len(draws))
        samples.append(sum(task_means) / len(task_means))
    samples.sort()
    return samples[int(0.025 * iterations)], samples[int(0.975 * iterations)]


def select_finalist(loaded: dict[str, dict[str, Any]], arms: list[str], baseline: str) -> str:
    candidate_arms = [arm for arm in arms if arm != baseline]
    if not candidate_arms:
        raise ValueError("at least one non-baseline candidate arm is required")
    return max(
        candidate_arms,
        key=lambda arm: (loaded[arm]["macro"], -loaded[arm]["logical_model_bytes"]),
    )


def load_promoted_math_score(
    run_dir: Path, expected_count: int, lock: dict[str, Any]
) -> tuple[dict[str, Any], list[Path]]:
    score_path = run_dir / "math-score.json"
    receipt_path = run_dir / "math-score.receipt.json"
    if not score_path.is_file() or not receipt_path.is_file():
        raise ValueError(f"promoted full MATH score evidence is incomplete under {run_dir}")
    score = json.loads(score_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    if (
        score.get("format") != PROMOTED_MATH_SCORE_FORMAT
        or score.get("policy") != MATH_SCORE_POLICY
        or score.get("versions") != MATH_SCORE_VERSIONS
        or score.get("total") != expected_count
        or score.get("by_task", {}).get(PROMOTED_MATH_TASK, {}).get("total") != expected_count
        or not isinstance(score.get("passed"), int)
        or not 0 <= score["passed"] <= expected_count
        or not isinstance(score.get("samples"), list)
        or len(score["samples"]) != expected_count
    ):
        raise ValueError(f"invalid promoted MATH score: {score_path}")
    if (
        receipt.get("format") != PROMOTED_MATH_RECEIPT_FORMAT
        or Path(receipt.get("run_dir", "")).resolve() != run_dir.resolve()
        or Path(receipt.get("output", "")).resolve() != score_path.resolve()
        or receipt.get("output_sha256") != sha256(score_path)
        or receipt.get("expected_sample_count") != expected_count
        or receipt.get("suite_lock_canonical_sha256") != canonical_json_sha256(lock)
        or receipt.get("sandbox") != PROMOTED_MATH_SANDBOX
        or not isinstance(receipt.get("tool_sha256"), str)
        or len(receipt["tool_sha256"]) != 64
        or not str(receipt.get("image_id", "")).startswith("sha256:")
    ):
        raise ValueError(f"invalid promoted MATH score receipt: {receipt_path}")
    return score, [score_path, receipt_path]


def load_arm(
    out_root: Path,
    run_id: str,
    arm: str,
    specs: list[dict[str, str]],
    expected_counts: dict[str, int],
    full_run: bool,
    lock: dict[str, Any],
) -> dict[str, Any]:
    run_dir = out_root / arm / run_id
    manifest_paths = sorted(run_dir.rglob("artifact-manifest.json"))
    if not manifest_paths:
        raise ValueError(f"{arm}: no artifact manifests under {run_dir}")
    manifest_payloads = {path.read_bytes() for path in manifest_paths}
    if len(manifest_payloads) != 1:
        raise ValueError(f"{arm}: artifact manifests differ across shards")
    manifest = json.loads(manifest_payloads.pop())
    artifact_bytes = manifest.get("artifact_bytes")
    if isinstance(artifact_bytes, bool) or not isinstance(artifact_bytes, int) or artifact_bytes < 0:
        raise ValueError(f"{arm}: invalid artifact_bytes {artifact_bytes!r}")
    result_paths = sorted(run_dir.rglob("results_*.json"))
    if not result_paths:
        raise ValueError(f"{arm}: no result files under {run_dir}")
    result_payloads = [(path, json.loads(path.read_text())) for path in result_paths]
    receipt_paths = sorted(run_dir.rglob("run-metadata.json"))
    if not receipt_paths:
        raise ValueError(f"{arm}: no run receipts under {run_dir}")
    receipts = [(path, json.loads(path.read_text())) for path in receipt_paths]
    server_log_paths: list[Path] = []
    lock_paths = [path.parent / "suite.lock.json" for path in receipt_paths]
    copied_locks: list[dict[str, Any]] = []
    for path in lock_paths:
        if not path.is_file():
            raise ValueError(f"{arm}: copied suite lock missing: {path}")
        copied = json.loads(path.read_text())
        if frozen_suite_contract(copied) != frozen_suite_contract(lock):
            raise ValueError(f"{arm}: frozen suite contract differs: {path}")
        if full_run and copied.get("eval_documents") != lock.get("eval_documents"):
            raise ValueError(f"{arm}: full-run document counts differ: {path}")
        copied_locks.append(copied)
    copied_lock_hashes = {canonical_json_sha256(value) for value in copied_locks}
    if len(copied_lock_hashes) != 1:
        raise ValueError(f"{arm}: copied suite locks differ across shards")
    shared_config = None
    reference = receipts[0][1]
    if full_run:
        expected_tasks = set(expected_counts)
        observed_tasks: set[str] = set()
        for key in FULL_WITHIN_ARM_RECEIPT_KEYS:
            if reference.get(key) is None:
                raise ValueError(f"{arm}: full receipt missing {key}")
        for path, receipt in receipts:
            for key in FULL_WITHIN_ARM_RECEIPT_KEYS:
                if receipt.get(key) != reference.get(key):
                    raise ValueError(f"{arm}: receipt {path} differs on {key}")
            if receipt.get("limit") != "all":
                raise ValueError(f"{arm}: receipt {path} does not record limit=all")
            elapsed = receipt.get("elapsed_seconds")
            if (
                receipt.get("completed_successfully") is not True
                or receipt.get("evaluator_exit_code") != 0
                or receipt.get("tee_exit_code") != 0
                or isinstance(elapsed, bool)
                or not isinstance(elapsed, (int, float))
                or not math.isfinite(float(elapsed))
                or elapsed <= 0
                or not isinstance(receipt.get("started_utc"), str)
                or not isinstance(receipt.get("completed_utc"), str)
            ):
                raise ValueError(f"{arm}: receipt {path} is not a successful timed completion")
            server_log = Path(receipt.get("server_log") or "")
            if (
                not server_log.is_file()
                or receipt.get("server_log_sha256") != sha256(server_log)
            ):
                raise ValueError(f"{arm}: receipt {path} has invalid server-log evidence")
            server_log_paths.append(server_log)
            spill_delta = receipt.get("spill_delta")
            if not isinstance(spill_delta, dict) or set(spill_delta) != set(SPILL_COUNTER_KEYS):
                raise ValueError(f"{arm}: receipt {path} has invalid spill delta")
            if any(
                isinstance(spill_delta[key], bool)
                or not isinstance(spill_delta[key], int)
                or spill_delta[key] < 0
                for key in SPILL_COUNTER_KEYS
            ):
                raise ValueError(f"{arm}: receipt {path} has non-monotonic spill counters")
            if spill_delta["reads"] <= 0 or spill_delta["bytes"] <= 0:
                raise ValueError(f"{arm}: receipt {path} did not record spill reads")
            if spill_delta["errors"] != 0 or spill_delta["short_reads"] != 0:
                raise ValueError(f"{arm}: receipt {path} recorded spill I/O failure")
            tasks = receipt.get("tasks")
            if not isinstance(tasks, list) or not tasks or not all(isinstance(task, str) for task in tasks):
                raise ValueError(f"{arm}: receipt {path} has invalid tasks")
            shard_id = receipt.get("shard_id")
            if len(receipts) == 1:
                if set(tasks) != expected_tasks or shard_id is not None:
                    raise ValueError(f"{arm}: monolithic full receipt has wrong tasks or shard ID")
            else:
                if len(tasks) != 1 or shard_id != tasks[0]:
                    raise ValueError(f"{arm}: receipt {path} does not match its single task shard")
            for task in tasks:
                if task in observed_tasks:
                    raise ValueError(f"{arm}: duplicate task receipt {task}")
                observed_tasks.add(task)
        if observed_tasks != expected_tasks:
            raise ValueError(f"{arm}: receipt tasks differ from pinned suite")
        shared_config = {key: reference[key] for key in FULL_SHARED_RECEIPT_KEYS}
    else:
        if len(receipts) != 1:
            raise ValueError(f"{arm}: N=50 run must have exactly one legacy receipt")
        for key in N50_SHARED_RECEIPT_KEYS:
            if reference.get(key) is None:
                raise ValueError(f"{arm}: N=50 receipt missing {key}")
        if (
            reference.get("arm") != arm
            or reference.get("model") != arm
            or reference.get("suite") != "candidate"
            or reference.get("max_gen_toks_override") != 256
        ):
            raise ValueError(f"{arm}: N=50 receipt has wrong arm/model/suite/generation config")
        shared_config = {key: reference[key] for key in N50_SHARED_RECEIPT_KEYS}
    manifest_hash = sha256(manifest_paths[0])
    if reference.get("artifact_identity_sha256") != manifest_hash:
        raise ValueError(f"{arm}: copied manifest does not match receipt artifact identity")

    expected_limit = None if full_run else float(next(iter(expected_counts.values())))
    expected_model_args = {
        "model": arm,
        "base_url": reference["base_url"],
        "num_concurrent": reference.get("num_concurrent", 1),
        "max_retries": 3,
        "tokenized_requests": False,
        "tokenizer_backend": "none",
    }
    task_hashes: dict[str, str] = {}
    task_versions: dict[str, Any] = {}
    for path, result in result_payloads:
        config = result.get("config", {})
        limit = config.get("limit")
        if full_run:
            limit_matches = limit is None
        else:
            limit_matches = (
                isinstance(limit, (int, float))
                and not isinstance(limit, bool)
                and float(limit) == expected_limit
            )
        try:
            evaluation_seconds = float(result.get("total_evaluation_time_seconds"))
        except (TypeError, ValueError):
            evaluation_seconds = math.nan
        if (
            result.get("model_name") != arm
            or result.get("model_source") != "local-completions"
            or config.get("model") != "local-completions"
            or config.get("model_args") != expected_model_args
            or str(config.get("batch_size")) != "1"
            or config.get("gen_kwargs") != {"max_gen_toks": 256}
            or config.get("random_seed") != 0
            or config.get("numpy_seed") != 1234
            or config.get("torch_seed") != 1234
            or config.get("fewshot_seed") != 1234
            or not limit_matches
            or not math.isfinite(evaluation_seconds)
            or evaluation_seconds <= 0
        ):
            raise ValueError(f"{arm}: result configuration/completion evidence differs in {path}")
        hashes = result.get("task_hashes")
        versions = result.get("versions")
        sample_counts = result.get("n-samples")
        if not all(isinstance(value, dict) for value in (hashes, versions, sample_counts)):
            raise ValueError(f"{arm}: result lacks task provenance in {path}")
        if set(hashes) != set(versions) or set(hashes) != set(sample_counts):
            raise ValueError(f"{arm}: task provenance keys differ in {path}")
        for task, task_hash in hashes.items():
            if task not in expected_counts or task in task_hashes:
                raise ValueError(f"{arm}: unexpected or duplicate task provenance for {task}")
            counts = sample_counts[task]
            if (
                not isinstance(task_hash, str)
                or not task_hash
                or not isinstance(counts, dict)
                or counts.get("effective") != expected_counts[task]
            ):
                raise ValueError(f"{arm}: invalid task hash/count for {task}")
            task_hashes[task] = task_hash
            task_versions[task] = versions[task]
    if set(task_hashes) != set(expected_counts):
        raise ValueError(f"{arm}: result task provenance differs from pinned suite")
    math_score = None
    scorer_paths: list[Path] = []
    if full_run:
        math_score, scorer_paths = load_promoted_math_score(
            run_dir, expected_counts[PROMOTED_MATH_TASK], lock
        )
    tasks = {}
    values_by_task: dict[str, dict[str, float]] = {}
    sample_paths: list[Path] = []
    for spec in specs:
        task = spec["result_task"]
        expected_n = expected_counts[task]
        metric_key = f"{spec['metric']},{spec['filter']}"
        matching_results = [
            (path, result)
            for path, result in result_payloads
            if metric_key in result.get(spec["result_section"], {}).get(task, {})
        ]
        if len(matching_results) != 1:
            raise ValueError(
                f"{arm}/{task}: expected exactly one result across shards, found {len(matching_results)}"
            )
        _, results = matching_results[0]
        aggregate = results.get(spec["result_section"], {}).get(task, {})
        sample_path = exactly_one(sorted(run_dir.rglob(spec["sample_glob"])), f"{arm}/{task} samples")
        sample_paths.append(sample_path)
        values: dict[str, float] = {}
        if full_run and task == PROMOTED_MATH_TASK:
            assert math_score is not None
            input_files = math_score.get("input_files")
            if (
                not isinstance(input_files, list)
                or len(input_files) != 1
                or input_files[0].get("sha256") != sha256(sample_path)
            ):
                raise ValueError(f"{arm}/{task}: external score input does not match samples")
            for row in math_score["samples"]:
                if row.get("task") != task or not isinstance(row.get("doc_id"), int):
                    raise ValueError(f"{arm}/{task}: invalid external score sample")
                identity_fields = (row.get("doc_hash"), row.get("prompt_hash"), row.get("target_hash"))
                if not all(isinstance(value, str) and value for value in identity_fields):
                    raise ValueError(f"{arm}/{task}: external score sample lacks identity")
                identity = ":".join(identity_fields)
                if identity in values or not isinstance(row.get("passed"), bool):
                    raise ValueError(f"{arm}/{task}: duplicate or invalid external score sample")
                values[identity] = float(row["passed"])
            aggregate_value = math_score["passed"] / expected_n
        else:
            aggregate_value = numeric(aggregate.get(metric_key), f"{arm}/{task} aggregate")
            with sample_path.open() as handle:
                for line_number, line in enumerate(handle, 1):
                    row = json.loads(line)
                    if row.get("filter") != spec["filter"]:
                        continue
                    value = numeric(row.get(spec["metric"]), f"{sample_path}:{line_number}")
                    if value not in (0.0, 1.0):
                        raise ValueError(f"{arm}/{task}: expected binary metric, got {value}")
                    doc_hash = row.get("doc_hash")
                    prompt_hash = row.get("prompt_hash")
                    target_hash = row.get("target_hash")
                    if not all(isinstance(value, str) and value for value in (doc_hash, prompt_hash, target_hash)):
                        raise ValueError(f"{arm}/{task}: missing sample identity")
                    identity = f"{doc_hash}:{prompt_hash}:{target_hash}"
                    if identity in values:
                        raise ValueError(f"{arm}/{task}: duplicate sample identity {identity}")
                    values[identity] = value
        if len(values) != expected_n:
            raise ValueError(f"{arm}/{task}: expected N={expected_n}, found {len(values)}")
        successes = int(sum(values.values()))
        rate = successes / expected_n
        if not math.isclose(rate, aggregate_value, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{arm}/{task}: aggregate {aggregate_value} != sample mean {rate}")
        low, high = wilson(successes, expected_n)
        tasks[task] = {"successes": successes, "n": expected_n, "rate": rate, "ci95": [low, high]}
        values_by_task[task] = values
    macro = sum(task["rate"] for task in tasks.values()) / len(tasks)
    return {
        "artifact_bytes": artifact_bytes,
        "artifact_manifest_sha256": manifest_hash,
        "result_file": str(result_paths[0]) if len(result_paths) == 1 else None,
        "result_files": [str(path) for path in result_paths],
        "receipt_files": [str(path) for path in receipt_paths],
        "result_evidence": evidence(result_paths),
        "receipt_evidence": evidence(receipt_paths),
        "sample_evidence": evidence(sample_paths),
        "scorer_evidence": evidence(scorer_paths),
        "server_log_evidence": evidence(server_log_paths),
        "suite_lock_evidence": evidence(lock_paths),
        "suite_lock_canonical_sha256": copied_lock_hashes.pop(),
        "analysis_lock_canonical_sha256": canonical_json_sha256(lock),
        "task_hashes": task_hashes,
        "task_versions": task_versions,
        "shared_run_config": shared_config,
        "tasks": tasks,
        "macro": macro,
        "values": values_by_task,
    }


def build_report(
    out_root: Path,
    run_id: str,
    arms: list[str],
    baseline: str,
    expected_n: int | str,
    lock: dict[str, Any],
    shared_model_bytes: int = DEFAULT_SHARED_MODEL_BYTES,
) -> dict[str, Any]:
    specs = candidate_specs(lock)
    if expected_n == "all":
        pinned = lock.get("eval_documents", {})
        expected_counts = {}
        for spec in specs:
            task = spec["result_task"]
            count = pinned.get(task)
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError(f"{task}: missing positive eval_documents count in suite lock")
            expected_counts[task] = count
    else:
        if isinstance(expected_n, bool) or not isinstance(expected_n, int) or expected_n <= 0:
            raise ValueError(f"expected_n must be a positive integer or 'all', got {expected_n!r}")
        expected_counts = {spec["result_task"]: expected_n for spec in specs}
    full_run = expected_n == "all"
    loaded = {
        arm: load_arm(out_root, run_id, arm, specs, expected_counts, full_run, lock)
        for arm in arms
    }
    reference_config = loaded[arms[0]]["shared_run_config"]
    for arm in arms[1:]:
        if loaded[arm]["shared_run_config"] != reference_config:
            label = "full" if full_run else "N=50"
            raise ValueError(f"{arm}: {label} run configuration differs from {arms[0]}")
        if (
            loaded[arm]["task_hashes"] != loaded[arms[0]]["task_hashes"]
            or loaded[arm]["task_versions"] != loaded[arms[0]]["task_versions"]
        ):
            raise ValueError(f"{arm}: task definitions differ from {arms[0]}")
        if loaded[arm]["suite_lock_canonical_sha256"] != loaded[arms[0]]["suite_lock_canonical_sha256"]:
            raise ValueError(f"{arm}: copied suite lock differs from {arms[0]}")
    for task in (spec["result_task"] for spec in specs):
        identities = {arm: set(data["values"][task]) for arm, data in loaded.items()}
        first = identities[arms[0]]
        if any(value != first for value in identities.values()):
            raise ValueError(f"{task}: sample identities differ across arms")
    paired = {}
    base_values = loaded[baseline]["values"]
    for arm in arms:
        if arm == baseline:
            continue
        candidate = loaded[arm]["values"]
        wins = losses = 0
        for task in base_values:
            for identity, base in base_values[task].items():
                value = candidate[task][identity]
                wins += value > base
                losses += value < base
        delta = loaded[arm]["macro"] - loaded[baseline]["macro"]
        low, high = bootstrap_delta(base_values, candidate)
        paired[arm] = {
            "macro_delta": delta,
            "bootstrap_ci95": [low, high],
            "paired_wins": wins,
            "paired_losses": losses,
            "exact_sign_p": exact_sign_p(wins, losses),
        }
    for data in loaded.values():
        data.pop("values")
        data["logical_model_bytes"] = shared_model_bytes + data["artifact_bytes"]
    baseline_bytes = loaded[baseline]["logical_model_bytes"]
    for data in loaded.values():
        data["size_reduction_vs_baseline"] = 1.0 - data["logical_model_bytes"] / baseline_bytes
    pareto_arms = []
    for arm, data in loaded.items():
        dominated = any(
            other["logical_model_bytes"] <= data["logical_model_bytes"]
            and other["macro"] >= data["macro"]
            and (
                other["logical_model_bytes"] < data["logical_model_bytes"]
                or other["macro"] > data["macro"]
            )
            for other_arm, other in loaded.items()
            if other_arm != arm
        )
        if not dominated:
            pareto_arms.append(arm)
    # Freeze the N=50 down-selection rule before looking at the result. Quality is primary; exact
    # point-estimate ties go to the smaller finished model. The baseline is always retained for the
    # full comparison, and the uncertainty fields remain separate so this choice cannot be read as
    # a claim of equivalence or statistically proven superiority.
    selected_finalist = select_finalist(loaded, arms, baseline)
    return {
        "format": "bw24-promoted-candidate-v1",
        "run_id": run_id,
        "n_per_task": expected_n if expected_n != "all" else expected_counts,
        "documents_per_arm": sum(expected_counts.values()),
        "baseline": baseline,
        "shared_model_bytes": shared_model_bytes,
        "arms": loaded,
        "paired_vs_baseline": paired,
        "point_estimate_pareto_arms": pareto_arms,
        "selection": {
            "rule": "highest candidate macro point estimate; exact tie chooses smaller logical model",
            "selected_finalist": selected_finalist,
            "full_eval_arms": [baseline, selected_finalist],
            "note": "Directional down-selection only; uncertainty is reported separately and this is not an equivalence claim.",
        },
        "tasks": [{"task": spec["result_task"], "label": spec["label"]} for spec in specs],
    }


def markdown(report: dict[str, Any]) -> str:
    n_per_task = report["n_per_task"]
    if isinstance(n_per_task, int):
        sample_description = f"N={n_per_task} per task"
    else:
        sample_description = f"full pinned N={report['documents_per_arm']:,} per arm"
    lines = [
        "# Promoted-arm matched evaluation",
        "",
        f"Run ID: `{report['run_id']}` · {sample_description} · baseline `{report['baseline']}`",
        "",
        "| Arm | Logical size | Reduction vs baseline | Expert overlay bytes | Macro accuracy | Delta vs baseline | Paired W/L | 95% paired-bootstrap CI | Exact sign p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, data in report["arms"].items():
        pair = report["paired_vs_baseline"].get(arm)
        if pair is None:
            cells = [
                arm, f"{data['logical_model_bytes']:,} ({data['logical_model_bytes'] / 2**30:.3f} GiB)",
                f"{data['size_reduction_vs_baseline']:.1%}", f"{data['artifact_bytes']:,}",
                f"{data['macro']:.1%}", "—", "—", "—", "—",
            ]
        else:
            lo, hi = pair["bootstrap_ci95"]
            cells = [
                arm, f"{data['logical_model_bytes']:,} ({data['logical_model_bytes'] / 2**30:.3f} GiB)",
                f"{data['size_reduction_vs_baseline']:.1%}", f"{data['artifact_bytes']:,}", f"{data['macro']:.1%}",
                f"{pair['macro_delta']:+.1%}", f"{pair['paired_wins']}/{pair['paired_losses']}",
                f"[{lo:+.1%}, {hi:+.1%}]", f"{pair['exact_sign_p']:.4f}",
            ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "Point-estimate quality/size Pareto frontier: "
        + ", ".join(f"`{arm}`" for arm in report["point_estimate_pareto_arms"])
        + ". This is descriptive; use the paired intervals above for uncertainty.",
        "",
        f"Selected finalist: **`{report['selection']['selected_finalist']}`**. "
        f"Full comparison arms: **`{'` and `'.join(report['selection']['full_eval_arms'])}`**.",
        "",
        report["selection"]["note"],
        "",
        "## Per-task accuracy (Wilson 95% CI)",
        "",
    ]
    for task in report["tasks"]:
        lines += [f"### {task['label']}", "", "| Arm | Correct/N | Accuracy | Wilson 95% CI |", "|---|---:|---:|---:|"]
        for arm, data in report["arms"].items():
            value = data["tasks"][task["task"]]
            lines.append(
                f"| {arm} | {value['successes']}/{value['n']} | {value['rate']:.1%} | "
                f"[{value['ci95'][0]:.1%}, {value['ci95'][1]:.1%}] |"
            )
        lines.append("")
    lines.append("All comparisons use identical sample hashes across arms. The paired bootstrap is stratified by task.")
    return "\n".join(lines) + "\n"


def write_report_new(out: Path, markdown_text: str, json_text: str) -> tuple[Path, Path]:
    json_out = out.with_suffix(".json")
    if out == json_out:
        raise ValueError("markdown and JSON report paths must be distinct")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary: list[Path] = []
    linked: list[Path] = []
    try:
        for content in (markdown_text, json_text):
            with tempfile.NamedTemporaryFile(
                mode="w", dir=out.parent, prefix=".promoted-results-", delete=False
            ) as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
                temporary.append(Path(handle.name))
            os.chmod(temporary[-1], 0o644)
        for source, destination in zip(temporary, (out, json_out)):
            os.link(source, destination)
            linked.append(destination)
    except FileExistsError as exc:
        for path in linked:
            path.unlink(missing_ok=True)
        raise ValueError(f"refusing to overwrite promoted report: {out} / {json_out}") from exc
    except Exception:
        for path in linked:
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)
    return out, json_out


def self_test(lock: dict[str, Any]) -> None:
    selection_fixture = {
        "plain": {"macro": 0.8, "logical_model_bytes": 200},
        "larger": {"macro": 0.7, "logical_model_bytes": 120},
        "smaller": {"macro": 0.7, "logical_model_bytes": 100},
        "better": {"macro": 0.71, "logical_model_bytes": 150},
    }
    assert select_finalist(selection_fixture, ["plain", "larger", "smaller"], "plain") == "smaller"
    assert select_finalist(selection_fixture, ["plain", "smaller", "better"], "plain") == "better"
    specs = candidate_specs(lock)
    with tempfile.TemporaryDirectory(prefix="bw24-promoted-summary-") as tmp:
        root = Path(tmp)
        arms = ["plain_quant", "candidate"]
        for arm_index, arm in enumerate(arms):
            run_dir = root / arm / "fixture"
            model_dir = run_dir / arm
            model_dir.mkdir(parents=True)
            manifest_path = run_dir / "artifact-manifest.json"
            manifest_path.write_text(json.dumps({"artifact_bytes": 100 + arm_index}))
            (run_dir / "suite.lock.json").write_text(json.dumps(lock))
            server_log_path = run_dir / "server.log"
            server_log_path.write_text(f"server evidence for {arm}\n")
            receipt = {
                "suite": "candidate", "base_url": "http://127.0.0.1:8080/v1/completions",
                "bw24_commit": "bw24", "lm_eval_commit": "harness", "eval_timeout_s": 100,
                "max_gen_toks_override": 256, "num_concurrent": 1,
                "server_binary_sha256": "server", "platform": "linux", "nvidia_smi": "gpu",
                "declared_spill_io": "worker",
                "declared_spill_pread_depth": "8", "declared_spill_stats": "1",
                "declared_serve_spec": "0", "arm": arm, "model": arm,
                "artifact_identity_sha256": sha256(manifest_path), "limit": "all",
                "tasks": [spec["result_task"] for spec in specs], "shard_id": None,
                "started_utc": "start", "completed_utc": "end", "elapsed_seconds": 1.0,
                "evaluator_exit_code": 0, "tee_exit_code": 0, "completed_successfully": True,
                "server_log_source": f"source-server-{arm}.log",
                "server_log": str(server_log_path),
                "server_log_sha256": sha256(server_log_path),
                "spill_delta": {
                    "reads": 1, "bytes": 1, "errors": 0, "short_reads": 0,
                    "fallbacks": 0, "buffer_waits": 0, "ring_full": 0,
                },
            }
            (run_dir / "run-metadata.json").write_text(json.dumps(receipt))
            results = {
                "model_name": arm,
                "model_source": "local-completions",
                "total_evaluation_time_seconds": "1.0",
                "config": {
                    "model": "local-completions",
                    "model_args": {
                        "model": arm,
                        "base_url": "http://127.0.0.1:8080/v1/completions",
                        "num_concurrent": 1,
                        "max_retries": 3,
                        "tokenized_requests": False,
                        "tokenizer_backend": "none",
                    },
                    "batch_size": "1", "limit": 4.0,
                    "gen_kwargs": {"max_gen_toks": 256},
                    "random_seed": 0, "numpy_seed": 1234,
                    "torch_seed": 1234, "fewshot_seed": 1234,
                },
                "task_hashes": {}, "versions": {}, "n-samples": {},
            }
            for task_index, spec in enumerate(specs):
                task = spec["result_task"]
                results["task_hashes"][task] = f"task-hash-{task_index}"
                results["versions"][task] = task_index + 1
                results["n-samples"][task] = {"original": 100 + task_index, "effective": 4}
                rows = []
                for i in range(4):
                    value = float((i + task_index + arm_index) % 3 == 0)
                    rows.append({
                        "filter": spec["filter"], spec["metric"]: value,
                        "doc_hash": f"doc-{task_index}-{i}",
                        "prompt_hash": f"prompt-{task_index}-{i}",
                        "target_hash": f"target-{task_index}-{i}",
                    })
                results.setdefault(spec["result_section"], {})[spec["result_task"]] = {
                    f"{spec['metric']},{spec['filter']}": sum(row[spec["metric"]] for row in rows) / 4,
                }
                sample = model_dir / spec["sample_glob"].replace("*", "fixture")
                sample.write_text("".join(json.dumps(row) + "\n" for row in rows))
            (model_dir / "results_fixture.json").write_text(json.dumps(results))
        report = build_report(root, "fixture", arms, "plain_quant", 4, lock)
        assert report["n_per_task"] == 4
        assert report["documents_per_arm"] == 4 * len(specs)
        assert report["paired_vs_baseline"]["candidate"]["paired_wins"] > 0
        assert report["arms"]["plain_quant"]["logical_model_bytes"] == DEFAULT_SHARED_MODEL_BYTES + 100
        assert report["arms"]["candidate"]["size_reduction_vs_baseline"] < 0
        assert report["point_estimate_pareto_arms"] == ["plain_quant"]
        assert report["selection"]["selected_finalist"] == "candidate"
        assert report["selection"]["full_eval_arms"] == ["plain_quant", "candidate"]
        assert "Selected finalist" in markdown(report)
        assert "Wilson 95% CI" in markdown(report)
        report_out = root / "reports" / "promoted-results.md"
        write_report_new(report_out, markdown(report), json.dumps(report))
        original_markdown = report_out.read_text()
        try:
            write_report_new(report_out, "replacement", "replacement")
        except ValueError as exc:
            assert "refusing to overwrite" in str(exc)
        else:
            raise AssertionError("promoted report was overwritten")
        assert report_out.read_text() == original_markdown
        candidate_result_path = root / "candidate" / "fixture" / "candidate" / "results_fixture.json"
        candidate_result = json.loads(candidate_result_path.read_text())
        candidate_result["config"]["model_args"]["max_retries"] = 4
        candidate_result_path.write_text(json.dumps(candidate_result))
        try:
            build_report(root, "fixture", arms, "plain_quant", 4, lock)
        except ValueError as exc:
            assert "result configuration/completion evidence differs" in str(exc)
        else:
            raise AssertionError("mismatched lm-eval config was accepted")
        candidate_result["config"]["model_args"]["max_retries"] = 3
        candidate_result_path.write_text(json.dumps(candidate_result))
        full_lock = dict(lock)
        full_lock["eval_documents"] = {spec["result_task"]: 4 for spec in specs}
        math_spec = next(spec for spec in specs if spec["result_task"] == PROMOTED_MATH_TASK)
        for arm in arms:
            run_dir = root / arm / "fixture"
            result_path = run_dir / arm / "results_fixture.json"
            result = json.loads(result_path.read_text())
            result["config"]["limit"] = None
            # The raw aggregate is deliberately poisoned. Full reporting must use only the
            # independently rescored immutable MATH evidence below.
            result[math_spec["result_section"]][PROMOTED_MATH_TASK][
                f"{math_spec['metric']},{math_spec['filter']}"
            ] = 0.123456
            result_path.write_text(json.dumps(result))
            (run_dir / "suite.lock.json").write_text(json.dumps(full_lock))
            sample_path = exactly_one(
                sorted((run_dir / arm).glob(math_spec["sample_glob"])),
                f"{arm} fixture MATH samples",
            )
            raw_rows = [json.loads(line) for line in sample_path.read_text().splitlines()]
            scored_rows = [{
                "task": PROMOTED_MATH_TASK,
                "doc_id": index,
                "doc_hash": row["doc_hash"],
                "prompt_hash": row["prompt_hash"],
                "target_hash": row["target_hash"],
                "answer": "fixture",
                "normalized_answer": "fixture",
                "passed": bool(row[math_spec["metric"]]),
                "method": "fixture",
            } for index, row in enumerate(raw_rows)]
            math_score = {
                "format": PROMOTED_MATH_SCORE_FORMAT,
                "policy": MATH_SCORE_POLICY,
                "versions": MATH_SCORE_VERSIONS,
                "input_files": [{"path": str(sample_path), "sha256": sha256(sample_path)}],
                "by_task": {PROMOTED_MATH_TASK: {
                    "passed": sum(row["passed"] for row in scored_rows), "total": 4,
                }},
                "passed": sum(row["passed"] for row in scored_rows),
                "total": 4,
                "samples": scored_rows,
            }
            math_score_path = run_dir / "math-score.json"
            math_score_path.write_text(json.dumps(math_score))
            math_receipt = {
                "format": PROMOTED_MATH_RECEIPT_FORMAT,
                "run_dir": str(run_dir), "output": str(math_score_path),
                "output_sha256": sha256(math_score_path),
                "image": "fixture", "image_id": "sha256:fixture",
                "tool_sha256": "f" * 64,
                "suite_lock_sha256": "fixture",
                "suite_lock_canonical_sha256": canonical_json_sha256(full_lock),
                "expected_sample_count": 4,
                "sandbox": PROMOTED_MATH_SANDBOX,
            }
            (run_dir / "math-score.receipt.json").write_text(json.dumps(math_receipt))
        full_report = build_report(root, "fixture", arms, "plain_quant", "all", full_lock)
        assert full_report["documents_per_arm"] == 4 * len(specs)
        assert isinstance(full_report["n_per_task"], dict)
        assert "full pinned" in markdown(full_report)
        candidate_math = root / "candidate" / "fixture" / "math-score.json"
        original_candidate_math = candidate_math.read_text()
        candidate_math.write_text(original_candidate_math + "\n")
        try:
            build_report(root, "fixture", arms, "plain_quant", "all", full_lock)
        except ValueError as exc:
            assert "invalid promoted MATH score receipt" in str(exc)
        else:
            raise AssertionError("tampered promoted MATH score was accepted")
        candidate_math.write_text(original_candidate_math)
        plain_server_log = root / "plain_quant" / "fixture" / "server.log"
        original_server_log = plain_server_log.read_text()
        plain_server_log.write_text(original_server_log + "tampered\n")
        try:
            build_report(root, "fixture", arms, "plain_quant", "all", full_lock)
        except ValueError as exc:
            assert "invalid server-log evidence" in str(exc)
        else:
            raise AssertionError("tampered full-run server log was accepted")
        plain_server_log.write_text(original_server_log)
        candidate_run = root / "candidate" / "fixture"
        candidate_model = candidate_run / "candidate"
        candidate_results_path = candidate_model / "results_fixture.json"
        candidate_results = json.loads(candidate_results_path.read_text())
        manifest_text = (candidate_run / "artifact-manifest.json").read_text()
        candidate_receipt_path = candidate_run / "run-metadata.json"
        candidate_receipt = json.loads(candidate_receipt_path.read_text())
        for spec in specs:
            task = spec["result_task"]
            shard = candidate_run / "shards" / task
            shard_model = shard / "candidate"
            shard_model.mkdir(parents=True)
            sample = exactly_one(sorted(candidate_model.glob(spec["sample_glob"])), task)
            sample.rename(shard_model / sample.name)
            section = spec["result_section"]
            shard_results = {
                key: value for key, value in candidate_results.items()
                if key not in {spec["result_section"] for spec in specs}
            }
            for key in ("task_hashes", "versions", "n-samples"):
                shard_results[key] = {task: candidate_results[key][task]}
            shard_results[section] = {task: candidate_results[section][task]}
            (shard_model / f"results_{task}.json").write_text(json.dumps(shard_results))
            (shard / "artifact-manifest.json").write_text(manifest_text)
            (shard / "suite.lock.json").write_text(json.dumps(full_lock))
            shard_server_log = shard / "server.log"
            shard_server_log.write_text(f"server evidence for candidate/{task}\n")
            shard_receipt = dict(
                candidate_receipt,
                tasks=[task],
                shard_id=task,
                server_log_source=str(shard_server_log),
                server_log=str(shard_server_log),
                server_log_sha256=sha256(shard_server_log),
            )
            (shard / "run-metadata.json").write_text(json.dumps(shard_receipt))
        candidate_results_path.unlink()
        (candidate_run / "artifact-manifest.json").unlink()
        candidate_receipt_path.unlink()
        sharded_report = build_report(root, "fixture", arms, "plain_quant", "all", full_lock)
        assert len(sharded_report["arms"]["candidate"]["result_files"]) == len(specs)
        print("promoted result summarizer self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--arms", help="comma-separated arm names")
    parser.add_argument("--baseline", default="plain_quant")
    parser.add_argument("--expected-n", default="50", help="positive integer per task, or 'all'")
    parser.add_argument("--shared-model-bytes", type=int, default=DEFAULT_SHARED_MODEL_BYTES)
    parser.add_argument("--lock", type=Path, default=Path(__file__).with_name("suite.lock.json"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text())
    if args.self_test:
        self_test(lock)
        return 0
    if args.out_root is None or not args.run_id or not args.arms:
        parser.error("--out-root, --run-id, and --arms are required")
    arms = [arm for arm in args.arms.split(",") if arm]
    if args.baseline not in arms:
        parser.error("--baseline must be present in --arms")
    if args.shared_model_bytes < 0:
        parser.error("--shared-model-bytes must be non-negative")
    if args.expected_n == "all":
        expected_n: int | str = "all"
    else:
        try:
            expected_n = int(args.expected_n)
        except ValueError:
            parser.error("--expected-n must be a positive integer or 'all'")
        if expected_n <= 0:
            parser.error("--expected-n must be a positive integer or 'all'")
    report = build_report(
        args.out_root, args.run_id, arms, args.baseline, expected_n, lock, args.shared_model_bytes
    )
    out = args.out or args.out_root / "_runs" / args.run_id / "promoted-results.md"
    json_out = out.with_suffix(".json")
    write_report_new(
        out,
        markdown(report),
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    print(f"wrote {out} and {json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
