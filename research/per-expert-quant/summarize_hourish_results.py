#!/usr/bin/env python3
"""Validate and summarize a frozen matched capability screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import random
import re
import tempfile
from typing import Any

from validate_capability_panel import SHA256_RE, validate_panel


# Kept for the existing MLX reference analyzer. New matched screens derive the
# active panel hash from their validated frozen lock.
PANEL_SHA256 = "770135c560b590844fcf09418e965a42ecb876a5eb9566564e19e8fb02bb6ce1"
SHARED_MODEL_BYTES = 24_999_514_624
SERVER_SHA256 = "7acee929499d1cd59cb118debd876af398dd8191f510e287b88cde515e7501d8"
TASKS = {
    "humaneval_instruct": ("HumanEval pass@1", None, None),
    "hendrycks_math500": ("MATH-500 exact match", "exact_match", "none"),
    "mmlu_pro_history": ("MMLU-Pro history", "exact_match", "custom-extract"),
    "mmlu_pro_other": ("MMLU-Pro other", "exact_match", "custom-extract"),
}
MATH_SCORE_POLICY = {
    "answer_selection": "first_nonempty_line_then_same_line_answer_clause",
    "later_lines_ignored": True,
    "max_answer_chars": 4096,
    "verification_timeout_seconds": 5,
}
MATH_SCORE_VERSIONS = {
    "antlr4-python3-runtime": "4.11.0",
    "latex2sympy2-extended": "1.11.0",
    "math-verify": "0.9.0",
    "mpmath": "1.3.0",
    "sympy": "1.14.0",
}
MMLU_CHOICES = "ABCDEFGHIJ"
LEGACY_BASE_URL = "http://127.0.0.1:8080/v1/completions"
LOOPBACK_BASE_URL_RE = re.compile(
    r"^http://127\.0\.0\.1:([1-9][0-9]{0,4})/v1/completions$"
)
MMLU_PREFIXES = {
    "mmlu_pro_history": (
        8465,
        "78d34da644b9f246f93937bcd75d7f589ca0be50876f2d0d8d97626806755586",
    ),
    "mmlu_pro_other": (
        3079,
        "a659aa9acf5720423e3588e16d8db6dd0ba7af91b715e4bd4d7c438c15370be7",
    ),
}


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_string(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hash_string(raw)


def locked_prompt(task: str, doc: dict[str, Any]) -> str:
    if task == "humaneval_instruct":
        return (
            "Write a solution to the following problem and make sure that it passes the tests:\n"
            f"```python\n{doc['prompt']}\n```\n"
        )
    if task == "hendrycks_math500":
        return f"Problem: {doc['problem']}\nAnswer:"
    if task.startswith("mmlu_pro_"):
        prompt = f"Question:\n{doc['question']}\nOptions:\n"
        for choice, option in zip(MMLU_CHOICES, doc["options"]):
            prompt += f"{choice}. {option.strip()}\n"
        return prompt + "Answer: Let's think step by step."
    raise ValueError(f"{task}: no frozen prompt renderer")


def locked_documents(panel: dict[str, Any], task: str) -> dict[int, dict[str, Any]]:
    rows = panel["selected_documents"][task]
    result = {row["index"]: row for row in rows}
    if len(result) != panel["task_counts"][task]:
        raise ValueError(f"{task}: invalid frozen document fingerprints")
    return result


def logged_prompt(row: dict[str, Any], context: str) -> str:
    arguments = row.get("arguments")
    if not isinstance(arguments, dict) or len(arguments) != 1:
        raise ValueError(f"{context}: malformed generation arguments")
    request = next(iter(arguments.values()))
    if not isinstance(request, dict) or not isinstance(request.get("arg_0"), str):
        raise ValueError(f"{context}: missing logged prompt")
    return request["arg_0"]


def validate_logged_prompt(
    task: str, doc: dict[str, Any], prompt: str, context: str
) -> None:
    current = locked_prompt(task, doc)
    if task == "humaneval_instruct":
        expected = (
            current
            + "Here is the completed function:\n"
            + f"```python\n{doc['prompt']}\n"
        )
        if prompt != expected:
            raise ValueError(f"{context}: logged prompt differs from the frozen task")
        return
    if task == "hendrycks_math500":
        if prompt != current:
            raise ValueError(f"{context}: logged prompt differs from the frozen task")
        return
    if not prompt.endswith(current):
        raise ValueError(f"{context}: logged prompt differs from the frozen task")
    prefix = prompt[: -len(current)]
    expected_length, expected_hash = MMLU_PREFIXES[task]
    if len(prefix) != expected_length or hash_string(prefix) != expected_hash:
        raise ValueError(f"{context}: few-shot prefix differs from the frozen task")


def validate_locked_sample(
    row: dict[str, Any],
    task: str,
    frozen: dict[int, dict[str, Any]],
    context: str,
) -> None:
    doc_id = row.get("doc_id")
    expected = frozen.get(doc_id)
    doc = row.get("doc")
    target = row.get("target")
    if expected is None or not isinstance(doc, dict) or not isinstance(target, str):
        raise ValueError(f"{context}: sample is absent from the frozen panel")
    actual_fingerprints = {
        "document_sha256": canonical_hash(doc),
        "prompt_sha256": canonical_hash(locked_prompt(task, doc)),
        "target_sha256": canonical_hash(target),
    }
    for field, actual in actual_fingerprints.items():
        if expected.get(field) != actual:
            raise ValueError(f"{context}: {field} differs from the frozen panel")

    prompt = logged_prompt(row, context)
    validate_logged_prompt(task, doc, prompt, context)
    runtime_fingerprints = {
        "doc_hash": hash_string(json.dumps(doc, indent=2, ensure_ascii=False)),
        "prompt_hash": hash_string(prompt),
        "target_hash": hash_string(str(target)),
    }
    for field, actual in runtime_fingerprints.items():
        if row.get(field) != actual:
            raise ValueError(f"{context}: {field} differs from the logged payload")


def exactly_one(paths: list[pathlib.Path], context: str) -> pathlib.Path:
    if len(paths) != 1:
        raise ValueError(f"{context}: expected exactly one file, found {len(paths)}")
    return paths[0]


def finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{context}: expected a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context}: expected a finite number")
    return result


def exact_sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def resolve_server_sha(panel_format: str, requested: str | None) -> str | None:
    if panel_format == "bw24-hourish-eval-panel-v1":
        if requested is not None and requested != SERVER_SHA256:
            raise ValueError(
                "legacy hourish panel requires its frozen server binary SHA"
            )
        return SERVER_SHA256
    return requested


def bootstrap_domain_macro_delta(
    baseline: dict[str, dict[str, float]],
    candidate: dict[str, dict[str, float]],
    iterations: int = 10_000,
) -> list[float]:
    rng = random.Random(20260711)
    samples = []
    for _ in range(iterations):
        task_deltas = []
        for task in TASKS:
            identities = sorted(baseline[task])
            draws = [identities[rng.randrange(len(identities))] for _ in identities]
            task_deltas.append(
                sum(
                    candidate[task][identity] - baseline[task][identity]
                    for identity in draws
                )
                / len(draws)
            )
        samples.append(sum(task_deltas) / len(task_deltas))
    samples.sort()
    return [samples[int(0.025 * iterations)], samples[int(0.975 * iterations)]]


def successful_receipt(
    path: pathlib.Path,
    arm: str,
    task: str,
    panel: dict[str, Any],
    manifest_sha: str,
    panel_sha: str = PANEL_SHA256,
    expected_server_sha: str | None = SERVER_SHA256,
) -> dict[str, Any]:
    receipt = json.loads(path.read_text())
    expected_samples = {task: panel["samples"][task]}
    expected_predict_only = task == "humaneval_instruct"
    expected_max_tokens = panel["max_gen_toks"][task]
    required = {
        "arm": arm,
        "model": arm,
        "lm_eval_commit": panel["lm_eval_commit"],
        "tasks": [task],
        "shard_id": task,
        "samples": expected_samples,
        "panel_lock_sha256": panel_sha,
        "predict_only": expected_predict_only,
        "max_gen_toks_override": expected_max_tokens,
        "num_concurrent": 1,
        "declared_spill_io": "worker",
        "declared_spill_pread_depth": "8",
        "declared_spill_stats": "1",
        "declared_serve_spec": "0",
        "artifact_identity_sha256": manifest_sha,
        "completed_successfully": True,
        "evaluator_exit_code": 0,
        "tee_exit_code": 0,
    }
    if expected_server_sha is not None:
        required["server_binary_sha256"] = expected_server_sha
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise ValueError(
                f"{arm}/{task}: receipt {key}={receipt.get(key)!r}, expected {expected!r}"
            )
    if (
        expected_server_sha is None
        and SHA256_RE.fullmatch(str(receipt.get("server_binary_sha256"))) is None
    ):
        raise ValueError(f"{arm}/{task}: invalid server binary SHA")
    validate_base_url(receipt.get("base_url"), panel["format"], f"{arm}/{task}")
    elapsed = finite_number(receipt.get("elapsed_seconds"), f"{arm}/{task} elapsed")
    if elapsed <= 0:
        raise ValueError(f"{arm}/{task}: elapsed time must be positive")
    server_log = pathlib.Path(receipt.get("server_log") or "")
    if not server_log.is_file() or receipt.get("server_log_sha256") != sha256(
        server_log
    ):
        raise ValueError(f"{arm}/{task}: invalid copied server log")
    return receipt


def validate_base_url(value: Any, panel_format: str, context: str) -> str:
    if panel_format == "bw24-hourish-eval-panel-v1":
        if value != LEGACY_BASE_URL:
            raise ValueError(f"{context}: base URL differs from legacy panel")
        return str(value)
    match = LOOPBACK_BASE_URL_RE.fullmatch(str(value))
    if match is None or int(match.group(1)) > 65535:
        raise ValueError(f"{context}: invalid loopback completion base URL")
    return str(value)


def comparison_config(config: dict[str, Any], panel_format: str) -> dict[str, Any]:
    comparable = dict(config)
    if panel_format == "bw24-expanded-capability-panel-v1":
        comparable.pop("base_url", None)
    return comparable


def validate_result_config(
    result: dict[str, Any],
    arm: str,
    task: str,
    panel: dict[str, Any],
    base_url: str,
) -> None:
    expected_model_args = {
        "model": arm,
        "base_url": base_url,
        "num_concurrent": 1,
        "max_retries": 3,
        "tokenized_requests": False,
        "tokenizer_backend": "none",
    }
    config = result.get("config", {})
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
    if (
        result.get("model_name") != arm
        or result.get("model_source") != "local-completions"
    ):
        raise ValueError(f"{arm}/{task}: wrong model identity in result")
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"{arm}/{task}: result config differs on {key}")
    if (
        result.get("n-samples", {}).get(task, {}).get("effective")
        != panel["task_counts"][task]
    ):
        raise ValueError(f"{arm}/{task}: wrong effective sample count")
    if set(result.get("task_hashes", {})) != {task} or set(
        result.get("versions", {})
    ) != {task}:
        raise ValueError(f"{arm}/{task}: missing task provenance")
    if (
        finite_number(
            result.get("total_evaluation_time_seconds"), f"{arm}/{task} evaluation time"
        )
        <= 0
    ):
        raise ValueError(f"{arm}/{task}: non-positive evaluation time")


def sample_identity(row: dict[str, Any], context: str) -> str:
    values = (row.get("doc_hash"), row.get("prompt_hash"), row.get("target_hash"))
    if not all(isinstance(value, str) and value for value in values):
        raise ValueError(f"{context}: missing sample hashes")
    return ":".join(values)


def load_regular_task(
    run_dir: pathlib.Path,
    arm: str,
    task: str,
    result: dict[str, Any],
    panel: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, float], pathlib.Path]:
    _, metric, filter_name = TASKS[task]
    assert metric is not None and filter_name is not None
    sample_path = exactly_one(
        sorted(run_dir.rglob(f"samples_{task}_*.jsonl")), f"{arm}/{task} samples"
    )
    frozen = locked_documents(panel, task)
    values: dict[str, float] = {}
    doc_ids: list[int] = []
    with sample_path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if row.get("filter") != filter_name:
                continue
            validate_locked_sample(row, task, frozen, f"{sample_path}:{line_number}")
            value = finite_number(
                row.get(metric), f"{sample_path}:{line_number} {metric}"
            )
            if value not in (0.0, 1.0):
                raise ValueError(f"{arm}/{task}: metric must be binary")
            identity = sample_identity(row, f"{sample_path}:{line_number}")
            if identity in values:
                raise ValueError(f"{arm}/{task}: duplicate sample identity")
            values[identity] = value
            doc_ids.append(row.get("doc_id"))
    if sorted(doc_ids) != sorted(panel["samples"][task]):
        raise ValueError(f"{arm}/{task}: sample indices differ from panel")
    successes = int(sum(values.values()))
    total = panel["task_counts"][task]
    aggregate = result.get("results", {}).get(task, {}).get(f"{metric},{filter_name}")
    if not math.isclose(
        finite_number(aggregate, f"{arm}/{task} aggregate"), successes / total
    ):
        raise ValueError(f"{arm}/{task}: aggregate differs from sample mean")
    return (
        {"successes": successes, "n": total, "rate": successes / total},
        values,
        sample_path,
    )


def load_code_task(
    run_dir: pathlib.Path,
    arm: str,
    panel: dict[str, Any],
    panel_sha: str = PANEL_SHA256,
) -> tuple[dict[str, Any], dict[str, float], list[pathlib.Path], dict[str, Any]]:
    score_path = run_dir / "code-score.json"
    receipt_path = run_dir / "code-score.receipt.json"
    if not score_path.is_file() or not receipt_path.is_file():
        raise ValueError(f"{arm}: missing code score evidence")
    score = json.loads(score_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    expected_n = panel["task_counts"]["humaneval_instruct"]
    panel_receipt = (
        receipt.get("panel_lock_sha256"),
        receipt.get("expected_sample_count"),
    )
    if (
        score.get("format") != "bw24-hourish-code-score-v1"
        or score.get("total") != expected_n
        or score.get("by_task", {}).get("humaneval_instruct", {}).get("total")
        != expected_n
        or receipt.get("format") != "bw24-hourish-code-score-receipt-v1"
        or receipt.get("output_sha256") != sha256(score_path)
        or receipt.get("sandbox", {}).get("network") != "none"
        or receipt.get("sandbox", {}).get("read_only_root") is not True
        or receipt.get("sandbox", {}).get("capabilities") != "all dropped"
        or receipt.get("sandbox", {}).get("no_new_privileges") is not True
        or receipt.get("sandbox", {}).get("pids_limit") != 32
        or receipt.get("sandbox", {}).get("memory_bytes") != 768 * 1024 * 1024
        or receipt.get("sandbox", {}).get("cpus") != 1
        or (
            panel["format"] != "bw24-hourish-eval-panel-v1"
            and receipt.get("sandbox", {}).get("cpu_shares") != 2
        )
        or score.get("limits")
        != {
            "cpu_seconds": 5,
            "wall_seconds": 10,
            "address_space_bytes": 512 * 1024 * 1024,
        }
        or not isinstance(receipt.get("tool_sha256"), str)
        or not receipt["tool_sha256"]
        or not isinstance(receipt.get("image_id"), str)
        or not receipt["image_id"]
        or (
            (
                panel["format"] != "bw24-hourish-eval-panel-v1"
                or panel_receipt != (None, None)
            )
            and panel_receipt != (panel_sha, expected_n)
        )
    ):
        raise ValueError(f"{arm}: invalid code score or sandbox receipt")
    values: dict[str, float] = {}
    doc_ids = []
    for row in score.get("samples", []):
        if row.get("task") != "humaneval_instruct" or not isinstance(
            row.get("passed"), bool
        ):
            raise ValueError(f"{arm}: malformed code sample")
        hashes = (row.get("doc_hash"), row.get("prompt_hash"), row.get("target_hash"))
        if not all(isinstance(value, str) and value for value in hashes):
            raise ValueError(f"{arm}: missing code sample hashes")
        identity = ":".join(hashes)
        if identity in values:
            raise ValueError(f"{arm}: duplicate code sample")
        values[identity] = float(row["passed"])
        doc_ids.append(row.get("doc_id"))
    if sorted(doc_ids) != sorted(panel["samples"]["humaneval_instruct"]):
        raise ValueError(f"{arm}: code sample indices differ from panel")
    sample_path = exactly_one(
        sorted(run_dir.rglob("samples_humaneval_instruct_*.jsonl")),
        f"{arm}/humaneval_instruct samples",
    )
    generated_identities = {}
    frozen = locked_documents(panel, "humaneval_instruct")
    with sample_path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            validate_locked_sample(
                row,
                "humaneval_instruct",
                frozen,
                f"{sample_path}:{line_number}",
            )
            identity = sample_identity(row, f"{sample_path}:{line_number}")
            if identity in generated_identities:
                raise ValueError(f"{arm}: duplicate generated code sample")
            generated_identities[identity] = row.get("doc_id")
    if set(generated_identities) != set(values) or sorted(
        generated_identities.values()
    ) != sorted(panel["samples"]["humaneval_instruct"]):
        raise ValueError(f"{arm}: generated and scored code samples differ")
    successes = int(sum(values.values()))
    if score.get("passed") != successes:
        raise ValueError(f"{arm}: code pass total differs from samples")
    scorer = {"tool_sha256": receipt["tool_sha256"], "image_id": receipt["image_id"]}
    if panel_receipt != (None, None):
        scorer.update(
            {"panel_lock_sha256": panel_sha, "expected_sample_count": expected_n}
        )
    return (
        {"successes": successes, "n": expected_n, "rate": successes / expected_n},
        values,
        [sample_path, score_path, receipt_path],
        scorer,
    )


def load_math_task(
    run_dir: pathlib.Path,
    arm: str,
    panel: dict[str, Any],
    panel_sha: str = PANEL_SHA256,
) -> tuple[dict[str, Any], dict[str, float], list[pathlib.Path], dict[str, Any]]:
    score_path = run_dir / "math-score.json"
    receipt_path = run_dir / "math-score.receipt.json"
    if not score_path.is_file() or not receipt_path.is_file():
        raise ValueError(f"{arm}: missing math score evidence")
    score = json.loads(score_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    expected_n = panel["task_counts"]["hendrycks_math500"]
    panel_receipt = (
        receipt.get("panel_lock_sha256"),
        receipt.get("expected_sample_count"),
    )
    if (
        score.get("format") != "bw24-hourish-math-score-v1"
        or score.get("total") != expected_n
        or score.get("by_task", {}).get("hendrycks_math500", {}).get("total")
        != expected_n
        or score.get("policy") != MATH_SCORE_POLICY
        or score.get("versions") != MATH_SCORE_VERSIONS
        or receipt.get("format") != "bw24-hourish-math-score-receipt-v1"
        or receipt.get("output_sha256") != sha256(score_path)
        or receipt.get("sandbox", {}).get("network") != "none"
        or receipt.get("sandbox", {}).get("read_only_root") is not True
        or receipt.get("sandbox", {}).get("capabilities") != "all dropped"
        or receipt.get("sandbox", {}).get("no_new_privileges") is not True
        or receipt.get("sandbox", {}).get("pids_limit") != 32
        or receipt.get("sandbox", {}).get("memory_bytes") != 1024 * 1024 * 1024
        or receipt.get("sandbox", {}).get("cpus") != 1
        or (
            panel["format"] != "bw24-hourish-eval-panel-v1"
            and receipt.get("sandbox", {}).get("cpu_shares") != 2
        )
        or not isinstance(receipt.get("tool_sha256"), str)
        or not receipt["tool_sha256"]
        or not isinstance(receipt.get("image_id"), str)
        or not receipt["image_id"]
        or (
            (
                panel["format"] != "bw24-hourish-eval-panel-v1"
                or panel_receipt != (None, None)
            )
            and panel_receipt != (panel_sha, expected_n)
        )
    ):
        raise ValueError(f"{arm}: invalid math score or sandbox receipt")
    sample_path = exactly_one(
        sorted(run_dir.rglob("samples_hendrycks_math500_*.jsonl")),
        f"{arm}/hendrycks_math500 samples",
    )
    input_files = score.get("input_files")
    if (
        not isinstance(input_files, list)
        or len(input_files) != 1
        or input_files[0].get("sha256") != sha256(sample_path)
    ):
        raise ValueError(f"{arm}: math score input hash differs from generation")
    generated_identities = {}
    frozen = locked_documents(panel, "hendrycks_math500")
    with sample_path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            validate_locked_sample(
                row,
                "hendrycks_math500",
                frozen,
                f"{sample_path}:{line_number}",
            )
            identity = sample_identity(row, f"{sample_path}:{line_number}")
            if identity in generated_identities:
                raise ValueError(f"{arm}: duplicate generated math sample")
            generated_identities[identity] = row.get("doc_id")
    values: dict[str, float] = {}
    doc_ids = []
    for row in score.get("samples", []):
        if (
            row.get("task") != "hendrycks_math500"
            or not isinstance(row.get("passed"), bool)
            or row.get("method") not in ("math_verify", "normalized_literal", "none")
            or not isinstance(row.get("answer"), str)
            or not isinstance(row.get("normalized_answer"), str)
        ):
            raise ValueError(f"{arm}: malformed math sample")
        hashes = (row.get("doc_hash"), row.get("prompt_hash"), row.get("target_hash"))
        if not all(isinstance(value, str) and value for value in hashes):
            raise ValueError(f"{arm}: missing math sample hashes")
        identity = ":".join(hashes)
        if identity in values:
            raise ValueError(f"{arm}: duplicate scored math sample")
        values[identity] = float(row["passed"])
        doc_ids.append(row.get("doc_id"))
    if (
        set(values) != set(generated_identities)
        or sorted(doc_ids) != sorted(panel["samples"]["hendrycks_math500"])
        or sorted(generated_identities.values())
        != sorted(panel["samples"]["hendrycks_math500"])
    ):
        raise ValueError(f"{arm}: generated and scored math samples differ")
    successes = int(sum(values.values()))
    if score.get("passed") != successes:
        raise ValueError(f"{arm}: math pass total differs from samples")
    scorer = {
        "tool_sha256": receipt["tool_sha256"],
        "image_id": receipt["image_id"],
        "policy": score["policy"],
        "versions": score["versions"],
    }
    if panel_receipt != (None, None):
        scorer.update(
            {"panel_lock_sha256": panel_sha, "expected_sample_count": expected_n}
        )
    return (
        {"successes": successes, "n": expected_n, "rate": successes / expected_n},
        values,
        [sample_path, score_path, receipt_path],
        scorer,
    )


def load_arm(
    out_root: pathlib.Path,
    run_id: str,
    arm: str,
    panel: dict[str, Any],
    panel_sha: str = PANEL_SHA256,
    expected_server_sha: str | None = SERVER_SHA256,
) -> dict[str, Any]:
    run_dir = out_root / arm / run_id
    if not run_dir.is_dir():
        raise ValueError(f"{arm}: missing run directory {run_dir}")
    shard_dirs = {
        path.name: path for path in (run_dir / "shards").iterdir() if path.is_dir()
    }
    if set(shard_dirs) != set(TASKS):
        raise ValueError(f"{arm}: shard set differs from panel")
    manifest_payloads = set()
    suite_lock_payloads = set()
    shared_config = None
    task_hashes = {}
    evidence = []
    result_by_task = {}
    for task, shard_dir in shard_dirs.items():
        panel_copy = shard_dir / "panel.lock.json"
        suite_copy = shard_dir / "suite.lock.json"
        manifest_path = shard_dir / "artifact-manifest.json"
        metadata_path = shard_dir / "run-metadata.json"
        if sha256(panel_copy) != panel_sha:
            raise ValueError(f"{arm}/{task}: copied panel lock differs")
        suite_lock_payloads.add(suite_copy.read_bytes())
        manifest_payloads.add(manifest_path.read_bytes())
        manifest_sha = sha256(manifest_path)
        receipt = successful_receipt(
            metadata_path,
            arm,
            task,
            panel,
            manifest_sha,
            panel_sha,
            expected_server_sha,
        )
        comparable = {
            key: receipt.get(key)
            for key in (
                "bw24_commit",
                "lm_eval_commit",
                "base_url",
                "num_concurrent",
                "declared_spill_io",
                "declared_spill_pread_depth",
                "declared_spill_stats",
                "declared_serve_spec",
                "server_binary_sha256",
            )
        }
        if shared_config is None:
            shared_config = comparable
        elif comparison_config(comparable, panel["format"]) != comparison_config(
            shared_config, panel["format"]
        ):
            raise ValueError(f"{arm}/{task}: shared run configuration differs")
        result_path = exactly_one(
            sorted(shard_dir.rglob("results_*.json")), f"{arm}/{task} result"
        )
        result = json.loads(result_path.read_text())
        validate_result_config(result, arm, task, panel, receipt["base_url"])
        task_hashes[task] = result["task_hashes"][task]
        result_by_task[task] = result
        evidence.extend(
            (panel_copy, suite_copy, manifest_path, metadata_path, result_path)
        )
    if len(manifest_payloads) != 1:
        raise ValueError(f"{arm}: artifact manifests differ across shards")
    if len(suite_lock_payloads) != 1:
        raise ValueError(f"{arm}: suite locks differ across shards")
    suite_lock_sha = hashlib.sha256(suite_lock_payloads.pop()).hexdigest()
    manifest = json.loads(manifest_payloads.pop())
    artifact_bytes = manifest.get("artifact_bytes")
    if (
        isinstance(artifact_bytes, bool)
        or not isinstance(artifact_bytes, int)
        or artifact_bytes <= 0
    ):
        raise ValueError(f"{arm}: invalid artifact size")

    tasks = {}
    values = {}
    code_task, code_values, code_evidence, code_scorer = load_code_task(
        run_dir, arm, panel, panel_sha
    )
    tasks["humaneval_instruct"] = code_task
    values["humaneval_instruct"] = code_values
    evidence.extend(code_evidence)
    math_task, math_values, math_evidence, math_scorer = load_math_task(
        run_dir, arm, panel, panel_sha
    )
    tasks["hendrycks_math500"] = math_task
    values["hendrycks_math500"] = math_values
    evidence.extend(math_evidence)
    for task in TASKS:
        if task in ("humaneval_instruct", "hendrycks_math500"):
            continue
        task_result, task_values, sample_path = load_regular_task(
            run_dir, arm, task, result_by_task[task], panel
        )
        tasks[task] = task_result
        values[task] = task_values
        evidence.append(sample_path)
    macro = sum(row["rate"] for row in tasks.values()) / len(tasks)
    total_successes = sum(row["successes"] for row in tasks.values())
    total_n = sum(row["n"] for row in tasks.values())
    return {
        "run_dir": str(run_dir),
        "artifact_bytes": artifact_bytes,
        "logical_model_bytes": artifact_bytes + SHARED_MODEL_BYTES,
        "logical_model_gib": (artifact_bytes + SHARED_MODEL_BYTES) / (1 << 30),
        "tasks": tasks,
        "domain_macro": macro,
        "total_correct": total_successes,
        "total_questions": total_n,
        "question_weighted": total_successes / total_n,
        "values": values,
        "task_hashes": task_hashes,
        "suite_lock_sha256": suite_lock_sha,
        "code_scorer": code_scorer,
        "math_scorer": math_scorer,
        "shared_config": shared_config,
        "evidence": [
            {"path": str(path), "sha256": sha256(path)}
            for path in sorted(set(evidence))
        ],
    }


def build_report(
    out_root: pathlib.Path,
    run_id: str,
    arms: list[str],
    baseline: str,
    panel_path: pathlib.Path,
    suite_lock_path: pathlib.Path | None = None,
    server_sha256: str | None = None,
) -> dict[str, Any]:
    if baseline not in arms:
        raise ValueError("baseline must be included in arms")
    if suite_lock_path is None:
        suite_lock_path = pathlib.Path(__file__).with_name("suite.lock.json")
    panel = validate_panel(panel_path, suite_lock_path)
    panel_sha = sha256(panel_path)
    suite_lock_sha = sha256(suite_lock_path)
    expected_server_sha = resolve_server_sha(panel["format"], server_sha256)
    loaded = {
        arm: load_arm(out_root, run_id, arm, panel, panel_sha, expected_server_sha)
        for arm in arms
    }
    reference = loaded[baseline]
    for arm, data in loaded.items():
        if data["suite_lock_sha256"] != suite_lock_sha:
            raise ValueError(f"{arm}: copied suite lock differs from analysis lock")
        comparable_config = comparison_config(data["shared_config"], panel["format"])
        reference_config = comparison_config(
            reference["shared_config"], panel["format"]
        )
        if comparable_config != reference_config:
            raise ValueError(f"{arm}: run configuration differs from baseline")
        if data["task_hashes"] != reference["task_hashes"]:
            raise ValueError(f"{arm}: task definitions differ from baseline")
        if data["suite_lock_sha256"] != reference["suite_lock_sha256"]:
            raise ValueError(f"{arm}: suite lock differs from baseline")
        if data["code_scorer"] != reference["code_scorer"]:
            raise ValueError(f"{arm}: code scorer identity differs from baseline")
        if data["math_scorer"] != reference["math_scorer"]:
            raise ValueError(f"{arm}: math scorer identity differs from baseline")
        for task in TASKS:
            if set(data["values"][task]) != set(reference["values"][task]):
                raise ValueError(f"{arm}/{task}: paired sample identities differ")

    comparisons = {}
    for arm, data in loaded.items():
        wins = losses = ties = 0
        if arm != baseline:
            for task in TASKS:
                for identity, base_value in reference["values"][task].items():
                    delta = data["values"][task][identity] - base_value
                    wins += delta > 0
                    losses += delta < 0
                    ties += delta == 0
        comparisons[arm] = {
            "domain_macro_delta": data["domain_macro"] - reference["domain_macro"],
            "domain_macro_delta_bootstrap_ci95": (
                [0.0, 0.0]
                if arm == baseline
                else bootstrap_domain_macro_delta(reference["values"], data["values"])
            ),
            "question_weighted_delta": data["question_weighted"]
            - reference["question_weighted"],
            "paired_wins": wins,
            "paired_losses": losses,
            "paired_ties": ties,
            "paired_exact_sign_p": exact_sign_p(wins, losses),
        }
    pareto = []
    for arm, data in loaded.items():
        dominated = any(
            other != arm
            and loaded[other]["logical_model_bytes"] <= data["logical_model_bytes"]
            and loaded[other]["domain_macro"] >= data["domain_macro"]
            and (
                loaded[other]["logical_model_bytes"] < data["logical_model_bytes"]
                or loaded[other]["domain_macro"] > data["domain_macro"]
            )
            for other in arms
        )
        if not dominated:
            pareto.append(arm)
    for data in loaded.values():
        data.pop("values")
    return {
        "format": (
            "bw24-hourish-capability-screen-v1"
            if panel["format"] == "bw24-hourish-eval-panel-v1"
            else "bw24-expanded-capability-screen-v1"
        ),
        "purpose": "directional screen; not a final capability claim",
        "run_id": run_id,
        "panel_lock_sha256": panel_sha,
        "baseline": baseline,
        "arms": loaded,
        "comparisons_to_baseline": comparisons,
        "point_estimate_pareto": pareto,
    }


def self_test() -> None:
    assert exactly_one([pathlib.Path("x")], "fixture") == pathlib.Path("x")
    assert finite_number(1, "fixture") == 1.0
    assert exact_sign_p(0, 0) == 1.0
    assert exact_sign_p(3, 0) == 0.25
    assert resolve_server_sha("bw24-hourish-eval-panel-v1", None) == SERVER_SHA256
    assert resolve_server_sha("bw24-expanded-capability-panel-v1", None) is None
    assert (
        validate_base_url(
            LEGACY_BASE_URL, "bw24-hourish-eval-panel-v1", "fixture"
        )
        == LEGACY_BASE_URL
    )
    assert (
        validate_base_url(
            "http://127.0.0.1:8087/v1/completions",
            "bw24-expanded-capability-panel-v1",
            "fixture",
        )
        == "http://127.0.0.1:8087/v1/completions"
    )
    port_8080 = {"base_url": LEGACY_BASE_URL, "server_binary_sha256": "a" * 64}
    port_8087 = {
        "base_url": "http://127.0.0.1:8087/v1/completions",
        "server_binary_sha256": "a" * 64,
    }
    assert comparison_config(
        port_8080, "bw24-expanded-capability-panel-v1"
    ) == comparison_config(port_8087, "bw24-expanded-capability-panel-v1")
    assert comparison_config(
        port_8080, "bw24-hourish-eval-panel-v1"
    ) != comparison_config(port_8087, "bw24-hourish-eval-panel-v1")
    try:
        validate_base_url(
            "http://127.0.0.1:8087/v1/completions",
            "bw24-hourish-eval-panel-v1",
            "fixture",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("legacy panel accepted a non-default port")
    for invalid_url in (
        "http://localhost:8081/v1/completions",
        "http://127.0.0.1:0/v1/completions",
        "http://127.0.0.1:65536/v1/completions",
    ):
        try:
            validate_base_url(
                invalid_url, "bw24-expanded-capability-panel-v1", "fixture"
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid expanded base URL: {invalid_url}")
    try:
        resolve_server_sha("bw24-hourish-eval-panel-v1", "0" * 64)
    except ValueError:
        pass
    else:
        raise AssertionError("legacy panel accepted a different server SHA")
    doc = {
        "problem": "What is 2 + 2?",
        "answer": "4",
        "subject": "Algebra",
        "level": 1,
        "unique_id": "fixture/math/1",
    }
    prompt = locked_prompt("hendrycks_math500", doc)
    target = "4"
    frozen = {
        7: {
            "index": 7,
            "id": doc["unique_id"],
            "document_sha256": canonical_hash(doc),
            "prompt_sha256": canonical_hash(prompt),
            "target_sha256": canonical_hash(target),
        }
    }
    row = {
        "doc_id": 7,
        "doc": doc,
        "target": target,
        "arguments": {"gen_args_0": {"arg_0": prompt}},
        "doc_hash": hash_string(json.dumps(doc, indent=2, ensure_ascii=False)),
        "prompt_hash": hash_string(prompt),
        "target_hash": hash_string(target),
    }
    validate_locked_sample(row, "hendrycks_math500", frozen, "fixture")
    wrong_logged_prompt = json.loads(json.dumps(row))
    wrong_logged_prompt["arguments"]["gen_args_0"]["arg_0"] = "arbitrary prompt"
    wrong_logged_prompt["prompt_hash"] = hash_string("arbitrary prompt")
    try:
        validate_locked_sample(
            wrong_logged_prompt, "hendrycks_math500", frozen, "fixture"
        )
    except ValueError as exc:
        assert "logged prompt" in str(exc)
    else:
        raise AssertionError("wrong logged prompt passed frozen fingerprint validation")
    wrong = json.loads(json.dumps(row))
    wrong["doc"]["problem"] = "What is 2 + 3?"
    wrong["target"] = "5"
    wrong_prompt = locked_prompt("hendrycks_math500", wrong["doc"])
    wrong["arguments"]["gen_args_0"]["arg_0"] = wrong_prompt
    wrong["doc_hash"] = hash_string(
        json.dumps(wrong["doc"], indent=2, ensure_ascii=False)
    )
    wrong["prompt_hash"] = hash_string(wrong_prompt)
    wrong["target_hash"] = hash_string(wrong["target"])
    try:
        validate_locked_sample(wrong, "hendrycks_math500", frozen, "fixture")
    except ValueError as exc:
        assert "frozen panel" in str(exc)
    else:
        raise AssertionError("wrong content passed frozen fingerprint validation")
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "x"
        path.write_bytes(b"abc")
        assert (
            sha256(path)
            == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        )
    print("hourish result summarizer self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=pathlib.Path)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--arms",
        default="plain_quant,plain_reap_quant,mix_quant,mix_quant_prune25,traffic_mix_quant",
    )
    parser.add_argument("--baseline", default="plain_quant")
    parser.add_argument(
        "--panel-lock",
        type=pathlib.Path,
        default=pathlib.Path(__file__).with_name("hourish-panel.lock.json"),
    )
    parser.add_argument(
        "--suite-lock",
        type=pathlib.Path,
        default=pathlib.Path(__file__).with_name("suite.lock.json"),
    )
    parser.add_argument(
        "--server-sha256",
        help="require this server binary hash; the expanded panel otherwise derives equality across arms",
    )
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.out_root is None or not args.run_id:
        parser.error("--out-root and --run-id are required")
    if (
        args.server_sha256 is not None
        and SHA256_RE.fullmatch(args.server_sha256) is None
    ):
        parser.error("--server-sha256 must be 64 lowercase hexadecimal characters")
    arms = [arm for arm in args.arms.split(",") if arm]
    report = build_report(
        args.out_root,
        args.run_id,
        arms,
        args.baseline,
        args.panel_lock,
        args.suite_lock,
        args.server_sha256,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite {args.output}")
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
