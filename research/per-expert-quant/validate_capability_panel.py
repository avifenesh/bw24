#!/usr/bin/env python3
"""Validate a frozen matched capability-screen panel and emit runner inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import tempfile
from typing import Any


TASKS = (
    "humaneval_instruct",
    "hendrycks_math500",
    "mmlu_pro_history",
    "mmlu_pro_other",
)
FORMATS = {
    "bw24-hourish-eval-panel-v1",
    "bw24-expanded-capability-panel-v1",
}
FROZEN_PANEL_SHA256 = {
    "bw24-hourish-eval-panel-v1": "770135c560b590844fcf09418e965a42ecb876a5eb9566564e19e8fb02bb6ce1",
    "bw24-expanded-capability-panel-v1": "33ca7c2a86ed52ab3ee06ec408ceda890e50447e5cc4a204a755afcd3368c64b",
}
EXPANDED_COUNTS = {
    "humaneval_instruct": 32,
    "hendrycks_math500": 56,
    "mmlu_pro_history": 10,
    "mmlu_pro_other": 17,
}
EXPANDED_SEED = "bw24-expanded-panel-v1-20260712"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
REVISION_RE = re.compile(r"[0-9a-f]{40}")


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_lock(path: pathlib.Path, context: str) -> dict[str, Any]:
    require(path.is_file(), f"missing {context}: {path}")
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"{context} must be a JSON object")
    return value


def validate_suite_provenance(panel: dict[str, Any], suite: dict[str, Any]) -> None:
    require(
        panel.get("lm_eval_commit") == suite.get("lm_eval_commit"),
        "panel and suite lock lm-eval commits differ",
    )
    suite_datasets = suite.get("datasets", {})
    revisions = panel.get("dataset_revisions")
    require(
        isinstance(revisions, dict) and set(revisions) == set(TASKS),
        "wrong dataset task set",
    )
    for task, spec in revisions.items():
        require(isinstance(spec, dict), f"{task}: invalid dataset revision entry")
        repo = spec.get("repo")
        revision = spec.get("revision")
        require(isinstance(repo, str) and repo, f"{task}: missing dataset repo")
        require(
            REVISION_RE.fullmatch(str(revision)) is not None,
            f"{task}: invalid dataset revision",
        )
        require(
            suite_datasets.get(repo, {}).get("revision") == revision,
            f"{task}: dataset revision differs from suite lock",
        )


def validate_exclusions(
    panel: dict[str, Any],
    excluded_panel_path: pathlib.Path | None,
) -> None:
    if panel["format"] != "bw24-expanded-capability-panel-v1":
        return
    require(panel.get("seed") == EXPANDED_SEED, "wrong expanded-panel seed")
    require(
        panel.get("task_counts") == EXPANDED_COUNTS, "wrong expanded-panel task counts"
    )
    if excluded_panel_path is None:
        excluded_panel_path = pathlib.Path(__file__).with_name(
            "hourish-panel.lock.json"
        )
    excluded_panel = load_lock(excluded_panel_path, "excluded panel lock")
    excluded_samples = excluded_panel.get("samples")
    require(
        isinstance(excluded_samples, dict) and set(excluded_samples) == set(TASKS),
        "excluded panel has the wrong task set",
    )
    metadata = panel.get("excluded_panel")
    require(
        isinstance(metadata, dict), "expanded panel is missing excluded-panel metadata"
    )
    require(
        metadata.get("format") == excluded_panel.get("format"),
        "excluded panel format differs",
    )
    require(
        metadata.get("seed") == excluded_panel.get("seed"),
        "excluded panel seed differs",
    )
    require(
        metadata.get("sha256") == file_sha256(excluded_panel_path),
        "excluded panel SHA differs",
    )
    require(
        metadata.get("task_counts") == excluded_panel.get("task_counts"),
        "excluded panel counts differ",
    )
    recorded = panel.get("excluded_indices")
    require(
        recorded == excluded_samples,
        "recorded exclusions differ from the excluded panel",
    )
    for task in TASKS:
        require(
            not set(panel["samples"][task]) & set(excluded_samples[task]),
            f"{task}: expanded panel overlaps excluded panel",
        )


def validate_panel(
    panel_path: pathlib.Path,
    suite_lock_path: pathlib.Path,
    excluded_panel_path: pathlib.Path | None = None,
) -> dict[str, Any]:
    panel = load_lock(panel_path, "panel lock")
    suite = load_lock(suite_lock_path, "suite lock")
    require(panel.get("format") in FORMATS, "unexpected panel format")
    require(
        file_sha256(panel_path) == FROZEN_PANEL_SHA256[panel["format"]],
        "panel lock SHA differs from its frozen format",
    )
    require(panel.get("temperature") == 0.0, "panel must use greedy decoding")
    require(panel.get("num_concurrent") == 1, "panel must use one request at a time")
    require(
        panel.get("code_execution") == "network-disabled sandbox required for scoring",
        "wrong code-scoring contract",
    )

    counts = panel.get("task_counts")
    samples = panel.get("samples")
    max_tokens = panel.get("max_gen_toks")
    document_counts = panel.get("task_document_counts")
    selected = panel.get("selected_documents")
    require(
        isinstance(counts, dict) and set(counts) == set(TASKS), "wrong task-count set"
    )
    require(
        isinstance(samples, dict) and set(samples) == set(TASKS),
        "wrong sample task set",
    )
    require(
        isinstance(max_tokens, dict) and set(max_tokens) == set(TASKS),
        "wrong token-limit set",
    )
    require(
        isinstance(document_counts, dict) and set(document_counts) == set(TASKS),
        "wrong document-count set",
    )
    require(
        isinstance(selected, dict) and set(selected) == set(TASKS),
        "wrong selected-doc set",
    )

    calibration = panel.get("calibration_indices")
    require(
        calibration == {"humaneval_instruct": [0, 1]}, "wrong calibration exclusion set"
    )
    for task in TASKS:
        count = counts[task]
        indices = samples[task]
        rows = selected[task]
        require(
            isinstance(count, int) and not isinstance(count, bool) and count > 0,
            f"{task}: invalid count",
        )
        require(
            isinstance(indices, list)
            and len(indices) == count
            and len(indices) == len(set(indices))
            and all(
                isinstance(index, int) and not isinstance(index, bool) and index >= 0
                for index in indices
            ),
            f"{task}: invalid sample indices",
        )
        require(
            isinstance(document_counts[task], int)
            and document_counts[task] > max(indices),
            f"{task}: sample index exceeds document count",
        )
        require(
            isinstance(max_tokens[task], int) and max_tokens[task] > 0,
            f"{task}: invalid generation limit",
        )
        require(
            isinstance(rows, list) and len(rows) == count,
            f"{task}: wrong selected-doc count",
        )
        require(
            [row.get("index") for row in rows] == indices,
            f"{task}: selected-doc order differs",
        )
        for row in rows:
            for field in ("document_sha256", "prompt_sha256", "target_sha256"):
                require(
                    SHA256_RE.fullmatch(str(row.get(field))) is not None,
                    f"{task}: invalid {field}",
                )
    require(
        not set(samples["humaneval_instruct"]) & set(calibration["humaneval_instruct"]),
        "HumanEval panel overlaps calibration indices",
    )
    validate_suite_provenance(panel, suite)
    validate_exclusions(panel, excluded_panel_path)
    return panel


def self_test(here: pathlib.Path) -> None:
    panel_path = here / "hourish-panel.lock.json"
    expanded_path = here / "expanded-capability-panel.lock.json"
    suite_path = here / "suite.lock.json"
    panel = validate_panel(panel_path, suite_path)
    require(sum(panel["task_counts"].values()) == 56, "legacy panel total changed")
    require(
        file_sha256(panel_path)
        == "770135c560b590844fcf09418e965a42ecb876a5eb9566564e19e8fb02bb6ce1",
        "legacy panel bytes changed",
    )
    expanded = validate_panel(expanded_path, suite_path, panel_path)
    require(
        sum(expanded["task_counts"].values()) == 115, "expanded panel total changed"
    )
    require(
        file_sha256(expanded_path)
        == "33ca7c2a86ed52ab3ee06ec408ceda890e50447e5cc4a204a755afcd3368c64b",
        "expanded panel bytes changed",
    )
    for task in TASKS:
        require(
            not set(panel["samples"][task]) & set(expanded["samples"][task]),
            f"{task}: frozen panels overlap",
        )
    with tempfile.TemporaryDirectory() as tmp:
        mutated_path = pathlib.Path(tmp) / "hourish-panel.lock.json"
        mutated = json.loads(panel_path.read_text())
        mutated["purpose"] = "mutated"
        mutated_path.write_text(json.dumps(mutated, indent=2, sort_keys=True) + "\n")
        try:
            validate_panel(mutated_path, suite_path)
        except ValueError as exc:
            require(
                "SHA differs" in str(exc), "mutated panel failed for the wrong reason"
            )
        else:
            raise AssertionError("mutated frozen panel passed validation")
    print("capability panel validator self-test: PASS")


def main() -> None:
    here = pathlib.Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument("panel_lock", type=pathlib.Path, nargs="?")
    parser.add_argument(
        "--suite-lock", type=pathlib.Path, default=here / "suite.lock.json"
    )
    parser.add_argument("--excluded-panel", type=pathlib.Path)
    parser.add_argument("--print-sha", action="store_true")
    parser.add_argument("--task-rows", action="store_true")
    parser.add_argument("--task-count", choices=TASKS)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(here)
        return
    if args.panel_lock is None:
        parser.error("panel_lock is required")
    panel = validate_panel(args.panel_lock, args.suite_lock, args.excluded_panel)
    if args.print_sha:
        print(file_sha256(args.panel_lock))
    elif args.task_rows:
        for task in TASKS:
            print(
                "\t".join(
                    (
                        task,
                        json.dumps(
                            {task: panel["samples"][task]},
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        str(panel["max_gen_toks"][task]),
                    )
                )
            )
    elif args.task_count:
        print(panel["task_counts"][args.task_count])
    else:
        print("capability panel validation: PASS")


if __name__ == "__main__":
    main()
