#!/usr/bin/env python3
"""Build the deterministic, time-budgeted bw24 screening panel.

Run this from the pinned lm-eval environment recorded in suite.lock.json.  The
script emits a complete lock document on stdout; it never calls a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict, deque
from typing import Any


FORMAT = "bw24-hourish-eval-panel-v1"
SEED = "bw24-hourish-panel-v1-20260711"
HARNESS_COMMIT = "97a5e2c710e2b56b9dd48f367bb6fe87bbb2c176"
TASK_COUNTS = {
    "humaneval_instruct": 6,
    "mbpp_instruct": 6,
    "hendrycks_math500": 32,
    "mmlu_pro_history": 5,
    "mmlu_pro_other": 5,
}
CALIBRATION_INDICES = {
    "humaneval_instruct": [0, 1],
    "mbpp_instruct": [0, 1],
}
DATASETS = {
    "humaneval_instruct": {
        "repo": "openai/openai_humaneval",
        "revision": "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
    },
    "mbpp_instruct": {
        "repo": "google-research-datasets/mbpp",
        "revision": "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
    },
    "hendrycks_math500": {
        "repo": "HuggingFaceH4/MATH-500",
        "revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
    },
    "mmlu_pro_history": {
        "repo": "TIGER-Lab/MMLU-Pro",
        "revision": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
    },
    "mmlu_pro_other": {
        "repo": "TIGER-Lab/MMLU-Pro",
        "revision": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
    },
}
MAX_GEN_TOKS = {
    "humaneval_instruct": 512,
    "mbpp_instruct": 512,
    "hendrycks_math500": 256,
    "mmlu_pro_history": 256,
    "mmlu_pro_other": 256,
}
TARGET_MINUTES = {
    "programming": 30,
    "math": 15,
    "history": 10,
    "other": 10,
}


def stable_key(namespace: str, value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(f"{SEED}\0{namespace}\0{raw}".encode()).hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def code_indices(task_name: str, docs: list[dict[str, Any]]) -> list[int]:
    excluded = set(CALIBRATION_INDICES[task_name])
    ranked = sorted(
        (i for i in range(len(docs)) if i not in excluded),
        key=lambda i: stable_key(task_name, docs[i].get("task_id", i)),
    )
    return sorted(ranked[: TASK_COUNTS[task_name]])


def math_indices(docs: list[dict[str, Any]]) -> list[int]:
    by_subject_level: dict[str, dict[str, deque[int]]] = defaultdict(lambda: defaultdict(deque))
    for i, doc in enumerate(docs):
        by_subject_level[str(doc["subject"])][str(doc["level"])].append(i)
    for subject, levels in by_subject_level.items():
        for level, indices in levels.items():
            levels[level] = deque(
                sorted(indices, key=lambda i: stable_key(f"math:{subject}:{level}", docs[i]["unique_id"]))
            )

    subjects = sorted(by_subject_level)
    base, extra = divmod(TASK_COUNTS["hendrycks_math500"], len(subjects))
    extra_subjects = set(sorted(subjects, key=lambda s: stable_key("math-extra", s))[:extra])
    chosen: list[int] = []
    for subject in subjects:
        quota = base + int(subject in extra_subjects)
        levels = by_subject_level[subject]
        level_order = sorted(levels, key=lambda level: stable_key(f"math-level:{subject}", level))
        while quota:
            progressed = False
            for level in level_order:
                if quota and levels[level]:
                    chosen.append(levels[level].popleft())
                    quota -= 1
                    progressed = True
            if not progressed:
                raise RuntimeError(f"not enough MATH documents for subject {subject}")
    if len(chosen) != TASK_COUNTS["hendrycks_math500"]:
        raise AssertionError("wrong MATH panel size")
    return sorted(chosen)


def mmlu_indices(task_name: str, docs: list[dict[str, Any]]) -> list[int]:
    by_source: dict[str, list[int]] = defaultdict(list)
    for i, doc in enumerate(docs):
        by_source[str(doc["src"])].append(i)
    sources = sorted(by_source, key=lambda source: stable_key(f"{task_name}:source", source))
    ranked = {
        source: deque(
            sorted(
                by_source[source],
                key=lambda i: stable_key(f"{task_name}:{source}", docs[i]["question_id"]),
            )
        )
        for source in sources
    }
    chosen = []
    while len(chosen) < TASK_COUNTS[task_name]:
        progressed = False
        for source in sources:
            if len(chosen) == TASK_COUNTS[task_name]:
                break
            if ranked[source]:
                chosen.append(ranked[source].popleft())
                progressed = True
        if not progressed:
            raise RuntimeError(f"not enough documents for {task_name}")
    return sorted(chosen)


def selected_indices(task_name: str, docs: list[dict[str, Any]]) -> list[int]:
    if task_name in CALIBRATION_INDICES:
        return code_indices(task_name, docs)
    if task_name == "hendrycks_math500":
        return math_indices(docs)
    return mmlu_indices(task_name, docs)


def document_id(task_name: str, doc: dict[str, Any]) -> Any:
    for key in ("task_id", "unique_id", "question_id"):
        if key in doc:
            return doc[key]
    raise KeyError(f"{task_name} document has no stable ID")


def build() -> dict[str, Any]:
    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    from lm_eval.tasks import TaskManager, get_task_dict

    tasks = get_task_dict(list(TASK_COUNTS), task_manager=TaskManager())
    samples: dict[str, list[int]] = {}
    selected: dict[str, list[dict[str, Any]]] = {}
    document_counts: dict[str, int] = {}
    for task_name in TASK_COUNTS:
        task = tasks[task_name]
        docs = list(task.test_docs())
        document_counts[task_name] = len(docs)
        indices = selected_indices(task_name, docs)
        if len(indices) != TASK_COUNTS[task_name] or len(indices) != len(set(indices)):
            raise AssertionError(f"invalid selection for {task_name}")
        samples[task_name] = indices
        selected[task_name] = []
        for index in indices:
            doc = docs[index]
            selected[task_name].append(
                {
                    "index": index,
                    "id": document_id(task_name, doc),
                    "document_sha256": canonical_hash(doc),
                    "prompt_sha256": canonical_hash(task.doc_to_text(doc)),
                    "target_sha256": canonical_hash(task.doc_to_target(doc)),
                    "stratum": (
                        {"subject": doc["subject"], "level": doc["level"]}
                        if task_name == "hendrycks_math500"
                        else ({"source": doc["src"]} if task_name.startswith("mmlu_pro_") else None)
                    ),
                }
            )

    return {
        "format": FORMAT,
        "seed": SEED,
        "lm_eval_commit": HARNESS_COMMIT,
        "purpose": "directional screen; not a final capability claim",
        "target_minutes": TARGET_MINUTES,
        "temperature": 0.0,
        "num_concurrent": 1,
        "code_execution": "network-disabled sandbox required for scoring",
        "task_counts": TASK_COUNTS,
        "max_gen_toks": MAX_GEN_TOKS,
        "calibration_indices": CALIBRATION_INDICES,
        "dataset_revisions": DATASETS,
        "task_document_counts": document_counts,
        "samples": samples,
        "selected_documents": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="write the lock here instead of stdout")
    args = parser.parse_args()
    rendered = json.dumps(build(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
