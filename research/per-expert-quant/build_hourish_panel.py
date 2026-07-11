#!/usr/bin/env python3
"""Build deterministic bw24 matched capability-screen panels.

Run this from the pinned lm-eval environment recorded in suite.lock.json.  The
script emits a complete lock document on stdout; it never calls a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


FORMAT = "bw24-hourish-eval-panel-v1"
SEED = "bw24-hourish-panel-v1-20260711"
HARNESS_COMMIT = "97a5e2c710e2b56b9dd48f367bb6fe87bbb2c176"
TASK_COUNTS = {
    "humaneval_instruct": 14,
    "hendrycks_math500": 32,
    "mmlu_pro_history": 5,
    "mmlu_pro_other": 5,
}
CALIBRATION_INDICES = {
    "humaneval_instruct": [0, 1],
}
EXPANDED_FORMAT = "bw24-expanded-capability-panel-v1"
EXPANDED_SEED = "bw24-expanded-panel-v1-20260712"
EXPANDED_TASK_COUNTS = {
    "humaneval_instruct": 32,
    "hendrycks_math500": 56,
    "mmlu_pro_history": 10,
    "mmlu_pro_other": 17,
}
DATASETS = {
    "humaneval_instruct": {
        "repo": "openai/openai_humaneval",
        "revision": "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(namespace: str, value: Any, seed: str = SEED) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(f"{seed}\0{namespace}\0{raw}".encode()).hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def code_indices(
    task_name: str,
    docs: list[dict[str, Any]],
    *,
    count: int | None = None,
    excluded_indices: set[int] | None = None,
    seed: str = SEED,
) -> list[int]:
    excluded = set(CALIBRATION_INDICES[task_name])
    excluded.update(excluded_indices or ())
    ranked = sorted(
        (i for i in range(len(docs)) if i not in excluded),
        key=lambda i: stable_key(task_name, docs[i].get("task_id", i), seed),
    )
    return sorted(ranked[: count if count is not None else TASK_COUNTS[task_name]])


def math_indices(
    docs: list[dict[str, Any]],
    *,
    count: int | None = None,
    excluded_indices: set[int] | None = None,
    seed: str = SEED,
) -> list[int]:
    expected = count if count is not None else TASK_COUNTS["hendrycks_math500"]
    excluded = excluded_indices or set()
    by_subject_level: dict[str, dict[str, deque[int]]] = defaultdict(
        lambda: defaultdict(deque)
    )
    for i, doc in enumerate(docs):
        if i in excluded:
            continue
        by_subject_level[str(doc["subject"])][str(doc["level"])].append(i)
    for subject, levels in by_subject_level.items():
        for level, indices in levels.items():
            levels[level] = deque(
                sorted(
                    indices,
                    key=lambda i: stable_key(
                        f"math:{subject}:{level}", docs[i]["unique_id"], seed
                    ),
                )
            )

    subjects = sorted(by_subject_level)
    base, extra = divmod(expected, len(subjects))
    extra_subjects = set(
        sorted(subjects, key=lambda subject: stable_key("math-extra", subject, seed))[
            :extra
        ]
    )
    chosen: list[int] = []
    for subject in subjects:
        quota = base + int(subject in extra_subjects)
        levels = by_subject_level[subject]
        level_order = sorted(
            levels, key=lambda level: stable_key(f"math-level:{subject}", level, seed)
        )
        while quota:
            progressed = False
            for level in level_order:
                if quota and levels[level]:
                    chosen.append(levels[level].popleft())
                    quota -= 1
                    progressed = True
            if not progressed:
                raise RuntimeError(f"not enough MATH documents for subject {subject}")
    if len(chosen) != expected:
        raise AssertionError("wrong MATH panel size")
    return sorted(chosen)


def mmlu_indices(
    task_name: str,
    docs: list[dict[str, Any]],
    *,
    count: int | None = None,
    excluded_indices: set[int] | None = None,
    seed: str = SEED,
) -> list[int]:
    expected = count if count is not None else TASK_COUNTS[task_name]
    excluded = excluded_indices or set()
    by_source: dict[str, list[int]] = defaultdict(list)
    for i, doc in enumerate(docs):
        if i in excluded:
            continue
        by_source[str(doc["src"])].append(i)
    sources = sorted(
        by_source, key=lambda source: stable_key(f"{task_name}:source", source, seed)
    )
    ranked = {
        source: deque(
            sorted(
                by_source[source],
                key=lambda i: stable_key(
                    f"{task_name}:{source}", docs[i]["question_id"], seed
                ),
            )
        )
        for source in sources
    }
    chosen = []
    while len(chosen) < expected:
        progressed = False
        for source in sources:
            if len(chosen) == expected:
                break
            if ranked[source]:
                chosen.append(ranked[source].popleft())
                progressed = True
        if not progressed:
            raise RuntimeError(f"not enough documents for {task_name}")
    return sorted(chosen)


def selected_indices(
    task_name: str,
    docs: list[dict[str, Any]],
    *,
    task_counts: dict[str, int] = TASK_COUNTS,
    excluded_indices: set[int] | None = None,
    seed: str = SEED,
) -> list[int]:
    options = {
        "count": task_counts[task_name],
        "excluded_indices": excluded_indices,
        "seed": seed,
    }
    if task_name in CALIBRATION_INDICES:
        return code_indices(task_name, docs, **options)
    if task_name == "hendrycks_math500":
        return math_indices(docs, **options)
    return mmlu_indices(task_name, docs, **options)


def document_id(task_name: str, doc: dict[str, Any]) -> Any:
    for key in ("task_id", "unique_id", "question_id"):
        if key in doc:
            return doc[key]
    raise KeyError(f"{task_name} document has no stable ID")


def panel_exclusions(
    path: Path | None,
) -> tuple[dict[str, set[int]], dict[str, Any] | None]:
    if path is None:
        return {task: set() for task in TASK_COUNTS}, None
    lock = json.loads(path.read_text())
    samples = lock.get("samples")
    if not isinstance(samples, dict) or set(samples) != set(TASK_COUNTS):
        raise ValueError("excluded panel has the wrong task set")
    exclusions: dict[str, set[int]] = {}
    for task, indices in samples.items():
        if (
            not isinstance(indices, list)
            or any(
                isinstance(index, bool) or not isinstance(index, int) or index < 0
                for index in indices
            )
            or len(indices) != len(set(indices))
        ):
            raise ValueError(f"excluded panel has invalid indices for {task}")
        exclusions[task] = set(indices)
    return exclusions, {
        "format": lock.get("format"),
        "seed": lock.get("seed"),
        "sha256": file_sha256(path),
        "task_counts": lock.get("task_counts"),
    }


def validate_suite_lock(path: Path) -> None:
    lock = json.loads(path.read_text())
    if lock.get("lm_eval_commit") != HARNESS_COMMIT:
        raise ValueError("suite lock lm-eval commit differs from the panel builder")
    datasets = lock.get("datasets", {})
    for spec in DATASETS.values():
        if datasets.get(spec["repo"], {}).get("revision") != spec["revision"]:
            raise ValueError(f"suite lock revision differs for {spec['repo']}")


def build(
    *,
    format_name: str = FORMAT,
    seed: str = SEED,
    task_counts: dict[str, int] = TASK_COUNTS,
    exclude_panel: Path | None = None,
    suite_lock: Path | None = None,
) -> dict[str, Any]:
    if suite_lock is not None:
        validate_suite_lock(suite_lock)
    exclusions, excluded_panel = panel_exclusions(exclude_panel)
    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    from lm_eval.tasks import TaskManager, get_task_dict

    tasks = get_task_dict(list(task_counts), task_manager=TaskManager())
    samples: dict[str, list[int]] = {}
    selected: dict[str, list[dict[str, Any]]] = {}
    document_counts: dict[str, int] = {}
    for task_name in task_counts:
        task = tasks[task_name]
        docs = list(task.test_docs())
        document_counts[task_name] = len(docs)
        indices = selected_indices(
            task_name,
            docs,
            task_counts=task_counts,
            excluded_indices=exclusions[task_name],
            seed=seed,
        )
        if len(indices) != task_counts[task_name] or len(indices) != len(set(indices)):
            raise AssertionError(f"invalid selection for {task_name}")
        if set(indices) & exclusions[task_name]:
            raise AssertionError(f"excluded documents selected for {task_name}")
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
                        else (
                            {"source": doc["src"]}
                            if task_name.startswith("mmlu_pro_")
                            else None
                        )
                    ),
                }
            )

    result = {
        "format": format_name,
        "seed": seed,
        "lm_eval_commit": HARNESS_COMMIT,
        "purpose": "directional screen; not a final capability claim",
        "target_minutes": TARGET_MINUTES,
        "temperature": 0.0,
        "num_concurrent": 1,
        "code_execution": "network-disabled sandbox required for scoring",
        "task_counts": task_counts,
        "max_gen_toks": MAX_GEN_TOKS,
        "calibration_indices": CALIBRATION_INDICES,
        "dataset_revisions": DATASETS,
        "task_document_counts": document_counts,
        "samples": samples,
        "selected_documents": selected,
    }
    if excluded_panel is not None:
        result["excluded_panel"] = excluded_panel
        result["excluded_indices"] = {
            task: sorted(indices) for task, indices in exclusions.items()
        }
    return result


def self_test() -> None:
    code_docs = [{"task_id": f"code-{index}"} for index in range(20)]
    selected = code_indices(
        "humaneval_instruct",
        code_docs,
        count=6,
        excluded_indices={2, 3, 4},
        seed=EXPANDED_SEED,
    )
    assert len(selected) == 6
    assert not set(selected) & {0, 1, 2, 3, 4}
    assert selected == code_indices(
        "humaneval_instruct",
        code_docs,
        count=6,
        excluded_indices={2, 3, 4},
        seed=EXPANDED_SEED,
    )

    math_docs = [
        {"subject": subject, "level": level, "unique_id": f"{subject}-{level}-{index}"}
        for subject in ("a", "b")
        for level in ("1", "2")
        for index in range(5)
    ]
    math = math_indices(math_docs, count=8, excluded_indices={0, 5}, seed=EXPANDED_SEED)
    assert len(math) == 8 and not set(math) & {0, 5}

    mmlu_docs = [
        {"src": source, "question_id": f"{source}-{index}"}
        for source in ("a", "b", "c")
        for index in range(5)
    ]
    mmlu = mmlu_indices(
        "mmlu_pro_other",
        mmlu_docs,
        count=7,
        excluded_indices={0, 6},
        seed=EXPANDED_SEED,
    )
    assert len(mmlu) == 7 and not set(mmlu) & {0, 6}
    print("capability panel builder self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="write the lock here instead of stdout")
    parser.add_argument("--profile", choices=("hourish", "expanded"), default="hourish")
    parser.add_argument(
        "--exclude-panel",
        type=Path,
        help="panel whose exact document indices must be excluded",
    )
    parser.add_argument(
        "--suite-lock",
        type=Path,
        default=Path(__file__).with_name("suite.lock.json"),
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.profile == "expanded":
        exclude_panel = args.exclude_panel or Path(__file__).with_name(
            "hourish-panel.lock.json"
        )
        result = build(
            format_name=EXPANDED_FORMAT,
            seed=EXPANDED_SEED,
            task_counts=EXPANDED_TASK_COUNTS,
            exclude_panel=exclude_panel,
            suite_lock=args.suite_lock,
        )
        result["purpose"] = (
            "expanded matched directional screen; not a final capability claim"
        )
        result.pop("target_minutes")
    else:
        if args.exclude_panel is not None:
            parser.error("--exclude-panel is only valid with --profile expanded")
        result = build(suite_lock=args.suite_lock)
    rendered = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
