#!/usr/bin/env python3
"""Require two lm-eval runs to contain identical matched generations."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any


def load_samples(run_dir: Path) -> dict[tuple[str, str, str, str, str], dict[str, Any]]:
    samples: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    paths = sorted(run_dir.rglob("samples_*.jsonl"))
    if not paths:
        raise ValueError(f"no sample logs under {run_dir}")
    for path in paths:
        task = path.name.removeprefix("samples_").rsplit("_", 1)[0]
        with path.open() as handle:
            for line_number, line in enumerate(handle, 1):
                row = json.loads(line)
                hashes = tuple(row.get(key) for key in ("doc_hash", "prompt_hash", "target_hash"))
                if not all(isinstance(value, str) for value in hashes):
                    raise ValueError(f"{path}:{line_number}: missing sample hashes")
                filter_name = row.get("filter")
                if not isinstance(filter_name, str):
                    raise ValueError(f"{path}:{line_number}: missing filter name")
                identity = (task, filter_name, *hashes)
                if identity in samples:
                    raise ValueError(f"{path}:{line_number}: duplicate sample identity {identity}")
                samples[identity] = {
                    "resps": row.get("resps"),
                    "filtered_resps": row.get("filtered_resps"),
                }
    return samples


def compare(baseline_dir: Path, candidate_dir: Path) -> int:
    baseline = load_samples(baseline_dir)
    candidate = load_samples(candidate_dir)
    if baseline.keys() != candidate.keys():
        missing = sorted(baseline.keys() - candidate.keys())[:5]
        extra = sorted(candidate.keys() - baseline.keys())[:5]
        raise ValueError(f"sample identities differ; missing={missing}, extra={extra}")
    mismatches = [identity for identity in baseline if baseline[identity] != candidate[identity]]
    if mismatches:
        raise ValueError(f"{len(mismatches)} generation mismatches; first={mismatches[:5]}")
    return len(baseline)


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="bw24-generation-compare-") as tmp:
        root = Path(tmp)
        baseline, candidate = root / "baseline", root / "candidate"
        baseline.mkdir()
        candidate.mkdir()
        row = {
            "doc_hash": "doc", "prompt_hash": "prompt", "target_hash": "target",
            "filter": "fixture", "resps": [["answer"]], "filtered_resps": ["answer"],
        }
        for directory in (baseline, candidate):
            (directory / "samples_task_fixture.jsonl").write_text(json.dumps(row) + "\n")
        assert compare(baseline, candidate) == 1
        row["resps"] = [["different"]]
        (candidate / "samples_task_fixture.jsonl").write_text(json.dumps(row) + "\n")
        try:
            compare(baseline, candidate)
        except ValueError as exc:
            assert "generation mismatches" in str(exc)
        else:
            raise AssertionError("mismatch was not detected")
    print("generation comparator self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.baseline is None or args.candidate is None:
        parser.error("--baseline and --candidate are required")
    count = compare(args.baseline, args.candidate)
    print(f"generation comparison: PASS ({count} matched samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
