#!/usr/bin/env python3
"""Verify the pinned lm-eval checkout and inject locked Hub dataset revisions."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import tempfile
from pathlib import Path


def inject_revision(path: Path, dataset: str, revision: str) -> bool:
    text = path.read_text()
    if f"dataset_path: {dataset}" not in text:
        raise ValueError(f"{path}: expected dataset_path {dataset!r}")
    lines = text.splitlines()
    if "dataset_kwargs:" in lines:
        start = lines.index("dataset_kwargs:")
        for i in range(start + 1, min(start + 8, len(lines))):
            if lines[i].startswith("  revision:"):
                lines[i] = f"  revision: {revision}"
                break
        else:
            lines.insert(start + 1, f"  revision: {revision}")
    else:
        insert_at = next(
            (i + 1 for i, line in enumerate(lines) if line.startswith("dataset_name:")),
            next(i + 1 for i, line in enumerate(lines) if line.startswith("dataset_path:")),
        )
        lines[insert_at:insert_at] = ["dataset_kwargs:", f"  revision: {revision}"]
    updated = "\n".join(lines) + "\n"
    if updated == text:
        return False

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(updated)
            temp_path = Path(handle.name)
        temp_path.chmod(path.stat().st_mode & 0o7777)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return True


def self_test() -> None:
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "task.yaml"
        path.write_text("dataset_path: org/data\ndataset_name: default\n")
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            changed = list(
                pool.map(
                    lambda _: inject_revision(path, "org/data", "abc123"),
                    range(64),
                )
            )
        assert any(changed)
        assert path.read_text() == (
            "dataset_path: org/data\n"
            "dataset_name: default\n"
            "dataset_kwargs:\n"
            "  revision: abc123\n"
        )
        assert inject_revision(path, "org/data", "abc123") is False
        assert not list(path.parent.glob(f".{path.name}.*.tmp"))
    print("lm-eval harness preparation self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("harness", type=Path, nargs="?")
    parser.add_argument("--lock", type=Path, default=Path(__file__).with_name("suite.lock.json"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.harness is None:
        parser.error("harness is required unless --self-test is used")
    lock = json.loads(args.lock.read_text())
    got = subprocess.check_output(
        ["git", "-C", str(args.harness), "rev-parse", "HEAD"], text=True
    ).strip()
    expected = lock["lm_eval_commit"]
    if got != expected:
        raise SystemExit(f"lm-eval checkout is {got}, expected {expected}")
    changed = 0
    for dataset, spec in lock["datasets"].items():
        for rel in spec["task_files"]:
            changed += inject_revision(
                args.harness / rel, dataset, spec["revision"]
            )
    print(
        f"lm-eval {got}: dataset revisions pinned "
        f"({len(lock['datasets'])} datasets, {changed} files changed)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
