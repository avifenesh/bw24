#!/usr/bin/env python3
"""Verify the pinned lm-eval checkout and inject locked Hub dataset revisions."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def inject_revision(path: Path, dataset: str, revision: str) -> None:
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
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("harness", type=Path)
    parser.add_argument("--lock", type=Path, default=Path(__file__).with_name("suite.lock.json"))
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text())
    got = subprocess.check_output(
        ["git", "-C", str(args.harness), "rev-parse", "HEAD"], text=True
    ).strip()
    expected = lock["lm_eval_commit"]
    if got != expected:
        raise SystemExit(f"lm-eval checkout is {got}, expected {expected}")
    for dataset, spec in lock["datasets"].items():
        for rel in spec["task_files"]:
            inject_revision(args.harness / rel, dataset, spec["revision"])
    print(f"lm-eval {got}: dataset revisions pinned ({len(lock['datasets'])} datasets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
