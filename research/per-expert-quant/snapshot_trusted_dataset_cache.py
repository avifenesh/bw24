#!/usr/bin/env python3
"""Freeze exact offline dataset rows and Arrow files used by trusted capability evals."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATASETS = (
    ("Idavidrein/gpqa", "gpqa_diamond"),
    ("HuggingFaceH4/MATH-500", None),
    ("TIGER-Lab/MMLU-Pro", None),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_digest(rows: Any) -> str:
    digest = hashlib.sha256()
    for row in rows:
        payload = json.dumps(
            row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        digest.update(payload + b"\n")
    return digest.hexdigest()


def self_test() -> None:
    rows = [{"b": [2, 3], "a": "x"}, {"a": "y", "b": []}]
    first = row_digest(rows)
    assert first == row_digest([{"a": "x", "b": [2, 3]}, {"b": [], "a": "y"}])
    assert first != row_digest(rows + [{"a": "z", "b": []}])
    print("trusted dataset cache snapshot self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("HF_DATASETS_OFFLINE") != "1":
        raise SystemExit("trusted dataset snapshot requires HF_HUB_OFFLINE=1 and HF_DATASETS_OFFLINE=1")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite dataset receipt: {args.output}")

    from datasets import load_dataset

    lock = json.loads(args.lock.read_text())
    records = []
    for repo, config in DATASETS:
        revision = lock["datasets"][repo]["revision"]
        dataset = load_dataset(repo, config, revision=revision)
        splits = {}
        for name, rows in sorted(dataset.items()):
            cache_files = []
            for entry in rows.cache_files:
                path = Path(entry["filename"]).resolve()
                if revision not in str(path):
                    raise ValueError(f"{repo}/{name}: cache path is outside locked revision: {path}")
                cache_files.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
            splits[name] = {
                "rows": len(rows),
                "columns": rows.column_names,
                "row_sha256": row_digest(rows),
                "cache_files": sorted(cache_files, key=lambda item: item["path"]),
            }
        records.append({"repo": repo, "config": config or "default", "revision": revision, "splits": splits})

    payload = {
        "format": "bw24-trusted-dataset-cache-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "offline": True,
        "datasets_package": importlib.metadata.version("datasets"),
        "suite_lock": {"path": str(args.lock.resolve()), "sha256": sha256(args.lock)},
        "datasets": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"trusted dataset cache snapshot: PASS ({args.output})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
