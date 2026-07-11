#!/usr/bin/env python3
"""Freeze a small, domain-balanced subset of the private routing calibration corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def select(source: Path, per_stratum: int) -> list[dict]:
    if per_stratum <= 0:
        raise ValueError("per-stratum count must be positive")
    records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    available = Counter(str(record["stratum"]) for record in records)
    short = {stratum: count for stratum, count in available.items() if count < per_stratum}
    if short:
        raise ValueError(f"strata do not have {per_stratum} records: {short}")
    used: Counter[str] = Counter()
    selected: list[dict] = []
    for record in records:
        stratum = str(record["stratum"])
        if used[stratum] < per_stratum:
            selected.append(record)
            used[stratum] += 1
    if not selected:
        raise ValueError("source calibration corpus is empty")
    return selected


def write_subset(source: Path, out: Path, manifest: Path, per_stratum: int) -> None:
    selected = select(source, per_stratum)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(record, sort_keys=True) for record in selected) + "\n")
    counts = Counter(str(record["stratum"]) for record in selected)
    payload = {
        "format": "bw24-confidence-calibration-lock-v1",
        "source": {"path": str(source.resolve()), "sha256": sha256(source)},
        "requests": {"path": str(out.resolve()), "sha256": sha256(out)},
        "selection": "first records per stratum from the already seeded and frozen source corpus",
        "per_stratum": per_stratum,
        "stratum_counts": dict(sorted(counts.items())),
        "request_count": len(selected),
        "prompt_tokens": sum(int(record["prompt_tokens"]) for record in selected),
        "public_eval_data_used_for_selection": False,
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="bw24-confidence-subset-") as tmp:
        root = Path(tmp)
        source = root / "source.jsonl"
        out = root / "requests.jsonl"
        manifest = root / "lock.json"
        source.write_text("\n".join(json.dumps({
            "ordinal": index,
            "stratum": stratum,
            "prompt_tokens": 2,
            "prompt_ids": [index, index + 1],
        }) for index, stratum in enumerate(["code", "math", "code", "math", "code", "math"])) + "\n")
        write_subset(source, out, manifest, 2)
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        assert [row["ordinal"] for row in rows] == [0, 1, 2, 3]
        lock = json.loads(manifest.read_text())
        assert lock["stratum_counts"] == {"code": 2, "math": 2}
        assert lock["prompt_tokens"] == 8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--per-stratum", type=int, default=4)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("confidence calibration subset self-test: PASS")
        return
    if args.source is None or args.out is None or args.manifest is None:
        parser.error("--source, --out, and --manifest are required")
    write_subset(args.source, args.out, args.manifest, args.per_stratum)
    print(f"wrote {args.out} and {args.manifest}")


if __name__ == "__main__":
    main()
