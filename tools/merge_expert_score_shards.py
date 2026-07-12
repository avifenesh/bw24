#!/usr/bin/env python3
"""Merge disjoint bw24 expert-retention score shards into one frozen score file."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any


FORMAT = "bw24-expert-retention-scores-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def without_layers(model: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in model.items() if key != "moe_layers"}


def merge(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one score shard is required")
    shards = [(path, json.loads(path.read_text())) for path in paths]
    first = shards[0][1]
    if first.get("format") != FORMAT:
        raise ValueError(f"score format must be {FORMAT!r}")
    expected_layers = [int(layer) for layer in first["model"]["complete_moe_layers"]]
    common_model = without_layers(first["model"])
    common_sections = {
        key: first[key] for key in ("rank_metric", "policy", "calibration", "source")
    }
    seen_layers: set[int] = set()
    scores: list[dict[str, Any]] = []
    teacher_targets: dict[str, Any] = {}
    expect_teacher_targets = bool(first.get("teacher_targets"))
    receipts = []
    for path, shard in shards:
        if shard.get("format") != FORMAT:
            raise ValueError(f"{path}: unsupported score format")
        if without_layers(shard["model"]) != common_model:
            raise ValueError(f"{path}: model metadata differs from the first shard")
        if any(shard.get(key) != value for key, value in common_sections.items()):
            raise ValueError(f"{path}: policy, calibration, or source metadata differs")
        layers = {int(layer) for layer in shard["model"]["moe_layers"]}
        overlap = seen_layers & layers
        if overlap:
            raise ValueError(f"{path}: duplicate layer coverage {sorted(overlap)}")
        row_layers = {int(row["layer"]) for row in shard["scores"]}
        if row_layers != layers:
            raise ValueError(f"{path}: score rows do not match declared layer coverage")
        shard_targets = shard.get("teacher_targets", {})
        if bool(shard_targets) != expect_teacher_targets:
            raise ValueError(f"{path}: teacher-target presence differs from the first shard")
        if expect_teacher_targets and {int(layer) for layer in shard_targets} != layers:
            raise ValueError(f"{path}: teacher targets do not match declared layer coverage")
        for layer, receipt in shard_targets.items():
            target_path = Path(receipt["path"])
            if target_path.stat().st_size != int(receipt["bytes"]):
                raise ValueError(f"{path}: teacher target {target_path} size changed")
            if sha256(target_path) != receipt["sha256"]:
                raise ValueError(f"{path}: teacher target {target_path} hash changed")
            teacher_targets[str(layer)] = receipt
        seen_layers.update(layers)
        scores.extend(shard["scores"])
        receipts.append({
            "path": str(path.resolve()), "sha256": sha256(path), "layers": sorted(layers)
        })
    if seen_layers != set(expected_layers):
        raise ValueError(
            f"score shards cover {sorted(seen_layers)}, expected {sorted(expected_layers)}"
        )
    scores.sort(key=lambda row: (int(row["layer"]), int(row["expert"])))
    expert_count = int(first["model"]["expert_count"])
    expected_rows = len(expected_layers) * expert_count
    keys = {(int(row["layer"]), int(row["expert"])) for row in scores}
    if len(scores) != expected_rows or len(keys) != expected_rows:
        raise ValueError(
            f"merged scores have rows/unique={len(scores)}/{len(keys)}, expected {expected_rows}"
        )
    result = dict(first)
    result["model"] = dict(first["model"])
    result["model"]["moe_layers"] = expected_layers
    result["scores"] = scores
    result["teacher_targets"] = dict(
        sorted(teacher_targets.items(), key=lambda item: int(item[0]))
    )
    result["shards"] = receipts
    return result


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="bw24-score-merge-") as tmp:
        root = Path(tmp)
        paths = []
        for layer in (1, 2):
            path = root / f"score-{layer}.json"
            path.write_text(json.dumps({
                "format": FORMAT,
                "rank_metric": "self_test",
                "model": {
                    "expert_count": 2,
                    "moe_layers": [layer],
                    "complete_moe_layers": [1, 2],
                },
                "policy": {"x": 1},
                "calibration": {"public_eval_data_used_for_selection": False},
                "source": {"sha256": "abc"},
                "teacher_targets": {},
                "scores": [
                    {"layer": layer, "expert": expert, "retain_score": layer + expert}
                    for expert in range(2)
                ],
            }))
            paths.append(path)
        result = merge(paths)
        assert result["model"]["moe_layers"] == [1, 2]
        assert len(result["scores"]) == 4 and len(result["shards"]) == 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        print("expert score shard merge self-test: PASS")
        return
    args = parse_args()
    result = merge(args.shard)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out} sha256={sha256(args.out)} scores={len(result['scores'])}")


if __name__ == "__main__":
    main()
