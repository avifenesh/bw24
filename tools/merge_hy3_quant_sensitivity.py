#!/usr/bin/env python3
"""Merge disjoint Hy3 quant-sensitivity lane outputs with strict provenance checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any


FORMAT = "bw24-hy3-quant-sensitivity-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def merge(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one lane is required")
    payloads = [json.loads(path.read_text()) for path in paths]
    if any(item.get("format") != FORMAT for item in payloads):
        raise ValueError(f"all lanes must use format {FORMAT}")
    for field in ("measurement", "calibration", "source"):
        if len({canonical(item[field]) for item in payloads}) != 1:
            raise ValueError(f"lane {field} provenance differs")
    complete = [int(x) for x in payloads[0]["model"]["complete_moe_layers"]]
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    for path, item in zip(paths, payloads):
        for row in item["scores"]:
            key = (int(row["layer"]), int(row["expert"]))
            if key in rows:
                raise ValueError(f"duplicate score {key} in {path}")
            rows[key] = row
    expert_count = int(payloads[0]["model"]["expert_count"])
    expected = {(layer, expert) for layer in complete for expert in range(expert_count)}
    if rows.keys() != expected:
        missing, extra = expected - rows.keys(), rows.keys() - expected
        raise ValueError(f"score coverage mismatch missing={len(missing)} extra={len(extra)}")
    model = dict(payloads[0]["model"])
    model["moe_layers"] = complete
    return {
        "format": FORMAT,
        "model": model,
        "measurement": payloads[0]["measurement"],
        "calibration": payloads[0]["calibration"],
        "source": payloads[0]["source"],
        "lanes": [
            {"path": str(path.resolve()), "sha256": sha256(path)} for path in paths
        ],
        "scores": [rows[key] for key in sorted(rows)],
    }


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="bw24-merge-quant-sensitivity-") as tmp:
        root = Path(tmp); paths = []
        for layer in (1, 2):
            path = root / f"lane-{layer}.json"; paths.append(path)
            path.write_text(json.dumps({
                "format": FORMAT,
                "model": {"expert_count": 2, "moe_layers": [layer],
                          "complete_moe_layers": [1, 2]},
                "measurement": {"qtypes": ["Q2_K"]},
                "calibration": {"public_eval_data_used_for_selection": False},
                "source": {"index_sha256": "a"},
                "scores": [
                    {"layer": layer, "expert": expert, "quantization": {"Q2_K": {}}}
                    for expert in range(2)
                ],
            }))
        result = merge(paths)
        assert len(result["scores"]) == 4
        assert result["model"]["moe_layers"] == [1, 2]


def main() -> None:
    if sys.argv[1:] == ["--self-test"]:
        self_test(); print("Hy3 quant sensitivity merge self-test: PASS"); return
    parser = argparse.ArgumentParser()
    parser.add_argument("lanes", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=False)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(); print("Hy3 quant sensitivity merge self-test: PASS"); return
    if args.out is None:
        raise SystemExit("--out is required")
    result = merge(args.lanes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out} sha256={sha256(args.out)} rows={len(result['scores'])}")


if __name__ == "__main__":
    main()
