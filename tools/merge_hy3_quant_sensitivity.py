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
    importance_sidecars: dict[str, dict[str, Any]] = {}
    for path, item in zip(paths, payloads):
        lane_layers = {str(int(layer)) for layer in item["model"]["moe_layers"]}
        sidecars = item.get("importance_sidecars", {})
        if sidecars and set(sidecars) != lane_layers:
            raise ValueError(f"importance sidecar coverage differs from lane layers in {path}")
        for layer, receipt in sidecars.items():
            if layer in importance_sidecars:
                raise ValueError(f"duplicate importance sidecar for layer {layer}")
            importance_sidecars[layer] = receipt
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
        "importance_sidecars": importance_sidecars,
        "lanes": [
            {"path": str(path.resolve()), "sha256": sha256(path)} for path in paths
        ],
        "scores": [rows[key] for key in sorted(rows)],
    }


def _quantizer_provenance(measurement: dict[str, Any]) -> dict[str, Any]:
    raw = measurement.get("exact_quantizer_implementation")
    if isinstance(raw, dict):
        return raw
    return {qtype: {"implementation": raw} for qtype in measurement["qtypes"]}


def merge_qtypes(paths: list[Path]) -> dict[str, Any]:
    """Merge full-coverage maps that measured disjoint qtype sets on identical private data."""
    if len(paths) < 2:
        raise ValueError("qtype merge requires at least two maps")
    payloads = [json.loads(path.read_text()) for path in paths]
    if any(item.get("format") != FORMAT for item in payloads):
        raise ValueError(f"all maps must use format {FORMAT}")
    for field in ("model", "calibration", "source"):
        if len({canonical(item[field]) for item in payloads}) != 1:
            raise ValueError(f"qtype map {field} provenance differs")
    common_measurement = (
        "max_tokens_per_expert", "sampling", "metric", "projection_ablation",
    )
    for field in common_measurement:
        if len({canonical(item["measurement"].get(field)) for item in payloads}) != 1:
            raise ValueError(f"qtype map measurement.{field} differs")

    qtypes: list[str] = []
    quantizer_provenance: dict[str, Any] = {}
    for path, item in zip(paths, payloads):
        for qtype in item["measurement"]["qtypes"]:
            if qtype in quantizer_provenance:
                raise ValueError(f"duplicate qtype {qtype} in {path}")
            qtypes.append(qtype)
            quantizer_provenance[qtype] = _quantizer_provenance(item["measurement"])[qtype]

    row_sets = []
    for item in payloads:
        row_sets.append({(int(row["layer"]), int(row["expert"])): row for row in item["scores"]})
    if any(rows.keys() != row_sets[0].keys() for rows in row_sets[1:]):
        raise ValueError("qtype map score coverage differs")
    merged_rows = []
    for key in sorted(row_sets[0]):
        base = dict(row_sets[0][key])
        quantization: dict[str, Any] = {}
        for rows in row_sets:
            row = rows[key]
            left = {name: value for name, value in base.items() if name != "quantization"}
            right = {name: value for name, value in row.items() if name != "quantization"}
            if canonical(left) != canonical(right):
                raise ValueError(f"qtype map routed evidence differs for {key}")
            overlap = quantization.keys() & row["quantization"].keys()
            if overlap:
                raise ValueError(f"duplicate quantization evidence for {key}: {sorted(overlap)}")
            quantization.update(row["quantization"])
        base["quantization"] = quantization
        merged_rows.append(base)

    importance_sidecars: dict[str, dict[str, Any]] = {}
    for item in payloads:
        for layer, receipt in item.get("importance_sidecars", {}).items():
            previous = importance_sidecars.get(layer)
            if previous is not None and canonical(previous) != canonical(receipt):
                raise ValueError(f"qtype maps bind different importance sidecars for layer {layer}")
            importance_sidecars[layer] = receipt
    measurement = {
        field: payloads[0]["measurement"].get(field) for field in common_measurement
    }
    measurement.update({
        "qtypes": qtypes,
        "exact_quantizer_implementation": quantizer_provenance,
        "importance_metric": next(
            (
                item["measurement"]["importance_metric"] for item in payloads
                if item["measurement"].get("importance_metric")
            ),
            None,
        ),
    })
    return {
        "format": FORMAT,
        "model": payloads[0]["model"],
        "measurement": measurement,
        "calibration": payloads[0]["calibration"],
        "source": payloads[0]["source"],
        "importance_sidecars": importance_sidecars,
        "qtype_components": [
            {"path": str(path.resolve()), "sha256": sha256(path)} for path in paths
        ],
        "scores": merged_rows,
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
        full = root / "full-q2.json"
        full.write_text(json.dumps(result))
        q4 = json.loads(full.read_text())
        q4["measurement"] = {"qtypes": ["Q4_K"]}
        for row in q4["scores"]:
            row["quantization"] = {"Q4_K": {}}
        q4_path = root / "full-q4.json"; q4_path.write_text(json.dumps(q4))
        combined = merge_qtypes([full, q4_path])
        assert combined["measurement"]["qtypes"] == ["Q2_K", "Q4_K"]
        assert set(combined["scores"][0]["quantization"]) == {"Q2_K", "Q4_K"}


def main() -> None:
    if sys.argv[1:] == ["--self-test"]:
        self_test(); print("Hy3 quant sensitivity merge self-test: PASS"); return
    parser = argparse.ArgumentParser()
    parser.add_argument("lanes", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=False)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--merge-qtypes", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(); print("Hy3 quant sensitivity merge self-test: PASS"); return
    if args.out is None:
        raise SystemExit("--out is required")
    result = merge_qtypes(args.lanes) if args.merge_qtypes else merge(args.lanes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out} sha256={sha256(args.out)} rows={len(result['scores'])}")


if __name__ == "__main__":
    main()
