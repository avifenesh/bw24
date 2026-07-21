#!/usr/bin/env python3
"""Validate the private-only 120/137 GB layer-balanced bridge lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


FORMAT = "bw24-hy3-layer-balanced-bridge-lock-v1"
ARMS = {"layer_balanced120": 120_000_000_000, "layer_balanced137": 137_459_192_320}
QTYPES = ["Q8_0", "NVFP4", "Q3_K", "Q2_K", "IQ3_S", "IQ4_XS", "Q4_K"]
INPUTS = {
    "retention_scores", "confidence_plan", "quant_sensitivity", "layer_constraints",
    "reference_plan", "trace_lock", "layer100_plan", "source_config", "source_index",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(data: dict, verify_inputs: bool) -> None:
    if data.get("format") != FORMAT:
        raise ValueError("wrong bridge lock format")
    if data.get("public_eval_data_used_for_selection") is not False:
        raise ValueError("bridge allocation must be private-only")
    arms = data.get("arms")
    if not isinstance(arms, dict) or {
        name: row.get("target_logical_bytes") for name, row in arms.items()
    } != ARMS:
        raise ValueError("bridge arm names or target bytes changed")
    allocator = data.get("allocator", {})
    if allocator.get("candidate_qtypes") != QTYPES:
        raise ValueError("candidate qtypes changed")
    if allocator.get("importance_weights") != {"retention": 0.0, "confidence": 0.0, "layer": 0.0}:
        raise ValueError("bridge importance weights changed")
    if allocator.get("min_survivors_per_layer") != 96 or allocator.get("mip_relative_gap") != 1e-6:
        raise ValueError("allocator limits changed")
    inputs = data.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != INPUTS:
        raise ValueError("bridge inputs changed")
    for name, row in inputs.items():
        path = Path(row.get("path", ""))
        expected = row.get("sha256")
        if not path.is_absolute() or not isinstance(expected, str) or not SHA256_RE.fullmatch(expected):
            raise ValueError(f"invalid locked input {name}")
        if verify_inputs:
            if not path.is_file() or sha256(path) != expected:
                raise ValueError(f"locked input mismatch: {name}")
    receipts = data.get("joint_receipts", {})
    receipt_dir = Path(receipts.get("path", ""))
    inventory_sha = receipts.get("inventory_sha256")
    if receipts.get("count") != 79 or not receipt_dir.is_absolute() or not SHA256_RE.fullmatch(str(inventory_sha)):
        raise ValueError("invalid joint receipt lock")
    if verify_inputs:
        paths = sorted(receipt_dir.glob("layer-*.receipt.json"))
        if len(paths) != 79:
            raise ValueError("joint receipt count changed")
        lines = b"".join(f"{sha256(path)}  ./{path.name}\n".encode() for path in paths)
        if hashlib.sha256(lines).hexdigest() != inventory_sha:
            raise ValueError("joint receipt inventory changed")


def self_test() -> None:
    lock = Path(__file__).with_name("layer-balanced-bridge.lock.json")
    data = json.loads(lock.read_text())
    validate(data, False)
    changed = json.loads(json.dumps(data))
    changed["public_eval_data_used_for_selection"] = True
    try:
        validate(changed, False)
    except ValueError:
        pass
    else:
        raise AssertionError("public-derived bridge lock was accepted")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=Path(__file__).with_name("layer-balanced-bridge.lock.json"))
    parser.add_argument("--verify-inputs", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("layer-balanced bridge lock self-test: PASS")
        return
    validate(json.loads(args.lock.read_text()), args.verify_inputs)
    print("layer-balanced bridge lock validation: PASS")


if __name__ == "__main__":
    main()
