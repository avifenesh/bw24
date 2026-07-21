#!/usr/bin/env python3
"""Validate frozen private held-out and routing gates before public evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
import tempfile
from typing import Any


LOCK_FORMAT = "bw24-hy3-prune-heal-overlay-v1"
GATE_FORMAT = "bw24-hy3-heal-quality-gates-v1"
OUTPUT_FORMAT = "bw24-hy3-heal-quality-gate-v1"


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_nonnegative(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) >= 0


def validate_mode(
    path: pathlib.Path, expected_mode: str, gates: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lock = json.loads(path.read_text())
    if lock.get("format") != LOCK_FORMAT or lock.get("mode") != expected_mode:
        raise ValueError(f"{path}: wrong heal lock format or mode")
    expected_layers = int(gates["expected_layers"])
    if lock.get("layers") != list(range(1, expected_layers + 1)):
        raise ValueError(f"{path}: layer coverage is not exactly 1..{expected_layers}")

    rows = []
    for shard in lock.get("shards", []):
        receipt_path = pathlib.Path(shard["receipt_path"])
        if sha256(receipt_path) != shard["receipt_sha256"]:
            raise ValueError(f"{receipt_path}: receipt hash changed")
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("mode") != expected_mode:
            raise ValueError(f"{receipt_path}: mode differs from overlay lock")
        before, after = receipt["before"], receipt["after"]
        metrics = {
            "layer": int(receipt["layer"]),
            "active_experts": len(receipt["active_experts"]),
            "before_normalized_mse": before["normalized_mse"],
            "after_normalized_mse": after["normalized_mse"],
            "before_routing_entropy_nats": before["routing_entropy_nats"],
            "after_routing_entropy_nats": after["routing_entropy_nats"],
            "after_dead_active_experts": after["dead_active_experts"],
            "after_max_active_load_fraction": after["max_active_load_fraction"],
        }
        if not all(finite_nonnegative(value) for key, value in metrics.items() if key != "layer"):
            raise ValueError(f"{receipt_path}: non-finite or negative quality metric")
        rows.append(metrics)
    if len(rows) != expected_layers or sorted(row["layer"] for row in rows) != list(
        range(1, expected_layers + 1)
    ):
        raise ValueError(f"{path}: receipt coverage is incomplete or duplicated")
    return lock, rows


def evaluate(router_path: pathlib.Path, joint_path: pathlib.Path, gates: dict[str, Any]) -> dict[str, Any]:
    if gates.get("format") != GATE_FORMAT:
        raise ValueError("wrong heal-quality gate format")
    router_lock, router_rows = validate_mode(router_path, "router", gates)
    joint_lock, joint_rows = validate_mode(joint_path, "joint", gates)
    if router_lock["plan"] != joint_lock["plan"] or router_lock["scores"] != joint_lock["scores"]:
        raise ValueError("router and joint heals use different plan or score evidence")

    safety = gates["routing_safety"]
    modes = {}
    for mode, rows in (("router", router_rows), ("joint", joint_rows)):
        checks = []
        for row in rows:
            active = max(int(row["active_experts"]), 1)
            before_entropy = float(row["before_routing_entropy_nats"])
            entropy_floor = before_entropy * float(safety["min_entropy_ratio_vs_before"])
            checks.append({
                "layer": row["layer"],
                "dead_fraction_ok": float(row["after_dead_active_experts"]) / active
                <= float(safety["max_dead_active_fraction"]),
                "max_load_ok": float(row["after_max_active_load_fraction"])
                <= float(safety["max_active_load_fraction"]),
                "entropy_ok": float(row["after_routing_entropy_nats"]) >= entropy_floor,
            })
        modes[mode] = {
            "routing_safety_passed": all(
                item[check] for item in checks for check in ("dead_fraction_ok", "max_load_ok", "entropy_ok")
            ),
            "layer_checks": checks,
        }

    before_mean = sum(float(row["before_normalized_mse"]) for row in joint_rows) / len(joint_rows)
    after_mean = sum(float(row["after_normalized_mse"]) for row in joint_rows) / len(joint_rows)
    improved = sum(
        float(row["after_normalized_mse"]) < float(row["before_normalized_mse"])
        for row in joint_rows
    )
    joint_cfg = gates["joint_reconstruction"]
    reconstruction_checks = {
        "mean_improved": (
            after_mean < before_mean
            if joint_cfg["require_mean_normalized_mse_improvement"]
            else True
        ),
        "improved_layer_fraction": improved / len(joint_rows)
        >= float(joint_cfg["min_improved_layer_fraction"]),
    }
    passed = (
        modes["router"]["routing_safety_passed"]
        and modes["joint"]["routing_safety_passed"]
        and all(reconstruction_checks.values())
    )
    return {
        "format": OUTPUT_FORMAT,
        "passed": passed,
        "modes": modes,
        "joint_reconstruction": {
            "mean_before_normalized_mse": before_mean,
            "mean_after_normalized_mse": after_mean,
            "improved_layers": improved,
            "total_layers": len(joint_rows),
            "checks": reconstruction_checks,
        },
        "evidence": {
            "router_lock": {"path": str(router_path.resolve()), "sha256": sha256(router_path)},
            "joint_lock": {"path": str(joint_path.resolve()), "sha256": sha256(joint_path)},
        },
    }


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="bw24-heal-quality-") as tmp:
        root = pathlib.Path(tmp)
        gates = {
            "format": GATE_FORMAT,
            "expected_layers": 2,
            "routing_safety": {
                "max_dead_active_fraction": 0.25,
                "max_active_load_fraction": 0.25,
                "min_entropy_ratio_vs_before": 0.5,
            },
            "joint_reconstruction": {
                "require_mean_normalized_mse_improvement": True,
                "min_improved_layer_fraction": 0.5,
            },
        }
        paths = {}
        for mode in ("router", "joint"):
            shards = []
            for layer in (1, 2):
                receipt_path = root / f"{mode}-{layer}.json"
                before = 1.0
                after = 0.8 if mode == "joint" else 1.1
                receipt_path.write_text(json.dumps({
                    "mode": mode,
                    "layer": layer,
                    "active_experts": list(range(8)),
                    "before": {
                        "normalized_mse": before,
                        "routing_entropy_nats": 2.0,
                    },
                    "after": {
                        "normalized_mse": after,
                        "routing_entropy_nats": 1.5,
                        "dead_active_experts": 0,
                        "max_active_load_fraction": 0.2,
                    },
                }))
                shards.append({
                    "receipt_path": str(receipt_path),
                    "receipt_sha256": sha256(receipt_path),
                })
            lock_path = root / f"{mode}.lock.json"
            lock_path.write_text(json.dumps({
                "format": LOCK_FORMAT,
                "mode": mode,
                "layers": [1, 2],
                "plan": {"sha256": "plan"},
                "scores": {"sha256": "scores"},
                "shards": shards,
            }))
            paths[mode] = lock_path
        result = evaluate(paths["router"], paths["joint"], gates)
        assert result["passed"]
        assert result["joint_reconstruction"]["improved_layers"] == 2

        receipt_path = pathlib.Path(json.loads(paths["joint"].read_text())["shards"][0]["receipt_path"])
        receipt = json.loads(receipt_path.read_text())
        receipt["after"]["routing_entropy_nats"] = 0.1
        receipt_path.write_text(json.dumps(receipt))
        joint_lock = json.loads(paths["joint"].read_text())
        joint_lock["shards"][0]["receipt_sha256"] = sha256(receipt_path)
        paths["joint"].write_text(json.dumps(joint_lock))
        assert not evaluate(paths["router"], paths["joint"], gates)["passed"]


def main() -> None:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        print("Hy3 heal quality gate self-test: PASS")
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-lock", type=pathlib.Path, required=True)
    parser.add_argument("--joint-lock", type=pathlib.Path, required=True)
    parser.add_argument("--gate-lock", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    gates = json.loads(args.gate_lock.read_text())
    result = evaluate(args.router_lock, args.joint_lock, gates)
    result["evidence"]["gate_lock"] = {
        "path": str(args.gate_lock.resolve()),
        "sha256": sha256(args.gate_lock),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not result["passed"]:
        raise SystemExit("Hy3 heal quality gate failed")
    print(f"wrote {args.output} passed=true")


if __name__ == "__main__":
    main()
