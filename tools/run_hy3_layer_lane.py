#!/usr/bin/env python3
"""Run one restartable GPU lane of Hy3 scoring or prune healing, one layer at a time."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


SCORE_FORMAT = "bw24-expert-retention-scores-v1"
HEAL_FORMAT = "bw24-hy3-prune-heal-layer-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_layers(raw: str) -> list[int]:
    if "-" in raw:
        lo, hi = (int(value) for value in raw.split("-", 1))
        if lo > hi:
            raise ValueError("layer range is descending")
        return list(range(lo, hi + 1))
    layers = [int(value) for value in raw.split(",") if value]
    if not layers:
        raise ValueError("at least one layer is required")
    return layers


def lane_layers(layers: list[int], lane_index: int, lane_count: int) -> list[int]:
    if lane_count <= 0 or lane_index < 0 or lane_index >= lane_count:
        raise ValueError("lane-index must be in [0, lane-count)")
    return [layer for position, layer in enumerate(layers) if position % lane_count == lane_index]


def valid_score(path: Path, layer: int, expert_count: int) -> bool:
    try:
        value = json.loads(path.read_text())
        if value.get("format") != SCORE_FORMAT:
            return False
        if value.get("calibration", {}).get("public_eval_data_used_for_selection") is not False:
            return False
        if [int(item) for item in value["model"]["moe_layers"]] != [layer]:
            return False
        rows = value["scores"]
        keys = {(int(row["layer"]), int(row["expert"])) for row in rows}
        if len(rows) != expert_count or keys != {(layer, expert) for expert in range(expert_count)}:
            return False
        targets = value.get("teacher_targets", {})
        if set(targets) != {str(layer)}:
            return False
        target = targets[str(layer)]
        target_path = Path(target["path"])
        return (
            target_path.stat().st_size == int(target["bytes"])
            and sha256(target_path) == target["sha256"]
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def valid_heal(
    receipt_path: Path, layer: int, mode: str, plan: Path, scores: Path
) -> bool:
    try:
        value = json.loads(receipt_path.read_text())
        if (
            value.get("format") != HEAL_FORMAT
            or int(value["layer"]) != layer
            or value["mode"] != mode
            or value.get("public_eval_data_used_for_healing") is not False
            or value["plan"]["sha256"] != sha256(plan)
            or value["scores"]["sha256"] != sha256(scores)
        ):
            return False
        output = value["output"]
        output_path = Path(output["path"])
        return (
            output_path.stat().st_size == int(output["bytes"])
            and sha256(output_path) == output["sha256"]
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def common_receipt(args: argparse.Namespace, layers: list[int], outputs: list[dict[str, Any]]) -> dict:
    inputs = {
        "worker_tool": {"path": str(args.tool.resolve()), "sha256": sha256(args.tool)},
        "source_config": {
            "path": str((args.source_dir / "config.json").resolve()),
            "sha256": sha256(args.source_dir / "config.json"),
        },
        "source_index": {
            "path": str((args.source_dir / "model.safetensors.index.json").resolve()),
            "sha256": sha256(args.source_dir / "model.safetensors.index.json"),
        },
    }
    if args.command == "score":
        inputs.update({
            "trace_lock": {"path": str(args.trace_lock.resolve()), "sha256": sha256(args.trace_lock)},
            "weight_trace": {"path": str(args.weight_trace.resolve()), "sha256": sha256(args.weight_trace)},
            "requests": {"path": str(args.requests.resolve()), "sha256": sha256(args.requests)},
        })
    else:
        inputs.update({
            "plan": {"path": str(args.plan.resolve()), "sha256": sha256(args.plan)},
            "scores": {"path": str(args.scores.resolve()), "sha256": sha256(args.scores)},
        })
    return {
        "format": "bw24-hy3-layer-lane-v1",
        "command": args.command,
        "lane_index": args.lane_index,
        "lane_count": args.lane_count,
        "layers": layers,
        "inputs": inputs,
        "outputs": outputs,
    }


def run_score(args: argparse.Namespace, layers: list[int]) -> list[dict[str, Any]]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.teacher_target_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for layer in layers:
        out = args.out_dir / f"layer-{layer:03}.json"
        if not valid_score(out, layer, args.expert_count):
            out.unlink(missing_ok=True)
            command = [
                sys.executable, str(args.tool),
                "--trace-lock", str(args.trace_lock),
                "--weight-trace", str(args.weight_trace),
                "--requests", str(args.requests),
                "--source-dir", str(args.source_dir),
                "--layers", str(layer),
                "--expert-count", str(args.expert_count),
                "--top-k", str(args.top_k),
                "--hidden-size", str(args.hidden_size),
                "--intermediate-size", str(args.intermediate_size),
                "--batch-tokens", str(args.batch_tokens),
                "--sketch-dim", str(args.sketch_dim),
                "--seed", str(args.seed),
                "--reap-weight", str(args.reap_weight),
                "--traffic-weight", str(args.traffic_weight),
                "--diversity-weight", str(args.diversity_weight),
                "--rare-weight", str(args.rare_weight),
                "--protect-per-stratum", str(args.protect_per_stratum),
                "--device", args.device,
                "--teacher-target-dir", str(args.teacher_target_dir),
                "--out", str(out),
            ]
            subprocess.run(command, check=True)
        if not valid_score(out, layer, args.expert_count):
            raise RuntimeError(f"layer {layer} score output failed validation")
        outputs.append({"layer": layer, "path": str(out.resolve()), "sha256": sha256(out)})
        print(f"score lane {args.lane_index}: layer {layer} complete", flush=True)
    return outputs


def run_heal(args: argparse.Namespace, layers: list[int]) -> list[dict[str, Any]]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.receipt_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for layer in layers:
        shard = args.out_dir / f"layer-{layer:03}.safetensors"
        receipt = args.receipt_dir / f"layer-{layer:03}.receipt.json"
        if not valid_heal(receipt, layer, args.mode, args.plan, args.scores):
            shard.unlink(missing_ok=True)
            receipt.unlink(missing_ok=True)
            command = [
                sys.executable, str(args.tool),
                "--mode", args.mode,
                "--layer", str(layer),
                "--plan", str(args.plan),
                "--scores", str(args.scores),
                "--source-dir", str(args.source_dir),
                "--expert-count", str(args.expert_count),
                "--top-k", str(args.top_k),
                "--hidden-size", str(args.hidden_size),
                "--intermediate-size", str(args.intermediate_size),
                "--rank", str(args.rank),
                "--lora-alpha", str(args.lora_alpha),
                "--steps", str(args.steps),
                "--batch-tokens", str(args.batch_tokens),
                "--eval-batch-tokens", str(args.eval_batch_tokens),
                "--learning-rate", str(args.learning_rate),
                "--bias-learning-rate", str(args.bias_learning_rate),
                "--bias-max-delta", str(args.bias_max_delta),
                "--router-anchor-weight", str(args.router_anchor_weight),
                "--max-grad-norm", str(args.max_grad_norm),
                "--holdout-modulus", str(args.holdout_modulus),
                "--log-every", str(args.log_every),
                "--seed", str(args.seed),
                "--device", args.device,
                "--out-shard", str(shard),
                "--receipt", str(receipt),
            ]
            subprocess.run(command, check=True)
        if not valid_heal(receipt, layer, args.mode, args.plan, args.scores):
            raise RuntimeError(f"layer {layer} heal output failed validation")
        outputs.append({
            "layer": layer, "receipt": str(receipt.resolve()), "sha256": sha256(receipt)
        })
        print(f"{args.mode} lane {args.lane_index}: layer {layer} complete", flush=True)
    return outputs


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tool", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--lane-index", type=int, required=True)
    parser.add_argument("--lane-count", type=int, default=8)
    parser.add_argument("--layers", default="1-79")
    parser.add_argument("--expert-count", type=int, default=192)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=1536)
    parser.add_argument("--batch-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lane-receipt", type=Path, required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    score = sub.add_parser("score")
    add_common(score)
    score.add_argument("--trace-lock", type=Path, required=True)
    score.add_argument("--weight-trace", type=Path, required=True)
    score.add_argument("--requests", type=Path, required=True)
    score.add_argument("--teacher-target-dir", type=Path, required=True)
    score.add_argument("--sketch-dim", type=int, default=32)
    score.add_argument("--reap-weight", type=float, default=0.65)
    score.add_argument("--traffic-weight", type=float, default=0.10)
    score.add_argument("--diversity-weight", type=float, default=0.15)
    score.add_argument("--rare-weight", type=float, default=0.10)
    score.add_argument("--protect-per-stratum", type=int, default=2)
    heal = sub.add_parser("heal")
    add_common(heal)
    heal.add_argument("--mode", choices=("router", "joint"), required=True)
    heal.add_argument("--plan", type=Path, required=True)
    heal.add_argument("--scores", type=Path, required=True)
    heal.add_argument("--receipt-dir", type=Path, required=True)
    heal.add_argument("--rank", type=int, default=8)
    heal.add_argument("--lora-alpha", type=float, default=8.0)
    heal.add_argument("--steps", type=int, default=600)
    heal.add_argument("--eval-batch-tokens", type=int, default=256)
    heal.add_argument("--learning-rate", type=float, default=2e-4)
    heal.add_argument("--bias-learning-rate", type=float, default=0.01)
    heal.add_argument("--bias-max-delta", type=float, default=0.1)
    heal.add_argument("--router-anchor-weight", type=float, default=1e-4)
    heal.add_argument("--max-grad-norm", type=float, default=1.0)
    heal.add_argument("--holdout-modulus", type=int, default=10)
    heal.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


def self_test() -> None:
    assert lane_layers(list(range(1, 10)), 0, 3) == [1, 4, 7]
    assert lane_layers(list(range(1, 10)), 2, 3) == [3, 6, 9]
    with tempfile.TemporaryDirectory(prefix="bw24-layer-lane-") as tmp:
        root = Path(tmp)
        target = root / "target.f32"; target.write_bytes(b"target")
        score = root / "score.json"
        score.write_text(json.dumps({
            "format": SCORE_FORMAT,
            "model": {"moe_layers": [1]},
            "calibration": {"public_eval_data_used_for_selection": False},
            "teacher_targets": {"1": {
                "path": str(target), "bytes": target.stat().st_size, "sha256": sha256(target),
            }},
            "scores": [{"layer": 1, "expert": expert} for expert in range(2)],
        }))
        assert valid_score(score, 1, 2)
        plan = root / "plan.json"; plan.write_text("plan")
        scores = root / "scores.json"; scores.write_text("scores")
        output = root / "heal.bin"; output.write_bytes(b"heal")
        receipt = root / "heal.json"
        receipt.write_text(json.dumps({
            "format": HEAL_FORMAT, "layer": 1, "mode": "joint",
            "public_eval_data_used_for_healing": False,
            "plan": {"sha256": sha256(plan)}, "scores": {"sha256": sha256(scores)},
            "output": {"path": str(output), "bytes": output.stat().st_size, "sha256": sha256(output)},
        }))
        assert valid_heal(receipt, 1, "joint", plan, scores)


def main() -> None:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        print("Hy3 layer lane runner self-test: PASS")
        return
    args = parse_args()
    layers = lane_layers(parse_layers(args.layers), args.lane_index, args.lane_count)
    outputs = run_score(args, layers) if args.command == "score" else run_heal(args, layers)
    receipt = common_receipt(args, layers, outputs)
    args.lane_receipt.parent.mkdir(parents=True, exist_ok=True)
    args.lane_receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"lane {args.lane_index}/{args.lane_count} complete: {len(layers)} layers")


if __name__ == "__main__":
    main()
