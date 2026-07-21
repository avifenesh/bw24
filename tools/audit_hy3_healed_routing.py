#!/usr/bin/env python3
"""Audit post-heal routing coverage on every frozen private calibration token."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from build_hy3_reap_scores import parse_layers
from heal_hy3_pruned_layer import bias_name, router_name


FORMAT = "bw24-hy3-post-heal-routing-audit-v1"
PLAN_FORMAT = "bw24-expert-tier-plan-v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def audit(args: argparse.Namespace) -> dict[str, Any]:
    plan = json.loads(args.plan.read_text())
    lock = json.loads(args.trace_lock.read_text())
    if plan.get("format") != PLAN_FORMAT:
        raise ValueError("unsupported plan format")
    if plan.get("calibration", {}).get("public_eval_data_used_for_selection") is not False:
        raise ValueError("routing audit requires a private-only plan")
    layers = parse_layers(args.layers)
    if not set(layers) <= {int(x) for x in plan["model"]["moe_layers"]}:
        raise ValueError("audit layers are not covered by the plan")
    if not isinstance(lock.get("trace_dir"), str) or not isinstance(lock.get("files"), dict):
        raise ValueError("invalid hidden-trace lock")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
    expert_count = int(plan["model"]["expert_count"])
    trace_dir = Path(lock["trace_dir"])
    result: dict[str, Any] = {}
    for layer in layers:
        shard = args.overlay_dir / f"layer-{layer:03}.safetensors"
        hidden_path = trace_dir / f"layer-{layer:03}.f32"
        receipt = lock["files"].get(hidden_path.name)
        if not shard.is_file() or not isinstance(receipt, dict):
            raise ValueError(f"missing overlay or trace for layer {layer}")
        if sha256(hidden_path) != receipt["sha256"]:
            raise ValueError(f"layer {layer} hidden trace hash mismatch")
        hidden_size = int(lock["hidden_size"])
        if hidden_path.stat().st_size % (4 * hidden_size):
            raise ValueError(f"layer {layer} hidden trace has invalid extent")
        tokens = hidden_path.stat().st_size // (4 * hidden_size)
        hidden = np.memmap(hidden_path, dtype="<f4", mode="r", shape=(tokens, hidden_size))
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            names = set(handle.keys())
            if router_name(layer) not in names or bias_name(layer) not in names:
                raise ValueError(f"layer {layer} overlay lacks router or correction bias")
            router = handle.get_tensor(router_name(layer)).float().to(device)
            bias = handle.get_tensor(bias_name(layer)).float().to(device)
        if router.shape != (expert_count, hidden_size) or bias.shape != (expert_count,):
            raise ValueError(f"layer {layer} router/bias shape mismatch")
        pruned = {int(x) for x in plan["pruned_experts"].get(str(layer), [])}
        active = torch.ones(expert_count, dtype=torch.bool, device=device)
        if pruned:
            active[torch.tensor(sorted(pruned), dtype=torch.long, device=device)] = False
        counts = torch.zeros(expert_count, dtype=torch.int64, device=device)
        for start in range(0, tokens, args.batch_tokens):
            values = torch.from_numpy(
                np.array(hidden[start : start + args.batch_tokens], dtype=np.float32, copy=True)
            ).to(device)
            choice = torch.sigmoid(torch.nn.functional.linear(values.float(), router)) + bias
            choice.masked_fill_(~active, float("-inf"))
            selected = torch.topk(choice, args.top_k, dim=-1, sorted=False).indices
            counts += torch.bincount(selected.reshape(-1), minlength=expert_count)
        active_ids = torch.where(active)[0]
        active_counts = counts[active_ids]
        dead_local = torch.where(active_counts == 0)[0]
        dead_ids = active_ids[dead_local].cpu().tolist()
        total = int(active_counts.sum().cpu())
        result[str(layer)] = {
            "tokens": int(tokens),
            "route_assignments": total,
            "active_experts": int(active.sum().cpu()),
            "pruned_experts": len(pruned),
            "dead_active_experts": len(dead_ids),
            "dead_active_expert_ids": dead_ids,
            "max_active_load_fraction": float(active_counts.max().cpu()) / max(total, 1),
            "overlay_sha256": sha256(shard),
            "hidden_trace_sha256": receipt["sha256"],
        }
        del hidden, router, bias, active, counts
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {
        "format": FORMAT,
        "plan": {"path": str(args.plan.resolve()), "sha256": sha256(args.plan)},
        "trace_lock": {"path": str(args.trace_lock.resolve()), "sha256": sha256(args.trace_lock)},
        "overlay_dir": str(args.overlay_dir.resolve()),
        "layers": result,
        "summary": {
            "layers": len(result),
            "dead_active_experts": sum(x["dead_active_experts"] for x in result.values()),
            "all_layers_have_full_active_coverage": all(
                x["dead_active_experts"] == 0 for x in result.values()
            ),
        },
        "public_eval_data_used": False,
    }


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="bw24-routing-audit-") as tmp:
        root = Path(tmp); overlay = root / "overlay"; trace = root / "trace"
        overlay.mkdir(); trace.mkdir()
        hidden = np.eye(4, dtype="<f4").repeat(4, axis=0)
        hidden_path = trace / "layer-001.f32"; hidden.tofile(hidden_path)
        router = torch.tensor([[4, 0, 0, 0], [0, 4, 0, 0], [0, 0, 4, 0]], dtype=torch.float32)
        save_file({router_name(1): router, bias_name(1): torch.zeros(3)}, overlay / "layer-001.safetensors")
        lock = root / "lock.json"
        lock.write_text(json.dumps({"trace_dir": str(trace), "hidden_size": 4,
            "files": {hidden_path.name: {"sha256": sha256(hidden_path)}}}))
        plan = root / "plan.json"
        plan.write_text(json.dumps({"format": PLAN_FORMAT,
            "model": {"expert_count": 3, "moe_layers": [1]}, "pruned_experts": {"1": []},
            "calibration": {"public_eval_data_used_for_selection": False}}))
        args = argparse.Namespace(plan=plan, trace_lock=lock, overlay_dir=overlay,
            layers="1", top_k=1, batch_tokens=3, device="cpu")
        result = audit(args)
        assert result["summary"]["all_layers_have_full_active_coverage"]
        assert result["layers"]["1"]["route_assignments"] == len(hidden)


def main() -> None:
    if sys.argv[1:] == ["--self-test"]:
        self_test(); print("Hy3 healed-routing audit self-test: PASS"); return
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--trace-lock", type=Path, required=True)
    parser.add_argument("--overlay-dir", type=Path, required=True)
    parser.add_argument("--layers", default="1-79")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--batch-tokens", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output} sha256={sha256(args.output)} dead={result['summary']['dead_active_experts']}")


if __name__ == "__main__":
    main()
