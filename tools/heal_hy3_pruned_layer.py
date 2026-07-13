#!/usr/bin/env python3
"""Functionally repair one pruned Hy3 MoE layer against frozen private teacher targets."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from ggml_quant_bridge import EXTERNAL_QTYPES, GgmlQuantBridge, load_importance_sidecar


PLAN_FORMAT = "bw24-expert-tier-plan-v2"
SCORE_FORMAT = "bw24-expert-retention-scores-v1"
CHECKPOINT_FORMAT = "bw24-hy3-prune-heal-layer-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_name(layer: int, expert: int, projection: str) -> str:
    return f"model.layers.{layer}.mlp.experts.{expert}.{projection}_proj.weight"


def router_name(layer: int) -> str:
    return f"model.layers.{layer}.mlp.router.gate.weight"


def bias_name(layer: int) -> str:
    return f"model.layers.{layer}.mlp.expert_bias"


def load_source_tensor(
    name: str,
    source_dir: Path,
    weight_map: dict[str, str],
    handles: dict[str, Any],
) -> torch.Tensor:
    if name not in weight_map:
        raise ValueError(f"source index is missing {name}")
    return handles[weight_map[name]].get_tensor(name)


def quantized_base(
    tensor: torch.Tensor,
    qtype: str,
    *,
    importance: np.ndarray | None = None,
    bridge: GgmlQuantBridge | None = None,
) -> torch.Tensor:
    """Round-trip through the artifact builder's exact bytes before repair."""
    from build_hy3_quant_sensitivity import quant_dequant
    restored, _ = quant_dequant(
        tensor, qtype, importance=importance, bridge=bridge
    )
    return restored


class LayerStudent(nn.Module):
    def __init__(
        self,
        *,
        active_ids: list[int],
        gate: torch.Tensor,
        up: torch.Tensor,
        down: torch.Tensor,
        router: torch.Tensor,
        correction_bias: torch.Tensor,
        router_scaling: float,
        top_k: int,
        expert_count: int,
        mode: str,
        rank: int,
        lora_alpha: float,
        seed: int,
    ):
        super().__init__()
        self.active_ids = active_ids
        self.top_k = top_k
        self.expert_count = expert_count
        self.router_scaling = router_scaling
        self.mode = mode
        self.rank = rank
        self.lora_scale = lora_alpha / rank if rank else 0.0
        self.register_buffer("gate_base", gate)
        self.register_buffer("up_base", up)
        self.register_buffer("down_base", down)
        self.router = nn.Parameter(router.float())
        self.register_buffer("initial_router", router.float().clone())
        self.register_buffer("correction_bias", correction_bias.float())
        self.register_buffer("initial_correction_bias", correction_bias.float().clone())
        active = torch.zeros(expert_count, dtype=torch.bool, device=router.device)
        active[active_ids] = True
        self.register_buffer("active_mask", active)
        original_to_local = torch.full(
            (expert_count,), -1, dtype=torch.long, device=router.device
        )
        original_to_local[active_ids] = torch.arange(len(active_ids), device=router.device)
        self.register_buffer("original_to_local", original_to_local)

        if mode == "joint":
            generator = torch.Generator(device=router.device)
            generator.manual_seed(seed)
            survivors, intermediate, hidden = gate.shape
            self.gate_a = nn.Parameter(
                torch.randn(survivors, rank, hidden, generator=generator, device=router.device) * 0.01
            )
            self.gate_b = nn.Parameter(torch.zeros(survivors, intermediate, rank, device=router.device))
            self.up_a = nn.Parameter(
                torch.randn(survivors, rank, hidden, generator=generator, device=router.device) * 0.01
            )
            self.up_b = nn.Parameter(torch.zeros(survivors, intermediate, rank, device=router.device))
            self.down_a = nn.Parameter(
                torch.randn(survivors, rank, intermediate, generator=generator, device=router.device) * 0.01
            )
            self.down_b = nn.Parameter(torch.zeros(survivors, hidden, rank, device=router.device))

    def _adapter(self, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return F.linear(F.linear(x.float(), a), b) * self.lora_scale

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = F.linear(hidden.float(), self.router)
        routing = torch.sigmoid(logits)
        choice = routing + self.correction_bias
        choice = choice.masked_fill(~self.active_mask, float("-inf"))
        _, selected = torch.topk(choice, self.top_k, dim=-1, sorted=False)
        weights = routing.gather(1, selected)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * self.router_scaling
        output = torch.zeros(hidden.shape[0], hidden.shape[1], dtype=torch.float32, device=hidden.device)
        for original_id in selected.unique().tolist():
            token, slot = torch.where(selected == original_id)
            local_id = int(self.original_to_local[original_id])
            current = hidden[token]
            gate = F.linear(current.to(torch.bfloat16), self.gate_base[local_id]).float()
            up = F.linear(current.to(torch.bfloat16), self.up_base[local_id]).float()
            if self.mode == "joint":
                gate = gate + self._adapter(current, self.gate_a[local_id], self.gate_b[local_id])
                up = up + self._adapter(current, self.up_a[local_id], self.up_b[local_id])
            activated = F.silu(gate) * up
            expert_output = F.linear(
                activated.to(torch.bfloat16), self.down_base[local_id]
            ).float()
            if self.mode == "joint":
                expert_output = expert_output + self._adapter(
                    activated, self.down_a[local_id], self.down_b[local_id]
                )
            output.index_add_(
                0, token, expert_output * weights[token, slot, None]
            )
        return output, selected, weights

    @torch.no_grad()
    def update_bias(
        self,
        selected: torch.Tensor,
        weights: torch.Tensor,
        target_load: torch.Tensor,
        learning_rate: float,
        max_delta: float,
    ) -> None:
        observed = torch.zeros(self.expert_count, dtype=torch.float32, device=selected.device)
        observed.scatter_add_(0, selected.reshape(-1), weights.reshape(-1).float())
        observed = observed * self.active_mask
        observed = observed / observed.sum().clamp_min(1.0)
        self.correction_bias += learning_rate * (target_load - observed)
        delta = self.correction_bias - self.initial_correction_bias
        delta[self.active_mask] -= delta[self.active_mask].mean()
        delta.clamp_(-max_delta, max_delta)
        self.correction_bias.copy_(self.initial_correction_bias + delta)

    @torch.no_grad()
    def merged_source_tensors(self, layer: int) -> dict[str, torch.Tensor]:
        result = {
            # Router top-k is discontinuous around near ties. Preserve the trained FP32 values;
            # export_hy3_router_overrides.py embeds these bytes as an F32 runtime override.
            router_name(layer): self.router.float().cpu().contiguous(),
            bias_name(layer): self.correction_bias.cpu().contiguous(),
        }
        if self.mode != "joint":
            return result
        for local, original in enumerate(self.active_ids):
            gate_delta = (self.gate_b[local] @ self.gate_a[local]) * self.lora_scale
            up_delta = (self.up_b[local] @ self.up_a[local]) * self.lora_scale
            down_delta = (self.down_b[local] @ self.down_a[local]) * self.lora_scale
            result[tensor_name(layer, original, "gate")] = (
                self.gate_base[local].float() + gate_delta
            ).to(torch.bfloat16).cpu().contiguous()
            result[tensor_name(layer, original, "up")] = (
                self.up_base[local].float() + up_delta
            ).to(torch.bfloat16).cpu().contiguous()
            result[tensor_name(layer, original, "down")] = (
                self.down_base[local].float() + down_delta
            ).to(torch.bfloat16).cpu().contiguous()
        return result


def load_student(
    args: argparse.Namespace,
    active_ids: list[int],
    config: dict[str, Any],
    weight_map: dict[str, str],
    device: torch.device,
    quant_assignments: dict[tuple[int, int, str], str] | None = None,
    quant_bridge: GgmlQuantBridge | None = None,
    quant_importance: dict[str, np.ndarray] | None = None,
) -> LayerStudent:
    names = [router_name(args.layer), bias_name(args.layer)] + [
        tensor_name(args.layer, expert, projection)
        for expert in active_ids
        for projection in ("gate", "up", "down")
    ]
    shards = sorted({weight_map[name] for name in names if name in weight_map})
    with contextlib.ExitStack() as stack:
        handles = {
            shard: stack.enter_context(
                safe_open(str(args.source_dir / shard), framework="pt", device="cpu")
            )
            for shard in shards
        }
        router = load_source_tensor(
            router_name(args.layer), args.source_dir, weight_map, handles
        ).to(device)
        bias = load_source_tensor(
            bias_name(args.layer), args.source_dir, weight_map, handles
        ).to(device)
        survivors = len(active_ids)
        gate = torch.empty(
            survivors, args.intermediate_size, args.hidden_size,
            dtype=torch.bfloat16, device=device,
        )
        up = torch.empty_like(gate)
        down = torch.empty(
            survivors, args.hidden_size, args.intermediate_size,
            dtype=torch.bfloat16, device=device,
        )
        for local, expert in enumerate(active_ids):
            for projection, destination in (("gate", gate), ("up", up), ("down", down)):
                source = load_source_tensor(
                    tensor_name(args.layer, expert, projection),
                    args.source_dir, weight_map, handles,
                )
                if quant_assignments is not None:
                    qtype = quant_assignments[(args.layer, expert, projection)]
                    importance = None
                    if qtype in EXTERNAL_QTYPES:
                        if quant_importance is None:
                            raise ValueError(f"{qtype} healing lacks private importance")
                        key = "input" if projection in ("gate", "up") else "down"
                        importance = quant_importance[key][expert]
                    source = quantized_base(
                        source, qtype, importance=importance, bridge=quant_bridge
                    )
                destination[local].copy_(source)
    return LayerStudent(
        active_ids=active_ids,
        gate=gate,
        up=up,
        down=down,
        router=router,
        correction_bias=bias,
        router_scaling=float(config["router_scaling_factor"]),
        top_k=args.top_k,
        expert_count=args.expert_count,
        mode=args.mode,
        rank=args.rank,
        lora_alpha=args.lora_alpha,
        seed=args.seed + args.layer,
    )


@torch.no_grad()
def evaluate(
    model: LayerStudent,
    hidden: np.memmap,
    teacher: np.memmap,
    indices: np.ndarray,
    batch_tokens: int,
    device: torch.device,
) -> dict[str, Any]:
    squared_error = 0.0
    teacher_energy = 0.0
    selected_counts = torch.zeros(model.expert_count, dtype=torch.float64, device=device)
    entropy_sum = 0.0
    rows = 0
    for start in range(0, len(indices), batch_tokens):
        batch_index = indices[start : start + batch_tokens]
        x = torch.from_numpy(np.asarray(hidden[batch_index])).to(device)
        target = torch.from_numpy(np.asarray(teacher[batch_index])).to(device)
        output, selected, weights = model(x)
        squared_error += float(torch.square(output - target).sum(dtype=torch.float64).cpu())
        teacher_energy += float(torch.square(target).sum(dtype=torch.float64).cpu())
        selected_counts += torch.bincount(
            selected.reshape(-1), minlength=model.expert_count
        ).to(torch.float64)
        normalized = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        entropy_sum += float(
            (-(normalized * normalized.clamp_min(1e-20).log()).sum(dim=-1)).sum().cpu()
        )
        rows += len(batch_index)
    active_counts = selected_counts[model.active_mask]
    return {
        "normalized_mse": squared_error / max(teacher_energy, 1e-30),
        "rmse": math.sqrt(squared_error / max(rows * hidden.shape[1], 1)),
        "routing_entropy_nats": entropy_sum / max(rows, 1),
        "dead_active_experts": int((active_counts == 0).sum().cpu()),
        "max_active_load_fraction": float(
            (active_counts.max() / active_counts.sum().clamp_min(1.0)).cpu()
        ),
    }


def heal(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode not in ("router", "joint"):
        raise ValueError("mode must be router or joint")
    plan = json.loads(args.plan.read_text())
    scores = json.loads(args.scores.read_text())
    if plan.get("format") != PLAN_FORMAT or scores.get("format") != SCORE_FORMAT:
        raise ValueError("unsupported plan or score format")
    if scores["calibration"].get("public_eval_data_used_for_selection") is not False:
        raise ValueError("score calibration provenance is not private")
    pruned = {int(expert) for expert in plan["pruned_experts"].get(str(args.layer), [])}
    active_ids = [expert for expert in range(args.expert_count) if expert not in pruned]
    if len(active_ids) < args.top_k:
        raise ValueError("pruning leaves fewer active experts than top-k")
    target_receipt = scores.get("teacher_targets", {}).get(str(args.layer))
    if not isinstance(target_receipt, dict):
        raise ValueError(f"score file has no teacher target for layer {args.layer}")
    teacher_path = Path(target_receipt["path"])
    if teacher_path.stat().st_size != int(target_receipt["bytes"]):
        raise ValueError("teacher target size changed")
    if sha256(teacher_path) != target_receipt["sha256"]:
        raise ValueError("teacher target hash changed")
    trace_lock_path = Path(scores["calibration"]["trace_lock"]["path"])
    if sha256(trace_lock_path) != scores["calibration"]["trace_lock"]["sha256"]:
        raise ValueError("trace lock hash changed")
    trace_lock = json.loads(trace_lock_path.read_text())
    hidden_receipt = trace_lock["files"][f"layer-{args.layer:03}.f32"]
    hidden_path = Path(trace_lock["trace_dir"]) / f"layer-{args.layer:03}.f32"
    if sha256(hidden_path) != hidden_receipt["sha256"]:
        raise ValueError("hidden-state trace hash changed")
    tokens = int(target_receipt["tokens"])
    hidden = np.memmap(hidden_path, dtype="<f4", mode="r", shape=(tokens, args.hidden_size))
    teacher = np.memmap(teacher_path, dtype="<f4", mode="r", shape=(tokens, args.hidden_size))
    if not np.isfinite(teacher).all():
        raise ValueError("teacher target contains non-finite values")

    index_path = args.source_dir / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    config = json.loads((args.source_dir / "config.json").read_text())
    device = torch.device(args.device)
    torch.manual_seed(args.seed + args.layer)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
    quant_assignments = None
    quant_bridge = None
    quant_importance = None
    importance_receipt = None
    if args.quantization_aware:
        from prepare_mixed_expert_repack import (
            _external_quantization_context,
            load_assignments,
        )
        _, quant_assignments, plan_pruned = load_assignments(args.plan)
        if plan_pruned[args.layer] != pruned:
            raise ValueError("quantization-aware plan pruning differs from healer pruning")
        quant_bridge, importance_receipts = _external_quantization_context(
            plan, quant_assignments, args
        )
        if quant_bridge is not None:
            importance_receipt = importance_receipts[str(args.layer)]
            quant_importance = load_importance_sidecar(
                importance_receipt,
                args.expert_count,
                args.hidden_size,
                args.intermediate_size,
            )
    model = load_student(
        args, active_ids, config, weight_map, device,
        quant_assignments=quant_assignments,
        quant_bridge=quant_bridge,
        quant_importance=quant_importance,
    ).to(device)

    score_rows = {
        int(row["expert"]): row
        for row in scores["scores"] if int(row["layer"]) == args.layer
    }
    target_load = torch.zeros(args.expert_count, dtype=torch.float32, device=device)
    for expert in active_ids:
        target_load[expert] = max(float(score_rows[expert]["router_weight_mass"]), 1e-8)
    target_load /= target_load.sum()

    all_indices = np.arange(tokens, dtype=np.int64)
    holdout = all_indices[all_indices % args.holdout_modulus == 0]
    train = all_indices[all_indices % args.holdout_modulus != 0]
    before = evaluate(model, hidden, teacher, holdout, args.eval_batch_tokens, device)
    parameters = [model.router]
    if args.mode == "joint":
        parameters.extend(
            parameter for name, parameter in model.named_parameters() if name != "router"
        )
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )
    generator = np.random.default_rng(args.seed + args.layer)
    history = []
    model.train()
    for step in range(1, args.steps + 1):
        batch_index = generator.choice(train, size=min(args.batch_tokens, len(train)), replace=False)
        x = torch.from_numpy(np.asarray(hidden[batch_index])).to(device)
        target = torch.from_numpy(np.asarray(teacher[batch_index])).to(device)
        output, selected, weights = model(x)
        target_energy = torch.square(target).mean().detach().clamp_min(1e-12)
        reconstruction_loss = torch.square(output - target).mean() / target_energy
        router_anchor = torch.square(model.router - model.initial_router).mean()
        router_anchor /= torch.square(model.initial_router).mean().detach().clamp_min(1e-12)
        loss = reconstruction_loss + args.router_anchor_weight * router_anchor
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
        optimizer.step()
        model.update_bias(
            selected, weights, target_load, args.bias_learning_rate, args.bias_max_delta
        )
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            record = {
                "step": step,
                "train_normalized_mse": float(reconstruction_loss.detach().cpu()),
                "router_anchor": float(router_anchor.detach().cpu()),
            }
            history.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
    model.eval()
    after_pre_requantization = evaluate(
        model, hidden, teacher, holdout, args.eval_batch_tokens, device
    )

    args.out_shard.parent.mkdir(parents=True, exist_ok=True)
    tensors = model.merged_source_tensors(args.layer)
    if quant_assignments is not None:
        # The artifact builder quantizes these merged tensors again.  Measure that exact terminal
        # state rather than reporting the optimistic full-precision adapter output.
        for local, expert in enumerate(active_ids):
            for projection, destination in (
                ("gate", model.gate_base), ("up", model.up_base), ("down", model.down_base),
            ):
                name = tensor_name(args.layer, expert, projection)
                qtype = quant_assignments[(args.layer, expert, projection)]
                importance = None
                if qtype in EXTERNAL_QTYPES:
                    assert quant_importance is not None
                    key = "input" if projection in ("gate", "up") else "down"
                    importance = quant_importance[key][expert]
                destination[local].copy_(
                    quantized_base(
                        tensors[name], qtype, importance=importance, bridge=quant_bridge
                    ).to(device)
                )
        model.mode = "router"
        after = evaluate(model, hidden, teacher, holdout, args.eval_batch_tokens, device)
    else:
        after = after_pre_requantization
    save_file(tensors, args.out_shard)
    receipt = {
        "format": CHECKPOINT_FORMAT,
        "layer": args.layer,
        "mode": args.mode,
        "source_dir": str(args.source_dir.resolve()),
        "source_config_sha256": sha256(args.source_dir / "config.json"),
        "source_index_sha256": sha256(index_path),
        "plan": {"path": str(args.plan.resolve()), "sha256": sha256(args.plan)},
        "scores": {"path": str(args.scores.resolve()), "sha256": sha256(args.scores)},
        "hidden_trace": {"path": str(hidden_path.resolve()), "sha256": hidden_receipt["sha256"]},
        "teacher_target": target_receipt,
        "active_experts": active_ids,
        "pruned_experts": sorted(pruned),
        "training": {
            "seed": args.seed,
            "steps": args.steps,
            "batch_tokens": args.batch_tokens,
            "learning_rate": args.learning_rate,
            "rank": args.rank if args.mode == "joint" else 0,
            "lora_alpha": args.lora_alpha if args.mode == "joint" else 0,
            "bias_learning_rate": args.bias_learning_rate,
            "bias_max_delta": args.bias_max_delta,
            "router_anchor_weight": args.router_anchor_weight,
            "holdout_modulus": args.holdout_modulus,
            "quantization_aware": args.quantization_aware,
            "quantization_base": (
                "exact GGUF quantizer bytes dequantized to BF16 before joint repair"
                if args.quantization_aware else "source checkpoint precision"
            ),
        },
        "before": before,
        "after_pre_requantization": after_pre_requantization,
        "after": after,
        "history": history,
        "output": {
            "path": str(args.out_shard.resolve()),
            "bytes": args.out_shard.stat().st_size,
            "sha256": sha256(args.out_shard),
            "tensor_count": len(tensors),
        },
        "public_eval_data_used_for_healing": False,
    }
    if quant_bridge is not None:
        receipt["external_quantizer"] = quant_bridge.provenance
        receipt["importance_sidecar"] = importance_receipt
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def self_test() -> None:
    probe = torch.linspace(-1, 1, 512, dtype=torch.float32).reshape(2, 256).to(torch.bfloat16)
    q2 = quantized_base(probe, "Q2_K")
    assert q2.dtype == torch.bfloat16 and q2.shape == probe.shape
    assert not torch.equal(q2, probe)
    with tempfile.TemporaryDirectory(prefix="bw24-hy3-layer-heal-") as tmp:
        root = Path(tmp)
        source, trace, targets = root / "source", root / "trace", root / "targets"
        source.mkdir(); trace.mkdir(); targets.mkdir()
        hidden = np.random.default_rng(3).normal(size=(20, 3)).astype("<f4")
        hidden_path = trace / "layer-001.f32"; hidden.tofile(hidden_path)
        teacher = np.zeros((20, 3), dtype="<f4")
        teacher[:, 0] = hidden[:, 0] * 0.5
        teacher_path = targets / "layer-001.teacher.f32"; teacher.tofile(teacher_path)
        tensors = {
            router_name(1): torch.ones(2, 3, dtype=torch.bfloat16) * 0.1,
            bias_name(1): torch.zeros(2),
        }
        weight_map = {}
        for expert in range(2):
            tensors[tensor_name(1, expert, "gate")] = torch.eye(2, 3, dtype=torch.bfloat16)
            tensors[tensor_name(1, expert, "up")] = torch.ones(2, 3, dtype=torch.bfloat16) * 0.2
            tensors[tensor_name(1, expert, "down")] = torch.ones(3, 2, dtype=torch.bfloat16) * 0.1
        for name in tensors:
            weight_map[name] = "model.safetensors"
        save_file(tensors, source / "model.safetensors")
        (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
        (source / "config.json").write_text(json.dumps({"router_scaling_factor": 1.0}))
        trace_lock = root / "trace-lock.json"
        trace_lock.write_text(json.dumps({
            "trace_dir": str(trace),
            "files": {hidden_path.name: {"sha256": sha256(hidden_path)}},
        }))
        scores_path = root / "scores.json"
        scores_path.write_text(json.dumps({
            "format": SCORE_FORMAT,
            "calibration": {
                "public_eval_data_used_for_selection": False,
                "trace_lock": {"path": str(trace_lock), "sha256": sha256(trace_lock)},
            },
            "teacher_targets": {"1": {
                "path": str(teacher_path), "sha256": sha256(teacher_path),
                "bytes": teacher_path.stat().st_size, "tokens": 20,
            }},
            "scores": [
                {"layer": 1, "expert": expert, "router_weight_mass": 1.0}
                for expert in range(2)
            ],
        }))
        plan_path = root / "plan.json"
        plan_path.write_text(json.dumps({
            "format": PLAN_FORMAT, "pruned_experts": {"1": [1]}
        }))
        args = argparse.Namespace(
            mode="joint", layer=1, plan=plan_path, scores=scores_path, source_dir=source,
            expert_count=2, top_k=1, hidden_size=3, intermediate_size=2, rank=2,
            lora_alpha=2.0, steps=3, batch_tokens=4, eval_batch_tokens=4,
            learning_rate=1e-3, bias_learning_rate=0.01, bias_max_delta=0.1,
            router_anchor_weight=1e-4,
            quantization_aware=False,
            max_grad_norm=1.0, holdout_modulus=5, log_every=1, seed=11, device="cpu",
            out_shard=root / "healed.safetensors", receipt=root / "receipt.json",
        )
        result = heal(args)
        assert result["output"]["tensor_count"] == 5
        assert result["active_experts"] == [0]
        assert math.isfinite(result["after"]["normalized_mse"])
        with safe_open(str(args.out_shard), framework="pt", device="cpu") as handle:
            assert handle.get_tensor(router_name(1)).dtype == torch.float32
            assert handle.get_tensor(bias_name(1)).dtype == torch.float32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("router", "joint"), required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--expert-count", type=int, default=192)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=1536)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-tokens", type=int, default=256)
    parser.add_argument("--eval-batch-tokens", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--bias-learning-rate", type=float, default=0.01)
    parser.add_argument("--bias-max-delta", type=float, default=0.1)
    parser.add_argument("--router-anchor-weight", type=float, default=1e-4)
    parser.add_argument(
        "--quantization-aware", action="store_true",
        help="initialize retained projections from their plan-assigned exact quantized bytes",
    )
    parser.add_argument("--ggml-lib", type=Path)
    parser.add_argument("--ggml-lib-sha256")
    parser.add_argument("--ggml-source-commit")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--holdout-modulus", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-shard", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        print("Hy3 pruned-layer healer self-test: PASS")
        return
    args = parse_args()
    receipt = heal(args)
    print(
        f"wrote {args.out_shard} sha256={receipt['output']['sha256']} "
        f"before={receipt['before']['normalized_mse']:.6g} "
        f"after={receipt['after']['normalized_mse']:.6g}"
    )


if __name__ == "__main__":
    main()
