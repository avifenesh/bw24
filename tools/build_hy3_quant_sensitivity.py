#!/usr/bin/env python3
"""Measure exact-format Hy3 expert quantization damage on private routed activations.

The scorer reuses the artifact builder's GGUF quantizers, then dequantizes their exact bytes and
measures output-space damage on a deterministic sample of tokens that actually routed to each
expert.  It reports joint expert damage plus gate/up/down projection ablations.  Layer subsets make
the full 79-layer sweep embarrassingly parallel without changing the frozen calibration corpus.
"""

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
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from build_hy3_reap_scores import load_requests, load_routes, parse_layers, tensor_name
from ggml_quant_bridge import (
    EXTERNAL_QTYPES,
    GgmlQuantBridge,
    sha256_file as bridge_sha256,
)
from prepare_mixed_expert_repack import (
    E2M1,
    QUANTIZERS,
    UE4M3,
    _dequant_nvfp4,
    _dequant_q2k,
    _dequant_q3k,
    _dequant_q8_0,
)


FORMAT = "bw24-hy3-quant-sensitivity-v1"
TRACE_LOCK_FORMAT = "bw24-moe-input-trace-lock-v1"
QTYPES = ("Q8_0", "NVFP4", "IQ4_XS", "Q4_K", "Q3_K", "Q2_K")
DEFAULT_QTYPES = ("Q8_0", "NVFP4", "Q3_K", "Q2_K")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dequant_q8(raw: bytes, rows: int, cols: int) -> np.ndarray:
    block = np.frombuffer(raw, dtype=np.uint8).reshape(rows, cols // 32, 34)
    scale = block[..., :2].copy().view("<f2").reshape(rows, cols // 32).astype(np.float32)
    values = block[..., 2:].view(np.int8).astype(np.float32)
    return (values * scale[..., None]).reshape(rows, cols)


def dequant_q2(raw: bytes, rows: int, cols: int) -> np.ndarray:
    block = np.frombuffer(raw, dtype=np.uint8).reshape(rows, cols // 256, 84)
    scale_code = block[..., :16] & 15
    min_code = block[..., :16] >> 4
    packed = block[..., 16:80].reshape(rows, cols // 256, 2, 32)
    values = np.empty((rows, cols // 256, 256), dtype=np.float32)
    for half in range(2):
        for lane in range(4):
            start = half * 128 + lane * 32
            values[..., start : start + 32] = (packed[..., half, :] >> (2 * lane)) & 3
    d = block[..., 80:82].copy().view("<f2").reshape(rows, cols // 256).astype(np.float32)
    dm = block[..., 82:84].copy().view("<f2").reshape(rows, cols // 256).astype(np.float32)
    scale = np.repeat(scale_code.astype(np.float32), 16, axis=-1)
    minimum = np.repeat(min_code.astype(np.float32), 16, axis=-1)
    return (d[..., None] * scale * values - dm[..., None] * minimum).reshape(rows, cols)


def dequant_q3(raw: bytes, rows: int, cols: int) -> np.ndarray:
    block = np.frombuffer(raw, dtype=np.uint8).reshape(rows, cols // 256, 110)
    encoded = np.empty((rows, cols // 256, 16), dtype=np.int16)
    for group in range(16):
        low = block[..., 96 + (group if group < 8 else group - 8)]
        low = (low & 15) if group < 8 else (low >> 4)
        high = (block[..., 104 + group % 4] >> (2 * (group // 4))) & 3
        encoded[..., group] = low | (high << 4)
    packed = block[..., 32:96].reshape(rows, cols // 256, 2, 32)
    high_mask = block[..., :32]
    values = np.empty((rows, cols // 256, 256), dtype=np.float32)
    for half in range(2):
        for lane in range(4):
            index = half * 128 + lane * 32
            low = (packed[..., half, :] >> (2 * lane)) & 3
            high = (high_mask >> (index // 32)) & 1
            values[..., index : index + 32] = low.astype(np.float32) - np.where(high, 0, 4)
    d = block[..., 108:110].copy().view("<f2").reshape(rows, cols // 256).astype(np.float32)
    scales = np.repeat((encoded - 32).astype(np.float32), 16, axis=-1)
    return (d[..., None] * scales * values).reshape(rows, cols)


def dequant_nvfp4(raw: bytes, rows: int, cols: int) -> np.ndarray:
    block = np.frombuffer(raw, dtype=np.uint8).reshape(rows, cols // 64, 36)
    scales = UE4M3[block[..., :4]]
    packed = block[..., 4:].reshape(rows, cols // 64, 4, 8)
    codes = np.empty((rows, cols // 64, 4, 16), dtype=np.uint8)
    codes[..., :8] = packed & 15
    codes[..., 8:] = packed >> 4
    signs = np.where(codes & 8, -1.0, 1.0).astype(np.float32)
    values = signs * E2M1[codes & 7] * scales[..., None]
    return values.reshape(rows, cols)


DEQUANTIZERS = {
    "Q8_0": dequant_q8,
    "NVFP4": dequant_nvfp4,
    "Q3_K": dequant_q3,
    "Q2_K": dequant_q2,
}


def quant_dequant(
    tensor: torch.Tensor,
    qtype: str,
    *,
    importance: np.ndarray | None = None,
    bridge: GgmlQuantBridge | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    source = tensor.float().cpu().numpy()
    if qtype in EXTERNAL_QTYPES:
        if bridge is None or importance is None:
            raise ValueError(f"{qtype} requires a pinned ggml bridge and private importance")
        raw, restored = bridge.quant_dequant(source, qtype, importance)
    else:
        raw = QUANTIZERS[qtype](source)
        restored = DEQUANTIZERS[qtype](raw, *source.shape)
    error = restored - source
    metrics = {
        "encoded_bytes": len(raw),
        "weight_normalized_mse": float(np.square(error, dtype=np.float64).sum())
        / max(float(np.square(source, dtype=np.float64).sum()), 1e-30),
        "weight_max_abs_error": float(np.abs(error).max(initial=0.0)),
    }
    return torch.from_numpy(restored).to(torch.bfloat16), metrics


def deterministic_sample(indices: np.ndarray, limit: int) -> np.ndarray:
    if len(indices) <= limit:
        return indices
    positions = np.linspace(0, len(indices) - 1, num=limit, dtype=np.int64)
    return indices[positions]


def error_metrics(
    candidate: torch.Tensor, reference: torch.Tensor, weight: torch.Tensor,
) -> dict[str, float]:
    delta = (candidate.float() - reference.float()) * weight[:, None]
    baseline = reference.float() * weight[:, None]
    squared_error = float(torch.square(delta).sum(dtype=torch.float64).cpu())
    baseline_energy = float(torch.square(baseline).sum(dtype=torch.float64).cpu())
    return {
        "squared_error": squared_error,
        "baseline_energy": baseline_energy,
        "normalized_mse": squared_error / max(baseline_energy, 1e-30),
    }


@torch.inference_mode()
def expert_metrics(
    *,
    x: torch.Tensor,
    weights: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    qtypes: tuple[str, ...],
    device: torch.device,
    bridge: GgmlQuantBridge | None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    x = x.to(device=device, dtype=torch.bfloat16)
    weights = weights.to(device=device, dtype=torch.float32)
    # Keep the source weights on CPU for exact quantization.  Separate device copies avoid a
    # needless device-to-host round trip for every expert and format.
    gate = gate.to(dtype=torch.bfloat16)
    up = up.to(dtype=torch.bfloat16)
    down = down.to(dtype=torch.bfloat16)
    gate_device = gate.to(device)
    up_device = up.to(device)
    down_device = down.to(device)
    gate_ref = F.linear(x, gate_device).float()
    up_ref = F.linear(x, up_device).float()
    activated_ref = F.silu(gate_ref) * up_ref
    output_ref = F.linear(activated_ref.to(torch.bfloat16), down_device).float()
    importance = {
        "input": np.maximum(
            torch.square(x.float()).mean(dim=0).cpu().numpy(), 1e-12
        ).astype(np.float32),
        "down": np.maximum(
            torch.square(activated_ref).mean(dim=0).cpu().numpy(), 1e-12
        ).astype(np.float32),
    }
    quantized = {
        qtype: {
            "gate": quant_dequant(
                gate, qtype, importance=importance["input"], bridge=bridge
            ),
            "up": quant_dequant(
                up, qtype, importance=importance["input"], bridge=bridge
            ),
            "down": quant_dequant(
                down, qtype, importance=importance["down"], bridge=bridge
            ),
        }
        for qtype in qtypes
    }
    result: dict[str, Any] = {}
    for qtype in qtypes:
        gate_q, gate_weight = quantized[qtype]["gate"]
        up_q, up_weight = quantized[qtype]["up"]
        down_q, down_weight = quantized[qtype]["down"]
        gate_q = gate_q.to(device); up_q = up_q.to(device); down_q = down_q.to(device)
        gate_out = F.linear(x, gate_q).float()
        up_out = F.linear(x, up_q).float()
        activated_q = F.silu(gate_out) * up_out
        gate_only = F.linear(
            (F.silu(gate_out) * up_ref).to(torch.bfloat16), down_device
        ).float()
        up_only = F.linear(
            (F.silu(gate_ref) * up_out).to(torch.bfloat16), down_device
        ).float()
        down_only = F.linear(activated_ref.to(torch.bfloat16), down_q).float()
        joint = F.linear(activated_q.to(torch.bfloat16), down_q).float()
        result[qtype] = {
            "joint_output_error": error_metrics(joint, output_ref, weights),
            "projection_output_error": {
                "gate": error_metrics(gate_only, output_ref, weights),
                "up": error_metrics(up_only, output_ref, weights),
                "down": error_metrics(down_only, output_ref, weights),
            },
            "projection_weight_error": {
                "gate": gate_weight,
                "up": up_weight,
                "down": down_weight,
            },
        }
        del gate_q, up_q, down_q, gate_out, up_out, activated_q, gate_only, up_only, down_only, joint
    del quantized
    return result, importance


def score(args: argparse.Namespace) -> dict[str, Any]:
    qtypes = tuple(x for x in args.qtypes.split(",") if x)
    if not qtypes or any(x not in QTYPES for x in qtypes) or len(set(qtypes)) != len(qtypes):
        raise ValueError(f"qtypes must be distinct values drawn from {QTYPES}")
    external = sorted(set(qtypes) & set(EXTERNAL_QTYPES))
    bridge = None
    if external:
        if not all((args.ggml_lib, args.ggml_lib_sha256, args.ggml_source_commit)):
            raise ValueError(f"{external} require --ggml-lib, its SHA-256, and source commit")
        if args.importance_dir is None:
            raise ValueError(f"{external} require --importance-dir")
        bridge = GgmlQuantBridge(
            args.ggml_lib, args.ggml_lib_sha256, args.ggml_source_commit
        )
    if args.importance_dir is not None:
        args.importance_dir.mkdir(parents=True, exist_ok=True)
    trace_lock = json.loads(args.trace_lock.read_text())
    if trace_lock.get("format") != TRACE_LOCK_FORMAT:
        raise ValueError(f"trace lock format must be {TRACE_LOCK_FORMAT}")
    if trace_lock.get("public_eval_data_used_for_selection") is not False:
        raise ValueError("trace lock must attest that public eval data was not used")
    requests, token_strata, strata = load_requests(args.requests)
    if sha256(args.requests) != trace_lock["requests"]["sha256"]:
        raise ValueError("request corpus hash does not match trace lock")
    all_layers = [int(x) for x in trace_lock["layers"]]
    layers = parse_layers(args.layers)
    selected, route_weights = load_routes(
        args.weight_trace, all_layers, layers, len(token_strata), args.expert_count, args.top_k
    )
    index_path = args.source_dir / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
    rows: list[dict[str, Any]] = []
    importance_sidecars: dict[str, dict[str, Any]] = {}
    for layer in layers:
        hidden_path = Path(trace_lock["trace_dir"]) / f"layer-{layer:03}.f32"
        receipt = trace_lock["files"][hidden_path.name]
        if sha256(hidden_path) != receipt["sha256"]:
            raise ValueError(f"layer {layer} hidden trace hash mismatch")
        hidden = np.memmap(
            hidden_path, dtype="<f4", mode="r", shape=(len(token_strata), args.hidden_size)
        )
        input_importance = np.empty(
            (args.expert_count, args.hidden_size), dtype=np.float32
        )
        down_importance = np.empty(
            (args.expert_count, args.intermediate_size), dtype=np.float32
        )
        names = [
            tensor_name(layer, expert, projection)
            for expert in range(args.expert_count) for projection in ("gate", "up", "down")
        ]
        shards = sorted({weight_map[name] for name in names})
        with contextlib.ExitStack() as stack:
            handles = {
                shard: stack.enter_context(
                    safe_open(str(args.source_dir / shard), framework="pt", device="cpu")
                ) for shard in shards
            }
            for expert in range(args.expert_count):
                token_index, slot = np.nonzero(selected[layer] == expert)
                sampled = deterministic_sample(np.arange(len(token_index)), args.max_tokens_per_expert)
                chosen_tokens = token_index[sampled]
                chosen_slots = slot[sampled]
                observed_routing = bool(len(chosen_tokens))
                if not observed_routing:
                    # Preserve function-space quantization evidence for experts absent from
                    # the frozen top-k trace, but give it zero routed-mass in allocation.
                    chosen_tokens = deterministic_sample(
                        np.arange(len(token_strata)), args.max_tokens_per_expert
                    )
                x = torch.from_numpy(np.asarray(hidden[chosen_tokens], dtype=np.float32))
                weights = (
                    torch.from_numpy(
                        route_weights[layer][chosen_tokens, chosen_slots].astype(np.float32)
                    )
                    if observed_routing else torch.ones(len(chosen_tokens), dtype=torch.float32)
                )
                tensors = {
                    projection: handles[weight_map[tensor_name(layer, expert, projection)]].get_tensor(
                        tensor_name(layer, expert, projection)
                    ) for projection in ("gate", "up", "down")
                }
                metrics, importance = expert_metrics(
                    x=x, weights=weights, gate=tensors["gate"], up=tensors["up"],
                    down=tensors["down"], qtypes=qtypes, device=device, bridge=bridge,
                )
                input_importance[expert] = importance["input"]
                down_importance[expert] = importance["down"]
                rows.append({
                    "layer": layer,
                    "expert": expert,
                    "routed_tokens": int(len(token_index)),
                    "sampled_tokens": int(len(chosen_tokens)),
                    "sample_scale": (
                        float(len(token_index) / len(chosen_tokens)) if observed_routing else 0.0
                    ),
                    "measurement_source": (
                        "routed_top_k" if observed_routing else "unrouted_function_probe"
                    ),
                    "sampled_router_weight_mass": float(weights.sum(dtype=torch.float64)),
                    "quantization": metrics,
                })
                if args.progress_every and (expert + 1) % args.progress_every == 0:
                    print(f"layer={layer} experts={expert + 1}/{args.expert_count}", flush=True)
                del x, weights, tensors
        del hidden
        if args.importance_dir is not None:
            sidecar = args.importance_dir / f"layer-{layer:03}.npz"
            temporary = sidecar.with_name(sidecar.name + ".tmp")
            with temporary.open("wb") as handle:
                np.savez(handle, input=input_importance, down=down_importance)
            temporary.replace(sidecar)
            importance_sidecars[str(layer)] = {
                "path": str(sidecar.resolve()),
                "bytes": sidecar.stat().st_size,
                "sha256": bridge_sha256(sidecar),
                "input_shape": list(input_importance.shape),
                "down_shape": list(down_importance.shape),
            }
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {
        "format": FORMAT,
        "model": {
            "expert_count": args.expert_count,
            "top_k": args.top_k,
            "hidden_size": args.hidden_size,
            "intermediate_size": args.intermediate_size,
            "moe_layers": layers,
            "complete_moe_layers": all_layers,
        },
        "measurement": {
            "qtypes": list(qtypes),
            "max_tokens_per_expert": args.max_tokens_per_expert,
            "sampling": (
                "deterministic evenly spaced routed-token positions; experts absent from the "
                "frozen top-k trace use a deterministic function probe with zero routed mass"
            ),
            "metric": "router-weighted expert-output normalized MSE",
            "projection_ablation": "one quantized projection with the other two at source precision",
            "exact_quantizer_implementation": {
                qtype: (
                    bridge.provenance if qtype in EXTERNAL_QTYPES
                    else {"implementation": "prepare_mixed_expert_repack.py"}
                )
                for qtype in qtypes
            },
            "importance_metric": "per-column mean squared private calibration activation",
        },
        "calibration": {
            "requests": {"path": str(args.requests.resolve()), "sha256": sha256(args.requests)},
            "trace_lock": {"path": str(args.trace_lock.resolve()), "sha256": sha256(args.trace_lock)},
            "weighted_routes": {"path": str(args.weight_trace.resolve()), "sha256": sha256(args.weight_trace)},
            "strata": strata,
            "public_eval_data_used_for_selection": False,
        },
        "source": {
            "directory": str(args.source_dir.resolve()),
            "index_sha256": sha256(index_path),
            "config_sha256": sha256(args.source_dir / "config.json"),
        },
        "importance_sidecars": importance_sidecars,
        "scores": rows,
    }


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="bw24-hy3-quant-sensitivity-") as tmp:
        root = Path(tmp); source = root / "source"; trace = root / "trace"
        source.mkdir(); trace.mkdir()
        rng = np.random.default_rng(7)
        hidden = rng.normal(size=(8, 256)).astype("<f4")
        hidden_path = trace / "layer-001.f32"; hidden.tofile(hidden_path)
        requests = root / "requests.jsonl"
        requests.write_text(json.dumps({
            "ordinal": 0, "stratum": "code", "prompt_tokens": 8,
            "prompt_ids": list(range(8)),
        }) + "\n")
        routes = root / "routes.trace"
        routes.write_text("\n".join(f"1 1 {i % 2}:1.0" for i in range(8)) + "\n")
        lock = root / "trace-lock.json"
        lock.write_text(json.dumps({
            "format": TRACE_LOCK_FORMAT, "trace_dir": str(trace), "layers": [1],
            "hidden_size": 256, "requests": {"sha256": sha256(requests)},
            "files": {hidden_path.name: {"sha256": sha256(hidden_path)}},
            "public_eval_data_used_for_selection": False,
        }))
        tensors: dict[str, torch.Tensor] = {}; weight_map = {}
        for expert in range(3):
            for projection, shape in {
                "gate": (256, 256), "up": (256, 256), "down": (256, 256),
            }.items():
                name = tensor_name(1, expert, projection)
                tensors[name] = torch.from_numpy(rng.normal(0, 0.02, size=shape)).to(torch.bfloat16)
                weight_map[name] = "model.safetensors"
        save_file(tensors, source / "model.safetensors")
        (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
        (source / "config.json").write_text("{}")
        args = argparse.Namespace(
            trace_lock=lock, weight_trace=routes, requests=requests, source_dir=source,
            layers="1", expert_count=3, top_k=1, hidden_size=256, intermediate_size=256,
            max_tokens_per_expert=2, qtypes="Q8_0,NVFP4,Q3_K,Q2_K", device="cpu",
            progress_every=0, importance_dir=root / "importance",
            ggml_lib=None, ggml_lib_sha256=None, ggml_source_commit=None,
        )
        result = score(args)
        assert len(result["scores"]) == 3
        for row in result["scores"]:
            for qtype in DEFAULT_QTYPES:
                value = row["quantization"][qtype]["joint_output_error"]["normalized_mse"]
                assert math.isfinite(value) and value >= 0
        assert result["scores"][0]["quantization"]["Q8_0"]["joint_output_error"]["normalized_mse"] \
            < result["scores"][0]["quantization"]["Q2_K"]["joint_output_error"]["normalized_mse"]
        assert result["scores"][2]["measurement_source"] == "unrouted_function_probe"
        assert result["scores"][2]["routed_tokens"] == 0
        assert result["scores"][2]["sample_scale"] == 0.0
        assert set(result["importance_sidecars"]) == {"1"}
        sidecar = result["importance_sidecars"]["1"]
        assert bridge_sha256(Path(sidecar["path"])) == sidecar["sha256"]
        probe = rng.normal(size=(3, 256)).astype(np.float32)
        scalar = {
            "Q8_0": _dequant_q8_0,
            "NVFP4": _dequant_nvfp4,
            "Q3_K": _dequant_q3k,
            "Q2_K": _dequant_q2k,
        }
        for qtype in DEFAULT_QTYPES:
            raw = QUANTIZERS[qtype](probe)
            expected = scalar[qtype](raw, probe.shape[1])
            actual = DEQUANTIZERS[qtype](raw, *probe.shape)
            assert np.array_equal(actual, expected), qtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-lock", type=Path, required=True)
    parser.add_argument("--weight-trace", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--layers", default="1-79")
    parser.add_argument("--expert-count", type=int, default=192)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=1536)
    parser.add_argument("--max-tokens-per-expert", type=int, default=16)
    parser.add_argument("--qtypes", default=",".join(DEFAULT_QTYPES))
    parser.add_argument("--importance-dir", type=Path)
    parser.add_argument("--ggml-lib", type=Path)
    parser.add_argument("--ggml-lib-sha256")
    parser.add_argument("--ggml-source-commit")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--progress-every", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    if sys.argv[1:] == ["--self-test"]:
        self_test(); print("Hy3 quant sensitivity self-test: PASS"); return
    args = parse_args(); result = score(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out} sha256={sha256(args.out)} rows={len(result['scores'])}")


if __name__ == "__main__":
    main()
