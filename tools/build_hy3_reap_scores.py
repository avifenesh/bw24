#!/usr/bin/env python3
"""Build frozen Hy3 REAP, traffic, diversity, and rare-stratum retention scores.

Inputs are a validated tokenwise BW24 MoE-input trace, the matched runtime weighted-route trace,
and the pinned high-precision Hugging Face checkpoint. Public evaluation data must not be present.
The tool may process a layer subset so independent GPU workers can run in parallel.
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


FORMAT = "bw24-expert-retention-scores-v1"
TRACE_LOCK_FORMAT = "bw24-moe-input-trace-lock-v1"
REAP_REFERENCE_COMMIT = "1970473c51ca3caeb98c10392f15b3a08a672974"


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


def load_requests(path: Path) -> tuple[list[dict[str, Any]], np.ndarray, list[str]]:
    requests = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not requests:
        raise ValueError("request corpus is empty")
    strata = sorted({str(row["stratum"]) for row in requests})
    stratum_id = {name: index for index, name in enumerate(strata)}
    token_strata: list[int] = []
    seen: set[str] = set()
    for row in requests:
        trace_id = str(row["ordinal"])
        if trace_id in seen:
            raise ValueError(f"duplicate request ordinal {trace_id}")
        seen.add(trace_id)
        tokens = int(row["prompt_tokens"])
        if tokens <= 0 or tokens != len(row["prompt_ids"]):
            raise ValueError(f"request {trace_id} has inconsistent prompt token coverage")
        token_strata.extend([stratum_id[str(row["stratum"])]] * tokens)
    return requests, np.asarray(token_strata, dtype=np.int16), strata


def load_routes(
    path: Path,
    all_layers: list[int],
    selected_layers: list[int],
    total_tokens: int,
    expert_count: int,
    top_k: int,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    selected_set = set(selected_layers)
    expert_ids = {
        layer: np.empty((total_tokens, top_k), dtype=np.int16) for layer in selected_layers
    }
    weights = {
        layer: np.empty((total_tokens, top_k), dtype=np.float32) for layer in selected_layers
    }
    expected_rows = total_tokens * len(all_layers)
    row_count = 0
    with path.open() as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if row_count >= expected_rows:
                raise ValueError(f"{path}:{line_no}: extra weighted-route row")
            token = row_count // len(all_layers)
            expected_layer = all_layers[row_count % len(all_layers)]
            fields = line.split(maxsplit=2)
            if len(fields) != 3:
                raise ValueError(f"{path}:{line_no}: malformed weighted-route row")
            layer, row_tokens = int(fields[0]), int(fields[1])
            if layer != expected_layer or row_tokens != 1:
                raise ValueError(
                    f"{path}:{line_no}: got layer/tokens={layer}/{row_tokens}, expected "
                    f"{expected_layer}/1"
                )
            pairs = []
            for raw in fields[2].split(","):
                expert_s, weight_s = raw.split(":", 1)
                expert, weight = int(expert_s), float(weight_s)
                if expert < 0 or expert >= expert_count or not math.isfinite(weight) or weight < 0:
                    raise ValueError(f"{path}:{line_no}: invalid expert/weight {raw!r}")
                pairs.append((expert, weight))
            if len(pairs) != top_k or len({expert for expert, _ in pairs}) != top_k:
                raise ValueError(f"{path}:{line_no}: expected {top_k} distinct experts")
            if layer in selected_set:
                expert_ids[layer][token] = [expert for expert, _ in pairs]
                weights[layer][token] = [weight for _, weight in pairs]
            row_count += 1
    if row_count != expected_rows:
        raise ValueError(f"weighted-route rows={row_count}, expected {expected_rows}")
    return expert_ids, weights


def scaled(values: np.ndarray) -> np.ndarray:
    values = np.maximum(values.astype(np.float64), 0.0)
    maximum = float(values.max(initial=0.0))
    if maximum == 0:
        return np.zeros_like(values)
    return np.log1p(values) / math.log1p(maximum)


def tensor_name(layer: int, expert: int, projection: str) -> str:
    return f"model.layers.{layer}.mlp.experts.{expert}.{projection}_proj.weight"


def score_layer(
    *,
    layer: int,
    hidden_path: Path,
    selected: np.ndarray,
    route_weights: np.ndarray,
    token_strata: np.ndarray,
    strata: list[str],
    source_dir: Path,
    weight_map: dict[str, str],
    expert_count: int,
    hidden_size: int,
    intermediate_size: int,
    batch_tokens: int,
    sketch_dim: int,
    seed: int,
    reap_weight: float,
    traffic_weight: float,
    diversity_weight: float,
    rare_weight: float,
    protect_per_stratum: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    total_tokens = selected.shape[0]
    hidden = np.memmap(hidden_path, dtype="<f4", mode="r", shape=(total_tokens, hidden_size))
    frequency = np.zeros(expert_count, dtype=np.int64)
    reap_sum = np.zeros(expert_count, dtype=np.float64)
    norm_sum = np.zeros(expert_count, dtype=np.float64)
    traffic = np.zeros(expert_count, dtype=np.float64)
    stratum_mass = np.zeros((expert_count, len(strata)), dtype=np.float64)
    signatures = np.zeros((expert_count, sketch_dim), dtype=np.float64)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + layer)
    projection = torch.randn(hidden_size, sketch_dim, generator=generator, dtype=torch.float32)
    projection = (projection / math.sqrt(sketch_dim)).to(device)

    names = [
        tensor_name(layer, expert, projection_name)
        for expert in range(expert_count)
        for projection_name in ("gate", "up", "down")
    ]
    missing = [name for name in names if name not in weight_map]
    if missing:
        raise ValueError(f"source index is missing {len(missing)} layer {layer} expert tensors")
    shards = sorted({weight_map[name] for name in names})
    with contextlib.ExitStack() as stack:
        handles = {
            shard: stack.enter_context(
                safe_open(str(source_dir / shard), framework="pt", device="cpu")
            )
            for shard in shards
        }
        with torch.inference_mode():
            for expert in range(expert_count):
                token_index, slot = np.nonzero(selected == expert)
                frequency[expert] = len(token_index)
                if len(token_index) == 0:
                    continue
                weights_np = route_weights[token_index, slot].astype(np.float32)
                traffic[expert] = weights_np.sum(dtype=np.float64)
                np.add.at(stratum_mass[expert], token_strata[token_index], weights_np)

                gate_name = tensor_name(layer, expert, "gate")
                up_name = tensor_name(layer, expert, "up")
                down_name = tensor_name(layer, expert, "down")
                gate = handles[weight_map[gate_name]].get_tensor(gate_name).to(
                    device=device, dtype=torch.bfloat16
                )
                up = handles[weight_map[up_name]].get_tensor(up_name).to(
                    device=device, dtype=torch.bfloat16
                )
                down = handles[weight_map[down_name]].get_tensor(down_name).to(
                    device=device, dtype=torch.bfloat16
                )
                if tuple(gate.shape) != (intermediate_size, hidden_size):
                    raise ValueError(f"{gate_name}: unexpected shape {tuple(gate.shape)}")
                if tuple(up.shape) != (intermediate_size, hidden_size):
                    raise ValueError(f"{up_name}: unexpected shape {tuple(up.shape)}")
                if tuple(down.shape) != (hidden_size, intermediate_size):
                    raise ValueError(f"{down_name}: unexpected shape {tuple(down.shape)}")

                signature_sum = torch.zeros(sketch_dim, dtype=torch.float64, device=device)
                for start in range(0, len(token_index), batch_tokens):
                    stop = min(start + batch_tokens, len(token_index))
                    x_np = np.asarray(hidden[token_index[start:stop]], dtype=np.float32)
                    x = torch.from_numpy(x_np).to(device=device, dtype=torch.bfloat16)
                    output = F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)
                    norms = torch.linalg.vector_norm(output.float(), dim=-1)
                    batch_weights = torch.from_numpy(weights_np[start:stop]).to(device)
                    reap_sum[expert] += float((norms * batch_weights).sum(dtype=torch.float64).cpu())
                    norm_sum[expert] += float(norms.sum(dtype=torch.float64).cpu())
                    sketch = F.normalize(output.float() @ projection, dim=-1)
                    signature_sum += (sketch * batch_weights[:, None]).sum(
                        dim=0, dtype=torch.float64
                    )
                    del x, output, norms, batch_weights, sketch
                signature = signature_sum.cpu().numpy()
                signature_norm = np.linalg.norm(signature)
                if signature_norm > 0:
                    signatures[expert] = signature / signature_norm
                del gate, up, down, signature_sum
    del hidden, projection
    torch.cuda.empty_cache()

    reap = np.divide(reap_sum, frequency, out=np.zeros_like(reap_sum), where=frequency > 0)
    mean_norm = np.divide(norm_sum, frequency, out=np.zeros_like(norm_sum), where=frequency > 0)
    similarity = signatures @ signatures.T
    valid_signature = np.linalg.norm(signatures, axis=1) > 0
    uniqueness = np.zeros(expert_count, dtype=np.float64)
    for expert in range(expert_count):
        candidates = similarity[expert, valid_signature]
        if not valid_signature[expert] or len(candidates) <= 1:
            continue
        candidates = np.sort(candidates[candidates < 1.0 - 1e-7])
        if len(candidates):
            uniqueness[expert] = max(0.0, 1.0 - float(candidates[-min(3, len(candidates)) :].mean()))

    stratum_relative = np.zeros_like(stratum_mass)
    for stratum in range(len(strata)):
        maximum = stratum_mass[:, stratum].max(initial=0.0)
        if maximum > 0:
            stratum_relative[:, stratum] = stratum_mass[:, stratum] / maximum
    rare = stratum_relative.max(axis=1, initial=0.0)
    components = {
        "reap": scaled(reap),
        "traffic": scaled(traffic),
        "diversity": scaled(uniqueness),
        "rare_stratum": scaled(rare),
    }
    composite = (
        reap_weight * components["reap"]
        + traffic_weight * components["traffic"]
        + diversity_weight * components["diversity"]
        + rare_weight * components["rare_stratum"]
    )
    protected: set[int] = set()
    if protect_per_stratum:
        for stratum in range(len(strata)):
            ranked = np.argsort(-stratum_mass[:, stratum], kind="stable")
            protected.update(
                int(expert) for expert in ranked[:protect_per_stratum]
                if stratum_mass[expert, stratum] > 0
            )

    rows = []
    for expert in range(expert_count):
        tie_break = (expert_count - expert) * 1e-12
        rows.append({
            "layer": layer,
            "expert": expert,
            "retain_score": float(composite[expert] + tie_break),
            "protected": expert in protected,
            "frequency": int(frequency[expert]),
            "reap": float(reap[expert]),
            "mean_output_norm": float(mean_norm[expert]),
            "router_weight_mass": float(traffic[expert]),
            "diversity_uniqueness": float(uniqueness[expert]),
            "rare_stratum_score": float(rare[expert]),
            "normalized_components": {
                name: float(values[expert]) for name, values in components.items()
            },
            "stratum_router_mass": {
                name: float(stratum_mass[expert, index]) for index, name in enumerate(strata)
            },
        })
    return rows


def build_scores(args: argparse.Namespace) -> dict[str, Any]:
    weights = (args.reap_weight, args.traffic_weight, args.diversity_weight, args.rare_weight)
    if any(value < 0 for value in weights) or not math.isclose(sum(weights), 1.0):
        raise ValueError("component weights must be non-negative and sum to one")
    trace_lock = json.loads(args.trace_lock.read_text())
    if trace_lock.get("format") != TRACE_LOCK_FORMAT:
        raise ValueError(f"trace lock format must be {TRACE_LOCK_FORMAT!r}")
    if trace_lock.get("public_eval_data_used_for_selection") is not False:
        raise ValueError("trace lock must attest that public eval data was not used")
    requests, token_strata, strata = load_requests(args.requests)
    if sha256(args.requests) != trace_lock["requests"]["sha256"]:
        raise ValueError("request corpus hash does not match trace lock")
    all_layers = [int(layer) for layer in trace_lock["layers"]]
    layers = parse_layers(args.layers)
    if not set(layers).issubset(all_layers):
        raise ValueError("requested layer subset is outside the trace lock")
    if int(trace_lock["hidden_size"]) != args.hidden_size:
        raise ValueError("hidden size does not match trace lock")
    expert_ids, route_weights = load_routes(
        args.weight_trace, all_layers, layers, len(token_strata), args.expert_count, args.top_k
    )

    index_path = args.source_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("source checkpoint index has no weight_map")
    config = json.loads((args.source_dir / "config.json").read_text())
    expected_config = {
        "hidden_size": args.hidden_size,
        "moe_intermediate_size": args.intermediate_size,
        "num_experts": args.expert_count,
        "num_experts_per_tok": args.top_k,
    }
    for key, expected in expected_config.items():
        if int(config[key]) != expected:
            raise ValueError(f"source config {key}={config[key]}, expected {expected}")
    if config.get("hidden_act") not in (None, "silu"):
        raise ValueError(f"unsupported Hy3 expert activation {config.get('hidden_act')!r}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA scoring was requested but torch.cuda.is_available() is false")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False

    rows: list[dict[str, Any]] = []
    for position, layer in enumerate(layers, 1):
        file_name = f"layer-{layer:03}.f32"
        hidden_path = Path(trace_lock["trace_dir"]) / file_name
        file_lock = trace_lock["files"].get(file_name)
        if not isinstance(file_lock, dict) or sha256(hidden_path) != file_lock["sha256"]:
            raise ValueError(f"hidden-state payload hash mismatch for layer {layer}")
        print(f"[{position}/{len(layers)}] scoring Hy3 layer {layer} on {device}", flush=True)
        rows.extend(score_layer(
            layer=layer,
            hidden_path=hidden_path,
            selected=expert_ids[layer],
            route_weights=route_weights[layer],
            token_strata=token_strata,
            strata=strata,
            source_dir=args.source_dir,
            weight_map=weight_map,
            expert_count=args.expert_count,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            batch_tokens=args.batch_tokens,
            sketch_dim=args.sketch_dim,
            seed=args.seed,
            reap_weight=args.reap_weight,
            traffic_weight=args.traffic_weight,
            diversity_weight=args.diversity_weight,
            rare_weight=args.rare_weight,
            protect_per_stratum=args.protect_per_stratum,
            device=device,
        ))
    return {
        "format": FORMAT,
        "rank_metric": "hy3_reap_traffic_activation_diversity_rare_stratum_v1",
        "model": {
            "expert_count": args.expert_count,
            "top_k": args.top_k,
            "hidden_size": args.hidden_size,
            "intermediate_size": args.intermediate_size,
            "moe_layers": layers,
            "complete_moe_layers": all_layers,
        },
        "policy": {
            "component_weights": {
                "reap": args.reap_weight,
                "traffic": args.traffic_weight,
                "diversity": args.diversity_weight,
                "rare_stratum": args.rare_weight,
            },
            "protect_per_stratum_per_layer": args.protect_per_stratum,
            "sketch_dim": args.sketch_dim,
            "seed": args.seed,
            "reap_formula": "mean(runtime_combine_weight * l2(expert_output)) over routed tokens",
            "reap_reference_commit": REAP_REFERENCE_COMMIT,
            "hy3_router_contract": "runtime sigmoid+bias top-k, selected-weight renorm and router scaling",
        },
        "calibration": {
            "requests": {"path": str(args.requests.resolve()), "sha256": sha256(args.requests)},
            "trace_lock": {"path": str(args.trace_lock.resolve()), "sha256": sha256(args.trace_lock)},
            "weighted_routes": {
                "path": str(args.weight_trace.resolve()), "sha256": sha256(args.weight_trace),
            },
            "requests_count": len(requests),
            "prompt_tokens": len(token_strata),
            "strata": strata,
            "public_eval_data_used_for_selection": False,
        },
        "source": {
            "directory": str(args.source_dir.resolve()),
            "config_sha256": sha256(args.source_dir / "config.json"),
            "index_sha256": sha256(index_path),
        },
        "scores": rows,
    }


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="bw24-hy3-reap-scores-") as tmp:
        root = Path(tmp)
        source, trace = root / "source", root / "trace"
        source.mkdir(); trace.mkdir()
        hidden = np.asarray([[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1]], dtype="<f4")
        hidden_path = trace / "layer-001.f32"; hidden.tofile(hidden_path)
        requests = root / "requests.jsonl"
        requests.write_text(json.dumps({
            "ordinal": 0, "stratum": "code", "prompt_tokens": 4, "prompt_ids": [1, 2, 3, 4]
        }) + "\n")
        routes = root / "routes.trace"
        routes.write_text("\n".join(["1 1 0:1.0", "1 1 1:1.0", "1 1 0:1.0", "1 1 1:1.0"]) + "\n")
        trace_lock = root / "trace-lock.json"
        trace_lock.write_text(json.dumps({
            "format": TRACE_LOCK_FORMAT,
            "trace_dir": str(trace),
            "requests": {"sha256": sha256(requests)},
            "layers": [1],
            "hidden_size": 3,
            "files": {hidden_path.name: {"sha256": sha256(hidden_path)}},
            "public_eval_data_used_for_selection": False,
        }))
        tensors = {}
        weight_map = {}
        for expert in range(2):
            tensors[tensor_name(1, expert, "gate")] = torch.eye(2, 3, dtype=torch.bfloat16)
            tensors[tensor_name(1, expert, "up")] = torch.ones(2, 3, dtype=torch.bfloat16)
            tensors[tensor_name(1, expert, "down")] = torch.ones(3, 2, dtype=torch.bfloat16)
            for projection in ("gate", "up", "down"):
                weight_map[tensor_name(1, expert, projection)] = "model.safetensors"
        save_file(tensors, source / "model.safetensors")
        (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
        (source / "config.json").write_text(json.dumps({
            "hidden_size": 3,
            "moe_intermediate_size": 2,
            "num_experts": 2,
            "num_experts_per_tok": 1,
            "hidden_act": "silu",
        }))
        args = argparse.Namespace(
            trace_lock=trace_lock, weight_trace=routes, requests=requests, source_dir=source,
            layers="1", expert_count=2, top_k=1, hidden_size=3, intermediate_size=2,
            batch_tokens=2, sketch_dim=2, seed=7, reap_weight=0.65, traffic_weight=0.10,
            diversity_weight=0.15, rare_weight=0.10, protect_per_stratum=1, device="cpu",
        )
        result = build_scores(args)
        assert len(result["scores"]) == 2
        assert sum(row["frequency"] for row in result["scores"]) == 4
        assert sum(bool(row["protected"]) for row in result["scores"]) == 1
        assert all(math.isfinite(row["retain_score"]) for row in result["scores"])


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
    parser.add_argument("--batch-tokens", type=int, default=256)
    parser.add_argument("--sketch-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--reap-weight", type=float, default=0.65)
    parser.add_argument("--traffic-weight", type=float, default=0.10)
    parser.add_argument("--diversity-weight", type=float, default=0.15)
    parser.add_argument("--rare-weight", type=float, default=0.10)
    parser.add_argument("--protect-per-stratum", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        print("Hy3 REAP score builder self-test: PASS")
        return
    args = parse_args()
    result = build_scores(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out} sha256={sha256(args.out)} scores={len(result['scores'])}")


if __name__ == "__main__":
    main()
