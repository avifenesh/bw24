#!/usr/bin/env python3
"""Selective official-Transformers-equivalent Hy3 layer-0 reference.

Only the requested embedding rows and layer-0 tensors are materialized from the sharded BF16
checkpoint.  No Transformers model object (and therefore no full-model load) is used.

Semantics are pinned to Hugging Face Transformers commit
d610229d0f0d80c7927694f164e3dd362750ca19, `models/hy_v3/modeling_hy_v3.py`:
RMSNorm in f32, Q/K per-head RMSNorm, full-head default RoPE, causal attention with f32 softmax,
and dense SwiGLU in layer 0.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from typing import Any, TextIO

import torch
import torch.nn.functional as F
from safetensors import safe_open


TRANSFORMERS_COMMIT = "d610229d0f0d80c7927694f164e3dd362750ca19"
SCHEMA = "bw24.hy3.layer0.v2"


class TensorStore:
    """Keep only the safetensors shards needed by requested keys open."""

    def __init__(self, checkpoint: Path, stack: ExitStack) -> None:
        index_path = checkpoint / "model.safetensors.index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"missing sharded checkpoint index: {index_path}")
        index = json.loads(index_path.read_text())
        self._checkpoint = checkpoint
        self._weight_map: dict[str, str] = index["weight_map"]
        self._stack = stack
        self._handles: dict[str, Any] = {}

    def _handle(self, key: str) -> Any:
        try:
            shard = self._weight_map[key]
        except KeyError as exc:
            raise KeyError(f"checkpoint index has no tensor {key!r}") from exc
        if shard not in self._handles:
            self._handles[shard] = self._stack.enter_context(
                safe_open(self._checkpoint / shard, framework="pt", device="cpu")
            )
        return self._handles[shard]

    def tensor(self, key: str, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return self._handle(key).get_tensor(key).to(device=device, dtype=dtype)

    def rows(
        self,
        key: str,
        row_ids: list[int],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        sliced = self._handle(key).get_slice(key)
        rows = [sliced[row : row + 1] for row in row_ids]
        return torch.cat(rows, dim=0).to(device=device, dtype=dtype)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Match HYV3RMSNorm: f32 variance, then cast normalized values back before weight."""
    input_dtype = x.dtype
    x_f32 = x.float()
    normalized = x_f32 * torch.rsqrt(x_f32.square().mean(dim=-1, keepdim=True) + eps)
    return weight * normalized.to(input_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def config_value(config: dict[str, Any], key: str, default: Any = None) -> Any:
    value = config.get(key, default)
    if value is None:
        raise ValueError(f"config is missing required field {key!r}")
    return value


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "model_type": "hy_v3",
        "hidden_act": "silu",
        "qk_norm": True,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"unsupported {key}={config.get(key)!r}; expected {expected!r}")
    if int(config_value(config, "first_k_dense_replace", 1)) < 1:
        raise ValueError("layer 0 is not configured as a dense MLP")
    rope = config_value(config, "rope_parameters")
    if rope.get("rope_type", "default") != "default":
        raise ValueError(f"only default RoPE is supported, got {rope!r}")
    if config.get("attention_bias", False) or config.get("mlp_bias", False):
        raise ValueError("this diagnostic expects bias-free Hy3 attention and MLP projections")


@torch.no_grad()
def layer0_reference(
    store: TensorStore,
    config: dict[str, Any],
    tokens: list[int],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    hidden_size = int(config_value(config, "hidden_size"))
    n_head = int(config_value(config, "num_attention_heads"))
    n_head_kv = int(config_value(config, "num_key_value_heads"))
    head_dim = int(config.get("head_dim", hidden_size // n_head))
    eps = float(config_value(config, "rms_norm_eps"))
    rope = config_value(config, "rope_parameters")
    rope_theta = float(config.get("rope_theta", rope.get("rope_theta")))
    if n_head % n_head_kv:
        raise ValueError(
            f"num_attention_heads={n_head} is not divisible by "
            f"num_key_value_heads={n_head_kv}"
        )

    prefix = "model.layers.0"
    embeddings = store.rows(
        "model.embed_tokens.weight", tokens, dtype=dtype, device=device
    )
    weights = {
        name: store.tensor(f"{prefix}.{key}", dtype=dtype, device=device)
        for name, key in {
            "input_norm": "input_layernorm.weight",
            "q": "self_attn.q_proj.weight",
            "k": "self_attn.k_proj.weight",
            "v": "self_attn.v_proj.weight",
            "o": "self_attn.o_proj.weight",
            "q_norm": "self_attn.q_norm.weight",
            "k_norm": "self_attn.k_norm.weight",
            "post_norm": "post_attention_layernorm.weight",
            "gate": "mlp.gate_proj.weight",
            "up": "mlp.up_proj.weight",
            "down": "mlp.down_proj.weight",
        }.items()
    }

    sequence = embeddings.unsqueeze(0)
    batch, seq_len, _ = sequence.shape
    normed = rms_norm(sequence, weights["input_norm"], eps)
    q = F.linear(normed, weights["q"]).view(batch, seq_len, n_head, head_dim).transpose(1, 2)
    k = F.linear(normed, weights["k"]).view(batch, seq_len, n_head_kv, head_dim).transpose(1, 2)
    v = F.linear(normed, weights["v"]).view(batch, seq_len, n_head_kv, head_dim).transpose(1, 2)
    q = rms_norm(q, weights["q_norm"], eps)
    k = rms_norm(k, weights["k_norm"], eps)

    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    freqs = torch.outer(positions, inv_freq)
    rope_emb = torch.cat((freqs, freqs), dim=-1)
    cos = rope_emb.cos().to(dtype=dtype)[None, None, :, :]
    sin = rope_emb.sin().to(dtype=dtype)[None, None, :, :]
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin

    groups = n_head // n_head_kv
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)
    scores = torch.matmul(q, k.transpose(2, 3)) * (head_dim**-0.5)
    causal_mask = torch.full(
        (seq_len, seq_len), float("-inf"), device=device, dtype=torch.float32
    ).triu(diagonal=1)
    scores = scores + causal_mask.to(dtype=scores.dtype)[None, None, :, :]
    attention_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    attention = torch.matmul(attention_weights, v)
    attention = attention.transpose(1, 2).contiguous().reshape(batch, seq_len, -1)
    attention = F.linear(attention, weights["o"])
    after_attention = sequence + attention

    mlp_input = rms_norm(after_attention, weights["post_norm"], eps)
    gate = F.linear(mlp_input, weights["gate"])
    up = F.linear(mlp_input, weights["up"])
    mlp = F.linear(F.silu(gate) * up, weights["down"])
    layer0 = after_attention + mlp
    return {
        "embedding": embeddings.float().cpu(),
        "attention_output": attention[0].float().cpu(),
        "after_attention": after_attention[0].float().cpu(),
        "mlp_output": mlp[0].float().cpu(),
        "layer0_residual": layer0[0].float().cpu(),
    }


def vector_stats(reference: torch.Tensor, observed: list[float]) -> dict[str, float]:
    candidate = torch.tensor(observed, dtype=torch.float32)
    if candidate.shape != reference.shape:
        raise ValueError(
            f"shape mismatch: reference {tuple(reference.shape)}, "
            f"bw24 {tuple(candidate.shape)}"
        )
    delta = candidate - reference
    candidate_norm = float(torch.linalg.vector_norm(candidate))
    reference_norm = float(torch.linalg.vector_norm(reference))
    denominator = candidate_norm * reference_norm
    if denominator:
        cosine = float(torch.dot(candidate, reference)) / denominator
    else:
        cosine = 1.0 if candidate_norm == reference_norm else 0.0
    return {
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "rmse": float(torch.sqrt(delta.square().mean())),
        "cosine": cosine,
    }


def load_bw24_records(path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("schema") != SCHEMA or record.get("producer") != "bw24":
                continue
            if "position" in record:
                position = int(record["position"])
                if position in records:
                    raise ValueError(f"duplicate bw24 position {position} at line {line_number}")
                records[position] = record
    return records


def emit(record: dict[str, Any], output: TextIO) -> None:
    print(json.dumps(record, separators=(",", ":"), allow_nan=False), file=output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="BF16 Hy3 safetensors directory")
    parser.add_argument("tokens", nargs="+", type=int, help="causal token-id sequence")
    parser.add_argument("--dtype", choices=("bf16", "f32"), default="bf16")
    parser.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    parser.add_argument("--compare", type=Path, help="optional bw24 JSONL from hy3-layer0-oracle")
    parser.add_argument("--output", type=Path, help="write JSONL here instead of stdout")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads((args.checkpoint / "config.json").read_text())
    validate_config(config)
    vocab_size = int(config_value(config, "vocab_size"))
    if any(token < 0 or token >= vocab_size for token in args.tokens):
        raise ValueError(f"token ids must be in [0, {vocab_size})")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    device = torch.device(args.device)

    with ExitStack() as stack:
        store = TensorStore(args.checkpoint, stack)
        stages = layer0_reference(
            store, config, args.tokens, dtype=dtype, device=device
        )
        bw24 = load_bw24_records(args.compare) if args.compare else {}
        if args.compare and set(bw24) != set(range(len(args.tokens))):
            raise ValueError(
                f"bw24 comparison positions are {sorted(bw24)}, expected "
                f"{list(range(len(args.tokens)))}"
            )
        output = args.output.open("w") if args.output else sys.stdout
        try:
            emit(
                {
                    "schema": SCHEMA,
                    "producer": "transformers_reference",
                    "checkpoint": str(args.checkpoint),
                    "tokens": args.tokens,
                    "n_embd": stages["embedding"].shape[-1],
                    "precision": args.dtype,
                    "transformers_commit": TRANSFORMERS_COMMIT,
                },
                output,
            )
            for position, token_id in enumerate(args.tokens):
                emit(
                    {
                        "schema": SCHEMA,
                        "producer": "transformers_reference",
                        "position": position,
                        "token_id": token_id,
                        **{
                            name: values[position].tolist()
                            for name, values in stages.items()
                        },
                    },
                    output,
                )
                if position in bw24:
                    observed = bw24[position]
                    if int(observed["token_id"]) != token_id:
                        raise ValueError(
                            f"token mismatch at position {position}: reference {token_id}, "
                            f"bw24 {observed['token_id']}"
                        )
                    emit(
                        {
                            "schema": SCHEMA,
                            "producer": "comparison",
                            "position": position,
                            "token_id": token_id,
                            **{
                                name: vector_stats(values[position], observed[name])
                                for name, values in stages.items()
                            },
                        },
                        output,
                    )
        finally:
            if args.output:
                output.close()


if __name__ == "__main__":
    main()
