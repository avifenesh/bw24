#!/usr/bin/env python3
"""Diff a candidate Qwen 3.8 config against the frozen Qwen 3.6 contract.

Exit 0: same-architecture runbook path.
Exit 1: hard STOP; open a real bring-up lane.
Exit 2: architecture matches, but FP8 metadata is inconclusive; inspect tensor headers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_fields(path: Path) -> dict[str, object]:
    cfg = json.loads(path.read_text())
    text = cfg.get("text_config", cfg)
    rope = text.get("rope_parameters") or {}
    layer_types = text.get("layer_types") or []
    interval = text.get("full_attention_interval")
    expected_types = []
    if isinstance(interval, int) and interval > 0:
        expected_types = [
            "full_attention" if (index + 1) % interval == 0 else "linear_attention"
            for index in range(len(layer_types))
        ]
    quant = cfg.get("quantization_config") or {}

    return {
        # Engine dispatch contracts. Any divergence is a hard STOP.
        "architectures": cfg.get("architectures"),
        "model_type": cfg.get("model_type"),
        "text_config.model_type": text.get("model_type"),
        "attention_bias": text.get("attention_bias"),
        "attention_dropout": text.get("attention_dropout"),
        "attn_output_gate": text.get("attn_output_gate"),
        "output_gate_type": text.get("output_gate_type"),
        "hidden_act": text.get("hidden_act"),
        "tie_word_embeddings": text.get(
            "tie_word_embeddings", cfg.get("tie_word_embeddings")
        ),
        "full_attention_interval": interval,
        "layer_types_cycle": layer_types[:interval] if isinstance(interval, int) else None,
        "layer_types_match_interval": bool(layer_types) and layer_types == expected_types,
        "layer_types_match_count": len(layer_types) == text.get("num_hidden_layers"),
        "num_attention_heads": text.get("num_attention_heads"),
        "num_key_value_heads": text.get("num_key_value_heads"),
        "head_dim": text.get("head_dim"),
        "linear_num_key_heads": text.get("linear_num_key_heads"),
        "linear_num_value_heads": text.get("linear_num_value_heads"),
        "linear_key_head_dim": text.get("linear_key_head_dim"),
        "linear_value_head_dim": text.get("linear_value_head_dim"),
        "linear_conv_kernel_dim": text.get("linear_conv_kernel_dim"),
        "rope_parameters": rope,
        "partial_rotary_factor": text.get("partial_rotary_factor"),
        "num_experts": text.get("num_experts"),
        "mtp_num_hidden_layers": text.get("mtp_num_hidden_layers"),
        "mtp_use_dedicated_embeddings": text.get("mtp_use_dedicated_embeddings"),
        # Parsed shape/value fields. Changes stay in the runbook path but require all gates.
        "num_hidden_layers": text.get("num_hidden_layers"),
        "layer_types_count": len(layer_types),
        "hidden_size": text.get("hidden_size"),
        "intermediate_size": text.get("intermediate_size"),
        "vocab_size": text.get("vocab_size"),
        "bos_token_id": text.get("bos_token_id"),
        "eos_token_id": text.get("eos_token_id"),
        "image_token_id": cfg.get("image_token_id"),
        "video_token_id": cfg.get("video_token_id"),
        "max_position_embeddings": text.get("max_position_embeddings"),
        "rms_norm_eps": text.get("rms_norm_eps"),
        # Metadata only. Tensor headers remain authoritative.
        "quantization_config.quant_method": quant.get("quant_method"),
        "quantization_config.fmt": quant.get("fmt"),
        "quantization_config.weight_block_size": quant.get("weight_block_size"),
        "quantization_config.activation_scheme": quant.get("activation_scheme"),
    }


STOP_FIELDS = {
    "architectures",
    "model_type",
    "text_config.model_type",
    "attention_bias",
    "attention_dropout",
    "attn_output_gate",
    "output_gate_type",
    "hidden_act",
    "tie_word_embeddings",
    "full_attention_interval",
    "layer_types_cycle",
    "layer_types_match_interval",
    "layer_types_match_count",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "linear_num_key_heads",
    "linear_num_value_heads",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "linear_conv_kernel_dim",
    "rope_parameters",
    "partial_rotary_factor",
    "num_experts",
    "mtp_num_hidden_layers",
    "mtp_use_dedicated_embeddings",
}

QUANT_FIELDS = {
    "quantization_config.quant_method",
    "quantization_config.fmt",
    "quantization_config.weight_block_size",
    "quantization_config.activation_scheme",
}


def fp8_metadata_verdict(fields: dict[str, object]) -> tuple[str, str]:
    method = fields["quantization_config.quant_method"]
    fmt = fields["quantization_config.fmt"]
    block = fields["quantization_config.weight_block_size"]
    activation = fields["quantization_config.activation_scheme"]

    if method not in (None, "fp8"):
        return "STOP", f"quant_method={method!r}, expected 'fp8'"
    if fmt not in (None, "e4m3"):
        return "STOP", f"fmt={fmt!r}, expected 'e4m3'"
    if block not in (None, [128, 128]):
        return "STOP", f"weight_block_size={block!r}, expected [128, 128] or tensor-proven scalar"
    if activation not in (None, "dynamic"):
        return "STOP", f"activation_scheme={activation!r}, expected 'dynamic'"
    if (method, fmt, block, activation) == ("fp8", "e4m3", [128, 128], "dynamic"):
        return "PASS", "official Qwen 3.6 block-128 metadata contract"
    return "REVIEW", "metadata is incomplete; tools/inspect-fp8-st.py must prove tensor classes"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect-fp8", action="store_true")
    parser.add_argument("candidate", type=Path)
    parser.add_argument("reference", type=Path, nargs="?")
    args = parser.parse_args()

    candidate = load_fields(args.candidate)
    if args.reference is None:
        for key, value in candidate.items():
            print(f"{key} = {value}")
        if args.expect_fp8:
            verdict, reason = fp8_metadata_verdict(candidate)
            print(f"\nFP8 metadata: {verdict} - {reason}")
            return {"PASS": 0, "STOP": 1, "REVIEW": 2}[verdict]
        return 0

    reference = load_fields(args.reference)
    hard_stops = 0
    diffs = 0
    for key, new_value in candidate.items():
        if key in QUANT_FIELDS:
            continue
        old_value = reference[key]
        if new_value == old_value:
            continue
        diffs += 1
        if key in STOP_FIELDS:
            hard_stops += 1
            tag = "STOP-ARCH"
        else:
            tag = "GO-WITH-GATES"
        print(f"DIFF [{tag}] {key}: ref={old_value!r} -> new={new_value!r}")

    missing = [
        key
        for key in STOP_FIELDS
        if reference.get(key) is not None and candidate.get(key) is None
    ]
    for key in sorted(missing):
        print(f"MISSING [STOP-ARCH] {key}: present in reference")
    hard_stops += len(missing)

    quant_exit = 0
    if args.expect_fp8:
        verdict, reason = fp8_metadata_verdict(candidate)
        print(f"FP8 metadata: {verdict} - {reason}")
        quant_exit = {"PASS": 0, "STOP": 1, "REVIEW": 2}[verdict]

    print(f"\n{diffs} architecture diffs, {hard_stops} hard stops")
    if hard_stops or quant_exit == 1:
        return 1
    return quant_exit


if __name__ == "__main__":
    raise SystemExit(main())
