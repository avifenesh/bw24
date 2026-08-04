#!/usr/bin/env python3
"""Arch-diff field extractor — the runbook §2 checklist, mechanized.

Usage: python3 arch-diff-fields.py <config.json> [reference-config.json]

One arg  = dump every checklist field (the day-one grab on the 3.8 config).
Two args = field-by-field diff against the reference (3.6-27B hf-min config),
           tagging each divergence FAST-PATH (numeric-only) or STOP-CLASS.
Self-test (2026-08-04): run on the 3.6 config against itself — all fields
extract, zero diffs. Receipt: arch-diff-selftest-36.txt.
"""
import json
import sys


def fields(path):
    cfg = json.load(open(path))
    tc = cfg.get("text_config", cfg)  # tolerate a flat (non-multimodal) config
    rp = tc.get("rope_parameters") or {}
    return {
        # STOP-CLASS on string/structure change
        "architectures": cfg.get("architectures"),
        "model_type": cfg.get("model_type"),
        "layer_types_pattern": "".join(
            "L" if t == "linear_attention" else "F" if t == "full_attention" else "?"
            for t in (tc.get("layer_types") or [])
        )[:16] or None,
        "full_attention_interval": tc.get("full_attention_interval"),
        "rope_type": rp.get("rope_type"),
        "mrope_interleaved": rp.get("mrope_interleaved"),
        "mrope_section": rp.get("mrope_section"),
        "num_experts": tc.get("num_experts"),  # any value = MoE-ification = STOP
        "attn_output_gate": tc.get("attn_output_gate"),
        "quantization_config.quant_method": (cfg.get("quantization_config") or {}).get("quant_method"),
        "quantization_config.fmt": (cfg.get("quantization_config") or {}).get("fmt"),
        "quantization_config.weight_block_size": (cfg.get("quantization_config") or {}).get("weight_block_size"),
        "quantization_config.activation_scheme": (cfg.get("quantization_config") or {}).get("activation_scheme"),
        # FAST-PATH on numeric-only change
        "num_hidden_layers": tc.get("num_hidden_layers"),
        "hidden_size": tc.get("hidden_size"),
        "intermediate_size": tc.get("intermediate_size"),
        "num_attention_heads": tc.get("num_attention_heads"),
        "num_key_value_heads": tc.get("num_key_value_heads"),
        "head_dim": tc.get("head_dim"),  # NEW value = FA-arm check (fa dispatch is head_dim-keyed)
        "linear_num_key_heads": tc.get("linear_num_key_heads"),
        "linear_num_value_heads": tc.get("linear_num_value_heads"),
        "linear_key_head_dim": tc.get("linear_key_head_dim"),
        "linear_value_head_dim": tc.get("linear_value_head_dim"),
        "linear_conv_kernel_dim": tc.get("linear_conv_kernel_dim"),
        "rope_theta": rp.get("rope_theta"),
        "partial_rotary_factor": rp.get("partial_rotary_factor"),
        "vocab_size": tc.get("vocab_size"),
        "bos_token_id": tc.get("bos_token_id"),
        "eos_token_id": tc.get("eos_token_id"),
        "image_token_id": cfg.get("image_token_id"),
        "video_token_id": cfg.get("video_token_id"),
        "mtp_num_hidden_layers": tc.get("mtp_num_hidden_layers"),
        "mtp_use_dedicated_embeddings": tc.get("mtp_use_dedicated_embeddings"),
        "max_position_embeddings": tc.get("max_position_embeddings"),
    }


STOP_CLASS = {
    "architectures", "model_type", "layer_types_pattern", "full_attention_interval",
    "rope_type", "mrope_interleaved", "num_experts", "attn_output_gate",
}

new = fields(sys.argv[1])
if len(sys.argv) < 3:
    for k, v in new.items():
        print(f"{k} = {v}")
    sys.exit(0)

ref = fields(sys.argv[2])
diffs = 0
for k in new:
    if new[k] != ref[k]:
        diffs += 1
        tag = "STOP-CLASS" if k in STOP_CLASS else "fast-path (numeric)"
        print(f"DIFF [{tag}] {k}: ref={ref[k]} -> new={new[k]}")
missing = [k for k in new if new[k] is None and ref[k] is not None]
for k in missing:
    print(f"MISSING (present in ref): {k} — treat as STOP until explained")
print(f"\n{diffs} diffs, {len(missing)} missing-vs-ref fields")
sys.exit(1 if any(k in STOP_CLASS for k in new if new[k] != ref[k]) or missing else 0)
