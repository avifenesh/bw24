# Compressed-Tensors NVFP4 Safetensors Mapping

## Source

apolo13x/Qwen3.5-9B-NVFP4 (produced by vllm/llm-compressor)
Format: compressed-tensors (HF ecosystem standard for quantized checkpoints)

## On-Disk Layout

Each quantized linear layer `<stem>` has three siblings:

| Tensor name               | dtype   | shape                | semantics                           |
|---------------------------|---------|----------------------|-------------------------------------|
| `<stem>.weight_packed`    | U8      | [out_features, in_features/2] | e2m1 4-bit codes, 2 codes/byte (low nibble first) |
| `<stem>.weight_scale`     | F8_E4M3 | [out_features, in_features/16] | per-16-element UE4M3 block scales |
| `<stem>.weight_global_scale` | F32  | scalar [1]           | per-tensor DIVISOR (NOT multiplier) |

Unquantized layers (attn_q/k/v/o on some models) remain BF16 at full shape.

## Critical Semantics: Global Scale is a DIVISOR

```
dequant = e2m1_code * ue4m3_scale / global_scale
```

Compare with modelopt (AxionML format):
```
dequant = e2m1_code * ue4m3_scale * scale_2
```

Relationship: `scale_2 = 1.0 / global_scale`

Typical value for Qwen3.5-9B: global_scale = 9408.0, scale_2 = 0.000106

## Engine Integration (source.rs)

Two inversion points are required:

1. **`nvfp4_quant()` path** — reads weight_global_scale, inverts to produce `macro_s = 1.0 / gs`.
   This feeds the repack path that writes the macro-scale into the GGUF block header.

2. **`.scale` sibling lookup in `find()`** — the engine reads `<stem>.scale` separately for
   the post-matmul device-side multiply. Must return Cow::Owned with inverted value
   (not raw borrowed bytes).

Both paths MUST agree on the inverted value. If either returns the raw divisor (9408.0),
the dequantized values are ~9408x too large and produce garbage (all-zero argmax).

## Detection Heuristic

In `nvfp4_quant()`:
1. Try `<stem>.weight_packed` (U8) — if missing, not an NVFP4 tensor
2. Try `<stem>.weight_scale` (F8_E4M3) — block scales
3. Try `<stem>.weight_scale_2` (F32) — if found: modelopt, use directly as multiplier
4. Try `<stem>.weight_global_scale` (F32) — if found: compressed-tensors, INVERT

## BF16 Attention Projections

The apolo13x checkpoint leaves attention Q/K/V/O projections unquantized (BF16).
These hit the Float-poison loader law: any BF16 2D tensor >= 1M elements MUST be
Q8_0 re-encoded to ride the fast q8 matvec path. The generalized gate in `find()`
handles this automatically (no name-pattern check needed, just shape threshold).

## Performance Note

CT arm measured 111.56 tok/s vs modelopt 127.88 tok/s (13% delta).
Root cause likely: 2-shard model (12GB + 487MB) vs modelopt single shard (9.4GB).
The multi-shard mmap has slightly higher lookup overhead per tensor.
Token quality is equivalent (first 27/32 greedy tokens identical; divergence from
1/gs FP rounding at the last bit is expected and acceptable for 4-bit quant).

## References

- vllm compressed-tensors spec: https://github.com/vllm-project/llm-compressor
- NVIDIA modelopt NVFP4: TensorRT-LLM modelopt quantization
- On-disk format identified by inspecting safetensors metadata headers
