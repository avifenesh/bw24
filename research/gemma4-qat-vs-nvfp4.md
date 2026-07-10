# gemma-4 26B: QAT-Q4_0 vs NVFP4 (assessment, 2026-07-10)

Question (goal item): does the QAT GGUF make sense as the daily format, and is the NVFP4 ST
checkpoint (`/data/ai-ml/hf-models/gemma4-26b-a4b-nvfp4/`, modelopt 0.43) worth porting?

## What the NVFP4 checkpoint actually is

From `hf_quant_config.json` + the tensor index (measured, not assumed):

- **Only the routed experts are NVFP4** — 11,520 `weight_scale` tensors = 128 experts x 3
  projections x 30 layers, exactly. group_size 16, per-expert `weight_scale_2` macro.
- `exclude_modules` lists **every one of the 30 layers'** `self_attn*`, `mlp*` (the shared
  parallel FFN) and `router*` — all stay **bf16**. `lm_head` excluded too (tied bf16 embd).
- KV-cache marked FP8 (serving-stack hint; irrelevant to weight bytes).
- File total 18.78 GB vs the QAT-Q4_0 GGUF's 13.4 GB.

## Decode byte accounting (the wall that sets tok/s on this rig)

Per decode step, weights read (26B geometry: trunk attn+shared ~925 MB at 0.5625 B/w):

| component            | QAT-Q4_0 (measured daily) | NVFP4 ST checkpoint |
|----------------------|---------------------------|---------------------|
| trunk attn + shared  | ~925 MB (Q4_0)            | ~3.29 GB (bf16, 3.56x) |
| routed experts (8/128)| ~1.00 GB (Q4_0 0.5625 B/w)| ~1.00 GB (NVFP4 0.5625 B/w — SAME) |
| lm_head              | ~605 MB (Q6_K tied)       | ~1.48 GB (bf16 tied) |
| **total / step**     | **~2.53 GB**              | **~5.77 GB**        |

At the 858 GB/s wall: QAT floor ~2.95 ms/step (339 tok/s ceiling; 178.7 achieved so far);
NVFP4 floor ~6.7 ms/step (**~149 tok/s ceiling — below what QAT already achieves**).

## Verdict

- **QAT-Q4_0 is the right daily format** for gemma-4 on this rig. The NVFP4 checkpoint is
  strictly worse for decode: identical expert bytes (NVFP4 and Q4_0 are both 0.5625 B/w),
  2-3.5x the bytes everywhere else. Its ceiling sits *below* our current QAT throughput.
- Quality side needs no measurement to rank: Q4_0 here is Google's quantization-aware-trained
  release (trained for this format); the NVFP4 experts are modelopt PTQ of the same weights.
  Equal bytes + PTQ-vs-QAT = no quality upside either.
- A "fixed" NVFP4 variant (re-quantizing the excluded trunk/head ourselves) would no longer
  be the NVFP4 checkpoint under assessment — it would be our own PTQ mix, forfeiting the QAT
  training for zero byte advantage (owner rule also blocks the W4A4 activation path on
  quality grounds).
- The NVFP4 ST port (gemma4 safetensors config arm + modelopt name mapping) is therefore NOT
  justified for the 26B. Revisit only if a future gemma NVFP4 drop quantizes the full trunk.
