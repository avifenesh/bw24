# Hy3 REAP50 port scope

Status as of 2026-07-09T01:02:05+03:00: CPU/disk-only preparation lane. No GPU loads, kernel checks,
or target-rig quality gates were run here. All quality and performance statements in this dossier are
unverified — pending target-rig gates.

## Scope law

This lane prepares Tencent Hy3 REAP50 for the bw24 local rig target, not a generic serving stack. The
owner scope in `HANDOVER.md` keeps the target on this machine, RTX 5090 Laptop 24 GB plus 60 GB RAM,
and calls Hy3 REAP50 the next spilled-MoE target after M3 because the disk tier is the specialty being
developed (`HANDOVER.md:13`, `HANDOVER.md:15`). The format decision says GGUF and safetensors are
import formats; the runtime target is a bw24 internal layout chosen per datapath, especially for
CPU-spilled experts (`docs/decisions/FORMAT-DECISION.md:9`, `docs/decisions/FORMAT-DECISION.md:45`).

## External facts checked

- Model artifact: [pipenetwork/Hy3-REAP50-MLX-4bit](https://huggingface.co/pipenetwork/Hy3-REAP50-MLX-4bit), MLX affine 4-bit, ~85.4 GB, REAP-pruned from Tencent Hy3 with 96 of 192 routed experts kept per layer.
- Base architecture: [Tencent Hy3](https://github.com/Tencent-Hunyuan/Hy3) and [Transformers hy_v3 docs](https://huggingface.co/docs/transformers/en/model_doc/hy_v3) describe dense-MoE hybrid structure, QK norm, sigmoid router, expert bias, shared expert, GQA, and MTP support.
- Training/runtime guidance: [NVIDIA NeMo Hy3 guide](https://docs.nvidia.com/nemo/automodel/nightly/guides/llm/hy3.html) confirms 80 layers, layer 0 dense, layers 1-79 MoE, 192 routed experts in the base, top-8, one shared expert, QK RMSNorm before RoPE, and MTP layer filtering.
- MLX quantization: [mlx.core.quantize](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html) documents affine quantization as packed integer weights plus per-group scales and biases; default 4-bit group size 64.
- Router precision note: [vLLM issue 47777](https://github.com/vllm-project/vllm/issues/47777) reports Hy3 `expert_bias` must stay F32 because near-tie top-k selection flips under downcast.

## Download dossier

Command used:

```bash
HF_HUB_DISABLE_XET=1 hf download pipenetwork/Hy3-REAP50-MLX-4bit --local-dir /data/ai-ml/hf-models/hy3-reap50-mlx --max-workers 1
```

`--max-workers 1` was chosen after a parallel/Xet download reached multi-GB RSS and then wedged on
this shared host. The final completed download used the non-Xet HTTP/LFS path and the HF CLI was
authenticated as `Avifenesh`. The top-level repo inventory is 24 completed files: 16 safetensor
shards, index, config, tokenizer files, README, chat template, and `.gitattributes`. `du -sh` on the
source directory currently reports 94G because the failed attempts left about 14G of hidden
`.cache/huggingface/download/*.incomplete` retry files; the indexed safetensor payload is
85,437,372,032 bytes.

Config and sidecar hashes already downloaded:

| file | bytes | sha256 |
|---|---:|---|
| `config.json` | 15,319 | `49db5afa1922a60f15dbbd51ba037b4f0427bc670210abbf24d0a5458dc7301f` |
| `generation_config.json` | 204 | `c683cd1812f6816c7c8248c56f6fe481e48f5a185ad1723bdfcf479db50073d8` |
| `tokenizer_config.json` | 165,971 | `1af226fc70b260371ca1f08053768b80a9d58b9d79e8eab718160f52783f7ceb` |
| `tokenizer.json` | 9,527,406 | `446e0b59cd941637c0ddfd84e12ee1f49480fd12097f86a9d2fec8ebd0c7ff6c` |
| `model.safetensors.index.json` | 274,220 | `5947de06359ab0fc06835a569bed7ec67718170db5fa368657b05984c52dbc09` |

Completed top-level source file inventory:

| file | bytes |
|---|---:|
| `.gitattributes` | 1,519 |
| `README.md` | 452 |
| `chat_template.jinja` | 10,223 |
| `config.json` | 15,319 |
| `generation_config.json` | 204 |
| `model-00001-of-00016.safetensors` | 5,086,063,650 |
| `model-00002-of-00016.safetensors` | 5,363,698,009 |
| `model-00003-of-00016.safetensors` | 5,363,698,184 |
| `model-00004-of-00016.safetensors` | 5,363,698,202 |
| `model-00005-of-00016.safetensors` | 5,363,698,080 |
| `model-00006-of-00016.safetensors` | 5,363,698,250 |
| `model-00007-of-00016.safetensors` | 5,363,698,230 |
| `model-00008-of-00016.safetensors` | 5,363,698,106 |
| `model-00009-of-00016.safetensors` | 5,363,698,186 |
| `model-00010-of-00016.safetensors` | 5,363,698,162 |
| `model-00011-of-00016.safetensors` | 5,363,698,176 |
| `model-00012-of-00016.safetensors` | 5,363,698,170 |
| `model-00013-of-00016.safetensors` | 5,363,698,256 |
| `model-00014-of-00016.safetensors` | 5,363,698,246 |
| `model-00015-of-00016.safetensors` | 5,363,698,148 |
| `model-00016-of-00016.safetensors` | 5,259,895,240 |
| `model.safetensors.index.json` | 274,220 |
| `tokenizer.json` | 9,527,406 |
| `tokenizer_config.json` | 165,971 |

Index metadata:

| item | value |
|---|---:|
| tensors in weight map | 3,034 |
| total safetensor payload bytes | 85,437,372,032 |
| total parameters | 151,859,257,344 |
| shards | `model-00001-of-00016.safetensors` through `model-00016-of-00016.safetensors` |

Full tensor headers are materialized in `research/hy3-reap50-tensor-inventory.tsv`:

| dtype | tensors | payload bytes |
|---|---:|---:|
| BF16 | 2,077 | 9,492,520,960 |
| F32 | 79 | 30,336 |
| U32 | 878 | 75,944,820,736 |
| total | 3,034 | 85,437,372,032 |

## Architecture from config and local headers

| area | Hy3 REAP50 artifact |
|---|---|
| HF architecture | `HYV3ForCausalLM`, `model_type=hy_v3` |
| layers | 80 transformer layers |
| hidden | `hidden_size=4096` |
| attention | GQA, `num_attention_heads=64`, `num_key_value_heads=8`, `head_dim=128`, QK norm on every layer |
| Q/O projection wrinkle | local header: `q_proj.weight` packed `[8192,512]` -> logical `[8192,4096]`; `o_proj.weight` packed `[4096,1024]` -> logical `[4096,8192]` |
| RoPE/context | `max_position_embeddings=262144`, `rope_theta=11158840.0`, full `head_dim` rotary in this config |
| dense FFN | layer 0 dense, `intermediate_size=13312` |
| MoE layers | layers 1-79 |
| routed experts | REAP kept 96 experts/layer from original 192 |
| routing | top-8, sigmoid router, route normalization, `router_scaling_factor=2.826`, F32 `expert_bias` selection bias |
| expert hidden | `moe_intermediate_size=1536` / `expert_hidden_dim=1536` |
| shared expert | one shared MLP per MoE layer; local headers show width 1536, same as one routed expert |
| tokenizer | `vocab_size=120832`, local tokenizer files present |
| MTP | config says `num_nextn_predict_layers=1`; the weight index has no `mtp`, `next`, or layer index >=80 tensor names. Treat as absent/stripped in this artifact until source-code parity work proves otherwise. |
| quantization | default MLX affine 4-bit, group size 64; router gates override to 8-bit affine; norms and expert bias are F32 |

Representative local header shapes from shard 1:

| tensor | dtype | stored shape | logical shape |
|---|---|---:|---:|
| `model.embed_tokens.weight` | U32 | `[120832,512]` | `[120832,4096]` |
| `model.layers.0.self_attn.q_proj.weight` | U32 | `[8192,512]` | `[8192,4096]` |
| `model.layers.0.self_attn.k_proj.weight` | U32 | `[1024,512]` | `[1024,4096]` |
| `model.layers.0.self_attn.v_proj.weight` | U32 | `[1024,512]` | `[1024,4096]` |
| `model.layers.0.self_attn.o_proj.weight` | U32 | `[4096,1024]` | `[4096,8192]` |
| `model.layers.0.mlp.gate_proj.weight` | U32 | `[13312,512]` | `[13312,4096]` |
| `model.layers.1.mlp.router.expert_bias` | F32 | `[96]` | `[96]` |
| `model.layers.1.mlp.router.gate.weight` | U32 | `[96,1024]` | `[96,4096]` at 8-bit affine |
| `model.layers.1.mlp.shared_mlp.gate_proj.weight` | U32 | `[1536,512]` | `[1536,4096]` |
| `model.layers.1.mlp.switch_mlp.gate_proj.weight` | U32 | `[96,1536,512]` | `[96,1536,4096]` |
| `model.layers.1.mlp.switch_mlp.down_proj.weight` | U32 | `[96,4096,192]` | `[96,4096,1536]` |

Index pattern counts:

| pattern | count |
|---|---:|
| per-layer attention/norm weight groups | 80 each |
| `mlp.router.{gate,expert_bias}` groups | 79 each |
| `mlp.shared_mlp.{gate,up,down}_proj` groups | 79 each |
| `mlp.switch_mlp.{gate,up,down}_proj` groups | 79 each |
| layer 0 dense `mlp.{gate,up,down}_proj` groups | 1 each |
| explicit MTP/next tensors | 0 |

## Diff against bw24

| item | current bw24 support | classification | effort |
|---|---|---|---|
| `hy_v3` arch string | `Arch::from_hf_model_type` maps qwen/olmoe/M3 but not `hy_v3` (`crates/bw24-gguf/src/config.rs:36`) | new forward/config code | Small: add `Hy3` or equivalent dense-attn MoE arch, parse Hy3 config names, keep `is_hybrid=false`, `is_moe=true`. |
| Dense full attention with QK norm/GQA/hd128 | dense path already uses tensor `out_features()` for Q and applies per-head QK norm/RoPE/SDPA (`crates/bw24-engine/src/forward.rs:37`) | loads as-is after mapping | Low: no CUDA kernel expected; validate q_out=8192/o_in=8192 with argmax gate later. |
| Hybrid attention dispatch | `run-safetensors` sends `is_hybrid()` arches to `HybridModel`, else dense `Model` (`crates/bw24-engine/src/bin/run_safetensors.rs:24`) | new arch dispatch decision | Hy3 should remain dense-attn MoE, not qwen35 hybrid. |
| Safetensors U32 MLX weights | safetensors dtype mapping does not support U32 model weights (`crates/bw24-gguf/src/safetensors.rs:28`) and current source adapter knows modelopt/Reza NVFP4, FP8, BF16/F32 (`crates/bw24-gguf/src/source.rs:170`) | new import/transcode code | Medium: tool-level MLX affine -> Q4_K now added; runtime can consume a repack dir/manifest in the port lane. |
| Tensor names | current M3 mapping is `block_sparse_moe.*` and qwen MoE mapping is `mlp.experts.*` (`crates/bw24-gguf/src/hf_mapping.rs:37`, `crates/bw24-gguf/src/hf_mapping.rs:84`) | new loader mapping | Small/medium: map Hy3 `mlp.router.*`, `mlp.shared_mlp.*`, and stacked `mlp.switch_mlp.*` to bw24 `ffn_*` names. |
| Stacked routed experts | `HostExps::load_stacked_from_source` supports a 3D stacked tensor and asserts expert stride (`crates/bw24-engine/src/model.rs:606`) | loads as-is after source exposes Q4_K bytes | Low runtime effort once the repack manifest provides contiguous expert-axis-slowest files. |
| Disk spill tier | GGUF tiering mmaps contiguous expert blocks (`crates/bw24-engine/src/hybrid.rs:53`); M3 safetensors path stream-repacks to `.bw24-repack` and mmaps it (`crates/bw24-engine/src/model.rs:744`) | new loader glue | Medium: adapt the M3 safetensors-dir precedent to read this Q4_K repack directory and per-expert manifest. |
| Sigmoid router + bias | M3 host oracle selects over `sigmoid + bias`, weights use plain sigmoid, then normalize and scale (`crates/bw24-engine/src/hybrid_forward.rs:1094`) | loads as-is conceptually | Medium: add Hy3 router config fields (`moe_router_use_sigmoid`, `moe_router_enable_expert_bias`, `route_norm`, `router_scaling_factor`); keep router/expert_bias F32. |
| Device router | M3 sigmoid routing currently must not enter softmax fused-router device arms (`crates/bw24-engine/src/hybrid_forward.rs:773`) | new kernel only for optimization | Not required for first correctness port; a sigmoid fused-router kernel is later perf work. |
| Shared MLP | shared expert tensors are optional and run through existing `gate_shexp/up_shexp/down_shexp` path (`crates/bw24-engine/src/hybrid.rs:101`) | loads as-is after mapping | Low: Hy3 shared expert is ungated, matching M3's add-direct behavior (`crates/bw24-engine/src/hybrid_forward.rs:977`). |
| Router precision | loader law keeps `ffn_gate_inp.weight` F32 because top-k selection is discontinuous (`crates/bw24-engine/src/model.rs:92`) | loads as-is policy | Low: router gate is 8-bit affine on disk but transcoder maps it to F32; expert_bias remains F32. |
| MTP | config has one next-token-predict layer but tensor index has no MTP names | unknown/new forward code if present | Blocker for speculation only, not plain decode. Re-check after full source-code comparison. |

## Transcode decision

Use MLX affine 4-bit -> GGUF `Q4_K`, not NVFP4.

Reason:

- MLX affine stores per-group scale plus bias/zero-point; `Q4_K` keeps an asymmetric scale+min form.
- `NVFP4` is symmetric e2m1. bw24 already measured a real acceptance tax when converting asymmetric
  k-quants to NVFP4: hard content p2 74.0 -> 70.7 and p3 66.9 -> 64.9 (`research/tune-data/rig5090.jsonl:220`).
- The project flag docs keep `BW24_KQ_NVFP4` opt-in because bpw equality is not quality-class equality
  for this conversion (`docs/FLAGS.md:90`).

The new tool is `tools/hy3_mlx_to_q4k.py`. It is CPU-only, uses mmap/json/NumPy, and streams row
chunks. It writes:

- `manifest.json`
- `tensors/*.q4k` / `tensors/*.f32` / `tensors/*.bin`
- `experts/blkL-{gate,up,down}-96xOxI.q4k`

The separate inventory subcommand writes `research/hy3-reap50-tensor-inventory.tsv`.

The manifest carries per-expert offsets:

```text
offset = expert_id * expert_stride
expert_stride = out_f * row_bytes
row_bytes = in_f / 256 * 144
```

For Hy3 routed experts, each gate/up/down projection has the same Q4_K stride:

| projection | logical shape per layer | row bytes | bytes/expert/proj |
|---|---:|---:|---:|
| gate | `[96,1536,4096]` | 2,304 | 3,538,944 |
| up | `[96,1536,4096]` | 2,304 | 3,538,944 |
| down | `[96,4096,1536]` | 864 | 3,538,944 |

One full routed expert is 10,616,832 bytes across gate/up/down.

Completed transcode command:

```bash
python3 tools/hy3_mlx_to_q4k.py transcode /data/ai-ml/hf-models/hy3-reap50-mlx /data/ai-ml/hf-models/hy3-reap50-q4k-bw24 --max-work-mb 64
```

The first transcode attempt showed source-mmap page accumulation above 2 GB RSS. The tool now closes
cached shard mmaps after each tensor and calls `malloc_trim` on Linux; the completed run stayed in
the roughly 225-500 MB RSS band while writing the 85.53 GB manifest payload.

Completed output:

| item | value |
|---|---:|
| output directory | `/data/ai-ml/hf-models/hy3-reap50-q4k-bw24` |
| `du -sh` | 80G |
| manifest payload bytes | 85,528,622,720 |
| manifest tensor records | 1,278 |
| qtype counts | Q4_K 799, BF16 321, F32 158 |
| routed expert files | 237 |
| routed expert manifest entries | 237 |
| MoE layers covered | 79, layers 1-79 |
| per-projection entries | gate 79, up 79, down 79 |
| expert offsets per file | 96 |
| expert stride per projection | 3,538,944 bytes |
| quality label | unverified — pending target-rig gates |

Byte-level tests run:

```bash
python3 -m py_compile tools/hy3_mlx_to_q4k.py
python3 tools/hy3_mlx_to_q4k.py test --real-model-dir /data/ai-ml/hf-models/hy3-reap50-mlx --real-limit 5
```

Test coverage:

- synthetic MLX affine encode/dequant bound check
- Q4_K byte determinism
- vectorized Q4_K dequant equality against a scalar GGUF-layout reference
- streamed safetensors transcode equality against in-memory transcode
- synthetic expert-offset manifest check
- sampled real tensors: `lm_head.weight` max_abs 0.005059, `model.embed_tokens.weight` max_abs 0.010481, layer-0 dense `down/gate/up` max_abs 0.008331 / 0.005579 / 0.003926

The sampled max_abs values are Q4_K transcode rounding evidence only; they are not model quality
claims. End-to-end quality remains unverified — pending target-rig gates.

## Spill-tier sizing memo

All numbers are byte-layout estimates, unverified — pending target-rig gates.

| component | estimate |
|---|---:|
| routed experts | 79 layers * 96 experts * 10,616,832 bytes = 80.52 GB |
| attention Q/K/V/O | about 6.04B params * 0.5625 B/w = 3.40 GB |
| shared experts | 79 layers * one 1536-wide expert * 0.5625 B/w = 0.84 GB |
| layer 0 dense FFN | about 0.09 GB |
| token embedding + lm head | about 0.56 GB |
| router gates dequantized to F32 | about 0.12 GB |
| norms/biases | tiny |
| completed repack payload | 85.53 GB manifest bytes, 80G by `du -sh` |

Initial placement recommendation for the port lane:

- VRAM resident: non-expert body first, roughly 5 GB plus runtime/KV/scratch; use remaining VRAM for the existing MoE SLRU/resident expert machinery.
- RAM/page cache: keep the Q4_K repack files mmap-backed and let the page cache carry as much of the 80.52 GB expert set as the live host budget allows. Avoid pinned slabs by default; M3 measured pinning 26 GB as a 30x regression because it evicted page cache (`HANDOVER.md:26`, `docs/FLAGS.md:88`).
- NVMe: expected to remain active but smaller than M3. If Hy3 follows M3's measured 77% touched profile, touched routed-expert bytes over a similar trace are about 62 GB, versus roughly 94 GB for M3's 122 GB expert set (`HANDOVER.md:26`). That suggests lower absolute miss traffic, but this is unverified — pending target-rig gates.

Done criteria status in this lane:

- full HF download visible with shard inventory: done
- full `research/hy3-reap50-tensor-inventory.tsv`: done
- full Q4_K repack directory plus manifest under `/data/ai-ml/hf-models/`: done
- CPU-only transcoder tests with sampled real tensors: done
- JSONL research row and commit on `lane/hy3-prep`: done before handoff
- target-rig forward/quality/perf gates: intentionally not run in this CPU/disk-only lane
