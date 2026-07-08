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

## Loader glue addendum (2026-07-09)

This continuation implements the CPU-only loader pieces for the transcoded REAP checkpoint. No GPU
model load, `kernel-check`, `run-gen`, or `run-spec` was run in this lane.

### Implemented loader surface

| item | implementation | status |
|---|---|---|
| `hy_v3` config parse | `Arch::Hy3`, dense-attention MoE, `is_hybrid=false`, `is_moe=true`, nested `rope_parameters.rope_theta`, `expert_hidden_dim`, sigmoid router, expert bias, route norm, router scale 2.826, QK norm, default layer-0 dense marker | implemented in `crates/bw24-gguf/src/config.rs:16`, `crates/bw24-gguf/src/config.rs:67`, `crates/bw24-gguf/src/config.rs:113`, `crates/bw24-gguf/src/config.rs:503` |
| HF/MLX name map | `mlp.router.*`, `mlp.shared_mlp.*`, stacked `mlp.switch_mlp.*`, and layer-0 dense `mlp.*_proj` map to bw24 `ffn_*` names | implemented in `crates/bw24-gguf/src/hf_mapping.rs:55` |
| Repack source | `Hy3RepackSource` reads `manifest.json`, mmaps tensor files, exposes Q4_K 2D/3D bytes, F32 router/bias tensors, and dequantized F32 one-dimensional BF16 norm tensors | implemented in `crates/bw24-gguf/src/source.rs:115`, `crates/bw24-gguf/src/source.rs:195`, `crates/bw24-gguf/src/source.rs:204` |
| Stripped MTP override | REAP config says `num_nextn_predict_layers=1`, but the repack manifest only has `blk.0..blk.79`; source config is overridden to `n_layer=80`, `nextn_predict_layers=0` for this artifact | implemented in `crates/bw24-gguf/src/source.rs:242` |
| Layer-0 dense FFN | Dense-attention loader uses `Ffn::Dense` for Hy3 layers `< first_k_dense_replace`, then existing MoE loader for layers 1-79 | implemented in `crates/bw24-engine/src/model.rs:485` |

The source adapter is intentionally a repack-directory reader, not a GGUF writer. It keeps the
single-file GGUF question out of the loader lane while preserving the existing `TensorSource`
contract used by GGUF and safetensors paths.

### CPU-only tests added

The new tests are:

- `config::hf_tests::parse_hy3_reap_config`
- `hf_mapping::tests::hy3_moe_names`
- `source::hy3_repack_probe::hy3_manifest_offset_roundtrip`
- `source::hy3_repack_probe::hy3_inventory_dtype_shape_assertions`
- `source::hy3_repack_probe::hy3_load_plan_dry_run_no_cuda`

The manifest offset test checks expert 37 in `blk.1.ffn_gate_exps.weight` by comparing the source
view against a 64-byte direct read at `37 * 3,538,944`. The inventory test checks TSV source
dtype/shape rows against the source adapter's exposed qtype/ne for representative attention, dense
MLP, router, shared MLP, stacked expert, and norm tensors. The dry-run test walks every tensor name
the dense loader will request for all 80 REAP layers without constructing an `Engine` or touching
CUDA.

### MTP base-check result

Current Hugging Face metadata was checked on 2026-07-09 with `hf download` metadata-only pulls for
[`tencent/Hy3`](https://huggingface.co/tencent/Hy3),
[`tencent/Hy3-preview-Base`](https://huggingface.co/tencent/Hy3-preview-Base), and
[`tencent/Hy3-preview`](https://huggingface.co/tencent/Hy3-preview). All three configs report
`num_hidden_layers=80` and `num_nextn_predict_layers=1`. None of the indexes contains tensor names
matching `mtp`, `nextn`, or `next`, but all three carry an appended `model.layers.80.*` block.

For `tencent/Hy3`, layer 80 has 593 tensor keys:

| class | evidence |
|---|---|
| MTP glue | `model.layers.80.eh_proj.weight`, `enorm.weight`, `hnorm.weight` |
| full-attention block | q/k/v/o projections plus q/k norms and layer norms |
| MoE block | router gate, expert bias, shared MLP, and 192 experts x gate/up/down |
| shard files | current repo layer-80 tensors are in `model-00018-of-00099`, `model-00066-of-00099`, `model-00091-of-00099`, `model-00096-of-00099`, `model-00097-of-00099` |

Conclusion: the base checkpoint does carry the `num_nextn_predict_layers=1` head, stored as appended
`model.layers.80.*`; the REAP artifact stripped that block. A later spec lane can extract it by
downloading only the layer-80 shard set, header-scanning those safetensors, and transcoding the
appended block through the same Q4_K path. Expected Q4_K expert payload for the layer-80 MoE portion
is about `192 * 10,616,832 = 2.04 GB`, plus attention/glue/norm tensors. The extraction lane must
also handle the base head's 192-expert MLP while the REAP trunk has 96 routed experts; that is a
spec-head loader/config detail, not part of this plain-loader lane.

If a future metadata refresh shows the official repo has moved, treat these as the 2026-07-09 HF
index facts and re-run the metadata-only check before extracting.

### Remaining GPU-gated work

- Hy3 forward correctness: sigmoid router, `expert_bias` selection, route normalization, and
  `router_scaling_factor=2.826` need target-rig validation against a reference.
- Argmax/quality gates: all quality claims remain unverified — pending target-rig gates.
- Spill-tier bring-up: resident/RAM/NVMe split must be validated with real miss traffic on the
  RTX 5090 Laptop path.
- Speculative decode: base layer-80 MTP extraction/transcode plus per-head expert-count handling
  remain future work. (Extraction/transcode done 2026-07-09 — see the MTP head addendum below;
  the rig-side loader/forward work remains.)

## MTP head extraction + transcode addendum (2026-07-09, lane/hy3-mtp)

CPU/disk-only lane. No GPU loads, `kernel-check`, `run-gen`, or `run-spec`. All quality statements
remain unverified — pending target-rig gates.

### Selective download

The 593 `model.layers.80.*` keys of [tencent/Hy3](https://huggingface.co/tencent/Hy3) live in five
of the repo's 99 shards. Mapped via `model.safetensors.index.json` first, then only those shards
were pulled:

| shard | bytes |
|---|---:|
| `model-00018-of-00099.safetensors` | 1.1 GB (eh_proj, o_proj) |
| `model-00066-of-00099.safetensors` | 1.0 GB (q/k/v_proj) |
| `model-00091-of-00099.safetensors` | 306 MB (norms, router, bias, shared MLP) |
| `model-00096-of-00099.safetensors` | 4.8 GB (experts, first span) |
| `model-00097-of-00099.safetensors` | 2.4 GB (experts, second span) |

Command shape (non-Xet, single worker, per the trunk download lesson):

```bash
HF_HUB_DISABLE_XET=1 hf download tencent/Hy3 model.safetensors.index.json config.json \
  model-000{18,66,91,96,97}-of-00099.safetensors --local-dir /data/ai-ml/hf-models/hy3-base-mtp-shards --max-workers 1
```

Actual download: 9,654,796,051 bytes (9.65 GB) against the <100 GB budget. Layer-80 payload inside
those shards is 7.505 GB (7.248 GB experts + 257.5 MB glue/attention/router/shared). After the
transcode and byte-level verification passed, all five shards were deleted — 9.65 GB freed;
`config.json` + `model.safetensors.index.json` (4.3 MB) kept in
`/data/ai-ml/hf-models/hy3-base-mtp-shards/` for provenance and re-pull instructions.

### Head architecture (from shard headers + vLLM `hy_v3_mtp.py` + HF `modeling_hy_v3.py`)

HF transformers does not implement the head — `modeling_hy_v3.py` lists
`_keys_to_ignore_on_load_unexpected = [r"model\.layers\.80.*"]` with a "Not supporting
multi-token prediction (MTP) atm" comment. The working reference is vLLM's
`model_executor/models/hy_v3_mtp.py` (`HYV3MultiTokenPredictorLayer`). Structure is
DeepSeek-V3-style NextN, the same family as the Qwen3.5 35B-MoE head bw24 already runs:

1. `inputs_embeds[positions == 0] = 0` (mask the position-0 embedding),
2. `e = enorm(embed(next_tok))`, `h = hnorm(prev_hidden)`,
3. `x = eh_proj(concat[e; h])` with `eh_proj` `[4096, 8192]` (out, in),
4. one full `HYV3DecoderLayer` — the SAME block class as trunk layers: GQA 64/8 heads, hd128,
   QK norm, RoPE, then MoE FFN with its OWN router (sigmoid + expert_bias + route_norm +
   router_scaling_factor), 192 routed experts, one shared expert,
5. `final_layernorm`, then logits through the TRUNK `lm_head` (the checkpoint has no
   `model.layers.80.shared_head.*` and no layer-80 `embed_tokens` — both are shared with the
   trunk; vLLM maps `lm_head.weight -> shared_head.head`).

Chaining for K>1 draft steps: `num_nextn_predict_layers=1`, so the single head is applied
recursively (vLLM indexes `layers[str(80 + spec_step_idx % 1)]` — always layer 80), exactly like
bw24's Qwen path where `mtp_head_forward` is looped with its own scratch KV.

Layer-80 tensor inventory (17 non-expert + 192 x 3 expert keys = 593):

| source (model.layers.80.*) | dtype | shape | mapped bw24 name |
|---|---|---:|---|
| `eh_proj.weight` | BF16 | [4096, 8192] | `blk.80.nextn.eh_proj.weight` |
| `enorm.weight` / `hnorm.weight` | BF16 | [4096] | `blk.80.nextn.{enorm,hnorm}.weight` |
| `final_layernorm.weight` | BF16 | [4096] | `blk.80.nextn.shared_head_norm.weight` |
| `input_layernorm.weight` / `post_attention_layernorm.weight` | BF16 | [4096] | `blk.80.{attn_norm,ffn_norm}.weight` |
| `self_attn.{q,k}_norm.weight` | BF16 | [128] | `blk.80.attn_{q,k}_norm.weight` |
| `self_attn.q_proj.weight` | BF16 | [8192, 4096] | `blk.80.attn_q.weight` |
| `self_attn.{k,v}_proj.weight` | BF16 | [1024, 4096] | `blk.80.attn_{k,v}.weight` |
| `self_attn.o_proj.weight` | BF16 | [4096, 8192] | `blk.80.attn_output.weight` |
| `mlp.router.gate.weight` | BF16 | [192, 4096] | `blk.80.ffn_gate_inp.weight` (-> F32) |
| `mlp.expert_bias` | F32 | [192] | `blk.80.exp_probs_b.bias` |
| `mlp.shared_mlp.{gate,up,down}_proj.weight` | BF16 | [1536,4096]/[4096,1536] | `blk.80.ffn_{gate,up,down}_shexp.weight` |
| `mlp.experts.{0..191}.{gate,up,down}_proj.weight` | BF16 | [1536,4096]/[4096,1536] | stacked `blk.80.ffn_{gate,up,down}_exps.weight` |

Base-vs-REAP naming wrinkles: the base head stores the selection bias as `mlp.expert_bias` (the
REAP trunk artifact used `mlp.router.expert_bias`) and stores experts per-expert under
`mlp.experts.{i}.*` (the MLX artifact used stacked `mlp.switch_mlp.*`). Attention shapes here are
plain BF16 logical shapes — no MLX U32 packing.

### Expert mismatch decision: transcode the full 192-expert head (single variant)

The dossier's 192-vs-96 caveat resolves to full-192, on this evidence:

1. **No kept-index list exists.** The REAP artifact's `config.json` carries only
   `"reap": {"kept_per_layer": 96, "orig_experts": 192}` — no per-layer index lists. The HF repo
   file listing has no sidecar metadata (no saliency dumps, no pruning maps), and the README says
   only "kept 96/192 routed experts/layer by REAP saliency". The kept experts were renumbered
   0..95 per layer with their original identities unrecorded.
2. **Routing is per-layer-independent.** In `hy_v3` every decoder layer owns its router
   (`mlp.router.gate` `[n_experts, 4096]` + `expert_bias`); there is no cross-layer routing state.
   Layer 80 has its own `[192, 4096]` gate and `[192]` F32 bias. A trunk layer's kept-list — even
   if recovered — would be meaningless for the head's expert space.
3. **Layer 80 was never REAP-pruned.** REAP saliency is computed per layer from calibration
   activations; the REAP50 release simply stripped layer 80. No saliency data exists for the head,
   so ANY 96-subset would be arbitrary, not a REAP-consistent choice. This is why only the full-192
   variant was transcoded: the "both variants" branch was for genuine ambiguity, and there is none —
   a subset variant has no principled construction. Bytes were not the constraint (2.1 GB).
4. **The head is draft-only.** Verification runs on the REAP trunk; head expert count affects
   acceptance rate, never output correctness. Keeping all 192 maximizes draft fidelity to the
   base head's training.

Known open risk (rig-side, unverifiable here): the head was trained on BASE-trunk hiddens; it will
consume REAP-trunk hiddens whose distribution has shifted. That is an acceptance-rate question for
the target-rig gates, not a transcode decision. If acceptance disappoints, recovering the trunk
kept-lists by weight-matching REAP experts against base experts (downloading base trunk shards)
is possible future forensics, but it would still not produce a principled head subset.

### Transcode output

New `transcode-mtp` subcommand in `tools/hy3_mlx_to_q4k.py` — the BF16 -> Q4_K sibling of the MLX
path. It reuses `quantize_q4k_rows`, the streaming row-chunk machinery (`--max-work-mb 64`,
per-tensor shard-mmap drop + `malloc_trim`), and the manifest schema. Command:

```bash
python3 tools/hy3_mlx_to_q4k.py transcode-mtp /data/ai-ml/hf-models/hy3-base-mtp-shards \
  /data/ai-ml/hf-models/hy3-reap50-q4k-bw24/mtp --max-work-mb 64
```

| item | value |
|---|---:|
| output directory | `/data/ai-ml/hf-models/hy3-reap50-q4k-bw24/mtp` |
| manifest format | `bw24-hy3-mtp-q4k-repack-v1`, `head_layer: 80` |
| tensor records | 20 (13 Q4_K, 5 BF16 norms, 2 F32) |
| expert slabs | 3 (`blk80-{gate,up,down}-192x…​.q4k`), 192 offsets each |
| expert stride | 3,538,944 bytes/expert/projection — identical to the trunk repack stride |
| payload bytes | 2,113,578,240 (2.11 GB), `du -sh` 2.0G |
| source keys consumed | 593 of 593 (hard-checked; missing keys raise) |
| quality label | unverified — pending target-rig gates |

Dtype policy matches the trunk repack: router gate BF16 -> F32 (loader law: selection tensors stay
F32), `expert_bias` F32 copied byte-exact, 1-D norms copied as BF16 (Hy3RepackSource already
dequantizes 1-D BF16 to F32 at load), all matmul weights Q4_K.

Byte-level tests (`python3 tools/hy3_mlx_to_q4k.py test --real-mtp-dir … --real-mtp-out …`):

- synthetic end-to-end `transcode-mtp` on a fake BF16 layer-80 checkpoint: name mapping to
  `nextn.*`, F32 router/bias exactness, expert-axis-slowest slab stacking, per-expert offset
  byte-equality against direct quantization, Q4_K-class dequant bound;
- real sampled rows: on-disk Q4_K bytes byte-equal to recomputed quantization of the BF16 source
  for `attn_k/attn_output/attn_q/attn_v/ffn_down_shexp/ffn_gate_shexp` (dequant max_abs
  0.000575–0.002438) plus expert 191's gate slab at offset 675,938,304 (max_abs 0.002313).
  These are transcode rounding evidence only, not quality claims.

One dead end recorded: the first real-sample test run crashed with `BufferError: cannot close
exported pointers exist` — safetensors memoryviews must be materialized (`bytes(raw)`) and dropped
before `SafeTensorDir.close()`, the same mmap-lifetime lesson the trunk sampler already encoded
with `del wraw, sraw, braw`.

### What the bw24 spec path needs from this head (vs Qwen MTP in `crates/bw24-engine/src/spec.rs`)

The head maps 1:1 onto the existing `MtpHead` contract (`crates/bw24-engine/src/hybrid.rs:234`):
`enorm`/`hnorm`/`eh_proj` glue, `attn_norm`/`post_attn_norm`, a `Mixer::Full` attention block, an
FFN, `shared_head_norm` = the transcoded `final_layernorm`, `shared_head_head` = None -> reuse
trunk `output` (exactly the existing fallback at `crates/bw24-engine/src/spec.rs:280-285`).
`mtp_head_forward_dev`'s op sequence (embed -> enorm/hnorm -> concat -> eh_proj -> full-attn on
scratch KV -> FFN -> shared-head norm -> lm_head) is the same graph vLLM runs for Hy3. Remaining
rig-side deltas:

1. **MoE FFN in the head with n_expert != trunk.** Qwen35's head FFN is MoE and spec.rs already
   keys head experts under a separate layer index (`u16::MAX`, `spec.rs:270`), so the SLRU/spill
   machinery does not collide with trunk layers. But the Hy3 head has 192 experts while trunk
   layers have 96 — loader/config paths that assume a single global `n_expert` must take the head
   count from the mtp manifest (`n_expert` per expert record), not from the trunk config.
2. **Sigmoid router in the head.** Same sigmoid + expert_bias + route_norm + scale-2.826 selection
   as the trunk Hy3 layers (single forward implementation shared, still GPU-gated). Must stay out
   of the softmax fused-router arms, like the trunk.
3. **Manifest wiring.** The head lives in the SIBLING `mtp/` manifest, not the trunk
   `manifest.json`, so `apply_stripped_mtp_override` (`crates/bw24-gguf/src/source.rs:242`) still
   correctly strips nextn for trunk-only loads. Spec bring-up needs `Hy3RepackSource` (or a
   wrapper) to open both manifests and re-assert `nextn_predict_layers=1`, `n_layer_total=81`,
   exposing `blk.80.*`/`blk.80.nextn.*` alongside trunk names.
4. **h-seed convention.** vLLM feeds the head raw trunk hidden states and applies `hnorm` inside
   (pre-norm seed). bw24's `BW24_SPEC_HPOST` seam (`spec.rs:17`) covers both conventions; which
   seed drafts better on the REAP trunk is a rig-side acceptance experiment.
5. **Position-0 masking.** vLLM zeroes `inputs_embeds` at absolute position 0 before enorm. bw24's
   qwen35 path has no such mask; for Hy3 parity the draft embed at `mtp_pos=0` should be zeroed —
   a one-liner in `mtp_head_forward_dev`, draft-quality-only.
6. **Budget.** 2.11 GB Q4_K head: glue+attention (~150 MB) VRAM-resident beside the trunk body;
   the 679.5 MB x 3 expert slabs join the spill tier with the same stride/offset scheme as trunk
   experts.

Acceptance/quality/perf of spec decode with this head: unverified — pending target-rig gates.
