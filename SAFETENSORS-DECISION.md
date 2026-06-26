# SAFETENSORS-DECISION.md

Decision doc + build spec for adding SafeTensors (HF) weight loading to the bw24 engine.
This is the spec the loader is built against. Everything below is verified against the
on-disk models at `/home/avifenesh/ai-ml/hf-models/{qwen35-9b-hf, qwen35-4b-hf, eagle3-qwen35-9b}`,
the safetensors spec/crate, the local llama.cpp conversion source, and the current engine code.

**Goal (from the user):** "obviously it should support not only gguf." The engine forward/decode
already consumes weights via ggml tensor names + a `ModelConfig`. The loader's job is to produce
those exact ggml-named `GpuTensor`s + a `ModelConfig` from HF safetensors + config.json so the
existing forward runs **unchanged**.

---

## 0. The one invariant that makes this tractable

The engine reads weights only through `GpuTensor::load(e, g, ggml_name)` (model.rs:25-47) and
hyperparams only through `ModelConfig` (config.rs). It never re-reads the file. So the loader is
purely: **(produce ggml-named GpuTensors) + (produce a ModelConfig)**. No forward/kernel change is
required for bf16 qwen3/qwen35; the kernels already get f32 (or resident-quant) `GpuTensor`s.

Two shapes of work:
- **Parse + name-map + transform** safetensors -> ggml-named in-memory weights (this doc, parts 1-3).
- **config.json -> ModelConfig** (part 4), mirroring `ModelConfig::from_gguf`.

---

## 1. Parser design: USE the `safetensors` crate, mmap per shard, route via index.json

### Decision: depend on the `safetensors` crate. Do NOT hand-roll.

| | hand-roll (serde_json + memmap2) | `safetensors` crate |
|---|---|---|
| LOC | ~40 | ~0 (header parse) |
| DOS validation (100MB header cap, no-holes, overlap checks) | must re-derive | free (`Metadata::validate`) |
| zero-copy | yes | yes (`TensorView` borrows mmap) |
| canonical / matches candle, mistral.rs, vLLM-rust | no | yes |

The GGUF hand-roll was justified (custom KV binary + quant-block sizing). SafeTensors is just
`u64 len + JSON header + blob`; the crate is ~400 lines Apache-2.0 and is the reference impl.
Add to `crates/bw24-gguf/Cargo.toml` (or a new `bw24-safetensors` crate — see below):

```toml
safetensors = "0.4"
serde_json  = "1"        # for index.json + config.json
memmap2     = "0.9"      # already present
```

### On-disk format (verified)

```
[ 8 bytes ] N  = header length, unsigned LITTLE-ENDIAN u64
[ N bytes ] JSON UTF-8 header (starts '{' 0x7B, MAY be right-padded with spaces 0x20)
[ rest    ] raw tensor data blob
```
- `data_start = 8 + N`. Tensor `data_offsets:[BEGIN,END]` are **RELATIVE to data_start**, not absolute.
  Bytes = `mmap[8 + N + BEGIN .. 8 + N + END]`. **No alignment padding** anywhere (unlike GGUF, which
  pads data_start to `general.alignment=32`).
- Header JSON: `{"__metadata__":{str->str}, "tensor.name":{"dtype":"BF16","shape":[..],"data_offsets":[B,E]}, ...}`.
  Skip the `__metadata__` key when iterating tensors (it's str->str, not a tensor; absent in qwen35 files).
- VERIFIED qwen35-9b shard-1: first 8 bytes `f0 06 00 00 00 00 00 00` => N=1776; dtype `BF16`;
  e.g. `lm_head.weight {dtype:BF16, shape:[248320,4096], data_offsets:[0,2034237440]}`.

### Sharding via `model.safetensors.index.json` (verified)

```json
{"metadata": {"total_size": 19306216416},
 "weight_map": {"<tensor.name>": "<shard-filename>", ...}}
```
- Detection: if `dir/model.safetensors.index.json` exists -> sharded (route per-tensor through
  `weight_map`). Else open `dir/model.safetensors` directly (single file).
- VERIFIED qwen35-9b: 775 entries, 4 shards. **`weight_map` is NOT contiguous by layer** — layers 14
  and 25-27 are in shard-1 while 3 and 9 are in shard-2. **Always route by name; never compute
  `shard = floor(layer/k)`.**
- **Shard filenames on disk use the dash form** `model.safetensors-00001-of-00004.safetensors`
  (NOT the HF-canonical `model-00001-of-00004.safetensors`). Use the literal string in `weight_map`;
  do not reconstruct it.
- qwen35-4b = 2 shards; eagle3 = single file; no `total_size` use needed.

### Rust struct (mirror `GgufFile`, avoid the self-referential-mmap problem)

The crate's `SafeTensors` view borrows the mmap (self-referential). candle solves this with the
`yoke` crate. Simpler and matching `GgufFile`: parse each shard header into **owned** info structs
up front and index `&mmap[..]` yourself at read time.

```rust
use memmap2::Mmap;
use safetensors::tensor::Dtype;

pub struct StTensorInfo { pub dtype: Dtype, pub shape: Vec<usize>, pub begin: usize, pub end: usize, pub shard: usize }

pub struct SafetensorsModel {
    mmaps: Vec<Mmap>,            // one per shard file
    data_start: Vec<usize>,     // 8 + N per shard
    tensors: std::collections::BTreeMap<String, StTensorInfo>,
}
impl SafetensorsModel {
    pub fn open_dir<P: AsRef<Path>>(dir: P) -> std::io::Result<Self> {
        // 1. index.json present? serde_json -> weight_map -> distinct shard set -> mmap each.
        //    else single model.safetensors.
        // 2. per shard: read 8-byte LE u64 N; SafeTensors::read_metadata(&header_bytes)
        //    (or serde_json on the N bytes); record data_start=8+N and each {dtype,shape,begin,end,shard}.
    }
    pub fn tensor_bytes(&self, name: &str) -> &[u8] {           // zero-copy, mirrors GgufFile::tensor_data
        let t = &self.tensors[name];
        &self.mmaps[t.shard][self.data_start[t.shard] + t.begin .. self.data_start[t.shard] + t.end]
    }
    pub fn has(&self, name: &str) -> bool { self.tensors.contains_key(name) }
}
```
candle reference: `candle-core/src/safetensors.rs` `MmapedSafetensors::multi(&paths)` (mmap per path +
`routing: HashMap<name,idx>` + `yoke::Yoke<SafeTensors_<'static>, Mmap>`). Use the owned-struct form
above to skip `yoke`.

### Where it lives / how it plugs into the engine

`GpuTensor::load` currently takes `&GgufFile` and does `g.find` + `g.tensor_data` + dispatch on
`t.ggml_type`. The clean drop-in is a small **source trait** both backends implement:

```rust
// what the loader needs from a weight source, per ggml name:
pub trait WeightSource {
    fn find_bytes(&self, ggml_name: &str) -> Option<(&[u8], DType, &[u64] /*ne, already ggml order*/)>;
}
```
For safetensors, `find_bytes` does the name-map (part 2), the transforms (part 3), and returns
**ne = reversed(shape)** + a dtype tag the existing dequant path understands. Because transforms
(A_log negate, norm +1, conv1d squeeze, fused split) produce NEW bytes, the safetensors source
materializes those transformed tensors once (owned `Vec<u8>`/`Vec<f32>`) rather than borrowing the
mmap for them. Pure pass-through tensors (q/k/v/o/gate/up/down) stay zero-copy from the mmap.

Recommendation: put it in a new module `crates/bw24-gguf/src/safetensors.rs` (reuses `dequant`,
`GgmlType`, `ModelConfig`) OR a sibling crate `bw24-safetensors`. Module is simpler; the crate name
`bw24-gguf` is then a misnomer but the dependency graph stays flat.

---

## 2. HF name -> engine (ggml) name map

`bid` = layer index N. Ground truth: `/home/avifenesh/projects/llama.cpp/conversion/qwen.py`,
`gguf-py/gguf/tensor_mapping.py`, `gguf-py/gguf/constants.py`. Consumer side: `hybrid.rs` /
`model.rs` (which exact ggml names the forward loads).

### 2.0 Prefix normalization (do this FIRST, before any match)

qwen35-9b/4b are multimodal `Qwen3_5ForConditionalGeneration` checkpoints. VERIFIED on disk:
- layer/embed/norm tensors are prefixed `model.language_model.` (e.g. `model.language_model.layers.14.mlp.down_proj.weight`, `model.language_model.embed_tokens.weight`).
- `lm_head.weight` is **top-level** (no prefix).
- `model.visual.*` (ViT) weights also exist — **skip them all** for text-only inference.

Rule: strip the `language_model.` segment so `model.language_model.X` -> `model.X`, then match the
patterns below. (This mirrors `_Qwen35MtpMixin.filter_tensors` stripping `model.mtp.` -> `mtp.`,
qwen.py:573-574.) Drop anything starting `model.visual.` or `model.*.merger`.

### 2.1 Global (non-layer)

| HF name (after prefix strip)   | ggml name (engine)   | notes |
|--------------------------------|----------------------|-------|
| `model.embed_tokens.weight`    | `token_embd.weight`  | host EmbedHost (model.rs:74) |
| `model.norm.weight`            | `output_norm.weight` | **+1** for qwen35 (part 3) |
| `lm_head.weight`               | `output.weight`      | ABSENT if `tie_word_embeddings` -> engine falls back to `token_embd.weight` (model.rs:111-113 / hybrid.rs:60) |

### 2.2 Dense attention (qwen3 all layers; qwen35 full-attn layers, `(N+1)%full_attention_interval==0`)

| HF name | ggml name | notes |
|---|---|---|
| `model.layers.N.self_attn.q_proj.weight` | `blk.N.attn_q.weight` | dim-reverse only |
| `model.layers.N.self_attn.k_proj.weight` | `blk.N.attn_k.weight` | dim-reverse only |
| `model.layers.N.self_attn.v_proj.weight` | `blk.N.attn_v.weight` | dim-reverse only |
| `model.layers.N.self_attn.o_proj.weight` | `blk.N.attn_output.weight` | dim-reverse only |
| `model.layers.N.self_attn.q_norm.weight` | `blk.N.attn_q_norm.weight` | **+1** for qwen35; raw for qwen3 |
| `model.layers.N.self_attn.k_norm.weight` | `blk.N.attn_k_norm.weight` | **+1** for qwen35; raw for qwen3 |
| `model.layers.N.input_layernorm.weight` | `blk.N.attn_norm.weight` | **+1** for qwen35 |
| `model.layers.N.post_attention_layernorm.weight` | `blk.N.ffn_norm.weight` | **+1** for qwen35. Engine accepts `post_attention_norm.weight` OR `ffn_norm.weight` (hybrid.rs:89-90); emit either. |
| `model.layers.N.mlp.gate_proj.weight` | `blk.N.ffn_gate.weight` | dim-reverse only |
| `model.layers.N.mlp.up_proj.weight` | `blk.N.ffn_up.weight` | dim-reverse only |
| `model.layers.N.mlp.down_proj.weight` | `blk.N.ffn_down.weight` | dim-reverse only |

### 2.3 qwen35 hybrid linear-attention (Gated DeltaNet). HF prefix `model.layers.N.linear_attn.*`

This is family **(a) Qwen3.5** = the SPLIT form (`in_proj_qkv` / `in_proj_z` / `in_proj_a` /
`in_proj_b`), which is exactly what `hybrid.rs` expects. (Family (b) Qwen3Next FUSES into
`in_proj_qkvz` + `in_proj_ba` — see 2.5; not directly consumable without porting the split.)

| HF name | ggml name (engine, hybrid.rs) | transform (part 3) |
|---|---|---|
| `linear_attn.in_proj_qkv.weight` | `blk.N.attn_qkv.weight` | dim-reverse (+ V-reorder rows, see 3.6) |
| `linear_attn.in_proj_z.weight`   | `blk.N.attn_gate.weight` | dim-reverse (+ V-reorder) |
| `linear_attn.in_proj_b.weight`   | `blk.N.ssm_beta.weight`  | dim-reverse (+ V-reorder rows) |
| `linear_attn.in_proj_a.weight`   | `blk.N.ssm_alpha.weight` | dim-reverse (+ V-reorder rows) |
| `linear_attn.conv1d.weight`      | `blk.N.ssm_conv1d.weight`| **squeeze** singleton, then dim-reverse |
| `linear_attn.A_log`              | `blk.N.ssm_a`            | **`-exp(A_log)`** (bare name, no `.weight`) |
| `linear_attn.dt_bias`            | `blk.N.ssm_dt.bias`      | RENAME only (1D bias, no value transform) |
| `linear_attn.norm.weight`        | `blk.N.ssm_norm.weight`  | **NO +1** (the one norm excluded) |
| `linear_attn.out_proj.weight`    | `blk.N.ssm_out.weight`   | dim-reverse (+ V-reorder cols) |

Engine consumer fields: `hybrid.rs:75-83` (`attn_qkv`, `attn_gate`, `ssm_beta`, `ssm_alpha`,
`ssm_a`, `ssm_dt.bias`, `ssm_conv1d.weight`, `ssm_norm.weight`, `ssm_out.weight`).

### 2.4 MoE (qwen3moe / qwen35moe) — not yet wired in engine (TaskCreate #10), map prepared

HF stores per-expert stacked tensors; ggml uses 3D expert tensors. When the MoE forward lands:
| HF | ggml |
|---|---|
| `model.layers.N.mlp.gate.weight` (router) | `blk.N.ffn_gate_inp.weight` |
| `model.layers.N.mlp.experts.{e}.gate_proj.weight` (stack over e) | `blk.N.ffn_gate_exps.weight` (3D) |
| ...`.up_proj` | `blk.N.ffn_up_exps.weight` (3D) |
| ...`.down_proj` | `blk.N.ffn_down_exps.weight` (3D) |
| shared expert `mlp.shared_expert.*` | `blk.N.ffn_*_shexp.weight` |
The HF->ggml stack/transpose for experts is non-trivial (see `base.py` gate/up fuse, qwen.py MoE
classes); defer until the engine MoE path exists. The 9b/4b daily models are DENSE — no MoE tensors.

### 2.5 Qwen3Next FUSED variant (reference; arch `Qwen3NextForCausalLM`, NOT the daily models)

`linear_attn.in_proj_qkvz` and `linear_attn.in_proj_ba` are fused. To feed the engine's SPLIT
expectation, port `Qwen3NextModel.modify_tensors` qwen.py:319-333: `permute(1,0)`, `view(-1,
num_k_heads, head_k+head_k+v_per_k*head_v*2)`, split into q,k,v,z, reshape, `qkv=cat([q,k,v]).permute(1,0)`,
`z=z.permute(1,0)`; and split `in_proj_ba` into beta/alpha. Out of scope until a Qwen3Next checkpoint
is targeted.

### 2.6 EAGLE3 draft (spec-decode) — BESPOKE loader, do NOT route through Hybrid/Dense

VERIFIED single `model.safetensors`, arch `LlamaForCausalLMEagle3`, FLAT config. Names are
`midlayer.*` (one fused decoder layer) + top-level `fc.weight`/`norm.weight`/`lm_head.weight` +
two vocab-remap tensors. It needs its own struct, not `blk.N.*`.

| HF name | shape (verified) | dtype | role |
|---|---|---|---|
| `fc.weight` | [4096,12288] | BF16 | fuses 3 aux hidden states (3*4096) -> 4096 |
| `midlayer.input_layernorm.weight` | [4096] | BF16 | |
| `midlayer.hidden_norm.weight` | [4096] | BF16 | |
| `midlayer.post_attention_layernorm.weight` | [4096] | BF16 | |
| `midlayer.self_attn.q_proj.weight` | [4096,8192] | BF16 | in=8192 = 2*hidden (prev-embed ++ hidden) |
| `midlayer.self_attn.k_proj.weight` | [1024,8192] | BF16 | |
| `midlayer.self_attn.v_proj.weight` | [1024,8192] | BF16 | |
| `midlayer.self_attn.o_proj.weight` | [4096,4096] | BF16 | |
| `midlayer.mlp.{gate,up}_proj.weight` | [12288,4096] | BF16 | |
| `midlayer.mlp.down_proj.weight` | [4096,12288] | BF16 | |
| `norm.weight` | [4096] | BF16 | |
| `lm_head.weight` | [32000,4096] | BF16 | DRAFT vocab (32000) |
| `d2t` | [32000] | **I64** | draft->target token id map |
| `t2d` | [248320] | **BOOL** | target->draft membership mask |

Loader must handle **I64 (8B) and BOOL (8B/1B)** dtypes for `d2t`/`t2d`, not just float weights.
EAGLE3 borrows trunk hparams via `--target-model-dir` in llama.cpp (convert_hf_to_gguf.py:156-163);
top-level `rope_theta` is null on disk — take it from the target (qwen35 = 1e7).

---

## 3. Per-tensor TRANSFORMS (HF vs GGUF). The engine consumes GGUF-baked values, so replicate ALL of these.

### 3.1 Dim-reverse for EVERY 2D weight (NO byte transpose) — load-bearing

HF stores nn.Linear as `shape=[out_features, in_features]` row-major (VERIFIED: `gate_proj=[12288,4096]`,
`down_proj=[4096,12288]`). GGUF reverses the dim LABELS only and keeps bytes identical
(`gguf_writer.py:265-268` writes `ti.shape[n_dims-1-j]`). The engine: `in_features()=ne[0]`,
`out_features()=ne[1]` (model.rs:21-22), and `GpuTensor::load` computes `out_f=ne[1]`,
`row_bytes=raw.len()/out_f` (model.rs:36-37).

**Rule:** `ne = reversed(hf_shape)` => for 2D `ne = [shape[1], shape[0]] = [in, out]`. Copy bytes
**verbatim** (dequant only). A real data transpose would be a correctness bug — `y=W·x` is the same
byte layout because both are row-major `W[out][in]`. Setting ne wrong silently corrupts `row_bytes`.

### 3.2 `ssm_a = -exp(A_log)` (CRITICAL)

HF `linear_attn.A_log` (shape [32], F32 on disk) is raw. GGUF/engine value is `-exp(A_log)`
elementwise (qwen.py:296-297). Compute it in the loader; emit under bare name `ssm_a` (no suffix).

### 3.3 Norm `+1` for qwen35 (CRITICAL) — NOT for plain qwen3

qwen35/qwen3next add `1.0` to **every** tensor whose name ends `norm.weight`, **EXCEPT**
`linear_attn.norm.weight` (qwen.py:302-303). The engine's `rms_norm_f32` kernel does
`dr[i]=xr[i]*scale*w[i]` with no built-in +1 (kernels.cu:28), so the +1 MUST be pre-baked.

Apply +1 to: `input_layernorm`, `post_attention_layernorm`, `self_attn.q_norm`, `self_attn.k_norm`,
`model.norm` (-> output_norm). Do NOT apply to `linear_attn.norm` (-> ssm_norm, stays raw).
For **plain qwen3** (non-hybrid) there is NO +1 — load all norms raw.

### 3.4 conv1d squeeze

HF `linear_attn.conv1d.weight` is `[conv_dim, 1, d_conv] = [8192,1,4]` (depthwise, groups=conv_dim).
qwen.py:300-301 does `.squeeze()` -> `[conv_dim, d_conv] = [8192,4]`, THEN dim-reverse -> ne `[4,8192]`.
Forgetting the squeeze leaves a phantom singleton dim and a wrong `ne`/row layout.

### 3.5 `dt_bias` rename (no value transform)

HF `linear_attn.dt_bias` [32] -> qwen.py renames to `dt_proj.bias` -> tensor_mapping SSM_DT +
`.bias` suffix -> engine loads literal `ssm_dt.bias` (hybrid.rs:80). 1D bias; copy bytes as-is.

### 3.6 V-head reorder (qwen35 only, when `linear_num_key_heads != linear_num_value_heads`)

VERIFIED 9b: num_k=16, num_v=32 -> they DIFFER, so this applies. llama.cpp's
`_LinearAttentionVReorderBase` (qwen.py:366-376, 469-518) reorders V heads grouped->tiled on:
`in_proj_qkv` (V rows only), `in_proj_z`, `in_proj_a`/`in_proj_b` (rows), `conv1d` (V channel portion),
`out_proj` (columns). This is a llama.cpp optimization (PR 19468) for `ggml_repeat` tiled broadcast.

**Open verification item:** the engine's deltanet (`decode.rs`/`hybrid_forward.rs`) was validated
against GGUF produced by THIS local llama.cpp (Task #9 fixed the argmax). If so, the engine expects
the **TILED (reordered)** layout and the loader MUST replicate `_reorder_v_heads`. Confirm with an
on-device argmax diff: load 9b via safetensors with reorder ON vs OFF, compare first-token argmax to
the validated GGUF run. Wrong choice manifests ONLY when num_v != num_k (so it's invisible on any
model where they match). Resolve this BEFORE declaring 9b correct.

### 3.7 No-op transforms (copy bytes verbatim, only dim-reverse): q/k/v/o, gate/up/down, embed, lm_head.

### 3.8 fused split/concat summary

- **gate_up fuse:** llama.cpp base sometimes fuses gate+up (`base.py` modify_tensors). The daily
  qwen35 stores SPLIT `mlp.gate_proj`/`mlp.up_proj` (VERIFIED) -> map separately, no split needed.
- **qkvz/ba split:** only for Qwen3Next fused variant (2.5) — not the daily models.

---

## 4. config.json -> ModelConfig (mirror `ModelConfig::from_gguf`)

### 4.0 Nesting (CRITICAL)

qwen35 (and gemma-4) are multimodal wrappers: ALL text hyperparams live under
`config["text_config"]`, NOT top-level. EAGLE3 is FLAT. Rule: `let tc = cfg.get("text_config", cfg)`.
Read `tie_word_embeddings` from BOTH top-level and tc and OR them (VERIFIED: 9b sets it only
top-level=false; 4b sets it in both=true).

Arch dispatch from `config["architectures"][0]`:
`Qwen3ForCausalLM`->Qwen3; `Qwen3_5ForConditionalGeneration`/`Qwen3_5ForCausalLM`->Qwen35;
`Qwen3MoeForCausalLM`->Qwen3Moe; `Qwen3_5MoeForConditionalGeneration`/`..ForCausalLM`->Qwen35Moe;
`Qwen3NextForCausalLM`->(maps to Qwen35 split path only after porting 2.5); `LlamaForCausalLMEagle3`->bespoke.

### 4.1 Field map (HF text_config key -> ModelConfig field). VERIFIED values for 9b / 4b.

| ModelConfig field | HF text_config key | derivation | 9b | 4b |
|---|---|---|---|---|
| `n_layer` | `num_hidden_layers` | direct | 32 | 32 |
| `n_embd` | `hidden_size` | direct | 4096 | 2560 |
| `n_head` | `num_attention_heads` | direct | 16 | 16 |
| `n_head_kv` | `num_key_value_heads` | direct | 4 | 4 |
| `head_dim_k`/`head_dim_v` | `head_dim` | **EXPLICIT — read, never derive** | 256 | 256 |
| `n_ff` | `intermediate_size` | direct | 12288 | 9216 |
| `n_vocab` | `vocab_size` | direct | 248320 | 248320 |
| `context_length` | `max_position_embeddings` | direct | | |
| `rms_eps` | `rms_norm_eps` | direct | 1e-6 | 1e-6 |
| `rope_freq_base` | `rope_parameters.rope_theta` | direct | **1e7** | 1e7 |
| `rope_dim_count` | — | `round(head_dim * rope_parameters.partial_rotary_factor)` = `256*0.25` | **64** | 64 |
| `rope_sections` | `rope_parameters.mrope_section` | `[11,11,10]` (interleaved MRoPE) | | |
| `full_attention_interval` | `full_attention_interval` | direct (0 if non-hybrid) | 4 | 4 |
| `nextn_predict_layers` | `mtp_num_hidden_layers` | direct | 1 | 1 |
| `n_layer_total` | — | `n_layer + nextn_predict_layers` | 33 | 33 |

`head_dim` MUST be read explicitly: `n_embd/n_head` = 256 for 9b only by coincidence; for 4b it's
`2560/16=160` which is WRONG. (`q_proj` out=4096 ... note attn_output_gate doubling lives in the
hybrid linear-attn path, not the full-attn q_proj here.)

### 4.2 SsmConfig (qwen35) — fields are OVERLOADED ggml aliases, NOT literal Mamba names

The engine's kernels (`hybrid_forward.rs:126-133`) read these aliases; fill them or break GDN.
HF GDN dims (VERIFIED 9b): key_dim = num_k*head_k = 16*128 = 2048; value_dim = num_v*head_v =
32*128 = 4096; conv_dim = 2*key_dim + value_dim = 8192.

| SsmConfig field (engine var) | HF text_config source | value (9b) |
|---|---|---|
| `state_size` (`d_state` = head_k=head_v) | `linear_key_head_dim` | 128 |
| `group_count` (`num_k`) | `linear_num_key_heads` | 16 |
| `time_step_rank` (`num_v`) | `linear_num_value_heads` | 32 |
| `conv_kernel` (`d_conv`) | `linear_conv_kernel_dim` | 4 |
| `inner_size` (conv_dim) | `2*key_dim + value_dim` = `2*(num_k*head_k) + num_v*head_v` | 8192 |

Disk shapes confirm: `in_proj_qkv [8192,4096]`, `in_proj_z [4096,4096]`, `in_proj_a/b [32,4096]`,
`conv1d [8192,1,4]`, `A_log [32]`, `dt_bias [32]`, `linear_attn.norm [128]`, `out_proj [4096,4096]`.

### 4.3 MoeConfig (when MoE path lands)

`expert_count <- num_experts`; `expert_used_count <- num_experts_per_tok`;
`expert_ff_length <- moe_intermediate_size`. Routing: MoE layer iff `idx not in mlp_only_layers &&
num_experts>0 && (idx+1)%decoder_sparse_step==0`. Daily 9b/4b are dense (no MoE keys).

---

## 5. Dtype handling

| safetensors dtype | engine handling | path |
|---|---|---|
| BF16 (all weights, all 4 models) | `dequant::bf16_to_f32` -> f32 -> `htod` | EXISTING Float arm (model.rs:40-44); ZERO new dequant math |
| F16 | `dequant::fp16_to_f32` | EXISTING |
| F32 (some `A_log`, `linear_attn.norm` are f32 in HF) | direct LE bytes | EXISTING |
| I64 (`d2t`) | new: read LE i64 (EAGLE3 only) | small helper |
| BOOL (`t2d`) | new: 1 byte/elem (EAGLE3 only) | small helper |
| F8_E4M3 / NVFP4 | NOT supported by current dequant oracle | OUT OF SCOPE (daily models are bf16) |

Implementation: map safetensors `Dtype::{F16,BF16,F32}` -> existing `GgmlType::{F16,BF16,F32}` and
route through `dequant::dequantize` unchanged. `EmbedHost` (model.rs:74) reads `token_embd` bytes +
a `GgmlType` tag and gathers per-row — feed it the bf16 bytes + `GgmlType::BF16` and the existing
per-row gather works (block_size=1, type_size=2).

For the resident-quant trait, safetensors has NO Q8_0/Q4_K/Q6_K, so every weight currently takes the
Float (f32) arm — which is the memory problem in part 6.

---

## 6. Memory caveat + what this unlocks NOW

**The caveat:** the Float arm dequants bf16 -> f32 in VRAM (model.rs:43), doubling memory vs disk.
The resident-Quant path (Task #8) that fixed the earlier OOM only triggers for Q8_0/Q4_K/Q6_K, which
safetensors never has. So per model, f32-resident VRAM ~= 2x the bf16 file size.

| model | disk (bf16) | f32-resident | fits 32GB 5090 NOW? |
|---|---|---|---|
| **qwen35-4b-hf** | ~8GB | ~16GB | YES |
| **qwen35-9b-hf** | ~19.3GB (total_size 19306216416) | ~38GB | **NO as pure f32** — needs bf16-resident path |
| **EAGLE3 draft** | ~0.6GB | ~1.2GB | YES (tiny; used alongside trunk) |
| gemma-4 / 27B-class | larger | >50GB | NO until quantize/bf16-on-load |

**Unlocked immediately (f32 path, no new GPU work):**
- **qwen35-4b-hf** — full hybrid forward, fits comfortably.
- **EAGLE3-qwen35-9b draft** — bespoke loader; tiny; the spec-decode draft head.

**Needs a bf16-resident `GpuTensor` variant first (recommended next GPU task):**
- **qwen35-9b-hf** — 38GB f32 OOMs a 32GB 5090. Add a `GpuTensor::Bf16 { bytes: CudaSlice<u8>, ne }`
  that htod's raw bf16 (19.3GB, fits) + a bf16-aware GEMM/upconvert (mirror the resident-Quant pattern
  from Task #8). This keeps VRAM ~= disk size and unblocks the 9b — the flagship daily target.
- 27B/35B-MoE need quantize-on-load (bf16 -> Q8_0/Q4_K at load) on top of that.

So safetensors loading delivers **4b + EAGLE3 today** with the existing f32 path, and **9b** as soon
as the bf16-resident `GpuTensor` lands (small, well-scoped — same shape as the existing Quant arm).

---

## 7. Build order (decisive)

1. `SafetensorsModel::open_dir` (mmap + index.json routing + header parse, owned StTensorInfo). Verify
   against 9b: 775 tensors across 4 shards, `tensor_bytes("lm_head.weight").len() == 248320*4096*2`.
2. config.json -> ModelConfig (part 4); assert 9b/4b values in the table match.
3. Name-map + transforms (parts 2-3) behind the `WeightSource` trait; emit ggml-named transformed
   tensors. Unit-check: `ssm_a == -exp(A_log)`, norms +1 (except ssm_norm), conv1d squeezed,
   ne reversed.
4. Wire `HybridModel::load` / `Model::load_dense` to accept a `&dyn WeightSource`. Run **qwen35-4b-hf**
   end-to-end; argmax-diff vs its GGUF twin.
5. Resolve the V-reorder open item (3.6) on 4b/9b via argmax diff vs validated GGUF.
6. Bespoke EAGLE3 loader (2.6) — `midlayer.*` + I64/BOOL.
7. (separate GPU task) bf16-resident `GpuTensor::Bf16` -> unblock qwen35-9b-hf.

## Exact references

- safetensors spec: huggingface/safetensors README "Format"; crate `safetensors/src/tensor.rs:390-425`
  (`read_metadata`, MAX_HEADER_SIZE=100MB, no-holes check), `:811-892` (Dtype enum + bitsize).
- candle loader: `candle-core/src/safetensors.rs` (`MmapedSafetensors::multi`, routing HashMap,
  `yoke::Yoke<SafeTensors_<'static>, Mmap>`, `convert_slice`; `save_single_tensor` test shows literal bytes).
- llama.cpp HF->ggml: `conversion/qwen.py` (A_log negate :296-297, dt_bias rename :298-299,
  conv1d squeeze :300-301, norm +1 :302-303, Qwen3Next qkvz split :305-333,
  `_LinearAttentionVReorderBase` :366-376/:469-518, `_Qwen35MtpMixin` :537-617,
  `Qwen3_5TextModel` :620-622); `conversion/base.py` (no-transpose, gate/up fuse :615-640);
  `gguf-py/gguf/tensor_mapping.py` (ATTN_QKV<-in_proj_qkv :251, ATTN_GATE<-in_proj_z, SSM_A<-A_log,
  SSM_ALPHA<-in_proj_a :885, SSM_BETA<-in_proj_b :909, SSM_CONV1D, SSM_OUT, SSM_NORM);
  `gguf-py/gguf/constants.py` TENSOR_NAMES :1099-1200; `gguf-py/gguf/gguf_writer.py:265-268`
  (dim-reverse on write = proof no byte transpose); `convert_hf_to_gguf.py` (298-line CLI shim:
  `--mtp/--no-mtp` :120-127, `--target-model-dir` EAGLE3 :156-163, arch dispatch :238-245).
- bw24 consumer: `crates/bw24-engine/src/model.rs:14-57` (GpuTensor + load, in=ne[0]/out=ne[1],
  Q8_0/Q4_K/Q6_K only resident; else dequant->f32), `:68-90` (EmbedHost per-row gather);
  `crates/bw24-engine/src/hybrid.rs:53-99` (all ggml names the forward loads);
  `crates/bw24-gguf/src/config.rs:87-171` (ModelConfig::from_gguf + layer_kind);
  `crates/bw24-gguf/src/dequant.rs:8-61` (fp16/bf16 -> f32, dispatch);
  `crates/bw24-gguf/src/lib.rs:122-129` (TensorInfo.ne, ne[0] fastest); kernels.cu rms_norm (no +1).
- on-disk (verified this session): qwen35-9b-hf index (775 tensors, 4 dash-named shards,
  total_size 19306216416, non-contiguous-by-layer; `model.language_model.*` prefix + top-level
  `lm_head.weight`), 9b/4b config.json (head_dim 256 explicit, rope_theta 1e7, partial_rotary 0.25,
  mrope [11,11,10], full_attention_interval 4, linear_* dims, mtp 1; 4b tie=true both levels,
  9b tie=false top-level only), eagle3 single file (15 tensors: midlayer.* BF16 + d2t I64[32000] +
  t2d BOOL[248320], lm_head [32000,4096] draft vocab, fc [4096,12288], q_proj in=8192).
```
