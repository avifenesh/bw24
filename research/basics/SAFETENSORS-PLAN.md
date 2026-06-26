# Safetensors Loader — Build Plan

All citations verified. The findings claimed `dequant.rs:40-50` for the dtype dispatch — confirmed at `dequant.rs:41-53`. Now I have enough grounded detail to produce the plan.

# bw24 safetensors loader — concrete implementation plan

Verified against the live tree. Every bw24 `file:line` below was read this session.

## 0. Guiding decisions (resolves the findings' open questions)

- **`TensorSource` trait lives in `bw24-gguf`** (new `src/source.rs`), not `bw24-engine`. `bw24-engine` already depends on `bw24-gguf` (`crates/bw24-engine/Cargo.toml:7`), and the trait returns `GgmlType` + `&[u8]` which are bw24-gguf types. This keeps `bw24-safetensors` depending only on `bw24-gguf`, and `bw24-engine` stays compute-only. No circular dep.
- **F16/BF16 reuse the existing CPU dequant path.** `dequant::dequantize` already handles `F16`/`BF16`/`F32` (`crates/bw24-gguf/src/dequant.rs:44-53`). A safetensors F16/BF16 tensor maps to `GgmlType::F16`/`BF16` and flows through the *unchanged* `None =>` arm in `GpuTensor::load` (`crates/bw24-engine/src/model.rs:56-61`). Zero new dequant code for the common case.
- **FP8 is deferred with an explicit error in v1** (panic on `F8_E4M3`/`F8_E5M2` in the dtype map). No `GgmlType::F8_*` variant added yet — the enum (`crates/bw24-gguf/src/lib.rs:27-34`) stays untouched.
- **Minimal-diff strategy:** existing GGUF entry points (`Model::load_dense`, `HybridModel::load`, `GpuTensor::load`, `EmbedHost::from_gguf`) keep their exact signatures and behavior by becoming thin wrappers over new `*_from_source` methods. All current GGUF tests pass unchanged.

---

## (A) The mmap safetensors reader — `crates/bw24-safetensors/src/lib.rs`

New crate, parallel to `bw24-gguf`. `Cargo.toml` deps: `memmap2 = "0.9"` (same as `crates/bw24-gguf/Cargo.toml`), `serde`/`serde_json` (for header + config.json), `bw24-gguf = { path = "../bw24-gguf" }` (for `GgmlType`, the `TensorSource` trait).

**Layout** (per findings, safetensors-0.7.0 `tensor.rs:353-417`): `[u64 LE header_len][header_len bytes UTF-8 JSON][raw tensor buffer]`. `data_offsets` are byte ranges into the buffer that starts at `8 + header_len`.

```rust
// crates/bw24-safetensors/src/lib.rs
use memmap2::Mmap;
use std::collections::HashMap;

const N_LEN: usize = 8;                       // size_of::<u64>(), tensor.rs:10
const MAX_HEADER: usize = 100_000_000;        // DoS guard, tensor.rs:10/325-366

#[derive(serde::Deserialize)]
pub struct StInfo {                            // tensor.rs TensorInfo 767-774
    pub dtype: String,                         // "F16","BF16","F32","F8_E4M3",...
    pub shape: Vec<usize>,                     // row-major, outer..inner
    pub data_offsets: [usize; 2],              // [start,end) into post-header buffer
}

pub struct StShard {
    mmap: Mmap,
    data_base: usize,                          // 8 + header_len
    infos: HashMap<String, StInfo>,            // "__metadata__" stripped out
}

impl StShard {
    pub fn open<P: AsRef<std::path::Path>>(p: P) -> std::io::Result<Self> {
        let f = std::fs::File::open(p)?;
        let mmap = unsafe { Mmap::map(&f)? };
        let hlen = u64::from_le_bytes(mmap[..N_LEN].try_into().unwrap()) as usize;
        assert!(hlen <= MAX_HEADER && N_LEN + hlen <= mmap.len(), "bad/oversized st header");
        let json: serde_json::Map<String, serde_json::Value> =
            serde_json::from_slice(&mmap[N_LEN..N_LEN + hlen]).expect("st header json");
        let mut infos = HashMap::new();
        for (k, v) in json {
            if k == "__metadata__" { continue; }            // free-form KV, tensor.rs:507-514
            infos.insert(k, serde_json::from_value(v).expect("st TensorInfo"));
        }
        Ok(Self { mmap, data_base: N_LEN + hlen, infos })
    }
    /// Zero-copy bytes for a tensor (mirrors GgufFile::tensor_data, lib.rs:239-242).
    pub fn raw(&self, name: &str) -> Option<(&StInfo, &[u8])> {
        let i = self.infos.get(name)?;
        let s = self.data_base + i.data_offsets[0];
        let e = self.data_base + i.data_offsets[1];
        Some((i, &self.mmap[s..e]))
    }
    pub fn names(&self) -> impl Iterator<Item = &String> { self.infos.keys() }
}
```

**Multi-shard** (`crates/bw24-safetensors/src/lib.rs`, same file): from the findings, `model.safetensors.index.json` = `{ "metadata": {"total_size": N}, "weight_map": { tensor_name -> "model-0000X-of-0000N.safetensors" } }` (verified shape: `models--Qwen--Qwen3-1.7B/.../model.safetensors.index.json`).

```rust
pub struct StModel {                          // holds all shards mmap'd
    shards: Vec<StShard>,
    map: HashMap<String, usize>,              // tensor_name -> shard index
}
impl StModel {
    pub fn open(dir: &std::path::Path) -> std::io::Result<Self> {
        let idx = dir.join("model.safetensors.index.json");
        if idx.exists() {
            #[derive(serde::Deserialize)] struct Index { weight_map: HashMap<String,String> }
            let i: Index = serde_json::from_slice(&std::fs::read(idx)?)?;
            let mut files: Vec<String> = i.weight_map.values().cloned().collect();
            files.sort(); files.dedup();
            let pos: HashMap<&String, usize> = files.iter().enumerate().map(|(n,f)|(f,n)).collect();
            let shards = files.iter().map(|f| StShard::open(dir.join(f))).collect::<Result<_,_>>()?;
            let map = i.weight_map.iter().map(|(t,f)| (t.clone(), pos[f])).collect();
            Ok(Self { shards, map })
        } else {                              // single-file model.safetensors
            let sh = StShard::open(dir.join("model.safetensors"))?;
            let map = sh.names().map(|n| (n.clone(), 0)).collect();
            Ok(Self { shards: vec![sh], map })
        }
    }
    pub fn raw(&self, name: &str) -> Option<(&StInfo, &[u8])> {
        self.shards[*self.map.get(name)?].raw(name)
    }
}
```

**Dtype map** — `crates/bw24-safetensors/src/dtype.rs`. Maps the canonical safetensors strings (findings: `tensor.rs:863-888`) to bw24's `GgmlType` (`crates/bw24-gguf/src/lib.rs:27-34`):

```rust
pub fn st_dtype_to_ggml(s: &str) -> bw24_gguf::GgmlType {
    use bw24_gguf::GgmlType::*;
    match s {
        "F32"  => F32,
        "F16"  => F16,
        "BF16" => BF16,
        "F64"  => F64,
        "I8"=>I8, "I16"=>I16, "I32"=>I32, "I64"=>I64,
        // v1: FP8 deferred — explicit failure, NOT silent. (findings open-question resolution)
        "F8_E4M3" | "F8_E5M2" | "F8_E8M0" =>
            panic!("FP8 ({s}) safetensors not yet supported; use the GGUF twin or F16/BF16"),
        other => panic!("unsupported safetensors dtype {other}"),
    }
}
```

Shape note: bw24's `ne` is **inner-fastest** (`ne[0]` fastest — `crates/bw24-gguf/src/lib.rs:125`), safetensors `shape` is **outer..inner** (row-major). So `ne = shape.iter().rev().cloned().collect()`. For a `[out, in]` weight, safetensors `shape=[out,in]` → `ne=[in,out]`, which matches what `GpuTensor::in_features=ne[0]`/`out_features=ne[1]` expect (`crates/bw24-engine/src/model.rs:21-22`). This is the single most bug-prone line — assert it in the validation harness (E).

---

## (B) `config.json` → `ModelConfig` — `crates/bw24-safetensors/src/config.rs`

HF `config.json` has no GGUF-style `{arch}.` prefixed keys, so we parse it with serde and build the *existing* `ModelConfig` struct (`crates/bw24-gguf/src/config.rs:62-86`) field-for-field. `Arch::parse` already exists but is private (`crates/bw24-gguf/src/config.rs:18-27`) — make it `pub(crate)`-accessible via a new `pub fn from_hf` constructor on `ModelConfig`, or expose `Arch::parse` as `pub`.

```rust
// crates/bw24-safetensors/src/config.rs
#[derive(serde::Deserialize)]
pub struct HfConfig {
    pub model_type: String,                    // "qwen3","qwen2","llama","qwen3_moe"...
    pub num_hidden_layers: u32,                // -> n_layer
    pub hidden_size: u32,                       // -> n_embd
    pub num_attention_heads: u32,               // -> n_head
    #[serde(default)] pub num_key_value_heads: Option<u32>, // -> n_head_kv
    #[serde(default)] pub head_dim: Option<u32>,            // -> head_dim_k/v
    pub intermediate_size: u32,                 // -> n_ff
    pub vocab_size: u32,                        // -> n_vocab
    #[serde(default)] pub max_position_embeddings: u32,     // -> context_length
    #[serde(default = "default_eps")] pub rms_norm_eps: f32,
    #[serde(default = "default_theta")] pub rope_theta: f32,
    // MoE (qwen3_moe): num_experts, num_experts_per_tok, moe_intermediate_size,
    //                  shared_expert_intermediate_size
    #[serde(default)] pub num_experts: Option<u32>,
    #[serde(default)] pub num_experts_per_tok: Option<u32>,
    #[serde(default)] pub moe_intermediate_size: Option<u32>,
    #[serde(default)] pub shared_expert_intermediate_size: Option<u32>,
}
```

`model_type` strings differ from GGUF arch strings. HF `"qwen3"`→GGUF `"qwen3"` (lucky), but HF `"qwen3_moe"`→GGUF `"qwen3moe"`, HF `"llama"`→`"llama"`. Add a small normalizer before `Arch::parse` (`crates/bw24-gguf/src/config.rs:18-27`):

```rust
fn hf_model_type_to_arch(mt: &str) -> bw24_gguf::config::Arch {
    let ggml = match mt {
        "qwen3_moe" => "qwen3moe",
        "qwen3_next" | "qwen3.5" => "qwen35",   // hybrid family
        other => other,                          // qwen3, llama pass through
    };
    bw24_gguf::config::Arch::parse(ggml)         // make pub
}
```

`ModelConfig::from_hf(cfg: &HfConfig) -> ModelConfig` fills the struct. Lenient defaults (resolving the strict-vs-lenient open question): `head_dim_k = head_dim.unwrap_or(hidden_size / num_attention_heads)` (matches the GGUF fallback at `crates/bw24-gguf/src/config.rs:96-99`); `n_head_kv = num_key_value_heads.unwrap_or(num_attention_heads)` (matches `crates/bw24-gguf/src/config.rs:135`); `rope_sections = vec![]`, `full_attention_interval = 0`, `nextn_predict_layers = 0` unless the hybrid/MoE fields are present. MoE config maps `num_experts→expert_count`, `num_experts_per_tok→expert_used_count`, `moe_intermediate_size→expert_ff_length`, `shared_expert_intermediate_size→expert_shared_ff_length` (the `MoeConfig` fields at `crates/bw24-gguf/src/config.rs:54-59`).

---

## (C) HF → ggml tensor-name mapping — `crates/bw24-safetensors/src/mapping.rs`

The whole abstraction's payoff: the engine only ever asks for **ggml names** (`"blk.{il}.attn_q.weight"`, hardcoded at `crates/bw24-engine/src/model.rs:134` and `crates/bw24-engine/src/hybrid.rs:86`). The mapper translates a requested **ggml name** → the **HF name** to look up in the safetensors file. Direction is ggml→HF (we resolve on demand, no full-table build needed).

```rust
// crates/bw24-safetensors/src/mapping.rs
// Returns the HF tensor name for a requested ggml name, or None if unmapped.
pub fn ggml_to_hf(ggml: &str, arch: &bw24_gguf::config::Arch) -> Option<String> {
    // top-level (arch-independent for Llama/Qwen dense+MoE)
    match ggml {
        "token_embd.weight"   => return Some("model.embed_tokens.weight".into()),
        "output_norm.weight"  => return Some("model.norm.weight".into()),
        "output.weight"       => return Some("lm_head.weight".into()),
        _ => {}
    }
    // per-layer: "blk.{il}.{suffix}"
    let rest = ggml.strip_prefix("blk.")?;
    let (il, suffix) = rest.split_once('.')?;
    let p = format!("model.layers.{il}");
    let hf_suffix = match suffix {
        "attn_norm.weight"        => "input_layernorm.weight",
        "attn_q.weight"           => "self_attn.q_proj.weight",
        "attn_k.weight"           => "self_attn.k_proj.weight",
        "attn_v.weight"           => "self_attn.v_proj.weight",
        "attn_output.weight"      => "self_attn.o_proj.weight",
        "attn_q_norm.weight"      => "self_attn.q_norm.weight",   // qwen3 only
        "attn_k_norm.weight"      => "self_attn.k_norm.weight",
        "ffn_norm.weight"         => "post_attention_layernorm.weight",
        "ffn_gate.weight"         => "mlp.gate_proj.weight",      // dense SwiGLU
        "ffn_up.weight"           => "mlp.up_proj.weight",
        "ffn_down.weight"         => "mlp.down_proj.weight",
        // MoE router + shared expert (qwen3_moe)
        "ffn_gate_inp.weight"     => "mlp.gate.weight",
        "ffn_gate_shexp.weight"   => "mlp.shared_expert.gate_proj.weight",
        "ffn_up_shexp.weight"     => "mlp.shared_expert.up_proj.weight",
        "ffn_down_shexp.weight"   => "mlp.shared_expert.down_proj.weight",
        "ffn_gate_inp_shexp.weight" => "mlp.shared_expert_gate.weight",
        _ => return None,
    };
    Some(format!("{p}.{hf_suffix}"))
}
```

**MoE expert tensors are the hard case.** GGUF stacks all 256 experts into one 3D tensor `ffn_gate_exps.weight` `ne=[in,out,256]` (consumed by `HostExps::load`, `crates/bw24-engine/src/model.rs:186-213` and `crates/bw24-engine/src/hybrid.rs:112-114`). HF stores them as **256 separate 2D tensors** `model.layers.{il}.mlp.experts.{e}.gate_proj.weight`. So `HostExps` cannot use a single `raw()` for safetensors — it needs a per-expert gather + concat into one contiguous `bytes: Vec<u8>` buffer (the struct already owns a `Vec<u8>`, `crates/bw24-engine/src/model.rs:175`). This is the one place the `find(name)->bytes` abstraction is insufficient; add a `TensorSource::find_stacked_experts(layer, proj, n_expert) -> (GgmlType, in_f, out_f, Vec<u8>)` method whose GGUF impl just returns the existing 3D blob and whose safetensors impl concatenates the 256 rows in expert order. For v1, scope expert mapping to dense + shared-expert only and gate MoE-expert safetensors behind an explicit "not yet" error; dense Qwen3/Llama validate first (E).

---

## (D) Source-agnostic abstraction — `crates/bw24-gguf/src/source.rs` + minimal engine changes

**New trait + GGUF impl** in `crates/bw24-gguf/src/source.rs` (add `pub mod source;` to `crates/bw24-gguf/src/lib.rs:16-17` alongside `dequant`/`config`):

```rust
// crates/bw24-gguf/src/source.rs
use crate::{GgufFile, GgmlType, config::ModelConfig};

pub struct TensorView<'a> { pub bytes: &'a [u8], pub ggml_type: GgmlType, pub ne: Vec<u64> }

pub trait TensorSource {
    fn config(&self) -> ModelConfig;
    fn find(&self, ggml_name: &str) -> Option<TensorView<'_>>;
    fn has(&self, ggml_name: &str) -> bool { self.find(ggml_name).is_some() }
}

pub struct GgufSource<'g>(pub &'g GgufFile);
impl<'g> TensorSource for GgufSource<'g> {
    fn config(&self) -> ModelConfig { ModelConfig::from_gguf(self.0) }
    fn find(&self, name: &str) -> Option<TensorView<'_>> {
        let t = self.0.find(name)?;                          // lib.rs:244-246
        Some(TensorView { bytes: self.0.tensor_data(t),      // lib.rs:239-242
                          ggml_type: t.ggml_type, ne: t.ne.clone() })
    }
}
```

`SafetensorsSource` (`crates/bw24-safetensors/src/source.rs`) wraps `StModel` + `HfConfig` + the arch, and in `find` calls `mapping::ggml_to_hf(name, &arch)` → `StModel::raw(hf_name)` → builds `TensorView` with `dtype::st_dtype_to_ggml` and reversed shape.

**`crates/bw24-engine/src/model.rs` changes (surgical).** Refactor `GpuTensor::load` (currently `model.rs:25-63`) so its body becomes `load_from_source`, then keep the old signature as a one-line wrapper:

```rust
pub fn load(e: &Engine, g: &GgufFile, name: &str) -> Result<Self, Box<dyn std::error::Error>> {
    Self::load_from_source(e, &GgufSource(g), name)          // unchanged callers, same behavior
}
pub fn load_from_source(e: &Engine, src: &dyn TensorSource, name: &str)
    -> Result<Self, Box<dyn std::error::Error>>
{
    let v = src.find(name).unwrap_or_else(|| panic!("missing tensor {name}"));
    // qtype match is IDENTICAL to model.rs:28-38 but on v.ggml_type;
    // NVFP4 sibling-".scale" lookup (model.rs:47-53) -> src.find(stem.scale);
    // None arm -> dequant::dequantize(v.ggml_type, v.bytes, n) (model.rs:56-61, unchanged).
}
```

`load_opt` (`model.rs:65-67`) gets a `load_opt_from_source` twin using `src.has`. `EmbedHost::from_gguf` (`model.rs:90-93`) gets `EmbedHost::from_source(src, name)` reading `v.bytes.to_vec()`, `v.ggml_type`, `v.ne[0]`. `HostExps::load` (`model.rs:186-213`) gets a `from_source` twin (using the `find_stacked_experts` method from §C for safetensors; GGUF impl unchanged).

**`crates/bw24-engine/src/hybrid.rs` changes.** The `load_t`/`load_opt` helpers (`hybrid.rs:11-16`) already wrap `GpuTensor::load`; redefine them to take `&dyn TensorSource` and call `load_from_source`. `HybridModel::load(e, g)` (`hybrid.rs:72`) becomes a wrapper that builds `GgufSource(g)` and calls a new `HybridModel::load_from_source(e, &dyn TensorSource)`. The entire layer loop (`hybrid.rs:85-135`) — including every `format!("blk.{il}.{s}")` — is **unchanged**, because it already speaks ggml names. Same one-line-wrapper treatment for `Model::load_dense` (`model.rs:119`). `cfg` now comes from `src.config()` instead of `ModelConfig::from_gguf(g)` (`hybrid.rs:73`, `model.rs:120`).

**`Engine` needs no changes:** `htod` (`crates/bw24-engine/src/lib.rs:253`) and `htod_bytes` (`crates/bw24-engine/src/lib.rs:61`) already take plain `&[f32]`/`&[u8]`, source-agnostic. The `QT_*` constants (`lib.rs:23-30`) are reused as-is.

---

## (E) Validation — argmax-match against the GGUF twin

Two tiers, both as a new bin `crates/bw24-engine/src/bin/st_validate.rs` (the repo already uses standalone `bin/*` check tools — see `dtype_gpu_check.rs` in git status).

**Tier 0 — pure-Rust loader unit tests** (no GPU), `crates/bw24-safetensors/src/lib.rs` `#[cfg(test)]`:
- Round-trip: write `[u64 len][json][buf]` by hand for a 2-tensor toy file, assert `StShard::raw` returns the exact bytes and `data_offsets` are honored.
- Shape-reversal assert: HF `shape=[out,in]` ⇒ `ne=[in,out]` ⇒ `GpuTensor::in_features()==in` (`crates/bw24-engine/src/model.rs:21`).
- Index sharding: synth a 2-shard `index.json`, assert every `weight_map` name resolves to the right shard.
- FP8 dtype panics with the explicit message.

**Tier 1 — argmax twin match (the real proof).** The repo's existing validation gold standard is "argmax == llama.cpp" (commit `02af8fc`: *"MoE+EDGE-1 VALIDATED — 35B-A3B argmax=1178 == llama.cpp"*). Mirror it for safetensors-vs-GGUF:

1. Pick a small dense model present on disk in **both** formats — `Qwen3-1.7B` already has safetensors (`models--Qwen--Qwen3-1.7B/.../*.safetensors` + `index.json`, cited in findings). Obtain/convert its GGUF twin.
2. `let gguf = Model::load_dense(&e, &GgufFile::open(gguf_path)?)?;`
   `let st   = Model::load_dense_from_source(&e, &SafetensorsSource::open(hf_dir)?)?;`
3. Feed an identical fixed prompt through the existing forward graph for both; compute the final-token logits.
4. **Primary assert:** `argmax(logits_st) == argmax(logits_gguf)` for the first N generated tokens (greedy). This is the same bar the project already trusts.
5. **Secondary asserts (catch silent corruption even when argmax happens to agree):** all logits finite (`is_finite()`), and per-element `|logit_st − logit_gguf|` within tolerance — looser than GGUF-vs-GGUF because F16/BF16→f32 dequant differs from the model's GGUF quant (Q4_K/Q6_K). Expect small numeric drift; argmax should still match on an unambiguous prompt.

**Tier 1 fallback** (if no GGUF twin is convertible): load safetensors alone, run forward, assert (a) every expected ggml name resolved through the mapper (no `None` panics — proves table coverage), (b) logits all finite, (c) argmax is stable across two runs (determinism). Weaker, but unblocks bring-up.

**Per-tensor spot check** (fastest corruption catch, run before any forward): for one weight, dequantize the safetensors F16/BF16 bytes and the GGUF-quant bytes of the twin and compare means/norms within tolerance. Pinpoints a shape-reversal or name-mapping bug at a single tensor instead of debugging a wrong argmax 40 layers deep.

---

## File-change summary

New crate `crates/bw24-safetensors/`: `Cargo.toml`, `src/lib.rs` (StShard/StModel mmap reader, §A + Tier-0 tests), `src/dtype.rs` (§A dtype map), `src/config.rs` (§B HfConfig→ModelConfig), `src/mapping.rs` (§C ggml↔HF table), `src/source.rs` (§D SafetensorsSource impl TensorSource).

Edited:
- `crates/bw24-gguf/src/lib.rs` — add `pub mod source;` (near line 16).
- `crates/bw24-gguf/src/source.rs` — NEW: `TensorView`, `TensorSource` trait, `GgufSource`.
- `crates/bw24-gguf/src/config.rs` — make `Arch::parse` pub (line 18); add `ModelConfig::from_hf`.
- `crates/bw24-engine/src/model.rs` — `GpuTensor::{load,load_opt}` become wrappers over new `*_from_source`; add `EmbedHost::from_source`, `HostExps::from_source`, `Model::load_dense_from_source`; old signatures preserved (lines 25, 65, 90, 119).
- `crates/bw24-engine/src/hybrid.rs` — `load_t`/`load_opt` (lines 11-16) take `&dyn TensorSource`; add `HybridModel::load_from_source`; `HybridModel::load` wraps it (line 72). Layer loop unchanged.
- `crates/bw24-engine/Cargo.toml` — add `bw24-safetensors = { path = "../bw24-safetensors" }` (only the validation bin needs it; the trait lives in bw24-gguf which engine already depends on, line 7).
- `crates/bw24-engine/src/bin/st_validate.rs` — NEW: §E Tier-1 harness.

**Risk-ranked unknowns:** (1) shape-reversal HF row-major vs bw24 `ne[0]`-fastest — the #1 silent-corruption source, asserted per-tensor in §E. (2) MoE expert layout mismatch (1 stacked 3D GGUF tensor vs 256 HF 2D tensors) — needs `find_stacked_experts`; deferred behind an explicit error for dense-first v1. (3) FP8 — explicitly unsupported in v1, panics with a clear message rather than producing garbage.
