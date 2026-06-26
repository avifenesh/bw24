//! Arch-agnostic model configuration extracted from GGUF metadata.
//! One ModelConfig per loaded model; the forward pass reads it. Arch-specific
//! fields (SSM, MoE, MTP) are Option — present only for the arches that use them.

use crate::{GgufFile, MetaValue};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Arch {
    Qwen3,        // vanilla dense transformer
    Qwen3Moe,
    Qwen35,       // hybrid: gated-deltanet linear-attn + periodic full-attn + MTP
    Qwen35Moe,
    Llama,
    Other(String),
}

impl Arch {
    pub fn parse(s: &str) -> Self {
        match s {
            "qwen3" => Arch::Qwen3,
            "qwen3moe" => Arch::Qwen3Moe,
            "qwen35" => Arch::Qwen35,
            "qwen35moe" => Arch::Qwen35Moe,
            "llama" => Arch::Llama,
            other => Arch::Other(other.to_string()),
        }
    }

    /// Map an HF `model_type` (config.json) to the ggml-style Arch. HF uses different strings
    /// than GGUF (`qwen3_moe` vs `qwen3moe`, `qwen3_5` vs `qwen35`), so normalize first.
    pub fn from_hf_model_type(mt: &str) -> Self {
        let ggml = match mt {
            "qwen3" => "qwen3",
            "qwen3_moe" => "qwen3moe",
            "qwen3_5" | "qwen3_5_text" | "qwen3_next" => "qwen35",
            "qwen3_5_moe" | "qwen3_next_moe" => "qwen35moe",
            "llama" => "llama",
            other => other,
        };
        Arch::parse(ggml)
    }
    /// True for arches with interleaved linear-attention (SSM) layers.
    pub fn is_hybrid(&self) -> bool {
        matches!(self, Arch::Qwen35 | Arch::Qwen35Moe)
    }
    pub fn is_moe(&self) -> bool {
        matches!(self, Arch::Qwen3Moe | Arch::Qwen35Moe)
    }
}

/// What kind of token-mixing a given layer performs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    FullAttention,   // softmax attention with growing KV cache
    LinearAttention, // gated-deltanet / SSM with fixed recurrent state
}

#[derive(Debug, Clone)]
pub struct SsmConfig {
    pub conv_kernel: u32,
    pub inner_size: u32,
    pub state_size: u32,
    pub time_step_rank: u32,
    pub group_count: u32,
}

#[derive(Debug, Clone)]
pub struct MoeConfig {
    pub expert_count: u32,
    pub expert_used_count: u32,
    pub expert_ff_length: u32,
    pub expert_shared_ff_length: u32,   // NEW: qwen35moe.expert_shared_feed_forward_length = 512
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub arch: Arch,
    pub name: String,
    pub n_layer: u32,
    pub n_embd: u32,
    pub n_head: u32,
    pub n_head_kv: u32,
    pub head_dim_k: u32,
    pub head_dim_v: u32,
    pub n_ff: u32,
    pub n_vocab: u32,
    pub context_length: u32,
    pub rms_eps: f32,
    pub rope_freq_base: f32,
    pub rope_dim_count: u32,            // partial rotary: only this many head dims get RoPE
    pub rope_sections: Vec<i32>,        // M-RoPE sections (qwen35), empty if plain
    // hybrid (qwen35)
    pub full_attention_interval: u32,   // 0 if not hybrid; else every Nth layer is full-attn
    pub ssm: Option<SsmConfig>,
    // moe
    pub moe: Option<MoeConfig>,
    // multi-token-predict / NextN
    pub nextn_predict_layers: u32,
    pub n_layer_total: u32,             // includes appended MTP layers
}

impl ModelConfig {
    pub fn from_gguf(g: &GgufFile) -> Self {
        let arch = Arch::parse(g.arch().unwrap_or("unknown"));
        let u = |k: &str| g.meta_arch(k).and_then(|v| v.as_u64()).map(|x| x as u32);
        let f = |k: &str| g.meta_arch(k).and_then(|v| v.as_f32());

        let n_layer = u("block_count").expect("block_count");
        let n_embd = u("embedding_length").expect("embedding_length");
        let head_dim_k = u("attention.key_length").unwrap_or_else(|| {
            // fall back to n_embd / n_head if not present
            n_embd / u("attention.head_count").unwrap_or(1)
        });
        let head_dim_v = u("attention.value_length").unwrap_or(head_dim_k);

        let rope_sections = match g.meta_arch("rope.dimension_sections") {
            Some(MetaValue::Array(a)) => a.iter().filter_map(|v| v.as_u64().map(|x| x as i32)).collect(),
            _ => Vec::new(),
        };

        let ssm = if arch.is_hybrid() {
            Some(SsmConfig {
                conv_kernel: u("ssm.conv_kernel").unwrap_or(0),
                inner_size: u("ssm.inner_size").unwrap_or(0),
                state_size: u("ssm.state_size").unwrap_or(0),
                time_step_rank: u("ssm.time_step_rank").unwrap_or(0),
                group_count: u("ssm.group_count").unwrap_or(0),
            })
        } else { None };

        let moe = if arch.is_moe() {
            Some(MoeConfig {
                expert_count: u("expert_count").unwrap_or(0),
                expert_used_count: u("expert_used_count").unwrap_or(0),
                expert_ff_length: u("expert_feed_forward_length").unwrap_or(0),
                // meta_arch tries "qwen35moe.expert_shared_feed_forward_length" first, then bare key
                expert_shared_ff_length: u("expert_shared_feed_forward_length").unwrap_or(0),
            })
        } else { None };

        let nextn = u("nextn_predict_layers").unwrap_or(0);

        ModelConfig {
            arch,
            name: g.metadata.get("general.name").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            n_layer,
            n_embd,
            n_head: u("attention.head_count").expect("head_count"),
            n_head_kv: u("attention.head_count_kv").unwrap_or_else(|| u("attention.head_count").unwrap()),
            head_dim_k,
            head_dim_v,
            n_ff: u("feed_forward_length").unwrap_or(0),
            n_vocab: u("vocab_size").unwrap_or_else(|| {
                // vocab size from token_embd tensor's last dim if metadata absent
                g.find("token_embd.weight").map(|t| *t.ne.last().unwrap() as u32).unwrap_or(0)
            }),
            context_length: u("context_length").unwrap_or(0),
            rms_eps: f("attention.layer_norm_rms_epsilon").unwrap_or(1e-6),
            rope_freq_base: f("rope.freq_base").unwrap_or(10000.0),
            rope_dim_count: u("rope.dimension_count").unwrap_or(head_dim_k),
            rope_sections,
            full_attention_interval: u("full_attention_interval").unwrap_or(0),
            ssm,
            moe,
            nextn_predict_layers: nextn,
            n_layer_total: n_layer + nextn,
        }
    }

    /// Build a ModelConfig from an HF `config.json` (read parallel to a safetensors checkpoint).
    /// HF has no `{arch}.`-prefixed keys (unlike GGUF), so we read its flat field names. Hybrid
    /// (qwen3_5) nests the transformer fields under `text_config`; `from_config_json` flattens that
    /// before calling here. Lenient defaults mirror the GGUF fallbacks in `from_gguf`.
    pub fn from_hf(c: &HfConfig) -> Self {
        let arch = Arch::from_hf_model_type(&c.model_type);
        let n_head = c.num_attention_heads;
        let head_dim_k = c.head_dim.unwrap_or_else(|| c.hidden_size / n_head.max(1));
        let head_dim_v = head_dim_k;
        let n_head_kv = c.num_key_value_heads.unwrap_or(n_head);

        let moe = if c.num_experts.is_some() || arch.is_moe() {
            Some(MoeConfig {
                expert_count: c.num_experts.unwrap_or(0),
                expert_used_count: c.num_experts_per_tok.unwrap_or(0),
                expert_ff_length: c.moe_intermediate_size.unwrap_or(0),
                expert_shared_ff_length: c.shared_expert_intermediate_size.unwrap_or(0),
            })
        } else {
            None
        };

        let ssm = if arch.is_hybrid() {
            // qwen3_5 linear-attn config keys (text_config). Best-effort; full hybrid safetensors
            // bring-up is out of scope (no validated checkpoint), so missing keys default to 0.
            Some(SsmConfig {
                conv_kernel: c.linear_conv_kernel_dim.unwrap_or(0),
                inner_size: c.linear_value_head_dim.unwrap_or(0) * c.linear_num_value_heads.unwrap_or(0),
                state_size: c.linear_key_head_dim.unwrap_or(0),
                time_step_rank: 0,
                group_count: c.linear_num_key_heads.unwrap_or(0),
            })
        } else {
            None
        };

        ModelConfig {
            arch,
            name: c.name.clone().unwrap_or_default(),
            n_layer: c.num_hidden_layers,
            n_embd: c.hidden_size,
            n_head,
            n_head_kv,
            head_dim_k,
            head_dim_v,
            n_ff: c.intermediate_size,
            n_vocab: c.vocab_size,
            context_length: c.max_position_embeddings,
            rms_eps: c.rms_norm_eps,
            rope_freq_base: c.rope_theta,
            rope_dim_count: head_dim_k,
            rope_sections: Vec::new(),
            full_attention_interval: c.full_attention_interval.unwrap_or(0),
            ssm,
            moe,
            nextn_predict_layers: c.num_nextn_predict_layers.unwrap_or(0),
            n_layer_total: c.num_hidden_layers + c.num_nextn_predict_layers.unwrap_or(0),
        }
    }

    /// Read + parse an HF `config.json` directly from disk and build a ModelConfig.
    pub fn from_config_json(path: &std::path::Path) -> std::io::Result<Self> {
        let txt = std::fs::read_to_string(path)?;
        let cfg = HfConfig::parse(&txt);
        Ok(Self::from_hf(&cfg))
    }

    /// Classify a layer index. For hybrid models, layer il is full-attention when
    /// (il+1) % full_attention_interval == 0, else linear-attention (matches llama.cpp qwen35).
    /// Non-hybrid models are always full-attention.
    pub fn layer_kind(&self, il: u32) -> LayerKind {
        if self.full_attention_interval == 0 {
            return LayerKind::FullAttention;
        }
        if (il + 1) % self.full_attention_interval == 0 {
            LayerKind::FullAttention
        } else {
            LayerKind::LinearAttention
        }
    }

    /// Count of full-attention layers (the ones that carry a growing KV cache).
    pub fn n_full_attn_layers(&self) -> u32 {
        (0..self.n_layer).filter(|&il| self.layer_kind(il) == LayerKind::FullAttention).count() as u32
    }
}

/// Subset of HF `config.json` fields bw24 needs. Defaults mirror GGUF fallbacks. Hybrid models
/// (qwen3_5) nest the transformer fields under `text_config` — `parse` flattens that automatically.
#[derive(Debug, Clone)]
pub struct HfConfig {
    pub model_type: String,
    pub name: Option<String>,
    pub num_hidden_layers: u32,
    pub hidden_size: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: Option<u32>,
    pub head_dim: Option<u32>,
    pub intermediate_size: u32,
    pub vocab_size: u32,
    pub max_position_embeddings: u32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub full_attention_interval: Option<u32>,
    pub num_nextn_predict_layers: Option<u32>,
    // MoE
    pub num_experts: Option<u32>,
    pub num_experts_per_tok: Option<u32>,
    pub moe_intermediate_size: Option<u32>,
    pub shared_expert_intermediate_size: Option<u32>,
    // hybrid linear-attn (qwen3_5 text_config)
    pub linear_conv_kernel_dim: Option<u32>,
    pub linear_key_head_dim: Option<u32>,
    pub linear_value_head_dim: Option<u32>,
    pub linear_num_key_heads: Option<u32>,
    pub linear_num_value_heads: Option<u32>,
}

impl Default for HfConfig {
    fn default() -> Self {
        HfConfig {
            model_type: String::new(),
            name: None,
            num_hidden_layers: 0,
            hidden_size: 0,
            num_attention_heads: 0,
            num_key_value_heads: None,
            head_dim: None,
            intermediate_size: 0,
            vocab_size: 0,
            max_position_embeddings: 0,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            full_attention_interval: None,
            num_nextn_predict_layers: None,
            num_experts: None,
            num_experts_per_tok: None,
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            linear_conv_kernel_dim: None,
            linear_key_head_dim: None,
            linear_value_head_dim: None,
            linear_num_key_heads: None,
            linear_num_value_heads: None,
        }
    }
}

impl HfConfig {
    /// Parse an HF config.json. Reads scalar fields at the top level; if a `text_config`
    /// object is present (vision-language / hybrid wrappers like qwen3_5), its scalar fields
    /// override the top-level ones for the transformer config. `architectures[0]` and the
    /// top-level `model_type` seed the arch when `text_config.model_type` is more specific.
    pub fn parse(json: &str) -> Self {
        let top = JsonObj::parse(json);
        let mut cfg = HfConfig::default();
        cfg.apply(&top);
        // text_config (hybrid / VLM wrappers) — its transformer fields take precedence.
        if let Some(tc) = top.object("text_config") {
            cfg.apply(&tc);
        }
        // model_type fallback chain: text_config.model_type > model_type > architectures[0].
        if cfg.model_type.is_empty() {
            if let Some(arch0) = top.first_string_in_array("architectures") {
                cfg.model_type = arch0;
            }
        }
        cfg
    }

    fn apply(&mut self, o: &JsonObj) {
        if let Some(s) = o.string("model_type") { self.model_type = s; }
        if let Some(s) = o.string("name_or_path").or_else(|| o.string("_name_or_path")) { self.name = Some(s); }
        if let Some(v) = o.u32("num_hidden_layers") { self.num_hidden_layers = v; }
        if let Some(v) = o.u32("hidden_size") { self.hidden_size = v; }
        if let Some(v) = o.u32("num_attention_heads") { self.num_attention_heads = v; }
        if let Some(v) = o.u32("num_key_value_heads") { self.num_key_value_heads = Some(v); }
        if let Some(v) = o.u32("head_dim") { self.head_dim = Some(v); }
        if let Some(v) = o.u32("intermediate_size") { self.intermediate_size = v; }
        if let Some(v) = o.u32("vocab_size") { self.vocab_size = v; }
        if let Some(v) = o.u32("max_position_embeddings") { self.max_position_embeddings = v; }
        if let Some(v) = o.f32("rms_norm_eps") { self.rms_norm_eps = v; }
        if let Some(v) = o.f32("rope_theta") { self.rope_theta = v; }
        if let Some(v) = o.u32("full_attention_interval") { self.full_attention_interval = Some(v); }
        if let Some(v) = o.u32("num_nextn_predict_layers") { self.num_nextn_predict_layers = Some(v); }
        if let Some(v) = o.u32("num_experts").or_else(|| o.u32("num_local_experts")) { self.num_experts = Some(v); }
        if let Some(v) = o.u32("num_experts_per_tok") { self.num_experts_per_tok = Some(v); }
        if let Some(v) = o.u32("moe_intermediate_size") { self.moe_intermediate_size = Some(v); }
        if let Some(v) = o.u32("shared_expert_intermediate_size") { self.shared_expert_intermediate_size = Some(v); }
        if let Some(v) = o.u32("linear_conv_kernel_dim") { self.linear_conv_kernel_dim = Some(v); }
        if let Some(v) = o.u32("linear_key_head_dim") { self.linear_key_head_dim = Some(v); }
        if let Some(v) = o.u32("linear_value_head_dim") { self.linear_value_head_dim = Some(v); }
        if let Some(v) = o.u32("linear_num_key_heads") { self.linear_num_key_heads = Some(v); }
        if let Some(v) = o.u32("linear_num_value_heads") { self.linear_num_value_heads = Some(v); }
    }
}

// ============================ minimal flat JSON object reader ============================
//
// config.json is a flat-ish object; we only need scalar fields + one level of nested object
// (text_config) + the architectures string array. Rather than add serde to bw24-gguf, parse
// the value-bearing tokens for the keys we care about. Nested objects/arrays are captured as
// raw substrings so they can be re-parsed on demand.

struct JsonObj {
    // key -> raw value substring (trimmed). Objects/arrays keep their braces/brackets.
    fields: std::collections::BTreeMap<String, String>,
}

impl JsonObj {
    fn parse(json: &str) -> Self {
        let b = json.as_bytes();
        let mut i = 0usize;
        let mut fields = std::collections::BTreeMap::new();
        // find opening brace
        while i < b.len() && b[i] != b'{' { i += 1; }
        if i >= b.len() { return JsonObj { fields }; }
        i += 1; // past '{'
        loop {
            skip_ws(b, &mut i);
            if i >= b.len() || b[i] == b'}' { break; }
            if b[i] != b'"' {
                // unexpected; bail gracefully
                break;
            }
            let key = read_string(b, &mut i);
            skip_ws(b, &mut i);
            if i >= b.len() || b[i] != b':' { break; }
            i += 1; // ':'
            skip_ws(b, &mut i);
            let val = read_value_raw(b, &mut i);
            fields.insert(key, val);
            skip_ws(b, &mut i);
            if i < b.len() && b[i] == b',' { i += 1; continue; }
            break;
        }
        JsonObj { fields }
    }

    fn raw(&self, key: &str) -> Option<&str> {
        self.fields.get(key).map(|s| s.as_str())
    }

    fn string(&self, key: &str) -> Option<String> {
        let v = self.raw(key)?.trim();
        if v.starts_with('"') && v.ends_with('"') && v.len() >= 2 {
            Some(v[1..v.len() - 1].to_string())
        } else {
            None
        }
    }

    fn u32(&self, key: &str) -> Option<u32> {
        let v = self.raw(key)?.trim();
        if v == "null" { return None; }
        // accept integers (and floats that are whole, e.g. "8.0")
        v.parse::<u64>().ok().map(|x| x as u32)
            .or_else(|| v.parse::<f64>().ok().map(|x| x as u32))
    }

    fn f32(&self, key: &str) -> Option<f32> {
        let v = self.raw(key)?.trim();
        if v == "null" { return None; }
        v.parse::<f32>().ok()
    }

    fn object(&self, key: &str) -> Option<JsonObj> {
        let v = self.raw(key)?.trim();
        if v.starts_with('{') { Some(JsonObj::parse(v)) } else { None }
    }

    /// First string element of a string array field (e.g. architectures[0]).
    fn first_string_in_array(&self, key: &str) -> Option<String> {
        let v = self.raw(key)?.trim();
        let inner = v.strip_prefix('[')?.trim_start();
        let q = inner.find('"')? + 1;
        let rest = &inner[q..];
        let end = rest.find('"')?;
        Some(rest[..end].to_string())
    }
}

fn skip_ws(b: &[u8], i: &mut usize) {
    while *i < b.len() && matches!(b[*i], b' ' | b'\t' | b'\n' | b'\r') { *i += 1; }
}

fn read_string(b: &[u8], i: &mut usize) -> String {
    // assumes b[*i] == '"'
    *i += 1;
    let mut s = String::new();
    while *i < b.len() {
        let c = b[*i];
        *i += 1;
        match c {
            b'"' => break,
            b'\\' => {
                if *i < b.len() {
                    let e = b[*i];
                    *i += 1;
                    s.push(match e { b'n' => '\n', b't' => '\t', b'r' => '\r', other => other as char });
                }
            }
            _ => s.push(c as char),
        }
    }
    s
}

/// Read a raw value substring (string with quotes, number, bool/null, or a balanced {}/[] block).
fn read_value_raw(b: &[u8], i: &mut usize) -> String {
    skip_ws(b, i);
    let start = *i;
    match b.get(*i).copied() {
        Some(b'"') => {
            // string value — include the quotes
            *i += 1;
            while *i < b.len() {
                let c = b[*i];
                *i += 1;
                if c == b'\\' { *i += 1; continue; }
                if c == b'"' { break; }
            }
            String::from_utf8_lossy(&b[start..*i]).into_owned()
        }
        Some(b'{') | Some(b'[') => {
            // balanced block, respecting strings inside
            let open = b[*i];
            let close = if open == b'{' { b'}' } else { b']' };
            let mut depth = 0i32;
            let mut in_str = false;
            while *i < b.len() {
                let c = b[*i];
                *i += 1;
                if in_str {
                    if c == b'\\' { *i += 1; }
                    else if c == b'"' { in_str = false; }
                    continue;
                }
                match c {
                    b'"' => in_str = true,
                    x if x == open => depth += 1,
                    x if x == close => { depth -= 1; if depth == 0 { break; } }
                    _ => {}
                }
            }
            String::from_utf8_lossy(&b[start..*i]).into_owned()
        }
        _ => {
            // scalar: number / true / false / null — until , } ] or whitespace
            while *i < b.len() && !matches!(b[*i], b',' | b'}' | b']') {
                *i += 1;
            }
            String::from_utf8_lossy(&b[start..*i]).trim().to_string()
        }
    }
}

#[cfg(test)]
mod hf_tests {
    use super::*;

    const QWEN3_17B: &str = r#"{
      "architectures": ["Qwen3ForCausalLM"],
      "head_dim": 128,
      "hidden_size": 2048,
      "intermediate_size": 6144,
      "max_position_embeddings": 40960,
      "model_type": "qwen3",
      "num_attention_heads": 16,
      "num_hidden_layers": 28,
      "num_key_value_heads": 8,
      "rms_norm_eps": 1e-06,
      "rope_theta": 1000000,
      "tie_word_embeddings": true,
      "torch_dtype": "bfloat16",
      "vocab_size": 151936
    }"#;

    #[test]
    fn parse_qwen3_dense() {
        let c = HfConfig::parse(QWEN3_17B);
        assert_eq!(c.model_type, "qwen3");
        assert_eq!(c.num_hidden_layers, 28);
        assert_eq!(c.hidden_size, 2048);
        assert_eq!(c.num_attention_heads, 16);
        assert_eq!(c.num_key_value_heads, Some(8));
        assert_eq!(c.head_dim, Some(128));
        assert_eq!(c.intermediate_size, 6144);
        assert_eq!(c.vocab_size, 151936);
        assert_eq!(c.max_position_embeddings, 40960);
        assert!((c.rms_norm_eps - 1e-6).abs() < 1e-12);
        assert!((c.rope_theta - 1_000_000.0).abs() < 1.0);

        let mc = ModelConfig::from_hf(&c);
        assert_eq!(mc.arch, Arch::Qwen3);
        assert_eq!(mc.n_layer, 28);
        assert_eq!(mc.n_embd, 2048);
        assert_eq!(mc.n_head, 16);
        assert_eq!(mc.n_head_kv, 8);
        assert_eq!(mc.head_dim_k, 128);
        assert_eq!(mc.n_ff, 6144);
        assert_eq!(mc.n_vocab, 151936);
        assert!(mc.moe.is_none());
        assert!(mc.ssm.is_none());
        assert_eq!(mc.full_attention_interval, 0);
    }

    #[test]
    fn head_dim_fallback() {
        // no head_dim -> hidden_size / num_attention_heads
        let json = r#"{"model_type":"llama","num_hidden_layers":2,"hidden_size":256,"num_attention_heads":8,"intermediate_size":512,"vocab_size":1000,"max_position_embeddings":2048}"#;
        let c = HfConfig::parse(json);
        let mc = ModelConfig::from_hf(&c);
        assert_eq!(mc.arch, Arch::Llama);
        assert_eq!(mc.head_dim_k, 32); // 256/8
        assert_eq!(mc.n_head_kv, 8); // defaults to n_head when absent
    }

    #[test]
    fn nested_text_config_hybrid() {
        // qwen3_5 wraps the transformer config in text_config and uses HF model_type "qwen3_5".
        let json = r#"{
          "architectures": ["Qwen3_5ForConditionalGeneration"],
          "model_type": "qwen3_5",
          "text_config": {
            "model_type": "qwen3_5_text",
            "full_attention_interval": 4,
            "head_dim": 256,
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "num_key_value_heads": 8,
            "vocab_size": 151936,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-06,
            "rope_theta": 5000000,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32
          }
        }"#;
        let c = HfConfig::parse(json);
        // text_config fields win
        assert_eq!(c.hidden_size, 4096);
        assert_eq!(c.num_hidden_layers, 32);
        assert_eq!(c.full_attention_interval, Some(4));
        assert_eq!(c.model_type, "qwen3_5_text");

        let mc = ModelConfig::from_hf(&c);
        assert_eq!(mc.arch, Arch::Qwen35);
        assert!(mc.arch.is_hybrid());
        assert_eq!(mc.full_attention_interval, 4);
        assert!(mc.ssm.is_some());
        // periodic full-attn classification still works
        assert_eq!(mc.layer_kind(3), LayerKind::FullAttention); // (3+1)%4==0
        assert_eq!(mc.layer_kind(0), LayerKind::LinearAttention);
    }

    #[test]
    fn moe_config() {
        let json = r#"{"model_type":"qwen3_moe","num_hidden_layers":4,"hidden_size":2048,"num_attention_heads":16,"num_key_value_heads":4,"intermediate_size":6144,"vocab_size":151936,"max_position_embeddings":40960,"num_experts":128,"num_experts_per_tok":8,"moe_intermediate_size":768,"shared_expert_intermediate_size":0}"#;
        let c = HfConfig::parse(json);
        let mc = ModelConfig::from_hf(&c);
        assert_eq!(mc.arch, Arch::Qwen3Moe);
        let moe = mc.moe.expect("moe");
        assert_eq!(moe.expert_count, 128);
        assert_eq!(moe.expert_used_count, 8);
        assert_eq!(moe.expert_ff_length, 768);
    }
}
