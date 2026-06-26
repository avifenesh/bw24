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
    fn parse(s: &str) -> Self {
        match s {
            "qwen3" => Arch::Qwen3,
            "qwen3moe" => Arch::Qwen3Moe,
            "qwen35" => Arch::Qwen35,
            "qwen35moe" => Arch::Qwen35Moe,
            "llama" => Arch::Llama,
            other => Arch::Other(other.to_string()),
        }
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
