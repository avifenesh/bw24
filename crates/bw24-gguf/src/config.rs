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
    Olmoe,        // dense full-attention + MoE FFN (no shared expert, no SSM, no MTP)
    MinimaxM3,    // dense full-attention (MSA later) + MoE FFN: sigmoid router + shared expert,
                  // gemma-norm, swigluoai, GQA 64/4 hd128 partial-RoPE, QK-norm
    Hy3,          // dense full-attention + MoE FFN: sigmoid router + bias + shared MLP, QK-norm
    Gemma4,       // hybrid SWA(1024)/global 5:1, per-layer kv-heads+head_dim+rope, K=V globals,
                  // 128-expert MoE + parallel shared FFN, gelu_tanh, softcap 30, layer_output_scale
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
            "olmoe" => Arch::Olmoe,
            "minimax-m3" => Arch::MinimaxM3,
            "hy3" => Arch::Hy3,
            "gemma4" => Arch::Gemma4,
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
            "olmoe" => "olmoe",
            // MiniMax-M3 (incl the VL wrapper model_type; text_config flattening handles the rest)
            "minimax_m3" | "minimax_m3_vl" | "minimax_m3_text" => "minimax-m3",
            "hy_v3" | "hy3" => "hy3",
            "llama" => "llama",
            other => other,
        };
        Arch::parse(ggml)
    }
    /// Arches the HybridModel loader/forward handles. MinimaxM3 and Hy3 qualify as the degenerate
    /// hybrid: full_attention_interval=0 -> every layer Mixer::Full, no SSM state, MoE FFN.
    /// (Hy3 joined 2026-07-09: the decode/KV/spec machinery lives on HybridModel only — the dense
    /// `Model` has no decode path — and M3 already proved the dense-attention-MoE-as-degenerate-
    /// hybrid shape end to end. "Not qwen35 hybrid" holds where it matters: zero SSM/linear layers.)
    pub fn is_hybrid(&self) -> bool {
        matches!(self, Arch::Qwen35 | Arch::Qwen35Moe | Arch::MinimaxM3 | Arch::Hy3 | Arch::Gemma4)
    }
    /// True for arches with a routed-expert FFN. `Olmoe` is dense-attention + MoE-FFN.
    pub fn is_moe(&self) -> bool {
        matches!(self, Arch::Qwen3Moe | Arch::Qwen35Moe | Arch::Olmoe | Arch::MinimaxM3 | Arch::Hy3 | Arch::Gemma4)
    }
    /// MiniMax-M3: sigmoid router (+e_score_correction_bias), gemma-norm, swigluoai clamp,
    /// Mixtral-style expert tensor names. Full attention v0 (MSA is bit-exact-degenerate <=2048
    /// ctx — the sparse indexer selects everything; the MSA kernel is a later arc).
    pub fn is_minimax(&self) -> bool { matches!(self, Arch::MinimaxM3) }
    /// Tencent HunYuan/Hunyuan Hy3 (`hy_v3` in HF config.json).
    pub fn is_hy3(&self) -> bool { matches!(self, Arch::Hy3) }
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

/// MiniMax-M3-specific forward-pass knobs (config.json, minimax_m3_vl text_config).
#[derive(Debug, Clone)]
pub struct M3Config {
    pub use_gemma_norm: bool,           // (1+w) RMSNorm — folded into weights at load
    pub sigmoid_routing: bool,          // scoring_func == "sigmoid" (DeepSeek-V3 style)
    pub use_routing_bias: bool,         // e_score_correction_bias on SELECTION only
    pub routed_scaling_factor: f32,     // 2.0 — multiplies the normalized routing weights
    pub n_shared_experts: u32,          // 1
    pub swiglu_alpha: f32,              // swigluoai: gate*sigmoid(alpha*gate), clamp at limit
    pub swiglu_limit: f32,              // 7.0
    pub rotary_dim: u32,                // partial RoPE (64 of head_dim 128)
    pub dense_intermediate_size: u32,   // dense-FFN layers' n_ff (12288)
    pub moe_layer_freq: Vec<u32>,       // per-layer 0=dense 1=moe (len == n_layer)
}

/// Hy3-specific loader metadata. Forward/kernel support is a later GPU-gated lane; these fields
/// let the CPU-side loader distinguish REAP's dense layer 0 from routed layers 1..79 and preserve
/// the routing contract documented in the port dossier.
#[derive(Debug, Clone)]
pub struct Hy3Config {
    pub sigmoid_routing: bool,
    pub use_routing_bias: bool,
    pub route_norm: bool,
    pub router_scaling_factor: f32,
    pub n_shared_experts: u32,
    pub first_k_dense_replace: u32,
    pub qk_norm: bool,
    pub hidden_act: String,
}

/// Gemma-4 per-layer attention geometry + block extras (P0 census 2026-07-10).
#[derive(Debug, Clone)]
pub struct Gemma4Config {
    pub head_count_kv: Vec<u32>,     // per layer (8 SWA / 2 global on the 26B)
    pub swa_pattern: Vec<bool>,      // true = sliding-window layer
    pub sliding_window: u32,         // 1024
    pub key_length_global: u32,      // 512
    pub key_length_swa: u32,         // 256
    pub rope_base_global: f32,       // 1e6 (+ rope_freqs.weight factors tensor)
    pub rope_base_swa: f32,          // 1e4
    pub rope_dims_global: u32,       // 512 metadata (p-RoPE partial applies via rope_freqs)
    pub rope_dims_swa: u32,          // 256
    pub final_logit_softcapping: f32, // 30.0
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
    // MiniMax-M3 extras (None for every other arch)
    pub m3: Option<M3Config>,
    // Hy3 extras (None for every other arch)
    pub hy3: Option<Hy3Config>,
    pub gemma4: Option<Gemma4Config>,
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

        let gemma4 = if matches!(&arch, Arch::Gemma4) {
                let arr_u = |k: &str| -> Vec<u32> {
                    match g.meta_arch(k) {
                        Some(MetaValue::Array(a)) => a.iter().filter_map(|v| v.as_u64().map(|x| x as u32)).collect(),
                        _ => Vec::new(),
                    }
                };
                Some(Gemma4Config {
                    head_count_kv: arr_u("attention.head_count_kv"),
                    swa_pattern: arr_u("attention.sliding_window_pattern").iter().map(|&x| x == 1).collect(),
                    sliding_window: u("attention.sliding_window").unwrap_or(1024),
                    key_length_global: u("attention.key_length").unwrap_or(512),
                    key_length_swa: u("attention.key_length_swa").unwrap_or(256),
                    rope_base_global: f("rope.freq_base").unwrap_or(1e6),
                    rope_base_swa: f("rope.freq_base_swa").unwrap_or(1e4),
                    rope_dims_global: u("rope.dimension_count").unwrap_or(512),
                    rope_dims_swa: u("rope.dimension_count_swa").unwrap_or(256),
                    final_logit_softcapping: f("final_logit_softcapping").unwrap_or(30.0),
                })
            } else { None };

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
            m3: None,   // GGUF M3 metadata keys are a later arc (ST import first)
            hy3: None,  // GGUF Hy3 metadata keys are a later arc (repack source first)
            gemma4,
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

        let moe = if c.num_experts.is_some() || c.num_local_experts.is_some() || arch.is_moe() {
            let expert_ff_length = c.moe_intermediate_size
                .or(c.expert_hidden_dim)
                .unwrap_or(c.intermediate_size);
            let n_shared = c.n_shared_experts.unwrap_or(0);
            let shared_ff_length = c.shared_expert_intermediate_size
                .or(c.shared_intermediate_size)
                .or_else(|| if arch.is_hy3() && n_shared > 0 { Some(expert_ff_length * n_shared) } else { None })
                .unwrap_or(0);
            Some(MoeConfig {
                // M3 names the count `num_local_experts`, the shared FF `shared_intermediate_size`.
                expert_count: c.num_experts.or(c.num_local_experts).unwrap_or(0),
                expert_used_count: c.num_experts_per_tok.unwrap_or(0),
                // OLMoE has no separate `moe_intermediate_size`; its experts use `intermediate_size`.
                expert_ff_length,
                expert_shared_ff_length: shared_ff_length,
            })
        } else {
            None
        };

        let m3 = if arch.is_minimax() {
            Some(M3Config {
                use_gemma_norm: c.use_gemma_norm.unwrap_or(false),
                sigmoid_routing: c.scoring_func.as_deref() == Some("sigmoid"),
                use_routing_bias: c.use_routing_bias.unwrap_or(false),
                routed_scaling_factor: c.routed_scaling_factor.unwrap_or(1.0),
                n_shared_experts: c.n_shared_experts.unwrap_or(0),
                swiglu_alpha: c.swiglu_alpha.unwrap_or(1.702),
                swiglu_limit: c.swiglu_limit.unwrap_or(7.0),
                rotary_dim: c.rotary_dim.unwrap_or(0),
                dense_intermediate_size: c.dense_intermediate_size.unwrap_or(c.intermediate_size),
                moe_layer_freq: c.moe_layer_freq.clone().unwrap_or_default(),
            })
        } else { None };

        let hy3 = if arch.is_hy3() {
            Some(Hy3Config {
                sigmoid_routing: c.moe_router_use_sigmoid.unwrap_or(false),
                use_routing_bias: c.moe_router_enable_expert_bias.unwrap_or(false),
                route_norm: c.route_norm.unwrap_or(false),
                router_scaling_factor: c.router_scaling_factor.unwrap_or(1.0),
                n_shared_experts: c.n_shared_experts.unwrap_or(0),
                first_k_dense_replace: c.first_k_dense_replace.unwrap_or(1),
                qk_norm: c.qk_norm.unwrap_or(false),
                hidden_act: c.hidden_act.clone().unwrap_or_else(|| "silu".to_string()),
            })
        } else { None };

        let ssm = if arch.is_hybrid() {
            // qwen3_5 linear-attn config keys (text_config). Mirror the GGUF ssm.* fields the hybrid
            // forward reads: state_size=key_head_dim(128), group_count=num_key_heads(16),
            // time_step_rank=num_value_heads(32), conv_kernel(4), inner_size=value_head_dim*num_value.
            Some(SsmConfig {
                conv_kernel: c.linear_conv_kernel_dim.unwrap_or(0),
                inner_size: c.linear_value_head_dim.unwrap_or(0) * c.linear_num_value_heads.unwrap_or(0),
                state_size: c.linear_key_head_dim.unwrap_or(0),
                time_step_rank: c.linear_num_value_heads.unwrap_or(0),
                group_count: c.linear_num_key_heads.unwrap_or(0),
            })
        } else {
            None
        };

        // NextN/MTP depth: 35B-MoE HF uses `num_nextn_predict_layers`; qwen3.6-27B (dense hybrid,
        // NVIDIA + local text ckpts) uses `mtp_num_hidden_layers`. Same meaning (head depth = 1).
        let nextn = c.num_nextn_predict_layers.or(c.mtp_num_hidden_layers).unwrap_or(0);

        ModelConfig {
            arch,
            name: c.name.clone().unwrap_or_default(),
            // GGUF `block_count` INCLUDES the MTP/NextN block(s) (hybrid.rs n_trunk = n_layer -
            // nextn); HF `num_hidden_layers` EXCLUDES them. Add nextn so both sources agree.
            n_layer: c.num_hidden_layers + nextn,
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
            // partial RoPE: M3 rotates only rotary_dim (64) of head_dim (128).
            rope_dim_count: c.rotary_dim.unwrap_or(head_dim_k),
            rope_sections: Vec::new(),
            full_attention_interval: c.full_attention_interval.unwrap_or(0),
            ssm,
            moe,
            m3,
            hy3,
            gemma4: None,   // ST gemma4 route: config wiring when that arc opens
            // NextN/MTP depth: 35B-MoE HF uses `num_nextn_predict_layers`; the 27B (dense hybrid)
            // uses `mtp_num_hidden_layers` (NVIDIA + local text ckpts) — same meaning, both = 1.
            nextn_predict_layers: c.num_nextn_predict_layers.or(c.mtp_num_hidden_layers).unwrap_or(0),
            n_layer_total: c.num_hidden_layers
                + c.num_nextn_predict_layers.or(c.mtp_num_hidden_layers).unwrap_or(0),
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

    /// qwen35-class fused [q|gate] attention output gate: wq packs q AND a per-head sigmoid gate
    /// (out = 2*n_head*head_dim) that `q_gate_split` separates. M3 and Hy3 have NO output gate —
    /// their wq out is exactly n_head*head_dim, and running the split would read 2x out of bounds.
    /// One predicate so every full-attn site (prefill/prime/decode/dc/spec) agrees.
    pub fn attn_out_gate(&self) -> bool {
        self.m3.is_none() && self.hy3.is_none()
    }

    /// DeepSeek-V3-class sigmoid routing knobs, arch-agnostic: `Some((scaling_factor, route_norm))`
    /// when the router scores with sigmoid (+ optional selection bias via `exp_probs_b`), `None`
    /// for the softmax archs. route_norm: sum-normalize the selected weights before scaling
    /// (true for M3 — its modeling code always normalizes — and for Hy3's `route_norm=true`).
    /// Sites that must NOT enter the fused SOFTMAX device-router arms key off `is_some()`.
    pub fn sigmoid_router(&self) -> Option<(f32, bool)> {
        if let Some(m3) = self.m3.as_ref() {
            if m3.sigmoid_routing { return Some((m3.routed_scaling_factor, true)); }
        }
        if let Some(hy3) = self.hy3.as_ref() {
            if hy3.sigmoid_routing { return Some((hy3.router_scaling_factor, hy3.route_norm)); }
        }
        None
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
    pub mtp_num_hidden_layers: Option<u32>,   // qwen3_5/3_6 HF key for the MTP head depth (27B: 1)
    // MoE
    pub num_experts: Option<u32>,
    pub num_experts_per_tok: Option<u32>,
    pub moe_intermediate_size: Option<u32>,
    pub expert_hidden_dim: Option<u32>,
    pub shared_expert_intermediate_size: Option<u32>,
    // hybrid linear-attn (qwen3_5 text_config)
    pub linear_conv_kernel_dim: Option<u32>,
    pub linear_key_head_dim: Option<u32>,
    pub linear_value_head_dim: Option<u32>,
    pub linear_num_key_heads: Option<u32>,
    pub linear_num_value_heads: Option<u32>,
    // ---- MiniMax-M3 (minimax_m3_vl text_config) ----
    pub num_local_experts: Option<u32>,        // M3 name for expert_count
    pub dense_intermediate_size: Option<u32>,  // layers 0..2 dense FFN (12288)
    pub shared_intermediate_size: Option<u32>, // shared expert FF (3072)
    pub n_shared_experts: Option<u32>,
    pub rotary_dim: Option<u32>,               // partial RoPE (64 of head_dim 128)
    pub use_gemma_norm: Option<bool>,
    pub scoring_func: Option<String>,          // "sigmoid"
    pub routed_scaling_factor: Option<f32>,    // 2.0
    pub use_routing_bias: Option<bool>,
    pub swiglu_alpha: Option<f32>,             // swigluoai clamp params
    pub swiglu_limit: Option<f32>,
    pub moe_layer_freq: Option<Vec<u32>>,      // per-layer 0=dense 1=moe
    // ---- Hy3 (`hy_v3`) ----
    pub first_k_dense_replace: Option<u32>,
    pub moe_router_use_sigmoid: Option<bool>,
    pub moe_router_enable_expert_bias: Option<bool>,
    pub route_norm: Option<bool>,
    pub router_scaling_factor: Option<f32>,
    pub qk_norm: Option<bool>,
    pub hidden_act: Option<String>,
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
            mtp_num_hidden_layers: None,
            num_experts: None,
            num_experts_per_tok: None,
            moe_intermediate_size: None,
            expert_hidden_dim: None,
            shared_expert_intermediate_size: None,
            linear_conv_kernel_dim: None,
            linear_key_head_dim: None,
            linear_value_head_dim: None,
            linear_num_key_heads: None,
            linear_num_value_heads: None,
            num_local_experts: None,
            dense_intermediate_size: None,
            shared_intermediate_size: None,
            n_shared_experts: None,
            rotary_dim: None,
            use_gemma_norm: None,
            scoring_func: None,
            routed_scaling_factor: None,
            use_routing_bias: None,
            swiglu_alpha: None,
            swiglu_limit: None,
            moe_layer_freq: None,
            first_k_dense_replace: None,
            moe_router_use_sigmoid: None,
            moe_router_enable_expert_bias: None,
            route_norm: None,
            router_scaling_factor: None,
            qk_norm: None,
            hidden_act: None,
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
        if let Some(rp) = o.object("rope_parameters") {
            if let Some(v) = rp.f32("rope_theta") { self.rope_theta = v; }
        }
        if let Some(v) = o.u32("full_attention_interval") { self.full_attention_interval = Some(v); }
        if let Some(v) = o.u32("num_nextn_predict_layers") { self.num_nextn_predict_layers = Some(v); }
        if let Some(v) = o.u32("mtp_num_hidden_layers") { self.mtp_num_hidden_layers = Some(v); }
        if let Some(v) = o.u32("num_experts").or_else(|| o.u32("num_local_experts")) { self.num_experts = Some(v); }
        if let Some(v) = o.u32("num_experts_per_tok") { self.num_experts_per_tok = Some(v); }
        if let Some(v) = o.u32("moe_intermediate_size") { self.moe_intermediate_size = Some(v); }
        if let Some(v) = o.u32("expert_hidden_dim") { self.expert_hidden_dim = Some(v); }
        if let Some(v) = o.u32("shared_expert_intermediate_size") { self.shared_expert_intermediate_size = Some(v); }
        if let Some(v) = o.u32("linear_conv_kernel_dim") { self.linear_conv_kernel_dim = Some(v); }
        if let Some(v) = o.u32("linear_key_head_dim") { self.linear_key_head_dim = Some(v); }
        if let Some(v) = o.u32("linear_value_head_dim") { self.linear_value_head_dim = Some(v); }
        if let Some(v) = o.u32("linear_num_key_heads") { self.linear_num_key_heads = Some(v); }
        if let Some(v) = o.u32("linear_num_value_heads") { self.linear_num_value_heads = Some(v); }
        // ---- MiniMax-M3 keys ----
        if let Some(v) = o.u32("num_local_experts") { self.num_local_experts = Some(v); }
        if let Some(v) = o.u32("dense_intermediate_size") { self.dense_intermediate_size = Some(v); }
        if let Some(v) = o.u32("shared_intermediate_size") { self.shared_intermediate_size = Some(v); }
        if let Some(v) = o.u32("n_shared_experts").or_else(|| o.u32("num_shared_experts")) { self.n_shared_experts = Some(v); }
        if let Some(v) = o.u32("rotary_dim") { self.rotary_dim = Some(v); }
        if let Some(v) = o.boolean("use_gemma_norm") { self.use_gemma_norm = Some(v); }
        if let Some(v) = o.string("scoring_func") { self.scoring_func = Some(v); }
        if let Some(v) = o.f32("routed_scaling_factor") { self.routed_scaling_factor = Some(v); }
        if let Some(v) = o.boolean("use_routing_bias") { self.use_routing_bias = Some(v); }
        if let Some(v) = o.f32("swiglu_alpha") { self.swiglu_alpha = Some(v); }
        if let Some(v) = o.f32("swiglu_limit") { self.swiglu_limit = Some(v); }
        if let Some(v) = o.u32_array("moe_layer_freq") { self.moe_layer_freq = Some(v); }
        // ---- Hy3 keys ----
        if let Some(v) = o.u32("first_k_dense_replace") { self.first_k_dense_replace = Some(v); }
        if let Some(v) = o.boolean("moe_router_use_sigmoid") { self.moe_router_use_sigmoid = Some(v); }
        if let Some(v) = o.boolean("moe_router_enable_expert_bias") { self.moe_router_enable_expert_bias = Some(v); }
        if let Some(v) = o.boolean("route_norm") { self.route_norm = Some(v); }
        if let Some(v) = o.f32("router_scaling_factor") { self.router_scaling_factor = Some(v); }
        if let Some(v) = o.boolean("qk_norm") { self.qk_norm = Some(v); }
        if let Some(v) = o.string("hidden_act") { self.hidden_act = Some(v); }
    }
}

// ============================ minimal flat JSON object reader ============================
//
// config.json is a flat-ish object; we only need scalar fields + one level of nested object
// (text_config) + the architectures string array. Rather than add serde to bw24-gguf, parse
// the value-bearing tokens for the keys we care about. Nested objects/arrays are captured as
// raw substrings so they can be re-parsed on demand.

pub(crate) struct JsonObj {
    // key -> raw value substring (trimmed). Objects/arrays keep their braces/brackets.
    fields: std::collections::BTreeMap<String, String>,
}

impl JsonObj {
    pub(crate) fn parse(json: &str) -> Self {
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

    pub(crate) fn fields(&self) -> impl Iterator<Item = (&str, &str)> {
        self.fields.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }

    pub(crate) fn raw(&self, key: &str) -> Option<&str> {
        self.fields.get(key).map(|s| s.as_str())
    }

    pub(crate) fn string(&self, key: &str) -> Option<String> {
        let v = self.raw(key)?.trim();
        if v.starts_with('"') && v.ends_with('"') && v.len() >= 2 {
            Some(v[1..v.len() - 1].to_string())
        } else {
            None
        }
    }

    pub(crate) fn u32(&self, key: &str) -> Option<u32> {
        let v = self.raw(key)?.trim();
        if v == "null" { return None; }
        // accept integers (and floats that are whole, e.g. "8.0")
        v.parse::<u64>().ok().map(|x| x as u32)
            .or_else(|| v.parse::<f64>().ok().map(|x| x as u32))
    }

    pub(crate) fn u64(&self, key: &str) -> Option<u64> {
        let v = self.raw(key)?.trim();
        if v == "null" { return None; }
        v.parse::<u64>().ok()
            .or_else(|| v.parse::<f64>().ok().map(|x| x as u64))
    }

    pub(crate) fn f32(&self, key: &str) -> Option<f32> {
        let v = self.raw(key)?.trim();
        if v == "null" { return None; }
        v.parse::<f32>().ok()
    }

    pub(crate) fn boolean(&self, key: &str) -> Option<bool> {
        match self.raw(key)?.trim() { "true" => Some(true), "false" => Some(false), _ => None }
    }

    /// Integer array field (e.g. moe_layer_freq: [0,0,0,1,...]).
    pub(crate) fn u32_array(&self, key: &str) -> Option<Vec<u32>> {
        let v = self.raw(key)?.trim();
        if !v.starts_with('[') || !v.ends_with(']') { return None; }
        Some(v[1..v.len()-1].split(',')
            .filter_map(|x| x.trim().parse::<u32>().ok()).collect())
    }

    pub(crate) fn u64_array(&self, key: &str) -> Option<Vec<u64>> {
        let v = self.raw(key)?.trim();
        if !v.starts_with('[') || !v.ends_with(']') { return None; }
        Some(v[1..v.len()-1].split(',')
            .filter_map(|x| x.trim().parse::<u64>().ok()).collect())
    }

    pub(crate) fn object(&self, key: &str) -> Option<JsonObj> {
        let v = self.raw(key)?.trim();
        if v.starts_with('{') { Some(JsonObj::parse(v)) } else { None }
    }

    /// First string element of a string array field (e.g. architectures[0]).
    pub(crate) fn first_string_in_array(&self, key: &str) -> Option<String> {
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

    #[test]
    fn parse_hy3_reap_config() {
        let json = r#"{
          "model_type": "hy_v3",
          "num_hidden_layers": 80,
          "hidden_size": 4096,
          "num_attention_heads": 64,
          "num_key_value_heads": 8,
          "head_dim": 128,
          "intermediate_size": 13312,
          "vocab_size": 120832,
          "max_position_embeddings": 262144,
          "rms_norm_eps": 1e-05,
          "rope_parameters": {"rope_theta": 11158840.0, "rope_type": "default"},
          "num_nextn_predict_layers": 1,
          "num_experts": 96,
          "num_experts_per_tok": 8,
          "moe_intermediate_size": 1536,
          "expert_hidden_dim": 1536,
          "num_shared_experts": 1,
          "moe_router_use_sigmoid": true,
          "moe_router_enable_expert_bias": true,
          "route_norm": true,
          "router_scaling_factor": 2.826,
          "qk_norm": true,
          "hidden_act": "silu"
        }"#;
        let c = HfConfig::parse(json);
        assert_eq!(Arch::from_hf_model_type(&c.model_type), Arch::Hy3);
        assert!((c.rope_theta - 11_158_840.0).abs() < 1.0);
        let mc = ModelConfig::from_hf(&c);
        assert_eq!(mc.arch, Arch::Hy3);
        assert!(mc.arch.is_moe());
        // Degenerate hybrid (the M3 class): rides HybridModel with every layer full-attention.
        assert!(mc.arch.is_hybrid());
        assert_eq!(mc.full_attention_interval, 0, "Hy3 has no linear-attention layers");
        assert!(!mc.attn_out_gate(), "Hy3 wq has no fused [q|gate] output gate");
        let (sf, norm) = mc.sigmoid_router().expect("Hy3 routes with sigmoid");
        assert!((sf - 2.826).abs() < 1e-6);
        assert!(norm);
        assert_eq!(mc.n_layer, 81, "HF config convention includes the appended MTP block");
        assert_eq!(mc.nextn_predict_layers, 1);
        assert_eq!(mc.n_embd, 4096);
        assert_eq!(mc.n_head, 64);
        assert_eq!(mc.n_head_kv, 8);
        assert_eq!(mc.head_dim_k, 128);
        assert_eq!(mc.n_ff, 13312);
        assert_eq!(mc.n_vocab, 120832);
        assert_eq!(mc.context_length, 262144);
        assert_eq!(mc.rope_dim_count, 128);
        let moe = mc.moe.as_ref().unwrap();
        assert_eq!(moe.expert_count, 96);
        assert_eq!(moe.expert_used_count, 8);
        assert_eq!(moe.expert_ff_length, 1536);
        assert_eq!(moe.expert_shared_ff_length, 1536);
        let hy3 = mc.hy3.as_ref().unwrap();
        assert!(hy3.sigmoid_routing);
        assert!(hy3.use_routing_bias);
        assert!(hy3.route_norm);
        assert!((hy3.router_scaling_factor - 2.826).abs() < 1e-6);
        assert_eq!(hy3.n_shared_experts, 1);
        assert_eq!(hy3.first_k_dense_replace, 1);
        assert!(hy3.qk_norm);
        assert_eq!(hy3.hidden_act, "silu");
    }
}

#[cfg(test)]
mod minimax_tests {
    use super::*;
    #[test]
    fn parse_minimax_m3_vl() {
        let cfg = HfConfig::parse(&std::fs::read_to_string(
            "/data/ai-ml/hf-models/minimax-m3-nvfp4-reap50/config.json").unwrap());
        assert_eq!(Arch::from_hf_model_type(&cfg.model_type), Arch::MinimaxM3);
        assert_eq!(cfg.num_hidden_layers, 60);
        assert_eq!(cfg.num_local_experts, Some(64));   // REAP50 artifact
        assert_eq!(cfg.num_experts_per_tok, Some(4));
        assert_eq!(cfg.hidden_size, 6144);
        assert_eq!(cfg.dense_intermediate_size, Some(12288));
        assert_eq!(cfg.shared_intermediate_size, Some(3072));
        assert_eq!(cfg.rotary_dim, Some(64));
        assert_eq!(cfg.use_gemma_norm, Some(true));
        assert_eq!(cfg.scoring_func.as_deref(), Some("sigmoid"));
        assert_eq!(cfg.routed_scaling_factor, Some(2.0));
        assert_eq!(cfg.moe_layer_freq.as_ref().map(|v| (v.len(), v[0], v[3])), Some((60, 0, 1)));
        let mc = ModelConfig::from_hf(&cfg);
        assert!(mc.arch.is_moe() && mc.arch.is_minimax());
        assert_eq!(mc.moe.as_ref().unwrap().expert_count, 64);
        assert_eq!(mc.moe.as_ref().unwrap().expert_shared_ff_length, 3072);
        assert_eq!(mc.rope_dim_count, 64);   // partial RoPE from rotary_dim
        let m3 = mc.m3.as_ref().unwrap();
        assert!(m3.use_gemma_norm && m3.sigmoid_routing && m3.use_routing_bias);
        assert_eq!(m3.routed_scaling_factor, 2.0);
        assert_eq!(m3.n_shared_experts, 1);
        assert_eq!((m3.swiglu_alpha, m3.swiglu_limit), (1.702, 7.0));
        assert_eq!(m3.dense_intermediate_size, 12288);
        assert_eq!(m3.moe_layer_freq.iter().filter(|&&x| x == 0).count(), 3); // 3 dense layers
    }

    /// Name-mapping against the REAL REAP50 shard index: every text-model tensor pattern the
    /// loader will request must resolve to a name present in the safetensors index.
    #[test]
    fn minimax_name_mapping_against_index() {
        use crate::hf_mapping::{ggml_to_hf, hf_expert_name, resolve_ggml, HfTarget};
        let cfg = ModelConfig::from_hf(&HfConfig::parse(&std::fs::read_to_string(
            "/data/ai-ml/hf-models/minimax-m3-nvfp4-reap50/config.json").unwrap()));
        let idx: std::collections::HashSet<String> = {
            let txt = std::fs::read_to_string(
                "/data/ai-ml/hf-models/minimax-m3-nvfp4-reap50/model.safetensors.index.json").unwrap();
            // crude but sufficient: harvest every JSON key that looks like a tensor name
            txt.split('"').filter(|s| s.contains('.') && !s.contains(' '))
                .map(|s| s.to_string()).collect()
        };
        // the VL wrapper prefixes the text model with `language_model.` — the source's lookup()
        // fallback strips/adds it; here emulate that for the assertion.
        let has = |hf: &str| idx.contains(hf) || idx.contains(&format!("language_model.{hf}"));

        // top-level + dense attention/norm names (layer 0 = dense-FFN layer, layer 3 = MoE)
        for g in ["token_embd.weight", "output_norm.weight", "output.weight"] {
            let hf = ggml_to_hf(g, &cfg.arch).unwrap();
            assert!(has(&hf), "{g} -> {hf} not in index");
        }
        for g in ["blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight",
                  "blk.0.attn_output.weight", "blk.0.attn_q_norm.weight", "blk.0.attn_k_norm.weight",
                  "blk.0.attn_norm.weight", "blk.0.ffn_norm.weight",
                  "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight",
                  "blk.3.ffn_gate_inp.weight", "blk.3.exp_probs_b.bias",
                  "blk.3.ffn_gate_shexp.weight", "blk.3.ffn_up_shexp.weight",
                  "blk.3.ffn_down_shexp.weight"] {
            let hf = ggml_to_hf(g, &cfg.arch).unwrap_or_else(|| panic!("{g} unmapped"));
            assert!(has(&hf), "{g} -> {hf} not in index");
        }
        // Mixtral-style per-expert names (w1=gate, w2=down, w3=up)
        for proj in ["gate", "down", "up"] {
            let hf = hf_expert_name(3, 63, proj, &cfg.arch);
            assert!(has(&hf), "expert {proj} -> {hf} not in index");
        }
        // gemma-norm fold: norms must resolve through the Transform(NormPlusOne) arm
        match resolve_ggml("blk.0.attn_norm.weight", &cfg) {
            Some(HfTarget::Transform { kind: crate::hf_mapping::TransformKind::NormPlusOne, .. }) => {}
            _ => panic!("gemma-norm fold not applied to attn_norm"),
        }
    }
}
