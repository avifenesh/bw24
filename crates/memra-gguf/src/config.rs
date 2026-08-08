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
    GlmDsa,       // GLM-5/5.2: MLA attention (latent KV, MQA decode) + DSA sparse indexer +
                  // deepseek-style MoE (sigmoid router + noaux_tc bias) + 1 NextN/MTP layer
    Step35,       // StepFun Step-3.5/3.7-Flash: SWA(512) 3:1 + PER-LAYER q-head count (64 full /
                  // 96 swa), head-wise attn gate (separate `attn_gate` tensor), dual rope base,
                  // half-rotary on full layers, 288-expert sigmoid-router MoE + shared expert,
                  // per-layer swiglu clamp arrays, 3 NextN/MTP blocks (shipped in a separate GGUF)
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
            // upstream llama.cpp writes the hybrid class as qwen3next (round 46: needed to
            // load public 27B GGUFs on the sm_90a board — same layer stack as qwen35).
            "qwen3next" => Arch::Qwen35,
            "qwen3nextmoe" => Arch::Qwen35Moe,
            "olmoe" => Arch::Olmoe,
            "minimax-m3" => Arch::MinimaxM3,
            "hy3" => Arch::Hy3,
            "gemma4" => Arch::Gemma4,
            "glm-dsa" => Arch::GlmDsa,
            // StepFun writes 3.5 AND 3.7-Flash under the same arch name (upstream llama.cpp
            // `step35`, PR #23845/#19283 — 3.7 is the 196B-A11B sibling of 3.5).
            "step35" => Arch::Step35,
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
            "qwen3_5_moe" | "qwen3_5_moe_text" | "qwen3_next_moe" => "qwen35moe",
            "olmoe" => "olmoe",
            // MiniMax-M3 (incl the VL wrapper model_type; text_config flattening handles the rest)
            "minimax_m3" | "minimax_m3_vl" | "minimax_m3_text" => "minimax-m3",
            "hy_v3" | "hy3" => "hy3",
            // GLM-5/5.2 (HF `GlmMoeDsaForCausalLM`, model_type `glm_moe_dsa`)
            "glm_moe_dsa" => "glm-dsa",
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
        matches!(self, Arch::Qwen35 | Arch::Qwen35Moe | Arch::MinimaxM3 | Arch::Hy3 | Arch::Gemma4
                     | Arch::GlmDsa | Arch::Step35)
    }
    /// True for arches with a routed-expert FFN. `Olmoe` is dense-attention + MoE-FFN.
    pub fn is_moe(&self) -> bool {
        matches!(self, Arch::Qwen3Moe | Arch::Qwen35Moe | Arch::Olmoe | Arch::MinimaxM3 | Arch::Hy3
                     | Arch::Gemma4 | Arch::GlmDsa | Arch::Step35)
    }
    /// MiniMax-M3: sigmoid router (+e_score_correction_bias), gemma-norm, swigluoai clamp,
    /// Mixtral-style expert tensor names. Full attention v0 (MSA is bit-exact-degenerate <=2048
    /// ctx — the sparse indexer selects everything; the MSA kernel is a later arc).
    pub fn is_minimax(&self) -> bool { matches!(self, Arch::MinimaxM3) }
    /// Tencent HunYuan/Hunyuan Hy3 (`hy_v3` in HF config.json).
    pub fn is_hy3(&self) -> bool { matches!(self, Arch::Hy3) }
    /// StepFun Step-3.5 / Step-3.7-Flash (GGUF arch `step35`).
    pub fn is_step35(&self) -> bool { matches!(self, Arch::Step35) }
}

/// What kind of token-mixing a given layer performs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    FullAttention,   // softmax attention with growing KV cache
    LinearAttention, // gated-deltanet / SSM with fixed recurrent state
}

/// How an attention layer gates its output before the output projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionGateKind {
    None,
    /// Qwen3.5 packs a per-dimension sigmoid gate beside Q in `attn_q.weight`.
    FusedQ,
    /// Step35 projects one sigmoid scalar per head through `attn_gate.weight`.
    SeparateHead,
}

/// Complete attention geometry for one architecture-defined layer class.
///
/// The table is intentionally small: it holds values that execution arms otherwise reconstruct
/// independently. Architecture-specific math, tensor names, clamps, and routing stay in their
/// existing configs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerGeometry {
    pub mixer: LayerKind,
    pub n_head: u32,
    pub n_head_kv: u32,
    pub head_dim_k: u32,
    pub head_dim_v: u32,
    pub n_rot: u32,
    pub rope_base: f32,
    pub window: Option<u32>,
    pub rope_factors: bool,
    pub attention_gate: AttentionGateKind,
}

impl LayerGeometry {
    pub fn attention_scale(self) -> f32 {
        1.0 / (self.head_dim_k as f32).sqrt()
    }
}

/// Declarative per-architecture geometry: a compact class table plus one class id per layer.
///
/// Qwen3.5 and Step35 are the first migrated architectures. Other architectures keep their
/// existing scalar/per-arch config paths until they are deliberately migrated.
#[derive(Debug, Clone)]
pub struct ArchGeometryTable {
    classes: Vec<LayerGeometry>,
    layer_classes: Vec<u16>,
}

impl ArchGeometryTable {
    fn qwen35(
        n_layer: u32,
        nextn: u32,
        full_attention_interval: u32,
        n_head: u32,
        n_head_kv: u32,
        head_dim_k: u32,
        head_dim_v: u32,
        n_rot: u32,
        rope_base: f32,
    ) -> Self {
        let linear = LayerGeometry {
            mixer: LayerKind::LinearAttention,
            n_head,
            n_head_kv,
            head_dim_k,
            head_dim_v,
            n_rot,
            rope_base,
            window: None,
            rope_factors: false,
            attention_gate: AttentionGateKind::None,
        };
        let full = LayerGeometry {
            mixer: LayerKind::FullAttention,
            attention_gate: AttentionGateKind::FusedQ,
            ..linear
        };
        let n_trunk = n_layer.saturating_sub(nextn);
        let layer_classes = (0..n_layer)
            .map(|il| {
                let full_layer = il >= n_trunk
                    || full_attention_interval == 0
                    || (il + 1) % full_attention_interval == 0;
                if full_layer { 1 } else { 0 }
            })
            .collect();
        Self { classes: vec![linear, full], layer_classes }
    }

    fn step35(
        n_layer: u32,
        head_dim_k: u32,
        head_dim_v: u32,
        step35: &Step35Config,
    ) -> Self {
        let mut classes = Vec::new();
        let mut layer_classes = Vec::with_capacity(n_layer as usize);
        for il in 0..n_layer {
            let swa = step35.is_swa(il);
            let geometry = LayerGeometry {
                mixer: LayerKind::FullAttention,
                n_head: step35.n_head(il),
                n_head_kv: step35.n_head_kv(il),
                head_dim_k,
                head_dim_v,
                n_rot: step35.n_rot(il),
                rope_base: step35.rope_base(il),
                window: swa.then_some(step35.sliding_window),
                rope_factors: !swa,
                attention_gate: AttentionGateKind::SeparateHead,
            };
            let class = match classes.iter().position(|candidate| *candidate == geometry) {
                Some(class) => class,
                None => {
                    classes.push(geometry);
                    classes.len() - 1
                }
            };
            assert!(class <= u16::MAX as usize, "too many architecture geometry classes");
            layer_classes.push(class as u16);
        }
        Self { classes, layer_classes }
    }

    pub fn classes(&self) -> &[LayerGeometry] {
        &self.classes
    }

    pub fn layer_classes(&self) -> &[u16] {
        &self.layer_classes
    }

    pub fn layer(&self, il: u32) -> Option<LayerGeometry> {
        let class = *self.layer_classes.get(il as usize)? as usize;
        self.classes.get(class).copied()
    }
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
    // ---- E4B (per-layer-embedding + KV-sharing variant; 0 on 26B/31B) ----
    /// n_embd_per_layer (E4B: 256). 0 = no per-layer-embedding machinery.
    pub n_embd_per_layer: u32,
    /// trailing layers WITHOUT own KV (E4B: 18); they attend an earlier layer's cache:
    /// il >= n_layer - shared_kv_layers reads layer (n_layer - shared_kv_layers) - (swa ? 2 : 1).
    pub shared_kv_layers: u32,
    /// tokenizer.ggml.suppress_tokens — ids the model card forbids at sampling (the 12B QAT
    /// ships two control ids); empty on 26B/31B/E4B. Masked to -inf before every argmax/sample.
    pub suppress_tokens: Vec<u32>,
}

/// StepFun Step-3.5/3.7-Flash (`step35`) per-layer geometry + block extras. Values in comments are
/// the 3.7-Flash 196B-A11B artifact (official IQ4_XS GGUF header, receipt
/// `research/step37-bringup-20260802/raw/gguf-header-stepfun-iq4xs-shard1-20260802.txt`).
///
/// Reference semantics: upstream llama.cpp `src/models/step35.cpp` (PR #23845, merged 2026-06-02)
/// + `llama-hparams.cpp` `n_rot()`/`is_swa()`. Three things make this arch different from every
/// arch memra already loads, and all three are per-LAYER:
///   1. `n_head` is an ARRAY (64 on full-attn layers, 96 on SWA layers) — KV heads are uniform 8,
///      so KV geometry is unaffected, but wq/wo/attn_gate out-features vary per layer.
///   2. RoPE: dual base (5e6 full / 1e4 SWA) AND half-rotary on the FULL layers only
///      (`n_rot_full = n_rot_full/2` = 64 of head_dim 128; SWA keeps the full 128).
///      `rope_freqs.weight` (llama3 factors) applies to the FULL layers only — SWA passes null.
///   3. The head-wise attention gate is a SEPARATE tensor `blk.N.attn_gate.weight [n_embd, n_head]`
///      producing ONE sigmoid scalar per head (broadcast over head_dim), NOT the qwen35 form where
///      the gate is fused into wq as a per-dim block. `ModelConfig::attn_out_gate()` must be false
///      for this arch or the wq split reads 2x out of bounds.
#[derive(Debug, Clone)]
pub struct Step35Config {
    /// Per-layer query-head count — `step35.attention.head_count` is an ARRAY (45 items: 64 on
    /// full-attn layers, 96 on SWA layers). len == n_layer_total (the MTP GGUF carries 48).
    pub head_count: Vec<u32>,
    /// Per-layer KV-head count (`attention.head_count_kv`, uniform 8 on 3.7 — kept as an array
    /// because the key IS an array in the artifact and a future sibling may vary it).
    pub head_count_kv: Vec<u32>,
    /// `attention.sliding_window_pattern` [bool; n_layer]: true = sliding-window layer.
    /// 3.7-Flash is 3:1 — [false,true,true,true] repeating = 12 full (il%4==0) + 33 SWA.
    pub swa_pattern: Vec<bool>,
    pub sliding_window: u32,          // attention.sliding_window = 512
    pub rope_base_global: f32,        // rope.freq_base = 5e6 (full-attn layers)
    pub rope_base_swa: f32,           // rope.freq_base_swa = 1e4 (SWA layers)
    /// Rotary dims on FULL-attn layers = head_dim_k/2 (64). Upstream halves `n_rot_full` in
    /// `load_arch_hparams` AFTER the generic loader defaults it to `n_embd_head_k` (128).
    pub rope_dims_full: u32,
    /// Rotary dims on SWA layers = head_dim_k (128, unhalved — `n_rot_swa` is copied from
    /// `n_rot_full` BEFORE the arch hook halves it, so SWA keeps the full width).
    pub rope_dims_swa: u32,
    /// `swiglu_clamp_exp` [f32; n_layer] — routed-expert SwiGLU clamp limit per layer.
    /// Nonzero only on layers 43-44 of 3.7-Flash. Semantics (llama-graph.cpp:2146): the limit
    /// applies when > 1e-6 as `up = clamp(up, -L, L); act = min(silu(gate), L); out = act * up`.
    pub swiglu_clamp_exp: Vec<f32>,
    /// `swiglu_clamp_shexp` [f32; n_layer] — same for the shared expert (llama-graph.cpp:1751).
    pub swiglu_clamp_shexp: Vec<f32>,
    // ---- MoE (deepseek-V3-class sigmoid router; the Hy3/M3/glm-dsa recipe verbatim) ----
    pub sigmoid_routing: bool,        // expert_gating_func == 2; ABSENT defaults to sigmoid (BC)
    pub routed_scaling_factor: f32,   // expert_weights_scale = 3.0
    pub route_norm: bool,             // expert_weights_norm = true
    pub first_k_dense_replace: u32,   // leading_dense_block_count = 3
}

impl Step35Config {
    /// True when layer `il` is a sliding-window layer. Out-of-range indices (the MTP blocks of a
    /// trunk-only GGUF) fall back to `true`: upstream's `is_swa_impl` array covers n_layer_all and
    /// every 3.7 MTP block is SWA-type (blocks 45/46/47, none at il%4==0).
    pub fn is_swa(&self, il: u32) -> bool {
        self.swa_pattern.get(il as usize).copied().unwrap_or(true)
    }
    /// Query-head count for layer `il` (64 full / 96 SWA on 3.7-Flash).
    pub fn n_head(&self, il: u32) -> u32 {
        self.head_count.get(il as usize).copied()
            .or_else(|| self.head_count.last().copied())
            .expect("step35: attention.head_count array is empty")
    }
    /// KV-head count for layer `il` (uniform 8 on 3.7-Flash).
    pub fn n_head_kv(&self, il: u32) -> u32 {
        self.head_count_kv.get(il as usize).copied()
            .or_else(|| self.head_count_kv.last().copied())
            .expect("step35: attention.head_count_kv array is empty")
    }
    /// Rotary width for layer `il` — upstream `llama_hparams::n_rot(il)`:
    /// `is_swa(il) ? n_rot_swa : n_rot_full` (128 SWA / 64 full on 3.7-Flash).
    pub fn n_rot(&self, il: u32) -> u32 {
        if self.is_swa(il) { self.rope_dims_swa } else { self.rope_dims_full }
    }
    /// RoPE base for layer `il` (1e4 SWA / 5e6 full).
    pub fn rope_base(&self, il: u32) -> f32 {
        if self.is_swa(il) { self.rope_base_swa } else { self.rope_base_global }
    }
    /// Routed-expert SwiGLU clamp for layer `il`, `None` when unset/<=eps (upstream uses a 1e-6
    /// epsilon, not != 0.0 — a tiny nonzero limit must not silently clamp everything to ~0).
    pub fn clamp_exp(&self, il: u32) -> Option<f32> {
        self.swiglu_clamp_exp.get(il as usize).copied().filter(|&l| l > 1e-6)
    }
    /// Shared-expert SwiGLU clamp for layer `il`.
    pub fn clamp_shexp(&self, il: u32) -> Option<f32> {
        self.swiglu_clamp_shexp.get(il as usize).copied().filter(|&l| l > 1e-6)
    }
    /// Count of full-attention (non-SWA) layers over the trunk — the layers whose KV cache grows
    /// unbounded with context. 12 on 3.7-Flash; the KV-budget arithmetic keys off this.
    pub fn n_full_attn(&self, n_trunk: u32) -> u32 {
        (0..n_trunk).filter(|&il| !self.is_swa(il)).count() as u32
    }
}

/// DSA (DeepSeek Sparse Attention) lightning-indexer geometry (GLM-5.2). Parsed when the GGUF
/// carries the `attention.indexer.*` keys; consumed by increment 6 (indexer arm). GLM-5.2:
/// 32 heads x 128 (64 rope + 64 nope), top-k 2048, 21 "full" layers (own top-k) + 57 "shared"
/// (reuse the previous full layer's indices — IndexShare/IndexCache, arXiv 2603.12201).
#[derive(Debug, Clone)]
pub struct DsaConfig {
    pub index_n_heads: u32,      // glm-dsa.attention.indexer.head_count (32)
    pub index_head_dim: u32,     // glm-dsa.attention.indexer.key_length (128)
    pub index_top_k: u32,        // glm-dsa.attention.indexer.top_k (2048)
    /// Per TRUNK layer: true = "full" indexer layer (own top-k selection + indexer tensors),
    /// false = "shared" (reuses the previous full layer's top-k; NO indexer tensors in the GGUF).
    /// glm-dsa.attention.indexer.types, [bool; n_trunk]. Empty if the key is absent (pre-5.2
    /// GLM GGUFs: all layers full — llama.cpp BC default).
    pub indexer_full: Vec<bool>,
}

/// llama.cpp `GLM_5_2_DEFAULT_INDEXER_TYPES` (glm-dsa.cpp): full-indexer layers at 0,1 then
/// every 4th from 2 — {0,1,2,6,10,...} — i.e. 21 full / 57 shared over 78 trunk layers. This is
/// NOT just BC: the 2026-06 unsloth GLM-5.2 GGUFs ship WITHOUT the `attention.indexer.types`
/// key (verified from the artifact header 2026-08-01, research/mla-inc2-20260801/ARTIFACT.md),
/// so the default table is what actually drives layer classification on the real artifact.
pub fn glm52_default_indexer_types(n_trunk: usize) -> Vec<bool> {
    (0..n_trunk).map(|i| i < 2 || (i - 2) % 4 == 0).collect()
}

/// MLA (multi-head latent attention) geometry + glm-dsa router knobs, parsed from the GGUF keys
/// the llama.cpp converter writes (pinned in research/mla-bringup-20260801/RECEIPTS.md §5).
/// GLM-5.2 values in comments. The latent KV-cache row is `latent_dim()` = kv_lora_rank +
/// qk_rope_head_dim (576); V is the first `kv_lora_rank` (512) elements of the SAME row.
#[derive(Debug, Clone)]
pub struct MlaConfig {
    pub q_lora_rank: u32,        // glm-dsa.attention.q_lora_rank (2048)
    pub kv_lora_rank: u32,       // glm-dsa.attention.kv_lora_rank (512)
    /// Per-head qk dim AFTER decompression (nope + rope) — glm-dsa.attention.key_length_mla (256).
    /// The softmax scale is 1/sqrt(THIS), not of the absorbed 576 width (DESIGN.md §1.3).
    pub qk_head_dim: u32,
    pub qk_nope_head_dim: u32,   // qk_head_dim - qk_rope_head_dim (192)
    pub qk_rope_head_dim: u32,   // glm-dsa.rope.dimension_count (64)
    /// Per-head v dim after decompression — glm-dsa.attention.value_length_mla (256).
    pub v_head_dim: u32,
    // ---- deepseek-style sigmoid router (the Hy3/M3-class knobs, glm-dsa key names) ----
    pub sigmoid_routing: bool,       // expert_gating_func == 2 (sigmoid); absent => sigmoid (BC)
    pub routed_scaling_factor: f32,  // glm-dsa.expert_weights_scale (2.5)
    pub route_norm: bool,            // glm-dsa.expert_weights_norm (norm_topk_prob: true)
    pub n_shared_experts: u32,       // glm-dsa.expert_shared_count (1)
    pub first_k_dense_replace: u32,  // glm-dsa.leading_dense_block_count (3)
    // ---- DSA indexer (None when the GGUF carries no indexer keys) ----
    pub dsa: Option<DsaConfig>,
}

impl MlaConfig {
    /// Latent KV-cache row width: [rmsnorm(c_kv) | rope(k_pe)] — 576 on GLM-5.2. This is what
    /// `attention.key_length` carries in a glm-dsa GGUF (cross-checked at parse).
    pub fn latent_dim(&self) -> u32 { self.kv_lora_rank + self.qk_rope_head_dim }
    /// The V view is the first kv_lora_rank (512) elements of each latent row — what
    /// `attention.value_length` carries in a glm-dsa GGUF. No separate V plane exists.
    pub fn v_view_dim(&self) -> u32 { self.kv_lora_rank }
    /// Softmax scale: 1/sqrt(qk_head_dim) = 1/16 on GLM-5.2 (mscale = 1, no yarn).
    pub fn scale(&self) -> f32 { 1.0 / (self.qk_head_dim as f32).sqrt() }
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
    // MLA extras — glm-dsa only (None for every other arch)
    pub mla: Option<MlaConfig>,
    // Step-3.5/3.7-Flash extras — `step35` only (None for every other arch)
    pub step35: Option<Step35Config>,
    // Declarative per-layer geometry for migrated architectures.
    pub geometry: Option<ArchGeometryTable>,
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
                    n_embd_per_layer: u("embedding_length_per_layer_input").unwrap_or(0),
                    shared_kv_layers: u("attention.shared_kv_layers").unwrap_or(0),
                    suppress_tokens: match g.metadata.get("tokenizer.ggml.suppress_tokens") {
                        Some(MetaValue::Array(a)) =>
                            a.iter().filter_map(|v| v.as_u64().map(|x| x as u32)).collect(),
                        _ => Vec::new(),
                    },
                })
            } else { None };

        // step35 (Step-3.5/3.7-Flash). Reference: upstream `src/models/step35.cpp`
        // `load_arch_hparams` + `llama-model.cpp:1190-1235` (the generic n_rot defaulting that
        // runs BEFORE the arch hook halves n_rot_full).
        let step35 = if matches!(&arch, Arch::Step35) {
            let arr_u = |k: &str| -> Vec<u32> {
                match g.meta_arch(k) {
                    Some(MetaValue::Array(a)) =>
                        a.iter().filter_map(|v| v.as_u64().map(|x| x as u32)).collect(),
                    // The key may legitimately be a SCALAR (upstream `get_key_or_arr` accepts
                    // both, and a uniform-geometry sibling would write one) — broadcast it.
                    Some(v) => match v.as_u64() {
                        Some(x) => vec![x as u32; n_layer as usize],
                        None => Vec::new(),
                    },
                    None => Vec::new(),
                }
            };
            let arr_f = |k: &str| -> Vec<f32> {
                match g.meta_arch(k) {
                    Some(MetaValue::Array(a)) => a.iter().filter_map(|v| v.as_f32()).collect(),
                    Some(v) => match v.as_f32() {
                        Some(x) => vec![x; n_layer as usize],
                        None => Vec::new(),
                    },
                    None => Vec::new(),
                }
            };
            let head_count = arr_u("attention.head_count");
            assert!(!head_count.is_empty(), "step35: attention.head_count missing");
            // The array covers every block INCLUDING the MTP ones (the 3.7 trunk GGUF writes 45,
            // the standalone MTP GGUF writes 48 = 45 trunk + 3 nextn). Short arrays are a
            // mis-converted file — a silent last-value broadcast would give the wrong wq width.
            assert!(head_count.len() as u32 >= n_layer,
                "step35: attention.head_count has {} entries, need >= block_count {n_layer}",
                head_count.len());
            let swa_pattern: Vec<bool> = match g.meta_arch("attention.sliding_window_pattern") {
                // The artifact writes a BOOL array (llama.cpp `get_key_or_arr` into is_swa_impl).
                // `as_u64` maps Bool -> 0/1, so one reader covers bool and int serializations.
                Some(MetaValue::Array(a)) =>
                    a.iter().filter_map(|v| v.as_u64().map(|x| x != 0)).collect(),
                // Scalar form = llama.cpp's n_pattern convention: layer il is SWA unless
                // il % n_pattern == 0 (llama-hparams.cpp:11, set_swa_pattern).
                Some(v) => match v.as_u64() {
                    Some(np) if np > 0 =>
                        (0..n_layer).map(|il| il as u64 % np != 0).collect(),
                    _ => Vec::new(),
                },
                None => Vec::new(),
            };
            assert!(swa_pattern.len() as u32 >= n_layer,
                "step35: attention.sliding_window_pattern has {} entries, need >= {n_layer}",
                swa_pattern.len());
            // Upstream: the generic loader sets n_rot_full = attention.key_length (128), copies
            // n_rot_swa from it, THEN step35.cpp halves n_rot_full -> 64. So SWA = 128, full = 64.
            let rope_dims_swa = u("rope.dimension_count").unwrap_or(head_dim_k);
            Some(Step35Config {
                head_count,
                head_count_kv: {
                    let kv = arr_u("attention.head_count_kv");
                    assert!(kv.len() as u32 >= n_layer,
                        "step35: attention.head_count_kv has {} entries, need >= {n_layer}",
                        kv.len());
                    kv
                },
                swa_pattern,
                sliding_window: u("attention.sliding_window")
                    .expect("step35: attention.sliding_window (SWA layers need a window)"),
                rope_base_global: f("rope.freq_base").unwrap_or(10000.0),
                // ABSENT => same as global (upstream get_key(..., false) leaves the copied value).
                rope_base_swa: f("rope.freq_base_swa")
                    .unwrap_or_else(|| f("rope.freq_base").unwrap_or(10000.0)),
                rope_dims_full: rope_dims_swa / 2,
                rope_dims_swa,
                swiglu_clamp_exp: arr_f("swiglu_clamp_exp"),
                swiglu_clamp_shexp: arr_f("swiglu_clamp_shexp"),
                // expert_gating_func 2 = sigmoid; ABSENT defaults to sigmoid (step35.cpp:19-21).
                sigmoid_routing: u("expert_gating_func").map(|v| v == 2).unwrap_or(true),
                routed_scaling_factor: f("expert_weights_scale").unwrap_or(1.0),
                route_norm: u("expert_weights_norm").map(|v| v != 0).unwrap_or(false),
                first_k_dense_replace: u("leading_dense_block_count").unwrap_or(0),
            })
        } else { None };

        // glm-dsa MLA + DSA keys (RECEIPTS.md §5 — the exact set the llama.cpp converter writes).
        let mla = if matches!(&arch, Arch::GlmDsa) {
            let q_lora_rank = u("attention.q_lora_rank").expect("glm-dsa: attention.q_lora_rank");
            let kv_lora_rank = u("attention.kv_lora_rank").expect("glm-dsa: attention.kv_lora_rank");
            let qk_head_dim =
                u("attention.key_length_mla").expect("glm-dsa: attention.key_length_mla");
            let v_head_dim =
                u("attention.value_length_mla").expect("glm-dsa: attention.value_length_mla");
            let qk_rope_head_dim =
                u("rope.dimension_count").expect("glm-dsa: rope.dimension_count");
            assert!(qk_rope_head_dim < qk_head_dim,
                    "glm-dsa: rope dim {qk_rope_head_dim} >= qk head dim {qk_head_dim}");
            // Cross-checks (DESIGN.md §3.1): attention.key_length is the LATENT cache row
            // (kv_lora_rank + rope), attention.value_length its V prefix view (kv_lora_rank).
            // A projection-wide mismatch here means a mis-converted GGUF — fail at load, loudly.
            if let Some(kl) = u("attention.key_length") {
                assert_eq!(kl, kv_lora_rank + qk_rope_head_dim,
                    "glm-dsa: attention.key_length {kl} != kv_lora_rank + rope {}",
                    kv_lora_rank + qk_rope_head_dim);
            }
            if let Some(vl) = u("attention.value_length") {
                assert_eq!(vl, kv_lora_rank,
                    "glm-dsa: attention.value_length {vl} != kv_lora_rank {kv_lora_rank}");
            }
            // Router: expert_gating_func 2 = sigmoid; ABSENT defaults to sigmoid (llama.cpp
            // glm-dsa BC — load_arch_hparams maps NONE -> SIGMOID).
            let sigmoid_routing = u("expert_gating_func").map(|v| v == 2).unwrap_or(true);
            // DSA indexer: present iff the converter wrote the indexer keys (GLM-5/5.1/5.2 all do).
            let dsa = u("attention.indexer.head_count").map(|index_n_heads| DsaConfig {
                index_n_heads,
                index_head_dim: u("attention.indexer.key_length")
                    .expect("glm-dsa: attention.indexer.key_length"),
                index_top_k: u("attention.indexer.top_k")
                    .expect("glm-dsa: attention.indexer.top_k"),
                indexer_full: match g.meta_arch("attention.indexer.types") {
                    Some(MetaValue::Array(a)) =>
                        a.iter().filter_map(|v| v.as_u64().map(|x| x != 0)).collect(),
                    // Key absent (the real 2026-06 unsloth GLM-5.2 GGUF!): llama.cpp BC —
                    // 5.2-class (ctx >= 1M) takes the hardcoded 21-full/57-shared table,
                    // pre-5.2 GLM (ctx < 1M) is all-full.
                    _ => {
                        let n_trunk = (n_layer - nextn) as usize;
                        if u("context_length").unwrap_or(0) >= 1_048_576 {
                            glm52_default_indexer_types(n_trunk)
                        } else {
                            vec![true; n_trunk]
                        }
                    }
                },
            });
            Some(MlaConfig {
                q_lora_rank,
                kv_lora_rank,
                qk_head_dim,
                qk_nope_head_dim: qk_head_dim - qk_rope_head_dim,
                qk_rope_head_dim,
                v_head_dim,
                sigmoid_routing,
                routed_scaling_factor: f("expert_weights_scale").unwrap_or(1.0),
                route_norm: u("expert_weights_norm").map(|v| v != 0).unwrap_or(false),
                n_shared_experts: u("expert_shared_count").unwrap_or(0),
                first_k_dense_replace: u("leading_dense_block_count").unwrap_or(0),
                dsa,
            })
        } else { None };

        let n_head = u("attention.head_count").unwrap_or_else(|| {
            step35.as_ref()
                .and_then(|s| s.head_count.iter().copied().max())
                .expect("head_count")
        });
        let n_head_kv = u("attention.head_count_kv")
            .or_else(|| step35.as_ref().and_then(|s| s.head_count_kv.iter().copied().max()))
            .unwrap_or_else(|| u("attention.head_count")
                .or_else(|| step35.as_ref()
                    .and_then(|s| s.head_count.iter().copied().max()))
                .expect("head_count_kv fallback"));
        let rope_freq_base = f("rope.freq_base").unwrap_or(10000.0);
        let rope_dim_count = u("rope.dimension_count").unwrap_or(head_dim_k);
        let full_attention_interval = u("full_attention_interval").unwrap_or(0);
        let geometry = match &arch {
            Arch::Qwen35 | Arch::Qwen35Moe => Some(ArchGeometryTable::qwen35(
                n_layer,
                nextn,
                full_attention_interval,
                n_head,
                n_head_kv,
                head_dim_k,
                head_dim_v,
                rope_dim_count,
                rope_freq_base,
            )),
            Arch::Step35 => Some(ArchGeometryTable::step35(
                n_layer,
                head_dim_k,
                head_dim_v,
                step35.as_ref().expect("step35 geometry needs step35 config"),
            )),
            _ => None,
        };

        ModelConfig {
            arch,
            name: g.metadata.get("general.name").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            n_layer,
            n_embd,
            // `attention.head_count` is a SCALAR on every arch but step35, where it is a
            // per-layer ARRAY (`as_u64` returns None on an Array, so the bare `.expect` would
            // panic). For step35 the global scalar is the MAX over layers: it sizes shared
            // scratch/workspace buffers, while every per-layer shape comes from
            // `step35.n_head(il)`. Max (96, not the 64 of a full layer) so no buffer under-sizes.
            n_head,
            n_head_kv,
            head_dim_k,
            head_dim_v,
            n_ff: u("feed_forward_length").unwrap_or(0),
            n_vocab: u("vocab_size").unwrap_or_else(|| {
                // vocab size from token_embd tensor's last dim if metadata absent
                g.find("token_embd.weight").map(|t| *t.ne.last().unwrap() as u32).unwrap_or(0)
            }),
            context_length: u("context_length").unwrap_or(0),
            rms_eps: f("attention.layer_norm_rms_epsilon").unwrap_or(1e-6),
            rope_freq_base,
            rope_dim_count,
            rope_sections,
            full_attention_interval,
            ssm,
            moe,
            m3: None,   // GGUF M3 metadata keys are a later arc (ST import first)
            hy3: None,  // GGUF Hy3 metadata keys are a later arc (repack source first)
            gemma4,
            mla,
            step35,
            geometry,
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
        let n_layer = c.num_hidden_layers + nextn;
        let full_attention_interval = c.full_attention_interval.unwrap_or(0);
        let geometry = match &arch {
            Arch::Qwen35 | Arch::Qwen35Moe => Some(ArchGeometryTable::qwen35(
                n_layer,
                nextn,
                full_attention_interval,
                n_head,
                n_head_kv,
                head_dim_k,
                head_dim_v,
                c.rotary_dim.unwrap_or(head_dim_k),
                c.rope_theta,
            )),
            _ => None,
        };

        ModelConfig {
            arch,
            name: c.name.clone().unwrap_or_default(),
            // GGUF `block_count` INCLUDES the MTP/NextN block(s) (hybrid.rs n_trunk = n_layer -
            // nextn); HF `num_hidden_layers` EXCLUDES them. Add nextn so both sources agree.
            n_layer,
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
            full_attention_interval,
            ssm,
            moe,
            m3,
            hy3,
            gemma4: None,   // ST gemma4 route: config wiring when that arc opens
            mla: None,      // GGUF-first arch (glm-dsa): HF/safetensors import is a later arc
            step35: None,   // GGUF-first arch: the official prequantized GGUF is the artifact
                            // (phase-1 §3.1 — no safetensors conversion in the bring-up path)
            geometry,
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
        if let Some(geometry) = self.layer_geometry(il) {
            return geometry.mixer;
        }
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

    /// qwen35-class FUSED [q|gate] attention output gate: wq packs q AND a per-head sigmoid gate
    /// (out = 2*n_head*head_dim) that `q_gate_split` separates. M3 and Hy3 have NO output gate —
    /// their wq out is exactly n_head*head_dim, and running the split would read 2x out of bounds.
    /// One predicate so every full-attn site (prefill/prime/decode/dc/spec) agrees.
    ///
    /// step35 is NOT in this class even though it HAS a head-wise gate: its gate is a separate
    /// `blk.N.attn_gate.weight [n_embd, n_head]` tensor (one scalar per head, broadcast over
    /// head_dim) and its wq out is exactly n_head*head_dim — see `attn_gate_separate()`. Running
    /// the fused split on it would read 2x out of bounds, which is why this deny-list must name it.
    pub fn attn_out_gate(&self) -> bool {
        if let Some(table) = self.geometry.as_ref() {
            return table.classes().iter()
                .any(|geometry| geometry.attention_gate == AttentionGateKind::FusedQ);
        }
        self.m3.is_none() && self.hy3.is_none() && self.gemma4.is_none() && self.mla.is_none()
            && self.step35.is_none()
    }

    /// step35-class SEPARATE head-wise attention gate: `blk.N.attn_gate.weight [n_embd, n_head_l]`
    /// yields one pre-sigmoid scalar PER HEAD, broadcast across head_dim over the attention output
    /// before wo (upstream `step35.cpp:267-285`: `attn_out * sigmoid(g_proj(attn_norm_out))`).
    /// Distinct from `attn_out_gate()` (fused-in-wq, per-DIM) — the two are mutually exclusive.
    /// Note the gate input is the POST-attn_norm hidden state (`cur`), not the raw residual.
    pub fn attn_gate_separate(&self) -> bool {
        if let Some(table) = self.geometry.as_ref() {
            return table.classes().iter()
                .any(|geometry| geometry.attention_gate == AttentionGateKind::SeparateHead);
        }
        self.step35.is_some()
    }

    /// Geometry row for a migrated architecture. `None` means the caller must use the existing
    /// architecture path; an out-of-range layer is never fabricated from another row.
    pub fn layer_geometry(&self, il: u32) -> Option<LayerGeometry> {
        self.geometry.as_ref()?.layer(il)
    }

    /// Resolve geometry for a full-attention execution arm. Migrated architectures read their
    /// declarative row; legacy architectures receive the exact scalar geometry they used before.
    pub fn full_attention_geometry_at(&self, il: u32) -> LayerGeometry {
        self.layer_geometry(il).unwrap_or(LayerGeometry {
            mixer: LayerKind::FullAttention,
            n_head: self.n_head,
            n_head_kv: self.n_head_kv,
            head_dim_k: self.head_dim_k,
            head_dim_v: self.head_dim_v,
            n_rot: self.rope_dim_count,
            rope_base: self.rope_freq_base,
            window: None,
            rope_factors: false,
            attention_gate: if self.attn_out_gate() {
                AttentionGateKind::FusedQ
            } else {
                AttentionGateKind::None
            },
        })
    }

    pub fn attn_out_gate_at(&self, il: u32) -> bool {
        self.layer_geometry(il)
            .map(|geometry| geometry.attention_gate == AttentionGateKind::FusedQ)
            .unwrap_or_else(|| self.attn_out_gate())
    }

    pub fn attn_gate_separate_at(&self, il: u32) -> bool {
        self.layer_geometry(il)
            .map(|geometry| geometry.attention_gate == AttentionGateKind::SeparateHead)
            .unwrap_or_else(|| self.attn_gate_separate())
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
        // glm-dsa: sigmoid + noaux_tc selection bias (exp_probs_b) + routed_scaling 2.5,
        // norm_topk_prob=true — the exact DeepSeek-V3 recipe M3/Hy3 already ride.
        if let Some(mla) = self.mla.as_ref() {
            if mla.sigmoid_routing { return Some((mla.routed_scaling_factor, mla.route_norm)); }
        }
        // step35: sigmoid + exp_probs_b selection bias + expert_weights_scale 3.0 +
        // expert_weights_norm true — the same DeepSeek-V3 recipe, different key names.
        if let Some(s) = self.step35.as_ref() {
            if s.sigmoid_routing { return Some((s.routed_scaling_factor, s.route_norm)); }
        }
        None
    }

    /// Per-layer query-head count. Global scalar for every arch except step35, whose
    /// `attention.head_count` is an array (64 on full-attn layers, 96 on SWA). Sites that build
    /// wq/wo/attn_gate shapes or size per-head loops MUST use this, not the `n_head` field.
    pub fn n_head_at(&self, il: u32) -> u32 {
        if let Some(geometry) = self.layer_geometry(il) {
            return geometry.n_head;
        }
        match self.step35.as_ref() {
            Some(s) => s.n_head(il),
            None => self.n_head,
        }
    }

    /// Per-layer KV-head count. gemma4 carries a per-layer array; step35's is uniform-8 but is
    /// serialized as an array. Every other arch is the global scalar.
    pub fn n_head_kv_at(&self, il: u32) -> u32 {
        if let Some(geometry) = self.layer_geometry(il) {
            return geometry.n_head_kv;
        }
        if let Some(g) = self.gemma4.as_ref() {
            if let Some(&n) = g.head_count_kv.get(il as usize) { return n; }
        }
        if let Some(s) = self.step35.as_ref() { return s.n_head_kv(il); }
        self.n_head_kv
    }

    /// True when layer `il` uses sliding-window attention. gemma4 and step35 carry an explicit
    /// per-layer pattern; every other arch is unwindowed (returns false).
    pub fn is_swa_at(&self, il: u32) -> bool {
        if let Some(geometry) = self.layer_geometry(il) {
            return geometry.window.is_some();
        }
        if let Some(g) = self.gemma4.as_ref() {
            return g.swa_pattern.get(il as usize).copied().unwrap_or(false);
        }
        if let Some(s) = self.step35.as_ref() { return s.is_swa(il); }
        false
    }

    /// Per-layer ROUTED-expert SwiGLU clamp limit, `None` when the arch/layer has none.
    /// step35-only today (`swiglu_clamp_exp`, live on layers 43-44 of 3.7-Flash).
    pub fn clamp_exp_at(&self, il: u32) -> Option<f32> {
        self.step35.as_ref().and_then(|s| s.clamp_exp(il))
    }

    /// Per-layer SHARED-expert SwiGLU clamp limit (`swiglu_clamp_shexp`).
    pub fn clamp_shexp_at(&self, il: u32) -> Option<f32> {
        self.step35.as_ref().and_then(|s| s.clamp_shexp(il))
    }

    /// True when ANY FFN branch on layer `il` needs the clamped SwiGLU form. This is the
    /// FUSED-EPILOGUE DENY predicate: memra's fused SiLU epilogues (grouped-decode,
    /// moe_pairs_silu_mul, the dev-path kernels) hardcode plain `silu(gate)*up`, so a layer
    /// with a live clamp must fall through to the unfused `ffn_act_*` seam. Substituting the
    /// plain form compiles, runs, and produces plausible-but-wrong logits — exactly the
    /// failure mode `swiglu_clamped_mul_scaled_f32`'s kernel-check cell guards against.
    pub fn swiglu_clamped_at(&self, il: u32) -> bool {
        self.clamp_exp_at(il).is_some() || self.clamp_shexp_at(il).is_some()
    }

    /// True when the model has a live SwiGLU clamp on ANY layer — the cheap whole-model
    /// question the no-`il` `ffn_act` seam asserts against (a clamped model reaching a seam
    /// that cannot see `il` means the caller has to be migrated to `ffn_act_exp`/`_shexp`).
    pub fn swiglu_clamped_anywhere(&self) -> bool {
        self.step35.as_ref().is_some_and(|s| {
            s.swiglu_clamp_exp.iter().chain(s.swiglu_clamp_shexp.iter()).any(|&l| l > 1e-6)
        })
    }
}

/// Subset of HF `config.json` fields memra needs. Defaults mirror GGUF fallbacks. Hybrid models
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
// (text_config) + the architectures string array. Rather than add serde to memra-gguf, parse
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
        let table = mc.geometry.as_ref().expect("qwen35 has a geometry table");
        assert_eq!(table.classes().len(), 2);
        assert_eq!(table.layer_classes().len(), 32);
        let linear = mc.layer_geometry(0).unwrap();
        assert_eq!(linear.mixer, LayerKind::LinearAttention);
        assert_eq!(linear.attention_gate, AttentionGateKind::None);
        let full = mc.layer_geometry(3).unwrap();
        assert_eq!(full.mixer, LayerKind::FullAttention);
        assert_eq!(full.n_head, 32);
        assert_eq!(full.n_head_kv, 8);
        assert_eq!(full.head_dim_k, 256);
        assert_eq!(full.n_rot, 256);
        assert_eq!(full.rope_base, 5_000_000.0);
        assert_eq!(full.window, None);
        assert!(!full.rope_factors);
        assert_eq!(full.attention_gate, AttentionGateKind::FusedQ);
    }

    #[test]
    fn qwen35_mtp_layer_is_explicit_full_attention_geometry() {
        let json = r#"{
          "model_type": "qwen3_5",
          "num_hidden_layers": 32,
          "num_nextn_predict_layers": 1,
          "hidden_size": 4096,
          "num_attention_heads": 32,
          "num_key_value_heads": 8,
          "head_dim": 128,
          "intermediate_size": 12288,
          "vocab_size": 151936,
          "max_position_embeddings": 262144,
          "full_attention_interval": 4
        }"#;
        let mc = ModelConfig::from_hf(&HfConfig::parse(json));
        assert_eq!(mc.n_layer, 33);
        assert_eq!(mc.layer_kind(31), LayerKind::FullAttention);
        assert_eq!(mc.layer_kind(32), LayerKind::FullAttention);
        assert_eq!(
            mc.layer_geometry(32).unwrap().attention_gate,
            AttentionGateKind::FusedQ
        );
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
    /// Checkpoint dir for the on-disk MiniMax tests below. Like `real_qwen3_17b_header`,
    /// they SKIP (not fail) when the model is absent from the box.
    const MINIMAX_DIR: &str = "/data/ai-ml/hf-models/minimax-m3-nvfp4-reap50";

    #[test]
    fn parse_minimax_m3_vl() {
        let Ok(txt) = std::fs::read_to_string(format!("{MINIMAX_DIR}/config.json")) else {
            eprintln!("SKIP parse_minimax_m3_vl: no model at {MINIMAX_DIR}");
            return;
        };
        let cfg = HfConfig::parse(&txt);
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
        let Ok(cfg_txt) = std::fs::read_to_string(format!("{MINIMAX_DIR}/config.json")) else {
            eprintln!("SKIP minimax_name_mapping_against_index: no model at {MINIMAX_DIR}");
            return;
        };
        let cfg = ModelConfig::from_hf(&HfConfig::parse(&cfg_txt));
        let idx: std::collections::HashSet<String> = {
            let txt = std::fs::read_to_string(
                format!("{MINIMAX_DIR}/model.safetensors.index.json")).unwrap();
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
