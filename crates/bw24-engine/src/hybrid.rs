//! Qwen3.5/3.6 hybrid model: linear-attention (Gated DeltaNet) layers + periodic full-attention
//! layers + SwiGLU FFN. Loads weights, runs the forward, dual cache. Builds on the validated
//! conv1d + gdn_scan kernels (M2/M3) and the dense full-attn path (M0).

use cudarc::driver::CudaSlice;
use bw24_gguf::{GgufFile, dequant};
use bw24_gguf::config::{ModelConfig, LayerKind};
use crate::Engine;
use crate::model::GpuTensor;

fn load_t(e: &Engine, g: &GgufFile, name: &str) -> Result<GpuTensor, Box<dyn std::error::Error>> {
    let t = g.find(name).unwrap_or_else(|| panic!("missing tensor {name}"));
    let n = t.n_elements() as usize;
    let f32v = dequant::dequantize(t.ggml_type, g.tensor_data(t), n);
    Ok(GpuTensor { data: e.htod(&f32v)?, ne: t.ne.clone() })
}
fn load_opt(e: &Engine, g: &GgufFile, name: &str) -> Result<Option<GpuTensor>, Box<dyn std::error::Error>> {
    match g.find(name) { Some(_) => Ok(Some(load_t(e, g, name)?)), None => Ok(None) }
}

pub struct FullAttnLayer {
    pub wq: GpuTensor, pub wk: GpuTensor, pub wv: GpuTensor, pub wo: GpuTensor,
    pub q_norm: GpuTensor, pub k_norm: GpuTensor,
}

pub struct LinearAttnLayer {
    pub wqkv: GpuTensor,       // [n_embd, conv_dim] -> qkv_mixed
    pub wqkv_gate: GpuTensor,  // [n_embd, value_dim] -> z
    pub ssm_beta: GpuTensor,   // [n_embd, num_v_heads]
    pub ssm_alpha: GpuTensor,  // [n_embd, num_v_heads]
    pub ssm_a: GpuTensor,      // [num_v_heads] (pre-negated -exp(A_log))
    pub ssm_dt: GpuTensor,     // [num_v_heads] bias
    pub ssm_conv1d: GpuTensor, // [d_conv, conv_dim]
    pub ssm_norm: GpuTensor,   // [head_v_dim]
    pub ssm_out: GpuTensor,    // [value_dim, n_embd]
}

pub enum Mixer { Full(FullAttnLayer), Linear(LinearAttnLayer) }

pub struct HybridLayer {
    pub attn_norm: GpuTensor,
    pub post_attn_norm: GpuTensor,  // "post_attention_norm" = PRE-FFN norm
    pub mixer: Mixer,
    pub ffn_gate: GpuTensor, pub ffn_up: GpuTensor, pub ffn_down: GpuTensor,
}

pub struct HybridModel {
    pub cfg: ModelConfig,
    pub tok_embd: GpuTensor,
    pub output_norm: GpuTensor,
    pub output: GpuTensor,
    pub layers: Vec<HybridLayer>,
}

impl HybridModel {
    pub fn load(e: &Engine, g: &GgufFile) -> Result<Self, Box<dyn std::error::Error>> {
        let cfg = ModelConfig::from_gguf(g);
        assert!(cfg.arch.is_hybrid(), "not a hybrid arch");
        assert!(cfg.moe.is_none(), "MoE hybrid not yet wired (dense FFN only)");

        let tok_embd = load_t(e, g, "token_embd.weight")?;
        let output_norm = load_t(e, g, "output_norm.weight")?;
        let output = match g.find("output.weight") { Some(_) => load_t(e, g, "output.weight")?, None => load_t(e, g, "token_embd.weight")? };

        let mut layers = Vec::with_capacity(cfg.n_layer as usize);
        for il in 0..cfg.n_layer {
            let p = |s: &str| format!("blk.{il}.{s}");
            let mixer = match cfg.layer_kind(il) {
                LayerKind::FullAttention => Mixer::Full(FullAttnLayer {
                    wq: load_t(e, g, &p("attn_q.weight"))?,
                    wk: load_t(e, g, &p("attn_k.weight"))?,
                    wv: load_t(e, g, &p("attn_v.weight"))?,
                    wo: load_t(e, g, &p("attn_output.weight"))?,
                    q_norm: load_t(e, g, &p("attn_q_norm.weight"))?,
                    k_norm: load_t(e, g, &p("attn_k_norm.weight"))?,
                }),
                LayerKind::LinearAttention => Mixer::Linear(LinearAttnLayer {
                    wqkv: load_t(e, g, &p("attn_qkv.weight"))?,
                    wqkv_gate: load_t(e, g, &p("attn_gate.weight"))?,
                    ssm_beta: load_t(e, g, &p("ssm_beta.weight"))?,
                    ssm_alpha: load_t(e, g, &p("ssm_alpha.weight"))?,
                    ssm_a: load_t(e, g, &p("ssm_a"))?,
                    ssm_dt: load_t(e, g, &p("ssm_dt.bias"))?,
                    ssm_conv1d: load_t(e, g, &p("ssm_conv1d.weight"))?,
                    ssm_norm: load_t(e, g, &p("ssm_norm.weight"))?,
                    ssm_out: load_t(e, g, &p("ssm_out.weight"))?,
                }),
            };
            // attn_norm always; post_attention_norm is the pre-FFN norm in qwen35
            layers.push(HybridLayer {
                attn_norm: load_t(e, g, &p("attn_norm.weight"))?,
                post_attn_norm: load_opt(e, g, &p("post_attention_norm.weight"))?
                    .or(load_opt(e, g, &p("ffn_norm.weight"))?)
                    .expect("need post_attention_norm or ffn_norm"),
                mixer,
                ffn_gate: load_t(e, g, &p("ffn_gate.weight"))?,
                ffn_up: load_t(e, g, &p("ffn_up.weight"))?,
                ffn_down: load_t(e, g, &p("ffn_down.weight"))?,
            });
        }
        Ok(HybridModel { cfg, tok_embd, output_norm, output, layers })
    }

    pub fn embed(&self, e: &Engine, tokens: &[u32]) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let n_embd = self.cfg.n_embd as usize;
        let embd = e.dtoh(&self.tok_embd.data)?;
        let mut x = vec![0f32; tokens.len() * n_embd];
        for (ti, &tok) in tokens.iter().enumerate() {
            let s = tok as usize * n_embd;
            x[ti * n_embd..ti * n_embd + n_embd].copy_from_slice(&embd[s..s + n_embd]);
        }
        Ok(e.htod(&x)?)
    }
}
