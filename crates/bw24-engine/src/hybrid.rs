//! Qwen3.5/3.6 hybrid model: linear-attention (Gated DeltaNet) layers + periodic full-attention
//! layers + SwiGLU FFN. Loads weights, runs the forward, dual cache. Builds on the validated
//! conv1d + gdn_scan kernels (M2/M3) and the dense full-attn path (M0).

use cudarc::driver::CudaSlice;
use bw24_gguf::GgufFile;
use bw24_gguf::config::{ModelConfig, LayerKind};
use crate::Engine;
use crate::model::{GpuTensor, EmbedHost, HostExps};

fn load_t(e: &Engine, g: &GgufFile, name: &str) -> Result<GpuTensor, Box<dyn std::error::Error>> {
    GpuTensor::load(e, g, name)
}
fn load_opt(e: &Engine, g: &GgufFile, name: &str) -> Result<Option<GpuTensor>, Box<dyn std::error::Error>> {
    GpuTensor::load_opt(e, g, name)
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

/// MoE weights for one layer. Router + shared expert stay GPU-RESIDENT (tiny); the 256 routed
/// experts stay HOST-RESIDENT (HostExps) and are staged per-token (EDGE-1).
pub struct MoeWeights {
    pub gate_inp: GpuTensor,        // F32 [2048,256] router          (GPU resident, Float)
    pub gate_inp_shexp: GpuTensor,  // F32 [2048] 1-D shared gate dot (GPU resident, Float, out_f=1)
    pub gate_exps: HostExps,        // Q6_K [2048,512,256]            (HOST)
    pub up_exps: HostExps,          // Q6_K [2048,512,256]            (HOST)
    pub down_exps: HostExps,        // Q8_0 [512,2048,256] TRANSPOSED (HOST; in=512,out=2048)
    pub gate_shexp: GpuTensor,      // Q8_0 [2048,512]                (GPU resident)
    pub up_shexp: GpuTensor,        // Q8_0 [2048,512]                (GPU resident)
    pub down_shexp: GpuTensor,      // Q8_0 [512,2048]                (GPU resident)
}

/// Per-layer FFN: dense SwiGLU (qwen35) or 256-expert MoE (qwen35moe).
pub enum Ffn {
    Dense { ffn_gate: GpuTensor, ffn_up: GpuTensor, ffn_down: GpuTensor },
    Moe(MoeWeights),
}

pub struct HybridLayer {
    pub attn_norm: GpuTensor,
    pub post_attn_norm: GpuTensor,  // "post_attention_norm" = PRE-FFN norm
    pub mixer: Mixer,
    pub ffn: Ffn,
}

pub struct HybridModel {
    pub cfg: ModelConfig,
    pub embd: EmbedHost,
    pub output_norm: GpuTensor,
    pub output: GpuTensor,
    pub layers: Vec<HybridLayer>,
}

impl HybridModel {
    pub fn load(e: &Engine, g: &GgufFile) -> Result<Self, Box<dyn std::error::Error>> {
        let cfg = ModelConfig::from_gguf(g);
        assert!(cfg.arch.is_hybrid(), "not a hybrid arch");

        let embd = EmbedHost::from_gguf(g, "token_embd.weight");
        let output_norm = load_t(e, g, "output_norm.weight")?;
        let output = match g.find("output.weight") { Some(_) => load_t(e, g, "output.weight")?, None => load_t(e, g, "token_embd.weight")? };

        // B0 FIX: cfg.n_layer == block_count INCLUDES the MTP/NextN block(s) (41 for the 35B-MoE).
        // Running the MTP block as a trunk layer is wrong; iterate only the trunk layers.
        // 9B (nextn=0): n_trunk = 32 (unchanged). 35B-MoE (nextn=1): n_trunk = 40 (drops MTP block).
        let n_trunk = (cfg.n_layer - cfg.nextn_predict_layers) as usize;
        let mut layers = Vec::with_capacity(n_trunk);
        for il in 0..n_trunk as u32 {
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
            let ffn = if cfg.moe.is_some() {
                Ffn::Moe(MoeWeights {
                    gate_inp:       load_t(e, g, &p("ffn_gate_inp.weight"))?,        // F32 -> Float
                    gate_inp_shexp: load_t(e, g, &p("ffn_gate_inp_shexp.weight"))?,  // F32 1-D
                    gate_exps: HostExps::load(g, &p("ffn_gate_exps.weight"))?,       // HOST Q6_K
                    up_exps:   HostExps::load(g, &p("ffn_up_exps.weight"))?,         // HOST Q6_K
                    down_exps: HostExps::load(g, &p("ffn_down_exps.weight"))?,       // HOST Q8_0
                    gate_shexp: load_t(e, g, &p("ffn_gate_shexp.weight"))?,
                    up_shexp:   load_t(e, g, &p("ffn_up_shexp.weight"))?,
                    down_shexp: load_t(e, g, &p("ffn_down_shexp.weight"))?,
                })
            } else {
                Ffn::Dense {
                    ffn_gate: load_t(e, g, &p("ffn_gate.weight"))?,
                    ffn_up:   load_t(e, g, &p("ffn_up.weight"))?,
                    ffn_down: load_t(e, g, &p("ffn_down.weight"))?,
                }
            };
            // attn_norm always; post_attention_norm is the pre-FFN norm in qwen35
            layers.push(HybridLayer {
                attn_norm: load_t(e, g, &p("attn_norm.weight"))?,
                post_attn_norm: load_opt(e, g, &p("post_attention_norm.weight"))?
                    .or(load_opt(e, g, &p("ffn_norm.weight"))?)
                    .expect("need post_attention_norm or ffn_norm"),
                mixer,
                ffn,
            });
        }
        Ok(HybridModel { cfg, embd, output_norm, output, layers })
    }

    pub fn embed(&self, e: &Engine, tokens: &[u32]) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let n_embd = self.cfg.n_embd as usize;
        let x = self.embd.gather(n_embd, tokens);
        Ok(e.htod(&x)?)
    }
}
