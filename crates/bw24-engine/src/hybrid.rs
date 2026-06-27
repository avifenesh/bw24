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

/// Load the mixer (full-attn or linear-attn) for block `il`. Shared by the trunk loop and the MTP head.
/// `kind` overrides cfg.layer_kind (the MTP/NextN block is ALWAYS full-attn regardless of the periodic
/// interval — its GGUF carries attn_q/k/v, not ssm_*/attn_qkv).
fn load_mixer_kind(e: &Engine, g: &GgufFile, il: u32, kind: LayerKind)
              -> Result<Mixer, Box<dyn std::error::Error>> {
    let p = |s: &str| format!("blk.{il}.{s}");
    Ok(match kind {
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
    })
}

/// Load the FFN (dense SwiGLU or 256-expert MoE) for block `il`. Shared by trunk loop and MTP head.
fn load_ffn(e: &Engine, g: &GgufFile, cfg: &ModelConfig, il: u32)
            -> Result<Ffn, Box<dyn std::error::Error>> {
    let p = |s: &str| format!("blk.{il}.{s}");
    Ok(if cfg.moe.is_some() {
        Ffn::Moe(MoeWeights {
            gate_inp:       load_t(e, g, &p("ffn_gate_inp.weight"))?,
            gate_inp_shexp: load_t(e, g, &p("ffn_gate_inp_shexp.weight"))?,
            gate_exps: HostExps::load(e, g, &p("ffn_gate_exps.weight"))?,
            up_exps:   HostExps::load(e, g, &p("ffn_up_exps.weight"))?,
            down_exps: HostExps::load(e, g, &p("ffn_down_exps.weight"))?,
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
    })
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

/// Qwen3.5 NextN/MTP head: a full transformer block (attn+FFN, same tensors as a trunk layer)
/// plus the MTP glue (enorm/hnorm/eh_proj that fold the next-token embedding into the trunk
/// hidden, and an optional shared_head_norm/head). Loaded from blk.{n_trunk}.* — the block the
/// trunk loop drops. Used for speculative decode (drafts 1 token per call). See research/mtp/MTP-PLAN.md.
pub struct MtpHead {
    pub enorm: GpuTensor,            // blk.N.nextn.enorm   — RMSNorm of the next-token embedding
    pub hnorm: GpuTensor,            // blk.N.nextn.hnorm   — RMSNorm of the trunk hidden
    pub eh_proj: GpuTensor,          // blk.N.nextn.eh_proj [2*n_embd, n_embd]: [e_norm; h_norm] -> n_embd
    pub attn_norm: GpuTensor,        // blk.N.attn_norm
    pub post_attn_norm: GpuTensor,   // blk.N.post_attention_norm (pre-FFN)
    pub mixer: Mixer,                // full-attn block (qwen35 MTP block is full-attn)
    pub ffn: Ffn,                    // Dense or Moe, same loader as trunk
    pub shared_head_norm: Option<GpuTensor>,  // blk.N.nextn.shared_head_norm (else reuse output_norm)
    pub shared_head_head: Option<GpuTensor>,  // blk.N.nextn.shared_head      (else reuse output)
}

pub struct HybridModel {
    pub cfg: ModelConfig,
    pub embd: EmbedHost,
    pub output_norm: GpuTensor,
    pub output: GpuTensor,
    pub layers: Vec<HybridLayer>,
    pub mtp: Option<MtpHead>,        // NextN spec-decode head (None if nextn_predict_layers == 0)
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
            // attn_norm always; post_attention_norm is the pre-FFN norm in qwen35
            layers.push(HybridLayer {
                attn_norm: load_t(e, g, &p("attn_norm.weight"))?,
                post_attn_norm: load_opt(e, g, &p("post_attention_norm.weight"))?
                    .or(load_opt(e, g, &p("ffn_norm.weight"))?)
                    .expect("need post_attention_norm or ffn_norm"),
                mixer: load_mixer_kind(e, g, il, cfg.layer_kind(il))?,
                ffn: load_ffn(e, g, &cfg, il)?,
            });
        }

        // MTP/NextN head: load the block the trunk loop drops (il = n_trunk). It is a full
        // transformer block PLUS the nextn.{enorm,hnorm,eh_proj} glue. Only when nextn>0 and the
        // eh_proj tensor actually exists in the file (some MTP GGUFs ship the draft separately).
        let mtp = if cfg.nextn_predict_layers > 0 {
            let n = n_trunk as u32;
            let p = |s: &str| format!("blk.{n}.{s}");
            match g.find(&p("nextn.eh_proj.weight")) {
                Some(_) => Some(MtpHead {
                    enorm:  load_t(e, g, &p("nextn.enorm.weight"))?,
                    hnorm:  load_t(e, g, &p("nextn.hnorm.weight"))?,
                    eh_proj: load_t(e, g, &p("nextn.eh_proj.weight"))?,
                    attn_norm: load_t(e, g, &p("attn_norm.weight"))?,
                    post_attn_norm: load_opt(e, g, &p("post_attention_norm.weight"))?
                        .or(load_opt(e, g, &p("ffn_norm.weight"))?)
                        .expect("MTP block needs post_attention_norm or ffn_norm"),
                    mixer: load_mixer_kind(e, g, n, LayerKind::FullAttention)?,
                    ffn: load_ffn(e, g, &cfg, n)?,
                    shared_head_norm: load_opt(e, g, &p("nextn.shared_head_norm.weight"))?,
                    shared_head_head: load_opt(e, g, &p("nextn.shared_head.weight"))?,
                }),
                None => None,  // nextn>0 but no embedded eh_proj (external draft GGUF) -> no head
            }
        } else { None };

        Ok(HybridModel { cfg, embd, output_norm, output, layers, mtp })
    }

    pub fn embed(&self, e: &Engine, tokens: &[u32]) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let n_embd = self.cfg.n_embd as usize;
        let x = self.embd.gather(n_embd, tokens);
        Ok(e.htod(&x)?)
    }
}
