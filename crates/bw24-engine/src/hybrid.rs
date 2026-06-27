//! Qwen3.5/3.6 hybrid model: linear-attention (Gated DeltaNet) layers + periodic full-attention
//! layers + SwiGLU FFN. Loads weights, runs the forward, dual cache. Builds on the validated
//! conv1d + gdn_scan kernels (M2/M3) and the dense full-attn path (M0).

use cudarc::driver::CudaSlice;
use bw24_gguf::GgufFile;
use bw24_gguf::config::{ModelConfig, LayerKind};
use bw24_gguf::source::{TensorSource, GgufSource};
use crate::Engine;
use crate::model::{GpuTensor, EmbedHost, HostExps};

// Source-agnostic load helpers (GGUF or safetensors). The GGUF wrappers below keep `load()`
// byte-identical; only the source object differs.
fn load_t(e: &Engine, src: &dyn TensorSource, name: &str) -> Result<GpuTensor, Box<dyn std::error::Error>> {
    GpuTensor::load_from_source(e, src, name)
}
fn load_opt(e: &Engine, src: &dyn TensorSource, name: &str) -> Result<Option<GpuTensor>, Box<dyn std::error::Error>> {
    GpuTensor::load_opt_from_source(e, src, name)
}

/// Load the mixer (full-attn or linear-attn) for block `il`. Shared by the trunk loop and the MTP head.
/// `kind` overrides cfg.layer_kind (the MTP/NextN block is ALWAYS full-attn regardless of the periodic
/// interval — its GGUF carries attn_q/k/v, not ssm_*/attn_qkv).
fn load_mixer_kind(e: &Engine, src: &dyn TensorSource, il: u32, kind: LayerKind)
              -> Result<Mixer, Box<dyn std::error::Error>> {
    let p = |s: &str| format!("blk.{il}.{s}");
    Ok(match kind {
        LayerKind::FullAttention => Mixer::Full(FullAttnLayer {
            wq: load_t(e, src, &p("attn_q.weight"))?,
            wk: load_t(e, src, &p("attn_k.weight"))?,
            wv: load_t(e, src, &p("attn_v.weight"))?,
            wo: load_t(e, src, &p("attn_output.weight"))?,
            q_norm: load_t(e, src, &p("attn_q_norm.weight"))?,
            k_norm: load_t(e, src, &p("attn_k_norm.weight"))?,
        }),
        LayerKind::LinearAttention => Mixer::Linear(LinearAttnLayer {
            wqkv: load_t(e, src, &p("attn_qkv.weight"))?,
            wqkv_gate: load_t(e, src, &p("attn_gate.weight"))?,
            ssm_beta: load_t(e, src, &p("ssm_beta.weight"))?,
            ssm_alpha: load_t(e, src, &p("ssm_alpha.weight"))?,
            ssm_a: load_t(e, src, &p("ssm_a"))?,
            ssm_dt: load_t(e, src, &p("ssm_dt.bias"))?,
            ssm_conv1d: load_t(e, src, &p("ssm_conv1d.weight"))?,
            ssm_norm: load_t(e, src, &p("ssm_norm.weight"))?,
            ssm_out: load_t(e, src, &p("ssm_out.weight"))?,
        }),
    })
}

/// Load the FFN (dense SwiGLU or routed MoE) for block `il`. Source-agnostic (GGUF or safetensors
/// via `TensorSource`); shared by the hybrid trunk/MTP loops AND the dense-attention MoE path (OLMoE).
/// Shared-expert tensors are OPTIONAL (`load_opt`): qwen35moe has them, OLMoE/vanilla-MoE do not.
/// When `spill` is `Some` (BW24_SPILL_DISK on) AND the source is the GGUF on disk, MoE experts load
/// through the per-expert tier split (`HostExps::load_tiered`: hottest pinned, rest mmap'd from disk);
/// otherwise experts take the all-host / gather path. Spill tiering is GGUF-only (needs the file mmap).
pub(crate) fn load_ffn(e: &Engine, src: &dyn TensorSource, cfg: &ModelConfig, il: u32,
            spill: Option<(&GgufFile, &mut crate::spill::SpillCtx)>)
            -> Result<Ffn, Box<dyn std::error::Error>> {
    let p = |s: &str| format!("blk.{il}.{s}");
    Ok(if let Some(moe) = cfg.moe.as_ref() {
        let n_expert = moe.expert_count as usize;
        // Expert loader. `spill` carries an optional (GgufFile, SpillCtx) — only the GGUF on-disk
        // path can tier (it needs the file mmap); safetensors always gathers/stacks all-host.
        //  - spill Some -> per-expert tier split (hottest pinned, rest mmap'd from the GGUF).
        //  - GGUF 3D stacked name resolves -> load_stacked_from_source (all-host).
        //  - else (safetensors) -> gather N separate 2D expert tensors.
        let (gate_exps, up_exps, down_exps) = match spill {
            Some((g, ctx)) => (
                HostExps::load_tiered(e, g, &p("ffn_gate_exps.weight"), ctx)?,
                HostExps::load_tiered(e, g, &p("ffn_up_exps.weight"), ctx)?,
                HostExps::load_tiered(e, g, &p("ffn_down_exps.weight"), ctx)?),
            None => {
                let exps = |e: &Engine, n: &str| -> Result<HostExps, Box<dyn std::error::Error>> {
                    if src.has(n) { HostExps::load_stacked_from_source(e, src, n) }
                    else { HostExps::load_from_source(e, src, n, n_expert) }
                };
                (exps(e, &p("ffn_gate_exps.weight"))?,
                 exps(e, &p("ffn_up_exps.weight"))?,
                 exps(e, &p("ffn_down_exps.weight"))?)
            }
        };
        Ffn::Moe(MoeWeights {
            gate_inp:       load_t(e, src, &p("ffn_gate_inp.weight"))?,
            gate_inp_shexp: load_opt(e, src, &p("ffn_gate_inp_shexp.weight"))?,
            gate_exps, up_exps, down_exps,
            gate_shexp: load_opt(e, src, &p("ffn_gate_shexp.weight"))?,
            up_shexp:   load_opt(e, src, &p("ffn_up_shexp.weight"))?,
            down_shexp: load_opt(e, src, &p("ffn_down_shexp.weight"))?,
        })
    } else {
        Ffn::Dense {
            ffn_gate: load_t(e, src, &p("ffn_gate.weight"))?,
            ffn_up:   load_t(e, src, &p("ffn_up.weight"))?,
            ffn_down: load_t(e, src, &p("ffn_down.weight"))?,
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

/// MoE weights for one layer. Router + shared expert stay GPU-RESIDENT (tiny); the routed
/// experts stay HOST-RESIDENT (HostExps) and are staged per-token (EDGE-1).
///
/// The shared-expert fields are `Option`: qwen35moe carries a shared expert, but OLMoE (and most
/// vanilla MoE) have none (`shared_expert_intermediate_size` absent) — those layers `load_opt` the
/// shexp tensors to `None` (ST-MOE-PLAN §1.3, §3.2). When `None` the shared-expert branch is skipped.
pub struct MoeWeights {
    pub gate_inp: GpuTensor,        // F32 [n_embd, n_expert] router  (GPU resident, Float)
    pub gate_inp_shexp: Option<GpuTensor>,  // F32 [n_embd] 1-D shared gate dot (qwen35moe only)
    pub gate_exps: HostExps,        // [n_embd, n_ff_exp, n_expert]   (HOST)
    pub up_exps: HostExps,          // [n_embd, n_ff_exp, n_expert]   (HOST)
    pub down_exps: HostExps,        // [n_ff_exp, n_embd, n_expert] TRANSPOSED (HOST)
    pub gate_shexp: Option<GpuTensor>,
    pub up_shexp: Option<GpuTensor>,
    pub down_shexp: Option<GpuTensor>,
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
    /// Load a hybrid (qwen35) model from GGUF. Thin byte-identical wrapper over `load_from_source`.
    pub fn load(e: &Engine, g: &GgufFile) -> Result<Self, Box<dyn std::error::Error>> {
        Self::load_from_source(e, &GgufSource(g))
    }

    /// Load a hybrid model from any `TensorSource` (GGUF or a safetensors HF checkpoint). The whole
    /// loop speaks ggml names; the source maps them (and, for safetensors, applies the SSM value
    /// transforms via the owned-buffer seam). The forward graph is untouched.
    pub fn load_from_source(e: &Engine, src: &dyn TensorSource) -> Result<Self, Box<dyn std::error::Error>> {
        let cfg = src.config();
        assert!(cfg.arch.is_hybrid(), "not a hybrid arch");

        let embd = EmbedHost::from_source(src, "token_embd.weight");
        let output_norm = load_t(e, src, "output_norm.weight")?;
        // tied embeddings: fall back to tok_embd if output.weight absent.
        let output = if src.has("output.weight") { load_t(e, src, "output.weight")? }
                     else { load_t(e, src, "token_embd.weight")? };

        // SPILLING-PLAN §2: build the tiered-spill context ONCE, before loading any experts, but
        // only for a MoE model with the disk tier forced on (`BW24_SPILL_DISK`). It probes free VRAM
        // + host RAM at runtime (never hardcoded) and opens one shared GGUF mmap; all expert tensors
        // draw down its single pinned-RAM budget (hottest pinned, the rest mmap'd from disk). When
        // unset/dense this stays `None` and the load takes the byte-identical all-host path.
        // Disk spill is GGUF-only (needs the on-disk file mmap); src.gguf() is None for safetensors.
        let gguf: Option<&GgufFile> = src.gguf();
        let mut spill: Option<crate::spill::SpillCtx> =
            if cfg.moe.is_some() && crate::spill::disk_tier_enabled() && gguf.is_some() {
                let budget = crate::spill::MemBudget::probe(e)?;
                let ctx = crate::spill::SpillCtx::open(gguf.unwrap().path(), &budget)?;
                eprintln!("[spill] disk tier ON: free_vram={} MiB  pinnable_ram={} MiB (MemAvailable*frac)",
                          budget.free_vram >> 20, budget.free_pinnable_ram >> 20);
                Some(ctx)
            } else { None };

        // B0 FIX: cfg.n_layer == block_count INCLUDES the MTP/NextN block(s) (41 for the 35B-MoE).
        // Running the MTP block as a trunk layer is wrong; iterate only the trunk layers.
        // 9B (nextn=0): n_trunk = 32 (unchanged). 35B-MoE (nextn=1): n_trunk = 40 (drops MTP block).
        let n_trunk = (cfg.n_layer - cfg.nextn_predict_layers) as usize;
        let mut layers = Vec::with_capacity(n_trunk);
        for il in 0..n_trunk as u32 {
            let p = |s: &str| format!("blk.{il}.{s}");
            // attn_norm always; post_attention_norm is the pre-FFN norm in qwen35
            layers.push(HybridLayer {
                attn_norm: load_t(e, src, &p("attn_norm.weight"))?,
                post_attn_norm: load_opt(e, src, &p("post_attention_norm.weight"))?
                    .or(load_opt(e, src, &p("ffn_norm.weight"))?)
                    .expect("need post_attention_norm or ffn_norm"),
                mixer: load_mixer_kind(e, src, il, cfg.layer_kind(il))?,
                ffn: load_ffn(e, src, &cfg, il, spill.as_mut().map(|c| (gguf.unwrap(), c)))?,
            });
        }

        // MTP/NextN head: load the block the trunk loop drops (il = n_trunk). It is a full
        // transformer block PLUS the nextn.{enorm,hnorm,eh_proj} glue. Only when nextn>0 and the
        // eh_proj tensor actually exists in the file (some MTP GGUFs ship the draft separately).
        let mtp = if cfg.nextn_predict_layers > 0 {
            let n = n_trunk as u32;
            let p = |s: &str| format!("blk.{n}.{s}");
            match src.has(&p("nextn.eh_proj.weight")) {
                true => Some(MtpHead {
                    enorm:  load_t(e, src, &p("nextn.enorm.weight"))?,
                    hnorm:  load_t(e, src, &p("nextn.hnorm.weight"))?,
                    eh_proj: load_t(e, src, &p("nextn.eh_proj.weight"))?,
                    attn_norm: load_t(e, src, &p("attn_norm.weight"))?,
                    post_attn_norm: load_opt(e, src, &p("post_attention_norm.weight"))?
                        .or(load_opt(e, src, &p("ffn_norm.weight"))?)
                        .expect("MTP block needs post_attention_norm or ffn_norm"),
                    mixer: load_mixer_kind(e, src, n, LayerKind::FullAttention)?,
                    ffn: load_ffn(e, src, &cfg, n, spill.as_mut().map(|c| (gguf.unwrap(), c)))?,
                    shared_head_norm: load_opt(e, src, &p("nextn.shared_head_norm.weight"))?,
                    shared_head_head: load_opt(e, src, &p("nextn.shared_head.weight"))?,
                }),
                false => None,  // nextn>0 but no embedded eh_proj (external draft GGUF) -> no head
            }
        } else { None };

        if let Some(ctx) = spill.as_ref() {
            eprintln!("[spill] experts placed: {} pinned (Tier 1), {} mmap'd from disk (Tier 2, {} MiB)",
                      ctx.n_pinned, ctx.n_mmap, ctx.mmap_bytes >> 20);
        }

        Ok(HybridModel { cfg, embd, output_norm, output, layers, mtp })
    }

    pub fn embed(&self, e: &Engine, tokens: &[u32]) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let n_embd = self.cfg.n_embd as usize;
        let x = self.embd.gather(n_embd, tokens);
        Ok(e.htod(&x)?)
    }
}
