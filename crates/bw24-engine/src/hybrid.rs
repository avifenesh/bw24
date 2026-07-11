//! Qwen3.5/3.6 hybrid model: linear-attention (Gated DeltaNet) layers + periodic full-attention
//! layers + SwiGLU FFN. Loads weights, runs the forward, dual cache. Builds on the validated
//! conv1d + gdn_scan kernels (M2/M3) and the dense full-attn path (M0).

use cudarc::driver::CudaSlice;
use bw24_gguf::{GgufFile, GgmlType};
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
            // gemma4 global layers ship NO v_proj (attention_k_eq_v): V = the K projection
            // output pre-rope (llama gemma4.cpp: `Vcur = wv ? mm(wv,cur) : Kcur`). Loading
            // wv := wk reproduces that exactly with zero forward changes; the gemma forward
            // adds the weightless V rms_norm (R7 part 2).
            wv: match load_opt(e, src, &p("attn_v.weight"))? {
                Some(v) => v,
                None => load_t(e, src, &p("attn_k.weight"))?,
            },
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
    // MiniMax-M3: moe_layer_freq[il]==0 -> this layer is a DENSE-FFN layer (layers 0..2) even
    // though the arch is MoE; force the Dense arm (its mlp.{p}_proj names map via ggml_to_hf).
    // Hy3: `first_k_dense_replace` leading layers are dense-FFN (REAP50: layer 0 only).
    let dense_override = cfg.m3.as_ref()
        .is_some_and(|m| m.moe_layer_freq.get(il as usize).copied() == Some(0))
        || cfg.hy3.as_ref().is_some_and(|h| il < h.first_k_dense_replace)
        // gemma4 DENSE variants (31B/E4B): the arch is MoE-capable but the file ships no
        // expert tensors at all — tensor presence decides.
        || (cfg.gemma4.is_some() && !src.has(&p("ffn_gate_exps.weight"))
            && !src.has(&p("ffn_gate_up_exps.weight")));
    Ok(if let Some(moe) = cfg.moe.as_ref().filter(|_| !dense_override) {
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
                // gemma4: gate+up ship FUSED (ffn_gate_up_exps, gate rows first) — split at load.
                let fused = p("ffn_gate_up_exps.weight");
                if !src.has(&p("ffn_gate_exps.weight")) && src.has(&fused) {
                    let ff = moe.expert_ff_length as usize;
                    (HostExps::load_stacked_split_from_source(e, src, &fused, 0, ff)?,
                     HostExps::load_stacked_split_from_source(e, src, &fused, ff, 2 * ff)?,
                     exps(e, &p("ffn_down_exps.weight"))?)
                } else {
                (exps(e, &p("ffn_gate_exps.weight"))?,
                 exps(e, &p("ffn_up_exps.weight"))?,
                 exps(e, &p("ffn_down_exps.weight"))?)
                }
            }
        };
        // FITS-VRAM RESIDENT EXPERTS: upload this layer's 3 expert slabs to device when a global
        // budget (BW24_MOE_RESIDENT_GB, default = 80% of free VRAM at first-layer load) covers
        // the whole model's expert bytes. Decision is made ONCE (first MoE layer): total expert
        // bytes = per-layer bytes x n_moe_layers (uniform layers; UD-quant variance is small and
        // the budget has 20% slack). Failure to fit => None => the SLRU spill machinery.
        let dev_exps = build_dev_exps(e, cfg, &gate_exps, &up_exps, &down_exps)?;
        // e_score_correction_bias (M3 sigmoid routing): tiny [n_expert] f32, host-side.
        let exp_probs_b = src.find(&p("exp_probs_b.bias")).map(|v| {
            bw24_gguf::dequant::dequantize(v.ggml_type, &v.bytes, n_expert)
        });
        Ffn::Moe(MoeWeights {
            gate_inp:       load_t(e, src, &p("ffn_gate_inp.weight"))?,
            gate_inp_shexp: load_opt(e, src, &p("ffn_gate_inp_shexp.weight"))?,
            exp_probs_b,
            gate_exps, up_exps, down_exps,
            gate_shexp: load_opt(e, src, &p("ffn_gate_shexp.weight"))?,
            up_shexp:   load_opt(e, src, &p("ffn_up_shexp.weight"))?,
            down_shexp: load_opt(e, src, &p("ffn_down_shexp.weight"))?,
            dev_exps,
        })
    } else {
        Ffn::Dense {
            ffn_gate: load_t(e, src, &p("ffn_gate.weight"))?,
            ffn_up:   load_t(e, src, &p("ffn_up.weight"))?,
            ffn_down: load_t(e, src, &p("ffn_down.weight"))?,
        }
    })
}

/// Decide + build the resident expert slabs for one layer. Budget check runs once (static):
/// projected total = this layer's expert bytes x (n_layer MoE layers, approximated as all);
/// fits => every subsequent layer uploads too (uniform). BW24_MOE_RESIDENT=0 forces the SLRU path.
fn build_dev_exps(e: &Engine, cfg: &ModelConfig, gate: &HostExps, up: &HostExps, down: &HostExps)
                  -> Result<Option<crate::hybrid::DevExps>, Box<dyn std::error::Error>> {
    use std::sync::OnceLock;
    static DECISION: OnceLock<bool> = OnceLock::new();
    let per_layer = gate.bytes.as_bytes().len() + up.bytes.as_bytes().len() + down.bytes.as_bytes().len();
    let fits = *DECISION.get_or_init(|| {
        if std::env::var("BW24_MOE_RESIDENT").as_deref() == Ok("0") { return false; }
        if gate.tiers.is_some() { return false; }   // tiered/spill loads keep the cache path
        let (free, _total) = match e.ctx().mem_get_info() { Ok(v) => v, Err(_) => return false };
        let budget = std::env::var("BW24_MOE_RESIDENT_GB").ok()
            .and_then(|v| v.parse::<f64>().ok())
            .map(|gb| (gb * 1e9) as usize)
            .unwrap_or((free as f64 * 0.80) as usize);
        let projected = per_layer * cfg.n_layer as usize;   // upper bound (dense layers shrink it)
        let ok = projected <= budget;
        eprintln!("[moe] resident-experts decision: per-layer {}MB x {} layers = {:.1}GB vs budget {:.1}GB -> {}",
                  per_layer / 1_000_000, cfg.n_layer, projected as f64 / 1e9, budget as f64 / 1e9,
                  if ok { "RESIDENT" } else { "SLRU cache" });
        ok
    });
    if !fits { return Ok(None); }
    use cudarc::driver::DevicePtr;
    let gu_il = std::env::var("BW24_MOE_GU_IL").as_deref() == Ok("1")
        && gate.out_f == up.out_f && gate.in_f == up.in_f;
    let n_expert = gate.n_expert;
    let (g, u) = if gu_il {
        // interleave gate/up rows: [ex][row o] = gate-row-o bytes ++ up-row-o bytes.
        let (rbg, rbu) = (gate.row_bytes, up.row_bytes);
        let n_rows = gate.out_f;
        let gb = gate.bytes.as_bytes();
        let ub = up.bytes.as_bytes();
        let mut il = vec![0u8; n_expert * n_rows * (rbg + rbu)];
        for ex in 0..n_expert {
            for o in 0..n_rows {
                let dst = (ex * n_rows + o) * (rbg + rbu);
                let sg = ex * gate.expert_stride + o * rbg;
                let su = ex * up.expert_stride + o * rbu;
                il[dst..dst + rbg].copy_from_slice(&gb[sg..sg + rbg]);
                il[dst + rbg..dst + rbg + rbu].copy_from_slice(&ub[su..su + rbu]);
            }
        }
        let ild = e.htod_bytes_padded(&il, 8)?;
        // `up` slot points into the same buffer via ptr math; keep a tiny placeholder alloc so
        // the struct shape is unchanged (the table below carries the real pointers).
        (ild, e.htod_bytes(&[0u8; 16])?)
    } else {
        (e.htod_bytes_padded(gate.bytes.as_bytes(), 8)?,
         e.htod_bytes_padded(up.bytes.as_bytes(), 8)?)
    };
    let d = e.htod_bytes_padded(down.bytes.as_bytes(), 8)?;
    let mut host = vec![0u64; 3 * n_expert];
    let (pg, pu, pd) = {
        let (pg, _e0) = g.device_ptr(e.stream());
        let (pu, _e1) = u.device_ptr(e.stream());
        let (pd, _e2) = d.device_ptr(e.stream());
        (pg as u64, pu as u64, pd as u64)
    };
    for ex in 0..n_expert {
        if gu_il {
            let stride = gate.out_f * (gate.row_bytes + up.row_bytes);
            host[ex]            = pg + (ex * stride) as u64;
            host[n_expert + ex] = pg + (ex * stride + gate.row_bytes) as u64;
        } else {
            host[ex]            = pg + (ex * gate.expert_stride) as u64;
            host[n_expert + ex] = pu + (ex * up.expert_stride) as u64;
        }
        host[2 * n_expert + ex] = pd + (ex * down.expert_stride) as u64;
    }
    if gu_il { eprintln!("[moe] gate/up dev slab INTERLEAVED (BW24_MOE_GU_IL)"); }
    let ptr_row = e.htod_u64(&host)?;
    Ok(Some(crate::hybrid::DevExps { gate: g, up: u, down: d, ptr_row, gu_il }))
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
    /// DeepSeek-V3/MiniMax-M3 `e_score_correction_bias` [n_expert]: added to the sigmoid scores
    /// for expert SELECTION only; the routing weights use the un-biased scores. Kept host-side —
    /// routing's top-k is a host loop and this is n_expert floats.
    pub exp_probs_b: Option<Vec<f32>>,
    pub gate_exps: HostExps,        // [n_embd, n_ff_exp, n_expert]   (HOST)
    pub up_exps: HostExps,          // [n_embd, n_ff_exp, n_expert]   (HOST)
    pub down_exps: HostExps,        // [n_ff_exp, n_embd, n_expert] TRANSPOSED (HOST)
    pub gate_shexp: Option<GpuTensor>,
    pub up_shexp: Option<GpuTensor>,
    pub down_shexp: Option<GpuTensor>,
    /// FITS-VRAM RESIDENT EXPERTS (2026-07-06): when the WHOLE model's expert bytes fit the VRAM
    /// budget, each (proj) slab is uploaded once as a contiguous device buffer and the fused
    /// _dev kernels take base+ex*stride pointers — no SLRU, no dispatch, no residency checks
    /// (llama's full-offload regime; measured 169.55 vs bw24's cache path 28.5 on the local 35B).
    /// None => the SLRU host-expert machinery (the spill regime, where it WINS vs llama's
    /// CPU-offload degradation). Decided at load in `load_ffn` (BW24_MOE_RESIDENT=0 forces off).
    pub dev_exps: Option<DevExps>,
}

/// Device-resident expert slabs for one layer (gate/up/down) + the prebuilt [3, n_expert]
/// pointer row the _dev kernels consume.
pub struct DevExps {
    pub gate: CudaSlice<u8>,
    pub up: CudaSlice<u8>,
    pub down: CudaSlice<u8>,
    /// [3*n_expert] u64 device row: gate ptrs, up ptrs, down ptrs (proj-major like layer_dev_row).
    pub ptr_row: CudaSlice<u64>,
    /// WALL-GAP ARC (BW24_MOE_GU_IL=1): gate/up rows INTERLEAVED in one slab — row o of gate at
    /// base + o*(rb_g+rb_u), up at +rb_g. Consumers on the dev path must use (rb_g+rb_u) as the
    /// row stride for BOTH projections (see MoeWeights::dev_rb_gu). One contiguous 1760B stream
    /// per (expert,row) instead of two scattered 880B streams — the measured 56%-of-wall fix
    /// candidate. Kernels unchanged (stride is already a parameter everywhere).
    pub gu_il: bool,
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
    pub gemma4: Option<Gemma4LayerBits>,
}

/// Gemma-4 per-layer extras (R8 wiring, HANDOVER "R8 VERIFIED WIRING"): the parallel shared
/// FFN branch, the four extra norms, the router prologue scale vector, per-expert output
/// scales, and the layer output scalar.
pub struct Gemma4LayerBits {
    pub ffn_norm: GpuTensor,           // ffn pre-norm (dense: THE ffn norm; moe: shared branch)
    pub post_ffw_norm: GpuTensor,      // combined post (before the attn_out residual)
    /// MoE-layer extras (None on the dense gemma4 variants — 31B/E4B): the parallel shared
    /// branch norms + tensors, the router prologue vector, per-expert output scales.
    pub moe_bits: Option<Gemma4MoeBits>,
    pub layer_scale: f32,              // layer_output_scale [1]
    /// E4B extras (None on 26B/31B): the per-layer-embedding tail block + KV-share target.
    pub e4b: Option<Gemma4E4bLayer>,
}

/// gemma-4 E4B per-layer bits (see research/gemma4-bringup/e4b-arch-map.md):
/// tail block  cur += rms_norm(proj . (gelu(inp_gate . cur) * inp_pl[il]), post_norm)
/// and the KV-share map — layers il >= n_layer-shared_kv_layers have NO own k/v projections
/// and attend the cache of layer (n_layer-shared) - (swa ? 2 : 1) with their own Q.
pub struct Gemma4E4bLayer {
    pub inp_gate: GpuTensor,           // blk.N.inp_gate  [n_embd, n_epl]
    pub proj: GpuTensor,               // blk.N.proj      [n_epl, n_embd]
    pub post_norm: GpuTensor,          // blk.N.post_norm [n_embd]
    /// Some(target_layer) on KV-shared layers (wk/wv here are the TARGET layer's tensors,
    /// loaded for shape symmetry only — the forward must skip k/v compute + append and read
    /// the target's cache; TODO dedupe the duplicate weight upload ~63MB).
    pub kv_share: Option<u32>,
}

/// gemma-4 E4B model-level per-layer-embedding tensors (prologue inputs). The token table
/// stays HOST-side raw GGUF bytes at load (Q6_K [n_epl*n_layer, n_vocab], ~2.3GB VRAM when
/// uploaded — the forward arc decides resident-vs-gather placement).
pub struct Gemma4E4bModel {
    /// device copy of the per-layer token table, uploaded on first use (the 26B embd_gpu
    /// pattern — keeps the ~2.3GB off load-critical paths that never decode).
    pub tok_tbl_gpu: std::sync::OnceLock<CudaSlice<u8>>,
    pub tok_embd_bytes: Vec<u8>,
    pub tok_embd_qt: i32,
    pub tok_embd_row_bytes: usize,
    pub model_proj: GpuTensor,         // per_layer_model_proj [n_embd, n_epl*n_layer] F16
    pub proj_norm: GpuTensor,          // per_layer_proj_norm [n_epl]
    pub n_epl: usize,
}

pub struct Gemma4MoeBits {
    pub post_ffw_norm_1: GpuTensor,    // shared-branch post
    pub pre_ffw_norm_2: GpuTensor,     // moe-branch pre
    pub post_ffw_norm_2: GpuTensor,    // moe-branch post
    pub shared_gate: GpuTensor,
    pub shared_up: GpuTensor,
    pub shared_down: GpuTensor,
    /// ffn_gate_inp.scale [n_embd] PRE-multiplied by 1/sqrt(n_embd) at load: the router
    /// prologue (weightless rms_norm x 1/sqrt(n_embd) x scale-vec) collapses to ONE rms_norm
    /// with this as the norm weight (x_hat * (v*s) vs llama's (x_hat*s)*v — one reassociation;
    /// the argmax gate arbitrates).
    pub router_scale_pre: CudaSlice<f32>,
    pub per_expert_scale: Vec<f32>,    // ffn_down_exps.scale [n_expert] (host)
    pub per_expert_scale_d: CudaSlice<f32>,  // device copy (router-weight fold kernel)
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
    /// FR-Spec draft->target vocab map: the draft lm_head is TRIMMED to the highest-frequency
    /// tokens (e.g. 32768 rows of the full 248320-row head); `d2t[draft_idx]` = the target vocab
    /// token id of trimmed row `draft_idx`. `None` for a full-vocab head (identity map). Host-side:
    /// the draft argmax already lands on host as one u32, so the map is a single Vec index.
    pub d2t: Option<Vec<u32>>,
}

impl MtpHead {
    /// Load an MTP/NextN head from a STANDALONE draft GGUF (BW24_MTP_DRAFT override). The draft
    /// file carries ONLY the NextN block (blk.N.nextn.* glue + attn/ffn) plus its own lm_head
    /// (`output.weight`) — which for an FR-Spec draft is TRIMMED to the top-frequency rows, with
    /// a `d2t` (i32/i64) tensor mapping trimmed-row index -> target vocab token id. Draft-token
    /// embedding still uses the MAIN model's token_embd (identical weights, saves VRAM), so the
    /// draft file's full-vocab token_embd copy is ignored.
    pub fn load_draft(e: &Engine, g: &GgufFile, main_cfg: &ModelConfig)
                      -> Result<Self, Box<dyn std::error::Error>> {
        let src = GgufSource(g);
        let dcfg = src.config();
        // The head forward runs with the MAIN model's cfg (eps/rope/head geometry) — the draft
        // block must be the same shape or the forward is garbage.
        assert_eq!(dcfg.n_embd, main_cfg.n_embd, "draft n_embd != model n_embd");
        assert_eq!(dcfg.n_head, main_cfg.n_head, "draft n_head != model n_head");
        assert_eq!(dcfg.n_head_kv, main_cfg.n_head_kv, "draft n_head_kv != model n_head_kv");
        assert_eq!(dcfg.head_dim_k, main_cfg.head_dim_k, "draft head_dim != model head_dim");
        assert!(dcfg.nextn_predict_layers > 0, "draft GGUF has no nextn_predict_layers");

        // NextN block index INSIDE THE DRAFT FILE (its block_count includes the trunk numbering).
        let n = dcfg.n_layer - dcfg.nextn_predict_layers;
        let p = |s: &str| format!("blk.{n}.{s}");

        // Draft lm_head: the file's own output.weight (+ shared_head_norm / output_norm). For
        // FR-Spec this is [n_embd, draft_vocab] with draft_vocab << n_vocab.
        let head = load_t(e, &src, "output.weight")?;
        let head_norm = match load_opt(e, &src, &p("nextn.shared_head_norm.weight"))? {
            Some(t) => Some(t),
            None => load_opt(e, &src, "output_norm.weight")?,
        };

        // d2t: draft-row -> target-token-id map (absolute ids, verified against the tokenizer).
        let d2t: Option<Vec<u32>> = g.find("d2t").map(|t| {
            let bytes = g.tensor_data(t);
            match t.ggml_type {
                GgmlType::I32 => bytes.chunks_exact(4)
                    .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as u32).collect(),
                GgmlType::I64 => bytes.chunks_exact(8)
                    .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as u32).collect(),
                other => panic!("d2t must be I32/I64, got {other:?}"),
            }
        });
        if let Some(map) = &d2t {
            assert_eq!(map.len(), head.out_features(),
                       "d2t len {} != draft head rows {}", map.len(), head.out_features());
            let n_vocab = main_cfg.n_vocab as u64;
            assert!(map.iter().all(|&t| (t as u64) < n_vocab),
                    "d2t contains token id >= model n_vocab {n_vocab}");
        }
        eprintln!("[mtp-draft] external draft head: blk.{n}, head_vocab={}{}",
                  head.out_features(),
                  if d2t.is_some() { " (trimmed, d2t map)" } else { " (full)" });

        Ok(MtpHead {
            enorm:  load_t(e, &src, &p("nextn.enorm.weight"))?,
            hnorm:  load_t(e, &src, &p("nextn.hnorm.weight"))?,
            eh_proj: load_t(e, &src, &p("nextn.eh_proj.weight"))?,
            attn_norm: load_t(e, &src, &p("attn_norm.weight"))?,
            post_attn_norm: load_opt(e, &src, &p("post_attention_norm.weight"))?
                .or(load_opt(e, &src, &p("ffn_norm.weight"))?)
                .expect("draft NextN block needs post_attention_norm or ffn_norm"),
            mixer: load_mixer_kind(e, &src, n, LayerKind::FullAttention)?,
            ffn: load_ffn(e, &src, &dcfg, n, None)?,
            shared_head_norm: head_norm,
            shared_head_head: Some(head),
            d2t,
        })
    }
}

/// gemma4 model-level auxiliaries.
pub struct GemmaAux {
    /// rope_freqs.weight [hd_global/2] freq factors — global layers' RoPE (R9).
    pub rope_freqs: Option<CudaSlice<f32>>,
    /// all-ones norm weight [512] (max head_dim) — the weightless rms_norms (R7 V-norm).
    pub ones: CudaSlice<f32>,
    /// E4B per-layer-embedding model tensors (None on 26B/31B).
    pub e4b: Option<Gemma4E4bModel>,
}

pub struct HybridModel {
    pub cfg: ModelConfig,
    pub embd: EmbedHost,
    pub output_norm: GpuTensor,
    pub output: GpuTensor,
    pub layers: Vec<HybridLayer>,
    pub mtp: Option<MtpHead>,        // NextN spec-decode head (None if nextn_predict_layers == 0)
    /// Lazily-uploaded DEVICE copy of the raw embed table (spec/graph hot loops gather rows
    /// on-device instead of host-dequant + htod). ~0.5GB; uploaded once on first use.
    pub embd_gpu: std::sync::OnceLock<cudarc::driver::CudaSlice<u8>>,
    pub gemma4_aux: Option<GemmaAux>,
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
                mixer: {
                    // E4B KV-shared layers ship NO attn_k/attn_v — load the SHARE TARGET's
                    // k/v tensors for shape symmetry (forward skips k/v compute there and
                    // reads the target layer's cache; see Gemma4E4bLayer::kv_share).
                    let g4_shared = cfg.gemma4.as_ref().map(|g| g.shared_kv_layers).unwrap_or(0);
                    let kv_from = n_trunk as u32 - g4_shared;
                    if g4_shared > 0 && il >= kv_from
                        && !src.has(&format!("blk.{il}.attn_k.weight")) {
                        let g4 = cfg.gemma4.as_ref().unwrap();
                        let swa = g4.swa_pattern.get(il as usize).copied().unwrap_or(true);
                        let tgt = kv_from - if swa { 2 } else { 1 };
                        let tp = |s: &str| format!("blk.{tgt}.{s}");
                        Mixer::Full(FullAttnLayer {
                            wq: load_t(e, src, &p("attn_q.weight"))?,
                            wk: load_t(e, src, &tp("attn_k.weight"))?,
                            wv: load_t(e, src, &tp("attn_v.weight"))?,
                            wo: load_t(e, src, &p("attn_output.weight"))?,
                            q_norm: load_t(e, src, &p("attn_q_norm.weight"))?,
                            k_norm: load_t(e, src, &tp("attn_k_norm.weight"))?,
                        })
                    } else {
                        load_mixer_kind(e, src, il, cfg.layer_kind(il))?
                    }
                },
                ffn: load_ffn(e, src, &cfg, il, spill.as_mut().map(|c| (gguf.unwrap(), c)))?,
                gemma4: if cfg.gemma4.is_some() {
                    let scalar = |n: &str| -> f32 {
                        let t = src.find(&p(n)).unwrap_or_else(|| panic!("missing {n}"));
                        bw24_gguf::dequant::dequantize(t.ggml_type, &t.bytes, 1)[0]
                    };
                    let vecf = |n: &str| -> Vec<f32> {
                        let t = src.find(&p(n)).unwrap_or_else(|| panic!("missing {n}"));
                        bw24_gguf::dequant::dequantize(t.ggml_type, &t.bytes,
                                                       t.ne.iter().product::<u64>() as usize)
                    };
                    let moe_bits = if src.find(&p("ffn_gate_inp.scale")).is_some() {
                        Some(crate::hybrid::Gemma4MoeBits {
                            post_ffw_norm_1: load_t(e, src, &p("post_ffw_norm_1.weight"))?,
                            pre_ffw_norm_2: load_t(e, src, &p("pre_ffw_norm_2.weight"))?,
                            post_ffw_norm_2: load_t(e, src, &p("post_ffw_norm_2.weight"))?,
                            shared_gate: load_t(e, src, &p("ffn_gate.weight"))?,
                            shared_up: load_t(e, src, &p("ffn_up.weight"))?,
                            shared_down: load_t(e, src, &p("ffn_down.weight"))?,
                            router_scale_pre: {
                                let inv = 1.0 / (cfg.n_embd as f32).sqrt();
                                let v: Vec<f32> = vecf("ffn_gate_inp.scale").iter().map(|x| x * inv).collect();
                                e.htod(&v)?
                            },
                            per_expert_scale: vecf("ffn_down_exps.scale"),
                            per_expert_scale_d: e.htod(&vecf("ffn_down_exps.scale"))?,
                        })
                    } else { None };
                    // E4B extras (tensor-presence: blk.N.inp_gate only exists on E4B)
                    let e4b = if src.has(&p("inp_gate.weight")) {
                        let g4 = cfg.gemma4.as_ref().unwrap();
                        let kv_from = n_trunk as u32 - g4.shared_kv_layers;
                        let kv_share = if g4.shared_kv_layers > 0 && il >= kv_from {
                            let swa = g4.swa_pattern.get(il as usize).copied().unwrap_or(true);
                            Some(kv_from - if swa { 2 } else { 1 })
                        } else { None };
                        Some(crate::hybrid::Gemma4E4bLayer {
                            inp_gate: load_t(e, src, &p("inp_gate.weight"))?,
                            proj: load_t(e, src, &p("proj.weight"))?,
                            post_norm: load_t(e, src, &p("post_norm.weight"))?,
                            kv_share,
                        })
                    } else { None };
                    Some(Gemma4LayerBits {
                        ffn_norm: load_t(e, src, &p("ffn_norm.weight"))?,
                        post_ffw_norm: load_t(e, src, &p("post_ffw_norm.weight"))?,
                        moe_bits,
                        layer_scale: scalar("layer_output_scale.weight"),
                        e4b,
                    })
                } else { None },
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
                    d2t: None,
                }),
                false => None,  // nextn>0 but no embedded eh_proj (external draft GGUF) -> no head
            }
        } else { None };

        // BW24_MTP_DRAFT=<path.gguf>: REPLACE the MTP head with one loaded from a standalone
        // draft GGUF (e.g. an FR-Spec trimmed-vocab draft). Verify-based spec decode stays exact
        // regardless of the draft — a different draft only changes WHICH tokens get proposed.
        let mtp = match std::env::var("BW24_MTP_DRAFT") {
            Ok(path) if !path.is_empty() => {
                eprintln!("[mtp-draft] loading external MTP draft: {path}");
                let dg = GgufFile::open(&path)?;
                Some(MtpHead::load_draft(e, &dg, &cfg)?)
            }
            _ => mtp,
        };

        // BW24_FRSPEC_TRIM=<frspec.gguf>: SELF-TRIMMED draft head. Reads ONLY the d2t ranked-token
        // list from the given file and gathers those rows from the MAIN model's own output.weight
        // bytes (quantized rows are independent — a byte-level row gather, zero requant). The MTP
        // block, norms, and head quant all stay main-model, so there is no cross-file quality
        // mismatch (the external Q4_K draft file measured -15pts acceptance vs the native block).
        // Draft lm_head reads drop vocab/32768-fold; verify stays full-vocab -> exactness unchanged.
        // FULL_PREC (MTP-heal ceiling): the self-trim gathers rows into `from_quant_bytes` (Quant
        // only) and, more to the point, the full-precision ceiling wants the model's NATURAL full
        // head — trimming the draft vocab is a speed lever, not part of the exactness measurement.
        // Disable trim under the flag (documented resolution, §item 2).
        let trim_env = std::env::var("BW24_FRSPEC_TRIM");
        if crate::model::full_prec_enabled() && trim_env.as_deref().map(|p| !p.is_empty()).unwrap_or(false) {
            eprintln!("[frspec-trim] DISABLED under BW24_FULL_PREC — using the natural full MTP head");
        }
        let mtp = match (if crate::model::full_prec_enabled() { Err(std::env::VarError::NotPresent) } else { trim_env }, mtp) {
            (Ok(path), Some(mut head)) if !path.is_empty() => {
                let tg = GgufFile::open(&path)?;
                let d2t_t = tg.find("d2t").expect("BW24_FRSPEC_TRIM file has no d2t tensor");
                let d2t_bytes = tg.tensor_data(d2t_t);
                let d2t: Vec<u32> = match d2t_t.ggml_type {
                    GgmlType::I32 => d2t_bytes.chunks_exact(4)
                        .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as u32).collect(),
                    GgmlType::I64 => d2t_bytes.chunks_exact(8)
                        .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as u32).collect(),
                    other => panic!("d2t must be I32/I64, got {other:?}"),
                };
                let v = src.find("output.weight")
                    .or_else(|| src.find("token_embd.weight"))
                    .expect("model has no output.weight for FR-Spec trim");
                let out_f = v.ne[1] as usize;
                let row_bytes = v.bytes.len() / out_f;
                assert!(d2t.iter().all(|&t| (t as usize) < out_f),
                        "d2t token id >= lm_head rows {out_f}");
                let mut gathered = Vec::with_capacity(d2t.len() * row_bytes);
                for &t in &d2t {
                    let off = t as usize * row_bytes;
                    gathered.extend_from_slice(&v.bytes[off..off + row_bytes]);
                }
                let trimmed = GpuTensor::from_quant_bytes(
                    e, &gathered, v.ggml_type, v.ne[0], d2t.len() as u64,
                    /*nvfp4 macro-scale*/ match src.find("output.scale") {
                        Some(sv) => f32::from_le_bytes(sv.bytes[..4].try_into().unwrap()),
                        None => 1.0,
                    })?;
                eprintln!("[frspec-trim] self-trimmed head: {} rows of main output.weight ({:?})",
                          d2t.len(), v.ggml_type);
                head.shared_head_head = Some(trimmed);
                head.d2t = Some(d2t);
                Some(head)
            }
            (_, m) => m,
        };

        if let Some(ctx) = spill.as_ref() {
            eprintln!("[spill] experts placed: {} pinned (Tier 1), {} mmap'd from disk (Tier 2, {} MiB)",
                      ctx.n_pinned, ctx.n_mmap, ctx.mmap_bytes >> 20);
        }

        if cfg.gemma4.is_some() {
            // gemma4 fa-vec crossover default (measured sweep 2026-07-10; env overrides).
            crate::FA_VEC_MIN_DEFAULT.store(1, std::sync::atomic::Ordering::Relaxed);
            // gemma4 rms_norm block 1024 (single-row 2816-col norms; battery-arbitrated per model).
            crate::RMS_BLOCK_DEFAULT.store(1024, std::sync::atomic::Ordering::Relaxed);
            // gemma4 fa split ladder (d1736 sweep; see fa_split_keys).
            crate::FA_SP_GEMMA.store(true, std::sync::atomic::Ordering::Relaxed);
            // depth fa: PARITY LAW (2026-07-10) — decode and verify share the rows_w/rows_dpl16
            // kernel symbols (decode t=1), so lane choice is freely tunable; v4 measured the
            // depth winner. Seams: BW24_FA_V4_MAX / BW24_FA_SMEM_TKV / BW24_GEMMA_ROWS_W.
        }
        // gemma4: the dc serving loop + spec draft gather read the device embed table every
        // step — upload it AT LOAD (OnceLock init) so first-use cost never lands in a timed span.
        let force_embd_gpu = cfg.gemma4.is_some();
        let gemma4_aux = if cfg.gemma4.is_some() {
            let rope_freqs = match src.find("rope_freqs.weight") {
                Some(t) => Some(e.htod(&bw24_gguf::dequant::dequantize(
                    t.ggml_type, &t.bytes, t.ne.iter().product::<u64>() as usize))?),
                None => None,
            };
            // E4B per-layer-embedding model tensors (tensor-presence gated).
            let e4b = match src.find("per_layer_token_embd.weight") {
                Some(t) => {
                    let n_epl = cfg.gemma4.as_ref().map(|g| g.n_embd_per_layer as usize).unwrap_or(0);
                    let row = t.ne[0] as usize;   // n_epl * n_layer
                    let row_bytes = t.bytes.len() / (t.ne[1] as usize);
                    eprintln!("[gemma4-e4b] per-layer-embed model detected (n_epl={n_epl}, row {row}) — \
                               first-light forward (eager decode + prime); dc/graph/spec unwired \
                               (HANDOVER-E4B.md)");
                    Some(crate::hybrid::Gemma4E4bModel {
                        tok_tbl_gpu: std::sync::OnceLock::new(),
                        tok_embd_bytes: t.bytes.to_vec(),
                        tok_embd_qt: match t.ggml_type {
                            bw24_gguf::GgmlType::Q6_K => crate::QT_Q6_K,
                            bw24_gguf::GgmlType::Q8_0 => crate::QT_Q8_0,
                            other => panic!("e4b per-layer tok embd: unhandled dtype {other:?}"),
                        },
                        tok_embd_row_bytes: row_bytes,
                        model_proj: load_t(e, src, "per_layer_model_proj.weight")?,
                        proj_norm: load_t(e, src, "per_layer_proj_norm.weight")?,
                        n_epl,
                    })
                }
                None => None,
            };
            Some(GemmaAux { rope_freqs, ones: e.htod(&[1.0f32; 512])?, e4b })
        } else { None };
        let mut layers = layers;
        // Q4_0 SPLIT-PLANE DECODE MIRRORS (2026-07-10, BW24_Q4RP seam): gemma-4 MoE-class trunk
        // (26B — attn wq/wk/wv/wo + the parallel shared FFN triple). The 18B GGUF block stride
        // costs ~25-35% decode bandwidth in sector overfetch (rp_q4_probe: m=1 1.34x, m=3 1.17x,
        // bitwise); the mirror (~0.7GB for the 26B) fixes the m<=8 mmvq/batched/fused family.
        // Dense 31B is NOT mirrored (its 15GB trunk mirror does not fit 24GB — the full layout
        // swap is the follow-up arc); raw bytes stay for prefill/gemm/Stage-A either way.
        if cfg.gemma4.is_some() && crate::Engine::q4rp_enabled() {
            let mut nmir = 0usize;
            for layer in layers.iter_mut() {
                let mirror_layer = layer.gemma4.as_ref().is_some_and(|g| g.moe_bits.is_some());
                if !mirror_layer { continue; }
                if let Mixer::Full(fa) = &mut layer.mixer {
                    for w in [&mut fa.wq, &mut fa.wk, &mut fa.wv, &mut fa.wo] {
                        e.build_q4_rp4(w)?; nmir += 1;
                    }
                }
                if let Some(mb) = layer.gemma4.as_mut().unwrap().moe_bits.as_mut() {
                    for w in [&mut mb.shared_gate, &mut mb.shared_up, &mut mb.shared_down] {
                        e.build_q4_rp4(w)?; nmir += 1;
                    }
                }
            }
            if nmir > 0 { eprintln!("[q4rp] split-plane decode mirrors built: {nmir} trunk tensors"); }
        }
        let model = HybridModel { cfg, embd, output_norm, output, layers, mtp,
                                  embd_gpu: std::sync::OnceLock::new(), gemma4_aux };
        if force_embd_gpu {
            let _ = model.embd_gpu.get_or_init(|| {
                e.upload_u8(&model.embd.raw).expect("embed table upload")
            });
        }
        Ok(model)
    }

    pub fn embed(&self, e: &Engine, tokens: &[u32]) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let n_embd = self.cfg.n_embd as usize;
        let x = self.embd.gather(n_embd, tokens);
        Ok(e.htod(&x)?)
    }
}
