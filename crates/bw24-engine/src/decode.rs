//! Incremental decode (T=1) with the dual cache + greedy generation loop. Serves end-to-end.
//! Reuses the validated kernels; threads KV (full-attn) and conv/SSM state (linear-attn) across steps.

use cudarc::driver::CudaSlice;
use std::collections::HashMap;
use crate::Engine;
use crate::hybrid::{HybridModel, Mixer, FullAttnLayer, LinearAttnLayer};
use crate::cache::{Cache, RecurLayer};
use crate::forward::argmax;

/// Persistent CUDA-graph decode state (CUDA-GRAPH-PLAN Phase 3). Holds the device-resident counters
/// the captured graph reads/writes (`token_d` = current/next token id, `pos_d` = rope position) — both
/// at FIXED addresses baked into every captured graph — plus the per-`t_kv`-bucket graph cache. The
/// bucket key is the eager `(fa_vec, n_splits)` pair (see `Engine::fa_bucket_key`): every t_kv that
/// maps to the same key reproduces eager's split geometry, so one captured graph replays bit-identically
/// for the whole bucket. A new key triggers a re-capture (n_splits changes ~every 64 tokens).
pub struct GraphDecodeState {
    pub token_d: CudaSlice<u32>,        // [1] resident next-token id (argmax writes, embed reads)
    pub pos_d: CudaSlice<i32>,          // [1] resident rope position counter
    pub graphs: HashMap<(bool, usize), cudarc::driver::CudaGraph>,
    pub bucket_max: HashMap<(bool, usize), usize>,  // bucket key -> bucket_max fed to the capture
    pub captures: usize,                // count of (re)captures, for reporting
}

impl GraphDecodeState {
    pub fn new(e: &Engine) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(GraphDecodeState {
            token_d: e.stream().clone_htod(&[0u32])?,
            pos_d: e.htod_i32(&[0])?,
            graphs: HashMap::new(),
            bucket_max: HashMap::new(),
            captures: 0,
        })
    }
}

/// Generation parameters for the reusable serving API (`generate_with`).
#[derive(Clone, Debug)]
pub struct GenParams {
    pub max_new: usize,            // hard cap on generated tokens
    pub max_ctx: Option<usize>,    // context-length guard; None => prompt+max_new+8
    pub eos: Vec<u32>,             // stop on any of these token ids (eos/eog + specials)
}
impl Default for GenParams {
    fn default() -> Self { GenParams { max_new: 128, max_ctx: None, eos: Vec::new() } }
}

/// Why generation stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason { Eos, MaxNew, ContextFull, Callback }

/// Result of `generate_with`: the generated token ids + why it stopped.
pub struct GenOutput {
    pub tokens: Vec<u32>,
    pub stop_reason: StopReason,
}

impl HybridModel {
    /// One decode step for `token` at cache.pos; returns logits [n_vocab] (host f32). Advances cache.
    pub fn decode_step(&self, e: &Engine, token: u32, cache: &mut Cache) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        Ok(self.decode_step_h(e, token, cache)?.0)
    }

    /// Dense-FFN SwiGLU (T=1 decode): `down @ (silu(gate@z) * (up@z))`. Two fused levers stack here:
    ///  - RANK3 LEVER 2: gate+up NVFP4 macro-scales fold into ONE `silu_mul_scaled*` launch (via
    ///    `matmul_pre_noscale`), saving the two separate `scale_inplace` launches.
    ///  - RANK2 LEVER (q8_1 quant-fold): when ffn_down is ALSO on the q8_1 fast path, the SwiGLU
    ///    epilogue EMITS the q8_1 quantization of `act` directly (`silu_mul_scaled_q8_1`) and feeds
    ///    ffn_down via `matmul_pre`, removing ffn_down's standalone `quantize_q8_1` launch (the
    ///    down-proj activation has one consumer, so the quant folds into its producer for free).
    /// BIT-IDENTICAL to matmul_pre(gate)+matmul_pre(up)+silu_mul+quantize_q8_1+matmul(down): same
    /// float silu*mul, same amax/127 q8_1 rounding, same dp4a/mmvq dot. Falls back to the f32 `act`
    /// + plain matmul(down) path whenever any of the three is off the fast path.
    fn ffn_swiglu_decode(&self, e: &Engine, ffn_gate: &crate::model::GpuTensor,
                         ffn_up: &crate::model::GpuTensor, ffn_down: &crate::model::GpuTensor,
                         z: &CudaSlice<f32>, n_embd: usize, n_ff: usize)
                         -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        // M3 dense layers use swigluoai (clamped) — the silu_mul fused fast paths below encode
        // plain SiLU; route through ffn_act (macro-scales folded via matmul_pre) until clamped
        // fused twins exist.
        if self.cfg.m3.is_some() {
            let (zq, zd) = e.quantize_q8_1(z, 1, n_embd)?;
            let gate = e.matmul_pre(ffn_gate, &zq, &zd, z, 1)?;
            let up = e.matmul_pre(ffn_up, &zq, &zd, z, 1)?;
            let mut act = e.uninit(n_ff)?;
            Self::ffn_act(e, &self.cfg, &gate, &up, &mut act, n_ff)?;
            return Ok(e.matmul(ffn_down, &act, 1)?);
        }
        if e.uses_q8_1_fast(ffn_gate) && e.uses_q8_1_fast(ffn_up) {
            let (zq, zd) = e.quantize_q8_1(z, 1, n_embd)?;
            // DUAL mm-fusion first (NVFP4 gate+up in ONE launch), else two noscale launches.
            let pair = match e.matmul_pre_dual_noscale(ffn_gate, ffn_up, &zq, &zd, 1)? {
                Some((g, u)) => (Some(g), Some(u)),
                None => (e.matmul_pre_noscale(ffn_gate, &zq, &zd, 1)?, e.matmul_pre_noscale(ffn_up, &zq, &zd, 1)?),
            };
            match pair {
                (Some((gate, gs)), Some((up, us))) => {
                    // RANK2 fold: if ffn_down is q8_1-fast, emit act PRE-QUANTIZED and skip the
                    // standalone quantize_q8_1 before ffn_down.
                    if e.uses_q8_1_fast(ffn_down) {
                        let (aq, ad) = e.silu_mul_scaled_q8_1(&gate, &up, gs, us, n_ff)?;
                        return Ok(e.matmul_pre(ffn_down, &aq, &ad, /*x_fallback unused on fast path*/ &gate, 1)?);
                    }
                    let mut act = e.uninit(n_ff)?;
                    e.silu_mul_scaled(&gate, &up, gs, us, &mut act, n_ff)?;
                    return Ok(e.matmul(ffn_down, &act, 1)?);
                }
                _ => {
                    // one (or both) not on the separable-scale fast path: scaled matmul + plain silu_mul.
                    let gate = e.matmul_pre(ffn_gate, &zq, &zd, z, 1)?;
                    let up = e.matmul_pre(ffn_up, &zq, &zd, z, 1)?;
                    let mut act = e.uninit(n_ff)?;
                    Self::ffn_act(e, &self.cfg, &gate, &up, &mut act, n_ff)?;
                    return Ok(e.matmul(ffn_down, &act, 1)?);
                }
            }
        }
        let gate = e.matmul(ffn_gate, z, 1)?;
        let up = e.matmul(ffn_up, z, 1)?;
        let mut act = e.uninit(n_ff)?;
        Self::ffn_act(e, &self.cfg, &gate, &up, &mut act, n_ff)?;
        Ok(e.matmul(ffn_down, &act, 1)?)
    }

    /// Like `ffn_swiglu_decode` but the input is ALREADY q8_1-quantized `(zq, zd)` — used by the
    /// DECODE NORM-FUSION lever where `add_rms_norm_q8_1` emits the post-attn-normed activation
    /// pre-quantized (no f32 `z` materialized, no standalone quantize_q8_1 launch). Caller GUARANTEES
    /// ffn_gate and ffn_up are q8_1-fast (so `matmul_pre_noscale` returns Some at m=1). BIT-IDENTICAL
    /// to ffn_swiglu_decode(z) when (zq,zd) == quantize_q8_1(z): same matmul_pre_noscale, same
    /// silu_mul_scaled_q8_1 / silu_mul_scaled, same ffn_down dot.
    fn ffn_swiglu_decode_pre(&self, e: &Engine, ffn_gate: &crate::model::GpuTensor,
                             ffn_up: &crate::model::GpuTensor, ffn_down: &crate::model::GpuTensor,
                             zq: &CudaSlice<i8>, zd: &CudaSlice<f32>, n_ff: usize)
                             -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let pair = match e.matmul_pre_dual_noscale(ffn_gate, ffn_up, zq, zd, 1)? {
            Some((g, u)) => (Some(g), Some(u)),
            None => (e.matmul_pre_noscale(ffn_gate, zq, zd, 1)?, e.matmul_pre_noscale(ffn_up, zq, zd, 1)?),
        };
        match pair {
            (Some((gate, gs)), Some((up, us))) => {
                if e.uses_q8_1_fast(ffn_down) {
                    let (aq, ad) = e.silu_mul_scaled_q8_1(&gate, &up, gs, us, n_ff)?;
                    Ok(e.matmul_pre(ffn_down, &aq, &ad, &gate, 1)?)
                } else {
                    let mut act = e.uninit(n_ff)?;
                    e.silu_mul_scaled(&gate, &up, gs, us, &mut act, n_ff)?;
                    Ok(e.matmul(ffn_down, &act, 1)?)
                }
            }
            // Unreachable when the caller's q8_1-fast guarantee holds (m==1 + fast => Some). Guard
            // anyway: re-quant from the dequantized pair would need f32; surface a clear error.
            _ => Err("ffn_swiglu_decode_pre: gate/up not separable-scale at m=1 (caller must guarantee q8_1-fast)".into()),
        }
    }

    /// Shared post-attention residual + post-attn-norm + FFN for ONE decode layer, routed by ALL
    /// decode loops (eager + dc + dc_cap) so they stay bit-identical by construction. DECODE
    /// NORM-FUSION LEVER: when the layer is Dense AND ffn_gate/ffn_up are q8_1-fast (the daily NVFP4
    /// case), fuses residual-add + post_attn_norm + q8_1-quantize into ONE `add_rms_norm_q8_1` launch
    /// and feeds the FFN the pre-quantized activation (skipping its internal quantize_q8_1) — removing
    /// 1-2 launches + the f32 `z` HBM round-trip per layer. BIT-IDENTICAL to the unfused
    /// add_rms_norm(or add+rms_norm) + quantize_q8_1 + ffn (all proven bit-identical in kernel_check).
    /// BW24_NO_FUSE_NORMQ forces the unfused f32 path. Returns (x1 residual f32, ffn_out f32).
    /// True when ALL of a mixer's input projections are on the q8_1 fast path (so the attn-input
    /// rms_norm can emit q8_1 directly and the mixer skips its internal quantize_q8_1).
    pub(crate) fn mixer_in_q8_1_fast(&self, e: &Engine, mixer: &Mixer) -> bool {
        match mixer {
            Mixer::Full(fa) => e.uses_q8_1_fast(&fa.wq) && e.uses_q8_1_fast(&fa.wk) && e.uses_q8_1_fast(&fa.wv),
            Mixer::Linear(la) => e.uses_q8_1_fast(&la.wqkv) && e.uses_q8_1_fast(&la.wqkv_gate)
                && e.uses_q8_1_fast(&la.ssm_beta) && e.uses_q8_1_fast(&la.ssm_alpha),
        }
    }

    /// attn_norm + mixer for the EAGER loop, with the attn-input NORM-FUSION. BW24_NO_FUSE_NORMQ
    /// forces the unfused (separate rms_norm + mixer-internal quantize) path.
    fn attn_in_norm_mixer(&self, e: &Engine, layer: &crate::hybrid::HybridLayer, x: &CudaSlice<f32>,
                          pos_d: &CudaSlice<i32>, pos: usize, cache: &mut Cache, il: usize,
                          n_embd: usize, eps: f32)
                          -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let anorm = layer.attn_norm.float_data();
        let fuse = std::env::var("BW24_NO_FUSE_NORMQ").is_err() && self.mixer_in_q8_1_fast(e, &layer.mixer);
        if fuse {
            let (hq, hd) = e.rms_norm_q8_1(x, anorm, n_embd, 1, eps)?;
            // h is unused on the fast path (matmul_pre x_fallback only used at m>=16); pass a zero-len.
            let h0 = e.zeros(0)?;
            match &layer.mixer {
                Mixer::Full(fa) => self.full_attn_decode_pre(e, fa, &h0, Some((&hq, &hd)), pos_d, pos, cache, il),
                Mixer::Linear(la) => self.linear_attn_decode_pre(e, la, &h0, &hq, &hd, cache, il, false),
            }
        } else {
            let mut h = e.uninit(n_embd)?;
            e.rms_norm(x, anorm, &mut h, n_embd, 1, eps)?;
            match &layer.mixer {
                Mixer::Full(fa) => self.full_attn_decode(e, fa, &h, pos_d, pos, cache, il),
                Mixer::Linear(la) => self.linear_attn_decode(e, la, &h, cache, il),
            }
        }
    }

    /// attn_norm + mixer for the DEVICE-COUNTER loop (decode_step_dc). Full-attn uses the dc path;
    /// linear uses the eager-state path (persistent=false), same as decode_step_dc. NORM-FUSED.
    fn attn_in_norm_mixer_dc(&self, e: &Engine, layer: &crate::hybrid::HybridLayer, x: &CudaSlice<f32>,
                             pos_d: &CudaSlice<i32>, cache: &mut Cache, il: usize, n_embd: usize, eps: f32)
                             -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let anorm = layer.attn_norm.float_data();
        let fuse = std::env::var("BW24_NO_FUSE_NORMQ").is_err() && self.mixer_in_q8_1_fast(e, &layer.mixer);
        if fuse {
            let (hq, hd) = e.rms_norm_q8_1(x, anorm, n_embd, 1, eps)?;
            let h0 = e.zeros(0)?;
            match &layer.mixer {
                Mixer::Full(fa) => self.full_attn_decode_dc_pre(e, fa, &h0, &hq, &hd, pos_d, cache, il),
                Mixer::Linear(la) => self.linear_attn_decode_pre(e, la, &h0, &hq, &hd, cache, il, false),
            }
        } else {
            let mut h = e.uninit(n_embd)?;
            e.rms_norm(x, anorm, &mut h, n_embd, 1, eps)?;
            match &layer.mixer {
                Mixer::Full(fa) => self.full_attn_decode_dc(e, fa, &h, pos_d, cache, il),
                Mixer::Linear(la) => self.linear_attn_decode(e, la, &h, cache, il),
            }
        }
    }

    /// attn_norm + mixer for the CAPTURE loop (decode_step_dc_cap). Full-attn uses the dc_cap path
    /// (fixed bucket_max); linear uses the persistent-state path. NORM-FUSED; capture-safe (rms_norm_q8_1
    /// + the *_pre mixers enqueue the same kernels every replay, stable buffers).
    fn attn_in_norm_mixer_dc_cap(&self, e: &Engine, layer: &crate::hybrid::HybridLayer, x: &CudaSlice<f32>,
                                 pos_d: &CudaSlice<i32>, cache: &mut Cache, il: usize, bucket_max: usize,
                                 n_embd: usize, eps: f32)
                                 -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let anorm = layer.attn_norm.float_data();
        let fuse = std::env::var("BW24_NO_FUSE_NORMQ").is_err() && self.mixer_in_q8_1_fast(e, &layer.mixer);
        if fuse {
            let (hq, hd) = e.rms_norm_q8_1(x, anorm, n_embd, 1, eps)?;
            let h0 = e.zeros(0)?;
            match &layer.mixer {
                Mixer::Full(fa) => self.full_attn_decode_dc_cap_pre(e, fa, &h0, &hq, &hd, pos_d, cache, il, bucket_max),
                Mixer::Linear(la) => self.linear_attn_decode_pre(e, la, &h0, &hq, &hd, cache, il, true),
            }
        } else {
            let mut h = e.uninit(n_embd)?;
            e.rms_norm(x, anorm, &mut h, n_embd, 1, eps)?;
            match &layer.mixer {
                Mixer::Full(fa) => self.full_attn_decode_dc_cap(e, fa, &h, pos_d, cache, il, bucket_max),
                Mixer::Linear(la) => self.linear_attn_decode_cap(e, la, &h, cache, il),
            }
        }
    }

    fn residual_norm_ffn(&self, e: &Engine, layer: &crate::hybrid::HybridLayer, x: &CudaSlice<f32>,
                         mixed: &CudaSlice<f32>, n_embd: usize, il: usize, eps: f32)
                         -> Result<(CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let pnorm = layer.post_attn_norm.float_data();
        match &layer.ffn {
            crate::hybrid::Ffn::Dense { ffn_gate, ffn_up, ffn_down } => {
                let n_ff = ffn_gate.out_features();
                // cfg.m3: the fused-pre chain's silu_mul_scaled* epilogues are plain SiLU —
                // M3's swigluoai must route through ffn_swiglu_decode's m3 arm (FAST-gate
                // MISMATCH root cause #2, 2026-07-07: L0 dense FFN clamp skipped under FAST).
                let fuse = std::env::var("BW24_NO_FUSE_NORMQ").is_err()
                    && self.cfg.m3.is_none()
                    && e.uses_q8_1_fast(ffn_gate) && e.uses_q8_1_fast(ffn_up);
                if fuse {
                    let mut x1 = e.uninit(n_embd)?;
                    let (zq, zd) = e.add_rms_norm_q8_1(x, mixed, pnorm, &mut x1, n_embd, 1, eps)?;
                    let ffn_out = self.ffn_swiglu_decode_pre(e, ffn_gate, ffn_up, ffn_down, &zq, &zd, n_ff)?;
                    Ok((x1, ffn_out))
                } else {
                    let mut x1 = e.uninit(n_embd)?;
                    let mut z = e.uninit(n_embd)?;
                    e.add_rms_norm(x, mixed, pnorm, &mut x1, &mut z, n_embd, 1, eps)?;
                    let ffn_out = self.ffn_swiglu_decode(e, ffn_gate, ffn_up, ffn_down, &z, n_embd, n_ff)?;
                    Ok((x1, ffn_out))
                }
            }
            crate::hybrid::Ffn::Moe(m) => {
                let mut x1 = e.uninit(n_embd)?;
                let mut z = e.uninit(n_embd)?;
                // z-quantize fuse (add_rms_norm_zq8) measured NEGATIVE here (158.8 vs 160.6:
                // the fused warp-per-block quantize pass re-reads z slower than the dedicated
                // coalesced quantize_q8_1). Kernel + threading kept for graph-capture use where
                // launch count matters more; eager default = unfused (no gain = no change).
                e.add_rms_norm(x, mixed, pnorm, &mut x1, &mut z, n_embd, 1, eps)?;
                let ffn_out = self.moe_ffn_il_zq8(e, m, &z, None, 1, il as u16)?;
                Ok((x1, ffn_out))
            }
        }
    }

    /// EAGLE3 aux-hidden capture (EAGLE-PLAN N1): one decode step that ALSO returns the trunk
    /// residual-stream `x` taken AFTER each of the blocks in `aux_layers` (the EAGLE3 encoder feeds
    /// these 3 layer hiddens through `fc`). Returns (logits[n_vocab] host, aux: Vec<[n_embd] dev>),
    /// one device buffer per requested aux layer, in `aux_layers` order. The captured tensor is the
    /// residual `x` produced by that block (`x2` at the loop tail), cloned before the next block
    /// overwrites it — cheap (one clone_dtod of [n_embd] per aux layer). T=1 decode regime.
    pub fn decode_step_aux(&self, e: &Engine, token: u32, cache: &mut Cache, aux_layers: &[usize])
                           -> Result<(Vec<f32>, Vec<CudaSlice<f32>>), Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let n_embd = cfg.n_embd as usize;
        let eps = cfg.rms_eps;
        let pos = cache.pos;
        let pos_d = e.htod_i32(&[pos as i32])?;

        let mut x = e.htod(&self.embd.gather(n_embd, &[token]))?;
        let mut aux: Vec<CudaSlice<f32>> = Vec::with_capacity(aux_layers.len());

        for (il, layer) in self.layers.iter().enumerate() {
            // attn-input NORM-FUSION (eager); shared with decode_step_h.
            let mixed = self.attn_in_norm_mixer(e, layer, &x, &pos_d, pos, cache, il, n_embd, eps)?;
            // DECODE NORM-FUSION LEVER (residual_norm_ffn): residual add + post_attn RMSNorm +
            // q8_1-quantize fused into ONE add_rms_norm_q8_1 launch on the Dense q8_1-fast path, then
            // the FFN consumes the pre-quantized activation. Bit-identical to the unfused path.
            let (x1, ffn_out) = self.residual_norm_ffn(e, layer, &x, &mixed, n_embd, il, eps)?;
            let mut x2 = e.uninit(n_embd)?;
            e.add(&x1, &ffn_out, &mut x2, n_embd)?;
            // EAGLE3 N1: capture this block's residual output if it is an aux layer.
            if aux_layers.contains(&il) { aux.push(e.clone_dtod(&x2)?); }
            x = x2;
        }
        // re-order aux to match aux_layers order (contains() pushes in il order; aux_layers is the
        // canonical order the encoder concats in — they coincide since aux_layers is ascending).
        let mut hn = e.uninit(n_embd)?;
        e.rms_norm(&x, self.output_norm.float_data(), &mut hn, n_embd, 1, eps)?;
        let logits = e.matmul(&self.output, &hn, 1)?;
        let host = e.dtoh(&logits)?;
        cache.pos += 1;
        Ok((host, aux))
    }

    /// Like `decode_step`, but ALSO returns the trunk's hidden state `x` taken BEFORE the final
    /// `output_norm` (MTP-PLAN §A: this is `h_seed` for the NextN head). Device buffer [n_embd].
    pub fn decode_step_h(&self, e: &Engine, token: u32, cache: &mut Cache)
                         -> Result<(Vec<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        if self.is_gemma4_e4b() { return self.gemma4_e4b_decode_step_h(e, token, cache); }
        if self.cfg.gemma4.is_some() { return self.gemma4_decode_step_h(e, token, cache); }
        let cfg = &self.cfg;
        let n_embd = cfg.n_embd as usize;
        let eps = cfg.rms_eps;
        let pos = cache.pos;
        let pos_d = e.htod_i32(&[pos as i32])?;

        // embed the single token -> [1, n_embd]
        let mut x = e.htod(&self.embd.gather(n_embd, &[token]))?;

        // CROSS-LAYER ADD+NORM FUSION (launch-arc 2026-07-07): layer il's post-FFN residual add
        // (x2 = x1 + ffn_out) and layer il+1's attn_norm+quantize are consecutive row-wise ops —
        // add_rms_norm_q8_1 does all three in ONE launch (bit-identity proven in kernel_check:
        // add_rms_norm == add then rms_norm; _q8_1 == then quantize_q8_1). Carry the un-added
        // (x1, ffn_out) pair into the next iteration; the fused launch materializes x2 (the
        // residual this layer needs) as its `res` output. Falls back to the separate add when
        // the next mixer is off the q8_1 fast path.
        let mut pending: Option<(CudaSlice<f32>, CudaSlice<f32>)> = None;
        for (il, layer) in self.layers.iter().enumerate() {
            let anorm = layer.attn_norm.float_data();
            let fuse = std::env::var("BW24_NO_FUSE_NORMQ").is_err()
                && self.mixer_in_q8_1_fast(e, &layer.mixer);
            // NOTE: take() FIRST, branch on fuse after — a tuple pattern like
            // `if let (Some(p), true) = (pending.take(), fuse)` DROPS the taken pair when
            // fuse is false (pattern fails post-take) and silently loses the residual add.
            let taken = pending.take();
            let mixed = match (taken, fuse) {
                (Some((x1, f1)), true) => {
                    // fused add + attn_norm + q8_1 (this layer's mixer input), res -> x2
                    let mut x2 = e.uninit(n_embd)?;
                    let (hq, hd) = e.add_rms_norm_q8_1(&x1, &f1, anorm, &mut x2, n_embd, 1, eps)?;
                    x = x2;
                    let h0 = e.zeros(0)?;
                    match &layer.mixer {
                        Mixer::Full(fa) => self.full_attn_decode_pre(e, fa, &h0, Some((&hq, &hd)), &pos_d, pos, cache, il)?,
                        Mixer::Linear(la) => self.linear_attn_decode_pre(e, la, &h0, &hq, &hd, cache, il, false)?,
                    }
                }
                (taken, _) => {
                    if let Some((x1, f1)) = taken {
                        let mut x2 = e.uninit(n_embd)?;
                        e.add(&x1, &f1, &mut x2, n_embd)?;
                        x = x2;
                    }
                    self.attn_in_norm_mixer(e, layer, &x, &pos_d, pos, cache, il, n_embd, eps)?
                }
            };

            // DECODE NORM-FUSION LEVER (residual_norm_ffn): add+post_attn_norm+q8_1 fused on the Dense
            // fast path. Bit-identical to add + rms_norm + ffn (add_rms_norm == add then rms_norm,
            // proven in kernel_check; add_rms_norm_q8_1 == add_rms_norm then quantize_q8_1).
            let (x1, ffn_out) = self.residual_norm_ffn(e, layer, &x, &mixed, n_embd, il, eps)?;
            pending = Some((x1, ffn_out));
        }
        // final layer's add (no next norm to fuse with — output_norm is f32-out)
        if let Some((x1, f1)) = pending.take() {
            let mut x2 = e.uninit(n_embd)?;
            e.add(&x1, &f1, &mut x2, n_embd)?;
            x = x2;
        }

        let mut hn = e.uninit(n_embd)?;
        e.rms_norm(&x, self.output_norm.float_data(), &mut hn, n_embd, 1, eps)?;
        // h_seed = trunk hidden BEFORE output_norm (default, §A) or AFTER it (BW24_SPEC_HPOST,
        // the reference engines' convention — see spec::spec_hpost).
        let h_seed = if crate::spec::spec_hpost() { e.clone_dtod(&hn)? } else { e.clone_dtod(&x)? };
        // head-MIPS feasibility probe (BW24_DUMP_HN=<path>): append pre-head hiddens for
        // offline bound analysis. Diagnostic only.
        if let Ok(path) = std::env::var("BW24_DUMP_HN") {
            let hh = e.dtoh(&hn)?;
            use std::io::Write;
            let mut fo = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
            for v in &hh { fo.write_all(&v.to_le_bytes())?; }
        }
        let logits = e.matmul(&self.output, &hn, 1)?;
        let host = e.dtoh(&logits)?;
        cache.pos += 1;
        Ok((host, h_seed))
    }

    /// DEVICE-COUNTER decode step (CUDA-GRAPH-PLAN Phase 2). A clone of `decode_step_h` that removes
    /// the two per-step VARYING host kernel-args by reading them from device counters:
    ///   1. the KV-append write slot  -> per-layer `kvl.len_d` (device i32[1])
    ///   2. the fa_decode t_kv bound   -> the same `kvl.len_d` after `inc_seqlen`
    /// plus it keeps the token id + rope pos DEVICE-RESIDENT (embed_gather_device, device rope pos,
    /// argmax_token_device). NO graph capture yet — runs the kernels eagerly through the counter
    /// path. Must be BIT-IDENTICAL to `decode_step_h`'s token stream (the gate).
    ///
    /// Args: `token_d` = resident device token id [1] (this step's input token); `pos_d` = resident
    /// device rope pos i32[1] (== cache.pos at entry; INCREMENTED in-path); `embd_gpu` = resident embed
    /// table; (qt,row_bytes) from EmbedHost::qt_and_row_bytes. Returns the NEXT token id device buffer.
    /// `cache.pos` and each `kvl.len`/`kvl.len_d` are advanced to match `decode_step_h`.
    pub fn decode_step_dc(&self, e: &Engine, token_d: &CudaSlice<u32>, pos_d: &mut CudaSlice<i32>,
                          embd_gpu: &CudaSlice<u8>, embd_qt: i32, embd_row_bytes: usize,
                          cache: &mut Cache, n_vocab: usize)
                          -> Result<CudaSlice<u32>, Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let n_embd = cfg.n_embd as usize;
        let eps = cfg.rms_eps;

        // embed the single (DEVICE-resident) token -> [1, n_embd], no host round-trip of the id.
        let mut x = e.embed_gather_device(embd_gpu, token_d, n_embd, embd_qt, embd_row_bytes)?;

        for (il, layer) in self.layers.iter().enumerate() {
            // attn-input NORM-FUSION (dc path); bit-identical to decode_step_h (Phase-2 gate).
            let mixed = self.attn_in_norm_mixer_dc(e, layer, &x, pos_d, cache, il, n_embd, eps)?;

            // DECODE NORM-FUSION LEVER (residual_norm_ffn): see decode_step_h. Shared helper -> dc
            // path stays bit-identical to decode_step_h's token stream (the Phase-2 gate).
            let (x1, ffn_out) = self.residual_norm_ffn(e, layer, &x, &mixed, n_embd, il, eps)?;
            let mut x2 = e.uninit(n_embd)?;
            e.add(&x1, &ffn_out, &mut x2, n_embd)?;
            x = x2;
        }

        let mut hn = e.uninit(n_embd)?;
        e.rms_norm(&x, self.output_norm.float_data(), &mut hn, n_embd, 1, eps)?;
        let logits = e.matmul(&self.output, &hn, 1)?;
        // device argmax -> next token id stays resident (no logits dtoh).
        let next_tok = e.argmax_token_device(&logits, n_vocab)?;
        // advance rope pos counter on-device (replaces the per-step htod_i32(&[pos])).
        e.inc_seqlen(pos_d)?;
        cache.pos += 1;
        Ok(next_tok)
    }

    /// CAPTURE body for CUDA-graph replay (CUDA-GRAPH-PLAN Phase 3). One full decode step enqueued
    /// entirely on `e.stream()` with ZERO host sync and ZERO per-step varying host kernel-args:
    ///   - embed reads the PERSISTENT device `token_d` (last step's argmax), writes scratch `x`.
    ///   - full-attn layers size n_splits from `bucket_max` (fixed for this capture); the kernel reads
    ///     the ACTUAL t_kv from the device counter `kvl.len_d`. KV append + device-counter inc happen
    ///     in-graph. The host `kvl.len`/`cache.pos` are NOT advanced here (the driver advances the host
    ///     mirrors once per replay; only the DEVICE counters advance inside the graph).
    ///   - linear-attn layers use the persistent-state variant (copy-back, stable pointers).
    ///   - lm_head -> parallel 2-pass argmax (`argmax_partial_f32`+`argmax_final_f32`) writes the
    ///     next id into the PERSISTENT `token_d`.
    ///   - `inc_seqlen(pos_d)` advances the rope-pos device counter in-graph.
    /// Captured ONCE per `bucket_max`; replayed for every t_kv in that bucket. Bit-identical to eager
    /// when `bucket_max` reproduces eager's n_splits for the replayed t_kv (the bucket-key contract).
    pub fn decode_step_dc_cap(&self, e: &Engine, token_d: &mut CudaSlice<u32>, pos_d: &mut CudaSlice<i32>,
                          embd_gpu: &CudaSlice<u8>, embd_qt: i32, embd_row_bytes: usize,
                          cache: &mut Cache, n_vocab: usize, bucket_max: usize)
                          -> Result<(), Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let n_embd = cfg.n_embd as usize;
        let eps = cfg.rms_eps;

        let mut x = e.embed_gather_device(embd_gpu, token_d, n_embd, embd_qt, embd_row_bytes)?;

        for (il, layer) in self.layers.iter().enumerate() {
            // attn-input NORM-FUSION (capture path); capture-safe + bit-identical to eager.
            let mixed = self.attn_in_norm_mixer_dc_cap(e, layer, &x, pos_d, cache, il, bucket_max, n_embd, eps)?;
            // DECODE NORM-FUSION LEVER (residual_norm_ffn): see decode_step_aux. Shared helper keeps
            // the capture path bit-identical to eager by construction.
            let (x1, ffn_out) = self.residual_norm_ffn(e, layer, &x, &mixed, n_embd, il, eps)?;
            let mut x2 = e.uninit(n_embd)?;
            e.add(&x1, &ffn_out, &mut x2, n_embd)?;
            x = x2;
        }

        let mut hn = e.uninit(n_embd)?;
        e.rms_norm(&x, self.output_norm.float_data(), &mut hn, n_embd, 1, eps)?;
        let logits = e.matmul(&self.output, &hn, 1)?;
        // argmax into the PERSISTENT token_d (next step's embed reads it) — same buffer pointer baked
        // at capture, written each replay, so the token id never round-trips to host in steady state.
        e.argmax_token_device_into(&logits, token_d, n_vocab)?;
        e.inc_seqlen(pos_d)?;
        Ok(())
    }

    /// CUDA-GRAPH decode driver (CUDA-GRAPH-PLAN Phase 3). Primes the prompt EAGERLY (device-counter
    /// `decode_step_dc`, advancing host + device counters together), then generates `max_new` tokens by
    /// CUDA-graph REPLAY: per step it picks the t_kv bucket key, captures a graph on first sight of that
    /// key (re-using the SAME persistent counters/cache so replays continue the sequence), and replays.
    /// The argmax-written next token stays device-resident in `gs.token_d`; we read back only the [1]
    /// u32 after each launch (the gate compares it; a real server can defer this). Returns the generated
    /// token ids. Greedy. Bit-identical to eager `decode_step` (the gate).
    ///
    /// CAPTURE STATE HYGIENE: `capture_graph` runs the step body 3x (2 warmup + 1 capture), each of
    /// which mutates the device KV/conv/ssm/counter state. We SNAPSHOT the cache + device counters +
    /// token id before capturing and RESTORE them after, so the 3 throwaway runs leave zero residue and
    /// replay resumes from the true pre-capture state.
    pub fn generate_graph(&self, e: &Engine, gs: &mut GraphDecodeState, prompt: &[u32], max_new: usize)
                          -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let n_embd = self.cfg.n_embd as usize;
        let head_dim = self.cfg.head_dim_k as usize;
        let (qt, row_bytes) = self.embd.qt_and_row_bytes(n_embd);

        // EVENT TRACKING OFF for the WHOLE graph-decode session. cudarc records a per-CudaSlice event
        // (the Engine is in multi-stream mode via copy_stream) and inserts `stream.wait(event)` on every
        // kernel arg whose buffer was touched — those waits are illegal inside a capture region. The
        // captured decode step is strictly single-stream, so this tracking is unnecessary. Disable it
        // BEFORE allocating ANY buffer the captured graph will reference (cache, embd, counters,
        // scratch) so none of them carry events. SAFETY: decode-dc touches only gpu.stream.
        let was_tracking = e.ctx().is_event_tracking();
        if was_tracking { unsafe { e.ctx().disable_event_tracking(); } }
        let r = self.generate_graph_inner(e, gs, prompt, max_new, n_embd, head_dim, qt, row_bytes);
        if was_tracking { unsafe { e.ctx().enable_event_tracking(); } }
        r
    }

    fn generate_graph_inner(&self, e: &Engine, gs: &mut GraphDecodeState, prompt: &[u32],
                            max_new: usize, n_embd: usize, head_dim: usize, qt: i32, row_bytes: usize)
                            -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let _ = n_embd;
        let embd_gpu = e.upload_u8(&self.embd.raw)?;
        let max_ctx = prompt.len() + max_new + 8;
        let mut cache = Cache::new(e, &self.cfg, max_ctx)?;

        // (Re)create the persistent counters tracking-OFF so they carry no events (the caller's
        // GraphDecodeState::new may have allocated them with tracking on).
        gs.pos_d = e.htod_i32(&[0])?;
        gs.token_d = e.stream().clone_htod(&[0u32])?;
        // PRIME eagerly: feed each prompt token; advance host + device counters together.
        let mut next_in = 0u32;
        for &tok in prompt {
            e.set_u32_one(&mut gs.token_d, tok)?;
            let nt = self.decode_step_dc(e, &gs.token_d, &mut gs.pos_d, &embd_gpu, qt, row_bytes, &mut cache, /*n_vocab*/ self.output.out_features())?;
            next_in = e.dtoh_u32_one(&nt)?;
        }
        // gs.token_d now must hold the first generated INPUT token (= argmax of the last prime step).
        e.set_u32_one(&mut gs.token_d, next_in)?;

        let n_vocab = self.output.out_features();
        let mut out = Vec::with_capacity(max_new);
        for _ in 0..max_new {
            // t_kv for THIS step = (cache.pos)+1 (the new token's KV length after append).
            let t_kv = cache.pos + 1;
            let key = e.fa_bucket_key(t_kv, head_dim, self.cfg.n_head_kv as usize, crate::Engine::kv_fp8_on());
            if !gs.graphs.contains_key(&key) {
                // bucket_max = t_kv that produces this key's n_splits; t_kv itself works (same key).
                let bucket_max = t_kv;
                // --- snapshot device + host state so the 3 capture-warmup runs leave no residue ---
                let snap = cache.snapshot(e)?;
                let pos_save = e.dtoh_i32_one(&gs.pos_d)?;
                let len_save: Vec<Option<i32>> = cache.kv.iter()
                    .map(|k| k.as_ref().map(|kvl| { e.dtoh_i32_one(&kvl.len_d).unwrap() })).collect();
                let tok_save = e.dtoh_u32_one(&gs.token_d)?;

                // capture (runs the body 3x on the REAL cache + counters).
                let graph = {
                    // split borrows: pull the fields out so the closure can take &mut of each.
                    let GraphDecodeState { token_d, pos_d, .. } = gs;
                    let token_d: &mut CudaSlice<u32> = token_d;
                    let pos_d: &mut CudaSlice<i32> = pos_d;
                    let cache_ref = &mut cache;
                    let embd_ref = &embd_gpu;
                    e.capture_graph(|e| {
                        self.decode_step_dc_cap(e, token_d, pos_d, embd_ref, qt, row_bytes,
                                                cache_ref, n_vocab, bucket_max)
                    })?
                };

                // --- restore the true pre-capture state (undo the 3 throwaway runs) ---
                cache.rollback(e, &snap, 0)?;   // restores conv/ssm + sets len = snapshot len
                e.set_i32_one(&mut gs.pos_d, pos_save)?;
                for (il, ls) in len_save.iter().enumerate() {
                    if let (Some(kvl), Some(v)) = (cache.kv[il].as_mut(), ls) {
                        e.set_i32_one(&mut kvl.len_d, *v)?;
                    }
                }
                e.set_u32_one(&mut gs.token_d, tok_save)?;

                gs.graphs.insert(key, graph);
                gs.bucket_max.insert(key, bucket_max);
                gs.captures += 1;
            }
            // REPLAY: one dispatch runs the whole step; argmax wrote gs.token_d, counters advanced.
            gs.graphs.get(&key).unwrap().launch()?;
            // advance the HOST mirrors to match the in-graph DEVICE advance (host len/pos used only for
            // bucket selection next step; device counters are the source of truth for the kernels).
            cache.pos += 1;
            for kvl in cache.kv.iter_mut().filter_map(|k| k.as_mut()) { kvl.len += 1; }
            // read back the [1] u32 next token (the only D2H in steady state).
            let tok = e.dtoh_u32_one(&gs.token_d)?;
            out.push(tok);
        }
        Ok(out)
    }

    /// Device-counter full-attention decode (CUDA-GRAPH-PLAN Phase 2): clone of `full_attn_decode`
    /// using the `_dc` KV-append (write slot from `kvl.len_d`) + `_dc` fa_decode (t_kv from `kvl.len_d`
    /// after inc), and the resident device rope `pos_d`. Bit-identical to `full_attn_decode` (the
    /// `_dc` kernels reproduce the same math; fa_decode_dc with bucket_max==t_kv reproduces the same
    /// n_splits/per/combine). Advances `kvl.len`/`kvl.len_d`.
    pub(crate) fn full_attn_decode_dc(&self, e: &Engine, fa: &FullAttnLayer, h: &CudaSlice<f32>,
                            pos_d: &CudaSlice<i32>, cache: &mut Cache, il: usize)
                            -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        // eager-mirror path: advance host counters and size n_splits from the live t_kv (bit-identical
        // to fa_decode). The capture path uses full_attn_decode_dc_cap (fixed bucket_max, no host
        // advance, full-buffer K/V view).
        self.full_attn_decode_dc_inner(e, fa, h, None, pos_d, cache, il, None)
    }

    /// PRE-QUANTIZED-INPUT dc full-attn (device-counter path). See full_attn_decode_pre. BIT-IDENTICAL.
    pub(crate) fn full_attn_decode_dc_pre(&self, e: &Engine, fa: &FullAttnLayer, h: &CudaSlice<f32>,
                            hq: &CudaSlice<i8>, hd: &CudaSlice<f32>,
                            pos_d: &CudaSlice<i32>, cache: &mut Cache, il: usize)
                            -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        self.full_attn_decode_dc_inner(e, fa, h, Some((hq, hd)), pos_d, cache, il, None)
    }

    /// PRE-QUANTIZED-INPUT CAPTURE dc full-attn (graph path, fixed bucket_max). BIT-IDENTICAL.
    pub(crate) fn full_attn_decode_dc_cap_pre(&self, e: &Engine, fa: &FullAttnLayer, h: &CudaSlice<f32>,
                            hq: &CudaSlice<i8>, hd: &CudaSlice<f32>,
                            pos_d: &CudaSlice<i32>, cache: &mut Cache, il: usize, bucket_max: usize)
                            -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        self.full_attn_decode_dc_inner(e, fa, h, Some((hq, hd)), pos_d, cache, il, Some(bucket_max))
    }

    /// CAPTURE variant of `full_attn_decode_dc` (CUDA-GRAPH-PLAN Phase 3). `bucket_max` sizes the
    /// fa_decode_dc grid (n_splits) at capture time; the kernel reads the ACTUAL t_kv from the device
    /// counter `kvl.len_d`. Does NOT advance the host `kvl.len` (only the DEVICE counter via inc_seqlen,
    /// which is captured and replays each launch). Views the FULL K/V cache buffer so the kernel may
    /// safely read up to any t_kv within the bucket on replay. Bit-identical to eager when
    /// `bucket_max` yields the same n_splits as eager for the replayed t_kv (the bucket-key contract).
    pub(crate) fn full_attn_decode_dc_cap(&self, e: &Engine, fa: &FullAttnLayer, h: &CudaSlice<f32>,
                            pos_d: &CudaSlice<i32>, cache: &mut Cache, il: usize, bucket_max: usize)
                            -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        self.full_attn_decode_dc_inner(e, fa, h, None, pos_d, cache, il, Some(bucket_max))
    }

    fn full_attn_decode_dc_inner(&self, e: &Engine, fa: &FullAttnLayer, h: &CudaSlice<f32>,
                            pre_q: Option<(&CudaSlice<i8>, &CudaSlice<f32>)>,
                            pos_d: &CudaSlice<i32>, cache: &mut Cache, il: usize,
                            cap_bucket_max: Option<usize>)
                            -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let n_head = cfg.n_head as usize;
        let n_head_kv = cfg.n_head_kv as usize;
        let head_dim = cfg.head_dim_k as usize;
        let eps = cfg.rms_eps;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let n_embd = cfg.n_embd as usize;
        // Q8 TRUNK-FUSION (2026-07-05): wq+wk+wv share input h — on the 35B every full-attn
        // projection is Q8_0, so ONE fused3 launch (block-offset split, out_f 8192/512/512)
        // replaces three launch-latency-class m=1 launches. BIT-IDENTICAL per (tensor,row) to
        // the three matmul_pre MMVQ dispatches (same kernel body). BW24_Q8_DUAL=0 rollback.
        let qkv_fused = |e: &Engine, hq: &CudaSlice<i8>, hd: &CudaSlice<f32>|
            -> Result<(CudaSlice<f32>, CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
            if let Some((qf, k, v)) = e.matmul_q8_fused3(&fa.wq, &fa.wk, &fa.wv, hq, hd)? {
                return Ok((qf, k, v));
            }
            Ok((e.matmul_pre(&fa.wq, hq, hd, h, 1)?, e.matmul_pre(&fa.wk, hq, hd, h, 1)?,
                e.matmul_pre(&fa.wv, hq, hd, h, 1)?))
        };
        let (qf, mut k, v) = if e.uses_q8_1_fast(&fa.wq) && e.uses_q8_1_fast(&fa.wk) && e.uses_q8_1_fast(&fa.wv) {
            match pre_q {
                Some((hq, hd)) => qkv_fused(e, hq, hd)?,
                None => { let (hq, hd) = e.quantize_q8_1(h, 1, n_embd)?;
                    qkv_fused(e, &hq, &hd)? }
            }
        } else {
            (e.matmul(&fa.wq, h, 1)?, e.matmul(&fa.wk, h, 1)?, e.matmul(&fa.wv, h, 1)?)
        };
        // M3/Hy3 have no attention output gate — wq out is exactly q; skip the split.
        let gated = self.cfg.attn_out_gate();
        let (mut q, gate) = if gated {
            let mut q = e.uninit(n_head * head_dim)?;
            let mut gate = e.uninit(n_head * head_dim)?;
            e.q_gate_split(&qf, &mut q, &mut gate, head_dim, n_head, 1)?;
            (q, Some(gate))
        } else { (qf, None) };

        let mut qn = e.uninit(n_head * head_dim)?;
        e.rms_norm(&q, fa.q_norm.float_data(), &mut qn, head_dim, n_head, eps)?;
        q = qn;
        let mut kn = e.uninit(n_head_kv * head_dim)?;
        e.rms_norm(&k, fa.k_norm.float_data(), &mut kn, head_dim, n_head_kv, eps)?;
        k = kn;
        let rope_dims = cfg.rope_dim_count as usize;
        // rope pos from the resident device counter (no per-step host upload).
        e.rope_neox(&mut q, pos_d, head_dim, rope_dims, n_head, 1, cfg.rope_freq_base, 1.0)?;
        e.rope_neox(&mut k, pos_d, head_dim, rope_dims, n_head_kv, 1, cfg.rope_freq_base, 1.0)?;

        let kvl = cache.kv[il].as_mut().unwrap();
        // (1) append at the device write slot kvl.len_d (== old len).
        e.append_kv_quantized_dc(&k, &v, &mut kvl.k, &mut kvl.v, &kvl.len_d,
                                 kvl.kv_dim_k, kvl.kv_dim_v, kvl.k_tok_bytes, kvl.v_tok_bytes,
                                 crate::Engine::kv_fp8_on())?;
        // (2) advance the device counter: kvl.len_d now holds new len == t_kv.
        e.inc_seqlen(&mut kvl.len_d)?;
        // n_splits sizing + K/V view extent:
        //  - eager path (cap_bucket_max==None): advance host len; size from live t_kv == bit-identical
        //    to fa_decode; view exactly t_kv*tok_bytes.
        //  - capture path (Some(bucket_max)): DO NOT touch host len (replay advances only the device
        //    counter); size n_splits from bucket_max; view the FULL cache buffer so any in-bucket t_kv
        //    is in range on replay.
        let (bucket_max, k_view, v_view) = match cap_bucket_max {
            None => {
                kvl.len += 1;
                let t_kv = kvl.len;
                (t_kv,
                 e.view_u8(&kvl.k, t_kv * kvl.k_tok_bytes),
                 e.view_u8(&kvl.v, t_kv * kvl.v_tok_bytes))
            }
            Some(bm) => {
                (bm,
                 e.view_u8(&kvl.k, kvl.k.len()),
                 e.view_u8(&kvl.v, kvl.v.len()))
            }
        };
        let (ktb, vtb) = (kvl.k_tok_bytes, kvl.v_tok_bytes);
        let mut attn = e.uninit(n_head * head_dim)?;
        if std::env::var("BW24_NOFA").is_ok() {
            return Err("BW24_NOFA (naive f32 SDPA) is incompatible with the quantized KV cache; \
                        unset BW24_NOFA to use fa_decode_dc".into());
        }
        // (3) fa_decode reads t_kv from kvl.len_d; bucket_max yields the eager n_splits -> bit-identical.
        e.fa_decode_dc(&q, &k_view, &v_view, &mut attn, head_dim, n_head, n_head_kv,
                       &kvl.len_d, bucket_max, scale, ktb, vtb, crate::Engine::kv_fp8_on())?;

        let attn_g = match &gate {
            Some(gate) => {
                let mut gsig = e.uninit(n_head * head_dim)?;
                e.sigmoid(gate, &mut gsig, n_head * head_dim)?;
                let mut ag = e.uninit(n_head * head_dim)?;
                e.mul(&attn, &gsig, &mut ag, n_head * head_dim)?;
                ag
            }
            None => attn,
        };
        Ok(e.matmul(&fa.wo, &attn_g, 1)?)
    }

    /// Greedy generation: prime with prompt tokens (decode them in sequence to build state),
    /// then generate `max_new` tokens. Returns the generated token ids. (Back-compat: greedy,
    /// no EOS/stop — used by the decode==prefill validation gate. New code uses `generate_with`.)
    pub fn generate(&self, e: &Engine, prompt: &[u32], max_new: usize)
                    -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let max_ctx = prompt.len() + max_new + 8;
        let mut cache = Cache::new(e, &self.cfg, max_ctx)?;
        let mut last_logits = Vec::new();
        // prime: BATCHED cache prime (prime_cache — the prefill-throughput path, the measured #1
        // e2e gap: tokenwise primed at ~102/38 tok/s vs ~2000-5900 tok/s batched). Prompts below
        // PRIME_MIN_T, and BW24_PRIME_TOKENWISE=1 (the escape seam), take the tokenwise loop.
        let t_prime = std::time::Instant::now();
        let batched_prime = prompt.len() >= crate::hybrid_forward::PRIME_MIN_T
            && std::env::var("BW24_PRIME_TOKENWISE").is_err();
        if batched_prime {
            let (l, _h_seed, _hiddens) = self.prime_cache(e, prompt, &mut cache)?;
            last_logits = l;
        } else {
            for &tok in prompt {
                last_logits = self.decode_step(e, tok, &mut cache)?;
            }
        }
        e.stream().synchronize()?;
        // Harness timing contract: prime wall time published for gen-only throughput math
        // (bench binaries read this right after the call; subtraction-from-total breaks down
        // when prime >> gen — measured ±80% error at 6k-token prompts).
        crate::PRIME_NANOS.store(t_prime.elapsed().as_nanos() as u64,
                                 std::sync::atomic::Ordering::Relaxed);
        let mut out = Vec::with_capacity(max_new);
        // E4B: dc/graph serving arms UNWIRED (HANDOVER-E4B.md) — the eager loop below routes
        // through gemma4_e4b_decode_step_h.
        if self.cfg.gemma4.is_some() && !self.is_gemma4_e4b() {
            // Graph serving probed FLAT vs this dc loop (2026-07-12, 1.7k N=2: 174.6/174.2 vs
            // 174.5/174.3) — the GRAPH-GATE's +2.5% is over the plain-eager loop, and the dc
            // arc already banked that; the gate (IDENTICAL at every ctx since the wkv
            // capture-arm fix) stays as the correctness harness.
            // DEVICE-COUNTER greedy loop (the dc arc): stream-identical to eager (DC-GATE).
            let n_vocab = self.output.out_features();
            let embd_gpu = self.embd_gpu.get_or_init(|| {
                e.upload_u8(&self.embd.raw).expect("embed table upload")
            });
            let (qt, rb) = self.embd.qt_and_row_bytes(self.cfg.n_embd as usize);
            for kvl in cache.kv.iter_mut().flatten() {
                e.set_i32_one(&mut kvl.len_d, kvl.len as i32)?;
            }
            let mut token_d = e.stream().clone_htod(&[argmax(&last_logits) as u32])?;
            let mut pos_d = e.htod_i32(&[cache.pos as i32])?;
            for _ in 0..max_new {
                out.push(e.dtoh_u32(&token_d)?[0]);
                token_d = self.gemma4_decode_step_dc(e, &token_d, &mut pos_d, embd_gpu, qt, rb,
                                                     &mut cache, n_vocab, None)?;
            }
            return Ok(out);
        }
        for _ in 0..max_new {
            let next = argmax(&last_logits) as u32;
            out.push(next);
            last_logits = self.decode_step(e, next, &mut cache)?;
        }
        Ok(out)
    }

    /// The reusable serving generation API (BASE-3). Primes the prompt, then samples up to
    /// `params.max_new` tokens, stopping on EOS, any stop-token, or the context-length guard.
    /// Calls `on_token(id)` after each emitted token (for streaming; return `false` to stop early).
    /// Returns `GenOutput { tokens, stop_reason }`. Does NOT detokenize — the caller (which owns
    /// the tokenizer) handles text + stop-STRING matching on the detokenized tail.
    pub fn generate_with<F: FnMut(u32) -> bool>(
        &self, e: &Engine, prompt: &[u32], params: &GenParams,
        sampler: &mut crate::sampler::Sampler, mut on_token: F,
    ) -> Result<GenOutput, Box<dyn std::error::Error>> {
        // Context guard: prompt + generated must fit max_ctx (caller-supplied or model default).
        let ctx_cap = params.max_ctx.unwrap_or(prompt.len() + params.max_new + 8);
        if prompt.len() >= ctx_cap {
            return Ok(GenOutput { tokens: Vec::new(), stop_reason: StopReason::ContextFull });
        }
        let room = ctx_cap - prompt.len();
        let budget = params.max_new.min(room);

        let mut cache = Cache::new(e, &self.cfg, ctx_cap)?;
        let mut last_logits = Vec::new();
        // BATCHED PRIME (2026-07-06 fix — generate_with was still tokenwise! run-gen's "decode"
        // numbers folded a ~40-100 tok/s tokenwise prime into the rate) + PRIME_NANOS contract.
        let t_prime = std::time::Instant::now();
        let batched = prompt.len() >= crate::hybrid_forward::PRIME_MIN_T
            && std::env::var("BW24_PRIME_TOKENWISE").is_err();
        if batched {
            let (l, _h, _x) = self.prime_cache(e, prompt, &mut cache)?;
            last_logits = l;
            for &tok in prompt { sampler.accept(tok); }
        } else {
            for &tok in prompt {
                last_logits = self.decode_step(e, tok, &mut cache)?;
                sampler.accept(tok);
            }
        }
        e.stream().synchronize()?;
        crate::PRIME_NANOS.store(t_prime.elapsed().as_nanos() as u64,
                                 std::sync::atomic::Ordering::Relaxed);
        // BW24_PROFILE_GEN=2: profiler capture starts HERE — after the prime — so an
        // `nsys -c cudaProfilerApi` capture contains ONLY the decode loop (the run-spec
        // BW24_PROFILE_SPEC=2 pattern; =1 in run_gen brackets prime+decode).
        if std::env::var("BW24_PROFILE_GEN").as_deref() == Ok("2") {
            unsafe extern "C" { fn cudaProfilerStart() -> i32; }
            unsafe { cudaProfilerStart(); }
        }
        let mut out = Vec::with_capacity(budget);
        let mut reason = StopReason::MaxNew;
        // gemma4 DEVICE-COUNTER greedy serving loop (the dc arc): token/pos/kv-lens live in
        // device counters, argmax on device — host sees 4B/token. Stream-identical to the
        // eager chain (DC-GATE). Penalties/temp fall through to the host-logits loop.
        if self.cfg.gemma4.is_some() && !self.is_gemma4_e4b()
            && sampler.is_greedy() && sampler.penalty_last_n() == 0 {
            let n_vocab = self.output.out_features();
            let embd_gpu = self.embd_gpu.get_or_init(|| {
                e.upload_u8(&self.embd.raw).expect("embed table upload")
            });
            let (qt, rb) = self.embd.qt_and_row_bytes(self.cfg.n_embd as usize);
            for kvl in cache.kv.iter_mut().flatten() {
                e.set_i32_one(&mut kvl.len_d, kvl.len as i32)?;
            }
            let first = crate::forward::argmax(&last_logits) as u32;
            let mut token_d = e.stream().clone_htod(&[first])?;
            let mut pos_d = e.htod_i32(&[cache.pos as i32])?;
            let mut next = first;
            for _ in 0..budget {
                sampler.accept(next);
                out.push(next);
                if params.eos.contains(&next) { reason = StopReason::Eos; break; }
                if !on_token(next) { reason = StopReason::Callback; break; }
                if cache.pos >= ctx_cap { reason = StopReason::ContextFull; break; }
                token_d = self.gemma4_decode_step_dc(e, &token_d, &mut pos_d, embd_gpu, qt, rb,
                                                     &mut cache, n_vocab, None)?;
                next = e.dtoh_u32(&token_d)?[0];
            }
            return Ok(GenOutput { tokens: out, stop_reason: reason });
        }
        for _ in 0..budget {
            let next = sampler.sample(&last_logits);
            sampler.accept(next);
            out.push(next);
            if params.eos.contains(&next) { reason = StopReason::Eos; break; }
            if !on_token(next) { reason = StopReason::Callback; break; }
            if cache.pos >= ctx_cap { reason = StopReason::ContextFull; break; }
            last_logits = self.decode_step(e, next, &mut cache)?;
        }
        Ok(GenOutput { tokens: out, stop_reason: reason })
    }

    /// Full-attention decode: project q/gate/k/v for the new token, QK-norm, RoPE at pos,
    /// append k,v to the layer KV cache, attend over the full [0..=pos] context.
    pub(crate) fn full_attn_decode(&self, e: &Engine, fa: &FullAttnLayer, h: &CudaSlice<f32>,
                        pos_d: &CudaSlice<i32>, pos: usize, cache: &mut Cache, il: usize)
                        -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        self.full_attn_decode_pre(e, fa, h, None, pos_d, pos, cache, il)
    }

    /// PRE-QUANTIZED-INPUT eager full-attn (attn-input NORM-FUSION lever): caller passes the
    /// attn-normed activation already q8_1 `(hq,hd)` (rms_norm_q8_1) -> skips internal quantize_q8_1.
    /// `None` = quantize h here (the spec / non-fused path). BIT-IDENTICAL.
    pub(crate) fn full_attn_decode_pre(&self, e: &Engine, fa: &FullAttnLayer, h: &CudaSlice<f32>,
                        pre_q: Option<(&CudaSlice<i8>, &CudaSlice<f32>)>,
                        pos_d: &CudaSlice<i32>, pos: usize, cache: &mut Cache, il: usize)
                        -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let n_head = cfg.n_head as usize;
        let n_head_kv = cfg.n_head_kv as usize;
        let head_dim = cfg.head_dim_k as usize;
        let eps = cfg.rms_eps;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // LATENCY-HIDING (BW24_KV_PREFETCH=1): warm this layer's KV stream into L2 while the
        // q/k/v projections run ahead of the fa (fa is latency-bound; its lines land warm).
        // Value-free scheduling — no numeric config change.
        static KV_PF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *KV_PF.get_or_init(|| std::env::var("BW24_KV_PREFETCH").as_deref() == Ok("1")) {
            let kvl = cache.kv[il].as_ref().unwrap();
            let t_kv = kvl.len + 1;
            e.prefetch_l2(&kvl.k, t_kv * kvl.k_tok_bytes)?;
            e.prefetch_l2(&kvl.v, t_kv * kvl.v_tok_bytes)?;
        }

        // wq|wk|wv all take the same input `h` (in_f = n_embd) — quantize q8_1 ONCE, feed all three.
        // Q8 TRUNK-FUSION: on Q8_0 trunks (35B) the three fold into ONE fused3 launch (same MMVQ
        // body per (tensor,row) — bit-identical; see full_attn_decode_dc_inner). BW24_Q8_DUAL=0 off.
        let n_embd = cfg.n_embd as usize;
        let qkv_fused = |e: &Engine, hq: &CudaSlice<i8>, hd: &CudaSlice<f32>|
            -> Result<(CudaSlice<f32>, CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
            if let Some((qf, k, v)) = e.matmul_q8_fused3(&fa.wq, &fa.wk, &fa.wv, hq, hd)? {
                return Ok((qf, k, v));
            }
            Ok((e.matmul_pre(&fa.wq, hq, hd, h, 1)?, e.matmul_pre(&fa.wk, hq, hd, h, 1)?,
                e.matmul_pre(&fa.wv, hq, hd, h, 1)?))
        };
        let (qf, mut k, v) = if e.uses_q8_1_fast(&fa.wq) && e.uses_q8_1_fast(&fa.wk) && e.uses_q8_1_fast(&fa.wv) {
            match pre_q {
                Some((hq, hd)) => qkv_fused(e, hq, hd)?,
                None => { let (hq, hd) = e.quantize_q8_1(h, 1, n_embd)?;
                    qkv_fused(e, &hq, &hd)? }
            }
        } else {
            (e.matmul(&fa.wq, h, 1)?, e.matmul(&fa.wk, h, 1)?, e.matmul(&fa.wv, h, 1)?)
        };
        // q|gate fused: [2*head_dim per head]. Split on-device (no dtoh/host-loop/htod).
        // M3/Hy3 have no attention output gate — wq out is exactly q; skip the split.
        let gated = self.cfg.attn_out_gate();
        let (mut q, gate) = if gated {
            let mut q = e.uninit(n_head * head_dim)?;
            let mut gate = e.uninit(n_head * head_dim)?;
            e.q_gate_split(&qf, &mut q, &mut gate, head_dim, n_head, 1)?;
            (q, Some(gate))
        } else { (qf, None) };

        // QK-norm + RoPE at position `pos`
        let mut qn = e.uninit(n_head * head_dim)?;
        e.rms_norm(&q, fa.q_norm.float_data(), &mut qn, head_dim, n_head, eps)?;
        q = qn;
        let mut kn = e.uninit(n_head_kv * head_dim)?;
        e.rms_norm(&k, fa.k_norm.float_data(), &mut kn, head_dim, n_head_kv, eps)?;
        k = kn;
        let rope_dims = cfg.rope_dim_count as usize;
        e.rope_neox(&mut q, pos_d, head_dim, rope_dims, n_head, 1, cfg.rope_freq_base, 1.0)?;
        e.rope_neox(&mut k, pos_d, head_dim, rope_dims, n_head_kv, 1, cfg.rope_freq_base, 1.0)?;

        // append k,v into the RESIDENT GPU QUANTIZED KV cache at the current position (q8_0 K /
        // q5_1 V, on-device append-quantize kernel; no host round-trip). KVQUANT-PLAN §C/E2.
        let kvl = cache.kv[il].as_mut().unwrap();
        e.append_kv_quantized(&k, &v, &mut kvl.k, &mut kvl.v, kvl.len,
                              kvl.kv_dim_k, kvl.kv_dim_v, kvl.k_tok_bytes, kvl.v_tok_bytes,
                              crate::Engine::kv_fp8_on())?;
        kvl.len += 1;
        let t_kv = kvl.len;

        // attend: q[hd,nh,1] over the resident byte K/V (view first t_kv*tok_bytes BYTES).
        let k_view = e.view_u8(&kvl.k, t_kv * kvl.k_tok_bytes);
        let v_view = e.view_u8(&kvl.v, t_kv * kvl.v_tok_bytes);
        let (ktb, vtb) = (kvl.k_tok_bytes, kvl.v_tok_bytes);
        let mut attn = e.uninit(n_head * head_dim)?;
        if std::env::var("BW24_NOFA").is_ok() {
            return Err("BW24_NOFA (naive f32 SDPA) is incompatible with the quantized KV cache; \
                        unset BW24_NOFA to use fa_decode".into());
        }
        e.fa_decode_kvmod(&q, &k_view, &v_view, &mut attn, head_dim, n_head, n_head_kv, t_kv,
                          scale, ktb, vtb, crate::Engine::kv_fp8_on())?;
        let _ = pos;

        // output gate: attn * sigmoid(gate), then o-proj
        let attn_g = match &gate {
            Some(gate) => {
                let mut gsig = e.uninit(n_head * head_dim)?;
                e.sigmoid(gate, &mut gsig, n_head * head_dim)?;
                let mut ag = e.uninit(n_head * head_dim)?;
                e.mul(&attn, &gsig, &mut ag, n_head * head_dim)?;
                ag
            }
            None => attn,
        };
        Ok(e.matmul(&fa.wo, &attn_g, 1)?)
    }

    /// Linear-attention decode: conv with ring-buffer state, GDN scan carrying SSM state.
    pub fn linear_attn_decode(&self, e: &Engine, la: &LinearAttnLayer, h: &CudaSlice<f32>,
                          cache: &mut Cache, il: usize)
                          -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        self.linear_attn_decode_inner(e, la, h, None, cache, il, false)
    }

    /// PRE-QUANTIZED-INPUT variant (DECODE attn-input NORM-FUSION lever): the caller passes the
    /// post-attn-norm activation ALREADY q8_1-quantized `(hq,hd)` (produced by rms_norm_q8_1, fusing
    /// the attn_norm + the mixer's internal quantize_q8_1). Skips the internal quantize. Caller
    /// GUARANTEES the projections are q8_1-fast. `persistent` selects the capture-safe state plumbing.
    /// BIT-IDENTICAL to linear_attn_decode(h) when (hq,hd)==quantize_q8_1(rms_norm(x)*w).
    pub fn linear_attn_decode_pre(&self, e: &Engine, la: &LinearAttnLayer, h: &CudaSlice<f32>,
                          hq: &CudaSlice<i8>, hd: &CudaSlice<f32>, cache: &mut Cache, il: usize,
                          persistent: bool)
                          -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        self.linear_attn_decode_inner(e, la, h, Some((hq, hd)), cache, il, persistent)
    }

    /// CAPTURE variant of `linear_attn_decode` (CUDA-GRAPH-PLAN Phase 3). The GDN scan needs distinct
    /// in/out SSM-state buffers; the eager path SWAPS a fresh scratch into `rl.ssm_state` (new pointer
    /// each step), which is a CAPTURE HAZARD — the graph bakes capture-time pointers and never re-runs
    /// the host swap, so replay would read a stale state buffer. Here we instead COPY the scratch back
    /// into the STABLE `rl.ssm_state` buffer (memcpy_dtod, captured, same pointers every replay). Math
    /// is identical; only the buffer plumbing differs. `conv_state` is already mutated in place (no
    /// pointer change) so it is capture-safe as-is.
    pub(crate) fn linear_attn_decode_cap(&self, e: &Engine, la: &LinearAttnLayer, h: &CudaSlice<f32>,
                          cache: &mut Cache, il: usize)
                          -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        self.linear_attn_decode_inner(e, la, h, None, cache, il, true)
    }

    fn linear_attn_decode_inner(&self, e: &Engine, la: &LinearAttnLayer, h: &CudaSlice<f32>,
                          pre_q: Option<(&CudaSlice<i8>, &CudaSlice<f32>)>,
                          cache: &mut Cache, il: usize, persistent_state: bool)
                          -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let ssm = cfg.ssm.as_ref().unwrap();
        let d_state = ssm.state_size as usize;
        let num_k = ssm.group_count as usize;
        let num_v = ssm.time_step_rank as usize;
        let d_conv = ssm.conv_kernel as usize;
        let head_k = d_state;
        let key_dim = head_k * num_k;
        let value_dim = d_state * num_v;
        let conv_dim = key_dim * 2 + value_dim;
        let eps = cfg.rms_eps;
        let scale = 1.0 / (d_state as f32).sqrt();

        // projections (T=1): wqkv, wqkv_gate, ssm_beta, ssm_alpha ALL take input `h` (in_f = n_embd)
        // -> quantize q8_1 ONCE, feed all four (was 4x redundant quantize_q8_1 of the same row).
        let n_embd = cfg.n_embd as usize;
        let all_fast = e.uses_q8_1_fast(&la.wqkv) && e.uses_q8_1_fast(&la.wqkv_gate)
            && e.uses_q8_1_fast(&la.ssm_beta) && e.uses_q8_1_fast(&la.ssm_alpha);
        // beta+alpha DUAL fuse (2026-07-05): ssm_beta and ssm_alpha are the same tiny shape
        // ([n_embd -> num_v=32]) — out_f=32 launches are pure launch latency (15-16us each,
        // HANDOVER b4-headroom note). The existing dual mr2 kernel (FFN gate+up) folds them into
        // ONE launch. Bit-identical per row: same MMVQ warp-per-row body, blockIdx.y picks the
        // weight; the separable macro-scale multiply is the same single f32 mul as matmul_pre's
        // in-kernel scale. Falls back to two matmul_pre when ineligible (Float layers 1/2/4 etc).
        let beta_alpha = |e: &Engine, hq: &CudaSlice<i8>, hd: &CudaSlice<f32>|
            -> Result<(CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
            if let Some(((mut b, bs), (mut a, as_))) =
                e.matmul_pre_dual_noscale(&la.ssm_beta, &la.ssm_alpha, hq, hd, 1)? {
                if bs != 1.0 { e.scale_inplace(&mut b, bs, la.ssm_beta.out_features())?; }
                if as_ != 1.0 { e.scale_inplace(&mut a, as_, la.ssm_alpha.out_features())?; }
                return Ok((b, a));
            }
            // Q8_0 twin of the NVFP4 dual (9B GGUFs store ssm_beta/alpha as Q8_0 on most layers):
            // one fused2 launch, bit-identical per row, no macro-scale (q8_0 scale==1.0).
            if let Some((b, a)) = e.matmul_q8_fused2(&la.ssm_beta, &la.ssm_alpha, hq, hd)? {
                return Ok((b, a));
            }
            Ok((e.matmul_pre(&la.ssm_beta, hq, hd, h, 1)?,
                e.matmul_pre(&la.ssm_alpha, hq, hd, h, 1)?))
        };
        // Q8 TRUNK-FUSION (2026-07-05): wqkv+wqkv_gate share (hq,hd) and in_f — on the 35B both
        // are Q8_0 (out_f 8192/4096), so ONE fused2 launch replaces the two biggest
        // launch-latency-class m=1 launches of every linear layer. BIT-IDENTICAL per (tensor,row)
        // (same MMVQ body, block-offset split). Falls back per-tensor when ineligible.
        let qkv_pair = |e: &Engine, hq: &CudaSlice<i8>, hd: &CudaSlice<f32>|
            -> Result<(CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
            if let Some((qkv, z)) = e.matmul_q8_fused2(&la.wqkv, &la.wqkv_gate, hq, hd)? {
                return Ok((qkv, z));
            }
            Ok((e.matmul_pre(&la.wqkv, hq, hd, h, 1)?,
                e.matmul_pre(&la.wqkv_gate, hq, hd, h, 1)?))
        };
        let (qkv_mixed, z, beta_raw, alpha) = if all_fast {
            // attn-input NORM-FUSION: use the caller's pre-quantized (hq,hd) when provided (the
            // attn_norm already emitted q8_1 via rms_norm_q8_1), else quantize h here. Bit-identical.
            match pre_q {
                Some((hq, hd)) => { let (b, a) = beta_alpha(e, hq, hd)?;
                    let (qkv, z) = qkv_pair(e, hq, hd)?;
                    (qkv, z, b, a) }
                None => { let (hq, hd) = e.quantize_q8_1(h, 1, n_embd)?;
                    let (b, a) = beta_alpha(e, &hq, &hd)?;
                    let (qkv, z) = qkv_pair(e, &hq, &hd)?;
                    (qkv, z, b, a) }
            }
        } else {
            // 35B trunk lands HERE: wqkv/wqkv_gate are Q8_0 but ssm_beta/alpha are F32, so
            // all_fast is false. Still fuse the two Q8_0 projections (one quantize + ONE launch
            // instead of two matmuls each re-quantizing h) — matmul_q8_fused2_x is bit-identical
            // to the two m=1 MMVQ dispatches. beta/alpha keep the Float cuBLAS path.
            let (qm, zg) = match e.matmul_q8_fused2_x(&la.wqkv, &la.wqkv_gate, h)? {
                Some(pair) => pair,
                None => (e.matmul(&la.wqkv, h, 1)?, e.matmul(&la.wqkv_gate, h, 1)?),
            };
            (qm, zg, e.matmul(&la.ssm_beta, h, 1)?, e.matmul(&la.ssm_alpha, h, 1)?)
        };

        // RANK3 LEVER (conv fuse): assemble [conv_state | new col], depthwise causal conv + SiLU, and
        // roll the ring — ALL in ONE kernel (`ssm_conv1d_fused_decode`), never materializing conv_in
        // to HBM. Replaces conv_assemble_and_roll + ssm_conv1d. Bit-identical (same accumulation order).
        let rl = cache.recur[il].as_mut().unwrap();
        let mut conv_out = e.uninit(conv_dim)?;  // [conv_dim, 1] channel-major, SiLU
        e.ssm_conv1d_fused_decode(&qkv_mixed, &mut rl.conv_state, la.ssm_conv1d.float_data(),
                                  &mut conv_out, conv_dim, d_conv)?;

        // GDN scan: SSM state stays RESIDENT on GPU. gdn needs DISTINCT in/out state buffers.
        // DECODE DETERMINISM FIX: write the new state into the PERSISTENT spare buffer
        // (`ssm_state_alt`) and PING-PONG the two owned buffers in place — instead of allocating a
        // fresh `state_scratch` via `e.uninit` each step and swapping its pointer in. The old
        // per-step alloc/free churned the stream-ordered async pool; the freed prior state block was
        // recycled by a later step's scratch while a kernel still referenced the swapped-in state,
        // a use-after-reuse that made decode RUN-TO-RUN nondeterministic (two identical primes
        // diverged). With two stable resident buffers there is no per-step alloc/free and no pool
        // churn; the math is byte-identical. `o` is a true per-step output (consumed immediately by
        // gated_rmsnorm below) so it stays a normal scratch.
        let mut o = e.uninit(d_state * num_v)?;
        let n_state = d_state * d_state * num_v;
        let _ = head_k;  // head_k == d_state; the kernels use head_k = d_state internally.
        // GDN PREP, FUSED (2026-07-03): repack + q/k L2-norm + beta sigmoid + g_log in ONE
        // gdn_prep_decode launch (was 5 tiny serialized kernels: qkv_to_gdn_repack, 2x l2_norm,
        // sigmoid, gdn_glog). Same math; the L2 reduce runs a 32-lane warp tree instead of the
        // 256-thread two-level tree (different FP sum order) — gates: argmax + run-spec exactness.
        // (A prep+scan single-launch fusion — lane/gdnfuse, BW24_GDN_FUSE — measured NEUTRAL on
        // eager decode 2026-07-08 and was removed in the flag audit; rig5090.jsonl holds the record.)
        {
            let mut q_l2 = e.uninit(d_state * num_v)?;
            let mut k_l2 = e.uninit(d_state * num_v)?;
            let mut v_gd = e.uninit(d_state * num_v)?;
            let mut beta = e.uninit(num_v)?;
            let mut g_log = e.uninit(num_v)?;
            e.gdn_prep_decode(&conv_out, &beta_raw, &alpha,
                              la.ssm_dt.float_data(), la.ssm_a.float_data(),
                              &mut q_l2, &mut k_l2, &mut v_gd, &mut beta, &mut g_log,
                              d_state, num_v, num_k, key_dim, eps)?;
            // gdn reads ssm_state, writes the spare ssm_state_alt (disjoint resident fields).
            let RecurLayer { ssm_state, ssm_state_alt, .. } = rl;
            e.gdn_scan_s128(&q_l2, &k_l2, &v_gd, &g_log, &beta, ssm_state, ssm_state_alt, &mut o, num_v, 1, scale)?;
        }
        if persistent_state {
            // CAPTURE-safe (graph replay): the canonical state every replay reads must stay at a
            // FIXED pointer (baked into the captured graph). Copy the freshly-written spare BACK
            // into ssm_state (captured, replays each launch). No host pointer swap.
            let alt = std::mem::replace(&mut rl.ssm_state_alt, e.zeros(0)?);
            e.copy_into(&mut rl.ssm_state, 0, &alt, n_state)?;
            rl.ssm_state_alt = alt;
        } else {
            // EAGER: swap the two OWNED resident buffers in place (stable pointers, no alloc/free).
            std::mem::swap(&mut rl.ssm_state, &mut rl.ssm_state_alt);
        }

        // gated RMSNorm + ssm_out. FUSED-QUANTIZE ARM (launch-arc): when ssm_out rides the
        // q8_1 fast path, emit q8_1 straight from the gated norm (bit-identical bytes to
        // gated_rmsnorm + quantize_q8_1) and feed matmul_pre — one launch instead of three
        // (norm, quantize, scale all fold away). Fallback = the original f32 chain.
        if e.uses_q8_1_fast(&la.ssm_out) {
            // norm is PER d_state-ROW (num_v rows), exactly like the f32 twin's grid; the q8_1
            // block stream is row-major so the flat bytes feed the matvec unchanged.
            let (gq, gd) = e.gated_rmsnorm_q8_1(&o, la.ssm_norm.float_data(), &z, d_state, num_v, eps)?;
            let g0 = e.zeros(0)?;
            return Ok(e.matmul_pre(&la.ssm_out, &gq, &gd, &g0, 1)?);
        }
        let mut gn = e.uninit(d_state * num_v)?;
        e.gated_rmsnorm(&o, la.ssm_norm.float_data(), &z, &mut gn, d_state, num_v, eps)?;
        Ok(e.matmul(&la.ssm_out, &gn, 1)?)
    }
}
