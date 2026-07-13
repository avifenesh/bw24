//! DFlash block-diffusion drafter (DFLASH-BRINGUP-PLAN.md, 2026-07-13).
//!
//! 5-layer qwen3-class mini-transformer that drafts a 16-token block in ONE non-causal
//! forward, conditioned on the TARGET's hidden states at 6 tapped layers (concatenated
//! through `fc` + `hidden_norm`). No embed / lm_head of its own — the round reuses the
//! target's. Reference: z-lab/dflash `dflash/model.py` (semantics frozen in the plan doc);
//! oracle: tools/dflash_oracle.py -> /data/cache/dflash-oracle.npz.
//!
//! FIRST LIGHT = f32-resident weights + fresh full-context forward (no draft KV cache) —
//! correctness vs the oracle, then the cache/quant/window arms land measurement-gated.

use crate::Engine;
use crate::model::GpuTensor;
use cudarc::driver::CudaSlice;

pub struct DflashCfg {
    pub hidden: usize,        // 5376
    pub n_head: usize,        // 64
    pub n_kv: usize,          // 8
    pub head_dim: usize,      // 128
    pub n_ff: usize,          // 10752
    pub n_layer: usize,       // 5
    pub eps: f32,             // 1e-6
    pub rope_theta: f32,      // 1e6
    pub block_size: usize,    // 16
    pub mask_token_id: u32,   // 4
    pub target_layer_ids: Vec<usize>, // [1,12,23,35,46,57]
    pub sliding_window: usize,        // 2048
    /// true = sliding_attention for that layer (4x true + 1x false on the 31B draft).
    pub layer_sliding: Vec<bool>,
}

pub struct DflashLayer {
    pub wq: GpuTensor,   // [nh*hd, hidden] row-major (out_f rows)
    pub wk: GpuTensor,   // [nkv*hd, hidden]
    pub wv: GpuTensor,   // [nkv*hd, hidden]
    pub wo: GpuTensor,   // [hidden, nh*hd]
    pub w_gate: GpuTensor, // [n_ff, hidden]
    pub w_up: GpuTensor,   // [n_ff, hidden]
    pub w_down: GpuTensor, // [hidden, n_ff]
    pub ln_in: CudaSlice<f32>,   // [hidden]
    pub ln_post: CudaSlice<f32>, // [hidden]
    pub q_norm: CudaSlice<f32>,  // [hd]
    pub k_norm: CudaSlice<f32>,  // [hd]
}

pub struct DflashDraft {
    pub cfg: DflashCfg,
    pub layers: Vec<DflashLayer>,
    pub fc: GpuTensor,               // [hidden, n_taps*hidden]
    pub hidden_norm: CudaSlice<f32>, // [hidden]
    pub norm: CudaSlice<f32>,        // [hidden]
}

fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

/// Host q8_0 encode (ggml block layout: [d f16][32 x i8] = 34B/32 vals). The drafter's
/// weights ride the dp4a fast path at 1.6GB resident (bf16 3.1GB + the 31B trunk OOM'd
/// 24GB; f32 6.2GB worse). Drafter quantization moves ACCEPTANCE only — verify exactness
/// is structural.
fn encode_q8_0(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() / 32 * 34);
    for blk in vals.chunks_exact(32) {
        let amax = blk.iter().fold(0f32, |a, v| a.max(v.abs()));
        let d = amax / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        let dh = half_from_f32(d);
        out.extend_from_slice(&dh.to_le_bytes());
        for &v in blk {
            out.push(((v * id).round().clamp(-127.0, 127.0)) as i8 as u8);
        }
    }
    out
}

/// Host q4_0 encode (ggml: [d f16][16B packed nibbles] = 18B/32 vals; q = round(v/d)+8,
/// d = amax/-7 sign trick NOT used — plain amax/7? ggml uses d = max/-8 .. follow ggml:
/// d = amax / -8 when the max is negative-dominant; reference quantize_row_q4_0: d =
/// max(|v|)/-8 signed-max form). Implemented to match ggml quantize_row_q4_0_ref.
fn encode_q4_0(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() / 32 * 18);
    for blk in vals.chunks_exact(32) {
        // ggml ref: pick the value with the LARGEST |v| (keeping sign), d = that / -8
        let mut amax = 0f32; let mut mx = 0f32;
        for &v in blk { if v.abs() > amax { amax = v.abs(); mx = v; } }
        let d = mx / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&half_from_f32(d).to_le_bytes());
        for j in 0..16 {
            let x0 = (blk[j] * id + 8.5).clamp(0.0, 15.0) as u8;
            let x1 = (blk[j + 16] * id + 8.5).clamp(0.0, 15.0) as u8;
            out.push(x0 | (x1 << 4));
        }
    }
    out
}

fn half_from_f32(v: f32) -> u16 {
    // f32 -> IEEE f16 (round-to-nearest-even; range of q8_0 d values is tame)
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let man = b & 0x7fffff;
    if exp <= 0 { return sign; }              // flush tiny d to zero
    if exp >= 31 { return sign | 0x7c00; }    // inf (unreachable for sane d)
    let mut h = sign | ((exp as u16) << 10) | ((man >> 13) as u16);
    // round to nearest even on the truncated 13 bits
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) { h += 1; }
    h
}

impl DflashDraft {
    /// Load the backbone-only checkpoint dir (config.json + model.safetensors, bf16).
    /// Config scalars ride a minimal extractor (no json dep in-tree — HfConfig precedent).
    pub fn load(e: &Engine, dir: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let txt = std::fs::read_to_string(dir.join("config.json"))?;
        fn num(txt: &str, key: &str) -> Option<f64> {
            let i = txt.find(&format!("\"{key}\""))?;
            let rest = &txt[i..];
            let colon = rest.find(':')?;
            let val: String = rest[colon + 1..].trim_start().chars()
                .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-' || *c == 'e' || *c == 'E' || *c == '+')
                .collect();
            val.parse().ok()
        }
        fn num_list(txt: &str, key: &str) -> Vec<usize> {
            let Some(i) = txt.find(&format!("\"{key}\"")) else { return Vec::new() };
            let rest = &txt[i..];
            let (Some(a), Some(b)) = (rest.find('['), rest.find(']')) else { return Vec::new() };
            rest[a + 1..b].split(',').filter_map(|s| s.trim().parse().ok()).collect()
        }
        let g = |k: &str| num(&txt, k).unwrap_or_else(|| panic!("config missing {k}")) as usize;
        // layer_types order: count entries, mark sliding ones
        let layer_sliding: Vec<bool> = {
            let i = txt.find("\"layer_types\"").expect("layer_types");
            let rest = &txt[i..];
            let (a, b) = (rest.find('[').unwrap(), rest.find(']').unwrap());
            rest[a + 1..b].split(',').map(|s| s.contains("sliding_attention")).collect()
        };
        let cfg = DflashCfg {
            hidden: g("hidden_size"),
            n_head: g("num_attention_heads"),
            n_kv: g("num_key_value_heads"),
            head_dim: g("head_dim"),
            n_ff: g("intermediate_size"),
            n_layer: g("num_hidden_layers"),
            eps: num(&txt, "rms_norm_eps").expect("rms_norm_eps") as f32,
            rope_theta: num(&txt, "rope_theta").expect("rope_theta") as f32,
            block_size: g("block_size"),
            mask_token_id: g("mask_token_id") as u32,
            target_layer_ids: num_list(&txt, "target_layer_ids"),
            sliding_window: g("sliding_window"),
            layer_sliding,
        };
        let st = bw24_gguf::safetensors::StModel::open(&dir.join("model.safetensors"))?;
        // 1D norm weights ride raw slices; 2D matmul weights ride GpuTensor::Float
        // (cuBLASLt f32 arm — the Stage-A numeric class, right for oracle parity).
        let up = |name: &str| -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
            let (_info, bytes) = st.raw(name).ok_or_else(|| format!("missing tensor {name}"))?;
            Ok(e.htod(&bf16_to_f32(bytes))?)
        };
        // Precision policy (BW24_DFLASH_PREC seam): "q8" = all q8_0 (1.6GB, default);
        // "mixed" = bf16 attn+fc (the ctx-conditioning path) + q8_0 ffn (~2.2GB — fits the
        // ~2.8GB headroom beside the 31B trunk); "bf16" = all bf16 (parity runs, no target).
        let prec = std::env::var("BW24_DFLASH_PREC").unwrap_or_else(|_| "q8".into());
        let upw = |name: &str| -> Result<GpuTensor, Box<dyn std::error::Error>> {
            let (info, bytes) = st.raw(name).ok_or_else(|| format!("missing tensor {name}"))?;
            let shape = info.ne(); // ggml order: ne[0]=in_f, ne[1]=out_f
            let in_f = shape[0] as usize;
            let is_ffn = name.contains(".mlp.");
            let bf16 = prec == "bf16" || (prec == "mixed" && !is_ffn)
                || (prec == "fc" && name == "fc.weight");
            if bf16 {
                return Ok(GpuTensor::FloatBf16 { data: e.upload_u8(bytes)?, ne: shape.to_vec() });
            }
            let f32s = bf16_to_f32(bytes);
            if prec == "q4" {
                let q = encode_q4_0(&f32s);
                return Ok(GpuTensor::Quant {
                    bytes: e.upload_u8(&q)?, qtype: crate::QT_Q4_0,
                    row_bytes: in_f / 32 * 18, ne: shape.to_vec(), scale: 1.0, rp: false,
                    #[cfg(bw24_cutlass)]
                    cutlass: None,
                    fp8: None, rp4: None,
                });
            }
            let q = encode_q8_0(&f32s);
            Ok(GpuTensor::Quant {
                bytes: e.upload_u8(&q)?, qtype: crate::QT_Q8_0,
                row_bytes: in_f / 32 * 34, ne: shape.to_vec(), scale: 1.0, rp: false,
                #[cfg(bw24_cutlass)]
                cutlass: None,
                fp8: None, rp4: None,
            })
        };
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let p = |s: &str| format!("layers.{i}.{s}");
            layers.push(DflashLayer {
                wq: upw(&p("self_attn.q_proj.weight"))?,
                wk: upw(&p("self_attn.k_proj.weight"))?,
                wv: upw(&p("self_attn.v_proj.weight"))?,
                wo: upw(&p("self_attn.o_proj.weight"))?,
                w_gate: upw(&p("mlp.gate_proj.weight"))?,
                w_up: upw(&p("mlp.up_proj.weight"))?,
                w_down: upw(&p("mlp.down_proj.weight"))?,
                ln_in: up(&p("input_layernorm.weight"))?,
                ln_post: up(&p("post_attention_layernorm.weight"))?,
                q_norm: up(&p("self_attn.q_norm.weight"))?,
                k_norm: up(&p("self_attn.k_norm.weight"))?,
            });
        }
        Ok(Self {
            fc: upw("fc.weight")?,
            hidden_norm: up("hidden_norm.weight")?,
            norm: up("norm.weight")?,
            cfg,
            layers,
        })
    }

    /// f32 GEMM helper via the engine Float arm (cuBLASLt): y[t, out_f].
    fn mm(&self, e: &Engine, w: &GpuTensor, x: &CudaSlice<f32>, t: usize, _in_f: usize,
          _out_f: usize) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        Ok(e.matmul(w, x, t)?)
    }

    /// FIRST-LIGHT forward (oracle contract): full non-causal attention over
    /// [ctx_features ; block], NO draft KV cache, NO sliding window (the oracle bypasses
    /// the reference mask machinery the same way — window/caching land in the round arm).
    ///
    /// `target_hidden`: [ctx, n_taps*hidden] (f32, device)  — raw tapped states.
    /// `noise_emb`:     [block, hidden] — target embed rows for [accepted, MASK x b-1].
    /// `pos`:           absolute positions for ctx rows THEN block rows (ctx+block i32).
    /// Returns final normed hidden [block, hidden] (feed target lm_head for draft logits).
    /// ctx features for `t` tapped rows: hidden_norm(fc(taps)) — the drafter's context
    /// representation, cacheable across rounds (append-only in committed-token order).
    pub fn ctx_features(&self, e: &Engine, taps: &CudaSlice<f32>, t: usize)
                        -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let c = &self.cfg;
        let n_taps = c.target_layer_ids.len();
        let fc_out = self.mm(e, &self.fc, taps, t, n_taps * c.hidden, c.hidden)?;
        let mut out = e.uninit(t * c.hidden)?;
        e.rms_norm(&fc_out, &self.hidden_norm, &mut out, c.hidden, t, c.eps)?;
        Ok(out)
    }

    pub fn forward(
        &self, e: &Engine, target_hidden: &CudaSlice<f32>, noise_emb: &CudaSlice<f32>,
        pos: &[i32], ctx: usize,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let ctx_f = self.ctx_features(e, target_hidden, ctx)?;
        if let Ok(dir) = std::env::var("BW24_DFLASH_DUMP") {
            let v = e.dtoh(&ctx_f)?;
            let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
            std::fs::write(format!("{dir}/bw24-ctx_features.f32"), bytes)?;
        }
        self.forward_block(e, &ctx_f, noise_emb, pos, ctx)
    }

    /// Block forward over PRECOMPUTED ctx features (the round arm's entry: features are
    /// cached across rounds; only the block work repeats).
    pub fn forward_block(
        &self, e: &Engine, ctx_f: &CudaSlice<f32>, noise_emb: &CudaSlice<f32>,
        pos: &[i32], ctx: usize,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let c = &self.cfg;
        let (h, nh, nkv, hd) = (c.hidden, c.n_head, c.n_kv, c.head_dim);
        let b = c.block_size;
        assert_eq!(pos.len(), ctx + b, "pos covers ctx rows then block rows");

        let pos_blk = e.htod_i32(&pos[ctx..])?;

        let mut x = e.clone_dtod(noise_emb)?; // [b, hidden] residual stream
        for (li, l) in self.layers.iter().enumerate() {
            let _ = li;
            // input_layernorm on the block rows only (ctx features are norm-free per ref:
            // k/v project the SAME ctx_f every layer, un-layernormed).
            let mut xn = e.uninit(b * h)?;
            e.rms_norm(&x, &l.ln_in, &mut xn, h, b, c.eps)?;

            // q from block; k/v from [ctx_f ; block-normed]
            let q0 = self.mm(e, &l.wq, &xn, b, h, nh * hd)?;
            let k0c = self.mm(e, &l.wk, ctx_f, ctx, h, nkv * hd)?;
            let v0c = self.mm(e, &l.wv, ctx_f, ctx, h, nkv * hd)?;
            let k0b = self.mm(e, &l.wk, &xn, b, h, nkv * hd)?;
            let v0b = self.mm(e, &l.wv, &xn, b, h, nkv * hd)?;

            // per-head q/k rms norm (v passes through: ones weight trick not needed — the
            // qkv kernel norms rq+rk rows; concatenate k first).
            let mut k0 = e.uninit((ctx + b) * nkv * hd)?;
            e.copy_into(&mut k0, 0, &k0c, ctx * nkv * hd)?;
            e.copy_into(&mut k0, ctx * nkv * hd, &k0b, b * nkv * hd)?;
            let mut v = e.uninit((ctx + b) * nkv * hd)?;
            e.copy_into(&mut v, 0, &v0c, ctx * nkv * hd)?;
            e.copy_into(&mut v, ctx * nkv * hd, &v0b, b * nkv * hd)?;

            if li == 0 { if let Ok(dir) = std::env::var("BW24_DFLASH_DUMP") {
                let v = e.dtoh(&q0)?;
                let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
                std::fs::write(format!("{dir}/bw24-l0_q0.f32"), bytes)?;
            }}
            let mut q = e.uninit(b * nh * hd)?;
            let mut k = e.uninit((ctx + b) * nkv * hd)?;
            // rms over head_dim rows: q has b*nh rows, k has (ctx+b)*nkv rows.
            e.rms_norm(&q0, &l.q_norm, &mut q, hd, b * nh, c.eps)?;
            if li == 0 { if let Ok(dir) = std::env::var("BW24_DFLASH_DUMP") {
                let v = e.dtoh(&q)?;
                let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
                std::fs::write(format!("{dir}/bw24-l0_qn.f32"), bytes)?;
            }}
            e.rms_norm(&k0, &l.k_norm, &mut k, hd, (ctx + b) * nkv, c.eps)?;

            // rope: q at block positions, k at ctx-then-block positions (absolute).
            let norope = std::env::var("BW24_DFLASH_NOROPE").is_ok();
            if !norope { e.rope_neox(&mut q, &pos_blk, hd, hd, nh, b, c.rope_theta, 1.0)?; }
            if li == 0 { if let Ok(dir) = std::env::var("BW24_DFLASH_DUMP") {
                let dump = |name: &str, t: &cudarc::driver::CudaSlice<f32>| -> Result<(), Box<dyn std::error::Error>> {
                    let v = e.dtoh(t)?;
                    let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
                    std::fs::write(format!("{dir}/bw24-l0_{name}.f32"), bytes)?;
                    Ok(())
                };
                dump("xn", &xn)?; dump("q_prerope", &q)?;
            }}
            // k rows are laid out [row, nkv, hd] with row-major tokens — rope_neox expects
            // (n_heads, n_tokens); ctx and block ropes run as one call over ctx+b tokens.
            let pos_all = e.htod_i32(pos)?;
            if !norope { e.rope_neox(&mut k, &pos_all, hd, hd, nkv, ctx + b, c.rope_theta, 1.0)?; }

            // full non-causal attention: every block query sees all ctx+b keys.
            let mut attn = e.uninit(b * nh * hd)?;
            let scale = 1.0f32 / (hd as f32).sqrt();
            // NAIVE SDPA for first light: fa_prefill's NON-CAUSAL arm with T != T_kv is
            // BROKEN (attn maxdiff 0.34 vs the torch oracle; q/k inputs bit-close — no
            // existing caller exercises that shape class, jsonl 2026-07-13). The 16 x
            // (ctx+16) block attention is tiny; the fa arm returns behind this seam once
            // its kernel is fixed + parity-gated.
            if std::env::var("BW24_DFLASH_FA").is_ok() {
                e.fa_prefill(&q, &k, &v, &mut attn, hd, nh, nkv, b, ctx + b, scale, false)?;
            } else {
                e.sdpa_naive(&q, &k, &v, &mut attn, hd, nh, nkv, b, ctx + b, scale, false)?;
            }

            let o = self.mm(e, &l.wo, &attn, b, nh * hd, h)?;
            let mut x1 = e.uninit(b * h)?;
            e.add(&o, &x, &mut x1, b * h)?;
            if li == 0 { if let Ok(dir) = std::env::var("BW24_DFLASH_DUMP") {
                let dump = |name: &str, t: &cudarc::driver::CudaSlice<f32>| -> Result<(), Box<dyn std::error::Error>> {
                    let v = e.dtoh(t)?;
                    let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
                    std::fs::write(format!("{dir}/bw24-l0_{name}.f32"), bytes)?;
                    Ok(())
                };
                dump("q", &q)?; dump("k", &k)?; dump("attn", &attn)?; dump("x1", &x1)?;
            }}

            // mlp
            let mut x1n = e.uninit(b * h)?;
            e.rms_norm(&x1, &l.ln_post, &mut x1n, h, b, c.eps)?;
            let gate = self.mm(e, &l.w_gate, &x1n, b, h, c.n_ff)?;
            let up_ = self.mm(e, &l.w_up, &x1n, b, h, c.n_ff)?;
            let mut act = e.uninit(b * c.n_ff)?;
            e.silu_mul(&gate, &up_, &mut act, b * c.n_ff)?;
            let down = self.mm(e, &l.w_down, &act, b, c.n_ff, h)?;
            let mut x2 = e.uninit(b * h)?;
            e.add(&down, &x1, &mut x2, b * h)?;
            x = x2;
            if let Ok(dir) = std::env::var("BW24_DFLASH_DUMP") {
                let v = e.dtoh(&x)?;
                let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
                std::fs::write(format!("{dir}/bw24-layer{li}_out.f32"), bytes)?;
            }
        }
        let mut out = e.uninit(b * h)?;
        e.rms_norm(&x, &self.norm, &mut out, h, b, c.eps)?;
        Ok(out)
    }
}

/// Draft KV cache (round-cost fix, 2026-07-13): per-layer normed+roped ctx K and raw ctx V,
/// append-only in committed order. Block K/V land TRANSIENTLY at [len..len+b] each round
/// (never committed — the reference crops them identically). Kills the per-round full-ctx
/// projection recompute (first light was O(ctx)/round -> 7 tok/s).
pub struct DflashKv {
    pub k: Vec<CudaSlice<f32>>,   // per layer [cap + block, nkv*hd]
    pub v: Vec<CudaSlice<f32>>,
    pub len: usize,
    pub cap: usize,
}

impl DflashKv {
    pub fn new(e: &Engine, cfg: &DflashCfg, cap: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let rowsz = cfg.n_kv * cfg.head_dim;
        let mut k = Vec::with_capacity(cfg.n_layer);
        let mut v = Vec::with_capacity(cfg.n_layer);
        for _ in 0..cfg.n_layer {
            k.push(e.uninit((cap + cfg.block_size) * rowsz)?);
            v.push(e.uninit((cap + cfg.block_size) * rowsz)?);
        }
        Ok(Self { k, v, len: 0, cap })
    }
}

impl DflashDraft {
    /// Ingest `t` NEW ctx-feature rows (committed order, absolute positions `pos_new`) into
    /// the draft KV: per layer k/v projections + k head-norm + rope, appended at kv.len.
    pub fn ingest_ctx(&self, e: &Engine, kv: &mut DflashKv, feats: &CudaSlice<f32>,
                      pos_new: &[i32], t: usize) -> Result<(), Box<dyn std::error::Error>> {
        let c = &self.cfg;
        let (h, nkv, hd) = (c.hidden, c.n_kv, c.head_dim);
        assert!(kv.len + t <= kv.cap, "draft kv overflow");
        let pos_d = e.htod_i32(pos_new)?;
        for (li, l) in self.layers.iter().enumerate() {
            let k0 = self.mm(e, &l.wk, feats, t, h, nkv * hd)?;
            let v0 = self.mm(e, &l.wv, feats, t, h, nkv * hd)?;
            let mut kn = e.uninit(t * nkv * hd)?;
            e.rms_norm(&k0, &l.k_norm, &mut kn, hd, t * nkv, c.eps)?;
            e.rope_neox(&mut kn, &pos_d, hd, hd, nkv, t, c.rope_theta, 1.0)?;
            e.copy_into(&mut kv.k[li], kv.len * nkv * hd, &kn, t * nkv * hd)?;
            e.copy_into(&mut kv.v[li], kv.len * nkv * hd, &v0, t * nkv * hd)?;
        }
        kv.len += t;
        Ok(())
    }

    /// Block forward over the CACHED ctx KV: only the 16 block rows are projected per layer;
    /// block K/V land transiently at kv[len..len+b]. Bit-class-identical to forward_block
    /// (same kernels, same per-row programs; ONLY the ctx K/V recompute is cached).
    pub fn forward_round(&self, e: &Engine, kv: &mut DflashKv, noise_emb: &CudaSlice<f32>,
                         pos_block: &[i32]) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let c = &self.cfg;
        let (h, nh, nkv, hd) = (c.hidden, c.n_head, c.n_kv, c.head_dim);
        let b = c.block_size;
        assert_eq!(pos_block.len(), b);
        let ctx = kv.len;
        let pos_blk = e.htod_i32(pos_block)?;
        let mut x = e.clone_dtod(noise_emb)?;
        for (li, l) in self.layers.iter().enumerate() {
            let mut xn = e.uninit(b * h)?;
            e.rms_norm(&x, &l.ln_in, &mut xn, h, b, c.eps)?;
            let q0 = self.mm(e, &l.wq, &xn, b, h, nh * hd)?;
            let k0b = self.mm(e, &l.wk, &xn, b, h, nkv * hd)?;
            let v0b = self.mm(e, &l.wv, &xn, b, h, nkv * hd)?;
            let mut q = e.uninit(b * nh * hd)?;
            let mut kb = e.uninit(b * nkv * hd)?;
            e.rms_norm(&q0, &l.q_norm, &mut q, hd, b * nh, c.eps)?;
            e.rms_norm(&k0b, &l.k_norm, &mut kb, hd, b * nkv, c.eps)?;
            e.rope_neox(&mut q, &pos_blk, hd, hd, nh, b, c.rope_theta, 1.0)?;
            e.rope_neox(&mut kb, &pos_blk, hd, hd, nkv, b, c.rope_theta, 1.0)?;
            e.copy_into(&mut kv.k[li], ctx * nkv * hd, &kb, b * nkv * hd)?;
            e.copy_into(&mut kv.v[li], ctx * nkv * hd, &v0b, b * nkv * hd)?;
            let mut attn = e.uninit(b * nh * hd)?;
            let scale = 1.0f32 / (hd as f32).sqrt();
            if std::env::var("BW24_DFLASH_FA").is_ok() {
                e.fa_prefill(&q, &kv.k[li], &kv.v[li], &mut attn, hd, nh, nkv, b, ctx + b,
                             scale, false)?;
            } else {
                e.sdpa_naive(&q, &kv.k[li], &kv.v[li], &mut attn, hd, nh, nkv, b, ctx + b,
                             scale, false)?;
            }
            let o = self.mm(e, &l.wo, &attn, b, nh * hd, h)?;
            let mut x1 = e.uninit(b * h)?;
            e.add(&o, &x, &mut x1, b * h)?;
            let mut x1n = e.uninit(b * h)?;
            e.rms_norm(&x1, &l.ln_post, &mut x1n, h, b, c.eps)?;
            let gate = self.mm(e, &l.w_gate, &x1n, b, h, c.n_ff)?;
            let up_ = self.mm(e, &l.w_up, &x1n, b, h, c.n_ff)?;
            let mut act = e.uninit(b * c.n_ff)?;
            e.silu_mul(&gate, &up_, &mut act, b * c.n_ff)?;
            let down = self.mm(e, &l.w_down, &act, b, c.n_ff, h)?;
            let mut x2 = e.uninit(b * h)?;
            e.add(&down, &x1, &mut x2, b * h)?;
            x = x2;
        }
        let mut out = e.uninit(b * h)?;
        e.rms_norm(&x, &self.norm, &mut out, h, b, c.eps)?;
        Ok(out)
    }
}

// ================= DFlash spec round (greedy, first light) =================
// Exact contract: identical output stream to plain greedy decode BY CONSTRUCTION — the
// target's batched verify argmax decides every committed token; the drafter only proposes.
// (Same verify+rewind pattern as generate_spec_gemma's eager round; t=16 verify rides the
// straddle-split-safe fa_decode_rows.)
impl crate::hybrid::HybridModel {
    pub fn generate_spec_dflash(
        &self, e: &Engine, draft: &DflashDraft, prompt: &[u32], max_new: usize, eos: &[u32],
    ) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        use crate::cache::{Cache, DflashTapSink};
        let n_embd = self.cfg.n_embd as usize;
        let c = &draft.cfg;
        assert_eq!(n_embd, c.hidden, "draft hidden must match target n_embd");
        let b = c.block_size;
        let n_taps = c.target_layer_ids.len();
        let max_ctx = prompt.len() + max_new + b + 8;
        // First light holds ctx <= sliding_window: the draft was trained with 4 sliding
        // layers (window 2048) and the first-light attention is windowless full — inside
        // the window the two are identical. The depth cell (1736 + 128) fits.
        assert!(max_ctx <= c.sliding_window,
                "first-light dflash round is windowless — ctx cap {} exceeds the draft window {}",
                max_ctx, c.sliding_window);
        let mut cache = Cache::new(e, &self.cfg, max_ctx)?;

        // ---- prime with taps armed ----
        let tp = prompt.len();
        cache.dflash_taps = Some(DflashTapSink {
            layer_ids: c.target_layer_ids.clone(),
            buf: e.uninit(tp * n_taps * n_embd)?,
            hidden: n_embd, t: tp,
        });
        let t_prime = std::time::Instant::now();
        let (logits, _h_seed, _hiddens) = self.prime_cache(e, prompt, &mut cache)?;
        let mut last = crate::forward::argmax(&logits) as u32;
        // draft KV cache: ingest the prompt's ctx features once; per round only the kept
        // rows ingest + the block projects (round cost O(block), not O(ctx)).
        let mut dkv = DflashKv::new(e, &draft.cfg, max_ctx)?;
        {
            // CHUNKED ingest (depth OOM fix): the 1736-row prompt tap buffer is ~224MB f32;
            // running fc + 5-layer k/v projection over it in one shot stacks another
            // ~300MB of transients on the ~21.3GB trunk peak. 256-row windows bound the
            // transient set; identical values (row-independent ops).
            let taps = cache.dflash_taps.take().unwrap();
            let n_taps_h = n_taps * n_embd;
            let mut r0 = 0usize;
            while r0 < tp {
                let t_c = (tp - r0).min(256);
                let tv = e.view(&taps.buf, tp * n_taps_h);
                let win = tv.slice(r0 * n_taps_h..(r0 + t_c) * n_taps_h);
                let mut chunk = e.uninit(t_c * n_taps_h)?;
                e.copy_view_into(&mut chunk, 0, &win, t_c * n_taps_h)?;
                let f = draft.ctx_features(e, &chunk, t_c)?;
                let pos_c: Vec<i32> = ((r0 as i32)..(r0 + t_c) as i32).collect();
                draft.ingest_ctx(e, &mut dkv, &f, &pos_c, t_c)?;
                r0 += t_c;
            }
        }
        let mut ctx_len = tp;
        e.stream().synchronize()?;
        // published prime wall (the run-spec/gemma-gate timing contract subtracts it)
        crate::PRIME_NANOS.store(t_prime.elapsed().as_nanos() as u64,
                                 std::sync::atomic::Ordering::Relaxed);

        // embed-scale seam (BW24_DFLASH_EMB_SCALE): gemma trunks scale embeddings by
        // sqrt(n_embd) INSIDE the forward; whether the z-lab gemma4 training fed the
        // drafter scaled or raw embed rows is not visible from the reference (qwen path
        // uses raw embed_tokens). Acceptance arbitrates; default raw.
        let emb_scale = if std::env::var("BW24_DFLASH_EMB_SCALE").as_deref() == Ok("1") {
            (n_embd as f32).sqrt()
        } else { 1.0 };

        let mut out = Vec::with_capacity(max_new);
        let n_vocab = self.output.out_features();
        // VERIFY WIDTH (BW24_DFLASH_VERIFY_T, default 8): the drafter always drafts a full
        // block (its trained mask pattern) but only the first vt rows go through the target
        // verify — the t=16 verify rides the untuned b16 tier at ~32% of the byte wall
        // (65ms/verify) while b8 rides the tuned r2 tier; with ~2.7 committed/round the
        // deep block positions almost never survive anyway. Exactness unaffected (verify
        // still decides every committed token).
        let vt: usize = std::env::var("BW24_DFLASH_VERIFY_T").ok()
            .and_then(|v| v.parse().ok()).unwrap_or(8).clamp(2, b);
        let mut attempted = 0usize;
        let mut accepted = 0usize;
        // The whole round runs in the decode-exact matmul scope: the m=16 draft mms were
        // otherwise falling into the prefill-GEMM class (770us/matmul, 17% of the depth
        // round). Prime (before this loop) keeps the prefill GEMM path.
        e.set_verify_exact(true);
        'outer: while out.len() < max_new {
            let start = cache.pos; // committed length
            // ---- draft: block = [last, MASK x b-1] ----
            let mut block: Vec<u32> = vec![c.mask_token_id; b];
            block[0] = last;
            let mut noise = e.htod(&self.embd.gather(n_embd, &block))?;
            if emb_scale != 1.0 { e.scale_inplace(&mut noise, emb_scale, b * n_embd)?; }
            if std::env::var("BW24_DFLASH_DEBUG").as_deref() == Ok("1") && start == cache.pos {
                let nv = e.dtoh(&noise)?;
                let r0: f32 = nv[..n_embd].iter().map(|x| x * x).sum::<f32>().sqrt();
                let r1: f32 = nv[n_embd..2 * n_embd].iter().map(|x| x * x).sum::<f32>().sqrt();
                eprintln!("[dflash noise] |row0(last)|={r0:.3} |row1(MASK id {})|={r1:.3}",
                          c.mask_token_id);
            }
            let pos_block: Vec<i32> = ((start as i32)..(start + b) as i32).collect();
            let dh = draft.forward_round(e, &mut dkv, &noise, &pos_block)?;
            // draft tokens = argmax(lm_head(h rows 1..b))
            let mut rows = e.uninit((b - 1) * n_embd)?;
            {
                let dv = e.view(&dh, b * n_embd);
                let tail = dv.slice(n_embd..b * n_embd);
                e.copy_view_into(&mut rows, 0, &tail, (b - 1) * n_embd)?;
            }
            let dl = e.matmul(&self.output, &rows, b - 1)?;
            let mut dtoks_d = e.stream().alloc_zeros::<u32>(b - 1)?;
            for i in 0..(b - 1) {
                e.argmax_token_device_col(&dl, i, n_vocab, &mut dtoks_d, i)?;
            }
            let dtoks = e.dtoh_u32(&dtoks_d)?;
            for (i, &dt) in dtoks.iter().enumerate() { block[i + 1] = dt; }
            let dbg = std::env::var("BW24_DFLASH_DEBUG").as_deref() == Ok("1");

            // ---- verify: one t=vt target forward with taps armed ----
            let vblock = &block[..vt];
            cache.dflash_taps = Some(DflashTapSink {
                layer_ids: c.target_layer_ids.clone(),
                buf: e.uninit(vt * n_taps * n_embd)?,
                hidden: n_embd, t: vt,
            });
            let (vam, _vh) = self.gemma4_decode_step_t_am(e, vblock, start, &mut cache)?;
            let taps = cache.dflash_taps.take().unwrap();
            if dbg {
                eprintln!("[dflash r] start={start} last={last}\n  draft={:?}\n  vam  ={:?}",
                          &block[1..], &vam);
            }

            // ---- accept ----
            let mut m = 0usize;
            while m < vt - 1 && block[m + 1] as usize == vam[m] as usize { m += 1; }
            attempted += vt - 1;
            accepted += m;
            out.push(last);
            if eos.contains(&last) { break 'outer; }
            for &dt in &block[1..=m] {
                out.push(dt);
                if eos.contains(&dt) { break 'outer; }
                if out.len() >= max_new { break 'outer; }
            }
            let next = vam[m] as u32;

            // ---- commit/rollback: keep m+1 of the b appended rows ----
            let keep = m + 1;
            for kvl in cache.kv.iter_mut().flatten() {
                kvl.len -= vt - keep;
                e.set_i32_one(&mut kvl.len_d, kvl.len as i32)?;
            }
            cache.pos -= vt - keep;

            // ---- ingest the kept rows' ctx features into the draft KV ----
            {
                let tv = e.view(&taps.buf, vt * n_taps * n_embd);
                let keep_view = tv.slice(0..keep * n_taps * n_embd);
                let mut kept = e.uninit(keep * n_taps * n_embd)?;
                e.copy_view_into(&mut kept, 0, &keep_view, keep * n_taps * n_embd)?;
                let f = draft.ctx_features(e, &kept, keep)?;
                let pos_k: Vec<i32> = ((ctx_len as i32)..(ctx_len + keep) as i32).collect();
                draft.ingest_ctx(e, &mut dkv, &f, &pos_k, keep)?;
                ctx_len += keep;
            }
            last = next;
        }
        e.set_verify_exact(false);
        if std::env::var("BW24_SPEC_STATS").as_deref() == Ok("1") {
            eprintln!("[dflash] acceptance {accepted}/{attempted} = {:.3}",
                      accepted as f64 / attempted.max(1) as f64);
        }
        Ok(out)
    }
}
