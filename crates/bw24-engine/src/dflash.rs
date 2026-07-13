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
        let upw = |name: &str| -> Result<GpuTensor, Box<dyn std::error::Error>> {
            let (info, bytes) = st.raw(name).ok_or_else(|| format!("missing tensor {name}"))?;
            let shape = info.ne(); // ggml order: ne[0]=in_f, ne[1]=out_f
            Ok(GpuTensor::Float { data: e.htod(&bf16_to_f32(bytes))?,
                                  ne: shape.to_vec() })
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
    pub fn forward(
        &self, e: &Engine, target_hidden: &CudaSlice<f32>, noise_emb: &CudaSlice<f32>,
        pos: &[i32], ctx: usize,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let c = &self.cfg;
        let (h, nh, nkv, hd) = (c.hidden, c.n_head, c.n_kv, c.head_dim);
        let b = c.block_size;
        assert_eq!(pos.len(), ctx + b, "pos covers ctx rows then block rows");
        let n_taps = c.target_layer_ids.len();

        // ctx features: hidden_norm(fc(target_hidden))  [ctx, hidden]
        let fc_out = self.mm(e, &self.fc, target_hidden, ctx, n_taps * h, h)?;
        let mut ctx_f = e.uninit(ctx * h)?;
        e.rms_norm(&fc_out, &self.hidden_norm, &mut ctx_f, h, ctx, c.eps)?;
        if let Ok(dir) = std::env::var("BW24_DFLASH_DUMP") {
            let v = e.dtoh(&ctx_f)?;
            let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
            std::fs::write(format!("{dir}/bw24-ctx_features.f32"), bytes)?;
        }

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
            let k0c = self.mm(e, &l.wk, &ctx_f, ctx, h, nkv * hd)?;
            let v0c = self.mm(e, &l.wv, &ctx_f, ctx, h, nkv * hd)?;
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
