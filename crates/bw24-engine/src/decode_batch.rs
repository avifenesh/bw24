//! Batched decode step — B sequences share one fused pass (ARCHITECTURE-H100.md §3 B2').
//!
//! The bandwidth thesis: decode is weight-stream-bound, so every projection at m=B rows
//! amortizes one weight read across B sequences. Row-parallel ops (norm/rope/quantize/
//! activation) batch trivially — they are the SAME kernels prefill already runs at T rows.
//! Only truly per-sequence state stays in a loop: KV append + fa_decode over each cache,
//! and the GDN/conv recurrent step (v1: per-seq loop via the existing single-seq path;
//! a blockIdx.z-batched GDN state kernel is the v2 fusion).
//!
//! EXACTNESS CONTRACT (the law this module lives under):
//! - B == 1 must be BIT-IDENTICAL to `decode_step_h` (gate: decode-batch-gate).
//! - 2 <= B <= 8: each row rides the m=2..9 verify-tier mmvq kernels, which are per-row
//!   bit-identical to m=1 (the spec-exactness machinery decode_step_t relies on). Each
//!   sequence's token stream must equal its isolated single-seq run (worker.rs contract:
//!   "byte-identical to isolated").
//! - B >= 16 would cross into the GEMM tier (block-scale f32 rounding — a DIFFERENT
//!   numeric config). Deliberately refused in v1: assert B <= 8.
//!
//! v1 scope: the hybrid (Qwen3.5-class) non-gemma4 trunk. Fused m=1 micro-launches
//! (fused3 QKV, cross-layer add+norm+q8 chain) are NOT used — the unfused sequence is
//! bit-identical (kernel_check: add_rms_norm == add;rms_norm; _q8_1 == +quantize_q8_1)
//! and keeps the batched path simple. Batched fusions are tuning work, not correctness.

use crate::cache::Cache;
use crate::hybrid::{HybridModel, Mixer};
use crate::Engine;
use cudarc::driver::CudaSlice;

impl HybridModel {
    /// One batched greedy-decode step over B independent sequences.
    /// `tokens[b]` is sequence b's input token; `caches[b]` its private cache (position,
    /// quantized KV, GDN/conv state). Returns the B logits rows (host, [n_vocab] each).
    /// Each cache's pos/len advance exactly as `decode_step_h` would.
    pub fn decode_step_batch(
        &self,
        e: &Engine,
        tokens: &[u32],
        caches: &mut [&mut Cache],
    ) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        let b_n = tokens.len();
        assert!(b_n >= 1 && b_n == caches.len(), "tokens/caches length mismatch");
        assert!(
            b_n <= 8,
            "decode_step_batch: B={b_n} > 8 crosses the m>=16 GEMM tier (a different \
             numeric config) — refused until the batched-tier exactness policy lands"
        );
        assert!(
            !self.is_gemma4_e4b() && self.cfg.gemma4.is_none(),
            "decode_step_batch v1 covers the hybrid non-gemma4 trunk only"
        );
        let cfg = &self.cfg;
        let n_embd = cfg.n_embd as usize;
        let eps = cfg.rms_eps;
        let n_head = cfg.n_head as usize;
        let n_head_kv = cfg.n_head_kv as usize;
        let head_dim = cfg.head_dim_k as usize;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let rope_dims = cfg.rope_dim_count as usize;

        // Per-row rope positions (each sequence at its own depth).
        let pos_v: Vec<i32> = caches.iter().map(|c| c.pos as i32).collect();
        let pos_d = e.htod_i32(&pos_v)?;

        // Embed all B tokens -> x [B, n_embd] (host gather, one H2D).
        let mut x = e.htod(&self.embd.gather(n_embd, tokens))?;

        for (il, layer) in self.layers.iter().enumerate() {
            // ---- attn_norm + q8_1 quantize, batched (B rows) ----
            let anorm = layer.attn_norm.float_data();
            let mut xn = e.uninit(b_n * n_embd)?;
            e.rms_norm(&x, anorm, &mut xn, n_embd, b_n, eps)?;
            let (hq, hd) = e.quantize_q8_1(&xn, b_n, n_embd)?;

            // ---- mixer ----
            let mixed: CudaSlice<f32> = match &layer.mixer {
                Mixer::Full(fa) => {
                    // Batched projections: one weight read serves all B rows.
                    let qf = e.matmul_pre(&fa.wq, &hq, &hd, &xn, b_n)?;
                    let mut k = e.matmul_pre(&fa.wk, &hq, &hd, &xn, b_n)?;
                    let v = e.matmul_pre(&fa.wv, &hq, &hd, &xn, b_n)?;

                    let gated = cfg.attn_out_gate();
                    let (mut q, gate) = if gated {
                        let mut qs = e.uninit(b_n * n_head * head_dim)?;
                        let mut gs = e.uninit(b_n * n_head * head_dim)?;
                        e.q_gate_split(&qf, &mut qs, &mut gs, head_dim, n_head, b_n)?;
                        (qs, Some(gs))
                    } else {
                        (qf, None)
                    };

                    // QK-norm over B*n_head rows, rope with per-row positions.
                    let mut qn = e.uninit(b_n * n_head * head_dim)?;
                    e.rms_norm(&q, fa.q_norm.float_data(), &mut qn, head_dim, b_n * n_head, eps)?;
                    q = qn;
                    let mut kn = e.uninit(b_n * n_head_kv * head_dim)?;
                    e.rms_norm(&k, fa.k_norm.float_data(), &mut kn, head_dim, b_n * n_head_kv, eps)?;
                    k = kn;
                    e.rope_neox(&mut q, &pos_d, head_dim, rope_dims, n_head, b_n,
                                cfg.rope_freq_base, 1.0)?;
                    e.rope_neox(&mut k, &pos_d, head_dim, rope_dims, n_head_kv, b_n,
                                cfg.rope_freq_base, 1.0)?;

                    // Per-sequence: append this row's k/v into that cache, attend, place the
                    // attention row. Row views — no scratch copies on the append side.
                    let q_dim = n_head * head_dim;
                    let kv_dim = n_head_kv * head_dim;
                    let mut attn = e.uninit(b_n * q_dim)?;
                    for (bi, cache) in caches.iter_mut().enumerate() {
                        let kvl = cache.kv[il].as_mut().unwrap();
                        let k_row = k.slice(bi * kv_dim..(bi + 1) * kv_dim);
                        let v_row = v.slice(bi * kv_dim..(bi + 1) * kv_dim);
                        e.append_kv_quantized_view(
                            &k_row, &v_row, &mut kvl.k, &mut kvl.v, kvl.len,
                            kvl.kv_dim_k, kvl.kv_dim_v, kvl.k_tok_bytes, kvl.v_tok_bytes,
                            Engine::kv_fp8_on(),
                        )?;
                        kvl.len += 1;
                        let t_kv = kvl.len;
                        let k_view = e.view_u8(&kvl.k, t_kv * kvl.k_tok_bytes);
                        let v_view = e.view_u8(&kvl.v, t_kv * kvl.v_tok_bytes);
                        // fa_decode wants a q slice starting at row bi: v1 scratch-copies the
                        // row (q8-class µs cost). A q-view fa_decode twin is a v2 cleanup.
                        let mut q_row = e.uninit(q_dim)?;
                        e.dtod_copy_view(&q.slice(bi * q_dim..(bi + 1) * q_dim), &mut q_row)?;
                        let mut a_row = e.uninit(q_dim)?;
                        e.fa_decode_kvmod(
                            &q_row, &k_view, &v_view, &mut a_row, head_dim, n_head, n_head_kv,
                            t_kv, scale, kvl.k_tok_bytes, kvl.v_tok_bytes, Engine::kv_fp8_on(),
                        )?;
                        e.dtod_copy_into(&a_row, &mut attn, bi * q_dim)?;
                    }

                    // Output gate (element-wise — batches whole) + o-proj at m=B.
                    let attn_g = match &gate {
                        Some(g) => {
                            let n = b_n * q_dim;
                            let mut gsig = e.uninit(n)?;
                            e.sigmoid(g, &mut gsig, n)?;
                            let mut ag = e.uninit(n)?;
                            e.mul(&attn, &gsig, &mut ag, n)?;
                            ag
                        }
                        None => attn,
                    };
                    e.matmul(&fa.wo, &attn_g, b_n)?
                }
                Mixer::Linear(la) => {
                    // v1: per-seq loop through the EXISTING single-seq GDN/conv path with
                    // row views of the batched norm+quantize outputs. The recurrent state is
                    // per-sequence dense state — batching it is a kernel change (blockIdx.z
                    // over a state array), scheduled with the paged-KV work, not here.
                    let mut mixed = e.uninit(b_n * n_embd)?;
                    let hq_row_bytes = hq.len() / b_n;
                    let hd_row_len = hd.len() / b_n;
                    for (bi, cache) in caches.iter_mut().enumerate() {
                        let mut xn_row = e.uninit(n_embd)?;
                        e.dtod_copy_view(&xn.slice(bi * n_embd..(bi + 1) * n_embd), &mut xn_row)?;
                        let mut hq_row = e.uninit_i8(hq_row_bytes)?;
                        e.dtod_copy_view_i8(
                            &hq.slice(bi * hq_row_bytes..(bi + 1) * hq_row_bytes), &mut hq_row)?;
                        let mut hd_row = e.uninit(hd_row_len)?;
                        e.dtod_copy_view(
                            &hd.slice(bi * hd_row_len..(bi + 1) * hd_row_len), &mut hd_row)?;
                        let out_row = if self.mixer_in_q8_1_fast(e, &layer.mixer) {
                            self.linear_attn_decode_pre(e, la, &xn_row, &hq_row, &hd_row,
                                                        cache, il, false)?
                        } else {
                            self.linear_attn_decode(e, la, &xn_row, cache, il)?
                        };
                        e.dtod_copy_into(&out_row, &mut mixed, bi * n_embd)?;
                    }
                    mixed
                }
            };

            // ---- residual add + post_attn_norm + FFN, batched ----
            let pnorm = layer.post_attn_norm.float_data();
            let mut x1 = e.uninit(b_n * n_embd)?;
            let mut z = e.uninit(b_n * n_embd)?;
            e.add_rms_norm(&x, &mixed, pnorm, &mut x1, &mut z, n_embd, b_n, eps)?;
            let ffn_out = match &layer.ffn {
                crate::hybrid::Ffn::Dense { ffn_gate, ffn_up, ffn_down } => {
                    // v1 covers the SiLU family; M3's swigluoai clamp rides a scaled epilogue
                    // (m=1 fused tier) — batched M3 lands with the batched-fusion pass.
                    assert!(self.cfg.m3.is_none(),
                            "decode_step_batch v1: M3 swigluoai FFN not yet batched");
                    let n_ff = ffn_gate.out_features();
                    let (zq, zd) = e.quantize_q8_1(&z, b_n, n_embd)?;
                    let g = e.matmul_pre(ffn_gate, &zq, &zd, &z, b_n)?;
                    let u = e.matmul_pre(ffn_up, &zq, &zd, &z, b_n)?;
                    let mut act = e.uninit(b_n * n_ff)?;
                    e.silu_mul(&g, &u, &mut act, b_n * n_ff)?;
                    let (aq, ad) = e.quantize_q8_1(&act, b_n, n_ff)?;
                    e.matmul_pre(ffn_down, &aq, &ad, &act, b_n)?
                }
                crate::hybrid::Ffn::Moe(m) => self.moe_ffn_il_zq8(e, m, &z, None, b_n, il as u16)?,
            };
            // next-layer input x = x1 + ffn_out (batched element-wise add)
            let mut x2 = e.uninit(b_n * n_embd)?;
            e.add(&x1, &ffn_out, &mut x2, b_n * n_embd)?;
            x = x2;
        }

        // ---- output norm + lm_head at m=B, one D2H ----
        let mut hn = e.uninit(b_n * n_embd)?;
        e.rms_norm(&x, self.output_norm.float_data(), &mut hn, n_embd, b_n, eps)?;
        let logits = e.matmul(&self.output, &hn, b_n)?;
        let host = e.dtoh(&logits)?;
        let n_vocab = host.len() / b_n;
        for c in caches.iter_mut() {
            c.pos += 1;
        }
        Ok((0..b_n).map(|bi| host[bi * n_vocab..(bi + 1) * n_vocab].to_vec()).collect())
    }
}
