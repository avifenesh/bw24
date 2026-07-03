//! Qwen3.5 MTP (NextN) greedy speculative decode (research/mtp/MTP-PLAN.md §A/§B/§C/§D).
//!
//! Greedy spec decode is MATHEMATICALLY EXACT: the accepted+bonus token stream is token-for-token
//! identical to plain greedy `generate`. This module provides:
//!   - `mtp_head_forward`  (§A, T=1): one NextN draft-token forward.
//!   - `decode_step_t`     (§D.3, T=K+1): batched target verify forward, all-column logits.
//!   - `generate_spec`     (§B): the draft/verify/accept/rollback orchestrator.
//! Cache snapshot/rollback lives in cache.rs (§D.4). The MTP head uses its OWN scratch KV (§D.6).

use cudarc::driver::CudaSlice;
use crate::Engine;
use crate::hybrid::{HybridModel, Mixer, FullAttnLayer, LinearAttnLayer, MtpHead};
use crate::cache::{Cache, KvLayer};
use crate::forward::argmax;

/// Tiny scratch KV for the MTP block (one full-attn layer). Reset each draft round (§D.6).
struct MtpScratch {
    kv: KvLayer,
}
impl MtpScratch {
    fn new(e: &Engine, cfg: &bw24_gguf::config::ModelConfig, cap: usize)
           -> Result<Self, Box<dyn std::error::Error>> {
        let n_head_kv = cfg.n_head_kv as usize;
        let head_dim_k = cfg.head_dim_k as usize;
        let head_dim_v = cfg.head_dim_v as usize;
        assert!(head_dim_k % 32 == 0 && head_dim_v % 32 == 0,
                "KVQUANT requires head_dim%32==0 (MTP scratch)");
        let kv_dim_k = head_dim_k * n_head_kv;
        let kv_dim_v = head_dim_v * n_head_kv;
        let k_tok_bytes = (kv_dim_k / 32) * 34;
        let v_tok_bytes = (kv_dim_v / 32) * 24;
        Ok(MtpScratch { kv: KvLayer {
            k: e.alloc_u8(cap * k_tok_bytes)?, v: e.alloc_u8(cap * v_tok_bytes)?,
            kv_dim_k, kv_dim_v, k_tok_bytes, v_tok_bytes, len: 0,
            len_d: e.htod_i32(&[0])?,
        } })
    }
    fn reset(&mut self) { self.kv.len = 0; }
}

impl HybridModel {
    /// NextN head forward for ONE draft token (§A ops 1-13, T=1).
    /// Inputs: `e_tok` = the token to predict FROM (last committed / previous draft); `h_seed` =
    /// the trunk's pre-output_norm hidden of that token (§A op 2 input). `mtp_pos` = absolute
    /// position of the token being predicted from. Returns (draft_logits[n_vocab] host, h_nextn dev).
    /// `h_nextn` (§A op 10) becomes `h_seed` for the next autoregressive draft step.
    /// Device-resident: returns draft logits ON DEVICE (no [n_vocab] dtoh). The greedy draft
    /// loop only needs argmax — paired with `argmax_token_device` this cuts the ~600KB logits
    /// transfer + host argmax per draft token from the K-token draft chain.
    fn mtp_head_forward_dev(&self, e: &Engine, mtp: &MtpHead, e_tok: u32, h_seed: &CudaSlice<f32>,
                            scratch: &mut MtpScratch, mtp_pos: usize)
                            -> Result<(CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let n_embd = cfg.n_embd as usize;
        let eps = cfg.rms_eps;
        let pos_d = e.htod_i32(&[mtp_pos as i32])?;

        // op A: embed the predict-from token e -> [n_embd]
        let e_emb = e.htod(&self.embd.gather(n_embd, &[e_tok]))?;

        // op 1/2: e_norm = RMSNorm(e, enorm); h_norm = RMSNorm(h_seed, hnorm)
        let mut e_norm = e.zeros(n_embd)?;
        e.rms_norm(&e_emb, mtp.enorm.float_data(), &mut e_norm, n_embd, 1, eps)?;
        let mut h_norm = e.zeros(n_embd)?;
        e.rms_norm(h_seed, mtp.hnorm.float_data(), &mut h_norm, n_embd, 1, eps)?;

        // op 3: concat = [e_norm ; h_norm] -> [2*n_embd], e_norm in [0,n_embd), h_norm in [n_embd,2n_embd)
        let mut concat = e.zeros(2 * n_embd)?;
        e.copy_into(&mut concat, 0, &e_norm, n_embd)?;
        e.copy_into(&mut concat, n_embd, &h_norm, n_embd)?;

        // op 4: inpSA = eh_proj @ concat  (eh_proj [2*n_embd, n_embd]) -> [n_embd]
        let inp_sa = e.matmul(&mtp.eh_proj, &concat, 1)?;

        // op 5: a_norm = RMSNorm(inpSA, attn_norm)
        let mut a_norm = e.zeros(n_embd)?;
        e.rms_norm(&inp_sa, mtp.attn_norm.float_data(), &mut a_norm, n_embd, 1, eps)?;

        // op 6: attention (same body as full_attn_decode, on the MTP block's own scratch KV).
        let attn_out = match &mtp.mixer {
            Mixer::Full(fa) => self.mtp_full_attn(e, fa, &a_norm, &pos_d, scratch)?,
            Mixer::Linear(_) => panic!("MTP block is full-attn in qwen35; linear MTP not supported"),
        };

        // op 7: x1 = inpSA + attn_out
        let mut x1 = e.zeros(n_embd)?;
        e.add(&inp_sa, &attn_out, &mut x1, n_embd)?;

        // op 8: z = RMSNorm(x1, post_attn_norm)  (pre-FFN norm)
        let mut z = e.zeros(n_embd)?;
        e.rms_norm(&x1, mtp.post_attn_norm.float_data(), &mut z, n_embd, 1, eps)?;

        // op 9: FFN (Dense or MoE) — same as the trunk decode FFN
        let ffn_out = match &mtp.ffn {
            crate::hybrid::Ffn::Dense { ffn_gate, ffn_up, ffn_down } => {
                let n_ff = ffn_gate.out_features();
                let (gate, up) = if e.uses_q8_1_fast(ffn_gate) && e.uses_q8_1_fast(ffn_up) {
                    let (zq, zd) = e.quantize_q8_1(&z, 1, n_embd)?;
                    (e.matmul_pre(ffn_gate, &zq, &zd, &z, 1)?, e.matmul_pre(ffn_up, &zq, &zd, &z, 1)?)
                } else {
                    (e.matmul(ffn_gate, &z, 1)?, e.matmul(ffn_up, &z, 1)?)
                };
                let mut act = e.zeros(n_ff)?;
                e.silu_mul(&gate, &up, &mut act, n_ff)?;
                e.matmul(ffn_down, &act, 1)?
            }
            // MTP head is a distinct block — key its experts under a separate layer index (u16::MAX)
            // so they never alias trunk layer 0's cache keys.
            crate::hybrid::Ffn::Moe(m) => self.moe_ffn_il(e, m, &z, 1, u16::MAX)?,
        };

        // op 10: h_nextn = x1 + ffn_out
        let mut h_nextn = e.zeros(n_embd)?;
        e.add(&x1, &ffn_out, &mut h_nextn, n_embd)?;

        // op 11: final = RMSNorm(h_nextn, shared_head_norm OR output_norm)
        let final_norm = mtp.shared_head_norm.as_ref().unwrap_or(&self.output_norm);
        let mut final_h = e.zeros(n_embd)?;
        e.rms_norm(&h_nextn, final_norm.float_data(), &mut final_h, n_embd, 1, eps)?;

        // op 12: draft_logits = (shared_head_head OR output) @ final — stays ON DEVICE.
        let head = mtp.shared_head_head.as_ref().unwrap_or(&self.output);
        let logits = e.matmul(head, &final_h, 1)?;
        Ok((logits, h_nextn))
    }

    /// MTP-block full attention, T=1, on the scratch KV (mirror of full_attn_decode but on a
    /// caller-owned KvLayer instead of `cache.kv[il]`).
    fn mtp_full_attn(&self, e: &Engine, fa: &FullAttnLayer, h: &CudaSlice<f32>,
                     pos_d: &CudaSlice<i32>, scratch: &mut MtpScratch)
                     -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let n_head = cfg.n_head as usize;
        let n_head_kv = cfg.n_head_kv as usize;
        let head_dim = cfg.head_dim_k as usize;
        let eps = cfg.rms_eps;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let n_embd = cfg.n_embd as usize;

        let (qf, mut k, v) = if e.uses_q8_1_fast(&fa.wq) && e.uses_q8_1_fast(&fa.wk) && e.uses_q8_1_fast(&fa.wv) {
            let (hq, hd) = e.quantize_q8_1(h, 1, n_embd)?;
            (e.matmul_pre(&fa.wq, &hq, &hd, h, 1)?, e.matmul_pre(&fa.wk, &hq, &hd, h, 1)?, e.matmul_pre(&fa.wv, &hq, &hd, h, 1)?)
        } else {
            (e.matmul(&fa.wq, h, 1)?, e.matmul(&fa.wk, h, 1)?, e.matmul(&fa.wv, h, 1)?)
        };
        let mut q = e.zeros(n_head * head_dim)?;
        let mut gate = e.zeros(n_head * head_dim)?;
        e.q_gate_split(&qf, &mut q, &mut gate, head_dim, n_head, 1)?;

        let mut qn = e.zeros(n_head * head_dim)?;
        e.rms_norm(&q, fa.q_norm.float_data(), &mut qn, head_dim, n_head, eps)?;
        q = qn;
        let mut kn = e.zeros(n_head_kv * head_dim)?;
        e.rms_norm(&k, fa.k_norm.float_data(), &mut kn, head_dim, n_head_kv, eps)?;
        k = kn;
        let rope_dims = cfg.rope_dim_count as usize;
        e.rope_neox(&mut q, pos_d, head_dim, rope_dims, n_head, 1, cfg.rope_freq_base, 1.0)?;
        e.rope_neox(&mut k, pos_d, head_dim, rope_dims, n_head_kv, 1, cfg.rope_freq_base, 1.0)?;

        let kv = &mut scratch.kv;
        e.append_kv_quantized(&k, &v, &mut kv.k, &mut kv.v, kv.len,
                              kv.kv_dim_k, kv.kv_dim_v, kv.k_tok_bytes, kv.v_tok_bytes)?;
        kv.len += 1;
        let t_kv = kv.len;
        let (ktb, vtb) = (kv.k_tok_bytes, kv.v_tok_bytes);
        let k_view = e.view_u8(&kv.k, t_kv * kv.k_tok_bytes);
        let v_view = e.view_u8(&kv.v, t_kv * kv.v_tok_bytes);
        let mut attn = e.zeros(n_head * head_dim)?;
        e.fa_decode(&q, &k_view, &v_view, &mut attn, head_dim, n_head, n_head_kv, t_kv, scale, ktb, vtb)?;

        let mut gsig = e.zeros(n_head * head_dim)?;
        e.sigmoid(&gate, &mut gsig, n_head * head_dim)?;
        let mut attn_g = e.zeros(n_head * head_dim)?;
        e.mul(&attn, &gsig, &mut attn_g, n_head * head_dim)?;
        Ok(e.matmul(&fa.wo, &attn_g, 1)?)
    }

    /// Batched target verify forward over `tokens` at positions `pos0..pos0+T` (§D.3, T=K+1).
    /// Returns ALL T logit columns (host f32, [T*n_vocab]); appends T cols to every full-attn KV
    /// and advances every linear-attn recur state by T steps (the recur steps are SEQUENTIAL T=1).
    /// Advances `cache.pos` by T.
    pub fn decode_step_t(&self, e: &Engine, tokens: &[u32], pos0: usize, cache: &mut Cache)
                         -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        Ok(self.decode_step_t_h(e, tokens, pos0, cache)?.0)
    }

    /// Like `decode_step_t` but ALSO returns the LAST column's pre-output_norm hidden (h_seed for
    /// the next draft round). This lets partial-accept replay run as ONE batched T=(n_acc+1) forward
    /// (single weight read) instead of n_acc+1 separate T=1 decode_steps (n_acc+1 weight reads).
    /// At batch=1 decode is bandwidth-bound, so batching the replay is THE MTP profitability lever.
    pub fn decode_step_t_h(&self, e: &Engine, tokens: &[u32], pos0: usize, cache: &mut Cache)
                         -> Result<(Vec<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let n_embd = cfg.n_embd as usize;
        let eps = cfg.rms_eps;
        let t = tokens.len();
        let pos_vec: Vec<i32> = (0..t).map(|i| (pos0 + i) as i32).collect();
        let pos_d = e.htod_i32(&pos_vec)?;

        // embed T tokens -> [T, n_embd] token-major
        let mut x = e.htod(&self.embd.gather(n_embd, tokens))?;

        for (il, layer) in self.layers.iter().enumerate() {
            let mut h = e.zeros(t * n_embd)?;
            e.rms_norm(&x, layer.attn_norm.float_data(), &mut h, n_embd, t, eps)?;

            let mixed = match &layer.mixer {
                Mixer::Full(fa) => self.full_attn_verify(e, fa, &h, &pos_d, t, cache, il)?,
                Mixer::Linear(la) => {
                    // BATCHED linear verify (2026-07-03, the MTP-profit lever): one T-token pass —
                    // batched projections (weight read ONCE, hits the m=2-4 weight-resident matvec),
                    // carried-state conv (ssm_conv1d_tm_state), GDN prep on the prefill kernels, and
                    // ONE gdn_scan whose internal sequential t-loop is the SAME recurrence as T
                    // chained T=1 steps (bit-identical). Falls back to the sequential per-column
                    // chain when T < d_conv-1 (conv ring update needs T >= pad).
                    if t >= 3 {
                        self.linear_attn_verify_t(e, la, &h, t, cache, il)?
                    } else {
                        let mut out = e.zeros(t * n_embd)?;
                        for col in 0..t {
                            let mut h_col = e.zeros(n_embd)?;
                            let src = h.slice(col * n_embd..(col + 1) * n_embd);
                            e.copy_view_into(&mut h_col, 0, &src, n_embd)?;
                            let m_col = self.linear_attn_decode(e, la, &h_col, cache, il)?;
                            e.copy_into(&mut out, col * n_embd, &m_col, n_embd)?;
                        }
                        out
                    }
                }
            };

            let mut x1 = e.zeros(t * n_embd)?;
            e.add(&x, &mixed, &mut x1, t * n_embd)?;

            let mut z = e.zeros(t * n_embd)?;
            e.rms_norm(&x1, layer.post_attn_norm.float_data(), &mut z, n_embd, t, eps)?;
            let ffn_out = match &layer.ffn {
                crate::hybrid::Ffn::Dense { ffn_gate, ffn_up, ffn_down } => {
                    let n_ff = ffn_gate.out_features();
                    let (gate, up) = if e.uses_q8_1_fast(ffn_gate) && e.uses_q8_1_fast(ffn_up) {
                        let (zq, zd) = e.quantize_q8_1(&z, t, n_embd)?;
                        (e.matmul_pre(ffn_gate, &zq, &zd, &z, t)?, e.matmul_pre(ffn_up, &zq, &zd, &z, t)?)
                    } else {
                        (e.matmul(ffn_gate, &z, t)?, e.matmul(ffn_up, &z, t)?)
                    };
                    let mut act = e.zeros(t * n_ff)?;
                    e.silu_mul(&gate, &up, &mut act, t * n_ff)?;
                    e.matmul(ffn_down, &act, t)?
                }
                crate::hybrid::Ffn::Moe(m) => self.moe_ffn_il(e, m, &z, t, il as u16)?,
            };
            let mut x2 = e.zeros(t * n_embd)?;
            e.add(&x1, &ffn_out, &mut x2, t * n_embd)?;
            x = x2;
        }

        // h_seed for the next round = LAST column's pre-output_norm hidden ([n_embd]).
        let h_seed = {
            let mut hs = e.zeros(n_embd)?;
            let src = x.slice((t - 1) * n_embd..t * n_embd);
            e.copy_view_into(&mut hs, 0, &src, n_embd)?;
            hs
        };
        let mut hn = e.zeros(t * n_embd)?;
        e.rms_norm(&x, self.output_norm.float_data(), &mut hn, n_embd, t, eps)?;
        let logits = e.matmul(&self.output, &hn, t)?;
        let host = e.dtoh(&logits)?;
        cache.pos += t;
        Ok((host, h_seed))
    }

    /// BATCHED linear-attn verify (T=K+1): the whole layer in ~10 launches instead of T x the
    /// T=1 decode chain (T x ~12 launches + T weight reads of the four projections). The GDN
    /// recurrence itself is inherently sequential — gdn_scan_s128 runs its internal t-loop with
    /// the SAME per-token math as chained T=1 calls (bit-identical state evolution); everything
    /// around it (projections, conv, prep, gated norm, out-proj) batches. Advances conv ring +
    /// ssm state exactly like T sequential decode steps.
    fn linear_attn_verify_t(&self, e: &Engine, la: &LinearAttnLayer, h: &CudaSlice<f32>, t: usize,
                            cache: &mut Cache, il: usize)
                            -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let ssm = cfg.ssm.as_ref().unwrap();
        let d_state = ssm.state_size as usize;
        let num_k = ssm.group_count as usize;
        let num_v = ssm.time_step_rank as usize;
        let d_conv = ssm.conv_kernel as usize;
        let key_dim = d_state * num_k;
        let conv_dim = key_dim * 2 + d_state * num_v;
        let eps = cfg.rms_eps;
        let scale = 1.0 / (d_state as f32).sqrt();

        // batched projections ([T, n_embd] @ W^T — the m=2-4 band hits the weight-resident matvec)
        let qkv_mixed = e.matmul(&la.wqkv, h, t)?;
        let z = e.matmul(&la.wqkv_gate, h, t)?;
        let beta_raw = e.matmul(&la.ssm_beta, h, t)?;
        let alpha = e.matmul(&la.ssm_alpha, h, t)?;

        // conv with CARRIED state + ring roll (T >= pad guaranteed by caller).
        let rl = cache.recur[il].as_mut().unwrap();
        let mut conv_out = e.uninit(conv_dim * t)?;
        e.ssm_conv1d_tm_state(&qkv_mixed, &mut rl.conv_state, la.ssm_conv1d.float_data(),
                              &mut conv_out, conv_dim, t, d_conv)?;

        // GDN prep via the prefill kernels (repack + L2 + sigmoid + glog), T-wide.
        let mut q_g = e.uninit(d_state * num_v * t)?;
        let mut k_g = e.uninit(d_state * num_v * t)?;
        let mut v_g = e.uninit(d_state * num_v * t)?;
        e.qkv_to_gdn_repack(&conv_out, &mut q_g, &mut k_g, &mut v_g, d_state, num_v, num_k, key_dim, t)?;
        let mut q_l2 = e.uninit(d_state * num_v * t)?;
        e.l2_norm(&q_g, &mut q_l2, d_state, num_v * t, eps)?;
        let mut k_l2 = e.uninit(d_state * num_v * t)?;
        e.l2_norm(&k_g, &mut k_l2, d_state, num_v * t, eps)?;
        let mut beta = e.uninit(t * num_v)?;
        e.sigmoid(&beta_raw, &mut beta, t * num_v)?;
        let mut g_log = e.uninit(t * num_v)?;
        e.gdn_glog(&alpha, la.ssm_dt.float_data(), la.ssm_a.float_data(), &mut g_log, num_v, t)?;

        // ONE gdn_scan over T tokens from the carried state (internal sequential loop ==
        // T chained T=1 steps). Ping-pong the resident buffers like eager decode.
        let mut o = e.uninit(d_state * num_v * t)?;
        {
            let crate::cache::RecurLayer { ssm_state, ssm_state_alt, .. } = rl;
            e.gdn_scan_s128(&q_l2, &k_l2, &v_g, &g_log, &beta, ssm_state, ssm_state_alt, &mut o, num_v, t, scale)?;
        }
        std::mem::swap(&mut rl.ssm_state, &mut rl.ssm_state_alt);

        // gated RMSNorm + out projection, T-wide.
        let mut gn = e.uninit(d_state * num_v * t)?;
        e.gated_rmsnorm(&o, la.ssm_norm.float_data(), &z, &mut gn, d_state, num_v * t, eps)?;
        Ok(e.matmul(&la.ssm_out, &gn, t)?)
    }

    /// EAGLE3 aux-capturing verify forward over `tokens` (T) — mirrors `decode_step_t_h` exactly
    /// (same KV append, same causal verify, same recur advance) but ALSO clones the aux residual-
    /// stream hiddens (blocks in `aux_layers`) for TWO columns: the LAST column (always) and the
    /// optional `pred_col` (the EAGLE seed = bonus's predecessor). Returns
    /// (all_T_logits host, last_col_aux, pred_col_aux?). Used by the EAGLE3 orchestrator's commit.
    pub fn decode_step_t_aux2(&self, e: &Engine, tokens: &[u32], pos0: usize, cache: &mut Cache,
                              aux_layers: &[usize], pred_col: Option<usize>)
        -> Result<(Vec<f32>, Vec<CudaSlice<f32>>, Option<Vec<CudaSlice<f32>>>), Box<dyn std::error::Error>>
    {
        let cfg = &self.cfg;
        let n_embd = cfg.n_embd as usize;
        let eps = cfg.rms_eps;
        let t = tokens.len();
        let pos_vec: Vec<i32> = (0..t).map(|i| (pos0 + i) as i32).collect();
        let pos_d = e.htod_i32(&pos_vec)?;
        let mut x = e.htod(&self.embd.gather(n_embd, tokens))?;
        let mut aux_last: Vec<CudaSlice<f32>> = Vec::with_capacity(aux_layers.len());
        let mut aux_pred: Vec<CudaSlice<f32>> = Vec::new();
        let want_pred = pred_col.is_some();

        for (il, layer) in self.layers.iter().enumerate() {
            let mut h = e.zeros(t * n_embd)?;
            e.rms_norm(&x, layer.attn_norm.float_data(), &mut h, n_embd, t, eps)?;
            let mixed = match &layer.mixer {
                Mixer::Full(fa) => self.full_attn_verify(e, fa, &h, &pos_d, t, cache, il)?,
                Mixer::Linear(la) => {
                    let mut out = e.zeros(t * n_embd)?;
                    for col in 0..t {
                        let mut h_col = e.zeros(n_embd)?;
                        let src = h.slice(col * n_embd..(col + 1) * n_embd);
                        e.copy_view_into(&mut h_col, 0, &src, n_embd)?;
                        let m_col = self.linear_attn_decode(e, la, &h_col, cache, il)?;
                        e.copy_into(&mut out, col * n_embd, &m_col, n_embd)?;
                    }
                    out
                }
            };
            let mut x1 = e.zeros(t * n_embd)?;
            e.add(&x, &mixed, &mut x1, t * n_embd)?;
            let mut z = e.zeros(t * n_embd)?;
            e.rms_norm(&x1, layer.post_attn_norm.float_data(), &mut z, n_embd, t, eps)?;
            let ffn_out = match &layer.ffn {
                crate::hybrid::Ffn::Dense { ffn_gate, ffn_up, ffn_down } => {
                    let n_ff = ffn_gate.out_features();
                    let (gate, up) = if e.uses_q8_1_fast(ffn_gate) && e.uses_q8_1_fast(ffn_up) {
                        let (zq, zd) = e.quantize_q8_1(&z, t, n_embd)?;
                        (e.matmul_pre(ffn_gate, &zq, &zd, &z, t)?, e.matmul_pre(ffn_up, &zq, &zd, &z, t)?)
                    } else {
                        (e.matmul(ffn_gate, &z, t)?, e.matmul(ffn_up, &z, t)?)
                    };
                    let mut act = e.zeros(t * n_ff)?;
                    e.silu_mul(&gate, &up, &mut act, t * n_ff)?;
                    e.matmul(ffn_down, &act, t)?
                }
                crate::hybrid::Ffn::Moe(m) => self.moe_ffn_il(e, m, &z, t, il as u16)?,
            };
            let mut x2 = e.zeros(t * n_embd)?;
            e.add(&x1, &ffn_out, &mut x2, t * n_embd)?;
            if aux_layers.contains(&il) {
                let mut a = e.zeros(n_embd)?;
                e.copy_view_into(&mut a, 0, &x2.slice((t - 1) * n_embd..t * n_embd), n_embd)?;
                aux_last.push(a);
                if let Some(pc) = pred_col {
                    let mut ap = e.zeros(n_embd)?;
                    e.copy_view_into(&mut ap, 0, &x2.slice(pc * n_embd..(pc + 1) * n_embd), n_embd)?;
                    aux_pred.push(ap);
                }
            }
            x = x2;
        }
        let mut hn = e.zeros(t * n_embd)?;
        e.rms_norm(&x, self.output_norm.float_data(), &mut hn, n_embd, t, eps)?;
        let logits = e.matmul(&self.output, &hn, t)?;
        let host = e.dtoh(&logits)?;
        cache.pos += t;
        Ok((host, aux_last, if want_pred { Some(aux_pred) } else { None }))
    }

    /// Full-attention mixer over T query tokens with a GROWING resident KV (verify path, §D.3).
    /// Appends the T new K/V columns to cache.kv[il] then attends causally over [0..len) via
    /// fa_prefill. Token-major [T, kv_dim] projection layout == cache row layout (single copy).
    fn full_attn_verify(&self, e: &Engine, fa: &FullAttnLayer, h: &CudaSlice<f32>,
                        pos_d: &CudaSlice<i32>, t: usize, cache: &mut Cache, il: usize)
                        -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let n_head = cfg.n_head as usize;
        let n_head_kv = cfg.n_head_kv as usize;
        let head_dim = cfg.head_dim_k as usize;
        let eps = cfg.rms_eps;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let n_embd = cfg.n_embd as usize;

        let (qf, mut k, v) = if e.uses_q8_1_fast(&fa.wq) && e.uses_q8_1_fast(&fa.wk) && e.uses_q8_1_fast(&fa.wv) {
            let (hq, hd) = e.quantize_q8_1(h, t, n_embd)?;
            (e.matmul_pre(&fa.wq, &hq, &hd, h, t)?, e.matmul_pre(&fa.wk, &hq, &hd, h, t)?, e.matmul_pre(&fa.wv, &hq, &hd, h, t)?)
        } else {
            (e.matmul(&fa.wq, h, t)?, e.matmul(&fa.wk, h, t)?, e.matmul(&fa.wv, h, t)?)
        };
        let mut q = e.zeros(t * n_head * head_dim)?;
        let mut gate = e.zeros(t * n_head * head_dim)?;
        e.q_gate_split(&qf, &mut q, &mut gate, head_dim, n_head, t)?;

        let mut qn = e.zeros(t * n_head * head_dim)?;
        e.rms_norm(&q, fa.q_norm.float_data(), &mut qn, head_dim, n_head * t, eps)?;
        q = qn;
        let mut kn = e.zeros(t * n_head_kv * head_dim)?;
        e.rms_norm(&k, fa.k_norm.float_data(), &mut kn, head_dim, n_head_kv * t, eps)?;
        k = kn;
        let rope_dims = cfg.rope_dim_count as usize;
        e.rope_neox(&mut q, pos_d, head_dim, rope_dims, n_head, t, cfg.rope_freq_base, 1.0)?;
        e.rope_neox(&mut k, pos_d, head_dim, rope_dims, n_head_kv, t, cfg.rope_freq_base, 1.0)?;

        // append T new K/V columns to the resident QUANTIZED cache. k/v are token-major [T, kv_dim]
        // f32; append-quantize each of the T token rows into the byte cache (q8_0 K / q5_1 V).
        let kvl = cache.kv[il].as_mut().unwrap();
        let (kv_dim_k, kv_dim_v, ktb, vtb) = (kvl.kv_dim_k, kvl.kv_dim_v, kvl.k_tok_bytes, kvl.v_tok_bytes);
        for i in 0..t {
            let k_row = k.slice(i * kv_dim_k..(i + 1) * kv_dim_k);
            let v_row = v.slice(i * kv_dim_v..(i + 1) * kv_dim_v);
            e.append_kv_quantized_view(&k_row, &v_row, &mut kvl.k, &mut kvl.v, kvl.len + i,
                                       kv_dim_k, kv_dim_v, ktb, vtb)?;
        }
        kvl.len += t;

        // BIT-IDENTICAL VERIFY ATTENTION (spec-exactness fix): run fa_decode PER TOKEN ROW so the
        // FP accumulation order is byte-for-byte identical to the eager decode path. fa_prefill uses
        // a different tile size (BLOCK_Q=64, BK=32) and online-softmax structure than fa_decode's
        // split-K + combine, which changes FP summation order and can flip argmax at tight logit
        // margins. Query row r attends to keys [0..base_len+r+1) — each successive row sees one
        // more key (the causal property). This matches eager: decode appends k at len, then fa_decode
        // sees t_kv = len+1 keys. The verify appends all T tokens first but adjusts the view per row.
        let mut attn = e.zeros(t * n_head * head_dim)?;
        let base_len = kvl.len - t;   // KV len BEFORE this round's T tokens were appended
        for r in 0..t {
            let t_kv_r = base_len + r + 1; // this row sees keys [0..t_kv_r)
            let k_view_r = e.view_u8(&kvl.k, t_kv_r * ktb);
            let v_view_r = e.view_u8(&kvl.v, t_kv_r * vtb);
            // copy q row into an owned buffer (fa_decode takes &CudaSlice, not CudaView)
            let mut q_row = e.zeros(n_head * head_dim)?;
            let q_src = q.slice(r * n_head * head_dim..(r + 1) * n_head * head_dim);
            e.copy_view_into(&mut q_row, 0, &q_src, n_head * head_dim)?;
            let mut attn_row = e.zeros(n_head * head_dim)?;
            e.fa_decode(&q_row, &k_view_r, &v_view_r, &mut attn_row, head_dim, n_head, n_head_kv, t_kv_r, scale, ktb, vtb)?;
            e.copy_into(&mut attn, r * n_head * head_dim, &attn_row, n_head * head_dim)?;
        }

        let mut gsig = e.zeros(t * n_head * head_dim)?;
        e.sigmoid(&gate, &mut gsig, t * n_head * head_dim)?;
        let mut attn_g = e.zeros(t * n_head * head_dim)?;
        e.mul(&attn, &gsig, &mut attn_g, t * n_head * head_dim)?;
        Ok(e.matmul(&fa.wo, &attn_g, t)?)
    }

    /// Greedy MTP speculative decode (§B). Token-identical to `generate(prompt, max_new)` but uses
    /// the NextN head to draft K tokens then verifies them in one batched target forward.
    /// Returns (generated tokens, total_drafted, total_accepted) so the caller can report
    /// acceptance rate. `k` = draft length per round.
    pub fn generate_spec(&self, e: &Engine, prompt: &[u32], max_new: usize, k: usize)
                         -> Result<(Vec<u32>, usize, usize), Box<dyn std::error::Error>> {
        assert!(k >= 1, "k must be >= 1");
        let mtp = self.mtp.as_ref().expect("generate_spec requires an MTP head (nextn_predict_layers>0)");
        let n_vocab = self.output.out_features();
        // FR-Spec: the draft head may be TRIMMED (fewer rows than n_vocab); the draft argmax runs
        // over the draft vocab and the winning index maps through d2t to a TARGET token id.
        // Everything downstream (verify/accept/commit) sees target ids only — exactness unchanged.
        let d_vocab = mtp.shared_head_head.as_ref().unwrap_or(&self.output).out_features();
        let n_embd = self.cfg.n_embd as usize;
        let max_ctx = prompt.len() + max_new + k + 8;
        let mut cache = Cache::new(e, &self.cfg, max_ctx)?;

        // prime: feed the prompt token-by-token (decode_step_h builds KV+state, returns h_seed).
        assert!(!prompt.is_empty(), "prompt must be non-empty");
        let mut prime_logits = Vec::new();
        for &tok in prompt {
            let (l, _h) = self.decode_step_h(e, tok, &mut cache)?;
            prime_logits = l;
        }

        let mut scratch = MtpScratch::new(e, &self.cfg, k + 1)?;
        let mut out: Vec<u32> = Vec::with_capacity(max_new);
        let mut total_drafted = 0usize;
        let mut total_accepted = 0usize;

        // First generated token = argmax of the prompt's last logits (== greedy's first token).
        // Emit it, then FEED it to establish the loop invariant below.
        let mut last_token = argmax(&prime_logits) as u32;
        out.push(last_token);
        // INVARIANT at loop top: `last_token` is the most-recently-committed/emitted token, its
        // KV+recur state IS in `cache` (cache.pos = position right AFTER last_token), `last_logits`
        // predicts the token that should FOLLOW last_token, and `h_seed` = last_token's
        // pre-output_norm hidden. Establish it by feeding last_token once (mirrors plain greedy).
        let (mut last_logits, mut h_seed) = self.decode_step_h(e, last_token, &mut cache)?;

        let debug_spec = std::env::var("BW24_DEBUG_SPEC").is_ok();
        let mut round = 0usize;
        // PERSISTENT snapshot buffers: allocate ONCE, refresh in place each round (was 2 fresh
        // D2D clones per linear layer per round = 48 allocs + ~50MB of pool churn per round).
        let mut snap = cache.snapshot(e)?;
        // BONUS FOLD (2026-07-04): after a FULL accept the bonus token is NOT committed with a
        // separate T=1 trunk pass (a full weight read per round). It stays PENDING and rides as
        // column 0 of the NEXT round's verify batch; the next draft chain seeds from the MTP
        // block's pseudo-hidden at the bonus position (one extra MTP-block pass, ~1/33 of a trunk
        // read). Verify still checks every emitted token against the target -> exactness holds by
        // construction; only DRAFT QUALITY can shift (pseudo-hidden seed), which the acceptance
        // numbers arbitrate. Partial accepts keep the batched replay (commits the bonus with a
        // TRUE trunk hidden) — so pseudo-seeding never compounds past a rejection.
        let mut pending: Option<u32> = None;   // bonus emitted but not yet committed to cache
        while out.len() < max_new {
            let pos = cache.pos;            // #tokens committed (EXCLUDES a pending bonus)
            cache.snapshot_into(e, &mut snap)?;  // §C: snapshot BEFORE draft+verify

            // --- 1. DRAFT k tokens with the NextN head (autoregressive, T=1 each) ---
            scratch.reset();
            let mut draft: Vec<u32> = Vec::with_capacity(k);
            let mut e_tok = last_token;
            let mut d_seed = e.clone_dtod(&h_seed)?;
            // P-MIN CONFIDENCE GATE (BW24_SPEC_PMIN, the serve script's --spec-draft-p-min
            // mechanism): stop the draft chain early when the head's softmax confidence in its
            // own pick drops below p_min — low-confidence tail tokens are the ones the verify
            // rejects, and every drafted-but-rejected token costs a full verify column. Exactness
            // is untouched (verify still checks whatever WAS drafted; a shorter draft is just a
            // smaller K this round). p=0.0/unset = always draft K (the prior behavior).
            static PMIN: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
            let p_min = *PMIN.get_or_init(|| {
                std::env::var("BW24_SPEC_PMIN").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0)
            });
            for j in 0..k {
                // GPU-ARGMAX DRAFT (2026-07-03): device logits + device argmax + 4-byte token read
                // instead of the ~600KB full-vocab dtoh + host argmax per draft token. Same
                // smallest-index tie-break as host argmax (argmax_gate-validated) -> same tokens.
                let mtp_pos = pos + if pending.is_some() { 1 } else { 0 } + j;
                let (dl_d, h_nextn) = self.mtp_head_forward_dev(e, mtp, e_tok, &d_seed, &mut scratch, mtp_pos)?;
                let tok_d = e.argmax_token_device(&dl_d, d_vocab)?;
                let idx = e.dtoh_u32_one(&tok_d)?;
                // trimmed draft vocab -> target token id (identity when no d2t map)
                let d = match &mtp.d2t { Some(map) => map[idx as usize], None => idx };
                if p_min > 0.0 {
                    let p_d = e.prob_of_token_device(&dl_d, &tok_d, d_vocab)?;
                    let p = e.dtoh(&p_d)?[0];
                    if p < p_min && j > 0 {
                        // keep at least 1 draft token (j==0 always drafts — a 0-draft round would
                        // degenerate to plain decode plus overhead).
                        break;
                    }
                }
                draft.push(d);
                e_tok = d;
                d_seed = h_nextn;
            }
            let k_round = draft.len();

            // --- 2. VERIFY: one batched target forward. With a pending bonus, it rides as col 0
            //         (committing its KV/recur inside the SAME weight read); drafts follow. ---
            let verify_tokens: Vec<u32> = match pending {
                Some(b) => { let mut v = Vec::with_capacity(k_round + 1); v.push(b); v.extend_from_slice(&draft); v }
                None => draft.clone(),
            };
            let base = if pending.is_some() { 1 } else { 0 };
            let (tlogits, vh_seed) = self.decode_step_t_h(e, &verify_tokens, pos, &mut cache)?;

            // --- 3. GREEDY ACCEPT (walk prefix, stop at first mismatch) ---
            // t_pred[j] = target's greedy prediction for the slot after draft[j-1] (j>=1) or after
            // last_token (j==0). With a pending bonus, col 0 IS the prediction after last_token
            // (== the bonus), so every index shifts by `base` and last_logits is unused.
            let t_pred = |j: usize| -> u32 {
                if j == 0 && base == 0 { argmax(&last_logits) as u32 }
                else { argmax(&tlogits[(base + j - 1) * n_vocab..(base + j) * n_vocab]) as u32 }
            };
            let mut n_acc = 0usize;
            for j in 0..k_round {
                if t_pred(j) == draft[j] { n_acc += 1; } else { break; }
            }
            // bonus = target's own token at the first non-accepted slot. n_acc in 0..=k; t_pred is
            // defined for j in 0..=k (j==0 -> last_logits, j>=1 -> tlogits col j-1, last col = k-1).
            let bonus = t_pred(n_acc);
            total_drafted += k_round;
            total_accepted += n_acc;

            if debug_spec {
                eprintln!("[R{round}] pos={pos} out_len={} last_tok={last_token} draft={draft:?} n_acc={n_acc} bonus={bonus} t_pred0={}", out.len(), t_pred(0));
            }

            // --- 4. COMMIT: draft[0..n_acc] then bonus (n_acc + 1 tokens) ---
            for j in 0..n_acc {
                if out.len() >= max_new { break; }
                out.push(draft[j]);
            }
            let bonus_emitted = out.len() < max_new;
            if bonus_emitted { out.push(bonus); }
            last_token = bonus;

            // --- 5. ROLLBACK + advance (§C) ---
            if n_acc == k_round {
                // FULL ACCEPT, BONUS FOLD: all verify columns (pending? + drafts) are committed in
                // cache; the NEW bonus stays PENDING for the next round's verify batch — NO extra
                // T=1 trunk pass. The next draft chain seeds from the MTP block's h_nextn at the
                // bonus position: one MTP-block pass (~1/33 trunk cost) replaces the trunk read.
                // last_logits is dead in the pending path (t_pred reads verify col 0).
                let (_dl_d, h_bonus_pseudo) = self.mtp_head_forward_dev(
                    e, mtp, bonus, &vh_seed, &mut scratch, pos + base + k_round)?;
                pending = Some(bonus);
                h_seed = h_bonus_pseudo;
                if debug_spec { eprintln!("  -> FULL ACCEPT (bonus pending)"); }
            } else {
                // PARTIAL ACCEPT: verify appended k draft columns but only n_acc are committed.
                // Restore EVERYTHING to the pre-round snapshot (KV truncate to pos + recur restore),
                // then replay the committed prefix draft[0..n_acc] ++ [bonus] as ONE batched
                // T=(n_acc+1) forward — single weight read, bit-identical to greedy (the verify-all-
                // columns path is the same math). This is the MTP profitability lever: replaying
                // n_acc+1 tokens in one decode_step_t_h reads every weight ONCE vs n_acc+1 times in
                // the old per-token loop (decode is bandwidth-bound at batch=1). KV+recur rebuilt.
                let _ = n_embd;
                cache.rollback(e, &snap, 0)?;   // accept_len=0: KV len = pos, recur = snapshot
                // Replay = pending bonus (if any) ++ accepted drafts ++ new bonus: commits the
                // pending token with a TRUE trunk hidden, so pseudo-seeding never survives a
                // rejection round.
                let mut replay: Vec<u32> = Vec::with_capacity(base + n_acc + 1);
                if let Some(b) = pending.take() { replay.push(b); }
                replay.extend_from_slice(&draft[0..n_acc]);
                replay.push(bonus);
                let (rl, rh) = self.decode_step_t_h(e, &replay, pos, &mut cache)?;
                // last_logits = last column's logits (predicts the token after `bonus`).
                last_logits = rl[(replay.len() - 1) * n_vocab..replay.len() * n_vocab].to_vec();
                h_seed = rh;
                if debug_spec { eprintln!("  -> PARTIAL(replay={replay:?}), next_pred={}", argmax(&last_logits)); }
            }
            round += 1;
        }

        out.truncate(max_new);
        Ok((out, total_drafted, total_accepted))
    }
}
