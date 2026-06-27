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
use crate::hybrid::{HybridModel, Mixer, FullAttnLayer, MtpHead};
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
    fn mtp_head_forward(&self, e: &Engine, mtp: &MtpHead, e_tok: u32, h_seed: &CudaSlice<f32>,
                        scratch: &mut MtpScratch, mtp_pos: usize)
                        -> Result<(Vec<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
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

        // op 12: draft_logits = (shared_head_head OR output) @ final
        let head = mtp.shared_head_head.as_ref().unwrap_or(&self.output);
        let logits = e.matmul(head, &final_h, 1)?;
        let host = e.dtoh(&logits)?;
        Ok((host, h_nextn))
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
                    // recurrent: cannot batch — run T sequential T=1 steps advancing the recur state
                    // exactly like decode (MTP-PLAN §D.3.3). Each step reads its own column of h.
                    let mut out = e.zeros(t * n_embd)?;
                    for col in 0..t {
                        // extract column `col` of h ([T,n_embd] token-major) as an owned [n_embd] buffer
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
        let t_kv = kvl.len;

        // causal attention of the T query rows over [0..t_kv) resident byte K/V. fa_prefill is causal:
        // query row r (absolute pos pos0+r) sees keys [0 .. (t_kv - t) + r]. With pos0 == t_kv - t
        // (the verify tokens are appended at the tail), the causal mask aligns: q row r attends
        // up to absolute key (t_kv - t) + r == pos0 + r. Correct.
        let k_view = e.view_u8(&kvl.k, t_kv * ktb);
        let v_view = e.view_u8(&kvl.v, t_kv * vtb);
        let mut attn = e.zeros(t * n_head * head_dim)?;
        e.fa_prefill_view(&q, &k_view, &v_view, &mut attn, head_dim, n_head, n_head_kv, t, t_kv, scale, true, ktb, vtb)?;

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

        while out.len() < max_new {
            let pos = cache.pos;            // #tokens already committed in the cache
            let snap = cache.snapshot(e)?;  // §C: snapshot BEFORE draft+verify

            // --- 1. DRAFT k tokens with the NextN head (autoregressive, T=1 each) ---
            scratch.reset();
            let mut draft: Vec<u32> = Vec::with_capacity(k);
            let mut e_tok = last_token;
            let mut d_seed = e.clone_dtod(&h_seed)?;
            for j in 0..k {
                let (dl, h_nextn) = self.mtp_head_forward(e, mtp, e_tok, &d_seed, &mut scratch, pos + j)?;
                let d = argmax(&dl) as u32;
                draft.push(d);
                e_tok = d;
                d_seed = h_nextn;
            }

            // --- 2. VERIFY: one batched target forward over draft[0..k] at positions pos..pos+k-1
            //         (T=k). Column j follows draft[j], predicting slot pos+1+j. ---
            let tlogits = self.decode_step_t(e, &draft, pos, &mut cache)?;

            // --- 3. GREEDY ACCEPT (walk prefix, stop at first mismatch) ---
            // t_pred[j] = target's greedy prediction for slot pos+j:
            //   t_pred[0] = argmax(last_logits)           (predicts the token after last_token)
            //   t_pred[j] = argmax(tlogits col j-1)        (predicts the token after draft[j-1])
            let t_pred = |j: usize| -> u32 {
                if j == 0 { argmax(&last_logits) as u32 }
                else { argmax(&tlogits[(j - 1) * n_vocab..j * n_vocab]) as u32 }
            };
            let mut n_acc = 0usize;
            for j in 0..k {
                if t_pred(j) == draft[j] { n_acc += 1; } else { break; }
            }
            // bonus = target's own token at the first non-accepted slot. n_acc in 0..=k; t_pred is
            // defined for j in 0..=k (j==0 -> last_logits, j>=1 -> tlogits col j-1, last col = k-1).
            let bonus = t_pred(n_acc);
            total_drafted += k;
            total_accepted += n_acc;

            // --- 4. COMMIT: draft[0..n_acc] then bonus (n_acc + 1 tokens) ---
            for j in 0..n_acc {
                if out.len() >= max_new { break; }
                out.push(draft[j]);
            }
            let bonus_emitted = out.len() < max_new;
            if bonus_emitted { out.push(bonus); }
            last_token = bonus;

            // --- 5. ROLLBACK + advance to exactly pos + n_acc + 1 committed tokens (§C) ---
            if n_acc == k {
                // FULL ACCEPT: all k draft columns are committed; KV+recur already correct.
                // cache.pos == pos + k. Feed `bonus` (T=1) to commit its KV/recur and obtain the
                // next-round last_logits + h_seed. (No rollback — Reader-3 full-accept path.)
                let (l, h) = self.decode_step_h(e, bonus, &mut cache)?;
                last_logits = l; h_seed = h;
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
                let mut replay: Vec<u32> = draft[0..n_acc].to_vec();
                replay.push(bonus);
                let (rl, rh) = self.decode_step_t_h(e, &replay, pos, &mut cache)?;
                // last_logits = last column's logits (predicts the token after `bonus`).
                last_logits = rl[(replay.len() - 1) * n_vocab..replay.len() * n_vocab].to_vec();
                h_seed = rh;
            }
        }

        out.truncate(max_new);
        Ok((out, total_drafted, total_accepted))
    }
}
