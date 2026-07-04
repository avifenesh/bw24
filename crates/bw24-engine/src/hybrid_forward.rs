//! Hybrid forward pass (Stage-1, f32, prefill, single sequence). Per layer dispatches to a
//! linear-attention (Gated DeltaNet) or full-attention mixer, then SwiGLU FFN. Matches
//! llama.cpp src/models/qwen35.cpp node-for-node.

use cudarc::driver::CudaSlice;
use bw24_gguf::config::ModelConfig;
use crate::Engine;
use crate::cache::Cache;
use crate::hybrid::{HybridModel, Mixer, FullAttnLayer, LinearAttnLayer, MoeWeights};

/// STAGE-2 GROUPED DECODE gate (BW24_MOE_GDEC, default ON; `=0` restores the sequential
/// per-expert launch chain). See `moe_gdec_token`.
fn gdec_enabled() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| std::env::var("BW24_MOE_GDEC").map(|v| v != "0").unwrap_or(true))
}

/// Minimum prompt length for the BATCHED cache prime (`prime_cache`). Below this the tokenwise
/// decode loop wins anyway (the batched path's GEMM dispatch needs m>=16, and the stateful conv
/// kernel needs T >= d_conv-1). Callers: generate / generate_spec.
pub const PRIME_MIN_T: usize = 16;

impl HybridModel {
    /// Prefill forward over `tokens`; returns logits [T, n_vocab] (host f32).
    pub fn forward(&self, e: &Engine, tokens: &[u32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let n_embd = cfg.n_embd as usize;
        let t = tokens.len();
        let eps = cfg.rms_eps;
        let pos: Vec<i32> = (0..t as i32).collect();
        let pos_d = e.htod_i32(&pos)?;

        let mut x = self.embed(e, tokens)?;   // [T, n_embd]

        for (il, layer) in self.layers.iter().enumerate() {
            // attn_norm
            let mut h = e.zeros(t * n_embd)?;
            e.rms_norm(&x, layer.attn_norm.float_data(), &mut h, n_embd, t, eps)?;

            let mixed = match &layer.mixer {
                Mixer::Full(fa) => self.full_attn(e, fa, &h, &pos_d, t)?,
                Mixer::Linear(la) => self.linear_attn(e, la, &h, t)?,
            };

            // residual 1
            let mut x1 = e.zeros(t * n_embd)?;
            e.add(&x, &mixed, &mut x1, t * n_embd)?;

            // pre-FFN norm (post_attention_norm), FFN (Dense or MoE), residual 2
            let mut z = e.zeros(t * n_embd)?;
            e.rms_norm(&x1, layer.post_attn_norm.float_data(), &mut z, n_embd, t, eps)?;
            let ffn_out = match &layer.ffn {
                crate::hybrid::Ffn::Dense { ffn_gate, ffn_up, ffn_down } => {
                    let n_ff = ffn_gate.out_features();
                    let gate = e.matmul(ffn_gate, &z, t)?;
                    let up = e.matmul(ffn_up, &z, t)?;
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

        let mut hn = e.zeros(t * n_embd)?;
        e.rms_norm(&x, self.output_norm.float_data(), &mut hn, n_embd, t, eps)?;
        let logits = e.matmul(&self.output, &hn, t)?;
        Ok(e.dtoh(&logits)?)
    }

    /// Prefill that returns ONLY the last token's logits — the common case (greedy/sample needs
    /// just the final position to start decode). Runs the trunk over all T, then the lm_head
    /// (output.weight, the largest matrix — 248320 rows) on the LAST hidden row ONLY, not all T.
    /// On a 512-token prompt this turns a [512,248320] GEMM into [1,248320] — the dominant prefill
    /// cost (nsys: ~99ms when done for all T). Bit-identical last-row logits to forward()[last].
    pub fn forward_last(&self, e: &Engine, tokens: &[u32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let n_embd = cfg.n_embd as usize;
        let t = tokens.len();
        let eps = cfg.rms_eps;
        let pos: Vec<i32> = (0..t as i32).collect();
        let pos_d = e.htod_i32(&pos)?;

        let mut x = self.embed(e, tokens)?;   // [T, n_embd]
        for (il, layer) in self.layers.iter().enumerate() {
            let mut h = e.zeros(t * n_embd)?;
            e.rms_norm(&x, layer.attn_norm.float_data(), &mut h, n_embd, t, eps)?;
            let mixed = match &layer.mixer {
                Mixer::Full(fa) => self.full_attn(e, fa, &h, &pos_d, t)?,
                Mixer::Linear(la) => self.linear_attn(e, la, &h, t)?,
            };
            let mut x1 = e.zeros(t * n_embd)?;
            e.add(&x, &mixed, &mut x1, t * n_embd)?;
            let mut z = e.zeros(t * n_embd)?;
            e.rms_norm(&x1, layer.post_attn_norm.float_data(), &mut z, n_embd, t, eps)?;
            let ffn_out = match &layer.ffn {
                crate::hybrid::Ffn::Dense { ffn_gate, ffn_up, ffn_down } => {
                    let n_ff = ffn_gate.out_features();
                    let gate = e.matmul(ffn_gate, &z, t)?;
                    let up = e.matmul(ffn_up, &z, t)?;
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
        // norm over all T, then slice the LAST row and run lm_head on that single row.
        let mut hn = e.zeros(t * n_embd)?;
        e.rms_norm(&x, self.output_norm.float_data(), &mut hn, n_embd, t, eps)?;
        let last = e.view(&hn, t * n_embd);            // [T, n_embd]
        let last_row = last.slice((t - 1) * n_embd..t * n_embd);  // [1, n_embd]
        let mut hlast = e.zeros(n_embd)?;
        e.copy_view_into(&mut hlast, 0, &last_row, n_embd)?;
        let logits = e.matmul(&self.output, &hlast, 1)?;   // [1, n_vocab] — lm_head on ONE row
        Ok(e.dtoh(&logits)?)
    }

    /// BATCHED PROMPT PRIME (the measured #1 e2e gap, e2e-image-1): `forward_last`'s batched
    /// prefill body EXTENDED to leave a DECODE-READY cache behind — vs the tokenwise prime's
    /// ~102/38 tok/s (9B/27B) decode_step loop, this runs the whole prompt at prefill throughput.
    ///   (a) full-attn layers append their T post-RoPE K/V rows into `cache.kv[il]` via the SAME
    ///       per-row quantize kernel as the decode append (bit-identical cache bytes per row);
    ///   (b) linear layers run STATEFULLY from the cache's current recurrent state (zero at a
    ///       fresh prime): carried-ring conv (ssm_conv1d_tm_state) + ONE gdn_scan(state_in,
    ///       state_out) whose internal sequential t-loop equals T chained T=1 steps — but with
    ///       the NORMAL prefill matmul dispatch (GEMM at m>=16), NOT the decode-exact MMVQ the
    ///       spec verify uses (prime is a prefill-regime pass; the run-gen prefill==decode
    ///       argmax gate is the accuracy authority, exactly as for forward_last);
    ///   (c) `cache.pos`/KV len/len_d advance by T.
    /// Returns (last-row logits host, h_seed = last-row PRE-output_norm hidden [n_embd],
    /// hiddens = the full pre-output_norm hidden stack [T, n_embd] — generate_spec's prompt_h).
    /// FRESH-PROMPT ONLY (cache.pos == 0): the fa_prefill tiles attend within `tokens` alone.
    /// forward_last itself stays untouched (kernel-check / run-gen gate on it).
    pub fn prime_cache(&self, e: &Engine, tokens: &[u32], cache: &mut Cache)
                       -> Result<(Vec<f32>, CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let n_embd = cfg.n_embd as usize;
        let t = tokens.len();
        let eps = cfg.rms_eps;
        assert!(cache.pos == 0, "prime_cache is a fresh-prompt prime (cache.pos must be 0)");
        assert!(t >= PRIME_MIN_T, "prime_cache needs T >= {PRIME_MIN_T} (caller gates)");
        assert!(t <= cache.max_ctx, "prime_cache: prompt exceeds cache max_ctx");
        let pos: Vec<i32> = (0..t as i32).collect();
        let pos_d = e.htod_i32(&pos)?;

        let mut x = self.embed(e, tokens)?;   // [T, n_embd]
        for (il, layer) in self.layers.iter().enumerate() {
            let mut h = e.zeros(t * n_embd)?;
            e.rms_norm(&x, layer.attn_norm.float_data(), &mut h, n_embd, t, eps)?;
            let mixed = match &layer.mixer {
                Mixer::Full(fa) => self.full_attn_prime(e, fa, &h, &pos_d, t, cache, il)?,
                Mixer::Linear(la) => self.linear_attn_prime(e, la, &h, t, cache, il)?,
            };
            let mut x1 = e.zeros(t * n_embd)?;
            e.add(&x, &mixed, &mut x1, t * n_embd)?;
            let mut z = e.zeros(t * n_embd)?;
            e.rms_norm(&x1, layer.post_attn_norm.float_data(), &mut z, n_embd, t, eps)?;
            let ffn_out = match &layer.ffn {
                crate::hybrid::Ffn::Dense { ffn_gate, ffn_up, ffn_down } => {
                    let n_ff = ffn_gate.out_features();
                    let gate = e.matmul(ffn_gate, &z, t)?;
                    let up = e.matmul(ffn_up, &z, t)?;
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

        // h_seed = LAST row of x BEFORE output_norm (MTP-PLAN §A seed convention).
        let mut h_seed = e.zeros(n_embd)?;
        e.copy_view_into(&mut h_seed, 0, &x.slice((t - 1) * n_embd..t * n_embd), n_embd)?;
        // last-row logits, exactly like forward_last (norm all T — per-row op — then lm_head on 1 row).
        let mut hn = e.zeros(t * n_embd)?;
        e.rms_norm(&x, self.output_norm.float_data(), &mut hn, n_embd, t, eps)?;
        let last = e.view(&hn, t * n_embd);
        let last_row = last.slice((t - 1) * n_embd..t * n_embd);
        let mut hlast = e.zeros(n_embd)?;
        e.copy_view_into(&mut hlast, 0, &last_row, n_embd)?;
        let logits = e.matmul(&self.output, &hlast, 1)?;
        cache.pos += t;
        Ok((e.dtoh(&logits)?, h_seed, x))
    }

    /// `full_attn` (batched prefill mixer) + the cache side-effect: append the T post-RoPE K/V
    /// rows into the resident quantized KV cache (q8_0 K / q5_1 V) and advance len/len_d. Row
    /// bytes are BIT-IDENTICAL to the decode append (same per-warp quant kernel per row; the
    /// batched `append_kv_quantized_rows` runs that exact warp math on a (block, token) grid).
    /// The attention itself is unchanged prefill math (fa_prefill over the f32 K/V).
    fn full_attn_prime(&self, e: &Engine, fa: &FullAttnLayer, h: &CudaSlice<f32>,
                       pos_d: &CudaSlice<i32>, t: usize, cache: &mut Cache, il: usize)
                       -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let n_head = cfg.n_head as usize;
        let n_head_kv = cfg.n_head_kv as usize;
        let head_dim = cfg.head_dim_k as usize;
        let eps = cfg.rms_eps;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let qf = e.matmul(&fa.wq, h, t)?;
        let mut q = e.zeros(t * n_head * head_dim)?;
        let mut gate = e.zeros(t * n_head * head_dim)?;
        e.q_gate_split(&qf, &mut q, &mut gate, head_dim, n_head, t)?;
        let mut k = e.matmul(&fa.wk, h, t)?;
        let v = e.matmul(&fa.wv, h, t)?;

        let mut qn = e.zeros(t * n_head * head_dim)?;
        e.rms_norm(&q, fa.q_norm.float_data(), &mut qn, head_dim, n_head * t, eps)?;
        q = qn;
        let mut kn = e.zeros(t * n_head_kv * head_dim)?;
        e.rms_norm(&k, fa.k_norm.float_data(), &mut kn, head_dim, n_head_kv * t, eps)?;
        k = kn;
        let rope_dims = cfg.rope_dim_count as usize;
        e.rope_neox(&mut q, pos_d, head_dim, rope_dims, n_head, t, cfg.rope_freq_base, 1.0)?;
        e.rope_neox(&mut k, pos_d, head_dim, rope_dims, n_head_kv, t, cfg.rope_freq_base, 1.0)?;

        // CACHE SIDE-EFFECT: append the T post-rope K/V token rows (token-major [T, kv_dim] ==
        // the cache row layout) quantized into cache.kv[il], then advance len + device len_d.
        {
            let kvl = cache.kv[il].as_mut().unwrap();
            assert!(kvl.len + t <= cache.max_ctx, "prime_cache: KV overflow");
            e.append_kv_quantized_rows(&k, &v, &mut kvl.k, &mut kvl.v, kvl.len, t,
                                       kvl.kv_dim_k, kvl.kv_dim_v, kvl.k_tok_bytes, kvl.v_tok_bytes)?;
            kvl.len += t;
            let new_len = kvl.len as i32;
            e.set_i32_one(&mut kvl.len_d, new_len)?;
        }

        // batched prefill attention (unchanged forward_last math).
        let mut attn = e.zeros(t * n_head * head_dim)?;
        if std::env::var("BW24_NOFA").is_ok() {
            e.sdpa_naive(&q, &k, &v, &mut attn, head_dim, n_head, n_head_kv, t, t, scale, true)?;
        } else {
            e.fa_prefill(&q, &k, &v, &mut attn, head_dim, n_head, n_head_kv, t, t, scale, true)?;
        }

        let mut gsig = e.zeros(t * n_head * head_dim)?;
        e.sigmoid(&gate, &mut gsig, t * n_head * head_dim)?;
        let mut attn_g = e.zeros(t * n_head * head_dim)?;
        e.mul(&attn, &gsig, &mut attn_g, t * n_head * head_dim)?;
        Ok(e.matmul(&fa.wo, &attn_g, t)?)
    }

    /// STATEFUL batched linear-attention prime: `linear_attn`'s prefill-dispatch pass (normal
    /// `e.matmul` — GEMM at m>=16 — plus the prefill repack/L2/glog kernels) but with the state
    /// carried THROUGH the cache like the spec verify does: carried-ring conv
    /// (ssm_conv1d_tm_state writes the final ring back) + ONE gdn_scan from cache.recur[il]'s
    /// current state (zero at a fresh prime) whose final state ping-pongs back into the cache.
    /// Wiring mirrors `linear_attn_verify_t` (spec.rs); dispatch mirrors `linear_attn` (prefill).
    fn linear_attn_prime(&self, e: &Engine, la: &LinearAttnLayer, h: &CudaSlice<f32>, t: usize,
                         cache: &mut Cache, il: usize)
                         -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let ssm = cfg.ssm.as_ref().unwrap();
        let d_state = ssm.state_size as usize;       // 128
        let num_k = ssm.group_count as usize;        // 16
        let num_v = ssm.time_step_rank as usize;     // 32
        let d_conv = ssm.conv_kernel as usize;       // 4
        let key_dim = d_state * num_k;               // 2048
        let value_dim = d_state * num_v;             // 4096
        let conv_dim = key_dim * 2 + value_dim;      // 8192
        let eps = cfg.rms_eps;
        let scale = 1.0 / (d_state as f32).sqrt();
        debug_assert!(t >= d_conv - 1, "stateful conv needs T >= pad (PRIME_MIN_T gates)");

        // NORMAL prefill dispatch (GEMM at m>=16) — same as linear_attn/forward_last.
        let qkv_mixed = e.matmul(&la.wqkv, h, t)?;       // [T, conv_dim] token-major
        let z = e.matmul(&la.wqkv_gate, h, t)?;          // [T, value_dim]
        let beta_raw = e.matmul(&la.ssm_beta, h, t)?;    // [T, num_v]
        let alpha = e.matmul(&la.ssm_alpha, h, t)?;      // [T, num_v]

        // conv with CARRIED ring state + ring roll (state read + final-window write-back).
        let rl = cache.recur[il].as_mut().unwrap();
        let mut conv_out = e.uninit(conv_dim * t)?;      // [conv_dim, T] channel-major, SiLU
        e.ssm_conv1d_tm_state(&qkv_mixed, &mut rl.conv_state, la.ssm_conv1d.float_data(),
                              &mut conv_out, conv_dim, t, d_conv)?;

        // GDN prep via the PREFILL kernels (repack + 256-thread l2_norm + sigmoid + glog) —
        // the same kernels forward_last's fused path reproduces value-for-value.
        let mut q_g = e.uninit(d_state * num_v * t)?;
        let mut k_g = e.uninit(d_state * num_v * t)?;
        let mut v_g = e.uninit(d_state * num_v * t)?;
        e.qkv_to_gdn_repack(&conv_out, &mut q_g, &mut k_g, &mut v_g, d_state, num_v, num_k, key_dim, t)?;
        let mut q_l2 = e.zeros(d_state * num_v * t)?;
        e.l2_norm(&q_g, &mut q_l2, d_state, num_v * t, eps)?;
        let mut k_l2 = e.zeros(d_state * num_v * t)?;
        e.l2_norm(&k_g, &mut k_l2, d_state, num_v * t, eps)?;
        let mut beta = e.zeros(t * num_v)?;
        e.sigmoid(&beta_raw, &mut beta, t * num_v)?;
        let mut g_log = e.zeros(t * num_v)?;
        e.gdn_glog(&alpha, la.ssm_dt.float_data(), la.ssm_a.float_data(), &mut g_log, num_v, t)?;

        // ONE gdn_scan over T from the cache's CURRENT state (zero at fresh prime); the final
        // state lands in the spare buffer and ping-pongs back (stable resident pointers, the
        // decode-determinism discipline from linear_attn_decode_inner). A4: `gdn_scan_prefill`
        // dispatches the chunked WY form under BW24_GDN_CHUNKED (prefill-only seam; decode +
        // verify keep the sequential kernel).
        let mut o = e.zeros(d_state * num_v * t)?;
        {
            let crate::cache::RecurLayer { ssm_state, ssm_state_alt, .. } = rl;
            e.gdn_scan_prefill(&q_l2, &k_l2, &v_g, &g_log, &beta, ssm_state, ssm_state_alt, &mut o, num_v, t, scale)?;
        }
        std::mem::swap(&mut rl.ssm_state, &mut rl.ssm_state_alt);

        // gated RMSNorm + out projection (prefill dispatch).
        let mut gn = e.zeros(d_state * num_v * t)?;
        e.gated_rmsnorm(&o, la.ssm_norm.float_data(), &z, &mut gn, d_state, num_v * t, eps)?;
        Ok(e.matmul(&la.ssm_out, &gn, t)?)
    }

    /// Full-attention mixer with QK-norm, partial RoPE, sigmoid output gate (qwen35 :257-336).
    fn full_attn(&self, e: &Engine, fa: &FullAttnLayer, h: &CudaSlice<f32>, pos_d: &CudaSlice<i32>, t: usize)
                 -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let _n_embd = cfg.n_embd as usize;
        let n_head = cfg.n_head as usize;
        let n_head_kv = cfg.n_head_kv as usize;
        let head_dim = cfg.head_dim_k as usize;
        let eps = cfg.rms_eps;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // wq output = head_dim*2*n_head (fused [q|gate] per head, stride 2*head_dim).
        let qf = e.matmul(&fa.wq, h, t)?;
        // split per head ON-DEVICE: q = [head_dim] at offset 0 within each 2*head_dim block; gate at
        // offset head_dim. -> q,gate [head_dim, n_head, T]. No dtoh/host-loop/htod.
        let mut q = e.zeros(t * n_head * head_dim)?;
        let mut gate = e.zeros(t * n_head * head_dim)?;
        e.q_gate_split(&qf, &mut q, &mut gate, head_dim, n_head, t)?;
        let mut k = e.matmul(&fa.wk, h, t)?;
        let v = e.matmul(&fa.wv, h, t)?;

        // QK-norm (per head_dim row), then partial RoPE.
        let mut qn = e.zeros(t * n_head * head_dim)?;
        e.rms_norm(&q, fa.q_norm.float_data(), &mut qn, head_dim, n_head * t, eps)?;
        q = qn;
        let mut kn = e.zeros(t * n_head_kv * head_dim)?;
        e.rms_norm(&k, fa.k_norm.float_data(), &mut kn, head_dim, n_head_kv * t, eps)?;
        k = kn;
        let rope_dims = cfg.rope_dim_count as usize;
        e.rope_neox(&mut q, pos_d, head_dim, rope_dims, n_head, t, cfg.rope_freq_base, 1.0)?;
        e.rope_neox(&mut k, pos_d, head_dim, rope_dims, n_head_kv, t, cfg.rope_freq_base, 1.0)?;

        // SDPA
        let mut attn = e.zeros(t * n_head * head_dim)?;
        // hand-written FlashAttention prefill (head_dim 256). BW24_NOFA falls back to naive sdpa.
        if std::env::var("BW24_NOFA").is_ok() {
            e.sdpa_naive(&q, &k, &v, &mut attn, head_dim, n_head, n_head_kv, t, t, scale, true)?;
        } else {
            e.fa_prefill(&q, &k, &v, &mut attn, head_dim, n_head, n_head_kv, t, t, scale, true)?;
        }

        // output gate: attn * sigmoid(gate)
        let mut gsig = e.zeros(t * n_head * head_dim)?;
        e.sigmoid(&gate, &mut gsig, t * n_head * head_dim)?;
        let mut attn_g = e.zeros(t * n_head * head_dim)?;
        // reuse silu_mul? no — need plain mul. do it via a tiny host path is wasteful; use mul kernel.
        e.mul(&attn, &gsig, &mut attn_g, t * n_head * head_dim)?;

        // o projection
        let o = e.matmul(&fa.wo, &attn_g, t)?;
        Ok(o)
    }

    /// Linear-attention (Gated DeltaNet) mixer (qwen35 :338-470).
    fn linear_attn(&self, e: &Engine, la: &LinearAttnLayer, h: &CudaSlice<f32>, t: usize)
                   -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let _n_embd = cfg.n_embd as usize;
        let ssm = cfg.ssm.as_ref().unwrap();
        let d_state = ssm.state_size as usize;       // 128
        let num_k = ssm.group_count as usize;        // 16
        let num_v = ssm.time_step_rank as usize;     // 32
        let d_conv = ssm.conv_kernel as usize;       // 4
        let head_k = d_state; let head_v = d_state;
        let key_dim = head_k * num_k;                // 2048
        let value_dim = head_v * num_v;              // 4096
        let conv_dim = key_dim * 2 + value_dim;      // 8192
        let eps = cfg.rms_eps;
        let scale = 1.0 / (d_state as f32).sqrt();

        // projections
        let qkv_mixed = e.matmul(&la.wqkv, h, t)?;       // [T, conv_dim] token-major
        let z = e.matmul(&la.wqkv_gate, h, t)?;          // [T, value_dim]
        let beta_raw = e.matmul(&la.ssm_beta, h, t)?;    // [T, num_v]
        let alpha = e.matmul(&la.ssm_alpha, h, t)?;      // [T, num_v]

        // conv + GDN repack, FUSED (2026-07-03): ssm_conv1d_gdn reads qkv_mixed [T, conv_dim]
        // token-major DIRECTLY (causal window rows t-pad..t, rows<0 = zero prefill state), applies
        // the 8-tap conv + SiLU, and scatters straight into the GDN [d_state, num_v, T] q/k/v
        // layout with the modulo head-repeat. Replaces transpose + zeros + conv_left_pad +
        // ssm_conv1d + qkv_to_gdn_repack (5 launches, conv_in/conv_out scratch + a 16MB@T=512
        // round-trip). BIT-IDENTICAL accumulation and scatter mapping.
        let _ = (head_k, head_v);
        let mut q_g = e.uninit(d_state * num_v * t)?;
        let mut k_g = e.uninit(d_state * num_v * t)?;
        let mut v_g = e.uninit(d_state * num_v * t)?;
        e.ssm_conv1d_gdn(&qkv_mixed, la.ssm_conv1d.float_data(), &mut q_g, &mut k_g, &mut v_g,
                         conv_dim, t, d_conv, d_state, num_v, num_k, key_dim)?;
        // L2-norm q,k per (head_dim) row — rows are contiguous d_state in q_g.
        let mut q_l2 = e.zeros(d_state * num_v * t)?;
        e.l2_norm(&q_g, &mut q_l2, d_state, num_v * t, eps)?;
        let mut k_l2 = e.zeros(d_state * num_v * t)?;
        e.l2_norm(&k_g, &mut k_l2, d_state, num_v * t, eps)?;
        let v_gd = v_g;

        // beta = sigmoid(beta_raw) ; g_log = a * softplus(alpha + dt). Both need [num_v, T] layout
        // (g[t*num_v + h]). beta_raw/alpha are [T, num_v] token-major == that layout already.
        let mut beta = e.zeros(t * num_v)?;
        e.sigmoid(&beta_raw, &mut beta, t * num_v)?;
        // gdn_glog expects alpha [H,T] with alpha[t*H+h] and dt_bias/a [H] — matches token-major [T,num_v].
        let mut g_log = e.zeros(t * num_v)?;
        e.gdn_glog(&alpha, la.ssm_dt.float_data(), la.ssm_a.float_data(), &mut g_log, num_v, t)?;

        // GDN scan (A4: gdn_scan_prefill dispatches chunked WY under BW24_GDN_CHUNKED)
        let state_in = e.zeros(d_state * d_state * num_v)?;  // zero state (prefill)
        let mut state_out = e.zeros(d_state * d_state * num_v)?;
        let mut o = e.zeros(d_state * num_v * t)?;
        e.gdn_scan_prefill(&q_l2, &k_l2, &v_gd, &g_log, &beta, &state_in, &mut state_out, &mut o, num_v, t, scale)?;

        // gated RMSNorm: dst = RMSNorm(o, ssm_norm[head_v]) * silu(z). o is [d_state, num_v, T];
        // rows of head_v=d_state, nrows = num_v*T. z must match row layout: z is [T, value_dim] token-major
        // = [T, num_v*head_v]; per (t, vh) the head_v slice is contiguous -> rows align as (t*num_v+vh).
        // o rows are (t*num_v+vh) too. Good.
        let mut gn = e.zeros(d_state * num_v * t)?;
        e.gated_rmsnorm(&o, la.ssm_norm.float_data(), &z, &mut gn, d_state, num_v * t, eps)?;

        // ssm_out projection: gn is [d_state, num_v, T] = [value_dim, T] viewed token-major as [T, value_dim]?
        // gn layout: (t*num_v+vh)*d_state + i  == token t, then (vh,i) = channel vh*d_state+i. That's
        // token-major [T, value_dim]. linear wants [T, in=value_dim]. Good.
        let out = e.matmul(&la.ssm_out, &gn, t)?;
        Ok(out)
    }
}

impl HybridModel {
    /// MoE FFN (EDGE-1). z: [T, n_embd] (already post-attention-normed). Returns moe_out [T, n_embd].
    /// Node-for-node vs llama.cpp build_moe_ffn + qwen35moe::build_layer_ffn.
    ///
    /// `il` is the trunk layer index — the residency-cache key prefix (a gate-expert of layer 3 is a
    /// different 860160-byte block than the same expert of layer 7).
    ///
    /// Routing: host softmax+sort (default) OR the fused router kernel (BW24_FUSED_ROUTER).
    /// Dispatch: stage-every-token into 3 scratch slots (default) OR the SLRU residency cache
    /// (BW24_MOE_CACHE). The cache-HIT weight path is bit-identical to stage-every-token (§B.3).
    /// Convenience wrapper used by the hybrid trunk/MTP loops: pulls dims + max-block from `self`.
    pub(crate) fn moe_ffn_il(&self, e: &Engine, m: &MoeWeights, z: &CudaSlice<f32>, t: usize, il: u16)
               -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        Self::moe_ffn(e, m, z, t, &self.cfg, il, self.max_moe_block())
    }

    /// MoE FFN (EDGE-1), source-/model-agnostic. z: [T, n_embd] (already post-attention-normed).
    /// Returns moe_out [T, n_embd]. Node-for-node vs llama.cpp build_moe_ffn. Shared by the hybrid
    /// (qwen35moe, shared expert present) and the dense-attention MoE (OLMoE, no shared expert) paths;
    /// `cfg.moe` supplies the dims and the optional shexp fields decide whether step 3 runs.
    ///
    /// `il` is the layer index — the residency-cache key prefix. `max_block` is the global max expert
    /// stride (fixed cache-slot size); pass `self.max_moe_block()`.
    pub(crate) fn moe_ffn(e: &Engine, m: &MoeWeights, z: &CudaSlice<f32>, t: usize,
                          cfg: &ModelConfig, il: u16, max_block: usize)
               -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        // A2: Expert-grouped dispatch for prefill (T>1). BW24_MOE_GROUPED=1 routes here.
        if t > 1 && std::env::var("BW24_MOE_GROUPED").is_ok() {
            let grouped_out = Self::moe_ffn_grouped(e, m, z, t, cfg, il, max_block)?;
            // BW24_MOE_GATE: byte-identity comparison vs sequential path.
            if std::env::var("BW24_MOE_GATE").is_ok() {
                let seq_out = Self::moe_ffn_sequential(e, m, z, t, cfg, il, max_block)?;
                let g_host = e.dtoh(&grouped_out)?;
                let s_host = e.dtoh(&seq_out)?;
                let g_bytes: &[u8] = unsafe { std::slice::from_raw_parts(g_host.as_ptr() as *const u8, g_host.len() * 4) };
                let s_bytes: &[u8] = unsafe { std::slice::from_raw_parts(s_host.as_ptr() as *const u8, s_host.len() * 4) };
                if g_bytes == s_bytes {
                    if il == 0 { println!("moe-gate il={il} t={t} BYTE-IDENTICAL (first layer only printed)"); }
                } else {
                    let diffs = g_host.iter().zip(s_host.iter()).enumerate()
                        .filter(|(_, (a, b))| a != b).count();
                    let maxdiff = g_host.iter().zip(s_host.iter())
                        .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                    panic!("moe-gate il={il} t={t} MISMATCH: {diffs}/{} elems differ, maxdiff={maxdiff:.6e}", g_host.len());
                }
            }
            return Ok(grouped_out);
        }
        Self::moe_ffn_sequential(e, m, z, t, cfg, il, max_block)
    }

    /// Sequential (per-token) MoE FFN -- the original path. Factored out for the gate comparison.
    pub(crate) fn moe_ffn_sequential(e: &Engine, m: &MoeWeights, z: &CudaSlice<f32>, t: usize,
                                     cfg: &ModelConfig, il: u16, max_block: usize)
                   -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        use crate::moe_cache::{PROJ_GATE, PROJ_UP, PROJ_DOWN};
        let moe = cfg.moe.as_ref().unwrap();
        let n_embd = cfg.n_embd as usize;          // 2048 (gate/up in_f, down out_f)
        let n_expert = moe.expert_count as usize;  // 256
        let n_used = moe.expert_used_count as usize; // 8
        let n_ff_exp = moe.expert_ff_length as usize; // 512 (gate/up out_f, down in_f)

        // verify the HostExps dims match cfg (catches a wrong-file / transpose mixup)
        debug_assert_eq!(m.gate_exps.in_f, n_embd);
        debug_assert_eq!(m.gate_exps.out_f, n_ff_exp);
        debug_assert_eq!(m.down_exps.in_f, n_ff_exp);  // down is TRANSPOSED: in=512
        debug_assert_eq!(m.down_exps.out_f, n_embd);   //                     out=2048
        debug_assert_eq!(m.gate_exps.n_expert, n_expert);

        // 1. ROUTER: logits = ffn_gate_inp @ z  -> [T, 256]. gate_inp is F32 -> e.linear.
        let logits = e.matmul(&m.gate_inp, z, t)?;
        // Per-token (sel[8], w[8]) — either fused-router (device top-k) or host softmax+sort.
        let (sel_all, w_all) = Self::moe_route(e, &logits, t, n_expert, n_used)?;

        // BW24_MOE_STATS: per-layer routing stats for the A2 (expert-grouped prefill) baseline —
        // per-token expert-id entropy, active-expert coverage, tokens-per-expert group sizes.
        if t > 1 && std::env::var("BW24_MOE_STATS").is_ok() {
            let mut cnt = vec![0u32; n_expert];
            for &s in sel_all.iter() { cnt[s as usize] += 1; }
            let total = sel_all.len() as f64;
            let mut h = 0.0f64;
            let mut active = 0usize;
            for &c in &cnt { if c > 0 { active += 1; let p = c as f64 / total; h -= p * p.log2(); } }
            let maxc = cnt.iter().copied().max().unwrap_or(0);
            println!("moe-stats il={} t={} assignments={} active={}/{} entropy={:.3}b (max {:.3}b) mean_tok_per_active={:.2} max_tok_per_expert={}",
                     il, t, sel_all.len(), active, n_expert, h, (n_expert as f64).log2(), total / active.max(1) as f64, maxc);
        }

        let mut moe_out = e.zeros(t * n_embd)?;

        // GPU scratch: one slot per proj, big enough for ONE expert (default stage-every-token path).
        let g_len = m.gate_exps.expert_stride;  // 860160
        let u_len = m.up_exps.expert_stride;    // 860160
        let d_len = m.down_exps.expert_stride;  // 1114112
        let mut scratch_g = e.alloc_u8(g_len)?;
        let mut scratch_u = e.alloc_u8(u_len)?;
        let mut scratch_d = e.alloc_u8(d_len)?;
        // `max_block` (the GLOBAL max expert stride across all layers) is passed in — the cache slots
        // are FIXED-ADDRESS and must fit any layer's block (UD/dynamic GGUFs vary quant per layer).

        let use_cache = Engine::moe_cache_enabled();

        // EDGE-1 §C.2/C.3 (async H2D prefetch) — TODO, deliberately NOT wired into the hot loop.
        // The infrastructure is in place and validated: `HostExps` bytes are pinned under
        // BW24_MOE_CACHE (§C.1, true-DMA H2D, argmax-1178 confirmed), and `Engine` exposes a copy
        // stream + `stage_expert_async`/`compute_wait` (event-synced). What is left is the in-token
        // pipeline-by-one: while `qmatvec_view` for sel[j] runs on compute, prefetch the MISS blocks
        // of sel[j+1..] on the copy stream into double-buffered staging slots, then `compute_wait`
        // the event before each dependent GEMM. It is deferred because (a) the SLRU cache already
        // serves ~91% of blocks with ZERO H2D so prefetch only helps the ~9% miss tail, and (b) a
        // mis-ordered copy-stream event would silently corrupt the bit-identity gate. Shipping A+B
        // validated (per the build directive) rather than risk the argmax-1178 gate on the C tail.

        // 2. PER TOKEN: routed-expert loop. The ONE dispatch change vs Stage-1: a resident slot
        //    (cache HIT, no H2D) OR a staged slot (MISS) feeds the SAME unchanged qmatvec_view.
        for tok in 0..t {
            let sel = &sel_all[tok * n_used..(tok + 1) * n_used];
            let w = &w_all[tok * n_used..(tok + 1) * n_used];
            let zt = z.slice(tok * n_embd..(tok + 1) * n_embd);  // CudaView<f32>

            // STAGE-2 GROUPED DECODE (2026-07-04, BW24_MOE_GDEC default ON, =0 rollback): fold
            // this token's whole routed-expert FFN (8x gate/up/silu + 8x down/axpy = 40 launches)
            // into TWO launches via expert-pointer indirection over the fixed-address cache slots.
            // Fires only when ALL 3*n_used blocks are ALREADY cache-resident (pure-HIT: zero
            // memcpy, zero admission, so no slot can move under the collected pointers) — any
            // miss falls through to the sequential loop below, which admits as before. In steady
            // state on a fully-resident rig every token-layer takes the grouped path.
            // BIT-IDENTITY: each in-kernel dot reproduces qmatvec_f32's exact reduction; SiLU is
            // silu_mul_f32's exact expression; the down accumulation is a slot-ordered
            // __fmaf_rn chain == the sequential axpy_f32 chain (BW24_MOE_GDEC_GATE compares).
            if use_cache && n_used <= 8 && gdec_enabled()
                && Self::moe_gdec_token(e, m, il, max_block, &zt, sel, w,
                                        &mut moe_out, tok, n_embd, n_ff_exp, n_used)? {
                continue;
            }

            for (j, &ex) in sel.iter().enumerate() {
                let ex = ex as usize;
                if use_cache {
                    // SLRU residency cache: per-projection, dispatch the block (HIT => resident slot,
                    // MISS => staged slot) then run the SAME unchanged qmatvec_view from that slot.
                    // The bytes the kernel reads are byte-for-byte the same GGUF block (§B.3); the
                    // only difference between HIT and MISS is whether the memcpy_htod ran.
                    let gate = Self::moe_cached_gemm(e, il, PROJ_GATE, ex, m, max_block, &zt)?;
                    let up   = Self::moe_cached_gemm(e, il, PROJ_UP,   ex, m, max_block, &zt)?;
                    let mut act = e.zeros(n_ff_exp)?;
                    e.silu_mul(&gate, &up, &mut act, n_ff_exp)?;
                    let actv = act.slice(0..n_ff_exp);
                    let y = Self::moe_cached_gemm(e, il, PROJ_DOWN, ex, m, max_block, &actv)?;
                    let mut dst = moe_out.slice_mut(tok * n_embd..(tok + 1) * n_embd);
                    e.axpy_into(&y, w[j], &mut dst, n_embd)?;
                } else {
                    // Stage-1: stage gate/up/down for expert `ex` into the scratch slots, then GEMM.
                    e.stage_expert(m.gate_exps.expert_bytes(ex), &mut scratch_g, 0)?;
                    let gate = e.qmatvec_view(&scratch_g, 0..g_len, &zt, 1,
                        m.gate_exps.in_f, m.gate_exps.out_f, m.gate_exps.qtype, m.gate_exps.row_bytes)?;

                    e.stage_expert(m.up_exps.expert_bytes(ex), &mut scratch_u, 0)?;
                    let up = e.qmatvec_view(&scratch_u, 0..u_len, &zt, 1,
                        m.up_exps.in_f, m.up_exps.out_f, m.up_exps.qtype, m.up_exps.row_bytes)?;

                    let mut act = e.zeros(n_ff_exp)?;
                    e.silu_mul(&gate, &up, &mut act, n_ff_exp)?;

                    e.stage_expert(m.down_exps.expert_bytes(ex), &mut scratch_d, 0)?;
                    let actv = act.slice(0..n_ff_exp);
                    let y = e.qmatvec_view(&scratch_d, 0..d_len, &actv, 1,
                        m.down_exps.in_f, m.down_exps.out_f, m.down_exps.qtype, m.down_exps.row_bytes)?;

                    let mut dst = moe_out.slice_mut(tok * n_embd..(tok + 1) * n_embd);
                    e.axpy_into(&y, w[j], &mut dst, n_embd)?;
                }
            }
        }

        // 3. SHARED EXPERT (ALWAYS-ON, no routing) on the SAME z — qwen35moe only. OLMoE and most
        //    vanilla MoE have NO shared expert (the shexp tensors are absent / `None`); skip it then.
        if let (Some(gate_shexp), Some(up_shexp), Some(down_shexp), Some(gate_inp_shexp)) =
            (&m.gate_shexp, &m.up_shexp, &m.down_shexp, &m.gate_inp_shexp)
        {
            let n_ff_sh = gate_shexp.out_features();  // 512
            let sg_gate = e.matmul(gate_shexp, z, t)?;  // [T, 512]
            let sg_up = e.matmul(up_shexp, z, t)?;      // [T, 512]
            let mut sa = e.zeros(t * n_ff_sh)?;
            e.silu_mul(&sg_gate, &sg_up, &mut sa, t * n_ff_sh)?;
            let sh = e.matmul(down_shexp, &sa, t)?;     // [T, n_embd]

            // BUG-2 FIX: ffn_gate_inp_shexp is 1-D ne=[2048] -> out_f=1. Use e.linear(.., out_f=1),
            // NOT matmul/out_features (which would index ne[1] out of bounds).
            let gs = e.linear(z, gate_inp_shexp.float_data(), t, n_embd, 1)?;  // [T, 1]
            let mut g = e.zeros(t)?;
            e.sigmoid(&gs, &mut g, t)?;

            // moe_out[r, :] += sh[r, :] * g[r]   (per-token scalar gate)
            e.add_scaled_rows(&sh, &g, &mut moe_out, n_embd, t)?;
        }

        Ok(moe_out)
    }

    /// Stage-1 (no-cache) per-DECODE-TOKEN H2D bytes: every routed block re-staged every layer every
    /// token = sum over MoE layers of n_used * (gate+up+down expert_stride). The §D.4 PCIe baseline.
    pub fn stage1_h2d_per_token(&self) -> u64 {
        use crate::hybrid::Ffn;
        let n_used = self.cfg.moe.as_ref().map(|m| m.expert_used_count as u64).unwrap_or(0);
        let mut bytes = 0u64;
        for l in self.layers.iter() {
            if let Ffn::Moe(m) = &l.ffn {
                bytes += n_used * (m.gate_exps.expert_stride + m.up_exps.expert_stride
                                   + m.down_exps.expert_stride) as u64;
            }
        }
        bytes
    }

    /// Largest expert block (bytes) across ALL MoE layers + the MTP head — the fixed cache slot size.
    /// UD/dynamic GGUFs quant different layers differently, so `expert_stride` varies per layer; the
    /// residency cache slots are fixed-address and must fit any block, so size to this global max.
    pub(crate) fn max_moe_block(&self) -> usize {
        use crate::hybrid::Ffn;
        let mut mx = 0usize;
        let mut scan = |ffn: &Ffn| {
            if let Ffn::Moe(m) = ffn {
                mx = mx.max(m.gate_exps.expert_stride)
                       .max(m.up_exps.expert_stride)
                       .max(m.down_exps.expert_stride);
            }
        };
        for l in self.layers.iter() { scan(&l.ffn); }
        if let Some(mtp) = self.mtp.as_ref() { scan(&mtp.ffn); }
        mx
    }

    /// Routing for the whole batch: returns (sel [T*n_used] expert ids, w [T*n_used] renorm weights),
    /// token-major. Default = the Stage-1 host path (dtoh logits, softmax-256, stable DESC top-k,
    /// renorm). BW24_FUSED_ROUTER = the device kernel (§A) which reproduces the same numerics; we
    /// still dtoh the tiny [T,n_used] sel/w buffers (64 B/token vs 1 KB/token) — the host loop
    /// indexes HostExps.bytes on the CPU to choose the DMA source (§A.2 output staging).
    fn moe_route(e: &Engine, logits: &CudaSlice<f32>, t: usize, n_expert: usize, n_used: usize)
                 -> Result<(Vec<u32>, Vec<f32>), Box<dyn std::error::Error>> {
        if std::env::var("BW24_FUSED_ROUTER").is_ok() {
            let (sel_d, w_d) = e.moe_router_topk(logits, t, n_expert, n_used)?;
            let sel_i = e.dtoh_i32(&sel_d)?;
            let w = e.dtoh(&w_d)?;
            let sel: Vec<u32> = sel_i.iter().map(|&i| i as u32).collect();
            return Ok((sel, w));
        }
        // Host oracle (the §D bit-identity reference).
        let lg = e.dtoh(logits)?;   // [T*n_expert] host
        let mut sel = vec![0u32; t * n_used];
        let mut w_out = vec![0f32; t * n_used];
        for tok in 0..t {
            let row = &lg[tok * n_expert..(tok + 1) * n_expert];
            // softmax over ALL n_expert (stable: subtract max)
            let maxl = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut probs = vec![0f32; n_expert];
            let mut den = 0f32;
            for i in 0..n_expert { let x = (row[i] - maxl).exp(); probs[i] = x; den += x; }
            for p in probs.iter_mut() { *p /= den; }
            // stable DESC sort: prob DESC, ascending-index tiebreak.
            let mut idx: Vec<usize> = (0..n_expert).collect();
            idx.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]).then(a.cmp(&b)));
            let sl = &idx[..n_used];
            let mut wv: Vec<f32> = sl.iter().map(|&i| probs[i]).collect();
            let mut ws: f32 = wv.iter().sum();
            ws = ws.max(6.103515625e-5_f32);  // F16 smallest normal, clamp BEFORE divide
            for x in wv.iter_mut() { *x /= ws; }
            for j in 0..n_used {
                sel[tok * n_used + j] = sl[j] as u32;
                w_out[tok * n_used + j] = wv[j];
            }
        }
        Ok((sel, w_out))
    }

    /// STAGE-2 GROUPED DECODE (2026-07-04): run ONE token's whole routed-expert FFN in TWO
    /// launches when every one of its 3*n_used blocks is ALREADY cache-resident. Returns
    /// Ok(true) if the grouped path ran (caller skips the sequential loop for this token);
    /// Ok(false) on ANY miss (caller falls through — the sequential loop stages/admits as
    /// before, so the NEXT occurrence takes the grouped path). Pointer safety: cache slots are
    /// fixed-address for the engine's lifetime and the pure-HIT path performs no admission, so
    /// the collected raw pointers cannot move between collection and launch (single-threaded
    /// decode; the lock is held only for collection, launches are stream-ordered after any
    /// prior same-stream staging writes).
    #[allow(clippy::too_many_arguments)]
    fn moe_gdec_token(e: &Engine, m: &MoeWeights, il: u16, max_block: usize,
                      zt: &cudarc::driver::CudaView<f32>, sel: &[u32], w: &[f32],
                      moe_out: &mut CudaSlice<f32>, tok: usize,
                      n_embd: usize, n_ff_exp: usize, n_used: usize)
                      -> Result<bool, Box<dyn std::error::Error>> {
        use crate::moe_cache::{BlockId, PROJ_GATE, PROJ_UP, PROJ_DOWN};
        use cudarc::driver::DevicePtr;
        // One lock hold: residency-check all 3*n_used blocks, collect raw slot pointers.
        let ptrs = e.with_moe_cache(max_block, |c, eng| {
            let mut g = [0u64; 8];
            let mut u = [0u64; 8];
            let mut d = [0u64; 8];
            for (j, &ex) in sel.iter().enumerate() {
                let ex = ex as u16;
                let (Some(sg), Some(su), Some(sd)) = (c.resident(BlockId::new(il, PROJ_GATE, ex)),
                                                      c.resident(BlockId::new(il, PROJ_UP,   ex)),
                                                      c.resident(BlockId::new(il, PROJ_DOWN, ex)))
                else { return Ok(None); };
                let (pg, _e0) = c.slot(sg).device_ptr(eng.stream());
                let (pu, _e1) = c.slot(su).device_ptr(eng.stream());
                let (pd, _e2) = c.slot(sd).device_ptr(eng.stream());
                g[j] = pg as u64; u[j] = pu as u64; d[j] = pd as u64;
            }
            c.hits += (3 * n_used) as u64;   // instrumentation parity with dispatch()
            Ok(Some((g, u, d)))
        })?;
        let Some((g, u, d)) = ptrs else { return Ok(false) };
        let mut wv = [0f32; 8];
        wv[..n_used].copy_from_slice(w);
        // 2 launches: (gate+up+silu) x8, then (down + slot-ordered FMA accumulate) x8.
        let act = e.moe_gate_up_silu8(crate::WPtr8(g), crate::WPtr8(u), zt,
                                      n_embd, n_ff_exp, n_used,
                                      m.gate_exps.qtype, m.up_exps.qtype,
                                      m.gate_exps.row_bytes, m.up_exps.row_bytes)?;
        let mut dst = moe_out.slice_mut(tok * n_embd..(tok + 1) * n_embd);
        e.moe_down8_fma_into(crate::WPtr8(d), crate::F32x8(wv), &act, &mut dst,
                             n_ff_exp, n_embd, n_used,
                             m.down_exps.qtype, m.down_exps.row_bytes)?;
        Ok(true)
    }

    /// EDGE-1 §B.3: dispatch one expert projection through the SLRU cache, then run the SAME
    /// `qmatvec_view` from whichever slot it landed in (resident HIT or staged MISS). `x` is the
    /// sliced activation row. `proj` selects the gate/up/down HostExps tensor. Returns y = W_expert @ x.
    fn moe_cached_gemm(e: &Engine, il: u16, proj: u8, ex: usize, m: &MoeWeights,
                       max_block: usize, x: &cudarc::driver::CudaView<f32>)
                       -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        use crate::moe_cache::{BlockId, DispatchSlot, PROJ_GATE, PROJ_UP};
        let exps = match proj { PROJ_GATE => &m.gate_exps, PROJ_UP => &m.up_exps, _ => &m.down_exps };
        let len = exps.expert_stride;
        let id = BlockId::new(il, proj, ex as u16);
        let host_bytes = exps.expert_bytes(ex);
        // dispatch under the lock (lookup/admit/memcpy-issue), then resolve the slot and GEMM.
        e.with_moe_cache(max_block, |c, eng| {
            let slot = c.dispatch(id, host_bytes, eng)?;
            // resolve the device buffer for this slot; the GEMM is enqueued on the compute stream
            // (the same stream the memcpy was issued on, so ordering holds without extra sync).
            let buf = match slot { DispatchSlot::Resident(s) => c.slot(s), DispatchSlot::Staging(s) => c.buf(DispatchSlot::Staging(s)) };
            eng.qmatvec_view(buf, 0..len, x, 1, exps.in_f, exps.out_f, exps.qtype, exps.row_bytes)
        })
    }
}

// ================================================================================================
// A2: EXPERT-GROUPED MoE PREFILL (BW24_MOE_GROUPED=1). Resident-case prototype.
//
// Instead of the per-token loop (T * 8 experts * 3 projections = 12024 individual m=1 matvecs),
// this groups tokens by expert and runs ONE matmul per active expert per projection at m=m_e.
// On a 501-token prefill with ~170 active experts, that's ~510 matmuls (vs 12024).
//
// EXACTNESS: per-token accumulation across its 8 experts is reordered (grouped processes experts
// in expert-id order, not the router's top-k order). To preserve bit-identity with the sequential
// loop, we use an 8-SLOT scheme: expert outputs are scattered into slots keyed by the token's
// top-k position (0..7), then reduced in that fixed order. This makes the f32 addition order
// identical to the per-token loop regardless of expert processing order.
//
// Memory: T * 8 * n_embd * 4 = 501 * 8 * 2048 * 4 = ~32 MB (slot buffer). Fine on 96GB.
// ================================================================================================

impl HybridModel {
    /// A2 expert-grouped MoE FFN (prefill path, BW24_MOE_GROUPED=1). Same semantics as moe_ffn:
    /// z [T, n_embd] -> moe_out [T, n_embd]. BIT-IDENTICAL to moe_ffn when using the slot scheme.
    pub(crate) fn moe_ffn_grouped(e: &Engine, m: &MoeWeights, z: &CudaSlice<f32>, t: usize,
                                  cfg: &ModelConfig, il: u16, _max_block: usize)
                   -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let moe = cfg.moe.as_ref().unwrap();
        let n_embd = cfg.n_embd as usize;
        let n_expert = moe.expert_count as usize;
        let n_used = moe.expert_used_count as usize;
        let n_ff_exp = moe.expert_ff_length as usize;

        // 1. ROUTER (identical to moe_ffn).
        let logits = e.matmul(&m.gate_inp, z, t)?;
        let (sel_all, w_all) = Self::moe_route(e, &logits, t, n_expert, n_used)?;

        // 2. BUILD PER-EXPERT TOKEN LISTS (host-side grouping).
        // For each expert e, we need: which tokens use it, their positions in z, their top-k
        // slot index (for bit-identical accumulation), and their weights.
        struct ExpertGroup {
            tok_indices: Vec<i32>,   // indices into z rows (0..T-1)
            slot_indices: Vec<i32>,  // top-k slot (0..n_used-1) for that token-expert pair
            weights: Vec<f32>,       // renormalized weight for that token-expert pair
        }
        let mut groups: Vec<ExpertGroup> = (0..n_expert).map(|_| ExpertGroup {
            tok_indices: Vec::new(), slot_indices: Vec::new(), weights: Vec::new(),
        }).collect();

        for tok in 0..t {
            for j in 0..n_used {
                let ex = sel_all[tok * n_used + j] as usize;
                let w = w_all[tok * n_used + j];
                groups[ex].tok_indices.push(tok as i32);
                groups[ex].slot_indices.push(j as i32);
                groups[ex].weights.push(w);
            }
        }

        // 3. ALLOCATE SLOT BUFFER: [T, n_used, n_embd] f32, zero-initialized.
        // Each token's 8 expert contributions land in their respective slots.
        let mut slot_buf = e.zeros(t * n_used * n_embd)?;
        let mut wbuf = e.zeros(t * n_used)?;  // [T, n_used] weight buffer for FMA reduce

        // Expert weight dimensions (used in both cache and staging paths).
        let g_len = m.gate_exps.expert_stride;
        let u_len = m.up_exps.expert_stride;
        let d_len = m.down_exps.expert_stride;
        let use_cache = Engine::moe_cache_enabled();
        let max_block = _max_block;

        // GPU scratch for staging (only allocated when NOT using cache).
        let (mut scratch_g, mut scratch_u, mut scratch_d) = if !use_cache {
            (Some(e.alloc_u8(g_len)?), Some(e.alloc_u8(u_len)?), Some(e.alloc_u8(d_len)?))
        } else {
            (None, None, None)
        };

        // 4. PER ACTIVE EXPERT: gather, compute, scatter.
        // Processing ORDER: DEFAULT = DESCENDING m_e (biggest token batches first);
        // BW24_MOE_ORDER=id restores ascending expert id. Measured (rig5090, 2026-07-04): desc is
        // a first-forward win at partial cache capacity — the hot (big-m_e) experts are admitted
        // to the SLRU before the small-m_e tail can pollute it, so residency converges in ONE
        // forward instead of several: auto-cache T=501 126.9 -> 169.9 tok/s (1.34x), cap512
        // 119.6 -> 160.8 (and kills the rep-to-rep bimodal); wash (<2%) at cap64 pure-spill and
        // at long prompts where every expert stages regardless. Order is FREE to change without
        // breaking the byte-identity gate: the slot scheme pins each token's accumulation order
        // regardless of expert processing order (the whole point of the slots).
        let desc = std::env::var("BW24_MOE_ORDER").map(|v| v != "id").unwrap_or(true);
        let mut order: Vec<usize> =
            (0..n_expert).filter(|&ex| !groups[ex].tok_indices.is_empty()).collect();
        if desc {
            order.sort_by(|&a, &b| groups[b].tok_indices.len()
                .cmp(&groups[a].tok_indices.len()).then(a.cmp(&b)));
        }
        let mut m_dist: Vec<usize> = Vec::new();  // for stats
        for ex in order {
            let grp = &groups[ex];
            let m_e = grp.tok_indices.len();
            m_dist.push(m_e);

            // Upload index/weight arrays to device.
            let tok_idx_d = e.htod_i32(&grp.tok_indices)?;
            let slot_idx_d = e.htod_i32(&grp.slot_indices)?;
            let weight_d = e.htod(&grp.weights)?;

            // GATHER: collect m_e activation rows from z into a contiguous buffer.
            let mut gathered = e.zeros(m_e * n_embd)?;
            e.gather_rows(z, &tok_idx_d, &mut gathered, n_embd, m_e)?;
            let gv = gathered.slice(0..m_e * n_embd);

            // Compute gate/up/down matmuls -- two paths: cache-resident or host-staged.
            let y = if use_cache {
                use crate::moe_cache::{BlockId, PROJ_GATE, PROJ_UP, PROJ_DOWN};
                // CACHE PATH: dispatch through MOE cache, get device-resident buffer, GEMM at m=m_e.
                let gate = e.with_moe_cache(max_block, |c, eng| {
                    let id = BlockId::new(il, PROJ_GATE, ex as u16);
                    let slot = c.dispatch(id, m.gate_exps.expert_bytes(ex), eng)?;
                    let buf = c.buf(slot);
                    eng.qmatvec_view(buf, 0..g_len, &gv, m_e,
                        m.gate_exps.in_f, m.gate_exps.out_f, m.gate_exps.qtype, m.gate_exps.row_bytes)
                })?;
                let up = e.with_moe_cache(max_block, |c, eng| {
                    let id = BlockId::new(il, PROJ_UP, ex as u16);
                    let slot = c.dispatch(id, m.up_exps.expert_bytes(ex), eng)?;
                    let buf = c.buf(slot);
                    eng.qmatvec_view(buf, 0..u_len, &gv, m_e,
                        m.up_exps.in_f, m.up_exps.out_f, m.up_exps.qtype, m.up_exps.row_bytes)
                })?;
                // SiLU-MUL activation.
                let mut act = e.zeros(m_e * n_ff_exp)?;
                e.silu_mul(&gate, &up, &mut act, m_e * n_ff_exp)?;
                let actv = act.slice(0..m_e * n_ff_exp);
                e.with_moe_cache(max_block, |c, eng| {
                    let id = BlockId::new(il, PROJ_DOWN, ex as u16);
                    let slot = c.dispatch(id, m.down_exps.expert_bytes(ex), eng)?;
                    let buf = c.buf(slot);
                    eng.qmatvec_view(buf, 0..d_len, &actv, m_e,
                        m.down_exps.in_f, m.down_exps.out_f, m.down_exps.qtype, m.down_exps.row_bytes)
                })?
            } else {
                // STAGING PATH: H2D the expert blocks into scratch buffers, then GEMM.
                let sg = scratch_g.as_mut().unwrap();
                let su = scratch_u.as_mut().unwrap();
                let sd = scratch_d.as_mut().unwrap();
                e.stage_expert(m.gate_exps.expert_bytes(ex), sg, 0)?;
                e.stage_expert(m.up_exps.expert_bytes(ex), su, 0)?;
                e.stage_expert(m.down_exps.expert_bytes(ex), sd, 0)?;
                let gate = e.qmatvec_view(sg, 0..g_len, &gv, m_e,
                    m.gate_exps.in_f, m.gate_exps.out_f, m.gate_exps.qtype, m.gate_exps.row_bytes)?;
                let up = e.qmatvec_view(su, 0..u_len, &gv, m_e,
                    m.up_exps.in_f, m.up_exps.out_f, m.up_exps.qtype, m.up_exps.row_bytes)?;
                // SiLU-MUL activation.
                let mut act = e.zeros(m_e * n_ff_exp)?;
                e.silu_mul(&gate, &up, &mut act, m_e * n_ff_exp)?;
                let actv = act.slice(0..m_e * n_ff_exp);
                e.qmatvec_view(sd, 0..d_len, &actv, m_e,
                    m.down_exps.in_f, m.down_exps.out_f, m.down_exps.qtype, m.down_exps.row_bytes)?
            };

            // SCATTER into slot buffer: each row goes to slot_buf[tok, slot, :].
            e.scatter_slot(&y, &tok_idx_d, &slot_idx_d, &weight_d,
                           &mut slot_buf, &mut wbuf, n_embd, n_used, m_e)?;
        }

        // 5. REDUCE SLOTS: sum the 8 slots per token into the final moe_out.
        let mut moe_out = e.zeros(t * n_embd)?;
        e.reduce_slots(&slot_buf, &wbuf, &mut moe_out, n_embd, n_used, t)?;

        // STATS: print m-distribution when BW24_MOE_STATS is set.
        if std::env::var("BW24_MOE_STATS").is_ok() && !m_dist.is_empty() {
            m_dist.sort_unstable();
            let active = m_dist.len();
            let mean = m_dist.iter().sum::<usize>() as f64 / active as f64;
            let median = m_dist[active / 2];
            let max_m = *m_dist.last().unwrap();
            let min_m = m_dist[0];
            let above16 = m_dist.iter().filter(|&&x| x >= 16).count();
            println!("moe-grouped il={il} t={t} active={active}/{n_expert} \
                      m_e: min={min_m} median={median} mean={mean:.1} max={max_m} \
                      above_gemm_threshold(>=16)={above16}/{active}");
        }

        // 6. SHARED EXPERT (same as moe_ffn — untouched).
        if let (Some(gate_shexp), Some(up_shexp), Some(down_shexp), Some(gate_inp_shexp)) =
            (&m.gate_shexp, &m.up_shexp, &m.down_shexp, &m.gate_inp_shexp)
        {
            let n_ff_sh = gate_shexp.out_features();
            let sg_gate = e.matmul(gate_shexp, z, t)?;
            let sg_up = e.matmul(up_shexp, z, t)?;
            let mut sa = e.zeros(t * n_ff_sh)?;
            e.silu_mul(&sg_gate, &sg_up, &mut sa, t * n_ff_sh)?;
            let sh = e.matmul(down_shexp, &sa, t)?;
            let gs = e.linear(z, gate_inp_shexp.float_data(), t, n_embd, 1)?;
            let mut g = e.zeros(t)?;
            e.sigmoid(&gs, &mut g, t)?;
            e.add_scaled_rows(&sh, &g, &mut moe_out, n_embd, t)?;
        }

        Ok(moe_out)
    }
}
