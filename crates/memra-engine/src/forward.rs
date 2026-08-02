//! Dense forward pass (Stage-1, all f32, prefill of T tokens, batch=1).
//! Matches llama.cpp qwen3 graph: embed → per layer {RMSNorm, QKV, QK-norm, RoPE, SDPA, O,
//! residual, RMSNorm, SwiGLU, residual} → output_norm → lm_head.
//!
//! Activation layout: x is [n_embd, T] but we store it row-major-per-token as [T, n_embd]
//! (token t at offset t*n_embd) so cuBLASLt linear (m=T tokens, in=n_embd) works directly.

use crate::model::Model;
use crate::Engine;

impl Model {
    /// Run prefill over `tokens`, return logits [T, n_vocab] (host f32). positions = 0..T.
    pub fn forward(
        &self,
        e: &Engine,
        tokens: &[u32],
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let n_embd = cfg.n_embd as usize;
        let n_head = cfg.n_head as usize;
        let n_head_kv = cfg.n_head_kv as usize;
        let head_dim = cfg.head_dim_k as usize;
        let t = tokens.len();
        let eps = cfg.rms_eps;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // positions 0..T
        let pos: Vec<i32> = (0..t as i32).collect();
        let pos_d = e.htod_i32(&pos)?;

        // x: [T, n_embd] (token-major)
        let mut x = self.embed_tokens(e, tokens)?;
        // Fixed MoE cache-slot size (0 for a non-MoE dense model). Computed once for the whole run.
        let max_block = self.max_moe_block();

        for (il, layer) in self.layers.iter().enumerate() {
            // --- attention block ---
            let mut h = e.zeros(t * n_embd)?;
            e.rms_norm(&x, layer.attn_norm.float_data(), &mut h, n_embd, t, eps)?;

            // QKV projections: q[T, n_head*head_dim], k/v[T, n_head_kv*head_dim]
            let q_out = layer.wq.out_features(); // n_head*head_dim
            let k_out = layer.wk.out_features();
            let mut q = e.matmul(&layer.wq, &h, t)?;
            let mut k = e.matmul(&layer.wk, &h, t)?;
            let v = e.matmul(&layer.wv, &h, t)?;

            // QK-norm: RMSNorm over head_dim, per (token, head). Layout [head_dim, n_head, T]
            // == row-major rows of length head_dim. q currently [T, n_head*head_dim] which is
            // exactly n_head*T rows of head_dim if we treat each head slice as a row. The memory
            // for token t is [head0(head_dim) head1(head_dim) ...]; so rows of head_dim are
            // contiguous and number n_head*T. RMSNorm with ncols=head_dim, nrows=n_head*T works.
            // rms_norm(ncols=head_dim, nrows=n_head*T) multiplies each row by q_norm[head_dim] —
            // exactly per-head QK-norm. Rows of head_dim are contiguous in the token-major buffer.
            if let Some(qn) = &layer.q_norm {
                let mut qn_out = e.zeros(t * q_out)?;
                e.rms_norm(&q, qn.float_data(), &mut qn_out, head_dim, n_head * t, eps)?;
                q = qn_out;
            }
            if let Some(kn) = &layer.k_norm {
                let mut kn_out = e.zeros(t * k_out)?;
                e.rms_norm(
                    &k,
                    kn.float_data(),
                    &mut kn_out,
                    head_dim,
                    n_head_kv * t,
                    eps,
                )?;
                k = kn_out;
            }

            // RoPE on q,k. Layout per token is [head_dim, n_head] contiguous. Our buffer is
            // [T, n_head*head_dim]; rope_neox expects [head_dim, n_heads, n_tokens] with grid
            // n_heads*n_tokens and head index = blockIdx % n_heads. Token-major works if we pass
            // n_heads and the kernel treats hr = token*n_heads + head. Our layout: token t at
            // t*(n_head*head_dim), head h at +h*head_dim. So hr index = t*n_head + h → matches
            // kernel's head=hr%n_heads, tok=hr/n_heads ONLY if hr = tok*n_head+head. Good.
            e.rope_neox(
                &mut q,
                &pos_d,
                head_dim,
                cfg.rope_dim_count as usize,
                n_head,
                t,
                cfg.rope_freq_base,
                1.0,
            )?;
            e.rope_neox(
                &mut k,
                &pos_d,
                head_dim,
                cfg.rope_dim_count as usize,
                n_head_kv,
                t,
                cfg.rope_freq_base,
                1.0,
            )?;

            // SDPA: q[head_dim,n_head,T], k/v[head_dim,n_head_kv,T] (T_kv = T for prefill).
            // Our buffers are token-major [T, heads*head_dim] == [head_dim, heads, T] interpreting
            // index (d, head, tok) at tok*(heads*head_dim)+head*head_dim+d. The SDPA kernel indexes
            // Q at (qt*n_head+head)*head_dim+d — identical. Good.
            let mut attn = e.zeros(t * q_out)?;
            e.sdpa_naive(
                &q, &k, &v, &mut attn, head_dim, n_head, n_head_kv, t, t, scale, true,
            )?;

            // O projection: attn[T, n_head*head_dim] @ wo[n_embd, n_head*head_dim]^T
            let o = e.matmul(&layer.wo, &attn, t)?;

            // residual 1
            let mut x1 = e.zeros(t * n_embd)?;
            e.add(&x, &o, &mut x1, t * n_embd)?;

            // --- ffn block: dense SwiGLU or routed MoE (OLMoE) ---
            let mut z = e.zeros(t * n_embd)?;
            e.rms_norm(&x1, layer.ffn_norm.float_data(), &mut z, n_embd, t, eps)?;
            let down = match &layer.ffn {
                crate::hybrid::Ffn::Dense {
                    ffn_gate,
                    ffn_up,
                    ffn_down,
                } => {
                    let n_ff = ffn_gate.out_features();
                    let gate = e.matmul(ffn_gate, &z, t)?;
                    let up = e.matmul(ffn_up, &z, t)?;
                    let mut act = e.zeros(t * n_ff)?;
                    crate::hybrid::HybridModel::ffn_act(e, cfg, &gate, &up, &mut act, t * n_ff)?;
                    e.matmul(ffn_down, &act, t)?
                }
                crate::hybrid::Ffn::Moe(m) => {
                    crate::hybrid::HybridModel::moe_ffn(e, m, &z, t, cfg, il as u16, max_block)?
                }
            };

            // residual 2
            let mut x2 = e.zeros(t * n_embd)?;
            e.add(&x1, &down, &mut x2, t * n_embd)?;
            x = x2;
        }

        // final norm + lm_head
        let mut hn = e.zeros(t * n_embd)?;
        e.rms_norm(&x, self.output_norm.float_data(), &mut hn, n_embd, t, eps)?;
        let logits = e.matmul(&self.output, &hn, t)?;
        let host = e.dtoh(&logits)?;
        Ok(host)
    }

    /// Logits for just the last token (the decode-relevant row).
    pub fn forward_last(
        &self,
        e: &Engine,
        tokens: &[u32],
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let all = self.forward(e, tokens)?;
        let n_vocab = self.output.out_features();
        let t = tokens.len();
        Ok(all[(t - 1) * n_vocab..t * n_vocab].to_vec())
    }
}

/// argmax helper.
pub fn argmax(logits: &[f32]) -> usize {
    let mut best = 0;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best
}

/// top-1/top-2 (id, value) pairs — the greedy near-tie margin is v1 - v2.
pub fn top2(logits: &[f32]) -> (usize, f32, usize, f32) {
    let (mut i1, mut v1, mut i2, mut v2) = (0usize, f32::NEG_INFINITY, 0usize, f32::NEG_INFINITY);
    for (i, &v) in logits.iter().enumerate() {
        if v > v1 {
            i2 = i1;
            v2 = v1;
            i1 = i;
            v1 = v;
        } else if v > v2 {
            i2 = i;
            v2 = v;
        }
    }
    (i1, v1, i2, v2)
}

/// Gate #46 verdict: BATCHED-PRIME last-position logits vs the TOKENWISE-PRIME reference.
///
/// The two primes are different numeric configs by design (prefill GEMM m=T vs decode
/// GEMV m=1 — cross-config drift class, like forward_last vs decode). An argmax flip on a
/// near-tie under small logit drift is within that law; a flip on a WIDE margin, or drift
/// beyond the calibrated ceiling, cannot come from the FP-composition class and fails hard.
///
/// Bounds (env-overridable; calibrated on the 2026-08-02 supported-set sweep,
/// research/prime-gate-coverage-20260802 — recalibrate when the kernels under them move,
/// the H100 stale-verdict law):
///   MEMRA_PRIME_GATE_MAXDIFF — full-vocab logit maxdiff ceiling (default 8.0). Measured
///     legal cross-config drift across the 144-prompt supported-set sweep: dense Q8_0 up
///     to ~1.0, MoE IQ4_XS/Q4_K_M up to 3.1, gemma QAT Q4_0 up to 5.5 (its logit scale is
///     larger); run-gen's own accepted forward-vs-decode drift is 1.39 on the q35 probe.
///     A real defect (indexing/wrong-weights) lands decades above this.
///   MEMRA_PRIME_GATE_MARGIN — tokenwise top1-top2 margin above which an argmax flip is
///     treated as structured (default 1.0). All 10 measured legal first-token flips sat
///     at margins <= 0.70 (per-position flips <= 0.92).
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum PrimeGateClass {
    Match,
    NearTieFlip,
    Structured,
}

pub struct PrimeGateVerdict {
    pub tw_argmax: usize,
    pub bp_argmax: usize,
    pub tw_margin: f32,
    pub bp_margin: f32,
    pub maxdiff: f32,
    pub class: PrimeGateClass,
}

fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

pub fn prime_gate_verdict(tokenwise: &[f32], batched: &[f32]) -> PrimeGateVerdict {
    let (t1, tv1, _, tv2) = top2(tokenwise);
    let (b1, bv1, _, bv2) = top2(batched);
    let maxdiff = tokenwise
        .iter()
        .zip(batched)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let maxdiff_bound = env_f32("MEMRA_PRIME_GATE_MAXDIFF", 8.0);
    let margin_bound = env_f32("MEMRA_PRIME_GATE_MARGIN", 1.0);
    let tw_margin = tv1 - tv2;
    let class = if !maxdiff.is_finite() || maxdiff > maxdiff_bound {
        PrimeGateClass::Structured
    } else if t1 == b1 {
        PrimeGateClass::Match
    } else if tw_margin <= margin_bound {
        PrimeGateClass::NearTieFlip
    } else {
        PrimeGateClass::Structured
    };
    PrimeGateVerdict {
        tw_argmax: t1,
        bp_argmax: b1,
        tw_margin,
        bp_margin: bv1 - bv2,
        maxdiff,
        class,
    }
}
