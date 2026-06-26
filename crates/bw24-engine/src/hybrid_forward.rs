//! Hybrid forward pass (Stage-1, f32, prefill, single sequence). Per layer dispatches to a
//! linear-attention (Gated DeltaNet) or full-attention mixer, then SwiGLU FFN. Matches
//! llama.cpp src/models/qwen35.cpp node-for-node.

use cudarc::driver::CudaSlice;
use crate::Engine;
use crate::hybrid::{HybridModel, Mixer, FullAttnLayer, LinearAttnLayer};

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

        for layer in self.layers.iter() {
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

            // pre-FFN norm (post_attention_norm), SwiGLU, residual 2
            let mut z = e.zeros(t * n_embd)?;
            e.rms_norm(&x1, layer.post_attn_norm.float_data(), &mut z, n_embd, t, eps)?;
            let n_ff = layer.ffn_gate.out_features();
            let gate = e.matmul(&layer.ffn_gate, &z, t)?;
            let up = e.matmul(&layer.ffn_up, &z, t)?;
            let mut act = e.zeros(t * n_ff)?;
            e.silu_mul(&gate, &up, &mut act, t * n_ff)?;
            let down = e.matmul(&layer.ffn_down, &act, t)?;
            let mut x2 = e.zeros(t * n_embd)?;
            e.add(&x1, &down, &mut x2, t * n_embd)?;
            x = x2;
        }

        let mut hn = e.zeros(t * n_embd)?;
        e.rms_norm(&x, self.output_norm.float_data(), &mut hn, n_embd, t, eps)?;
        let logits = e.matmul(&self.output, &hn, t)?;
        Ok(e.dtoh(&logits)?)
    }

    pub fn forward_last(&self, e: &Engine, tokens: &[u32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let all = self.forward(e, tokens)?;
        let n_vocab = self.output.out_features();
        let t = tokens.len();
        Ok(all[(t - 1) * n_vocab..t * n_vocab].to_vec())
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
        // split per head: q = [head_dim] at offset 0 within each 2*head_dim block; gate at offset head_dim.
        // Build q [head_dim, n_head, T] and gate [head_dim, n_head, T] by host repack (Stage 1).
        let qf_host = e.dtoh(&qf)?;
        let mut q_host = vec![0f32; t * n_head * head_dim];
        let mut gate_host = vec![0f32; t * n_head * head_dim];
        let stride = 2 * head_dim;
        for tok in 0..t {
            for hh in 0..n_head {
                let src = tok * (n_head * stride) + hh * stride;
                let dst = (tok * n_head + hh) * head_dim;
                q_host[dst..dst + head_dim].copy_from_slice(&qf_host[src..src + head_dim]);
                gate_host[dst..dst + head_dim].copy_from_slice(&qf_host[src + head_dim..src + 2 * head_dim]);
            }
        }
        let mut q = e.htod(&q_host)?;
        let gate = e.htod(&gate_host)?;
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

        // conv: need channel-major [conv_dim, T+pad]. transpose qkv_mixed -> [conv_dim, T],
        // then prepend (d_conv-1) zero cols of state (prefill from zero state).
        let qkv_cm = e.transpose(&qkv_mixed, t, conv_dim)?;   // [conv_dim, T]
        let pad = d_conv - 1;
        let tp = t + pad;
        // build padded channel-major buffer on host (Stage 1)
        let qkv_cm_host = e.dtoh(&qkv_cm)?;
        let mut conv_in = vec![0f32; conv_dim * tp];
        for c in 0..conv_dim {
            for tt in 0..t { conv_in[c * tp + pad + tt] = qkv_cm_host[c * t + tt]; }
        }
        let conv_in_d = e.htod(&conv_in)?;
        let mut conv_out = e.zeros(conv_dim * t)?;  // [conv_dim, T] channel-major, SiLU applied
        e.ssm_conv1d(&conv_in_d, la.ssm_conv1d.float_data(), &mut conv_out, conv_dim, t, d_conv, true)?;

        // split conv_out channels into q/k/v and repack to GDN [d_state, num_v, T].
        // conv_out channel c, time tt at c*t + tt. q channels [0,key_dim), k [key_dim,2key_dim), v [2key_dim,conv_dim).
        let conv_host = e.dtoh(&conv_out)?;
        // q,k: [head_k, num_k, T]; v: [head_v, num_v, T]. GDN wants [d_state, num_v, T] for all.
        let mut q_g = vec![0f32; d_state * num_v * t];
        let mut k_g = vec![0f32; d_state * num_v * t];
        let mut v_g = vec![0f32; d_state * num_v * t];
        for tt in 0..t {
            for vh in 0..num_v {
                let kh = vh % num_k;  // ggml_repeat_4d head mapping is MODULO (vh % num_k), not block
                for i in 0..d_state {
                    // q channel = kh*head_k + i ; time tt
                    let qc = kh * head_k + i;
                    let kc = key_dim + kh * head_k + i;
                    let vc = 2 * key_dim + vh * head_v + i;
                    let dst = (tt * num_v + vh) * d_state + i;
                    q_g[dst] = conv_host[qc * t + tt];
                    k_g[dst] = conv_host[kc * t + tt];
                    v_g[dst] = conv_host[vc * t + tt];
                }
            }
        }
        // L2-norm q,k per (head_dim) row — rows are contiguous d_state in q_g.
        let q_gd = e.htod(&q_g)?; let mut q_l2 = e.zeros(d_state * num_v * t)?;
        e.l2_norm(&q_gd, &mut q_l2, d_state, num_v * t, eps)?;
        let k_gd = e.htod(&k_g)?; let mut k_l2 = e.zeros(d_state * num_v * t)?;
        e.l2_norm(&k_gd, &mut k_l2, d_state, num_v * t, eps)?;
        let v_gd = e.htod(&v_g)?;

        // beta = sigmoid(beta_raw) ; g_log = a * softplus(alpha + dt). Both need [num_v, T] layout
        // (g[t*num_v + h]). beta_raw/alpha are [T, num_v] token-major == that layout already.
        let mut beta = e.zeros(t * num_v)?;
        e.sigmoid(&beta_raw, &mut beta, t * num_v)?;
        // gdn_glog expects alpha [H,T] with alpha[t*H+h] and dt_bias/a [H] — matches token-major [T,num_v].
        let mut g_log = e.zeros(t * num_v)?;
        e.gdn_glog(&alpha, la.ssm_dt.float_data(), la.ssm_a.float_data(), &mut g_log, num_v, t)?;

        // GDN scan
        let state_in = e.zeros(d_state * d_state * num_v)?;  // zero state (prefill)
        let mut state_out = e.zeros(d_state * d_state * num_v)?;
        let mut o = e.zeros(d_state * num_v * t)?;
        e.gdn_scan_s128(&q_l2, &k_l2, &v_gd, &g_log, &beta, &state_in, &mut state_out, &mut o, num_v, t, scale)?;

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
