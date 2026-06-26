//! Hybrid forward pass (Stage-1, f32, prefill, single sequence). Per layer dispatches to a
//! linear-attention (Gated DeltaNet) or full-attention mixer, then SwiGLU FFN. Matches
//! llama.cpp src/models/qwen35.cpp node-for-node.

use cudarc::driver::CudaSlice;
use crate::Engine;
use crate::hybrid::{HybridModel, Mixer, FullAttnLayer, LinearAttnLayer, MoeWeights};

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
                crate::hybrid::Ffn::Moe(m) => self.moe_ffn(e, m, &z, t)?,
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

        // conv: need channel-major [conv_dim, T+pad]. transpose qkv_mixed -> [conv_dim, T],
        // then prepend (d_conv-1) zero cols of state (prefill from zero state).
        let qkv_cm = e.transpose(&qkv_mixed, t, conv_dim)?;   // [conv_dim, T]
        let pad = d_conv - 1;
        let tp = t + pad;
        // build padded channel-major buffer ON-DEVICE (zero state prepended). No dtoh/host-loop/htod.
        let mut conv_in = e.zeros(conv_dim * tp)?;
        e.conv_left_pad(&qkv_cm, &mut conv_in, conv_dim, t, pad)?;
        let mut conv_out = e.zeros(conv_dim * t)?;  // [conv_dim, T] channel-major, SiLU applied
        e.ssm_conv1d(&conv_in, la.ssm_conv1d.float_data(), &mut conv_out, conv_dim, t, d_conv, true)?;

        // split conv_out channels into q/k/v and repack to GDN [d_state, num_v, T] ON-DEVICE.
        // conv_out channel c, time tt at c*t + tt. q channels [0,key_dim), k [key_dim,2key_dim),
        // v [2key_dim,conv_dim). q/k head-repeat kh = vh % num_k (ggml_repeat_4d MODULO mapping).
        // head_v == head_k == d_state. No dtoh/host-loop/3x-htod.
        let _ = (head_k, head_v);
        let mut q_g = e.zeros(d_state * num_v * t)?;
        let mut k_g = e.zeros(d_state * num_v * t)?;
        let mut v_g = e.zeros(d_state * num_v * t)?;
        e.qkv_to_gdn_repack(&conv_out, &mut q_g, &mut k_g, &mut v_g, d_state, num_v, num_k, key_dim, t)?;
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

impl HybridModel {
    /// MoE FFN (EDGE-1 Stage-1, host-resident experts, per-token H2D of ONLY the routed 8).
    /// z: [T, n_embd] (already post-attention-normed). Returns moe_out [T, n_embd].
    /// Node-for-node vs llama.cpp build_moe_ffn + qwen35moe::build_layer_ffn.
    pub(crate) fn moe_ffn(&self, e: &Engine, m: &MoeWeights, z: &CudaSlice<f32>, t: usize)
               -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
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
        let lg = e.dtoh(&logits)?;   // [T*256] host

        let mut moe_out = e.zeros(t * n_embd)?;

        // GPU scratch: one slot per proj, big enough for ONE expert (Stage-1, no cache).
        let g_len = m.gate_exps.expert_stride;  // 860160
        let u_len = m.up_exps.expert_stride;    // 860160
        let d_len = m.down_exps.expert_stride;  // 1114112
        let mut scratch_g = e.alloc_u8(g_len)?;
        let mut scratch_u = e.alloc_u8(u_len)?;
        let mut scratch_d = e.alloc_u8(d_len)?;

        // 2. PER TOKEN: softmax-over-256, stable top-8, renorm, routed-expert loop.
        for tok in 0..t {
            let row = &lg[tok * n_expert..(tok + 1) * n_expert];

            // softmax over ALL 256 (stable: subtract max)
            let maxl = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut probs = vec![0f32; n_expert];
            let mut den = 0f32;
            for i in 0..n_expert { let x = (row[i] - maxl).exp(); probs[i] = x; den += x; }
            for p in probs.iter_mut() { *p /= den; }

            // BUG-1 FIX: stable DESC sort matching CUDA cub radix (argsort DESC, stable on ties).
            // total_cmp is NaN-safe; .then(a.cmp(&b)) gives ascending-index tiebreak.
            let mut idx: Vec<usize> = (0..n_expert).collect();
            idx.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]).then(a.cmp(&b)));
            let sel = &idx[..n_used];

            // gather UNBIASED probs as weights, then renorm: sum -> clamp -> divide. NO w_scale.
            let mut w: Vec<f32> = sel.iter().map(|&i| probs[i]).collect();
            let mut ws: f32 = w.iter().sum();
            ws = ws.max(6.103515625e-5_f32);  // F16 smallest normal, clamp BEFORE divide
            for x in w.iter_mut() { *x /= ws; }

            let zt = z.slice(tok * n_embd..(tok + 1) * n_embd);  // CudaView<f32>

            for (j, &ex) in sel.iter().enumerate() {
                // stage gate/up/down for expert `ex` (async H2D, ordered before the qmatvec below)
                // use the expert tensor's ACTUAL qtype (IQ3_S/IQ4_XS/Q6_K/... per file), not a hardcode.
                e.stage_expert(m.gate_exps.expert_bytes(ex), &mut scratch_g, 0)?;
                let gate = e.qmatvec_view(&scratch_g, 0..g_len, &zt, 1,
                    m.gate_exps.in_f, m.gate_exps.out_f, m.gate_exps.qtype, m.gate_exps.row_bytes)?;

                e.stage_expert(m.up_exps.expert_bytes(ex), &mut scratch_u, 0)?;
                let up = e.qmatvec_view(&scratch_u, 0..u_len, &zt, 1,
                    m.up_exps.in_f, m.up_exps.out_f, m.up_exps.qtype, m.up_exps.row_bytes)?;

                // act = silu(gate) * up   (length n_ff_exp = 512)
                let mut act = e.zeros(n_ff_exp)?;
                e.silu_mul(&gate, &up, &mut act, n_ff_exp)?;

                // down: TRANSPOSED, in=512 out=2048. x arg must be a CudaView (BUG-8).
                e.stage_expert(m.down_exps.expert_bytes(ex), &mut scratch_d, 0)?;
                let actv = act.slice(0..n_ff_exp);
                let y = e.qmatvec_view(&scratch_d, 0..d_len, &actv, 1,
                    m.down_exps.in_f, m.down_exps.out_f, m.down_exps.qtype, m.down_exps.row_bytes)?;

                // moe_out[tok] += w[j] * y  (BUG-9: slice_mut -> CudaViewMut)
                let mut dst = moe_out.slice_mut(tok * n_embd..(tok + 1) * n_embd);
                e.axpy_into(&y, w[j], &mut dst, n_embd)?;
            }
        }

        // 3. SHARED EXPERT (ALWAYS-ON, no routing) on the SAME z.
        let n_ff_sh = m.gate_shexp.out_features();  // 512
        let sg_gate = e.matmul(&m.gate_shexp, z, t)?;  // [T, 512]
        let sg_up = e.matmul(&m.up_shexp, z, t)?;      // [T, 512]
        let mut sa = e.zeros(t * n_ff_sh)?;
        e.silu_mul(&sg_gate, &sg_up, &mut sa, t * n_ff_sh)?;
        let sh = e.matmul(&m.down_shexp, &sa, t)?;     // [T, n_embd]

        // BUG-2 FIX: ffn_gate_inp_shexp is 1-D ne=[2048] -> out_f=1. Use e.linear(.., out_f=1),
        // NOT matmul/out_features (which would index ne[1] out of bounds).
        let gs = e.linear(z, m.gate_inp_shexp.float_data(), t, n_embd, 1)?;  // [T, 1]
        let mut g = e.zeros(t)?;
        e.sigmoid(&gs, &mut g, t)?;

        // moe_out[r, :] += sh[r, :] * g[r]   (per-token scalar gate)
        e.add_scaled_rows(&sh, &g, &mut moe_out, n_embd, t)?;

        Ok(moe_out)
    }
}
