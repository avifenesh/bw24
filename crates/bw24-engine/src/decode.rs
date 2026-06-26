//! Incremental decode (T=1) with the dual cache + greedy generation loop. Serves end-to-end.
//! Reuses the validated kernels; threads KV (full-attn) and conv/SSM state (linear-attn) across steps.

use cudarc::driver::CudaSlice;
use crate::Engine;
use crate::hybrid::{HybridModel, Mixer, FullAttnLayer, LinearAttnLayer};
use crate::cache::Cache;
use crate::forward::argmax;

impl HybridModel {
    /// One decode step for `token` at cache.pos; returns logits [n_vocab] (host f32). Advances cache.
    pub fn decode_step(&self, e: &Engine, token: u32, cache: &mut Cache) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let n_embd = cfg.n_embd as usize;
        let eps = cfg.rms_eps;
        let pos = cache.pos;
        let pos_d = e.htod_i32(&[pos as i32])?;

        // embed the single token -> [1, n_embd]
        let mut x = e.htod(&self.embd.gather(n_embd, &[token]))?;

        for (il, layer) in self.layers.iter().enumerate() {
            let mut h = e.zeros(n_embd)?;
            e.rms_norm(&x, layer.attn_norm.float_data(), &mut h, n_embd, 1, eps)?;

            let mixed = match &layer.mixer {
                Mixer::Full(fa) => self.full_attn_decode(e, fa, &h, &pos_d, pos, cache, il)?,
                Mixer::Linear(la) => self.linear_attn_decode(e, la, &h, cache, il)?,
            };

            let mut x1 = e.zeros(n_embd)?;
            e.add(&x, &mixed, &mut x1, n_embd)?;

            let mut z = e.zeros(n_embd)?;
            e.rms_norm(&x1, layer.post_attn_norm.float_data(), &mut z, n_embd, 1, eps)?;
            let n_ff = layer.ffn_gate.out_features();
            let gate = e.matmul(&layer.ffn_gate, &z, 1)?;
            let up = e.matmul(&layer.ffn_up, &z, 1)?;
            let mut act = e.zeros(n_ff)?;
            e.silu_mul(&gate, &up, &mut act, n_ff)?;
            let down = e.matmul(&layer.ffn_down, &act, 1)?;
            let mut x2 = e.zeros(n_embd)?;
            e.add(&x1, &down, &mut x2, n_embd)?;
            x = x2;
        }

        let mut hn = e.zeros(n_embd)?;
        e.rms_norm(&x, self.output_norm.float_data(), &mut hn, n_embd, 1, eps)?;
        let logits = e.matmul(&self.output, &hn, 1)?;
        let host = e.dtoh(&logits)?;
        cache.pos += 1;
        Ok(host)
    }

    /// Greedy generation: prime with prompt tokens (decode them in sequence to build state),
    /// then generate `max_new` tokens. Returns the generated token ids.
    pub fn generate(&self, e: &Engine, prompt: &[u32], max_new: usize)
                    -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let max_ctx = prompt.len() + max_new + 8;
        let mut cache = Cache::new(e, &self.cfg, max_ctx)?;
        let mut last_logits = Vec::new();
        // prime: feed each prompt token (decode_step builds KV + state incrementally)
        for &tok in prompt {
            last_logits = self.decode_step(e, tok, &mut cache)?;
        }
        let mut out = Vec::with_capacity(max_new);
        for _ in 0..max_new {
            let next = argmax(&last_logits) as u32;
            out.push(next);
            last_logits = self.decode_step(e, next, &mut cache)?;
        }
        Ok(out)
    }

    /// Full-attention decode: project q/gate/k/v for the new token, QK-norm, RoPE at pos,
    /// append k,v to the layer KV cache, attend over the full [0..=pos] context.
    fn full_attn_decode(&self, e: &Engine, fa: &FullAttnLayer, h: &CudaSlice<f32>,
                        pos_d: &CudaSlice<i32>, pos: usize, cache: &mut Cache, il: usize)
                        -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let n_head = cfg.n_head as usize;
        let n_head_kv = cfg.n_head_kv as usize;
        let head_dim = cfg.head_dim_k as usize;
        let eps = cfg.rms_eps;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // q|gate fused: [2*head_dim per head]
        let qf = e.matmul(&fa.wq, h, 1)?;
        let qf_host = e.dtoh(&qf)?;
        let mut q_host = vec![0f32; n_head * head_dim];
        let mut gate_host = vec![0f32; n_head * head_dim];
        let stride = 2 * head_dim;
        for hh in 0..n_head {
            let src = hh * stride;
            let dst = hh * head_dim;
            q_host[dst..dst + head_dim].copy_from_slice(&qf_host[src..src + head_dim]);
            gate_host[dst..dst + head_dim].copy_from_slice(&qf_host[src + head_dim..src + 2 * head_dim]);
        }
        let mut q = e.htod(&q_host)?;
        let gate = e.htod(&gate_host)?;
        let mut k = e.matmul(&fa.wk, h, 1)?;
        let v = e.matmul(&fa.wv, h, 1)?;

        // QK-norm + RoPE at position `pos`
        let mut qn = e.zeros(n_head * head_dim)?;
        e.rms_norm(&q, fa.q_norm.float_data(), &mut qn, head_dim, n_head, eps)?;
        q = qn;
        let mut kn = e.zeros(n_head_kv * head_dim)?;
        e.rms_norm(&k, fa.k_norm.float_data(), &mut kn, head_dim, n_head_kv, eps)?;
        k = kn;
        let rope_dims = cfg.rope_dim_count as usize;
        e.rope_neox(&mut q, pos_d, head_dim, rope_dims, n_head, 1, cfg.rope_freq_base, 1.0)?;
        e.rope_neox(&mut k, pos_d, head_dim, rope_dims, n_head_kv, 1, cfg.rope_freq_base, 1.0)?;

        // append k,v into the RESIDENT GPU KV cache at the current position (no host round-trip)
        let kvl = cache.kv[il].as_mut().unwrap();
        let off = kvl.len * kvl.kv_dim;
        e.copy_into(&mut kvl.k, off, &k, kvl.kv_dim)?;
        e.copy_into(&mut kvl.v, off, &v, kvl.kv_dim)?;
        kvl.len += 1;
        let t_kv = kvl.len;

        // attend: q[hd,nh,1] over the resident K/V[hd,nhkv,t_kv] (view first t_kv*kv_dim elems)
        let k_view = e.view(&kvl.k, t_kv * kvl.kv_dim);
        let v_view = e.view(&kvl.v, t_kv * kvl.kv_dim);
        let mut attn = e.zeros(n_head * head_dim)?;
        if std::env::var("BW24_NOFA").is_ok() {
            e.sdpa_naive_view(&q, &k_view, &v_view, &mut attn, head_dim, n_head, n_head_kv, 1, t_kv, scale, true)?;
        } else {
            e.fa_decode(&q, &k_view, &v_view, &mut attn, head_dim, n_head, n_head_kv, t_kv, scale)?;
        }
        let _ = pos;

        // output gate: attn * sigmoid(gate), then o-proj
        let mut gsig = e.zeros(n_head * head_dim)?;
        e.sigmoid(&gate, &mut gsig, n_head * head_dim)?;
        let mut attn_g = e.zeros(n_head * head_dim)?;
        e.mul(&attn, &gsig, &mut attn_g, n_head * head_dim)?;
        Ok(e.matmul(&fa.wo, &attn_g, 1)?)
    }

    /// Linear-attention decode: conv with ring-buffer state, GDN scan carrying SSM state.
    fn linear_attn_decode(&self, e: &Engine, la: &LinearAttnLayer, h: &CudaSlice<f32>,
                          cache: &mut Cache, il: usize)
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
        let pad = d_conv - 1;

        // projections (T=1)
        let qkv_mixed = e.matmul(&la.wqkv, h, 1)?;     // [conv_dim] (token-major, single token)
        let z = e.matmul(&la.wqkv_gate, h, 1)?;        // [value_dim]
        let beta_raw = e.matmul(&la.ssm_beta, h, 1)?;  // [num_v]
        let alpha = e.matmul(&la.ssm_alpha, h, 1)?;    // [num_v]

        // conv input = [conv_state (pad cols) | new col]  channel-major [conv_dim, pad+1]
        let qkv_host = e.dtoh(&qkv_mixed)?;            // [conv_dim] (channel = index, single token)
        let rl = cache.recur[il].as_mut().unwrap();
        let tp = pad + 1;
        let mut conv_in = vec![0f32; conv_dim * tp];
        for c in 0..conv_dim {
            // prior state cols
            for j in 0..pad { conv_in[c * tp + j] = rl.conv_state[c * pad + j]; }
            // new input col
            conv_in[c * tp + pad] = qkv_host[c];
        }
        // update conv_state ring: keep last `pad` columns of conv_in (cols 1..=pad)
        for c in 0..conv_dim {
            for j in 0..pad { rl.conv_state[c * pad + j] = conv_in[c * tp + 1 + j]; }
        }
        let conv_in_d = e.htod(&conv_in)?;
        let mut conv_out = e.zeros(conv_dim)?;  // [conv_dim, 1] channel-major, SiLU
        e.ssm_conv1d(&conv_in_d, la.ssm_conv1d.float_data(), &mut conv_out, conv_dim, 1, d_conv, true)?;
        let conv_host = e.dtoh(&conv_out)?;

        // split + repack to GDN [d_state, num_v, 1]; q/k repeat 16->32 via modulo (ggml_repeat_4d)
        let mut q_g = vec![0f32; d_state * num_v];
        let mut k_g = vec![0f32; d_state * num_v];
        let mut v_g = vec![0f32; d_state * num_v];
        for vh in 0..num_v {
            let kh = vh % num_k;
            for i in 0..d_state {
                let qc = kh * head_k + i;
                let kc = key_dim + kh * head_k + i;
                let vc = 2 * key_dim + vh * d_state + i;
                let dst = vh * d_state + i;
                q_g[dst] = conv_host[qc];
                k_g[dst] = conv_host[kc];
                v_g[dst] = conv_host[vc];
            }
        }
        let q_gd = e.htod(&q_g)?; let mut q_l2 = e.zeros(d_state * num_v)?;
        e.l2_norm(&q_gd, &mut q_l2, d_state, num_v, eps)?;
        let k_gd = e.htod(&k_g)?; let mut k_l2 = e.zeros(d_state * num_v)?;
        e.l2_norm(&k_gd, &mut k_l2, d_state, num_v, eps)?;
        let v_gd = e.htod(&v_g)?;

        let mut beta = e.zeros(num_v)?;
        e.sigmoid(&beta_raw, &mut beta, num_v)?;
        let mut g_log = e.zeros(num_v)?;
        e.gdn_glog(&alpha, la.ssm_dt.float_data(), la.ssm_a.float_data(), &mut g_log, num_v, 1)?;

        // GDN scan: SSM state stays RESIDENT on GPU. Read from cache buffer, write to a scratch,
        // then swap scratch into the cache slot (gdn needs distinct in/out buffers).
        let mut o = e.zeros(d_state * num_v)?;
        let mut state_scratch = e.zeros(d_state * d_state * num_v)?;
        e.gdn_scan_s128(&q_l2, &k_l2, &v_gd, &g_log, &beta, &rl.ssm_state, &mut state_scratch, &mut o, num_v, 1, scale)?;
        rl.ssm_state = state_scratch;   // resident swap, no host round-trip

        // gated RMSNorm + ssm_out
        let mut gn = e.zeros(d_state * num_v)?;
        e.gated_rmsnorm(&o, la.ssm_norm.float_data(), &z, &mut gn, d_state, num_v, eps)?;
        Ok(e.matmul(&la.ssm_out, &gn, 1)?)
    }
}
