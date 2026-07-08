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

/// LAUNCH-STRUCTURE STAGE 3 gate (BW24_MOE_DEV, default ON; `=0` restores host routing). The
/// zero-DtoH device-dispatch path for fully-resident layers: router top-k output stays on device,
/// expert weight pointers come from the per-layer device table. Requires the fused router (the
/// dev path consumes the device sel/w directly), so BW24_FUSED_ROUTER=0 also disables it.
fn moe_dev_enabled() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| std::env::var("BW24_MOE_DEV").map(|v| v != "0").unwrap_or(true)
        && !matches!(std::env::var("BW24_FUSED_ROUTER").as_deref(), Ok("0")))
}

/// MoE EXPERT dp4a gate (BW24_MOE_Q8, default ON; `=0` restores the Stage-A f32-dequant expert
/// kernels). Applies when gate/up/down expert qtypes are all in the dp4a body set (IQ3_S/IQ4_XS).
/// FP-order differs from Stage-A (int dp4a + warp tree) — argmax/run-gen/stream-identity gates
/// arbitrate; the sequential and fused q8 paths ship as a matched pair (BW24_MOE_GATE contract).
fn moe_q8_enabled() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| std::env::var("BW24_MOE_Q8").map(|v| v != "0").unwrap_or(true))
}
fn q8_expert_supported(qt: i32) -> bool {
    // k-quant arms added 2026-07-06 (Q3_K/Q4_K/Q6_K bodies for the UD tail layers). Briefly
    // default-excluded the same day when they appeared to break 35B real-prompt spec — the
    // ACTUAL culprit was the MoE router's cuBLASLt n-dependence (d994271); with the router
    // decode-exact at verify t, the k-quant arms pass the full spec battery (p1/p2/p3 + raw
    // K=1..8) and are DEFAULT ON again (+9 tok/s: 148.9 -> 157.9). BW24_MOE_Q8_KQ=0 excludes.
    static KQ: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let kq = *KQ.get_or_init(|| {
        std::env::var("BW24_MOE_Q8_KQ").map(|v| v != "0").unwrap_or(true)
    });
    // NVFP4 experts (MiniMax-M3): dot body exists (expert_dot_nvfp4_g) but enabling it here
    // broke the M3 gate (decode-vs-verify MISMATCH 3.4e1) — the q8 arms' macro handling differs
    // between t-regimes somewhere; M3 stays on the f32 arm until that parity is proven. ALSO
    // measured irrelevant for now: M3 decode is PCIe-staging-bound (11.9s HtoD in a 32-tok
    // window — SLRU misses dominate), not kernel-bound. BW24_MOE_Q8_NVFP4=1 re-enables for debug.
    let nvfp4_q8 = std::env::var("BW24_MOE_Q8_NVFP4").map(|v| v == "1").unwrap_or(false);
    qt == crate::QT_IQ3_S || qt == crate::QT_IQ4_XS || (nvfp4_q8 && qt == crate::QT_NVFP4)
        || (kq && (qt == crate::QT_Q3_K || qt == crate::QT_Q4_K || qt == crate::QT_Q6_K))
}

/// The decode-once (_dec) and IQ-MMA expert kernels dequant via IQ-specific extractors —
/// k-quant tensors must fall to the _em dot path instead.
fn q8_expert_dec_supported(qt: i32) -> bool {
    qt == crate::QT_IQ3_S || qt == crate::QT_IQ4_XS
}

/// STAGE 3 prewarm gate (BW24_MOE_PREWARM, default ON; `=0` leaves residency organic). One-shot
/// per layer: force-admit every block while FREE slots cover the whole layer (never evicts).
fn moe_prewarm_enabled() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| std::env::var("BW24_MOE_PREWARM").map(|v| v != "0").unwrap_or(true))
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
                    Self::ffn_act(e, &self.cfg, &gate, &up, &mut act, t * n_ff)?;
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
        // BW24_LAYER_PROBE=1: synchronize + print after every stage — bisects an in-graph
        // ILLEGAL_ADDRESS to (layer, stage) at ~1 line of output per layer (M3 bring-up tool).
        let probe = std::env::var("BW24_LAYER_PROBE").is_ok();
        for (il, layer) in self.layers.iter().enumerate() {
            let mut h = e.zeros(t * n_embd)?;
            e.rms_norm(&x, layer.attn_norm.float_data(), &mut h, n_embd, t, eps)?;
            if probe { e.stream().synchronize()?; eprintln!("[probe] L{il} norm ok"); }
            let mixed = match &layer.mixer {
                Mixer::Full(fa) => self.full_attn(e, fa, &h, &pos_d, t)?,
                Mixer::Linear(la) => self.linear_attn(e, la, &h, t)?,
            };
            if probe { e.stream().synchronize()?; eprintln!("[probe] L{il} mixer ok"); }
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
                    Self::ffn_act(e, &self.cfg, &gate, &up, &mut act, t * n_ff)?;
                    e.matmul(ffn_down, &act, t)?
                }
                crate::hybrid::Ffn::Moe(m) => self.moe_ffn_il(e, m, &z, t, il as u16)?,
            };
            if probe { e.stream().synchronize()?; eprintln!("[probe] L{il} ffn ok"); }
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
        let n_embd = self.cfg.n_embd as usize;
        let t = tokens.len();
        // SESSION CONTINUATION (2026-07-05): cache.pos > 0 = priming a NEW SUFFIX onto a live
        // session cache — every chunk (including the first) takes the continuation arm
        // (fa_prefill_view over the quantized past + this chunk). Fresh prime (pos==0) unchanged.
        assert!(t >= PRIME_MIN_T, "prime_cache needs T >= {PRIME_MIN_T} (caller gates)");
        assert!(cache.pos + t <= cache.max_ctx, "prime_cache: prompt exceeds cache max_ctx");

        // CHUNKED PRIME (2026-07-05, the long-ctx OOM fix): the monolithic prime allocates
        // per-layer transients proportional to T (gate/up/act = T*n_ff*4B EACH — 1.5GB apiece at
        // 16k on the 27B), which OOMs a 24GB card around 16k prompt tokens. Chunk the prompt:
        // each chunk runs the full layer stack with transients sized to the chunk, appending its
        // K/V to the resident quantized cache and carrying the GDN conv-ring + recurrent state
        // through `cache.recur` (linear_attn_prime is already stateful — a chunk boundary is
        // exactly the state carry it was built for). Full-attn chunks after the first attend to
        // the QUANTIZED past KV via fa_prefill_view (the spec-verify pattern) — same numeric
        // class as decode reading the cache. Prompts <= one chunk take the ORIGINAL monolithic
        // body byte-for-byte (chunk 0 short-circuits to the f32 fa_prefill path).
        // BW24_PRIME_CHUNK sets the chunk size (tokens); 0 disables chunking (monolithic).
        let chunk: usize = std::env::var("BW24_PRIME_CHUNK").ok()
            .and_then(|v| v.parse().ok()).unwrap_or(4096);
        if chunk == 0 || t <= chunk {
            return self.prime_chunk(e, tokens, cache);
        }
        let mut hiddens = e.uninit(t * n_embd)?;
        let mut last: Option<(Vec<f32>, CudaSlice<f32>)> = None;
        let mut start = 0usize;
        while start < t {
            // keep the tail chunk >= PRIME_MIN_T (the stateful conv needs T >= d_conv-1).
            let mut end = (start + chunk).min(t);
            if t - end > 0 && t - end < PRIME_MIN_T { end = t; }
            let (l, hs, x) = self.prime_chunk(e, &tokens[start..end], cache)?;
            e.copy_into(&mut hiddens, start * n_embd, &x, (end - start) * n_embd)?;
            last = Some((l, hs));
            start = end;
        }
        let (logits, h_seed) = last.unwrap();
        Ok((logits, h_seed, hiddens))
    }

    /// One prime chunk: the full layer stack over `tokens`, continuing from the cache's current
    /// state (`cache.pos` = tokens already primed; 0 = fresh). Positions/RoPE are absolute
    /// (cache.pos + i). Returns (last-row logits, h_seed, this chunk's hidden stack [T, n_embd]).
    fn prime_chunk(&self, e: &Engine, tokens: &[u32], cache: &mut Cache)
                       -> Result<(Vec<f32>, CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let n_embd = cfg.n_embd as usize;
        let t = tokens.len();
        let eps = cfg.rms_eps;
        let base = cache.pos;
        let pos: Vec<i32> = (base as i32..(base + t) as i32).collect();
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
                    Self::ffn_act(e, &self.cfg, &gate, &up, &mut act, t * n_ff)?;
                    e.matmul(ffn_down, &act, t)?
                }
                crate::hybrid::Ffn::Moe(m) => self.moe_ffn_il(e, m, &z, t, il as u16)?,
            };
            let mut x2 = e.zeros(t * n_embd)?;
            e.add(&x1, &ffn_out, &mut x2, t * n_embd)?;
            x = x2;
        }

        // h_seed = LAST row of x BEFORE output_norm (MTP-PLAN §A default) or AFTER it
        // (BW24_SPEC_HPOST — reference convention; hn is computed just below either way, so
        // the post-norm copy happens after hn exists).
        let mut h_seed = e.zeros(n_embd)?;
        if !crate::spec::spec_hpost() {
            e.copy_view_into(&mut h_seed, 0, &x.slice((t - 1) * n_embd..t * n_embd), n_embd)?;
        }
        // last-row logits, exactly like forward_last (norm all T — per-row op — then lm_head on 1 row).
        let mut hn = e.zeros(t * n_embd)?;
        e.rms_norm(&x, self.output_norm.float_data(), &mut hn, n_embd, t, eps)?;
        if crate::spec::spec_hpost() {
            e.copy_view_into(&mut h_seed, 0, &hn.slice((t - 1) * n_embd..t * n_embd), n_embd)?;
        }
        let last = e.view(&hn, t * n_embd);
        let last_row = last.slice((t - 1) * n_embd..t * n_embd);
        let mut hlast = e.zeros(n_embd)?;
        e.copy_view_into(&mut hlast, 0, &last_row, n_embd)?;
        let logits = e.matmul(&self.output, &hlast, 1)?;
        cache.pos += t;
        // Hidden stack handed to generate_spec as prompt_h: pre-norm x (default) or the full
        // post-norm stack hn (BW24_SPEC_HPOST).
        Ok((e.dtoh(&logits)?, h_seed, if crate::spec::spec_hpost() { hn } else { x }))
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

        // qwen35 fuses [q|gate] per head in wq (2*head_dim stride); M3 has NO output gate
        // (attention_output_gate=false) — wq out = n_head*head_dim exactly, and q_gate_split
        // would read 2x out of bounds. `gated` keys both the split and the sigmoid epilogue.
        let gated = cfg.m3.is_none();
        let qf = e.matmul(&fa.wq, h, t)?;
        let (mut q, gate) = if gated {
            let mut q = e.zeros(t * n_head * head_dim)?;
            let mut gate = e.zeros(t * n_head * head_dim)?;
            e.q_gate_split(&qf, &mut q, &mut gate, head_dim, n_head, t)?;
            (q, Some(gate))
        } else {
            (qf, None)
        };
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

        // batched prefill attention. FRESH prime (no past KV): unchanged forward_last math over
        // the f32 K/V of this batch. CONTINUATION chunk (past KV present): the chunk's queries
        // must attend to [0 .. base+t) — run fa_prefill_view over the resident QUANTIZED cache
        // (the spec-verify pattern; kernel's causal mask offsets by T_kv-T). Numerically this
        // reads q8_0/q5_1-dequantized K/V for the past AND the current chunk — the same class as
        // decode reading the cache; the run-gen/first-16 battery is the accuracy authority.
        let base_len = {
            let kvl = cache.kv[il].as_ref().unwrap();
            kvl.len - t   // KV rows present BEFORE this chunk's append above
        };
        let mut attn = e.zeros(t * n_head * head_dim)?;
        if base_len == 0 {
            // fa_prefill's smem layout is compile-time HEAD_DIM: stamped twins exist for 256
            // (qwen35) and 128 (M3, `_hd128` — 2026-07-07). Other dims would overrun the
            // runtime-sized allocation -> ILLEGAL_ADDRESS; fall to naive SDPA there.
            if std::env::var("BW24_NOFA").is_ok() || !(head_dim == 256 || head_dim == 128) {
                e.sdpa_naive(&q, &k, &v, &mut attn, head_dim, n_head, n_head_kv, t, t, scale, true)?;
            } else {
                e.fa_prefill(&q, &k, &v, &mut attn, head_dim, n_head, n_head_kv, t, t, scale, true)?;
            }
        } else {
            let kvl = cache.kv[il].as_ref().unwrap();
            let t_kv = base_len + t;
            let k_view = e.view_u8(&kvl.k, t_kv * kvl.k_tok_bytes);
            let v_view = e.view_u8(&kvl.v, t_kv * kvl.v_tok_bytes);
            // ARC B (2026-07-05): dequant-once workspace, DEFAULT ON. fa_prefill_q's inline
            // dequant re-reads+re-dequants the whole quantized KV stream from every one of the
            // T/64 x n_head CTAs (64x+ redundant at chunk=4096; 30.5% of the 32k prime wall).
            // fa_prefill_view_ws dequants K/V ONCE into a resident bf16 workspace then runs the
            // bit-identical bf16 twin (fa_prefill_qw) — same staged values, same FP order, token-
            // identical output (gate: BW24_PRIME_CHUNK=4096 ws-on vs ws-off vs monolithic).
            // BW24_PRIME_DEQW=0 reverts to the inline-dequant kernel.
            let deqw = std::env::var("BW24_PRIME_DEQW").map(|v| v != "0").unwrap_or(true);
            if deqw {
                e.fa_prefill_view_ws(&q, &k_view, &v_view, &mut attn, head_dim, n_head, n_head_kv,
                                     t, t_kv, scale, true, kvl.k_tok_bytes, kvl.v_tok_bytes)?;
            } else {
                e.fa_prefill_view(&q, &k_view, &v_view, &mut attn, head_dim, n_head, n_head_kv,
                                  t, t_kv, scale, true, kvl.k_tok_bytes, kvl.v_tok_bytes)?;
            }
        }

        let attn_g = match &gate {
            Some(gate) => {
                let mut gsig = e.zeros(t * n_head * head_dim)?;
                e.sigmoid(gate, &mut gsig, t * n_head * head_dim)?;
                let mut ag = e.zeros(t * n_head * head_dim)?;
                e.mul(&attn, &gsig, &mut ag, t * n_head * head_dim)?;
                ag
            }
            None => attn,
        };
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
    pub fn full_attn(&self, e: &Engine, fa: &FullAttnLayer, h: &CudaSlice<f32>, pos_d: &CudaSlice<i32>, t: usize)
                 -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let _n_embd = cfg.n_embd as usize;
        let n_head = cfg.n_head as usize;
        let n_head_kv = cfg.n_head_kv as usize;
        let head_dim = cfg.head_dim_k as usize;
        let eps = cfg.rms_eps;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // qwen35: wq output = head_dim*2*n_head (fused [q|gate] per head). M3: NO output gate —
        // wq out = n_head*head_dim, no split (see prime-path note).
        let gated = cfg.m3.is_none();
        let qf = e.matmul(&fa.wq, h, t)?;
        let (mut q, gate) = if gated {
            let mut q = e.zeros(t * n_head * head_dim)?;
            let mut gate = e.zeros(t * n_head * head_dim)?;
            e.q_gate_split(&qf, &mut q, &mut gate, head_dim, n_head, t)?;
            (q, Some(gate))
        } else {
            (qf, None)
        };
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
        // hand-written FlashAttention prefill (head_dim 256/128 stamped twins). BW24_NOFA
        // falls back to naive sdpa.
        if std::env::var("BW24_NOFA").is_ok() || !(head_dim == 256 || head_dim == 128) {
            // head_dim gate: see prime-path note (fa_prefill is stamped at 256 and 128 only).
            e.sdpa_naive(&q, &k, &v, &mut attn, head_dim, n_head, n_head_kv, t, t, scale, true)?;
        } else {
            e.fa_prefill(&q, &k, &v, &mut attn, head_dim, n_head, n_head_kv, t, t, scale, true)?;
        }

        // output gate: attn * sigmoid(gate) — qwen35 only (M3 has no gate).
        let attn_g = match &gate {
            Some(gate) => {
                let mut gsig = e.zeros(t * n_head * head_dim)?;
                e.sigmoid(gate, &mut gsig, t * n_head * head_dim)?;
                let mut ag = e.zeros(t * n_head * head_dim)?;
                e.mul(&attn, &gsig, &mut ag, t * n_head * head_dim)?;
                ag
            }
            None => attn,
        };

        // o projection
        let o = e.matmul(&fa.wo, &attn_g, t)?;
        Ok(o)
    }

    /// Linear-attention (Gated DeltaNet) mixer (qwen35 :338-470).
    pub fn linear_attn(&self, e: &Engine, la: &LinearAttnLayer, h: &CudaSlice<f32>, t: usize)
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
    pub fn moe_ffn_il(&self, e: &Engine, m: &MoeWeights, z: &CudaSlice<f32>, t: usize, il: u16)
               -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        Self::moe_ffn(e, m, z, t, &self.cfg, il, self.max_moe_block())
    }

    /// Decode-path twin with a PRE-QUANTIZED z (from add_rms_norm_zq8): threads (zq, zd) into the
    /// t=1 dev arm so the per-layer standalone quantize_q8_1 launch folds away. Identical bytes
    /// (the fused kernel reproduces quantize_q8_1 exactly); every other path ignores the pair.
    pub fn moe_ffn_il_zq8(&self, e: &Engine, m: &MoeWeights, z: &CudaSlice<f32>,
                          zq8: Option<&(CudaSlice<i8>, CudaSlice<f32>)>, t: usize, il: u16)
               -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        Self::moe_ffn_inner(e, m, z, zq8, t, &self.cfg, il, self.max_moe_block())
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
        Self::moe_ffn_inner(e, m, z, None, t, cfg, il, max_block)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn moe_ffn_inner(e: &Engine, m: &MoeWeights, z: &CudaSlice<f32>,
                          zq8: Option<&(CudaSlice<i8>, CudaSlice<f32>)>, t: usize,
                          cfg: &ModelConfig, il: u16, max_block: usize)
               -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        // A2: Expert-grouped dispatch for prefill (T>1). BW24_MOE_GROUPED=1 routes here.
        if t > 1 && std::env::var("BW24_MOE_GROUPED").is_ok() {
            let grouped_out = Self::moe_ffn_grouped(e, m, z, t, cfg, il, max_block)?;
            // BW24_MOE_GATE: byte-identity comparison vs sequential path.
            // KNOWN t>1 MISMATCH maxdiff ~3.4e-4 (deterministic, 5x bit-identical 2026-07-05): the
            // sequential arm routes resident experts through the dev_q8 dp4a path (q8_1-quantized z
            // and act rows) while grouped stays f32-dequant qmatvec — a quantize-path difference,
            // not a bug (per-stage: act q8-vs-f32 ~4-9e-3 abs on |act|<=3, down-only ~1-3e-4; the
            // q8_1 activation-quantize error class). BW24_MOE_Q8=0 restores BYTE-IDENTICAL.
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
        Self::moe_ffn_sequential_zq8(e, m, z, zq8, t, cfg, il, max_block)
    }

    /// Sequential (per-token) MoE FFN -- the original path. Factored out for the gate comparison.
    pub(crate) fn moe_ffn_sequential(e: &Engine, m: &MoeWeights, z: &CudaSlice<f32>, t: usize,
                          cfg: &ModelConfig, il: u16, max_block: usize)
               -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        Self::moe_ffn_sequential_zq8(e, m, z, None, t, cfg, il, max_block)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn moe_ffn_sequential_zq8(e: &Engine, m: &MoeWeights, z: &CudaSlice<f32>,
                          zq8: Option<&(CudaSlice<i8>, CudaSlice<f32>)>, t: usize,
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

        let use_cache = Engine::moe_cache_enabled();
        let moe_q8 = moe_q8_enabled()
            && q8_expert_supported(m.gate_exps.qtype) && q8_expert_supported(m.up_exps.qtype)
            && q8_expert_supported(m.down_exps.qtype);

        // 1. ROUTER: logits = ffn_gate_inp @ z  -> [T, 256]. gate_inp is F32 -> cuBLASLt, whose
        // reductions are n-DEPENDENT (lt_ndep probe: m=1 vs m=2 col0 differs every bit). At
        // small t (spec verify, 2..15) that shifts router logits vs the T=1 decode chain ->
        // top-k WEIGHTS (and at tie margins the SELECTION) differ -> verify != decode. Route
        // small-t through per-column m=1 calls (decode-exact contract); real prefill keeps the
        // batched GEMM.
        let logits = if t > 1 && t < PRIME_MIN_T {
            e.matmul_decode_exact(&m.gate_inp, z, t)?
        } else {
            e.matmul(&m.gate_inp, z, t)?
        };

        // LAUNCH-STRUCTURE STAGE 3 (2026-07-05, BW24_MOE_DEV default ON, =0 rollback): ZERO-DtoH
        // device-dispatch when this layer's expert blocks are ALL cache-resident. The fused
        // router's sel/w stay ON DEVICE; the expert weight pointers come from the per-layer
        // device table of fixed slot addresses; gate/up/silu + down/fma run as the same TWO
        // launches per token as gdec. Removes the per-layer router DtoH + stream sync — the
        // per-token host stall that dominated the 35B decode wall after stages 1+2.
        // BIT-IDENTITY: the router kernel is selection-exact vs the host oracle (kernel-check
        // tie gate) and the _dev matvec twins reproduce the gdec kernels' exact FP chains; the
        // only difference is where sel/w/pointers are READ from (device instead of params).
        // Residency: one-shot PREWARM force-admits the layer while free slots cover it
        // (BW24_MOE_PREWARM=0 -> organic residency, dev path fires when the SLRU fills).
        // Any non-resident layer falls through to host routing + the gdec/sequential path.
        // FITS-VRAM RESIDENT EXPERTS (2026-07-06): the layer's expert slabs are device-resident
        // (load-time decision) — fire the zero-DtoH dev path unconditionally with the prebuilt
        // pointer row. No cache, no dispatch, no residency check: the llama full-offload regime
        // (it measured 169.55 vs the cache path's 28.5 on the local 35B — the residency-gate
        // all-or-nothing fallback was the 6x). BIT-IDENTITY: same _dev kernels, same math; only
        // the pointer table's provenance differs (slab base+stride vs SLRU slot addresses).
        // MoE PREFILL PAIR-BATCH (2026-07-06, the 16x pp hole): t>1 on resident experts — ONE
        // launch per proj covers ALL (token,expert) pairs (grid.y=pair, warp-per-row), replacing
        // the per-expert loop (256 experts x 3-4 launches x tiny m_e). Scatter is slot-ordered
        // per token (the sequential-axpy bit-identity class). Requires q8-supported qtypes +
        // resident slabs. BW24_MOE_PAIRS=0 rollback.
        // t >= PRIME_MIN_T only (2026-07-06 exactness fix, verify-probe proof): spec VERIFY
        // batches (t = 2..K+2) previously rode these pairs kernels while T=1 decode rode the
        // dev_q8 loop — different FP chains, verify-T2 logit maxdiff 2.6e-1 vs eager -> greedy
        // flips at tight margins -> 35B real-prompt spec self-consistency FAIL (the 27B
        // "verify must be kernel-DISPATCH-identical to decode" lesson, MoE edition). Small-t
        // now rides the dev loop below (same kernels per token as decode); pairs serves real
        // prefill (t >= 16, where spec never verifies).
        if cfg.m3.is_none() && t >= PRIME_MIN_T && m.dev_exps.is_some() && moe_q8_enabled()
            && q8_expert_supported(m.gate_exps.qtype) && q8_expert_supported(m.up_exps.qtype)
            && q8_expert_supported(m.down_exps.qtype)
            && std::env::var("BW24_MOE_PAIRS").map(|v| v != "0").unwrap_or(true)
            && std::env::var("BW24_MOE_STATS").is_err() {
            return Self::moe_ffn_pairs(e, m, z, &logits, t, cfg);
        }

        // t < PRIME_MIN_T: moe_ffn_dev loops tokens serially (1 launch-pair per token) — the
        // decode path AND the spec-verify path (dispatch parity = exactness; see pairs gate
        // above). Serial launches are fine at t<=10 (K+2); real prefill never lands here.
        // moe_ffn_dev routes via the FUSED SOFTMAX device router (moe_router_topk) — M3's
        // sigmoid routing (+e_score_correction_bias) has no device kernel yet, so M3 must NOT
        // enter the dev arms: with MOE_CACHE=1 it silently routed softmax = wrong experts
        // (gate MISMATCH 74602 vs 92, caught 2026-07-07). Host sigmoid path below is correct.
        let dev_ok = cfg.m3.is_none();
        if dev_ok && t < PRIME_MIN_T && m.dev_exps.is_some() && n_used <= 8 && moe_dev_enabled()
            && std::env::var("BW24_MOE_STATS").is_err() {
            return Self::moe_ffn_dev(e, m, z, zq8, &logits, t, cfg, il, max_block);
        }
        if dev_ok && use_cache && n_used <= 8 && moe_dev_enabled()
            && std::env::var("BW24_MOE_STATS").is_err() {
            let row_ok = e.with_moe_cache(max_block, |c, eng| {
                if moe_prewarm_enabled() { c.prewarm_layer(il, m, eng)?; }
                Ok(c.layer_dev_row(il, n_expert, eng)?.is_some())
            })?;
            if row_ok {
                return Self::moe_ffn_dev(e, m, z, zq8, &logits, t, cfg, il, max_block);
            }
        }

        // Per-token (sel[8], w[8]) — either fused-router (device top-k) or host softmax+sort.
        let (sel_all, w_all) = if let Some(m3) = cfg.m3.as_ref().filter(|m| m.sigmoid_routing) {
            Self::moe_route_cfg(e, &logits, t, n_expert, n_used,
                                m.exp_probs_b.as_deref(), Some(m3.routed_scaling_factor))?
        } else {
            Self::moe_route(e, &logits, t, n_expert, n_used)?
        };

        // BW24_MOE_TRACE=<path>: append one line per (layer, step) with the selected expert ids —
        // offline analysis derives the decode working set + step-to-step reuse (the go/no-go
        // measurement for resident-expert tiering; see rig5090.jsonl 2026-07-07 pinned-tier row).
        if let Ok(path) = std::env::var("BW24_MOE_TRACE") {
            use std::io::Write as _;
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
                let ids: Vec<String> = sel_all.iter().map(|s| s.to_string()).collect();
                let _ = writeln!(f, "{} {} {}", il, t, ids.join(","));
            }
        }

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

        // LAUNCH-STRUCTURE STAGE 2 (2026-07-05): moe_out memset elision on the gdec path.
        // moe_down8_fma_f32 FULLY overwrites its token row (dst[o] = the in-kernel FMA chain that
        // starts at 0.0f — numerically the axpy-into-zeroed-row chain), so when the grouped-decode
        // path fires the upfront `e.zeros(t*n_embd)` memset is pure launch churn. Allocate uninit
        // when gdec CAN fire (any t — decode t=1 AND the spec verify t=K+1 route here per token)
        // and lazily zero ONLY the row of a token that falls through to the sequential axpy loop.
        // BIT-IDENTITY: unchanged — every row is either fully overwritten (gdec) or
        // zeroed-then-accumulated exactly as before (fallback).
        let gdec_may_fire = use_cache && n_used <= 8 && gdec_enabled();
        let mut moe_out = if gdec_may_fire { e.uninit(t * n_embd)? } else { e.zeros(t * n_embd)? };

        // GPU scratch: one slot per proj, big enough for ONE expert (default stage-every-token path).
        // STAGE 2: LAZY — allocated only if the no-cache staging path actually runs (under
        // BW24_MOE_CACHE they were 3 dead ~1MB alloc_zeros + memset + free per layer per token,
        // measured ~123 memsets/token of the decode wall).
        let g_len = m.gate_exps.expert_stride;  // 860160
        let u_len = m.up_exps.expert_stride;    // 860160
        let d_len = m.down_exps.expert_stride;  // 1114112
        let mut scratch_g: Option<CudaSlice<u8>> = None;
        let mut scratch_u: Option<CudaSlice<u8>> = None;
        let mut scratch_d: Option<CudaSlice<u8>> = None;
        // `max_block` (the GLOBAL max expert stride across all layers) is passed in — the cache slots
        // are FIXED-ADDRESS and must fit any layer's block (UD/dynamic GGUFs vary quant per layer).

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
            let mut tok_q8: Option<(CudaSlice<i8>, CudaSlice<f32>)> = None;

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
            // cfg.m3: the grouped kernels' fused epilogues are plain SiLU — M3's swigluoai must
            // NOT take them until the kernels grow the clamped variant. NVFP4 experts carry
            // per-expert macro-scales the fused kernels don't fold — those fall through too.
            let no_macros = m.gate_exps.macros.is_none() && m.up_exps.macros.is_none()
                && m.down_exps.macros.is_none();
            if gdec_may_fire && moe_q8 && cfg.m3.is_none() && no_macros {
                if tok_q8.is_none() {
                    tok_q8 = Some(e.quantize_q8_1_view(&zt, 1, n_embd)?);
                }
                let (zq, zd) = tok_q8.as_ref().unwrap();
                if Self::moe_gdec_token_q8(e, m, il, max_block, zq, zd, sel, w,
                                           &mut moe_out, tok, n_embd, n_ff_exp, n_used)? {
                    continue;
                }
            } else if gdec_may_fire && cfg.m3.is_none() && no_macros
                && Self::moe_gdec_token(e, m, il, max_block, &zt, sel, w,
                                        &mut moe_out, tok, n_embd, n_ff_exp, n_used)? {
                continue;
            }

            // STAGE 2 memset-elision invariant: moe_out was allocated UNINIT when gdec could fire.
            // This token fell through to the sequential axpy loop, which ACCUMULATES — zero its row
            // first (row-sized memset, replaces the old full-buffer zeros; other rows are gdec-owned).
            if gdec_may_fire {
                let mut row = moe_out.slice_mut(tok * n_embd..(tok + 1) * n_embd);
                e.memset_zeros_view(&mut row)?;
            }

            for (j, &ex) in sel.iter().enumerate() {
                let ex = ex as usize;
                if use_cache && moe_q8 {
                    // dp4a EXPERT PATH (BW24_MOE_Q8): quantize z-row once per token (hoisted
                    // below via zq/zd lazies), int-dot the three projections. Same dispatch/
                    // residency mechanics as the f32 arm; only the matvec kernel differs.
                    if tok_q8.is_none() {
                        tok_q8 = Some(e.quantize_q8_1_view(&zt, 1, n_embd)?);
                    }
                    let (zq, zd) = tok_q8.as_ref().unwrap();
                    let gate = Self::moe_cached_gemm_q8(e, il, PROJ_GATE, ex, m, max_block, zq, zd)?;
                    let up   = Self::moe_cached_gemm_q8(e, il, PROJ_UP,   ex, m, max_block, zq, zd)?;
                    let mut act = e.uninit(n_ff_exp)?;
                    Self::ffn_act_scaled(e, cfg, &gate, &up,
                        m.gate_exps.macro_scale(ex), m.up_exps.macro_scale(ex), &mut act, n_ff_exp)?;
                    let (aq2, ad2) = e.quantize_q8_1(&act, 1, n_ff_exp)?;
                    let y = Self::moe_cached_gemm_q8(e, il, PROJ_DOWN, ex, m, max_block, &aq2, &ad2)?;
                    let mut dst = moe_out.slice_mut(tok * n_embd..(tok + 1) * n_embd);
                    // down-proj macro folds into the accumulate weight (1.0 for non-macro archs).
                    e.axpy_into(&y, w[j] * m.down_exps.macro_scale(ex), &mut dst, n_embd)?;
                } else if use_cache {
                    // SLRU residency cache: per-projection, dispatch the block (HIT => resident slot,
                    // MISS => staged slot) then run the SAME unchanged qmatvec_view from that slot.
                    // The bytes the kernel reads are byte-for-byte the same GGUF block (§B.3); the
                    // only difference between HIT and MISS is whether the memcpy_htod ran.
                    let gate = Self::moe_cached_gemm(e, il, PROJ_GATE, ex, m, max_block, &zt)?;
                    let up   = Self::moe_cached_gemm(e, il, PROJ_UP,   ex, m, max_block, &zt)?;
                    let mut act = e.uninit(n_ff_exp)?;  // activation fully overwrites
                    Self::ffn_act_scaled(e, cfg, &gate, &up,
                        m.gate_exps.macro_scale(ex), m.up_exps.macro_scale(ex), &mut act, n_ff_exp)?;
                    let actv = act.slice(0..n_ff_exp);
                    let y = Self::moe_cached_gemm(e, il, PROJ_DOWN, ex, m, max_block, &actv)?;
                    let mut dst = moe_out.slice_mut(tok * n_embd..(tok + 1) * n_embd);
                    // down-proj macro folds into the accumulate weight (post-matmul linear scale).
                    e.axpy_into(&y, w[j] * m.down_exps.macro_scale(ex), &mut dst, n_embd)?;
                } else {
                    // Stage-1: stage gate/up/down for expert `ex` into the scratch slots, then GEMM.
                    // Lazy scratch: first no-cache expert allocates the 3 slots (uninit — stage_expert
                    // fully overwrites the byte range the GEMM reads).
                    if scratch_g.is_none() {
                        scratch_g = Some(e.alloc_u8_uninit(g_len)?);
                        scratch_u = Some(e.alloc_u8_uninit(u_len)?);
                        scratch_d = Some(e.alloc_u8_uninit(d_len)?);
                    }
                    let (sg, su, sd) = (scratch_g.as_mut().unwrap(), scratch_u.as_mut().unwrap(),
                                        scratch_d.as_mut().unwrap());
                    e.stage_expert(m.gate_exps.expert_bytes(ex), sg, 0)?;
                    let gate = e.qmatvec_view(sg, 0..g_len, &zt, 1,
                        m.gate_exps.in_f, m.gate_exps.out_f, m.gate_exps.qtype, m.gate_exps.row_bytes)?;

                    e.stage_expert(m.up_exps.expert_bytes(ex), su, 0)?;
                    let up = e.qmatvec_view(su, 0..u_len, &zt, 1,
                        m.up_exps.in_f, m.up_exps.out_f, m.up_exps.qtype, m.up_exps.row_bytes)?;

                    let mut act = e.uninit(n_ff_exp)?;  // activation fully overwrites
                    Self::ffn_act_scaled(e, cfg, &gate, &up,
                        m.gate_exps.macro_scale(ex), m.up_exps.macro_scale(ex), &mut act, n_ff_exp)?;

                    e.stage_expert(m.down_exps.expert_bytes(ex), sd, 0)?;
                    let actv = act.slice(0..n_ff_exp);
                    let y = e.qmatvec_view(sd, 0..d_len, &actv, 1,
                        m.down_exps.in_f, m.down_exps.out_f, m.down_exps.qtype, m.down_exps.row_bytes)?;

                    let mut dst = moe_out.slice_mut(tok * n_embd..(tok + 1) * n_embd);
                    e.axpy_into(&y, w[j] * m.down_exps.macro_scale(ex), &mut dst, n_embd)?;
                }
            }
        }

        // 3. SHARED EXPERT (ALWAYS-ON, no routing) on the SAME z — qwen35moe only. OLMoE and most
        //    vanilla MoE have NO shared expert (the shexp tensors are absent / `None`); skip it then.
        // gate_inp_shexp is OPTIONAL: qwen35moe gates the shared expert (sigmoid(gate_inp) x sh);
        // MiniMax-M3 (DeepSeek-V3 class) has NO shexp gate — the shared expert adds directly.
        if let (Some(gate_shexp), Some(up_shexp), Some(down_shexp)) =
            (&m.gate_shexp, &m.up_shexp, &m.down_shexp)
        {
            let n_ff_sh = gate_shexp.out_features();  // 512
            // Q8 TRUNK-FUSION (decode t=1): gate_shexp+up_shexp are Q8_0 same-shape on the 35B —
            // ONE fused2 launch (also folds the two per-matmul re-quantizes of z into one).
            // Bit-identical per (tensor,row); falls back to the two matmul calls when ineligible.
            // Small-t (spec verify 2..15) rides matmul_decode_exact so shexp FP chains match the
            // t==1 decode chain per column (cuBLASLt n-dependence + dp4a-vs-mmvq class); real
            // prefill keeps the batched matmul. Activation routes through ffn_act (SiLU for
            // softmax archs, clamped swigluoai for M3 — identical to silu_mul when cfg.m3 is None).
            let verify_t = t > 1 && t < PRIME_MIN_T;
            let (sg_gate, sg_up) = if t == 1 {
                match e.matmul_q8_fused2_x(gate_shexp, up_shexp, z)? {
                    Some(pair) => pair,
                    None => (e.matmul(gate_shexp, z, t)?, e.matmul(up_shexp, z, t)?),
                }
            } else if verify_t {
                (e.matmul_decode_exact(gate_shexp, z, t)?, e.matmul_decode_exact(up_shexp, z, t)?)
            } else {
                (e.matmul(gate_shexp, z, t)?, e.matmul(up_shexp, z, t)?)   // [T, 512] each
            };
            let mut sa = e.uninit(t * n_ff_sh)?;  // activation fully overwrites
            Self::ffn_act(e, cfg, &sg_gate, &sg_up, &mut sa, t * n_ff_sh)?;
            let sh = if verify_t { e.matmul_decode_exact(down_shexp, &sa, t)? }
                     else { e.matmul(down_shexp, &sa, t)? };     // [T, n_embd]

            // shexp gate: qwen35moe sigmoid-gates via ffn_gate_inp_shexp (1-D ne=[n_embd] ->
            // out_f=1, e.linear NOT matmul); M3 has no gate tensor -> weight 1.0.
            let g = match &m.gate_inp_shexp {
                Some(gate_inp_shexp) => {
                    let gs = if verify_t {
                        e.linear_decode_exact(z, gate_inp_shexp.float_data(), t, n_embd, 1)?
                    } else {
                        e.linear(z, gate_inp_shexp.float_data(), t, n_embd, 1)?  // [T, 1]
                    };
                    let mut g = e.uninit(t)?;  // sigmoid fully overwrites
                    e.sigmoid(&gs, &mut g, t)?;
                    g
                }
                None => e.htod(&vec![1.0f32; t])?,
            };
            // moe_out[r, :] += sh[r, :] * g[r]   (per-token scalar gate; g=1 ungated)
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

    /// FFN activation dispatch: swigluoai (clamped, alpha/limit) when cfg.m3 says so, else the
    /// standard SiLU*up. One seam so every FFN site (dense, routed expert, shared expert) follows
    /// the model's activation exactly.
    pub fn ffn_act(e: &Engine, cfg: &ModelConfig, gate: &CudaSlice<f32>, up: &CudaSlice<f32>,
               act: &mut CudaSlice<f32>, n: usize) -> Result<(), Box<dyn std::error::Error>> {
        Self::ffn_act_scaled(e, cfg, gate, up, 1.0, 1.0, act, n)
    }

    /// ffn_act with per-tensor post-matmul macro-scales folded in (gs/us == 1.0 -> identical
    /// float ops to ffn_act; used by the ModelOpt NVFP4 expert path where each expert tensor
    /// carries a `weight_scale_2`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn ffn_act_scaled(e: &Engine, cfg: &ModelConfig, gate: &CudaSlice<f32>, up: &CudaSlice<f32>,
               gs: f32, us: f32, act: &mut CudaSlice<f32>, n: usize)
               -> Result<(), Box<dyn std::error::Error>> {
        if let Some(m3) = cfg.m3.as_ref() {
            return e.swigluoai_mul_scaled(gate, up, gs, us, m3.swiglu_alpha, m3.swiglu_limit, act, n);
        }
        if gs == 1.0 && us == 1.0 { return e.silu_mul(gate, up, act, n); }
        e.silu_mul_scaled(gate, up, gs, us, act, n)
    }

    /// Routing for the whole batch: returns (sel [T*n_used] expert ids, w [T*n_used] renorm weights),
    /// token-major. Default = the Stage-1 host path (dtoh logits, softmax-256, stable DESC top-k,
    /// renorm). BW24_FUSED_ROUTER = the device kernel (§A) which reproduces the same numerics; we
    /// still dtoh the tiny [T,n_used] sel/w buffers (64 B/token vs 1 KB/token) — the host loop
    /// indexes HostExps.bytes on the CPU to choose the DMA source (§A.2 output staging).
    fn moe_route(e: &Engine, logits: &CudaSlice<f32>, t: usize, n_expert: usize, n_used: usize)
                 -> Result<(Vec<u32>, Vec<f32>), Box<dyn std::error::Error>> {
        Self::moe_route_cfg(e, logits, t, n_expert, n_used, None, None)
    }

    /// MiniMax-M3 (DeepSeek-V3-style) sigmoid routing, host oracle. Reference:
    /// M3 modeling code — scores = sigmoid(logits); selection over scores + e_score_correction_bias;
    /// weights = un-biased scores of the selected experts, sum-normalized, x routed_scaling_factor.
    /// `m3` = (bias, routed_scaling_factor). Softmax arch passes None -> the qwen35moe/OLMoE path.
    fn moe_route_cfg(e: &Engine, logits: &CudaSlice<f32>, t: usize, n_expert: usize, n_used: usize,
                     bias: Option<&[f32]>, scale: Option<f32>)
                 -> Result<(Vec<u32>, Vec<f32>), Box<dyn std::error::Error>> {
        if let Some(sf) = scale {
            // sigmoid routing (M3). Host path only for now (fused-router kernel is softmax-top-k).
            let lg = e.dtoh(logits)?;
            let mut sel = vec![0u32; t * n_used];
            let mut w_out = vec![0f32; t * n_used];
            for tok in 0..t {
                let row = &lg[tok * n_expert..(tok + 1) * n_expert];
                let scores: Vec<f32> = row.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
                // selection score = sigmoid + bias; weight = plain sigmoid.
                let selsc: Vec<f32> = match bias {
                    Some(b) => scores.iter().zip(b).map(|(s, bb)| s + bb).collect(),
                    None => scores.clone(),
                };
                let mut idx: Vec<usize> = (0..n_expert).collect();
                idx.sort_by(|&a, &b| selsc[b].total_cmp(&selsc[a]).then(a.cmp(&b)));
                let sl = &idx[..n_used];
                let mut wv: Vec<f32> = sl.iter().map(|&i| scores[i]).collect();
                let ws: f32 = wv.iter().sum::<f32>().max(1e-20);
                for x in wv.iter_mut() { *x = *x / ws * sf; }
                for j in 0..n_used {
                    sel[tok * n_used + j] = sl[j] as u32;
                    w_out[tok * n_used + j] = wv[j];
                }
            }
            return Ok((sel, w_out));
        }
        // LAUNCH-STRUCTURE STAGE 1 (2026-07-05): fused router DEFAULT ON (BW24_FUSED_ROUTER=0
        // rollback) via the single-sync pinned readback — softmax arch only; the M3 sigmoid arm
        // above returns before this (host path until a sigmoid fused-router kernel exists).
        if !matches!(std::env::var("BW24_FUSED_ROUTER").as_deref(), Ok("0")) {
            return e.moe_router_topk_host(logits, t, n_expert, n_used);
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

    /// LAUNCH-STRUCTURE STAGE 3: the ZERO-DtoH fully-resident MoE FFN. Caller guarantees the
    /// layer's device pointer row exists (checked under the cache lock). Router top-k runs on
    /// device; sel/w are consumed by the `_dev` matvec twins directly; NOTHING crosses PCIe.
    /// Same numerics as the fused-router + gdec chain (kernel-level bit-identity, see the
    /// MoE PREFILL PAIR-BATCH: host routing (sel/w like the sequential path), then 5 launches
    /// TOTAL per layer (quantize z, gate-pairs, up-pairs, silu, act-quantize, down-pairs,
    /// scatter) regardless of T or expert count. Bit-identity class: per (pair,row) dot =
    /// qmatvec_expert_q8 order; per-token accumulation slot-ordered (scatter kernel).
    fn moe_ffn_pairs(e: &Engine, m: &MoeWeights, z: &CudaSlice<f32>, logits: &CudaSlice<f32>,
                     t: usize, cfg: &ModelConfig)
                     -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let moe = cfg.moe.as_ref().unwrap();
        let n_embd = cfg.n_embd as usize;
        let n_expert = moe.expert_count as usize;
        let n_used = moe.expert_used_count as usize;
        let n_ff_exp = moe.expert_ff_length as usize;
        let dev = m.dev_exps.as_ref().unwrap();

        let (sel_all, w_all) = Self::moe_route(e, logits, t, n_expert, n_used)?;
        let n_pairs = t * n_used;
        // pair arrays: pair p = (token p/n_used, slot p%n_used) — ALREADY slot-ordered per token,
        // so the CSR is trivial: tok_pair_off[tok] = tok*n_used, ids identity.
        let pair_tok: Vec<i32> = (0..n_pairs).map(|p| (p / n_used) as i32).collect();
        let pair_ex:  Vec<i32> = sel_all.iter().map(|&x| x as i32).collect();
        let pair_w:   Vec<f32> = w_all.clone();
        let tok_off:  Vec<i32> = (0..=t).map(|tok| (tok * n_used) as i32).collect();
        let tok_ids:  Vec<i32> = (0..n_pairs as i32).collect();
        let pt = e.htod_i32(&pair_tok)?;
        let px = e.htod_i32(&pair_ex)?;
        let pw = e.htod(&pair_w)?;
        let toff = e.htod_i32(&tok_off)?;
        let tids = e.htod_i32(&tok_ids)?;

        // z quantized ONCE for all tokens; gate/up pair matvecs; silu; act quantize; down; scatter.
        // EXPERT-MAJOR CSR (rung 2): pairs grouped by expert -> the kernel reuses each weight
        // row across the expert's token group (llama-MMQ's core win). Host grouping is O(pairs).
        let mut by_ex: Vec<Vec<i32>> = vec![Vec::new(); n_expert];
        for p in 0..n_pairs { by_ex[pair_ex[p] as usize].push(p as i32); }
        let mut ex_ids: Vec<i32> = Vec::new();
        let mut ex_off: Vec<i32> = vec![0];
        let mut ex_pairs: Vec<i32> = Vec::with_capacity(n_pairs);
        for (ex, list) in by_ex.iter().enumerate() {
            if list.is_empty() { continue; }
            ex_ids.push(ex as i32);
            ex_pairs.extend_from_slice(list);
            ex_off.push(ex_pairs.len() as i32);
        }
        let n_active = ex_ids.len();
        let exi = e.htod_i32(&ex_ids)?;
        let exo = e.htod_i32(&ex_off)?;
        let exp_d = e.htod_i32(&ex_pairs)?;
        let _ = &px;   // pair-major twin keeps it; em path uses CSR

        // INT8-MMA EXPERT MMQ (BW24_MOE_MMA=1, opt-in): the m16n8k16.s8 tensor-core analog of the
        // _dec dp4a kernel (cu/mmq_iq_experts.cu). Same CSR grouping; per-expert matvec runs as a
        // 128x128-tile int8 MMA GEMM over the expert's token group. Weight IQ nibbles decode to int8
        // at tile-load + per-32 float scale; activation is q8_1_mmq (D4, same quant class as dp4a).
        // FP-ORDER differs from dp4a (MMA reduction) — logits SHIFT, gated on argmax/spec/closeness,
        // NOT byte-identity (like the W4A8 path). Requires IQ3_S/IQ4_XS + in_f % 256 == 0.
        // t >= 16 (GEMM_M-class rule): the MMA tile needs token volume (crossover ~200 tok/expert;
        // microbench: dp4a wins at tiny groups). ALSO an exactness requirement — spec verify
        // batches (t=2..K+2) must ride the dp4a path whose FP order matches the T=1 decode chain,
        // else K=1 self-consistency FAILs (caught 2026-07-06: MMA at T=2 flipped a verify argmax).
        // DEFAULT ON (2026-07-06, third flip — this time with the real culprit fixed): the
        // "MMA prime breaks spec" failure was the ROUTER's cuBLASLt n-dependence (d994271),
        // not MMA's own FP order — both this and the k-quant arms were innocent suspects whose
        // margin shifts surfaced the router bug. With the router decode-exact at verify t, the
        // full battery is green with MMA on (spec p1/p2/p3 PASS, raw K=1..8 PASS, argmax MATCH,
        // pp6257 2862 = 2.1x dec). t>=16 floor still required: verify batches must ride dp4a
        // (dispatch parity with the T=1 decode chain). BW24_MOE_MMA=0 rollback;
        // BW24_MOE_MMA_T overrides the floor (bisect seam).
        static MMA_T: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        let mma_t = *MMA_T.get_or_init(|| {
            std::env::var("BW24_MOE_MMA_T").ok().and_then(|v| v.parse().ok()).unwrap_or(16)
        });
        let use_mma = std::env::var("BW24_MOE_MMA").map(|v| v != "0").unwrap_or(true)
            && t >= mma_t
            && q8_expert_dec_supported(m.gate_exps.qtype) && q8_expert_dec_supported(m.up_exps.qtype)
            && q8_expert_dec_supported(m.down_exps.qtype)
            && n_embd % 256 == 0 && n_ff_exp % 256 == 0;
        if use_mma {
            // gate/up: activation = z, token-major over t tokens; pair_tok gathers the routed row.
            let z_scr = e.mmq_iq_quantize_act(z, n_embd, t)?;
            let gate = e.mmq_iq_experts(&dev.ptr_row, 0, n_expert, &exi, &exo, &exp_d, &pt, &z_scr,
                                        n_embd, n_ff_exp, n_active, n_pairs, t,
                                        m.gate_exps.qtype, m.gate_exps.row_bytes)?;
            let up = e.mmq_iq_experts(&dev.ptr_row, 1, n_expert, &exi, &exo, &exp_d, &pt, &z_scr,
                                      n_embd, n_ff_exp, n_active, n_pairs, t,
                                      m.up_exps.qtype, m.up_exps.row_bytes)?;
            let act = e.moe_pairs_silu_mul(&gate, &up, n_pairs * n_ff_exp)?;
            // down: activation = act, pair-major [n_pairs, n_ff_exp]; pair_tok = identity.
            let a_scr = e.mmq_iq_quantize_act(&act, n_ff_exp, n_pairs)?;
            let pair_self: Vec<i32> = (0..n_pairs as i32).collect();
            let pself = e.htod_i32(&pair_self)?;
            let y_down = e.mmq_iq_experts(&dev.ptr_row, 2, n_expert, &exi, &exo, &exp_d, &pself, &a_scr,
                                          n_ff_exp, n_embd, n_active, n_pairs, n_pairs,
                                          m.down_exps.qtype, m.down_exps.row_bytes)?;
            let mut moe_out = e.uninit(t * n_embd)?;
            e.moe_pairs_scatter(&y_down, &pw, &toff, &tids, &mut moe_out, t, n_embd)?;
            if let (Some(gate_shexp), Some(up_shexp), Some(down_shexp)) =
                (&m.gate_shexp, &m.up_shexp, &m.down_shexp)
            {
                let n_ff_sh = gate_shexp.out_features();
                let sg_gate = e.matmul(gate_shexp, z, t)?;
                let sg_up = e.matmul(up_shexp, z, t)?;
                let mut sa = e.uninit(t * n_ff_sh)?;
                Self::ffn_act(e, cfg, &sg_gate, &sg_up, &mut sa, t * n_ff_sh)?;
                let sh = e.matmul(down_shexp, &sa, t)?;
                // shexp gate: qwen35moe sigmoid-gates; M3 has no gate tensor -> weight 1.0.
                let g = match &m.gate_inp_shexp {
                    Some(gate_inp_shexp) => {
                        let gs = e.linear(z, gate_inp_shexp.float_data(), t, n_embd, 1)?;
                        let mut g = e.uninit(t)?;
                        e.sigmoid(&gs, &mut g, t)?;
                        g
                    }
                    None => e.htod(&vec![1.0f32; t])?,
                };
                e.add_scaled_rows(&sh, &g, &mut moe_out, n_embd, t)?;
            }
            return Ok(moe_out);
        }

        // DECODE-ONCE MMQ (rung 3, BW24_MOE_DEC=1 default-on): dequant each weight group once per
        // (row,group) then dp4a across the expert's tokens. _em re-decoded per token (NEUTRAL).
        let dec = std::env::var("BW24_MOE_DEC").map(|v| v != "0").unwrap_or(true);
        let matvec = |proj, exi: &_, exo: &_, exp_d: &_, pt: &_, aq: &_, ad: &_,
                      inf, outf, qtype, rb| -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
            // _dec's decode-once extractors are IQ-only; k-quant expert layers take the _em dot path.
            let dec = dec && q8_expert_dec_supported(qtype);
            if dec { e.moe_pairs_matvec_q8_dec(&dev.ptr_row, proj, exi, exo, exp_d, pt, aq, ad,
                                               inf, outf, n_expert, n_active, n_pairs, qtype, rb) }
            else   { e.moe_pairs_matvec_q8_em (&dev.ptr_row, proj, exi, exo, exp_d, pt, aq, ad,
                                               inf, outf, n_expert, n_active, n_pairs, qtype, rb) }
        };
        let (zq, zd) = e.quantize_q8_1(z, t, n_embd)?;
        let gate = matvec(0, &exi, &exo, &exp_d, &pt, &zq, &zd,
                          n_embd, n_ff_exp, m.gate_exps.qtype, m.gate_exps.row_bytes)?;
        let up = matvec(1, &exi, &exo, &exp_d, &pt, &zq, &zd,
                        n_embd, n_ff_exp, m.up_exps.qtype, m.up_exps.row_bytes)?;
        let act = e.moe_pairs_silu_mul(&gate, &up, n_pairs * n_ff_exp)?;
        let (aq2, ad2) = e.quantize_q8_1(&act, n_pairs, n_ff_exp)?;
        // down consumes PAIR-major activation rows: pair_tok = identity.
        let pair_self: Vec<i32> = (0..n_pairs as i32).collect();
        let pself = e.htod_i32(&pair_self)?;
        let y_down = matvec(2, &exi, &exo, &exp_d, &pself, &aq2, &ad2,
                            n_ff_exp, n_embd, m.down_exps.qtype, m.down_exps.row_bytes)?;
        let mut moe_out = e.uninit(t * n_embd)?;   // scatter fully overwrites per (token,col)
        e.moe_pairs_scatter(&y_down, &pw, &toff, &tids, &mut moe_out, t, n_embd)?;

        // SHARED EXPERT epilogue — same as the other paths.
        // gate_inp_shexp is OPTIONAL: qwen35moe gates the shared expert (sigmoid(gate_inp) x sh);
        // MiniMax-M3 (DeepSeek-V3 class) has NO shexp gate — the shared expert adds directly.
        if let (Some(gate_shexp), Some(up_shexp), Some(down_shexp)) =
            (&m.gate_shexp, &m.up_shexp, &m.down_shexp)
        {
            let n_ff_sh = gate_shexp.out_features();
            let sg_gate = e.matmul(gate_shexp, z, t)?;
            let sg_up = e.matmul(up_shexp, z, t)?;
            let mut sa = e.uninit(t * n_ff_sh)?;
            e.silu_mul(&sg_gate, &sg_up, &mut sa, t * n_ff_sh)?;
            let sh = e.matmul(down_shexp, &sa, t)?;
            // shexp gate: qwen35moe sigmoid-gates; M3 has no gate tensor -> weight 1.0.
            let g = match &m.gate_inp_shexp {
                Some(gate_inp_shexp) => {
                    let gs = e.linear(z, gate_inp_shexp.float_data(), t, n_embd, 1)?;
                    let mut g = e.uninit(t)?;
                    e.sigmoid(&gs, &mut g, t)?;
                    g
                }
                None => e.htod(&vec![1.0f32; t])?,
            };
            e.add_scaled_rows(&sh, &g, &mut moe_out, n_embd, t)?;
        }
        Ok(moe_out)
    }

    /// kernel headers); the shared-expert epilogue is byte-identical to moe_ffn_sequential's.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn moe_ffn_dev(e: &Engine, m: &MoeWeights, z: &CudaSlice<f32>,
                   zq8: Option<&(CudaSlice<i8>, CudaSlice<f32>)>, logits: &CudaSlice<f32>,
                   t: usize, cfg: &ModelConfig, il: u16, max_block: usize)
                   -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let moe = cfg.moe.as_ref().unwrap();
        let n_embd = cfg.n_embd as usize;
        let n_expert = moe.expert_count as usize;
        let n_used = moe.expert_used_count as usize;
        let n_ff_exp = moe.expert_ff_length as usize;

        // device top-k: sel [t, n_used] i32, w [t, n_used] f32 — stays on device.
        let (sel_d, w_d) = e.moe_router_topk(logits, t, n_expert, n_used)?;

        // moe_out rows are FULLY overwritten by moe_down8_fma_dev — uninit (stage-2 rule).
        let mut moe_out = e.uninit(t * n_embd)?;

        // RESIDENT-EXPERTS arm: the pointer row comes from the load-time slab (no cache, no
        // lock). Same kernels/loop as the SLRU arm below — only the row's provenance differs.
        if let Some(dev) = m.dev_exps.as_ref() {
            let q8 = moe_q8_enabled()
                && q8_expert_supported(m.gate_exps.qtype) && q8_expert_supported(m.up_exps.qtype)
                && q8_expert_supported(m.down_exps.qtype);
            // SMALL-M ROWS ARM (BW24_SPEC_M2, lane/spec-m2): batch the verify token loop —
            // ONE batched z-quantize + ONE gate_up rows launch + ONE act quantize + ONE down
            // rows launch (4 launches/layer, was 4t). BIT-IDENTICAL per token to the serial
            // loop below (rows twins = the _v/w8h2v per-token programs on a grid.z token axis;
            // quantize_q8_1 is per-32-block row-independent). Gated to the AUTO kernel modes —
            // a custom BW24_MOE_DEVQ8_GU/DOWN diagnostic run keeps the serial loop so the
            // dispatched kernel stays exactly the env-selected one — and to the w8h2v shape
            // (n_ff_exp==512, n_used<=8), the same contract the AUTO down dispatch keys on.
            let rows_arm = q8 && t > 1 && crate::spec::spec_m2()
                && n_ff_exp == 512 && n_used <= 8
                && std::env::var("BW24_MOE_DEVQ8_GU").map(|v| v.is_empty() || v == "v").unwrap_or(true)
                && std::env::var("BW24_MOE_DEVQ8_DOWN").map(|v| v.is_empty() || v == "w8h2v").unwrap_or(true);
            if rows_arm {
                let (zq, zd) = e.quantize_q8_1(z, t, n_embd)?;
                let act = e.moe_gate_up_silu8_dev_q8_rows(&dev.ptr_row, &sel_d, &zq, &zd, t,
                                                          n_embd, n_ff_exp, n_used, n_expert,
                                                          m.gate_exps.qtype, m.up_exps.qtype,
                                                          m.gate_exps.row_bytes, m.up_exps.row_bytes)?;
                let (aq2, ad2) = e.quantize_q8_1(&act, t * n_used, n_ff_exp)?;
                e.moe_down8_fma_dev_q8_rows(&dev.ptr_row, &sel_d, &w_d, &aq2, &ad2, &mut moe_out,
                                            t, n_ff_exp, n_embd, n_used, n_expert,
                                            m.down_exps.qtype, m.down_exps.row_bytes)?;
            } else {
            for tok in 0..t {
                let zt = z.slice(tok * n_embd..(tok + 1) * n_embd);
                let selt = sel_d.slice(tok * n_used..(tok + 1) * n_used);
                let wt = w_d.slice(tok * n_used..(tok + 1) * n_used);
                let mut dst = moe_out.slice_mut(tok * n_embd..(tok + 1) * n_embd);
                if q8 {
                    let (zq, zd) = match (t, zq8) {
                        (1, Some((q, d))) => (q.clone(), d.clone()),
                        _ => e.quantize_q8_1_view(&zt, 1, n_embd)?,
                    };
                    let act = e.moe_gate_up_silu8_dev_q8(&dev.ptr_row, &selt, &zq, &zd,
                                                         n_embd, n_ff_exp, n_used, n_expert,
                                                         m.gate_exps.qtype, m.up_exps.qtype,
                                                         m.gate_exps.row_bytes, m.up_exps.row_bytes)?;
                    let (aq2, ad2) = e.quantize_q8_1(&act, n_used, n_ff_exp)?;
                    e.moe_down8_fma_dev_q8(&dev.ptr_row, &selt, &wt, &aq2, &ad2, &mut dst,
                                           n_ff_exp, n_embd, n_used, n_expert,
                                           m.down_exps.qtype, m.down_exps.row_bytes)?;
                } else {
                    let act = e.moe_gate_up_silu8_dev(&dev.ptr_row, &selt, &zt, n_embd, n_ff_exp,
                                                      n_used, n_expert,
                                                      m.gate_exps.qtype, m.up_exps.qtype,
                                                      m.gate_exps.row_bytes, m.up_exps.row_bytes)?;
                    e.moe_down8_fma_dev(&dev.ptr_row, &selt, &wt, &act, &mut dst,
                                        n_ff_exp, n_embd, n_used, n_expert,
                                        m.down_exps.qtype, m.down_exps.row_bytes)?;
                }
            }
            }
        } else {
        // Launch under the cache lock: the row borrow lives as long as the closure, and the
        // lock covers only launch ISSUE (µs), same policy as moe_cached_gemm.
        // Q8 ARM PARITY (2026-07-06): the SLRU arm ran the f32-dequant _dev kernels only —
        // 80us/launch vs the q8 twins' 15us on the SAME shapes (fixed-build profile: 228
        // f32 launches = 36ms of the 64-tok window). Same q8 gate + kernels as the resident
        // arm above; BW24_MOE_Q8=0 restores the byte-identical f32 path.
        let q8 = moe_q8_enabled()
            && q8_expert_supported(m.gate_exps.qtype) && q8_expert_supported(m.up_exps.qtype)
            && q8_expert_supported(m.down_exps.qtype);
        e.with_moe_cache(max_block, |c, eng| {
            let row = c.layer_dev_row(il, n_expert, eng)?
                .ok_or("moe_ffn_dev: layer row vanished under the lock")?;
            for tok in 0..t {
                let zt = z.slice(tok * n_embd..(tok + 1) * n_embd);
                let selt = sel_d.slice(tok * n_used..(tok + 1) * n_used);
                let wt = w_d.slice(tok * n_used..(tok + 1) * n_used);
                let mut dst = moe_out.slice_mut(tok * n_embd..(tok + 1) * n_embd);
                if q8 {
                    let (zq, zd) = match (t, zq8) {
                        (1, Some((q, d))) => (q.clone(), d.clone()),
                        _ => eng.quantize_q8_1_view(&zt, 1, n_embd)?,
                    };
                    let act = eng.moe_gate_up_silu8_dev_q8(row, &selt, &zq, &zd,
                                                           n_embd, n_ff_exp, n_used, n_expert,
                                                           m.gate_exps.qtype, m.up_exps.qtype,
                                                           m.gate_exps.row_bytes, m.up_exps.row_bytes)?;
                    let (aq2, ad2) = eng.quantize_q8_1(&act, n_used, n_ff_exp)?;
                    eng.moe_down8_fma_dev_q8(row, &selt, &wt, &aq2, &ad2, &mut dst,
                                             n_ff_exp, n_embd, n_used, n_expert,
                                             m.down_exps.qtype, m.down_exps.row_bytes)?;
                } else {
                    let act = eng.moe_gate_up_silu8_dev(row, &selt, &zt, n_embd, n_ff_exp,
                                                        n_used, n_expert,
                                                        m.gate_exps.qtype, m.up_exps.qtype,
                                                        m.gate_exps.row_bytes, m.up_exps.row_bytes)?;
                    eng.moe_down8_fma_dev(row, &selt, &wt, &act, &mut dst,
                                          n_ff_exp, n_embd, n_used, n_expert,
                                          m.down_exps.qtype, m.down_exps.row_bytes)?;
                }
            }
            // instrumentation parity with the host paths (3 blocks/expert-slot, all hits).
            c.hits += (t * 3 * n_used) as u64;
            Ok(())
        })?;
        }

        // SHARED EXPERT epilogue — byte-identical to moe_ffn_sequential step 3 (incl. its Q8
        // TRUNK-FUSION arm: fused2 is bit-identical to the two matmul calls per (tensor,row)).
        // gate_inp_shexp is OPTIONAL: qwen35moe gates the shared expert (sigmoid(gate_inp) x sh);
        // MiniMax-M3 (DeepSeek-V3 class) has NO shexp gate — the shared expert adds directly.
        if let (Some(gate_shexp), Some(up_shexp), Some(down_shexp)) =
            (&m.gate_shexp, &m.up_shexp, &m.down_shexp)
        {
            let n_ff_sh = gate_shexp.out_features();
            // verify-t (2..15) decode-exact arm: this fn now serves the spec verify batches
            // (pairs gate moved to t>=PRIME_MIN_T), so the shexp chain must match t==1 per col.
            let verify_t = t > 1 && t < PRIME_MIN_T;
            let (sg_gate, sg_up) = if t == 1 {
                match e.matmul_q8_fused2_x(gate_shexp, up_shexp, z)? {
                    Some(pair) => pair,
                    None => (e.matmul(gate_shexp, z, t)?, e.matmul(up_shexp, z, t)?),
                }
            } else if verify_t {
                (e.matmul_decode_exact(gate_shexp, z, t)?, e.matmul_decode_exact(up_shexp, z, t)?)
            } else {
                (e.matmul(gate_shexp, z, t)?, e.matmul(up_shexp, z, t)?)
            };
            let mut sa = e.uninit(t * n_ff_sh)?;  // silu_mul fully overwrites
            e.silu_mul(&sg_gate, &sg_up, &mut sa, t * n_ff_sh)?;
            let sh = if verify_t { e.matmul_decode_exact(down_shexp, &sa, t)? }
                     else { e.matmul(down_shexp, &sa, t)? };
            // shexp gate: qwen35moe sigmoid-gates; M3 has no gate tensor -> weight 1.0.
            let g = match &m.gate_inp_shexp {
                Some(gate_inp_shexp) => {
                    let gs = if verify_t {
                e.linear_decode_exact(z, gate_inp_shexp.float_data(), t, n_embd, 1)?
            } else {
                e.linear(z, gate_inp_shexp.float_data(), t, n_embd, 1)?
            };
                    let mut g = e.uninit(t)?;
                    e.sigmoid(&gs, &mut g, t)?;
                    g
                }
                None => e.htod(&vec![1.0f32; t])?,
            };
            e.add_scaled_rows(&sh, &g, &mut moe_out, n_embd, t)?;
        }

        Ok(moe_out)
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
    /// q8 twin of moe_gdec_token (dp4a arc): same residency check + 2-launch shape; the fused
    /// kernels consume the pre-quantized z-row and re-quantize act per slot batch.
    #[allow(clippy::too_many_arguments)]
    fn moe_gdec_token_q8(e: &Engine, m: &MoeWeights, il: u16, max_block: usize,
                      zq: &CudaSlice<i8>, zd: &CudaSlice<f32>, sel: &[u32], w: &[f32],
                      moe_out: &mut CudaSlice<f32>, tok: usize,
                      n_embd: usize, n_ff_exp: usize, n_used: usize)
                      -> Result<bool, Box<dyn std::error::Error>> {
        use crate::moe_cache::{BlockId, PROJ_GATE, PROJ_UP, PROJ_DOWN};
        use cudarc::driver::DevicePtr;
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
            c.hits += (3 * n_used) as u64;
            Ok(Some((g, u, d)))
        })?;
        let Some((g, u, d)) = ptrs else { return Ok(false) };
        let mut wv = [0f32; 8];
        wv[..n_used].copy_from_slice(w);
        let act = e.moe_gate_up_silu8_q8(crate::WPtr8(g), crate::WPtr8(u), zq, zd,
                                         n_embd, n_ff_exp, n_used,
                                         m.gate_exps.qtype, m.up_exps.qtype,
                                         m.gate_exps.row_bytes, m.up_exps.row_bytes)?;
        // per-slot act quantize: [n_used, n_ff] rows in one quantize launch.
        let (aq2, ad2) = e.quantize_q8_1(&act, n_used, n_ff_exp)?;
        let mut dst = moe_out.slice_mut(tok * n_embd..(tok + 1) * n_embd);
        e.moe_down8_fma_q8(crate::WPtr8(d), crate::F32x8(wv), &aq2, &ad2, &mut dst,
                           n_ff_exp, n_embd, n_used,
                           m.down_exps.qtype, m.down_exps.row_bytes)?;
        Ok(true)
    }

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
    /// q8 twin of moe_cached_gemm: same dispatch/slot mechanics, dp4a expert kernel.
    fn moe_cached_gemm_q8(e: &Engine, il: u16, proj: u8, ex: usize, m: &MoeWeights,
                          max_block: usize, aq: &CudaSlice<i8>, ad: &CudaSlice<f32>)
                          -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        use crate::moe_cache::{BlockId, DispatchSlot, PROJ_GATE, PROJ_UP};
        let exps = match proj { PROJ_GATE => &m.gate_exps, PROJ_UP => &m.up_exps, _ => &m.down_exps };
        let len = exps.expert_stride;
        let id = BlockId::new(il, proj, ex as u16);
        let host_bytes = exps.expert_bytes(ex);
        e.with_moe_cache(max_block, |c, eng| {
            let slot = c.dispatch(id, host_bytes, eng)?;
            let DispatchSlot::Resident(sl) = slot;
            let buf = c.slot(sl);
            eng.qmatvec_expert_q8(buf, 0..len, aq, ad, 1, exps.in_f, exps.out_f, exps.qtype, exps.row_bytes)
        })
    }

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
            let DispatchSlot::Resident(sl) = slot;
            let buf = c.slot(sl);
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
        let (sel_all, w_all) = if let Some(m3) = cfg.m3.as_ref().filter(|m| m.sigmoid_routing) {
            Self::moe_route_cfg(e, &logits, t, n_expert, n_used,
                                m.exp_probs_b.as_deref(), Some(m3.routed_scaling_factor))?
        } else {
            Self::moe_route(e, &logits, t, n_expert, n_used)?
        };

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
        // Processing ORDER: DESCENDING m_e (biggest token batches first) — the concluded winner
        // (rig5090 2026-07-04, the ascending-id arm and its BW24_MOE_ORDER seam removed): desc is
        // a first-forward win at partial cache capacity — the hot (big-m_e) experts are admitted
        // to the SLRU before the small-m_e tail can pollute it, so residency converges in ONE
        // forward instead of several: auto-cache T=501 126.9 -> 169.9 tok/s (1.34x), cap512
        // 119.6 -> 160.8 (and kills the rep-to-rep bimodal); wash (<2%) at cap64 pure-spill and
        // at long prompts where every expert stages regardless. Order is FREE to change without
        // breaking the byte-identity gate: the slot scheme pins each token's accumulation order
        // regardless of expert processing order (the whole point of the slots).
        let mut order: Vec<usize> =
            (0..n_expert).filter(|&ex| !groups[ex].tok_indices.is_empty()).collect();
        order.sort_by(|&a, &b| groups[b].tok_indices.len()
            .cmp(&groups[a].tok_indices.len()).then(a.cmp(&b)));
        let mut m_dist: Vec<usize> = Vec::new();  // for stats
        for ex in order {
            let grp = &groups[ex];
            let m_e = grp.tok_indices.len();
            m_dist.push(m_e);

            // Upload index/weight arrays to device. The down-proj per-expert macro-scale
            // (ModelOpt weight_scale_2) folds into the scatter weights — post-matmul linear,
            // same fold as the sequential loop's `w[j] * macro_scale(ex)`. 1.0 for GGUF experts.
            let tok_idx_d = e.htod_i32(&grp.tok_indices)?;
            let slot_idx_d = e.htod_i32(&grp.slot_indices)?;
            let dmac = m.down_exps.macro_scale(ex);
            let weight_d = if dmac == 1.0 { e.htod(&grp.weights)? } else {
                let scaled: Vec<f32> = grp.weights.iter().map(|&w| w * dmac).collect();
                e.htod(&scaled)?
            };

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
                // SiLU-MUL activation (per-expert macro-scales folded).
                let mut act = e.zeros(m_e * n_ff_exp)?;
                Self::ffn_act_scaled(e, cfg, &gate, &up,
                    m.gate_exps.macro_scale(ex), m.up_exps.macro_scale(ex), &mut act, m_e * n_ff_exp)?;
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
                // SiLU-MUL activation (per-expert macro-scales folded).
                let mut act = e.zeros(m_e * n_ff_exp)?;
                Self::ffn_act_scaled(e, cfg, &gate, &up,
                    m.gate_exps.macro_scale(ex), m.up_exps.macro_scale(ex), &mut act, m_e * n_ff_exp)?;
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
        // gate_inp_shexp is OPTIONAL: qwen35moe gates the shared expert (sigmoid(gate_inp) x sh);
        // MiniMax-M3 (DeepSeek-V3 class) has NO shexp gate — the shared expert adds directly.
        if let (Some(gate_shexp), Some(up_shexp), Some(down_shexp)) =
            (&m.gate_shexp, &m.up_shexp, &m.down_shexp)
        {
            let n_ff_sh = gate_shexp.out_features();
            let sg_gate = e.matmul(gate_shexp, z, t)?;
            let sg_up = e.matmul(up_shexp, z, t)?;
            let mut sa = e.zeros(t * n_ff_sh)?;
            Self::ffn_act(e, cfg, &sg_gate, &sg_up, &mut sa, t * n_ff_sh)?;
            let sh = e.matmul(down_shexp, &sa, t)?;
            // shexp gate: qwen35moe sigmoid-gates; M3 has no gate tensor -> weight 1.0.
            let g = match &m.gate_inp_shexp {
                Some(gate_inp_shexp) => {
                    let gs = e.linear(z, gate_inp_shexp.float_data(), t, n_embd, 1)?;
                    let mut g = e.uninit(t)?;
                    e.sigmoid(&gs, &mut g, t)?;
                    g
                }
                None => e.htod(&vec![1.0f32; t])?,
            };
            e.add_scaled_rows(&sh, &g, &mut moe_out, n_embd, t)?;
        }

        Ok(moe_out)
    }
}
