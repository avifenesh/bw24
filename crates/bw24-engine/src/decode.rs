//! Incremental decode (T=1) with the dual cache + greedy generation loop. Serves end-to-end.
//! Reuses the validated kernels; threads KV (full-attn) and conv/SSM state (linear-attn) across steps.

use cudarc::driver::CudaSlice;
use std::collections::HashMap;
use crate::Engine;
use crate::hybrid::{HybridModel, Mixer, FullAttnLayer, LinearAttnLayer};
use crate::cache::{Cache, RecurLayer};
use crate::forward::argmax;

/// Persistent CUDA-graph decode state (CUDA-GRAPH-PLAN Phase 3). Holds the device-resident counters
/// the captured graph reads/writes (`token_d` = current/next token id, `pos_d` = rope position) — both
/// at FIXED addresses baked into every captured graph — plus the per-`t_kv`-bucket graph cache. The
/// bucket key is the eager `(fa_vec, n_splits)` pair (see `Engine::fa_bucket_key`): every t_kv that
/// maps to the same key reproduces eager's split geometry, so one captured graph replays bit-identically
/// for the whole bucket. A new key triggers a re-capture (n_splits changes ~every 64 tokens).
pub struct GraphDecodeState {
    pub token_d: CudaSlice<u32>,        // [1] resident next-token id (argmax writes, embed reads)
    pub pos_d: CudaSlice<i32>,          // [1] resident rope position counter
    pub graphs: HashMap<(bool, usize), cudarc::driver::CudaGraph>,
    pub bucket_max: HashMap<(bool, usize), usize>,  // bucket key -> bucket_max fed to the capture
    pub captures: usize,                // count of (re)captures, for reporting
}

impl GraphDecodeState {
    pub fn new(e: &Engine) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(GraphDecodeState {
            token_d: e.stream().clone_htod(&[0u32])?,
            pos_d: e.htod_i32(&[0])?,
            graphs: HashMap::new(),
            bucket_max: HashMap::new(),
            captures: 0,
        })
    }
}

/// Generation parameters for the reusable serving API (`generate_with`).
#[derive(Clone, Debug)]
pub struct GenParams {
    pub max_new: usize,            // hard cap on generated tokens
    pub max_ctx: Option<usize>,    // context-length guard; None => prompt+max_new+8
    pub eos: Vec<u32>,             // stop on any of these token ids (eos/eog + specials)
}
impl Default for GenParams {
    fn default() -> Self { GenParams { max_new: 128, max_ctx: None, eos: Vec::new() } }
}

/// Why generation stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason { Eos, MaxNew, ContextFull, Callback }

/// Result of `generate_with`: the generated token ids + why it stopped.
pub struct GenOutput {
    pub tokens: Vec<u32>,
    pub stop_reason: StopReason,
}

impl HybridModel {
    /// One decode step for `token` at cache.pos; returns logits [n_vocab] (host f32). Advances cache.
    pub fn decode_step(&self, e: &Engine, token: u32, cache: &mut Cache) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        Ok(self.decode_step_h(e, token, cache)?.0)
    }

    /// EAGLE3 aux-hidden capture (EAGLE-PLAN N1): one decode step that ALSO returns the trunk
    /// residual-stream `x` taken AFTER each of the blocks in `aux_layers` (the EAGLE3 encoder feeds
    /// these 3 layer hiddens through `fc`). Returns (logits[n_vocab] host, aux: Vec<[n_embd] dev>),
    /// one device buffer per requested aux layer, in `aux_layers` order. The captured tensor is the
    /// residual `x` produced by that block (`x2` at the loop tail), cloned before the next block
    /// overwrites it — cheap (one clone_dtod of [n_embd] per aux layer). T=1 decode regime.
    pub fn decode_step_aux(&self, e: &Engine, token: u32, cache: &mut Cache, aux_layers: &[usize])
                           -> Result<(Vec<f32>, Vec<CudaSlice<f32>>), Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let n_embd = cfg.n_embd as usize;
        let eps = cfg.rms_eps;
        let pos = cache.pos;
        let pos_d = e.htod_i32(&[pos as i32])?;

        let mut x = e.htod(&self.embd.gather(n_embd, &[token]))?;
        let mut aux: Vec<CudaSlice<f32>> = Vec::with_capacity(aux_layers.len());

        for (il, layer) in self.layers.iter().enumerate() {
            let mut h = e.uninit(n_embd)?;
            e.rms_norm(&x, layer.attn_norm.float_data(), &mut h, n_embd, 1, eps)?;
            let mixed = match &layer.mixer {
                Mixer::Full(fa) => self.full_attn_decode(e, fa, &h, &pos_d, pos, cache, il)?,
                Mixer::Linear(la) => self.linear_attn_decode(e, la, &h, cache, il)?,
            };
            let mut x1 = e.uninit(n_embd)?;
            e.add(&x, &mixed, &mut x1, n_embd)?;
            let mut z = e.uninit(n_embd)?;
            e.rms_norm(&x1, layer.post_attn_norm.float_data(), &mut z, n_embd, 1, eps)?;
            let ffn_out = match &layer.ffn {
                crate::hybrid::Ffn::Dense { ffn_gate, ffn_up, ffn_down } => {
                    let n_ff = ffn_gate.out_features();
                    let (gate, up) = if e.uses_q8_1_fast(ffn_gate) && e.uses_q8_1_fast(ffn_up) {
                        let (zq, zd) = e.quantize_q8_1(&z, 1, n_embd)?;
                        (e.matmul_pre(ffn_gate, &zq, &zd, &z, 1)?, e.matmul_pre(ffn_up, &zq, &zd, &z, 1)?)
                    } else {
                        (e.matmul(ffn_gate, &z, 1)?, e.matmul(ffn_up, &z, 1)?)
                    };
                    let mut act = e.uninit(n_ff)?;
                    e.silu_mul(&gate, &up, &mut act, n_ff)?;
                    e.matmul(ffn_down, &act, 1)?
                }
                crate::hybrid::Ffn::Moe(m) => self.moe_ffn_il(e, m, &z, 1, il as u16)?,
            };
            let mut x2 = e.uninit(n_embd)?;
            e.add(&x1, &ffn_out, &mut x2, n_embd)?;
            // EAGLE3 N1: capture this block's residual output if it is an aux layer.
            if aux_layers.contains(&il) { aux.push(e.clone_dtod(&x2)?); }
            x = x2;
        }
        // re-order aux to match aux_layers order (contains() pushes in il order; aux_layers is the
        // canonical order the encoder concats in — they coincide since aux_layers is ascending).
        let mut hn = e.uninit(n_embd)?;
        e.rms_norm(&x, self.output_norm.float_data(), &mut hn, n_embd, 1, eps)?;
        let logits = e.matmul(&self.output, &hn, 1)?;
        let host = e.dtoh(&logits)?;
        cache.pos += 1;
        Ok((host, aux))
    }

    /// Like `decode_step`, but ALSO returns the trunk's hidden state `x` taken BEFORE the final
    /// `output_norm` (MTP-PLAN §A: this is `h_seed` for the NextN head). Device buffer [n_embd].
    pub fn decode_step_h(&self, e: &Engine, token: u32, cache: &mut Cache)
                         -> Result<(Vec<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let n_embd = cfg.n_embd as usize;
        let eps = cfg.rms_eps;
        let pos = cache.pos;
        let pos_d = e.htod_i32(&[pos as i32])?;

        // embed the single token -> [1, n_embd]
        let mut x = e.htod(&self.embd.gather(n_embd, &[token]))?;

        for (il, layer) in self.layers.iter().enumerate() {
            let mut h = e.uninit(n_embd)?;
            e.rms_norm(&x, layer.attn_norm.float_data(), &mut h, n_embd, 1, eps)?;

            let mixed = match &layer.mixer {
                Mixer::Full(fa) => self.full_attn_decode(e, fa, &h, &pos_d, pos, cache, il)?,
                Mixer::Linear(la) => self.linear_attn_decode(e, la, &h, cache, il)?,
            };

            let mut x1 = e.uninit(n_embd)?;
            e.add(&x, &mixed, &mut x1, n_embd)?;

            let mut z = e.uninit(n_embd)?;
            e.rms_norm(&x1, layer.post_attn_norm.float_data(), &mut z, n_embd, 1, eps)?;
            let ffn_out = match &layer.ffn {
                crate::hybrid::Ffn::Dense { ffn_gate, ffn_up, ffn_down } => {
                    let n_ff = ffn_gate.out_features();
                    // gate and up share input `z` (in_f = n_embd) — quantize q8_1 ONCE, feed both,
                    // instead of re-quantizing the identical row twice (quantize_q8_1 was 13.5% of decode).
                    let (gate, up) = if e.uses_q8_1_fast(ffn_gate) && e.uses_q8_1_fast(ffn_up) {
                        let (zq, zd) = e.quantize_q8_1(&z, 1, n_embd)?;
                        (e.matmul_pre(ffn_gate, &zq, &zd, &z, 1)?, e.matmul_pre(ffn_up, &zq, &zd, &z, 1)?)
                    } else {
                        (e.matmul(ffn_gate, &z, 1)?, e.matmul(ffn_up, &z, 1)?)
                    };
                    let mut act = e.uninit(n_ff)?;
                    e.silu_mul(&gate, &up, &mut act, n_ff)?;
                    e.matmul(ffn_down, &act, 1)?
                }
                crate::hybrid::Ffn::Moe(m) => self.moe_ffn_il(e, m, &z, 1, il as u16)?,
            };
            let mut x2 = e.uninit(n_embd)?;
            e.add(&x1, &ffn_out, &mut x2, n_embd)?;
            x = x2;
        }

        // h_seed = trunk hidden BEFORE output_norm (the NextN head's `h` input, §A).
        let h_seed = e.clone_dtod(&x)?;
        let mut hn = e.uninit(n_embd)?;
        e.rms_norm(&x, self.output_norm.float_data(), &mut hn, n_embd, 1, eps)?;
        let logits = e.matmul(&self.output, &hn, 1)?;
        let host = e.dtoh(&logits)?;
        cache.pos += 1;
        Ok((host, h_seed))
    }

    /// DEVICE-COUNTER decode step (CUDA-GRAPH-PLAN Phase 2). A clone of `decode_step_h` that removes
    /// the two per-step VARYING host kernel-args by reading them from device counters:
    ///   1. the KV-append write slot  -> per-layer `kvl.len_d` (device i32[1])
    ///   2. the fa_decode t_kv bound   -> the same `kvl.len_d` after `inc_seqlen`
    /// plus it keeps the token id + rope pos DEVICE-RESIDENT (embed_gather_device, device rope pos,
    /// argmax_token_device). NO graph capture yet — runs the kernels eagerly through the counter
    /// path. Must be BIT-IDENTICAL to `decode_step_h`'s token stream (the gate).
    ///
    /// Args: `token_d` = resident device token id [1] (this step's input token); `pos_d` = resident
    /// device rope pos i32[1] (== cache.pos at entry; INCREMENTED in-path); `embd_gpu` = resident embed
    /// table; (qt,row_bytes) from EmbedHost::qt_and_row_bytes. Returns the NEXT token id device buffer.
    /// `cache.pos` and each `kvl.len`/`kvl.len_d` are advanced to match `decode_step_h`.
    pub fn decode_step_dc(&self, e: &Engine, token_d: &CudaSlice<u32>, pos_d: &mut CudaSlice<i32>,
                          embd_gpu: &CudaSlice<u8>, embd_qt: i32, embd_row_bytes: usize,
                          cache: &mut Cache, n_vocab: usize)
                          -> Result<CudaSlice<u32>, Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let n_embd = cfg.n_embd as usize;
        let eps = cfg.rms_eps;

        // embed the single (DEVICE-resident) token -> [1, n_embd], no host round-trip of the id.
        let mut x = e.embed_gather_device(embd_gpu, token_d, n_embd, embd_qt, embd_row_bytes)?;

        for (il, layer) in self.layers.iter().enumerate() {
            let mut h = e.uninit(n_embd)?;
            e.rms_norm(&x, layer.attn_norm.float_data(), &mut h, n_embd, 1, eps)?;

            let mixed = match &layer.mixer {
                Mixer::Full(fa) => self.full_attn_decode_dc(e, fa, &h, pos_d, cache, il)?,
                Mixer::Linear(la) => self.linear_attn_decode(e, la, &h, cache, il)?,
            };

            let mut x1 = e.uninit(n_embd)?;
            e.add(&x, &mixed, &mut x1, n_embd)?;

            let mut z = e.uninit(n_embd)?;
            e.rms_norm(&x1, layer.post_attn_norm.float_data(), &mut z, n_embd, 1, eps)?;
            let ffn_out = match &layer.ffn {
                crate::hybrid::Ffn::Dense { ffn_gate, ffn_up, ffn_down } => {
                    let n_ff = ffn_gate.out_features();
                    let (gate, up) = if e.uses_q8_1_fast(ffn_gate) && e.uses_q8_1_fast(ffn_up) {
                        let (zq, zd) = e.quantize_q8_1(&z, 1, n_embd)?;
                        (e.matmul_pre(ffn_gate, &zq, &zd, &z, 1)?, e.matmul_pre(ffn_up, &zq, &zd, &z, 1)?)
                    } else {
                        (e.matmul(ffn_gate, &z, 1)?, e.matmul(ffn_up, &z, 1)?)
                    };
                    let mut act = e.uninit(n_ff)?;
                    e.silu_mul(&gate, &up, &mut act, n_ff)?;
                    e.matmul(ffn_down, &act, 1)?
                }
                crate::hybrid::Ffn::Moe(m) => self.moe_ffn_il(e, m, &z, 1, il as u16)?,
            };
            let mut x2 = e.uninit(n_embd)?;
            e.add(&x1, &ffn_out, &mut x2, n_embd)?;
            x = x2;
        }

        let mut hn = e.uninit(n_embd)?;
        e.rms_norm(&x, self.output_norm.float_data(), &mut hn, n_embd, 1, eps)?;
        let logits = e.matmul(&self.output, &hn, 1)?;
        // device argmax -> next token id stays resident (no logits dtoh).
        let next_tok = e.argmax_token_device(&logits, n_vocab)?;
        // advance rope pos counter on-device (replaces the per-step htod_i32(&[pos])).
        e.inc_seqlen(pos_d)?;
        cache.pos += 1;
        Ok(next_tok)
    }

    /// CAPTURE body for CUDA-graph replay (CUDA-GRAPH-PLAN Phase 3). One full decode step enqueued
    /// entirely on `e.stream()` with ZERO host sync and ZERO per-step varying host kernel-args:
    ///   - embed reads the PERSISTENT device `token_d` (last step's argmax), writes scratch `x`.
    ///   - full-attn layers size n_splits from `bucket_max` (fixed for this capture); the kernel reads
    ///     the ACTUAL t_kv from the device counter `kvl.len_d`. KV append + device-counter inc happen
    ///     in-graph. The host `kvl.len`/`cache.pos` are NOT advanced here (the driver advances the host
    ///     mirrors once per replay; only the DEVICE counters advance inside the graph).
    ///   - linear-attn layers use the persistent-state variant (copy-back, stable pointers).
    ///   - lm_head -> `argmax_logits_f32_to_u32` writes the next id into the PERSISTENT `token_d`.
    ///   - `inc_seqlen(pos_d)` advances the rope-pos device counter in-graph.
    /// Captured ONCE per `bucket_max`; replayed for every t_kv in that bucket. Bit-identical to eager
    /// when `bucket_max` reproduces eager's n_splits for the replayed t_kv (the bucket-key contract).
    pub fn decode_step_dc_cap(&self, e: &Engine, token_d: &mut CudaSlice<u32>, pos_d: &mut CudaSlice<i32>,
                          embd_gpu: &CudaSlice<u8>, embd_qt: i32, embd_row_bytes: usize,
                          cache: &mut Cache, n_vocab: usize, bucket_max: usize)
                          -> Result<(), Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let n_embd = cfg.n_embd as usize;
        let eps = cfg.rms_eps;

        let mut x = e.embed_gather_device(embd_gpu, token_d, n_embd, embd_qt, embd_row_bytes)?;

        for (il, layer) in self.layers.iter().enumerate() {
            let mut h = e.uninit(n_embd)?;
            e.rms_norm(&x, layer.attn_norm.float_data(), &mut h, n_embd, 1, eps)?;
            let mixed = match &layer.mixer {
                Mixer::Full(fa) => self.full_attn_decode_dc_cap(e, fa, &h, pos_d, cache, il, bucket_max)?,
                Mixer::Linear(la) => self.linear_attn_decode_cap(e, la, &h, cache, il)?,
            };
            let mut x1 = e.uninit(n_embd)?;
            e.add(&x, &mixed, &mut x1, n_embd)?;
            let mut z = e.uninit(n_embd)?;
            e.rms_norm(&x1, layer.post_attn_norm.float_data(), &mut z, n_embd, 1, eps)?;
            let ffn_out = match &layer.ffn {
                crate::hybrid::Ffn::Dense { ffn_gate, ffn_up, ffn_down } => {
                    let n_ff = ffn_gate.out_features();
                    let (gate, up) = if e.uses_q8_1_fast(ffn_gate) && e.uses_q8_1_fast(ffn_up) {
                        let (zq, zd) = e.quantize_q8_1(&z, 1, n_embd)?;
                        (e.matmul_pre(ffn_gate, &zq, &zd, &z, 1)?, e.matmul_pre(ffn_up, &zq, &zd, &z, 1)?)
                    } else {
                        (e.matmul(ffn_gate, &z, 1)?, e.matmul(ffn_up, &z, 1)?)
                    };
                    let mut act = e.uninit(n_ff)?;
                    e.silu_mul(&gate, &up, &mut act, n_ff)?;
                    e.matmul(ffn_down, &act, 1)?
                }
                crate::hybrid::Ffn::Moe(m) => self.moe_ffn_il(e, m, &z, 1, il as u16)?,
            };
            let mut x2 = e.uninit(n_embd)?;
            e.add(&x1, &ffn_out, &mut x2, n_embd)?;
            x = x2;
        }

        let mut hn = e.uninit(n_embd)?;
        e.rms_norm(&x, self.output_norm.float_data(), &mut hn, n_embd, 1, eps)?;
        let logits = e.matmul(&self.output, &hn, 1)?;
        // argmax into the PERSISTENT token_d (next step's embed reads it) — same buffer pointer baked
        // at capture, written each replay, so the token id never round-trips to host in steady state.
        e.argmax_token_device_into(&logits, token_d, n_vocab)?;
        e.inc_seqlen(pos_d)?;
        Ok(())
    }

    /// CUDA-GRAPH decode driver (CUDA-GRAPH-PLAN Phase 3). Primes the prompt EAGERLY (device-counter
    /// `decode_step_dc`, advancing host + device counters together), then generates `max_new` tokens by
    /// CUDA-graph REPLAY: per step it picks the t_kv bucket key, captures a graph on first sight of that
    /// key (re-using the SAME persistent counters/cache so replays continue the sequence), and replays.
    /// The argmax-written next token stays device-resident in `gs.token_d`; we read back only the [1]
    /// u32 after each launch (the gate compares it; a real server can defer this). Returns the generated
    /// token ids. Greedy. Bit-identical to eager `decode_step` (the gate).
    ///
    /// CAPTURE STATE HYGIENE: `capture_graph` runs the step body 3x (2 warmup + 1 capture), each of
    /// which mutates the device KV/conv/ssm/counter state. We SNAPSHOT the cache + device counters +
    /// token id before capturing and RESTORE them after, so the 3 throwaway runs leave zero residue and
    /// replay resumes from the true pre-capture state.
    pub fn generate_graph(&self, e: &Engine, gs: &mut GraphDecodeState, prompt: &[u32], max_new: usize)
                          -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let n_embd = self.cfg.n_embd as usize;
        let head_dim = self.cfg.head_dim_k as usize;
        let (qt, row_bytes) = self.embd.qt_and_row_bytes(n_embd);

        // EVENT TRACKING OFF for the WHOLE graph-decode session. cudarc records a per-CudaSlice event
        // (the Engine is in multi-stream mode via copy_stream) and inserts `stream.wait(event)` on every
        // kernel arg whose buffer was touched — those waits are illegal inside a capture region. The
        // captured decode step is strictly single-stream, so this tracking is unnecessary. Disable it
        // BEFORE allocating ANY buffer the captured graph will reference (cache, embd, counters,
        // scratch) so none of them carry events. SAFETY: decode-dc touches only gpu.stream.
        let was_tracking = e.ctx().is_event_tracking();
        if was_tracking { unsafe { e.ctx().disable_event_tracking(); } }
        let r = self.generate_graph_inner(e, gs, prompt, max_new, n_embd, head_dim, qt, row_bytes);
        if was_tracking { unsafe { e.ctx().enable_event_tracking(); } }
        r
    }

    fn generate_graph_inner(&self, e: &Engine, gs: &mut GraphDecodeState, prompt: &[u32],
                            max_new: usize, n_embd: usize, head_dim: usize, qt: i32, row_bytes: usize)
                            -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let _ = n_embd;
        let embd_gpu = e.upload_u8(&self.embd.raw)?;
        let max_ctx = prompt.len() + max_new + 8;
        let mut cache = Cache::new(e, &self.cfg, max_ctx)?;

        // (Re)create the persistent counters tracking-OFF so they carry no events (the caller's
        // GraphDecodeState::new may have allocated them with tracking on).
        gs.pos_d = e.htod_i32(&[0])?;
        gs.token_d = e.stream().clone_htod(&[0u32])?;
        // PRIME eagerly: feed each prompt token; advance host + device counters together.
        let mut next_in = 0u32;
        for &tok in prompt {
            e.set_u32_one(&mut gs.token_d, tok)?;
            let nt = self.decode_step_dc(e, &gs.token_d, &mut gs.pos_d, &embd_gpu, qt, row_bytes, &mut cache, /*n_vocab*/ self.output.out_features())?;
            next_in = e.dtoh_u32_one(&nt)?;
        }
        // gs.token_d now must hold the first generated INPUT token (= argmax of the last prime step).
        e.set_u32_one(&mut gs.token_d, next_in)?;

        let n_vocab = self.output.out_features();
        let mut out = Vec::with_capacity(max_new);
        for _ in 0..max_new {
            // t_kv for THIS step = (cache.pos)+1 (the new token's KV length after append).
            let t_kv = cache.pos + 1;
            let key = e.fa_bucket_key(t_kv, head_dim);
            if !gs.graphs.contains_key(&key) {
                // bucket_max = t_kv that produces this key's n_splits; t_kv itself works (same key).
                let bucket_max = t_kv;
                // --- snapshot device + host state so the 3 capture-warmup runs leave no residue ---
                let snap = cache.snapshot(e)?;
                let pos_save = e.dtoh_i32_one(&gs.pos_d)?;
                let len_save: Vec<Option<i32>> = cache.kv.iter()
                    .map(|k| k.as_ref().map(|kvl| { e.dtoh_i32_one(&kvl.len_d).unwrap() })).collect();
                let tok_save = e.dtoh_u32_one(&gs.token_d)?;

                // capture (runs the body 3x on the REAL cache + counters).
                let graph = {
                    // split borrows: pull the fields out so the closure can take &mut of each.
                    let GraphDecodeState { token_d, pos_d, .. } = gs;
                    let token_d: &mut CudaSlice<u32> = token_d;
                    let pos_d: &mut CudaSlice<i32> = pos_d;
                    let cache_ref = &mut cache;
                    let embd_ref = &embd_gpu;
                    e.capture_graph(|e| {
                        self.decode_step_dc_cap(e, token_d, pos_d, embd_ref, qt, row_bytes,
                                                cache_ref, n_vocab, bucket_max)
                    })?
                };

                // --- restore the true pre-capture state (undo the 3 throwaway runs) ---
                cache.rollback(e, &snap, 0)?;   // restores conv/ssm + sets len = snapshot len
                e.set_i32_one(&mut gs.pos_d, pos_save)?;
                for (il, ls) in len_save.iter().enumerate() {
                    if let (Some(kvl), Some(v)) = (cache.kv[il].as_mut(), ls) {
                        e.set_i32_one(&mut kvl.len_d, *v)?;
                    }
                }
                e.set_u32_one(&mut gs.token_d, tok_save)?;

                gs.graphs.insert(key, graph);
                gs.bucket_max.insert(key, bucket_max);
                gs.captures += 1;
            }
            // REPLAY: one dispatch runs the whole step; argmax wrote gs.token_d, counters advanced.
            gs.graphs.get(&key).unwrap().launch()?;
            // advance the HOST mirrors to match the in-graph DEVICE advance (host len/pos used only for
            // bucket selection next step; device counters are the source of truth for the kernels).
            cache.pos += 1;
            for kvl in cache.kv.iter_mut().filter_map(|k| k.as_mut()) { kvl.len += 1; }
            // read back the [1] u32 next token (the only D2H in steady state).
            let tok = e.dtoh_u32_one(&gs.token_d)?;
            out.push(tok);
        }
        Ok(out)
    }

    /// Device-counter full-attention decode (CUDA-GRAPH-PLAN Phase 2): clone of `full_attn_decode`
    /// using the `_dc` KV-append (write slot from `kvl.len_d`) + `_dc` fa_decode (t_kv from `kvl.len_d`
    /// after inc), and the resident device rope `pos_d`. Bit-identical to `full_attn_decode` (the
    /// `_dc` kernels reproduce the same math; fa_decode_dc with bucket_max==t_kv reproduces the same
    /// n_splits/per/combine). Advances `kvl.len`/`kvl.len_d`.
    pub(crate) fn full_attn_decode_dc(&self, e: &Engine, fa: &FullAttnLayer, h: &CudaSlice<f32>,
                            pos_d: &CudaSlice<i32>, cache: &mut Cache, il: usize)
                            -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        // eager-mirror path: advance host counters and size n_splits from the live t_kv (bit-identical
        // to fa_decode). The capture path uses full_attn_decode_dc_cap (fixed bucket_max, no host
        // advance, full-buffer K/V view).
        self.full_attn_decode_dc_inner(e, fa, h, pos_d, cache, il, None)
    }

    /// CAPTURE variant of `full_attn_decode_dc` (CUDA-GRAPH-PLAN Phase 3). `bucket_max` sizes the
    /// fa_decode_dc grid (n_splits) at capture time; the kernel reads the ACTUAL t_kv from the device
    /// counter `kvl.len_d`. Does NOT advance the host `kvl.len` (only the DEVICE counter via inc_seqlen,
    /// which is captured and replays each launch). Views the FULL K/V cache buffer so the kernel may
    /// safely read up to any t_kv within the bucket on replay. Bit-identical to eager when
    /// `bucket_max` yields the same n_splits as eager for the replayed t_kv (the bucket-key contract).
    pub(crate) fn full_attn_decode_dc_cap(&self, e: &Engine, fa: &FullAttnLayer, h: &CudaSlice<f32>,
                            pos_d: &CudaSlice<i32>, cache: &mut Cache, il: usize, bucket_max: usize)
                            -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        self.full_attn_decode_dc_inner(e, fa, h, pos_d, cache, il, Some(bucket_max))
    }

    fn full_attn_decode_dc_inner(&self, e: &Engine, fa: &FullAttnLayer, h: &CudaSlice<f32>,
                            pos_d: &CudaSlice<i32>, cache: &mut Cache, il: usize,
                            cap_bucket_max: Option<usize>)
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
        let mut q = e.uninit(n_head * head_dim)?;
        let mut gate = e.uninit(n_head * head_dim)?;
        e.q_gate_split(&qf, &mut q, &mut gate, head_dim, n_head, 1)?;

        let mut qn = e.uninit(n_head * head_dim)?;
        e.rms_norm(&q, fa.q_norm.float_data(), &mut qn, head_dim, n_head, eps)?;
        q = qn;
        let mut kn = e.uninit(n_head_kv * head_dim)?;
        e.rms_norm(&k, fa.k_norm.float_data(), &mut kn, head_dim, n_head_kv, eps)?;
        k = kn;
        let rope_dims = cfg.rope_dim_count as usize;
        // rope pos from the resident device counter (no per-step host upload).
        e.rope_neox(&mut q, pos_d, head_dim, rope_dims, n_head, 1, cfg.rope_freq_base, 1.0)?;
        e.rope_neox(&mut k, pos_d, head_dim, rope_dims, n_head_kv, 1, cfg.rope_freq_base, 1.0)?;

        let kvl = cache.kv[il].as_mut().unwrap();
        // (1) append at the device write slot kvl.len_d (== old len).
        e.append_kv_quantized_dc(&k, &v, &mut kvl.k, &mut kvl.v, &kvl.len_d,
                                 kvl.kv_dim_k, kvl.kv_dim_v, kvl.k_tok_bytes, kvl.v_tok_bytes)?;
        // (2) advance the device counter: kvl.len_d now holds new len == t_kv.
        e.inc_seqlen(&mut kvl.len_d)?;
        // n_splits sizing + K/V view extent:
        //  - eager path (cap_bucket_max==None): advance host len; size from live t_kv == bit-identical
        //    to fa_decode; view exactly t_kv*tok_bytes.
        //  - capture path (Some(bucket_max)): DO NOT touch host len (replay advances only the device
        //    counter); size n_splits from bucket_max; view the FULL cache buffer so any in-bucket t_kv
        //    is in range on replay.
        let (bucket_max, k_view, v_view) = match cap_bucket_max {
            None => {
                kvl.len += 1;
                let t_kv = kvl.len;
                (t_kv,
                 e.view_u8(&kvl.k, t_kv * kvl.k_tok_bytes),
                 e.view_u8(&kvl.v, t_kv * kvl.v_tok_bytes))
            }
            Some(bm) => {
                (bm,
                 e.view_u8(&kvl.k, kvl.k.len()),
                 e.view_u8(&kvl.v, kvl.v.len()))
            }
        };
        let (ktb, vtb) = (kvl.k_tok_bytes, kvl.v_tok_bytes);
        let mut attn = e.uninit(n_head * head_dim)?;
        if std::env::var("BW24_NOFA").is_ok() {
            return Err("BW24_NOFA (naive f32 SDPA) is incompatible with the quantized KV cache; \
                        unset BW24_NOFA to use fa_decode_dc".into());
        }
        // (3) fa_decode reads t_kv from kvl.len_d; bucket_max yields the eager n_splits -> bit-identical.
        e.fa_decode_dc(&q, &k_view, &v_view, &mut attn, head_dim, n_head, n_head_kv,
                       &kvl.len_d, bucket_max, scale, ktb, vtb)?;

        let mut gsig = e.uninit(n_head * head_dim)?;
        e.sigmoid(&gate, &mut gsig, n_head * head_dim)?;
        let mut attn_g = e.uninit(n_head * head_dim)?;
        e.mul(&attn, &gsig, &mut attn_g, n_head * head_dim)?;
        Ok(e.matmul(&fa.wo, &attn_g, 1)?)
    }

    /// Greedy generation: prime with prompt tokens (decode them in sequence to build state),
    /// then generate `max_new` tokens. Returns the generated token ids. (Back-compat: greedy,
    /// no EOS/stop — used by the decode==prefill validation gate. New code uses `generate_with`.)
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

    /// The reusable serving generation API (BASE-3). Primes the prompt, then samples up to
    /// `params.max_new` tokens, stopping on EOS, any stop-token, or the context-length guard.
    /// Calls `on_token(id)` after each emitted token (for streaming; return `false` to stop early).
    /// Returns `GenOutput { tokens, stop_reason }`. Does NOT detokenize — the caller (which owns
    /// the tokenizer) handles text + stop-STRING matching on the detokenized tail.
    pub fn generate_with<F: FnMut(u32) -> bool>(
        &self, e: &Engine, prompt: &[u32], params: &GenParams,
        sampler: &mut crate::sampler::Sampler, mut on_token: F,
    ) -> Result<GenOutput, Box<dyn std::error::Error>> {
        // Context guard: prompt + generated must fit max_ctx (caller-supplied or model default).
        let ctx_cap = params.max_ctx.unwrap_or(prompt.len() + params.max_new + 8);
        if prompt.len() >= ctx_cap {
            return Ok(GenOutput { tokens: Vec::new(), stop_reason: StopReason::ContextFull });
        }
        let room = ctx_cap - prompt.len();
        let budget = params.max_new.min(room);

        let mut cache = Cache::new(e, &self.cfg, ctx_cap)?;
        let mut last_logits = Vec::new();
        for &tok in prompt {
            last_logits = self.decode_step(e, tok, &mut cache)?;
            sampler.accept(tok);
        }
        let mut out = Vec::with_capacity(budget);
        let mut reason = StopReason::MaxNew;
        for _ in 0..budget {
            let next = sampler.sample(&last_logits);
            sampler.accept(next);
            out.push(next);
            if params.eos.contains(&next) { reason = StopReason::Eos; break; }
            if !on_token(next) { reason = StopReason::Callback; break; }
            if cache.pos >= ctx_cap { reason = StopReason::ContextFull; break; }
            last_logits = self.decode_step(e, next, &mut cache)?;
        }
        Ok(GenOutput { tokens: out, stop_reason: reason })
    }

    /// Full-attention decode: project q/gate/k/v for the new token, QK-norm, RoPE at pos,
    /// append k,v to the layer KV cache, attend over the full [0..=pos] context.
    pub(crate) fn full_attn_decode(&self, e: &Engine, fa: &FullAttnLayer, h: &CudaSlice<f32>,
                        pos_d: &CudaSlice<i32>, pos: usize, cache: &mut Cache, il: usize)
                        -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let n_head = cfg.n_head as usize;
        let n_head_kv = cfg.n_head_kv as usize;
        let head_dim = cfg.head_dim_k as usize;
        let eps = cfg.rms_eps;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // wq|wk|wv all take the same input `h` (in_f = n_embd) — quantize q8_1 ONCE, feed all three.
        let n_embd = cfg.n_embd as usize;
        let (qf, mut k, v) = if e.uses_q8_1_fast(&fa.wq) && e.uses_q8_1_fast(&fa.wk) && e.uses_q8_1_fast(&fa.wv) {
            let (hq, hd) = e.quantize_q8_1(h, 1, n_embd)?;
            (e.matmul_pre(&fa.wq, &hq, &hd, h, 1)?, e.matmul_pre(&fa.wk, &hq, &hd, h, 1)?, e.matmul_pre(&fa.wv, &hq, &hd, h, 1)?)
        } else {
            (e.matmul(&fa.wq, h, 1)?, e.matmul(&fa.wk, h, 1)?, e.matmul(&fa.wv, h, 1)?)
        };
        // q|gate fused: [2*head_dim per head]. Split on-device (no dtoh/host-loop/htod).
        let mut q = e.uninit(n_head * head_dim)?;
        let mut gate = e.uninit(n_head * head_dim)?;
        e.q_gate_split(&qf, &mut q, &mut gate, head_dim, n_head, 1)?;

        // QK-norm + RoPE at position `pos`
        let mut qn = e.uninit(n_head * head_dim)?;
        e.rms_norm(&q, fa.q_norm.float_data(), &mut qn, head_dim, n_head, eps)?;
        q = qn;
        let mut kn = e.uninit(n_head_kv * head_dim)?;
        e.rms_norm(&k, fa.k_norm.float_data(), &mut kn, head_dim, n_head_kv, eps)?;
        k = kn;
        let rope_dims = cfg.rope_dim_count as usize;
        e.rope_neox(&mut q, pos_d, head_dim, rope_dims, n_head, 1, cfg.rope_freq_base, 1.0)?;
        e.rope_neox(&mut k, pos_d, head_dim, rope_dims, n_head_kv, 1, cfg.rope_freq_base, 1.0)?;

        // append k,v into the RESIDENT GPU QUANTIZED KV cache at the current position (q8_0 K /
        // q5_1 V, on-device append-quantize kernel; no host round-trip). KVQUANT-PLAN §C/E2.
        let kvl = cache.kv[il].as_mut().unwrap();
        e.append_kv_quantized(&k, &v, &mut kvl.k, &mut kvl.v, kvl.len,
                              kvl.kv_dim_k, kvl.kv_dim_v, kvl.k_tok_bytes, kvl.v_tok_bytes)?;
        kvl.len += 1;
        let t_kv = kvl.len;

        // attend: q[hd,nh,1] over the resident byte K/V (view first t_kv*tok_bytes BYTES).
        let k_view = e.view_u8(&kvl.k, t_kv * kvl.k_tok_bytes);
        let v_view = e.view_u8(&kvl.v, t_kv * kvl.v_tok_bytes);
        let (ktb, vtb) = (kvl.k_tok_bytes, kvl.v_tok_bytes);
        let mut attn = e.uninit(n_head * head_dim)?;
        if std::env::var("BW24_NOFA").is_ok() {
            return Err("BW24_NOFA (naive f32 SDPA) is incompatible with the quantized KV cache; \
                        unset BW24_NOFA to use fa_decode".into());
        }
        e.fa_decode(&q, &k_view, &v_view, &mut attn, head_dim, n_head, n_head_kv, t_kv, scale, ktb, vtb)?;
        let _ = pos;

        // output gate: attn * sigmoid(gate), then o-proj
        let mut gsig = e.uninit(n_head * head_dim)?;
        e.sigmoid(&gate, &mut gsig, n_head * head_dim)?;
        let mut attn_g = e.uninit(n_head * head_dim)?;
        e.mul(&attn, &gsig, &mut attn_g, n_head * head_dim)?;
        Ok(e.matmul(&fa.wo, &attn_g, 1)?)
    }

    /// Linear-attention decode: conv with ring-buffer state, GDN scan carrying SSM state.
    pub(crate) fn linear_attn_decode(&self, e: &Engine, la: &LinearAttnLayer, h: &CudaSlice<f32>,
                          cache: &mut Cache, il: usize)
                          -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        self.linear_attn_decode_inner(e, la, h, cache, il, false)
    }

    /// CAPTURE variant of `linear_attn_decode` (CUDA-GRAPH-PLAN Phase 3). The GDN scan needs distinct
    /// in/out SSM-state buffers; the eager path SWAPS a fresh scratch into `rl.ssm_state` (new pointer
    /// each step), which is a CAPTURE HAZARD — the graph bakes capture-time pointers and never re-runs
    /// the host swap, so replay would read a stale state buffer. Here we instead COPY the scratch back
    /// into the STABLE `rl.ssm_state` buffer (memcpy_dtod, captured, same pointers every replay). Math
    /// is identical; only the buffer plumbing differs. `conv_state` is already mutated in place (no
    /// pointer change) so it is capture-safe as-is.
    pub(crate) fn linear_attn_decode_cap(&self, e: &Engine, la: &LinearAttnLayer, h: &CudaSlice<f32>,
                          cache: &mut Cache, il: usize)
                          -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        self.linear_attn_decode_inner(e, la, h, cache, il, true)
    }

    fn linear_attn_decode_inner(&self, e: &Engine, la: &LinearAttnLayer, h: &CudaSlice<f32>,
                          cache: &mut Cache, il: usize, persistent_state: bool)
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

        // projections (T=1): wqkv, wqkv_gate, ssm_beta, ssm_alpha ALL take input `h` (in_f = n_embd)
        // -> quantize q8_1 ONCE, feed all four (was 4x redundant quantize_q8_1 of the same row).
        let n_embd = cfg.n_embd as usize;
        let (qkv_mixed, z, beta_raw, alpha) = if e.uses_q8_1_fast(&la.wqkv) && e.uses_q8_1_fast(&la.wqkv_gate)
            && e.uses_q8_1_fast(&la.ssm_beta) && e.uses_q8_1_fast(&la.ssm_alpha) {
            let (hq, hd) = e.quantize_q8_1(h, 1, n_embd)?;
            (e.matmul_pre(&la.wqkv, &hq, &hd, h, 1)?, e.matmul_pre(&la.wqkv_gate, &hq, &hd, h, 1)?,
             e.matmul_pre(&la.ssm_beta, &hq, &hd, h, 1)?, e.matmul_pre(&la.ssm_alpha, &hq, &hd, h, 1)?)
        } else {
            (e.matmul(&la.wqkv, h, 1)?, e.matmul(&la.wqkv_gate, h, 1)?,
             e.matmul(&la.ssm_beta, h, 1)?, e.matmul(&la.ssm_alpha, h, 1)?)
        };

        // conv input = [conv_state (pad cols) | new col]  channel-major [conv_dim, pad+1].
        // Assemble + roll the ring ON-DEVICE from the resident conv_state (no dtoh/host-loop/htod).
        let rl = cache.recur[il].as_mut().unwrap();
        let tp = pad + 1;
        let mut conv_in = e.uninit(conv_dim * tp)?;
        e.conv_assemble_and_roll(&qkv_mixed, &mut rl.conv_state, &mut conv_in, conv_dim, pad)?;
        let mut conv_out = e.uninit(conv_dim)?;  // [conv_dim, 1] channel-major, SiLU
        e.ssm_conv1d(&conv_in, la.ssm_conv1d.float_data(), &mut conv_out, conv_dim, 1, d_conv, true)?;

        // split + repack to GDN [d_state, num_v, 1] ON-DEVICE; q/k repeat 16->32 via modulo
        // (ggml_repeat_4d, kh = vh % num_k). No dtoh/host-loop/3x-htod.
        let _ = head_k;  // head_k == d_state; the kernel uses head_k = d_state internally.
        let mut q_g = e.uninit(d_state * num_v)?;
        let mut k_g = e.uninit(d_state * num_v)?;
        let mut v_g = e.uninit(d_state * num_v)?;
        e.qkv_to_gdn_repack(&conv_out, &mut q_g, &mut k_g, &mut v_g, d_state, num_v, num_k, key_dim, 1)?;
        let mut q_l2 = e.uninit(d_state * num_v)?;
        e.l2_norm(&q_g, &mut q_l2, d_state, num_v, eps)?;
        let mut k_l2 = e.uninit(d_state * num_v)?;
        e.l2_norm(&k_g, &mut k_l2, d_state, num_v, eps)?;
        let v_gd = v_g;

        let mut beta = e.uninit(num_v)?;
        e.sigmoid(&beta_raw, &mut beta, num_v)?;
        let mut g_log = e.uninit(num_v)?;
        e.gdn_glog(&alpha, la.ssm_dt.float_data(), la.ssm_a.float_data(), &mut g_log, num_v, 1)?;

        // GDN scan: SSM state stays RESIDENT on GPU. gdn needs DISTINCT in/out state buffers.
        // DECODE DETERMINISM FIX: write the new state into the PERSISTENT spare buffer
        // (`ssm_state_alt`) and PING-PONG the two owned buffers in place — instead of allocating a
        // fresh `state_scratch` via `e.uninit` each step and swapping its pointer in. The old
        // per-step alloc/free churned the stream-ordered async pool; the freed prior state block was
        // recycled by a later step's scratch while a kernel still referenced the swapped-in state,
        // a use-after-reuse that made decode RUN-TO-RUN nondeterministic (two identical primes
        // diverged). With two stable resident buffers there is no per-step alloc/free and no pool
        // churn; the math is byte-identical. `o` is a true per-step output (consumed immediately by
        // gated_rmsnorm below) so it stays a normal scratch.
        let mut o = e.uninit(d_state * num_v)?;
        let n_state = d_state * d_state * num_v;
        // gdn reads ssm_state, writes the spare ssm_state_alt (disjoint resident fields).
        {
            let RecurLayer { ssm_state, ssm_state_alt, .. } = rl;
            e.gdn_scan_s128(&q_l2, &k_l2, &v_gd, &g_log, &beta, ssm_state, ssm_state_alt, &mut o, num_v, 1, scale)?;
        }
        if persistent_state {
            // CAPTURE-safe (graph replay): the canonical state every replay reads must stay at a
            // FIXED pointer (baked into the captured graph). Copy the freshly-written spare BACK
            // into ssm_state (captured, replays each launch). No host pointer swap.
            let alt = std::mem::replace(&mut rl.ssm_state_alt, e.zeros(0)?);
            e.copy_into(&mut rl.ssm_state, 0, &alt, n_state)?;
            rl.ssm_state_alt = alt;
        } else {
            // EAGER: swap the two OWNED resident buffers in place (stable pointers, no alloc/free).
            std::mem::swap(&mut rl.ssm_state, &mut rl.ssm_state_alt);
        }

        // gated RMSNorm + ssm_out
        let mut gn = e.uninit(d_state * num_v)?;
        e.gated_rmsnorm(&o, la.ssm_norm.float_data(), &z, &mut gn, d_state, num_v, eps)?;
        Ok(e.matmul(&la.ssm_out, &gn, 1)?)
    }
}
