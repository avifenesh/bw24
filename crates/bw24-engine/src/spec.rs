//! Qwen3.5 MTP (NextN) greedy speculative decode (research/mtp/MTP-PLAN.md §A/§B/§C/§D).
//!
//! Greedy spec decode is MATHEMATICALLY EXACT: the accepted+bonus token stream is token-for-token
//! identical to plain greedy `generate`. This module provides:
//!   - `mtp_head_forward`  (§A, T=1): one NextN draft-token forward.
//!   - `decode_step_t`     (§D.3, T=K+1): batched target verify forward, all-column logits.
//!   - `generate_spec`     (§B): the draft/verify/accept/rollback orchestrator.
//! Cache snapshot/rollback lives in cache.rs (§D.4). The MTP head uses its OWN scratch KV (§D.6),
//! PERSISTENT over the committed sequence (see `MtpScratch`).

use cudarc::driver::CudaSlice;
use crate::Engine;
use crate::hybrid::{HybridModel, Mixer, FullAttnLayer, LinearAttnLayer, MtpHead};
use crate::cache::{Cache, KvLayer};
use crate::forward::argmax;

/// H-SEED CONVENTION (BW24_SPEC_HPOST=1): feed the MTP head the POST-norm hidden — trunk rows
/// hand over `output_norm(x)` and the draft chain recurrence hands over `shared_head_norm(h_nextn)`
/// (= final_h) — matching the reference engines: llama.cpp #24025 ("qwen35: use post-norm hidden
/// state for MTP", t_h_nextn is taken AFTER the final norm in both trunk and MTP graphs) and
/// SGLang's qwen3_5_mtp (spec_info.hidden_states = the target model's post-norm output). bw24's
/// historical convention (default, MTP-PLAN §A) is PRE-norm x. Draft-quality-only: exactness is
/// the verify's job either way; acceptance arbitrates. OnceLock: read once, hot-loop safe.
pub(crate) fn spec_hpost() -> bool {
    static H: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *H.get_or_init(|| std::env::var("BW24_SPEC_HPOST").map(|v| v != "0").unwrap_or(false))
}

/// LEAN VERIFY (default ON since 2026-07-08; BW24_SPEC_LEAN=0 reverts — close35 lane): the verify m-scaling
/// probe + nsys diff showed the verify t-path pays ~1.0ms/call at m=1 over eager decode on the
/// 35B, and the kernels are NOT the cause (dev-MoE identical, kernel-time delta only +179us).
/// The overhead is (a) ~250 extra cuMemsetD8Async/call from `e.zeros()` on buffers every kernel
/// fully overwrites (~0.9ms host issue + ~0.35ms GPU) and (b) the t=1 FA rows dispatch (rows_v2 +
/// combine_rows, +50us vs the eager fa_decode pair). This flag switches (a) fully-overwritten
/// verify buffers to `e.uninit` (identical bytes: every element is written before read) and
/// (b) t==1 verify FA to the eager `fa_decode` entry (byte-identical: kernel-check pins the
/// rows-vs-loop identity and the per-row loop at t=1 IS fa_decode on the same q). Gates arbitrate.
pub(crate) fn spec_lean() -> bool {
    static L: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // DEFAULT ON since 2026-07-08 (BW24_SPEC_LEAN=0 reverts): bit-identical (buffers fully
    // overwritten; gates green incl maxdiff-identical run-gen) and measured +2.4% e2e p3 /
    // +1.5% p2 at the daily 35B config. m=1 verify now costs eager-decode parity.
    *L.get_or_init(|| std::env::var("BW24_SPEC_LEAN").map(|v| v != "0").unwrap_or(true))
}

/// SMALL-M BATCHED VERIFY (default ON since 2026-07-09; BW24_SPEC_M2=0 reverts — lane/spec-m2): extend the
/// batched linear-attn verify arm down to t=2 and batch the MoE dev token loop over a
/// grid.z=token axis at every verify t. The close35 m-scaling probe put the m=2 verify tier at
/// x1.54 of m=1 (llama x1.14); the per-column linear chain (t<3) and the serial MoE dev token
/// loop are the two launch-structure causes. Both changes are LAUNCH-STRUCTURE ONLY:
/// (a) the batched conv's t<pad ring update is pure copies (ssm_conv_ring_rebuild from a cloned
///     ring — the ring stores raw input columns); every arithmetic kernel is the same one the
///     t>=3 arm already runs (matmul_decode_exact bit-identical at m=2-4, gdn_scan's internal
///     t-loop == chained T=1 steps);
/// (b) the MoE dev-rows twins run the serial loop's per-token warp program with tok-offset
///     pointers (same sel/w/aq/ad bytes, same dot order, same slot-ordered FMA chain).
/// Gates arbitrate: run-spec K=1..8 self-consistency (35B+9B), kernel-check, run-gen argmax.
pub(crate) fn spec_m2() -> bool {
    static M: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // DEFAULT ON since 2026-07-09 (BW24_SPEC_M2=0 reverts): launch-structure only — t=2
    // batched linear arm (ring-roll copies, zero new FP order) + MoE dev-rows kernels
    // (grid.z=token, 4 launches/layer at any verify t). Acceptance bit-identical at every K;
    // 35B p2 +3.4% / p3 +3.6%; the profitable-K plateau widens (new optimum K=3 at 223).
    *M.get_or_init(|| std::env::var("BW24_SPEC_M2").map(|v| v != "0").unwrap_or(true))
}

/// VERIFY-TIER TRUNK LAUNCH-FUSION (default ON since 2026-07-09; BW24_SPEC_FUSED_T=0 reverts — lane/close35b): extend
/// the t=1 fused2/fused3 Q8_0 trunk launches to the batched verify tier (t=2-4, the K=1..3
/// verify shapes). At t>1 the trunk pairs/triples (35B wqkv+wqkv_gate, wq/wk/wv,
/// gate_shexp+up_shexp) each run a separate `matmul_decode_exact` — one q8_1 re-quantize of the
/// SAME activation plus one _b2/_b4 launch per tensor. The fused twins share ONE quantize and
/// ONE launch per group; per (tensor,token,row) the kernel body is q8_0_mmvq_batched verbatim
/// with the identical row mapping -> BIT-IDENTICAL by construction (kernel-check pins it,
/// run-spec K=1..8 + acceptance identity arbitrate e2e).
pub(crate) fn spec_fused_t() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // DEFAULT ON since 2026-07-09 (BW24_SPEC_FUSED_T=0 reverts): verify t=2-4 trunk launch-fusion
    // (fused2/fused3 Q8_0 batched twins, bit-identical by construction — m=1 block-offset split on
    // the batched body). m=2 marginal token 2117->1762us; 35B daily: p3 +3.7% (crosses llama), p2 +5%.
    *F.get_or_init(|| std::env::var("BW24_SPEC_FUSED_T").map(|v| v != "0").unwrap_or(true))
}

/// zeros/uninit switch for verify-path buffers that are FULLY OVERWRITTEN before any read.
/// Only call this on such buffers — the lean contract is "identical bytes by construction".
fn vbuf(e: &Engine, n: usize) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
    if spec_lean() { e.uninit(n) } else { e.zeros(n) }
}

/// Scratch KV for the MTP block (one full-attn layer).
///
/// PERSISTENT MODE (default, 2026-07-03 — the acceptance lever): sized cap = max_ctx and kept in
/// sync with the COMMITTED sequence — slot p holds the MTP block's K/V for committed token p
/// (roped p+1, the chain's rope convention), so the draft chain's self-attention sees the FULL
/// committed history instead of only the current round's 1..K+1 chain tokens (the reference
/// engine's "mtp_update" design). Entries come from two sources:
///   - chain appends: accepted positions KEEP their chain-computed entries (embedding exact,
///     hidden chain-approximate — the reference engine accepts the same);
///   - `mtp_kv_fill` batches: prompt positions + the last-draft position on full accept, computed
///     from EXACT trunk hiddens (K/V-only MTP-block pass, no attention/FFN/lm_head).
/// Rejected drafts / p-min extras / pseudo-seed appends are all discarded by the round-start
/// `set_len` truncation (the KvLayer len mechanism — §C rollback for the draft side).
/// Multi-turn spec-decode session (2026-07-05): trunk Cache + persistent MTP draft scratch +
/// the committed token list, alive across generate_spec_session calls. Turn N+1 primes ONLY its
/// suffix (chunked continuation prime over the quantized past) and mtp_kv_fill's its suffix rows,
/// then the round loop runs unchanged. `last_h` carries the pre-output_norm hidden of the last
/// committed row across turns (the predecessor-pairing seed + fill anchor).
/// Per-request sampling config for the sampled-spec serve path.
#[derive(Clone, Copy, Debug)]
pub struct SpecSampling {
    pub temp: f32,
    pub seed: u64,
    pub top_k: i32,        // 0 = off
    pub top_p: f32,        // 1.0 = off
    pub min_p: f32,        // 0.0 = off
    pub penalty_last_n: usize, // 0 = penalties off
    pub penalty_repeat: f32,
    pub penalty_freq: f32,
    pub penalty_present: f32,
}

pub struct SpecSession {
    pub(crate) cache: Cache,
    pub(crate) scratch: MtpScratch,
    /// Every token whose state the caches hold, in order (prompt turns + generated), INCLUDING
    /// overshoot: spec commits accepted drafts past max_new; those rows are in the caches, so the
    /// session must count them. Callers render output from this, not from their own echo.
    pub committed: Vec<u32>,
    /// Pre-output_norm hidden of the LAST committed row (device). None before the first turn.
    pub(crate) last_h: Option<CudaSlice<f32>>,
    /// Greedy argmax predicting the token AFTER committed.last() (from the last turn's final
    /// logits). Fuels empty-suffix continuation bursts (serve): the next turn emits this token
    /// first, feeds it, and the round loop resumes without any prime. None before the first turn.
    pub next_pred: Option<u32>,
    /// SAMPLED-SPEC stream continuity across bursts: Philox event counters persist here so a
    /// session's randomness never repeats between generate_spec_session calls. (0,0) at admit.
    pub sctr: u32,
    pub uctr: u32,
}
impl SpecSession {
    /// Context capacity of the session's caches (the server's ContextFull guard).
    pub fn cache_max_ctx(&self) -> usize { self.cache.max_ctx }
}

pub(crate) struct MtpScratch {
    kv: KvLayer,
    /// Row capacity. Doubles as the fa_decode_dc bucket_max for BOTH draft paths (graph + eager):
    /// n_splits is sized from it ONCE, so the graph captured at round 0 stays valid for every
    /// later t_kv (splits beyond the device len_d exit empty; the shared combine skips them) —
    /// KV growth without recapture. Eager uses the SAME bucket_max -> identical dispatch ->
    /// bit-identical drafts (the graph-vs-eager parity gate).
    cap: usize,
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
        let (kbb, vbb) = crate::kv_blk_bytes();   // env-selected KV formats (default 34/24)
        let k_tok_bytes = (kv_dim_k / 32) * kbb;
        let v_tok_bytes = (kv_dim_v / 32) * vbb;
        Ok(MtpScratch { kv: KvLayer {
            k: e.alloc_u8(cap * k_tok_bytes)?, v: e.alloc_u8(cap * v_tok_bytes)?,
            kv_dim_k, kv_dim_v, k_tok_bytes, v_tok_bytes, len: 0,
            len_d: e.htod_i32(&[0])?,
        }, cap })
    }
    /// Set BOTH length counters: the host mirror AND the device len_d the captured append/fa read
    /// (a 4-byte in-place htod — the counter pointer is baked into the graph, never realloc'd).
    /// This is the ONLY truncation/rollback mechanism the persistent draft KV needs.
    fn set_len(&mut self, e: &Engine, n: usize) -> Result<(), Box<dyn std::error::Error>> {
        self.kv.len = n;
        e.set_i32_one(&mut self.kv.len_d, n as i32)
    }
}

/// Retained verify intermediates for the REPLAY-FREE partial accept (2026-07-03, the profiled
/// #1 spec cost at long ctx: the partial-accept replay was a DUPLICATE trunk pass — ~0.54 extra
/// full weight reads per round — recomputing columns the verify had already produced
/// bit-identically). Holds, per linear layer, everything needed to rebuild its recurrent state
/// to "after the first j verify columns" WITHOUT re-running the trunk:
/// - BATCHED-path layers (`gdn`): the exact token-major inputs the round's ONE gdn_scan
///   consumed. A prefix re-run of the SAME kernel (t=j) from the snapshot state is bit-identical
///   to the first j iterations of the verify's scan — the kernel's t-loop carries state in
///   registers and iteration t never depends on T. `qkv_mixed` (the conv input) feeds the
///   pure-copy ring rebuild.
/// - PER-COLUMN-path layers (`cols`): dtod clones of (conv_state, ssm_state) taken after each
///   column 0..t-2 — pure copies of the actual chain states (the last column is never a rebuild
///   target: j <= t-1).
/// Full-attn layers need nothing: their verify KV rows are bit-identical to eager's (the
/// decode-exact contract; verify-probe pins it), so rollback = len truncation.
struct GdnStash {
    qkv_mixed: CudaSlice<f32>,                     // [t, conv_dim] token-major (conv input)
    q_l2: CudaSlice<f32>, k_l2: CudaSlice<f32>, v_g: CudaSlice<f32>,  // [t, num_v, d_state]
    g_log: CudaSlice<f32>, beta: CudaSlice<f32>,   // [t, num_v]
}
struct VerifyCkpt {
    gdn: Vec<Option<GdnStash>>,                    // [n_layer], Some iff batched linear path ran
    cols: Vec<Option<Vec<(CudaSlice<f32>, CudaSlice<f32>)>>>, // [n_layer][col] = (conv, ssm) after col
}
impl VerifyCkpt {
    fn new(n_layer: usize) -> Self {
        VerifyCkpt { gdn: (0..n_layer).map(|_| None).collect(),
                     cols: (0..n_layer).map(|_| None).collect() }
    }
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
    #[allow(clippy::too_many_arguments)]
    fn mtp_head_forward_dev(&self, e: &Engine, mtp: &MtpHead, e_tok: u32, h_seed: &CudaSlice<f32>,
                            scratch: &mut MtpScratch, mtp_pos: usize,
                            embd_gpu: &CudaSlice<u8>, embd_qt: i32, embd_rb: usize)
                            -> Result<(CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let n_embd = cfg.n_embd as usize;
        let eps = cfg.rms_eps;
        let pos_d = e.htod_i32(&[mtp_pos as i32])?;

        // op A: embed the predict-from token ON DEVICE (was host dequant + 14-20KB htod per draft
        // token — with the resident table the transfer is one 4B token id).
        let e_emb = e.embed_gather_device_t(embd_gpu, &[e_tok], n_embd, embd_qt, embd_rb)?;

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

        // op 6: attention on the scratch KV. SAME dc launcher as the graph path (bucket_max =
        // scratch.cap, length from the device len_d) so eager drafts match graph drafts
        // bit-for-bit at any t_kv (the parity gate). Host len mirrored here (the dc append
        // advances only the device counter).
        let attn_out = match &mtp.mixer {
            Mixer::Full(fa) => {
                let out = self.mtp_full_attn_dc(e, fa, &a_norm, &pos_d, scratch)?;
                scratch.kv.len += 1;
                out
            }
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
                Self::ffn_act(e, &self.cfg, &gate, &up, &mut act, n_ff)?;
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
        // Chain recurrence hand-over: pre-norm h_nextn (default) or post-norm final_h
        // (BW24_SPEC_HPOST — llama.cpp #24025's t_h_nextn is taken AFTER the head norm).
        Ok((logits, if spec_hpost() { final_h } else { h_nextn }))
    }

    /// MTP-block full attention, T=1, on the scratch KV (BOTH draft paths — eager and graph):
    /// the scratch write slot and the attention bound come from `scratch.kv.len_d` (device i32[1])
    /// so the launch args are FIXED across draft steps — ONE captured graph serves the whole
    /// chain, and replays keep seeing KV growth through the device counter (no recapture).
    /// Geometry contract: n_splits is sized from `scratch.cap` (the persistent capacity); splits
    /// whose key range lies beyond the device t_kv exit empty and the shared combine skips them
    /// (fa_decode_dc bit-correct-for-any-t_kv<=bucket_max contract). The eager path uses the SAME
    /// launcher with the SAME bucket_max -> identical dispatch -> bit-identical draft tokens (the
    /// graph-vs-eager parity gate). Host len is NOT advanced here (graph contract); callers mirror.
    fn mtp_full_attn_dc(&self, e: &Engine, fa: &FullAttnLayer, h: &CudaSlice<f32>,
                        pos_d: &CudaSlice<i32>, scratch: &mut MtpScratch)
                        -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let n_head = cfg.n_head as usize;
        let n_head_kv = cfg.n_head_kv as usize;
        let head_dim = cfg.head_dim_k as usize;
        let eps = cfg.rms_eps;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let n_embd = cfg.n_embd as usize;
        let bucket_max = scratch.cap;   // < 96 guaranteed by the graph_draft eligibility gate

        let (qf, mut k, v) = if e.uses_q8_1_fast(&fa.wq) && e.uses_q8_1_fast(&fa.wk) && e.uses_q8_1_fast(&fa.wv) {
            let (hq, hd) = e.quantize_q8_1(h, 1, n_embd)?;
            (e.matmul_pre(&fa.wq, &hq, &hd, h, 1)?, e.matmul_pre(&fa.wk, &hq, &hd, h, 1)?, e.matmul_pre(&fa.wv, &hq, &hd, h, 1)?)
        } else {
            (e.matmul(&fa.wq, h, 1)?, e.matmul(&fa.wk, h, 1)?, e.matmul(&fa.wv, h, 1)?)
        };
        // M3/Hy3 have no attention output gate — wq out is exactly q; skip the split.
        let gated = self.cfg.attn_out_gate();
        let (mut q, gate) = if gated {
            let mut q = e.zeros(n_head * head_dim)?;
            let mut gate = e.zeros(n_head * head_dim)?;
            e.q_gate_split(&qf, &mut q, &mut gate, head_dim, n_head, 1)?;
            (q, Some(gate))
        } else { (qf, None) };

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
        // append at the DEVICE slot (kv.len_d == old len), then advance the counter in-graph.
        e.append_kv_quantized_dc(&k, &v, &mut kv.k, &mut kv.v, &kv.len_d,
                                 kv.kv_dim_k, kv.kv_dim_v, kv.k_tok_bytes, kv.v_tok_bytes)?;
        e.inc_seqlen(&mut kv.len_d)?;
        // full-buffer views (any in-round t_kv stays in range on replay); the kernel bounds the
        // key range from the device counter.
        let k_view = e.view_u8(&kv.k, kv.k.len());
        let v_view = e.view_u8(&kv.v, kv.v.len());
        let (ktb, vtb) = (kv.k_tok_bytes, kv.v_tok_bytes);
        let mut attn = e.zeros(n_head * head_dim)?;
        e.fa_decode_dc(&q, &k_view, &v_view, &mut attn, head_dim, n_head, n_head_kv,
                       &kv.len_d, bucket_max, scale, ktb, vtb)?;

        let attn_g = match &gate {
            Some(gate) => {
                let mut gsig = e.zeros(n_head * head_dim)?;
                e.sigmoid(gate, &mut gsig, n_head * head_dim)?;
                let mut ag = e.zeros(n_head * head_dim)?;
                e.mul(&attn, &gsig, &mut ag, n_head * head_dim)?;
                ag
            }
            None => attn,
        };
        Ok(e.matmul(&fa.wo, &attn_g, 1)?)
    }

    /// PERSISTENT-DRAFT-KV fill (the reference engine's "mtp_update" analogue): compute the MTP
    /// block's K/V for `tokens` (committed tokens at positions pos0..pos0+T) from their EXACT
    /// trunk hiddens `h` ([T, n_embd] token-major, pre-output_norm) and append at slots pos0.. of
    /// the scratch KV. K/V-ONLY — ops A/1-5 plus the K-side of op 6 (wk/wv + k_norm + rope +
    /// quantized append); no wq/attention/FFN/lm_head, so per-token cost ~= eh_proj + wk/wv (a
    /// small fraction of one trunk layer), T-batched. Rope follows the chain convention
    /// rope(token@p) = p+1. Runs at round boundaries OUTSIDE the captured graph in BOTH draft
    /// modes -> draft parity by construction. Caller must have scratch.kv.len == pos0.
    #[allow(clippy::too_many_arguments)]
    fn mtp_kv_fill(&self, e: &Engine, mtp: &MtpHead, tokens: &[u32], h: &CudaSlice<f32>,
                   pos0: usize, scratch: &mut MtpScratch,
                   embd_gpu: &CudaSlice<u8>, embd_qt: i32, embd_rb: usize)
                   -> Result<(), Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let n_embd = cfg.n_embd as usize;
        let eps = cfg.rms_eps;
        let t = tokens.len();
        assert_eq!(scratch.kv.len, pos0, "mtp_kv_fill: append slot mismatch");
        assert!(pos0 + t <= scratch.cap, "mtp_kv_fill: scratch overflow");
        let Mixer::Full(fa) = &mtp.mixer else {
            panic!("MTP block is full-attn in qwen35; linear MTP not supported")
        };
        let pos_vec: Vec<i32> = (0..t).map(|i| (pos0 + i + 1) as i32).collect();
        let pos_d = e.htod_i32(&pos_vec)?;

        // ops A/1/2: embed + the two input norms, T-wide.
        let e_emb = e.embed_gather_device_t(embd_gpu, tokens, n_embd, embd_qt, embd_rb)?;
        let mut e_norm = e.zeros(t * n_embd)?;
        e.rms_norm(&e_emb, mtp.enorm.float_data(), &mut e_norm, n_embd, t, eps)?;
        let mut h_norm = e.zeros(t * n_embd)?;
        e.rms_norm(h, mtp.hnorm.float_data(), &mut h_norm, n_embd, t, eps)?;

        // op 3: per-row [e_norm ; h_norm] concat, token-major [T, 2*n_embd].
        let mut concat = e.zeros(t * 2 * n_embd)?;
        for i in 0..t {
            e.copy_view_into(&mut concat, i * 2 * n_embd,
                             &e_norm.slice(i * n_embd..(i + 1) * n_embd), n_embd)?;
            e.copy_view_into(&mut concat, i * 2 * n_embd + n_embd,
                             &h_norm.slice(i * n_embd..(i + 1) * n_embd), n_embd)?;
        }

        // ops 4/5: eh_proj + attn_norm, T-wide.
        let inp_sa = e.matmul(&mtp.eh_proj, &concat, t)?;
        let mut a_norm = e.zeros(t * n_embd)?;
        e.rms_norm(&inp_sa, mtp.attn_norm.float_data(), &mut a_norm, n_embd, t, eps)?;

        // op 6 (K/V half): wk/wv + k_norm + rope + per-row quantized append. No wq/attention —
        // the fill only has to leave correct K/V rows behind for later chains to attend over.
        let n_head_kv = cfg.n_head_kv as usize;
        let head_dim = cfg.head_dim_k as usize;
        let mut k = e.matmul(&fa.wk, &a_norm, t)?;
        let v = e.matmul(&fa.wv, &a_norm, t)?;
        let mut kn = e.zeros(t * n_head_kv * head_dim)?;
        e.rms_norm(&k, fa.k_norm.float_data(), &mut kn, head_dim, n_head_kv * t, eps)?;
        k = kn;
        let rope_dims = cfg.rope_dim_count as usize;
        e.rope_neox(&mut k, &pos_d, head_dim, rope_dims, n_head_kv, t, cfg.rope_freq_base, 1.0)?;

        let kv = &mut scratch.kv;
        for i in 0..t {
            let k_row = k.slice(i * kv.kv_dim_k..(i + 1) * kv.kv_dim_k);
            let v_row = v.slice(i * kv.kv_dim_v..(i + 1) * kv.kv_dim_v);
            e.append_kv_quantized_view(&k_row, &v_row, &mut kv.k, &mut kv.v, kv.len + i,
                                       kv.kv_dim_k, kv.kv_dim_v, kv.k_tok_bytes, kv.v_tok_bytes)?;
        }
        kv.len += t;
        e.set_i32_one(&mut kv.len_d, kv.len as i32)?;
        Ok(())
    }

    /// CAPTURE body for the GRAPH DRAFT (stage 2 of graph-grade spec): ONE MTP head forward with
    /// every varying input device-resident —
    ///   - token id from the persistent `tok_d` (the previous replay's in-graph argmax wrote it,
    ///     so the chain feeds itself; the host reads the same 4 bytes for the draft list),
    ///   - h_seed from the persistent `h_seed_d` (h_nextn is copied BACK into it at the end),
    ///   - rope pos from the persistent `pos_d` counter (inc'd in-graph),
    ///   - scratch KV slot/bound from `scratch.kv.len_d` (see mtp_full_attn_dc).
    /// The p-min confidence lands in the persistent `p_d` iff `with_prob` (env is fixed per run).
    /// Same kernels, same dispatch as the eager mtp_head_forward_dev chain -> same draft tokens
    /// (exactness never depends on drafts — the verify arbitrates — but acceptance parity does).
    /// `with_head=false` captures the HEAD-LESS twin for the pseudo-seed replay (2026-07-03):
    /// the pseudo pass only needs h_nextn (op 10) + the scratch append — the lm_head read
    /// (~1.06ms q6_K on the 9B), argmax and prob are dead weight there. h_nextn's inputs are
    /// untouched, so the seed value is identical; round-start resets overwrite tok_d/p_d anyway.
    /// `sampled_cap` = Some((ctr_d, perturb_d, q_out_d, seed, temp)) captures the SAMPLED twin
    /// (step 3 of the sampled-spec arc): head logits are retained in the persistent `q_out_d`
    /// (host D2Ds them to the round's q slot after each replay), the DEVICE event counter is
    /// bumped in-graph, and the argmax reads GUMBEL-PERTURBED logits — one categorical draw per
    /// replay, bit-identical to the eager arm's gumbel_perturb at the same (seed, sctr, temp).
    /// seed/temp are capture-time constants (fixed per generate call, like p_min).
    #[allow(clippy::too_many_arguments)]
    fn mtp_head_forward_cap(&self, e: &Engine, mtp: &MtpHead,
                            tok_d: &mut CudaSlice<u32>, pos_d: &mut CudaSlice<i32>,
                            h_seed_d: &mut CudaSlice<f32>, p_d: &mut CudaSlice<f32>,
                            scratch: &mut MtpScratch, with_prob: bool, with_head: bool,
                            embd_gpu: &CudaSlice<u8>, embd_qt: i32, embd_rb: usize, d_vocab: usize,
                            sampled_cap: Option<(&mut CudaSlice<u32>, &mut CudaSlice<f32>,
                                                 &mut CudaSlice<f32>, u64, f32)>)
                            -> Result<(), Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let n_embd = cfg.n_embd as usize;
        let eps = cfg.rms_eps;
        let e_emb = e.embed_gather_device(embd_gpu, tok_d, n_embd, embd_qt, embd_rb)?;
        let mut e_norm = e.zeros(n_embd)?;
        e.rms_norm(&e_emb, mtp.enorm.float_data(), &mut e_norm, n_embd, 1, eps)?;
        let mut h_norm = e.zeros(n_embd)?;
        e.rms_norm(&*h_seed_d, mtp.hnorm.float_data(), &mut h_norm, n_embd, 1, eps)?;
        let mut concat = e.zeros(2 * n_embd)?;
        e.copy_into(&mut concat, 0, &e_norm, n_embd)?;
        e.copy_into(&mut concat, n_embd, &h_norm, n_embd)?;
        let inp_sa = e.matmul(&mtp.eh_proj, &concat, 1)?;
        let mut a_norm = e.zeros(n_embd)?;
        e.rms_norm(&inp_sa, mtp.attn_norm.float_data(), &mut a_norm, n_embd, 1, eps)?;
        let attn_out = match &mtp.mixer {
            Mixer::Full(fa) => self.mtp_full_attn_dc(e, fa, &a_norm, pos_d, scratch)?,
            Mixer::Linear(_) => panic!("MTP block is full-attn in qwen35; linear MTP not supported"),
        };
        let mut x1 = e.zeros(n_embd)?;
        e.add(&inp_sa, &attn_out, &mut x1, n_embd)?;
        let mut z = e.zeros(n_embd)?;
        e.rms_norm(&x1, mtp.post_attn_norm.float_data(), &mut z, n_embd, 1, eps)?;
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
                Self::ffn_act(e, &self.cfg, &gate, &up, &mut act, n_ff)?;
                e.matmul(ffn_down, &act, 1)?
            }
            crate::hybrid::Ffn::Moe(_) => return Err("graph draft requires a Dense MTP FFN".into()),
        };
        let mut h_nextn = e.zeros(n_embd)?;
        e.add(&x1, &ffn_out, &mut h_nextn, n_embd)?;
        // BW24_SPEC_HPOST needs final_h even head-less (it IS the next seed under that convention).
        let final_h = if with_head || spec_hpost() {
            let final_norm = mtp.shared_head_norm.as_ref().unwrap_or(&self.output_norm);
            let mut fh = e.zeros(n_embd)?;
            e.rms_norm(&h_nextn, final_norm.float_data(), &mut fh, n_embd, 1, eps)?;
            Some(fh)
        } else { None };
        if with_head {
            let head = mtp.shared_head_head.as_ref().unwrap_or(&self.output);
            let logits = e.matmul(head, final_h.as_ref().unwrap(), 1)?;
            if let Some((ctr_d, perturb_d, q_out_d, seed, temp)) = sampled_cap {
                // SAMPLED chain: retain q (raw head logits -> persistent q_out_d; the matmul's
                // own buffer is pool-recycled after the capture body returns, so it can't be the
                // retention target), bump the device event counter, gumbel-perturb reading it,
                // and argmax the PERTURBED logits into tok_d — the in-graph categorical draw.
                e.copy_into(q_out_d, 0, &logits, d_vocab)?;
                e.sctr_inc(ctr_d)?;
                e.gumbel_perturb_ctr(&logits, perturb_d, d_vocab, seed, ctr_d, temp)?;
                e.argmax_token_device_into(perturb_d, tok_d, d_vocab)?;
                // p-min prob = the head's RAW softmax confidence in the SAMPLED pick — same
                // semantics as the eager sampled arm's prob_of_token_device(dl_d, tok_d).
                if with_prob { e.prob_of_token_device_into(&logits, tok_d, p_d, d_vocab)?; }
            } else {
                // draft token -> persistent tok_d (next replay's embed reads it; host reads the 4 bytes).
                e.argmax_token_device_into(&logits, tok_d, d_vocab)?;
                if with_prob { e.prob_of_token_device_into(&logits, tok_d, p_d, d_vocab)?; }
            }
        }
        // Next draft step's h_seed: pre-norm h_nextn (default) or post-norm final_h (HPOST).
        if spec_hpost() {
            e.copy_into(h_seed_d, 0, final_h.as_ref().unwrap(), n_embd)?;
        } else {
            e.copy_into(h_seed_d, 0, &h_nextn, n_embd)?;
        }
        // advance the draft rope position in-graph.
        e.inc_seqlen(pos_d)?;
        Ok(())
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
        self.decode_step_t_h_emb(e, tokens, pos0, cache, None)
    }

    /// Like `decode_step_t_h` with an optional RESIDENT embed table (spec hot loop): device
    /// gather instead of host dequant + [T, n_embd] f32 htod. Bit-identical rows.
    pub fn decode_step_t_h_emb(&self, e: &Engine, tokens: &[u32], pos0: usize, cache: &mut Cache,
                               embd_dev: Option<(&CudaSlice<u8>, i32, usize)>)
                         -> Result<(Vec<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let (logits_d, h_seed) = self.decode_step_t_h_emb_dev(e, tokens, pos0, cache, embd_dev)?;
        Ok((e.dtoh(&logits_d)?, h_seed))
    }

    /// DEVICE-LOGITS verify forward (spec device-argmax lever): identical kernel chain to
    /// `decode_step_t_h_emb` but returns the [T, n_vocab] logits ON DEVICE — the accept walk
    /// argmaxes each column on-device and reads back ONE [T] u32 instead of dtoh'ing the full
    /// T x n_vocab f32 block (~1-4 MB + T host argmaxes, every round). Kernel dispatch is
    /// UNCHANGED (same decode-exact kernels); only the post-logits transfer moves.
    pub fn decode_step_t_h_emb_dev(&self, e: &Engine, tokens: &[u32], pos0: usize, cache: &mut Cache,
                               embd_dev: Option<(&CudaSlice<u8>, i32, usize)>)
                         -> Result<(CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let n_embd = self.cfg.n_embd as usize;
        let t = tokens.len();
        let (logits, x) = self.decode_step_t_core(e, tokens, pos0, cache, embd_dev, None)?;
        // h_seed for the next round = LAST column's pre-output_norm hidden ([n_embd]).
        let mut hs = vbuf(e, n_embd)?;   // fully written by copy_view_into below
        e.copy_view_into(&mut hs, 0, &x.slice((t - 1) * n_embd..t * n_embd), n_embd)?;
        Ok((logits, hs))
    }

    /// CORE verify forward: the `decode_step_t_h_emb_dev` kernel chain, returning the FULL
    /// pre-output_norm hidden stack x ([T, n_embd], any column extractable) and optionally
    /// filling a `VerifyCkpt` (retained per-layer state-rebuild inputs) for the REPLAY-FREE
    /// partial accept. `ckpt: None` => byte-for-byte the old behavior (the ckpt writes are pure
    /// retains/copies — they never change what any kernel computes).
    fn decode_step_t_core(&self, e: &Engine, tokens: &[u32], pos0: usize, cache: &mut Cache,
                          embd_dev: Option<(&CudaSlice<u8>, i32, usize)>,
                          mut ckpt: Option<&mut VerifyCkpt>)
                         -> Result<(CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let n_embd = cfg.n_embd as usize;
        let eps = cfg.rms_eps;
        let t = tokens.len();
        let pos_vec: Vec<i32> = (0..t).map(|i| (pos0 + i) as i32).collect();
        let pos_d = e.htod_i32(&pos_vec)?;

        // embed T tokens -> [T, n_embd] token-major (device gather on the spec hot loop)
        let mut x = match embd_dev {
            Some((g, qt, rb)) => e.embed_gather_device_t(g, tokens, n_embd, qt, rb)?,
            None => e.htod(&self.embd.gather(n_embd, tokens))?,
        };

        for (il, layer) in self.layers.iter().enumerate() {
            // DISPATCH-MIRRORED attn-input RMSNorm (FP-order lesson #8): eager decode fuses the
            // 1024-thread rms_norm_q8_1 ONLY when every mixer projection is q8_1-fast; layers with
            // Float projections (ssm_beta/ssm_alpha on layers 1/2/4 of the 9B NVFP4 GGUF) take the
            // UNFUSED 256-thread rms_norm. The verify norm must mirror that PER-LAYER choice —
            // blockDim changes the sum-of-squares reduce order, and the ULP shift amplifies through
            // the GDN recurrence into argmax flips (measured: 9B text prompt, 1 ULP at layer 2 ->
            // 2.3e-1 logit maxdiff at the head -> K=1..8 divergence at a 0.03-margin token).
            let mixer_fast = self.mixer_in_q8_1_fast(e, &layer.mixer);
            let norm_fused = std::env::var("BW24_NO_FUSE_NORMQ").is_err() && mixer_fast;
            let mut h = vbuf(e, t * n_embd)?;   // fully written by either rms_norm arm
            if norm_fused {
                e.rms_norm_decode(&x, layer.attn_norm.float_data(), &mut h, n_embd, t, eps)?;
            } else {
                e.rms_norm(&x, layer.attn_norm.float_data(), &mut h, n_embd, t, eps)?;
            }

            let mixed = match &layer.mixer {
                Mixer::Full(fa) => self.full_attn_verify(e, fa, &h, &pos_d, t, cache, il)?,
                Mixer::Linear(la) => {
                    // BATCHED linear verify (2026-07-03, the MTP-profit lever): one T-token pass —
                    // batched projections (weight read ONCE, hits the m=2-4 weight-resident matvec),
                    // carried-state conv (ssm_conv1d_tm_state), GDN prep on the prefill kernels, and
                    // ONE gdn_scan whose internal sequential t-loop is the SAME recurrence as T
                    // chained T=1 steps (bit-identical). Falls back to the sequential per-column
                    // chain when T < d_conv-1 (conv ring update needs T >= pad) — or when ANY
                    // projection is off the q8_1 fast path: matmul_decode_exact would route a Float
                    // tensor to cuBLAS at m=t (different FP accumulation than eager's per-token
                    // GEMV), so mixed-dtype layers stay on the eager-identical per-column chain.
                    // BW24_SPEC_M2 (lane/spec-m2): the t==2 batch rides the same arm — the conv
                    // wrapper handles t<pad with a pure-copy ring rebuild; see spec_m2() header.
                    if (t >= 3 || (t == 2 && spec_m2())) && mixer_fast && e.uses_q8_1_fast(&la.ssm_out) {
                        let want = ckpt.is_some();
                        let (out, stash) = self.linear_attn_verify_t(e, la, &h, t, cache, il, want)?;
                        if let (Some(ck), Some(st)) = (ckpt.as_deref_mut(), stash) {
                            ck.gdn[il] = Some(st);
                        }
                        out
                    } else {
                        let mut out = vbuf(e, t * n_embd)?;   // every col written by copy_into
                        let mut col_states: Option<Vec<(CudaSlice<f32>, CudaSlice<f32>)>> =
                            if ckpt.is_some() && t >= 2 { Some(Vec::with_capacity(t - 1)) } else { None };
                        for col in 0..t {
                            let mut h_col = vbuf(e, n_embd)?;   // fully written by copy_view_into
                            let src = h.slice(col * n_embd..(col + 1) * n_embd);
                            e.copy_view_into(&mut h_col, 0, &src, n_embd)?;
                            let m_col = self.linear_attn_decode(e, la, &h_col, cache, il)?;
                            e.copy_into(&mut out, col * n_embd, &m_col, n_embd)?;
                            // REPLAY-FREE ckpt: clone the chain's ACTUAL state after this column
                            // (pure dtod — cannot change any computed value). Last column skipped:
                            // rebuild targets are j <= t-1 columns.
                            if let Some(cs) = col_states.as_mut() {
                                if col + 1 < t {
                                    let rl = cache.recur[il].as_ref().unwrap();
                                    cs.push((e.clone_dtod(&rl.conv_state)?,
                                             e.clone_dtod(&rl.ssm_state)?));
                                }
                            }
                        }
                        if let (Some(ck), Some(cs)) = (ckpt.as_deref_mut(), col_states) {
                            ck.cols[il] = Some(cs);
                        }
                        out
                    }
                }
            };

            // DISPATCH-MIRRORED post-attn norm: eager residual_norm_ffn fuses add+norm+quant
            // (1024-thread add_rms_norm_q8_1) only for Dense FFNs whose gate+up are q8_1-fast;
            // otherwise (and for MoE) it runs the 256-thread fused add_rms_norm. Mirror per layer.
            let ffn_fuse = match &layer.ffn {
                crate::hybrid::Ffn::Dense { ffn_gate, ffn_up, .. } =>
                    std::env::var("BW24_NO_FUSE_NORMQ").is_err()
                        && e.uses_q8_1_fast(ffn_gate) && e.uses_q8_1_fast(ffn_up),
                crate::hybrid::Ffn::Moe(_) => false,
            };
            let mut x1 = vbuf(e, t * n_embd)?;   // fully written by add / add_rms_norm
            let mut z = vbuf(e, t * n_embd)?;    // fully written by rms_norm_decode / add_rms_norm
            if ffn_fuse {
                e.add(&x, &mixed, &mut x1, t * n_embd)?;
                e.rms_norm_decode(&x1, layer.post_attn_norm.float_data(), &mut z, n_embd, t, eps)?;
            } else {
                e.add_rms_norm(&x, &mixed, layer.post_attn_norm.float_data(), &mut x1, &mut z,
                               n_embd, t, eps)?;
            }
            // DECODE-EXACT FFN projections: force MMVQ for gate/up/down at any T to match the
            // T=1 decode FP accumulation order. At T>=5 the generic matmul/matmul_pre falls to dp4a
            // (128-thread, different FP sum order). At T=2-4 the batched MMVQ is already bit-identical.
            let ffn_out = match &layer.ffn {
                crate::hybrid::Ffn::Dense { ffn_gate, ffn_up, ffn_down } => {
                    let n_ff = ffn_gate.out_features();
                    let gate = e.matmul_decode_exact(ffn_gate, &z, t)?;
                    let up = e.matmul_decode_exact(ffn_up, &z, t)?;
                    let mut act = vbuf(e, t * n_ff)?;   // fully written by ffn_act
                    Self::ffn_act(e, &self.cfg, &gate, &up, &mut act, t * n_ff)?;
                    e.matmul_decode_exact(ffn_down, &act, t)?
                }
                crate::hybrid::Ffn::Moe(m) => self.moe_ffn_il(e, m, &z, t, il as u16)?,
            };
            let mut x2 = vbuf(e, t * n_embd)?;   // fully written by add
            e.add(&x1, &ffn_out, &mut x2, t * n_embd)?;
            x = x2;
        }

        let mut hn = vbuf(e, t * n_embd)?;   // fully written by rms_norm_decode
        e.rms_norm_decode(&x, self.output_norm.float_data(), &mut hn, n_embd, t, eps)?;
        let logits = e.matmul_decode_exact(&self.output, &hn, t)?;
        cache.pos += t;
        // Hidden stack for seeds/refresh-fills: pre-norm x (default) or post-norm hn (HPOST).
        Ok((logits, if spec_hpost() { hn } else { x }))
    }

    /// BATCHED linear-attn verify (T=K+1): the whole layer in ~10 launches instead of T x the
    /// T=1 decode chain (T x ~12 launches + T weight reads of the four projections). The GDN
    /// recurrence itself is inherently sequential — gdn_scan_s128 runs its internal t-loop with
    /// the SAME per-token math as chained T=1 calls (bit-identical state evolution); everything
    /// around it (projections, conv, prep, gated norm, out-proj) batches. Advances conv ring +
    /// ssm state exactly like T sequential decode steps.
    /// `want_stash`: additionally RETAIN the gdn-scan inputs (pure buffer keep-alives, zero extra
    /// kernels) so a partial accept can rebuild the state after any column prefix (REPLAY-FREE).
    fn linear_attn_verify_t(&self, e: &Engine, la: &LinearAttnLayer, h: &CudaSlice<f32>, t: usize,
                            cache: &mut Cache, il: usize, want_stash: bool)
                            -> Result<(CudaSlice<f32>, Option<GdnStash>), Box<dyn std::error::Error>> {
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

        // DECODE-EXACT projections: matmul_decode_exact forces the MMVQ (warp-per-row, 32-thread)
        // accumulation order for EVERY m, matching the T=1 decode path bit-for-bit. The generic
        // `matmul` at m>=5 falls to dp4a (128-thread, two-level reduce) which has a different FP
        // sum order — ULP differences propagate through gdn_scan and flip argmax on the 27B.
        // Q8 TRUNK-FUSION at T=1 (35B: wqkv+wqkv_gate both Q8_0): one fused2 launch, bit-identical
        // per (tensor,row) to the two m=1 MMVQ dispatches below — decode-exact contract holds.
        // VERIFY-TIER TRUNK FUSION (BW24_SPEC_FUSED_T, t=2-4): quantize h ONCE for every
        // fused-eligible same-input Q8_0 pair of this layer (35B wqkv+wqkv_gate; 9B
        // ssm_beta+ssm_alpha) — each fused2 batched launch then replaces two decode-exact
        // calls (each of which re-quantizes the same h + runs its own _b2/_b4 launch).
        // Bit-identical per (tensor,token,row) — see spec_fused_t().
        let h_q8_t = if spec_fused_t() && (2..=4).contains(&t)
            && ((e.uses_q8_1_fast(&la.wqkv) && e.uses_q8_1_fast(&la.wqkv_gate))
                || (e.uses_q8_1_fast(&la.ssm_beta) && e.uses_q8_1_fast(&la.ssm_alpha))) {
            Some(e.quantize_q8_1(h, t, cfg.n_embd as usize)?)
        } else { None };
        let (qkv_mixed, z) = {
            let mut fused = None;
            if t == 1 && e.uses_q8_1_fast(&la.wqkv) && e.uses_q8_1_fast(&la.wqkv_gate) {
                let (hq, hd) = e.quantize_q8_1(h, 1, cfg.n_embd as usize)?;
                fused = e.matmul_q8_fused2(&la.wqkv, &la.wqkv_gate, &hq, &hd)?;
            } else if let Some((hq, hd)) = h_q8_t.as_ref() {
                fused = e.matmul_q8_fused2_t(&la.wqkv, &la.wqkv_gate, hq, hd, t)?;
            }
            match fused {
                Some(pair) => pair,
                None => (e.matmul_decode_exact(&la.wqkv, h, t)?,
                         e.matmul_decode_exact(&la.wqkv_gate, h, t)?),
            }
        };
        // beta+alpha DUAL at T=1 (75% of p3 rounds run T=1 verify — p-min chain cuts): the dual
        // mr2 kernel is bit-identical per element to the m=1 MMVQ matmul_decode_exact dispatches
        // (same warp-per-row body, blockIdx.y picks the weight), so the decode-exact contract
        // holds; the run-spec battery is the arbiter. T>1 keeps the per-tensor decode-exact path.
        let (beta_raw, alpha) = if t == 1 {
            let (hq, hd) = e.quantize_q8_1(h, 1, cfg.n_embd as usize)?;
            match e.matmul_pre_dual_noscale(&la.ssm_beta, &la.ssm_alpha, &hq, &hd, 1)? {
                Some(((mut b, bs), (mut a, as_))) => {
                    if bs != 1.0 { e.scale_inplace(&mut b, bs, la.ssm_beta.out_features())?; }
                    if as_ != 1.0 { e.scale_inplace(&mut a, as_, la.ssm_alpha.out_features())?; }
                    (b, a)
                }
                // Q8_0 fused2 twin (9B stores beta/alpha as Q8_0): DISPATCH-MIRRORS the eager
                // decode's beta_alpha closure — the fused body is qmatvec_q8_0_mmvq verbatim,
                // bit-identical per row (kernel-check rel=0.00e0 gate), so decode==verify holds.
                None => match e.matmul_q8_fused2(&la.ssm_beta, &la.ssm_alpha, &hq, &hd)? {
                    Some((b, a)) => (b, a),
                    None => (e.matmul_decode_exact(&la.ssm_beta, h, 1)?,
                             e.matmul_decode_exact(&la.ssm_alpha, h, 1)?),
                },
            }
        } else {
            // fused-t twin (9B stores beta/alpha as Q8_0): same shared-quantize + one launch
            // contract as the wqkv pair above; 35B beta/alpha are Float -> None -> fallback.
            let mut fused = None;
            if let Some((hq, hd)) = h_q8_t.as_ref() {
                fused = e.matmul_q8_fused2_t(&la.ssm_beta, &la.ssm_alpha, hq, hd, t)?;
            }
            match fused {
                Some(pair) => pair,
                None => (e.matmul_decode_exact(&la.ssm_beta, h, t)?,
                         e.matmul_decode_exact(&la.ssm_alpha, h, t)?),
            }
        };

        // conv with CARRIED state + ring roll (T >= pad rides the input-column update kernel;
        // T < pad — the BW24_SPEC_M2 t=2 arm — rolls via the pure-copy ring rebuild).
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
        e.l2_norm_decode(&q_g, &mut q_l2, d_state, num_v * t, eps)?;
        let mut k_l2 = e.uninit(d_state * num_v * t)?;
        e.l2_norm_decode(&k_g, &mut k_l2, d_state, num_v * t, eps)?;
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
        // DECODE-EXACT out-projection: same MMVQ path as the T=1 decode (ssm_out at m>=5 would
        // fall to dp4a with a different FP reduction order — same class of bug as the input projs).
        let out = e.matmul_decode_exact(&la.ssm_out, &gn, t)?;
        let stash = if want_stash {
            Some(GdnStash { qkv_mixed, q_l2, k_l2, v_g, g_log, beta })
        } else { None };
        Ok((out, stash))
    }

    /// REPLAY-FREE partial-accept commit (2026-07-03): make the cache state == "committed through
    /// the first `j` verify columns" WITHOUT the legacy rollback + duplicate trunk replay.
    /// - Full-attn KV: truncate len to snapshot + j. The verify's appended rows for those columns
    ///   are bit-identical to what an eager T=1 chain writes (the decode-exact contract the
    ///   verify-probe gates), so keeping them == replaying them.
    /// - Linear layers, batched path: rebuild the conv ring by PURE COPIES (ring holds raw input
    ///   columns) and the ssm state by a prefix re-run of the SAME gdn_scan kernel (t=j) from the
    ///   snapshot state over the stash's identical inputs — the kernel's t-loop carries state in
    ///   registers and writes it once at the end, so iterations 0..j-1 are independent of T:
    ///   bit-identical to the verify's own state after j tokens == the eager chain state.
    /// - Linear layers, per-column path: restore the cloned actual state after column j-1.
    /// Caller guarantees 1 <= j <= t-1 (j==0 rounds take the legacy rollback; j==t is full accept).
    fn commit_verified_prefix(&self, e: &Engine, cache: &mut Cache,
                              snap: &crate::cache::CacheSnapshot, ckpt: &VerifyCkpt, j: usize)
                              -> Result<(), Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let ssm = cfg.ssm.as_ref().unwrap();
        let d_state = ssm.state_size as usize;
        let num_k = ssm.group_count as usize;
        let num_v = ssm.time_step_rank as usize;
        let d_conv = ssm.conv_kernel as usize;
        let conv_dim = d_state * num_k * 2 + d_state * num_v;
        let scale = 1.0 / (d_state as f32).sqrt();
        for il in 0..self.layers.len() {
            if let (Some(kvl), Some(saved)) = (cache.kv[il].as_mut(), snap.kv_len[il]) {
                kvl.len = saved + j;
                e.set_i32_one(&mut kvl.len_d, kvl.len as i32)?;
            }
            if let Some(rl) = cache.recur[il].as_mut() {
                if let Some(st) = &ckpt.gdn[il] {
                    let ring_old = snap.conv[il].as_ref().expect("snapshot missing conv");
                    e.ssm_conv_ring_rebuild(&st.qkv_mixed, ring_old, &mut rl.conv_state,
                                            conv_dim, j, d_conv)?;
                    let state_in = snap.ssm[il].as_ref().expect("snapshot missing ssm");
                    let mut o = e.uninit(d_state * num_v * j)?;   // scan output, discarded
                    e.gdn_scan_s128(&st.q_l2, &st.k_l2, &st.v_g, &st.g_log, &st.beta,
                                    state_in, &mut rl.ssm_state, &mut o, num_v, j, scale)?;
                } else if let Some(cols) = &ckpt.cols[il] {
                    let (c, s) = &cols[j - 1];
                    e.copy_into(&mut rl.conv_state, 0, c, c.len())?;
                    e.copy_into(&mut rl.ssm_state, 0, s, s.len())?;
                } else {
                    return Err("commit_verified_prefix: verify ckpt missing for linear layer".into());
                }
            }
        }
        cache.pos = snap.pos + j;
        Ok(())
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
            // DISPATCH-MIRRORED norms (FP-order lesson #8) — see decode_step_t_h_emb.
            let mixer_fast = self.mixer_in_q8_1_fast(e, &layer.mixer);
            let norm_fused = std::env::var("BW24_NO_FUSE_NORMQ").is_err() && mixer_fast;
            let mut h = vbuf(e, t * n_embd)?;   // fully written by either rms_norm arm
            if norm_fused {
                e.rms_norm_decode(&x, layer.attn_norm.float_data(), &mut h, n_embd, t, eps)?;
            } else {
                e.rms_norm(&x, layer.attn_norm.float_data(), &mut h, n_embd, t, eps)?;
            }
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
            let ffn_fuse = match &layer.ffn {
                crate::hybrid::Ffn::Dense { ffn_gate, ffn_up, .. } =>
                    std::env::var("BW24_NO_FUSE_NORMQ").is_err()
                        && e.uses_q8_1_fast(ffn_gate) && e.uses_q8_1_fast(ffn_up),
                crate::hybrid::Ffn::Moe(_) => false,
            };
            let mut x1 = vbuf(e, t * n_embd)?;   // fully written by add / add_rms_norm
            let mut z = vbuf(e, t * n_embd)?;    // fully written by rms_norm_decode / add_rms_norm
            if ffn_fuse {
                e.add(&x, &mixed, &mut x1, t * n_embd)?;
                e.rms_norm_decode(&x1, layer.post_attn_norm.float_data(), &mut z, n_embd, t, eps)?;
            } else {
                e.add_rms_norm(&x, &mixed, layer.post_attn_norm.float_data(), &mut x1, &mut z,
                               n_embd, t, eps)?;
            }
            let ffn_out = match &layer.ffn {
                crate::hybrid::Ffn::Dense { ffn_gate, ffn_up, ffn_down } => {
                    let n_ff = ffn_gate.out_features();
                    let gate = e.matmul_decode_exact(ffn_gate, &z, t)?;
                    let up = e.matmul_decode_exact(ffn_up, &z, t)?;
                    let mut act = vbuf(e, t * n_ff)?;   // fully written by ffn_act
                    Self::ffn_act(e, &self.cfg, &gate, &up, &mut act, t * n_ff)?;
                    e.matmul_decode_exact(ffn_down, &act, t)?
                }
                crate::hybrid::Ffn::Moe(m) => self.moe_ffn_il(e, m, &z, t, il as u16)?,
            };
            let mut x2 = vbuf(e, t * n_embd)?;   // fully written by add
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
        let mut hn = vbuf(e, t * n_embd)?;   // fully written by rms_norm_decode
        e.rms_norm_decode(&x, self.output_norm.float_data(), &mut hn, n_embd, t, eps)?;
        let logits = e.matmul_decode_exact(&self.output, &hn, t)?;
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

        // DECODE-EXACT Q/K/V projections: matmul_decode_exact forces the MMVQ (warp-per-row) path
        // for every m, matching the T=1 decode's FP accumulation order. matmul_pre at m>=5 would
        // fall to dp4a (128-thread, two-level reduce) with a different FP sum order.
        // Q8 TRUNK-FUSION at T=1: DISPATCH-MIRRORS the eager decode's fused3 (bit-identical body).
        let (qf, mut k, v) = {
            let mut fused = None;
            if t == 1 && e.uses_q8_1_fast(&fa.wq) && e.uses_q8_1_fast(&fa.wk)
                && e.uses_q8_1_fast(&fa.wv) {
                let (hq, hd) = e.quantize_q8_1(h, 1, n_embd)?;
                fused = e.matmul_q8_fused3(&fa.wq, &fa.wk, &fa.wv, &hq, &hd)?;
            } else if spec_fused_t() && (2..=4).contains(&t) && e.uses_q8_1_fast(&fa.wq)
                && e.uses_q8_1_fast(&fa.wk) && e.uses_q8_1_fast(&fa.wv) {
                // VERIFY-TIER TRUNK FUSION (BW24_SPEC_FUSED_T): one shared quantize + one
                // fused3 batched launch replaces three decode-exact calls (3 re-quantizes of
                // the same h + 3 _b2/_b4 launches). Bit-identical per (tensor,token,row).
                let (hq, hd) = e.quantize_q8_1(h, t, n_embd)?;
                fused = e.matmul_q8_fused3_t(&fa.wq, &fa.wk, &fa.wv, &hq, &hd, t)?;
            }
            match fused {
                Some(triple) => triple,
                None => (e.matmul_decode_exact(&fa.wq, h, t)?,
                         e.matmul_decode_exact(&fa.wk, h, t)?,
                         e.matmul_decode_exact(&fa.wv, h, t)?),
            }
        };
        // M3/Hy3 have no attention output gate — wq out is exactly q; skip the split.
        let gated = self.cfg.attn_out_gate();
        let (mut q, gate) = if gated {
            let mut q = vbuf(e, t * n_head * head_dim)?;      // fully written by q_gate_split
            let mut gate = vbuf(e, t * n_head * head_dim)?;   // fully written by q_gate_split
            e.q_gate_split(&qf, &mut q, &mut gate, head_dim, n_head, t)?;
            (q, Some(gate))
        } else { (qf, None) };

        let mut qn = vbuf(e, t * n_head * head_dim)?;   // fully written by rms_norm
        e.rms_norm(&q, fa.q_norm.float_data(), &mut qn, head_dim, n_head * t, eps)?;
        q = qn;
        let mut kn = vbuf(e, t * n_head_kv * head_dim)?;   // fully written by rms_norm
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

        // BIT-IDENTICAL VERIFY ATTENTION (spec-exactness fix): the FP accumulation order must be
        // byte-for-byte identical to the eager decode path. fa_prefill uses a different tile size
        // (BLOCK_Q=64, BK=32) and online-softmax structure than fa_decode's split-K + combine,
        // which changes FP summation order and can flip argmax at tight logit margins. Query row r
        // attends to keys [0..base_len+r+1) — each successive row sees one more key (the causal
        // property). This matches eager: decode appends k at len, then fa_decode sees t_kv = len+1
        // keys. The verify appends all T tokens first but bounds the key range per row.
        //
        // MULTI-ROW FUSED PATH (the long-ctx spec fix, 2026-07-03): when every row takes the vec
        // kernel (base_len+1 >= FA_VEC_MIN_TKV), ONE fa_decode_rows launch executes the exact
        // per-row program for all T rows (grid.z = row, per-row n_splits from the same
        // fa_split_keys formula) — replacing T x (2 launches + 2 dtod copies + 5 partial allocs)
        // and multiplying resident CTAs by T on a latency-bound kernel. Bit-identical per row by
        // construction; kernel-check pins rows-vs-loop byte identity, run-spec is the end gate.
        // Short ctx (any row below the vec crossover) and BW24_NO_FA_VEC/BW24_FA_ROWS_OFF keep the
        // per-row loop (whose fa_decode picks scalar/vec per row exactly like eager decode).
        let mut attn = vbuf(e, t * n_head * head_dim)?;   // fully written by every FA arm below
        let base_len = kvl.len - t;   // KV len BEFORE this round's T tokens were appended
        // T=1 INCLUDED (2026-07-05): p-min cuts the draft to 1 in ~75% of rounds on hard
        // (agentic) content — the old t>1 gate sent those rounds to the per-row loop (262us/row
        // + q-row copy + per-row allocs vs 93us/row through the fused kernel at grid.z=1, same
        // program). nsys accounting: 1088 of 1456 verify FA launches were T=1 escapees.
        // LEAN T=1 ARM (BW24_SPEC_LEAN, close35): at t==1, q IS one row and fa_decode on it is
        // the EXACT eager decode dispatch (vec_q_v2 + combine_f32; the rows pair measured +50us
        // at m=1). Byte-identical: kernel-check pins rows-vs-loop identity, and the per-row loop
        // at t=1 is fa_decode on the same q with zero-offset copies. Gates arbitrate.
        if spec_lean() && t == 1 {
            let t_kv = base_len + 1;
            let k_view = e.view_u8(&kvl.k, t_kv * ktb);
            let v_view = e.view_u8(&kvl.v, t_kv * vtb);
            e.fa_decode(&q, &k_view, &v_view, &mut attn, head_dim, n_head, n_head_kv,
                        t_kv, scale, ktb, vtb)?;
        } else if e.fa_rows_eligible(base_len, head_dim) {
            let k_view = e.view_u8(&kvl.k, (base_len + t) * ktb);
            let v_view = e.view_u8(&kvl.v, (base_len + t) * vtb);
            e.fa_decode_rows(&q, &k_view, &v_view, &mut attn, head_dim, n_head, n_head_kv,
                             base_len, t, scale, ktb, vtb)?;
        } else {
            for r in 0..t {
                let t_kv_r = base_len + r + 1; // this row sees keys [0..t_kv_r)
                let k_view_r = e.view_u8(&kvl.k, t_kv_r * ktb);
                let v_view_r = e.view_u8(&kvl.v, t_kv_r * vtb);
                // copy q row into an owned buffer (fa_decode takes &CudaSlice, not CudaView)
                let mut q_row = vbuf(e, n_head * head_dim)?;   // fully written by copy_view_into
                let q_src = q.slice(r * n_head * head_dim..(r + 1) * n_head * head_dim);
                e.copy_view_into(&mut q_row, 0, &q_src, n_head * head_dim)?;
                let mut attn_row = vbuf(e, n_head * head_dim)?;   // fully written by fa_decode
                e.fa_decode(&q_row, &k_view_r, &v_view_r, &mut attn_row, head_dim, n_head, n_head_kv, t_kv_r, scale, ktb, vtb)?;
                e.copy_into(&mut attn, r * n_head * head_dim, &attn_row, n_head * head_dim)?;
            }
        }

        let attn_g = match &gate {
            Some(gate) => {
                let mut gsig = vbuf(e, t * n_head * head_dim)?;   // fully written by sigmoid
                e.sigmoid(gate, &mut gsig, t * n_head * head_dim)?;
                let mut ag = vbuf(e, t * n_head * head_dim)?;     // fully written by mul
                e.mul(&attn, &gsig, &mut ag, t * n_head * head_dim)?;
                ag
            }
            None => attn,
        };
        // DECODE-EXACT wo projection: at m>=5 (K=4+ with pending) the generic matmul would use dp4a
        // (128-thread, different FP sum order than MMVQ). Force MMVQ for bit-identity with decode.
        Ok(e.matmul_decode_exact(&fa.wo, &attn_g, t)?)
    }

    /// Greedy MTP speculative decode (§B). Token-identical to `generate(prompt, max_new)` but uses
    /// the NextN head to draft K tokens then verifies them in one batched target forward.
    /// Returns (generated tokens, total_drafted, total_accepted) so the caller can report
    /// acceptance rate. `k` = draft length per round.
    ///
    /// GRAPH DRAFT (stage 2 of graph-grade spec): when the model is all-Dense and the MTP head is
    /// Dense (no MoE host readbacks), the fixed-shape T=1 MTP forward is CUDA-graph-captured ONCE
    /// and replayed per draft step — the ~40 eager launches per drafted token collapse into one
    /// graph dispatch; only the 4-byte token id (and 4-byte p-min confidence) round-trip per step.
    /// Event tracking is disabled for the whole call (generate_graph pattern) so every buffer the
    /// captured graph references is event-free; the spec loop is strictly single-stream.
    /// BW24_SPEC_NOGRAPH=1 forces the eager draft chain.
    /// SAMPLED mode (BW24_SPEC_TEMP>0) has its OWN capture (gumbel-perturbed in-graph argmax,
    /// device Philox event counter, persistent q retention) — graph-vs-eager sampled streams are
    /// bit-identical for the same (seed, prompt, K, temp); see the sampled-graph setup in
    /// generate_spec_inner2.
    /// Multi-turn session: trunk cache + MTP draft scratch persist across generate calls, so
    /// turn N+1 primes ONLY its new suffix (the 124k-conversation daily pattern — re-priming a
    /// 32k history costs ~54s; a suffix prime costs seconds). APPEND-ONLY by construction: the
    /// hybrid linear-attn states are in-place (no position index), so a session can extend but
    /// never rewind — `committed` is the exact token list whose state the caches hold (includes
    /// any overshoot tokens past max_new; the caller renders from `committed`, not its own echo).
    pub fn new_session(&self, e: &Engine, max_ctx: usize)
                       -> Result<SpecSession, Box<dyn std::error::Error>> {
        Ok(SpecSession {
            cache: Cache::new(e, &self.cfg, max_ctx)?,
            scratch: MtpScratch::new(e, &self.cfg, max_ctx)?,
            committed: Vec::new(),
            last_h: None,
            next_pred: None,
            sctr: 0,
            uctr: 0,
        })
    }

    /// One spec-decode turn on a live session. `suffix` = the NEW tokens only (turn N+1's user
    /// message rendered through the chat template continuation). Returns (new tokens emitted,
    /// drafted, accepted); session.committed grows by suffix + emitted.
    pub fn generate_spec_session(&self, e: &Engine, sess: &mut SpecSession, suffix: &[u32],
                                 max_new: usize, k: usize)
                                 -> Result<(Vec<u32>, usize, usize), Box<dyn std::error::Error>> {
        self.generate_spec_session_sampled(e, sess, suffix, max_new, k, None)
    }

    /// Serve-path sampled spec: routes the burst through the rejection-sampling verify with
    /// per-SESSION Philox continuity (sess.sctr/uctr). None = env-driven (CLI) or greedy.
    /// Filters (top-k/p/min-p) apply SYMMETRICALLY to draft q and verify p — distribution-exact
    /// for the filtered target (feat/filtered-spec).
    pub fn generate_spec_session_sampled(&self, e: &Engine, sess: &mut SpecSession, suffix: &[u32],
                                 max_new: usize, k: usize, sampling: Option<SpecSampling>)
                                 -> Result<(Vec<u32>, usize, usize), Box<dyn std::error::Error>> {
        let mtp_dense = self.mtp.as_ref()
            .map(|m| matches!(m.ffn, crate::hybrid::Ffn::Dense { .. })).unwrap_or(false);
        let trunk_dense = self.layers.iter()
            .all(|l| matches!(l.ffn, crate::hybrid::Ffn::Dense { .. }));
        // FULL_PREC forces the EAGER draft: the graph capture would enclose cuBLASLt f32 GEMV
        // (the FloatBf16 else-branches) and a bf16_to_f32 dequant alloc — neither is stream-capture
        // safe. Eager rides matmul/matmul_decode_exact, which dequant FloatBf16 on use. (§item 2.)
        let graph_draft = std::env::var("BW24_SPEC_NOGRAPH").is_err()
            && mtp_dense && trunk_dense && k + 2 < 96 && !crate::model::full_prec_enabled();
        let was_tracking = e.ctx().is_event_tracking();
        if graph_draft && was_tracking { unsafe { e.ctx().disable_event_tracking(); } }
        let r = self.generate_spec_inner2(e, suffix, max_new, k, graph_draft, Some(sess), sampling);
        if graph_draft && was_tracking { unsafe { e.ctx().enable_event_tracking(); } }
        let (out, d, a) = r?;
        Ok((out, d, a))
    }

    pub fn generate_spec(&self, e: &Engine, prompt: &[u32], max_new: usize, k: usize)
                         -> Result<(Vec<u32>, usize, usize), Box<dyn std::error::Error>> {
        let mtp_dense = self.mtp.as_ref()
            .map(|m| matches!(m.ffn, crate::hybrid::Ffn::Dense { .. })).unwrap_or(false);
        let trunk_dense = self.layers.iter()
            .all(|l| matches!(l.ffn, crate::hybrid::Ffn::Dense { .. }));
        // FULL_PREC forces eager (see generate_spec_session note): CUDA graph capture cannot
        // enclose cuBLASLt f32 GEMV or the bf16_to_f32 dequant alloc the FloatBf16 path needs.
        let graph_draft = std::env::var("BW24_SPEC_NOGRAPH").is_err()
            && mtp_dense && trunk_dense && k + 2 < 96 && !crate::model::full_prec_enabled();
        if !graph_draft {
            return self.generate_spec_inner2(e, prompt, max_new, k, false, None, None);
        }
        let was_tracking = e.ctx().is_event_tracking();
        if was_tracking { unsafe { e.ctx().disable_event_tracking(); } }
        let r = self.generate_spec_inner2(e, prompt, max_new, k, true, None, None);
        if was_tracking { unsafe { e.ctx().enable_event_tracking(); } }
        r
    }

    fn generate_spec_inner2(&self, e: &Engine, prompt: &[u32], max_new: usize, k: usize,
                           graph_draft: bool, mut sess: Option<&mut SpecSession>,
                           sampling: Option<SpecSampling>)
                         -> Result<(Vec<u32>, usize, usize), Box<dyn std::error::Error>> {
        assert!(k >= 1, "k must be >= 1");
        let mtp = self.mtp.as_ref().expect("generate_spec requires an MTP head (nextn_predict_layers>0)");
        let n_vocab = self.output.out_features();
        // FR-Spec: the draft head may be TRIMMED (fewer rows than n_vocab); the draft argmax runs
        // over the draft vocab and the winning index maps through d2t to a TARGET token id.
        // Everything downstream (verify/accept/commit) sees target ids only — exactness unchanged.
        let d_vocab = mtp.shared_head_head.as_ref().unwrap_or(&self.output).out_features();
        let n_embd = self.cfg.n_embd as usize;
        // SESSION MODE: reuse the live cache/scratch, prime only the suffix. `base` = tokens
        // already committed (their state is in the caches); 0 = fresh single-shot call.
        let session_mode = sess.is_some();
        let max_ctx = match sess.as_ref() {
            Some(s) => s.cache.max_ctx,
            None => prompt.len() + max_new + k + 8,
        };
        let mut own_cache;
        let mut own_scratch;
        let (cache, scratch, mut sess_tail): (&mut Cache, &mut MtpScratch,
                                              Option<(&mut Vec<u32>, &mut Option<CudaSlice<f32>>,
                                                      &mut Option<u32>, &mut u32, &mut u32)>) =
            match sess.take() {
                Some(sr) => {
                    let SpecSession { cache, scratch, committed, last_h, next_pred, sctr: s_sctr, uctr: s_uctr } = sr;
                    (cache, scratch, Some((committed, last_h, next_pred, s_sctr, s_uctr)))
                }
                None => {
                    own_cache = Cache::new(e, &self.cfg, max_ctx)?;
                    // Persistent scratch = max_ctx rows (~2KB/token quantized).
                    own_scratch = MtpScratch::new(e, &self.cfg, max_ctx)?;
                    (&mut own_cache, &mut own_scratch, None)
                }
            };
        let base = cache.pos;
        // PERSISTENT DRAFT KV (the only mode since 2026-07-08 — the legacy round-local scratch,
        // BW24_SPEC_KVLOCAL, measured -35 acceptance pts on the 27B p3 sweep and was removed;
        // acceptance-only — exactness is verify's job either way).
        // HIDDEN-PAIRING CONVENTION (DEFAULT = predecessor-row, 2026-07-04 — the 27B acceptance
        // unlock, +16pts): the MTP head is TRAINED on rows pairing token x_p with the trunk
        // hidden of its PREDECESSOR h_{p-1} (the reference engine's mtp_update shifts the target
        // hiddens right by one; its draft step 0 feeds (id_last, TRUE hidden of the row id_last
        // was sampled from)). bw24's historical convention paired SAME-ROW (x_p, h_p) in the fill
        // and seeded chain step 0 through an extra MTP pass on a duplicated token (the
        // pseudo-seed) — measured 27B p2 K=3 acceptance 0.569 vs 0.731, p3 0.445 vs 0.63+, and
        // the chain steps j>=1 were already predecessor-shaped, so ONLY the fill + step-0 seed
        // move. The fill shifts by one and the chain seeds from the predecessor's true hidden
        // DIRECTLY (vh_seed / vx[j-1]) — the pseudo pass disappears (one MTP-block pass saved
        // per round on top of the acceptance win). Draft-quality-only: exactness stays the
        // verify's job either way. (The legacy same-row pairing seam, BW24_SPEC_HSAME, and its
        // pseudo-seed passes were removed 2026-07-08 — predecessor pairing won by +16 acc pts;
        // the legacy round-local scratch, BW24_SPEC_KVLOCAL, went with it.)
        // REPLAY-FREE PARTIAL ACCEPT (default, 2026-07-03): partial rounds keep the verify's own
        // bit-identical committed-prefix state (KV truncate + recur rebuild from the VerifyCkpt)
        // and leave the bonus PENDING — no duplicate trunk pass (profiled ~0.54 extra full weight
        // reads/round at long ctx). BW24_SPEC_REPLAY=1 restores the legacy rollback+replay (A/B
        // + fallback seam).
        let spec_replay = std::env::var("BW24_SPEC_REPLAY").is_ok();
        // TRUE-HIDDEN REFRESH (default in persistent-draft-KV mode): every round overwrites the
        // committed positions' scratch entries from the verify's exact hiddens (mtp_kv_fill batch)
        // instead of keeping chain-approximate entries. BW24_SPEC_NOREFRESH=1 = legacy (A/B seam).
        let refresh = std::env::var("BW24_SPEC_NOREFRESH").is_err();

        // prime: BATCHED cache prime (prime_cache — the measured #1 e2e gap: tokenwise primed at
        // ~102/38 tok/s vs the engine's ~2000-5900 tok/s batched prefill). prime_cache returns the
        // full pre-output_norm hidden stack [T, n_embd], which IS prompt_h (the persistent-draft-KV
        // mtp_kv_fill input) — no per-token collection needed. Prompts below PRIME_MIN_T, and
        // BW24_PRIME_TOKENWISE=1 (the escape seam), take the tokenwise decode_step_h loop.
        // EMPTY-SUFFIX CONTINUATION (serve bursts): a session turn with NO new tokens resumes
        // generation exactly where the last turn stopped — no prime at all. The stashed
        // `next_pred` plays prime_logits' argmax role (it IS the argmax of the logits after
        // committed.last()); `last_h` seeds the predecessor pairing below. Fresh calls and
        // non-empty suffixes take the normal path.
        let continuation = prompt.is_empty();
        if continuation {
            assert!(session_mode, "empty prompt requires a session");
            assert!(sess_tail.as_ref().map_or(false, |(c, lh, np, _, _)|
                !c.is_empty() && lh.is_some() && np.is_some()),
                "empty-suffix continuation needs a primed session (committed + last_h + next_pred)");
        }
        let mut prime_logits;
        let mut prompt_h: Option<CudaSlice<f32>> = None;
        let t_prime = std::time::Instant::now();
        let batched_prime = !continuation
            && prompt.len() >= crate::hybrid_forward::PRIME_MIN_T
            && std::env::var("BW24_PRIME_TOKENWISE").is_err();
        if continuation {
            prime_logits = Vec::new();
        } else if batched_prime {
            let (l, _h_seed, hiddens) = self.prime_cache(e, prompt, &mut *cache)?;
            prime_logits = l;
            prompt_h = Some(hiddens);
        } else {
            prime_logits = Vec::new();
            prompt_h = Some(e.uninit(prompt.len() * n_embd)?);
            for (i, &tok) in prompt.iter().enumerate() {
                let (l, h) = self.decode_step_h(e, tok, &mut *cache)?;
                if let Some(ph) = prompt_h.as_mut() { e.copy_into(ph, i * n_embd, &h, n_embd)?; }
                prime_logits = l;
            }
        }
        e.stream().synchronize()?;
        // Harness timing contract (see crate::PRIME_NANOS): gen-only throughput without the
        // prime-subtraction hack.
        crate::PRIME_NANOS.store(t_prime.elapsed().as_nanos() as u64,
                                 std::sync::atomic::Ordering::Relaxed);

        // Resident embed table (model-lifetime, lazy first-use upload; kills the per-draft-token
        // and per-verify host-dequant+htod that nsys measured at 84% of spec API time).
        let embd_gpu = self.embd_gpu.get_or_init(|| {
            e.upload_u8(&self.embd.raw).expect("embed table upload")
        });
        let (embd_qt, embd_rb) = self.embd.qt_and_row_bytes(n_embd);
        let mut out: Vec<u32> = Vec::with_capacity(max_new);
        let mut total_drafted = 0usize;
        let mut total_accepted = 0usize;

        // First generated token = argmax of the prompt's last logits (== greedy's first token).
        // Emit it, then FEED it to establish the loop invariant below.
        let mut last_token = if continuation {
            sess_tail.as_ref().unwrap().2.unwrap()
        } else { argmax(&prime_logits) as u32 };
        out.push(last_token);
        if continuation {
            // draft-KV invariant: entries [0..base) are the session's exact fills; truncate any
            // overhang so the chain's first append lands at slot base (== committed.len()).
            scratch.set_len(e, base)?;
        }
        // INVARIANT at loop top: `last_token` is the most-recently-committed/emitted token, its
        // KV+recur state IS in `cache` (cache.pos = position right AFTER last_token), `last_pred`
        // is the greedy ARGMAX of the logits that predict the token FOLLOWING last_token, and
        // `h_seed` = last_token's pre-output_norm hidden. Establish it by feeding last_token once
        // (mirrors plain greedy). DEVICE-ARGMAX lever: the accept walk only ever consumes the
        // argmax of those logits — never the full vector — so a host u32 replaces the Vec<f32>.
        // --- SAMPLED SPEC (BW24_SPEC_TEMP>0, research/sampled-spec-impl-map.md): rejection-
        // sampling verify (Leviathan/Chen) — accept draft x at u < p(x)/q(x), resample from
        // norm(max(0,p-q)) on reject, bonus sampled from p on full accept. Counter-based Philox
        // everywhere (seed, event) -> reproducible. temp==0/unset = the greedy path, untouched.
        let sp = sampling.unwrap_or_else(|| SpecSampling {
            temp: std::env::var("BW24_SPEC_TEMP").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0),
            seed: std::env::var("BW24_SEED").ok().and_then(|v| v.parse().ok()).unwrap_or(42),
            top_k: std::env::var("BW24_TOP_K").ok().and_then(|v| v.parse().ok()).unwrap_or(0),
            top_p: std::env::var("BW24_TOP_P").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0),
            min_p: std::env::var("BW24_MIN_P").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0),
            penalty_last_n: std::env::var("BW24_PENALTY_LAST_N").ok().and_then(|v| v.parse().ok()).unwrap_or(0),
            penalty_repeat: std::env::var("BW24_PENALTY_REPEAT").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0),
            penalty_freq: std::env::var("BW24_PENALTY_FREQ").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0),
            penalty_present: std::env::var("BW24_PENALTY_PRESENT").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0),
        });
        let (sp_temp, sp_seed) = (sp.temp, sp.seed);
        let sampled = sp_temp > 0.0;
        // Trimmed heads: q lives on the trimmed vocab; accept gathers use the TRIMMED index and
        // the residual scatters q into target-id space (q=-inf off-trim — the head cannot propose
        // those, so their residual mass is p(x), correct by construction).
        let d2t_dev: Option<CudaSlice<u32>> = if sampled {
            match &mtp.d2t { Some(map) => Some(e.htod_u32_v(map)?), None => None }
        } else { None };
        let mut q_full_buf: Option<CudaSlice<f32>> = None;
        // Counters resume from the session (burst continuity: randomness must never repeat
        // across generate_spec_session calls); one-shot callers start at (0,0). Read through
        // sess_tail — `sess` was take()n into it above, so sess.as_ref() here is always None.
        let mut sctr: u32 = sess_tail.as_ref().map(|(_, _, _, s, _)| **s).unwrap_or(0);
        let mut uctr: u32 = sess_tail.as_ref().map(|(_, _, _, _, u)| **u).unwrap_or(0);
        // host Philox4x32-10 (mirrors spec_sample.cu; independent stream via ctr_lo tag)
        let host_u01 = |seed: u64, ctr: u32| -> f32 {
            let (m0, m1) = (0xD2511F53u32, 0xCD9E8D57u32);
            let (mut c0, mut c1, mut c2, mut c3) = (0xFFFF_FFFEu32, ctr, 0u32, 0u32);
            let (mut k0, mut k1) = ((seed & 0xFFFF_FFFF) as u32, (seed >> 32) as u32);
            for _ in 0..10 {
                let (h0, l0) = (((m0 as u64 * c0 as u64) >> 32) as u32, m0.wrapping_mul(c0));
                let (h1, l1) = (((m1 as u64 * c2 as u64) >> 32) as u32, m1.wrapping_mul(c2));
                let (n0, n1, n2, n3) = (h1 ^ c1 ^ k0, l1, h0 ^ c3 ^ k1, l0);
                c0 = n0; c1 = n1; c2 = n2; c3 = n3;
                k0 = k0.wrapping_add(0x9E3779B9); k1 = k1.wrapping_add(0xBB67AE85);
            }
            (c0 as f32 + 1.0) * (1.0 / 4294967296.0)
        };
        let mut draft_logits: Vec<CudaSlice<f32>> = Vec::new();   // retained head logits (q), per slot
        let mut draft_stats: Vec<(f32, f32, f32)> = Vec::new();   // (row_max, th_e, z_e) per slot
        let mut perturb_buf: Option<CudaSlice<f32>> = None;       // gumbel scratch (max(n_vocab,d_vocab))
        let mut sample_tok = e.alloc_u32_zeroed(1)?;              // residual/bonus sample out
        let mut col_buf: Option<CudaSlice<f32>> = None;           // materialized verify column
        // Penalties (v2.1): applied to COPIES of q rows and p columns symmetrically (exactness
        // for the penalized+filtered target). History = generated tokens, host-tracked window.
        let pen_on = sampled && sp.penalty_last_n > 0
            && (sp.penalty_repeat != 1.0 || sp.penalty_freq != 0.0 || sp.penalty_present != 0.0);
        let mut pen_hist: Vec<u32> = if pen_on {
            prompt.iter().rev().take(64).rev().cloned().collect()  // llama-parity: history spans prompt tail too
        } else { Vec::new() };
        let mut pen_hist_d: Option<CudaSlice<u32>> = None;
        let mut pcol_buf: Option<CudaSlice<f32>> = None;          // penalized p-column scratch
        let (init_logits, h_seed0) = self.decode_step_h(e, last_token, &mut *cache)?;
        let mut last_pred = argmax(&init_logits) as u32;
        // sampled mode: p-distribution after last_token, for the j==0/base==0 accept test.
        let mut last_col_logits: Option<CudaSlice<f32>> = if sampled { Some(e.htod(&init_logits)?) } else { None };
        let mut last_col_stats: Option<(f32, f32, f32)> = None;
        // PERSISTENT h_seed buffer (allocated BEFORE any graph capture so no captured scratch can
        // alias it): every path that updates the round seed copies INTO it — no per-round allocs,
        // stable pointer for the graph-draft round-start copy.
        let mut h_seed_buf = e.clone_dtod(&h_seed0)?;
        // Predecessor-pairing trackers: `fill_prev` = trunk hidden AT the last COMMITTED row (the
        // predecessor of the next verify's col 0 — the reference's carried pending-h analogue;
        // also the predecessor-row hidden for the round-0 legacy-replay seed). At round 0 that
        // row is last_token's own (h_seed0). The chain step-0 seed under the pairing default =
        // hidden of the row BEFORE last_token = the prompt's last row at round 0 (h_seed_buf
        // overwritten below).
        let mut fill_prev = e.clone_dtod(&h_seed0)?;
        {
            if let Some(ph) = &prompt_h {
                let np = prompt.len();
                e.copy_view_into(&mut h_seed_buf, 0, &ph.slice((np - 1) * n_embd..np * n_embd),
                                 n_embd)?;
            } else if continuation {
                if let Some((_, lh, _, _, _)) = sess_tail.as_ref() {
                    if let Some(lh) = lh.as_ref() { e.copy_into(&mut h_seed_buf, 0, lh, n_embd)?; }
                }
            }
        }
        // Persistent device prediction slots for the accept walk (max k+1 verify columns).
        let mut preds_d = e.alloc_u32_zeroed(k + 2)?;

        let debug_spec = std::env::var("BW24_DEBUG_SPEC").is_ok();
        // BW24_SPEC_STATS=1: per-slot accept histogram + draft-length histogram, printed once at
        // the end. Metric normalization vs the reference engine: BOTH engines count
        // accepted/drafted where the chain stopped at p-min and the sub-threshold token is
        // discarded uncounted — per-slot decay + chain-length mix are the extra dimensions.
        let spec_stats = std::env::var("BW24_SPEC_STATS").is_ok();
        let mut st_drafted = vec![0usize; k];
        let mut st_accepted = vec![0usize; k];
        let mut st_len_hist = vec![0usize; k + 1];
        let mut st_full = 0usize;
        // P-MIN CONFIDENCE GATE (BW24_SPEC_PMIN, the serve script's --spec-draft-p-min mechanism):
        // stop the draft chain early when the head's softmax confidence in its own pick drops
        // below p_min. Hoisted above the loop: the graph capture bakes the prob kernels iff on.
        static PMIN: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
        let p_min = *PMIN.get_or_init(|| {
            std::env::var("BW24_SPEC_PMIN").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0)
        });
        // ZERO-DRAFT ROUNDS (BW24_SPEC_PMIN0=1, vendored from llama.cpp's draft gating): let the
        // p-min gate apply at j==0 too, so a low-confidence round drafts NOTHING and the verify
        // batch is just the pending bonus (m=1 = a plain decode step). llama's 35B win rides
        // exactly this — draft acceptance 76% at mean len 2.5 because unpredictable stretches
        // never pay draft+verify overhead. Only legal when a pending bonus exists (an empty
        // verify batch is not); the j==0 exemption stays for pending-less rounds.
        let pmin0 = std::env::var("BW24_SPEC_PMIN0").map(|v| v == "1").unwrap_or(false);

        // --- GRAPH DRAFT setup: persistent I/O buffers + ONE capture (2 warmups inside). The
        // warmups mutate scratch len_d / pos / tok / seed — all reset at every round start, so the
        // only restore needed is the scratch counter. Capture failure (e.g. a non-capturable
        // cuBLAS path in an exotic head) falls back to the eager draft chain.
        let mut g_tok = e.alloc_u32_zeroed(1)?;
        let mut g_pos = e.htod_i32(&[0])?;
        let mut g_seed = e.zeros(n_embd)?;
        let mut g_p = e.zeros(1)?;
        let mut draft_graph: Option<cudarc::driver::CudaGraph> = None;
        if graph_draft && !sampled {
            let cap_res = e.capture_graph(|e| {
                self.mtp_head_forward_cap(e, mtp, &mut g_tok, &mut g_pos, &mut g_seed, &mut g_p,
                                          &mut *scratch, p_min > 0.0, true, embd_gpu, embd_qt,
                                          embd_rb, d_vocab, None)
            });
            match cap_res {
                Ok(g) => { scratch.set_len(e, 0)?; draft_graph = Some(g); }
                Err(err) => {
                    scratch.set_len(e, 0)?;
                    if debug_spec { eprintln!("[spec] draft-graph capture failed ({err}); eager fallback"); }
                }
            }
        }
        // --- SAMPLED GRAPH DRAFT setup (step 3 of the sampled-spec arc): a SECOND capture, own
        // graph object, built only when sampled && graph-eligible — the greedy capture above is
        // untouched (and skipped when sampled: its graph would never be launched). Same head
        // forward, but the in-graph argmax reads GUMBEL-PERTURBED logits; the Philox event
        // counter lives in the persistent device g_ctr (bumped in-graph, host-seeded from sctr
        // once per round); the raw head logits land in the persistent g_q for the host's
        // per-replay async D2D into the round's q slot (q_slots, K x d_vocab, allocated once).
        // seed/temp are capture-time constants — capture happens once per generate call, exactly
        // like the greedy graph (no extra recapture cost beyond today's per-call capture).
        let mut g_ctr = e.alloc_u32_zeroed(1)?;
        let mut g_q = e.zeros(if sampled { d_vocab } else { 1 })?;
        let mut g_perturb = e.zeros(if sampled { d_vocab } else { 1 })?;
        let mut q_slots: Vec<CudaSlice<f32>> = Vec::new();
        let mut draft_graph_s: Option<cudarc::driver::CudaGraph> = None;
        // COMPOSITION RULE (fspec x gsd merge): the in-graph chain samples from the RAW
        // softmax — it can hold neither per-row filter stats nor the varying penalty history.
        // The sampled graph therefore engages only in the PURE-TEMP regime; filters/penalties
        // force the eager draft (which computes stats/penalties per row).
        let pure_temp = sp.top_k == 0 && sp.top_p >= 1.0 && sp.min_p <= 0.0 && !pen_on;
        if graph_draft && sampled && pure_temp {
            let cap_res = e.capture_graph(|e| {
                self.mtp_head_forward_cap(e, mtp, &mut g_tok, &mut g_pos, &mut g_seed, &mut g_p,
                                          &mut *scratch, p_min > 0.0, true, embd_gpu, embd_qt,
                                          embd_rb, d_vocab,
                                          Some((&mut g_ctr, &mut g_perturb, &mut g_q,
                                                sp_seed, sp_temp)))
            });
            match cap_res {
                Ok(g) => {
                    scratch.set_len(e, 0)?;
                    for _ in 0..k { q_slots.push(e.zeros(d_vocab)?); }
                    draft_graph_s = Some(g);
                }
                Err(err) => {
                    scratch.set_len(e, 0)?;
                    if debug_spec { eprintln!("[spec] sampled draft-graph capture failed ({err}); eager fallback"); }
                }
            }
        }
        // PERSISTENT DRAFT KV: fill the MTP block's K/V for every prompt position from the exact
        // trunk hiddens collected during prime — ONE batched K/V-only pass (overwrites any
        // capture-warmup garbage; capture left len at 0). last_token (the init feed) needs no
        // fill: the first chain step processes it and appends its entry at slot prompt.len().
        if let Some(ph) = &prompt_h {
            // SESSION: rows [0..base) are the previous turns' exact fills (refresh overwrote them
            // with true verify hiddens) — truncate any draft overhang, fill ONLY the suffix at
            // global positions [base..base+tp). Fresh call: base==0, identical to before.
            scratch.set_len(e, base)?;
            // CHUNKED FILL (long-ctx OOM fix, 2026-07-05): mtp_kv_fill's transients scale with its
            // T (concat = T*2*n_embd*4B — 1.5GB at 40k) and its concat loop is 2*T launches. The
            // fill is a pure sequential append, so chunking is exact: each chunk appends its rows
            // at pos0=base+start with the identical per-row math. Same knob as the trunk prime.
            let fill_chunk: usize = std::env::var("BW24_PRIME_CHUNK").ok()
                .and_then(|v| v.parse().ok()).unwrap_or(4096);
            let tp = prompt.len();
            let fill_chunk = if fill_chunk == 0 { tp } else { fill_chunk };
            let mut start = 0usize;
            while start < tp {
                let end = (start + fill_chunk).min(tp);
                let tc = end - start;
                {
                    // PREDECESSOR pairing: row i gets h[i-1]; global row 0 a zeros row (the
                    // reference engine's initial pending-h is zeroed too); a session turn's row 0
                    // gets the PREVIOUS turn's last committed hidden (sess.last_h). Per chunk:
                    // rows start..end read h[start-1..end-1] — one dtod into a chunk buffer.
                    let mut phs = e.zeros(tc * n_embd)?;
                    let (src_lo, dst_off) = if start == 0 { (0, n_embd) } else { ((start - 1) * n_embd, 0) };
                    let n_copy = if start == 0 { (tc - 1) * n_embd } else { tc * n_embd };
                    if start == 0 {
                        if let Some((_, lh, _, _, _)) = sess_tail.as_ref() {
                            if let Some(lh) = lh.as_ref() {
                                e.copy_into(&mut phs, 0, lh, n_embd)?;
                            }
                        }
                    }
                    if n_copy > 0 {
                        e.copy_view_into(&mut phs, dst_off, &ph.slice(src_lo..src_lo + n_copy), n_copy)?;
                    }
                    self.mtp_kv_fill(e, mtp, &prompt[start..end], &phs, base + start, &mut *scratch,
                                     embd_gpu, embd_qt, embd_rb)?;
                }
                start = end;
            }
        }
        let mut round = 0usize;
        // (Adaptive-K — BW24_SPEC_ADAPT, acceptance-EMA draft length — measured an HONEST LOSS
        // to static per-class optima on 2026-07-07 (115.0/85.8/73.4 vs 121.6/92.7/75.6, EMA lag)
        // and was removed 2026-07-08; rig5090.jsonl has the record. K is fixed per call.)
        // PERSISTENT snapshot buffers: allocate ONCE, refresh in place each round (was 2 fresh
        // D2D clones per linear layer per round = 48 allocs + ~50MB of pool churn per round).
        let mut snap = cache.snapshot(e)?;
        // BONUS FOLD (2026-07-04): after a FULL accept the bonus token is NOT committed with a
        // separate T=1 trunk pass (a full weight read per round). It stays PENDING and rides as
        // column 0 of the NEXT round's verify batch. Under predecessor pairing the next chain
        // seeds from the bonus's predecessor's TRUE verify hidden (free — no extra
        // pass of any kind). Verify still
        // checks every emitted token against the target -> exactness holds by construction; only
        // DRAFT QUALITY can shift, which the acceptance numbers arbitrate.
        let mut pending: Option<u32> = None;   // bonus emitted but not yet committed to cache
        while out.len() < max_new {
            let pos = cache.pos;            // #tokens committed (EXCLUDES a pending bonus)
            cache.snapshot_into(e, &mut snap)?;  // §C: snapshot BEFORE draft+verify

            // --- 1. DRAFT k tokens with the NextN head (autoregressive, T=1 each) ---
            // p-min semantics (both paths): stop the chain early when the head's confidence in
            // its own pick drops below p_min — the just-drafted token is DISCARDED, but its
            // scratch append stands (identical to the eager chain's ordering). j==0 always drafts.
            let base0 = if pending.is_some() { 1usize } else { 0usize };
            // Round-start draft-KV sync (BOTH paths). Persistent: truncate/align to the committed
            // history — slots 0..P hold entries for the tokens before last_token@P (P = pos +
            // base0 - 1); this single set_len IS the draft-side rollback (drops last round's
            // rejected drafts and p-min extras via the len mechanism).
            scratch.set_len(e, pos + base0 - 1)?;
            if pen_on {
                let w0 = pen_hist.len().saturating_sub(sp.penalty_last_n);
                pen_hist_d = Some(e.htod_u32_v(&pen_hist[w0..])?);
            }
            let k_this = k;   // fixed draft length (adaptive-K removed 2026-07-08, honest loss)
            let mut draft: Vec<u32> = Vec::with_capacity(k);
            let mut draft_idx: Vec<u32> = Vec::with_capacity(k);   // trimmed-vocab ids (== draft when untrimmed)
            if sampled { draft_logits.clear(); draft_stats.clear(); }
            if let (false, Some(gr)) = (sampled || pen_on, &draft_graph) {
                // GRAPH DRAFT: one dispatch per drafted token. The chain feeds itself on-device
                // (in-graph argmax -> tok_d -> next replay's embed; h_nextn -> h_seed_d; pos_d
                // inc'd in-graph); the host only reads 4B token (+4B p) and decides the break.
                e.set_i32_one(&mut g_pos, (pos + base0) as i32)?;
                e.set_u32_one(&mut g_tok, last_token)?;
                e.copy_into(&mut g_seed, 0, &h_seed_buf, n_embd)?;
                for j in 0..k_this {
                    gr.launch()?;
                    scratch.kv.len += 1;   // host mirror (len_d advanced in-graph)
                    let idx = e.dtoh_u32_one(&g_tok)?;
                    // trimmed draft vocab -> target token id (identity when no d2t map)
                    let d = match &mtp.d2t { Some(map) => map[idx as usize], None => idx };
                    if p_min > 0.0 {
                        let p = e.dtoh(&g_p)?[0];
                        if p < p_min && (j > 0 || (pmin0 && base0 == 1)) { break; }
                    }
                    draft.push(d);
                    // with a trimmed head the NEXT embed must read the TARGET id, not the draft
                    // index the argmax wrote — patch the persistent token buffer (4B htod).
                    if d != idx { e.set_u32_one(&mut g_tok, d)?; }
                }
            } else if let (true, Some(gr)) = (sampled, &draft_graph_s) {
                // SAMPLED GRAPH DRAFT: one replay per drafted token — head forward + gumbel +
                // argmax in ONE dispatch; the host reads 4B token (+4B p), D2Ds q into slot j,
                // and decides the break. Event-counter continuity: g_ctr is host-seeded to
                // sctr-1 ONCE per round (outside the graph); the in-graph bump runs BEFORE the
                // perturb, so replay j consumes counter sctr+j — exactly the eager arm's Philox
                // stream. Host sctr advances in lockstep (computed, no readback needed).
                e.set_i32_one(&mut g_pos, (pos + base0) as i32)?;
                e.set_u32_one(&mut g_tok, last_token)?;
                e.copy_into(&mut g_seed, 0, &h_seed_buf, n_embd)?;
                e.set_u32_one(&mut g_ctr, sctr.wrapping_sub(1))?;
                for j in 0..k_this {
                    gr.launch()?;
                    scratch.kv.len += 1;   // host mirror (len_d advanced in-graph)
                    sctr += 1;             // mirrors the in-graph g_ctr bump (eager parity:
                                           // counts the p-min-discarded token too)
                    // q retention: ONE async D2D of the persistent head-logits buffer into this
                    // round's slot j (stream-ordered after the replay, before the next one).
                    e.copy_into(&mut q_slots[j], 0, &g_q, d_vocab)?;
                    let idx = e.dtoh_u32_one(&g_tok)?;
                    let d = match &mtp.d2t { Some(map) => map[idx as usize], None => idx };
                    draft_idx.push(idx);
                    if p_min > 0.0 {
                        let p = e.dtoh(&g_p)?[0];
                        if p < p_min && (j > 0 || (pmin0 && base0 == 1)) { break; }
                    }
                    draft.push(d);
                    // trimmed head: the NEXT embed must read the TARGET id (see the greedy arm).
                    if d != idx { e.set_u32_one(&mut g_tok, d)?; }
                }
                // uniform accept path: fill draft_stats per used slot (pure-temp regime — the
                // stats degenerate to th=0 / full-Z; one filter_stats launch per slot, tiny).
                for j in 0..draft.len().max(draft_idx.len()) {
                    let rows0 = e.htod_i32(&[0])?;
                    let (mut th_d, mut z_d, mut mx_d) = (e.zeros(1)?, e.zeros(1)?, e.zeros(1)?);
                    e.filter_stats(&q_slots[j], d_vocab, &rows0, &mut th_d, &mut z_d, &mut mx_d,
                                   d_vocab, 1, sp_temp, sp.top_k, sp.top_p, sp.min_p)?;
                    draft_stats.push((e.dtoh(&mx_d)?[0], e.dtoh(&th_d)?[0], e.dtoh(&z_d)?[0]));
                }
            } else {
                // EAGER DRAFT (fallback: MoE head/trunk, huge k, BW24_SPEC_NOGRAPH, capture fail).
                let mut e_tok = last_token;
                let mut d_seed = e.clone_dtod(&h_seed_buf)?;
                for j in 0..k_this {
                    // GPU-ARGMAX DRAFT (2026-07-03): device logits + device argmax + 4-byte token
                    // read instead of the ~600KB full-vocab dtoh + host argmax per draft token.
                    let mtp_pos = pos + base0 + j;
                    let (dl_d, h_nextn) = self.mtp_head_forward_dev(e, mtp, e_tok, &d_seed, &mut *scratch, mtp_pos, embd_gpu, embd_qt, embd_rb)?;
                    let tok_d = if sampled {
                        // FILTERED Gumbel-max: stats -> masked perturb -> argmax = one draw from
                        // the filtered softmax (filters off => th=0, exact v1 semantics).
                        if perturb_buf.is_none() { perturb_buf = Some(e.zeros(d_vocab.max(n_vocab))?); }
                        let mut q_row = e.clone_dtod(&dl_d)?;      // retained q (penalized when on)
                        if pen_on {
                            let h = pen_hist_d.as_ref().unwrap();
                            let nh = h.len();
                            e.penalize_logits(&mut q_row, h, nh, sp.penalty_repeat, sp.penalty_freq,
                                              sp.penalty_present, d_vocab)?;
                        }
                        let rows0 = e.htod_i32(&[0])?;
                        let (mut th_d, mut z_d, mut mx_d) = (e.zeros(1)?, e.zeros(1)?, e.zeros(1)?);
                        e.filter_stats(&q_row, d_vocab, &rows0, &mut th_d, &mut z_d, &mut mx_d,
                                       d_vocab, 1, sp_temp, sp.top_k, sp.top_p, sp.min_p)?;
                        let (th, z, mx) = (e.dtoh(&th_d)?[0], e.dtoh(&z_d)?[0], e.dtoh(&mx_d)?[0]);
                        let pb = perturb_buf.as_mut().unwrap();
                        e.gumbel_perturb_filtered(&q_row, pb, d_vocab, sp_seed, sctr, sp_temp, mx, th)?;
                        sctr += 1;
                        draft_logits.push(q_row);
                        draft_stats.push((mx, th, z));
                        e.argmax_token_device(pb, d_vocab)?
                    } else {
                        e.argmax_token_device(&dl_d, d_vocab)?
                    };
                    let idx = e.dtoh_u32_one(&tok_d)?;
                    let d = match &mtp.d2t { Some(map) => map[idx as usize], None => idx };
                    if sampled { draft_idx.push(idx); }
                    if p_min > 0.0 {
                        let p_d = e.prob_of_token_device(&dl_d, &tok_d, d_vocab)?;
                        let p = e.dtoh(&p_d)?[0];
                        if p < p_min && (j > 0 || (pmin0 && base0 == 1)) { break; }
                    }
                    draft.push(d);
                    e_tok = d;
                    d_seed = h_nextn;
                }
            }
            let k_round = draft.len();

            // --- 2. VERIFY: one batched target forward. With a pending bonus, it rides as col 0
            //         (committing its KV/recur inside the SAME weight read); drafts follow. ---
            let verify_tokens: Vec<u32> = match pending {
                Some(b) => { let mut v = Vec::with_capacity(k_round + 1); v.push(b); v.extend_from_slice(&draft); v }
                None => draft.clone(),
            };
            let base = if pending.is_some() { 1 } else { 0 };
            // ckpt (REPLAY-FREE partial accept): retain per-layer state-rebuild inputs alongside
            // the verify. Pure buffer keep-alives + dtod clones — kernel work is unchanged.
            let mut ckpt = if spec_replay { None } else { Some(VerifyCkpt::new(self.layers.len())) };
            let (tlogits_d, vx) = self.decode_step_t_core(e, &verify_tokens, pos, &mut *cache,
                                                          Some((embd_gpu, embd_qt, embd_rb)),
                                                          ckpt.as_mut())?;

            // --- 3. GREEDY ACCEPT (walk prefix, stop at first mismatch) ---
            // DEVICE-ARGMAX ACCEPT: argmax every verify column ON DEVICE (same 2-pass kernels +
            // smallest-index tie-break as host argmax, argmax_gate-validated) and read back ONE
            // [T] u32 — replaces the T x n_vocab f32 dtoh + T host argmaxes per round.
            // t_pred[j] = target's greedy prediction for the slot after draft[j-1] (j>=1) or after
            // last_token (j==0). With a pending bonus, col 0 IS the prediction after last_token
            // (== the bonus), so every index shifts by `base` and last_pred is unused.
            let t_v = verify_tokens.len();
            let mut preds: Vec<u32> = Vec::new();
            if !sampled {
                for j in 0..t_v {
                    e.argmax_token_device_col(&tlogits_d, j, n_vocab, &mut preds_d, j)?;
                }
                preds = e.dtoh_u32(&preds_d)?;
            }
            let t_pred = |j: usize| -> u32 {
                if j == 0 && base == 0 { last_pred } else { preds[base + j - 1] }
            };
            let (n_acc, bonus) = if !sampled {
                let mut n_acc = 0usize;
                for j in 0..k_round {
                    if t_pred(j) == draft[j] { n_acc += 1; } else { break; }
                }
                // bonus = target's own token at the first non-accepted slot. n_acc in 0..=k; t_pred
                // is defined for j in 0..=k (j==0 -> last_logits, j>=1 -> col j-1, last col = k-1).
                (n_acc, t_pred(n_acc))
            } else {
                // --- SAMPLED ACCEPT (rejection sampling): u_j < p_j(x_j)/q_j(x_j) walk ---
                if col_buf.is_none() { col_buf = Some(e.zeros(n_vocab)?); }
                // FILTERED p_j: per-verify-col stats (one batched filter_stats call), then the
                // filtered gather. j==0&&base==0 reads last_col (its own stats row appended).
                let mut pj = vec![0f32; k_round.max(1)];
                let mut col_stats: Vec<(f32, f32, f32)> = Vec::new();  // (max, th, z) per verify col used
                if k_round > 0 {
                    let mut ids: Vec<u32> = Vec::new();
                    let mut rows: Vec<i32> = Vec::new();
                    for j in 0..k_round {
                        if j > 0 || base == 1 { ids.push(draft[j]); rows.push((base + j) as i32 - 1); }
                    }
                    if !ids.is_empty() {
                        let nr = rows.len();
                        // penalties: materialize the used columns into one contiguous penalized
                        // buffer (rows remapped 0..nr) so stats+gathers see the penalized p.
                        // penalties: materialize used columns contiguously, penalize all rows in
                        // one launch, and point stats+gathers at the penalized buffer (rows 0..nr).
                        let p_rows: Vec<i32> = if pen_on { (0..nr as i32).collect() } else { rows.clone() };
                        if pen_on {
                            if pcol_buf.as_ref().map(|b| b.len()).unwrap_or(0) < nr * n_vocab {
                                pcol_buf = Some(e.zeros(nr * n_vocab)?);
                            }
                            let pc = pcol_buf.as_mut().unwrap();
                            for (i2, &r) in rows.iter().enumerate() {
                                let c = r as usize;
                                e.copy_view_into(pc, i2 * n_vocab,
                                                 &tlogits_d.slice(c * n_vocab..(c + 1) * n_vocab), n_vocab)?;
                            }
                            let h = pen_hist_d.as_ref().unwrap();
                            let nh = h.len();
                            e.penalize_logits_rows(pc, h, nh, sp.penalty_repeat, sp.penalty_freq,
                                                   sp.penalty_present, n_vocab, nr)?;
                        }
                        let p_src: &CudaSlice<f32> = if pen_on { pcol_buf.as_ref().unwrap() } else { &tlogits_d };
                        let rowsd = e.htod_i32(&p_rows)?;
                        let (mut th_d, mut z_d, mut mx_d) = (e.zeros(nr)?, e.zeros(nr)?, e.zeros(nr)?);
                        e.filter_stats(p_src, n_vocab, &rowsd, &mut th_d, &mut z_d, &mut mx_d,
                                       n_vocab, nr, sp_temp, sp.top_k, sp.top_p, sp.min_p)?;
                        let idsd = e.htod_u32_v(&ids)?;
                        let mut outd = e.zeros(nr)?;
                        e.softmax_gather_filtered(p_src, n_vocab, &idsd, &rowsd, &th_d, &z_d,
                                                  &mut outd, n_vocab, nr, sp_temp)?;
                        let outv = e.dtoh(&outd)?;
                        let (thv, zv, mxv) = (e.dtoh(&th_d)?, e.dtoh(&z_d)?, e.dtoh(&mx_d)?);
                        let mut oi = 0usize;
                        for j in 0..k_round { if j > 0 || base == 1 { pj[j] = outv[oi]; oi += 1; } }
                        col_stats = (0..nr).map(|i| (mxv[i], thv[i], zv[i])).collect();
                    }
                    if base == 0 {
                        let lc: &CudaSlice<f32> = if pen_on {
                            if col_buf.is_none() { col_buf = Some(e.zeros(n_vocab)?); }
                            let cb = col_buf.as_mut().unwrap();
                            e.copy_into(cb, 0, last_col_logits.as_ref().expect("sampled: last_col_logits unset"), n_vocab)?;
                            let h = pen_hist_d.as_ref().unwrap();
                            let nh = h.len();
                            e.penalize_logits(cb, h, nh, sp.penalty_repeat, sp.penalty_freq, sp.penalty_present, n_vocab)?;
                            col_buf.as_ref().unwrap()
                        } else {
                            last_col_logits.as_ref().expect("sampled: last_col_logits unset")
                        };
                        let rows0 = e.htod_i32(&[0])?;
                        let (mut th_d, mut z_d, mut mx_d) = (e.zeros(1)?, e.zeros(1)?, e.zeros(1)?);
                        e.filter_stats(lc, n_vocab, &rows0, &mut th_d, &mut z_d, &mut mx_d,
                                       n_vocab, 1, sp_temp, sp.top_k, sp.top_p, sp.min_p)?;
                        let idsd = e.htod_u32_v(&[draft[0]])?;
                        let mut outd = e.zeros(1)?;
                        e.softmax_gather_filtered(lc, n_vocab, &idsd, &rows0, &th_d, &z_d,
                                                  &mut outd, n_vocab, 1, sp_temp)?;
                        pj[0] = e.dtoh(&outd)?[0];
                        last_col_stats = Some((e.dtoh(&mx_d)?[0], e.dtoh(&th_d)?[0], e.dtoh(&z_d)?[0]));
                    }
                }
                // q source: the graph arm retained the head logits in the persistent q_slots;
                // the eager arm in per-round draft_logits clones. Same raw-logit values either way.
                // FILTERED q_j: stats from draft_stats (eager pushes in-chain; the graph arm
                // computes them post-replay — graph engages only filter/penalty-free, so the
                // stats degenerate to th=0/full-Z there, keeping ONE accept path).
                let q_bufs: &[CudaSlice<f32>] =
                    if draft_graph_s.is_some() { &q_slots } else { &draft_logits };
                let mut n_acc = 0usize;
                for j in 0..k_round {
                    let (qmx, qth, qz) = draft_stats[j];
                    let idsd = e.htod_u32_v(&[draft_idx[j]])?;
                    let rowsd = e.htod_i32(&[0])?;
                    let thd = e.htod(&[qth])?; let zd = e.htod(&[qz])?;
                    let _ = qmx;
                    let mut outd = e.zeros(1)?;
                    e.softmax_gather_filtered(&q_bufs[j], d_vocab, &idsd, &rowsd, &thd, &zd,
                                              &mut outd, d_vocab, 1, sp_temp)?;
                    let qj = e.dtoh(&outd)?[0];
                    let u = host_u01(sp_seed, uctr); uctr += 1;
                    if (u as f64) * (qj as f64) < pj[j] as f64 { n_acc += 1; } else { break; }
                }
                let bonus = if n_acc == k_round {
                    // FULL ACCEPT: bonus ~ FILTERED softmax at the last verify column.
                    let col = base + k_round - 1;
                    let cb = col_buf.as_mut().unwrap();
                    e.copy_view_into(cb, 0, &tlogits_d.slice(col * n_vocab..(col + 1) * n_vocab), n_vocab)?;
                    if pen_on {
                        let h = pen_hist_d.as_ref().unwrap();
                        let nh = h.len();
                        e.penalize_logits(cb, h, nh, sp.penalty_repeat, sp.penalty_freq, sp.penalty_present, n_vocab)?;
                    }
                    if perturb_buf.is_none() { perturb_buf = Some(e.zeros(d_vocab.max(n_vocab))?); }
                    // stats for the last used col: reuse col_stats when it covers it, else compute.
                    let (mx, th, _z) = if !col_stats.is_empty() {
                        *col_stats.last().unwrap()
                    } else {
                        let rows0 = e.htod_i32(&[0])?;
                        let (mut th_d, mut z_d, mut mx_d) = (e.zeros(1)?, e.zeros(1)?, e.zeros(1)?);
                        let cb0 = col_buf.as_ref().unwrap();
                        e.filter_stats(cb0, n_vocab, &rows0, &mut th_d, &mut z_d, &mut mx_d,
                                       n_vocab, 1, sp_temp, sp.top_k, sp.top_p, sp.min_p)?;
                        (e.dtoh(&mx_d)?[0], e.dtoh(&th_d)?[0], e.dtoh(&z_d)?[0])
                    };
                    let pb = perturb_buf.as_mut().unwrap();
                    let cb2 = col_buf.as_ref().unwrap();
                    e.gumbel_perturb_filtered(cb2, pb, n_vocab, sp_seed, sctr, sp_temp, mx, th)?;
                    sctr += 1;
                    let td = e.argmax_token_device(pb, n_vocab)?;
                    e.dtoh_u32_one(&td)?
                } else {
                    // REJECT at n_acc: bonus ~ norm(max(0, softmax_T(p) - softmax_T(q))).
                    let cb = col_buf.as_mut().unwrap();
                    if n_acc > 0 || base == 1 {
                        let col = base + n_acc - 1;
                        e.copy_view_into(cb, 0, &tlogits_d.slice(col * n_vocab..(col + 1) * n_vocab), n_vocab)?;
                    } else {
                        let lc = last_col_logits.as_ref().unwrap();
                        e.copy_into(cb, 0, lc, n_vocab)?;
                    }
                    if pen_on {
                        let h = pen_hist_d.as_ref().unwrap();
                        let nh = h.len();
                        e.penalize_logits(cb, h, nh, sp.penalty_repeat, sp.penalty_freq, sp.penalty_present, n_vocab)?;
                    }
                    let cb2 = col_buf.as_ref().unwrap();
                    let sc = sctr; sctr += 1;
                    // p-stats for the reject column: from col_stats when the col was gathered,
                    // else (j==0&&base==0) from last_col_stats.
                    let p_stats = if n_acc > 0 || base == 1 {
                        // col index within the gathered set == number of gathered cols before n_acc
                        let gi = if base == 1 { n_acc } else { n_acc - 1 };
                        col_stats.get(gi).copied().unwrap_or_else(|| {
                            (0.0, 0.0, 1.0) // unreachable: gathered cols always cover the reject slot
                        })
                    } else {
                        last_col_stats.expect("sampled: last_col_stats unset at reject")
                    };
                    let q_stats = draft_stats[n_acc];
                    if let Some(map) = &d2t_dev {
                        if q_full_buf.is_none() { q_full_buf = Some(e.zeros(n_vocab)?); }
                        let qf = q_full_buf.as_mut().unwrap();
                        e.scatter_trim_logits(&q_bufs[n_acc], map, qf, d_vocab, n_vocab)?;
                        let qf2 = q_full_buf.as_ref().unwrap();
                        e.residual_sample_filtered(cb2, Some(qf2), n_vocab, sp_temp, sp_seed, sc, p_stats, q_stats, &mut sample_tok)?;
                    } else {
                        e.residual_sample_filtered(cb2, Some(&q_bufs[n_acc]), n_vocab, sp_temp, sp_seed, sc, p_stats, q_stats, &mut sample_tok)?;
                    }
                    e.dtoh_u32(&sample_tok)?[0]
                };
                (n_acc, bonus)
            };
            total_drafted += k_round;
            total_accepted += n_acc;
            if spec_stats {
                st_len_hist[k_round] += 1;
                for j in 0..k_round { st_drafted[j] += 1; }
                for j in 0..n_acc { st_accepted[j] += 1; }
                if n_acc == k_round { st_full += 1; }
            }

            if debug_spec {
                eprintln!("[R{round}] pos={pos} out_len={} last_tok={last_token} draft={draft:?} n_acc={n_acc} bonus={bonus} t_pred0={}", out.len(), t_pred(0));
            }

            // --- 4. COMMIT: draft[0..n_acc] then bonus (n_acc + 1 tokens) ---
            // SESSION MODE: every accepted column is already in the CACHE — `out` must carry all
            // of them (overshoot past max_new included) or `committed` under-counts the cache rows
            // and the next turn's continuation seeds one token off (gate-caught 2026-07-05). The
            // single-shot path keeps the cap (its caller truncates + drops the cache anyway).
            for j in 0..n_acc {
                if !session_mode && out.len() >= max_new { break; }
                out.push(draft[j]);
            }
            if pen_on {
                pen_hist.extend_from_slice(&draft[0..n_acc]);
                pen_hist.push(bonus);
            }
            let bonus_emitted = session_mode || out.len() < max_new;
            if bonus_emitted { out.push(bonus); }
            last_token = bonus;

            // --- 5. ROLLBACK + advance (§C) ---
            if n_acc == k_round {
                // FULL ACCEPT, BONUS FOLD: all verify columns (pending? + drafts) are committed in
                // cache; the NEW bonus stays PENDING for the next round's verify batch — NO extra
                // T=1 trunk pass. The next draft chain seeds from the MTP block's h_nextn at the
                // bonus position: one MTP-block pass (~1/33 trunk cost) replaces the trunk read.
                // last_pred is dead in the pending path (t_pred reads verify col 0).
                //
                // PERSISTENT DRAFT KV, full-accept fill: the chain covered last_token +
                // draft[0..k_round-2] as INPUTS (slots P..P'-2); draft[k_round-1] (slot P'-1) was
                // only ever an output, so its entry is MISSING. Fill it from vh_seed — its EXACT
                // trunk hidden (the last verify column). set_len first: a p-min break may have
                // left one extra chain append at that slot. Partial accepts need NO fill (the
                // chain already covered every accepted position; round-start set_len truncates).
                let mut vh_seed = e.zeros(n_embd)?;
                e.copy_view_into(&mut vh_seed, 0, &vx.slice((t_v - 1) * n_embd..t_v * n_embd), n_embd)?;
                if refresh {
                    // TRUE-HIDDEN REFRESH (2026-07-03, the HANDOVER-listed acceptance lever):
                    // overwrite ALL committed positions' scratch entries with K/V from their EXACT
                    // verify hiddens — the reference engine's mtp_update fills from true hiddens;
                    // the full stack (vx) is already resident from the verify. Replaces both the
                    // chain-approximate entries AND the old last-token-only fill. Acceptance-only
                    // (draft attention quality); exactness stays the verify's job.
                    scratch.set_len(e, pos)?;
                    // PREDECESSOR pairing: row i gets vx[i-1]; row 0 the carried fill_prev
                    // (hidden of the last committed row before this verify batch).
                    let mut vxs = e.zeros(t_v * n_embd)?;
                    e.copy_into(&mut vxs, 0, &fill_prev, n_embd)?;
                    if t_v > 1 {
                        e.copy_view_into(&mut vxs, n_embd, &vx.slice(0..(t_v - 1) * n_embd),
                                         (t_v - 1) * n_embd)?;
                    }
                    self.mtp_kv_fill(e, mtp, &verify_tokens, &vxs, pos, &mut *scratch,
                                     embd_gpu, embd_qt, embd_rb)?;
                } else {
                    scratch.set_len(e, pos + base + k_round - 1)?;
                    // predecessor of the last draft = verify col t_v-2 (or fill_prev at t_v==1)
                    let mut hp = e.zeros(n_embd)?;
                    if t_v >= 2 {
                        e.copy_view_into(&mut hp, 0,
                                         &vx.slice((t_v - 2) * n_embd..(t_v - 1) * n_embd),
                                         n_embd)?;
                    } else {
                        e.copy_into(&mut hp, 0, &fill_prev, n_embd)?;
                    }
                    self.mtp_kv_fill(e, mtp, &[draft[k_round - 1]], &hp,
                                     pos + base + k_round - 1, &mut *scratch,
                                     embd_gpu, embd_qt, embd_rb)?;
                }
                // REFERENCE SEEDING: no pseudo pass — the next chain's step 0 IS the
                // reference's (id_last, h_prev) draft row; it appends the bonus's scratch
                // entry itself. Seed = TRUE hidden of the bonus's predecessor (last verify
                // col). Saves one MTP-block pass per round on top of the pairing fix.
                e.copy_into(&mut h_seed_buf, 0, &vh_seed, n_embd)?;
                e.copy_into(&mut fill_prev, 0, &vh_seed, n_embd)?;
                pending = Some(bonus);
                if debug_spec { eprintln!("  -> FULL ACCEPT (bonus pending, prev-h seed)"); }
            } else if !spec_replay && base + n_acc >= 1 {
                // PARTIAL ACCEPT, REPLAY-FREE (2026-07-03 — the profiled #1 long-ctx spec cost):
                // the verify's first j = base+n_acc columns ARE the committed sequence, computed
                // bit-identically to eager (decode-exact contract) — so KEEP them: KV truncates to
                // pos+j, recurrent state rebuilds from the VerifyCkpt (same-kernel gdn prefix
                // re-run / pure state-clone restore), and the bonus stays PENDING exactly like the
                // full-accept path — the legacy duplicate trunk replay is gone. The next chain
                // seeds from the MTP pseudo-hidden of the bonus, whose seed = the TRUE verify
                // hidden of its predecessor (col j-1) — same one-hop pseudo structure as full
                // accept (never compounds: the next verify recomputes true hiddens for all
                // committed columns).
                let j = base + n_acc;
                self.commit_verified_prefix(e, &mut *cache, &snap, ckpt.as_ref().unwrap(), j)?;
                let mut seed = e.zeros(n_embd)?;
                e.copy_view_into(&mut seed, 0, &vx.slice((j - 1) * n_embd..j * n_embd), n_embd)?;
                // Draft scratch: TRUE-HIDDEN REFRESH of the committed prefix (see the full-accept
                // branch); without it the chain entries stand and only the tail truncates. Either
                // way len ends at pos+j so the pseudo append lands at the bonus's slot pos+j
                // (persistent mode), rope pos+j+1 (chain convention).
                if refresh {
                    scratch.set_len(e, pos)?;
                    let mut vxs = e.zeros(j * n_embd)?;
                    e.copy_into(&mut vxs, 0, &fill_prev, n_embd)?;
                    if j > 1 {
                        e.copy_view_into(&mut vxs, n_embd, &vx.slice(0..(j - 1) * n_embd),
                                         (j - 1) * n_embd)?;
                    }
                    self.mtp_kv_fill(e, mtp, &verify_tokens[0..j], &vxs, pos, &mut *scratch,
                                     embd_gpu, embd_qt, embd_rb)?;
                } else {
                    scratch.set_len(e, pos + j)?;
                }
                // REFERENCE SEEDING (see the full-accept branch): seed = TRUE hidden of the
                // bonus's predecessor (verify col j-1); no pseudo pass.
                e.copy_into(&mut h_seed_buf, 0, &seed, n_embd)?;
                e.copy_into(&mut fill_prev, 0, &seed, n_embd)?;
                pending = Some(bonus);
                if debug_spec { eprintln!("  -> PARTIAL(replay-free j={j}, bonus pending, prev-h seed)"); }
            } else if !spec_replay {
                // ZERO ROUND FOLD (2026-07-10, verify-cost target #3): base+n_acc == 0 — a
                // pending-less round where nothing was accepted (PMIN0 zero-draft chains after a
                // replay/commit, or plain 0-accept rounds at round 0). The old path replayed
                // [bonus] through a FULL m=1 trunk+head forward (the 489us full-vocab head pass
                // measured at ~0.75/round on PMIN0 configs). Instead: restore the pre-round
                // snapshot and let the bonus ride the NEXT round's verify as col 0 — the existing
                // base=1 pending machinery, bit-identical by the decode-exact verify contract.
                // Seed: the bonus's predecessor is the last COMMITTED token, whose hidden
                // fill_prev already carries (same seeding as the 1-token-replay case it replaces).
                cache.rollback(e, &snap, 0)?;
                scratch.set_len(e, pos)?;
                e.copy_into(&mut h_seed_buf, 0, &fill_prev, n_embd)?;
                pending = Some(bonus);
                if debug_spec { eprintln!("  -> ZERO-ROUND FOLD (bonus pending, fill_prev seed)"); }
            } else {
                // PARTIAL ACCEPT, LEGACY REPLAY (seam BW24_SPEC_REPLAY=1 — or j==0: nothing of
                // this round survives, only possible before the first pending exists, ~round 0):
                // restore EVERYTHING to the pre-round snapshot (KV truncate to pos + recur
                // restore), then replay the committed prefix pending? ++ draft[0..n_acc] ++
                // [bonus] as ONE batched T forward — single weight read, bit-identical to greedy
                // (the verify-all-columns path is the same math). Commits the bonus with a TRUE
                // trunk hidden.
                cache.rollback(e, &snap, 0)?;   // accept_len=0: KV len = pos, recur = snapshot
                let mut replay: Vec<u32> = Vec::with_capacity(base + n_acc + 1);
                if let Some(b) = pending.take() { replay.push(b); }
                replay.extend_from_slice(&draft[0..n_acc]);
                replay.push(bonus);
                // Full-stack forward (decode_step_t_core = decode_step_t_h_emb_dev's body):
                // Predecessor pairing seeds from the PREDECESSOR row (col len-2) — the same-row path takes the
                // last col exactly as before (byte-identical to the old _h_emb_dev call).
                let (rl_d, rx) = self.decode_step_t_core(e, &replay, pos, &mut *cache,
                                                         Some((embd_gpu, embd_qt, embd_rb)), None)?;
                // last_pred = argmax of the LAST column's logits (predicts the token after `bonus`)
                // — device argmax + one 4-byte read instead of the full-vocab column dtoh.
                e.argmax_token_device_col(&rl_d, replay.len() - 1, n_vocab, &mut preds_d, 0)?;
                last_pred = e.dtoh_u32(&preds_d)?[0];
                if sampled {
                    let lr0 = replay.len();
                    let lc = last_col_logits.as_mut().expect("sampled: last_col_logits unset");
                    e.copy_view_into(lc, 0, &rl_d.slice((lr0 - 1) * n_vocab..lr0 * n_vocab), n_vocab)?;
                }
                let lr = replay.len();
                if lr >= 2 {
                    e.copy_view_into(&mut h_seed_buf, 0,
                                     &rx.slice((lr - 2) * n_embd..(lr - 1) * n_embd), n_embd)?;
                } else {
                    // 1-token replay (round-0 miss): the bonus's predecessor is the OLD
                    // last_token, whose own-row hidden fill_prev still holds.
                    e.copy_into(&mut h_seed_buf, 0, &fill_prev, n_embd)?;
                }
                // the bonus is COMMITTED here — it becomes the last committed row.
                let mut rh_last = e.zeros(n_embd)?;
                e.copy_view_into(&mut rh_last, 0, &rx.slice((lr - 1) * n_embd..lr * n_embd),
                                 n_embd)?;
                e.copy_into(&mut fill_prev, 0, &rh_last, n_embd)?;
                if debug_spec { eprintln!("  -> PARTIAL(replay={replay:?}), next_pred={last_pred}"); }
            }
            round += 1;
        }

        if spec_stats {
            let per_slot: Vec<String> = (0..k).map(|j| if st_drafted[j] > 0 {
                format!("{}/{}={:.3}", st_accepted[j], st_drafted[j],
                        st_accepted[j] as f64 / st_drafted[j] as f64)
            } else { "0/0".into() }).collect();
            let acc = if total_drafted > 0 {
                total_accepted as f64 / total_drafted as f64 } else { 0.0 };
            eprintln!("[spec-stats] rounds={round} full_accept={st_full} len_hist={st_len_hist:?} \
                       per_slot=[{}] total={total_accepted}/{total_drafted}={acc:.3} \
                       tok_per_round={:.3}",
                      per_slot.join(" "),
                      (total_accepted + round) as f64 / round.max(1) as f64);
        }
        // SESSION TAIL: leave the session in the exact invariant the next turn's suffix prime
        // expects — every row in `committed` has trunk KV/recur state AND an exact draft-KV row.
        if let Some((committed, last_h, next_pred_slot, sctr_slot, uctr_slot)) = sess_tail.take() {
            *sctr_slot = sctr;
            *uctr_slot = uctr;
            *next_pred_slot = Some(last_pred);
            if let Some(b) = pending.take() {
                // pending bonus: in `out` but NOT in the caches — commit it (one T=1 pass) and
                // fill its draft-KV row (pairing: its row carries the predecessor's hidden,
                // which is exactly the carried fill_prev).
                let pos_b = cache.pos;
                scratch.set_len(e, pos_b)?;
                let (lg_b, hb) = self.decode_step_h(e, b, &mut *cache)?;
                // after a FULL-accept exit `last_pred` is STALE (it predicted the bonus itself —
                // the prediction AFTER the bonus never materialized; it would have been the next
                // round's verify col 0). The bonus commit's own logits ARE that prediction.
                *next_pred_slot = Some(argmax(&lg_b) as u32);
                self.mtp_kv_fill(e, mtp, &[b], &fill_prev, pos_b, &mut *scratch,
                                 embd_gpu, embd_qt, embd_rb)?;
                *last_h = Some(hb);
            } else {
                // fill_prev tracks the hidden of the last COMMITTED row throughout the loop.
                *last_h = Some(e.clone_dtod(&fill_prev)?);
            }
            committed.extend_from_slice(prompt);
            committed.extend_from_slice(&out);   // FULL out incl. overshoot — it's all committed
            debug_assert_eq!(cache.pos, committed.len(),
                "session invariant: cache rows == committed tokens");
            return Ok((out, total_drafted, total_accepted));
        }
        out.truncate(max_new);
        Ok((out, total_drafted, total_accepted))
    }

    /// TEACHER-FORCED REPLAY ACCEPTANCE (hqmtp MTP-heal protocol): walk a FIXED token
    /// sequence and, at sampled positions, compare the MTP head's K-token draft chain against
    /// the trunk's own teacher-forced greedy predictions. Nothing is generated — the context is
    /// the corpus text itself, so (a) degenerate self-generated loops cannot inflate acceptance
    /// and (b) two arms (bf16 ceiling vs NVFP4) score on IDENTICAL contexts, isolating the
    /// quant-induced head/hidden-state mismatch from text drift.
    ///
    /// Per eval position p (context = tokens[0..=p], predecessor pairing as in spec decode):
    ///   draft_j  = chain token j from (tokens[p], h_{p-1}), then its own drafts — the exact
    ///              eager spec-decode chain (same mtp_head_forward_dev, same rope positions).
    ///   target_j = teacher-forced greedy pick for position p+1+j (argmax of the trunk logits
    ///              at forced context tokens[0..p+j]). For j==0 this equals live spec
    ///              acceptance; for j>=1 live verify would condition on the drafts, here it
    ///              conditions on the corpus — deterministic and arm-comparable by design.
    ///
    /// Returns (rows, bg): one (p, drafts[k], targets[k]) row per eval position (ascending p),
    /// plus the full teacher-forced greedy track bg (bg[i] = greedy pick for position i, i>=1)
    /// so harnesses can cross-check runs (e.g. different chunk sizes must give identical bg).
    ///
    /// `hdump`: when Some, every position's pre-output_norm trunk hidden (the exact rows the
    /// draft-KV fill pairs from) streams to the file as little-endian f32 [t_total, n_embd] —
    /// the head-distillation extraction (hqmtp): the ENGINE is the source of truth for trunk
    /// hiddens (HF torch reproductions of the hybrid trunk measured only ~0.5 greedy
    /// agreement vs this path — not usable as a training-data source).
    pub fn replay_acceptance(&self, e: &Engine, tokens: &[u32], k: usize, stride: usize,
                             chunk: usize, mut hdump: Option<&mut std::fs::File>)
        -> Result<(Vec<(usize, Vec<u32>, Vec<u32>)>, Vec<u32>), Box<dyn std::error::Error>> {
        assert!(k >= 1 && stride >= 1 && chunk >= 2);
        let mtp = self.mtp.as_ref().expect("replay_acceptance requires an MTP head");
        let n_vocab = self.output.out_features();
        let d_vocab = mtp.shared_head_head.as_ref().unwrap_or(&self.output).out_features();
        let n_embd = self.cfg.n_embd as usize;
        let t_total = tokens.len();
        assert!(t_total >= 8, "corpus too short ({t_total} tokens)");
        let mut cache = Cache::new(e, &self.cfg, t_total + k + 8)?;
        let mut scratch = MtpScratch::new(e, &self.cfg, t_total + k + 8)?;
        let embd_gpu = self.embd_gpu.get_or_init(|| {
            e.upload_u8(&self.embd.raw).expect("embed table upload")
        });
        let (embd_qt, embd_rb) = self.embd.qt_and_row_bytes(n_embd);

        // bg[i] = the trunk's greedy pick for position i under the forced context (i >= 1).
        let mut bg: Vec<u32> = vec![0; t_total + 1];
        let mut rows: Vec<(usize, Vec<u32>, Vec<u32>)> = Vec::new();
        let mut prev_last_h = e.zeros(n_embd)?;   // predecessor hidden entering the chunk
        let mut seed_buf = e.zeros(n_embd)?;
        let mut preds_d = e.alloc_u32_zeroed(chunk)?;
        let nll_on = std::env::var("BW24_REPLAY_NLL").as_deref() == Ok("1");
        let (mut nll_sum, mut nll_cnt) = (0f64, 0u64);
        let mut s = 0usize;
        while s < t_total {
            let cend = (s + chunk).min(t_total);
            let tc = cend - s;
            let ch = &tokens[s..cend];
            // 1. forced trunk pass — verify path (decode-exact contract): all-column logits +
            //    the chunk's true hiddens.
            let (tl_d, vx) = self.decode_step_t_core(e, ch, s, &mut cache,
                                                     Some((embd_gpu, embd_qt, embd_rb)), None)?;
            for j in 0..tc { e.argmax_token_device_col(&tl_d, j, n_vocab, &mut preds_d, j)?; }
            let preds = e.dtoh_u32(&preds_d)?;
            for j in 0..tc { bg[s + j + 1] = preds[j]; }
            // BW24_REPLAY_NLL=1: teacher-forced NLL/perplexity over the same forced pass — the
            // checkpoint-quality metric (position j's logits score the GOLD next token).
            if nll_on {
                let jmax = if cend < t_total { tc } else { tc - 1 };  // last pos has no gold next
                if jmax > 0 {
                    let ids: Vec<u32> = (0..jmax).map(|j| tokens[s + j + 1]).collect();
                    let rows: Vec<i32> = (0..jmax as i32).collect();
                    let idsd = e.htod_u32_v(&ids)?;
                    let rowsd = e.htod_i32(&rows)?;
                    let mut outd = e.zeros(jmax)?;
                    e.softmax_gather(&tl_d, n_vocab, &idsd, &rowsd, &mut outd, n_vocab, jmax, 1.0)?;
                    for pr in e.dtoh(&outd)? {
                        nll_sum += -((pr.max(1e-30)) as f64).ln();
                        nll_cnt += 1;
                    }
                }
            }
            if let Some(f) = hdump.as_deref_mut() {
                use std::io::Write;
                let host: Vec<f32> = e.dtoh(&vx)?;
                // bf16 round-to-nearest-even — f32 doubled the disk bill at bulk
                // extraction scale (20M tokens x 4096 = 320GB f32 vs 160GB bf16).
                let mut bytes = Vec::with_capacity(tc * n_embd * 2);
                for v in &host[..tc * n_embd] {
                    let b = v.to_bits();
                    let r = b.wrapping_add(0x7FFF + ((b >> 16) & 1));
                    bytes.extend_from_slice(&((r >> 16) as u16).to_le_bytes());
                }
                f.write_all(&bytes)?;
            }
            // CHAINLESS extraction (stride > corpus, the bulk-hdump mode): no chunk ever
            // drafts, so the draft-KV fills are pure waste — skip them (2 MTP-block passes
            // per token saved; the forced trunk pass + hdump is all the mode needs).
            let chainless = stride > t_total;
            if chainless {
                e.copy_view_into(&mut prev_last_h, 0,
                                 &vx.slice((tc - 1) * n_embd..tc * n_embd), n_embd)?;
                s = cend;
                continue;
            }
            // 2. TRUE predecessor-paired draft-KV fill for the chunk (row i carries h_{i-1};
            //    row s reads the previous chunk's last true hidden, zeros at corpus start).
            let mut vxs = e.zeros(tc * n_embd)?;
            e.copy_into(&mut vxs, 0, &prev_last_h, n_embd)?;
            if tc > 1 {
                e.copy_view_into(&mut vxs, n_embd, &vx.slice(0..(tc - 1) * n_embd),
                                 (tc - 1) * n_embd)?;
            }
            scratch.set_len(e, s)?;
            self.mtp_kv_fill(e, mtp, ch, &vxs, s, &mut scratch, embd_gpu, embd_qt, embd_rb)?;
            // 3. draft chains at sampled positions, DESCENDING: a chain reads only slots
            //    [0..p) (true fills) and appends at >= p; the next (smaller-p) chain's set_len
            //    truncates those approximate appends before they can ever be read.
            let ps: Vec<usize> = (s..cend)
                .filter(|p| *p >= 1 && *p % stride == 0 && *p + k <= t_total).collect();
            for &p in ps.iter().rev() {
                scratch.set_len(e, p)?;
                if p == s { e.copy_into(&mut seed_buf, 0, &prev_last_h, n_embd)?; }
                else {
                    e.copy_view_into(&mut seed_buf, 0,
                                     &vx.slice((p - 1 - s) * n_embd..(p - s) * n_embd), n_embd)?;
                }
                let mut e_tok = tokens[p];
                let mut d_seed = e.clone_dtod(&seed_buf)?;
                let mut drafts: Vec<u32> = Vec::with_capacity(k);
                for j in 0..k {
                    let (dl_d, h_nextn) = self.mtp_head_forward_dev(
                        e, mtp, e_tok, &d_seed, &mut scratch, p + 1 + j,
                        embd_gpu, embd_qt, embd_rb)?;
                    let tok_d = e.argmax_token_device(&dl_d, d_vocab)?;
                    let idx = e.dtoh_u32_one(&tok_d)?;
                    let d = match &mtp.d2t { Some(map) => map[idx as usize], None => idx };
                    drafts.push(d);
                    e_tok = d;
                    d_seed = h_nextn;
                }
                // targets may live in a LATER chunk's bg — resolved after the walk.
                rows.push((p, drafts, Vec::new()));
            }
            // 4. restore TRUE entries for the whole chunk (the next chunk's chains and fills
            //    expect scratch.len == cend with exact rows).
            scratch.set_len(e, s)?;
            self.mtp_kv_fill(e, mtp, ch, &vxs, s, &mut scratch, embd_gpu, embd_qt, embd_rb)?;
            e.copy_view_into(&mut prev_last_h, 0, &vx.slice((tc - 1) * n_embd..tc * n_embd),
                             n_embd)?;
            s = cend;
        }
        for (p, drafts, targets) in rows.iter_mut() {
            for j in 0..drafts.len() { targets.push(bg[*p + 1 + j]); }
        }
        rows.sort_by_key(|r| r.0);
        if nll_cnt > 0 {
            let mean = nll_sum / nll_cnt as f64;
            println!("[replay-nll] tokens={nll_cnt} nll/token={mean:.5} ppl={:.4}", mean.exp());
        }
        Ok((rows, bg))
    }
}
