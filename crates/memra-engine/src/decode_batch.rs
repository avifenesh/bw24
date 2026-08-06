//! Batched decode step — B sequences share one fused pass (ARCHITECTURE-H100.md §3 B2').
//!
//! The bandwidth thesis: decode is weight-stream-bound, so every projection at m=B rows
//! amortizes one weight read across B sequences. Row-parallel ops (norm/rope/quantize/
//! activation) batch trivially — they are the SAME kernels prefill already runs at T rows.
//! Only truly per-sequence state stays in a loop: KV append + fa_decode over each cache,
//! and the GDN/conv recurrent step (v1: per-seq loop via the existing single-seq path;
//! a blockIdx.z-batched GDN state kernel is the v2 fusion).
//!
//! EXACTNESS CONTRACT (the law this module lives under):
//! - B == 1 must be BIT-IDENTICAL to `decode_step_h` (gate: decode-batch-gate).
//! - 2 <= B <= 8: each row rides the m=2..9 verify-tier mmvq kernels, which are per-row
//!   bit-identical to m=1 (the spec-exactness machinery decode_step_t relies on). Each
//!   sequence's token stream must equal its isolated single-seq run (worker.rs contract:
//!   "byte-identical to isolated").
//! - 9 <= B <= 16 (the EXACT-16 tier, inc3 2026-08-01): admitted iff
//!   `decode_batch_exact16_ok` — every matmul rides the b16 batched-mmvq class
//!   (bit-identical per (token,row) to m=1; Q8_0 needs the q8rp mirror) under a
//!   verify_exact scope that disables the m>=16 GEMM/MMQ arms. gate2 bit-strength
//!   PASS at B=12/16 (research/batched-tick-inc3-20260801). Refused otherwise.
//! - B > 16 crosses into GEMM/dp4a-tail numeric configs with NO exact kernel class —
//!   refused (MEMRA_DECODE_BATCH_CAP stays a measurement door).
//!
//! v1 scope: the hybrid (Qwen3.5-class) non-gemma4 trunk. Fused m=1 micro-launches
//! (fused3 QKV, cross-layer add+norm+q8 chain) are NOT used — the unfused sequence is
//! bit-identical (kernel_check: add_rms_norm == add;rms_norm; _q8_1 == +quantize_q8_1)
//! and keeps the batched path simple. Batched fusions are tuning work, not correctness.

use crate::cache::Cache;
use crate::hybrid::{HybridModel, Mixer};
use crate::Engine;
use cudarc::driver::CudaSlice;

// ---- MEMRA_BATCH_PHASE=1 (diagnostics): sync-bounded per-phase accumulators for the batched
// tick. Each boundary syncs the stream, so the TOTAL inflates (launch pipelining is destroyed);
// the value is the RANKING/shares, not absolute ms. Read via `batch_phase_report()`.
pub(crate) static BATCH_PHASE: std::sync::Mutex<[f64; 12]> = std::sync::Mutex::new([0.0; 12]);
pub const BATCH_PHASE_NAMES: [&str; 12] = [
    "setup(ptrs+embed H2D)",
    "attn batched pre (norm/qkv/rope)",
    "attn per-seq: kv append",
    "attn per-seq: q/a dtod copies",
    "attn per-seq: fa_decode",
    "attn post (gate+o-proj)",
    "gdn batched projections",
    "gdn state ops (conv/prep/scan)",
    "gdn out (gated norm+proj)",
    "ffn (add/norm/gate/up/act/down)",
    "lm_head (norm+matmul)",
    "logits D2H + host split",
];
pub fn batch_phase_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_BATCH_PHASE").as_deref() == Ok("1"))
}
pub fn batch_phase_report() -> String {
    let ph = BATCH_PHASE.lock().unwrap();
    let tot: f64 = ph.iter().sum();
    let mut rows: Vec<(usize, f64)> = ph.iter().copied().enumerate().collect();
    rows.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut s = format!("[batch-phase] total {:.1} ms (sync-bounded; shares rank, not walltime)\n", tot * 1e3);
    for (i, v) in rows {
        s += &format!("  {:>6.1} ms {:>5.1}%  {}\n", v * 1e3, v / tot * 100.0, BATCH_PHASE_NAMES[i]);
    }
    s
}

impl HybridModel {
    /// Batched-decode width cap. 8 = the exactness-tier default (see the assert below);
    /// MEMRA_DECODE_BATCH_CAP overrides for tier-probe measurement, clamped to 32.
    pub fn decode_batch_cap() -> usize {
        use std::sync::OnceLock;
        static CAP: OnceLock<usize> = OnceLock::new();
        *CAP.get_or_init(|| {
            std::env::var("MEMRA_DECODE_BATCH_CAP").ok()
                .and_then(|v| v.parse().ok())
                .map(|c: usize| c.clamp(1, 32))
                .unwrap_or(8)
        })
    }

    /// EXACT-16 TIER admission (increment 3a, 2026-08-01, 5090 receipts
    /// research/batched-tick-inc3-20260801): true iff EVERY matmul the batched decode step
    /// runs has a per-(token,row) bit-exact kernel class at m=9..16 under the verify_exact
    /// scope — i.e. the batched-mmvq b16 family (32-thread warp reduce, the exact m=1 mmvq
    /// program per column) or the e4m3 grid.y=m mmvq catch-all. Q8_0 qualifies only with
    /// the split-plane mirror (rp4, MEMRA_Q8RP): its b16 kernel exists only as the _rp twin.
    /// Float matmuls (cuBLASLt, n-dependent reductions) and MoE FFNs disqualify the model.
    /// Measured attribution for WHY the naked m=16 tier is not exact: the m>=16 arms
    /// (MMQ int8-MMA `mul_mat_q` — MEMRA_PP_Q8MMQ default-on — and `qmatvec_gemm`, both
    /// block-scale f32) and the m=9..15 dp4a tail (128-thread two-level reduce) all break
    /// per-row bit-identity vs isolated decode (gate2 step-0 bit-diffs, maxdiff ~1.3-2.3e-1).
    pub fn decode_batch_exact16_ok(&self) -> bool {
        fn ok(w: &crate::model::GpuTensor) -> bool {
            match w {
                crate::model::GpuTensor::Quant { qtype, rp4, .. } =>
                    *qtype == crate::QT_Q4_0 || *qtype == crate::QT_Q6_K
                    || *qtype == crate::QT_F8_E4M3
                    || (*qtype == crate::QT_Q8_0 && rp4.is_some()),
                _ => false,
            }
        }
        if self.cfg.m3.is_some() || self.is_gemma4_e4b() || self.cfg.gemma4.is_some() {
            return false;
        }
        self.layers.iter().all(|l| {
            let mix_ok = match &l.mixer {
                Mixer::Full(fa) => [&fa.wq, &fa.wk, &fa.wv, &fa.wo].into_iter().all(ok),
                Mixer::Linear(la) => [&la.wqkv, &la.wqkv_gate, &la.ssm_beta,
                                      &la.ssm_alpha, &la.ssm_out].into_iter().all(ok),
                // MLA rides its own increment-4 arm; never admitted to the exact-16 tier here.
                Mixer::Mla(_) => false,
            };
            let ffn_ok = match &l.ffn {
                crate::hybrid::Ffn::Dense { ffn_gate, ffn_up, ffn_down } =>
                    [ffn_gate, ffn_up, ffn_down].into_iter().all(ok),
                crate::hybrid::Ffn::Moe(_) => false,
            };
            mix_ok && ffn_ok
        }) && ok(&self.output)
    }

    /// H3 rollback/A-B seam (serve-path phase 2): `MEMRA_SERVE_B1FAST=0` sends B=1 back
    /// through the batched body (the pre-change tick, bit-for-bit). Default ON.
    ///
    /// EXACTNESS, stated precisely (measured on-box 2026-08-05, sm_120 q9 NVFP4-MTP):
    /// the fast path is BIT-IDENTICAL TO `decode_step_h` — decode-batch-gate's STRICT
    /// gate1 (`--mode strict`) PASSes with it ON and FAILs with it OFF at maxdiff
    /// 1.591e-1. It is deliberately NOT bit-identical to the batched body: the two
    /// carry the long-accepted decode-config FP-composition gap (same class gate1's
    /// config mode tolerates), and this lever moves solo sessions onto the NAKED side
    /// of it. That is the desired direction — a c=1 serve request now computes exactly
    /// what `run-gen` computes for the same prompt. Token-stream receipts:
    /// research/servepath-p2-20260805 (greedy 150 ids + seeded-sampled identical to the
    /// run-gen oracle AND cross-arm, so the gap is sub-token here as designed).
    ///
    /// Read fresh (an `AtomicU8` memo, not a `OnceLock`): decode-batch-gate flips this
    /// seam BETWEEN gates in-process — gate1 needs the fast path ON to prove bit-identity,
    /// gate2 needs it pinned OFF to keep testing the batched body. A latch-once read would
    /// bake whichever gate ran first, so the gate could never test both sides. The memo
    /// caches the parse but `set_b1_fast` invalidates it.
    pub fn b1_fast_on() -> bool {
        // 0 = unknown/invalidated, 1 = off, 2 = on
        match Self::b1_fast_memo().load(std::sync::atomic::Ordering::Relaxed) {
            1 => false,
            2 => true,
            _ => {
                let on = std::env::var("MEMRA_SERVE_B1FAST").as_deref() != Ok("0");
                Self::b1_fast_memo()
                    .store(if on { 2 } else { 1 }, std::sync::atomic::Ordering::Relaxed);
                on
            }
        }
    }

    fn b1_fast_memo() -> &'static std::sync::atomic::AtomicU8 {
        static MEMO: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
        &MEMO
    }

    /// Test/gate seam: force the B=1 fast path on or off for the rest of the process,
    /// overriding the env. Used by decode-batch-gate to pin gate2's reference arm.
    pub fn set_b1_fast(on: bool) {
        Self::b1_fast_memo()
            .store(if on { 2 } else { 1 }, std::sync::atomic::Ordering::Relaxed);
    }

    /// H3 body: the m=1 FUSED trunk (`decode_layers_eager` — shared verbatim with
    /// `decode_step_h`/the ppN stages) plus the batched path's own serving epilogue
    /// (grammar mask, device sample, lean-logits park). See the call-site comment in
    /// `decode_step_batch_sampled_lean_masked` for why this is bit-identical.
    fn decode_step_b1_fast(
        &self,
        e: &Engine,
        token: u32,
        caches: &mut [&mut Cache],
        samp: &[Option<(f32, u64, u32)>],
        masks: &[Option<(&CudaSlice<u32>, usize)>],
        lean: bool,
    ) -> Result<(Vec<Vec<f32>>, Vec<Option<u32>>), Box<dyn std::error::Error>> {
        let n_embd = self.cfg.n_embd as usize;
        let eps = self.cfg.rms_eps;
        let pos = caches[0].pos;
        let pos_d = e.htod_i32(&[pos as i32])?;
        let x = e.htod(&self.embd.gather(n_embd, &[token]))?;
        // the SHARED m=1 trunk: same function decode_step_h runs, so every m=1 fusion
        // (cross-layer add+norm+q8_1, fused SwiGLU, lever 1's gate+up dual) fires here.
        let x = self.decode_layers_eager(e, x, 0, self.layers.len(), &pos_d, pos, caches[0])?;
        let mut hn = e.uninit(n_embd)?;
        e.rms_norm(&x, self.output_norm.float_data(), &mut hn, n_embd, 1, eps)?;
        let logits = e.matmul(&self.output, &hn, 1)?;

        // ---- epilogue: byte-for-byte the batched path's, at b_n=1 ----
        let n_vocab = self.output.out_features();
        let mut logits = logits;
        let mut pristine: Option<CudaSlice<f32>> = None;
        if let Some((mask, words)) = masks.first().copied().flatten() {
            assert!(samp.first().copied().flatten().is_some(),
                    "grammar-masked row 0 must request a device sample");
            if lean {
                let cache = &mut caches[0];
                if cache.last_logits_dev.as_ref().map(|d| d.len() < n_vocab).unwrap_or(true) {
                    cache.last_logits_dev = Some(e.uninit(n_vocab)?);
                }
                let dst = cache.last_logits_dev.as_mut().unwrap();
                e.dtod_copy_view(&logits.slice(0..n_vocab), dst)?;
            } else {
                let mut p = e.uninit(n_vocab)?;
                e.dtod_copy_view(&logits.slice(0..n_vocab), &mut p)?;
                pristine = Some(p);
            }
            e.mask_logits_col(&mut logits, mask, 0, n_vocab, words)?;
        }

        let mut next: Vec<Option<u32>> = vec![None; 1];
        if let Some((temp, seed, ctr)) = samp.first().copied().flatten() {
            let mut toks = e.alloc_u32_zeroed(1)?;
            if temp <= 0.0 {
                e.argmax_token_device_col(&logits, 0, n_vocab, &mut toks, 0)?;
            } else {
                let mut pb = e.zeros(n_vocab)?;
                e.gumbel_perturb_col(&logits, 0, &mut pb, n_vocab, seed, ctr, temp)?;
                e.argmax_token_device_col(&pb, 0, n_vocab, &mut toks, 0)?;
            }
            next[0] = Some(e.dtoh_u32(&toks)?[0]);
        }

        let sampled = samp.first().copied().flatten().is_some();
        let rows: Vec<Vec<f32>> = if lean && sampled {
            if masks.first().copied().flatten().is_none() {
                let cache = &mut caches[0];
                if cache.last_logits_dev.as_ref().map(|d| d.len() < n_vocab).unwrap_or(true) {
                    cache.last_logits_dev = Some(e.uninit(n_vocab)?);
                }
                let dst = cache.last_logits_dev.as_mut().unwrap();
                e.dtod_copy_view(&logits.slice(0..n_vocab), dst)?;
            }
            vec![Vec::new()]
        } else if let Some(p) = pristine.as_ref() {
            vec![e.dtoh(p)?]
        } else {
            vec![e.dtoh(&logits)?]
        };
        // decode_layers_eager does NOT advance cache.pos (decode_step_h advances it after
        // the head); the batched path advances every cache at the tail — same here.
        caches[0].pos += 1;
        Ok((rows, next))
    }

    /// One batched greedy-decode step over B independent sequences.
    /// `tokens[b]` is sequence b's input token; `caches[b]` its private cache (position,
    /// quantized KV, GDN/conv state). Returns the B logits rows (host, [n_vocab] each).
    /// Each cache's pos/len advance exactly as `decode_step_h` would.
    pub fn decode_step_batch(
        &self,
        e: &Engine,
        tokens: &[u32],
        caches: &mut [&mut Cache],
    ) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        let (rows, _) = self.decode_step_batch_sampled(e, tokens, caches, &[])?;
        Ok(rows)
    }

    /// `decode_step_batch` + DEVICE-SIDE SAMPLING for eligible rows (the batched-tick lever,
    /// 2026-08-01): the host sampler's temp-path is O(n_vocab) with a full-vocab exp per row
    /// (measured 1.36 ms/row at the 9B's 248320 vocab = 10.9 ms/tick at B=8 — the single
    /// largest component of the serving tick). Here each requested row samples ON DEVICE
    /// between the lm_head matmul and the logits D2H:
    ///   temp <= 0 (greedy): the 2-pass device argmax — bit-identical to host argmax
    ///     (argmax-gate contract, same kernels as the dc serving path).
    ///   temp > 0: gumbel_perturb(seed, ctr, temp) + the same argmax = ONE categorical draw
    ///     from softmax(logits/temp) — the sampled-spec Philox machinery. Deterministic per
    ///     (seed, ctr) and INDEPENDENT of batch composition (the isolation contract;
    ///     decode-batch-gate gate3). NOTE: the draw stream differs from the host sampler's
    ///     SplitMix64 (distribution-equal, seed-deterministic, NOT byte-equal to the old
    ///     host draws) — greedy rows are unchanged bit-exact.
    /// `samp[bi] = Some((temp, seed, ctr))` requests a device sample for row bi; the full
    /// logits rows are still returned (worker keeps last_logits semantics + fallback rows).
    pub fn decode_step_batch_sampled(
        &self,
        e: &Engine,
        tokens: &[u32],
        caches: &mut [&mut Cache],
        samp: &[Option<(f32, u64, u32)>],
    ) -> Result<(Vec<Vec<f32>>, Vec<Option<u32>>), Box<dyn std::error::Error>> {
        self.decode_step_batch_sampled_lean(e, tokens, caches, samp, false)
    }

    /// `decode_step_batch_sampled` + LEAN LOGITS (increment 2 component 3, 2026-08-01):
    /// with `lean`, device-sampled rows SKIP the [n_vocab] logits D2H (9.4%/32.5% of the
    /// pre-/post-inc2 tick profile) — their returned row is EMPTY. The audit-mapped
    /// consumers: (a) the next tick's host sample — never fires, `device_next` carries the
    /// token; (b) the graph-promotion argmax — reads only prefill logits (generated empty);
    /// (c) the KV-reuse pool park at retire — the REAL consumer, served by a per-cache
    /// device park: the row is dtod-copied into `cache.last_logits_dev` (device bandwidth)
    /// and D2H'd ONCE at retire by the worker. Rows without a device sample keep a per-row
    /// D2H. `lean=false` is bit-for-bit the previous method (gates + non-serving callers).
    pub fn decode_step_batch_sampled_lean(
        &self,
        e: &Engine,
        tokens: &[u32],
        caches: &mut [&mut Cache],
        samp: &[Option<(f32, u64, u32)>],
        lean: bool,
    ) -> Result<(Vec<Vec<f32>>, Vec<Option<u32>>), Box<dyn std::error::Error>> {
        self.decode_step_batch_sampled_lean_masked(e, tokens, caches, samp, &[], lean)
    }

    /// `decode_step_batch_sampled_lean` + GRAMMAR MASKS (constrained decoding, 2026-08-03):
    /// `masks[bi] = Some((packed_bitset, words))` bans every unset-bit vocab id on row bi
    /// (mask_logits_f32, -FLT_MAX) BETWEEN the lm_head matmul and the device sampler, so a
    /// constrained row rides the SAME device-sample/lean-logits tick as everyone else — no
    /// full-row D2H, no host O(n_vocab) sample. Contract: a masked row must also request a
    /// device sample. The row's PRISTINE logits are preserved for their consumers before the
    /// in-place ban: lean rows park the unmasked row into `cache.last_logits_dev` (the
    /// retire-time reuse-pool park stays unmasked — continuations resume grammar-free, the
    /// v1 host-path contract), non-lean rows D2H the unmasked row. `masks = &[]` is
    /// bit-for-bit the unmasked method.
    pub fn decode_step_batch_sampled_lean_masked(
        &self,
        e: &Engine,
        tokens: &[u32],
        caches: &mut [&mut Cache],
        samp: &[Option<(f32, u64, u32)>],
        masks: &[Option<(&CudaSlice<u32>, usize)>],
        lean: bool,
    ) -> Result<(Vec<Vec<f32>>, Vec<Option<u32>>), Box<dyn std::error::Error>> {
        // NOTE (inc3 3c, 2026-08-01, KILLED ARM): a deferred-token-readback variant (all
        // chunks of a tick writing device-sampled tokens into one shared buffer, ONE
        // dtoh_u32 after the last chunk instead of one per chunk) measured FLAT at serve
        // level on the 5090 (N=4 medians within +-0.7% at c=8/16/32 — 3 saved syncs
        // against a ~100 ms weight-bound tick is ~0.1%, below resolution). Killed per the
        // flags doctrine; receipts research/batched-tick-inc3-20260801 (serve-points.jsonl
        // base vs defer arms) are the record. The per-chunk [B]-u32 readback below IS the
        // tick's only steady-state D2H — one per chunk, none per seq.
        let b_n = tokens.len();
        assert!(b_n >= 1 && b_n == caches.len(), "tokens/caches length mismatch");
        // ---- H3: B=1 FAST-PATH (serve-path phase 2, 2026-08-05) ----------------------------
        // At b_n==1 every projection below calls `matmul_pre(.., b_n)` with m=1, which is
        // ALREADY the m=1 mmvq dispatch — so the m=1 *kernel family* was never the gap. What
        // this body does NOT have is the m=1 *fusion chain* that `decode_step_h` carries:
        //   - the cross-layer add+norm+quantize fusion (`add_rms_norm_q8_1`: 3 launches -> 1),
        //   - the fused SwiGLU epilogue (`silu_mul_scaled_q8_1`: folds ffn_down's quantize
        //     into its producer) and, with it, `matmul_pre_dual_noscale`'s gate+up pair
        //     fusion — i.e. phase-1 LEVER 1.
        // Routing b_n==1 through `decode_layers_eager` (the SHARED trunk `decode_step_h` and
        // the ppN stages already use, lifted verbatim — not a copy) makes every present and
        // future m=1 lever fire on the serve path automatically, which is the durable half of
        // this change. The epilogue (grammar mask -> device sample -> lean logits park) is
        // kept EXACTLY as the batched path runs it, so the serving contract is untouched.
        // BIT-IDENTITY: the trunk is the same function `decode_step_h` calls, and every
        // fusion it enables is kernel-check-pinned bit-identical to its unfused sequence
        // (add_rms_norm == add;rms_norm | _q8_1 == +quantize_q8_1 | dual_noscale == two
        // matmul_pre_noscale). Gate: decode-batch-gate B=1 vs decode_step_h + serve stream
        // identity. MEMRA_SERVE_B1FAST=0 is the rollback/A-B seam.
        if b_n == 1
            && Self::b1_fast_on()
            && !self.is_gemma4_e4b()
            && self.cfg.gemma4.is_none()
            && self.cfg.m3.is_none()
            && crate::pp::pp_cuts(self.layers.len()).is_none()
            && !e.verify_exact_on()
        {
            return self.decode_step_b1_fast(e, tokens[0], caches, samp, masks, lean);
        }
        // MEMRA_DECODE_BATCH_CAP (experimental door, serving-lane tier probe 2026-08-01):
        // default 8 keeps the v1 exactness policy — B=2..8 rides the verify-tier batched
        // mmvq arms, per-row bit-identical to isolated m=1 decode. Values >8 are a
        // MEASUREMENT DOOR ONLY: m=9..15 falls to the grid.y=m dp4a tail (m weight
        // re-reads + a different reduce shape) and m>=16 crosses into the GEMM tier
        // (block-scale f32 rounding) — BOTH break the "byte-identical to isolated"
        // serving contract. Never default this above 8 without the batched-tier
        // exactness policy landing.
        let cap = Self::decode_batch_cap();
        // EXACT-16 TIER (increment 3a): chunks of 9..=16 are admitted WITHOUT the env door
        // when every matmul has a bit-exact b16-class kernel (see decode_batch_exact16_ok).
        // The verify_exact scope below pins that dispatch for the whole step: it turns off
        // the m>=16 GEMM arms (qmatvec_gemm + MMQ + fp8/f16/fp4 — all block-scale/foreign
        // numeric configs) so every projection rides the batched-mmvq b16 tier, which is
        // per-(token,row) bit-identical to isolated m=1 decode (gate2 bit-strength PASS at
        // B=12/16, s32+s160, 5090 receipts research/batched-tick-inc3-20260801). Without
        // the exact tier, B>cap stays refused; the env door (MEMRA_DECODE_BATCH_CAP) keeps
        // its old meaning as the non-exact measurement probe.
        let exact16 = b_n > 8 && b_n <= 16 && self.decode_batch_exact16_ok();
        assert!(
            b_n <= cap || exact16,
            "decode_step_batch: B={b_n} > cap {cap} with no exact tier (Q8_0 m>8 needs the \
             q8rp mirror's b16 class; m>16 crosses GEMM/dp4a numeric configs) — refused"
        );
        struct ExactScope<'a>(&'a Engine, bool);
        impl Drop for ExactScope<'_> {
            fn drop(&mut self) {
                if self.1 {
                    self.0.set_verify_exact(false);
                }
            }
        }
        let _exact_scope = ExactScope(e, exact16);
        if exact16 {
            e.set_verify_exact(true);
        }
        assert!(
            !self.is_gemma4_e4b() && self.cfg.gemma4.is_none(),
            "decode_step_batch v1 covers the hybrid non-gemma4 trunk only"
        );
        // step35: the batched body below is the generic Full arm (uniform n_head, 128-dim rope
        // everywhere, no window, no head-wise gate). B=1 already routed to the shared eager
        // trunk above; B>1 has no step35 twin.
        assert!(
            self.cfg.step35.is_none(),
            "decode_step_batch has no step35 arm at B>1 (per-layer n_head / partial rope / SWA \
             offset view / head-wise gate) — B=1 rides the shared eager trunk"
        );
        let cfg = &self.cfg;
        let n_embd = cfg.n_embd as usize;
        let eps = cfg.rms_eps;
        let n_head = cfg.n_head as usize;
        let n_head_kv = cfg.n_head_kv as usize;
        let head_dim = cfg.head_dim_k as usize;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let rope_dims = cfg.rope_dim_count as usize;

        // MEMRA_BATCH_PHASE=1: sync-bounded phase accumulation (diagnostics — see header note).
        // Initialized BEFORE the tick-input assembly below so slot 0 covers the HOST side of
        // setup (pos_v/ptr-table builds, embed gather) as well as the H2D sync — the audit-fix
        // lane's Q6 instrumentation gap (research/audit-fixes2-20260805): the old placement
        // started the clock after the assembly, so slot 0 under-reported setup.
        let ph_on = batch_phase_on();
        let mut ph_last = std::time::Instant::now();
        let ph_mark = |slot: usize,
                       last: &mut std::time::Instant|
         -> Result<(), Box<dyn std::error::Error>> {
            if ph_on {
                e.stream().synchronize()?;
                let now = std::time::Instant::now();
                BATCH_PHASE.lock().unwrap()[slot] += (now - *last).as_secs_f64();
                *last = now;
            }
            Ok(())
        };

        // Per-row rope positions (each sequence at its own depth).
        let pos_v: Vec<i32> = caches.iter().map(|c| c.pos as i32).collect();
        let pos_d = e.htod_i32(&pos_v)?;

        // Per-step STATE POINTER TABLE (one H2D): for every linear layer, [conv x B]
        // [ssm_in x B][ssm_out x B] device addresses. The batched state kernels read their
        // sequence's pointer from these arrays — states stay per-cache (no pooling refactor),
        // yet conv/prep/scan collapse from 3xB launches per layer to 3. Rebuilt every step
        // because the ssm ping-pong swaps pointers host-side after each scan.
        // INCREMENT 2 (2026-08-01): the SAME table now also carries, for every FULL-attn
        // layer, [k0,v0,k1,v1,...] cache base addresses — the z-batched seqs append and
        // seqs fa_decode kernels read their sequence's cache through it (the MoE
        // expert-table pattern), collapsing 2xB launches per attn layer to 2.
        let mut lin_base: Vec<Option<usize>> = vec![None; self.layers.len()];
        let mut attn_base: Vec<Option<usize>> = vec![None; self.layers.len()];
        let mut ptrs: Vec<u64> = Vec::new();
        {
            use cudarc::driver::DevicePtr;
            let s = &e.gpu.stream();
            for (il, layer) in self.layers.iter().enumerate() {
                match &layer.mixer {
                    Mixer::Linear(_) => {
                        lin_base[il] = Some(ptrs.len());
                        for c in caches.iter() {
                            let rl = c.recur[il].as_ref().unwrap();
                            let (p, _g) = rl.conv_state.device_ptr(s);
                            ptrs.push(p as u64);
                        }
                        for c in caches.iter() {
                            let rl = c.recur[il].as_ref().unwrap();
                            let (p, _g) = rl.ssm_state.device_ptr(s);
                            ptrs.push(p as u64);
                        }
                        for c in caches.iter() {
                            let rl = c.recur[il].as_ref().unwrap();
                            let (p, _g) = rl.ssm_state_alt.device_ptr(s);
                            ptrs.push(p as u64);
                        }
                    }
                    Mixer::Full(_) => {
                        attn_base[il] = Some(ptrs.len());
                        for c in caches.iter() {
                            let kvl = c.kv[il].as_ref().unwrap();
                            let (pk, _g) = kvl.k.device_ptr(s);
                            let (pv, _g2) = kvl.v.device_ptr(s);
                            ptrs.push(pk as u64);
                            ptrs.push(pv as u64);
                        }
                    }
                    Mixer::Mla(_) => crate::hybrid::mla_forward_unimplemented(),
                }
            }
        }
        let ptr_table = if ptrs.is_empty() { None } else { Some(e.htod_u64(&ptrs)?) };

        // INCREMENT 2 arm picks (per STEP — t_kv is layer-invariant within a tick):
        // - seqs APPEND: format-only condition (per-row program is t_kv-independent);
        //   default flash module only (fp8-KV rides the per-seq g-module path).
        // - seqs FA: every row must take the v4 eager arm at ITS OWN t_kv AND all rows
        //   must share ONE fa_split_keys rung (the rows-twins' straddle law) — a rung
        //   crossing inside the batch keeps the per-seq loop for that step, so each
        //   sequence always executes the exact program its isolated run would.
        // MEMRA_BATCH_APPEND=0 / MEMRA_BATCH_FA=0 are the rollback/A-B seams.
        let t_kvs: Vec<usize> = caches.iter().map(|c| c.pos + 1).collect();
        let t_kv_max = *t_kvs.iter().max().unwrap();
        let seqs_append = {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| std::env::var("MEMRA_BATCH_APPEND").as_deref() != Ok("0"))
        } && !Engine::kv_fp8_on();
        let sp0 = crate::fa_split_keys(t_kvs[0], cfg.n_head_kv as usize);
        let seqs_fa = {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| std::env::var("MEMRA_BATCH_FA").as_deref() != Ok("0"))
        } && t_kvs.iter().all(|&t| crate::fa_seqs_eligible(t, head_dim))
          && t_kvs.iter().all(|&t| crate::fa_split_keys(t, cfg.n_head_kv as usize) == sp0);

        // Embed all B tokens -> x [B, n_embd] (host gather, one H2D).
        let mut x = e.htod(&self.embd.gather(n_embd, tokens))?;
        ph_mark(0, &mut ph_last)?;

        for (il, layer) in self.layers.iter().enumerate() {
            // ---- attn_norm + q8_1 quantize, batched (B rows) ----
            let anorm = layer.attn_norm.float_data();
            let mut xn = e.uninit(b_n * n_embd)?;
            e.rms_norm(&x, anorm, &mut xn, n_embd, b_n, eps)?;
            let (hq, hd) = e.quantize_q8_1(&xn, b_n, n_embd)?;

            // ---- mixer ----
            let mixed: CudaSlice<f32> = match &layer.mixer {
                Mixer::Mla(_) => crate::hybrid::mla_forward_unimplemented(),
                Mixer::Full(fa) => {
                    // Batched projections: one weight read serves all B rows.
                    let qf = e.matmul_pre(&fa.wq, &hq, &hd, &xn, b_n)?;
                    let mut k = e.matmul_pre(&fa.wk, &hq, &hd, &xn, b_n)?;
                    let v = e.matmul_pre(&fa.wv, &hq, &hd, &xn, b_n)?;

                    let gated = cfg.attn_out_gate();
                    let (mut q, gate) = if gated {
                        let mut qs = e.uninit(b_n * n_head * head_dim)?;
                        let mut gs = e.uninit(b_n * n_head * head_dim)?;
                        e.q_gate_split(&qf, &mut qs, &mut gs, head_dim, n_head, b_n)?;
                        (qs, Some(gs))
                    } else {
                        (qf, None)
                    };

                    // QK-norm over B*n_head rows, rope with per-row positions.
                    let mut qn = e.uninit(b_n * n_head * head_dim)?;
                    e.rms_norm(&q, fa.q_norm.float_data(), &mut qn, head_dim, b_n * n_head, eps)?;
                    q = qn;
                    let mut kn = e.uninit(b_n * n_head_kv * head_dim)?;
                    e.rms_norm(&k, fa.k_norm.float_data(), &mut kn, head_dim, b_n * n_head_kv, eps)?;
                    k = kn;
                    e.rope_neox(&mut q, &pos_d, head_dim, rope_dims, n_head, b_n,
                                cfg.rope_freq_base, 1.0)?;
                    e.rope_neox(&mut k, &pos_d, head_dim, rope_dims, n_head_kv, b_n,
                                cfg.rope_freq_base, 1.0)?;
                    ph_mark(1, &mut ph_last)?;

                    // INCREMENT 2 (2026-08-01): the per-seq (append, attend) launch train
                    // becomes two phases. Phase A appends all B rows (one z-batched launch,
                    // or the per-seq loop on the seam/fp8 path); phase B attends all B
                    // sequences (one blockIdx.z launch + one combine on the batched arm —
                    // which also reads q / writes attn at row offsets, killing the per-seq
                    // q/a dtod copies — or the per-seq loop when any row is outside the v4
                    // arm / a split rung crosses inside the batch). Caches are disjoint per
                    // sequence, so the phase split leaves every row's math untouched.
                    let q_dim = n_head * head_dim;
                    let kv_dim = n_head_kv * head_dim;
                    let mut attn = e.uninit(b_n * q_dim)?;
                    // ---- phase A: KV append (all B rows) ----
                    if seqs_append {
                        let (kdk, kdv, ktb, vtb) = {
                            let kvl = caches[0].kv[il].as_ref().unwrap();
                            (kvl.kv_dim_k, kvl.kv_dim_v, kvl.k_tok_bytes, kvl.v_tok_bytes)
                        };
                        let base = attn_base[il].expect("full layer missing from pointer table");
                        let table = ptr_table.as_ref().expect("pointer table missing");
                        let kv_view = table.slice(base..base + 2 * b_n);
                        e.append_kv_quantized_seqs(&k, &v, &kv_view, &pos_d, b_n,
                                                   kdk, kdv, ktb, vtb)?;
                        for cache in caches.iter_mut() {
                            let kvl = cache.kv[il].as_mut().unwrap();
                            debug_assert_eq!(kvl.len, cache.pos, "kv len / pos out of lockstep");
                            kvl.len += 1;
                        }
                    } else {
                        for (bi, cache) in caches.iter_mut().enumerate() {
                            let kvl = cache.kv[il].as_mut().unwrap();
                            let k_row = k.slice(bi * kv_dim..(bi + 1) * kv_dim);
                            let v_row = v.slice(bi * kv_dim..(bi + 1) * kv_dim);
                            e.append_kv_quantized_view(
                                &k_row, &v_row, &mut kvl.k, &mut kvl.v, kvl.len,
                                kvl.kv_dim_k, kvl.kv_dim_v, kvl.k_tok_bytes, kvl.v_tok_bytes,
                                Engine::kv_fp8_on(),
                            )?;
                            kvl.len += 1;
                        }
                    }
                    ph_mark(2, &mut ph_last)?;
                    // ---- phase B: attention (all B sequences) ----
                    if seqs_fa {
                        let (ktb, vtb) = {
                            let kvl = caches[0].kv[il].as_ref().unwrap();
                            (kvl.k_tok_bytes, kvl.v_tok_bytes)
                        };
                        let base = attn_base[il].expect("full layer missing from pointer table");
                        let table = ptr_table.as_ref().expect("pointer table missing");
                        let kv_view = table.slice(base..base + 2 * b_n);
                        e.fa_decode_batch_seqs_v4(&q, &kv_view, &pos_d, &mut attn,
                                                  head_dim, n_head, n_head_kv, b_n,
                                                  t_kv_max, scale, sp0, ktb, vtb)?;
                        ph_mark(4, &mut ph_last)?;
                    } else {
                        for (bi, cache) in caches.iter_mut().enumerate() {
                            let kvl = cache.kv[il].as_mut().unwrap();
                            let t_kv = kvl.len;
                            let k_view = e.view_u8(&kvl.k, t_kv * kvl.k_tok_bytes);
                            let v_view = e.view_u8(&kvl.v, t_kv * kvl.v_tok_bytes);
                            // fa_decode wants a q slice starting at row bi: the fallback arm
                            // scratch-copies the row (q8-class µs cost); the seqs arm above
                            // reads/writes row offsets in place.
                            let mut q_row = e.uninit(q_dim)?;
                            e.dtod_copy_view(&q.slice(bi * q_dim..(bi + 1) * q_dim), &mut q_row)?;
                            ph_mark(3, &mut ph_last)?;
                            let mut a_row = e.uninit(q_dim)?;
                            e.fa_decode_kvmod(
                                &q_row, &k_view, &v_view, &mut a_row, head_dim, n_head, n_head_kv,
                                t_kv, scale, kvl.k_tok_bytes, kvl.v_tok_bytes, Engine::kv_fp8_on(),
                            )?;
                            ph_mark(4, &mut ph_last)?;
                            e.dtod_copy_into(&a_row, &mut attn, bi * q_dim)?;
                            ph_mark(3, &mut ph_last)?;
                        }
                    }

                    // Output gate (element-wise — batches whole) + o-proj at m=B.
                    let attn_g = match &gate {
                        Some(g) => {
                            let n = b_n * q_dim;
                            let mut gsig = e.uninit(n)?;
                            e.sigmoid(g, &mut gsig, n)?;
                            let mut ag = e.uninit(n)?;
                            e.mul(&attn, &gsig, &mut ag, n)?;
                            ag
                        }
                        None => attn,
                    };
                    let o = e.matmul(&fa.wo, &attn_g, b_n)?;
                    ph_mark(5, &mut ph_last)?;
                    o
                }
                Mixer::Linear(la) => {
                    // v2 (the B-scaling fix): the GDN mixer's PROJECTIONS carry the layer's
                    // weight mass — batch them at m=B so wqkv/gate/beta/alpha/ssm_out stream
                    // ONCE per step instead of once per sequence. Only the recurrent state ops
                    // (fused conv ring, gdn prep, gdn scan) stay per-seq — they are state-bound
                    // micro-kernels, not weight readers. Composition unchanged vs v1 (matmul_pre
                    // == fused2 per (tensor,row); _bN mmvq per-row == m=1): same numeric config.
                    let ssm = cfg.ssm.as_ref().expect("linear mixer requires ssm cfg");
                    let d_state = ssm.state_size as usize;
                    let num_k = ssm.group_count as usize;
                    let num_v = ssm.time_step_rank as usize;
                    let d_conv = ssm.conv_kernel as usize;
                    let key_dim = d_state * num_k;
                    let value_dim = d_state * num_v;
                    let conv_dim = key_dim * 2 + value_dim;
                    let gdn_scale = 1.0 / (d_state as f32).sqrt();

                    // ---- batched projections (the weight win) ----
                    let qkv_mixed = e.matmul_pre(&la.wqkv, &hq, &hd, &xn, b_n)?;
                    let z = e.matmul_pre(&la.wqkv_gate, &hq, &hd, &xn, b_n)?;
                    let beta_raw = e.matmul_pre(&la.ssm_beta, &hq, &hd, &xn, b_n)?;
                    let alpha = e.matmul_pre(&la.ssm_alpha, &hq, &hd, &xn, b_n)?;
                    ph_mark(6, &mut ph_last)?;

                    // ---- batched recurrent state ops (3 launches for all B sequences) ----
                    let base = lin_base[il].expect("linear layer missing from pointer table");
                    let table = ptr_table.as_ref().expect("pointer table missing");
                    let conv_view = table.slice(base..base + b_n);
                    let in_view = table.slice(base + b_n..base + 2 * b_n);
                    let out_view = table.slice(base + 2 * b_n..base + 3 * b_n);
                    let mut conv_outs = e.uninit(b_n * conv_dim)?;
                    e.ssm_conv1d_fused_decode_b(&qkv_mixed, &conv_view,
                                                la.ssm_conv1d.float_data(), &mut conv_outs,
                                                conv_dim, d_conv, b_n)?;
                    let mut q_l2 = e.uninit(b_n * value_dim)?;
                    let mut k_l2 = e.uninit(b_n * value_dim)?;
                    let mut v_gd = e.uninit(b_n * value_dim)?;
                    let mut beta_b = e.uninit(b_n * num_v)?;
                    let mut g_log = e.uninit(b_n * num_v)?;
                    e.gdn_prep_decode_b(&conv_outs, &beta_raw, &alpha,
                                        la.ssm_dt.float_data(), la.ssm_a.float_data(),
                                        &mut q_l2, &mut k_l2, &mut v_gd, &mut beta_b, &mut g_log,
                                        d_state, num_v, num_k, key_dim, eps, conv_dim, b_n)?;
                    let mut o_all = e.uninit(b_n * value_dim)?;
                    e.gdn_scan_s128_batched(&q_l2, &k_l2, &v_gd, &g_log, &beta_b,
                                            &in_view, &out_view, &mut o_all,
                                            num_v, b_n, gdn_scale)?;
                    // ping-pong: scan wrote each seq's alt buffer; swap host handles (the
                    // NEXT step's table rebuild picks up the new canonical pointers).
                    for cache in caches.iter_mut() {
                        let rl = cache.recur[il].as_mut().unwrap();
                        std::mem::swap(&mut rl.ssm_state, &mut rl.ssm_state_alt);
                    }
                    ph_mark(7, &mut ph_last)?;

                    // ---- batched gated norm + out-projection ----
                    let o = if e.uses_q8_1_fast(&la.ssm_out) {
                        let (gq, gd) = e.gated_rmsnorm_q8_1(&o_all, la.ssm_norm.float_data(),
                                                            &z, d_state, b_n * num_v, eps)?;
                        let g0 = e.zeros(0)?;
                        e.matmul_pre(&la.ssm_out, &gq, &gd, &g0, b_n)?
                    } else {
                        let mut gn = e.uninit(b_n * value_dim)?;
                        e.gated_rmsnorm(&o_all, la.ssm_norm.float_data(), &z, &mut gn,
                                        d_state, b_n * num_v, eps)?;
                        e.matmul(&la.ssm_out, &gn, b_n)?
                    };
                    ph_mark(8, &mut ph_last)?;
                    o
                }
            };

            // ---- residual add + post_attn_norm + FFN, batched ----
            let pnorm = layer.post_attn_norm.float_data();
            let mut x1 = e.uninit(b_n * n_embd)?;
            let mut z = e.uninit(b_n * n_embd)?;
            e.add_rms_norm(&x, &mixed, pnorm, &mut x1, &mut z, n_embd, b_n, eps)?;
            let ffn_out = match &layer.ffn {
                crate::hybrid::Ffn::Dense { ffn_gate, ffn_up, ffn_down } => {
                    // v1 covers the SiLU family; M3's swigluoai clamp rides a scaled epilogue
                    // (m=1 fused tier) — batched M3 lands with the batched-fusion pass.
                    assert!(self.cfg.m3.is_none(),
                            "decode_step_batch v1: M3 swigluoai FFN not yet batched");
                    let n_ff = ffn_gate.out_features();
                    let (zq, zd) = e.quantize_q8_1(&z, b_n, n_embd)?;
                    // REFUTED ARM (lane/q27-deepdive, 2026-08-05): fusing this gate+up pair
                    // into `matmul_q8_fused2_t` (the fused2_b8 tier) measured FLAT-TO-NEGATIVE
                    // at the serving tick — bench c=8 213.1/213.8, 213.9/214.4, 214.4/213.5
                    // (sign flips) and serve c=8 paired mean −0.20% over 3 passes. Mechanism:
                    // unlike m=1 (where the pair is 128 of 1015 launches in a 7.67%-gap tick),
                    // the c=8 tick is 73.2% one weight-bound kernel class with launch cost
                    // already hidden — halving 128 launches of ~28k buys nothing. The m=1 arm
                    // in `matmul_pre_dual_noscale` (+0.94%) stays; this call site keeps the two
                    // launches. Kernel + fused2_b8 wrapper retained: kernel-check gates it at
                    // m=5/8 and matmul_q8_fused2_t serves the verify tier. Receipts:
                    // research/q27-deepdive-20260805/ (lever3-bench-*, serve-points.jsonl).
                    let g = e.matmul_pre(ffn_gate, &zq, &zd, &z, b_n)?;
                    let u = e.matmul_pre(ffn_up, &zq, &zd, &z, b_n)?;
                    let mut act = e.uninit(b_n * n_ff)?;
                    e.silu_mul(&g, &u, &mut act, b_n * n_ff)?;
                    let (aq, ad) = e.quantize_q8_1(&act, b_n, n_ff)?;
                    e.matmul_pre(ffn_down, &aq, &ad, &act, b_n)?
                }
                crate::hybrid::Ffn::Moe(m) => self.moe_ffn_il_zq8(e, m, &z, None, b_n, il as u16)?,
            };
            // next-layer input x = x1 + ffn_out (batched element-wise add)
            let mut x2 = e.uninit(b_n * n_embd)?;
            e.add(&x1, &ffn_out, &mut x2, b_n * n_embd)?;
            x = x2;
            ph_mark(9, &mut ph_last)?;
        }

        // ---- output norm + lm_head at m=B, one D2H ----
        let mut hn = e.uninit(b_n * n_embd)?;
        e.rms_norm(&x, self.output_norm.float_data(), &mut hn, n_embd, b_n, eps)?;
        let logits = e.matmul(&self.output, &hn, b_n)?;
        ph_mark(10, &mut ph_last)?;

        // GRAMMAR MASKS (constrained decoding): preserve each masked row's PRISTINE logits
        // for its consumer (lean park into cache.last_logits_dev — the reuse-pool park stays
        // unmasked, the v1 contract — or the non-lean D2H), then ban in place BEFORE the
        // device sampler reads the row. All stream-ordered; masks=&[] takes no new branch.
        let n_vocab = self.output.out_features();
        let mut logits = logits;
        let mut pristine: Vec<Option<CudaSlice<f32>>> = Vec::new();
        if masks.iter().take(b_n).any(|m| m.is_some()) {
            pristine.resize_with(b_n, || None);
            for (bi, m) in masks.iter().take(b_n).enumerate() {
                let Some((mask, words)) = m else { continue };
                assert!(samp.get(bi).copied().flatten().is_some(),
                        "grammar-masked row {bi} must request a device sample");
                if lean {
                    let cache = &mut caches[bi];
                    if cache.last_logits_dev.as_ref().map(|d| d.len() < n_vocab).unwrap_or(true) {
                        cache.last_logits_dev = Some(e.uninit(n_vocab)?);
                    }
                    let dst = cache.last_logits_dev.as_mut().unwrap();
                    e.dtod_copy_view(&logits.slice(bi * n_vocab..(bi + 1) * n_vocab), dst)?;
                } else {
                    let mut p = e.uninit(n_vocab)?;
                    e.dtod_copy_view(&logits.slice(bi * n_vocab..(bi + 1) * n_vocab), &mut p)?;
                    pristine[bi] = Some(p);
                }
                e.mask_logits_col(&mut logits, mask, bi, n_vocab, *words)?;
            }
        }

        // Device-side sampling for requested rows (see the method doc). Enqueued before the
        // big logits D2H so the tiny [B] token readback rides the same sync.
        let mut next: Vec<Option<u32>> = vec![None; b_n];
        if samp.iter().take(b_n).any(|s| s.is_some()) {
            let mut toks = e.alloc_u32_zeroed(b_n)?;
            let mut perturb: Option<CudaSlice<f32>> = None;
            for (bi, s) in samp.iter().take(b_n).enumerate() {
                let Some((temp, seed, ctr)) = s else { continue };
                if *temp <= 0.0 {
                    e.argmax_token_device_col(&logits, bi, n_vocab, &mut toks, bi)?;
                } else {
                    if perturb.is_none() {
                        perturb = Some(e.zeros(n_vocab)?);
                    }
                    let pb = perturb.as_mut().unwrap();
                    e.gumbel_perturb_col(&logits, bi, pb, n_vocab, *seed, *ctr, *temp)?;
                    e.argmax_token_device_col(pb, 0, n_vocab, &mut toks, bi)?;
                }
            }
            let host_toks = e.dtoh_u32(&toks)?;
            for (bi, s) in samp.iter().take(b_n).enumerate() {
                if s.is_some() {
                    next[bi] = Some(host_toks[bi]);
                }
            }
        }

        let lean_any = lean && samp.iter().take(b_n).any(|s| s.is_some());
        let rows: Vec<Vec<f32>> = if lean_any {
            // LEAN: park device-sampled rows on-device (per-cache buffer, dtod); D2H only
            // the rows that still need host logits. No sampled rows + no fallback rows =
            // the big D2H disappears (the [B] token readback above already synced).
            for (bi, s) in samp.iter().take(b_n).enumerate() {
                if s.is_none() { continue; }
                // grammar-masked rows already parked their PRISTINE copy above — the
                // in-place ban has since poisoned this row for the reuse-pool consumer.
                if masks.get(bi).copied().flatten().is_some() { continue; }
                let cache = &mut caches[bi];
                if cache.last_logits_dev.as_ref().map(|d| d.len() < n_vocab).unwrap_or(true) {
                    cache.last_logits_dev = Some(e.uninit(n_vocab)?);
                }
                let dst = cache.last_logits_dev.as_mut().unwrap();
                e.dtod_copy_view(&logits.slice(bi * n_vocab..(bi + 1) * n_vocab), dst)?;
            }
            (0..b_n)
                .map(|bi| {
                    if samp.get(bi).copied().flatten().is_some() {
                        Ok(Vec::new())
                    } else {
                        e.dtoh_view(&logits.slice(bi * n_vocab..(bi + 1) * n_vocab))
                    }
                })
                .collect::<Result<_, _>>()?
        } else {
            let host = e.dtoh(&logits)?;
            (0..b_n).map(|bi| {
                // grammar-masked non-lean rows return the PRISTINE copy (the in-place ban
                // must never leak into last_logits — reuse-pool/park semantics unchanged).
                if let Some(p) = pristine.get(bi).and_then(|p| p.as_ref()) {
                    return e.dtoh(p);
                }
                Ok(host[bi * n_vocab..(bi + 1) * n_vocab].to_vec())
            }).collect::<Result<_, _>>()?
        };
        for c in caches.iter_mut() {
            c.pos += 1;
        }
        ph_mark(11, &mut ph_last)?;
        Ok((rows, next))
    }
}
