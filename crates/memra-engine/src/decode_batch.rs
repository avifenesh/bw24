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

/// Per-step, per-LAYER-RANGE invariants the batched trunk needs: the device state-pointer
/// table for the range's layers, the arm picks, and the per-row `t_kv` snapshot. Built once
/// per step per range by `HybridModel::batch_layer_ctx`, consumed by `decode_batch_layers`.
///
/// WHY IT IS RANGE-SCOPED AND NOT STEP-SCOPED (this is the whole point of the struct):
/// `ptr_table` is a `CudaSlice<u64>` of DEVICE ADDRESSES, uploaded through `e` — so it lives
/// on `e`'s device, and its entries are pointers into caches that live on the device that
/// OWNS those layers. Under a pp stage split, stage s runs layers [fence[s], fence[s+1])
/// whose cache state was allocated by stage s's engine (`pp::new_cache` -> `Cache::new_ppn`),
/// so stage s must build its OWN table through its OWN engine. One step-wide table built on
/// the primary would put every stage's kernel arguments in stage-0's HBM — a peer read per
/// pointer fetch, which is the exact cliff `pp::refuse_unsplit_if_remote` exists to stop.
/// `lo`/`hi` are recorded so the consumer can assert the ctx it was handed matches the range
/// it was asked to run (the offsets in `lin_base`/`attn_base` are only valid for that range).
pub(crate) struct BatchLayerCtx {
    /// Offset into `ptr_table` of layer il's [conv x B][ssm_in x B][ssm_out x B] block
    /// (linear-attn layers only). Indexed by ABSOLUTE layer id; `None` off-range.
    lin_base: Vec<Option<usize>>,
    /// Offset into `ptr_table` of layer il's [k0,v0,k1,v1,..] block (full-attn layers only).
    /// Indexed by ABSOLUTE layer id; `None` off-range.
    attn_base: Vec<Option<usize>>,
    ptr_table: Option<CudaSlice<u64>>,
    /// Per-row `pos + 1` — the t_kv each sequence attends at this step. Layer-invariant
    /// within a step, so the arm picks below are decided once.
    t_kvs: Vec<usize>,
    t_kv_max: usize,
    /// The single `fa_split_keys` rung every row shares (the rows-twins straddle law).
    sp0: usize,
    seqs_append: bool,
    seqs_fa: bool,
    lo: usize,
    hi: usize,
}

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
/// Accumulate the elapsed time since `last` into phase slot `slot` and re-stamp `last`.
/// No-op unless `MEMRA_BATCH_PHASE=1`. Syncs the ambient stream first, so under a pp stage
/// scope this bounds the STAGE's stream, which is what the caller is timing.
///
/// A free fn rather than the closure it replaced: `decode_batch_layers` (the pp stage seam)
/// runs the instrumented layer loop, so the marker has to be callable from both the seam
/// and its caller's epilogue. `batch_phase_on()` is a `OnceLock` memo, so per-call cost is
/// the same atomic load the hoisted `ph_on` local was.
fn ph_mark(
    e: &Engine,
    slot: usize,
    last: &mut std::time::Instant,
) -> Result<(), Box<dyn std::error::Error>> {
    if batch_phase_on() {
        e.stream().synchronize()?;
        let now = std::time::Instant::now();
        BATCH_PHASE.lock().unwrap()[slot] += (now - *last).as_secs_f64();
        *last = now;
    }
    Ok(())
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
                crate::model::GpuTensor::Quant { qtype, .. } =>
                    *qtype == crate::QT_Q4_0 || *qtype == crate::QT_Q6_K
                    || *qtype == crate::QT_F8_E4M3
                    // BLOCK-128 FP8-ST (lane/rp-on-st, 2026-08-06): admitted now that the class
                    // has a b16 batched kernel (`qmatvec_e4m3_blk_mmvq_b16`), bit-identical per
                    // (token,row) to its m=1 launch. Before that kernel existed this class fell to
                    // the grid.y=m form at every width — still EXACT, so the tier's correctness
                    // bar was met, but it re-read the weight m times, which is why admitting it
                    // without the kernel would have been a throughput trap rather than a win.
                    || *qtype == crate::QT_F8_E4M3_BLK
                    // NVFP4 (lane/rp-on-st, 2026-08-06) — THE blocker this lane measured. The
                    // mixed FP8-ST 27B is 193 NVFP4 dense-MLP tensors, and this predicate is an
                    // ALL over every matmul, so NVFP4's missing b16 refused the whole checkpoint
                    // (`B=16 > cap 8 with no exact tier ... refused`) even with both e4m3 classes
                    // admitted. It now has base + _rp b16 twins off its existing batched template
                    // (bit-identical per (token,row) to the m=1 mmvq: same nibble decode, dp4a
                    // order, ue4m3 scale, warp reduce). This also opens the tier for pure-NVFP4
                    // GGUF models, which is a behavior change on the primary format — hence the
                    // full decode-batch config+strict battery on both.
                    || *qtype == crate::QT_NVFP4
                    // Q4_K (lane/rp-on-st): named by MEMRA_EXACT16_WHY as the 9B NVFP4 GGUF's
                    // refusing class (`L0.wqkv qtype=1`) — mixed NVFP4 checkpoints keep Q4_K
                    // attention. Now has base + _rp b16.
                    || *qtype == crate::QT_Q4_K
                    // Q5_K (lane/rp-on-st): the FOURTH class the diagnostic named on the same 9B
                    // GGUF (`L0.wqkv_gate qtype=3`). A shipped mixed checkpoint spreads ~500
                    // matmuls over four/five classes, and this predicate is an ALL — so chunk 16
                    // was unreachable for every real artifact until every class had a b16.
                    || *qtype == crate::QT_Q5_K
                    // Q8_0 NO LONGER requires the mirror (rp4): it has a base b16 too, so the
                    // tier is reachable at zero VRAM. Named by the diagnostic as the FP8-ST
                    // refusal — `L0.ssm_beta qtype=0 rp4=false`, a 23.9 MiB residual class that
                    // was gating chunk 16 for a 16.4 GiB checkpoint.
                    || *qtype == crate::QT_Q8_0,
                _ => false,
            }
        }
        // WHY-NOT DIAGNOSTIC (lane/rp-on-st, 2026-08-06): this predicate is a bare bool over
        // ~500 tensors, so a refusal produced only `B=16 > cap 8 with no exact tier ... refused`
        // with no way to tell WHICH class refused. That cost this lane two wrong hypotheses (the
        // rp mirror, then e4m3-only) before the NVFP4 gap was found. MEMRA_EXACT16_WHY=1 names
        // the first refusing tensor + its qtype. Diagnostic-only per flags doctrine; default off,
        // zero cost when unread.
        let why = std::env::var("MEMRA_EXACT16_WHY").is_ok();
        macro_rules! chk {
            ($t:expr, $label:expr) => {{
                let r = ok($t);
                if !r && why {
                    // qtype = -1 means the tensor is NOT Quant at all (a float/BF16/F16
                    // container), which the tier can never admit — a distinct diagnosis from
                    // "quantized, but in a class with no b16 kernel".
                    let (qt, rp4) = match $t {
                        crate::model::GpuTensor::Quant { qtype, rp4, .. } => (*qtype, rp4.is_some()),
                        _ => (-1, false),
                    };
                    eprintln!("[exact16] REFUSED by {} qtype={qt} rp4={rp4}", $label);
                }
                r
            }};
        }
        if self.cfg.m3.is_some() || self.is_gemma4_e4b() || self.cfg.gemma4.is_some() {
            if why { eprintln!("[exact16] REFUSED by architecture (m3/gemma4)"); }
            return false;
        }
        self.layers.iter().enumerate().all(|(li, l)| {
            let mix_ok = match &l.mixer {
                Mixer::Full(fa) => chk!(&fa.wq, format!("L{li}.wq")) && chk!(&fa.wk, format!("L{li}.wk"))
                    && chk!(&fa.wv, format!("L{li}.wv")) && chk!(&fa.wo, format!("L{li}.wo")),
                Mixer::Linear(la) => chk!(&la.wqkv, format!("L{li}.wqkv"))
                    && chk!(&la.wqkv_gate, format!("L{li}.wqkv_gate"))
                    && chk!(&la.ssm_beta, format!("L{li}.ssm_beta"))
                    && chk!(&la.ssm_alpha, format!("L{li}.ssm_alpha"))
                    && chk!(&la.ssm_out, format!("L{li}.ssm_out")),
                // MLA rides its own increment-4 arm; never admitted to the exact-16 tier here.
                Mixer::Mla(_) => { if why { eprintln!("[exact16] REFUSED by L{li} MLA mixer"); } false }
            };
            let ffn_ok = match &l.ffn {
                crate::hybrid::Ffn::Dense { ffn_gate, ffn_up, ffn_down } =>
                    chk!(ffn_gate, format!("L{li}.ffn_gate")) && chk!(ffn_up, format!("L{li}.ffn_up"))
                    && chk!(ffn_down, format!("L{li}.ffn_down")),
                crate::hybrid::Ffn::Moe(_) => { if why { eprintln!("[exact16] REFUSED by L{li} MoE ffn"); } false }
            };
            mix_ok && ffn_ok
        }) && chk!(&self.output, "output".to_string())
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
        // ---- PP DOOR: THE BATCHED STAGE SPLIT (pp2-batch 2026-08-06) ----------------------
        // Until this increment this body had NO pp arm: it walked lo=0..n_layers on the
        // primary engine's stream, with no stage split, no boundary, and no `rt.enter()`. With
        // the door open and a sharded cross-device placement, every projection for the remote
        // stages' layers was read over PCIe, per step, silently — measured 7.4 vs 208.9 tok/s
        // at B=1 (28x), 47.4 vs 657.0 at B=8 (13.9x) on a PRO 6000 pair over Gen5 x16 P2P.
        // Nothing failed or warned, because peer reads return identical bytes and all three
        // `decode-batch-gate` gates PASS on that config — the failure mode was performance,
        // and a green exactness battery hid it. `pp2-hardening` made that regime FAIL CLOSED
        // (research/pp2-hardening-20260806); this lane makes it legitimately split, so the
        // refusal lifts for the batched path.
        //
        // `decode_step_batch_ppn` runs each stage's layer range through that stage's engine
        // and stream with a [B, n_embd] boundary transfer between them, i.e. every stage
        // touches only LOCAL weights and LOCAL cache state. The refusal below still guards
        // the residue: the door open with `MEMRA_PP_STREAMS=0` (the same-stream rollback,
        // which also disables the sharded loader, so nothing is remote — `pp_shard_off` and
        // `pp2_streams_off` both make `pp_sharded_cross_device()` false) or a placement whose
        // PpNRt fails to build. Keeping the call means a future path that reaches here in a
        // remote regime still refuses instead of regressing 28x.
        if let Some(fence) = crate::pp::pp_cuts(self.layers.len()) {
            if !crate::pp::pp2_streams_off() && crate::pp::batch_pp_on() {
                return self.decode_step_batch_ppn(
                    e, tokens, caches, samp, masks, lean, &fence,
                );
            }
        }
        crate::pp::refuse_unsplit_if_remote(
            "decode_step_batch",
            "drop MEMRA_PP_STREAMS=0 / MEMRA_BATCH_PP=0 so the batched path takes its OWN \
             stage split (decode_step_batch_ppn), or serve single-stream over the eager pp \
             arm (decode_step_h), which is also split",
        )?;
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
            "decode_step_batch: B={b_n} > cap {cap} with no exact tier — refused. Either \
             B>16 (there is NO exact kernel class above 16: m>16 crosses GEMM/dp4a numeric \
             configs; the serve scheduler chunks wider concurrency into <=16 groups instead), \
             or some matmul in this checkpoint has no bit-exact b16 kernel — run with \
             MEMRA_EXACT16_WHY=1 to see which tensor and qtype refuses"
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
        let n_embd = self.cfg.n_embd as usize;
        let eps = self.cfg.rms_eps;

        // MEMRA_BATCH_PHASE=1: sync-bounded phase accumulation (diagnostics — see header note).
        // Initialized BEFORE the tick-input assembly below so slot 0 covers the HOST side of
        // setup (pos_v/ptr-table builds, embed gather) as well as the H2D sync — the audit-fix
        // lane's Q6 instrumentation gap (research/audit-fixes2-20260805): the old placement
        // started the clock after the assembly, so slot 0 under-reported setup.
        let mut ph_last = std::time::Instant::now();

        // Per-row rope positions (each sequence at its own depth).
        let pos_v: Vec<i32> = caches.iter().map(|c| c.pos as i32).collect();
        let pos_d = e.htod_i32(&pos_v)?;

        // Per-step, whole-trunk layer context: state pointer table + arm picks. Under a pp
        // split this call is made once PER STAGE with that stage's engine and range instead
        // (see `batch_layer_ctx`'s doc for why the table cannot be shared across devices).
        let n_layers = self.layers.len();
        let ctx = self.batch_layer_ctx(e, caches, 0, n_layers)?;

        // Embed all B tokens -> x [B, n_embd] (host gather, one H2D).
        let x = e.htod(&self.embd.gather(n_embd, tokens))?;
        ph_mark(e, 0, &mut ph_last)?;

        let x = self.decode_batch_layers(e, x, caches, &ctx, &pos_d, &mut ph_last)?;

        // ---- output norm + lm_head at m=B, one D2H ----
        let mut hn = e.uninit(b_n * n_embd)?;
        e.rms_norm(&x, self.output_norm.float_data(), &mut hn, n_embd, b_n, eps)?;
        let logits = e.matmul(&self.output, &hn, b_n)?;
        ph_mark(e, 10, &mut ph_last)?;

        self.decode_batch_epilogue(e, caches, samp, masks, lean, logits, b_n, &mut ph_last)
    }

    /// THE BATCHED PP-N STEP (pp2-batch increment 2, 2026-08-06): the batched tick split
    /// across `fence.len()-1` stages, each stage running ONLY its own layer range through
    /// ITS OWN engine and stream, with a `[B, n_embd]` boundary activation between them.
    /// The batched twin of `decode_step_h_ppn`, and the #1 item on the PP-2 serving bill —
    /// without it a >VRAM SKU (Step-3.7-Flash: 105 GB, fits only across two cards) serves
    /// SINGLE-STREAM only, because the batched path was the one loop with no stage split.
    ///
    /// STRUCTURE (mirrors the eager arm exactly, so the two stay comparable):
    ///   stage 0        `rt.enter(0)` -> per-stage pos_d + embed -> range -> `rt.tx`
    ///   middle stages  `rt.rx` -> per-stage pos_d -> range -> `rt.tx`
    ///   last stage     `rt.rx` -> per-stage pos_d -> range -> output_norm + lm_head ->
    ///                  the batched serving epilogue (masks, device sample, lean park)
    ///
    /// FOUR THINGS ARE PER-STAGE, and each is per-stage for a measured reason:
    ///
    /// 1. THE ENGINE (`rt.engine(s, e)`). Not just for the remote device: `Engine` owns
    ///    lazily-grown stable-pointer scratch pools (`fa_part_pool`, `fa_vf16_scratch`,
    ///    `argmax_partials`) that are single-stream-safe BY DESIGN. Two stage streams
    ///    through one Engine is the shared-scratch race the pp2 lane hit (2026-08-02
    ///    nondeterministic all-logits divergence, 35% flake). `PpNRt::build` already gives
    ///    every stage s>0 its own Engine even on the primary device, so honouring
    ///    `rt.engine(s, e)` here is what scopes the pools per stage — the batched path
    ///    allocates MORE of that scratch than the eager one (fa at m=B), so this is the
    ///    load-bearing half of the trap's mitigation, not an inherited nicety.
    ///
    /// 2. THE POINTER TABLE (`batch_layer_ctx(es, caches, lo, hi)`). See [`BatchLayerCtx`]:
    ///    it holds DEVICE ADDRESSES of that range's cache state, uploaded through that
    ///    stage's engine. One step-wide table on the primary would put every stage's kernel
    ///    arguments in stage-0's HBM — a peer read per pointer fetch, the exact cliff this
    ///    whole lane exists to remove.
    ///
    /// 3. `pos_d` (the M2 pipelining law, learned on the eager arm): each stage uploads its
    ///    own copy of the step's per-row positions on ITS stream, so the buffer is
    ///    allocated, consumed and freed on one stream. A shared stage-0 `pos_d` freed at fn
    ///    return breaks under deferred readback — the free enqueues on stream 0 while later
    ///    stages still dereference it.
    ///
    /// 4. THE HEAD + EPILOGUE run on the LAST stage: `output_norm`/`output` were uploaded
    ///    through the last stage's engine by the sharded loader (`hybrid.rs`: `e_head =
    ///    layer_engine(e, n_trunk, n_trunk-1)`), and `cache.last_logits_dev` must be
    ///    allocated where the logits are.
    ///
    /// EXACTNESS: PP-N adds ZERO deviation. Each stage runs the SAME kernels on the SAME
    /// bytes in the same order — the split only moves where the residual is materialized,
    /// and the boundary is a straight f32 copy (dtod same-device / `cudaMemcpyPeerAsync`
    /// cross-device, no conversion). So batched PP-N must be BIT-IDENTICAL to single-device
    /// batched at the same B, in both placement orders. Gate: `decode-batch-gate --mode
    /// pp` (logit-dump, both orders) — the batched analogue of the eager arm's 48 steps x
    /// 248,320 f32 logits with zero differing bits.
    ///
    /// The B=1 fast path is NOT taken here (its condition already excludes an open door):
    /// it routes through `decode_layers_eager` whole-trunk on one engine, which is exactly
    /// the unsplit walk. B=1 under the door rides this function's B=1 case instead — the
    /// same trade the eager arm's own ppn step makes, and the reason the pp2 lane measured
    /// B=1 door-open at 0.854x (the lost fusion chain), not a cliff.
    #[allow(clippy::too_many_arguments)]
    fn decode_step_batch_ppn(
        &self,
        e: &Engine,
        tokens: &[u32],
        caches: &mut [&mut Cache],
        samp: &[Option<(f32, u64, u32)>],
        masks: &[Option<(&CudaSlice<u32>, usize)>],
        lean: bool,
        fence: &[usize],
    ) -> Result<(Vec<Vec<f32>>, Vec<Option<u32>>), Box<dyn std::error::Error>> {
        let b_n = tokens.len();
        assert!(b_n >= 1 && b_n == caches.len(), "tokens/caches length mismatch");
        assert!(
            !self.is_gemma4_e4b() && self.cfg.gemma4.is_none(),
            "decode_step_batch_ppn covers the hybrid non-gemma4 trunk only"
        );
        // Same width policy as the unsplit body — the stage split changes WHERE kernels run,
        // never WHICH tier admits the width. Duplicated deliberately rather than hoisted:
        // the exact-16 scope must wrap the whole multi-stage walk (`set_verify_exact` is
        // per-Engine state read at dispatch on every stage), so it has to be established
        // here, and a shared helper returning a guard would have to own `e` plus the flag.
        let cap = Self::decode_batch_cap();
        let exact16 = b_n > 8 && b_n <= 16 && self.decode_batch_exact16_ok();
        assert!(
            b_n <= cap || exact16,
            "decode_step_batch_ppn: B={b_n} > cap {cap} with no exact tier — refused"
        );
        let rt = crate::pp::PpNRt::get(e)?;
        let n_st = fence.len() - 1;
        assert_eq!(
            rt.n_stages(), n_st,
            "PpNRt stage count {} != fence stages {n_st}", rt.n_stages()
        );
        let n_embd = self.cfg.n_embd as usize;
        let eps = self.cfg.rms_eps;
        let payload = b_n * n_embd;

        // EXACT-16 SCOPE, PER STAGE ENGINE: `verify_exact` is per-Engine state (an AtomicBool
        // on the Engine the dispatch reads), and each stage runs through a DIFFERENT Engine —
        // so setting it on the primary alone would leave stages 1..N-1 dispatching the m>=16
        // GEMM/MMQ arms while stage 0 used the exact b16 tier. That is a silent per-stage
        // numeric split (the failure this tier exists to prevent), so the flag is set on
        // every stage engine and cleared on all of them at scope exit.
        struct ExactScopeN<'a>(Vec<&'a Engine>);
        impl Drop for ExactScopeN<'_> {
            fn drop(&mut self) {
                for eng in &self.0 {
                    eng.set_verify_exact(false);
                }
            }
        }
        let _exact_scope = if exact16 {
            let engines: Vec<&Engine> = (0..n_st).map(|s| rt.engine(s, e)).collect();
            for eng in &engines {
                eng.set_verify_exact(true);
            }
            Some(ExactScopeN(engines))
        } else {
            None
        };

        let mut ph_last = std::time::Instant::now();

        // B=1 PER-STAGE FAST PATH (measured 2026-08-06, PRO 6000 pair). The unsplit body's
        // b1_fast guard includes `pp_cuts().is_none()`, so opening the pp door dropped every
        // solo session off the m=1 FUSION chain (cross-layer add+norm+q8_1, fused SwiGLU,
        // lever 1's gate+up dual) and onto the batched m=1 walk. Cost, arm A vs arm C at B=1:
        // 208.5 vs 177.3 tok/s = -15.0% — and NOT a split cost, since arm B (stages=2 on ONE
        // card) pays the same 177, and the prior lane's `MEMRA_PP_SHARD=0` batched-body B=1
        // was 178.5. It was the fusion chain going missing, on the config the Step SKU serves
        // solo requests from.
        //
        // `decode_layers_eager(lo, hi)` is ALREADY range-scoped and is exactly what the eager
        // ppn arm (`decode_step_h_ppn`) calls per stage, so B=1 rides the same per-stage
        // structure: same engines, same streams, same [1, n_embd] boundary slots, same
        // stage-owned caches. Only the trunk kernels differ, and they differ identically to
        // how they differ off-door. Exactness is therefore the SAME accepted decode-config FP
        // class the unsplit b1_fast lever already carries (strict gate1 PASSes with it on,
        // FAILs with it off at maxdiff 1.591e-1) — which is why the pp gate pins
        // `set_b1_fast(false)`: with it on, the B=1 reference and the split arm would
        // legitimately sit on opposite sides of that gap and the bit-identity arm would
        // report a fake stage-split failure. Gates that DO cover this: run-gen argmax MATCH
        // and serve-smoke greedy-determinism over the split.
        let b1_stage_fast = b_n == 1
            && Self::b1_fast_on()
            && !self.is_gemma4_e4b()
            && self.cfg.gemma4.is_none()
            && self.cfg.m3.is_none()
            && !e.verify_exact_on();
        // Hoisted: `caches[0].pos` as a value argument alongside `caches[0]` as `&mut` in one
        // call is a borrow conflict; `pos` is Copy and the epilogue is what advances it.
        let pos0 = if b1_stage_fast { caches[0].pos } else { 0 };

        // ---- STAGE 0: embed (the table lives with stage 0) + layers [0, fence[1]) + TX ----
        let mut slot = {
            let _st0 = rt.enter(0);
            let e0 = rt.engine(0, e);
            let pos_v: Vec<i32> = caches.iter().map(|c| c.pos as i32).collect();
            let pos_d = e0.htod_i32(&pos_v)?;
            let x = e0.htod(&self.embd.gather(n_embd, tokens))?;
            ph_mark(e0, 0, &mut ph_last)?;
            let x = if b1_stage_fast {
                self.decode_layers_eager(e0, x, fence[0], fence[1], &pos_d, pos0, caches[0])?
            } else {
                let ctx = self.batch_layer_ctx(e0, caches, fence[0], fence[1])?;
                self.decode_batch_layers(e0, x, caches, &ctx, &pos_d, &mut ph_last)?
            };
            rt.tx(0, &x, payload)?
            // x + pos_d + ctx.ptr_table drop here: freed stream-ordered on stage-0's stream.
        };

        // ---- MIDDLE STAGES: RX boundary s-1 -> range -> TX boundary s ----
        for s in 1..n_st - 1 {
            let _st = rt.enter(s);
            let es = rt.engine(s, e);
            let pos_v: Vec<i32> = caches.iter().map(|c| c.pos as i32).collect();
            let pos_d = es.htod_i32(&pos_v)?;
            let x = rt.rx(s - 1, slot, payload)?;
            let x = if b1_stage_fast {
                self.decode_layers_eager(es, x, fence[s], fence[s + 1], &pos_d, pos0, caches[0])?
            } else {
                let ctx = self.batch_layer_ctx(es, caches, fence[s], fence[s + 1])?;
                self.decode_batch_layers(es, x, caches, &ctx, &pos_d, &mut ph_last)?
            };
            slot = rt.tx(s, &x, payload)?;
        }

        // ---- LAST STAGE: RX + final range + head + the batched serving epilogue ----
        let _stl = rt.enter(n_st - 1);
        let el = rt.engine(n_st - 1, e);
        let pos_v: Vec<i32> = caches.iter().map(|c| c.pos as i32).collect();
        let pos_d = el.htod_i32(&pos_v)?;
        let x = rt.rx(n_st - 2, slot, payload)?;
        let x = if b1_stage_fast {
            self.decode_layers_eager(el, x, fence[n_st - 1], fence[n_st], &pos_d, pos0, caches[0])?
        } else {
            let ctx = self.batch_layer_ctx(el, caches, fence[n_st - 1], fence[n_st])?;
            self.decode_batch_layers(el, x, caches, &ctx, &pos_d, &mut ph_last)?
        };

        let mut hn = el.uninit(payload)?;
        el.rms_norm(&x, self.output_norm.float_data(), &mut hn, n_embd, b_n, eps)?;
        let logits = el.matmul(&self.output, &hn, b_n)?;
        ph_mark(el, 10, &mut ph_last)?;

        self.decode_batch_epilogue(el, caches, samp, masks, lean, logits, b_n, &mut ph_last)
    }

    /// Build the per-step layer context for layers `[lo, hi)`: the device state-pointer
    /// table plus the step's arm picks. See [`BatchLayerCtx`] for why this is RANGE-scoped
    /// (the table holds device addresses and must be uploaded through the engine whose
    /// device runs those layers).
    ///
    /// Table layout is unchanged from the whole-trunk version — `lin_base`/`attn_base` are
    /// still indexed by ABSOLUTE layer id, so `decode_batch_layers`' body indexes them
    /// exactly as the old inline loop did. Only layers in `[lo, hi)` contribute entries; the
    /// rest stay `None`, which is a loud `expect` if a range ever reads outside its own.
    pub(crate) fn batch_layer_ctx(
        &self,
        e: &Engine,
        caches: &[&mut Cache],
        lo: usize,
        hi: usize,
    ) -> Result<BatchLayerCtx, Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let head_dim = cfg.head_dim_k as usize;
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
            for il in lo..hi {
                match &self.layers[il].mixer {
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
        //
        // The picks are t_kv-driven, and t_kv is layer-INVARIANT within a step, so every
        // stage of a pp split independently computes the SAME arms from the same `caches`
        // — a stage cannot silently take a different program than its unsplit self.
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

        Ok(BatchLayerCtx {
            lin_base,
            attn_base,
            ptr_table,
            t_kvs,
            t_kv_max,
            sp0,
            seqs_append,
            seqs_fa,
            lo,
            hi,
        })
    }

    /// THE PP SEAM (pp2-batch increment 1, 2026-08-06): run the batched trunk over layers
    /// `[ctx.lo, ctx.hi)`, entering with a materialized `[B, n_embd]` residual and exiting
    /// with the range's final residual materialized. The batched twin of
    /// `decode_layers_eager` — the eager arm has had this seam since M1-PP2 and every ppN
    /// stage calls it; the batched body had no equivalent, which is why every later PP-2
    /// increment (and spec-over-PP2, whose verify is a batched T=K+1 forward) waited on this
    /// extraction (`research/pp2-hardening-20260806/PROGRESS.md` bill item 1).
    ///
    /// SINGLE-DEVICE SEMANTICS ARE UNCHANGED BY CONSTRUCTION: the body is the old
    /// `for (il, layer) in self.layers.iter().enumerate()` loop moved verbatim, with `for il
    /// in ctx.lo..ctx.hi` as the header and the per-step invariants (`ptr_table`, arm picks,
    /// `t_kv`) read from `ctx` instead of enclosing locals. At `lo=0, hi=n_layers` — every
    /// call today — the launch sequence is identical, so the exactness contract in this
    /// module's header carries over untouched rather than needing a re-proof.
    ///
    /// UNLIKE the eager seam, this one is NOT yet stage-callable: `caches` is `&mut [&mut
    /// Cache]` mutated in place (KV `len` bumps, ssm ping-pong swaps), and `pos_d`/`x` come
    /// from the caller's device. Wiring a stage split means per-stage `pos_d` + a boundary
    /// `[B, n_embd]` transfer around this call, which is the NEXT increment. The seam exists
    /// so that increment is a call-site change, not a 250-line surgery.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn decode_batch_layers(
        &self,
        e: &Engine,
        mut x: CudaSlice<f32>,
        caches: &mut [&mut Cache],
        ctx: &BatchLayerCtx,
        pos_d: &CudaSlice<i32>,
        ph_last: &mut std::time::Instant,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let b_n = caches.len();
        let cfg = &self.cfg;
        let n_embd = cfg.n_embd as usize;
        let eps = cfg.rms_eps;
        let n_head = cfg.n_head as usize;
        let n_head_kv = cfg.n_head_kv as usize;
        let head_dim = cfg.head_dim_k as usize;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let rope_dims = cfg.rope_dim_count as usize;
        let (lin_base, attn_base) = (&ctx.lin_base, &ctx.attn_base);
        let ptr_table = &ctx.ptr_table;
        let (seqs_append, seqs_fa, sp0, t_kv_max) =
            (ctx.seqs_append, ctx.seqs_fa, ctx.sp0, ctx.t_kv_max);
        debug_assert_eq!(ctx.t_kvs.len(), b_n, "ctx built for a different batch width");

        for il in ctx.lo..ctx.hi {
            let layer = &self.layers[il];
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
                    ph_mark(e, 1, ph_last)?;

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
                    ph_mark(e, 2, ph_last)?;
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
                        ph_mark(e, 4, ph_last)?;
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
                            ph_mark(e, 3, ph_last)?;
                            let mut a_row = e.uninit(q_dim)?;
                            e.fa_decode_kvmod(
                                &q_row, &k_view, &v_view, &mut a_row, head_dim, n_head, n_head_kv,
                                t_kv, scale, kvl.k_tok_bytes, kvl.v_tok_bytes, Engine::kv_fp8_on(),
                            )?;
                            ph_mark(e, 4, ph_last)?;
                            e.dtod_copy_into(&a_row, &mut attn, bi * q_dim)?;
                            ph_mark(e, 3, ph_last)?;
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
                    ph_mark(e, 5, ph_last)?;
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
                    ph_mark(e, 6, ph_last)?;

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
                    ph_mark(e, 7, ph_last)?;

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
                    ph_mark(e, 8, ph_last)?;
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
            ph_mark(e, 9, ph_last)?;
        }
        Ok(x)
    }

    /// The batched tick's TAIL, after the trunk: grammar masks -> device sampling -> lean
    /// logits park -> `pos` bump. Split out with the pp seam (`decode_batch_layers`) because
    /// under a stage split this runs on the LAST stage's engine and device — the lm_head, the
    /// masks, the sampler, and `cache.last_logits_dev` all live where the final residual
    /// lands, and the caller must be able to place them there without duplicating 90 lines of
    /// serving contract. `logits` is `[b_n, n_vocab]` already computed by the caller (the
    /// output_norm + lm_head pair stays at the call site so a stage split can fence around
    /// it); everything after it is here, verbatim.
    #[allow(clippy::too_many_arguments)]
    fn decode_batch_epilogue(
        &self,
        e: &Engine,
        caches: &mut [&mut Cache],
        samp: &[Option<(f32, u64, u32)>],
        masks: &[Option<(&CudaSlice<u32>, usize)>],
        lean: bool,
        logits: CudaSlice<f32>,
        b_n: usize,
        ph_last: &mut std::time::Instant,
    ) -> Result<(Vec<Vec<f32>>, Vec<Option<u32>>), Box<dyn std::error::Error>> {
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
        ph_mark(e, 11, ph_last)?;
        Ok((rows, next))
    }
}
