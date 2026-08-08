//! Hybrid forward pass (Stage-1, f32, prefill, single sequence). Per layer dispatches to a
//! linear-attention (Gated DeltaNet) or full-attention mixer, then SwiGLU FFN. Matches
//! llama.cpp src/models/qwen35.cpp node-for-node.

use cudarc::driver::CudaSlice;
use memra_gguf::config::ModelConfig;
use crate::Engine;
use crate::cache::Cache;

/// Resident trunk transients for the eager prime (piecewise-graph foundation; see
/// HybridModel::prime_slabs). Every buffer is fully overwritten before use per prime.
pub struct PrimeSlabs {
    pub t_cap: usize,
    pub h: CudaSlice<f32>,
    pub x1: CudaSlice<f32>,
    pub z: CudaSlice<f32>,
    pub act: CudaSlice<f32>,
    pub xa: CudaSlice<f32>,
    pub xb: CudaSlice<f32>,
    pub h16: CudaSlice<u8>,
    pub z16: CudaSlice<u8>,
    /// piecewise boundary slabs (increment 2): GEMM outputs land here so the
    /// downstream captured segments see fixed addresses.
    pub gate: CudaSlice<f32>,     // t * n_ff_max
    pub up: CudaSlice<f32>,       // t * n_ff_max
    pub ffn_out: CudaSlice<f32>,  // t * n_embd
    /// piecewise increment 3: per-layer S-glue segment graphs (down-add + next
    /// attn-norm, ALL-slab IO, zero in-graph allocations -> keeperless capture is
    /// clean). Baked at this t_cap; replay only when t == t_cap. seg_glue[il] fires
    /// between layer il and il+1 (ping-pong parity is deterministic per il).
    pub seg_glue: Vec<Option<cudarc::driver::CudaGraph>>,
    /// increment 5 (core-split edition): the mixer out-GEMM writes _into_ `mixed`
    /// directly (no staging copy — the increment-4 copy route was refuted), making
    /// S-mid [add + post-norm] all-slab and capturable.
    pub mixed: CudaSlice<f32>,
    pub seg_mid: Vec<Option<cudarc::driver::CudaGraph>>,
    pub seg_t: usize,
}

// Split prime ranges cannot enter the full-range segment-graph arm, and every slab access
// is serialized by its device mutex after binding that device's CUDA context on the thread.
unsafe impl Send for PrimeSlabs {}

fn empty_cache_layers<T>(n: usize) -> Vec<Option<T>> {
    std::iter::repeat_with(|| None).take(n).collect()
}

/// Temporarily move a PP-2 cache's layer state into two independently-owned cache shells.
/// The stage walkers then receive disjoint `&mut Cache` values and can run on separate host
/// threads without aliasing. GPU buffers are moved, not copied; Drop restores every layer
/// and publishes the last position completed by both stages.
struct PrimeCacheStages<'a> {
    parent: &'a mut Cache,
    cut: usize,
    stage0: Cache,
    stage1: Cache,
}

impl<'a> PrimeCacheStages<'a> {
    fn new(parent: &'a mut Cache, cut: usize) -> Self {
        let n = parent.kv.len();
        assert_eq!(parent.recur.len(), n, "cache layer vectors disagree");
        assert!(cut <= n, "PP-2 cache cut {cut} exceeds {n} layers");
        let mut kv0 = empty_cache_layers(n);
        let mut kv1 = empty_cache_layers(n);
        let mut recur0 = empty_cache_layers(n);
        let mut recur1 = empty_cache_layers(n);
        for i in 0..cut {
            kv0[i] = parent.kv[i].take();
            recur0[i] = parent.recur[i].take();
        }
        for i in cut..n {
            kv1[i] = parent.kv[i].take();
            recur1[i] = parent.recur[i].take();
        }
        let pos = parent.pos;
        let max_ctx = parent.max_ctx;
        Self {
            parent,
            cut,
            stage0: Cache {
                kv: kv0,
                recur: recur0,
                pos,
                max_ctx,
                last_logits_dev: None,
                dflash_taps: None,
            },
            stage1: Cache {
                kv: kv1,
                recur: recur1,
                pos,
                max_ctx,
                last_logits_dev: None,
                dflash_taps: None,
            },
        }
    }

    fn parts(&mut self) -> (&mut Cache, &mut Cache) {
        (&mut self.stage0, &mut self.stage1)
    }
}

impl Drop for PrimeCacheStages<'_> {
    fn drop(&mut self) {
        let n = self.parent.kv.len();
        for i in 0..n {
            let source = if i < self.cut {
                &mut self.stage0
            } else {
                &mut self.stage1
            };
            debug_assert!(self.parent.kv[i].is_none());
            debug_assert!(self.parent.recur[i].is_none());
            self.parent.kv[i] = source.kv[i].take();
            self.parent.recur[i] = source.recur[i].take();
        }
        self.parent.pos = self.stage0.pos.min(self.stage1.pos);
    }
}


/// task #18 (attn side): one sequence's pre-attention outputs (post-rope q/k, v, out-gate).
pub(crate) struct AttnPre {
    pub q: cudarc::driver::CudaSlice<f32>,
    pub k: cudarc::driver::CudaSlice<f32>,
    pub v: cudarc::driver::CudaSlice<f32>,
    pub gate: Option<cudarc::driver::CudaSlice<f32>>,
}

/// task #18: one sequence's GDN prep outputs (the scan inputs).
pub(crate) struct GdnPrep {
    pub hk: usize,
    pub q_l2: cudarc::driver::CudaSlice<f32>,
    pub k_l2: cudarc::driver::CudaSlice<f32>,
    pub v_g: cudarc::driver::CudaSlice<f32>,
    pub beta: cudarc::driver::CudaSlice<f32>,
    pub g_log: cudarc::driver::CudaSlice<f32>,
    pub kb16: Option<cudarc::driver::CudaSlice<u8>>,
    pub qb16: Option<cudarc::driver::CudaSlice<u8>>,
}

/// Device scratch for the burst verify stream (see `verify_stream_scratch`).
pub(crate) struct VerifyStreamScratch {
    pub pos_d: CudaSlice<i32>,
    pub row_ctrs: Vec<CudaSlice<i32>>,
}
use crate::hybrid::{HybridModel, Mixer, FullAttnLayer, LinearAttnLayer, MoeWeights};

struct MoeInputTraceWriter {
    dir: std::path::PathBuf,
    index: std::fs::File,
    payloads: std::collections::HashMap<u16, (std::fs::File, u64)>,
}

static MOE_INPUT_TRACE_WRITER: std::sync::OnceLock<
    std::sync::Mutex<Option<MoeInputTraceWriter>>,
> = std::sync::OnceLock::new();

/// STAGE-2 GROUPED DECODE gate (MEMRA_MOE_GDEC, default ON; `=0` restores the sequential
/// per-expert launch chain). See `moe_gdec_token`.
fn gdec_enabled() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| std::env::var("MEMRA_MOE_GDEC").map(|v| v != "0").unwrap_or(true))
}

/// SLAB-LOCAL RESIDENT ARM gate (lane/pp-leverb 2026-08-08, MEMRA_MOE_SLAB, default ON;
/// `=0` restores the SLRU dispatch even when resident slabs exist). Read PER CALL, never
/// memoized — probes A/B the two provenances in one process (the MEMRA_PRIME_PP pattern).
/// See `moe_ffn_sequential_zq8`'s slab_local arm: the sigmoid-router archs (step35/M3/Hy3)
/// are denied every `dev_exps` consumer (pairs/dev route softmax), so before this arm the
/// fits-VRAM resident slabs were UPLOADED for them but never READ — the SLRU kept staging
/// the same bytes beside a dead copy (37 GB H2D per pp4096 prime on the Step SKU, anatomy
/// receipt). The arm reads the SAME bytes through the SAME kernels; only the pointer
/// PROVENANCE changes (slab base + ex*stride vs SLRU slot address) — the bit-identity class
/// `moe_ffn_dev`'s resident arm already documents against its SLRU arm.
fn moe_slab_enabled() -> bool {
    std::env::var("MEMRA_MOE_SLAB").as_deref() != Ok("0")
}

/// Expert-grouped dispatch is the Step35 prefill default. The env remains a live per-call seam:
/// `=0` restores the sequential oracle, while any other explicit value preserves the historical
/// opt-in for non-Step35/non-prefill callers.
fn moe_grouped_enabled(cfg: &ModelConfig, prefill: bool) -> bool {
    std::env::var("MEMRA_MOE_GROUPED")
        .map(|value| value != "0")
        .unwrap_or(prefill && cfg.step35.is_some())
}

/// Deterministic in-token expert prefetch. `MEMRA_MOE_PREFETCH=1` overlaps memory-source H2D on the
/// copy stream; selecting the opt-in worker spill backend enables the same known-next hook for disk.
fn moe_prefetch_enabled() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| std::env::var("MEMRA_MOE_PREFETCH").as_deref() == Ok("1")
        || crate::spill_pread::worker_enabled())
}

/// Best-effort OS page-cache prefetch distance for mmap-backed expert ranges. Independent of the
/// H2D copy-stream experiment so storage->RAM and RAM->HBM overlap can be measured separately.
/// The opt-in default stays one expert to preserve the original experiment; spill rigs can widen
/// it with `MEMRA_MOE_PAGE_PREFETCH_WINDOW` to cover NVMe latency.
fn moe_page_prefetch_window() -> usize {
    static W: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *W.get_or_init(|| page_prefetch_window_from_values(
        std::env::var("MEMRA_MOE_PAGE_PREFETCH").as_deref() == Ok("1"),
        std::env::var("MEMRA_MOE_PAGE_PREFETCH_WINDOW").ok().as_deref(),
    ))
}

fn page_prefetch_window_from_values(enabled: bool, raw_window: Option<&str>) -> usize {
    if !enabled {
        return 0;
    }
    raw_window
        .and_then(|value| value.parse().ok())
        .unwrap_or(1)
}

/// Return only the newly exposed positions in a rolling lookahead window. Position zero seeds the
/// full window; each later position adds one expert at the far edge. Thus widening the window does
/// not repeatedly issue `MADV_WILLNEED` for the same range.
fn page_prefetch_positions(
    position: usize,
    len: usize,
    window: usize,
) -> std::ops::Range<usize> {
    if window == 0 || position >= len {
        return len..len;
    }
    let (start, count) = if position == 0 {
        (1, window)
    } else {
        (position.saturating_add(window), 1)
    };
    let start = start.min(len);
    start..start.saturating_add(count).min(len)
}

/// Grouped worker-I/O schedule: prime the first active expert before the loop, then queue exactly
/// one known-next expert at each iteration. Returning positions keeps expert ordering authoritative.
fn grouped_worker_prefetch_position(order_len: usize, current: Option<usize>) -> Option<usize> {
    let position = current.map_or(0, |position| position.saturating_add(1));
    (position < order_len).then_some(position)
}

/// Fill the worker ring with complete experts, retaining one pinned buffer for an unexpected
/// demand miss. Each expert has gate/up/down extents, so depth 16 admits a rolling five-expert
/// window. Position zero primes the current expert too: its three independent reads can run in
/// parallel instead of demand-serializing gate, up, and down before any useful GPU work exists.
fn worker_prefetch_window() -> usize {
    static WINDOW: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *WINDOW.get_or_init(|| {
        let automatic = crate::spill_pread::configured_depth().saturating_sub(1) / 3;
        std::env::var("MEMRA_SPILL_WORKER_EXPERT_WINDOW")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(automatic.max(1))
    })
}

/// Return only positions newly exposed by a rolling worker-I/O window. Unlike mmap page advice,
/// this includes the current expert when the window is seeded so all three current projections
/// enter the CPU pool together.
fn worker_prefetch_positions(position: usize, len: usize, window: usize) -> std::ops::Range<usize> {
    if window == 0 || position >= len {
        return len..len;
    }
    let (start, count) = if position == 0 {
        (0, window)
    } else {
        (position.saturating_add(window).saturating_sub(1), 1)
    };
    let start = start.min(len);
    start..start.saturating_add(count).min(len)
}

/// LAUNCH-STRUCTURE STAGE 3 gate (MEMRA_MOE_DEV, default ON; `=0` restores host routing). The
/// zero-DtoH device-dispatch path for fully-resident layers: router top-k output stays on device,
/// expert weight pointers come from the per-layer device table. Requires the fused router (the
/// dev path consumes the device sel/w directly), so MEMRA_FUSED_ROUTER=0 also disables it.
fn moe_dev_enabled() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| std::env::var("MEMRA_MOE_DEV").map(|v| v != "0").unwrap_or(true)
        && !matches!(std::env::var("MEMRA_FUSED_ROUTER").as_deref(), Ok("0")))
}

/// MoE EXPERT dp4a gate (MEMRA_MOE_Q8, default ON; `=0` restores the Stage-A f32-dequant expert
/// kernels). Applies when gate/up/down expert qtypes are all in the dp4a body set (IQ3_S/IQ4_XS).
/// FP-order differs from Stage-A (int dp4a + warp tree) — argmax/run-gen/stream-identity gates
/// arbitrate; the sequential and fused q8 paths ship as a matched pair (MEMRA_MOE_GATE contract).
fn moe_q8_enabled() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| std::env::var("MEMRA_MOE_Q8").map(|v| v != "0").unwrap_or(true))
}

/// gemma4 fast-arm gate: qtypes with an `expert_dot_g` dp4a body (superset used by the gelu
/// dev arm; the qwen q8 arms keep their own battery-gated q8_expert_supported policy).
fn expert_dp4a_supported(qt: i32) -> bool {
    qt == crate::QT_Q4_0 || qt == crate::QT_IQ3_S || qt == crate::QT_IQ4_XS
        || qt == crate::QT_Q3_K || qt == crate::QT_Q4_K || qt == crate::QT_Q6_K
}

fn q8_expert_supported(qt: i32) -> bool {
    // k-quant arms added 2026-07-06 (Q3_K/Q4_K/Q6_K bodies for the UD tail layers). Briefly
    // default-excluded the same day when they appeared to break 35B real-prompt spec — the
    // ACTUAL culprit was the MoE router's cuBLASLt n-dependence (d994271); with the router
    // decode-exact at verify t, the k-quant arms pass the full spec battery (p1/p2/p3 + raw
    // K=1..8) and are DEFAULT ON again (+9 tok/s: 148.9 -> 157.9). MEMRA_MOE_Q8_KQ=0 excludes.
    static KQ: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let kq = *KQ.get_or_init(|| {
        std::env::var("MEMRA_MOE_Q8_KQ").map(|v| v != "0").unwrap_or(true)
    });
    // NVFP4 experts: DEFAULT ON (2026-07-17). The M3-era "decode-vs-verify MISMATCH 3.4e1"
    // that had this excluded was the missing per-expert macro-scale fold, fixed in the
    // dev-kernel epilogues + moe_w_scale_by_expert; the 35B ct-NVFP4 artifact now runs the
    // q8 arm at parity with the IQ4_XS daily (174-178 tok/s, spec K=1..8 exact). M3/Hy3
    // never reach the q8 arms regardless (sigmoid-router cfg gates on pairs/dev/gdec).
    // MEMRA_MOE_Q8_NVFP4=0 restores the f32 arm.
    let nvfp4_q8 = std::env::var("MEMRA_MOE_Q8_NVFP4").map(|v| v != "0").unwrap_or(true);
    qt == crate::QT_IQ3_S || qt == crate::QT_IQ4_XS || (nvfp4_q8 && qt == crate::QT_NVFP4)
        || (kq && (qt == crate::QT_Q3_K || qt == crate::QT_Q4_K || qt == crate::QT_Q6_K))
}

/// The decode-once (_dec) and IQ-MMA expert kernels dequant via IQ-specific extractors —
/// k-quant tensors must fall to the _em dot path instead.
fn q8_expert_dec_supported(qt: i32) -> bool {
    qt == crate::QT_IQ3_S || qt == crate::QT_IQ4_XS || qt == crate::QT_Q4_0
}

/// Grouped-f16 door (MEMRA_MOE_F16G) per-projection admission: the qtype has a dequant-to-f16
/// kernel in cu/moe_f16_grouped.cu AND the projection's k dimension tiles its block size.
/// Round 49 widened coverage to q35's UD mix (gate/up IQ3_S x39 + Q3_K x1 + IQ4_XS x1; down
/// IQ4_XS x37 + Q6_K x3 + Q4_K x1) — the round-47 IQ4_XS/Q4_0-only table admitted ~1 of 41
/// q35 layers, which is why that cell measured FLAT.
fn f16g_proj_ok(qt: i32, in_f: usize) -> bool {
    match qt {
        crate::QT_Q4_0 => in_f % 32 == 0,
        crate::QT_IQ4_XS | crate::QT_IQ3_S | crate::QT_Q3_K | crate::QT_Q4_K
        | crate::QT_Q6_K => in_f % 256 == 0,
        _ => false,
    }
}

/// STAGE 3 prewarm gate (MEMRA_MOE_PREWARM, default ON; `=0` leaves residency organic). One-shot
/// per layer: force-admit every block while FREE slots cover the whole layer (never evicts).
fn moe_prewarm_enabled() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| std::env::var("MEMRA_MOE_PREWARM").map(|v| v != "0").unwrap_or(true))
}

/// During a discarded fixed-residency profile, admit CPU-routed misses after their current-token
/// CPU result is complete. The current result and numeric path are unchanged; later warmup tokens
/// can then vote for and exercise those experts on GPU before the cache is frozen.
fn cpu_expert_profile_admit_enabled() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| std::env::var("MEMRA_CPU_EXPERT_FREEZE_PROFILE_ADMIT").as_deref() == Ok("1"))
}

/// Minimum prompt length for the BATCHED cache prime (`prime_cache`). Below this the tokenwise
/// decode loop wins anyway (the batched path's GEMM dispatch needs m>=16, and the stateful conv
/// kernel needs T >= d_conv-1). Callers: generate / generate_spec.
pub const PRIME_MIN_T: usize = 16;
const PRIME_PIPE_MICROBATCHES: usize = 8;
const PRIME_PIPE_MIN_CHUNK: usize = 128;

/// Effective internal prime chunk. An explicit MEMRA_PRIME_CHUNK is authoritative.
/// Naked PP-2 primes use the measured pipeline geometry: up to eight microchunks, never
/// below 128 tokens, while the legacy 4096-token cap remains the long-context bound.
pub fn prime_chunk_tokens(t: usize, n_layers: usize) -> usize {
    if let Ok(value) = std::env::var("MEMRA_PRIME_CHUNK") {
        return value.parse().unwrap_or(4096);
    }
    let chunk = 4096usize;
    let pp2 = crate::pp::prime_pp_on()
        && !crate::pp::pp2_streams_off()
        && crate::pp::pp_cuts(n_layers).is_some_and(|cuts| cuts.len() == 3);
    if pp2 && t >= 2 * PRIME_PIPE_MIN_CHUNK {
        chunk.min(
            t.div_ceil(PRIME_PIPE_MICROBATCHES)
                .max(PRIME_PIPE_MIN_CHUNK),
        )
    } else {
        chunk
    }
}

impl HybridModel {
    /// CHUNK-INVARIANCE BISECT SEAM (lane/chunk-invariance): `MEMRA_PRIME_TRACE=<path>`
    /// appends one JSONL row per (chunk, layer) with a hash of that layer's last-row
    /// post-residual hidden. Diagnostic only — never on in a measured or gated run
    /// (it forces a dtoh + host hash per layer).
    fn prime_trace_path() -> Option<&'static str> {
        static P: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
        P.get_or_init(|| std::env::var("MEMRA_PRIME_TRACE").ok())
            .as_deref()
    }

    /// Prefill forward over `tokens`; returns logits [T, n_vocab] (host f32).
    pub fn forward(&self, e: &Engine, tokens: &[u32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        if self.is_gemma4_e4b() { return self.gemma4_e4b_forward(e, tokens, false); }
        if self.cfg.gemma4.is_some() { return self.gemma4_forward(e, tokens, false); }
        let cfg = &self.cfg;
        let n_embd = cfg.n_embd as usize;
        let t = tokens.len();
        let eps = cfg.rms_eps;
        let pos: Vec<i32> = (0..t as i32).collect();
        let pos_d = e.htod_i32(&pos)?;

        let mut x = self.embed(e, tokens)?;   // [T, n_embd]

        for (il, layer) in self.layers.iter().enumerate() {
            // attn_norm
            let mut h = e.uninit(t * n_embd)?;
            e.rms_norm(&x, layer.attn_norm.float_data(), &mut h, n_embd, t, eps)?;

            let mixed = match &layer.mixer {
                Mixer::Full(fa) => self.full_attn(e, fa, &h, &pos_d, t, il)?,
                Mixer::Linear(la) => self.linear_attn(e, la, &h, t)?,
                Mixer::Mla(_) => crate::hybrid::mla_forward_unimplemented(),
            };

            // residual 1
            let mut x1 = e.uninit(t * n_embd)?;
            e.add(&x, &mixed, &mut x1, t * n_embd)?;

            // pre-FFN norm (post_attention_norm), FFN (Dense or MoE), residual 2
            let mut z = e.uninit(t * n_embd)?;
            e.rms_norm(&x1, layer.post_attn_norm.float_data(), &mut z, n_embd, t, eps)?;
            let ffn_out = match &layer.ffn {
                crate::hybrid::Ffn::Dense { ffn_gate, ffn_up, ffn_down } => {
                    let n_ff = ffn_gate.out_features();
                    let mut g2 = e.matmul_group(&[ffn_gate, ffn_up], &z, t)?;
                    let up = g2.pop().unwrap();
                    let gate = g2.pop().unwrap();
                    let mut act = e.uninit(t * n_ff)?;
                    // A DENSE FFN reads the SHEXP clamp array: upstream's one `build_ffn` serves
                    // both the dense MLP and the shared expert, and its limit is
                    // `swiglu_clamp_shexp[il]` (llama-graph.cpp:1751). step35's leading dense
                    // blocks 0-2 therefore key off clamp_shexp, not clamp_exp.
                    Self::ffn_act_lim(e, &self.cfg, &gate, &up, 1.0, 1.0,
                                      self.cfg.clamp_shexp_at(il as u32), &mut act, t * n_ff)?;
                    e.matmul(ffn_down, &act, t)?
                }
                crate::hybrid::Ffn::Moe(m) => self.moe_ffn_il_prefill(e, m, &z, t, il as u16)?,
            };
            let mut x2 = e.uninit(t * n_embd)?;
            e.add(&x1, &ffn_out, &mut x2, t * n_embd)?;
            x = x2;
        }

        let mut hn = e.uninit(t * n_embd)?;
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
        if self.cfg.gemma4.is_some() { return self.gemma4_forward(e, tokens, true); }
        let cfg = &self.cfg;
        let n_embd = cfg.n_embd as usize;
        let t = tokens.len();
        let eps = cfg.rms_eps;
        let pos: Vec<i32> = (0..t as i32).collect();
        let pos_d = e.htod_i32(&pos)?;

        let mut x = self.embed(e, tokens)?;   // [T, n_embd]
        // MEMRA_LAYER_PROBE=1: synchronize + print after every stage — bisects an in-graph
        // ILLEGAL_ADDRESS to (layer, stage) at ~1 line of output per layer (M3 bring-up tool).
        let probe = std::env::var("MEMRA_LAYER_PROBE").is_ok();
        for (il, layer) in self.layers.iter().enumerate() {
            let mut h = e.uninit(t * n_embd)?;
            e.rms_norm(&x, layer.attn_norm.float_data(), &mut h, n_embd, t, eps)?;
            if probe { e.stream().synchronize()?; eprintln!("[probe] L{il} norm ok"); }
            let mixed = match &layer.mixer {
                Mixer::Full(fa) => self.full_attn(e, fa, &h, &pos_d, t, il)?,
                Mixer::Linear(la) => self.linear_attn(e, la, &h, t)?,
                Mixer::Mla(_) => crate::hybrid::mla_forward_unimplemented(),
            };
            if probe { e.stream().synchronize()?; eprintln!("[probe] L{il} mixer ok"); }
            let mut x1 = e.uninit(t * n_embd)?;
            e.add(&x, &mixed, &mut x1, t * n_embd)?;
            let mut z = e.uninit(t * n_embd)?;
            e.rms_norm(&x1, layer.post_attn_norm.float_data(), &mut z, n_embd, t, eps)?;
            let ffn_out = match &layer.ffn {
                crate::hybrid::Ffn::Dense { ffn_gate, ffn_up, ffn_down } => {
                    let n_ff = ffn_gate.out_features();
                    let mut g2 = e.matmul_group(&[ffn_gate, ffn_up], &z, t)?;
                    let up = g2.pop().unwrap();
                    let gate = g2.pop().unwrap();
                    let mut act = e.uninit(t * n_ff)?;
                    // dense FFN keys off the SHEXP clamp array — see forward()'s note.
                    Self::ffn_act_lim(e, &self.cfg, &gate, &up, 1.0, 1.0,
                                      self.cfg.clamp_shexp_at(il as u32), &mut act, t * n_ff)?;
                    e.matmul(ffn_down, &act, t)?
                }
                crate::hybrid::Ffn::Moe(m) => self.moe_ffn_il_prefill(e, m, &z, t, il as u16)?,
            };
            if probe { e.stream().synchronize()?; eprintln!("[probe] L{il} ffn ok"); }
            let mut x2 = e.uninit(t * n_embd)?;
            e.add(&x1, &ffn_out, &mut x2, t * n_embd)?;
            x = x2;
        }
        // norm over all T, then slice the LAST row and run lm_head on that single row.
        let mut hn = e.uninit(t * n_embd)?;
        e.rms_norm(&x, self.output_norm.float_data(), &mut hn, n_embd, t, eps)?;
        let last = e.view(&hn, t * n_embd);            // [T, n_embd]
        let last_row = last.slice((t - 1) * n_embd..t * n_embd);  // [1, n_embd]
        let mut hlast = e.uninit(n_embd)?;
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
    ///
    /// `queued_after` (lane/tick-seg, 2026-08-07): the number of prompt tokens of the SAME
    /// REQUEST that the caller will prime in LATER calls — 0 when this call is the whole
    /// request (every single-shot caller). Serve splits a long prompt across SEVERAL
    /// prime_cache calls (one per scheduler tick, plus the prefix-cache LCP split), and the
    /// request's absolute end position `seq_end = cache.pos + t + queued_after` steers step35's
    /// SWA prefill arm — computing it per CALL made the arm a function of the tick budget
    /// (budgets 512/256/64 DIFFER 1.813e0 vs monolithic, greedy diverging at step 6; dark lanes
    /// default to 256 AND cap by live SLO headroom, so identical judge requests primed
    /// differently under load — research/tick-seg-20260807, receipt in
    /// research/step35-chunkfix-20260807 §9). The parameter is what prime_cache structurally
    /// lacked: it cannot know from `tokens` and `cache` alone whether more of the request is
    /// coming. A SESSION CONTINUATION (a NEW user turn primed onto a live cache) is a NEW
    /// request — its arithmetic is keyed to its own extent, so those callers pass 0; only a
    /// caller that SPLITS one request across calls passes the remainder.
    pub fn prime_cache(&self, e: &Engine, tokens: &[u32], cache: &mut Cache, queued_after: usize)
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
        // MEMRA_PRIME_CHUNK sets the chunk size (tokens); 0 disables chunking (monolithic).
        if self.is_gemma4_e4b() {
            return self.gemma4_e4b_prime(e, tokens, cache);
        }
        if self.cfg.gemma4.is_some() {
            // gemma4 v0: monolithic fresh-prompt prime (chunked/continuation arms later).
            return self.gemma4_prime(e, tokens, cache);
        }
        let chunk = prime_chunk_tokens(t, self.layers.len());
        // CHUNK-ORDER INVARIANCE (lane/chunk-invariance, 2026-08-05; vLLM #38561 shape).
        // MEMRA_PRIME_CHUNK is documented as a memory-transient knob, but it also decides
        // the prefill's ARITHMETIC, so two rigs with different values produced different
        // greedy text for the same prompt (research/session-affinity-20260805: 97- and
        // 149-token prompts). ROOT CAUSE, measured in research/chunk-invariance-20260805
        // (VERDICT.md) — and it is NOT what docs originally said:
        //   * NOT trunk GEMM m / reduction order. REFUTED: the prefill GEMM is m-INVARIANT
        //     (rows [0,32) bit-identical at m=32 vs m=33..80, both quantized wq and the
        //     output head), so growing a chunk cannot move an existing row's value.
        //   * NOT the GDN scan segmentation. REFUTED: MEMRA_GDN_CHUNKED=0 (sequential scan,
        //     no WY segmentation at all) still diverges, so vLLM's mamba-boundary fix does
        //     not describe our leak.
        //   * IT IS the attention numeric CLASS edge in full_attn_prime_fa_dispatch, which
        //     selects on `base_len == 0`: chunk 0 attends over this batch's f32 K/V
        //     (fa_prefill) while every later chunk attends over the q8_0/q5_1 quantized KV
        //     cache (fa_prefill_view_ws). The chunk size therefore decides WHERE in the
        //     prompt that precision edge falls. Signature: per-row maxdiff is exactly 0.0
        //     before the first boundary and O(1) right after, first_div_pos == chunk size.
        // The grain-free fix (full_attn_prime_fa_dispatch below) removed that class edge at
        // the source — every row is in one numeric class, so the chunk size no longer steers
        // arithmetic and MEMRA_PRIME_CHUNK is a pure memory/transient knob again. The interim
        // MEMRA_PRIME_INVARIANT/MEMRA_PRIME_GRAIN pin-the-boundary door was superseded by that
        // fix and KILLED at v0.71 per the flags doctrine (the jsonl + VERDICT.md are the
        // record); the chunkinv gate asserts byte-identity across chunk sizes naked.
        // The REQUEST's absolute end position, computed ONCE before the loop: every chunk sees the
        // same value, whatever the chunk size. step35's SWA arm selects on it so that kernel
        // selection — and therefore the logits — cannot depend on MEMRA_PRIME_CHUNK
        // (research/step35-chunkfix-20260807; see step35_attn_pre_wo's doc note).
        // `+ queued_after` closes the SECOND axis (lane/tick-seg): when serve splits the request
        // across calls, the request still ends at the same absolute position, whatever the tick
        // budget or LCP split point. MEMRA_PRIME_CALLLOCAL=1 is the ROLLBACK SEAM to the FULL
        // pre-fix arithmetic: the per-call value here AND the unaligned FA view offset in
        // step35_attn_pre_wo. Both halves are required for tickinv35's canary teeth under the FA
        // default. Read per call, not cached (the probe flips it in-process between arms). Never
        // on in a measured default run.
        let legacy_calllocal =
            std::env::var("MEMRA_PRIME_CALLLOCAL").as_deref() == Ok("1");
        let seq_end = if legacy_calllocal {
            cache.pos + t
        } else {
            cache.pos + t + queued_after
        };
        if chunk == 0 || t <= chunk {
            return self.prime_chunk(e, tokens, cache, seq_end);
        }
        // PIPELINED PP-2 PRIME (lane/cx-pipeline-prime, 2026-08-08): overlap stage 0 of
        // chunk N+1 with stage 1 of chunk N. The serial split stays reachable through
        // MEMRA_PRIME_PIPE=0 and is the exactness oracle. N>2 keeps the serial walker;
        // this lane owns the balanced two-stage schedule only.
        if crate::pp::prime_pipe_on()
            && crate::pp::prime_pp_on()
            && !crate::pp::pp2_streams_off()
        {
            if let Some(fence) = crate::pp::pp_cuts(self.layers.len()).filter(|f| f.len() == 3) {
                if crate::pp::pp_multi_stream_same_device() {
                    return Err(
                        "prime chunk pipeline refused with 2 stage streams on one device — \
                         that concurrent-stream placement remains quarantined by the deferred \
                         pp flake record. Use one device per stage or MEMRA_PRIME_PIPE=0 for \
                         the serial split."
                            .into(),
                    );
                }
                return self.prime_cache_pp2_pipelined(
                    e, tokens, cache, seq_end, chunk, &fence,
                );
            }
        }
        let mut hiddens = e.uninit(t * n_embd)?;
        let mut last: Option<(Vec<f32>, CudaSlice<f32>)> = None;
        let mut start = 0usize;
        while start < t {
            // keep the tail chunk >= PRIME_MIN_T (the stateful conv needs T >= d_conv-1).
            let mut end = (start + chunk).min(t);
            if t - end > 0 && t - end < PRIME_MIN_T { end = t; }
            let (l, hs, x) = self.prime_chunk(e, &tokens[start..end], cache, seq_end)?;
            e.copy_into(&mut hiddens, start * n_embd, &x, (end - start) * n_embd)?;
            last = Some((l, hs));
            start = end;
        }
        let (logits, h_seed) = last.unwrap();
        Ok((logits, h_seed, hiddens))
    }

    /// PP-2 chunk scheduler: stage 1 of chunk N and stage 0 of chunk N+1 are both queued
    /// before N's epilogue D2H drains the last-stage stream. Arithmetic is unchanged:
    /// every chunk still runs the same two `prime_layers` ranges, boundary copy, output
    /// norm, lm head, and caller hidden-stack copy as the serial split.
    fn prime_cache_pp2_pipelined(
        &self,
        e: &Engine,
        tokens: &[u32],
        cache: &mut Cache,
        seq_end: usize,
        chunk: usize,
        fence: &[usize],
    ) -> Result<(Vec<f32>, CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        debug_assert_eq!(fence.len(), 3);
        let rt = crate::pp::PpNRt::get(e)?;
        assert_eq!(rt.n_stages(), 2, "prime pipeline requires exactly two PP stages");
        let n_embd = self.cfg.n_embd as usize;
        let t = tokens.len();
        let initial_base = cache.pos;
        let caller_stream = e.stream();

        let mut ranges = Vec::new();
        let mut start = 0usize;
        while start < t {
            let mut end = (start + chunk).min(t);
            if t - end > 0 && t - end < PRIME_MIN_T {
                end = t;
            }
            ranges.push((start, end));
            start = end;
        }
        debug_assert!(ranges.len() >= 2);

        // #87 reverse publication before any new stage allocation, then prewarm both
        // boundary slots while the stage streams are otherwise empty. Lazy-growing slot B
        // after stage 1(N) is queued would synchronize that stream and erase the first
        // overlap on a two-chunk prompt.
        rt.fence_stages_behind(&caller_stream)?;
        let max_payload = ranges
            .iter()
            .map(|(s, e)| (e - s) * n_embd)
            .max()
            .unwrap();
        rt.prepare_overlap_slots(0, max_payload)?;

        let mut hiddens = e.uninit(t * n_embd)?;
        let mut last: Option<(Vec<f32>, CudaSlice<f32>)> = None;
        let mut stage_caches = PrimeCacheStages::new(cache, fence[1]);
        let (cache0, cache1) = stage_caches.parts();
        let (first_start, first_end) = ranges[0];
        let mut slot = self.prime_pp2_stage0_enqueue(
            e,
            rt,
            &tokens[first_start..first_end],
            cache0,
            seq_end,
            fence,
            initial_base + first_start,
            true,
        )?;
        cache0.pos = initial_base + first_end;

        for (i, &(start, end)) in ranges.iter().enumerate() {
            let base = initial_base + start;
            debug_assert_eq!(
                cache1.pos, base,
                "stage 1 must drain chunks in original position order"
            );
            let (out, next_slot) = if let Some(&(next_start, next_end)) = ranges.get(i + 1) {
                let next_base = initial_base + next_start;
                debug_assert_eq!(
                    cache0.pos, next_base,
                    "stage 0 must issue chunks in original position order"
                );
                let cache0_stage = &mut *cache0;
                // Step's MoE router readback synchronizes once per layer. Two CUDA streams
                // on one host thread therefore serialize even if the calls are ordered as
                // a pipeline. Drive the disjoint stage caches from two scoped host threads:
                // stage 1 consumes slot N while stage 0 produces slot N+1.
                std::thread::scope(
                    |scope| -> Result<_, Box<dyn std::error::Error>> {
                        let stage0 = scope.spawn(move || -> Result<usize, String> {
                            let next = self
                                .prime_pp2_stage0_enqueue(
                                    e,
                                    rt,
                                    &tokens[next_start..next_end],
                                    cache0_stage,
                                    seq_end,
                                    fence,
                                    next_base,
                                    true,
                                )
                                .map_err(|err| err.to_string())?;
                            cache0_stage.pos = initial_base + next_end;
                            Ok(next)
                        });
                        let x = self.prime_pp2_stage1_enqueue(
                            e,
                            rt,
                            slot,
                            end - start,
                            cache1,
                            seq_end,
                            fence,
                            base,
                            true,
                        )?;
                        let out = {
                            rt.bind_stage(1)?;
                            let _st1 = rt.enter(1);
                            let e1 = rt.engine(1, e);
                            self.prime_chunk_epilogue(e1, x, end - start, cache1)?
                        };
                        let next = stage0
                            .join()
                            .map_err(|_| "pipeprime stage-0 host walker panicked")?
                            .map_err(|err| -> Box<dyn std::error::Error> { err.into() })?;
                        Ok((out, Some(next)))
                    },
                )?
            } else {
                let x = self.prime_pp2_stage1_enqueue(
                    e,
                    rt,
                    slot,
                    end - start,
                    cache1,
                    seq_end,
                    fence,
                    base,
                    true,
                )?;
                let out = {
                    rt.bind_stage(1)?;
                    let _st1 = rt.enter(1);
                    let e1 = rt.engine(1, e);
                    self.prime_chunk_epilogue(e1, x, end - start, cache1)?
                };
                (out, None)
            };

            rt.publish_to(1, &caller_stream)?;
            e.copy_into(
                &mut hiddens,
                start * n_embd,
                &out.2,
                (end - start) * n_embd,
            )?;
            last = Some((out.0, out.1));
            crate::pp::PRIME_SPLIT_CHUNKS
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

            if let Some(next) = next_slot {
                // The caller copy above reads a stage-1 allocation. Before stage 1 of the
                // next chunk can allocate/reuse blocks, mirror #87's body-entry fence.
                // Stage 0(N+1) is already queued before this wait is appended, so its
                // overlap with stage 1(N) is preserved.
                rt.fence_stages_behind(&caller_stream)?;
                slot = next;
            }
        }

        debug_assert_eq!(cache0.pos, initial_base + t);
        debug_assert_eq!(cache1.pos, initial_base + t);
        let (logits, h_seed) = last.unwrap();
        Ok((logits, h_seed, hiddens))
    }

    /// One prime chunk: the full layer stack over `tokens`, continuing from the cache's current
    /// state (`cache.pos` = tokens already primed; 0 = fresh). Positions/RoPE are absolute
    /// task #21 de-broadcast: the q/k head count PREP must emit so the scan's consumers
    /// agree. Compact (num_k) ONLY when the scan will take the chunked+mma route AND
    /// num_k == num_v/2 (the engine-side hint mirrors this exact formula); everything
    /// else (s128 verify tier, chunked-off, mma-off) keeps the broadcast layout.
    fn gdn_hk(e: &Engine, t: usize, num_v: usize, num_k: usize) -> usize {
        if Engine::gdn_db_on()
            && Engine::gdn_chunked_enabled() && t >= 16
            && e.gdn_mma_enabled(Engine::gdn_chunk_size())
            && num_k * 2 == num_v
        {
            num_k
        } else {
            num_v
        }
    }

    /// task #17: gate for the fused fp16-operand epilogues (silu_mul/gated_rmsnorm/sig_mul
    /// `_f16out` twins). Bit-identical class (the twins emit the cvt kernel's exact halves),
    /// but seam-gated (MEMRA_F16OUT=0) so a bit-check can arbitrate, and OFF under verify-exact
    /// (matmul_group skips the f16 lane there; the twins must not resurrect it).
    fn f16out_on(e: &Engine, t: usize) -> bool {
        crate::f16_ffi::pp_f16_enabled() && t >= 16 && !e.verify_exact_on()
            && std::env::var("MEMRA_F16OUT").as_deref() != Ok("0")
    }

    /// (cache.pos + i). Returns (last-row logits, h_seed, this chunk's hidden stack [T, n_embd]).
    /// See HybridModel::prime_slabs — the eager prime's resident trunk transients.
    /// PER-DEVICE since lane/pp-leverb (2026-08-08): the map is keyed by the allocating
    /// engine's CUDA ordinal — under the prime stage split each stage's range walks through
    /// its OWN slabs on its own device (a dev0 slab dereferenced by a dev1 kernel would be
    /// a peer read per GEMM operand, the exact class Lever B removes). Single-device rigs
    /// see one entry, byte-identical behavior.
    pub fn prime_slabs_get(
        &self,
        e: &Engine,
        t: usize,
        n_embd: usize,
        n_ff_max: usize,
    ) -> Result<std::sync::Arc<std::sync::Mutex<PrimeSlabs>>, Box<dyn std::error::Error>> {
        let mut slabs = self.prime_slabs.lock().unwrap();
        let dev = e.ctx().ordinal();
        let need_new = match slabs.get(&dev) {
            None => true,
            Some(sl) => sl.lock().unwrap().t_cap < t,
        };
        if need_new {
            slabs.insert(dev, std::sync::Arc::new(std::sync::Mutex::new(PrimeSlabs {
                t_cap: t,
                h: e.uninit(t * n_embd)?,
                x1: e.uninit(t * n_embd)?,
                z: e.uninit(t * n_embd)?,
                act: e.uninit(t * n_ff_max)?,
                xa: e.uninit(t * n_embd)?,
                xb: e.uninit(t * n_embd)?,
                h16: e.alloc_u8_uninit(t * n_embd * 2)?,
                z16: e.alloc_u8_uninit(t * n_embd * 2)?,
                gate: e.uninit(t * n_ff_max)?,
                up: e.uninit(t * n_ff_max)?,
                ffn_out: e.uninit(t * n_embd)?,
                seg_glue: Vec::new(),
                mixed: e.uninit(t * n_embd)?,
                seg_mid: Vec::new(),
                seg_t: 0,
            })));
        }
        Ok(slabs.get(&dev).expect("prime slab inserted").clone())
    }

    /// `seq_end` = the whole REQUEST's absolute end position (`cache.pos + prompt_len` at
    /// `prime_cache` entry), NOT this chunk's end. Chunk-size-invariant by construction; step35's
    /// SWA arm selects on it (see `step35_attn_pre_wo`). Every other arch ignores it.
    fn prime_chunk(&self, e: &Engine, tokens: &[u32], cache: &mut Cache, seq_end: usize)
                       -> Result<(Vec<f32>, CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        // LEVER B (lane/pp-leverb, 2026-08-08): the ppN door for the chunked prime. With the
        // door open + per-stage streams, each chunk walks its layer ranges on the OWNING
        // stage's engine/device (the anatomy receipt this kills: dev1 ran ZERO prefill
        // kernels; stage-1 trunk weights were peer-read = 22% of the pp4096 wall).
        // MEMRA_PRIME_PP=0 = the unsplit rollback (also the prime-split-gate's reference
        // arm — prime deliberately keeps NO refuse_unsplit_if_remote, see pp.rs). The
        // MEMRA_PP_STREAMS=0 seam keeps the unsplit walk too: in that regime the sharded
        // loader is off and there is nothing remote to split for.
        if self.cfg.gemma4.is_none()
            && !crate::pp::pp2_streams_off()
            && crate::pp::prime_pp_on()
        {
            if let Some(fence) = crate::pp::pp_cuts(self.layers.len()) {
                return self.prime_chunk_ppn(e, tokens, cache, seq_end, &fence);
            }
        }
        let t = tokens.len();
        let base = cache.pos;
        debug_assert!(seq_end >= base + t, "prime_chunk: seq_end must cover this chunk");
        let pos: Vec<i32> = (base as i32..(base + t) as i32).collect();
        let pos_d = e.htod_i32(&pos)?;

        let x_embed = self.embed(e, tokens)?;   // [T, n_embd]
        let x = self.prime_layers(
            e, x_embed, 0, self.layers.len(), &pos_d, t, base, cache, seq_end,
        )?;
        self.prime_chunk_epilogue(e, x, t, cache)
    }

    /// PRIME RANGE SUBGRAPH (lane/pp-leverb, 2026-08-08): layers `[lo, hi)` of the chunked
    /// prime walk — `prime_chunk`'s trunk loop extracted verbatim to the
    /// `decode_layers_eager(lo, hi)` / `verify_layers(lo, hi)` contract: enters with a
    /// MATERIALIZED `[T, n_embd]` residual, exits with the range's final residual
    /// materialized (cloned out of the slab). At `lo=0, hi=n_layers` — the unsplit call —
    /// the launch sequence is byte-identical to the pre-extraction body. Range semantics:
    ///   - the cross-layer [down-add + NEXT attn-norm] fusion is range-LOCAL (`il + 1 < hi`):
    ///     layer `hi`'s attn_norm belongs to the NEXT stage's device, so the range ends with
    ///     the plain add (materialize) and the next stage hoists its own first norm — the
    ///     kernel-check-pinned `add_rms_norm == add then rms_norm` identity, the same law
    ///     the decode split rests on (`prime-split-gate` arbitrates end-to-end);
    ///   - prime slabs are PER-DEVICE (`prime_slabs_get` keys on the engine ordinal), so
    ///     each stage walks through its own resident transients;
    ///   - the S-glue/S-mid capture path requires the FULL range (its lookahead fuses
    ///     `self.layers[il+1]` unconditionally) — `use_seg` gains `lo == 0 && hi == n_layers`.
    #[allow(clippy::too_many_arguments)]
    fn prime_layers(&self, e: &Engine, x_in: CudaSlice<f32>, lo: usize, hi: usize,
                    pos_d: &CudaSlice<i32>, t: usize, base: usize, cache: &mut Cache,
                    seq_end: usize)
                    -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let n_embd = cfg.n_embd as usize;
        let eps = cfg.rms_eps;
        // task #14: fuse the fp16 GEMM-operand emission into the trunk norms (kills the
        // standalone convert launches). Only when the f16 lane serves and T reaches the
        // GEMM tier; bit-identical either way.
        let f16fuse = crate::f16_ffi::pp_f16_enabled() && t >= 16;
        // PRIME SLABS (piecewise foundation): trunk transients in resident buffers —
        // ~224 fewer alloc/free calls per prime and FROZEN Lt operand addresses
        // (nvjet's alignment-variant selection becomes run-to-run stable). Every slab is
        // fully overwritten before use; x ping-pongs xa<->xb; the hidden-stack return
        // clones the final x (the slab cannot leave). Non-slab fallback: MEMRA_PRIME_SLABS=0.
        let n_ff_max = self.layers.iter().map(|l| match &l.ffn {
            crate::hybrid::Ffn::Dense { ffn_gate, .. } => ffn_gate.out_features(),
            _ => n_embd,
        }).max().unwrap_or(n_embd).max(n_embd);
        let use_slabs = std::env::var("MEMRA_PRIME_SLABS").as_deref() != Ok("0");
        let slab = if use_slabs {
            Some(self.prime_slabs_get(e, t, n_embd, n_ff_max)?)
        } else {
            None
        };
        let mut slab_guard = slab.as_ref().map(|sl| sl.lock().unwrap());
        let mut x_own;   // fallback storage when slabs are off
        type SlabRefs<'a> = (&'a mut CudaSlice<f32>, &'a mut CudaSlice<f32>, &'a mut CudaSlice<f32>, &'a mut CudaSlice<f32>, &'a mut CudaSlice<u8>, &'a mut CudaSlice<u8>, &'a mut CudaSlice<f32>, &'a mut CudaSlice<f32>, &'a mut CudaSlice<f32>);
        let (mut x_cur, mut x_nxt, sl): (&mut CudaSlice<f32>, &mut CudaSlice<f32>, Option<SlabRefs>);
        let mut seg: Option<(&mut Vec<Option<cudarc::driver::CudaGraph>>, &mut Vec<Option<cudarc::driver::CudaGraph>>, &mut CudaSlice<f32>, &mut usize)> = None;
        let mut x_own2;
        match slab_guard.as_mut() {
            Some(g) => {
                let slabs = &mut **g;
                e.copy_into(&mut slabs.xa, 0, &x_in, t * n_embd)?;
                let PrimeSlabs { xa, xb, h, x1, z, act, h16, z16, gate, up, ffn_out, seg_glue, mixed, seg_mid, seg_t, .. } = slabs;
                x_cur = xa;
                x_nxt = xb;
                seg = Some((seg_glue, seg_mid, mixed, seg_t));
                sl = Some((h, x1, z, act, h16, z16, gate, up, ffn_out));
            }
            None => {
                x_own = x_in;
                x_own2 = e.uninit(t * n_embd)?;
                x_cur = &mut x_own;
                x_nxt = &mut x_own2;
                sl = None;
            }
        }
        let mut alloc_h; let mut alloc_x1; let mut alloc_z; let mut alloc_act;
        let mut alloc_h16; let mut alloc_z16;
        let mut alloc_gate; let mut alloc_up; let mut alloc_fo;
        let (h, x1, z, act): (&mut CudaSlice<f32>, &mut CudaSlice<f32>, &mut CudaSlice<f32>, &mut CudaSlice<f32>);
        let (h16, z16): (&mut CudaSlice<u8>, &mut CudaSlice<u8>);
        let (sl_gate, sl_up, sl_fo): (&mut CudaSlice<f32>, &mut CudaSlice<f32>, &mut CudaSlice<f32>);
        match sl {
            Some((a, b, c, d, e16, f16b, g, u, fo)) => {
                h = a; x1 = b; z = c; act = d; h16 = e16; z16 = f16b;
                sl_gate = g; sl_up = u; sl_fo = fo;
            }
            None => {
                alloc_h = e.uninit(t * n_embd)?;
                alloc_x1 = e.uninit(t * n_embd)?;
                alloc_z = e.uninit(t * n_embd)?;
                alloc_act = e.uninit(t * n_ff_max)?;
                alloc_h16 = e.alloc_u8_uninit(t * n_embd * 2)?;
                alloc_z16 = e.alloc_u8_uninit(t * n_embd * 2)?;
                alloc_gate = e.uninit(t * n_ff_max)?;
                alloc_up = e.uninit(t * n_ff_max)?;
                alloc_fo = e.uninit(t * n_embd)?;
                h = &mut alloc_h; x1 = &mut alloc_x1; z = &mut alloc_z; act = &mut alloc_act;
                h16 = &mut alloc_h16; z16 = &mut alloc_z16;
                sl_gate = &mut alloc_gate; sl_up = &mut alloc_up; sl_fo = &mut alloc_fo;
            }
        }
        // piecewise increment 3: layer-0 norm hoisted; the per-layer tail fuses
        // [down-add + NEXT layer's attn-norm] into one captured S-glue segment
        // (all-slab IO, zero in-graph allocations). Capture happens lazily on the
        // first prime at this t (capture does not execute -> launch right after).
        let n_layers = self.layers.len();
        // OPT-IN (2026-07-26 interleaved verdict): 2-kernel segments measured NET
        // -0.7%-to-neutral under the interleaved A/B protocol — one cuGraphLaunch costs
        // about what two kernel submissions do. The earlier "+0.9%" was cross-run clock
        // drift (the repo's interleaved-A/B law exists for exactly this). Larger segments
        // (S-prep/S-attn, 7-9 kernels) remain the open hypothesis; the core-split
        // machinery stays (byte-identical) as their foundation.
        // step35 is excluded: the core-split path calls `full_attn_prime_core_inner`, which is
        // the GENERIC attn core (uniform n_head, rope_dim_count, no window, no head-wise gate).
        // step35 rides its own mixer through the normal per-layer arm below.
        let use_seg = f16fuse && seg.is_some() && self.cfg.step35.is_none()
            && lo == 0 && hi == n_layers
            && std::env::var("MEMRA_PRIME_SEG").as_deref() == Ok("1");
        if let Some((sg, sm, _, st)) = seg.as_mut() {
            if **st != t {
                sg.clear();
                sg.extend((0..n_layers).map(|_| None));
                sm.clear();
                sm.extend((0..n_layers).map(|_| None));
                **st = t;
            }
        }
        {
            let layer_lo = &self.layers[lo];
            if f16fuse {
                e.rms_norm_f16out(x_cur, layer_lo.attn_norm.float_data(), h, h16, n_embd, t, eps)?;
            } else {
                e.rms_norm(x_cur, layer_lo.attn_norm.float_data(), h, n_embd, t, eps)?;
            }
        }
        for il in lo..hi {
            let layer = &self.layers[il];
            let hx16 = if f16fuse { Some(&*h16) } else { None };
            if use_seg {
                // core-split path: projections -> _inner core -> out-GEMM INTO the mixed
                // slab (no copies) -> S-mid segment [add + post-norm] as one graph launch.
                let (pre, pre16, w_out) = match &layer.mixer {
                    Mixer::Full(fa) => {
                        let g3 = match hx16 {
                            Some(xh) => e.matmul_group_xh(&[&fa.wq, &fa.wk, &fa.wv], h, xh, t)?,
                            None => e.matmul_group(&[&fa.wq, &fa.wk, &fa.wv], h, t)?,
                        };
                        let (pre, pre16) = self.full_attn_prime_core_inner(e, fa, g3, &pos_d, t, cache, il)?;
                        (pre, pre16, &fa.wo)
                    }
                    Mixer::Mla(_) => crate::hybrid::mla_forward_unimplemented(),
                    Mixer::Linear(la) => {
                        let ws = [&la.wqkv, &la.wqkv_gate, &la.ssm_beta, &la.ssm_alpha];
                        let g4 = match hx16 {
                            Some(xh) => e.matmul_group_xh(&ws, h, xh, t)?,
                            None => e.matmul_group(&ws, h, t)?,
                        };
                        let (pre, pre16) = self.linear_attn_prime_core_pad_inner(e, la, g4, t, cache, il, None)?;
                        (pre, pre16, &la.ssm_out)
                    }
                };
                {
                    let (_, sm, mslab, _) = seg.as_mut().unwrap();
                    let pre_n = pre.len() / t;
                    let xh_pre = match pre16 {
                        Some(x) => x,
                        None => e.f16_act(&pre, t * pre_n, pre_n)?,
                    };
                    if !e.try_f16_gemm_pre_into(w_out, &xh_pre, t, mslab)? {
                        let y = e.matmul(w_out, &pre, t)?;
                        e.copy_into(mslab, 0, &y, t * n_embd)?;
                    }
                    if sm[il].is_none() {
                        use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
                        let w_post = layer.post_attn_norm.float_data();
                        e.stream().synchronize()?;
                        e.stream().begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)?;
                        let r = (|| -> Result<(), Box<dyn std::error::Error>> {
                            e.add(x_cur, mslab, x1, t * n_embd)?;
                            e.rms_norm_f16out(x1, w_post, z, z16, n_embd, t, eps)?;
                            Ok(())
                        })();
                        let g = e.stream().end_capture(
                            CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH);
                        r?;
                        sm[il] = Some(g?.ok_or("S-mid capture produced no graph")?);
                    }
                    sm[il].as_ref().unwrap().launch()?;
                }
            } else {
                let mixed = match &layer.mixer {
                    Mixer::Full(fa) => self.full_attn_prime(e, fa, h, hx16, &pos_d, t, cache, il,
                                                            seq_end)?,
                    Mixer::Linear(la) => self.linear_attn_prime(e, la, h, hx16, t, cache, il)?,
                    Mixer::Mla(_) => crate::hybrid::mla_forward_unimplemented(),
                };
                if f16fuse {
                    // round 28: residual+norm in ONE kernel (add_rms_norm precedent,
                    // bit-identical) — the standalone add pass disappears.
                    e.add_rms_norm_f16out(x_cur, &mixed, layer.post_attn_norm.float_data(),
                                          x1, z, z16, n_embd, t, eps)?;
                } else {
                    e.add(x_cur, &mixed, x1, t * n_embd)?;
                    e.rms_norm(x1, layer.post_attn_norm.float_data(), z, n_embd, t, eps)?;
                }
            }
            let zx16 = if f16fuse { Some(&*z16) } else { None };
            match &layer.ffn {
                crate::hybrid::Ffn::Dense { ffn_gate, ffn_up, ffn_down } => {
                    let n_ff = ffn_gate.out_features();
                    // gate/up INTO boundary slabs (piecewise increment 2); fall back to
                    // the allocating group + copy when a mirror is missing.
                    let mut into_ok = false;
                    if let Some(xh) = zx16 {
                        into_ok = e.try_f16_gemm_pre_into(ffn_gate, xh, t, sl_gate)?
                            && e.try_f16_gemm_pre_into(ffn_up, xh, t, sl_up)?;
                    }
                    if !into_ok {
                        let mut g2 = match zx16 {
                            Some(xh) => e.matmul_group_xh(&[ffn_gate, ffn_up], z, xh, t)?,
                            None => e.matmul_group(&[ffn_gate, ffn_up], z, t)?,
                        };
                        let up_y = g2.pop().unwrap();
                        let gate_y = g2.pop().unwrap();
                        e.copy_into(sl_gate, 0, &gate_y, t * n_ff)?;
                        e.copy_into(sl_up, 0, &up_y, t * n_ff)?;
                    }
                    // task #17: the silu arm's f16out twin emits the down GEMM's fp16
                    // operand in-epilogue; non-silu activations keep the standalone convert.
                    // silu_mul_f16out is PLAIN silu(gate)*up — a clamped layer (step35 dense
                    // blocks under a live swiglu_clamp_shexp) must take the ffn_act_lim arm.
                    let d_lim = self.cfg.clamp_shexp_at(il as u32);
                    let act16 = if Self::f16out_on(e, t) && self.cfg.m3.is_none()
                        && d_lim.is_none() {
                        let mut a16 = e.alloc_u8_uninit(t * n_ff * 2)?;
                        e.silu_mul_f16out(sl_gate, sl_up, act, &mut a16, t * n_ff)?;
                        Some(a16)
                    } else {
                        Self::ffn_act_lim(e, &self.cfg, sl_gate, sl_up, 1.0, 1.0, d_lim,
                                          act, t * n_ff)?;
                        None
                    };
                    // down GEMM into the ffn_out slab (f16 arm; fallback copies)
                    let xh_act = match act16 {
                        Some(x) => x,
                        None => e.f16_act(act, t * n_ff, n_ff)?,
                    };
                    if !e.try_f16_gemm_pre_into(ffn_down, &xh_act, t, sl_fo)? {
                        let y = e.matmul(ffn_down, &*act, t)?;
                        e.copy_into(sl_fo, 0, &y, t * n_embd)?;
                    }
                }
                crate::hybrid::Ffn::Moe(m) => {
                    let y = self.moe_ffn_il_prefill(e, m, z, t, il as u16)?;
                    e.copy_into(sl_fo, 0, &y, t * n_embd)?;
                }
            }
            if use_seg && il + 1 < hi {
                // S-glue segment: [add + next attn-norm(+f16out)] — one cuGraphLaunch
                let w_next = self.layers[il + 1].attn_norm.float_data();
                let (sg, _, _, _) = seg.as_mut().unwrap();
                if sg[il].is_none() {
                    use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
                    e.stream().synchronize()?;
                    e.stream().begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)?;
                    let r = (|| -> Result<(), Box<dyn std::error::Error>> {
                        e.add(x1, sl_fo, x_nxt, t * n_embd)?;
                        e.rms_norm_f16out(x_nxt, w_next, h, h16, n_embd, t, eps)?;
                        Ok(())
                    })();
                    let g = e.stream().end_capture(
                        CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH);
                    r?;
                    sg[il] = Some(g?.ok_or("S-glue capture produced no graph")?);
                }
                sg[il].as_ref().unwrap().launch()?;
            } else {
                if il + 1 < hi {
                    let w_next = self.layers[il + 1].attn_norm.float_data();
                    if f16fuse {
                        e.add_rms_norm_f16out(x1, sl_fo, w_next, x_nxt, h, h16, n_embd, t, eps)?;
                    } else {
                        e.add(x1, sl_fo, x_nxt, t * n_embd)?;
                        e.rms_norm(x_nxt, w_next, h, n_embd, t, eps)?;
                    }
                } else {
                    e.add(x1, sl_fo, x_nxt, t * n_embd)?;
                }
            }
            // CHUNK-INVARIANCE BISECT SEAM (lane/chunk-invariance, 2026-08-05): dump the
            // per-layer post-residual hidden for the LAST row of this chunk, keyed by
            // absolute position, so two runs at different MEMRA_PRIME_CHUNK can be diffed
            // layer-by-layer to find the FIRST diverging layer. Diagnostic only —
            // unset (the default) costs one OnceLock read per layer.
            if let Some(path) = Self::prime_trace_path() {
                let row = (base + t - 1) as usize;
                let host = e.dtoh(x_nxt)?;
                let last = &host[(t - 1) * n_embd..t * n_embd];
                use std::io::Write as _;
                let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
                let mut h64: u64 = 0xcbf29ce484222325;
                for v in last {
                    h64 ^= v.to_bits() as u64;
                    h64 = h64.wrapping_mul(0x100000001b3);
                }
                writeln!(f, "{{\"pos\":{row},\"layer\":{il},\"t\":{t},\"base\":{base},\
                             \"hash\":\"{h64:016x}\",\"v0\":{:.9e},\"v1\":{:.9e},\"v2\":{:.9e}}}",
                         last[0], last[1], last[2])?;
            }
            std::mem::swap(&mut x_cur, &mut x_nxt);
        }
        // hidden-stack return: clone the final x out of the slab
        let mut x = e.uninit(t * n_embd)?;
        e.copy_into(&mut x, 0, x_cur, t * n_embd)?;
        drop(slab_guard);
        Ok(x)
    }

    /// The prime chunk's tail — h_seed + output_norm + one-row lm head + cache.pos advance —
    /// shared verbatim by the unsplit walk and the last stage of the ppN walk (`e` = the
    /// engine that produced `x`, i.e. the last stage's under the split; output_norm/output
    /// were loaded through that engine by the sharded loader, hybrid.rs `e_head`).
    fn prime_chunk_epilogue(&self, e: &Engine, x: CudaSlice<f32>, t: usize, cache: &mut Cache)
                            -> Result<(Vec<f32>, CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let n_embd = self.cfg.n_embd as usize;
        let eps = self.cfg.rms_eps;
        // h_seed = LAST row of x BEFORE output_norm (MTP-PLAN §A default) or AFTER it
        // (MEMRA_SPEC_HPOST — reference convention; hn is computed just below either way, so
        // the post-norm copy happens after hn exists).
        let mut h_seed = e.uninit(n_embd)?;
        if !crate::spec::spec_hpost() {
            e.copy_view_into(&mut h_seed, 0, &x.slice((t - 1) * n_embd..t * n_embd), n_embd)?;
        }
        // last-row logits, exactly like forward_last (norm all T — per-row op — then lm_head on 1 row).
        let mut hn = e.uninit(t * n_embd)?;
        e.rms_norm(&x, self.output_norm.float_data(), &mut hn, n_embd, t, eps)?;
        if crate::spec::spec_hpost() {
            e.copy_view_into(&mut h_seed, 0, &hn.slice((t - 1) * n_embd..t * n_embd), n_embd)?;
        }
        let last = e.view(&hn, t * n_embd);
        let last_row = last.slice((t - 1) * n_embd..t * n_embd);
        let mut hlast = e.uninit(n_embd)?;
        e.copy_view_into(&mut hlast, 0, &last_row, n_embd)?;
        let logits = e.matmul(&self.output, &hlast, 1)?;
        cache.pos += t;
        // Hidden stack handed to generate_spec as prompt_h: pre-norm x (default) or the full
        // post-norm stack hn (MEMRA_SPEC_HPOST).
        Ok((e.dtoh(&logits)?, h_seed, if crate::spec::spec_hpost() { hn } else { x }))
    }

    /// THE PRIME STAGE SPLIT (lane/pp-leverb, 2026-08-08): one prime chunk as N stage
    /// subgraphs, each range on ITS OWN engine/stream/device, with the `[T, n_embd]`
    /// boundary handoff at every fence cut — the prime twin of `decode_step_h_ppn` /
    /// `decode_step_t_core_ppn`, and the kill for the anatomy's two receipts: stage-1 trunk
    /// weights stop being peer-read (22% of the pp4096 wall) and dev1 stops running zero
    /// prefill kernels. Structure mirrors the verify split exactly:
    ///   stage 0        `rt.enter(0)` → per-stage pos_d + embed (the table lives with
    ///                  stage 0) → `prime_layers(fence[0], fence[1])` → `rt.tx`
    ///   middle stages  `rt.rx` → per-stage pos_d → range → `rt.tx`
    ///   last stage     `rt.rx` → range → the shared epilogue (output_norm + head live
    ///                  there via the sharded loader) → `publish_to`
    /// Laws inherited (not relearned): `fence_stages_behind` at entry (#87 — a previous
    /// round's stage-freed buffers must not be reused under the caller's queued reads);
    /// per-stage `pos_d` (allocated/consumed/freed on one stream); per-stage Engines from
    /// `PpNRt` (shared-scratch race); EXIT PUBLICATION for the device-resident returns
    /// (h_seed + hidden stack live on the last stage — the caller's stream must wait).
    /// KV/MoE locality falls out: each range's KV appends run on the owning stage
    /// (`pp::new_cache` placed the buffers there), each stage engine owns its own SLRU
    /// pool (per-Engine `moe_cache`, sized on ITS device — the SGLang #33666 law), and the
    /// slab-local MoE arm's `DevExps.dev` gate now matches on stage-1 layers too.
    /// EXACTNESS: the split adds zero deviation by construction (same kernels, same bytes,
    /// boundary = straight f32 copy); `prime-split-gate` (ppsplit) arbitrates bit-for-bit
    /// and its liveness counter is bumped here — the gate goes green with this function.
    fn prime_chunk_ppn(&self, e: &Engine, tokens: &[u32], cache: &mut Cache, seq_end: usize,
                       fence: &[usize])
                       -> Result<(Vec<f32>, CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let rt = crate::pp::PpNRt::get(e)?;
        let n_st = fence.len() - 1;
        assert_eq!(
            rt.n_stages(), n_st,
            "PpNRt stage count {} != fence stages {n_st}", rt.n_stages()
        );
        let n_embd = self.cfg.n_embd as usize;
        let t = tokens.len();
        let base = cache.pos;
        debug_assert!(seq_end >= base + t, "prime_chunk_ppn: seq_end must cover this chunk");
        let payload = t * n_embd;
        // The caller's ambient stream, captured BEFORE any enter() pushes a stage stream
        // (inside a stage scope e.stream() IS the stage stream and the exit wait would
        // self-order into a no-op) — the decode_step_t_core_ppn pattern.
        let caller_stream = e.stream();
        rt.fence_stages_behind(&caller_stream)?;

        if n_st == 2 {
            let slot = self.prime_pp2_stage0_enqueue(
                e, rt, tokens, cache, seq_end, fence, base, false,
            )?;
            let x = self.prime_pp2_stage1_enqueue(
                e, rt, slot, t, cache, seq_end, fence, base, false,
            )?;
            let out = {
                rt.bind_stage(1)?;
                let _st1 = rt.enter(1);
                let e1 = rt.engine(1, e);
                self.prime_chunk_epilogue(e1, x, t, cache)?
            };
            rt.publish_to(1, &caller_stream)?;
            crate::pp::PRIME_SPLIT_CHUNKS
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok(out);
        }

        let pos: Vec<i32> = (base as i32..(base + t) as i32).collect();

        // ---- STAGE 0: embed + layers [0, fence[1]) + boundary-0 TX ----
        let mut slot = {
            let _st0 = rt.enter(0);
            let e0 = rt.engine(0, e);
            let pos_d = e0.htod_i32(&pos)?;
            let x = self.embed(e0, tokens)?;
            let x = self.prime_layers(
                e0, x, fence[0], fence[1], &pos_d, t, base, cache, seq_end,
            )?;
            rt.tx(0, &x, payload)?
            // x + pos_d drop here: freed stream-ordered on stage-0's stream after use.
        };

        // ---- MIDDLE STAGES: RX boundary s-1 -> range -> TX boundary s ----
        for s in 1..n_st - 1 {
            let _st = rt.enter(s);
            let es = rt.engine(s, e);
            let pos_d = es.htod_i32(&pos)?;
            let x = rt.rx(s - 1, slot, payload)?;
            let x = self.prime_layers(
                es, x, fence[s], fence[s + 1], &pos_d, t, base, cache, seq_end,
            )?;
            slot = rt.tx(s, &x, payload)?;
        }

        // ---- LAST STAGE: RX + final range + the shared epilogue ----
        let _stl = rt.enter(n_st - 1);
        let el = rt.engine(n_st - 1, e);
        let pos_d = el.htod_i32(&pos)?;
        let x = rt.rx(n_st - 2, slot, payload)?;
        let x = self.prime_layers(
            el, x, fence[n_st - 1], fence[n_st], &pos_d, t, base, cache, seq_end,
        )?;
        let out = self.prime_chunk_epilogue(el, x, t, cache)?;
        // EXIT PUBLICATION: h_seed + the hidden stack are device-resident on the last
        // stage's stream; the caller resumes on its own stream (chunk-loop copy_into /
        // generate_spec's prompt_h consumer). The logits dtoh above already drained the
        // stage stream host-side, but the law is stated in events, not in a dtoh side
        // effect a later deferred form would remove.
        rt.publish_to(n_st - 1, &caller_stream)?;
        crate::pp::PRIME_SPLIT_CHUNKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(out)
    }

    fn prime_pp2_stage0_enqueue(
        &self,
        e: &Engine,
        rt: &crate::pp::PpNRt,
        tokens: &[u32],
        cache: &mut Cache,
        seq_end: usize,
        fence: &[usize],
        base: usize,
        pipelined: bool,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let t = tokens.len();
        let n_embd = self.cfg.n_embd as usize;
        let pos: Vec<i32> = (base as i32..(base + t) as i32).collect();
        rt.bind_stage(0)?;
        let _st0 = rt.enter(0);
        let e0 = rt.engine(0, e);
        let pos_d = e0.htod_i32(&pos)?;
        let x = self.embed(e0, tokens)?;
        let _overlap = pipelined.then(crate::pp::enter_prime_pipe_stage);
        let x = self.prime_layers(
            e0, x, fence[0], fence[1], &pos_d, t, base, cache, seq_end,
        )?;
        if pipelined {
            rt.tx_pipelined(0, &x, t * n_embd)
        } else {
            rt.tx(0, &x, t * n_embd)
        }
    }

    fn prime_pp2_stage1_enqueue(
        &self,
        e: &Engine,
        rt: &crate::pp::PpNRt,
        slot: usize,
        t: usize,
        cache: &mut Cache,
        seq_end: usize,
        fence: &[usize],
        base: usize,
        pipelined: bool,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let n_embd = self.cfg.n_embd as usize;
        let pos: Vec<i32> = (base as i32..(base + t) as i32).collect();
        rt.bind_stage(1)?;
        let _st1 = rt.enter(1);
        let e1 = rt.engine(1, e);
        let pos_d = e1.htod_i32(&pos)?;
        let x = rt.rx(0, slot, t * n_embd)?;
        let _overlap = pipelined.then(crate::pp::enter_prime_pipe_stage);
        self.prime_layers(
            e1, x, fence[1], fence[2], &pos_d, t, base, cache, seq_end,
        )
    }

    /// CAPTURE-SAFE prime trunk (task #14 increment 1, ARCHITECTURE-H100.md design v2):
    /// prime_chunk's layer stack with every capture hazard hoisted — `x` is the
    /// PRE-EMBEDDED input in a STABLE graph-input buffer (embed + its 8MB htod stay
    /// eager, one launch), `pos_d` is a baked device param (fresh prime = 0..T, constant
    /// per bucket), logits/h_seed/hidden stay DEVICE-resident (no dtoh), and cache.pos is
    /// NOT advanced (host state — the replay wrapper owns it). Body mirrors prime_chunk
    /// (the prime-graph-gate pins them together). KNOWN smoke-scope gap: append host-len
    /// bookkeeping still runs on the host per call — the real replay path moves the write
    /// slot to the len_d device counter (increment 3).
    /// GRAPH-OUTPUT CONTRACT: results are COPIED into caller-provided stable buffers
    /// (`logits_out` [n_vocab], `h_seed_out` [n_embd]) — every internal allocation drops
    /// INSIDE the capture region (alloc+free node pairs). Retaining an in-capture
    /// allocation across end_capture makes instantiate throw INVALID_VALUE (smoke finding
    /// 2026-07-26), and under AUTO_FREE_ON_LAUNCH its address wouldn't survive a launch
    /// anyway — the decode GraphSession's pre-allocated-output pattern is the law here.
    pub fn prime_chunk_captured(&self, e: &Engine, x_in: &CudaSlice<f32>, pos_d: &CudaSlice<i32>,
                                t: usize, cache: &mut Cache,
                                len_d: &CudaSlice<i32>,
                                logits_out: &mut CudaSlice<f32>, h_seed_out: &mut CudaSlice<f32>)
                                -> Result<(), Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let n_embd = cfg.n_embd as usize;
        let eps = cfg.rms_eps;
        let f16fuse = crate::f16_ffi::pp_f16_enabled() && t >= 16;
        let mut x = e.uninit(t * n_embd)?;
        e.copy_into(&mut x, 0, x_in, t * n_embd)?;
        for (il, layer) in self.layers.iter().enumerate() {
            let mut h = e.uninit(t * n_embd)?;
            let mut hx16: Option<CudaSlice<u8>> = None;
            if f16fuse {
                let mut b16 = e.alloc_u8_uninit(t * n_embd * 2)?;
                e.rms_norm_f16out(&x, layer.attn_norm.float_data(), &mut h, &mut b16, n_embd, t, eps)?;
                hx16 = Some(b16);
            } else {
                e.rms_norm(&x, layer.attn_norm.float_data(), &mut h, n_embd, t, eps)?;
            }
            let mixed = match &layer.mixer {
                // The captured prime is ONE unchunked bucket over a FRESH cache (pos == 0), so the
                // request ends at t. If bucketed capture ever composes with chunking, seq_end must
                // come from the caller (see step35_attn_pre_wo's doc note).
                Mixer::Full(fa) => self.full_attn_prime(e, fa, &h, hx16.as_ref(), pos_d, t, cache,
                                                        il, t)?,
                Mixer::Mla(_) => crate::hybrid::mla_forward_unimplemented(),
                Mixer::Linear(la) => {
                    let ws = [&la.wqkv, &la.wqkv_gate, &la.ssm_beta, &la.ssm_alpha];
                    let g4 = match hx16.as_ref() {
                        Some(xh) => e.matmul_group_xh(&ws, &h, xh, t)?,
                        None => e.matmul_group(&ws, &h, t)?,
                    };
                    self.linear_attn_prime_core_pad(e, la, g4, t, cache, il, Some(len_d))?
                }
            };
            let mut x1 = e.uninit(t * n_embd)?;
            e.add(&x, &mixed, &mut x1, t * n_embd)?;
            let mut z = e.uninit(t * n_embd)?;
            let mut zx16: Option<CudaSlice<u8>> = None;
            if f16fuse {
                let mut b16 = e.alloc_u8_uninit(t * n_embd * 2)?;
                e.rms_norm_f16out(&x1, layer.post_attn_norm.float_data(), &mut z, &mut b16, n_embd, t, eps)?;
                zx16 = Some(b16);
            } else {
                e.rms_norm(&x1, layer.post_attn_norm.float_data(), &mut z, n_embd, t, eps)?;
            }
            let ffn_out = match &layer.ffn {
                crate::hybrid::Ffn::Dense { ffn_gate, ffn_up, ffn_down } => {
                    let n_ff = ffn_gate.out_features();
                    let mut g2 = match &zx16 {
                        Some(xh) => e.matmul_group_xh(&[ffn_gate, ffn_up], &z, xh, t)?,
                        None => e.matmul_group(&[ffn_gate, ffn_up], &z, t)?,
                    };
                    let up = g2.pop().unwrap();
                    let gate = g2.pop().unwrap();
                    let mut act = e.uninit(t * n_ff)?;
                    // dense FFN keys off the SHEXP clamp array — see forward()'s note.
                    Self::ffn_act_lim(e, &self.cfg, &gate, &up, 1.0, 1.0,
                                      self.cfg.clamp_shexp_at(il as u32), &mut act, t * n_ff)?;
                    e.matmul(ffn_down, &act, t)?
                }
                crate::hybrid::Ffn::Moe(m) => self.moe_ffn_il_prefill(e, m, &z, t, il as u16)?,
            };
            let mut x2 = e.uninit(t * n_embd)?;
            e.add(&x1, &ffn_out, &mut x2, t * n_embd)?;
            x = x2;
        }
        // device-indexed TRUE last row (pads sit past it in a bucketed graph)
        if !crate::spec::spec_hpost() {
            e.row_gather_dev(&x, h_seed_out, len_d, n_embd)?;
        }
        let mut hn = e.uninit(t * n_embd)?;
        e.rms_norm(&x, self.output_norm.float_data(), &mut hn, n_embd, t, eps)?;
        if crate::spec::spec_hpost() {
            e.row_gather_dev(&hn, h_seed_out, len_d, n_embd)?;
        }
        let mut hlast = e.uninit(n_embd)?;
        e.row_gather_dev(&hn, &mut hlast, len_d, n_embd)?;
        let logits = e.matmul(&self.output, &hlast, 1)?;
        let nv = logits.len();
        e.copy_into(logits_out, 0, &logits, nv)?;
        Ok(())
    }

    fn step35_prime_batch_on() -> bool {
        std::env::var("MEMRA_STEP35_PRIME_BATCH").as_deref() != Ok("0")
    }

    /// Step35 cross-request prime range: weight-streaming work runs once at `m=sum(T)`;
    /// sequence-scoped attention/KV work stays on each request's own cache and positions.
    #[allow(clippy::too_many_arguments)]
    fn step35_prime_batch_layers(
        &self,
        e: &Engine,
        mut x: CudaSlice<f32>,
        lo: usize,
        hi: usize,
        ts: &[usize],
        offs: &[usize],
        pos_ds: &[CudaSlice<i32>],
        caches: &mut [&mut Cache],
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let n_embd = cfg.n_embd as usize;
        let eps = cfg.rms_eps;
        let b = ts.len();
        let total: usize = ts.iter().sum();
        let f16fuse = crate::f16_ffi::pp_f16_enabled() && total >= 16;

        let split = |e: &Engine, y: &CudaSlice<f32>, dim: usize|
                     -> Result<Vec<CudaSlice<f32>>, Box<dyn std::error::Error>> {
            let mut out = Vec::with_capacity(b);
            for s in 0..b {
                let mut ys = e.uninit(ts[s] * dim)?;
                e.copy_view_into(
                    &mut ys,
                    0,
                    &y.slice(offs[s] * dim..(offs[s] + ts[s]) * dim),
                    ts[s] * dim,
                )?;
                out.push(ys);
            }
            Ok(out)
        };

        for il in lo..hi {
            let layer = &self.layers[il];
            let Mixer::Full(fa) = &layer.mixer else {
                return Err(format!("step35 layer {il} is not full-attn — corrupt config").into());
            };

            let mut h = e.uninit(total * n_embd)?;
            let mut hx16 = e.alloc_u8_uninit(total * n_embd * 2)?;
            if f16fuse {
                e.rms_norm_f16out(
                    &x,
                    layer.attn_norm.float_data(),
                    &mut h,
                    &mut hx16,
                    n_embd,
                    total,
                    eps,
                )?;
            } else {
                e.rms_norm(
                    &x,
                    layer.attn_norm.float_data(),
                    &mut h,
                    n_embd,
                    total,
                    eps,
                )?;
            }

            // Q/K/V + gate projections and wo are batched weight streams. The step35 core
            // remains per sequence, so its partial RoPE, SWA view, KV append, and gate
            // application stay verbatim.
            let gate_w = fa
                .attn_gate
                .as_ref()
                .ok_or("step35 layer is missing attn_gate.weight (head-wise attention gate)")?;
            let mut g4 = if f16fuse {
                e.matmul_group_xh(&[&fa.wq, &fa.wk, &fa.wv, gate_w], &h, &hx16, total)?
            } else {
                e.matmul_group(&[&fa.wq, &fa.wk, &fa.wv, gate_w], &h, total)?
            };
            let gate = g4.pop().unwrap();
            let mut parts: Vec<Vec<CudaSlice<f32>>> =
                (0..b).map(|_| Vec::with_capacity(3)).collect();
            for (w, y) in [&fa.wq, &fa.wk, &fa.wv].iter().zip(g4) {
                for (s, ys) in split(e, &y, w.out_features())?.into_iter().enumerate() {
                    parts[s].push(ys);
                }
            }
            let gates = split(e, &gate, gate_w.out_features())?;
            let (hd, _, nh, _, _, _) = self.step35_geom(il);
            let mut ag_cat = e.uninit(total * nh * hd)?;
            for (s, (g3s, gate)) in parts.into_iter().zip(gates).enumerate() {
                let ag = self.step35_attn_pre_wo(
                    e,
                    fa,
                    g3s,
                    None,
                    Some(&gate),
                    &pos_ds[s],
                    ts[s],
                    Some(&mut *caches[s]),
                    il,
                    ts[s],
                )?;
                e.copy_into(
                    &mut ag_cat,
                    offs[s] * nh * hd,
                    &ag,
                    ts[s] * nh * hd,
                )?;
            }
            let mixed = e.matmul(&fa.wo, &ag_cat, total)?;

            let mut x1 = e.uninit(total * n_embd)?;
            let mut z = e.uninit(total * n_embd)?;
            let mut zx16 = e.alloc_u8_uninit(total * n_embd * 2)?;
            if f16fuse {
                e.add_rms_norm_f16out(
                    &x,
                    &mixed,
                    layer.post_attn_norm.float_data(),
                    &mut x1,
                    &mut z,
                    &mut zx16,
                    n_embd,
                    total,
                    eps,
                )?;
            } else {
                e.add(&x, &mixed, &mut x1, total * n_embd)?;
                e.rms_norm(
                    &x1,
                    layer.post_attn_norm.float_data(),
                    &mut z,
                    n_embd,
                    total,
                    eps,
                )?;
            }

            let ffn_out = match &layer.ffn {
                crate::hybrid::Ffn::Dense { ffn_gate, ffn_up, ffn_down } => {
                    let n_ff = ffn_gate.out_features();
                    let mut g2 = if f16fuse {
                        e.matmul_group_xh(&[ffn_gate, ffn_up], &z, &zx16, total)?
                    } else {
                        e.matmul_group(&[ffn_gate, ffn_up], &z, total)?
                    };
                    let up = g2.pop().unwrap();
                    let gate = g2.pop().unwrap();
                    let mut act = e.uninit(total * n_ff)?;
                    let d_lim = cfg.clamp_shexp_at(il as u32);
                    if Self::f16out_on(e, total) && cfg.m3.is_none() && d_lim.is_none() {
                        let mut a16 = e.alloc_u8_uninit(total * n_ff * 2)?;
                        e.silu_mul_f16out(&gate, &up, &mut act, &mut a16, total * n_ff)?;
                        match e.try_f16_gemm_pre(ffn_down, &a16, total)? {
                            Some(y) => y,
                            None => e.matmul(ffn_down, &act, total)?,
                        }
                    } else {
                        Self::ffn_act_lim(
                            e,
                            cfg,
                            &gate,
                            &up,
                            1.0,
                            1.0,
                            d_lim,
                            &mut act,
                            total * n_ff,
                        )?;
                        e.matmul(ffn_down, &act, total)?
                    }
                }
                crate::hybrid::Ffn::Moe(m) => self.moe_ffn_il(e, m, &z, total, il as u16)?,
            };
            let mut x2 = e.uninit(total * n_embd)?;
            e.add(&x1, &ffn_out, &mut x2, total * n_embd)?;
            x = x2;
        }
        Ok(x)
    }

    fn step35_prime_batch_epilogue(
        &self,
        e: &Engine,
        x: CudaSlice<f32>,
        ts: &[usize],
        offs: &[usize],
        caches: &mut [&mut Cache],
    ) -> Result<Vec<(Vec<f32>, CudaSlice<f32>, CudaSlice<f32>)>, Box<dyn std::error::Error>> {
        let n_embd = self.cfg.n_embd as usize;
        let total: usize = ts.iter().sum();
        let mut hn = e.uninit(total * n_embd)?;
        e.rms_norm(
            &x,
            self.output_norm.float_data(),
            &mut hn,
            n_embd,
            total,
            self.cfg.rms_eps,
        )?;

        let hidden_src = if crate::spec::spec_hpost() { &hn } else { &x };
        let mut out = Vec::with_capacity(ts.len());
        for s in 0..ts.len() {
            let mut hidden = e.uninit(ts[s] * n_embd)?;
            e.copy_view_into(
                &mut hidden,
                0,
                &hidden_src.slice(offs[s] * n_embd..(offs[s] + ts[s]) * n_embd),
                ts[s] * n_embd,
            )?;
            let last0 = (offs[s] + ts[s] - 1) * n_embd;
            let mut h_seed = e.uninit(n_embd)?;
            e.copy_view_into(
                &mut h_seed,
                0,
                &hidden_src.slice(last0..last0 + n_embd),
                n_embd,
            )?;
            // Exactness-first: the serial reference runs the output head at m=1.
            let mut hlast = e.uninit(n_embd)?;
            e.copy_view_into(
                &mut hlast,
                0,
                &hn.slice(last0..last0 + n_embd),
                n_embd,
            )?;
            let logits = e.dtoh(&e.matmul(&self.output, &hlast, 1)?)?;
            caches[s].pos += ts[s];
            out.push((logits, h_seed, hidden));
        }
        Ok(out)
    }

    fn step35_prime_cache_batch(
        &self,
        e: &Engine,
        prompts: &[&[u32]],
        caches: &mut [&mut Cache],
    ) -> Result<Vec<(Vec<f32>, CudaSlice<f32>, CudaSlice<f32>)>, Box<dyn std::error::Error>> {
        if !Self::step35_prime_batch_on() {
            return Err("step35 batched prime is disabled (MEMRA_STEP35_PRIME_BATCH=0)".into());
        }
        if caches.iter().any(|c| c.pos != 0) {
            return Err(
                "step35 batched prime currently supports complete fresh prompts only; \
                 continuation/tick chunks require per-request queued_after"
                    .into(),
            );
        }

        let ts: Vec<usize> = prompts.iter().map(|p| p.len()).collect();
        for &t in &ts {
            assert!(t >= PRIME_MIN_T, "step35 batched prime needs T >= {PRIME_MIN_T}");
        }
        for (s, c) in caches.iter().enumerate() {
            assert!(ts[s] <= c.max_ctx, "step35 batched prime exceeds cache max_ctx");
        }
        let offs: Vec<usize> = ts
            .iter()
            .scan(0usize, |a, &t| {
                let o = *a;
                *a += t;
                Some(o)
            })
            .collect();
        let total: usize = ts.iter().sum();
        let payload = total * self.cfg.n_embd as usize;
        let cat_tokens: Vec<u32> = prompts.iter().flat_map(|p| p.iter().copied()).collect();
        let positions: Vec<Vec<i32>> = ts
            .iter()
            .map(|&t| (0..t as i32).collect())
            .collect();
        let upload_positions = |e: &Engine|
                                -> Result<Vec<CudaSlice<i32>>, Box<dyn std::error::Error>> {
            positions
                .iter()
                .map(|p| e.htod_i32(p))
                .collect::<Result<_, _>>()
        };

        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            eprintln!(
                "[step35-prime-batch] first concat prime: B={} tokens={total}",
                prompts.len()
            );
        });

        let out = if !crate::pp::pp2_streams_off() && crate::pp::prime_pp_on() {
            if let Some(fence) = crate::pp::pp_cuts(self.layers.len()) {
                let rt = crate::pp::PpNRt::get(e)?;
                let n_st = fence.len() - 1;
                assert_eq!(rt.n_stages(), n_st, "step35 prime batch stage count mismatch");
                let caller_stream = e.stream();
                rt.fence_stages_behind(&caller_stream)?;

                let mut slot = {
                    let _st0 = rt.enter(0);
                    let e0 = rt.engine(0, e);
                    let pos_ds = upload_positions(e0)?;
                    let x = self.embed(e0, &cat_tokens)?;
                    let x = self.step35_prime_batch_layers(
                        e0,
                        x,
                        fence[0],
                        fence[1],
                        &ts,
                        &offs,
                        &pos_ds,
                        caches,
                    )?;
                    rt.tx(0, &x, payload)?
                };
                for s in 1..n_st - 1 {
                    let _st = rt.enter(s);
                    let es = rt.engine(s, e);
                    let pos_ds = upload_positions(es)?;
                    let x = rt.rx(s - 1, slot, payload)?;
                    let x = self.step35_prime_batch_layers(
                        es,
                        x,
                        fence[s],
                        fence[s + 1],
                        &ts,
                        &offs,
                        &pos_ds,
                        caches,
                    )?;
                    slot = rt.tx(s, &x, payload)?;
                }

                let _stl = rt.enter(n_st - 1);
                let el = rt.engine(n_st - 1, e);
                let pos_ds = upload_positions(el)?;
                let x = rt.rx(n_st - 2, slot, payload)?;
                let x = self.step35_prime_batch_layers(
                    el,
                    x,
                    fence[n_st - 1],
                    fence[n_st],
                    &ts,
                    &offs,
                    &pos_ds,
                    caches,
                )?;
                let out = self.step35_prime_batch_epilogue(el, x, &ts, &offs, caches)?;
                rt.publish_to(n_st - 1, &caller_stream)?;
                crate::pp::STEP35_PRIME_BATCH_SPLITS.fetch_add(
                    1,
                    std::sync::atomic::Ordering::Relaxed,
                );
                out
            } else {
                let pos_ds = upload_positions(e)?;
                let x = self.embed(e, &cat_tokens)?;
                let x = self.step35_prime_batch_layers(
                    e,
                    x,
                    0,
                    self.layers.len(),
                    &ts,
                    &offs,
                    &pos_ds,
                    caches,
                )?;
                self.step35_prime_batch_epilogue(e, x, &ts, &offs, caches)?
            }
        } else {
            let pos_ds = upload_positions(e)?;
            let x = self.embed(e, &cat_tokens)?;
            let x = self.step35_prime_batch_layers(
                e,
                x,
                0,
                self.layers.len(),
                &ts,
                &offs,
                &pos_ds,
                caches,
            )?;
            self.step35_prime_batch_epilogue(e, x, &ts, &offs, caches)?
        };
        crate::pp::STEP35_PRIME_BATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(out)
    }

    /// Cross-request BATCHED fresh prime (task #13, design in ARCHITECTURE-H100.md): the
    /// trunk's token-parallel ops (embed, norms, adds, ffn, projection GROUPS) run once on
    /// the CONCATENATION of B sequences — GEMMs at m = sum_T, the continuous-batching win
    /// the serving-lane bench attributed the remaining vLLM gap to. The stateful mixer
    /// CORES (QK-norm/RoPE/FA/append, conv/GDN scans) run per sequence on split projection
    /// buffers (D2D row copies; the mixers' own out-projections stay per-seq this
    /// increment). CONTINUATION primes (increment (b), 2026-07-30): cache.pos > 0 seqs
    /// batch the projections/FFN/lm_head exactly like fresh; the mixer cores take the
    /// per-seq CONTINUATION arms (Full: core_inner with carried pos_d + fa_prefill_view
    /// over the quantized past; Linear: the stateful pad_view twin — the same state
    /// carry the chunked single-seq prime rides). The fresh-only favl/gdn-vl fast paths
    /// stay byte-identical (gated on !carried). gemma4 models have no continuation
    /// prime (v0 monolithic fresh) — carried gemma4 batches return Err (caller falls
    /// back to single-chunk serving).
    /// NUMERIC CONFIG: a concat GEMM tiles K differently than per-seq GEMMs — same class
    /// as every prefill GEMM change; prime_batch_gate arbitrates (argmax + stream battery).
    pub fn prime_cache_batch(&self, e: &Engine, prompts: &[&[u32]], caches: &mut [&mut Cache])
                             -> Result<Vec<(Vec<f32>, CudaSlice<f32>, CudaSlice<f32>)>, Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let n_embd = cfg.n_embd as usize;
        let eps = cfg.rms_eps;
        let b = prompts.len();
        assert!(b >= 1 && b == caches.len());
        let pos0s: Vec<usize> = caches.iter().map(|c| c.pos).collect();
        let carried = pos0s.iter().any(|&p| p > 0);
        // gemma4: refuse UNCONDITIONALLY (2026-08-07, lane/gemma4-serve-gaps). The old guard
        // covered only `carried` — two concurrent FRESH gemma4 prompts batched into the
        // generic concat attn core below (uniform geometry, no per-layer swa window, no
        // softcapped head): compiles, runs, wrong logits. Same silent-wrong class as the
        // step35 refusal beneath. The per-sequence `gemma4_prime` is the supported prefill.
        if cfg.gemma4.is_some() {
            return Err("prime_cache_batch: gemma4 has no batched prime core (per-layer \
                        swa/global geometry, softcapped head) — use gemma4_prime per sequence".into());
        }
        // Step35 has a dedicated concat walk: the generic core below cannot express its
        // per-layer geometry/SWA/head gate, and under PP the dedicated path is stage-scoped.
        if cfg.step35.is_some() {
            return self.step35_prime_cache_batch(e, prompts, caches);
        }
        let ts: Vec<usize> = prompts.iter().map(|p| p.len()).collect();
        for &t in &ts { assert!(t >= PRIME_MIN_T, "prime_cache_batch needs T >= {PRIME_MIN_T}"); }
        for (s, c) in caches.iter().enumerate() {
            assert!(c.pos + ts[s] <= c.max_ctx, "prime_cache_batch: prompt exceeds cache max_ctx");
        }
        let total: usize = ts.iter().sum();
        let offs: Vec<usize> = ts.iter().scan(0usize, |a, &t| { let o = *a; *a += t; Some(o) }).collect();
        // per-seq positions (fresh: 0..T_s; continuation: pos0..pos0+T_s)
        let pos_ds: Vec<CudaSlice<i32>> = ts.iter().zip(&pos0s)
            .map(|(&t, &p0)| e.htod_i32(&(p0 as i32..(p0 + t) as i32).collect::<Vec<_>>()))
            .collect::<Result<_, _>>()?;
        // split a concat [total, dim] buffer into per-seq copies
        let split = |e: &Engine, y: &CudaSlice<f32>, dim: usize|
                     -> Result<Vec<CudaSlice<f32>>, Box<dyn std::error::Error>> {
            let mut out = Vec::with_capacity(b);
            for s in 0..b {
                let mut ys = e.uninit(ts[s] * dim)?;
                e.copy_view_into(&mut ys, 0, &y.slice(offs[s] * dim..(offs[s] + ts[s]) * dim), ts[s] * dim)?;
                out.push(ys);
            }
            Ok(out)
        };

        let cat_tokens: Vec<u32> = prompts.iter().flat_map(|p| p.iter().copied()).collect();
        let mut x = self.embed(e, &cat_tokens)?;   // [total, n_embd]
        for (il, layer) in self.layers.iter().enumerate() {
            let mut h = e.uninit(total * n_embd)?;
            let mut hx16 = e.alloc_u8_uninit(total * n_embd * 2)?;
            e.rms_norm_f16out(&x, layer.attn_norm.float_data(), &mut h, &mut hx16, n_embd, total, eps)?;
            // mixer: projection GROUP on the concat (m = total), stateful core per seq
            let mut mixed = e.uninit(total * n_embd)?;
            match &layer.mixer {
                Mixer::Full(fa) => {
                    let g3 = e.matmul_group_xh(&[&fa.wq, &fa.wk, &fa.wv], &h, &hx16, total)?;
                    // task #18 (attn side): the WHOLE attn core is varlen for fresh gated
                    // batches — split/QK-norm/RoPE/append (attn_pre_vl8, view inputs: the
                    // q/k/v split copies vanish) + ONE varlen FA. Per-block math identical
                    // everywhere (bit-gateable); MEMRA_FA_VL=0 or a non-bf16kv config falls
                    // back to the per-seq dispatch.
                    let (n_head, n_head_kv, head_dim) =
                        (self.cfg.n_head as usize, self.cfg.n_head_kv as usize, self.cfg.head_dim_k as usize);
                    let fa_scale = 1.0 / (head_dim as f32).sqrt();
                    let use_favl = !carried
                        && (2..=8).contains(&b)
                        && (head_dim == 256 || head_dim == 128)
                        && self.cfg.attn_out_gate()
                        && std::env::var("MEMRA_NOFA").is_err()
                        && std::env::var("MEMRA_FA_FLOOR").is_err()
                        && std::env::var("MEMRA_FA_PP_W2").as_deref() != Ok("1")
                        && std::env::var("MEMRA_FA_BF16KV").as_deref() != Ok("0")
                        && std::env::var("MEMRA_FA_VL").as_deref() != Ok("0");
                    if use_favl {
                        let (qf_w, kf_w, vf_w) =
                            (fa.wq.out_features(), fa.wk.out_features(), fa.wv.out_features());
                        struct APre {
                            q: CudaSlice<f32>, gate: Option<CudaSlice<f32>>,
                            qn: CudaSlice<f32>, kn: CudaSlice<f32>,
                        }
                        let mut aps = Vec::with_capacity(b);
                        for &t in ts.iter().take(b) {
                            aps.push(APre {
                                q: e.uninit(t * n_head * head_dim)?,
                                gate: Some(e.uninit(t * n_head * head_dim)?),
                                qn: e.uninit(t * n_head * head_dim)?,
                                kn: e.uninit(t * n_head_kv * head_dim)?,
                            });
                        }
                        let (kv_dim_k, kv_dim_v, ktb, vtb) = {
                            let kvl = caches[0].kv[il].as_ref().unwrap();
                            (kvl.kv_dim_k, kvl.kv_dim_v, kvl.k_tok_bytes, kvl.v_tok_bytes)
                        };
                        let pargs: Vec<crate::AttnPreVl> = (0..b).map(|s| {
                            let (o, t) = (offs[s], ts[s]);
                            let kvl = caches[s].kv[il].as_ref().unwrap();
                            assert!(kvl.len == 0 && kvl.len + t <= caches[s].max_ctx,
                                    "prime_cache_batch attn vl: fresh + capacity");
                            crate::AttnPreVl {
                                qf: e.addr_f32v(&g3[0].slice(o * qf_w..(o + t) * qf_w)),
                                kf: e.addr_f32v(&g3[1].slice(o * kf_w..(o + t) * kf_w)),
                                vf: e.addr_f32v(&g3[2].slice(o * vf_w..(o + t) * vf_w)),
                                q: e.addr_f32(&aps[s].q),
                                gate: e.addr_f32(aps[s].gate.as_ref().unwrap()),
                                qn: e.addr_f32(&aps[s].qn), kn: e.addr_f32(&aps[s].kn),
                                kc: e.addr_u8(&kvl.k), vc: e.addr_u8(&kvl.v),
                                t: t as i32, pad: 0,
                            }
                        }).collect();
                        e.attn_pre_vl8(&pargs, fa.q_norm.float_data(), fa.k_norm.float_data(),
                                       head_dim, self.cfg.rope_dim_count as usize, n_head, n_head_kv,
                                       self.cfg.rms_eps, self.cfg.rope_freq_base, 1.0,
                                       kv_dim_k, kv_dim_v, ktb, vtb)?;
                        for s in 0..b {
                            let kvl = caches[s].kv[il].as_mut().unwrap();
                            kvl.len += ts[s];
                            let new_len = kvl.len as i32;
                            e.set_i32_one(&mut kvl.len_d, new_len)?;
                        }
                        let mut attns = Vec::with_capacity(b);
                        let mut mirrors = Vec::with_capacity(b);
                        for &t in ts.iter().take(b) {
                            attns.push(e.uninit(t * n_head * head_dim)?);
                            let n = t * n_head_kv * head_dim;
                            mirrors.push((e.alloc_u8_uninit(n * 2)?, e.alloc_u8_uninit(n * 2)?));
                        }
                        // FA3 batched twin (round 31): TMA-swizzled wgmma vl when the
                        // promoted single-seq config is on; else the mma favl.
                        let fa3_on = match std::env::var("MEMRA_FA3").as_deref() {
                            Ok("0") => false,
                            Ok("1") => true,
                            _ => cfg!(memra_hopper_mma),
                        };
                        if fa3_on {
                            let mut q16s = Vec::with_capacity(b);
                            let mut v16s = Vec::with_capacity(b);
                            for s in 0..b {
                                let t = ts[s];
                                let mut q16 = e.alloc_u8_uninit(t * n_head * head_dim * 2)?;
                                e.f32_to_bf16_into(&aps[s].qn, &mut q16, t * n_head * head_dim)?;
                                let mut k16 = e.alloc_u8_uninit(t * n_head_kv * head_dim * 2)?;
                                e.f32_to_bf16_into(&aps[s].kn, &mut k16, t * n_head_kv * head_dim)?;
                                let mut v16 = e.alloc_u8_uninit(t * n_head_kv * head_dim * 2)?;
                                e.f32_to_bf16_v(&g3[2].slice(offs[s] * vf_w..(offs[s] + t) * vf_w),
                                                &mut v16, t * n_head_kv * head_dim)?;
                                q16s.push(q16);
                                v16s.push((k16, v16));
                            }
                            let mut qp = [core::ptr::null::<core::ffi::c_void>(); 8];
                            let mut kp = qp;
                            let mut vp = qp;
                            let mut op = [core::ptr::null_mut::<f32>(); 8];
                            let mut tsv = [0i32; 8];
                            for s in 0..b {
                                qp[s] = e.addr_u8(&q16s[s]) as *const core::ffi::c_void;
                                kp[s] = e.addr_u8(&v16s[s].0) as *const core::ffi::c_void;
                                vp[s] = e.addr_u8(&v16s[s].1) as *const core::ffi::c_void;
                                op[s] = e.addr_f32(&attns[s]) as *mut f32;
                                tsv[s] = ts[s] as i32;
                            }
                            let rc = unsafe {
                                crate::fa3_vl_raw(qp.as_ptr(), kp.as_ptr(), vp.as_ptr(), op.as_ptr(),
                                                  tsv.as_ptr(), b as i32, n_head as i32,
                                                  n_head_kv as i32, head_dim as i32, fa_scale,
                                                  e.stream().cu_stream() as *mut core::ffi::c_void)
                            };
                            if rc != 0 {
                                return Err(format!("memra_fa3_vl rc={rc}").into());
                            }
                        } else {
                            let fargs: Vec<crate::FaSeqVl> = (0..b).map(|s| crate::FaSeqVl {
                                q: e.addr_f32(&aps[s].qn), k16: e.addr_u8(&mirrors[s].0),
                                v16: e.addr_u8(&mirrors[s].1), o: e.addr_f32(&attns[s]),
                                kf: e.addr_f32(&aps[s].kn),
                                vf: e.addr_f32v(&g3[2].slice(offs[s] * vf_w..(offs[s] + ts[s]) * vf_w)),
                                t: ts[s] as i32, pad: 0,
                            }).collect();
                            e.fa_prefill_vl8(&fargs, head_dim, n_head, n_head_kv, fa_scale)?;
                        }
                        for (s, attn) in attns.into_iter().enumerate() {
                            let (attn_g, ag16) = self.full_attn_prime_post_fa(
                                e, attn, &aps[s].gate, ts[s], n_head, head_dim)?;
                            let mut done = false;
                            if let Some(xh) = &ag16 {
                                done = e.try_f16_gemm_pre_into_off(&fa.wo, xh, ts[s], &mut mixed, offs[s] * n_embd)?;
                            }
                            if !done {
                                let m = e.matmul(&fa.wo, &attn_g, ts[s])?;
                                e.copy_into(&mut mixed, offs[s] * n_embd, &m, ts[s] * n_embd)?;
                            }
                        }
                    } else {
                        let mut parts: Vec<Vec<CudaSlice<f32>>> = (0..b).map(|_| Vec::new()).collect();
                        for (w, y) in [&fa.wq, &fa.wk, &fa.wv].iter().zip(g3) {
                            for (s, ys) in split(e, &y, w.out_features())?.into_iter().enumerate() {
                                parts[s].push(ys);
                            }
                        }
                        for (s, g3s) in parts.into_iter().enumerate() {
                            // task #16 gather removal: wo writes into `mixed` at offs[s] directly.
                            let (attn_g, ag16) = self.full_attn_prime_core_inner(
                                e, fa, g3s, &pos_ds[s], ts[s], caches[s], il)?;
                            let mut done = false;
                            if let Some(xh) = &ag16 {
                                done = e.try_f16_gemm_pre_into_off(&fa.wo, xh, ts[s], &mut mixed, offs[s] * n_embd)?;
                            }
                            if !done {
                                let m = e.matmul(&fa.wo, &attn_g, ts[s])?;
                                e.copy_into(&mut mixed, offs[s] * n_embd, &m, ts[s] * n_embd)?;
                            }
                        }
                    }
                }
                Mixer::Mla(_) => crate::hybrid::mla_forward_unimplemented(),
                Mixer::Linear(la) => {
                    // task #16: NO split copies (cores read row-offset views of the concat
                    // outputs; out-GEMMs write into `mixed` at offs[s]). task #18: the core
                    // itself is BATCHED — per-seq prep/K1-K3, then ONE varlen K4 + ONE
                    // varlen K5 launch for all sequences.
                    let ws = [&la.wqkv, &la.wqkv_gate, &la.ssm_beta, &la.ssm_alpha];
                    let g4 = e.matmul_group_xh(&ws, &h, &hx16, total)?;
                    let outs = self.linear_attn_prime_core_batch(e, la, &g4, &offs, &ts, caches, il)?;
                    for (s, (gn, gn16)) in outs.into_iter().enumerate() {
                        let (o, t) = (offs[s], ts[s]);
                        let mut done = false;
                        if let Some(xh) = &gn16 {
                            done = e.try_f16_gemm_pre_into_off(&la.ssm_out, xh, t, &mut mixed, o * n_embd)?;
                        }
                        if !done {
                            let m = e.matmul(&la.ssm_out, &gn, t)?;
                            e.copy_into(&mut mixed, o * n_embd, &m, t * n_embd)?;
                        }
                    }
                }
            }
            let mut x1 = e.uninit(total * n_embd)?;
            let mut z = e.uninit(total * n_embd)?;
            let mut zx16 = e.alloc_u8_uninit(total * n_embd * 2)?;
            e.add_rms_norm_f16out(&x, &mixed, layer.post_attn_norm.float_data(),
                                  &mut x1, &mut z, &mut zx16, n_embd, total, eps)?;
            let ffn_out = match &layer.ffn {
                crate::hybrid::Ffn::Dense { ffn_gate, ffn_up, ffn_down } => {
                    let n_ff = ffn_gate.out_features();
                    let mut g2 = e.matmul_group_xh(&[ffn_gate, ffn_up], &z, &zx16, total)?;
                    let up = g2.pop().unwrap();
                    let gate = g2.pop().unwrap();
                    let mut act = e.uninit(total * n_ff)?;
                    // task #17 (batch trunk): silu twin emits the down GEMM's fp16 operand
                    // in-epilogue (nsys round-26: this trunk still paid 32 cvt passes).
                    // A clamped layer must skip the plain-SiLU twin (see prime_chunk's note).
                    let d_lim = self.cfg.clamp_shexp_at(il as u32);
                    if Self::f16out_on(e, total) && self.cfg.m3.is_none() && d_lim.is_none() {
                        let mut a16 = e.alloc_u8_uninit(total * n_ff * 2)?;
                        e.silu_mul_f16out(&gate, &up, &mut act, &mut a16, total * n_ff)?;
                        match e.try_f16_gemm_pre(ffn_down, &a16, total)? {
                            Some(y) => y,
                            None => e.matmul(ffn_down, &act, total)?,
                        }
                    } else {
                        Self::ffn_act_lim(e, &self.cfg, &gate, &up, 1.0, 1.0, d_lim,
                                          &mut act, total * n_ff)?;
                        e.matmul(ffn_down, &act, total)?
                    }
                }
                crate::hybrid::Ffn::Moe(m) => {
                    self.moe_ffn_il_prefill(e, m, &z, total, il as u16)?
                }
            };
            let mut x2 = e.uninit(total * n_embd)?;
            e.add(&x1, &ffn_out, &mut x2, total * n_embd)?;
            x = x2;
        }
        // epilogue per seq (identical math to prime_chunk: norm all rows, lm_head on last row)
        let mut hn = e.uninit(total * n_embd)?;
        e.rms_norm(&x, self.output_norm.float_data(), &mut hn, n_embd, total, eps)?;
        // batched lm_head (nsys round-26: B sequential m=1 matvecs re-read the 600MB
        // lm_head weight B times — 2.2ms at B=6): gather the B last rows and run ONE
        // m=B GEMM (f16 lane; falls back to the per-seq matvec when no mirror). The
        // f16-vs-mmvq logits delta is the prefill GEMM numeric class — prime_batch_gate's
        // argmax battery arbitrates, same as every other prefill GEMM change.
        let mut hcat = e.uninit(b * n_embd)?;
        for s in 0..b {
            let last0 = (offs[s] + ts[s] - 1) * n_embd;
            e.copy_view_into(&mut hcat, s * n_embd, &hn.slice(last0..last0 + n_embd), n_embd)?;
        }
        let logits_cat = if b >= 2 { e.try_f16_gemm(&self.output, &hcat, b)? } else { None };
        let logits_host: Option<Vec<f32>> = match &logits_cat {
            Some(lc) => Some(e.dtoh(lc)?),
            None => None,
        };
        let n_vocab = self.output.out_features();
        let mut hidden_all = if crate::spec::spec_hpost() {
            split(e, &hn, n_embd)?
        } else {
            split(e, &x, n_embd)?
        };
        let mut out = Vec::with_capacity(b);
        for s in 0..b {
            let last0 = (offs[s] + ts[s] - 1) * n_embd;
            let mut h_seed = e.uninit(n_embd)?;
            if !crate::spec::spec_hpost() {
                e.copy_view_into(&mut h_seed, 0, &x.slice(last0..last0 + n_embd), n_embd)?;
            } else {
                e.copy_view_into(&mut h_seed, 0, &hn.slice(last0..last0 + n_embd), n_embd)?;
            }
            let logits = match &logits_host {
                Some(lh) => lh[s * n_vocab..(s + 1) * n_vocab].to_vec(),
                None => {
                    let mut hlast = e.uninit(n_embd)?;
                    e.copy_view_into(&mut hlast, 0, &hn.slice(last0..last0 + n_embd), n_embd)?;
                    e.dtoh(&e.matmul(&self.output, &hlast, 1)?)?
                }
            };
            caches[s].pos += ts[s];
            out.push((logits, h_seed, hidden_all.remove(0)));
        }
        Ok(out)
    }

    /// `full_attn` (batched prefill mixer) + the cache side-effect: append the T post-RoPE K/V
    /// rows into the resident quantized KV cache (q8_0 K / q5_1 V) and advance len/len_d. Row
    /// bytes are BIT-IDENTICAL to the decode append (same per-warp quant kernel per row; the
    /// batched `append_kv_quantized_rows` runs that exact warp math on a (block, token) grid).
    /// The attention itself is unchanged prefill math (fa_prefill over the f32 K/V).
    ///
    /// `seq_end` = the ABSOLUTE end position of the WHOLE prime request (`cache.pos + prompt_len`
    /// measured BEFORE the chunk loop starts), i.e. a chunk-size-invariant property of the request.
    /// Only step35's SWA arm reads it (`step35_attn_pre_wo`'s doc note explains why keying on the
    /// chunk's own `t_kv` made the output depend on MEMRA_PRIME_CHUNK); every other arch ignores it.
    #[allow(clippy::too_many_arguments)]
    fn full_attn_prime(&self, e: &Engine, fa: &FullAttnLayer, h: &CudaSlice<f32>,
                       hx: Option<&CudaSlice<u8>>,
                       pos_d: &CudaSlice<i32>, t: usize, cache: &mut Cache, il: usize,
                       seq_end: usize)
                       -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        if self.cfg.step35.is_some() {
            return self.step35_attn_prime(e, fa, h, hx, pos_d, t, cache, il, seq_end);
        }
        // PROJ/CORE SPLIT (task #13, 2026-07-26): the q/k/v group projection is hoisted so
        // the cross-request batch driver can run it at m = sum_T over concatenated tokens;
        // this single-seq path composes proj+core identically (byte-for-byte the old body).
        // task #14: `hx` = the norm-fused fp16 twin of `h` (skips the convert launch).
        let g3 = match hx {
            Some(xh) => e.matmul_group_xh(&[&fa.wq, &fa.wk, &fa.wv], h, xh, t)?,
            None => e.matmul_group(&[&fa.wq, &fa.wk, &fa.wv], h, t)?,
        };
        self.full_attn_prime_core(e, fa, g3, pos_d, t, cache, il)
    }

    /// Everything after the q/k/v projections (split/QK-norm/RoPE/FA/gate/append + wo).
    /// `g3` = matmul_group([wq, wk, wv]) output rows for THIS sequence.
    /// Composes inner + wo (== the pre-split core; every existing caller unchanged).
    fn full_attn_prime_core(&self, e: &Engine, fa: &FullAttnLayer, g3: Vec<CudaSlice<f32>>,
                            pos_d: &CudaSlice<i32>, t: usize, cache: &mut Cache, il: usize)
                            -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let (attn_g, ag16) = self.full_attn_prime_core_inner(e, fa, g3, pos_d, t, cache, il)?;
        if let Some(xh) = &ag16 {
            if let Some(y) = e.try_f16_gemm_pre(&fa.wo, xh, t)? {
                return Ok(y);
            }
        }
        Ok(e.matmul(&fa.wo, &attn_g, t)?)
    }

    fn full_attn_prime_core_inner(&self, e: &Engine, fa: &FullAttnLayer, g3: Vec<CudaSlice<f32>>,
                            pos_d: &CudaSlice<i32>, t: usize, cache: &mut Cache, il: usize)
                            -> Result<(CudaSlice<f32>, Option<CudaSlice<u8>>), Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let n_head = cfg.n_head as usize;
        let n_head_kv = cfg.n_head_kv as usize;
        let head_dim = cfg.head_dim_k as usize;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let (pre, base_len) = self.full_attn_prime_pre_fa(e, fa, g3, pos_d, t, cache, il)?;
        let AttnPre { q, k, v, gate } = pre;
        let mut attn = e.uninit(t * n_head * head_dim)?;
        self.full_attn_prime_fa_dispatch(e, &q, &k, &v, &mut attn, base_len, t, cache, il,
                                         head_dim, n_head, n_head_kv, scale)?;
        self.full_attn_prime_post_fa(e, attn, &gate, t, n_head, head_dim)
    }

    /// task #18 (attn side): projections tail through KV append — everything before the
    /// attention kernel. Returns the post-rope q/k, v, optional out-gate, and the KV rows
    /// present BEFORE this chunk's append (base_len; 0 == fresh).
    #[allow(clippy::type_complexity)]
    fn full_attn_prime_pre_fa(&self, e: &Engine, fa: &FullAttnLayer, mut g3: Vec<CudaSlice<f32>>,
                            pos_d: &CudaSlice<i32>, t: usize, cache: &mut Cache, il: usize)
                            -> Result<(AttnPre, usize), Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let n_head = cfg.n_head as usize;
        let n_head_kv = cfg.n_head_kv as usize;
        let head_dim = cfg.head_dim_k as usize;
        let eps = cfg.rms_eps;

        // qwen35 fuses [q|gate] per head in wq (2*head_dim stride); M3/Hy3 have NO output gate
        // (attention_output_gate=false) — wq out = n_head*head_dim exactly, and q_gate_split
        // would read 2x out of bounds. `gated` keys both the split and the sigmoid epilogue.
        let gated = cfg.attn_out_gate();
        let v = g3.pop().unwrap();
        let mut k = g3.pop().unwrap();
        let qf = g3.pop().unwrap();
        let (mut q, gate) = if gated {
            let mut q = e.uninit(t * n_head * head_dim)?;
            let mut gate = e.uninit(t * n_head * head_dim)?;
            e.q_gate_split(&qf, &mut q, &mut gate, head_dim, n_head, t)?;
            (q, Some(gate))
        } else {
            (qf, None)
        };

        let mut qn = e.uninit(t * n_head * head_dim)?;
        e.rms_norm(&q, fa.q_norm.float_data(), &mut qn, head_dim, n_head * t, eps)?;
        q = qn;
        let mut kn = e.uninit(t * n_head_kv * head_dim)?;
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
                                       kvl.kv_dim_k, kvl.kv_dim_v, kvl.k_tok_bytes, kvl.v_tok_bytes,
                                       crate::Engine::kv_fp8_on())?;
            kvl.len += t;
            let new_len = kvl.len as i32;
            e.set_i32_one(&mut kvl.len_d, new_len)?;
        }

        let base_len = {
            let kvl = cache.kv[il].as_ref().unwrap();
            kvl.len - t   // KV rows present BEFORE this chunk's append above
        };
        Ok((AttnPre { q, k, v, gate }, base_len))
    }

    /// batched prefill attention. FRESH prime (no past KV): unchanged forward_last math over
    /// the f32 K/V of this batch. CONTINUATION chunk (past KV present): the chunk's queries
    /// must attend to [0 .. base+t) — run fa_prefill_view over the resident QUANTIZED cache
    /// (the spec-verify pattern; kernel's causal mask offsets by T_kv-T). Numerically this
    /// reads q8_0/q5_1-dequantized K/V for the past AND the current chunk — the same class as
    /// decode reading the cache; the run-gen/first-16 battery is the accuracy authority.
    #[allow(clippy::too_many_arguments)]
    fn full_attn_prime_fa_dispatch(&self, e: &Engine, q: &CudaSlice<f32>, k: &CudaSlice<f32>,
                            v: &CudaSlice<f32>, attn: &mut CudaSlice<f32>, base_len: usize,
                            t: usize, cache: &mut Cache, il: usize,
                            head_dim: usize, n_head: usize, n_head_kv: usize, scale: f32)
                            -> Result<(), Box<dyn std::error::Error>> {
        // GRAIN-FREE CHUNK-INVARIANCE FIX (lane/chunkinv-flip, 2026-08-05): the base_len == 0
        // f32 special case (fa_prefill over this batch's f32 K/V) is DROPPED. pre_fa appends
        // the chunk's quantized rows into cache.kv[il] BEFORE this dispatch, so chunk 0 can
        // attend through the quantized cache exactly like every later chunk (quantize-then-
        // attend). One numeric class for every row => the chunk size cannot decide where a
        // precision edge falls, so chunked prefill is reduction-order-stable with NO grain
        // knob (the stronger fix VERDICT.md filed; supersedes the MEMRA_PRIME_INVARIANT door's
        // pin-the-boundary approach).
        // MEMRA_PRIME_F32CHUNK0=1 is the ROLLBACK SEAM to the old arithmetic (chunk 0 attends
        // f32 K/V) — flags-doctrine rollback door AND the chunkinv gate's canary injection:
        // with the fix unconditional, only re-introducing the class edge can prove the gate
        // still detects the mechanism. Never on in a measured default run.
        if base_len == 0 && std::env::var("MEMRA_PRIME_F32CHUNK0").as_deref() == Ok("1") {
            if std::env::var("MEMRA_NOFA").is_ok() || !(head_dim == 256 || head_dim == 128) {
                e.sdpa_naive(q, k, v, attn, head_dim, n_head, n_head_kv, t, t, scale, true)?;
            } else {
                e.fa_prefill(q, k, v, attn, head_dim, n_head, n_head_kv, t, t, scale, true)?;
            }
            return Ok(());
        }
        let kvl = cache.kv[il].as_ref().unwrap();
        let t_kv = base_len + t;
        let k_view = e.view_u8(&kvl.k, t_kv * kvl.k_tok_bytes);
        let v_view = e.view_u8(&kvl.v, t_kv * kvl.v_tok_bytes);
        // fa_prefill_q/_qw twins are stamped for head_dim 256 (qwen35) and 128 (M3) only;
        // other dims (and MEMRA_NOFA) take the naive quantized-view SDPA — SAME cache bytes,
        // same numeric class, so the uniform contract holds on the fallback too.
        if std::env::var("MEMRA_NOFA").is_ok() || !(head_dim == 256 || head_dim == 128) {
            e.sdpa_naive_quantized_view(q, &k_view, &v_view, attn, head_dim, n_head,
                                        n_head_kv, t, t_kv, scale, true,
                                        kvl.k_tok_bytes, kvl.v_tok_bytes)?;
            return Ok(());
        }
        // ARC B (2026-07-05): dequant-once workspace, DEFAULT ON. fa_prefill_q's inline
        // dequant re-reads+re-dequants the whole quantized KV stream from every one of the
        // T/64 x n_head CTAs (64x+ redundant at chunk=4096; 30.5% of the 32k prime wall).
        // fa_prefill_view_ws dequants K/V ONCE into a resident bf16 workspace then runs the
        // bit-identical bf16 twin (fa_prefill_qw) — same staged values, same FP order, token-
        // identical output (gate: MEMRA_PRIME_CHUNK=4096 ws-on vs ws-off vs monolithic).
        // MEMRA_PRIME_DEQW=0 reverts to the inline-dequant kernel.
        let deqw = std::env::var("MEMRA_PRIME_DEQW").map(|v| v != "0").unwrap_or(true);
        if deqw {
            e.fa_prefill_view_ws(q, &k_view, &v_view, attn, head_dim, n_head, n_head_kv,
                                 t, t_kv, scale, true, kvl.k_tok_bytes, kvl.v_tok_bytes,
                                 crate::Engine::kv_fp8_on())?;
        } else {
            e.fa_prefill_view(q, &k_view, &v_view, attn, head_dim, n_head, n_head_kv,
                              t, t_kv, scale, true, kvl.k_tok_bytes, kvl.v_tok_bytes,
                              crate::Engine::kv_fp8_on())?;
        }
        Ok(())
    }

    /// task #17: sig_mul_f16out fuses [sigmoid + mul + f16 convert] into one launch
    /// (bit-identical composition) and hands wo its fp16 operand directly.
    fn full_attn_prime_post_fa(&self, e: &Engine, attn: CudaSlice<f32>,
                            gate: &Option<CudaSlice<f32>>, t: usize,
                            n_head: usize, head_dim: usize)
                            -> Result<(CudaSlice<f32>, Option<CudaSlice<u8>>), Box<dyn std::error::Error>> {
        let (attn_g, ag16) = match gate {
            Some(gate) => {
                let n = t * n_head * head_dim;
                let mut ag = e.uninit(n)?;
                if Self::f16out_on(e, t) {
                    let mut a16 = e.alloc_u8_uninit(n * 2)?;
                    e.sig_mul_f16out(&attn, gate, &mut ag, &mut a16, n)?;
                    (ag, Some(a16))
                } else {
                    let mut gsig = e.uninit(n)?;
                    e.sigmoid(gate, &mut gsig, n)?;
                    e.mul(&attn, &gsig, &mut ag, n)?;
                    (ag, None)
                }
            }
            None => (attn, None),
        };
        Ok((attn_g, ag16))
    }

    /// STATEFUL batched linear-attention prime: `linear_attn`'s prefill-dispatch pass (normal
    /// `e.matmul` — GEMM at m>=16 — plus the prefill repack/L2/glog kernels) but with the state
    /// carried THROUGH the cache like the spec verify does: carried-ring conv
    /// (ssm_conv1d_tm_state writes the final ring back) + ONE gdn_scan from cache.recur[il]'s
    /// current state (zero at a fresh prime) whose final state ping-pongs back into the cache.
    /// Wiring mirrors `linear_attn_verify_t` (spec.rs); dispatch mirrors `linear_attn` (prefill).
    fn linear_attn_prime(&self, e: &Engine, la: &LinearAttnLayer, h: &CudaSlice<f32>,
                         hx: Option<&CudaSlice<u8>>, t: usize,
                         cache: &mut Cache, il: usize)
                         -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        // PROJ/CORE SPLIT (task #13): see full_attn_prime — same hoist for the GDN 4-tuple.
        let ws = [&la.wqkv, &la.wqkv_gate, &la.ssm_beta, &la.ssm_alpha];
        let g4 = match hx {
            Some(xh) => e.matmul_group_xh(&ws, h, xh, t)?,
            None => e.matmul_group(&ws, h, t)?,
        };
        self.linear_attn_prime_core(e, la, g4, t, cache, il)
    }

    /// Everything after the GDN 4-tuple projections (conv/chunk stack/gated norm + ssm_out).
    fn linear_attn_prime_core(&self, e: &Engine, la: &LinearAttnLayer, mut g4: Vec<CudaSlice<f32>>,
                              t: usize, cache: &mut Cache, il: usize)
                              -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        self.linear_attn_prime_core_pad(e, la, g4.drain(..).collect(), t, cache, il, None)
    }

    /// task #14: `pad_len` = device true length for PADDED prime graphs. Pads become
    /// identity GDN steps (gdn_pad_mask zeroes beta/g_log past the true length) and the
    /// conv ring writes back from the true tail. None = classic path, byte-identical.
    #[allow(clippy::too_many_arguments)]
    fn linear_attn_prime_core_pad_inner(&self, e: &Engine, la: &LinearAttnLayer, mut g4: Vec<CudaSlice<f32>>,
                              t: usize, cache: &mut Cache, il: usize,
                              pad_len: Option<&CudaSlice<i32>>)
                              -> Result<(CudaSlice<f32>, Option<CudaSlice<u8>>), Box<dyn std::error::Error>> {
        // shim over the view twin (task #16): full-range views of the owned buffers.
        let ssm = self.cfg.ssm.as_ref().unwrap();
        let d_state = ssm.state_size as usize;
        let num_k = ssm.group_count as usize;
        let num_v = ssm.time_step_rank as usize;
        let key_dim = d_state * num_k;
        let value_dim = d_state * num_v;
        let conv_dim = key_dim * 2 + value_dim;
        let alpha = g4.pop().unwrap();                   // [T, num_v]
        let beta_raw = g4.pop().unwrap();                // [T, num_v]
        let z = g4.pop().unwrap();                       // [T, value_dim]
        let qkv_mixed = g4.pop().unwrap();               // [T, conv_dim] token-major
        self.linear_attn_prime_core_pad_view(
            e, la,
            &qkv_mixed.slice(0..t * conv_dim), &z.slice(0..t * value_dim),
            &beta_raw.slice(0..t * num_v), &alpha.slice(0..t * num_v),
            t, cache, il, pad_len)
    }

    /// task #18: the GDN prep stage (conv + repack + l2 x2 + sigmoid + glog + pad-mask) —
    /// shared verbatim by the per-seq scan path and the varlen batched path.
    #[allow(clippy::too_many_arguments)]
    fn linear_attn_gdn_prep(&self, e: &Engine, la: &LinearAttnLayer,
                            qkv_mixed: &cudarc::driver::CudaView<f32>,
                            beta_raw: &cudarc::driver::CudaView<f32>,
                            alpha: &cudarc::driver::CudaView<f32>,
                            t: usize, cache: &mut Cache, il: usize,
                            pad_len: Option<&CudaSlice<i32>>)
                            -> Result<GdnPrep, Box<dyn std::error::Error>> {
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
        debug_assert!(t >= d_conv - 1, "stateful conv needs T >= pad (PRIME_MIN_T gates)");

        // conv with CARRIED ring state + ring roll (state read + final-window write-back).
        // task #18 conv-fuse (default ON): conv + SiLU + repack in one pass — the 67MB
        // conv_out intermediate and its transposed re-read disappear (values bit-identical).
        // MEMRA_CONV_FUSE=0 reverts to the two-kernel chain.
        let rl = cache.recur[il].as_mut().unwrap();
        let hk = Self::gdn_hk(e, t, num_v, num_k);
        let conv_fuse = std::env::var("MEMRA_CONV_FUSE").as_deref() != Ok("0");
        let hk = if conv_fuse { hk } else { num_v };   // de-broadcast rides the fused conv
        let mut q_g = e.uninit(d_state * hk * t)?;
        let mut k_g = e.uninit(d_state * hk * t)?;
        let mut v_g = e.uninit(d_state * num_v * t)?;
        if conv_fuse {
            e.ssm_conv1d_gdn_state_pad(qkv_mixed, &mut rl.conv_state, la.ssm_conv1d.float_data(),
                                  &mut q_g, &mut k_g, &mut v_g,
                                  conv_dim, t, d_conv, d_state, num_v, num_k, key_dim, hk, pad_len)?;
        } else {
            let mut conv_out = e.uninit(conv_dim * t)?;      // [conv_dim, T] channel-major, SiLU
            e.ssm_conv1d_tm_state_pad_v(qkv_mixed, &mut rl.conv_state, la.ssm_conv1d.float_data(),
                                  &mut conv_out, conv_dim, t, d_conv, pad_len)?;
            e.qkv_to_gdn_repack(&conv_out, &mut q_g, &mut k_g, &mut v_g, d_state, num_v, num_k, key_dim, t)?;
        }
        let mut q_l2 = e.uninit(d_state * hk * t)?;
        // mirror-fold (round 35): q's bf16 twin (wgmma K45/K2 A-operand) in-epilogue too.
        // Emitted only where a consumer exists (the wgmma config) — on other arches the
        // alloc + epilogue stores would be pure waste.
        let qb16 = if Engine::l2_v2_on(d_state) && e.gdn_wgmma_on(32) {
            let mut qb = e.alloc_u8_uninit(d_state * hk * t * 2)?;
            e.l2_norm_pp(&q_g, &mut q_l2, Some(&mut qb), d_state, hk * t, eps)?;
            Some(qb)
        } else {
            e.l2_norm_pp(&q_g, &mut q_l2, None, d_state, hk * t, eps)?;
            None
        };
        let mut k_l2 = e.uninit(d_state * hk * t)?;
        // mirror-fold: emit k's bf16 twin (the chunked-scan kb16 mirror) in-epilogue
        let kb16 = if Engine::l2_v2_on(d_state) {
            let mut kb = e.alloc_u8_uninit(d_state * hk * t * 2)?;
            e.l2_norm_pp(&k_g, &mut k_l2, Some(&mut kb), d_state, hk * t, eps)?;
            Some(kb)
        } else {
            e.l2_norm_pp(&k_g, &mut k_l2, None, d_state, hk * t, eps)?;
            None
        };
        let mut beta = e.uninit(t * num_v)?;
        e.sigmoid_v(beta_raw, &mut beta, t * num_v)?;
        let mut g_log = e.uninit(t * num_v)?;
        e.gdn_glog_v(alpha, la.ssm_dt.float_data(), la.ssm_a.float_data(), &mut g_log, num_v, t)?;
        if let Some(len_d) = pad_len {
            e.gdn_pad_mask(&mut beta, &mut g_log, len_d, num_v, t)?;
        }
        Ok(GdnPrep { hk, q_l2, k_l2, v_g, beta, g_log, kb16, qb16 })
    }

    /// task #18: batched GDN mixer core — per-seq prep + K1-K3, then ONE varlen K4 and
    /// ONE varlen K5 launch for all B sequences (full-machine occupancy vs B underfilled
    /// per-seq trains). Per-block math identical to the per-seq path (bit-gateable);
    /// MEMRA_GDN_VL=0 or a non-mma/non-C32 config falls back to the per-seq loop.
    #[allow(clippy::too_many_arguments)]
    fn linear_attn_prime_core_batch(&self, e: &Engine, la: &LinearAttnLayer,
                                    g4: &[CudaSlice<f32>], offs: &[usize], ts: &[usize],
                                    caches: &mut [&mut Cache], il: usize)
                                    -> Result<Vec<(CudaSlice<f32>, Option<CudaSlice<u8>>)>, Box<dyn std::error::Error>> {
        let ssm = self.cfg.ssm.as_ref().unwrap();
        let d_state = ssm.state_size as usize;
        let num_k = ssm.group_count as usize;
        let num_v = ssm.time_step_rank as usize;
        let key_dim = d_state * num_k;
        let value_dim = d_state * num_v;
        let conv_dim = key_dim * 2 + value_dim;
        let eps = self.cfg.rms_eps;
        let scale = 1.0 / (d_state as f32).sqrt();
        let b = ts.len();
        let c = Engine::gdn_chunk_size();
        // CONTINUATION batches (increment (b)) take the per-seq stateful pad_view twin
        // below — the varlen K4/K5 chain is fresh-only (zero initial state assumed).
        let carried = caches.iter().any(|c| c.pos > 0);
        let use_vl = !carried
            && (2..=8).contains(&b)
            && Engine::gdn_chunked_enabled() && ts.iter().all(|&t| t >= 16)
            && e.gdn_mma_enabled(c)
            && std::env::var("MEMRA_GDN_VL").as_deref() != Ok("0");
        if !use_vl {
            return (0..b).map(|s| {
                let (o, t) = (offs[s], ts[s]);
                self.linear_attn_prime_core_pad_view(
                    e, la,
                    &g4[0].slice(o * conv_dim..(o + t) * conv_dim),
                    &g4[1].slice(o * value_dim..(o + t) * value_dim),
                    &g4[2].slice(o * num_v..(o + t) * num_v),
                    &g4[3].slice(o * num_v..(o + t) * num_v),
                    t, caches[s], il, None)
            }).collect();
        }
        // increment 3: the ENTIRE core is varlen — allocs only per seq, then 13 launches
        // total for the whole batch: [conv, ring, repack, l2(q+k), gate-prep] +
        // [k-mirror, K1, K2, K3, w-mirror, K4, K5] + [gated-norm tail].
        struct SeqBufs {
            conv_out: CudaSlice<f32>, q_g: CudaSlice<f32>, k_g: CudaSlice<f32>, v_g: CudaSlice<f32>,
            q_l2: CudaSlice<f32>, k_l2: CudaSlice<f32>, beta: CudaSlice<f32>, g_log: CudaSlice<f32>,
            gn: CudaSlice<f32>, gn16: CudaSlice<u8>,
        }
        let d_conv = ssm.conv_kernel as usize;
        let f16o = Self::f16out_on(e, 16);
        let hk = Self::gdn_hk(e, 16, num_v, num_k);   // vl path is always chunked+mma
        let mut sb = Vec::with_capacity(b);
        let mut pres = Vec::with_capacity(b);
        for &t in ts.iter().take(b) {
            sb.push(SeqBufs {
                conv_out: e.uninit(conv_dim * t)?,
                q_g: e.uninit(d_state * hk * t)?,
                k_g: e.uninit(d_state * hk * t)?,
                v_g: e.uninit(d_state * num_v * t)?,
                q_l2: e.uninit(d_state * hk * t)?,
                k_l2: e.uninit(d_state * hk * t)?,
                beta: e.uninit(t * num_v)?,
                g_log: e.uninit(t * num_v)?,
                gn: e.uninit(d_state * num_v * t)?,
                gn16: e.alloc_u8_uninit(d_state * num_v * t * 2)?,
            });
            pres.push(e.gdn_chunk_alloc(num_v, t, c, hk)?);
        }
        let prep_args: Vec<crate::GdnPrepVl> = (0..b).map(|s| {
            let (o, t) = (offs[s], ts[s]);
            let rl = caches[s].recur[il].as_ref().unwrap();
            crate::GdnPrepVl {
                qkv: e.addr_f32v(&g4[0].slice(o * conv_dim..(o + t) * conv_dim)),
                conv_state: e.addr_f32(&rl.conv_state),
                conv_out: e.addr_f32(&sb[s].conv_out),
                q_g: e.addr_f32(&sb[s].q_g), k_g: e.addr_f32(&sb[s].k_g), v_g: e.addr_f32(&sb[s].v_g),
                q_l2: e.addr_f32(&sb[s].q_l2), k_l2: e.addr_f32(&sb[s].k_l2),
                beta_raw: e.addr_f32v(&g4[2].slice(o * num_v..(o + t) * num_v)),
                alpha: e.addr_f32v(&g4[3].slice(o * num_v..(o + t) * num_v)),
                beta: e.addr_f32(&sb[s].beta), g_log: e.addr_f32(&sb[s].g_log),
                o: e.addr_f32(&pres[s].o),
                z: e.addr_f32v(&g4[1].slice(o * value_dim..(o + t) * value_dim)),
                gn: e.addr_f32(&sb[s].gn), gn16: e.addr_u8(&sb[s].gn16),
                kb16: if Engine::l2_v2_on(d_state) { e.addr_u8(&pres[s].kb16) } else { 0 },
                qb16: if Engine::l2_v2_on(d_state) && e.gdn_wgmma_on(c) { e.addr_u8(&pres[s].qb16) } else { 0 },
                t: t as i32, pad: 0,
            }
        }).collect();
        let args: Vec<crate::GdnSeqVl> = (0..b).map(|s| {
            let rl = caches[s].recur[il].as_ref().unwrap();
            crate::GdnSeqVl {
                kb16: e.addr_u8(&pres[s].kb16), gcum: e.addr_f32(&pres[s].gcum),
                beta: e.addr_f32(&sb[s].beta), u: e.addr_f32(&pres[s].u),
                wb16: e.addr_u8(&pres[s].wb16), y: e.addr_u8(&pres[s].y16),
                ssnap: e.addr_u8(&pres[s].ssnap16),
                state_in: e.addr_f32(&rl.ssm_state), state_out: e.addr_f32(&rl.ssm_state_alt),
                q: e.addr_f32(&sb[s].q_l2), p: e.addr_f32(&pres[s].p),
                o: e.addr_f32(&pres[s].o),
                k: e.addr_f32(&sb[s].k_l2), v: e.addr_f32(&sb[s].v_g),
                g: e.addr_f32(&sb[s].g_log), a: e.addr_f32(&pres[s].a),
                w: e.addr_f32(&pres[s].w),
                t: ts[s] as i32, nc: pres[s].nc as i32,
            }
        }).collect();
        e.gdn_prep_vl8(&prep_args, la.ssm_conv1d.float_data(), la.ssm_dt.float_data(),
                       la.ssm_a.float_data(), conv_dim, d_conv, d_state, num_v, num_k, key_dim, hk, eps)?;
        // mirror-fold (round 27): l2 v2 emits kb16 in-epilogue; K3's store emits wb16 —
        // both standalone mirror launches vanish on the default config.
        if !Engine::l2_v2_on(d_state) {
            e.gdn_mirror_vl8(&args, num_v, 0, hk)?;
        }
        // task #22: wgmma-fused vl twins — qb16 mirrors + GdnWVl8 side-struct.
        let wq8: Option<crate::GdnWVl8> = if e.gdn_wgmma_on(c) {
            // qb16 emitted by the vl l2 mirror-fold when l2-v2 serves; bulk cvt otherwise
            if !Engine::l2_v2_on(d_state) {
                for s in 0..b {
                    e.f32_to_bf16_into(&sb[s].q_l2, &mut pres[s].qb16, d_state * hk * ts[s])?;
                }
            }
            let mut wa = [crate::GdnWVl::default(); 8];
            for s in 0..b {
                wa[s] = crate::GdnWVl { qb16: e.addr_u8(&pres[s].qb16), pb16: e.addr_u8(&pres[s].pb16) };
            }
            Some(crate::GdnWVl8(wa))
        } else { None };
        e.gdn_chunk_k123_vl8(&args, num_v, hk, wq8.as_ref())?;
        e.gdn_chunk_vl8(&args, num_v, scale, hk, wq8.as_ref())?;
        if f16o {
            e.gdn_tail_vl8(&prep_args, la.ssm_norm.float_data(), d_state, num_v, eps)?;
        }
        // per-seq state swap (+ non-f16out tail fallback)
        let mut out = Vec::with_capacity(b);
        for (s, bufs) in sb.into_iter().enumerate() {
            let rl = caches[s].recur[il].as_mut().unwrap();
            std::mem::swap(&mut rl.ssm_state, &mut rl.ssm_state_alt);
            let (o, t) = (offs[s], ts[s]);
            let SeqBufs { mut gn, gn16, .. } = bufs;
            if f16o {
                out.push((gn, Some(gn16)));
            } else {
                let z_v = g4[1].slice(o * value_dim..(o + t) * value_dim);
                e.gated_rmsnorm_zv(&pres[s].o, la.ssm_norm.float_data(), &z_v, &mut gn,
                                   d_state, num_v * t, eps)?;
                out.push((gn, None));
            }
        }
        Ok(out)
    }

    /// task #16: view-consuming GDN prime core — the batched prime hands row-offset
    /// views of the CONCAT projection outputs directly (no per-seq split copies).
    /// Same kernels, same values, byte-identical to the Vec shim above.
    #[allow(clippy::too_many_arguments)]
    fn linear_attn_prime_core_pad_view(&self, e: &Engine, la: &LinearAttnLayer,
                              qkv_mixed: &cudarc::driver::CudaView<f32>,
                              z: &cudarc::driver::CudaView<f32>,
                              beta_raw: &cudarc::driver::CudaView<f32>,
                              alpha: &cudarc::driver::CudaView<f32>,
                              t: usize, cache: &mut Cache, il: usize,
                              pad_len: Option<&CudaSlice<i32>>)
                              -> Result<(CudaSlice<f32>, Option<CudaSlice<u8>>), Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let ssm = cfg.ssm.as_ref().unwrap();
        let d_state = ssm.state_size as usize;       // 128
        let num_v = ssm.time_step_rank as usize;     // 32
        let eps = cfg.rms_eps;
        let scale = 1.0 / (d_state as f32).sqrt();

        let prep = self.linear_attn_gdn_prep(e, la, qkv_mixed, beta_raw, alpha, t, cache, il, pad_len)?;

        // ONE gdn_scan over T from the cache's CURRENT state (zero at fresh prime); the final
        // state lands in the spare buffer and ping-pongs back (stable resident pointers, the
        // decode-determinism discipline from linear_attn_decode_inner). A4: `gdn_scan_prefill`
        // dispatches the chunked WY form under MEMRA_GDN_CHUNKED (prefill-only seam; decode +
        // verify keep the sequential kernel).
        let mut o = e.uninit(d_state * num_v * t)?;
        let rl = cache.recur[il].as_mut().unwrap();
        {
            let crate::cache::RecurLayer { ssm_state, ssm_state_alt, .. } = rl;
            e.gdn_scan_prefill(&prep.q_l2, &prep.k_l2, &prep.v_g, &prep.g_log, &prep.beta,
                               prep.kb16.as_ref(), prep.qb16.as_ref(), ssm_state, ssm_state_alt, &mut o, num_v, t, scale,
                               prep.hk)?;
        }
        std::mem::swap(&mut rl.ssm_state, &mut rl.ssm_state_alt);

        // gated RMSNorm + out projection (prefill dispatch). task #17: the f16out twin also
        // emits the ssm_out GEMM's fp16 operand in-epilogue (kills the standalone convert).
        let mut gn = e.uninit(d_state * num_v * t)?;
        let gn16 = if Self::f16out_on(e, t) {
            let mut g16 = e.alloc_u8_uninit(d_state * num_v * t * 2)?;
            e.gated_rmsnorm_f16out_zv(&o, la.ssm_norm.float_data(), z, &mut gn, &mut g16,
                                      d_state, num_v * t, eps)?;
            Some(g16)
        } else {
            e.gated_rmsnorm_zv(&o, la.ssm_norm.float_data(), z, &mut gn, d_state, num_v * t, eps)?;
            None
        };
        Ok((gn, gn16))
    }

    /// Task #15 core-split wrapper: composes inner + ssm_out (== the old core).
    #[allow(clippy::too_many_arguments)]
    fn linear_attn_prime_core_pad(&self, e: &Engine, la: &LinearAttnLayer, g4: Vec<CudaSlice<f32>>,
                              t: usize, cache: &mut Cache, il: usize,
                              pad_len: Option<&CudaSlice<i32>>)
                              -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let (gn, gn16) = self.linear_attn_prime_core_pad_inner(e, la, g4, t, cache, il, pad_len)?;
        if let Some(xh) = &gn16 {
            if let Some(y) = e.try_f16_gemm_pre(&la.ssm_out, xh, t)? {
                return Ok(y);
            }
        }
        Ok(e.matmul(&la.ssm_out, &gn, t)?)
    }

    /// Full-attention mixer with QK-norm, partial RoPE, sigmoid output gate (qwen35 :257-336).
    ///
    /// `il` = layer index: step35 needs it (per-layer n_head / rope width / window / gate) and
    /// routes to its own mixer. Every other arch ignores it (uniform geometry).
    pub fn full_attn(&self, e: &Engine, fa: &FullAttnLayer, h: &CudaSlice<f32>, pos_d: &CudaSlice<i32>, t: usize, il: usize)
                 -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        if self.cfg.step35.is_some() {
            return self.step35_attn(e, fa, h, pos_d, t, il);
        }
        let cfg = &self.cfg;
        let _n_embd = cfg.n_embd as usize;
        let n_head = cfg.n_head as usize;
        let n_head_kv = cfg.n_head_kv as usize;
        let head_dim = cfg.head_dim_k as usize;
        let eps = cfg.rms_eps;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // qwen35: wq output = head_dim*2*n_head (fused [q|gate] per head). M3/Hy3: NO output
        // gate — wq out = n_head*head_dim, no split (see prime-path note).
        let gated = cfg.attn_out_gate();
        // grouped: one f16 activation convert feeds q/k/v (matmul_group)
        let mut g3 = e.matmul_group(&[&fa.wq, &fa.wk, &fa.wv], h, t)?;
        let v = g3.pop().unwrap();
        let mut k = g3.pop().unwrap();
        let qf = g3.pop().unwrap();
        let (mut q, gate) = if gated {
            let mut q = e.uninit(t * n_head * head_dim)?;
            let mut gate = e.uninit(t * n_head * head_dim)?;
            e.q_gate_split(&qf, &mut q, &mut gate, head_dim, n_head, t)?;
            (q, Some(gate))
        } else {
            (qf, None)
        };

        // QK-norm (per head_dim row), then partial RoPE.
        let mut qn = e.uninit(t * n_head * head_dim)?;
        e.rms_norm(&q, fa.q_norm.float_data(), &mut qn, head_dim, n_head * t, eps)?;
        q = qn;
        let mut kn = e.uninit(t * n_head_kv * head_dim)?;
        e.rms_norm(&k, fa.k_norm.float_data(), &mut kn, head_dim, n_head_kv * t, eps)?;
        k = kn;
        let rope_dims = cfg.rope_dim_count as usize;
        e.rope_neox(&mut q, pos_d, head_dim, rope_dims, n_head, t, cfg.rope_freq_base, 1.0)?;
        e.rope_neox(&mut k, pos_d, head_dim, rope_dims, n_head_kv, t, cfg.rope_freq_base, 1.0)?;

        // SDPA
        let mut attn = e.uninit(t * n_head * head_dim)?;
        // hand-written FlashAttention prefill (head_dim 256/128 stamped twins). MEMRA_NOFA
        // falls back to naive sdpa.
        if std::env::var("MEMRA_NOFA").is_ok() || !(head_dim == 256 || head_dim == 128) {
            // head_dim gate: see prime-path note (fa_prefill is stamped at 256 and 128 only).
            e.sdpa_naive(&q, &k, &v, &mut attn, head_dim, n_head, n_head_kv, t, t, scale, true)?;
        } else {
            e.fa_prefill(&q, &k, &v, &mut attn, head_dim, n_head, n_head_kv, t, t, scale, true)?;
        }

        // output gate: attn * sigmoid(gate) — qwen35 only (M3 has no gate).
        let attn_g = match &gate {
            Some(gate) => {
                let mut gsig = e.uninit(t * n_head * head_dim)?;
                e.sigmoid(gate, &mut gsig, t * n_head * head_dim)?;
                let mut ag = e.uninit(t * n_head * head_dim)?;
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
        // grouped: one f16 activation convert feeds all four projections (matmul_group)
        let mut g4 = e.matmul_group(&[&la.wqkv, &la.wqkv_gate, &la.ssm_beta, &la.ssm_alpha], h, t)?;
        let alpha = g4.pop().unwrap();                   // [T, num_v]
        let beta_raw = g4.pop().unwrap();                // [T, num_v]
        let z = g4.pop().unwrap();                       // [T, value_dim]
        let qkv_mixed = g4.pop().unwrap();               // [T, conv_dim] token-major

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
        let mut q_l2 = e.uninit(d_state * num_v * t)?;
        e.l2_norm(&q_g, &mut q_l2, d_state, num_v * t, eps)?;
        let mut k_l2 = e.uninit(d_state * num_v * t)?;
        e.l2_norm(&k_g, &mut k_l2, d_state, num_v * t, eps)?;
        let v_gd = v_g;

        // beta = sigmoid(beta_raw) ; g_log = a * softplus(alpha + dt). Both need [num_v, T] layout
        // (g[t*num_v + h]). beta_raw/alpha are [T, num_v] token-major == that layout already.
        let mut beta = e.uninit(t * num_v)?;
        e.sigmoid(&beta_raw, &mut beta, t * num_v)?;
        // gdn_glog expects alpha [H,T] with alpha[t*H+h] and dt_bias/a [H] — matches token-major [T,num_v].
        let mut g_log = e.uninit(t * num_v)?;
        e.gdn_glog(&alpha, la.ssm_dt.float_data(), la.ssm_a.float_data(), &mut g_log, num_v, t)?;

        // GDN scan (A4: gdn_scan_prefill dispatches chunked WY under MEMRA_GDN_CHUNKED)
        let state_in = e.zeros(d_state * d_state * num_v)?;  // zero state (prefill)
        let mut state_out = e.zeros(d_state * d_state * num_v)?;
        let mut o = e.uninit(d_state * num_v * t)?;
        e.gdn_scan_prefill(&q_l2, &k_l2, &v_gd, &g_log, &beta, None, None, &state_in, &mut state_out, &mut o, num_v, t, scale, num_v)?;

        // gated RMSNorm: dst = RMSNorm(o, ssm_norm[head_v]) * silu(z). o is [d_state, num_v, T];
        // rows of head_v=d_state, nrows = num_v*T. z must match row layout: z is [T, value_dim] token-major
        // = [T, num_v*head_v]; per (t, vh) the head_v slice is contiguous -> rows align as (t*num_v+vh).
        // o rows are (t*num_v+vh) too. Good.
        let mut gn = e.uninit(d_state * num_v * t)?;
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
    /// Routing: host softmax+sort (default) OR the fused router kernel (MEMRA_FUSED_ROUTER).
    /// Dispatch: stage-every-token into 3 scratch slots (default) OR the SLRU residency cache
    /// (MEMRA_MOE_CACHE). The cache-HIT weight path is bit-identical to stage-every-token (§B.3).
    /// Convenience wrapper used by the hybrid trunk/MTP loops: pulls dims + max-block from `self`.
    pub fn moe_ffn_il(&self, e: &Engine, m: &MoeWeights, z: &CudaSlice<f32>, t: usize, il: u16)
               -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        Self::moe_ffn_inner(e, m, z, None, t, &self.cfg, il, self.max_moe_block(), false)
    }

    /// Prefill twin: Step35 promotes expert-grouped dispatch by default while decode/spec callers
    /// keep `moe_ffn_il` and therefore retain their existing dispatch class.
    pub fn moe_ffn_il_prefill(
        &self,
        e: &Engine,
        m: &MoeWeights,
        z: &CudaSlice<f32>,
        t: usize,
        il: u16,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        Self::moe_ffn_inner(e, m, z, None, t, &self.cfg, il, self.max_moe_block(), true)
    }

    /// Decode-path twin with a PRE-QUANTIZED z (from add_rms_norm_zq8): threads (zq, zd) into the
    /// t=1 dev arm so the per-layer standalone quantize_q8_1 launch folds away. Identical bytes
    /// (the fused kernel reproduces quantize_q8_1 exactly); every other path ignores the pair.
    pub fn moe_ffn_il_zq8(&self, e: &Engine, m: &MoeWeights, z: &CudaSlice<f32>,
                          zq8: Option<&(CudaSlice<i8>, CudaSlice<f32>)>, t: usize, il: u16)
               -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        Self::moe_ffn_inner(
            e, m, z, zq8, t, &self.cfg, il, self.max_moe_block(), false,
        )
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
        Self::moe_ffn_inner(e, m, z, None, t, cfg, il, max_block, false)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn moe_ffn_inner(
        e: &Engine,
        m: &MoeWeights,
        z: &CudaSlice<f32>,
        zq8: Option<&(CudaSlice<i8>, CudaSlice<f32>)>,
        t: usize,
        cfg: &ModelConfig,
        il: u16,
        max_block: usize,
        prefill: bool,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let worker_io = crate::spill_pread::worker_enabled();
        let epoch_lfu = std::env::var_os("MEMRA_MOE_LFU_DECAY").is_some();
        if Engine::moe_cache_enabled() && (worker_io || epoch_lfu) {
            e.with_moe_cache(max_block, |cache, _| {
                cache.begin_forward_epoch(il, t);
                if worker_io {
                    cache.begin_worker_scope();
                }
                Ok(())
            })?;
        }
        // Expert-grouped dispatch for prefill (T>1). Step35 prefill defaults here; the live
        // MEMRA_MOE_GROUPED seam can restore sequential or opt other callers in explicitly.
        if t > 1 && moe_grouped_enabled(cfg, prefill) {
            let grouped_out = Self::moe_ffn_grouped(e, m, z, t, cfg, il, max_block)?;
            // MEMRA_MOE_GATE: byte-identity comparison vs sequential path.
            // The grouped q8 path uses the same row-wise quantize/dot/activation/FMA programs as
            // sequential dispatch; mixed or q8-disabled layouts fall back to the matching f32
            // path. A mismatch is therefore a correctness failure, not an accepted numeric class.
            if std::env::var("MEMRA_MOE_GATE").is_ok() {
                let seq_out = Self::moe_ffn_sequential(e, m, z, t, cfg, il, max_block)?;
                let g_host = e.dtoh(&grouped_out)?;
                let s_host = e.dtoh(&seq_out)?;
                let g_bytes: &[u8] = unsafe { std::slice::from_raw_parts(g_host.as_ptr() as *const u8, g_host.len() * 4) };
                let s_bytes: &[u8] = unsafe { std::slice::from_raw_parts(s_host.as_ptr() as *const u8, s_host.len() * 4) };
                if g_bytes == s_bytes {
                    println!("moe-gate il={il} t={t} BYTE-IDENTICAL");
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

    /// One router-logit selector shared by sequential and grouped dispatch. Real prefill must use
    /// the row-wise GEMV when exact routing is enabled: cuBLASLt's reduction changes with `m`,
    /// which makes expert selection depend on the caller's chunk or concat-batch shape.
    fn moe_router_logits(
        e: &Engine,
        m: &MoeWeights,
        z: &CudaSlice<f32>,
        t: usize,
        cfg: &ModelConfig,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        if t < PRIME_MIN_T {
            // Decode and speculative verify use one fixed per-row reduction program.
            if crate::router_kernel_on() {
                e.router_gemv(
                    m.gate_inp.float_data(),
                    z,
                    cfg.n_embd as usize,
                    m.gate_exps.n_expert,
                    t,
                )
            } else {
                e.matmul_decode_exact(&m.gate_inp, z, t)
            }
        } else if crate::router_prefill_exact_on() && crate::router_kernel_on() {
            e.router_gemv(
                m.gate_inp.float_data(),
                z,
                cfg.n_embd as usize,
                m.gate_exps.n_expert,
                t,
            )
        } else {
            e.matmul(&m.gate_inp, z, t)
        }
    }

    /// Append the host-visible router selection for one layer/forward when calibration tracing is
    /// enabled. Both sequential and expert-grouped prefill must call this after routing so the
    /// trace is independent of the dispatch optimization selected for the forward.
    fn trace_moe_routes(il: u16, t: usize, sel_all: &[u32], weights: &[f32])
                        -> Result<(), Box<dyn std::error::Error>> {
        use std::io::Write as _;
        if let Ok(path) = std::env::var("MEMRA_MOE_TRACE") {
            let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
            let ids: Vec<String> = sel_all.iter().map(|s| s.to_string()).collect();
            writeln!(f, "{} {} {}", il, t, ids.join(","))?;
        }
        if let Ok(path) = std::env::var("MEMRA_MOE_WEIGHT_TRACE") {
            let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
            let pairs: Vec<String> = sel_all.iter().zip(weights)
                .map(|(expert, weight)| format!("{expert}:{weight:.9}"))
                .collect();
            writeln!(f, "{} {} {}", il, t, pairs.join(","))?;
        }
        Ok(())
    }

    /// Append the f32 input to one MoE layer for offline, layerwise calibration. This diagnostic
    /// intentionally performs a DtoH copy and is therefore disabled unless an explicit fresh trace
    /// directory is supplied. Each layer owns one payload file; index.jsonl records byte offsets so
    /// a validator can prove request/layer coverage before the trace is used for pruning or healing.
    fn trace_moe_input(e: &Engine, il: u16, t: usize, n_embd: usize, z: &CudaSlice<f32>)
                       -> Result<(), Box<dyn std::error::Error>> {
        use std::io::Write as _;
        let Ok(dir) = std::env::var("MEMRA_MOE_INPUT_TRACE_DIR") else { return Ok(()) };
        let host = e.dtoh(z)?;
        if host.len() != t * n_embd {
            return Err(format!(
                "MoE input trace shape mismatch at layer {il}: got {} values, expected {}x{}",
                host.len(), t, n_embd
            ).into());
        }
        let bytes = unsafe {
            std::slice::from_raw_parts(
                host.as_ptr().cast::<u8>(), host.len() * std::mem::size_of::<f32>()
            )
        };
        let state = MOE_INPUT_TRACE_WRITER.get_or_init(|| std::sync::Mutex::new(None));
        let mut state = state.lock().map_err(|_| "MoE input trace writer lock is poisoned")?;
        if state.is_none() {
            let dir = std::path::PathBuf::from(&dir);
            std::fs::create_dir_all(&dir)?;
            let index = std::fs::OpenOptions::new().create(true).append(true)
                .open(dir.join("index.jsonl"))?;
            *state = Some(MoeInputTraceWriter {
                dir,
                index,
                payloads: std::collections::HashMap::new(),
            });
        }
        let writer = state.as_mut().unwrap();
        if writer.dir != std::path::Path::new(&dir) {
            return Err("MEMRA_MOE_INPUT_TRACE_DIR changed after capture started".into());
        }
        let file_name = format!("layer-{il:03}.f32");
        if !writer.payloads.contains_key(&il) {
            let payload = std::fs::OpenOptions::new().create(true).append(true)
                .open(writer.dir.join(&file_name))?;
            let offset = payload.metadata()?.len();
            writer.payloads.insert(il, (payload, offset));
        }
        let (payload, offset) = writer.payloads.get_mut(&il).unwrap();
        let row_offset = *offset;
        payload.write_all(bytes)?;
        *offset += bytes.len() as u64;
        writeln!(
            writer.index,
            "{{\"format\":\"memra-moe-input-trace-v1\",\"layer\":{il},\"tokens\":{t},\
             \"hidden_size\":{n_embd},\"file\":\"{file_name}\",\"offset\":{row_offset},\
             \"payload_bytes\":{}}}",
            bytes.len()
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn moe_ffn_sequential_zq8(
        e: &Engine,
        m: &MoeWeights,
        z: &CudaSlice<f32>,
        zq8: Option<&(CudaSlice<i8>, CudaSlice<f32>)>,
        t: usize,
        cfg: &ModelConfig,
        il: u16,
        max_block: usize,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        use crate::moe_cache::{BlockId, PROJ_DOWN, PROJ_GATE, PROJ_UP};
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

        // step35 PER-LAYER SwiGLU clamp (None on every other arch and on every unclamped layer).
        // Routed experts and the shared expert read SEPARATE arrays — never share one value.
        let lim_exp = cfg.clamp_exp_at(il as u32);
        let lim_shexp = cfg.clamp_shexp_at(il as u32);
        let use_cache = Engine::moe_cache_enabled();
        let uniform_experts = m.has_uniform_expert_layout();
        let moe_q8 = uniform_experts && moe_q8_enabled()
            && q8_expert_supported(m.gate_exps.qtype) && q8_expert_supported(m.up_exps.qtype)
            && q8_expert_supported(m.down_exps.qtype);
        // Experimental secondary backend: complete experts already resident in the SLRU stay on
        // CUDA; any expert missing one or more projections runs from the original host GGUF bytes
        // through llama.cpp CPU quant dots. Small-t only keeps decode and speculative verification
        // in the same numeric/dispatch class; real prefill remains on the established GPU path and
        // seeds the residency cache. An explicit library path is the build/runtime gate, so naked
        // commands and CI have no llama.cpp or OpenMP dependency.
        let cpu_expert_requested = crate::cpu_experts::configured();
        if cpu_expert_requested && (cfg.hy3.is_none() || cfg.m3.is_some()) {
            return Err(std::io::Error::other(
                "MEMRA_CPU_EXPERT_LIB is experimental and currently gated to Hy3",
            )
            .into());
        }
        let cpu_hybrid = cpu_expert_requested && t < PRIME_MIN_T && m.dev_exps.is_none();
        // A dynamic SLRU changes which complete experts run on CUDA versus llama.cpp CPU dots.
        // Those backends are each deterministic but are different numeric configurations, so a
        // later prefill eviction can change greedy output. Freeze after the first real prefill;
        // decode/spec reads the fixed resident set, while later prefill misses use transient GPU
        // staging below and cannot change backend assignment.
        let freeze_cpu_residency = cpu_expert_requested
            && std::env::var("MEMRA_CPU_EXPERT_FREEZE_CACHE").as_deref() == Ok("1");
        let caller_warms_before_freeze = std::env::var("MEMRA_CPU_EXPERT_FREEZE_WARMUP_TOKENS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .is_some_and(|tokens| tokens > 0);
        if cpu_hybrid && freeze_cpu_residency && !caller_warms_before_freeze {
            e.freeze_moe_cache();
        }
        let cache_frozen = use_cache && e.moe_cache_frozen();
        let cache_dispatch = use_cache && (!cache_frozen || cpu_hybrid);

        // 1. ROUTER: one selector is shared with expert-grouped prefill so changing dispatch
        // cannot change logits, selected expert ids, or routing weights.
        let logits = Self::moe_router_logits(e, m, z, t, cfg)?;

        // LAUNCH-STRUCTURE STAGE 3 (2026-07-05, MEMRA_MOE_DEV default ON, =0 rollback): ZERO-DtoH
        // device-dispatch when this layer's expert blocks are ALL cache-resident. The fused
        // router's sel/w stay ON DEVICE; the expert weight pointers come from the per-layer
        // device table of fixed slot addresses; gate/up/silu + down/fma run as the same TWO
        // launches per token as gdec. Removes the per-layer router DtoH + stream sync — the
        // per-token host stall that dominated the 35B decode wall after stages 1+2.
        // BIT-IDENTITY: the router kernel is selection-exact vs the host oracle (kernel-check
        // tie gate) and the _dev matvec twins reproduce the gdec kernels' exact FP chains; the
        // only difference is where sel/w/pointers are READ from (device instead of params).
        // Residency: one-shot PREWARM force-admits the layer while free slots cover it
        // (MEMRA_MOE_PREWARM=0 -> organic residency, dev path fires when the SLRU fills).
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
        // resident slabs. MEMRA_MOE_PAIRS=0 rollback.
        // t >= PRIME_MIN_T only (2026-07-06 exactness fix, verify-probe proof): spec VERIFY
        // batches (t = 2..K+2) previously rode these pairs kernels while T=1 decode rode the
        // dev_q8 loop — different FP chains, verify-T2 logit maxdiff 2.6e-1 vs eager -> greedy
        // flips at tight margins -> 35B real-prompt spec self-consistency FAIL (the 27B
        // "verify must be kernel-DISPATCH-identical to decode" lesson, MoE edition). Small-t
        // now rides the dev loop below (same kernels per token as decode); pairs serves real
        // prefill (t >= 16, where spec never verifies).
        // sigmoid-router archs (M3, Hy3) must NOT enter the pairs/dev arms: those route via the
        // fused SOFTMAX device router (moe_router_topk) — silently wrong experts (the M3
        // gate-MISMATCH 74602-vs-92 lesson, 2026-07-07). Host sigmoid routing below is correct.
        // Per-expert macro-scales (compressed-tensors NVFP4 ST class, e.g. unsloth 35B-A3B):
        // the device-dispatch kernels (pairs/dev) do NOT fold them — those checkpoints must
        // ride the macro-aware sequential/staged paths below or every expert output is off by
        // its global scale (~3e4x, measured garbage 2026-07-16). GGUF experts: macros None.
        let no_exp_macros = m.gate_exps.macros.is_none() && m.up_exps.macros.is_none()
            && m.down_exps.macros.is_none();
        // A clamped-SwiGLU layer (step35 43/44) also cannot ride pairs: moe_pairs_silu_mul's
        // epilogue is plain silu(gate)*up with no clamp form, and moe_ffn_pairs takes no `il`
        // so it cannot even see the per-layer limit.
        if cfg.sigmoid_router().is_none() && cfg.m3.is_none() && cfg.hy3.is_none()
            && !cfg.swiglu_clamped_at(il as u32)
            && no_exp_macros
            && t >= PRIME_MIN_T && m.dev_exps.is_some() && moe_q8_enabled()
            && q8_expert_supported(m.gate_exps.qtype) && q8_expert_supported(m.up_exps.qtype)
            && q8_expert_supported(m.down_exps.qtype)
            && std::env::var("MEMRA_MOE_PAIRS").map(|v| v != "0").unwrap_or(true)
            && std::env::var("MEMRA_MOE_STATS").is_err() {
            return Self::moe_ffn_pairs(e, m, z, &logits, t, cfg);
        }

        // t < PRIME_MIN_T: moe_ffn_dev loops tokens serially (1 launch-pair per token) — the
        // decode path AND the spec-verify path (dispatch parity = exactness; see pairs gate
        // above). Serial launches are fine at t<=10 (K+2); real prefill never lands here.
        // moe_ffn_dev routes via the FUSED SOFTMAX device router (moe_router_topk) — sigmoid
        // routing (M3, Hy3: +expert bias) has no device kernel yet, so those arches must NOT
        // enter the dev arms: with MOE_CACHE=1 M3 silently routed softmax = wrong experts
        // (gate MISMATCH 74602 vs 92, caught 2026-07-07). Host sigmoid path below is correct.
        // macro-carrying experts are handled inside the dev path now (epilogue fold + w-scale);
        // pairs/gdec/csr keep their macro gates until their kernels grow the fold.
        // step35 (2026-08-06) is the third sigmoid-router arch and the deny above was written as
        // two arch names, not as the mechanism — so `dev_ok` let it into moe_ffn_dev's
        // moe_router_topk (softmax, no exp_probs_b bias, no expert_weights_scale) = silently
        // wrong experts, the same failure the pairs gate at :2254 already blocks by predicate.
        // Keyed off sigmoid_router() so arch #4 is denied by construction.
        // The clamped-SwiGLU layers must also fall through: every moe_ffn_dev kernel's fused
        // epilogue is plain silu(gate)*up (moe_gate_up_silu8_dev_*), which has no clamp form.
        let dev_ok = uniform_experts && cfg.sigmoid_router().is_none()
            && cfg.m3.is_none() && cfg.hy3.is_none()
            && !cfg.swiglu_clamped_at(il as u32);
        // Observation modes must route through the host-visible selection below. Otherwise a fully
        // resident layer returns through device dispatch before its trace/stats row is recorded,
        // silently biasing calibration toward only non-resident layers on large-VRAM machines.
        let observe_routes = std::env::var("MEMRA_MOE_STATS").is_ok()
            || std::env::var("MEMRA_MOE_TRACE").is_ok()
            || std::env::var("MEMRA_MOE_WEIGHT_TRACE").is_ok()
            || std::env::var("MEMRA_MOE_INPUT_TRACE_DIR").is_ok();
        if dev_ok && t < PRIME_MIN_T && m.dev_exps.is_some() && n_used <= 8 && moe_dev_enabled()
            && !observe_routes {
            return Self::moe_ffn_dev(e, m, z, zq8, &logits, t, cfg, il, max_block);
        }
        if dev_ok && use_cache && n_used <= 8 && moe_dev_enabled()
            && !observe_routes {
            let row_ok = e.with_moe_cache(max_block, |c, eng| {
                if moe_prewarm_enabled() { c.prewarm_layer(il, m, eng)?; }
                Ok(c.layer_dev_row(il, n_expert, eng)?.is_some())
            })?;
            if row_ok {
                return Self::moe_ffn_dev(e, m, z, zq8, &logits, t, cfg, il, max_block);
            }
        }

        // Per-token (sel[8], w[8]) — sigmoid host oracle (M3/Hy3), else fused-router/softmax.
        let (sel_all, w_all, routed_cpu_input) = if let Some(sig) = cfg.sigmoid_router() {
            if cpu_hybrid {
                let (sel, w, input) = Self::moe_route_sigmoid_with_input(
                    e,
                    &logits,
                    z,
                    t,
                    n_expert,
                    n_used,
                    m.exp_probs_b.as_deref(),
                    sig,
                    m.active_experts.as_deref(),
                )?;
                (sel, w, Some(input))
            } else {
                let (sel, w) = Self::moe_route_cfg(
                    e,
                    &logits,
                    t,
                    n_expert,
                    n_used,
                    m.exp_probs_b.as_deref(),
                    Some(sig),
                    m.active_experts.as_deref(),
                )?;
                (sel, w, None)
            }
        } else {
            let (sel, w) = Self::moe_route_cfg(
                e,
                &logits,
                t,
                n_expert,
                n_used,
                None,
                None,
                m.active_experts.as_deref(),
            )?;
            (sel, w, None)
        };

        // MEMRA_MOE_TRACE=<path>: append one line per (layer, step) with the selected expert ids —
        // offline analysis derives the decode working set + step-to-step reuse (the go/no-go
        // measurement for resident-expert tiering; see rig5090.jsonl 2026-07-07 pinned-tier row).
        Self::trace_moe_routes(il, t, &sel_all, &w_all)?;
        Self::trace_moe_input(e, il, t, n_embd, z)?;

        // Worker reads ordinarily overlap NVMe with the current expert but leave their H2D copies
        // demand-serialized on the compute stream. Hy3 decode already has a safe owner-thread
        // boundary here: sigmoid routing read `logits` back to the host, which synchronized all
        // earlier-layer compute. Submit the complete selected set first, then reserve only victims
        // outside that set and queue its H2D copies on the copy stream. Dispatch inserts an event
        // wait for each pending block, so later copies can overlap the earlier expert kernels while
        // the selected weights and numerical kernels remain unchanged. Restrict this experiment to
        // T=1; batched forwards can have token-local consumers still in flight between selections.
        // The CPU backend reads mmap-backed model bytes directly. Positioned-read worker buffers
        // are CUDA write-combined staging allocations and are intentionally not CPU-compute inputs;
        // do not spend NVMe bandwidth filling them for a layer whose misses go to CPU.
        let worker_disk_prefetch =
            cache_dispatch && crate::spill_pread::worker_enabled() && !cpu_hybrid;
        let promote_worker_h2d =
            t == 1 && worker_disk_prefetch && crate::spill_pread::copy_h2d_enabled();
        if promote_worker_h2d {
            let mut selected_blocks = Vec::with_capacity(n_used * 3);
            for &ex in sel_all.iter().take(n_used) {
                let ex = ex as u16;
                selected_blocks.extend([
                    BlockId::new(il, PROJ_GATE, ex),
                    BlockId::new(il, PROJ_UP, ex),
                    BlockId::new(il, PROJ_DOWN, ex),
                ]);
            }
            for &ex in sel_all.iter().take(n_used) {
                Self::moe_prefetch_disk_expert(e, il, ex as usize, m, max_block, &selected_blocks)?;
            }
            e.with_moe_cache(max_block, |cache, eng| {
                cache.promote_worker_reads_at_safe_boundary(
                    &selected_blocks,
                    &selected_blocks,
                    eng,
                )?;
                Ok(())
            })?;
        }

        // MEMRA_MOE_STATS: per-layer routing stats for the A2 (expert-grouped prefill) baseline —
        // per-token expert-id entropy, active-expert coverage, tokens-per-expert group sizes.
        if t > 1 && std::env::var("MEMRA_MOE_STATS").is_ok() {
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
        // `!swiglu_clamped_at`: the grouped-decode kernels' fused epilogue is plain
        // silu(gate)*up (see the m3 note at the call sites below) — a clamped layer must fall
        // through to the sequential loop's ffn_act_lim. Hoisted into the fire predicate so the
        // uninit/memset invariant at :2418 stays consistent with what actually dispatches.
        let gdec_may_fire = uniform_experts && use_cache && n_used <= 8 && gdec_enabled()
            && !cfg.swiglu_clamped_at(il as u32);
        // SLAB-LOCAL RESIDENT ARM (lane/pp-leverb 2026-08-08): the fits-VRAM resident slabs
        // (`dev_exps`, sized per PP device by cx-503b) were consumed ONLY by the pairs/dev
        // arms — every one of which DENIES sigmoid-router archs (step35/M3/Hy3). So on those
        // archs the slabs were uploaded but never read, and every expert went through the
        // SLRU (37 GB H2D staging per pp4096 prime on the Step SKU; and post-cx-503b the
        // dead slabs additionally STARVE the SLRU, which sizes itself on free-VRAM-after-
        // residents). This arm reads the slabs through the SAME kernels the SLRU arm runs —
        // gdec's fused pair when eligible, the per-expert qmatvec twins otherwise — with the
        // slab base + ex*stride as the pointer provenance (the exact bit-identity class
        // `moe_ffn_dev` documents for its resident-vs-SLRU arms; bytes are the same HostExps
        // bytes either way). LOCALITY GATE: `d.dev == e.ctx().ordinal()` — a slab on another
        // device must NOT be dereferenced (m=1 peer reads are the measured 34-150x class,
        // strictly worse than staging); under PP-2 without the prime walker this admits
        // stage-0 layers on dev0 and leaves stage-1 layers on the SLRU, which is the honest
        // interim shape. MEMRA_MOE_SLAB=0 = rollback/A-B seam (read per call).
        let slab_local = m.dev_exps.as_ref()
            .filter(|d| !d.gu_il && moe_slab_enabled() && d.dev == e.ctx().ordinal());
        let slab_bases = slab_local.map(|d| {
            use cudarc::driver::DevicePtr;
            let s = e.stream();
            let (pg, _g0) = d.gate.device_ptr(&s);
            let (pu, _g1) = d.up.device_ptr(&s);
            let (pd, _g2) = d.down.device_ptr(&s);
            (pg as u64, pu as u64, pd as u64)
        });
        // Fused-pair eligibility mirrors the gdec call sites exactly (plain-SiLU epilogue,
        // no macros, m3 clamp excluded, q8-supported qtypes, <=8 experts) — INCLUDING
        // `gdec_enabled()`: the pair IS the gdec kernel pair, so MEMRA_MOE_GDEC=0 must
        // disable it here too. That seam is also the exactness localizer: with GDEC=0 both
        // provenances run the SAME per-expert qmatvec kernels (slab base+stride vs SLRU
        // slot) and must be BIT-IDENTICAL — the true provenance-only pair — while GDEC=1
        // compares the fused-pair class against the SLRU's hit/miss MIX (gdec for
        // all-resident tokens, staged loop for misses), which is a dispatch-class
        // comparison, not a provenance one.
        let slab_fused_may_fire = slab_bases.is_some() && n_used <= 8 && gdec_enabled()
            && !cfg.swiglu_clamped_at(il as u32) && cfg.m3.is_none()
            && no_exp_macros && moe_q8;
        // moe_out memset elision: BOTH full-row-overwrite arms (gdec + slab fused) allocate
        // uninit; a token that falls through to any accumulating loop zeroes its own row.
        let mut moe_out = if gdec_may_fire || slab_fused_may_fire {
            e.uninit(t * n_embd)?
        } else {
            e.zeros(t * n_embd)?
        };
        // The router readback above already established a host boundary. Copy each small-t hidden
        // row once so CPU miss experts can start while the owner thread queues resident GPU work.
        let cpu_input = if cpu_hybrid {
            Some(routed_cpu_input.ok_or("CPU expert routing did not return the MoE input")?)
        } else {
            None
        };

        // GPU scratch: one slot per proj, big enough for ONE expert (default stage-every-token path).
        // STAGE 2: LAZY — allocated only if the no-cache staging path actually runs (under
        // MEMRA_MOE_CACHE they were 3 dead ~1MB alloc_zeros + memset + free per layer per token,
        // measured ~123 memsets/token of the decode wall).
        let g_len = m.gate_exps.max_expert_bytes();  // 860160 for the uniform 35B gate
        let u_len = m.up_exps.max_expert_bytes();    // 860160 for the uniform 35B up
        let d_len = m.down_exps.max_expert_bytes();  // 1114112 for the uniform 35B down
        let mut scratch_g: Option<CudaSlice<u8>> = None;
        let mut scratch_u: Option<CudaSlice<u8>> = None;
        let mut scratch_d: Option<CudaSlice<u8>> = None;
        // `max_block` (the GLOBAL max expert stride across all layers) is passed in — the cache slots
        // are FIXED-ADDRESS and must fit any layer's block (UD/dynamic GGUFs vary quant per layer).

        // EDGE-1 §C.2/C.3: the optional pipeline queues the next selected expert's MISS blocks on
        // the copy stream before launching the current expert's compute. Pending slots stay invisible
        // to cache hits until dispatch inserts the completion-event wait; current gate/up/down ids are
        // protected from eviction. This changes scheduling only, never the GGUF bytes or GEMM path.
        let page_window = moe_page_prefetch_window();

        // 2. PER TOKEN: routed-expert loop. The ONE dispatch change vs Stage-1: a resident slot
        //    (cache HIT, no H2D) OR a staged slot (MISS) feeds the SAME unchanged qmatvec_view.
        for tok in 0..t {
            let sel = &sel_all[tok * n_used..(tok + 1) * n_used];
            let w = &w_all[tok * n_used..(tok + 1) * n_used];
            let zt = z.slice(tok * n_embd..(tok + 1) * n_embd);  // CudaView<f32>
            let mut tok_q8: Option<(CudaSlice<i8>, CudaSlice<f32>)> = None;

            // STAGE-2 GROUPED DECODE (2026-07-04, MEMRA_MOE_GDEC default ON, =0 rollback): fold
            // this token's whole routed-expert FFN (8x gate/up/silu + 8x down/axpy = 40 launches)
            // into TWO launches via expert-pointer indirection over the fixed-address cache slots.
            // Fires only when ALL 3*n_used blocks are ALREADY cache-resident (pure-HIT: zero
            // memcpy, zero admission, so no slot can move under the collected pointers) — any
            // miss falls through to the sequential loop below, which admits as before. In steady
            // state on a fully-resident rig every token-layer takes the grouped path.
            // BIT-IDENTITY: each in-kernel dot reproduces qmatvec_f32's exact reduction; SiLU is
            // silu_mul_f32's exact expression; the down accumulation is a slot-ordered
            // __fmaf_rn chain == the sequential axpy_f32 chain (MEMRA_MOE_GDEC_GATE compares).
            // cfg.m3: the grouped kernels' fused epilogues are plain SiLU — M3's swigluoai must
            // NOT take them until the kernels grow the clamped variant. NVFP4 experts carry
            // per-expert macro-scales the fused kernels don't fold — those fall through too.
            let no_macros = m.gate_exps.macros.is_none() && m.up_exps.macros.is_none()
                && m.down_exps.macros.is_none();
            // SLAB-LOCAL FUSED PAIR (lane/pp-leverb 2026-08-08): the gdec launch pair
            // (moe_gate_up_silu8_q8 + moe_down8_fma_q8 — the EXACT kernels, same FP chains)
            // with pointers computed from the resident slab base + ex*stride instead of
            // collected SLRU slot addresses. No cache lock, no residency predicate — the
            // slab holds every expert by construction, so this arm never falls through
            // (gdec's P(all 24 resident) ≈ 0.37 coin-flip and the miss path's ~49-launch
            // staging both die). Bit-identity class: pointer provenance only, the same
            // slab-vs-SLRU equivalence moe_ffn_dev documents. Ordered BEFORE gdec: when a
            // slab exists it is strictly better (no lock, no miss).
            if slab_fused_may_fire {
                let (pg, pu, pd) = slab_bases.unwrap();
                let mut gp = [0u64; 8];
                let mut up = [0u64; 8];
                let mut dp = [0u64; 8];
                for (j, &ex) in sel.iter().enumerate() {
                    let ex = ex as usize;
                    gp[j] = pg + (ex * m.gate_exps.expert_stride) as u64;
                    up[j] = pu + (ex * m.up_exps.expert_stride) as u64;
                    dp[j] = pd + (ex * m.down_exps.expert_stride) as u64;
                }
                let mut wv = [0f32; 8];
                wv[..n_used].copy_from_slice(w);
                if tok_q8.is_none() {
                    tok_q8 = Some(e.quantize_q8_1_view(&zt, 1, n_embd)?);
                }
                let (zq, zd) = tok_q8.as_ref().unwrap();
                let act = e.moe_gate_up_silu8_q8(crate::WPtr8(gp), crate::WPtr8(up), zq, zd,
                                                 n_embd, n_ff_exp, n_used,
                                                 m.gate_exps.qtype, m.up_exps.qtype,
                                                 m.gate_exps.row_bytes, m.up_exps.row_bytes)?;
                let (aq2, ad2) = e.quantize_q8_1(&act, n_used, n_ff_exp)?;
                let mut dst = moe_out.slice_mut(tok * n_embd..(tok + 1) * n_embd);
                e.moe_down8_fma_q8(crate::WPtr8(dp), crate::F32x8(wv), &aq2, &ad2, &mut dst,
                                   n_ff_exp, n_embd, n_used,
                                   m.down_exps.qtype, m.down_exps.row_bytes)?;
                continue;
            }
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

            // STAGE 2 memset-elision invariant: moe_out was allocated UNINIT when gdec or the
            // slab pair could fire. This token fell through to a sequential axpy loop, which
            // ACCUMULATES — zero its row first (row-sized memset; other rows are owned by the
            // full-overwrite arms). slab_fused_may_fire never actually reaches here (its arm
            // has no fallible predicate), included for the allocation invariant's symmetry.
            if gdec_may_fire || slab_fused_may_fire {
                let mut row = moe_out.slice_mut(tok * n_embd..(tok + 1) * n_embd);
                e.memset_zeros_view(&mut row)?;
            }

            // Fiddler-style split at whole-expert granularity. Partial residency is deliberately a
            // CPU assignment: transferring the missing projection would reintroduce the PCIe/NVMe
            // stall this path exists to remove, while mixing projections would require another
            // activation round-trip. Weight addresses remain valid until this worker is joined at
            // the bottom of the token scope.
            let mut cpu_mask = vec![false; sel.len()];
            let cpu_worker = if let Some(host_input) = cpu_input.as_ref() {
                let gpu_resident = if use_cache {
                    e.with_moe_cache(max_block, |cache, _| {
                        Ok(sel
                            .iter()
                            .map(|&expert| {
                                let expert = expert as u16;
                                [PROJ_GATE, PROJ_UP, PROJ_DOWN]
                                    .into_iter()
                                    .filter(|&projection| {
                                        cache
                                            .resident(BlockId::new(il, projection, expert))
                                            .is_some()
                                    })
                                    .count()
                            })
                            .collect::<Vec<_>>())
                    })?
                } else {
                    vec![0; sel.len()]
                };
                let mut cpu_selected = Vec::new();
                for (index, (&expert, &route_weight)) in sel.iter().zip(w).enumerate() {
                    if gpu_resident[index] != 3 {
                        cpu_mask[index] = true;
                        crate::cpu_experts::record_incomplete_gpu_residency(gpu_resident[index]);
                        let expert = expert as usize;
                        cpu_selected.push((expert, route_weight));
                    }
                }
                if crate::cpu_experts::predictor_enabled() {
                    // Fire-and-forget lookahead: the predictor worker scores layers il+1..
                    // from this layer's MoE input and prefetches predicted-and-missing
                    // experts into the companion RAM cache. Never blocks this thread.
                    let row = &host_input[tok * n_embd..(tok + 1) * n_embd];
                    crate::cpu_experts::predictor_submit(il, row);
                }
                if cpu_selected.is_empty() {
                    None
                } else {
                    let row = &host_input[tok * n_embd..(tok + 1) * n_embd];
                    let job = crate::cpu_experts::prepare_job(m, il, &cpu_selected, row)
                        .map_err(std::io::Error::other)?;
                    Some(crate::cpu_experts::submit(job).map_err(std::io::Error::other)?)
                }
            } else {
                None
            };

            let worker_window = worker_disk_prefetch
                .then(worker_prefetch_window)
                .unwrap_or(0);
            for (j, &ex) in sel.iter().enumerate() {
                if cpu_mask[j] {
                    continue;
                }
                let ex = ex as usize;
                // PER-EXPERT SLAB READ (lane/pp-leverb 2026-08-08): the layers the fused
                // pair above excludes — step35's CLAMPED layers 43/44 (ffn_act_lim has no
                // fused form) and macro-carrying artifacts — still have their bytes in the
                // local resident slab. Same kernels as the SLRU arms (`qmatvec_expert_q8` /
                // `qmatvec_view`), same ffn_act_lim/macro folds; provenance = slab base +
                // ex*stride. No dispatch lock, no admission, no prefetch — nothing to miss.
                if let Some(d) = slab_local {
                    let gl = m.gate_exps.expert_layout(ex);
                    let ul = m.up_exps.expert_layout(ex);
                    let dl = m.down_exps.expert_layout(ex);
                    let (g0, u0, d0) = (ex * m.gate_exps.expert_stride,
                                        ex * m.up_exps.expert_stride,
                                        ex * m.down_exps.expert_stride);
                    let (gate, up) = if moe_q8 {
                        if tok_q8.is_none() {
                            tok_q8 = Some(e.quantize_q8_1_view(&zt, 1, n_embd)?);
                        }
                        let (zq, zd) = tok_q8.as_ref().unwrap();
                        (e.qmatvec_expert_q8(&d.gate, g0..g0 + gl.len, zq, zd, 1,
                                             m.gate_exps.in_f, m.gate_exps.out_f,
                                             gl.qtype, gl.row_bytes)?,
                         e.qmatvec_expert_q8(&d.up, u0..u0 + ul.len, zq, zd, 1,
                                             m.up_exps.in_f, m.up_exps.out_f,
                                             ul.qtype, ul.row_bytes)?)
                    } else {
                        (e.qmatvec_view(&d.gate, g0..g0 + gl.len, &zt, 1,
                                        m.gate_exps.in_f, m.gate_exps.out_f,
                                        gl.qtype, gl.row_bytes)?,
                         e.qmatvec_view(&d.up, u0..u0 + ul.len, &zt, 1,
                                        m.up_exps.in_f, m.up_exps.out_f,
                                        ul.qtype, ul.row_bytes)?)
                    };
                    let mut act = e.uninit(n_ff_exp)?;
                    Self::ffn_act_lim(e, cfg, &gate, &up, m.gate_exps.macro_scale(ex),
                                      m.up_exps.macro_scale(ex), lim_exp, &mut act, n_ff_exp)?;
                    let y = if moe_q8 {
                        let (aq2, ad2) = e.quantize_q8_1(&act, 1, n_ff_exp)?;
                        e.qmatvec_expert_q8(&d.down, d0..d0 + dl.len, &aq2, &ad2, 1,
                                            m.down_exps.in_f, m.down_exps.out_f,
                                            dl.qtype, dl.row_bytes)?
                    } else {
                        let actv = act.slice(0..n_ff_exp);
                        e.qmatvec_view(&d.down, d0..d0 + dl.len, &actv, 1,
                                       m.down_exps.in_f, m.down_exps.out_f,
                                       dl.qtype, dl.row_bytes)?
                    };
                    let mut dst = moe_out.slice_mut(tok * n_embd..(tok + 1) * n_embd);
                    e.axpy_into(&y, w[j] * m.down_exps.macro_scale(ex), &mut dst, n_embd)?;
                    continue;
                }
                for next in page_prefetch_positions(j, sel.len(), page_window) {
                    Self::moe_prefetch_host_expert(sel[next] as usize, m);
                }
                let keep = [
                    crate::moe_cache::BlockId::new(il, crate::moe_cache::PROJ_GATE, ex as u16),
                    crate::moe_cache::BlockId::new(il, crate::moe_cache::PROJ_UP, ex as u16),
                    crate::moe_cache::BlockId::new(il, crate::moe_cache::PROJ_DOWN, ex as u16),
                ];
                if worker_disk_prefetch && worker_window > 0 {
                    for next in worker_prefetch_positions(j, sel.len(), worker_window) {
                        Self::moe_prefetch_disk_expert(
                            e,
                            il,
                            sel[next] as usize,
                            m,
                            max_block,
                            &keep,
                        )?;
                    }
                } else if cache_dispatch
                    && !cpu_hybrid
                    && moe_prefetch_enabled()
                    && j + 1 < sel.len()
                {
                    let next = sel[j + 1] as usize;
                    Self::moe_prefetch_expert(e, il, next, m, max_block, &keep)?;
                }
                let [gate_q8, up_q8, down_q8] = [moe_q8; 3];
                if cache_dispatch && (gate_q8 || up_q8 || down_q8) {
                    // dp4a EXPERT PATH (MEMRA_MOE_Q8): quantize z-row once per token. Mixed expert
                    // layouts stay on the metadata-aware f32 path.
                    if (gate_q8 || up_q8) && tok_q8.is_none() {
                        tok_q8 = Some(e.quantize_q8_1_view(&zt, 1, n_embd)?);
                    }
                    let gate = if gate_q8 {
                        let (zq, zd) = tok_q8.as_ref().unwrap();
                        Self::moe_cached_gemm_q8(e, il, PROJ_GATE, ex, m, max_block, zq, zd)?
                    } else {
                        Self::moe_cached_gemm(e, il, PROJ_GATE, ex, m, max_block, &zt)?
                    };
                    let up = if up_q8 {
                        let (zq, zd) = tok_q8.as_ref().unwrap();
                        Self::moe_cached_gemm_q8(e, il, PROJ_UP, ex, m, max_block, zq, zd)?
                    } else {
                        Self::moe_cached_gemm(e, il, PROJ_UP, ex, m, max_block, &zt)?
                    };
                    let mut act = e.uninit(n_ff_exp)?;
                    Self::ffn_act_lim(
                        e,
                        cfg,
                        &gate,
                        &up,
                        m.gate_exps.macro_scale(ex),
                        m.up_exps.macro_scale(ex),
                        lim_exp,
                        &mut act,
                        n_ff_exp,
                    )?;
                    let y = if down_q8 {
                        let (aq2, ad2) = e.quantize_q8_1(&act, 1, n_ff_exp)?;
                        Self::moe_cached_gemm_q8(e, il, PROJ_DOWN, ex, m, max_block, &aq2, &ad2)?
                    } else {
                        let actv = act.slice(0..n_ff_exp);
                        Self::moe_cached_gemm(e, il, PROJ_DOWN, ex, m, max_block, &actv)?
                    };
                    let mut dst = moe_out.slice_mut(tok * n_embd..(tok + 1) * n_embd);
                    // down-proj macro folds into the accumulate weight (1.0 for non-macro archs).
                    e.axpy_into(&y, w[j] * m.down_exps.macro_scale(ex), &mut dst, n_embd)?;
                } else if cache_dispatch {
                    // SLRU residency cache: per-projection, dispatch the block (HIT => resident slot,
                    // MISS => staged slot) then run the SAME unchanged qmatvec_view from that slot.
                    // The bytes the kernel reads are byte-for-byte the same GGUF block (§B.3); the
                    // only difference between HIT and MISS is whether the memcpy_htod ran.
                    let gate = Self::moe_cached_gemm(e, il, PROJ_GATE, ex, m, max_block, &zt)?;
                    let up   = Self::moe_cached_gemm(e, il, PROJ_UP,   ex, m, max_block, &zt)?;
                    let mut act = e.uninit(n_ff_exp)?;  // activation fully overwrites
                    Self::ffn_act_lim(e, cfg, &gate, &up, m.gate_exps.macro_scale(ex),
                        m.up_exps.macro_scale(ex), lim_exp, &mut act, n_ff_exp)?;
                    let actv = act.slice(0..n_ff_exp);
                    let y = Self::moe_cached_gemm(e, il, PROJ_DOWN, ex, m, max_block, &actv)?;
                    let mut dst = moe_out.slice_mut(tok * n_embd..(tok + 1) * n_embd);
                    // down-proj macro folds into the accumulate weight (post-matmul linear scale).
                    e.axpy_into(&y, w[j] * m.down_exps.macro_scale(ex), &mut dst, n_embd)?;
                } else if cache_frozen {
                    // A later prompt prime must not change the CPU/GPU assignment frozen after the
                    // first prime. Reuse every fixed resident projection directly and stage only a
                    // true miss through the ordinary scratch slot. This preserves the established
                    // f32-dequant numeric path while avoiding a full-bank reread on every prime.
                    let gate = Self::moe_frozen_gemm(
                        e,
                        il,
                        PROJ_GATE,
                        ex,
                        m,
                        max_block,
                        &zt,
                        &mut scratch_g,
                        g_len,
                    )?;
                    let up = Self::moe_frozen_gemm(
                        e,
                        il,
                        PROJ_UP,
                        ex,
                        m,
                        max_block,
                        &zt,
                        &mut scratch_u,
                        u_len,
                    )?;
                    let mut act = e.uninit(n_ff_exp)?;
                    Self::ffn_act_lim(
                        e,
                        cfg,
                        &gate,
                        &up,
                        m.gate_exps.macro_scale(ex),
                        m.up_exps.macro_scale(ex),
                        lim_exp,
                        &mut act,
                        n_ff_exp,
                    )?;
                    let actv = act.slice(0..n_ff_exp);
                    let y = Self::moe_frozen_gemm(
                        e,
                        il,
                        PROJ_DOWN,
                        ex,
                        m,
                        max_block,
                        &actv,
                        &mut scratch_d,
                        d_len,
                    )?;
                    let mut dst = moe_out.slice_mut(tok * n_embd..(tok + 1) * n_embd);
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
                    let gl = m.gate_exps.expert_layout(ex);
                    let ul = m.up_exps.expert_layout(ex);
                    let dl = m.down_exps.expert_layout(ex);
                    e.stage_expert(m.gate_exps.expert_bytes(ex), sg, 0)?;
                    let gate = e.qmatvec_view(sg, 0..gl.len, &zt, 1,
                        m.gate_exps.in_f, m.gate_exps.out_f, gl.qtype, gl.row_bytes)?;

                    e.stage_expert(m.up_exps.expert_bytes(ex), su, 0)?;
                    let up = e.qmatvec_view(su, 0..ul.len, &zt, 1,
                        m.up_exps.in_f, m.up_exps.out_f, ul.qtype, ul.row_bytes)?;

                    let mut act = e.uninit(n_ff_exp)?;  // activation fully overwrites
                    Self::ffn_act_lim(e, cfg, &gate, &up, m.gate_exps.macro_scale(ex),
                        m.up_exps.macro_scale(ex), lim_exp, &mut act, n_ff_exp)?;

                    e.stage_expert(m.down_exps.expert_bytes(ex), sd, 0)?;
                    let actv = act.slice(0..n_ff_exp);
                    let y = e.qmatvec_view(sd, 0..dl.len, &actv, 1,
                        m.down_exps.in_f, m.down_exps.out_f, dl.qtype, dl.row_bytes)?;

                    let mut dst = moe_out.slice_mut(tok * n_embd..(tok + 1) * n_embd);
                    e.axpy_into(&y, w[j] * m.down_exps.macro_scale(ex), &mut dst, n_embd)?;
                }
            }
            if let Some(worker) = cpu_worker {
                let cpu_output = worker.wait().map_err(std::io::Error::other)?;
                let cpu_output = e.htod(&cpu_output)?;
                let mut dst = moe_out.slice_mut(tok * n_embd..(tok + 1) * n_embd);
                e.axpy_into(&cpu_output, 1.0, &mut dst, n_embd)?;
            }
            if cpu_hybrid && !cache_frozen && cpu_expert_profile_admit_enabled() {
                for (j, &ex) in sel.iter().enumerate() {
                    if cpu_mask[j] {
                        Self::moe_profile_admit_expert(e, il, ex as usize, m, max_block)?;
                    }
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
            Self::ffn_act_lim(e, cfg, &sg_gate, &sg_up, 1.0, 1.0, lim_shexp, &mut sa, t * n_ff_sh)?;
            let sh = if verify_t { e.matmul_decode_exact(down_shexp, &sa, t)? }
                     else { e.matmul(down_shexp, &sa, t)? };     // [T, n_embd]

            // shexp gate: qwen35moe sigmoid-gates via ffn_gate_inp_shexp (1-D ne=[n_embd] ->
            // out_f=1); M3 has no gate tensor -> weight 1.0. Decode + verify ride the fused
            // sigmoid-dot kernel (one fold order for both chains; kills the per-layer
            // cuBLASLt m=1 splitK GEMM — 40x/step, ~10% of the H100 q35 decode step).
            // SERVE ISOLATION (lane/concat-prime-exact, 2026-08-02): PREFILL rides it too.
            // This out_f=1 cuBLASLt GEMV is the SECOND m-dependent op in the trunk (probed:
            // rows [0,19) move by 1.07e-4 between m=74 and m=75 while sigmoid_dot_rows is
            // BIT-IDENTICAL — allw-shexpgate-o35b.log). The gate multiplies the shared
            // expert's contribution into every token's residual, so under cross-request
            // concat prefill a session's hidden state depended on its co-arrivals' token
            // count. MEMRA_ROUTER_PREFILL_EXACT=0 reverts to the batched cuBLASLt linear.
            let g = match &m.gate_inp_shexp {
                Some(gate_inp_shexp) => {
                    if t < PRIME_MIN_T || crate::router_prefill_exact_on() {
                        e.sigmoid_dot_rows(z, gate_inp_shexp.float_data(), n_embd, t)?
                    } else {
                        let gs = e.linear(z, gate_inp_shexp.float_data(), t, n_embd, 1)?;
                        let mut g = e.uninit(t)?;  // sigmoid fully overwrites
                        e.sigmoid(&gs, &mut g, t)?;
                        g
                    }
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
                bytes += n_used * (m.gate_exps.max_expert_bytes() + m.up_exps.max_expert_bytes()
                                   + m.down_exps.max_expert_bytes()) as u64;
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
                mx = mx.max(m.gate_exps.max_expert_bytes())
                       .max(m.up_exps.max_expert_bytes())
                       .max(m.down_exps.max_expert_bytes());
            }
        };
        for l in self.layers.iter() { scan(&l.ffn); }
        if let Some(mtp) = self.mtp.as_ref() { scan(&mtp.ffn); }
        mx
    }

    /// Exact retained projection lengths for cache class sizing. Pruned ids keep router positions
    /// but have no bytes and therefore consume no residency slot.
    pub(crate) fn moe_cache_block_sizes(&self) -> Vec<usize> {
        use crate::hybrid::Ffn;
        let mut sizes = Vec::new();
        let mut scan = |ffn: &Ffn| {
            let Ffn::Moe(m) = ffn else { return };
            for ex in 0..m.gate_exps.n_expert {
                if m.active_experts.as_ref().is_some_and(|active| !active[ex]) {
                    continue;
                }
                for exps in [&m.gate_exps, &m.up_exps, &m.down_exps] {
                    let len = exps.expert_layout(ex).len;
                    if len > 0 {
                        sizes.push(len);
                    }
                }
            }
        };
        for layer in &self.layers {
            scan(&layer.ffn);
        }
        if let Some(mtp) = &self.mtp {
            scan(&mtp.ffn);
        }
        sizes
    }

    /// Persist the frozen residency set so a later process can restage it directly and skip
    /// the profiling warmup. Plain text: a versioned header binding slot geometry, then one
    /// `layer proj ex` triple per line. A mismatched or stale profile is rejected at load
    /// (header check) or degrades to fewer restaged blocks (per-id checks); either way the
    /// post-freeze argmax gate still validates the serving assignment.
    pub fn save_cpu_expert_residency_profile(
        &self,
        e: &Engine,
        path: &std::path::Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let Some(ids) = e.export_moe_residency() else {
            return Err("no MoE residency cache to persist".into());
        };
        let mut body = format!(
            "memra-freeze-profile v1 max_block={} blocks={}\n",
            self.max_moe_block(),
            ids.len()
        );
        for (layer, proj, ex) in &ids {
            body.push_str(&format!("{layer} {proj} {ex}\n"));
        }
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, body)?;
        std::fs::rename(&tmp, path)?;
        println!(
            "[moe-cache] freeze profile saved: {} blocks -> {}",
            ids.len(),
            path.display()
        );
        Ok(())
    }

    /// Restage a saved freeze profile and freeze immediately, skipping the profiling warmup.
    /// Returns false (leaving the cache untouched for a normal warmup) when the profile is
    /// missing or its header does not match this model's slot geometry.
    pub fn restore_cpu_expert_residency_profile(
        &self,
        e: &Engine,
        path: &std::path::Path,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        use crate::hybrid::Ffn;
        use crate::moe_cache::BlockId;
        let Ok(content) = std::fs::read_to_string(path) else {
            return Ok(false);
        };
        let mut lines = content.lines();
        let Some(header) = lines.next() else { return Ok(false) };
        let expected = format!("memra-freeze-profile v1 max_block={}", self.max_moe_block());
        if !header.starts_with(&expected) {
            println!(
                "[moe-cache] freeze profile ignored (geometry mismatch): {}",
                path.display()
            );
            return Ok(false);
        }
        let mut by_layer: std::collections::HashMap<u16, Vec<BlockId>> =
            std::collections::HashMap::new();
        for line in lines {
            let mut fields = line.split_whitespace();
            let (Some(layer), Some(proj), Some(ex)) =
                (fields.next(), fields.next(), fields.next())
            else {
                continue;
            };
            let (Ok(layer), Ok(proj), Ok(ex)) =
                (layer.parse::<u16>(), proj.parse::<u8>(), ex.parse::<u16>())
            else {
                continue;
            };
            by_layer
                .entry(layer)
                .or_default()
                .push(BlockId::new(layer, proj, ex));
        }
        let requested: usize = by_layer.values().map(Vec::len).sum();
        if requested == 0 {
            return Ok(false);
        }
        let max_block = self.max_moe_block();
        let mut restaged = 0usize;
        let mut stage_layer = |layer_index: u16,
                               ffn: &Ffn|
         -> Result<(), Box<dyn std::error::Error>> {
            let Ffn::Moe(m) = ffn else { return Ok(()) };
            let Some(ids) = by_layer.get(&layer_index) else {
                return Ok(());
            };
            e.with_moe_cache(max_block, |cache, eng| {
                for id in ids {
                    if cache.restage_block(*id, m, eng)? {
                        restaged += 1;
                    }
                }
                Ok(())
            })
        };
        for (index, layer) in self.layers.iter().enumerate() {
            stage_layer(index as u16, &layer.ffn)?;
        }
        if let Some(mtp) = self.mtp.as_ref() {
            stage_layer(u16::MAX, &mtp.ffn)?;
        }
        e.freeze_moe_cache();
        println!(
            "[moe-cache] freeze profile restored: {restaged}/{requested} blocks restaged from {}",
            path.display()
        );
        Ok(true)
    }

    /// Freeze the heterogeneous CPU/GPU split after the caller's discarded profile warmup.
    pub fn freeze_cpu_expert_residency(
        &self,
        e: &Engine,
    ) -> Result<(), Box<dyn std::error::Error>> {
        e.freeze_moe_cache();
        Ok(())
    }

    /// FFN activation dispatch: swigluoai (clamped, alpha/limit) when cfg.m3 says so, else the
    /// standard SiLU*up. One seam so every FFN site (dense, routed expert, shared expert) follows
    /// the model's activation exactly.
    ///
    /// NO-`il` FORM: cannot apply step35's PER-LAYER SwiGLU clamp. Only call it from a site whose
    /// layer provably has no live limit (dense-FFN layers, MTP blocks) — `ffn_act_lim` is the
    /// form for anything that can land on a clamped layer.
    pub fn ffn_act(e: &Engine, cfg: &ModelConfig, gate: &CudaSlice<f32>, up: &CudaSlice<f32>,
               act: &mut CudaSlice<f32>, n: usize) -> Result<(), Box<dyn std::error::Error>> {
        Self::ffn_act_scaled(e, cfg, gate, up, 1.0, 1.0, act, n)
    }

    /// ffn_act with per-tensor post-matmul macro-scales folded in (gs/us == 1.0 -> identical
    /// float ops to ffn_act; used by the ModelOpt NVFP4 expert path where each expert tensor
    /// carries a `weight_scale_2`). Same no-`il` contract as `ffn_act`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn ffn_act_scaled(e: &Engine, cfg: &ModelConfig, gate: &CudaSlice<f32>, up: &CudaSlice<f32>,
               gs: f32, us: f32, act: &mut CudaSlice<f32>, n: usize)
               -> Result<(), Box<dyn std::error::Error>> {
        Self::ffn_act_lim(e, cfg, gate, up, gs, us, None, act, n)
    }

    /// ffn_act_scaled + step35's PER-LAYER clamped SwiGLU. `limit`:
    ///   * `None`   -> the unclamped dispatch (every arch except step35's layers 43-44).
    ///   * `Some(l)`-> `min(silu(gate*gs), l) * clamp(up*us, +-l)` (llama-graph.cpp:2146/1751,
    ///                 non-DEEPSEEK4 branch). Callers source it from `cfg.clamp_exp_at(il)`
    ///                 (routed experts) or `cfg.clamp_shexp_at(il)` (shared expert) — the two
    ///                 arrays are SEPARATE and a layer can have one without the other.
    /// The `> 1e-6` eps gate lives in `clamp_exp_at`/`clamp_shexp_at`, so a `Some` here is
    /// already known live.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn ffn_act_lim(e: &Engine, cfg: &ModelConfig, gate: &CudaSlice<f32>, up: &CudaSlice<f32>,
               gs: f32, us: f32, limit: Option<f32>, act: &mut CudaSlice<f32>, n: usize)
               -> Result<(), Box<dyn std::error::Error>> {
        if let Some(m3) = cfg.m3.as_ref() {
            debug_assert!(limit.is_none(), "m3 swigluoai and step35 clamp are different archs");
            return e.swigluoai_mul_scaled(gate, up, gs, us, m3.swiglu_alpha, m3.swiglu_limit, act, n);
        }
        if let Some(l) = limit {
            return e.swiglu_clamped_mul_scaled(gate, up, gs, us, l, act, n);
        }
        if gs == 1.0 && us == 1.0 { return e.silu_mul(gate, up, act, n); }
        e.silu_mul_scaled(gate, up, gs, us, act, n)
    }

    /// Routing for the whole batch: returns (sel [T*n_used] expert ids, w [T*n_used] renorm weights),
    /// token-major. Default = the Stage-1 host path (dtoh logits, softmax-256, stable DESC top-k,
    /// renorm). MEMRA_FUSED_ROUTER = the device kernel (§A) which reproduces the same numerics; we
    /// still dtoh the tiny [T,n_used] sel/w buffers (64 B/token vs 1 KB/token) — the host loop
    /// indexes HostExps.bytes on the CPU to choose the DMA source (§A.2 output staging).
    fn moe_route(e: &Engine, logits: &CudaSlice<f32>, t: usize, n_expert: usize, n_used: usize)
                 -> Result<(Vec<u32>, Vec<f32>), Box<dyn std::error::Error>> {
        Self::moe_route_cfg(e, logits, t, n_expert, n_used, None, None, None)
    }

    /// DeepSeek-V3-class sigmoid routing (MiniMax-M3, Hy3), host oracle. Reference:
    /// M3/Hy3 modeling code — scores = sigmoid(logits); selection over scores + expert bias
    /// (M3 `e_score_correction_bias` / Hy3 `expert_bias`, both surfaced as `exp_probs_b`);
    /// weights = un-biased scores of the selected experts, sum-normalized when `route_norm`,
    /// x scaling factor (M3 routed_scaling_factor 2.0 / Hy3 router_scaling_factor 2.826).
    /// `sig` = (scaling_factor, route_norm) from `cfg.sigmoid_router()`; softmax archs pass
    /// None -> the qwen35moe/OLMoE path below.
    fn moe_route_cfg(e: &Engine, logits: &CudaSlice<f32>, t: usize, n_expert: usize, n_used: usize,
                     bias: Option<&[f32]>, sig: Option<(f32, bool)>, active: Option<&[bool]>)
                 -> Result<(Vec<u32>, Vec<f32>), Box<dyn std::error::Error>> {
        if let Some((sf, route_norm)) = sig {
            // sigmoid routing. Host path only for now (fused-router kernel is softmax-top-k).
            let lg = e.dtoh(logits)?;
            return Self::moe_route_sigmoid_host(
                &lg, t, n_expert, n_used, bias, sf, route_norm, active,
            );
        }
        // LAUNCH-STRUCTURE STAGE 1 (2026-07-05): fused router DEFAULT ON (MEMRA_FUSED_ROUTER=0
        // rollback) via the single-sync pinned readback — softmax arch only; the M3 sigmoid arm
        // above returns before this (host path until a sigmoid fused-router kernel exists).
        if active.is_none() && !matches!(std::env::var("MEMRA_FUSED_ROUTER").as_deref(), Ok("0")) {
            return e.moe_router_topk_host(logits, t, n_expert, n_used);
        }
        // Host oracle (the §D bit-identity reference).
        let lg = e.dtoh(logits)?;   // [T*n_expert] host
        let mut sel = vec![0u32; t * n_used];
        let mut w_out = vec![0f32; t * n_used];
        for tok in 0..t {
            let row = &lg[tok * n_expert..(tok + 1) * n_expert];
            // softmax over ALL n_expert (stable: subtract max)
            let maxl = row.iter().enumerate()
                .filter(|(i, _)| active.is_none_or(|mask| mask[*i]))
                .map(|(_, &x)| x).fold(f32::NEG_INFINITY, f32::max);
            let mut probs = vec![0f32; n_expert];
            let mut den = 0f32;
            for i in 0..n_expert {
                if active.is_some_and(|mask| !mask[i]) { continue; }
                let x = (row[i] - maxl).exp(); probs[i] = x; den += x;
            }
            for p in probs.iter_mut() { *p /= den; }
            // stable DESC sort: prob DESC, ascending-index tiebreak.
            let mut idx: Vec<usize> = (0..n_expert)
                .filter(|&i| active.is_none_or(|mask| mask[i])).collect();
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

    #[allow(clippy::too_many_arguments)]
    fn moe_route_sigmoid_with_input(
        e: &Engine,
        logits: &CudaSlice<f32>,
        input: &CudaSlice<f32>,
        t: usize,
        n_expert: usize,
        n_used: usize,
        bias: Option<&[f32]>,
        (sf, route_norm): (f32, bool),
        active: Option<&[bool]>,
    ) -> Result<(Vec<u32>, Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
        let (lg, input) = e.dtoh_pair(logits, input)?;
        let (sel, w) =
            Self::moe_route_sigmoid_host(&lg, t, n_expert, n_used, bias, sf, route_norm, active)?;
        Ok((sel, w, input))
    }

    /// Start the prediction-guided prefetch worker (MEMRA_MOE_PREFETCH=depth). Call after
    /// residency freeze: the worker filters against a static snapshot of the frozen HBM set.
    /// Builds a fully-owned per-layer table (host router copies via one-time DtoH, bias,
    /// active mask, prebuilt projection descriptors) so no model reference escapes.
    pub fn start_moe_prefetch_predictor(
        &self,
        e: &Engine,
        cfg: &ModelConfig,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use crate::hybrid::Ffn;
        let Some(sig) = cfg.sigmoid_router() else {
            return Err("prefetch predictor requires a sigmoid-router arch".into());
        };
        let resident: std::collections::HashSet<(u16, u8, u16)> = e
            .export_moe_residency()
            .ok_or("prefetch predictor needs the frozen MoE residency cache")?
            .into_iter()
            .collect();
        let mut layers = Vec::new();
        for (index, layer) in self.layers.iter().enumerate() {
            let Ffn::Moe(m) = &layer.ffn else { continue };
            let crate::model::GpuTensor::Float { data, .. } = &m.gate_inp else { continue };
            let router = e.dtoh(data)?;
            let n_expert = m.gate_exps.n_expert;
            let n_embd = m.gate_exps.in_f;
            if router.len() != n_embd * n_expert {
                continue;
            }
            let build = |exps: &crate::model::HostExps| {
                (0..n_expert)
                    .map(|expert| crate::cpu_experts::predictor_projection(exps, expert))
                    .collect::<Vec<_>>()
            };
            layers.push((index as u16, crate::cpu_experts::PredictLayerInit {
                router,
                bias: m.exp_probs_b.clone(),
                active: m.active_experts.clone(),
                n_embd,
                n_used: cfg
                    .moe
                    .as_ref()
                    .map(|moe| moe.expert_used_count as usize)
                    .ok_or("prefetch predictor requires MoE config")?,
                sig,
                weights_n_expert: n_expert,
                gate: build(&m.gate_exps),
                up: build(&m.up_exps),
                down: build(&m.down_exps),
            }));
        }
        crate::cpu_experts::start_prefetch_predictor(layers, resident)
            .map_err(|error| error.into())
    }

    /// Crate-visible sigmoid-routing oracle for the prefetch predictor: identical selection
    /// math to the runtime router, applied to host-computed lookahead logits.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn moe_route_sigmoid_host_public(
        logits: &[f32],
        t: usize,
        n_expert: usize,
        n_used: usize,
        bias: Option<&[f32]>,
        sf: f32,
        route_norm: bool,
        active: Option<&[bool]>,
    ) -> Result<(Vec<u32>, Vec<f32>), Box<dyn std::error::Error>> {
        Self::moe_route_sigmoid_host(logits, t, n_expert, n_used, bias, sf, route_norm, active)
    }

    #[allow(clippy::too_many_arguments)]
    fn moe_route_sigmoid_host(
        lg: &[f32],
        t: usize,
        n_expert: usize,
        n_used: usize,
        bias: Option<&[f32]>,
        sf: f32,
        route_norm: bool,
        active: Option<&[bool]>,
    ) -> Result<(Vec<u32>, Vec<f32>), Box<dyn std::error::Error>> {
        if lg.len() != t * n_expert {
            return Err(format!(
                "sigmoid router logits length mismatch: got {}, expected {}",
                lg.len(),
                t * n_expert,
            )
            .into());
        }
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
            let mut idx: Vec<usize> = (0..n_expert)
                .filter(|&i| active.is_none_or(|mask| mask[i]))
                .collect();
            idx.sort_by(|&a, &b| selsc[b].total_cmp(&selsc[a]).then(a.cmp(&b)));
            let sl = &idx[..n_used];
            let mut wv: Vec<f32> = sl.iter().map(|&i| scores[i]).collect();
            if route_norm {
                let ws: f32 = wv.iter().sum::<f32>().max(1e-20);
                for x in wv.iter_mut() {
                    *x = *x / ws * sf;
                }
            } else {
                for x in wv.iter_mut() {
                    *x *= sf;
                }
            }
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
        // This arm has no `il` and every kernel in it fuses PLAIN silu(gate)*up, so a clamped
        // model must never reach it. The caller's gate at `moe_ffn_sequential_zq8` denies per
        // layer via `swiglu_clamped_at(il)`; assert the whole-model form here so a future caller
        // that forgets the gate fails loudly in debug instead of returning wrong logits.
        debug_assert!(!cfg.swiglu_clamped_anywhere(),
                      "moe_ffn_pairs has no per-layer clamp: fused epilogues are plain SiLU");
        let dev = m.dev_exps.as_ref().unwrap();
        // WALL-GAP ARC: interleaved gate/up slab strides (see moe_ffn_dev).
        let (rbg_d, rbu_d) = if dev.gu_il {
            let sxx = m.gate_exps.row_bytes + m.up_exps.row_bytes; (sxx, sxx)
        } else { (m.gate_exps.row_bytes, m.up_exps.row_bytes) };

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

        // INT8-MMA EXPERT MMQ (MEMRA_MOE_MMA=1, opt-in): the m16n8k16.s8 tensor-core analog of the
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
        // (dispatch parity with the T=1 decode chain). MEMRA_MOE_MMA=0 rollback;
        // MEMRA_MOE_MMA_T overrides the floor (bisect seam).
        static MMA_T: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        let mma_t = *MMA_T.get_or_init(|| {
            std::env::var("MEMRA_MOE_MMA_T").ok().and_then(|v| v.parse().ok()).unwrap_or(16)
        });
        let use_mma = std::env::var("MEMRA_MOE_MMA").map(|v| v != "0").unwrap_or(true)
            && t >= mma_t
            && q8_expert_dec_supported(m.gate_exps.qtype) && q8_expert_dec_supported(m.up_exps.qtype)
            && q8_expert_dec_supported(m.down_exps.qtype)
            && n_embd % 256 == 0 && n_ff_exp % 256 == 0;
        // GROUPED f16 LANE admission (MEMRA_MOE_F16G, rounds 46-49): its own door, no longer a
        // subset of the MMQ arm — q35's k-quant stragglers (Q6_K/Q4_K down, one Q3_K gate/up
        // layer) fail q8_expert_dec_supported but dequant fine to f16, so f16g must be able to
        // take a layer the MMQ arm would reject. Same t >= mma_t floor as MMA: decode and
        // spec-verify batches must ride the dp4a path whose FP order matches the T=1 chain.
        // MODE 2 (sm_120a naked default, lane/f16g-default-rearb 2026-08-02): every layer
        // whose three projections pass f16g_proj_ok rides the sk visitor with direct tile
        // loaders — with IQ4_XS/IQ3_S direct coverage the sk arm beats the int8-MMA MMQ
        // tiles on the IQ-bank models too (q35 board-2048 +33.9%, KAT pp512 +46.7%).
        // AUTO-KQUANT (mode 3, MEMRA_MOE_F16G=3, lane/q4k-expert-prefill): admit f16g ONLY
        // where the MMA arm can't take the layer's QTYPES — the k-quant expert class whose
        // baseline is the per-pair _em fallback (Ornith-35B Q4_K: board-2048 3.14x). Its
        // MMA-capable carve-out (IQ3_S/IQ4_XS/Q4_0 banks on MMQ) was priced pre-IQ-direct;
        // it survives as the rollback seam. Keyed on qtype capability, NOT use_mma, so
        // MEMRA_MOE_MMA=0 stays a pure dp4a rollback seam.
        let mma_capable = q8_expert_dec_supported(m.gate_exps.qtype)
            && q8_expert_dec_supported(m.up_exps.qtype)
            && q8_expert_dec_supported(m.down_exps.qtype)
            && n_embd % 256 == 0 && n_ff_exp % 256 == 0;
        let f16g_mode = crate::moe_f16g_mode();
        let f16g = f16g_mode != 0 && t >= mma_t
            && (f16g_mode != 3 || !mma_capable)
            && f16g_proj_ok(m.gate_exps.qtype, n_embd)
            && f16g_proj_ok(m.up_exps.qtype, n_embd)
            && f16g_proj_ok(m.down_exps.qtype, n_ff_exp);
        if use_mma || f16g {
            // GROUPED f16 LANE (MEMRA_MOE_F16G, rounds 46-49, experimental door): dequant
            // the active experts ONCE per projection to f16 and run one grouped f16 GEMM over
            // the CSR groups. f16-mirror numeric class — argmax/spec gated before promotion.
            // NOTE: the pair-gather kernel needs pair p ordered EXPERT-MAJOR (ex_pairs order);
            // the y rows come back in that same CSR order, so gate/up/down all stay pair-major
            // in ex_pairs order — but moe_pairs_silu_mul and the scatter consume PAIR-ID order.
            // We therefore gather activations per ex_pairs and scatter y back through ex_pairs.
            let y_down = if f16g {
                // CSR order end-to-end: gather z rows by the pair's TOKEN (pair p's token is
                // p / n_used — the trivial CSR above), silu in CSR order (elementwise), one
                // permute at the very end back to pair-id order for the scatter.
                let csr_tok: Vec<i32> = ex_pairs.iter().map(|&p| p / n_used as i32).collect();
                let csr_tok_d = e.htod_i32(&csr_tok)?;
                let (z_f16, z_s) = e.moe_f16g_act(z, Some(&csr_tok_d), n_embd, n_pairs)?;
                let g_csr = e.moe_f16_grouped(&dev.ptr_row, 0, n_expert, &exi, &ex_off, &exo,
                                              &z_f16, &z_s, n_embd, n_ff_exp, n_active, n_pairs,
                                              m.gate_exps.qtype, rbg_d)?;
                let u_csr = e.moe_f16_grouped(&dev.ptr_row, 1, n_expert, &exi, &ex_off, &exo,
                                              &z_f16, &z_s, n_embd, n_ff_exp, n_active, n_pairs,
                                              m.up_exps.qtype, rbu_d)?;
                let act_csr = e.moe_pairs_silu_mul(&g_csr, &u_csr, n_pairs * n_ff_exp)?;
                let (a_f16, a_s) = e.moe_f16g_act(&act_csr, None, n_ff_exp, n_pairs)?;
                let d_csr = e.moe_f16_grouped(&dev.ptr_row, 2, n_expert, &exi, &ex_off, &exo,
                                              &a_f16, &a_s, n_ff_exp, n_embd, n_active, n_pairs,
                                              m.down_exps.qtype, m.down_exps.row_bytes)?;
                e.rows_permute(&d_csr, &exp_d, n_pairs, n_embd)?
            } else {
            // gate/up: activation = z, token-major over t tokens; pair_tok gathers the routed row.
            let z_scr = e.mmq_iq_quantize_act(z, n_embd, t)?;
            let gate = e.mmq_iq_experts(&dev.ptr_row, 0, n_expert, &exi, &exo, &exp_d, &pt, &z_scr,
                                        n_embd, n_ff_exp, n_active, n_pairs, t,
                                        m.gate_exps.qtype, rbg_d)?;
            let up = e.mmq_iq_experts(&dev.ptr_row, 1, n_expert, &exi, &exo, &exp_d, &pt, &z_scr,
                                      n_embd, n_ff_exp, n_active, n_pairs, t,
                                      m.up_exps.qtype, rbu_d)?;
            // down: activation = silu(gate)*up, pair-major [n_pairs, n_ff_exp]; pair_tok =
            // identity. FUSED ACT-EPILOGUE (default on): one launch computes the activation in
            // registers and writes ONLY the quantized scratch — the two-pass chain
            // (moe_pairs_silu_mul writes act f32, mmq_iq_quantize_act re-reads it) is the
            // MEMRA_MOE_FUSE_ACTQ=0 rollback. Scratch bytes are BYTE-IDENTICAL (kernel-check).
            let a_scr = if crate::moe_fuse_actq_on() {
                e.mmq_iq_fused_act_quant(&gate, &up, n_ff_exp, n_pairs, 0)?
            } else {
                let act = e.moe_pairs_silu_mul(&gate, &up, n_pairs * n_ff_exp)?;
                e.mmq_iq_quantize_act(&act, n_ff_exp, n_pairs)?
            };
            let pair_self: Vec<i32> = (0..n_pairs as i32).collect();
            let pself = e.htod_i32(&pair_self)?;
            e.mmq_iq_experts(&dev.ptr_row, 2, n_expert, &exi, &exo, &exp_d, &pself, &a_scr,
                             n_ff_exp, n_embd, n_active, n_pairs, n_pairs,
                             m.down_exps.qtype, m.down_exps.row_bytes)?
            };
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
                // SERVE ISOLATION (lane/concat-prime-exact): the fused sigmoid-dot is the
                // m-INVARIANT form — see the sequential arm's note. This is the PAIRS arm,
                // i.e. the one real prefill actually takes on a resident-expert MoE model,
                // so the concat-prime isolation fix has to land here as well.
                let g = match &m.gate_inp_shexp {
                    Some(gate_inp_shexp) if crate::router_prefill_exact_on() => {
                        e.sigmoid_dot_rows(z, gate_inp_shexp.float_data(), n_embd, t)?
                    }
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

        // DECODE-ONCE MMQ (rung 3, MEMRA_MOE_DEC=1 default-on): dequant each weight group once per
        // (row,group) then dp4a across the expert's tokens. _em re-decoded per token (NEUTRAL).
        let dec = std::env::var("MEMRA_MOE_DEC").map(|v| v != "0").unwrap_or(true);
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
                          n_embd, n_ff_exp, m.gate_exps.qtype, rbg_d)?;
        let up = matvec(1, &exi, &exo, &exp_d, &pt, &zq, &zd,
                        n_embd, n_ff_exp, m.up_exps.qtype, rbu_d)?;
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
            // m-INVARIANT fused sigmoid-dot under router_prefill_exact (serve isolation,
            // lane/concat-prime-exact) — every shexp-gate arm shares the same form so a
            // dispatch choice cannot change bits.
            let g = match &m.gate_inp_shexp {
                Some(gate_inp_shexp) if crate::router_prefill_exact_on() => {
                    e.sigmoid_dot_rows(z, gate_inp_shexp.float_data(), n_embd, t)?
                }
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
        // moe_router_topk is SOFTMAX-only (no exp_probs_b bias, no expert_weights_scale) and every
        // gate_up kernel here fuses PLAIN silu(gate)*up. `dev_ok` denies sigmoid-router archs and
        // clamped layers; assert both so a future caller that skips the gate fails loudly.
        debug_assert!(cfg.sigmoid_router().is_none(),
                      "moe_ffn_dev routes SOFTMAX: a sigmoid-router arch would pick wrong experts");
        debug_assert!(!cfg.swiglu_clamped_at(il as u32),
                      "moe_ffn_dev's fused epilogue is plain SiLU: no clamped form");

        // device top-k: sel [t, n_used] i32, w [t, n_used] f32 — stays on device.
        let (sel_d, mut w_d) = e.moe_router_topk(logits, t, n_expert, n_used)?;
        // Down-projection macro fold (compressed-tensors NVFP4 artifacts): one tiny launch,
        // skipped entirely for macro-free experts (every k-quant GGUF).
        if m.has_macros {
            e.moe_w_scale_by_expert(&mut w_d, &sel_d, &m.dev_macros, n_expert, t * n_used)?;
        }

        // moe_out rows are FULLY overwritten by moe_down8_fma_dev — uninit (stage-2 rule).
        let mut moe_out = e.uninit(t * n_embd)?;

        // RESIDENT-EXPERTS arm: the pointer row comes from the load-time slab (no cache, no
        // lock). Same kernels/loop as the SLRU arm below — only the row's provenance differs.
        if let Some(dev) = m.dev_exps.as_ref() {
            // WALL-GAP ARC: interleaved gate/up slab (MEMRA_MOE_GU_IL) -> both projections use
            // the combined stride; up's base is offset in the ptr table. Down unchanged.
            let (rbg_d, rbu_d) = if dev.gu_il {
                let sxx = m.gate_exps.row_bytes + m.up_exps.row_bytes; (sxx, sxx)
            } else { (m.gate_exps.row_bytes, m.up_exps.row_bytes) };
            let q8 = moe_q8_enabled()
                && q8_expert_supported(m.gate_exps.qtype) && q8_expert_supported(m.up_exps.qtype)
                && q8_expert_supported(m.down_exps.qtype);
            // SMALL-M ROWS ARM (MEMRA_SPEC_M2, lane/spec-m2): batch the verify token loop —
            // ONE batched z-quantize + ONE gate_up rows launch + ONE act quantize + ONE down
            // rows launch (4 launches/layer, was 4t). BIT-IDENTICAL per token to the serial
            // loop below (rows twins = the _v/w8h2v per-token programs on a grid.z token axis;
            // quantize_q8_1 is per-32-block row-independent). Gated to the AUTO kernel modes —
            // a custom MEMRA_MOE_DEVQ8_GU/DOWN diagnostic run keeps the serial loop so the
            // dispatched kernel stays exactly the env-selected one — and to the w8h2v shape
            // (n_ff_exp==512, n_used<=8), the same contract the AUTO down dispatch keys on.
            let rows_arm = q8 && t > 1 && crate::spec::spec_m2()
                && n_ff_exp == 512 && n_used <= 8
                && std::env::var("MEMRA_MOE_DEVQ8_GU").map(|v| v.is_empty() || v == "v").unwrap_or(true)
                && std::env::var("MEMRA_MOE_DEVQ8_DOWN").map(|v| v.is_empty() || v == "w8h2v").unwrap_or(true);
            // CSR EXPERT-DEDUP gate_up (verify-cost target #1, DEFAULT ON 2026-07-10): the
            // owner-scan kernel serves every (token, slot) pair of an expert from ONE block,
            // deduping the 38-40% duplicated weight-stream+decode the overlap probe measured
            // (gate_up 55.0 -> 39.7us/launch; +1.3-2.1% spec e2e p2, +0.6-1.7% p3, all K).
            // Bit-identical to the _rows twins (explicit-intrinsic accumulate — the ULP/fmad
            // lesson; =2 byte-compare verified zero diffs). down stays on _rows (CSR down
            // measured 23.5 -> 37.5us: 16-group rows can't amortize the serial pair loop).
            // MEMRA_MOE_CSR=0 rollback; =2 runs BOTH paths and byte-compares (debug).
            let csr_mode = std::env::var("MEMRA_MOE_CSR").ok()
                .and_then(|v| v.parse::<i32>().ok()).unwrap_or(1);
            let csr_qt = |qt: i32| qt == crate::QT_IQ4_XS || qt == crate::QT_IQ3_S;
            let csr_arm = rows_arm && csr_mode > 0 && t <= 10
                && csr_qt(m.gate_exps.qtype) && csr_qt(m.up_exps.qtype)
                && csr_qt(m.down_exps.qtype);
            if csr_arm {
                if csr_mode == 2 {
                    static ENGAGED: std::sync::Once = std::sync::Once::new();
                    ENGAGED.call_once(|| eprintln!("[memra] moe CSR byte-compare mode ON (t={t})"));
                }
                let n_pairs = t * n_used;
                let (zq, zd) = e.quantize_q8_1(z, t, n_embd)?;
                let act = e.moe_gate_up_silu8_dev_q8_csr(&dev.ptr_row, &sel_d, &zq, &zd, n_pairs,
                                                         n_embd, n_ff_exp, n_used, n_expert,
                                                         m.gate_exps.qtype, m.up_exps.qtype,
                                                         rbg_d, rbu_d)?;
                let (aq2, ad2) = e.quantize_q8_1(&act, n_pairs, n_ff_exp)?;
                // down stays on the _rows twin — BOTH CSR down variants measured negative
                // (v1 serial pairs 23.5->37.5us; v2 warp-parallel+SMEM -14% e2e, K=8 -37%):
                // 16-group rows have too little decode to amortize any dedup structure.
                e.moe_down8_fma_dev_q8_rows(&dev.ptr_row, &sel_d, &w_d, &aq2, &ad2, &mut moe_out,
                                            t, n_ff_exp, n_embd, n_used, n_expert,
                                            m.down_exps.qtype, m.down_exps.row_bytes)?;
                if csr_mode == 2 {
                    // DEBUG BYTE-COMPARE: run the _rows twins on the same inputs, diff bits.
                    let act_r = e.moe_gate_up_silu8_dev_q8_rows(&dev.ptr_row, &sel_d, &zq, &zd, t,
                                                                n_embd, n_ff_exp, n_used, n_expert,
                                                                m.gate_exps.qtype, m.up_exps.qtype,
                                                                rbg_d, rbu_d, &m.dev_macros)?;
                    let mut out_r = e.uninit(t * n_embd)?;
                    let (aq2r, ad2r) = e.quantize_q8_1(&act_r, n_pairs, n_ff_exp)?;
                    e.moe_down8_fma_dev_q8_rows(&dev.ptr_row, &sel_d, &w_d, &aq2r, &ad2r, &mut out_r,
                                                t, n_ff_exp, n_embd, n_used, n_expert,
                                                m.down_exps.qtype, m.down_exps.row_bytes)?;
                    let (a1, a2) = (e.dtoh(&act)?, e.dtoh(&act_r)?);
                    let (o1, o2) = (e.dtoh(&moe_out)?, e.dtoh(&out_r)?);
                    let ba = a1.iter().zip(&a2).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
                    let bo = o1.iter().zip(&o2).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
                    if ba + bo > 0 {
                        eprintln!("[csr-check] il={il} t={t} ACT diffs={ba}/{} OUT diffs={bo}/{}",
                                  a1.len(), o1.len());
                        // First 4 differing ACT elements: (pair, o, csr, rows) + that pair's expert
                        let sel_h = e.dtoh_i32(&sel_d)?;
                        let mut shown = 0;
                        for (i, (x, y)) in a1.iter().zip(&a2).enumerate() {
                            if x.to_bits() != y.to_bits() && shown < 4 {
                                let (p, o) = (i / n_ff_exp, i % n_ff_exp);
                                let ex = sel_h[p];
                                let npx = sel_h.iter().filter(|&&v| v == ex).count();
                                eprintln!("  ACT p={p} ex={ex} np={npx} o={o} csr={x:e} rows={y:e}");
                                shown += 1;
                            }
                        }
                        std::process::exit(3);
                    }
                }
            } else if rows_arm {
                // TEMP PROBE (MEMRA_MOE_OVERLAP=1): cross-token expert-activation overlap at
                // verify — sizes the CSR dedup win (unique experts vs t*n_used pairs).
                if std::env::var("MEMRA_MOE_OVERLAP").as_deref() == Ok("1") {
                    use std::sync::atomic::{AtomicU64, Ordering};
                    static PAIRS: AtomicU64 = AtomicU64::new(0);
                    static UNIQ: AtomicU64 = AtomicU64::new(0);
                    static CALLS: AtomicU64 = AtomicU64::new(0);
                    let sel_h = e.dtoh_i32(&sel_d)?;
                    let mut u: Vec<i32> = sel_h.clone(); u.sort_unstable(); u.dedup();
                    PAIRS.fetch_add(sel_h.len() as u64, Ordering::Relaxed);
                    UNIQ.fetch_add(u.len() as u64, Ordering::Relaxed);
                    let c = CALLS.fetch_add(1, Ordering::Relaxed) + 1;
                    if c % 480 == 0 {
                        let p = PAIRS.load(Ordering::Relaxed); let q = UNIQ.load(Ordering::Relaxed);
                        eprintln!("[overlap] calls={c} pairs={p} unique={q} ratio={:.3} (t={t})",
                                  q as f64 / p as f64);
                    }
                }
                let (zq, zd) = e.quantize_q8_1(z, t, n_embd)?;
                let act = e.moe_gate_up_silu8_dev_q8_rows(&dev.ptr_row, &sel_d, &zq, &zd, t,
                                                          n_embd, n_ff_exp, n_used, n_expert,
                                                          m.gate_exps.qtype, m.up_exps.qtype,
                                                          rbg_d, rbu_d, &m.dev_macros)?;
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
                                                         rbg_d, rbu_d, &m.dev_macros)?;
                    let (aq2, ad2) = e.quantize_q8_1(&act, n_used, n_ff_exp)?;
                    e.moe_down8_fma_dev_q8(&dev.ptr_row, &selt, &wt, &aq2, &ad2, &mut dst,
                                           n_ff_exp, n_embd, n_used, n_expert,
                                           m.down_exps.qtype, m.down_exps.row_bytes)?;
                } else {
                    let act = e.moe_gate_up_silu8_dev(&dev.ptr_row, &selt, &zt, n_embd, n_ff_exp,
                                                      n_used, n_expert,
                                                      m.gate_exps.qtype, m.up_exps.qtype,
                                                      rbg_d, rbu_d, &m.dev_macros)?;
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
        // arm above; MEMRA_MOE_Q8=0 restores the byte-identical f32 path.
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
                                                           m.gate_exps.row_bytes, m.up_exps.row_bytes,
                                                           &m.dev_macros)?;
                    let (aq2, ad2) = eng.quantize_q8_1(&act, n_used, n_ff_exp)?;
                    eng.moe_down8_fma_dev_q8(row, &selt, &wt, &aq2, &ad2, &mut dst,
                                             n_ff_exp, n_embd, n_used, n_expert,
                                             m.down_exps.qtype, m.down_exps.row_bytes)?;
                } else {
                    let act = eng.moe_gate_up_silu8_dev(row, &selt, &zt, n_embd, n_ff_exp,
                                                        n_used, n_expert,
                                                        m.gate_exps.qtype, m.up_exps.qtype,
                                                        m.gate_exps.row_bytes, m.up_exps.row_bytes,
                                                        &m.dev_macros)?;
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
                // VERIFY-TIER TRUNK FUSION (MEMRA_SPEC_FUSED_T, t=2-4): the shexp gate+up pair
                // rides one shared quantize + one fused2 batched launch instead of two
                // decode-exact calls. Bit-identical per (tensor,token,row) — see spec_fused_t.
                let mut fused = None;
                if crate::spec::spec_fused_t() && (2..=4).contains(&t)
                    && e.uses_q8_1_fast(gate_shexp) && e.uses_q8_1_fast(up_shexp) {
                    let (zq, zd) = e.quantize_q8_1(z, t, n_embd)?;
                    fused = e.matmul_q8_fused2_t(gate_shexp, up_shexp, &zq, &zd, t)?;
                }
                match fused {
                    Some(pair) => pair,
                    None => (e.matmul_decode_exact(gate_shexp, z, t)?,
                             e.matmul_decode_exact(up_shexp, z, t)?),
                }
            } else {
                (e.matmul(gate_shexp, z, t)?, e.matmul(up_shexp, z, t)?)
            };
            let mut sa = e.uninit(t * n_ff_sh)?;  // silu_mul fully overwrites
            e.silu_mul(&sg_gate, &sg_up, &mut sa, t * n_ff_sh)?;
            let sh = if verify_t { e.matmul_decode_exact(down_shexp, &sa, t)? }
                     else { e.matmul(down_shexp, &sa, t)? };
            // shexp gate: qwen35moe sigmoid-gates; M3 has no gate tensor -> weight 1.0.
            // Same fused sigmoid-dot as moe_ffn_sequential step 3 (byte-identity contract
            // between the two arms; prefill keeps the batched cuBLASLt linear).
            let g = match &m.gate_inp_shexp {
                Some(gate_inp_shexp) => {
                    // router_prefill_exact_on(): the fused sigmoid-dot is m-INVARIANT and
                    // serves EVERY t (serve isolation, lane/concat-prime-exact).
                    if t < PRIME_MIN_T || crate::router_prefill_exact_on() {
                        e.sigmoid_dot_rows(z, gate_inp_shexp.float_data(), n_embd, t)?
                    } else {
                        let gs = e.linear(z, gate_inp_shexp.float_data(), t, n_embd, 1)?;
                        let mut g = e.uninit(t)?;
                        e.sigmoid(&gs, &mut g, t)?;
                        g
                    }
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
                let __s = eng.stream();
                let (pg, _e0) = c.slot(sg).device_ptr(&__s);
                let (pu, _e1) = c.slot(su).device_ptr(&__s);
                let (pd, _e2) = c.slot(sd).device_ptr(&__s);
                g[j] = pg as u64; u[j] = pu as u64; d[j] = pd as u64;
            }
            if cpu_expert_profile_admit_enabled() && !c.is_frozen() {
                for &ex in sel {
                    let ex = ex as u16;
                    for proj in [PROJ_GATE, PROJ_UP, PROJ_DOWN] {
                        c.note_profile_hit(BlockId::new(il, proj, ex));
                    }
                }
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
                let __s = eng.stream();
                let (pg, _e0) = c.slot(sg).device_ptr(&__s);
                let (pu, _e1) = c.slot(su).device_ptr(&__s);
                let (pd, _e2) = c.slot(sd).device_ptr(&__s);
                g[j] = pg as u64; u[j] = pu as u64; d[j] = pd as u64;
            }
            if cpu_expert_profile_admit_enabled() && !c.is_frozen() {
                for &ex in sel {
                    let ex = ex as u16;
                    for proj in [PROJ_GATE, PROJ_UP, PROJ_DOWN] {
                        c.note_profile_hit(BlockId::new(il, proj, ex));
                    }
                }
            }
            c.hits += (3 * n_used) as u64; // instrumentation parity with dispatch()
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
        let layout = exps.expert_layout(ex);
        let id = BlockId::new(il, proj, ex as u16);
        let source = exps.expert_source(ex);
        e.with_moe_cache(max_block, |c, eng| {
            let slot = c.dispatch_source(id, source, eng)?;
            let DispatchSlot::Resident(sl) = slot;
            let buf = c.slot(sl);
            eng.qmatvec_expert_q8(buf, 0..layout.len, aq, ad, 1, exps.in_f, exps.out_f,
                                  layout.qtype, layout.row_bytes)
        })
    }

    fn moe_cached_gemm(e: &Engine, il: u16, proj: u8, ex: usize, m: &MoeWeights,
                       max_block: usize, x: &cudarc::driver::CudaView<f32>)
                       -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        use crate::moe_cache::{BlockId, DispatchSlot, PROJ_GATE, PROJ_UP};
        let exps = match proj { PROJ_GATE => &m.gate_exps, PROJ_UP => &m.up_exps, _ => &m.down_exps };
        let layout = exps.expert_layout(ex);
        let id = BlockId::new(il, proj, ex as u16);
        let source = exps.expert_source(ex);
        // dispatch under the lock (lookup/admit/memcpy-issue), then resolve the slot and GEMM.
        e.with_moe_cache(max_block, |c, eng| {
            let slot = c.dispatch_source(id, source, eng)?;
            // resolve the device buffer for this slot; the GEMM is enqueued on the compute stream
            // (the same stream the memcpy was issued on, so ordering holds without extra sync).
            let DispatchSlot::Resident(sl) = slot;
            let buf = c.slot(sl);
            eng.qmatvec_view(buf, 0..layout.len, x, 1, exps.in_f, exps.out_f,
                             layout.qtype, layout.row_bytes)
        })
    }

    /// Populate the warmup cache for an expert whose current-token output intentionally ran on
    /// CPU. No GEMM is launched and callers invoke this only after the CPU result has completed,
    /// so the current forward's backend assignment and output remain unchanged.
    fn moe_profile_admit_expert(
        e: &Engine,
        il: u16,
        ex: usize,
        m: &MoeWeights,
        max_block: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use crate::moe_cache::{BlockId, PROJ_DOWN, PROJ_GATE, PROJ_UP};
        e.with_moe_cache(max_block, |cache, eng| {
            for (proj, exps) in [
                (PROJ_GATE, &m.gate_exps),
                (PROJ_UP, &m.up_exps),
                (PROJ_DOWN, &m.down_exps),
            ] {
                let id = BlockId::new(il, proj, ex as u16);
                let _ = cache.dispatch_source(id, exps.expert_source(ex), eng)?;
            }
            Ok(())
        })
    }

    /// Read a projection from the immutable residency set when present; otherwise use one
    /// transient slot without admitting or evicting anything. Used only by post-freeze prefill.
    #[allow(clippy::too_many_arguments)]
    fn moe_frozen_gemm(
        e: &Engine,
        il: u16,
        proj: u8,
        ex: usize,
        m: &MoeWeights,
        max_block: usize,
        x: &cudarc::driver::CudaView<f32>,
        scratch: &mut Option<CudaSlice<u8>>,
        scratch_len: usize,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        use crate::moe_cache::{BlockId, PROJ_GATE, PROJ_UP};
        let exps = match proj {
            PROJ_GATE => &m.gate_exps,
            PROJ_UP => &m.up_exps,
            _ => &m.down_exps,
        };
        let layout = exps.expert_layout(ex);
        let id = BlockId::new(il, proj, ex as u16);
        if let Some(output) = e.with_moe_cache(max_block, |cache, eng| {
            let Some(slot) = cache.resident(id) else {
                return Ok(None);
            };
            let buf = cache.slot(slot);
            Ok(Some(eng.qmatvec_view(
                buf,
                0..layout.len,
                x,
                1,
                exps.in_f,
                exps.out_f,
                layout.qtype,
                layout.row_bytes,
            )?))
        })? {
            return Ok(output);
        }
        if scratch.is_none() {
            *scratch = Some(e.alloc_u8_uninit(scratch_len)?);
        }
        let scratch = scratch.as_mut().unwrap();
        e.stage_expert(exps.expert_bytes(ex), scratch, 0)?;
        e.qmatvec_view(
            scratch,
            0..layout.len,
            x,
            1,
            exps.in_f,
            exps.out_f,
            layout.qtype,
            layout.row_bytes,
        )
    }

    fn moe_prefetch_expert(
        e: &Engine,
        il: u16,
        ex: usize,
        m: &MoeWeights,
        max_block: usize,
        keep: &[crate::moe_cache::BlockId],
    ) -> Result<(), Box<dyn std::error::Error>> {
        use crate::moe_cache::{BlockId, PROJ_DOWN, PROJ_GATE, PROJ_UP};
        e.with_moe_cache(max_block, |c, eng| {
            for (proj, exps) in [(PROJ_GATE, &m.gate_exps), (PROJ_UP, &m.up_exps),
                                 (PROJ_DOWN, &m.down_exps)] {
                let id = BlockId::new(il, proj, ex as u16);
                let _ = c.prefetch_source(id, exps.expert_source(ex), keep, eng)?;
            }
            Ok(())
        })
    }

    /// Worker-mode disk lookahead for grouped prefill. Memory sources are deliberately skipped so
    /// selecting `worker` changes only storage scheduling here; all CUDA work stays in dispatch.
    fn moe_prefetch_disk_expert(e: &Engine, il: u16, ex: usize, m: &MoeWeights,
                                max_block: usize, keep: &[crate::moe_cache::BlockId])
                                -> Result<(), Box<dyn std::error::Error>> {
        use crate::moe_cache::{BlockId, PROJ_DOWN, PROJ_GATE, PROJ_UP};
        e.with_moe_cache(max_block, |c, eng| {
            for (proj, exps) in [(PROJ_GATE, &m.gate_exps), (PROJ_UP, &m.up_exps),
                                 (PROJ_DOWN, &m.down_exps)] {
                let source = exps.expert_source(ex);
                if let crate::model::ExpertSource::Disk { .. } = &source {
                    let id = BlockId::new(il, proj, ex as u16);
                    let _ = c.prefetch_source(id, source, keep, eng)?;
                }
            }
            Ok(())
        })
    }

    #[inline]
    fn moe_prefetch_host_expert(ex: usize, m: &MoeWeights) {
        let _ = m.gate_exps.prefetch_expert_pages(ex);
        let _ = m.up_exps.prefetch_expert_pages(ex);
        let _ = m.down_exps.prefetch_expert_pages(ex);
    }
}

// ================================================================================================
// A2: EXPERT-GROUPED MoE PREFILL (MEMRA_MOE_GROUPED=1). Resident-case prototype.
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
    /// Resident-slab fast path for host-routed grouped prefill. Unclamped layers batch the
    /// sequential fused q8 program over the token axis; clamped layers use the separate
    /// expert-major q8 chain so `ffn_act_lim` remains authoritative.
    #[allow(clippy::too_many_arguments)]
    fn moe_ffn_grouped_resident_q8(
        e: &Engine,
        m: &MoeWeights,
        z: &CudaSlice<f32>,
        t: usize,
        cfg: &ModelConfig,
        il: u16,
        sel_all: &[u32],
        w_all: &[f32],
        table: &CudaSlice<u64>,
        gu_il: bool,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let moe = cfg.moe.as_ref().unwrap();
        let n_embd = cfg.n_embd as usize;
        let n_expert = moe.expert_count as usize;
        let n_used = moe.expert_used_count as usize;
        let n_ff_exp = moe.expert_ff_length as usize;
        let n_pairs = t * n_used;
        debug_assert_eq!(sel_all.len(), n_pairs);
        debug_assert_eq!(w_all.len(), n_pairs);
        debug_assert!(
            m.gate_exps.macros.is_none()
                && m.up_exps.macros.is_none()
                && m.down_exps.macros.is_none(),
            "resident grouped q8 does not fold per-expert macro scales",
        );

        // The rows twins run the resident sequential program verbatim on grid.z = token:
        // fused gate/up/SiLU per slot, batched activation quantization, then the original
        // slot-ordered down/FMA chain. Routing remains the host sigmoid oracle above; these
        // kernels consume sel/w only and never enter the softmax device router.
        if !cfg.swiglu_clamped_at(il as u32) {
            let sel: Vec<i32> = sel_all.iter().map(|&expert| expert as i32).collect();
            let sel_d = e.htod_i32(&sel)?;
            let w_d = e.htod(w_all)?;
            let (gate_row_bytes, up_row_bytes) = if gu_il {
                let combined = m.gate_exps.row_bytes + m.up_exps.row_bytes;
                (combined, combined)
            } else {
                (m.gate_exps.row_bytes, m.up_exps.row_bytes)
            };
            let (zq, zd) = e.quantize_q8_1(z, t, n_embd)?;
            let act = e.moe_gate_up_silu8_dev_q8_rows(
                table,
                &sel_d,
                &zq,
                &zd,
                t,
                n_embd,
                n_ff_exp,
                n_used,
                n_expert,
                m.gate_exps.qtype,
                m.up_exps.qtype,
                gate_row_bytes,
                up_row_bytes,
                &m.dev_macros,
            )?;
            let (aq2, ad2) = e.quantize_q8_1(&act, n_pairs, n_ff_exp)?;
            let mut moe_out = e.uninit(t * n_embd)?;
            e.moe_down8_fma_dev_q8_rows_g(
                table,
                &sel_d,
                &w_d,
                &aq2,
                &ad2,
                &mut moe_out,
                t,
                n_ff_exp,
                n_embd,
                n_used,
                n_expert,
                m.down_exps.qtype,
                m.down_exps.row_bytes,
            )?;

            if std::env::var("MEMRA_MOE_STATS").is_ok() {
                let mut counts = vec![0usize; n_expert];
                for &expert in sel_all {
                    counts[expert as usize] += 1;
                }
                let mut sizes: Vec<usize> =
                    counts.into_iter().filter(|&count| count != 0).collect();
                sizes.sort_unstable();
                let mean = sizes.iter().sum::<usize>() as f64 / sizes.len().max(1) as f64;
                println!(
                    "moe-grouped il={il} t={t} dispatch=resident-q8-rows active={}/{} \
                     m_e: min={} median={} mean={mean:.1} max={}",
                    sizes.len(),
                    n_expert,
                    sizes.first().copied().unwrap_or(0),
                    sizes.get(sizes.len() / 2).copied().unwrap_or(0),
                    sizes.last().copied().unwrap_or(0),
                );
            }
            return Ok(moe_out);
        }

        // Clamped layers keep gate/up, activation, and down as separate stages. The pair-major
        // matvec body is qmatvec_expert_q8 verbatim, while one launch covers every routed pair.
        // Pair ids stay in router slot order so scatter preserves the sequential FMA chain.
        let pair_tok: Vec<i32> = (0..n_pairs).map(|pair| (pair / n_used) as i32).collect();
        let pair_ex: Vec<i32> = sel_all.iter().map(|&expert| expert as i32).collect();
        let tok_off: Vec<i32> = (0..=t).map(|tok| (tok * n_used) as i32).collect();
        let tok_ids: Vec<i32> = (0..n_pairs as i32).collect();

        let mut by_expert: Vec<Vec<i32>> = vec![Vec::new(); n_expert];
        for (pair, &expert) in pair_ex.iter().enumerate() {
            by_expert[expert as usize].push(pair as i32);
        }

        let pair_tok_d = e.htod_i32(&pair_tok)?;
        let pair_ex_d = e.htod_i32(&pair_ex)?;
        let pair_w_d = e.htod(w_all)?;
        let tok_off_d = e.htod_i32(&tok_off)?;
        let tok_ids_d = e.htod_i32(&tok_ids)?;

        let matvec = |
            proj: i32,
            pair_rows: &CudaSlice<i32>,
            aq: &CudaSlice<i8>,
            ad: &CudaSlice<f32>,
            in_f: usize,
            out_f: usize,
            qtype: i32,
            row_bytes: usize,
        | -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
            e.moe_pairs_matvec_q8(
                table,
                proj,
                pair_rows,
                &pair_ex_d,
                aq,
                ad,
                in_f,
                out_f,
                n_expert,
                n_pairs,
                qtype,
                row_bytes,
            )
        };

        let (gate_row_bytes, up_row_bytes) = if gu_il {
            let combined = m.gate_exps.row_bytes + m.up_exps.row_bytes;
            (combined, combined)
        } else {
            (m.gate_exps.row_bytes, m.up_exps.row_bytes)
        };
        let (zq, zd) = e.quantize_q8_1(z, t, n_embd)?;
        let gate = matvec(
            0,
            &pair_tok_d,
            &zq,
            &zd,
            n_embd,
            n_ff_exp,
            m.gate_exps.qtype,
            gate_row_bytes,
        )?;
        let up = matvec(
            1,
            &pair_tok_d,
            &zq,
            &zd,
            n_embd,
            n_ff_exp,
            m.up_exps.qtype,
            up_row_bytes,
        )?;
        let mut act = e.uninit(n_pairs * n_ff_exp)?;
        Self::ffn_act_lim(
            e,
            cfg,
            &gate,
            &up,
            1.0,
            1.0,
            cfg.clamp_exp_at(il as u32),
            &mut act,
            n_pairs * n_ff_exp,
        )?;
        let (aq2, ad2) = e.quantize_q8_1(&act, n_pairs, n_ff_exp)?;
        let pair_self: Vec<i32> = (0..n_pairs as i32).collect();
        let pair_self_d = e.htod_i32(&pair_self)?;
        let down = matvec(
            2,
            &pair_self_d,
            &aq2,
            &ad2,
            n_ff_exp,
            n_embd,
            m.down_exps.qtype,
            m.down_exps.row_bytes,
        )?;
        let mut moe_out = e.uninit(t * n_embd)?;
        e.moe_pairs_scatter(
            &down,
            &pair_w_d,
            &tok_off_d,
            &tok_ids_d,
            &mut moe_out,
            t,
            n_embd,
        )?;

        if std::env::var("MEMRA_MOE_STATS").is_ok() {
            let mut sizes: Vec<usize> = by_expert
                .iter()
                .filter_map(|pairs| (!pairs.is_empty()).then_some(pairs.len()))
                .collect();
            sizes.sort_unstable();
            let mean = sizes.iter().sum::<usize>() as f64 / sizes.len().max(1) as f64;
            println!(
                "moe-grouped il={il} t={t} dispatch=resident-q8-clamped-pairs active={}/{} \
                 m_e: min={} median={} mean={mean:.1} max={}",
                sizes.len(),
                n_expert,
                sizes.first().copied().unwrap_or(0),
                sizes.get(sizes.len() / 2).copied().unwrap_or(0),
                sizes.last().copied().unwrap_or(0),
            );
        }
        Ok(moe_out)
    }

    fn moe_ffn_grouped_add_shared(
        e: &Engine,
        m: &MoeWeights,
        z: &CudaSlice<f32>,
        t: usize,
        cfg: &ModelConfig,
        il: u16,
        moe_out: &mut CudaSlice<f32>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let n_embd = cfg.n_embd as usize;
        if let (Some(gate_shexp), Some(up_shexp), Some(down_shexp)) =
            (&m.gate_shexp, &m.up_shexp, &m.down_shexp)
        {
            let n_ff_sh = gate_shexp.out_features();
            let sg_gate = e.matmul(gate_shexp, z, t)?;
            let sg_up = e.matmul(up_shexp, z, t)?;
            let mut sa = e.uninit(t * n_ff_sh)?;
            Self::ffn_act_lim(
                e,
                cfg,
                &sg_gate,
                &sg_up,
                1.0,
                1.0,
                cfg.clamp_shexp_at(il as u32),
                &mut sa,
                t * n_ff_sh,
            )?;
            let sh = e.matmul(down_shexp, &sa, t)?;
            let gate = match &m.gate_inp_shexp {
                Some(gate_inp_shexp) => {
                    if t < PRIME_MIN_T || crate::router_prefill_exact_on() {
                        e.sigmoid_dot_rows(z, gate_inp_shexp.float_data(), n_embd, t)?
                    } else {
                        let raw = e.linear(z, gate_inp_shexp.float_data(), t, n_embd, 1)?;
                        let mut gate = e.uninit(t)?;
                        e.sigmoid(&raw, &mut gate, t)?;
                        gate
                    }
                }
                None => e.htod(&vec![1.0f32; t])?,
            };
            e.add_scaled_rows(&sh, &gate, moe_out, n_embd, t)?;
        }
        Ok(())
    }

    /// A2 expert-grouped MoE FFN (prefill path, MEMRA_MOE_GROUPED=1). Same semantics as moe_ffn:
    /// z [T, n_embd] -> moe_out [T, n_embd]. BIT-IDENTICAL to moe_ffn when using the slot scheme.
    pub(crate) fn moe_ffn_grouped(e: &Engine, m: &MoeWeights, z: &CudaSlice<f32>, t: usize,
                                  cfg: &ModelConfig, il: u16, max_block: usize)
                   -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let moe = cfg.moe.as_ref().unwrap();
        let n_embd = cfg.n_embd as usize;
        let n_expert = moe.expert_count as usize;
        let n_used = moe.expert_used_count as usize;
        let n_ff_exp = moe.expert_ff_length as usize;
        // step35 per-layer SwiGLU clamp; None on every other arch / unclamped layer.
        let lim_exp = cfg.clamp_exp_at(il as u32);

        // 1. ROUTER: exactly the same m-invariant selector and host sigmoid oracle as the
        // sequential path. The grouped dispatch never enters the softmax-only pairs/dev router.
        let logits = Self::moe_router_logits(e, m, z, t, cfg)?;
        let (sel_all, w_all) = if let Some(sig) = cfg.sigmoid_router() {
            Self::moe_route_cfg(e, &logits, t, n_expert, n_used,
                                m.exp_probs_b.as_deref(), Some(sig), m.active_experts.as_deref())?
        } else {
            Self::moe_route_cfg(e, &logits, t, n_expert, n_used,
                                None, None, m.active_experts.as_deref())?
        };
        Self::trace_moe_routes(il, t, &sel_all, &w_all)?;
        Self::trace_moe_input(e, il, t, n_embd, z)?;

        // Fits-VRAM Step35 under the PP stage split: use the expert-major q8 arithmetic directly
        // from the owning device's uniform resident slab. This is not `moe_ffn_pairs`: routing
        // already happened above and the activation is clamp-aware. Mixed layouts, remote slabs,
        // macro-scaled experts, q8-disabled configs, and spill fall through to metadata-aware A2.
        let no_exp_macros = m.gate_exps.macros.is_none()
            && m.up_exps.macros.is_none()
            && m.down_exps.macros.is_none();
        let resident_q8 = m.dev_exps.as_ref().filter(|dev| {
            m.has_uniform_expert_layout()
                && no_exp_macros
                && moe_q8_enabled()
                && q8_expert_supported(m.gate_exps.qtype)
                && q8_expert_supported(m.up_exps.qtype)
                && q8_expert_supported(m.down_exps.qtype)
                && moe_slab_enabled()
                && dev.dev == e.ctx().ordinal()
        });
        if let Some(dev) = resident_q8 {
            let mut moe_out = Self::moe_ffn_grouped_resident_q8(
                e,
                m,
                z,
                t,
                cfg,
                il,
                &sel_all,
                &w_all,
                &dev.ptr_row,
                dev.gu_il,
            )?;
            Self::moe_ffn_grouped_add_shared(e, m, z, t, cfg, il, &mut moe_out)?;
            return Ok(moe_out);
        }

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
        let g_len = m.gate_exps.max_expert_bytes();
        let u_len = m.up_exps.max_expert_bytes();
        let d_len = m.down_exps.max_expert_bytes();
        let moe_q8 = m.has_uniform_expert_layout()
            && moe_q8_enabled()
            && q8_expert_supported(m.gate_exps.qtype)
            && q8_expert_supported(m.up_exps.qtype)
            && q8_expert_supported(m.down_exps.qtype);
        // The per-expert slab view is legal only for the ordinary contiguous uniform layout.
        // Interleaved GU slabs require the pointer-table fast path above.
        let slab_local = m.dev_exps.as_ref().filter(|dev| {
            !dev.gu_il && moe_slab_enabled() && dev.dev == e.ctx().ordinal()
        });
        let use_cache =
            slab_local.is_none() && Engine::moe_cache_enabled() && !e.moe_cache_frozen();
        // The sequential no-cache/frozen staging oracle is f32. Use q8 only where sequential
        // also does: a local resident slab or a live SLRU dispatch.
        let grouped_q8 = moe_q8 && (slab_local.is_some() || use_cache);

        // GPU scratch for staging (only allocated without a local slab or cache).
        let (mut scratch_g, mut scratch_u, mut scratch_d) = if slab_local.is_none() && !use_cache {
            (Some(e.alloc_u8(g_len)?), Some(e.alloc_u8(u_len)?), Some(e.alloc_u8(d_len)?))
        } else {
            (None, None, None)
        };

        // 4. PER ACTIVE EXPERT: gather, compute, scatter.
        // Processing ORDER: DESCENDING m_e (biggest token batches first) — the concluded winner
        // (rig5090 2026-07-04, the ascending-id arm and its MEMRA_MOE_ORDER seam removed): desc is
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
        let page_window = moe_page_prefetch_window();
        let worker_disk_prefetch = use_cache && crate::spill_pread::worker_enabled();
        if worker_disk_prefetch {
            if let Some(first) = grouped_worker_prefetch_position(order.len(), None) {
                Self::moe_prefetch_disk_expert(e, il, order[first], m, max_block, &[])?;
            }
        }
        for (order_pos, &ex) in order.iter().enumerate() {
            for next in page_prefetch_positions(order_pos, order.len(), page_window) {
                Self::moe_prefetch_host_expert(order[next], m);
            }
            if worker_disk_prefetch {
                if let Some(next) = grouped_worker_prefetch_position(order.len(), Some(order_pos)) {
                    use crate::moe_cache::{BlockId, PROJ_DOWN, PROJ_GATE, PROJ_UP};
                    let keep = [
                        BlockId::new(il, PROJ_GATE, ex as u16),
                        BlockId::new(il, PROJ_UP, ex as u16),
                        BlockId::new(il, PROJ_DOWN, ex as u16),
                    ];
                    Self::moe_prefetch_disk_expert(e, il, order[next], m, max_block, &keep)?;
                }
            }
            let grp = &groups[ex];
            let m_e = grp.tok_indices.len();
            m_dist.push(m_e);
            let gl = m.gate_exps.expert_layout(ex);
            let ul = m.up_exps.expert_layout(ex);
            let dl = m.down_exps.expert_layout(ex);

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

            // Compute gate/up/down from the local slab, metadata-aware cache, or staging. The q8
            // form keeps each gathered row in the sequential arithmetic class at `m=m_e`.
            let y = if let Some(dev) = slab_local {
                let gate_start = ex * m.gate_exps.expert_stride;
                let up_start = ex * m.up_exps.expert_stride;
                let down_start = ex * m.down_exps.expert_stride;
                if grouped_q8 {
                    let (zq, zd) = e.quantize_q8_1(&gathered, m_e, n_embd)?;
                    let gate = e.qmatvec_expert_q8(
                        &dev.gate,
                        gate_start..gate_start + gl.len,
                        &zq,
                        &zd,
                        m_e,
                        m.gate_exps.in_f,
                        m.gate_exps.out_f,
                        gl.qtype,
                        gl.row_bytes,
                    )?;
                    let up = e.qmatvec_expert_q8(
                        &dev.up,
                        up_start..up_start + ul.len,
                        &zq,
                        &zd,
                        m_e,
                        m.up_exps.in_f,
                        m.up_exps.out_f,
                        ul.qtype,
                        ul.row_bytes,
                    )?;
                    let mut act = e.uninit(m_e * n_ff_exp)?;
                    Self::ffn_act_lim(
                        e,
                        cfg,
                        &gate,
                        &up,
                        m.gate_exps.macro_scale(ex),
                        m.up_exps.macro_scale(ex),
                        lim_exp,
                        &mut act,
                        m_e * n_ff_exp,
                    )?;
                    let (aq2, ad2) = e.quantize_q8_1(&act, m_e, n_ff_exp)?;
                    e.qmatvec_expert_q8(
                        &dev.down,
                        down_start..down_start + dl.len,
                        &aq2,
                        &ad2,
                        m_e,
                        m.down_exps.in_f,
                        m.down_exps.out_f,
                        dl.qtype,
                        dl.row_bytes,
                    )?
                } else {
                    let gate = e.qmatvec_view(
                        &dev.gate,
                        gate_start..gate_start + gl.len,
                        &gv,
                        m_e,
                        m.gate_exps.in_f,
                        m.gate_exps.out_f,
                        gl.qtype,
                        gl.row_bytes,
                    )?;
                    let up = e.qmatvec_view(
                        &dev.up,
                        up_start..up_start + ul.len,
                        &gv,
                        m_e,
                        m.up_exps.in_f,
                        m.up_exps.out_f,
                        ul.qtype,
                        ul.row_bytes,
                    )?;
                    let mut act = e.uninit(m_e * n_ff_exp)?;
                    Self::ffn_act_lim(
                        e,
                        cfg,
                        &gate,
                        &up,
                        m.gate_exps.macro_scale(ex),
                        m.up_exps.macro_scale(ex),
                        lim_exp,
                        &mut act,
                        m_e * n_ff_exp,
                    )?;
                    let actv = act.slice(0..m_e * n_ff_exp);
                    e.qmatvec_view(
                        &dev.down,
                        down_start..down_start + dl.len,
                        &actv,
                        m_e,
                        m.down_exps.in_f,
                        m.down_exps.out_f,
                        dl.qtype,
                        dl.row_bytes,
                    )?
                }
            } else if use_cache {
                use crate::moe_cache::{BlockId, PROJ_DOWN, PROJ_GATE, PROJ_UP};
                if grouped_q8 {
                    let (zq, zd) = e.quantize_q8_1(&gathered, m_e, n_embd)?;
                    let gate = e.with_moe_cache(max_block, |cache, eng| {
                        let id = BlockId::new(il, PROJ_GATE, ex as u16);
                        let slot =
                            cache.dispatch_source(id, m.gate_exps.expert_source(ex), eng)?;
                        eng.qmatvec_expert_q8(
                            cache.buf(slot),
                            0..gl.len,
                            &zq,
                            &zd,
                            m_e,
                            m.gate_exps.in_f,
                            m.gate_exps.out_f,
                            gl.qtype,
                            gl.row_bytes,
                        )
                    })?;
                    let up = e.with_moe_cache(max_block, |cache, eng| {
                        let id = BlockId::new(il, PROJ_UP, ex as u16);
                        let slot =
                            cache.dispatch_source(id, m.up_exps.expert_source(ex), eng)?;
                        eng.qmatvec_expert_q8(
                            cache.buf(slot),
                            0..ul.len,
                            &zq,
                            &zd,
                            m_e,
                            m.up_exps.in_f,
                            m.up_exps.out_f,
                            ul.qtype,
                            ul.row_bytes,
                        )
                    })?;
                    let mut act = e.uninit(m_e * n_ff_exp)?;
                    Self::ffn_act_lim(
                        e,
                        cfg,
                        &gate,
                        &up,
                        m.gate_exps.macro_scale(ex),
                        m.up_exps.macro_scale(ex),
                        lim_exp,
                        &mut act,
                        m_e * n_ff_exp,
                    )?;
                    let (aq2, ad2) = e.quantize_q8_1(&act, m_e, n_ff_exp)?;
                    e.with_moe_cache(max_block, |cache, eng| {
                        let id = BlockId::new(il, PROJ_DOWN, ex as u16);
                        let slot =
                            cache.dispatch_source(id, m.down_exps.expert_source(ex), eng)?;
                        eng.qmatvec_expert_q8(
                            cache.buf(slot),
                            0..dl.len,
                            &aq2,
                            &ad2,
                            m_e,
                            m.down_exps.in_f,
                            m.down_exps.out_f,
                            dl.qtype,
                            dl.row_bytes,
                        )
                    })?
                } else {
                    let gate = e.with_moe_cache(max_block, |cache, eng| {
                        let id = BlockId::new(il, PROJ_GATE, ex as u16);
                        let slot =
                            cache.dispatch_source(id, m.gate_exps.expert_source(ex), eng)?;
                        eng.qmatvec_view(
                            cache.buf(slot),
                            0..gl.len,
                            &gv,
                            m_e,
                            m.gate_exps.in_f,
                            m.gate_exps.out_f,
                            gl.qtype,
                            gl.row_bytes,
                        )
                    })?;
                    let up = e.with_moe_cache(max_block, |cache, eng| {
                        let id = BlockId::new(il, PROJ_UP, ex as u16);
                        let slot =
                            cache.dispatch_source(id, m.up_exps.expert_source(ex), eng)?;
                        eng.qmatvec_view(
                            cache.buf(slot),
                            0..ul.len,
                            &gv,
                            m_e,
                            m.up_exps.in_f,
                            m.up_exps.out_f,
                            ul.qtype,
                            ul.row_bytes,
                        )
                    })?;
                    let mut act = e.uninit(m_e * n_ff_exp)?;
                    Self::ffn_act_lim(
                        e,
                        cfg,
                        &gate,
                        &up,
                        m.gate_exps.macro_scale(ex),
                        m.up_exps.macro_scale(ex),
                        lim_exp,
                        &mut act,
                        m_e * n_ff_exp,
                    )?;
                    let actv = act.slice(0..m_e * n_ff_exp);
                    e.with_moe_cache(max_block, |cache, eng| {
                        let id = BlockId::new(il, PROJ_DOWN, ex as u16);
                        let slot =
                            cache.dispatch_source(id, m.down_exps.expert_source(ex), eng)?;
                        eng.qmatvec_view(
                            cache.buf(slot),
                            0..dl.len,
                            &actv,
                            m_e,
                            m.down_exps.in_f,
                            m.down_exps.out_f,
                            dl.qtype,
                            dl.row_bytes,
                        )
                    })?
                }
            } else {
                let sg = scratch_g.as_mut().unwrap();
                let su = scratch_u.as_mut().unwrap();
                let sd = scratch_d.as_mut().unwrap();
                e.stage_expert(m.gate_exps.expert_bytes(ex), sg, 0)?;
                e.stage_expert(m.up_exps.expert_bytes(ex), su, 0)?;
                e.stage_expert(m.down_exps.expert_bytes(ex), sd, 0)?;
                if grouped_q8 {
                    let (zq, zd) = e.quantize_q8_1(&gathered, m_e, n_embd)?;
                    let gate = e.qmatvec_expert_q8(
                        sg,
                        0..gl.len,
                        &zq,
                        &zd,
                        m_e,
                        m.gate_exps.in_f,
                        m.gate_exps.out_f,
                        gl.qtype,
                        gl.row_bytes,
                    )?;
                    let up = e.qmatvec_expert_q8(
                        su,
                        0..ul.len,
                        &zq,
                        &zd,
                        m_e,
                        m.up_exps.in_f,
                        m.up_exps.out_f,
                        ul.qtype,
                        ul.row_bytes,
                    )?;
                    let mut act = e.uninit(m_e * n_ff_exp)?;
                    Self::ffn_act_lim(
                        e,
                        cfg,
                        &gate,
                        &up,
                        m.gate_exps.macro_scale(ex),
                        m.up_exps.macro_scale(ex),
                        lim_exp,
                        &mut act,
                        m_e * n_ff_exp,
                    )?;
                    let (aq2, ad2) = e.quantize_q8_1(&act, m_e, n_ff_exp)?;
                    e.qmatvec_expert_q8(
                        sd,
                        0..dl.len,
                        &aq2,
                        &ad2,
                        m_e,
                        m.down_exps.in_f,
                        m.down_exps.out_f,
                        dl.qtype,
                        dl.row_bytes,
                    )?
                } else {
                    let gate = e.qmatvec_view(
                        sg,
                        0..gl.len,
                        &gv,
                        m_e,
                        m.gate_exps.in_f,
                        m.gate_exps.out_f,
                        gl.qtype,
                        gl.row_bytes,
                    )?;
                    let up = e.qmatvec_view(
                        su,
                        0..ul.len,
                        &gv,
                        m_e,
                        m.up_exps.in_f,
                        m.up_exps.out_f,
                        ul.qtype,
                        ul.row_bytes,
                    )?;
                    let mut act = e.uninit(m_e * n_ff_exp)?;
                    Self::ffn_act_lim(
                        e,
                        cfg,
                        &gate,
                        &up,
                        m.gate_exps.macro_scale(ex),
                        m.up_exps.macro_scale(ex),
                        lim_exp,
                        &mut act,
                        m_e * n_ff_exp,
                    )?;
                    let actv = act.slice(0..m_e * n_ff_exp);
                    e.qmatvec_view(
                        sd,
                        0..dl.len,
                        &actv,
                        m_e,
                        m.down_exps.in_f,
                        m.down_exps.out_f,
                        dl.qtype,
                        dl.row_bytes,
                    )?
                }
            };

            // SCATTER into slot buffer: each row goes to slot_buf[tok, slot, :].
            e.scatter_slot(&y, &tok_idx_d, &slot_idx_d, &weight_d,
                           &mut slot_buf, &mut wbuf, n_embd, n_used, m_e)?;
        }

        // 5. REDUCE SLOTS: sum the 8 slots per token into the final moe_out.
        let mut moe_out = e.zeros(t * n_embd)?;
        e.reduce_slots(&slot_buf, &wbuf, &mut moe_out, n_embd, n_used, t)?;

        // STATS: print m-distribution when MEMRA_MOE_STATS is set.
        if std::env::var("MEMRA_MOE_STATS").is_ok() && !m_dist.is_empty() {
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

        Self::moe_ffn_grouped_add_shared(e, m, z, t, cfg, il, &mut moe_out)?;
        Ok(moe_out)
    }

    /// Lane-3 M2: cross-stream MoE for lockstep decode. Routes all m stream rows in one
    /// batch, executes fully-HBM-resident experts through the grouped gather/GEMM/scatter
    /// machinery at m_e>1 (weight reads amortized across streams), and assigns any expert
    /// with a missing projection to that row's CPU companion call (whole-expert granularity,
    /// same rule as the sequential frozen path). Slot-pinned accumulation keeps each row's
    /// expert-sum order identical to the sequential path.
    pub(crate) fn moe_ffn_lockstep(
        &self,
        e: &Engine,
        m: &MoeWeights,
        zbatch: &CudaSlice<f32>,
        mrows: usize,
        il: u16,
        max_block: usize,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        use crate::moe_cache::{BlockId, PROJ_DOWN, PROJ_GATE, PROJ_UP};
        let cfg = &self.cfg;
        let moe = cfg.moe.as_ref().unwrap();
        let n_embd = cfg.n_embd as usize;
        let n_expert = moe.expert_count as usize;
        let n_used = moe.expert_used_count as usize;
        let n_ff_exp = moe.expert_ff_length as usize;
        // step35 per-layer SwiGLU clamp; None on every other arch / unclamped layer.
        let lim_exp = cfg.clamp_exp_at(il as u32);
        let lim_shexp = cfg.clamp_shexp_at(il as u32);

        let logits = e.matmul(&m.gate_inp, zbatch, mrows)?;
        let (sel_all, w_all) = if let Some(sig) = cfg.sigmoid_router() {
            Self::moe_route_cfg(e, &logits, mrows, n_expert, n_used,
                                m.exp_probs_b.as_deref(), Some(sig), m.active_experts.as_deref())?
        } else {
            Self::moe_route_cfg(e, &logits, mrows, n_expert, n_used,
                                None, None, m.active_experts.as_deref())?
        };
        Self::trace_moe_routes(il, mrows, &sel_all, &w_all)?;

        // Residency split at whole-expert granularity against the (frozen) cache.
        let resident_expert: Vec<bool> = e.with_moe_cache(max_block, |c, _| {
            Ok((0..n_expert)
                .map(|ex| {
                    [PROJ_GATE, PROJ_UP, PROJ_DOWN].into_iter().all(|p| {
                        c.resident(BlockId::new(il, p, ex as u16)).is_some()
                    })
                })
                .collect())
        })?;

        struct Group {
            rows: Vec<i32>,
            slots: Vec<i32>,
            weights: Vec<f32>,
        }
        let mut groups: std::collections::HashMap<usize, Group> = Default::default();
        let mut cpu_rows: Vec<Vec<(usize, f32)>> = vec![Vec::new(); mrows];
        let mut cpu_by_expert: std::collections::HashMap<usize, Vec<(usize, f32)>> =
            Default::default();
        for row in 0..mrows {
            for j in 0..n_used {
                let ex = sel_all[row * n_used + j] as usize;
                let w = w_all[row * n_used + j];
                if resident_expert[ex] {
                    let group = groups.entry(ex).or_insert_with(|| Group {
                        rows: Vec::new(),
                        slots: Vec::new(),
                        weights: Vec::new(),
                    });
                    group.rows.push(row as i32);
                    group.slots.push(j as i32);
                    group.weights.push(w);
                } else {
                    crate::cpu_experts::record_incomplete_gpu_residency(0);
                    cpu_rows[row].push((ex, w));
                    cpu_by_expert.entry(ex).or_default().push((row, w));
                }
            }
        }

        // CPU tickets first: reads/compute overlap the GPU grouped work below. Experts routed
        // by >=2 streams go through the multi-row ABI (weight decode amortized across rows);
        // each row's remaining experts stay one ordinary per-row call. Contribution FP-sum
        // order per row differs from the sequential single-call chunk — part of the
        // documented lockstep numeric class.
        let host_rows = e.dtoh(zbatch)?;
        let rows_ok = crate::cpu_experts::rows_supported();
        enum CpuPart {
            Single { row: usize },
            Rows { rows: Vec<usize> },
        }
        let mut tickets: Vec<(CpuPart, crate::cpu_experts::CpuExpertTicket)> = Vec::new();
        let mut rows_served: std::collections::HashSet<(usize, usize)> = Default::default();
        if rows_ok {
            let mut shared: Vec<(usize, Vec<(usize, f32)>)> = cpu_by_expert
                .into_iter()
                .filter(|(_, rows)| rows.len() >= 2)
                .collect();
            shared.sort_by_key(|(ex, _)| *ex);
            for (ex, mut row_weights) in shared {
                row_weights.sort_by_key(|(row, _)| *row);
                let inputs: Vec<(&[f32], f32)> = row_weights
                    .iter()
                    .map(|&(row, w)| (&host_rows[row * n_embd..(row + 1) * n_embd], w))
                    .collect();
                let job = crate::cpu_experts::prepare_rows_job(m, ex, &inputs)
                    .map_err(std::io::Error::other)?;
                for &(row, _) in &row_weights {
                    rows_served.insert((row, ex));
                }
                tickets.push((
                    CpuPart::Rows {
                        rows: row_weights.iter().map(|&(row, _)| row).collect(),
                    },
                    crate::cpu_experts::submit_rows(job).map_err(std::io::Error::other)?,
                ));
            }
        }
        for (row, selected) in cpu_rows.iter().enumerate() {
            let leftover: Vec<(usize, f32)> = selected
                .iter()
                .copied()
                .filter(|&(ex, _)| !rows_served.contains(&(row, ex)))
                .collect();
            if leftover.is_empty() {
                continue;
            }
            let host_row = &host_rows[row * n_embd..(row + 1) * n_embd];
            let job = crate::cpu_experts::prepare_job(m, il, &leftover, host_row)
                .map_err(std::io::Error::other)?;
            tickets.push((
                CpuPart::Single { row },
                crate::cpu_experts::submit(job).map_err(std::io::Error::other)?,
            ));
        }

        let mut slot_buf = e.zeros(mrows * n_used * n_embd)?;
        let mut wbuf = e.zeros(mrows * n_used)?;
        let mut order: Vec<usize> = groups.keys().copied().collect();
        order.sort_by(|&a, &b| {
            groups[&b].rows.len().cmp(&groups[&a].rows.len()).then(a.cmp(&b))
        });
        for &ex in &order {
            let group = &groups[&ex];
            let m_e = group.rows.len();
            let gl = m.gate_exps.expert_layout(ex);
            let ul = m.up_exps.expert_layout(ex);
            let dl = m.down_exps.expert_layout(ex);
            let row_idx_d = e.htod_i32(&group.rows)?;
            let slot_idx_d = e.htod_i32(&group.slots)?;
            let dmac = m.down_exps.macro_scale(ex);
            let weight_d = if dmac == 1.0 {
                e.htod(&group.weights)?
            } else {
                let scaled: Vec<f32> = group.weights.iter().map(|&w| w * dmac).collect();
                e.htod(&scaled)?
            };
            let mut gathered = e.zeros(m_e * n_embd)?;
            e.gather_rows(zbatch, &row_idx_d, &mut gathered, n_embd, m_e)?;
            let gv = gathered.slice(0..m_e * n_embd);
            let gate = e.with_moe_cache(max_block, |c, eng| {
                let slot = c
                    .resident(BlockId::new(il, PROJ_GATE, ex as u16))
                    .ok_or("lockstep resident expert vanished (cache not frozen?)")?;
                eng.qmatvec_view(c.buf(crate::moe_cache::DispatchSlot::Resident(slot)), 0..gl.len, &gv, m_e,
                    m.gate_exps.in_f, m.gate_exps.out_f, gl.qtype, gl.row_bytes)
            })?;
            let up = e.with_moe_cache(max_block, |c, eng| {
                let slot = c
                    .resident(BlockId::new(il, PROJ_UP, ex as u16))
                    .ok_or("lockstep resident expert vanished (cache not frozen?)")?;
                eng.qmatvec_view(c.buf(crate::moe_cache::DispatchSlot::Resident(slot)), 0..ul.len, &gv, m_e,
                    m.up_exps.in_f, m.up_exps.out_f, ul.qtype, ul.row_bytes)
            })?;
            let mut act = e.zeros(m_e * n_ff_exp)?;
            Self::ffn_act_lim(e, cfg, &gate, &up, m.gate_exps.macro_scale(ex),
                m.up_exps.macro_scale(ex), lim_exp, &mut act, m_e * n_ff_exp)?;
            let actv = act.slice(0..m_e * n_ff_exp);
            let y = e.with_moe_cache(max_block, |c, eng| {
                let slot = c
                    .resident(BlockId::new(il, PROJ_DOWN, ex as u16))
                    .ok_or("lockstep resident expert vanished (cache not frozen?)")?;
                eng.qmatvec_view(c.buf(crate::moe_cache::DispatchSlot::Resident(slot)), 0..dl.len, &actv, m_e,
                    m.down_exps.in_f, m.down_exps.out_f, dl.qtype, dl.row_bytes)
            })?;
            e.scatter_slot(&y, &row_idx_d, &slot_idx_d, &weight_d,
                           &mut slot_buf, &mut wbuf, n_embd, n_used, m_e)?;
        }
        let mut moe_out = e.zeros(mrows * n_embd)?;
        e.reduce_slots(&slot_buf, &wbuf, &mut moe_out, n_embd, n_used, mrows)?;

        // CPU contributions join BEFORE the shared expert (the sequential path's placement).
        let mut row_sums: Vec<Option<Vec<f32>>> = vec![None; mrows];
        for (part, ticket) in tickets {
            let cpu_output = ticket.wait().map_err(std::io::Error::other)?;
            let mut add_row = |row: usize, chunk: &[f32]| {
                let sum = row_sums[row].get_or_insert_with(|| vec![0.0f32; n_embd]);
                for (accumulator, value) in sum.iter_mut().zip(chunk) {
                    *accumulator += value;
                }
            };
            match part {
                CpuPart::Single { row } => add_row(row, &cpu_output),
                CpuPart::Rows { rows } => {
                    for (slot, row) in rows.into_iter().enumerate() {
                        add_row(row, &cpu_output[slot * n_embd..(slot + 1) * n_embd]);
                    }
                }
            }
        }
        for (row, sum) in row_sums.into_iter().enumerate() {
            let Some(sum) = sum else { continue };
            let cpu_output = e.htod(&sum)?;
            let mut dst = moe_out.slice_mut(row * n_embd..(row + 1) * n_embd);
            e.axpy_into(&cpu_output, 1.0, &mut dst, n_embd)?;
        }

        if let (Some(gate_shexp), Some(up_shexp), Some(down_shexp)) =
            (&m.gate_shexp, &m.up_shexp, &m.down_shexp)
        {
            let n_ff_sh = gate_shexp.out_features();
            let sg_gate = e.matmul(gate_shexp, zbatch, mrows)?;
            let sg_up = e.matmul(up_shexp, zbatch, mrows)?;
            let mut sa = e.zeros(mrows * n_ff_sh)?;
            Self::ffn_act_lim(e, cfg, &sg_gate, &sg_up, 1.0, 1.0, lim_shexp,
                              &mut sa, mrows * n_ff_sh)?;
            let sh = e.matmul(down_shexp, &sa, mrows)?;
            // lockstep rows ARE decode tokens: fused sigmoid-dot per row so batched serving
            // decode matches the single-sequence decode chain bit-for-bit.
            let g = match &m.gate_inp_shexp {
                Some(gate_inp_shexp) => {
                    e.sigmoid_dot_rows(zbatch, gate_inp_shexp.float_data(), n_embd, mrows)?
                }
                None => e.htod(&vec![1.0f32; mrows])?,
            };
            e.add_scaled_rows(&sh, &g, &mut moe_out, n_embd, mrows)?;
        }

        Ok(moe_out)
    }
}

// ============================ gemma4 (R8 verified wiring) ==================================
// Node-for-node vs llama.cpp src/models/gemma4.cpp:180-405 (HANDOVER "R8 VERIFIED WIRING").
// v0 bring-up: full attention everywhere (exact for prompts < sliding_window 1024 — R6 masking
// later), sdpa_naive (hd 512 has no FA stamp), sequential host-staged MoE (the perf arms grow
// gemma variants after the correctness gate).
impl HybridModel {
    /// Per-layer attention geometry (R5): (head_dim, n_kv, n_head, rope_base, scale, is_swa).
    pub(crate) fn gemma4_geom(&self, il: usize) -> (usize, usize, usize, f32, f32, bool) {
        let g = self.cfg.gemma4.as_ref().unwrap();
        let swa = g.swa_pattern[il];
        let hd = if swa { g.key_length_swa } else { g.key_length_global } as usize;
        // attention scale = 1.0 (llama gemma4.cpp:11 "Gemma4 uses self.scaling = 1.0" — q/k are
        // per-head rms-normed; NOT the 1/sqrt(hd) default. Bring-up bug: 1/sqrt(hd) left token-0
        // rows exact (softmax over one element) while every later position drifted).
        (hd, g.head_count_kv[il] as usize, self.cfg.n_head as usize,
         if swa { g.rope_base_swa } else { g.rope_base_global },
         1.0, swa)
    }

    /// Suppress-token mask over t logits rows (tokenizer.ggml.suppress_tokens; no-op when the
    /// model ships none). NOT monotonic like softcap — must run before every argmax/sample, so
    /// every gemma4 logits tail (forward/prime/decode/dc/verify/e4b) calls this on device ld.
    fn gemma4_suppress(&self, e: &Engine, ld: &mut CudaSlice<f32>, t: usize)
                       -> Result<(), Box<dyn std::error::Error>> {
        if let Some((ids, n)) = self.gemma4_aux.as_ref().and_then(|a| a.suppress_d.as_ref()) {
            e.mask_ids_rows(ld, ids, *n, self.output.out_features(), t)?;
        }
        Ok(())
    }

    /// gemma4 attention (R5 geometry, R7 weightless V-norm on the RAW K projection, R9 dual rope).
    /// `cache`: Some => PRIME mode — append the T post-rope K / normed V rows into the quantized
    /// KV cache (same per-row quantize math as the decode append) and advance len. Fresh-prompt
    /// only (v0): attends within `tokens` via the f32 sdpa.
    fn gemma4_attn_prime(&self, e: &Engine, fa: &crate::hybrid::FullAttnLayer, il: usize,
                         h: &CudaSlice<f32>, pos_d: &CudaSlice<i32>, t: usize,
                         cache: Option<&mut Cache>)
                         -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let (hd, nkv, nh, base, scale, swa) = self.gemma4_geom(il);
        let eps = self.cfg.rms_eps;
        let aux = self.gemma4_aux.as_ref().unwrap();

        // quantize-once window: q/k/v share `h` — the MMQ D4 activation quantizes once
        // (h stays borrowed across the triple, so the cache key can't go stale).
        e.mmq_act_begin();
        let q0 = e.matmul(&fa.wq, h, t)?;   // [t, nh*hd]
        let k0 = e.matmul(&fa.wk, h, t)?;   // [t, nkv*hd]
        // globals ship no v_proj (wv := wk at load) — V input is the SAME projection output;
        // reuse k0 instead of re-running the identical matmul (K=V dedup, 5 layers).
        let v0 = if swa { e.matmul(&fa.wv, h, t)? } else { e.clone_dtod(&k0)? };

        let mut q = e.uninit(t * nh * hd)?;
        let mut k = e.uninit(t * nkv * hd)?;
        // R7: V = weightless rms_norm of the raw projection; NEVER roped.
        let mut v = e.uninit(t * nkv * hd)?;
        // 31B glue lane: producers emit the bf16 FA operands (norm emits vb; rope emits qb/kb
        // post-rope) — kills 3 f32->bf16 converts + re-reads per layer. Bit-identical operands
        // (same __float2bfloat16); MEMRA_FA_EMIT=0 reverts to the convert-in-FA path.
        static EMIT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let emit = t >= 16 && crate::Engine::qkvnorm_w_on_prefill(nh * t + 2 * nkv * t, hd)
            && *EMIT.get_or_init(|| std::env::var("MEMRA_FA_EMIT").map(|s| s != "0").unwrap_or(true));
        let mut qb = e.alloc_uninit::<u8>(if emit { t * nh * hd * 2 } else { 1 })?;
        let mut kb = e.alloc_uninit::<u8>(if emit { t * nkv * hd * 2 } else { 1 })?;
        let mut vb = e.alloc_uninit::<u8>(if emit { t * nkv * hd * 2 } else { 1 })?;
        // f16-P/V door: emit V as f16 straight from the norm when this layer's FA consumer
        // reads f16 — kills the per-layer bf16->f16 re-encode (1.4GB/pass on 31B SWA).
        let v_f16 = emit && crate::fa_f16pv_on() && match hd {
            512 => true,
            256 => swa && crate::faw_hp_on() && nh % 2 == 0 && (nh / nkv) % 2 == 0,
            _ => false,
        };
        if emit {
            e.rms_norm_qkv_w4b(&q0, &k0, &v0, fa.q_norm.float_data(), fa.k_norm.float_data(),
                               &aux.ones, &mut q, &mut k, &mut v, &mut vb,
                               hd, nh * t, nkv * t, eps, v_f16)?;
        } else {
            e.rms_norm_qkv(&q0, &k0, &v0, fa.q_norm.float_data(), fa.k_norm.float_data(),
                           &aux.ones, &mut q, &mut k, &mut v, hd, nh * t, nkv * t, eps)?;
        }

        let ff = if swa { None } else {
            Some(aux.rope_freqs.as_ref().expect("gemma4 global rope needs rope_freqs.weight"))
        };
        if emit {
            e.rope_neox2_bf16e(&mut q, &mut k, &mut qb, &mut kb, pos_d, hd, hd, nh, nkv, t,
                               base, 1.0, ff)?;
        } else {
            e.rope_neox2(&mut q, &mut k, pos_d, hd, hd, nh, nkv, t, base, 1.0, ff)?;
        }

        if let Some(cache) = cache {
            let kvl = cache.kv[il].as_mut().unwrap();
            assert_eq!(kvl.len, 0, "gemma4 prime is fresh-prompt only (v0)");
            e.append_kv_quantized_rows(&k, &v, &mut kvl.k, &mut kvl.v, kvl.len, t,
                                       kvl.kv_dim_k, kvl.kv_dim_v, kvl.k_tok_bytes, kvl.v_tok_bytes, (!swa && crate::Engine::gkv_on()) || (swa && crate::Engine::wkv_on()))?;
            kvl.len += t;
        }
        let mut attn = e.zeros(t * nh * hd)?;
        // R6: SWA layers mask keys older than sliding_window once the prompt exceeds it
        // (windowed naive twin; fa windowed stamps later). Under the window, full attention
        // is exact — SWA rides fa_prefill (hd-256 stamp), the hd-512 globals stay naive.
        let win = self.cfg.gemma4.as_ref().unwrap().sliding_window as usize;
        if swa && t > win {
            if hd == 256 && std::env::var("MEMRA_NOFA").is_err() {
                if emit { e.fa_prefill_w_pre(&qb, &kb, &vb, &mut attn, hd, nh, nkv, t, t,
                                             scale, true, win, v_f16)?; }
                else { e.fa_prefill_w(&q, &k, &v, &mut attn, hd, nh, nkv, t, t, scale, true,
                                      win)?; }
            } else {
                e.sdpa_naive_w(&q, &k, &v, &mut attn, hd, nh, nkv, t, t, scale, true, win)?;
            }
        } else if hd == 256 && std::env::var("MEMRA_NOFA").is_err() {
            e.fa_prefill(&q, &k, &v, &mut attn, hd, nh, nkv, t, t, scale, true)?;
        } else if hd == 512 && std::env::var("MEMRA_NOFA").is_err() {
            if emit { e.fa_prefill_hd512_pre(&qb, &kb, &vb, &mut attn, hd, nh, nkv, t, t,
                                             scale, true, v_f16)?; }
            else { e.fa_prefill_hd512(&q, &k, &v, &mut attn, hd, nh, nkv, t, t, scale, true)?; }
        } else {
            e.sdpa_naive(&q, &k, &v, &mut attn, hd, nh, nkv, t, t, scale, true)?;
        }
        Ok(e.matmul(&fa.wo, &attn, t)?)
    }

    /// Back-compat wrapper (pure prefill, no cache).
    fn gemma4_attn(&self, e: &Engine, fa: &crate::hybrid::FullAttnLayer, il: usize,
                   h: &CudaSlice<f32>, pos_d: &CudaSlice<i32>, t: usize)
                   -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        self.gemma4_attn_prime(e, fa, il, h, pos_d, t, None)
    }

    /// gemma4 MoE with the expert input PRE-QUANTIZED (the q8z tail fusion). Caller guarantees
    /// the fast-arm conditions (resident dev slabs + dp4a qtypes) — decode t=1 and verify rows
    /// arms only, per-token kernel chains identical to the f32-input path (same quantize bytes:
    /// the q8z epilogue is quantize_q8_1 verbatim).
    fn gemma4_moe_q8(&self, e: &Engine, m: &crate::hybrid::MoeWeights,
                     bits: &crate::hybrid::Gemma4MoeBits,
                     mq: &(CudaSlice<i8>, CudaSlice<f32>),
                     router_in: &CudaSlice<f32>, t: usize)
                     -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let moe = cfg.moe.as_ref().unwrap();
        let n_embd = cfg.n_embd as usize;
        let n_expert = moe.expert_count as usize;
        let n_used = moe.expert_used_count as usize;
        let n_ff_exp = moe.expert_ff_length as usize;
        // router stays the two-launch pair: THREE fuse variants measured worse (serial-dot
        // -50%, 1024-thread warp-parallel -12% on topk sync overhead — jsonl 2026-07-11);
        // the pair's 12us is kernel time, not launch gaps.
        let logits = if crate::router_kernel_on() {
            e.router_gemv(m.gate_inp.float_data(), router_in, n_embd, n_expert, t)?
        } else {
            e.matmul(&m.gate_inp, router_in, t)?
        };
        let dev = m.dev_exps.as_ref().unwrap();
        let (sel_d, w_d) = e.moe_router_topk_scaled(&logits, t, n_expert, n_used,
                                                    &bits.per_expert_scale_d)?;
        let (zq, zd) = mq;
        if t == 1 {
            let selv = sel_d.slice(0..n_used);
            let wv = w_d.slice(0..n_used);
            let act = e.moe_gate_up_gelu8_dev_q8(&dev.ptr_row, &selv, zq, zd,
                                                 n_embd, n_ff_exp, n_used, n_expert,
                                                 m.gate_exps.qtype, m.up_exps.qtype,
                                                 m.gate_exps.row_bytes, m.up_exps.row_bytes)?;
            let (aq2, ad2) = e.quantize_q8_1(&act, n_used, n_ff_exp)?;
            let mut moe_out = e.uninit(n_embd)?;
            e.moe_down8_fma_dev_q8(&dev.ptr_row, &selv, &wv, &aq2, &ad2,
                                   &mut moe_out.slice_mut(0..n_embd), n_ff_exp, n_embd,
                                   n_used, n_expert, m.down_exps.qtype, m.down_exps.row_bytes)?;
            return Ok(moe_out);
        }
        let csr = t <= 10 && std::env::var("MEMRA_GEMMA_CSR").as_deref() != Ok("0");
        let act = if csr {
            e.moe_gate_up_gelu8_dev_q8_csr(&dev.ptr_row, &sel_d, zq, zd, t * n_used,
                                           n_embd, n_ff_exp, n_used, n_expert,
                                           m.gate_exps.qtype, m.up_exps.qtype,
                                           m.gate_exps.row_bytes, m.up_exps.row_bytes)?
        } else {
            e.moe_gate_up_gelu8_dev_q8_rows(&dev.ptr_row, &sel_d, zq, zd, t,
                                            n_embd, n_ff_exp, n_used, n_expert,
                                            m.gate_exps.qtype, m.up_exps.qtype,
                                            m.gate_exps.row_bytes, m.up_exps.row_bytes)?
        };
        let (aq2, ad2) = e.quantize_q8_1(&act, t * n_used, n_ff_exp)?;
        let mut moe_out = e.uninit(t * n_embd)?;
        // down stays rows_g: the CSR dedup twin measured NEGATIVE at nsb=22 too (189.4 vs
        // 207.0 depth spec, bitwise-exact — jsonl 2026-07-10; qwen nsb=16 same verdict).
        e.moe_down8_fma_dev_q8_rows_g(&dev.ptr_row, &sel_d, &w_d, &aq2, &ad2, &mut moe_out, t,
                                      n_ff_exp, n_embd, n_used, n_expert,
                                      m.down_exps.qtype, m.down_exps.row_bytes)?;
        Ok(moe_out)
    }

    /// gemma4 MoE (R2 router prologue input supplied by caller, R3 per-expert output scale).
    /// Sequential host-staged v0 — softmax gating + renorm (moe_route, the qwen recipe), GELU
    /// experts, scale folded into the accumulate weight (post-matmul linear scale, exact fold).
    fn gemma4_moe(&self, e: &Engine, m: &crate::hybrid::MoeWeights,
                  bits: &crate::hybrid::Gemma4MoeBits, moe_in: &CudaSlice<f32>,
                  router_in: &CudaSlice<f32>, t: usize)
                  -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let moe = cfg.moe.as_ref().unwrap();
        let n_embd = cfg.n_embd as usize;
        let n_expert = moe.expert_count as usize;
        let n_used = moe.expert_used_count as usize;
        let n_ff_exp = moe.expert_ff_length as usize;

        // Router: the in-house GEMV for ALL small t (decode AND verify ride the same per-column
        // kernel — the cuBLASLt n-dependence flipped top-k at verify t on the 27B, d994271);
        // batched matmul only at real prefill.
        let logits = if t < PRIME_MIN_T && crate::router_kernel_on() {
            e.router_gemv(m.gate_inp.float_data(), router_in, n_embd, n_expert, t)?
        } else {
            e.matmul(&m.gate_inp, router_in, t)?
        };

        // FAST SMALL-T ARM (decode t=1 AND spec verify t=2..15): device softmax-topk router,
        // then PER TOKEN the same fused gate_up GELU + down8 FMA launch pair over the resident
        // dev slabs — verify rides the EXACT decode kernel chain per token (dispatch-parity
        // law; the qwen "verify must be kernel-dispatch-identical to decode" lesson).
        if t < PRIME_MIN_T && m.dev_exps.as_ref().is_some_and(|d| !d.gu_il)
            && expert_dp4a_supported(m.gate_exps.qtype) && expert_dp4a_supported(m.up_exps.qtype)
            && expert_dp4a_supported(m.down_exps.qtype)
            && std::env::var("MEMRA_GEMMA_MOE_FAST").as_deref() != Ok("0") {
            let dev = m.dev_exps.as_ref().unwrap();
            let (sel_d, w_d) = e.moe_router_topk_scaled(&logits, t, n_expert, n_used,
                                                        &bits.per_expert_scale_d)?;
            if t == 1 {
                let (zq, zd) = e.quantize_q8_1(moe_in, 1, n_embd)?;
                let selv = sel_d.slice(0..n_used);
                let wv = w_d.slice(0..n_used);
                let act = e.moe_gate_up_gelu8_dev_q8(&dev.ptr_row, &selv, &zq, &zd,
                                                     n_embd, n_ff_exp, n_used, n_expert,
                                                     m.gate_exps.qtype, m.up_exps.qtype,
                                                     m.gate_exps.row_bytes, m.up_exps.row_bytes)?;
                let (aq2, ad2) = e.quantize_q8_1(&act, n_used, n_ff_exp)?;
                let mut moe_out = e.uninit(n_embd)?;
                e.moe_down8_fma_dev_q8(&dev.ptr_row, &selv, &wv, &aq2, &ad2,
                                       &mut moe_out.slice_mut(0..n_embd), n_ff_exp, n_embd,
                                       n_used, n_expert, m.down_exps.qtype, m.down_exps.row_bytes)?;
                return Ok(moe_out);
            }
            // VERIFY ROWS TWINS (t=2..15): ONE launch pair for all tokens; per (token,row,slot)
            // bodies are the t=1 kernels VERBATIM (bit-identical to the per-token loop).
            // gate_up rides the CSR owner-scan dedup when t <= 10 (each duplicated expert's
            // weight stream decoded once; the qwen CSR class). MEMRA_GEMMA_CSR=0 -> rows.
            let (zq, zd) = e.quantize_q8_1(moe_in, t, n_embd)?;
            let csr = t <= 10 && std::env::var("MEMRA_GEMMA_CSR").as_deref() != Ok("0");
            let act = if csr {
                e.moe_gate_up_gelu8_dev_q8_csr(&dev.ptr_row, &sel_d, &zq, &zd, t * n_used,
                                               n_embd, n_ff_exp, n_used, n_expert,
                                               m.gate_exps.qtype, m.up_exps.qtype,
                                               m.gate_exps.row_bytes, m.up_exps.row_bytes)?
            } else {
                e.moe_gate_up_gelu8_dev_q8_rows(&dev.ptr_row, &sel_d, &zq, &zd, t,
                                                n_embd, n_ff_exp, n_used, n_expert,
                                                m.gate_exps.qtype, m.up_exps.qtype,
                                                m.gate_exps.row_bytes, m.up_exps.row_bytes)?
            };
            let (aq2, ad2) = e.quantize_q8_1(&act, t * n_used, n_ff_exp)?;
            let mut moe_out = e.uninit(t * n_embd)?;
            e.moe_down8_fma_dev_q8_rows_g(&dev.ptr_row, &sel_d, &w_d, &aq2, &ad2, &mut moe_out, t,
                                          n_ff_exp, n_embd, n_used, n_expert,
                                          m.down_exps.qtype, m.down_exps.row_bytes)?;
            return Ok(moe_out);
        }

        let (sel_all, mut w_all) = Self::moe_route(e, &logits, t, n_expert, n_used)?;
        for (i, &sx) in sel_all.iter().enumerate() {
            w_all[i] *= bits.per_expert_scale[sx as usize];
        }

        // PREFILL PAIRS ARM (t >= 16): expert-major CSR over the resident slabs — ONE launch per
        // projection covers ALL (token,expert) pairs (the qwen pairs recipe, _em dot = expert_dot_g
        // which has the Q4_0 body; GELU pairs epilogue; R3 scale folded into pair_w).
        if t >= PRIME_MIN_T && m.dev_exps.as_ref().is_some_and(|d| !d.gu_il)
            && expert_dp4a_supported(m.gate_exps.qtype) && expert_dp4a_supported(m.up_exps.qtype)
            && expert_dp4a_supported(m.down_exps.qtype)
            && std::env::var("MEMRA_GEMMA_MOE_PAIRS").as_deref() != Ok("0") {
            let dev = m.dev_exps.as_ref().unwrap();
            let n_pairs = t * n_used;
            let pair_ex: Vec<i32> = sel_all.iter().map(|&x| x as i32).collect();
            let pair_tok: Vec<i32> = (0..n_pairs).map(|p| (p / n_used) as i32).collect();
            let tok_off: Vec<i32> = (0..=t).map(|tok| (tok * n_used) as i32).collect();
            let tok_ids: Vec<i32> = (0..n_pairs as i32).collect();
            let pt = e.htod_i32(&pair_tok)?;
            let pw = e.htod(&w_all)?;
            let toff = e.htod_i32(&tok_off)?;
            let tids = e.htod_i32(&tok_ids)?;
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
            // GROUPED f16 LANE (MEMRA_MOE_F16G=1, round 46 arc 2): dequant active experts to
            // f16 once per projection + one grouped f16 GEMM over the CSR groups; CSR order
            // end-to-end (gelu is elementwise), one row permute before the scatter. The
            // ragged down k (704) needs no padding here — cublas takes any k.
            // f16-mirror numeric class, argmax/spec gated. PER-MODEL default OFF (round 50):
            // the gelu class regressed g26 board-2048 prefill -8.3% under the round-49
            // Hopper default — see moe_f16g_gemma_on.
            if crate::moe_f16g_gemma_on()
                && f16g_proj_ok(m.gate_exps.qtype, n_embd)
                && f16g_proj_ok(m.up_exps.qtype, n_embd)
                && f16g_proj_ok(m.down_exps.qtype, n_ff_exp) {
                let csr_tok: Vec<i32> = ex_pairs.iter().map(|&p| p / n_used as i32).collect();
                let csr_tok_d = e.htod_i32(&csr_tok)?;
                let (z_f16, z_s) = e.moe_f16g_act(moe_in, Some(&csr_tok_d), n_embd, n_pairs)?;
                let g_csr = e.moe_f16_grouped(&dev.ptr_row, 0, n_expert, &exi, &ex_off, &exo,
                                              &z_f16, &z_s, n_embd, n_ff_exp, n_active, n_pairs,
                                              m.gate_exps.qtype, m.gate_exps.row_bytes)?;
                let u_csr = e.moe_f16_grouped(&dev.ptr_row, 1, n_expert, &exi, &ex_off, &exo,
                                              &z_f16, &z_s, n_embd, n_ff_exp, n_active, n_pairs,
                                              m.up_exps.qtype, m.up_exps.row_bytes)?;
                let act_csr = e.moe_pairs_gelu_mul(&g_csr, &u_csr, n_pairs * n_ff_exp)?;
                let (a_f16, a_s) = e.moe_f16g_act(&act_csr, None, n_ff_exp, n_pairs)?;
                let d_csr = e.moe_f16_grouped(&dev.ptr_row, 2, n_expert, &exi, &ex_off, &exo,
                                              &a_f16, &a_s, n_ff_exp, n_embd, n_active, n_pairs,
                                              m.down_exps.qtype, m.down_exps.row_bytes)?;
                let y_down = e.rows_permute(&d_csr, &exp_d, n_pairs, n_embd)?;
                let mut moe_out = e.uninit(t * n_embd)?;
                e.moe_pairs_scatter(&y_down, &pw, &toff, &tids, &mut moe_out, t, n_embd)?;
                if std::env::var("MEMRA_F16G_DEBUG").is_ok() {
                    let scan = |v: &[f32]| v.iter().filter(|x| !x.is_finite()).count();
                    let (yd, mo) = (e.dtoh(&y_down)?, e.dtoh(&moe_out)?);
                    eprintln!("[f16g-debug] post-permute bad={} post-scatter bad={}",
                              scan(&yd), scan(&mo));
                }
                return Ok(moe_out);
            }
            // gate/up: int8-MMA expert GEMM (in_f = n_embd 2816 = 11x256 tiles ok); down keeps
            // the decode-once dp4a (in_f 704 fails the 256-superblock tiling).
            let mma = n_embd % 256 == 0
                && std::env::var("MEMRA_GEMMA_MOE_MMA").as_deref() != Ok("0");
            let (gate, up) = if mma {
                let z_scr = e.mmq_iq_quantize_act(moe_in, n_embd, t)?;
                (e.mmq_iq_experts(&dev.ptr_row, 0, n_expert, &exi, &exo, &exp_d, &pt, &z_scr,
                                  n_embd, n_ff_exp, n_active, n_pairs, t,
                                  m.gate_exps.qtype, m.gate_exps.row_bytes)?,
                 e.mmq_iq_experts(&dev.ptr_row, 1, n_expert, &exi, &exo, &exp_d, &pt, &z_scr,
                                  n_embd, n_ff_exp, n_active, n_pairs, t,
                                  m.up_exps.qtype, m.up_exps.row_bytes)?)
            } else {
                let (zq, zd) = e.quantize_q8_1(moe_in, t, n_embd)?;
                (e.moe_pairs_matvec_q8_dec(&dev.ptr_row, 0, &exi, &exo, &exp_d, &pt, &zq, &zd,
                                           n_embd, n_ff_exp, n_expert, n_active, n_pairs,
                                           m.gate_exps.qtype, m.gate_exps.row_bytes)?,
                 e.moe_pairs_matvec_q8_dec(&dev.ptr_row, 1, &exi, &exo, &exp_d, &pt, &zq, &zd,
                                           n_embd, n_ff_exp, n_expert, n_active, n_pairs,
                                           m.up_exps.qtype, m.up_exps.row_bytes)?)
            };
            let pair_self: Vec<i32> = (0..n_pairs as i32).collect();
            let pself = e.htod_i32(&pair_self)?;
            // DOWN through the int8-MMA expert GEMM (2026-07-31, g26 prefill lever): the
            // ragged k (n_ff_exp=704 on the 26B) rides a PADDED k-walk — in_f rounds up
            // to the 256-val superblock (768) while the act quantizer's zero padding
            // makes every padded-k product exactly zero (weight overread bytes multiply
            // zero int8 act values; the dev slab carries 144B tail slack for the OOB).
            // The old dp4a matvec was 11.3ms/call at m=T (the 0.07x prefill wall).
            // MEMRA_GEMMA_MOE_MMA=0 reverts down together with gate/up.
            // FUSED ACT-EPILOGUE (default on): gelu_tanh(gate)*up + D4 quantize in one
            // launch — no f32 act buffer (the fused kernel zero-pads the ragged tail
            // exactly like the two-pass quantizer). MEMRA_MOE_FUSE_ACTQ=0 rollback.
            // Scratch bytes are BYTE-IDENTICAL (kernel-check gated).
            let y_down = if mma {
                let in_pad = n_ff_exp.div_ceil(256) * 256;
                let a_scr = if crate::moe_fuse_actq_on() {
                    e.mmq_iq_fused_act_quant(&gate, &up, n_ff_exp, n_pairs, 1)?
                } else {
                    let act = e.moe_pairs_gelu_mul(&gate, &up, n_pairs * n_ff_exp)?;
                    e.mmq_iq_quantize_act(&act, n_ff_exp, n_pairs)?
                };
                e.mmq_iq_experts(&dev.ptr_row, 2, n_expert, &exi, &exo, &exp_d, &pself, &a_scr,
                                 in_pad, n_embd, n_active, n_pairs, n_pairs,
                                 m.down_exps.qtype, m.down_exps.row_bytes)?
            } else {
                let act = e.moe_pairs_gelu_mul(&gate, &up, n_pairs * n_ff_exp)?;
                let (aq2, ad2) = e.quantize_q8_1(&act, n_pairs, n_ff_exp)?;
                e.moe_pairs_matvec_q8_dec(&dev.ptr_row, 2, &exi, &exo, &exp_d, &pself, &aq2, &ad2,
                                          n_ff_exp, n_embd, n_expert, n_active, n_pairs,
                                          m.down_exps.qtype, m.down_exps.row_bytes)?
            };
            let mut moe_out = e.uninit(t * n_embd)?;
            e.moe_pairs_scatter(&y_down, &pw, &toff, &tids, &mut moe_out, t, n_embd)?;
            return Ok(moe_out);
        }

        let g_len = m.gate_exps.expert_stride;
        let u_len = m.up_exps.expert_stride;
        let d_len = m.down_exps.expert_stride;
        // Resident dev slabs (fits-VRAM regime): read each expert straight from the device slab
        // at ex*stride — zero H2D, SAME qmatvec_view kernel/bytes as the staged path. Staging is
        // the spill fallback.
        let dev = m.dev_exps.as_ref().filter(|d| !d.gu_il);
        let (mut sg, mut su, mut sd) = if dev.is_some() { (None, None, None) } else {
            (Some(e.alloc_u8_uninit(g_len)?), Some(e.alloc_u8_uninit(u_len)?), Some(e.alloc_u8_uninit(d_len)?))
        };
        let mut moe_out = e.zeros(t * n_embd)?;
        for tok in 0..t {
            let sel = &sel_all[tok * n_used..(tok + 1) * n_used];
            let w = &w_all[tok * n_used..(tok + 1) * n_used];
            let zt = moe_in.slice(tok * n_embd..(tok + 1) * n_embd);
            for (j, &ex) in sel.iter().enumerate() {
                let ex = ex as usize;
                let gate = match dev {
                    Some(d) => e.qmatvec_view(&d.gate, ex * g_len..(ex + 1) * g_len, &zt, 1,
                        m.gate_exps.in_f, m.gate_exps.out_f, m.gate_exps.qtype, m.gate_exps.row_bytes)?,
                    None => {
                        let sg = sg.as_mut().unwrap();
                        e.stage_expert(m.gate_exps.expert_bytes(ex), sg, 0)?;
                        e.qmatvec_view(sg, 0..g_len, &zt, 1,
                            m.gate_exps.in_f, m.gate_exps.out_f, m.gate_exps.qtype, m.gate_exps.row_bytes)?
                    }
                };
                let up = match dev {
                    Some(d) => e.qmatvec_view(&d.up, ex * u_len..(ex + 1) * u_len, &zt, 1,
                        m.up_exps.in_f, m.up_exps.out_f, m.up_exps.qtype, m.up_exps.row_bytes)?,
                    None => {
                        let su = su.as_mut().unwrap();
                        e.stage_expert(m.up_exps.expert_bytes(ex), su, 0)?;
                        e.qmatvec_view(su, 0..u_len, &zt, 1,
                            m.up_exps.in_f, m.up_exps.out_f, m.up_exps.qtype, m.up_exps.row_bytes)?
                    }
                };
                let mut act = e.uninit(n_ff_exp)?;
                e.gelu_tanh_mul(&gate, &up, &mut act, n_ff_exp)?;
                let actv = act.slice(0..n_ff_exp);
                let y = match dev {
                    Some(d) => e.qmatvec_view(&d.down, ex * d_len..(ex + 1) * d_len, &actv, 1,
                        m.down_exps.in_f, m.down_exps.out_f, m.down_exps.qtype, m.down_exps.row_bytes)?,
                    None => {
                        let sd = sd.as_mut().unwrap();
                        e.stage_expert(m.down_exps.expert_bytes(ex), sd, 0)?;
                        e.qmatvec_view(sd, 0..d_len, &actv, 1,
                            m.down_exps.in_f, m.down_exps.out_f, m.down_exps.qtype, m.down_exps.row_bytes)?
                    }
                };
                let mut dst = moe_out.slice_mut(tok * n_embd..(tok + 1) * n_embd);
                e.axpy_into(&y, w[j], &mut dst, n_embd)?;
            }
        }
        Ok(moe_out)
    }

    /// One gemma4 trunk layer (R8): x -> x_next.
    fn gemma4_layer(&self, e: &Engine, il: usize, layer: &crate::hybrid::HybridLayer,
                    x: &CudaSlice<f32>, pos_d: &CudaSlice<i32>, t: usize)
                    -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let n_embd = self.cfg.n_embd as usize;
        let eps = self.cfg.rms_eps;

        let mut h = e.zeros(t * n_embd)?;
        e.rms_norm(x, layer.attn_norm.float_data(), &mut h, n_embd, t, eps)?;
        let Mixer::Full(fa) = &layer.mixer else { panic!("gemma4 layer {il} not full-attn") };
        let o = self.gemma4_attn(e, fa, il, &h, pos_d, t)?;
        // gemma order: post_attention_norm applies to the ATTENTION OUTPUT, then the residual.
        let mut cur = e.zeros(t * n_embd)?;
        e.rms_norm(&o, layer.post_attn_norm.float_data(), &mut cur, n_embd, t, eps)?;
        self.gemma4_layer_tail_add(e, layer, &cur, x, t)
    }

    /// Everything after the attention output in a gemma4 layer: the residual add (cur + x ->
    /// attn_out) FUSED with the three attn_out norms, then shared FFN + router + MoE + combine +
    /// layer scale — shared verbatim by the prefill, decode and verify paths.
    fn gemma4_layer_tail_add(&self, e: &Engine, layer: &crate::hybrid::HybridLayer,
                             cur: &CudaSlice<f32>, x: &CudaSlice<f32>, t: usize)
                             -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        Ok(self.gemma4_layer_tail_add_n(e, layer, cur, x, t, None)?.0)
    }

    /// tail_add with the NEXT layer's attn_norm fused into the closing add+scale (one launch
    /// produces both x_next and h_next — the cross-layer fusion). `next_norm` None = last layer.
    fn gemma4_layer_tail_add_n(&self, e: &Engine, layer: &crate::hybrid::HybridLayer,
                               cur: &CudaSlice<f32>, x: &CudaSlice<f32>, t: usize,
                               next_norm: Option<&CudaSlice<f32>>)
                               -> Result<(CudaSlice<f32>, Option<CudaSlice<f32>>), Box<dyn std::error::Error>> {
        let n_embd = self.cfg.n_embd as usize;
        let bits = layer.gemma4.as_ref().unwrap();
        let (sn, attn_out) = self.gemma4_layer_tail_core(e, layer, cur, x, t)?;
        let mut xn = e.uninit(t * n_embd)?;
        match next_norm {
            Some(w) => {
                let mut hn = e.uninit(t * n_embd)?;
                e.add_scale_rms_norm(&sn, &attn_out, bits.layer_scale, w, &mut xn, &mut hn,
                                     n_embd, t, self.cfg.rms_eps)?;
                Ok((xn, Some(hn)))
            }
            None => {
                e.add_scale(&sn, &attn_out, bits.layer_scale, &mut xn, t * n_embd)?;
                Ok((xn, None))
            }
        }
    }

    /// Tail core: attn_out = cur+x (fused with the 3 norms), both FFN branches, the combine
    /// norm — returns (sn, attn_out) for the closing add+scale variants.
    fn gemma4_layer_tail_core(&self, e: &Engine, layer: &crate::hybrid::HybridLayer,
                              cur: &CudaSlice<f32>, x: &CudaSlice<f32>, t: usize)
                              -> Result<(CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        self.gemma4_layer_tail_core_pn(e, layer, cur, x, t, None, false)
    }

    /// tail_core with an optional PRE-NORM fold (glue-fusion lane): `pre_norm = Some(wa)`
    /// means `cur` is the RAW attention output and the dense entry runs
    /// rms(cur, wa) + residual-add + ffn_norm as ONE launch (E4B's post-attn norm fold).
    /// `defer_post_norm`: dense-arm exit returns RAW f0 (ffn_down output) instead of
    /// sn = rms(f0, post_ffw) — the caller fuses the post-norm into its residual emit
    /// (rms_pre_add_q8_1, E4B glue wave 5). MoE arm ignores it.
    fn gemma4_layer_tail_core_pn(&self, e: &Engine, layer: &crate::hybrid::HybridLayer,
                                 cur: &CudaSlice<f32>, x: &CudaSlice<f32>, t: usize,
                                 pre_norm: Option<&CudaSlice<f32>>, defer_post_norm: bool)
                                 -> Result<(CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let n_embd = self.cfg.n_embd as usize;
        let eps = self.cfg.rms_eps;
        let bits = layer.gemma4.as_ref().unwrap();

        // DENSE gemma4 variants (31B/E4B — no MoE, no parallel branch): attn_out = cur + x
        // fused with the single ffn_norm; GELU_PAR ffn; post_ffw_norm.
        let Some(mbits) = bits.moe_bits.as_ref() else {
            let crate::hybrid::Ffn::Dense { ffn_gate, ffn_up, ffn_down } = &layer.ffn
            else { panic!("gemma4 dense layer without Dense ffn") };
            let mut attn_out = e.uninit(t * n_embd)?;
            let mut zsh = e.uninit(t * n_embd)?;
            // wave-2: with the pre-norm fold active the entry ALSO emits zsh q8_1 — the
            // t=1 fused2 gate/up consume the pair with no standalone quantize launch.
            let mut zpair: Option<(CudaSlice<i8>, CudaSlice<f32>)> = None;
            match pre_norm {
                Some(wa) if t == 1 => {
                    zpair = Some(e.rms_pre_add_rms_norm_q8z(cur, wa, x,
                                                            bits.ffn_norm.float_data(),
                                                            &mut attn_out, &mut zsh,
                                                            n_embd, t, eps)?);
                }
                Some(wa) => e.rms_pre_add_rms_norm(cur, wa, x, bits.ffn_norm.float_data(),
                                                   &mut attn_out, &mut zsh, n_embd, t, eps)?,
                None => e.add_rms_norm(cur, x, bits.ffn_norm.float_data(), &mut attn_out,
                                       &mut zsh, n_embd, t, eps)?,
            }
            let n_ff = ffn_gate.out_features();
            // FFN persistent slab (counter-barrier form) FALSIFIED here 2026-07-14
            // (falsification #7, jsonl row): PDL glue already hides the launch boundaries
            // it fused; barrier + worst-segment occupancy net −0.3% (31B depth) / −2.3%
            // (E4B spec). Down's act dependency is all-to-all, so sentinel sync cannot
            // rescue segment C — the megakernel front is closed for the dense tail.
            let (gate, up) = if t == 1 {
                let (zq, zd) = match zpair {
                    Some(p) => p,
                    None => e.quantize_q8_1(&zsh, 1, n_embd)?,
                };
                match e.matmul_q4_fused2(ffn_gate, ffn_up, &zq, &zd)? {
                    Some(p) => p,
                    None => (e.matmul_pre(ffn_gate, &zq, &zd, &zsh, 1)?,
                             e.matmul_pre(ffn_up, &zq, &zd, &zsh, 1)?),
                }
            } else {
                // BATCHED FUSED2 (DEFAULT ON 2026-07-13, MEMRA_F2B=0 seam): one segmented
                // launch for the verify's gate+up — the up segment's blocks fill SMs as
                // the gate segment drains (the launch-tail mechanism behind the b-tier
                // plateau; first positive after six falsified in-kernel variants).
                static F2B: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                let f2b = *F2B.get_or_init(|| std::env::var("MEMRA_F2B").as_deref() != Ok("0"));
                let fused = if f2b {
                    let (zq, zd) = e.quantize_q8_1(&zsh, t, n_embd)?;
                    e.matmul_q4_fused2_batched(ffn_gate, ffn_up, &zq, &zd, t)?
                } else { None };
                match fused {
                    Some(p) => p,
                    None => {
                        // quantize-once window: gate/up share `zsh` (borrowed across the pair).
                        e.mmq_act_begin();
                        (e.matmul(ffn_gate, &zsh, t)?, e.matmul(ffn_up, &zsh, t)?)
                    }
                }
            };
            let mut act = e.uninit(t * n_ff)?;
            // act quantize folds into the GELU epilogue (bit-identical q8_1 rounding);
            // ffn_down rides matmul_pre — one quantize launch fewer per layer.
            let f0 = if e.uses_q8_1_fast(ffn_down) {
                let upv = e.view(&up, t * n_ff);
                let up_all = upv.slice(0..t * n_ff);
                let (aq, ad) = e.gelu_tanh_mul_q8_1(&gate, &up_all, &mut act, n_ff, t)?;
                e.matmul_pre(ffn_down, &aq, &ad, &act, t)?
            } else {
                e.gelu_tanh_mul(&gate, &up, &mut act, t * n_ff)?;
                e.matmul(ffn_down, &act, t)?
            };
            if defer_post_norm { return Ok((f0, attn_out)); }
            let mut sn = e.uninit(t * n_embd)?;
            e.rms_norm(&f0, bits.post_ffw_norm.float_data(), &mut sn, n_embd, t, eps)?;
            return Ok((sn, attn_out));
        };

        assert!(pre_norm.is_none(), "pre-norm fold is dense-entry only");
        // MoE variant (26B): attn_out = cur + x fused with the three attn_out norms
        // (ffn_norm + router-scale + pre_ffw_norm_2): ONE launch, chains verbatim. At small t
        // (decode + verify) the zsh and moe_in outputs are EMITTED q8_1 (both consumers are
        // quantized matmuls — two quantize launches + two f32 round-trips fold away).
        let mut attn_out = e.uninit(t * n_embd)?;
        let mut router_in = e.uninit(t * n_embd)?;
        let fast_moe = match &layer.ffn {
            crate::hybrid::Ffn::Moe(m) => m.dev_exps.as_ref().is_some_and(|d| !d.gu_il)
                && expert_dp4a_supported(m.gate_exps.qtype)
                && expert_dp4a_supported(m.up_exps.qtype)
                && expert_dp4a_supported(m.down_exps.qtype)
                && std::env::var("MEMRA_GEMMA_MOE_FAST").as_deref() != Ok("0"),
            _ => false,
        };
        let q8z = t < PRIME_MIN_T && fast_moe;
        let (zsh_f32, zsh_q8, moe_q8) = if q8z {
            let (z0, m2) = e.add_rms_norm3_q8z(cur, x, bits.ffn_norm.float_data(),
                                               &mbits.router_scale_pre,
                                               mbits.pre_ffw_norm_2.float_data(),
                                               &mut attn_out, &mut router_in, n_embd, t, eps)?;
            (None, Some(z0), Some(m2))
        } else {
            let mut zsh = e.uninit(t * n_embd)?;
            let mut moe_in = e.uninit(t * n_embd)?;
            e.add_rms_norm3(cur, x, bits.ffn_norm.float_data(), &mbits.router_scale_pre,
                            mbits.pre_ffw_norm_2.float_data(), &mut attn_out, &mut zsh,
                            &mut router_in, &mut moe_in, n_embd, t, eps)?;
            (Some((zsh, moe_in)), None, None)
        };
        let attn_out2 = attn_out;
        #[allow(unused_variables)]
        let attn_out = &attn_out2;
        let n_ff = mbits.shared_gate.out_features();
        let (gate, up) = if let Some((zq, zd)) = zsh_q8.as_ref() {
            if t == 1 {
                match e.matmul_q4_fused2(&mbits.shared_gate, &mbits.shared_up, zq, zd)? {
                    Some(p) => p,
                    None => {
                        let h0 = e.zeros(0)?;
                        (e.matmul_pre(&mbits.shared_gate, zq, zd, &h0, 1)?,
                         e.matmul_pre(&mbits.shared_up, zq, zd, &h0, 1)?)
                    }
                }
            } else {
                // verify t 2..15: the batched mmvq twins consume the pre-quantized pair.
                let h0 = e.zeros(0)?;
                (e.matmul_pre(&mbits.shared_gate, zq, zd, &h0, t)?,
                 e.matmul_pre(&mbits.shared_up, zq, zd, &h0, t)?)
            }
        } else {
            let (zsh, _) = zsh_f32.as_ref().unwrap();
            (e.matmul(&mbits.shared_gate, zsh, t)?, e.matmul(&mbits.shared_up, zsh, t)?)
        };
        let mut act = e.uninit(t * n_ff)?;
        e.gelu_tanh_mul(&gate, &up, &mut act, t * n_ff)?;
        let mlp0 = e.matmul(&mbits.shared_down, &act, t)?;
        let crate::hybrid::Ffn::Moe(m) = &layer.ffn else { panic!("gemma4 layer not MoE") };
        let moe0 = match (&moe_q8, &zsh_f32) {
            (Some(mq), _) => self.gemma4_moe_q8(e, m, mbits, mq, &router_in, t)?,
            (None, Some((_, moe_in))) => self.gemma4_moe(e, m, mbits, moe_in, &router_in, t)?,
            _ => unreachable!(),
        };
        // post_ffw_norm_1(mlp0) + post_ffw_norm_2(moe0): one fused launch, per-row verbatim.
        let mut mlp = e.uninit(t * n_embd)?;
        let mut moe = e.uninit(t * n_embd)?;
        e.rms_norm2x(&mlp0, &moe0, mbits.post_ffw_norm_1.float_data(),
                     mbits.post_ffw_norm_2.float_data(), &mut mlp, &mut moe, n_embd, t, eps)?;

        // combine: rms_norm(mlp + moe, post_ffw_norm) + attn_out, then the layer output scalar.
        // add+norm fused (add_rms_norm == add then rms_norm, kernel-check-pinned identity).
        let mut sum = e.uninit(t * n_embd)?;
        let mut sn = e.uninit(t * n_embd)?;
        e.add_rms_norm(&mlp, &moe, bits.post_ffw_norm.float_data(), &mut sum, &mut sn,
                       n_embd, t, eps)?;
        Ok((sn, attn_out2))
    }

    /// tail_add with the next attn_norm emitted PRE-QUANTIZED q8_1 (decode/verify loops).
    fn gemma4_layer_tail_add_nq(&self, e: &Engine, layer: &crate::hybrid::HybridLayer,
                                cur: &CudaSlice<f32>, x: &CudaSlice<f32>, t: usize,
                                next_norm: Option<&CudaSlice<f32>>)
                                -> Result<(CudaSlice<f32>, Option<(CudaSlice<i8>, CudaSlice<f32>)>), Box<dyn std::error::Error>> {
        let n_embd = self.cfg.n_embd as usize;
        let bits = layer.gemma4.as_ref().unwrap();
        let (sn, attn_out) = self.gemma4_layer_tail_core(e, layer, cur, x, t)?;
        let mut xn = e.uninit(t * n_embd)?;
        match next_norm {
            Some(w) => {
                let pair = e.add_scale_rms_norm_q8_1(&sn, &attn_out, bits.layer_scale, w, &mut xn,
                                                     n_embd, t, self.cfg.rms_eps)?;
                Ok((xn, Some(pair)))
            }
            None => {
                e.add_scale(&sn, &attn_out, bits.layer_scale, &mut xn, t * n_embd)?;
                Ok((xn, None))
            }
        }
    }

    /// gemma4 prefill: `last_only` = forward_last semantics (lm_head on the final row only).
    /// R4: final logits softcapped 30*tanh(l/30) on host (monotonic — argmax unaffected).
    fn gemma4_forward(&self, e: &Engine, tokens: &[u32], last_only: bool)
                      -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        // E4B routes to its own forward regardless of the caller's entry point (forward /
        // forward_last / prime paths all funnel here for gemma4).
        if self.is_gemma4_e4b() { return self.gemma4_e4b_forward(e, tokens, last_only); }
        let n_embd = self.cfg.n_embd as usize;
        let t = tokens.len();
        let pos: Vec<i32> = (0..t as i32).collect();
        let pos_d = e.htod_i32(&pos)?;

        let mut x = self.embed(e, tokens)?;
        e.scale_inplace(&mut x, (n_embd as f32).sqrt(), t * n_embd)?;
        // MEMRA_GEMMA_PROBE=1: per-layer trunk stats (host rms + max of the LAST token row) —
        // the bring-up bisect vs llama-eval-callback node stats.
        let probe = std::env::var("MEMRA_GEMMA_PROBE").is_ok();
        let stat = |e: &Engine, x: &CudaSlice<f32>, tag: &str| -> Result<(), Box<dyn std::error::Error>> {
            let h = e.dtoh(x)?;
            let bad = h.iter().filter(|v| !v.is_finite()).count();
            let mx = h.iter().filter(|v| v.is_finite()).fold(0.0f32, |m, v| m.max(v.abs()));
            eprintln!("[gemma-probe] {tag}: tok0_first3={:?} bad={bad} max={mx:.3e}", &h[..3]);
            Ok(())
        };
        if probe { stat(e, &x, "embed")?; }
        for (il, layer) in self.layers.iter().enumerate() {
            x = self.gemma4_layer(e, il, layer, &x, &pos_d, t)?;
            if probe { stat(e, &x, &format!("L{il}"))?; }
        }
        let mut hn = e.zeros(t * n_embd)?;
        e.rms_norm(&x, self.output_norm.float_data(), &mut hn, n_embd, t, self.cfg.rms_eps)?;
        let cap = self.cfg.gemma4.as_ref().unwrap().final_logit_softcapping;
        let n_vocab = self.output.out_features();
        let logits = if last_only {
            let hv = e.view(&hn, t * n_embd);
            let last_row = hv.slice((t - 1) * n_embd..t * n_embd);
            let mut hlast = e.zeros(n_embd)?;
            e.copy_view_into(&mut hlast, 0, &last_row, n_embd)?;
            let mut ld = e.matmul(&self.output, &hlast, 1)?;
            e.softcap(&mut ld, cap, n_vocab)?;
            self.gemma4_suppress(e, &mut ld, 1)?;
            e.dtoh(&ld)?
        } else {
            let mut ld = e.matmul(&self.output, &hn, t)?;
            e.softcap(&mut ld, cap, t * n_vocab)?;
            self.gemma4_suppress(e, &mut ld, t)?;
            e.dtoh(&ld)?
        };
        Ok(logits)
    }

    /// gemma4 BATCHED PROMPT PRIME v0 (fresh cache only): the prefill graph over the whole
    /// prompt with each layer's post-rope K / weightless-normed V appended into the quantized
    /// KV cache (decode-append row math). Returns (last-row softcapped logits, h_seed = last
    /// pre-output_norm hidden, hiddens = full pre-output_norm stack [T, n_embd]).
    pub(crate) fn gemma4_prime(&self, e: &Engine, tokens: &[u32], cache: &mut Cache)
                               -> Result<(Vec<f32>, CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        // Err, not assert (2026-08-07, lane/gemma4-serve-gaps): a served gemma4 prompt longer
        // than the worker's prefill tick used to chunk here, and chunk 2 (pos > 0) killed the
        // whole worker process on this line. The worker now primes gemma4 monolithically and
        // routes continuation suffixes tokenwise; this is the per-request backstop.
        if cache.pos != 0 {
            return Err("gemma4 prime v0 is fresh-prompt only (no continuation/chunked prime) \
                        — prime the full prompt in one call or decode tokenwise".into());
        }
        let n_embd = self.cfg.n_embd as usize;
        let eps = self.cfg.rms_eps;
        let t = tokens.len();
        let pos: Vec<i32> = (0..t as i32).collect();
        let pos_d = e.htod_i32(&pos)?;
        let mut x = self.embed(e, tokens)?;
        e.scale_inplace(&mut x, (n_embd as f32).sqrt(), t * n_embd)?;
        for (il, layer) in self.layers.iter().enumerate() {
            let mut h = e.zeros(t * n_embd)?;
            e.rms_norm(&x, layer.attn_norm.float_data(), &mut h, n_embd, t, eps)?;
            let Mixer::Full(fa) = &layer.mixer else { panic!("gemma4 layer not full-attn") };
            let o = self.gemma4_attn_prime(e, fa, il, &h, &pos_d, t, Some(cache))?;
            let mut cur = e.zeros(t * n_embd)?;
            e.rms_norm(&o, layer.post_attn_norm.float_data(), &mut cur, n_embd, t, eps)?;
            x = self.gemma4_layer_tail_add(e, layer, &cur, &x, t)?;
            self.dflash_tap(e, cache, il, &x, t)?;
        }
        cache.pos += t;
        let hiddens = e.clone_dtod(&x)?;
        let xv = e.view(&x, t * n_embd);
        let last_row = xv.slice((t - 1) * n_embd..t * n_embd);
        let mut h_seed = e.zeros(n_embd)?;
        e.copy_view_into(&mut h_seed, 0, &last_row, n_embd)?;
        let mut hn = e.uninit(n_embd)?;
        e.rms_norm(&h_seed, self.output_norm.float_data(), &mut hn, n_embd, 1, eps)?;
        let mut ld = e.matmul(&self.output, &hn, 1)?;
        let cap = self.cfg.gemma4.as_ref().unwrap().final_logit_softcapping;
        e.softcap(&mut ld, cap, self.output.out_features())?;
        self.gemma4_suppress(e, &mut ld, 1)?;
        let logits = e.dtoh(&ld)?;
        Ok((logits, h_seed, hiddens))
    }

    /// gemma4 T=1 decode attention: per-layer geometry, quantized-KV append + fa_decode
    /// (vec kernels at hd 256, generic scalar at the globals' hd 512), weightless V-norm,
    /// dual rope, scale 1.0. Takes the attn-normed input PRE-QUANTIZED (the cross-layer
    /// fused norm emits q8 directly — the f32 h never materializes).
    fn gemma4_decode_attn(&self, e: &Engine, fa: &crate::hybrid::FullAttnLayer, il: usize,
                          hq: &CudaSlice<i8>, hdq: &CudaSlice<f32>,
                          pos_d: &CudaSlice<i32>, cache: &mut Cache)
                          -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let (hd, nkv, nh, base, scale, swa) = self.gemma4_geom(il);
        let eps = self.cfg.rms_eps;
        let aux = self.gemma4_aux.as_ref().unwrap();
        let (hq, hdq) = (hq, hdq);
        let h0 = e.zeros(0)?;
        let h = &h0;
        let (q0, k0, v0) = if swa {
            match e.matmul_q4_fused3(&fa.wq, &fa.wk, &fa.wv, &hq, &hdq)? {
                Some(t3) => t3,
                None => (e.matmul_pre(&fa.wq, &hq, &hdq, h, 1)?,
                         e.matmul_pre(&fa.wk, &hq, &hdq, h, 1)?,
                         e.matmul_pre(&fa.wv, &hq, &hdq, h, 1)?),
            }
        } else {
            let (q0, k0) = match e.matmul_q4_fused2(&fa.wq, &fa.wk, &hq, &hdq)? {
                Some(p) => p,
                None => (e.matmul_pre(&fa.wq, &hq, &hdq, h, 1)?,
                         e.matmul_pre(&fa.wk, &hq, &hdq, h, 1)?),
            };
            let v0 = e.clone_dtod(&k0)?;
            (q0, k0, v0)
        };
        let mut q = e.uninit(nh * hd)?;
        let mut k = e.uninit(nkv * hd)?;
        let mut v = e.uninit(nkv * hd)?;
        // E4B wave-3 fold, m=1 completion (2026-07-23): the rows arms took this fold at
        // 550fcfa5; the decode-step trio kept 2 launches/layer it doesn't need.
        let ff = if swa { None } else {
            Some(aux.rope_freqs.as_ref().expect("gemma4 global rope needs rope_freqs.weight"))
        };
        e.rms_norm_qkv_rope(&q0, &k0, &v0, fa.q_norm.float_data(), fa.k_norm.float_data(),
                            &aux.ones, &mut q, &mut k, &mut v, hd, nh, nkv,
                            pos_d, nh, nkv, base, 1.0, ff, eps)?;
        let kvl = cache.kv[il].as_mut().unwrap();
        e.append_kv_quantized(&k, &v, &mut kvl.k, &mut kvl.v, kvl.len,
                              kvl.kv_dim_k, kvl.kv_dim_v, kvl.k_tok_bytes, kvl.v_tok_bytes, (!swa && crate::Engine::gkv_on()) || (swa && crate::Engine::wkv_on()))?;
        kvl.len += 1;
        // R6 decode: SWA layers attend only the last `sliding_window` keys — a token-aligned
        // VIEW OFFSET into the quantized cache (keys carry absolute rope; the mask is purely
        // positional). Globals attend the full history.
        let win = self.cfg.gemma4.as_ref().unwrap().sliding_window as usize;
        let mut attn = e.uninit(nh * hd)?;
        // global layers: SAME hd512 rows twin as verify with t=1 (parity law).
        if !swa && hd == 512 && kvl.len >= crate::fa512_min_tkv()
            && std::env::var("MEMRA_GEMMA_ROWS_W").as_deref() != Ok("0") {
            let kp = e.view_u8(&kvl.k, kvl.len * kvl.k_tok_bytes);
            let vp = e.view_u8(&kvl.v, kvl.len * kvl.v_tok_bytes);
            // device-len: eager syncs the counter (async arg-store), dc keeps it live.
            let base = kvl.len as i32;
            e.i32_set_k(&mut kvl.len_d, base)?;
            e.fa_decode_rows(&q, &kp, &vp, &mut attn, hd, nh, nkv, kvl.len - 1, 1, scale,
                             kvl.k_tok_bytes, kvl.v_tok_bytes, Some((&kvl.len_d, -1)), false,
                             false, None)?;
            return Ok(e.matmul(&fa.wo, &attn, 1)?);
        }
        // windowed regime: SAME rows_w kernel as verify with t=1 (parity law — see verify_attn).
        if swa && kvl.len > win && hd == 256
            && std::env::var("MEMRA_GEMMA_ROWS_W").as_deref() != Ok("0") {
            let kp = e.view_u8(&kvl.k, kvl.len * kvl.k_tok_bytes);
            let vp = e.view_u8(&kvl.v, kvl.len * kvl.v_tok_bytes);
            let base = kvl.len as i32;
            e.i32_set_k(&mut kvl.len_d, base)?;
            e.fa_decode_rows_w(&q, &kp, &vp, &mut attn, hd, nh, nkv, &kvl.len_d, -1, 1, scale,
                               win, kvl.k_tok_bytes, kvl.v_tok_bytes, None)?;
            return Ok(e.matmul(&fa.wo, &attn, 1)?);
        }
        let (off_tok, t_kv) = if swa && kvl.len > win { (kvl.len - win, win) } else { (0, kvl.len) };
        let k_view = e.view_u8_range(&kvl.k, off_tok * kvl.k_tok_bytes,
                                     (off_tok + t_kv) * kvl.k_tok_bytes);
        let v_view = e.view_u8_range(&kvl.v, off_tok * kvl.v_tok_bytes,
                                     (off_tok + t_kv) * kvl.v_tok_bytes);
        e.fa_decode_kvmod(&q, &k_view, &v_view, &mut attn, hd, nh, nkv, t_kv, scale,
                    kvl.k_tok_bytes, kvl.v_tok_bytes, swa && crate::Engine::wkv_on())?;
        Ok(e.matmul(&fa.wo, &attn, 1)?)
    }

    /// gemma4 DEVICE-COUNTER decode step (graph arc): token id + rope pos + KV lengths live in
    /// device counters; ZERO varying host kernel args. `cap_bucket_max` = Some(bucket) for graph
    /// capture (host mirrors untouched, n_splits from bucket, full-buffer KV views) / None for
    /// the eager-dc gate path (host mirrors advanced, live geometry — bit-identical target =
    /// gemma4_decode_step_h's token stream). V1 scope: t_kv <= sliding_window (no window views
    /// in-graph; the driver gates).
    #[allow(clippy::too_many_arguments)]
    pub fn gemma4_decode_step_dc(&self, e: &Engine, token_d: &CudaSlice<u32>,
                                 pos_d: &mut CudaSlice<i32>, embd_gpu: &CudaSlice<u8>,
                                 embd_qt: i32, embd_rb: usize, cache: &mut Cache,
                                 n_vocab: usize, cap_bucket_max: Option<(usize, usize)>)
                                 -> Result<CudaSlice<u32>, Box<dyn std::error::Error>> {
        let mut tok_out = e.stream().alloc_zeros::<u32>(1)?;
        self.gemma4_decode_step_dc_into(e, token_d, pos_d, embd_gpu, embd_qt, embd_rb, cache,
                                        n_vocab, cap_bucket_max, &mut tok_out)?;
        Ok(tok_out)
    }

    /// CAPTURE body: argmax lands in the PERSISTENT `tok_out` (same buffer = same address on
    /// every replay; pass `token_d` itself for the self-feeding graph loop).
    #[allow(clippy::too_many_arguments)]
    pub fn gemma4_decode_step_dc_into(&self, e: &Engine, token_d: &CudaSlice<u32>,
                                      pos_d: &mut CudaSlice<i32>, embd_gpu: &CudaSlice<u8>,
                                      embd_qt: i32, embd_rb: usize, cache: &mut Cache,
                                      n_vocab: usize, cap_bucket_max: Option<(usize, usize)>,
                                      tok_out: &mut CudaSlice<u32>)
                                      -> Result<(), Box<dyn std::error::Error>> {
        let n_embd = self.cfg.n_embd as usize;
        let eps = self.cfg.rms_eps;
        let mut x = e.embed_gather_device(embd_gpu, token_d, n_embd, embd_qt, embd_rb)?;
        e.scale_inplace(&mut x, (n_embd as f32).sqrt(), n_embd)?;
        let mut h_carry: Option<(CudaSlice<i8>, CudaSlice<f32>)> = None;
        let n_layers = self.layers.len();
        for (il, layer) in self.layers.iter().enumerate() {
            let (hq, hdq) = match h_carry.take() {
                Some(p) => p,
                None => e.rms_norm_q8_1(&x, self.layers[0].attn_norm.float_data(), n_embd, 1, eps)?,
            };
            let Mixer::Full(fa) = &layer.mixer else { panic!("gemma4 layer {il} not full-attn") };
            let o = self.gemma4_decode_attn_dc(e, fa, il, &hq, &hdq, pos_d, cache, cap_bucket_max)?;
            let mut cur = e.uninit(n_embd)?;
            e.rms_norm(&o, layer.post_attn_norm.float_data(), &mut cur, n_embd, 1, eps)?;
            let next_norm = if il + 1 < n_layers {
                Some(self.layers[il + 1].attn_norm.float_data())
            } else { None };
            let (xn, hn) = self.gemma4_layer_tail_add_nq(e, layer, &cur, &x, 1, next_norm)?;
            x = xn;
            h_carry = hn;
        }
        let mut hn = e.uninit(n_embd)?;
        e.rms_norm(&x, self.output_norm.float_data(), &mut hn, n_embd, 1, eps)?;
        let mut logits = e.matmul(&self.output, &hn, 1)?;
        self.gemma4_suppress(e, &mut logits, 1)?;   // cap skipped (monotonic); the mask is not
        e.argmax_token_device_into(&logits, tok_out, n_vocab)?;
        e.inc_seqlen(pos_d)?;
        if cap_bucket_max.is_none() { cache.pos += 1; }
        Ok(())
    }

    /// Persistent transient slots for the ALLOC-FREE captured dc step (the graph door):
    /// every buffer the step produces per token lives here, allocated ONCE pre-capture, so
    /// the captured graph carries zero cuMemAllocAsync/Free nodes (the 226us/launch tax,
    /// osrt 2026-07-23). Sized for the model's max per-layer shapes.

    /// Build the slot set (call OUTSIDE any capture).
    pub fn g4_dc_slots(&self, e: &Engine) -> Result<G4DcSlots, Box<dyn std::error::Error>> {
        let n_embd = self.cfg.n_embd as usize;
        let n_vocab = self.output.out_features();
        let n_layers = self.layers.len();
        let (mut qmax, mut kvmax, mut ffmax) = (0usize, 0usize, 0usize);
        for il in 0..n_layers {
            let (hd, nkv, nh, _b, _s, _w) = self.gemma4_geom(il);
            qmax = qmax.max(nh * hd);
            kvmax = kvmax.max(nkv * hd);
            if let crate::hybrid::Ffn::Dense { ffn_gate, .. } = &self.layers[il].ffn {
                ffmax = ffmax.max(ffn_gate.out_features());
            }
        }
        Ok(G4DcSlots {
            x: e.uninit(n_embd)?, xn: e.uninit(n_embd)?, cur: e.uninit(n_embd)?,
            hq: e.alloc_i8_uninit(n_embd)?, hd_: e.uninit(n_embd / 32)?,
            q0: e.uninit(qmax)?, k0: e.uninit(kvmax)?, v0: e.uninit(kvmax)?,
            q: e.uninit(qmax)?, k: e.uninit(kvmax)?, v: e.uninit(kvmax)?,
            attn: e.uninit(qmax)?, o: e.uninit(n_embd)?,
            attn_out: e.uninit(n_embd)?, zsh: e.uninit(n_embd)?,
            // zq/zd feed the wo matvec (nh*hd rows — 4096 on hd512 globals > n_embd),
            // the ffn entry (n_embd) and the lm_head (n_embd): size for the max.
            zq: e.alloc_i8_uninit(n_embd.max(qmax))?, zd: e.uninit(n_embd.max(qmax) / 32)?,
            gate: e.uninit(ffmax)?, up: e.uninit(ffmax)?,
            act: e.uninit(ffmax)?, actq: e.alloc_i8_uninit(ffmax)?, actd: e.uninit(ffmax / 32)?,
            f0: e.uninit(n_embd)?, sn: e.uninit(n_embd)?,
            hn: e.uninit(n_embd)?, logits: e.uninit(n_vocab)?,
        })
    }

    /// m=1 pre-quantized matvec into a slot — mirrors matmul_pre's m=1 mmvq route exactly
    /// (rp4-mirror bytes; mmvq_supports guaranteed for gemma4 q4_0/q6_K).
    fn g4_matvec_m1_into(&self, e: &Engine, w: &crate::model::GpuTensor,
                         aq: &CudaSlice<i8>, ad: &CudaSlice<f32>, y: &mut CudaSlice<f32>)
                         -> Result<(), Box<dyn std::error::Error>> {
        use crate::model::GpuTensor;
        let (bytes, qtype, row_bytes, scale, rp) = match w {
            GpuTensor::Quant { bytes, qtype, row_bytes, scale, rp, .. } =>
                (bytes, *qtype, *row_bytes, *scale, *rp),
            _ => return Err("g4_matvec_m1_into: non-quant tensor".into()),
        };
        let (mbytes, mrp) = match w {
            GpuTensor::Quant { rp4: Some(m4), .. } => (m4, true),
            _ => (bytes, rp),
        };
        e.qmatvec_mmvq_into(mbytes, aq, ad, 1, w.in_features(), w.out_features(),
                            qtype, row_bytes, scale, mrp, y)
    }

    /// ALLOC-FREE dc step (capture body): kernel-for-kernel mirror of
    /// `gemma4_decode_step_dc_into` at t=1 with every transient slot-fed. Dense gemma4 only
    /// (12B/31B; uniform q4_0 trunk guarantees the fused2/3 arms).
    #[allow(clippy::too_many_arguments)]
    pub fn gemma4_decode_step_dc_slotted(&self, e: &Engine, token_d: &CudaSlice<u32>,
                                         pos_d: &mut CudaSlice<i32>, embd_gpu: &CudaSlice<u8>,
                                         embd_qt: i32, embd_rb: usize, cache: &mut Cache,
                                         n_vocab: usize, cap_bucket_max: Option<(usize, usize)>,
                                         sl: &mut G4DcSlots, tok_out: &mut CudaSlice<u32>,
                                         ring: Option<(&mut CudaSlice<u32>, usize)>)
                                         -> Result<(), Box<dyn std::error::Error>> {
        let n_embd = self.cfg.n_embd as usize;
        let eps = self.cfg.rms_eps;
        e.embed_gather_device_into(embd_gpu, token_d, &mut sl.x, n_embd, embd_qt, embd_rb)?;
        e.scale_inplace(&mut sl.x, (n_embd as f32).sqrt(), n_embd)?;
        let n_layers = self.layers.len();
        let mut has_carry = false;
        for il in 0..n_layers {
            if !has_carry {
                e.rms_norm_q8_1_into(&sl.x, self.layers[il].attn_norm.float_data(), n_embd, 1,
                                     eps, &mut sl.hq, &mut sl.hd_)?;
            }
            has_carry = true;
            let layer = &self.layers[il];
            let Mixer::Full(fa) = &layer.mixer else { panic!("gemma4 layer {il} not full-attn") };
            self.gemma4_decode_attn_dc_slotted(e, fa, il, pos_d, cache, cap_bucket_max, sl)?;
            e.rms_norm(&sl.o, layer.post_attn_norm.float_data(), &mut sl.cur, n_embd, 1, eps)?;
            let next_norm = if il + 1 < n_layers {
                Some(self.layers[il + 1].attn_norm.float_data())
            } else { None };
            self.gemma4_layer_tail_slotted(e, layer, next_norm, sl)?;
            std::mem::swap(&mut sl.x, &mut sl.xn);
        }
        e.rms_norm(&sl.x, self.output_norm.float_data(), &mut sl.hn, n_embd, 1, eps)?;
        e.quantize_q8_1_into(&sl.hn, 1, n_embd, &mut sl.zq, &mut sl.zd)?;
        // lm_head via the same m=1 mmvq route as matmul (q6_K on the gemma4 family).
        {
            let (zq, zd) = (&sl.zq, &sl.zd);
            let zq = unsafe { &*(zq as *const CudaSlice<i8>) };
            let zd = unsafe { &*(zd as *const CudaSlice<f32>) };
            self.g4_matvec_m1_into(e, &self.output, zq, zd, &mut sl.logits)?;
        }
        self.gemma4_suppress(e, &mut sl.logits, 1)?;
        e.argmax_token_device_into(&sl.logits, tok_out, n_vocab)?;
        if let Some((ring, base)) = ring {
            // in-graph token parking: slot = (pos - base) % ring.len(); the door drains
            // with ONE sync per chunk instead of a per-token dtoh (the launch-serialization
            // fix — llama's 885us/launch overlaps GPU work because nothing waits per token).
            e.plain_tok_ring(tok_out, pos_d, base, ring)?;
        }
        e.inc_seqlen(pos_d)?;
        if cap_bucket_max.is_none() { cache.pos += 1; }
        Ok(())
    }

    /// Slot-fed dc attention: mirrors gemma4_decode_attn_dc's CAPTURE arm kernel-for-kernel.
    #[allow(clippy::too_many_arguments)]
    fn gemma4_decode_attn_dc_slotted(&self, e: &Engine, fa: &crate::hybrid::FullAttnLayer,
                                     il: usize, pos_d: &CudaSlice<i32>, cache: &mut Cache,
                                     cap_bucket_max: Option<(usize, usize)>, sl: &mut G4DcSlots)
                                     -> Result<(), Box<dyn std::error::Error>> {
        let (hd, nkv, nh, base, scale, swa) = self.gemma4_geom(il);
        let eps = self.cfg.rms_eps;
        let aux = self.gemma4_aux.as_ref().unwrap();
        {
            let hq = unsafe { &*(&sl.hq as *const CudaSlice<i8>) };
            let hdq = unsafe { &*(&sl.hd_ as *const CudaSlice<f32>) };
            if swa {
                if !e.matmul_q4_fused3_into(&fa.wq, &fa.wk, &fa.wv, hq, hdq,
                                            &mut sl.q0, &mut sl.k0, &mut sl.v0)? {
                    return Err("slotted step: fused3 unavailable (non-uniform trunk)".into());
                }
            } else {
                if !e.matmul_q4_fused2_into(&fa.wq, &fa.wk, hq, hdq, &mut sl.q0, &mut sl.k0)? {
                    return Err("slotted step: fused2 unavailable".into());
                }
                let k0r = unsafe { &*(&sl.k0 as *const CudaSlice<f32>) };
                e.copy_into(&mut sl.v0, 0, k0r, nkv * hd)?;
            }
        }
        // E4B wave-3 fold, m=1 completion (2026-07-23) — MUST mirror the dc_into arm
        // kernel-for-kernel (graph stream-identity gate).
        let ff = if swa { None } else {
            Some(aux.rope_freqs.as_ref().expect("gemma4 global rope needs rope_freqs.weight"))
        };
        let kvl = cache.kv[il].as_mut().unwrap();
        let kv_fp8 = (!swa && crate::Engine::gkv_on()) || (swa && crate::Engine::wkv_on());
        if crate::Engine::qkv_append_on() {
            // append fold (2026-07-23): mirrors dc_into.
            e.rms_norm_qkv_rope_append_dc(&sl.q0, &sl.k0, &sl.v0, fa.q_norm.float_data(),
                fa.k_norm.float_data(), &aux.ones, &mut sl.q, &mut sl.k, &mut sl.v, hd, nh, nkv,
                pos_d, nh, nkv, base, 1.0, ff, eps,
                &mut kvl.k, &mut kvl.v, &kvl.len_d, kvl.k_tok_bytes, kvl.v_tok_bytes, kv_fp8)?;
        } else {
            e.rms_norm_qkv_rope(&sl.q0, &sl.k0, &sl.v0, fa.q_norm.float_data(), fa.k_norm.float_data(),
                                &aux.ones, &mut sl.q, &mut sl.k, &mut sl.v, hd, nh, nkv,
                                pos_d, nh, nkv, base, 1.0, ff, eps)?;
            e.append_kv_quantized_dc(&sl.k, &sl.v, &mut kvl.k, &mut kvl.v, &kvl.len_d,
                                     kvl.kv_dim_k, kvl.kv_dim_v, kvl.k_tok_bytes, kvl.v_tok_bytes,
                                     kv_fp8)?;
        }
        e.inc_seqlen(&mut kvl.len_d)?;
        let (b_swa, b_glob) = cap_bucket_max.expect("slotted step is capture-only");
        let k_view = e.view_u8(&kvl.k, kvl.k.len());
        let v_view = e.view_u8(&kvl.v, kvl.v.len());
        let rows_on = std::env::var("MEMRA_GEMMA_ROWS_W").as_deref() != Ok("0");
        let win = self.cfg.gemma4.as_ref().unwrap().sliding_window as usize;
        // combine-q8 emit (wave-5b m=1 port): the rows arms quantize inside the combine —
        // the standalone quantize launch runs only on the non-rows fallback. MUST mirror
        // the dc_into arm branch-for-branch (stream gate).
        let mut fa_q8 = false;
        if !swa && hd == 512 && b_glob >= crate::fa512_min_tkv() && rows_on {
            e.fa_decode_rows(&sl.q, &k_view, &v_view, &mut sl.attn, hd, nh, nkv, b_glob - 1,
                             1, scale, kvl.k_tok_bytes, kvl.v_tok_bytes,
                             Some((&kvl.len_d, -1)), false, false,
                             Some((&mut sl.zq, &mut sl.zd)))?;
            fa_q8 = true;
        } else if swa && b_swa > win && hd == 256 && rows_on {
            e.fa_decode_rows_w(&sl.q, &k_view, &v_view, &mut sl.attn, hd, nh, nkv,
                               &kvl.len_d, -1, 1, scale, win,
                               kvl.k_tok_bytes, kvl.v_tok_bytes,
                               Some((&mut sl.zq, &mut sl.zd)))?;
            fa_q8 = true;
        } else {
            let b = if swa { b_swa } else { b_glob };
            e.fa_decode_dc(&sl.q, &k_view, &v_view, &mut sl.attn, hd, nh, nkv, &kvl.len_d, b,
                           scale, kvl.k_tok_bytes, kvl.v_tok_bytes,
                           swa && crate::Engine::wkv_on())?;
        }
        if !fa_q8 {
            let aq = unsafe { &*(&sl.attn as *const CudaSlice<f32>) };
            e.quantize_q8_1_into(aq, 1, nh * hd, &mut sl.zq, &mut sl.zd)?;
        }
        {
            let zq = unsafe { &*(&sl.zq as *const CudaSlice<i8>) };
            let zd = unsafe { &*(&sl.zd as *const CudaSlice<f32>) };
            self.g4_matvec_m1_into(e, &fa.wo, zq, zd, &mut sl.o)?;
        }
        Ok(())
    }

    /// Slot-fed dense layer tail: mirrors gemma4_layer_tail_core (t=1, no pre-norm fold) +
    /// tail_add_nq kernel-for-kernel; the next layer's (hq, hd_) carry lands in the slots.
    fn gemma4_layer_tail_slotted(&self, e: &Engine, layer: &crate::hybrid::HybridLayer,
                                 next_norm: Option<&CudaSlice<f32>>, sl: &mut G4DcSlots)
                                 -> Result<(), Box<dyn std::error::Error>> {
        let n_embd = self.cfg.n_embd as usize;
        let eps = self.cfg.rms_eps;
        let bits = layer.gemma4.as_ref().unwrap();
        let crate::hybrid::Ffn::Dense { ffn_gate, ffn_up, ffn_down } = &layer.ffn
        else { return Err("slotted tail: dense ffn only".into()) };
        e.add_rms_norm(&sl.cur, &sl.x, bits.ffn_norm.float_data(), &mut sl.attn_out,
                       &mut sl.zsh, n_embd, 1, eps)?;
        let n_ff = ffn_gate.out_features();
        {
            let zshr = unsafe { &*(&sl.zsh as *const CudaSlice<f32>) };
            e.quantize_q8_1_into(zshr, 1, n_embd, &mut sl.zq, &mut sl.zd)?;
        }
        {
            let zq = unsafe { &*(&sl.zq as *const CudaSlice<i8>) };
            let zd = unsafe { &*(&sl.zd as *const CudaSlice<f32>) };
            if !e.matmul_q4_fused2_into(ffn_gate, ffn_up, zq, zd, &mut sl.gate, &mut sl.up)? {
                return Err("slotted tail: ffn fused2 unavailable".into());
            }
        }
        debug_assert!(e.uses_q8_1_fast(ffn_down));
        {
            let upr = unsafe { &*(&sl.up as *const CudaSlice<f32>) };
            let upv = e.view(upr, n_ff);
            let up_all = upv.slice(0..n_ff);
            let gr = unsafe { &*(&sl.gate as *const CudaSlice<f32>) };
            e.gelu_tanh_mul_q8_1_into(gr, &up_all, &mut sl.act, n_ff, 1,
                                      &mut sl.actq, &mut sl.actd)?;
        }
        {
            let aq = unsafe { &*(&sl.actq as *const CudaSlice<i8>) };
            let ad = unsafe { &*(&sl.actd as *const CudaSlice<f32>) };
            self.g4_matvec_m1_into(e, ffn_down, aq, ad, &mut sl.f0)?;
        }
        e.rms_norm(&sl.f0, bits.post_ffw_norm.float_data(), &mut sl.sn, n_embd, 1, eps)?;
        match next_norm {
            Some(w) => {
                e.add_scale_rms_norm_q8_1_into(&sl.sn, &sl.attn_out, bits.layer_scale, w,
                                               &mut sl.xn, n_embd, 1, eps,
                                               &mut sl.hq, &mut sl.hd_)?;
            }
            None => {
                e.add_scale(&sl.sn, &sl.attn_out, bits.layer_scale, &mut sl.xn, n_embd)?;
            }
        }
        Ok(())
    }

    /// dc attention: same math as gemma4_decode_attn, KV slot/lengths from device counters.
    #[allow(clippy::too_many_arguments)]
    fn gemma4_decode_attn_dc(&self, e: &Engine, fa: &crate::hybrid::FullAttnLayer, il: usize,
                             hq: &CudaSlice<i8>, hdq: &CudaSlice<f32>,
                             pos_d: &CudaSlice<i32>, cache: &mut Cache,
                             cap_bucket_max: Option<(usize, usize)>)
                             -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let (hd, nkv, nh, base, scale, swa) = self.gemma4_geom(il);
        let eps = self.cfg.rms_eps;
        let aux = self.gemma4_aux.as_ref().unwrap();
        let (q0, k0, v0) = if swa {
            match e.matmul_q4_fused3(&fa.wq, &fa.wk, &fa.wv, hq, hdq)? {
                Some(t3) => t3,
                None => {
                    let h0 = e.zeros(0)?;
                    (e.matmul_pre(&fa.wq, hq, hdq, &h0, 1)?,
                     e.matmul_pre(&fa.wk, hq, hdq, &h0, 1)?,
                     e.matmul_pre(&fa.wv, hq, hdq, &h0, 1)?)
                }
            }
        } else {
            let (q0, k0) = match e.matmul_q4_fused2(&fa.wq, &fa.wk, hq, hdq)? {
                Some(p) => p,
                None => {
                    let h0 = e.zeros(0)?;
                    (e.matmul_pre(&fa.wq, hq, hdq, &h0, 1)?,
                     e.matmul_pre(&fa.wk, hq, hdq, &h0, 1)?)
                }
            };
            let v0 = e.clone_dtod(&k0)?;
            (q0, k0, v0)
        };
        let mut q = e.uninit(nh * hd)?;
        let mut k = e.uninit(nkv * hd)?;
        let mut v = e.uninit(nkv * hd)?;
        // E4B wave-3 fold, m=1 completion (2026-07-23): mirrored by the slotted arm.
        let ff = if swa { None } else {
            Some(aux.rope_freqs.as_ref().expect("gemma4 global rope needs rope_freqs.weight"))
        };
        let kvl = cache.kv[il].as_mut().unwrap();
        let kv_fp8 = (!swa && crate::Engine::gkv_on()) || (swa && crate::Engine::wkv_on());
        if crate::Engine::qkv_append_on() {
            // append fold (2026-07-23): norm+rope+cache-append in ONE launch.
            e.rms_norm_qkv_rope_append_dc(&q0, &k0, &v0, fa.q_norm.float_data(),
                fa.k_norm.float_data(), &aux.ones, &mut q, &mut k, &mut v, hd, nh, nkv,
                pos_d, nh, nkv, base, 1.0, ff, eps,
                &mut kvl.k, &mut kvl.v, &kvl.len_d, kvl.k_tok_bytes, kvl.v_tok_bytes, kv_fp8)?;
        } else {
            e.rms_norm_qkv_rope(&q0, &k0, &v0, fa.q_norm.float_data(), fa.k_norm.float_data(),
                                &aux.ones, &mut q, &mut k, &mut v, hd, nh, nkv,
                                pos_d, nh, nkv, base, 1.0, ff, eps)?;
            e.append_kv_quantized_dc(&k, &v, &mut kvl.k, &mut kvl.v, &kvl.len_d,
                                     kvl.kv_dim_k, kvl.kv_dim_v, kvl.k_tok_bytes, kvl.v_tok_bytes, kv_fp8)?;
        }
        e.inc_seqlen(&mut kvl.len_d)?;
        let mut attn = e.uninit(nh * hd)?;
        // combine-q8 emit carrier (wave-5b m=1 port): rows arms fill this; the tail then
        // rides g4_matvec_m1_into instead of matmul's internal quantize.
        let mut fa_q8: Option<(CudaSlice<i8>, CudaSlice<f32>)> = None;
        // Weight prefetch NOT wired here (26B/31B/qwen probes 2026-07-13: 26B flat
        // 196.6/196.0 vs 196.4/196.1 — MoE ffn dilutes wo; 31B −0.2% — dense decode sits
        // at the DRAM wall, no idle window to front-load into). E4B keeps the arm
        // (gemma4_e4b_attn, +0.65% valid window).
        match cap_bucket_max {
            None => {
                // dc-EAGER: host knows the length — R6 window views EXACTLY like the eager
                // decode (SWA layers attend the last `sliding_window` keys); the device
                // counters carry only the append slot + the graph seam.
                kvl.len += 1;
                let win = self.cfg.gemma4.as_ref().unwrap().sliding_window as usize;
                if !swa && hd == 512 && kvl.len >= crate::fa512_min_tkv()
                    && std::env::var("MEMRA_GEMMA_ROWS_W").as_deref() != Ok("0") {
                    // global layers: SAME hd512 rows twin as verify, t=1 (parity law).
                    // dc: len_d is live (inc_seqlen) — device-len rides it, plus=-1.
                    let kp = e.view_u8(&kvl.k, kvl.len * kvl.k_tok_bytes);
                    let vp = e.view_u8(&kvl.v, kvl.len * kvl.v_tok_bytes);
                    let (mut aq8, mut ad8) = e.uninit_q8_pair(nh * hd)?;
                    e.fa_decode_rows(&q, &kp, &vp, &mut attn, hd, nh, nkv, kvl.len - 1, 1,
                                     scale, kvl.k_tok_bytes, kvl.v_tok_bytes,
                                     Some((&kvl.len_d, -1)), false, false,
                                     Some((&mut aq8, &mut ad8)))?;
                    fa_q8 = Some((aq8, ad8));
                } else if swa && kvl.len > win && hd == 256
                    && std::env::var("MEMRA_GEMMA_ROWS_W").as_deref() != Ok("0") {
                    // windowed regime: SAME rows_w kernel as verify, t=1 (parity law).
                    let kp = e.view_u8(&kvl.k, kvl.len * kvl.k_tok_bytes);
                    let vp = e.view_u8(&kvl.v, kvl.len * kvl.v_tok_bytes);
                    let (mut aq8, mut ad8) = e.uninit_q8_pair(nh * hd)?;
                    e.fa_decode_rows_w(&q, &kp, &vp, &mut attn, hd, nh, nkv, &kvl.len_d, -1,
                                       1, scale, win, kvl.k_tok_bytes, kvl.v_tok_bytes,
                                       Some((&mut aq8, &mut ad8)))?;
                    fa_q8 = Some((aq8, ad8));
                } else {
                    let (off_tok, t_kv) = if swa && kvl.len > win { (kvl.len - win, win) }
                                          else { (0, kvl.len) };
                    let k_view = e.view_u8_range(&kvl.k, off_tok * kvl.k_tok_bytes,
                                                 (off_tok + t_kv) * kvl.k_tok_bytes);
                    let v_view = e.view_u8_range(&kvl.v, off_tok * kvl.v_tok_bytes,
                                                 (off_tok + t_kv) * kvl.v_tok_bytes);
                    e.fa_decode_kvmod(&q, &k_view, &v_view, &mut attn, hd, nh, nkv, t_kv, scale,
                                kvl.k_tok_bytes, kvl.v_tok_bytes, swa && crate::Engine::wkv_on())?;
                }
            }
            Some((b_swa, b_glob)) => {
                // capture: full-buffer views + device length. PARITY LAW port (graph arc step
                // 3): the SAME regime branches as dc-eager, pinned per arm — b_swa is the
                // exact-key t_kv for the dc family (n_splits constant per bucket), b_glob is
                // the RUNG max for the rows family (kernels derive per-replay splits from
                // kvl.len_d; grid/partials sized for the rung end, excess splits exit).
                let k_view = e.view_u8(&kvl.k, kvl.k.len());
                let v_view = e.view_u8(&kvl.v, kvl.v.len());
                let rows_on = std::env::var("MEMRA_GEMMA_ROWS_W").as_deref() != Ok("0");
                let win = self.cfg.gemma4.as_ref().unwrap().sliding_window as usize;
                if !swa && hd == 512 && b_glob >= crate::fa512_min_tkv() && rows_on {
                    let (mut aq8, mut ad8) = e.uninit_q8_pair(nh * hd)?;
                    e.fa_decode_rows(&q, &k_view, &v_view, &mut attn, hd, nh, nkv, b_glob - 1,
                                     1, scale, kvl.k_tok_bytes, kvl.v_tok_bytes,
                                     Some((&kvl.len_d, -1)), false, false,
                                     Some((&mut aq8, &mut ad8)))?;
                    fa_q8 = Some((aq8, ad8));
                } else if swa && b_swa > win && hd == 256 && rows_on {
                    let (mut aq8, mut ad8) = e.uninit_q8_pair(nh * hd)?;
                    e.fa_decode_rows_w(&q, &k_view, &v_view, &mut attn, hd, nh, nkv,
                                       &kvl.len_d, -1, 1, scale, win,
                                       kvl.k_tok_bytes, kvl.v_tok_bytes,
                                       Some((&mut aq8, &mut ad8)))?;
                    fa_q8 = Some((aq8, ad8));
                } else {
                    let b = if swa { b_swa } else { b_glob };
                    e.fa_decode_dc(&q, &k_view, &v_view, &mut attn, hd, nh, nkv, &kvl.len_d, b,
                                   scale, kvl.k_tok_bytes, kvl.v_tok_bytes,
                                   swa && crate::Engine::wkv_on())?;
                }
            }
        }
        // combine-q8 emit (wave-5b m=1 port): the rows arms produced the wo activation pair
        // in-combine — ride the slotted arm's exact matvec route (parity by construction).
        if let Some((aq8, ad8)) = fa_q8 {
            let mut y = e.uninit(fa.wo.out_features())?;
            self.g4_matvec_m1_into(e, &fa.wo, &aq8, &ad8, &mut y)?;
            return Ok(y);
        }
        Ok(e.matmul(&fa.wo, &attn, 1)?)
    }

    /// gemma4 GRAPH-REPLAY greedy loop: per (swa-key, global-key) fa bucket, capture ONE full
    /// dc step (self-feeding: argmax writes token_d in-graph) and replay it — one graph launch
    /// per token, one 4B dtoh. V1 scope: whole generation under the sliding window (no window
    /// views in-graph); caller gates and falls back to the dc-eager loop.
    pub fn gemma4_generate_graph(&self, e: &Engine, prompt_pos: usize, first_token: u32,
                                 cache: &mut Cache, max_new: usize, eos: &[u32],
                                 mut on_token: impl FnMut(u32) -> bool)
                                 -> Result<(Vec<u32>, crate::decode::StopReason), Box<dyn std::error::Error>> {
        if self.is_gemma4_e4b() {
            return Err("E4B graph serving is unwired (HANDOVER-E4B.md) — dc-eager is the serving arm".into());
        }
        use crate::decode::StopReason;
        let n_vocab = self.output.out_features();
        let n_embd = self.cfg.n_embd as usize;
        let embd_gpu = self.embd_gpu.get_or_init(|| {
            e.upload_u8(&self.embd.raw).expect("embed table upload")
        });
        let (qt, rb) = self.embd.qt_and_row_bytes(n_embd);
        for kvl in cache.kv.iter_mut().flatten() {
            e.set_i32_one(&mut kvl.len_d, kvl.len as i32)?;
        }
        let mut token_d = e.stream().clone_htod(&[first_token])?;
        let mut pos_d = e.htod_i32(&[prompt_pos as i32])?;
        let g4 = self.cfg.gemma4.as_ref().unwrap();
        let (hd_s, hd_g) = (g4.key_length_swa as usize, g4.key_length_global as usize);
        // per-layer nkv: swa vs global counts from the pattern (uniform within class).
        let nkv_s = g4.head_count_kv.iter().zip(g4.swa_pattern.iter())
            .find(|p| *p.1).map(|p| *p.0 as usize).unwrap_or(8);
        let nkv_g = g4.head_count_kv.iter().zip(g4.swa_pattern.iter())
            .find(|p| !*p.1).map(|p| *p.0 as usize).unwrap_or(2);
        let mut graphs: std::collections::HashMap<((bool, usize), (bool, usize), bool, bool),
                                                  (cudarc::driver::CudaGraph,
                                                   Vec<Box<dyn std::any::Any + Send>>)> = Default::default();
        // ALLOC-FREE capture: persistent transient slots — the captured graph carries zero
        // mem nodes (the 226us/launch tax). Slots must outlive every cached graph.
        let mut slots = self.g4_dc_slots(e)?;
        // Chunked replay ring: tokens park on-device; ONE drain sync per chunk. ring_base is
        // baked at the door entry (the modulo keeps every capture valid indefinitely).
        const RING: usize = 64;
        // DRAIN pinned at 1 (2026-07-23): relaunching the SAME graph exec before its prior
        // launch completes is ILLEGAL (chunk=4 -> ILLEGAL_ADDRESS; chunk=1 clean). Pipelining
        // needs alternating exec instances — and the measured payoff was ~0 (the ~200us
        // cuGraphLaunch host cost already overlaps its own launch's GPU work; llama pays
        // 885us/launch the same way). MEMRA_GRAPH_DRAIN raises it only for experiments.
        const DRAIN: usize = 1;
        let mut ring = e.stream().alloc_zeros::<u32>(RING)?;
        let ring_base = prompt_pos;
        let mut out = Vec::with_capacity(max_new);
        let mut reason = StopReason::MaxNew;
        let mut next = first_token;
        let mut captures = 0usize;
        for _ in 0..max_new {
            out.push(next);
            if eos.contains(&next) { reason = StopReason::Eos; break; }
            if !on_token(next) { reason = StopReason::Callback; break; }
            let t_kv = cache.pos + 1;
            // Bucket key per ARM (graph arc step 3):
            //  - swa component: dc family under the window (exact (fa_vec, n_splits) key —
            //    n_splits must be constant per bucket); rows_w above it (window-constant, so
            //    the component collapses to a single marker).
            //  - global component: dc family under the fa512 floor (exact key); rows_dpl16
            //    at/above it — the kernel derives splits from len_d per replay, so buckets
            //    are power-of-2 RUNGS (one capture per doubling; grid sized for the rung end).
            let win = self.cfg.gemma4.as_ref().unwrap().sliding_window as usize;
            let f512 = crate::fa512_min_tkv();
            let key_s = if t_kv > win { (true, usize::MAX) }
                        else { e.fa_bucket_key(t_kv, hd_s, nkv_s, crate::Engine::wkv_on()) };
            let (key_g, rung_end) = if t_kv >= f512 {
                // strict upper bound: the rung must ROLL at exact powers (t_kv==1024 starts
                // the [1024,2048) bucket) — sizing covers every replayed T_kv < end.
                let end = (t_kv + 1).next_power_of_two().max(f512 * 2);
                ((true, end), end)
            } else { (e.fa_bucket_key(t_kv, hd_g, nkv_g, false), t_kv) };
            let key = (key_s, key_g, t_kv >= f512, t_kv > win);
            if !graphs.contains_key(&key) {
                let bucket_max = (t_kv, rung_end);
                // snapshot device+host state (the 3 capture-warmup runs leave no residue).
                let snap = cache.snapshot(e)?;
                let pos_save = e.dtoh_i32_one(&pos_d)?;
                let len_save: Vec<Option<i32>> = cache.kv.iter()
                    .map(|k| k.as_ref().map(|kvl| e.dtoh_i32_one(&kvl.len_d).unwrap())).collect();
                let tok_save = e.dtoh_u32_one(&token_d)?;
                // RETAINED capture (2026-07-23): the plain capture_graph left pool-transient
                // clones as dead COPY NODES replayed every launch — the E4B 0.74ms/token
                // regression class, and this door's measured -8.8%. The keeper pins warmup
                // transients so the captured graph holds kernel nodes only.
                let graph = {
                    let tok_ref = &mut token_d;
                    let pos_ref = &mut pos_d;
                    let cache_ref = &mut *cache;
                    let slots_ref = &mut slots;
                    let ring_ref = &mut ring;
                    e.capture_graph_retained_flags(
                        cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY,
                        |e| {
                        // self-feeding: the argmax writes token_d itself.
                        let tok_in = unsafe { &*(tok_ref as *const CudaSlice<u32>) };
                        let sl = unsafe { &mut *(slots_ref as *mut G4DcSlots) };
                        let rg = unsafe { &mut *(ring_ref as *mut CudaSlice<u32>) };
                        self.gemma4_decode_step_dc_slotted(e, tok_in, pos_ref, embd_gpu, qt, rb,
                                                           cache_ref, n_vocab, Some(bucket_max),
                                                           sl, tok_ref, Some((rg, ring_base)))
                    })?
                };
                cache.rollback(e, &snap, 0)?;
                e.set_i32_one(&mut pos_d, pos_save)?;
                for (il, ls) in len_save.iter().enumerate() {
                    if let (Some(kvl), Some(v)) = (cache.kv[il].as_mut(), ls) {
                        e.set_i32_one(&mut kvl.len_d, *v)?;
                    }
                }
                e.set_u32_one(&mut token_d, tok_save)?;
                if std::env::var("MEMRA_GRAPH_CENSUS").as_deref() == Ok("1") {
                    if let Ok(c) = crate::graph_update::node_census(&graph.0) {
                        eprintln!("[graph-census] {c:?}");
                    }
                }
                graphs.insert(key, graph);
                captures += 1;
            }
            // CHUNKED REPLAY: enqueue up to DRAIN launches back-to-back (the ~200us host
            // cost of each cuGraphLaunch overlaps the PREVIOUS launch's ~11ms GPU work),
            // then ONE sync + ring drain. The chunk must stay inside this bucket key and
            // the budget; capture warmups already emitted their tokens through the ring.
            let mut chunk = 1usize;
            let drain_cap: usize = std::env::var("MEMRA_GRAPH_DRAIN").ok()
                .and_then(|v| v.parse().ok()).unwrap_or(DRAIN);
            while chunk < drain_cap && out.len() + chunk < max_new {
                let t_next = cache.pos + 1 + chunk;
                let key_s2 = if t_next > win { (true, usize::MAX) }
                             else { e.fa_bucket_key(t_next, hd_s, nkv_s, crate::Engine::wkv_on()) };
                let key_g2 = if t_next >= f512 {
                    (true, (t_next + 1).next_power_of_two().max(f512 * 2))
                } else { e.fa_bucket_key(t_next, hd_g, nkv_g, false) };
                if (key_s2, key_g2, t_next >= f512, t_next > win) != key { break; }
                chunk += 1;
            }
            let g = &graphs.get(&key).unwrap().0;
            for _ in 0..chunk { g.launch()?; }
            e.stream().synchronize()?;
            let ringh = e.dtoh_u32(&ring)?;
            for j in 0..chunk {
                let pos_j = cache.pos + j;
                let tok_j = ringh[(pos_j - ring_base) % RING];
                cache.pos += 0; // advanced below in one shot
                if j + 1 == chunk { next = tok_j; }
                else {
                    out.push(tok_j);
                    if eos.contains(&tok_j) || !on_token(tok_j) {
                        reason = if eos.contains(&tok_j) { StopReason::Eos }
                                 else { StopReason::Callback };
                        // roll device/host state back to the stop point.
                        let keep = cache.pos + j + 1;
                        e.set_i32_one(&mut pos_d, keep as i32)?;
                        for kvl in cache.kv.iter_mut().filter_map(|k| k.as_mut()) {
                            e.set_i32_one(&mut kvl.len_d, keep as i32)?;
                            kvl.len = keep;
                        }
                        cache.pos = keep;
                        if std::env::var("MEMRA_GRAPH_STATS").is_ok() {
                            eprintln!("[gemma-graph] captures={captures} buckets={}", graphs.len());
                        }
                        return Ok((out, reason));
                    }
                }
            }
            cache.pos += chunk;
            for kvl in cache.kv.iter_mut().filter_map(|k| k.as_mut()) { kvl.len += chunk; }
        }
        if std::env::var("MEMRA_GRAPH_STATS").is_ok() {
            eprintln!("[gemma-graph] captures={captures} buckets={}", graphs.len());
        }
        Ok((out, reason))
    }

    /// gemma4 VERIFY step (spec decode): t tokens batched through the trunk at positions
    /// pos0..pos0+t-1, K/V rows appended to the quantized cache (caller rolls back rejected
    /// rows via kvl.len), per-token causal windowed attend (fa_decode per token — the same
    /// kernel family as T=1 decode at each token's t_kv). Returns [t, n_vocab] softcapped
    /// logits (host) + advances cache.pos by t.
    pub(crate) fn gemma4_decode_step_t(&self, e: &Engine, tokens: &[u32], pos0: usize,
                                       cache: &mut Cache)
                                       -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        Ok(self.gemma4_decode_step_t_h(e, tokens, pos0, cache)?.0)
    }

    /// GREEDY verify: per-row DEVICE argmax (t x 4B host traffic instead of the t x 1MB logits
    /// stack — softcap skipped: tanh is monotonic, per-row argmax unaffected). Returns
    /// (argmax ids [t], post-output_norm hidden stack [t, n_embd]).
    pub(crate) fn gemma4_decode_step_t_am(&self, e: &Engine, tokens: &[u32], pos0: usize,
                                          cache: &mut Cache)
                                          -> Result<(Vec<u32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let (ld, hn) = self.gemma4_verify_trunk(e, tokens, pos0, cache, None)?;
        let t = tokens.len();
        let n_vocab = self.output.out_features();
        let mut toks = e.stream().alloc_zeros::<u32>(t)?;
        for i in 0..t {
            e.argmax_token_device_col(&ld, i, n_vocab, &mut toks, i)?;
        }
        Ok((e.dtoh_u32(&toks)?, hn))
    }

    /// Device-token verify (async spec round): tokens live in tok_d[0..t]; per-row argmax
    /// lands in a DEVICE buffer (no host logits). Returns (vam_d [t] u32 device, hn stack).
    pub(crate) fn gemma4_decode_step_t_am_dev(&self, e: &Engine, tok_d: &CudaSlice<u32>, t: usize,
                                              pos0: usize, cache: &mut Cache)
                                              -> Result<(CudaSlice<u32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let (ld, hn) = self.gemma4_verify_trunk(e, &vec![0u32; t], pos0, cache, Some(tok_d))?;
        let n_vocab = self.output.out_features();
        let mut vam = e.stream().alloc_zeros::<u32>(t)?;
        for i in 0..t {
            e.argmax_token_device_col(&ld, i, n_vocab, &mut vam, i)?;
        }
        Ok((vam, hn))
    }

    /// gemma4 verify + the POST-output_norm hidden stack [t, n_embd] (the drafter's h input —
    /// llama's h_nextn convention).
    pub(crate) fn gemma4_decode_step_t_h(&self, e: &Engine, tokens: &[u32], pos0: usize,
                                         cache: &mut Cache)
                                         -> Result<(Vec<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let (mut ld, hn) = self.gemma4_verify_trunk(e, tokens, pos0, cache, None)?;
        let t = tokens.len();
        let cap = self.cfg.gemma4.as_ref().unwrap().final_logit_softcapping;
        e.softcap(&mut ld, cap, t * self.output.out_features())?;
        Ok((e.dtoh(&ld)?, hn))
    }

    /// Persistent device scratch for the BURST verify stream (one per generation): rope pos
    /// rows [cap] + per-row t_kv counters (device-filled from the round counter, zero H2D).
    pub(crate) fn verify_stream_scratch(&self, e: &Engine, cap: usize)
                                        -> Result<VerifyStreamScratch, Box<dyn std::error::Error>> {
        Ok(VerifyStreamScratch {
            pos_d: e.htod_i32(&vec![0i32; cap])?,
            row_ctrs: (0..cap).map(|_| e.htod_i32(&[0])).collect::<Result<_, _>>()?,
        })
    }

    /// BURST verify trunk (device-slot twin): tokens from a device buffer, rope positions
    /// iota'd from `ctr`, appends/attention at the layers' len_d counters (verify_attn_stream),
    /// NO host cache.pos/len advance. Returns (per-row device argmaxes [t], post-norm hidden
    /// stack [t, n_embd]) — the burst's accept/seed inputs, zero host readbacks.
    /// `scr` = PERSISTENT per-gen scratch (pos rows + per-row t_kv counters): the first cut
    /// htod_i32-allocated these per call — 9 pageable H2D copies per round, each a stream
    /// sync, exactly the turnaround the burst exists to remove.
    pub(crate) fn gemma4_verify_t_am_stream(&self, e: &Engine, tok_d: &CudaSlice<u32>, t: usize,
                                            ctr: &CudaSlice<i32>, hint: usize,
                                            cache: &mut Cache,
                                            scr: &mut VerifyStreamScratch)
                                            -> Result<(CudaSlice<u32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let n_embd = self.cfg.n_embd as usize;
        let eps = self.cfg.rms_eps;
        assert!(t <= scr.row_ctrs.len() && t <= 64);
        e.i32_iota_from(ctr, &mut scr.pos_d, t)?;
        for i in 0..t {
            e.i32_copy_add(ctr, &mut scr.row_ctrs[i], (i + 1) as i32)?;
        }
        let (pos_d, row_ctrs) = (&scr.pos_d, &scr.row_ctrs);
        let embd_gpu = self.embd_gpu.get_or_init(|| {
            e.upload_u8(&self.embd.raw).expect("embed table upload")
        });
        let (qt, rb) = self.embd.qt_and_row_bytes(n_embd);
        let mut x = e.embed_gather_device_td(embd_gpu, tok_d, t, n_embd, qt, rb)?;
        e.scale_inplace(&mut x, (n_embd as f32).sqrt(), t * n_embd)?;
        let mut h_carry: Option<(CudaSlice<i8>, CudaSlice<f32>)> = None;
        let n_layers = self.layers.len();
        for (il, layer) in self.layers.iter().enumerate() {
            let (hq, hdq) = match h_carry.take() {
                Some(p) => p,
                None => e.rms_norm_q8_1(&x, self.layers[0].attn_norm.float_data(), n_embd, t, eps)?,
            };
            let Mixer::Full(fa) = &layer.mixer else { panic!("gemma4 layer {il} not full-attn") };
            let o = self.gemma4_verify_attn_stream(e, fa, il, &hq, &hdq, pos_d, t, cache,
                                                    hint, row_ctrs)?;
            let mut cur = e.uninit(t * n_embd)?;
            e.rms_norm(&o, layer.post_attn_norm.float_data(), &mut cur, n_embd, t, eps)?;
            let next_norm = if il + 1 < n_layers {
                Some(self.layers[il + 1].attn_norm.float_data())
            } else { None };
            let (xn, hn) = self.gemma4_layer_tail_add_nq(e, layer, &cur, &x, t, next_norm)?;
            x = xn;
            h_carry = hn;
            self.dflash_tap(e, cache, il, &x, t)?;
        }
        let mut hn = e.uninit(t * n_embd)?;
        e.rms_norm(&x, self.output_norm.float_data(), &mut hn, n_embd, t, eps)?;
        let ld = e.matmul(&self.output, &hn, t)?;
        let n_vocab = self.output.out_features();
        let mut vam = e.stream().alloc_zeros::<u32>(t)?;
        for i in 0..t {
            e.argmax_token_device_col(&ld, i, n_vocab, &mut vam, i)?;
        }
        Ok((vam, hn))
    }

    /// Verify trunk core: returns (UN-softcapped logits device [t, n_vocab], post-output_norm
    /// hidden stack [t, n_embd]); appends KV rows + advances cache.pos.
    /// DFlash tap write (dflash lane): copy the post-layer residual rows of `x` into the
    /// armed sink at the tap slot for `il` (row-major [t, n_taps*hidden]). Per-row D2D
    /// copies — t <= block_size on verify; prime pays t*n_taps once per prompt (dedicated
    /// kernel later if it shows in the profile).
    fn dflash_tap(&self, e: &Engine, cache: &mut Cache, il: usize, x: &CudaSlice<f32>, t: usize)
                  -> Result<(), Box<dyn std::error::Error>> {
        let Some(taps) = cache.dflash_taps.as_mut() else { return Ok(()) };
        let Some(slot) = taps.layer_ids.iter().position(|&l| l == il) else { return Ok(()) };
        let h = taps.hidden;
        let n_taps = taps.layer_ids.len();
        debug_assert_eq!(taps.t, t);
        let xv = e.view(x, t * h);
        for r in 0..t {
            let row = xv.slice(r * h..(r + 1) * h);
            e.copy_view_into(&mut taps.buf, r * n_taps * h + slot * h, &row, h)?;
        }
        Ok(())
    }

    fn gemma4_verify_trunk(&self, e: &Engine, tokens: &[u32], pos0: usize, cache: &mut Cache,
                           tok_dev: Option<&CudaSlice<u32>>)
                           -> Result<(CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let n_embd = self.cfg.n_embd as usize;
        let eps = self.cfg.rms_eps;
        let t = tokens.len();
        let pos: Vec<i32> = (0..t).map(|i| (pos0 + i) as i32).collect();
        let pos_d = e.htod_i32(&pos)?;
        let mut x = match tok_dev {
            Some(td) => {
                let embd_gpu = self.embd_gpu.get_or_init(|| {
                    e.upload_u8(&self.embd.raw).expect("embed table upload")
                });
                let (qt, rb) = self.embd.qt_and_row_bytes(n_embd);
                e.embed_gather_device_td(embd_gpu, td, t, n_embd, qt, rb)?
            }
            None => e.htod(&self.embd.gather(n_embd, tokens))?,
        };
        e.scale_inplace(&mut x, (n_embd as f32).sqrt(), t * n_embd)?;
        let mut h_carry: Option<(CudaSlice<i8>, CudaSlice<f32>)> = None;
        let n_layers = self.layers.len();
        for (il, layer) in self.layers.iter().enumerate() {
            let (hq, hdq) = match h_carry.take() {
                Some(p) => p,
                None => e.rms_norm_q8_1(&x, self.layers[0].attn_norm.float_data(), n_embd, t, eps)?,
            };
            let Mixer::Full(fa) = &layer.mixer else { panic!("gemma4 layer {il} not full-attn") };
            let o = self.gemma4_verify_attn(e, fa, il, &hq, &hdq, &pos_d, t, cache)?;
            let mut cur = e.uninit(t * n_embd)?;
            e.rms_norm(&o, layer.post_attn_norm.float_data(), &mut cur, n_embd, t, eps)?;
            let next_norm = if il + 1 < n_layers {
                Some(self.layers[il + 1].attn_norm.float_data())
            } else { None };
            let (xn, hn) = self.gemma4_layer_tail_add_nq(e, layer, &cur, &x, t, next_norm)?;
            x = xn;
            h_carry = hn;
            self.dflash_tap(e, cache, il, &x, t)?;
        }
        let mut hn = e.uninit(t * n_embd)?;
        e.rms_norm(&x, self.output_norm.float_data(), &mut hn, n_embd, t, eps)?;
        let mut ld = e.matmul(&self.output, &hn, t)?;
        self.gemma4_suppress(e, &mut ld, t)?;   // before the per-row argmax consumers
        cache.pos += t;
        Ok((ld, hn))
    }

    /// Verify attention: project/norm/rope t rows, append them to the cache, then attend each
    /// row causally over [win_off_i .. base+i] via the SAME fa_decode dispatch as T=1 decode.
    /// BURST verify attention (device-slot twin of `gemma4_verify_attn`): the append lands
    /// at the layer's len_d counter (rows_dc), attention bases ride the counter (rows /
    /// rows_w with base_dev), and NO host len is read or bumped — `hint` is a host UPPER
    /// bound on base_len used only for split sizing and arm gating (the burst loop passes
    /// pos0 + burst slack). Host len mirrors re-sync at the burst drain.
    #[allow(clippy::too_many_arguments)]
    fn gemma4_verify_attn_stream(&self, e: &Engine, fa: &crate::hybrid::FullAttnLayer, il: usize,
                                 hq: &CudaSlice<i8>, hdq: &CudaSlice<f32>,
                                 pos_d: &CudaSlice<i32>, t: usize,
                                 cache: &mut Cache, hint: usize,
                                 row_ctrs: &[CudaSlice<i32>])
                                 -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let (hd, nkv, nh, base, scale, swa) = self.gemma4_geom(il);
        let eps = self.cfg.rms_eps;
        let aux = self.gemma4_aux.as_ref().unwrap();
        let h0 = e.zeros(0)?;
        let h = &h0;
        // BATCHED FUSED qkv (MEMRA_F2B=1, megakernel microcosm): swa layers fuse all three,
        // globals fuse q,k (v := k clone). Bit-identical per row; segments tail-fill.
        static F2B_QKV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let f2b = *F2B_QKV.get_or_init(|| std::env::var("MEMRA_F2B").as_deref() != Ok("0"));
        let fused_qkv = if f2b {
            if swa {
                e.matmul_q4_fused3_batched(&fa.wq, &fa.wk, &fa.wv, hq, hdq, t)?
                    .map(|(a, b, c)| (a, b, Some(c)))
            } else {
                e.matmul_q4_fused2_batched(&fa.wq, &fa.wk, hq, hdq, t)?
                    .map(|(a, b)| (a, b, None))
            }
        } else { None };
        let (q0, k0, v0) = match fused_qkv {
            Some((a, b, cv)) => {
                let v = match cv { Some(c) => c, None => e.clone_dtod(&b)? };
                (a, b, v)
            }
            None => {
                let q0 = e.matmul_pre(&fa.wq, hq, hdq, h, t)?;
                let k0 = e.matmul_pre(&fa.wk, hq, hdq, h, t)?;
                let v0 = if swa { e.matmul_pre(&fa.wv, hq, hdq, h, t)? }
                         else { e.clone_dtod(&k0)? };
                (q0, k0, v0)
            }
        };
        let mut q = e.uninit(t * nh * hd)?;
        let mut k = e.uninit(t * nkv * hd)?;
        let mut v = e.uninit(t * nkv * hd)?;
        // E4B wave-3 fold (2026-07-23 decode-dust port): q/k/v norms + q/k rope in ONE
        // launch — rope math verbatim on the normed rows; V ones-rms, never roped.
        let ff = if swa { None } else {
            Some(aux.rope_freqs.as_ref().expect("gemma4 global rope needs rope_freqs.weight"))
        };
        e.rms_norm_qkv_rope(&q0, &k0, &v0, fa.q_norm.float_data(), fa.k_norm.float_data(),
                            &aux.ones, &mut q, &mut k, &mut v, hd, nh * t, nkv * t,
                            pos_d, nh, nkv, base, 1.0, ff, eps)?;
        let kvl = cache.kv[il].as_mut().unwrap();
        // append at the DEVICE slot; the counter advances by t on-device.
        e.append_kv_quantized_rows_dc(&k, &v, &mut kvl.k, &mut kvl.v, &kvl.len_d, t,
                                      kvl.kv_dim_k, kvl.kv_dim_v,
                                      kvl.k_tok_bytes, kvl.v_tok_bytes,
                                      (!swa && crate::Engine::gkv_on())
                                          || (swa && crate::Engine::wkv_on()))?;
        // len_d is NOT advanced here: the burst's device rollback (spec_rollback_stream) is
        // the sole len writer after this round's attention (base stays = old len, plus = 0).
        let win = self.cfg.gemma4.as_ref().unwrap().sliding_window as usize;
        let mut attn = e.uninit(t * nh * hd)?;
        let k_view = e.view_u8(&kvl.k, kvl.k.len());
        let v_view = e.view_u8(&kvl.v, kvl.v.len());
        // arm gating on the host HINT (upper bound; burst entry guards hint >= the vec floor
        // and a stable window regime — the same rung/regime keys as the draft graph).
        if swa && hint + 1 >= win {
            // fully-windowed rows: per-row window geometry from the counter (plus = 0: len_d
            // still holds the pre-append len; row r's T_kv = ctr + r + 1).
            e.fa_decode_rows_w(&q, &k_view, &v_view, &mut attn, hd, nh, nkv,
                               &kvl.len_d, 0, t, scale, win,
                               kvl.k_tok_bytes, kvl.v_tok_bytes, None)?;
        } else if hd == 512 && hint + t < crate::fa512_min_tkv() {
            // globals UNDER the fa512 crossover: eager runs the per-row fa_decode_kvmod
            // fallback there (parity with t=1 decode) — mirror it with per-row fa_decode_dc
            // (bit-correct for any t_kv <= bucket; the drafter dc arc proved the pairing).
            // Burst entry gates the horizon onto one side of the crossover, so hint decides
            // for every row.
            // NO .max(512): fa_decode_dc gates its hd512 dpl16-vs-scalar pick on bucket_max
            // (mirroring eager's fa512 floor) — forcing 512 here flipped the arm to dpl16
            // while eager ran scalar (il=5 KV drift, the burst's 4/128). SAME LAW capped
            // from above (2026-07-13): a regime-pinned hint near the floor (round-graph
            // captures use f512-1) pow2-rounds PAST it — clamp under the floor, or the arm
            // re-flips to dpl16 (the round-graph 4/64). The scalar unified self-splits, so
            // any bucket >= the live length is exact.
            let bucket = (hint + t + 2).next_power_of_two()
                .min(crate::fa512_min_tkv().saturating_sub(1));
            let qv = e.view(&q, t * nh * hd);
            for i in 0..t {
                let q_row = qv.slice(i * nh * hd..(i + 1) * nh * hd);
                let mut q_one = e.uninit(nh * hd)?;
                e.copy_view_into(&mut q_one, 0, &q_row, nh * hd)?;
                let mut a_one = e.uninit(nh * hd)?;
                e.fa_decode_dc(&q_one, &k_view, &v_view, &mut a_one, hd, nh, nkv,
                               &row_ctrs[i], bucket, scale,
                               kvl.k_tok_bytes, kvl.v_tok_bytes, false)?;
                e.copy_into(&mut attn, i * nh * hd, &a_one, nh * hd)?;
            }
        } else if hd == 512 {
            // globals past the crossover: dpl16 dc twin via the shared rows wrapper (hint
            // sizes splits — upper bound; splits beyond the device len exit in-kernel).
            e.fa_decode_rows(&q, &k_view, &v_view, &mut attn, hd, nh, nkv, hint, t, scale,
                             kvl.k_tok_bytes, kvl.v_tok_bytes,
                             Some((&kvl.len_d, 0)), false, false, None)?;
        } else {
            // hd256 under-window: v4 device-len rows twin.
            e.fa_decode_rows_dc(&q, &k_view, &v_view, &mut attn, hd, nh, nkv,
                                &kvl.len_d, hint + t, t, scale,
                                kvl.k_tok_bytes, kvl.v_tok_bytes, 0,
                                swa && crate::Engine::wkv_on())?;
        }
        Ok(e.matmul(&fa.wo, &attn, t)?)
    }

    fn gemma4_verify_attn(&self, e: &Engine, fa: &crate::hybrid::FullAttnLayer, il: usize,
                          hq: &CudaSlice<i8>, hdq: &CudaSlice<f32>,
                          pos_d: &CudaSlice<i32>, t: usize,
                          cache: &mut Cache)
                          -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let (hd, nkv, nh, base, scale, swa) = self.gemma4_geom(il);
        let eps = self.cfg.rms_eps;
        let aux = self.gemma4_aux.as_ref().unwrap();
        let n_embd = self.cfg.n_embd as usize;
        let _ = n_embd;

        let h0 = e.zeros(0)?;
        let h = &h0;
        // BATCHED FUSED qkv (MEMRA_F2B=1, megakernel microcosm): swa layers fuse all three,
        // globals fuse q,k (v := k clone). Bit-identical per row; segments tail-fill.
        static F2B_QKV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let f2b = *F2B_QKV.get_or_init(|| std::env::var("MEMRA_F2B").as_deref() != Ok("0"));
        let fused_qkv = if f2b {
            if swa {
                e.matmul_q4_fused3_batched(&fa.wq, &fa.wk, &fa.wv, hq, hdq, t)?
                    .map(|(a, b, c)| (a, b, Some(c)))
            } else {
                e.matmul_q4_fused2_batched(&fa.wq, &fa.wk, hq, hdq, t)?
                    .map(|(a, b)| (a, b, None))
            }
        } else { None };
        let (q0, k0, v0) = match fused_qkv {
            Some((a, b, cv)) => {
                let v = match cv { Some(c) => c, None => e.clone_dtod(&b)? };
                (a, b, v)
            }
            None => {
                let q0 = e.matmul_pre(&fa.wq, hq, hdq, h, t)?;
                let k0 = e.matmul_pre(&fa.wk, hq, hdq, h, t)?;
                let v0 = if swa { e.matmul_pre(&fa.wv, hq, hdq, h, t)? }
                         else { e.clone_dtod(&k0)? };
                (q0, k0, v0)
            }
        };
        let mut q = e.uninit(t * nh * hd)?;
        let mut k = e.uninit(t * nkv * hd)?;
        let mut v = e.uninit(t * nkv * hd)?;
        // E4B wave-3 fold (2026-07-23 decode-dust port): q/k/v norms + q/k rope in ONE
        // launch — rope math verbatim on the normed rows; V ones-rms, never roped.
        let ff = if swa { None } else {
            Some(aux.rope_freqs.as_ref().expect("gemma4 global rope needs rope_freqs.weight"))
        };
        e.rms_norm_qkv_rope(&q0, &k0, &v0, fa.q_norm.float_data(), fa.k_norm.float_data(),
                            &aux.ones, &mut q, &mut k, &mut v, hd, nh * t, nkv * t,
                            pos_d, nh, nkv, base, 1.0, ff, eps)?;
        let kvl = cache.kv[il].as_mut().unwrap();
        let base_len = kvl.len;
        e.append_kv_quantized_rows(&k, &v, &mut kvl.k, &mut kvl.v, base_len, t,
                                   kvl.kv_dim_k, kvl.kv_dim_v, kvl.k_tok_bytes, kvl.v_tok_bytes, (!swa && crate::Engine::gkv_on()) || (swa && crate::Engine::wkv_on()))?;
        kvl.len += t;
        let win = self.cfg.gemma4.as_ref().unwrap().sliding_window as usize;
        let mut attn = e.uninit(t * nh * hd)?;
        // ROWS twin when no window offset is in play (globals are hd512-ineligible; SWA rows
        // under the window need no offset): ONE launch, per-row causal == the per-token loop.
        let rows_ok = (hd == 256 && base_len + 1 >= crate::fa_vec_min_tkv())
            // gemma globals: hd512 rows twin in the dpl16 vec regime (row 0 gates the batch);
            // decode rides the SAME symbol at t=1 (parity law).
            || (hd == 512 && !swa && base_len + 1 >= crate::fa512_min_tkv());
        if rows_ok && (!swa || base_len + t <= win) {
            let k_view = e.view_u8(&kvl.k, (base_len + t) * kvl.k_tok_bytes);
            let v_view = e.view_u8(&kvl.v, (base_len + t) * kvl.v_tok_bytes);
            if hd == 512 {
                // device-len twin: sync the counter to the verify base (async arg-store).
                e.i32_set_k(&mut kvl.len_d, base_len as i32)?;
                e.fa_decode_rows(&q, &k_view, &v_view, &mut attn, hd, nh, nkv, base_len, t,
                                 scale, kvl.k_tok_bytes, kvl.v_tok_bytes,
                                 Some((&kvl.len_d, 0)), false,
                                 swa && crate::Engine::wkv_on(), None)?;
            } else {
                // hd256: the SAME v4_dc symbol the burst verify launches (PARITY LAW — the
                // host-base rows_v4 twin compiles apart from v4_dc and the two-symbol split
                // drifted the burst's persisted KV at il=8; one symbol, counter arg-stored).
                e.i32_set_k(&mut kvl.len_d, base_len as i32)?;
                e.fa_decode_rows_dc(&q, &k_view, &v_view, &mut attn, hd, nh, nkv,
                                    &kvl.len_d, base_len + t, t, scale,
                                    kvl.k_tok_bytes, kvl.v_tok_bytes, 0,
                                    swa && crate::Engine::wkv_on())?;
            }
            return Ok(e.matmul(&fa.wo, &attn, t)?);
        }
        // WINDOWED rows twin (deep ctx, every row fully windowed): one launch, ABSOLUTE-index
        // per-row geometry over the prefix view. PARITY LAW (2026-07-10 root-cause): textually
        // identical kernels do NOT compile bit-identically (nvcc unrolls fa_decode_vec_q 2x vs
        // its rows_w clone — SASS-proven; the unpinned score `+=` chain then rounds apart), so
        // decode-vs-verify parity comes from BOTH sides launching THIS SAME rows_w kernel
        // (decode passes t=1) — bitwise equal per position by symbol identity, any lane.
        // MEMRA_GEMMA_ROWS_W=0 -> per-token loop (decode falls back to fa_decode views too).
        if hd == 256 && swa && base_len + 1 >= win
            && std::env::var("MEMRA_GEMMA_ROWS_W").as_deref() != Ok("0") {
            let k_view = e.view_u8(&kvl.k, (base_len + t) * kvl.k_tok_bytes);
            let v_view = e.view_u8(&kvl.v, (base_len + t) * kvl.v_tok_bytes);
            e.i32_set_k(&mut kvl.len_d, base_len as i32)?;
            e.fa_decode_rows_w(&q, &k_view, &v_view, &mut attn, hd, nh, nkv, &kvl.len_d, 0,
                               t, scale, win, kvl.k_tok_bytes, kvl.v_tok_bytes, None)?;
            return Ok(e.matmul(&fa.wo, &attn, t)?);
        }
        for i in 0..t {
            let avail = base_len + i + 1;
            let (off_tok, t_kv) = if swa && avail > win { (avail - win, win) } else { (0, avail) };
            let k_view = e.view_u8_range(&kvl.k, off_tok * kvl.k_tok_bytes,
                                         (off_tok + t_kv) * kvl.k_tok_bytes);
            let v_view = e.view_u8_range(&kvl.v, off_tok * kvl.v_tok_bytes,
                                         (off_tok + t_kv) * kvl.v_tok_bytes);
            let qi = e.view(&q, t * nh * hd);
            let q_row = qi.slice(i * nh * hd..(i + 1) * nh * hd);
            let mut q_one = e.uninit(nh * hd)?;
            e.copy_view_into(&mut q_one, 0, &q_row, nh * hd)?;
            let mut a_one = e.uninit(nh * hd)?;
            // straddle rounds: windowed rows must ride the SAME rows_w kernel decode uses
            // (parity law above); global hd512 rows past the fa512 floor likewise ride the
            // rows_dpl16 twin; remaining rows keep the gated fa_decode pair.
            if swa && avail > win && hd == 256
                && std::env::var("MEMRA_GEMMA_ROWS_W").as_deref() != Ok("0") {
                let kp = e.view_u8(&kvl.k, avail * kvl.k_tok_bytes);
                let vp = e.view_u8(&kvl.v, avail * kvl.v_tok_bytes);
                e.i32_set_k(&mut kvl.len_d, (avail - 1) as i32)?;
                e.fa_decode_rows_w(&q_one, &kp, &vp, &mut a_one, hd, nh, nkv, &kvl.len_d, 0,
                                   1, scale, win, kvl.k_tok_bytes, kvl.v_tok_bytes, None)?;
            } else if !swa && hd == 512 && avail >= crate::fa512_min_tkv()
                && std::env::var("MEMRA_GEMMA_ROWS_W").as_deref() != Ok("0") {
                let kp = e.view_u8(&kvl.k, avail * kvl.k_tok_bytes);
                let vp = e.view_u8(&kvl.v, avail * kvl.v_tok_bytes);
                e.i32_set_k(&mut kvl.len_d, (avail - 1) as i32)?;
                e.fa_decode_rows(&q_one, &kp, &vp, &mut a_one, hd, nh, nkv, avail - 1, 1,
                                 scale, kvl.k_tok_bytes, kvl.v_tok_bytes,
                                 Some((&kvl.len_d, 0)), false, false, None)?;
            } else {
                e.fa_decode_kvmod(&q_one, &k_view, &v_view, &mut a_one, hd, nh, nkv, t_kv, scale,
                            kvl.k_tok_bytes, kvl.v_tok_bytes, swa && crate::Engine::wkv_on())?;
            }
            e.copy_into(&mut attn, i * nh * hd, &a_one, nh * hd)?;
        }
        Ok(e.matmul(&fa.wo, &attn, t)?)
    }

    /// gemma4 T=1 decode step: R8 layer graph over the cache; returns (softcapped logits host,
    /// h_seed = pre-output_norm hidden). Advances cache.pos.
    pub(crate) fn gemma4_decode_step_h(&self, e: &Engine, token: u32, cache: &mut Cache)
                                       -> Result<(Vec<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        // M1-PP2 door (crate::pp): 2-stage split with an explicit activation handoff.
        // Default OFF — unset env means this branch is never taken. The gemma4 arm stays
        // 2-stage in M2 (the N-stage gate model is the generic-arm 9B); N>2 warns + runs
        // unsplit rather than guessing a fence.
        if let Some(split) = crate::pp::pp2_split(self.layers.len()) {
            return self.gemma4_decode_step_h_pp2(e, token, cache, split);
        }
        if crate::pp::pp_cuts(self.layers.len()).is_some() {
            crate::pp::warn_unwired_once("gemma4 eager decode (N>2)");
        }
        let n_embd = self.cfg.n_embd as usize;
        let eps = self.cfg.rms_eps;
        let pos_d = e.htod_i32(&[cache.pos as i32])?;
        let mut x = e.htod(&self.embd.gather(n_embd, &[token]))?;
        e.scale_inplace(&mut x, (n_embd as f32).sqrt(), n_embd)?;
        // cross-layer fusion: each tail's closing add+scale also EMITS the next layer's
        // attn-normed input pre-quantized q8_1 (the mixer consumes only quantized matmuls).
        let mut h_carry: Option<(CudaSlice<i8>, CudaSlice<f32>)> = None;
        let n_layers = self.layers.len();
        for (il, layer) in self.layers.iter().enumerate() {
            let (hq, hdq) = match h_carry.take() {
                Some(p) => p,
                None => e.rms_norm_q8_1(&x, self.layers[0].attn_norm.float_data(), n_embd, 1, eps)?,
            };
            let Mixer::Full(fa) = &layer.mixer else { panic!("gemma4 layer {il} not full-attn") };
            let o = self.gemma4_decode_attn(e, fa, il, &hq, &hdq, &pos_d, cache)?;
            let mut cur = e.uninit(n_embd)?;
            e.rms_norm(&o, layer.post_attn_norm.float_data(), &mut cur, n_embd, 1, eps)?;
            let next_norm = if il + 1 < n_layers {
                Some(self.layers[il + 1].attn_norm.float_data())
            } else { None };
            let (xn, hn) = self.gemma4_layer_tail_add_nq(e, layer, &cur, &x, 1, next_norm)?;
            x = xn;
            h_carry = hn;
        }
        let mut hn = e.uninit(n_embd)?;
        e.rms_norm(&x, self.output_norm.float_data(), &mut hn, n_embd, 1, eps)?;
        let h_seed = e.clone_dtod(&x)?;
        let mut ld = e.matmul(&self.output, &hn, 1)?;
        let cap = self.cfg.gemma4.as_ref().unwrap().final_logit_softcapping;
        e.softcap(&mut ld, cap, self.output.out_features())?;   // R4 on device (262k host tanh ~ms/step)
        self.gemma4_suppress(e, &mut ld, 1)?;
        let logits = e.dtoh(&ld)?;
        cache.pos += 1;
        Ok((logits, h_seed))
    }

    /// M1-PP2 stage subgraph (gemma4 arm): layers [lo, hi) of the gemma4 T=1 walk with the
    /// pre-quantized next-norm carry LOCAL to the range. Enters with a materialized residual
    /// `x` (the range head runs its own `rms_norm_q8_1` against ITS layer's attn_norm — for
    /// lo == 0 that is exactly the unsplit loop's il==0 arm); exits with the residual
    /// materialized (range tail passes next_norm = None, exactly the unsplit last-layer arm).
    /// Bit-identity of the cut relies on the kernel-check-pinned `add_scale_rms_norm_q8_1 ==
    /// add_scale then rms_norm_q8_1` identity (`pp2-gate` verifies end-to-end).
    fn gemma4_decode_layers(&self, e: &Engine, mut x: CudaSlice<f32>, lo: usize, hi: usize,
                            pos_d: &CudaSlice<i32>, cache: &mut Cache)
                            -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let n_embd = self.cfg.n_embd as usize;
        let eps = self.cfg.rms_eps;
        let mut h_carry: Option<(CudaSlice<i8>, CudaSlice<f32>)> = None;
        for il in lo..hi {
            let layer = &self.layers[il];
            let (hq, hdq) = match h_carry.take() {
                Some(p) => p,
                // range head: il == lo — norm against THIS layer's attn_norm.
                None => e.rms_norm_q8_1(&x, self.layers[il].attn_norm.float_data(), n_embd, 1, eps)?,
            };
            let Mixer::Full(fa) = &layer.mixer else { panic!("gemma4 layer {il} not full-attn") };
            let o = self.gemma4_decode_attn(e, fa, il, &hq, &hdq, pos_d, cache)?;
            let mut cur = e.uninit(n_embd)?;
            e.rms_norm(&o, layer.post_attn_norm.float_data(), &mut cur, n_embd, 1, eps)?;
            let next_norm = if il + 1 < hi {
                Some(self.layers[il + 1].attn_norm.float_data())
            } else { None };
            let (xn, hn) = self.gemma4_layer_tail_add_nq(e, layer, &cur, &x, 1, next_norm)?;
            x = xn;
            h_carry = hn;
        }
        Ok(x)
    }

    /// M1-PP2 (increment 2): `gemma4_decode_step_h` as TWO stage subgraphs, each on its
    /// own stream (and device, under MEMRA_PP_DEVICES) with the transport-selected
    /// boundary handoff — same choreography as the generic arm (decode.rs), same
    /// ownership contract (crate::pp). Stage 0 = embed+scale + layers [0, split);
    /// stage 1 = layers [split, n) + output_norm + softcapped head.
    /// MEMRA_PP_STREAMS=0 = the increment-1 same-stream seam. Gate: `pp2-gate`.
    fn gemma4_decode_step_h_pp2(&self, e: &Engine, token: u32, cache: &mut Cache, split: usize)
                                -> Result<(Vec<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        if crate::pp::pp2_streams_off() {
            return self.gemma4_decode_step_h_pp2_samestream(e, token, cache, split);
        }
        let rt = crate::pp::Pp2Rt::get(e)?;
        let e0 = rt.engine(0, e);
        let e1 = rt.engine(1, e);
        let n_embd = self.cfg.n_embd as usize;
        let eps = self.cfg.rms_eps;

        // ---- STAGE 0 (its own stream): embed + sqrt(n_embd) scale + layers [0, split) ----
        let (pos_d, slot) = {
            let _st0 = rt.enter(0);
            let pos_d = e0.htod_i32(&[cache.pos as i32])?;
            let mut x = e0.htod(&self.embd.gather(n_embd, &[token]))?;
            e0.scale_inplace(&mut x, (n_embd as f32).sqrt(), n_embd)?;
            let x = self.gemma4_decode_layers(e0, x, 0, split, &pos_d, cache)?;
            let slot = rt.tx(0, &x, n_embd)?;
            (pos_d, slot)
        };

        // ---- STAGE 1 (its own stream): RX + layers [split, n) + softcapped head ----
        let _st1 = rt.enter(1);
        let x = rt.rx(0, slot, n_embd)?;
        let x = self.gemma4_decode_layers(e1, x, split, self.layers.len(), &pos_d, cache)?;

        let mut hn = e1.uninit(n_embd)?;
        e1.rms_norm(&x, self.output_norm.float_data(), &mut hn, n_embd, 1, eps)?;
        let h_seed = e1.clone_dtod(&x)?;
        let mut ld = e1.matmul(&self.output, &hn, 1)?;
        let cap = self.cfg.gemma4.as_ref().unwrap().final_logit_softcapping;
        e1.softcap(&mut ld, cap, self.output.out_features())?;
        self.gemma4_suppress(e1, &mut ld, 1)?;
        let logits = e1.dtoh(&ld)?;
        cache.pos += 1;
        Ok((logits, h_seed))
    }

    /// MEMRA_PP_STREAMS=0 rollback seam: the increment-1 gemma4 pp2 body verbatim — both
    /// stage subgraphs on the ambient compute stream, boundary = two plain dtod copies.
    fn gemma4_decode_step_h_pp2_samestream(&self, e: &Engine, token: u32, cache: &mut Cache,
                                           split: usize)
                                           -> Result<(Vec<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let n_embd = self.cfg.n_embd as usize;
        let eps = self.cfg.rms_eps;
        let pos_d = e.htod_i32(&[cache.pos as i32])?;

        // ---- STAGE 0: embed + sqrt(n_embd) scale + layers [0, split) ----
        let mut x = e.htod(&self.embd.gather(n_embd, &[token]))?;
        e.scale_inplace(&mut x, (n_embd as f32).sqrt(), n_embd)?;
        let x = self.gemma4_decode_layers(e, x, 0, split, &pos_d, cache)?;

        // ---- STAGE BOUNDARY: explicit [n_embd] activation handoff (TX copy, RX copy) ----
        let boundary_tx = e.clone_dtod(&x)?;
        let boundary_rx = e.clone_dtod(&boundary_tx)?;

        // ---- STAGE 1: layers [split, n) + output_norm + softcapped head ----
        let x = self.gemma4_decode_layers(e, boundary_rx, split, self.layers.len(), &pos_d, cache)?;

        let mut hn = e.uninit(n_embd)?;
        e.rms_norm(&x, self.output_norm.float_data(), &mut hn, n_embd, 1, eps)?;
        let h_seed = e.clone_dtod(&x)?;
        let mut ld = e.matmul(&self.output, &hn, 1)?;
        let cap = self.cfg.gemma4.as_ref().unwrap().final_logit_softcapping;
        e.softcap(&mut ld, cap, self.output.out_features())?;
        self.gemma4_suppress(e, &mut ld, 1)?;
        let logits = e.dtoh(&ld)?;
        cache.pos += 1;
        Ok((logits, h_seed))
    }
}

// ============================ step35 (Step-3.7-Flash) ==================================
// Node-for-node vs llama.cpp src/models/step35.cpp:216-300. WHY THIS IS A DEDICATED MIXER
// FAMILY and not a few branches inside the generic `full_attn*` chain:
//
//   1. `n_head` IS PER LAYER (`step35.attention.head_count` is an ARRAY: 64 on full-attn
//      layers, 96 on SWA). Every generic site reads the `cfg.n_head` scalar, so wq/wo/attn_gate
//      shapes and the FA head counts would be wrong on 33 of 45 layers.
//   2. `cfg.rope_dim_count` is 128 for this arch, but FULL-attn layers rotate only 64 dims
//      (upstream halves `n_rot_full` after the generic loader defaults it). A generic mixer
//      would rotate 128 dims on the full layers — silently wrong, plausible-looking logits.
//   3. Dual rope base (5e6 full / 1e4 SWA) + `rope_freqs.weight` on FULL layers ONLY.
//   4. SWA 3:1 (`sliding_window` 512, pattern [false,true,true,true]).
//   5. The head-wise gate is a SEPARATE `attn_gate.weight [n_embd, n_head_l]` (one pre-sigmoid
//      scalar per head, broadcast over head_dim) applied to the attention output BEFORE wo —
//      not the qwen35 fused-in-wq per-DIM gate that `attn_out_gate()`/`q_gate_split` handle.
//      Its input is the POST-attn_norm hidden (`cur`), not the residual.
//
// Attention scale is the DEFAULT 1/sqrt(n_embd_head_k) (step35.cpp:255) — NOT gemma4's 1.0.
impl HybridModel {
    /// Per-layer attention geometry: (head_dim, n_kv, n_head, rope_base, scale, is_swa).
    pub(crate) fn step35_geom(&self, il: usize) -> (usize, usize, usize, f32, f32, bool) {
        let s = self.cfg.step35.as_ref().unwrap();
        let hd = self.cfg.head_dim_k as usize;
        (hd,
         s.n_head_kv(il as u32) as usize,        // 8 (uniform on 3.7-Flash)
         s.n_head(il as u32) as usize,           // 64 full / 96 SWA
         s.rope_base(il as u32),                 // 5e6 full / 1e4 SWA
         1.0 / (hd as f32).sqrt(),               // step35.cpp:255 kq_scale
         s.is_swa(il as u32))
    }

    /// step35 attention core: everything from the q/k/v projection outputs through the head-wise
    /// gate, returning the GATED attention output PRE-`wo` (the caller runs wo, so this composes
    /// with both the plain `matmul(&fa.wo, ..)` sites and the f16/into-slab GEMM sites).
    ///
    /// `hg` = the POST-attn_norm hidden state (upstream's `cur`) — the gate projection's input
    /// on single-sequence paths. `gt_pre` supplies the already-projected per-head gate for a
    /// cross-request batch; exactly one of `hg` or `gt_pre` must be present.
    /// `cache`:
    ///   * `Some` => PRIME mode: append this chunk's post-rope K / raw V rows into the resident
    ///     quantized cache and attend THROUGH the cache view, exactly like the generic
    ///     `full_attn_prime_fa_dispatch` (quantize-then-attend on chunk 0 too, so chunk size
    ///     cannot decide where a precision edge falls — the grain-free chunk-invariance
    ///     contract, lane/chunkinv-flip).
    ///   * `None` => pure prefill (forward/forward_last/t2probe): attend over this batch's f32
    ///     q/k/v, no cache side effect.
    ///
    /// SWA WINDOW, and why prefill needs `sdpa_naive_w_quantized_view`: the view is trimmed to
    /// the oldest key ANY query in this chunk can reach (`off = base_len - (win-1)`), which
    /// bounds the view at `win-1+t` rows, but the mask is still required — inside the chunk,
    /// query `qt` may only see view keys `[qt, qt+win-1]`, so the earlier keys the trimmed view
    /// still contains must be masked per query. memra's window convention
    /// (`sdpa_naive_w_f32`: mask `t < q_pos-(window-1)`, `q_pos = (T_kv-T)+qt`) is bit-for-bit
    /// upstream's `LLAMA_SWA_TYPE_STANDARD` (`llama-hparams.h:359`: mask `p1-p0 >= n_swa`), and
    /// step35.cpp:6 sets exactly that swa_type — verified verbatim on both sides.
    ///
    /// Note the f32 `sdpa_naive_w` floor's shared memory is `t_kv*4` bytes, so the SWA arm needs
    /// `t_kv = win-1+t <= 12287` — i.e. it relies on chunked prefill (MEMRA_PRIME_CHUNK, default
    /// 4096). A monolithic 32k prime would exceed the 48 KiB dynamic-smem default.
    ///
    /// SWA ARM SELECTION IS KEYED ON `seq_end`, NOT ON THE CHUNK'S OWN `t_kv` (lane/step35-chunkfix,
    /// 2026-08-07). `seq_end` = the ABSOLUTE end position of the whole prime call
    /// (`cache.pos + prompt_len` at entry; `t` on the cacheless prefill path) — a property of the
    /// REQUEST, identical at every MEMRA_PRIME_CHUNK value. The old predicate `swa && t_kv > win`
    /// read the chunk's own extent, which made kernel selection — and therefore the logits, the
    /// hidden rows, and the generated text — a function of the chunk size:
    ///   a chunk [b,e) has off = max(0, b-(win-1)), t_kv = e-off, so b <= win-1 => off=0 => FA iff
    ///   e <= win, while every later chunk has t_kv = t+(win-1) > win. The FA rows were therefore a
    ///   contiguous PREFIX [0,P) with P = c*floor(win/c) for c <= win (else 0), and the output
    ///   depended only on P. Measured (research/step37-p2-20260806, closed-form receipt
    ///   `raw/chunkinv-step35-GAP2-CONFIRMED-20260807.txt`): at T=4883, c=512 and c=64 (P=512) are
    ///   byte-identical to each other but DIFFER from the c=4096 default (P=0) by maxdiff 1.813e0
    ///   with greedy text diverging at step 6; c=513 (P=0) is bit-identical to the default. A
    ///   one-token change in a documented machine-config knob changed the answer.
    /// So the pre-fix comment "same cache bytes, same numeric class" was false as written: same
    /// bytes, DIFFERENT numeric class — swapping `fa_prefill_view_ws` for the f32 windowed floor on
    /// the same rows moves the logits by ~1.8.
    ///
    /// Keying on `seq_end` makes P identically 0 for every chunk size, so MEMRA_PRIME_CHUNK is a
    /// pure memory/transient knob again (research/step35-chunkfix-20260807). It is
    /// correct-by-construction rather than a tolerance argument — the windowed kernel computes the
    /// same masked attention the FA arm computed (for e <= win the window mask is a no-op under
    /// causal, which is exactly why FA was legal there), it is now simply the ONLY arm used once
    /// the request passes the window. It also cannot move the shipped default: at chunk=4096 every
    /// chunk of a `seq_end > win` prime ALREADY had t_kv > win (chunk 0 has t_kv = min(chunk,
    /// seq_end) > win; every later chunk has t_kv >= t+win-1 > win since `PRIME_MIN_T` keeps
    /// t >= 16), and a `seq_end <= win` prime keeps every chunk on FA exactly as before — so this
    /// is a no-op on both default regimes and only removes the FA prefix at small chunk values.
    /// The `t_kv <= 12287` smem ceiling is respected: the chunks whose arm changes are precisely
    /// those with t_kv <= win = 512.
    #[allow(clippy::too_many_arguments)]
    fn step35_attn_pre_wo(&self, e: &Engine, fa: &FullAttnLayer, mut g3: Vec<CudaSlice<f32>>,
                          hg: Option<&CudaSlice<f32>>, gt_pre: Option<&CudaSlice<f32>>,
                          pos_d: &CudaSlice<i32>, t: usize,
                          cache: Option<&mut Cache>, il: usize, seq_end: usize)
                          -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let (hd, nkv, nh, rbase, scale, swa) = self.step35_geom(il);
        let eps = self.cfg.rms_eps;
        let s = self.cfg.step35.as_ref().unwrap();
        let win = s.sliding_window as usize;
        let n_rot = s.n_rot(il as u32) as usize;

        let v = g3.pop().unwrap();
        let k0 = g3.pop().unwrap();
        let q0 = g3.pop().unwrap();

        // q/k RMSNorm over head_dim rows (attn_q_norm/attn_k_norm [128] F32), then the
        // per-layer PARTIAL NEOX rope: n_rot = 64 on full layers (dims 64..127 pass through
        // unrotated), 128 on SWA. rope_freqs (llama3-style per-dim factors) on FULL only.
        let mut q = e.uninit(t * nh * hd)?;
        e.rms_norm(&q0, fa.q_norm.float_data(), &mut q, hd, nh * t, eps)?;
        let mut k = e.uninit(t * nkv * hd)?;
        e.rms_norm(&k0, fa.k_norm.float_data(), &mut k, hd, nkv * t, eps)?;
        let ff = if swa { None } else {
            self.step35_aux.as_ref().and_then(|a| a.rope_freqs.as_ref())
        };
        e.rope_neox2(&mut q, &mut k, pos_d, hd, n_rot, nh, nkv, t, rbase, 1.0, ff)?;

        let mut attn = e.uninit(t * nh * hd)?;
        match cache {
            Some(cache) => {
                let base_len = {
                    let kvl = cache.kv[il].as_mut().unwrap();
                    assert!(kvl.len + t <= cache.max_ctx, "step35 prime: KV overflow");
                    let base_len = kvl.len;
                    e.append_kv_quantized_rows(&k, &v, &mut kvl.k, &mut kvl.v, base_len, t,
                                               kvl.kv_dim_k, kvl.kv_dim_v, kvl.k_tok_bytes,
                                               kvl.v_tok_bytes, crate::Engine::kv_fp8_on())?;
                    kvl.len += t;
                    let new_len = kvl.len as i32;
                    e.set_i32_one(&mut kvl.len_d, new_len)?;
                    base_len
                };
                let kvl = cache.kv[il].as_ref().unwrap();
                // Both segmentation seams restore the FULL pre-fix arithmetic: their legacy
                // arm predicate (`t_kv` below, per-call `seq_end` in prime_cache) AND the
                // unaligned view offset here. Both halves are load-bearing for the canaries:
                // on the FA default the predicate arms agree bitwise wherever they can differ
                // (a t_kv<=win view has no maskable key, so windowed==unwindowed FA bit-for-bit).
                // That made predicate-only seams INERT under Lever A — first chunkinv35c, then
                // tickinv35c. The raw offset restores the live segmentation-variant mechanism
                // on the current FA path: its tile grid starts at the chunk/call boundary.
                // Read per layer call, never in a measured default.
                let legacy_tkv = std::env::var("MEMRA_STEP35_SWA_TKV").as_deref() == Ok("1");
                let legacy_calllocal =
                    std::env::var("MEMRA_PRIME_CALLLOCAL").as_deref() == Ok("1");
                // SWA: trim the view to the oldest key any query in this chunk can reach —
                // ALIGNED DOWN to the FA tile size (BK=32). The raw trim is chunk-DEPENDENT
                // (off = base_len-(win-1), and base_len is a chunk boundary), and the FA
                // kernel's online-softmax recurrence groups keys into BK tiles relative to
                // the VIEW START — so an unaligned off regroups the same absolute keys into
                // different tiles at different chunk sizes = different (m,l) rounding =
                // chunk-dependent bits. chunkinv35 caught exactly this on the first FA-arm
                // battery (first_div == chunk size; raw/leverA-gates-20260807T135541Z.log).
                // Aligning off to 32 pins tiles to ABSOLUTE key positions for every chunk
                // size; the <=31 extra leading keys are older than EVERY query's window
                // (all queries sit at >= base_len, so keys < base_len-(win-1) are masked
                // for all of them) and a fully-masked key is an exact-0.0 no-op in both
                // kernels (NEG_INF -> p=0.0; l+=0.0 and O+=0.0 are bitwise identity), so
                // the floor arm's bits do not move either (gated: G2f, battery 2).
                let off = if swa {
                    let raw = base_len.saturating_sub(win - 1);
                    if legacy_tkv || legacy_calllocal { raw } else { raw & !31usize }
                } else {
                    0
                };
                let t_kv = base_len + t - off;
                let k_view = e.view_u8_range(&kvl.k, off * kvl.k_tok_bytes,
                                             (off + t_kv) * kvl.k_tok_bytes);
                let v_view = e.view_u8_range(&kvl.v, off * kvl.v_tok_bytes,
                                             (off + t_kv) * kvl.v_tok_bytes);
                // CHUNK-INVARIANT PREDICATE: `seq_end` (the request's absolute end position), not
                // this chunk's `t_kv`. See the doc note — keying on t_kv made P = c*floor(win/c)
                // rows take FA and the output a function of MEMRA_PRIME_CHUNK.
                // MEMRA_STEP35_SWA_TKV=1 is the ROLLBACK SEAM to the pre-fix arithmetic — BOTH
                // halves: this predicate AND the unaligned view offset above (`legacy_tkv`).
                // It is what gives the step35 chunkinv gate its canary teeth: chunk-VARIANT by
                // construction, so the invariance assertion MUST break under it (the seam whose
                // absence made the original canary inert — GAP 1 in research/step37-p2-20260806;
                // and a predicate-only seam went inert AGAIN under the FA default, battery 2's
                // CANARY UNEXPECTEDLY MATCHED — see the offset comment). Read per call, not
                // cached (probes flip it in-process). Never on in a measured default run.
                let swa_naive = if legacy_tkv { t_kv > win } else { seq_end > win };
                if swa && swa_naive {
                    // Windowed mask needed (see the doc note). DEFAULT since lane/pp-prefill
                    // 2026-08-07: the windowed hd128 FA stamp (`fa_prefill_view_ws_w_hd128`) —
                    // the anatomy profile measured the f32 floor at 565 ms/layer on a pp4096
                    // (41% of the whole prime) while the unwindowed hd128 FA family did the
                    // strictly harder causal-4096 in 3.3 ms. NOTE t_kv can be <= win here
                    // (a chunk-0 view on a small chunk); the windowed kernel handles that
                    // identically to the unwindowed one modulo the mask, which is the point.
                    // NEW NUMERIC CLASS vs the floor (bf16-MMA online softmax vs f32 serial),
                    // selected on `seq_end` like every arm here, so the class is uniform for
                    // the whole request at every MEMRA_PRIME_CHUNK — chunkinv holds by the
                    // same construction as the chunkfix. MEMRA_STEP35_SWA_FA=0 = rollback to
                    // the f32 floor (the previous numeric config, kept as the A/B seam).
                    if std::env::var("MEMRA_STEP35_SWA_FA").as_deref() == Ok("0") {
                        e.sdpa_naive_w_quantized_view(&q, &k_view, &v_view, &mut attn, hd, nh,
                                                      nkv, t, t_kv, scale, true, win,
                                                      kvl.k_tok_bytes, kvl.v_tok_bytes)?;
                    } else {
                        e.fa_prefill_view_ws_w_hd128(&q, &k_view, &v_view, &mut attn, hd, nh,
                                                     nkv, t, t_kv, scale, true, win,
                                                     kvl.k_tok_bytes, kvl.v_tok_bytes)?;
                    }
                } else if std::env::var("MEMRA_NOFA").is_ok() {
                    e.sdpa_naive_quantized_view(&q, &k_view, &v_view, &mut attn, hd, nh, nkv,
                                                t, t_kv, scale, true,
                                                kvl.k_tok_bytes, kvl.v_tok_bytes)?;
                } else {
                    // seq_end <= win (or a full-attn layer): no query in the WHOLE request can
                    // reach past the window, so the window mask is a no-op under causal and every
                    // chunk rides the hd128 dequant-once FA prefill — one arm for the whole
                    // request either way, which is what makes the chunk size arithmetic-free.
                    e.fa_prefill_view_ws(&q, &k_view, &v_view, &mut attn, hd, nh, nkv,
                                         t, t_kv, scale, true,
                                         kvl.k_tok_bytes, kvl.v_tok_bytes,
                                         crate::Engine::kv_fp8_on())?;
                }
            }
            None => {
                // Cacheless prefill (forward/forward_last/t2probe) is MONOLITHIC — there is no
                // chunk loop on this path, so seq_end == t and the predicate is unchanged. The
                // assert pins that: if a chunked cacheless prefill is ever added, it must thread
                // seq_end here too or it re-opens the same door.
                debug_assert_eq!(seq_end, t, "step35 cacheless prefill is monolithic (seq_end == t)");
                if swa && seq_end > win {
                    e.sdpa_naive_w(&q, &k, &v, &mut attn, hd, nh, nkv, t, t, scale, true, win)?;
                } else if std::env::var("MEMRA_NOFA").is_ok() {
                    e.sdpa_naive(&q, &k, &v, &mut attn, hd, nh, nkv, t, t, scale, true)?;
                } else {
                    e.fa_prefill(&q, &k, &v, &mut attn, hd, nh, nkv, t, t, scale, true)?;
                }
            }
        }

        // HEAD-WISE GATE (step35.cpp:267-285): gate = attn_gate(cur) -> [t, n_head_l];
        // attn *= sigmoid(gate) broadcast over head_dim. BEFORE wo.
        let gw = fa.attn_gate.as_ref()
            .ok_or("step35 layer is missing attn_gate.weight (head-wise attention gate)")?;
        let gt_owned = if gt_pre.is_none() {
            Some(e.matmul(
                gw,
                hg.ok_or("step35 attention needs hg when gt_pre is absent")?,
                t,
            )?)
        } else {
            None
        };
        let gt = gt_pre.or(gt_owned.as_ref()).unwrap();
        let mut ag = e.uninit(t * nh * hd)?;
        e.attn_head_gate(&attn, gt, &mut ag, None, hd, nh, t)?;
        Ok(ag)
    }

    /// step35 PREFILL mixer, no cache side effect (the `full_attn` contract: `forward`,
    /// `forward_last`, t2probe). Post-`wo`.
    pub(crate) fn step35_attn(&self, e: &Engine, fa: &FullAttnLayer, h: &CudaSlice<f32>,
                              pos_d: &CudaSlice<i32>, t: usize, il: usize)
                              -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let g3 = e.matmul_group(&[&fa.wq, &fa.wk, &fa.wv], h, t)?;
        // Monolithic path: the request ends at t (no chunk loop). See step35_attn_pre_wo's note.
        let ag = self.step35_attn_pre_wo(e, fa, g3, Some(h), None, pos_d, t, None, il, t)?;
        Ok(e.matmul(&fa.wo, &ag, t)?)
    }

    /// step35 PRIME mixer (the `full_attn_prime` contract: append this chunk's K/V into the
    /// resident quantized cache, attend through the cache view). Post-`wo`.
    ///
    /// `seq_end` = the ABSOLUTE end position of the whole prime request (chunk-invariant); it
    /// selects the SWA arm. See `step35_attn_pre_wo`'s doc note for why it cannot be this chunk's
    /// own extent.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn step35_attn_prime(&self, e: &Engine, fa: &FullAttnLayer, h: &CudaSlice<f32>,
                                    hx: Option<&CudaSlice<u8>>, pos_d: &CudaSlice<i32>, t: usize,
                                    cache: &mut Cache, il: usize, seq_end: usize)
                                    -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let g3 = match hx {
            Some(xh) => e.matmul_group_xh(&[&fa.wq, &fa.wk, &fa.wv], h, xh, t)?,
            None => e.matmul_group(&[&fa.wq, &fa.wk, &fa.wv], h, t)?,
        };
        let ag = self.step35_attn_pre_wo(
            e,
            fa,
            g3,
            Some(h),
            None,
            pos_d,
            t,
            Some(cache),
            il,
            seq_end,
        )?;
        Ok(e.matmul(&fa.wo, &ag, t)?)
    }

    /// step35 T=1 decode attention (post-`wo`, matching `full_attn_decode_pre`'s contract).
    /// `pre_q` = the attn-input norm's q8_1 pair when the caller took the norm-fusion lever
    /// (then `h` is a zero-length placeholder and EVERY projection here — including the gate —
    /// must be on the q8_1 fast path; `mixer_in_q8_1_fast` enforces that for step35 by also
    /// requiring `attn_gate`).
    ///
    /// SWA decode is a token-aligned VIEW OFFSET into the quantized cache (the gemma4 R6
    /// pattern): keys carry absolute rope and the mask is purely positional, so the single
    /// query at `len-1` attending the last `win` rows IS the windowed result — no mask kernel.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn step35_decode_attn(&self, e: &Engine, fa: &FullAttnLayer, il: usize,
                          h: &CudaSlice<f32>,
                          pre_q: Option<(&CudaSlice<i8>, &CudaSlice<f32>)>,
                          pos_d: &CudaSlice<i32>, cache: &mut Cache)
                          -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let (hd, nkv, nh, rbase, scale, swa) = self.step35_geom(il);
        let eps = self.cfg.rms_eps;
        let s = self.cfg.step35.as_ref().unwrap();
        let win = s.sliding_window as usize;
        let n_rot = s.n_rot(il as u32) as usize;
        let n_embd = self.cfg.n_embd as usize;
        let gw = fa.attn_gate.as_ref()
            .ok_or("step35 layer is missing attn_gate.weight (head-wise attention gate)")?;

        let (q0, k0, v0, gt) = match pre_q {
            Some((hq, hdq)) => {
                debug_assert!(e.uses_q8_1_fast(gw),
                    "step35 pre-quantized decode requires attn_gate on the q8_1 fast path \
                     (h is a zero-length placeholder here) — see mixer_in_q8_1_fast");
                let (a, b, c) = match e.matmul_q8_fused3(&fa.wq, &fa.wk, &fa.wv, hq, hdq)? {
                    Some(t3) => t3,
                    None => (e.matmul_pre(&fa.wq, hq, hdq, h, 1)?,
                             e.matmul_pre(&fa.wk, hq, hdq, h, 1)?,
                             e.matmul_pre(&fa.wv, hq, hdq, h, 1)?),
                };
                let gt = e.matmul_pre(gw, hq, hdq, h, 1)?;
                (a, b, c, gt)
            }
            None => {
                if e.uses_q8_1_fast(&fa.wq) && e.uses_q8_1_fast(&fa.wk)
                    && e.uses_q8_1_fast(&fa.wv) && e.uses_q8_1_fast(gw) {
                    let (hq, hdq) = e.quantize_q8_1(h, 1, n_embd)?;
                    let (a, b, c) = match e.matmul_q8_fused3(&fa.wq, &fa.wk, &fa.wv, &hq, &hdq)? {
                        Some(t3) => t3,
                        None => (e.matmul_pre(&fa.wq, &hq, &hdq, h, 1)?,
                                 e.matmul_pre(&fa.wk, &hq, &hdq, h, 1)?,
                                 e.matmul_pre(&fa.wv, &hq, &hdq, h, 1)?),
                    };
                    let gt = e.matmul_pre(gw, &hq, &hdq, h, 1)?;
                    (a, b, c, gt)
                } else {
                    (e.matmul(&fa.wq, h, 1)?, e.matmul(&fa.wk, h, 1)?,
                     e.matmul(&fa.wv, h, 1)?, e.matmul(gw, h, 1)?)
                }
            }
        };

        let mut q = e.uninit(nh * hd)?;
        e.rms_norm(&q0, fa.q_norm.float_data(), &mut q, hd, nh, eps)?;
        let mut k = e.uninit(nkv * hd)?;
        e.rms_norm(&k0, fa.k_norm.float_data(), &mut k, hd, nkv, eps)?;
        let ff = if swa { None } else {
            self.step35_aux.as_ref().and_then(|a| a.rope_freqs.as_ref())
        };
        e.rope_neox2(&mut q, &mut k, pos_d, hd, n_rot, nh, nkv, 1, rbase, 1.0, ff)?;

        if std::env::var("MEMRA_NOFA").is_ok() {
            return Err("MEMRA_NOFA (naive f32 SDPA) is incompatible with the quantized KV \
                        cache; unset MEMRA_NOFA to use fa_decode".into());
        }
        let kvl = cache.kv[il].as_mut().unwrap();
        e.append_kv_quantized(&k, &v0, &mut kvl.k, &mut kvl.v, kvl.len,
                              kvl.kv_dim_k, kvl.kv_dim_v, kvl.k_tok_bytes, kvl.v_tok_bytes,
                              crate::Engine::kv_fp8_on())?;
        kvl.len += 1;
        let (off, t_kv) = if swa && kvl.len > win { (kvl.len - win, win) } else { (0, kvl.len) };
        let k_view = e.view_u8_range(&kvl.k, off * kvl.k_tok_bytes,
                                     (off + t_kv) * kvl.k_tok_bytes);
        let v_view = e.view_u8_range(&kvl.v, off * kvl.v_tok_bytes,
                                     (off + t_kv) * kvl.v_tok_bytes);
        let mut attn = e.uninit(nh * hd)?;
        e.fa_decode_kvmod(&q, &k_view, &v_view, &mut attn, hd, nh, nkv, t_kv, scale,
                          kvl.k_tok_bytes, kvl.v_tok_bytes, crate::Engine::kv_fp8_on())?;

        let mut ag = e.uninit(nh * hd)?;
        e.attn_head_gate(&attn, &gt, &mut ag, None, hd, nh, 1)?;
        Ok(e.matmul(&fa.wo, &ag, 1)?)
    }
}

// ===================================================================================== //
//  gemma-4 E4B (per-layer embeddings + KV-sharing) — FIRST-LIGHT forward.               //
//  Dedicated simple path (Stage-B matmuls, per-row causal attention over the quantized  //
//  cache) so the tuned 26B/31B paths stay untouched. Arch: research/gemma4-bringup/     //
//  e4b-arch-map.md; llama reference: src/models/gemma4.cpp (E4B arms). Wired: forward   //
//  (prefill logits), prime (tokenwise-equivalent batched trunk), eager decode_step_h.   //
//  NOT wired: dc/graph serving arms, verify/spec, chunked prime (HANDOVER-E4B.md).      //
// ===================================================================================== //
impl HybridModel {
    pub fn is_gemma4_e4b(&self) -> bool {
        self.gemma4_aux.as_ref().is_some_and(|a| a.e4b.is_some())
    }

    /// E4B per-layer geometry: nh/nkv derive from the layer's OWN tensor shapes (the E4B GGUF
    /// ships a scalar head_count_kv; swa q 8x256 / kv 2x256, global q 4x512 / kv 1x512).
    /// KV-shared layers report the SHARE TARGET's kv count (their wk IS the target's tensor).
    fn gemma4_e4b_geom(&self, il: usize) -> (usize, usize, usize, f32, f32, bool) {
        let g = self.cfg.gemma4.as_ref().unwrap();
        let swa = g.swa_pattern[il];
        let hd = if swa { g.key_length_swa } else { g.key_length_global } as usize;
        let Mixer::Full(fa) = &self.layers[il].mixer else { panic!("e4b layer {il} not full-attn") };
        let nh = fa.wq.out_features() / hd;
        let nkv = fa.wk.out_features() / hd;
        (hd, nkv, nh, if swa { g.rope_base_swa } else { g.rope_base_global }, 1.0, swa)
    }

    /// E4B KV-share target: Some(target) for the trailing shared layers, None = own cache.
    fn gemma4_e4b_kv_target(&self, il: usize) -> Option<usize> {
        self.layers[il].gemma4.as_ref()
            .and_then(|b| b.e4b.as_ref())
            .and_then(|e4| e4.kv_share.map(|t| t as usize))
    }

    /// E4B prologue: inp_pl [t][n_layer][n_epl] =
    ///   ( gather(per_layer_tok_embd, tok)*sqrt(n_epl)
    ///   + rms_norm(model_proj . x_scaled * 1/sqrt(n_embd), proj_norm) ) * 1/sqrt(2)
    /// (llama gemma4.cpp build_inp_per_layer + project_per_layer_inputs, exact order).
    fn gemma4_e4b_inp_pl(&self, e: &Engine, tokens: &[u32], x_scaled: &CudaSlice<f32>, t: usize)
                         -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let tok_d = e.stream().clone_htod(&tokens.to_vec())?;
        self.gemma4_e4b_inp_pl_dev(e, &tok_d, x_scaled, t)
    }

    /// Device-token prologue core (dc arm shares it: token ids never touch the host).
    fn gemma4_e4b_inp_pl_dev(&self, e: &Engine, tok_d: &CudaSlice<u32>,
                             x_scaled: &CudaSlice<f32>, t: usize)
                             -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let aux = self.gemma4_aux.as_ref().unwrap();
        let m = aux.e4b.as_ref().unwrap();
        let n_embd = self.cfg.n_embd as usize;
        let n_layer = self.layers.len();
        let width = m.n_epl * n_layer;
        let tbl = m.tok_tbl_gpu.get_or_init(|| {
            e.upload_u8(&m.tok_embd_bytes).expect("e4b per-layer token table upload")
        });
        let mut a = e.embed_gather_device_td(tbl, tok_d, t, width, m.tok_embd_qt,
                                             m.tok_embd_row_bytes)?;
        e.scale_inplace(&mut a, (m.n_epl as f32).sqrt(), t * width)?;
        let mut p = e.matmul(&m.model_proj, x_scaled, t)?;
        e.scale_inplace(&mut p, 1.0 / (n_embd as f32).sqrt(), t * width)?;
        let mut pn = e.uninit(t * width)?;
        e.rms_norm(&p, m.proj_norm.float_data(), &mut pn, m.n_epl, t * n_layer,
                   self.cfg.rms_eps)?;
        let mut out = e.uninit(t * width)?;
        e.add_scale(&a, &pn, 1.0 / 2f32.sqrt(), &mut out, t * width)?;
        Ok(out)
    }

    /// E4B attention (t-wide, causal, per-row fa over the QUANTIZED cache — first-light
    /// correctness path; the fa rows arms come later). Own-KV layers project+norm+rope k/v
    /// and append t rows; KV-shared layers are Q-only over the target layer's cache (which
    /// already holds this forward's rows — the target runs earlier in the stack).
    #[allow(clippy::too_many_arguments)]
    fn gemma4_e4b_attn(&self, e: &Engine, il: usize,
                       hq: &CudaSlice<i8>, hdq: &CudaSlice<f32>,
                       pos_d: &CudaSlice<i32>, t: usize, cache: &mut Cache,
                       dc_bucket: Option<usize>)
                       -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let (hd, nkv, nh, base, scale, swa) = self.gemma4_e4b_geom(il);
        let eps = self.cfg.rms_eps;
        let aux = self.gemma4_aux.as_ref().unwrap();
        let Mixer::Full(fa) = &self.layers[il].mixer else { unreachable!() };
        // pre-quantized layer input (E4B fusion port, 2026-07-12): ONE norm+quant feeds
        // wq/wk/wv via matmul_pre — the first-light path quantized the same h three times
        // (the profile's 342 quantize_q8_1/token; glue = 26% of the 6.1ms token).
        let h0 = e.zeros(0)?;
        let h = &h0;

        let ff = if swa { None } else {
            Some(aux.rope_freqs.as_ref().expect("e4b global rope needs rope_freqs.weight"))
        };
        let share = self.gemma4_e4b_kv_target(il);
        // Own-KV arms keep the f32 (k, v) alive for the prime fa arm below.
        let mut kv_f32: Option<(CudaSlice<f32>, CudaSlice<f32>)> = None;
        let mut q;
        if let Some(_tgt) = share {
            let q0 = e.matmul_pre(&fa.wq, hq, hdq, h, t)?;
            q = e.uninit(t * nh * hd)?;
            // Q-only through the same fused norm+rope kernel (rk = 0: the k/v segments are
            // empty; q0 stands in for the unused k/v pointers).
            let mut kdummy = e.uninit(1)?;
            let mut vdummy = e.uninit(1)?;
            e.rms_norm_qkv_rope(&q0, &q0, &q0, fa.q_norm.float_data(),
                                fa.q_norm.float_data(), &aux.ones,
                                &mut q, &mut kdummy, &mut vdummy, hd, nh * t, 0,
                                pos_d, nh, 1, base, 1.0, ff, eps)?;
        } else {
            // wave-4b: ONE concat matvec (wq|wk|wv) at t == 1 when the cat tensor exists;
            // else the fused3 grid launch; else per-matvec. The cat output is contiguous
            // q|k|v rows — the cat norm+rope twin consumes it directly.
            let e4bits = self.layers[il].gemma4.as_ref().and_then(|g| g.e4b.as_ref());
            let cat = e4bits.and_then(|e4| e4.qkv_cat.as_ref());
            q = e.uninit(t * nh * hd)?;
            let mut k = e.uninit(t * nkv * hd)?;
            let mut v = e.uninit(t * nkv * hd)?;
            if t == 1 && cat.is_some() {
                let qkv0 = e.matmul_pre(cat.unwrap(), hq, hdq, h, 1)?;
                e.rms_norm_qkv_rope_cat(&qkv0, fa.q_norm.float_data(), fa.k_norm.float_data(),
                                        &aux.ones, &mut q, &mut k, &mut v, hd, nh, nkv,
                                        pos_d, nh, nkv, base, 1.0, ff, eps)?;
            } else {
                let (q0, k0, v0) = match if t == 1 {
                    e.matmul_q4_fused3(&fa.wq, &fa.wk, &fa.wv, hq, hdq)?
                } else {
                    // E4B verify f3 port (2026-07-14): the 31B segmented-grid batched qkv
                    // on E4B's real-V triple — same MEMRA_F2B seam, bit-identical per row.
                    static F2B_QKV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                    if *F2B_QKV.get_or_init(|| std::env::var("MEMRA_F2B").as_deref() != Ok("0")) {
                        e.matmul_q4_fused3_batched(&fa.wq, &fa.wk, &fa.wv, hq, hdq, t)?
                    } else { None }
                } {
                    Some(triple) => triple,
                    None => (e.matmul_pre(&fa.wq, hq, hdq, h, t)?,
                             e.matmul_pre(&fa.wk, hq, hdq, h, t)?,
                             e.matmul_pre(&fa.wv, hq, hdq, h, t)?),   // E4B: real v (K != V)
                };
                // wave-3 fold: q/k/v norms + q/k rope in ONE launch (rope math verbatim on
                // the normed rows; V ones-rms, never roped).
                e.rms_norm_qkv_rope(&q0, &k0, &v0, fa.q_norm.float_data(),
                                    fa.k_norm.float_data(), &aux.ones, &mut q, &mut k, &mut v,
                                    hd, nh * t, nkv * t, pos_d, nh, nkv, base, 1.0, ff, eps)?;
            }
            let kvl = cache.kv[il].as_mut().unwrap();
            // class flag must match the cache dims (the g-threading sweep hardcoded `false`
            // here and the wkv default corrupted E4B: q8_0 bytes into an e4m3 cache — the
            // degenerate tok-0 stream, 2026-07-12).
            let cls = (!swa && crate::Engine::gkv_on()) || (swa && crate::Engine::wkv_on());
            if dc_bucket.is_some() {
                // DC arm (graph serving): append at the len_d slot, advance the counter
                // in-stream — replay-correct, no host len in the launch args. Host mirrors
                // are NOT touched here (the replay loop owns them; a bump at capture-record
                // time would double-count the capture iteration).
                debug_assert!(t == 1);
                // wave 5c: append + len_d inc fused (one launch; single-block ordering).
                e.append_kv_quantized_row_dc_inc(&k, &v, &mut kvl.k, &mut kvl.v,
                                                 &mut kvl.len_d, kvl.kv_dim_k, kvl.kv_dim_v,
                                                 kvl.k_tok_bytes, kvl.v_tok_bytes, cls)?;
            } else {
                e.append_kv_quantized_rows(&k, &v, &mut kvl.k, &mut kvl.v, kvl.len, t,
                                           kvl.kv_dim_k, kvl.kv_dim_v, kvl.k_tok_bytes,
                                           kvl.v_tok_bytes, cls)?;
                kvl.len += t;
            }
            kv_f32 = Some((k, v));
        }
        // attention: per-row causal fa over the (own or target) quantized cache. The cache
        // already contains this forward's rows in both arms; row i attends [.., base+i].
        let kvl_idx = share.unwrap_or(il);
        let kvl = cache.kv[kvl_idx].as_ref().unwrap();
        let base_len = kvl.len - t;   // pre-append length (target appended this forward too)
        let win = self.cfg.gemma4.as_ref().unwrap().sliding_window as usize;
        let mut attn = e.uninit(t * nh * hd)?;
        // PRIME FA ARMS (perf lane 3, completed 2026-07-31 — the H100 board pinned E4B
        // prefill at 482 tok/s: the per-row loop ran the DECODE kernel once per token per
        // layer). Fresh-prompt (base_len == 0) batched attention, one launch per layer:
        //   - own-KV hd256 under the window: f32 fa_prefill (full causal exact — 26B pattern)
        //   - own-KV hd256 above the window (swa): fa_prefill_w windowed twin (12B pattern)
        //   - own-KV hd512 globals: fa_prefill_hd512 (12B/31B globals pattern)
        //   - KV-shared layers (no f32 k/v): fa_prefill_view over the target's QUANTIZED
        //     rows (the T=K verify kernel; the target appended this forward's rows already).
        //     swa-shared above the window keeps the per-row loop (no windowed view twin).
        // Same numeric class split as the 26B prime (f32/batched prefill vs per-row
        // quantized decode); run-gen argmax + chat gates arbitrate. MEMRA_NOFA=1 reverts all.
        if t > 1 && base_len == 0 && std::env::var("MEMRA_NOFA").is_err() {
            if let Some((kf, vf)) = &kv_f32 {
                if hd == 256 && t <= win {
                    e.fa_prefill(&q, kf, vf, &mut attn, hd, nh, nkv, t, t, scale, true)?;
                    return Ok(e.matmul(&fa.wo, &attn, t)?);
                }
                if hd == 256 && swa && t > win {
                    e.fa_prefill_w(&q, kf, vf, &mut attn, hd, nh, nkv, t, t, scale, true,
                                   win)?;
                    return Ok(e.matmul(&fa.wo, &attn, t)?);
                }
                if hd == 512 && !swa {
                    e.fa_prefill_hd512(&q, kf, vf, &mut attn, hd, nh, nkv, t, t, scale,
                                       true)?;
                    return Ok(e.matmul(&fa.wo, &attn, t)?);
                }
            } else if share.is_some() {
                let g = (!swa && crate::Engine::gkv_on()) || (swa && crate::Engine::wkv_on());
                let k_view = e.view_u8(&kvl.k, kvl.k.len());
                let v_view = e.view_u8(&kvl.v, kvl.v.len());
                if hd == 256 && (!swa || t <= win) {
                    // inline-dequant quantized-view prefill (fa_prefill_q stamps 256/128)
                    e.fa_prefill_view(&q, &k_view, &v_view, &mut attn, hd, nh, nkv, t, t,
                                      scale, true, kvl.k_tok_bytes, kvl.v_tok_bytes, g)?;
                    return Ok(e.matmul(&fa.wo, &attn, t)?);
                }
                // remaining shared classes (swa above the window; hd512 globals): dequant
                // the target's t rows ONCE to f32, then the same f32 twins as own-KV.
                let kv_dim = nkv * hd;
                let mut kf = e.uninit(t * kv_dim)?;
                let mut vf = e.uninit(t * kv_dim)?;
                e.fa_dequant_kv_view_f32(&k_view, &v_view, &mut kf, &mut vf, kv_dim, kv_dim,
                                         t, kvl.k_tok_bytes, kvl.v_tok_bytes, g)?;
                if hd == 512 {
                    e.fa_prefill_hd512(&q, &kf, &vf, &mut attn, hd, nh, nkv, t, t, scale,
                                       true)?;
                } else {
                    e.fa_prefill_w(&q, &kf, &vf, &mut attn, hd, nh, nkv, t, t, scale, true,
                                   win)?;
                }
                return Ok(e.matmul(&fa.wo, &attn, t)?);
            }
        }
        if let Some(bucket) = dc_bucket {
            // DC arm (t == 1, under-window regime — the generate gate enforces it): ONE
            // fa_decode_dc over the live counter. len_d already advanced past this token
            // (t_kv = len_d[0], the cache.rs contract). KV-shared layers read the target's
            // counter (advanced when the target ran earlier in the stack).
            assert!(t == 1);
            // hd512 globals: eager (kvmod) picks the SCALAR unified below the fa512 floor,
            // and under the window every live t_kv sits below it — cap the capture bucket
            // under the floor so fa_decode_dc bakes the same scalar symbol, or the graph
            // rides the dpl16 twin and its numeric class diverges from the dc-eager stream
            // (E4B-GRAPH-GATE 2/64, 2026-07-12).
            let bucket = if hd == 512 && win <= crate::fa512_min_tkv() {
                bucket.min(crate::fa512_min_tkv().saturating_sub(1))
            } else { bucket };
            let k_view = e.view_u8(&kvl.k, kvl.k.len());
            let v_view = e.view_u8(&kvl.v, kvl.v.len());
            let g = (!swa && crate::Engine::gkv_on()) || (swa && crate::Engine::wkv_on());
            // Weight prefetch (SOTA item 3, 2026-07-13): wo's decode plane prefetched into
            // L2 across the fa window — fa reads KV only, the weight DRAM lanes are idle
            // there (E4B valid window +0.65%: 196.8 vs 195.6). Value-free scheduling op,
            // captured into the dc graph like any other launch. Extending the cascade to
            // the ffn gate/up planes measured NEGATIVE (193.9 vs 195.8 — 29MB/layer floods
            // the fill path and evicts still-hot lines); wo-only is the shipped shape.
            // MEMRA_WPF=0 rollback seam.
            if crate::Engine::wpf_level() >= 1 {
                e.prefetch_weight_l2(&fa.wo)?;
            }
            // wave 5b: the combine emits the wo input q8 pair directly — the standalone
            // quantize launch + the f32 attn round-trip fold away (t=1 fast path only).
            if e.uses_q8_1_fast(&fa.wo) {
                let mut oq = e.alloc_i8_uninit(nh * hd)?;
                let mut od = e.zeros(nh * hd / 32)?;
                e.fa_decode_dc_q8(&q, &k_view, &v_view, &mut attn, hd, nh, nkv,
                                  &kvl.len_d, bucket, scale,
                                  kvl.k_tok_bytes, kvl.v_tok_bytes, g,
                                  Some((&mut oq, &mut od)))?;
                return Ok(e.matmul_pre(&fa.wo, &oq, &od, &attn, t)?);
            }
            e.fa_decode_dc(&q, &k_view, &v_view, &mut attn, hd, nh, nkv,
                           &kvl.len_d, bucket, scale,
                           kvl.k_tok_bytes, kvl.v_tok_bytes, g)?;
            return Ok(e.matmul(&fa.wo, &attn, t)?);
        }
        for i in 0..t {
            let avail = base_len + i + 1;
            let (off_tok, t_kv) = if swa && avail > win { (avail - win, win) } else { (0, avail) };
            let k_view = e.view_u8_range(&kvl.k, off_tok * kvl.k_tok_bytes,
                                         (off_tok + t_kv) * kvl.k_tok_bytes);
            let v_view = e.view_u8_range(&kvl.v, off_tok * kvl.v_tok_bytes,
                                         (off_tok + t_kv) * kvl.v_tok_bytes);
            let qv = e.view(&q, t * nh * hd);
            let q_row = qv.slice(i * nh * hd..(i + 1) * nh * hd);
            let mut q_one = e.uninit(nh * hd)?;
            e.copy_view_into(&mut q_one, 0, &q_row, nh * hd)?;
            let mut a_one = e.uninit(nh * hd)?;
            // read class MUST match the append class (globals are e4m3 under gkv): the
            // swa-only flag decoded global-layer e4m3 bytes as q8_0/q5_1 — attention over
            // garbage on EVERY E4B global layer, the cross-mode maxdiff-30 root (2026-07-12).
            e.fa_decode_kvmod(&q_one, &k_view, &v_view, &mut a_one, hd, nh, nkv, t_kv, scale,
                        kvl.k_tok_bytes, kvl.v_tok_bytes,
                        (!swa && crate::Engine::gkv_on())
                            || (swa && crate::Engine::wkv_on()))?;
            e.copy_into(&mut attn, i * nh * hd, &a_one, nh * hd)?;
        }
        Ok(e.matmul(&fa.wo, &attn, t)?)
    }

    /// E4B trunk: embed -> prologue -> layers (attn + dense ffn via the 31B tail_core + the
    /// per-layer-embedding tail + layer scale) -> output_norm. Returns (softcapped logits
    /// device [t, n_vocab], pre-output_norm hidden [t, n_embd]). Appends t rows per own-KV
    /// layer; does NOT advance cache.pos (caller owns pos).
    fn gemma4_e4b_trunk(&self, e: &Engine, tokens: &[u32], pos0: usize, cache: &mut Cache,
                        head_last: bool)
                        -> Result<(CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let n_embd = self.cfg.n_embd as usize;
        let t = tokens.len();
        let pos: Vec<i32> = (0..t).map(|i| (pos0 + i) as i32).collect();
        let pos_d = e.htod_i32(&pos)?;
        let mut x = e.htod(&self.embd.gather(n_embd, tokens))?;
        e.scale_inplace(&mut x, (n_embd as f32).sqrt(), t * n_embd)?;
        let inp_pl = self.gemma4_e4b_inp_pl(e, tokens, &x, t)?;
        self.gemma4_e4b_trunk_core(e, x, inp_pl, &pos_d, t, cache, None, true, head_last)
    }

    /// Layer stack + head over prebuilt (x_scaled, inp_pl, device pos) — everything below
    /// here is device-driven, so the dc arm shares it verbatim (stream identity with the
    /// eager chain by construction: SAME functions, not twins).
    fn gemma4_e4b_trunk_core(&self, e: &Engine, x_in: CudaSlice<f32>, inp_pl: CudaSlice<f32>,
                             pos_d: &CudaSlice<i32>, t: usize, cache: &mut Cache,
                             dc_bucket: Option<usize>, cap_logits: bool, head_last: bool)
                             -> Result<(CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let n_embd = self.cfg.n_embd as usize;
        let eps = self.cfg.rms_eps;
        let n_layer = self.layers.len();
        let mut x = x_in;
        let aux_e4b = self.gemma4_aux.as_ref().unwrap().e4b.as_ref().unwrap();
        let n_epl = aux_e4b.n_epl;

        // cross-layer fusion (2026-07-12 port of the 26B/31B trunk structure): each layer's
        // closing add_scale also EMITS the next layer's attn-normed input pre-quantized
        // q8_1 (add_scale_rms_norm_q8_1); the LAST layer emits through output_norm so the
        // head rides matmul_pre too. First layer's pair comes from a standalone fused
        // norm+quant.
        let mut h_carry: Option<(CudaSlice<i8>, CudaSlice<f32>)> = None;
        for il in 0..n_layer {
            let layer = &self.layers[il];
            let (hq, hdq) = match h_carry.take() {
                Some(p) => p,
                None => e.rms_norm_q8_1(&x, layer.attn_norm.float_data(), n_embd, t, eps)?,
            };
            let o = self.gemma4_e4b_attn(e, il, &hq, &hdq, pos_d, t, cache, dc_bucket)?;
            // dense ffn tail with the post-attn norm FOLDED into its entry (one launch for
            // rms(o, post_attn_norm) + residual add + ffn_norm — glue-fusion lane).
            let bits = layer.gemma4.as_ref().unwrap();
            let e4b = bits.e4b.as_ref().expect("e4b layer bits");
            // glue wave 5: the tail DEFERS its post_ffw norm — the FFN exit fuses
            // rms(f0, post_ffw) + residual add + q8 emit into ONE launch (rms_pre_add_q8_1),
            // killing the rms_norm + add_q8_1 pair per layer. T-GENERIC since 2026-07-13:
            // the fused single-phase reduction is NOT FP-order-identical to the unfused
            // rms_norm+add pair (the original "bit-identical chain" claim was FALSE — the
            // t==1 gate left the batched VERIFY on the unfused chain and split E4B verify
            // from decode by logit maxdiff ~0.45, the greedy tie-flip at depth: gate 135/256.
            // NOFUSE bisect: disabling ONLY this fusion -> verify maxdiff 0.000e0). With the
            // gate dropped, decode AND verify ride the same fused chain — parity by
            // construction, VERIFY-GATE 0.000e0.
            let fuse_exit = e.uses_q8_1_fast(&e4b.inp_gate);
            let (sn, attn_out) = self.gemma4_layer_tail_core_pn(
                e, layer, &o, &x, t, Some(layer.post_attn_norm.float_data()), fuse_exit)?;
            let mut resid = e.uninit(t * n_embd)?;
            // per-layer-embedding tail: resid += rms(proj . (gelu(inp_gate . resid) * inp_pl[il]))
            // wave-2: the residual add emits q8_1 alongside — inp_gate rides matmul_pre.
            // (PLE one-block mega-fusion PROBED NEGATIVE 2026-07-13: argmax-correct but
            // 126 vs 189 tok/s — one SM pulling 0.74MB of weights loses to the multi-block
            // launch chain it replaced; jsonl row. Kernel deleted per doctrine.)
            let g = if fuse_exit {
                // sn here = RAW f0 (post_ffw deferred).
                let (rq, rd) = e.rms_pre_add_q8_1(&sn, bits.post_ffw_norm.float_data(),
                                                  &attn_out, &mut resid, n_embd, t,
                                                  self.cfg.rms_eps)?;
                e.matmul_pre(&e4b.inp_gate, &rq, &rd, &resid, t)?
            } else {
                e.add(&sn, &attn_out, &mut resid, t * n_embd)?;
                e.matmul(&e4b.inp_gate, &resid, t)?
            };
            let mut act = e.uninit(t * n_epl)?;
            let y = if t == 1 && e.uses_q8_1_fast(&e4b.proj) {
                let ipv = e.view(&inp_pl, n_epl * n_layer);
                let row = ipv.slice(il * n_epl..(il + 1) * n_epl);
                let (aq, ad) = e.gelu_tanh_mul_q8_1(&g, &row, &mut act, n_epl, 1)?;
                e.matmul_pre(&e4b.proj, &aq, &ad, &act, t)?
            } else {
                let mut inp_this = e.uninit(t * n_epl)?;
                e.copy_rows_strided(&inp_pl, &mut inp_this, n_epl, t, n_epl * n_layer,
                                    il * n_epl)?;
                e.gelu_tanh_mul(&g, &inp_this, &mut act, t * n_epl)?;
                e.matmul(&e4b.proj, &act, t)?
            };
            // rms(y, post_norm) + (yn + resid)*layer_scale + next-layer norm+quant emit,
            // ONE launch (glue-fusion lane; last layer emits through output_norm).
            let next_norm = if il + 1 < n_layer {
                self.layers[il + 1].attn_norm.float_data()
            } else {
                self.output_norm.float_data()
            };
            let mut xn = e.uninit(t * n_embd)?;
            let pair = e.rms_pre_add_scale_rms_norm_q8_1(&y, e4b.post_norm.float_data(),
                                                         &resid, bits.layer_scale, next_norm,
                                                         &mut xn, n_embd, t, eps)?;
            h_carry = Some(pair);
            x = xn;
        }
        // the head consumes the last layer's fused (output_norm) emit. head_last callers
        // (prime, last_only forward) need only the final row's logits — the all-T head is
        // t*n_vocab of discarded work (E4B prime: ~134ms GEMM + a 2.26GB dtoh kept 1 row).
        let (oq, odq) = h_carry.take().unwrap();
        let h0 = e.zeros(0)?;
        let hm = if head_last { 1 } else { t };
        let (hq, hd) = if head_last && t > 1 {
            let mut q1 = e.uninit_i8(n_embd)?;
            e.dtod_copy_view_i8(&oq.slice((t - 1) * n_embd..t * n_embd), &mut q1)?;
            let nb = n_embd / 32;
            let mut d1 = e.uninit(nb)?;
            e.dtod_copy_view(&odq.slice((t - 1) * nb..t * nb), &mut d1)?;
            (q1, d1)
        } else {
            (oq, odq)
        };
        let mut ld = e.matmul_pre(&self.output, &hq, &hd, &h0, hm)?;
        // softcap is strictly monotonic — greedy (argmax-only) consumers skip it, matching
        // the 26B/31B dc precedent (their dc head goes matmul -> argmax with no cap).
        // Logit-returning callers (host logits / spec prime) keep the capped emit.
        if cap_logits {
            let cap = self.cfg.gemma4.as_ref().unwrap().final_logit_softcapping;
            e.softcap(&mut ld, cap, hm * self.output.out_features())?;
        }
        self.gemma4_suppress(e, &mut ld, hm)?;  // mask both capped and argmax-only consumers
        Ok((ld, x))
    }

    /// E4B batched VERIFY (device tokens, the spec round's t=K+1 step): t rows through the
    /// e4b trunk (per-row causal attention; own-KV layers append t rows host-len, KV-shared
    /// layers ride their targets), per-row device argmax + the POST-output_norm hidden
    /// stack (the drafter's h convention). Advances cache.pos/kvl.len by t — the spec
    /// round rolls back rejected rows (shared layers have no KvLayer, so the plain rewind
    /// covers exactly the layers that appended).
    pub fn gemma4_e4b_decode_step_t_am_dev(&self, e: &Engine, tok_d: &CudaSlice<u32>,
                                                  t: usize, pos0: usize, cache: &mut Cache)
                                                  -> Result<(CudaSlice<u32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let n_embd = self.cfg.n_embd as usize;
        let eps = self.cfg.rms_eps;
        let pos: Vec<i32> = (0..t).map(|i| (pos0 + i) as i32).collect();
        let pos_d = e.htod_i32(&pos)?;
        let embd_gpu = self.embd_gpu.get_or_init(|| {
            e.upload_u8(&self.embd.raw).expect("embed table upload")
        });
        let (qt, rb) = self.embd.qt_and_row_bytes(n_embd);
        let mut x = e.embed_gather_device_td(embd_gpu, tok_d, t, n_embd, qt, rb)?;
        e.scale_inplace(&mut x, (n_embd as f32).sqrt(), t * n_embd)?;
        let inp_pl = self.gemma4_e4b_inp_pl_dev(e, tok_d, &x, t)?;
        let (ld, xp) = self.gemma4_e4b_trunk_core(e, x, inp_pl, &pos_d, t, cache, None, true,
                                                  false)?;
        // softcap is monotonic — the per-row argmax is invariant to it (the trunk's head
        // emit is already capped, matching the eager chain bit-for-bit).
        let n_vocab = self.output.out_features();
        let mut vam = e.stream().alloc_zeros::<u32>(t)?;
        for i in 0..t {
            e.argmax_token_device_col(&ld, i, n_vocab, &mut vam, i)?;
        }
        let mut hn = e.uninit(t * n_embd)?;
        e.rms_norm(&xp, self.output_norm.float_data(), &mut hn, n_embd, t, eps)?;
        cache.pos += t;
        Ok((vam, hn))
    }

    /// E4B verify + host logits + POST-output_norm hidden stack (the short-prompt spec
    /// prime path — mirror of `gemma4_decode_step_t_h`).
    pub(crate) fn gemma4_e4b_decode_step_t_h(&self, e: &Engine, tokens: &[u32], pos0: usize,
                                             cache: &mut Cache)
                                             -> Result<(Vec<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let n_embd = self.cfg.n_embd as usize;
        let eps = self.cfg.rms_eps;
        let t = tokens.len();
        let (ld, xp) = self.gemma4_e4b_trunk(e, tokens, pos0, cache, false)?;
        let mut hn = e.uninit(t * n_embd)?;
        e.rms_norm(&xp, self.output_norm.float_data(), &mut hn, n_embd, t, eps)?;
        cache.pos += t;
        Ok((e.dtoh(&ld)?, hn))
    }

    /// E4B GRAPH-CAPTURABLE dc step: same trunk as the dc step but token_d is updated IN
    /// PLACE (self-feeding replay) and every launch arg is a device counter — pos from
    /// pos_d (inc'd in-stream), KV slots from len_d (advanced in-stream), attention from
    /// fa_decode_dc at `bucket`. Host mirrors (cache.pos / kvl.len) advance in the caller's
    /// replay loop. UNDER-WINDOW regime only (the caller gates pos + budget < window).
    pub fn gemma4_e4b_decode_step_dcg(&self, e: &Engine, token_d: &mut CudaSlice<u32>,
                                      pos_d: &mut CudaSlice<i32>, embd_gpu: &CudaSlice<u8>,
                                      embd_qt: i32, embd_rb: usize, cache: &mut Cache,
                                      n_vocab: usize, bucket: usize)
                                      -> Result<(), Box<dyn std::error::Error>> {
        let n_embd = self.cfg.n_embd as usize;
        let mut x = e.embed_gather_device(embd_gpu, token_d, n_embd, embd_qt, embd_rb)?;
        e.scale_inplace(&mut x, (n_embd as f32).sqrt(), n_embd)?;
        let inp_pl = self.gemma4_e4b_inp_pl_dev(e, token_d, &x, 1)?;
        let (ld, _x) = self.gemma4_e4b_trunk_core(e, x, inp_pl, pos_d, 1, cache, Some(bucket),
                                                  false, false)?;
        e.argmax_token_device_into(&ld, token_d, n_vocab)?;
        e.inc_seqlen(pos_d)?;
        Ok(())
    }

    /// E4B DEVICE-COUNTER decode step (the dc serving arm): token id rides `token_d`, the
    /// greedy argmax lands in the returned device buffer — 4B/token host traffic. The layer
    /// stack is gemma4_e4b_trunk_core, i.e. the SAME functions the eager chain runs (stream
    /// identity by construction, not by twin-kernel parity). Host KV mirrors advance like the
    /// 26B dc-eager arm (window views are host math); len_d stays synced by the caller's
    /// entry sync + the appends here don't read it. Graph capture is NOT wired (no
    /// cap_bucket_max) — the E4B graph arc comes after the perf gates.
    #[allow(clippy::too_many_arguments)]
    pub fn gemma4_e4b_decode_step_dc(&self, e: &Engine, token_d: &CudaSlice<u32>,
                                     pos_d: &mut CudaSlice<i32>, embd_gpu: &CudaSlice<u8>,
                                     embd_qt: i32, embd_rb: usize, cache: &mut Cache,
                                     n_vocab: usize)
                                     -> Result<CudaSlice<u32>, Box<dyn std::error::Error>> {
        let n_embd = self.cfg.n_embd as usize;
        let eps = self.cfg.rms_eps;
        let mut x = e.embed_gather_device(embd_gpu, token_d, n_embd, embd_qt, embd_rb)?;
        e.scale_inplace(&mut x, (n_embd as f32).sqrt(), n_embd)?;
        let inp_pl = self.gemma4_e4b_inp_pl_dev(e, token_d, &x, 1)?;
        let (ld, _x) = self.gemma4_e4b_trunk_core(e, x, inp_pl, pos_d, 1, cache, None, false,
                                                  false)?;
        let mut tok_out = e.stream().alloc_zeros::<u32>(1)?;
        e.argmax_token_device_into(&ld, &mut tok_out, n_vocab)?;
        e.inc_seqlen(pos_d)?;
        cache.pos += 1;
        let _ = eps;
        Ok(tok_out)
    }

    /// E4B eager T=1 decode step (decode_step_h contract): returns (softcapped logits host,
    /// pre-output_norm hidden). Advances cache.pos.
    pub(crate) fn gemma4_e4b_decode_step_h(&self, e: &Engine, token: u32, cache: &mut Cache)
                                           -> Result<(Vec<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let (ld, x) = self.gemma4_e4b_trunk(e, &[token], cache.pos, cache, false)?;
        let logits = e.dtoh(&ld)?;
        cache.pos += 1;
        Ok((logits, x))
    }

    /// E4B batched prime (prime_cache contract: (last-row logits host, h_seed pre-norm last
    /// row, hidden stack)). First-light: the t-wide trunk (per-row attention) — correct, not
    /// fast; the prefill fa arms come later.
    pub(crate) fn gemma4_e4b_prime(&self, e: &Engine, tokens: &[u32], cache: &mut Cache)
                                   -> Result<(Vec<f32>, CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        // Err, not assert (2026-08-07, lane/gemma4-serve-gaps): same served-chunked-prompt
        // process-kill as gemma4_prime — refuse per-request.
        if cache.pos != 0 {
            return Err("e4b prime is fresh-prompt only (v0) — prime the full prompt in one \
                        call or decode tokenwise".into());
        }
        let n_embd = self.cfg.n_embd as usize;
        let t = tokens.len();
        let (ld, x) = self.gemma4_e4b_trunk(e, tokens, 0, cache, true)?;
        cache.pos += t;
        let last = e.dtoh(&ld)?;   // head_last: ld is already the final row only
        let xv = e.view(&x, t * n_embd);
        let row = xv.slice((t - 1) * n_embd..t * n_embd);
        let mut h_seed = e.uninit(n_embd)?;
        e.copy_view_into(&mut h_seed, 0, &row, n_embd)?;
        Ok((last, h_seed, x))
    }

    /// E4B prefill logits (forward contract — no persistent cache; scratch cache internally).
    pub(crate) fn gemma4_e4b_forward(&self, e: &Engine, tokens: &[u32], last_only: bool)
                                     -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let mut cache = Cache::new(e, &self.cfg, tokens.len() + 8)?;
        let (ld, _x) = self.gemma4_e4b_trunk(e, tokens, 0, &mut cache, last_only)?;
        Ok(e.dtoh(&ld)?)   // head_last already reduced to the final row when last_only
    }
}

#[cfg(test)]
mod page_prefetch_tests {
    use super::{
        grouped_worker_prefetch_position, page_prefetch_positions,
        page_prefetch_window_from_values, worker_prefetch_positions,
    };

    #[test]
    fn page_prefetch_window_keeps_existing_opt_in_default() {
        assert_eq!(page_prefetch_window_from_values(false, None), 0);
        assert_eq!(page_prefetch_window_from_values(false, Some("8")), 0);
        assert_eq!(page_prefetch_window_from_values(true, None), 1);
        assert_eq!(page_prefetch_window_from_values(true, Some("bad")), 1);
        assert_eq!(page_prefetch_window_from_values(true, Some("0")), 0);
        assert_eq!(page_prefetch_window_from_values(true, Some("8")), 8);
    }

    #[test]
    fn rolling_page_prefetch_advises_each_future_expert_once() {
        let advised: Vec<_> = (0..7)
            .flat_map(|position| page_prefetch_positions(position, 7, 3))
            .collect();
        assert_eq!(advised, vec![1, 2, 3, 4, 5, 6]);

        let one_ahead: Vec<_> = (0..4)
            .flat_map(|position| page_prefetch_positions(position, 4, 1))
            .collect();
        assert_eq!(one_ahead, vec![1, 2, 3]);
        assert!(page_prefetch_positions(0, 4, 0).is_empty());
    }

    #[test]
    fn grouped_worker_prefetch_primes_first_then_each_known_next_once() {
        assert_eq!(grouped_worker_prefetch_position(0, None), None);
        let positions: Vec<_> = std::iter::once(grouped_worker_prefetch_position(4, None).unwrap())
            .chain((0..4).filter_map(|position| {
                grouped_worker_prefetch_position(4, Some(position))
            }))
            .collect();
        assert_eq!(positions, vec![0, 1, 2, 3]);
        assert_eq!(grouped_worker_prefetch_position(1, Some(0)), None);
    }

    #[test]
    fn rolling_worker_prefetch_primes_current_and_each_future_expert_once() {
        let queued: Vec<_> = (0..8)
            .flat_map(|position| worker_prefetch_positions(position, 8, 5))
            .collect();
        assert_eq!(queued, (0..8).collect::<Vec<_>>());

        let one_at_a_time: Vec<_> = (0..4)
            .flat_map(|position| worker_prefetch_positions(position, 4, 1))
            .collect();
        assert_eq!(one_at_a_time, vec![0, 1, 2, 3]);
        assert!(worker_prefetch_positions(0, 4, 0).is_empty());
    }
}

pub struct G4DcSlots {
    x: CudaSlice<f32>, xn: CudaSlice<f32>, cur: CudaSlice<f32>,
    hq: CudaSlice<i8>, hd_: CudaSlice<f32>,
    q0: CudaSlice<f32>, k0: CudaSlice<f32>, v0: CudaSlice<f32>,
    q: CudaSlice<f32>, k: CudaSlice<f32>, v: CudaSlice<f32>,
    attn: CudaSlice<f32>, o: CudaSlice<f32>,
    attn_out: CudaSlice<f32>, zsh: CudaSlice<f32>,
    zq: CudaSlice<i8>, zd: CudaSlice<f32>,
    gate: CudaSlice<f32>, up: CudaSlice<f32>,
    act: CudaSlice<f32>, actq: CudaSlice<i8>, actd: CudaSlice<f32>,
    f0: CudaSlice<f32>, sn: CudaSlice<f32>,
    hn: CudaSlice<f32>, logits: CudaSlice<f32>,
}
