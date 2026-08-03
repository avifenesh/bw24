//! FFI to the MMQ prefill GEMMs (cu/mmq_fp4.cu + cu/mmq_q45k.cu) — vendored floor kernels.
//!
//! NVFP4: the 5150-pp512 kernel from llama.cpp, ggml-decoupled into a static lib with a C-ABI host
//! launcher. The launcher quantizes the f32 activation to block_fp4_mmq internally (llama's 2-level
//! FP8-e8m0/UE4M3 scale = the accurate W4A8-via-FP8 path that fixes memra's W4A4 maxdiff 1.46), then
//! launches the native mxf4nvf4 block-scale tensor-core mma.
//!
//! Q4_K/Q5_K: llama's k-quant int8-MMA MMQ (dequant to int8 at tile-load, q8_1 DS4 activation with
//! the (d, sum) pair that feeds the k-quant min-offset term, shared m16n8k32 s8 mma inner loop).
//! Replaces the hand-rolled qmatvec_gemm k-quant GEMMs that dominate prefill (32% + 28% busy).
//!
//! All dispatched behind MEMRA_MMQ=1. Always built (no external deps) — unlike cutlass_ffi which is
//! MEMRA_CUTLASS-gated.

use crate::Engine;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

/// Quantize-once seam state (see `Engine::mmq_act_begin`): window epoch + one cached
/// (epoch, act_ptr, m, in_f, D4 scratch) slot. Slot drops (freeing the scratch) on each new window.
static MMQ_ACT_EPOCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[allow(clippy::type_complexity)]
static MMQ_ACT_SLOT: std::sync::Mutex<Option<(u64, u64, usize, usize, CudaSlice<u8>)>> =
    std::sync::Mutex::new(None);
/// Stream-k fixup scratch (lazy; sized once per process — one slot per SM).
static MMQ_FIXUP_SLOT: std::sync::Mutex<Option<cudarc::driver::CudaSlice<u8>>> =
    std::sync::Mutex::new(None);

unsafe extern "C" {
    /// Bytes needed for the block_fp4_mmq activation scratch for (in_f, n_tokens).
    pub fn memra_mmq_nvfp4_act_bytes(in_f: i32, n_tokens: i32) -> usize;
    /// Run the NVFP4 W4A4 MMQ prefill GEMM. y[n_tokens, out_f] = act[n_tokens, in_f] @ W[out_f, in_f]^T.
    ///   W_nvfp4_blocks : raw memra NVFP4 weight rows (block_nvfp4 36B blocks, in_f/64 per row).
    ///   act_f32        : f32 activation [n_tokens, in_f] (contiguous).
    ///   y              : f32 output [n_tokens, out_f].
    ///   act_scratch    : pre-alloc'd quant buffer >= memra_mmq_nvfp4_act_bytes(in_f, n_tokens).
    /// Returns 0 on success, else (1000 + cudaError).
    pub fn memra_mmq_nvfp4(
        w_nvfp4_blocks: *const core::ffi::c_void,
        act_f32: *const f32,
        y: *mut f32,
        in_f: i32,
        out_f: i32,
        n_tokens: i32,
        act_scratch: *mut core::ffi::c_void,
        stream: *mut core::ffi::c_void,
        out_scale: f32,
    ) -> i32;
    /// Same as `memra_mmq_nvfp4`, plus the activation-quantizer selector.
    ///   per_token_scale = 1: two-level scaling (per-token row amax folded into the GEMM epilogue
    ///     + per-sub-block UE4M3). This is what `memra_mmq_nvfp4` does.
    ///   per_token_scale = 0: the v1 sub-block-only quantizer, retained as the numeric oracle so
    ///     kernel-check can measure what the row scale bought, and as the rollback seam.
    pub fn memra_mmq_nvfp4_ex(
        w_nvfp4_blocks: *const core::ffi::c_void,
        act_f32: *const f32,
        y: *mut f32,
        in_f: i32,
        out_f: i32,
        n_tokens: i32,
        act_scratch: *mut core::ffi::c_void,
        stream: *mut core::ffi::c_void,
        out_scale: f32,
        per_token_scale: i32,
    ) -> i32;
    /// Same as `memra_mmq_nvfp4_ex`, plus the residual high-precision channel count.
    ///   residual_k = 0: off.
    ///   residual_k > 0: the k largest-magnitude activation channels (ranked across the batch) are
    ///     zeroed before quantization and their exact f32 contribution is added back as a rank-k
    ///     correction. Requires per_token_scale = 1. Clamped to the kernel's max (16).
    pub fn memra_mmq_nvfp4_ex2(
        w_nvfp4_blocks: *const core::ffi::c_void,
        act_f32: *const f32,
        y: *mut f32,
        in_f: i32,
        out_f: i32,
        n_tokens: i32,
        act_scratch: *mut core::ffi::c_void,
        stream: *mut core::ffi::c_void,
        out_scale: f32,
        per_token_scale: i32,
        residual_k: i32,
    ) -> i32;
    /// Bytes needed for the block_q8_1_mmq activation scratch for the NVFP4 W4A8 path.
    pub fn memra_mmq_nvfp4_w4a8_act_bytes(in_f: i32, n_tokens: i32) -> usize;
    /// Run the NVFP4 W4A8 MMQ prefill GEMM (STAGE 2 accuracy-safe rung). Same fast MMQ tile as
    /// memra_mmq_nvfp4 (W4A4) but the non-Blackwell int8 pair: weight FP4 LUT-dequantized to int8 at
    /// tile-load, activation stays q8_1 int8 (D4, the same quant class as the default int8 GEMM).
    /// `rp`: 0 = GGUF 36B-block weight layout, 1 = A6 split-plane repack (the resident decode
    /// layout). The rp tile loader is a pure address remap of the GGUF loader (same dequant math,
    /// same FP op order) — output is bit-identical either way.
    /// Same contract as memra_mmq_nvfp4 otherwise. Returns 0 or (1000 + cudaError).
    pub fn memra_mmq_nvfp4_w4a8(
        w_nvfp4_blocks: *const core::ffi::c_void,
        act_f32: *const f32,
        y: *mut f32,
        in_f: i32,
        out_f: i32,
        n_tokens: i32,
        act_scratch: *mut core::ffi::c_void,
        stream: *mut core::ffi::c_void,
        out_scale: f32,
        rp: i32,
    ) -> i32;
    /// Bytes for the block_e4m3_mmq activation scratch (footprint-identical to block_q8_1_mmq).
    pub fn memra_mmq_nvfp4_f8f4_act_bytes(in_f: i32, n_tokens: i32) -> usize;
    /// R-B W4A8-FP8 MMQ prefill GEMM (research/prefill-mxf8f6f4-design.md): NVFP4 per-16 scales
    /// fold into e4m3 weight VALUES at tile load; e4m3 activations; ONE kind::f8f6f4 m16n8k32
    /// MMA (381-TF class) where the int8 path issues two imma k16. NEW NUMERIC CONFIG — own
    /// battery. Same contract/rp semantics as memra_mmq_nvfp4_w4a8. Returns 0 / 1000+cudaError /
    /// 2000+cudaError.
    pub fn memra_mmq_nvfp4_f8f4(
        w_nvfp4_blocks: *const core::ffi::c_void,
        act_f32: *const f32,
        y: *mut f32,
        in_f: i32,
        out_f: i32,
        n_tokens: i32,
        act_scratch: *mut core::ffi::c_void,
        stream: *mut core::ffi::c_void,
        out_scale: f32,
        rp: i32,
    ) -> i32;
    /// Bytes needed for the block_q8_1_mmq activation scratch (shared by Q4_K and Q5_K).
    pub fn memra_mmq_q45k_act_bytes(in_f: i32, n_tokens: i32) -> usize;
    /// Run the Q4_K W4A8 MMQ prefill GEMM. Same contract as memra_mmq_nvfp4 (raw ggml block_q4_K
    /// weight rows, in_f/256 144B superblocks per row). Returns 0 or (1000 + cudaError).
    pub fn memra_mmq_q4_K(
        w_q4k_blocks: *const core::ffi::c_void,
        act_f32: *const f32,
        y: *mut f32,
        in_f: i32,
        out_f: i32,
        n_tokens: i32,
        act_scratch: *mut core::ffi::c_void,
        stream: *mut core::ffi::c_void,
    ) -> i32;
    /// Run the Q5_K W4A8 MMQ prefill GEMM (176B superblocks). Same contract as memra_mmq_q4_K.
    pub fn memra_mmq_q5_K(
        w_q5k_blocks: *const core::ffi::c_void,
        act_f32: *const f32,
        y: *mut f32,
        in_f: i32,
        out_f: i32,
        n_tokens: i32,
        act_scratch: *mut core::ffi::c_void,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    /// Bytes needed for the block_q8_1_mmq (D4) activation scratch for the Q8_0 MMQ path.
    pub fn memra_mmq_q8_0_act_bytes(in_f: i32, n_tokens: i32) -> usize;
    /// Run the Q8_0 int8-MMA MMQ prefill GEMM (MEMRA_PP_Q8MMQ). Conventional xy-tiling only (no fixup
    /// scratch). Weight = raw ggml block_q8_0 rows (34B blocks, in_f/32 per row); activation is
    /// quantized internally to q8_1 D4. Requires in_f % 32 == 0. Returns 0 or (1000 + cudaError).
    pub fn memra_mmq_q8_0(
        w_q8_0_blocks: *const core::ffi::c_void,
        act_f32: *const f32,
        y: *mut f32,
        in_f: i32,
        out_f: i32,
        n_tokens: i32,
        act_scratch: *mut core::ffi::c_void,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    /// Bytes needed for the block_q8_1_mmq (D4) activation scratch for the Q4_0 MMQ path.
    pub fn memra_mmq_q4_0_act_bytes(in_f: i32, n_tokens: i32) -> usize;
    /// Run the Q4_0 int8-MMA MMQ prefill GEMM (MEMRA_PP_Q4MMQ). Nibbles dequant to int8 at
    /// tile-load (the -8 zero-point folds into the quants, D4 epilogue — same accuracy class as
    /// the Q8_0 MMQ). `rp`: 0 = raw ggml 18B blocks, 1 = MEMRA_Q4RP split-plane repack (qs plane +
    /// fp16 d plane) — pure address remap, bit-identical output either way. Requires
    /// in_f % 32 == 0. Returns 0 or (1000 + cudaError).
    pub fn memra_mmq_q4_0(
        w_q4_0: *const core::ffi::c_void,
        act_f32: *const f32,
        y: *mut f32,
        in_f: i32,
        out_f: i32,
        n_tokens: i32,
        act_scratch: *mut core::ffi::c_void,
        stream: *mut core::ffi::c_void,
        rp: i32,
    ) -> i32;
    /// Quantize-only entry (quantize-once seam): f32 activation -> block_q8_1_mmq scratch.
    pub fn memra_mmq_q4_0_quant_act(
        act_f32: *const f32,
        act_scratch: *mut core::ffi::c_void,
        in_f: i32,
        n_tokens: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
    /// GEMM-only entry: consumes a pre-quantized scratch (from memra_mmq_q4_0_quant_act).
    pub fn memra_mmq_q4_0_gemm(
        w_q4_0: *const core::ffi::c_void,
        act_scratch: *const core::ffi::c_void,
        y: *mut f32,
        in_f: i32,
        out_f: i32,
        n_tokens: i32,
        stream: *mut core::ffi::c_void,
        rp: i32,
    ) -> i32;
    /// Stream-k fixup scratch bytes (one [MMQ_X x MMQ_Y] f32 slot per SM).
    pub fn memra_mmq_q4_0_fixup_bytes() -> usize;
    /// Force the CLC work-stealing arm: 1 = on, 0 = off (static grid), -1 = MEMRA_MMQ_CLC env
    /// default. Schedule-only swap of the xy-tiling kernel — bit-identical output by
    /// construction (perf-frontier lever #1). Returns 1 when the CLC kernel is compiled in
    /// (SM_100+ gencode), 0 on sm_89/90a builds (force is a no-op there; static grid always).
    pub fn memra_mmq_q4_0_set_clc(force: i32) -> i32;
    /// Stream-k GEMM entry: tiling when wave efficiency >= 90%, else stream-k + fixup.
    pub fn memra_mmq_q4_0_gemm_sk(
        w_q4_0: *const core::ffi::c_void,
        act_scratch: *const core::ffi::c_void,
        y: *mut f32,
        fixup_scratch: *mut core::ffi::c_void,
        in_f: i32,
        out_f: i32,
        n_tokens: i32,
        stream: *mut core::ffi::c_void,
        rp: i32,
    ) -> i32;

    // ---- IQ3_S / IQ4_XS expert-segmented int8-MMA MMQ (cu/mmq_iq_experts.cu, MEMRA_MOE_MMA) ----
    /// Bytes for the token-major block_q8_1_mmq activation scratch (in_f, n_tokens).
    pub fn memra_mmq_iq_experts_act_bytes(in_f: i32, n_tokens: i32) -> usize;
    /// Quantize token-major f32 activation [n_tokens, in_f] -> block_q8_1_mmq (D4). Returns 0 or 1000+err.
    pub fn memra_mmq_iq_quantize_act(
        act_f32: *const f32,
        act_scratch: *mut core::ffi::c_void,
        in_f: i32,
        n_tokens: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
    /// Fused act-epilogue: silu/gelu(gate)*up + q8_1_mmq (D4) quantize in ONE launch — no f32 act
    /// buffer. gate/up pair-major [n_tokens, in_f]; scratch identical to memra_mmq_iq_quantize_act.
    /// act_kind: 0=silu*mul, 1=gelu_tanh*mul. Byte-identical to the two-pass path (kernel-check gated).
    pub fn memra_mmq_iq_fused_act_quant(
        gate: *const f32,
        up: *const f32,
        act_scratch: *mut core::ffi::c_void,
        in_f: i32,
        n_tokens: i32,
        act_kind: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
    /// Expert-segmented IQ MMA MMQ. Same CSR shape as moe_pairs_matvec_q8_dec: `table` = [3,n_expert]
    /// device slab ptrs, CSR ex_ids/ex_off/ex_pairs group pairs by expert, pair_tok gathers the
    /// activation row. y = [n_pairs, out_f] pair-major. `act_scratch` pre-quantized over n_tokens.
    /// qtype: 5=IQ4_XS, 6=IQ3_S. Returns 0 or 1000+cudaError.
    /// Dense-trunk IQ4_XS MMQ (lane/kquant-tile-loaders): the dense analog of the expert
    /// kernel for non-expert IQ4_XS 2-D matmuls (the KAT-Coder trunk class). Quantizes the
    /// f32 activation to D4 q8_1_mmq internally; `act_scratch` sized by
    /// `memra_mmq_iq_experts_act_bytes`. Requires in_f % 256 == 0.
    pub fn memra_mmq_iq4xs_dense(
        w_blocks: *const core::ffi::c_void,
        act_f32: *const f32,
        y: *mut f32,
        in_f: i32,
        out_f: i32,
        n_tokens: i32,
        row_bytes: i64,
        act_scratch: *mut core::ffi::c_void,
        stream: *mut core::ffi::c_void,
    ) -> i32;
    pub fn memra_mmq_iq_experts(
        table: *const u64,
        proj: i32,
        n_expert: i32,
        ex_ids: *const i32,
        ex_off: *const i32,
        ex_pairs: *const i32,
        pair_tok: *const i32,
        act_scratch: *const core::ffi::c_void,
        y: *mut f32,
        in_f: i32,
        out_f: i32,
        n_active: i32,
        n_tokens: i32,
        qtype: i32,
        row_bytes: i64,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    // ---- MoE grouped f16 GEMM (cu/moe_f16_grouped.cu, round 46 arc 2) ----
    pub fn memra_moe_f16g_dequant(
        table: *const u64, proj: i32, n_expert: i32, ex_ids: *const i32,
        w_f16: *mut core::ffi::c_void, in_f: i32, out_f: i32, n_active: i32,
        qtype: i32, row_bytes: i64, stream: *mut core::ffi::c_void,
    ) -> i32;
    pub fn memra_moe_f16g_gather_act(
        x: *const f32, pair_tok_or_null: *const i32, act_f16: *mut core::ffi::c_void,
        row_scale: *mut f32, in_f: i32, n_pairs: i32, stream: *mut core::ffi::c_void,
    ) -> i32;
    pub fn memra_moe_f16g_h2f_scaled(
        src_f16: *const core::ffi::c_void, dst: *mut f32, row_scale: *const f32,
        ncols: i32, nrows: i32, stream: *mut core::ffi::c_void,
    ) -> i32;
    pub fn memra_moe_f16g_gemm(
        w_f16: *const core::ffi::c_void, act_f16: *const core::ffi::c_void,
        y_f16: *mut core::ffi::c_void,
        ex_off_host: *const i32, n_active: i32, in_f: i32, out_f: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
    pub fn memra_moe_f16g_h2f(
        src_f16: *const core::ffi::c_void, dst: *mut f32, n: usize,
        stream: *mut core::ffi::c_void,
    ) -> i32;
    // Single-kernel grouped GEMM (MEMRA_MOE_F16G=2, rounds 49+51): on OUR stream, f32 C with
    // the act row-scale folded in — no cublas internal-stream race, no sync. Round 51 runs it
    // as a persistent problem-visitor over the real tiles with two tile forms (32x64 tail
    // / 128x64x64 3-stage): shape_sel < 0 = the round-49 grid-scan kernel (rollback
    // arm); else groups with m_e >= cross ride the 128 form. ex_off_host sizes the visitor
    // grids host-side (the offsets are already there at the call site — no extra transfer).
    // tail != 0 (lane/sk-tail-form): sub-cross groups ride the DEEP tail (32x64x64 3-stage);
    // 0 = the round-51 2-stage 32x64x32 (MEMRA_F16G_TAIL=0 rollback). Byte-identical arms.
    pub fn memra_moe_f16g_gemm_sk(
        w_f16: *const core::ffi::c_void, act_f16: *const core::ffi::c_void,
        y_f32: *mut f32, row_scale: *const f32, ex_off_dev: *const i32,
        ex_off_host: *const i32,
        n_active: i32, max_m: i32, in_f: i32, out_f: i32, shape_sel: i32, cross: i32,
        tail: i32, stream: *mut core::ffi::c_void,
    ) -> i32;
    // DIRECT-FROM-QUANT sk visitor grouped GEMM (lane/kquant-tile-loaders + iq-direct-loaders):
    // the visitor forms with the B (weight) tiles dequanted in-register from the expert
    // superblocks — no f16 dequant workspace pass. Bit-identical to the workspace path by
    // construction (kernel-check "f16g-kq-direct"). qtype: QT_Q4_K | QT_Q6_K | QT_IQ4_XS |
    // QT_IQ3_S; rc=2 = not admitted here (caller keeps the dequant-workspace path).
    // tail: as memra_moe_f16g_gemm_sk.
    pub fn memra_moe_kq_gemm_sk(
        table: *const u64, proj: i32, n_expert: i32, ex_ids: *const i32,
        act_f16: *const core::ffi::c_void, y_f32: *mut f32,
        row_scale: *const f32, ex_off_dev: *const i32, ex_off_host: *const i32,
        n_active: i32, max_m: i32, in_f: i32, out_f: i32, qtype: i32, cross: i32,
        tail: i32, row_bytes: i64, stream: *mut core::ffi::c_void,
    ) -> i32;
}

/// W4A8-MMQ DEFAULT-FLIP seam (2026-07-05): the vendored MMQ prefill suite is DEFAULT-ON — NVFP4
/// takes the W4A8 MMQ tile (same int8 accuracy class as the int8 GEMM it replaces, all exactness
/// gates hold, ~1.9x pp512; the rp tile-loader arm coexists with the A6 split-plane repack) and
/// Q4_K/Q5_K take the vendored k-quant int8-MMA MMQ (also int8-class; gated with W4A8 in the same
/// battery — the predecessor's `MEMRA_MMQ_W4A8=1` arm engaged BOTH, this flip preserves exactly
/// that measured config). `MEMRA_MMQ_W4A8=0` = escape hatch back to the int8 GEMM prefill
/// everywhere. `MEMRA_MMQ=1` additionally switches GGUF-layout NVFP4 to the W4A4 mxf4nvf4 tile
/// (speed/accuracy tradeoff opt-in, unchanged).
pub fn mmq_w4a8_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("MEMRA_MMQ_W4A8")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

/// Residual high-precision activation channels for the W4A4 MMQ prefill path.
/// `MEMRA_MMQ_RESIDUAL_K=<k>` keeps the k largest-magnitude activation channels out of the e2m1
/// quantized path and adds their exact f32 contribution back as a rank-k correction. k=0 (default)
/// is off; the kernel clamps to 16.
///
/// Read LIVE per call, not OnceLock'd, for the same reason `MEMRA_MMQ` is: the W4A4 exactness gate
/// sweeps arms inside ONE process against ONE set of loaded weights, and a cached first read would
/// pin every later arm to whatever the first one saw.
pub fn mmq_residual_k() -> i32 {
    std::env::var("MEMRA_MMQ_RESIDUAL_K")
        .ok()
        .and_then(|v| v.parse::<i32>().ok())
        .unwrap_or(0)
        .clamp(0, 16)
}

/// Q8_0 MMQ prefill seam (lane/ppmmq lever 2, DEFAULT ON since 2026-07-09 — `MEMRA_PP_Q8MMQ=0`
/// reverts): routes Q8_0 dense
/// projections (m>=16) through the vendored int8-MMA MMQ (cu/mmq_q8_0.cu) instead of the hand-rolled
/// `qmatvec_gemm_q8_0` tiling GEMM. Its own numeric config (MMA f32 reduction order != the tiling
/// GEMM's) — gated with the full exactness battery. Default OFF until the battery is green.
pub fn mmq_q8_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // Promotion battery (2026-07-09): argmax MATCH on 35B p1/p2/p3 + 9B p2/p3 (p4-16k OOMs
    // identically with and without the flag — pre-existing gate capacity limit, not this seam);
    // kernel-check ALL GREEN; run-spec K=1..8 PASS on 9B+35B. 35B pp 2456->3069 free-clock.
    *ON.get_or_init(|| {
        std::env::var("MEMRA_PP_Q8MMQ")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

/// IQ4_XS dense-trunk MMQ prefill seam (lane/kquant-tile-loaders, 2026-08-02): routes
/// NON-expert IQ4_XS 2-D projections (m>=16) through the vendored-machinery int8-MMA dense
/// MMQ (cu/mmq_iq_experts.cu `mmq_iq4xs_dense_kernel`) instead of the per-column dp4a grid
/// — the KAT-Coder prefill wall (0.169x vs llama; zero weight reuse across tokens,
/// research/kat-anomaly-20260802 §6). Its own numeric config (MMA reduction order) — gated
/// with the full exactness battery. m=1..15 decode/verify keep dp4a (dispatch parity).
/// `MEMRA_PP_IQMMQ=0` reverts.
pub fn mmq_iq4xs_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("MEMRA_PP_IQMMQ")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

/// Q4_0 MMQ prefill seam (gemma-4-12B lane, 2026-07-22): routes Q4_0 dense projections (m>=16)
/// through the vendored int8-MMA MMQ (cu/mmq_q4_0.cu) instead of the hand-rolled
/// `qmatvec_gemm_q4_0[_rp]` tiling GEMM (measured 77% of the 12B prime pass). Its own numeric
/// config (MMA f32 reduction order != the tiling GEMM's) — gated with the full exactness battery
/// before default-flip; `MEMRA_PP_Q4MMQ=0` reverts.
pub fn mmq_q4_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("MEMRA_PP_Q4MMQ")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

impl Engine {
    /// True if `w` should take a vendored MMQ GEMM under the current env policy (see
    /// `mmq_w4a8_enabled`): NVFP4 needs in_f % 64 == 0, Q4_K/Q5_K need in_f % 256 == 0.
    pub fn mmq_supports(&self, w: &crate::model::GpuTensor) -> bool {
        use crate::model::GpuTensor;
        if crate::portable_mma_gated() {
            return false;
        }
        let mmq_opt_in = std::env::var("MEMRA_MMQ").is_ok();
        match w {
            // A6 split-plane repacked NVFP4: ONLY the W4A8 loader has an rp arm (pure address
            // remap, bit-identical output — mmq_nvfp4_w4a8.cu load_tiles_nvfp4_w4a8<is_rp>).
            // The W4A4 loader (mmq_fp4.cu load_tiles_nvfp4_nvfp4) reads 36B GGUF blocks only,
            // so an rp weight with W4A8 disabled falls through to the rp-ported int8 GEMM.
            // NVFP4 W4A8/W4A4 launchers use .kind::f8f6f4 / mxf4nvf4 tile MMA — sm_100a+/
            // sm_120a-only. On every portable build (incl. the 90a Hopper-MMA lane) they are
            // fail-closed link stubs (build.rs), so never offer them here.
            GpuTensor::Quant { qtype, rp, .. } if *qtype == crate::QT_NVFP4 && *rp => {
                !cfg!(memra_portable_cuda) && mmq_w4a8_enabled() && w.in_features() % 64 == 0
            }
            // GGUF-layout NVFP4 (MEMRA_RP=0): W4A8 (default-on) or the explicit W4A4 opt-in.
            GpuTensor::Quant { qtype, .. } if *qtype == crate::QT_NVFP4 => {
                !cfg!(memra_portable_cuda) && (mmq_w4a8_enabled() || mmq_opt_in)
                    && w.in_features() % 64 == 0
            }
            GpuTensor::Quant { qtype, .. }
                if *qtype == crate::QT_Q4_K || *qtype == crate::QT_Q5_K =>
            {
                (mmq_w4a8_enabled() || mmq_opt_in) && w.in_features() % 256 == 0
            }
            // Q8_0 dense projections (35B attn/ssm/shexp): opt-in only (MEMRA_PP_Q8MMQ=1), its own
            // numeric config vs qmatvec_gemm_q8_0. in_f % 256 == 0: MMQ_ITER_K=256 loads 8-block
            // groups, so a non-multiple row would read a garbage weight tail (fp16 d bytes can be
            // NaN-pattern, and NaN * 0-padded-activation = NaN — the 26B ffn_down lesson).
            GpuTensor::Quant { qtype, .. } if *qtype == crate::QT_Q8_0 => {
                mmq_q8_enabled() && w.in_features() % 256 == 0
            }
            // Q4_0 dense projections (gemma QAT ggufs): MEMRA_PP_Q4MMQ seam. Both weight layouts
            // (raw 18B blocks and the MEMRA_Q4RP split-plane repack) have loader arms. Same
            // in_f % 256 == 0 tail rule as Q8_0 (26B ffn_down in_f=2112 NaN'd on the %32 gate);
            // non-multiples fall back to the hand-rolled qmatvec_gemm_q4_0[_rp].
            GpuTensor::Quant { qtype, .. } if *qtype == crate::QT_Q4_0 => {
                mmq_q4_enabled() && w.in_features() % 256 == 0
            }
            // IQ4_XS dense projections (KAT-Coder trunk): m>=16 prefill only — decode and
            // spec-verify (m<16) keep the qmatvec_iq4_XS_dp4a per-column program (the
            // kat-anomaly dispatch-parity law). Requires the dp4a fast path itself enabled:
            // MEMRA_IQ_FAST=0 (the Stage-A oracle rollback) must also kill this arm so the
            // rollback stays a full-path seam. in_f % 256: MMQ_ITER_K walks whole superblocks.
            GpuTensor::Quant { qtype, .. } if *qtype == crate::QT_IQ4_XS => {
                mmq_iq4xs_enabled() && Self::iq_fast_enabled() && w.in_features() % 256 == 0
            }
            _ => false,
        }
    }

    /// Unified vendored-MMQ dispatch: routes to the NVFP4 or Q4_K/Q5_K launcher by qtype.
    /// Caller MUST have checked `mmq_supports(w)`. `x` is the RAW f32 activation.
    pub fn qmatvec_mmq(
        &self,
        w: &crate::model::GpuTensor,
        x: &CudaSlice<f32>,
        m: usize,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        use crate::model::GpuTensor;
        let (in_f, out_f) = (w.in_features(), w.out_features());
        let GpuTensor::Quant {
            bytes,
            scale,
            qtype,
            rp,
            ..
        } = w
        else {
            return Err("qmatvec_mmq: not a Quant tensor".into());
        };
        // NVFP4 tile choice: W4A8 (accuracy-safe int8 pair, DEFAULT since the flip) vs W4A4
        // (mxf4nvf4 mma, explicit MEMRA_MMQ=1 speed/accuracy tradeoff). An rp weight ALWAYS takes
        // W4A8 — only its loader has the split-plane arm (pure address remap, bit-identical).
        // Explicit MEMRA_MMQ_W4A8=1 still overrides a simultaneous MEMRA_MMQ=1 (predecessor rule).
        let w4a8_explicit = std::env::var("MEMRA_MMQ_W4A8")
            .map(|v| v != "0")
            .unwrap_or(false);
        let use_w4a8 =
            *rp || w4a8_explicit || (mmq_w4a8_enabled() && std::env::var("MEMRA_MMQ").is_err());
        match *qtype {
            // STAGE 2: the accuracy-safe int8 W4A8 MMQ tile (weight FP4->int8 dequant + q8_1
            // activation) — handles BOTH weight layouts (rp = A6 split-plane vs GGUF blocks).
            q if q == crate::QT_NVFP4 && use_w4a8 => {
                self.qmatvec_mmq_nvfp4_w4a8(bytes, x, m, in_f, out_f, *scale, *rp)
            }
            q if q == crate::QT_NVFP4 => self.qmatvec_mmq_nvfp4(bytes, x, m, in_f, out_f, *scale),
            q if q == crate::QT_Q4_K || q == crate::QT_Q5_K => {
                let mut y = self.qmatvec_mmq_q45k_raw(bytes, x, m, in_f, out_f, q)?;
                if *scale != 1.0 {
                    self.scale_inplace(&mut y, *scale, m * out_f)?;
                }
                Ok(y)
            }
            q if q == crate::QT_Q8_0 => {
                // wgmma arm (sm_90a, task 8): OPT-IN via MEMRA_WGMMA=1 — v0 measured 3845
                // vs MMQ 8692 tok/s pp512 (2026-07-26 N=5), so MMQ stays the default until
                // the pipelined wgmma wins. Reads the rp4 split-plane mirror + the engine's
                // q8_1 activation planes. Same numeric class as MMQ (exact s32 per 32-block,
                // one f32 fold per block, ascending K) — kernel-check tolerance-gated.
                if cfg!(memra_hopper_mma) && out_f % 64 == 0 && crate::wgmma_gemm_enabled() {
                    if let GpuTensor::Quant { rp4: Some(m4), .. } = w {
                        let (aq, ad) = self.quantize_q8_1(x, m, in_f)?;
                        let mut y = self.qmatvec_gemm_q8_0_wgmma_raw(m4, &aq, &ad, m, in_f, out_f)?;
                        if *scale != 1.0 {
                            self.scale_inplace(&mut y, *scale, m * out_f)?;
                        }
                        return Ok(y);
                    }
                }
                let mut y = self.qmatvec_mmq_q8_0_raw(bytes, x, m, in_f, out_f)?;
                if *scale != 1.0 {
                    self.scale_inplace(&mut y, *scale, m * out_f)?;
                }
                Ok(y)
            }
            q if q == crate::QT_Q4_0 => {
                let mut y = self.qmatvec_mmq_q4_0_raw(bytes, x, m, in_f, out_f, *rp)?;
                if *scale != 1.0 {
                    self.scale_inplace(&mut y, *scale, m * out_f)?;
                }
                Ok(y)
            }
            q if q == crate::QT_IQ4_XS => {
                let GpuTensor::Quant { row_bytes, .. } = w else { unreachable!() };
                let mut y = self.qmatvec_mmq_iq4xs_raw(bytes, x, m, in_f, out_f, *row_bytes)?;
                if *scale != 1.0 {
                    self.scale_inplace(&mut y, *scale, m * out_f)?;
                }
                Ok(y)
            }
            q => Err(format!("qmatvec_mmq: unsupported qtype {q}").into()),
        }
    }

    /// Bare IQ4_XS dense MMQ launch (no macro-scale) — also the kernel_check gate entry.
    pub fn qmatvec_mmq_iq4xs_raw(
        &self,
        bytes: &CudaSlice<u8>,
        x: &CudaSlice<f32>,
        m: usize,
        in_f: usize,
        out_f: usize,
        row_bytes: usize,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        assert!(
            in_f % 256 == 0,
            "MMQ IQ4_XS requires in_f % 256 == 0, got {in_f}"
        );
        let act_bytes = unsafe { memra_mmq_iq_experts_act_bytes(in_f as i32, m as i32) };
        let mut scratch = self.alloc_uninit::<u8>(act_bytes)?;
        let mut y = self.alloc_uninit::<f32>(m * out_f)?;
        {
            let stream = self.gpu.stream();
            let (w_p, _gw) = bytes.device_ptr(&stream);
            let (x_p, _gx) = x.device_ptr(&stream);
            let (y_p, _gy) = y.device_ptr_mut(&stream);
            let (s_p, _gs) = scratch.device_ptr_mut(&stream);
            let rc = unsafe {
                memra_mmq_iq4xs_dense(
                    w_p as *const core::ffi::c_void,
                    x_p as *const f32,
                    y_p as *mut f32,
                    in_f as i32,
                    out_f as i32,
                    m as i32,
                    row_bytes as i64,
                    s_p as *mut core::ffi::c_void,
                    stream.cu_stream() as *mut core::ffi::c_void,
                )
            };
            if rc != 0 {
                return Err(format!("memra_mmq_iq4xs_dense rc={rc}").into());
            }
        }
        Ok(y)
    }

    /// Bare Q4_K/Q5_K MMQ launch (no macro-scale) — also the kernel_check accuracy-gate entry.
    /// Conventional xy-tiling only (the vendored stream-K arm — MEMRA_MMQ_STREAMK — was removed
    /// 2026-07-08: 1.11x per-GEMM but its k-split f32 reorder flipped the model argmax gate;
    /// rig5090.jsonl 2026-07-03 has the record).
    pub fn qmatvec_mmq_q45k_raw(
        &self,
        bytes: &CudaSlice<u8>,
        x: &CudaSlice<f32>,
        m: usize,
        in_f: usize,
        out_f: usize,
        qtype: i32,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        assert!(
            in_f % 256 == 0,
            "MMQ Q4_K/Q5_K requires in_f % 256 == 0, got {in_f}"
        );
        let act_bytes = unsafe { memra_mmq_q45k_act_bytes(in_f as i32, m as i32) };
        let mut scratch = self.alloc_uninit::<u8>(act_bytes)?;
        let mut y = self.alloc_uninit::<f32>(m * out_f)?;
        {
            let stream = self.gpu.stream();
            let (w_p, _gw) = bytes.device_ptr(&stream);
            let (x_p, _gx) = x.device_ptr(&stream);
            let (y_p, _gy) = y.device_ptr_mut(&stream);
            let (s_p, _gs) = scratch.device_ptr_mut(&stream);
            let launcher = if qtype == crate::QT_Q4_K {
                memra_mmq_q4_K
            } else {
                memra_mmq_q5_K
            };
            let rc = unsafe {
                launcher(
                    w_p as *const core::ffi::c_void,
                    x_p as *const f32,
                    y_p as *mut f32,
                    in_f as i32,
                    out_f as i32,
                    m as i32,
                    s_p as *mut core::ffi::c_void,
                    stream.cu_stream() as *mut core::ffi::c_void,
                )
            };
            if rc != 0 {
                return Err(format!("memra_mmq_q45k(qtype={qtype}) rc={rc}").into());
            }
        }
        Ok(y)
    }

    /// Bare Q8_0 int8-MMA MMQ launch (no macro-scale) — the kernel_check accuracy-gate entry and
    /// the `qmatvec_mmq` dispatch body. Conventional xy-tiling only (no stream-K / fixup scratch).
    pub fn qmatvec_mmq_q8_0_raw(
        &self,
        bytes: &CudaSlice<u8>,
        x: &CudaSlice<f32>,
        m: usize,
        in_f: usize,
        out_f: usize,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        assert!(
            in_f % 32 == 0,
            "MMQ Q8_0 requires in_f % 32 == 0, got {in_f}"
        );
        let act_bytes = unsafe { memra_mmq_q8_0_act_bytes(in_f as i32, m as i32) };
        let mut scratch = self.alloc_uninit::<u8>(act_bytes)?;
        let mut y = self.alloc_uninit::<f32>(m * out_f)?;
        {
            let stream = self.gpu.stream();
            let (w_p, _gw) = bytes.device_ptr(&stream);
            let (x_p, _gx) = x.device_ptr(&stream);
            let (y_p, _gy) = y.device_ptr_mut(&stream);
            let (s_p, _gs) = scratch.device_ptr_mut(&stream);
            let rc = unsafe {
                memra_mmq_q8_0(
                    w_p as *const core::ffi::c_void,
                    x_p as *const f32,
                    y_p as *mut f32,
                    in_f as i32,
                    out_f as i32,
                    m as i32,
                    s_p as *mut core::ffi::c_void,
                    stream.cu_stream() as *mut core::ffi::c_void,
                )
            };
            if rc != 0 {
                return Err(format!("memra_mmq_q8_0 rc={rc}").into());
            }
        }
        Ok(y)
    }

    /// Open a quantize-once sharing window for the NEXT activation (quantize-once seam): sibling
    /// Q4_0 MMQ matmuls on the SAME input (q/k/v; gate/up) quantize its D4 scratch once. Safe by
    /// construction: a hit requires the same window epoch AND the same (ptr, m, in_f) — the caller
    /// opens a window while it holds the shared input alive, so its address can neither change nor
    /// be recycled inside the window. Paths that never call this never hit the cache.
    pub fn mmq_act_begin(&self) {
        use std::sync::atomic::Ordering;
        MMQ_ACT_EPOCH.fetch_add(1, Ordering::Relaxed);
        *MMQ_ACT_SLOT.lock().unwrap() = None;
    }

    /// Bare Q4_0 int8-MMA MMQ launch (no macro-scale) — the kernel_check accuracy-gate entry and
    /// the `qmatvec_mmq` dispatch body. `rp` selects the weight layout (MEMRA_Q4RP split-plane vs
    /// raw ggml 18B blocks) — pure address remap, bit-identical output.
    pub fn qmatvec_mmq_q4_0_raw(
        &self,
        bytes: &CudaSlice<u8>,
        x: &CudaSlice<f32>,
        m: usize,
        in_f: usize,
        out_f: usize,
        rp: bool,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        use std::sync::atomic::Ordering;
        assert!(
            in_f % 32 == 0,
            "MMQ Q4_0 requires in_f % 32 == 0, got {in_f}"
        );
        let mut y = self.alloc_uninit::<f32>(m * out_f)?;
        let stream = self.gpu.stream();
        let (x_p, _gx) = x.device_ptr(&stream);
        let epoch = MMQ_ACT_EPOCH.load(Ordering::Relaxed);
        // quantize-once: reuse the window's scratch when the SAME activation comes back.
        let mut slot = MMQ_ACT_SLOT.lock().unwrap();
        let hit = matches!(&*slot,
            Some((e, p, mm, inf, _)) if *e == epoch && *p == x_p as u64 && *mm == m && *inf == in_f);
        if !hit {
            let act_bytes = unsafe { memra_mmq_q4_0_act_bytes(in_f as i32, m as i32) };
            let mut scratch = self.alloc_uninit::<u8>(act_bytes)?;
            {
                let (s_p, _gs) = scratch.device_ptr_mut(&stream);
                let rc = unsafe {
                    memra_mmq_q4_0_quant_act(
                        x_p as *const f32,
                        s_p as *mut core::ffi::c_void,
                        in_f as i32,
                        m as i32,
                        stream.cu_stream() as *mut core::ffi::c_void,
                    )
                };
                if rc != 0 {
                    return Err(
                        format!("memra_mmq_q4_0_quant_act(in_f={in_f}, m={m}) rc={rc}").into()
                    );
                }
            }
            *slot = Some((epoch, x_p as u64, m, in_f, scratch));
        }
        let scratch = &slot.as_ref().unwrap().4;
        {
            let (w_p, _gw) = bytes.device_ptr(&stream);
            let (y_p, _gy) = y.device_ptr_mut(&stream);
            let (s_p, _gs) = scratch.device_ptr(&stream);
            // Stream-k arm (DEFAULT since 2026-07-23; MEMRA_MMQ_SK=0 reverts to xy-tiling):
            // small-batch tail-wave fix — the sk entry itself falls back to (bit-identical)
            // tiling at >=90% wave efficiency. Band-class fold order below that. Gate: 12B
            // pp512 +3.3% (1.005x vs llama), pp1736 +1.0%; 31B +0.5%; D512 sentinel MATCH.
            //
            // SPEC-SERVING FLIP (2026-07-27, the f16pv/wkv acceptance-law pattern): with
            // MEMRA_DRAFT set the default is OFF. Two mechanisms, both measured: (1) sk's
            // fold order shifts PRIME numerics in the drafter's sub-argmax logit space
            // (26B d1736 accept 0.846 w/o sk vs 0.826 — bisect row 0.861 vs 0.405-era);
            // (2) the shape-keyed AUTOTUNE re-times both forms per process, so knife-edge
            // shapes pick DIFFERENT kernels run-to-run — the 12B depth cell was BIMODAL
            // (205 @ 0.756 / 260 @ 0.943 across identical invocations; sk-off x6 = stable
            // 263-269 @ 0.953). Explicit MEMRA_MMQ_SK always wins; plain serving keeps sk.
            // HOPPER DEFAULT OFF (2026-07-31, #23): on sm_90a the SK arm computes WRONG
            // values for the 26B a4b's non-rp Q4_0 shapes once the autotune m-bucket
            // crosses 256 (prefill argmax garbage, maxdiff ~10; MEMRA_MMQ_SK=0 -> MATCH,
            // one-variable kill x confirmed on-box). The winner is timing-picked and the
            // SK split/fixup are SM-count dependent (132 vs 170) — until the kernel is
            // fixed for that class, Hopper fails CLOSED to the bit-identical xy-tiling
            // (cost on the healthy models: g12 -1.4%, g31 -0.6% prefill, N=3 on-box).
            // sm_120a keeps SK on (rig-divergence law). MEMRA_MMQ_SK=1 still forces.
            static SK_ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let sk = match crate::MMQ_SK_FORCE.load(std::sync::atomic::Ordering::Relaxed) {
                0 => false,
                1 => true,
                _ => *SK_ON.get_or_init(|| std::env::var("MEMRA_MMQ_SK")
                        .map(|v| v != "0").unwrap_or(!cfg!(memra_hopper_mma))),
            };
            let rc = if sk {
                let mut fx = MMQ_FIXUP_SLOT.lock().unwrap();
                if fx.is_none() {
                    let nb = unsafe { memra_mmq_q4_0_fixup_bytes() };
                    *fx = Some(self.alloc_uninit::<u8>(nb)?);
                }
                let (f_p, _gf) = fx.as_mut().unwrap().device_ptr_mut(&stream);
                unsafe {
                    memra_mmq_q4_0_gemm_sk(
                        w_p as *const core::ffi::c_void,
                        s_p as *const core::ffi::c_void,
                        y_p as *mut f32,
                        f_p as *mut core::ffi::c_void,
                        in_f as i32,
                        out_f as i32,
                        m as i32,
                        stream.cu_stream() as *mut core::ffi::c_void,
                        rp as i32,
                    )
                }
            } else { unsafe {
                memra_mmq_q4_0_gemm(
                    w_p as *const core::ffi::c_void,
                    s_p as *const core::ffi::c_void,
                    y_p as *mut f32,
                    in_f as i32,
                    out_f as i32,
                    m as i32,
                    stream.cu_stream() as *mut core::ffi::c_void,
                    rp as i32,
                )
            } };
            if rc != 0 {
                return Err(format!(
                    "memra_mmq_q4_0_gemm(rp={rp}, in_f={in_f}, out_f={out_f}, m={m}, wbytes={}) rc={rc}",
                    bytes.len()
                )
                .into());
            }
        }
        Ok(y)
    }

    /// Run the vendored NVFP4 MMQ prefill GEMM from raw weight bytes + f32 activation.
    /// y[m, out_f] = x[m, in_f] @ W^T. The per-tensor NVFP4 macro-scale is FOLDED into the MMQ
    /// write-back epilogue (was a separate scale_inplace launch + full y round-trip per matmul).
    /// Same elementwise multiply -> bit-identical to the two-launch form.
    /// `x` is the RAW f32 activation (the launcher quantizes it to block_fp4_mmq internally).
    pub fn qmatvec_mmq_nvfp4(
        &self,
        bytes: &CudaSlice<u8>,
        x: &CudaSlice<f32>,
        m: usize,
        in_f: usize,
        out_f: usize,
        scale: f32,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        self.qmatvec_mmq_nvfp4_scaled(bytes, x, m, in_f, out_f, scale)
    }

    /// Bare MMQ launch (no macro-scale) — for the kernel_check accuracy gate.
    pub fn qmatvec_mmq_nvfp4_raw(
        &self,
        bytes: &CudaSlice<u8>,
        x: &CudaSlice<f32>,
        m: usize,
        in_f: usize,
        out_f: usize,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        self.qmatvec_mmq_nvfp4_scaled(bytes, x, m, in_f, out_f, 1.0)
    }

    /// Bare MMQ launch on the PRE-PORT activation quantizer (per-sub-block UE4M3 scale only, no
    /// per-token row amax). The numeric oracle for the two-level quantizer: kernel-check runs both
    /// and reports the accuracy delta, so the port's value is measured rather than asserted.
    pub fn qmatvec_mmq_nvfp4_raw_v1(
        &self,
        bytes: &CudaSlice<u8>,
        x: &CudaSlice<f32>,
        m: usize,
        in_f: usize,
        out_f: usize,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        self.qmatvec_mmq_nvfp4_inner(bytes, x, m, in_f, out_f, 1.0, false, 0)
    }

    /// Bare MMQ launch with an explicit residual-channel count — for the kernel-check k sweep.
    pub fn qmatvec_mmq_nvfp4_raw_res(
        &self,
        bytes: &CudaSlice<u8>,
        x: &CudaSlice<f32>,
        m: usize,
        in_f: usize,
        out_f: usize,
        residual_k: i32,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        self.qmatvec_mmq_nvfp4_inner(bytes, x, m, in_f, out_f, 1.0, true, residual_k)
    }

    fn qmatvec_mmq_nvfp4_scaled(
        &self,
        bytes: &CudaSlice<u8>,
        x: &CudaSlice<f32>,
        m: usize,
        in_f: usize,
        out_f: usize,
        scale: f32,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        self.qmatvec_mmq_nvfp4_inner(bytes, x, m, in_f, out_f, scale, true, mmq_residual_k())
    }

    fn qmatvec_mmq_nvfp4_inner(
        &self,
        bytes: &CudaSlice<u8>,
        x: &CudaSlice<f32>,
        m: usize,
        in_f: usize,
        out_f: usize,
        scale: f32,
        per_token_scale: bool,
        residual_k: i32,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        assert!(
            in_f % 64 == 0,
            "MMQ NVFP4 requires in_f % 64 == 0, got {in_f}"
        );
        let act_bytes = unsafe { memra_mmq_nvfp4_act_bytes(in_f as i32, m as i32) };
        let mut scratch = self.alloc_uninit::<u8>(act_bytes)?;
        let mut y = self.alloc_uninit::<f32>(m * out_f)?;
        {
            let stream = self.gpu.stream();
            let (w_p, _gw) = bytes.device_ptr(&stream);
            let (x_p, _gx) = x.device_ptr(&stream);
            let (y_p, _gy) = y.device_ptr_mut(&stream);
            let (s_p, _gs) = scratch.device_ptr_mut(&stream);
            let rc = unsafe {
                memra_mmq_nvfp4_ex2(
                    w_p as *const core::ffi::c_void,
                    x_p as *const f32,
                    y_p as *mut f32,
                    in_f as i32,
                    out_f as i32,
                    m as i32,
                    s_p as *mut core::ffi::c_void,
                    stream.cu_stream() as *mut core::ffi::c_void,
                    scale,
                    per_token_scale as i32,
                    residual_k,
                )
            };
            if rc != 0 {
                return Err(format!("memra_mmq_nvfp4_ex2 rc={rc}").into());
            }
        }
        Ok(y)
    }

    /// STAGE 2 W4A8 MMQ NVFP4: same tile as the W4A4 path, but weight FP4 is LUT-dequantized to
    /// int8 at tile-load and the activation stays q8_1 int8 — the accuracy-safe rung. Macro-scale
    /// folded into the write-back epilogue (bit-identical to a post-matmul scale_inplace).
    /// `rp` selects the weight layout (A6 split-plane vs GGUF blocks) — bit-identical output.
    pub fn qmatvec_mmq_nvfp4_w4a8(
        &self,
        bytes: &CudaSlice<u8>,
        x: &CudaSlice<f32>,
        m: usize,
        in_f: usize,
        out_f: usize,
        scale: f32,
        rp: bool,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        self.qmatvec_mmq_nvfp4_w4a8_scaled(bytes, x, m, in_f, out_f, scale, rp)
    }

    /// Bare W4A8 MMQ launch (no macro-scale, GGUF layout) — for the kernel_check accuracy gate.
    pub fn qmatvec_mmq_nvfp4_w4a8_raw(
        &self,
        bytes: &CudaSlice<u8>,
        x: &CudaSlice<f32>,
        m: usize,
        in_f: usize,
        out_f: usize,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        self.qmatvec_mmq_nvfp4_w4a8_scaled(bytes, x, m, in_f, out_f, 1.0, false)
    }

    /// Bare W4A8 MMQ launch on an A6 split-plane repacked weight — the rp-loader bit-identity gate
    /// compares this against `qmatvec_mmq_nvfp4_w4a8_raw` on the same weight.
    pub fn qmatvec_mmq_nvfp4_w4a8_raw_rp(
        &self,
        bytes: &CudaSlice<u8>,
        x: &CudaSlice<f32>,
        m: usize,
        in_f: usize,
        out_f: usize,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        self.qmatvec_mmq_nvfp4_w4a8_scaled(bytes, x, m, in_f, out_f, 1.0, true)
    }

    fn qmatvec_mmq_nvfp4_w4a8_scaled(
        &self,
        bytes: &CudaSlice<u8>,
        x: &CudaSlice<f32>,
        m: usize,
        in_f: usize,
        out_f: usize,
        scale: f32,
        rp: bool,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        assert!(
            in_f % 64 == 0,
            "MMQ NVFP4 W4A8 requires in_f % 64 == 0, got {in_f}"
        );
        let act_bytes = unsafe { memra_mmq_nvfp4_w4a8_act_bytes(in_f as i32, m as i32) };
        let mut scratch = self.alloc_uninit::<u8>(act_bytes)?;
        let mut y = self.alloc_uninit::<f32>(m * out_f)?;
        {
            let stream = self.gpu.stream();
            let (w_p, _gw) = bytes.device_ptr(&stream);
            let (x_p, _gx) = x.device_ptr(&stream);
            let (y_p, _gy) = y.device_ptr_mut(&stream);
            let (s_p, _gs) = scratch.device_ptr_mut(&stream);
            // MEMRA_MMQ_F8F4=1: the R-B W4A8-FP8 tile (own numeric config; battery-gated seam).
            // Scratch layouts are footprint-identical, so only the entry point swaps.
            static F8F4: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let f8f4 = *F8F4.get_or_init(|| std::env::var("MEMRA_MMQ_F8F4").as_deref() == Ok("1"));
            let rc = unsafe {
                if f8f4 {
                    memra_mmq_nvfp4_f8f4(
                        w_p as *const core::ffi::c_void,
                        x_p as *const f32,
                        y_p as *mut f32,
                        in_f as i32,
                        out_f as i32,
                        m as i32,
                        s_p as *mut core::ffi::c_void,
                        stream.cu_stream() as *mut core::ffi::c_void,
                        scale,
                        rp as i32,
                    )
                } else {
                    memra_mmq_nvfp4_w4a8(
                        w_p as *const core::ffi::c_void,
                        x_p as *const f32,
                        y_p as *mut f32,
                        in_f as i32,
                        out_f as i32,
                        m as i32,
                        s_p as *mut core::ffi::c_void,
                        stream.cu_stream() as *mut core::ffi::c_void,
                        scale,
                        rp as i32,
                    )
                }
            };
            if rc != 0 {
                return Err(format!("memra_mmq_nvfp4_w4a8(f8f4={f8f4}) rc={rc}").into());
            }
        }
        Ok(y)
    }

    /// Quantize token-major f32 activation [n_tokens, in_f] to the block_q8_1_mmq (D4) scratch the
    /// IQ expert-MMA kernel consumes. Returns the scratch buffer (one per proj input per layer).
    pub fn mmq_iq_quantize_act(
        &self,
        x: &CudaSlice<f32>,
        in_f: usize,
        n_tokens: usize,
    ) -> Result<CudaSlice<u8>, Box<dyn std::error::Error>> {
        let act_bytes = unsafe { memra_mmq_iq_experts_act_bytes(in_f as i32, n_tokens as i32) };
        let mut scratch = self.alloc_uninit::<u8>(act_bytes)?;
        {
            let stream = self.gpu.stream();
            let (x_p, _gx) = x.device_ptr(&stream);
            let (s_p, _gs) = scratch.device_ptr_mut(&stream);
            let rc = unsafe {
                memra_mmq_iq_quantize_act(
                    x_p as *const f32,
                    s_p as *mut core::ffi::c_void,
                    in_f as i32,
                    n_tokens as i32,
                    stream.cu_stream() as *mut core::ffi::c_void,
                )
            };
            if rc != 0 {
                return Err(format!("memra_mmq_iq_quantize_act rc={rc}").into());
            }
        }
        Ok(scratch)
    }

    /// Fused act-epilogue (research lever #3): silu/gelu(gate)*up + D4 quantize in one launch —
    /// replaces moe_pairs_{silu,gelu}_mul + mmq_iq_quantize_act without materializing the f32 act
    /// buffer (saves one full write + one full read pass over [n_pairs x n_ff]). Scratch bytes are
    /// BYTE-IDENTICAL to the two-pass path (kernel-check `iq fused act+quant` gates it).
    /// `act_kind`: 0 = silu*mul (qwen35moe), 1 = gelu_tanh*mul (gemma4).
    pub fn mmq_iq_fused_act_quant(
        &self,
        gate: &CudaSlice<f32>,
        up: &CudaSlice<f32>,
        in_f: usize,
        n_tokens: usize,
        act_kind: i32,
    ) -> Result<CudaSlice<u8>, Box<dyn std::error::Error>> {
        let act_bytes = unsafe { memra_mmq_iq_experts_act_bytes(in_f as i32, n_tokens as i32) };
        let mut scratch = self.alloc_uninit::<u8>(act_bytes)?;
        {
            let stream = self.gpu.stream();
            let (g_p, _gg) = gate.device_ptr(&stream);
            let (u_p, _gu) = up.device_ptr(&stream);
            let (s_p, _gs) = scratch.device_ptr_mut(&stream);
            let rc = unsafe {
                memra_mmq_iq_fused_act_quant(
                    g_p as *const f32,
                    u_p as *const f32,
                    s_p as *mut core::ffi::c_void,
                    in_f as i32,
                    n_tokens as i32,
                    act_kind,
                    stream.cu_stream() as *mut core::ffi::c_void,
                )
            };
            if rc != 0 {
                return Err(format!("memra_mmq_iq_fused_act_quant rc={rc}").into());
            }
        }
        Ok(scratch)
    }

    /// Expert-segmented IQ3_S/IQ4_XS int8-MMA MMQ (the m16n8k16.s8 analog of moe_pairs_matvec_q8_dec).
    /// Same CSR inputs (table/ex_ids/ex_off/ex_pairs/pair_tok) + a pre-quantized q8_1_mmq activation
    /// scratch (from `mmq_iq_quantize_act` over n_tokens). y = [n_pairs, out_f] pair-major.
    #[allow(clippy::too_many_arguments)]
    pub fn mmq_iq_experts(
        &self,
        table: &CudaSlice<u64>,
        proj: i32,
        n_expert: usize,
        ex_ids: &CudaSlice<i32>,
        ex_off: &CudaSlice<i32>,
        ex_pairs: &CudaSlice<i32>,
        pair_tok: &CudaSlice<i32>,
        act_scratch: &CudaSlice<u8>,
        in_f: usize,
        out_f: usize,
        n_active: usize,
        n_pairs: usize,
        n_tokens: usize,
        qtype: i32,
        row_bytes: usize,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let mut y = self.alloc_uninit::<f32>(n_pairs * out_f)?;
        {
            let stream = self.gpu.stream();
            let (tab_p, _g0) = table.device_ptr(&stream);
            let (ei_p, _g1) = ex_ids.device_ptr(&stream);
            let (eo_p, _g2) = ex_off.device_ptr(&stream);
            let (ep_p, _g3) = ex_pairs.device_ptr(&stream);
            let (pt_p, _g4) = pair_tok.device_ptr(&stream);
            let (as_p, _g5) = act_scratch.device_ptr(&stream);
            let (y_p, _g6) = y.device_ptr_mut(&stream);
            let rc = unsafe {
                memra_mmq_iq_experts(
                    tab_p as *const u64,
                    proj,
                    n_expert as i32,
                    ei_p as *const i32,
                    eo_p as *const i32,
                    ep_p as *const i32,
                    pt_p as *const i32,
                    as_p as *const core::ffi::c_void,
                    y_p as *mut f32,
                    in_f as i32,
                    out_f as i32,
                    n_active as i32,
                    n_tokens as i32,
                    qtype,
                    row_bytes as i64,
                    stream.cu_stream() as *mut core::ffi::c_void,
                )
            };
            if rc != 0 {
                return Err(format!("memra_mmq_iq_experts rc={rc}").into());
            }
        }
        Ok(y)
    }

    /// Gather+convert the activation to f16 pair-major [n_pairs, in_f] for the grouped
    /// GEMM, normalized per row by its amax (raw f16 overflows on gemma's activation
    /// spikes — round 46 NaN find). Returns (act_f16, row_scales) — the scales fold back
    /// into the GEMM output. `pair_tok` = None when the input is already pair-major.
    pub fn moe_f16g_act(
        &self,
        x: &CudaSlice<f32>,
        pair_tok: Option<&CudaSlice<i32>>,
        in_f: usize,
        n_pairs: usize,
    ) -> Result<(CudaSlice<u8>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let mut act = self.alloc_uninit::<u8>(n_pairs * in_f * 2)?;
        let mut scales = self.alloc_uninit::<f32>(n_pairs)?;
        {
            let stream = self.gpu.stream();
            let (x_p, _gx) = x.device_ptr(&stream);
            let pt_p = match pair_tok {
                Some(pt) => { let (p, _g) = pt.device_ptr(&stream); p as *const i32 }
                None => std::ptr::null(),
            };
            let (a_p, _ga) = act.device_ptr_mut(&stream);
            let (s_p, _gs) = scales.device_ptr_mut(&stream);
            let rc = unsafe {
                memra_moe_f16g_gather_act(x_p as *const f32, pt_p,
                    a_p as *mut core::ffi::c_void, s_p as *mut f32,
                    in_f as i32, n_pairs as i32,
                    stream.cu_stream() as *mut core::ffi::c_void)
            };
            if rc != 0 { return Err(format!("memra_moe_f16g_gather_act rc={rc}").into()); }
        }
        Ok((act, scales))
    }

    /// One projection through the grouped f16 lane: dequant the active experts' rows to an
    /// f16 workspace, then ONE grouped GEMM over the CSR groups (variable m per expert).
    /// y = f32 [n_pairs, out_f] pair-major — same layout as mmq_iq_experts.
    /// MEMRA_MOE_F16G=1: cublasGemmGroupedBatchedEx (+ h2f pass + per-projection sync — the
    /// grouped API runs on internal streams unordered with ours, round-47 ledger).
    /// MEMRA_MOE_F16G=2: single-kernel grouped GEMM on the engine stream (round 49) — the
    /// row scale folds into the kernel epilogue; no f16 C, no h2f, NO sync (ordered by
    /// construction). f16-MIRROR numeric class either way (argmax/spec gated, not
    /// byte-identity). Errors on unsupported qtype (caller keeps the MMQ arm as fallback).
    #[allow(clippy::too_many_arguments)]
    pub fn moe_f16_grouped(
        &self,
        table: &CudaSlice<u64>,
        proj: i32,
        n_expert: usize,
        ex_ids: &CudaSlice<i32>,
        ex_off_host: &[i32],
        ex_off_dev: &CudaSlice<i32>,
        act_f16: &CudaSlice<u8>,
        act_scale: &CudaSlice<f32>,
        in_f: usize,
        out_f: usize,
        n_active: usize,
        n_pairs: usize,
        qtype: i32,
        row_bytes: usize,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let sk = crate::moe_f16g_mode() >= 2 && in_f % 32 == 0;
        // DIRECT-FROM-QUANT lane (lane/kquant-tile-loaders + lane/iq-direct-loaders, default
        // ON — MEMRA_F16G_DIRECT=0 is the rollback seam): Q4_K/Q6_K/IQ4_XS/IQ3_S expert
        // projections skip the dequant-workspace pass entirely; the sk visitor forms dequant
        // B tiles in-register from the superblocks. Bit-identical to the workspace path by
        // construction (kernel-check "f16g-kq-direct") — this is a pure data-movement change,
        // not a numeric-class change. Admission mirrors the C-side guards; the grid-scan
        // rollback arm (MEMRA_F16G_SK=0) keeps the workspace.
        let (shape_sel, cross) = crate::moe_f16g_sk_params();
        if sk && shape_sel >= 0 && crate::moe_f16g_direct_on(qtype)
            && (qtype == crate::QT_Q4_K || qtype == crate::QT_Q6_K
                || qtype == crate::QT_IQ4_XS || qtype == crate::QT_IQ3_S)
            && in_f % 256 == 0 && n_active <= 512 && n_active > 0
        {
            let max_m = ex_off_host.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(0);
            let mut y = self.alloc_uninit::<f32>(n_pairs * out_f)?;
            {
                let stream = self.gpu.stream();
                let (tab_p, _g0) = table.device_ptr(&stream);
                let (ei_p, _g1) = ex_ids.device_ptr(&stream);
                let (a_p, _g2) = act_f16.device_ptr(&stream);
                let (s_p, _g3) = act_scale.device_ptr(&stream);
                let (off_p, _g4) = ex_off_dev.device_ptr(&stream);
                let (y_p, _g5) = y.device_ptr_mut(&stream);
                let rc = unsafe {
                    memra_moe_kq_gemm_sk(tab_p as *const u64, proj, n_expert as i32,
                        ei_p as *const i32, a_p as *const core::ffi::c_void, y_p as *mut f32,
                        s_p as *const f32, off_p as *const i32, ex_off_host.as_ptr(),
                        n_active as i32, max_m, in_f as i32, out_f as i32, qtype, cross,
                        crate::moe_f16g_tail_on() as i32, row_bytes as i64,
                        stream.cu_stream() as *mut core::ffi::c_void)
                };
                if rc != 0 { return Err(format!("memra_moe_kq_gemm_sk rc={rc}").into()); }
            }
            return Ok(y);
        }
        // one-time cublas grouped init (algo heuristics + module load cost ~10% of a cold
        // g26 prime when paid inside the first projection): a tiny dummy grouped GEMM at
        // first use, synced, so the real prime runs warm. The =2 path never touches cublas.
        if !sk {
            static WARM: std::sync::Once = std::sync::Once::new();
            let mut warm_err = None;
            WARM.call_once(|| {
                let r = (|| -> Result<(), Box<dyn std::error::Error>> {
                    let w = self.alloc_uninit::<u8>(2 * 32 * 64 * 2)?;
                    let a = self.alloc_uninit::<u8>(4 * 64 * 2)?;
                    let mut yw = self.alloc_uninit::<u8>(4 * 32 * 2)?;
                    let off = [0i32, 2, 4];
                    let stream = self.gpu.stream();
                    let (w_p, _a1) = w.device_ptr(&stream);
                    let (a_p, _a2) = a.device_ptr(&stream);
                    let (y_p, _a3) = yw.device_ptr_mut(&stream);
                    let rc = unsafe {
                        memra_moe_f16g_gemm(w_p as *const core::ffi::c_void,
                            a_p as *const core::ffi::c_void, y_p as *mut core::ffi::c_void,
                            off.as_ptr(), 2, 64, 32,
                            stream.cu_stream() as *mut core::ffi::c_void)
                    };
                    if rc != 0 { return Err(format!("f16g warmup rc={rc}").into()); }
                    self.gpu.stream().synchronize()?;
                    Ok(())
                })();
                if let Err(e) = r { warm_err = Some(e.to_string()); }
            });
            if let Some(we) = warm_err { return Err(we.into()); }
        }
        let w_bytes = n_active * out_f * in_f * 2;
        let mut w_f16 = self.alloc_uninit::<u8>(w_bytes)?;
        let mut y = self.alloc_uninit::<f32>(n_pairs * out_f)?;
        {
            let stream = self.gpu.stream();
            let (tab_p, _g0) = table.device_ptr(&stream);
            let (ei_p, _g1) = ex_ids.device_ptr(&stream);
            let (w_p, _g2) = w_f16.device_ptr_mut(&stream);
            let rc = unsafe {
                memra_moe_f16g_dequant(tab_p as *const u64, proj, n_expert as i32,
                    ei_p as *const i32, w_p as *mut core::ffi::c_void,
                    in_f as i32, out_f as i32, n_active as i32, qtype, row_bytes as i64,
                    stream.cu_stream() as *mut core::ffi::c_void)
            };
            if rc != 0 { return Err(format!("memra_moe_f16g_dequant rc={rc}").into()); }
            let (a_p, _g3) = act_f16.device_ptr(&stream);
            let (s_p, _g6) = act_scale.device_ptr(&stream);
            let (y_p, _g5) = y.device_ptr_mut(&stream);
            if sk {
                let max_m = ex_off_host.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(0);
                let (off_p, _g7) = ex_off_dev.device_ptr(&stream);
                let (shape_sel, cross) = crate::moe_f16g_sk_params();
                let rc = unsafe {
                    memra_moe_f16g_gemm_sk(w_p as *const core::ffi::c_void,
                        a_p as *const core::ffi::c_void, y_p as *mut f32,
                        s_p as *const f32, off_p as *const i32, ex_off_host.as_ptr(),
                        n_active as i32, max_m, in_f as i32, out_f as i32, shape_sel, cross,
                        crate::moe_f16g_tail_on() as i32,
                        stream.cu_stream() as *mut core::ffi::c_void)
                };
                if rc != 0 { return Err(format!("memra_moe_f16g_gemm_sk rc={rc}").into()); }
            } else {
                let mut y16 = self.alloc_uninit::<u8>(n_pairs * out_f * 2)?;
                let (y16_p, _g4) = y16.device_ptr_mut(&stream);
                let rc = unsafe {
                    memra_moe_f16g_gemm(w_p as *const core::ffi::c_void,
                        a_p as *const core::ffi::c_void, y16_p as *mut core::ffi::c_void,
                        ex_off_host.as_ptr(), n_active as i32, in_f as i32, out_f as i32,
                        stream.cu_stream() as *mut core::ffi::c_void)
                };
                if rc != 0 { return Err(format!("memra_moe_f16g_gemm rc={rc}").into()); }
                let rc = unsafe {
                    memra_moe_f16g_h2f_scaled(y16_p as *const core::ffi::c_void, y_p as *mut f32,
                        s_p as *const f32, out_f as i32, n_pairs as i32,
                        stream.cu_stream() as *mut core::ffi::c_void)
                };
                if rc != 0 { return Err(format!("memra_moe_f16g_h2f_scaled rc={rc}").into()); }
            }
        }
        // MODE 1 ONLY: cublasGemmGroupedBatchedEx issues through internal streams NOT ordered
        // with ours (round 46: NaN race, clean under sync — 205=205 MATCH). Full sync per
        // projection. Mode 2 (single kernel, our stream) is ordered by construction — no sync,
        // that is the point of this arc.
        if !sk {
            self.gpu.stream().synchronize()?;
        }
        if std::env::var("MEMRA_F16G_DEBUG").is_ok() {
            // FULL NaN/Inf scan of w, act (through h2f) and y — localizes the corrupt stage.
            let wn = n_active * out_f * in_f;
            let an = n_pairs * in_f;
            let mut wf = self.alloc_uninit::<f32>(wn)?;
            let mut af = self.alloc_uninit::<f32>(an)?;
            {
                let stream = self.gpu.stream();
                let (w_p, _a) = w_f16.device_ptr(&stream);
                let (a_p, _b) = act_f16.device_ptr(&stream);
                let (wf_p, _c) = wf.device_ptr_mut(&stream);
                let (af_p, _d) = af.device_ptr_mut(&stream);
                unsafe {
                    memra_moe_f16g_h2f(w_p as *const core::ffi::c_void, wf_p as *mut f32, wn,
                        stream.cu_stream() as *mut core::ffi::c_void);
                    memra_moe_f16g_h2f(a_p as *const core::ffi::c_void, af_p as *mut f32, an,
                        stream.cu_stream() as *mut core::ffi::c_void);
                }
            }
            let (wh, ah, yh) = (self.dtoh(&wf)?, self.dtoh(&af)?, self.dtoh(&y)?);
            let scan = |v: &[f32]| -> (usize, f32) {
                let bad = v.iter().filter(|x| !x.is_finite()).count();
                let mx = v.iter().filter(|x| x.is_finite()).fold(0.0f32, |m, x| m.max(x.abs()));
                (bad, mx)
            };
            let (wb, wm) = scan(&wh); let (ab, am) = scan(&ah); let (yb, ym) = scan(&yh);
            eprintln!("[f16g-debug] proj={proj} w: bad={wb} max={wm:.3e} | act: bad={ab} \
                       max={am:.3e} | y: bad={yb} max={ym:.3e} (na={n_active} np={n_pairs} \
                       in={in_f} out={out_f})");
        }
        Ok(y)
    }

    /// Raw sk grouped-GEMM entry for kernel-check ("f16g-sk" section): explicit shape/cross
    /// instead of the env policy. shape_sel < 0 = the round-49 grid-scan rollback arm; else
    /// the round-51 problem-visitor split at `cross` (1 forces all-128, i32::MAX all-32).
    /// tail: 1 = the deep tail (32x64x64 3-stage, lane/sk-tail-form) on sub-cross groups,
    /// 0 = the round-51 2-stage 32x64x32 tail.
    /// w_f16 = [n_active][out_f][in_f] f16 bytes, act_f16 = [n_pairs][in_f] f16 bytes.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_f16g_gemm_sk_raw(&self, w_f16: &CudaSlice<u8>, act_f16: &CudaSlice<u8>,
        row_scale: &CudaSlice<f32>, ex_off_host: &[i32], ex_off_dev: &CudaSlice<i32>,
        in_f: usize, out_f: usize, n_pairs: usize, shape_sel: i32, cross: i32, tail: i32)
        -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let n_active = ex_off_host.len() - 1;
        let max_m = ex_off_host.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(0);
        let mut y = self.alloc_uninit::<f32>(n_pairs * out_f)?;
        {
            let stream = self.gpu.stream();
            let (w_p, _g0) = w_f16.device_ptr(&stream);
            let (a_p, _g1) = act_f16.device_ptr(&stream);
            let (s_p, _g2) = row_scale.device_ptr(&stream);
            let (off_p, _g3) = ex_off_dev.device_ptr(&stream);
            let (y_p, _g4) = y.device_ptr_mut(&stream);
            let rc = unsafe {
                memra_moe_f16g_gemm_sk(w_p as *const core::ffi::c_void,
                    a_p as *const core::ffi::c_void, y_p as *mut f32,
                    s_p as *const f32, off_p as *const i32, ex_off_host.as_ptr(),
                    n_active as i32, max_m, in_f as i32, out_f as i32, shape_sel, cross,
                    tail, stream.cu_stream() as *mut core::ffi::c_void)
            };
            if rc != 0 { return Err(format!("memra_moe_f16g_gemm_sk rc={rc}").into()); }
        }
        Ok(y)
    }

    /// Raw direct-from-quant sk grouped-GEMM entry for kernel-check ("f16g-kq-direct"):
    /// explicit cross/tail instead of the env policy. `table` = device u64 pointer table
    /// (proj-major, [n_proj][n_expert] — same contract as moe_f16_grouped), `ex_ids` =
    /// active-expert ids (device). Visitor forms only (the C side rejects anything else).
    #[allow(clippy::too_many_arguments)]
    pub fn moe_kq_gemm_sk_raw(&self, table: &CudaSlice<u64>, proj: i32, n_expert: usize,
        ex_ids: &CudaSlice<i32>, act_f16: &CudaSlice<u8>, row_scale: &CudaSlice<f32>,
        ex_off_host: &[i32], ex_off_dev: &CudaSlice<i32>,
        in_f: usize, out_f: usize, n_pairs: usize, qtype: i32, row_bytes: usize, cross: i32,
        tail: i32)
        -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let n_active = ex_off_host.len() - 1;
        let max_m = ex_off_host.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(0);
        let mut y = self.alloc_uninit::<f32>(n_pairs * out_f)?;
        {
            let stream = self.gpu.stream();
            let (tab_p, _g0) = table.device_ptr(&stream);
            let (ei_p, _g1) = ex_ids.device_ptr(&stream);
            let (a_p, _g2) = act_f16.device_ptr(&stream);
            let (s_p, _g3) = row_scale.device_ptr(&stream);
            let (off_p, _g4) = ex_off_dev.device_ptr(&stream);
            let (y_p, _g5) = y.device_ptr_mut(&stream);
            let rc = unsafe {
                memra_moe_kq_gemm_sk(tab_p as *const u64, proj, n_expert as i32,
                    ei_p as *const i32, a_p as *const core::ffi::c_void, y_p as *mut f32,
                    s_p as *const f32, off_p as *const i32, ex_off_host.as_ptr(),
                    n_active as i32, max_m, in_f as i32, out_f as i32, qtype, cross,
                    tail, row_bytes as i64,
                    stream.cu_stream() as *mut core::ffi::c_void)
            };
            if rc != 0 { return Err(format!("memra_moe_kq_gemm_sk rc={rc}").into()); }
        }
        Ok(y)
    }

    /// Raw dequant-workspace entry for kernel-check: dequant the active experts' rows to a
    /// fresh f16 workspace via the same kernel `moe_f16_grouped` uses (the direct loaders'
    /// bitwise reference).
    pub fn moe_f16g_dequant_raw(&self, table: &CudaSlice<u64>, proj: i32, n_expert: usize,
        ex_ids: &CudaSlice<i32>, in_f: usize, out_f: usize, n_active: usize, qtype: i32,
        row_bytes: usize)
        -> Result<CudaSlice<u8>, Box<dyn std::error::Error>> {
        let mut w_f16 = self.alloc_uninit::<u8>(n_active * out_f * in_f * 2)?;
        {
            let stream = self.gpu.stream();
            let (tab_p, _g0) = table.device_ptr(&stream);
            let (ei_p, _g1) = ex_ids.device_ptr(&stream);
            let (w_p, _g2) = w_f16.device_ptr_mut(&stream);
            let rc = unsafe {
                memra_moe_f16g_dequant(tab_p as *const u64, proj, n_expert as i32,
                    ei_p as *const i32, w_p as *mut core::ffi::c_void,
                    in_f as i32, out_f as i32, n_active as i32, qtype, row_bytes as i64,
                    stream.cu_stream() as *mut core::ffi::c_void)
            };
            if rc != 0 { return Err(format!("memra_moe_f16g_dequant rc={rc}").into()); }
        }
        Ok(w_f16)
    }
}
