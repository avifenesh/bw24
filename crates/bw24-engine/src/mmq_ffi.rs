//! FFI to the MMQ prefill GEMMs (cu/mmq_fp4.cu + cu/mmq_q45k.cu) — vendored floor kernels.
//!
//! NVFP4: the 5150-pp512 kernel from llama.cpp, ggml-decoupled into a static lib with a C-ABI host
//! launcher. The launcher quantizes the f32 activation to block_fp4_mmq internally (llama's 2-level
//! FP8-e8m0/UE4M3 scale = the accurate W4A8-via-FP8 path that fixes bw24's W4A4 maxdiff 1.46), then
//! launches the native mxf4nvf4 block-scale tensor-core mma.
//!
//! Q4_K/Q5_K: llama's k-quant int8-MMA MMQ (dequant to int8 at tile-load, q8_1 DS4 activation with
//! the (d, sum) pair that feeds the k-quant min-offset term, shared m16n8k32 s8 mma inner loop).
//! Replaces the hand-rolled qmatvec_gemm k-quant GEMMs that dominate prefill (32% + 28% busy).
//!
//! All dispatched behind BW24_MMQ=1. Always built (no external deps) — unlike cutlass_ffi which is
//! BW24_CUTLASS-gated.

use crate::Engine;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

unsafe extern "C" {
    /// Bytes needed for the block_fp4_mmq activation scratch for (in_f, n_tokens).
    pub fn bw24_mmq_nvfp4_act_bytes(in_f: i32, n_tokens: i32) -> usize;
    /// Run the NVFP4 W4A4 MMQ prefill GEMM. y[n_tokens, out_f] = act[n_tokens, in_f] @ W[out_f, in_f]^T.
    ///   W_nvfp4_blocks : raw bw24 NVFP4 weight rows (block_nvfp4 36B blocks, in_f/64 per row).
    ///   act_f32        : f32 activation [n_tokens, in_f] (contiguous).
    ///   y              : f32 output [n_tokens, out_f].
    ///   act_scratch    : pre-alloc'd quant buffer >= bw24_mmq_nvfp4_act_bytes(in_f, n_tokens).
    /// Returns 0 on success, else (1000 + cudaError).
    pub fn bw24_mmq_nvfp4(
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
    /// Bytes needed for the block_q8_1_mmq activation scratch for the NVFP4 W4A8 path.
    pub fn bw24_mmq_nvfp4_w4a8_act_bytes(in_f: i32, n_tokens: i32) -> usize;
    /// Run the NVFP4 W4A8 MMQ prefill GEMM (STAGE 2 accuracy-safe rung). Same fast MMQ tile as
    /// bw24_mmq_nvfp4 (W4A4) but the non-Blackwell int8 pair: weight FP4 LUT-dequantized to int8 at
    /// tile-load, activation stays q8_1 int8 (D4, the same quant class as the default int8 GEMM).
    /// `rp`: 0 = GGUF 36B-block weight layout, 1 = A6 split-plane repack (the resident decode
    /// layout). The rp tile loader is a pure address remap of the GGUF loader (same dequant math,
    /// same FP op order) — output is bit-identical either way.
    /// Same contract as bw24_mmq_nvfp4 otherwise. Returns 0 or (1000 + cudaError).
    pub fn bw24_mmq_nvfp4_w4a8(
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
    pub fn bw24_mmq_nvfp4_f8f4_act_bytes(in_f: i32, n_tokens: i32) -> usize;
    /// R-B W4A8-FP8 MMQ prefill GEMM (research/prefill-mxf8f6f4-design.md): NVFP4 per-16 scales
    /// fold into e4m3 weight VALUES at tile load; e4m3 activations; ONE kind::f8f6f4 m16n8k32
    /// MMA (381-TF class) where the int8 path issues two imma k16. NEW NUMERIC CONFIG — own
    /// battery. Same contract/rp semantics as bw24_mmq_nvfp4_w4a8. Returns 0 / 1000+cudaError /
    /// 2000+cudaError.
    pub fn bw24_mmq_nvfp4_f8f4(
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
    pub fn bw24_mmq_q45k_act_bytes(in_f: i32, n_tokens: i32) -> usize;
    /// Run the Q4_K W4A8 MMQ prefill GEMM. Same contract as bw24_mmq_nvfp4 (raw ggml block_q4_K
    /// weight rows, in_f/256 144B superblocks per row). Returns 0 or (1000 + cudaError).
    pub fn bw24_mmq_q4_K(
        w_q4k_blocks: *const core::ffi::c_void,
        act_f32: *const f32,
        y: *mut f32,
        in_f: i32,
        out_f: i32,
        n_tokens: i32,
        act_scratch: *mut core::ffi::c_void,
        stream: *mut core::ffi::c_void,
    ) -> i32;
    /// Run the Q5_K W4A8 MMQ prefill GEMM (176B superblocks). Same contract as bw24_mmq_q4_K.
    pub fn bw24_mmq_q5_K(
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
    pub fn bw24_mmq_q8_0_act_bytes(in_f: i32, n_tokens: i32) -> usize;
    /// Run the Q8_0 int8-MMA MMQ prefill GEMM (BW24_PP_Q8MMQ). Conventional xy-tiling only (no fixup
    /// scratch). Weight = raw ggml block_q8_0 rows (34B blocks, in_f/32 per row); activation is
    /// quantized internally to q8_1 D4. Requires in_f % 32 == 0. Returns 0 or (1000 + cudaError).
    pub fn bw24_mmq_q8_0(
        w_q8_0_blocks: *const core::ffi::c_void,
        act_f32: *const f32,
        y: *mut f32,
        in_f: i32,
        out_f: i32,
        n_tokens: i32,
        act_scratch: *mut core::ffi::c_void,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    // ---- IQ3_S / IQ4_XS expert-segmented int8-MMA MMQ (cu/mmq_iq_experts.cu, BW24_MOE_MMA) ----
    /// Bytes for the token-major block_q8_1_mmq activation scratch (in_f, n_tokens).
    pub fn bw24_mmq_iq_experts_act_bytes(in_f: i32, n_tokens: i32) -> usize;
    /// Quantize token-major f32 activation [n_tokens, in_f] -> block_q8_1_mmq (D4). Returns 0 or 1000+err.
    pub fn bw24_mmq_iq_quantize_act(
        act_f32: *const f32,
        act_scratch: *mut core::ffi::c_void,
        in_f: i32,
        n_tokens: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
    /// Expert-segmented IQ MMA MMQ. Same CSR shape as moe_pairs_matvec_q8_dec: `table` = [3,n_expert]
    /// device slab ptrs, CSR ex_ids/ex_off/ex_pairs group pairs by expert, pair_tok gathers the
    /// activation row. y = [n_pairs, out_f] pair-major. `act_scratch` pre-quantized over n_tokens.
    /// qtype: 5=IQ4_XS, 6=IQ3_S. Returns 0 or 1000+cudaError.
    pub fn bw24_mmq_iq_experts(
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
}

/// W4A8-MMQ DEFAULT-FLIP seam (2026-07-05): the vendored MMQ prefill suite is DEFAULT-ON — NVFP4
/// takes the W4A8 MMQ tile (same int8 accuracy class as the int8 GEMM it replaces, all exactness
/// gates hold, ~1.9x pp512; the rp tile-loader arm coexists with the A6 split-plane repack) and
/// Q4_K/Q5_K take the vendored k-quant int8-MMA MMQ (also int8-class; gated with W4A8 in the same
/// battery — the predecessor's `BW24_MMQ_W4A8=1` arm engaged BOTH, this flip preserves exactly
/// that measured config). `BW24_MMQ_W4A8=0` = escape hatch back to the int8 GEMM prefill
/// everywhere. `BW24_MMQ=1` additionally switches GGUF-layout NVFP4 to the W4A4 mxf4nvf4 tile
/// (speed/accuracy tradeoff opt-in, unchanged).
pub fn mmq_w4a8_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("BW24_MMQ_W4A8")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

/// Q8_0 MMQ prefill seam (lane/ppmmq lever 2, DEFAULT ON since 2026-07-09 — `BW24_PP_Q8MMQ=0`
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
        std::env::var("BW24_PP_Q8MMQ")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

impl Engine {
    /// True if `w` should take a vendored MMQ GEMM under the current env policy (see
    /// `mmq_w4a8_enabled`): NVFP4 needs in_f % 64 == 0, Q4_K/Q5_K need in_f % 256 == 0.
    pub fn mmq_supports(&self, w: &crate::model::GpuTensor) -> bool {
        use crate::model::GpuTensor;
        if cfg!(bw24_portable_cuda) {
            return false;
        }
        let mmq_opt_in = std::env::var("BW24_MMQ").is_ok();
        match w {
            // A6 split-plane repacked NVFP4: ONLY the W4A8 loader has an rp arm (pure address
            // remap, bit-identical output — mmq_nvfp4_w4a8.cu load_tiles_nvfp4_w4a8<is_rp>).
            // The W4A4 loader (mmq_fp4.cu load_tiles_nvfp4_nvfp4) reads 36B GGUF blocks only,
            // so an rp weight with W4A8 disabled falls through to the rp-ported int8 GEMM.
            GpuTensor::Quant { qtype, rp, .. } if *qtype == crate::QT_NVFP4 && *rp => {
                mmq_w4a8_enabled() && w.in_features() % 64 == 0
            }
            // GGUF-layout NVFP4 (BW24_RP=0): W4A8 (default-on) or the explicit W4A4 opt-in.
            GpuTensor::Quant { qtype, .. } if *qtype == crate::QT_NVFP4 => {
                (mmq_w4a8_enabled() || mmq_opt_in) && w.in_features() % 64 == 0
            }
            GpuTensor::Quant { qtype, .. }
                if *qtype == crate::QT_Q4_K || *qtype == crate::QT_Q5_K =>
            {
                (mmq_w4a8_enabled() || mmq_opt_in) && w.in_features() % 256 == 0
            }
            // Q8_0 dense projections (35B attn/ssm/shexp): opt-in only (BW24_PP_Q8MMQ=1), its own
            // numeric config vs qmatvec_gemm_q8_0. in_f % 32 == 0 (integral q8_0 blocks per row).
            GpuTensor::Quant { qtype, .. } if *qtype == crate::QT_Q8_0 => {
                mmq_q8_enabled() && w.in_features() % 32 == 0
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
        // (mxf4nvf4 mma, explicit BW24_MMQ=1 speed/accuracy tradeoff). An rp weight ALWAYS takes
        // W4A8 — only its loader has the split-plane arm (pure address remap, bit-identical).
        // Explicit BW24_MMQ_W4A8=1 still overrides a simultaneous BW24_MMQ=1 (predecessor rule).
        let w4a8_explicit = std::env::var("BW24_MMQ_W4A8")
            .map(|v| v != "0")
            .unwrap_or(false);
        let use_w4a8 =
            *rp || w4a8_explicit || (mmq_w4a8_enabled() && std::env::var("BW24_MMQ").is_err());
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
                let mut y = self.qmatvec_mmq_q8_0_raw(bytes, x, m, in_f, out_f)?;
                if *scale != 1.0 {
                    self.scale_inplace(&mut y, *scale, m * out_f)?;
                }
                Ok(y)
            }
            q => Err(format!("qmatvec_mmq: unsupported qtype {q}").into()),
        }
    }

    /// Bare Q4_K/Q5_K MMQ launch (no macro-scale) — also the kernel_check accuracy-gate entry.
    /// Conventional xy-tiling only (the vendored stream-K arm — BW24_MMQ_STREAMK — was removed
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
        let act_bytes = unsafe { bw24_mmq_q45k_act_bytes(in_f as i32, m as i32) };
        let mut scratch = self.alloc_uninit::<u8>(act_bytes)?;
        let mut y = self.alloc_uninit::<f32>(m * out_f)?;
        {
            let stream = &self.gpu.stream;
            let (w_p, _gw) = bytes.device_ptr(stream);
            let (x_p, _gx) = x.device_ptr(stream);
            let (y_p, _gy) = y.device_ptr_mut(stream);
            let (s_p, _gs) = scratch.device_ptr_mut(stream);
            let launcher = if qtype == crate::QT_Q4_K {
                bw24_mmq_q4_K
            } else {
                bw24_mmq_q5_K
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
                return Err(format!("bw24_mmq_q45k(qtype={qtype}) rc={rc}").into());
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
        let act_bytes = unsafe { bw24_mmq_q8_0_act_bytes(in_f as i32, m as i32) };
        let mut scratch = self.alloc_uninit::<u8>(act_bytes)?;
        let mut y = self.alloc_uninit::<f32>(m * out_f)?;
        {
            let stream = &self.gpu.stream;
            let (w_p, _gw) = bytes.device_ptr(stream);
            let (x_p, _gx) = x.device_ptr(stream);
            let (y_p, _gy) = y.device_ptr_mut(stream);
            let (s_p, _gs) = scratch.device_ptr_mut(stream);
            let rc = unsafe {
                bw24_mmq_q8_0(
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
                return Err(format!("bw24_mmq_q8_0 rc={rc}").into());
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

    fn qmatvec_mmq_nvfp4_scaled(
        &self,
        bytes: &CudaSlice<u8>,
        x: &CudaSlice<f32>,
        m: usize,
        in_f: usize,
        out_f: usize,
        scale: f32,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        assert!(
            in_f % 64 == 0,
            "MMQ NVFP4 requires in_f % 64 == 0, got {in_f}"
        );
        let act_bytes = unsafe { bw24_mmq_nvfp4_act_bytes(in_f as i32, m as i32) };
        let mut scratch = self.alloc_uninit::<u8>(act_bytes)?;
        let mut y = self.alloc_uninit::<f32>(m * out_f)?;
        {
            let stream = &self.gpu.stream;
            let (w_p, _gw) = bytes.device_ptr(stream);
            let (x_p, _gx) = x.device_ptr(stream);
            let (y_p, _gy) = y.device_ptr_mut(stream);
            let (s_p, _gs) = scratch.device_ptr_mut(stream);
            let rc = unsafe {
                bw24_mmq_nvfp4(
                    w_p as *const core::ffi::c_void,
                    x_p as *const f32,
                    y_p as *mut f32,
                    in_f as i32,
                    out_f as i32,
                    m as i32,
                    s_p as *mut core::ffi::c_void,
                    stream.cu_stream() as *mut core::ffi::c_void,
                    scale,
                )
            };
            if rc != 0 {
                return Err(format!("bw24_mmq_nvfp4 rc={rc}").into());
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
        let act_bytes = unsafe { bw24_mmq_nvfp4_w4a8_act_bytes(in_f as i32, m as i32) };
        let mut scratch = self.alloc_uninit::<u8>(act_bytes)?;
        let mut y = self.alloc_uninit::<f32>(m * out_f)?;
        {
            let stream = &self.gpu.stream;
            let (w_p, _gw) = bytes.device_ptr(stream);
            let (x_p, _gx) = x.device_ptr(stream);
            let (y_p, _gy) = y.device_ptr_mut(stream);
            let (s_p, _gs) = scratch.device_ptr_mut(stream);
            // BW24_MMQ_F8F4=1: the R-B W4A8-FP8 tile (own numeric config; battery-gated seam).
            // Scratch layouts are footprint-identical, so only the entry point swaps.
            static F8F4: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let f8f4 = *F8F4.get_or_init(|| std::env::var("BW24_MMQ_F8F4").as_deref() == Ok("1"));
            let rc = unsafe {
                if f8f4 {
                    bw24_mmq_nvfp4_f8f4(
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
                    bw24_mmq_nvfp4_w4a8(
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
                return Err(format!("bw24_mmq_nvfp4_w4a8(f8f4={f8f4}) rc={rc}").into());
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
        let act_bytes = unsafe { bw24_mmq_iq_experts_act_bytes(in_f as i32, n_tokens as i32) };
        let mut scratch = self.alloc_uninit::<u8>(act_bytes)?;
        {
            let stream = &self.gpu.stream;
            let (x_p, _gx) = x.device_ptr(stream);
            let (s_p, _gs) = scratch.device_ptr_mut(stream);
            let rc = unsafe {
                bw24_mmq_iq_quantize_act(
                    x_p as *const f32,
                    s_p as *mut core::ffi::c_void,
                    in_f as i32,
                    n_tokens as i32,
                    stream.cu_stream() as *mut core::ffi::c_void,
                )
            };
            if rc != 0 {
                return Err(format!("bw24_mmq_iq_quantize_act rc={rc}").into());
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
            let stream = &self.gpu.stream;
            let (tab_p, _g0) = table.device_ptr(stream);
            let (ei_p, _g1) = ex_ids.device_ptr(stream);
            let (eo_p, _g2) = ex_off.device_ptr(stream);
            let (ep_p, _g3) = ex_pairs.device_ptr(stream);
            let (pt_p, _g4) = pair_tok.device_ptr(stream);
            let (as_p, _g5) = act_scratch.device_ptr(stream);
            let (y_p, _g6) = y.device_ptr_mut(stream);
            let rc = unsafe {
                bw24_mmq_iq_experts(
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
                return Err(format!("bw24_mmq_iq_experts rc={rc}").into());
            }
        }
        Ok(y)
    }
}
