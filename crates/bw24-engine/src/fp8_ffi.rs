//! FP8-ACT PREFILL (BW24_PP_FP8=1): cuBLASLt FP8-E4M3 TN GEMM for the F8-E4M3-origin projections.
//!
//! Probe verdict 2026-07-08 (probe/fp8_lt_prefill.cu, JSONL row in research/tune-data): cuBLASLt
//! FP8 GEMM runs 620-795 TF at the 27B prefill shapes vs 47-72 TF for the qmatvec_gemm_q8_0 class
//! those weights ride today (46.5% of pp GPU time) — projected ~1.85x pp from the F8-native
//! layers alone. The weight side is EXACT: the checkpoint's raw e4m3 bytes + per-tensor f32
//! weight_scale are stashed at load next to the Q8_0 re-encode (`GpuTensor::Quant { fp8 }`,
//! following the `cutlass` optional-operand precedent). The only new rounding vs today is the
//! ACTIVATION: f32 -> e4m3 with ONE per-batch scalar scale (amax/448) instead of q8_1's per-32
//! int8 — finer mantissa lost, coarser scale granularity; the run-gen argmax gate arbitrates.
//!
//! Dispatch: `matmul`/`matmul_pre` m>=16 arms ONLY (prefill). Decode (m<16) keeps the Q8_0
//! dp4a/MMVQ chain bit-for-bit — the spec-exactness law is untouched, and the m=K+1 verify tier
//! (m<=9) never reaches this path.
//!
//! All device work (amax reduce, scale finalize, e4m3 quantize, cublasLtMatmul) runs on the one
//! `gpu.stream` inside a single C-ABI call (cu/fp8_prefill.cu) — no host sync anywhere: the act
//! scale is folded with weight_scale into a device scalar fed to the GEMM's B_SCALE_POINTER
//! (per-token OUTER_VEC B-scales are NOT supported on sm_120 — probed; scalar scales verified
//! exact there).

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

unsafe extern "C" {
    /// One FP8 prefill GEMM: quantize act f32->e4m3 (per-batch scalar) + cublasLtMatmul TN.
    /// Returns 0 on success (see cu/fp8_prefill.cu for the error-code bands).
    fn bw24_fp8_pp_gemm(
        w_e4m3: *const core::ffi::c_void,
        x_f32: *const f32,
        xq_e4m3: *mut core::ffi::c_void,
        scales: *mut f32,
        y_f32: *mut f32,
        m: i32,
        n: i32,
        k: i32,
        w_scale: f32,
        ws: *mut core::ffi::c_void,
        ws_bytes: usize,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

/// `BW24_PP_FP8=1` gate (default OFF), read once. Gates BOTH the loader stash (model.rs) and the
/// prefill dispatch — unset means zero VRAM / zero dispatch change.
pub fn pp_fp8_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("BW24_PP_FP8")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// `BW24_ST_E4M3=1` gate (default OFF; lane e4m3dec 2026-07-08): F8-E4M3-origin safetensors
/// projections load as RAW e4m3 (QT_F8_E4M3) instead of the Q8_0 re-encode. NEW NUMERIC CONFIG:
/// decode reads the checkpoint's own e4m3 precision (the Q8_0 re-encode was a lossy extra hop) via
/// qmatvec_e4m3_mmvq; prefill (m>=16) rides the cuBLASLt FP8 GEMM on the SAME resident bytes —
/// one weight copy total (frees the ~GBs the BW24_PP_FP8 stash duplicated, no budget cap needed).
/// Superset relationship: with this on, BW24_PP_FP8 and its budget are irrelevant for F8-origin
/// tensors (they never surface as Q8_0, so the stash arm never fires).
pub fn st_e4m3_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("BW24_ST_E4M3")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// Resident scratch for the FP8 prefill GEMM (mirrors `CutlassScratch`): the quantized activation
/// (grown to the largest m*k seen), the 4-float scale block ([0]=amax, [1]=quant mul, [2]=folded
/// B_SCALE — the GEMM desc holds a POINTER to slot 2, so the buffer must be resident/stable), and
/// the cuBLASLt workspace (64MB, the probe's size). Single GPU worker => no concurrent use; the
/// Mutex guards lazy build/grow only (matches moe_cache / cutlass_scratch).
pub struct Fp8Scratch {
    pub xq: CudaSlice<u8>,
    pub scales: CudaSlice<f32>,
    pub ws: CudaSlice<u8>,
    cap_xq: usize,
}

/// cuBLASLt workspace size — same 64MB the probe ran its heuristics with.
const FP8_WS_BYTES: usize = 64 << 20;

impl crate::Engine {
    /// FP8 prefill GEMM for a weight carrying the fp8 operand: y[m,out] = x[m,in] @ (e4m3 W)^T
    /// with the per-batch act scale and per-tensor weight_scale folded in-GEMM. Returns None when
    /// the env is off or the weight has no fp8 operand (caller falls through to the Q8_0 path).
    pub fn try_fp8_gemm(
        &self,
        w: &crate::model::GpuTensor,
        x: &CudaSlice<f32>,
        m: usize,
    ) -> Result<Option<CudaSlice<f32>>, Box<dyn std::error::Error>> {
        use crate::model::GpuTensor;
        if cfg!(bw24_portable_cuda) {
            return Ok(None);
        }
        // Two e4m3 operand sources, one GEMM:
        //  * QT_F8_E4M3 (BW24_ST_E4M3): the RESIDENT decode bytes ARE the raw checkpoint e4m3 —
        //    prefill rides them directly (one copy, no budget). Unconditional: this dtype has no
        //    other prefill GEMM class, so the FP8 path is inherent to the config, not a flag.
        //  * fp8 stash (BW24_PP_FP8=1): the Q8_0-decode config's optional duplicate operand.
        let (w_bytes, w_scale, ne) = match w {
            GpuTensor::Quant {
                qtype,
                bytes,
                scale,
                ne,
                ..
            } if *qtype == crate::QT_F8_E4M3 => (bytes, *scale, ne),
            GpuTensor::Quant {
                fp8: Some(f8), ne, ..
            } if pp_fp8_enabled() => (&f8.bytes, f8.scale, ne),
            _ => return Ok(None),
        };
        let (in_f, out_f) = (ne[0] as usize, ne[1] as usize);

        // lazy build / grow the resident scratch to this m*k
        let need_xq = m * in_f;
        let mut guard = self.fp8_scratch.lock().unwrap();
        if guard.is_none() {
            *guard = Some(Fp8Scratch {
                xq: self.alloc_u8_uninit(need_xq)?,
                scales: self.alloc_uninit::<f32>(4)?,
                ws: self.alloc_u8_uninit(FP8_WS_BYTES)?,
                cap_xq: need_xq,
            });
        }
        let s = guard.as_mut().unwrap();
        if need_xq > s.cap_xq {
            s.xq = self.alloc_u8_uninit(need_xq)?;
            s.cap_xq = need_xq;
        }

        let mut y = self.uninit(m * out_f)?; // full-overwrite GEMM output: skip memset
        let rc = {
            let stream = &self.gpu.stream;
            // Hold every SyncOnDrop guard across the FFI call (same pattern as cutlass_ffi);
            // the block scope drops them before `y` is returned.
            let (w_p, _gw) = w_bytes.device_ptr(stream);
            let (x_p, _gx) = x.device_ptr(stream);
            let (q_p, _gq) = s.xq.device_ptr_mut(stream);
            let (sc_p, _gs) = s.scales.device_ptr_mut(stream);
            let (y_p, _gy) = y.device_ptr_mut(stream);
            let (ws_p, _gws) = s.ws.device_ptr_mut(stream);
            unsafe {
                bw24_fp8_pp_gemm(
                    w_p as *const core::ffi::c_void,
                    x_p as *const f32,
                    q_p as *mut core::ffi::c_void,
                    sc_p as *mut f32,
                    y_p as *mut f32,
                    m as i32,
                    out_f as i32,
                    in_f as i32,
                    w_scale,
                    ws_p as *mut core::ffi::c_void,
                    FP8_WS_BYTES,
                    stream.cu_stream() as *mut core::ffi::c_void,
                )
            }
        };
        if rc != 0 {
            return Err(format!(
                "bw24_fp8_pp_gemm rc={rc} (m={m} n={out_f} k={in_f}; 1xxxx=cudaError quant chain, \
                 2xxxx=no cublasLt algo, 3xxxx=matmul status)"
            )
            .into());
        }
        Ok(Some(y))
    }
}
