//! FP16-MIRROR PREFILL (BW24_PP_F16=1): cuBLASLt FP16 TN GEMM on a resident fp16 dequant
//! mirror of the Q8_0 trunk weights.
//!
//! Probe verdict 2026-07-26 (tools/bench_lt_f16.cu on the H100 box): 611-687 TF at the 9B
//! m=512 prefill shapes vs the vendored MMQ per-shape medians = **3.2-3.7x per launch**
//! (MMQ = 60% of prime). Why fp16 and not faster int8: the exact wgmma arc proved Q8_0's
//! per-32-block scale fold serializes Hopper's warpgroup MMA pipe (ptxas C7514, ledger'd);
//! fp16 f32-accumulate has no mid-loop accumulator reads and streams at tensor-core rate.
//!
//! NUMERIC CONFIG (new, explicit, opt-in — BW24_PP_FP8/GDN-chunked precedent): the int8 part
//! of the dequant is exact in fp16 (7 mantissa bits into 11); rounding enters at d*q products
//! and the activation f32->fp16 cast. run-gen argmax battery + kernel-check tolerance gate
//! arbitrate. Decode (m<16) keeps the Q8_0 dp4a/MMVQ chain untouched — decode==verify law holds.
//!
//! VRAM: the mirror duplicates every 2D Q8_0 projection at 2 B/w (9B model ~+17GB) — an 80GB
//! H100 lane feature. BW24_PP_F16_BUDGET_MB (default 32768) caps the spend, layer-order prefix.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

unsafe extern "C" {
    /// One FP16 prefill GEMM: f32->fp16 activation convert + cublasLtMatmul TN on one stream.
    fn bw24_f16_pp_gemm(
        w_f16: *const core::ffi::c_void,
        x_f32: *const f32,
        xh_f16: *mut core::ffi::c_void,
        y_f32: *mut f32,
        m: i32,
        n: i32,
        k: i32,
        ws: *mut core::ffi::c_void,
        ws_bytes: usize,
        stream: *mut core::ffi::c_void,
    ) -> i32;
    /// Standalone f32->fp16 convert (grouped dispatch: one convert feeds N GEMMs).
    fn bw24_f16_cvt(
        x_f32: *const f32,
        xh_f16: *mut core::ffi::c_void,
        nelem: usize,
        stream: *mut core::ffi::c_void,
    ) -> i32;
    /// GEMM on a PRE-CONVERTED fp16 activation (see bw24_f16_cvt).
    fn bw24_f16_pp_gemm_pre(
        w_f16: *const core::ffi::c_void,
        xh_f16: *const core::ffi::c_void,
        y_f32: *mut f32,
        m: i32,
        n: i32,
        k: i32,
        ws: *mut core::ffi::c_void,
        ws_bytes: usize,
        stream: *mut core::ffi::c_void,
    ) -> i32;
    /// GGUF Q8_0 34B blocks -> row-major fp16 mirror (load-time).
    fn bw24_q8_0_dequant_f16(
        w_q8: *const core::ffi::c_void,
        w_f16: *mut core::ffi::c_void,
        out_f: i64,
        nblk_row: i64,
        stream: *mut core::ffi::c_void,
    ) -> i32;
    /// GGUF Q4_0 18B blocks -> row-major fp16 mirror (campaign A, 2026-07-31).
    fn bw24_q4_0_dequant_f16(
        w_q4: *const core::ffi::c_void,
        w_f16: *mut core::ffi::c_void,
        out_f: i64,
        nblk_row: i64,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

/// BW24_PP_F16 gate, read once. DEFAULT ON on the Hopper lane (80GB — the mirror costs
/// 2 B/w, ~17GB on the 9B; the box carries it), opt-in elsewhere; =1/=0 overrides either way.
/// Promotion battery (2026-07-26, H100): kernel-check ALL GREEN (f16 rel <= 6.5e-3, band 1e-2);
/// run-gen argmax MATCH on p1/p2/p3 long prompts; greedy streams IDENTICAL to the MMQ config
/// on all three; pp512 8674 -> 15626 tok/s (+80%, N=5 medians). Decode untouched (m>=16 arm).
pub fn pp_f16_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("BW24_PP_F16").as_deref() {
        Ok("1") => true,
        Ok("0") => false,
        _ => cfg!(bw24_hopper_mma),
    })
}

/// Resident scratch (fp8_ffi::Fp8Scratch pattern): fp16 activation (grown to the largest m*k
/// seen) + the cuBLASLt workspace. Single GPU worker; the Mutex guards lazy build/grow only.
pub struct F16Scratch {
    pub xh: CudaSlice<u8>,
    pub ws: CudaSlice<u8>,
    cap_xh: usize,
}

impl F16Scratch {
    /// Pre-sized scratch (task #14: the captured prime gets a PRIVATE scratch so the
    /// graph's baked cvt/Lt pointers are never mutated by eager GEMMs between replays).
    pub fn with_capacity(e: &crate::Engine, xh_bytes: usize)
                         -> Result<Self, Box<dyn std::error::Error>> {
        Ok(F16Scratch {
            xh: e.alloc_u8_uninit(xh_bytes)?,
            ws: e.alloc_u8_uninit(F16_WS_BYTES)?,
            cap_xh: xh_bytes,
        })
    }
}

const F16_WS_BYTES: usize = 64 << 20;

impl crate::Engine {
    /// Swap the resident f16 scratch (task #14 capture isolation). Returns the previous
    /// contents; pass them back to restore.
    pub fn f16_scratch_swap(&self, new: Option<F16Scratch>) -> Option<F16Scratch> {
        std::mem::replace(&mut *self.f16_scratch.lock().unwrap(), new)
    }

    /// FP16 prefill GEMM for a weight carrying the f16 mirror: y[m,out] = x[m,in] @ (fp16 W)^T,
    /// f32 accumulate. Returns None when the weight has no mirror (caller falls through to MMQ).
    pub fn try_f16_gemm(
        &self,
        w: &crate::model::GpuTensor,
        x: &CudaSlice<f32>,
        m: usize,
    ) -> Result<Option<CudaSlice<f32>>, Box<dyn std::error::Error>> {
        use crate::model::GpuTensor;
        let (w16, ne, scale) = match w {
            GpuTensor::Quant {
                f16: Some(w16),
                ne,
                scale,
                ..
            } => (w16, ne, *scale),
            _ => return Ok(None),
        };
        let (in_f, out_f) = (ne[0] as usize, ne[1] as usize);
        let mut y = self.qmatvec_gemm_f16_raw(w16, x, m, in_f, out_f)?;
        if scale != 1.0 {
            self.scale_inplace(&mut y, scale, m * out_f)?;
        }
        Ok(Some(y))
    }

    /// Bare FP16 GEMM launch on an fp16 mirror — also the kernel_check gate entry.
    pub fn qmatvec_gemm_f16_raw(
        &self,
        w16: &CudaSlice<u8>,
        x: &CudaSlice<f32>,
        m: usize,
        in_f: usize,
        out_f: usize,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let need_xh = m * in_f * 2;
        let mut guard = self.f16_scratch.lock().unwrap();
        if guard.is_none() {
            *guard = Some(F16Scratch {
                xh: self.alloc_u8_uninit(need_xh)?,
                ws: self.alloc_u8_uninit(F16_WS_BYTES)?,
                cap_xh: need_xh,
            });
        }
        let s = guard.as_mut().unwrap();
        if need_xh > s.cap_xh {
            s.xh = self.alloc_u8_uninit(need_xh)?;
            s.cap_xh = need_xh;
        }
        let mut y = self.uninit(m * out_f)?; // full-overwrite GEMM output: skip memset
        let rc = {
            let stream = &self.gpu.stream;
            let (w_p, _gw) = w16.device_ptr(stream);
            let (x_p, _gx) = x.device_ptr(stream);
            let (h_p, _gh) = s.xh.device_ptr_mut(stream);
            let (y_p, _gy) = y.device_ptr_mut(stream);
            let (ws_p, _gws) = s.ws.device_ptr_mut(stream);
            unsafe {
                bw24_f16_pp_gemm(
                    w_p as *const core::ffi::c_void,
                    x_p as *const f32,
                    h_p as *mut core::ffi::c_void,
                    y_p as *mut f32,
                    m as i32,
                    out_f as i32,
                    in_f as i32,
                    ws_p as *mut core::ffi::c_void,
                    F16_WS_BYTES,
                    stream.cu_stream() as *mut core::ffi::c_void,
                )
            }
        };
        if rc != 0 {
            return Err(format!(
                "bw24_f16_pp_gemm rc={rc} (m={m} n={out_f} k={in_f}; 1xxxx=cudaError convert, \
                 2xxxx=no cublasLt algo, 3xxxx=matmul status)"
            )
            .into());
        }
        Ok(y)
    }

    /// f32 -> fp16 activation convert into a fresh buffer (matmul_group: ONE convert feeds
    /// every mirror-carrying weight in the group; the standalone per-GEMM converts were ~250
    /// launches/prime of gap-cluster fuel, nsys 2026-07-26).
    pub fn f16_act(
        &self,
        x: &CudaSlice<f32>,
        nelem: usize,
    ) -> Result<CudaSlice<u8>, Box<dyn std::error::Error>> {
        let mut xh = self.alloc_u8_uninit(nelem * 2)?;
        let rc = {
            let stream = &self.gpu.stream;
            let (x_p, _gx) = x.device_ptr(stream);
            let (h_p, _gh) = xh.device_ptr_mut(stream);
            unsafe {
                bw24_f16_cvt(
                    x_p as *const f32,
                    h_p as *mut core::ffi::c_void,
                    nelem,
                    stream.cu_stream() as *mut core::ffi::c_void,
                )
            }
        };
        if rc != 0 {
            return Err(format!("bw24_f16_cvt rc={rc}").into());
        }
        Ok(xh)
    }

    /// `_into` twin of `try_f16_gemm_pre` (piecewise-slab plumbing): the GEMM writes into
    /// a caller-provided buffer (a resident slab view) instead of a fresh allocation —
    /// the FFI has always taken the y pointer; only the wrapper allocated. Returns
    /// Ok(false) when the weight has no mirror (caller falls back and copies).
    pub fn try_f16_gemm_pre_into(
        &self,
        w: &crate::model::GpuTensor,
        xh: &CudaSlice<u8>,
        m: usize,
        y: &mut CudaSlice<f32>,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        use crate::model::GpuTensor;
        let (w16, ne, scale) = match w {
            GpuTensor::Quant { f16: Some(w16), ne, scale, .. } => (w16, ne, *scale),
            _ => return Ok(false),
        };
        let (in_f, out_f) = (ne[0] as usize, ne[1] as usize);
        assert!(y.len() >= m * out_f, "try_f16_gemm_pre_into: output slab too small");
        let mut guard = self.f16_scratch.lock().unwrap();
        if guard.is_none() {
            *guard = Some(F16Scratch {
                xh: self.alloc_u8_uninit(2)?,
                ws: self.alloc_u8_uninit(F16_WS_BYTES)?,
                cap_xh: 2,
            });
        }
        let s = guard.as_mut().unwrap();
        let rc = {
            let stream = &self.gpu.stream;
            let (w_p, _gw) = w16.device_ptr(stream);
            let (h_p, _gh) = xh.device_ptr(stream);
            let (y_p, _gy) = y.device_ptr_mut(stream);
            let (ws_p, _gws) = s.ws.device_ptr_mut(stream);
            unsafe {
                bw24_f16_pp_gemm_pre(
                    w_p as *const core::ffi::c_void,
                    h_p as *const core::ffi::c_void,
                    y_p as *mut f32,
                    m as i32,
                    out_f as i32,
                    in_f as i32,
                    ws_p as *mut core::ffi::c_void,
                    F16_WS_BYTES,
                    stream.cu_stream() as *mut core::ffi::c_void,
                )
            }
        };
        if rc != 0 {
            return Err(format!("bw24_f16_pp_gemm_pre(into) rc={rc} (m={m} n={out_f} k={in_f})").into());
        }
        if scale != 1.0 {
            self.scale_inplace(y, scale, m * out_f)?;
        }
        Ok(true)
    }

    /// `_into` at a ROW OFFSET (task #16): the batched prime's per-seq out-GEMMs write
    /// straight into the concat `mixed` trunk at offs[s] — removing the per-seq gather
    /// copy. off_elems must keep the pointer's alignment class (n_embd rows do).
    pub fn try_f16_gemm_pre_into_off(
        &self,
        w: &crate::model::GpuTensor,
        xh: &CudaSlice<u8>,
        m: usize,
        y: &mut CudaSlice<f32>,
        off_elems: usize,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        use crate::model::GpuTensor;
        let (w16, ne, scale) = match w {
            GpuTensor::Quant { f16: Some(w16), ne, scale, .. } => (w16, ne, *scale),
            _ => return Ok(false),
        };
        let (in_f, out_f) = (ne[0] as usize, ne[1] as usize);
        assert!(y.len() >= off_elems + m * out_f, "try_f16_gemm_pre_into_off: output slab too small");
        if scale != 1.0 {
            return Ok(false); // post-scale would need a strided view; caller falls back
        }
        let mut guard = self.f16_scratch.lock().unwrap();
        if guard.is_none() {
            *guard = Some(F16Scratch {
                xh: self.alloc_u8_uninit(2)?,
                ws: self.alloc_u8_uninit(F16_WS_BYTES)?,
                cap_xh: 2,
            });
        }
        let s = guard.as_mut().unwrap();
        let rc = {
            let stream = &self.gpu.stream;
            let (w_p, _gw) = w16.device_ptr(stream);
            let (h_p, _gh) = xh.device_ptr(stream);
            let (y_p, _gy) = y.device_ptr_mut(stream);
            let (ws_p, _gws) = s.ws.device_ptr_mut(stream);
            unsafe {
                bw24_f16_pp_gemm_pre(
                    w_p as *const core::ffi::c_void,
                    h_p as *const core::ffi::c_void,
                    (y_p as *mut f32).add(off_elems),
                    m as i32,
                    out_f as i32,
                    in_f as i32,
                    ws_p as *mut core::ffi::c_void,
                    F16_WS_BYTES,
                    stream.cu_stream() as *mut core::ffi::c_void,
                )
            }
        };
        if rc != 0 {
            return Err(format!("bw24_f16_pp_gemm_pre(into_off) rc={rc} (m={m} n={out_f} k={in_f})").into());
        }
        Ok(true)
    }

    /// FP16 GEMM on a pre-converted activation — the matmul_group arm. Same contract as
    /// `try_f16_gemm` minus the convert.
    pub fn try_f16_gemm_pre(
        &self,
        w: &crate::model::GpuTensor,
        xh: &CudaSlice<u8>,
        m: usize,
    ) -> Result<Option<CudaSlice<f32>>, Box<dyn std::error::Error>> {
        use crate::model::GpuTensor;
        let (w16, ne, scale) = match w {
            GpuTensor::Quant {
                f16: Some(w16),
                ne,
                scale,
                ..
            } => (w16, ne, *scale),
            _ => return Ok(None),
        };
        let (in_f, out_f) = (ne[0] as usize, ne[1] as usize);
        // workspace from the shared scratch (xh is caller-owned here)
        let mut guard = self.f16_scratch.lock().unwrap();
        if guard.is_none() {
            *guard = Some(F16Scratch {
                xh: self.alloc_u8_uninit(2)?,
                ws: self.alloc_u8_uninit(F16_WS_BYTES)?,
                cap_xh: 2,
            });
        }
        let s = guard.as_mut().unwrap();
        let mut y = self.uninit(m * out_f)?;
        let rc = {
            let stream = &self.gpu.stream;
            let (w_p, _gw) = w16.device_ptr(stream);
            let (h_p, _gh) = xh.device_ptr(stream);
            let (y_p, _gy) = y.device_ptr_mut(stream);
            let (ws_p, _gws) = s.ws.device_ptr_mut(stream);
            unsafe {
                bw24_f16_pp_gemm_pre(
                    w_p as *const core::ffi::c_void,
                    h_p as *const core::ffi::c_void,
                    y_p as *mut f32,
                    m as i32,
                    out_f as i32,
                    in_f as i32,
                    ws_p as *mut core::ffi::c_void,
                    F16_WS_BYTES,
                    stream.cu_stream() as *mut core::ffi::c_void,
                )
            }
        };
        if rc != 0 {
            return Err(format!("bw24_f16_pp_gemm_pre rc={rc} (m={m} n={out_f} k={in_f})").into());
        }
        if scale != 1.0 {
            self.scale_inplace(&mut y, scale, m * out_f)?;
        }
        Ok(Some(y))
    }

    /// Raw fp16 mirror build from GGUF Q8_0 device bytes (gates/benches; also the loader's
    /// worker via `build_q8_f16`).
    pub fn build_q8_f16_raw(
        &self,
        bytes: &CudaSlice<u8>,
        in_f: usize,
        out_f: usize,
    ) -> Result<CudaSlice<u8>, Box<dyn std::error::Error>> {
        assert!(in_f % 32 == 0);
        let nblk = in_f / 32;
        let mut dst = self.alloc_u8_uninit(out_f * in_f * 2)?;
        let rc = {
            let stream = &self.gpu.stream;
            let (s_p, _gs) = bytes.device_ptr(stream);
            let (d_p, _gd) = dst.device_ptr_mut(stream);
            unsafe {
                bw24_q8_0_dequant_f16(
                    s_p as *const core::ffi::c_void,
                    d_p as *mut core::ffi::c_void,
                    out_f as i64,
                    nblk as i64,
                    stream.cu_stream() as *mut core::ffi::c_void,
                )
            }
        };
        if rc != 0 {
            return Err(format!("bw24_q8_0_dequant_f16 rc={rc}").into());
        }
        Ok(dst)
    }

    /// Load-time fp16 mirror pass for one tensor (hybrid.rs calls this under BW24_PP_F16=1,
    /// next to `build_q8_rp4`). No-op unless 2D Q8_0 with integral rows and budget headroom.
    pub fn build_q8_f16(
        &self,
        t: &mut crate::model::GpuTensor,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use crate::model::GpuTensor;
        let GpuTensor::Quant {
            bytes,
            qtype,
            row_bytes,
            ne,
            f16,
            ..
        } = t
        else {
            return Ok(());
        };
        // Q4_0 admitted 2026-07-31 (campaign A): the gemma QAT trunk rides the same Lt
        // f16 lane — int4 magnitudes exact in fp16, same rounding class as Q8_0.
        let q4 = *qtype == crate::QT_Q4_0;
        if (*qtype != crate::QT_Q8_0 && !q4) || f16.is_some() || ne.len() != 2 {
            return Ok(());
        }
        let (in_f, out_f) = (ne[0] as usize, ne[1] as usize);
        if in_f % 32 != 0 || *row_bytes != (in_f / 32) * (if q4 { 18 } else { 34 }) {
            return Ok(());
        }
        // Budget (layer-order prefix, BW24_PP_FP8_BUDGET_MB pattern): default 32GB — the whole
        // 9B mirror on an 80GB box; smaller rigs set BW24_PP_F16_BUDGET_MB down.
        use std::sync::atomic::{AtomicUsize, Ordering};
        static SPENT: AtomicUsize = AtomicUsize::new(0);
        static BUDGET: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        let budget = *BUDGET.get_or_init(|| {
            std::env::var("BW24_PP_F16_BUDGET_MB")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(32768)
                << 20
        });
        let sz = out_f * in_f * 2;
        if SPENT.fetch_add(sz, Ordering::Relaxed) + sz > budget {
            SPENT.fetch_sub(sz, Ordering::Relaxed);
            return Ok(());
        }
        *f16 = Some(if q4 { self.build_q4_f16_raw(bytes, in_f, out_f)? }
                    else { self.build_q8_f16_raw(bytes, in_f, out_f)? });
        Ok(())
    }

    /// Q4_0 twin of `build_q8_f16_raw` (18B blocks, campaign A 2026-07-31).
    pub fn build_q4_f16_raw(
        &self,
        bytes: &CudaSlice<u8>,
        in_f: usize,
        out_f: usize,
    ) -> Result<CudaSlice<u8>, Box<dyn std::error::Error>> {
        assert!(in_f % 32 == 0);
        let nblk = in_f / 32;
        let mut dst = self.alloc_u8_uninit(out_f * in_f * 2)?;
        let rc = {
            let stream = &self.gpu.stream;
            let (s_p, _gs) = bytes.device_ptr(stream);
            let (d_p, _gd) = dst.device_ptr_mut(stream);
            unsafe {
                bw24_q4_0_dequant_f16(
                    s_p as *const core::ffi::c_void,
                    d_p as *mut core::ffi::c_void,
                    out_f as i64,
                    nblk as i64,
                    stream.cu_stream() as *mut core::ffi::c_void,
                )
            }
        };
        if rc != 0 {
            return Err(format!("bw24_q4_0_dequant_f16 rc={rc}").into());
        }
        Ok(dst)
    }
}
