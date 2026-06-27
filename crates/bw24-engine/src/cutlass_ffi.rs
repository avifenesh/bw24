//! CUTLASS sm_120a NVFP4 GEMM FFI (Phase 0).
//!
//! Bridges cudarc (driver API) to the CUTLASS host adapter (runtime API). They share device state via
//! the CUDA **primary context** — cudarc retains it (cuDevicePrimaryCtxRetain), so runtime-API calls
//! inside CUTLASS bind to the same context and pointers are interchangeable. Verified by the Phase-0
//! smoke test (`bin/cutlass_smoke.rs`): a real GEMM launches, cudaGetLastError()==0, result correct.
//!
//! All of this is compiled only under `cfg(bw24_cutlass)` (set by build.rs when BW24_CUTLASS is on),
//! so the default build links no CUTLASS and is unaffected.

#![cfg(bw24_cutlass)]

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};

/// Resident scratch for the CUTLASS NVFP4 prefill GEMM, allocated ONCE and grown to the largest
/// prefill GEMM shape seen, then reused for every call (no per-call cudaMalloc / alpha htod).
///
/// The per-call path used to allocate 6 buffers + an htod of a single f32 EVERY matmul (~200/prefill):
/// a_packed [m,k/2], sfa_linear [m,k/16], sfa_sw [sfa_size], workspace [ws_bytes], y [m,n], alpha [1].
/// All six are now resident here. Each buffer is sized for the MAX shape encountered and the GEMM /
/// quant kernels touch only the leading `m*..` / `n*..` prefix (row-major, contiguous from offset 0,
/// bounded by the m,n,k passed to the FFI), so passing the full resident buffer is bit-identical to a
/// freshly-allocated exact-size buffer. `y` and `a_packed`/`sfa*` are fully overwritten each call.
///
/// SINGLE-STREAM SAFETY: bw24 runs ONE GPU worker thread / one compute stream (`Engine::gpu.stream`);
/// the HTTP server serializes all GPU work on that worker, so no two CUTLASS GEMMs touch this scratch
/// concurrently. The `Mutex<Option<>>` on the Engine guards only the lazy build/grow (matches moe_cache);
/// the GEMM itself runs on `gpu.stream` under that lock for the call's duration (one worker => no contention).
pub struct CutlassScratch {
    pub workspace: CudaSlice<u8>,   // >= max over shapes of cutlass_fp4_workspace_size(m,n,k)
    pub a_packed: CudaSlice<u8>,    // >= max m*k/2
    pub sfa_linear: CudaSlice<u8>,  // >= max m*k/16
    pub sfa_sw: CudaSlice<u8>,      // >= max cutlass_sfa_size(m,k)
    pub y: CudaSlice<f32>,          // >= max m*n
    pub alpha: CudaSlice<f32>,      // resident [1], written in-place via memcpy_htod each call
    // Current capacities (in elements/bytes) so we only grow when a bigger shape appears.
    cap_ws: usize,
    cap_a: usize,
    cap_sfa_lin: usize,
    cap_sfa_sw: usize,
    cap_y: usize,
}

#[allow(dead_code)]
unsafe extern "C" {
    /// Host-only workspace size for an (m,n,k) GEMM. No launch.
    pub fn bw24_cutlass_fp4_workspace(m: i32, n: i32, k: i32) -> usize;
    /// Swizzled SFA byte count for an (m,k) activation operand.
    pub fn bw24_cutlass_sfa_size(m: i32, k: i32) -> usize;
    /// Swizzled SFB byte count for an (n,k) weight operand.
    pub fn bw24_cutlass_sfb_size(n: i32, k: i32) -> usize;
    /// Run one NVFP4 W4A4 GEMM. Returns 0 on success, else a CUTLASS-status-derived code.
    /// alpha_dev: device float* == 1/scale (epilogue LinearCombination scalar). d: f32 [m,n] RowMajor.
    pub fn bw24_cutlass_fp4_gemm(
        a_e2m1: *const core::ffi::c_void,
        b_e2m1: *const core::ffi::c_void,
        sfa: *const core::ffi::c_void,
        sfb: *const core::ffi::c_void,
        alpha_dev: *const f32,
        d: *mut core::ffi::c_void,
        m: i32, n: i32, k: i32,
        workspace: *mut core::ffi::c_void,
        workspace_bytes: usize,
        stream: *mut core::ffi::c_void,
    ) -> i32;
    /// Scatter linear [n, k/16] ue4m3 weight scales -> swizzled SFB layout. Returns cudaError as int.
    pub fn bw24_cutlass_repack_sfb(
        sfb_linear: *const core::ffi::c_void, sfb_swizzled: *mut core::ffi::c_void,
        n: i32, k: i32, stream: *mut core::ffi::c_void) -> i32;
    /// De-interleave a GGUF NVFP4 weight tensor (raw [n] rows of `row_bytes`) into the CUTLASS B
    /// operand: b_packed [n, k/2] plain K-contiguous packed e2m1, sfb_linear [n, k/16] ue4m3 scales
    /// (linear, then feed to bw24_cutlass_repack_sfb). Returns cudaError as int. One-time, at load.
    pub fn bw24_gguf_nvfp4_deinterleave(
        src: *const core::ffi::c_void, row_bytes: i64,
        b_packed: *mut core::ffi::c_void, sfb_linear: *mut core::ffi::c_void,
        n: i32, k: i32, stream: *mut core::ffi::c_void) -> i32;
    /// Scatter linear [m, k/16] ue4m3 activation scales -> swizzled SFA layout. Returns cudaError as int.
    pub fn bw24_cutlass_repack_sfa(
        sfa_linear: *const core::ffi::c_void, sfa_swizzled: *mut core::ffi::c_void,
        m: i32, k: i32, stream: *mut core::ffi::c_void) -> i32;
    /// TEST oracle: quantize [rows,k] f32 -> packed e2m1 (rows*k/2 B) + linear ue4m3 scales (rows*k/16 B).
    pub fn bw24_nvfp4_quant_ref(
        src_f32: *const core::ffi::c_void, packed_e2m1: *mut core::ffi::c_void,
        scales_linear: *mut core::ffi::c_void, rows: i32, k: i32, stream: *mut core::ffi::c_void) -> i32;
    /// TEST oracle: dequantize packed e2m1 + linear ue4m3 scales -> f32 [rows,k].
    pub fn bw24_nvfp4_dequant_ref(
        packed_e2m1: *const core::ffi::c_void, scales_linear: *const core::ffi::c_void,
        dst_f32: *mut core::ffi::c_void, rows: i32, k: i32, stream: *mut core::ffi::c_void) -> i32;
}

impl crate::Engine {
    /// Workspace size (bytes) CUTLASS needs for the largest (m,n,k) prefill GEMM. Host-only.
    pub fn cutlass_fp4_workspace_size(&self, m: usize, n: usize, k: usize) -> usize {
        unsafe { bw24_cutlass_fp4_workspace(m as i32, n as i32, k as i32) }
    }

    /// Swizzled scale-factor buffer sizes for activation (m,k) and weight (n,k) operands.
    pub fn cutlass_sfa_size(&self, m: usize, k: usize) -> usize {
        unsafe { bw24_cutlass_sfa_size(m as i32, k as i32) }
    }
    pub fn cutlass_sfb_size(&self, n: usize, k: usize) -> usize {
        unsafe { bw24_cutlass_sfb_size(n as i32, k as i32) }
    }

    /// Scatter linear ue4m3 weight scales into CUTLASS's swizzled SFB layout (one-time, at load).
    pub fn cutlass_repack_sfb(&self, sfb_linear: &CudaSlice<u8>, sfb_swizzled: &mut CudaSlice<u8>,
                              n: usize, k: usize) -> Result<(), Box<dyn std::error::Error>> {
        let stream = &self.gpu.stream;
        let (src, _g1) = sfb_linear.device_ptr(stream);
        let (dst, _g2) = sfb_swizzled.device_ptr_mut(stream);
        let rc = unsafe {
            bw24_cutlass_repack_sfb(src as *const core::ffi::c_void, dst as *mut core::ffi::c_void,
                                    n as i32, k as i32, stream.cu_stream() as *mut core::ffi::c_void)
        };
        if rc != 0 { return Err(format!("bw24_cutlass_repack_sfb cudaError={rc}").into()); }
        Ok(())
    }

    /// Scatter linear ue4m3 activation scales into CUTLASS's swizzled SFA layout (per-prefill).
    pub fn cutlass_repack_sfa(&self, sfa_linear: &CudaSlice<u8>, sfa_swizzled: &mut CudaSlice<u8>,
                              m: usize, k: usize) -> Result<(), Box<dyn std::error::Error>> {
        let stream = &self.gpu.stream;
        let (src, _g1) = sfa_linear.device_ptr(stream);
        let (dst, _g2) = sfa_swizzled.device_ptr_mut(stream);
        let rc = unsafe {
            bw24_cutlass_repack_sfa(src as *const core::ffi::c_void, dst as *mut core::ffi::c_void,
                                    m as i32, k as i32, stream.cu_stream() as *mut core::ffi::c_void)
        };
        if rc != 0 { return Err(format!("bw24_cutlass_repack_sfa cudaError={rc}").into()); }
        Ok(())
    }

    /// De-interleave a GGUF NVFP4 weight tensor into the CUTLASS B operand layout (one-time, at load).
    /// `src` = raw GGUF rows (n rows of row_bytes); produces b_packed [n,k/2] plain packed e2m1 and
    /// sfb_linear [n,k/16] ue4m3 scales (caller then scatters via cutlass_repack_sfb).
    pub fn cutlass_gguf_nvfp4_deinterleave(&self, src: &CudaSlice<u8>, row_bytes: usize,
                                           b_packed: &mut CudaSlice<u8>, sfb_linear: &mut CudaSlice<u8>,
                                           n: usize, k: usize)
                                           -> Result<(), Box<dyn std::error::Error>> {
        let stream = &self.gpu.stream;
        let (s, _g0) = src.device_ptr(stream);
        let (p, _g1) = b_packed.device_ptr_mut(stream);
        let (sc, _g2) = sfb_linear.device_ptr_mut(stream);
        let rc = unsafe {
            bw24_gguf_nvfp4_deinterleave(s as *const core::ffi::c_void, row_bytes as i64,
                                         p as *mut core::ffi::c_void, sc as *mut core::ffi::c_void,
                                         n as i32, k as i32, stream.cu_stream() as *mut core::ffi::c_void)
        };
        if rc != 0 { return Err(format!("bw24_gguf_nvfp4_deinterleave cudaError={rc}").into()); }
        Ok(())
    }

    /// Build a CUTLASS-ready NVFP4 weight (B operand) from raw GGUF bytes: de-interleave to plain
    /// packed e2m1 + scatter the per-16 ue4m3 scales into CUTLASS's swizzled SFB. One-time, at load.
    /// Returns (b_packed [n,k/2], sfb_swizzled [bw24_cutlass_sfb_size]).
    pub fn build_cutlass_weight(&self, raw: &CudaSlice<u8>, n: usize, k: usize, row_bytes: usize)
                                -> Result<(CudaSlice<u8>, CudaSlice<u8>), Box<dyn std::error::Error>> {
        let mut b_packed = self.alloc_u8(n * k / 2)?;
        let mut sfb_linear = self.alloc_u8(n * (k / 16))?;
        self.cutlass_gguf_nvfp4_deinterleave(raw, row_bytes, &mut b_packed, &mut sfb_linear, n, k)?;
        let sfb_bytes = self.cutlass_sfb_size(n, k);
        let mut sfb_sw = self.alloc_u8(sfb_bytes)?;
        self.cutlass_repack_sfb(&sfb_linear, &mut sfb_sw, n, k)?;
        Ok((b_packed, sfb_sw))
    }

    /// Quantize an activation [m,k] f32 to the CUTLASS A operand: plain packed e2m1 [m,k/2] + swizzled
    /// SFA. Uses the same NVFP4 quantizer the smoke test proved correct (CUTLASS dtype ctors), so the
    /// bytes are exactly what the GEMM decodes. Per-prefill (per-token amax). Returns (a_packed, sfa_sw).
    pub fn quantize_fp4_act_cutlass(&self, x: &CudaSlice<f32>, m: usize, k: usize)
                                    -> Result<(CudaSlice<u8>, CudaSlice<u8>), Box<dyn std::error::Error>> {
        let mut a_packed = self.alloc_u8(m * k / 2)?;
        let mut sfa_linear = self.alloc_u8(m * (k / 16))?;
        self.cutlass_nvfp4_quant_ref(x, &mut a_packed, &mut sfa_linear, m, k)?;
        let sfa_bytes = self.cutlass_sfa_size(m, k);
        let mut sfa_sw = self.alloc_u8(sfa_bytes)?;
        self.cutlass_repack_sfa(&sfa_linear, &mut sfa_sw, m, k)?;
        Ok((a_packed, sfa_sw))
    }

    /// High-level CUTLASS NVFP4 GEMM for the dispatch seam: given the repacked weight (b_packed,
    /// sfb_swizzled), quantize the activation to CUTLASS layout, run the GEMM with alpha=scale folded
    /// into the epilogue (replaces the post-matmul scale_inplace), and return y [m,n] f32 RowMajor.
    ///
    /// All scratch (a_packed, sfa_linear, sfa_sw, workspace, y, alpha) is RESIDENT: allocated once and
    /// grown to the largest prefill GEMM shape, then reused. No per-call cudaMalloc / alpha htod. The
    /// kernels touch only the leading m*../n*.. prefix of each (row-major contiguous from offset 0,
    /// bounded by m,n,k), and y/a_packed/sfa are fully overwritten — bit-identical to a per-call alloc.
    /// Returns an owned y copy (caller expects an owned CudaSlice); the resident y is the work buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn cutlass_fp4_gemm(&self, b_packed: &CudaSlice<u8>, sfb_swizzled: &CudaSlice<u8>,
                            x: &CudaSlice<f32>, alpha: f32, m: usize, n: usize, k: usize)
                            -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        // Size the scratch for THIS shape (grows if a bigger shape appears; no-op once at max).
        self.ensure_cutlass_scratch(m, n, k)?;
        let mut guard = self.cutlass_scratch.lock().unwrap();
        let s = guard.as_mut().unwrap();

        // 1) Quantize activation x[m,k] -> a_packed[m,k/2] + sfa_linear[m,k/16], then swizzle SFA.
        //    Full-overwrite of the leading prefix; resident buffers reused.
        self.cutlass_nvfp4_quant_ref(x, &mut s.a_packed, &mut s.sfa_linear, m, k)?;
        {
            let stream = &self.gpu.stream;
            let (src, _g1) = s.sfa_linear.device_ptr(stream);
            let (dst, _g2) = s.sfa_sw.device_ptr_mut(stream);
            let rc = unsafe {
                bw24_cutlass_repack_sfa(src as *const core::ffi::c_void, dst as *mut core::ffi::c_void,
                                        m as i32, k as i32, stream.cu_stream() as *mut core::ffi::c_void)
            };
            if rc != 0 { return Err(format!("bw24_cutlass_repack_sfa cudaError={rc}").into()); }
        }

        // 2) Write alpha IN PLACE into the resident [1] f32 (no fresh htod alloc per call).
        self.gpu.stream.memcpy_htod(&[alpha], &mut s.alpha)?;

        // 3) Run the GEMM into the resident y[m,n] (full overwrite), reusing the resident workspace.
        //    The raw FFI reads workspace.len() as the byte count; the resident workspace is sized to
        //    the max shape's query and CUTLASS accepts a workspace >= its requirement, so a larger
        //    resident workspace is safe for any smaller shape.
        {
            let stream = &self.gpu.stream;
            let ws_bytes = s.workspace.len();
            let (a_p, _ga) = s.a_packed.device_ptr(stream);
            let (b_p, _gb) = b_packed.device_ptr(stream);
            let (sfa_p, _gsa) = s.sfa_sw.device_ptr(stream);
            let (sfb_p, _gsb) = sfb_swizzled.device_ptr(stream);
            let (al_p, _gal) = s.alpha.device_ptr(stream);
            let (d_p, _gd) = s.y.device_ptr_mut(stream);
            let (ws_p, _gw) = s.workspace.device_ptr_mut(stream);
            let rc = unsafe {
                bw24_cutlass_fp4_gemm(
                    a_p as *const core::ffi::c_void,
                    b_p as *const core::ffi::c_void,
                    sfa_p as *const core::ffi::c_void,
                    sfb_p as *const core::ffi::c_void,
                    al_p as *const f32,
                    d_p as *mut core::ffi::c_void,
                    m as i32, n as i32, k as i32,
                    ws_p as *mut core::ffi::c_void, ws_bytes,
                    stream.cu_stream() as *mut core::ffi::c_void,
                )
            };
            if rc != 0 { return Err(format!("bw24_cutlass_fp4_gemm returned {rc} (CUTLASS status / workspace code)").into()); }
        }

        // 4) Copy the [m,n] result out of the resident work buffer into an owned slice (D2D, on-stream).
        //    The caller's downstream consumes an owned CudaSlice; the resident y stays for the next call.
        let mut out = self.alloc_uninit::<f32>(m * n)?;
        self.gpu.stream.memcpy_dtod(&s.y.slice(0..m * n), &mut out)?;
        Ok(out)
    }

    /// Lazily build / grow the resident CutlassScratch so every buffer covers the (m,n,k) shape.
    /// Each buffer is sized to the MAX shape seen; the workspace takes the max over per-shape queries
    /// (CUTLASS accepts a workspace >= its requirement). Only reallocates a buffer when a bigger shape
    /// appears — in steady prefill this fires a handful of times then never again (zero per-call alloc).
    fn ensure_cutlass_scratch(&self, m: usize, n: usize, k: usize)
                              -> Result<(), Box<dyn std::error::Error>> {
        let need_ws = self.cutlass_fp4_workspace_size(m, n, k).max(1);
        let need_a = m * k / 2;
        let need_sfa_lin = m * (k / 16);
        let need_sfa_sw = self.cutlass_sfa_size(m, k).max(1);
        let need_y = m * n;

        let mut guard = self.cutlass_scratch.lock().unwrap();
        if guard.is_none() {
            *guard = Some(CutlassScratch {
                workspace: self.alloc_u8(need_ws)?,
                a_packed: self.alloc_u8(need_a)?,
                sfa_linear: self.alloc_u8(need_sfa_lin)?,
                sfa_sw: self.alloc_u8(need_sfa_sw)?,
                y: self.alloc_uninit::<f32>(need_y)?,
                alpha: self.alloc_uninit::<f32>(1)?,
                cap_ws: need_ws, cap_a: need_a, cap_sfa_lin: need_sfa_lin,
                cap_sfa_sw: need_sfa_sw, cap_y: need_y,
            });
            return Ok(());
        }
        let s = guard.as_mut().unwrap();
        if need_ws > s.cap_ws { s.workspace = self.alloc_u8(need_ws)?; s.cap_ws = need_ws; }
        if need_a > s.cap_a { s.a_packed = self.alloc_u8(need_a)?; s.cap_a = need_a; }
        if need_sfa_lin > s.cap_sfa_lin { s.sfa_linear = self.alloc_u8(need_sfa_lin)?; s.cap_sfa_lin = need_sfa_lin; }
        if need_sfa_sw > s.cap_sfa_sw { s.sfa_sw = self.alloc_u8(need_sfa_sw)?; s.cap_sfa_sw = need_sfa_sw; }
        if need_y > s.cap_y { s.y = self.alloc_uninit::<f32>(need_y)?; s.cap_y = need_y; }
        Ok(())
    }

    /// TEST oracle: quantize a [rows,k] f32 device matrix to packed e2m1 + linear ue4m3 scales using
    /// CUTLASS's own dtype constructors (so bytes match what the GEMM decodes). Phase-0 smoke test only.
    pub fn cutlass_nvfp4_quant_ref(&self, src: &CudaSlice<f32>, packed: &mut CudaSlice<u8>,
                                   scales_linear: &mut CudaSlice<u8>, rows: usize, k: usize)
                                   -> Result<(), Box<dyn std::error::Error>> {
        let stream = &self.gpu.stream;
        let (s, _g0) = src.device_ptr(stream);
        let (p, _g1) = packed.device_ptr_mut(stream);
        let (sc, _g2) = scales_linear.device_ptr_mut(stream);
        let rc = unsafe {
            bw24_nvfp4_quant_ref(s as *const core::ffi::c_void, p as *mut core::ffi::c_void,
                                 sc as *mut core::ffi::c_void, rows as i32, k as i32,
                                 stream.cu_stream() as *mut core::ffi::c_void)
        };
        if rc != 0 { return Err(format!("bw24_nvfp4_quant_ref cudaError={rc}").into()); }
        Ok(())
    }

    /// TEST oracle: dequantize packed e2m1 + linear ue4m3 scales back to f32 [rows,k] (Phase-0 only).
    pub fn cutlass_nvfp4_dequant_ref(&self, packed: &CudaSlice<u8>, scales_linear: &CudaSlice<u8>,
                                     dst: &mut CudaSlice<f32>, rows: usize, k: usize)
                                     -> Result<(), Box<dyn std::error::Error>> {
        let stream = &self.gpu.stream;
        let (p, _g0) = packed.device_ptr(stream);
        let (sc, _g1) = scales_linear.device_ptr(stream);
        let (d, _g2) = dst.device_ptr_mut(stream);
        let rc = unsafe {
            bw24_nvfp4_dequant_ref(p as *const core::ffi::c_void, sc as *const core::ffi::c_void,
                                   d as *mut core::ffi::c_void, rows as i32, k as i32,
                                   stream.cu_stream() as *mut core::ffi::c_void)
        };
        if rc != 0 { return Err(format!("bw24_nvfp4_dequant_ref cudaError={rc}").into()); }
        Ok(())
    }

    /// Run one NVFP4 W4A4 GEMM through CUTLASS. Operands must already be in CUTLASS layout:
    ///   a_e2m1 [m,k/2] RowMajor packed e2m1; b_e2m1 [n,k/2] K-major packed e2m1;
    ///   sfa/sfb swizzled (see cutlass_repack_sf{a,b}); alpha_dev a device float* == 1/scale.
    /// Output is f32 [m,n] RowMajor. The SyncOnDrop guards are held across the whole FFI call.
    #[allow(clippy::too_many_arguments)]
    pub fn cutlass_fp4_gemm_raw(
        &self,
        a_e2m1: &CudaSlice<u8>, b_e2m1: &CudaSlice<u8>,
        sfa: &CudaSlice<u8>, sfb: &CudaSlice<u8>,
        alpha_dev: &CudaSlice<f32>,
        d: &mut CudaSlice<f32>,
        m: usize, n: usize, k: usize,
        workspace: &mut CudaSlice<u8>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let stream = &self.gpu.stream;
        let ws_bytes = workspace.len();
        // Hold every SyncOnDrop guard for the lifetime of the FFI call (cudarc's sync-on-drop token).
        let (a_p, _ga) = a_e2m1.device_ptr(stream);
        let (b_p, _gb) = b_e2m1.device_ptr(stream);
        let (sfa_p, _gsa) = sfa.device_ptr(stream);
        let (sfb_p, _gsb) = sfb.device_ptr(stream);
        let (al_p, _gal) = alpha_dev.device_ptr(stream);
        let (d_p, _gd) = d.device_ptr_mut(stream);
        let (ws_p, _gw) = workspace.device_ptr_mut(stream);
        let rc = unsafe {
            bw24_cutlass_fp4_gemm(
                a_p as *const core::ffi::c_void,
                b_p as *const core::ffi::c_void,
                sfa_p as *const core::ffi::c_void,
                sfb_p as *const core::ffi::c_void,
                al_p as *const f32,
                d_p as *mut core::ffi::c_void,
                m as i32, n as i32, k as i32,
                ws_p as *mut core::ffi::c_void, ws_bytes,
                stream.cu_stream() as *mut core::ffi::c_void,
            )
        };
        if rc != 0 { return Err(format!("bw24_cutlass_fp4_gemm returned {rc} (CUTLASS status / workspace code)").into()); }
        Ok(())
    }
}
