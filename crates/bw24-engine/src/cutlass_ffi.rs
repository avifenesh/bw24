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
    /// sfb_swizzled), quantize the activation to CUTLASS layout, run the GEMM with alpha=1/scale folded
    /// into the epilogue (replaces the post-matmul scale_inplace), and return y [m,n] f32 RowMajor.
    /// The CUTLASS workspace is allocated per-call (sized via the host-only query); the SyncOnDrop
    /// guards are held across the FFI in cutlass_fp4_gemm_raw.
    #[allow(clippy::too_many_arguments)]
    pub fn cutlass_fp4_gemm(&self, b_packed: &CudaSlice<u8>, sfb_swizzled: &CudaSlice<u8>,
                            x: &CudaSlice<f32>, alpha: f32, m: usize, n: usize, k: usize)
                            -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let (a_packed, sfa_sw) = self.quantize_fp4_act_cutlass(x, m, k)?;
        let alpha_d = self.htod(&[alpha])?;
        let ws_bytes = self.cutlass_fp4_workspace_size(m, n, k);
        let mut workspace = self.alloc_u8(ws_bytes.max(1))?;
        let mut y = self.alloc_uninit::<f32>(m * n)?;
        self.cutlass_fp4_gemm_raw(&a_packed, b_packed, &sfa_sw, sfb_swizzled, &alpha_d,
                                  &mut y, m, n, k, &mut workspace)?;
        Ok(y)
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
