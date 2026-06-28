//! FFI to the VENDORED llama NVFP4 W4A4 MMQ prefill GEMM (cu/llama_mmq_nvfp4.cu).
//!
//! This is the 5150-pp512 kernel from llama.cpp, ggml-decoupled into a static lib with a C-ABI host
//! launcher. The launcher quantizes the f32 activation to block_fp4_mmq internally (llama's 2-level
//! FP8-e8m0/UE4M3 scale = the accurate W4A8-via-FP8 path that fixes bw24's W4A4 maxdiff 1.46), then
//! launches the native mxf4nvf4 block-scale tensor-core mma. Dispatched behind BW24_MMQ=1.
//!
//! Always built (no external deps) — unlike cutlass_ffi which is BW24_CUTLASS-gated.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use crate::Engine;

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
        in_f: i32, out_f: i32, n_tokens: i32,
        act_scratch: *mut core::ffi::c_void,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

impl Engine {
    /// True if `w` is an NVFP4 weight eligible for the vendored MMQ GEMM (in_f % 64 == 0).
    pub fn mmq_supports(&self, w: &crate::model::GpuTensor) -> bool {
        use crate::model::GpuTensor;
        matches!(w, GpuTensor::Quant { qtype, .. } if *qtype == crate::QT_NVFP4)
            && w.in_features() % 64 == 0
    }

    /// Run the vendored llama NVFP4 MMQ prefill GEMM from raw weight bytes + f32 activation.
    /// y[m, out_f] = x[m, in_f] @ W^T. Applies the per-tensor NVFP4 macro-scale post (scale==1.0 no-op).
    /// `x` is the RAW f32 activation (the launcher quantizes it to block_fp4_mmq internally).
    pub fn qmatvec_mmq_nvfp4(&self, bytes: &CudaSlice<u8>, x: &CudaSlice<f32>, m: usize,
                             in_f: usize, out_f: usize, scale: f32)
                             -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        assert!(in_f % 64 == 0, "MMQ NVFP4 requires in_f % 64 == 0, got {in_f}");
        let mut y = self.qmatvec_mmq_nvfp4_raw(bytes, x, m, in_f, out_f)?;
        if scale != 1.0 { self.scale_inplace(&mut y, scale, m * out_f)?; }
        Ok(y)
    }

    /// Bare MMQ launch (no macro-scale) — for the kernel_check accuracy gate.
    pub fn qmatvec_mmq_nvfp4_raw(&self, bytes: &CudaSlice<u8>, x: &CudaSlice<f32>, m: usize,
                                 in_f: usize, out_f: usize)
                                 -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        assert!(in_f % 64 == 0, "MMQ NVFP4 requires in_f % 64 == 0, got {in_f}");
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
                    in_f as i32, out_f as i32, m as i32,
                    s_p as *mut core::ffi::c_void,
                    stream.cu_stream() as *mut core::ffi::c_void,
                )
            };
            if rc != 0 { return Err(format!("bw24_mmq_nvfp4 rc={rc}").into()); }
        }
        Ok(y)
    }
}
