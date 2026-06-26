//! bw24 engine: Stage-1 correctness-first forward-pass kernels + ops, on sm_120 via cudarc.

use std::sync::Arc;
use cudarc::driver::{CudaContext, CudaStream, CudaModule, CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

pub use bw24_gguf;
pub use bw24_runtime;

pub mod model;
pub mod forward;
pub mod hybrid;
pub mod hybrid_forward;
pub mod cache;
pub mod decode;
pub mod spec;

const FATBIN_PATH: &str = env!("BW24_ENGINE_FATBIN");
const HYBRID_FATBIN_PATH: &str = env!("BW24_HYBRID_FATBIN");
const QMATVEC_FATBIN_PATH: &str = env!("BW24_QMATVEC_FATBIN");
const FLASH_FATBIN_PATH: &str = env!("BW24_FLASH_FATBIN");

/// Quant type codes matching qmatvec.cu QType enum.
pub const QT_Q8_0: i32 = 0;
pub const QT_Q4_K: i32 = 1;
pub const QT_Q6_K: i32 = 2;
pub const QT_Q5_K: i32 = 3;
pub const QT_Q3_K: i32 = 4;
pub const QT_IQ4_XS: i32 = 5;
pub const QT_IQ3_S: i32 = 6;
pub const QT_NVFP4: i32 = 7;

/// Engine device context: CUDA context, stream, loaded kernel modules, cuBLASLt (via runtime::Gpu).
pub struct Engine {
    pub gpu: bw24_runtime::Gpu,
    module: Arc<CudaModule>,
    hybrid: Arc<CudaModule>,
    qmatvec: Arc<CudaModule>,
    flash: Arc<CudaModule>,
}

impl Engine {
    pub fn new(ordinal: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let gpu = bw24_runtime::Gpu::new(ordinal)?;
        let module = gpu.ctx.load_module(Ptx::from_file(FATBIN_PATH))?;
        let hybrid = gpu.ctx.load_module(Ptx::from_file(HYBRID_FATBIN_PATH))?;
        let qmatvec = gpu.ctx.load_module(Ptx::from_file(QMATVEC_FATBIN_PATH))?;
        let flash = gpu.ctx.load_module(Ptx::from_file(FLASH_FATBIN_PATH))?;
        Ok(Self { gpu, module, hybrid, qmatvec, flash })
    }

    pub fn ctx(&self) -> &Arc<CudaContext> { &self.gpu.ctx }
    pub fn stream(&self) -> &Arc<CudaStream> { &self.gpu.stream }
    fn func(&self, name: &str) -> CudaFunction {
        self.module.load_function(name)
            .or_else(|_| self.hybrid.load_function(name))
            .or_else(|_| self.qmatvec.load_function(name))
            .or_else(|_| self.flash.load_function(name))
            .unwrap_or_else(|_| panic!("kernel {name} not in any fatbin"))
    }

    pub fn htod_bytes(&self, v: &[u8]) -> Result<CudaSlice<u8>, Box<dyn std::error::Error>> {
        Ok(self.gpu.stream.clone_htod(v)?)
    }

    /// Device-to-device copy of `src` into `dst[off..off+len]` (f32). For in-place KV append.
    pub fn copy_into(&self, dst: &mut CudaSlice<f32>, off: usize, src: &CudaSlice<f32>, len: usize)
                     -> Result<(), Box<dyn std::error::Error>> {
        let mut view = dst.slice_mut(off..off + len);
        self.gpu.stream.memcpy_dtod(&src.slice(0..len), &mut view)?;
        Ok(())
    }

    /// View a sub-range of a device buffer (for attending over [0..len) of a KV cache).
    pub fn view<'a>(&self, b: &'a CudaSlice<f32>, len: usize) -> cudarc::driver::CudaView<'a, f32> {
        b.slice(0..len)
    }

    /// View the first `len` BYTES of a u8 device buffer (quantized KV cache: [0..t_kv*tok_bytes)).
    pub fn view_u8<'a>(&self, b: &'a CudaSlice<u8>, len: usize) -> cudarc::driver::CudaView<'a, u8> {
        b.slice(0..len)
    }

    /// Append-quantize ONE token's post-RoPE K (q8_0) and V (q5_1) into the resident byte caches at
    /// token index `t` (KVQUANT-PLAN §C). One CTA (one warp) per 32-element block; the kernel writes
    /// the f16 scale(s) + packed quants for K and V. k_row/v_row are f32 [kv_dim_k]/[kv_dim_v].
    pub fn append_kv_quantized(&self, k_row: &CudaSlice<f32>, v_row: &CudaSlice<f32>,
                               kc: &mut CudaSlice<u8>, vc: &mut CudaSlice<u8>, t: usize,
                               kv_dim_k: usize, kv_dim_v: usize,
                               k_tok_bytes: usize, v_tok_bytes: usize)
                               -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("append_quantize_kv_q8_0_q5_1");
        let nblk = (kv_dim_k.max(kv_dim_v) / 32) as u32;
        let cfg = LaunchConfig { grid_dim: (nblk, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
        let (ti, kdk, kdv) = (t as i32, kv_dim_k as i32, kv_dim_v as i32);
        let (ktb, vtb) = (k_tok_bytes as i64, v_tok_bytes as i64);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(k_row).arg(v_row).arg(kc).arg(vc).arg(&ti).arg(&kdk).arg(&kdv).arg(&ktb).arg(&vtb);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// Like `append_kv_quantized` but k_row/v_row are CudaViews (one token's row sliced out of a
    /// token-major [T, kv_dim] activation buffer — the MTP verify path appends T tokens).
    pub fn append_kv_quantized_view(&self, k_row: &cudarc::driver::CudaView<f32>,
                                    v_row: &cudarc::driver::CudaView<f32>,
                                    kc: &mut CudaSlice<u8>, vc: &mut CudaSlice<u8>, t: usize,
                                    kv_dim_k: usize, kv_dim_v: usize,
                                    k_tok_bytes: usize, v_tok_bytes: usize)
                                    -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("append_quantize_kv_q8_0_q5_1");
        let nblk = (kv_dim_k.max(kv_dim_v) / 32) as u32;
        let cfg = LaunchConfig { grid_dim: (nblk, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
        let (ti, kdk, kdv) = (t as i32, kv_dim_k as i32, kv_dim_v as i32);
        let (ktb, vtb) = (k_tok_bytes as i64, v_tok_bytes as i64);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(k_row).arg(v_row).arg(kc).arg(vc).arg(&ti).arg(&kdk).arg(&kdv).arg(&ktb).arg(&vtb);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// Device-to-device copy of a CudaView `src` into `dst[off..off+len]` (f32). Like `copy_into`
    /// but the source is a sub-view (e.g. one column of a token-major activation buffer).
    pub fn copy_view_into(&self, dst: &mut CudaSlice<f32>, off: usize,
                          src: &cudarc::driver::CudaView<f32>, len: usize)
                          -> Result<(), Box<dyn std::error::Error>> {
        let mut view = dst.slice_mut(off..off + len);
        self.gpu.stream.memcpy_dtod(&src.slice(0..len), &mut view)?;
        Ok(())
    }

    /// Real device-to-device COPY of `src` into a freshly allocated buffer (NOT an Arc clone).
    /// Used for cache snapshots (MTP-PLAN §D.4): `CudaSlice::clone()` only bumps a refcount and
    /// would alias the live buffer; this allocs new device memory and memcpy_dtod's the contents.
    pub fn clone_dtod(&self, src: &CudaSlice<f32>) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let mut dst = self.gpu.stream.alloc_zeros::<f32>(src.len())?;
        self.gpu.stream.memcpy_dtod(src, &mut dst)?;
        Ok(dst)
    }

    /// Resident-quantized linear (Stage-A: f32 dequant-in-kernel). y[m,out]=x[m,in]@W[out,in]^T.
    pub fn qmatvec(&self, w: &CudaSlice<u8>, x: &CudaSlice<f32>, m: usize, in_f: usize, out_f: usize,
                   qtype: i32, row_bytes: usize) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let f = self.func("qmatvec_f32");
        let mut y = self.gpu.stream.alloc_zeros::<f32>(m * out_f)?;
        let cfg = LaunchConfig { grid_dim: (out_f as u32, m as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let (inf, outf, mi, qt, rb) = (in_f as i32, out_f as i32, m as i32, qtype, row_bytes as i64);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(w).arg(x).arg(&mut y).arg(&inf).arg(&outf).arg(&mi).arg(&qt).arg(&rb);
        unsafe { b.launch(cfg)?; }
        Ok(y)
    }

    /// Allocate a reusable u8 GPU scratch buffer (for staged expert weights).
    pub fn alloc_u8(&self, n: usize) -> Result<CudaSlice<u8>, Box<dyn std::error::Error>> {
        Ok(self.gpu.stream.alloc_zeros::<u8>(n)?)
    }

    /// EDGE-1 staging: copy `host_bytes` (a sub-slice of a HostExps buffer) into `scratch`
    /// at byte offset `off` (async H2D on the default stream). Length is host_bytes.len().
    /// The qmatvec_view that reads `scratch[off..]` is enqueued on the SAME stream after this,
    /// so ordering is guaranteed without an explicit sync (Stage-1; Stage-2 prefetch on a 2nd
    /// stream would require an event).
    pub fn stage_expert(&self, host_bytes: &[u8], scratch: &mut CudaSlice<u8>, off: usize)
                        -> Result<(), Box<dyn std::error::Error>> {
        let mut dst = scratch.slice_mut(off..off + host_bytes.len());  // CudaViewMut<u8>
        self.gpu.stream.memcpy_htod(host_bytes, &mut dst)?;            // accepts &[u8] HostSlice src
        Ok(())
    }

    /// qmatvec over a byte sub-range of a (resident/scratch) CudaSlice<u8> holding ONE expert
    /// matrix. x is a CudaView<f32> (a sliced row of z, or a sliced activation). Reuses the
    /// validated qmatvec_f32 dequant path (NOT a fast path — the correctness gate). The
    /// CudaView base+offset pointer is honored by the launch arg.
    pub fn qmatvec_view(&self, w: &CudaSlice<u8>, range: std::ops::Range<usize>,
                        x: &cudarc::driver::CudaView<f32>, m: usize, in_f: usize, out_f: usize,
                        qtype: i32, row_bytes: usize)
                        -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let f = self.func("qmatvec_f32");
        let wv = w.slice(range);  // CudaView<u8>, offset honored
        let mut y = self.gpu.stream.alloc_zeros::<f32>(m * out_f)?;
        let cfg = LaunchConfig { grid_dim: (out_f as u32, m as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let (inf, outf, mi, qt, rb) = (in_f as i32, out_f as i32, m as i32, qtype, row_bytes as i64);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(&wv).arg(x).arg(&mut y).arg(&inf).arg(&outf).arg(&mi).arg(&qt).arg(&rb);
        unsafe { b.launch(cfg)?; }
        Ok(y)
    }

    /// dst[i] += alpha * src[i], i in 0..n. dst is a CudaViewMut (a row of moe_out).
    pub fn axpy_into(&self, src: &CudaSlice<f32>, alpha: f32,
                     dst: &mut cudarc::driver::CudaViewMut<f32>, n: usize)
                     -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("axpy_f32");
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let (a, ni) = (alpha, n as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(src).arg(dst).arg(&a).arg(&ni);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// dst[r*ncols + c] += src[r*ncols + c] * scale[r]. Per-row scalar accumulate (shared expert).
    pub fn add_scaled_rows(&self, src: &CudaSlice<f32>, scale: &CudaSlice<f32>,
                           dst: &mut CudaSlice<f32>, ncols: usize, nrows: usize)
                           -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("add_scaled_rows_f32");
        let cfg = LaunchConfig::for_num_elems((ncols * nrows) as u32);
        let (nc, nr) = (ncols as i32, nrows as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(src).arg(scale).arg(dst).arg(&nc).arg(&nr);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// Stage-B: quantize activation [m,in] f32 -> q8_1 (int8 qs + per-block f32 scale).
    /// Quantize an activation [m, in_f] to q8_1 (int8 qs + per-32 f32 scale). Public so the
    /// forward can quantize a SHARED activation ONCE and feed it to several matmuls (gate+up
    /// share `z`; q/k/v and wqkv/gate/beta/alpha share `h`) — quantize_q8_1 was 13.5% of decode
    /// GPU time, ~half of it redundant re-quantization of the same row.
    pub fn quantize_q8_1(&self, x: &CudaSlice<f32>, m: usize, in_f: usize)
                     -> Result<(CudaSlice<i8>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let f = self.func("quantize_q8_1");
        let nblk = in_f / 32;
        let mut q = self.gpu.stream.alloc_zeros::<i8>(m * in_f)?;
        let mut d = self.gpu.stream.alloc_zeros::<f32>(m * nblk)?;
        let cfg = LaunchConfig::for_num_elems((m * nblk) as u32);
        let (inf, mi) = (in_f as i32, m as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(x).arg(&mut q).arg(&mut d).arg(&inf).arg(&mi);
        unsafe { b.launch(cfg)?; }
        Ok((q, d))
    }

    /// Stage-B: Q8_0 weight x q8_1 activation int8 dp4a matmul. y[m,out]=x@W^T.
    pub fn qmatvec_q8_0_fast(&self, w: &CudaSlice<u8>, x: &CudaSlice<f32>, m: usize, in_f: usize,
                             out_f: usize, row_bytes: usize) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let (aq, ad) = self.quantize_q8_1(x, m, in_f)?;
        let f = self.func("qmatvec_q8_0_dp4a");
        let mut y = self.gpu.stream.alloc_zeros::<f32>(m * out_f)?;
        let cfg = LaunchConfig { grid_dim: (out_f as u32, m as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
        let (inf, outf, mi, rb) = (in_f as i32, out_f as i32, m as i32, row_bytes as i64);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(w).arg(&aq).arg(&ad).arg(&mut y).arg(&inf).arg(&outf).arg(&mi).arg(&rb);
        unsafe { b.launch(cfg)?; }
        Ok(y)
    }

    /// Stage-B: Q4_K weight x q8_1 activation int8 dp4a (decode). Min-offset via q8_1 sum term.
    pub fn qmatvec_q4_K_fast(&self, w: &CudaSlice<u8>, x: &CudaSlice<f32>, m: usize, in_f: usize,
                             out_f: usize, row_bytes: usize) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let (aq, ad) = self.quantize_q8_1(x, m, in_f)?;
        let f = self.func("qmatvec_q4_K_dp4a");
        let mut y = self.gpu.stream.alloc_zeros::<f32>(m * out_f)?;
        let cfg = LaunchConfig { grid_dim: (out_f as u32, m as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
        let (inf, outf, mi, rb) = (in_f as i32, out_f as i32, m as i32, row_bytes as i64);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(w).arg(&aq).arg(&ad).arg(&mut y).arg(&inf).arg(&outf).arg(&mi).arg(&rb);
        unsafe { b.launch(cfg)?; }
        Ok(y)
    }

    /// Stage-B: Q6_K weight x q8_1 activation int8 dp4a (decode, symmetric).
    pub fn qmatvec_q6_K_fast(&self, w: &CudaSlice<u8>, x: &CudaSlice<f32>, m: usize, in_f: usize,
                             out_f: usize, row_bytes: usize) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let (aq, ad) = self.quantize_q8_1(x, m, in_f)?;
        let f = self.func("qmatvec_q6_K_dp4a");
        let mut y = self.gpu.stream.alloc_zeros::<f32>(m * out_f)?;
        let cfg = LaunchConfig { grid_dim: (out_f as u32, m as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
        let (inf, outf, mi, rb) = (in_f as i32, out_f as i32, m as i32, row_bytes as i64);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(w).arg(&aq).arg(&ad).arg(&mut y).arg(&inf).arg(&outf).arg(&mi).arg(&rb);
        unsafe { b.launch(cfg)?; }
        Ok(y)
    }

    /// Stage-B: Q5_K weight x q8_1 activation int8 dp4a (decode). Min-offset via q8_1 sum term.
    pub fn qmatvec_q5_K_fast(&self, w: &CudaSlice<u8>, x: &CudaSlice<f32>, m: usize, in_f: usize,
                             out_f: usize, row_bytes: usize) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        self.qmatvec_dp4a_named("qmatvec_q5_K_dp4a", w, x, m, in_f, out_f, row_bytes)
    }
    /// Stage-B: Q3_K weight x q8_1 activation int8 dp4a (decode, symmetric).
    pub fn qmatvec_q3_K_fast(&self, w: &CudaSlice<u8>, x: &CudaSlice<f32>, m: usize, in_f: usize,
                             out_f: usize, row_bytes: usize) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        self.qmatvec_dp4a_named("qmatvec_q3_K_dp4a", w, x, m, in_f, out_f, row_bytes)
    }
    /// Stage-B: NVFP4 weight x q8_1 activation int8 dp4a (decode, symmetric, codebook lookup).
    pub fn qmatvec_nvfp4_fast(&self, w: &CudaSlice<u8>, x: &CudaSlice<f32>, m: usize, in_f: usize,
                              out_f: usize, row_bytes: usize) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        // B1: the NVFP4 dp4a kernel maps two 32-elem q8_1 blocks onto one 64-elem block_nvfp4
        // (sblk = g >> 1). in_f must be a multiple of 64 or the last block reads a partial superblock.
        assert!(in_f % 64 == 0, "NVFP4 dp4a requires in_f % 64 == 0, got {in_f}");
        self.qmatvec_dp4a_named("qmatvec_nvfp4_dp4a", w, x, m, in_f, out_f, row_bytes)
    }
    /// Stage-B (optional perf): IQ4_XS codebook int8 dp4a.
    pub fn qmatvec_iq4_XS_fast(&self, w: &CudaSlice<u8>, x: &CudaSlice<f32>, m: usize, in_f: usize,
                               out_f: usize, row_bytes: usize) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        self.qmatvec_dp4a_named("qmatvec_iq4_XS_dp4a", w, x, m, in_f, out_f, row_bytes)
    }

    /// Shared dp4a launcher: quantize_q8_1 then call the named kernel (grid (out,m), block 64).
    fn qmatvec_dp4a_named(&self, name: &str, w: &CudaSlice<u8>, x: &CudaSlice<f32>, m: usize,
                          in_f: usize, out_f: usize, row_bytes: usize)
                          -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let (aq, ad) = self.quantize_q8_1(x, m, in_f)?;
        let f = self.func(name);
        let mut y = self.gpu.stream.alloc_zeros::<f32>(m * out_f)?;
        let cfg = LaunchConfig { grid_dim: (out_f as u32, m as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
        let (inf, outf, mi, rb) = (in_f as i32, out_f as i32, m as i32, row_bytes as i64);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(w).arg(&aq).arg(&ad).arg(&mut y).arg(&inf).arg(&outf).arg(&mi).arg(&rb);
        unsafe { b.launch(cfg)?; }
        Ok(y)
    }

    pub fn htod(&self, v: &[f32]) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        Ok(self.gpu.stream.clone_htod(v)?)
    }
    pub fn htod_i32(&self, v: &[i32]) -> Result<CudaSlice<i32>, Box<dyn std::error::Error>> {
        Ok(self.gpu.stream.clone_htod(v)?)
    }
    pub fn dtoh(&self, d: &CudaSlice<f32>) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let v = self.gpu.stream.clone_dtoh(d)?;
        self.gpu.stream.synchronize()?;
        Ok(v)
    }
    /// Device-to-host copy of a u8 buffer (used to read back the quantized KV cache for validation).
    pub fn dtoh_u8(&self, d: &CudaSlice<u8>) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let v = self.gpu.stream.clone_dtoh(d)?;
        self.gpu.stream.synchronize()?;
        Ok(v)
    }
    pub fn zeros(&self, n: usize) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        Ok(self.gpu.stream.alloc_zeros::<f32>(n)?)
    }

    /// RMSNorm: x[ncols,nrows] row-major, weight[ncols] -> dst. One block/row, 256 threads.
    pub fn rms_norm(&self, x: &CudaSlice<f32>, w: &CudaSlice<f32>, dst: &mut CudaSlice<f32>,
                    ncols: usize, nrows: usize, eps: f32) -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("rms_norm_f32");
        let cfg = LaunchConfig { grid_dim: (nrows as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let (nc, e) = (ncols as i32, eps);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(x).arg(w).arg(dst).arg(&nc).arg(&e);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// L2 norm per row (head_dim), no weight.
    pub fn l2_norm(&self, x: &CudaSlice<f32>, dst: &mut CudaSlice<f32>, ncols: usize, nrows: usize,
                   eps: f32) -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("l2_norm_f32");
        let cfg = LaunchConfig { grid_dim: (nrows as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let (nc, e) = (ncols as i32, eps);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(x).arg(dst).arg(&nc).arg(&e);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// RoPE NEOX in-place. x:[head_dim, n_heads, n_tokens], pos:[n_tokens].
    pub fn rope_neox(&self, x: &mut CudaSlice<f32>, pos: &CudaSlice<i32>, head_dim: usize,
                     n_dims: usize, n_heads: usize, n_tokens: usize, freq_base: f32, freq_scale: f32)
                     -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("rope_neox_f32");
        let theta_scale = (freq_base).powf(-2.0 / n_dims as f32);
        let grid = (n_heads * n_tokens) as u32;
        let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: ((head_dim / 2) as u32, 1, 1), shared_mem_bytes: 0 };
        let (hd, nd, nh) = (head_dim as i32, n_dims as i32, n_heads as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(x).arg(pos).arg(&hd).arg(&nd).arg(&nh).arg(&theta_scale).arg(&freq_scale);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    pub fn silu_mul(&self, gate: &CudaSlice<f32>, up: &CudaSlice<f32>, dst: &mut CudaSlice<f32>, n: usize)
                    -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("silu_mul_f32");
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let ni = n as i32;
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(gate).arg(up).arg(dst).arg(&ni);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    pub fn add(&self, a: &CudaSlice<f32>, b_in: &CudaSlice<f32>, dst: &mut CudaSlice<f32>, n: usize)
               -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("add_f32");
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let ni = n as i32;
        let mut bld = self.gpu.stream.launch_builder(&f);
        bld.arg(a).arg(b_in).arg(dst).arg(&ni);
        unsafe { bld.launch(cfg)?; }
        Ok(())
    }

    pub fn mul(&self, a: &CudaSlice<f32>, b_in: &CudaSlice<f32>, dst: &mut CudaSlice<f32>, n: usize)
               -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("mul_f32");
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let ni = n as i32;
        let mut bld = self.gpu.stream.launch_builder(&f);
        bld.arg(a).arg(b_in).arg(dst).arg(&ni);
        unsafe { bld.launch(cfg)?; }
        Ok(())
    }

    /// Unified weight-tensor matmul: dispatches quant tensors to qmatvec (weights packed) and
    /// float tensors to cuBLASLt. y[m,out] = x[m,in] @ W[out,in]^T.
    pub fn matmul(&self, w: &crate::model::GpuTensor, x: &CudaSlice<f32>, m: usize)
                  -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        use crate::model::GpuTensor;
        let in_f = w.in_features();
        let out_f = w.out_features();
        // Stage-A (f32 dequant) is the validated correctness path. Stage-B fast Q8_0 is gated
        // behind BW24_FAST until it passes the isolation gate vs Stage-A.
        let fast = std::env::var("BW24_FAST").is_ok();
        let mut y = match w {
            GpuTensor::Quant { bytes, qtype, row_bytes, .. } if fast && *qtype == QT_Q8_0 =>
                self.qmatvec_q8_0_fast(bytes, x, m, in_f, out_f, *row_bytes)?,
            GpuTensor::Quant { bytes, qtype, row_bytes, .. } if fast && *qtype == QT_Q4_K =>
                self.qmatvec_q4_K_fast(bytes, x, m, in_f, out_f, *row_bytes)?,
            GpuTensor::Quant { bytes, qtype, row_bytes, .. } if fast && *qtype == QT_Q6_K =>
                self.qmatvec_q6_K_fast(bytes, x, m, in_f, out_f, *row_bytes)?,
            GpuTensor::Quant { bytes, qtype, row_bytes, .. } if fast && *qtype == QT_Q5_K =>
                self.qmatvec_q5_K_fast(bytes, x, m, in_f, out_f, *row_bytes)?,
            GpuTensor::Quant { bytes, qtype, row_bytes, .. } if fast && *qtype == QT_Q3_K =>
                self.qmatvec_q3_K_fast(bytes, x, m, in_f, out_f, *row_bytes)?,
            GpuTensor::Quant { bytes, qtype, row_bytes, .. } if fast && *qtype == QT_NVFP4 =>
                self.qmatvec_nvfp4_fast(bytes, x, m, in_f, out_f, *row_bytes)?,
            // IQ4_XS optional fast path (gate behind a second env var; Stage-A is the default).
            GpuTensor::Quant { bytes, qtype, row_bytes, .. }
                if fast && *qtype == QT_IQ4_XS && std::env::var("BW24_IQ_FAST").is_ok() =>
                self.qmatvec_iq4_XS_fast(bytes, x, m, in_f, out_f, *row_bytes)?,
            // B3: IQ3_S and (default) IQ4_XS use the Stage-A f32 dequant-in-kernel path. There is
            // NO qmatvec_iq3_s_dp4a / (default) iq4_XS fast kernel — do NOT add a `*qtype == QT_IQ3_S`
            // (or unconditional QT_IQ4_XS) fast guard here without first writing the matching kernel,
            // or func() will panic "kernel ... not in any fatbin".
            GpuTensor::Quant { bytes, qtype, row_bytes, .. } =>
                self.qmatvec(bytes, x, m, in_f, out_f, *qtype, *row_bytes)?,
            GpuTensor::Float { data, .. } => self.linear(x, data, m, in_f, out_f)?,
        };
        // NVFP4 per-tensor macro-scale (post-matmul). scale==1.0 for all other quants/float -> no-op.
        if let GpuTensor::Quant { scale, .. } = w {
            if *scale != 1.0 { self.scale_inplace(&mut y, *scale, m * out_f)?; }
        }
        Ok(y)
    }

    /// True if `w` would take the int8-dp4a fast path under BW24_FAST (so its activation can be
    /// pre-quantized once and shared across sibling matmuls via `matmul_pre`).
    pub fn uses_q8_1_fast(&self, w: &crate::model::GpuTensor) -> bool {
        use crate::model::GpuTensor;
        if std::env::var("BW24_FAST").is_err() { return false; }
        match w {
            GpuTensor::Quant { qtype, .. } => matches!(*qtype,
                QT_Q8_0 | QT_Q4_K | QT_Q6_K | QT_Q5_K | QT_Q3_K | QT_NVFP4)
                || (*qtype == QT_IQ4_XS && std::env::var("BW24_IQ_FAST").is_ok()),
            GpuTensor::Float { .. } => false,
        }
    }

    /// matmul with a PRE-QUANTIZED q8_1 activation (aq,ad from `quantize_q8_1`). Skips the
    /// per-matmul re-quantize so sibling matmuls that share an input (gate+up share `z`;
    /// q/k/v + wqkv/gate/beta/alpha share `h`) quantize ONCE. Caller MUST have checked
    /// `uses_q8_1_fast(w)`; falls back to plain `matmul` otherwise (Stage-A / Float / non-fast).
    pub fn matmul_pre(&self, w: &crate::model::GpuTensor, aq: &CudaSlice<i8>, ad: &CudaSlice<f32>,
                      x_fallback: &CudaSlice<f32>, m: usize)
                      -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        use crate::model::GpuTensor;
        if !self.uses_q8_1_fast(w) { return self.matmul(w, x_fallback, m); }
        let in_f = w.in_features();
        let out_f = w.out_features();
        let (bytes, qtype, row_bytes, scale) = match w {
            GpuTensor::Quant { bytes, qtype, row_bytes, scale, .. } => (bytes, *qtype, *row_bytes, *scale),
            _ => unreachable!("uses_q8_1_fast guaranteed Quant"),
        };
        let name = match qtype {
            QT_Q8_0 => "qmatvec_q8_0_dp4a", QT_Q4_K => "qmatvec_q4_K_dp4a",
            QT_Q6_K => "qmatvec_q6_K_dp4a", QT_Q5_K => "qmatvec_q5_K_dp4a",
            QT_Q3_K => "qmatvec_q3_K_dp4a", QT_NVFP4 => "qmatvec_nvfp4_dp4a",
            QT_IQ4_XS => "qmatvec_iq4_XS_dp4a",
            _ => unreachable!(),
        };
        let f = self.func(name);
        let mut y = self.gpu.stream.alloc_zeros::<f32>(m * out_f)?;
        let cfg = LaunchConfig { grid_dim: (out_f as u32, m as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
        let (inf, outf, mi, rb) = (in_f as i32, out_f as i32, m as i32, row_bytes as i64);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(bytes).arg(aq).arg(ad).arg(&mut y).arg(&inf).arg(&outf).arg(&mi).arg(&rb);
        unsafe { b.launch(cfg)?; }
        if scale != 1.0 { self.scale_inplace(&mut y, scale, m * out_f)?; }
        Ok(y)
    }

    /// y[i] *= s. NVFP4 per-tensor macro-scale broadcast over the whole output.
    pub fn scale_inplace(&self, y: &mut CudaSlice<f32>, s: f32, n: usize)
                         -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("scale_f32");
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let (sf, ni) = (s, n as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(y).arg(&sf).arg(&ni);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// On-device linear: y[m,out] = x[m,in] @ W[out,in]^T, weights row-major [out,in] (ggml).
    /// cuBLASLt col-major mapping (see bw24_runtime::Gpu::linear_f32 for the derivation).
    pub fn linear(&self, x: &CudaSlice<f32>, w: &CudaSlice<f32>, m_tokens: usize, in_f: usize, out_f: usize)
                  -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        use cudarc::cublaslt::{Matmul, MatmulConfig};
        let mut c = self.gpu.stream.alloc_zeros::<f32>(m_tokens * out_f)?;
        let cfg = MatmulConfig {
            transa: true, transb: false, transc: false,
            m: out_f as u64, n: m_tokens as u64, k: in_f as u64,
            alpha: 1.0, lda: in_f as i64, ldb: in_f as i64, beta: 0.0, ldc: out_f as i64,
            stride_a: None, stride_b: None, stride_c: None, stride_bias: None, batch_size: None,
        };
        unsafe { self.gpu.blas.matmul(cfg, w, x, &mut c, None, None)?; }
        Ok(c)
    }

    /// Naive SDPA. Q:[head_dim,n_head,T], K/V:[head_dim,n_head_kv,T_kv] -> O:[head_dim,n_head,T].
    pub fn sdpa_naive(&self, q: &CudaSlice<f32>, k: &CudaSlice<f32>, v: &CudaSlice<f32>,
                      o: &mut CudaSlice<f32>, head_dim: usize, n_head: usize, n_head_kv: usize,
                      t: usize, t_kv: usize, scale: f32, causal: bool)
                      -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("sdpa_naive_f32");
        let cfg = LaunchConfig {
            grid_dim: (n_head as u32, t as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: (t_kv * 4) as u32,
        };
        let (hd, nh, nhkv, ti, tkvi, cz) = (head_dim as i32, n_head as i32, n_head_kv as i32, t as i32, t_kv as i32, causal as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(q).arg(k).arg(v).arg(o).arg(&hd).arg(&nh).arg(&nhkv).arg(&ti).arg(&tkvi).arg(&scale).arg(&cz);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// SDPA where K/V are CudaViews into a resident KV cache (decode hot path, no host round-trip).
    pub fn sdpa_naive_view(&self, q: &CudaSlice<f32>, k: &cudarc::driver::CudaView<f32>,
                           v: &cudarc::driver::CudaView<f32>, o: &mut CudaSlice<f32>,
                           head_dim: usize, n_head: usize, n_head_kv: usize, t: usize, t_kv: usize,
                           scale: f32, causal: bool) -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("sdpa_naive_f32");
        let cfg = LaunchConfig {
            grid_dim: (n_head as u32, t as u32, 1), block_dim: (128, 1, 1),
            shared_mem_bytes: (t_kv * 4) as u32,
        };
        let (hd, nh, nhkv, ti, tkvi, cz) = (head_dim as i32, n_head as i32, n_head_kv as i32, t as i32, t_kv as i32, causal as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(q).arg(k).arg(v).arg(o).arg(&hd).arg(&nh).arg(&nhkv).arg(&ti).arg(&tkvi).arg(&scale).arg(&cz);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// Hand-written FlashAttention prefill (sm_120, FA-2 online softmax on validated mma.sync,
    /// head_dim 256, GQA, causal). Replaces sdpa_naive for T>1. Q/K/V/O [head_dim, n_head(_kv), T].
    pub fn fa_prefill(&self, q: &CudaSlice<f32>, k: &CudaSlice<f32>, v: &CudaSlice<f32>,
                      o: &mut CudaSlice<f32>, head_dim: usize, n_head: usize, n_head_kv: usize,
                      t: usize, t_kv: usize, scale: f32, causal: bool)
                      -> Result<(), Box<dyn std::error::Error>> {
        const M_ROWS: usize = 16; const BK: usize = 64;
        let f = self.func("fa_prefill_f32");
        // smem: bf16*(M*hd + 2*BK*hd + M*BK) + f32*(M*hd + M*BK + 2*M)
        let shmem = (2 * (M_ROWS * head_dim + 2 * BK * head_dim + M_ROWS * BK)
                   + 4 * (M_ROWS * head_dim + M_ROWS * BK + 2 * M_ROWS)) as u32;
        use cudarc::driver::sys::CUfunction_attribute_enum as A;
        f.set_attribute(A::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmem as i32)?;
        let cfg = LaunchConfig {
            grid_dim: ((t as u32 + 15) / 16, n_head as u32, 1),
            block_dim: (32, 1, 1), shared_mem_bytes: shmem,
        };
        let (hd, nh, nhkv, ti, tkvi, cz) = (head_dim as i32, n_head as i32, n_head_kv as i32, t as i32, t_kv as i32, causal as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(q).arg(k).arg(v).arg(o).arg(&hd).arg(&nh).arg(&nhkv).arg(&ti).arg(&tkvi).arg(&scale).arg(&cz);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// FA prefill where K/V are QUANTIZED CudaViews into the resident byte KV cache (the T=K verify
    /// path, MTP-PLAN §D.3). Uses `fa_prefill_q` (inline-dequant during stage-to-smem). The view's
    /// base+offset pointer is honored; the kernel reads [0..t_kv*tok_bytes). Q is the T fresh query
    /// rows; t = T, t_kv = cache len. k_tok_bytes/v_tok_bytes are the per-token byte strides.
    pub fn fa_prefill_view(&self, q: &CudaSlice<f32>, k: &cudarc::driver::CudaView<u8>,
                           v: &cudarc::driver::CudaView<u8>, o: &mut CudaSlice<f32>,
                           head_dim: usize, n_head: usize, n_head_kv: usize,
                           t: usize, t_kv: usize, scale: f32, causal: bool,
                           k_tok_bytes: usize, v_tok_bytes: usize)
                           -> Result<(), Box<dyn std::error::Error>> {
        const M_ROWS: usize = 16; const BK: usize = 64;
        let f = self.func("fa_prefill_q");
        let shmem = (2 * (M_ROWS * head_dim + 2 * BK * head_dim + M_ROWS * BK)
                   + 4 * (M_ROWS * head_dim + M_ROWS * BK + 2 * M_ROWS)) as u32;
        use cudarc::driver::sys::CUfunction_attribute_enum as A;
        f.set_attribute(A::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmem as i32)?;
        let cfg = LaunchConfig {
            grid_dim: ((t as u32 + 15) / 16, n_head as u32, 1),
            block_dim: (32, 1, 1), shared_mem_bytes: shmem,
        };
        let (hd, nh, nhkv, ti, tkvi, cz) = (head_dim as i32, n_head as i32, n_head_kv as i32, t as i32, t_kv as i32, causal as i32);
        let (ktb, vtb) = (k_tok_bytes as i64, v_tok_bytes as i64);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(q).arg(k).arg(v).arg(o).arg(&hd).arg(&nh).arg(&nhkv).arg(&ti).arg(&tkvi).arg(&scale).arg(&cz)
         .arg(&ktb).arg(&vtb);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// FA decode (T=1 split-K) over the resident QUANTIZED KV cache (q8_0 K / q5_1 V) as u8 views.
    /// Replaces sdpa_naive_view for decode; inline-dequants per element. k_tok_bytes/v_tok_bytes are
    /// the per-token byte strides (differ: q8_0=34*nblk, q5_1=24*nblk per token).
    pub fn fa_decode(&self, q: &CudaSlice<f32>, k: &cudarc::driver::CudaView<u8>,
                     v: &cudarc::driver::CudaView<u8>, o: &mut CudaSlice<f32>,
                     head_dim: usize, n_head: usize, n_head_kv: usize, t_kv: usize, scale: f32,
                     k_tok_bytes: usize, v_tok_bytes: usize)
                     -> Result<(), Box<dyn std::error::Error>> {
        let n_splits = ((t_kv + 255) / 256).max(1);
        let mut part_o = self.zeros(n_head * n_splits * head_dim)?;
        let mut part_m = self.zeros(n_head * n_splits)?;
        let mut part_l = self.zeros(n_head * n_splits)?;
        let (hd, nh, nhkv, tkvi, nsp) = (head_dim as i32, n_head as i32, n_head_kv as i32, t_kv as i32, n_splits as i32);
        let (ktb, vtb) = (k_tok_bytes as i64, v_tok_bytes as i64);
        let f = self.func("fa_decode_f32");
        let cfg = LaunchConfig { grid_dim: (n_head as u32, n_splits as u32, 1),
            block_dim: (head_dim as u32, 1, 1), shared_mem_bytes: (4 * (head_dim + 32)) as u32 };
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(q).arg(k).arg(v).arg(&mut part_o).arg(&mut part_m).arg(&mut part_l)
         .arg(&hd).arg(&nh).arg(&nhkv).arg(&tkvi).arg(&scale).arg(&nsp).arg(&ktb).arg(&vtb);
        unsafe { b.launch(cfg)?; }
        let fc = self.func("fa_decode_combine_f32");
        let cfg2 = LaunchConfig { grid_dim: (n_head as u32, 1, 1), block_dim: (head_dim as u32, 1, 1), shared_mem_bytes: 0 };
        let mut b2 = self.gpu.stream.launch_builder(&fc);
        b2.arg(&part_o).arg(&part_m).arg(&part_l).arg(o).arg(&hd).arg(&nh).arg(&nsp);
        unsafe { b2.launch(cfg2)?; }
        Ok(())
    }

    /// gdn_scan variant where state_in/out are CudaViews (resident SSM state, in-place per step).
    pub fn gdn_scan_s128_view(&self, q: &CudaSlice<f32>, k: &CudaSlice<f32>, v: &CudaSlice<f32>,
                              g: &CudaSlice<f32>, beta: &CudaSlice<f32>,
                              state_in: &cudarc::driver::CudaView<f32>,
                              state_out: &mut cudarc::driver::CudaViewMut<f32>,
                              o: &mut CudaSlice<f32>, n_head: usize, t: usize, scale: f32)
                              -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("gdn_scan_s128");
        const S_V: u32 = 128; const WARP: u32 = 32; const COLS: u32 = 4;
        let cfg = LaunchConfig { grid_dim: (n_head as u32, 1, S_V / COLS), block_dim: (WARP, COLS, 1), shared_mem_bytes: 0 };
        let (h, ti) = (n_head as i32, t as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(q).arg(k).arg(v).arg(g).arg(beta).arg(state_in).arg(state_out).arg(o).arg(&h).arg(&ti).arg(&scale);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// conv1d where the input is a CudaView (resident conv state assembled in place).
    pub fn ssm_conv1d_view(&self, x: &cudarc::driver::CudaView<f32>, w: &CudaSlice<f32>, y: &mut CudaSlice<f32>,
                           conv_dim: usize, t: usize, d_conv: usize, silu: bool)
                           -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("ssm_conv1d_silu_f32");
        let cfg = LaunchConfig::for_num_elems(conv_dim as u32);
        let (cd, ti, dc, s) = (conv_dim as i32, t as i32, d_conv as i32, silu as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(x).arg(w).arg(y).arg(&cd).arg(&ti).arg(&dc).arg(&s);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// Depthwise causal conv1d + optional SiLU.
    /// x:[conv_dim, T+d_conv-1] channel-major (first d_conv-1 cols = carried state),
    /// w:[d_conv, conv_dim] kernel-major, y:[conv_dim, T] channel-major.
    pub fn ssm_conv1d(&self, x: &CudaSlice<f32>, w: &CudaSlice<f32>, y: &mut CudaSlice<f32>,
                      conv_dim: usize, t: usize, d_conv: usize, silu: bool)
                      -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("ssm_conv1d_silu_f32");
        let cfg = LaunchConfig::for_num_elems(conv_dim as u32);
        let (cd, ti, dc, s) = (conv_dim as i32, t as i32, d_conv as i32, silu as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(x).arg(w).arg(y).arg(&cd).arg(&ti).arg(&dc).arg(&s);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// Gated DeltaNet scan, S_v=128. q,k,v:[128,H,T]; g,beta:[H,T]; state:[128,128,H] transposed;
    /// o:[128,H,T]. Single sequence.
    pub fn gdn_scan_s128(&self, q: &CudaSlice<f32>, k: &CudaSlice<f32>, v: &CudaSlice<f32>,
                         g: &CudaSlice<f32>, beta: &CudaSlice<f32>, state_in: &CudaSlice<f32>,
                         state_out: &mut CudaSlice<f32>, o: &mut CudaSlice<f32>,
                         n_head: usize, t: usize, scale: f32)
                         -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("gdn_scan_s128");
        const S_V: u32 = 128; const WARP: u32 = 32; const COLS_PER_BLOCK: u32 = 4;
        let cfg = LaunchConfig {
            grid_dim: (n_head as u32, 1, S_V / COLS_PER_BLOCK),
            block_dim: (WARP, COLS_PER_BLOCK, 1),
            shared_mem_bytes: 0,
        };
        let (h, ti) = (n_head as i32, t as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(q).arg(k).arg(v).arg(g).arg(beta).arg(state_in).arg(state_out).arg(o).arg(&h).arg(&ti).arg(&scale);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// softplus-based g_log: g_log[h,t] = a[h] * softplus(alpha[h,t] + dt_bias[h]). a pre-negated.
    pub fn gdn_glog(&self, alpha: &CudaSlice<f32>, dt_bias: &CudaSlice<f32>, a: &CudaSlice<f32>,
                    g_log: &mut CudaSlice<f32>, n_head: usize, t: usize)
                    -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("gdn_glog_f32");
        let cfg = LaunchConfig::for_num_elems((n_head * t) as u32);
        let (h, ti) = (n_head as i32, t as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(alpha).arg(dt_bias).arg(a).arg(g_log).arg(&h).arg(&ti);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    pub fn sigmoid(&self, x: &CudaSlice<f32>, y: &mut CudaSlice<f32>, n: usize)
                   -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("sigmoid_f32");
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let ni = n as i32;
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(x).arg(y).arg(&ni);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// gated RMSNorm: dst = RMSNorm(o, w[ncols]) * silu(z), per row of ncols. nrows blocks.
    pub fn gated_rmsnorm(&self, o: &CudaSlice<f32>, w: &CudaSlice<f32>, z: &CudaSlice<f32>,
                         dst: &mut CudaSlice<f32>, ncols: usize, nrows: usize, eps: f32)
                         -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("gated_rmsnorm_f32");
        let cfg = LaunchConfig { grid_dim: (nrows as u32, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
        let (nc, e) = (ncols as i32, eps);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(o).arg(w).arg(z).arg(dst).arg(&nc).arg(&e);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// transpose [rows,cols] row-major -> [cols,rows] row-major.
    pub fn transpose(&self, inp: &CudaSlice<f32>, rows: usize, cols: usize)
                     -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let f = self.func("transpose_f32");
        let mut out = self.zeros(rows * cols)?;
        let cfg = LaunchConfig::for_num_elems((rows * cols) as u32);
        let (r, c) = (rows as i32, cols as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(inp).arg(&mut out).arg(&r).arg(&c);
        unsafe { b.launch(cfg)?; }
        Ok(out)
    }

    /// repeat-interleave heads: in[head_dim,n_in,T] -> out[head_dim,n_out,T].
    pub fn repeat_heads(&self, inp: &CudaSlice<f32>, out: &mut CudaSlice<f32>,
                        head_dim: usize, n_in: usize, n_out: usize, t: usize)
                        -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("repeat_heads_f32");
        let cfg = LaunchConfig::for_num_elems((head_dim * n_out * t) as u32);
        let (hd, ni, no, ti) = (head_dim as i32, n_in as i32, n_out as i32, t as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(inp).arg(out).arg(&hd).arg(&ni).arg(&no).arg(&ti);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// q|gate split (on-device). qf:[T, n_head*2*head_dim] -> q_out,gate_out:[head_dim,n_head,T].
    /// Replaces the dtoh->host-double-loop->htod in full_attn / full_attn_decode.
    pub fn q_gate_split(&self, qf: &CudaSlice<f32>, q_out: &mut CudaSlice<f32>,
                        gate_out: &mut CudaSlice<f32>, head_dim: usize, n_head: usize, t: usize)
                        -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("q_gate_split_f32");
        let cfg = LaunchConfig::for_num_elems((head_dim * n_head * t) as u32);
        let (hd, nh, ti) = (head_dim as i32, n_head as i32, t as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(qf).arg(q_out).arg(gate_out).arg(&hd).arg(&nh).arg(&ti);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// qkv->GDN repack (on-device). conv_out:[conv_dim,T] channel-major ->
    /// q_g/k_g/v_g:[d_state,num_v,T] with q/k head-repeat kh = vh % num_k (validated modulo mapping).
    /// Replaces the dtoh->host-q/k/v-repack->3x-htod in linear_attn / linear_attn_decode.
    pub fn qkv_to_gdn_repack(&self, conv_out: &CudaSlice<f32>, q_g: &mut CudaSlice<f32>,
                             k_g: &mut CudaSlice<f32>, v_g: &mut CudaSlice<f32>,
                             d_state: usize, num_v: usize, num_k: usize, key_dim: usize, t: usize)
                             -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("qkv_to_gdn_repack_f32");
        let cfg = LaunchConfig::for_num_elems((d_state * num_v * t) as u32);
        let (ds, nv, nk, kd, ti) = (d_state as i32, num_v as i32, num_k as i32, key_dim as i32, t as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(conv_out).arg(q_g).arg(k_g).arg(v_g).arg(&ds).arg(&nv).arg(&nk).arg(&kd).arg(&ti);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// conv left zero-pad (prefill from zero state). src:[conv_dim,T] -> dst:[conv_dim,T+pad],
    /// cols 0..pad = 0, cols pad..pad+T = src. `dst` MUST be pre-zeroed. No dtoh/host-loop/htod.
    pub fn conv_left_pad(&self, src: &CudaSlice<f32>, dst: &mut CudaSlice<f32>,
                         conv_dim: usize, t: usize, pad: usize)
                         -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("conv_left_pad_f32");
        let cfg = LaunchConfig::for_num_elems((conv_dim * t) as u32);
        let (cd, ti, p) = (conv_dim as i32, t as i32, pad as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(src).arg(dst).arg(&cd).arg(&ti).arg(&p);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// conv-state assemble + ring roll (decode T=1). conv_state:[conv_dim,pad] (resident),
    /// qkv_col:[conv_dim] -> conv_in:[conv_dim,pad+1]; AND rolls conv_state (keep last pad cols).
    /// Replaces the dtoh->host-conv-ring-assemble->ring-update->htod in linear_attn_decode.
    pub fn conv_assemble_and_roll(&self, qkv_col: &CudaSlice<f32>, conv_state: &mut CudaSlice<f32>,
                                  conv_in: &mut CudaSlice<f32>, conv_dim: usize, pad: usize)
                                  -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("conv_assemble_and_roll_f32");
        let cfg = LaunchConfig::for_num_elems(conv_dim as u32);
        let (cd, p) = (conv_dim as i32, pad as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(qkv_col).arg(conv_state).arg(conv_in).arg(&cd).arg(&p);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// Copy a contiguous range [start, start+len) out of src into a fresh slice (device→device via host).
    /// Used for qkv split views. Small/rare; not perf-critical in Stage 1.
    pub fn slice_range(&self, src: &CudaSlice<f32>, start: usize, len: usize)
                       -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let host = self.gpu.stream.clone_dtoh(src)?;
        self.gpu.stream.synchronize()?;
        Ok(self.htod(&host[start..start + len])?)
    }
}
