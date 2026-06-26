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

const FATBIN_PATH: &str = env!("BW24_ENGINE_FATBIN");
const HYBRID_FATBIN_PATH: &str = env!("BW24_HYBRID_FATBIN");
const QMATVEC_FATBIN_PATH: &str = env!("BW24_QMATVEC_FATBIN");

/// Quant type codes matching qmatvec.cu QType enum.
pub const QT_Q8_0: i32 = 0;
pub const QT_Q4_K: i32 = 1;
pub const QT_Q6_K: i32 = 2;

/// Engine device context: CUDA context, stream, loaded kernel modules, cuBLASLt (via runtime::Gpu).
pub struct Engine {
    pub gpu: bw24_runtime::Gpu,
    module: Arc<CudaModule>,
    hybrid: Arc<CudaModule>,
    qmatvec: Arc<CudaModule>,
}

impl Engine {
    pub fn new(ordinal: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let gpu = bw24_runtime::Gpu::new(ordinal)?;
        let module = gpu.ctx.load_module(Ptx::from_file(FATBIN_PATH))?;
        let hybrid = gpu.ctx.load_module(Ptx::from_file(HYBRID_FATBIN_PATH))?;
        let qmatvec = gpu.ctx.load_module(Ptx::from_file(QMATVEC_FATBIN_PATH))?;
        Ok(Self { gpu, module, hybrid, qmatvec })
    }

    pub fn ctx(&self) -> &Arc<CudaContext> { &self.gpu.ctx }
    pub fn stream(&self) -> &Arc<CudaStream> { &self.gpu.stream }
    fn func(&self, name: &str) -> CudaFunction {
        self.module.load_function(name)
            .or_else(|_| self.hybrid.load_function(name))
            .or_else(|_| self.qmatvec.load_function(name))
            .unwrap_or_else(|_| panic!("kernel {name} not in any fatbin"))
    }

    pub fn htod_bytes(&self, v: &[u8]) -> Result<CudaSlice<u8>, Box<dyn std::error::Error>> {
        Ok(self.gpu.stream.clone_htod(v)?)
    }

    /// Resident-quantized linear: y[m,out] = x[m,in] @ W[out,in]^T, W in GGUF block bytes.
    /// Weights stay packed in VRAM; dequant happens in-register inside the kernel.
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
        match w {
            GpuTensor::Quant { bytes, qtype, row_bytes, .. } =>
                self.qmatvec(bytes, x, m, in_f, out_f, *qtype, *row_bytes),
            GpuTensor::Float { data, .. } => self.linear(x, data, m, in_f, out_f),
        }
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

    /// Copy a contiguous range [start, start+len) out of src into a fresh slice (device→device via host).
    /// Used for qkv split views. Small/rare; not perf-critical in Stage 1.
    pub fn slice_range(&self, src: &CudaSlice<f32>, start: usize, len: usize)
                       -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let host = self.gpu.stream.clone_dtoh(src)?;
        self.gpu.stream.synchronize()?;
        Ok(self.htod(&host[start..start + len])?)
    }
}
