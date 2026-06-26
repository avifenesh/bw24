//! bw24 engine: Stage-1 correctness-first forward-pass kernels + ops, on sm_120 via cudarc.

use std::sync::Arc;
use cudarc::driver::{CudaContext, CudaStream, CudaModule, CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

pub use bw24_gguf;
pub use bw24_runtime;

pub mod model;
pub mod forward;

const FATBIN_PATH: &str = env!("BW24_ENGINE_FATBIN");

/// Engine device context: CUDA context, stream, loaded kernel module, cuBLASLt (via runtime::Gpu).
pub struct Engine {
    pub gpu: bw24_runtime::Gpu,
    module: Arc<CudaModule>,
}

impl Engine {
    pub fn new(ordinal: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let gpu = bw24_runtime::Gpu::new(ordinal)?;
        let module = gpu.ctx.load_module(Ptx::from_file(FATBIN_PATH))?;
        Ok(Self { gpu, module })
    }

    pub fn ctx(&self) -> &Arc<CudaContext> { &self.gpu.ctx }
    pub fn stream(&self) -> &Arc<CudaStream> { &self.gpu.stream }
    fn func(&self, name: &str) -> CudaFunction {
        self.module.load_function(name).unwrap_or_else(|_| panic!("kernel {name} not in fatbin"))
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
}
