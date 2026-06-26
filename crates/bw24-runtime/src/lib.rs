//! bw24 inference runtime. Correctness-first: every GPU op is validated against a
//! CPU reference before any sm_120 fast-path replaces it.

use std::sync::Arc;
use cudarc::driver::{CudaContext, CudaStream};
use cudarc::cublaslt::{CudaBlasLT, Matmul, MatmulConfig};

pub use bw24_gguf;

/// CPU reference matmul for a linear layer y = x @ W^T.
/// Conventions (ggml/GGUF): a weight tensor with ne=[in, out] is stored row-major as
/// `out` rows of `in` contiguous elements — i.e. W[o*in + i]. A linear layer computes
/// y[o] = sum_i x[i] * W[o*in + i], for each of `out` outputs. Batched over `m` tokens:
///   x: [m, in] row-major (x[t*in + i]); w: [out, in] row-major (w[o*in + i]); y: [m, out].
pub fn cpu_linear(x: &[f32], w: &[f32], m: usize, in_f: usize, out_f: usize) -> Vec<f32> {
    assert_eq!(x.len(), m * in_f);
    assert_eq!(w.len(), out_f * in_f);
    let mut y = vec![0f32; m * out_f];
    for t in 0..m {
        for o in 0..out_f {
            let mut acc = 0f32;
            let xr = &x[t * in_f..t * in_f + in_f];
            let wr = &w[o * in_f..o * in_f + in_f];
            for i in 0..in_f {
                acc += xr[i] * wr[i];
            }
            y[t * out_f + o] = acc;
        }
    }
    y
}

/// GPU runtime handle: a context + stream + cuBLASLt.
pub struct Gpu {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub blas: CudaBlasLT,
}

impl Gpu {
    pub fn new(ordinal: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let ctx = CudaContext::new(ordinal)?;
        let stream = ctx.default_stream();
        let blas = CudaBlasLT::new(stream.clone())?;
        Ok(Self { ctx, stream, blas })
    }

    /// GPU linear y = x @ W^T using cuBLASLt (f32), matching `cpu_linear` exactly.
    ///
    /// Layout reasoning (cuBLASLt is column-major):
    /// We want y[m,out] row-major = y^T[out,m] column-major. Treat:
    ///   - x[m,in] row-major == x^T[in,m] col-major  (an in×m col-major matrix)
    ///   - w[out,in] row-major == w^T[in,out] col-major (an in×out col-major matrix)
    /// Compute C[out,m] col-major = W_colmajor(out×in) * X_colmajor(in×m)
    ///   => set A = w (interpreted col-major as in×out, so transa to get out×in),
    ///      B = x (col-major in×m), C = y (col-major out×m == y[m,out] row-major).
    /// cfg: m_=out, n_=m_tokens, k=in. A is in×out (lda=in, transa=true -> out×in),
    ///      B is in×m (ldb=in), C is out×m (ldc=out).
    pub fn linear_f32(
        &self,
        x: &cudarc::driver::CudaSlice<f32>,
        w: &cudarc::driver::CudaSlice<f32>,
        m_tokens: usize,
        in_f: usize,
        out_f: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let mut c = self.stream.alloc_zeros::<f32>(m_tokens * out_f)?;
        let cfg = MatmulConfig {
            transa: true,            // A stored in×out col-major -> use as out×in
            transb: false,
            transc: false,
            m: out_f as u64,
            n: m_tokens as u64,
            k: in_f as u64,
            alpha: 1.0,
            lda: in_f as i64,        // A leading dim = in (col-major in×out)
            ldb: in_f as i64,        // B leading dim = in (col-major in×m)
            beta: 0.0,
            ldc: out_f as i64,       // C leading dim = out (col-major out×m)
            stride_a: None, stride_b: None, stride_c: None, stride_bias: None,
            batch_size: None,
        };
        unsafe { self.blas.matmul(cfg, w, x, &mut c, None, None)?; }
        let y = self.stream.clone_dtoh(&c)?;
        self.stream.synchronize()?;
        Ok(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_linear_tiny() {
        // m=1, in=2, out=2; x=[1,2], W=[[1,0],[0,1]] (identity) -> y=[1,2]
        let x = vec![1.0, 2.0];
        let w = vec![1.0, 0.0, 0.0, 1.0]; // row0=[1,0], row1=[0,1]
        let y = cpu_linear(&x, &w, 1, 2, 2);
        assert_eq!(y, vec![1.0, 2.0]);
        // W=[[1,1],[2,0]] -> y[0]=1*1+2*1=3, y[1]=1*2+2*0=2
        let w2 = vec![1.0, 1.0, 2.0, 0.0];
        let y2 = cpu_linear(&x, &w2, 1, 2, 2);
        assert_eq!(y2, vec![3.0, 2.0]);
    }
}
