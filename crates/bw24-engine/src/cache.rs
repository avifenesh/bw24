//! Dual cache for the hybrid arch (PHASE1-HYBRID.md §3), GPU-RESIDENT (Stage-2):
//! - Growing KV cache for full-attention layers (kept on GPU, appended in place).
//! - Fixed recurrent state (conv ring + SSM state) for linear-attention layers (kept on GPU).
//! No host round-trips per step. Single sequence.

use cudarc::driver::CudaSlice;
use bw24_gguf::config::{ModelConfig, LayerKind};
use crate::Engine;

/// Per-full-attn-layer growing KV cache, resident on GPU (f32 Stage-2.0; fp16 later).
/// Stored as a flat device buffer with capacity; k/v interleaved per layer as [token, kv_dim].
pub struct KvLayer {
    pub k: CudaSlice<f32>,   // capacity max_ctx*kv_dim
    pub v: CudaSlice<f32>,
    pub kv_dim: usize,
    pub len: usize,
}

/// Per-linear-attn-layer fixed recurrent state.
/// conv_state is small (~0.1MB) → kept on host (negligible transfer). ssm_state is large
/// (~2.1MB) → kept RESIDENT on GPU (was the dominant per-step round-trip, ~100MB/token).
pub struct RecurLayer {
    pub conv_state: Vec<f32>,        // host [conv_dim, d_conv-1]
    pub ssm_state: CudaSlice<f32>,   // GPU [d_state, d_state, num_v] transposed M[col][i]
}

pub struct Cache {
    pub kv: Vec<Option<KvLayer>>,
    pub recur: Vec<Option<RecurLayer>>,
    pub pos: usize,
    pub max_ctx: usize,
}

impl Cache {
    /// Allocate GPU-resident caches sized by arch + max context.
    pub fn new(e: &Engine, cfg: &ModelConfig, max_ctx: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let n = cfg.n_layer as usize;
        let mut kv = Vec::with_capacity(n);
        let mut recur = Vec::with_capacity(n);
        let kv_dim = cfg.head_dim_k as usize * cfg.n_head_kv as usize;
        let (conv_dim, d_state, num_v, d_conv) = if let Some(s) = &cfg.ssm {
            let num_k = s.group_count as usize;
            let num_v = s.time_step_rank as usize;
            let ds = s.state_size as usize;
            (ds * num_k * 2 + ds * num_v, ds, num_v, s.conv_kernel as usize)
        } else { (0, 0, 0, 0) };
        for il in 0..cfg.n_layer {
            match cfg.layer_kind(il) {
                LayerKind::FullAttention => {
                    kv.push(Some(KvLayer {
                        k: e.zeros(max_ctx * kv_dim)?,
                        v: e.zeros(max_ctx * kv_dim)?,
                        kv_dim, len: 0,
                    }));
                    recur.push(None);
                }
                LayerKind::LinearAttention => {
                    kv.push(None);
                    recur.push(Some(RecurLayer {
                        conv_state: vec![0.0; conv_dim * (d_conv - 1)],
                        ssm_state: e.zeros(d_state * d_state * num_v)?,
                    }));
                }
            }
        }
        Ok(Cache { kv, recur, pos: 0, max_ctx })
    }
}
