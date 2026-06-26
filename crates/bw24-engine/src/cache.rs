//! Dual cache for the hybrid arch (PHASE1-HYBRID.md §3):
//! - Growing KV cache for full-attention layers (only ~25% of layers grow with context).
//! - Fixed recurrent state (conv ring buffer + SSM state) for linear-attention layers.
//! Single sequence. Stage-1: KV stored f32 on host, re-uploaded per step (correctness-first;
//! Stage-2 keeps KV resident on GPU + fp16/quant). State stays on host between steps, tiny.

use bw24_gguf::config::{ModelConfig, LayerKind};

/// Per-full-attn-layer growing KV cache (host f32, [head_dim*n_head_kv] per token).
pub struct KvLayer {
    pub k: Vec<f32>,   // appended: token t at t*kv_dim
    pub v: Vec<f32>,
    pub kv_dim: usize, // head_dim * n_head_kv
    pub len: usize,    // tokens stored
}

/// Per-linear-attn-layer fixed recurrent state.
pub struct RecurLayer {
    pub conv_state: Vec<f32>,  // [conv_dim, d_conv-1] channel-major: last (d_conv-1) conv inputs
    pub ssm_state: Vec<f32>,   // [d_state, d_state, num_v] transposed M[col][i], = gdn state
}

pub struct Cache {
    pub kv: Vec<Option<KvLayer>>,
    pub recur: Vec<Option<RecurLayer>>,
    pub pos: usize,   // current sequence length (absolute position of next token)
}

impl Cache {
    /// Allocate empty caches sized by arch. conv_dim/d_state/num_v from SSM config; kv_dim from attn.
    pub fn new(cfg: &ModelConfig) -> Self {
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
                    kv.push(Some(KvLayer { k: Vec::new(), v: Vec::new(), kv_dim, len: 0 }));
                    recur.push(None);
                }
                LayerKind::LinearAttention => {
                    kv.push(None);
                    recur.push(Some(RecurLayer {
                        conv_state: vec![0.0; conv_dim * (d_conv - 1)],
                        ssm_state: vec![0.0; d_state * d_state * num_v],
                    }));
                }
            }
        }
        Cache { kv, recur, pos: 0 }
    }
}
