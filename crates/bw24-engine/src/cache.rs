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
/// conv_state and ssm_state are BOTH kept RESIDENT on GPU — the conv ring assemble + roll runs
/// on-device (conv_assemble_and_roll), so there is no per-step dtoh/htod for either.
pub struct RecurLayer {
    pub conv_state: CudaSlice<f32>,  // GPU [conv_dim, d_conv-1] (channel c, tap j at c*pad + j)
    pub ssm_state: CudaSlice<f32>,   // GPU [d_state, d_state, num_v] transposed M[col][i]
}

pub struct Cache {
    pub kv: Vec<Option<KvLayer>>,
    pub recur: Vec<Option<RecurLayer>>,
    pub pos: usize,
    pub max_ctx: usize,
}

/// Snapshot of the dual cache taken BEFORE a spec-decode draft+verify round (MTP-PLAN §C/§D.4).
/// - Full-attn KV: only the per-layer `len` is recorded; rollback truncates (append-only,
///   position-addressed — no copy). C.1.
/// - Linear-attn conv/ssm: real device-to-device COPIES of the recurrent state, because those
///   buffers are mutated IN PLACE by the verify pass and have no position index to truncate. C.2.
///   (CudaSlice::clone is an Arc refcount, NOT a buffer copy — so we alloc fresh + memcpy_dtod.)
pub struct CacheSnapshot {
    pub kv_len: Vec<Option<usize>>,            // per layer (Some for full-attn layers)
    pub conv: Vec<Option<CudaSlice<f32>>>,     // per layer (Some for linear-attn layers, D2D copy)
    pub ssm: Vec<Option<CudaSlice<f32>>>,
    pub pos: usize,
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
                        conv_state: e.zeros(conv_dim * (d_conv - 1))?,
                        ssm_state: e.zeros(d_state * d_state * num_v)?,
                    }));
                }
            }
        }
        Ok(Cache { kv, recur, pos: 0, max_ctx })
    }

    /// Snapshot the dual cache before a spec-decode draft+verify round (MTP-PLAN §C/§D.4).
    /// Records each full-attn `len` (cheap) and makes a REAL device copy of each linear-attn
    /// conv_state/ssm_state (a fresh alloc + memcpy_dtod — NOT an Arc clone).
    pub fn snapshot(&self, e: &Engine) -> Result<CacheSnapshot, Box<dyn std::error::Error>> {
        let n = self.kv.len();
        let mut kv_len = Vec::with_capacity(n);
        let mut conv = Vec::with_capacity(n);
        let mut ssm = Vec::with_capacity(n);
        for il in 0..n {
            match &self.kv[il] {
                Some(kvl) => kv_len.push(Some(kvl.len)),
                None => kv_len.push(None),
            }
            match &self.recur[il] {
                Some(rl) => {
                    conv.push(Some(e.clone_dtod(&rl.conv_state)?));
                    ssm.push(Some(e.clone_dtod(&rl.ssm_state)?));
                }
                None => { conv.push(None); ssm.push(None); }
            }
        }
        Ok(CacheSnapshot { kv_len, conv, ssm, pos: self.pos })
    }

    /// Roll the cache back to exactly `snap.pos + accept_len` committed tokens (MTP-PLAN §C).
    /// - Full-attn KV (C.1): set len = snapshot_len + accept_len (truncate, no copy).
    /// - Linear-attn (C.2): RESTORE the snapshot conv/ssm (real D2D copy back into the resident
    ///   buffers). The caller must then REPLAY the `accept_len` committed tokens through the full
    ///   T=1 decode path to rebuild the recurrent state for those positions. We restore (not
    ///   replay here) because replay needs the model; this only resets state to the pre-round value.
    /// `cache.pos` is set to `snap.pos` so the caller's replay advances it back to the commit point.
    pub fn rollback(&mut self, e: &Engine, snap: &CacheSnapshot, accept_len: usize)
                    -> Result<(), Box<dyn std::error::Error>> {
        for il in 0..self.kv.len() {
            if let (Some(kvl), Some(saved)) = (self.kv[il].as_mut(), snap.kv_len[il]) {
                kvl.len = saved + accept_len;
            }
            if let Some(rl) = self.recur[il].as_mut() {
                if let Some(c) = &snap.conv[il] { e.copy_into(&mut rl.conv_state, 0, c, c.len())?; }
                if let Some(s) = &snap.ssm[il]  { e.copy_into(&mut rl.ssm_state,  0, s, s.len())?; }
            }
        }
        self.pos = snap.pos;
        Ok(())
    }
}
