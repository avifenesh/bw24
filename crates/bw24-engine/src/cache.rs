//! Dual cache for the hybrid arch (PHASE1-HYBRID.md §3), GPU-RESIDENT (Stage-2):
//! - Growing KV cache for full-attention layers (kept on GPU, appended in place).
//! - Fixed recurrent state (conv ring + SSM state) for linear-attention layers (kept on GPU).
//! No host round-trips per step. Single sequence.

use cudarc::driver::CudaSlice;
use bw24_gguf::config::{ModelConfig, LayerKind};
use crate::Engine;

/// Per-full-attn-layer growing KV cache, resident on GPU. QUANTIZED (KVQUANT-PLAN §B):
/// K stored q8_0 (34 B/32 elem), V stored q5_1 (24 B/32 elem). Per-token byte layout keeps the
/// [token, kv_head, dim] element order so a 32-block never straddles a head (assert head_dim%32==0).
/// Element-within-token index = kv_head*head_dim + d; block = idx/32; lane = idx%32.
pub struct KvLayer {
    pub k: CudaSlice<u8>,        // q8_0 packed, capacity max_ctx*k_tok_bytes
    pub v: CudaSlice<u8>,        // q5_1 packed, capacity max_ctx*v_tok_bytes
    pub kv_dim_k: usize,         // head_dim_k * n_head_kv  (K elements per token)
    pub kv_dim_v: usize,         // head_dim_v * n_head_kv  (V elements per token)
    pub k_tok_bytes: usize,      // (kv_dim_k/32)*34
    pub v_tok_bytes: usize,      // (kv_dim_v/32)*24
    pub len: usize,
    /// Device-resident mirror of `len` (CUDA-GRAPH-PLAN Phase 2). Holds the KV write SLOT for the
    /// append-dc kernel (old len, before this step's append); after `inc_seqlen` it holds the new
    /// len == t_kv for fa_decode_dc. Kept in lock-step with the host `len`. i32[1].
    pub len_d: CudaSlice<i32>,
}

/// Per-linear-attn-layer fixed recurrent state.
/// conv_state and ssm_state are BOTH kept RESIDENT on GPU — the conv ring assemble + roll runs
/// on-device (conv_assemble_and_roll), so there is no per-step dtoh/htod for either.
pub struct RecurLayer {
    pub conv_state: CudaSlice<f32>,  // GPU [conv_dim, d_conv-1] (channel c, tap j at c*pad + j)
    pub ssm_state: CudaSlice<f32>,   // GPU [d_state, d_state, num_v] transposed M[col][i]
    /// PERSISTENT second SSM-state buffer for the gdn-scan double buffer (DECODE DETERMINISM FIX).
    /// gdn_scan needs DISTINCT in/out state buffers. The old eager path allocated a fresh
    /// `state_scratch` via `e.uninit` every step and swapped its pointer into `ssm_state`; that
    /// per-step alloc/free churned the stream-ordered async pool, and the freed prior `ssm_state`
    /// block was recycled by the next step's scratch while a kernel referencing the swapped-in state
    /// was still in flight — a use-after-reuse that produced RUN-TO-RUN nondeterministic decode
    /// (two identical prompt primes diverged). We instead PING-PONG between two STABLE resident
    /// buffers (no per-step alloc/free, no pool churn): step writes into the spare, then swaps the
    /// two owned buffers in place. Stable pointers, identical math. Sized like `ssm_state`.
    pub ssm_state_alt: CudaSlice<f32>,
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
        let n_head_kv = cfg.n_head_kv as usize;
        let head_dim_k = cfg.head_dim_k as usize;
        let head_dim_v = cfg.head_dim_v as usize;
        assert!(head_dim_k % 32 == 0 && head_dim_v % 32 == 0,
                "KVQUANT requires head_dim_k%32==0 && head_dim_v%32==0 (got k={head_dim_k} v={head_dim_v})");
        let kv_dim_k = head_dim_k * n_head_kv;
        let kv_dim_v = head_dim_v * n_head_kv;
        // per-block bytes follow the env-selected KV formats (kvbytes lane; default q8_0/q5_1
        // = 34/24 — MUST match the flash fatbin Engine::new loaded, both read the same env).
        let (kbb, vbb) = crate::kv_blk_bytes();
        let k_tok_bytes = (kv_dim_k / 32) * kbb;
        let v_tok_bytes = (kv_dim_v / 32) * vbb;
        let (conv_dim, d_state, num_v, d_conv) = if let Some(s) = &cfg.ssm {
            let num_k = s.group_count as usize;
            let num_v = s.time_step_rank as usize;
            let ds = s.state_size as usize;
            (ds * num_k * 2 + ds * num_v, ds, num_v, s.conv_kernel as usize)
        } else { (0, 0, 0, 0) };
        for il in 0..cfg.n_layer {
            // gemma4 R5: per-layer KV geometry (SWA 256hd x 8kv = 2048 / global 512hd x 2kv = 1024).
            let (kv_dim_k, kv_dim_v) = match &cfg.gemma4 {
                Some(g) => {
                    let hd = if g.swa_pattern[il as usize] { g.key_length_swa } else { g.key_length_global } as usize;
                    // E4B ships a SCALAR head_count_kv (per-layer vec empty; scalar = 2 in
                    // the gguf, landing in cfg.n_head_kv): kv_dim = hd * 2 for BOTH kinds —
                    // swa 2x256 = 512, global 2x512 = 1024. The old fallback used
                    // key_length_global (512) for both, which HALVED the global layers' K/V
                    // (the attn writes wk.out_features = 1024 rows): every E4B global layer
                    // stored/attended half its K/V and the batched append read row strides
                    // wrong — THE cross-mode maxdiff-30 root (2026-07-12 bisect, il=5 slot-1
                    // byte forensics). 26B/31B keep the per-layer vec.
                    let d = match g.head_count_kv.get(il as usize) {
                        Some(n) => hd * *n as usize,
                        None => hd * n_head_kv,
                    };
                    (d, d)
                }
                None => (kv_dim_k, kv_dim_v),
            };
            // E4B KV-SHARING: the trailing shared_kv_layers have no k/v of their own — they
            // attend an earlier layer's cache (hybrid_forward resolves the target). No KvLayer
            // here: any accidental use is a loud unwrap at bring-up, and rewind/len loops
            // (iter_mut().flatten()) skip None naturally.
            let g4_shared = cfg.gemma4.as_ref().map(|g| g.shared_kv_layers).unwrap_or(0);
            if g4_shared > 0 && il >= cfg.n_layer - g4_shared {
                kv.push(None);
                recur.push(None);
                continue;
            }
            // FP8-GLOBALS (gemma, 2026-07-11): global (hd512) layers hold e4m3 K/V (32B/32elem
            // both planes — the dequant-latency arc); windowed layers keep the default pair.
            let g4_global_fp8 = crate::Engine::gkv_on()
                && cfg.gemma4.as_ref().is_some_and(|g| !g.swa_pattern[il as usize]);
            let g4_windowed_fp8 = crate::Engine::wkv_on()
                && cfg.gemma4.as_ref().is_some_and(|g| g.swa_pattern[il as usize]);
            // QWEN FP8-KV (BW24_KV_FP8, bring-up): non-gemma full-attn layers, uniform class.
            let qwen_fp8 = crate::Engine::kv_fp8_on() && cfg.gemma4.is_none();
            let (kbb_l, vbb_l) = if g4_global_fp8 || g4_windowed_fp8 || qwen_fp8 { (32, 32) }
                                 else { (kbb, vbb) };
            let k_tok_bytes = (kv_dim_k / 32) * kbb_l;
            let v_tok_bytes = (kv_dim_v / 32) * vbb_l;
            match cfg.layer_kind(il) {
                LayerKind::FullAttention => {
                    kv.push(Some(KvLayer {
                        // +8B tail pad: the v4 stage's aligned funnelshift window reads up to
                        // 4B past the final block (PR #3's finding, adopted pad-style — the
                        // expert-dot precedent; zero hot-loop branches, values discarded).
                        k: e.alloc_u8(max_ctx * k_tok_bytes + 8)?,
                        v: e.alloc_u8(max_ctx * v_tok_bytes + 8)?,
                        kv_dim_k, kv_dim_v, k_tok_bytes, v_tok_bytes, len: 0,
                        len_d: e.htod_i32(&[0])?,
                    }));
                    recur.push(None);
                }
                LayerKind::LinearAttention => {
                    kv.push(None);
                    recur.push(Some(RecurLayer {
                        conv_state: e.zeros(conv_dim * (d_conv - 1))?,
                        ssm_state: e.zeros(d_state * d_state * num_v)?,
                        ssm_state_alt: e.zeros(d_state * d_state * num_v)?,
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

    /// PERSISTENT-BUFFER snapshot (spec-decode hot loop): refresh `snap` IN PLACE — same values as
    /// `snapshot()` but the conv/ssm device buffers are reused across rounds (D2D copy-into, ZERO
    /// allocations vs 2 fresh clones per linear layer per round). `snap` must come from a prior
    /// `snapshot()` of THIS cache (same layer shapes).
    pub fn snapshot_into(&self, e: &Engine, snap: &mut CacheSnapshot)
                         -> Result<(), Box<dyn std::error::Error>> {
        let n = self.kv.len();
        for il in 0..n {
            snap.kv_len[il] = self.kv[il].as_ref().map(|kvl| kvl.len);
            if let Some(rl) = &self.recur[il] {
                let dc = snap.conv[il].as_mut().expect("snapshot_into: shape mismatch (conv)");
                let ds = snap.ssm[il].as_mut().expect("snapshot_into: shape mismatch (ssm)");
                let (cn, sn) = (rl.conv_state.len(), rl.ssm_state.len());
                e.copy_into(dc, 0, &rl.conv_state, cn)?;
                e.copy_into(ds, 0, &rl.ssm_state, sn)?;
            }
        }
        snap.pos = self.pos;
        Ok(())
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
                // keep the device mirror in lock-step (CUDA-GRAPH-PLAN Phase 2). Set IN PLACE
                // (stable pointer): a fresh htod_i32 would reallocate len_d, but its old pointer is
                // baked into the captured decode graph's append/inc/fa_decode kernels — replacing it
                // strands the graph on a freed buffer (stale-pointer hazard). memcpy_htod in place.
                e.set_i32_one(&mut kvl.len_d, kvl.len as i32)?;
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
