//! PrimeGraph (task #14, design v3): a per-bucket CUDA graph of the FULL fresh-prime
//! trunk, bound to a dedicated SCRATCH cache; serving replays it (one cuGraphLaunch,
//! ~23ms vs ~26ms eager at bucket 512) and COPIES the outputs into the session's cache
//! (KV rows + conv rings + recurrent states, ~tens of us D2D — the copy-out beats both
//! table-indirect kernels and graphExec node patching, ledger design v3).
//!
//! Correctness story (all bit-proven by prime-graph-smoke + the gate):
//! - fresh-prime semantics are BAKED as graph-head memset nodes (state/ring/len_d zero);
//! - pads past the true length are invisible (gdn_pad_mask identity steps, causal
//!   attention, device-indexed last-row gathers) — replay logits are bit-identical to
//!   the eager true-length prime;
//! - the GRAPH-OUTPUT CONTRACT: only the stable IO buffers and the scratch cache's
//!   resident state survive a launch (in-graph transient addresses recycle).
//! - ssm ping-pong: the capture-time core swapped the scratch cache's host fields; after
//!   capture they name exactly the buffer the graph WRITES, and no further swaps happen,
//!   so `scratch.recur[il].ssm_state` is the copy-out source on every replay.

use crate::cache::Cache;
use crate::hybrid::HybridModel;
use crate::Engine;
use cudarc::driver::{CudaGraph, CudaSlice};

pub struct PrimeGraph {
    pub bucket: usize,
    graph: CudaGraph,
    /// CAPTURE-RETAIN keeper (draft-graph law): holds every allocation the closure made so
    /// the pool NEVER re-issues the graph's baked addresses to later eager work — without
    /// it, any post-capture allocation can land on graph-internal addresses and every
    /// replay scribbles it (the prime-graph-gate T=512 corruption, 2026-07-26).
    _keeper: Vec<Box<dyn std::any::Any + Send>>,
    scratch: Cache,
    x_in: CudaSlice<f32>,
    len_d: CudaSlice<i32>,
    logits_out: CudaSlice<f32>,
    h_seed_out: CudaSlice<f32>,
    n_embd: usize,
}

impl PrimeGraph {
    /// Gate/debug accessor: the graph's bound scratch cache (read-only).
    pub fn scratch(&self) -> &Cache {
        &self.scratch
    }
}

impl HybridModel {
    /// Capture the fresh-prime graph for `bucket` tokens (13-15ms measured). Manual staged
    /// capture — capture_graph_retained's keeper path trips on the prime (smoke finding 4).
    pub fn prime_graph_new(&self, e: &Engine, bucket: usize)
                           -> Result<PrimeGraph, Box<dyn std::error::Error>> {
        use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
        let n_embd = self.cfg.n_embd as usize;
        let n_vocab = self.output.out_features();
        let mut scratch = Cache::new(e, &self.cfg, bucket + 8)?;
        let x_in = e.zeros(bucket * n_embd)?;
        let pos_d = e.htod_i32(&(0..bucket as i32).collect::<Vec<_>>())?;
        let len_d = e.htod_i32(&[bucket as i32])?;
        let mut logits_out = e.uninit(n_vocab)?;
        let mut h_seed_out = e.uninit(n_embd)?;

        let scratch_cell = std::cell::RefCell::new(&mut scratch);
        let lo_cell = std::cell::RefCell::new(&mut logits_out);
        let hs_cell = std::cell::RefCell::new(&mut h_seed_out);
        let (graph, keeper) = e.capture_graph_retained(|e| {
            let sc: &mut Cache = &mut scratch_cell.borrow_mut();
            for kvl in sc.kv.iter_mut().flatten() {
                kvl.len = 0;
                e.stream().memset_zeros(&mut kvl.len_d)?;
            }
            for rl in sc.recur.iter_mut().flatten() {
                e.stream().memset_zeros(&mut rl.conv_state)?;
                e.stream().memset_zeros(&mut rl.ssm_state)?;
                e.stream().memset_zeros(&mut rl.ssm_state_alt)?;
            }
            self.prime_chunk_captured(e, &x_in, &pos_d, bucket, sc, &len_d,
                                      &mut lo_cell.borrow_mut(), &mut hs_cell.borrow_mut())
        })?;
        drop(scratch_cell);
        drop(lo_cell);
        drop(hs_cell);
        let _ = CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED;
        let _ = CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
        Ok(PrimeGraph {
            bucket,
            graph,
            _keeper: keeper,
            scratch,
            x_in,
            len_d,
            logits_out,
            h_seed_out,
            n_embd,
        })
    }

    /// Replay the graph for `tokens` (len <= bucket) and copy the outputs into `session`
    /// (a FRESH cache: pos == 0). Returns host logits (the prefill_tick contract).
    pub fn prime_graph_run(&self, e: &Engine, pg: &mut PrimeGraph, tokens: &[u32],
                           session: &mut Cache)
                           -> Result<(Vec<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let t = tokens.len();
        assert!(t >= 2 && t <= pg.bucket, "prime_graph_run: 2 <= T <= bucket");
        assert!(session.pos == 0, "prime_graph_run: fresh sessions only");
        let n_embd = pg.n_embd;
        // graph inputs: embed rows + zeroed pad tail + true length (all OUTSIDE capture,
        // so host-sourced writes are legal here)
        let x = self.embed(e, tokens)?;
        e.copy_into(&mut pg.x_in, 0, &x, t * n_embd)?;
        if t < pg.bucket {
            let mut tail = pg.x_in.slice_mut(t * n_embd..pg.bucket * n_embd);
            e.stream().memset_zeros(&mut tail)?;
        }
        e.set_i32_one(&mut pg.len_d, t as i32)?;
        pg.graph.launch()?;
        // copy-out: quantized KV rows [0,T), conv rings, recurrent state
        for (il, kvl) in pg.scratch.kv.iter().enumerate() {
            let (Some(src), Some(dst)) = (kvl.as_ref(), session.kv[il].as_mut()) else { continue };
            let kb = t * src.k_tok_bytes;
            let vb = t * src.v_tok_bytes;
            e.stream().memcpy_dtod(&src.k.slice(0..kb), &mut dst.k.slice_mut(0..kb))?;
            e.stream().memcpy_dtod(&src.v.slice(0..vb), &mut dst.v.slice_mut(0..vb))?;
            dst.len = t;
            e.set_i32_one(&mut dst.len_d, t as i32)?;
        }
        for (il, rl) in pg.scratch.recur.iter().enumerate() {
            let (Some(src), Some(dst)) = (rl.as_ref(), session.recur[il].as_mut()) else { continue };
            let cn = src.conv_state.len();
            let sn = src.ssm_state.len();
            e.copy_into(&mut dst.conv_state, 0, &src.conv_state, cn)?;
            e.copy_into(&mut dst.ssm_state, 0, &src.ssm_state, sn)?;
        }
        session.pos = t;
        let logits = e.dtoh(&pg.logits_out)?;
        let mut h_seed = e.uninit(n_embd)?;
        let hn = pg.h_seed_out.len();
        e.copy_into(&mut h_seed, 0, &pg.h_seed_out, hn)?;
        Ok((logits, h_seed))
    }
}
