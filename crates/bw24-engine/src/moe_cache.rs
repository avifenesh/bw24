//! EDGE-1 §B: SLRU GPU expert-residency cache (MOE-SLRU-PLAN §B).
//!
//! Stage-1 `moe_ffn` re-stages EVERY routed expert EVERY token over PCIe into one scratch slot.
//! The same ~15-20% of experts recur (the "hot expert" mass), so an SLRU residency cache makes
//! the steady-state re-stage count -> ~0. The cache holds N fixed-address GPU slots (never
//! re-allocated, never fragmented), a `BlockId -> slot` residency table, an SLRU eviction policy
//! (probation + protected segments) and a second-miss "ghost" admission filter so a one-off cold
//! expert can never evict a genuinely hot one.
//!
//! THE bit-identity property (MOE-SLRU-PLAN §B.3): a cache HIT and a MISS feed `qmatvec_view` the
//! *same* GGUF block bytes — the only difference is whether the `memcpy_htod` ran. So the cache-hit
//! weight path is byte-for-byte identical to stage-every-token; the §D.2 gate pins this.
//!
//! Gated behind `BW24_MOE_CACHE` (default off => current stage-every-token behavior).

use std::collections::{HashMap, HashSet, VecDeque};
use cudarc::driver::CudaSlice;
use crate::Engine;

/// Which projection of an expert (gate/up/down are three distinct GGUF blocks per expert).
pub const PROJ_GATE: u8 = 0;
pub const PROJ_UP: u8 = 1;
pub const PROJ_DOWN: u8 = 2;

/// Residency key: expert `ex` of layer `layer` projection `proj` is a distinct block.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BlockId {
    pub layer: u16,
    pub proj: u8,
    pub ex: u16,
}
impl BlockId {
    #[inline]
    pub fn new(layer: u16, proj: u8, ex: u16) -> Self { BlockId { layer, proj, ex } }
}

/// Where a dispatched block landed: a retained resident slot or a transient staging slot.
#[derive(Clone, Copy, Debug)]
pub enum DispatchSlot {
    Resident(usize),
    Staging(usize),
}

/// SLRU GPU expert-residency cache. N fixed slots, each sized to the largest block so any block
/// fits any slot (fixed-address, never re-allocated). One global instance shared across all layers.
pub struct MoeSlotCache {
    slots: Vec<CudaSlice<u8>>,        // N fixed GPU buffers, each `max_block_bytes`
    occupant: Vec<Option<BlockId>>,   // slots[s] currently holds occupant[s]  (the residency bitmask)
    table: HashMap<BlockId, usize>,   // BlockId -> slot index (O(1) residency lookup)
    probation: VecDeque<usize>,       // SLRU segment 1 (slot indices, LRU front = coldest)
    protected: VecDeque<usize>,       // SLRU segment 2 (hot; capped at ~0.8*N)
    free: Vec<usize>,                 // unused slot indices (startup)
    ghost: HashSet<BlockId>,          // SECOND-MISS admission: seen-once keys, no payload
    /// Transient double-buffered staging slots for blocks NOT admitted (first miss). Two so an
    /// in-flight prefetch and the consuming GEMM don't clobber each other; sized `max_block_bytes`.
    staging: Vec<CudaSlice<u8>>,
    staging_next: usize,

    n: usize,
    protected_cap: usize,
    max_block_bytes: usize,

    /// SPILLING-PLAN §4: true when async (copy-stream) prefetch can write into cache slots. While
    /// false (today's default), the store-before-evict barrier in `admit` is skipped (no copy-stream
    /// H2D is in flight, so reusing a slot cannot race one). The disk-tier prefetch sets this on.
    prefetch_active: bool,

    // --- §D.4 instrumentation ---
    pub hits: u64,
    pub misses: u64,
    pub staged_bytes: u64,    // total H2D bytes the cache caused (admit + first-miss transient)
}

impl MoeSlotCache {
    /// Build the cache sizing N from free VRAM (MOE-SLRU-PLAN §B.4): probe free VRAM AFTER residents
    /// are loaded; N is shared across ALL layers so it must hold the WHOLE-MODEL hot set, not one
    /// layer's. The 35B-A3B keeps its 256 experts HOST-resident, so the GPU has ~20+ GB free at
    /// decode — empirically a 256-slot cache thrashes (~2-7% hit) while a few-thousand-slot cache
    /// reaches ~85%+ steady-state. So the DEFAULT auto-sizes N to fill `BW24_MOE_VRAM_FRAC` (default
    /// 0.40) of free VRAM, clamped to [256, ~hot-set]. `BW24_MOE_SLOTS` forces an exact N.
    pub fn new(e: &Engine, max_block_bytes: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let (free, _total) = e.ctx().mem_get_info()?;
        // hard headroom: never use more than 80% of free; reserve 2 staging buffers.
        let max_by_vram = ((free as f64 * 0.80) as usize / max_block_bytes).saturating_sub(2).max(8);
        let want = if let Some(n) = std::env::var("BW24_MOE_SLOTS").ok().and_then(|s| s.parse::<usize>().ok()) {
            n
        } else {
            // auto: fill BW24_MOE_VRAM_FRAC of free VRAM with slots (default 40%).
            let frac = std::env::var("BW24_MOE_VRAM_FRAC").ok().and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.40);
            (((free as f64 * frac) as usize) / max_block_bytes).max(256)
        };
        let n = want.min(max_by_vram).max(8);

        let mut slots = Vec::with_capacity(n);
        let mut occupant = Vec::with_capacity(n);
        let mut free_list = Vec::with_capacity(n);
        for s in 0..n {
            slots.push(e.alloc_u8(max_block_bytes)?);
            occupant.push(None);
            free_list.push(n - 1 - s); // push reversed so pop() yields 0,1,2,... (deterministic fill)
        }
        let staging = vec![e.alloc_u8(max_block_bytes)?, e.alloc_u8(max_block_bytes)?];
        let protected_cap = ((n as f64 * 0.8) as usize).max(1);

        Ok(MoeSlotCache {
            slots, occupant, table: HashMap::with_capacity(n * 2),
            probation: VecDeque::new(), protected: VecDeque::new(), free: free_list,
            ghost: HashSet::new(), staging, staging_next: 0,
            n, protected_cap, max_block_bytes,
            prefetch_active: false,
            hits: 0, misses: 0, staged_bytes: 0,
        })
    }

    /// SPILLING-PLAN §4: enable the store-before-evict barrier (arm it once disk-tier prefetch is
    /// turned on so a copy-stream H2D into a slot can never race the slot's reuse on eviction).
    #[inline]
    pub fn set_prefetch_active(&mut self, on: bool) { self.prefetch_active = on; }

    #[inline]
    pub fn n_slots(&self) -> usize { self.n }
    #[inline]
    pub fn max_block_bytes(&self) -> usize { self.max_block_bytes }

    /// O(1) residency check (the ktransformers `generate_gpu_experts_masks` analog).
    #[inline]
    pub fn resident(&self, id: BlockId) -> Option<usize> { self.table.get(&id).copied() }

    /// HIT promotion (SLRU): on a probation hit promote to protected; on a protected hit bump to MRU.
    fn on_hit(&mut self, slot: usize) {
        if let Some(pos) = self.probation.iter().position(|&x| x == slot) {
            self.probation.remove(pos);
            self.push_protected(slot);
        } else if let Some(pos) = self.protected.iter().position(|&x| x == slot) {
            self.protected.remove(pos);
            self.protected.push_back(slot); // MRU
        } else {
            // not in either segment (shouldn't happen for a resident slot) — treat as protected MRU
            self.push_protected(slot);
        }
    }

    /// Push a slot to protected MRU; if protected exceeds its cap, demote its LRU front to probation.
    fn push_protected(&mut self, slot: usize) {
        self.protected.push_back(slot);
        while self.protected.len() > self.protected_cap {
            if let Some(demoted) = self.protected.pop_front() {
                self.probation.push_back(demoted);
            } else { break; }
        }
    }

    /// Pick + free an eviction victim slot, removing its occupant from the table. Victim = probation
    /// LRU front; if probation empty, demote protected LRU -> probation then evict. Returns the slot.
    fn evict_one(&mut self) -> usize {
        if let Some(s) = self.probation.pop_front() {
            if let Some(old) = self.occupant[s].take() { self.table.remove(&old); }
            return s;
        }
        if let Some(s) = self.protected.pop_front() {
            if let Some(old) = self.occupant[s].take() { self.table.remove(&old); }
            return s;
        }
        unreachable!("evict_one called with no resident slots (n>=8 guarantees occupancy)");
    }

    /// Stage `host_bytes` into a NON-retained transient staging slot (first miss, ghost-only). The
    /// caller runs the GEMM from this slot THIS token but the block is not retained. Double-buffered.
    fn stage_transient(&mut self, host_bytes: &[u8], e: &Engine)
                       -> Result<usize, Box<dyn std::error::Error>> {
        let idx = self.staging_next;
        self.staging_next ^= 1; // toggle 0<->1
        e.stage_expert(host_bytes, &mut self.staging[idx], 0)?;
        self.staged_bytes += host_bytes.len() as u64;
        Ok(idx)
    }

    /// Admit a block: evict a victim, stage `host_bytes` into its slot, register residency, place in
    /// probation (new admissions enter probation — they earn promotion on a later hit).
    fn admit(&mut self, id: BlockId, host_bytes: &[u8], e: &Engine)
             -> Result<usize, Box<dyn std::error::Error>> {
        let reused = self.free.is_empty();   // no free slot => we are about to evict + reuse one
        let slot = if let Some(s) = self.free.pop() { s } else { self.evict_one() };
        // SPILLING-PLAN §4 — STORE-BEFORE-EVICT BARRIER. Before staging into a REUSED (just-evicted)
        // slot, drain the copy stream so an in-flight async H2D into that slot (issued by disk-tier
        // prefetch on the copy stream) cannot race the new occupant -> use-after-free -> silent wrong
        // tokens (would break the argmax GATE). This is a NO-OP today: the stage below runs on the
        // DEFAULT compute stream (`stage_expert`), and no prefetch issues copy-stream H2D into cache
        // slots yet. It is wired now (gated on `prefetch_active`) so the disk-tier prefetch is correct
        // from the moment it is turned on. The vLLM/LMCache eviction-barrier pattern.
        if reused && self.prefetch_active {
            e.copy_stream.synchronize()?;
        }
        e.stage_expert(host_bytes, &mut self.slots[slot], 0)?;
        self.staged_bytes += host_bytes.len() as u64;
        self.occupant[slot] = Some(id);
        self.table.insert(id, slot);
        self.probation.push_back(slot);
        Ok(slot)
    }

    /// The dispatch decision for one (BlockId, host_bytes). Returns where the block landed; resolve
    /// the device buffer with `buf()`. On the bit-identity-critical path the buffer holds EXACTLY
    /// `host_bytes` either way (a HIT skipped the copy; the prior stage wrote the same bytes).
    ///
    /// Policy (MOE-SLRU-PLAN §B.2):
    /// - HIT  (table[id] = s): promote, return s. ZERO PCIe.
    /// - MISS, id not in ghost (FIRST miss): insert into ghost, stage transient, DON'T retain.
    /// - MISS, id in ghost   (SECOND miss): admit (stage into a retained slot), remove from ghost.
    pub fn dispatch(&mut self, id: BlockId, host_bytes: &[u8], e: &Engine)
                    -> Result<DispatchSlot, Box<dyn std::error::Error>> {
        if let Some(s) = self.table.get(&id).copied() {
            self.hits += 1;
            self.on_hit(s);
            return Ok(DispatchSlot::Resident(s));
        }
        self.misses += 1;
        if self.ghost.contains(&id) {
            self.ghost.remove(&id);
            let s = self.admit(id, host_bytes, e)?;
            Ok(DispatchSlot::Resident(s))
        } else {
            self.ghost.insert(id);
            let s = self.stage_transient(host_bytes, e)?;
            Ok(DispatchSlot::Staging(s))
        }
    }

    /// Pre-warm: force-admit a block (used by the §D.2 bit-identity gate to make all blocks resident).
    /// Bypasses the ghost filter — admits on the spot.
    pub fn force_admit(&mut self, id: BlockId, host_bytes: &[u8], e: &Engine)
                       -> Result<usize, Box<dyn std::error::Error>> {
        if let Some(s) = self.table.get(&id).copied() { return Ok(s); }
        self.admit(id, host_bytes, e)
    }

    /// Resolve a `DispatchSlot` to the device buffer to feed `qmatvec_view`.
    #[inline]
    pub fn buf(&self, d: DispatchSlot) -> &CudaSlice<u8> {
        match d {
            DispatchSlot::Resident(s) => &self.slots[s],
            DispatchSlot::Staging(s) => &self.staging[s],
        }
    }

    /// Read-only access to a slot's device buffer (the `qmatvec_view` source on a HIT).
    #[inline]
    pub fn slot(&self, s: usize) -> &CudaSlice<u8> { &self.slots[s] }

    /// Hit rate over this cache's lifetime (for the §D.4 print).
    pub fn hit_rate(&self) -> f64 {
        let tot = self.hits + self.misses;
        if tot == 0 { 0.0 } else { self.hits as f64 / tot as f64 }
    }

    /// Reset the per-window perf counters (lets the run print steady-state vs warmup separately).
    pub fn reset_counters(&mut self) {
        self.hits = 0; self.misses = 0; self.staged_bytes = 0;
    }
}
