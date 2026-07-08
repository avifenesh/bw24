//! EDGE-1 §B: SLRU GPU expert-residency cache (MOE-SLRU-PLAN §B).
//!
//! Stage-1 `moe_ffn` re-stages EVERY routed expert EVERY token over PCIe into one scratch slot.
//! The same ~15-20% of experts recur (the "hot expert" mass), so an SLRU residency cache makes
//! the steady-state re-stage count -> ~0. The cache holds N fixed-address GPU slots (never
//! re-allocated, never fragmented), a `BlockId -> slot` residency table, an SLRU eviction policy
//! (probation + protected segments; the second-miss "ghost" admission filter was measured a net
//! loss in both regimes and removed 2026-07-08 — first-miss admit is the policy) so a one-off cold
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

/// Where a dispatched block landed (always a retained resident slot since the first-miss-admit
/// policy, 2026-07-08 — the transient staging tier went with the ghost filter).
#[derive(Clone, Copy, Debug)]
pub enum DispatchSlot {
    Resident(usize),
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

    n: usize,
    protected_cap: usize,
    max_block_bytes: usize,

    /// SPILLING-PLAN §4: true when async (copy-stream) prefetch can write into cache slots. While
    /// false (today's default), the store-before-evict barrier in `admit` is skipped (no copy-stream
    /// H2D is in flight, so reusing a slot cannot race one). The disk-tier prefetch sets this on.
    prefetch_active: bool,

    // --- LAUNCH-STRUCTURE STAGE 3 (2026-07-05): device-side expert-pointer indirection ---
    /// Resident-block count per LAYER (all 3 projections summed). When a layer reaches
    /// 3*n_expert every routed block of that layer is cache-resident at a fixed address, so the
    /// whole layer can dispatch via the DEVICE pointer table with ZERO host routing (no router
    /// DtoH, no per-layer stream sync — the round-trip stall the decode profile measured at
    /// ~36us x 40 layers/token). Maintained by admit/evict.
    per_layer: HashMap<u16, u32>,
    /// Per-layer device pointer row [3, n_expert] of slot base addresses (u64), uploaded lazily
    /// when the layer first reads as fully resident. Slots are fixed-address for the cache's
    /// lifetime, so a row stays valid until an eviction touches that layer (which drops the row
    /// -> re-upload on next full residency).
    dev_rows: HashMap<u16, CudaSlice<u64>>,
    /// Layers whose one-shot prewarm was already attempted (success or not) — spill rigs whose
    /// free slots can't hold a full layer must not re-scan 3*n_expert blocks every token.
    prewarm_tried: HashSet<u16>,

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
        // hard headroom: never use more than 80% of free; keep 2 blocks of slack.
        let max_by_vram = ((free as f64 * 0.80) as usize / max_block_bytes).saturating_sub(2).max(8);
        let want = if let Some(n) = std::env::var("BW24_MOE_SLOTS").ok().and_then(|s| s.parse::<usize>().ok()) {
            n
        } else {
            // auto: fill BW24_MOE_VRAM_FRAC of free VRAM with slots (default 40%).
            // DEFAULT 0.85 (2026-07-06 local sweep: 0.40=25.0, 0.60=28.0, 0.85=28.5 tok/s on the
            // spill-regime 35B — hit-rate 87.8% -> 99.2%, PCIe 55 -> 3.8 MB/tok; the 0.80
            // hard-headroom cap below still bounds the true allocation, so 0.85 requests the max).
            // Rigs co-running other GPU work should set BW24_MOE_VRAM_FRAC lower.
            let frac = std::env::var("BW24_MOE_VRAM_FRAC").ok().and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.85);
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
        let protected_cap = ((n as f64 * 0.8) as usize).max(1);

        Ok(MoeSlotCache {
            slots, occupant, table: HashMap::with_capacity(n * 2),
            probation: VecDeque::new(), protected: VecDeque::new(), free: free_list,
            n, protected_cap, max_block_bytes,
            prefetch_active: false,
            per_layer: HashMap::new(), dev_rows: HashMap::new(), prewarm_tried: HashSet::new(),
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
    ///
    /// O(1) EARLY-OUT (STAGING-ELISION stage, 2026-07-04): while FREE slots remain, `admit` pops
    /// `free` and `evict_one` is unreachable — recency order is dead state until the cache fills.
    /// The promotion below is a linear scan of two VecDeques (O(n_slots) PER HIT; at ~46k slots x
    /// ~850 hits/token that was ~40M host ops/token — measured as the fast-admit A/B regression
    /// 48.5 -> 46.0 tok/s on the 35B g7e decode). Skip it until eviction is possible. On 96GB
    /// (slots >= whole-model block count) every HIT stays an O(1) table lookup forever; on spill
    /// rigs this only defers SLRU ordering to when eviction pressure actually exists.
    /// Bookkeeping-only: the dispatched bytes are identical either way (the D.2 gate pins it).
    fn on_hit(&mut self, slot: usize) {
        if !self.free.is_empty() { return; }
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
            if let Some(old) = self.occupant[s].take() {
                self.table.remove(&old);
                self.on_block_evicted(old.layer);
            }
            return s;
        }
        if let Some(s) = self.protected.pop_front() {
            if let Some(old) = self.occupant[s].take() {
                self.table.remove(&old);
                self.on_block_evicted(old.layer);
            }
            return s;
        }
        unreachable!("evict_one called with no resident slots (n>=8 guarantees occupancy)");
    }

    /// STAGE 3 bookkeeping: a resident block of `layer` was evicted — the layer is no longer fully
    /// resident, so its device pointer row (if uploaded) must be invalidated. NOTE: the row's device
    /// buffer is dropped here, which is safe because the fully-resident fast path is only taken when
    /// `dev_rows` contains the layer at DISPATCH time and all launches consuming the row were
    /// enqueued BEFORE this eviction's staging memcpy on the same stream (single-stream ordering).
    fn on_block_evicted(&mut self, layer: u16) {
        if let Some(c) = self.per_layer.get_mut(&layer) { *c -= 1; }
        self.dev_rows.remove(&layer);
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
        *self.per_layer.entry(id.layer).or_insert(0) += 1;  // STAGE 3 residency count
        Ok(slot)
    }

    /// The dispatch decision for one (BlockId, host_bytes). Returns where the block landed; resolve
    /// the device buffer with `buf()`. On the bit-identity-critical path the buffer holds EXACTLY
    /// `host_bytes` either way (a HIT skipped the copy; the prior stage wrote the same bytes).
    ///
    /// Policy (MOE-SLRU-PLAN §B.2, first-miss admit since 2026-07-06):
    /// - HIT  (table[id] = s): promote, return s. ZERO PCIe.
    /// - MISS: admit (stage into a retained slot, evicting an SLRU victim when full).
    pub fn dispatch(&mut self, id: BlockId, host_bytes: &[u8], e: &Engine)
                    -> Result<DispatchSlot, Box<dyn std::error::Error>> {
        if let Some(s) = self.table.get(&id).copied() {
            self.hits += 1;
            self.on_hit(s);
            return Ok(DispatchSlot::Resident(s));
        }
        self.misses += 1;
        // FIRST-MISS ADMIT (the only policy since 2026-07-08; the second-miss "ghost" filter and
        // its seams BW24_MOE_GHOST / BW24_MOE_FAST_ADMIT are gone). Measured record: while FREE
        // slots remain, admission evicts nothing — filtering only delayed residency (96GB: 83.7%
        // steady hit-rate instead of ~100%, 74 MB/token avoidable PCIe; 2026-07-04). In the SPILL
        // regime (cache permanently full, local 35B) the filter made every cold block pay TWO H2D
        // copies — ~6% of token PCIe, measured ABOVE its eviction-protection benefit (24.2 -> 25.0
        // tok/s with it off, 2026-07-06). First-miss admit evicts an SLRU victim when full; the
        // SLRU probation segment still protects the protected set. Bit-identity unchanged: the
        // slot holds byte-for-byte the same GGUF block (D.2 gate).
        let s = self.admit(id, host_bytes, e)?;
        Ok(DispatchSlot::Resident(s))
    }

    /// Pre-warm: force-admit a block (used by the §D.2 bit-identity gate to make all blocks resident).
    pub fn force_admit(&mut self, id: BlockId, host_bytes: &[u8], e: &Engine)
                       -> Result<usize, Box<dyn std::error::Error>> {
        if let Some(s) = self.table.get(&id).copied() { return Ok(s); }
        self.admit(id, host_bytes, e)
    }

    /// STAGE 3 one-shot PREWARM: force-admit every block of `layer` while FREE slots can hold it
    /// (never evicts — a spill rig whose cache can't fit the layer just skips; organic residency
    /// still applies). Runs at most once per layer (success or not). The H2D copies are the SAME
    /// stage_expert bytes the miss path would issue — bit-identity unchanged; this only front-loads
    /// them so the device-dispatch fast path fires from token 0 instead of after the SLRU fill.
    pub fn prewarm_layer(&mut self, layer: u16, m: &crate::hybrid::MoeWeights, e: &Engine)
                         -> Result<(), Box<dyn std::error::Error>> {
        if !self.prewarm_tried.insert(layer) { return Ok(()); }
        let n_expert = m.gate_exps.n_expert;
        let resident = self.per_layer.get(&layer).copied().unwrap_or(0) as usize;
        let missing = 3 * n_expert - resident;
        if self.free.len() < missing { return Ok(()); }  // won't evict for a prewarm
        for ex in 0..n_expert {
            for (proj, exps) in [(PROJ_GATE, &m.gate_exps), (PROJ_UP, &m.up_exps),
                                 (PROJ_DOWN, &m.down_exps)] {
                let id = BlockId::new(layer, proj, ex as u16);
                if self.table.contains_key(&id) { continue; }
                self.admit(id, exps.expert_bytes(ex), e)?;
            }
        }
        Ok(())
    }

    /// STAGE 3: device pointer row for a FULLY-RESIDENT layer. Returns the [3, n_expert] u64 slot
    /// base-address table (proj-major: gate row, up row, down row) if EVERY block of `layer` is
    /// cache-resident, else None (caller falls back to host routing). The row is built+uploaded on
    /// first full residency and reused until an eviction touches the layer. `n_expert` is the
    /// layer's expert count (the full-residency threshold is 3*n_expert blocks).
    pub fn layer_dev_row(&mut self, layer: u16, n_expert: usize, e: &Engine)
                         -> Result<Option<&CudaSlice<u64>>, Box<dyn std::error::Error>> {
        if self.per_layer.get(&layer).copied().unwrap_or(0) as usize != 3 * n_expert {
            return Ok(None);
        }
        if !self.dev_rows.contains_key(&layer) {
            use cudarc::driver::DevicePtr;
            let mut host = vec![0u64; 3 * n_expert];
            for proj in 0..3u8 {
                for ex in 0..n_expert {
                    let Some(&s) = self.table.get(&BlockId::new(layer, proj, ex as u16)) else {
                        // count said fully resident but a block is missing — inconsistent; bail safe.
                        return Ok(None);
                    };
                    let (p, _ev) = self.slots[s].device_ptr(e.stream());
                    host[proj as usize * n_expert + ex] = p as u64;
                }
            }
            let row = e.stream().clone_htod(&host)?;
            self.dev_rows.insert(layer, row);
        }
        Ok(self.dev_rows.get(&layer))
    }

    /// Resolve a `DispatchSlot` to the device buffer to feed `qmatvec_view`.
    #[inline]
    pub fn buf(&self, d: DispatchSlot) -> &CudaSlice<u8> {
        match d {
            DispatchSlot::Resident(s) => &self.slots[s],
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
