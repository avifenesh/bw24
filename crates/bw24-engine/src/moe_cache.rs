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
use std::sync::Arc;
use cudarc::driver::{CudaEvent, CudaSlice, CudaStream, HostSlice, SyncOnDrop};
use crate::Engine;
use crate::model::{ExpertKeepalive, ExpertSource};
use crate::spill_pread::{PreadPool, PreadStats};

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
    /// Copy-stream prefetches that have reserved a slot but are not visible in `table` until the
    /// consumer inserts an explicit compute-stream wait for `ready`. Pending slots are absent from
    /// both SLRU queues, so neither synchronous admission nor another prefetch can evict them.
    pending: HashMap<BlockId, PendingBlock>,
    /// Source owners whose copy completed submission but not yet DMA completion. They are reaped
    /// only after the recorded copy-stream event reports complete.
    inflight_sources: Vec<(Arc<CudaEvent>, ExpertKeepalive)>,
    /// Owners for copies whose completion could not be proved. Kept until a whole-stream drain;
    /// leaked with the GPU slots if teardown cannot establish safety.
    quarantined_sources: Vec<ExpertKeepalive>,
    /// Unique owners used by demand/fallback H2D on the compute stream. `stage_expert` receives a
    /// raw byte slice, so cudarc cannot attach its own source-lifetime event. Retain each backing
    /// allocation once until cache teardown instead of paying one CUDA event per miss.
    compute_sources: HashMap<KeepaliveKey, ExpertKeepalive>,
    /// Opt-in blocking positioned-read proof backend. Its pinned buffers remain owned here until
    /// their explicit compute-stream completion events fire.
    pread: Option<PreadPool>,
    pread_requested: bool,
    pread_fallbacks: u64,
    /// Retained so an event-creation failure after copy submission can be drained again during
    /// teardown. A slot touched by an unprovable copy is quarantined outside every cache queue.
    copy_stream: Arc<CudaStream>,
    copy_stream_unknown: bool,
    compute_stream: Arc<CudaStream>,
    compute_stream_unknown: bool,

    n: usize,
    protected_cap: usize,
    max_block_bytes: usize,

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

struct PendingBlock {
    slot: usize,
    ready: Arc<CudaEvent>,
    keepalive: Option<ExpertKeepalive>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum KeepaliveKey {
    Pinned(usize),
    Buffer(usize),
    Mmap(usize),
}

impl KeepaliveKey {
    fn from_owner(owner: &ExpertKeepalive) -> Self {
        match owner {
            ExpertKeepalive::Pinned(value) => Self::Pinned(Arc::as_ptr(value) as usize),
            ExpertKeepalive::Buffer(value) => Self::Buffer(Arc::as_ptr(value) as usize),
            ExpertKeepalive::Mmap(value) => Self::Mmap(Arc::as_ptr(value) as usize),
        }
    }
}

/// Exact-length view over one CUDA-pinned pool allocation. cudarc's raw `&[u8]` HostSlice waits
/// for the whole stream before returning, while passing `PinnedHostSlice` would copy its full
/// capacity. This wrapper submits exactly the expert prefix; the caller records and retains the
/// completion event before the backing allocation can be reused.
struct ExactPinnedPrefix<'a>(&'a [u8]);

impl HostSlice<u8> for ExactPinnedPrefix<'_> {
    fn len(&self) -> usize { self.0.len() }

    unsafe fn stream_synced_slice<'a>(
        &'a self,
        _stream: &'a CudaStream,
    ) -> (&'a [u8], SyncOnDrop<'a>) {
        // SAFETY: stage_pread_on_compute_stream records an explicit event immediately after the
        // async memcpy and PreadPool retains both allocation and event until it completes.
        (self.0, SyncOnDrop::Record(None))
    }

    unsafe fn stream_synced_mut_slice<'a>(
        &'a mut self,
        _stream: &'a CudaStream,
    ) -> (&'a mut [u8], SyncOnDrop<'a>) {
        panic!("ExactPinnedPrefix is a source-only HostSlice")
    }
}

fn stage_on_copy_stream(
    e: &Engine,
    host_bytes: &[u8],
    slot: &mut CudaSlice<u8>,
) -> Result<Arc<CudaEvent>, (Box<dyn std::error::Error>, bool)> {
    // Protect all earlier compute-stream users of a reused slot before the copy stream overwrites it.
    let prior = match e.stream().record_event(None) {
        Ok(prior) => prior,
        Err(err) => return Err((err.into(), true)),
    };
    if let Err(err) = e.copy_stream.wait(&prior) {
        return Err((err.into(), true));
    }
    match e.stage_expert_async(host_bytes, slot, 0) {
        Ok(ready) => Ok(Arc::new(ready)),
        Err(err) => {
            // The H2D may have been submitted before event creation failed. Never release either the
            // destination slot or pinned source until the copy stream has drained.
            match e.copy_stream.synchronize() {
                Ok(()) => Err((err, true)),
                Err(sync_err) => Err((std::io::Error::other(format!(
                    "copy-stream H2D setup failed ({err}); stream drain also failed ({sync_err})"
                )).into(), false)),
            }
        }
    }
}

fn stage_pread_on_compute_stream(
    e: &Engine,
    host_bytes: &[u8],
    slot: &mut CudaSlice<u8>,
) -> Result<Arc<CudaEvent>, Box<dyn std::error::Error>> {
    let ready = Arc::new(e.ctx().new_event(None)?);
    let source = ExactPinnedPrefix(host_bytes);
    let mut dst = slot.slice_mut(0..host_bytes.len());
    e.stream().memcpy_htod(&source, &mut dst)?;
    ready.record(e.stream())?;
    Ok(ready)
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
        let pread_requested = crate::spill_pread::enabled();
        let pread = if pread_requested {
            match PreadPool::try_new(e, max_block_bytes) {
                Ok(pool) => Some(pool),
                Err(err) => {
                    eprintln!("[spill-pread] pinned-buffer initialization failed ({err}); using mmap");
                    None
                }
            }
        } else { None };

        Ok(MoeSlotCache {
            slots, occupant, table: HashMap::with_capacity(n * 2),
            probation: VecDeque::new(), protected: VecDeque::new(), free: free_list,
            pending: HashMap::new(), inflight_sources: Vec::new(),
            quarantined_sources: Vec::new(), compute_sources: HashMap::new(),
            pread, pread_requested, pread_fallbacks: 0,
            copy_stream: e.copy_stream.clone(), copy_stream_unknown: false,
            compute_stream: e.stream().clone(), compute_stream_unknown: false,
            n, protected_cap, max_block_bytes,
            per_layer: HashMap::new(), dev_rows: HashMap::new(), prewarm_tried: HashSet::new(),
            hits: 0, misses: 0, staged_bytes: 0,
        })
    }

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

    /// Pick a resident victim that is not needed by the expert currently being computed. Pending
    /// slots never enter the SLRU queues, so they are excluded automatically. Returns `None` rather
    /// than evicting a protected block; the caller then leaves this block to the synchronous path.
    fn evict_one_excluding(&mut self, keep: &[BlockId]) -> Option<usize> {
        let take = |q: &mut VecDeque<usize>, occupant: &[Option<BlockId>]| {
            q.iter()
                .position(|&s| occupant[s].is_some_and(|id| !keep.contains(&id)))
                .and_then(|pos| q.remove(pos))
        };
        let slot = take(&mut self.probation, &self.occupant)
            .or_else(|| take(&mut self.protected, &self.occupant))?;
        if let Some(old) = self.occupant[slot].take() {
            self.table.remove(&old);
            self.on_block_evicted(old.layer);
        }
        Some(slot)
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

    fn reserve_slot(&mut self) -> usize {
        if let Some(slot) = self.free.pop() { slot } else { self.evict_one() }
    }

    fn release_reserved_slot(&mut self, slot: usize) {
        debug_assert!(self.occupant[slot].is_none());
        self.free.push(slot);
    }

    fn publish(&mut self, id: BlockId, slot: usize) {
        self.occupant[slot] = Some(id);
        self.table.insert(id, slot);
        self.probation.push_back(slot);
        *self.per_layer.entry(id.layer).or_insert(0) += 1;
    }

    fn reap_copy_sources(&mut self) {
        self.inflight_sources.retain(|(ready, _)| !ready.is_complete());
    }

    fn retain_compute_source(&mut self, owner: Option<ExpertKeepalive>) {
        if let Some(owner) = owner {
            let key = KeepaliveKey::from_owner(&owner);
            self.compute_sources.entry(key).or_insert(owner);
        }
    }

    /// Admit a block: evict a victim, stage `host_bytes` into its slot, register residency, place in
    /// probation (new admissions enter probation — they earn promotion on a later hit).
    fn admit(&mut self, id: BlockId, host_bytes: &[u8], e: &Engine)
             -> Result<usize, Box<dyn std::error::Error>> {
        let slot = self.reserve_slot();
        // Pending copy-stream admissions are not in either SLRU queue, so `evict_one` cannot return
        // an in-flight slot. This synchronous copy and its consumer remain ordered on gpu.stream.
        if let Err(err) = e.stage_expert(host_bytes, &mut self.slots[slot], 0) {
            return match e.stream().synchronize() {
                Ok(()) => {
                    self.release_reserved_slot(slot);
                    Err(err)
                }
                Err(sync_err) => {
                    // Keep the slot outside free/table/SLRU. Drop retries the stream drain and
                    // leaks every slot if CUDA never provides a completion proof.
                    self.compute_stream_unknown = true;
                    Err(std::io::Error::other(format!(
                        "compute-stream H2D setup failed ({err}); stream drain also failed ({sync_err})"
                    )).into())
                }
            };
        }
        self.staged_bytes += host_bytes.len() as u64;
        self.publish(id, slot);
        Ok(slot)
    }

    fn note_pread_fallback(&mut self, reason: &dyn std::fmt::Display) {
        self.pread_fallbacks += 1;
        if let Some(pool) = self.pread.as_mut() {
            pool.note_fallback();
        }
        if self.pread_fallbacks <= 3 {
            eprintln!("[spill-pread] falling back to mmap: {reason}");
        }
    }

    fn dispatch_disk(
        &mut self,
        id: BlockId,
        file: &Arc<std::fs::File>,
        offset: u64,
        len: usize,
        fallback: &[u8],
        e: &Engine,
    ) -> Result<DispatchSlot, Box<dyn std::error::Error>> {
        if self.pread.is_none() {
            if self.pread_requested {
                self.note_pread_fallback(&"pinned-buffer backend unavailable");
            }
            return Ok(DispatchSlot::Resident(self.admit(id, fallback, e)?));
        }

        let read = self.pread.as_mut().unwrap().read(file.as_ref(), offset, len);
        let index = match read {
            Ok(index) => index,
            Err(err) => {
                self.note_pread_fallback(err.as_ref());
                return Ok(DispatchSlot::Resident(self.admit(id, fallback, e)?));
            }
        };

        // The blocking read happens before eviction, so an I/O failure leaves cache residency
        // untouched and can safely use the mmap oracle.
        let slot = self.reserve_slot();
        let ready = {
            let bytes = match self.pread.as_ref().unwrap().bytes(index, len) {
                Ok(bytes) => bytes,
                Err(err) => {
                    self.pread.as_mut().unwrap().abort_read(index);
                    self.release_reserved_slot(slot);
                    self.note_pread_fallback(err.as_ref());
                    return Ok(DispatchSlot::Resident(self.admit(id, fallback, e)?));
                }
            };
            stage_pread_on_compute_stream(e, bytes, &mut self.slots[slot])
        };
        let ready = match ready {
            Ok(ready) => ready,
            Err(err) => {
                // A memcpy or event-record failure can occur after submission. Synchronize the
                // retained compute stream before either source or destination is reused. If CUDA
                // cannot prove completion, quarantine both and fail instead of risking UAF.
                match e.stream().synchronize() {
                    Ok(()) => {
                        self.pread.as_mut().unwrap().abort_read(index);
                        self.release_reserved_slot(slot);
                        self.note_pread_fallback(err.as_ref());
                        return Ok(DispatchSlot::Resident(self.admit(id, fallback, e)?));
                    }
                    Err(sync_err) => {
                        self.pread.as_mut().unwrap().mark_unknown_h2d(index);
                        return Err(std::io::Error::other(format!(
                            "pread H2D setup failed ({err}); CUDA stream drain also failed ({sync_err})"
                        )).into());
                    }
                }
            }
        };
        self.pread.as_mut().unwrap().mark_h2d(index, ready);
        // Copy and dependent GEMM share the compute stream, so stream order is the consumer fence.
        // Publish only after both memcpy submission and explicit completion-event recording.
        self.staged_bytes += len as u64;
        self.publish(id, slot);
        Ok(DispatchSlot::Resident(slot))
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
        self.dispatch_source(id, ExpertSource::Memory { bytes: host_bytes, keepalive: None }, e)
    }

    pub(crate) fn dispatch_source(&mut self, id: BlockId, source: ExpertSource<'_>, e: &Engine)
                                  -> Result<DispatchSlot, Box<dyn std::error::Error>> {
        self.reap_copy_sources();
        if let Some(s) = self.table.get(&id).copied() {
            self.hits += 1;
            self.on_hit(s);
            return Ok(DispatchSlot::Resident(s));
        }
        if let Some(pending) = self.pending.remove(&id) {
            if let Err(err) = e.compute_wait(pending.ready.as_ref()) {
                self.pending.insert(id, pending);
                return Err(err);
            }
            self.misses += 1;
            let slot = pending.slot;
            if let Some(keepalive) = pending.keepalive {
                self.inflight_sources.push((pending.ready, keepalive));
            }
            self.publish(id, slot);
            return Ok(DispatchSlot::Resident(slot));
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
        match source {
            ExpertSource::Memory { bytes, keepalive } => {
                // Retain before H2D submission so even the setup-error path cannot release a
                // pinned/mapped source while CUDA may still be reading it.
                self.retain_compute_source(keepalive);
                let slot = self.admit(id, bytes, e)?;
                Ok(DispatchSlot::Resident(slot))
            }
            ExpertSource::Disk { file, offset, len, fallback, keepalive } => {
                // The owner is only needed when dispatch_disk falls back to mmap, but retaining
                // the usually shared mmap Arc once keeps every fallback branch simple and safe.
                self.retain_compute_source(Some(keepalive));
                self.dispatch_disk(id, file, offset, len, fallback, e)
            }
        }
    }

    /// Deterministically stage a known-future block on the copy stream. The slot is reserved but is
    /// not considered resident until `dispatch` inserts a compute-stream wait for the returned copy
    /// event. Before overwriting a reused slot, the copy stream waits for all compute work already
    /// queued at this call site; the caller issues prefetch before the current expert's kernels, so
    /// the transfer can overlap those kernels without racing any earlier consumer of the victim.
    ///
    /// `keep` is the current expert's gate/up/down ids. If no safe victim exists, return `false` and
    /// let the normal synchronous miss path handle the block.
    pub fn prefetch(&mut self, id: BlockId, host_bytes: &[u8], keep: &[BlockId], e: &Engine)
                    -> Result<bool, Box<dyn std::error::Error>> {
        self.prefetch_source(
            id,
            ExpertSource::Memory { bytes: host_bytes, keepalive: None },
            keep,
            e,
        )
    }

    fn reserve_prefetch_slot(&mut self, keep: &[BlockId]) -> Option<usize> {
        self.free.pop().or_else(|| self.evict_one_excluding(keep))
    }

    fn prefetch_bytes(&mut self, id: BlockId, host_bytes: &[u8],
                      keepalive: Option<ExpertKeepalive>, keep: &[BlockId], e: &Engine)
                      -> Result<bool, Box<dyn std::error::Error>> {
        let Some(slot) = self.reserve_prefetch_slot(keep) else { return Ok(false) };
        let ready = match stage_on_copy_stream(e, host_bytes, &mut self.slots[slot]) {
            Ok(ready) => ready,
            Err((err, reusable)) => {
                if reusable {
                    self.release_reserved_slot(slot);
                } else {
                    // The slot is absent from free/table/SLRU and cannot be reused. Drop retries a
                    // whole copy-stream drain and leaks all slots if CUDA still cannot prove safety.
                    self.copy_stream_unknown = true;
                    if let Some(keepalive) = keepalive {
                        self.quarantined_sources.push(keepalive);
                    }
                    eprintln!("[moe-cache] quarantining slot {slot} after unprovable copy completion");
                }
                return Err(err);
            }
        };
        self.occupant[slot] = Some(id);
        self.pending.insert(id, PendingBlock { slot, ready, keepalive });
        self.staged_bytes += host_bytes.len() as u64;
        Ok(true)
    }

    pub(crate) fn prefetch_source(
        &mut self,
        id: BlockId,
        source: ExpertSource<'_>,
        keep: &[BlockId],
        e: &Engine,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        self.reap_copy_sources();
        if self.table.contains_key(&id) || self.pending.contains_key(&id) {
            return Ok(false);
        }
        match source {
            ExpertSource::Memory { bytes, keepalive } => {
                self.prefetch_bytes(id, bytes, keepalive, keep, e)
            }
            // A FileExt read here would block the GPU worker before the current expert launches.
            // True explicit-I/O lookahead needs a worker/io_uring runtime; v1 leaves demand to
            // dispatch. In mmap mode retain the existing fallback-byte prefetch behavior.
            ExpertSource::Disk { fallback, keepalive, .. } => {
                if self.pread.is_some() { Ok(false) }
                else { self.prefetch_bytes(id, fallback, Some(keepalive), keep, e) }
            }
        }
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
        if self.pread.is_some() && (0..n_expert).any(|ex| {
            matches!(m.gate_exps.expert_source(ex), ExpertSource::Disk { .. })
                || matches!(m.up_exps.expert_source(ex), ExpertSource::Disk { .. })
                || matches!(m.down_exps.expert_source(ex), ExpertSource::Disk { .. })
        }) {
            // Prewarm is a whole-layer scan. It must not silently turn explicit demand I/O back
            // into an mmap walk; organic misses will populate the cache through dispatch_source.
            return Ok(());
        }
        let resident = self.per_layer.get(&layer).copied().unwrap_or(0) as usize;
        let missing = 3 * n_expert - resident;
        if self.free.len() < missing { return Ok(()); }  // won't evict for a prewarm
        for ex in 0..n_expert {
            for (proj, exps) in [(PROJ_GATE, &m.gate_exps), (PROJ_UP, &m.up_exps),
                                 (PROJ_DOWN, &m.down_exps)] {
                let id = BlockId::new(layer, proj, ex as u16);
                if self.table.contains_key(&id) { continue; }
                match exps.expert_source(ex) {
                    ExpertSource::Memory { bytes, keepalive } => {
                        self.retain_compute_source(keepalive);
                        self.admit(id, bytes, e)?;
                    }
                    ExpertSource::Disk { fallback, keepalive, .. } => {
                        self.retain_compute_source(Some(keepalive));
                        self.admit(id, fallback, e)?;
                    }
                }
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

    pub(crate) fn pread_stats(&self) -> Option<PreadStats> {
        if !self.pread_requested { return None; }
        let mut stats = self.pread.as_ref().map(PreadPool::stats).unwrap_or_default();
        stats.fallbacks = self.pread_fallbacks;
        Some(stats)
    }
}

impl Drop for MoeSlotCache {
    fn drop(&mut self) {
        // Event tracking is intentionally disabled in Engine. Drain explicit copy-stream handoffs
        // before either the destination slots or pinned read buffers begin field destruction.
        let mut safe_to_drop_slots = true;
        if self.compute_stream_unknown || !self.compute_sources.is_empty() {
            if let Err(err) = self.compute_stream.synchronize() {
                safe_to_drop_slots = false;
                eprintln!("[moe-cache] unknown compute-stream drain failed ({err}); leaking GPU slots for safety");
                for (_, keepalive) in self.compute_sources.drain() {
                    std::mem::forget(keepalive);
                }
            } else {
                self.compute_stream_unknown = false;
                self.compute_sources.clear();
            }
        }
        let need_copy_drain = self.copy_stream_unknown || !self.pending.is_empty()
            || !self.inflight_sources.is_empty() || !self.quarantined_sources.is_empty();
        if need_copy_drain {
            if let Err(err) = self.copy_stream.synchronize() {
                safe_to_drop_slots = false;
                eprintln!("[moe-cache] unknown copy-stream drain failed ({err}); leaking GPU slots for safety");
                for (_, keepalive) in self.inflight_sources.drain(..) {
                    std::mem::forget(keepalive);
                }
                for keepalive in self.quarantined_sources.drain(..) {
                    std::mem::forget(keepalive);
                }
                for (_, pending) in self.pending.drain() {
                    if let Some(keepalive) = pending.keepalive {
                        std::mem::forget(keepalive);
                    }
                }
            } else {
                self.copy_stream_unknown = false;
                self.inflight_sources.clear();
                self.quarantined_sources.clear();
                self.pending.clear();
            }
        }
        if let Some(pool) = self.pread.as_mut() {
            safe_to_drop_slots &= pool.drain();
        } else if self.pread_requested && self.pread_fallbacks != 0 {
            eprintln!(
                "[spill-pread] backend unavailable; mmap_fallbacks={}",
                self.pread_fallbacks
            );
        }
        if !safe_to_drop_slots {
            for slot in self.slots.drain(..) {
                std::mem::forget(slot);
            }
        }
    }
}
