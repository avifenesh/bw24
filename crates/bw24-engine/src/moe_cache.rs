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

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::Arc;
use cudarc::driver::{CudaEvent, CudaSlice, CudaStream, HostSlice, SyncOnDrop};
use crate::Engine;
use crate::model::{ExpertKeepalive, ExpertSource};
use crate::spill_pread::{PreadPool, PreadStats, ReadTicket, SpillIoMode};

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

/// One fixed-address size class with an independent SLRU. Separating queues by capacity prevents a
/// small mixed-layout block from consuming the scarce slots that can hold a larger block.
struct SlotClass {
    capacity: usize,
    probation: VecDeque<usize>,
    protected: VecDeque<usize>,
    free: Vec<usize>,
    protected_cap: usize,
}

/// SLRU GPU expert-residency cache. Slots remain fixed-address for the cache lifetime. Uniform
/// models use one class; mixed-layout models may preallocate several exact-capacity classes.
pub struct MoeSlotCache {
    slots: Vec<CudaSlice<u8>>, // fixed GPU buffers; capacities live in `classes`
    slot_class: Vec<usize>,    // slot index -> size-class index
    classes: Vec<SlotClass>,
    occupant: Vec<Option<BlockId>>, // slots[s] currently holds occupant[s]  (the residency bitmask)
    table: HashMap<BlockId, usize>, // BlockId -> slot index (O(1) residency lookup)
    /// Exponentially aged online access scores for the optional mixed-layout LFU victim policy.
    /// Scores survive eviction and perf-counter resets; an opt-in decode-epoch decay prevents a
    /// batched prompt from permanently outweighing recent token-to-token reuse.
    frequencies: HashMap<BlockId, f32>,
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
    /// Opt-in positioned-read backends. Pinned buffers remain owned here until their explicit
    /// compute-stream completion events fire.
        pread: Option<PreadPool>,
    /// Known-next reads submitted to disk workers but not yet consumed by dispatch. They own pinned
    /// buffers, not GPU slots; all CUDA submission remains on the caller thread.
    worker_reads: HashMap<BlockId, WorkerRead>,
    pread_requested: bool,
    pread_fallbacks: u64,
    /// Retained so an event-creation failure after copy submission can be drained again during
    /// teardown. A slot touched by an unprovable copy is quarantined outside every cache queue.
    copy_stream: Arc<CudaStream>,
    copy_stream_unknown: bool,
    compute_stream: Arc<CudaStream>,
        compute_stream_unknown: bool,

    n: usize,
    max_block_bytes: usize,
    size_aware: bool,
    frequency_evict: bool,
    frequency_decay: Option<f32>,
    /// Relative LFU value of a NextN/MTP access. The MTP block is keyed at `u16::MAX` and is
    /// latency-critical during speculative decode, but contributes only one layer of observations
    /// versus the full trunk. Keep the neutral default; local fixed-residency profiling may raise
    /// it after an exact throughput sweep.
    mtp_frequency_weight: f32,
    last_forward_layer: Option<u16>,
    last_forward_t: usize,
    /// Stable-residency mode for heterogeneous CPU/GPU expert execution. Once frozen, callers may
    /// still read resident slots, but must stage cache misses through transient scratch instead of
    /// changing which experts execute on each backend.
    frozen: bool,

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

#[derive(Clone, Copy)]
struct WorkerRead {
    ticket: ReadTicket,
    len: usize,
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
        // SAFETY: the pread staging helpers record an explicit event immediately after the async
        // memcpy and PreadPool retains both allocation and event until it completes.
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

fn stage_pread_prefetch_on_copy_stream(
    e: &Engine,
    host_bytes: &[u8],
    slot: &mut CudaSlice<u8>,
) -> Result<Arc<CudaEvent>, (Box<dyn std::error::Error>, bool)> {
    let ready = match e.ctx().new_event(None) {
        Ok(ready) => Arc::new(ready),
        Err(err) => return Err((err.into(), true)),
    };
    let source = ExactPinnedPrefix(host_bytes);
    let mut dst = slot.slice_mut(0..host_bytes.len());
    let submitted = e
        .copy_stream
        .memcpy_htod(&source, &mut dst)
        .and_then(|()| ready.record(&e.copy_stream));
    match submitted {
        Ok(()) => Ok(ready),
        Err(err) => match e.copy_stream.synchronize() {
            Ok(()) => Err((err.into(), true)),
            Err(sync_err) => Err((
                std::io::Error::other(format!(
                "pread copy-stream H2D setup failed ({err}); stream drain also failed ({sync_err})"
            ))
                .into(),
                false,
            )),
        },
    }
}

/// Allocate the same fraction of every exact block-size class under one byte budget. This avoids
/// biasing residency toward either low-bit or high-bit tiers while eliminating max-slot padding.
fn size_class_plan(block_bytes: &[usize], budget_bytes: usize) -> Vec<(usize, usize)> {
    let mut counts: BTreeMap<usize, usize> = BTreeMap::new();
    for &bytes in block_bytes.iter().filter(|&&bytes| bytes > 0) {
        *counts.entry(bytes).or_insert(0) += 1;
    }
    if counts.is_empty() || budget_bytes == 0 {
        return Vec::new();
    }
    let total_bytes: u128 = counts
        .iter()
        .map(|(&bytes, &count)| (bytes as u128 + 8) * count as u128)
        .sum();
    let fraction = (budget_bytes as f64 / total_bytes as f64).min(1.0);
    let mut plan: Vec<(usize, usize, f64)> = counts
        .iter()
        .map(|(&bytes, &count)| {
            let exact = count as f64 * fraction;
            (bytes, exact.floor() as usize, exact.fract())
        })
        .collect();
    let mut used: usize = plan
        .iter()
        .map(|(bytes, count, _)| (bytes + 8) * count)
        .sum();

    // Hamilton-style remainder pass keeps class proportions close after flooring. There are only
    // a handful of layout classes, so one additional slot per class covers all rounding loss.
    let mut order: Vec<usize> = (0..plan.len()).collect();
    order.sort_by(|&a, &b| plan[b].2.total_cmp(&plan[a].2).then(a.cmp(&b)));
    for index in order {
        let (bytes, count, _) = plan[index];
        let available = counts[&bytes];
        if count < available && used.saturating_add(bytes + 8) <= budget_bytes {
            plan[index].1 += 1;
            used += bytes + 8;
        }
    }
    plan.into_iter()
        .filter_map(|(bytes, count, _)| (count > 0).then_some((bytes, count)))
        .collect()
}

impl MoeSlotCache {
    /// Build the cache sizing N from free VRAM (MOE-SLRU-PLAN §B.4): probe free VRAM AFTER residents
    /// are loaded; N is shared across ALL layers so it must hold the WHOLE-MODEL hot set, not one
    /// layer's. The 35B-A3B keeps its 256 experts HOST-resident, so the GPU has ~20+ GB free at
    /// decode — empirically a 256-slot cache thrashes (~2-7% hit) while a few-thousand-slot cache
    /// reaches ~85%+ steady-state. So the DEFAULT auto-sizes N to fill `BW24_MOE_VRAM_FRAC` (default
    /// 0.85) of free VRAM, clamped to [256, ~hot-set]. `BW24_MOE_SLOTS` forces an exact N.
    pub fn new(e: &Engine, max_block_bytes: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let (free, _total) = e.ctx().mem_get_info()?;
        // Keep two blocks of slack after the machine-specific hard ceiling. The default remains
        // 80%; tightly provisioned spill rigs may raise it only after an OOM-gated local sweep.
        let hard_frac = cache_hard_vram_frac();
        let hard_bytes =
            ((free as f64 * hard_frac) as usize).saturating_sub(2 * (max_block_bytes + 8));
        let forced_slots = std::env::var("BW24_MOE_SLOTS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok());
        let requested_bytes = if let Some(n) = forced_slots {
            n.saturating_mul(max_block_bytes + 8)
        } else {
            // auto: fill BW24_MOE_VRAM_FRAC of free VRAM with slots (default 85%).
            // DEFAULT 0.85 (2026-07-06 local sweep: 0.40=25.0, 0.60=28.0, 0.85=28.5 tok/s on the
            // spill-regime 35B — hit-rate 87.8% -> 99.2%, PCIe 55 -> 3.8 MB/tok; the 0.80
            // hard-headroom cap below still bounds the true allocation, so 0.85 requests the max).
            // Rigs co-running other GPU work should set BW24_MOE_VRAM_FRAC lower.
            let frac = std::env::var("BW24_MOE_VRAM_FRAC")                .ok()
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(0.85);
            (free as f64 * frac) as usize
        };
        let budget_bytes = requested_bytes.min(hard_bytes);
        let layout = e.moe_cache_layout().unwrap_or_default();
        let size_aware = forced_slots.is_none()
            && std::env::var("BW24_MOE_SIZE_AWARE").as_deref() == Ok("1")
            && !layout.is_empty();
        let frequency_evict = std::env::var("BW24_MOE_LFU").as_deref() == Ok("1");
        let frequency_decay = if frequency_evict {
            cache_lfu_decay()
        } else {
            None
        };
        let mtp_frequency_weight = cache_lfu_mtp_weight();
        let mut class_plan = if size_aware {
            size_class_plan(&layout, budget_bytes)
        } else {
            Vec::new()
        };
        if class_plan.iter().map(|(_, count)| count).sum::<usize>() < 8 {
            let n = (budget_bytes / (max_block_bytes + 8)).max(8);
            class_plan = vec![(max_block_bytes, n)];
        }
        let n: usize = class_plan.iter().map(|(_, count)| count).sum();

        let mut slots = Vec::with_capacity(n);
        let mut slot_class = Vec::with_capacity(n);
        let mut classes = Vec::with_capacity(class_plan.len());
        let mut occupant = Vec::with_capacity(n);
        for (class_index, &(capacity, count)) in class_plan.iter().enumerate() {
            let start = slots.len();
            for _ in 0..count {
                // +8 tail pad: wide expert dots may issue an aligned read past the final block.
                slots.push(e.alloc_u8(capacity + 8)?);
                slot_class.push(class_index);
                occupant.push(None);
            }
            let free_slots = (start..start + count).rev().collect();
            classes.push(SlotClass {
                capacity,
                probation: VecDeque::new(),
                protected: VecDeque::new(),
                free: free_slots,
                protected_cap: ((count as f64 * 0.8) as usize).max(1),
            });
        }
        if size_aware {
            let allocated: usize = class_plan
                .iter()
                .map(|(bytes, count)| (bytes + 8) * count)
                .sum();
            eprintln!(
                "[moe-cache] size-aware fixed slots: {n} slots in {} classes, {:.2} GB / {:.2} GB budget",
                class_plan.len(),
                allocated as f64 / 1e9,
                budget_bytes as f64 / 1e9
            );
        }
        let pread_mode = crate::spill_pread::configured_mode();
        let pread_requested = pread_mode != SpillIoMode::Mmap;
        let pread = if pread_requested {

            match PreadPool::try_new(e, max_block_bytes, pread_mode) {
                Ok(pool) => Some(pool),
                Err(err) => {
                    eprintln!("[spill-pread] pinned-buffer initialization failed ({err}); using mmap");
                    None
                }
            }
        } else { None };


        Ok(MoeSlotCache {
            slots,
            slot_class,
            classes,
            occupant,
            table: HashMap::with_capacity(n * 2),
            frequencies: HashMap::with_capacity(layout.len().max(n * 2)),
            pending: HashMap::new(),
            inflight_sources: Vec::new(),
            quarantined_sources: Vec::new(),
 compute_sources: HashMap::new(),
            pread, worker_reads: HashMap::new(), pread_requested, pread_fallbacks: 0,
            copy_stream: e.copy_stream.clone(), copy_stream_unknown: false,
                        compute_stream: e.stream().clone(),
            compute_stream_unknown: false,
            n,
            max_block_bytes,
            size_aware,
            frequency_evict,
            frequency_decay,
            mtp_frequency_weight,
            last_forward_layer: None,
            last_forward_t: 0,
            frozen: false,
            per_layer: HashMap::new(),
            dev_rows: HashMap::new(),
            prewarm_tried: HashSet::new(),
            hits: 0, misses: 0, staged_bytes: 0,
        })
    }

    #[inline]
    pub fn n_slots(&self) -> usize {         self.n
    }
    #[inline]
    pub fn is_frozen(&self) -> bool {
        self.frozen
    }
    pub fn freeze(&mut self) {
        if !self.frozen {
            self.frozen = true;
            let (_, complete, one_projection, two_projections, stranded_blocks) =
                self.expert_residency_shape();
            eprintln!(
                "[moe-cache] residency frozen: {} slots, {} resident blocks; \
                 {complete} complete experts, {one_projection} one-projection fragments, \
                 {two_projections} two-projection fragments ({stranded_blocks} stranded blocks)",
                self.n,
                self.table.len()
            );
            let mut mtp_masks = HashMap::<u16, u8>::new();
            for id in self.table.keys().filter(|id| id.layer == u16::MAX) {
                *mtp_masks.entry(id.ex).or_insert(0) |= 1u8 << id.proj;
            }
            if !mtp_masks.is_empty() {
                let complete = mtp_masks.values().filter(|&&mask| mask == 0b111).count();
                eprintln!(
                    "[moe-cache] frozen MTP residency: {} blocks, {complete} complete experts",
                    mtp_masks.values().map(|mask| mask.count_ones() as usize).sum::<usize>()
                );
            }
        }
    }

    pub(crate) fn expert_residency_shape(&self) -> (usize, usize, usize, usize, usize) {
        let mut masks = HashMap::<(u16, u16), u8>::new();
        for id in self.table.keys() {
            *masks.entry((id.layer, id.ex)).or_insert(0) |= 1u8 << id.proj;
        }
        let complete = masks.values().filter(|&&mask| mask == 0b111).count();
        let one_projection = masks
            .values()
            .filter(|&&mask| mask.count_ones() == 1)
            .count();
        let two_projections = masks
            .values()
            .filter(|&&mask| mask.count_ones() == 2)
            .count();
        let stranded_blocks = one_projection + 2 * two_projections;
        (
            masks.len(),
            complete,
            one_projection,
            two_projections,
            stranded_blocks,
        )
    }

    #[inline]
    pub fn max_block_bytes(&self) -> usize {
        self.max_block_bytes
    }


    /// O(1) residency check (the ktransformers `generate_gpu_experts_masks` analog).
    #[inline]
    pub fn resident(&self, id: BlockId) -> Option<usize> { self.table.get(&id).copied() }

    #[inline]
    fn frequency_increment(&self, id: BlockId) -> f32 {
        if id.layer == u16::MAX { self.mtp_frequency_weight } else { 1.0 }
    }

    /// Record a routed block that a fused all-hit path consumed without going through dispatch.
    /// Warmup-only callers use this to make the LFU profile reflect actual grouped GPU traffic;
    /// frozen serving skips it because residency can no longer change.
    pub(crate) fn note_profile_hit(&mut self, id: BlockId) {
        if self.frozen || !self.table.contains_key(&id) {
            return;
        }
        let increment = self.frequency_increment(id);
        *self.frequencies.entry(id).or_insert(0.0) += increment;
    }

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
        let class_index = self.slot_class[slot];
        let class = &mut self.classes[class_index];
        if !class.free.is_empty() {
            return;
        }
        if let Some(pos) = class.probation.iter().position(|&x| x == slot) {
            class.probation.remove(pos);
            self.push_protected(slot);
        } else if let Some(pos) = class.protected.iter().position(|&x| x == slot) {
            class.protected.remove(pos);
            class.protected.push_back(slot); // MRU
        } else {
            // not in either segment (shouldn't happen for a resident slot) — treat as protected MRU
            self.push_protected(slot);

        }
    }


    /// Push a slot to protected MRU; if protected exceeds its cap, demote its LRU front to probation.
    fn push_protected(&mut self, slot: usize) {
        let class = &mut self.classes[self.slot_class[slot]];
        class.protected.push_back(slot);
        while class.protected.len() > class.protected_cap {
            if let Some(demoted) = class.protected.pop_front() {
                class.probation.push_back(demoted);
            } else {
                break;
            }
        }
    }

    fn remove_occupant(&mut self, slot: usize) {
        if let Some(old) = self.occupant[slot].take() {
            self.table.remove(&old);
            self.on_block_evicted(old.layer);
        }
    }

    /// Lowest cumulative-frequency resident in one class; ties keep ordinary LRU order. A cold
    /// admission therefore becomes the sacrificial slot on the next miss instead of displacing a
    /// prompt-proven hot expert. `keep` protects the expert whose kernels are currently queued.
    fn frequency_victim_in_class(&mut self, class_index: usize, keep: &[BlockId]) -> Option<usize> {
        let class = &self.classes[class_index];
        let probation_len = class.probation.len();
        let candidate = class
            .probation
            .iter()
            .chain(class.protected.iter())
            .enumerate()
            .filter_map(|(position, &slot)| {
                let id = self.occupant[slot]?;
                (!keep.contains(&id)).then_some((
                    self.frequencies.get(&id).copied().unwrap_or(0.0),
                    position,
                    slot,
                ))
            })
            .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let (_, position, slot) = candidate?;
        if position < probation_len {
            self.classes[class_index].probation.remove(position);
        } else {
            self.classes[class_index]
                .protected
                .remove(position - probation_len);
        }
        Some(slot)
    }

    /// Pick the LRU victim from the smallest class that can hold `required` bytes.
    fn evict_one(&mut self, required: usize) -> Option<usize> {
        for class_index in 0..self.classes.len() {
            if self.classes[class_index].capacity < required {
                continue;
            }
            let slot = if self.frequency_evict {
                self.frequency_victim_in_class(class_index, &[])
            } else {
                self.classes[class_index]
                    .probation
                    .pop_front()
                    .or_else(|| self.classes[class_index].protected.pop_front())
            };
            if let Some(slot) = slot {
                self.remove_occupant(slot);
                return Some(slot);
            }
        }
        None
    }

    /// Pick a resident victim that is not needed by the expert currently being computed. Pending
    /// slots never enter the SLRU queues, so they are excluded automatically. Returns `None` rather
    /// than evicting a protected block; the caller then leaves this block to the synchronous path.
    fn evict_one_excluding(&mut self, required: usize, keep: &[BlockId]) -> Option<usize> {
        let take = |q: &mut VecDeque<usize>, occupant: &[Option<BlockId>]| {
            q.iter()
                .position(|&s| occupant[s].is_some_and(|id| !keep.contains(&id)))
                .and_then(|pos| q.remove(pos))
        };
        for class_index in 0..self.classes.len() {
            if self.classes[class_index].capacity < required {
                continue;
            }
            let slot = if self.frequency_evict {
                self.frequency_victim_in_class(class_index, keep)
            } else {
                take(&mut self.classes[class_index].probation, &self.occupant)
                    .or_else(|| take(&mut self.classes[class_index].protected, &self.occupant))
            };
            if let Some(slot) = slot {
                self.remove_occupant(slot);
                return Some(slot);
            }
        }
        None
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

    fn reserve_slot(&mut self, required: usize) -> Option<usize> {
        for class in &mut self.classes {
            if class.capacity >= required {
                if let Some(slot) = class.free.pop() {
                    return Some(slot);
                }
            }
        }
        self.evict_one(required)
    }

    fn release_reserved_slot(&mut self, slot: usize) {
        debug_assert!(self.occupant[slot].is_none());
        self.classes[self.slot_class[slot]].free.push(slot);
    }

    fn publish(&mut self, id: BlockId, slot: usize) {
        self.occupant[slot] = Some(id);
        self.table.insert(id, slot);
        self.classes[self.slot_class[slot]]
            .probation
            .push_back(slot);
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
        let slot = self.reserve_slot(host_bytes.len()).ok_or_else(|| {
            std::io::Error::other(format!(
                "no MoE cache slot can hold {} bytes (max class {})",
                host_bytes.len(),
                self.classes.last().map(|class| class.capacity).unwrap_or(0)
            ))
        })?;
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

    /// Start one MoE forward's worker-I/O scope. Any ticket left by an earlier error/early return is
    /// no longer a valid lookahead target; cancel it before this scope submits its own known-next
    /// reads. In-flight CPU reads keep their buffers until completion restores them safely.
    pub(crate) fn begin_worker_scope(&mut self) {
        if self.worker_reads.is_empty() { return; }
                let tickets: Vec<_> = self
            .worker_reads
            .drain()
            .map(|(_, read)| read.ticket)
            .collect();
        if let Some(pool) = self.pread.as_mut().filter(|pool| pool.is_worker()) {
            for ticket in tickets {
 let _ = pool.cancel_worker(ticket); }
                }
    }

    /// Age cumulative LFU at decode-token boundaries. A batched prompt may touch one block many
    /// times before decode begins; treating those touches as permanent future-use votes poisons a
    /// spill cache. The first T=1 sweep starts a fresh frequency epoch while preserving populated
    /// GPU slots. Later decode sweeps exponentially age history so recent cross-token reuse can
    /// displace stale prompt-specific experts.
    ///
    /// MoE layers are visited in ascending order and the cache is model-global, so
    /// `layer <= previous_layer` marks a new model forward. This changes victim selection only;
    /// every hit and miss still feeds identical expert bytes to the same GPU kernel.
    pub(crate) fn begin_forward_epoch(&mut self, layer: u16, t: usize) {
        let Some(decay) = self.frequency_decay else {
            self.last_forward_layer = Some(layer);
            self.last_forward_t = t;
            return;
        };
        let new_sweep = self
            .last_forward_layer
            .is_some_and(|previous| layer <= previous);
        if new_sweep && t == 1 {
            if self.last_forward_t != 1 {
                self.frequencies.clear();
            } else {
                self.frequencies.retain(|_, score| {
                    *score *= decay;
                    *score >= 1.0e-3
                });
            }
        }
        self.last_forward_layer = Some(layer);
        self.last_forward_t = t;
    }

    /// Turn already-submitted disk reads into copy-stream GPU admissions at a host-routing
    /// boundary. The caller must invoke this only after the router's DtoH synchronization has
    /// completed all earlier-layer compute, and `keep` must contain every block selected in the
    /// current layer. A reserved victim is therefore neither in use nor about to be used, so its
    /// H2D can start immediately while the CPU workers finish later reads. Consumers still insert
    /// an explicit compute-stream wait through the ordinary `pending` dispatch path.
    pub(crate) fn promote_worker_reads_at_safe_boundary(
        &mut self,
        order: &[BlockId],
        keep: &[BlockId],
        e: &Engine,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        if !crate::spill_pread::copy_h2d_enabled() {
            return Ok(0);
        }
        let mut promoted = 0usize;
        for &id in order {
            if self.table.contains_key(&id) || self.pending.contains_key(&id) {
                if let Some(read) = self.worker_reads.remove(&id) {
                    if let Some(pool) = self.pread.as_mut() {
                        let _ = pool.cancel_worker(read.ticket);
                    }
                }
                continue;
            }
            let Some(read) = self.worker_reads.get(&id).copied() else {
                continue;
            };
            let Some(slot) = self.reserve_prefetch_slot(read.len, keep) else {
                continue;
            };
            self.worker_reads.remove(&id);

            let index = match self.pread.as_mut().unwrap().wait_worker(read.ticket) {
                Ok(index) => index,
                Err(err) => {
                    let _ = self.pread.as_mut().unwrap().cancel_worker(read.ticket);
                    self.release_reserved_slot(slot);
                    self.note_pread_fallback(err.as_ref());
                    continue;
                }
            };
            let ready = {
                let bytes = match self.pread.as_ref().unwrap().bytes(index, read.len) {
                    Ok(bytes) => bytes,
                    Err(err) => {
                        self.pread.as_mut().unwrap().abort_read(index);
                        self.release_reserved_slot(slot);
                        self.note_pread_fallback(err.as_ref());
                        continue;
                    }
                };
                stage_pread_prefetch_on_copy_stream(e, bytes, &mut self.slots[slot])
            };
            let ready = match ready {
                Ok(ready) => ready,
                Err((err, reusable)) => {
                    if reusable {
                        self.pread.as_mut().unwrap().abort_read(index);
                        self.release_reserved_slot(slot);
                        self.note_pread_fallback(err.as_ref());
                        continue;
                    }
                    self.pread.as_mut().unwrap().mark_unknown_h2d(index);
                    self.copy_stream_unknown = true;
                    return Err(err);
                }
            };
            self.pread.as_mut().unwrap().mark_h2d(index, ready.clone());
            self.occupant[slot] = Some(id);
            self.pending.insert(
                id,
                PendingBlock {
                    slot,
                    ready,
                    keepalive: None,
                },
            );
            self.staged_bytes += read.len as u64;
            promoted += 1;
        }
        Ok(promoted)
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

        let pending = self.worker_reads.remove(&id);
                let pool = self.pread.as_mut().unwrap();
        let read = if pool.is_worker() {
            let ticket = match pending {
                Some(read) => Ok(Some(read.ticket)),
                None => pool.submit_worker(file.clone(), offset, len),
            };
            match ticket {

                Ok(Some(ticket)) => match pool.wait_worker(ticket) {
                    Ok(index) => Ok(index),
                    Err(err) => {
                        // Read errors normally release in wait_worker. A worker/channel failure may
                        // return earlier; cancel defensively so the next scope cannot lose the slot.
                        let _ = pool.cancel_worker(ticket);
                        Err(err)
                    }
                },
                Ok(None) => Err(std::io::Error::other("worker read ring is busy").into()),
                Err(err) => Err(err),
            }
        } else {
            debug_assert!(pending.is_none());
            pool.read(file.as_ref(), offset, len)
        };
        let index = match read {
            Ok(index) => index,
            Err(err) => {
                self.note_pread_fallback(err.as_ref());
                return Ok(DispatchSlot::Resident(self.admit(id, fallback, e)?));
            }
        };


        // The blocking read happens before eviction, so an I/O failure leaves cache residency
        // untouched and can safely use the mmap oracle.
        let slot = self.reserve_slot(len).ok_or_else(|| {
            std::io::Error::other(format!(
                "no MoE cache slot can hold {len} bytes (max class {})",
                self.classes.last().map(|class| class.capacity).unwrap_or(0)
            ))
        })?;
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
        let increment = self.frequency_increment(id);
        *self.frequencies.entry(id).or_insert(0.0) += increment;
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

    fn reserve_prefetch_slot(&mut self, required: usize, keep: &[BlockId]) -> Option<usize> {
        for class in &mut self.classes {
            if class.capacity >= required {
                if let Some(slot) = class.free.pop() {
                    return Some(slot);
                }
            }
        }
        self.evict_one_excluding(required, keep)
    }

    fn prefetch_bytes(
&mut self, id: BlockId, host_bytes: &[u8],
                      keepalive: Option<ExpertKeepalive>, keep: &[BlockId], e: &Engine)
                      -> Result<bool, Box<dyn std::error::Error>> {
        let Some(slot) = self.reserve_prefetch_slot(host_bytes.len(), keep) else { return Ok(false) };
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
        if self.table.contains_key(&id) || self.pending.contains_key(&id)
            || self.worker_reads.contains_key(&id) {
            return Ok(false);
        }
        match source {
            ExpertSource::Memory { bytes, keepalive } => {
                self.prefetch_bytes(id, bytes, keepalive, keep, e)
            }
            ExpertSource::Disk { file, offset, len, fallback, keepalive } => {
                if self.pread.as_ref().is_some_and(PreadPool::is_worker) {
                    match self.pread.as_mut().unwrap()
                        .submit_worker_speculative(file.clone(), offset, len) {
                        Ok(Some(ticket)) => {
                            self.worker_reads.insert(id, WorkerRead { ticket, len });
                            Ok(true)
                        }
                        Ok(None) => Ok(false),
                        Err(err) => {
                            self.note_pread_fallback(err.as_ref());
                            Ok(false)
                        }
                    }
                } else if self.pread.is_some() {
                    // Blocking `pread` remains demand-only so it cannot delay current compute.
                    Ok(false)
                } else {
                    self.prefetch_bytes(id, fallback, Some(keepalive), keep, e)
                }
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
        if self.size_aware {
            return Ok(());
        } // heterogeneous prewarm needs a per-class fit proof
        if self
            .classes
            .iter()
            .map(|class| class.free.len())
            .sum::<usize>()
            < missing
        {
            return Ok(()); // won't evict for a prewarm
        }
        for ex in 0..n_expert {
            for (proj, exps) in [
                (PROJ_GATE, &m.gate_exps),
 (PROJ_UP, &m.up_exps),
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

fn cache_lfu_decay() -> Option<f32> {
    let raw = std::env::var("BW24_MOE_LFU_DECAY").ok()?;
    match parse_cache_lfu_decay(Some(&raw)) {
        Ok(value) => value,
        Err(reason) => {
            eprintln!(
                "[moe-cache] invalid BW24_MOE_LFU_DECAY={raw:?} ({reason}); disabling LFU decay"
            );
            None
        }
    }
}

fn cache_lfu_mtp_weight() -> f32 {
    const DEFAULT: f32 = 1.0;
    let raw = std::env::var("BW24_MOE_LFU_MTP_WEIGHT").ok();
    match parse_cache_lfu_mtp_weight(raw.as_deref()) {
        Ok(value) => value,
        Err(reason) => {
            eprintln!(
                "[moe-cache] invalid BW24_MOE_LFU_MTP_WEIGHT={:?} ({reason}); using {DEFAULT}",
                raw.as_deref().unwrap_or("")
            );
            DEFAULT
        }
    }
}

fn parse_cache_lfu_mtp_weight(raw: Option<&str>) -> Result<f32, &'static str> {
    let value = raw
        .unwrap_or("1")
        .parse::<f32>()
        .map_err(|_| "expected a number")?;
    if value.is_finite() && (0.25..=64.0).contains(&value) {
        Ok(value)
    } else {
        Err("expected a finite multiplier from 0.25 through 64")
    }
}

fn parse_cache_lfu_decay(raw: Option<&str>) -> Result<Option<f32>, &'static str> {
    let Some(raw) = raw else { return Ok(None) };
    let value = raw.parse::<f32>().map_err(|_| "expected a number")?;
    if value.is_finite() && value > 0.0 && value <= 1.0 {
        Ok(Some(value))
    } else {
        Err("expected a finite fraction greater than 0 and at most 1")
    }
}

fn cache_hard_vram_frac() -> f64 {
    const DEFAULT: f64 = 0.80;
    let raw = std::env::var("BW24_MOE_HARD_VRAM_FRAC").ok();
    match parse_cache_hard_vram_frac(raw.as_deref()) {
        Ok(value) => value,
        Err(reason) => {
            eprintln!(
                "[moe-cache] invalid BW24_MOE_HARD_VRAM_FRAC={:?} ({reason}); using {DEFAULT}",
                raw.as_deref().unwrap_or("")
            );
            DEFAULT
        }
    }
}

fn parse_cache_hard_vram_frac(raw: Option<&str>) -> Result<f64, &'static str> {
    let value = raw
        .unwrap_or("0.80")
        .parse::<f64>()
        .map_err(|_| "expected a number")?;
    if value.is_finite() && (0.10..=0.95).contains(&value) {
        Ok(value)
    } else {
        Err("expected a finite fraction from 0.10 through 0.95")
    }
}

#[cfg(test)]
mod vram_fraction_tests {
    use super::{
        parse_cache_hard_vram_frac, parse_cache_lfu_decay, parse_cache_lfu_mtp_weight,
        size_class_plan,
    };

    #[test]
    fn hard_vram_fraction_defaults_and_rejects_unsafe_values() {
        assert_eq!(parse_cache_hard_vram_frac(None), Ok(0.80));
        assert_eq!(parse_cache_hard_vram_frac(Some("0.82")), Ok(0.82));
        assert!(parse_cache_hard_vram_frac(Some("NaN")).is_err());
        assert_eq!(parse_cache_hard_vram_frac(Some("0.95")), Ok(0.95));
        assert!(parse_cache_hard_vram_frac(Some("0.96")).is_err());
        assert!(parse_cache_hard_vram_frac(Some("1.0")).is_err());
        assert!(parse_cache_hard_vram_frac(Some("bad")).is_err());
    }

    #[test]
    fn lfu_decay_is_opt_in_and_bounded() {
        assert_eq!(parse_cache_lfu_decay(None), Ok(None));
        assert_eq!(parse_cache_lfu_decay(Some("0.8")), Ok(Some(0.8)));
        assert_eq!(parse_cache_lfu_decay(Some("1")), Ok(Some(1.0)));
        for value in ["0", "-0.1", "1.1", "NaN", "bad"] {
            assert!(
                parse_cache_lfu_decay(Some(value)).is_err(),
                "accepted {value}"
            );
        }
    }

    #[test]
    fn lfu_mtp_weight_defaults_and_is_bounded() {
        assert_eq!(parse_cache_lfu_mtp_weight(None), Ok(1.0));
        assert_eq!(parse_cache_lfu_mtp_weight(Some("4")), Ok(4.0));
        for value in ["0", "0.1", "65", "NaN", "bad"] {
            assert!(
                parse_cache_lfu_mtp_weight(Some(value)).is_err(),
                "accepted {value}"
            );
        }
    }

    #[test]
    fn size_class_plan_preserves_classes_and_never_exceeds_budget() {
        let blocks = [100usize, 100, 100, 200, 200, 400];
        let budget = (108 * 2) + 208 + 408;
        let plan = size_class_plan(&blocks, budget);
        assert!(plan.iter().all(|(_, count)| *count > 0));
        assert!(
            plan.iter()
                .map(|(bytes, count)| (bytes + 8) * count)
                .sum::<usize>()
                <= budget
        );
        assert!(plan.iter().all(|(bytes, count)| {
            *count <= blocks.iter().filter(|block| **block == *bytes).count()
        }));
    }

    #[test]
    fn size_class_plan_returns_full_inventory_when_it_fits() {
        let blocks = [100usize, 100, 200, 400];
        let budget: usize = blocks.iter().map(|bytes| bytes + 8).sum();
        assert_eq!(
            size_class_plan(&blocks, budget),
            vec![(100, 2), (200, 1), (400, 1)]
        );
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
