//! SPILLING-PLAN: full tiered spilling (VRAM ↔ pinned-host ↔ mmap-disk).
//!
//! Today bw24 has the VRAM↔pinned-host leg (the `MoeSlotCache` GPU slot cache + the pinned
//! `HostExps` host store). This module adds the **third tier**: a `HostBuf::Mmap` arm (model.rs)
//! so cold experts are demand-faulted from the GGUF file on disk instead of held in RAM, plus the
//! runtime memory probe (`MemBudget`) that decides — per expert, at load — which tier each block
//! lives in. Never hardcode: VRAM is queried via `cuMemGetInfo`, host RAM via `/proc/meminfo`.
//!
//! THE GATE (SPILLING-PLAN §8): spilling is a memory-PLACEMENT change, never a numerics change. A
//! `Mmap` expert and a `Pinned` expert feed `qmatvec_view` byte-for-byte identical GGUF bytes — the
//! `Pinned`/`Paged` stores copied FROM exactly those on-disk bytes — so argmax is unchanged.
//!
//! The disk tier is gated behind `BW24_SPILL_DISK`. Unset (default) = the current all-host
//! behavior, byte-identical: `HostExps::tiers` stays `None` and every expert slices the single
//! pinned/paged backing store. The daily models (9B/27B) fit 24 GB and NEVER trigger spill.

use std::sync::Arc;
use memmap2::Mmap;
use crate::Engine;
use crate::model::HostBuf;

/// Runtime free-memory budget (SPILLING-PLAN §2). Both numbers are QUERIED at load, never
/// hardcoded — free host RAM "varies with other LLM servers", so the split between pinned (Tier 1)
/// and disk (Tier 2) must be decided against the live machine state.
#[derive(Clone, Copy, Debug)]
pub struct MemBudget {
    /// Free VRAM in bytes, from `cuMemGetInfo` (authoritative; accounts for other GPU processes).
    pub free_vram: usize,
    /// Bytes of host RAM safe to pin: `/proc/meminfo MemAvailable` × `pinned_frac` (default 0.60).
    /// Capped so `cudaHostAlloc` can neither OOM nor evict the page cache the Tier-2 mmap depends on.
    pub free_pinnable_ram: usize,
}

impl MemBudget {
    pub fn probe(e: &Engine) -> Result<Self, Box<dyn std::error::Error>> {
        let (free_vram, _total) = e.ctx().mem_get_info()?;     // same call moe_cache.rs:77 uses
        let avail = read_meminfo_kb("MemAvailable")? * 1024;   // MemAvailable (NOT MemFree)
        let frac = std::env::var("BW24_SPILL_PINNED_FRAC")
            .ok().and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.60);
        Ok(MemBudget {
            free_vram,
            free_pinnable_ram: (avail as f64 * frac) as usize,
        })
    }
}

/// Parse one `/proc/meminfo` field (a value in kB) by key, e.g. "MemAvailable".
fn read_meminfo_kb(key: &str) -> Result<usize, Box<dyn std::error::Error>> {
    let s = std::fs::read_to_string("/proc/meminfo")?;
    for line in s.lines() {
        // line form: "MemAvailable:   12345678 kB"
        if let Some(rest) = line.strip_prefix(key) {
            let rest = rest.trim_start_matches(':').trim();
            let kb: usize = rest.split_whitespace().next()
                .ok_or("malformed /proc/meminfo line")?
                .parse()?;
            return Ok(kb);
        }
    }
    Err(format!("/proc/meminfo: key {key} not found").into())
}

/// Is the disk tier (Tier 2) enabled? Gated behind `BW24_SPILL_DISK`. Default (unset) = off =>
/// the unchanged all-host path (`HostExps::tiers` stays `None`). Set to anything to force-on.
#[inline]
pub fn disk_tier_enabled() -> bool {
    std::env::var("BW24_SPILL_DISK").is_ok()
}

/// Shared load-time spill context (SPILLING-PLAN §2 step 4). Built ONCE per model load when the
/// disk tier is on, then handed by `&mut` to each `HostExps::load` so all layers/projections share
/// ONE file mmap and draw down a single running pinned-RAM budget. Greedy in load order: pin until
/// `pinned_remaining` is exhausted, then spill every later expert to `Mmap`.
pub struct SpillCtx {
    /// One `MAP_SHARED` mmap of the whole GGUF, shared (`Arc`) across every spilled expert block.
    pub file_map: Arc<Mmap>,
    /// Pinned-RAM budget still available (bytes); decremented as experts are pinned.
    pub pinned_remaining: usize,
    /// Diagnostics: how many experts landed pinned vs. mmap'd, and total disk-tier bytes.
    pub n_pinned: usize,
    pub n_mmap: usize,
    pub mmap_bytes: usize,
}

impl SpillCtx {
    /// Open a `MAP_SHARED` mmap of the GGUF and seed the pinned budget from a live `MemBudget` probe.
    /// The whole-map expert advice defaults to random (the historical behavior); setting
    /// `BW24_MOE_MMAP_ADVICE=normal` restores ordinary Linux readahead. SPILLING-PLAN §1.
    pub fn open(path: &std::path::Path, budget: &MemBudget) -> Result<Self, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(path)?;
        // MAP_SHARED, no MAP_POPULATE (memmap2's default Mmap::map): zero upfront copy, demand-fault.
        let map = unsafe { Mmap::map(&file)? };
        let _ = bw24_gguf::source::apply_expert_mmap_advice(&map);
        Ok(SpillCtx {
            file_map: Arc::new(map),
            pinned_remaining: budget.free_pinnable_ram,
            n_pinned: 0,
            n_mmap: 0,
            mmap_bytes: 0,
        })
    }
}

/// Build one expert's `HostBuf`, choosing its tier under the running budget (SPILLING-PLAN §1.1):
/// pin (Tier 1) while `pinned_remaining` covers the block, else `Mmap` it (Tier 2). `file_off` is
/// this expert's absolute byte offset within the GGUF file (= `data_start + tensor.offset + e*stride`).
/// Returns the chosen `HostBuf`; the bytes are bit-identical whichever tier is picked.
pub fn place_expert(
    ctx: &mut SpillCtx,
    e: &Engine,
    raw: &[u8],
    file_off: usize,
) -> Result<HostBuf, Box<dyn std::error::Error>> {
    let len = raw.len();
    if ctx.pinned_remaining >= len {
        // Tier 1: pinned host memory — true async DMA at full PCIe (matches the no-spill path).
        ctx.pinned_remaining -= len;
        ctx.n_pinned += 1;
        let mut p = unsafe { e.ctx().alloc_pinned::<u8>(len)? };
        { let dst = p.as_mut_slice()?; dst.copy_from_slice(raw); }
        let base = p.as_ptr()? as *const u8;
        Ok(HostBuf::Pinned { slice: p, base, len })
    } else {
        // Tier 2: mmap the GGUF region — demand-faulted from NVMe on first H2D. Zero RAM cost.
        ctx.n_mmap += 1;
        ctx.mmap_bytes += len;
        Ok(HostBuf::Mmap { map: ctx.file_map.clone(), off: file_off, len })
    }
}

/// SPILLING-PLAN §3/§5: a single spillable weight block over the same `{Pinned, Mmap}` substrate.
/// Lifted from the `HostExps` fields so dense weights (dense-70B case) can reuse the disk tier
/// without the 256-expert stacking. Carried for the requested generalization; the MoE path uses
/// `HostExps` directly (which now embeds the same tier machinery via `HostBuf`).
pub struct SpillBlock {
    pub host: HostBuf,
    pub qtype: i32,
    pub in_f: usize,
    pub out_f: usize,
    pub row_bytes: usize,
}

impl SpillBlock {
    /// The H2D DMA source for this block — resolves the tier (`Pinned` fast / `Mmap` demand-fault).
    #[inline]
    pub fn bytes(&self) -> &[u8] { self.host.as_bytes() }
}

/// SPILLING-PLAN §3: the requested `Tiered` generalization. Structurally it is the existing
/// `HostExps` (Tier 1/2 host backing, per-block) composed with the existing `MoeSlotCache`
/// (Tier 0 GPU residency). Both seams are already present and unchanged; this names the composition.
/// The MoE hot loop drives the two seams directly (`expert_bytes()` + `with_moe_cache`), so this is
/// a documentation/structural alias, not a new hot path.
pub struct Tiered {
    pub host: crate::model::HostExps,            // Tier 1/2 (Pinned hot / Mmap cold), per-expert
    pub slots: crate::moe_cache::MoeSlotCache,   // Tier 0 GPU residency (existing slot cache)
}
