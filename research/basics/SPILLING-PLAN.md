# SPILLING-PLAN — Full Tiered Spilling (VRAM ↔ pinned-host ↔ NVMe disk)

bw24 capability **"spilling"**: a model whose weights exceed VRAM **and** exceed free host RAM
must still load and run **argmax-correct** by faulting cold weight blocks from an NVMe disk tier.
Today bw24 has the *VRAM↔pinned-host* leg (EDGE-1: MoE-SLRU GPU slot cache + host-resident pinned
expert store) but **no disk tier**, so any model whose host store doesn't fit RAM cannot even load.

This plan: generalize the host store behind the existing `expert_bytes()` seam into a
**`Tiered` backend** over `{Vram-slot, Pinned-host, Mmap-disk}`, add a **runtime free-mem query**
(VRAM + host RAM), wire **async prefetch** through the already-present copy stream, and add the
**store-before-evict barrier** so async staging can never corrupt a re-used slot.

> Honesty up front (§7): **the daily set does not need this.** Qwen3.5-9B and Qwen3.6-27B fit 24 GB
> resident-quant and never trigger spill. The disk tier exists only for `>24 GB` models that also
> overflow host RAM (the 35B-A3B Q6_K ≈ 31 GB case, and dense 70B). The GATE is a `>24 GB` model
> loading + running correct via disk **while the daily set takes the unchanged in-RAM path.**

---

## 0. What exists today (verified — this is the foundation, not a rewrite)

The current code is already past the stale `SPILL-PLAN.md` draft. Verified:

- **VRAM↔pinned-host leg is built and validated.** `HostExps` holds one layer's stacked 256-expert
  quant bytes host-resident, in **pinned** `cudaHostAlloc` memory under `BW24_MOE_CACHE`/`BW24_MOE_PINNED`
  (`crates/bw24-engine/src/model.rs:203-229` `HostBuf::{Paged,Pinned}`, `:283-294` load chooses pinned).
  `expert_bytes(e)` (`model.rs:297-302`) is the **single seam** returning the contiguous DMA source
  slice for expert `e` — `&self.bytes.as_bytes()[e*stride..(e+1)*stride]`.
- **The GPU slot cache (MoE-SLRU) is built.** `MoeSlotCache` (`crates/bw24-engine/src/moe_cache.rs:46-238`):
  N fixed-address GPU slots (`:47`), `BlockId→slot` table (`:49`), SLRU probation/protected
  (`:50-51`), second-miss ghost admission (`:53`, `:196-204`), double-buffered transient staging
  (`:56-57`), counters (`:64-66`). `new()` sizes N from a **free-VRAM query** at build time
  (`:76-87`, `e.ctx().mem_get_info()`).
- **Copy-stream + event infra is in place** but **not wired into the hot loop**: `Engine.copy_stream`
  (`lib.rs:50-51`), `stage_expert_async` issues on copy stream + returns an event (`lib.rs:241-246`),
  `compute_wait` makes the compute stream wait (`lib.rs:248-252`). The MoE loop explicitly leaves
  prefetch as TODO (`hybrid_forward.rs:291-300`).
- **Single H2D primitive:** `stage_expert` (`lib.rs:212-217`) = `memcpy_htod` on the **default**
  stream, ordered-before its `qmatvec_view` consumer (`lib.rs:258-269`) with no explicit event.
- **The cache dispatch path:** `moe_cached_gemm` (`hybrid_forward.rs:446-462`) →
  `cache.dispatch(id, host_bytes, eng)` (`moe_cache.rs:188-205`) → `qmatvec_view` from the resolved
  slot. **Bit-identity** (`moe_cache.rs:10-12`): HIT and MISS feed `qmatvec_view` the *same bytes*;
  only the `memcpy_htod` differs.

**The gaps that block `>24 GB` models** (this plan closes them):
1. `HostExps` holds **all** experts host-resident. For 35B Q6_K ≈ 31 GB > free host RAM → load OOMs.
   There is **no disk tier**: `HostBuf` is only `Paged(Vec)` or `Pinned(cudaHostAlloc)` (`model.rs:203-208`).
2. **No host-RAM budget query.** Only VRAM is probed (`moe_cache.rs:77`). Host RAM is never queried,
   so the loader cannot decide what stays pinned vs. what spills to disk.
3. **No store-before-evict barrier** for async staging — fine today (Stage-1 is same-stream,
   `lib.rs:209-211`), but a prerequisite once prefetch/disk faults run on the copy stream.
4. Tiers are hardcoded to **expert** tensors; no general `SpillBlock` for dense weights.

---

## 1. The disk tier: `HostBuf::Mmap` behind the existing `expert_bytes()` seam

The whole point of `expert_bytes()` (`model.rs:297-302`) is that callers (`moe_cached_gemm`
`hybrid_forward.rs:453`, Stage-1 `:326/:330/:337`) only ever see `&[u8]`. Add a third `HostBuf`
variant whose `as_bytes()` resolves into an **mmap'd region of the GGUF file** instead of a heap
allocation. This is the minimal change that unlocks the disk tier — the GEMM path is untouched.

```rust
// crates/bw24-engine/src/model.rs : extend the existing enum at :203
pub enum HostBuf {
    Paged(Vec<u8>),
    Pinned { slice: PinnedHostSlice<u8>, base: *const u8, len: usize },   // existing
    Mmap   { map: memmap2::Mmap, off: usize, len: usize },                // NEW (Tier 2)
}
impl HostBuf {
    pub fn as_bytes(&self) -> &[u8] {                 // extend the match at :216
        match self {
            HostBuf::Paged(v) => v.as_slice(),
            HostBuf::Pinned { base, len, .. } => unsafe { std::slice::from_raw_parts(*base, *len) },
            HostBuf::Mmap { map, off, len } => &map[*off .. *off + *len],   // page-faults on read
        }
    }
}
```

- `memmap2::Mmap` over the GGUF file, **`MAP_SHARED`, no `MAP_POPULATE`** (zero upfront copy).
  `posix_fadvise(POSIX_FADV_RANDOM)` on the fd (expert access is random, not sequential).
- `expert_bytes(e)` is **unchanged** — slicing an mmap region is the same `&[u8]`. On a cold expert
  the first `memcpy_htod` of those bytes triggers the kernel page-fault → NVMe read → DMA. This is
  the demand-fault disk path; correctness is automatic because the bytes are bit-identical to the
  GGUF on disk (which is what `Paged`/`Pinned` copied from in the first place — `model.rs:287/292`).
- **`io_uring` O_DIRECT prefetch (deferred optimization, §4):** for *predicted-cold* blocks, an
  explicit `O_DIRECT` read into a pinned bounce buffer dodges page-cache thrash. The demand mmap
  fault is the correctness fallback and ships first; io_uring is a latency optimization layered on
  the same `SpillBlock` (§5), not a correctness requirement for the GATE.

### 1.1 Three-tier residence chosen at **load** per block

`HostExps::load` (`model.rs:256-295`) currently makes a binary pinned/paged choice (`:283-284`).
Generalize to a **per-block tier decision** driven by `MemBudget` (§2). Add a `tier` field to track
where each expert's bytes live:

```rust
pub struct HostExps {
    pub bytes: HostBuf,        // ONE backing store for the whole stacked tensor (today)
    // NEW: when the layer is larger than the pinned budget, split per-expert:
    pub tiers: Vec<HostBuf>,   // Some => per-expert backing (Pinned hot / Mmap cold); else use `bytes`
    ...                        // qtype/in_f/out_f/n_expert/row_bytes/expert_stride unchanged (:244-249)
}
```

Tier assignment (greedy, by §2 budget): the hottest experts (warm-start from a static prior, then
observed gate frequency) get `Pinned`; the rest get `Mmap` into the GGUF. `expert_bytes(e)` resolves
`tiers[e]` if present, else slices `bytes` — one extra match arm, hot path unchanged.

**Tier 0 (VRAM, 24 GB):** residents (norms/embeddings/output, attention `wq/wk/wv/wo`, router
`gate_inp`, shared expert, dense FFN, KV cache `cache.rs:54-93`) **+** the `MoeSlotCache` N slots.
**Tier 1 (pinned host RAM):** hottest experts in `cudaHostAlloc` — true async DMA at full PCIe.
**Tier 2 (mmap NVMe):** cold experts faulted from the GGUF; demand-fault (slow) or io_uring-prefetched.

---

## 2. Runtime free-mem query — `MemBudget` (never hardcode)

Today only VRAM is queried (`moe_cache.rs:77`). Add a host-RAM query so the loader can split experts
between pinned (Tier 1) and disk (Tier 2). New `crates/bw24-engine/src/spill.rs`:

```rust
pub struct MemBudget { pub free_vram: usize, pub free_pinnable_ram: usize }
impl MemBudget {
    pub fn probe(e: &Engine) -> Result<Self, Box<dyn std::error::Error>> {
        let (free_vram, _) = e.ctx().mem_get_info()?;          // authoritative; accounts for other GPU procs
        let avail = read_meminfo_kb("MemAvailable")? * 1024;   // /proc/meminfo MemAvailable (NOT MemFree)
        // cap pinned to 60% of MemAvailable: cudaHostAlloc must not OOM nor evict the page cache
        // that the Tier-2 mmap depends on.
        Ok(MemBudget { free_vram, free_pinnable_ram: (avail as f64 * 0.60) as usize })
    }
}
```

- **VRAM:** `cuMemGetInfo` via `e.ctx().mem_get_info()` (already used at `moe_cache.rs:77`).
- **Host RAM:** parse `/proc/meminfo` `MemAvailable` (the kernel's own pageable-without-OOM estimate),
  cap pinned to a fraction so `cudaHostAlloc` can't OOM or evict the mmap page cache.
  `ARCHITECTURE.md:3` mandate: free host RAM is "~12-16 GB free *right now*, varies with other LLM
  servers — query at runtime, never hardcode."

**Load-time allocation order** (in `HybridModel::load`, before decode):
1. `MemBudget::probe` → `free_vram`, `free_pinnable_ram`.
2. Reserve residents + KV from `free_vram` (KV size from `cache.rs:63-66` per-token bytes × `max_ctx`).
3. Size `MoeSlotCache` N from remaining free VRAM (existing logic `moe_cache.rs:79-87`, already does this).
4. Compute total expert bytes = `Σ_layers Σ_proj expert_stride×n_expert`. If `≤ free_pinnable_ram`
   → **all pinned (current behavior, no disk).** Else → greedily pin the hottest experts up to
   `free_pinnable_ram`; everything else → `HostBuf::Mmap`. **This branch is the spill trigger.**
5. **Refresh** `MemBudget` every K decode tokens (deferred): if `free_pinnable_ram` drops (another
   LLM server started), demote the coldest pinned experts to `Mmap` and `cuMemFreeHost` the tail.

Env override (matches existing `BW24_MOE_*` convention `moe_cache.rs:75,80,84`): `BW24_SPILL_DISK=1`
forces disk tier on (testing the GATE on a model that *would* fit RAM), `BW24_SPILL_PINNED_FRAC`
overrides the 0.60 cap. Default = auto, never hardcoded.

---

## 3. Generalize `MoeSlotCache` → `Tiered` over `{Vram, Pinned, Mmap}`

The cache already abstracts the VRAM tier perfectly (`dispatch` returns `Resident`/`Staging`,
`moe_cache.rs:188-205`). The **disk tier slots underneath** the cache, not inside it: the cache's
job is "is this block in a GPU slot?"; `expert_bytes()`'s job is "where do the host bytes come from?"
So generalization is two orthogonal seams, both already present:

- **Above the cache (VRAM residency):** `MoeSlotCache` unchanged. HIT → resident GPU slot, 0 PCIe.
  MISS → stage from whatever `host_bytes` resolves to (`moe_cache.rs:198/202` call `stage_expert`,
  which calls `memcpy_htod` on `host_bytes`). The MISS source being pinned vs. mmap is **invisible**
  to the cache — it just sees `&[u8]`.
- **Below the cache (host residency):** `expert_bytes()` (`model.rs:297-302`) resolves the tier.
  Pinned → fast async DMA; Mmap → page-fault then DMA.

So the only cache-side change is **none for correctness**; the disk path is entirely in `HostBuf`.
Optionally add a `Tiered<T>` wrapper for the *naming* requested, but it is a thin alias:

```rust
// spill.rs — the requested generalization, structurally = HostExps + MoeSlotCache composed
pub struct Tiered {
    pub host: HostExps,        // Tier 1/2 backing (Pinned hot / Mmap cold), per-block
    pub slots: MoeSlotCache,   // Tier 0 GPU residency (existing)
}
```

`SpillBlock` for the dense-weight generalization (§5):

```rust
pub struct SpillBlock { pub bytes_len: usize, pub qtype: i32, pub in_f: usize, pub out_f: usize,
                        pub row_bytes: usize, pub host: HostBuf }   // fields lifted from HostExps :244-249
```

---

## 4. Async prefetch + store-before-evict barrier (correctness for the disk tier)

The disk tier makes a cold MISS cost ~1-5 ms (NVMe fault), vs. ~30 µs pinned. To hide it, wire the
**already-built** copy stream into the hot loop and add the eviction barrier.

**Prefetch (overlap stage[j+1] under GEMM[j]):** in the expert loop (`hybrid_forward.rs:309-323`),
while `qmatvec_view` for `sel[j]` runs on the compute stream, issue `stage_expert_async`
(`lib.rs:241-246`) for the MISS blocks of `sel[j+1]` on the copy stream into a *different* staging
slot (the cache already has 2: `moe_cache.rs:56-57`). Consumer calls `compute_wait(event)`
(`lib.rs:248-252`) before its GEMM. For the disk tier, the io_uring `O_DIRECT` read runs on a block
I/O thread into a pinned bounce buffer, then `stage_expert_async` does the H2D; the returned event
gates the GEMM.

**Store-before-evict barrier (the mandatory correctness rule once prefetch is on):** before a slot
is re-used on eviction (`MoeSlotCache::evict_one` `moe_cache.rs:144-154`, and `admit` reusing it
`:169-178`), the copy stream must be drained so an in-flight async H2D into that slot cannot race the
new occupant. Add to the eviction path:

```rust
// moe_cache.rs : in admit(), before staging into a reused slot
if slot_had_inflight_copy { eng.copy_stream.synchronize()?; }   // store-before-evict (vLLM/LMCache pattern)
```

This is a **no-op today** (Stage-1 is same-stream, ordered, `lib.rs:209-211`) and only activates once
prefetch issues copies on the copy stream — but it is wired now so the disk-tier prefetch is correct
from the first commit. Without it: an async fault-in to slot S races eviction returning S to the
free list → use-after-free → silent wrong tokens (breaks the argmax GATE).

**Speculative cross-layer prefetch (deferred, optional):** layer L+1's routed experts can't be known
until L's output exists. Deterministic prefetch that *is* safe: the **shared expert** (always routed,
`hybrid_forward.rs:348-354`) and dense weights of L+1. Routed experts: warm-start top-k by historical
gate frequency. This is an optimization on top of the GATE, not required for it.

---

## 5. Generalize to dense weights (the dense-70B case)

Routed experts are sparse (8 of 256 per token) so the SLRU cache wins big. **Dense** weights
(`wq/wk/wv/wo`, `ffn_gate/up/down`, `model.rs:160-168`) are touched **every token** — if they fit
VRAM they stay Tier-0 resident (never spill). They only spill when the model itself can't fit VRAM
(dense 70B Q4): then the same `SpillBlock` + `stage_block` substrate streams layer L's weights while
computing L-1. Access is **fully sequential** (layer order), so prefetch is 100% accurate — unlike
expert routing. The cache degenerates to a per-layer streaming window; decode becomes PCIe-bound but
**correct**. Same `HostBuf::Mmap` + copy-stream machinery; only the access predictor differs.

`spill_priority = access_frequency / bytes`: reside highest-priority blocks until VRAM (§2) exhausts,
demote the rest Tier1→Tier2 by the same metric. Dense weights have frequency=1 (every token) so they
sort to the top and never spill unless VRAM is literally too small.

---

## 6. The `>24 GB` case worked end-to-end (the GATE target)

Target: qwen36-35B-A3B Q6_K, experts ≈ 31 GB (`ROADMAP.md` task #8 / current-model-targets), vs.
24 GB VRAM and ~12-16 GB free host RAM. **31 GB experts fit neither VRAM nor pinned RAM** — requires
all three tiers. (The validated 35B-A3B run today only works because IQ3_S/IQ4_XS shrink experts to
~9-14.5 GB that *fits* host RAM; at Q6_K it does not, and the loader OOMs without the disk tier.)

1. **Load:** mmap the GGUF (`MAP_SHARED`, no `MAP_POPULATE`) → zero upfront copy. `MemBudget::probe`
   (§2): say 14 GB free RAM → pin ~8 GB of hottest experts, the other ~23 GB → `HostBuf::Mmap`.
   Free VRAM after residents+KV → `MoeSlotCache` N (existing sizing `moe_cache.rs:79-87`).
2. **Warm-up:** every expert is first a Tier-2 miss; `expert_bytes` faults from mmap, `memcpy_htod`
   DMAs to a staging slot, GEMM runs (correct from byte 0 — bit-identity `moe_cache.rs:10-12`).
   Second-miss admission (`moe_cache.rs:196-204`) fills the SLRU; gate frequency identifies the hot
   ~15-20% (`ARCHITECTURE.md:92`).
3. **Steady state:** SLRU serves the per-token hot set from VRAM (≈0 PCIe). Cold experts stage from
   pinned (fast) or disk (slow, prefetch-hidden where possible).
4. **Honest budget:** with only 12-16 GB free RAM, the 31 GB working set genuinely spills to ~7 GB/s
   SSD on cold misses. Realistic target **~1.5-2× over llama.cpp `-ncmoe`** in this offload regime
   (`ARCHITECTURE.md:135,189-190`), measured on-box — *not* the downgraded 6-25×. The **correctness**
   GATE (argmax-match) is the bar; throughput is a separate, honestly-modest claim.

---

## 7. Honest assessment — does the daily set need this? **No.**

| Model | Quant | Weights | Fits 24 GB resident? | Spill? |
|---|---|---|---|---|
| Qwen3.5-9B (P0 daily) | NVFP4 / Q8_0 | 5.3 / 8.9 GB | **Yes** | Never |
| Qwen3.6-27B (P0 daily) | NVFP4 / Q4_K_M | 5.6 / 15 GB | **Yes** | Never |
| Qwen3.6-35B-A3B (P1) | IQ3_S/IQ4_XS | 9-14.5 GB | host-RAM-resident | **pinned, no disk** |
| Qwen3.6-35B-A3B (P1) | Q6_K | ~31 GB | No (>RAM too) | **disk tier (this plan)** |
| dense 70B (future) | Q4 | ~40 GB | No | **disk tier, sequential** |

- The **daily models do not trigger spill** — they take the unchanged in-RAM path. The disk tier is a
  `>24 GB`-model capability, dormant for daily use (`ARCHITECTURE.md:134-135`: "the 30B *primary*
  target doesn't spill at all").
- Hybrid daily models also have a **shrunken KV footprint** (only 8/32 layers grow full-attention KV,
  the rest use fixed recurrent state, `cache.rs:73-91`), so 2-4 daily models coexist in 24 GB without
  ever touching disk.
- **Therefore the GATE has two halves:** (a) a `>24 GB` model loads + runs argmax-correct via the disk
  tier; (b) the daily set's path and numbers are **byte-identical** before/after (the disk variant is
  an additive `HostBuf` arm + a load-time branch, gated by `MemBudget`, off when experts fit RAM).

---

## 8. Implementation order (extends built code, never rewrites the validated path)

1. **`HostBuf::Mmap` arm** (`model.rs:203-229`) + `memmap2` dep + `posix_fadvise`. `expert_bytes`
   gains one match arm; `qmatvec_view`/`moe_cached_gemm` untouched.
2. **`MemBudget::probe`** (new `spill.rs`, §2) — `mem_get_info` + `/proc/meminfo MemAvailable`.
3. **Tiered load decision** in `HostExps::load`/`HybridModel::load` (§1.1, §2 step 4): pin hottest,
   mmap the rest, gated by budget. Default auto; `BW24_SPILL_DISK` to force for testing.
4. **Store-before-evict barrier** (`moe_cache.rs` admit/evict, §4) — wired now (no-op until prefetch).
5. **Copy-stream prefetch** in the expert loop (`hybrid_forward.rs:309-323`, §4) — pipeline j+1 under
   GEMM[j]; for disk, io_uring `O_DIRECT` bounce → `stage_expert_async`.
6. **Dense-weight generalization** (§5) via `SpillBlock` for not-fit dense models; sequential prefetch.

**Validation gate at every step (the GATE):**
- (a) **Correctness, daily:** Qwen3.5-9B / Qwen3.6-27B argmax-match vs. llama.cpp oracle, **unchanged**
  before/after each commit. Spilling is a memory-placement change, never a numerics change — the
  `qmatvec_view` dequant path (`lib.rs:254-269`) is untouched and bit-identity (`moe_cache.rs:10-12`)
  pins it.
- (b) **Correctness, spill:** a `>24 GB` model (35B-A3B Q6_K, or any model with `BW24_SPILL_DISK=1`)
  **loads** (no OOM) and produces the **same argmax sequence** with the disk tier as it would fully
  in-RAM — proving the mmap/pinned/VRAM tiers are byte-equivalent.
- (c) **No-spill-for-daily:** assert via `MemBudget` log that the daily set chooses the all-pinned
  (or all-resident) path — `HostBuf::Mmap` count = 0, disk faults = 0.

---

## Key file:line anchors

- `crates/bw24-engine/src/model.rs:203-229` — `HostBuf::{Paged,Pinned}` + `as_bytes`; **add `Mmap` arm** (§1).
- `crates/bw24-engine/src/model.rs:242-302` — `HostExps` + `load` (binary pinned/paged choice :283-284 → per-block tier) + `expert_bytes` (the seam, :297-302).
- `crates/bw24-engine/src/moe_cache.rs:46-238` — `MoeSlotCache` (VRAM tier, unchanged); `:77` VRAM query; `:144-178` evict/admit (**add store-before-evict barrier**, §4); `:188-205` dispatch; `:10-12` bit-identity.
- `crates/bw24-engine/src/lib.rs:50-51` — `copy_stream`; `:212-217` `stage_expert`; `:241-252` `stage_expert_async` + `compute_wait` (wire into prefetch, §4); `:254-269` `qmatvec_view` (validated consumer, unchanged).
- `crates/bw24-engine/src/hybrid_forward.rs:291-300` — prefetch TODO (this plan wires it); `:309-323` expert loop; `:446-462` `moe_cached_gemm`; `:386-399` `max_moe_block`.
- `crates/bw24-engine/src/cache.rs:54-93` — KV residents (Tier-0 reservation in `MemBudget` math, §2 step 2).
- new `crates/bw24-engine/src/spill.rs` — `MemBudget::probe` (§2), `SpillBlock`/`Tiered` (§3).
- `ARCHITECTURE.md:3` — query-RAM-at-runtime mandate; `:92` resident split + SLRU/EAM; `:99-101` 3-tier spilling; `:108` mmap loader; `:112` streams/prefetch; `:134-135,189-190` honest 1.5-2× target.
