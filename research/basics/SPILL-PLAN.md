# Tiered Spilling — Build Plan

I have verified all the load-bearing references. Now I'll synthesize the plan.

# Concrete Plan: General Tiered Weight Spilling in bw24 (Beyond EDGE-1 Experts)

## 0. What exists today (the foundation to extend)

EDGE-1 already implements the *expert* case of tiered spilling, but only the hot path of one tier transition (host RAM -> VRAM scratch) with no cache, no prefetch, and no disk tier:

- `HostExps` (`crates/bw24-engine/src/model.rs:174-221`) holds one layer's stacked 256-expert quant bytes **host-resident** in a plain `Vec<u8>`; `expert_stride` (`model.rs:181`, `:207`) = `raw.len()/n_expert` (860160 gate/up Q6_K, 1114112 down Q8_0); `expert_bytes(e)` (`model.rs:217-220`) returns the contiguous DMA source slice for expert `e`.
- `Engine::stage_expert` (`crates/bw24-engine/src/lib.rs:101-106`) is the single H2D primitive: `memcpy_htod(host_bytes, scratch.slice_mut(off..off+len))` on the default stream, ordered before the consuming `qmatvec_view` (`lib.rs:112-119`) without an explicit event.
- `moe_ffn` (`crates/bw24-engine/src/hybrid_forward.rs:203-303`) allocates **one scratch slot per projection** (`scratch_g/u/d`, `hybrid_forward.rs:229-231`) and re-stages all 3 matrices of every routed expert **every token, with no cache** (`:261`, `:265`, `:274`). Router top-k runs on host after a `dtoh` (`:220-221`).

The gaps relative to the full design (`ARCHITECTURE.md:99-101`, `ROADMAP.md:56-57`): (1) no GPU slot cache, so warm experts re-DMA; (2) no pinned host buffer (`Vec<u8>` is pageable -> H2D goes through a driver bounce buffer, halving effective PCIe); (3) no disk tier, so the 35B-31GB case can't even load; (4) no prefetch; (5) tiers are hardcoded to expert tensors, not general weights.

This plan generalizes that machinery into a reusable `SpillTier` substrate and a `SlotCache`, then re-points `moe_ffn` at it.

---

## 1. Tier design: a `Tiered<T>` weight handle over VRAM / pinned-host / mmap-disk

Replace `HostExps`'s single pageable `Vec<u8>` with a 3-tier descriptor that any spillable weight (expert matrix **or** dense `wq/wk/wv/wo`, `ffn_gate/up/down`) can use. Define in a new `crates/bw24-engine/src/spill.rs`:

```
enum Residence { Vram { slot: u32 }, Pinned { off: usize }, Disk { file_off: u64 } }

struct SpillBlock {        // one matrix (one expert, or one dense weight)
    bytes_len: usize,      // = expert_stride for experts; row_bytes/qtype carried alongside
    qtype: i32, in_f: usize, out_f: usize, row_bytes: usize,  // lifted verbatim from HostExps fields (model.rs:176-181)
    res: Residence,
}

struct SpillTier {
    // Tier 1: ONE contiguous cudaHostAlloc pinned buffer (ARCHITECTURE.md:100)
    pinned: PinnedHostBuf,        // see §1.1
    pinned_cap: usize,            // sized at runtime from free-RAM query (§3)
    pinned_map: Vec<(BlockId, usize)>,   // which blocks live pinned + their offset
    // Tier 2: mmap of the GGUF region for cold blocks (ARCHITECTURE.md:108)
    mmap: Mmap,                   // MAP_SHARED, posix_fadvise SEQUENTIAL, NO MAP_POPULATE
    file_off: Vec<u64>,           // per-block byte offset into the file
}
```

**Tier 0 (24 GB VRAM):** the residents already on GPU (norms, embeddings via `EmbedHost`, output layer, all *dense* layer weights for fitting models, attention, router, shared expert — per `ARCHITECTURE.md:92` "keep attention + shared expert + dense FFN + router + KV resident") **plus** the `SlotCache` (§2) of N fixed-address staging buffers. This replaces the 3 throwaway scratch slots at `hybrid_forward.rs:229-231`.

**Tier 1 (pinned host RAM, ~10-12 GB):** one `cudaHostAlloc` buffer holding the hottest blocks that don't fit VRAM. Pinned (not pageable) is the key correctness/perf fix over today's `Vec<u8>` — `cudaMemcpyAsync` from pinned memory is true async DMA at full PCIe; from pageable it stalls on a bounce-copy. `expert_bytes(e)` (`model.rs:217`) becomes "pointer into the pinned buffer at `pinned_map[e]`".

**Tier 2 (mmap NVMe, ~14 GB/s RAID0 ceiling, treat as slowest):** the full GGUF stays mmap'd (we already do `mmap` in the loader per `ARCHITECTURE.md:108`). Cold blocks are read from the mapping. For *predicted-cold* prefetch, add an `io_uring` O_DIRECT reader to dodge page-cache thrash (`ARCHITECTURE.md:100`, `:163`); for demand misses, the plain mmap fault path is the fallback.

### 1.1 PinnedHostBuf — the missing primitive (extend `Engine`)

Today `Engine::alloc_u8` (`lib.rs:92-94`) only gives **device** memory; there is no pinned-host allocator. Add alongside it:

```
impl Engine {
    fn alloc_pinned(&self, n: usize) -> Result<PinnedHostBuf>   // wraps cuMemHostAlloc / cudarc HostSlice
    fn stage_block(&self, src: &SpillBlock, tier: &SpillTier, slot: &mut CudaSlice<u8>, off: usize, copy_stream: &CudaStream) -> Result<Event>
}
```

`stage_block` is the generalization of `stage_expert` (`lib.rs:101-106`): it resolves `src.res`, copies from pinned (fast) or faults in from mmap (slow) on the **copy stream**, and returns a CUDA event so the consumer can wait without serializing the whole default stream (today `stage_expert` relies on same-stream ordering, `lib.rs:98-100`).

---

## 2. SLRU hot-set: a fixed-address GPU slot cache

This is the "persistent GPU expert-slot cache (N≪n_expert fixed-address slots, SLRU + second-miss admission)" of `ARCHITECTURE.md:92`. Define in `spill.rs`:

```
struct SlotCache {
    slots: Vec<CudaSlice<u8>>,    // N fixed-address device buffers, each = max block bytes
    occupant: Vec<Option<BlockId>>,
    // SLRU: two segments — probationary + protected
    probation: VecDeque<u32>,     // recency order, slot indices
    protected: VecDeque<u32>,
    // second-miss admission: a block only enters cache on its 2nd miss within a window
    miss_ghost: HashSet<BlockId>, // "ghost" keys seen-once (no payload), drives admission
}
```

**Slot sizing/count (budget, §5):** each slot is `max(expert_stride)` = 1114112 B (down Q8_0). With Tier-0 headroom after residents+KV, a 35B-A3B at Q6_K can hold roughly N=8-16 slots per the design's "8-16 at ~860-1114 KB each" (`ARCHITECTURE.md:100` "per-layer LRU hot-expert slot cache + 4-8 async staging buffers"). N is computed at startup from the live free-VRAM query (§3), never hardcoded.

**Lookup/admission policy:**
1. On routed expert `ex` (the loop at `hybrid_forward.rs:258`): `cache.get(block_id(layer, proj, ex))`.
2. **Hit** -> use the resident slot directly as the `qmatvec_view` weight (`lib.rs:112`); zero PCIe. This is the entire win — "the ~15-20% hot experts stay resident -> per-token PCIe ≈ 0 after warmup" (`ARCHITECTURE.md:92`).
3. **Miss** -> stage from Tier 1/2 into a *staging* slot (not yet cached). **Second-miss admission:** if `block_id ∈ miss_ghost`, promote into the SLRU probationary segment (evicting probation LRU); else insert into `miss_ghost`. This prevents one-off cold experts from evicting genuinely hot ones (the classic SLRU scan-resistance property).
4. **Promotion:** a probationary slot hit -> move to protected segment. Protected eviction demotes to probation.

**Hot-expert signal beyond pure recency** (`ARCHITECTURE.md:92-93`, `:101`): bias admission by the router gate logit. We already compute the full softmax `probs[256]` on host (`hybrid_forward.rs:239-242`); pass the selected experts' renormalized weights `w[j]` (`hybrid_forward.rs:251-254`) as a warmth score, so a low-probability routed expert (barely top-8) does not displace a high-probability resident one. This is the "gate-logit warmness" signal; copy the priority structure from MoE-Infinity EAM / SGLang `eplb/expert_distribution.py` (`ARCHITECTURE.md:93`, `:101`).

Copy the LRU+ghost mechanics structurally from `dvmazur/mixtral-offloading` `expert_cache.py` and llama.cpp's `copy_experts` selective sub-row copy (`ARCHITECTURE.md:93`, `:101`).

---

## 3. Runtime elastic free-RAM / free-VRAM query (never hardcode)

`ARCHITECTURE.md:3` is explicit: free host RAM is "~12-16 GB free *right now*, varies with other LLM servers — query at runtime, never hardcode." Today nothing queries it; all sizes are static (`hybrid_forward.rs:226-228`). Add a `MemBudget` probed at startup and refreshed periodically:

```
struct MemBudget { free_vram: usize, free_pinnable_ram: usize }
impl MemBudget {
    fn probe() -> Self {
        // VRAM: cuMemGetInfo (free, total) via cudarc — authoritative, accounts for other GPU procs
        // RAM:  /proc/meminfo MemAvailable (NOT MemFree) — the kernel's own estimate of pageable-without-OOM;
        //       cap pinned request to a fraction (e.g. 60%) of MemAvailable so cudaHostAlloc can't trigger OOM
        //       or evict the page cache that mmap Tier 2 depends on.
    }
}
```

**Allocation order at load:** (1) `cuMemGetInfo` -> reserve residents + KV (`cache.rs:50-53` already pre-allocs `max_ctx*kv_dim`); (2) from remaining free VRAM, size `SlotCache` N = `floor(free_vram_after_residents / max_block_bytes)`, clamped to [4, 16]; (3) from `MemAvailable*0.6`, size `pinned_cap` and greedily place the hottest blocks (by static estimate first, then refined by observed gate stats); (4) everything else -> Tier 2 mmap. Refresh `MemBudget` every K tokens; if `free_pinnable_ram` drops (another LLM server started), demote coldest pinned blocks to disk and `cuMemFreeHost` the tail. This is the "write-back + write-through-selective (host RAM is only 12-16 GB free)" policy (`ARCHITECTURE.md:84`).

Open implementation choice (per the findings' open question): `MemAvailable` from `/proc/meminfo` is the safe RAM signal; `cuMemGetInfo` is the safe VRAM signal. Both are cheap syscalls; poll, don't trust a config file.

---

## 4. Prefetch-next-layer via a dedicated copy stream

`ARCHITECTURE.md:112`: "Streams: 1 compute + 1 H2D + 1 D2H/token-out; event-fork the copy stream for next-expert prefetch during matmul; PDL edges." Today there is one stream and `stage_expert` is synchronous-by-ordering (`lib.rs:98-100`).

**Add a second `CudaStream` (the copy stream) to `Engine`.** Then overlap the current expert's GEMM (compute stream) with the next expert's H2D (copy stream):

- **Within a token's expert loop** (`hybrid_forward.rs:258-282`): software-pipeline by one expert. While `qmatvec_view` for `sel[j]` runs on the compute stream, `stage_block` for `sel[j+1]` runs on the copy stream into a *different* staging slot (double-buffer the 4-8 staging buffers of `ARCHITECTURE.md:100`). Consumer waits on the stage event before its GEMM. This hides the ~30-120 µs host->VRAM copy (860 KB at ~7-31 GB/s) behind the GEMM.
- **Across layers (speculative next-layer-gate prefetch):** the router logits for layer L+1 cannot be known until layer L's output exists, so true cross-layer expert prefetch is speculative. Practical version: prefetch the **shared expert** (always routed, `hybrid_forward.rs:285-291`) and the dense weights of L+1 during L's attention/FFN compute (deterministic, no speculation). For routed experts, prefetch the **top-k by historical frequency** for that layer (the EAM priority table) as a warm-start while L+1's true gate is computed; copy the speculative-prefetch structure from `dvmazur/mixtral-offloading` (`ARCHITECTURE.md:101`).

Synchronization: `stage_block` returns an event (§1.1); the compute stream issues `cuStreamWaitEvent` before the dependent `qmatvec_view`. This is the "event-synced double-buffer + PDL edges" of `ARCHITECTURE.md:100`. Note the open question on PCIe re-clocking (`ARCHITECTURE.md:196`): the link idles at Gen1 and re-clocks to Gen5 x8 (~31 GB/s) under load — prefetch break-even must be measured at full power, so keep a small priming copy at decode start.

---

## 5. Spill-candidate selection (which weights spill, which stay)

The split is driven by access pattern and size, lifted directly from `ARCHITECTURE.md:92` ("keep attention + shared expert + dense FFN + router + KV resident; routed experts in pinned host RAM"):

**Always Tier-0 resident (never spill):**
- Norms (`attn_norm`, `ffn_norm` — tiny f32 RMS, `model.rs:75-81`), token embeddings (`EmbedHost`, `model.rs:84-106` — already host-gathered on demand), output/lm_head layer.
- Attention `wq/wk/wv/wo`, the router `gate_inp` (`hybrid_forward.rs:220`), the shared expert (`hybrid_forward.rs:285-291`) — all accessed *every token unconditionally*, so spilling them would pay PCIe on the critical path with a ~100% miss rate.
- KV cache (`cache.rs:50-53`).

**Spill candidates (Tier 1/2, demand-staged via SlotCache):**
- **Routed expert matrices** — the primary case. Of 256 experts, only 8 touched per token (`hybrid_forward.rs:248`), and the hot ~15-20% concentrate (`ARCHITECTURE.md:92`). High spill leverage: huge bytes (29.75 GB for 35B, `model.rs:165`), sparse access.
- **Dense layer matmul weights** *only when the model itself doesn't fit VRAM* — e.g. a dense 70B at Q4. These are accessed every token (100% hit needed), so they only spill if VRAM literally can't hold them; then the SlotCache becomes a per-layer streaming window (stage layer L's weights while computing L-1), accepting that dense decode degrades to PCIe-bound. The same `SpillBlock`/`stage_block` substrate handles this; the difference is just the access predictor (sequential layer order, fully predictable -> prefetch is 100% accurate, unlike expert routing).

Selection rule: `spill_priority = access_frequency_estimate / bytes`; reside the highest-priority blocks until VRAM budget (§3) is exhausted, demote the rest to Tier 1 then Tier 2 by the same metric.

---

## 6. The 35B-31GB case (worked end-to-end)

Target: qwen36-35B Q6_K, experts = ~31 GB (`ROADMAP.md:57`), against 24 GB VRAM and 12-16 GB free host RAM. **31 GB experts fit neither VRAM nor pinned RAM** — this is the case that *requires* all three tiers + disk.

Today bw24's 35B-A3B *validated* run (`ROADMAP.md:17`, commit `02af8fc`) keeps experts in pageable `Vec<u8>` host RAM and works only because IQ3_S/IQ4_XS shrink it to ~9-14.5 GB (`ROADMAP.md:21`, `:31`) that fits host RAM. At Q6_K = 31 GB it does **not** fit host RAM either, so:

1. **Load:** mmap the GGUF (Tier 2), `MAP_SHARED`, no `MAP_POPULATE` (`ARCHITECTURE.md:108`) — zero upfront copy. Probe `MemBudget` (§3): say 14 GB free RAM -> `pinned_cap ≈ 8 GB` (60% of MemAvailable), free VRAM after residents+KV -> N=12 slots.
2. **Warm-up (first ~tens of tokens):** every expert is a Tier-2 miss; `stage_block` faults pages from mmap (or io_uring O_DIRECT for predicted-cold) at ~7-14 GB/s. Second-miss admission (§2) populates the SlotCache; the EAM frequency table identifies the hot ~15-20%.
3. **Pinned-tier fill:** the hottest ~8 GB of the 31 GB (the experts that recur) get promoted into the pinned buffer; their subsequent VRAM staging is full-PCIe async (~31 GB/s) instead of disk-bound.
4. **Steady state:** SlotCache serves the per-token hot set from VRAM (≈0 PCIe); occasional cold experts stage from pinned (fast) or disk (slow). The design's claim is "steady-state hit rate ~98-100% leaves PCIe mostly idle" (`ARCHITECTURE.md:100`) — the *primary lever is the large resident SLRU cache*, not the io_uring machinery.
5. **Honest budget** (`ARCHITECTURE.md:100`, `:135`, `:189`): with only 12-16 GB free RAM, the 31 GB working set genuinely spills to ~7 GB/s SSD on cold misses; the realistic target is **~1.5-2× over llama.cpp `-ncmoe`** in this offload regime (`ARCHITECTURE.md:135`, `:190`), measured on-box — *not* the 6-25× that was downgraded. The 30B-A3B-Q4 *fits 24 GB and runs fully resident*, so spilling gives it zero benefit (`ARCHITECTURE.md:135`) — Edge-1's megakernel is that target's lever, not this path.

---

## 7. Implementation order (extends what's built, doesn't rewrite it)

1. **`PinnedHostBuf` + `alloc_pinned`** (`Engine`, `lib.rs` near `:92`). Swap `HostExps.bytes: Vec<u8>` (`model.rs:175`) for a pinned buffer. Pure win: makes the existing `stage_expert` (`lib.rs:101`) a true async DMA. Re-validate 35B-A3B argmax=1178 (`ROADMAP.md:17`) unchanged.
2. **`MemBudget::probe`** (§3) — `cuMemGetInfo` + `/proc/meminfo`. Wire sizing into `moe_ffn`'s currently-static scratch (`hybrid_forward.rs:226-231`).
3. **`SlotCache`** (§2, SLRU + second-miss) replacing the throwaway `scratch_g/u/d` and re-stage-every-token loop (`hybrid_forward.rs:258-282`). First with recency-only, then add gate-warmth admission. This is where the headline win lands.
4. **Copy stream + event-synced double-buffer** (§4): second `CudaStream` in `Engine`; `stage_block` returns an event; pipeline expert `j+1` staging under expert `j` GEMM.
5. **Tier 2 mmap + io_uring O_DIRECT reader** (§1) for blocks exceeding pinned capacity — unlocks the 35B-31GB case (§6).
6. **Generalize to dense weights** (§5) via the same `SpillBlock` for not-fit dense models; sequential prefetch.

Validation gate throughout: every step must preserve the existing oracle match (argmax==llama.cpp, `ROADMAP.md:17`) before/after — spilling is a memory-placement change, never a numerics change, so the `qmatvec_view` correctness path (`lib.rs:110-111`, the validated dequant gate) is untouched.

---

## Key file:line anchors

- `crates/bw24-engine/src/model.rs:174-221` — `HostExps` (host-resident bytes, `expert_stride`, `expert_bytes`); swap `Vec<u8>`->pinned (step 1).
- `crates/bw24-engine/src/lib.rs:92-94` — `alloc_u8` (add `alloc_pinned` beside it).
- `crates/bw24-engine/src/lib.rs:101-106` — `stage_expert` (generalize to `stage_block` w/ copy stream + event).
- `crates/bw24-engine/src/lib.rs:112-119` — `qmatvec_view` (the validated consumer; reads any slot/scratch, unchanged).
- `crates/bw24-engine/src/hybrid_forward.rs:226-231` — static scratch slots -> `SlotCache`.
- `crates/bw24-engine/src/hybrid_forward.rs:239-254` — host softmax/top-k/weights -> feed gate-warmth into admission.
- `crates/bw24-engine/src/hybrid_forward.rs:258-282` — per-token re-stage loop -> cache lookup + prefetch pipeline.
- `crates/bw24-engine/src/cache.rs:50-53` — KV residents (Tier-0 reservation in budget math).
- `ARCHITECTURE.md:3` — elastic-RAM-query-at-runtime mandate; `:84` write-back policy; `:92-93` SLRU/EAM offload executor + copy-from list; `:99-101` 3-tier spilling tiers + copy-from; `:108` mmap loader; `:112` streams/graphs/prefetch; `:135`/`:189-190` honest 1.5-2× target; `:196`/`:198` open measurement questions.
- `ROADMAP.md:17` validated 35B-A3B argmax=1178; `:56-57` the spilling work item (qwen36-35B Q6_K = 31 GB spills).
