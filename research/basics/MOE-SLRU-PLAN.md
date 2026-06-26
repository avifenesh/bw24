# MOE-SLRU-PLAN — EDGE-1 completion (`moe_slru`)

**Lever:** `moe_ffn` re-stages every routed expert every token over PCIe with no cross-token
cache. Add a fused-router kernel + an SLRU GPU expert-residency cache + async pinned-H2D
next-layer prefetch so per-token PCIe → ~0 after warmup at B=2–4.

**Target box:** sm_120 (RTX 5090 / Blackwell consumer), 2–4 concurrent agents, GGUF
resident-quant, Rust + raw CUDA via cudarc 0.19.8.

**Validation gate (authoritative, do not soften):** 35B-A3B prefill argmax stays **1178**
(`ROADMAP.md:17`, `run_gen.rs:64-66`); the cache-**hit** weight path is **bit-identical** to
stage-every-token; per-token PCIe → ~0 after warmup. Build-map row: `BW24-BUILD-MAP.md:155-158`.

---

## 0. Where we are today (Stage-1, exact code)

`HybridModel::moe_ffn` at `crates/bw24-engine/src/hybrid_forward.rs:247-347` is the whole MoE FFN.
The hot loop, per **token** `tok` (`:278`), per **routed expert** `ex` of 8 (`:302`):

```
e.stage_expert(m.gate_exps.expert_bytes(ex), &mut scratch_g, 0)?;   // H2D  860160 B  (:305)
let gate = e.qmatvec_view(&scratch_g, 0..g_len, &zt, 1, ...)?;       //       (:306)
e.stage_expert(m.up_exps.expert_bytes(ex), &mut scratch_u, 0)?;     // H2D  860160 B  (:309)
let up   = e.qmatvec_view(&scratch_u, 0..u_len, &zt, 1, ...)?;       //       (:310)
e.silu_mul(&gate, &up, &mut act, n_ff_exp)?;                         //       (:315)
e.stage_expert(m.down_exps.expert_bytes(ex), &mut scratch_d, 0)?;   // H2D 1114112 B  (:318)
let y    = e.qmatvec_view(&scratch_d, 0..d_len, &actv, 1, ...)?;     //       (:320)
e.axpy_into(&y, w[j], &mut dst, n_embd)?;                            //       (:325)
```

Per token the H2D bytes = `8 * (860160 + 860160 + 1114112)` = **22.67 MB** (gate+up Q6_K
`expert_stride=860160`, down Q8_0 `1114112`; `model.rs:203,227-230`). The router is host-side:
`logits = matmul(gate_inp, z)` on GPU (`:264`), then `dtoh` the whole `[T,256]` (`:265`), then a
per-token CPU softmax-over-256 + stable DESC top-8 sort + renorm (`:281-298`). Scratch is **three
fixed slots** (`scratch_g/u/d`, `:273-275`) sized to ONE expert, overwritten every expert, no
retention across tokens or layers. `HostExps.bytes` is a pageable `Vec<u8>` (`model.rs:197`).

Two independent costs to kill:
- **PCIe (this plan's headline):** 22.67 MB/token of redundant re-staging. The same ~15–20% of
  experts recur (the "hot expert" mass) → an SLRU residency cache makes the steady-state
  re-stage count → ~0.
- **Host routing latency:** the `dtoh` at `:265` + per-token CPU sort (`:291`). A fused-router
  kernel removes the round-trip and is the prerequisite for ever graphing the MoE WHILE-loop
  (`ARCHITECTURE.md:112`).

Decode (`decode.rs:79`) calls the SAME `moe_ffn` with `t=1`, so every change here lands on both
prefill and decode automatically.

---

## A. Fused-router kernel — replace host `dtoh`+softmax+sort

### A.1 What it replaces
`hybrid_forward.rs:264-298`: GPU `matmul` → `dtoh` `[T*256]` → host softmax → host stable
DESC argsort → top-8 → renorm. The argsort is `O(T·256·log256)` on one CPU thread and the `dtoh`
is a hard stream barrier (`e.dtoh` calls `synchronize()`, `lib.rs:327-331`).

### A.2 New kernel — `cu/moe_router.cu`, function `moe_router_topk_f32`
One CTA per token row (grid `(T,1,1)`, block `(256,1,1)` = one thread per expert). Reuses the
exact Stage-1 numerics so the result is bit-identical to the host loop:

```cuda
// in:  logits  [T, 256] f32 (router output, device, token-major)
// out: sel_idx [T, 8]   i32   (expert ids, DESC by prob, ascending-index tiebreak)
//      sel_w   [T, 8]   f32   (renormalized weights, F16-min clamp BEFORE divide)
extern "C" __global__ void moe_router_topk_f32(
    const float* __restrict__ logits, int* __restrict__ sel_idx,
    float* __restrict__ sel_w, int n_expert /*256*/, int n_used /*8*/)
```

Per CTA: (1) block-max reduce over 256 (matches `row.iter().fold(NEG_INF,max)`, `:282`);
(2) `expf(l-max)`, block-sum denom, `p[i]=exp/den` (`:285-286`); (3) **iterative argmax** —
`n_used` rounds of a block argmax that on each round picks the max `(prob, -idx)` pair
(`prob` primary DESC, index ascending tiebreak == `total_cmp(b,a).then(a.cmp(b))`, `:291`) and
masks the winner to `-INF` for the next round; (4) accumulate `ws += p[winner]`, after 8 rounds
`ws = max(ws, 6.103515625e-5f)` (F16 smallest normal, clamp **before** divide, `:297`), write
`sel_w[j] = p[sel[j]] / ws`.

**Bit-identity note:** the host path subtracts max then `f32::exp`; the device must use `expf`
(IEEE single). These differ in the last ULP, which can flip a tie in pathological cases. To
**guarantee** the argmax-1178 gate, the router kernel keeps `BW24_FUSED_ROUTER` OFF by default at
first (Stage-1 host router stays the oracle), and the gate is the bit-identity test in §D.2 (the
*selection indices and renormalized weights* must match, not the intermediate exp). If a tie
flips, fall back to a full-256 device sort (CUB `BlockRadixSort` on `(prob_bits, idx)` keys) which
reproduces the stable Rust sort exactly. Prefer iterative-argmax (cheaper, fits registers); escalate
to the radix sort only if the equality test fails.

### A.3 Rust seam — `lib.rs` near the other launchers (`:147`)
```rust
pub fn moe_router_topk(&self, logits: &CudaSlice<f32>, t: usize, n_expert: usize, n_used: usize)
    -> Result<(CudaSlice<i32>, CudaSlice<f32>), Box<dyn std::error::Error>>
```
Launch `(t,1,1)` × `(256,1,1)`. Add `"moe_router.cu"` to `build.rs` (the modified `build.rs` already
in the working tree compiles per-`.cu` fatbins; mirror the `QMATVEC_FATBIN` env pattern, `lib.rs:21`)
and a `router: Arc<CudaModule>` field on `Engine` (`lib.rs:36-43`) + a branch in `func()` (`lib.rs:58-65`).

**Output staging.** The dispatch loop still needs `sel_idx`/`sel_w` on the host (it indexes
`HostExps.bytes` on the CPU to choose the DMA source — §B). So we still `dtoh` two tiny buffers
`[T,8]` i32 + `[T,8]` f32 = **64 B/token** vs today's 1 KB/token, and the expensive sort moves to
the GPU. (Full elimination of this `dtoh` requires on-device residency lookup + on-device DMA
issue, which is the `cudaGraph` MoE-WHILE work in `ARCHITECTURE.md:112` and explicitly out of
scope here — see §F.) This is the honest scope: the fused router removes the **256-wide sort and
the 1 KB `dtoh`**, not the last 64 B.

---

## B. SLRU GPU expert-residency cache — N slots, second-miss admission, residency bitmask

### B.1 The data structure (`crates/bw24-engine/src/moe_cache.rs`, new file)
The cache is **layer-keyed and proj-keyed** because expert `ex` of layer L proj `gate` is a
different 860160-byte block than layer L proj `up`. Key = `BlockId(layer: u16, proj: u8, ex: u16)`.
All slots are sized to `max_block_bytes = 1114112` (down Q8_0, the largest) so any block fits any
slot — fixed-address, never re-allocated, never fragmented (`SPILL-PLAN.md:81`).

```rust
pub struct MoeSlotCache {
    slots: Vec<CudaSlice<u8>>,         // N fixed GPU buffers, each max_block_bytes (alloc_u8, lib.rs:160)
    occupant: Vec<Option<BlockId>>,    // slots[s] currently holds occupant[s]
    table: HashMap<BlockId, usize>,    // BlockId -> slot index (the O(1) residency lookup)
    probation: VecDeque<usize>,        // SLRU segment 1 (slot indices, LRU front = coldest)
    protected: VecDeque<usize>,        // SLRU segment 2 (hot; capped at ~0.8*N)
    free: Vec<usize>,                  // unused slot indices (startup)
    ghost: HashSet<BlockId>,           // SECOND-MISS admission: seen-once keys, no payload
    n: usize, protected_cap: usize, max_block_bytes: usize,
}
```

`occupant` IS the **residency bitmask** in the design's sense: "is `(layer,proj,ex)` resident?" is
`table.contains_key(&id)` in O(1) (the ktransformers `generate_gpu_experts_masks` analog,
`research/inference-maps/ktransformers.md:29-34`).

### B.2 Lookup / admission / eviction (SLRU + second-miss, scan-resistant)
```
fn lookup(&mut self, id: BlockId) -> Option<usize>      // returns resident slot, promotes on hit
fn admit(&mut self, id, host_bytes, stage) -> usize     // miss path: stage into a slot, return it
```
- **Hit** (`table[id] = s`): if `s ∈ probation`, move to `protected` (promotion); if already
  protected, bump to MRU. Return `s`. **Zero PCIe** — the entire win.
- **Miss**: **second-miss admission.** If `id ∉ ghost`: insert into `ghost`, **do NOT** admit —
  this token stages into a transient staging slot (§C double-buffer) and runs but is not retained.
  If `id ∈ ghost` (its second miss): admit. Eviction victim = `free.pop()` else `probation` LRU
  front; if probation empty, demote `protected` LRU → probation then evict that. This is the
  classic SLRU property that a one-off cold expert can never evict a genuinely hot one
  (`SPILL-PLAN.md:79,103-108`).
- **Warm-start admission prior (optional, kills the cold cliff):** seed `ghost` with the top-K
  experts per (layer,proj) by a static frequency table (ktransformers front-loading /
  `eplb/expert_distribution.py`, `ARCHITECTURE.md:92-93`). With the prior, a hot expert is admitted
  on its **first** miss instead of second, halving warmup misses. Gate behind a config flag;
  default off until the static table is measured for Qwen3.5-35B-A3B.

### B.3 Dispatch split (the one-change seam in `moe_ffn`)
Replace `hybrid_forward.rs:305-321` (the three `stage_expert`+`qmatvec_view` pairs) with, per proj:
```rust
let id = BlockId { layer: il, proj: GATE, ex: ex as u16 };
let slot = match cache.lookup(id) {
    Some(s) => s,                                              // HIT: resident, no H2D
    None    => cache.admit(id, m.gate_exps.expert_bytes(ex), &e)?, // MISS: stage (+maybe retain)
};
let gate = e.qmatvec_view(&cache.slots[slot], 0..g_len, &zt, 1,
    m.gate_exps.in_f, m.gate_exps.out_f, m.gate_exps.qtype, m.gate_exps.row_bytes)?;
```
`qmatvec_view` is unchanged — it already takes a `&CudaSlice<u8>` + range (`lib.rs:180-193`), so a
resident slot and a freshly-staged scratch are the *same* call site. **This is exactly why the
hit path is bit-identical to stage-every-token: the only difference between hit and miss is whether
the `memcpy_htod` ran; the bytes the kernel reads are byte-for-byte the same GGUF block** (the same
`m.gate_exps.expert_bytes(ex)` either copied this token or a prior token). No dequant, no re-pack,
no scale change. The gate in §D.2 asserts this.

`moe_ffn` signature gains `il: u16` and `cache: &mut MoeSlotCache` (or the cache lives on `Engine`
behind a `RefCell`/`Mutex` keyed by layer — see §E for the multi-agent decision). Callers:
`hybrid_forward.rs:47,94` (prefill) and `decode.rs:79` (decode) thread the layer index `il` (the
prefill loops already have `layer` but not its index — add `.enumerate()` at `:21,74`; decode
already has `il` at `decode.rs:50`).

### B.4 Slot count N (never hardcode)
At model load, after residents+KV are allocated, probe free VRAM via `cuMemGetInfo` (cudarc exposes
it; `SPILL-PLAN.md:113-130`) and set `N = floor(free_vram_after_residents / 1114112)` clamped to
`[8, 64]`. For 35B-A3B Q6_K with ~4 GB peak resident in a 24 GB card (`ROADMAP.md:17`), that is
hundreds of slots of headroom — but the working set per layer is only `8 routed × 3 proj = 24`
blocks, and across 40 layers the hot set is ~15–20% of `40×256×3` blocks. **N is shared across all
layers** (one global cache), so size it to the measured hot-set, not the theoretical max; start
N=256 (≈285 MB) and tune by the hit-rate counter (§D.3).

---

## C. Async pinned H2D + next-layer prefetch

### C.1 Pinned host buffer (low effort, ~2× H2D on miss)
Replace `HostExps.bytes: Vec<u8>` (`model.rs:197`) with a pinned allocation so the miss-path
`memcpy_htod` is a true DMA, not a pageable bounce copy. cudarc: `ctx.alloc_pinned::<u8>(len)`
returns a `PinnedHostSlice<u8>` (`cudarc .../core.rs:1412`); `stage_expert`'s `memcpy_htod`
(`lib.rs:172`) accepts it via the `HostSlice` trait. **Caveat (must document in code):**
`alloc_pinned` uses `CU_MEMHOSTALLOC_WRITECOMBINED` — great for H2D-only (the expert bytes are
never read by the CPU on the hot path), but **write-combined memory is slow for CPU reads**, so a
future CPU-VNNI cold-expert fallback (`ARCHITECTURE.md`, open-q #4) must NOT read from this buffer.
Load path: `HostExps::load` (`model.rs:208-235`) takes `&Engine`, allocs pinned, copies the GGUF
block bytes in once. Re-validate argmax=1178 unchanged (`SPILL-PLAN.md:164`).

### C.2 Second stream + event for prefetch
Add a copy stream to the runtime: `Gpu` (`bw24-runtime/src/lib.rs:34-46`) gains
`pub copy_stream: Arc<CudaStream>` via `ctx.new_stream()` (`cudarc .../core.rs:674`). `Engine`
exposes `stage_expert_async(host_bytes, slot, copy_stream)` that issues the `memcpy_htod` on the
copy stream and returns a recorded `CudaEvent` (`stream.record_event(...)`, `core.rs:751`); the
compute stream calls `stream.wait(&event)` (`core.rs:766`) before the dependent `qmatvec_view`.
This is the `ARCHITECTURE.md:112` "1 compute + 1 H2D; event-fork the copy stream for next-expert
prefetch" pattern, structurally the LMCache async-H2D discipline
(`research/inference-maps/lmcache.md:32-41`).

### C.3 What to prefetch (and what NOT to)
- **Within a token's 8-expert loop:** software-pipeline by one. While `qmatvec_view` for `sel[j]`
  runs on compute, `stage_expert_async` for the **miss** experts in `sel[j+1..]` runs on the copy
  stream into double-buffered staging slots. This hides the ~30–50 µs per-block H2D behind the
  ~per-expert GEMM. The `sel` array is fully known up-front (router ran for the whole token), so
  this prefetch is **deterministic**, not speculative.
- **The shared expert** (always routed, `hybrid_forward.rs:329-344`) and **layer L+1's dense /
  router / shared weights** are deterministic — prefetch them during L's compute. (These are
  GPU-resident already, so this only matters once a tiered-spill demotes them; out of scope for the
  EDGE-1 residency win, noted for §F.)
- **Cross-layer routed-expert prefetch is speculative** and we DON'T do it blind: L+1's router
  needs L's output, so the routed experts of L+1 are unknown during L. The honest version is to
  prefetch L+1's **historically-hottest** experts (the EAM/frequency table) as a warm-start — but
  that competes for copy-stream bandwidth with the in-token prefetch and risks evicting a resident
  hot block on a wrong guess. **Defer** until the in-token pipeline + SLRU steady-state is measured;
  it is upside, not a pillar (`SPILL-PLAN.md:131`, `ARCHITECTURE.md:196` PCIe re-clock caveat).

---

## D. Validation gates (in order; each must pass before the next)

1. **Router equality (A standalone).** With `BW24_FUSED_ROUTER=1`, dump `sel_idx`/`sel_w` from the
   kernel and from the host loop (`:291-298`) for the 35B-A3B prompt; assert **identical indices**
   and `sel_w` within 0 ULP (they should match exactly; the renorm divide is the only float op and
   both use `f32`). If any tie flips, switch the kernel to the CUB radix-sort variant (§A.2) and
   re-assert. Add to `bin/kernel_check.rs` (already modified in the tree; it hosts the dp4a-vs-GEMM
   equality gates at `:360-447`).
2. **Cache-hit bit-identity (B standalone, THE correctness gate).** Run `moe_ffn` twice on the same
   `z`: once stage-every-token (Stage-1, cache disabled), once with the SLRU cache pre-warmed so all
   24 blocks of the layer are resident. Assert the `moe_out` buffers are **bitwise equal** (`dtoh` +
   `==`, not `rel<1e-3`). This is mechanically guaranteed by §B.3 (same bytes, same kernel) but the
   test pins it against a future refactor that accidentally re-packs.
3. **End-to-end argmax (the real gate).** `run_gen` on 35B-A3B with `BW24_MOE_SLRU=1`
   (`run_gen.rs:64-66`): **prefill argmax MUST be 1178** (`ROADMAP.md:17`). Also run the full greedy
   continuation and diff against the Stage-1 baseline token stream — must be identical. Any drift =
   revert. This subsumes 1+2 end-to-end.
4. **PCIe → ~0 after warmup (the perf claim).** Instrument `MoeSlotCache` with `staged_bytes` and
   `hit/miss` counters; print per-N-tokens. The claim is **per-token re-staged bytes → ~0 after the
   first ~few hundred decode tokens once the hot set is resident**, NOT zero during warmup (§G).
   Cross-check with `nsys` H2D byte total: it must drop from `22.67 MB × T` toward the one-time
   hot-set fill (`hot_blocks × ~1 MB`). Compare against the `pp` timing already wired in commit
   `8d1c0b7` ("time prefill pp in run-gen").

---

## E. Multi-agent (2–4 concurrent) interaction — HONEST

The cache lives in process; with 2–4 agents each running their own `generate`/`decode` loop over
the **same** `HybridModel`, their routing **differs** (different prompts → different top-8 per
layer). Three real consequences:

1. **Shared cache contention / pollution.** A single global `MoeSlotCache` is shared across agents
   only if they share one `Engine`. If agent A's hot set and agent B's hot set diverge, they evict
   each other's blocks → lower hit rate, more PCIe. **SLRU's second-miss admission is exactly the
   mitigation:** a block another agent touched once doesn't evict A's protected hot block; only a
   block hot for *some* agent (seen twice) gets retained. The **union** of 2–4 agents' hot sets is
   still ~15–20% × a small constant, well within an N=256 cache for 35B-A3B (§B.4). Net: hit rate
   degrades gracefully, not catastrophically, as long as N ≥ union-hot-set.
2. **Concurrency model decision.** The bw24 HTTP scheduler is step-interleaved over 2–4 agents on
   one process (`BASE-4`, task #15). Two options: **(a) one shared cache behind a `Mutex`** —
   simplest, but the lock is held across `lookup`+`admit`+`memcpy_htod` issue (not the GEMM), so
   contention is the H2D-issue critical section (~µs), acceptable at B=2–4. **(b) per-agent caches**
   — no contention but N×agents VRAM and no cross-agent hot-set sharing (wasteful: agents on the
   same model DO share hot experts). **Recommendation: one shared cache, `Mutex`-guarded, sized to
   the union hot-set.** The lock does NOT cover the matmul, so streams still overlap.
3. **Per-agent residency bitmask is wrong; per-cache is right.** Residency is a property of the GPU
   slots (physical), not the agent, so the `occupant`/`table` is per-cache, not per-agent. The
   router output (`sel_idx`) is per-agent (different tokens) and flows through as a plain argument —
   no per-agent cache state needed beyond that.

**The honest failure mode:** if 3–4 agents have *fully disjoint* hot sets (e.g., wildly different
domains), the union can exceed N and the cache thrashes back toward Stage-1 PCIe. Mitigation is to
size N to the *measured* union (§B.4 tuning) and, if VRAM-bound, fall back to per-agent smaller
caches. This must be measured under the real 2–4-agent workload, not assumed.

---

## F. Explicitly OUT of scope (don't sell these as part of EDGE-1)

- **On-device DMA issue / fully eliminating the 64 B `sel_idx` dtoh** — requires device-side
  residency lookup + device-issued copies, which is the `cudaGraph` MoE-WHILE-loop work
  (`ARCHITECTURE.md:112`, "real MoE graph win requires on-device routing"). EDGE-1 stops at the
  fused router + host-driven SLRU dispatch.
- **Tier-2 mmap/NVMe spill + io_uring** — that is task #8 / `SPILL-PLAN.md`. This plan is Tier-0
  (VRAM slots) + Tier-1 (pinned host RAM) only; the routed bytes already fit in host RAM
  (`HostExps`), so disk is unnecessary for EDGE-1.
- **CPU-VNNI cold-expert recompute** — `ARCHITECTURE.md` open-q #4, a measured fallback, not a pillar.
- **Cross-layer speculative prefetch** — deferred (§C.3), upside only.

---

## G. Warmup transient — HONEST

"Per-token PCIe → ~0" is a **steady-state** claim, not an instant one. Concretely:
- **Cold start:** the first time each (layer,proj,ex) block is routed it misses; with second-miss
  admission it misses **twice** before retention (so the warmup re-stages each eventually-hot block
  ~2×). For a 40-layer model the hot set fills over roughly the first few hundred decode tokens (each
  token touches 24 blocks/layer × 40 = 960 block-accesses, of which the hot ~15–20% recur). The
  **warmup transient still pays full or near-full PCIe** — the win is amortized over a long
  generation, not on token 1.
- **Prefill is the worst case for the cache:** a single prefill forward touches each layer **once**,
  so within one prefill there is no cross-token reuse *within a layer* — reuse is across the T tokens
  of the SAME layer (the per-token loop at `:278` for T tokens of one layer DOES revisit hot experts).
  So prefill benefits when T is large (long prompt) and the same experts recur across positions; it
  benefits little for very short prompts. **Decode** (the steady stream, `decode.rs`) is where the
  ~0 PCIe claim is strongest: token after token through all 40 layers, the resident hot set serves
  the vast majority of routes.
- **Second-miss admission deliberately trades a slower warmup for a cleaner steady state.** The
  optional warm-start prior (§B.2) buys back most of the cold-start cliff by admitting predicted-hot
  experts on first miss. Report both warmup-included and steady-state PCIe numbers; do not quote only
  steady state.
- **PCIe re-clocking** (`ARCHITECTURE.md:196`): the link idles at Gen1 and re-clocks to Gen5 x8 under
  load. A burst of warmup misses re-clocks the link; once the cache is warm and PCIe goes idle, a
  later cold miss eats the re-clock latency. Keep a small priming copy at decode start so break-even
  is measured at full power.

---

## H. Build order (smallest reversible steps, each gated)

1. **Pinned `HostExps` (C.1)** — `model.rs:197,208-235` + `lib.rs:160`. Gate: argmax 1178 unchanged.
   Pure latency win, no logic change. Smallest, do first.
2. **`MoeSlotCache` + dispatch split (B)** — new `moe_cache.rs`; rewire `hybrid_forward.rs:305-321`;
   thread `il`/`cache` through `moe_ffn` + callers (`:47,94`, `decode.rs:79`). Behind `BW24_MOE_SLRU`.
   Gates: D.2 bit-identity, then D.3 argmax 1178, then D.4 PCIe counter.
3. **Fused router (A)** — `cu/moe_router.cu` + `lib.rs` launcher + `build.rs`. Behind
   `BW24_FUSED_ROUTER`. Gate: D.1 equality, then D.3 argmax 1178.
4. **Async prefetch (C.2–C.3)** — copy stream on `Gpu`; `stage_expert_async`; in-token pipeline.
   Gate: argmax 1178 unchanged + measured H2D-hidden-under-GEMM (nsys overlap).
5. **Multi-agent shared `Mutex` cache (E.2)** — wire into the `BASE-4` scheduler; size N to union
   hot-set. Gate: 2–4-agent run, per-agent token streams each match their single-agent baseline,
   measured aggregate hit rate.

**Files touched:** `crates/bw24-engine/src/moe_cache.rs` (new), `hybrid_forward.rs:247-347`,
`model.rs:196-235`, `lib.rs:147-193` (+ Engine fields `:36-65`), `decode.rs:79`,
`crates/bw24-engine/cu/moe_router.cu` (new), `crates/bw24-engine/build.rs`,
`crates/bw24-runtime/src/lib.rs:34-46`, `crates/bw24-engine/src/bin/kernel_check.rs` (gates).
