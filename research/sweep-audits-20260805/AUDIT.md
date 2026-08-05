# Sweep audits 2026-08-05 — CPU-only code audit vs the upstream trap catalog

Source: `research/upstream-sweeps.md` § Sweep 2026-08-05T12:20:53Z (ranked shortlist item 2 +
the three audit-tagged entries). Tree audited: `restructure/public-split` @ c025ac5b
(worktree `lane/sweep-audits`). Read-only analysis — every fix that needs GPU measurement is
QUEUED as a brief at the bottom, not executed here.

Verdict summary:

| # | Trap class | Verdict |
|---|---|---|
| 1a | Alloc/fill inside CUDA-graph capture (SGLang #33063) | **EXPOSED (known, partially mitigated)** — generic dc_cap capture carries per-layer alloc nodes + 3 captured fa-partials memsets/layer replayed every token |
| 1b | Fill kernel inside a PDL chain (SGLang #33063) | **EXPOSED (narrow)** — fa-partials memset-prefix sits directly before PDL-launched fa rows kernels |
| 1c | Silent permanent graph-coverage loss (TRT #16072) | **EXPOSED (two sites)** — draft-graph `graph_failed` memoization is silent by default and survives session parking; worker swallows graph-session step errors as `MaxNew` |
| 2 | Quadratic/linear eviction scaling with pool size (vLLM #50992) | **EXPOSED (two paths)** — PrefixCache LRU full-rescan per victim; MoE SLRU `on_hit` O(n_slots)/hit in the spill regime. F5 evict-first + ladder itself NOT exposed |
| 3 | Host-materialized sentinel forcing sync H2D per tick (SGLang #32575) | **EXPOSED (qualified)** — per-tick pageable `pos_v`/`ptr_table`/embed-gather uploads + per-chunk syncing D2H; one arm has a measured-flat receipt, the H2D uploads have none |
| 4 | Int32 row-addressing overflow (FlashInfer #4263) | **MOSTLY SAFE, one EXPOSED pattern** — all GEMV/MMVQ row addressing 64-bit; the vendored MMQ launchers' `offset_dst` is a 32-bit product (11 sites) wrapping at n_tokens·out_f ≥ 2^31 (quantized lm_head, single call T ≥ 8193 at 256k vocab) |

---

## 1. Decode-graph scratch + PDL chain (SGLang #33063 + TRT #16072 class)

### 1a. Allocation/fill INSIDE capture — EXPOSED (known, partially mitigated, residual cost measured)

The generic graph-session capture path — the one memra-serve's interactive greedy promotion
uses (`worker.rs:1306` → `graph_session_from_cache_masked` → `graph_capture_segment_masked`,
`decode.rs:1723`: `e.capture_graph(|e| { self.decode_step_dc_cap_masked(...) })`) — allocates
transients inside the captured step:

`crates/memra-engine/src/decode.rs:1353-1370` (`decode_step_dc_cap_masked`):

```rust
let mut x = e.embed_gather_device(embd_gpu, token_d, n_embd, embd_qt, embd_row_bytes)?;
for (il, layer) in self.layers.iter().enumerate() {
    let mixed = self.attn_in_norm_mixer_dc_cap(...)?;
    let (x1, ffn_out) = self.residual_norm_ffn(e, layer, &x, &mixed, n_embd, il, eps)?;
    let mut x2 = e.uninit(n_embd)?;          // <- cuMemAllocAsync node per layer, captured
    e.add(&x1, &ffn_out, &mut x2, n_embd)?;
    x = x2;
}
let mut hn = e.uninit(n_embd)?;              // <- captured alloc
e.rms_norm(&x, self.output_norm.float_data(), &mut hn, n_embd, 1, eps)?;
let mut logits = e.matmul(&self.output, &hn, 1)?;   // <- captured alloc inside matmul
```

The codebase itself confirms these become graph ALLOC nodes —
`crates/memra-engine/src/graph_update.rs:232-234`:

```
// programmatic encoding worked in pdl_probe (2690 -> 2434 ns/pair) but the ENGINE's captured
// graphs contain cuMemAllocAsync ALLOC NODES (cudarc allocs inside the captured step) and
// CUDA returns CUDA_ERROR_NOT_SUPPORTED for edge topology edits on such graphs.
```

And the fa-decode dc path fills (3 memsets) inside every captured layer —
`crates/memra-engine/src/lib.rs:9848-9850` (inside `fa_decode_dc_q8`, the capture-path fa):

```rust
self.gpu.stream().memset_zeros(&mut pg.0.slice_mut(0..o_len))?;
self.gpu.stream().memset_zeros(&mut pg.1.slice_mut(0..ml_len))?;
self.gpu.stream().memset_zeros(&mut pg.2.slice_mut(0..ml_len))?;
```

3 memset nodes/layer × ~48 layers = the "~144 mem nodes per decode token" the pool's own doc
names as a live tax — `crates/memra-engine/src/lib.rs:612-614`:

```
/// Pooled fa-decode split partials (part_o, part_m, part_l): per-call zeros() was 3
/// alloc+memset pairs per fa launch (~144 mem nodes per decode token — the graph door's
/// residual launch tax) — lazy-grow, memset-prefix per use, stream-ordered reuse.
```

(The pooling killed the *alloc* half of each pair; the *memset* half is still captured and
replayed every token.) The memsets are currently load-bearing —
`crates/memra-engine/src/graph_update.rs:130-134`:

```
/// the partial buffers are `zeros()` allocations whose memset is CAPTURED — every replay
/// re-zeroes them, so any split slot the main doesn't write holds m=0.0 (NOT the NEG_INF
/// empty the combine skips). The combine's merge count must therefore exactly equal the
/// main's written split count.
```

Mitigations already in place (why this is not the raw SGLang bug):

- Engine init pins the async mem-pool `RELEASE_THRESHOLD` to `u64::MAX` so captured alloc
  nodes replay as pointer bumps, not re-maps — `lib.rs:927-930`: "freed blocks return to the
  OS at every sync, so cuMemAllocAsync NODES inside captured graphs re-map memory on EVERY
  cuGraphLaunch (measured 226us/launch on the gemma graph door, 2026-07-23 osrt)".
- The known residual: `AUTO_FREE_ON_LAUNCH`'s "launch-time mem-pool scan was measured at
  ~0.25us/node (205us on the 826-node step) even with nothing to free" (`lib.rs:10004-10007`),
  which is why the alloc-free gemma slotted door instantiates with a non-AUTO_FREE flag
  (`hybrid_forward.rs:5849-5850` passes `USE_NODE_PRIORITY`; `lib.rs:10004-10006` documents
  `UPLOAD` for zero-mem-node graphs) — `gemma4_decode_step_dc_slotted` is the existing proof
  of the fix shape: slot-fed twins for every transient, zero mem nodes.

**Exposure**: the generic `GraphSession` path (every non-gemma graph promotion in serve, and
`graph_decode_loop`) still replays per-layer alloc/free nodes + 144 memset nodes per token and
pays the AUTO_FREE launch scan. Known, bounded, and exactly the class SGLang deleted for a win.

NOT exposed on the adjacent capture sites checked: the prefill S-mid/S-glue segment graphs
(`hybrid_forward.rs:644-656, 726-739`) capture only `e.add` + `e.rms_norm_f16out` — both
alloc-free, caller-owned-buffer launches (`lib.rs:5339-5350`, `lib.rs:7335-7347`) — zero mem
nodes by construction; and the gemma slotted door is alloc-free by design (§ above).

**Fix brief (Q1)**: extend the slotted alloc-free discipline to `decode_step_dc_cap_masked` —
slot-feed the per-layer `x2`/`hn`/`logits` transients (the `G4DcSlots` pattern already exists),
elide the fa-partials memset by NEG_INF-init + combine-skip-empty (or write all grid slots
in-kernel), then instantiate with `UPLOAD` instead of `AUTO_FREE_ON_LAUNCH`.
Gate: graph-decode-gate bit-identity (256 steps), graph-session-gate, `MEMRA_GRAPH_CENSUS`
before/after showing mem nodes → 0, and an interleaved ×5 tok/s A/B on the graph-session path.
Est. size: 2-4 days (the slotted door is the template; the combine change touches every fa
combine kernel's empty-slot semantics and must be re-proven bit-exact).

### 1b. Fill inside a PDL chain — EXPOSED (narrow: the fa rows PDL lanes)

memra's PDL is launch-side `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`
(`lib.rs:1394-1445`, `launch_pdl` / `launch_pdl_flash`; kernel-side
`MEMRA_PDL_ENTRY() = cudaGridDependencySynchronize()`, e.g. `cu/flash_attn.cu:60`,
`cu/qmatvec.cu:18`). PSS pairs a launch with its immediately preceding stream op. In the
rows/verify fa lanes the 3-fill memset-prefix sits DIRECTLY between the producer chain and a
PDL-attributed fa kernel:

`crates/memra-engine/src/lib.rs:9484-9486` (memsets) →
`lib.rs:9536-9538` (`launch_pdl_flash(wg, "fa_decode_vec_q_rows_v4_w_sp", ...)`); same shape at
`lib.rs:9297-9299` → `lib.rs:9335-9339` (`fa_decode_vec_q_rows_v4_512_tb`) and the dc-combine
PDL arm `lib.rs:9389`.

If the driver implements `memsetAsync` as a fill kernel (it does on current drivers for f32
patterns — the exact SGLang FillFunctor mechanism), the PDL fa kernel's early-launch pairs with
the ~µs fill instead of the real producer (the QKV projection GEMV), forfeiting the ~120ns/kernel
overlap the PDL arm exists for on precisely the lane where fa is hottest. The decode dc path is
NOT exposed here (its vec `_dc` twins launch through the plain builder, `lib.rs:9925-9928`).

**Fix brief (Q4)**: rides Q1 — eliding the memset-prefix removes the interposed fill; if Q1's
combine change is deferred, a cheap alternative is issuing the 3 memsets BEFORE the producer
GEMV (they have no dependency on it) so the PDL pair is producer→fa again. Gate: pdl_probe-style
nsys pair timing on the rows lane + the standard battery. Est. size: half a day if reordering,
free if Q1 lands.

### 1c. Silent permanent graph-coverage loss — EXPOSED (two sites)

**Site 1 — draft-graph capture failure memoized silently, survives parking.**
`crates/memra-engine/src/spec.rs:3204-3210`:

```rust
Err(err) => {
    scratch.set_len(e, base)?;
    dctx.graph_failed = true;
    if debug_spec {
        eprintln!("[spec] draft-graph capture failed ({err}); eager fallback");
    }
}
```

(the sampled twin: `spec.rs:3279-3285`, `graph_s_failed`). The memoization is by design
(`spec.rs:356-357`: "`*_failed` memoizes a failed capture so the eager fallback doesn't pay a
doomed capture attempt every burst") — but the log line is gated behind `MEMRA_DEBUG_SPEC`, so
by default a capture failure under transient memory pressure drops the draft graph with ZERO
log output. And the `DraftGraphCtx` parks on the session (`spec.rs:4713-4715`: "Park the
draft-graph ctx back on the session") and spec sessions park in the reuse pool across requests
(`worker.rs:201-209`), so `graph_failed=true` persists across future requests that resume the
parked session — the TRT #16072 shape: pressure-triggered, silent, long-lived coverage loss.
Only a dmask shape change (`spec.rs:3155,3160`) or sampled `s_key` change (`spec.rs:3240`)
resets it; a plain resume never does.

**Site 2 — graph-session step error swallowed as MaxNew.**
`crates/memra-server/src/worker.rs:1351-1354`:

```rust
match g.step(&engine, &lm.model) {
    Ok(next) => { s.graph_pending = Some(next); }
    Err(_) => { finish(s, StopReason::MaxNew); finished.push(0); }
}
```

`GraphSession::step` errors for real causes besides budget exhaustion — recapture failure at a
kernel-class boundary (`decode.rs:64` → `graph_session_recapture`, which allocates), and
`fa_apply` exec-update failure. All of them are reported to the client as a normal `MaxNew`
stop with the error text discarded (`Err(_)`). A recapture OOM under KV pressure truncates the
generation silently. Contrast: the PROMOTE failure right above is loud
(`worker.rs:1316-1317` sends `Event::Error("graph promote failed: ...")`), and the gen-graph
MoE door-close is loud (`decode.rs:2498-2503` `eprintln!` once).

Not exposed on the adjacent TRT trap arms: padding/partials are preallocated or
retire-on-grow (never freed) — `lib.rs:615-619` `fa_part_pool`/`fa_part_retired`, and capture
warmup transients are pinned by the keeper (`lib.rs:10021-10041`); graph buffers are sized at
capture from `bucket_max` (`decode.rs:1780`: `bucket_max = cache.pos + max_new + 1`), not
lazily under pressure.

**Fix brief (Q2)**: (a) unconditional single-shot `eprintln!`/metrics counter when
`graph_failed`/`graph_s_failed` flips true (the `NOTICE: Once` pattern from `decode.rs:2498`);
(b) reset `graph_failed` on session resume-from-park (one line where the pool hands the session
back) so a transient-pressure failure isn't permanent; (c) `worker.rs:1353`: log the step error
and use a distinct stop reason (or at least `Event::Error`) instead of silent `MaxNew`.
Gate: serve-smoke + a forced-failure unit (capture with event tracking on errors deterministically
— `decode.rs:1622-1624`). Est. size: half a day, CPU-only except the smoke.

---

## 2. ARC/eviction complexity (vLLM #50992 class)

### EXPOSED path 1 — PrefixCache LRU eviction: full rescan per evicted entry (O(E²) aggregate)

`crates/memra-server/src/worker.rs:629-664` (`PrefixCache::insert`), loop proper at 646-663:

```rust
self.entries.entry(key.clone()).or_default().push(e);
while self.total_bytes > budget {
    let mut victim: Option<(PoolKey, usize, Instant)> = None;
    for (k, pool) in &self.entries {
        for (i, e) in pool.iter().enumerate() {
            if victim.as_ref().is_none_or(|&(_, _, t)| e.last_use < t) {
                victim = Some((k.clone(), i, e.last_use));
            }
        }
    }
    let Some((k, i, _)) = victim else { break };
    let dead = self.entries.get_mut(&k).map(|p| p.remove(i));
    ...
}
```

Each `while` iteration re-scans EVERY entry in EVERY (model, ns) pool for the global LRU
minimum — O(E) per victim plus `Vec::remove(i)` — so an insert that evicts k entries costs
O(k·E); flushing a pool of many small entries is O(E²). Exactly the vLLM #50992
rescan-from-head shape. Realistic bound: budget = `MEMRA_PREFIX_CACHE_MB` default 256 MB
(`worker.rs:530-538`), entries ≥ 64 tokens of deep-copied KV (`PREFIX_CACHE_MIN_TOKENS`,
`worker.rs:528`), so E is tens-to-hundreds at default — but E is NOT structurally capped:
raising the budget knob on a small-KV model grows the per-insert cost quadratically. The
admission-side probes `lookup`/`best_lcp`/`has_covering` (`worker.rs:593-624`) are additionally
O(E × prompt_len) token-compare scans per admit.

### EXPOSED path 2 — MoE SLRU `on_hit`: O(n_slots) scan per HIT once full (spill regime)

`crates/memra-engine/src/moe_cache.rs:540-557`:

```rust
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
        self.push_protected(slot);
    }
}
```

`position()` + `VecDeque::remove(pos)` are each O(n_slots) — PER HIT. The early-out only
protects a not-yet-full cache; in the spill regime the module's own docs state the cache is
"permanently full" (`moe_cache.rs:1023-1024`), so every hit pays the scan. The module
self-documents the magnitude (`moe_cache.rs:532-537`): "~46k slots x ~850 hits/token that was
~40M host ops/token — measured as the fast-admit A/B regression 48.5 -> 46.0 tok/s on the 35B
g7e decode" — that fix DEFERRED the scan until the cache fills; it did not remove it.
n_slots auto-sizes to 85% of free VRAM (`moe_cache.rs:317-345`) — tens of thousands, unbounded
by any constant. Aggregate O(hits × n_slots) per decoded token, every token, whenever the model
spills. Related same-module scans: `frequency_victim_in_class` (`moe_cache.rs:583-609`,
O(n_slots)/eviction — LFU opt-in only), `evict_one_excluding` (`moe_cache.rs:636-658`,
O(n_slots) worst per prefetch eviction). The default `evict_one` (`moe_cache.rs:612-631`) is
`pop_front()` O(1) — NOT exposed.

### NOT-EXPOSED — the F5 evict-first + right-size ladder itself

`crates/memra-server/src/worker.rs:2280-2298`:

```rust
if spec_sizing.evict_first.contains(&req.model) {
    if let Some(n) = spec_reuse.get_mut(&pool_key)
        .map(|p| { let n = p.len(); p.clear(); n }).filter(|&n| n > 0)
    { ... }
}
match lm.model.new_session(engine, ctx_cap) {
    Ok(sess) => Some(sess),
    Err(first_err) => {
        let evicted = spec_reuse.get_mut(&pool_key)
            .map(|p| { let n = p.len(); p.clear(); n }).unwrap_or(0);
```

`p.clear()` is O(pool) with pool ≤ `MEMRA_REUSE_POOL` = 2 default (`worker.rs:222-233`); the
ladder loop (`worker.rs:2306-2353`) halves `ask` per miss — O(log(ctx_cap/need)) attempts,
memoized via `learned_ctx` (`worker.rs:2346`). No pool-size scaling. Also NOT exposed:
continuation/spec/affinity admit probes (`worker.rs:1965-1971, 2126-2143, 2178-2218` — pool
depth 1-2), park-at-retire eviction (`worker.rs:1772,1797`, ≤1 removal/retire at the cap),
tick scans (active ≤ 64+4+8), `StepStats::p` sort (window hard-capped 16,384,
`memra-lanes/src/lib.rs:69-93`), and memra-kv (no eviction code at all).

**Fix brief (Q3)** — PrefixCache: keep a `total_bytes`-ordered auxiliary index or a
monotonic-iterator sweep (the vLLM fix shape: never rescan from head; collect victims in one
ordered pass, apply mutations after). O(E log E) worst, O(1) amortized per victim. CPU-only
change + unit test over a synthetic 10k-entry pool; no GPU gate needed beyond serve-smoke.
Est. size: half a day.
**Fix brief (Q5)** — MoE SLRU `on_hit`: intrusive doubly-linked recency list (slot → node
handle) or index-map beside the VecDeques, making promotion O(1). The 48.5→46.0 receipt says
~5% decode is on the table in the spill regime. Gate: spill-regime decode A/B on the 35B
(interleaved ×5) + cache hit-rate parity. Est. size: 1-2 days (GPU-gated — queue for the rig).

---

## 3. Host-materialized sentinel H2D sync (SGLang #32575 class)

**Verdict: EXPOSED (qualified)** — the hot batched tick materializes three small host buffers
every step and uploads them from pageable memory, plus one explicitly-synchronizing D2H per
chunk per tick. One arm carries a measured-flat receipt; the H2D uploads carry none.

Hot path: `worker.rs:890` `run()` → main loop `worker.rs:1068` → batched decode
`decode_step_batch_sampled_lean_masked` per chunk per tick
(`crates/memra-engine/src/decode_batch.rs:319`).

Per-tick host materializations (the #32575 shape):

```rust
// decode_batch.rs:416-417 — host Vec built + pageable H2D EVERY tick
let pos_v: Vec<i32> = caches.iter().map(|c| c.pos as i32).collect();
let pos_d = e.htod_i32(&pos_v)?;
// decode_batch.rs:468 — pointer table rebuilt host-side + uploaded EVERY step
let ptr_table = if ptrs.is_empty() { None } else { Some(e.htod_u64(&ptrs)?) };
// decode_batch.rs:491-492 — host embed gather + pageable H2D of [B, n_embd] f32 per tick
let mut x = e.htod(&self.embd.gather(n_embd, tokens))?;
```

B=1 twins: `decode_batch.rs:184-185` (`pos_d` + embed gather per token) and `:225`
(`dtoh_u32` per token). Per-constrained-row grammar-mask upload per step:
`worker.rs:2569` (`engine.htod_u32_into(d, words)` — ~n_vocab/8 bytes pageable H2D).
Spec-burst per-round setters + per-drafted-token syncing readbacks on the non-round-stream
arm: `spec.rs:3705-3706` (`set_i32_one`/`set_u32_one`), `:3730` (`dtoh_u32_one`),
`:3737` (`dtoh(&dctx.g_p)`); the round-stream arm amortizes M rounds behind ONE
`synchronize` (`spec.rs:3574-3576, 3636`).

Helper sync classes (`crates/memra-engine/src/lib.rs`):

- Pageable-async (driver-staged, no stream sync): `htod_i32` `lib.rs:3818-3820`,
  `htod`/`htod_u64`/`htod_u32_v`, `htod_u32_into` `lib.rs:4042-4047`, and the one-element
  setters `set_i32_one`/`set_u32_one` `lib.rs:4093-4102`. The repo's own doc classifies the
  setter as a trap — `lib.rs:4078-4080`: "set_i32_one below is the SYNCING pageable copy (fine
  at stream-idle boundaries, poison mid-round)" — and provides the async kernel-arg
  alternative `i32_set_k` (`lib.rs:4081-4091`).
- Explicitly synchronizing: EVERY dtoh helper — `dtoh_u32_one` `lib.rs:4104-4108`
  (`clone_dtoh` + `stream().synchronize()`), `dtoh_i32_one` `:4070-4074`, `dtoh` `:3835-3839`,
  `dtoh_u32` `:4034-4038`, `dtoh_view` `:3829-3834`.
- Pinned staging exists (`PinnedStage`, `lib.rs:747-766`) but is used ONLY for the MoE router
  readback (`lib.rs:2478-2490`) — no worker-tick H2D/D2H rides pinned memory.

The per-chunk syncing D2H (`decode_batch.rs:808` `let host_toks = e.dtoh_u32(&toks)?;`) has a
receipt — `decode_batch.rs:329-335`: a deferred-token-readback variant "measured FLAT at serve
level on the 5090 (N=4 medians within +-0.7%)"; the tick is weight-bound there. The per-tick
`pos_d`/`ptr_table`/embed-gather pageable uploads and the per-row mask upload have NO such
receipt and no pinned staging.

Per-session costs verified fine (not the trap): `prefix_restore` `worker.rs:747`, graph
promotion `decode.rs:1630-1636`, retire-time `worker.rs:1786-1788`, mask first-alloc
`worker.rs:2572-2574`.

**Fix brief (Q6)**: move `pos_v`/`ptr_table`/embed-gather/grammar-mask uploads onto a small
persistent pinned staging ring (the `PinnedStage` type already exists), and make per-row rope
positions device-resident (the graph path already keeps `pos_d` on device and bumps it with
`inc_seqlen` — the batched path can inc a `[B]` i32 buffer in-kernel instead of re-uploading).
Gate: interleaved ×5 serve A/B on short-turn TTFT + batched decode tok/s (this targets the
dogfood short-turn deficit), nsys confirming no pageable-staging stalls in the tick.
Est. size: 1-2 days (GPU-gated).

---

## 4. Int32 row-addressing overflow (FlashInfer #4263 class)

**Verdict: MOSTLY SAFE — one EXPOSED pattern (11 instances, all vendored MMQ launchers) plus
two latent-only sites.** Reference scales: n_embd 5376-8192, n_ff up to 40960, vocab 262144;
largest matrix = lm_head 262144 × 5376 ≈ 1.4e9 elements; prefill chunks M ≤ 8192, so
M·K ≤ 8192·40960 ≈ 3.36e8 per call on the chunked paths.

### SAFE — the GEMV/MMVQ row addressing (the audit's named targets)

Every weight-row base multiply is 64-bit; `row_bytes` is `long` in every kernel ABI and
crosses FFI as `i64`:

- Q8_0 MMVQ — `cu/qmatvec.cu:634`: `const unsigned char* wrow = W + (long)o * row_bytes;`
  (fused row1 twin `:665`; batched `:2918, :2968`; dual-row `:3023-3024`
  `W + (long)(o0 + 1) * row_bytes`). Max product 262143 × 8704 ≈ 2.28e9 > 2^31 at
  vocab×8192 — the `(long)` is load-bearing. Output writes `y[(size_t)t * out_f + o]`
  (`:652`), activation rows `aq + (size_t)t * in_f` (`:635`).
- e4m3 mmvq — `cu/qmatvec.cu:3296`:
  `e4m3_row_dot(W + (long)o * row_bytes, aq + (size_t)t * in_f, ...)`; row_bytes == in_f
  (raw checkpoint bytes), max 262143 × 8192 ≈ 2.15e9 > 2^31 — cast load-bearing. Batched
  twin `:3318`, writes `y[(size_t)c * out_f + o]` `:3351`.
- NVFP4 MMVQ — `cu/qmatvec.cu:1088`: `const unsigned char* wrow = W + (long)o * row_bytes;`;
  superblock `:1169` `W + (long)o * row_bytes + (long)sblk * 36`; split-plane rp/ca twins
  `:1919-1923` (`(size_t)out_f * nsb64 * 32` plane offsets, `long qstride`); prefetch twin
  `:1406` `wrow + (long)(g >> 1) * 36`. All 64-bit.
- mmq_fp8_blk tile bases — `cu/mmq_fp8_blk.cu:427-428`:
  `x + (size_t) it * mmq_y * (size_t) stride_row_x` (byte offset, 262144·8192 = 2.15e9 at
  the largest shape — casts load-bearing) and the scale-grid pointer, both `size_t`.
  Row loader `:240` `x + (size_t) row * (size_t) stride_row + (size_t) kv`. SAFE.
- fp8_blk_dequant — `cu/fp8_blk_dequant.cu:81-83` decomposes a flat 1-D grid in
  `long long`; `:103` `f8_weights[(size_t)row * (size_t)in_dim + (size_t)col]`; `:117`
  `out_q8 + ((size_t)row * (size_t)blocks_per_row + (size_t)qb) * Q8_0_BYTES` (max ≈ 2.28e9,
  cast load-bearing). SAFE.
- Vendored MMQ activation quantizers all-int64 — `cu/mmq_fp8_blk.cu:442-452`
  (`quantize_mmq_e4m3_d128_kernel`: `int64_t i0/i1/ib/iqs`), same in `mmq_q8_0.cu:405`,
  `mmq_nvfp4_f8f4.cu:53`, `mmq_fp4.cu:532-541, 720-731`. SAFE.
- Host-side: every `row_bytes` passes as `i64` (`lib.rs:2372, 2523, 3702, ...`;
  `mmq_ffi.rs:323/342/350/396` extern decls). No i32 stride found. MoE expert bases are u64
  device-pointer tables (`qmatvec.cu:4770/4785` `table[(size_t)proj * n_expert + ex]`), built
  host-side in usize (`hybrid.rs:304-306`) — no in-kernel expert×stride 32-bit multiply.

### EXPOSED — `offset_dst` int product in ALL vendored MMQ launchers (11 instances)

```c
const int offset_dst = jt * mmq_x * stride_col_dst + it * mmq_y;
```

All factors `int`; `jt = blockIdx.y` (token tile), `mmq_x` = 128-256,
`stride_col_dst = out_f`. `jt*mmq_x*out_f` computes entirely in 32-bit, then feeds
`dst + offset_dst` — a negative pointer add on overflow. Instances (verified verbatim):

- `cu/mmq_q8_0.cu:387`, `cu/mmq_q8_0_f32acc.cu:414`
- `cu/mmq_q4_0.cu:445, :514, :581, :604`
- `cu/mmq_q45k.cu:481`
- `cu/mmq_nvfp4_w4a8.cu:788, :1367`
- `cu/mmq_fp4.cu:435`
- `cu/mmq_fp8_blk.cu:422`

**Overflow arithmetic**: needs `jt·mmq_x·out_f ≥ 2^31` ≈ `n_tokens·out_f ≥ 2^31` in ONE MMQ
call. At lm_head out_f = 262144: n_tokens ≥ 8193 (jt = 64 at mmq_x = 128 → 64·128·262144 =
2^31 exactly; wraps negative from jt ≥ 64). At n_ff 40960: needs n_tokens ≥ 52429 — out of
range. So only vocab-sized out_f is realistically reachable.

**Reachability**: the MMQ dispatch (`lib.rs:5417`: `if m >= GEMM_M_THRESHOLD && out_f >=
GEMM_MIN_OUT_F && self.mmq_supports(w) { return self.qmatvec_mmq(w, x, m); }`) passes the
call's FULL m; the launchers grid `ntx = (n_tokens + MX - 1)/MX` with no internal chunking
(`mmq_q8_0.cu:498`, `mmq_fp8_blk.cu:606`). The full-T lm_head matmul exists:
`forward.rs:146` `let logits = e.matmul(&self.output, &hn, t)?;` (`ModelForward::forward`,
the dense/prompt-logits path). The production hybrid prime path computes lm_head on the last
row only and chunks the trunk at `MEMRA_PRIME_CHUNK` default 4096, so decode/prime never
reaches the bound (≤ 3.4e8, ~6x margin). Exposure = a quantized-lm_head model evaluated
through the non-chunked full-logits entry with a single-call T ≥ 8193 on a 256k-vocab model
— a supported shape class (prompt-logits eval), not the serve hot path.

**Latent-only (NOT exposed at supported shapes)**: memra's own `quantize_q8_1`
(`qmatvec.cu:481-487`) and `quantize_fp4_act` (`qmatvec.cu:546-548`) compute the flat
block-of-32 thread id in `int` (`int blk = (blockIdx.x * blockDim.x + threadIdx.x) >> 5` with
bound `blk >= m * nblk_row`); overflow needs m·in_f ≥ 2^31 ≈ M ≥ 52429 tokens at in_f 40960 —
~6x above the largest chunked-prefill call. The byte offsets themselves are `size_t`
(`qmatvec.cu:486` `size_t off = (size_t)t * in_f + b * 32 + lane;`). Secondary int products
in the MMQ kernels (`offset_y`, y-chunk pointers, `offset_x` in block units, write-back
`ids_dst[j] * stride + i` post-offset) are all bounded ≥ 20x under 2^31 at max shapes.

**Fix brief (Q7)**: one-line change at all 11 sites —
`const int64_t offset_dst = (int64_t)jt * mmq_x * stride_col_dst + (int64_t)it * mmq_y;`
(the exact FlashInfer #4263 remediation), plus widen the two latent quantizer thread-id
bounds to `size_t`/`long long` while touching the area (defense-in-depth, zero cost).
Gate: kernel-check (the MMQ kernels are all bit-identity-gated) + a synthetic large-M
lm_head MMQ call (m = 8320, out_f = 262144 random weights) asserting no OOB write
(compute-sanitizer). No perf gate needed — index-type change only, but per doctrine the
battery runs before merge. Est. size: half a day including the sanitizer repro.

---

## Ranked fix queue

Ranked by (expected win on the felt decode/serve path) × (confidence) ÷ (size). GPU-gated
items are QUEUED for the rig; CPU-only items are executable immediately on a lane branch.

| Rank | Brief | From | Cost class | GPU-gated? | Est. size |
|---|---|---|---|---|---|
| 1 | **Q2 — loud + resettable graph fallbacks**: unconditional one-shot log/counter on `graph_failed`/`graph_s_failed` flip (`spec.rs:3206,3280`), reset on pool resume, and `worker.rs:1353` step-error surfaced instead of silent `MaxNew` | 1c | Correctness-of-evidence: silent permanent coverage loss is exactly the TRT #16072 trap; costs nothing at runtime | smoke only | 0.5 day |
| 2 | **Q7 — 64-bit `offset_dst` in the 11 MMQ launchers** (+ widen the two latent quantizer thread-id bounds) | 4 | Real OOB-write class on a supported shape (quantized lm_head, T ≥ 8193, 256k vocab); one-line-per-site, bit-identity gated | YES (kernel-check + sanitizer repro) | 0.5 day |
| 3 | **Q3 — PrefixCache O(1)-amortized eviction**: ordered sweep / aux index replacing the per-victim full rescan (`worker.rs:646-663`) | 2 | Latent quadratic wall behind one env knob; the vLLM fix shape is known (16-81x on eviction batches) | no (unit test) | 0.5 day |
| 4 | **Q1 — alloc-free generic graph capture**: slot-feed `decode_step_dc_cap_masked` transients, elide fa-partials memsets, instantiate non-AUTO_FREE | 1a | ~205us/step scan receipt + 144 replayed mem nodes/token on every non-gemma graph session; the slotted door proves the shape | YES (bit-identity battery + A/B) | 2-4 days |
| 5 | **Q5 — MoE SLRU O(1) `on_hit`**: intrusive recency list replacing the O(n_slots) scans (`moe_cache.rs:546-551`) | 2 | ~5% decode receipt (48.5→46.0) in the spill regime; only matters when spilling | YES (35B spill A/B) | 1-2 days |
| 6 | **Q6 — pinned/device-resident tick inputs**: `pos_v`/`ptr_table`/embed-gather/grammar-mask off pageable per-tick uploads (`decode_batch.rs:416-492`, `worker.rs:2569`) | 3 | Unreceipted pageable staging in the hot tick; targets the dogfood short-turn deficit, but the one measured arm was flat — win uncertain | YES (serve A/B + nsys) | 1-2 days |
| 7 | **Q4 — PDL chain hygiene on the rows fa lanes**: reorder/elide the memset-prefix between producer and PDL fa kernel (`lib.rs:9484-9486` → `:9536`) | 1b | ~120ns/kernel-class win at most; free if Q1 lands | YES (nsys pair timing) | 0.5 day (or free) |

Execution note: Q2 + Q3 are CPU-only and safe to land from this audit lane's follow-up;
Q1/Q4/Q5/Q6/Q7 touch kernels or need the target-rig battery per the correctness discipline
and are queued as briefs only (no kernel edits in this lane).
