# lane/pp-leverb — LEVER B: the stage-split chunked prime over PP-2 (#94)

**Mission:** dev1 ran ZERO prefill kernels in the anatomy trace
(`research/pp-prefill-20260807/raw/anatomy-20260807T114137Z-*`): prime walks all 45 layers on
dev0, peer-reading stage-1 trunk weights (22% of wall: MMQ 34x, router 100x, dp4a slow-class on
the fence [0,22,45]) and peer-writing stage-1 KV. Baseline after Lever A: **~141 tok/s pp4096**
(153.3 first-warm, 140.6-141.2 steady; `raw/leverA-gates2-*.log` G6). A+B projection 420-450.

**Scope re-order (coordinator, 2026-08-08, v0.73 target):**
increment 1 = this file + the split-vs-unsplit prime gate registered RED;
increment 2 = the RESIDENCY FLIP ALONE (rebase onto the train tip first — cx-503b already
merged per-PP-device resident-expert SIZING at `6afc4f65`; this lane owns the
placement/decision flip on top); increment 3 = the per-stage prime walker, only after 2 is
gated + measured. Measure after each increment: pp4096 N=5 interleaved vs the ~141 Lever-A
arm, one flock hold.

---

## Increment 0 — structural reading (code facts; branch base 80f47796 = train tip incl. Lever A)

### A. The prime path today — what a split must move

- `prime_cache` (`crates/memra-engine/src/hybrid_forward.rs:422`): computes the request's
  `seq_end = cache.pos + t + queued_after` ONCE before the chunk loop (the chunkfix +
  tick-seg laws), then loops `prime_chunk` per `MEMRA_PRIME_CHUNK` (default 4096), copying
  each chunk's hidden stack into a request-wide `hiddens` buffer on the PRIMARY engine.
- `prime_chunk` (`hybrid_forward.rs:569-884`): the full 45-layer walk on ONE engine `e`.
  Call-site inventory (all `e.`-routed, so an engine-handle swap is the mechanical part):
  * entry: `htod_i32(pos)`, `embed` (host gather + htod) — stage-0 work (embed table is
    host-side with stage 0, per pp.rs ownership law);
  * **PrimeSlabs** (`hybrid.rs:1031` `Mutex<Option<PrimeSlabs>>`, struct at
    `hybrid_forward.rs:12`, getter `prime_slabs_get` :539): SINGLE-ENGINE resident trunk
    transients — xa/xb ping-pong, h/x1/z/act, h16/z16 (f16 operand slabs), gate/up/ffn_out,
    plus seg_glue/seg_mid capture arrays. Allocated on whichever engine first primes at this
    t. **A per-stage walker needs per-stage slabs** (split the struct per stage or key a
    second slab set off the stage engine). This is the largest structural cost after the
    loop split itself.
  * per layer: `rms_norm(_f16out)` / `add_rms_norm_f16out` (norm+residual chain, with a
    cross-layer fusion carry exactly like decode's — the carry is range-LOCAL, so a fence
    cut only needs the residual materialized at the boundary, same as
    `decode_layers_eager`'s contract);
  * mixer: `full_attn_prime` (:1336) → for step35 `step35_attn_prime` →
    `step35_attn_pre_wo` (:6867) — matmul_group projections, QK-norm, partial rope,
    **KV append into `cache.kv[il]`** (stage-owned rows under `pp::new_cache` — the
    append must run on the OWNING stage's engine to stop being a peer write), the
    SWA/full FA view dispatch (Lever A's `fa_prefill_view_ws_w_hd128`, selection keyed on
    `seq_end`, tile-grid `off &= !31` alignment), head-wise attn gate, wo;
    `linear_attn_prime` (:1542) for GDN layers (qwen — step35 has none; walker must still
    thread it for the generic arm);
  * FFN: `moe_ffn_il` (:2036) → dispatch web below; dense arm for non-MoE;
  * S-glue/S-mid graph captures (`use_seg`): **step35 EXCLUDED by predicate**
    (`cfg.step35.is_none()` at :663) — the walker does not need capture-safety for the
    Step SKU; the generic arm keeps them only if `MEMRA_PRIME_SEG=1` (opt-in, off default);
  * epilogue: `output_norm` + lm head on the LAST layers' engine — the sharded loader
    already uploads `output_norm`/`output` through `e_head = layer_engine(n_trunk-1)`
    (`hybrid.rs:1098`), so the head belongs to the last stage for free;
  * returns `(host logits, h_seed [n_embd] dev, hiddens [T,n_embd] dev)`. h_seed + hiddens
    are consumed by generate_spec (prompt_h) and run-gen; under a split they are produced
    on the LAST stage's stream while callers resume on the primary — **`rt.publish_to`
    (pp.rs:781) at exit is mandatory** (the pp2-spec device-resident-output law; measured
    NaN/garbage class when skipped on one device).
  * `cache.pos += t` per chunk (host state, caller-ordered).
- Boundary payload: `[chunk_t, n_embd]` f32 = 64 MB at 4096x4096. `rt.tx/rx` already take a
  payload ELEMENT COUNT with grow-only slots (the batched arm passes `b_n * n_embd`) — prime
  passes `chunk_t * n_embd`, no transport work needed. 1.1 ms at the measured 56 GB/s,
  hidden under chunk compute.
- Chunk-invariance composition: the walker changes WHERE layers run, never which kernel a
  row takes — selection stays keyed on `seq_end`. chunkinv35/tickinv35 must stay green and
  are the early-warning gates (they caught 2 defects in Lever A).

### B. The decode-side pattern to mirror (paid bill, reuse verbatim)

- `decode_step_h_ppn` (`decode.rs:827`): stage 0 = `rt.enter(0)` + `rt.engine(0,e)` +
  per-stage `pos_d` + embed + `decode_layers_eager(fence[s], fence[s+1])` + `rt.tx`;
  middle = `rt.rx` → range → `rt.tx`; last = `rt.rx` → range → output_norm + head.
- `decode_step_batch_ppn` (`decode_batch.rs:665`): adds (1) per-stage `BatchLayerCtx`
  pointer tables uploaded through the stage engine; (2) **per-Engine flag scoping** — the
  exact16 `verify_exact` scope must be set on EVERY stage engine (per-Engine atomic state);
  any per-Engine state the prime path touches has the same trap; (3) the step35 refusal
  outside B=1 (generic batched body lacks step35 geometry) — the PRIME walker must thread
  the step35 mixer per stage, which it gets for free by calling the same `prime_chunk`
  layer body, not the generic batched one.
- Laws inherited (do not relearn): per-stage Engine isolation (shared scratch pools =
  the 35% flake class), slot first-use ordering (host-sync per slot alloc), publish_to at
  exit, `sync_stages_after_load` barrier, per-stage pos_d freed on its own stream.
- `ppn-gate` (`bin/ppn_gate.rs`) = the bit-identity gate shape: reference recorded first,
  identical token sequence replayed through the split, every f32 bit compared, teeth via
  seams. pp2-batch's `--mode pp` refinement: reference arm = door OPEN but split seam OFF
  over the SAME sharded load (a door-off load of a 105 GB model doesn't fit one card).

### C. MoE dispatch facts — what the residency flip must respect

The step35 dispatch reality (verified in `moe_ffn_sequential_zq8`, `hybrid_forward.rs:2200+):

| arm | step35 verdict | why |
|---|---|---|
| `moe_ffn_pairs` (:3357) | DENIED | `sigmoid_router().is_none()` + clamp layers 43/44 |
| `moe_ffn_dev` (:3629) | DENIED | same predicate (`dev_ok`) — softmax-only device router |
| `moe_ffn_grouped` | env-OFF | `MEMRA_MOE_GROUPED` + m-dependent cuBLASLt router (lever C) |
| sequential loop + gdec fold | **RUNS** | gdec fires per token iff ALL 3x8 blocks SLRU-resident |

**CRITICAL FINDING: `m.dev_exps` (the fits-VRAM resident slabs, `build_dev_exps`
`hybrid.rs:227`) is consumed ONLY by the pairs/dev/gemma4 arms — every one of which is
DENIED for a sigmoid-router arch.** The sequential loop and both gdec folds
(`moe_gdec_token_q8` :3897, `moe_gdec_token` :3947, `moe_cached_gemm*` :4001+) read
exclusively from the per-Engine SLRU (`e.with_moe_cache`). So flipping the
`build_dev_exps` decision to "fits" on step35 would upload ~50 GB/card of slabs **that no
dispatched kernel reads**, while the SLRU keeps allocating beside them — double
allocation, zero dispatch change, near-certain OOM. The residency flip for the Step SKU is
therefore NOT `dev_exps`; the two honest shapes are:

1. **Teach the sequential q8 arm to read dev slabs** — `qmatvec_expert_q8` /
   `moe_gate_up_silu8_q8`+`moe_down8_fma_q8` over `slab_base + ex*stride` pointers instead
   of SLRU slot pointers. Same kernels, same bytes, same FP chains — pointer PROVENANCE
   only (the exact bit-identity class `moe_ffn_dev`'s resident arm documents vs its SLRU
   arm). gdec's pointer collection becomes unconditional (residency predicate = true by
   construction), the 37 GB/prime staging and the miss-path launches die.
2. **Per-stage SLRU convergence** — with the walker, each stage engine owns its own
   `moe_cache` (per-Engine field, `lib.rs:594`); a per-card budget covering the stage's
   ~50 GB share makes the SLRU fully resident after warmup and gdec always-fires. No new
   kernel paths, but only live WITH the walker (without it, all MoE compute runs on the
   primary engine and only dev0's SLRU exists).

- SLRU sizing: `MoeSlotCache::new` (`moe_cache.rs:467`) fills `MEMRA_MOE_VRAM_FRAC`
  (default 0.85) of free VRAM probed AFTER residents load; `MEMRA_MOE_SLOTS` forces N.
- Boot receipt (step-sku): 101.07 GB expert bank vs 94.96 GB budget → SLRU; ~96% per-block
  steady hit; P(all 24 blocks resident) ≈ 0.37 → 63% of token-layers take the ~49-launch
  miss path with H2D staging.
- `build_dev_exps` decision is a `static DECISION: OnceLock<bool>` — ONE decision for the
  whole model, probed on the engine that loads the FIRST MoE layer (= stage 0 under the
  sharded loader). **The train tip carries cx-503b `6afc4f65` "size resident experts per
  PP device"** — my branch base (80f47796) PREDATES it; rebase/merge before building
  increment 2 and read what it already changed (sizing is claimed; placement/decision flip
  is this lane's remainder).

### D. What the residency flip ALONE can and cannot buy (pre-registered, honest)

Where the peer traffic actually is (anatomy receipts):
- The routed experts are NOT peer-read today: SLRU stages host→dev0 and the expert kernels
  run on dev0 reading dev0 slots. The 22% peer-read tax is **stage-1 TRUNK tensors**
  (MMQ attn/proj weights, router `gate_inp`, shexp dp4a) dereferenced by dev0 kernels —
  killed only by running stage-1 layers on dev1 (the walker) — the flip does not touch it.
- What the flip kills: the 37 GB/prime H2D staging + eviction churn, the staged-miss
  launches (`qmatvec_expert_q8` n=255K ≈ 1.3 s + `quantize_q8_1` n=419K ≈ 0.55 s), and it
  makes gdec always-fire (49 → 3 launches/token-layer for the ~63% of tokens currently
  missing). It does NOT shrink the gdec m=1 pair cost itself
  (`moe_gate_up_silu8_q8`/`moe_down8_fma_q8` at n=161K ≈ 10.7 s — that is lever C's
  grouped-GEMM territory).
- Placement subtlety without the walker: stage-1 experts must stay dev0-resident (or
  SLRU) — uploading them to dev1 while compute stays on dev0 CONVERTS staged reads into
  peer reads (worse). The clean interim shape: resident slabs for STAGE-0's share on dev0
  (~50 GB fits beside the dev0 trunk shard), stage-1 keeps SLRU until the walker lands.
  cx-503b's per-device sizing may already express part of this — read it first.
- Honest projection: increment 2 alone moves cost #3's staging/miss fraction, not the 22%
  trunk tax and not the 28% m=1 dispatch core → expect ~141 → ~160-190, NOT the 400 class.
  The 400 class needs the walker (increment 3). Stated now so the measurement can refute
  the model rather than the narrative absorbing it.

### E. The split-vs-unsplit PRIME bit-identity gate (increment 1, registered RED)

Design (the ppn-gate/decode-batch-gate `--mode pp` shape, adapted to prime):
- **Reference arm = the unsplit walk over the SAME sharded load.** Prime deliberately has
  NO `refuse_unsplit_if_remote` (its unsplit walk is a 22% amortized tax, not the decode
  28x cliff) and MUST STAY callable — it is the gate's reference. Seam:
  **`MEMRA_PRIME_PP`** (mirrors `MEMRA_BATCH_PP`/`MEMRA_SPEC_PP`; default ON when the
  door+devices are open; `=0` = unsplit; read per call, never memoized).
- **Split arm liveness teeth:** bit-identity of two identical unsplit walks is vacuous —
  the gate must FAIL, not pass, while the walker is absent. The engine exports a split
  counter (`pp::PRIME_SPLIT_CHUNKS`, bumped once per chunk by the walker); the gate
  requires it to ADVANCE during the split arm. Walker absent ⇒ counter frozen ⇒ **RED**.
  This is the tickinv35 pattern: the gate exists and fails before the mechanism does.
- Comparison: full last-row logits bit-for-bit, the `[T, n_embd]` hidden stack
  bit-for-bit, h_seed, and a 24-step greedy stream, at chunk sizes {4096, 513, 256} (the
  chunkinv-composition arms), fresh `pp::new_cache` per arm, one process, one load.
- Implementation: new `ppsplit` mode in `concat-prime-probe` (already loads step35, owns
  chunkinv/tickinv/ppprime and the pinned prompts) + `tools/prime-split-gate.sh`
  (SKIP-if-no-model, teeth documented; canary = forcing `MEMRA_PRIME_PP=0` into the split
  arm must trip the liveness check once the gate is green).

### F. Box workflow notes (2x RTX PRO 6000, 18.195.123.14)

- Source tree `~/tokparity-memra` is **NOT a git repo** (rsync'd) and sits at `80b2ddf4`
  (lane/pp2spec-crash) **WITHOUT Lever A** (`grep MEMRA_STEP35_SWA_FA` = 0 hits). It
  belongs to the pp2spec co-tenant — do NOT clobber it. This lane uses its OWN tree
  (`~/leverb-memra`), rsync'd from this worktree, built there.
- Any Lever-B measurement against a tree without Lever A would baseline at the 90.9 floor,
  not the ~141 FA arm — sync first, then measure.
- Model `~/step37/models/step-3.7-flash/IQ4_XS/*.gguf` (3 shards + MTP Q8_0), prompt
  `~/step37/prompt-pp4096.txt`, raw logs `~/ppserve-raw/`, `.nsys-rep` in `/tmp` only,
  every GPU window under `flock /tmp/memra-gpu.lock` (decode-batch-gate co-tenant observed
  holding it 08-07). Box: 48 cores, 499 GB RAM, 189 GB free disk, cards ~3.2/97.9 GB used
  at read time. Config: `MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1`, fence [0,22,45].
- Prior-lane scripts to reuse: `research/pp-prefill-20260807/{anatomy-pp4096,leverA-gates*}.sh`
  (flock + tee + interleave shapes), `research/pp2-batch-20260806/run-ppbatch-*.sh`.

### G. Cache/KV facts (increment-0 reading, continued below at Increment 2)

- `pp::new_cache` (pp.rs:844) → `Cache::new_ppn` (`memra-kv/src/lib.rs:202`): stage-owned
  KV allocation already lands stage-1 KV on dev1 (serve worker + gates go through it since
  pp2-batch). The unsplit prime therefore peer-WRITES stage-1 KV appends and peer-reads
  them in FA views — both die with the walker, both are part of the 22% bucket's KV share.
- `concat-prime-probe` builds caches with plain `Cache::new` everywhere (~28 sites) — the
  ppsplit gate must use `pp::new_cache` for its arms (the pp2-batch "wrong-card KV"
  harness-bug class, found there in the server worker + bench).

---

## Increment 1 — the gate, registered RED (commits `eae50c02` + the rebase `cf3bbc3a`)

Rebased onto train tip `6afc4f65` (cx-503b's per-PP-device residency sizing `238beae0`
included). Landed: `ppsplit` probe mode (split-vs-unsplit bit-identity over last-row logits +
h_seed + the full [T,n_embd] hidden stack + teacher-forced decode steps THROUGH the primed KV,
per chunk size), the `MEMRA_PRIME_PP` seam, the `pp::PRIME_SPLIT_CHUNKS` liveness counter,
`tools/prime-split-gate.sh` (+canary), fast-gate rows `ppsplit`/`ppsplitc`, FLAGS.md row.

**RED receipt on the box** (`raw/inc2-battery-20260808T010302Z.log` G4): both chunk arms
report `logits diff 0 … hidden diff 0 … split_chunks ref=0 split=0 (need split >= 2/10)` →
`*** SPLIT-NOT-LIVE (bit-identity vacuous)` → exit 1. Exactly the designed shape: the
comparison machinery is proven able to read zero where zero is, and the verdict is RED
because the walker doesn't exist — not because bits moved.

## Increment 2 — the residency flip: slab-local MoE arm (commit `ec6bfad0`)

**The finding that reshaped this increment:** `dev_exps` (fits-VRAM resident slabs) is
consumed ONLY by `moe_ffn_pairs`/`moe_ffn_dev`/gemma4 arms — all of which DENY
sigmoid-router archs (step35/M3/Hy3). cx-503b flipped the Step SKU to per-device RESIDENT
(boot receipt in the battery: `PP dev0: experts 45.72GB … -> RESIDENT`, `PP dev1: 55.35GB
… -> RESIDENT`), so the train tip uploads ~101 GB of slabs **that nothing reads** on this
arch, while the SLRU — which sizes itself on free-VRAM-after-residents — is starved beside
them. What landed: a slab-local arm in `moe_ffn_sequential_zq8` — gdec's exact fused kernel
pair (`moe_gate_up_silu8_q8` + `moe_down8_fma_q8`) over `slab base + ex*stride` pointers
(no cache lock, no residency coin-flip, no miss staging), with the clamped layers 43/44 +
macro artifacts riding the same per-expert `qmatvec` twins from the slab. LOCALITY GATE:
`DevExps.dev == e.ctx().ordinal()` — a remote slab is never dereferenced (m=1 peer reads =
the 34-150x class); without the walker, stage-0 layers fire and stage-1 stays SLRU.
Seams: `MEMRA_MOE_SLAB=0` (provenance rollback), `MEMRA_MOE_GDEC=0` disables the fused pair
here too (and is the exactness localizer: with GDEC=0 both provenances run identical
per-expert kernels → the true provenance-only bit pair).

### Battery (box, one flock hold, `raw/inc2-battery-20260808T010302Z.log`)

| gate | verdict |
|---|---|
| G1 kernel-check model-backed FULL | **ALL GREEN** |
| G2 chunkinv35 naked (slab arm live) | **PASS** (INVARIANT, all 5 chunk sizes) |
| G3 run-gen over PP-2, naked | **MATCH** (prefill=decode argmax 6776, batched-prime MATCH) |
| G3b run-gen `MEMRA_MOE_SLAB=0` | **MATCH** (same argmax) |
| G4 prime-split-gate | **RED as registered** (SPLIT-NOT-LIVE, see increment 1) |
| G5 run-spec K=1..8 | **8/8 PASS**, acceptance digit-for-digit at the pin (14/17 = 82.4% K=1, flat-15 K=2..8) |

G3cmp harness bug (caught in the raw log, not repeated as a finding): the battery's logits
dump ran `run-gen` without `MEMRA_PP_ONLY`, which is the mode that actually writes
`MEMRA_PP_LOGITS` — `cmp` failed on a MISSING file, not differing bytes. The focused
provenance bit-cmp re-ran with `MEMRA_PP_ONLY=1` (4 arms: naked / SLAB=0 / GDEC=0 x both) —
see `raw/inc2-bitcmp-*.log`.

### Perf: pp4096, N=5 rep-major interleaved, one hold (medians of 5)

| arm | pp4096 tok/s (5 reps) | median |
|---|---|---|
| A naked = slabs + slab arm (the default after this commit) | 142.7, 140.7, 140.4, 140.7, 140.6 | **140.7** |
| B `MEMRA_MOE_SLAB=0` = dead slabs (**the bare train tip today**) | 137.0, 137.2, 136.9, 136.8, 136.8 | **136.9** |
| C `MEMRA_MOE_RESIDENT=0` = no slabs, big SLRU (the Lever-A state) | 141.0, 141.5, 141.5, 141.1, 140.8 | **141.1** |

Ranges are disjoint (B's max 137.2 < A's min 140.4 < C's overlap with A). Thermal regime:
30-37C, 2325-2400 MHz, cards 0 MiB before/after.

**Honest verdict:**
1. **cx-503b's residency flip is a ~3% pp regression on the Step SKU as merged** (B vs C:
   136.9 vs 141.1) — the slabs it uploads are dead on a sigmoid-router arch and starve the
   SLRU. This lane's slab arm recovers it (A vs B: +2.8%).
2. **The residency flip alone does NOT beat the Lever-A baseline** (A 140.7 vs C 141.1 —
   parity, slightly under). Consistent with the anatomy: the 37 GB H2D staging was already
   OVERLAPPED (0.83 s GPU-side of a 45.8 s wall), so killing it buys ~nothing at pp4096;
   the real MoE cost is the m=1 launch-pair dispatch (28%, lever C's territory), and the
   22% peer-read trunk tax needs the walker. The increment-0 §D projection (~160-190) was
   too optimistic — the measured answer is parity, and the pre-registered conclusion holds
   with more force: **increment 3 (the walker) is REQUIRED for the 400 class; the flip
   alone gets nowhere near it.**
3. Decode side effect (single-run, not a claim): G3 gen-only 18.74 tok/s (A) vs 16.12 (B).
   The slab arm can only help decode (no lock, no miss on stage-0 layers); a real decode
   receipt needs its own interleaved N=5 — deferred until after the walker.

### The provenance bit-cmp (follow-up to the G3cmp harness bug; `raw/inc2-bitcmp-20260808T013715Z.log`)

The battery's cmp slot compared a MISSING file (its run-gen invocation lacked
`MEMRA_PP_ONLY`, the mode that writes `MEMRA_PP_LOGITS`) — harness bug, fixed by a focused
4-arm rerun over the pp4096 prompt: naked vs `MEMRA_MOE_SLAB=0` (dispatch-class pair) AND
`MEMRA_MOE_GDEC=0` x both (the true provenance-only pair, identical per-expert kernels).
**All three comparisons: 0 differing bytes** — slab-vs-SLRU provenance exact, fused-pair vs
per-expert qmatvec exact, class pair exact at this prompt.

## Increment 3 — THE PRIME STAGE SPLIT (walker commit `564fb04d`), gate GREEN, 141 → 266 tok/s

What landed: `prime_layers` extracted from `prime_chunk` to the `decode_layers_eager(lo,hi)`
contract (range-local add+norm fusion; the S-glue capture path gated to the full range);
per-device prime slabs (HashMap on engine ordinal); shared `prime_chunk_epilogue` on the
last stage; `prime_chunk_ppn` — the decode/verify split structure verbatim (stage 0 embed +
range + tx; middle rx/range/tx; last rx + range + epilogue), with `fence_stages_behind` at
entry (#87), per-stage `pos_d`, `publish_to` at exit, and the `PRIME_SPLIT_CHUNKS` liveness
bump. `ppprime` now builds `pp::new_cache` caches (the wrong-card-KV harness class).

### Battery (`raw/inc3-battery-20260808T022651Z.log`, one flock hold)

| gate | verdict |
|---|---|
| G4 prime-split-gate | **GREEN**: `SPLIT BIT-IDENTICAL + LIVE (T=4883, chunks=4096,513, 8 decode steps)` — logits/h_seed/[T,n_embd] hiddens/teacher-forced decode all 0 differing bits, liveness counter advanced |
| G4c ppsplitc canary | **teeth proven**: forced-unsplit flips the verdict RED |
| G2/G2c chunkinv35 + canary | **PASS / teeth** (split prime live — chunk-invariance composes with the split) |
| G1 kernel-check FULL | **ALL GREEN** |
| G3 run-gen PP-2 | **MATCH** (argmax 6776, batched-prime MATCH) |
| G5 run-spec K=1..8 | **8/8 PASS**, acceptance digit-for-digit at the pin (82.4% K=1, flat-15) |

### Perf: pp4096, split vs unsplit, N=5 rep-major interleaved, one hold

| arm | pp4096 tok/s (5 reps) | median |
|---|---|---|
| S split (naked default) | 265.6, 266.3, 266.1, 266.4, 266.1 | **266.1** |
| U unsplit (`MEMRA_PRIME_PP=0`) | 141.5, 141.6, 141.5, 141.8, 141.4 | **141.4** |

**141 → 266 tok/s = 1.88x, spread 0.3%, ranges disjoint by 124 tok/s.** Thermal 30-37C,
2325-2400 MHz, cards 0 MiB before/after. The U arm reproduces the Lever-A baseline
(141.4 vs 141.1-141.5 across receipts) — same-binary, same-hold control.

Against the A+B projection (~420-450): the walker alone lands at 266 — the projection's
remaining share was (a) the residency flip's staging kill (measured ~nothing at pp4096 —
already overlapped; the slab arm's win shows up as stage-locality now that stage-1 MoE
dispatch runs on dev1's engine with its own SLRU/slabs) and (b) CHUNK PIPELINING
(stage-0 chunk N+1 under stage-1 chunk N — step 6, explicitly cut-to-follow-up material).
At 266 tok/s serial-split, the two stages are ~balanced (22+23 layers), so pipelining's
ceiling is ~2x → the 400-500 class is exactly its territory, consistent with the original
projection arithmetic.

### The serve receipt: 4k-prompt TTFT (`raw/ttft4k-20260808T025454Z.log`)

Serve-level, streaming, N=5 + 1 warmup per arm, one lock hold, spec OFF, drafter attached,
think-mode delta counting (the leverA-ttft4k shape verbatim):

| arm | 4k TTFT p50 | p95 | min-max |
|---|---|---:|---|
| SPLIT (naked walker) | **15.47 s** | 15.52 s | 15.46-15.55 |
| UNSPLIT (`MEMRA_PRIME_PP=0`) | 32.15 s | 32.34 s | 32.04-32.45 |

**2.08x on TTFT** — bigger than the probe's 1.88x because the worker primes in
`PREFILL_TICK_T=1024` chunks: at chunk_t=1024 the boundary/epilogue overheads amortize
differently and the unsplit walk's peer-read tax is per-chunk-constant. The unsplit arm
reproduces Lever A's 32.04 s receipt to within noise (same-binary control). The 4k TTFT
arc across the two lanes: **38.2 s (floor) → 32.0 s (Lever A) → 15.5 s (Lever B)**.

### What remains (the honest cut line)

- **Chunk pipelining (step 6)** — stage-0 chunk N+1 under stage-1 chunk N, the SGLang
  #33666 per-stage-budget + TRT-LLM #16170 drain-before-block laws. The remaining ~2x to
  the 400-500 class. Cut to a follow-up lane per the brief's "only if time to spare":
  the walker + gates + receipts consumed this lane's window, and pipelining wants the
  deferred-boundary design done carefully (the 2026-08-02 pipelined-arm flake history says
  this is not an evening bolt-on).
- **Decode over the split with the slab arm** — G3's single-run 18.74 vs 16.12 gen-only
  suggests the slab arm helps decode too; needs its own interleaved N=5.
- The serve worker's `MEMRA_SERVE_B1FAST` B=1 path and batched ticks already had their
  splits (pp2-batch); prime was the last unsplit serving path. `decode_step_dc` + graph
  capture remain fail-closed (unchanged, not on the serving path).
