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

### G. Cache/KV facts

- `pp::new_cache` (pp.rs:844) → `Cache::new_ppn` (`memra-kv/src/lib.rs:202`): stage-owned
  KV allocation already lands stage-1 KV on dev1 (serve worker + gates go through it since
  pp2-batch). The unsplit prime therefore peer-WRITES stage-1 KV appends and peer-reads
  them in FA views — both die with the walker, both are part of the 22% bucket's KV share.
- `concat-prime-probe` builds caches with plain `Cache::new` everywhere (~28 sites) — the
  ppsplit gate must use `pp::new_cache` for its arms (the pp2-batch "wrong-card KV"
  harness-bug class, found there in the server worker + bench).
