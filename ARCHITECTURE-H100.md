# bw24 on H100 — the serving lane (sm_90a)

Companion to `ARCHITECTURE.md` (the sm_120 laptop engine). This document is the
architecture for the **H100 serving lane**: multi-tenant, batched, lane-scheduled
serving with every kernel driven to its sm_90a wall. It folds in three ground-truth
maps (crate architecture, full kernel inventory, darklanes lane design) and the
2026-07-25 engine-decision bench. Perf claims follow the repo law: N=5 medians,
kernel-check + argmax gates before any measurement.

Validation box: AWS p5.4xlarge (1×H100 80GB SXM, driver 595.71, nvcc 13.1),
`ssh ubuntu@13.203.16.133` (darklanes OPS.md holds lifecycle + dead-man switch).

---

## 0. Hard facts (what changes vs the sm_120 doc)

1. **sm_90a is the inverse of sm_120.** H100 HAS: wgmma (async warpgroup MMA),
   TMA, 228 KB smem/SM opt-in, clusters, FP8 tensor cores via cuBLASLt, 132 SMs,
   3.35 TB/s HBM3. H100 LACKS: the sm_120a FP4 block-scale mma kinds
   (`mxf4nvf4`, `kind::f8f6f4` is sm_100a+ — f8f6f4 REFUTED on 90a by ptxas,
   commit aa8b51d). Everything gated on `mxf4/f8f6f4` stays dead here; everything
   gated on tcgen05 stays dead everywhere (that's sm_100a).
2. **The portable int8/bf16 MMA already in-tree runs on sm_90a.**
   `mma.sync.m16n8k32.s8`, `m16n8k16.s8`, bf16 `m16n8k16`, `ldmatrix`,
   `cp.async` are sm_80-class PTX. The 90a build disables them by **Rust-side
   gate only** (build.rs "portable boot" decision), not silicon:
   - `legacy_quant_gemm_allowed = !portable` → `crates/bw24-engine/src/lib.rs:65-70`
   - `mmq_supports` → `false` on portable → `crates/bw24-engine/src/mmq_ffi.rs:194`
   - `try_fp8_gemm` → `Ok(None)` on portable → `crates/bw24-engine/src/fp8_ffi.rs:96`
   - FA prefill → `sdpa_naive` (single-thread softmax oracle) → `lib.rs:5580-5583, 5625-5631`
   - GDN chunked prefill compiled out → `cu/hybrid.cu:420`
   - NVFP4-W4A8 → link stub on portable → `build.rs:118-121` (main arm is
     m16n8k16.s8 — candidate for re-enable; f8f4 arm stays dead)
3. **Measured baseline (2026-07-25 bench, Qwen3.5-9B Q8_0, N=5 medians):**
   decode 181.2 tok/s (dp4a, 51% of 352 tok/s weight wall) — already 101% of
   vLLM 0.26 w8a8 single-seq. Prefill ~398 tok/s vs vLLM 35,000 (88×): that gap
   is the fact-2 gates, i.e. the documented dp4a-fallback class ("298 vs 1413
   pp512" — lib.rs:5620-5624), not a kernel-quality mystery.
4. **The engine is architecturally B=1.** One `Cache` ("Single sequence",
   cache.rs:4), one recurrent GDN state (hybrid.cu:3 "n_seqs=1"), FA decode Q =
   `[head_dim, n_head, 1]` (flash_attn.cu:1779). bw24-server exists (axum +
   GPU-worker thread) but multi-sequence = time-interleaved B=1 steps
   (worker.rs:277-288, MAX_ACTIVE=4, private cache per session). Token-batching
   exists only along ONE sequence (spec verify m=2..16, prefill m≥16) — the
   m-tier dispatch in `Engine::matmul` (lib.rs:3894) is the seam cross-sequence
   batching rides.
5. **Spec-exactness law carries over**: decode must stay bit-identical to the
   sequential oracle at every m (the verify tier already obeys it). Batched
   paths inherit the law; graphs bake device counters (decode.rs:17-23) so
   batch geometry changes mean re-capture, never silent reuse.
6. **The lane model is measured, not aspirational** (darklanes B2/D1/D4):
   interactive = protected + IS the SLO sensor; judge (prefill-shaped 2048/4)
   sheds at 100% of SLO; harvest (decode-shaped 64/256) sheds at 90%. Shed
   happens OUTSIDE the engine (429 + Retry-After), never queued inside. The
   headline defect a native engine must fix: vLLM's ONE global chunked-prefill
   budget (2048) taxes baseline interactive p99 11.6→40.6 ms even at zero
   parasite load. Native answer = **per-lane prefill budgets**.

---

## 1. Target: what "done" means

An engine that on one H100:
- serves interactive streams at p99 TPOT ≤ 50 ms while judge+harvest lanes ride
  the dark compute, with **per-lane admission and per-lane prefill budgets
  native in the scheduler** (not proxied through a global knob);
- decodes batched: B sequences share every weight read (QKV/O/FFN as m=B GEMM),
  attention per-sequence over a paged KV pool;
- prefills through tensor cores (target TTFT ≤ 300 ms @ 2048 tok, ≥ 60× today);
- decode single-seq reaches 95-98% of the 352 tok/s weight wall via the wgmma
  lane (334-345 tok/s, = ~1.9× vLLM B=1);
- every kernel in `cu/` has an sm_90a tuning verdict (kept / re-tuned / replaced)
  with before/after N=5 numbers;
- shared machinery (KV pool, lane scheduler, sampling, launch helpers,
  validation harness) extracted into crates the sm_120 lane can also consume.

---

## 2. Phase A — un-gate the portable tensor-core paths (days, ~5× prefill)

No new kernels. Flip the fact-2 gates behind an arch predicate
(`portable_boot` → `hopper_mma`) and validate each step on the box:

| Step | Change | Expected effect | Gate |
|---|---|---|---|
| A1 | Allow `qmatvec_gemm_q8_0` (int8 m16n8k32.s8) on 90a: lib.rs:65-70 + gemm_supports lib.rs:5629-5645 | prefill matmuls leave dp4a class (repo analog: 298→1413 pp512) | kernel-check + argmax MATCH + pp512 N=5 |
| A2 | FA prefill bf16 mma path on 90a (lib.rs:5580-5583): `fa_prefill_f32_pp`/`_qw` replace sdpa_naive | attention prefill leaves single-thread-softmax class | fa-sanitize, fa-hd128-check, argmax |
| A3 | GDN chunked prefill kernels compile on 90a (hybrid.cu:420) | hybrid-arch prompts chunk-scan instead of tokenwise | run-hybrid logits vs llama.cpp |
| A4 | MMQ static lib on 90a (mmq_ffi.rs:194; keep 120a-only quants stubbed) | Q4_K/Q5_K/Q8_0 prefill via MMQ where profitable vs A1 | mmq gates + N=5 A/B vs A1 |
| A5 | cuBLASLt FP8 on 90a (fp8_ffi.rs:96) — H100 is the first-class FP8 arch | F8 checkpoints prefill via Lt (620-795 TF measured on 5090; H100 ceiling ~2× higher) | gemm-check vs cpu_linear, fp8 gates |
| A6 | smem/launch audit of un-gated kernels for 132 SM / 228 KB (occupancy re-tune only where ncu shows a wall) | free margins | ncu evidence per change |

Order A1→A2 first (Q8_0 bench model exercises both); A4/A5 after since they
need quant-specific or checkpoint-specific traffic. Every step lands with the
Rust predicate split: `portable_cuda` (89) vs `sm90a` (90a, portable + hopper
MMA subset on) so 89 stays honest.

**MEASURED (2026-07-26, commit 1b7cdad8, H100 box, N=5):** A1+A2+A3 landed as
one flip (`bw24_hopper_mma`). kernel-check ALL GREEN, 34 policy tests pass.
Prime 2048 tok: **5.15 s → 0.230 s (22×, ~8.9k tok/s prefill)** — TTFT target
(≤300 ms) met before any wgmma work. Decode 179.4 tok/s median — unchanged
(path untouched, as predicted). Prefill now ~25% of vLLM's 35k tok/s (was
1.1%). A4 (MMQ A/B) and A5 (FP8 Lt, needs an F8 checkpoint) remain open;
run-spec n/a on this checkpoint (no MTP head).

## 3. Phase B — the batched, lane-scheduled serving engine (the core build)

### B1. Paged KV pool (replaces single-sequence Cache)
- Block-paged KV: 32-token blocks, per-layer pools, quantized as today
  (K=q8_0, V=q5_1 — formats preserved so append/decode kernels change
  addressing, not math). Block table per sequence: `[seq][layer] -> Vec<block_id>`.
- `Cache` splits: `SeqState` (pos, block table, GDN/conv state, sampler state)
  vs `KvPool` (shared blocks + free list + per-lane accounting).
- Per-lane accounting on the pool IS the admission currency: lane quotas in
  blocks; harvest evictable (its contract tolerates requeue — shed-first lane).
- Recurrent layers (GDN/conv ring) are per-sequence dense state — they move to
  `SeqState` wholesale, `[B]`-indexed batched variants of the state kernels.
- Snapshot/rollback (spec) becomes block-table clone + refcount, not byte copy.

### B2. Batched step (replaces worker.rs round-robin)
One engine step = one fused pass over the active set:
1. gather per-seq tokens → `[B]` token ids (embed gather widened, decode.rs:83-94);
2. batched QKV/gate/FFN/lm_head: existing matmul m-tier with m=B rows —
   the weight stream amortizes across sequences (THE bandwidth win; the m=2-9
   fused-matvec tier and m≥16 GEMM tier already exist for spec verify);
3. attention per sequence over block tables: fa_decode gains a sequence axis
   (blockIdx.z or per-seq launch on the same stream — decide by ncu; split-K
   geometry per seq as today, combine unchanged);
4. batched KV append (`_rows` variant generalizes: per-row (seq, pos, block) triple);
5. batched sampler: extend `spec_sample.cu` Philox machinery to `[B]` rows
   device-side; argmax already has partial/final split — widen to row-major B.
   Kills the per-step full-vocab D2H (decode.rs:505) — return B sampled ids +
   B logprob scalars, not B×vocab floats.
6. Exactness ladder: B=1 batched path must produce bit-identical output to
   today's decode_step (extend `graph-decode-gate`); B>1 verified per-sequence
   vs isolated runs (worker.rs's own "byte-identical to isolated" contract).

### B3. Lane scheduler (yieldgate invariants, native)
Per-step admission planner replacing MAX_ACTIVE round-robin:
- Three queues (interactive / judge / harvest). Interactive always admitted to
  the step; judge admitted while measured interactive p99 < 100% SLO; harvest
  < 90%. Shed = immediate 429 at the server edge (never engine-queued) —
  identical thresholds to yieldgate.py:33 so the sidecar stays compatible.
- **Per-lane prefill budgets**: each step carries token budgets
  {interactive: unbounded-first, judge: J tok, harvest: H tok}; judge/harvest
  prefill chunks fill leftover step capacity only. This deletes the global-knob
  tax (the 11.6→40.6 ms baseline p99 cost of vLLM's single budget).
- True per-step timing: the engine records per-step decode latency per lane —
  replaces the sidecar's network-gap estimator with ground truth; exported at
  /yield/metrics-compatible endpoint.
- Preemption: harvest sequences preempt at block-pool pressure (evict via
  block-table park; resume = re-admit); judge chunks are naturally preemptible
  between chunks. Interactive never preempted.
- bw24-server keeps the axum surface + adds `x-lane` header intake (default
  interactive), SSE cadence unchanged (sidecar contract: `data:` chunks +
  `[DONE]`).

### B4. Serving integration
- Existing worker-thread model stays (one GPU thread, cmd channel); the
  scheduler owns the step loop; sessions become SeqState handles.
- Streaming path unchanged for clients; lane metrics endpoint added;
  graceful backpressure = shed responses at the edge.

**MEASURED (2026-07-26, commits 855162ae/f42fd94e/05f90270, H100 box):**
- decode_step_batch v1+v2: exactness battery ALL GREEN (strict bit-identity
  under equalized composition; config-mode argmax authority + bit-checked
  B=8 isolation). B=8 aggregate 306 tok/s (2.10×); remaining scaling gap =
  per-seq state-kernel launches → pooled state + blockIdx.z batching (B1 lane).
- Lane server LIVE: 14 concurrent sessions (4 interactive + 4 judge 2k-prompts
  + 6 harvest) — interactive p99 32.6 ms < 50 ms SLO at full mixed load,
  per-lane accounting exact, shed path 429, /yield/metrics engine-truth.
- **Serving-regime B-curve (ctx=512, N=3 medians, 2026-07-26):** B=1 148.8 /
  B=2 236.2 / B=4 367.8 / **B=8 487.2 tok/s aggregate (3.27×)** — the earlier
  306 figure was a short-prompt artifact (prompts under fa_vec_min_tkv profiled
  the f32 attention fallback at 21% of step time; nsys). BW24_BVAR sweep at
  m=8: auto≈base (490 vs 489) — the picker is fine; the residual efficiency
  gap (~30-35% of peak BW in mmvq_b8) lives inside the b8 kernel (ncu next).

**LANE BATTERY (2026-07-26, the B2-style demo record):** four scheduler defects
found by measurement and fixed (fixed-chunk stalls → per-tick stall bounds;
estimator starvation blind spot → sentinel; interactive cap serializing clients
→ cap follows batched capacity; decode-only estimator → full-tick TPOT +
SLO-headroom-adaptive dark chunks). Final NINT=12 battery: interactive p50 flat
41.5 ms at judge rates 0-8, p99 bounded 42-75 ms, zero starvation; dark lanes
duty-cycle honestly (judge 24 admitted/1348 shed at saturation). Envelope: 12
streams saturate the 50 ms SLO at today's decode ceiling — dark yield is real
at NINT≤8 (measured 15-18 ms TPOT + thousands of judge tok/s); widening it is
exactly the Phase C decode-wall work.

## 4. Phase C — the wgmma lane (the 3-4× decode headroom)

New kernels, guarded `bw24_hopper_mma`, tuned for 132 SM / 228 KB smem / HBM3:
- **C1 decode GEMM** (the prize): weight-streaming int8 wgmma
  (`wgmma.mma_async.m64nNk32.s8`) with TMA bulk loads + 3-4 stage smem
  pipeline, warp-specialized producer/consumer. Target 95-98% of the 352 tok/s
  wall single-seq (334-345 tok/s); at batch B the same kernel serves m=B rows.
  Validation: argmax MATCH vs dp4a path per shape; N=5 decode-bench A/B.
- **C2 prefill GEMM**: wgmma twin of qmatvec_gemm (int8) — chases whatever gap
  remains after Phase A (A1 may already sit near the roofline for m≤2k; ncu
  decides if C2 is worth it before building it).
- **C3 FA-3-class attention**: prefill mainloop on wgmma bf16 + TMA KV loads;
  decode split-K stays GEMV-shaped (bandwidth-bound; wgmma irrelevant there) —
  decode attention tuning is smem/cp.async/occupancy work, not MMA work.
- **C4 kernel-by-kernel sweep** (task 9): every kernel in cu/ gets an ncu pass
  on the box; keep/re-tune/replace verdict recorded in a table appended to this
  doc. Small kernels count (norm/rope/append/router): they set the non-GEMM
  floor that caps batched-step latency.

## 5. Phase D — shared extraction

What becomes shared crates (consumed by both the sm_120 laptop lane and this one):
- `bw24-kv`: block pool, block tables, quantized append/dequant views;
- `bw24-lanes`: lane types, admission policy, per-lane budgets, step planner
  (pure host logic — reusable over any backend, including the darklanes sidecar
  as an out-of-process fallback);
- `bw24-sampling`: host sampler + device Philox sampling (already
  graph-safe) behind one trait;
- `bw24-cuda-util`: launch helpers, smem calculators, fatbin loading, ncu
  hooks (today duplicated across engine/probe);
- `bw24-validate`: kernel-check harness generalized (CPU references, tolerance
  policy, batched-path B=1..32 equivalence gates, N=5 protocol runner).
Extraction happens per-phase as pieces stabilize, never speculatively.

## 6. Validation protocol (all phases)

- Correctness: kernel-check ALL GREEN on box before any bench (repo law);
  argmax MATCH per engine change; spec gates (run-spec K=1..8) stay green on
  paths they cover; batched: B=1 bit-identity + per-seq equivalence at B∈{2,8,32}.
- Perf: N=5 medians, interleaved A/B where comparing engines; ncu evidence for
  every tuning claim; decode-bench + pp512 + TTFT tracked in a results table
  per phase.
- Lanes: interference.py --lanes against the served engine (judge/harvest rate
  sweeps); acceptance = B2/D1-class yields at ≤ their p99 cost, minus the
  global-knob baseline tax (the native scheduler's whole point).
- Fleet: H100 box per darklanes OPS.md; dead-man switch active; every session
  logs cost.

## 7. Task map

Tasks tracked in-session: 2=this doc, 3=B1, 4=B3, 5=B2, 6=A1-A6(+C2 if needed),
7=B4, 8=C1, 9=C4, 10=D, 11=validation harness, 12=end-to-end lane demo.
Sequencing: A first (cheap, unblocks everything), B1→B2→B3→B4 as the core
build, C1 parallel to B once A validates the toolchain, C4+D continuous.

**Q8_0 split-plane result (2026-07-26, commit fec8f234):** ALL GATES GREEN incl
strict bit-identity through the mirrors (249 tensors). Serving curve: B=2
236→266, B=4 368→420 (+14%), B=8 487→**526 tok/s (3.34×, 2.9× vLLM
single-seq)**. m=1 essentially flat (183.6) — the m=1 kernel's per-warp walk
was already sector-coalesced; its remaining gap to the 352 tok/s wall is
latency-bound (next: multi-block ILP / cp.async staging / mr2-style rows —
the continuing task-9 lane, with wgmma prefill as task 8).
