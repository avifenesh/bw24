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

**Tuning-lane ledger (2026-07-26, all N≥2 measured on the box):**
| probe | verdict |
|---|---|
| Q8_0 split-plane mirrors | **+14% B=4, +8% B=8, +2.8% m=1** — kept (fec8f234) |
| Q8_0 mr2 (q4_0 recipe) | −8% on H100 (132-SM grid math) — default mr1, kernel behind seam |
| MMVQ_ROWS 2/8 | kernel-check FAIL (cross-kernel invariant) — blocked by gates |
| KV_PREFETCH | +0.6% (noise) — not taken |
| batched-state pointer-array kernels | perf-neutral, launch hygiene kept — states pooled for future |
| prefill GEMM tiles (NSTAGE/BM/BN/K1_*) | ALL FLAT at 0.230s — GEMM is not the prime bound; profile says fa_prefill_f32_pp (415µs/call) + gdn_chunk pass are |
| MMQ vs qmatvec_gemm prefill | tie (0.231s both) |

Next frontier (evidence-ranked): (1) fa_prefill bf16 mainloop perf on H100
(415µs/call — wgmma/TMA candidate, task 8's real target, NOT the int8 GEMM);
(2) m=1 latency class (186→352 wall: cp.async staged weight ring);
(3) fa_decode_combine batching (64 launches/step at B=8);
(4) remaining shared extraction (bw24-validate, bw24-sampling, bw24-kv).

**m=1 ncu verdict (2026-07-26, qmatvec_q8_0_mmvq_rp on H100):** issue every
6-9 cycles, 0.16-0.29 eligible of ~7.5 active warps/scheduler (49% occupancy,
grid-limited on small out_f shapes) — long-scoreboard stall on the weight
stream. k-split (rpks) is BANNED (decode==verify FP-order law); the legal
lever is the rpca recipe (cp.async double-buffered weight ring, same
accumulation order = bit-identical): port q8_0_mmvq_rp -> _rpca next.
Expected class: NVFP4 rpca's long_scoreboard fix; target is the 250-300
tok/s band en route to the 334-345 wall goal (which likely also needs the
fused-launch family widened to rp).

**rpca probe (2026-07-26):** Q8_0 m=1 cp.async ring measured −2% (181.8 vs rp
185.5) — smem staging costs more than it hides for 8-bit direct-dp4a. Opt-in
seam kept. m=1 stands at 186 (~57% of achievable BW); the remaining gap is
per-shape latency work (grid fill on small out_f, register-window ILP) and
the non-matvec per-token floor — next probes need per-shape microbenches
(tools/bench_mapped_qmatvec.cu pattern), not whole-engine A/Bs.

**m=1 mystery RESOLVED (2026-07-26, per-shape microbench + graph A/B):**
tools/bench_q8_shapes.cu isolates the rp matvec per trunk shape: 97-100% of
peak on square shapes (4096x4096), 80-90% on wide (11-12k out, lm_head), 66%
on the sub-wave attn qkv, launch-floor on beta/alpha(32). Kernel times sum to
~297 tok/s — the kernels are essentially AT the wall. The 186 e2e gap is
~370 launches/token of gap overhead + per-token D2H sampling. PROOF: the
in-tree CUDA-graph decode (generate_graph) measures **214.1 tok/s (+14%)**
with zero new code. The m=1 road to 334-345: graph decode as the serving
default (graph-decode-gate covers bit-identity) + device-resident sampling
(argmax_token_device / spec_sample machinery exists) to kill the D2H — an
INTEGRATION lane now, not a kernel lane. Wide-shape 80% + attn-qkv 66% are
the only true kernel work left in the m=1 stack.

**g2 probe (2026-07-26):** sub-wave grid-fill twin kept (bit-identical, wins
in isolation on the 66%-peak qkv shape) but e2e-neutral at N=3 resolution —
the shape is 4 launches/token. Confirms the ledger rule: e2e A/Bs resolve
nothing below ~2%; per-shape microbench is the instrument. The one remaining
>5% m=1 lever is the GRAPH-SERVING integration (+14% measured standalone):
worker sessions on generate_graph + device sampling — an integration arc.

**Load-policy A/B (2026-07-26, per-shape tool):** __ldcs streaming beats plain
ldg by 13-37% on all weight-heavy shapes (+1.2% ldg only on lm_head) — load
policy exonerated. Wide-shape 80% decomposes as wave-quantization tail (5.8
waves at 3072 blocks ≈ 14%) + residual; next testable fix = persistent-CTA
row loop (grid = exact fill, per-row program unchanged = legal). ffn_down at
95.7%, squares at 97%, lm_head 90% — five of eight shapes effectively AT wall.

**Persistent-CTA probe (2026-07-26):** REFUTED — −11% wqkv, −12% ffn, −31%
ffn_down, −9% lm_head (stride-loop overhead + lost locality exceed the wave
tail). m=1 matvec per-shape survey COMPLETE: squares/down AT wall (96-100%),
lm_head 90%, wide 79-82% (best-known after ldg/persistent/rpca all refuted),
qkv 66% (g2 wins in isolation, e2e-invisible), beta/alpha launch-floor
(already dual-fused in the m=1 chain). The kernel family is surveyed; the
m=1 frontier is CONFIRMED as integration (graph serving + device sampler,
+14% measured floor) — kernel-side, only the FA-prefill mainloop remains.

**Graph-serving design pinned (2026-07-26):** generate_graph is whole-generation
(owns its Cache, primes internally — decode.rs:742) — routing lanes through it
whole-request blocks the scheduler tick ~1.2s/request = interactive p99
destroyed; DISQUALIFIED for concurrent serving. The +14% (and the device-
sampler D2H kill) therefore requires the step-wise refactor: a GraphSession
API — capture per (fa_vec, split) bucket ONCE against a session's resident
counters/cache, replay ONE step per scheduler tick, recapture on bucket cross.
The capture-region constraints are already solved in generate_graph's body
(event-tracking off, stable pointers, bucketed t_kv) — the refactor lifts that
loop body into a stepping struct. THIS is the single next arc for m=1-to-wall.

**GraphSession landed (2026-07-26, commit 7479e9ec):** step-wise CUDA-graph
decode — gate PASS (token stream == generate_graph exactly), **233.8 tok/s vs
179.3 eager (+30.4%)** = 66% of the 352 wall, from 53%, with zero kernel
changes. Confirms the integration thesis. Remaining m=1 ladder: worker wiring
(single-interactive-session policy), then multi-step replay between D2H syncs.

**Step decomposition (2026-07-26):** fa_apply 1µs + launch 25µs + gpu 4.25ms —
the graph step is 99% GPU-bound; multi-step replay KILLED pre-build (would
save 0.6%). Gap to wall = ~1.4ms/step of in-graph non-matvec GPU work (norm
rows ~0.5ms, attention+combine, state ops, argmax-248k, embed). Next
instruments: nsys graph-node timing; next levers: further norm-chain fusion,
combine batching, device argmax split tuning. Session A/B improved to +34%
(234.0 vs 174.5 eager on the 48-tok-prompt shape).

**Graph-serving LIVE (2026-07-26, commits b90cae99..190ba549):** single-session
lane serving real HTTP at **217 tok/s for long generations** (512 tok; +20% vs
eager serving, 62% of wall end-to-end incl. HTTP+prime). Promotion gated at
BW24_GS_MIN=384 tokens (capture+snapshot ~340ms one-time = ~330-tok
break-even); degrade-to-batched live-validated with 3 concurrent clients
(correct output through the cache handoff). Serve-mode flags OnceLock'd,
metrics publish throttled. Remaining serving polish: ~0.2s fixed per-request
setup on the short path; capture-cost reduction would move break-even down.

**Task-8 opening measurement (2026-07-26, OPEN):** two nsys attempts at a
single-2048 prime profile captured run-gen on the TOKENWISE prime path
(245k m=1 matvecs = 2048 stepwise) — run-gen's batched-prime branch engages
under conditions not yet located (bench_bw24's own timer reports 0.230s
batched prime for the same invocation shape). Before the wgmma FA build:
find the branch, profile the true batched prime, size FA's share. The
wgmma payoff bound is unknown until then — do not build blind.

**Task-8 scope CORRECTED by clean profile (2026-07-26, BW24_PP_ONLY):** pure
batched-prime kernel shares: **MMQ int8 GEMM 60.1%** (688µs/launch), split-
plane build 9.5% (load-time artifact), **fa_prefill_f32_pp 9.3%** (2.65ms/call
× 8 full-attn layers), gdn_chunk 6%+. The earlier "FA mainloop dominates"
read came from a contaminated mixed capture — task 8's wgmma target is the
PREFILL GEMM (m≥16 int8 class, both MMQ and qmatvec_gemm tie today), worth
up to ~2.4× on prime if wgmma reaches its int8 ceiling; FA prefill is the
secondary 9% target. Measurement discipline saved building the wrong kernel
a second time.

**wgmma arc OPENED (2026-07-26):** tools/bench_q8_gemm_wgmma.cu — standalone
dev harness (synthetic Q8_0, CPU reference, timing) + v0 kernel
(m64n64k32.s8 warpgroup mainloop, per-block scale fold). ptxas ACCEPTS wgmma
on the 90a build (the build.rs "separate later lane" is open). v0 status:
illegal memory access on first launch — smem descriptor encoding (LBO/SBO,
canonical no-swizzle layout) is the live bug; fragment row/col mapping
unverified until it runs. Iteration cycle: 30s standalone compile+run.
Target: beat the 688µs int8-mma class = up to 2.4× on prime.
Debug hint for the v0 fault: plain row-major smem does NOT match wgmma's
canonical no-swizzle operand layout — the tile must be arranged in CORE-MATRIX
order (8-row × 16B cores; A 64×32 = 8 M-cores × 2 K-cores; store core(m,k)
at ((m*2+k)*8 + r)*16, then LBO/SBO encode inter-core strides: try LBO=128,
SBO=256 first, then the transposed pairing). Verify against PTX ISA §9.7.14
asynchronous-warpgroup-level matrix shape/layout tables before trusting signs.
v0 UPDATE (same day): core-matrix smem order + LBO=128/SBO=256 → kernel RUNS:
**169µs = 4.1× the MMQ class raw** (101.6 int8 TOP/s, unpipelined, correctness
still FAIL rel~1e2 — the s32 D-fragment row/col mapping needs the exact PTX
ISA table; speed already proves the arc: 60%-of-prime × 4 ≈ prime 0.23→0.13s
before pipelining). Next mechanical step: fix fragment mapping (PTX ISA
wgmma.mma_async D layout, m64nN), verify vs CPU ref, then cp.async pipeline.
Layout search verdict (160+ combos, best rel 12.6, none pass): the descriptor
stride space cannot express the fix — B arrives token-major but wgmma consumes
K-major B; the smem WRITE arrangement for B must perform the transpose into
the canonical K-major core order (and/or SW128 swizzle), not the descriptor.
Next: derive from PTX ISA §wgmma matrix-layout tables + a known-good open
int8 wgmma kernel, write the ONE correct arrangement, verify vs CPU ref.
The 4.1× raw-speed result stands — only operand plumbing remains.
v0 status (end of stretch): B transpose-in-write confirmed directionally
(error class changed; gather costs 169→411µs, still under MMQ 688 — v1 will
quantize activations directly into K-major, killing the gather). Exact core
arrangement still wrong after ~170 tested combos — STOP guessing: the fix
requires the PTX ISA wgmma matrix-layout table (§9.7.15) or cross-reading a
known-good open int8 wgmma kernel (marlin-hopper / FA3 source). Both are
fetchable references; one derivation, one verify run. Everything else in the
arc (harness, CPU ref, fragment mapping, 4.1× speed proof) is in place.
**v0 CORRECT (2026-07-26, the arc's breakthrough):** rel err 1.6e-05 OK at
**179µs = 3.84× the MMQ class**, unpipelined. Root causes (via Colfax/PTX
canonical reference): (1) the missing `fence.proxy.async.shared::cta` between
generic-proxy smem writes and async-proxy wgmma reads — THE correctness bug
all along; (2) original core-matrix arrangements were canonical (A and B both
core(i,j) = i*SBO(256) + j*LBO(128) + row*16, K-major no-swizzle) — the
"transpose fix" detour was wrong. Path to integration: k-slice pipeline
(cp.async or TMA), Q8_0 dequant-fold epilogue check vs qmatvec_gemm bit
policy, engine dispatch at m>=16, kernel-check case, prime A/B (expect
0.23s -> ~0.10-0.13s = prefill toward 60-80% of vLLM).

**wgmma engine integration + the honest verdict (2026-07-26, task 8 closed
for the exact path):** the "3.84×" was a BASELINE ERROR — the harness's
"688µs MMQ ref" was a pp2048-shape figure; the real per-launch MMQ medians at
m=512 (nsys per-shape, grid-dim split + duration clustering) are: wqkv
4096→12288 253µs, mid→8192 168µs, square 4096² **84-99µs**, ffn_down
11008→4096 ~247µs, gate/up 4096→11008 236µs, small→1024 82µs.

Kernel ladder (all CORRECT, rel ≤5e-5 vs CPU ref; engine gate rel ~3e-7 vs
MMQ on real GGUF tensors, kernel-check ALL GREEN):
- v0 unpipelined 64×64: 516µs wqkv — e2e pp512 3845 vs MMQ 8692 tok/s (2.26× LOSS).
- v1 64×128 single-acc: 246 regs → occupancy collapse, worse than v0.
- v2a dual-acc via acc[2][32] runtime index → LOCAL MEMORY (256B stack): 1116µs. Fixed
  by pair-unrolling (acc0/acc1 static): 157 regs, 459µs.
- v3 = v2 + TRANSPOSED activation scales ([blk][tok], coalesced 16B cp.async): 414µs.
  Untransposed engine layout costs ~15% (measured) — any engine port stages a
  transposed twin.
- v4 = 2-warpgroup ping-pong, shared A tile, N=128/CTA, __launch_bounds__(256,2):
  wqkv 344µs, square 116µs, small 72µs. NSTAGE/LA sweep flat (4/2 fine).
- CEILING PROBE (fold law lifted, scale_d=1 full-K s32): wqkv 313µs vs 120µs W-traffic
  floor — even fold-free, the n64-class pipeline only reaches MMQ's level.

ROOT CAUSE (architectural, not tuning): Q8_0's per-32-block scale fold reads
wgmma accumulator registers every 32-K step; ptxas C7514 "wgmma serialized due
to non-wgmma instructions reading accumulator registers" — the dual-acc
overlap is compiled out, the warpgroup tensor pipe drains every block. The
Ampere-style mma.m16n8k32 path (vendored MMQ) tolerates per-block folds via
warp-level ILP; DeepGEMM-class fp8 kernels live with per-128 folds (4× fewer)
plus SASS-level interleaving. Per-32 exact int8 block-scale GEMM on Hopper
belongs to mma, not wgmma.

VERDICT: v4 wins ONLY out_f=1024 (72 vs 82µs, ~0.8% of prime). BW24_WGMMA=1
opt-in seam kept; MMQ stays default (N=5 pp512: MMQ 8692 vs wgmma-v0 3845 —
law holds). Correctness stays pinned: kernel-check wgmma case is cfg-gated.
NEXT SWING (prefill is still 60% MMQ at 178µs avg): non-exact numeric-config
probe — fp16-dequant GEMM (resident fp16 mirror + cuBLASLt, or in-kernel
dequant + fp16 wgmma full-K f32-accum, which streams with ZERO mid-loop acc
reads). Precedent: BW24_PP_FP8 (620-795TF) / BW24_FP4 opt-in seams; gate =
argmax battery + logit tolerance, not bit-identity.

**FP16-mirror prefill — the fold-free swing (2026-07-26, PROMOTED default on
the Hopper lane):** the wgmma verdict pointed the way: the per-32 fold law is
the wall, so lift the operand format instead of fighting the pipe. Probe
(tools/bench_lt_f16.cu): cuBLASLt FP16 TN runs 611-687 TF at the m=512 model
shapes = 3.2-3.7x the MMQ class per launch. Engine arm (BW24_PP_F16,
fp8_prefill.cu pattern): resident fp16 dequant mirror of every 2D Q8_0
projection built device-side at load (f16_ffi::build_q8_f16, budget
BW24_PP_F16_BUDGET_MB default 32GB, layer-order prefix), dispatch in
matmul/matmul_pre m>=16 arms AFTER fp8, BEFORE MMQ. Decode (m<16) untouched —
decode==verify law holds by construction.

Numeric config (explicit, gated): int8 part exact in fp16 (7 mantissa bits
into 11); rounding at d*q products + activation f32->fp16 cast (NO per-32
rescale on the act side). Battery: kernel-check f16 case rel <= 6.5e-3
(band 1e-2) ALL GREEN; run-gen argmax MATCH p1/p2/p3; greedy streams
IDENTICAL to MMQ config on all three; serving smoke (600-tok prompt) coherent.

MEASURED (N=5 medians, 9B-Q8_0):
- pp512:  8674 -> 15626 tok/s (+80%)
- pp2048: 8260 -> 14543 tok/s (+76%)  [the old "0.230s prime" -> 0.141s]
- VRAM: 18.1 -> 31.8GB served (mirror ~2B/w on 2D Q8_0 trunk) — 80GB lane feature
- validate-h100.sh ALL GATES GREEN; batch curve unchanged (decode path untouched)

Default ON under bw24_hopper_mma (BW24_PP_F16=0 reverts to MMQ; =1 forces on
smaller rigs at their own VRAM risk). MMQ stays the exact-config fallback and
the portable default. Next prefill ceiling: FA prefill 9.3% + GDN chunk
kernels now dominate prime — new profile needed for the next target.

**matmul_group / convert-once (2026-07-26, kept, perf-NEUTRAL):** grouped the
7 shared-activation call-site families (GDN 4-tuple, attn q/k/v, ffn gate/up)
through `matmul_group` — the f16 arm converts the activation once per group
(saves ~160 cvt launches/prime). Measured pp512 15626 -> 15391 -> 15545 across
runs = one noise band; the gap clusters were NOT cvt-bound (they sit at layer
boundaries before rms_norm — host submission cadence, not launch count).
Kept: no regression, and one-xh-per-group is the shape a future prefill graph
capture wants. Honest-neutral, recorded per the launch-collapse precedent.

Prime anatomy at 15.5k tok/s (per-prime, nsys): fp16 GEMMs ~10ms @660TF
(near HW for Lt-heuristic), GDN chunk family ~5.9ms, elementwise+norms ~5.4ms,
FA prefill f32 ~3.2ms, launch gaps ~7.5ms over ~1030 launches. vLLM prefill
ref 35k tok/s (w8a8 int8 GEMMs ~1300TF class + fused epilogues). Next arcs
ranked: (1) GDN chunk kernels to their wall (18%), (2) FA prefill tensor-core
port (10%, FA3-class arc), (3) norm/add fusion + prefill graph capture for the
gap structure (22%), (4) cutlass int8 per-row epilogue GEMM (the vLLM numeric
config) if (1)-(3) exhaust.

**Prime grind round 2 (2026-07-26, after the f16 promotion):** pp512
15626 -> 16679-16839 tok/s across three landed changes + two refuted probes:
- zeros->uninit on 47 full-overwrite prefill buffers (+4.5%): nsys cuda_api_sum
  showed 1230 memsets/trace + cuMemAllocAsync pool-miss tails (med 980ns,
  avg 22us) as the layer-boundary gap fuel. state_in/out keep semantic zeros.
- elementwise float4 wave: silu_mul + add + f16 cvt (+2.7%): bit-identical per
  element; decode/graph token-exact gates green. Norm reductions untouched
  (sum order pinned by decode==verify).
- matmul_group convert-once: neutral, kept (launch structure).
- REFUTED: Lt algo autotune (neutral + cold-start cost, reverted); GDN chunk
  C sweep re-validated shipped default C=32 on the new profile (C=128 tanks 32%).
Launch gaps 22% -> 12-18%; launches 1028 -> 910/prime.

REMAINING ARCS (each multi-day, evidence attached in this ledger):
1. GDN K4 (gdn_chunk_state) runs ~9TF f32, smem+serial-bound over NC chunks —
   a bf16-mma rewrite of its C x D GEMM steps is the chunked-config upgrade path.
2. FA prefill is ALREADY the bf16-mma FA-2 floor port (P0a/P0b/C4 arcs) at
   ~400-540us/launch — next step is an ncu-driven stall analysis, not a rewrite.
3. Prefill CUDA-graph capture (gap floor ~12% remains).
4. cutlass int8 per-row-epilogue GEMM (vLLM's numeric config) if 1-3 exhaust.
Standing vs vLLM: decode 101%, prefill ~48% (16.8k vs 35k) — was 1.1% at boot.

**FA prefill ncu diagnosis (2026-07-26, arc #2 sharpened):** fa_prefill_f32_pp
at T=512: SM throughput 3.6%, memory 9.2% — pure latency exposure. ncu full:
255 regs/thread (the P0a/P0b Q-in-reg + O-in-reg design) -> Block Limit
Registers = 2, theoretical occupancy 12.5%, ACHIEVED 6.25% (grid 128 CTAs on
132 SMs, <1 wave, 4 warps/SM effectively); 67% of stall cycles = long
scoreboard on the synchronous f32->bf16 K/V stage-to-smem. mma m16 pins 16
query rows/warp, so occupancy can't come from smaller tiles — the fix is
HIDING the latency: software-pipeline / double-buffer the K/V staging (same
tiles, same math order -> bit-identical, no new numeric config). GDN K4 by
contrast measures 59.6% memory SOL at 130us — much closer to its config's
wall; its upgrade stays the bf16-mma rewrite (numeric-config class).
FA pipeline arc projected: 625us -> plausibly 150-250us/launch = +8-10% pp512.

**Phase D extraction CLOSED (2026-07-26):** the shared-crate scoreboard —
- bw24-lanes ✓ (lane types, admission, budgets — serving consumes it)
- bw24-sampling ✓ (host sampler + device Philox behind one trait)
- bw24-validate ✓ (protocol core: maxdiff/rel banding, deterministic pr vectors,
  N-median runner, GateTally ALL-GREEN contract; 4 gate bins ported verbatim;
  fa_sanitize's 16-bit pr variant deliberately stays local — different vectors)
- bw24-kv ✓ (dual cache + KV format policy behind the KvDev seam. The documented
  "Engine-trait blocker" dissolved on measurement: the cache uses exactly 7
  device ops — zeros/uninit/alloc_u8/htod_i32/clone_dtod/copy_into/set_i32_one —
  so the seam is that trait, not an engine-wide abstraction. Engine impl
  delegates to inherent methods; every call site unchanged via re-export.)
- bw24-cuda-util ✗ REFUTED as speculative: bw24-probe's "duplication" is 64
  lines of raw cudarc idiom appropriate to a probe; no shared body exists.
  Extraction law ("never speculatively") applies.
All gates green on box after each move (validate-h100 + graph-session token-exact).

**FA prefill W2 probe (2026-07-26, REFUTED at T=512):** the 2-warp/32-row CTA
variant (grid.x x2, bit-identical per-row math) measured pp512 15452 vs 16677
default — the doubled K/V staging traffic + halved per-CTA warps cost more
than the coverage gain. THIRD refuted FA hypothesis (after MINBLOCKS 3/4):
this kernel is at a real local optimum; the remaining path is the FA3-class
producer/consumer redesign (TMA staging + warp specialization). Seam kept
(BW24_FA_PP_W2=1): at dark-lane chunk sizes (T=256 -> grid 64 CTAs, half the
SMs idle) W2's coverage argument doubles — an untested SERVING-side hypothesis
the lane battery should arbitrate, not pp512.

**W8A8 per-row arc REFUTED BY PROBE (2026-07-26, tools/bench_lt_i8.cu):** the
"vLLM numeric config" (int8 W per-row x int8 act per-token, s32 GEMM + f32
dequant epilogue) measures NET 0.87-1.04x vs the SHIPPED fp16 mirror at every
m=512 shape: int8 IMMA only reaches 654-892 TOP/s here (not the 2x-of-fp16
datasheet ratio), and the epilogue pass (5-19us) eats the remainder. VERDICT:
the fp16-mirror path is at the practical H100 GEMM ceiling for this workload —
no dtype change buys more; a fused-epilogue cutlass kernel would at best
reclaim the 5-19us epilogue, which the probe shows is not worth the arc.
STRATEGIC COROLLARY: vLLM's remaining prefill edge (35k vs 16.7k) is
STRUCTURAL, not GEMM-rate — fused epilogues/norms into GEMMs, TC attention,
fewer launches, graphs. The remaining roadmap is exactly the three structural
arcs: FA3 staging redesign, GDN K4 bf16-mma, prefill graph capture.

**GDN K4 bf16-mma arc — KERNEL PROVEN (2026-07-26, tools/bench_gdn_k4.cu):**
v2 measures **68.3us vs the shipped f32 K4's 119.4us (1.75x)** at the real
dims (H=32, T=512, C=32, D=128 — harness calibrated: v0 port reproduces the
engine's ~130us). Design: M state lives in mma accumulator FRAGMENTS across
all chunks (16 f32/thread; bC fold = register scale); step A (Y = U - W.M)
and step B (M += ys^T.k) are m16n8k16 bf16 warp-tiled GEMMs (FA's proven
ldmatrix helpers); W and k arrive PRE-CONVERTED bf16 (engine: K3 casts W on
store for free; k gets one mirror pass) through a 2-deep cp.async ring —
the probe showed synchronous global->bf16 staging was 72us of v1's 133.
Numerics: operands round to bf16 per chunk, state carry stays f32 —
rel 1.3e-2 vs CPU ref on hostile (exploding) synthetic data; a change WITHIN
the gated chunked prefill config (BW24_GDN_CHUNKED), arbitrated by the
BW24_GDN_DIFF oracle + argmax battery on adoption.
Debug ledger: (1) B operands must be [n][k] in smem — Mb mirror goes NATURAL
[col][i] (no transpose!), kb needs ld_A_trans (both are FA GEMM0/PV patterns);
(2) remaining known slack: Ssnap fragment-scatter ~15us (stage via smem later).
Engine integration next: K3 bf16-W store, k bf16 mirror, BW24_GDN_MMA seam,
oracle + battery. Projected prime: K4 2.8 -> ~1.6ms (+4% pp512).

**GDN K4-MMA engine integration (2026-07-26): landed OPT-IN, promotion gated
on a state-carry battery.** Engine kernel gdn_chunk_state_mma (hybrid.cu,
k4mma helpers) + f32_to_bf16_bulk mirrors + BW24_GDN_MMA seam (C==32 only).
Battery: pp512 16694 -> 17286 (+3.5%), argmax MATCH x3, greedy streams
identical, oracle out mean_rel ~1e-4. HOWEVER the kernel-check f64-truth
STATE pin (2.5e-4) reads 4.25e-1 under mma on hostile synthetics — bf16
rounding accumulates in the recurrent state, which feeds decode and session
continuation; 16-token battery windows cannot rule out long-generation drift.
DEFAULT STAYS f32-chunked (its tight pin untouched); the mma config got its
OWN kernel-check pin (out<8e-2, state<8e-1 — regression guard, measured
4.30e-2/4.25e-1). PROMOTION CRITERION: long-context chunked prime -> long
decode + multi-turn continuation battery showing no stream drift vs f32.
Ragged-T edge verified clean in harness (T=200/488 in band).

**GDN K4-MMA PROMOTED (2026-07-26, state-carry battery green):** promotion
criterion met — 2048-token prime (64 in-kernel state carries) -> 256 greedy
decode tokens IDENTICAL to f32 on 3 seeds; chunked-continuation prime
(BW24_PRIME_CHUNK=512, 4 cross-call carries through cache.recur) IDENTICAL on
2 seeds. Default ON on the Hopper lane; kernel-check pins BOTH configs by
forcing the seam env per case (f32 tight band + mma 8e-2/8e-1 band).
pp512 DEFAULT: 16694 -> 17240 tok/s. Night cumulative: 8674 -> 17240 (+99%).
Remaining task-9 arcs: FA3 staging redesign, prefill graph capture.

**GDN K5-MMA + coupled pair (2026-07-26): LANDED default, pp512 17786.** K5
(gdn_chunk_output) followed the K4 playbook: harness v1 (mma, f32 sources) was
correct but NEUTRAL (62->62us — K5 is staging-bound like K4 was); v2 with
bf16 sources + cp.async double-buffer hit 35.3us (1.78x). Engine adoption
coupled the pair: K4-mma writes Y/Ssnap directly in half precision (their only
consumer is K5-mma, which rounds anyway — no extra numeric hop, half the
traffic). FIRST coupling attempt in bf16 FAILED its own config pin (out
2.19e-1 vs 8e-2 band: K4 error compounding through K5's bf16 rounding) —
switched the coupled channel + K5 operands to FP16 (11 mantissa bits): pin
back to 4.28e-2 in band, pp 17786. mma m16n8k16.f16 = same rate as bf16;
ld_A/ldmatrix are type-agnostic b16 moves.
decode-batch-gate note: the config-mode gate1 (decode_step_batch vs
decode_step_h, step-16 threshold) flipped at step 1 under the mma prime —
STRICT bit-gate + gate2 still PASS, so decode is untouched; the mma-primed
state shifted near-tie logits on the gate's fixed prompt. Fixed by PINNING the
gate's prime config to f32 (the gate tests DECODE configs; prime is a nuisance
variable there — doc'd in the bin).
Battery (coupled pair default): validate-h100 ALL GREEN, graph-session
token-exact, state-carry IDENTICAL x2 seeds, 3-prompt streams IDENTICAL to
f32, argmax MATCH. Night cumulative: pp512 8674 -> 17786 (+105%).

**FA BF16-KV staging (2026-07-26): +11% pp512, BIT-IDENTICAL, default ON.**
The FA3-lite the ncu evidence was pointing at all along: the kernel already
rounds K/V to bf16 during staging (64 scalar f32 loads + converts per thread —
the 67%-of-stalls long-scoreboard). Pre-converting K/V to bf16 mirrors
(f32_to_bf16_bulk, 2 launches per fa_prefill call) feeds the SAME
__float2bfloat16 values into the SAME mma -> outputs bit-identical (verified:
argmax + logit maxdiff lines byte-equal across arms); staging becomes 8 int4
vector copies per thread. fa_prefill_bf16kv_pp twins (body templated on
BF16KV); BW24_FA_BF16KV=0 reverts. pp512 17777 -> 19718 (+11%).
NIGHT CUMULATIVE: 8674 -> 19718 (+127%); prefill now ~56% of vLLM's 35k.
The three refuted FA occupancy probes stand; the remaining FA headroom is the
full FA3 producer/consumer redesign (cp.async ring on the now-bf16 tiles is
the next increment — the mirrors make it a plain byte ring, no convert).
Remaining arcs: FA cp.async ring, prefill graph capture, K2/K3/conv GDN passes.

**FA cp.async ring on bf16 tiles (2026-07-26): +0.85% (pp512 19886),
bit-identical.** The 2-stage ring prefetches tile k0+BK behind the current
tile's mma (only copy TIMING changes — bit-check byte-equal). The vectorized
bf16 staging had already absorbed most of the stall; the ring takes the rest.
FA slice now effectively at its non-redesign wall. pp512 night cumulative:
8674 -> 19886 (+129%); prefill ~57% of vLLM. Remaining structural arcs:
prefill graph capture (gap floor), full FA3 warp specialization (diminishing
vs the above), remaining GDN small kernels (conv/cumgate/solve/attn).

**Round-9 wall audit (2026-07-26, anatomy at 19.9k):** per-prime: GEMMs 10.3ms
(practical ceiling — W8A8 + autotune refuted), GDN state-mma 1.56 + output-mma
0.85 (post-mma walls), FA 1.07 (was 4.3 — bf16kv+ring), conv 0.95, norms+cvt
~3.2, gaps 3.7ms (15%). Probes this round:
- ssm_conv1d_gdn float4: NEUTRAL, reverted — the kernel's wall is the GQA
  broadcast WRITE amplification (~25MB materialized q/k copies), not tap loads.
  Mapped option for later: de-broadcast layout (k/q stay [T, num_k, 128];
  consumers map vh -> vh % num_k) — touches K2/K4/K5 mirrors, prefill-only.
- Norm reductions (l2/rms, ~1.7ms) stay pinned: any load-width change reorders
  the reduction tree and the SAME kernels serve decode (decode==verify law).
THE remaining structural arc is prefill graph capture (3.7ms gap floor);
everything else measured at or near its wall for this design generation.

**TASK-9 KERNEL SWEEP CLOSED (2026-07-26):** every prime-path kernel measured
at or near its wall, each with landed wins or refutation evidence: GEMMs (fp16
mirror @660TF; W8A8 + Lt-autotune refuted by probe), FA (bf16kv + ring, 4x;
three occupancy probes refuted; FA3 full rewrite = diminishing), GDN K4/K5
(mma coupled pair, 1.75x/1.78x, fp16 channel), conv (scatter-wall, float4
refuted, de-broadcast mapped), norms (decode-law-pinned), elementwise (float4
wave landed), launch diet (uninit conversion). Serving-lane measurement closed
the attribution: scheduler+HTTP cost 7%; the remaining 2x to vLLM serving
prefill is cross-request prefill CONCATENATION (their continuous batching runs
bigger GEMM m) — scheduler work, not kernels. OPEN ARCS (tracked as tasks):
cross-request prefill batching (decode_step_batch pattern applied to prime)
and prefill graph capture (15% gap floor). Night: pp512 8674 -> 19886 (+129%),
bench-shape prefill 398 -> 18659 (47x), TTFT 5.15s -> 0.119s, decode 102% vLLM.

## Task #13 design — cross-request prefill batching (the measured 2x)

WHY (measured 2026-07-26): serving prefill 17.3k vs vLLM 35k; scheduler costs
only 7% — the whole remaining gap is GEMM batch size (vLLM concatenates
prefill chunks across requests; our tick primes one request at a time; nvjet
at m=512 runs ~660TF and larger m climbs toward the fp16 ceiling).

DESIGN (decode_step_batch precedent, applied to prime):
- New `prime_cache_batch(e, prompts: &[&[u32]], caches: &mut [&mut Cache])` in
  hybrid_forward. Embed CONCATENATED tokens [sum_T, n_embd]; per layer:
  * BATCHED on the concat buffer (one launch, m = sum_T): rms_norm chains,
    qkv/out/gate-up/down projections (matmul_group / f16 GEMMs), elementwise,
    ffn. These are token-parallel — rows independent.
  * PER-SEQ on contiguous VIEWS of the concat buffer (offsets = prefix sums,
    zero copies): rope (positions restart per seq), QK-norm is token-parallel
    (safe either way), FA prefill, conv+GDN chunk stack (chunk kernels take
    per-seq T), KV quantize-append, per-seq last-token logits.
- Worker tick: collect up to BW24_PRIME_BATCH (default 4) pending interactive
  FRESH prefills (continuation primes stay single — the suffix arm is
  session-stateful); dispatch prime_cache_batch; per-seq TTFT emitted as each
  seq's logits land.
- NUMERIC CONFIG: batched GEMM at m=sum_T tiles K differently than per-seq m
  -> NOT bit-identical to single primes (same class as every prefill GEMM
  change). Gate: batch-vs-sequential ARGMAX equality per seq on the prompt
  battery + logit-band + the standard batteries. Decode untouched (per-seq
  caches identical structure after prime).
- GATE BIN: prime_batch_gate — N prompts, prime individually vs batched,
  compare per-seq prefill argmax + decode-16 streams + state maxdiff bands.
- EXPECTED: GEMM m 512->2048 lifts the 10.3ms GEMM slice toward the ceiling
  (+15-25% aggregate prefill), amortizes the per-tick fixed costs; stacks
  with task #14 (graphs) toward the 35k lane target.

**Task #13 increment 2 (2026-07-26): prime_cache_batch LANDED + regime mapped.**
Driver: trunk (embed/norms/adds/ffn/projection groups) on CONCAT tokens
(m = sum_T); stateful mixer cores per seq on split projection buffers (D2D).
prime-batch-gate ALL GREEN (uneven lengths: per-seq prefill argmax MATCH +
16-step decode streams MATCH vs individual primes). MEASURED REGIME (N=5):
  B=8 T=64:  5034 -> 9073 tok/s  (+80.2%)
  B=8 T=128: 8741 -> 13009 tok/s (+48.8%)
  B=4 T=128: 8859 -> 12783 tok/s (+44.3%)
  B=4 T=512: 17980 -> 16326 tok/s (-9.2%)  <- per-seq m already at plateau;
    split/gather copies eat the margin. Crossover ~T=256-384.
POLICY: batch prefills when per-seq T <= BW24_PRIME_BATCH_MAX_T (default 320),
else single-prime — exactly the serving mix (chat prompts are mostly short;
the aggregate serving prefill number was measured on 937-token prompts, i.e.
the UNFAVORABLE side; the favorable side was previously WORSE than measured).
Next: worker tick integration + serving re-measurement.

**Task #13 increment 3 (2026-07-26): worker batch-prime LANDED, +21.6% serving
throughput at the short-prompt load.** Tick phase (b) collects fresh short
interactive prefills (T in [PRIME_MIN_T, BW24_PRIME_BATCH_MAX_T=320], same
model, budget-fitting) and dispatches prime_cache_batch in ROUNDS of up to
BW24_PRIME_BATCH=4; a lone fresh candidate holds up to
BW24_PRIME_BATCH_HOLD_MS=4 so staggered arrivals coalesce (TTFT cost <= 4ms).
Debug ledger (telemetry-driven): v1 fired on only 25% of a 32-concurrent burst
(arrivals staggered across ~1ms ticks) -> the hold; then a tick with 8 pending
batched 4 and SINGLE-primed the rest -> rounds. Final: 98% of burst tokens
batched (14288/14592). Serving A/B (96 x 152-tok, conc 32, max_tokens=1):
off 7971 -> on 9689 tok/s (+21.6%). Long prompts (937t > MAX_T) byte-unchanged
(17188 vs 17189). prime-batch-gate + validate-h100 ALL GREEN.

**Task #14 verdict (2026-07-26): the gap floor is NOT launch-count-bound —
graphs or acceptance.** Norm+cvt fusion landed (rms_norm_f16out emits the
GEMM's fp16 operand in the norm epilogue — BIT-IDENTICAL, byte-checked vs the
pre-fusion build; kills ~64 convert launches/prime + their re-read traffic)
and measured NEUTRAL (19,872 = the 19.9k band), exactly like matmul_group's
convert-once before it. TWO independent launch-diet passes now show the 3.7ms
gap floor does not shrink with launch count at this granularity — the residual
is per-launch host cost x remaining ~700 launches plus legitimate inter-op
drains. Reclaiming it means TRUE prime graph capture (single cuGraphLaunch;
the pointer-table generalization) — the one remaining structural arc — or
accepting the floor. Fusion KEPT (bit-identical, less traffic, groundwork:
the fp16-operand plumbing is what a graphed prime wants anyway).
All gates green (validate, graph-session, prime-batch); serving short-burst
9334 tok/s (9689 band).

## Task #14 design v2 — prime graph capture IS tractable (pad-to-bucket analysis)

The 50-kernel device-length estimate was WRONG. With prompts PADDED to a
bucket (graph shape fully static), the causal structure absorbs the pads:
- FA prefill: real query i attends keys <= i — all REAL rows. Pad-query
  outputs are garbage and DISCARDED. No kernel change.
- GEMMs/norms/elementwise: token-parallel; pad rows compute garbage, harmless.
- GDN recurrence is the crux — pads would UPDATE state. But the update law is
  state' = exp(g)*state + beta*(...): forcing beta[pad]=0 AND g_log[pad]=0
  makes pads IDENTITY steps. ONE tiny mask kernel (reads true_len from a
  device int) after the beta/g_log producers.
Device-length spots (the ONLY dynamic scalars): (1) that beta/g mask,
(2) conv ring writeback must take rows [true_len-pad, true_len) not the pad
tail — device-int variant of the ring update, (3) KV len_d finalize = host
memcpy after replay (rows beyond true_len sit inert past len), (4) last-token
logits row = device-index gather before lm_head.
Session pointers: the fresh-prime graph touches the cache ONLY via KV append
(K, V, len_d), conv ring, ssm_state/alt — ~5 ptrs x 32 layers = one ~1.3KB
device pointer TABLE, memcpy'd per replay (kernels index the table; the
decode pointer-table precedent is decode_step_batch's u64 tables).
Capture: once per (bucket, model) at server start (~340ms x buckets
{128,256,384,512} ~ 1.4s startup); replay = memcpy table+tokens+len, ONE
cuGraphLaunch. Waste <= 25% on bucket padding (policy: nearest bucket >= T;
below 128 the batch-prime path already dominates).
Gate: graphed prime vs eager prime — per-seq argmax + decode-stream + state
maxdiff battery (prime-batch-gate pattern). Prize: the 3.7ms/prime gap floor
(~15%) + host freed for scheduling.

**Task #14 implementation state (2026-07-26, next increment = capture smoke):**
`Engine::capture_graph_retained` (the decode-graph machinery) takes the prime
closure as-is ONCE a capture-safe prime variant exists. Constraints audited:
- prime_chunk's `e.htod_i32(&pos)` must HOIST out (H2D inside capture bakes a
  node reading a dropped host Vec — replay UAF); pos_d becomes a param.
- the tail `e.dtoh(&logits)` must go device-resident (return CudaSlice; the
  worker reads it post-replay) — same for h_seed/hidden.
- `cache.pos += t` is HOST state — advance outside the closure.
- KV append position: the batched append takes pos as a HOST int — baked 0 is
  CORRECT for fresh-prime graphs (the only graphed class; continuation stays
  eager).
- warmup pool stability + capture_keep retention already handled by
  capture_graph_retained (draft-graph precedent).
INCREMENT ORDER: (1) prime_chunk_captured (capture-safe copy, ~70 lines) +
smoke bin proving the ~900-launch prime captures + replays byte-equal on a
fixed cache (riskiest unknown: cuBLASLt fp16 + cp.async under RELAXED capture
— Lt plan cache is warm after warmups, no events in the call);
(2) the 4 device-length pieces + beta/g pad mask; (3) pointer table variants
(append/conv/gdn-state); (4) per-bucket capture at server start + worker
replay path + prime-graph-gate. Prize: 3.7ms/prime (~15%) + freed host.

**Task #14 SMOKE GREEN (2026-07-26): the prime graph WORKS.** prime-graph-smoke
at T=512: capture INSTANTIATES in 13ms (the 340ms fear was the decode-session
snapshot machinery, not capture itself); replay 23.3-24.0ms vs eager 25.7ms
(**+10% immediately, the gap-floor reclaim**) with logits maxdiff 0.000e0 —
BIT-IDENTICAL. Debug ledger for the arc:
1. set_i32_one is a SYNCHRONOUS host memcpy — capture-illegal. Fresh len_d=0
   goes through a memset node instead.
2. Warmups polluted the recurrent state (warmup 2 primed as a continuation)
   and overflowed KV host-lens — fixed by BAKING fresh-prime semantics into
   the graph head: memset nodes zero conv ring + ssm_state(+alt) + len_d and
   host lens reset per closure entry. This is the CORRECT per-replay behavior,
   not a workaround.
3. Retaining an in-capture allocation across end_capture -> INVALID_VALUE at
   instantiate (and AUTO_FREE would UAF it anyway). GRAPH-OUTPUT CONTRACT:
   results copy into caller-preallocated stable buffers; every transient
   drops inside the capture (alloc+free node pairs). prime_chunk_captured
   signature carries the contract.
4. capture_graph_retained's keeper path also trips on the prime; the smoke
   uses manual staged capture — the serving wrapper will too.
REMAINING for serving: per-bucket capture at boot, session pointer TABLE
(this smoke bakes ONE cache's pointers), pad-to-bucket + the 4 device-length
pieces, worker replay path + prime-graph-gate. Replay math: 512/23.3ms =
21,973 tok/s pp512-equivalent (+10.3% over eager 19,922).

**Task #14 design v3 (2026-07-26, post-smoke): COPY-OUT beats the pointer
table.** Economics: one graph per bucket binds a DEDICATED scratch cache;
after replay, D2D-copy the outputs into the session's cache — KV rows [0,T)
(~12MB), conv rings + ssm states (~64MB) ≈ 25-50us total vs the 2.3ms the
graph saves. ZERO kernel changes, ZERO cudaGraphExec node patching (the
160-param patching alternative costs ~320us/session-switch AND graph_update
surgery; copy-out wins on both simplicity and cost). AUTO_FREE LAW (smoke
finding 3 corollary): in-graph transient ADDRESSES recycle per launch — every
replay-consumed output MUST be copy-noded into a stable buffer inside the
graph; the copy-out sources are exactly those stable buffers plus the scratch
cache's resident state.
PAD-PROOFING (the 4 pieces, insertion points pinned):
1. gdn_pad_mask(beta_raw, g_log, len_d, H, T_bucket) — tiny kernel; insert in
   linear_attn_prime_core between the g4 pops and gdn_glog/sigmoid consumers
   (zeroed beta + g_log make pad rows identity state-steps).
2. conv-ring writeback from device true_len (the fresh prime's ring must hold
   rows [len-3, len), not the pad tail) — device-int variant of the ring
   update call in linear_attn_prime_core.
3. row_gather_device(hn/x, len_d) for h_seed + hlast (the smoke gathers row
   T-1 host-side — padded graphs need the true last row).
4. len_d/host-len finalize post-replay (host memcpy, already trivial).
Worker path: buckets {128, 256, 384, 512} captured at boot (13ms each) on the
scratch cache; fresh prime with T <= 512 routes: pad x_in to bucket, memcpy
tokens' embed rows + len_d := T, replay, copy-out, host lens := T. The
prime-batch path (task #13) keeps priority below T=320 at batch >= 2; graphs
serve the singles. Gate: prime-graph-gate (graphed vs eager prime: argmax +
decode-stream + state maxdiff — the prime-batch-gate pattern).

**Task #14 PAD-PROOF GREEN (2026-07-26): the engine core is COMPLETE.**
len_d threaded through the captured prime (gdn_pad_mask after the beta/g
producers, ssm_conv_ring_update_dev writeback, row_gather_dev for
h_seed/hlast — eager paths byte-unchanged via Option<None>). Smoke at
bucket 512 / true_len 300: replayed logits maxdiff 0.000e0 vs the EAGER
300-token prime — pads are provably invisible (identity GDN steps, causal
attention, discarded pad rows), and even the m=512-padded GEMMs bit-match the
m=300 eager run at these shapes. Exact-length case still bit-identical.
Capture 13-15ms; replay 23.2-24.0ms. Remaining = the serving WRAPPER only:
PrimeGraph { graph, scratch cache, x_in/pos_d/len_d/outs } per bucket at
boot, copy-out into session caches (design v3), worker route below the
batch-prime threshold, prime-graph-gate formalization.

**Task #14 gate status (2026-07-26): padded replays GREEN end-to-end; exact-
bucket-length has ONE pinned defect.** prime-graph-gate (eager-vs-graphed
prime + copy-out + 16-step decode streams): T=47/128/300 all MATCH through
decode; copy-out fidelity byte-exact (session==scratch 0.000e0). Findings:
- CONTROL (eager-vs-eager, pool-shifted): streams MATCH — eager is address-
  robust; the graph arm is its own numeric config (keeper-era pool layout ->
  baked Lt addresses -> a different valid rounding; the keeperless smoke
  measured 0.000e0, the retained capture does not).
- Gate convention adopted: prefill argmax MATCH hard + decode divergence
  before step 12 fails (decode_batch config-gate precedent). Smoke-prompt
  T=512 drifts at step 14 = accepted class.
- REMAINING DEFECT: T == bucket with prompt-A diverges at STEP 2 with conv
  max 4.9e-1 (prompt-B: 4.3e-2 / step-14) — prompt-dependent magnitude at
  exact length only; conv-implicated; a real localized bug, NOT the drift
  class. Next bisection: per-LAYER conv/ssm diff at T=512 + T=511 (one pad)
  to isolate the t==bucket edge.
- KEEPER LAW confirmed for primes: capture without the retained keeper leaves
  graph-baked pool addresses reissuable (the earlier corruption class).
SAFE SERVING SUBSET AVAILABLE NOW: replay only for T < bucket (strict pads) —
all green; exact-length primes stay eager pending the fix.

**Task #14 defect hunt — hypothesis kill-list + the live lead (2026-07-26):**
The graphed prime's cache diverges from eager by percent-scale conv values,
deterministically per (prompt, T), while streams still MATCH at all padded
lengths. Killed by experiment: (1) uninit-reads — re-zeroing all 33 prime-path
buffers left every diff BYTE-IDENTICAL; (2) pool-address corruption — keeper
vs keeperless produced identical diffs; (3) copy-out — session==scratch at
0.000e0 everywhere; (4) alignment-lottery — eager-vs-eager pool-shifted
control streams MATCH; (5) launch-stream race — CudaGraph::launch uses the
capturing stream. LIVE LEAD (order experiment): adding an UNRELATED eager
prime BEFORE capture changed a LATER case's outputs (argmax MISMATCH, conv
max 9.8e-2 -> 1.13e1) — replay numerics depend on pre-capture history =
GRAPH-BAKED SHARED MUTABLE STATE. Prime candidate: the engine-resident
f16_scratch (xh activation + 64MB Lt ws) — its pointers bake into the graph's
cvt/Lt nodes while every EAGER f16 GEMM mutates the same buffers between
replays. FIX CANDIDATE (next unit, one file): give the captured prime a
PRIVATE f16 scratch (per-PrimeGraph xh/ws, threaded like len_d) — removes the
sharing entirely. Serving policy meanwhile: graphs stay OFF; the eager+
batch-prime stack (all green) serves.

**Task #14 CLOSING VERDICT (2026-07-26): prime graphs are blocked by
cuBLASLt's address-variant numerics — mechanism identified, reclaim path
priced.** The global discriminator (BW24_DEBUG_ZERO_ALLOCS=1: memset EVERY
engine allocation) left all diffs BYTE-IDENTICAL — contents-independent,
allocation-LAYOUT-dependent. Seven hypotheses tested across the arc (uninit
x2 scopes, pool corruption, keeper, copy-out, alignment-lottery control,
launch stream, shared f16 scratch — the private-scratch isolation MOVED the
diffs, confirming layout sensitivity, but did not remove them). MECHANISM:
Lt/nvjet algos contain pointer-alignment-specialized variants; a baked layout
that differs from eager's seeds ~1e-3 GEMM deviations which the 32-layer
recurrent GDN gating chain AMPLIFIES to percent-scale state diffs — the
keeperless smoke's exact 0.000e0 was the one layout where capture reused the
eager warmups' pool slots verbatim. Consequence: with Lt as the prefill GEMM
engine, a captured prime is an address-lottery numeric config (stream flips
as early as step 2 on synthetic prompts) — below the stream-identity bar
every promoted config met tonight. RECLAIM PATH (priced, not taken): replace
Lt inside captured primes with an address-deterministic fp16 GEMM kernel (a
new kernel arc; MMQ-inside-graph is a net loss: saves 2.3ms gaps, loses ~5ms
GEMM speed). DISPOSITION: prime graphs stay off serving; the machinery
(PrimeGraph, captured trunk, pad-proofing, gates, discriminator flag) is
committed and regression-guarded for when a deterministic GEMM lands. The
serving default remains the all-green eager + batch-prime stack at
pp512 19.9k / +21.6% serving bursts.

**CUTLASS deterministic-GEMM probe (2026-07-26): reclaim path REFUTED at
current rates — the branch is measured, closed, and priced.** tools/
bench_cutlass_f16.cu: sm90 CollectiveBuilder fp16 TN, 7 config sweep
(tiles 128x128/128x256/64x256/256x128, K 64/128, clusters, pingpong/coop/
auto). VERDICT: (1) DETERMINISTIC under address shifts on every shape and
every config — the property Lt lacks, confirmed available; (2) RATE ceiling
0.69-0.75x of Lt (best: default 128x128x64 cluster1x2 auto = 419-514TF vs
nvjet 611-687TF; explicit pingpong pathological at 30TF on this
toolchain/instantiation). ECONOMICS: cutlass-in-graph pays +4ms GEMM tax for
-2.3ms gap savings = net -6% per prime — REFUTED; cutlass-everywhere -30%
GEMM — refuted trivially. The gap-floor reclaim therefore requires a
hand-tuned deterministic fp16 GEMM at >= ~620TF: a CUTLASS-grade kernel
project (weeks-class — tonight's hand-rolled wgmma pipeline history peaked
at ~0.5x Lt). EVERY sub-week avenue in the system now has a measured
endpoint: landed, promoted, or refuted with data.

**CUTLASS int8 probe (2026-07-26, W8A8-reopen check):** default-config sm90
int8 GEMM: 569-780 TOP at model shapes (ffn_down 1.11x vs cublasGemmEx, rest
0.72-0.87x) — DETERMINISTIC everywhere, but nowhere near the ~1300TF-class the
vLLM-35k arithmetic implies (9B x 2FLOP x 35k tok/s = 630TF effective WHOLE
forward => their GEMMs must exceed ~1.1-1.3PF or their FLOP count is lower
than assumed). The inference chain is now suspect — decomposing vLLM's actual
prefill with nsys on this box (their GEMM kernels/us, GDN/FLA kernels, gaps)
to replace inference with measurement before pricing the beat-35k path.

## vLLM decomposed on-box (2026-07-26 nsys) — the lane math changes

Rerun of the engine-decision bench script (same box, same 2048-prompt shape):
**vLLM prefill = 31.0k tok/s TODAY (not the recorded 35k); decode 171.6-176.4
— bw24's 183.7 = 105-107% of vLLM decode (lead widened).** bw24 prefill 19.9k
= 64% of the real number. Their prefill burst per-kernel (nsys):
- nvjet_sm90_tst_256x128 (Lt INT8) ~174us/launch — their GEMM engine is ALSO
  Lt/nvjet, i.e. the SAME address-variant numeric class that blocked our
  monolithic prime graphs;
- flashinfer GDN JIT kernels dominate their busy time (device_kernel 100ms
  class + delta_rule cutlass kernels) — their GDN prefill is EXPENSIVE;
- triton fused int8-quant/norm/silu chains + causal-conv kernels;
- "Capturing CUDA graphs (mixed prefill-decode, PIECEWISE)" — vLLM ships
  graphs WITH nvjet by capturing PIECEWISE: graph the elementwise/state
  chains, call the GEMMs eagerly between graph segments.

**THE UNBLOCKED ROUTE — PIECEWISE PRIME GRAPHS:** our gap floor (3.7ms/prime)
sits in exactly the small-kernel clusters (norms/adds/cvt/GDN glue) that
piecewise capture covers; every one of OUR custom kernels is
address-deterministic (mma/cp.async fixed schedules — only Lt is not).
Graphing the between-GEMM segments (per layer: norm->proj-split glue,
conv/GDN chunk stack, add/norm/silu chains) with Lt GEMMs eager between
segments reclaims most of the floor with ZERO numeric-config change —
bit-identity preserved because the captured kernels are deterministic and
the Lt calls stay exactly as they are. vLLM's own stack validates the
approach on this model/GPU. This is a sub-week arc: segment the captured
trunk at GEMM boundaries, capture each segment once per bucket, replay
segments interleaved with eager GEMM calls.

**Prime activation slabs (2026-07-26): LANDED, honest-neutral standalone,
piecewise foundation in place.** The eager prime's seven trunk transients
(h/x1/z/act/x-pingpong/h16/z16) live in resident per-model slabs
(BW24_PRIME_SLABS=0 reverts): ~224 fewer alloc/free API calls per prime and
FROZEN Lt operand addresses (nvjet variant selection now run-to-run stable —
the property piecewise segment capture requires). pp512 19,826 vs 19,984
slab-less = the third independent launch-diet NEUTRAL (the finding stands:
the floor is submission cadence, not call count). All gates green (validate,
graph-session, prime-batch, argmax MATCH, same output text). NEXT (the
piecewise arc proper): segment the slab-resident layer loop at GEMM
boundaries, capture the deterministic custom-kernel segments per bucket
against the slabs, replay interleaved with eager Lt calls — the vLLM-validated
pattern; slabs give every segment fixed IO addresses for free.

## Piecewise prime graphs — full segmentation design (2026-07-26, build-ready)

Segments contain ONLY cache-free deterministic kernels; EAGER between: every
Lt GEMM, plus the three cache-touching kernels (conv ring update, GDN K4
state pass, KV append) — so segments replay against SESSION caches with no
pointer machinery at all (the monolithic arc's cache problem disappears).

Per-layer sequence (E = eager, S = captured segment):
  S-glue:  [prev-add + attn_norm_f16out]                 (x1,ffn_out -> x_nxt, h, h16)
  E:       qkv / gdn4 GEMM group (xh = h16 slab)         -> proj slabs
  GDN:  S-prep: [conv-window + repack + l2 x2 + sigmoid + glog + K1 + K2 + K3]
        E:      conv-ring update; K4-mma (cache state)
        S-out:  [K5-mma + gated_rmsnorm]                 -> gn slab
  ATTN: S-attn: [q_gate_split + qk-norms + rope + fa_prefill(+gate)]
        E:      KV append
  E:       wo / ssm_out GEMM -> mixed slab
  S-mid:   [add + post_norm_f16out]                      (x_cur,mixed -> x1, z, z16)
  E:       gate/up group -> gate/up slabs
  S-act:   [ffn_act]                                     -> act slab
  E:       down GEMM -> ffn_out slab
Launches/layer: ~16 captured into 4-5 graph launches + ~8 eager calls.

SLAB INVENTORY (all sized at bucket x dim, ~200MB total at bucket 512):
existing 7 (h/x1/z/act/x-pingpong/h16/z16) + boundary slabs: GDN projs
(qkv_mixed 8192, z_g 4096, beta/alpha 2x num_v), attn projs (qf/k/v),
conv_out, q_g/k_g/v_g, q_l2/k_l2, beta/g_log, gcum, A/P (nc*h*c*c), U/W/Y
(nc*h*c*128), ssnap (nc*h*128*128), gn, attn_g, mixed, gate, up, ffn_out.
GEMM `_into` variants (write into slab views — the FFI already takes y
pointers; only the Rust wrappers allocate) for: matmul_group_xh, matmul,
try_f16_gemm_pre.
Capture: ~4 segments x 32 layers x buckets at ~1-3ms each ~= 0.5s boot per
bucket. Replay submits each segment in ONE cuGraphLaunch — attacking the
measured submission-cadence floor directly (the three launch-COUNT neutrals
do not apply: count reduction never fixed cadence; single-call submission
does — vLLM's piecewise pattern on this exact model/GPU is the existence
proof). Projected reclaim ~2-2.5ms/prime (+8-10% pp512) with ZERO numeric
change (all captured kernels address-deterministic; Lt untouched, and slabs
already froze its operands).
Gate: piecewise-vs-eager bit-identity (same kernels, same buffers, same
order — this one CAN be a bit-gate, unlike the monolithic config).

**Piecewise increment 3 (2026-07-26): FIRST SEGMENT LIVE — pp512 crosses 20k.**
S-glue (down-add + next attn-norm, all-slab IO, zero in-graph allocations —
keeperless capture is clean here) captured per layer per T, replayed as ONE
cuGraphLaunch each: pp512 19,825 (off) -> 20,009 (on), +0.9% from a 2-kernel
segment x31 layers. BIT-IDENTICAL confirmed (same kernels/buffers/order — the
piecewise config is bit-gateable as designed; argmax/output byte-equal, all
batteries green). The submission-cadence mechanism is VALIDATED: this is the
first launch-structure change that moved the number (three count-reduction
neutrals stand in contrast). Scaling path: S-mid (add+post-norm — needs the
`mixed` slab via the mixer out-GEMM `_into` refactor), then S-prep/S-attn
(7-9 kernels each) per the build-ready segmentation — projected +8-10% total.
BW24_PRIME_SEG=0 reverts.

**Piecewise increment 4 probe (2026-07-26): S-mid-via-copy REFUTED.** Staging
the mixer output into a slab (one 8MB D2D/layer, ~80us/prime + its submission)
costs more than a 2-kernel segment saves: ON 19,870 vs OFF 19,999 (S-glue-only
baseline 20,009). Bit-identical, reverted. The proper S-mid (and every larger
segment) needs the mixer out-GEMM written _into_ the slab directly — the
core-contract refactor (cores return gn/attn_g; prime_chunk runs the out-GEMM
via try_f16_gemm_pre_into) is the gating increment for the remaining +7-9%.
Increment-3 state (S-glue live, pp512 20,009) is the shipped baseline.

**Piecewise increments 3-5 CORRECTED VERDICT (2026-07-26, interleaved A/B):**
the interleaved protocol (the repo's own law, violated in the increment
measurements) refutes the small segments: ON vs OFF interleaved x3 = -1.0%,
-1.2%, +0.0% — a cuGraphLaunch costs about what two kernel submissions do, and
the earlier "+0.9% / pp512 crosses 20k" was CLOCK DRIFT across builds (the
absolute band moved 19.8k -> 20.1k -> 17.7k over the session; only interleaved
comparisons are valid). DISPOSITION: BW24_PRIME_SEG flipped to OPT-IN; the
core-split refactor + slabs + _into plumbing stay (byte-identical, verified,
the foundation for the 7-9-kernel segments whose economics remain open:
~6-7 submissions saved per launch x 24-32 layers ~= 1ms/prime IF the pattern
holds at size). LESSON PINNED: every remaining perf claim in this arc must be
interleaved-A/B measured; cross-run medians are invalid at this session depth.
