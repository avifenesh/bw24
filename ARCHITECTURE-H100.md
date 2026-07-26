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
