# lane/pp-prefill-serve — the PP-2 PREFILL serving bill

**Mission:** Step-3.7-Flash over PP-2 prefills at **90.9 tok/s** on pp4096
(`research/step-sku-20260807/raw/capacity-20260807T075551Z.log`, N=5, spread 0.12%) because the
pp door is eager-decode only. At the 89.5:1 prompt-heavy traffic ratio that caps the SKU at ~$2/day;
every 1K tok/s of sustained prefill ≈ $18/day. Target: multi-thousand tok/s class.

Box: 2x RTX PRO 6000 Blackwell Server 96GB (`18.195.123.14`), PP-2
(`MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1`), artifact `~/step37/models/step-3.7-flash/`
(IQ4_XS 3 shards + MTP Q8_0). Every GPU window under `flock /tmp/memra-gpu.lock` (shared with
pp2spec + tick-seg lanes). Raw receipts land in `raw/` here; `.nsys-rep` stays in `/tmp` on the
box, never in git.

---

## Increment 0 — structural reading (code facts, established before the profile)

Read first: `research/step37-p2-20260806/PROGRESS.md` (bring-up), `research/pp2-batch-20260806/RESULTS.md`
(the paid decode side of this bill), `research/pp2-hardening-20260806/PROGRESS.md` (P2P verdict,
refusal audit), `research/step35-chunkfix-20260807/PROGRESS.md` (seq_end chunk-invariance law).

### Fact 1 — the prime path has NO pp arm and FAILS OPEN over a sharded split

`prime_cache` / `prime_chunk` (`crates/memra-engine/src/hybrid_forward.rs:407-860`) contain zero
`pp::` references. The 2026-08-06 hardening audit fixed the four fail-open doors
(`decode_step_batch`, `decode_step_dc`, graph capture, spec verify — all now call
`pp::refuse_unsplit_if_remote`) but **prime was never in that audit**. Under
`MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1` with the sharded loader (the Step SKU's only placement):

- every prime chunk walks all 45 layers on the PRIMARY engine's stream (dev0);
- layers 22-44's trunk weights (attn, norms, shexp, router) live on dev1 → peer-read per GEMM.
  At m=4096 the weight read amortizes over the chunk (~2 GB of stage-1 trunk per chunk ≈ ~36 ms
  at the measured 56 GB/s P2P — real but not the 45 s), unlike decode where it was the 28x cliff;
- layers 22-44's KV (allocated on dev1 by `pp::new_cache`) is peer-WRITTEN by the append and
  peer-read by the FA view kernels;
- **dev1 contributes zero FLOPs to prefill.** The second card is a KV closet.

### Fact 2 — step35's MoE prefill rides the PER-TOKEN sequential expert loop

`moe_ffn_inner` dispatch (`hybrid_forward.rs:2005-2560`) for step35 at prefill t:

| arm | gate | step35 verdict |
|---|---|---|
| `moe_ffn_pairs` (one launch per proj for ALL pairs) | `cfg.sigmoid_router().is_none()` | **DENIED** (sigmoid router, and clamp layers 43/44 have no fused clamp form) |
| `moe_ffn_dev` (device router, zero-DtoH) | `dev_ok` requires `sigmoid_router().is_none()` | **DENIED** (softmax-only device router — the M3 74602-vs-92 lesson) |
| `moe_ffn_grouped` (A2: group tokens by expert, m=m_e GEMMs) | `MEMRA_MOE_GROUPED=1` env | **OFF by default** |
| per-token sequential loop | fallback | **THIS RUNS** |

The sequential loop per token per MoE layer: `quantize_q8_1_view` + 8 experts x
(`moe_cached_gemm_q8` gate + up + `ffn_act_lim` + `quantize_q8_1` + down + `axpy_into`)
≈ ~49 kernel launches per token-layer at m=1, **unless** all 24 of the token's blocks are
SLRU-resident, in which case `moe_gdec_token_q8` folds it to ~3 launches — but the model does NOT
go resident on this box (101.07 GB experts vs 94.96 GB budget → SLRU, boot receipt), and a token
needs ALL 8 experts x 3 projections resident to take gdec: at the measured ~96% per-block
steady-state hit rate, P(all 24) ≈ 0.96^24 ≈ 0.37, so most tokens fall through to the m=1 loop
with H2D staging of misses.

Launch-count arithmetic at pp4096: 42 MoE layers x 4096 tokens x O(3..49) launches per token-layer
= **O(0.5M..8M) kernel launches per prime**, every expert GEMM at m=1 with zero operand reuse.
45.06 s / 4096 tokens / 45 layers ≈ **244 us per token-layer** — the decode shape, at prefill.
This is why 90.9 tok/s prefill sits within 3x of the 34 tok/s decode: for the dominant MoE cost,
prefill IS per-token decode today.

By contrast `moe_ffn_grouped` runs ~288 active experts x 3 GEMMs at m_e ≈ 114 (avg over
4096x8/288) per layer ≈ ~1K launches/layer of real GEMMs, reading each expert block once per
layer-chunk instead of once per token-hit.

### Fact 3 — the chunk-invariance laws that bind any prefill change

- step35 kernel selection MUST key on the request's `seq_end`, never a chunk-local `t_kv`
  (`step35-chunkfix`: `P ≡ 0` by construction; `chunkinv35` is the gate; `tickinv` may still be
  red on this branch — the tick-seg lane owns it, coordinate via gate status only).
- `moe_ffn_grouped` as written routes via `e.matmul(&m.gate_inp, z, t)` — the cuBLASLt router GEMM
  is **m-DEPENDENT** (research/concat-prime-exact-20260802), so chunk size would steer router
  logits → expert selection → chunk-dependent text: the exact class chunkinv exists to kill.
  Any grouped adoption must route selection through the m-invariant `router_gemv`
  (`router_prefill_exact_on()`, already the sequential path's default) so selection is
  bit-identical to the sequential arm and chunk-size-invariant by construction.
- Sequential-vs-grouped expert math differ in class (q8 dp4a vs f32 dequant qmatvec — the
  documented ~3.4e-4 t>1 mismatch; `MEMRA_MOE_Q8=0` restores byte-identity). A dispatch-class
  change on the served prefill path needs the full exactness battery + before/after numbers,
  and must be uniform across all chunks of a request.

### Fact 4 — what the decode side of this bill already proved (reusable)

- The `[B, n_embd]` boundary transfer is ~free (pp2-batch: split costs 0.5-1.5% at B=4-16;
  transport alone 0.986-0.997x). At prefill the payload is `[chunk_t, n_embd]` f32 = 64 MB at
  4096x4096 — at 56 GB/s uni ≈ 1.1 ms per boundary crossing, trivially hidden by chunk compute.
- `PpNRt` already has per-stage engines/streams/contexts, grow-only boundary slots (`tx`/`rx`
  take a payload element count — the batched arm already sends `b_n * n_embd`), and the
  slot first-use ordering + publish_to laws are receipted. A chunked prime split reuses all of it.
- SGLang #33666 law: per-stage resources budget on the stage's OWN layer slice; TRT-LLM #16170
  law: drain sends before blocking in compute, missing-relay = loud error.

## Increment 1 — anatomy profile (RUNNING)

`anatomy-pp4096.sh` (committed here) launched detached on the box 2026-08-07T11:41Z, queued
behind a co-tenant chunkinv window per flock discipline. nsys 2026.1.3 installed
(`/opt/nvidia/nsight-systems/2026.1.3`). Two arms in one lock hold:
1. nsys-traced ppprime (1 warmup + 1 rep), `--trace=cuda --sample=none`, `.nsys-rep` in /tmp;
2. untraced control (2 reps) — the nsys-overhead check against the 90.9 baseline.

Extraction: `cuda_gpu_kern_sum`, `cuda_api_sum`, `cuda_gpu_mem_time_sum` CSVs → `raw/`.

**Pre-registered predictions (written before the profile is read):**
- P1: GPU kernel time is dominated by m=1-class expert kernels (`qmatvec`/`moe_*_q8`/dp4a
  family), not by MMQ prefill GEMMs and not by `cudaMemcpyPeerAsync`.
- P2: a large fraction of the 45 s is NOT covered by GPU kernel time at all (launch/host gaps —
  the per-token loop's ~5-6 us/launch shape), visible as cuLaunchKernel dominating the API sum.
- P3: H2D memcpy (SLRU staging of expert misses) is material (multi-GB) but not the majority.
- P4: dev1's kernel time ≈ 0 (KV appends/reads only — no compute split exists).

If P1/P2 hold, the biggest single lever is the MoE prefill dispatch shape (grouped/batched expert
GEMMs for the sigmoid-router arch), with the PP chunk pipeline as the second multiplier on top —
and the increment order below gets re-scoped accordingly per the stop-and-report clause.

---

## Increment 1 — ANATOMY VERDICT: the mission's premise is refuted; three named levers

Receipts: `raw/anatomy-20260807T114137Z{-kernsum,-apisum,-memsum,-gpusum}.csv` + `.log`
(nsys 2026.1.3, `--trace=cuda --sample=none`, 1 warmup + 1 timed prime; deeper queries run
against the sqlite export on the box — the queries and outputs are quoted below). One lock
window 11:41→12:12Z, cards 0 MiB at exit. **nsys distortion check:** traced rep 45.80 s vs
untraced same-window control 47.51/45.07 s and the capacity baseline 45.04-45.10 s — the trace
is inside the control spread, numbers are representative.

### Prediction scorecard (honest, against the pre-registration above)

| pred | verdict | measured |
|---|---|---|
| P1 m=1 expert kernels dominate | **PARTIAL** | largest *launch count* (835K launches/prime, 28% of GPU time) — but the top kernel was unpredicted (below) |
| P2 wall dominated by host/launch gaps | **REFUTED** | dev0 kernel-union busy = 93.4% of span (87.63 s of 93.77 s, both primes); 1.84M cuLaunchKernel calls are fully overlapped |
| P3 H2D staging material, not majority | CONFIRMED | 37 GB H2D per prime (SLRU expert staging), only 0.83 s GPU-side, overlapped |
| P4 dev1 contributes ~0 FLOPs | **CONFIRMED exactly** | `kernels per device: [(0, 2337323, 87.6s)]` — **zero kernels on dev1 in the whole trace** |

### Where the 45.8 s actually goes (per prime; accounting reaches 44.4 s of 45.8)

| # | cost | s/prime | % | mechanism |
|---|---|---:|---:|---|
| 1 | `sdpa_naive_w_f32` | **18.6** | **41%** | 33 SWA layers x 565 ms. The windowed f32 floor from the bring-up ("no windowed FA prefill stamp at hd128"). Its inner loops iterate ALL t_kv=4096 keys per query (mask discards 87%), and thread 0 does a serial 4096-element softmax per (head,query) block. The full-attn layers answer what this should cost: `fa_prefill_qw_db_hd128` = **3.3 ms** for a STRICTLY HARDER workload (causal 4096 vs window 512) — the floor is ~170x off FA class on an easier problem |
| 2 | peer-read tax (stage-1 weights from dev0) | **10.2** | **22%** | three kernel families are bimodal with the split EXACTLY on the fence [0,22,45]: MMQ `fx132 Sx5fS...` = 22 stage-0 layers x 6 fast, then stage-1 slow (fast med 1.0 ms, slow med 33.5 ms = **34x**); router `fx19 Sx23` = 19 vs 23 MoE layers (0.7 → 72 ms = **100x**); `qmatvec_iq4_XS_dp4a` `fx22 Sx23` (0.4 → 63 ms). Slow classes: mmq 6.89 s + dp4a 1.7 s + router 1.65 s. Same cliff class as decode's 28x, amortized by m=4096 but still 22% of wall |
| 3 | MoE m=1 per-token dispatch | **12.6** | **28%** | `moe_gate_up_silu8_q8` + `moe_down8_fma_q8` at n=161,383 each (= 4096 tok x ~39 hit layer-tokens, gdec m=1 pairs) 5.9+4.8 s, plus `qmatvec_expert_q8` n=255K (staged misses) 1.3 s + `quantize_q8_1` n=419K 0.55 s. The gdec hit path DID fire for most tokens — but it is still one m=1 launch pair per (token, layer): 67 us of kernel time per token-layer for what grouped m_e≈114 GEMMs do with one weight read per expert per layer |
| 4 | local MMQ + everything else | 3.0 | 7% | stage-0 trunk GEMMs 0.93 s, q45k 0.72 s, H2D 0.83 s, rms/rope/gate/quantize tails |

Boundary transport is invisible: D2D total 5.4 ms per trace. `cudaMemcpyPeerAsync` is nowhere
in the cost. The 63.5 s of `cuMemcpyDtoHAsync` API time is the host WAITING behind enqueued GPU
work at the 84 per-MoE-layer router-logit readbacks (4.6 MB each — sigmoid host routing), not a
transfer cost; it becomes a real serialization risk only after the kernels shrink.

### The verdict against the brief

The brief's increment 2 ("chunked pipelined prefill... the gap is PP transport + scheduling, not
GEMM") is **refuted as the primary lever**. dev0 is 93% busy — there is no scheduling stall to
overlap away; pipelining today's work would at best approach 2x (≈180 tok/s), nowhere near the
multi-thousand class. The prefill is slow because of three compute-shape defects, and the PP
split fixes only the second:

**Lever A — windowed FA prefill stamp at hd128** (projected: 18.6 s → ~0.1 s).
`fa_prefill_qw_db_hd128` exists (used by the 12 full-attn layers, 3.3 ms/layer); every WINDOWED
prefill stamp is hd256-only (step37-p2 increment 6 named this gap). A windowed hd128 twin (the
same mask delta the hd256 pair already implements) replaces the f32 floor on all 33 SWA layers.
Kernel-selection stays keyed on `seq_end` (the chunkfix law) — the class changes UNIFORMLY for
the whole request, so chunk-invariance holds by the same construction. New numeric class on SWA
rows → full battery arbitrates (kernel-check cell vs CPU windowed oracle, chunkinv35, run-gen,
ppn-gate). Single-card measurable. **90.9 → ~154 tok/s alone.**

**Lever B — stage-split chunked prime over PP-2** (projected: kills the 10.2 s tax + doubles
throughput via overlap + flips expert residency). This IS the brief's increment 2, demoted to
second: per-stage layer ranges through `decode_layers_eager`-style prime ranges on each stage's
engine, `[chunk_t, n_embd]` boundary (64 MB at 4096 — 1.1 ms at the measured 56 GB/s, trivially
hidden). Two receipted side effects the brief did not price: (a) dev1 starts computing at all
(today: zero kernels); (b) the PP-blind residency numerator (step37-p2 finding B) currently
compares the WHOLE 101 GB bank against one card — split per stage, each card's ~50 GB share fits
RESIDENT in ~95 GB free, which kills the 37 GB/prime SLRU staging and makes the gdec hit
predicate always-true. **A+B ≈ 420-450 tok/s** (17.1 s serial halved across cards + fill).

**Lever C — grouped expert GEMM prefill for the sigmoid-router arch** (projected: 12.6 s → ~3 s).
`moe_ffn_grouped` (A2) already exists with the bit-identity slot scheme, shexp, and the step35
clamp threaded — but is env-gated OFF and routes selection through the m-DEPENDENT cuBLASLt
router GEMM (chunk-dependent routing = the chunkinv class). Adoption = route its selection
through the same `router_gemv` + sigmoid host oracle the sequential path uses (bit-identical
selection by construction), then A/B the expert-math class (grouped f32 qmatvec vs sequential q8
dp4a — the documented ~3.4e-4 divergence; `MEMRA_MOE_Q8=0` is the byte-identity control).
**A+B+C ≈ 1400-1600 tok/s** — the multi-thousand class the bill needs. C is the largest
complexity and the most exactness-sensitive; it goes last.

Re-scoped order: **A → B → C**, each with its own exactness battery + interleaved perf receipt
before the next. TTFT on the 4k prompt (2.18 s p50 today) rides lever A immediately
(prefill is ~95% of TTFT).

---

## Increment 2 — LEVER A: the windowed hd128 FA prefill stamp (commit `8b425742` + fix `5c523d5e`)

### What landed

- `flash_attn.cu`: `fa_prefill_qw_body` / `fa_prefill_qw_db_body` gain `int window = 0` —
  `fa_prefill_f32_body`'s exact mask predicate (`k < q_pos-(win-1)` → NEG_INF) plus its
  whole-tile skip; in the db twin the skip folds into the loop **bound** (`t_start`) so the
  cp.async prefetch chain is unbroken, buffer parity following `t_start`. `window=0` is the
  default-arg body — the existing stamps are byte-unchanged. New stamps
  `fa_prefill_qw_w_hd128` / `fa_prefill_qw_db_w_hd128`.
- `lib.rs`: `fa_prefill_view_ws_w_hd128` — the windowed twin of `fa_prefill_view_ws` (same
  dequant-once bf16 workspace/slab, db default, `MEMRA_PRIME_DEQW_DB=0` seam).
- `hybrid_forward.rs` `step35_attn_pre_wo`: the SWA arm defaults to the FA twin; selection
  still keys on `seq_end` (chunkfix law). **`MEMRA_STEP35_SWA_FA=0`** = rollback to the f32
  floor (documented in docs/FLAGS.md).
- `kernel_check.rs`: 4 assertions x 3 shapes (CPU windowed oracle at the 2e-2 fa band; vs the
  floor same band; window BITES vs unwindowed FA; db-vs-single-buffer **bit-identical**, incl.
  the odd-`t_start` shape) + `window=0` bitdiff=0 vs `fa_prefill_view_ws`.

### The gate that earned its keep: chunkinv35 caught a NEW chunk-dependence door

Battery 1 (`raw/leverA-gates-20260807T135541Z.log`): G1 kernel-check ALL GREEN, G3 run-gen
MATCH, G4 ppn-gate bit-identical both arms, G5 run-spec 8/8 PASS at baseline acceptance
(82.4% K=1) — and **G2 chunkinv35 FAILED**: 513 diverging at row 513 (1.115e0), 512/256/64
mutually identical diverging at row 512 (1.164e0). Mechanism, distinct from the chunkfix
class: the SWA view offset `off = base_len-(win-1)` starts the FA **tile grid** at a
chunk-dependent absolute position; the qw kernel's online-softmax recurrence groups keys into
BK=32 tiles **relative to the view start**, so the same absolute keys regroup at different
chunk sizes → different (m,l) rounding → different bits. The f32 floor was immune (serial
per-key softmax, no tile grouping), which is exactly why the chunkfix lane never saw this door.

Fix (`5c523d5e`): `off &= !31` — the tile grid pins to absolute key positions at every chunk
size. The ≤31 extra leading keys are older than every query's window (all queries ≥
`base_len`) and a fully-masked key is a bitwise no-op in both kernels (NEG_INF → p = exact
0.0; l += 0.0 / O += 0.0 are IEEE identities), so the floor arm's bits cannot move either —
battery 2 measures that claim (G2f) rather than arguing it.

### Perf (battery 1, N=5 interleaved FA/floor in one hold, pre-alignment arm)

| arm | pp4096 tok/s (5 reps) |
|---|---|
| FA (default) | 153.4, 140.8, 140.6, 140.6, 141.0 |
| FLOOR (`MEMRA_STEP35_SWA_FA=0`) | 85.6, 85.8, 85.8, 85.8, 85.7 |

**90.9 → ~141 tok/s (1.55x)** — the anatomy's lever-A projection (~154) was close; the floor
arm reproduces the capacity baseline (85.7 ≈ the 86-91 window). TTFT p50 2.182 s (G7) —
**unchanged**, which is itself a finding: the serve prime path chunks at `PREFILL_TICK_T=1024`
and the 228-token TTFT probe prompt sits under the 512 window where both arms ride
`fa_prefill_view_ws` anyway; the 4k-prompt TTFT is the one lever A moves.

Battery 2 (running): G2 must flip INVARIANT, G2c canary stays red, G2f floor unmoved,
G3/G5 re-receipts, G6 N=5 interleaved on the aligned arm.
