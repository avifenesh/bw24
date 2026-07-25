# Owner-selected lanes (2026-07-23): trained draft head + multi-request serving

## Lane 1 — trained draft head for Hy3 (EAGLE-class)

No existing Hy3 draft/EAGLE head on the Hub (searched 2026-07-23). The shipped layer-80 MTP
head ceilings at 77% gated acceptance (owner bar: 0.80; receipts in
`../per-expert-quant/local-5090-10toks-plan.md`). The gating framework (BW24_SPEC_PMIN,
K>2) is proven and waiting — a head drafting at 85-90% makes K=4 spec a 1.3-1.5x multiplier
on the whole stack.

Plan:
1. **Data**: self-distillation corpus — prompts through Hy3 itself (greedy, the serving
   regime), capturing per-token hidden state (pre-lm_head) + sampled token. The MoE input
   trace hooks already capture hidden states; extend to the final-norm state or reuse the
   layer-79 capture. Volume target: 50-100M tokens (EAGLE-3 recipes use ~68M).
2. **Architecture**: EAGLE-style single-layer autoregressive head over (hidden, token-embed)
   pairs, sized to Hy3's 4096 hidden. Reuse the existing MTP serving path in bw24 (the spec
   plumbing is arch-agnostic given a draft head that emits logits) — the head replaces the
   layer-80 weights, keeping the verify machinery untouched.
3. **Compute**: training runs on a rented research GPU (vast/G7e class per project rules —
   never the serving rig). Head is ~0.5-1B params: single-GPU trainable.
4. **Gate**: gated acceptance ≥0.80 at PMIN 0.7-0.85 AND net ≥1.15x vs plain at NGEN≥64,
   N=3 interleaved, before any default flip.

## Lane 3 — multi-request serving (expert-io amortization across streams)

Measured overlap (route trace, 300 trials/m): concurrent streams share routed experts at
**1.12x (m=2), 1.32x (m=4), 1.66x (m=8)** — per-token expert io and weight-decode cost
divide by these factors when execution is grouped by expert; GPU-side resident experts gain
additionally from batched matvec weight reuse. Projected aggregate at m=8: ~7-9 tok/s.

Build (in order):
1. **Recon**: prefill's `moe_ffn_grouped` (T>1 expert-grouped dispatch) and the verify pass's
   batched decode-like path — the two existing seams closest to lockstep multi-stream decode.
2. **Multi-stream state**: per-stream KV cache + GDN linear-attention state; block-diagonal
   attention for T=m lockstep steps (each stream attends only its own history).
3. **Scheduler**: m prompts advanced in lockstep; per-layer router batches m tokens; expert
   execution grouped by unique expert across streams (CPU companion already accepts
   arbitrary expert lists per call — one call per unique expert with m-row activation is the
   natural extension; the multi-token ABI experiment from 2026-07-21 was removed under
   winners-only but its receipt documents the grouping approach).
4. **Gate**: aggregate tok/s at m=2/4/8 vs m× single-stream baseline; per-stream latency
   reported alongside (aggregate wins must not hide unacceptable per-stream tails);
   correctness = each stream's output identical to its single-stream run.

## Lane 3 seam recon (2026-07-23, full map in agent transcript; key facts)

- `Cache` is hard single-sequence (scalar `len`/`pos`, one recurrent state per layer) — but
  v1 lockstep AVOIDS the refactor: m independent `Cache` objects, one per stream.
- Attention stays per-stream (existing `full_attn_decode`/`linear_attn_decode` calls, T=1
  each) — no block-diagonal mask needed until attention batching becomes worth it
  (`fa_decode_rows` per-row key ranges are the seam when it does).
- `moe_ffn_grouped` (hybrid_forward.rs:2742) gathers/scatters by flat row index, stream-
  agnostic — reusable for the cross-stream MoE stage, GPU side.
- CPU companion ABI is one-row-per-call: within-step io amortization needs NO ABI change
  (stream 1's miss fills the shared RAM cache; siblings hit). Weight-decode compute
  amortization needs a multi-row ABI v3 — deferred to M3.

Increments (each battery-gated):
- **M1**: `decode_step_lockstep(streams)` — per-layer walk, per-stream mixers, per-stream
  sequential MoE (correctness first: each stream's tokens identical to its single-stream
  run; aggregate baseline measured).
- **M2**: cross-stream MoE stage — route all m rows, dispatch CPU experts stream-ordered so
  shared-cache reuse lands within the step; measure io amortization vs the 1.12/1.32/1.66x
  curve.
- **M3**: companion ABI v3 multi-row-per-expert + GPU grouped dispatch across streams.
- **M4**: m=4/8 scaling, serve loop, per-stream latency reporting.

## M1 first gate (2026-07-23, lockstep-m*.log)

`decode_step_lockstep` + `run_lockstep`: per-stream math is `decode_step_h`'s; the harness
gate requires all streams (same prompt) to emit identical tokens — PASS at m=1/2/4.

| m | aggregate | per stream | whole-run cache hit rate |
|---|---:|---:|---:|
| 1 | 4.44 | 4.44 | 58.6% |
| 2 | **6.10 (+37%)** | 3.05 | 67.4% |
| 4 | 2.52 (COLLAPSE) | 0.63 | 72.0% |

Cross-stream cache amortization works (hit rate climbs with m; m=2 beats the 1.12x overlap
prediction because GPU-side residency adjacency stacks on top). The m=4 collapse is a
per-step nonlinearity (step wall 328 ms -> 1590 ms), not RAM (RSS flat, zero swaps) —
prime suspect is VRAM-edge pressure from 4 streams' recurrent states allocated after the
0.90-frac expert slab sized itself; discriminator arms (m=4 at frac 0.85/0.80, m=3 at 0.90)
in flight.

## M1 discriminator (2026-07-23, lockstep2-*.log): VRAM-edge confirmed

| arm | aggregate | identity |
|---|---:|---|
| m=3 @ frac 0.90 | 6.03 | PASS |
| m=4 @ frac 0.85 | **5.23** (was 2.52 @ 0.90) | PASS |
| m=4 @ frac 0.80 | 5.13 | PASS |

The m=4 collapse was VRAM-edge pressure: stream recurrent/KV state allocated after the
0.90-frac expert slab left no headroom. At 0.85 the collapse vanishes; 0.80 trades too much
residency back. Standing concurrency scoreboard: 4.44 single -> 6.10 (m=2) / 6.03 (m=3) /
5.23 (m=4@0.85), all streams bit-identical to single-stream. Aggregate peaks at m=2-3 while
the MoE stage is still per-stream sequential — M2 (cross-stream expert batching) and M3
(multi-row companion ABI) target exactly that; the frac knob needs stream-count-aware
sizing in the serve loop (M4).

## M2 gate (2026-07-23, m2gate-*.log)

| arm | aggregate | identity |
|---|---:|---|
| m=2 base / grouped | 6.17 / 5.85 | PASS / PASS |
| m=3 grouped | **6.31 (campaign best)** | PASS |
| m=4 @0.85 base / grouped | 5.34 / 5.66 | PASS / PASS |

Grouped MoE wins from m>=3 (+5-6%), loses at m=2 under the default q8 lanes (the sequential
path's dp4a arms are faster than grouped's f32-dequant at tiny m); under BW24_MOE_Q8=0
grouped already wins at m=2 (6.20 vs 6.09). Policy encoded: auto-grouped at m>=3.

Numeric class, measured precisely: grouped-lockstep tokens are NOT bitwise-identical to the
per-stream path even at BW24_MOE_Q8=0 — the m_e>1 batched GEMM reduces each row in a
different FP order than m=1 (a single near-tie greedy flip at token 8 in the exactness pair;
streams within any arm remain bit-identical, and cross-frac assignment changes are the other,
previously documented divergence class). Exact parity with sequential is impossible by
design once m_e>1 kernels engage; the gate standard for lockstep arms is per-arm stream
identity + this documented class, mirroring the BW24_MOE_GROUPED prefill receipt.

## M3 gate (2026-07-23, m2gate-m3-rows*/m4-rows logs)

Multi-row expert dispatch live end-to-end: experts routed by >=2 streams go through
bw24_cpu_expert_rows_v2 (weight decode amortized across rows; Q2_K fused, other formats
generic). m=3: 6.33/6.40 (best 6.40, campaign peak); m=4: 5.93 vs M2's 5.66 (+4.8% — the
gain grows with m, as sharing does). All stream-identity gates PASS. Cross-expert CPU
contribution FP-sum order documented as part of the lockstep numeric class.

Concurrency scoreboard after M1-M3: **4.44 single -> 6.17 (m=2) -> 6.40 (m=3) -> 5.93 (m=4)**.
Remaining M4 items: m=6/8 arms with stream-count-aware frac sizing, serve loop, per-stream
latency reporting, fused multi-row IQ3_S/Q4_K kernels (currently generic fallback).

## M4 scaling sweep (2026-07-23, fused multi-row kernels live)

| m | frac | aggregate | identity |
|---|---|---:|---|
| 4 | 0.85 | **6.32 (campaign best; 5.93 before fusing, +6.6%)** | PASS |
| 6 | 0.82 | 5.72 | PASS |
| 8 | 0.78 | 5.53 | PASS |
| 3 | 0.90 | **6.92 (campaign best — guarded rerun; the first run's 3.72 was mid-run interference)** | PASS |

Fused-M4 scoreboard: m=3 6.92 / m=4 6.32 / m=6 5.72 / m=8 5.53 — 1.56x the single-stream
ceiling at the m=3 optimum. The optimum moved m=2 -> m=3/m=4 across M1 -> fused-M4, tracking each
amortization increment. Past m=4 the binding costs are (a) residency given back through
stream-state frac headroom and (b) per-stream attention/glue, which scales linearly with m —
batched attention (the fa_decode_rows block-diagonal seam from the recon) is the next
structural lever, alongside stream-count-aware frac in a real serve loop.

## Distinct-prompt correction (2026-07-23 evening, mix*.log) — the honest serving numbers

| m (mixed prompts) | aggregate | vs single |
|---|---:|---|
| 1 | 4.74 | baseline |
| 2 | 3.96 | **worse** |
| 3 | 3.62 | worse |
| 4 | 3.50 | worse |

Every same-prompt gain (6.10-6.92) was the identical-routing artifact: identical streams
route identically, so siblings ride pure cache hits and the grouped/rows machinery sees
100% overlap. That regime is only real for identical-batch workloads. Under mixed prompts,
cache hit rate stays flat (59-61% — cache contention is NOT the mechanism) but expert sets
are disjoint: per step, m nearly-full expert loads serialize through the single CPU-executor
thread (each call serial io-then-compute), so aggregate falls below single-stream.

Lane-3 standing: the M1-M4 machinery (lockstep, grouped GPU dispatch, multi-row ABI, fused
kernels) is correct, gated, and pays in identical-batch regimes — but mixed-workload serving
needs CROSS-STREAM io/compute overlap (stream A's compute under stream B's reads): parallel
per-stream CPU dispatch with the fabric-interference constraint the pipeline receipts
mapped. That is the next design problem, not a tuning knob. Method lesson recorded: the
overlap simulation modeled sharing but not dispatch serialization; and same-prompt harness
runs must never be promoted as serving numbers.

## Executor-pool arms (2026-07-23 night, m2gate-ovl*.log) — cross-stream overlap works

| arm (mixed prompts) | aggregate | vs serial-executor mixed |
|---|---:|---|
| m=2, 2 executors x 4 threads | **4.65** | 3.96 → **+17%** (parity with single-stream 4.74) |
| m=3, 2x4 | 4.11 | 3.62 → +14% |
| m=4, 2x4 | 3.95 | 3.50 → +13% |
| m=3, 3x3 | 2.97 | worse — three 3-thread teams are too thin; 2 executors is the knee |

Stream A's compute under stream B's reads is real and the fabric tax does not eat it at
per-stream volume. Mixed serving is now at single-stream parity at m=2; the remaining gap is
the in-call serial io (each call still reads-then-computes). This compounds with the
tail-Q2_K requant (-22% miss bytes) — the io the executors serialize on shrinks by the same
fraction. BW24_CPU_EXPERT_EXECUTORS default stays 1 (winners-only: parity is not a win);
the serve-loop arm flips it when the compound clears the bar.

## Compound arms on the demoted artifact (2026-07-24, compound*/compound2*/compound3* logs)

Confound round first: frac 0.88 silently broke freeze-profile fidelity (6431 blocks restaged
into 6302 slots — 129 evicted during restage) and the executor receipt's 2x4 thread shape
does not transfer to the demoted artifact (io shrank, so calls are compute-bound: 2x4 = 3.42
vs 2x8 = 4.80 at m=2). Method note: profile-bearing arms must keep the profile's frac; the
harness now prints the CPU decode-window counters so no arm runs blind.

Honest mixed scoreboard on tail-Q2_K (frac 0.90, t8): m=1 5.45 / m=2 4.80 / **m=3 5.29** /
executors 2-vs-1 +2.8%. In-call pipeline at fractional volume (third falsification): m=2
3.01, m=3 5.09 — backend wall inflates again (11.6 -> 19.2 s at m=2); the in-call overlap
mechanism is dead on this fabric at every tested volume and stays retired permanently.

Standing: mixed-workload concurrency ceilings at ~parity with single-stream under lockstep.
The remaining mechanisms are different in kind: batched block-diagonal attention
(fa_decode_rows seam) shrinks the per-stream GPU serial share, and a real serve loop gets
continuous batching (prefill-under-decode phase overlap), which lockstep cannot express.
Identical-batch regimes keep their 6.9 aggregate win; single-stream keeps 6.0.

## M4a batched full-attention projections (2026-07-25, battn-*.log) — FLAT, door left open

Built `full_attn_decode_batched`: a generic m-band primitive splitting the mixer by what the
hardware cares about — WEIGHT-BOUND work (q/k/v + output projections) runs once at m because
all streams share the same weights, KV-bound work (append, fa_decode) stays per stream. Reuses
the existing m-band kernels (`matmul_q8_fused3_t`, `matmul_pre`) and `copy_view_into`; the
elementwise ops needed no new kernels (`rms_norm` takes nrows=m*n_head, `rope_neox` takes
n_tokens=m with a per-stream position vector, `q_gate_split` takes t=m).

| arm (mixed prompts, tail-Q2_K, 2x8 executors) | off | on |
|---|---:|---:|
| m=2 | 4.72 | 4.72 |
| m=3 | 5.35 | 5.24 |

Correctness: same-prompt m=2 tokens bit-identical between arms (PASS) — per-row quantize/norm,
per-token rope, and the same m-band kernels spec verify is gated on.

Verdict FLAT/slightly negative; predicted +10% did not appear. Mechanism: full-attn is the
minority layer type in this hybrid (GDN/linear dominates the 80 layers), so the m-band
weight-read saving covers few layers, and on exactly those layers the path gives up the
per-stream norm->q8_1 fusion (a measured +3% lever) and adds gather/scatter copies. Gain and
loss cancel. This is also consistent with the campaign's standing finding that decode matvec
is latency-bound at low m rather than weight-bandwidth-bound.

`BW24_LOCKSTEP_BATCH_ATTN` defaults OFF and is documented as an explicitly-blocked
experimental door. The primitive stays in-tree deliberately: a continuous-batching serve loop
batches across *requests* at higher m, where there is no fused per-stream alternative to lose
and the m-band weight read is the whole point. Extending the same split to the GDN layers
would be the only way to test the lever at full coverage — worth doing only inside that serve
loop, not in lockstep.

## Serve-loop batching: NOT INDICATED on this workload (2026-07-25 design note)

Mapped `bw24-server` before building continuous batching. The scheduler already admits up to
`MAX_ACTIVE = 4` sessions and round-robins them (`worker.rs:277-288`), each session owning its
own `Cache` — so m concurrent requests today cost m independent forward passes, i.e. aggregate
throughput ~= the single-stream rate (6.0 tok/s on tail-Q2_K), split across sessions.

Batched lockstep at mixed prompts measures 5.35 aggregate at m=3 and 4.72 at m=2 — BELOW that
round-robin baseline. Wiring `decode_step_lockstep` into the scheduler would therefore make the
server slower, for the reason the distinct-prompt correction already established: concurrent
requests route to disjoint expert sets, so every added stream costs nearly a full expert
io+compute load and batching only adds coordination on top. Request batching pays when streams
SHARE weights they would otherwise re-read; here the shared part (dense trunk) is the small
part and the unshared part (experts) is the wall.

So the serve loop is not the lever. The precondition for revisiting it is CPU-expert throughput
that scales with m — i.e. the asymmetric P+E executor work below. If concurrency cannot beat
single-stream in the CLI harness, it cannot beat it in the server either.

The engine-side pieces stay ready for the day that changes (`decode_step_lockstep`,
grouped-MoE, the m-band mixer, per-session caches already in the right shape).

## Asymmetric P/E core partitioning (2026-07-25) — first round + the single-stream design

Topology (kernel hybrid masks): `cpu_core` = 0-7 (8 P), `cpu_atom` = 8-23 (16 E). Every
measurement in this campaign so far ran expert compute on the 8 P-cores only; the 16 E-cores
carried at most io threads.

Built per-executor pinning: `BW24_CPU_EXPERT_EXECUTOR_CPUSETS="0-7;8-15"` gives each executor
its own core group and sizes its OMP team to that group (`BW24_CPU_EXPERT_EXECUTOR_THREADS`
overrides). This is the shape the 2026-07-23 receipt said was the only viable way to use
heterogeneous cores — ONE team per group, so a slow group never straggles a fast group's
barrier. Naive widening of a single team across P+E measured catastrophic (compute 2.8 -> 5.2 s
at 16 threads, 14.6 s at 20).

First round, m=3 mixed prompts on tail-Q2_K:

| arm | aggregate |
|---|---:|
| baseline (P only, 2 executors x 8 thr) | 5.30, 4.94 |
| **P + 8E (2 groups)** | **5.74** |
| P + 14E (3 groups: 8P + 8E + 6E) | 5.10 |

CONFIRMED (N=3 each): **P+8E 5.74 / 5.73 / 5.77** (spread 0.7%) vs **baseline 5.30 / 4.94 /
5.27** — **+11%**, and the pinned arm is markedly more reproducible than the P-only baseline.
A single 14-core E team (`0-7;8-21`) measured 5.29, i.e. back at baseline: 8 E-cores in a
second group is the knee, and piling more cores into one E team does not pay.
The 3-group arm being worse is instructive: with m=3 streams and 3 executors, the weak 6-core
group takes a whole call and becomes the critical path — heterogeneous executors want FEWER
groups than in-flight calls so the shared queue can self-balance.

### Next: intra-call P/E split (the single-stream lever)

Executors only help when several calls are in flight, so this round cannot move single-stream
(one companion call at a time). The single-stream version is to split ONE call's expert set
across two pinned teams. It is structurally clean because expert e's whole chain (gate/up ->
SwiGLU -> down-activation quantize -> down) depends only on expert e: partition experts, run
`compute_expert_stages(subset)` on each pinned team concurrently, join, then accumulate in
expert-index order exactly as now. No cross-team barrier, and bit-identity holds by
construction since accumulation order is independent of which team computed a row.

Open question the microbench answers first: the P:E throughput ratio at these shapes, which
sets the split. A rough 8-P-core vs 8-E-core team ratio near 1:0.55 would put the balance
around 5:3 of routed experts and cut CPU compute ~35%. Granularity is coarse (only 4-8 CPU
experts per call), so the split must be measured, not assumed.

### P vs E throughput on the real expert kernels (microbench, cpu_native_check)

Relative to an 8-thread P team (higher = faster):

| qtype | P8 (0-7) | E8 (8-15) | E16 (8-23) |
|---|---:|---:|---:|
| Q2_K | 1.00 | 0.77 | **1.10** |
| IQ3_S | 1.00 | 0.50 | 0.81 |

Two things follow. First, 16 E-cores out-throughput 8 P-cores on Q2_K — E silicon is not a
rounding error on this workload. Second, the ratio is format-dependent (Q2_K's cheap 2-bit
decode suits E-cores; IQ3_S's grid gathers do not), which compounds favourably with the
tail-Q2_K demotion that made Q2_K the dominant CPU-side format. It also means a STATIC
intra-call split would be fragile — the split wants to be dynamic (both teams pulling experts
from a shared atomic counter), which self-balances across formats and core types with no
cross-team barrier.

### Intra-call P/E split: REJECTED, and the microbench that predicted it was measuring the wrong regime

Built the single-stream analog — `ComputeTeams` in the companion: persistent pinned per-core-group
teams pulling experts from a shared atomic counter, dynamic hand-out so no cross-team barrier.
Correctness was fine (ALL GREEN, oracle-identical, inert when unset). Performance was not:

| arm (m=1, tail-Q2_K) | tok/s | CPU compute |
|---|---:|---:|
| P only (8 thr) | 5.43 / 5.41 | 2.915 / 2.946 s |
| P + 8E teams | 4.96 / 4.86 | 3.360 / 3.509 s |
| P + 14E teams | 4.09 | 4.646 s |

Compute got monotonically WORSE as E-cores were added — the opposite of the prediction. Mechanism:
expert weights stream from RAM (16.79 GB of fills per run), so the loop is memory-bandwidth-bound;
E-cores add no bandwidth, only contention for the same controller, plus a per-expert OMP region
entry that replaced whole-subset worksharing. Code reverted under winners-only.

METHOD LESSON (the important part): the P-vs-E microbench that motivated this used
`cpu_native_check`'s fixture, which repeats a single weight row and therefore fits in cache. It
measured ALU throughput in a cache-hot regime and reported E16 > P8 — a ratio that simply does not
transfer to the bandwidth-bound production path. A microbench must reproduce the memory regime of
the path it is used to predict, or its ratios are decoration. This is the same "micro != e2e" trap
the KV-fp8 receipt recorded, in a new costume.

Reconciling with the executor win: pinned executors pay (+11%) because the E team computes ANOTHER
stream's experts while the P team is blocked on io — E-cores fill otherwise-idle time. Intra-call
teams lose because both teams contend for the same bandwidth on the SAME call's experts. E-cores
are useful here for overlap, not for parallel bandwidth-bound throughput.

## Lane 3 CLOSURE (2026-07-25): concurrency ends at parity; the runtime is at its floor

Final honest numbers, mixed prompts (the only regime that represents real serving):

| | tok/s |
|---|---:|
| single-stream | 6.0 (run-gen) / 5.4 (lockstep m=1 path) |
| mixed m=3, pinned P+E executors | **5.74** |
| identical-batch m=3 (shared routing) | 6.9 |

Pinned executors took mixed concurrency from ~0.88x to ~0.96x of single-stream — a real +11%,
but it does not cross 1.0x. Every batching mechanism built for this lane (lockstep, grouped MoE,
multi-row ABI, fused multi-row kernels, m-band attention) is correct and gated, and none of them
makes m concurrent DISTINCT requests cheaper than running them one after another, because
concurrent requests route to disjoint expert sets: each added stream costs a nearly-full expert
load, and only coordination is shared. Concurrency wins only in the identical-batch regime.

The runtime as a whole is now at its floor, and the budget says why:

- **compute 50%** of the step — the paired AVX-VNNI kernels are at the ISA ceiling. This CPU
  exposes `avx_vnni` only: no AVX-512, no AMX (consumer Arrow Lake). No wider instruction exists.
- **io 34%** — 8.45 GB/s effective against a measured ~9.6 GB/s dual-NVMe practical ceiling, i.e.
  88% of the device. Storage backends (io_uring et al.) cannot add bandwidth that is not there.
- **GPU+glue 16%** — long since at SOL for this decode shape.

So the remaining distance to 10 tok/s cannot come from the runtime. It has to come from moving
BYTES or TOKENS:

1. **Artifact axis** — fewer bytes per expert. This is the only lever with a proven large win:
   the tail-Q2_K demotion cut the io wall 37% and gave +23% single-stream (4.8 -> 6.0). Two or
   three further demotion steps of that size reach 10, with a model-quality cost that must be
   measured per step, not assumed. Requires the five-arm quality discipline, not a runtime knob.
2. **Draft head** — a head clearing the 0.80 acceptance bar turns K=4 spec into a ~1.3-1.5x
   multiplier on whatever the runtime does. The gating framework is built and proven (it lifted
   the shipped head 48% -> 77%); the head itself needs training compute the local rig cannot
   supply.

Both are owner decisions — one spends model quality, the other spends money. The engine work
that would consume either is already in place.

## Lane 1: HF survey (2026-07-25) — no ready head exists for Hy3

Searched the Hub for an existing draft/EAGLE head before committing to training:

- **No EAGLE/EAGLE3 head for `hy_v3` exists.** The ecosystem has EAGLE3 drafters for Qwen3.6-35B-A3B,
  MiniMax-M2.7, gemma-4-26B, step-3.5-flash, MiniCPM-SALA and others — none for Hy3/Hunyuan.
  Heads are architecture- and vocabulary-specific (Hy3: hidden 4096, vocab 120832), so none transfer.
- **`canada-quant/hy3-w4a16-mtp`** is tagged `mtp`/`speculative-decoding`, but it is a W4A16 GPTQ
  quantization of full Hy3 that PRESERVES the stock layer-80 MTP head; its listed datasets are
  llmcompressor calibration data, not draft-head training data. Same head we already serve.
- **`num_nextn_predict_layers = 1`** in the pinned source config — the model ships exactly one MTP
  layer and we already use it. No unused speculative capacity.

Conclusion: training is required, and it is self-distillation rather than architecture search.

### The mismatch hypothesis (why a trained head should clear the bar, not just tie it)

The stock MTP head was trained against the ORIGINAL full-precision, full-expert Hy3. What we serve
is REAP50-pruned (half the expert bank removed) and 90.7% Q2_K. The head therefore predicts a model
that is not the one verifying, and every such disagreement is a rejected draft. Self-distillation
against OUR artifact removes exactly that error source — which is why the ceiling for a trained head
is meaningfully above the stock head's 74-77%, not marginally.

Measuring the mismatch directly: acceptance at K=4/PMIN=0.8 on the demoted artifact we now serve vs
the pre-demotion Layer103.5 artifact (the gated sweep's 74-77% was measured on the latter). A drop
on the demoted artifact quantifies the cost the demotion silently imposed on spec, and establishes
the true baseline any trained head must beat.
