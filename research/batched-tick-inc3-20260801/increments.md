# Batched-tick increment 3: exact-16 chunk tier (3a) + deferred token readback (3c) — 2026-08-01

Mission (from increment 2's map): (a) a B=16/32 exactness-tier policy for the batched
serving tick, (c) the host emit-path D2H drop (one D2H per tick, not per seq), (b)
graphed batched tick only if a+c land clean. Box: the local RTX 5090 Laptop (sm_120a,
24463 MiB, 82 SMs — the deployment target), 9B Q8_0
(`~/models/qwen3.5-9b-judge-q8_0.gguf`). Co-resident: a llama-server --embedding
(332 MiB, untouched); one other memra lane interleaved via /tmp/gpu5090.lock (all runs
here are GPU-exclusive under flock).

## (3a) B=16/32 bit-isolation: verdicts per chunk size

Arbiter: decode-batch-gate gate2 (per-seq logits vs isolated, BIT-checked within config)
+ gate3a/b/c. Steps 32 AND 160 (160 crosses the t_kv=96 vec floor so the z-batched seqs
arm engages — inc2 law). Final binary receipts = `fN-*.log`.

| chunk | config | gate2 (bit) | gate3 | verdict |
|---|---|---|---|---|
| 8 (control) | naked | PASS s32+s160 | PASS | the shipped tier, unchanged |
| 12 | q8rp mirror (auto exact-16) | PASS s32 | PASS | exact |
| 16 | q8rp mirror (auto exact-16) | **PASS s32+s160** | PASS | **exact — new tier** |
| 16 | naked (no mirror) | — | — | REFUSED (engine assert; receipt fN-b16-refuse) |
| 16 | naked + cap door (pre-policy) | FAIL step 0, maxdiff 1.3-2.1e-1, 16/16 seqs | FAIL (3b diverges step 8-11) | GEMM tier bit-shifts (receipt dbg-config-b16-*) |
| 32 | cap door + mirror | FAIL step 0, maxdiff 1.3-2.3e-1, all seqs | FAIL | **no exact kernel class exists at m=32** — chunk policy stays <=16 |

Strict mode (equalized env MEMRA_MMVQ=0 MEMRA_NO_FUSE_NORMQ=1): B=1 BIT-IDENTITY vs
decode_step_h through 160 steps + B=4-vs-isolated ALL GREEN on this rig (fN-strict-b4) —
the plumbing pin.

### Which kernel breaks isolation at 16/32 (the attribution chain)

1. `MEMRA_BATCH_FA=0 MEMRA_BATCH_APPEND=0` at B=16: gate2 still FAILs step 0, same
   maxdiff class → NOT the z-batched fa/append kernels (dbg-b16-noattn-s32).
2. `MEMRA_NO_GEMM=1`: unchanged → not (only) `qmatvec_gemm`.
3. Root cause: at m>=16 BOTH `matmul` and `matmul_pre` route Q8_0 projections to the
   **MMQ int8-MMA GEMM (`mul_mat_q`, MEMRA_PP_Q8MMQ default-on since 2026-07-09)** —
   in_f=4096 %256==0 on every 9B projection — and small-out_f tensors (ssm_beta/alpha,
   out_f=32) to `qmatvec_gemm`. Both are block-scale-f32 GEMM classes: per-row results
   != the isolated m=1 mmvq program. `MEMRA_NO_GEMM` never touches the MMQ arm.
4. `NO_GEMM=1 + PP_Q8MMQ=0 + Q8RP=1` (all m>=16 arms off + the b16 mmvq class): gate2
   **PASS bit-checked** at B=16, s32+s160 (dbg-b16-exact16v2-*). B=32 same env: FAIL —
   m=32 exceeds the b16 tier and lands on the dp4a grid.y=m tail (128-thread two-level
   reduce != the m=1 mmvq 32-thread warp reduce).

### The shipped policy (no env juggling)

`decode_batch_exact16_ok()`: every matmul in the batched decode step must have a
bit-exact b16-class kernel — batched-mmvq b16 family (Q4_0/Q6_K native; Q8_0 only as
the `_rp` twin => requires the q8rp split-plane mirror) or the e4m3 mmvq catch-all;
Float matmuls (cuBLASLt) and MoE FFNs disqualify. Qualifying steps at B=9..16 scope
`verify_exact` for the step, which turns off every m>=16 GEMM arm (fp8/f16/MMQ/fp4/
qmatvec_gemm) so the whole step rides the b16 mmvq class. B>8 without the tier stays
refused (assert); MEMRA_DECODE_BATCH_CAP keeps its meaning as the explicit
measurement door. Worker: per-model chunk width = door if set, else 16 iff
exact16_ok, else 8 — **the naked 5090 (no mirror) is byte-for-byte unchanged**.

Prior art note: the pre-inc2 H100 probe (research/darklane-serving-20260801, FLAGS.md
row) had already shown cap-15 b16_rp bit-exactness but measured it SLOWER on the then
per-seq-serial-bound tick ("do not raise as a serving default"). Increment 2 removed
the per-seq serial train; this increment re-swept the stale verdict (H100 law 2) and
adds the m=16 admission (the old cap-16 probe hit the GEMM tier; the verify_exact scope
is what makes 16 itself exact).

### Chunk-size sweep (32 seqs, ceil(32/C) chunked steps/tick, 5 interleaved rounds,
### median of 5 single-rep invocations, prompt 512, steps 128, greedy)

| arm | chunk | env | median agg tok/s | ms/tick | vs c8-naked | vs c8-q8rp |
|---|---|---|---|---|---|---|
| c8-naked | 8 | — | 302.7 | 105.7 | — | — |
| c8-q8rp | 8 | mirror | 320.5 | 99.8 | +5.9% | — |
| **c16-q8rp (EXACT)** | 16 | mirror | **366.5** | **87.3** | **+21.1%** | **+14.4%** |
| c16-gemm (non-exact ref) | 16 | cap door | 227.2 | 140.8 | -24.9% | — |
| c32-gemm (non-exact ref) | 32 | cap door | 341.9 | 93.6 | +12.9% | — |

(raw: chunksweep.log; per-arm spreads tight — c16-q8rp [360.6, 372.8], c8-naked
[279.9, 306.2] with one 279.9 thermal dip in round 4.) Findings: (1) the exact-16 tier
is the fastest measured 32-seq tick config on this rig — it beats even the NON-exact
32-wide GEMM door by +7.2%; (2) the m=16 MMQ/GEMM class is not just inexact but SLOW
at decode shapes (grid starvation at m=16 — 227 tok/s); (3) the mirror alone is worth
+5.9% at chunk 8 (aligned-16B reads, the H100 lane's original rationale). The worker
default (chunk 16 iff exact16_ok) follows directly; the naked 5090 is unchanged.

## (3c) Host emit-path D2H audit + the drop

Audit of what crossed D2H per token per seq in the worker tick (devsample + lean on):

- **Token ids: the only remaining per-token D2H** — `decode_step_batch_sampled_lean`
  ran ONE `dtoh_u32([B])` + stream sync PER CHUNK (4/tick at 32 sessions chunked by 8).
  The readback also serialized the chunks host-side (launch bubble between chunks).
- Stop-check strings: host-only (detok of host-held ids) — no D2H.
- Metrics: host counters, published every 32nd tick — no D2H.
- Non-devsample rows (penalties/top-k/top-p configs): per-row [n_vocab] logits D2H
  remains BY DESIGN (host sampler needs the row; not the load-harness path).
- Retire-time pool park: one D2H per RETIRED session (inc2's lean design) — untouched.

The drop (BUILT, MEASURED FLAT, KILLED): a deferred variant wrote each device-sampled
row's token into a caller-owned per-tick device buffer (same argmax/gumbel launches,
different output slot — values bit-unchanged, greedy-hash + check-batch-exact receipts
identical); the worker accumulated every chunk of the tick there and performed ONE
dtoh_u32 after the last chunk = one D2H + one sync per tick instead of one per chunk.
Serve A/B (N=4 medians, base=seam-off vs defer=on, same binary): c8 377.3 vs 375.3,
c16 369.5 vs 372.2, c32 370.3 vs 368.6 tok/s — FLAT within ±0.7% at every load point,
as the arithmetic predicts (3 saved syncs ≈ 0.1% of a ~100 ms weight-bound tick; at
the new chunk-16 default it is 1 saved sync). Killed per the flags doctrine (flat ⇒
kill the flag and dispatch arm; the JSONL rows are the record). The audit conclusion
stands as the increment's 3c finding: the serving tick's steady-state D2H floor is one
[B]-u32 token readback per CHUNK and NOTHING per seq; the per-tick fold is below
measurement resolution on this rig.

## Serving receipts (single 5090 replica, fresh server per point, arms interleaved)

Exactness (the fleet greedy-hash pattern + check-batch-exact):

| arm | check-batch-exact (16 greedy, batched vs SHARED isolated refs) | greedy-hash |
|---|---|---|
| base (TOKDEFER=0) | PASS 16/16, 0 err | `28ca31bb8fb5aae3` |
| defer (3c naked) | PASS 16/16, 0 err | `28ca31bb8fb5aae3` |
| c16m (mirror, auto chunk-16) | PASS 16/16, 0 err | `28ca31bb8fb5aae3` |

Same prompts, same bytes, all three arms — 3c changes nothing byte-wise and the serve
scheduler's chunk-16 path is byte-identical to isolated (the serving isolation contract
at the new width). Server-log receipt for the auto policy: `[worker] qwen: decode chunk
cap 16 (exact-16 tier)` (server-exact-c16m.log); naked arms log cap 8. Raw:
batch-exact-{base,defer,c16m}.{log,jsonl}, greedy-hash-inc3.log, exact-refs.json.

A/B round (temp 0.7, ~200-tok prompt, 128-tok gens, requests=4c, N=4 per (arm,c),
fresh server per point, gpu5090.lock held per point, raw: serve-points.jsonl /
sv-*.log / server-*.log / metrics-*.json):

Phase 1 — 3c isolate (same binary; base = MEMRA_SERVE_TOKDEFER=0):

| c | base med | defer med | delta |
|---|---|---|---|
| 8 | 377.3 | 375.3 | -0.5% (noise) |
| 16 | 369.5 | 372.2 | +0.7% (noise) |
| 32 | 370.3 | 368.6 | -0.5% (noise) |

→ FLAT; arm killed (see 3c section). All 24 points 0 errors. Aggregate is
c-independent ~370 — the weight-stream-bound signature from inc2 reproduces on the
5090 at chunk 8.

Phase 2 — chunk width isolate (BOTH arms carry the q8rp mirror; c8m pins width 8 via
the door, c16m rides the auto exact-16 policy — server-log receipts):

| c | c8m med | c16m med | chunk-16 delta | c16m vs naked base |
|---|---|---|---|---|
| 16 | 416.4 [405.1,419.2] | **494.5** [472.7,509.9] | **+18.8%** (p50 4.92→4.16 s) | **+33.8%** |
| 32 @ctx8192 | 368.6* | 471.0* | (capacity-clipped cells) | — |
| 32 @MEMRA_CTX=2048 | — | **502.1**, 128/128 ok (N=1) | — | **+35.6%** |

(*) every c=32 mirror point at the default MEMRA_CTX=8192 admitted only ~27 of 32
sessions — 101/128 requests failed with the captured
`cache alloc failed: CUDA_ERROR_OUT_OF_MEMORY` (server also logs `[prime-batch] failed
(CUDA_ERROR_OUT_OF_MEMORY)` fallbacks; concurrent GPU state: 9.5 GB weights + 8.4 GB
mirror + 32×~119 MB ctx-8192 sessions > 24 GB). The 5090 deployment envelope for the
mirror config: ~27 ctx-8192 sessions, OR set MEMRA_CTX to the workload (machine-specific
config knob per the flags doctrine) — at MEMRA_CTX=2048 the same cell runs clean at
502.1 tok/s. Naked (mirror-less) c=32 serves 128/128 at every point.

Post-kill confirmation (final binary, killed 3c arm): base-c16 383.4 | c8m-c16 421.4
(chunk-cap-8 log receipt) | c16m-c16 502.9 — the chunk-16 win stands without the defer
plumbing; check-batch-exact postkill-c16m PASS 16/16 vs the same shared refs;
greedy-hash still `28ca31bb8fb5aae3` (4/4 arms across the day).

## (3b) graphed batched tick — blocker map (not forced)

- The batched step rebuilds the state POINTER TABLE host-side every step (one
  htod_u64): conv/ssm pointers change because the ssm ping-pong swap is a host-side
  `std::mem::swap`. Under capture the table is baked; replay would reuse stale
  pointers. Fix = device-side parity (double-buffer flip via device counter) or
  even/odd twin graphs — a real build, not plumbing.
- Every intermediate allocates through the stream pool per step; capture needs a
  persistent-address buffer set (the GraphDecodeState pattern) replicated at [B,*]
  widths — the dc machinery has no batched twin yet.
- Variable B per tick (admits/retires) + per-seq t_kv rungs: the seqs-fa arm demands
  ONE split rung per batch (straddle => per-seq fallback, which a graph cannot take);
  capture domain = (B bucket, rung segment, kernel-class segment per v0.55.0 rules) —
  at serving churn this recaptures constantly.
- Payoff bound on THIS rig: the 32-seq tick is ~88-105 ms (weight-stream-bound);
  host/launch overhead is a ~1-2% slice — vs +12% from the exact-16 chunk lever that
  landed instead. Graph the tick when B is pinned (dedicated-replica mode), not in the
  general scheduler.

## Gate battery (final binary)

| gate | verdict |
|---|---|
| kernel-check 9B, naked AND MEMRA_Q8RP=1 | ALL GREEN both (kernel-check-inc3.log) |
| run-gen argmax (9B, board-2048), naked AND MEMRA_Q8RP=1 | MATCH both; 32-token streams IDENTICAL naked vs mirror (rungen-9b-inc3.log) |
| decode-batch-gate strict B=4 s160 (equalized env) | ALL GREEN — B=1 bit-identity through 160 steps on this rig |
| decode-batch-gate config B=8 s32+s160 | gate2 bit PASS + gate3 PASS (gate1 = the near-tie roulette, see incidents) |
| decode-batch-gate config B=12/16 + mirror, s32+s160 | gate2 bit PASS + gate3 PASS — no cap door needed (auto exact-16 admission) |
| decode-batch-gate B=16 naked | REFUSED (engine assert — the exact-tier guard) |
| decode-batch-gate B=32 door | gate2 FAIL step 0 (no exact class at 32 — documented wall) |
| check-batch-exact x4 arms (incl post-kill) | PASS 16/16 each, shared refs |
| run-spec K=1..8 | N/A for this GGUF — no MTP head (nextn=0, captured runspec-9b-inc3.log); spec path untouched by this lane |
| cargo test -p memra-server | 7/7 pass |

## Incidents

1. The first gate matrix (dbg-config-*) ran on the pre-3c binary; the rebuild landed
   mid-matrix (same class as inc2's rebuild-race note). The ENTIRE matrix was re-run
   on the final binary (fN-*) — dbg-* logs are retained as the discovery record.
2. decode-batch-gate OOM'd in gate3 at B=16 with the mirror on 24GB (captured
   CUDA_ERROR_OUT_OF_MEMORY; ~18 GB weights+mirror + the harness's ~80 live
   ctx-independent GDN caches). Fixed IN the gate (drop gate2's herd + gate3a scoping)
   — harness capacity, no verdict was wrong.
3. gate1-config (cross-config argmax drift rule, calibrated on H100 2026-07-31) fails
   on this rig via ONE near-tie draw (seed 0, step 1). 18-draw sweep: 15/18 full
   32-step agreement, 1 step-3 WARN, 1 step-19 WARN, 1 step-1 FAIL; strict-mode
   bit-identity 160 steps GREEN. The plumbing-bug signature (step 0-2 on EVERY draw)
   is absent — this is the accepted near-tie roulette; the 6-seed rule needs a 5090
   recalibration pass (left for the gate's owner lane; gate2/gate3 arbitrate here).
4. zsh env-splitting bug in an INLINE post-kill point: `env $VAR` with
   VAR="MEMRA_Q8RP=1 MEMRA_DECODE_BATCH_CAP=8" does not word-split under zsh — the
   whole string became MEMRA_Q8RP's value, CAP stayed unset, and the point silently
   ran the auto chunk-16 policy. Caught by the server-log `decode chunk cap` receipt
   (the point read 505.4 ≈ the c16m twin — serve-points.jsonl label
   `c8m-c16-r5postkill` is really a second c16m sample; the corrected bash re-run
   `c8m-c16-r5bpostkill` = 421.4 with the cap-8 log line). The phase-1/2 script ran
   under bash (correct splitting, per-point cap receipts verified). Rule: pass env as
   explicit `env K=V K=V` argv, never a single string through a shell variable.
