# spec-scaling — why serve-spec does not scale with concurrency

Lane: `lane/spec-scaling`, task #88 (raised by `research/pp2-spec-20260806/RESULTS.md`).
Box commit `8711a9209f1a38f4f97618e92a80f170b1576759`, `NVIDIA GeForce RTX 5090 Laptop GPU, 595.84`
(`BOX-COMMIT.txt`). Model: `Qwen3.5-9B-NVFP4-MTP-GGUF.gguf` + `draft-9b-owntrim-nvfp4head-q4blk.gguf`
(the accept-gate q9 cell's production drafter, attached via `MEMRA_MODELS "+draft"` so it REPLACES
the embedded head — a bare embedded head is a different acceptance regime). `MEMRA_CTX=4096`,
`MEMRA_SPEC_K=3` (the serve default, `worker.rs:3262`), greedy, one card, PP door shut.

**VERDICT: BRIEF-ONLY. No code change.** The serialization point is named below and it is not a
bug — it is the absence of a batched-verify entry point. A batched cross-session verify is
*buildable*, but the exactness contract caps it at **16 pooled columns total** (4 sessions at the
production K=3) and the measured amortization inside that cap bounds the whole fix at
**1.27x (draft left serial) to 1.44x (draft also batched)**. Spec-OFF already scales **3.77x** on
the same box. A 1.3x fix on the losing arm does not reach the winning arm, so the honest product
answer is a concurrency-gated spec policy, not a batched verify. Details and the seam that a
future lane would need are in §5-§7.

---

## 1. The serialization point

Two lines, both in `crates/memra-server/src/worker.rs`:

```
worker.rs:1686        for i in spec_order {
worker.rs:1871                        && active[i].spec.is_none() && active[i].prefill_done
```

- **`worker.rs:1686`** — scheduler phase (a). Every spec-capable session is stepped in a **serial
  host loop**, and each `step_session` call (`worker.rs:3236`) runs that session's **whole burst**
  — up to `MEMRA_SPEC_BURST`=32 tokens, i.e. ~8 rounds at K=3 (`worker.rs:3260`) — to completion
  before the next session is touched.
- **`worker.rs:1871`** — scheduler phase (c), batched decode. The filter `active[i].spec.is_none()`
  **excludes spec sessions from batching entirely.** Spec sessions are structurally barred from
  `decode_step_batch_sampled_lean_masked` (`worker.rs:1964`).

So concurrency on the spec path is round-robin at burst granularity over a single-session engine
entry point. `generate_spec_session_sampled` (`spec.rs:2874`) takes one `&mut SpecSession`, one
cache, one `t`; `decode_step_t_core_stream` (`spec.rs:1316`) — the single funnel every verify
forward reaches — has no batch axis at all.

**Hypothesis (b) from the brief is correct, in its strongest form.** There is no engine mutex, no
CUDA-stream contention, and no draft-model bottleneck. There is simply **no batched verify to
enter**, and the batched path actively filters spec rows out.

## 2. Mechanism receipt

### 2.1 The c-ladder (N=3 interleaved, rep-major, arm order alternating by rep parity)

`run-cladder.sh` → `logs/cladder/points.jsonl` (24 points), server logs per arm per rep.
Medians of N=3; GPU 56C/12W pre, 74C/21W post (`logs/cladder/gpu-{pre,post}.csv`) — warm, no
thermal cliff. Spread is tight enough that the medians are not close calls (worst arm spread
1.6%).

| c | spec ON agg tok/s | spec OFF agg tok/s | S/N | spec scale vs c=1 | spec-OFF scale vs c=1 |
|---|---|---|---|---|---|
| 1 | 253.2 | 139.4 | **1.82x** | 1.00x | 1.00x |
| 2 | 252.3 | 223.2 | **1.13x** | 1.00x | 1.60x |
| 4 | 251.2 | 386.7 | **0.65x** | 0.99x | 2.77x |
| 8 | 249.7 | 525.9 | **0.47x** | **0.99x** | **3.77x** |

Spec is not merely equal at the endpoints — it is **monotonically flat, slightly negative**
(253.2 → 249.7, -1.4% across an 8x load increase). This reproduces the predecessor lane's
PRO-6000 finding (346.5 → 345.2 flat vs 223.7 → 872.9 scaling) on a different card with a
different model, confirming it as a **single-card property of the scheduler**, not a PP artifact.

Per-session latency is the other half of the proof:

| c | spec p50 | vs solo | spec-OFF p50 | vs solo |
|---|---|---|---|---|
| 1 | 0.510s | 1.00x | 0.917s | 1.00x |
| 2 | 1.023s | 2.01x | 1.147s | 1.25x |
| 4 | 1.899s | 3.73x | 1.322s | 1.44x |
| 8 | 3.951s | **7.75x** | 1.949s | **2.12x** |

Spec p50 is **linear in c** (1.00 / 2.01 / 3.73 / 7.75). That is the signature of a queue, not of
resource contention: each session pays exactly its solo cost and waits its turn.

### 2.2 `ready=0`: spec never enters the batched path

`MEMRA_TICK_TRACE=1` prints `[tick] act= int= priming= ready= decode_ms=`, where `ready` is the
count of rows entering phase (c) batched decode (`worker.rs:1994`). Distinct tick shapes:

```
r1-S-spec-server.log     34 act=8 int=8 priming=0 ready=0     (101 ticks total, ready=0 on ALL)
r1-N-nospec-server.log  508 act=8 int=8 priming=0 ready=8     (3392 ticks)
```

Every tick of the spec arm at every concurrency: `ready=0`, `decode_ms=0.0`. Direct observable for
the filter at `worker.rs:1871`.

Note the tick counts: 101 ticks for the spec arm vs 3392 for spec-OFF over the same 60 measured
requests. One tick = one full sweep of whole bursts.

### 2.3 Per-round cost is FLAT under 8x load — serialization, not contention

`MEMRA_SPEC_PHASE=1` per-burst decomposition, divided by that burst's `rounds`. A **separate
server boot per concurrency** (`logs/phase/ph-c{1,2,4,8}-server.log`) so no cross-c contamination.
Medians, N = number of `[spec-phase]` bursts:

| c | N | round ms | draft | verify-issue | verify-wait | commit-host |
|---|---|---|---|---|---|---|
| 1 | 17 | 10.663 | 1.493 | 1.500 | 7.440 | 0.230 |
| 2 | 38 | 10.700 | 1.490 | 1.500 | 7.480 | 0.230 |
| 4 | 77 | 10.725 | 1.480 | 1.530 | 7.475 | 0.240 |
| 8 | 153 | 10.758 | 1.480 | 1.562 | 7.475 | 0.240 |

**1.009x round cost across an 8x load increase**, while p50 latency rose 7.75x. If sessions were
contending for a shared resource, the round would inflate. It does not move. The engine does the
same work at the same speed; the sessions are simply queued behind each other.

Round budget at K=3 (T=4), c=1: verify-wait 7.44 ms (**69.8%**), verify-issue 1.50 ms (14.1%),
draft 1.49 ms (14.0%), commit-host 0.23 ms (2.2%). **The verify forward is 83.9% of the round and
is the only phase worth batching.**

### 2.4 Lockstep at burst granularity

`[spec-acc]` liveness lines at c=8 (`logs/cladder/r2-S-spec-server.log`) show eight identical
bursts back-to-back per context level, e.g. `ctx=328 burst=19/24 cum=88/120=0.733` repeated 6x
then `ctx=294 burst=20/42` — round-robin, one whole burst each, in index order.

Acceptance is identical across reps and unaffected by concurrency (`logs/cladder/r*-S-spec-metrics.txt`):
r1 `acceptance_rate=0.7351, tokens_per_round=3.2053, accept_rate_per_pos=[0.826, 0.752, 0.627]`;
r2 `0.7351 / 3.2054 / [0.826, 0.752, 0.628]`. Concurrency changes nothing about the spec math —
only the scheduling.

## 3. What the fix would have to beat: the T-amortization curve

A pooled verify's whole value is that verify cost grows sublinearly in T. Measured at c=1 by
sweeping `MEMRA_SPEC_K` and reading per-round `[spec-phase]` medians (`logs/ksweep/`, N=17 bursts
per cell, one server boot per K). verify = issue + wait; T = K+1 columns:

| K | T | round ms | draft | v-issue | v-wait | verify | **ms/column** |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 8.789 | 0.565 | 1.559 | 6.459 | 8.018 | 4.009 |
| 2 | 3 | 9.678 | 1.040 | 1.487 | 6.933 | 8.420 | 2.807 |
| 3 | 4 | 11.582 | 1.650 | 1.557 | 8.137 | 9.695 | 2.424 |
| 5 | 6 | 15.280 | 2.614 | 1.617 | 10.762 | 12.378 | 2.063 |
| 7 | 8 | 17.436 | 3.325 | 1.725 | 12.086 | 13.811 | 1.726 |
| 11 | 12 | 26.281 | 5.013 | 1.755 | 19.214 | 20.969 | 1.747 |
| 15 | **16** | 33.852 | 6.714 | 1.800 | 25.038 | **26.838** | **1.677** |
| 16 | **17** | 74.420 | 8.029 | 1.650 | 64.429 | **66.079** | **3.887** |
| 17 | 18 | 78.789 | 8.629 | 1.673 | 68.175 | 69.848 | 3.880 |
| 19 | 20 | 87.352 | 9.629 | 1.675 | 75.736 | 77.411 | 3.871 |

### THE HARD WIDTH CEILING IS T=16

T=16 → 1.677 ms/column. T=17 → **3.887 ms/column**. A **2.3x per-column cliff between 16 and 17**,
and everything above stays on the bad side (3.88 / 3.87). The cliff matches
`matmul_decode_exact`'s batched-weight-resident gate exactly:

```
lib.rs:5846        if (2..=16).contains(&m) && self.batched_supports(qtype) && self.mmvq_supports(qtype)
```

At m<=16 one weight read serves m columns. At m>16 the dispatch falls to `qmatvec_mmvq` with
`grid.y=m` — **m full weight reads per launch**. This is not a tuning threshold: above m=16 there
is no exact kernel class (`decode_batch.rs` module law: "B > 16 crosses into GEMM/dp4a-tail numeric
configs with NO exact kernel class — refused"), so raising it breaks verify==decode bit-identity,
which is the entire basis of the greedy accept walk.

**Consequence: a pooled verify may carry at most 16 columns across ALL sessions.** At the
production K=3 (T=4) that is exactly **4 sessions per shared launch**. c=8 would need two pooled
groups run serially; c=64 sixteen.

## 4. The ceiling arithmetic

Using measured numbers only. Serial baseline at K=3: 10.725 ms/round/session, 3.205 accepted
tokens/round.

**4 sessions, serial (today):** 4 x 10.725 = **42.90 ms** for 4 rounds.

**4 sessions, pooled verify at T=16, draft+commit left serial:**
- draft: 4 x 1.480 = 5.92 ms (per-session MTP chain, own `draft_ctx` + own persistent graph)
- verify: 26.838 ms (the measured T=16 cell)
- commit: 4 x 0.240 = 0.96 ms (per-session `j`, see §5.3)
- **= 33.72 ms → 1.27x**

**4 sessions, pooled verify AND B-way draft** (a second new kernel family):
- draft ~1.9 ms + verify 26.838 + commit 0.96 = **29.70 ms → 1.44x**

Past c=4 the gain is flat (ceil(c/4) pooled groups run serially), so at c=8 spec-ON would move
from the measured 249.7 to roughly **317-360 tok/s**. Spec-OFF is **525.9** on the same box in the
same run. **The fix loses to the existing default by 1.5-1.7x at c=8**, and the gap widens with c.

That is the whole verdict. Everything below is what a future lane would need if the calculus
changes (e.g. a much larger K, or a model where spec-OFF batching is weaker).

## 5. Seam analysis — what a batched verify actually needs

### 5.1 GDN batched scan: REACHABLE (correcting an earlier read)

`gdn_scan_s128_batched` (`lib.rs:11095`) looks like a blocker — it takes `b_n` and **no `t`**,
i.e. B sequences x 1 token. But the kernel it launches is a thin wrapper over the same template
the single-seq path uses:

```
hybrid.cu:1366  extern "C" __global__ void gdn_scan_s128_b(...) {
                    int b = blockIdx.y;
                    size_t row = (size_t)b * H * 128;      // [B, H*S_v] activation rows (T=1)
                    size_t sc  = (size_t)b * H;            // [B, H] scalar rows
                    gdn_scan_kernel<128, 32>(q + row, ..., state_ins[b], state_outs[b], o + row, H, 1, scale);
                }
```

`gdn_scan_kernel<S_v, WARP>` (`hybrid.cu:337`) already **takes `T` and loops `for (int t = 0; t < T; t++)`
sequentially per column with the state carried in registers** — the batched wrapper simply passes
`1`. A B-way x T-token verify scan is a ~5-line new entry point: stride by `b*T*H*S_v` /
`b*T*H`, pass `T`. Each `(b, h, col)` warp then executes the exact register program the single-seq
`gdn_scan_s128(..., T, ...)` call runs, so it is bit-identical by construction. **GDN is not the
obstacle.**

### 5.2 FA: THE MISSING KERNEL — the two batch axes do not compose

memra has each axis separately and neither has both:

| kernel | axis | per-row key bound | per-seq KV base | split rung |
|---|---|---|---|---|
| `fa_decode_rows` (`lib.rs:9762`) | T causal **rows**, grid.z=row | `base_len + r + 1`, host `base_len` | **one** k/v view | grouped per row-range |
| `fa_decode_batch_seqs_v4` (`lib.rs:9645`) | B **sequences**, grid.z=seq | `pos_seq[z] + 1` | `kv_ptrs` [2B] table | **one** `split_keys` for all |

A pooled verify needs `B sessions x T rows` with **per-session** KV pointer, **per-session**
`base_len` (`spec.rs:2583`: `let base_len = kvl.len - t;` — a per-session quantity), and the
split-ladder straddle law honored **per session**:

> `fa_decode_rows` (lib.rs, LADDER-RUNG STRADDLE FIX, issue #10): "one sp for every row diverges
> from eager decode when a split-ladder rung falls INSIDE the batch — row r's eager twin used
> `fa_split_keys(t_kv_r)`, the batch used `fa_split_keys(t_kv_max)`, and the different partition
> changes the combine's FP order (greedy tie flips at depth)."

With B sessions at independent depths, one launch cannot carry one `sp`. The grouping becomes
per-(session, row-range) — up to 2B groups on crossing rounds — and `fa_decode_combine_seqs`
takes a single scalar `split_keys`. **This is the real new work: a `fa_decode_vec_q_rows_seqs`
kernel + combine with a per-seq base table and per-(seq,rung) grouping.** It is the one piece with
no precedent to extract from.

### 5.3 Commit stays per-session (cheap, so: fine)

`commit_verified_prefix` (`spec.rs:2104`) rolls back per-layer `kvl.len = saved + j` and replays
`ssm_conv_ring_rebuild` + `gdn_scan_s128` from the `VerifyCkpt`, with a per-session `CacheSnapshot`.
`j` (accept length) differs per session per round, so B serial commits are unavoidable. Measured
cost 0.24 ms of a 10.7 ms round (**2.2%**) — so leaving it serial costs ~1 ms at B=4. Not a
blocker; just does not amortize.

### 5.4 `verify_layers` must be mirrored, not duplicated

`verify_layers` (`spec.rs:1570`) carries the per-layer dispatch-mirroring state whose bit-identity
IS the exactness contract: the `pending: Option<(CudaSlice<f32>, CudaSlice<f32>)>` cross-layer
add+norm+q8 fusion carry, the per-layer `mixer_fast` / `norm_fused` / `lin_q8_only` picks, the
`t>=3 || (t==2 && spec_m2())` batched-linear window. A B-way variant must reproduce every one of
those per (session, column). The pp2-batch lane's `decode_batch_layers` + range-scoped
`BatchLayerCtx` (`decode_batch.rs:36`, `:581`) is the right structural template, but it mirrors
the **decode** dispatch, not the verify dispatch — the two are deliberately distinct code paths.

### 5.5 Width bookkeeping

`decode_batch_exact16_ok` (`decode_batch.rs:149`) is an ALL over ~500 matmuls and already exists
as the "may I use 16 columns" predicate; a pooled verify would gate on it plus a running
`sum(T_i) <= 16` admission rule, refusing to pool a session whose K would overflow the tier
(rather than padding, which would change per-row `base_len` and therefore the FA program).

## 6. The cheap alternative, and why it is not this lane's to ship

The ladder says spec ON wins at low concurrency and loses at high, with the crossover **between
c=2 and c=4**:

```
c=1  1.82x WIN      c=2  1.13x win      c=4  0.65x LOSS      c=8  0.47x LOSS
```

So the product-correct behavior is a **concurrency-gated spec policy**: keep spec for c<=2, take
the batched path above. The eligibility predicate is already a single per-admit expression
(`spec_eligible`, `worker.rs:2461`) and the tick already runs phase (a) and phase (c) in the same
sweep, so mixed spec+batched sessions are structurally supported today.

Two reasons this lane does not ship it:

1. **Demotion is not free.** A session admitted spec has `Session.cache == None` — the
   `SpecSession` owns the caches (`worker.rs:974` comment: "legacy tokenwise cache — None on the
   spec path"). Gating at admit only affects *new* arrivals; the already-admitted spec sessions
   keep serializing. Draining-then-demoting needs a cache handoff.
2. **The mixed tick is unmeasured.** Phase (a) runs whole bursts before phase (c) is reached, so 2
   spec sessions holding ~21 ms of serial burst per tick would inflate the batched rows' TTFT and
   inter-token latency. That interaction has no receipt in this lane and a policy shipped without
   it would be a latency regression dressed as a throughput fix.

Both are scheduler-policy work with their own gate battery (serve-smoke, serve-stress c=64, the
lane ladder), i.e. a lane, not a footnote to this one.

## 7. Honest disclosure: one non-clean point

`S-spec-c8-r2` reports `n_ok=31, n_err=1` with the error, quoted from `points.jsonl`:

```
HTTPError: HTTP Error 503: Service Unavailable {"error":{"message":"server is draining
(shutdown in progress); retry","type":"server_error",...}}
```

Server-side, quoted from `logs/cladder/r2-S-spec-server.log:496`:

```
[server] SIGTERM: draining (8 in flight, deadline 30s)
```

A SIGTERM reached that server mid-point. Admit counts confirm the scope precisely: r1-S and r3-S
each admitted 64 requests (4 points x (1 warmup + 4/8/16/32)); r2-S admitted **63** — exactly the
one request that got the 503 never entered the engine. The 31 that did complete normally, and the
point's 249.74 tok/s sits inside the r1/r3 spread (249.10-253.18), so the median is unaffected.

**The sender of that SIGTERM is not identified and I will not infer it.** The harness's own
`kill $pid` runs only after the c-loop and the metrics curl; no `pkill -f memra-server` appears in
this lane's scripts, in `tools/serve-smoke.sh` / `tools/serve-stress-gate.sh` (both kill a captured
`$SPID` only), or in any concurrent lane's tooling that I could find. Per the evidence law this is
recorded as "SIGTERM received mid-point, sender unknown", not as a diagnosis. It is a harness-
lifecycle artifact, not an engine fault: no CUDA error, no OOM, no worker death anywhere in the
log, and the arm's own acceptance metrics are identical to the other two reps.

## 8. Files

| path | what |
|---|---|
| `run-cladder.sh` | the N=3 interleaved c-ladder (arms S/N x c=1,2,4,8), stale-listener + pid-ownership guards |
| `logs/cladder/points.jsonl` | 24 raw load points |
| `logs/cladder/r{1,2,3}-{S-spec,N-nospec}-server.log` | per-arm server logs (tick trace, spec-acc, drains) |
| `logs/cladder/r*-*-metrics.txt` | `/metrics` acceptance snapshots per arm |
| `logs/cladder/gpu-{pre,post}.csv` | thermal regime bounds |
| `logs/phase/ph-c{1,2,4,8}-server.log` | `MEMRA_SPEC_PHASE` per-round decomposition, one boot per c |
| `logs/ksweep/k*-server.log` | the T-amortization sweep (K=1..19), one boot per K |
| `logs/quick/` | the first two-point reproduction (superseded by the ladder) |

## 9. Answer to task #88, in one paragraph

Serve-spec does not scale with concurrency because spec sessions are stepped one whole burst at a
time in a serial host loop (`worker.rs:1686`) and are explicitly excluded from batched decode
(`worker.rs:1871`); the engine has no batched-verify entry point (`spec.rs:1316` /
`spec.rs:2874` are single-session all the way down). It is a queue, not contention — per-round cost
moves 1.009x under an 8x load increase while per-session p50 goes 7.75x, and `[tick] ready=0` on
every spec tick. Building the batched verify is possible (GDN needs a 5-line B-way-x-T entry point;
commit and draft can stay serial) except for one genuinely new kernel — an FA that batches
**sessions and causal rows together** with per-session key bounds and per-session split-ladder
grouping (`lib.rs:9762` has rows, `lib.rs:9645` has seqs, nothing has both). It would not be worth
it: the decode-exact width tier caps a pooled verify at 16 columns (`lib.rs:5846`; measured cliff
1.677 → 3.887 ms/column between T=16 and T=17), which is 4 sessions at the production K=3 and
bounds the entire fix at 1.27-1.44x on an arm that spec-OFF already beats 2.1x at c=8. The
concurrency crossover is between c=2 and c=4; the shippable answer is a gated spec policy, which
is scheduler work with an unmeasured mixed-tick latency interaction and belongs in its own lane.
