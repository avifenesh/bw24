# q27 Q8_0 extreme-perf deep dive — PHASE 1 verdict

> **5090-ARBITER ADDENDUM (2026-08-05, `local5090/VERDICT.md`).** The shipping arbiter changed
> lever 2's final form: the 48 graph key **does not transfer** to the 82-SM local rig (q27
> graph arm −1.61% at n=128, still negative through n=512), so the shipped default is
> **SM-gated** — `budget >= 48` at `sm_count() >= 180`, `budget >= 256` everywhere else.
> References to "landed at 48" below describe the pod-lane tree, not the final shipped form.
> Lever 1 measured flat (order-paired −0.04%) on 82 SM and ships default-ON everywhere
> (bit-identical, big-rig +0.94%). Full local gate battery ALL GREEN; receipts in `local5090/`.

Rig: **pro6000wk-runpod-community** (RTX PRO 6000 Blackwell WK 96GB, 188 SM, driver 570.211.01,
510W cap, mem clock droops 13365/14001 → ~1711 GB/s effective, 89C under spin).

> **BOARD CAVEAT.** This is a *community* RunPod board, measured ~5-11% below the 20260804
> prod-class board at identical code (q8 d512 49.82 here vs 52.61 there = −5.3%; pp512 4091.9 vs
> 4591 = −10.9%). **Relative deltas are the currency**; every absolute row here gets re-minted on
> prod-class silicon before it goes near a published board.

Artifact: `Qwen3.6-27B-Q8_0.gguf`, 28,595,763,424 bytes,
sha256 `f93f517f38e696d35a1a7df2c0e3155a64f4c4dcd662107a146ae263f7fb14ce`
(`unsloth/Qwen3.6-27B-GGUF`, same provenance as the prod battery).
Prompt: `research/e2e/prompts/pp512.txt` (512 tokens — the prod-anchor denominator);
long-prompt cells use `research/e2e/prompts/p3-agentic-long.txt` (6257 tokens).

Scope: plain decode + prefill + serve. MTP/drafter is phase 2 and is **not** measured here.

> **BUILD PROVENANCE (disclosed, verified benign).** The pod tree `/root/bw24` is an rsync'd working
> copy, not a git checkout, and it was left warm by the preceding `pro6000-dev-20260804` lane. Every
> binary in this report therefore also carries that lane's **two uncommitted NVFP4 measurement
> seams** (`MEMRA_NV_DUAL` in `matmul_pre_dual_noscale`, `MEMRA_NV_MR` in the mmvq mr selector; patch
> at `research/pro6000-dev-20260804/pod-receipts/dev-seams.patch`), which are absent from this lane's
> diff. Both are **default-preserving** (unset → dual fusion ON, `mr=2`, i.e. the shipped defaults)
> and both are **gated on `qtype == QT_NVFP4`**, so they cannot touch the Q8_0 path that produces
> every number in §1-§4. Neither variable was ever set in this lane's runs. Verified by
> comment-stripped diff of pod-vs-local across all five touched files: `decode.rs`,
> `decode_batch.rs`, `qmatvec.cu`, `kernel_check.rs` are **code-identical** to the local tree, and
> `lib.rs` differs *only* by those two prior-lane seams. Consequence for lever 2's NVFP4-MTP column
> (n=16..128): those cells ran with the seams present at their default values — the arms are
> internally consistent (both sides of every A/B carried the same binary), but the NVFP4 absolute
> tok/s are not directly comparable to a clean-tree build.

---

## 0. Gates (correctness before numbers)

All gates re-run on the **final** tree (lever 1 landed + lever 3 call site reverted + lever 2 key at 48):

| gate | arm | result |
|---|---|---|
| `run-gen` prefill-vs-decode argmax | final tree | 791/791 **MATCH** |
| `run-gen` batched-prime vs tokenwise | final tree | **MATCH** |
| `kernel-check` full battery | final tree | **ALL GREEN**, 8 `bits=true` cells (incl. new `Q8-FUSED2-B m=5, m=8`) |
| `decode-batch-gate` exactness battery | fuse ON and OFF | **ALL GREEN** both arms |
| `run-spec` **K=1..8** self-consistency | final tree, MTP artifact | **PASS all 8 K** (`=== SELF-CONSISTENCY PASS ===`) |
| `graph-decode-gate` (256 steps) | key=48 tree | **BIT-IDENTICAL** vs `decode_step`, buckets=16, captures=2 |
| `graph-session-gate` (96 tokens) | key=48 tree | **PASS** |
| token-stream identity, 128 tokens | fuse OFF vs ON | **IDENTICAL** |

`run-spec` K=1..8 receipt, board caveat applies to the tok/s only — the gate is the
self-consistency column:

| K | acceptance | tok/s (vs generate) | self-consistency |
|---|---|---|---|
| 1 | 15/16 = 93.8% | 115.83 (1.53x) | PASS |
| 2 | 19/24 = 79.2% | 140.84 (1.86x) | PASS |
| 3 | 24/30 = 80.0% | 151.91 (2.01x) | PASS |
| 4 | 23/36 = 63.9% | 148.10 (1.96x) | PASS |
| 5 | 25/45 = 55.6% | 135.88 (1.80x) | PASS |
| 6 | 27/54 = 50.0% | 126.39 (1.67x) | PASS |
| 7 | 29/63 = 46.0% | 115.98 (1.53x) | PASS |
| 8 | 23/64 = 35.9% | 78.76 (1.04x) | PASS |

Note for phase 2: `run-spec` on the **Q8_0** artifact exits rc=2 —
`ERROR: model has no MTP/NextN head (nextn_predict_layers=0, no blk.N.nextn.eh_proj)`. The K sweep
is therefore run on `Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf`, which shares the engine code paths this
lane changed. Q8_0 + MTP would need an artifact rebuild — a phase-2 input, not a phase-1 gap.
Receipt: `logs/gate-r3-runspec-q8.log`.

## 0b. What could NOT be measured

**ncu is hard-blocked on this pod.** `ERR_NVGPUCTRPERM`; `/proc/driver/nvidia/params` carries
`RmProfilingAdminOnly: 1`, `capsh` shows `!cap_perfmon` and `!cap_sys_admin`, no `modprobe`.
Consequence: **achieved occupancy per top-5 kernel is unavailable** — the task asked for it and it
cannot be produced on this container. Achieved bandwidth below is therefore **derived
analytically** (nsys kernel duration × exact byte footprint), explicitly *not* ncu-measured.

---

## 1. Kernel-share tables

### 1a. Decode c=1 (`MEMRA_PROFILE_GEN=2`, decode loop only, prime excluded)

Top 13 by rank, verbatim from `nsys/nsys-q8-decode-c1_cuda_gpu_kern_sum.csv` (no rows skipped):

| # | share | kernel |
|---|---|---|
| 1 | **73.8%** | `qmatvec_q8_0_mmvq` |
| 2 | 15.2% | `qmatvec_q8_0_mmvq_fused2` |
| 3 | 4.2% | `qmatvec_q8_0_mmvq_fused3` |
| 4 | 1.1% | `add_rms_norm_q8_1` |
| 5 | 1.0% | `rms_norm_q8_1` |
| 6 | 0.8% | `gdn_scan_s128` |
| 7 | 0.5% | `silu_mul_scaled_q8_1` |
| 8 | 0.4% | `add_f32` |
| 9 | 0.4% | `ssm_conv1d_fused_decode_f32` |
| 10 | 0.4% | `fa_decode_vec_q_v4_deep_dc` |
| 11 | 0.4% | `gated_rmsnorm_q8_1` |
| 12 | 0.3% | `gdn_prep_decode_f32` |
| 13 | 0.3% | `fa_decode_combine_f32` |

**93.2% of the decode tick is three Q8_0 mmvq kernels.** Attention is **0.7% combined** (every
`fa_*` kernel in the whole profile: 0.4 + 0.3) — the 27B head geometry is *not* a decode lever.
There is no glue hotspot either: **nothing outside the matvec family clears 1.1%**, and the entire
tail from rank 4 down sums to under 7%. This is the structural reason lever value in this lane comes
from *launch count and graph replay*, not from kernel rewrites.

Timeline: **1015 kernel launches/token**, busy **92.52%**, idle gaps **7.48%**, median gap 1312 ns.

Launch accounting, so the number is unambiguous: the baseline trace has 136,213 GPU rows over 128
tokens, of which **129,920 are kernels (1015/token)** and 6,293 are `[CUDA memset]` /
`[CUDA memcpy]` (49.2/token — 6,146 of them memsets). Every "launches/token" figure in this
document counts **kernels only**; the memset/memcpy count is identical in both lever-1 arms, so it
never enters a delta.

### 1b. Derived achieved bandwidth (analytic — ncu blocked)

Reproduce with `python3 scripts/derive-bw.py` — every row below is recomputed from the committed
trace CSV, not transcribed. Kernel classes are keyed by **grid dimension** (`GrdX` = output rows,
one warp per row), because several distinct tensor shapes share the `qmatvec_q8_0_mmvq` symbol.
Shapes are authoritative from the GGUF header; Q8_0 `row_bytes = (in_f/32)*34`.

| tensor class | launches/tok | med ns | MB/launch | GB/s |
|---|---|---|---|---|
| lm_head (5120→248320) | 1 | 821529 | 1350.86 | **1644.3** |
| `fused2` (qkv+gate, 5120→16384, 48 GDN layers) | 48 | 56383 | 89.13 | **1580.8** |
| `fused3` (q/k/v, 5120→14336, 16 full-attn) | 16 | 49568 | 77.99 | **1573.4** |
| ffn_down (17408→5120) | 64 | 60224 | 94.70 | **1572.5** |
| ffn gate+up, separate launches (5120→17408) | 128 | 60383 | 94.70 | **1568.3** |
| attn_out / ssm_out (6144→5120) | 64 | 22143 | 33.42 | **1509.4** |
| `ssm_alpha`+`ssm_beta` (5120→96, tiny fused2) | 48 | 2848 | 0.52 | **183.4** |
| **weighted aggregate, whole matvec family** | — | — | 27.223 GB/tok | **1557.3** |

Against the *drooped* ~1711 GB/s effective board bandwidth the aggregate is **91.0% of achievable**,
and every individual class sits between 88% and 96%. The matvec family moves 27.223 GB/token in
17.481 ms of the 20.07 ms token.

**So the "last 16%" is not a bandwidth hole in the kernels.** The weight-streaming classes are
already at 88-96% of a board whose memory clock is drooping. What is left is (i) the **7.5%
launch-gap tax** — which is exactly what levers 1 and 2 attack, and together they took +4.8% of it —
and (ii) the small-shape classes.

**New finding out of the re-derivation** (H7 below): the `ssm_alpha`+`ssm_beta` fused2 launch runs at
**183.4 GB/s, 8.5x below the streaming classes**. It is only 0.52 MB and 2848 ns, so it is *not* a
throughput problem — at 48 launches/token it is **136.7 µs/token = 0.68% of the 20.07 ms tick**, i.e. latency-
and launch-bound on a 96-row output, not bandwidth-bound. It is the single clearest "this shape never
reaches streaming width" case in the profile, and it is a *fusion* candidate (fold alpha/beta into
the neighbouring GDN projection) rather than a kernel-tuning one.

> **Correction, logged rather than quietly fixed.** An earlier draft of this table carried
> `attn/gdn out = 1257.9` and `aggregate = 1538.8 GB/s (26.867 GB / 17.459 ms)`. Re-deriving from
> the committed trace gives **1509.4** and **1557.3 (27.223 GB / 17.481 ms)**. The other five class
> figures reproduced to the decimal. The two that moved had no committed derivation, which is why
> `scripts/derive-bw.py` now exists and is the source of this table — a number nobody can recompute
> is not evidence. This does not change any verdict: the conclusion was and is that the kernels are
> near the board's drooped bandwidth and the recoverable slack is launch overhead.

### 1c. Prefill pp512

| share | kernel |
|---|---|
| **75.1%** | `mul_mat_q_q8_0<128,0>` (int8-MMA GEMM) |
| 5.0% | `gdn_chunk_state_f32` |
| 2.9% | `qmatvec_q8_0_dp4a` |
| 1.9% | `gdn_chunk_output_f32` |
| 1.8% | `quantize_mmq_q8_1_d4_q8_0` |
| 1.6% | `silu_mul_f32` |
| 1.6% | `rms_norm_f32` |
| 1.4% | `fa_prefill_bf16kv_pp` |

Top 8 by rank from `nsys/nsys-q8-pp512-default_cuda_gpu_kern_sum.csv`; the cut is a truncation, not
a filter, and it lands on a tie — rank 9 is `l2_norm_f32`, also 1.4% (3,075,965 ns vs
`fa_prefill_bf16kv_pp`'s 3,104,956 ns).

Prefill is one GEMM class. The GDN (linear-attention) chunk kernels are the only non-GEMM
double-digit-adjacent block at 6.9% combined (5.0 + 1.9). No launch-overhead story at pp512.

### 1d. Decode c=8 (batched tick)

| share | kernel |
|---|---|
| **73.2%** | `qmatvec_q8_0_mmvq_b8` |
| 11.1% | `mul_mat_q_q8_0<128,0>` ← **prime phase**, not the tick |
| 3.0% | `add_rms_norm_f32` |
| 2.8% | `gdn_scan_s128_b` |
| 2.7% | `rms_norm_f32` |
| 1.3% | `fa_decode_vec_q_seqs_v4` |
| 0.9% | `quantize_q8_1` |
| 0.6% | `gdn_chunk_state_f32` |

Same shape as c=1: one weight-bound class owns the tick. The b8 kernel reads each weight once for
up to 8 columns, so the batched tick is *more* bandwidth-efficient per token, not less.

### 1e. The c=8 serial-fraction question — RESOLVED AS A BENCH ARTIFACT

`MEMRA_BATCH_PHASE=1` put **logits D2H + host split at 21.2% of the c=8 tick vs 0.5% at c=1**
(host argmax 183 µs/row × B=8 = 1.46 ms/tick) — above the task's >15% trigger. But that is
`decode-batch-bench`, which does a **host** `argmax` over `n_vocab=248320` per row.

The real serve path does not: `decode_step_batch_sampled_lean_masked` device-samples and skips the
per-row logits D2H entirely. Measured end-to-end:

| harness | c=8 aggregate |
|---|---|
| `decode-batch-bench` (host argmax) | 213.6 tok/s |
| `memra-server` + `tools/load-serve.py` (device sample + lean logits) | **289.3 tok/s** |

**+35.4%.** The 21.2% serial fraction is already eliminated in the shipping serve path. **No batch
lever list is triggered** — and the standing conclusion is that `decode-batch-bench` overstates
batched cost and should not be used to size serving levers (see §4).

---

## 2. Levers

### LEVER 1 — LANDED: Q8_0 dense-FFN gate+up launch fusion at m=1 (**+0.94%**)

`matmul_pre_dual_noscale` hard-rejected non-NVFP4, so the Q8_0 dense-FFN gate+up pair fell to two
separate `matmul_pre_noscale` launches — **128 of 1015 launches/token, the largest un-fused class**
in a tick that is 7.5% launch gaps. `q8_fused2_core` already served the identical pair shape for
the shared-expert gate/up, and its body is `qmatvec_q8_0_mmvq` verbatim per (tensor,row).

Engagement receipt (nsys, per 16 tokens): `fused2` 1536 → 2560 (**+64/token**), `mmvq` 4112 → 2064
(**−128/token**) — exactly the 2-for-1 trade, one fused launch replacing two singles per layer over
64 layers. Whole-tick effect, re-derived from the committed traces: **129,920 → 121,728 kernels over
128 tokens = 1015 → 951 kernels/token, −6.3%**.

A/B, N=5, arms interleaved **within** each rep, order alternated **across** reps:

| rep | OFF | ON | Δ |
|---|---|---|---|
| r1 (off first) | 49.83 | 50.30 | +0.94% |
| r2 (on first) | 49.81 | 50.29 | +0.96% |
| r3 (off first) | 49.81 | 50.28 | +0.94% |
| r4 (on first) | 49.78 | 50.27 | +0.98% |
| r5 (off first) | 49.78 | 50.26 | +0.96% |
| **median** | **49.81** | **50.28** | **+0.94%** |

5/5 pairs win, both orderings, spread 0.04. Exactness: token streams **IDENTICAL** over 128
tokens; `kernel-check` `Q8-FUSED2` cells `bits=true`. Seam `MEMRA_Q8_FFN_FUSE2=0`.
Guarded against the `rp4` split-plane mirror (`MEMRA_Q8RP`) — `fused2` has no `_rp` twin, so the
arm bails when a mirror exists rather than swapping dispatch families mid-model.

### LEVER 2 — LANDED: `MEMRA_GEN_GRAPH` budget key **256 → 48** (**+3.8% at 128, +7.7% cross-model**)

The CUDA-graph decode door defaulted ON at `budget >= 256`. That key came from the E4B
amortization **rule**, never from a measured crossover — so every ≤128-token generation, **including
the entire published board** (`--max-tokens 128`), was silently running **eager**.

The key is a *cross-model* shipped default, so one artifact is not enough evidence to move it. Swept
the real crossover on **two** models, N=3 per cell per arm, arms interleaved with order alternated
per rep:

Medians of N=3, re-derived from the raw logs (not transcribed):

| budget | q8 eager | q8 graph | q8 Δ | nvfp4 eager | nvfp4 graph | nvfp4 Δ |
|---|---|---|---|---|---|---|
| 16 | 50.18 | 46.43 | **−7.47%** | 78.47 | 66.49 | **−15.27%** |
| 32 | 50.25 | 49.57 | **−1.35%** | 78.74 | 78.91 | +0.22% (flat, see note) |
| **48** | 50.25 | 50.70 | **+0.90%** | 78.78 | 81.50 | **+3.45%** |
| 64 | 50.31 | 51.28 | **+1.93%** | 78.84 | 82.85 | **+5.09%** |
| 128 | 50.26 | 52.17 | **+3.80%** | 78.80 | 84.88 | **+7.72%** |
| 512 | 50.14 | 52.90 | **+5.50%** | — | — | — |

Both models: **clearly negative at 16, no reliable gain at 32, positive from 48 up**, monotone in
budget from 48 on. Capture cost amortizes in **~32 steps, not ~256** — the shipped key was ~5x too conservative.
**48 is the key**, not 64: 48 measures positive on both models, so keying at 64 would leave a
measured-positive cell on the table.

Honest note on the 32 cell: q8 is a clean −1.35% (per-arm spread ≤0.03), but nvfp4's graph arm at 32
is **noisy, not flat** — runs 79.02 / 78.91 / 77.09, spread 1.93 against an eager spread of 0.04.
The median lands +0.22%. Treating 32 as "no reliable gain" is what the data supports; it is not
evidence of a win, and it is the reason the key sits at 48 rather than 32.

Landed in `decode.rs` (`_ => budget >= 48`). Post-change **default-path** verification — the naked
command now lands on the graph arm:

| model | n=48 default | n=128 default |
|---|---|---|
| Q8_0 | **50.76** (was 50.25 eager) | **52.22** (was 50.26 eager) |
| NVFP4-MTP | **81.56** (was 78.78 eager) | **84.87** (was 78.80 eager) |

The NVFP4 n=48 figure is the r2 value from the drift-controlled re-probe, *not* the first
post-build run — see the watch-out below.

Exactness at the new key: `graph-decode-gate` 256 steps **BIT-IDENTICAL**, `graph-session-gate`
96 tokens PASS, `kernel-check` ALL GREEN, `run-spec` K=1..8 PASS, argmax MATCH on all 30+ sweep runs
across both models. The door still auto-closes when MoE experts are on the SLRU cache path
(capture-illegal) — unchanged.

> **Watch-out worth its own line:** the **first run after a rebuild is a cold-start outlier.**
> The initial default-path check read NVFP4 n=48 = 75.94 — *below both* sweep arms (eager 78.78,
> graph 81.50), which is impossible if the key merely selects one of them. Caught it only because it
> contradicted the sweep, then re-probed interleaved default/forced-off/forced-on x3:
>
> | rep | default | forced eager | forced graph |
> |---|---|---|---|
> | r1 | **75.95** (cold) | 78.79 | 79.34 (also warming) |
> | r2 | 81.56 | 78.75 | 81.50 |
> | r3 | 80.98 | 78.79 | 81.52 |
>
> From r2 on, `default` tracks the **graph** arm (81.56/80.98 vs 81.50/81.52) and beats forced-eager
> — the key is live and correct. Standing rule: **never take a post-build first run as a data
> point.** Receipts: `logs/key48-nv-n48-{default,0,1}-r{1,2,3}.log`.

### LEVER 3 — REFUTED: batched dense-FFN fusion at the serving tier (flat/negative)

Built the missing `qmatvec_q8_0_mmvq_fused2_b8` wrapper, widened `matmul_q8_fused2_t` from
`2..=4` to `2..=8`, and fused the `decode_step_batch` dense-FFN gate+up pair.

| harness | OFF | ON | verdict |
|---|---|---|---|
| bench c=8, 3 passes both orderings | 213.1 / 213.9 / 214.4 | 213.8 / 214.4 / 213.5 | sign-flipping |
| serve c=8, 3 passes both orderings | 291.0 / 290.1 / 289.0 | 292.2 / 289.3 / 286.9 | paired mean **−0.20%** |

Mechanism for the null: unlike m=1 — where the pair is 128 of 1015 launches inside a 7.5%-gap tick
— the c=8 tick is **73.2% one weight-bound kernel class with launch cost already hidden**. Halving
128 launches out of ~28k buys nothing.

Call site **reverted** per the flags doctrine (no dead dispatch arm). The **kernel and wrapper are
retained**: `matmul_q8_fused2_t` serves the verify tier, and `kernel-check` now gates the b8 tier
at `m=5` and `m=8` (`bits=true`) — that gate is the durable part of this refutation.

### LEVER 4 — REFUTED: `MEMRA_PRIME_CHUNK` sweep

At pp512 the sweep is a **null test** — every chunk ≥1024 is monolithic on a 512-token prompt
(measured flat 4130-4174, spread 0.9%). Re-run on the 6257-token prompt, the ascending sweep looked
like a monotone decline (r1 default 3976 → r3 chunk4096 3839, every step down) — but `default` was
always measured first, and total drift over 12 runs was **−4.2% monotone in run order**: a thermal
signature, not a chunk effect.

Drift-controlled A/B (chunk 2048 vs default, both `ab` and `ba` orderings per pass, N=3):

| pass | ab (A first) | ba (B first) | order-paired mean |
|---|---|---|---|
| p1 | −0.65% | +0.52% | −0.07% |
| p2 | −0.49% | +0.41% | −0.04% |
| p3 | −0.21% | +0.20% | −0.01% |

**Order-paired mean −0.04%** — indistinguishable from zero, with the apparent effect fully
explained by measurement order. `MEMRA_PRIME_CHUNK` is not a lever on a 96GB card at these depths.

---

## 3. Same-board denominators (for the phase-2 A/Bs)

Plain `run-gen`, N=3, gate-green, thermal regime noted per run in the logs:

| cell | this board | prod 20260804 | delta |
|---|---|---|---|
| q8 d512 (128 tok, eager — the *old* default) | 49.82 | 52.61 | −5.3% |
| q8 d512 (128 tok, **new default**: lever 1 + key=48) | **52.22** | — | +4.8% vs this board's eager |
| q8 pp512 | 4091.9 | 4591.5 | −10.9% |
| q8 pp6257 | 4030.6 | — | — |

Phase-1 net on the naked command, same board / same commit / same prompt: **49.82 → 52.22 tok/s
(+4.82%)** at the 128-token board shape, entirely from lever 1 (+0.94%) and lever 2 (+3.80%), with
bit-identity or BIT-IDENTICAL gates on both. The NVFP4-MTP artifact moves **78.80 → 84.87 (+7.70%)**
on lever 2 alone.

The two levers compose but are **not exactly multiplicative**: 49.82 × 1.0094 × 1.0380 = 52.19 vs
**52.22 measured**, a 0.06% gap that is inside this board's run-to-run spread. The +4.82% headline is
the *measured* end-to-end default-vs-default number, not the product of the two lever deltas — they
were measured against different baselines (lever 1 eager-vs-eager, lever 2 on the fused tree), so
stacking them is a sanity check, not the claim.

`memra-server` + `tools/load-serve.py` (temp 0.7, `max_tokens=128`, N=3 passes, server restarted
per arm), 0 errors / 0 shed throughout:

| cell | aggregate tok/s | p50 latency |
|---|---|---|
| serve c=1 | 46.09 | 2.777 s |
| serve c=8 | 289.3 | 3.53 s |

`decode-batch-bench` batch scaling (median of 3, two runs agreeing to 0.4%):

| B | aggregate | per-seq | scale |
|---|---|---|---|
| 1 | 45.4 | 45.4 | 1.00x |
| 2 | 86.4 | 43.2 | 1.90x |
| 4 | 156.5 | 39.1 | 3.45x |
| 8 | 213.6 | 26.7 | 4.70x |

---

## 4. THE headline finding: serve c=1 leaves 11.7% on the floor

| path | c=1 tok/s | vs serve |
|---|---|---|
| `run-gen`, **today's naked default** (lever 1 + key=48 → graph arm at 128 tok) | **52.22** | serve is **−11.74%** |
| `run-gen` (lever 1 + graph door forced) | 52.17 | serve is −11.65% |
| `run-gen` (lever 1, door *closed* — the pre-lever-2 default) | 50.28 | serve is −8.33% |
| **`memra-server` c=1** | **46.09** | — |

Note the framing shift lever 2 forces: against the *old* eager default the serve gap was −8.3%, but
the naked command now lands on the graph arm at the 128-token shape, so **the gap a user actually
sees today is −11.74%**. Both levers widened it, because neither reaches the serve path. Cause: the serve worker routes **B=1 through `decode_step_batch`**, which
(a) has no CUDA-graph door at all (the worker runs its own tick loop; `MEMRA_GEN_GRAPH` lives in
`generate_with`, which the worker does not call), and (b) dispatches the dense-FFN pair through
`matmul_pre` at `b_n=1`, so **lever 1 never fires on the serve path**.

Confirmed directly rather than inferred — the serve A/B for `MEMRA_Q8_FFN_FUSE2` at c=1 is
order-paired **+0.06%** (pairs +0.20% / −0.25% / +0.23%, i.e. sign-flipping noise), against +0.94%
with 5/5 winning pairs in `run-gen`. That contrast *is* the receipt that the serve path bypasses the
m=1 dispatch family.

For a single-tenant interactive request — the dominant darklane shape — the serving path is
**11.7% below what the same box already does with the naked `run-gen` default**. That is the largest
single number in this whole deep dive — bigger than everything phase 1 landed combined (+4.82%) — and
it is pure recoverable overhead, not silicon. H1/H3 are the entry points.

---

## 5. Phase-2 (MTP) entry points the profile suggests

1. **The verify tier is already fused, the trunk is now too.** With lever 1 landed, `fused2` owns
   **56.1%** of the m=1 tick and `fused3` **4.3%** (fuse-ON arm,
   `nsys/nsys-q8-decode-c1-eager_cuda_gpu_kern_sum.csv`; the fuse-OFF baseline in §1a has
   `fused2` at 15.2%). Lever 1 is visible in the launch counts, not just the wall clock: plain
   `qmatvec_q8_0_mmvq` drops 257→129 launches/token (−128 singles) while `fused2` rises 96→160
   (+64 fused), i.e. exactly the 128 gate+up singles replaced by 64 fused launches, for the
   1015→951 total. MTP verify at t=2..8 rides `fused2_b*`/`fused3_b*` —
   the same kernel bodies, now with the b8 wrapper built and **kernel-check-gated at m=5/8**. Phase
   2 inherits a gated batched-fusion tier instead of having to build one.
2. **Launch gaps are the only c=1 slack, and graph replay is the tool that closes them.** 7.5% gaps
   at 951 launches/token; the graph door converts that to +3.8..5.5%. A spec round is a *longer*
   launch chain than a decode step, so the round-graph door (`MEMRA_GEMMA_ROUND_GRAPH`, refuted
   −10/−11% on gemma at fixed shapes) deserves a q27 re-measure **now that the budget-key finding
   shows this board's crossover is 4x lower than assumed** — stale-verdict law.
3. **Do not size MTP levers on `decode-batch-bench`.** It overstates batched cost by 35% (host
   argmax). Every phase-2 batched claim must run through `memra-server` + `load-serve.py`.
4. **Attention is not the lever, at any concurrency.** 0.7% of the c=1 tick, 1.3% at c=8. Phase 2
   should not spend time on the 27B attention kernels.
5. **Prefill is one GEMM class (75.1%) with no launch story.** MTP changes decode, not prefill;
   the pp512 denominator (4091.9) should be treated as a fixed constant across phase 2.

## 6. Named next-hypothesis list

| # | hypothesis | why it is named | how to test |
|---|---|---|---|
| H1 | **The serve worker should route B=1 to the `generate_with`/graph-door path (or grow its own graph door).** Recovers up to **11.7%** on single-tenant interactive — the single largest number in this deep dive, larger than everything phase 1 landed combined. | §4: serve c=1 **46.09 vs 52.22** on the naked default, same board / commit / prompt | worker-side arm + serve c=1 A/B, N=5, both orderings; stream-identity gate vs the batched path |
| H2 | ~~budget key 256 → 64~~ **LANDED at 48** (§2 lever 2). Residual: sweep the *other* board models (gemma/KAT/q35) — 48 is validated on two q27 artifacts, not the whole board. | key is a cross-model default; two models is enough to move it, not enough to stop checking | same interleaved sweep per model; watch for any model whose crossover sits above 48 |
| H3 | **Serve B=1 should dispatch the m=1 kernel family, not the batched one.** Lever 1 is dead on the serve path today. | §4: `MEMRA_Q8_FFN_FUSE2` measures order-paired +0.06% (sign-flipping) at serve c=1 vs +0.94% with 5/5 winning pairs in run-gen | `b_n==1` fast-path in `decode_step_batch` → the m=1 dispatch; bit-identity gate + serve c=1 A/B |
| H4 | **The small-shape class (attn_out/ssm_out, 1509.4 GB/s vs 1568-1644 in the big streaming classes) is the residual bandwidth hole.** ~60-135 GB/s of width left on the table in ~5% of the tick — note the re-derivation shrank this gap a lot (the pre-correction draft read 1258 GB/s), so H4 is now a *small* hypothesis, ranked below H7. | §1b derived-BW table (`scripts/derive-bw.py`) | needs ncu occupancy → **blocked on this pod**; re-run on prod-class silicon with `cap_perfmon` |
| H5 | **`gdn_chunk_state_f32`+`gdn_chunk_output` (6.9% of prefill) are the only non-GEMM prefill target.** | §1c | isolated GDN prefill bench sweep; compare against the MMQ class's achieved TF |
| H6 | **The round-graph door may now be positive on q27** (refuted on gemma at fixed shapes pre-dating this board's crossover finding). | §5 item 2 + stale-verdict law | phase-2 spec-round A/B once the drafter is attached |
| H7 | **Fold `ssm_alpha`+`ssm_beta` into the neighbouring GDN projection.** 48 launches/token at 183.4 GB/s = 136.7 µs/token (**0.68%** of the 20.07 ms tick) for 0.52 MB of weights — pure launch/latency cost on a 96-row output, the same class of win as lever 1. | §1b: 8.5x below the streaming classes, surfaced by `scripts/derive-bw.py` | extend the existing fused-projection path; bit-identity gate (the body would be lifted verbatim like lever 1), then interleaved N=5 A/B |

---

## 7. Receipts

All raw per-run logs, nsys reports (`.nsys-rep` + `.sqlite` + kernel-sum/trace CSVs), and the
serve JSONL live under `nsys/`, `logs/`, `ncu/` next to this file. Driver scripts: `scripts/`.

- Baselines: `logs/anchor-q8-{d512,pp512}-r{1,2,3}.log`
- Gates (lever-1+3 tree): `logs/gate-rungen-argmax-q8.log`, `logs/gate-kernel-check-lever3.log`,
  `logs/gate-decode-batch-fuse-{off,on}.log`, `logs/fusebits-{off,on}.{log,txt}`
- Gates (reverted tree): `logs/gate-r3-kernel-check.log`, `logs/gate-r3-decode-batch-fuse{0,1}.log`,
  `logs/gate-r3-{graph-decode-gate,graph-session-gate}.log`,
  `logs/gate-r3-runspec-mtp-K1to8.log`, `logs/gate-r3-runspec-q8.log` (the `nextn=0` rc=2 receipt)
- Gates (final key=48 tree): `logs/gate-key48-{kernel-check,decode-batch,graph-decode-gate,graph-session-gate}.log`,
  `logs/gate-key48-runspec-K1to8.log`, `logs/gate-key48-default-{Q8_0,NVFP4-Q4_K_M-mtp}-n{48,128}.log`
- Lever 1: `nsys/engage-{off,on}*` (engagement receipt), `logs/ffnfuse-{off,on}-r{1..5}.log`
- Lever 2 (q8): `logs/gengraph-n{16,32,48,64,128,512}-{eager,graph}-r{1,2,3}.log`
- Lever 2 (cross-model): `logs/gengraph-m2-Qwen3.6-27B-NVFP4-Q4_K_M-mtp-n{16,32,48,64,128}-{eager,graph}-r{1,2,3}.log`,
  cold-start probe `logs/key48-nv-n48-{default,0,1}-r{1,2,3}.log`; driver `scripts/gengraph-m2.sh`
- Builds: `logs/build-revert3.log`, `logs/build-key48.log`
- Lever 3: `logs/lever3-bench-{off,on}-p{1,2,3}.log`, `logs/serve-points.jsonl`
- Lever 4: `logs/lever-chunk-*`, `logs/lever-chunklong-*`, `logs/chunkab-2048-{ab,ba}-{A,B}-r*.log`
- Profiles: `nsys/nsys-q8-decode-c1*`, `nsys/nsys-q8-pp512-default*`, `nsys/nsys-q8-decode-c8*`
- Batch phase: `logs/phase-q8-c{1,8}.log`; batch scaling: `logs/batchscale-q8-r{1,2}.log`
- ncu failures (the `ERR_NVGPUCTRPERM` evidence): `ncu/ncu-q8-{decode-mmvq,prefill-mmq}.log`

A note on the graph-door timeline: `nsys` traces a replayed graph as a **single launch node**, so the
gap analysis in §1a is **eager-arm only** — 7.48% gaps / 92.52% busy at 1015 kernels/token (fuse OFF)
and 7.39% / 92.61% at 951 kernels/token (fuse ON). The graph arm's own trace shows only 2,584 GPU rows
total (1,902 kernels + 682 memset/memcpy — vs 121,728 kernels in the eager arm) and a **98.44%** "gap",
which is a **tracing artifact** of graph replay, not a real idle measurement. The graph
door's benefit is established by the wall-clock A/B in §2, never by that trace.

Note that the fuse-ON arm's gap *fraction* (7.39%) is barely below fuse-OFF's (7.48%) even though it
issues 64 fewer kernels per token — removing a launch removes both its gap and its work, so the ratio
moves far less than the wall clock does. That is why lever 1's claim rests on the interleaved
tok/s A/B and the launch-count receipt, not on a gap-percentage delta.
