# RP-mirror-on-ST — verdict: **the mirror cannot be ported, must not be, and was never the gap**

Lane: `lane/rp-on-st` · 2026-08-06 · local RTX 5090 (24 GB, sm_120a) + RTX PRO 6000 Blackwell
Workstation 96 GB (community pod, deployment verdict rig)

Charter: port the Q8RP batch lever to the FP8-ST path so FP8-ST becomes the single prod artifact
for Qwen 3.8 day one, because "the +57% c=16/32 batch win is the ONLY thing keeping Q8_0 GGUF in
the prod picture."

Every raw run is in this directory. `RESULTS.jsonl` carries one row per measured fact. Cite
`git rev-parse HEAD` for hashes — this branch was rewritten in place during the 2026-08-06
credential scrub (old `b6f2d6d7` → `3a1d9b22`; content byte-identical, map at
`/home/avifenesh/projects/bw24-scrub-commit-map-20260806`).

---

## 0. The answer, up front

**The mission bar is not met, and the charter's premise inverted three times on the way to
finding out why.** Each inversion was measured, not argued, and each one made the next question
different from the one the charter asked.

1. **The mirror is a NO-OP on FP8-ST.** It cannot be ported because there is nothing to port to.
2. **The c=16 blocker was exact-16 ADMISSION, not bandwidth** — and it was blocking *every*
   shipped checkpoint, GGUF included, not just FP8-ST. Fixed here, as kernels, at zero VRAM cost.
3. **Admission was necessary but not sufficient.** With chunk 16 admitted, FP8-ST moved **+0.11%**.
   The ST batched path is not weight-bandwidth-bound at that width; it was **ALU-bound in the
   e4m3 dequant**, which the batched tier was running MCOLS times per weight fetch. Hoisting it:
   **+13.5% at c=16** (N=5 interleaved).

Net: FP8-ST at c=16 goes 307.0 → 348.3 tok/s on the 96 GB rig (+13.5%, N=5 interleaved), which closes a third of
the gap to GGUF-Q8RP's 454.1 but not the gap. **FP8-ST is not yet the single prod artifact.** The
honest day-one statement is in §7.

Three things landed after that answer, each with its own section:

4. **The ALU wall confirmed a second way, with no new kernel** (§7b). The existing `MEMRA_ST_E4M3=0`
   seam sends the same ST bytes down the `dp4a` Q8_0 slab loop: **+15.6% at c=16, +17.1% at c=32,
   −2.4% at c=1**, +448 MiB (N=5, all cells disjoint). That makes the best measured FP8-ST batch
   config **401.3 tok/s = 0.88x** of GGUF-Q8RP at **2.5x its throughput per resident GiB**. Not
   flipped as a default — it is a per-m regression at the single-stream width the owner runs daily.
5. **A coordinator-routed find, absorbed and shipped** (§7c): the per-block FP8 prefill tile was
   issuing the slow `kind::f8f6f4` MMA form. Swapping to the block_scale form at the ue8m0 identity
   scale is **+6.54% e2e pp512, bit-identical** (0 of 993280 logit bytes differ; `fp8-mmq-check`
   reports byte-for-byte equal across forms).
6. **The same defect's third site invalidates a published GO** (§7c-bis): the FP8-ST v3 gate's Q1
   instrument had the *same* form asymmetry against its s32 control, so its "+16.7/+23.1pp weighted
   → GO" was the MMA form, not the accumulator. Equalized, the sign flips to **−6.7/−8.8pp**. The
   receipted case for an FP8-ST v3 is refuted on its own instrument.

---

## 1. Inversion 1 — the mirror is a no-op on ST (measured, not reasoned)

`MEMRA_Q8RP=1` on the FP8-ST 27B produces a residency census **byte-for-byte identical** to naked
(`census-st-naked.log`, `census-st-q8rp1.log`):

| class | tensors | MiB | share |
|---|---|---|---|
| F8_E4M3 | 208 | 6880.000 | 41.0% |
| NVFP4 | 193 | 9862.031 | 58.8% |
| Q8_0 | 96 | 23.906 | **0.143%** |
| TOTAL | 497 | 16765.938 | |

Both arms argmax MATCH. The mirror walk **does** fire (`[q8rp] split-plane decode mirrors built:
96 tensors`) — `build_q8_rp4` accepts only `QT_Q8_0`, and on an ST checkpoint the only Q8_0 left is
23.9 MiB. The 401 tensors that hold 99.86% of the bytes are refused by dtype.

**And it shouldn't mirror them.** The mirror exists to fix a *coalescing* deficit specific to GGUF
Q8_0's 34 B block stride (H100 ncu 2026-07-26: Max BW pinned 41-46% from sector overfetch). Native
e4m3 has `row_bytes == in_f` at 1 B/weight, so every 32-weight block is already a 32 B-aligned pair
of LDG.128s. There is no skew to un-skew. A second residency copy would spend ~6.9 GB buying an
alignment the layout already has.

---

## 2. Inversion 2 — the real blocker was exact-16 admission, and it blocked EVERYTHING

### What the b8-class arms did on ST before this branch

Measured before touching anything (`gate-decode-batch-both.log`): both ST classes were **ALL GREEN
at config B=8 and strict B=4**, and both **rc=101 REFUSED at B=16**. That is the gap, and it is a
*kernel-class* gap, not a data-movement one:

* per-tensor `QT_F8_E4M3` had `_b2/_b4/_b8` only — m=9..16 fell through the `m<=8` dispatch gate
* block-128 `QT_F8_E4M3_BLK` likewise
* so the serve scheduler's `chunk_cap_for` returned 8, and c=16/c=32 traffic was chunked 8-wide

### The ALL-semantics discovery

`decode_batch_exact16_ok` is an **ALL over every matmul in the model** (~500 on the 27B). It admits
a checkpoint iff EVERY matmul has a per-(token,row) bit-exact b16-class kernel at m=9..16. **One
class without a b16 refuses the whole model, silently, back to chunk 8.**

Every real shipped checkpoint is MIXED. A Qwen3.5 9B "NVFP4" GGUF spreads its matmuls over NVFP4
(MLP) + Q4_K (qkv) + Q5_K (qkv_gate) + Q6_K (output) + Q8_0 (ssm). So chunk 16 was unreachable for
**every artifact memra ships**, and had been since the tier landed.

Four classes were blocking. Each was found by the new diagnostic, not by guessing — the previous
three iterations had each guessed and been wrong:

| class | how it surfaced | note |
|---|---|---|
| NVFP4 | `[exact16] REFUSED by L0.ffn_gate qtype=7` | had b2/b4/b8, no b16 — and NVFP4 is memra's *primary* quant |
| Q8_0 | `[exact16] REFUSED by L0.ssm_beta qtype=0 rp4=false` | b16 existed ONLY as the `_rp` twin |
| Q4_K | `[exact16] REFUSED by L0.wqkv qtype=1` | |
| Q5_K | `[exact16] REFUSED by L0.wqkv_gate qtype=3` | no rp twins at any width |

The Q8_0 entry is the one that matters for the charter: **the q8rp mirror was a *correctness
prerequisite* for the exact-16 tier, not the bandwidth lever it was designed as.** A checkpoint had
to pay a full second copy of its trunk to reach chunk 16 — for a kernel-shape reason.

### The instrument that made it findable

`MEMRA_EXACT16_WHY=1` (default off, diagnostics-only per the flags doctrine, documented in
`docs/FLAGS.md §4`). It prints which tensor and qtype refuses, plus the structural refusals
(architecture / MLA mixer / MoE ffn) that can never be admitted. Without it, each "still refused"
round was a guess; with it, four rounds closed four classes.

**Also fixed, one instrument earlier:** `decode_batch_gate` was GGUF-only, and `Mmap::map` on a
*directory* fd returns ENODEV — so pointing the gate at an ST checkpoint died with
`Os { code: 19, ... "No such device" }` **before loading anything**, which reads exactly like a
GPU-acquisition failure and is not one. The inherited "GPU contention" hypothesis was wrong. The
gate now takes the same directory branch as `run_gen`.

### Cost: zero VRAM

The fix is **kernels, not a residency copy**. Every new kernel is a pure MCOLS=16 instantiation of
an already-proven batched template, so bit-identity is structural rather than argued:

```
qmatvec_nvfp4_mmvq_b16     -> nvfp4_mmvq_batched<16>
qmatvec_nvfp4_mmvq_b16_rp  -> nvfp4_mmvq_batched_rp<16, 1>
qmatvec_q8_0_mmvq_b16      -> q8_0_mmvq_batched<16>      (base form; was rp-only)
qmatvec_q4_K_mmvq_b16      -> q4k_mmvq_batched<16>
qmatvec_q4_K_mmvq_b16_rp   -> q4k_mmvq_batched_rp<16>
qmatvec_q5_K_mmvq_b16      -> q5k_mmvq_batched<16>        (base only)
qmatvec_e4m3_mmvq_b16      -> e4m3_mmvq_batched<16>
qmatvec_e4m3_blk_mmvq_b16  -> e4m3_blk_mmvq_batched<16>
```

The `_rp` twins are mandatory, not optional: `qmatvec_mmvq_batched` forces
`variant = if rp { "rp" } else { "base" }` at `mcols == 16`, because **rp is a LAYOUT** (split-plane
qplane ++ dplane), not a perf variant. Feeding split-plane bytes to a GGUF-layout kernel is NaN.
NVFP4-from-safetensors is split-plane by default, so its `_rp` twin is load-bearing.

VRAM census, both classes, before and after: **16765.938 MiB, unchanged**. Local serving census
14818 MiB in both arms, every round.

---

## 3. Inversion 3 — admission was necessary but NOT sufficient

Verdict cells, PRO 6000 96 GB, **N=5 with all four arms interleaved within each round**, server
restarted per arm, spec OFF everywhere (so all arms exercise the same batched-decode path), sampled
load (t=0.7 + per-request seed) so batched decode sees divergent sequences, max_tokens=128,
requests=3*c. Idle box: 0 other compute-apps at every hold. 0 errors, 0 shed across all 40 points.
Spread ≤0.92%.

| arm | c=16 tok/s | c=32 tok/s | p50 lat c=16 | VRAM resident |
|---|---|---|---|---|
| `st_b16` FP8-ST, naked → chunk 16 | **307.90** | 306.45 | 6.64s | 17812 MiB |
| `st_b8` FP8-ST, cap=8 (BEFORE arm) | 307.56 | 306.72 | 6.66s | 17812 MiB |
| `q8_rp` GGUF Q8_0 + mirror, chunk 16 | **454.07** | 449.40 | 4.51s | 52532 MiB |
| `q8_norp` GGUF Q8_0, no mirror, chunk 16 | 290.83 | 290.25 | 7.04s | 26580 MiB |

| ratio | c=16 | c=32 |
|---|---|---|
| `st_b16 / st_b8` — what admission bought ST | **1.0011x** | 0.9991x |
| `q8_rp / q8_norp` — the mirror's own lever, reproduced | 1.5613x | 1.5483x |
| `st_b16 / q8_rp` — the mission bar | 0.6781x | 0.6819x |
| `st_b16 / q8_norp` | 1.0587x | 1.0558x |

Read the first two rows together. Widening the chunk to 16 moved FP8-ST by **0.11%** — nothing,
against a 0.4% spread — while the *same* widening is worth **+56%** on GGUF Q8_0 with its mirror.

A path that gains nothing from reading each weight once for 16 columns instead of 8 **is not
weight-bandwidth-bound at that width.** The batched tier's whole value proposition did not apply.

Note the `q8_rp / q8_norp` = 1.56x cell also re-prices the mirror on its own terms: it costs
**25952 MiB** — a second copy of the trunk, +98% resident — to buy +56% throughput. In tok/s per
resident GiB: `q8_rp` 8.85, `st_b16` 17.70 (pre-hoist). Affordable on 96 GB; not on the 24 GB
deployment target.

---

## 4. The actual wall: e4m3 dequant was running MCOLS times per weight fetch

Found by reading the kernel after the flat cell, not before.

The batched decode tier exists to amortize **per-weight** work across activation columns: read the
weight bytes once per (row, k32-block), reuse for all m columns. Both e4m3 batched row bodies read
the bytes once and then converted them **inside** the column loop:

```
for blk:                                  // weight bytes read ONCE
    wu[8] = load 32 e4m3 bytes
    for c in 0..MCOLS:                    // ... and CONVERTED MCOLS times
        e4m3x2_to_f32x2(wu[k]) x8
        8x fmaf against column c's int8 activations
```

At MCOLS=16 that is **sixteen** e4m3→f32 conversions of one weight fetch. "Weight read once,
weight converted sixteen times."

This is e4m3-specific, and the contrast proves it: `nvfp4_mmvq_batched` **already hoists** its
nibble decode + scale out of the column loop (`int2 wv[2][2]; float wscale[2];` computed once per
group, comment: *"decode the weight nibbles ONCE for this group (reused across all m token
columns)"*). dp4a classes get their dequant free inside the integer dot product. e4m3 pays ~32 fmaf
per k32 block **per column**. The ST 27B is 41% e4m3 by resident bytes, so this hit on every column
of every step.

Fix: hoist into a `float wf[32]` per k32 block, computed once before the column loop, in both
`e4m3_mmvq_batched_row` (per-tensor; also feeds `fused2_b`/`fused3_b`, which share the body
verbatim and inherit it) and `e4m3_blk_mmvq_batched_row` (block-128).

**Exactness is by construction**: identical values, identical `bs` accumulation order over k,
identical scale fold and `warp_reduce_sum`. Only the point of evaluation moves, and float
conversion is not order-dependent. Measured anyway (`hoist-gate.log`):

* `kernel-check` **ALL GREEN** — every `E4M3-BATCHED` and `E4M3-BLK-BATCHED` cell `bit-bad=0`,
  EXACT and RAND arms, m={2,5,9} × b2/b8/b16, all shapes including ragged 1184x200 and 5120x1536;
  the 254/254 legal-e4m3-code coverage cell still passes.
* `run-gen --verify-prefill`, **both** ST classes: logit maxdiff **exactly 0.000e0**, argmax MATCH,
  identical token streams — `nvidia-qwen36-27b-nvfp4` (per-tensor) and `qwen36-27b-blk128fp8`
  (block-128).

### The hoist A/B (interleaved, same box, same branch, only the hoist differs)

`hoist` = this build; `nohoist` = the preserved pre-hoist binary
(`cf018d2575d856927b24ba4def6d6719`), same branch otherwise — same b16 twins, same admission.
**N=5, the two arms interleaved within each round**, server restarted per arm, idle box, spec off,
sampled load, max_tokens=128, requests=3*c. 0 err / 0 shed across all 30 points
(`pod-hoist.log`, medians in `RESULTS.jsonl` as `hoist_ab_*`):

| c | hoist (median, N=5) | nohoist (median, N=5) | gain | spread hoist / nohoist |
|---|---|---|---|---|
| 1 | 71.82 | 71.66 | +0.21% (flat — m=1 path untouched, as designed) | 0.04% / 0.47% |
| 16 | **348.28** | 306.97 | **+13.46%** | 0.51% / 0.21% |
| 32 | **344.28** | 305.15 | **+12.82%** | 0.35% / 0.51% |

The gain is 26-64x the worst spread in the cell, and the two distributions are disjoint at both
widths (hoist min 347.19 vs nohoist max 307.04 at c=16).

Cross-session anchor: `nohoist` c=16 306.97 vs the verdict run's `st_b16` 307.90 — 0.3% apart on a
different session, which is what licenses comparing the hoisted number against that run's `q8_rp`.

The c=1 flat cell is the mechanism's own signature: at m=1 there is no column loop to hoist out of,
so a fix that only removes redundant per-column work must show up **only** at width. It does.

---

## 5. Exactness — the full battery, both rigs

Local RTX 5090 (flock-serialized, `battery-local.log`, `serve-gates-local.log`, `hoist-gate.log`):

| gate | artifact | verdict |
|---|---|---|
| `kernel-check` | 9B NVFP4 GGUF | ALL GREEN, 0 FAIL |
| `decode-batch-gate` config B=16 | 9B NVFP4 GGUF | **PASS rc=0** (was rc=101 REFUSED) |
| `decode-batch-gate` config B=12 | 9B NVFP4 GGUF | PASS rc=0 |
| `decode-batch-gate` strict B=4 equalized | 9B NVFP4 GGUF | PASS rc=0 |
| `run-gen` argmax + verify-prefill | 9B NVFP4 GGUF | MATCH |
| `run-gen` argmax + verify-prefill | FP8-ST 27B (e4m3+nvfp4) | MATCH, maxdiff 0.000e0 |
| `run-gen` argmax + verify-prefill | BLK128 27B | MATCH, maxdiff 0.000e0 |
| `run-spec` K=1..8 self-consistency | 9B NVFP4 + draft | PASS 8/8 (acceptance 11.0-90.6%) |
| `serve-smoke` | 9B NVFP4 + draft | **0 failed** |
| `serve-st-gate` | qwen35-4b-hf BF16 ST dir | **0 failed** |

New bitwise cells, all `bit-bad=0` at m={9,12,16} × {base, rp}: `B16-TIER` Q8_0 (0/288, 0/384,
0/512), Q4_K (0/73728, 0/98304, 0/131072), Q5_K (0/36864, 0/49152, 0/65536); `NVFP4-B16` (0/110592,
0/147456, 0/196608); `KQRP` Q4_K+Q6_K mcols=16.

`serve-smoke` matters more here than it looks: Q8_0/Q4_K/Q5_K/NVFP4 admission changes behavior on
the **primary delivery format**, not just on the ST lane. It is the no-regression receipt for GGUF.

PRO 6000 96 GB (`pod-gates.log`, fresh build of this branch — the pod's earlier binaries predated
all four b16 classes and were discarded):

| gate | artifact | verdict |
|---|---|---|
| `kernel-check` | 27B NVFP4 GGUF | ALL GREEN, 0 FAIL (Q5_K b16 `output.weight` m=16 bit-bad=0/3973120) |
| config B=16 | FP8-ST 27B | **PASS rc=0** |
| strict B=4 | FP8-ST 27B | PASS rc=0 |
| config B=32 | FP8-ST 27B | REFUSED rc=101 — correct, see below |
| config B=16 | GGUF Q8_0 27B | **PASS rc=0** |
| strict B=4 | GGUF Q8_0 27B | PASS rc=0 |
| config B=32 | GGUF Q8_0 27B | REFUSED rc=101 |

**The Q8_0 B=16 pass is the direct proof of the decoupling**: q8rp defaults OFF on Blackwell, so
that checkpoint entered the exact-16 tier with **no split-plane mirror resident**. Before this
branch it could only reach chunk 16 by paying ~27 GB of mirror.

B=32 refusing is correct and is not a serving limit: there is no exact kernel class above 16 (m>16
crosses GEMM/dp4a numeric configs), and the serve scheduler chunks wider concurrency into ≤16
groups. The assert message that fired still claimed "Q8_0 m>8 needs the q8rp mirror's b16 class" —
stale twice over — and was rewritten to say the real reason and point at `MEMRA_EXACT16_WHY`.

Local no-regression, 24 GB card, daily 27B NVFP4 GGUF, N=3 interleaved (`local-noreg.jsonl`):

| c | chunk 16 (new) | chunk 8 (before) | VRAM |
|---|---|---|---|
| 1 | 45.46 | 45.44 | 14818 MiB both |
| 8 | 171.51 | 171.54 | 14818 MiB both |

Both cells are ≤8 wide, so the cap cannot change which kernel runs — that is the point. This is the
receipt that **admission itself is free** on the low-concurrency path, and that the fix adds no
resident bytes.

---

## 6. Default / door decision

Per the flags doctrine (winners are defaults; naked commands get full speed):

* **b16 twins for NVFP4 / Q8_0-base / Q4_K / Q5_K / e4m3 / e4m3-blk: DEFAULT, no flag.** They are
  new kernels behind an existing exactness predicate. Admission is decided by
  `decode_batch_exact16_ok`, which is a proof obligation, not a preference — nothing to tune.
* **e4m3 dequant hoist: DEFAULT, no flag, no rollback seam.** It is bit-identical (maxdiff 0.000e0)
  and strictly less work. A flag would be dead code and the JSONL row is the record.
* **`MEMRA_EXACT16_WHY`: new, diagnostics-only, default off.** The one genuinely new env var; the
  doctrine permits diagnostics.
* **`MEMRA_Q8RP` stays a machine-config door, unchanged defaults** (ON Hopper only). Its docs were
  corrected: it is now *purely* a bandwidth lever, no longer the exact-16 admission ticket, and it
  is a documented no-op on FP8-ST.
* **No new door for chunk width.** `MEMRA_DECODE_BATCH_CAP` already exists as the measurement door
  and rollback seam; the BEFORE arm in every cell above is just `=8`.

Docs updated in the same commits: `docs/FLAGS.md` §3 (`MEMRA_Q8RP`), §4 (`MEMRA_EXACT16_WHY`), §7
(`MEMRA_DECODE_BATCH_CAP`), and `worker.rs`'s `chunk_cap_for` doc comment, all of which carried the
now-false "Q8_0 needs the q8rp mirror" claim.

---

## 7. The day-one question: is FP8-ST the single artifact?

**Not yet — and the charter's own framing is what has to change, not the answer.**

The charter said the +57% c=16/32 batch win is the ONLY thing keeping Q8_0 GGUF in the prod
picture, and asked for that lever to be ported. It cannot be ported: it is a fix for a GGUF layout
defect that FP8-ST does not have. What this lane did instead was find and remove the two things
actually holding FP8-ST off the batch tier — admission (zero VRAM) and the e4m3 ALU wall (+13.5%).

Where that leaves the two candidates at c=16 on the 96 GB rig:

| | tok/s | resident | tok/s per GiB |
|---|---|---|---|
| GGUF Q8_0 + q8rp mirror | 454.1 | 52532 MiB | 8.85 |
| FP8-ST, `MEMRA_ST_E4M3=0` dp4a slab (§7b) | 401.3 | 18260 MiB | **22.50** |
| FP8-ST (this branch, default e4m3) | 348.3 | 17812 MiB | 20.02 |
| GGUF Q8_0, no mirror | 290.8 | 26580 MiB | 11.20 |

FP8-ST is at **0.77x** of GGUF-Q8RP throughput on its default path and **0.88x** on the slab seam
(§7b, not a default), while using **34-35%** of its resident bytes. So the
honest statement for the 3.8 drop is conditional on the box, and the two cases point opposite ways:

* **On the 24 GB deployment target** (the local 5090, memra's stated final performance target), the
  question does not arise: a 27 GB Q8_0 artifact plus a ~27 GB mirror does not fit, mirror or not.
  FP8-ST at 17.8 GB resident is the only one of the three that serves at all. **FP8-ST is the
  day-one artifact there, and now it reaches the c=16 tier.**
* **On a 96 GB box serving batch traffic**, GGUF-Q8RP is still 1.31x faster than FP8-ST's default
  path (1.13x against the §7b slab seam) and remains the pick if VRAM is free. It is 2.3-2.5x worse
  per resident GiB, so it stops being the pick as soon as VRAM is contended (multi-model, longer
  contexts, more sessions).

So: **FP8-ST is the single prod artifact for the 24 GB target and for any VRAM-contended box, and
is not yet the single artifact for a VRAM-rich batch box.** Q8_0 GGUF stays in the picture for
exactly one reason, and it is now a *measured* reason with a named mechanism rather than an
unexplained +57%.

## 7b. The ALU wall, confirmed a second way: the dp4a slab WINS at width on the same bytes

§8 item 1 predicted that if the ST batched path is ALU-bound, then sending the *same checkpoint*
down the Q8_0 slab path — whose batched inner loop is `dp4a`, 4 int8 MACs per instruction — should
win at width even though it is known to lose at m=1. **It does, and it needed no new kernel**: the
`MEMRA_ST_E4M3=0` seam already exists (ARM B′ from `research/fp8dec-20260805/`, which measured it
losing 2.58pp at m=1 — a regime with no column loop, so it never priced this wall).

96 GB pod, same ST-27B bytes on disk, server restarted per arm, arms interleaved within each round,
**N=5**, spec off, sampled, max_tokens=128, requests=3·c, **0 err / 0 shed across all 30 points**
(`pod-slab-points.jsonl`, medians as `slab_ab_*` in `RESULTS.jsonl`):

| c | e4m3 native (hoisted, default) | Q8_0 slab (`MEMRA_ST_E4M3=0`) | slab / e4m3 | distributions |
|---|---|---|---|---|
| 1 | **71.82** | 70.09 | 0.976x | DISJOINT |
| 16 | 347.23 | **401.28** | **1.156x** | DISJOINT |
| 32 | 342.64 | **401.27** | **1.171x** | DISJOINT |

Resident: e4m3 17812 MiB vs slab 18260 MiB — **+448 MiB, 2.5%**. Per-arm spread ≤0.7% everywhere;
all three cells disjoint.

This is the ALU-bound signature stated twice, independently: hoisting redundant e4m3 converts bought
+13.5% at width and 0% at m=1 (§4); replacing the remaining converts with `dp4a` buys another +15.6%
at width and **−2.4% at m=1**. The crossover sits between c=1 and c=16 and is exactly where the
column loop starts amortizing weight fetches. It also re-prices the §7 table: **the best FP8-ST
config on a batch box is 401.3 tok/s, not 348.3** — 0.88x of GGUF-Q8RP's 454.1 at **22.5 tok/s per
resident GiB** (2.5x GGUF-Q8RP's 8.85), which moves the day-one statement without changing its shape.

**Not flipped as a default here.** The slab arm is a per-m regression at c=1, and single-stream is
the interactive path the owner runs daily; a width-conditional residency choice is a dispatch
decision with its own exactness surface, not a flag flip at the end of a lane. What this measurement
establishes is the *mechanism* and its size, which is what §8 item 1 needs before it is built.

---

## 7c. A routed find, absorbed: the per-block FP8 prefill tile was on the slow MMA form

Coordinator-routed from `research/w4a8-prefill-20260806/`: `cu/mmq_fp8_blk.cu:214` issued
`mma.sync.aligned.kind::f8f6f4.m16n8k32` on a live path (the `MEMRA_PP_FP8` direction, default ON
for `QT_F8_E4M3_BLK` native residency). On sm_120a that form costs **32.02 cyc/warp-MMA**, while
`kind::mxf8f6f4.block_scale.scale_vec::1X … ue8m0` with the identity scale `0x7F7F7F7F` computes the
**identical** e4m3×e4m3 m16n8k32 product at **16.06 cyc**. The line's own comment claimed the
"381-TF class"; the form it issued is a 155-TF class. Absorbed here rather than routed onward because
this lane was actively editing the file.

**Exactness, proven on this kernel and not inherited** (both directions, `fp8-mmq-check` +
live-path):

- `fp8-mmq-check` ALL GREEN on both forms, and the two reports are **byte-for-byte identical** —
  `diff` of the verdict lines is empty. 9 shapes × 2 arms; EXACT arm 0 bit-mismatch in every cell
  (0/16384 … 0/786432); RAND arm identical `max_abs`/`rms_rel`/mismatch counts everywhere; 254/254
  legal e4m3 codes covered. (`fp8-mmq-check-blksc.log`, `fp8-mmq-check-plainform-control.log`)
- Live prefill logits at pp512 on the 27B block-128 checkpoint, **1040 dispatches**: **0 of 993280
  bytes differ**, md5 `7dc6325379c0` under both forms, two independent passes.
  (`pplogits-{blksc,plain}-p{1,2}.log`)

**Perf**, N=6 adjacent alternating pairs (pairsweep law), idle card, 368 MiB resident at every pair
start, clocks 1807–1852 MHz: blksc median **1472.8** tok/s (min 1472.4, max 1473.2) vs plain
**1382.4** (1382.0–1382.8) → **1.0654x, DISJOINT**, per-arm spread <0.09%.
(`fp8blk-form-ab.sh`, `fp8blk-form-ab.log`)

+6.5% and not +99% is itself consistent with the correction: this tile measures 110–130 TF and was
never MMA-bound, so halving MMA issue cycles pays only its Amdahl share. The denser W4A8 tile got
1.2153x from the same one-line change. Rollback seam: `MEMRA_MMQ_FP8BLK_PLAIN=1`. No-regression
battery in `blksc-noreg.{sh,log}` (kernel-check ALL GREEN, 3/3 argmax MATCH, both 27B ST classes
`0.000e0`).

### 7c-bis. The same defect's THIRD site — and it invalidates a published GO

Auditing for the routed pattern found it in `cu/mmq_q8_0_f32acc.cu`, the **Q1 instrument of the
FP8-ST v3 gate** (`research/fp8v3-gate-20260805/`). That instrument's whole claim is that the
accumulator is its "one free variable": S32 arm on `m16n8k32.s8`, F32 arm on plain
`kind::f8f6f4.m16n8k32`. Those are 16.06 and 32.02 cyc/warp-MMA respectively — so the published
`delta_pp` was the **sum** of an accumulator effect and a 2× issue-interval effect. Its SASS census
validated MMA *count* and never checked MMA *rate*.

Re-measured with the form equalized (3 adjacent alternating binary pairs per cell, both
`ACCPROBE_DIST` settings, S32-arm time as a drift control: −0.83…+0.33% across 24 shape cells):

| weighted delta (gate's own multiplicity weights) | published arm, re-measured | equalized |
|---|---|---|
| m=512 | +17.3pp | **−6.7pp** |
| m=6257 | +17.6pp | **−8.8pp** |

**The sign flips.** All of Q1's delta was the MMA form; none of it was the accumulator. With the
interval equalized, f32 accumulate is 5–9% *faster* than s32 at fixed geometry — the s32 arm pays
512 `I2FP` converts per instantiation, an asymmetry the old census recorded and scored backwards as
"generous to F32". Q1's ≥10pp bar is not merely missed, it is missed in the wrong direction: **the
receipted case for an FP8-ST v3 is refuted on its own instrument.** Q2 (native-e4m3 decode) is
untouched and is not re-priced here, so the gate's "decode-v1 first" recommendation stands on Q2's
evidence alone.

`research/fp8v3-gate-20260805/VERDICT.md` now carries the correction inline at its head and at each
of the four claim sites; its `RESULTS.jsonl` carries a `q1_refutation` row. `ACCPROBE_F32_PLAIN=1`
rebuilds the published arm so the old receipts stay reproducible. Receipts:
`accprobe-form-ab.{sh,log}`, `accprobe-form-ab-summary.txt`, `accprobe-sass-census.txt`.

The fourth candidate site, `cu/mmq_nvfp4_f8f4.cu:42`, is referenced only by its own definition and
its launcher sits behind default-OFF `MEMRA_MMQ_F8F4=1` — no live consequence, left alone.

**Law worth keeping:** "one free variable" must be checked at the **cost** level (issue rate), not
only at the source level (fragment ABI, MMA count). A verdict is only as calibrated as its control
arm — the H100 lane's LAW 2 generalizes past thresholds to instruments.

---

## 8. What the remaining gap is, and what to do next

The gap is no longer a mystery to be surveyed; §4 named a mechanism and removed one instance of it,
and §7b confirmed the same mechanism a second way. The hoist recovered +13.5% of a 1.47x gap, and
the dp4a slab arm shows another +15.6% sitting in the same wall. Ranked by evidence already in hand:

1. **e4m3 still dequantizes to f32 and multiplies with fmaf, one weight at a time.** Q8_0's path is
   `dp4a` — 4 int8 MACs per instruction. That is a ~4-8x instruction-count difference on the
   multiply itself, and the hoist only removed the *redundant* conversions, not the conversion. The
   obvious next arm is an e4m3 → int8 pre-scale so the batched inner loop can use dp4a like every
   other class. **Now measured, not conjectured** (§7b): the existing `MEMRA_ST_E4M3=0` slab arm is
   **+15.6% at c=16 / +17.1% at c=32** and **−2.4% at c=1** on the same checkpoint bytes for +448 MiB,
   so the prize is real and the cost of the naive version is a single-stream regression. The kernel
   arm should keep e4m3 residency and pre-scale *inside* the batched tier so m=1 is untouched; it
   changes numerics and needs its own exactness verdict. Still the single highest-value item this
   lane surfaced.
2. **NVFP4 is 58.8% of the ST artifact and already dp4a-based** — so the ST path's remaining
   e4m3 cost is bounded at 41% of resident weight. That caps what item 1 can win and should be
   modeled before it is built.
3. **`q8_rp / q8_norp` = 1.56x is a *GGUF-layout* number, not a physics number.** It measures how
   bad the 34 B stride is, and FP8-ST's whole advantage is not having that defect. Comparing ST to
   `q8_rp` therefore compares against a competitor that paid 26 GB for its number. Both cells
   belong in any future board row, never just the favorable one.

Nothing here is blocked on owner input. Item 1 is the next lane.
