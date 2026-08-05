# PRODUCT-TRUTH — the single source of truth for product-facing claims

**Last reconciled: 2026-08-05.** Reconciled against branch `restructure/public-split` @ `ed815eee`.

## Rule zero (read this before writing anything product-facing)

Any agent or human writing a **product-facing surface** — website copy, landing pages,
pricing pages, blog posts, gateway/marketplace applications, README marketing prose,
social posts, investor or partner material — reads **THIS FILE**, not a research
directory. Research dirs are append-only lab records: they are correct as of their own
date and go stale silently. This file is the reconciled view.

Three rules with teeth:

1. **Every number carries its receipt path, its date, its rig, and its protocol caveat.**
   A number without those four is not publishable. The caveat is not optional garnish —
   several numbers below are wrong by 5-12% if you move them to a different board.
2. **Never soften a real receipt, never harden a soft one.** The exactness story, the
   fleet endurance, the 2.6x official-FP8 spec win, the miscompile catch are all solid
   and should be stated at full strength. The numbers marked SOFT below must not be
   stated at full strength.
3. **Any lane that moves a product number updates this file in the same commit.** Same
   pattern as the perf-board rule in CLAUDE.md. A lane that lands a number and leaves
   this file stale is the failure mode this document exists to prevent — it already
   happened once: a website build-agent followed stale specs and built the wrong product.

If a claim is not in this file, it is not cleared for publication. Add it here first,
with its receipt, and then use it.

---

## 1. Brand and entity architecture (owner decision, 2026-08-05)

The commercial serving brand sits **under** a parent lab. Name the lab first, build the
lab site, embed the inference product in it, and ship a separate landing page for the
inference product as the go-to-market asset.

| Layer | Entity | Name status |
|---|---|---|
| Parent | the research lab | **OWNER-PENDING — not named yet.** See §11. |
| Engine / research record | **memra** — the public Rust+CUDA engine repo. Already the lab's track record. | Settled |
| Serving product | **darklanes** — the QoS-lane serving business | Settled as the *product* name |

**Do not write "darklanes lab", "darklanes research", or "darklanes doctrine" on any
surface.** Until the owner names the parent, product-facing copy says "the lab" or "our
research lab" generically, and attributes engine/research properties to **memra** (the
engine) rather than to darklanes (the product). The owner's concern is explicit: a
research lab carrying the "dark" connotation is a liability the serving product can
absorb and the lab cannot.

### Two surfaces, not one

The earlier spec assumed a single `darklanes.ai` site. It is now two:

- **(a) The lab site** — the research record. Receipts ledger, blog, the operating-model
  story (serving revenue pays for hardware; the research is the product), about/founder,
  and a products section with the inference product embedded.
- **(b) The inference landing page** — the GTM asset. Conversion-focused: what the service
  commits to and how a user verifies it themselves, get-a-key flow, pricing, quickstart.

The operating-model narrative belongs on (a), not (b). A prospective customer on the
landing page does not need to know that their tokens fund GPU purchases.

---

## 2. Measured numbers — cleared for publication

### 2.1 The first-SKU serving board (RTX PRO 6000, rented pod)

Rig label **`pro6000wk-runpod`** — RTX PRO 6000 Blackwell Workstation 96 GB, 188 SM,
driver 610.43.02, 600 W, clocks pinned 2865 MHz, **zero throttle**, temp max 43 C on a
35 s spin probe. Date **2026-08-04**. Commit `2299ee0f+restructure/public-split`.
Receipt dir: `research/pro6000-prod-20260804/`. Journal: `pro6000wk-runpod.jsonl`.

Two artifacts, both Qwen3.6-27B, measured interleaved in the same session:
**`nv`** = `Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf` (15.7 GB) and **`q8`** =
`Qwen3.6-27B-Q8_0.gguf` (28.6 GB, `unsloth/Qwen3.6-27B-GGUF`).

| Claim | Value | Protocol | Receipt |
|---|---|---|---|
| Spec decode (MTP), NVFP4, K=3, **bare CLI** | **186.7 tok/s** | N=5 process reps, median (186.75/186.72/185.50/186.84/186.28). 2.17x the same-run plain 86.20. tg128 at d512 prompt, own-trim drafter, gen-only timing | `anchor/spec-bestk3-nv-n5-r{1..5}.log` |
| Spec decode, NVFP4, K=3, **through the serve surface** at c=1 | **170.5 tok/s** | N=5 median (170.68/170.39/170.22/170.62/170.55). Server restarted per K, 0 err / 0 shed | `serve/points-nv-spec.jsonl` |
| Plain decode tg128, NVFP4 | 86.8 tok/s | N=5 median | `anchor/plain-d512-nv-r*.log` |
| Plain decode tg128, Q8_0 | 52.6 tok/s | N=5 median | journal row 3 |
| Aggregate throughput at c=8, NVFP4 | **420.6 tok/s** | N=3 passes, median (421.18/420.57/420.43). p50 latency 2.42 s, 64 ok / 0 shed / 0 err per pass. Spec OFF, plain batched serve | `serve/points-nv-batch.jsonl` |
| Aggregate at c=8, Q8_0 | 308.7 tok/s | N=3 median | `serve/points-q8-batch.jsonl` |
| TTFT cold, NVFP4 | **0.182 s** | N=5 with per-rep `cache_salt` forcing a cache miss; median of reps 2-5. **Rep 1 = 0.340 s and is excluded as one-time session warmup.** pp512-class prompt, SSE, temp 0 | `serve/nv-ttft-cold.jsonl` |
| TTFT warm (prefix-cache hit), NVFP4 | **0.003 s** | reps 2-5: 0.0025/0.0023/0.0025/0.0022. 61x the cold number | `serve/nv-ttft.jsonl` |
| TTFT cold / warm, Q8_0 | 0.156 s / 0.004 s | same protocol | `serve/q8-ttft*.jsonl` |
| Prefill pp512 | 4118 (nv) / 4591 (q8) tok/s | N=5, arms interleaved within every rep, medians | journal line 3 |
| Q8 96 GB residency lever | +57% at c=16/32 (486 vs 310 agg tok/s, p50 6.61 → 4.21 s), 63.7 GB resident | — | `q8rp/` |

**Caveats that must travel with these numbers:**

- **`c=8` is the knee, and saturation is not a win.** c8 420.6 → c16 421.9 → c32 423.0 is
  flat *while p50 doubles at every step* (2.43 → 4.84 → 9.67 s). The journal's own word
  for c16/c32 is "queueing, not throughput." Publish c=8; do not publish c=32 as a
  throughput ceiling.
- **The TTFT protocol trap.** Unsalted repeat requests hit the prefix cache. A TTFT
  number measured without a fresh `cache_salt` per request is a warm number wearing a
  cold label. Always state cold vs warm separately.
- **4118 and 4591 are two different artifacts**, not two configurations of one. Fine to
  show side by side as "the NVFP4 arm vs the Q8_0 arm"; not fine to headline 4118 while
  implying the same weights.
- **This is a 188-SM workstation Blackwell board, not a 5090** (82 SM). A reader who buys
  a 5090 expecting these numbers will be disappointed. Label the rig, every time.
- **The board is rented.** See §3 — nothing in this class is owned yet.

Rounding note for copy: earlier drafts carried **170.6** and **421**. 170.6188 is the r4
single rep; the N=5 median is 170.55. 421.18 is the p1 pass; the N=3 median is 420.57 and
the journal itself records 420.6. Publish **170.5** and **420.6**, or "≈170.6"/"≈421" if a
single decimal is wanted — but never present a single rep as the headline.

### 2.2 Gate battery on that rig

`research/pro6000-prod-20260804/`: `kernel-check` **ALL GREEN** naked (184 `OK`) and
model-backed on real 27B weights (263 `OK`, `gate1b-kernel-check-nv.log`); `run-gen`
argmax **MATCH** both artifacts (`gate2-rungen-argmax-{nv,q8}.log`); `run-spec`
self-consistency **PASS** at K=1,2,3 (`gate3-runspec-k123-{nv,q8}.log`).

**Do not say "run-spec K=1..8" about this rig.** The prod board ran K=1..3 as a gate plus
K=4/K=5 as perf cells. The full **K=1..8 PASS** battery exists on the community board, on
the NVFP4-MTP artifact — `research/q27-deepdive-20260805/logs/gate-key48-runspec-K1to8.log`.
Q8_0 cannot run it at all (no MTP head: `RUNSPEC-Q8 rc=2`). The correct public form is
"K=1..8 self-consistency is a standing gate, run on the MTP-capable artifact."

### 2.3 Official FP8 checkpoint — the 2.6x spec result (SOLID, but a different rig)

Rig **`rig2x5090-serve`** — vast.ai 2x RTX 5090 32 GB, single card, CUDA 13.0.1,
nvcc 13.0.88. Date **2026-08-04**. Model: the official `Qwen/Qwen3.6-27B-FP8`
safetensors checkpoint (29 GB, e4m3, `weight_block_size [128,128]`, 407 block-128 scale
grids). Receipt: `research/fp8ship-20260804/official/RESULTS.md`, `rig2x5090.jsonl`,
per-rep `serve-perf.jsonl`.

| Arm | tok/s | vs plain |
|---|---|---|
| ST plain | 48.99 | — |
| **ST spec, checkpoint's own embedded MTP head** | **128.06** | **2.61x** |
| ST spec + own-trim drafter | 136.75 | 2.79x |
| ST e4m3 resident | 48.99 | flat *by construction* |

Protocol: N=5 medians per arm, SSE `/v1/chat/completions`, greedy, max_tokens 128,
pp512-class prompt, **fresh `cache_salt` per request with `cached_tokens=0` verified every
rep**, arms sequential in one session, 32-44 C.

Bit-identity on the official artifact: argmax 365==365, maxdiff 0.0 x3, prefill logit
vectors **BIT-IDENTICAL 993280/993280 bytes**. Load wall 843.9 s → 291.6 s = **2.89x**
(N=3 interleaved).

Caveats: the e4m3 arm is flat *because* every tensor on this checkpoint is block-128 and
falls through to the Q8_0 path — the win here is load time, not tok/s. TTFT **triples** on
the spec arm (0.170 → 0.466 s). The GGUF Q8_0 reference row (53.63) is cross-protocol and
cross-day and is explicitly not apples-to-apples. **There is no official-FP8 measurement
on any PRO 6000.** Label this rig `2x5090 (rented)` whenever the 2.6x is used.

### 2.4 Fleet endurance (SOLID — the strongest reliability receipt)

Receipt: `research/fleet-endurance-20260803/SUMMARY.txt`, `load-windows.jsonl`,
`greedy-hashes.txt`. Date **2026-08-03**, window 08:25:02Z → 10:46:48Z. Rig: **8x H100
80 GB HBM3, rented**, 1 replica/GPU, proxy cap 16. Model
**Qwen3.5-9B-Q8_0** (not the 27B SKU).

- **140 minutes sustained load** (70 back-to-back 120 s windows), concurrency 96
- **464,870 requests, 0 errors, 0 sheds**, 0 supervisor restarts
- Throughput drift **+0.045%** (first-10-min mean 7000.2 → last-10-min 7003.3 tok/s;
  all-70 spread 0.21%)
- p95 1.7560 → 1.7556 s (−0.4 ms). Worst single request 1.861 s. RSS delta max
  3064 KB/replica; GPU mem delta max 160 MiB
- 59,461,500 completion tokens

Three honesty requirements, all mandatory:

1. **The load ran at temperature 0.7, seeded — not greedy.** The determinism hash
   (`56b8502cfb8de57a`, identical on 16/16 lines = 8 replicas x pre/post) is a **separate
   greedy probe** before and after the soak. Correct phrasing: "a greedy probe hashed
   identical on all 8 replicas before and after the soak." Not: "464,870 deterministic
   requests."
2. **The prompts were prefix-cache hits** (`226 of 226 prompt tokens from cache`). This is
   a warm-prefix throughput number.
3. **It is a 9B-class fleet result on rented H100s**, not the 27B SKU on PRO 6000. Say
   "9B-class fleet endurance."

Minor: the proxy reports 464,940 vs the harness's 464,870 (+70, one per window, likely the
health probe). Cite 464,870 and the harness.

Companion chaos receipt: `research/fleet-v060-20260801/SUMMARY.md` — SIGKILL a replica
mid-load, breaker DOWN same second, restart +2 s, backend UP +9 s, 8/768 requests lost
(exactly the victim's in-flight cap), greedy hash identical on all 6 replicas 18/18.

### 2.5 The QoS lanes result (SOLID, with a mandatory attribution note)

Receipt: `research/qos-p95-20260802/RESULTS.md` + `box/`. Date **2026-08-02/03**. Rig:
8x H100, devices 4-7, 2 replicas/GPU. Model Qwen3.5-9B-Q8_0. Protocol: interactive c=4
alone, then bulk c=96 with interactive c=4 starting 5 s in; **4 conditions interleaved,
N=3 passes, full teardown/bring-up per cell** (12 bring-ups). One thermal window, peak
56 C. Zero request errors in every cell.

| Condition | Interactive contended p95 | Bulk aggregate |
|---|---|---|
| lane-blind, proxy cap 8 | **7.150 s** | 2010.5 tok/s |
| lanes on, cap 8 | 6.463 s | 2174.0 |
| lane-blind, cap 16 (control) | 4.335 s | 2490.2 |
| **lanes on, cap 16** | **3.690 s** | 2213.8 |

SLO dial at cap 16, lanes on, N=3 each: 50 ms → p95 3.69-4.15 s; 35 ms → 3.56 s;
**25 ms → p50 1.637 / p95 2.158 s, contended statistically equal to uncontended
(1.635/2.065), bulk pays −67%.** Inflation 4.2x → 1.24x.

**Attribution, non-negotiable:** raising the proxy cap from 8 to 16 *by itself* moves p95
7.15 → 4.34 s. The lane gate moves 4.34 → 3.69 s. **Roughly half the improvement is the
proxy cap, not the lane mechanism.** RESULTS.md says it plainly: "the engine gate cannot
fix a queue it never sees." Any copy that credits the full 7.15 → 3.69 s to lanes is
overclaiming. The honest framing: "admission-controlled lanes plus a correctly sized
queue hold contended interactive p95 at 3.69 s where a lane-blind fleet sits at 7.15 s;
at the tight SLO dial, contended p95 becomes statistically equal to an uncontended box,
and batch pays 67% of its throughput for it."

The 4x figure: contended/uncontended inflation measured 4.2x (1.74 → 7.33 s) in
`research/fleet-resweep-20260802/RESULTS.md` §3, reproduced here as 7.150 s.

### 2.6 Exactness (the core differentiator — state at full strength, with the right object)

**The isolation gate.** Driver `research/concat-prime-exact-20260802/run-serve-gate.sh`,
comparator `tools/check-batch-exact.py`. It fires 16 prompts sequentially (c=1) and then
concurrently (batched), against **the same running server**, and byte-compares every
stream. 16 prompts x 96 tokens x 4 models, temp 0, seed 0. Receipt:
`serve-gate-matrix.jsonl`, 2026-08-02.

Result: **PASS 16/16 on all four models at defaults.** With
`MEMRA_ROUTER_PREFILL_EXACT=0` (a numeric A/B seam, never a serving config) it FAILS
7/16 and 6/16 — which is how the defect was found.

The **only** correct public wording:

> Greedy serving is isolated-identical under concurrent load at defaults: a request's
> output tokens are byte-identical whether it arrives alone or inside a full batch.
> Gated, not assumed — the gate replays the same prompts at c=1 and c=16 and
> byte-compares every stream.

What you may **not** write: bare "byte-identical" with no object; "byte-identical to
single-token reference decode"; "one canonical output"; "token-identical to the CLI
oracle." The batched-plain path is **confirmed non-identical** to a tokenwise oracle —
near-tie FP flips at 0.19-0.81 per 100 tokens, 16/16 flips independent, all gaps
sub-0.2, n-stable, reproducible across restarts
(`research/plainbatch-20260804/RESULTS.md`, 2026-08-04, verdict: "FP NEAR-TIE CLASS
CONFIRMED, n-STABLE — no new bug, no code change"). A customer can reproduce the
counterexample in minutes. Scope the claim to the batch object and it is bulletproof.

**The defect the gate caught (the story worth telling).** Under batched prefill an
m-dependent router GEMM changed which **expert set** a MoE request activated depending on
who arrived with it: **121/760 = 15.9% of (layer,token) pairs** differed in set, plus a
further 217 pairs differing in order only. Model Ornith-1.0-35B Q4_K_M at total_m=75, arm
`exact0` = **pre-fix**. Receipt: `research/concat-prime-exact-20260802/findings.jsonl`,
`kind: route_trace`. Mechanism isolated in the same file: the router cuBLASLt GEMM is
m-dependent (maxdiff 0.0039), the in-house `router_gemv` is m-invariant (0.0), and 36
trunk weights are m-invariant — which pins the defect to two router GEMMs. Fixed with
m-invariant router twins; post-fix gate 16/16 on all four models.

Say "16% of (layer,token) pairs picked a different expert set depending on who arrived in
the same batch — 121/760, Ornith-35B Q4_K_M, before the fix." Note: a "post-trains have
tighter margins" explanation was tested and **REFUTED** in the same receipt; do not
reuse it.

**Other exactness receipts, all with named objects (safe as written):**

- Prefix-cache hit is bit-identical to the run that computed the prefix — gated 16/16 +
  16/16 (`research/prompt-cache-20260802/gate-exact.jsonl`)
- Device-mask constrained greedy is byte-identical to the host `-inf` oracle;
  spec-constrained byte-identical to plain-constrained; graphed byte-identical to eager;
  draft-masking ON byte-identical to OFF across 7 cells
  (`research/constrained-full-20260803/`, `research/draft-mask-20260804/`)
- Hopper promotions are compile-gated so the naked sm_120a build stays byte-identical
- The nvcc miscompile catch: nvcc 13.0.88 at -O3/sm_120a dropped one of two adjacent byte
  stores in the FP8 block-128 dequant kernel, zeroing the low byte of every block scale.
  Caught by the `kernel-check` bit-parity arm on **first run** (3968/680/8 bad bytes on
  three shapes). Unshipped consequence: first-token logit max_abs diff 3.04, greedy stream
  diverged at token 14. 40-line repro committed
  (`research/fp8ship-20260804/official/miscompile/halfstore_repro.cu` 8/8 fail,
  `halfstore_fix.cu` 0/8). Law extracted: parity gates re-run per **toolchain**, not per
  commit. One honesty note: the original failing logs were lost to `rsync --delete`; the
  repro survives, the raw pre-fix logs do not.

**Constrained-decoding cost — scope it to the plain lane.** 99.4% of unconstrained
(123.7 vs 124.4 tok/s) is the **plain** lane: rig local RTX 5090, model Qwen3.5-9B NVFP4,
256-token greedy, N=3 interleaved same session
(`research/constrained-full-20260803/RESULTS.md`). The **spec** lane pays far more:
153.4 vs 194.4 = **79%**. Never quote 99.4% for the spec path. Mask cost 0.006-0.007
ms/step. One documented exactness exception: unbounded-schema whitespace tail.

---

## 3. Target platform and hardware (owner override, 2026-08-03)

**The owned trajectory is RTX PRO 6000 Blackwell class, homogeneous.** Not 2x RTX 5090.

Owner verbatim: *"buying now 5090 that cant scale later with the 6000 is missuse."*
Receipts: `research/hw-growth-rethink-20260803/ASSESSMENT.md` §"OWNER OVERRIDE
(2026-08-03, post-assessment): box #1 is NOT 2x used 5090" (lines 377-402), and the
owner memory note `target-platform-2x5090.md`. The staged plan's box #1 (2x used 5090,
~$9.5-10k) is **rejected on scaling-continuity grounds**. Revised box #1: a single
used/refurb RTX PRO 6000 ($9.5-11k, the same money class), later the 2-card 192 GB box
($24-28k) when earnings justify. Trade-off accepted knowingly: 2x5090 has ~2x the
aggregate bandwidth per dollar (3.58 vs 1.79 TB/s at equal price); the owner prioritizes
scaling continuity — native P2P, cards compound into one scaling group, same sm_120a so
every receipt transfers.

| | Hardware | Status |
|---|---|---|
| **Owned today** | RTX 5090 **Laptop** (GB203, 82 SM, 858 GB/s) — local proof rig | The only owned GPU |
| **Owned trajectory** | RTX PRO 6000 Blackwell class, homogeneous | **Buy plan, earnings-gated. Nothing purchased.** |
| Rented — prod tuning | AWS G7e (PRO 6000 Server Edition) $3.36/hr OD; RunPod PRO 6000 $1.69-1.99/hr | Where every PRO 6000 number came from |
| Rented — dev/measurement | vast 2x5090 ~$0.75/hr | The measurement platform |
| Rented — fleet | 8x H100 80 GB | Where the fleet/QoS receipts came from |

Buy trigger: cumulative metered gross margin since the last purchase ≥ 0.5x the next
unit's price (`research/hw-buy-20260802/REPORT.md`).

**Three precise statements, to avoid both stale and false claims:**

- 2x5090 is **dead as an owned purchase** — explicitly rejected.
- 2x5090 is **alive and load-bearing** as the rental measurement platform and the
  small-SKU / customer-site serving-shape reference. Do not write "the 5090 is dead."
- **No datacenter or desktop card is owned.** Never write "we run on PRO 6000" without
  "rented" or "pod". The correct product-facing form is: "measured on RTX PRO 6000
  Blackwell (rented pod); the owned build-out targets the same silicon."

**Stale-text hazard:** the pre-override 2x5090 recommendation still sits un-struck inside
the two hardware studies (`research/hw-growth-rethink-20260803/ASSESSMENT.md` §0 lines
39-41, `research/hw-buy-20260802/REPORT.md` "First-box recommendation"). Only the appended
OWNER OVERRIDE section supersedes them — they are left as-is by design, because a research
dir is an append-only record of what was recommended on its date. Do not read a
first-box recommendation out of either file.

*Fixed 2026-08-05 (lane/product-truth):* the five files that called the local 5090 "the
deployment and final performance target" have been rescoped to "measuring and gating rig" —
`CLAUDE.md`, `CONTRIBUTING.md`, `HANDOVER.md` (plus a header warning that it is an
append-only log, not a copy source), `research/benchmarks.md` (rig/target correction under
its llama-freeze banner), `research/8bit-decision-20260803/DECISION.md` (superseded-target
note), `research/per-expert-quant/README.md`.

---

## 4. The SKU story

**Today: one model class — the Qwen 27B at 8-bit.** `Qwen3.6-27B`, served as Q8_0 GGUF
(the shipping 8-bit arm) with an NVFP4+MTP arm for the speculative fast lane.

**Day one for Qwen3.8-27B: planned, runbook written, date not owned by us.** Runbook:
`docs/qwen38-bringup-runbook.md` (written 2026-08-03, revised 2026-08-04). Two legs —
Leg A FP8-ST (official FP8 checkpoint served as a safetensors dir, the prod direction),
Leg B GGUF (house NVFP4+MTP, the only leg with spec/drafter rows day one). Deployment bar
before the word "supported" is used: ≥1.1x end-to-end with an own-gen trimmed drafter.

Release timing: **"week of 2026-08-10"**, from the official @Alibaba_Qwen post
2026-08-03 ("Next week … Qwen3.8-27B is also going open-weights") plus press
corroboration. As of the newest source in-repo (2026-08-05) it is **not released**.
Receipt: `research/qwen38-prep-20260803/WATCH.md`.

**What is not known and must not be implied:** the 3.8-27B architecture, license, active
parameter count, and benchmarks are all unpublished. Same-architecture is an
*expectation*; the runbook's arch-diff step is the verification. Safe copy: "a day-one
bring-up runbook is written and ready ahead of the expected release." Not: "day-one
support guaranteed."

Approved phrasing already in use and substantiated
(`research/or-application-20260805/APPLICATION.md`): *"the Qwen 27B class at 8-bit
(Qwen3.6-27B today; day-one bring-up runbook for Qwen3.8-27B ready ahead of its expected
release)."*

**Single-SKU is framed as depth, never apologized for.** What "supported" means here:
`kernel-check` ALL GREEN, `run-gen` argmax MATCH, `run-spec` K=1..8 self-consistency on
the MTP-capable artifact, the serve isolation gate, and the ≥1.1x deployment bar.

**Open conflicts to resolve before this ships** (both are owner calls, neither is
publishable as-is):

1. `research/sku-repick-20260802/REPORT.md` (2026-08-02) names a *different* launch SKU —
   Qwen3.6-35B-A3B with Step-3.7-Flash as flagship. Stale against the q27 story. Task
   #53 still tracks Step-3.7-Flash bring-up.
2. The 8-bit bridge story conflicts: `research/8bit-decision-20260803/DECISION.md` says
   "Q8_0 GGUF serves now, FP8-ST is the tuning track with a ≥1.1x promotion gate", while
   the owner memory (2026-08-03) hardens it to *"we will finish the st before 3.8 day
   one"* — no Q8_0 bridge, 3.8 launches directly on FP8-ST. DECISION.md is the stale
   half. Pick one before any Models page names a format.

---

## 5. Precision arms — what may be claimed today

**Q8_0 GGUF is the shipping 8-bit serving arm.** Everything FP8-native is either default
off or non-shippable.

| Arm | Status | Evidence |
|---|---|---|
| Q8_0 GGUF | **SHIPPING** | `research/8bit-decision-20260803/DECISION.md` |
| NVFP4+MTP GGUF | **SHIPPING** (the spec fast lane) | `research/pro6000-prod-20260804/` |
| `MEMRA_FP8_BLK_GPU` — GPU-side block-128 dequant | Exact, **byte-identical** to the CPU path, 3.87x faster load. **Default OFF** (opt-in until 27B-class rig gates run) | `docs/FLAGS.md`; official-artifact bit-identity in `research/fp8ship-20260804/official/` |
| `MEMRA_FP8_MMQ` — native per-block FP8 MMQ tile | Exactness **delivered**; performance **negative** (0.81-0.94x the Q8_0 MMQ floor). Does not clear the 1.1x bar. **Default OFF** | `research/fp8st-20260804/mmq/LANE-VERDICT.jsonl`, `mmq-v2/LANE-VERDICT.jsonl` |
| `MEMRA_FP8_FOLD` — lossy per-tensor amax fold | **NOT SHIPPABLE.** +18.4% pp512, but greedy diverges at generated position 20 (102/128 tokens differ); argmax MISMATCH | `research/fp8st-20260803/gemm-arm/armA-vs-floor.jsonl` |

Platform limit worth knowing: cuBLASLt block-scaled FP8 is **not supported on sm_120**
(heuristic status 7/15, nh=0, every m, both D dtypes; only per-tensor SCALAR_32F).
Receipt: `research/fp8st-20260803/gemm-arm/P1-VERDICT.md`. This is *why* the exact
compute arms stay off — not neglect.

**Cleared Models-page wording:**

> Q8_0 GGUF is the shipping 8-bit serving arm. FP8-E4M3 safetensors checkpoints
> (block-128 scale grids, including the official `Qwen/Qwen3.6-27B-FP8`) load, gate
> green, and serve — dequantized to the Q8_0 arm byte-identically, verified on the
> official artifact with prefill logits bit-identical at 993280/993280 bytes. GPU-side
> FP8 dequant cuts the 29 GB load wall 2.89x and is available opt-in. A native per-block
> FP8 MMQ tile is implemented and bit-exact but does not yet clear our 1.1x deployment
> bar, so it stays off by default. A lossy per-tensor scale fold buys 18.4% prefill and
> is not shipped — it changes what the model says.

**Three things you may not say:** that FP8 is "the" serving format (Q8_0 is); that an
FP8-native compute path is enabled by default (both exact arms are off); that FP8 is
faster at inference (the e4m3-resident serve arm is *flat by construction*; the win is
load time).

---

## 6. Serve surface — the v0.69 feature list

Authoritative source: `docs/SERVING.md`. Everything below is gated, with a receipt.

- **OpenAI-compatible** `/v1/chat/completions` + `/v1/completions`, validated against the
  official `openai` Python SDK. Envelope: id, created, `system_fingerprint = memra-<git
  sha>`, `x-request-id`. **Honest 400s** on params it cannot honor (`logit_bias`,
  `logprobs`, `n≠1`, `best_of≠1`) — never silent ignoring. (`research/serve-compat-20260802/`)
- **Streaming** SSE with keep-alives every 5 s; reasoning separation (OpenRouter dialect).
- **Tools / function calling**: `tools`, `tool_choice` auto/none, streaming `tool_calls`
  deltas, malformed blocks surfaced verbatim rather than dropped.
  (`research/serve-tools-20260802/`)
- **Real constrained decoding**: `response_format` json_object/json_schema via
  llguidance, on-device bitset masking, draft-side masking makes proposals legal by
  construction. Cost 99.4% of unconstrained **on the plain lane** (see §2.6 for the spec
  caveat).
- **Cross-request prefix caching**, LRU, `MEMRA_PREFIX_CACHE_MB` default 256 MB, with
  **per-tenant `cache_salt` isolation**. Hit is bit-identical to the computing run.
  (`research/prompt-cache-20260802/`, `research/pc-iso-20260802/`)
- **Honest usage metering**: worker-truth `usage` on every shape including streams;
  `prompt_tokens_details.cached_tokens` itemized; disconnected requests billed to the
  abort point; `/metrics` cumulative splits.
- **QoS lanes**: `x-lane` header (interactive / judge / harvest), shed at **admission**
  with 429 + Retry-After, never queued inside the engine; interactive never preempted;
  per-lane prefill budgets; `MEMRA_SLO_P99_MS` dial. Naked traffic = interactive.
- **Gateway-ready**: OpenRouter-schema `/v1/models` (context_length, architecture,
  pricing stub, honest nulls), truthful `X-RateLimit-*` headers riding the SSE stream,
  graceful drain SIGTERM → drain → exit 0 (`MEMRA_DRAIN_S` default 30 s).
  (`research/serve-tail-20260804/`)
- **Safetensors checkpoint serving**: official Qwen FP8 block-128 dirs load bit-exact,
  2.89x faster load, spec decode runs out of the box on the checkpoint's embedded MTP
  head. The former dir-checkpoint spec quarantine is **lifted**; dir checkpoints are
  spec-eligible by default (`MEMRA_SERVE_SPEC=0` is the rollback door).
- **NEW since the website spec was written — API keys and tenant auth**
  (`research/apikeys-20260805/RESULTS.md`, 2026-08-05): keyring TOML with
  `sha256`/`tenant`/`lane`/`enabled`/`rate_limit`; `--gen-key` / `--revoke-key`;
  401 on unknown, 403 on disabled; per-tenant cache namespace `t:<tenant>␟<cache_salt>`;
  a batch-class key on `x-lane: interactive` is refused 403; per-key rate limit is
  `min(override, global lane cap)`. Live gate **18/18 PASS**, hot revoke ≤2 s, two-tenant
  cache-hit oracle verified (same tenant shares, different tenant misses), back-compat
  serve-smoke 0 failed, 59/59 server bin tests. Protocol note: single interleaved run per
  gate — these are behavioral pass/fail results, not perf medians.

- **NEW, later the same day — session affinity (task #71)**
  (`research/session-affinity-20260805/RESULTS.md`, merged `70ce5a0f`): resuming a
  conversation whose history the client **rewrote**. Both earlier reuse tiers required the
  new prompt to *extend* what was cached; real agent clients rewrite instead (the owner's
  strips `<think>` blocks from prior assistant turns), so every turn missed and re-primed the
  whole growing conversation. Identity **nominates** (explicit `session_id`/`user`/
  `x-session-id`, or an implicit control-token-segment fingerprint chain needing 3 shared
  leading segments), **bytes decide** (resume only if the prompt reproduces the session's
  committed tokens exactly to its checkpoint; any divergence = full re-prime). A fingerprint
  collision therefore costs one wasted comparison, never a wrong resume.

  Measured, rig **local RTX 5090 Laptop**, owner's daily serve config verbatim, the same
  recorded transcript replayed by both arms, **N=3 interleaved** reps, per-turn median,
  thermal ramp 61 → 85 C spread across arms by the interleave:

  | Turn class | affinity ON | OFF | Note |
  |---|---|---|---|
  | 0 (cold prime) | 9.882 s | 9.962 s | nothing to resume — **1.01x** |
  | 1, 23 (pure extension) | 0.590 / 0.544 s | 0.591 / 0.541 s | the prefix probe already served both arms |
  | 2-22, 24 (rewritten history) | 0.525-0.645 s | 11.28-14.03 s | **20-24x**; sum-of-medians over 25 turns 23.1 s vs 287.2 s = 12.4x |

  **The publishable claim is FLATNESS, not the ratio:** TTFT stops scaling with conversation
  length (0.525 s at 13.1k prompt tokens → 0.548 s at 14.6k, where OFF goes 11.89 → 13.36 s).
  Quote the ratio only with the turn class attached — turn 0 and extension turns are 1.00x by
  construction, and a bare "24x faster" is refutable on the first cold request.

  **What it does NOT fix, and §7.2 still stands:** the ~0.53 s figure is the *fixed per-turn
  floor* (rewind + delta prime + first decode step), which is the same 0.53 s that loses
  cold-turn TTFT to llama.cpp's 0.19 s in the dogfood run. Affinity removes the re-prime that
  made turn N *worse than* turn 1; it does not lower the floor. No interactive-latency
  superiority claim follows from it. Wall-clock corroboration is smaller by construction
  (5.30x sum-of-medians) because decode time is identical in both arms.

  Gates: full battery GREEN, 0 failed, including a byte-identity arm (`MEMRA_AFFINITY=0` is
  the rollback seam) and `serve-smoke` check 10 added to guard what the lane owns.

This is a **product feature the earlier website spec predates.** The site's get-a-key flow
and the pricing page's per-tenant story both depend on it and should now assume it exists.
Session affinity is the stronger *story* of the two for an agentic-coding audience — "turn 20
answers as fast as turn 2" is the shape they feel. It has its own page block in the website
spec (§7.1 block 4b, "fig. 03"), its own wording rule (§2a), and its own blog post
(§10 post 6 / BLOG-EVIDENCE §G); all three were added 2026-08-05 in this lane.

**Crates / distribution.** 9 publishable crates (10 workspace members;
`memra-probe` is `publish = false`), all at **0.69.0**, MIT, intra-workspace deps pinned
`=0.69.0`. The first tagged publish landed **5 of 9** before crates.io's new-crate burst
limit returned 429; a resumable per-crate workflow then landed (skip-if-live via the
registry API, 6 attempts x 620 s backoff, plus a `publish=true` dispatch door). Receipts:
`docs/RELEASING.md`, `.github/workflows/publish.yml`, commit `c52edfd3`. **Do not claim
"all 9 crates live" without checking the registry** — the repo's only 9/9 evidence is a
dry run (`0d41fdec`). *(`docs/RELEASING.md` and `HANDOVER.md` carried the "all nine names
claimed" reading; both were corrected 2026-08-05 — commits `dbe12cb3`, `6c2a7e73`.)*

---

## 7. Honest gaps — publish these; they are the trust engine

The site's credibility comes from stating limits in the same breath as wins. Each item
below is a real, receipted gap. None of them is a reason to hide a number; all of them
are reasons not to overstate one.

### 7.1 The serve path is currently slower than the naked CLI

**This is a gap being closed, not a marketing number. Never publish it as a product
metric — but never let copy imply serve-path parity either.**

- **−11.74%** at c=1: `memra-server` 46.09 tok/s (N=3 passes, median) vs `run-gen`
  naked default 52.22 tok/s (**single run**, the post-lever verification log). Artifact
  Q8_0, rig `pro6000wk-runpod-community`, same board, same commit, same prompt.
  Receipts: `research/q27-deepdive-20260805/RESULTS.md` §4,
  `logs/serve-points.jsonl`, `logs/gate-key48-default-Q8_0-n128.log`.
- **−8.66%** on the NVFP4 spec path, prod board: serve 170.55 vs bare 186.72.
- **Root cause is known, not inferred.** The serve worker routes B=1 through
  `decode_step_batch`, which (a) has no CUDA-graph door — `MEMRA_GEN_GRAPH` lives in
  `generate_with`, which the worker never calls — and (b) dispatches dense-FFN gate+up via
  `matmul_pre` at `b_n=1`, so the m=1 fusion lever never fires. Proof for (b): the same
  A/B that wins +0.94% with 5/5 winning pairs in `run-gen` measures **+0.06% with
  sign-flipping pairs** through the serve path.
- **Filed** as `research/q27-deepdive-20260805/PHASE2-SPEC.md` hypotheses **H1** (route
  B=1 to the graph-door path) and **H3** (`b_n==1` fast path onto the m=1 dispatch).
  This is task **#70**. Earlier drafts cited "#71" — that is a different task
  (session-id affinity) and the reference is wrong.

### 7.2 llama.cpp currently wins the interactive-latency path

**Until §7.1 and PHASE2-SPEC land, no surface may claim interactive-latency
superiority over llama.cpp.**

Receipt: `research/memra-vs-llama-daily-20260805/RESULTS.md` + `META.md` (2026-08-05).
Note: this lane lives in the `bw24` tree, not `bw24-unified`. Rig: local RTX 5090 laptop.
Protocol, and it is a strong one: **inode-verified identical model artifact** for both
engines, each engine run with **its owner's exact daily flags**, spec-vs-spec (llama's MTP
draft active), **N=5 interleaved**, nonce-defeated prefix caches, seeds 1000+rep,
server-truth cross-checks agreeing within ~3%.

| Scenario | memra | llama.cpp | Winner |
|---|---|---|---|
| Short-turn cold TTFT | 0.53 s | **0.19 s** | llama, 2.8x |
| Short-agentic sampled decode | ~73 | **84-86** | llama |
| 4k prefill | 1.2k tok/s | **2.1k tok/s** | llama |
| Long-gen sampled decode | **89.8** | 76.5 | memra, +17% |
| t-matched ctx4k decode | **76.4** | 74.2 | memra |
| t=0.8 ctx4k | **80.5** | 68.1 | memra |

Three named causes: F5 evict-realloc firing on 12/12 requests (per-request full-floor
session realloc under VRAM pressure); the prefill wall; short-context spec acceptance
0.55 vs 0.73.

The receipt labels itself **"DOGFOOD-EXPERIENCE DIAGNOSTIC, NOT BOARD MATERIAL."** Do not
put this table on a Models or Benchmarks page as a competitive claim — it is honest-gaps
and blog material. Cleared wording:

> On a single 5090 laptop, same model file, each engine with its own daily config,
> memra leads on long-generation sampled throughput (+17%) and 4k-context sampled decode.
> llama.cpp currently leads on cold time-to-first-token (0.19 s vs 0.53 s), short agentic
> turns, and raw prefill. We are not claiming interactive-latency superiority; the three
> causes are identified and the lanes are open.

**Do not read the session-affinity result (§6) as closing this gap.** The two share the
same 0.53 s number and mean different things, so a surface that mixes them is refutable:

| | §7.2 cold TTFT | §6 session affinity |
|---|---|---|
| What is measured | *first* turn, nothing cached | turns 2+, history rewritten by the client |
| memra | 0.53 s | 0.525-0.645 s |
| Compared against | llama.cpp's 0.19 s **on the same turn** | memra with `MEMRA_AFFINITY=0` (11.3-14.0 s) |
| Claim it supports | none — this is a loss | TTFT does not grow with conversation length |

Affinity removed the re-prime that made turn 20 *worse than* turn 2; it did not lower the
per-turn floor, and the floor is what loses to llama. On turn 0 the two arms measure 9.882
vs 9.962 s — **1.01x**, i.e. affinity does nothing for a cold request. The §7.2 prohibition
stands until §7.1/PHASE2-SPEC land.

Standing posture note: **llama benching is stopped** (owner, 2026-08-03). The llama
numbers in the boards are **frozen reference points** recorded through 2026-08-03; all
forward work is self-competition. `research/benchmarks.md` carries the doctrine banner;
`docs/PERFORMANCE.md` got the same banner on 2026-08-05 (commit `8628bbc2`), including the
counter-example above.

### 7.3 First-token cross-config drift (~7%)

Across three distinct numeric prime configurations, **10 of 144 first tokens flip
(6.9%)**, and **every flip sits at a top1-top2 margin ≤ 0.70**. 6 models x 24 prompts.
Determinism control: 144/144 bit-identical. The dense-Q8_0 fleet class flipped **0/48**.
Escape: `MEMRA_PRIME_TOKENWISE=1`. Receipt:
`research/prime-gate-coverage-20260802/RESULTS.md`. Reported, bounded, documented — not
gated away. This is a *first-token, cross-config* class; it is not the same thing as the
isolation contract in §2.6, and the two must not be conflated in copy.

### 7.4 Batched-plain near-tie flips

The batched-plain serve path is not byte-identical to a tokenwise oracle: 0.19-0.81
independent flips per 100 tokens, all gaps sub-0.2, n-stable, reproducible across
restarts, and not accumulating error (post-flip separation, not drift). Verdict: accepted
FP near-tie class, no code change. `research/plainbatch-20260804/RESULTS.md`.

### 7.5 The sampler bug class (fixed — and the honest version is an asset)

Three bugs, one class: **a meaningful zero is not "unset."**

- **F4 (2026-08-04)**: `#[serde(default)] temperature: f32` → 0.0, and `is_greedy()` is
  `temperature <= 0.0`, so every client omitting temperature got **greedy argmax** instead
  of OpenAI's documented 1.0. Live symptom: the owner's own daily agent locked into ~10
  identical tool-call cycles. Fixing temperature was **not enough** — `#[serde(default)]
  seed: u64` → 0 is a valid *fixed* seed, and the loop survived. Fixes: default temperature
  1.0; `seed: Option<u64>` with fresh entropy. Explicit `temperature: 0` and `seed: 0` are
  still honored exactly. Receipt: `research/sampledspec-20260804/RESULTS.md`.
- **The `!` injection (found and fixed 2026-08-05)**: `top_p`/`min_p` truncation injected
  token id 0 (`!`) into output — `!bash`, `grep -!q`, `/!etc/hosts`. Cause: in the
  sampled-spec full-accept path, the filter stats were taken from
  `col_stats.last()` — a *neighbouring column* — so a foreign `row_max` mis-scaled the
  exponent, every id failed the threshold, the row masked to -3.4e38, and the argmax fell
  through to the smallest-index tie-break. Fragility scaled with the threshold: `min_p
  0.05` hit a **100% id-0 rate**; plain `top_k 40` was clean; **memra's own default (no
  truncation filter) is structurally immune**, which is exactly why the owner's daily
  never showed it. Fixed in `d1dc79b8`, merged `44c4c6a4`. A differential serve-smoke
  matrix (`9bbd3cca`) now catches it and was **proven in both directions** — 3 failures on
  the pre-fix binary, 0 on the post-fix. Receipts: `research/sampfix-20260805/`,
  and the isolation matrix in
  `research/memra-vs-llama-daily-20260805/logs/posthoc-lsampler.txt`.
- **The structural lesson, and the fix to the process**: every golden in the repo ran
  temp=0, which routes around the sampler chain entirely — a broken sampled path was
  **invisible to the entire argmax gate battery**. The hole is now closed by a
  distribution-level composition gate (arm 6 of `sample_check.rs`): the composed
  accept-walk output distribution vs the target p over 20k draws, L-inf 0.012 / TV 0.05
  thresholds. Its negative controls fail as designed — notably "forgot the residual"
  trips TV 0.0881 **with acceptance unchanged**, i.e. a bug that every isolation arm
  would have passed.

Publish this. It is a credibility asset: a bug that only bit clients configured like
*other* engines, found by dogfooding, fixed with a gate proven against the pre-fix binary.
What must be stated honestly alongside it: the bug shipped for a window, and a greedy-only
gate battery was structurally blind to it.

### 7.6 What does not exist yet

- **No SOC2, no multi-region, no 99.99% SLA.** Compensating controls: public receipts, a
  status page, drain-not-drop deploys.
- **No power-curve data on any healthy PRO 6000** — every prod cell ran at 600 W fixed
  because `nvidia-smi -pl` was container-blocked on every pod. Any 300/450/600 W
  comparison in the repo is third-party, not ours.
- **No tensor parallelism** — one GPU per engine process; multi-GPU boxes serve as a
  replica fleet. The pipeline-parallel seam is merged, default off.
- **No q27-at-8-bit row in the published board** (`research/tune-data/current-board.json`
  carries Qwen3.6-27B as NVFP4 / Q4_K_M MTP-baked only). The q27 Q8_0 absolutes measured
  so far came off a **community pod running 5-11% below the prod-class board** and its own
  RESULTS.md says "relative deltas are the currency; every absolute row here gets
  re-minted on prod-class silicon before it goes near a published board." Publish 52.6
  (prod q8 plain) and the +4.8% relative — never the community absolutes 49.82/52.22.

---

## 8. Pricing posture

**Structure is decided. Every number is an owner call and is marked UNDECIDED.**

Four products, priced by the latency guarantee:

1. **Interactive lane** — per-token, premium. Admission-controlled p95, never preempted,
   spec-decode single-stream speed. The headline price. **UNDECIDED.**
2. **Harvest lane** — per-token, discounted. Sheds with 429 + Retry-After under
   interactive pressure rather than silently queueing. Market anchor for batch tiers:
   40-60% of interactive (Novita batch = 50% off; Parasail publishes a parameter-band
   batch grid at $0.07/$0.22 for 21-41B). **UNDECIDED.**
3. **Dedicated** — per-hour/month reserved replica, anchored to GPU-hour market rates
   plus margin. **UNDECIDED.**
4. **Cached input** — the market convention is exactly **25% of input price** across the
   surveyed endpoints, and memra meters `cached_tokens` honestly either way.
   Recommendation 25%. **UNDECIDED.**

Market anchors, live as of 2026-08-05 (`research/darklanes-website-spec-20260804/RESEARCH-COMPETITORS.md`,
`research/or-provider-20260802/REPORT.md`):

- Qwen3-32B (nearest dense proxy): DeepInfra $0.08/$0.28, Nebius $0.10/$0.30,
  SiliconFlow $0.14/$0.57, Groq $0.29/$0.59 — only **5 endpoints**, a thin market.
- **Qwen3.6-27b, the direct comp: Groq prices it $0.60/$3.00** — an order above the
  DeepInfra floor, proving the class supports guarantee-priced tiers.
- Gemma-3-27B (27B-dense comp): $0.08-0.15 in / $0.16-0.46 out.
- The class clusters at **$0.08-0.15 in / $0.16-0.60 out**.

Three posture rules that are *not* undecided:

- **Do not price to the floor.** The floor fell ~35% in 90 days on one measured family,
  and a saturated replica at floor prices grosses only $2-4/hr. Cost coverage comes from
  the guarantee tiers.
- **Whatever ships must match `/v1/models` pricing metadata exactly** — OpenRouter and
  Hugging Face consume it programmatically and will cross-check.
- **Free credits: small and one-time**, not a perpetual free tier. A single-founder fleet
  cannot police abuse; one crypto-native provider's free tier was destroyed by >10k bot
  signups and killed. Amount **UNDECIDED**.

Business posture (owner-set): serving revenue covers hardware; the research is the
product. Success bar = **cost coverage + public reliability stats**, not market share.
Owner verbatim: *"Everything need to be honest, nothing is a dream that we try to make
true."*

---

## 9. Distribution channels (state of play)

- **Hugging Face Inference Providers** — the registration path is mapped
  (`research/hf-inference-20260804/`): OpenAI-compatible APIs skip most schema work; then
  a huggingface.js PR, the Model Mapping API (needs a Team/Enterprise Hub plan), a
  per-request nano-USD billing endpoint with `Inference-Id` headers, a Python client PR,
  and a provider docs page. Automated validation every 6 h covers TTFT < 5 s streaming,
  tool-calling, and structured output — all behaviors memra already gates. Default
  provider ordering on model pages is 7-day routed volume, so the listing compounds with
  traffic, not marketing.
- **OpenRouter** — application material exists
  (`research/or-application-20260805/APPLICATION.md`). Expect a backlog; open-weight
  models are non-prioritized. Treat as **distribution and public perf receipts, not
  revenue**.
- **GitHub / crates.io** — the memra repo is the lab's public record and the strongest
  trust asset. See §6 for the crates caveat.

---

## 10. Numbers the earlier specs got wrong (correction ledger)

Kept here so a stale claim is recognizable if it resurfaces.

| Old claim | Correction |
|---|---|
| serve spec 170.6 tok/s | N=5 median is **170.5**; 170.62 was the r4 single rep |
| c=8 aggregate 421 tok/s | N=3 median is **420.6**; 421.9 is c=16 |
| "2x5090 trajectory / the 5090 is the deployment target" | **RTX PRO 6000 owned trajectory** (override 2026-08-03); 5090 = rental + shape reference |
| "we run on PRO 6000" | **rented pods only**; nothing in that class is owned |
| serve path implied at parity with naked | serve c=1 is **−11.74%** (Q8_0) / **−8.66%** (NVFP4 spec); filed as PHASE2-SPEC H1/H3, task **#70** |
| "#71" as the serve-gap task | wrong task; #71 is session-id affinity — **merged 2026-08-05** (`70ce5a0f`), now a shipped feature in §6 |
| session affinity "makes memra 24x faster" | **turn class is load-bearing**: 20-24x on rewritten-history turns, **1.01x** on turn 0, ~1.00x on pure-extension turns; the claim is flatness (§6) |
| interactive-latency superiority over llama | **llama currently wins cold TTFT 0.19 vs 0.53 s**, short-agentic decode, and 4k prefill |
| 2.6x official-FP8 on the PRO 6000 | real (2.61x) but on **`rig2x5090-serve`**; no official-FP8 cell exists on any PRO 6000 |
| "run-spec K=1..8" on the prod PRO 6000 | prod ran K=1..3 gate + K=1..5 perf; the K=1..8 battery is on the community board, MTP artifact only |
| bare "byte-identical" / "token-identical to the CLI oracle" | scope to **alone vs inside a full batch**, serve-vs-serve, at defaults |
| "the exact FP8 arm ships" | **Q8_0 GGUF ships**; both exact FP8 arms are default OFF; the fold is non-shippable |
| "all 9 crates live at 0.69.0" | first publish landed **5/9** before a 429; resumable workflow landed; verify the registry |
| 464,870 deterministic requests | load was temp-0.7 seeded; the determinism hash is a **separate greedy probe** pre/post |
| lanes take p95 7.15 → 3.69 s | **half of that is the proxy cap** (7.15 → 4.34 s alone); the lane gate does 4.34 → 3.69 s |
| constrained decoding at 99.4% | **plain lane only**; the spec lane runs at 79% |
| "darklanes" as the lab/research entity | **darklanes = serving product only**; the lab is unnamed and owner-pending |

---

## 11. Open owner decisions (nothing below ships without a call)

1. **The lab name** — first-order, blocks both sites. See the decision brief in
   `research/darklanes-website-spec-20260804/SPEC.md` §5.
2. **Domain purchases** — nothing bought. Availability checked 2026-08-05 by RDAP:
   `darklanes.ai`/`.com`/`.dev` **available**; `memra.ai` **taken** (registered
   2025-08-20), `memra.dev` **taken** (2025-08-22), `memra.com` **taken** (2003);
   `memra.io`, `memralab.ai`, `memralabs.ai`, `memralabs.com` **available**.
3. **Pricing numbers** — all four lanes plus the free-credit amount (§8).
4. **The 8-bit posture for 3.8 day one** — Q8_0 bridge vs FP8-ST direct (§4, conflict 2).
5. **SKU scope** — retire or rescope the 35B-A3B / Step-3.7-Flash line (§4, conflict 1).
6. **Data-retention policy text** — a prerequisite for both gateway listings.
7. **Legal/invoicing entity** shown in the footer.
8. **Launch timing** relative to the Qwen3.8 drop.

---

## 12. Maintenance

This file is reconciled, not appended. When a lane moves a product number:

1. Update the row here, with receipt path + date + rig + protocol.
2. If the number is also a tracked board cell, follow the perf-board rule in CLAUDE.md
   (`current-board.json` → `tools/update-perf-board.py` → commit together).
3. If a claim became false, move it to §10 rather than deleting it — a recognizable stale
   claim is cheaper than a silent one.
4. Same commit. Not a follow-up.

### 12.1 What the 2026-08-05 reconciliation changed, by file

The lane that created this file also corrected the surfaces that had gone stale. Recorded
here so a future reader can tell "already reconciled" from "never checked."

| File | Commit | What changed |
|---|---|---|
| `docs/PRODUCT-TRUTH.md` (new) | `96fa6701` | this file |
| `research/darklanes-website-spec-20260804/SPEC.md` | `a47b07fa` | claim-by-claim corrections, §2a wording rules, §5A lab-name brief, §7.0 two-surface split + content mapping, §16 CHANGELOG for the build-agent |
| `.../EVIDENCE-REPO.md`, `.../BLOG-EVIDENCE.md` | `881ccf31` | rig labels, N values, pre-fix labels, refuted-explanation warnings, honest counterweights per post |
| `CLAUDE.md` | `e9b2c2d0` | the product-claims rule (this file is the only source; same-commit obligation) |
| `README.md`, `docs/SERVING.md`, `docs/RELEASING.md` | `dbe12cb3` | exactness/isolation wording scoped, FP8 rig-labeled, constrained 99.4% scoped to the plain lane, serve-gap + cold-TTFT gaps added, 5/9-crates publish history |
| `docs/PERFORMANCE.md`, `tools/update-perf-board.py` | `8628bbc2` | llama-freeze + rig-label banners, "Standing" refreshed, spine heading retitled, serve-gap section, generated footers carry the frozen-reference sentence |
| lab-name genericization (11 hits) + 5090-target rescope (6 files) | `6c2a7e73` | see the commit body |
| session-affinity reconciliation: this file (§6, §7.2, §10), `docs/FLAGS.md`, SPEC §2a/§7.1/§10/§16, EVIDENCE-REPO, BLOG-EVIDENCE | (this commit) | task #71 merged into the lane *after* the pass began — new capability + its turn-class wording rules; also fixed the last stale serve-st-gate item-4 wording in FLAGS.md |

Not touched by design: `research/hw-growth-rethink-20260803/ASSESSMENT.md` and
`research/hw-buy-20260802/REPORT.md` keep their pre-override recommendations (append-only
records — §3 explains how to read them), and every `PERF-*` marker block stays generated.
