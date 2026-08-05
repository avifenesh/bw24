# Blog-post source material — the receipts behind each candidate post (2026-08-05)

> **Corrected 2026-08-05.** Numbers and framings below were re-read against the raw logs.
> **`docs/PRODUCT-TRUTH.md` is the source of truth**; if it and this file disagree, this
> file is stale. Two structural notes: (1) **the blog lives on Surface A, the lab site**
> (SPEC.md §7.0) — these are the *lab's* posts, bylined to the founder, crediting the engine
> as **memra**, never as darklanes; (2) every number in a post carries its **rig label** and
> its **N**, and every post states its honest counterweight in the same breath as its win.

Companion to SPEC.md §10. Every post below is written from material that already exists in
the memra repo — the writer's job is narrative, not new evidence. Paths relative to repo
root (branch restructure/public-split).

## A. The nvcc miscompile catch (candidate: LAUNCH POST)

Commit `93e1b6a1` (2026-08-04); write-up `research/fp8ship-20260804/official/RESULTS.md`
("FINDING FIRST — nvcc 13.0.88 miscompiles the ARM B' dequant kernel").

- nvcc **13.0.88** at -O3/sm_120a drops the first of two adjacent byte stores
  (`dst[0]=lo; dst[1]=hi;` under `if (lane==0)`) in the FP8 block-128 dequant kernel —
  the LOW byte of the Q8_0 `d` f16 scale zeroed. The same commit was green on nvcc 13.1.
- Caught by the `kernel-check` `[fp8-blk-gpu]` bit-parity arm on FIRST run (3968/680/8 bad
  bytes on three shapes); byte-in-block histogram localized every bad byte to offset 0.
- Real-world consequence had it shipped: first-token logits max_abs diff 3.04, greedy
  stream diverged at token 14. "The bit-identity gate did exactly its job."
- 40-line isolated repro committed: `research/fp8ship-20260804/official/miscompile/
  halfstore_repro.cu` (8/8 blocks fail) + `halfstore_fix.cu` (0/8). Fix: one aligned u16
  store in `crates/memra-engine/cu/fp8_blk_dequant.cu`.
- The law extracted: "parity gates re-run per TOOLCHAIN, not per commit."
- Related compiler lore for the post's depth: ptxas C7514/15/17/19 wgmma
  form-sensitivity findings pinned in ARCHITECTURE-H100.md:446, 1372-1398; FLAGS.md:433.
- **Honesty note the post MUST carry:** the original failing logs were lost to
  `rsync --delete`. The committed 40-line repro survives (8/8 fail pre-fix, 0/8 post-fix)
  and is what the post should demonstrate from. Do not present reconstructed output as the
  original logs — on a verify-it-yourself brand that is the one unrecoverable mistake.

## B. Dogfood — the founder's own agent found THREE serving bugs (2026-08-04/05)

**Post scope corrected: three bugs, not two.** F4 commits 8032ab01 → 6f51d4a1 → 95df4f3c →
merged c716954b (receipts `research/sampledspec-20260804/`); the third fixed in `d1dc79b8`,
merged `44c4c6a4`, gate `9bbd3cca` (receipts `research/sampfix-20260805/`).

- Bug 1: `#[serde(default)] temperature: f32` = 0.0, and `is_greedy()` is
  `temperature <= 0.0` ⇒ every client omitting temperature got greedy argmax instead of
  OpenAI's documented 1.0. Live symptom: the owner's daily agent (sends no temperature)
  locked into ~10 identical tool-call cycles.
- Bug 2: fixing temperature was NOT enough — `#[serde(default)] seed: u64` pinned omitted
  seed to 0, a valid FIXED seed; the loop survived. Differential proof on the pre-fix
  binary is in the commit body (4/4 byte-identical with omitted params; distinct with
  explicit seeds).
- **Bug 3 (2026-08-05) — the server was injecting `!` into output.** `top_p`/`min_p`
  truncation in the sampled-spec **full-accept** path read its filter stats from
  `col_stats.last()` — a *neighbouring* column — so a foreign `row_max` mis-scaled
  `e0 = exp((x - row_max)/T)`, every id failed the threshold, the row masked to -3.4e38, and
  the argmax fell through to the smallest-index tie-break: **token id 0 = `!`**. Wild
  symptoms: `!bash`, `grep -!q`, `/!etc/hosts`. Fragility scaled with the threshold —
  `min_p 0.05` hit a **100% id-0 rate**, plain `top_k 40` was clean, and **memra's own
  default sets no truncation filter and is structurally immune**, which is exactly why the
  owner's daily never showed it: *it only bit clients configured like other engines.*
  Isolation matrix: `research/memra-vs-llama-daily-20260805/logs/posthoc-lsampler.txt`.
- Fixes: default_temperature=1.0; `seed: Option<u64>` + `fresh_seed()` (nanosecond clock
  x atomic counter through SplitMix64 finalizer, never 0); the full-accept path takes its
  own column's stats. Explicit temp 0 / seed 0 honored exactly — every determinism gate
  sends both explicitly.
- The one-line thesis for all three: **a meaningful zero is not "unset."**
- The meta-lesson that makes it a post: **every golden in the repo ran temp=0, which
  routes around the sampler chain entirely — a broken sampled-spec path is INVISIBLE to
  argmax goldens** (sample_check.rs top-of-file doc). New gate: the sampled-spec
  **composition** arm (composed accept-walk output distribution vs target p, 20k draws,
  L-inf 0.012 / TV 0.05). Its negative controls fail as designed — "forgot the residual"
  trips TV 0.0881 **with acceptance unchanged**, i.e. a bug every isolation arm would have
  passed. The `!` bug got its own differential serve-smoke matrix, **proven in both
  directions**: 3 failures on the pre-fix binary, 0 on the post-fix.
- Verification battery on the owner's exact config: ALL PASS; sampled-spec 73.92 tok/s
  (1.72x plain-sampled), 266 spec bursts, acceptance 0.59.
- **Required framing: all three are FIXED, and the post says so — while also saying the
  bugs shipped for a window and a greedy-only battery was structurally blind to them.**
  That combination is what makes it a credibility asset rather than a confession.

## C. "We bit-audit every kernel" — the exactness discipline

- README: "Exactness is the contract... speed never changes what the model says."
  "A MISMATCH voids every number after it."
- CONTRIBUTING.md:38: "A kernel that reduces in different floating-point order can flip
  an argmax at tight logit margins — has silently broken 'faster' kernels before."
- The concat-prime-exact defect (merge b3a5465f, receipts
  `research/concat-prime-exact-20260802/findings.jsonl`, `kind: route_trace`): the
  m-dependent F32 router GEMV changed EXPERT SELECTION by co-arrival — **121/760 (15.9%) of
  (layer,token) pairs picked a different expert *set*** depending on who arrived in the same
  batch, plus 217 more differing in order only. Model **Ornith-1.0-35B Q4_K_M at total_m=75,
  arm `exact0` = PRE-FIX**. Mechanism isolated, not guessed: the router cuBLASLt GEMM is
  m-dependent (maxdiff 0.0039), the in-house `router_gemv` is m-invariant (0.0), and 36
  trunk weights are m-invariant — pinning the defect to two router GEMMs. Fixed with
  m-invariant router twins; post-fix gate **16/16 on all four models**. New contract: greedy
  serving is isolated-identical under concurrent load at defaults.
  **Two precision requirements: label the 16% PRE-FIX every time, and do NOT reuse the
  "post-trains have tighter margins" explanation — it was tested and REFUTED in the same
  receipt.**
- **Claim-wording caution for this post specifically:** the contract's object is
  **serve-vs-serve, c=1 vs c=16, same server, at defaults**. Do not let the post drift into
  bare "byte-identical" or "token-identical to the CLI oracle" — the batched-plain path is
  confirmed *non*-identical to a tokenwise oracle (accepted FP near-tie class,
  `research/plainbatch-20260804/RESULTS.md`), and a reader could demonstrate the
  counterexample in minutes. SPEC.md §2a has the exact permitted forms.
- Honest-limits section for credibility: first-token cross-config drift — **10 of 144
  (6.9%)** across three numeric prime configurations, **every flip at a top1-top2 margin
  ≤ 0.70**, determinism control 144/144 bit-identical, dense-Q8_0 fleet class 0/48, escape
  `MEMRA_PRIME_TOKENWISE=1` (docs/SERVING.md). **Keep this class distinct from the
  isolation contract** — different mechanism, different object; conflating them in a post
  about determinism would be the worst place to do it.
- Post argument: determinism is not a benchmark aesthetic — it is the property that makes
  evals reproducible, agent loops debuggable, and caching auditable.

## D. The QoS lane model (the product-namesake post)

- Mechanism (crates/memra-lanes/src/lib.rs): interactive/judge/harvest; shed at
  ADMISSION, never inside the engine ("the engine's own queue is where the tail dies");
  interactive never preempted; per-lane prefill budgets replace vLLM's single global
  chunked-prefill knob (the knob that taxed baseline p99 11.6 → 40.6 ms at zero parasite
  load).
- Numbers (`research/qos-p95-20260802/RESULTS.md`; 8 replicas, Qwen3.5-9B-Q8_0 on rented
  H100s, 4 conditions interleaved, N=3 passes, full teardown/bring-up per cell): a c=96 bulk
  tenant inflates interactive p95 **4.2x** (1.74 → 7.33 s) on a lane-blind fleet; lanes on →
  **3.690 s** at −11% bulk; SLO dial 25 ms → contended p50 1.637 / p95 **2.158 s**,
  statistically equal to uncontended (1.635/2.065), bulk pays **−67%**.
- **THE ATTRIBUTION THIS POST MUST MAKE — and it makes the post better, not weaker.**
  Raising the proxy cap from 8 to 16 *by itself* moves p95 **7.15 → 4.335 s** (that control
  cell is in the same RESULTS.md); the lane gate moves 4.34 → 3.69 s. **Roughly half the
  headline improvement is the queue, not the gate.** RESULTS.md's own sentence is the better
  thesis: *"the engine gate cannot fix a queue it never sees."* A post that only claims
  7.15 → 3.69 s for lanes is overclaiming and is refutable from our own published log.
- **Harvest is a price TIER on the same endpoint, selected by the `x-lane` header — not a
  second product.** The post's pricing consequence ("batch is cheaper because it yields")
  holds either way; the framing must not imply a separate pipeline.
- Endurance (`research/fleet-endurance-20260803/SUMMARY.txt`): 140 min, **464,870 requests,
  0 errors, 0 sheds**, +0.045% throughput drift, p95 drift −0.4 ms.
  **Required wording:** the load ran at **temperature 0.7, seeded, on prefix-cache hits, on a
  9B-class model across 8 RENTED H100s** — the identical hash `56b8502cfb8de57a` is a
  **separate greedy probe** before and after. Write *"a greedy probe hashed identical on all
  8 replicas before and after the soak."* **Never** "464,870 deterministic requests."
- Companion chaos receipt worth a paragraph: `research/fleet-v060-20260801/SUMMARY.md` —
  SIGKILL a replica mid-load, breaker DOWN same second, restart +2 s, backend UP +9 s, 8/768
  requests lost (exactly the victim's in-flight cap), greedy hash identical on all 6
  replicas 18/18.
- Post argument: mixed workloads on shared GPUs are the actual economics of small fleets;
  hard p95 for interactive + a harvest tier for batch = the price/latency trade made
  explicit and receipted — including which half of the win came from which mechanism.

## E. Spec-decode economics on consumer-priced silicon

- `research/pro6000-prod-20260804/` — **rig `pro6000wk-runpod`: RTX PRO 6000 Blackwell
  Workstation 96 GB, 188 SM, 600 W, zero throttle, RENTED pod.** 27B-class NVFP4+MTP spec
  **186.7 tok/s bare** (N=5 median; **2.17x** the same-run plain 86.20) / **170.5 serve c=1**
  (N=5 median); **420.6 agg at c=8**; TTFT **0.182 s cold / 0.003 s warm**; Q8RP +57% at
  c16/32 on 96 GB.
  **CORRECTED: was "170.6 serve" and "421 agg".** 170.62 and 421.18 are single reps.
  **And c=8 is the knee, not a ceiling** — c16/c32 are flat (421.9 / 423.0) while p50 doubles
  per step (2.43 → 4.84 → 9.67 s): "queueing, not throughput."
- FP8 official checkpoint: embedded-MTP spec **128.06 tok/s (2.61x plain 48.99)**, out of the
  box from the checkpoint's own mtp.safetensors; +own-trim drafter 136.75; load wall
  843.9 → 291.6 s = 2.89x; prefill logits bit-identical 993280/993280 bytes
  (`research/fp8ship-20260804/official/`).
  **Rig is `rig2x5090-serve` (rented vast 2x RTX 5090) — NOT the PRO 6000. There is no
  official-FP8 cell on any PRO 6000 board; do not merge the two.** Honest counterweight the
  post must state: spec **triples TTFT** on this arm (0.170 → 0.466 s), and the
  e4m3-resident arm is flat *by construction* — the FP8 win is load time, not tok/s.
- **Framing correction: no llama.cpp comparison in this post.** The 5090 reference ratios
  (9B MTP spec 2.30/1.74/1.59x) are **frozen** reference points recorded through 2026-08-03,
  when llama benching stopped; and as of 2026-08-05 llama is *ahead* on cold TTFT (0.19 s vs
  0.53 s), short agentic turns, and raw prefill (PRODUCT-TRUTH §7.2). Use self-competition:
  2.17x over our own plain decode on the SKU, 2.61x on the official FP8 checkpoint. Those are
  bigger numbers anyway and they are ours to keep.
- Economics frame: workstation Blackwell $/token vs datacenter cards
  (`research/hw-buy-20260802/`: 2x5090 $0.47/Mtok; H100 TCO-negative on the small SKU) —
  **these are internal estimates; cite as estimates or not at all.** Hardware honesty for the
  post: every number above came off **rented** pods; the only owned GPU is an RTX 5090
  **Laptop**; the owned build-out targets **RTX PRO 6000 class, homogeneous** (the 2x5090
  purchase path was explicitly rejected on scaling-continuity grounds).
- Post argument: single-stream speed on workstation-priced silicon changes which workloads
  are economical to serve at hard-guarantee latency.

## F. Launch-announcement material (Qwen3.8 day-one)

- `docs/qwen38-bringup-runbook.md`: release-watch → published-as-supported in one day if
  the arch-diff is clean; two legs (FP8-ST exact arm = prod direction; GGUF NVFP4+MTP =
  spec leg); deployment bar **≥1.1x end-to-end** before "published as supported."
- **State the bar as ours, not as a live comparison.** Its llama denominator is frozen at
  2026-08-03; write "our ≥1.1x end-to-end deployment bar."
- **What is genuinely unknown and must not be implied:** the 3.8-27B architecture, license,
  active parameter count, and benchmarks are all unpublished. Release timing is **"week of
  2026-08-10"** from Alibaba's own 2026-08-03 post, and as of 2026-08-05 it is **not
  released**. Same-architecture is an *expectation*; the runbook's arch-diff step is the
  verification (`research/qwen38-prep-20260803/WATCH.md`).
- **Which format day one runs is an OPEN OWNER CALL** — Q8_0 bridge vs FP8-ST direct
  (PRODUCT-TRUTH §4, conflict 2). The post cannot name a format before that call.
- The launch post can only be written AFTER the drop lands and gates pass — the site
  ships with posts A–E ready and F slotted. Cleared advance phrasing: *"a day-one bring-up
  runbook is written and ready ahead of the expected release."* Never "day-one support
  guaranteed"; the word "supported" is earned only after the bar clears.
