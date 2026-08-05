# Repo evidence inventory — the receipts the website is allowed to cite (2026-08-05)

> **Corrected 2026-08-05.** Numbers below were re-read against the raw logs and several
> were off — see the corrections inline. **`docs/PRODUCT-TRUTH.md` is the source of truth**;
> this file is the *index* of where the receipts live. If the two disagree,
> PRODUCT-TRUTH wins and this file is stale. Every claim also needs its **rig label** and
> its **N** — the same number is wrong by 5-12% on a different board, which is exactly how
> a build-agent shipped the wrong product from an earlier revision of this directory.

Every marketing claim must trace to one of these. Paths are relative to the memra repo root
(bw24-unified, branch restructure/public-split as of 623ce27e). Rule inherited from the
repo's evidence discipline: **a claim whose raw runs exist nowhere in the repo is not
evidence** — no surface states a number the repo cannot back.

## 1. First-SKU serving numbers (Qwen 27B class on RTX PRO 6000 Blackwell, 96GB)

Source: `research/pro6000-prod-20260804/` (anchor/, serve/, levers/, q8rp/ raw logs +
JSONL; commit 623ce27e). Gates ALL GREEN on the 5th distinct GB202 die.

**Rig label, mandatory on every citation: `pro6000wk-runpod` — RTX PRO 6000 Blackwell
Workstation 96 GB, 188 SM, 600 W, clocks pinned 2865 MHz, zero throttle, RENTED pod.**
Never cite the `pro6000wk-runpod-community` dev pod alongside these (it runs 5-11% slower),
and never imply owned hardware.

Two artifacts, both Qwen3.6-27B, interleaved in one session: `nv` =
`Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf` (15.7 GB), `q8` = `Qwen3.6-27B-Q8_0.gguf` (28.6 GB).

- Plain decode tg128 (N=5 medians): NVFP4+MTP arm 86.8 tok/s, Q8_0 arm 52.6 tok/s.
- Spec decode (MTP), nv K=3: **186.7 tok/s bare CLI** (N=5 median; 2.17x the same-run plain
  86.20) and **170.5 tok/s through the serve surface** at c=1 (N=5 median).
  **CORRECTED: was "170.6 … serve tax −12.6%".** 170.62 is the r4 single rep; the median is
  170.55. And the serve/bare delta on this arm is **−8.66%**, not −12.6% — the −11.74%
  figure is a *different* measurement (Q8_0, community board; see §7 note below).
  **The serve gap is an open engineering gap, not a marketing number** — see
  PRODUCT-TRUTH §7.1 (root cause known: the worker routes B=1 through `decode_step_batch`,
  which has no CUDA-graph door and misses the m=1 dispatch fusion; filed as task **#70**).
  Q8 K=4 143.5 (reproduces the vast desktop 137.4 ladder digit-for-digit: acceptance
  77.8/75.0/71.4).
- Batched serving: **the knee is c=8 at 420.6 tok/s aggregate** (nv arm, N=3 median).
  **CORRECTED: was "421 tok/s".** 421.18 is a single pass. Do not publish c=16/c=32 as a
  ceiling — 420.6 → 421.9 → 423.0 is flat while p50 doubles at every step (2.43 → 4.84 →
  9.67 s); the journal's own word for it is "queueing, not throughput."
- TTFT: **cold 0.182 s / warm 0.003 s** (61x apart). Cold = fresh `cache_salt` per rep,
  median of reps 2-5 (rep 1 = 0.340 s, excluded as one-time session warmup). Warm =
  prefix-cache hit. **Always separate cells** — an unsalted repeat request hits the cache, so
  a TTFT without a salt is a warm number wearing a cold label.
- Prefill pp512: 4118 (nv) / 4591 (q8) tok/s — **two different artifacts**, not two configs
  of one. Fine side by side; not fine to headline 4118 while implying the same weights.
- Q8RP 96GB lever: +57% at c16/32 (486 vs 310 agg tok/s, p50 6.61→4.21 s), 63.7 GB resident.
- 5090 reference boards: README PERF-SAMPLES (2026-08-02): Qwen3.6-35B-A3B plain decode
  1.13x llama.cpp; 9B MTP spec 2.30/1.74/1.59x by prompt class. Full boards
  `docs/PERFORMANCE.md`, raw log `research/tune-data/rig5090.jsonl`.
  **Posture note: these llama ratios are FROZEN reference points recorded through
  2026-08-03, when llama benching stopped.** They are regression anchors, not a live
  scoreboard, and they must not be published as a current competitive claim — a
  same-artifact head-to-head on 2026-08-05 has llama *ahead* on cold TTFT and prefill
  (PRODUCT-TRUTH §7.2).
- **Official FP8 checkpoint, 2.61x spec — a different rig.** `Qwen/Qwen3.6-27B-FP8`
  safetensors: ST plain 48.99 → **ST spec 128.06 tok/s on the checkpoint's own embedded MTP
  head** (2.61x), +own-trim drafter 136.75; load wall 843.9 → 291.6 s = 2.89x. Prefill
  logits **bit-identical 993280/993280 bytes**. Receipt
  `research/fp8ship-20260804/official/RESULTS.md`. **Rig is `rig2x5090-serve` (rented vast
  2x RTX 5090), NOT the PRO 6000 — there is no official-FP8 cell on any PRO 6000 board.**
  Honest counterweight: spec **triples** TTFT here (0.170 → 0.466 s), and the e4m3-resident
  arm is flat *by construction* (every tensor is block-128 and falls through to the Q8_0
  path) — the win is load time, not tok/s.
- **No q27-at-8-bit row exists in the published board yet.** The Q8_0 absolutes measured so
  far came off the community pod, whose own RESULTS.md says every absolute gets re-minted on
  prod-class silicon first. Publish 52.6 (prod q8 plain) and relative deltas; never the
  community absolutes 49.82 / 52.22.

## 2. Exactness discipline (the core differentiator)

- README.md (public repo): "Exactness is the contract: speculative, graph-replay, and
  batched serving output is gated token-identical to plain decode — speed never changes
  what the model says." **Care with the second half of that sentence on a public surface:**
  the *batched-plain* path is confirmed **not** token-identical to a tokenwise oracle
  (accepted FP near-tie class, 0.19-0.81 independent flips per 100 tokens, all gaps
  sub-0.2, n-stable — `research/plainbatch-20260804/RESULTS.md`). The gated, defensible
  object is **serve-vs-serve, c=1 vs c=16, at defaults**. See the wording rules in
  SPEC.md §2a.
- Three standing gates (CLAUDE.md, CONTRIBUTING.md, docs/TESTING.md): `kernel-check`
  (every kernel vs CPU reference, ALL GREEN), `run-gen` argmax gate (printed MATCH before
  any generation; "a MISMATCH voids every number after it"), `run-spec` K=1..8
  self-consistency. **Attribution correction:** the K=1..8 PASS battery runs on the
  **MTP-capable artifact** (`research/q27-deepdive-20260805/logs/gate-key48-runspec-K1to8.log`,
  community board); the prod PRO 6000 board ran K=1..3 as its gate plus K=4/5 as perf cells,
  and Q8_0 cannot run it at all (no MTP head, `RUNSPEC-Q8 rc=2`). Public form: "K=1..8
  self-consistency is a standing gate, run on the MTP-capable artifact."
- Serve-surface isolation contract (docs/SERVING.md §"The isolation contract"): greedy
  output is byte-identical **alone vs inside a full batch** — gated by replaying the same
  prompts at c=1 and c=16 **against the same running server** and byte-comparing every
  stream. **PASS 16/16 on all four models at defaults**; with
  `MEMRA_ROUTER_PREFILL_EXACT=0` (a numeric A/B seam, never a serving config) it fails
  7/16 and 6/16, which is how the defect was found. Driver
  `research/concat-prime-exact-20260802/run-serve-gate.sh`, comparator
  `tools/check-batch-exact.py`, receipt `serve-gate-matrix.jsonl`.
  The m-dependent MoE router defect this gate caught: **121/760 = 15.9% of (layer,token)
  pairs activated a different expert *set* depending on who arrived in the same batch**
  (plus 217 more differing in order only), Ornith-1.0-35B Q4_K_M at total_m=75, arm
  `exact0` = **PRE-FIX**. Mechanism isolated, not guessed: the router cuBLASLt GEMM is
  m-dependent (maxdiff 0.0039), the in-house `router_gemv` is m-invariant (0.0), 36 trunk
  weights m-invariant. Fixed with m-invariant router twins. **Two care notes:** always label
  the 16% pre-fix, and do **not** reuse the "post-trains have tighter margins" story — it
  was tested and **REFUTED** in the same receipt (`findings.jsonl`, `kind: route_trace`).
- Prefix-cache exactness: cached hit is bit-identical to the run that computed the prefix,
  gated 16/16 + 16/16 (`research/prompt-cache-20260802/gate-exact.jsonl`).
- Constrained decoding exactness: device-mask greedy byte-identical to host oracle;
  spec-constrained byte-identical to plain-constrained; draft-mask ON byte-identical to
  OFF across 7 cells (`research/constrained-full-20260803/`, `research/draft-mask-20260804/`).
- Honest limits stated in-repo (the site should link, not hide): first-token cross-config
  drift on near-tie prompts under batched prime — **10 of 144 first tokens flip (6.9%)
  across three distinct numeric prime configurations, and every flip sits at a top1-top2
  margin ≤ 0.70**; determinism control 144/144 bit-identical; the dense-Q8_0 fleet class
  flipped 0/48. Escape hatch `MEMRA_PRIME_TOKENWISE=1`. Receipt
  `research/prime-gate-coverage-20260802/RESULTS.md`, docs/SERVING.md §"First-token
  cross-config drift". **This is a first-token, cross-config class — not the isolation
  contract above. Do not conflate the two in copy.**
- **The full honest-gaps list is PRODUCT-TRUTH §7 and it is required content**, not
  optional garnish: the serve-vs-naked gap (§7.1), the llama cold-TTFT loss (§7.2), this
  drift class (§7.3), the batched-plain near-tie class (§7.4), the fixed sampler-bug class
  (§7.5), and the does-not-exist-yet list (§7.6 — no SOC2, no multi-region, no tensor
  parallelism, no power-curve data on a healthy PRO 6000, no q27-8bit board row).

## 3. The lanes QoS story (the product namesake)

- Mechanism: `crates/memra-lanes/src/lib.rs` — three lanes (interactive / judge / harvest),
  shed at ADMISSION never inside the engine ("the engine's own queue is where the tail
  dies"), interactive never preempted, per-lane prefill budgets per tick. `x-lane` request
  header; naked traffic = interactive and byte-identical.
- Measured: `research/qos-p95-20260802/` — 8 replicas (devices 4-7, 2/GPU),
  Qwen3.5-9B-Q8_0, c=96 harvest + c=4 interactive, **4 conditions interleaved, N=3 passes,
  full teardown/bring-up per cell** (12 bring-ups), one thermal window, 0 request errors:
  lane-blind FIFO at proxy cap 8 inflates interactive p95 to **7.150 s** (~4.2x alone);
  lanes on at cap 16 → **3.690 s** at −11% bulk; SLO dial at 25 ms makes contended
  interactive statistically equal to alone (p50 1.637 / p95 **2.158 s** vs 1.635/2.065) with
  bulk paying **−67%**. `MEMRA_SLO_P99_MS` is the knob (FLAGS.md §1).
  **MANDATORY ATTRIBUTION — do not cite the 7.15 → 3.69 s move as the lane mechanism.**
  Raising the proxy cap from 8 to 16 *by itself* moves p95 7.15 → **4.335 s** (that's the
  control cell); the lane gate does 4.34 → 3.69 s. **Roughly half the improvement is the
  queue, not the gate.** RESULTS.md's own line: "the engine gate cannot fix a queue it
  never sees." The honest — and sharper — framing is in PRODUCT-TRUTH §2.5.
- Endurance receipt: `research/fleet-endurance-20260803/SUMMARY.txt` + `load-windows.jsonl`
  + `greedy-hashes.txt` — **140 min sustained load** (70 back-to-back 120 s windows,
  concurrency 96), **8x H100 80 GB (RENTED), 1 replica/GPU, proxy cap 16**, model
  **Qwen3.5-9B-Q8_0 — not the 27B SKU**: **464,870 requests, 0 errors, 0 sheds**, 0
  supervisor restarts, throughput drift **+0.045%** (7000.2 → 7003.3 tok/s), p95 drift
  −0.4 ms, worst single request 1.861 s, RSS plateau (max +3064 KB/replica), 59,461,500
  completion tokens.
  **Three honesty requirements, all mandatory:** (1) **the load ran at temperature 0.7,
  seeded — not greedy.** The identical hash `56b8502cfb8de57a` (16/16 lines = 8 replicas ×
  pre/post) is a **separate greedy probe** before and after the soak. Correct phrasing: "a
  greedy probe hashed identical on all 8 replicas before and after the soak." **Never**
  "464,870 deterministic requests." (2) The prompts were **prefix-cache hits** (226/226
  prompt tokens from cache) — this is a warm-prefix throughput number. (3) Say **"9B-class
  fleet endurance"** — it is not the 27B SKU and not owned hardware. Minor: the proxy
  reports 464,940 (+70, one per window, likely the health probe); cite 464,870 and the
  harness.
- Fleet chaos receipt: `research/fleet-v060-20260801/SUMMARY.md` — SIGKILL a replica
  mid-load: breaker DOWN same second, restart +2 s, backend UP +9 s, 8/768 requests lost
  (exactly the victim's in-flight cap), greedy hash identical on all 6 replicas 18/18.

## 4. Serve-surface features (product claims)

All in docs/SERVING.md with receipts:

- OpenAI-compatible `/v1/chat/completions` + `/v1/completions`, validated against the
  official `openai` Python SDK (`research/serve-compat-20260802/`): envelope (id/created/
  system_fingerprint/x-request-id), streaming with SSE keep-alives, reasoning separation
  (OpenRouter dialect), honest 400s on params it cannot honor (never silent ignoring of
  semantic params).
- Tools/function calling: `tools`, `tool_choice` auto/none, streaming tool_calls deltas,
  malformed-block policy = surface verbatim, never dropped (`research/serve-tools-20260802/`).
- REAL constrained decoding: `response_format` json_object/json_schema via llguidance,
  on-device masking at **99.4% of unconstrained speed on the PLAIN lane** (123.7 vs 124.4
  tok/s, local RTX 5090, Qwen3.5-9B NVFP4, 256-token greedy, N=3 interleaved same session);
  draft-side masking makes proposals legal by construction
  (`research/constrained-20260803/`, `-full-20260803/`, `research/draft-mask-20260804/`).
  **CORRECTED: never quote 99.4% for the spec path — the spec lane runs at 79%** (153.4 vs
  194.4). Mask cost 0.006-0.007 ms/step. One documented exactness exception: unbounded-schema
  whitespace tail.
- Prompt caching: cross-request prefix cache, LRU, per-tenant isolation via `cache_salt`
  (vLLM-compatible design; the CacheProbe/PROMPTPEEK mitigation), gates in
  `research/pc-iso-20260802/`. Usage carries `prompt_tokens_details.cached_tokens`.
- Honest usage metering: worker-truth `usage` on every shape, disconnect abort billed to
  the abort point, `/metrics` cumulative splits. QoS lanes were extracted from the
  dl-metering lane (FLAGS.md §1).
- Gateway-ready surface: OR-schema `/v1/models` (context_length, architecture, pricing
  stub, honest nulls), truthful rate-limit headers riding the SSE stream, graceful drain
  SIGTERM→drain→exit 0 (`research/serve-tail-20260804/`).
- Sampling honesty (dogfood F4, 2026-08-04): omitted temperature ⇒ 1.0, omitted seed ⇒
  fresh entropy; explicit temp 0 / seed 0 honored exactly. The two bugs this fixed were
  found by the founder's own agent running on the server (commits 6f51d4a1, c716954b);
  receipt `research/sampledspec-20260804/RESULTS.md`.
- **NEW 2026-08-05 — the third bug in the same class, found AND FIXED:** `top_p`/`min_p`
  truncation was injecting token id 0 (`!`) into output (`!bash`, `grep -!q`,
  `/!etc/hosts`). In the sampled-spec full-accept path the filter stats came from
  `col_stats.last()` — a *neighbouring* column — so a foreign `row_max` mis-scaled the
  exponent, every id failed the threshold, the row masked to -3.4e38, and argmax fell to the
  smallest-index tie-break. `min_p 0.05` hit a **100% id-0 rate**; plain `top_k 40` was
  clean; **memra's own default sets no truncation filter and is structurally immune**, so it
  only bit clients configured like other engines. Fixed `d1dc79b8`, merged `44c4c6a4`;
  differential serve-smoke matrix `9bbd3cca` **proven in both directions** (3 failures
  pre-fix, 0 post-fix). Receipts `research/sampfix-20260805/`,
  `research/memra-vs-llama-daily-20260805/logs/posthoc-lsampler.txt`. Publishable as a
  credibility asset **provided** the honest half travels with it: it shipped for a window,
  and a greedy-only gate battery was structurally blind to it.
- **New gate closing that hole:** a distribution-level composition arm (arm 6 of
  `sample_check.rs`) — composed accept-walk output distribution vs target p over 20k draws,
  L-inf 0.012 / TV 0.05. Negative controls fail as designed: "forgot the residual" trips
  TV 0.0881 **with acceptance unchanged**, i.e. a bug every isolation arm would have passed.
- **NEW 2026-08-05 — API keys and tenant auth** (`research/apikeys-20260805/RESULTS.md`):
  keyring TOML with `sha256`/`tenant`/`lane`/`enabled`/`rate_limit`; `--gen-key` /
  `--revoke-key`; 401 unknown, 403 disabled; per-tenant cache namespace
  `t:<tenant>␟<cache_salt>`; a batch-class key on `x-lane: interactive` refused **403**;
  per-key rate limit `min(override, global lane cap)`. Live gate **18/18 PASS**, hot revoke
  ≤2 s, two-tenant cache-hit oracle verified, back-compat serve-smoke 0 failed, 59/59 server
  bin tests. Protocol note: single interleaved run per gate — behavioural pass/fail, not perf
  medians. **The website spec predates this entirely**; the get-a-key flow is real now.
- **Safetensors checkpoint serving:** official Qwen FP8 block-128 dirs load bit-exact, 2.89x
  faster, and spec decode runs out of the box on the checkpoint's embedded MTP head. The
  former dir-checkpoint spec quarantine is **lifted** (`MEMRA_SERVE_SPEC=0` is the rollback
  door).

## 5. Business/market receipts (pricing section inputs)

- `research/or-provider-20260802/REPORT.md`: OpenRouter provider onboarding (application,
  backlog, proprietary-model priority), technical requirements (streaming usage, provider
  /v1/models schema, capacity_tpm), no published take rate (demand-side 5.5%), uptime
  accounting rules, hy3 endpoint price table (floor $0.129/$0.44 per Mtok in/out; cache
  read priced at exactly 25% of input across providers), revenue realism: saturated
  replica ≈ $2–4/hr gross at floor prices — "treat the first listing as distribution and
  public perf receipts, not revenue."
- `research/8bit-decision-20260803/DECISION.md`: 8-bit serving-format verdict — Q8_0 GGUF
  serves now, FP8-ST is the tuning track with a ≥1.1x promotion gate. **OPEN CONFLICT, owner
  call needed:** the owner memory (2026-08-03) hardens this to *"we will finish the st before
  3.8 day one"* — no Q8_0 bridge at all. DECISION.md is the stale half. No Models or Pricing
  page may name a day-one format until this is resolved (PRODUCT-TRUTH §4, conflict 2).
  Same file's line 5 also still calls the 5090 the deployment target — superseded by the
  2026-08-03 PRO 6000 override.
- **OPEN CONFLICT 2:** `research/sku-repick-20260802/REPORT.md` names a *different* launch
  SKU (Qwen3.6-35B-A3B with Step-3.7-Flash as flagship), stale against the q27 story; task
  #53 still tracks Step-3.7-Flash bring-up. Owner call (PRODUCT-TRUTH §4, conflict 1).
- `docs/qwen38-bringup-runbook.md`: the day-one Qwen3.8-27B bring-up plan (release watch,
  two legs: FP8-ST exact-arm prod direction + GGUF NVFP4+MTP spec leg), deployment bar
  **≥1.1x end-to-end** before "published as supported." **Two care notes:** the bar's llama
  denominator is a **frozen** reference (benching stopped 2026-08-03) — state it as "our
  ≥1.1x end-to-end bar", not as a live comparison; and the release is expected **"week of
  2026-08-10"** per Alibaba's own post and is **not out as of 2026-08-05**, with its
  architecture, license, and benchmarks all unpublished. Cleared copy: "a day-one bring-up
  runbook is written and ready ahead of the expected release." Never "day-one support
  guaranteed."
- `research/or-application-20260805/APPLICATION.md`: the OpenRouter application material,
  including the substantiated SKU phrasing *"the Qwen 27B class at 8-bit (Qwen3.6-27B today;
  day-one bring-up runbook for Qwen3.8-27B ready ahead of its expected release)."*
- Memory (owner-confirmed 2026-08-02, operating model): serving success bar = cost coverage
  + public reliability stats, NOT market-share wins; "everything need to be honest, nothing
  is a dream that we try to make true." **Naming note: this memory is filed under a
  "darklanes" label, but the doctrine belongs to the LAB, which is unnamed and
  owner-pending — darklanes is the serving product only. Do not write "darklanes doctrine"
  or "darklanes lab" on any surface** (PRODUCT-TRUTH §1).
- **Hardware, stated exactly** (`research/hw-growth-rethink-20260803/ASSESSMENT.md` OWNER
  OVERRIDE section, `research/hw-buy-20260802/REPORT.md`): the owned trajectory is **RTX PRO
  6000 Blackwell class, homogeneous** — the 2x5090 first-box path was **rejected** on
  scaling-continuity grounds (owner: "buying now 5090 that cant scale later with the 6000 is
  missuse"). Nothing in that class is owned; the only owned GPU is an RTX 5090 **Laptop**.
  2x5090 stays alive as the rental measurement platform. Buy trigger: cumulative metered
  gross margin since the last purchase ≥ 0.5x the next unit's price. **Caution: the pre-
  override 2x5090 recommendation still sits un-struck in §0 of the same assessment file and
  in REPORT.md's "First-box recommendation" — only the appended OWNER OVERRIDE supersedes
  them.** The `$0.47/Mtok` 2x5090 figure and the H100-TCO-negative conclusion are internal
  estimates; cite carefully or not at all.

## 6. HF Inference Providers channel (distribution)

huggingface.co/docs/inference-providers/register-as-a-provider (fetched 2026-08-05):
partner registration = task API compat (OpenAI-compatible LLM APIs "may skip most" of the
schema work) → huggingface.js PR → Model Mapping API (requires Team/Enterprise Hub plan)
→ billing endpoint (per-request cost in nano-USD, polled every minute, 30-min billing
window, idempotent, `Inference-Id` header on every response incl. streams) → Python client
PR → provider docs page. Automated validation every 6 h: TTFT < 5 s streaming, tool-calling
and structured-output behavioral tests — all surfaces memra already gates. Default
provider ordering on model pages = 7-day routed request volume. A separate lane owns the
mechanics; the site spec only needs the channel to exist in the distribution plan.
