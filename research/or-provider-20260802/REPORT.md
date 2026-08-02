# Listing darklanes on OpenRouter (and comparable channels) — research report

Date: 2026-08-02. No GPU used; all facts from live web sources fetched 2026-08-02 (source dates
noted inline) plus this repo's own docs. Raw API receipts in this directory:
`hy3-endpoints-api-20260802.json` (OpenRouter endpoints API for `tencent/hy3`) and
`models-api-tencent-20260802.json` (models-list entries). Question: what does it actually take
for darklanes (small H100 fleet, OpenAI-compatible memra-server fleet, initially 2–4 replicas of
tencent/hy3) to list on OpenRouter and comparable aggregators in 2026?

---

## 1. OpenRouter provider onboarding

### Process — application, reviewed, with an explicit backlog

- Entry point is a self-serve application form: `openrouter.ai/how-to-list` →
  `openrouter.ai/providers/apply` (fetched 2026-08-02). It is **not** self-serve listing —
  "We review every application … We review applications on a rolling basis. Due to high demand,
  not all providers will be accepted. Priority is given to providers that fill gaps in our
  current network."
- Direct quote on the queue (apply page, 2026-08-02): *"We currently have a large backlog of
  provider applications and are prioritizing providers with **proprietary models**."* darklanes
  serving an open-weight Tencent model is exactly the non-prioritized category — expect a slow
  queue unless we can argue a network gap (price floor, a quant/latency point nobody serves, or
  a proprietary variant).
- Four stages (apply page): (1) submit application (infrastructure, API endpoints, models, data
  policies), (2) technical review (API compat, endpoint reliability, pricing, performance
  "against network standards"), (3) integration with **test traffic** validating latency,
  throughput, error handling, (4) go live. Accepted providers get a **provider dashboard**
  (for-providers doc: "Once onboarded, our team can give you access to it").
- Network scale for context: apply page says "70+ providers … 10M+ developers"; the OpenRouter
  cost-guide blog (published 2026-06-12) says "80+ providers at roughly 100 trillion tokens per
  month". Sacra (updated 2026-05-31) estimates ~$50M annualized OpenRouter revenue as of March
  2026.

### Technical requirements (docs: `openrouter.ai/docs/guides/community/for-providers`, fetched 2026-08-02)

1. **OpenAI-compatible `/chat/completions`** — must support **streaming** and must **return
   usage tokens for both stream and non-stream** requests (apply page, requirement list).
2. **List-models endpoint** (their `/v1/models` provider schema) returning every model to be
   served, with: exact model `id` OpenRouter will call, `pricing` **in USD as strings**,
   context length, max output tokens, supported features, **datacenter locations**. Enumerated
   vocabularies: quantization ∈ {int4, int8, fp4, mxfp4, nvfp4, fp6, fp8, mxfp8, fp16, bf16,
   fp32}; sampling params ∈ {temperature, top_p, top_k, min_p, top_a, frequency_penalty,
   presence_penalty, repetition_penalty, stop, seed, max_tokens, logit_bias}; features ∈
   {tools, json_mode, structured_outputs, logprobs, web_search, reasoning}.
   Optional fields with real routing consequences: `capacity_tpm` (input tokens/min your infra
   can absorb — "OpenRouter's provider monitor auto-applies capacity changes when they appear
   in your `/v1/models` response"), `is_ready` (stage a model hidden / take one offline; new
   models are otherwise **auto-staged, baseline-tested, and auto-unhidden**), `is_free`,
   `deprecation_date` (auto-hides past models), `pricing.overrides` (long-context or peak/
   off-peak pricing, max 2 tiers/windows), `discount_to_user` (fractional discount applied to
   displayed prices — GMICloud runs 0.08 on hy3 today).
3. **Automated payment** — "Support monthly invoicing so OpenRouter can pay for inference
   without manual intervention", or auto top-up (for-providers §2: "auto top up or invoicing").
   Note the direction: OpenRouter is the paying customer; the "payout" is our invoice to them.
4. **Privacy & data policy** — "Have a published privacy policy and clear data retention terms.
   Providers must disclose whether prompts are logged and if data is used for training" (apply
   page). Users can filter us out via `data_collection: deny` and per-request `zdr: true`
   routing if we retain prompts (provider-selection doc, fetched 2026-08-02).

### Commercial terms

- **No provider-side take rate is published.** OpenRouter monetizes the demand side: "We pass
  through the pricing of the underlying model providers without any markup" (FAQ, fetched
  2026-08-02); the platform fee is **5.5% on credit purchases** (min $0.80/tx) and **5% on BYOK**
  usage (TrueFoundry pricing writeup 2026-06-25; OpenRouter's own cost-guide blog 2026-06-12
  confirms 5.5%). Sacra (2026-05-31) models it as "~5% take rate" on inference spend. Net for
  darklanes: we set list prices, and we should expect to be paid ~list price for tokens served,
  via monthly invoice.
- **Payout schedule:** monthly invoicing is the documented norm (apply page). No public
  provider agreement PDF exists; exact terms land in the private onboarding contract.
- **Minimum volumes: none documented.** What exists instead are minimum-data thresholds before
  stats/routing kick in (below).

### Reliability expectations (for-providers §3–5, fetched 2026-08-02)

- **Uptime** = successful ÷ total requests, excluding user errors. Counted against you: 401,
  402, 404, all 5xx, **mid-stream errors**, and "successful" responses with error finish
  reasons. NOT counted: 400, 413; **429 and geographic 403 are tracked separately**.
- **Routing tiers by uptime**: needs **100+ requests** of data first; ≥95% = normal routing;
  80–94% = "Degraded — receives lower priority"; <80% = "Down — only used as fallback".
- **Public performance stats**: TTFT and throughput are tracked and shown on every model page.
  Throughput = output tokens ÷ generation time **including fetch latency and queue time** —
  "any queueing on your end will show up in your throughput metrics." Their explicit advice:
  return **early 429s** instead of queueing; stream tokens immediately; send SSE keep-alive
  comments during long thinking phases or they cancel on fetch timeout and fail over.
- **Auto Exacto** (runs by default on **every tool-calling request**): reorders providers by
  throughput, tool-calling success rate, and internal benchmark accuracy. Deprioritization
  cutoffs: benchmark = fixed baseline (median − 2σ of the model's first ~21 days; **missing
  benchmark data ⇒ deprioritized**), throughput = 1.5σ below live median, tool-success = 2σ
  below live median; thresholds only computed once ≥4 providers. An endpoint needs **100
  general requests per 30-min window and 200 tool-call requests per 2-hour window** to be
  evaluated at all; insufficient-data endpoints sort behind known-good ones. "Consistent rate
  limiting (429s) can reduce the volume of successful requests available for evaluation" —
  i.e., a capacity-limited provider that 429s a lot may never accumulate top-tier standing.
- **Removal/hiding**: models past `deprecation_date` "may be automatically hidden from the
  marketplace"; `is_ready:false` auto-hides an endpoint; the reviewed application itself is the
  main gate. No published SLA in either direction (ofox.ai review, 2026-04-21: OpenRouter
  publishes no formal SLA to its own users either).

### What existing providers' experience shows

- Small/unusual providers do get in and get real volume: Chutes (a Bittensor subnet, i.e. a
  decentralized GPU network) became the **leading provider by usage on OpenRouter** in late
  2025 (Grayscale research, 2025-12-03) and peaked around **42B tokens/day on 2026-02-07**
  before declining to 8–12B/day by March (ownyourmind.ai, 2026-04-10). Parasail, GMICloud,
  AtlasCloud, NextBit-class small hosts all have provider pages. DigitalOcean announced itself
  as a new OpenRouter provider on 2026-06-03 (runtimewire) — the pipeline is active in 2026.
- Community keeps providers honest on quant: the classic r/LocalLLaMA thread "Be careful in
  selecting providers on openrouter" (2025-08-07) and OpenRouter's own response — quantization
  labels on endpoints, the `quantizations` routing filter, and the Auto Exacto benchmark
  program ("our own benchmark analysis found aggressively quantized endpoints that match
  full-precision competitors" — OR blog, 2026-06-12). Undisclosed or "unknown" quant is a
  trust penalty users can filter against.
- OpenRouter enforces upstream provider ToS on users (403s + regional rules — HN
  "Openrouter Going Rogue?", 2026-04-06): expect our published ToS to be enforced as written.

---

## 2. Routing/overflow behavior for a capacity-limited provider

Facts a 2–4-replica provider lives and dies by (provider-selection doc + cost-guide blog +
for-providers doc, all fetched 2026-08-02):

- **Default routing is price-weighted load balancing**: (1) drop providers with significant
  outages in the last 30 seconds; (2) among stable ones, pick weighted by **inverse square of
  price**; (3) rest are fallbacks. Documented worked example: providers at $1/$2/$3 per M —
  the $1 provider is ~9x more likely than the $3 one; an outage-marked provider sorts last.
- **`:floor` (sort by price) locks 100% of that traffic onto the single cheapest endpoint**,
  and `:nitro` onto the fastest; `max_price` hard-filters. So undercutting the floor captures
  the `:floor` segment immediately and shifts the default-routing weight in continuous fashion
  — no review cycle: the provider monitor picks pricing/capacity changes straight from your
  `/v1/models` (for-providers doc). Observed market behavior on hy3: cheapest input price fell
  **35.6% in ~90 days ($0.200 → $0.129/M)** (pricepertoken.com, updated 2026-08-01) — undercutting
  is routine and answered.
- **OpenRouter's own warning about being the cheap+small provider** (cost-guide blog,
  2026-06-12): naive cheapest-first routing makes the cheapest provider "the first to saturate
  under load, the first to degrade, and the slowest to recover" — the inverse-square weighting
  exists to spread load, but a deep undercut with 2–4 replicas will pull more traffic than the
  fleet can hold.
- **429 semantics**: early 429 under load is the *recommended* behavior; 429s are tracked
  separately and do **not** damage uptime; queueing instead damages the public throughput stat
  (queue time counts in the denominator). The cost: heavy 429ing starves the router of
  evaluation data (Auto Exacto explicitly says so), keeping you out of the top tool-calling
  tier. Failed/fallback requests are not billed to users; a 429'd request simply falls over to
  the next provider (FAQ + cost-guide blog).
- **Capacity and region signaling exist**: `capacity_tpm` (input tokens/min) is read
  automatically from `/v1/models`; datacenter locations are a required listing field; providers
  can run region-specific endpoint variants (e.g. `google-vertex/us-east5`) and users can
  target or ignore specific endpoint slugs. Geographic 403s are tracked separately from uptime.
- **Mid-stream failure is the one unforgivable error class** — it counts against uptime AND
  wastes the user's tokens. Admission-reject (429) early, never mid-stream.

Net for darklanes: the correct posture is *honest small capacity* — low `capacity_tpm`,
admission-capped replicas that 429 fast (we already do exactly this at the proxy), price near
but not drastically below the floor, and scale replicas before deepening the undercut.

---

## 3. The tencent/hy3 competitive landscape (OpenRouter endpoints API, fetched 2026-08-02)

Model: `tencent/hy3` — 295B MoE, 21B active, 192 experts top-8, 262,144-token context, created
1783344048 (2026-07-06). **5 endpoints**. Raw JSON receipt: `hy3-endpoints-api-20260802.json`.

| Provider | Quant | Ctx | Max out | $/M in | $/M out | $/M cache-rd | Disc. | Uptime 30m / 1d | Tools | Params |
|---|---|---|---|---|---|---|---|---|---|---|
| GMICloud | bf16 | 262k | — | 0.1288 | 0.5336 | 0.0322 | 0.08 → eff. 0.1185/0.4909 | 99.85 / 99.71 | **no** | 8 |
| Tencent | fp8 | 262k | 128k | 0.1320 | 0.5280 | 0.0330 | 0 | 99.97 / 99.88 | yes | 10 |
| DeepInfra | fp8 | 262k | 131k | 0.1400 | 0.5800 | 0.0350 | 0 | 98.33 / 98.88 | yes | 17 |
| Novita | unknown | 262k | 262k | 0.1400 | 0.5800 | 0.0350 | 0 | 100.00 / 99.17 | yes | 15 |
| AtlasCloud | fp8 | 262k | 131k | 0.2000 | 0.8000 | 0.0500 | 0 | 99.82 / 99.73 | yes | 17 |

Notes:
- `latency_last_30m` / `throughput_last_30m` return null on the unauthenticated API; the model
  page renders them client-side. Third-party medians: ~94 tok/s output, ~1.64 s TTFT
  (pricepertoken.com, updated 2026-08-01, sourced from OpenRouter+Helicone). Artificial
  Analysis benchmarks hy3 providers at `artificialanalysis.ai/models/hy3/providers` (charts are
  JS-rendered; page confirmed live 2026-08-02, now benchmarked at 10k-input workloads).
- Every endpoint supports `reasoning` + `reasoning_effort`; all five price cached input at
  exactly 25% of input price. 4/5 support `tools` (GMICloud — today's price floor — does not).
- There is also `tencent/hy3-preview` (GMICloud solo, bf16, $0.063/$0.21) — a second, less
  defended surface, single-provider.
- Price competition is live: cheapest input fell $0.200 → $0.129/M in ~90 days
  (pricepertoken.com, 2026-08-01); GMICloud's `discount_to_user` 0.08 is the current knife.

**What price/quant point would attract the router.** Effective floor is GMICloud at
0.1185/0.4909 (bf16, but no tools). Inverse-square math: matching ~0.115/0.48 gains only ~1.06x
weight vs GMICloud and ~1.3x vs Tencent — marginal. A meaningful default-routing pull and
outright `:floor` capture needs ~10%+ under the effective floor, e.g. **$0.105/M in, $0.44/M
out, $0.026/M cache-read** (~1.27x weight vs GMICloud, ~1.6x vs Tencent, 100% of `:floor`).
Quant positioning: three incumbents are fp8, the floor is bf16. A GGUF/NVFP4-class endpoint must
declare its label honestly (`nvfp4`/`int4` are valid enum values); quality-sensitive users
filter `quantizations:["fp8","bf16"]`, so sub-fp8 serving cedes that segment and must win purely
on price + published eval receipts. The differentiated wedge nobody currently holds: **cheapest
endpoint that also supports tools** (beating GMICloud on price while carrying the `tools`
feature the floor lacks) — that is both the `:floor` segment and Auto Exacto-eligible agentic
traffic on a model marketed for agentic workflows.

**Revenue realism at floor prices** (own estimate, stated assumptions): at $0.44/M out +
$0.105/M in, a replica sustaining e.g. 300 tok/s output earns ~$0.48/hr on output; with
prompt-heavy agentic traffic at ~10:1 in:out and prefill an order of magnitude faster, input
adds roughly $1–1.5/hr — call it **$2–4/hr gross per saturated replica** against H100 node cost
of the same order of magnitude. An OpenRouter listing at today's hy3 floor is distribution,
public perf stats, and utilization for idle capacity — not a standalone profit engine at 2–4
replicas. (Consistent with why the field keeps consolidating to big hosts.)

---

## 4. Comparable channels, ranked by effort-to-revenue

1. **OpenRouter** — best demand pool (~100T tokens/mo across the network, OR blog 2026-06-12),
   zero provider-side take documented, monthly invoicing. Effort: application + review queue
   (backlogged, proprietary-first), OR-schema `/v1/models`, usage-in-stream, privacy policy.
   The only channel with meaningful self-serve-ish demand at our size. **Rank 1.**
2. **The Grid (thegrid.ai)** — spot market where "suppliers bid to serve your requests in real
   time"; explicit supplier funnel ("Have inference capacity to sell? … Chat with our Sales
   Team", thegrid.ai, fetched 2026-08-02). Tier specs are benchmark-defined (intelligence/
   throughput/latency) with continuous evaluation and automatic replacement — no committed
   supply advertised, pay-per-clearing-price. Model-agnostic tiers mean hy3 competes inside
   "Text Prime/Standard"-class buckets rather than on a model page. Good fit for monetizing
   *idle* capacity at market-clearing prices; sales-contact onboarding. **Rank 2.**
3. **Hugging Face Inference Providers** — documented partner path
   (`huggingface.co/docs/inference-providers/register-as-a-provider`, fetched 2026-08-02):
   OpenAI-compatible LLM APIs "may skip most" of the task-schema work; requires a Hub org on a
   **Team/Enterprise plan**, PRs into `huggingface.js` + `huggingface_hub`, model-mapping API
   registration, and a **billing endpoint returning per-request cost in nano-USD** (polled
   every minute; requests unbilled after ~30 min are dropped and not charged). Automated
   validation every 6 h incl. tool-calling and structured-output behavioral tests; TTFT < 5 s
   required; failing providers are temporarily delisted. Real integration work (two client PRs,
   a billing service) for a demand pool skewed to experimentation; DeepInfra only joined
   2026-04-29 (their blog). **Rank 3.**
4. **Direct API sales** — prerequisites are exactly the OpenRouter checklist plus go-to-market:
   public pricing/docs/status page, metered billing (we have billing-grade metering), a privacy
   policy, and buyer trust signals (SOC 2 expectations in any enterprise deal). Highest
   margin, slowest revenue, all sales effort on us. Do it opportunistically, not as the
   channel. **Rank 4.**
5. **Vercel AI Gateway / Requesty / Eden AI-class gateways** — demand-side aggregators whose
   provider sets are curated integrations; Vercel documents no "become a provider" path
   (docs fetched 2026-08-02, last_updated 2026-07-08) — routing pool is chosen by Vercel.
   Being on OpenRouter partially covers these anyway (several gateways resell OR). **Passive.**
6. **AVOID at our size — committed-supply / stake / marketplace-partnership channels:**
   - **AWS Bedrock Marketplace, Azure AI Foundry, GCP Vertex Model Garden** — partnership/
     marketplace programs where the model runs on *their* managed infra or under enterprise
     agreements; nothing for a 2–4 replica independent host.
   - **Bittensor/Chutes-style decentralized networks** — capacity commitment via stake and
     miner mechanics, token-denominated revenue, and volatile demand (Chutes 42B→8–12B
     tokens/day between February and March 2026, ownyourmind.ai 2026-04-10).
   - **Compute-block markets (SF Compute-class) and any deal with contracted capacity floors**
     — a committed-supply obligation against a 2–4 replica fleet is an outage waiting to be
     invoiced.

---

## 5. Verdict: darklanes listing checklist and the gap

### Already have (receipts in-repo)

| OpenRouter requirement | darklanes status |
|---|---|
| OpenAI-compatible `/v1/chat/completions` + `/v1/completions`, SSE streaming | Yes — `crates/memra-server` (`MEMRA_COMPAT=openai`, bearer auth via `MEMRA_API_KEY`), smoke-gated (`tools/serve-smoke.sh`); docs/SERVING.md |
| Usage tokens on stream + non-stream | Yes on `lane/dl-metering` — worker-truth `prompt_tokens`/`completion_tokens` on all shapes **including the SSE final chunk** (docs/METERING.md); **not yet merged** into this branch |
| Early 429 instead of queueing; Retry-After | Yes — proxy bounded-FIFO deadline → 429 + Retry-After (`tools/serve-proxy.py`); lane QoS sheds with immediate 429 (dl-metering battery, N=3, 2026-08-02) |
| Billing-grade accounting for monthly invoicing | Yes on `lane/dl-metering` — per-request JSONL, `/usage`, reconciliation verdict EXACT 13/13 incl. chaos kill |
| Reliability under load / chaos | Fleet supervisor + breaker, SIGKILL chaos receipts, greedy-hash exactness 18/18 (`research/fleet-v060-20260801/`) |
| Mid-stream error avoidance | VRAM-aware admission waits instead of mid-request OOM (SERVING.md, 2026-08-02) |
| Per-model capacity signal | Measurable from load harness (`tools/load-serve.py`) → `capacity_tpm` |

### Missing

1. **Tool calling — the single biggest gap.** memra-server *intentionally rejects* tool calls
   (`crates/memra-server/src/main.rs:110`). hy3 is sold as an agentic model; 4/5 incumbent
   endpoints support `tools`; Auto Exacto governs **all** tool-calling traffic and deprioritizes
   endpoints missing benchmark data. Without `tools` (+ `tool_choice`, and ideally
   `structured_outputs`/`json_mode`), a darklanes endpoint forfeits the model's primary traffic
   class and its routing tier on that class. Everything else below is days of glue; this is the
   one real engine+server feature.
2. **`reasoning` / `reasoning_effort` / `include_reasoning`** — supported by all 5 incumbents;
   hy3 is a reasoning model. Template/parse work on the serving surface.
3. **OR-schema `/v1/models` listing endpoint** — current `/models` is bare; needs USD-string
   pricing, context, max output, quantization label, features, datacenter locations,
   `capacity_tpm`, `is_ready`. Small, spec is fully documented.
4. **Merge `lane/dl-metering`** — usage-in-stream and invoicing-grade records are on a side
   branch; they're listing prerequisites, not optional polish.
5. **Business surface** — published privacy policy + data-retention/training disclosure,
   monthly invoicing capability, support contact, ToS. Paper, not code.
6. **hy3-on-H100 throughput evidence** — SERVING.md fleet numbers are the 9B fleet model;
   HY3-SPILL.md's current state is the 5090 spill profile. OpenRouter's technical review sends
   test traffic; we need measured hy3 TTFT/throughput per replica (and honest `capacity_tpm`)
   before applying, or the review does the measuring for us.
7. **Competitive niceties** — cache-read pricing (all incumbents price it at 25% of input;
   requires prefix-cache accounting), penalties/min_p/logprobs parameter breadth (incumbents
   carry 15–17 params; we list what we truly support — `require_parameters` users filter on it).

### Go / no-go

**Go — but sequenced, and with priced-in queue time.** Nothing in OpenRouter's requirements is
out of reach: the fleet, exactness gates, 429 discipline, and metering are already
listing-grade, and there is a real wedge (cheapest hy3 endpoint *with* tools, at ~$0.105/$0.44
vs the 0.1185/0.4909 effective floor). But (a) the application backlog explicitly deprioritizes
open-weight-only providers, so submit early and expect to wait; (b) tool-calling + reasoning
support must exist *before* the technical review, or hy3's agentic traffic — the reason to be
on that model page — routes around us; (c) at floor prices a 2–4 replica fleet grosses on the
order of single-digit $/hr — treat the first listing as distribution and public perf receipts,
not revenue. Recommended order: merge dl-metering → implement tools/reasoning on the serve
surface → OR-schema `/v1/models` → measure hy3-on-H100 with the load harness → publish
privacy/ToS pages → submit the application, with The Grid's supplier funnel as the parallel
low-effort channel for idle capacity.

---

## Source index (all fetched 2026-08-02)

- OpenRouter for-providers doc — openrouter.ai/docs/guides/community/for-providers (live)
- OpenRouter provider application — openrouter.ai/providers/apply (live)
- OpenRouter provider routing — openrouter.ai/docs/guides/routing/provider-selection (live)
- OpenRouter cost guide (routing example, 429/fallback, quant gotchas, 5.5% fee) —
  openrouter.ai/blog/tutorials/how-to-get-the-lowest-cost-llm-inference-on-openrouter/ (2026-06-12)
- OpenRouter models API + hy3 endpoints API — api/v1/models, api/v1/models/tencent/hy3/endpoints
  (receipts in this dir)
- OpenRouter FAQ — openrouter.ai/docs/faq (live)
- Sacra OpenRouter report — sacra.com/c/openrouter (updated 2026-05-31)
- TrueFoundry OpenRouter pricing — truefoundry.com/blog/openrouter-pricing (2026-06-25)
- pricepertoken hy3 page — pricepertoken.com/pricing-page/model/tencent-hy3 (updated 2026-08-01)
- Artificial Analysis hy3 providers — artificialanalysis.ai/models/hy3/providers (live)
- HF register-as-a-provider — huggingface.co/docs/inference-providers/register-as-a-provider (live)
- DeepInfra HF-provider announcement — deepinfra.com/blog/huggingface-inference-provider (2026-04-29)
- The Grid — thegrid.ai (live); Vercel AI Gateway docs — vercel.com/docs/ai-gateway (2026-07-08)
- Chutes volume data — ownyourmind.ai (2026-04-10); Grayscale Bittensor report (2025-12-03)
- Community: r/LocalLLaMA "Be careful in selecting providers on openrouter" (2025-08-07);
  HN 47563884 "Openrouter Going Rogue?" (2026-04-06); DigitalOcean OR-provider news (2026-06-03)
