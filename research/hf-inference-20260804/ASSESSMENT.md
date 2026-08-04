# darklanes on Hugging Face as an inference offering — assessment

Date: 2026-08-05 (lane opened 2026-08-04). No GPU used. All web facts fetched live
2026-08-04/05 (source dates noted inline); repo receipts cited by path. Companion
research: `research/or-provider-20260802/REPORT.md` (OpenRouter listing study),
`research/gap-scan-20260802/REPORT.md` (serve-surface gap grades),
`research/serve-tail-20260804/RESULTS.md` (OR-schema `/v1/models`, rate-limit
headers, drain — the last serve-tail items).

Question: how does darklanes (memra engine, OpenAI-compatible fleet, 1–2 SKUs)
get onto Hugging Face as an INFERENCE OFFERING — the Inference Providers program,
or stepping stones first?

---

## Verdict up front

1. **HF Inference Providers listing is technically reachable but strategically
   premature as the FIRST channel.** The program has a fully documented,
   self-serve-ish path (docs live, fetched 2026-08-05) and for OpenAI-compatible
   LLM serving "you may be able to skip most" of the task-schema work. But it
   requires: a Team/Enterprise Hub plan ($20/seat/mo minimum), two client-library
   PRs (huggingface.js + huggingface_hub) that HF must review and merge, a
   per-request billing API in nano-USD polled every minute, 6-hourly automated
   revalidation incl. tool-calling and structured-output behavioral tests, and —
   decisive for a 1–2 SKU provider — HF's discretion ("Let's get in touch")
   at server-side registration. Every listed provider to date is a company
   (smallest precedent: Featherless AI, a small startup, entered June 2025 with
   6,700+ models — breadth was their pitch; Public AI, a nonprofit, entered
   Sept 2025 on sovereign-model uniqueness). A single-founder shop serving 1–2
   open-weight SKUs matches no precedent. Odds of acceptance this quarter: low
   (~10–20%, own estimate) without a uniqueness story; the ask costs weeks of
   integration before you learn the answer.
2. **OpenRouter first, HF Providers second.** OR is application-reviewed but has
   no plan fee, no client-library PRs, no billing API to build (they pay us by
   invoice), and the serve surface is already OR-ready (receipts below). HF's
   provider integration is a superset of effort for a demand pool that skews to
   experimentation ($0.10/mo free credits per free user, $2/mo per PRO seat —
   pricing doc, fetched 2026-08-05). Do HF's *free* surfaces now (org page, model
   cards, a Space demo), apply to OR now, and revisit HF Providers when there are
   ≥2 production SKUs plus OR uptime/latency receipts to show.
3. **Qwen licensing is clean.** Qwen3.6-27B is Apache-2.0 (verified via HF API
   `cardData.license: apache-2.0` + `license:apache-2.0` tag, fetched 2026-08-05).
   Qwen3.8 is NOT released yet (HF `search=Qwen3.8` under author Qwen returns
   empty, fetched 2026-08-04; expected week of 2026-08-10 per our WATCH.md
   re-check). Coverage as of 2026-08-03 (MarkTechPost) confirms **no license
   published yet** for 3.8 — verify on release day; the 3.6 precedent and the
   Wikipedia Qwen article (as of 2026-08-04: "Qwen3.6 model released under the
   Apache License"; 3.7 Max noted proprietary) both point Apache-2.0 for the
   open-weight line, but treat that as expectation, not fact.

---

## 1. HF Inference Providers program — what listing takes in 2026

Source: `huggingface.co/docs/inference-providers/register-as-a-provider` (fetched
2026-08-05), pricing doc (fetched 2026-08-05), provider launch blogs (dates below).

### Application process

There is no application form. The doc's front door is literally "Want to be listed
as an Inference Provider on the Hugging Face Hub? Let's get in touch! Please reach
out to us on social networks or here on the Hub" (the HuggingDiscussions thread
#49). The process is then nine documented steps, most of them work on our side:

1. Implement standard task APIs — for LLMs/VLMs, "If your implementation strictly
   follows the OpenAI API … you may be able to skip most of this section."
   We qualify: memra-server is OpenAI-compat, smoke-gated (docs/SERVING.md).
2. PR into `huggingface.js` `packages/inference` — a provider helper class
   (for chat: inherit `BaseConversationalTask`, cf. Cerebras/Fireworks impls),
   registration in `PROVIDERS` + `INFERENCE_PROVIDERS`, README update, tests.
3. Model Mapping API — `POST /api/partners/{provider}/models` linking HF model
   ids to our model ids. **Gate: "we have to enable your account server-side.
   Make sure you have an organization on the Hub for your company, and upgrade
   it to a Team or Enterprise plan."** This is the discretionary step — HF flips
   the partner bit; nothing in the docs promises they will.
4. Billing endpoint — an HTTP POST API on our side returning per-request cost in
   **nano-USD** for batches of up to 10,000 request IDs, polled **every minute**,
   idempotent, keyed on a response header (`Inference-Id` suggested) that must be
   present on every response **including streaming**. Requests unbilled after
   ~30 min are dropped and never charged — our loss. Our per-request JSONL
   metering (docs/METERING.md, reconciliation EXACT 13/13) is the right
   substrate; the endpoint itself is new glue (~1–2 days).
5. PR into `huggingface_hub` (Python) — mirror of step 2, "second step" after
   the Hub-side works.
6. Server-side registration + SVG icon — HF does it, on request.
7–9. Our own docs page (handlebars template PR into hub-docs), comms, "share
   share share."

### Technical requirements (the automated bar, ongoing)

Once mapped, every model is re-validated **every 6 hours** by live API calls:

- API reachable, HTTP success; output format compatible with the HF JS client.
- **Latency: under 5 s time-to-first-token (streaming) for conversational/text**;
  under 30 s for other tasks.
- **LLM behavioral tests: tool calling AND structured output** — specific
  requests whose responses must parse. Failing providers are temporarily
  delisted; failed mappings retest hourly; a status flip triggers immediate
  revalidation.
- If OpenAI-compatible, expose pricing + context length via `/v1/models`
  (`pricing: {input, output}`, `context_length`) — powers their comparison table
  and `:fastest` / `:cheapest` selection.

We already pass the behavioral bar on the serve surface: tools + streamed
tool_calls deltas (`research/serve-tools-20260802/`), structured outputs /
json_mode and reasoning-effort input (gap-scan §1 table), OR-schema `/v1/models`
with context_length + pricing stubs (`research/serve-tail-20260804/RESULTS.md`
item 1 — pricing strings currently "0", would need real numbers), rate-limit
headers and graceful drain (items 2–3). TTFT ≪ 5 s at our batch sizes is not the
constraint.

### Business terms

- **Take rate: none today.** "There's no additional markup from us; we just pass
  through the provider costs directly. (In the future, we may establish
  revenue-sharing agreements with our provider partners.)" — provider launch
  blogs (inference-providers-publicai, 2025-09-17) and pricing doc (fetched
  2026-08-05). So 100% of routed revenue reaches the provider… minus the
  future-rev-share reservation they print in every announcement.
- **Payout mechanics: not published.** The docs specify how HF *collects* cost
  per request (the nano-USD billing API) but not the remittance schedule to
  providers; that lands in the private partner arrangement. Plan for
  monthly-settlement working capital like OR.
- **Costs to us:** Team plan $20/user/month minimum for the org (HF pricing,
  fetched 2026-08-05 via docs pointer); engineering: two client PRs + billing
  endpoint + mapping integration — realistically 1–2 weeks of focused work plus
  HF review latency on the PRs (out of our control, historically days–weeks).
- **Demand context:** free users get $0.10/month in credits, PRO $2/month,
  Team/Enterprise $2/seat (pricing doc, fetched 2026-08-05). The routed pool is
  real but experimentation-weighted; HF publishes no per-provider token-volume
  numbers (searched 2026-08-05, none found). Provider default ordering on model
  pages = total HF-routed requests over the last 7 days (register doc FAQ) — a
  new small provider starts at the bottom of the widget order.

### Precedent — who got in, when, at what scale

Every provider in `huggingface.js` `INFERENCE_PROVIDERS` today (fetched
2026-08-05): baseten, cerebras, cohere, deepinfra, fal-ai, featherless-ai,
fireworks-ai, groq, hf-inference, novita, nscale, openai, ovhcloud, publicai,
replicate, scaleway, together, wavespeed, zai-org. Nineteen entries; all
companies or institutions.

- **Launch cohort (2025-01-28):** fal, Replicate, SambaNova, Together — invited
  partners at launch (HF blog, 2025-01-28).
- **Novita + Hyperbolic + Nebius + Fireworks (2025-02-18/19,** HF blog +
  huggingface_hub v0.29.0 release**):** weeks after launch. Novita and
  Hyperbolic were already multi-model GPU clouds with public price sheets and
  OR presence — small companies, not solo operations. (Hyperbolic has since
  dropped out of the current provider list — being listed is not permanent.)
- **Featherless AI (2025-06-12,** HF blog + featherless.ai blog**):** the
  smallest team to enter, and the instructive case. Their wedge was not scale of
  infra but **coverage**: "Hugging Face's largest LLM inference provider with
  6,700+ models" via their orchestration layer. A startup got in by being
  maximally useful to the long tail of Hub model pages. They raised $20M later
  (SiliconANGLE, 2026-04-30) — at entry they were seed-stage.
- **Public AI (2025-09-17,** HF blog**):** a nonprofit running donated
  vLLM clusters — got in on uniqueness (only route to Swiss AI / AI Singapore
  sovereign models), not scale.
- **DeepInfra (2026-04-29,** deepinfra.com blog**):** an established provider
  joining late — the pipeline stays open in 2026.

Pattern: HF lists providers that make *many Hub model pages* light up (
Featherless: thousands) or that serve models *nobody else serves* (Public AI).
A darklanes listing with 1–2 SKUs of models that already have 5+ providers
(hy3 has five endpoints on OR — or-provider REPORT §3) adds nothing to HF's
coverage story. The realistic single-founder angle is the Public AI angle:
**serve something unique** — e.g. the day-one fastest Qwen3.8-27B endpoint on
Blackwell, or house artifacts (NVFP4+MTP conversions) nobody else runs — and/or
enter after OR receipts prove reliability. Odds without that: low. Odds with a
unique-model + OR-track-record story: plausible (Featherless and Public AI prove
non-giants get in), but on HF's timeline, not ours.

---

## 2. Stepping stones (all available now, none gated on HF's discretion)

### (a) HF Space demo calling OUR endpoint — allowed and cheap

- Spaces may make outbound network requests on ports 80/443/8080
  (docs/hub/spaces-overview, fetched 2026-08-05: "you can make requests through
  the standard HTTP and HTTPS ports (80 and 443) along with port 8080"). A
  Gradio/static Space that calls the darklanes API is a normal, ToS-clean
  pattern — Spaces exist to demo things; secrets (our API key) go in Space
  Secrets, never in code.
- **Cost of always-on:** the demo needs NO Space GPU — our fleet does the
  compute. A CPU Basic Space (2 vCPU/16 GB) has no hourly cost; free-tier
  Spaces sleep when unused (cold start seconds), paid plans keep them warm.
  Note the 2026 gating: Gradio/Docker Spaces now require a paid plan (PRO $9/mo
  personal) — free personal accounts get up to 2 ZeroGPU Gradio Spaces
  (spaces-overview, fetched 2026-08-05). A **static** Space (transformers.js-
  style single-page HTML hitting our endpoint from the browser) is free for
  everyone and never sleeps in the container sense — but exposes the endpoint
  to client-side traffic, so it needs a rate-limited public demo key. Practical
  pick: PRO account + one Gradio Space proxying to us with a server-side key,
  $9/mo total.
- ZeroGPU is irrelevant for us (it runs the model *on HF's H200 slices* inside
  the Space — we want the opposite: traffic to OUR fleet). Only use ZeroGPU if
  we want a free-compute fallback demo decoupled from fleet uptime.
- Rate/ToS constraint that matters: a public demo is unauthenticated traffic
  into the fleet — front it with the existing proxy admission caps (serve-tail
  429 discipline) and a demo-scoped key.

### (b) Hub model repos + "inference pointer" — partial

- **The model-page inference widget is provider-gated.** "Widgets are displayed
  when the model is hosted by at least one Inference Provider" (docs/hub/
  models-widgets, fetched 2026-08-05). There is **no `inference:` metadata that
  points a widget at an arbitrary external endpoint** — that mechanism died
  with the old serverless Inference API; today widget = registered provider.
  So no, we cannot make `Avifenesh/…-bw24` model cards run against
  darklanes without being a listed provider.
- What model cards CAN do today: (1) `widget:` example inputs **with `output:`
  showing canned results** — explicitly recommended "when the model is not yet
  supported by Inference Providers, so that the model page can still showcase
  how the model works" (same doc); (2) ordinary markdown — a prominent
  "Try it live" link/badge to the Space demo and API docs; (3) `pipeline_tag`,
  `license`, `base_model` metadata so the repos index correctly; (4) the Space's
  README `models:` key links Space ↔ model repos, which also surfaces the Space
  on each model page's "Spaces using this model" list (spaces-overview, fetched
  2026-08-05) — that list is the free advertising slot: our demo Space appears
  ON `Qwen/Qwen3.6-27B` (100 Spaces listed there today, fetched 2026-08-05)
  if the Space references the model id.
- "Ask for provider support" button exists on unserved model pages
  (models-widgets, fetched 2026-08-05) — community demand signal we can watch
  for our artifacts, and a place users nudge providers.

### (c) The org presence — what polish buys

Current state (HF API, author=Avifenesh, fetched 2026-08-05): 16 public repos —
ModernBERT rankers (May), EAGLE3 drafts (June), `memra-bench` (July, 470
downloads, the healthiest asset), `Hy3-REAP-Layer103p5-bw24` (July, bw24-tagged,
0 downloads). It reads as a researcher's scratchpad, not a serving company.

A darklanes **organization** (orgs are free; Team plan only needed for the
provider program later) contributes: a canonical namespace for served artifacts
(`darklanes/qwen3.8-27b-nvfp4-mtp` style), model cards that double as SKU spec
sheets (quant ladder, exactness gates, perf receipts per our evidence
discipline), `memra-bench` and the demo Space under one brand, and the org page
as the "who is this provider" link every OR/HF application asks for. Cost: an
afternoon plus card-writing. It is also a hard prerequisite: the provider
program requires "an organization on the Hub for your company" before the
mapping step, and `PROVIDERS_HUB_ORGS` maps every provider to its org.

---

## 3. OpenRouter vs HF Providers — sequencing

| Axis | OpenRouter | HF Inference Providers |
|---|---|---|
| Demand | ~100T tokens/mo network-wide (OR blog, 2026-06-12); production-agent heavy | Volume unpublished; credit sizes ($0.10 free/$2 PRO per month) say experimentation-weighted |
| Take rate | None provider-side; ~5.5% demand-side fee (or-provider REPORT §1) | None today; rev-share explicitly reserved for the future |
| Gate | Application review, backlogged, proprietary-first priority | HF discretion + Team plan + merged client PRs |
| Integration lift | Zero new code — serve surface is OR-ready (receipts below) | Billing API + 2 client PRs + mapping + docs page: 1–2 weeks + review latency |
| Ongoing bar | Uptime tiers at 100+ req; Auto Exacto on tool traffic | 6-hourly revalidation, TTFT<5s, tools+structured tests |
| Payment direction | They pay us (monthly invoice) | They pay us (mechanics unpublished), we run the cost API |
| What it buys at N=1 provider | Model-page endpoint + public perf stats + routed traffic | Widget presence on model pages + SDK reach |

**Technical readiness — OR side is done.** The or-provider REPORT's seven gaps
have closed on `restructure/public-split`: tools/tool_choice/streamed deltas
(serve-tools-20260802), reasoning-effort input (gap-scan §1), usage-in-stream +
metering (merged; v0.69.0 release notes "OR-listing surface complete", commit
2c3099d2), OR-schema `/v1/models` + X-RateLimit trio + graceful drain
(serve-tail-20260804 RESULTS, all PASS). Gap-scan F13 (reasoning OUTPUT
separation into `reasoning`/`reasoning_details` response fields) remains the one
flagged item to re-verify before submitting — the gap-scan graded it
listing-relevant because all hy3 incumbents expose it.

**Sequencing: OR first.** Reasons: (1) zero incremental engineering vs ~2 weeks
for HF; (2) OR's review measures us on exactly what we've already gated; (3) an
OR listing generates the public TTFT/throughput/uptime stats that make a later
HF pitch credible ("here is our provider page, 99.9% 30-day uptime"); (4) HF's
discretionary gate is best approached with a uniqueness story (Qwen3.8 day-one,
house artifacts) that doesn't exist until 3.8 launch anyway. HF's free surfaces
(org, cards, Space) run in parallel now — they cost days and feed both channels.

---

## 4. Compliance/legal for commercial serving

- **Qwen3.6-27B: Apache-2.0.** Verified 2026-08-05 via HF API: tag
  `license:apache-2.0`, `cardData.license_link` → repo LICENSE. Apache-2.0
  permits commercial serving, modification (our NVFP4/MTP conversions are
  derivative works — fine), and resale of outputs. Obligations: ship the LICENSE
  text and NOTICE (if any) with redistributed artifacts (our repacked GGUF/HF
  repos must include the Apache LICENSE + attribution to the Qwen authors), and
  state changes. No use-restriction clauses, no MAU thresholds (that was the old
  Qwen/Tongyi license generations — the current open-weight line moved to
  Apache; Wikipedia Qwen article as of 2026-08-04 concurs).
- **Qwen3.8: license unpublished as of 2026-08-04** (MarkTechPost 2026-08-03:
  "no benchmark table, no license, no activated-parameter count published";
  our WATCH.md re-check 2026-08-04: no HF repo exists). Expected Apache-2.0 by
  precedent; the bring-up runbook should gate "published as supported" on
  reading the actual LICENSE file day-one. If it ships Apache-2.0, zero friction.
  If it ships a Qwen-style community license instead, re-run this section —
  those historically demanded attribution ("Built with Qwen") and had
  large-user-count clauses.
- **Attribution surface:** model cards for our conversions must carry
  `base_model:` metadata (already done on Hy3-REAP repo), the upstream LICENSE
  file, and a "derived from Qwen/Qwen3.8-27B" line. On the serving side
  (OR/HF listings), naming the model accurately ("Qwen3.8-27B, NVFP4 house
  conversion, quantization declared") is both a ToS matter (OR quant-label
  enforcement — or-provider REPORT §1) and Apache attribution hygiene.
- **Hy3 (the other SKU candidate):** tencent/Hy3 is served by five OR providers
  commercially; our repo already tags it. Verify its exact license on the
  pinned revision before it becomes a *listed* SKU (out of scope here; flagged).
- **Channel ToS:** HF provider program imposes the billing-API contract and
  revalidation; nothing IP-related beyond normal marketplace terms. Both
  channels require a published privacy policy + data-retention disclosure —
  still the open "business surface" item from the OR checklist (or-provider
  REPORT §5 Missing #5): no privacy/ToS pages exist in-repo as of 2026-08-05.

---

## 5. Concrete next actions

### This week (no GPU, no launch dependency)

1. **Create the darklanes HF org + move/mirror the serving-relevant repos**
   (memra-bench, Hy3-REAP, future SKU artifacts) with real model cards
   (license file, base_model, quant ladder, gate receipts). ~0.5 day.
2. **Write the privacy policy + ToS + data-retention page** (static site or
   repo docs). Blocks BOTH channel applications; it's paper, not code. ~0.5 day.
3. **Close/verify gap-scan F13** (reasoning output separation) on the serve
   surface and re-run the serve battery. The last flagged OR item. ~1 day if
   open; hours if the fp8st/serve-st work already covered it.
4. **Submit the OpenRouter provider application** (openrouter.ai/providers/apply)
   with the hy3-or-current-SKU story, honest capacity_tpm, and the uptime/
   exactness receipts. The backlog is long (proprietary-first priority — or-provider
   REPORT §1); being IN the queue early is the whole point. ~0.5 day of form +
   evidence assembly.
5. **Stand up the demo Space** (PRO account $9/mo, Gradio, server-side demo key,
   admission-capped): chat UI → darklanes endpoint, README `models:` linking to
   the SKU repos so the Space surfaces on those model pages. ~1 day.

### At Qwen3.8 launch (week of 2026-08-10 expected)

6. **Read the actual 3.8 LICENSE before "published as supported"** — one-line
   gate added to the bring-up runbook. Minutes.
7. **Publish the day-one artifact repos under the org** (house NVFP4+MTP GGUF +
   any FP8-ST notes), cards carrying the bring-up receipts; example `widget:`
   inputs with canned `output:` blocks so the pages demo without a provider.
   ~0.5 day on top of bring-up.
8. **Point the Space at 3.8 day-one** and post the launch comms (HN/X per the
   existing distribution queue). Hours.
9. **Add 3.8 to the OR application/listing** the moment gates pass — day-one
   fastest-Blackwell-endpoint is the differentiation window.

### Later (condition-gated, not calendar-gated)

10. **HF Inference Providers application — trigger conditions:** (a) OR listing
    live with ≥30 days of public uptime/latency stats, (b) ≥2 SKUs in
    production, (c) a uniqueness story (model or artifact no listed provider
    serves, or a day-one exclusive). Then: upgrade org to Team ($20/mo), open
    the huggingface.js PR (BaseConversationalTask subclass — days, not weeks),
    build the nano-USD billing endpoint on the metering JSONL (~1–2 days),
    reach out via HuggingDiscussions #49 + direct contact, then the
    huggingface_hub PR + docs page after HF flips the partner bit. Total
    engineering ~1–2 weeks; calendar time dominated by HF review.
11. **Real pricing in `/v1/models`** (currently "0" stubs — serve-tail item 1)
    once list prices are set for OR; the same numbers feed HF's comparison table.

---

## Source index

Fetched 2026-08-04/05 unless noted:
- HF register-as-a-provider — huggingface.co/docs/inference-providers/register-as-a-provider
- HF Inference Providers index — huggingface.co/docs/inference-providers/en/index
- HF pricing/billing — huggingface.co/docs/inference-providers/pricing
- HF models-widgets — huggingface.co/docs/hub/models-widgets
- HF spaces-overview (networking, plans, hardware table) — huggingface.co/docs/hub/spaces-overview
- huggingface.js provider registry — raw.githubusercontent.com/huggingface/huggingface.js/main/packages/inference/src/types.ts
- Launch blog (2025-01-28) — huggingface.co/blog/inference-providers
- Novita/Hyperbolic/Nebius/Fireworks joins (2025-02-18/19) — huggingface.co/blog/inference-providers-nebius-novita-hyperbolic; github.com/huggingface/huggingface_hub/releases/tag/v0.29.0
- Featherless join (2025-06-12) — huggingface.co/blog/inference-providers-featherless; featherless.ai blog same date; SiliconANGLE $20M raise (2026-04-30)
- Public AI join (2025-09-17) — huggingface.co/blog/inference-providers-publicai
- DeepInfra join (2026-04-29) — deepinfra.com/blog/huggingface-inference-provider
- Qwen3.6-27B license — huggingface.co/api/models/Qwen/Qwen3.6-27B (apache-2.0)
- Qwen3.8 absence — huggingface.co/api/models?author=Qwen&search=Qwen3.8 (empty, 2026-08-04); research/qwen38-prep-20260803/WATCH.md (TechTimes/MarkTechPost 2026-08-03: week of Aug 10, no license published)
- Qwen license posture — en.wikipedia.org Qwen article (as of 2026-08-04)
- Repo receipts — research/or-provider-20260802/REPORT.md; research/gap-scan-20260802/REPORT.md; research/serve-tools-20260802/; research/serve-tail-20260804/RESULTS.md; docs/METERING.md; docs/SERVING.md; commit 2c3099d2 (v0.69.0 "OR-listing surface complete")
