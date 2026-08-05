# darklanes placement map — every listing/integration surface beyond OpenRouter + HF

Date: 2026-08-05. No GPU used. All web facts fetched live 2026-08-05 (source dates
inline). Companion research (NOT re-covered here): OpenRouter application study
`research/or-provider-20260802/REPORT.md` (application in prep), Hugging Face strategy
`research/hf-inference-20260804/ASSESSMENT.md` (org/cards/Space now, Providers program
later, condition-gated).

Product being placed: **darklanes** — GPU inference on memra, receipts/determinism
positioning (bit-audited kernels, seeded sampling, batch-invariant routing), q27-class
SKU, OpenAI-compatible API (serve surface complete per v0.69.0 "OR-listing surface
complete", commit 2c3099d2).

**Standing dependency for almost everything below:** a *publicly reachable* API base
URL + self-serve or emailable API keys + published per-token pricing. The serve code
is done; the public endpoint/pricing/signup is the gate. Every verdict marked
"needs-live-endpoint" is blocked on that, not on the integration work itself.

---

## Ranked summary — (value x readiness)/effort

### THIS WEEK (pre-Qwen3.8) — dispatchable lanes

| # | Surface | What it is | Effort | Why now |
|---|---------|-----------|--------|---------|
| 1 | **models.dev PR** (opencode/AI-SDK ecosystem registry) | TOML files + logo.svg in a community repo | ~0.5 day | One PR lists darklanes in opencode's 75+ provider picker, Kilo Code CLI, and every models.dev consumer. Schema-validated by CI, no discretionary gate beyond maintainer review. |
| 2 | **LiteLLM providers.json PR** | ONE JSON entry: `{"darklanes": {"base_url": ..., "api_key_env": "DARKLANES_API_KEY"}}` | ~0.5 day incl. test | LiteLLM SDK + LiteLLM proxy + **aider** (uses LiteLLM for all providers) + everything downstream, in a single-file PR. Best effort-to-reach ratio found in this study. |
| 3 | **pi provider preset** (owner's own daily agent) | models.json snippet in darklanes docs + example extension; optionally fix upstream issue #3168 | hours (preset); ~1 day if fixing #3168 | Zero-cost distribution to the exact agentic-coding audience; the owner already dogfoods pi. Upstream bug: custom OpenAI-compatible providers from models.json "load but hang" (pi-mono #3168, 2026-04-14) — verify against memra-server, fix if it reproduces = credibility + a working preset. |
| 4 | **`llm-darklanes` plugin** (simonw's LLM CLI) | Clone llm-openrouter (~200 lines), publish PyPI, PR the plugin directory | ~1 day | The datasette/Willison audience is precisely the receipts-and-reproducibility crowd. Plugin directory listing is a docs PR, no gate. |
| 5 | **"Use darklanes with X" docs/recipes page** | One page: Continue.dev, Cline, Roo, Kilo, VS Code Copilot BYOK, JetBrains AI, aider, Cursor-style, Cloudflare AI Gateway, Portkey, LangChain `ChatOpenAI(base_url=...)`, LlamaIndex `OpenAILike` | ~1 day | These 12 surfaces need NO registry PR — they all accept a custom OpenAI-compatible base URL today. The recipe page converts every one of them into a supported integration + SEO surface at once. |

Also this week (prep, fire later): Artificial Analysis outreach package (#6 below),
Vercel AI SDK provider npm package (#7).

### LAUNCH WEEK

6. **Artificial Analysis submission** — email/contact-form with endpoint, model list,
   public pricing, integrity-terms compliance. Prep now, send when endpoint is GA.
7. **Vercel AI SDK community-provider docs PR** — after publishing the npm package.
8. **Content syndication battery** — HN (planned), lobste.rs (invite needed; the nvcc
   miscompile post, not the launch post), This Week in Rust PR (crates release),
   /r/rust (engine story), X thread. r/LocalLLaMA: see §5 — owner account filtered,
   note alternatives.
9. **Continue Hub provider block** + **llm-prices / Helicone cost-DB PRs** (need
   public pricing).
10. **ProductHunt** — launch-week timing or skip; weak fit for infra (below).

### LATER (condition-gated)

11. AWS Marketplace SaaS listing (B2B procurement pull only; weeks of metering API work).
12. Eval-vendor BD (Braintrust/Langfuse/promptfoo "deterministic inference partner") —
    needs the determinism blog post live first; it's the pitch artifact.
13. LangChain/LlamaIndex *dedicated* packages (recipes suffice until demand).
14. Kong AI Gateway per-provider docs page (watch their issue #4316).
15. RunPod Hub engine template (engine marketing, not darklanes revenue).
16. SKIP as channels: Replicate/Modal serverless (§2 honest take), RapidAPI-class
    marketplaces.

---

## 1. Aggregators / routers beyond OpenRouter

### LiteLLM — config-PR shaped, yes. VERDICT: THIS WEEK

Source: docs.litellm.ai/docs/contributing/adding_openai_compatible_providers +
/docs/provider_registration (both fetched 2026-08-05).

- For OpenAI-compatible providers the path is literally **one JSON entry** in
  `litellm/llms/openai_like/providers.json`: `base_url` + `api_key_env`; optional
  `param_mappings`, `constraints`, `supported_endpoints`. Precedents in the same file:
  Hyperbolic, Nscale, PublicAI. The full Python-config path (transformation.py,
  ~6 files) is only needed for custom auth/streaming — we don't need it.
- Reach: LiteLLM is the compatibility layer for **aider** ("Aider uses the LiteLLM
  package to connect to LLM providers" — aider.chat/docs/llms/other.html, fetched
  2026-08-05), the LiteLLM proxy/gateway deployments, and many eval/observability
  tools' provider lists. One PR, many surfaces.
- What it takes: live endpoint, a model-name convention (`darklanes/<model>`), one
  test file. Effort: ~0.5 day. Gate: normal PR review, no business gate.
- Note: model *pricing/context* registration in LiteLLM's `model_prices_and_context_window.json`
  is a separate optional PR once list prices exist — do it with the pricing drop.

### models.dev (opencode / SST) — the registry the agent tools actually read. VERDICT: THIS WEEK

Source: github.com/anomalyco/models.dev README (fetched 2026-08-05);
opencode.ai/docs/providers ("OpenCode uses the AI SDK and Models.dev to support 75+
LLM providers", fetched 2026-08-05); Kilo-Org/kilocode #6315 (2026-02-15: CLI bundles
`@ai-sdk/openai-compatible` as default fallback, config reads models.dev IDs).

- Contribution = TOML PR: `providers/darklanes/provider.toml` with
  `npm = "@ai-sdk/openai-compatible"` + `api = "https://api.darklanes.../v1"` +
  `env = ["DARKLANES_API_KEY"]` + `doc`, a **required logo.svg** (currentColor, no
  fixed size), and one model TOML per SKU (`base_model = "alibaba/qwen3.6-27b"`-style
  inheritance keeps it to cost/limits/overrides only). CI schema-validates.
- Reach: opencode's provider picker, Kilo CLI, anything consuming
  `models.dev/api.json`. The q27 SKU rides existing model metadata.
- Effort: ~0.5 day incl. the logo. Gate: maintainer review; open-weights SKUs of
  models they already track = uncontroversial.
- Note: quantization honesty — declare the NVFP4/house-conversion in the model name or
  description exactly as on OR; same quant-label discipline.

### pi (earendil-works/pi-mono) — the owner's own tool. VERDICT: THIS WEEK

Source: pi-mono docs/custom-provider.md + docs/providers.md (fetched 2026-08-05);
issue #3168 (2026-04-14).

- Two placement forms: (a) a **models.json preset** users paste
  (`~/.pi/agent/models.json` provider entry, `api: "openai-completions"`, per-model
  `compat` flags) — publish in darklanes docs; (b) an **extension** calling
  `pi.registerProvider()` — publishable as an npm package, PR-able into
  `examples/extensions/`.
- Known landmine: #3168 reports custom OpenAI-compatible providers from models.json
  "load but hang indefinitely". The owner's daily client already talks to memra-server
  (the F4 dogfood receipts, commits 8032ab01→c716954b), so we may already know the
  working `compat` shape — verify, and if the bug reproduces against memra-server, an
  upstream fix is both a placement and a credibility artifact in the exact community
  we want.
- Effort: hours for the preset + docs; ~1 day if the upstream fix is real.

### Vercel AI SDK community provider. VERDICT: package this week, docs-PR launch week

Source: ai-sdk.dev/providers/community-providers/custom-providers (fetched
2026-08-05).

- The docs explicitly invite it: "Please publish your custom provider in your own
  GitHub repository and as an NPM package... you can submit a PR to the AI SDK
  repository to add your provider to the Community Providers documentation section"
  and "If you open-source a provider, we'd love to promote it here."
- Effort truth: a **full** LanguageModelV4 implementation is ~1 week, but we don't
  need it — `@ai-sdk/openai-compatible` (npm, fetched 2026-08-05: "provides a
  foundation for implementing providers that expose an OpenAI-compatible API") lets
  `darklanes-ai-provider` be a thin factory (~100 lines: baseURL, auth header, model
  id types). OpenRouter's community provider is the documented template.
- Value: AI SDK is the default TS stack; a listed provider page is durable
  documentation-as-distribution. Effort: 1–2 days total (package + docs PR).

### Portkey, Cloudflare AI Gateway, Kong — user-side config, not registries. VERDICT: recipes now, listings later

- **Cloudflare AI Gateway custom providers** (developers.cloudflare.com/ai-gateway/
  configuration/custom-providers, updated 2026-06-15): any user creates a
  `custom-darklanes` slug with our base_url per-account via dashboard/API. There is
  **no global provider directory to be listed in** — placement = a recipe in our docs
  + theirs is self-serve. Zero effort beyond the recipe page (#5).
- **Portkey** (portkey.ai/docs/integrations/libraries/openai-compatible, 2025-12-22):
  same shape — users register any OpenAI-compatible base URL as a named provider.
  Native listing in Portkey-AI/gateway is a code PR into their providers directory —
  possible but low marginal value over the openai-compatible path; LATER if Portkey
  users ask.
- **Kong AI Gateway** (developer.konghq.com/ai-gateway/ai-providers/openai, fetched
  2026-08-05): `upstream_url` option covers any OpenAI-compatible endpoint. Their
  docs team is actively writing per-provider example pages for OpenAI-compatible
  providers ("Fireworks AI, OpenRouter, SambaNova, Together AI (full list TBD)" —
  Kong/developer.konghq.com issue #4316, 2026-02-25). **Watch that issue; a docs
  contribution adding darklanes when the pattern lands is a free listing.** LATER.

### LangChain / LlamaIndex. VERDICT: recipes now, packages later

- LangChain: `langchain-community` is **sunset** (repo notice, 2026-06-19). New
  integrations are standalone `langchain-<provider>` packages + a docs listing, at
  maintainer discretion. But `ChatOpenAI(base_url=...)` works today → recipe page
  covers it. A dedicated package is LATER, demand-gated.
- LlamaIndex: `OpenAILike` (llama-index-llms-openai-like, PyPI fetched 2026-08-05) is
  the sanctioned wrapper for third-party OpenAI-compatible endpoints → recipe.
  Contribution guide warns integrations "may be declined" at maintainer discretion —
  don't spend a package on it yet.

### Continue.dev / aider / Cline / Roo / Kilo — the agentic-coding audience. VERDICT: THIS WEEK via recipes + two PRs

This audience is the spec-decode strength match (interactive latency, tool calls).

- **aider**: covered by the LiteLLM PR (#2) + an `.aider.model.settings.yml` block in
  the recipe page (context window, costs) — aider.chat/docs/config/adv-model-settings.
- **Continue.dev**: config-based (`provider: openai, apiBase: ...` —
  docs.continue.dev, fetched 2026-08-05) → recipe. PLUS a **Continue Hub block**
  (hub.continue.dev; continuedev/hub-configs monorepo publishes provider/model blocks)
  — a free branded tile in their catalog. Launch week, ~0.5 day.
- **Cline / Roo Code / Kilo Code**: all ship an "OpenAI Compatible" provider type
  (docs fetched 2026-08-05; Cline docs updated 2026-05-12). No registry to PR (Cline's
  custom-provider registry is still an open feature request, #4633). Kilo is
  additionally covered by the models.dev PR. → recipes.

## 2. Marketplaces

- **RapidAPI-class hubs**: RapidAPI is in visible decline (multiple "alternatives"
  roundups through 2026-01; Zyla/Postman Network are the successors). LLM inference
  buyers do not shop there; commission models are unfavorable. **SKIP.**
- **AWS Marketplace SaaS listing**: there is now a dedicated "SaaS API-based AI agent
  products" path (docs.aws.amazon.com/marketplace/latest/userguide/listing-saas-ai-agents,
  fetched 2026-08-05). It requires seller registration, integration with Marketplace
  metering/entitlement APIs, and listing review — realistically **weeks** of work,
  and its value is *procurement* (enterprises burning committed AWS spend), not
  discovery. **LATER**, only when a real enterprise prospect asks to buy through AWS.
- **Replicate / Modal / RunPod-serverless as channels — honest take: SKIP as
  channels.** These platforms bill for compute *on their GPUs* (Replicate: "billed for
  the compute time used to run your models", docs fetched 2026-08-05; RunPod Hub
  shares "up to 7% of compute spend" with publishers — Sacra profile, fetched
  2026-08-05). Publishing memra-served models there means renting their silicon and
  handing over the engine as a template — it surrenders exactly the vertical
  integration (own 5090/fleet + own engine) darklanes' margin and determinism claims
  stand on, and the per-request receipts/cache behavior would run on hardware we
  don't control. The one defensible variant: a **RunPod Hub template of the open
  memra engine** as *engine marketing* (drives GitHub stars and validates portability)
  — LATER, and only framed as "run the engine yourself", never as darklanes capacity.

## 3. Discovery / directories

- **artificialanalysis.ai — the one that matters.** Methodology + integrity terms
  (artificialanalysis.ai/methodology/performance-benchmarking, fetched 2026-08-05)
  confirm a submission relationship exists: "By **submitting endpoints** to Artificial
  Analysis, providers acknowledge these terms". Contact is via their contact form /
  contact@artificialanalysis.ai (FAQ, fetched 2026-08-05). What listing takes: a
  generally-available endpoint (they explicitly test with anonymous accounts and
  cross-check that benchmark traffic isn't special-cased), public pricing, and a
  model they track (the q27-class SKU qualifies — they benchmark per model per
  provider, TTFT + tok/s, 8x/day from GCP us-central1). Their integrity terms are a
  *gift* to the receipts positioning: darklanes can state compliance affirmatively
  ("same config for all traffic — here are the public gates") in the submission.
  Listing is at their sole discretion and free. Prep the package THIS WEEK
  (endpoint URL, model mapping, pricing, quantization disclosure — they are "moving
  towards full disclosure of quantization methods", which favors us), send at launch
  when the endpoint is GA. Effort: ~0.5 day of assembly.
- **Price-comparison DBs** (all PR-able once pricing is public, minutes each):
  simonw/llm-prices (GitHub), Helicone's open-source LLM cost API ("largest
  open-source API pricing database with 300+ models", github.com/helicone/helicone,
  fetched 2026-08-05), pricepertoken.com. LAUNCH WEEK with the pricing drop.
- **Awesome-lists — honest fit check**: the big awesome-LLM-inference lists
  (xlite-dev, sihyeong's Awesome-LLM-Inference-Engine) catalog *papers and engines*,
  not API providers → a **memra** (engine) entry is legitimate in
  Awesome-LLM-Inference-Engine; a darklanes entry is not. The provider-shaped lists
  are mostly "free LLM API" lists (mnfst/awesome-free-llm-apis) — only relevant if a
  free tier ships. LAUNCH WEEK, ~1 hour, engine-entry only.
- **alternativeto / ProductHunt**: alternativeto entry ("alternative to
  Together/Fireworks") is free and takes minutes — fine, launch week. ProductHunt is
  a consumer-app arena; GPU inference APIs get little qualified traffic there. Do it
  only if launch-week bandwidth allows; never at cost to HN/TWiR. LOW priority.

## 4. Community / dev surfaces

- **pi** — covered in §1 (top-3 pick).
- **simonw's LLM CLI plugin directory** (llm.datasette.io/en/stable/plugins/directory,
  fetched 2026-08-05): plugins are independent PyPI packages; the directory is a docs
  PR. llm-openrouter is the perfect template (same OpenAI-compat shape + key
  management). `llm-darklanes` = ~1 day including README and directory PR. The
  audience overlap with the determinism/receipts story is the best of any surface in
  this file. **THIS WEEK.**
- **VS Code / Copilot BYOK**: VS Code now has a first-class "custom OpenAI Compatible
  provider" path (code.visualstudio.com/docs/agent-customization/language-models,
  updated 2026-06-11; BYOK blog 2025-10-22). No registry — recipe page material.
  Building a dedicated VS Code extension is not worth it (a community extension
  already generically bridges OpenAI-compatible endpoints — r/GithubCopilot,
  2026-05-12). Recipe only.
- **JetBrains AI Assistant**: "OpenAI Compatible — use a model served through an
  OpenAI-compatible endpoint" is a supported BYOK provider type
  (jetbrains.com/help/ai-assistant/use-custom-models, updated ~2026-07), plus AI
  Enterprise lets admins wire an OpenAI-compatible endpoint org-wide (IDE Services
  docs, 2026-06-24 — that's a B2B seam worth one line in the recipe). Recipe only.
- **Continue Hub block** — §1, launch week.

## 5. Content-led syndication

- **HN**: already planned (launch queue, memory: bw24 launch posts). The nvcc
  13.0.88 miscompile catch (BLOG-EVIDENCE.md §A) is the strongest single post asset —
  compiler-bug war stories with committed 40-line repros are exactly HN/lobste.rs
  material and carry the receipts positioning implicitly.
- **lobste.rs**: invite-only; self-promo must stay <25% of activity and it is "not a
  launch channel for SaaS announcements" (lobste.rs/about + community writeups,
  fetched 2026-08-05). Play: get an invite (owner's network or the open invite
  threads), submit the *miscompile* post tagged `c`/`compilers`/`rust`, not the
  darklanes announcement. LAUNCH WEEK, minutes once an invite exists — start the
  invite hunt now.
- **This Week in Rust**: openly developed on GitHub; submissions land via PR to
  this-week-in-rust.org (fetched 2026-08-05), plus Crate of the Week nominations via
  the users.rust-lang.org thread. The memra crates hit crates.io at v0.69.0 (per-crate
  resumable publish fixed in c52edfd3) — a Rust-native CUDA inference engine is a
  legitimate TWiR item and /r/rust story ("the engine story": Rust + hand-tuned
  sm_120a kernels + bit-exactness gates). LAUNCH WEEK, ~1 hour.
- **r/LocalLLaMA**: owner's account (u/code_things) is sitewide spam-filtered
  (memory), and the sub adopted stricter self-promo rules with karma minimums
  (May 2026 "new rules" check-in; 1/10 guideline + disclosure). Alternatives, in
  order: (1) organic — ship the HN post and let it get cross-posted (common for
  strong technical content); (2) a fresh account that *participates first* for 1–2
  weeks (bring-up threads for Qwen3.8 are natural, on-topic contributions), then one
  disclosed post; (3) skip Reddit entirely and lean HN/lobste.rs/X/TWiR (the standing
  posture from memory). Do NOT evade the filter with sockpuppet launch posts — one
  removal poisons the domain.
- **X/technical-twitter**: account live from the July launch. The eval-determinism
  thread (§6 angle) and the miscompile thread are the two X-native assets.

## 6. B2B direct — the "deterministic inference partner" angle

The pain is documented in their own words: Braintrust's eval guides open with "AI
outputs are non-deterministic" (braintrust.dev/articles/llm-evaluation-metrics-guide,
2026-05-16); the Thinking Machines "Defeating Nondeterminism in LLM Inference" post
(2025) made batch-variance the canonical explanation of why temp=0 still isn't
reproducible on hosted endpoints; academic follow-ups measured it (EVAL4NLP 2025).

darklanes' receipt that maps onto this narrative **exactly**: the concat-prime-exact
work (merge b3a5465f, `research/concat-prime-exact-20260802/findings.jsonl`) found and
fixed batch-composition-dependent MoE expert selection (121/760 (layer,token) pairs
flipped experts with batch size) and shipped m-invariant router twins — i.e.
**batch-invariant serving is a shipped, gated property, not a roadmap item**. Plus
seeded sampling (F4 fixes), temp-0 exactness gates, and the honest-limits disclosure
(docs/SERVING.md ~7% near-tie first-token drift cross-config, with the
`MEMRA_PRIME_TOKENWISE=1` escape). Nobody else markets this; Thinking Machines wrote
the problem statement and serves no endpoints.

- **Sequencing**: this is a CONTENT play before it is a BD play. (1) Write the
  "batch-invariant inference, with receipts" post (material already inventoried in
  BLOG-EVIDENCE.md §C); (2) syndicate (HN/X); (3) then the BD emails to
  promptfoo (custom provider config is trivial — promptfoo.dev/docs/providers,
  fetched 2026-08-05 — the ask is a *featured example/guide*, not an integration),
  Langfuse and Braintrust (both accept custom OpenAI-compatible base URLs today; the
  ask is a co-marketing "reproducible evals" guide). Eval vendors have an incentive:
  their statistical-aggregation workarounds exist *because* providers are
  nondeterministic.
- Effort: post ~1 day (evidence exists); three emails ~0.5 day. Verdict: post is
  LAUNCH WEEK; emails LATER (post must exist first).
- **Agent-framework companies + fine-tune serving**: LATER; nothing actionable this
  cycle — the fine-tune-serving angle waits on the finetune SKU verdict lane.

---

## Surprisingly high-value findings

1. **LiteLLM's single-JSON-file path** — since 2025 the OpenAI-compatible on-ramp is
   one `providers.json` entry, not the six-file Python integration. Because aider
   rides LiteLLM, one ~20-line PR covers the SDK, the proxy, and a major coding
   agent simultaneously. Cheapest wide-reach placement discovered.
2. **models.dev is the actual provider registry for the agent-tool generation**
   (opencode, Kilo, AI SDK lookups) — a schema-validated TOML PR, no business gate,
   and the same `@ai-sdk/openai-compatible` npm shim we'd cite everywhere else.
3. **The batch-invariance receipt is a marketing asset nobody else has.** The eval
   industry's canonical unsolved complaint (Thinking Machines framing) is a *merged,
   gated, JSONL-backed fix* in this repo. That's the wedge for artificialanalysis,
   the eval vendors, and the launch narrative all at once.
4. **pi issue #3168** — the owner's own tool has a known-broken custom
   OpenAI-compatible path; a verified preset (or upstream fix) is placement +
   credibility in one move.
5. **Artificial Analysis accepts endpoint submissions** (their integrity terms are
   written around provider-submitted endpoints) — a free, continuously-refreshed
   third-party TTFT/tok-s board entry, and their move toward quantization disclosure
   plays directly into the declared-quant receipts posture.

## Dependency ledger (what blocks what)

| Blocker | Blocks |
|---|---|
| Public GA endpoint + key issuance | models.dev, LiteLLM, llm-darklanes, AI SDK package, Artificial Analysis, all recipes being real |
| Published per-token pricing | Artificial Analysis, llm-prices/Helicone/pricepertoken PRs, models.dev `[cost]` blocks (can stub 0 only where the schema allows) |
| Privacy/ToS pages (still missing — or-provider REPORT §5, re-flagged in HF ASSESSMENT) | OR application, HF later, AA submission credibility |
| Determinism blog post | eval-vendor BD emails, half the syndication battery |
| lobste.rs invite | lobste.rs submission |
| Qwen3.8 release (~week of 2026-08-10) | day-one-fastest uniqueness story reused across AA/HF/OR |

## Source index (all fetched 2026-08-05 unless dated)

- LiteLLM: docs.litellm.ai/docs/contributing/adding_openai_compatible_providers; /docs/provider_registration
- models.dev: github.com/anomalyco/models.dev README; opencode.ai/docs/providers; Kilo-Org/kilocode#6315 (2026-02-15)
- pi: github.com/badlogic/pi-mono packages/coding-agent/docs/custom-provider.md, docs/providers.md; issue #3168 (2026-04-14)
- Vercel AI SDK: ai-sdk.dev/providers/community-providers/custom-providers; npmjs.com/package/@ai-sdk/openai-compatible
- Cloudflare: developers.cloudflare.com/ai-gateway/configuration/custom-providers (2026-06-15)
- Portkey: portkey.ai/docs/integrations/libraries/openai-compatible (2025-12-22)
- Kong: developer.konghq.com/ai-gateway/ai-providers/openai; Kong/developer.konghq.com#4316 (2026-02-25)
- LangChain: github.com/langchain-ai/langchain-community sunset notice (2026-06-19)
- LlamaIndex: pypi.org/project/llama-index-llms-openai-like; run-llama/llama_index contribution guide
- Continue: docs.continue.dev/customize/model-providers; github.com/continuedev/hub-configs
- Cline/Roo/Kilo: docs.cline.bot/provider-config/openai-compatible (2026-05-12); docs.roocode.com/providers/openai-compatible; kilo.ai/docs/ai-providers/openai-compatible
- aider: aider.chat/docs/llms/other.html (LiteLLM); /docs/config/adv-model-settings.html
- LLM CLI: llm.datasette.io/en/stable/plugins/directory; github.com/simonw/llm-openrouter
- VS Code: code.visualstudio.com/docs/agent-customization/language-models (2026-06-11); BYOK blog (2025-10-22)
- JetBrains: jetbrains.com/help/ai-assistant/use-custom-models; /help/ide-services/manage-aie (2026-06-24)
- Artificial Analysis: artificialanalysis.ai/methodology/performance-benchmarking (v2.2.0, 2026-03-02) incl. Integrity Terms; /faq
- AWS: docs.aws.amazon.com/marketplace/latest/userguide/listing-saas-ai-agents
- Replicate: replicate.com/docs/topics/billing; RunPod: docs.runpod.io/hub/overview; sacra.com/c/runpod (7% Hub share)
- RapidAPI decline: blog.apify.com/best-rapidapi-alternatives (2026-01-26)
- lobste.rs: lobste.rs/about; TWiR: this-week-in-rust.org issues 648–660 (2026-04→07)
- r/LocalLLaMA: "New rules 1 week check-in" (2026-05-01); 1/10 self-promo guideline
- Determinism narrative: thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference; braintrust.dev/articles/llm-evaluation-metrics-guide (2026-05-16); aclanthology.org/2025.eval4nlp-1.12
- Repo receipts: research/or-provider-20260802/REPORT.md; research/hf-inference-20260804/ASSESSMENT.md; research/darklanes-website-spec-20260804/{SPEC,BLOG-EVIDENCE}.md; research/concat-prime-exact-20260802/findings.jsonl; research/crates-release-20260804/AUDIT.md; commits 2c3099d2, c52edfd3, b3a5465f, c716954b
