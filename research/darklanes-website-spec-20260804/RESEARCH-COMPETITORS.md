# Research notes — inference-provider competitive teardown (web, fetched 2026-08-05)

Supporting evidence for SPEC.md §3, §4, §9. All pages fetched live 2026-08-05 unless
marked "search snippet". Prices per 1M tokens, input/output.

## Part I — the big four

All fetched live 2026-08-04/05. All four sites are heavily JS-rendered; "above the fold"
reconstructed from server-rendered HTML (headings, RSC payloads, meta tags).

### Together.ai — the research-credibility play, moved upmarket

- H1: "Build what's next on the AI Native Cloud." Stat cards under hero: "Faster
  inference 2x / Lower cost 60% / Faster pre-training 90%", each linking to proof.
  CTAs "Start building" / "Contact sales".
- Brand: collective/open-source heritage (RedPajama) shifted to full-stack enterprise
  cloud. The flagship trust asset is a homepage research wall — 70+ cards
  (FlashAttention-3/4, Mamba-3, ThunderKittens) with named authors incl. Tri Dao.
- Pricing page is no longer a per-token table: leads with PTUs (reserved throughput
  units) + savings calculator, GPU-hours (H100 $3.99/hr, B200 $8.19), fine-tuning
  tables. The serverless per-token board lives in docs
  (docs.together.ai/docs/serverless/models).
- Per-token (docs, 2026-08-04): **no Qwen3-32B/30B-A3B on the menu anymore** (catalog
  rotated); nearest 27B-class Gemma 4 31B $0.39/$0.97 (FP8); **Llama-3.3-70B Turbo
  $1.04/$1.04** flat. Free credits: conflicting third-party claims ($1 signup credit vs
  $5 minimum purchase) — snippet-only, unverified.
- Trust: status.together.ai publishes **per-model uptime** (e.g. "Llama 3.3 70B
  99.990%") — unusually granular; trust center subdomain; "99% uptime SLA" for
  committed capacity.
- Docs-first vs marketing: buyer/platform-first homepage, no code above fold.
  Distinctive: research wall + per-model status. Templated: the 2x/60%/90% stat-card
  hero.

### Fireworks.ai — the authority/ownership play; cautionary copy

- Above fold: Jensen Huang quote hero — "'Fireworks is the TSMC of AI Factories...'"
  CTA "Build your frontier". H1s: "Own your model. Own your future."
- Copy is abstract strategy language ("compounding specialized intelligence", "own the
  learning loop") — hard to parse what you buy without scrolling. Customer logos:
  Cursor, Sourcegraph, Notion, Quora, Perplexity, Vercel.
- Pricing: marketing page says "Pay per token... $1 in free credits", table delegated
  to docs. **Standard / Priority / Fast serving tiers** priced separately. Size-based
  catchalls: ">16B parameters $0.90/1M" uniform in+out; "MoE up to 56B $0.50/1M"
  (covers Qwen3-30B-A3B class). Llama-3.3-70B → $0.90 flat. On-demand H100 $7/hr,
  B200 $10/hr. Batch = 50% of serverless.
- Trust: status + trust-center subdomains; blog perf claims ("~250% higher
  throughput"); llms.txt agent-readable docs (Together has this too).
- Distinctive: three-tier serving-path pricing. Forgettable: the strategy-deck
  abstractions — the four's cautionary example of low information density.

### Groq — the hardware manifesto; pricing page deleted

- H1: "Groq is the premier neocloud for fast inference." Animated manifesto hero
  ("Training creates the possibility. Inference creates the value."). Minimal nav.
  Platform productized as GroqMetal/GroqCore/GroqAssured with hardware specs as proof
  ("40 PB/s SRAM bandwidth, 1,000 tokens/sec/user").
- **groq.com/pricing now redirects to the homepage** (observed 2026-08-04); per-token
  prices live only in console docs RSC payload: **qwen/qwen3.6-27b $0.60/$3.00**
  (the current Qwen mid-size flagship — direct comp for darklanes' SKU class);
  llama-3.3-70b-versatile $0.59/$0.79; gpt-oss-120b $0.15/$0.60. qwen3-32b model id
  exists but unpriced (rotating out); third-party snippets say $0.29/$0.59 at 662 T/s
  (unconfirmed). Free tier ~30 RPM no-card (snippets).
- Trust: groqstatus.com live; trust.groq.com; $650M raise in nav; per-model
  tokens/sec published in docs. Cleanest marketing/dev split of the four: groq.com =
  100% narrative, console.groq.com = 100% dev.
- Risk demonstrated: removing public pricing makes them opaque vs peers.

### DeepInfra — radical price transparency; the price-sheet-as-website

- Eyebrow "FAST SIMPLE RELIABLE LOW-COST" → H2 "AI Inference". CTAs "Let's Go" /
  "Book a consultation". **Live model cards with prices above the fold**, each showing
  $/M in/out/cached + quantization + context.
- Pricing pattern: per-token with tier MULTIPLIERS — Standard 1x, Priority 1.5x
  (faster TTFT), Flex 0.8x (non-production). "The provider with the most models on
  OpenRouter" as social proof.
- Per-token (live API catalog api.deepinfra.com/models/list, 2026-08-04 — the
  cheapest of the four): **Qwen3-32B $0.08/$0.28 (fp8)**, Qwen3-30B-A3B $0.12/$0.50,
  gemma-3-27b $0.08/$0.16, Llama-3.3-70B-Turbo $0.10/$0.32.
- Trust: "SOC 2 and ISO 27001 certified" + zero-retention claim on homepage; $107M
  Series B banner; live inference metrics on homepage; status page. No research blog.
- Most dev-first of the four. Distinctive: quantization honesty per card. Templated:
  MUI look, generic hero adjectives.

### Big-four cross-cutting findings

1. **Price-table placement IS the positioning:** DeepInfra above the fold
   (cheap+transparent), Fireworks/Together in docs, Groq deleted. A small provider
   earns trust at the DeepInfra end: exact in/out/cached + quant + context on the
   marketing page.
2. **Mid-size landscape (2026-08-04/05):** Qwen3-32B-class $0.08–0.29 in /
   $0.28–0.59 out; Groq prices qwen3.6-27b at a premium $0.60/$3.00 on speed;
   Llama-3.3-70B spans $0.10 (DeepInfra) → $1.04 flat (Together). Together and
   Fireworks have rotated Qwen3-32B OFF their menus — catalog freshness is itself a
   signal, and day-one support on a new Qwen drop is a visible differentiator.
3. **Table stakes across all four:** status page, trust-center subdomain,
   OpenAI-compatible pitch, tiered serving (priority/flex/batch multipliers). Absence
   reads amateur.
4. **Differentiators that actually distinguish:** Together's named-author research
   wall + per-model public uptime; Groq's per-model tokens/sec; DeepInfra's
   quantization disclosure. Fireworks' abstract copy is the anti-example.
5. **Quantization disclosure is an emerging trust axis** (Together, DeepInfra print
   quant per model; Parasail markets "no hidden quantization") — darklanes' bit-audit
   story is the strongest possible version of this axis.

## Part II — small providers + the marketplace lens

### Novita.ai — "full-stack AI cloud" template, executed well

- Hero (novita.ai): "Secure, isolated runtimes. Built for agents that actually do
  things." Products: agent sandboxes, GPU machines, serverless jobs, dedicated clusters;
  "Up to 50% less than major cloud providers"; "200+ models... Free to start."
- Distinctive: homepage carries an agent-facing note — full docs index at /llms.txt,
  markdown content negotiation. Treats AI agents as a first-class audience.
- Trust: status.novita.ai live; testimonial from Julien Chaumond (HF CTO): "often the
  first to get stable, production ready inference support online – often on Day One."
- Pricing: one giant per-model grid (novita.ai/models/llm). Llama 3.3 70B $0.135/$0.40
  (bf16, ctx capped 6k on OR); Gemma 3 27B $0.119/$0.20; Qwen3 Coder 30B-A3B
  $0.07/$0.27; Qwen3.5-35B-A3B $0.25/$2.00. Batch 50% off (snippet). Free: ~$0.50
  signup voucher (snippet) + rotating time-limited $0 models on the grid.
- Verdict: templated structure (hero → products → props → testimonials) rescued by
  day-0 model support speed and heavyweight testimonials.

### Hyperbolic — GPU-cloud-first; the anti-pattern exhibit

- Hero (hyperbolic.ai): "Hyperbolic makes building and running AI hyper simple";
  GPU-rental-led; stats bar "250K+ Engineers / Minutes, Not Weeks / Zero Quota Limit."
- Inference page pricing panel is STALE: still lists Qwen2.5-72B $0.40, Llama-3.1
  era — no Qwen3 family on the marketing page at all. Docs redirect chain 4 hops to a
  generic page (.xyz → .ai migration debt). Crypto-adjacent brand feel.
- Lesson for darklanes: stale price panels and broken doc redirects are what "not
  maintaining the site" looks like from outside — an argument for generated,
  CI-checked pricing surfaces.

### OpenRouter — the marketplace lens (the real storefront for a small provider)

- Homepage: "The Unified Interface For LLMs — Better prices, better uptime, no
  subscriptions." CTAs "Get API Key / Explore Models".
- Provider credibility IS public telemetry: per-endpoint quantization tag, context,
  prices, supported params, uptime_last_5m/30m/1d, TTFT + throughput charts. Example
  live contrast: Groq 100/100/99.94% uptime vs Nebius 54.9% 30-min uptime on the same
  model page — public and unforgiving.
- Routing: default = price-weighted load balancing (undercutting buys traffic);
  uptime tiers 95%+ normal, 80–94% deprioritized, <80% fallback-only (100+ requests
  before calculation). Guidance: return early 429s instead of queueing (memra's
  admission-shed design is literally this), stream immediately, SSE keep-alives
  (memra sends them). "Auto Exacto" reorders tool-calling traffic by benchmark
  accuracy/throughput/tool-call success — weak providers silently lose tool traffic.
- Live reference prices (OR endpoints API, 2026-08-05):
  - Qwen3-32B: DeepInfra $0.08/$0.28 (fp8, 41k), Nebius $0.10/$0.30, SiliconFlow
    $0.14/$0.57 (131k), Groq $0.29/$0.59 — only 5 endpoints, thin market.
  - Qwen3-30B-A3B: DeepInfra $0.12/$0.50, Alibaba $0.13/$0.52 — 2 endpoints, thinner.
  - Gemma-3-27B: DeepInfra $0.08/$0.16, Parasail $0.08/$0.45, Nebius $0.10/$0.30,
    Novita $0.119/$0.20, Phala $0.15/$0.46 (262k).
  - Llama-3.3-70B (reference): DeepInfra $0.10/$0.32 → Novita $0.135/$0.40 → Parasail
    $0.22/$0.50 → Groq $0.59/$0.79 → Together $1.04/$1.04 — 13 endpoints, crowded.
- User-side free tier: `:free` variants with daily caps gated on ~$10 lifetime credit
  purchase (exact numbers render via JS; flagged).

### Featherless.ai — the differentiated-pricing example

- Hero: "Cut your AI costs... One API key. Instant access... Browse 40,000+ models."
- Pricing pattern: FLAT-RATE subscription, not per-token — Premium $25/mo, concurrency
  units instead of tokens; "At 10M tokens/day, Featherless Scale consistently beats
  per-token pricing by 1.5–2x." No per-token menu exists — that's the point.
- Community-driven (Discord, roleplay/creative-writing segments), "no logs" privacy
  emphasis. Their 2024 Show HN: 7 points (see Part III).

### Parasail.io — the best-in-class small-provider site

- Hero: "The Inference Cloud for AI-native startups"; stats "30× Cheaper than legacy
  clouds", "Day 0 Support for frontier LLMs"; explicitly "Lossless by default, with
  **no hidden quantization**" — quantization disclosure is now a competitive axis.
- FAQ voice founder-direct: "shared Slack channel with your dedicated solutions
  engineer... Response time measured in minutes, not days."
- Pricing page pattern worth copying: four products (Serverless / Elastic / Dedicated /
  Batch), full per-token table (Llama 3.3 70B FP8 $0.22/$0.50, cache $0.11; Gemma 3
  27B $0.08/$0.45; Qwen3.5/3.6-35B-A3B $0.15/$1.00), plus a batch
  price-by-parameter-band grid (21–41B: $0.07/$0.22). Astro-built, light, unflashy —
  reads authentic.

### Chutes.ai — crypto-native cautionary tale

- Bittensor-based, TEE/confidential-compute angle; Plus/Pro monthly plans + PAYG.
- Free-tier cautionary tale: the $5-deposit/200-free-req/day tier was killed Aug 2025
  after >10,000 bot signups (snippets, Reddit) — gate free usage behind a small
  deposit from day one.

### Dropped candidates

- kluster.ai: pivoted/exited — homepage is a farewell note (joined MITO, AI video).
  Small providers disappearing mid-citation-cycle is a buyer fear darklanes' public
  receipts + public engine partially answer.
- inference.net: pivoted upmarket to custom fine-tuned SLMs; no public per-token menu.

## Part III — HN launch patterns (HN Algolia API, 2026-08-05)

Worked:
- "Show HN: Cloud GPUs for Deep Learning – At 1/3 the Cost of AWS/GCP" — 168 pts/93c
  (2021). Price-vs-incumbent number in the title.
- "Show HN: Oblivus GPU Cloud – from $0.29/hr" — 193 pts/129c (2023). Price in title.
- "Launch HN: RunAnywhere (YC W26) – Faster AI Inference on Apple Silicon" — 240
  pts/153c (2026-03); "Launch HN: Cactus – AI inference on smartphones" — 123/63.
  A specific technical wedge, not "another API".
- "Show HN: Echo – Fable-level results at 1/3 the cost using open-weight models" —
  484 pts/229c (2026-07). Quality-parity-at-fraction-of-cost is the current meta.
- "Show HN: NCompass – yet another AI Inference API, but hear us out" — 37/34
  (2024-12). Self-aware title bought a hearing.

Flopped:
- Featherless Show HN — 7 pts/2c (2024-06); generic "serverless GPU inference platform
  with predictable latency" — 5 pts/1c (2026-02); Novita early Show HNs — 2–4 pts.

Pattern: "we host models, here's an API" gets nothing. Traction = (a) a number in the
title, (b) a hardware/technical wedge with engineering depth in the thread, or (c) a
benchmark-parity claim. Engineering war stories earn comments only when the product is
differentiated — which supports leading darklanes' launch with the miscompile story
(a genuine technical wedge) rather than a product Show HN alone.

## Part IV — cross-cutting takeaways

1. Table-stakes trust kit: status page (all four live small providers have one),
   OpenAI-compatible API, single-page public per-token table, quantization disclosure,
   Discord, model-launch blog cadence.
2. OpenRouter is the real storefront for a small provider — uptime ≥95%, no queueing
   (early 429s), streaming immediacy, low price buys traffic automatically; the
   marketing site mainly must not contradict the marketplace telemetry.
3. Thin markets exist: Qwen3-32B has 5 endpoints, 30B-A3B has 2, vs 13 on
   Llama-3.3-70B. Cheapest-or-distinctive on a thin model is a visible position;
   DeepInfra ($0.08/$0.28 on Qwen3-32B) is the floor bar.
4. Distinctive ≠ expensive: Parasail's parameter-band batch grid and Featherless's
   flat rate are memorable because they break the template — darklanes' lane-priced
   tiers are the same kind of break.
5. Anti-patterns observed live: stale pricing panels (Hyperbolic), broken doc
   redirects, bot-destroyed free tiers (Chutes), providers vanishing (kluster.ai).
