# darklanes — launch website spec

Date: 2026-08-05. Lane: research+writing, no GPU. Status: ready for implementation
pending the owner sign-offs in §12.

**Who this is for:** an implementing agent (or human) with ZERO prior context on this
project. Everything you need is in this directory:

- `SPEC.md` (this file) — the build brief.
- `DOMAINS.md` — domain availability receipts (checked 2026-08-05, nothing purchased).
- `EVIDENCE-REPO.md` — the inventory of repo receipts every marketing claim must trace to.
- `BLOG-EVIDENCE.md` — the source material behind each blog post.
- `RESEARCH-DESIGN.md` — the web-research evidence behind the design choices.
- `RESEARCH-COMPETITORS.md` — the competitive teardown (providers, pricing, trust signals).

Rule zero, inherited from the engine's own discipline and non-negotiable on the site:
**no claim without a receipt.** Every number on the site links to a raw log in the public
memra repo or to a named third-party source with a date. If a claim can't be traced,
it doesn't ship. This is not a style preference — it IS the product positioning (§2).

---

## 1. Business context (self-contained)

**darklanes** is a GPU-inference business built by a single founder (Avi Fenesh, Tel
Aviv — systems engineer at AWS ElastiCache, maintainer of Valkey GLIDE, independent ML
researcher at avifenesh.ai). It serves open-source models on owned/rented NVIDIA
Blackwell hardware (RTX PRO 6000 96GB class trajectory), powered by **memra** — a
from-scratch Rust+CUDA inference engine, public on GitHub, with an unusual discipline:
every kernel is bit-audited against a CPU reference, every serving mode is gated
token-identical to plain decode, and every published number has its raw run logs
committed in-repo.

- **First SKU:** one model — the Qwen 27B class at 8-bit (Qwen3.6-27B today; Qwen3.8-27B
  day-one support when it drops ~Aug 10, runbook already written). Single-digit SKU count
  is deliberate: one model, served obsessively, beats a thousand-model menu on every
  quality axis a small fleet can control.
- **Measured serving numbers** (RTX PRO 6000, 2026-08-04, all receipted —
  EVIDENCE-REPO.md §1): MTP speculative decode **170.6 tok/s single-stream** through the
  serving surface (186.7 bare), **421 tok/s aggregate at c=8**, TTFT **0.182 s cold /
  3 ms warm** (prefix-cache hit). 140-minute endurance soak on an 8-GPU fleet:
  464,870 requests, **0 errors, 0 sheds**, throughput drift +0.045%, deterministic
  output hash identical on all replicas before and after.
- **The namesake mechanism:** QoS lanes. Requests tag `interactive` (protected — hard
  p95, never preempted), `judge`, or `harvest` (batch — rides spare capacity, shed at
  admission with 429 + Retry-After, never queued inside the engine). Interactive and
  batch traffic share the same GPUs without the batch tenant destroying interactive
  tail latency. Measured: a c=96 bulk tenant inflates a lane-blind fleet's interactive
  p95 by ~4x; with lanes on, the SLO dial can hold contended p95 statistically equal
  to an uncontended box.
- **Serve surface:** OpenAI-compatible (validated against the official `openai` SDK),
  streaming, tools/function-calling, REAL constrained decoding (`json_schema` via
  llguidance at 99.4% of unconstrained speed), cross-request prefix caching with
  per-tenant `cache_salt` isolation, honest usage accounting (cached tokens itemized,
  aborted requests billed to the abort point).
- **Business posture (owner-set):** serving revenue covers hardware; research is the
  product. Success bar = cost coverage + public reliability stats, NOT market-share
  wins. "Everything need to be honest, nothing is a dream that we try to make true."
  The site must never promise scale it doesn't have; it converts on trust density, not
  breadth.

## 2. Positioning (the choice, and why)

**Chosen positioning: "the inference provider that shows its work" — deterministic,
receipt-published serving for people whose workloads break on nondeterminism: evals,
agents, and CI.**

One paragraph, usable verbatim in briefs: *darklanes is a boutique GPU-inference
provider serving one open model class, obsessively. Where the big fleets sell a menu,
darklanes sells a property no one else markets: bit-audited determinism — same input,
same tokens, alone or under load, gated and receipted, with every performance number
traceable to a raw log in the public engine repo. It is built for eval pipelines that
must reproduce, agent loops that must be debuggable, and mixed workloads that need hard
interactive p95 while batch jobs ride the same GPUs at a discount — the dark lanes.*

Why this wins attention against the big fleets (each hypothesis tested against
evidence):

1. **Determinism/reproducibility is an unserved, named pain.** OpenRouter — the
   marketplace lens on the whole industry — teaches buyers that undisclosed
   quantization is "the hidden quality variable" and publishes per-provider percentile
   charts precisely because providers are not trusted (RESEARCH-DESIGN.md §4).
   Eval and agent builders currently cannot get a "same tokens every time" guarantee
   from any listed provider. memra has it as a gated contract, not a best-effort
   (EVIDENCE-REPO.md §2). Nobody else CAN market this easily: it requires owning the
   engine down to kernel reduction order.
2. **Receipts convert; adjectives don't.** The one published A/B with numbers (Evil
   Martians) moved conversion 0.1% → 2.0% by replacing vague claims with specific
   metrics (RESEARCH-DESIGN.md §1). darklanes' entire evidence discipline — raw JSONL
   next to every summary — is a conversion asset the incumbents' marketing teams cannot
   replicate without changing how their companies work.
3. **The lanes story monetizes a real trade-off instead of hiding it.** Every provider
   suffers the noisy-neighbor problem; none sell the solution explicitly. "Your
   interactive p95 is protected by admission control, and your batch jobs are cheaper
   BECAUSE they yield" is both a differentiated product and an honest price structure
   (§9).
4. **Built-in-the-open credibility.** The engine is public, the research logs are
   public, the founder has a verifiable track record (AWS, Valkey GLIDE, a research
   site whose motto — "evidence before adjectives" — is already the brand voice).
   Scrappy single-founder shops earn dev-audience attention exactly this way (HN norms;
   RESEARCH-COMPETITORS.md).
5. **Single-SKU is a feature, not a poverty signal — when framed as depth.** The site
   never apologizes for one model; it shows what "supported" means here: day-one
   bring-up runbook, per-model gate battery, published deployment bar (≥1.1x llama.cpp
   end-to-end before a model is called supported).

What darklanes does NOT position on (explicit anti-goals): cheapest-per-token (the
floor war is lost to fleets with scale; the or-provider report measured the hy3 floor
falling ~35% in 90 days), biggest model menu, enterprise compliance theater (no SOC2
badge yet — say so honestly), and raw speed leaderboards against Groq/Cerebras ASIC
silicon (compete on determinism-at-speed, not speed alone).

## 3. Competitive landscape (evidence: RESEARCH-COMPETITORS.md, fetched 2026-08-04/05)

One-line teardowns; the full detail with verbatim headlines, prices, and access dates is
in RESEARCH-COMPETITORS.md.

**The big fleets:**

- **Together.ai** ("Build what's next on the AI Native Cloud") — moved upmarket to
  full-stack cloud; per-token table pushed into docs; trust carried by a homepage
  research wall with named authors and a **per-model public uptime** status page.
  Notably: Qwen3-32B class has rotated OFF its serverless menu.
- **Fireworks.ai** ("Own your model. Own your future." under a Jensen Huang quote) —
  authority-play homepage, abstract strategy copy ("compounding specialized
  intelligence"); pricing by size-band catchalls ($0.90/1M >16B dense, $0.50/1M MoE
  ≤56B) with Standard/Priority/Fast serving tiers. The four's cautionary example of
  low information density.
- **Groq** ("the premier neocloud for fast inference") — hardware-manifesto homepage,
  zero code or prices on the marketing site; **public pricing page deleted** (redirects
  home; prices only in console docs — qwen3.6-27b at $0.60/$3.00, a speed premium).
  Cleanest marketing/dev split; also the cautionary tale on opacity.
- **DeepInfra** ("AI Inference — FAST SIMPLE RELIABLE LOW-COST") — the price sheet AS
  the website: live model cards with $/M in/out/cached + quantization + context above
  the fold; tier multipliers (Priority 1.5x, Flex 0.8x); SOC2/ISO badges; the price
  floor of the market (Qwen3-32B $0.08/$0.28).

**The small shops and the marketplace:**

- **Novita.ai** — templated full-stack-cloud site rescued by day-0 model support speed
  (HF CTO testimonial: "often the first to get stable, production ready inference
  support online – often on Day One") — direct validation of darklanes' Qwen3.8
  day-one strategy.
- **Hyperbolic** — GPU-rental-led; STALE inference pricing panel (no Qwen3 family at
  all) and 4-hop broken docs redirects — the anti-pattern exhibit for unmaintained
  marketing surfaces.
- **Parasail.io** — the best small-provider site: founder-direct FAQ voice, explicit
  "**no hidden quantization**" positioning, four clear products, full public per-token
  table + parameter-band batch grid. Closest existing analog to what darklanes should
  build.
- **Featherless.ai** — differentiated by flat-rate subscription instead of per-token;
  proof that breaking the pricing template is itself memorable.
- **Chutes.ai** — crypto-native; its free tier was destroyed by >10k bot signups and
  killed — the free-tier cautionary tale.
- **kluster.ai / inference.net** — pivoted/exited mid-2026; small providers vanishing
  is a live buyer fear that darklanes' public engine + receipts partially answer.
- **OpenRouter** (the marketplace lens) — providers are rendered as public telemetry:
  per-endpoint quantization tags, prices, uptime (5m/30m/1d), TTFT + throughput
  percentile charts. Routing is price-weighted; uptime <95% gets deprioritized;
  guidance explicitly favors early 429s over queueing and immediate streaming with SSE
  keep-alives — memra's admission-shed design and serve surface already match the
  marketplace's ideal-provider shape.

## 4. What the winners share / what looks templated

**Shared by everyone credible (= table stakes; absence reads amateur):**
status page, OpenAI-compatible API pitch, public per-token pricing somewhere, tiered
serving (priority/flex/batch multipliers), trust-center or security page, a model-launch
blog cadence, Discord or equivalent community door.

**What actually distinguishes (each winner owns ONE concrete proof device):**
Together's named-author research wall + per-model uptime; Groq's per-model tokens/sec in
docs; DeepInfra's per-card quantization disclosure; Parasail's "no hidden quantization"
+ founder-direct voice; Novita's day-0 support speed; Featherless's template-breaking
price model. The pattern: **the differentiator is always a verifiable artifact, not a
slogan.** darklanes' verifiable artifact is the receipt ledger — stronger than all of
the above because it's raw data, not curated stats.

**What looks templated/forgettable:** stat-card heroes (2x/60%/90%), abstract
strategy-deck copy (Fireworks), generic adjective strings ("FAST SIMPLE RELIABLE"),
MUI/stock component look, enterprise Solutions/Industries page trees, testimonial walls
without numbers. Two live failure modes to design against: stale pricing panels
(Hyperbolic) — solved by generated, CI-checked surfaces (§13); and deleted/hidden
pricing (Groq, Together's docs-only table) — solved by DeepInfra-style above-the-fold
transparency.

**Quantization disclosure is an emerging trust axis** (Together and DeepInfra print
quant per model; Parasail markets the absence of hidden quant; OpenRouter tags every
endpoint). darklanes serves a documented 8-bit arm with bit-audited kernels — the
strongest possible version of this axis; the Models page must state precision arms
explicitly and link the format-decision receipt.

## 5. Naming verdict: keep "darklanes"

**Verdict: darklanes is a strong name. Keep it.** Rationale:

1. **It names the mechanism.** The brand story is literally the product's scheduler:
   protected interactive lanes, dark lanes for batch. No inference provider has a name
   that maps to an architectural claim (Together = vague collectivism, Fireworks =
   generic energy, Novita/Hyperbolic = abstract). A visitor who learns why the company
   is called darklanes has already understood the product.
2. **The finance analogy is respectable, not shady:** dark pools — venues where large
   trades execute without moving the visible market — are the exact semantics of the
   harvest lane (big batch jobs that don't move interactive p95). Use this analogy once,
   in the About/lanes explainer, to inoculate against "dark = sketchy" readings.
3. **Collision-free:** web search (2026-08-05) finds no company/product named darklanes;
   nearest neighbor is Darklang (different word, different space). All key domains are
   unregistered — darklanes.ai/.dev/.com/.io/.cloud/.run all AVAILABLE (receipts:
   DOMAINS.md).
4. **It commits to the visual identity the audience already prefers:** dark-first design
   is the dev-tool norm (RESEARCH-DESIGN.md §3), and here it's brand-motivated rather
   than trend-following.
5. **Risks, stated honestly:** (a) "dark" carries darknet/dark-pattern adjacency for
   non-dev audiences — acceptable, because the audience is developers and the finance
   analogy plus instrument-panel aesthetics read as engineering, not edginess;
   (b) plural "lanes" invites the singular-domain typo — darklane.ai is also free;
   register it if cheap, otherwise accept.

Domain recommendation: **darklanes.ai primary** (the category convention — together.ai,
fireworks.ai, novita.ai, openrouter.ai; gateways and buyers pattern-match on .ai), plus
darklanes.com and darklanes.dev defensively. Owner sign-off required for purchase
(§12).

Alternatives were NOT explored further because the research argues the name is strong;
per the brief, alternatives are only warranted if evidence says it's weak. It doesn't.

## 6. Brand system

### Voice and tone

The voice already exists — it's the memra README and avifenesh.ai. Codify it:

- **Evidence before adjectives.** Never "blazing fast"; always "170.6 tok/s
  single-stream, N=5 median, raw log linked." Every number states its N and conditions
  or links to a page that does.
- **State the limits in the same breath as the wins.** The site publishes the ~7%
  first-token near-tie drift class and the cold-vs-warm TTFT distinction unprompted.
  This is the trust engine: a page that admits what it can't do is believed about what
  it can. (Heavybit: devs hate spam, not marketing — transparency about limits is the
  antidote. RESEARCH-DESIGN.md §1.)
- **First person singular is allowed.** "I run darklanes" — founder-run is a
  differentiator for this audience, not a weakness. No royal "we" theater; "we" only
  where it's literally the engine + the founder.
- **No emojis, no exclamation marks, no "🚀 launch" language.** Sentence case
  everywhere, including headlines and buttons.
- **Verbs of measurement, not verbs of magic:** measured, gated, receipted, published,
  bounded, refuted. Banned words: blazing, magical, supercharge, unleash, effortless,
  revolutionary, game-changing.
- Micro-copy pattern under CTAs: the friction-removers ("no credit card", "OpenAI SDK
  compatible — change one line"), per daily.dev evidence.

### Color

Dark-first, near-black substrate, ONE accent (RESEARCH-DESIGN.md §3 — the proven
formula; Vercel proves even zero-accent works, so one accent is the maximum).

- Background: near-black, not pure #000 — e.g. `#0A0B0E` surfaces, `#101218` raised
  panels. Thin 1px borders `#23262E`.
- Text: `#E6E8EC` primary, `#9BA1AC` secondary. Contrast ≥ 4.5:1 everywhere (also an
  OG-card requirement).
- **Accent: a single "lane-marker" hue.** Recommendation: amber/signal-yellow
  (`#F5B942` family) — reads as instrument-panel/roadway marking, avoids the
  overused dev-tool purple/violet cluster and the green already owned by NVIDIA and
  terminal clichés. Used ONLY for: the active lane in diagrams, primary CTA, live
  numbers. Secondary chart hues (for p50/p95/p99 series) stay desaturated blues/grays.
- Semantic colors reserved for receipts: green strictly for gate PASS states, red
  strictly for FAIL/refuted — never decorative, so their appearance always means
  something (this mirrors the repo's ALL GREEN language).
- Light mode: not required at launch (dark-first is brand-motivated); ensure code blocks
  and OG cards test legibly in both light and dark FEED contexts.

### Typography

- **Sans + mono pair, mono for anything measured.** Recommendation: Geist Sans + Geist
  Mono (purpose-built for dev tools, OFL-licensed, free — RESEARCH-DESIGN.md §2) or
  Inter + JetBrains Mono as the fallback pair.
- Rule with teeth: **every numeral that represents a measurement is set in the mono
  face, tabular figures** — tok/s, p95, TTFT, prices, request counts. Prose numerals
  stay in the sans. This single rule creates the "instrument panel" feel without any
  decoration.
- Headlines: sans, medium weight, sentence case, tight leading. No gradient-text
  cliché (the Linear-Look element that's now most worn out).

### Motifs (the visual language)

- **Lanes:** thin horizontal rules as a recurring layout device; feature panels are
  "lanes" separated by 1px lines. The hero diagram (§7 home) shows two lanes:
  interactive (accent, steady) and harvest (dim, dense) sharing one GPU band.
- **Latency-graph aesthetics:** percentile charts (p50/p95/p99) as decoration-that-is-
  data. Real charts from real receipts only — no illustrative fake curves, ever
  (a fake curve on this brand is self-refuting).
- **FIG-labeling:** technical panels carry schematic labels ("fig. 01 — admission
  gate") — the Linear device (RESEARCH-DESIGN.md §2), which dresses marketing as an
  engineering document. It matches the working-paper aesthetic of avifenesh.ai.
- **Terminal blocks:** real, runnable curl/SDK snippets with copy buttons — never
  screenshots of terminals.
- Logo direction (owner sign-off — §12): wordmark "darklanes" lowercase in the mono
  face, with the "l" or the "k" carrying a subtle lane-divider mark; or a minimal glyph
  of three horizontal lane lines with one lane highlighted. Avoid: shields, hexagons,
  brains, sparkles, anything gradient-blobbed.

## 7. Site architecture — page-by-page

Global chrome: header = wordmark, Models, Pricing, Docs, Receipts, Blog, GitHub icon
(with star count), single primary CTA "Get an API key". Footer = status link (separate
infra — RESEARCH-DESIGN.md §4), contact, privacy/data policy (required for gateway
listings — EVIDENCE-REPO.md §5-6), avifenesh.ai founder link, X/HF/GitHub.

Total public surface at launch: **6 pages + docs + status.** Nothing else. Resist the
enterprise-page temptation (Solutions/Industries/Partners pages are the "templated"
smell — §4).

### 7.1 Home

**Message:** deterministic speed, provable in one scroll. **Audience:** a developer
with an eval pipeline, an agent product, or a batch job, deciding in ~3 seconds whether
this is real.

Sections, in order:

1. **Hero.** Headline (3 candidates in §8) + subhead + stat strip + terminal.
   - Stat strip (the Fly.io device, mono, four numbers): `170.6 tok/s single-stream ·
     0.18 s TTFT cold / 3 ms warm · 464,870 req / 0 errors / 140 min soak ·
     byte-identical c=1 vs c=16`. Each number is a link to its receipt.
   - Terminal above the fold: a curl to `/v1/chat/completions` with
     `"model": "qwen3.8-27b"` and a visible `seed` param + a comment line
     `# same seed, same tokens — gated, not promised`. Copy button. One tab for
     Python (openai SDK, `base_url` swap — "change one line").
   - Dual CTA: **"Get an API key"** primary, **"Read the receipts"** secondary
     (links to /receipts, not docs — the differentiated path). Micro-copy: "OpenAI SDK
     compatible · free evaluation credits" (credits amount = owner call §12).
2. **The determinism block ("fig. 01 — the exactness contract").** Three short columns:
   kernel bit-audit (every kernel vs CPU reference), serving isolation (byte-identical
   alone vs full batch, replayed and compared), spec-decode identity (speculative output
   token-identical to plain decode). Each column ends with a mono link to the actual
   gate log. One honest line under the block: "Known bounded exception: near-tie
   first-token drift across prime configs, ~7% on a 144-prompt sweep, documented here."
3. **The lanes block ("fig. 02 — dark lanes").** The two-lane diagram: interactive lane
   holds p95 under a c=96 batch flood (real numbers: 7.15 s lane-blind → 3.69 s → 2.16 s
   at the tight SLO dial); harvest lane = same GPUs, discounted tokens, sheds instead of
   queueing. One sentence on the dark-pools analogy. CTA: "How lanes are priced →"
   (/pricing).
4. **Speed block.** The spec-decode numbers with the llama.cpp comparison posture the
   repo already uses (best-vs-best, same rig, interleaved). Keep it to 2-3 cells + a
   link to the full published board. Never a wall of comparison tables (repo posture:
   numbers are regression tracking, not a scoreboard).
5. **Built in the open.** GitHub card (memra repo, stars, license), one line on the
   founder with link to avifenesh.ai, the three most recent blog/receipt entries.
6. **Bottom CTA repeat** (devs read whole pages — RESEARCH-DESIGN.md §1): API key +
   pricing links.

**What NOT to put on home:** customer logos (there are none — never fake social proof),
model-menu grids, uptime SLA percentages not yet earned, roadmap promises, newsletter
modal, chat widget, cookie-consent theater beyond the legally minimal.

**Acceptance criteria:** headline + one measured number + runnable snippet + primary
CTA all visible at 1280×800 without scrolling; every number on the page is a working
link to a receipt; page weight < 1 MB, LCP < 1.5 s on fast 3G emulation, zero
third-party trackers; reads correctly with JS disabled (charts may degrade to static
SVG).

### 7.2 Models (+ live board)

**Message:** one model, and here is exactly what "supported" means.

- The SKU card: Qwen3.8-27B (or 3.6-27B until the drop): precision arms served (Q8_0 /
  FP8-ST exact arm), context length, spec-decode on by default, tools/json_schema
  support flags, price summary, the OpenAI model id string.
- **"What supported means here" panel** — the differentiated content: the gate battery
  a model passes before it's listed (kernel-check ALL GREEN, argmax MATCH, spec K=1..8
  self-consistency, serve isolation gate), the ≥1.1x deployment bar, link to the
  day-one bring-up runbook for Qwen3.8. This turns the thin menu into a depth story.
- **Live board:** rolling TTFT and tok/s percentiles (p50/p95/p99) from production
  probes, honest N and window, plus the current gate status (last battery run + hash).
  Publish tail percentiles prominently — competitors market p50; OpenRouter teaches
  buyers to shop p99 (RESEARCH-DESIGN.md §4). If live infra isn't ready at launch,
  ship the latest receipted battery as a dated static board — never a fake-live widget.
- Deprecation/versioning promise: model ids are pinned, `system_fingerprint` carries
  the engine build, changes announced with dates.

**NOT here:** "coming soon" model lists (a single "request a model" mailto is enough),
speculative hardware announcements.

**Acceptance:** board data timestamped and sourced; every support flag on the card maps
to a gated feature in EVIDENCE-REPO.md §4; SKU card renders as a clean OG card when the
page is shared.

### 7.3 Pricing

**Message:** three ways to buy the same tokens, priced by the latency guarantee — and
the meter is honest.

Structure (shape decided here; NUMBERS are owner sign-off, §12, with market anchors in
RESEARCH-COMPETITORS.md):

1. **Interactive lane** — per-token, premium. Hard admission-controlled p95, never
   preempted, spec-decode single-stream speed. This is the headline price.
2. **Harvest lane (the dark lane)** — per-token, discounted (anchor: 40–60% of
   interactive; competitors' "batch" tiers typically run ~50% — verify against
   RESEARCH-COMPETITORS.md). Honest semantics stated on the page: sheds with 429 +
   Retry-After under interactive pressure instead of silently queueing; designed for
   data-gen, evals, distillation.
3. **Dedicated lane** — per-hour/month reserved replica (a whole GPU's worth of the
   SKU), for tenants who want the box to themselves. Anchor to GPU-hour market rates.
4. **Cached input at 25% of input price** — the documented market convention
   (EVIDENCE-REPO.md §5) AND a mechanism darklanes actually meters honestly
   (`cached_tokens` itemized in every response).

Table styling: mono numerals, per-1M-token units, input/output/cached columns, no
"Contact us" cells for the core lanes. Below the table:

- **"How the meter works"** — the honesty block nobody else has: usage carries
  worker-truth token counts on every response including streams; disconnected requests
  billed to the abort point; cached tokens itemized. Link the serve-compat receipts.
- **Free evaluation credits** (recommendation: small one-time credit, e.g. $5–10 — a
  perpetual free tier invites abuse a single-founder fleet can't police, while
  credit-on-signup is the category norm — verify amounts in RESEARCH-COMPETITORS.md;
  owner sets the number).
- FAQ: rate limits (truthful headers), data retention (state the actual policy — needed
  for OpenRouter/HF listings anyway), what happens at capacity (admission wait vs shed,
  per lane).

**NOT here:** enterprise tier theater, "starting at" asterisks, seat-based anything,
crossed-out fake discounts.

**Acceptance:** a visitor can compute the cost of a concrete workload (e.g. "1M in /
200k out per day, half cacheable") from the page alone; lane semantics (shed vs queue)
stated on the page, not buried in docs; prices match `/v1/models` pricing metadata
exactly (gateways will cross-check — EVIDENCE-REPO.md §6).

### 7.4 Docs (entry)

**Message:** you already know this API; here's the one line that changes.

- Quickstart: API key → `base_url` swap in the openai SDK → first request, in Python /
  JS / curl tabs. Target: under 60 seconds ("time to first hello world" is the
  activation metric — RESEARCH-DESIGN.md §1).
- Then exactly five doc pages at launch: **Determinism** (seed semantics, temp-0
  behavior, what's gated — including the omitted-temperature/seed OpenAI-default
  semantics), **Lanes** (`x-lane` header, shed semantics, retry guidance), **Constrained
  decoding** (json_schema, the no-think interaction, bounded-schema guidance),
  **Caching** (prefix cache, `cache_salt` per-tenant isolation, when you must salt),
  **Errors & limits** (honest 400s on unsupported params, rate-limit headers, drain
  behavior).
- Docs are versioned in the open (public repo or a /docs source dir) — doc PRs are a
  trust signal.

**NOT here:** marketing repeated in doc voice; auto-generated API-reference bloat for
five endpoints.

**Acceptance:** quickstart executes verbatim against production; every documented
behavior corresponds to a gated behavior (docs/SERVING.md is the source of truth);
docs reachable from header on every page.

### 7.5 Receipts (the differentiated page — /receipts)

**Message:** this is the page the whole brand hangs on. An indexed, dated ledger of
every public claim: gate batteries, endurance soaks, perf boards, incident notes.

- Format mirrors the avifenesh.ai evidence ledger: dated entries, each with claim →
  receipt link (raw JSONL/log in the memra repo) → conditions (N, rig, thermal regime).
  Wins AND bounded limitations both listed (the drift class, the serve tax, cold-vs-warm
  TTFT).
- The soak entry, the miscompile catch, and the dogfood-bugs entry seed it at launch
  (BLOG-EVIDENCE.md A/B + EVIDENCE-REPO.md §1-3).
- Later this page absorbs the status history (post-incident notes live here).

**Acceptance:** every entry's receipt link resolves to raw data (not a summary blog);
entries are immutable once posted (corrections append, never edit-in-place).

### 7.6 Blog

Standard chronological blog, RSS, no gating. Post inventory and launch pick: §10.
Styling: working-paper aesthetic (the avifenesh.ai note style), FIG-labeled charts from
real data, code blocks with copy buttons. Comments: none; discussion happens on HN/X —
each post footer links its HN thread once one exists.

**Acceptance:** RSS validates; posts render OG cards per §11; each post's numbers link
to receipts.

### 7.7 About

**Message:** who runs this and why you can trust one person with your inference.

- The founder, plainly: name, face, track record (AWS ElastiCache systems engineer,
  Valkey GLIDE maintainer, the public research record at avifenesh.ai), Tel Aviv.
- Why darklanes exists (the operating model, honestly): serving pays for the hardware;
  the research is the product; the engine is public because the receipts are the moat.
- The name explained (lanes + the dark-pools analogy).
- What darklanes is NOT yet: no SOC2, no multi-region, no 99.99 SLA — with the
  compensating controls stated (public receipts, status page, drain-not-drop deploys).
  This section converts the skeptics the rest of the site attracts.

**Acceptance:** zero stock imagery; the "not yet" list is present and current.

### 7.8 Status (status.darklanes.ai)

Hosted on separate infrastructure/domain (RESEARCH-DESIGN.md §4). Uptime per lane,
current incident banner, history. Third-party hosted (e.g. a standard status provider)
is fine and MORE credible than self-hosted at this size. Link from footer + docs +
pricing FAQ.

### avifenesh.ai linkage (both directions, separate brands)

- darklanes → avifenesh.ai: About page founder link; blog author byline links.
- avifenesh.ai → darklanes: the portfolio's "experimental systems" section adds a
  darklanes card (the serving business built on the bw24/memra apparatus); research
  pages that used the engine footnote "the engine now serves production traffic at
  darklanes.ai".
- Keep brands distinct: avifenesh.ai = personal research record (working papers, "the
  result can be yes, no, or not yet"); darklanes = the service. Shared DNA (evidence
  ledger format, mono-numerals, dark restraint) makes the pair mutually reinforcing
  without merging. The darklanes About page should feel like it was written by the
  person whose site avifenesh.ai is — because it was.

## 8. Homepage headline candidates (3, with rationale)

1. **"Inference with receipts."**
   Subhead: "One open model, served fast and deterministic — every number on this page
   links to a raw log. Same seed, same tokens, alone or under load. Gated, not
   promised."
   *Rationale:* four words, brand-ownable, states the category (inference) and the
   differentiator (receipts) with zero adjectives. Passes the Evil Martians
   specificity test via the subhead + stat strip carrying the numbers. Risk: "receipts"
   is idiomatic — mitigated by the stat strip directly beneath. **Recommended.**

2. **"170 tok/s, byte-identical under load."**
   Subhead: "darklanes serves the Qwen 27B class on Blackwell silicon with bit-audited
   determinism — speculative decode included, receipts published, OpenAI-compatible."
   *Rationale:* the numbers-first pattern the conversion evidence favors most
   literally; unambiguous and falsifiable. Risk: a single perf number invites
   Groq-style speed one-upmanship framing, which darklanes doesn't want to fight on;
   number needs updating whenever the board moves (regenerated surface, same rule as
   the repo's perf blocks).

3. **"One model, served obsessively."**
   Subhead: "Bit-audited determinism, hard interactive p95, and batch that rides the
   dark lanes — from the engine kernels up. Every claim receipted in the open."
   *Rationale:* leads with the single-SKU depth story and the founder-scale honesty;
   most memorable voice, weakest information scent (doesn't say "inference API" without
   the subhead). Best if the owner wants personality-forward.

Owner picks one (§12). All three keep the same subhead elements: model class,
determinism, receipts, OpenAI-compat.

## 9. Pricing-page numbers — market anchors

How the same model class is priced TODAY (live OpenRouter endpoints API + provider
pages, 2026-08-05 — full table in RESEARCH-COMPETITORS.md):

- **Qwen3-32B (nearest dense proxy for the 27B class):** DeepInfra $0.08/$0.28 (fp8,
  41k ctx), Nebius $0.10/$0.30, SiliconFlow $0.14/$0.57 (131k ctx), Groq $0.29/$0.59.
  Only **5 endpoints** — a thin market.
- **Qwen3-30B-A3B:** DeepInfra $0.12/$0.50, Alibaba $0.13/$0.52 — **2 endpoints**.
- **Qwen3.6-27b (the direct SKU-class comp):** Groq prices it at **$0.60/$3.00** —
  a speed premium an order above the DeepInfra floor, proving the class supports
  guarantee-priced tiers. Together and Fireworks have no 27B/32B Qwen row at all
  (catalog rotated; Fireworks covers it via a $0.50–0.90/1M size-band catchall).
- **Gemma-3-27B (the 27B-dense comp):** DeepInfra $0.08/$0.16, Parasail $0.08/$0.45,
  Nebius $0.10/$0.30, Novita $0.119/$0.20, Phala $0.15/$0.46.
- **Llama-3.3-70B (crowded-market reference):** DeepInfra $0.10/$0.32 → Novita
  $0.135/$0.40 → Parasail $0.22/$0.50 → Groq $0.59/$0.79 → Together $1.04/$1.04
  (13 endpoints).
- **Cached input:** priced at exactly 25% of input across the surveyed hy3 endpoints
  (in-repo or-provider report, 2026-08-02); Parasail publishes cache-read at 50% of
  input on Llama-3.3-70B ($0.11 vs $0.22). darklanes meters cached tokens honestly
  either way — pick 25% as the convention.
- **Batch/off-peak convention:** Novita batch = 50% off; Parasail publishes a
  parameter-band batch grid (21–41B: $0.07/$0.22) — supports the harvest-lane discount
  anchor of 40–60% of interactive.

Implications for the owner's number-picking (sign-off §12):

1. The 27B/32B class clusters at **$0.08–0.15 in / $0.16–0.60 out**. A brand-new
   Qwen3.8-27B has a pricing vacuum for days–weeks (the 30B-A3B market has TWO
   endpoints) — day-one support briefly makes darklanes the price-setter, not taker.
2. Do NOT price to the DeepInfra floor: the or-provider report measured the floor
   falling ~35% in 90 days on hy3, and floor-chasing grosses $2–4/hr per saturated
   replica — cost coverage comes from the premium lanes, not the race.
3. Suggested SHAPE (numbers are owner's): interactive lane at the upper-middle of the
   cluster justified by the p95 guarantee + determinism (e.g. mid $0.20s out), harvest
   at ~50% of interactive, dedicated priced from GPU-hour market rates + margin,
   cached input at 25%. Free credits small and one-time (Chutes' bot-killed free tier
   is the cautionary tale; Novita's $0.50 voucher and Hyperbolic's deposit-gated
   rewards are the survivors' pattern).
4. Whatever ships must match the `/v1/models` pricing metadata exactly — OpenRouter
   and HF both consume it programmatically.

## 10. Content strategy — first posts, launch pick, distribution

Five posts, all writable today from committed material (sources: BLOG-EVIDENCE.md):

1. **LAUNCH POST — "The compiler ate my byte: how a bit-identity gate caught an nvcc
   miscompile."** Abstract: nvcc 13.0.88 at -O3 silently dropped one of two adjacent
   byte stores in an FP8 dequant kernel — zeroing the low byte of every block scale.
   Unit tests pass; generation diverges at token 14. The kernel bit-parity gate caught
   it on first run; a 40-line repro is in the repo; the fix is one aligned u16 store.
   Why gates re-run per toolchain, not per commit — and why an inference provider
   should have to show you this class of receipt.
   *Why this is the launch post:* compiler-bug war stories with minimal repros are
   evergreen HN material; it demonstrates the exactness discipline as a story rather
   than a claim; it's vendor-neutral enough to earn goodwill (helps anyone on nvcc
   13.0.88) while being the entire darklanes sales pitch in disguise.

2. **"My own agent found two bugs in my server (dogfood day)."** Abstract: the founder's
   daily agent locked into identical tool-call loops. Cause #1: omitted temperature
   deserialized to 0.0 = greedy. Cause #2, the one that survived the first fix: omitted
   seed deserialized to 0 — a valid *fixed* seed. Both are the same bug class: zero is a
   meaningful value, not "unset." Plus the uncomfortable meta-lesson: every golden test
   ran temp-0, so the sampled path was invisible to the entire gate battery — and the
   new distribution-level composition gate that closes the hole.
   *Slot:* week 1–2 after launch. Relatable, self-deprecating, credibility-compounding.

3. **"Serving is a determinism problem: the batch that changed the model's mind."**
   Abstract: under batched prefill, an m-dependent router GEMM changed which EXPERTS a
   MoE request activated depending on who arrived with it — 16% of (layer,token) pairs.
   The isolation gate (byte-compare c=1 vs c=16) that found it, the m-invariant fix,
   and the resulting contract: your output does not depend on your neighbors. Includes
   the honestly-documented bounded exception (near-tie first-token drift).
   *Slot:* week 3–4; the technical centerpiece for eval/agent audiences.

4. **"Dark lanes: hard p95 for interactive while batch eats the leftovers."** Abstract:
   the noisy-neighbor problem measured (bulk tenant inflates interactive p95 4x), why
   shedding at admission beats queueing in the engine ("the engine's own queue is where
   the tail dies"), per-lane prefill budgets vs the global chunked-prefill knob, and the
   140-minute zero-violation soak. Ends with the pricing consequence: batch is cheaper
   because it yields.
   *Slot:* aligned with any pricing/HF-listing push; this is the brand-namesake post.

5. **"Spec-decode economics on consumer-priced silicon."** Abstract: MTP speculative
   decoding turns a 27B into a 170 tok/s single-stream model on a workstation card —
   with output gated token-identical to plain decode. Where the wins come from
   (acceptance ladders, draft masking under json_schema), where they stop (batch
   crossover at c=2–4), and what that does to $/token on owned Blackwell vs rented
   datacenter fleets.
   *Slot:* week 4–6; the economics audience (and the HN "consumer GPU" perennial).

Plus the news-peg announcement (not a numbered post): **"Qwen3.8-27B, supported
day-one"** — published the day the gate battery goes green per the bring-up runbook,
linking the receipts. Short, factual, no launch theater.

**Distribution:**

- **HN:** launch post submitted as a plain title (no "Show HN" for the compiler story;
  "Show HN: darklanes — deterministic inference with published receipts" is the
  separate product submission, best timed with the Qwen3.8 day-one announcement).
  Founder answers questions in-thread; receipts links do the arguing. (Reddit is
  skipped — the owner's account is spam-filtered sitewide; HN/X/Discord instead.)
- **X/dev-twitter:** each post gets a thread whose first tweet is the single most
  concrete artifact (the miscompile repro diff, the p95 chart), not the link. Charts
  as dark-card images (§11).
- **Hugging Face:** the distribution/discovery channel — a darklanes org page, the
  provider-registration path (OpenAI-compat LLM APIs skip most schema work; Model
  Mapping API; per-request nano-USD billing endpoint with `Inference-Id` headers;
  automated 6-hourly validation incl. tool-calling and structured-output tests — all
  behaviors memra already gates; EVIDENCE-REPO.md §6). A separate lane owns the
  mechanics; the site's Models page and docs should carry the `HF Inference Providers`
  badge/link once live. Note HF's default provider ordering = 7-day routed volume, so
  the listing compounds with actual traffic, not marketing.
- **OpenRouter:** apply (expect the backlog; open-weight models are non-prioritized);
  the realistic near-term value is public perf/uptime charts on a neutral surface,
  i.e. more receipts — treat as distribution, not revenue (EVIDENCE-REPO.md §5).
- **GitHub:** the memra repo README links darklanes.ai once live ("memra in
  production"); release notes cross-post board-moving perf changes to /receipts.

## 11. OG image concept

One template, generated per page (Vercel OG / `ImageResponse` pattern —
RESEARCH-DESIGN.md §5):

- 1200×630, near-black `#0A0B0E`, three thin horizontal lane lines across the lower
  third with the middle lane in the accent amber; wordmark bottom-left, page title
  ≥60 px bold sans center-left (~40 chars max), and ONE mono-set metric top-right
  (per-page: home = "170.6 tok/s · byte-identical", blog posts = the post's headline
  number, models = TTFT/percentiles).
- Contrast ≥4.5:1; verify legibility at 200×105 thumbnail; test on light AND dark feed
  backgrounds (mid-gray dies in dark mode — cited guidance).
- The GitHub repo social-preview (manual upload, Settings → Social preview — no API)
  gets the home variant.

## 12. Owner sign-off required (nothing below ships without it)

1. **Domain purchase** — darklanes.ai (+ .com/.dev defensively; darklane.ai optional).
   Receipts in DOMAINS.md; ~$100–150/yr for the trio.
2. **Headline pick** — §8 (spec recommends #1, "Inference with receipts.").
3. **Pricing numbers** — lane prices, dedicated rate, free-credit amount (§9 anchors;
   structure is decided, numbers are not). Also: the public data-retention policy text.
4. **Logo/wordmark** — direction in §6; final mark is an owner call.
5. **Accent color** — spec recommends lane-marker amber; owner may prefer another hue
   (keep it single-accent regardless).
6. **Launch timing** — coupling the site launch to the Qwen3.8 day-one announcement
   (recommended: site goes live quietly first, HN product submission rides the day-one
   news peg).
7. **Free-tier/credits policy** and abuse posture.
8. **Company/legal identity** shown in the footer (invoicing entity, ToS/privacy
   pages — also prerequisites for OpenRouter/HF listings).

## 13. Implementation notes (for the building agent)

- **Stack:** static-first (Astro or Next.js SSG) + edge functions only for the live
  board and OG generation. No client-side framework payload for prose pages. Charts:
  server-rendered SVG from receipt JSON (the repo already generates SVG perf cards —
  `tools/update-perf-board.py` is prior art and a style reference).
- **The board pipeline is generated, never hand-edited** — same rule as the repo's
  perf surfaces: numbers live in a JSON artifact, pages regenerate from it, CI refuses
  drift. Reuse the current-board.json pattern.
- **Perf budget:** LCP < 1.5 s, total JS < 100 KB on home, zero third-party trackers
  (privacy-respecting analytics only, e.g. self-hosted or Plausible-class; this is
  also a brand statement for the audience).
- **Accessibility:** the dark theme must hold 4.5:1; charts need text alternatives
  (the receipts are text anyway — link them).
- **Deploy:** any static host + the status page on SEPARATE infrastructure. The site
  repo is public (the site itself is a receipt).

## 14. Overall acceptance criteria

1. Every number on every page resolves to a raw receipt (repo file or dated
   third-party source). Automated link check in CI.
2. The five §10 posts exist in draft; the launch post is publishable day one.
3. Quickstart executes verbatim against production; time-to-first-token for a new
   visitor following it < 60 s of active work.
4. Home passes: headline + number + snippet + CTA above the fold at 1280×800; LCP
   < 1.5 s; works without JS.
5. Pricing page lets a visitor compute a concrete workload's cost unaided; numbers
   match `/v1/models` metadata.
6. The "what we are NOT yet" honesty blocks exist on About and Pricing.
7. OG cards render legibly at 200×105 on light and dark feeds.
8. No banned vocabulary (§6 voice) anywhere, including alt text and meta descriptions.
9. Status page lives on separate infrastructure and is linked site-wide.
10. avifenesh.ai linkage implemented in both directions per §7.8.

## 15. Sources

- Repo receipts: EVIDENCE-REPO.md (this directory) — all paths verified 2026-08-05 on
  branch restructure/public-split.
- Competitive teardown with access dates: RESEARCH-COMPETITORS.md.
- Design/conversion evidence with access dates: RESEARCH-DESIGN.md.
- Domain checks: DOMAINS.md (RDAP + registry WHOIS, 2026-08-05).
- HF provider registration: huggingface.co/docs/inference-providers/register-as-a-provider
  (fetched 2026-08-05).
- OpenRouter provider onboarding: research/or-provider-20260802/REPORT.md (in-repo,
  sources fetched 2026-08-02).
