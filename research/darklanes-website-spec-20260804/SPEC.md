# darklanes — launch website spec

Date: 2026-08-05. Lane: research+writing, no GPU. Status: ready for implementation
pending the owner sign-offs in §12.

> ## ⚠️ READ FIRST — numbers and targets live in `docs/PRODUCT-TRUTH.md`
>
> **This spec is NOT the source of truth for any number, target platform, SKU, or
> capability claim.** `docs/PRODUCT-TRUTH.md` in the memra repo is. That file is
> reconciled; this one is a build brief that was written on 2026-08-04/05 and corrected
> on 2026-08-05 against receipts that had moved underneath it.
>
> **Corrected 2026-08-05.** An earlier revision of this spec was handed to a build-agent
> and produced the wrong product — wrong numbers, wrong target platform — because it
> predated seven landings. Everything that changed is listed in **§16 CHANGELOG**, which
> exists so the owner can re-hand the diff rather than the whole brief.
>
> Rules for whoever builds from this:
>
> 1. Before publishing any number, check it against `docs/PRODUCT-TRUTH.md`. If the two
>    disagree, **PRODUCT-TRUTH wins** and this file is stale again — say so.
> 2. `docs/PRODUCT-TRUTH.md` §7 (honest gaps) is **required content**, not optional
>    garnish. The gaps are the trust engine; a site that hides them is off-brand.
> 3. Two things this spec cannot settle and you must not guess: the **lab name** (§5) and
>    every **price** (§9). Both are owner calls.

**Who this is for:** an implementing agent (or human) with ZERO prior context on this
project. Everything you need is in this directory:

- `SPEC.md` (this file) — the build brief.
- **`docs/PRODUCT-TRUTH.md` (in the memra repo) — the live source for every number,
  target, SKU, capability, and gap. Read it before this file.**
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

**Entity architecture (owner decision, 2026-08-05) — read this before anything else.**
The serving brand sits **under** a parent research lab. There are three layers:

| Layer | Entity | Status |
|---|---|---|
| Parent | **the research lab** | **NOT NAMED YET — owner call, §5. Blocks both sites.** |
| Engine / research record | **memra** — the public Rust+CUDA engine repo | Settled |
| Serving product | **darklanes** — the QoS-lane serving business | Settled as the *product* name |

Consequences that bind every page of this spec:

- **Never write "darklanes lab", "darklanes research", or "darklanes doctrine."** Until
  the owner names the parent, copy says "the lab" or "our research lab" generically, and
  attributes engine/research properties to **memra** (the engine), not to darklanes (the
  product). The owner's concern is explicit and it is the reason this section exists: a
  research lab carrying the "dark" connotation is a liability the serving product can
  absorb and a lab cannot.
- **Two surfaces, not one site** (§7): the **lab site** carries the research record,
  receipts, blog, operating model, and an embedded products section; a **separate
  inference landing page** is the GTM asset (conversion, get-a-key, pricing).

**darklanes** is the GPU-inference product of a one-person lab (Avi Fenesh, Tel Aviv —
systems engineer at AWS ElastiCache, maintainer of Valkey GLIDE, independent ML
researcher at avifenesh.ai). It serves open-source models on NVIDIA Blackwell hardware,
powered by **memra** — a from-scratch Rust+CUDA inference engine, public on GitHub, with
an unusual discipline: every kernel is bit-audited against a CPU reference, every serving
mode is gated against a named reference, and every published number has its raw run logs
committed in-repo.

- **First SKU:** one model — the Qwen 27B class at 8-bit (Qwen3.6-27B today, served as
  Q8_0 GGUF with an NVFP4+MTP arm for the speculative fast lane). A day-one bring-up
  runbook for Qwen3.8-27B is written and ready ahead of its expected release (the
  publicly signalled window is the week of 2026-08-10; the date is not ours to promise,
  and the 3.8 architecture is unpublished, so **"day-one support guaranteed" is not
  claimable** — "runbook ready" is). Single-digit SKU count is deliberate: one model,
  served obsessively, beats a thousand-model menu on every quality axis a small fleet can
  control.
- **Measured serving numbers** — RTX PRO 6000 Blackwell 96 GB, **188 SM, rented pod**,
  2026-08-04, all receipted (EVIDENCE-REPO.md §1; canonical in
  `docs/PRODUCT-TRUTH.md` §2.1): MTP speculative decode **170.5 tok/s single-stream
  through the serving surface** (186.7 bare CLI), **420.6 tok/s aggregate at c=8**, TTFT
  **0.182 s cold / 3 ms warm** (prefix-cache hit). All are N=5 or N=3 medians. Two
  caveats that must travel with them on the site: **c=8 is the knee** — c=16 and c=32 add
  no throughput while p50 doubles at each step, so c=8 is the number, not a ceiling — and
  **the board is rented**; nothing in that class is owned yet (§below).
- **Endurance:** a 140-minute soak on a rented 8x H100 fleet served **464,870 requests
  with 0 errors and 0 sheds**, throughput drift **+0.045%**, p95 drift −0.4 ms. A greedy
  probe hashed **identical on all 8 replicas before and after** the soak. Precise
  wording matters here: the soak's own load ran at temperature 0.7 seeded, and the
  determinism hash is a *separate* pre/post greedy probe — say "a greedy probe hashed
  identical on all 8 replicas", never "464,870 deterministic requests." The soak model is
  a 9B-class Q8_0, not the 27B SKU: call it "9B-class fleet endurance."
- **The namesake mechanism:** QoS lanes. Requests tag `interactive` (protected — hard
  p95, never preempted), `judge`, or `harvest` (batch — rides spare capacity, shed at
  admission with 429 + Retry-After, never queued inside the engine). Interactive and
  batch traffic share the same GPUs without the batch tenant destroying interactive tail
  latency. Measured: a c=96 bulk tenant inflates a lane-blind fleet's interactive p95 by
  ~4x; admission-controlled lanes plus a correctly sized queue hold contended p95 at
  3.69 s where a lane-blind fleet sits at 7.15 s, and at the tight SLO dial contended p95
  becomes statistically equal to an uncontended box while batch pays 67% of its
  throughput. **Attribution honesty, non-negotiable:** roughly half of that 7.15 → 3.69 s
  is the proxy queue cap, not the lane mechanism (cap alone gets 7.15 → 4.34 s; the lane
  gate does 4.34 → 3.69 s). Copy that credits the whole move to lanes is overclaiming.
- **Serve surface:** OpenAI-compatible (validated against the official `openai` SDK),
  streaming, tools/function-calling, REAL constrained decoding (`json_schema` via
  llguidance at 99.4% of unconstrained speed **on the plain lane** — the spec lane pays
  far more, 79%, so never quote 99.4% for speculative decoding), cross-request prefix
  caching with per-tenant `cache_salt` isolation, honest usage accounting (cached tokens
  itemized, aborted requests billed to the abort point), and — **new since this spec was
  first written** — **API keys with per-tenant auth**: per-key tenant/lane/rate-limit,
  hot revoke ≤2 s, per-tenant cache namespacing, and a batch-class key refused 403 on the
  interactive lane. The get-a-key flow and the per-tenant pricing story both depend on
  this and can now assume it exists.
- **Business posture (owner-set):** serving revenue covers hardware; research is the
  product. Success bar = cost coverage + public reliability stats, NOT market-share
  wins. "Everything need to be honest, nothing is a dream that we try to make true."
  The site must never promise scale it doesn't have; it converts on trust density, not
  breadth. **This narrative belongs on the lab site, not the inference landing page** —
  a prospective customer does not need to know their tokens fund GPU purchases.
- **Hardware, stated exactly** (owner override 2026-08-03): the **owned** trajectory is
  **RTX PRO 6000 Blackwell class, homogeneous** — not 2x RTX 5090, which was explicitly
  rejected on scaling-continuity grounds. Today the only owned GPU is an RTX 5090
  *laptop* (the local proof rig); every PRO 6000 and H100 number above came from a
  **rented** pod, and the first purchase is earnings-gated. The site may say "measured on
  RTX PRO 6000 Blackwell (rented pod); the owned build-out targets the same silicon." It
  may **not** say "we run on PRO 6000" unqualified, and it may not call the 5090 the
  deployment target (superseded) or dead (false — it remains the rental measurement
  platform and the small-SKU/customer-site shape reference).
- **What is NOT true yet, and must appear on the site** (`docs/PRODUCT-TRUTH.md` §7): the
  serving path currently runs **~9-12% slower than the naked CLI at c=1** (root-caused,
  filed, being fixed); **llama.cpp currently wins cold time-to-first-token** (0.19 s vs
  0.53 s), short agentic turns, and raw prefill on a same-artifact head-to-head, so **no
  surface may claim interactive-latency superiority**; there is no SOC2, no multi-region,
  no 99.99% SLA, and no tensor parallelism.

## 2. Positioning (the choice, and why)

**Chosen positioning — INVERTED 2026-08-05 (owner call): "commitments you can verify
yourself," not "the provider that shows its work."**

The inversion, and why it matters. "We show our work" is provider-centric: it asks the
buyer to admire our logs. It sells our diligence. Inverted, the same evidence becomes the
buyer's instrument: **every commitment darklanes makes is one the customer can verify
from their own client, without trusting us.** Same seed, same tokens — run it twice and
compare. Cached tokens itemized — check the meter against your own prompt. A published
`system_fingerprint` — pin it and detect the day it changes. Our receipts stop being a
trust *appeal* and become a trust *mechanism*: the customer never has to believe us,
because they can check.

This is a stronger position for the same evidence, and it is the only honest one for a
one-person lab: we are not asking for the benefit of the doubt that a big fleet's brand
buys. We are removing the need for it.

Copy consequences:

- Lead with what the *customer* can do ("verify it yourself in two requests"), not with
  what we did ("we bit-audit every kernel"). The audit is the *reason* the commitment
  holds; it is support, not headline.
- Every commitment on the site ships with **its verification recipe** — the two-request
  curl, the field to diff, the header to pin. A commitment without a recipe is an
  adjective.
- Receipts remain load-bearing, but move a layer down: they answer "why should this
  work?" after the customer has already seen "here's how you check."

One paragraph, usable verbatim in briefs: *darklanes is a boutique GPU-inference product
serving one open model class, obsessively. Where the big fleets ask you to trust a brand,
darklanes makes commitments you can check from your own client: same seed and same
tokens, whether your request runs alone or inside a full batch; a usage meter that
itemizes what was cached; an engine build id you can pin. Each of those is verifiable in
two requests, and each traces to a raw log in the public engine repo. It is built for
eval pipelines that must reproduce, agent loops that must be debuggable, and mixed
workloads that need hard interactive p95 while batch work rides the same GPUs — the dark
lanes.*

Why this wins attention against the big fleets (each hypothesis tested against
evidence):

1. **Determinism/reproducibility is an unserved, named pain — and it is customer-checkable.**
   OpenRouter — the marketplace lens on the whole industry — teaches buyers that
   undisclosed quantization is "the hidden quality variable" and publishes per-provider
   percentile charts precisely because providers are not trusted (RESEARCH-DESIGN.md §4).
   Eval and agent builders currently cannot get a "same tokens every time" guarantee
   from any listed provider. memra has it as a gated contract, not a best-effort
   (EVIDENCE-REPO.md §2). Nobody else CAN market this easily: it requires owning the
   engine down to kernel reduction order. **The exact commitment, stated correctly:**
   greedy output is byte-identical whether a request runs alone or inside a full batch —
   verified by replaying the same prompts at c=1 and c=16 against the same server and
   byte-comparing every stream. It is **not** a claim of one canonical output across every
   configuration; see §2a for the wording rules, which are strict.
2. **Receipts convert; adjectives don't.** The one published A/B with numbers (Evil
   Martians) moved conversion 0.1% → 2.0% by replacing vague claims with specific
   metrics (RESEARCH-DESIGN.md §1). The lab's evidence discipline — raw JSONL next to
   every summary, which darklanes inherits — is a conversion asset the incumbents'
   marketing teams cannot replicate without changing how their companies work.
3. **The lanes story monetizes a real trade-off instead of hiding it.** Every provider
   suffers the noisy-neighbor problem; none sell the solution explicitly. "Your
   interactive p95 is protected by admission control, and batch tokens are cheaper
   BECAUSE they yield" is both a differentiated mechanism and an honest price structure
   (§9). **Scope correction (2026-08-05): harvest is a PRICE TIER, not a product.** It is
   the discounted lane on the same endpoint and the same SKU — one header, one price
   column. Do not give it a product page, a product card, or a slot in a
   three-products lineup: there is one product (inference on one model) sold with a
   latency-guarantee dial. Overbuilding harvest into a second product implies a second
   pipeline we do not have and invites questions about capacity we cannot answer.
4. **Built-in-the-open credibility.** The engine is public, the research logs are
   public, the founder has a verifiable track record (AWS, Valkey GLIDE, a research
   site whose motto — "evidence before adjectives" — is already the brand voice).
   Scrappy single-founder shops earn dev-audience attention exactly this way (HN norms;
   RESEARCH-COMPETITORS.md).
5. **Single-SKU is a feature, not a poverty signal — when framed as depth.** The site
   never apologizes for one model; it shows what "supported" means here: a day-one
   bring-up runbook, a per-model gate battery, and a published deployment bar of **≥1.1x
   end-to-end** before a model is called supported. *Wording care:* that bar is defined
   against **frozen** llama.cpp reference numbers recorded through 2026-08-03 (llama
   benching stopped that day; all forward work is self-competition). State it as "our
   ≥1.1x end-to-end deployment bar" — do not present it as a live, ongoing competitive
   comparison, and do not pair it with a current-day llama claim (§2a).

What darklanes does NOT position on (explicit anti-goals): cheapest-per-token (the
floor war is lost to fleets with scale; the or-provider report measured the hy3 floor
falling ~35% in 90 days), biggest model menu, enterprise compliance theater (no SOC2
badge yet — say so honestly), raw speed leaderboards against Groq/Cerebras ASIC
silicon (compete on determinism-at-speed, not speed alone), and — **added 2026-08-05** —
**interactive latency versus llama.cpp**, which we currently lose (§2a).

## 2a. Claim-wording rules (STRICT — these break the brand if broken)

Every rule below has a receipt behind it in `docs/PRODUCT-TRUTH.md`. A customer can
reproduce the counterexample to a broken version in minutes, and on a
verify-it-yourself brand a falsifiable overclaim is fatal — worse than saying nothing.

**The determinism commitment.** Say:

> Same seed, same tokens — whether your request runs alone or inside a full batch.

Never say: bare "byte-identical" with no object; "byte-identical to single-token
reference decode"; "token-identical to the CLI"; "one canonical output"; "deterministic"
unqualified. The gate compares the same server at c=1 vs c=16 and passes 16/16 at
defaults. It does **not** establish identity against a tokenwise oracle — that path has a
known, bounded, documented near-tie flip class. Scope the claim to the batch object and
it is bulletproof; unscope it and it is false.

**Latency vs llama.cpp.** A same-artifact head-to-head on 2026-08-05 found **llama.cpp
ahead on cold TTFT (0.19 s vs 0.53 s), short agentic turns, and raw prefill**; memra ahead
on long-generation sampled throughput (+17%) and 4k-context decode. Until the serve-path
work lands, **no page may claim interactive-latency superiority**. Speed copy leads with
sustained/long-generation throughput, or with the spec-decode ratio against *our own*
plain decode (2.17x on the SKU, 2.61x on the official FP8 checkpoint) — self-competition,
which is also the standing repo posture since llama benching stopped on 2026-08-03.

**Precision formats.** Q8_0 GGUF is the shipping 8-bit serving arm. FP8 checkpoints
*load* bit-exact and serve (dequantized to that arm), and GPU-side dequant cuts the load
wall 2.89x. Do not say FP8 is "the" serving format, that an FP8-native compute path is on
by default (both exact arms are off), or that FP8 is faster at inference (the win is load
time; the resident e4m3 arm is flat by construction).

**Every performance number** carries its rig label and its N. The headline board is a
**rented** RTX PRO 6000 Blackwell pod at 188 SM — not a 5090, and not owned hardware. The
2.6x official-FP8 spec figure is from a **rented 2x5090** box, not the PRO 6000.

**Aggregate throughput** is quoted at **c=8**, the knee. c=16/c=32 add nothing while p50
doubles per step; publishing them as a ceiling misrepresents queueing as capacity.

**The QoS result** credits the proxy cap for roughly half of the p95 improvement (§1).

**Constrained decoding at 99.4%** is the plain lane only; the spec lane runs at 79%.

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

## 5. Naming — TWO decisions, and the lab one blocks everything

**Owner input, 2026-08-05: "calling a lab darklanes might be problematic"** and "the
serving brand should sit under the lab name. we should name the lab first."

That splits the naming question in two, and reorders them:

- **5A. The LAB name — UNDECIDED, first-order, blocks both sites.** The lab is the parent
  entity. It carries the research record, the receipts ledger, the blog, and the About
  story; the serving product sits inside it.
- **5B. The PRODUCT name — settled: darklanes.** The verdict below (§5B) stands, and the
  owner's concern actually *strengthens* it: "dark lanes" is an excellent name for a
  batch-yields-to-interactive scheduler and a poor one for a research institution. Scoping
  it to the product resolves the tension rather than costing anything.

### 5A. Lab-name decision brief (owner picks; nothing purchased)

Requirements a lab name must meet: it fronts *research*, so it must survive being cited in
a paper, a HN thread, and a hiring conversation; it must not inherit the "dark"
connotation the owner flagged; it must pair cleanly with "darklanes" as a product beneath
it; and — since the lab's whole credibility asset is an existing public repo — it should
ideally cost **zero** new brand-building.

**Domain availability, RDAP-checked 2026-08-05. Nothing purchased.**

| Domain | Status |
|---|---|
| `memra.ai` | **TAKEN** (registered 2025-08-20) |
| `memra.dev` | **TAKEN** (registered 2025-08-22) |
| `memra.com` | **TAKEN** (registered 2003) |
| `memra.io` | AVAILABLE |
| `memralab.ai` | AVAILABLE |
| `memralabs.ai` / `memralabs.com` | AVAILABLE |
| `memraresearch.ai` | AVAILABLE |
| `darklanes.ai` / `.com` / `.dev` | AVAILABLE |
| `lanelab.ai` | AVAILABLE |
| `receiptlab.ai` | AVAILABLE |
| `evidencelab.ai` | TAKEN (2025-11-21) |

That `memra.ai` and `memra.dev` are both gone — and both registered in August 2025, i.e.
recently and by someone else — is the single most decision-relevant fact here. Option 1
below is otherwise the obvious answer, and this is its one real cost.

**Option 1 — "memra" as the lab identity** (standing candidate; recommended).
`memra` becomes the lab/research name; the engine keeps the name it already has;
darklanes is "the inference service by memra."
*For:* zero new brand-building — the public repo, the crates, the release history, the
`system_fingerprint` string, and every receipt already say "memra", so the lab's track
record is instantly legible; the engine-lab identity is genuinely the same thing here;
cleanest possible story ("the lab builds memra; memra serves darklanes").
*Against:* `memra.ai`/`.dev`/`.com` are all taken, so the lab site lands on `memra.io`,
`memralab.ai`, or `memralabs.com` — slightly weaker than the product's own
`darklanes.ai`, which is odd for a parent entity. Also collapses engine and lab into one
word, so a future second product has no neutral parent to hang from.

**Option 2 — a new neutral lab name, memra stays the engine, darklanes stays the product.**
Three layers, three names.
*For:* maximum structural clarity and the most room to grow — the lab can ship a second
product without renaming anything; a neutral parent is the easiest entity to put on an
invoice, a paper, or a hiring page; free choice of an available `.ai`.
*Against:* a third brand to build from zero, with no existing audience, at exactly the
moment attention is scarce; the owner is one person, and three brands is a lot of surface
for one person to keep coherent. This is the honest cost, and it is not small.

**Option 3 — the founder as the lab: `avifenesh.ai` is the parent.**
The existing personal research site becomes the lab surface; darklanes is its product.
*For:* zero new anything — the site, the audience, the "evidence before adjectives" motto,
and the verifiable track record already exist; founder-led is a genuine differentiator for
this audience; sidesteps the naming problem entirely.
*Against:* a personal name does not read as an institution, which weakens the
"lab" framing for partners and hiring; it also permanently couples the business's
credibility to one person's name, which is the thing a lab name is *for*. The spec's own
§7 linkage plan already keeps these brands deliberately distinct.

**Option 4 — "memra lab" / "memra research" as a two-word lab name, engine stays "memra".**
A middle path between 1 and 2: the lab is *named after* the engine without being
identical to it.
*For:* inherits the repo's credibility like Option 1, but keeps a nameable parent
distinct from the artifact, so a second product is possible; `memralab.ai` and
`memralabs.ai` are both available.
*Against:* two-word names get truncated in conversation back to "memra", so in practice it
may collapse into Option 1 anyway; "lab"/"labs" is a crowded suffix in AI right now.

**Recommendation: Option 1, falling back to Option 4 if the owner wants a parent distinct
from the engine.** The repo already *is* the lab's track record; spending brand-building
effort on a third name buys structure the lab does not yet need. Take `memra.io` or
`memralabs.com` for the lab site and let `darklanes.ai` be the product's front door — a
strong product domain under a plainer parent domain is a normal and unremarkable shape.

**Whatever the owner picks, the docs stay generic until they pick it.** Product-facing
copy says "the lab"; engine/research properties are attributed to memra.

### 5B. Product-name verdict: keep "darklanes"

**Verdict: darklanes is a strong name for the serving product. Keep it — scoped to the
product.** Rationale:

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
   non-dev audiences — acceptable **for a product** sold to developers, where the finance
   analogy plus instrument-panel aesthetics read as engineering rather than edginess. It
   is **not** acceptable for the parent lab, which is exactly the owner's 2026-08-05
   objection and why §5A exists; (b) plural "lanes" invites the singular-domain typo —
   darklane.ai is also free; register it if cheap, otherwise accept.

Domain recommendation: **darklanes.ai primary** for the product / inference landing page
(the category convention — together.ai, fireworks.ai, novita.ai, openrouter.ai; gateways
and buyers pattern-match on .ai), plus darklanes.com and darklanes.dev defensively. The
**lab site sits on its own domain** per §5A. Owner sign-off required for every purchase
(§12); RDAP confirms all three darklanes domains still available 2026-08-05.

Product-name alternatives were NOT explored further because the research argues the name
is strong for a product; per the brief, alternatives are only warranted if evidence says
it's weak. It isn't. The **lab** name is the open question (§5A).

## 6. Brand system

### Voice and tone

The voice already exists — it's the memra README and avifenesh.ai. Codify it:

- **Evidence before adjectives.** Never "blazing fast"; always "170.5 tok/s
  single-stream, N=5 median, rented RTX PRO 6000, raw log linked." Every number states its
  N, its rig, and its conditions, or links to a page that does.
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

## 7. Site architecture — TWO SURFACES (restructured 2026-08-05)

**Owner decision:** the lab site is the parent surface with the inference product embedded
in it; the inference landing page is a **separate** GTM asset. This replaces the earlier
single-site plan. The page inventory below is unchanged in *content* — it is re-homed.

### 7.0 Surface split and content mapping

**Surface A — the LAB site** (domain per §5A; lab name owner-pending). The research
record. Audience: peers, press, potential collaborators, hiring, and the technically
curious buyer doing due diligence.

Pages: Home (what the lab is + current work), **Research/Receipts** (the ledger),
**Blog**, **Products** (a section, with the inference product embedded and linking out to
Surface B), **About** (founder, operating model, "what we are NOT yet").

**Surface B — the INFERENCE LANDING PAGE** (`darklanes.ai`). The GTM asset. Audience: a
developer with an eval pipeline, an agent product, or a batch job, deciding in ~3 seconds
whether to try it. Conversion-focused, verify-it-yourself posture (§2).

Pages: Landing (hero + commitments + verification recipes + speed + CTA), **Models**,
**Pricing**, **Docs**, Status. Links *up* to the lab site for receipts and blog rather
than duplicating them.

**Content-to-surface mapping** (this is the table the build-agent needs):

| Content, as specced below | Goes to | Note |
|---|---|---|
| §7.1 Home hero, stat strip, terminal, dual CTA | **B** landing | Becomes the inference landing hero |
| §7.1 determinism block ("fig. 01") | **B** landing | Reframed as commitments + verification recipes (§2) |
| §7.1 lanes block ("fig. 02") | **B** landing | Harvest = price tier, not product (§2 point 3) |
| §7.1 speed block | **B** landing | Self-competition framing only (§2a) |
| §7.1 "Built in the open" | **both** | On B: a short trust strip linking to A. On A: it *is* the site |
| §7.2 Models + live board | **B** | |
| §7.3 Pricing | **B** | |
| §7.4 Docs | **B** | |
| §7.5 Receipts ledger | **A** | The lab's core artifact. B links to it |
| §7.6 Blog | **A** | Author byline = founder; posts are lab output |
| §7.7 About + operating model + "not yet" list | **A** | The "serving pays for hardware" narrative lives here, NOT on B |
| §7.8 Status | **B** (linked from both) | Separate infra either way |
| Lab Home + Products section | **A** | New — not in the original spec; see below |

**New content needed for Surface A** (the only genuinely new writing this restructure
creates):

- **Lab home**: one paragraph on what the lab does (builds inference systems and publishes
  the evidence), the current work, and the receipts/blog/products doors. No pricing, no
  CTA theater.
- **Products section**: the inference product described in two paragraphs with a link to
  Surface B. Written for a peer, not a buyer. If a second product ever exists, it lands
  here — which is the structural reason the lab is the parent (§5A).

Global chrome, Surface B: header = wordmark, Models, Pricing, Docs, Receipts (→ A), Blog
(→ A), GitHub icon (with star count), single primary CTA "Get an API key". Footer = status
link (separate infra — RESEARCH-DESIGN.md §4), contact, privacy/data policy (required for
gateway listings — EVIDENCE-REPO.md §5-6), lab-site link, avifenesh.ai founder link,
X/HF/GitHub.

Global chrome, Surface A: header = lab wordmark, Research, Blog, Products, About, GitHub.
No "Get an API key" CTA in the lab header — that is Surface B's job; the Products section
carries the one link.

Total public surface at launch: **Surface A 5 pages + Surface B 4 pages + docs +
status.** Nothing else. Resist the enterprise-page temptation (Solutions/Industries/
Partners pages are the "templated" smell — §4).

The page specs that follow (§7.1-7.8) are written as originally drafted; apply the mapping
table above to place each one.

### 7.1 Landing — **SURFACE B** (was "Home"; re-homed per §7.0)

**Message:** deterministic speed, provable in one scroll — **by you, not by us**.
**Audience:** a developer with an eval pipeline, an agent product, or a batch job, deciding
in ~3 seconds whether this is real. (Surface A gets its own home page — see §7.0 "New
content needed for Surface A" — which is a different page with a different job.)

Sections, in order:

1. **Hero.** Headline (3 candidates in §8) + subhead + stat strip + terminal.
   - Stat strip (the Fly.io device, mono, four numbers): `170.5 tok/s single-stream ·
     0.18 s TTFT cold / 3 ms warm · 464,870 req / 0 errors / 140 min soak · same tokens
     alone or in a full batch`. Each number is a link to its receipt. **Corrections vs the
     earlier draft:** 170.5 not 170.6 (N=5 median, not a single rep); the fourth cell must
     not read bare "byte-identical c=1 vs c=16" — it needs the object in plain English
     (§2a). Rig label ("RTX PRO 6000, rented pod") goes in the receipt link or a footnote,
     not omitted.
   - Terminal above the fold: a curl to `/v1/chat/completions` with the **current** model
     id — `qwen3.6-27b` today; `qwen3.8-27b` only once it has actually shipped and gated —
     and a visible `seed` param + a comment line `# same seed, same tokens — run it twice`.
     Copy button. One tab for Python (openai SDK, `base_url` swap — "change one line").
   - Dual CTA: **"Get an API key"** primary, **"Verify it yourself"** secondary (links to
     the verification recipes, then onward to the lab's receipts ledger — the
     differentiated path, and the inverted-positioning form of the original "Read the
     receipts"). Micro-copy: "OpenAI SDK compatible · free evaluation credits" (credits
     amount = owner call §12).
2. **The commitments block ("fig. 01 — what we commit to, and how you check").** This is
   the inverted-positioning centerpiece (§2): three commitments, each with **the
   customer's own verification recipe**, not our audit story.
   - *Same seed, same tokens — alone or inside a full batch.* Recipe: send the request
     twice with the same seed, diff the output; then send it during a load burst and diff
     again. (Support, one line: gated by replaying prompts at c=1 and c=16 against the
     same server and byte-comparing every stream — 16/16 on four models.)
   - *An honest meter.* Recipe: check `usage.prompt_tokens_details.cached_tokens` against
     your own prompt; abort a stream mid-flight and see it billed to the abort point.
   - *A pinnable build.* Recipe: pin `system_fingerprint` and detect the day the engine
     changes underneath you.
   Each commitment ends with a mono link to the actual gate log. One honest line under the
   block, unprompted: "Known bounded exception: near-tie first-token drift across prime
   configurations, ~7% of first tokens on a 144-prompt sweep (10/144, all at a top-2
   margin ≤ 0.70), documented here." **Do not** write a "spec-decode output is
   token-identical to plain decode" column in bare form — scope it or drop it (§2a).
3. **The lanes block ("fig. 02 — dark lanes").** The two-lane diagram: interactive lane
   holds p95 under a c=96 batch flood (real numbers: 7.15 s lane-blind → 3.69 s with lanes
   and a right-sized queue → 2.16 s at the tight SLO dial, where contended p95 becomes
   statistically equal to an uncontended box and batch pays −67%). **Attribution line is
   required on the page:** about half the 7.15 → 3.69 s move is the queue cap, not the
   lane gate. Harvest = **the discounted price tier on the same endpoint** (sheds with 429
   + Retry-After instead of queueing) — **not a second product** (§2 point 3). One
   sentence on the dark-pools analogy. CTA: "How lanes are priced →" (/pricing).
4. **Speed block.** **Self-competition framing only** (§2a) — llama benching stopped
   2026-08-03 and we currently lose cold TTFT, so a head-to-head here would be both
   off-posture and unflattering. Use the spec-decode ratio against our own plain decode:
   **2.17x on the SKU** (186.7 vs 86.2 tok/s, same run, bare CLI) and **2.61x on the
   official FP8 checkpoint** (128.06 vs 48.99, rented 2x5090). Keep it to 2-3 cells + a
   link to the full published board. Never a wall of comparison tables (repo posture:
   numbers are regression tracking, not a scoreboard). Aggregate throughput, if shown, is
   quoted at **c=8 = 420.6 tok/s** and labeled as the knee.
5. **Built in the open.** GitHub card (the **memra** repo — the engine and the lab's public
   record, stars, license), one line on the founder with link to avifenesh.ai, the three
   most recent receipt entries pulled from Surface A. On Surface B this is a short trust
   strip that links *up* to the lab site; on Surface A it is the whole site (§7.0).
6. **Bottom CTA repeat** (devs read whole pages — RESEARCH-DESIGN.md §1): API key +
   pricing links.

**What NOT to put on this page:** customer logos (there are none — never fake social
proof), model-menu grids, uptime SLA percentages not yet earned, roadmap promises,
newsletter modal, chat widget, cookie-consent theater beyond the legally minimal, **and the
operating-model narrative** ("serving pays for the hardware") — that belongs on Surface A's
About page, not in front of a prospective customer.

**Acceptance criteria:** headline + one measured number + runnable snippet + primary
CTA all visible at 1280×800 without scrolling; every number on the page is a working
link to a receipt; page weight < 1 MB, LCP < 1.5 s on fast 3G emulation, zero
third-party trackers; reads correctly with JS disabled (charts may degrade to static
SVG).

### 7.2 Models (+ live board)

**Message:** one model, and here is exactly what "supported" means.

- The SKU card: **Qwen3.6-27B today** — 3.8-27B only after it drops AND gates green
  (PRODUCT-TRUTH §4: the release is "week of 2026-08-10" per Alibaba's own post, its
  architecture/license/benchmarks are unpublished, and the *only* cleared phrasing is
  "a day-one bring-up runbook is written and ready ahead of the expected release" —
  never "day-one support guaranteed"). Card carries: precision arm served, context
  length, spec-decode on by default, tools/json_schema support flags, price summary, the
  OpenAI model id string.
  - **Precision wording is fixed copy, do not paraphrase** (PRODUCT-TRUTH §5): **Q8_0
    GGUF is the shipping 8-bit serving arm.** FP8-E4M3 safetensors checkpoints load, gate
    green, and serve — dequantized to the Q8_0 arm byte-identically (official artifact:
    prefill logits bit-identical 993280/993280 bytes), with GPU-side FP8 dequant cutting
    the 29 GB load wall **2.89x**, opt-in. A native per-block FP8 MMQ tile is implemented
    and bit-exact but does not clear the 1.1x bar, so it is off by default. A lossy
    per-tensor scale fold buys 18.4% prefill and is **not shipped**. Banned: "FP8 is our
    serving format", "FP8-native compute enabled", "FP8 is faster" (the e4m3-resident
    serve arm is *flat by construction* — the win is load time).
  - **Open owner conflict blocking this card:** whether 3.8 launches on a Q8_0 bridge or
    directly on FP8-ST (PRODUCT-TRUTH §4 conflict 2), and whether the 35B-A3B /
    Step-3.7-Flash line is retired (conflict 1). Do not name a day-one format until the
    owner calls it.
- **"What supported means here" panel** — the differentiated content: the gate battery
  a model passes before it's listed (kernel-check ALL GREEN, argmax MATCH, run-spec
  self-consistency, serve isolation gate), the ≥1.1x deployment bar, link to the
  day-one bring-up runbook for Qwen3.8. This turns the thin menu into a depth story.
  - **Wording fix:** write "**K=1..8 self-consistency is a standing gate, run on the
    MTP-capable artifact**" — not "K=1..8 on the production rig." The prod PRO 6000 board
    ran K=1..3 as the gate (plus K=4/5 as perf cells); the K=1..8 PASS battery lives on
    the community board's NVFP4-MTP artifact, and Q8_0 cannot run it at all (no MTP head).
    PRODUCT-TRUTH §2.2.
- **Live board:** rolling TTFT and tok/s percentiles (p50/p95/p99) from production
  probes, honest N and window, plus the current gate status (last battery run + hash).
  Publish tail percentiles prominently — competitors market p50; OpenRouter teaches
  buyers to shop p99 (RESEARCH-DESIGN.md §4). If live infra isn't ready at launch,
  ship the latest receipted battery as a dated static board — never a fake-live widget.
  - **Every cell needs its rig label and its N.** The prod numbers are `pro6000wk-runpod`
    (rented RTX PRO 6000 Blackwell 96 GB, 188 SM, 600 W, zero throttle), N=5 medians.
    Never mix them with the `pro6000wk-runpod-community` dev pod (5-11% slower) or the
    2x5090 rental. **There is no q27-at-8-bit row in the published board yet** — the Q8_0
    absolutes measured so far came off the community pod, and its own RESULTS.md says the
    absolutes get re-minted on prod-class silicon first. Publish **52.6** (prod q8 plain)
    and relative deltas; never the community absolutes 49.82/52.22. PRODUCT-TRUTH §7.6.
  - Cold and warm TTFT are **separate cells, always** (0.182 s vs 0.003 s — 61x). An
    unsalted repeat request hits the prefix cache, so a TTFT number without a fresh
    `cache_salt` is a warm number wearing a cold label.
- Deprecation/versioning promise: model ids are pinned, `system_fingerprint` carries
  the engine build, changes announced with dates.

**NOT here:** "coming soon" model lists (a single "request a model" mailto is enough),
speculative hardware announcements.

**Acceptance:** board data timestamped and sourced; every support flag on the card maps
to a gated feature in EVIDENCE-REPO.md §4; SKU card renders as a clean OG card when the
page is shared.

### 7.3 Pricing

**Message:** one product, one endpoint, priced by the latency guarantee you pick — and
the meter is honest.

**Framing correction (2026-08-05):** the earlier draft read as three *products*. It is
**one inference product with price tiers.** Harvest is a **tier on the same endpoint,
selected by a request header** (`x-lane: harvest`) — not a separate service with its own
signup, its own page, or its own brand. Copy says "tier" or "lane," never "our harvest
product." PRODUCT-TRUTH §1, §8.

Structure (shape decided here; **every NUMBER is owner sign-off, §12** — PRODUCT-TRUTH §8
marks all four UNDECIDED, with the market anchors below):

1. **Interactive tier** (default — naked traffic lands here) — per-token, premium.
   Admission-controlled p95, never preempted, spec-decode single-stream speed. This is
   the headline price. **UNDECIDED.**
2. **Harvest tier (the dark lane)** — per-token, discounted (anchor: 40–60% of
   interactive; Novita's batch tier = 50% off, Parasail publishes a parameter-band batch
   grid at $0.07/$0.22 for 21-41B). Honest semantics stated on the page: sheds with 429 +
   Retry-After under interactive pressure instead of silently queueing; designed for
   data-gen, evals, distillation. **UNDECIDED.**
3. **Dedicated** — per-hour/month reserved replica (a whole GPU's worth of the SKU), for
   tenants who want the box to themselves. Anchor to GPU-hour market rates plus margin.
   **UNDECIDED.**
4. **Cached input at 25% of input price** — the documented market convention, exactly 25%
   across every surveyed endpoint (EVIDENCE-REPO.md §5) AND a mechanism the service
   actually meters honestly (`cached_tokens` itemized in every response). Recommendation
   25%; **UNDECIDED.**

**Two posture rules that are NOT undecided and must survive into the built page**
(PRODUCT-TRUTH §8): (a) **do not price to the floor** — the floor fell ~35% in 90 days on
one measured family and a saturated replica at floor prices grosses $2-4/hr, so cost
coverage has to come from the guarantee tiers; the direct comp is **Groq pricing
Qwen3.6-27b at $0.60/$3.00**, an order above the DeepInfra floor, which is the proof the
class supports guarantee-priced tiers. (b) whatever ships **must match `/v1/models`
pricing metadata exactly** — OpenRouter and Hugging Face consume it programmatically.

Table styling: mono numerals, per-1M-token units, input/output/cached columns, no
"Contact us" cells for the core lanes. Below the table:

- **"How the meter works"** — the honesty block nobody else has: usage carries
  worker-truth token counts on every response including streams; disconnected requests
  billed to the abort point; cached tokens itemized. Link the serve-compat receipts.
- **Get-a-key flow is now real, not a mailto.** API keys and tenant auth landed
  2026-08-05 (`research/apikeys-20260805/RESULTS.md`, gate **18/18 PASS**): keyring with
  per-key `tenant` / `lane` / `enabled` / `rate_limit`, hot revoke ≤2 s, per-tenant prompt-
  cache namespace (same tenant shares cache hits, different tenant misses — verified by a
  two-tenant oracle), a batch-class key refused **403** on `x-lane: interactive`, and
  per-key rate limit = `min(override, global lane cap)`. The earlier spec predates this
  entirely and assumed manual provisioning — build the signup/key page as a real flow.
- **Free evaluation credits** (recommendation: small one-time credit, e.g. $5–10 — a
  perpetual free tier invites abuse a single-founder fleet can't police, while
  credit-on-signup is the category norm; one crypto-native provider's free tier was
  destroyed by >10k bot signups and killed. Verify amounts in RESEARCH-COMPETITORS.md;
  **owner sets the number**).
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

### 7.5 Receipts — **lives on SURFACE A (the lab site)**, per §7.0

**Placement changed 2026-08-05.** The receipts ledger is the *lab's* asset, not the
product's: it covers engine research that has nothing to do with serving (kernel work,
refuted mechanisms, the H100 lane). It lives on the lab site; the inference landing page
links to it and mirrors only the three or four service-relevant entries inline next to
the commitments they back.

**Message:** this is the page the whole brand hangs on. An indexed, dated ledger of
every public claim: gate batteries, endurance soaks, perf boards, incident notes.

- Format mirrors the avifenesh.ai evidence ledger: dated entries, each with claim →
  receipt link (raw JSONL/log in the memra repo) → **conditions (N, rig label, thermal
  regime, protocol caveat)**. The rig label is mandatory, not decorative — the same claim
  is wrong by 5-12% on a different board (PRODUCT-TRUTH §2.1).
- Wins AND bounded limitations both listed, and the limitations are **required entries**,
  not a courtesy: the ~7% first-token cross-config drift, the batched-plain near-tie
  class, the serve-vs-naked gap, cold-vs-warm TTFT, and the fixed sampler-bug class
  (PRODUCT-TRUTH §7).
- Seed entries at launch: the endurance soak, the nvcc miscompile catch, the router
  expert-selection defect, and the dogfood sampler bugs (BLOG-EVIDENCE.md A/B/C +
  EVIDENCE-REPO.md §1-3).
- Later this page absorbs the status history (post-incident notes live here).

**Acceptance:** every entry's receipt link resolves to raw data (not a summary blog);
every entry states N + rig + protocol; at least four honest-limit entries are present;
entries are immutable once posted (corrections append, never edit-in-place).

### 7.6 Blog

Standard chronological blog, RSS, no gating. Post inventory and launch pick: §10.
Styling: working-paper aesthetic (the avifenesh.ai note style), FIG-labeled charts from
real data, code blocks with copy buttons. Comments: none; discussion happens on HN/X —
each post footer links its HN thread once one exists.

**Acceptance:** RSS validates; posts render OG cards per §11; each post's numbers link
to receipts.

### 7.7 About — **lives on SURFACE A (the lab site)**, per §7.0

**Message:** who runs this lab and why you can trust one person with your inference.

- The founder, plainly: name, face, track record (AWS ElastiCache systems engineer,
  Valkey GLIDE maintainer, the public research record at avifenesh.ai), Tel Aviv.
- **The operating model, honestly — this is the lab's story and it belongs HERE, not on
  the product landing page:** serving revenue pays for the hardware; the research is the
  product; the engine is public because the receipts are the moat. A prospective customer
  on the landing page does not need to know their tokens fund GPU purchases
  (PRODUCT-TRUTH §1).
- The **product** name explained (lanes + the dark-pools analogy). Note this explains
  *darklanes the serving product* — the lab's own name is a separate owner decision (§5A)
  and this section must not imply the lab is called darklanes.
- **Hardware, stated exactly** (PRODUCT-TRUTH §3): every published number was measured on
  **rented** RTX PRO 6000 Blackwell pods, rented 8x H100, or rented 2x5090; the only owned
  GPU is an RTX 5090 **Laptop** development rig. The owned build-out targets **RTX PRO 6000
  class, homogeneous** — the 2x5090 purchase path was explicitly rejected on
  scaling-continuity grounds. Never "we run on PRO 6000" without "rented" or "pod";
  never a fleet photo or an implied datacenter.
- What the service is NOT yet: no SOC2, no multi-region, no 99.99 SLA, **no tensor
  parallelism** (one GPU per engine process; multi-GPU boxes are a replica fleet) — with
  the compensating controls stated (public receipts, status page, drain-not-drop deploys).
  This section converts the skeptics the rest of the site attracts.

**Acceptance:** zero stock imagery; no owned-datacenter implication; the "not yet" list is
present and current; the lab is never named "darklanes."

### 7.8 Status (status.<product-domain>)

Hosted on separate infrastructure/domain (RESEARCH-DESIGN.md §4). Uptime per lane,
current incident banner, history. Third-party hosted (e.g. a standard status provider)
is fine and MORE credible than self-hosted at this size. Link from footer + docs +
pricing FAQ.

### avifenesh.ai linkage (three brands now, still distinct)

Revised 2026-08-05 for the lab/product split. Three properties, one voice:

- **avifenesh.ai** = the personal research record (working papers, "the result can be
  yes, no, or not yet").
- **Surface A, the lab site** = the institution's record and the products shelf.
- **Surface B, the inference landing page** = the GTM asset for darklanes.

Link directions:

- Lab site → avifenesh.ai: About-page founder link; blog author bylines.
- avifenesh.ai → lab site: the portfolio's "experimental systems" section adds a card for
  the lab (with darklanes as the product inside it, not as a peer entry); research pages
  that used the engine footnote "the engine now serves production traffic" and link the
  product page.
- Landing page → lab site: the receipts ledger, the blog, and About all live on A; B links
  to them rather than duplicating them.
- **Do not make avifenesh.ai link "darklanes the lab"** — that framing is the thing the
  owner flagged. The personal site links the *lab* (name pending) and the lab shelves the
  product.

Shared DNA (evidence-ledger format, mono numerals, dark restraint) makes the set mutually
reinforcing without merging. The About page should feel like it was written by the person
whose site avifenesh.ai is — because it was.

## 8. Headline candidates — for SURFACE B, the inference landing page (3, with rationale)

Scope note (2026-08-05): these are the **product** landing-page headlines. Surface A (the
lab site) needs its own headline, which cannot be written until the lab is named (§5A); its
job is the institution and the research record, not conversion.

1. **"Inference you can verify yourself."**
   Subhead: "One open model, served fast and deterministic. Same seed, same tokens —
   alone or inside a full batch. Every number links to a raw log; every commitment ships
   with the recipe to check it."
   *Rationale:* the **inverted positioning** from §2 stated literally — the buyer's
   instrument, not our audit story. Keeps the receipts differentiator but frames it as
   something the visitor *does* rather than something we *claim*. Passes the specificity
   test via the subhead + stat strip. **Recommended.** (This replaces the earlier
   recommendation "Inference with receipts." — same asset, provider-centric voice; keep it
   as a fallback if the owner prefers the shorter line.)

2. **"170 tok/s, and the same tokens under load."**
   Subhead: "The Qwen 27B class on rented Blackwell silicon with bit-audited kernels —
   speculative decode included, receipts published, OpenAI-compatible."
   *Rationale:* the numbers-first pattern the conversion evidence favors most literally;
   unambiguous and falsifiable. **Two corrections vs the earlier draft:** the number is
   **170.5** (N=5 median; 170.6 was a single rep) so the headline should read "170" or
   "170.5", never "170.6"; and bare "byte-identical under load" is a **banned form** (§2a)
   — the object has to be there, hence "the same tokens under load." Risks: a single perf
   number invites Groq-style speed one-upmanship, which this product does not want to
   fight on — and it is *especially* wrong to invite right now, since llama.cpp currently
   wins cold TTFT (PRODUCT-TRUTH §7.2). Number needs updating whenever the board moves
   (same discipline as the repo's generated perf blocks).

3. **"One model, served obsessively."**
   Subhead: "Bit-audited kernels, hard interactive p95, and batch that rides the dark
   lanes — from the engine kernels up. Every claim receipted in the open, every claim
   reproducible by you."
   *Rationale:* leads with the single-SKU depth story and the founder-scale honesty;
   most memorable voice, weakest information scent (doesn't say "inference API" without
   the subhead). Best if the owner wants personality-forward.

Owner picks one (§12). All three keep the same subhead elements: model class, determinism
**with its object**, receipts + reproducibility, OpenAI-compat.

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
   endpoints) — being ready at the drop briefly makes us the price-setter, not taker.
   **But the pricing page cannot be built around day-one 3.8**: the release is expected
   "week of 2026-08-10" on Alibaba's own word, is not out as of 2026-08-05, and its
   architecture/license are unpublished. Price the 3.6-27B that exists; treat 3.8 as an
   upside, not a launch dependency (PRODUCT-TRUTH §4).
2. Do NOT price to the DeepInfra floor: the or-provider report measured the floor
   falling ~35% in 90 days on hy3, and floor-chasing grosses $2–4/hr per saturated
   replica — cost coverage comes from the premium tiers, not the race.
3. Suggested SHAPE (numbers are owner's): interactive tier at the upper-middle of the
   cluster justified by the p95 guarantee + determinism (e.g. mid $0.20s out), harvest
   at ~50% of interactive, dedicated priced from GPU-hour market rates + margin,
   cached input at 25%. Free credits small and one-time (Chutes' bot-killed free tier
   is the cautionary tale; Novita's $0.50 voucher and Hyperbolic's deposit-gated
   rewards are the survivors' pattern).
4. Whatever ships must match the `/v1/models` pricing metadata exactly — OpenRouter
   and HF both consume it programmatically.
5. **Do not justify the premium with a latency comparison.** The guarantee (admission-
   controlled p95, no preemption, an honest meter, reproducible output) is the
   justification. A head-to-head latency argument is currently false on the cold-TTFT
   path (PRODUCT-TRUTH §7.2) and off-posture besides (llama benching stopped 2026-08-03).

## 10. Content strategy — first posts, launch pick, distribution

**Placement (2026-08-05): the blog lives on SURFACE A, the lab site.** These are the
lab's research posts; the product landing page links the two or three that back its
commitments and hosts none of them. Bylines are the founder's; the engine is credited as
**memra**, never as "darklanes" (the product does not do research).

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
   13.0.88) while being the whole sales pitch in disguise.
   *Honesty note the post must carry:* the original failing logs were lost to
   `rsync --delete`. The committed repro survives (8/8 fail pre-fix, 0/8 post-fix) and is
   what the post should demonstrate from; do not present reconstructed logs as the
   originals.

2. **"My own agent found three bugs in my server (dogfood week)."** Abstract: the
   founder's daily agent locked into identical tool-call loops. Cause #1: omitted
   temperature deserialized to 0.0, and `is_greedy()` is `temperature <= 0.0` — so every
   client that omits temperature got greedy argmax instead of OpenAI's documented 1.0.
   Cause #2, the one that survived the first fix: omitted seed deserialized to 0 — a valid
   *fixed* seed. **Cause #3, found 2026-08-05: the server was injecting `!` into output.**
   `top_p`/`min_p` truncation in the sampled-spec full-accept path read its filter stats
   from `col_stats.last()` — a *neighbouring* column — so a foreign `row_max` mis-scaled
   the exponent, every id failed the threshold, the row masked to -3.4e38, and argmax fell
   through to the smallest-index tie-break: token id 0, `!`. Symptoms in the wild were
   `!bash`, `grep -!q`, `/!etc/hosts`; `min_p 0.05` hit a **100% id-0 rate**; plain
   `top_k 40` was clean; **the engine's own default sets no truncation filter and is
   structurally immune**, which is exactly why the owner's daily never showed it — it only
   bit clients configured like *other* engines.
   All three are one bug class: **a meaningful zero is not "unset."** Plus the
   uncomfortable meta-lesson: every golden in the repo ran temp-0, which routes around the
   sampler chain entirely, so a broken sampled path was invisible to the entire argmax
   battery — closed now by a distribution-level composition gate (composed accept-walk
   output vs target p, 20k draws, L-inf 0.012 / TV 0.05), whose negative controls fail as
   designed: "forgot the residual" trips TV 0.0881 **with acceptance unchanged**, i.e. a
   bug every isolation arm would have passed.
   *State it fixed, and state that it shipped for a window.* Fix `d1dc79b8`, merged
   `44c4c6a4`; the differential serve-smoke matrix (`9bbd3cca`) is **proven in both
   directions** — 3 failures on the pre-fix binary, 0 after. Receipts:
   `research/sampledspec-20260804/`, `research/sampfix-20260805/`.
   *Slot:* week 1–2 after launch. Relatable, self-deprecating, credibility-compounding.

3. **"Serving is a determinism problem: the batch that changed the model's mind."**
   Abstract: under batched prefill, an m-dependent router GEMM changed which EXPERTS a
   MoE request activated depending on who arrived with it — **121/760 = 16% of
   (layer,token) pairs differed in expert *set*** (plus 217 more differing in order only),
   on Ornith-1.0-35B Q4_K_M at total_m=75, **before the fix**. The isolation gate
   (byte-compare c=1 vs c=16 against the same server) that found it, the m-invariant router
   twins that fixed it, and the resulting contract: your output does not depend on your
   neighbours. Mechanism was *isolated*, not guessed — the cuBLASLt router GEMM is
   m-dependent at maxdiff 0.0039, the in-house `router_gemv` is m-invariant at 0.0, and 36
   trunk weights are m-invariant, which pins it to two router GEMMs.
   *Two required precision notes:* label the 16% **pre-fix** every time (post-fix gate is
   16/16 on four models), and do **not** reuse the "post-trains have tighter margins"
   explanation — it was tested and **REFUTED** in the same receipt. Include the honestly
   documented bounded exception (near-tie first-token drift, 10/144, every flip at a
   top-2 margin ≤ 0.70) and keep it distinct from the isolation contract — they are
   different classes and must not be conflated.
   *Slot:* week 3–4; the technical centerpiece for eval/agent audiences.

4. **"Dark lanes: hard p95 for interactive while batch eats the leftovers."** Abstract:
   the noisy-neighbour problem measured (a c=96 bulk tenant inflates interactive p95 4.2x,
   1.74 → 7.33 s), why shedding at admission beats queueing in the engine ("the engine's
   own queue is where the tail dies"), per-lane prefill budgets vs the global
   chunked-prefill knob, and the 140-minute soak. Ends with the pricing consequence: batch
   is cheaper because it yields.
   *The attribution the post MUST make, and which makes it a better post:* raising the
   proxy cap from 8 to 16 *by itself* moves p95 7.15 → 4.34 s; the lane gate moves
   4.34 → 3.69 s. **Roughly half the win is the queue, not the gate** — RESULTS.md's own
   line is "the engine gate cannot fix a queue it never sees," which is a sharper thesis
   than a single 2x claim. The tight dial (25 ms) is where the real result lands:
   contended p95 2.158 s, statistically equal to an uncontended box (2.065 s), bulk paying
   −67%.
   *Soak wording:* 140 min / 464,870 requests / 0 errors / 0 sheds / +0.045% drift is
   solid, but it ran at **temperature 0.7, seeded, on prefix-cache hits, on a 9B-class
   model across 8 rented H100s** — the identical determinism hash is a **separate greedy
   probe** before and after. Write "a greedy probe hashed identical on all 8 replicas
   before and after the soak," never "464,870 deterministic requests."
   *Slot:* aligned with any pricing/HF-listing push; this is the product-namesake post.

5. **"Spec-decode economics on consumer-priced silicon."** Abstract: MTP speculative
   decoding turns a 27B into a **186.7 tok/s bare / 170.5 tok/s served** single-stream
   model on a rented workstation card — **2.17x the same-run plain decode (86.2)** — and
   **2.61x on the official FP8 checkpoint out of the box** (128.06 vs 48.99) using the
   checkpoint's own MTP head. Where the wins come from (acceptance ladders, draft masking
   under json_schema), where they stop (aggregate saturates at c=8 = 420.6 tok/s while p50
   doubles at every step beyond it — "queueing, not throughput"), and what that does to
   $/token.
   *Required labels:* 186.7/170.5/420.6 are `pro6000wk-runpod` (rented RTX PRO 6000
   Blackwell 96 GB), N=5/N=3 medians; the **2.61x is on rented 2x5090** and there is **no
   official-FP8 cell on any PRO 6000** — do not merge the two boards. Also state that
   spec-decode **triples TTFT** on that arm (0.170 → 0.466 s), which is the honest
   counterweight and the reason cold TTFT is a separate cell everywhere.
   *Do not frame it against llama.cpp* — self-competition only (§2a); the ratio against our
   own plain decode is the number, and it is a bigger one anyway.
   *Slot:* week 4–6; the economics audience (and the HN "consumer GPU" perennial).

Plus the news-peg announcement (not a numbered post): **"Qwen3.8-27B, day one"** —
published the day the gate battery goes green per the bring-up runbook, linking the
receipts. Short, factual, no launch theater. It cannot be drafted with numbers in advance
(the model does not exist yet) and the word "supported" is earned only after the ≥1.1x
deployment bar clears.

**Distribution:**

- **HN:** launch post submitted as a plain title (no "Show HN" for the compiler story;
  "Show HN: darklanes — inference you can verify yourself" is the separate *product*
  submission, best timed with the Qwen3.8 day-one announcement). Founder answers questions
  in-thread; receipts links do the arguing. (Reddit is skipped — the owner's account is
  spam-filtered sitewide; HN/X/Discord instead.)
- **X/dev-twitter:** each post gets a thread whose first tweet is the single most
  concrete artifact (the miscompile repro diff, the p95 chart), not the link. Charts
  as dark-card images (§11).
- **Hugging Face:** the distribution/discovery channel — an org page (**named for the lab
  once the lab is named**, not for the product), the provider-registration path
  (OpenAI-compat LLM APIs skip most schema work; huggingface.js PR; Model Mapping API,
  which needs a Team/Enterprise Hub plan; per-request nano-USD billing endpoint with
  `Inference-Id` headers; automated 6-hourly validation incl. TTFT < 5 s streaming,
  tool-calling and structured-output tests — all behaviours memra already gates;
  EVIDENCE-REPO.md §6). A separate lane owns the mechanics; the Models page and docs carry
  the `HF Inference Providers` badge/link once live. HF's default provider ordering =
  7-day routed volume, so the listing compounds with actual traffic, not marketing.
- **OpenRouter:** apply (expect the backlog; open-weight models are non-prioritized);
  the realistic near-term value is public perf/uptime charts on a neutral surface,
  i.e. more receipts — treat as distribution, not revenue (EVIDENCE-REPO.md §5).
  Application material already exists: `research/or-application-20260805/APPLICATION.md`.
- **GitHub / crates.io:** the memra repo is the **lab's** public record — the README links
  the product page once live ("memra in production"); release notes cross-post
  board-moving perf changes to the receipts ledger on Surface A. Crates caveat for any
  "shipped" copy: 9 publishable crates at 0.69.0, but the first tagged publish landed
  **5 of 9** before crates.io's new-crate burst limit returned 429 (a resumable per-crate
  workflow then landed). **Verify the registry before writing "all nine crates are live"**
  — the only 9/9 evidence in-repo is a dry run.

## 11. OG image concept

One template, generated per page (Vercel OG / `ImageResponse` pattern —
RESEARCH-DESIGN.md §5):

- **Two wordmark variants** now (§7.0): the lab mark for Surface A pages, the darklanes
  mark for Surface B. Same template, same grid.
- 1200×630, near-black `#0A0B0E`, three thin horizontal lane lines across the lower
  third with the middle lane in the accent amber; wordmark bottom-left, page title
  ≥60 px bold sans center-left (~40 chars max), and ONE mono-set metric top-right
  (per-page: product home = `170.5 tok/s · same tokens under load` — **not** "170.6", and
  **not** bare "byte-identical" per §2a; blog posts = the post's headline number; models =
  TTFT/percentiles with cold and warm distinguished).
- Contrast ≥4.5:1; verify legibility at 200×105 thumbnail; test on light AND dark feed
  backgrounds (mid-gray dies in dark mode — cited guidance).
- The GitHub repo social-preview (manual upload, Settings → Social preview — no API)
  gets the home variant.

## 12. Owner sign-off required (nothing below ships without it)

**0. THE LAB NAME — blocks everything else on this list.** §5A carries the decision brief
(four candidates, RDAP availability, how each pairs with darklanes underneath). Until this
is called, Surface A cannot be built at all, no wordmark can be drawn, no HF org can be
created, and no domain should be bought — the lab domain is the primary purchase and the
product domain is secondary to it. Spec recommendation: **Option 1 (memra as the lab
identity)**, falling back to Option 4; the one real cost of Option 1 is that `memra.ai`,
`memra.dev`, and `memra.com` are all already registered (`memra.io`, `memralab.ai`,
`memralabs.ai`, `memralabs.com` are available).

1. **Domain purchases** — the lab domain (blocked on item 0) plus darklanes.ai
   (+ .com/.dev defensively; darklane.ai optional). All three darklanes TLDs verified
   **available** by RDAP 2026-08-05. Receipts in DOMAINS.md; ~$100–150/yr for the trio.
   **Nothing purchased.**
2. **Headline pick** — §8, for Surface B (spec recommends #1, "Inference you can verify
   yourself."). Surface A's headline is blocked on item 0.
3. **Pricing numbers** — the three tier prices, the dedicated rate, the cached-input
   fraction, and the free-credit amount (§9 anchors; **structure is decided, every number
   is UNDECIDED**). Also: the public data-retention policy text.
4. **Logo/wordmark** — direction in §6; two marks needed (lab + product); final call is
   the owner's.
5. **Accent color** — spec recommends lane-marker amber; owner may prefer another hue
   (keep it single-accent regardless).
6. **Launch timing** — coupling the launch to the Qwen3.8 day-one announcement
   (recommended: both surfaces go live quietly first, HN product submission rides the
   day-one news peg). Note the drop is expected "week of 2026-08-10" and is **not** in our
   control; nothing on either surface may depend on it.
7. **Free-tier/credits policy** and abuse posture.
8. **Company/legal identity** shown in the footer (invoicing entity, ToS/privacy
   pages — also prerequisites for OpenRouter/HF listings).
9. **The two SKU conflicts** (PRODUCT-TRUTH §4): whether the 35B-A3B / Step-3.7-Flash line
   is retired in favour of the q27 story, and whether 3.8 day one runs a Q8_0 bridge or
   goes directly to FP8-ST. The Models and Pricing pages cannot name a format until these
   are called.

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

Every criterion below applies to **both surfaces** unless it names one.

1. Every number on every page resolves to a raw receipt (repo file or dated
   third-party source). Automated link check in CI.
2. **Every published number carries its rig label, its N, and its cold/warm or
   bare/served distinction.** A number without those is a build failure, not a polish
   item — this is the single rule whose absence produced the wrong product the first time.
3. **Zero claims contradict `docs/PRODUCT-TRUTH.md`.** Mechanical check: no page contains
   `170.6`, `421 tok/s`, bare `byte-identical`, "2x5090 trajectory", "we run on PRO 6000"
   without "rented", "K=1..8" attributed to the prod rig, "FP8 is our serving format", or
   "all nine crates live".
4. **No page names the lab "darklanes."** darklanes appears only as the product.
5. The five §10 posts exist in draft on Surface A; the launch post is publishable day one.
6. Quickstart executes verbatim against production; time-to-first-token for a new
   visitor following it < 60 s of active work.
7. Surface B home passes: headline + number + snippet + CTA above the fold at 1280×800;
   LCP < 1.5 s; works without JS.
8. Pricing page lets a visitor compute a concrete workload's cost unaided; numbers
   match `/v1/models` metadata; harvest reads as a **tier**, not a second product.
9. **Every commitment on Surface B ships with its verification recipe** (§2 inversion) —
   a commitment the visitor cannot check themselves does not belong on the page.
10. The honest-gaps content is present, not buried: the ~7% drift note under the
    commitments block, the "not yet" list on About, cold-vs-warm TTFT split everywhere.
11. OG cards render legibly at 200×105 on light and dark feeds, per-surface wordmark.
12. No banned vocabulary (§6 voice) and no banned claim forms (§2a) anywhere, including
    alt text and meta descriptions.
13. Status page lives on separate infrastructure and is linked site-wide.
14. avifenesh.ai linkage implemented per the revised three-brand map in §7.

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
- **Reconciled product claims: `docs/PRODUCT-TRUTH.md` (memra repo).** Supersedes any
  number in this file.

## 16. CHANGELOG — 2026-08-05 correction pass (hand THIS to the build-agent)

An earlier revision of this spec was handed to a build-agent and produced the wrong
product. The cause was **staleness, not overclaim**: the spec was written 2026-08-04 from
append-only research directories that were each correct on their own date, and seven things
landed underneath it. This section is the diff, so the owner can re-hand ~4 pages instead of
40.

**The structural fix, so this cannot recur:** `docs/PRODUCT-TRUTH.md` is now the single
reconciled source for every product-facing claim, and CLAUDE.md carries a rule requiring
any lane that moves a product number to update it in the same commit — the same discipline
that already governs the generated perf boards. **Read PRODUCT-TRUTH first; if it and this
file disagree, this file is stale again.**

### 16.1 What changed, by category

**A. Three owner decisions folded in (these are new direction, not corrections).**

| # | Decision | Where it landed |
|---|---|---|
| A1 | **Positioning inverted** — from "the provider that shows its work" to "commitments you can verify yourself." Same evidence; the buyer's instrument instead of our diligence. | §2 rewritten; §7.1 point 2 became a commitments-plus-recipes block; §8 headline #1 replaced; §14.9 made it an acceptance criterion |
| A2 | **The lab is the parent entity, and it is NOT called darklanes.** Owner: "calling a lab darklanes might be problematic." darklanes = the serving product only. The lab name is **owner-pending and blocks everything.** | §5 split into §5A (lab decision brief) / §5B (product verdict); §12 item **0**; PRODUCT-TRUTH §1 |
| A3 | **Two surfaces, not one site** — a lab site with the product embedded, plus a separate inference landing page for GTM. | §7.0 (new) with the content mapping table; §7.1/7.5/7.7 re-homed; §7 avifenesh linkage revised to three brands |

**B. Numbers corrected.** Each was a real receipt read at the wrong precision or off the
wrong rig.

| Old | New | Why |
|---|---|---|
| `170.6 tok/s` serve | **170.5** | 170.62 is the r4 single rep; the N=5 median is 170.55 |
| `421 tok/s` aggregate | **420.6 at c=8** | N=3 median is 420.57; 421.9 is c=16, and c=16/32 are *queueing, not throughput* (p50 doubles per step) |
| "serve tax −12.6%" as a stat | **not a marketing number at all** | It is a gap being closed: −11.74% (Q8_0) / −8.66% (NVFP4 spec), root cause known, filed as task #70 |
| "2.6x on the PRO 6000" | **2.61x on a rented 2x5090** | There is no official-FP8 cell on any PRO 6000. The 2.6x is solid — it just belongs to a different box |
| "run-spec K=1..8" on the prod rig | **"K=1..8 is a standing gate on the MTP-capable artifact"** | Prod ran K=1..3 as the gate; the K=1..8 PASS is on the community board; Q8_0 cannot run it (no MTP head) |
| "464,870 deterministic requests" | **"a greedy probe hashed identical on all 8 replicas before and after the soak"** | The 140-min load ran temp 0.7 seeded on prefix-cache hits; the determinism hash is a separate pre/post probe. Also: 9B-class model, rented H100s — not the 27B SKU |
| "lanes take p95 7.15 → 3.69 s" | **half of that is the proxy cap** (7.15 → 4.34 s by itself); the lane gate does 4.34 → 3.69 s | RESULTS.md: "the engine gate cannot fix a queue it never sees." The tight-dial result (2.16 s ≈ uncontended, bulk −67%) is the real headline |
| "constrained decoding at 99.4%" | **plain lane only; the spec lane is 79%** | Different lane, different cost |
| "16% of router pairs" | **121/760 = 16%, Ornith-35B Q4_K_M, PRE-FIX** | Post-fix gate is 16/16 on four models. Also: the "post-trains have tighter margins" explanation was **refuted** — do not reuse it |
| "all nine crates live at 0.69.0" | **first publish landed 5/9** before a crates.io 429; resumable workflow landed | The only 9/9 evidence in-repo is a dry run — verify the registry |

**C. Claim wording tightened (new §2a, STRICT).** These are falsifiable-in-minutes forms
on a verify-it-yourself brand, which makes them worse than saying nothing.

- Bare **"byte-identical"** / "token-identical to the CLI oracle" / "one canonical output"
  → scope it: *same seed, same tokens, whether your request runs alone or inside a full
  batch*. The gate is serve-vs-serve, c=1 vs c=16, same server, at defaults. It does **not**
  establish identity against a tokenwise oracle — that path has a documented near-tie flip
  class a customer could demonstrate.
- **No interactive-latency claim against llama.cpp.** A same-artifact, N=5 interleaved
  head-to-head on 2026-08-05 has llama ahead on cold TTFT (0.19 s vs 0.53 s), short agentic
  turns, and raw prefill. Speed copy uses self-competition ratios instead — which is also
  the standing repo posture since llama benching stopped 2026-08-03.
- **Every number carries its rig label and N.** The headline board is a *rented* RTX PRO
  6000 Blackwell pod (188 SM), never "we run on PRO 6000" unqualified.
- **Cold and warm TTFT are always separate cells** (0.182 s vs 0.003 s). An unsalted repeat
  request hits the prefix cache; a TTFT without a fresh `cache_salt` is a warm number
  wearing a cold label.
- **FP8 is not the serving format.** Q8_0 GGUF ships; both exact FP8 compute arms are
  default off; the lossy fold is non-shippable; the FP8 win is *load time*, not tok/s.

**D. Target platform corrected.** The owned trajectory is **RTX PRO 6000 Blackwell class,
homogeneous** (owner override 2026-08-03) — not 2x RTX 5090, which was explicitly rejected
on scaling-continuity grounds. Nothing in that class is owned; the only owned GPU is an RTX
5090 **Laptop**. 2x5090 remains alive as the *rental measurement platform*, so do not write
"the 5090 is dead" either.

**E. Scope corrected: harvest is a PRICE TIER, not a product.** One product (inference on
one model class) sold with a latency-guarantee dial, selected by an `x-lane` header. The
earlier draft's three-products lineup implies a second pipeline that does not exist.

**F. Features the spec predated (now assume they exist).**

- **API keys and tenant auth** (2026-08-05, gate 18/18 PASS): keyring with per-key tenant /
  lane / enabled / rate_limit, hot revoke ≤2 s, per-tenant prompt-cache namespace, 403 when
  a batch-class key asks for the interactive lane. The get-a-key flow is a **real flow**
  now, not a mailto.
- **The `!`-injection sampler bug found AND fixed** (2026-08-05) — a third bug for blog
  post B, and a credibility asset: it only bit clients that set `top_p`/`min_p` (i.e.
  configured like other engines), the engine's own default is structurally immune, and the
  gate that now catches it was proven in both directions against the pre-fix binary.
- **Safetensors FP8 dir checkpoints** load bit-exact, 2.89x faster, and are spec-eligible by
  default (the old quarantine is lifted).

**G. Honest gaps that are now REQUIRED content, not optional garnish.** PRODUCT-TRUTH §7 is
the list: the serve-vs-naked gap, the llama cold-TTFT loss, the ~7% first-token
cross-config drift (10/144, every flip at a top-2 margin ≤ 0.70), the batched-plain
near-tie class, the fixed sampler-bug class, and the does-not-exist-yet list (no SOC2, no
multi-region, no tensor parallelism, no power-curve data, no q27-at-8-bit board row).

### 16.2 Content-to-surface mapping (the table the build-agent needs)

Surface **A** = the lab site (name pending, §5A). Surface **B** = the inference landing
page (darklanes). The existing draft site's content is mostly B; About and Blog move to A.

| Content | Surface | Notes |
|---|---|---|
| Hero, stat strip, terminal, dual CTA | **B** | Numbers per §16.1B; CTA secondary becomes "Verify it yourself" |
| Commitments + verification recipes ("fig. 01") | **B** | Was the determinism block; reframed per A1 |
| Lanes block ("fig. 02") | **B** | Proxy-cap attribution line required; harvest = tier |
| Speed block | **B** | Self-competition ratios only |
| Models + live board | **B** | Rig label + N on every cell; cold/warm split |
| Pricing | **B** | Three tiers of one product; every number owner-undecided |
| Docs (5 pages + quickstart) | **B** | Unchanged |
| Status | **B**, linked from both | Separate infra either way |
| **Receipts ledger** | **A** | The lab's core artifact; B links to it and mirrors 3-4 entries inline |
| **Blog (posts A-E)** | **A** | Lab output; engine credited as *memra*, never as darklanes |
| **About + operating model + "not yet" list** | **A** | "Serving pays for the hardware" belongs here, NOT in front of a customer |
| **Lab home + Products section** | **A** | NEW writing — the only genuinely new content this restructure creates |
| Founder / avifenesh.ai linkage | **A** | Three brands now: personal site → lab → product |

Chrome: Surface B header = wordmark, Models, Pricing, Docs, Receipts (→A), Blog (→A),
GitHub, one CTA "Get an API key". Surface A header = lab wordmark, Research, Blog, Products,
About, GitHub — **no API-key CTA**; the Products section carries the single link.
Launch total: **A 5 pages + B 4 pages + docs + status.**

### 16.3 Blocked on the owner (do not guess)

1. **The lab name** — §5A: four options, RDAP availability, spec recommends Option 1
   (*memra* as the lab identity, `darklanes-by-memra` as the product) falling back to
   Option 4. Its one real cost: `memra.ai`/`.dev`/`.com` are all already registered;
   `memra.io`, `memralab.ai`, `memralabs.ai/.com` are available. **This blocks Surface A
   entirely, both wordmarks, the HF org name, and the primary domain purchase.**
2. **Every price** — §9 anchors are researched; all four numbers plus the free-credit amount
   are undecided.
3. **The two SKU conflicts** — retire or rescope the 35B-A3B / Step-3.7-Flash line, and pick
   Q8_0-bridge vs FP8-ST-direct for 3.8 day one. The Models and Pricing pages cannot name a
   format until then.
4. Headline pick, accent color, wordmarks, launch timing, data-retention text, legal entity.
