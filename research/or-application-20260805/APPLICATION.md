# OpenRouter provider application — paste-ready package (2026-08-05)

Prepared for the owner to submit in one sitting. The submission itself is owner-only
(account, form click, legal identity). Everything below is drafted from repo truth;
items marked **OWNER** need the owner's decision or credentials before pasting.

Companion research: `research/or-provider-20260802/REPORT.md` (the listing study),
`research/hf-inference-20260804/ASSESSMENT.md` (the OR-first sequencing verdict),
`research/gap-scan-20260802/REPORT.md` (serve-surface grades),
`research/serve-tail-20260804/RESULTS.md` (/v1/models, rate-limit headers, drain),
`research/darklanes-website-spec-20260804/SPEC.md` (positioning + pricing anchors).

---

## 0. The actual application — what OR asks

Entry: `openrouter.ai/providers/apply` → the form at `/providers/apply/form`.
The form is an embedded HubSpot form (portal 244648951, form
e2fa81df-6179-4e3c-aefc-e9b8c5f9f20c); the FULL field definition was fetched live
2026-08-05 and committed as the raw receipt:
`or-apply-form-def-20260805.json` (this directory). Page banner (fetched live,
rendered page): "Complete the form below. We review applications on a rolling basis
and will follow up via email. Due to high demand, not all applications are accepted."

**All 15 fields, all required.** Submit button: "Submit Application". No file
uploads, no captcha field in the form definition (hCaptcha rides the page).
Contact email gets invited to a **Slack Connect channel** — that email is the
ongoing operational contact, choose it accordingly.

Review stages (apply page): submit → technical review (API compat, reliability,
pricing, performance) → integration with test traffic → go live. Backlog note
(apply page): "We currently have a large backlog of provider applications and are
prioritizing providers with proprietary models." Applications failing the technical
criteria are not reviewed. Being in the queue early is the point of submitting now.

Pre-submission technical bar (apply page, all four already met — receipts in §2):
1. OpenAI-compatible `/chat/completions`, streaming, usage tokens in stream + non-stream.
2. `/models` endpoint with pricing (USD strings), context length, max output tokens,
   features, datacenter locations.
3. Automated payment (monthly invoicing — OR pays us).
4. Published privacy policy + data-retention terms (**the one open gap — §4**).

---

## 1. Field-by-field draft answers

| # | Field (exact label) | Answer | Status |
|---|---|---|---|
| 1 | Company Name | darklanes | **OWNER**: confirm legal vs trade name — if the legal entity differs (sole proprietor / Ltd. in formation), decide what goes here |
| 2 | Website | https://darklanes.ai | **OWNER**: domain is AVAILABLE but UNREGISTERED (DOMAINS.md receipt, 2026-08-05). Must be purchased AND serving at least a landing + privacy/ToS pages before submission. Fallback until then: https://github.com/avifenesh/memra |
| 3 | Your Email | — | **OWNER**: company email (they invite it to Slack Connect). An @darklanes.ai address once the domain exists; avoid personal gmail |
| 4 | Display Name | darklanes | ready |
| 5 | Desired Slug | `darklanes` | ready (lowercase, no collision found — no existing provider or brand of that name, DOMAINS.md web-check) |
| 6 | Distinguishing Features (checkboxes) | **Low Latency** + **High Throughput** + **Unique Infrastructure** | see §1a below for the honest case per box; do NOT check Low Pricing (pricing posture is deliberately not floor-chasing, SPEC §9), nor Unique Models (we serve open weights), nor Decentralized / Strategic Partnership |
| 7 | Extra Details (textarea) | §1b below, paste-ready | ready |
| 8 | URL to /models API | `https://api.darklanes.ai/v1/models` (or chosen host) | **OWNER**: needs the public deployment live; the endpoint itself is implemented + battery-gated (serve-tail-20260804, PASS) |
| 9 | API Base URL | `https://api.darklanes.ai/v1` | **OWNER**: same dependency |
| 10 | URL to Privacy Policy | `https://darklanes.ai/privacy` | **OWNER**: page does not exist yet — §4 |
| 11 | URL to Terms of Service | `https://darklanes.ai/terms` | **OWNER**: page does not exist yet — §4 |
| 12 | Data Policy (textarea) | §1c below, paste-ready once owner confirms retention window | draft |
| 13 | Supported Output Modalities | **Text** only | ready (honest: text->text is the only modality) |
| 14 | Inference Location | US | **OWNER**: confirm — current fleet receipts are AWS us-east (H100 p5 box, G7e us-east-1/us-east-2); if launch fleet lands elsewhere, update. Format: country codes |
| 15 | HQ Location | Tel Aviv, Israel | **OWNER**: confirm (SPEC.md §1 states Tel Aviv) |

### 1a. Distinguishing Features — the honest case per checkbox

- **Low Latency**: TTFT 0.182 s cold / 3 ms warm (prefix-cache hit) on the 27B-class
  SKU, RTX PRO 6000, 2026-08-04 (`research/pro6000-prod-20260804/`). Soak p95 1.76 s
  end-to-end at c=96 fleet load, worst single request 1.861 s over 140 min
  (`research/fleet-endurance-20260803/SUMMARY.txt`).
- **High Throughput**: 170.6 tok/s single-stream through the serve surface (MTP
  speculative decode), 421 tok/s aggregate at c=8 on one GPU (same receipts);
  7,003 tok/s aggregate sustained on an 8-GPU fleet for 140 min, drift +0.045%.
- **Unique Infrastructure**: memra — a from-scratch Rust+CUDA engine, public on
  GitHub, every serving mode gated token-identical to plain decode, every published
  number's raw logs committed in-repo. QoS lanes (interactive/judge/harvest) let
  batch and interactive traffic share GPUs without tail-latency destruction.
  Deterministic serving: greedy output hash identical on all 8 replicas before and
  after a 464,870-request soak.

### 1b. Extra Details — paste-ready textarea

> darklanes serves open-weight models on NVIDIA Blackwell hardware (RTX PRO 6000
> 96GB class), powered by memra — a from-scratch Rust+CUDA inference engine that is
> public on GitHub (github.com/avifenesh/memra). The differentiator is receipts:
> every kernel is bit-audited against a CPU reference, speculative/graph/batched
> serving is gated token-identical to plain decode, and every performance number we
> publish has its raw run logs committed in the public repo. Determinism is a
> product feature: a seeded request replays byte-identical, alone or under load —
> built for eval harnesses, agent pipelines, and anyone whose workload breaks on
> nondeterminism.
>
> Serve surface: OpenAI-compatible (validated against the official openai SDK),
> streaming with usage on stream and non-stream, tools/function calling with
> streamed tool_call deltas, reasoning output separated into the OpenRouter
> reasoning/reasoning_details fields (include_reasoning honored), real constrained
> decoding (json_schema via llguidance at 99.4% of unconstrained speed),
> cross-request prefix caching with per-tenant cache_salt isolation, honest usage
> accounting (cached tokens itemized at the 25% convention, aborted requests billed
> to the abort point), X-RateLimit-* headers, graceful drain on deploys (in-flight
> streams finish; zero mid-stream errors on planned restarts), and early-429
> admission control — we shed at admission with Retry-After, never queue-and-degrade.
>
> Measured numbers (RTX PRO 6000, 2026-08-04, raw logs in-repo): 27B-class SKU at
> 170.6 tok/s single-stream via MTP speculative decode, 421 tok/s aggregate at c=8,
> TTFT 0.182 s cold / 3 ms warm. Fleet endurance: 8 GPUs, 140 minutes at c=96 —
> 464,870 requests, 0 errors, 0 sheds, throughput drift +0.045%, deterministic
> output hash identical on all replicas pre/post soak.
>
> Catalog: one model class served obsessively rather than a large menu — the Qwen
> 27B class at 8-bit (Qwen3.6-27B today; day-one bring-up runbook for Qwen3.8-27B
> ready ahead of its expected release). Capacity is honest and small at launch
> (capacity_tpm declared low, early 429s over queueing); replicas scale before
> pricing deepens. Quantization is declared exactly in /v1/models. Deployment
> trajectory: RTX PRO 6000 Blackwell fleet.

(Every claim above traces to `research/darklanes-website-spec-20260804/EVIDENCE-REPO.md`;
numbers: `research/pro6000-prod-20260804/`, `research/fleet-endurance-20260803/`.)

### 1c. Data Policy — draft textarea (owner confirms retention window)

The form asks exactly: "Do you have a training policy for prompts and completions?
If you're logging prompts, how long do you store them?"

> Prompts and completions are never used for training — no training pipeline exists
> and none is planned. Prompt and completion TEXT is not persisted: per-request
> records retain metadata only (request id, tenant/lane, token counts including
> cached-token split, timing, finish reason) for billing and reliability
> engineering, retained for [OWNER: pick — the or-provider report suggests
> stating a concrete window, e.g. 30 days] and never shared. Aborted requests are
> billed to the abort point from the same metadata. We support OpenRouter's
> data-collection disclosure as "deny" (no prompt logging).

**Truth check before pasting**: metadata-only logging matches the current serve
surface (usage/abort log lines carry token counts, not text — `[abort]` records
prompt/cached/generated counts, docs/SERVING.md). The per-request JSONL metering
lane (`/usage` endpoint, reconciliation EXACT 13/13) lives on the darklane fleet
branch (`darklane/main` commit 73d3adf4), not on `restructure/public-split` — the
production deployment must run a build with that metering surface, and whatever
retention window is stated here must match what the privacy page says. **OWNER:
confirm the deployed build + window.**

---

## 2. Requirements-vs-receipts matrix (for the technical review stage)

| OR requirement | Status | Receipt |
|---|---|---|
| OpenAI-compat /chat/completions, streaming | DONE | docs/SERVING.md (official openai SDK validation); serve-smoke battery |
| Usage tokens, stream + non-stream | DONE | final stream chunk carries usage (unit-pinned, main.rs); non-stream usage_json |
| /models with pricing/ctx/features/locations | DONE (pricing stub "0" until owner sets prices) | `research/serve-tail-20260804/v1-models.json` + RESULTS.md item 1 |
| Tools + streamed tool_calls | DONE | v0.67.0 serve surface; gap-scan F-items closed |
| Reasoning in + OUT separation (all 5 hy3 incumbents have it) | DONE | F13 closed, commit 0b2f4681 — see §3 |
| X-RateLimit-* headers | DONE | serve-tail item 2, live hammer receipts |
| Graceful drain (uptime protection on deploys) | DONE | serve-tail item 3, live SIGTERM sequence |
| Early 429 + Retry-After (recommended posture) | DONE | admission shed receipts (serve-tail, fleet lanes) |
| Monthly invoicing (they pay us) | Process answer, not code | state "monthly invoicing supported" |
| Privacy policy + ToS URLs | **MISSING — the only blocker** | §4 |
| Uptime evidence | Pre-launch honest answer | 140-min endurance soak: 464,870 req, 0 err, 0 shed (fleet-endurance-20260803); no public 30-day stat yet — say exactly that |
| Pricing (USD strings, set by us) | **OWNER decision** | anchors: SPEC.md §9 — 27B/32B class clusters $0.08–0.15 in / $0.16–0.60 out; suggested shape: interactive upper-middle (e.g. mid-$0.20s out), harvest ~50%, cached input 25%. Draft only — owner picks numbers; /v1/models metadata must match exactly |

Infra story (honest, for review conversation): current receipts are one RTX PRO 6000
(G7e) + an 8x H100 research box; the launch fleet is an RTX PRO 6000 trajectory with
honest low capacity_tpm at listing time. Do not claim a fleet that isn't racked.

## 3. F13 verify (the HF lane's flagged item) — CLOSED, verified 2026-08-05

The gap-scan (research/gap-scan-20260802/REPORT.md F13) flagged: `reasoning_effort`
accepted IN but `<think>` text streamed as plain `content` OUT — a listing-blocker
for a reasoning-model SKU (all five hy3 incumbents expose
`reasoning`/`reasoning_details` + `include_reasoning`).

Verified closed on `restructure/public-split`, commit **0b2f4681** ("feat(serve):
reasoning output separation — think text to the OR reasoning field", 2026-08-03):

- Streams: think text goes out as `delta.reasoning` chunks, never content
  (main.rs SSE path, `Piece::Reasoning` arm).
- Non-stream: `message.reasoning` + `message.reasoning_details`
  (`[{type:"reasoning.text", text}]`), content is post-think only (main.rs
  blocking_response; unit-pinned).
- `include_reasoning:false` / `reasoning.exclude:true` honored (separated AND dropped).
- Splitter arms on EVERY chat request whose rendered prompt ends with an open think
  tail (`ToolStreamParser::reasoning_only`); non-think models keep byte-identical
  no-parser streams.
- Re-verified this date: full memra-server unit suite green — **48 passed, 0 failed**
  (includes the reasoning-separation tests: char-by-char split, exclude-drop,
  unclosed-think flush).

No gap remains; no fix lane needed. (The serve-smoke gate also asserts the emitted
stream = reasoning + content, commit a8f71123.)

## 4. What ONLY the owner can do — submission checklist

Ordered; 1–4 are prerequisites of the form's own required fields, 5 is the click.

1. **Register the domain + minimal site** — darklanes.ai (available per DOMAINS.md;
   purchase is an owner call, SPEC.md §12). Fields 2/8/9/10/11 all want URLs on it.
2. **Publish privacy policy + ToS pages** — the known paper-not-code item
   (or-provider REPORT §5, hf-assessment action #2). Retention window stated there
   must match the §1c Data Policy answer.
3. **Stand up the public API endpoint** (base URL + /models reachable from outside)
   on a build carrying the metering surface (darklane/main 73d3adf4 class) — OR's
   review will hit these URLs.
4. **Decide** — legal/company name (field 1), contact email (field 3, becomes the
   Slack Connect invite), inference-location country codes (field 14), HQ wording
   (field 15), pricing numbers (feed /v1/models before review traffic), data-policy
   retention window (§1c).
5. **Submit** at openrouter.ai/providers/apply/form — paste §1 answers, tick
   Text-only modality, tick the three §1a feature boxes, click "Submit Application".
6. **After submit**: watch the email for the Slack Connect invite; expect queue time
   (open-weight providers are the non-prioritized class); when review starts, the
   test-traffic stage measures latency/throughput/error handling against the live
   endpoint — have the fleet in its honest launch config before that.
