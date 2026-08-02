# Inference gap scan — what else is missing, in the cache+tools class

Date: 2026-08-02/03 (web sources fetched on those dates; source dates inline). No GPU used.
Owner question, verbatim: *"what are we missing that needed in inference — same as we missed the
cache and the tool calls, what else is missing that we need?"*

Method: (a) inventory the actual serve surface on this branch (`crates/memra-server/src/main.rs`
1,153 lines, `worker.rs` 2,179, `toolcall.rs` 366; `docs/SERVING.md`; `tools/serve-proxy.py`),
(b) sweep five axes against what serious API consumers, the major engines (vLLM/SGLang/TRT-LLM),
the API leaders (OpenAI/Anthropic), and OpenRouter's provider requirements demand in 2026,
(c) grade every gap: severity (listing-blocker / revenue-lever / nice-to-have), implementation
size against our stack, and which customer class needs it.

Already-queued roadmap items are NOT counted as misses (per lane brief): constrained decoding /
`json_mode` / `structured_outputs`, the OR-schema `/v1/models` route, `parallel_tool_calls`.

---

## 0. What we HAVE (verified in-code, not from docs alone)

| Capability | Where | Receipt |
|---|---|---|
| `/v1/chat/completions` + `/v1/completions`, SSE streaming | `main.rs:444-456` | serve-smoke gate |
| `tools`, `tool_choice` auto/none, streaming `tool_calls` deltas, `finish_reason:"tool_calls"`, malformed-block verbatim policy | `main.rs:150-430`, `toolcall.rs` | `research/serve-tools-20260802/` (N=3 greedy, stream schema checker) |
| `reasoning_effort` + OR `reasoning` object (input side) | `main.rs:344-371` | tested `main.rs:1014-1034` |
| Cross-request prefix cache, continuation pool, `cached_tokens` usage split on every shape incl. SSE final chunk | `main.rs:214-225`, worker | `research/prompt-cache-20260802/gate-exact.jsonl` (16/16+16/16) |
| `cache_salt` per-tenant isolation (vLLM convention) | `main.rs:245-256`, `worker.rs:80-85` | `research/pc-iso-20260802/` |
| `max_completion_tokens` alias | `main.rs:159` | — |
| `stop` string/array/null, excluded from non-stream text | `main.rs:131-148,802-806` | tests `main.rs:1121-1152` |
| Early 429 + Retry-After at proxy, bounded FIFO deadline | `tools/serve-proxy.py:255-262` | OR-recommended posture (for-providers doc, fetched 2026-08-02) |
| VRAM-aware admission (waits, never mid-stream OOM) | `worker.rs:785-828` | `research/fast-router-20260802/` |
| `/health`, `/metrics` (+ proxy health loop, breaker, chaos receipts) | `main.rs:492-512` | `research/fleet-v060-20260801/SUMMARY.md` |
| Bearer auth, context-length 400 with token counts quoted | `main.rs:633-639`, `worker.rs:1330-1334` | — |
| Bit-exact isolation contract under batching; deterministic seed | SERVING.md | gate battery |

---

## 1. Axis findings

### 1.1 API surface

**F1 — OpenAI response-envelope non-compliance: `id` and `created` missing from every
`chat.completion` and `chat.completion.chunk`; error body is `{"error": "<string>"}` not the
OpenAI `{"error": {"message", "type", "code"}}` object; no `role` in the first stream delta;
no `system_fingerprint`.**
Verified: `main.rs:743-745, 768-777, 850-856` build the JSON with only
`object/model/choices/usage`. The official OpenAI SDKs validate responses with pydantic —
`ChatCompletion`/`ChatCompletionChunk` require `id: str` and `created: int`, so the standard
`openai` client **throws a validation error on our responses**; strict frameworks discard the
whole response (exact failure class shown in pydantic-ai issue #3994, 2026-01-13: *"5 validation
errors for _OpenRouterChatCompletion — id: Input should be a valid string"*). OpenRouter's stage-2
technical review is "API compat" testing (apply page, fetched 2026-08-02).
**Severity: listing-blocker.** The single most embarrassing possible find in test traffic —
"OpenAI-compatible" fails the OpenAI SDK. **Size: hours** (uuid + unix ts + build-hash
fingerprint + error-object wrapper + first-chunk role, ~100 lines in `main.rs` + tests).
Also fold in: echo/generate `x-request-id` header (vLLM does this in `serving_engine.py` —
docs.vllm.ai v0.9.1) for support/tracing.

**F2 — `max_tokens` omitted ⇒ silent truncation at 128.**
`default_max_tokens() -> 128` (`main.rs:197`). OpenAI's default when omitted is
model-maximum/context-bounded (help.openai.com, "Controlling the length of model responses");
SGLang shipped this exact bug class (sglang issue #582, 2024-07-02). Most agent frameworks omit
`max_tokens`. OR's baseline test traffic would see every long answer cut at 128 tokens with
`finish_reason:"length"` and score us as a broken endpoint.
**Severity: listing-blocker (quality-catastrophe class).** **Size: hours** — make the field
`Option<usize>`, default to context-remainder at admit (`worker.rs:1330` already computes the cap).

**F3 — sampling-parameter breadth: `frequency_penalty` / `presence_penalty` /
`repetition_penalty` are IMPLEMENTED in `SamplerConfig` (`memra-sampling/src/lib.rs:16-19`) but
never plumbed through the HTTP layer; `logit_bias` absent.**
Incumbent hy3 endpoints carry 8-17 params; DeepInfra/AtlasCloud/Novita all list the three
penalties + `logit_bias`/`min_p`/`top_k` (our own receipt
`research/or-provider-20260802/hy3-endpoints-api-20260802.json`). OR users filter with
`require_parameters` (provider-selection doc, fetched 2026-08-02) — every param we can't declare
is routed-around traffic. **Severity: revenue-lever.** **Size: hours** for the penalties (fields
exist end-to-end, it is request-struct plumbing); ~1 day for `logit_bias` (small sampler hook).

**F4 — unknown/unsupported request fields are silently swallowed.**
Neither request struct sets `deny_unknown_fields`, so `response_format`, `logprobs`, `n`,
`logit_bias`, `parallel_tool_calls`, `stream_options`, `user` are all accepted and ignored. For
semantic params this is worse than a 400: a client sending `response_format:{type:"json_object"}`
gets unvalidated free text and no error (contrast: we correctly 400 on `tool_choice:"required"` —
`main.rs:334`, "clean 400s, not silent downgrades" is already our stated policy; it just isn't
applied to fields serde never sees). **Severity: revenue-lever / trust.** **Size: hours** —
explicit reject-list for params we can't honor (until constrained decoding lands), keep
accept-and-ignore only for genuinely cosmetic fields (`user`, `stream_options`).

**F5 — `logprobs` / `top_logprobs`: absent.**
In OR's feature vocabulary and normalized request schema (`top_logprobs` — API reference, fetched
2026-08-03); required by lm-evaluation-harness for every loglikelihood/multiple-choice task
(EleutherAI harness `openai_completions.py`, main branch) — i.e., third-party evaluators
benchmarking our endpoint quality need it. Notably NOT carried by any of the 5 hy3 incumbents
(our endpoints receipt), so not table-stakes on that page. Our lean-logits device-sampling path
skips host logits transfer, so this costs a real (small) perf tax when requested.
**Severity: nice-to-have now, revenue-lever for eval-driven buyers.** **Size: 2-4 days**
(top-k logprob extraction on the sampled step + response plumbing, gated off unless requested).

**F6 — batch/async job API, embeddings, moderation endpoints: absent — and correctly deferred.**
OpenAI's 50%-discount class (Batch, and Flex `service_tier` at batch rates — OpenAI flex doc,
fetched 2026-08-03) is a direct-API revenue lever; OpenRouter routes none of it, and our SKU is
the OR listing first. Embeddings/moderation are different model classes. **Severity:
nice-to-have (direct-API SKU only).** No action for listing.

### 1.2 Streaming semantics

**F7 — no SSE keep-alive comments.**
`Sse::new(stream)` with no `.keep_alive(...)` (`main.rs:798`; axum 0.7 has the builder).
OpenRouter explicitly: send SSE keep-alive comments during long silent phases **or they cancel on
fetch timeout and fail over** (for-providers doc, fetched 2026-08-02 — cited in
`research/or-provider-20260802/REPORT.md` §1). Our silent window is real: long-prompt prefill
interleaves at PREFILL_TICK_T per tick, so a 100k-token prompt streams nothing for many seconds
before first token. An OR-side cancel is a **mid-stream failure — the one error class that
counts against uptime** (same doc). **Severity: listing-blocker.** **Size: one line** plus a
keep-alive interval choice.

**F8 — client disconnect does not cancel generation; billing on abort is wrong.**
Every token send is `let _ = s.tx.send(...)` (`worker.rs:1727,1761,1882,1950`) — send errors
ignored, no `tx.is_closed()` check anywhere in the tick loop. An aborted client burns GPU until
`max_tokens`/EOS, holds a session slot against admission, and the dl-metering lane would bill
tokens the caller never received. vLLM treats this as core correctness (abort-on-disconnect,
PR #11190 merged 2024-12; issue #10087). OpenRouter surfaces "Stream cancellation: Supported"
per endpoint and stops billing on abort for supported providers (OR streaming doc + zendesk
article, 2026-06-14) — a visible product flag we'd fail. **Severity: listing-blocker +
revenue-lever (wasted GPU under agent churn — agent frameworks abort constantly).**
**Size: ~1 day** — per-tick `s.tx.is_closed()` → `finish(s, Aborted)`, plus a metering rule
(bill-to-abort-point).

**F9 — streamed content leaks stop-sequence text.**
Non-stream truncates at the stop string (`truncate_at_stop`, `main.rs:802`), but the stream path
emits the token delta BEFORE the stop check (`worker.rs:1727-1730`), so streaming clients receive
the stop text (and any same-token overshoot) that non-stream clients never see. OpenAI semantics:
stop text excluded in both shapes. Stream-vs-non-stream divergence is exactly the kind of thing
marketplace test suites diff. **Severity: revenue-lever (correctness nit, cheap).**
**Size: hours** — hold back a `partial_suffix_len`-style buffer (the helper already exists in
`toolcall.rs`) before emitting.

Checked-and-fine on this axis: usage in the final stream chunk always present (OR hard
requirement — have); `finish_reason` correctness incl. `tool_calls` (gated); `[DONE]` terminator
present; tool-call streaming deltas schema-checked. One note: mid-stream worker errors go out as
a named SSE `event: error` (`main.rs:790-794`) — OpenAI clients only parse `data:` lines, so
they see a silent hang instead of an error; fold the fix into F1's error-shape work.

### 1.3 Serving robustness

**F10 — no per-request deadline/timeout server-side.** A stuck/slow session holds its slot; the
proxy has queue deadlines but nothing bounds an admitted request except token budget and context.
OpenAI SDK default client timeout is 10 min (flex doc, fetched 2026-08-03) — after which we're
generating for a dead socket (compounds F8). **Severity: nice-to-have once F8 lands** (disconnect
detection subsumes most of it). **Size: small.**

**F11 — no graceful drain.** Fleet restarts are SIGKILL-class; chaos receipts show in-flight loss
= the victim's cap (8/768, `research/fleet-v060-20260801/`). A SIGTERM handler that stops
admitting and finishes actives would zero the loss on planned deploys — OR counts mid-stream
errors against uptime on every deploy we do under load. **Severity: revenue-lever (uptime
stat).** **Size: ~1 day** (worker already has clean shutdown-on-channel-close; needs
stop-admission + await-actives + proxy `is_ready:false` signal).

**F12 — no `x-ratelimit-*` headers.** OpenAI-convention observability; nothing in OR requires
them (429+Retry-After is the requirement, and we have it at the proxy). **Severity:
nice-to-have.** **Size: hours** (proxy-side).

### 1.4 Marketplace-specific (OpenRouter)

Mostly covered by `research/or-provider-20260802/REPORT.md` (fetched 2026-08-02) and the queued
roadmap (OR-schema `/v1/models` with USD-string pricing, quantization disclosure, `capacity_tpm`,
`is_ready`). New findings this scan adds:

**F13 — reasoning OUTPUT separation is missing: we accept `reasoning_effort` in, but stream
`<think>` text as plain `content` out.**
Verified: the tools parser's think gate passes everything through *as content*
(`toolcall.rs:28-30, 99-107`; test at 304-311 asserts think text lands in content), and non-tools
chat has no think handling at all. OpenRouter normalizes reasoning into dedicated
`reasoning`/`reasoning_details` response fields with `include_reasoning` control (OR
reasoning-tokens doc, fetched 2026-08-03); **all five** hy3 incumbents list
`include_reasoning`+`reasoning` support (our endpoints receipt, 2026-08-02). A provider that dumps
raw `<think>...</think>` into `message.content` on a reasoning-model page looks broken to every
client and corrupts downstream agent transcripts. **Severity: listing-blocker for the hy3/Qwen
reasoning class — our exact SKU.** **Size: 1-2 days, template+parsing only, zero engine changes**
— the think-boundary scanner already exists in `toolcall.rs` (Prethink state, `</think>`
holdback); generalize it to all chat requests on think-class models and emit
`delta.reasoning`/`message.reasoning` (+ honor `include_reasoning:false` by dropping it).

**F14 — predicted outputs (`prediction: {type:"content", content}`): absent.**
In OpenRouter's normalized request schema (API reference, fetched 2026-08-03) and OpenAI's
latency-optimization surface. It is client-hinted speculative decoding — we own a full spec-decode
stack (MTP drafts, verify/rollback), so a prompt-provided draft source is a natural arm nobody
else in the hy3 provider set advertises. **Severity: nice-to-have now, differentiator later
(agent edit-loops: code editing is the canonical workload).** **Size: ~1 week** (new draft
source + acceptance gate in the spec tier).

**F15 — per-request priority tiers (`service_tier`): absent, and fine for OR** (OR expresses
priority as `:nitro`/`:floor` routing variants on their side). Relevant only to a future direct
API SKU alongside Batch (F6). **Severity: nice-to-have.**

Checked-and-fine on this axis: usage accounting incl. `cached_tokens` at the 25%-of-input OR
billing convention (have; the margin lever per or-provider report); early-429 posture (have);
uptime/chaos receipts (have); quantization disclosure + capacity signal (queued `/v1/models`
work); privacy/ToS pages (known paper-not-code item in or-provider report §5); BYOK — an
OR-user-side feature (5% fee, TrueFoundry 2026-06-25), nothing provider-side to build.

### 1.5 Agent-era features (the class the question is really about)

**F16 — Anthropic-style explicit `cache_control` blocks: NOT a gap for our lane.** The
OpenAI/OR convention is implicit prefix caching + `cached_tokens` receipts (OR prompt-caching
blog, 2026-07; Anthropic requires explicit breakpoints — platform.claude.com prompt-caching doc,
fetched 2026-08-02 — but that's their API dialect, not the OpenAI-compatible one we serve).
We already have the implicit tier + salt isolation. Checked-fine.

**F17 — sticky sessions / `user` tracking: OR-side feature.** OR folds a hashed `user` upstream
and uses `session_id` for sticky provider routing to keep caches warm (OR user-tracking doc +
prompt-caching/sticky-routing blog, fetched 2026-08-03). Provider-side obligation: accept the
field, don't break on it (we do), keep per-tenant salts documented (done, SERVING.md). Checked-fine.

**F18 — Responses API (`/v1/responses`): watch, don't build.** OR now exposes it user-side
(Portkey gateway issue #1527, 2026-02-18) but translates to chat/completions for providers; no
provider requirement found (for-providers doc, fetched 2026-08-02). Server-side conversation
state would collide with our session/cache architecture for zero listing credit today.
Checked-fine with a watch flag.

**F19 — MCP-native serving, computer-use tool types: not provider-surface.** MCP is
client/gateway-side; computer-use tool types are model-capability declarations that ride the
existing tools surface. No engine/server work indicated by any provider checklist found.
Checked-fine.

**F20 — server-side context compaction: OR runs it as their own plugin** (`context-compression`
in the OR plugin list, API reference fetched 2026-08-03). Not provider-side. Checked-fine.

---

## 2. THE TOP-5 — misses in the cache+tools class

Things that would embarrass us at listing time or leave money on the table, ranked. "Same class"
= invisible in every benchmark we run, table-stakes to the first real API consumer who connects.

| # | Miss | Severity | Size | Who needs it | Why it's the same class of miss as cache/tools |
|---|---|---|---|---|---|
| 1 | **OpenAI envelope compliance** — `id`/`created`/`system_fingerprint` fields, error-object shape, first-delta `role`, `x-request-id` (F1) | Listing-blocker | Hours | Every SDK client; OR technical review | We built "OpenAI-ish" and never ran the actual `openai` SDK against it — the official client pydantic-rejects every response we emit; found by inspection, would've been found by OR's first test request |
| 2 | **Reasoning output separation** — `<think>` streams as `content`; no `reasoning`/`reasoning_details`/`include_reasoning` (F13) | Listing-blocker (reasoning-model SKU) | 1-2 days, template+parse only | hy3/Qwen listing; all 5 incumbents have it | Exact mirror of the tools miss: a template/parsing surface feature, zero engine work, that 100% of the target model page's incumbents ship — we did the input half (`reasoning_effort`) and missed the output half |
| 3 | **Disconnect abort + SSE keep-alive + billing-on-abort** (F8+F7) | Listing-blocker + revenue-lever | ~1 day + 1 line | OR uptime/cancellation flags; agent frameworks that abort constantly | Like cache before we built it: every aborted agent request silently burns GPU we could sell — invisible in benchmarks (benchmarks never hang up), a metered loss and an OR-visible product flag in production |
| 4 | **`max_tokens`-omitted ⇒ 128-token truncation** (F2) | Listing-blocker (quality catastrophe) | Hours | Any client that omits `max_tokens` — most agent frameworks | A defaults blind spot exactly like prompt-cache-off-by-default: harness always sets `max_tokens`, real clients don't, and the first framework user sees a "broken model" |
| 5 | **Parameter breadth + honesty** — plumb the already-implemented penalties + `logit_bias`; explicit 400s for silently-swallowed semantic params (`response_format`, `logprobs`, `n`) (F3+F4) | Revenue-lever | Hours-1 day | OR `require_parameters` filtering; JSON-mode clients | The sampler feature exists and the HTTP surface hides it — capability built but not sold (the cache-tools pattern inverted); meanwhile silently ignoring `response_format` hands unvalidated output to a client we never warned |

Runners-up (not top-5): `logprobs` (F5 — eval-harness buyers), streamed stop-text leak (F9 —
cheap correctness), graceful drain (F11 — uptime stat), predicted outputs (F14 — the one
differentiator candidate).

## 3. Checked-and-fine count

**25 items checked and found fine** (have it, gated, or verified not-required for our
listing class): usage on stream+non-stream · `cached_tokens` split · `cache_salt` isolation ·
explicit `cache_control` not needed (implicit-caching dialect) · `max_completion_tokens` alias ·
stop string/array/null forms · stop excluded from non-stream text · finish_reason mapping ·
tool-call streaming deltas · malformed-tool-call verbatim policy · tools capability-gate 400 ·
`reasoning_effort`/`reasoning` input mapping · early 429 + Retry-After · health/metrics/breaker ·
VRAM-aware admission (no mid-stream OOM) · context-length 400 with token counts · multimodal
content parts clean-400 (text-only models) · bearer auth · seed determinism (stronger than
industry; only the fingerprint field missing → folded into F1) · fleet chaos/uptime receipts ·
`n>1`/`best_of` (not in OR's normalized schema) · MCP-native serving (client-side) · server-side
context compaction (OR-side plugin) · sticky sessions/`user` tracking (OR-side) · batch API +
`service_tier` tiers (direct-API SKU only, correctly deferred).

## 4. Source index

In-repo receipts: `crates/memra-server/src/{main,worker,toolcall}.rs` (line refs inline, read
2026-08-02/03) · `docs/SERVING.md` · `tools/serve-proxy.py` ·
`research/or-provider-20260802/REPORT.md` + `hy3-endpoints-api-20260802.json` ·
`research/cache-tools-20260802/REPORT.md` · `research/serve-tools-20260802/` ·
`research/pc-iso-20260802/` · `research/fleet-v060-20260801/SUMMARY.md`.

Web (fetch dates in text): OpenRouter API reference — openrouter.ai/docs/api_reference/overview
(fetched 2026-08-03) · OR reasoning-tokens doc (fetched 2026-08-03) · OR for-providers +
provider-selection + apply (fetched 2026-08-02, via or-provider report) · OR streaming doc +
zendesk stream-cancellation article (2026-06-14) · OR user-tracking doc + prompt-caching/
sticky-routing blog (fetched 2026-08-03) · OpenAI flex-processing doc (fetched 2026-08-03) ·
OpenAI response-length help-center article · pydantic-ai issue #3994 (2026-01-13) · sglang issue
#582 (2024-07-02) · vLLM PR #11190 (2024-12) + issues #10087/#20798 · EleutherAI
lm-evaluation-harness `openai_completions.py` (main) · platform.claude.com prompt-caching doc
(fetched 2026-08-02) · Portkey gateway issue #1527 (2026-02-18) · TrueFoundry OpenRouter pricing
(2026-06-25).
