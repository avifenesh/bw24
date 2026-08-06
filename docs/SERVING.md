# Serving — the OpenAI surface and the replica fleet

This is the serve-surface doc: fleet topology and measured throughput, the isolation
contract, the OpenAI tools surface, the gateway listing surface (`/v1/models` schema,
rate-limit headers, graceful drain), safetensors/FP8 checkpoint serving, cross-request
prompt caching with per-tenant `cache_salt` isolation, and the honestly-stated numeric
edges of batched serving.

> Numbers here are engineering receipts, each labeled with its rig — see
> [Rigs](PERFORMANCE.md#rigs--what-was-measured-on-what) for what each label is. A number
> without its rig label is not a number: the same cell moves 5-12% between two pods of the
> same SKU and ~2x between a 188-SM and an 82-SM board. The open gaps stated below travel
> with the wins.

memra's engine owns one GPU per process (`Engine::new(0)`; `CUDA_VISIBLE_DEVICES` is the
placement mechanism). Multi-GPU serving is therefore a **replica fleet**: N `memra-server`
processes fronted by an admission proxy. Tensor parallelism is a separate in-progress build
(M0 comms floor measured — ARCHITECTURE-H100.md).

## Fleet tooling

(Not to be confused with the OpenAI `tools` API surface — that is
"[OpenAI tools surface](#openai-tools-surface-serve-tools-lane-2026-08-02)" below.)

| tool | what it does |
|---|---|
| `tools/serve-fleet.sh start\|stop\|status\|restart` | declarative fleet supervisor: brings up `REPLICAS_PER_GPU` replicas per GPU in `GPUS`, fronts them with the proxy, health-loop restarts anything that dies. systemd-free; pidfiles under `$FLEET_RUN` |
| `tools/serve-proxy.py` | least-outstanding reverse proxy with per-backend admission cap (default 8 = the engine's exactness-tier batch width and the two-replicas-per-GPU anti-thrash bound). Bounded FIFO queue with deadline → 429 + Retry-After; `/health` + `/metrics` JSON |
| `tools/load-serve.py` | concurrent OpenAI-format load harness: aggregate output tok/s, p50/p95 latency, JSONL per load point |
| `tools/serve-smoke.sh` | OpenAI-surface smoke gate for a single server |

## Measured numbers (Qwen3.5-9B Q8_0; receipts in `research/`)

- **Single replica (H100, rented pod):** temp-0.7 c=8/16/32 medians **654/657/659 tok/s** after the
  batched decode tick (z-batched FA + KV append, device sampling, lean logits — +25-36%
  over the pre-batched tick; N=4, `research/batched-tick-inc2-20260801/`; chunk-8 era —
  see the exact-16 tier below).
- **Managed fleet, 3 rented H100s x 2 replicas (v0.60-validated):** **1,477 tok/s** through the
  admission proxy at c=96 (N=2 interleaved passes: 1477.0/1473.1), zero 429s/5xx —
  managed now matches the v0.59-era 1,480 direct number (the ~7% admission-overhead gap
  closed at the fleet level). Chaos-tested: SIGKILL a replica mid-load, breaker DOWN the
  same second, supervisor restart +2s, backend UP +9s, 8/768 requests lost (exactly the
  victim's in-flight cap), aggregate across the kill 1,487 tok/s; greedy hash identical
  on all 6 replicas in every condition, 18/18 (`research/fleet-v060-20260801/SUMMARY.md`).
  The proxy cap (8) was calibrated on the v0.59 core — the cap re-sweep is pending the
  next box window (stale-verdict risk flagged in the validation summary).
- **Single replica (RTX 5090 Laptop — the local rig, exact-16 tier):** with the Q8_0 split-plane mirror
  (`MEMRA_Q8RP=1` on 24GB; Hopper default), the worker auto-selects decode chunk 16 —
  c=16 median **494.5 tok/s vs 416.4** at chunk 8, same mirror, interleaved N=4
  (**+18.8%**; +33.8% vs the mirror-less baseline); c=32 at `MEMRA_CTX=2048` runs
  **502.1** with 128/128 ok (single run; `research/batched-tick-inc3-20260801/`).
- **Solo (c=1) decode rides the naked m=1 program** (serve-path phase 2, 2026-08-05): a lone
  session's tick runs `decode_layers_eager` verbatim instead of the batched body, inheriting
  the whole m=1 fusion chain. Order-paired N=5 on the 5090 (82 SM), decode-only `step_p50`:
  **+8.33%** on the 9B (123.7 → 134.1 tok/s, 5/5 wins) and **+5.19%** on the 27B (43.6 → 45.8,
  5/5); c=8 saturation flat (−0.00% / −0.18%). This closes the class of c=1 gap phase 1
  measured against naked decode — serve c=1 now sits level with the same-board `run-gen`
  denominator (134.8/134.5/134.0). Notably it also **retires the solo graph door as a win**:
  `GraphSession` replay amortized the same launch overhead this removes outright, so with the
  fast path in place the door is a net loss at every length measured out to mt=1024 and
  `MEMRA_GS_MIN=384` must NOT be lowered (FLAGS §serve; `research/servepath-p2-20260805/`).
- **Spec fast lane:** MTP speculative serving is a single-stream latency tier — 1.82x plain
  serving at c=1 on the 27B (131.8 vs 72.5 tok/s); plain batching overtakes between c=2 and
  c=4, so spec and bulk tiers run as separate server processes (`MEMRA_SERVE_SPEC`;
  `research/spec-serving-20260801/`).
- **The plain-serve c=1 gap (task #70) is closed by the fast path above — with one cell
  pending re-measure and one still open.** Phase 1 measured serve c=1 trailing the naked
  CLI **−11.74%** on a Q8_0 27B cell (`memra-server` 46.09 tok/s, N=3 median, vs `run-gen`
  naked 52.22, single run; rig `pro6000wk-runpod-community`, same commit and prompt); the
  measured cause — B=1 ran the batched body and missed the m=1 fusion chain — is exactly
  what `MEMRA_SERVE_B1FAST` fixes, and on the 82-SM 5090 serve c=1 now sits level with the
  same-board `run-gen` denominator. The −11.74% number itself is **pre-fix** and the 188-SM
  cell has not been re-measured since — do not quote it as current. Still open: the NVFP4
  **spec** serve path at **−8.66%** (serve 170.55 vs bare 186.72, rig `pro6000wk-runpod`,
  also a pre-H3 measurement) — the spec tier runs its own burst loop that the `b_n==1`
  fast path does not touch. Receipts: `research/q27-deepdive-20260805/RESULTS.md` §4
  (phase 1), `research/servepath-p2-20260805/RESULTS.md` (the fix + the H1 refutation).

## The isolation contract

Greedy serving is **isolated-identical under concurrent load at defaults**: a request's
output tokens are byte-identical whether it arrives alone or inside a full batch. This is
gated, not assumed — the serve gate replays the same prompts at c=1 and c=16 and
byte-compares every stream.

The contract is over **tokens**, not over the FP program that produces them, and since
2026-08-05 (`MEMRA_SERVE_B1FAST`, serve-path phase 2) a solo tick deliberately runs a
*different* program from a batched one: at `b_n==1` the tick uses the m=1 fused trunk
(`decode_layers_eager` — the same code `run-gen` runs), while `b_n>=2` uses the batched
body. Those two carry the long-accepted decode-config FP-composition gap
(`decode-batch-gate`'s jurisdiction), so the guarantee is exactly as stated — token
streams match — and is not a claim of bit-identical logits between a solo and a batched
tick. The direction is deliberate: a c=1 request now computes what the CLI computes
(strict bit-identity to `decode_step_h`, gated), which is why `serve-st-gate`'s
CLI-vs-server greedy token-stream check is a *stronger* assertion than before. Verified at
the stream level on q9 NVFP4-MTP: greedy 150 ids and seeded-sampled identical to the
`run-gen` oracle and across both arms (`research/servepath-p2-20260805/`). It is also a fixed defect, not a freebie: the batched
cuBLASLt prefill router and shared-expert-gate GEMMs were m-dependent, so under
cross-request prefill batching a MoE request's own expert selection changed with its
co-arrivals (the supported Qwen3.6-35B had the same defect as the onboards whose serve
gate exposed it — Ornith-35B 6/16, KAT 7/16, both 16/16 after the fix). Default
`MEMRA_ROUTER_PREFILL_EXACT` routes prefill through decode's m-invariant router/gate
kernels; a bit-identical batched twin recovers most of the prefill cost
(`MEMRA_ROUTER_BATCH`, FLAGS §3). Receipts: `research/concat-prime-exact-20260802/`,
`research/fast-router-20260802/`.

**Admission is VRAM-aware** (2026-08-02): once the first admitted session reveals the
model's per-session VRAM cost, further admissions require free ≥ 2x that cost — otherwise
the request *waits* in the same never-rejected FIFO as the session-count cap instead of
failing with a cache-alloc OOM (the c=16 8192-ctx failure mode under resident-if-fits,
caught by the serve gate as instant HTTP 400s — `research/fast-router-20260802/RESULTS.md`).
The first session always admits; an OOM with no active sessions is real capacity and
still errors loudly, with the CUDA error quoted.

## The exact-16 decode chunk tier

The batched tick decodes sessions in per-model chunks. Default width is **16 on models
where every matmul has a bit-exact 16-batch kernel class** (`decode_batch_exact16_ok`:
the b16 batched-mmvq family — Q8_0 qualifies only through its `_rp` mirror twin), **8
otherwise**; `MEMRA_DECODE_BATCH_CAP` stays the explicit measurement door. Qualifying
steps scope out every m>=16 GEMM/MMQ arm, so chunk-16 output is bit-identical to
isolated decode (gate2 bit-checked at steps 32 and 160). B=32 has no exact kernel class
— chunk policy stays <=16. On the H100 fleet model (9B Q8_0, mirror on by default) the
tier engages automatically on the next deploy; the H100 numbers above are chunk-8-era
and the chunk-16 fleet effect is pending on-box re-validation.

**Capacity envelope (24GB):** the mirror costs ~model-size VRAM, so c=32 sessions at the
default `MEMRA_CTX=8192` exceed VRAM (captured `CUDA_ERROR_OUT_OF_MEMORY` in the
pre-admission-wait receipts; ~27 sessions fit — since the VRAM-aware admission wait the
overflow queues instead of erroring). Set `MEMRA_CTX` to the workload — 2048 clears the
same cell (machine-specific config per the flags doctrine).

## OpenAI tools surface (serve-tools lane, 2026-08-02)

`/v1/chat/completions` accepts `tools`, `tool_choice` (`"auto"`|`"none"`; `"required"` and
named-function forms 400 — the grammar engine isn't wired to tool selection yet),
assistant-history `tool_calls`,
`role:"tool"` result turns, and `reasoning_effort`/`reasoning`. The path is **template +
parsing only — zero engine changes**:

- Tool schemas render into the model chat template's own `<tools>` branch (the qwen3.5/3.6
  ChatML convention: schemas JSON in the system region, `<tool_call>`/`<function=…>` call
  format, `<tool_response>` result turns), byte-per-byte per the committed template dumps
  (`research/onboard-ornith-20260801/templates/`). Models whose template has no tools
  branch (hy3 dialect, gemma4, bare ChatML) reject `tools` with a 400 at admission.
- Emitted `<tool_call>` blocks are parsed from the generated stream into OpenAI-shape
  `tool_calls` (streaming deltas + non-stream `message.tool_calls`, deterministic ids,
  `finish_reason:"tool_calls"`); argument values coerce per the declared JSON-schema types.
  **Malformed policy:** a block that does not parse is surfaced verbatim as content — never
  an error, never dropped bytes; unterminated blocks flush raw at end of generation.
- `reasoning_effort` `none|minimal|low` → the template's `enable_thinking=false` no-think
  switch; `medium|high`/absent → the template default (open `<think>`). Models without the
  switch ignore the parameter. `reasoning: {enabled, effort}` (OpenRouter form) maps the
  same way.
- **Isolation:** non-tools traffic bypasses the tools renderer AND the emission parser
  entirely (legacy render path, byte-identical streams); tools traffic is generation-
  identical for the identical rendered prompt (raw-completions bijection gate). `usage`
  now carries worker-truth `prompt_tokens` (rendered tools block included) +
  `completion_tokens` + `total_tokens` on stream and non-stream shapes, with the
  prompt-caching split (`prompt_tokens_details.cached_tokens`) — see "Prompt caching"
  below. Tools requests cache like any other: the prefix cache keys on the rendered
  prompt's token ids, so a repeated tools block is a cacheable prefix (no special-casing).

Receipts: `research/serve-tools-20260802/` (round-trip transcripts N=3 greedy on
Qwen3.6-35B + AgentWorld, streaming schema checker, malformed-policy transcript,
tok-check usage crosscheck, cross-binary c1 refs + c1-vs-c16) and
`research/integrate-cache-20260802/` (tools x cache intersection gate).

## OpenAI compatibility contract (serve-compat lane, 2026-08-03)

The five gap-scan listing-blockers (`research/gap-scan-20260802/REPORT.md`), fixed and
gated by the official `openai` Python SDK against a live server
(`research/serve-compat-20260802/`):

- **Envelope:** every OpenAI-shape completion and stream chunk carries `id`
  (`chatcmpl-…`/`cmpl-…`), `created`, and `system_fingerprint` (`memra-<git sha>`, baked
  at build); the id echoes as the `x-request-id` response header. The first stream delta
  carries `role:"assistant"`. Error bodies are the OpenAI object —
  `{"error": {"message","type","param","code"}}` — and mid-stream worker errors arrive as
  a final `data:` error chunk + `[DONE]`, never a named SSE event. SSE keep-alive
  comments flow every 5s (long-prompt prefill streams nothing before first token;
  OpenRouter cancels silent streams).
- **Reasoning separation:** on think-open prompts, `<think>` text routes to
  `message.reasoning` / `delta.reasoning` (+ `reasoning_details`, the OpenRouter
  dialect); `content` is post-think only. `include_reasoning:false` (or
  `reasoning: {exclude: true}`) drops the separated text. Non-think models keep
  byte-identical no-parser streams.
- **`max_tokens` omitted** ⇒ context-bounded budget (session ctx − prompt, capped at the
  model's trained context) — the OpenAI default-when-omitted semantics, not a silent
  128-token truncation. Explicit `max_tokens`/`max_completion_tokens` honored exactly.
- **`temperature` omitted ⇒ 1.0; `seed` omitted ⇒ fresh per request** (dogfood F4,
  2026-08-04). Both were `#[serde(default)]`, which is `0.0` for `f32` and `0` for `u64` —
  and both of those zeros are *meaningful values*, not "unset": temperature 0 is greedy
  argmax and seed 0 is a valid fixed stream. An omitting client (the OpenAI SDK's
  documented leave-it-out path, and this repo's own agentic driver) therefore got
  deterministic decoding pinned twice over: same context in, same token out, identical
  tool-call cycles forever. Now `temperature` omitted is OpenAI's documented 1.0 and an
  omitted `seed` draws fresh entropy per request. **Explicit values are honored exactly,
  including `temperature: 0` (greedy) and `seed: 0`** — every determinism gate in `tools/`
  and `research/` sends both explicitly, so all of them keep their behavior. Supply a
  `seed` whenever you want reproducibility; omit it to get variation.
  Corollary worth knowing: an omitted-`temperature` request is *pure* temperature-1.0
  sampling (`top_p` 1.0, `top_k`/`min_p` disabled, penalties off), which is exactly the
  regime that keeps the in-graph sampled draft chain — so the OpenAI default lands on the
  fast sampled-spec path, not a slow fallback.
- **Disconnect abort:** a hung-up client's session retires at the next tick (all serve
  paths: batched, graph, spec, legacy) and is billed to the abort point (the `[abort]`
  log line records prompt/cached/generated); queued requests from dead clients never
  reach the GPU.
- **Parameter breadth + honesty:** `frequency_penalty`/`presence_penalty`/
  `repetition_penalty` plumb to the sampler (whole-history window; greedy+penalized
  keeps the host-sampled path). `response_format` `json_object`/`json_schema` is REAL
  constrained decoding (see the section below). Semantic params we can't honor 400 with
  the param named (`logit_bias`, `logprobs`/`top_logprobs`, `n != 1`, `best_of != 1`,
  unknown `response_format` types); cosmetic fields (`user`, `stream_options`) are
  accepted and ignored. Streams exclude stop-sequence text exactly like non-stream
  responses (holdback buffer).

## Gateway listing surface (serve-tail lane, 2026-08-04)

The OR-listing tail — the last three surface gaps between memra and a marketplace
gateway listing — is closed and battery-gated (`research/serve-tail-20260804/`):

- **`/v1/models` OR-schema:** each entry carries `context_length` (from the loaded
  plan's config), `architecture` (`modality`, `tokenizer`, `instruct_type` — probed at
  spawn from the model itself, not hardcoded), and an OR-convention `pricing` stub.
  Unknowns are honest `null`s, never guesses.
- **Rate-limit headers:** `X-RateLimit-Limit` / `-Remaining` / `-Reset` on both
  completion routes with concurrency-slot semantics — a per-lane atomic gauge whose
  RAII slot rides the SSE stream to completion, so `Remaining` is truthful for the
  whole life of a stream. Sheds carry `429 + Retry-After`; `MEMRA_RL_RESET_S` is the
  no-signal fallback for `Reset` (with traffic, Reset = mean tokens/request x p50 step
  latency).
- **Graceful drain:** SIGTERM flips `/health` to `"draining"`, new completion requests
  get `503 + Retry-After`, in-flight requests — streams included — run to `[DONE]`
  within the `MEMRA_DRAIN_S` deadline (default 30s), then the process exits 0. Live
  receipt: a 1024-token stream completed mid-drain.

## Safetensors checkpoint serving (serve-st + fp8-ship lanes, 2026-08-04)

`MEMRA_MODELS` accepts safetensors checkpoint directories (`config.json` +
`model.safetensors[.index.json]`) and repack dirs alongside GGUF paths — validated at
parse time (a bogus dir fails naming the missing file). Chat templates come from the
checkpoint's own tokenizer config (`from_hf_dir`); template-less dirs 400 with a pointer
to `/v1/completions`. Official Qwen FP8 block-128 checkpoints load bit-exact (GPU
dequant, load wall 843.9 → 291.6 s = **2.89x faster load**) and **spec decode runs out of
the box on the checkpoint's embedded MTP head** — **128.06 tok/s** from the checkpoint's
own `mtp.safetensors` (**2.61x** the same-run plain 48.99), 136.75 with an own-trim
drafter, on rig **`rig2x5090-serve`** (rented 2x RTX 5090; there is no official-FP8 cell
on any RTX PRO 6000 board — do not merge the two). The win is **load time, not decode
throughput**: the e4m3-resident arm is flat by construction (weights dequantize onto the
Q8_0 arm), and spec **triples TTFT** on this arm (0.170 → 0.466 s).
Receipts: `research/fp8ship-20260804/official/`.

The ST-spec exactness scare (#68) was root-caused to a serve-side bug that was never
ST-specific: the per-session persistent draft graph replayed with dangling pool
addresses (capture transients not retained + the fa-partials pool freeing grown-past
buffers the capture baked) — reproducible on GGUF session bursts at n>=600 too. Fixed
via capture-retain keepers on `DraftGraphCtx` + retire-on-grow for the fa partials pool;
the quarantine is lifted and dir checkpoints are spec-eligible by default
(`MEMRA_SERVE_SPEC=0` is the rollback door). Gate: `tools/serve-st-gate.sh` — item 3 pins
the CLI ST-dir branch and the server to identical greedy token streams on a 64-token
window, and item 4 pins the DEFAULT (spec-on) server against the **tokenwise serve
oracle** (`MEMRA_SERVE_SPEC=0 MEMRA_SERVE_BATCH=0` — same worker, plain decode) at a
400-token window, prefix-tolerant for burst overshoot. Note what item 4 is *not*: the
comparator is deliberately the tokenwise **serve** arm, not the run-gen CLI, because both
the batched-plain path and the CLI carry their own accepted near-tie FP classes at long
windows (see [first-token cross-config
drift](#first-token-cross-config-drift-batched-prime--stated-honestly) below). Do not
restate this gate as "token-identical to the CLI oracle".

## Constrained decoding (`response_format`) — lanes constrained + constrained-full, 2026-08-03

`/v1/chat/completions` honors `response_format` `{"type":"json_object"}` and
`{"type":"json_schema","json_schema":{...,"schema":{...}}}` as REAL constrained decoding:
the schema compiles to an [llguidance](https://github.com/guidance-ai/llguidance) grammar,
and each step's packed token bitset uploads to a stable per-session device buffer (~31KB
H2D) where `mask_logits_f32` bans disallowed tokens on device — BEFORE the same
device-sample / lean-logits / CUDA-graph / speculative paths unconstrained sessions ride.
No path is lost to being constrained.

- **Cost:** plain constrained-greedy = **99.4% of unconstrained** (123.7 vs 124.4 tok/s,
  q9 N=3 same-session, local RTX 5090, Qwen3.5-9B NVFP4, 256-token greedy); per-step
  grammar compute 0.006–0.007 ms. **That 99.4% is the plain lane only — the speculative
  lane pays far more: 153.4 vs 194.4 = 79%.** Never quote 99.4% for the spec path. The
  remaining constrained-vs-unconstrained gap is draft acceptance under a tight grammar,
  not mask overhead.
- **Draft-side masking (lane/draft-mask, 2026-08-04):** the drafter is masked too. A
  constrained spec session clones the session's grammar matcher once per spec round
  (0.002 ms), advances the clone with each proposed token, and bans the illegal ids in the
  draft head's own logits — in-graph on the captured draft chain, permuted through `d2t`
  for trimmed draft heads. Proposals are legal by construction, so the verify-side
  truncation backstop (which stays, as the correctness backstop) stops firing:
  `gram_cuts` went 3/12, 3/15, 1/10, 28/30, 18/25 -> **0/N on every cell measured**.
  Bounded tight schema: acceptance 0.561 -> 0.651, 216.6 -> 227.5 tok/s (+5.0%, N=3 warm).
  Cells whose drafter already proposed legal tokens (json_object, loose prose) move inside
  noise; unconstrained traffic is inert. Rollback seam `MEMRA_DRAFT_MASK=0`. Receipts
  `research/draft-mask-20260804/`.
- **Exactness:** device-mask greedy is byte-identical to the host -inf oracle
  (`MEMRA_CONSTRAIN_HOST=1`), spec-constrained is byte-identical to plain-constrained,
  graphed is byte-identical to eager, draft-masking ON is byte-identical to OFF (greedy and
  seeded-sampled, 7 cells), and unconstrained requests are byte-identical to the pre-lane
  binary (the isolation contract). Kernel-check pins `mask_logits_col` bit-identity.
  One measured exception, documented because it is NOT a masking property: an unbounded
  schema that lets the model degenerate into arbitrary whitespace against a token cap has a
  draft-chain-SHAPE-dependent tail (verify batch shape T changes FP summation order, which
  flips argmax at the near-ties in that tail). The pre-lane binary shows the same
  divergence across `MEMRA_SPEC_K=3/2/1` on that cell with no draft-mask code present;
  with shape held fixed the arms are byte-identical. Bound the schema and it goes away.
- **Think interaction:** constrained requests force the template's no-think switch (a
  grammar masking from token 0 can never close an open `<think>` tail); a think-tail
  template without an `enable_thinking` switch is a loud 400.
- Unknown `response_format` types remain loud 400s. `/v1/completions` (non-chat) carries
  no `response_format`.

Receipts: `research/constrained-20260803/` (v1) + `research/constrained-full-20260803/`
(full battery: every path, cross-path identity, three-way perf).

## Prompt caching (cross-request prefix cache) — 2026-08-02

Two caching tiers serve prompt tokens without recomputing them:

1. **Continuation pool** (pre-existing, `MEMRA_KV_REUSE`): a retired session parks its whole
   (prompt + generation) state; a new prompt that EXACTLY EXTENDS it resumes. Single-use,
   exact-extension only — a new session that merely shares a system prompt always missed.
2. **Cross-request prefix cache** (`MEMRA_PREFIX_CACHE_MB`, default 256MB, 0 = off): compact
   device snapshots of primed state at token boundaries, keyed by the exact token-id prefix,
   per model, LRU under the byte budget. Entries are REUSABLE — a hit deep-copies the entry
   into the new session's cache, so one marketplace system prompt serves any number of
   sessions. Learning sequence for a shared-prefix pattern: request 1 seeds its full prompt,
   request 2 split-primes at the longest-common-prefix and inserts the boundary entry,
   request 3+ hit. Hybrid models are safe by construction: GDN conv/ssm state cannot be
   truncated to a shorter prefix, so the state is snapshotted AT the boundary while a fresh
   session primes — never rolled back.

**Exactness contract:** an entry stores the KV/recurrent bytes from WHATEVER prime config ran
(single, chunked, or concat batch-prime); decode from those bytes is deterministic, so a
cached hit is bit-identical to the run that computed the prefix — gated 16/16 partial-prefix
+ 16/16 full-prefix cached-vs-fresh greedy identity across depths
(`research/prompt-cache-20260802/gate-exact.jsonl`). Comparing a cached-hit stream against a
DIFFERENT prime config's fresh stream inherits the batched-prime near-tie first-token law
("First-token cross-config drift" below) — same documented class, reported not gated.

**Policy:** spec sessions bypass the prefix cache entirely (SpecSession owns trunk + draft
caches; a trunk-only prefix restore would leave draft state unprimed — the spec tier keeps
its own continuation pool). Legacy round-robin mode (`MEMRA_SERVE_BATCH=0`) also bypasses.
Sessions always win over the cache: a failed session-cache allocation evicts every entry and
retries before erroring.

**Per-tenant isolation (`cache_salt`) — PC-ISO, 2026-08-02:** every cross-request reuse
tier (prefix cache, continuation pool, spec pool) keys on (model, cache namespace), not
model alone. The namespace comes from the optional `cache_salt` string field on
`/v1/completions` and `/v1/chat/completions` (the vLLM `cache_salt` design, OpenAI-
compatible extension): requests only share cached prefixes with requests carrying the
SAME salt, in either direction, so `usage.prompt_tokens_details.cached_tokens` can only
ever reflect the caller's own namespace's history — the CacheProbe/PROMPTPEEK cross-tenant
hit-oracle mitigation (`research/cache-tools-20260802/REPORT.md` §1.4/§4). No salt = the
default `""` namespace: single-tenant deployments behave exactly as before (no new env
knob — the namespace is a request field, not a flag). The LRU byte budget stays GLOBAL
across namespaces (VRAM is one resource; only visibility is namespaced). A gateway
multiplexing many end-users through one API key — the marketplace listing shape — MUST set
a per-end-user/session salt. Gates: `research/pc-iso-20260802/` (same-salt hit, cross-salt
miss both directions, default-namespace blindness; the integrate-cache intersection gate
re-run unmodified as the no-salt regression).

**Accounting:** every response shape carries OpenAI-schema usage with the worker-truth split —
`usage.prompt_tokens`, `completion_tokens`, `total_tokens`, and
`prompt_tokens_details.cached_tokens` (tokens resumed from ANY cache tier: continuation pool,
spec resume, or prefix cache). `/metrics` exposes the cumulative split
(`prompt_tokens_in`/`cached_tokens_in`) plus `prefix_cache_hits`/`entries`/`bytes`. Cached
prefill costs ~0 to serve and bills at 25% of input on the OpenRouter hy3 endpoints — the
margin lever (`research/or-provider-20260802/REPORT.md`).

## Spec-decode acceptance telemetry (lane/accept-telemetry, 2026-08-05)

Always-on per-draft-position acceptance counters, the llama.cpp #26389 / vLLM spec-decode
counter schema. WHY: the 2026-08-05 dogfood head-to-head found short-context sampled
acceptance at 0.55 vs 0.73 full-draft — a posthoc dig that this surface turns into a live
gauge (drafter health on a new checkpoint is readable in minutes, and the K-policy work
gets a per-position decay curve for free).

**`GET /metrics` — the `spec` block**, per model, cumulative since the model loaded (models
load once per server process, so counters reset on restart, never mid-run). Absent until the
first spec burst — spec-off deployments see the exact pre-lane payload:

```json
"spec": {
  "q9": {
    "rounds": 118, "drafted": 354, "accepted": 213,
    "acceptance_rate": 0.602, "tokens_per_round": 2.805,
    "pos_drafted":  [118, 118, 118],
    "pos_accepted": [96, 71, 46],
    "accept_rate_per_pos": [0.814, 0.602, 0.390]
  }
}
```

`accept_rate_per_pos[j]` = P(draft position j accepted | a round offered position j) — healthy
spec decode decays monotonically from position 0 (acceptance is a prefix walk: position j can
only be accepted if 0..j-1 were). Arrays are trimmed to the deepest position ever drafted
(up to 8 tracked positions; totals count deeper drafts too). Normalization matches
`MEMRA_SPEC_STATS`: a p-min-cut chain token is counted in neither drafted nor accepted. The
opt-in round-stream arm (`MEMRA_SPEC_STREAM=1`) keeps its accept counts on device, so under
it per-position arrays cover the standard-path rounds only; totals stay complete.

**`usage.spec` — per-request summary.** Spec-decode requests carry their OWN
rounds/drafted/accepted + `acceptance_rate` in the response usage object (this request only —
pool-resumed sessions do not leak prior requests' counts). Additive and OpenAI-safe: official
SDKs ignore unknown usage fields, no existing field changes, and non-spec requests carry no
`spec` key at all.

**Cost:** host-side u64 adds at the round accounting the engine loop already does — zero
GPU syncs, zero per-token allocation, no hot-path lock (the worker merges per-burst deltas
into its own map; the metrics mutex is only taken on the existing 32nd-tick publish, plus a
force-publish when a spec session retires so one-shot requests are visible immediately).
Validation capture: `research/accept-telemetry-20260805/`.

## API keys — multi-key tenant auth (lane/api-keys, 2026-08-05)

Bearer auth that maps key → tenant, so cache isolation, QoS lane class, rate-limit
headers, and metering all key off a real tenant identity. Launch-shaped: a file-backed
keyring + a CLI, no web UI.

**Configuration.** `MEMRA_API_KEYS=/path/keys.toml` — TOML `[[keys]]` entries carrying
`sha256` (of the plaintext key — the plaintext is never stored), `tenant`
(`[A-Za-z0-9_-]+`), `lane` (`interactive` default | `batch`), `enabled`, and optional
`rate_limit`. An inline env form `tenant:sha256hex[:lane],...` exists for file-less
deploys. A malformed ring is a startup FATAL (never partially applied); the file
hot-reloads on mtime change (≤2s poll — chosen over SIGHUP: no signal thread, cannot be
missed), and a broken rewrite keeps the previous ring and logs loudly — auth never fails
open because of a typo.

**Lifecycle CLI.**
```
memra-server --gen-key acme [--lane batch] [--rate-limit 4] [--keys /path/keys.toml]
memra-server --revoke-key mk-acme-1a2b3c4d5e6f [--keys /path/keys.toml]
```
`--gen-key` prints the plaintext key (`mk-<tenant>-<48 hex>`) exactly ONCE on stdout and
appends the hash entry; `--revoke-key` disables by unambiguous prefix (or full key) — a
running server picks the revocation up on the next poll. `--keys` defaults to
`MEMRA_API_KEYS`.

**Request law.** `Authorization: Bearer <key>` on every `/v1` completion route:
- keyring match → that key's tenant context; **disabled key → 403** (actionable,
  distinct from unknown), **unknown key / missing header → 401**;
- `MEMRA_API_KEY` (the single static key — the daily driver and every serve script)
  keeps working unchanged as tenant `default`, with or without a keyring configured;
- neither configured → open (dev behavior), tenant `default`.

**What the tenant identity drives:**
- **Cache isolation:** with a keyring configured, the PC-ISO namespace is
  `t:<tenant>␟<cache_salt>` — one tenant's keys share cached prefixes, different tenants
  never do, and the `␟` (US, `\x1f`) separator is excluded from tenant ids so a
  client-controlled `cache_salt` cannot forge another tenant's namespace. `cache_salt`
  still sub-scopes WITHIN a tenant (a gateway multiplexing end-users through one key
  keeps setting per-user salts). No keyring → the raw-salt namespace, byte-identical to
  PC-ISO behavior.
- **QoS lane class:** `interactive`-class keys behave exactly like pre-lane traffic
  (default lane interactive, any `x-lane` honored). `batch`-class keys default to the
  harvest lane and are refused `x-lane: interactive` with a 403 — a bulk key cannot
  claim the protected class, by omission or by header.
- **Rate limits:** per-key `rate_limit` is a concurrency-slot override; the effective
  cap is **min(override, global lane cap)** — the global cap stays authoritative, an
  override can only narrow. The `X-RateLimit-*` trio reports the binding cap, with
  `Remaining` counting the tighter of the tenant and lane gauges.
- **Metering seam:** every admitted request logs one flat
  `[meter] admit id=<x-request-id> tenant=<t> lane=<l> model=<m>` line — the public-repo
  half; the private fork's metering layer joins these against the worker-truth usage
  lines by request id for per-tenant billing.

Gate: `tools/apikeys-gate.sh` (unit laws + live two-tenant isolation proof via
cache-hit behavior; receipts `research/apikeys-20260805/`).

## Session affinity — resuming a REWRITTEN conversation (lane/session-affinity, 2026-08-05)

Both reuse tiers above require the new prompt to EXTEND what is cached (token prefix, or
text prefix). Real agent clients do not extend — they REWRITE. The owner's client strips
`<think>` blocks out of prior assistant turns before re-sending them, so turn N's prompt is
not a prefix-extension of anything, both probes miss, the parked multi-GB session is
discarded, and every turn re-primes the whole growing conversation.

Affinity answers a different question: not "does this prompt extend that session's bytes?"
but "is this the SAME CONVERSATION?" — and then resumes it at a retained boundary.

**Two identity tiers (nomination only):**

- **Explicit** — the client names its conversation. Accepted from `session_id` or `user` in
  the request body of `/v1/completions` and `/v1/chat/completions`, or the `x-session-id`
  header. Body beats header (the body is the caller's own statement of identity; a header can
  be injected by an intermediary); `session_id` beats `user`. An explicit id on one side only
  never matches: a named conversation and an anonymous one are not the same conversation.
- **Implicit** — nothing named, so identity is STRUCTURAL: the conversation is split at its
  control tokens (the chat template's own role markers) and each segment contributes a hash of
  its first and last few tokens. A rewritten segment BODY does not perturb its hash, so the
  chain's leading run survives a think-strip; three shared segments are required before an
  implicit fingerprint may name a conversation (a bare system prompt is shared by every fresh
  conversation and must not cross-link them).

**Identity nominates, BYTES decide.** A nominated session is resumed only if the new prompt
reproduces its committed tokens EXACTLY up to the boundary its last turn checkpointed. A
fingerprint collision therefore costs one wasted comparison, never a wrong resume. If the
rewrite reached BELOW the boundary, affinity declines and the request re-primes in full —
correctness first. Declines are logged with their offsets (`history diverged at N of
checkpoint M`), because a silent decline is indistinguishable from a broken mechanism.

**The boundary.** Each turn checkpoints the state at its PROMPT END — before the first
generated token. That is the only boundary worth keeping: a history-rewriting client mutates
what the session GENERATED, never the prompt it was given. Full-attention KV is truncatable
by length, so the checkpoint copies only the GDN conv/ssm recurrent state; the draft scratch
needs no copy (the next turn's fill rewrites it).

**Scope.** Affinity is stored per (model, cache namespace), so it adds no cross-tenant reach
beyond what the reuse tiers already have: a `cache_salt` is an affinity boundary too.
Constrained (grammar) requests never resume. Resumed sessions respect the same evict-first +
right-size ladder as new ones, and are tested against the room the request actually needs, so
a right-sized session stays affinity-eligible.

`MEMRA_AFFINITY=0` turns the mechanism off (rollback seam / exactness A/B arm; the winner is
the default and needs no flag). Receipts, byte-identity gate, and TTFT curves:
`research/session-affinity-20260805/`.

## Multi-tenant QoS — the x-lane SLO gate (lane/qos-p95, 2026-08-02)

Requests may tag a service class via the `x-lane` header: `interactive` (protected;
also the default when the header is absent — naked traffic is byte-identical),
`judge` (prefill-shaped), or `harvest` (decode-shaped bulk). The gate is engine-side
admission control: interactive always admits (waits FIFO past `MEMRA_MAX_SESSIONS`,
never rejected); judge/harvest admit only while the measured interactive decode-step
p99 stays under their fraction of `MEMRA_SLO_P99_MS` (50ms default) and shed with an
immediate `429 + Retry-After` otherwise — dark work is never queued inside the engine.
Inside the tick, interactive decode rows batch first and dark-lane prefill runs after
decode within measured SLO headroom only. Per-lane counters + the engine-truth step
p50/p99 export at `GET /yield/metrics`.

Measured at fleet scale (8 replicas, Qwen3.5-9B-Q8_0 on rented H100s, c=96 harvest + c=4
interactive, 4 conditions interleaved, N=3 passes with full teardown/bring-up per cell,
`research/qos-p95-20260802/`): the lane-blind proxy FIFO alone inflates contended
interactive p95 to 7.15s (~4x alone); with lanes on and the proxy cap at 16 (so engine
admission owns the queue — the gate cannot fix a queue it never sees), p95 drops to
3.69s (~2x alone) at -11% bulk throughput vs the cap-16 ceiling. `MEMRA_SLO_P99_MS`
is the dial: 25ms makes contended interactive statistically equal to alone
(p50 1.637s / p95 2.158s) with bulk paying -67%. Lane knobs in [FLAGS.md §1](FLAGS.md).

**Attribution, required whenever the 7.15 → 3.69s figure is quoted:** raising the proxy
cap from 8 to 16 *by itself* moves p95 **7.15 → 4.335s** (that control cell is in the same
RESULTS.md); the lane gate accounts for **4.34 → 3.69s**. Roughly half the headline
improvement is the queue, not the engine gate — which is the point of the sentence above,
not a caveat to it. Quoting 7.15 → 3.69s as "what lanes do" is refutable from our own log.

## Streaming cadence + admission latency — the felt-TTFT arc (lanes sse-cadence 2026-08-05, admission-latency 2026-08-06)

Two fixes, one arc: solo first text went **0.41 s → 0.12 s** and contended first text
**1.60 s → 0.15 s** (27B NVFP4+MTP, K=3, local 5090, N=5 medians in one lock hold), and
neither number scales with `MEMRA_SPEC_BURST` anymore.

- **Round-cadence SSE** (lane/sse-cadence): spec-burst sessions used to emit ONE
  `Event::Token` per burst — at B128 that meant 2 chunks per response and 1.16 s to first
  text. The worker now flushes text at every spec-round commit through an `on_commit` seam
  in the engine's spec loop (same detokenize-tail + `utf8_delta` cursor, same
  EOS-text-never-streamed rule), so first text is ~0.12 s and inter-chunk gap p50 ~27 ms at
  ANY burst size for a solo stream (B32 fix-off was 0.41 s / 299 ms). Content is
  byte-identical either way — only chunk boundaries move. Throughput parity measured c=1
  and c=8. Rollback: `MEMRA_SSE_PER_BURST=1`. Receipts:
  `research/sse-cadence-20260805/VERDICT.md`.
- **Admission yield + cold-first ordering** (lane/admission-latency): a request arriving
  mid-burst used to wait the whole in-flight burst out (contended first text 0.54 s at B32 /
  1.60 s at B128, i.e. burst size set round-robin admission latency). Two pieces, one flag:
  a pending admit (`PENDING_ADMITS` gauge, polled by the round hook above) ends the
  in-flight burst at the next round boundary, and sessions that have emitted nothing yet
  burst before mid-generation peers. Contended first text is now **0.123 s (B32) / 0.152 s
  (B128)** — the solo class at any burst. Content byte-identical on/off, solo AND contended.
  The cost lives at c=8 saturation only: −3.4% agg tok/s for 3.8x better p50
  (newcomer-first vs lockstep-fair; p95 tail pays); c=1 parity. Rollback:
  `MEMRA_ADMIT_YIELD=0` (both pieces). Receipts: `research/admission-20260806/VERDICT.md`.

Burst default stays **B32** by the strict flip criterion, but the two old flip-blockers are
gone: B128 buys +8.4% (c=1) / +8.5% (c=8) and now trails B32's contended first text by one
29 ms round-cadence quantum instead of a 3x cliff — a live owner call, per the
`MEMRA_SPEC_BURST` row in FLAGS.md.

## Knobs

Serving flags (batch cap, device sampling, lean logits, prime batching, spec burst) are
cataloged in [FLAGS.md §7](FLAGS.md) under "Serving (memra-server)"; fleet topology knobs
(`GPUS`, `REPLICAS_PER_GPU`, `CAP`, ports, health cadence) are env-overridable at the top of
`tools/serve-fleet.sh`. The exactness contract holds under batching: the decode-batch gate
battery (gate1-3, gate3c lean-vs-full) runs inside `tools/validate-h100.sh`.

## First-token cross-config drift (batched prime) — stated honestly

Serving primes prompts BATCHED (`prime_cache`, prefill GEMMs) while the historical oracle
stream is tokenwise (`decode_step`, m=1). These are different numeric configs by design —
same law as forward-vs-decode and the decode-batch gate's config mode — so on near-tie
prompts the FIRST generated token of a request can differ from the tokenwise oracle
stream, and everything after it follows the new prefix. Measured on the six-model
2026-08-02 sweep (`research/prime-gate-coverage-20260802/`, 144 prompts): **10/144 first
tokens flip (~7%)**, every flip at a tokenwise top1-top2 margin <= 0.70, batched prime
bit-deterministic, no content leakage across chunk boundaries, and forward_last sides
with the batched prime in 8/10 flips — the tokenwise config is usually the outlier, so
this is config roulette on a near-tie, not a wrong path. On the gemma prefill lanes the
config can even move per PROCESS (cuBLASLt heuristic algo selection; one observed
instance in the 144-row double pass, bit-deterministic within a process). Dense Q8_0
models (9B judge, Ornith-9B — the fleet class) flipped 0/48. Consequences can be visible (the Qwen3.6-35B
pp512 probe greedy-emits `"\n"` + EOS at 2 tokens where the tokenwise stream writes 128):
within contract, but real. `MEMRA_PRIME_TOKENWISE=1` pins the oracle stream at prefill
cost; the run-gen `batched-prime` gate line + the `prime-gate` battery bound the class
(structured divergence fails hard, near-tie flips are reported).

### Chunked prefill is split-stable — since the grain-free fix (found 2026-08-05; mechanism corrected and FIXED same day)

The class below is **history**: since lane/chunkinv-flip, chunked prefill is bit-identical
across `MEMRA_PRIME_CHUNK` values by default (see "FIXED BY DEFAULT" further down). What
follows is the finding and root cause, kept because the mechanism correction is the
evidence for the fix. A sharper statement of the same class, found while building
serve-smoke check 10: **changing only the prefill chunk split changed greedy output.**
Arms were the same four recorded prompts with a per-turn `cache_salt` (so nothing
resumes — every request primes cold), `MEMRA_AFFINITY=0`, varying only `MEMRA_PRIME_CHUNK`:

| prompt tokens | 2048 vs 64 | 2048 vs 32 |
|---|---|---|
| 48 | identical | identical |
| 97 | identical | **differs @ char 45** |
| 149 | **differs @ char 172** | **differs @ char 52** |
| 195 | identical | identical |

No reuse required, and 149 tokens is far too short for a long-window explanation. Every
resume tier inherits this by construction: a resume primes `[rewind boundary .. end]` as
its own chunk sequence rather than one full prime.

**Mechanism — corrected 2026-08-05 by `lane/chunk-invariance`.** This section originally
said "a different split changes the reduction order in the prefill GEMMs." **That is
measurably wrong.** The prefill GEMM is m-INVARIANT: feeding the same activation rows at
m=32 and at m=33..80 leaves rows `[0,32)` BIT-IDENTICAL for both the quantized `wq` and the
`output` head, so growing a batch does not move an existing row's value. And the divergence
is not a distributed last-bit band — it is a **step at the first chunk boundary**: per-row
maxdiff is exactly `0.000e0` for every row before the boundary and O(1) (6.9) immediately
after it, with `first_div_pos` equal to the chunk size exactly in every arm.

The real cause is a numeric-**class** edge, in `full_attn_prime_fa_dispatch`
(`hybrid_forward.rs`), selected by `base_len == 0` — *"is this the first chunk?"*:

- chunk 0 → `fa_prefill` over this batch's **f32** K/V;
- every later chunk → `fa_prefill_view_ws` over the **q8_0/q5_1 quantized KV cache**.

So `MEMRA_PRIME_CHUNK` decides at which token position the prefill stops reading f32 K/V
and starts reading dequantized cache. Rows before that position are computed identically in
both configs (hence the bit-identity); rows after it carry q8_0/q5_1 quantization error, and
a near-tie argmax flips. Eliminated by measurement, not assumption: `MEMRA_PRIME_DEQW=0`
(the other quantized-cache FA kernel) diverges identically, and `MEMRA_GDN_CHUNKED=0`
(sequential GDN scan, no WY segmentation at all) still diverges — the GDN state carry is
**not** the cause.

**FIXED BY DEFAULT — grain-free (lane/chunkinv-flip, 2026-08-05).** The `base_len == 0`
f32 special case is gone: chunk 0 quantizes its K/V into the cache first and attends through
`fa_prefill_view_ws` exactly like every later chunk (quantize-then-attend). One numeric class
for every row means the chunk size cannot decide where a precision edge falls, so **chunked
prefill is byte-identical across `MEMRA_PRIME_CHUNK` values with no door and no grain knob**
(chunkinv gate, naked env, both pinned prompts EXACT at chunks 2048/64/32).
`MEMRA_PRIME_CHUNK` is again a pure memory/transient knob. Rollback seam:
`MEMRA_PRIME_F32CHUNK0=1` restores the legacy f32 first-chunk arithmetic (and is the gate
canary's injection). The interim `MEMRA_PRIME_INVARIANT`/`MEMRA_PRIME_GRAIN` pin-the-boundary
door was superseded by this fix and removed at v0.71 per the flags doctrine (the research
record keeps its history). History + root-cause receipts:
`research/chunk-invariance-20260805/VERDICT.md`; flip receipts:
`research/chunkinv-flip-20260805/`.

What this changes and what it does not:

1. Gates MAY now assert byte-equality between two prefills of the same prompt at different
   chunk boundaries — `tools/chunk-invariance-gate.sh` asserts exactly that as its default
   (`--expect-invariant`, no env). serve-smoke check 10's scoping note is retired with it.
2. The exactness CLASS of short (single-chunk) prompts changed at the flip: chunk 0 now
   reads quantized KV — the same arithmetic long prompts always had past the first boundary.
   Near-tie argmax flips vs the old f32-first-chunk output are the documented contract
   change (quantified teacher-forced in `research/chunkinv-flip-20260805/`), not a bug.

The behavior is **gated in both directions**: fast-gate ids `chunkinv` / `chunkinvc`
(routed from the `hybrid_forward.rs` map row): the default arm asserts byte-identity naked;
the canary arm injects `MEMRA_PRIME_F32CHUNK0=1` and must break, proving the gate detects
the mechanism. Reproducers + raw rows:
`research/session-affinity-20260805/chunk-order-probe.py` and `chunk-order.jsonl` (12 rows =
3 chunk sizes x 4 prompts, each with its text; under two minutes on the 9B), plus the
engine-level root-cause arm `concat-prime-probe chunkinv` and
`research/chunk-invariance-20260805/`.
