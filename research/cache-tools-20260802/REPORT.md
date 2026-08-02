# Caching & Tools — competitive, economic, and security research for the darklanes listing

Date: 2026-08-02. Lane: `lane/cache-tools-research` (from `restructure/public-split`). No GPU used.
Every external fact is cited inline with source + date (all web sources fetched 2026-08-02 unless a
publish date is given). Our own numbers cite in-repo receipts.

Owner framing: "cache and tools are a big deal, worth a real research." This sits on four prior
lanes and asks what the caching + tools surfaces are worth competitively, what they earn on
input-heavy SKUs, what must change on security before a marketplace listing, and what to build next.

**Prior lanes this builds on (all 2026-08-02, in-repo receipts):**
- `research/prompt-cache-20260802/` — cross-request prefix cache SHIPPED. TTFT -76% (3.005->0.710s
  p50 hit waves), hit-wave throughput +95% (165.6->323.3 tok/s), 99.1% cached fraction in hit waves
  / 49.5% per pass, bit-exact (16/16 exactness gate), OpenAI `cached_tokens` billing on every
  response shape, LRU under `MEMRA_PREFIX_CACHE_MB` budget, spec-tier bypass. N=3 interleaved.
- `research/serve-tools-20260802/` — OpenAI tools surface SHIPPED (template + parsing only, zero
  kernel change). Streaming `tool_calls` deltas, template-native render, malformed-verbatim policy,
  `finish_reason:"tool_calls"`, worker-truth usage. GAPS: `tool_choice:"required"`/named -> 400 (no
  constrained decoding), no `json_mode`/`structured_outputs`, `parallel_tool_calls` not a param.
- `research/or-provider-20260802/` — OR mechanics: inverse-square price-weighted routing, `:floor`
  capture, Auto Exacto governs ALL tool-calling traffic, hy3 endpoints price cache-read at 25% of input.
- `research/sku-repick-20260802/` — Step-3.7-Flash = flagship SKU: 93.5:1 input-heavy in:out,
  $89.3K/day pool over 3 held-price endpoints, prefill/cache-dominant economics. Wedge = cheapest
  endpoint WITH tools.

---

## 0. Executive summary

The wedge the sku-repick lane found — "cheapest Step/hy3-class endpoint that also carries tools" —
is not one feature. It is a **three-part interlock on exactly the traffic these SKUs are made of.**
Tool-loop requests are simultaneously (a) tools-gated (they need the `tools` feature and Auto Exacto
standing to route at all on an agentic model) and (b) the most cacheable traffic on the network
(87-91% of input tokens are exact prefix re-sends, section 3). We already ship the tools surface and
a bit-exact cross-request prefix cache with honest `cached_tokens` billing. The combination is a
margin multiplier precisely where the money is on an input-heavy SKU.

**Security verdict (the one thing that must change before listing).** Our prefix cache keys on **raw
token prefixes, per-model, with no tenant scope** (`crates/memra-server/src/worker.rs`:
`PrefixCache { entries: HashMap<String /*model*/, Vec<PrefixEntry>> }`, matched on `toks: Vec<u32>`).
That is the exact configuration the 2025-2026 literature shows is exploitable: PROMPTPEEK (NDSS 2025)
reconstructs prompts at 95-99% from a shared radix/prefix cache, and CacheProbe (arXiv 2605.30613,
May 2026) audited OpenRouter itself and found **all three tested default-mode providers leaked across
accounts**, using the `cached_tokens` field as a noise-free hit oracle (4.8% of cross-account OpenAI
requests). Our `cached_tokens` billing field is the same oracle. Before listing we **must** add a
per-tenant salt/namespace to the cache key (the vLLM `cache_salt` design, shipped v0.9.0 2025-05-15).
This is cheap (a key change) and non-negotiable.

**Intersection math (section 3).** On a saturated, prefill-bound replica — which Step at 93.5:1 is —
cache-read priced at 25% of input **multiplies billable input revenue ~1.58x at a 70% hit rate and
~2.42x at 85%**, on the *same* prefill compute, because cached tokens still bill (at 25%) while
costing ~zero to serve. End-to-end that is the Step **$3.20 -> $4.40/hr saturated, 3y net +$12.4K ->
+$21.9K (+76%)** jump the sku-tco model already shows for 0->70% cache. The lever exists only on
cache-heavy, prefill-bound, tools-gated traffic — i.e. it is worth the most on exactly the SKU we picked.

**Roadmap top pick (section 5): constrained (grammar) decoding.** It is the single investment that
turns four gaps into one build — `tool_choice:"required"`, named forcing, `json_mode`,
`structured_outputs` — and it is the capability Auto Exacto's tool-success/benchmark tiers reward.
The correctness delta is the biggest published number in the space: **61-72% valid tool/JSON output
without constrained decoding -> 96-100% with** (SqueezeBits, 2025-09-16). Our logit-mask kernel seam
already exists (`penalize_logits`, `add_row_inplace`, Gumbel masked-argmax in
`crates/memra-engine/src/lib.rs`); the missing piece is the grammar compiler (adopt XGrammar/
llguidance, don't build) + per-step mask + spec-loop integration. Co-requisite: per-tenant cache
isolation (security gate). Best pure-revenue follow-on: cache host/disk tier + TTL policy.

---

## 1. AXIS 1 - CACHING

### 1.1 How the field prices and implements caching (2026)

| Provider | Mechanism | Min length | TTL | Write cost | Read ratio | Hit disclosure fields | Source (fetched 2026-08-02) |
|---|---|---|---|---|---|---|---|
| **Anthropic** | Explicit `cache_control:{type:"ephemeral"}` breakpoints (<=4) + NEW 2026 automatic mode (single top-level field auto-places breakpoint, walks forward); lookback <=20 blocks | 512 tok (Opus/Fable/Mythos 5); 1,024 (Sonnet 5/older); 2,048-4,096 (legacy) | 5 min default, 1 h opt-in (`ttl:"1h"`); no longer | 1.25x (5m), 2x (1h) | **0.1x** input | `cache_read_input_tokens`, `cache_creation_input_tokens`, `cache_creation.{ephemeral_5m,1h}_input_tokens` | platform.claude.com/docs/.../prompt-caching |
| **OpenAI** | Automatic >=1024 tok; `prompt_cache_key` to shard/route | 1,024 tok | pre-5.6: 5-10 min idle, max 1h; GPT-5.6+: >=30 min; `prompt_cache_retention:"24h"` (GPU-local, no storage fee, since GPT-5.1 Nov 2025) | Free pre-5.6; **1.25x on GPT-5.6+** | **0.1x** on GPT-5/5.1/5.5/5.6; 0.25x legacy (4.1/o3); 0.5x (o1) | `prompt_tokens_details.cached_tokens`; newer: `cache_write_tokens` | developers.openai.com/api/docs/guides/prompt-caching |
| **DeepSeek** | Automatic disk-based context caching, on by default, best-effort | request/prefix boundaries (old 64-tok block figure dropped) | "few hours to few days", cleared when unused | none | **~2% (outlier)**: v4-flash hit $0.0028 vs miss $0.14 (2.0%); v4-pro 0.83% | `prompt_cache_hit_tokens`, `prompt_cache_miss_tokens` | api-docs.deepseek.com/guides/kv_cache |
| **Google Gemini** | Implicit on by default (2.5+); explicit `CachedContent` now only on Legacy generateContent API (Interactions API = implicit only, a 2026 regression) | implicit 2,048 (2.5), 4,096 (3.x) | explicit default 1h; implicit undisclosed | explicit: input price + storage **$4.50/M-tok/hr (Pro), $1.00 (Flash)**; implicit none | **0.1x** input (was 75% off at 2025 launch, now 90%) | `usage_metadata.cachedContentTokenCount` (legacy); `usage.total_cached_tokens` (Interactions) | ai.google.dev/gemini-api/docs/caching |
| **xAI (Grok)** | Automatic; `x-grok-conv-id` header for sticky-hit routing | not disclosed | not disclosed (memory-pressure eviction) | none | **0.15-0.20x** (grok-4.5 15%, grok-4.3 16%, build-0.1 20%) | `prompt_tokens_details.cached_tokens` (chat), `input_tokens_details.cached_tokens` (Responses) | docs.x.ai/developers/advanced-api-usage/prompt-caching |

**Pattern:** the model *owners* price cache-read at ~10% of input (Anthropic/OpenAI/Gemini) or lower
(DeepSeek 2%); write premiums (1.25x-2x) are back in fashion in 2026 (OpenAI added one on GPT-5.6+;
Anthropic and Gemini-explicit already had them). Anthropic now offers an automatic mode too — the
explicit-vs-automatic distinction is blurring toward "automatic by default, explicit control for
power users."

### 1.2 SOTA serving architectures

- **vLLM Automatic Prefix Caching (v1)** — block-hash chain (SHA-256 default since v0.11, 2025-10-02;
  hash includes LoRA id, MM hashes, and **cache salt**), LRU, on by default, <1% overhead at 0% hit
  (vllm.ai 2025-01-27). Connector ecosystem in-tree: LMCache, NIXL (RDMA P/D), Mooncake, Offloading.
  **CPU offload** (OffloadingConnector, v0.11.0 2025-10-02): CPU-hit TTFT **2-22x**, up to **9x
  throughput** at high hit (vllm.ai 2026-01-08). v0.26 (2026-07-27) adds object-store tier +
  hybrid-model partial hits. (docs.vllm.ai/en/latest/design/prefix_caching)
- **SGLang RadixAttention** — radix tree of KV across all requests, cache-aware scheduling; paper
  (arXiv 2312.07104, NeurIPS 2024) up to **6.4x throughput**. **HiCache** (2025-09-10) L1 GPU -> L2
  host DRAM -> L3 storage (Mooncake/3FS/NIXL/file): up to **6x throughput, 80% TTFT cut**; Ant on
  R1-671B **84% avg TTFT cut on hits**; Novita+3FS hit 40%->80%.
- **LMCache** — external KV layer (GPU/DRAM/disk/remote), CacheGen (SIGCOMM'24: KV **3.5-4.3x**
  smaller, delay **3.2-3.7x** lower), CacheBlend (EuroSys'25 Best Paper: TTFT **2.2-3.3x**,
  throughput 2.8-5x, non-prefix reuse). Creators founded Tensormesh ($20M, 2026-05-27); still OSS.
- **Mooncake (Moonshot/Kimi)** — KVCache-centric P/D disaggregation + distributed multi-tier KV pool
  over RDMA (arXiv 2407.00079, FAST'25 Best Paper): **+525% throughput** long-context under SLO,
  **75% more requests** on real Kimi trace. Now a backend under SGLang/vLLM/LMCache/Dynamo/NIXL.
- **NVIDIA Dynamo + llm-d** — Dynamo KV-aware Smart Router + KVBM (HBM->DRAM->SSD->object): up to
  **30x** requests (R1-671B on GB200); Dynamo 1.0 (2026-03-16) up to **4x lower TTFT**; Baseten
  reports **89% hit rate across 4 replicas**, OpenRouter prod p95 -48%, RPS +61%. **llm-d** IGW with
  *precise* (not approximate) prefix-cache scoring: **57x faster P90 TTFT vs approximate**, 170x vs
  cache-blind (llm-d.ai 2025-09-24). TensorRT-LLM native block reuse (128-tok blocks) default-on +
  pinned-host offload.
- **Tier economics** — KV bytes/token (DeepSeek arXiv 2505.09343): MLA V3 **70.3 KB/tok**,
  Qwen2.5-72B GQA **327.7 KB/tok**, Llama-3.1-405B **516 KB/tok** (128K ctx ~= 40 GB/request). DRAM
  and NVMe tiers raise addressable cache 10-100x GPU HBM at ~0 quality cost.

**What "top-tier" means, mid-2026 (the ladder we are on):**
- **Table stakes** (every serious stack): in-VRAM block/radix prefix cache, LRU, on by default,
  near-zero overhead + **per-tenant isolation via salting** (vLLM `cache_salt`, 2025). *We have the
  VRAM+LRU rung; we do NOT yet have the salting rung — see 1.4/section 4.*
- **Expected of production stacks**: host-DRAM 2nd tier with async DMA (2-22x TTFT on hits) +
  cache-aware routing across replicas (approximate radix mirroring baseline; precise per-pod
  tracking the 2026 refinement). *We are single-tier VRAM, single-replica cache; this is the
  section-5 item-3/item-6 gap.*
- **Frontier**: disk/NVMe + object-storage 3rd tier, cross-node KV pooling over RDMA, KVCache-centric
  P/D disaggregation, KV compression / non-prefix reuse (CacheGen/CacheBlend). Not needed at 2-4
  replicas.
- **Convergence note**: by mid-2026 the majors interoperate through common plumbing (NIXL transport,
  Mooncake store, LMCache portable layer) — the moat has moved from "do you have a tier" to routing
  precision + hit-rate engineering at fleet scale. At our size the win is a correct, isolated, host-
  tiered VRAM cache with honest billing, not a novel architecture.

### 1.3 Agentic-traffic cacheability (the math)

See section 3 for the full derivation and the earnings model; the caching-specific result: agent tool
loops re-send the whole transcript every turn, so cache-read fraction of input rises toward 1 with
turn count. For a K-turn loop (system+tools base B, per-turn growth d), our worked values are **86.7%
(K=3), 89.4% (K=10), up to 91.3%** (thin-increment). This is corroborated by production disclosures:
**DeepSeek fleet-wide 56.3%** (342B of 608B input tokens hit disk cache in 24h, open-infra-index
2025-03-01) across a *mixed* workload; **OpenRouter per-endpoint hit rates 84-90%** for agentic models
(DeepSeek V4-Pro 87.9%, StepFun 86.1%, Claude Sonnet 4.6 89.9%; dirac.run aggregation 2026-05-23);
real user bills **84-98%** (ofox.ai audits 2026-05); **Manus**: "KV-cache hit rate is the single most
important metric for a production-stage AI agent," ~100:1 in:out, 10x cached-vs-uncached cost ratio
(manus.im 2025-07-18). Our own 99.1% hit-wave receipt is the pure-shared-header ceiling; real agent
traffic sits in the 70-91% band once per-turn growth is included.

### 1.4 Multi-tenant security (the attack surface)

**Attacks (paper - venue/date - mechanism - result):**

| Attack | Venue / date | Mechanism | Result |
|---|---|---|---|
| Auditing Prompt Caching in LM APIs (Gu et al., Stanford) | ICML 2025 / arXiv 2502.07776, Feb 2025 | Statistical timing test: cached prefix returns faster; if global, cross-user hit detectable | 17 providers audited; **7 had global cross-user sharing** (Azure, DeepInfra, Fireworks, Lepton, OpenAI-embeddings, Perplexity, Replicate); >=5 remediated after disclosure. Anthropic/OpenAI-chat safe (per-org) |
| PROMPTPEEK: Prompt Leakage via KV-Cache Sharing (Wu et al., SUSTech+ByteDance) | **NDSS 2025** | Per-token probing walks SGLang radix tree via hit/miss timing | Full prompt reconstruction **up to 99%** (template known), **95%** (no prior knowledge) |
| InputSnatch (Zheng et al.) | arXiv 2411.18191, Nov 2024 | Timing side channel on prefix + semantic caches (vLLM 0.6.2) | 62% disease-name extraction; **87.1%** cache-hit prefix-length inference |
| **CacheProbe: Auditing Prompt Cache Isolation in Gateway APIs** (Fahey) | SAGAI'26 / arXiv 2605.30613, **May 2026** | Audits **OpenRouter**: timing + `cached_tokens` metadata oracle across accounts | **All 3 default-mode providers leaked cross-account** (OpenAI, Groq, Fireworks); **4.8% of cross-account OpenAI requests reported cached_tokens above threshold**; **BYOK restores isolation**. Root cause = shared org credentials |
| Early Bird (Song et al.) | arXiv 2409.20002, Sep 2024 | Prefix-cache timing + continuous-batching latency channel | Cached-prompt + concurrent-request-composition leakage |

**Provider practice (docs, dated):** OpenAI — "Prompt caches are not shared between organizations."
Anthropic — "Caches are isolated between organizations ... never share caches, even if identical,"
plus per-workspace. DeepSeek — "Each user's cache is isolated and logically invisible to others"
(2024-08-02; provider claim, not independently audited). Gemini — per-project resources; no explicit
cross-customer guarantee on the caching page (UNVERIFIED beyond per-project model). OpenRouter —
**no per-end-user isolation in default mode** (pools under shared org credentials); BYOK restores it.

**OSS mitigation:** vLLM **`cache_salt`** (PR #17045, merged 2025-04-30, v0.9.0) — optional per-request
salt injected into the first block's hash, propagates via parent-hash chaining; only same-salt requests
share blocks; docs cite the timing side-channel rationale explicitly. SGLang RadixAttention has **no
per-tenant isolation by default** (this is what PROMPTPEEK attacks). Tinfoil (enclave design,
2026-07-14) derives the cache namespace from authenticated tenant identity inside the enclave.

**Our design, audited from code (`crates/memra-server/src/worker.rs`):**
- `PrefixCache { entries: HashMap<String, Vec<PrefixEntry>> }` — the `String` key is the **model id
  only**; entries match on `toks: Vec<u32>` (raw token prefix). **No tenant/org/api-key dimension.**
- `lookup()` returns the longest entry exactly prefixing the incoming prompt (floor
  `PREFIX_CACHE_MIN_TOKENS = 64`); a hit **deep-copies another request's primed KV/recurrent state**
  into the new session.
- Auth is a single shared `MEMRA_API_KEY` bearer (`main.rs:337`) — the whole server is one trust
  domain. On OpenRouter, OpenRouter is that one API customer, but the *prompts* it forwards come from
  many distinct end-users. Two end-users sharing a system prompt (common: everyone on the same agent
  framework) cross-hit each other's cached prefixes, and our returned `cached_tokens` is a per-request
  hit oracle (non-zero proves a prior request shared that exact prefix).

**This is exactly the CacheProbe/PROMPTPEEK configuration.** It is safe today only because the server
runs as a single trust domain behind one key; it becomes a live cross-tenant leak the moment the
cache serves prompts from more than one end-user. See section 4 for the fix.

---

## 2. AXIS 2 - TOOLS

### 2.1 What top-tier tool serving means in 2026

**Constrained/grammar decoding (the core capability):**

| Engine | Approach | Overhead | Adoption 2026 |
|---|---|---|---|
| **XGrammar** | Byte-level pushdown automaton + adaptive token-mask cache (context-independent tokens precomputed); CPU mask-gen overlapped with GPU | <40us/tok (vendor); up to 14x e2e JSON serving | **Default grammar backend in vLLM, SGLang, TensorRT-LLM, MLC** |
| **XGrammar-2** (2026) | Adds dynamic per-request tool sets + tool-name-conditional argument schemas (`structural_tag`) | ~12.7us/tok (Llama tool fmt) vs llguidance 250us; ~10ms compile; <6% e2e overhead | Integrated in SGLang + vLLM (arXiv 2601.04426) |
| **llguidance** | Rust lexer/parser, Lark CFG + JSON Schema | ~50us/tok avg | 2nd backend in vLLM + TRT-LLM; **optional in llama.cpp** (`-DLLAMA_LLGUIDANCE=ON`) |
| **llama.cpp GBNF** | Native char-level grammar, JSON-schema->GBNF built in | no published per-tok fig | still the llama.cpp default |
| **OpenAI structured outputs** | schema compiled to CFG on first request then cached; `strict:true` | first-compile latency | the API contract everyone clones |

**Mechanics:** `tool_choice:"required"` = a grammar forcing a tool-call array (vLLM >= 0.8.3); named =
compile that tool's parameter schema; `json_mode` = "any valid JSON" grammar; `structured_outputs` =
schema-compiled grammar. Notably, `tool_choice:"auto"` is historically NOT grammar-enforced (relies on
model + a `--tool-call-parser` regex) — the open gap vLLM RFC #39848 wants to close. **A grammar that
enforces tool-call framing + schema-constrained arguments even under `auto` is a real differentiation
surface no major engine ships by default.**

**Exactness / determinism (our contract's lens):** constrained decoding = logit masking before
sampling (disallowed logits -> -inf). Under **greedy**, the output changes *only* at steps where the
unconstrained argmax token is masked; if the model would have produced valid output anyway,
masked-greedy is byte-identical — deterministic given (model, grammar, tokenizer, sampler). Two
caveats we must gate: (1) **spec decode + structured outputs compose** in vLLM V1 (per-draft-position
bitmask, PR #14702) but vLLM documents spec decode does not guarantee stable logprobs — our
spec==greedy exactness gate must be re-run under masks; (2) **jump-forward decoding** (SGLang
compressed FSM) injects grammar segments without forward passes and requires retokenization, which
**can change the conditional distribution** (dottxt Coalescence) — if we adopt it, it is a separate
numeric config needing its own argmax baseline. The safe first cut is plain per-step masking on the
greedy path (no jump-forward), which preserves bit-exactness.

**Correctness delta (the headline number):** **61-72% valid without constrained decoding -> 96-100%
with** (SqueezeBits 2025-09-16, Qwen3-8B/32B); OpenAI 100% schema adherence with structured outputs
vs <40% prompting. JSONSchemaBench (arXiv 2501.10868): easy schemas all >86%, GitHub-Hard splits
Guidance 41% / llama.cpp 39% / XGrammar 28% / Outlines 3%. The "constraints hurt reasoning" claim
("Let Me Speak Freely," arXiv 2408.02442) was rebutted (dottxt "Say What You Mean": structured >=
unstructured with corrected prompts).

**Parallel tool calls:** OpenAI `parallel_tool_calls` (default true) -> multiple `tool_calls` entries,
streaming deltas carry per-call `index`. Native emitters: DeepSeek (up to 128 tools, strict mode),
Qwen3 (Hermes `<tool_call>` blocks), Kimi K2 (dedicated parser). **StepFun parallel behavior:
UNVERIFIED.** Serving burden: split multiple blocks, assign ids, correct per-call `index`.

**Benchmarks:** BFCL v3/v4 (deterministic AST match; simple/multiple/parallel/multi-turn/irrelevance),
tau2-bench (user-simulator + policy adherence; frontier scores still low: airline ~38%),
ComplexFuncBench, K2 Vendor Verifier. Schema-adherence varies **materially by serving stack on
identical weights**: K2VV (2025-11-15) 100% (official/Fireworks/Groq) down to 84.6% (Together), 76%
(raw vLLM), 73.1% (SGLang) — proof the parser/stack, not the weights, decides tool quality.

**Provider failure receipts (our differentiation surface):** OpenRouter's own data — GLM-5 ~8%
tool-call error, gpt-oss-120b 5.6% pre-routing. Streaming bugs across hosts: `index` starting at 1 not
0 (LiteLLM #32759), SSE fragmentation executing tools with empty `{}` args (LangChain #35514), delta
chunks missing required fields breaking strict clients (pydantic-ai #3658, koog #489). Quant receipts:
Roo-Code #11325 asks to filter FP4/Int4 OR providers over malformed output. Most variance attributed
to serving stack/parser, not quant.

**Auto Exacto mechanics (OpenRouter):** announced 2026-03-12, **on by default for every request with
`tools`**, re-evaluated ~5 min. Three signals: (1) throughput; (2) tool-call success = 1 - error rate,
validated with `@cfworker/json-schema` (Draft 7) in three buckets (InvalidJson / UnknownName /
SchemaMismatch), denominator = requests with `finish_reason:"tool_calls"`; (3) benchmark accuracy =
OR's own harness running **GPQA Diamond** (temp 0.5, 10 epochs) + **Tau2-Bench Airline** (temp 0).
Deranks: throughput >1.5sigma below median, tool-success >2sigma below median, benchmark <baseline-2sigma
(baseline = median of first ~21 days, 32-day rolling window). Data minimums: 100 general req/30min,
200 tool-call req/2h, >=4 providers before thresholds compute. Claimed results: GLM-5 error -88%
(~8%->~1%). Opt-out via `:floor`/price sort. Capability flags: `/v1/models` `supported_parameters`
enumerates `tools`/`tool_choice`/`response_format`/`structured_outputs`; users pin with
`provider:{require_parameters:true}` (else silent fallback e.g. json_schema -> json_object).

### 2.2 Where we stand (audited from code)

| Capability | Status in `crates/memra-server` | Note |
|---|---|---|
| `tools` + `<tools>` template render | SHIPPED | byte-identical to embedded chat_template (serve-tools gate) |
| streaming `tool_calls` deltas w/ `index` | SHIPPED | `main.rs:656` increments `call_index` per emitted block |
| `tool_choice: auto`/`none` | SHIPPED | `main.rs:288` |
| `tool_choice: required` / named | **400** | needs constrained decoding (`main.rs:294`) |
| `json_mode` / `response_format` | **absent** | needs constrained decoding |
| `structured_outputs` | **absent** | needs schema-compiled grammar |
| `parallel_tool_calls` request flag | **absent** | but wire format already tolerates N calls (below) |
| malformed policy | SHIPPED | unparseable block surfaced verbatim, HTTP 200, 0 tool_calls |

**Honest downgrade on "parallel tool calls":** the serve-tools README lists it as a gap, but the
parser (`toolcall.rs`) loops Scan->InCall->Scan and pushes every `<tool_call>` block as a separate
`Piece::Call`, and the response layer (`main.rs:656-677, 772`) already assigns OpenAI `index` fields
and collects a `tool_calls` array in both streaming and blocking shapes. **The wire/serialization
support for parallel calls exists today.** What is genuinely missing is only (a) accepting/echoing
the `parallel_tool_calls` param and (b) a gate for a model emitting multiple blocks in one turn (the
qwen3.5/3.6 family tends to emit one call per turn and loop). This lowers the roadmap cost of
"parallel tool calls" materially.

### 2.3 The constrained-decoding seam we already have

`crates/memra-engine/src/lib.rs` exposes GPU logit-manipulation primitives: `penalize_logits`
(in-place logit edits over a history set), `add_row_inplace` (per-row bias add), `scatter_trim_logits`
(set -inf then fill), and a Gumbel-max **masked** perturb-then-argmax draw. Greedy = argmax over the
logit buffer is the bit-exact reference (`crates/memra-sampling`). A grammar mask is exactly one more
in-place logit edit (disallowed ids -> -inf) applied before argmax — **the kernel seam is present.**
Missing: the grammar compiler + per-step token-mask computation (adopt XGrammar/llguidance, don't
build) and its interaction with the spec draft+verify loop (the real exactness work, section 5).

---

## 3. AXIS 3 - THE INTERSECTION (why the combination is a moat multiplier)

Tool-loop traffic is the one class that is **both** cache-heavy **and** tools-gated. The three edges
compound on it.

**(a) Cacheability of agent loops.** A tool loop re-sends the whole transcript each turn: turn i sends
`B + (i-1)d` tokens (B = system+tools header, d = per-turn growth), and turn i's prompt has turn
(i-1)'s entire prompt as an exact prefix. Cache-read fraction of total input:

```
frac = [ (K-1)*B + d*(K-1)(K-2)/2  +  B(cross-session header hit) ] / [ K*B + d*K(K-1)/2 ]
```

| loop | B | d | K | total input | cacheable | fraction |
|---|---|---|---|---|---|---|
| short agent | 2000 | 500 | 3 | 7,500 | 6,500 | **86.7%** |
| typical agent | 2000 | 500 | 5 | 15,000 | 13,000 | **86.7%** |
| long agent | 2000 | 500 | 10 | 42,500 | 38,000 | **89.4%** |
| tool-heavy | 3000 | 800 | 8 | 46,400 | 40,800 | **87.9%** |
| thin-increment | 1500 | 300 | 12 | 37,800 | 34,500 | **91.3%** |

(scratch `intersection_math.py`.) Our cross-request cache is what captures the **cross-session header
hit** — turn 1 of a *new* session hits the shared system+tools prefix, which per-session continuation
pools cannot (the prompt-cache lane's audit: same system prompt = 0 prefill skipped as-is). Field
corroboration in 1.3: DeepSeek 56.3% mixed-fleet, OR agentic endpoints 84-90%, real bills 84-98%.

**(b) The revenue multiplier on a prefill-bound replica.** The naive read — "cache-read bills 25%, so
caching *lowers* per-token revenue" — is the wrong frame for a saturated endpoint. A saturated Step
replica is **prefill-bound** (93.5:1 in:out), so its binding constraint is prefill compute, not price.
With hit fraction h, the same prefill-compute budget serves `1/(1-h)` as many billable input tokens;
the extra ones bill at 25% of input on ~zero marginal compute. Billable-input-revenue multiplier vs
no-cache, cache-read = 25% of input (Step's own rate is 20% -> slightly lower multipliers, shown):

| hit fraction h | x at 25% cache-read | x at 20% (Step) |
|---|---|---|
| 50% | 1.25 | 1.20 |
| 70% | **1.58** | 1.47 |
| 85% | **2.42** | 2.13 |
| 90% | 3.25 | 2.80 |
| 99.1% (our hit-wave ceiling) | 28.5 | 23.0 |

End-to-end (folding in the fixed output revenue and the prefill/decode split), this is exactly the
`sku-tco.py` result: **Step-3.7-Flash $3.20/hr -> $4.40/hr saturated (+37%), 3-year net +$12.4K ->
+$21.9K (+76%)** going 0%->70% cache (`research/sku-repick-20260802/`, re-run 2026-08-02). Billable
input throughput in that model rises 4.4B -> 12.0B tok/day on the same box.

**Honest caveat:** this lever is worth ~nothing on a decode-bound, low-in:out SKU (gpt-oss-120b at 6:1
stays $0.53->$0.53/hr in the same run). It pays off **only** on cache-heavy, prefill-bound, input-heavy
traffic — precisely tool-loop traffic on agentic SKUs like Step. **The moat and the SKU pick are the
same bet.**

**(c) The tools gate is what lets us serve that traffic at all.** On an agentic model, Auto Exacto
governs *every* tool-calling request and deprioritizes endpoints missing the tools feature or
benchmark data (2.1). Without tools, none of the cacheable agent traffic routes to us; with tools + a
real cache + exact `cached_tokens` billing, we are eligible for the tier AND serve it at 1.5-2.4x
billable input.

**Realistic routing share.** Per OR's rules (`research/or-provider-20260802/`): a ~5-10% undercut of
the effective floor that *also* carries tools captures the entire `:floor`/price-sort segment
immediately (no review cycle — the provider monitor reads price/capacity from `/v1/models`) plus a
default-routing weight ~1.2-1.6x vs held-price incumbents (inverse-square). The binding limit is
capacity, not demand: at 30% utilization Step needs only ~0.04% of the page's current pool
(sku-repick section 7). The cache multiplier means each captured request is worth 1.5-2.4x more on
input than a no-cache competitor serving the same bytes — **so we can undercut on list price while
earning more per compute-hour.** That is the moat multiplier in one sentence.

---

## 4. AXIS 4 - SECURITY VERDICT (what must change before listing)

**BLOCKING (must ship before any marketplace listing):**

1. **PC-ISO - per-tenant cache keying/salting.** Today the key is model-id only. Add an isolation
   token to the `PrefixCache` key so entries never cross a trust boundary. Adopt the vLLM `cache_salt`
   design (per-request salt folded into the prefix hash; only same-salt requests share). The honest
   marketplace boundary is the *end-user*, not the API customer: on OpenRouter many end-users arrive
   through one OR org credential, so "per-org" isolation collapses to global sharing (CacheProbe: all
   default-mode providers leaked cross-account). Two viable postures:
   - **(a) Isolate by default, opt-in sharing:** key on a per-request namespace (OpenAI's
     `prompt_cache_key`/`user` convention, or an OR-passed end-user/session id — OR added `session_id`
     / `x-session-id` sticky routing exactly for this). No id -> no cross-request sharing.
   - **(b) Provider-blessed safe-prefix sharing:** allow cross-request sharing ONLY for a provider-
     owned safe-prefix set (system + tools headers the provider itself defines), keep all user-turn
     content per-session/per-namespace. This preserves most of the section-3 multiplier (the header is
     the bulk of the shared bytes) while leaking nothing user-specific.
   Cost: small (key change + plumb a namespace through `admit()`); it is a correctness/compliance gate.

2. **Suppress or scope the `cached_tokens` hit oracle.** With per-tenant keying the field only reveals
   a tenant's own history (acceptable). Without PC-ISO it is a proven cross-tenant oracle (CacheProbe
   4.8%). Keep billing transparency but ensure the count can never reflect *another* tenant's prefix.
   Do NOT ship the current model-only-keyed cache to a multi-end-user channel with `cached_tokens` on.

**SHOULD-DO (hardening, not blocking a PC-ISO'd v1):**

3. Timing channel: a hit is dramatically faster TTFT (our -76% receipt is the signal an attacker
   measures). Per-tenant keying closes the *cross-tenant* timing channel; no constant-time work is
   needed once the key is isolated (matching the field — Tinfoil et al. also decline timing padding).
4. Document the isolation boundary precisely (org/workspace/end-user), as Anthropic/OpenAI do;
   ambiguity is itself a gap (>=5 providers had to retrofit after Gu et al.). Keep cached state in
   memory, no raw prompt at rest, short TTL, and offer BYOK/no-cache as the isolation-restoring
   fallback for privacy-sensitive tenants.

**Verdict:** the cache is safe today only as a single-trust-domain server. It is a live cross-tenant
leak the instant it serves more than one end-user — which is what an OpenRouter listing is. PC-ISO is
the one non-negotiable change before listing; it is cheap.

---

## 5. ROADMAP VERDICT - ranked next investments

| # | Investment | Unlocks | Honest cost | Impact on the Step listing revenue |
|---|---|---|---|---|
| 1 | **Constrained (grammar) decoding** | `tool_choice:required` + named + `json_mode` + `structured_outputs` + Auto Exacto tool-success/benchmark tier | Adopt XGrammar/XGrammar-2 or llguidance (don't build); per-step mask on the existing logit seam; **spec-loop + exactness gate is the real work** (re-run spec==greedy under masks; keep jump-forward off for bit-exactness). Est. 2-4 weeks | **Highest.** Turns 4 gaps into 1 build; the capability Auto Exacto rewards on the exact traffic class; correctness 61-72%->96-100% is our exactness-discipline differentiator vs high-malformed-rate hosts |
| 2 | **PC-ISO - per-tenant cache isolation** | safe marketplace listing | small (key + namespace plumb; adopt cache_salt design) | **Gate.** Without it we cannot list responsibly. Near-zero perf cost |
| 3 | **Cache host/disk tier + TTL policy** | deeper/longer residency -> higher sustained hit fraction across the fleet | host-pinned DRAM 2nd tier under existing LRU (vLLM OffloadingConnector pattern: 2-22x TTFT on host hits) + TTL knob | **Best pure-revenue follow-on.** Directly deepens the section-3 multiplier (more of the 87-91% cacheable actually resident) |
| 4 | **Parallel tool calls (finish)** | `parallel_tool_calls` param + multi-block gate | **small** - wire format already done (2.2); param + gate only | Modest; table-stakes polish for agent frameworks |
| 5 | **Explicit `cache_control` API** | Anthropic-style breakpoints/TTL control | request-surface + policy | Low near-term; OR passes automatic caching, explicit control is a nicety on our SKU |
| 6 | **Cache-aware fleet routing** | prefix-affinity across replicas -> fleet-level hit fraction | router change (consistent-hash/approx-radix on prefix; precise tracking is the 2026 refinement, 57x vs approximate on llm-d) | Matters at >2-4 replicas; defer until the fleet grows |

**Top pick: constrained decoding (item 1), with PC-ISO (item 2) as the mandatory co-requisite before
listing.** Constrained decoding moves the Step listing's revenue on multiple axes at once (Auto Exacto
routing eligibility + `json_mode`/`structured_outputs` as `/v1/models` features users filter on) and
it converts our exactness discipline into a public differentiator (K2VV-style vendor verifiers are the
scoreboard; identical weights swing 73-100% on serving stack alone). PC-ISO ships in the same window
because it is a listing gate. The cache host tier (item 3) is the best pure-revenue follow-on because
it compounds the section-3 multiplier we already proved.

---

## Source index

**In-repo receipts:** `research/prompt-cache-20260802/RESULTS.md`, `research/serve-tools-20260802/README.md`,
`research/or-provider-20260802/REPORT.md` (+ raw endpoint JSON), `research/sku-repick-20260802/REPORT.md`
(+ `sku-tco.py`, re-run 2026-08-02). Code audited: `crates/memra-server/src/worker.rs` (PrefixCache,
model-only key), `.../main.rs` (tool_choice, streaming tool_calls, single-key auth), `.../toolcall.rs`
(parser, multi-block loop), `crates/memra-engine/src/lib.rs` (logit-mask seam), `crates/memra-sampling/src/lib.rs`.
Scratch math: `intersection_math.py` (cacheability + saturated-replica revenue multiplier).

**Web (all fetched 2026-08-02; publish dates inline):**
- Caching pricing: platform.claude.com/docs/en/build-with-claude/prompt-caching;
  developers.openai.com/api/docs/guides/prompt-caching (+/pricing); api-docs.deepseek.com/guides/kv_cache
  (+/quick_start/pricing, /news/news0802 2024-08-02); ai.google.dev/gemini-api/docs/caching (+/docs/pricing);
  docs.x.ai/developers/advanced-api-usage/prompt-caching; openrouter.ai/docs/features/prompt-caching;
  OR `/api/v1/models/{slug}/endpoints`.
- SOTA serving: docs.vllm.ai/en/latest/design/prefix_caching; vllm.ai/blog (2025-01-27, 2026-01-08,
  2026-05-18); github.com/vllm-project/vllm PR#17045 (cache_salt); lmsys.org/blog (2024-01-17,
  2024-12-04, 2025-09-10 HiCache); arXiv 2312.07104; github.com/LMCache/LMCache; arXiv 2310.07240
  (CacheGen), 2405.16444 (CacheBlend); arXiv 2407.00079 (Mooncake, FAST'25); github.com/ai-dynamo/dynamo;
  developer.nvidia.com/blog/nvidia-dynamo-1-production-ready (2026-03-16); baseten.co blog (2026-03-16);
  llm-d.ai/blog/kvcache-wins-you-can-see (2025-09-24); nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html;
  arXiv 2505.09343 (KV bytes/tok); github.com/deepseek-ai/open-infra-index (2025-03-01, 56.3%).
- Security: arXiv 2502.07776 (Gu et al., ICML 2025); ndss-symposium.org (PROMPTPEEK, NDSS 2025);
  arXiv 2411.18191 (InputSnatch); arXiv 2605.30613 (CacheProbe, May 2026); arXiv 2409.20002 (Early Bird);
  arXiv 2508.08438 (selective KV sharing); docs.vllm.ai design/prefix_caching (cache_salt);
  tinfoil.sh/blog/2026-07-14-secure-prompt-caching.
- Tools: blog.mlc.ai/2024/11/22 + arXiv 2411.15100 (XGrammar); arXiv 2601.04426 (XGrammar-2);
  github.com/ggml-org/llama.cpp docs/llguidance.md + grammars/README.md; docs.vllm.ai features/structured_outputs
  + tool_calling; vLLM RFC #39848, PR #14702; blog.squeezebits.com/guided-decoding-performance-vllm-sglang
  (2025-09-16); arXiv 2501.10868 (JSONSchemaBench); blog.dottxt.ai (say-what-you-mean, coalescence);
  gorilla.cs.berkeley.edu (BFCL); github.com/sierra-research/tau2-bench + arXiv 2506.07982;
  github.com/MoonshotAI/K2-Vendor-Verifier; openrouter.ai/blog/announcements/auto-exacto (2026-03-12) +
  /docs/guides/routing/auto-exacto + /docs/guides/community/for-providers; manus.im blog (2025-07-18).

**UNVERIFIED / flagged:** PROMPTPEEK exact %s (NDSS PDF did not text-parse; from abstract + secondary);
Gemini cross-customer isolation (no explicit doc guarantee); DeepSeek isolation claim (provider's own,
un-audited); StepFun parallel-tool-call behavior; no controlled quant-vs-tool-calling study exists; no
Anthropic/OpenAI/Gemini fleet hit-rate published; Dynamo "3x routing" and Mooncake "115K req/day" are
chart/secondary-only; DeepSeek current internal block size.
