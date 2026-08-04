# Repo evidence inventory — the receipts the website is allowed to cite (2026-08-05)

Every marketing claim on the darklanes site must trace to one of these. Paths are relative
to the memra repo root (bw24-unified, branch restructure/public-split as of 623ce27e).
Rule inherited from the repo's evidence discipline: **a claim whose raw runs exist nowhere
in the repo is not evidence** — the site never states a number the repo cannot back.

## 1. First-SKU serving numbers (Qwen 27B class on RTX PRO 6000 Blackwell, 96GB)

Source: `research/pro6000-prod-20260804/` (anchor/, serve/, levers/, q8rp/ raw logs +
JSONL; commit 623ce27e). Gates ALL GREEN on the 5th distinct GB202 die.

- Plain decode tg128: NVFP4+MTP arm 86.8 tok/s, Q8_0 arm 52.6 tok/s.
- Spec decode (MTP): **nv K=3 186.7 tok/s bare**, **170.6 tok/s through the serve surface**
  at c=1 (serve tax −12.6%). Q8 K=4 143.5 (reproduces the vast desktop 137.4 ladder
  digit-for-digit: acceptance 77.8/75.0/71.4).
- Batched serving: saturates at c=8, **421 tok/s aggregate** (nv arm).
- TTFT: **cold 0.182 s / warm 3 ms** (warm = prefix-cache hit; the protocol trap — unsalted
  TTFT hits the prefix cache — is documented in the commit, so the site must always state
  cold vs warm separately).
- Prefill pp512: 4118 (nv) / 4591 (q8) tok/s.
- Q8RP 96GB lever: +57% at c16/32 (486 vs 310 agg tok/s, p50 6.61→4.21 s), 63.7 GB resident.
- 5090 reference boards: README PERF-SAMPLES (2026-08-02): Qwen3.6-35B-A3B plain decode
  1.13x llama.cpp; 9B MTP spec 2.30/1.74/1.59x by prompt class. Full boards
  `docs/PERFORMANCE.md`, raw log `research/tune-data/rig5090.jsonl`.

## 2. Exactness discipline (the core differentiator)

- README.md (public repo): "Exactness is the contract: speculative, graph-replay, and
  batched serving output is gated token-identical to plain decode — speed never changes
  what the model says."
- Three standing gates (CLAUDE.md, CONTRIBUTING.md, docs/TESTING.md): `kernel-check`
  (every kernel vs CPU reference, ALL GREEN), `run-gen` argmax gate (printed MATCH before
  any generation; "a MISMATCH voids every number after it"), `run-spec` K=1..8
  self-consistency.
- Serve-surface isolation contract (docs/SERVING.md §"The isolation contract"): greedy
  output is byte-identical alone vs inside a full batch; gated by replaying prompts at
  c=1 and c=16 and byte-comparing every stream. The m-dependent MoE router defect this
  gate caught (expert selection changed with co-arrivals) is documented in the same
  section — a real bug the gate found, not a hypothetical.
- Prefix-cache exactness: cached hit is bit-identical to the run that computed the prefix,
  gated 16/16 + 16/16 (`research/prompt-cache-20260802/gate-exact.jsonl`).
- Constrained decoding exactness: device-mask greedy byte-identical to host oracle;
  spec-constrained byte-identical to plain-constrained; draft-mask ON byte-identical to
  OFF across 7 cells (`research/constrained-full-20260803/`, `research/draft-mask-20260804/`).
- Honest limits stated in-repo (the site should link, not hide): first-token cross-config
  drift on near-tie prompts under batched prime, ~7% of first tokens on a 144-prompt sweep,
  bounded and documented (docs/SERVING.md §"First-token cross-config drift").

## 3. The lanes QoS story (the brand namesake)

- Mechanism: `crates/memra-lanes/src/lib.rs` — three lanes (interactive / judge / harvest),
  shed at ADMISSION never inside the engine ("the engine's own queue is where the tail
  dies"), interactive never preempted, per-lane prefill budgets per tick. `x-lane` request
  header; naked traffic = interactive and byte-identical.
- Measured: `research/qos-p95-20260802/` — 8 replicas, c=96 harvest + c=4 interactive:
  lane-blind FIFO inflates interactive p95 to 7.15 s (~4x alone); lanes on → 3.69 s at
  −11% bulk; SLO dial at 25 ms makes contended interactive statistically equal to alone
  (p95 2.158 s) with bulk paying −67%. `MEMRA_SLO_P99_MS` is the knob (FLAGS.md §1).
- Endurance receipt: `research/fleet-endurance-20260803/SUMMARY.txt` — **140 min sustained
  load, 8x H100, 464,870 requests, 0 errors, 0 sheds**, throughput drift +0.045%, p95
  drift −0.4 ms, RSS plateau (max +3.0 MiB/replica), greedy determinism hash identical on
  all 8 replicas pre- and post-soak (56b8502cfb8de57a).
- Fleet chaos receipt: `research/fleet-v060-20260801/SUMMARY.md` — SIGKILL a replica
  mid-load: breaker DOWN same second, restart +2 s, backend UP +9 s, 8/768 requests lost
  (exactly the victim's in-flight cap), greedy hash identical on all 6 replicas 18/18.

## 4. Serve-surface features (product claims)

All in docs/SERVING.md with receipts:

- OpenAI-compatible `/v1/chat/completions` + `/v1/completions`, validated against the
  official `openai` Python SDK (`research/serve-compat-20260802/`): envelope (id/created/
  system_fingerprint/x-request-id), streaming with SSE keep-alives, reasoning separation
  (OpenRouter dialect), honest 400s on params it cannot honor (never silent ignoring of
  semantic params).
- Tools/function calling: `tools`, `tool_choice` auto/none, streaming tool_calls deltas,
  malformed-block policy = surface verbatim, never dropped (`research/serve-tools-20260802/`).
- REAL constrained decoding: `response_format` json_object/json_schema via llguidance,
  on-device masking at 99.4% of unconstrained speed; draft-side masking makes proposals
  legal by construction (`research/constrained-20260803/`, `-full-20260803/`,
  `research/draft-mask-20260804/`).
- Prompt caching: cross-request prefix cache, LRU, per-tenant isolation via `cache_salt`
  (vLLM-compatible design; the CacheProbe/PROMPTPEEK mitigation), gates in
  `research/pc-iso-20260802/`. Usage carries `prompt_tokens_details.cached_tokens`.
- Honest usage metering: worker-truth `usage` on every shape, disconnect abort billed to
  the abort point, `/metrics` cumulative splits. QoS lanes were extracted from the
  dl-metering lane (FLAGS.md §1).
- Gateway-ready surface: OR-schema `/v1/models` (context_length, architecture, pricing
  stub, honest nulls), truthful rate-limit headers riding the SSE stream, graceful drain
  SIGTERM→drain→exit 0 (`research/serve-tail-20260804/`).
- Sampling honesty (dogfood F4, 2026-08-04): omitted temperature ⇒ 1.0, omitted seed ⇒
  fresh entropy; explicit temp 0 / seed 0 honored exactly. The two bugs this fixed were
  found by the founder's own agent running on the server (commits 6f51d4a1, c716954b).

## 5. Business/market receipts (pricing section inputs)

- `research/or-provider-20260802/REPORT.md`: OpenRouter provider onboarding (application,
  backlog, proprietary-model priority), technical requirements (streaming usage, provider
  /v1/models schema, capacity_tpm), no published take rate (demand-side 5.5%), uptime
  accounting rules, hy3 endpoint price table (floor $0.129/$0.44 per Mtok in/out; cache
  read priced at exactly 25% of input across providers), revenue realism: saturated
  replica ≈ $2–4/hr gross at floor prices — "treat the first listing as distribution and
  public perf receipts, not revenue."
- `research/8bit-decision-20260803/DECISION.md`: 8-bit serving-format verdict — Q8_0 GGUF
  serves now, FP8-ST is the tuning track with a ≥1.1x promotion gate.
- `docs/qwen38-bringup-runbook.md`: the day-one Qwen3.8-27B bring-up plan (release watch,
  two legs: FP8-ST exact-arm prod direction + GGUF NVFP4+MTP spec leg), deployment bar
  ≥1.1x e2e vs llama.cpp before "published as supported."
- Memory (owner-confirmed 2026-08-02, darklanes-operating-model): serving success bar =
  cost coverage + public reliability stats, NOT market-share wins; "everything need to be
  honest, nothing is a dream that we try to make true."

## 6. HF Inference Providers channel (distribution)

huggingface.co/docs/inference-providers/register-as-a-provider (fetched 2026-08-05):
partner registration = task API compat (OpenAI-compatible LLM APIs "may skip most" of the
schema work) → huggingface.js PR → Model Mapping API (requires Team/Enterprise Hub plan)
→ billing endpoint (per-request cost in nano-USD, polled every minute, 30-min billing
window, idempotent, `Inference-Id` header on every response incl. streams) → Python client
PR → provider docs page. Automated validation every 6 h: TTFT < 5 s streaming, tool-calling
and structured-output behavioral tests — all surfaces memra already gates. Default
provider ordering on model pages = 7-day routed request volume. A separate lane owns the
mechanics; the site spec only needs the channel to exist in the distribution plan.
