# lane/step-sku-listing — Step-3.7-Flash listing prerequisites (Task #53, non-head-to-head)

Predecessors (read first): `research/step37-p2-20260806/PROGRESS.md` (onboarding, PP-2 boot,
exactness battery, drafter wiring), `research/step35-chunkfix-20260807/PROGRESS.md`
(chunk-dependence fix + gates). This lane: the four remaining listing prerequisites, plus the
owner's 2026-08-07 scope expansion (thinking control for every supported model).

Box: 2x RTX PRO 6000 Blackwell Server 96GB, PP-2, `18.195.123.14`; workspace synced to
`~/tokparity-memra` (built from this branch). Every GPU window under `flock /tmp/memra-gpu.lock`.
Local 5090 runs under `systemd-run` CPUQuota; boxes verified `0 MiB` at window exit throughout.

---

## Item 1 — tokenizer byte-check: 113/113 PASS, and the one mismatch was UPSTREAM's bug

Gate: new `tok-parity` bin (`crates/memra-tokenizer/src/bin/tok_parity.rs`) — memra's
GGUF-built tokenizer vs the HF reference (`tokenizers` over the sha-pinned `tokenizer.json`,
`raw/hf-ref-sha256.txt`; chat_template byte-identical to the phase-1 dump, f428623f). Corpus
(`build-tok-corpus.py`, 113 cases, hex transport): digit runs of every length mod 3, the exact
CJK/kana range bounds, RTL/Indic/combining marks, emoji ZWJ/keycap/skin-tone/flags,
NBSP/ZWSP/CRLF/ideographic-space runs, code/JSON/regex/URLs, special-token literals (incl.
partial/case-miss), and 10 full chat-template renders (jinja2 under the HF env — composition
with the committed render goldens end-to-ends the serve path). BOTH encode modes per case.

**First run: 112/113** (`raw/tok-parity-20260807T0640Z.log`). The miss — `" symbols ~ ^ | \ "`
splitting `" ", "~"` where HF has one `Ġ~` pre-token (id 6883 vs `223,96`) — is a real
**upstream llama.cpp defect**: `k_ucat_map`'s sub-128 SYMBOL expansion (``"$+<=>^`|"``,
`unicode.cpp:1244`, verified on master) omits `0x7E` `~` (U+007E, Sm) — the single
printable-ASCII codepoint where that map disagrees with real Unicode P/S (enumerated
0x21..0x7E). The HF tokenizer the model was TRAINED with does match `~` under `\p{S}`, so
memra now deliberately includes it (`c_is_symbol`, documented divergence; cross-engine
reference + committed output regenerated, exactly one corpus line moved; the per-mechanism
qwen35-divergence test re-pinned on ZWSP). Rerun: **113/113 BOTH modes, PASS**
(`raw/tok-parity-PASS-20260807T0655Z.log`). Commit `980a42ec`.

## Item 2 — reasoning_effort serve surface (then owner-expanded to ALL models — see below)

step35's headline control is a STRING in the system turn (`Reasoning: low|medium|high\n\n`).
Wired end-to-end: `apply_chat_template_tools` gains `reasoning_effort: Option<&str>` (only the
step35/hy3 arms consume it), `worker::Request::reasoning_effort` + ReplayPlan (park/re-admit
replays the identical render), `ModelCaps::effort_levels` spawn probe (keyed on the jinja
input test `reasoning_effort is defined` — true: step35+hy3, false: qwen/gemma4, verified
against all four templates), `parse_think` returns `(ThinkMode, Option<level>)`, and
`build_chat_request` gates the level on caps so non-consuming models' prompts are
byte-identical **by construction**. Docs: `docs/SERVING.md`. Commits `0676d33e`, `ca6edb8d`.

Serve-smoke on the box (`raw/effort-smoke-20260807T072741Z.log`, PP-2, spec OFF per #87,
drafter attached): caps line `effort_levels=true`; absent = prompt_tokens 19 (no `Reasoning:`
line — the template default); low/medium/high/none/OpenRouter-object = 29 (the system turn
gains the line; none clamps to low); `"extreme"` = 400 with the OpenAI error object.

## Owner scope expansion (2026-08-07): thinking for EVERY supported model

All supported models are thinking models; one surface, per-arch native mapping. Inventory from
the REAL shipped templates (goldens: `render-thinking-goldens.py` → `raw/thinking-goldens.txt`;
gemma4 template dumped from the local QAT GGUF header sha 36e3a42e, hy3 from the pinned
tencent/Hy3 snapshot sha 7fc351fe — both committed under `templates/`):

| class | native control | default |
|---|---|---|
| qwen ChatML (Qwen3.5/3.6, Ornith 9B/35B, AgentWorld, KAT — all 5 dumps identical markers) | `enable_thinking` | **ON** (open `<think>\n`) |
| gemma4 family | `enable_thinking | default(false)`; ON = `<|think|>` token atop the FIRST system turn (created if absent) + open generation turn | **OFF** (closed thought channel) |
| hy3 | its own `reasoning_effort:` — accepted set exactly `no_think|low|high` (jinja raises otherwise); low/high open `<think:opensource>` | **no_think** |
| step35 | `Reasoning: {level}` string; `<think>` tail unconditional | no `Reasoning:` line |

Mapping (`parse_think`, full table in `docs/SERVING.md`): `low|medium|high` = thinking ON at
that budget (the reasoning-model convention — low IS a reasoning mode; the old `low→NoThink`
was wrong and is retired), `none|minimal` = OFF, absent = the model's own default (no silent
behavior change — pinned by tests that Default == the legacy render byte-for-byte on every
arm), OpenRouter `{enabled:false}` = OFF / `{enabled:true}` = ON. `ThinkMode` gains `Think`.
hy3 medium clamps to low; step35 none clamps to low (no off level). Golden-pinned unit tests
per arm; memra-tokenizer 26 + memra-server 98 pass.

**think-smoke** (`raw/think-smoke-20260807T094506Z.log`, local 5090, qwen 9B + gemma4 12B on
one server): qwen absent/high/low → prompt_tokens=17 with populated `reasoning`; none /
`{enabled:false}` → 19, empty reasoning, instant `content='ok'`. gemma4 absent/none → 20;
low/high → 23 (+3 = the created `<|think|>` system turn) and the model visibly opens a
thought channel. step35 arm = effort-smoke above; hy3 has no local servable artifact — its
arm is golden-pinned in tests.

## Item 3 — drafter run-spec K=1..8 + acceptance-delta assertion

`raw/specgate-20260807T073545Z.log` (one window, 51s, cards 0 MiB at exit): self-consistency
**PASS at all 8 K** (token-identical to generate), acceptance **digit-for-digit equal** to the
pinned baseline at every K — K=1 14/18 = 77.8%, K=2..8 flat 15 accepted (44.1% → 11.0%).
`spec-gates.sh` adds the mechanized delta gate (±5pp vs baseline, red on parse failure —
the f8f4-flip lesson: self-consistency stays green under acceptance regressions).
**K-policy: K=1 is the served depth** (slot-0-only acceptance is structural — single-head
drafter reused past its trained +1 position; throughput monotone decreasing in K).
Commit `9b802f4d`.

## Item 4 — capacity/perf receipts ($/day shape; research-only, NOT the public board)

`raw/capacity-20260807T075551Z.log` + points/ttft JSONL (one window 07:55–08:14Z, thermal
sampled per cell — warm steady-state, never throttled). Trunk+drafter, PP-2, spec OFF (#87).

| cell | result |
|---|---|
| pp4096 prefill (ppprime, N=5 med) | **90.9 tok/s** (45.06s for 4k, spread 0.12%) |
| decode agg tok/s, c=1/2/4/8 (N=3 med each, MEMRA_SERVE_BATCH=0) | **34.13 / 34.10 / 34.28 / 34.15** — FLAT in c |
| tok/day sustained | **~2.95M output tok/day/box** at any c |
| p50 latency by c | 3.75s / 7.52s / 14.93s / 29.98s (128 tok/req — queueing, not throughput) |
| TTFT short-turn (228-tok prompt, stream, N=8) | **p50 2.183s, p95 2.187s** |

Aggregate flat in c: round-robin serializes decode; concurrency buys queueing only. That IS
the $/day shape today. Commit `2d2eb676`.

### The bug item 4's probe found (and this lane fixed): step35 B>1 over PP-2 was GARBAGE

Default batched serve at c>1 "worked" (n_ok, HTTP 200) — and `b2-geometry-ab.sh` showed the
text was wrong: `decode_step_batch_ppn` had **no step35 guard** (the unsplit body's B>1
refusal sits AFTER the ppn door), so over PP-2 — this SKU's only placement — a B>1 tick
walked the generic Full arm (global n_head=96, 128-dim rope everywhere, no SWA window, no
head-wise gate). Receipt `raw/b2ab-pre-20260807T091553Z.log`: c=1 clean, one c=4 row
`'::::\n\n::::::::...'`. Fixed fail-closed in the ppn body outside the B=1 eager walk +
`chunk_cap_for` pins step35 to **B=1 chunks** (not overridable upward — wider = wrong
logits). Post-fix (`raw/b2ab-post-20260807T094618Z.log`, rebuilt box binary, `decode chunk
cap 1` in the log): **all four c=4 responses byte-identical to the c=1 greedy reference.**
Commits `ca6edb8d` (fix), `a0ba3e36` (post receipt).

## Pre-existing gaps receipted, NOT fixed here (follow-up lane material)

1. **gemma4 has no serving decode path on the default scheduler** — any gemma4 chat panics
   the worker (`decode_step_batch v1 covers the hybrid non-gemma4 trunk only`,
   decode_batch.rs:553; the B=1 fast path also excludes gemma4), respawn re-panics, process
   FATALs. `MEMRA_SERVE_BATCH=0` serves it. Receipt `raw/think-smoke-20260807T093918Z.log`.
   Same failure class as the step35 B>1 hole; louder symptom; predates this lane.
2. **gemma4 thought/content separation unwired** — `<|channel>thought` text lands in
   `content` (the reasoning splitter keys on qwen `</think>` only) and `<turn|>` turn-end
   tokens leak into content. Receipt in `raw/think-smoke-20260807T094506Z.log`.
3. step35 `[prime-batch]` refusal (no batched prime core) — single primes serve; already a
   loud log line, recorded in the capacity receipts.

## Ledger

| item | state |
|---|---|
| 1. tokenizer byte-check vs HF reference | **PASS 113/113 both modes**; upstream `~` \p{S} omission found + fixed on memra's side |
| 2. reasoning_effort serve surface (step35) | **DONE** — wired, gated on caps, serve-smoke receipt, documented |
| 2+. thinking control for ALL supported models (owner 2026-08-07) | **DONE** — per-arch native mapping, golden-pinned, smoke on qwen+gemma4+step35; hy3 golden-pinned (no local artifact) |
| 3. run-spec K=1..8 + acceptance delta | **PASS 8/8**, acceptance == baseline digit-for-digit; K=1 = served depth |
| 4. capacity receipts | **DONE** — pp4096 90.9 tok/s; decode ~34.1 agg flat c=1..8 = ~2.95M tok/day/box; TTFT p50 2.18s |
| bonus: step35 B>1-over-PP2 garbage | **FIXED** fail-closed + B=1 chunk pin; pre/post receipts |
