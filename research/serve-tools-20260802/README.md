# serve-tools: OpenAI tools surface on memra-server — gate receipts (2026-08-02)

Lane: `lane/serve-tools` (from `restructure/public-split`). Mission: the OpenRouter market
wedge is "cheapest endpoint WITH tools on an agentic model"
(`research/or-provider-20260802/REPORT.md` — tool calling named the single biggest gap;
memra-server previously rejected `tools` outright). This lane adds `tools` + `tool_choice` +
streaming `tool_calls` deltas + `role:"tool"` turns + `reasoning_effort`/`reasoning` to
`/v1/chat/completions` as **template + parsing only — zero kernel/engine changes**.

## What changed (code)

| file | change |
|---|---|
| `crates/memra-tokenizer/src/chat.rs` | `apply_chat_template_tools`: exact reproduction of the qwen3.5/3.6-class templates' tools branch (`<tools>` system header, `<function=…>`/`<parameter=…>` call rendering, `<tool_response>` turn grouping, `enable_thinking=false` think switch). Legacy `apply_chat_template_str` untouched; plain requests pinned byte-identical (`tools_renderer_matches_legacy_when_plain`). |
| `crates/memra-server/src/toolcall.rs` | NEW — streaming `<tool_call>` emission parser (state machine over text deltas, think-gated, schema-typed argument coercion, deterministic FNV ids). Malformed policy: unparseable blocks surface VERBATIM as content; unterminated blocks flush raw. 8 unit tests. |
| `crates/memra-server/src/main.rs` | request surface (`tools`, `tool_choice` auto/none, `reasoning_effort`, `reasoning`, assistant `tool_calls`, `role:"tool"`, content parts), python-dumps tool-JSON rendering (client key order, `preserve_order`), OpenAI `tool_calls` response shapes (stream deltas + message), `finish_reason:"tool_calls"`, usage `prompt_tokens`/`total_tokens`. |
| `crates/memra-server/src/worker.rs` | `Request` carries turns/tools/think; admit() routes plain requests through the EXACT legacy render path (isolation by construction); template-capability probe at load (`ModelCaps`); `Done` carries worker-truth `prompt_tokens`. |
| `docs/SERVING.md` | "OpenAI tools surface" section. |

Template ground truth: the committed dumps (`research/onboard-ornith-20260801/templates/`)
were verified **byte-identical to the deployed GGUFs' embedded `tokenizer.chat_template`**
(q35 = Qwen3.6-35B-A3B-UD-IQ4_XS, AgentWorld = Qwen-AgentWorld-35B-A3B-UD-IQ4_XS) before
implementation. One brief-vs-template correction: the templates' generation default is
think-ON (`enable_thinking` undefined → `<think>\n`); `enable_thinking=false` is the
no-think switch. The absent-param default therefore stays think-ON (byte-identity contract);
`reasoning_effort none|minimal|low` maps to the no-think switch, `medium|high` to default.

## Gate battery (RTX 5090, all GPU runs under `flock /tmp/gpu5090.lock`)

Driver: `run-gates.sh` (three flock holds, one server boot each; server killed by PID —
co-resident llama-server untouched). Deterministic greedy (temperature 0, seed 0), N=3
byte-identity asserted per leg (`roundtrip_gate.py`). Models: `q35` =
`/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`, `aw` =
`/data/ai-ml/hf-models/agentworld-35b-gguf/Qwen-AgentWorld-35B-A3B-UD-IQ4_XS.gguf`.

### Isolation contract (non-tools traffic unchanged)

| gate | verdict |
|---|---|
| baseline binary, q35: greedy c1-vs-c16 (n=16, 96 tok) | **PASS 16/16** (`greedy-hash.jsonl` `baseline-q35`) |
| new binary, q35: c1-vs-c16 | **PASS 16/16 x2** (`new-q35-c16`, two boots) |
| new binary c16 vs BASELINE-binary c1 refs (cross-binary) | **PASS 16/16 x2** (`new-q35-vs-baseline-refs`) |
| cross-binary c1 refs byte-diff (baseline refs file == new refs file) | **PASS, identical 16/16** (`cross-binary-refs-verdict.json`) |
| renderer differential (tools renderer on plain requests == legacy, unit) | **PASS** (`tools_renderer_matches_legacy_when_plain`) |

Structural guarantee on top of the gates: plain chat requests take the *legacy* render
call and never construct a parser (worker admit branch + `build_chat_request`).

### Tools battery (deterministic greedy, N=3 per leg, byte-stable across reps)

| gate | q35 (Qwen3.6-35B IQ4_XS, spec path) | AgentWorld (35B-A3B IQ4_XS, tokenwise/graph) |
|---|---|---|
| A-leg1: call emitted + parsed | **PASS** — `get_weather` `{"city":"Paris"}`, `finish_reason:"tool_calls"`, 177 completion tok | **PASS** — same call, 231 tok |
| A-leg2: role:"tool" result -> answer | **PASS** — "sunny … 21°C … 40%" | **PASS** — "sunny … 21°C … 40%" |
| A': `reasoning_effort:"low"` | **PASS** — no-think prompt, call parses, 26 completion tok (vs 177 default) | **PASS** — 26 tok (vs 231) |
| B: streaming OpenAI schema | **PASS** — 9 chunks (spec bursts), stream call == non-stream | **PASS** — 208 chunks (tokenwise), equal |
| C: malformed emission | **PASS** — block surfaced verbatim, 0 tool_calls, HTTP 200 | **PASS** |
| D: usage exactness | **PASS** — worker `prompt_tokens` == tok-check: plain 27 / tools 330 / no-think 332 (tools block = 303 tok) | **PASS** — identical counts (same template family) |
| E: bijection (same rendered prompt, parser off vs on) | **PASS** — stream-order byte-equal + parse-equal (654 bytes) | **PASS** (830 bytes) |

Gate-law note: the first q35 E row is **FAIL by a gate bug, not a server bug** — the
reconstruction concatenated content-then-blocks, but on the spec path content legally
continues after the block (the rendered eos text, below). `raw_len == recon_len` and
`parse_equal: true` in the row show the same bytes in different concat order. The law was
fixed to stream-order reconstruction and the full battery re-run (second set of rows,
15:41-15:42): PASS.

Observation (pre-existing, NOT this lane): on the SPEC serve path the eos token's text
(`<|im_end|>`) is rendered into the final content delta (`decode_bytes_special(.., true)`
over a burst that includes eos), so q35 chat content carries a trailing `<|im_end|>`;
the tokenwise/graph path (AgentWorld) checks eos before emitting and does not. Baseline
and new binary are byte-identical here (cross-binary refs) — surfacing/stripping it is a
separate serve-quality call, out of this lane's exactness scope.

## Evidence-discipline note: one contaminated row

`greedy-hash.jsonl` row `new-q35-c16` (15:00:54) is **FAIL with 5 quoted "timed out"
errors — contaminated, not evidence of a regression**: the Claude Code harness process
restarted mid-run, killing the flock holder (lock released; orphaned server kept decoding —
`server-new-q35.log` spec-acc lines continue past the kill) while another lane's GPU work
was free to start. Cause of the timeouts is therefore unattributable (repro needed — and
the clean re-run directly after passed). The row stays in the append-only JSONL; the
`new-q35-c16` verdict that counts is the post-restart re-run row.

## Metering interaction (for the darklane side)

`usage` on both chat shapes (stream final chunk + non-stream) now carries worker-truth
`prompt_tokens` (the tokenized RENDERED prompt — the tools block is billable prompt bytes;
measured delta on the leg-1 request in `gates-*.jsonl` row `D-usage`,
`tools_block_tokens`) and `total_tokens = prompt + completion`. Previously `total_tokens`
equaled `completion_tokens` (no prompt count on this branch). `lane/dl-metering` reworked
usage independently — reconcile at merge: worker-truth counts must stay the single source.
