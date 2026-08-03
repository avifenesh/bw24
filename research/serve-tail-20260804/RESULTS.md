# serve-tail lane — OR-listing serve tail items, 2026-08-04

Rig: local RTX 5090. Model: q9 = Qwen3.5-9B NVFP4 MTP GGUF (no draft attached for the
battery; the smoke run attaches the regime draft in its spec arm). GPU serialized via
`flock /tmp/gpu5090.lock`; two other lanes share the rig — all runs short bursts.
Battery: `run-battery.sh` (this dir); full log `battery-run.log`; server stderr
`server-rl.log` (items 1-2) / `server-drain.log` (item 3). Raw headers/bodies:
`rl-hdr-*.txt`, `shed-hdr-*.txt`, `drain-*.{txt,json}`, `v1-models.json`.

Closes the last three gap-scan/serve-compat items on the serve surface:
F11 (graceful drain), F12 (x-ratelimit headers), and the queued OR-schema `/v1/models`
(gap-scan REPORT §1.4).

## Item 1 — /v1/models OR-schema enrichment  (PASS, live + unit)

New route `GET /v1/models` (the legacy `/models` stays byte-identical). Per model:
`id`, `name`, `object`, `created` (worker-ready unix seconds — the honest timestamp we
have), `context_length` (model config, 262144 live on q9), `architecture {modality:
"text->text", tokenizer, instruct_type}` (spawn-time ModelCaps probe: tokenizer = the
GGUF pre-tokenizer family, "qwen35" live; instruct_type from template turn markers,
"chatml" live), a pricing stub (OR-convention "0" USD strings), and `top_provider
{context_length, max_completion_tokens: null}` — null because max_tokens is
context-bounded (gap-scan F2), a static cap would be invented. Unknown metadata is an
honest null, never a fabricated value (unit-pinned both directions). Live entry:
`v1-models.json`.

## Item 2 — X-RateLimit-Limit / -Remaining / -Reset  (PASS, live hammer)

Concurrency-slot semantics (the only budget this server actually enforces — no
request/min quota exists to report): Limit = the lane's admission cap (the same
MEMRA_MAX_SESSIONS / LanePolicy values the worker gate enforces), Remaining = free
slots at submission (per-lane atomic gauge at the HTTP layer; RAII guard rides the
response, SSE streams hold their slot until fully written), Reset = 0 while free,
else mean-service estimate (tokens/request x p50 step from /metrics) or
MEMRA_RL_RESET_S=2 fallback — honestly coarse, a hint not a promise.

Hammer receipts (single run each, warm server, N=8 and N=6 concurrent):

- interactive, 8 concurrent vs cap 4: trio on all 8 responses;
  remainings = [3,2,1,0,0,0,0,0] — Remaining hits 0 at cap; over-cap requests queue
  (never shed) and correctly report 0. Reset was 0 while free, 2 at saturation.
- harvest, 6 concurrent vs cap 2: 2 served + 4 shed; every shed = HTTP 429 +
  Retry-After + the full trio; Remaining hit 0 before the sheds.

## Item 3 — graceful drain on SIGTERM  (PASS, live)

SIGTERM flips a process drain flag: new completion requests 503 immediately with
Retry-After (=MEMRA_DRAIN_S) and the OpenAI error object, `/health` reports
`"draining"` (the LB not-ready signal), axum graceful shutdown waits on the in-flight
gauge up to MEMRA_DRAIN_S (default 30s), then exit 0.

Live sequence (battery-run.log): 1024-token stream started, SIGTERM sent after first
stream bytes (server log: "draining (1 in flight)"), then — in-flight stream ran to
completion with `data: [DONE]` received; concurrent new request got 503 +
Retry-After; `/health` returned `"draining"`; process exited 0 in ~11s (drain log:
"drain complete", inside the 30s deadline).

Note: the first battery run's drain probe raced a fast 256-token generation (spec
bursts finished inside the 2s pre-SIGTERM sleep — drain correctly reported "0 in
flight" and exited instantly; probe fixed to wait for first stream bytes + 1024
tokens). The failure was the probe's, not the drain's; kept in git history.

## Cross-checks

- `cargo test -p memra-server`: **43/43 PASS** (38 pre-lane + 5 new: /v1/models shape
  incl. honest-null law, rate-limit math + gauge lifecycle, handler-level headers via
  a fake worker, drain 503/health-flip/admit-again).
- `tools/serve-smoke.sh`: **the EXACT 4 pre-existing fails** receipted in
  research/constrained-20260803/RESULTS.md (chat non-stream / greedy determinism /
  concurrency / spec-vs-plain — the think-tail-at-small-max_tokens box condition),
  log `serve-smoke.log`. No new failures; the passing set (models list, chat stream,
  /v1/completions, long generation) unchanged.
- No worker/model-loading changes beyond 3 additive ModelCaps metadata fields
  (context_length/tokenizer/instruct_type) + the probe that fills them — deliberately
  clear of lane/serve-st's model-plan work in the same file.

## Flags

`MEMRA_DRAIN_S` (30) and `MEMRA_RL_RESET_S` (2) documented in docs/FLAGS.md §7
(serving). Both are runtime/machine-config class, not experiment doors.
