# lane/integrate-cache — merge resolution notes (2026-08-02)

Merge: `lane/prompt-cache` (af72c3db) into `lane/integrate-cache` (from
`restructure/public-split` @ 22587a47, which carries lane/serve-tools). Merge base
1bc62d49. 14 conflict hunks across 3 files, all in the serve layer; zero engine code
touched by either side. Merge commit: f66fa478.

RESOLUTION LAW (the brief): BOTH features fully survive — merged usage carries
worker-truth `prompt_tokens` (rendered tools block included) AND
`prompt_tokens_details.cached_tokens`; a tools request is cacheable like any other
(the cache keys on rendered token ids); ONE prompt-count source of truth, no
double-count; spec-tier cache bypass and template-caps 400s both kept.

## Cluster 1 — `worker.rs` Event::Done + finish() (2 hunks)

Both lanes extended the same `Done` event: serve-tools added `prompt_tokens`
(tokenized rendered prompt), prompt-cache added `n_prompt`/`n_cached` (fed-or-resumed
+ cached split). **Resolution: unified on `n_prompt`/`n_cached`.** They are the same
number by construction on every non-resume path: the cache lane computes
`n_prompt = prompt.len()` from the SAME `prompt` vec the tools lane's renderer
produces in `admit()` (tools block already tokenized into it — the cache accounting
is downstream of the tools rendering, so the tools count is inherited, not
re-derived). On the spec TEXT-resume path `n_prompt = spec_resumed + suffix_len`
(actually-fed; can differ from a fresh whole-prompt tokenization at a BPE boundary) —
kept as the cache lane defined it, since it is the truer worker count (see Behavior
decisions). The tools lane's now-redundant `Session.prompt_tokens` field was REMOVED
(both sides' struct fields had auto-merged side by side — keeping both would be two
books for one number). Doc comment on `Done` states the unified contract.

## Cluster 2 — `main.rs` response shapes (8 hunks)

SSE Done arm, blocking Done arm, chat + text_completion usage blocks (stream and
non-stream), native done payload. **Resolution: tools lane's control flow + cache
lane's usage schema.**

- `finish_reason`: the tools lane's `finish` (parser-aware — `"tool_calls"` when the
  parser saw calls, parser flush pieces emitted before the final chunk) survives in
  all four OpenAI shapes; the cache side had kept base `stop_reason_to_finish`.
- `usage`: every shape now calls the cache lane's `usage_json(n_prompt, n_tokens,
  n_cached, elapsed_s)` — `prompt_tokens` + `completion_tokens` + `total_tokens =
  prompt + completion` (the tools lane's total law) + `prompt_tokens_details.
  cached_tokens` (the cache lane's split). The tools lane's inline usage JSON (no
  cached field) was dropped in favor of the helper — same fields plus the split.
- Native (non-OpenAI) done payload/CompletionResp: cache side's `prompt_tokens` +
  `cached_tokens` fields kept (superset of the tools side's `prompt_tokens`).

## Cluster 3 — `main.rs` tests (4 hunks)

Both lanes extended the same test module. All tests from both sides survive;
`Event::Done` constructions unified to `n_prompt`/`n_cached`.
`chat_response_has_openai_message_shape` keeps the cache side's richer usage
assertions (42/1/43 + cached 30). `blocking_tools_response_carries_tool_calls_and_
finish_reason` (tools side) GAINS the intersection assertion: a tool-call response's
usage carries the same worker-truth prompt/cached split as any other shape
(40/2/42 + cached 0). `cargo test -p memra-server`: 20/20 pass;
`-p memra-tokenizer` (renderer differential incl. `tools_renderer_matches_legacy_
when_plain`): 3/3 pass.

## Cluster 4 — `docs/SERVING.md` (1 hunk)

Both lanes appended a section at the same anchor. Both kept, tools surface first,
prompt caching second; the tools section's usage sentence now points at the caching
split and states the tools-are-cacheable law.

Auto-merged without conflict (reviewed, no action): `Cargo.toml`/`Cargo.lock` (cache
side's serde additions), `docs/FLAGS.md` (`MEMRA_PREFIX_CACHE_MB` row), the whole
PrefixCache module + admit() probe (cache side), the tools renderer/parser plumbing +
ModelCaps probe (tools side), `/metrics` cached counters (cache side).

## Behavior decisions for the owner

1. **`usage.prompt_tokens` on a spec TEXT-prefix resume reports the actually-fed
   count** (`spec_resumed + re-tokenized remainder`), which can differ by a token or
   two from a fresh tokenization of the whole rendered prompt at the resume's BPE
   boundary. This is the cache lane's semantics, kept as the single source of truth.
   The tools lane's D-usage gate (prompt_tokens == tok-check) ran on fresh sessions,
   where the two definitions coincide — re-verified GREEN on the merged binary.
2. **The tools lane's redundant `Session.prompt_tokens` was deleted** (not a
   user-visible change — `n_prompt` equals it wherever both were defined; this is the
   no-double-count enforcement).
3. **Intersection tier note:** on q35 NAKED (spec tier serves greedy chat), tools
   requests get `cached_tokens: 0` by the spec-bypass policy — caching for tools
   traffic shows on the bulk tier (`MEMRA_SERVE_SPEC=0`, the marketplace concurrency
   config), exactly like non-tools traffic. Both lanes' policies kept verbatim.
4. One pre-existing warning (`unused_mut` on `consume` in blocking_response) ships in
   HEAD's tools code — left untouched to keep merge attribution clean.

## Gate battery (merged binary, RTX 5090, every GPU run under flock /tmp/gpu5090.lock)

Runner: `run-gates.sh` (5 holds, one server boot per hold; servers killed by PID; the
fa-decode lane interleaved between holds throughout — lock discipline held). Raw logs
next to this file. Model: q35 = Qwen3.6-35B-A3B-UD-IQ4_XS.

| gate | verdict |
|---|---|
| 1. build (`nice cargo build --release`) | clean (one pre-existing `unused_mut` warning from HEAD) |
| 1. unit tests (`-p memra-server`, `-p memra-tokenizer`) | 20/20 + 3/3 PASS |
| 1. kernel-check q35 | rc=0, **388 OK / 0 FAIL** (`battery-kernel-check.log`) |
| 2. tools round-trip battery (A/A-leg2/A-nothink/B-stream/C-malformed/D-usage/E-bijection) | **7/7 PASS**, N=3 byte-stable per leg (`gates-q35.jsonl`) — D-usage tok-check exact: plain 27 / tools 330 / no-think 332, same counts as the serve-tools lane |
| 3. cache exactness (16 cells): partial==split-prime / full==cold / usage truth / cold==0 / control | **16/16 x5 ALL GREEN** (`gate-exact.jsonl`); cross-config REPORT moved 6/16 — same near-tie class AND count as the lane's own run |
| 4. THE INTERSECTION (tools x cache, bulk tier) | **PASS** — see below |
| 5. serve greedy c1-vs-c16, q35 NAKED | **PASS 16/16** (`greedy-hash-q35-naked.jsonl`) |

### Gate 4, the intersection — verdict detail

`intersection_gate.py`, bulk tier (`MEMRA_SERVE_SPEC=0`, the tier the cache serves —
on NAKED q35 the spec tier serves tools chat and bypasses the cache by the cache
lane's policy), default 256MB budget.

- Leg R (same get_weather tools request 3x): rep1 cold `cached_tokens: 0`; rep2/rep3
  FULL-prefix hits `cached_tokens == prompt_tokens == 330 > 0`. Every rep:
  `finish_reason: "tool_calls"`, `get_weather{"city":"Paris"}` parses, `prompt_tokens
  330 == tok-check 330` (worker truth, tools block included), `total = prompt +
  completion`, `cached <= prompt`. The parsed tool_calls are BYTE-IDENTICAL across
  cold and hits.
- Leg M (same tools + system, distinct user turns): A cold 0/361; B and C hit the
  shared boundary `cached_tokens: 301` with `0 < 301 < 360/361`; B re-sent ==
  B byte-identical. All four tok-check exact. The 301-token boundary is the tools
  block + shared system region — the tools rendering IS the cacheable prefix, no
  special-casing anywhere.

Gate-law history (append-only in `intersection-gates.jsonl`, all rows kept): the first
run's FAILs are a GATE-SCRIPT bug, not a server bug — the request body never carried
`tools` (rows show prompt_tokens 27 = the plain render vs tok_check 330). The second
run's single FAIL was a gate-law overreach: it demanded full content byte-identity
(think prose included) between cold and full-hit, which is NOT the specified law
("the tool_call still parses identically") — and the divergence it caught is real,
pre-existing, and NOT a tools/merge artifact (next section). Law recalibrated to gate
the parsed tool_calls + finish identity and REPORT content identity; battery re-run.

## FINDING for the owner: cold-vs-full-hit near-tie divergence (pre-dates the merge)

On the bulk tier, a full-prefix-hit generation can diverge from the cold generation
that seeded the entry, at a think-prose near-tie ("function" vs "tool", diverge char
155, ~40 tokens in; cold 158 completion tokens vs hit 177; the parsed tool_call and
final answer class are unchanged). Attribution (`run-attribution.sh`, one flock hold
per probe, transcripts `attr-*.json`):

| probe | result |
|---|---|
| merged binary, cache OFF, same tools request 2x (cold-vs-cold) | **byte-identical** — cold path deterministic |
| merged binary, cache ON, RAW /v1/completions of the rendered bytes 3x | rep1 158 tok vs rep2/rep3 177 tok, diverge char 155; **rep2 == rep3** (hit path deterministic) |
| PRE-MERGE cache-lane binary (af72c3db), same raw 3x | **identical divergence, same char 155** |

So: not a tools-surface effect (reproduces on raw token-id completions with no tools
API involved), not a merge effect (reproduces byte-for-byte on the pre-merge lane
binary), and not nondeterminism (each path is individually deterministic). It is a
cross-path FP difference between decode-from-restored-state and decode-from-live-
primed-state that surfaces at near-ties on longer generations — the cache lane's
16-cell gate-full law (A2==A1, ~96-token gens) passed 16/16 on both binaries and did
not have the near-tie exposure to catch it. Same family as the documented
batched-prime near-tie law, but it is a LATENT HOLE in the "full hit == cold"
exactness contract as stated in docs/SERVING.md — the owner should decide whether to
(a) hunt the restored-state FP delta in the engine (KV/recurrent restore is
byte-exact by construction, so the suspect is session-state/config asymmetry between
the two decode entries), or (b) soften the documented contract to "deterministic and
call-equivalent, near-tie prose may move" and add a longer-generation cell to the
cache lane's gate.
