# Session-affinity resume — resuming a REWRITTEN conversation

Lane `lane/session-affinity` (parent `restructure/public-split` @ `44c4c6a4`), 2026-08-05.
Rig: local RTX 5090 Laptop (24 GB), the owner's daily-driver serve config verbatim.

## The problem this lane closes

The spec pool resumed a parked session only when the new prompt EXACTLY EXTENDED it —
token-prefix, or (since 2026-07-06) text-prefix. The owner's agent client rewrites
conversation history between turns: `<think>` blocks are stripped out of PRIOR assistant
turns before re-sending. Turn N's prompt is therefore NOT a prefix-extension of turn N-1's
committed text, both probes miss on every turn, the parked ~4 GB session is discarded as
dead weight, and the whole growing conversation re-primes.

Receipts for the miss: `research/specpool-20260804/RESULTS.md` (F5 removed the
evict-realloc churn but not the re-prime) and `research/memra-vs-llama-daily-20260805/RESULTS.md`
(short-turn TTFT 0.53 s vs llama's 0.19 s; the prime wall named as one of three causes).

## Design as-built

Identity NOMINATES a candidate; BYTES decide whether resuming it is exact. That split is the
whole safety argument: a fingerprint collision can only ever cost a wasted probe, never a
wrong resume. Code + rationale: `crates/memra-server/src/worker.rs` (the
`SESSION AFFINITY` block); user-facing contract: `docs/SERVING.md` ("Session affinity").

### Tier (a) — explicit: the client names its conversation

`affinity_key()` in `main.rs` accepts three spellings, priority order:

1. `session_id` request-body field — the explicit spelling.
2. `user` request-body field — OpenAI's field, which real clients already send.
3. `x-session-id` request header — the convention proxies in front of vLLM/TGI use.

Body beats header (the body is the caller's own statement of identity; a header can be
injected by an intermediary); `session_id` beats `user`. Blank/whitespace values are treated
as absent, values are trimmed. Explicit-on-one-side-only NEVER matches: if the request names
a conversation and the parked session does not (or vice versa), affinity declines rather than
guess. Unit test: `affinity_key_honors_both_client_conventions_in_priority_order`.

### Tier (b) — implicit: a structural fingerprint

Nothing named, so identity is structural. `conversation_fingerprint()` splits the token stream
at the tokenizer's CONTROL tokens (exactly what a chat template emits at turn boundaries:
`<|im_start|>`/`<|im_end|>` and friends) and hashes, per segment, only its first and last
`FP_WINDOW = 8` tokens — never its interior.

- A CHAIN, not one digest: turn N+1 has strictly MORE segments than turn N, so a
  whole-conversation digest could never match across turns. Identity is a PREFIX relation over
  the chain, and the nomination bar is a shared leading run of `FP_MIN_SEGMENTS = 3`.
- BOUNDARY WINDOWS: the rewrite class mutates segment INTERIORS (a stripped `<think>` block is
  deleted text inside a turn), so interior-blind hashes are invariant under it. Where a rewrite
  does touch a head window, the chain degrades gracefully — that segment's hash changes, the
  shared run just ends earlier, and the candidate is still nominated on the stable prefix
  (system prompt + early turns, which no client rewrites).
- The 3-segment bar exists because a bare system prompt is byte-identical across every fresh
  conversation with the same client; nominating on it would cross-link unrelated conversations.
  Test: `fingerprint_declines_short_generic_openers`.
- Markerless raw prompts produce a 1-segment chain (and for a request, that single segment is
  the live turn, so the chain is empty) — they can never clear the bar, so non-chat callers
  keep the plain prefix probes untouched.
  Test: `fingerprint_handles_a_prompt_with_no_markers`.

The owner's client renders the chat template CLIENT-side and posts raw `/v1/completions`, so
there is no `chat_turns` structure to walk. Recovering segments from the control tokens in the
flat stream is what makes the implicit tier work on that path — `Tokenizer::encode` parses
specials, so client-rendered `<|im_start|>` arrive as control tokens.

### Resume semantics — the exactness contract

`affinity_match(prompt, committed[..rewind_pos])` is the only authority. Committed tokens are
authoritative state: the caches hold exactly their KV/recurrent state. Resuming is exact iff
the prompt reproduces the session's committed tokens up to its REWIND BOUNDARY exactly, with a
non-empty suffix left to prime. Any divergence inside that range means the caches hold state
for tokens this request does not have — and hybrid GDN recurrent state is mutated in place with
no per-position index to truncate, so no amount of suffix priming repairs it. Divergence =
full re-prime, always. A prompt SHORTER than committed is divergence too (extra committed rows,
no boundary to trim at).

The rewind boundary is a per-turn checkpoint the spec session retains at THE PROMPT END. That
placement is load-bearing — see the bug below.

Scope: per `PoolKey = (model, cache_ns)`, so affinity can only nominate a session inside its own
PC-ISO namespace and adds NO new cross-tenant reach. Constrained (grammar) requests never
resume: the park's stashed `next_pred`/`pending` is unconstrained state.

F5 interaction: the probe tests `need = prompt + budget + SPEC_SHRINK_SLACK`, not `ctx_cap`.
On a VRAM-tight rig F5's ladder lands sessions BELOW `ctx_cap` — and those are exactly the rigs
where every turn is a miss, so gating on `ctx_cap` would reject every laddered session forever.
A resumed session that no longer fits its next turn simply misses and follows the ladder as a
new one. Test: `affinity_room_test_accepts_f5_right_sized_sessions`.

## THE BUG: the checkpoint sat one token PAST the prompt end

Slice 4 wired the whole path and affinity fired ZERO times on the owner regime. The decline
diagnostic added in slice 5 named it in one line:

```
[worker] spec-affinity: declined (history diverged at 12233 of checkpoint 12234;
                                  1 parked, 12317 prompt tokens)
```

Diverging ONE TOKEN below the boundary is not a history rewrite. The capture sat after the
draft-KV fill, hence after the init feed `decode_step_h(last_token)`, so
`pos = base + prompt.len() + 1` and the boundary included the FIRST GENERATED TOKEN — which,
on a reasoning model, is the first thing inside the `<think>` block the client strips. Affinity
declined 100% of the time while looking like a safe correctness decline.

Fix (`96beb3a6`, `crates/memra-engine/src/spec.rs`): move the capture to before the init feed,
plus `debug_assert_eq!(pos, base + prompt.len())` so the invariant can never drift again.
Immediately after: turn 2 16.4 s -> 6.5 s, `cached_tokens` 0 -> 12233.

(An earlier comment on this decline hypothesized a re-tokenization seam. That hypothesis was
WRONG and is superseded by the above; the seam text survives in the code only as the
documented meaning of an `at` far below `pos`.)

## Byte-identity verdict — 25-turn rewrite pattern

Harness `drive-affinity.py`. It renders ChatML client-side, and after each turn deletes the
`<think>...</think>` span from the answer it stores — an interior edit under surviving
boundaries, i.e. the exact rewrite class. Every row carries `rewrote`, and the gate FAILS if no
turn rewrote its history: a run can never claim a regime it did not reproduce.

Both arms replay prompts rebuilt from ONE recorded transcript (`t25.json`, `--replay`). Two
independently-driven conversations would diverge into uninterpretable rows after the first
mismatch; replay keeps every turn's prompt byte-identical across arms.

| arm | file | rewinds | prefix resumes |
|---|---|---|---|
| resume (`MEMRA_AFFINITY=1`) | `g25-resume.jsonl` | 21 | 3 |
| resume, repeat | `g25-resume2.jsonl` | 21 | 3 |
| fresh (`MEMRA_AFFINITY=0`) | `g25-fresh.jsonl` | 0 | 2 |

**22/25 turns byte-identical.** Three turns differ: 2, 3, 24 (divergence 2236 / 882 / 137
chars deep). The affinity arm itself is DETERMINISTIC — `g25-resume` vs `g25-resume2`,
same flag, same prompts, **25/25 identical `text_sha`**.

### Those three are a PRE-EXISTING class, not affinity

Four independent receipts, each one able to falsify the claim on its own:

1. **`nx` (short answers, no rewrite)** — pool vs cold **4/4 identical** with 2 affinity
   rewinds. Affinity resumes ARE byte-exact when generation stops early.
   (`nx-pool.jsonl` / `nx-cold.jsonl`)
2. **`lx` (long answers, no rewrite)** — **0 affinity rewinds, 3 prefix resumes**, and pool
   still diverged from cold at turns 1-2. Divergence exists with affinity never firing.
   (`lx-pool.jsonl` / `lx-cold.jsonl`)
3. **BASE binary (pre-lane, `44c4c6a4`, `wt-affinity-base`)** — pool-on turn 1
   `61e5037737fe280d` vs cold `12f8a2903b5e552b` on the SAME prompt. The pre-lane code
   diverges resumed-vs-cold at long windows by itself. (`base-pool.jsonl` / `base-cold3.jsonl`)
4. **Cold ground truth is identical across binaries** — per-turn `--only` arms on a fresh
   server: turn 2 `a47e7602ae2da841`, turn 3 `1343148fc68b5e20`, turn 24 `8d5d710a27e1b814`,
   LANE == BASE on all three. (`one-lane-*.jsonl` / `one-base-*.jsonl`;
   `lx1-lane.jsonl` == `lx1-base.jsonl` == `12f8a2903b5e552b`)

Verdict: **resumed-vs-cold divergence at long generation windows is pre-existing in the
prefix-resume tier and is NOT introduced by affinity.** Affinity inherits it; it does not
create it. Naming it as a separate open item rather than folding it into this lane's result.

## The number that matters — TTFT/turn, owner regime

N=3 interleaved reps per arm (on, off, on, off, on, off — never all of one arm then the
other), same recorded transcript replayed by both, `flock /tmp/gpu5090.lock` held for the whole
sweep, no other GPU tenant. Thermal regime: warm steady-state, GPU 61 C / 180 MHz idle at
sweep start, cards otherwise unloaded (the daily driver was down). Per-turn MEDIAN over the 3
reps — never a mean over turns: turn 0 is a cold prime in both arms and the rewrite turns are
the interesting ones, so a single aggregate would hide the shape.

Rows: `ttft-{on,off}-r{1,2,3}.jsonl` + `.server.log`. Table via
`drive-affinity.py --curve wall_s ...`.

| turn | affinity ON | OFF | speedup | note |
|---|---|---|---|---|
| 0 | 16.283 | 16.347 | 1.00x | cold prime, both arms — nothing to resume |
| 1 | 4.012 | 4.036 | 1.01x | pure extension: the PREFIX probe serves both arms |
| 2 | 6.625 | 17.118 | 2.58x | first rewritten history — OFF re-primes 12.3k |
| 3 | 3.518 | 15.460 | 4.39x | |
| 4 | 3.496 | 14.445 | 4.13x | |
| 5-21 | 0.866-0.884 | 11.8-13.3 | **13.7-15.0x** | the steady-state agent regime |
| 22 | 6.334 | 18.699 | 2.95x | long answer (601 tok) |
| 23 | 0.884 | 0.883 | 1.00x | extension turn: prefix probe hits in both arms |
| 24 | 3.809 | 16.291 | 4.28x | |

Sum-of-medians over the 25 turns: **59.8 s ON vs 317.2 s OFF = 5.30x**.

Read the steady-state block, not the total: at turns 5-21 the conversation is 13k-14.4k tokens
and each turn's answer is ~63 tokens. `MEMRA_AFFINITY=0` spends ~12-13 s re-priming that whole
window before emitting a token; affinity primes the ~85-token delta and answers in 0.87 s. The
brief's target — turn-N drops from the full-re-prime class to the prefill-of-delta class — is
met: **~12.3 s -> 0.87 s, and the ON arm's per-turn cost is FLAT in conversation length**
(0.866 s at turn 5 / 13.0k tokens, 0.884 s at turn 21 / 14.4k) while the OFF arm's grows
monotonically with it (11.82 -> 13.30 s). That flatness is the actual result: TTFT stops
scaling with history.

`wall_s` bundles prefill with generation, so it is only a fair per-turn comparison where both
arms emit the same token count — which the replay harness enforces prompt-side, and the
`completion_tokens` columns confirm on 22 of 25 turns (the 3 exceptions are the pre-existing
divergence turns above). For the isolated first-token number the harness gained `--stream`,
which stops the clock on the first SSE chunk carrying text; streamed rows are
`st-{on,off}-r{1,2,3}.jsonl` with a populated `ttft_s` column.

## Gates

| gate | result |
|---|---|
| `tools/local-ci.sh` correctness | GREEN — kernel-check ALL GREEN, prime-gate 8/8 MATCH, run-gen argmax MATCH (31B + 12B), VERIFY-GATE K=7 PASS, spec self-consistency 64/64 PASS |
| `tools/serve-smoke.sh` | 0 failed |
| `tools/serve-st-gate.sh` | 0 failed |
| `cargo test -p memra-server --release` | 60 passed, 0 failed |

New permanent gate: **serve-smoke check 10** — session-affinity resume exactness. Records a
4-turn rewritten-history conversation with `MEMRA_AFFINITY=1`, replays the SAME prompts with
`MEMRA_AFFINITY=0`, and requires the texts to match (burst-overshoot prefix tolerance, same as
serve-st-gate check 4). Plus a LIVENESS assertion: the affinity arm's log must show a rewind,
because a binary where affinity never fires passes the identity half trivially — measured, it
did exactly that before `96beb3a6`. Check 10 lives INSIDE the battery on purpose (the H100
lane's law 3: a gate outside the battery rots silently).

## Fixed in passing

`MEMRA_REUSE_POOL=0` panicked the worker thread on the first session retire —
`removal index (is 0) should be < len (is 0)`, `pool.remove(0)` under a `len() >= cap` loop with
cap 0. Verified PRE-EXISTING on the base binary before fixing. Both park sites (spec pool and
legacy reuse pool) now guard the cap. Found while building this lane's control arm.

## Flag

`MEMRA_AFFINITY=0` — rollback seam / exactness A/B arm, not a tuning knob. Documented in
`docs/FLAGS.md` §serving. `MEMRA_REUSE_POOL=0` cannot substitute: it also kills the
token/text-prefix resumes, so a divergence could not be attributed to affinity.

## Divergence from the brief

The brief says "the api-keys merge added TenantCtx — affinity must never cross tenants".
`crates/memra-server/src/auth.rs` does not exist on `restructure/public-split`; that merge is
an ancestor of `lane/q27-deepdive-20260805` only. On this base the isolation boundary is
`PoolKey = (model, cache_ns)`, so affinity is scoped by `PoolKey` and inherits `TenantCtx` for
free when api-keys merges (its per-key namespace derivation flows into `cache_ns`).

## Files

- `drive-affinity.py` — driver: rewrite-pattern conversation, `--replay` (both arms, one
  transcript), `--only TURN` (purest cold control: fresh server, one request), `--cold`
  (per-turn `cache_salt`), `--stream` (TTFT), `--gate`, `--curve`.
- `run-arm.sh` — boots the owner regime verbatim; `MEMRA_AFFINITY` is the only thing an arm
  varies; tails resume counts into the output next to the rows.
- `t25.json` — the recorded 25-turn transcript both arms replay.
- `*.jsonl` + `*.server.log` — every run's raw rows and raw server log. The resume decisions
  live in the LOG, not the HTTP responses, so a summary can never drift from its receipt.
