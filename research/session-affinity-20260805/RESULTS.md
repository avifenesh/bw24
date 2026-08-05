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

The streamed TTFT sweep is an independent re-run of the same comparison, 3x: **22/25 in every
rep, with the SAME three mismatch turns (2, 3, 24) each time**, and the ON arm 25/25
identical across all three reps. The verdict and its exceptions are reproducible, not noise.

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

### ROOT CAUSE of that class: chunked prefill is not reduction-order-stable

Found while building serve-smoke check 10, which failed at turn 2 on the 9B — a 149-token
window, far too short for a "long window" explanation. Isolating it produced a stronger
statement than the attribution above: **resumed == cold is not a property this engine has, on
any reuse tier, and no reuse is required to break it.**

Arms: the SAME 4 recorded prompts, per-turn `cache_salt` (so no tier can hit — every request
primes cold), `MEMRA_AFFINITY=0`, varying ONLY `MEMRA_PRIME_CHUNK`:

| turn | prompt tokens | chunk 2048 vs 64 | chunk 2048 vs 32 |
|---|---|---|---|
| 0 | 48 | identical | identical |
| 1 | 97 | identical | **differs @ char 45** |
| 2 | 149 | **differs @ char 172** | **differs @ char 52** |
| 3 | 195 | identical | identical |

Chunk boundaries alone change greedy output: a different split changes the reduction order in
the prefill GEMMs, which perturbs logits in the last bits and flips a near-tie argmax. Turn 2
is the same turn check 10 failed on, and it is chunk-sensitive with zero reuse involved.

Every resume necessarily RE-CHUNKS the prefill — it primes `[rewind boundary .. end]` as its
own chunk sequence instead of one full prime — so every resume tier inherits this. That is why
the pre-lane BASE binary already diverged pool-vs-cold (receipt 3), and why the `lx` arm
diverged with 0 affinity rewinds (receipt 2).

Consequences taken here:
- The 22/25 result stands as reported, and its 3 exceptions are now explained by mechanism, not
  just correlated with a pre-existing arm.
- serve-smoke check 10 does NOT assert resumed == cold. It asserts what affinity owns:
  determinism of the resume path across servers, plus liveness. Wiring the naive assertion
  would have put a permanently-red gate in the battery and blamed affinity for chunked
  prefill's reduction order.
- OPEN ITEM, not this lane's: whether chunked prefill should be made reduction-order-stable
  (fixed accumulation order independent of chunk split). Until it is, no gate anywhere in the
  repo may assert byte-equality between two prefills of the same prompt at different chunk
  boundaries. Worth noting `MEMRA_PRIME_CHUNK` is a documented machine-config knob, so two rigs
  with different values already produce different greedy text on the same prompt.
- Reproducer + raw rows: `chunk-order-probe.py` and `chunk-order.jsonl` (12 rows = 3 chunk
  sizes x 4 prompts, each with its text; under two minutes on the 9B). Boot the 9B+draft with
  `MEMRA_PRIME_CHUNK=<n> MEMRA_AFFINITY=0` and run the probe once per value.

## The number that matters — TTFT/turn, owner regime

Measurement regime for both sweeps below: N=3 INTERLEAVED reps per arm (on, off, on, off, on,
off — never all of one arm then the other, per the H100 lane's law 1), the same recorded
transcript replayed by both arms so every turn's prompt is byte-identical across them,
`flock /tmp/gpu5090.lock` held for the whole sweep with no other GPU tenant (the daily driver
was down). Thermal: cold start at 61 C / 180 MHz idle, ending 85 C / 1717 MHz — i.e. the sweep
warms into steady state, and each arm's replicates are spread across that ramp by the
interleave rather than one arm owning the cold end. Per-turn MEDIAN over the 3 reps, never a
mean over turns: turn 0 is a cold prime in both arms and the rewrite turns are the interesting
ones, so a single aggregate would hide the shape.

### TTFT (the lane's number)

Streamed (`--stream`): the clock stops on the first SSE chunk carrying text, so this is
prefill + one decode step — exactly the quantity the resume path shortcuts. Rows:
`st-{on,off}-r{1,2,3}.jsonl` + `.server.log`. Table via `drive-affinity.py --curve ttft_s ...`.

| turn | affinity ON | OFF | speedup | note |
|---|---|---|---|---|
| 0 | 9.882 | 9.962 | 1.01x | cold prime, both arms — nothing to resume |
| 1 | 0.590 | 0.591 | 1.00x | pure extension: the PREFIX probe serves both arms |
| 2 | 0.562 | 11.283 | **20.1x** | first rewritten history — OFF re-primes 12.3k |
| 3 | 0.600 | 11.896 | 19.8x | |
| 4 | 0.573 | 11.899 | 20.8x | |
| 5-22 | 0.525-0.548 | 11.89-13.36 | **22.4-24.5x** | the steady-state agent regime |
| 23 | 0.544 | 0.541 | 0.99x | extension turn: prefix probe hits in both arms |
| 24 | 0.645 | 14.031 | 21.8x | |

Sum-of-medians over the 25 turns: **23.1 s ON vs 287.2 s OFF = 12.4x**.

The brief's target was turn-N TTFT dropping from the ~3 s full-re-prime class to the
prefill-of-delta class (~0.2-0.4 s). Measured on the owner's real conversation shape the
control is far worse than 3 s (11-14 s at 12-15k tokens) and the resumed arm lands at
**0.53-0.65 s** — just above the target band, and the residual is not prefill: at turn 5 the
delta is ~85 tokens, so ~0.53 s is the fixed per-turn floor (rewind + delta prime + first
decode step), not work that scales.

**The result is the FLATNESS, not the ratio.** ON: 0.525 s at 13.1k prompt tokens, 0.548 s at
14.6k. OFF: 11.89 s -> 13.36 s across the same span. TTFT stops scaling with conversation
length — which is the property a daily driver needs, and the ratio only grows as the
conversation does. Per-rep, at the extremes of the steady block: rep1 0.525 -> 0.541,
rep2 0.534 -> 0.548, rep3 0.521 -> 0.566.

### Total wall (the earlier sweep, kept as the corroborating arm)

`wall_s` bundles prefill with generation, so it is only a fair comparison where both arms emit
the same token count — the replay harness enforces the prompt side, and `completion_tokens`
confirms it on 22 of 25 turns (the 3 exceptions are the pre-existing divergence turns above).
Rows: `ttft-{on,off}-r{1,2,3}.jsonl` (named before the streamed mode existed; they carry
`wall_s` only).

| turn | ON | OFF | speedup |
|---|---|---|---|
| 0 | 16.283 | 16.347 | 1.00x |
| 1 | 4.012 | 4.036 | 1.01x |
| 2 | 6.625 | 17.118 | 2.58x |
| 5-21 | 0.866-0.884 | 11.8-13.3 | 13.7-15.0x |
| 22 | 6.334 | 18.699 | 2.95x |
| 24 | 3.809 | 16.291 | 4.28x |

Sum-of-medians: 59.8 s ON vs 317.2 s OFF = 5.30x. The end-to-end ratio is necessarily smaller
than the TTFT ratio because decode time is identical in both arms — affinity removes prefill,
not generation.

Resume counts (from the server logs, not the HTTP responses): every ON rep 21 affinity rewinds
+ 3 prefix resumes, every OFF rep 0 rewinds + 2 prefix resumes — identical across all 3 reps
of both sweeps.

## Gates

Every gate below ran on top of the lane's final code. **Full battery GREEN, 0 failed.**

| gate | result |
|---|---|
| `kernel-check` | ALL GREEN |
| `prime-gate` (q35 mixed prompts) | 8/8 MATCH, 0 flip-neartie, 0 structured, 0 det_fails |
| `run-gen` argmax 31B | MATCH (prefill 4694 == decode 4694, maxdiff 1.063e0) |
| `run-gen` argmax 12B depth | MATCH (623 == 623; batched-prime == tokenwise, maxdiff 2.438e0) |
| VERIFY-GATE K=7 depth 31B | PASS |
| VERIFY-GATE K=7 depth 12B | PASS |
| spec self-consistency 31B | **stream agreement 64/64** (plain 40.11 tok/s, spec 105.83, 2.64x) |
| `tools/serve-smoke.sh` (incl. check 10) | **0 failed** |
| `tools/serve-st-gate.sh` | **0 failed** |
| `cargo test -p memra-server --release` | 60 passed, 0 failed |

Sequencing note, because it matters for what the GREEN covers: the 31B/12B arms first OOM'd
when the owner's daily driver came back up mid-battery holding 14786 MiB of the 24 GB card
(receipts preserved in `gates-head.log` — `Error: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of
memory")` for the 31B, `embed table upload: DriverError(...)` for the 12B, with the
`nvidia-smi` compute-apps state at failure time). Those were capacity failures, not
correctness failures, and they were reported as blocked rather than as passes. The owner then
confirmed the driver was idle, it was stopped, and all five arms were re-run on the free card:
all GREEN, as tabulated above. Receipts: `gates-head-rerun.log`.

Code identity across the whole battery is auditable rather than asserted:
`git diff 96beb3a6 HEAD -- crates/ cu/` is a ONE-LINE doc-comment change in `worker.rs`
(a `docs/API.md` -> `docs/SERVING.md` reference). No engine or kernel code moved.

New permanent gate: **serve-smoke check 10** — session-affinity resume. Records a 4-turn
rewritten-history conversation, replays the SAME prompts against a FRESH resuming server, and
asserts four things: (a) the two arms' texts match (burst-overshoot prefix tolerance, same as
serve-st-gate check 4) — determinism of the resume path across servers; (b) the recording arm
logged a rewind; (c) the replay arm logged one too; (d) no `affinity rewind failed` line, which
is the one message meaning state was accepted and then could not be restored. Liveness is not
optional: a binary where affinity never fires passes (a) trivially, and that is measurably what
the pre-`96beb3a6` binary did.

Check 10 deliberately does NOT assert resumed == cold. That assertion was written first, failed
at turn 2, and isolation showed it is not a property of the engine (chunked prefill, above).
Check 10 lives INSIDE the battery on purpose (the H100 lane's law 3: a gate outside the battery
rots silently) — and no reuse-exactness gate existed anywhere in `tools/` before it.

Measured on the 9B NVFP4 + regime draft: `serve-smoke: 0 failed`, check 10 all four green with
3 rewinds per arm.

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
