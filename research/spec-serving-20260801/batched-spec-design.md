# Batched-spec serving: what exists, what it costs, what multi-session spec would take

Lane: spec-serving probe, darklanes H100 box GPU 4, 2026-08-01.
Model: q27 = Qwen3.6-27B-Q4_K_M (MTP head baked in), 65 layers, dense-hybrid (GDN linear-attn
+ full-attn layers). Spec config = round-47 sweep winner: `MEMRA_SPEC_K=3 MEMRA_SPEC_HPOST=1
MEMRA_SPEC_PMIN=0.3`. Raw points: `points.jsonl` / `per-request.jsonl` in this directory;
server logs per arm per round alongside.

## 1. What EXISTS today (recon, file-accurate)

Spec serving is a real, default-on serve path — but a **single-sequence** one.

- `crates/memra-server/src/worker.rs` — `Session.spec: Option<memra_engine::spec::SpecSession>`
  (line ~190). `admit()` attaches a SpecSession when: sampler is greedy OR pure-temp sampled,
  the model has an MTP head, the request is not a KV-reuse resume, and `MEMRA_SERVE_SPEC!=0`
  (default ON; read per-admit at line ~873). On an MTP model this captures effectively **all**
  traffic — there is no per-request opt-out, and no mixed spec/plain mode from the API.
- **Multiple concurrent spec sessions are structurally supported**: each `SpecSession`
  (`crates/memra-engine/src/spec.rs:167`) owns its own trunk `Cache`, MTP draft scratch KV
  (`MtpScratch`), committed-token list, and Philox counters. Sessions are fully isolated.
- **But the scheduler serializes them.** Tick phase (a) (`worker.rs:489-501`) steps each spec
  session with a SOLO burst — `generate_spec_session_sampled`, up to `MEMRA_SPEC_BURST`
  (default 32) tokens of draft-K/verify rounds for ONE session — then moves to the next.
  `worker.rs:384`: "(a) spec sessions burst solo (spec x batch composition is a later step)".
- Consequence: on an MTP model the batched-decode phase (c) (`decode_step_batch_sampled`,
  chunks <= `decode_batch_cap()`=8, `crates/memra-engine/src/decode_batch.rs:106`) is
  **unreachable** for greedy/temp traffic. Spec sessions are also excluded from prime-batching
  (`worker.rs:529` candidate filter `s.spec.is_none()`) — they prime inside the spec path.
  This is the ledger note "MTP models batch only under SERVE_SPEC=0"
  (`research/next-targets-20260730.md:142`), confirmed in code.

So: **nothing "breaks"** — no scheduler bug, no shape mismatch, no cache conflict. It is a
designed serialization. What breaks *throughput* is that every burst is a B=1 trunk pass:
N spec sessions time-slice one single-stream spec pipeline.

## 2. Measured (this directory, GPU 4, H100 80GB)

Protocol: `tools/load-serve.py`, greedy, max_tokens=128, ~200-token fixed prompt,
requests = max(8, 4c). Arms alternate per round (plain server :8185 with
`MEMRA_SERVE_SPEC=0`, spec server :8186 with defaults + round-47 config; one server resident
at a time — one loaded q27 server holds ~40.7 GB with the q8rp decode mirrors, two do not fit
80 GB; OOM receipts in `server-plain-8085.log` / `server-spec-8186.log`). N=3-4 interleaved
rounds per point, same box, same hour; medians below, ranges in `summarize.py` output.

| point | plain agg tok/s | spec agg tok/s | spec/plain | plain p50 lat | spec p50 lat |
|---|---|---|---|---|---|
| c=1 | 72.5 | 125.5 | **1.73x** | 1.76s | 1.01s |
| c=2 | 111.2 | 126.2 | 1.14x | 2.30s | 2.03s |
| c=4 | 156.9 | 126.4 | 0.81x | 3.25s | 4.05s |
| c=8 | 182.5 | 126.5 | **0.69x** | 5.60s | 8.09s |

- Spec aggregate is **flat at ~126 tok/s at every concurrency** — the serialization signature.
  Per-session latency scales linearly with c (1.01s -> 8.09s): each session waits its turn
  through everyone else's bursts.
- Plain batching scales 2.52x from c1 to c8 (decode chunks of <= 8).
- **Crossover between c=2 and c=4.** At c=2 spec still wins aggregate (and latency); from c=4
  up, plain batching wins aggregate and latency both.
- Acceptance: 89.0% over all 789 bursts (17421/19579 drafted accepted; per-burst
  `[spec-acc]` lines in the spec server logs). c1 ratio 1.73x sits at the top of the round-47
  single-stream window (1.35-1.76x) — this prompt drafts well.
- Correctness (the serving exactness contract): spec-serve output == plain-serve output,
  greedy, exact common-prefix on both probe prompts (`correctness-*.json`; spec p1 emitted
  161 vs 160 tokens = the documented commit-past-max_new overshoot, prefix identical).
- Data-quality note: round-3 plain c4/c8 points died (`RemoteDisconnected` then refused);
  the server log ends silently after healthy `[prime-batch]` lines, no user-space error
  captured. Concurrent recorded state: NVRM assertions on GPUs 2/4/5/6/7 at 07:11:32
  (`nvAssertFailedNoLog ... kernel_graphics.c:3425`) — a box-wide driver event in that exact
  window while three other lanes ran. Cause unknown by the evidence rules; a full make-up
  round 4 reproduced nothing and matched rounds 1-2 within 2%. Dead points carry
  `n_ok=0` in `points.jsonl` and are excluded from medians.

**The darklanes money answer**: MTP spec does NOT survive batched serving today — it survives
only as a **single-stream fast lane**. The real product framing that exists right now is a
latency tier: a dedicated spec server (1.7x per-session tokens/s, 1.01s vs 1.76s for a 128-token
turn) for 1-2 premium/interactive sessions, and plain batched servers for bulk traffic. The
switch is process-level (`MEMRA_SERVE_SPEC`), so the tiers are separate server processes —
which is also how you'd deploy them anyway.

### 2b. Burst-size sensitivity (the latency-tier tuning knob, measured)

`MEMRA_SPEC_BURST` (tokens per solo burst before the scheduler moves to the next session)
at c=4/c=8, x2 interleaved rounds (`burst-points.jsonl`):

| burst | agg tok/s (c4 / c8) | p50 lat (c4 / c8) | vs default 32 |
|---|---|---|---|
| 8   | 105.6 / 105.7 | 4.89s / 9.76s | -16% |
| 32  | 125.8 / 126.2 | 4.06s / 8.11s | — |
| 128 | 131.5 / 131.8 | 3.88s / 7.76s | +4.4% |

- The per-burst fixed cost is real and measurable: 128 tokens take 1.211s of engine time at
  burst=8 (16 bursts) vs 0.971s at burst=128 (1 burst) -> **~16 ms fixed cost per burst**
  (draft-graph recapture + session setup, the `worker.rs:1174` comment). Design item (f)
  (persistent per-session draft graphs) is worth ~4% aggregate at the default burst=32 all
  by itself, and is what would make SMALL bursts (fine-grained fairness) affordable.
- In this uniform closed-loop workload bigger bursts win latency too (fewer fixed costs,
  everyone finishes sooner). The fairness cost of large bursts appears only for a short
  interactive request arriving among long ones: worst-case stall = (c-1) x burst_time
  (~1s/128-tok burst -> ~7s at c=8) — that is the real reason burst=32 is the default, and
  it is a workload policy, not a throughput optimum.

## 3. What multi-session spec serving would take (file map, effort classes)

Target design: **round-lockstep batched verify.** Per tick, collect the live spec sessions,
run each session's K-token draft chain (cheap: MTP block = 1 layer vs 65 trunk layers), then
ONE varlen batched trunk verify over all sessions' K+1 columns, then per-session accept walks.
The spec multiplier then stacks on the batching multiplier instead of replacing it: with 89%
acceptance and the measured plain-c8 batch efficiency, the c=8 projection is roughly the plain
curve x ~1.7 ≈ 300+ tok/s aggregate (acceptance-dependent; treat as a ceiling estimate, not a
promise).

Pieces, with what already exists:

(a) **Per-session draft state — EXISTS, zero work.** `SpecSession` already isolates trunk
    cache, draft scratch KV, committed list, sampled-spec Philox counters
    (`spec.rs:167-190`). The worker already holds one per session.

(b) **Round-loop extraction — the big one. Effort: LARGE.**
    `generate_spec_inner2` (`spec.rs:2050`, the ~700-line burst orchestrator) is deeply
    single-session: persistent per-call device buffers, per-call draft CUDA-graph capture,
    the h_seed/fill_prev predecessor-pairing invariants, `VerifyCkpt` replay-free partial
    accept, `set_len` rollback. Lockstep needs it refactored into a resumable per-round state
    machine (draft -> yield -> verify_batch -> accept -> yield) the worker can drive across B
    sessions. Every loop invariant must survive the split, and the session-gate oracle +
    run-spec K=1..8 batteries must re-pin exactness after. This is the majority of the cost.

(c) **Batched verify kernel path — Effort: MEDIUM.** Today verify is `decode_step_t`
    (`spec.rs:838`): single-sequence, T=K+1 columns, all-column logits, retained hiddens,
    `GdnStash` for rollback. The batched twin is a varlen concat over B sessions
    (B x (K+1) rows) and most of the machinery exists in `prime_cache_batch`
    (`crates/memra-engine/src/hybrid_forward.rs:874`): per-seq pos0 carried positions
    (concat increment (b): `attn_rope_vl`, varlen FA vl twins with T_kv > T) and the varlen
    GDN core (task #18, `hybrid_forward.rs:1488`). The one genuinely missing kernel piece:
    **the varlen GDN K4/K5 chain is fresh-only — zero initial recurrent state assumed**
    (`hybrid_forward.rs:1509`). Verify enters with each session's LIVE conv/ssm state, so the
    varlen scan needs per-seq state-in/state-out (the per-seq scan path already takes initial
    state — the varlen twin needs the same parameter plus per-seq state gather/scatter), and
    `GdnStash`/`VerifyCkpt` become per-seq slices of the concat buffers for rollback.
    This is exactly the "GDN state + fa batching in verify" question from the round recon —
    answer: FA side is already solved by the increment-(b) vl twins; GDN state-in is the
    missing kernel parameter.
    For q27 dense-hybrid there is no MoE expert-dedup bonus (that ~2.5x unique-expert
    verify-4 effect is MoE-only); the win here is pure GEMM m-scaling (m: 1 -> B(K+1)),
    which the measured plain decode-batch scaling on this box already proves out.

(d) **Per-session accept/rollback — Effort: SMALL.** KV rollback is per-session len
    truncation (exists); GDN partial-accept rebuild is a per-seq prefix re-run from the
    stash (exists per-session, stays per-seq — t <= K columns, negligible).

(e) **Draft chains — keep serial initially; batched drafts deferrable (Effort: MEDIUM,
    second-order).** K sequential T=1 MTP-block steps per session; at B=8, K=3 that is 24
    tiny launches per round — launch-bound but ~1/65th of trunk FLOPs. A batched
    `mtp_head_forward` (B rows, per-session scratch via the rowed fa_decode_dc dispatch the
    trunk batch path already uses) is a follow-up, not a prerequisite.

(f) **Draft CUDA graph — Effort: MEDIUM, orthogonal win.** Today the draft graph is
    re-captured every burst (`worker.rs:1174-1176` comment); per-session persistent capture
    (capture once, replay across bursts) removes a per-burst fixed cost that exists TODAY at
    c=1 too.

(g) **Scheduler — Effort: MEDIUM.** Worker phase (a) becomes group round-stepping with
    accept-length divergence per session (already independent), EOS/retire mid-group
    (group shrinks, trivial), and `MEMRA_SPEC_BURST` reinterpreted as rounds-per-tick for
    latency fairness.

Honest total: (b) dominates, (c) is the only new kernel work. By this repo's measured pace
(the concat increments (a)/(b) each took multi-session efforts with full gate batteries),
multi-session spec serving is a **multi-week engine lane**, not a scheduler tweak. There is
no cheap intermediate: any composition that still issues B=1 trunk verifies spends the same
GPU seconds — the flat ~126 tok/s line above IS that ceiling.

## 3b. The per-burst fixed cost: found and killed (2026-08-01, same day)

The ~16ms/burst estimate from §2b was investigated and fixed in two steps — the receipts for
both live in this directory.

**Step 1 — the recapture hypothesis was WRONG (measured).** The worker comment blamed the
per-call draft-graph recapture. Implemented per-session graph persistence (`DraftGraphCtx` on
`SpecSession`: the captured graph(s) + every baked I/O buffer survive across bursts; keyed
sampled-graph invalidation; failed-capture memo). Result: **flat** — spec c1 125.4 -> 126.0
(+0.4%, x3 interleaved, `postfix-points.jsonl`), burst sweep unchanged. The capture was only
~2.2ms and only the FIRST burst of a session pays it now, but it was never the cost.

**Step 2 — profiled, then removed the real cost.** `MEMRA_SPEC_SETUP_TRACE=1` (new diagnostics
in `generate_spec_inner2`) decomposed a continuation burst: **init=11.4ms** (the setup's solo
`decode_step_h` feeding the burst's first token) + **tail=11.6ms** (the session tail's solo
pass committing the pending bonus) — two full trunk passes per burst boundary, each committing
ONE token where the in-loop steady state commits ~3.7 tokens per trunk pass via the batched
verify (`server-setuptrace.log`). Fix: **pending-carry** (`SpecSession::pending_tok`) — the
bonus is stashed instead of committed; the next empty-suffix greedy burst consumes it as
round-0 verify col 0, exactly like a mid-burst full-accept boundary. The burst boundary
becomes a plain round edge (trace after: init=0.01ms, tail=0.02ms, `server-carrytrace.log`).
Non-empty-suffix turns, sampled turns, and pool-parking flush first (`spec_flush_pending`,
one pass per retired request instead of two per burst).

Measured (x3 interleaved rounds, `carry-points.jsonl`, medians):

| point | pre | carry | delta |
|---|---|---|---|
| spec c=1 | 126.3 | **131.8** | +4.4% |
| spec c=8 | 126.4 | **132.2** | +4.6% |
| burst=8 (c4) | 105.6 | **131.0** | +24% |
| burst=32 (c4) | 125.9 | 132.4 | +5.2% |
| burst=128 (c4) | 132.2 | 132.2 | flat |

The burst sweep is now FLAT (131.0-132.6 across 8/32/128) — the fixed cost is gone, and the
fairness knob is free: serve can run burst=8 (4x finer round-robin interleave for multi-session
latency) at ~1% aggregate cost instead of -16%. Default stays 32.

Exactness receipts: `session-gate` ALL TURNS MATCH (incl. the empty-suffix carry-consume turn;
the harness's reference-prefix arithmetic was updated to track history independently instead of
deriving it from `committed` — the oracle no longer reads the system under test's internals),
`run-spec` K=1..8 self-consistency PASS (identical acceptance to pre-fix — single-shot path
untouched), serve outputs byte-equal to the PRE-fix captures on both probe prompts
(`summary-carry.txt`). Acceptance 0.884 vs 0.890 pre (boundary rounds now draft — a
throughput-shape change, not an exactness one).

**Updated crossover** (`carry-points.jsonl` vs same-window plain): c1 = **1.82x**, c2 = 1.19x,
c4 = 0.84x — the fast lane widened but the crossover stays between c=2 and c=4. The remaining
gap to the ~200ms/burst round loop is round-loop work itself (draft chains + verify + accept),
i.e. the §3 batched-verify design — no boundary overhead left to harvest.

## 4. Receipts

- `points.jsonl` — 28 load points (24 matrix + 4 make-up round), each with agg tok/s,
  p50/p95 latency, error counts. `per-request.jsonl` — every request.
- `burst-points.jsonl` / `burst-per-request.jsonl` / `burst-driver.log` /
  `server-spec-burst*.log` — the burst-size sensitivity sweep (`run_burst.sh`).
- `postfix-points.jsonl` + `summary-postfix.txt` (`run_postfix.sh`) — graph-persistence A/B
  (the flat step-1 result); `carry-points.jsonl` + `summary-carry.txt` (`run_carry.sh`) — the
  pending-carry A/B; `server-setuptrace.log` / `server-carrytrace.log` — the [spec-setup]
  boundary decomposition before/after; `session-gate-*.log`, `run-spec-*.log` — engine gates;
  `correctness-specpost-*`/`-speccarry-*`/`-plainpost-*` — serve byte-equality captures;
  `validate-h100-quick-carry.log` — the quick battery on the carry build.
- `summarize.py` — the median table + acceptance aggregation (run on box).
- `server-{plain,spec}-r*.log` — per-arm per-round server logs (spec logs carry per-burst
  `[spec-acc]` acceptance telemetry).
- `correctness-{plain,spec}-p{1,2}.json` — full greedy outputs for the exactness contract.
- `server-plain-8085.log`, `server-spec-8186.log`, `server-plain-8185.log` — the failed
  double-resident launches (OOM receipts for the "one server at a time" protocol).
- `run_matrix.sh` — the exact driver (params baked as literals).
- Thermal/clock regime: all points same box, same hour (07:04-07:24Z), arms interleaved
  per round; single-digit-percent ranges across rounds (see summary output).
