# spec-gate — concurrency-gated spec scheduling

Lane: `lane/spec-gate`, task #89 (raised by `research/spec-scaling-20260806/RESULTS.md`, merged
`fe2b3740`). Box `7490a18d08c39dd0ade2de84bb3a08e019fb1b30`, `NVIDIA GeForce RTX 5090 Laptop GPU,
595.84`, 24463 MiB (`BOX-COMMIT.txt`). Model: `Qwen3.5-9B-NVFP4-MTP-GGUF.gguf` +
`draft-9b-owntrim-nvfp4head-q4blk.gguf` (the accept-gate q9 cell's **production** drafter, attached
via `MEMRA_MODELS "+draft"` so it REPLACES the embedded head — a bare embedded head is a different
acceptance regime). `MEMRA_CTX=4096`, `MEMRA_SPEC_K=3`, greedy, one card, PP door shut.

**VERDICT: SHIPPED, DEFAULT ON.** The gate is the default (`MEMRA_SPEC_GATE=0` is the rollback
seam). It keeps spec's **1.81x** at c=1 and recovers batched decode's scaling above the crossover —
**2.03x** over always-spec at c=8 — and it is what makes the mixed tick survivable: demotion pulls
per-stream p95 from **11.243s to 7.249s** in the worst-case mixed state. Exactness holds
byte-for-byte. One in-scope mitigation from the brief is **REFUTED by measurement**: bounding the
burst quantum (`MEMRA_SPEC_BURST=4`) does not recover the mixed-tick penalty and costs throughput.

---

## 1. The policy, as shipped

Two thresholds, both read from the measured ladder, plus one-way demotion.

```
admit spec   while active + 1 <= MEMRA_SPEC_GATE_LOW   (default 2 — the last measured WIN rung)
demote live  when active      >= MEMRA_SPEC_GATE_HIGH  (default 4 — the first measured LOSS rung)
```

`act == 3` is the hysteresis band: a session already on spec keeps bursting, a new arrival is
admitted batched. Nothing switches mode inside the band, so mode changes track load crossings
rather than ticks (§5). `MEMRA_SPEC_GATE_HIGH <= LOW` would leave no band at all and is clamped to
`LOW + 1` with a loud warning — a misconfiguration that reintroduces thrash should not be silent.

Two code seams, both in `crates/memra-server/src/worker.rs`:

- **admit** — `spec_eligible` gains `n_active + 1 <= spec_gate_low()`. New arrivals above the
  ceiling never join the serial spec queue.
- **phase (a-) demotion**, inserted before the spec-burst loop — a live spec session hands its
  cache and next-token prediction to the plain batched path.

The admit gate alone is **not** sufficient, and that is the whole reason the demotion exists: a
session admitted while the box was quiet keeps its whole-burst hold on the tick after load arrives,
and §4 measures exactly what that costs the batched rows waiting behind it.

### 1.1 Why the handoff is exact (greedy)

At a burst boundary `SpecSession` satisfies `cache.pos == committed.len()`: every committed row's
trunk KV and recurrent state is exactly what a plain prime of that token sequence would have
produced. `next_pred` is the argmax of the verify's logits for the last committed row, which is
bit-identical to plain decode's logits at that position — that identity IS the greedy accept walk's
basis, so it is not a new assumption this lane introduces. `into_demoted` hands `(cache, next_pred)`
over and `Session.device_next` makes the next batched tick emit exactly that token into that same
cache, which is what `advance_sample_emit` does for any batched row. The structural precedent is the
graph-session DEGRADE path, which already performs this same `s.cache = Some(...)` + pending handoff
when concurrency arrives.

A carried pending bonus (`pending_tok`) must **commit first**: its row is emitted to the client but
deliberately absent from the caches, so handing over a cache one row short of the emitted stream
would silently drop a token. `spec_flush_pending` is that commit — one T=1 trunk pass, once per
demotion, never per burst.

**Excluded, and why (stated, not hidden):**

- **Sampled** sessions — the sampled tail's `next_pred` is the commit pass's *argmax*, so handing it
  over would inject a greedy token into a sampled stream. The tail keeps no logits row to draw from,
  and adding a per-burst `[n_vocab]` D2H (1.36 ms at the 9B's 248k vocab) to enable a rare handoff
  is the wrong trade.
- **Constrained** sessions — `next_pred` is the *unmasked* verify argmax and could be
  grammar-illegal.

Both residuals are bounded by the admit gate: at most `spec_gate_low()` sessions can ever hold the
spec path, so the worst case is that many serial bursts, not a full ladder's worth.

### 1.2 Re-promotion: NOT in v1, and the reason

Demotion is **one-way per session**. Re-promoting on drain-down would need an `mtp_kv_fill` over the
whole committed history plus a fresh draft-graph capture — not the "symmetric and cheap" handoff
that option was conditioned on. A demoted session stays demoted until it ends. The policy still
tracks a draining load because **new arrivals** get spec again as soon as the count falls back to
`LOW`, i.e. the policy is per-REQUEST, not per-session. This also makes the thrash bound structural
(§5): a session can switch mode at most once in its life.

## 2. EXACTNESS — PASS

`exactness.py`, 5 arms, one server boot each, greedy, 768-token budget.

| property | verdict |
|---|---|
| **demoted mid-generation == batched from the start** (B=1 both sides) | **PASS — byte-identical, 3122 bytes / 768 tokens** |
| forced demotion fired on the target | YES, at `generated=129` of 768 |
| control: spec solo vs batched solo | PASS (shared prefix byte-identical, 0 overshoot) |
| accept-gate pinned cells, spec active at c=1 | **6/6 PASS**, acceptance counts integer-identical, all text shas identical |
| `run-spec` K=1..8 self-consistency | **PASS** all K, acceptance 88.2% → 26.2% |
| serve-smoke (incl. "spec == plain greedy text") | **0 failed** |
| serve-stress c=64 | **ALL GREEN**, 64/64, wall p50 22.8s, ttfb p50 0.19s |
| apikeys-gate (scratch OUT dir) | **0 failed / 18 gates** |

Reproduced on the final binary: `PRIMARY_demote_solo_vs_ref_solo: PASS (byte-identical)`, same
forced point.

### 2.1 Three harness traps caught before they returned a false green

Recorded because each one *did* produce a green light that meant nothing, and a later reader
must not reintroduce them.

1. **VACUOUS PASS.** q9 is a thinking model: every generated token lands in `message.reasoning`
   and `content` is empty. v1 compared `content` — three arms, 0 bytes each, all "PASS", on a
   stream it never read. Fixed: compare both fields, and hard-fail an arm whose stream is
   near-empty.
2. **WRONG SESSION.** v2 fired load 2.0s after a 384-token target that finishes in ~1.5s solo, so
   `act` fell to 0 and a background filler took the spec slot — the demote line read
   `generated 1`. Fixed: 768-token budget, 0.5s delay, and the verdict now requires the demote line
   to prove it fired on the *target*.
3. **THE WRONG REFERENCE — with a real pre-existing finding underneath.** v3 diverged at byte 681,
   ~45 tokens *after* the handoff. A discriminator arm (`REF_LOAD`: spec OFF, gate absent, same
   background load) diverges from solo `REF` too — see §2.2.

### 2.2 Pre-existing finding: batch-vs-solo decode is NOT bit-identical

With spec OFF and none of this lane's code involved, the same greedy request diverges between a
solo run and one sharing its batched decode with concurrent rows. Two independent runs put the
first divergence at byte **2379** and byte **1347** — the byte moving between runs is itself the
proof that the loaded configuration is nondeterministic.

This is expected from the engine's own documented laws rather than a new bug:
`fa_decode_batch_seqs_v4` carries a single `split_keys` for sessions at different depths (the
LADDER-RUNG STRADDLE law that `fa_decode_rows` documents for the row axis), and the batched-linear
tier selection changes with B.

**The consequence for this lane is methodological and it is the reason the test looks the way it
does:** load-triggered demotion can *never* be a clean exactness test, because both the arrival
timing and the batch composition are nondeterministic. So the handoff is tested at **fixed batch
shape** via a diagnostics-only door, `MEMRA_SPEC_DEMOTE_AT=N`, which forces the demotion at a
pinned generated-token count with no load at all, holding B=1 across the boundary. The only
difference from a plain batched run is then that the first N tokens came off the spec path — exactly
the property under test. Never set in production.

(Recorded as a finding, not fixed here: it is a pre-existing property of the batched decode path,
outside this lane's scope, and it does not affect the gate's verdict.)

## 3. MEASUREMENT 1 — the c-ladder

`run-cladder.sh`. N=5 rep-major, arm order rotating by rep so no arm sits at a fixed point of a
thermal drift; 60 load points, **0 errors, 0 shed**; per-rep spread 0.7-2.9%. GPU 57C/9.95W pre,
75C/21.9W post — warm, no thermal cliff. Medians of N=5. TTFT requires `--stream` (§3.2).

### 3.1 Aggregate throughput

| c | gated | never-spec | always-spec | gated vs best fixed arm |
|---|---|---|---|---|
| 1 | **251.2** | 138.5 | 251.9 | 1.00x — tracks spec (**1.81x** over batched) |
| 2 | **250.0** | 221.5 | 250.8 | 1.00x — tracks spec (1.13x over batched) |
| 4 | **357.4** | 383.5 | 249.7 | 0.93x — tracks batched (**1.43x** over spec) |
| 8 | **504.7** | 521.2 | 248.4 | 0.97x — tracks batched (**2.03x** over spec) |

The gate follows the winning arm at every rung instead of picking one and eating the other's loss.
Per-stream p50 says the same thing from the latency side: 0.514s at c=1 (spec's number, vs batched
0.923s) and 1.963s at c=8 — **exactly** batched's 1.963s, vs always-spec's 3.973s.

The 3-7% given up to never-spec at c=4-8 is the first wave, not a steady-state tax (§3.3).

### 3.2 TTFT

| c | gated p50 | never-spec p50 | always-spec p50 | gated p95 | never-spec p95 | always-spec p95 |
|---|---|---|---|---|---|---|
| 1 | 0.066 | 0.003 | 0.066 | 0.066 | 0.003 | 0.066 |
| 2 | 0.077 | 0.009 | 0.077 | 0.221 | 0.011 | 0.221 |
| 4 | **0.012** | 0.011 | 0.078 | 0.423 | 0.017 | 0.525 |
| 8 | **0.016** | 0.015 | 0.079 | 0.324 | 0.019 | 1.000 |

At c=4/8 gated p50 **matches never-spec** and beats always-spec 6.5x / 4.9x. The p95 is the honest
cost and it is not hidden: 0.423s vs never-spec's 0.017s at c=4.

TTFT is only observable in streaming mode. `tools/load-serve.py` posted with `stream: False`, whose
only client-side timestamp is the whole response — a policy that delays the first token but not the
last would have measured as neutral. `--stream` (added by this lane, default off so other lanes'
numbers stay comparable) timestamps the first *content-bearing* SSE frame; role-only openers and
empty deltas are protocol overhead, and `reasoning` counts because on a thinking model that is the
visible stream.

### 3.3 The p95 is the FIRST WAVE, and the per-request rows prove it

Not inferred from the aggregate — read off arrival order. Gated, c=8, rep 1, by `req_index`:

```
 0: 0.221   1: 0.068   2: 0.315   3: 0.315   4: 0.315   5: 0.315   6: 0.315   7: 0.314
 8: 0.015   9: 0.015  10: 0.015  11: 0.015  12: 0.014  13: 0.014  14: 0.015  15: 0.015
16: 0.016  ...  all 24 remaining requests 0.014-0.017
```

The elevated TTFT is confined to `req_index` 0-7 — the requests admitted while the box was quiet,
which by design hold the spec path until the high-water mark. Pooled across all 5 reps, exactly
**25%** of requests exceed 50 ms at c=4 and at c=8, which is precisely the harness's first-wave
fraction (4 of 16, 8 of 32). The baseline's own first wave is 0.057-0.080s, so prefill is part of
this too.

So the residual is a **transient bounded by `LOW=2` sessions per load ramp**, not a steady-state
tax.

### 3.4 The gate's own observables

Demotions: **exactly 4 per rep in all 5 reps** (two spec sessions at c=4 + two at c=8 — `LOW=2`
caps how many can ever hold the spec path), and **0** in both fixed arms. Tick shape, gated rep 1:
1071 ticks, 41 with a spec session present, **2 genuinely MIXED** (`spec>0 AND ready>0`) — the mixed
state is rare under the default precisely because demotion resolves it in one tick.

## 4. MEASUREMENT 2 — the mixed tick (the ship-blocker), and a REFUTATION

The predecessor lane refused to ship a gated policy for this reason, quoted:

> "Phase (a) runs whole bursts before phase (c) is reached, so 2 spec sessions holding ~21 ms of
> serial burst per tick would inflate the batched rows' TTFT and inter-token latency. That
> interaction has no receipt in this lane and a policy shipped without it would be a latency
> regression dressed as a throughput fix."

`run-mixedtick.sh`, N=5 rep-major rotating order, **c=6** (above `HIGH`, so the gate is fully
engaged), 512-token generations, 0 errors, spread 0.8-2.8%. GPU 57C pre / 74C post.

| arm | agg tok/s | TTFT p50 | TTFT p95 | stream p50 | **stream p95** | MIXED ticks |
|---|---|---|---|---|---|---|
| B — no spec at all | 456.1 | 0.013 | 0.079 | 6.728 | 6.767 | 0 / 2586 |
| **G — gated (SHIPPED)** | **445.6** | 0.014 | 0.518 | 6.753 | **7.249** | 1 / 1540 |
| M — mixed, demotion disabled (`HIGH=99`) | 393.9 | 0.014 | 0.517 | 6.752 | **11.243** | 17 / 1540 |
| Q — mixed + `MEMRA_SPEC_BURST=4` | 390.4 | 0.014 | 0.365 | 6.839 | 11.075 | 109 / 1543 |

**The starvation is real, and visible per request.** Arm M, rep 1, by arrival — the first-wave
batched rows take **11.08s** while later ones take 6.6s, stuck behind whole serial spec bursts every
tick for the whole run because demotion never fires:

```
lat: 4.67 11.08 4.59 11.08 11.08 11.07 | 6.70 6.63 6.66 6.66 6.66 6.66 6.66 6.66 | 5.32 5.32 5.32 5.32
```

**The demotion is the fix.** Gated pulls those same rows back to 7.22s: stream p95 **11.243 →
7.249s, a 3.99s recovery**, with **+13.1%** throughput (393.9 → 445.6), reaching 97.7% of the
no-spec ceiling while still taking spec's 1.81x at c=1. It resolves the mixed state in **one tick**
(1 mixed tick of 1540 vs M's 17).

### 4.1 REFUTED: burst-quantum bounding

The brief put `MEMRA_SPEC_BURST` in scope *if* the burst starves the batched tick. It does starve —
and bounding it still does not fix it. `MEMRA_SPEC_BURST=4` leaves stream p95 at **11.075s** (vs
M's 11.243s, inside spread) and costs **1.4%** throughput (390.4 vs 393.9).

The tick shape names the reason: it converts 17 mixed ticks into **109**, spreading the *same total
serial verify work* across more ticks instead of removing it. **The serial work is the cost, not its
granularity.** Per flags doctrine, no flag ships from this arm; this row is the record.

## 5. MEASUREMENT 3 — thrash: PASS

`run-thrash.py`. 6 cycles of c=2 ↔ c=6 across the hysteresis band, 6s phases, 1600 ticks. Bar
stated in the harness docstring *before* the run.

| observable | value |
|---|---|
| load crossings | 12 |
| ticks | 1600 |
| ticks with a spec session | 62 |
| **demotions** | **12 — exactly one per crossing** |
| demotions per tick | **0.0075** |
| admit-batched events | 24 |
| verdict | **PASS — O(load changes), not O(ticks)** |

`demote_at_generated` = `[1,1,1,33,1,1,1,33,1,33,1,33]` — demotions land at the first burst boundary
after the crossing, which is the intended latency (a session mid-burst finishes its quantum).

The bound is structural, not lucky: one-way demotion means a session switches mode **at most once in
its life**, so the total is bounded by sessions admitted-spec that then met a high-water mark. The
per-tick flap this test exists to rule out cannot occur by construction; the run confirms the
implementation matches the design.

## 5.5 The pre-push battery, and the one red it raised (settled)

`tools/local-ci.sh --perf` is the pre-push gate for any engine-touching commit (this lane touched
`crates/memra-engine/src/spec.rs`). Correctness stage **fully green**: kernel-check GREEN,
prime-gate MATCH=8/8, run-gen argmax MATCH (31B + 12B depth), VERIFY-GATE K=7 PASS both, spec
self-consistency 64/64, decode-batch-gate config+strict ALL GREEN on both 9B encodings,
graph-warmup-stress GREEN, sampler-gate, serve-smoke 0 failed, serve-stress 64/64, accept-gate
1/1. Perf stage: **9 cells, 8 OK, 1 FAIL** — `31b-plain-d1736` at 38.02 tok/s, -3.03% against a
rolling median of 39.21.

**Settled the way the script itself prescribes, and it is NOT a regression.** Interleaved A/B/A/B,
N=5 each, one thermal window, one exclusive `flock` hold for the whole run
(`perf-ab-31b-plain-d1736.sh`, receipts `logs/perf-ab/`):

| arm | reps (tok/s) | median |
|---|---|---|
| A — merge-base `9e228f4c` | 39.04 38.31 38.23 38.22 37.55 | **38.23** |
| B — lane tip `faba56cf` | 38.35 38.27 38.08 37.89 37.64 | **38.08** |

**B vs A: -0.39%** — inside noise, and the merge-base does not reproduce 39.21 either. The rolling
median is the invalid side: its rows are from 2026-07-30..08-06 (a cross-day comparison, which this
project's measurement law forbids as evidence, denominator included), that series itself spans
39.2x on 08-03/04/05 and 35.8x twice on 08-06, and the failing row is `window_clean:false` by its
own admission. Scope agrees with the measurement: the cell is gemma-4-31B **plain** greedy decode
through `run-gen`, a `memra-engine` binary; this lane's engine diff is +50 lines of NEW
`impl SpecSession` methods (0 existing lines touched) and `memra-engine` does not depend on
`memra-server`, so `run-gen` never constructs a `SpecSession` at all.

### 5.5.1 A gate bug found and fixed while running it

The perf stage **hung**, and the first battery produced no rows. `run_cell`'s dirty-window retry
was `while ! window_free_now; do sleep 40; done` — unbounded. The co-resident here is the owner's
`hermes-gateway.service`, holding a 394 MiB idle CUDA context 24/7 at 0% GPU util; it is not a
lane's job to kill and it never leaves, so that loop could not exit and the script's own honest
fallback two lines below it (`DIRTY twice — recording with window_clean=false`) was unreachable.
Fixed in `tools/local-ci.sh`: the wait is bounded (`MEMRA_CI_DIRTY_WAIT`, default 600s) and
**latched** — once one cell proves the co-resident outlasts the wait, later cells skip straight to
the honestly-labeled retry instead of re-paying it, because 10 cells x 600s of sleeping is a hang
with progress output, not a gate. A gate that hangs forever is worse than one that records a row
labeled `window_clean=false`.

The A/B harness had its own two bugs, fixed before its numbers were used: the loop body was passed
to `bash -c` via a `tr`-collapsed `declare -f` (`syntax error near unexpected token 'done'`) and the
run still **exited 0** because the failure was inside a pipeline. Now an exported function plus a
`PIPESTATUS` check — a settle harness that can report success while measuring nothing is worse than
no harness.

## 6. Flags

Winners are defaults — the gate needs no flag to be on.

| env | default | purpose |
|---|---|---|
| `MEMRA_SPEC_GATE=0` | on | rollback seam — restores pre-lane always-spec at every concurrency |
| `MEMRA_SPEC_GATE_LOW` | 2 | admit-spec ceiling (last measured win rung) |
| `MEMRA_SPEC_GATE_HIGH` | 4 | demote floor (first measured loss rung); clamped to `LOW+1` if lower, loudly |
| `MEMRA_SPEC_DEMOTE_AT` | unset | **diagnostics only** — force demotion at a pinned token count, B=1, for the exactness test. Never set in production. |

All four are in `docs/FLAGS.md` (§1 Server for the policy + thresholds, §3 for the rollback seam,
§4 for the test door). The `MEMRA_SPEC_BURST` entry there also carries §4.1's refutation, so a later
reader reaching for that knob to fix a mixed-tick latency problem finds the measurement first.

## 7. Honest disclosure

- **`ttft_p95` at c=4 is worse than either fixed arm** (gated 0.423s; never-spec 0.017s;
  always-spec 0.525s). Located, not hand-waved: the first wave only (§3.3), bounded by `LOW`
  sessions per load ramp, with p50 identical to never-spec. A deployment that cares about
  cold-ramp p95 more than about c=1 throughput should set `MEMRA_SPEC_GATE_LOW=0` (never admit
  spec) rather than get a different policy silently.
- **Batch-vs-solo decode is not bit-identical** (§2.2). Pre-existing, reproduced with this lane's
  code absent, recorded here because it is what forced the deterministic test door — not fixed by
  this lane.
- **Sampled and constrained spec sessions do not demote** (§1.1). They keep the serial path until
  they end, bounded by the admit gate.
- **All numbers are single-card 5090 Laptop**, greedy, q9 + production drafter, `MEMRA_CTX=4096`,
  K=3. The mixed-tick arm is a single concurrency (c=6); the ladder covers c=1/2/4/8. No
  cross-day comparison appears anywhere in this report — every table is medians of same-rep
  arm comparisons.
- The `MEMRA_SPEC_DEMOTE_AT` door adds a `OnceLock` env read and a `generated.len()` scan on the
  demotion path, which is already gated behind `spec_gate_on() || demote_at.is_some()`.

## 8. Files

| path | what |
|---|---|
| `exactness.py` | the 5-arm exactness harness (3 traps recorded in its docstring) |
| `run-cladder.sh` | measurement 1 — 3 arms x c=1,2,4,8, N=5 rotating order |
| `run-mixedtick.sh` | measurement 2 — baseline / mixed / mixed+bounded / gated at c=6 |
| `run-thrash.py` | measurement 3 — oscillating load across the hysteresis band |
| `analyze.py` | medians per (arm, c) + per-rep spread + error accounting |
| `perf-ab-31b-plain-d1736.sh` | §5.5's settle — merge-base vs lane tip, interleaved N=5, one lock hold |
| `logs/perf-ab/` | the settle's per-rep logs, `A.toks`/`B.toks`, `interleave.txt`, `VERDICT.txt` |
| `logs/local-ci-perf.txt` | the full pre-push battery run (correctness + 9 perf cells) |
| `logs/exact/` | 5 arms: per-arm server logs, full streams, `exactness.json` |
| `logs/cladder/` | `points.jsonl` (60), `per-request.jsonl`, per-arm-per-rep server logs, tick traces, demote counts, `/metrics`, `gpu-{pre,post}.csv` |
| `logs/mixedtick/` | same shape + `*-tickshape.txt` (MIXED tick counts) |
| `logs/thrash/` | `thrash.json` + server log |
| `logs/{accept-gate,run-spec,serve-smoke,serve-stress,apikeys-gate}.txt` | gate battery receipts |
| `logs/{cladder,mixedtick}-tables.md` | generated tables |
| `BOX-COMMIT.txt` | box + driver + engine commit at measurement |

## 9. Answer to task #89, in one paragraph

Spec wins at low concurrency and loses at high because the spec path is a serial queue with no
batched-verify entry point (refuted at a 16-column exact-kernel width ceiling, `fe2b3740`), so the
shippable answer is policy: admit spec only while `active+1 <= 2`, and demote live spec sessions to
the batched phase once `active >= 4`, with `act==3` as a hysteresis band and demotion one-way per
session. The demotion is a real cache handoff — `(cache, next_pred)` into `Session.cache` +
`device_next`, with a carried pending flushed first — and it is byte-exact for greedy: a session
demoted mid-generation emits a stream **byte-identical** to one batched from the start (768 tokens,
forced demotion at 129, 3122 bytes identical), with accept-gate 6/6, run-spec K=1..8, serve-smoke,
serve-stress c=64 and apikeys-gate all green. The gated arm tracks always-spec at c=1-2 (251.2
tok/s, 1.81x over batched at c=1) and never-spec at c=4-8 (504.7 tok/s, 2.03x over always-spec at
c=8), and per-stream p50 at c=8 is batched's 1.963s exactly rather than spec's 3.973s. The mixed
tick the predecessor lane flagged as a ship-blocker is real — an unmitigated mixed state costs
first-wave batched rows 11.08s against 6.6s — and the demotion is what closes it, recovering 3.99s
of stream p95 (11.243 → 7.249s) and 13.1% throughput; bounding the burst quantum, the other
candidate mitigation, is **refuted** (p95 11.075s, -1.4% throughput: it multiplies mixed ticks 17 →
109 without removing the serial work). Thrash is bounded structurally and measured at exactly one
demotion per load crossing over 1600 ticks (0.0075/tick). The gate is the **default**, with
`MEMRA_SPEC_GATE=0` as the rollback seam; the one residual is a first-wave TTFT p95 transient
(0.423s vs 0.017s at c=4), confined to the at-most-`LOW` sessions admitted before the ramp.
