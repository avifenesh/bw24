# lane/chunkinv-flip — grain-free chunk-invariance fix: VERDICT

**Date:** 2026-08-05. **Box:** local 5090 laptop, flock'd (`/tmp/gpu5090.lock`, 3 other
lanes sharing). **Branch:** `lane/chunkinv-flip` (worktree `wt-chunkflip`), base
`e207c182` (the lane/chunk-invariance merge that filed this fix).

## The fix (commit c8f58504 + 9e411769)

`full_attn_prime_fa_dispatch` (`crates/memra-engine/src/hybrid_forward.rs`) DROPS the
`base_len == 0` special case. Previously chunk 0 attended this batch's **f32** K/V via
`fa_prefill` while every later chunk attended the **q8_0/q5_1 quantized KV cache** via
`fa_prefill_view_ws` — so `MEMRA_PRIME_CHUNK` decided WHERE that precision-class edge fell
and two rigs with different values produced different greedy text
(`research/chunk-invariance-20260805/VERDICT.md`). Now: `full_attn_prime_pre_fa` already
appends the chunk's quantized K/V rows into the cache BEFORE dispatch, so chunk 0 attends
through the quantized cache exactly like later chunks (**quantize-then-attend**). One
numeric class for every row ⇒ split points cannot move a class edge ⇒ chunked prefill is
reduction-order-stable with **no door and no grain knob**.

Rollback seam: `MEMRA_PRIME_F32CHUNK0=1` restores the legacy f32-chunk-0 arithmetic
(flags doctrine: winner is default, env is the rollback door). It doubles as the chunkinv
gate's canary injection. `MEMRA_NOFA`/unstamped head_dim falls to
`sdpa_naive_quantized_view` — same cache bytes, same class, so the uniform contract holds
on the fallback too.

## First result (BINDING first-60-minutes cell) — PASS

`tools/chunk-invariance-gate.sh` with NO door env, one flock hold, 2026-08-05 ~19:47+03:00
(receipt `first-result.log` + `first-result-raw-probe.log`, commit c8f58504):
both pinned prompts (97-tok turn1, 149-tok turn2) **CHUNK-INVARIANT — prefill logits
bit-identical at every chunk size** (chunks 2048/64/32, `first_div_pos=-1`,
`maxdiff=0.000e0`, streams identical). The then-current `--expect-variant` assertion broke
exactly as its own "invariance won back" arm prescribes → gate default flipped to
`--expect-invariant` (naked env), canary re-armed on the legacy seam.

## Gate flip battery (one lock hold, 21:23+03:00; `logs/gate-flip.log` + raw logs)

| arm | env | verdict | result |
|---|---|---|---|
| default `--expect-invariant` | none | CHUNK-INVARIANT both prompts | **PASS** |
| canary (injects `MEMRA_PRIME_F32CHUNK0=1`) | seam on | CHUNK-DEPENDENT (maxdiff 3.9e-1..6.4e-1, `first_div_pos` == chunk size exactly, streams step @16/18/47) | **PASS (teeth proven)** |
| legacy `--expect-variant` | seam on | pinned divergence reproduces on both prompts | **PASS (rollback exact)** |

## Full evaluation (battery `run-eval.sh`, 21:26–21:36+03:00; all cells exit 0)

**Exactness** (`logs/A-*.log`): run-gen argmax — 9B NVFP4 `prefill==decode` MATCH +
`batched-prime==tokenwise` MATCH; 27B NVFP4 both gates MATCH; 9B ST modelopt dir
verify-prefill MATCH `maxdiff=0.000e0`. kernel-check ALL GREEN (`E-kernel-check.log`,
includes the fa_prefill_view_ws bitdiff=0 pins). run-spec K=1..8 SELF-CONSISTENCY PASS
(9B+MTP, `E-runspec-9b.log`). serve-smoke **0 failed** (`E-serve-smoke.log`, includes
session-affinity resume determinism).

**Quality — NLL, 1024-token frozen mmq-v2 window through the SERVING PRIME** (the pass the
fix changes; instrument: `concat-prime-probe nllwin`, lm_head over `prime_cache`'s own
hidden stack):

| model | arm | mean_nll | ppl |
|---|---|---|---|
| 9B NVFP4 | grain-free (default) | 0.910005 | 2.484336 |
| 9B NVFP4 | legacy f32-chunk0 | 0.910269 | 2.484990 |
| 9B NVFP4, chunk 256 | grain-free | 0.910005 | 2.484336 |
| 9B NVFP4, chunk 256 | legacy | 0.910836 | 2.486401 |
| 27B NVFP4 | grain-free | **0.840739** | **2.318080** |
| 27B NVFP4 | legacy | 0.850404 | 2.340591 |

The fix is quality-FREE on the 9B (never worse; chunked-arm legacy is the worst cell) and
**IMPROVES 27B NLL by 1.1%**. Note the grain-free 9B number is identical at chunk 4096 and
chunk 256 — the invariance property showing up in the quality instrument itself.

**Contract change, quantified** (teacher-forced argmax new-vs-legacy, mmq-v2 protocol;
`concat-prime-probe tfcmp`): 9B **11/1024** disagreements, all near-tie (largest legacy
margin 0.063x the legacy median, 7.1th percentile; chunk-256 arm identical 11 flips). 27B
**16/1024**, 13/16 below 0.1x median; largest 0.66x median (35.7th pctile). This is the
documented near-tie class — reported as the contract change, not a failure.

**Perf** (`ppprime` = timed `prime_cache`, fresh cache/rep, median of 3 in-process reps,
arms interleaved N=5; `logs/D-perf.jsonl`): pp512-class (400 tok) median 4192.5 tok/s
grain-free vs 4188.4 legacy (+0.10%); pp6257-class (4881 tok) 4730.1 vs 4730.0 (+0.00%).
Mechanism perf-free; no cliff, slight positive lean. Thermal: shared-rig steady state,
same-hold interleave (cross-run comparisons invalid per repo law; these are same-hold).

**Goldens:** q9 20-token stream byte-identical pre/post flip; k27 re-pinned at
battery-green 869e0fcc (diverged at token 6 = the near-tie chain above). q35/q35slru/o35/
g12/q35spec goldens unaffected (fast-gate tier-1 vs main: ALL PASS, incl. flipped
chunkinv + chunkinvc).

## Decision

**Grain-free is the DEFAULT.** No flag needed for the tuned+invariant path (naked = full
speed + byte-stable across `MEMRA_PRIME_CHUNK`).

- `MEMRA_PRIME_F32CHUNK0` = the rollback seam (docs/FLAGS.md).
- `MEMRA_PRIME_INVARIANT` + `MEMRA_PRIME_GRAIN` = SUPERSEDED (redundant on the fixed
  arithmetic; kill-candidates once the flip ships a release — flags doctrine).
- `MEMRA_PRIME_CHUNK` returns to a pure memory/transient knob; the long-ctx OOM behavior
  is UNCHANGED (chunk transients still scale with chunk size — the door's owed 27B/32k
  footprint gate was a door-default question and the door is now moot).
- Track 2 (door-on/off OOM gates) NOT taken: first result PASSed and the full evaluation
  stayed green, so the door never became the shipping mechanism.

Docs updated: docs/FLAGS.md §prime rows, docs/SERVING.md chunk-stability section
(byte-equality assertions now allowed and gated), fast-gate models.tsv registry comment.

## Receipts

- `first-result.log`, `first-result-raw-probe.log` — the binding first cell.
- `logs/gate-flip.log`, `logs/gate-{default,canary,legacy}-raw.log` — flipped gate battery.
- `logs/A-*` `B-*` `C-*` `D-*` `E-*` + `logs/D-perf.jsonl`, `logs/run-eval.out` — full eval.
- `logs/fast-gate-tier1.log` — mapped tier-1 vs main after golden refresh.
- `RESULTS.jsonl` — machine-readable summary rows.

Commits: c8f58504 (fix + first result) → 9e411769 (seam + gate flip + instruments) →
b82e7c30 (tfcmp + battery script) → bea89fa2 (docs) → 11caa85b (gate receipts) →
869e0fcc (full eval receipts) → 379b829e (golden refresh).
