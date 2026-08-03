# MoE-SpeQ oracle ceiling on Hy3 spill — miss-hiding fraction from drafter lookahead

**Lane deliverable: ONE number per (class, K) — what fraction of expert-cache misses during
real greedy decode would have been hidden if the experts predicted by the MTP spec drafter
K tokens ahead had been prefetched. Gate: >30% = lever proceeds; <30% = lever dies.**

## Verdict

**PROCEED.** The strict ceiling crosses the 30% gate at K=1 on all three serving classes
(chat-prose 30.8%, code-gen 42.4%, summarize 33.6%) and rises with K (37.6-52.1% @K=2,
39.9-56.1% @K=4); the partial ceiling (counting rejected-draft columns, which a real
prefetcher also issues) is 42.9-67.6% on the serving classes and 33.8-50.9% even on the
adversarial case. d1736 — the known low-acceptance depth prompt (7.6% acceptance this run,
matching the K-sweep's 8.5%) — is the strict floor at 7.0%: when the drafter can't predict
TOKENS it can't predict experts either, so the lever's value tracks acceptance, exactly like
spec decode itself. Its partial ceiling still clears 30% from K=2 up because even
rejected-token routing overlaps the true expert set.

## The table (miss-hiding fraction, % of all missed expert-dispatches in 128-token decode)

STRICT = misses covered by an ACCEPTED verify column that predicted the missed expert for that
(position, layer) — routing there is bit-identical to plain decode, so a prefetch issued at
draft time is provably the right bytes. PARTIAL = same, crediting every verify column
(accepted or rejected) that targeted the position — the real prefetcher issues these too;
wrong-path columns still prefetch the right experts when routing overlaps.

| class | prompt tok | miss rate (plain) | K=1 strict | K=1 partial | K=2 strict | K=2 partial | K=4 strict | K=4 partial |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chat-prose-medium | 38 | 50.7% (41022/80896) | **30.8%** | 42.9% | **37.6%** | 54.3% | 40.2% | 62.2% |
| code-gen-short | 42 | 48.1% (38896/80896) | **42.4%** | 46.6% | **52.1%** | 58.6% | **56.1%** | 66.0% |
| summarize-medium | 2096 | 87.5% (70773/80896) | **33.6%** | 45.3% | **37.5%** | 57.9% | **39.9%** | 67.6% |
| d1736 (depth, RAW) | 1736 | 85.6% (69284/80896) | 7.0% | **33.8%** | 7.0% | **42.5%** | 7.0% | **50.9%** |

Strict hits by lookahead depth (how far ahead the prefetch fires): chat-prose K=4 =
{1: 11091, 2: 4184, 3: 1202}; code-gen K=4 = {1: 10830, 2: 7099, 3: 2691, 4: 1194};
summarize K=4 = {1: 20464, 2: 5531, 3: 1659, 4: 554}. Depth 1 dominates but depth >= 2
contributes 27-50% of hidden misses at K=4 — lookahead beyond the next token is real
signal, not noise.

## Regime

- Box: darklanes-bench 8x H100 80GB (Ohio, 13.58.112.171), devices 0-3 claimed per
  `~/receipts/gpu-assignment.txt`, one process per device, GPUs otherwise idle in-bracket,
  temps 34-46 C, all runs 2026-08-03 00:32Z-06:0xZ same-night regime.
- Artifact: `/opt/dlami/nvme/models/hy3-layer103p5-bw24-runtime` (bw24-expert-overlay-v2,
  103.5 arm, bytes untouched). Boot required restoring the expected sparse-source path:
  `/data/ai-ml/hf-models/hy3-layer103p5-sparse-source -> /opt/dlami/nvme/models/hy3-sparse-source`
  (config.json sha256 663036ce… matches the manifest pin).
- Binary: box M2-lane tree `~/memra` (binaries Aug 2 19:10) + a 21-line diagnostic patch
  (`tools/miss-trace.patch`): `MEMRA_MOE_MISS_TRACE=<path>` logs one `"<layer> <proj>
  <expert> <H|M>"` line per SLRU `dispatch_source` lookup (hit, pending-promote = miss,
  first-miss admit = miss). Rebuilt `-p memra-engine` only, in a clone tree
  (`~/memra-moespeq`); the M2 lane's tree and binaries untouched.
- Spill regime on this box: 80GB HBM holds the trunk + a 6568-slot SLRU expert cache;
  the expert bank overflows it and misses stage H2D from host RAM (mmap, 0.00 GB physical
  reads — page cache warm). Miss rate is 48-54% of expert dispatches at NGEN=128
  (~1.7-3.9 GB/token H2D). This is a REAL spill regime (mid-tier of the
  local-5090 NVMe ladder: same SLRU policy, same dispatch path, faster backing store).

## Method (one process = one measurement, no cross-run pairing)

One patched `run-spec` process per (class, K) emits three lockstep artifacts from the SAME
model instance: the plain greedy oracle decode (128 tok) with per-lookup H/M trace +
per-forward route trace, then the spec phase (same prompt, MEMRA_SPEC_K=K,
MEMRA_DEBUG_SPEC per-round accept log, self-consistency gate asserting the spec phase
commits EXACTLY the oracle's tokens).

- Denominator: missed expert-dispatches in the oracle decode sweeps ((layer, expert) with
  any of its 3 projection lookups missing; block-level misses are 3x, same ratios). Note
  the run-spec oracle decodes against a colder cache than standalone run-gen (which runs a
  tokenwise verify-prefill first): standalone chat-prose block-miss is 35.7% vs 50.7%
  in-process. The ceiling is a fraction of the SAME process's own misses, so this shifts
  the denominator's absolute size, not the fraction's validity; a warmer cache removes
  re-usable misses first, which if anything raises the drafter-hideable share.
- Numerator: verify batch bi pairs with debug round bi+shift; column j >= 1 is the
  drafter's lookahead-j routing for generated position out_len-1+j+delta. shift/delta are
  resolved empirically per run by maximizing accepted-column routing equality with the
  oracle — the resolved cell is EXACT on every run (e.g. chatprose-k1 3081/3081,
  codegen-k4 5688/5688 layer-routings identical), which is the self-consistency
  contract observed end-to-end and the validity gate for the whole pairing.
- Lockstep gate: route and miss traces pair 1:1 (miss-lines consumed = 100% on every run,
  0 route/miss disagreements on all 12 cells).

Why in-process: run-gen and run-spec's oracle are both greedy but numerically distinct
paths; on this artifact they diverge textually at a near-tie token (~pos 31 on chat-prose;
cross-process routing similarity collapses 0.95 -> 0.33 there). First-pass cross-process
pairing was therefore discarded; its raw logs are retained (`raw/spec-*`, `raw/spec2-*`,
`raw/trace-*`, `raw/route-*`, `raw/miss-*`) as the record of the falsification.

## What the ceiling is NOT

- It is an ORACLE ceiling in two stacked senses: (a) the drafter's token must be right
  (strict counts accepted columns only — those routings are bit-exact by self-consistency),
  and (b) the prefetcher is assumed to know the trunk's layer-by-layer routing for the
  drafted token (we read it from the verify forward). A real MoE-SpeQ implementation gets
  (b) only by running trunk layers on the draft (pipelined router-ahead-of-dispatch) or by
  an approximate router-input predictor — any such realization sits BELOW these numbers.
  That is what makes this the go/no-go ceiling: a lever whose oracle is under the gate is
  dead regardless of engineering; ours is over it with margin on 3 of 4 classes.
- Not a speedup claim: hiding a miss saves its H2D stage only if the prefetch pipeline
  overlaps it with compute in the j-round lead the drafter provides; the closed
  prefetch-prediction lane's failure (trained-predictor precision 0.2-2.6%) was the
  PREDICTOR, not the pipe. This measurement replaces that predictor with the drafter's
  measured routing.
- The spec phase itself is a net slowdown on this artifact at every K (0.39x-0.88x, matches
  the K-sweep lane) — MoE-SpeQ as a serving config needs draft cost < hidden-miss savings;
  that arithmetic is the follow-up lane, and it starts from these ceilings.
- Single rep per cell (N=1) for the ceiling fractions: routing and acceptance are
  bit-deterministic per prompt (byte-identical acceptance counts across reps were
  established by the K-sweep and accept-profile lanes on this box); tok/s in the raw logs
  are single runs and NOT quotable as performance medians.

## Baseline (plain decode, unpatched binary, N=3 interleaved same-hour)

d1736 RAW prompt, NGEN=128, dev0: 2.28 / 2.25 / 2.24 tok/s (median 2.25, N=3 sequential
same-hour, temps 34-46 C), cache 6476 slots, DECODE-WINDOW hit-rate 14.2% (that window
includes the tokenwise verify-prefill pass; the miss-trace decode-only rates are the
per-class miss rates in the table). Patched-binary run-gen matched the unpatched binary's
routing sweep-for-sweep (166/166 identical sweeps, chat-prose) — the instrumentation does
not perturb routing. Trace-writing costs throughput (chat-prose run-gen: 3.94 tok/s traced
vs 4.07 with route trace only), so no tok/s from tracing runs is quotable as baseline.
Observe mode (any MOE trace flag) pins the stats-visible dispatch path in BOTH the oracle
and spec phases of the measurement runs, so numerator and denominator see the same cache
policy — first-miss admit SLRU — as the naked binary.

## Files

- `tools/miss-trace.patch` — the moe_cache.rs diagnostic (apply to add MEMRA_MOE_MISS_TRACE)
- `tools/moespeq_ceiling_inproc.py` — the analyzer (timeline split, empirical shift/delta,
  strict/partial ceilings; run `--help` for inputs)
- `raw/ceiling-<class>-k<K>.json` — the 12 result cells, machine-readable
- `raw/spec3-<class>-k<K>.log` — oracle + spec + per-round accepts (the measurement runs)
- `raw/miss3-*.txt.gz`, `raw/route3-*.txt.gz` — per-lookup H/M + route traces (the raw
  evidence for every fraction in the table)
- `raw/boot-run*.log`, `raw/baseline-seq.sh` — boot verdict + N=3 baseline
- `raw/spec-*`, `raw/spec2-*`, `raw/gen-*`, `raw/capture-*`, `raw/misstrace-*` — the
  discarded cross-process first pass (kept: falsification record + acceptance/tok-s rows)
- Box receipts mirror: `~/receipts/moe-speq/` on 13.58.112.171
