# Testing: the tiered gate structure

Two regimes, one rule: **the full battery gates every merge and tag, unchanged; fast-gate
accelerates the dev loop between battery points.** Nothing in this document weakens the
merge/tag bar — a fast-gate green is a *keep going* signal, never a *ship* signal.

## The tiers

| Tier | Wall (5090 rig, measured 2026-08-02) | What runs | When |
|---|---|---|---|
| 0 | seconds (~2 s kernel-check scoped + build) | workspace compile + kernel-check scoped to the touched sections | every edit-compile loop |
| 1 | ~1–2 min | tier 0 + golden-token argmax probe on ONE model per affected kernel class (+ one single-K spec probe when the diff touches the spec pipeline) | before every dev-loop commit |
| 2 | tens of minutes | the full battery: `tools/local-ci.sh` — kernel-check ALL GREEN (~4.5 min), prime-gate, run-gen argmax per model, VERIFY-GATE, spec self-consistency, serve-smoke, `--perf` cell battery | **every merge, every tag** (unchanged) |

Entry point:

```bash
tools/fast-gate/fast-gate.sh                       # tier 1 vs HEAD (uncommitted work)
tools/fast-gate/fast-gate.sh --tier 0              # compile + scoped kernel-check only
tools/fast-gate/fast-gate.sh --diff main           # scope = everything since main
tools/fast-gate/fast-gate.sh --tier 2              # execs tools/local-ci.sh (the real gate)
tools/fast-gate/fast-gate.sh --smoke               # add the perf tripwire (see below)
```

## Change-scoped gating (tier 0)

`git diff --name-only <ref>` (plus untracked files) is mapped through
[`tools/fast-gate/map.tsv`](../tools/fast-gate/map.tsv) — an editable TSV that encodes the
dispatch structure: which kernels serve which model classes. Every matching row contributes
to the plan (union); an unmatched path falls back to the conservative full plan and prints a
warning (add a row when that happens).

kernel-check gained two loud diagnostics seams for this (see the `kc_model` header in
`crates/memra-engine/src/bin/kernel_check.rs`):

- `MEMRA_KC_FAST=1` — synthetic arms only, every weight-oracle section skipped (loudly).
  Measured: **~1.4 s** vs **~4.5 min** full (the model-backed GEMM oracles are >98% of the
  wall — 266 s of 268 s in the timed run,
  `research/fast-gate-20260802/kernel-check-full-timed.log`).
- `MEMRA_KC_ONLY=csv` — synthetic arms + only the weight-oracle sections whose name matches
  (`dtype5`, `nvfp4-gemm`, `q8mmq-gemm`, `q4_0-mmq`, `q4_0-sk-arm`, `iq4xs-mmq`,
  `f16g-kq-direct`, `nvfp4-27b-shape`, `nvfp4-mmvq`, `nvfp4-batched`, `a6-split-plane`,
  `d2-cache-bit-identity`, `fast-router-batch`).

Both seams print a `KC-SKIP` line per skipped section naming the env — a scoped run is never
silently narrower than it looks. They are diagnostics-class flags per the flags doctrine
(dev-loop scoping), not defaults: the battery runs kernel-check naked.

## Golden-token pinning (tier 1)

The battery's run-gen argmax gate re-derives its reference every run (prefill forward +
tokenwise decode + batched prime — three primes per invocation on a big model). fast-gate
pins the *output*: for each probe in
[`tools/fast-gate/models.tsv`](../tools/fast-gate/models.tsv), the greedy `tokens: [...]`
line at a battery-green commit is stored in `tools/fast-gate/goldens/<id>.tokens` with its
SHA and timestamp. A tier-1 probe then:

1. runs `run-gen` with `MEMRA_NGEN=20` (one model per affected kernel class),
2. requires the in-run gates green (prefill/decode argmax `MATCH`, no
   `MISMATCH-STRUCTURED` from the batched-prime gate; `FLIP-NEARTIE` stays reported,
   non-fatal, per the #46 contract),
3. byte-compares the `tokens:` line against the pinned golden — **any diverged token id is
   a FAIL**, with an instant verdict and no reference recompute.

Greedy decode is deterministic on this engine (run-to-run nondeterminism is itself a gated
bug class — see the ping-pong SSM-state fix in `cache.rs`), so token divergence == behavior
change. Exactness is clock-independent: goldens generated under any thermal/power regime
are bit-valid.

Spec-pipeline diffs add one single-K spec probe (`run-spec` + `MEMRA_SPEC_K` for the qwen
MTP family; `gemma-gate` + `MEMRA_SPEC` stream-agreement for the gemma drafter family).
The K=1..8 sweep stays tier-2.

`kind=cmd` probes (models.tsv) are self-gating commands — host unit tests or GPU oracle
gates like `sample-check` — whose gate is exit 0. They pin no golden and exist for code
the greedy token goldens structurally cannot see (the sampler chain).

### Golden refresh protocol

Goldens refresh **only at full-battery green points**, never mid-dev:

```bash
tools/local-ci.sh                                  # must be ALL GREEN first
tools/fast-gate/fast-gate.sh --refresh-goldens     # refuses a dirty tree (--force overrides, loudly)
git add tools/fast-gate/goldens && git commit ...  # goldens are checked in, SHA-stamped
```

If a legitimate behavior change moves tokens (new kernel numeric config promoted through the
full battery), the refresh happens in the same commit that lands the change *after* its
battery run — the golden diff in review is the visible record that tokens moved.

## Perf smoke (`--smoke`) — tripwire, not evidence

Each tier-1 probe already times its decode window; `--smoke` compares that **single rep**
against the tok/s recorded at the golden point (`goldens/<id>.perf`): WARN at >10% drop,
FAIL at >25%. This exists to catch catastrophic regressions (a kernel fell off its fast
path) inside the dev loop — it is explicitly **not** a publishable number and never moves a
board: publishable performance stays N≥5 interleaved same-session medians per
[`research/benchmarks.md`](../research/benchmarks.md), and drift detection at fine grain
stays `tools/local-ci.sh --perf`. A smoke WARN/FAIL means "re-measure with the real
protocol", nothing more.

## The probe-regime laws (learned by breaking kernels on purpose)

The catch demonstrations below exposed three ways a probe can be green while the touched
code is broken. The mapping table encodes the fixes; keep them in mind when adding rows:

1. **The probe must EXERCISE the touched dispatch class, not just the touched model
   family.** On a 24GB rig every daily MoE model loads RESIDENT — the SLRU cache, staged
   `moe_cached_gemm*`, and spill dispatch never run under a naked probe, and a deliberate
   gate/up weight swap there passed all four default probes. `q35slru`
   (`MEMRA_MOE_RESIDENT=0` + `MEMRA_MOE_SLOTS=1024`) forces that regime (68.5% hit rate,
   185k misses in its pin log) and caught the same break instantly.
2. **Depth is a dispatch axis.** The short probes decode at t_kv below/near the FA vec
   floor and windows; the gemma fp8-KV g-module arms (hd512 tb512 staging, windowed SWA)
   only execute at depth. `g12d` (the battery's 1736-id depth prompt) caught a K
   element-permutation break in the live tb512 staging arm that every short probe missed.
   (Its golden is 16 ids, not 20: token 17 of the g12 depth continuation is a real
   run-to-run near-tie flip — `g12-depth-nondeterminism.log` — and a 20-id golden
   false-fails ~1/8 runs. q9/q35/g31 deep continuations measured deterministic x5-x8.)
3. **Greedy goldens route around the sampler entirely.** temp=0 collapses to argmax, so a
   broken gumbel/softmax-gather kernel or a backwards top-k is invisible to every token
   golden. `samp` (the `sample-check` GPU oracle) and `sampt` (`cargo test -p
   memra-sampling`) are `kind=cmd` probes — self-gating commands, exit 0 = PASS, no golden.

### Catch demonstrations (all breaks reverted; diffs + consoles in receipts)

| Break (deliberate) | Caught by | Receipt |
|---|---|---|
| MoE staged dispatch: up-projection reads GATE weights (`moe_cached_gemm_q8`) | tier-1 `q35slru` (run-gen argmax gate, exit 101); plain `q35` was BLIND (resident regime) | `break-moe-staged-*` |
| FA v4 K-scale skew x1.001 (default hd256 staging arm) | tier-0 kernel-check synthetic arms — 46 bit-identity FAILs (`fa_decode_rows`/`seqs_v4`) | `break-fa-v4-*` |
| FA hd512 tb512 K element permutation (live gemma fp8 global arm) | tier-1 `g12d` depth probe (prefill/decode argmax MISMATCH, exit 101) | `break-fa-tb512-perm-*` |
| MMQ Q8_0 wrong-block scale (index mixup, `load_tiles_q8_0`) | tier-0 `MEMRA_KC_ONLY=q8mmq-gemm` — rel=2.4e-1 vs the f32 oracle, 8 FAILs | `break-mmq-q8-idx-*` |
| f16g IQ4_XS dequant off-by-one (`ls-31`) | tier-0 `MEMRA_KC_ONLY=f16g-kq-direct` — byte-identity maxdiff 1.15e2, 8 FAILs | `break-f16g-iq4xs-*` |
| device sampler acceptance-prob skew x1.001 (`softmax_gather_f32`) | tier-1 `samp` (sample-check vs CPU softmax, exit 1) | `break-sampling2-*` |
| host sampler top-k keeps WORST k (ascending sort) | tier-1 `sampt` (memra-sampling unit tests, exit 101) | `break-sampler-host-*` |

### Demonstrated coverage gaps (documented honestly, not closed)

- **Default-dead rollback seams**: kernels/helpers only reachable through non-default env
  seams are invisible to naked probes *by construction*. Verified twice: a `dq_K_lane`
  q8_0-branch lane swap (only live under `MEMRA_NO_FA_VEC`/v4-off arms at hd256, smem twin,
  `MEMRA_GEMMA_GKV=0` globals) passed everything, and so did an fp8 `dq_K_lane` lane swap
  (`break-fa-decode2-*`, `break-fa-fp8k-*`) and a kd requant skew (`break-fa-tb512-*` —
  int8 requant made the x1.001/126-vs-127 skews vanish at the __float2int_rn rounding,
  a magnitude-tolerant class, while the permutation break in the same loop was caught).
  Scale-skew breaks below the requant rounding step need value-exact oracles, not probes;
  the bit-identity kernel-check arms are the teeth there — extend those when adding arms.
- **A subtle *uniform* K-scale skew (x1.001) on an arm with no kernel-check bit-identity
  twin was caught by NO tier** (`break-fa-decode-*` — attention renormalizes softmax, so a
  uniform score scale barely moves greedy tokens at short depth). Wide-margin numeric skews
  are a tier-2/battery class; fast-gate's teeth are structural breaks (index/lane/element
  mixups), which it demonstrably catches.
- **Sampled serving path** (temp>0 end-to-end): `samp` oracles the kernels, but no probe
  runs a sampled generation stream; distribution-level drift stays a battery/eval concern.

## What fast-gate does NOT cover

- **Serving surface** (`crates/memra-server/`): run `tools/serve-smoke.sh` (fast-gate prints
  the pointer when the diff touches it).
- **Acceptance drift**: invisible to every exactness gate by construction (decode and verify
  shift together) — only the tier-2 perf battery's per-cell acceptance verdicts catch it.
- **H100/sm_90a lane**: `tools/validate-h100.sh` on an H100, per its own laws.
- **Cross-model blast radius**: tier 1 probes one model per kernel class; the full per-model
  matrix runs at tier 2.

## Receipts

Timings, the deliberate-break catch demonstrations (diffs, consoles, per-probe raw logs),
and the depth-determinism sweeps: [`research/fast-gate-20260802/`](../research/fast-gate-20260802/).
