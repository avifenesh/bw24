# v0.71.0 → darklanes sync note (post-release)

What `~/projects/darklanes` (private product repo) needs after v0.71.0 is public.
**This file is a note only — nothing in the darklanes repo is edited from the engine lane.**
Written 2026-08-06 on the tag-day train (@ 4cbf5e39 + release commits); supersedes the
RUNBOOK §7 sketch, which was written before the admit-oom and serving-density lanes landed.

## 1. Version pins

| where | change |
|---|---|
| serve configs / deployment docs | pinned memra version → **v0.71.0** (crates.io `memra-server 0.71.0`; tarball tag `v0.71.0`) |
| the pill's local-memra default (`:8002` serve scripts) | pin/refresh to v0.71.0 |
| internal state docs | record the engine release (v0.71.0, 2026-08-06, headline mechanisms below) per the release-discipline standing rule |

## 2. Serve-claim refreshers — what is now claimable, and at what exact number

These are the product-copy-facing changes. Every number below is engine-measured with
receipts in the public repo; quote them as-is or not at all.

**a) The felt-latency arc (round-cadence SSE + admission yield).** The product latency
story changed materially this release. 27B NVFP4+MTP, K=3, local 5090 laptop, N=5 medians
in one lock hold:

| claim | was | now |
|---|---|---|
| solo first text | 0.41 s (B32) / 1.16 s (B128) | **0.12 s at ANY burst size** |
| solo inter-chunk gap p50 | 299 ms | **27 ms** |
| contended first text, B32 | 0.54 s | **0.123 s** |
| contended first text, B128 | 1.60 s | **0.152 s** |

The load-bearing shape of the claim: felt latency **no longer scales with the speculative
burst size**. Content is byte-identical either way — only chunk boundaries move. Any
website-spec §perf or product copy quoting the old streaming/contended numbers updates.
Cost, if asked: −3.4% aggregate tok/s at c=8 saturation for 3.8x better p50 (p95 tail
pays); c=1 parity. Receipts: `research/sse-cadence-20260805/`, `research/admission-20260806/`.

**b) 64-client robustness is now a CLAIMABLE PROPERTY (new this release).** At
`MEMRA_MAX_SESSIONS=64` with speculation on, a 24GB card previously lost **all 64 streams**
to step-time CUDA OOM. Fixed and, more importantly, **gated**: `tools/serve-stress-gate.sh`
runs in the engine's local CI and as the `sstress` fast-gate arm, asserting 64/64
well-formed streams (peak 23.1 of 24.5 GB), with a teeth arm that forces a broken headroom
reserve and verifies the gate still catches it.

Claim it as: *"64 concurrent speculative streams on a 24GB card, asserted in CI — not a
best-effort number."* Do **not** claim a higher concurrency number; 64 is what is gated.
The c=8 control is behaviorally identical to pre-fix (+0.49% agg, zero defer/park events),
so nothing in the small-concurrency story changes. Receipts: `research/admit-oom-20260806/`.

**c) Chunk-size-invariant exactness (new contract wording).** Chunked prefill now produces
**bit-identical logits across `MEMRA_PRIME_CHUNK` values with no flags** — "one canonical
greedy output per prompt" is back as a shipping contract, gated by the `chunkinv` battery
arm plus a canary that injects the legacy arithmetic and must fail. Quality moved the right
way (27B NLL −1.1%).

Scope discipline for copy: this is invariance across *chunk sizes*, and it composes with the
existing c=1-vs-c=16 isolated-identical serving contract. It is **not** an identity claim
against a single-token reference decode — the batched-plain path still has a documented,
bounded near-tie flip class, and speculative decode is gated *self-consistent*
(`run-spec` K=1..8), not identical. One more scope note now in the public docs: one
canonical output **per FA-split config** — an 82-SM and a 188-SM card pick different
`fa_split_keys` rungs, a legal near-tie flip, byte-identical at matched split.

**d) Block-128 FP8 is native by default — the 3.8 day-one story needs NO flags.** The class
Qwen3.6-FP8 actually ships (`weight_block_size [128,128]`) is now the default residency and
prefill route: decode +1.69%, prefill +0.83%, 430 MiB freed at single residency, and
NaN/ragged checkpoints fail safe to the exact dequant arm. Verify the darklanes copy of the
qwen38 bringup runbook carries the §3b correction and states **no flags required**.

**e) The B128 throughput tier is now quotable for batch endpoints.** `MEMRA_SPEC_BURST=128`
buys +8.4% (c=1) / +8.5% (c=8) and, with both felt-latency fixes in, trails B32's contended
first text by one 29 ms round-cadence quantum instead of a 3x cliff. Default stays 32 for
the daily pill; 128 is a live owner call for throughput-tier/non-streaming endpoints.

## 3. Config checks to run against the darklanes serve configs

- **Kill any `MEMRA_PRIME_INVARIANT` / `MEMRA_PRIME_GRAIN` usage** — the door no longer
  exists (removed per the flags doctrine; superseded by the grain-free default). It was
  opt-in, so naked configs are unaffected, but a config still setting it now sets nothing.
- **Confirm none set `MEMRA_SSE_PER_BURST=1` or `MEMRA_ADMIT_YIELD=0`** — the defaults *are*
  the fixes. A pure-batch tier may deliberately set `ADMIT_YIELD=0` to get lockstep fairness
  back; if so, document it per config so the latency claims are not quoted against it.
- **Add an explicit `max_tokens` to serve configs and client defaults (NEW).** Admission
  sizes each session's KV ladder from the request's own bound; omitting it falls back to the
  context ceiling and strands a measured **6.3% (c=16) / 12.6% (c=32) of a 96GB card** in
  ladder slack at `MEMRA_CTX=32768`. Right-sized requests strand ~0%. Also keep `MEMRA_CTX`
  at the workload rather than the maximum. This is the cheapest density win available and it
  is pure config. Receipt: `research/serving-density-20260806/VERDICT.md` (Q1).
- **Do not set `MEMRA_ADMIT_RESERVE_MB`** — it is a teeth/diagnostics door, not a tuning
  knob, and it warns loudly when set.

## 4. What did NOT change (so nothing gets over-claimed)

- **Prefix/prompt-cache sharing did not improve.** Sealed-prefix sharing was investigated
  and is **receipted-dead at the agent-trace shape**: duplication measures 0.85–7.69% of a
  96GB card at c=16/32 with 4–8k shared prefixes, below the 10% bar. Revive only if a
  product shape lands ~22k+ sealed prefixes (RAG / repo-context agent farm, not the
  coding-agent trace). No product claim to make here either way.
- **The competitor head-to-head numbers are unchanged and frozen.** The memra-side latency
  stack moved a lot this release, but competitor benching is stopped by doctrine and the
  head-to-head was NOT re-run — the old 0.53-vs-0.19 s row stays frozen and labeled as a
  pre-fix measurement in the public docs. Do not compute a new ratio from the new memra
  numbers and the old competitor denominator; that is exactly the cross-run comparison the
  interleaving law forbids.
- **No tracked perf-board number moved** (bare-CLI + H100 cells only; verified
  `update-perf-board.py --check` green). The serving board and felt-latency numbers live in
  hand-written prose.
- **W4A4 / FP4 activations: door stays shut.** If a product conversation reaches for a "2.4x
  FP4" number, it does not exist as a net figure — no rotation method publishes runtime
  latency, and the estimate is 0.9–1.8x net after Hadamard overhead. W4A8 is the flagged
  pragmatic alternative, unfunded.

## 5. Shipped-state confirmation (filled in after the tag, 2026-08-06)

- **v0.71.0 is live and public.** Tag `v0.71.0` → `98da33bd`; GitHub release published
  (not a draft) at 11:41:31Z; `release` workflow SUCCESS; `publish` workflow SUCCESS in
  11m45s with **no crates.io 429** — the resumable per-crate path from v0.69's recovery did
  not need to engage. `origin/main` = `98da33bd`. Everything in §1–4 above is now quotable.
- **Pin exactly `0.71.0`.** All 8 intra-workspace deps are pinned `=0.71.0` and the publish
  workflow refuses a tag/version mismatch, so a partial pin will fail loudly rather than
  silently mixing versions.

### One security item, and it is NOT a product-copy item

An `nsys` profile committed by a research lane had captured the profiled process's whole
environment into the binary `.nsys-rep` blob, including live credentials. GitHub's push
protection blocked the release push and the blobs were removed from the (still unpublished)
history before anything reached the remote — **nothing leaked publicly**, `origin` never
received them, and no secret-scanning unblock URL was used. `*.nsys-rep` is now in
`.gitignore`.

Two consequences that touch darklanes:

1. **One of the captured secrets was `REVUTO_GITHUB_WEBHOOK_SECRET` — it belongs to a
   different project entirely.** Anything sharing that shell environment is in scope for
   review, not just this engine repo.
2. **Credential rotation is an owner action and is NOT done by the engine lane.** The
   captured `GH_TOKEN` and `AWS_BEARER_TOKEN_BEDROCK` should be treated as exposed
   regardless of the clean push, because the profile was taken on a rented pod and sat in
   world-readable files locally. Rotation status: **owner-blocked, unresolved at the time of
   this note.**

Standing lesson worth carrying into any darklanes profiling runbook: profilers serialize the
environment. Never commit a raw profiler capture; commit the derived summary (this release
kept the `_cuda_gpu_kern_sum.csv` files, which carry the actual evidence and are clean).
