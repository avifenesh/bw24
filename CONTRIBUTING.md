# Contributing to bw24

Issues are welcome anytime. PRs are welcome **only when they carry proof**, per the rules below —
this project has no CI runner on sm_120a hardware, so a human reviewer is the only gate standing
between a claim and merged code. Unproven PRs (no gates run, no numbers, "should be faster",
AI-generated diffs with no on-device verification) will be closed, not debated. This is not
gatekeeping for its own sake: every accepted kernel becomes a load-bearing part of a correctness
contract (see [Correctness discipline](README.md#correctness-discipline)), and reverting a bad
merge after the fact costs far more than rejecting an unproven one up front.

## Before you write code

1. Read [`research/tune-data/*.jsonl`](research/tune-data/) for your target kernel/model. This is
   a labeled corpus of *every* prior tuning attempt, wins and losses both — if your idea was
   already tried and rejected, the record says why. Re-proposing a measured loss without new
   evidence is spam.
2. Read [`ARCHITECTURE.md`](ARCHITECTURE.md) for the sm_120a hardware ledger — several plausible
   optimizations (e.g. NVFP4 grouped/MoE GEMM, sm_90/sm_100 kernel ports) are already known
   infeasible on this silicon; check before spending effort there.
3. Have access to an sm_120a (consumer Blackwell) GPU. If you don't, open an issue describing the
   idea instead of a PR — someone with the hardware can implement and measure it, crediting you.
   PRs that cannot be run and gated on the target hardware will not be merged sight-unseen.

## Required proof, in order

Every PR touching a kernel, forward pass, dispatch policy, or anything on the decode/prefill hot
path must include, in the PR description, evidence for **all** of the following. A PR missing any
one of these is incomplete, not "mostly done" — do not open it yet.

### 1. Correctness gates (all three, all green)

```bash
./target/release/kernel-check          # every quant kernel vs a CPU reference
./target/release/run-gen  ...          # prefill argmax MUST match decode argmax
./target/release/run-spec ...          # K=1..8 self-consistency: every K token-identical to plain decode
```

Paste the actual pass/fail output (or the relevant tail of it), not "gates pass." A kernel that
reduces in a different floating-point order than the one it replaces can flip an argmax at a tight
logit margin — this has silently broken "faster" kernels before (`research/tune-data/`) — so a
green run *right now, on your branch* is required, not an assumption that it still passes.

### 2. Performance: prefill AND decode, both, never just one

A kernel that helps decode and quietly regresses prefill (or vice versa) is a net loss, not a win —
report both every time, even if your change nominally targets only one:

| Metric | Baseline (main) | Your branch | Ratio |
|---|---|---|---|
| pp512 (prefill, tok/s) | | | |
| pp2048 (prefill, tok/s) | | | |
| tg128 @ 512-ctx (decode, tok/s) | | | |

Use the exact protocol in [`research/benchmarks.md`](research/benchmarks.md): **N≥3 medians**,
`gpu-full-power on` verified beforehand, and baseline + branch measured **interleaved in the same
session** (sequential cross-session runs have been measured to drift up to ~10% from clock/thermal
state alone — a same-session-only number is not evidence of anything).

### 3. Main runners exercised, not just the micro-kernel

Benchmark binaries alone (`decode-bench`, `mvq-msweep`, etc.) prove a kernel is fast in isolation;
they do not prove the engine still works. Every PR must also show a clean run through the actual
model-serving paths your change touches:

- `run-gen` — end-to-end generation on at least one real model (not a synthetic/random-weight
  smoke test), full output shown, prefill/decode argmax line included.
- `run-spec` — if your change touches anything upstream of speculative decoding's target forward
  (attention, GEMM, MoE dispatch, KV cache), run this too, not just `run-gen`.
- `bw24-server` — if your change touches request handling, batching, or anything server-side, one
  real request/response through the OpenAI-compatible endpoint.

"It compiles and the unit-level gate passed" is not evidence the runners still produce sane
output end to end — show them running.

## What gets a PR closed on sight

- No before/after numbers, or numbers from a different session without the interleaved-protocol
  disclosure above.
- Only one of {prefill, decode} measured when the change plausibly touches both.
- Correctness gates claimed "passing" with no pasted output.
- AI-generated kernel/algorithm changes with no evidence they were run on real sm_120a hardware.
- Portability changes (targeting sm_89, sm_90, datacenter Blackwell, etc.) without first reading
  [Limitations](README.md#limitations) and [Scope](#scope) below — this is a single-target engine,
  not a general runtime.
- Drive-by style-only diffs bundled with unrelated functional changes — split them.

## Scope

This is a from-scratch engine tuned for one exact machine (RTX 5090 Laptop, sm_120a). See
[Limitations](README.md#limitations) before proposing portability work — an `arch/sm89-l40s`
branch exists for Ada, but sm_120a is the only tuned target, and tuning choices elsewhere in the
codebase assume this exact memory/compute ratio.

## Where to look first

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — hard hardware constraints and the sm_120a feasibility ledger.
- [`docs/decisions/`](docs/decisions/) — design decision records.
- [`research/benchmarks.md`](research/benchmarks.md) — the exact A/B measurement protocol referenced above.
- [`research/tune-data/`](research/tune-data/) — labeled corpus of tuning experiments (config → measured result, wins and losses both) — check before re-trying something already measured.
