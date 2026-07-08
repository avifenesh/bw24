<!--
Read CONTRIBUTING.md before filling this in. PRs missing the sections below are considered
incomplete and will be closed, not reviewed in a partial state. Delete this comment block once
you've read it.
-->

## What this changes and why

<!-- One or two sentences. Not "makes it faster" — name the mechanism. -->

## Correctness gates (paste actual output, not "passed")

```
$ ./target/release/kernel-check
<paste tail>

$ ./target/release/run-gen ...
<paste tail — must show prefill/decode argmax MATCH>

$ ./target/release/run-spec ...
<paste tail — must show K=1..8 self-consistency PASS>
```

## Performance — prefill AND decode, interleaved, N≥3 medians

Protocol: [`research/benchmarks.md`](../research/benchmarks.md). `gpu-full-power on` verified.
Baseline and branch measured in the same session, interleaved (not sequential runs on different
days) — cross-session ratios have been measured to drift up to ~10% from clock/thermal state alone.

| Metric | Baseline (main) | This branch | Ratio |
|---|---|---|---|
| pp512 (prefill, tok/s) | | | |
| pp2048 (prefill, tok/s) | | | |
| tg128 @ 512-ctx (decode, tok/s) | | | |

## Main runners exercised (not just the isolated kernel benchmark)

- [ ] `run-gen` on a real model, full output attached/pasted, argmax line included
- [ ] `run-spec` (required if this touches attention, GEMM, MoE dispatch, or KV cache)
- [ ] `bw24-server` request/response (required if this touches request handling or batching)

## Scope check

- [ ] This targets sm_120a (RTX 5090 Laptop or equivalent consumer Blackwell) — the only tuned
      target. Portability PRs (sm_89, sm_90, datacenter Blackwell) must first address
      [Limitations](../README.md#limitations) in the PR description.
- [ ] I checked [`research/tune-data/`](../research/tune-data/) for prior attempts at this exact
      change and either found none, or am including new evidence that overturns a prior result
      (link the record and explain what's different).

## Hardware used to produce the numbers above

<!-- GPU model, driver/CUDA version. "I don't have sm_120a hardware" means this should be an issue, not a PR. -->
