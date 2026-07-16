---
name: Hardware validation report
about: You ran bw24 on a desktop RTX 50-series (or other sm_120) card — share the results
---

<!--
bw24 is tuned on ONE machine: an RTX 5090 Laptop (82 SMs, ~847 GB/s). Every other
sm_120 card — desktop 5090 (170 SMs, ~1792 GB/s), 5080, 5070 Ti, 5070 — shares the
architecture, so the kernels are expected to WORK, but the tuning ratios are unvalidated
there. These reports are how the whole 50-series gets blessed. Correctness output alone
is already valuable; perf cells with a llama.cpp pairing are gold.
-->

## Your hardware

- GPU: <e.g. RTX 5090 desktop, 32 GB>
- Driver / CUDA toolkit:
- OS:
- Power state during runs: <e.g. stock / power-limited, and whether it stayed pinned>

## Correctness (required)

```
<./target/release/kernel-check tail — expect "ALL GREEN: kernels match CPU reference.">
```

```
<run-gen output on any supported model, including the "verify-prefill ... MATCH" gate line>
```

If you have a supported model with an MTP drafter, also paste the `run-spec` PASS line.

## Performance cells (optional, but the valuable part)

If you have the models locally: `BW24_MODELS_DIR=<your model root> tools/local-ci.sh --perf`
— paste the per-cell verdict lines. Cells whose models you don't have skip cleanly.

For a llama.cpp comparison, follow the interleaved protocol in
[`research/benchmarks.md`](../../research/benchmarks.md) (same session, both orders,
N≥2, one engine on the GPU at a time) — sequential numbers drift up to ~10% and can't
be used. State llama.cpp's build commit and flags.

## Anything that broke

Kernel launch failures, wrong output, OOM at sizes that fit — exact output, per the
bug-report template rules (paste, don't paraphrase).
