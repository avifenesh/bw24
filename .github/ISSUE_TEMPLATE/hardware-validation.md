---
name: Hardware validation report
about: You ran bw24 on your rig — share the results, whatever the card
---

<!--
bw24 is tuned on ONE machine: an RTX 5090 Laptop (82 SMs, ~847 GB/s). Reports from EVERY
end-user rig are wanted:

- Other sm_120 cards (desktop 5090, 5080, 5070 Ti, 5070) share the architecture — kernels
  are expected to WORK, tuning ratios are unvalidated. These reports bless the 50-series.
  (The 188-SM RTX PRO 6000 server die already validated binary-compat + bit-consistency.)
- Older NVIDIA cards (Ada/Ampere): the main build targets sm_120a and an `arch/sm89-l40s`
  branch exists for Ada — "what breaks where" reports map the compatibility floor.
- Correctness output alone is already valuable; perf cells with a llama.cpp pairing are gold.
-->

## Your hardware

- GPU: <exact model + VRAM, e.g. RTX 5090 desktop, 32 GB>
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
