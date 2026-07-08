# Contributing to bw24

Issues and PRs welcome.

## Kernel PRs

Any kernel change must pass, in order:

1. `kernel-check` — every quant kernel vs a CPU reference.
2. `run-gen` argmax gate — prefill and decode paths must agree on the next token.
3. `run-spec` self-consistency — speculative output at K=1..8 must be token-identical to plain decode.

Include before/after numbers measured with the protocol in [`research/benchmarks.md`](research/benchmarks.md) (N≥3 medians, full power state verified — `gpu-full-power on`). "Faster" kernels that change floating-point summation order and flip an argmax at tight logit margins get rejected even if benchmarks look good — exactness is part of the contract, not an afterthought.

## Scope

This is a from-scratch engine tuned for one exact machine (RTX 5090 Laptop, sm_120a). See [Limitations](README.md#limitations) before proposing portability work — an `arch/sm89-l40s` branch exists for Ada, but sm_120a is the only tuned target.

## Where to look first

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — hard hardware constraints and the sm_120a feasibility ledger.
- [`docs/decisions/`](docs/decisions/) — design decision records.
- [`research/tune-data/`](research/tune-data/) — labeled corpus of tuning experiments (config → measured result, wins and losses both) — check before re-trying something already measured.
