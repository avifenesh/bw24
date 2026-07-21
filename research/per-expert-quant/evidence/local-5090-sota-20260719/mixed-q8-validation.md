# Mixed-layout per-projection q8 validation

`BW24_MOE_MIXED_Q8=1` was an opt-in Hy3 experiment. It read each retained expert projection's
authoritative `expert_layout()` metadata, uses `qmatvec_expert_q8` only for supported qtypes, and
keeps Q2_K and other unsupported projections on the established f32-dequant path.

## Resource envelope

All compilation and kernel checks ran at `CPUQuota=50%`. Full-model checks ran at an aggregate
half-core quota with `CPUWeight=1`, `IOWeight=1`, `MemoryHigh=28G`, `MemoryMax=32G`, an 8 GiB CPU
expert cache, a 16 GiB live-RAM reserve, passive OpenMP waiting, and a hard scope lifetime. No
completed validation process used swap.

An older orphaned full-model scope was discovered during the first kernel check. It held 41.6 GB
RSS and 22.8 GB VRAM, causing a captured CUDA OOM in that check. The scope was stopped, GPU
ownership was recorded, and the clean-GPU rerun completed `ALL GREEN`; the OOM is not attributed to
this change.

## Kernel evidence

- `mixed-q8-iq3-micro-n3.log`: IQ3_S 4096x1536 projection, N=3. The production q8 path, including
  one activation quantization, measured 0.0073-0.0074 ms versus 0.0245-0.0247 ms for resident f32,
  approximately 3.4x faster. The input quantization is shared by gate/up projections in the runtime.
- `kernel-check-mixed-q8-clean-gpu.log`: full kernel battery, `ALL GREEN`.
- `test-mixed-q8-unit.log`: metadata dispatch selection unit test passed.

## Model evidence

- `hy3-mixed-q8-argmax-safe-ngen2.log`: verify-prefill argmax 40129, decode argmax 40129, `MATCH`.
- `hy3-mixed-q8-k1to8-safe-ngen8.log`: K=1 through K=8 all report self-consistency `PASS`, ending
  in `SELF-CONSISTENCY PASS`.
- `hy3-mixed-q8-k1-safe-ngen8.log` versus `hy3-mixed-q8-control-safe-ngen8.log`: the one-pair short
  screen measured 1.68 versus 1.63 tok/s plain and 1.78 versus 1.75 tok/s at K=1. A later candidate
  sweep measured 1.56 tok/s plain as cache/thermal state moved. This is directional evidence only,
  not a publishable throughput result.
- `hy3-mixed-q8-control-safe-measure32-n3.log` versus
  `hy3-mixed-q8-candidate-safe-measure32-n3.log`: the controlled 32-token, N=3 medians were 1.16
  tok/s for the control and 1.08 tok/s for mixed q8, a 6.9% regression.

## Decision

The model-level N=3 gate rejected the path despite the isolated kernel gain. The implementation,
test, and flag were removed; these logs remain as the negative-result record. No performance-board
number moves.
