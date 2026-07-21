# Native Q2_K paired AVX-VNNI win

Target: NVIDIA GeForce RTX 5090 Laptop GPU, driver 595.71.05, 24,463 MiB HBM,
with an Intel Core Ultra 9 275HX host. Runtime base: merged native bw24 Hy3 serving at
`40f02fb`; implementation/evidence branch base at measurement time: `363741d`.

The implementation is entirely bw24-native. It does not compile, link, or load llama.cpp, ggml,
or another inference runtime.

## Measured bottleneck

The first instrumented N=32 control isolated the caller-blocked tail of each asynchronous CPU
expert ticket. Of 6.730 s native backend wall, 6.622 s was exposed at the join: 98.4% of CPU
expert time remained on the token critical path. The backend split was 3.627 s compute and
2.894 s direct-I/O/cache fill, so speculative verification batching was not the next lever.

Raw log: `exposed-wait-profile-ngen32-run1.log`.

## Implementation

Q2_K stores each pair of 16-value groups in one 32-byte packed span. The prior native path loaded
and reduced each group independently with 128-bit AVX-VNNI. The new guarded path:

- loads the packed 32-byte group pair once;
- decodes both groups together with AVX2;
- combines their two Q8/16 activation blocks into one 256-bit vector;
- evaluates both integer dots with one 256-bit `vpdpbusd`; and
- applies the two scale/min terms sequentially in original group order, preserving floating-point
  accumulation order.

The path is selected only when both `__AVXVNNI__` and `__AVX2__` are available. Other hosts and all
other quant formats retain the established native implementation.

## Paired microbenchmark

Five interleaved control/candidate pairs used the production four-expert
4096x1536x4096 Q2_K MoE shape, eight pinned P-cores, 10 warmups, and 100 timed calls per process.
Every call produced the identical checksum `8076798197760.00000000`.

| Path | Runs (ms/token) | Median | Delta |
|---|---|---:|---:|
| 128-bit groupwise control | 1.089, 1.094, 1.097, 1.091, 1.097 | 1.094 ms | — |
| 256-bit paired groups | 0.940, 0.925, 0.931, 0.924, 1.209 | 0.931 ms | -14.9% |

The same run passed the independent packed-row oracle for all 12 supported formats at widths 256,
1536, and 4096, the production MoE composition check, and non-finite/subnormal checks.

The fifth candidate process was an outlier, but the interleaved median remained 14.9% faster. The
raw log records the command, arm labels, binary and companion hashes, firmware power profile, and
start/end temperature: `cpu-q2k-avxvnni-pair-provenance-microbench.log`.

## Local 5090 end-to-end validation

Three interleaved control/candidate pairs used both orders. Every arm was a cold process cooled to a
55-56 C start. Pairs 2 and 3 record the firmware at its 25 W dynamic-boost, 150 W GPU-TGP, and
140/175 W CPU power maxima in each raw log; pair 1 records its start temperature but not the firmware
knobs. All used the same 45-token chat prompt, N=32 greedy output, 13.97 GiB HBM expert budget,
20 GiB CPU expert cache, eight pinned P-cores, direct I/O, verified dual-NVMe mirror, and 128
discarded tokens before residency freeze. Every arm froze exactly 5,285 HBM blocks: 1,719 complete
experts plus the same 128 stranded projection blocks. All six arms generated the same 32 token ids,
passed post-freeze argmax `40129 == 40129`, and reported zero process swaps.

| Path | Throughput runs | Median | CPU compute | Backend wall | Exposed wait | CPU reads |
|---|---|---:|---:|---:|---:|---:|
| control | 4.19, 4.42, 4.37 tok/s | 4.37 tok/s | 3.237 s | 6.424 s | 6.310 s | 27.80 GB |
| paired Q2_K | 4.60, 4.51, 4.67 tok/s | 4.60 tok/s | 3.025 s | 6.068 s | 5.956 s | 27.80 GB |

The arm medians improve end-to-end throughput by 5.3%, CPU compute by 6.5%, and exposed CPU wait by
5.6%, with unchanged I/O volume. The median of the three paired throughput deltas is +6.9%. This is
an N=3 interleaved local-Hy3 result in the active-desktop thermal regime, not a Qwen performance-board
row.

Raw logs: `exposed-wait-profile-ngen32-run1.log`,
`cpu-q2k-avxvnni-pair-e2e-ngen32-run1.log`, `e2e-pair2-candidate-ngen32.log`,
`e2e-pair2-control-ngen32.log`, `e2e-pair3-control-ngen32.log`, and
`e2e-pair3-candidate-ngen32.log`. Tested companion SHA-256:
`26303685576126a829933144be6af7dad6a6c19995b0b90421ca196d47c31621`. Pair 1's control log binds
the baseline to source commit `363741d` and the logged default companion path; the subsequent
control logs record the retained baseline companion SHA-256
`c6423d768bea95f8a5a63e99a370dd323590fb360d8e0bf3af52de64481afc71`.

## Merge gates

- native packed-row and production-MoE oracle: `ALL GREEN`;
- GPU/reference `kernel-check`: `ALL GREEN`;
- post-freeze `run-gen` prefill/decode argmax: `MATCH`, with identical control/candidate output;
- full-vocabulary Hy3 `run-spec`: self-consistency `PASS` for every K from 1 through 8; and
- build/check: release binaries built successfully, focused Rust CPU-expert tests passed 4/4.

Raw logs: `cpu-q2k-avxvnni-pair-provenance-microbench.log`,
`kernel-check-q2k-avxvnni-pair.log`, and `run-spec-q2k-avxvnni-pair-k1to8-ngen8.log`.

## Decision

Ship the paired AVX-VNNI Q2_K path as the native default on supported CPUs. Keep the portable
fallback unchanged. The measurement also establishes the next ceiling: after this win, 5.956 s of
the 6.955 s N=32 generation window is still exposed CPU expert wait, split between remaining mixed
quant compute and cache-miss I/O. Sustained 10 tok/s is not yet reached.
