# Hy3 native ABI v2 validation

Target: NVIDIA GeForce RTX 5090 Laptop GPU, driver 595.71.05, 24,463 MiB HBM.

This receipt covers bw24's self-contained CPU expert implementation. It does not compile, link, or
load llama.cpp, ggml, or another inference runtime.

## Pinned inputs

- native companion SHA-256: `c6423d768bea95f8a5a63e99a370dd323590fb360d8e0bf3af52de64481afc71`
- post-freeze-gate `run-gen` SHA-256: `c46c0af92ef7de234c1e6ad19d8943741208ae34a23df158d92769922475a733`
- default-profile `run-spec` SHA-256: `919d0aeae9b69e3b4f5327504c7f28fd9a701bd6a2ceecf5c1d2a6f16a15d778`
- mirror-map v2 SHA-256: `861f58c5ad506f0d62242bed5cd79a97313e83a9df4412ddc4930ce1b0159a15`
- mirror payload: 237 files, 78,490,288,128 bytes
- runtime model root: the generated `hy3-layer103p5-dual-nvme` view
- CPU topology: cores 0-7, eight OpenMP compute workers, eight positioned-read workers
- memory profile: 20 GiB requested/effective host cache, 4 GiB live-RAM reserve, 0.90 requested HBM fraction
- residency profile: 128 discarded tokens before freeze; speculative profiling uses K=3

The shared object's `ldd` output contains only the platform C++, math, OpenMP, GCC, and C runtimes.

## Correctness gates

| Gate | Result | Raw log |
|---|---|---|
| Native packed-row oracle | 12 formats at widths 256, 1536, and 4096; non-finite/subnormal checks and nonzero production MoE composition `PASS` | `cpu-native-v2l-validation.log` |
| GPU kernel/reference battery | `ALL GREEN` | `kernel-check-native-v2l.log` |
| Hy3 prefill/decode argmax | post-freeze serving assignment `40129 == 40129`, `MATCH` | `run-gen-native-v2o-final-postfreeze-argmax-t8-cache20-warm128-chat-ngen32.log` |
| Hy3 speculative K=1 through K=8 | launcher runtime defaults plus the all-depth gate selector; every depth `self-consistency: PASS`; aggregate `SELF-CONSISTENCY PASS` | `run-spec-native-v2s-final-default-warm128-k1to8-chat-ngen8.log` |

The first K=1..8 attempt intentionally remains in
`run-spec-native-v2l-final-k1to8-t8-cache20-chat-ngen8.log`. It paired the generation-pinned map
with the persistent source model root instead of the dual-NVMe view. ABI v2 rejected the first CPU
miss with `CPU expert source inode is absent from mirror map`; no generation was scored. The
corrected invocation used the exact view that produced the map and passed.

### Current-main integration

The native runtime was also integrated with current `main` (`e2656409` plus `31fa5d13`) and rebuilt
in the release profile on the same RTX 5090. The merged binary passed the production warm-128
chat-prompt argmax gate (`40129 == 40129`, `MATCH`) in
`run-gen-main-merge-argmax-ngen4.log`. A subsequent production warm-128 K=1 through K=8 sweep
passed self-consistency at every depth and reported aggregate `SELF-CONSISTENCY PASS`; its raw log
is `run-spec-main-merge-k1to8-ngen4.log`.

That integration sweep intentionally used the launcher's one-token raw smoke prompt and only four
generated tokens. It therefore had zero accepted draft tokens and is correctness evidence only,
not a speculative-quality or performance result. Nonzero acceptance remains established by the
earlier chat-prompt N=8 full-depth sweep above. The preceding K=1-only smoke is retained in
`run-spec-main-merge-k1-ngen4.log` for completeness.

Merged release artifacts:

- `run-gen` SHA-256: `4c1d274a2ba0afb9d45d74229b70c7309842f19309d9c36697ba4effe8ece74a`
- `run-spec` SHA-256: `cb86b28087007ed0f1a96cfe4c960b80ab29c1468cd33ab2cc0898a23f446cec`
- native companion SHA-256: `c6423d768bea95f8a5a63e99a370dd323590fb360d8e0bf3af52de64481afc71`

## Performance observations

These are single observations, not medians and not a performance-board move.

| Path | N | Result | Thermal regime | Storage/CPU observation |
|---|---:|---:|---|---|
| `run-gen`, post-freeze greedy decode | 32 tokens | 4.48 tok/s | active desktop, 55 C start, 64 C end | 27.80 GB physical reads; CPU backend 6.254 s wall, 2.803 s I/O, 3.246 s compute |
| `run-spec` plain control before K sweep | 7 tokens | 3.76 tok/s | active desktop, 55 C start, 68 C end | full sweep: 819.035 GB CPU reads; 83.893 s I/O, 117.794 s compute |

The default warmup depth was checked with an identical teacher-forced N=32 continuation
(`e45e52016561f87bfb168b6369cad636c6b7b77c726fcd2b268d3e56f4f89090`). In one
same-session, forward-order pair, 8 discarded tokens measured 4.46 tok/s from a 56 C start and 128
measured 4.52 tok/s from a 59 C start. Both passed the post-freeze argmax gate. This is a single
pair, not a median; the raw logs are `run-gen-native-v2q-forced-warm8-ngen32.log` and
`run-gen-native-v2r-forced-warm128-ngen32.log`.

Speculative K=1 through K=8 remained exact but was slower for this short prompt. Sustained 10 tok/s
is still the target. The final profile shows that storage and CPU compute are both first-order costs,
so the next tuning lane must overlap positioned reads, cache publication, and expert compute rather
than treating a faster row kernel alone as an end-to-end result.

The Hy3 MTP head is full-vocabulary in this receipt. `BW24_FRSPEC_TRIM` was unset and no `d2t`
trimmed-vocabulary artifact was loaded, so MTP vocabulary trimming remains a separate measured lane.

The controlled Q2_K microbenchmark used two reverse-order topology sweeps, with 10 warmups and 100
timed calls at every point on the active-desktop powersave regime (55 C start). The two-pass means
for eight and twelve threads differed by 0.7%, while the winner reversed by about 8% between the
individual passes; eight remains the lower-contention default. Raw data:
`cpu-native-v2k-q2k-thread-sweep.log`.
