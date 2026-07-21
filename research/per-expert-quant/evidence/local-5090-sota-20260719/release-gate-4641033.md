# RTX 5090 release gate for `4641033`

Target: NVIDIA GeForce RTX 5090 Laptop GPU, driver 595.71.05, 24,463 MiB HBM.

Product commit: `4641033f0fee83d0cc4fc77bbbffa0f9d3adc8d0`.

Pinned companion inputs:

- llama.cpp commit `bb090d1f1dbf3c29df6778fda123aa352329514e`;
- `libbw24-cpu-experts-release.so` SHA-256
  `0d9ea5293ffb12bf5c67b6bcb669e156c8f6e0d0561e68d940bdfd5a458e29a9`;
- `tools/bw24_cpu_experts.cpp` SHA-256
  `c56c7d004d5a951c322274a5af3221d9c6e19588480a7a935a8871870d7a27ca`.

## Correctness gates

| Gate | Result | Raw log |
|---|---|---|
| kernel CPU-reference battery | `ALL GREEN` | `release-gate-kernel-check-4641033.log` |
| Hy3 prefill/decode argmax | `40129 == 40129`, `MATCH` | `release-gate-run-gen-safe85-4641033.log` |
| speculative K=1 through K=8 | `SELF-CONSISTENCY PASS` | `release-gate-run-spec-k1to8-max95-4641033.log` |

## Current performance observations

Both measurements used N=64 once and are labeled single runs. They are not board-moving medians.

| Profile | Effective host cache | Result | Storage observation |
|---|---:|---:|---|
| safe 0.85 HBM, 4 GiB RAM reserve | 24.37 GiB | 5.92 tok/s warm | 50.65 GB physical reads / 64-token window |
| max 0.95 HBM, zero RAM reserve | 29.71 GiB | 5.09 tok/s plain; 3.42 tok/s K=1 | system swap pressure increased; 722.61 GB read across the full K sweep |

The result rejects the assumption that the largest admitted host cache is automatically fastest on
a busy desktop. The release launcher therefore retains the conservative live-headroom cap. The
historical quiet-machine 36 GiB fixed-LRU run remains the strongest observed result at 7.60 tok/s
K=1, but it is not presented as the pinned release default.

An initial maximum-profile attempt supplied `0.97`; the final parser correctly rejected it because
the supported range ends at `0.95` and fell back to `0.80`. That run was interrupted before
scoring. Its raw log is retained as `release-gate-run-spec-k1to8-maxprofile-4641033.log`.
