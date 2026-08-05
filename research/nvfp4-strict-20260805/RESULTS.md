# NVFP4 strict equalization — decode-batch-gate `--mode strict` now covers NVFP4

Lane: `lane/nvfp4-strict` off 93420980. Rig: local 5090 (sm_120a), runs serialized under
`flock /tmp/gpu5090.lock` (GPU shared with the fp8-decode-v1 lane). All runs below are
single runs — these are bit-identity/exactness gates, not perf medians.

## Root cause (confirmed)

`--mode strict`'s equalizing env (`MEMRA_MMVQ=0 MEMRA_NO_FUSE_NORMQ=1`) was Q8/dp4a-shaped:

- The Q8_0 fused launches (fused2/fused3/dual) all sit behind `q8_fused_params`, which
  refuses under `MEMRA_MMVQ=0` — so on a Q8_0 model both sides of the comparison fall to
  the same dp4a class and bit-identity holds.
- The three NVFP4 dual doors never consulted `mmvq_supports`:
  - `matmul_pre_dual_noscale` (m=1 gate+up / beta+alpha `qmatvec_nvfp4_mmvq_dual_mr2*`)
  - `matmul_decode_exact_dual` (verify t=2..4 dual)
  - `matmul_decode_exact_dual_pre` (verify t=2..7 dual, pre-quantized)

  Under `MEMRA_MMVQ=0` the oracle (`decode_step_h`) kept dispatching the MMVQ-family dual
  (32-thread warp reduce) while the batched body fell to dp4a (128-thread two-level
  reduce) — mixed kernel classes, FP-composition divergence (maxdiff class), not a real
  batching bug. This is why strict FAILED on ANY NVFP4 model at pristine trees.

## Fix

Each NVFP4 dual door returns `None` when `mmvq_supports(QT_NVFP4)` is false — the same
decode-parity/FP-order law the singles have enforced since 2026-07-07 (`batched iff MMVQ`).
The DEFAULT env (MMVQ on) is dispatch-unchanged: door eligibility is identical when the
env door is open. No new flag was needed — the existing `MEMRA_MMVQ` diagnostics seam now
pins the NVFP4 fused arms like it always pinned Q8_0's.

local-ci gains the NVFP4 strict arm (`--mode strict` B=4 equalized on the q9 NVFP4-MTP
artifact); it previously ran config-only with a doctrine note documenting this exact gap.

## What strict now pins (NVFP4, equalized env)

Both sides ride: dp4a matvec singles (no mmvq, no dual_mr2, no batched-dual), unfused
norms/quantize (`MEMRA_NO_FUSE_NORMQ=1`), same class per (dtype, m). Gate1 = bit-identity
B=1 vs `decode_step_h`; gate2 = B=N streams vs isolated `decode_step_h`; gate3 unchanged.

## PASS/FAIL table

| arm | mode | pre-fix | post-fix | log |
|---|---|---|---|---|
| q27 NVFP4 (`Qwen3.6-27B-NVFP4-Q4_K_M-mtp`) | strict B=4 equalized | FAIL (gate2 seq 2 diverged @ step 6) | ALL GREEN | `repro.log` / `strict-q27-postfix.log` |
| q9 NVFP4-MTP (`Qwen3.5-9B-NVFP4-MTP`) | strict B=4 equalized | FAIL (train-HEAD receipt: gate1 maxdiff 1.639e-1, servepath-p2) | ALL GREEN | `strict-q9-postfix.log` |
| q9 NVFP4-MTP | config B=8 | ALL GREEN | ALL GREEN (unregressed) | `config-q9-postfix.log` |
| Q8_0 9B (`ornith-1.0-9b-Q8_0`, MEMRA_Q8RP=1) | config B=8 | ALL GREEN | ALL GREEN | `config-q8-postfix.log` |
| Q8_0 9B | strict B=4 equalized | ALL GREEN | ALL GREEN | `strict-q8-postfix.log` |
| q9 canary (`MEMRA_GATE_CANARY=1`) | config | FAIL 5/6 early | FAIL 5/6 early (teeth intact) | `canary-config-q9.log` |
| q9 canary | strict equalized | — (arm did not exist) | FAIL (gate1 BIT-DIFF maxdiff 1.434e1 @ step 1) | `canary-strict-q9.log` |

The canary strict FAIL proves the new NVFP4 strict arm is not vacuous: a single
wrong-token perturbation still trips bit-identity immediately.

## Battery

- `kernel-check`: ALL GREEN (`kernel-check.log`)
- `run-gen` argmax: MATCH on q9 NVFP4-MTP and q27 NVFP4 (prefill/decode + batched-prime,
  `run-gen-q9.log`, `run-gen-q27.log`)
- `run-spec` K=1..8: self-consistency PASS 8/8 on q9 NVFP4-MTP (`run-spec-q9-k1-8.log`)
- `tools/local-ci.sh` (correctness stage, includes the new NVFP4 strict arm + serve-smoke):
  `local-ci.log`
