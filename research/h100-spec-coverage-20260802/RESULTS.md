# h100-spec-coverage: q35 drafter spec closes the flip caveat; q27 MTP spec is receipt-only (2026-08-02)

Lane `lane/h100-spec-coverage` (from `restructure/public-split` a70a13c2 — the v0.65 tip with
BOTH lane/f16g-default-rearb and lane/h100-flip-full merged). Rig: darklanes-bench H100 80GB
HBM3 (Mumbai). Box tree `~/memra` was the pre-flip-merge rsync (stale vs tip: 8 files + the two
merged research dirs) — re-rsynced to a70a13c2 (`SOURCE-COMMIT.txt`), `MEMRA_CUDA_ARCH=90a`
release rebuild 3m57s rc=0. Every GPU-touching process under `flock /tmp/gpu-h100.lock`; GPU
idle at session start (0 MiB, zero compute apps) and between phases (gpu/apps state logged
pre/post in every raw log). Session window 08:31-09:14Z, single sitting — all cross-arm
comparisons same-session.

## Gate first

`kernel-check` sm_90a on the rebuilt tip binary: **ALL GREEN rc=0, 240 OK, 0 FAIL, 12 SKIP**
(`logs/kc-speccov.log`; SKIPs are the known absent-model class — same set as the flip lane's
240-OK run).

## 1) q35 spec on Hopper — the flip lane's caveat, CLOSED

`research/h100-flip-full-20260802` shipped the Hopper mode-2 naked flip with run-spec
"ATTEMPTED, structurally unavailable" (IQ4_XS artifact `nextn=0`, drafter not on box; the
mode-2 spec class was carried only by 5090 receipts). The own-trim drafter is now staged:

- `~/models/draft-35b-owntrim-nvfp4head-q4blk.gguf`, sha256
  `ae5b7797cc10188bddd00d7e46394e6b8676c1d4e4c6768c8b7b3b10d8870b6a` — byte-identical to the
  5090 source (`/data/ai-ml/hf-models/qwen36-35b-moe/`), verified both sides before any run.

Battery: one `run-spec` process = plain oracle + K=1..8 (no `MEMRA_SPEC_K`), NAKED — i.e. the
flipped Hopper default (mode 2, direct loaders + deep tail), the exact class the caveat is
about. `MEMRA_MTP_DRAFT` head, board-2048 prompt (6400 chars -> 2048 tok), `MEMRA_NGEN=128`,
N=3 batteries (`logs/q35-spec-k1-8-r{1,2,3}.log`, table `q35-sweep-table.md`).

**Self-consistency: PASS x8, all three batteries (24/24), zero FAIL.** The flip lane's spec
gap on sm_90a is closed with on-box receipts. Plain oracle 187.0 tok/s median; prime 0.149s =
13.7k tok/s prefill — consistent with the flip lane's mode-2 board prefill (13.3k), i.e. the
drafter rides the flipped path, not a fallback.

Acceptance / speed, K=2..4 (medians of 3, acceptance greedy-deterministic and identical
across runs):

| K | acceptance | spec tok/s | vs plain |
|---|---|---|---|
| 2 | 50.0% (64/128) | **221.39** | **1.18x** |
| 3 | 37.8% (68/180) | 209.17 | 1.12x |
| 4 | 29.2% (69/236) | 178.98 | 0.96x |

Full K=1..8 in `q35-sweep-table.md` (K=1 1.09x, monotone decay past K=2; K=8 0.69x). Best
K=2 — same best-K as the 5090 receipts' class.

## 2) q27 MTP spec on sm_90a (v0.65 tip) — the v0.57 follow-up, measured and REFUTED

Artifact `/opt/dlami/nvme/models/Qwen3.6-27B-Q4_K_M.gguf` (MTP-baked, `nextn=1` in-file — no
drafter env). Same battery protocol, `MEMRA_NGEN=256`, N=3
(`logs/q27-spec-k1-8-r{1,2,3}.log`, table `q27-sweep-table.md`).

**Self-consistency: PASS x8 all three batteries (24/24).** Sweep (medians of 3):

| K | acceptance | spec tok/s | vs plain |
|---|---|---|---|
| 1 | 68.4% (104/152) | 96.44 | 1.04x |
| 2 | 54.9% (134/244) | **100.54** | **1.08x** |
| 3 | 42.2% (143/339) | 92.55 | 0.99x |
| 4 | 34.8% (149/428) | 82.12 | 0.88x |

Best swept K in {2,3,4} = **K=2**. The e2e cell (board-2048, `MEMRA_SPEC_K=2`,
`MEMRA_NGEN=512`, N=3; each rep = one process running plain oracle then spec — interleaved by
construction, same load, same clock; `logs/q27-e2e-k2-r{1,2,3}.log`):

| rep | plain dec | spec dec | prime s | plain e2e | spec e2e | ratio |
|---|---|---|---|---|---|---|
| 1 | 92.94 | 102.94 | 0.419 | 86.36 | 94.94 | 1.10x |
| 2 | 91.84 | 101.88 | 0.418 | 85.41 | 94.06 | 1.10x |
| 3 | 92.63 | 102.69 | 0.418 | 86.10 | 94.75 | 1.10x |
| **med** | **92.63** | **102.69** | | **86.10** | **94.75** | **1.10x** |

e2e = 512/(prime_s + 512/dec), the board formula; acceptance 273/478 = 57.1% every rep,
self-consistency PASS every rep.

**The v0.57 follow-up ("spec board row would read ~1.26x") does NOT survive v0.65 on this
box.** Two reasons, both visible in the receipts:

- The in-harness spec gain is only 1.08-1.11x now (v0.57-era receipts on darklanes-8x read
  1.30x at K=3 on this same prompt class). Plain decode moved 78 -> 93 tok/s since then —
  the decode-BW campaign sped up exactly the thing spec amortizes, so the verify-vs-decode
  price changed and the old best-K (3) is now a wash (0.99x). Stale-verdict law, again.
- run-spec's plain oracle (92.6 dec, 86.1 e2e) sits below the board harness's plain cell
  (103.7 dec, 95.5 e2e) — the spec harness carries per-token overhead the board harness
  doesn't. The honest absolute: spec-K=2 e2e **94.75** does not clear the PUBLISHED plain
  q27 row (**96**). A spec row would move the board number DOWN.

## 3) Best-vs-best: vLLM q27 FP8 + its MTP spec, same session

The published q27 H100 row is plain-vs-vLLM-plain (96 vs 73). vLLM's best config carries MTP
spec (v0.59 receipts, darklanes-8x: FP8 spec-k=3 e2e 140.9). Re-measured on THIS box, same
session as our spec arm: `bench_vllm.py --model Qwen/Qwen3.6-27B-FP8 --spec-k K --runs 3`
(board-2048 text, p2048/g512, `speculative_config {method: mtp, num_speculative_tokens: K}`),
K in {3,4,5} — K=3 was vLLM's swept best in v0.59 AND its sweep ceiling there, so K=4/5 are
included to not understate their best.

First attempt FAILED (both K, logs kept: `logs/q27-vllm-spec{3,4}-FAILED-nvrtc.log`): the
FP8+MTP spec path JIT-compiles flashinfer `fp8_blockscale_gemm_90` and the system CUDA has no
nvrtc dev headers — quoted cause: `fatal error: nvrtc.h: No such file or directory`. (The
plain-FP8 board runs never touch this JIT — that is why the board harness always worked.)
Fixed by pointing `CPATH`/`LIBRARY_PATH`/`LD_LIBRARY_PATH` at the vllm-env pip toolkit
(`nvidia/cu13/{include,lib}`) — no system or vllm-env mutation.

| arm | decode med | prefill med | e2e med | N |
|---|---|---|---|---|
| memra plain (published board cell, this box, v0.65) | 103.73 | 4820 | 95.5 | 5 |
| memra spec-K=2 (Q4_K_M, own MTP head) | 102.69 | 4888 | 94.75 | 3 |
| vLLM spec-k=3 (FP8 + MTP) | 142.32 | 14782.8 | 137.04 | 3 |
| vLLM spec-k=4 (FP8 + MTP) | **145.12** | 13411.9 | **139.1** | 3 |
| vLLM spec-k=5 (FP8 + MTP) | 141.56 | 13036.7 | 135.67 | 3 |

vLLM's best on this box is spec-k=4 (139.1 e2e; K=5 is past its peak — their K-curve
saturates exactly like ours, just from a much higher plain floor). Old-box v0.59 spec-k=3
(140.9) reproduces here within 3% (137.0) — the number was real, not that box.

**Best-vs-best: vLLM FP8+MTP spec-k=4 139.1 e2e vs memra's best q27 arm 95.5 e2e (the
PLAIN cell — our spec does not clear our own plain) = 0.69x. vLLM leads best-vs-best by
1.46x on q27/H100.** Spec-vs-spec is the same story: 94.75 vs 139.1 = 0.68x. The published
plain-vs-plain row (96 vs 73 = 1.31x) remains true and remains the honest row; the
best-vs-best gap is decode-bandwidth-shaped (their spec verify batch rides FP8 tensor-core
decode at 145 tok/s; our verify pays the same dp4a decode path plain pays) and the
decode-BW campaign owns it.

## Verdict: receipt, not row

- q35: the flip lane's caveat is closed — gate evidence (PASS x24), no board row implicated.
  Row-worthy as a GATE (the mode-2 spec class now has on-box sm_90a receipts), not as a number.
- q27: no spec row. Our spec e2e (94.75) is below our own published plain row (96); publishing
  a "spec row" would be publishing a regression. Best-vs-best belongs to vLLM's FP8+MTP arm
  (139.1 vs 95.5 = 1.46x their way) — the decode-BW campaign owns that gap; this dir is the
  honest receipt of its size as of v0.65.
- Session: all GPU work 08:31-09:14Z, single sitting, box swept (scratch/script/shim removed,
  GPU 0 MiB, zero compute apps at exit); `~/memra` left as the a70a13c2 rsync
  (SOURCE-COMMIT.txt) with its 90a release build; drafter left staged in `~/models` for
  future batteries.

## Files

`run-speccov.sh` (driver, phases kc/q35spec/q27spec/q27e2e/vllm — the committed copy is the
one that ran), `parse-sweep.py` (from research/hy3-spec-20260802, unmodified),
`q35-sweep-table.md`, `q27-sweep-table.md`, `receipts.jsonl`; raw logs under `logs/`:
`kc-speccov.log`, `q35-spec-k1-8-r{1,2,3}.log`, `q27-spec-k1-8-r{1,2,3}.log`,
`q27-e2e-k2-r{1,2,3}.log`, `q27-vllm-spec{3,4,5}.json`, `q27-vllm-spec{3,4,5}.log`,
`q27-vllm-spec{3,4}-FAILED-nvrtc.log`, `q27-vllm-spec{3,4}-FAILED-cpath.log`. Every log
carries pre/post GPU state (temp/power/clock/mem + compute-apps).
