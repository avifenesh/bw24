# sk-tail-form: the deep tail — 32x64x64 3-stage for sub-crossover groups (2026-08-02)

Lane `lane/sk-tail-form` (from `restructure/public-split` b9bd9d4c; kernel commit d55584bb).
Rig: RTX 5090 Laptop 24463 MiB sm_120a, platform_profile `performance`, `gpu-full-power on`.
Every GPU run under `flock /tmp/gpu5090.lock` (co-resident `llama-server --embedding` 332 MiB
allowlisted, inside every figure). Models: q35
`/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`, Ornith-35B
`/data/ai-ml/hf-models/ornith-1.0-35b-gguf/ornith-1.0-35b-Q4_K_M.gguf` (RESIDENT).
All perf claims interleaved process rounds (arms round-robin per rep, same box same hour,
temps 60-63 C across the battery); each pp value = the run-gen in-process median of 5 reps
(+1 warmup) — the sk-bm128 protocol.

## 1. What was built (mission #32's remaining sk rung)

The sk-bm128 H100 ncu pricing: under q35's routing skew the 32x64x32 2-stage tail form is
41.3 ms = 31% of the sk GEMM stage, and the x8/x16/x24 cross fine-sweep REFUTED tuning it
away — the priced next rung is a deeper tail form. Built here, picked by smem math BEFORE
building (receipts.jsonl row 1):

| candidate | smem | verdict |
|---|---|---|
| **BM=32/BN=64/BK=64 3-stage** | A 32x72 + B 64x72 = 13824 B/stage x3 + 2052 s_pre = **43524 B STATIC** (ptxas 43536) | **PICK** — under the 48 KB static limit (no opt-in), 2 CTA/SM on sm_120a (~100 KB/SM), 5 smem-wise on H100 (228 KB); same 32-row tile = zero extra padding, identical tile list (drop-in arms) |
| BM=64/BN=64/BK=64 3-stage | 57348 B | rejected: >48 KB opt-in, 1 CTA/SM on sm_120a, up to 2x padding on exactly the sub-cross groups (the refuted fine-sweep effect) |
| register double-buffered 2-stage | smem-neutral | rejected: keeps the 32-k global pipeline depth and per-32-k sync cadence — strictly weaker latency cover |

Two variants (the kquant-tile-loaders composition requirement):
`moe_f16g_sktail_kernel` (workspace f16) and `moe_kq_sktail_kernel<Q4_K|Q6_K>`
(direct-from-quant: A keeps the 3-stage cp.async pipeline, B is a single 64x72 tile with TWO
register-pipelined 16-value KqRaw windows per thread — 25104 B static). ptxas sm_120a:
94 / 101 / 129 regs, **0 spills** all three. Wired under the existing visitor shape split
(tail groups = m_e < `MEMRA_F16G_SK_CROSS`), default ON, `MEMRA_F16G_TAIL=0` rollback seam;
`in_f % 64 != 0` falls back to the 2-stage tail in-launcher (kernel-check's in_f=480 case).

**Numeric class: NONE — bit-identical by construction.** Each 64-k block runs the same four
ascending 16-k m16n8k16 f32-accumulate steps the 32-k form runs in pairs, on the same f16
operands, per output element.

## 2. Gates (all green)

- kernel-check **ALL GREEN, 0 FAIL** (`kernel-check-r1.log`): new f16g-sk arms
  `visitor-32-deep-tail` / `visitor-32-legacy-tail` vs grid-scan **maxdiff 0.00e0** x2 shape
  cases (incl the in_f=480 %64-fallback), explicit `f16g-sk-tail deep vs legacy 0.00e0`;
  f16g-kq-direct `all-32-deep-tail` / `all-32-legacy-tail` vs workspace **0.00e0** on q4_K +
  q6_K synthetic skew AND real Ornith weights (Q4_K gate_exps in=2048, Q6_K down_exps in=512).
- **argmax MATCH + token-sha identity, both arms** (`gates.jsonl`, `token-hashes.log`):
  q35 MEMRA_MOE_F16G=2 gen512 sha `e94b6553fde7b9a0` old == new; Ornith naked gen512 sha ==
  the cross-lane anchor `c0c12c3b350dc7f5` in EVERY run of BOTH arms (8/8 across gates + AB).
- **q35 guard** (naked x3 + x3 clean re-batch, reps 4-6 after box-clear): token sha
  `86dc5f7105a3716b` == the q4k-expert-prefill anchor **6/6 reps**; pp2048 4099.8-4108.6
  (kquant-lane cell 4090.1-4100.5 — flat-or-better, cross-session sha-anchored), gen512
  prefill 2510.9-2527.7 (prev 2493.9-2501.2). Note: rep2 gen512 decode measured 52.16 tok/s
  (all other reps 242-262) — single-run blip, cause unknown (pre-colbert window), sha still
  anchored; decode is not a claim of this lane. Reps 4-6 = the clean batch.
- run-spec self-consistency: q35 (F16G=2, owntrim draft, p2, NGEN=64) **PASS x8 (K=1..8)**;
  Ornith (owntrim draft, p2, NGEN=128) **PASS x8** on the clean re-run — both cover the
  K=1..4 mission gate (§4 for the first attempt's OOM).

## 3. 5090 perf — interleaved, same session (`ab.jsonl`, git=d55584bb, ranges disjoint)

Win condition here (5090): no regression + any gain. Both cells beat it outright.

**q35 board-2048 pp e2e, MEMRA_MOE_F16G=2, x5 process rounds:**

| arm | reps (tok/s) | median |
|---|---|---|
| old tail (MEMRA_F16G_TAIL=0) | 3608.1, 3592.5, 3591.5, 3596.0, 3595.7 | 3595.7 |
| deep tail (naked) | 3741.4, 3728.7, 3723.2, 3720.0, 3734.8 | **3728.7** |

**+3.7% e2e, zero overlap** (min new 3720.0 > max old 3608.1).

**Ornith-35B naked (direct kq tail), x3 process rounds:**

| leg | old tail | deep tail | delta |
|---|---|---|---|
| pp512 (gen512 prefill) | 3154.7 [3150.8-3156.1] | **3495.2** [3493.3-3502.6] | **+10.8%** |
| pp2048 (board pp-only) | 4777.9 [4777.6-4781.4] | **4908.5** [4907.8-4917.8] | **+2.7%** |
| decode tg128 | 209.75 | 209.78 | flat |

The small-m tail matters most at short prompts, as priced: pp512 +10.8%. Against the
kquant-tile-loaders same-session llama bar (pp512 3977.4): ~0.879x from 0.792x —
INDICATIVE ONLY (cross-session denominator, clock-drift-invalid); the owned claim is the
same-session old-vs-new delta. A same-session llama re-ratio belongs to the next bar check.

**Mechanism (nsys cuda_gpu_kern_sum, SINGLE RUN per arm — N=1, mechanism evidence not a
perf claim; q35 board-2048 F16G=2, PP_REPS=1, 234 sk launches + 6 kq Q6_K launches):**

| kernel slice | old | new | delta |
|---|---|---|---|
| tail (ws): sk32v -> sktail | 152.79 ms (653.0 us avg) | 109.26 ms (466.9 us avg) | **-28.5%** |
| tail (kq direct, Q6_K x6) | 4.44 ms | 3.47 ms | **-21.9%** |
| sk128v (same kernel both arms) | 183.60 ms | 183.35 ms | flat |
| whole sk GEMM stage | 336.4 ms | 292.6 ms | **-13.0%** |

## 4. Failures kept (evidence discipline)

- First o35b run-spec attempt: `Error: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of
  memory")` at draft load (`o35b-spec-k1-8-oom-colbert.log`). Concurrent-GPU state at
  failure: co-resident `colbert-2/.venv/bin/python` pid 2220706 holding 1390 MiB (started
  03:01:53Z, mid-battery; nvidia-smi capture in gates-console.log context). Not a lane
  regression — the same binary's Ornith runs (22.4 GB resident) were green all session.
  Clean re-run after the box cleared: see §2.

## 5. What the H100 pass (v0.65 three-arm probe) needs — measurement is NOT this lane's

The box-prep window owns the H100 run. To run this form directly:

- Arms: mode-2 old-tail (`MEMRA_F16G_TAIL=0`) / mode-2 deep-tail (naked) / cublas mode-1
  (`MEMRA_MOE_F16G=1`), q35 board-2048 pp-only, interleaved x5 rounds x3 arms round-robin,
  `MEMRA_F16G_SK_CROSS=32` (the swept H100 cross — re-sweep it if cores moved;
  stale-verdict law).
- The deep tail is sm_80-portable by construction (same cp.async/ldmatrix/mma.sync class,
  no wgmma/TMA). Smem: 43536 B static (f16g) / 25104 B (kq direct) — no opt-in anywhere;
  sm_90a expected occupancy 5 CTA/SM f16g-tail (smem- and reg-wise at 94 regs x128 thr),
  kq-tail 9 smem-wise / ~3 reg-bound (Q6_K 129 regs). The pricing target: the 41.3 ms
  sk32v slice of the 131.9 ms GEMM stage (nsys single-run, sk-bm128 receipts) — parity
  bar vs cutlass+h2f 112.4 ms; mode-1 keeps the Hopper default until beaten on-box. The
  5090 mechanism datum to compare against: tail slice -28.5% (§3) — if the H100 tail
  moves similarly (41.3 -> ~30 ms), the sk stage lands ~121 ms vs cutlass's 112.4 —
  closer but likely still short; measure, don't assume (wgmma-era form-sensitivity law).
- kernel-check on-box first (the f16g-sk + f16g-kq-direct sections carry the tail arms),
  then argmax before any timing — the k-chain identity makes any sha drift a hard stop.
- Every claim interleaved x5 on-box, including the cublas denominator (clock-drift law).

## Files

`run-gates.sh` (q35-ab | o35b-ab | q35-guard | spec; GUARD_REPS/SPEC_ONLY/SPEC_TAG re-run
seams), `run-ab.sh` (q35 x5, o35b x3); `gates.jsonl`, `ab.jsonl`, `receipts.jsonl`,
`token-hashes.log`; per-run logs `q35-f16g2-*`, `o35b-*`, `q35-guard-*`, `q35-ab-r*`,
`o35b-ab-r*`; `kernel-check-r1.log`; spec logs `q35-spec-k1-8.log`, `o35b-spec-k1-8.log`
(+ the kept OOM raw); consoles `gates-console.log`, `ab-console.log`; nsys mechanism
receipts `nsys-*` (§3 addendum).
