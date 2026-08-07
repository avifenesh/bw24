# lane/iq-experts-k32 — IQ-experts k16→k32 MMA tile rewrite (task #85)

**Mission:** the PTX audit (task #81, `research/ptx-audit-20260806/AUDIT.md` site 2) found
`mmq_iq_experts.cu`'s int8 MMA runs `m16n8k16.s8` where the int8 pipe is K-FREE on sm_120a
(k16==k32==16.06 cyc/warp-MMA) — a k32 tile does 2x the K-work per instruction, tile-level
1.42x measured (`ptx-audit-20260806/logs/k16-vs-k32-tileloop.log`). The audit calls the swap
"candidate bit-identical" (per-16 scale slots provably equal within each 32-block); this lane
VERIFIES that on real weights rather than inheriting it.

**Gate first (the task's own condition):** measure the e2e SHARE mmq_iq's kernels hold before
rewriting. If <3% on every model that matters → NOT WORTH IT, stop with receipts.

## Dispatch map (read from source, commit 45e98ad8)

The k16 `mma()` at `crates/memra-engine/cu/mmq_iq_experts.cu:157` is issued from `vec_dot_mma`,
which is shared by TWO kernels in the same file:

1. **`mmq_iq_experts_kernel`** (expert-segmented MoE prefill, IQ4_XS/IQ3_S/Q4_0):
   - qwen35moe-class (`moe_ffn_pairs`, hybrid_forward.rs:3448-3472): guarded by `use_mma`
     BUT the naked default `MEMRA_MOE_F16G` mode 2 (lib.rs:77) admits every layer whose three
     projections pass `f16g_proj_ok` — on q35/AgentWorld/KAT expert banks (IQ3_S/IQ4_XS/
     Q3_K/Q4_K/Q6_K, in_f%256==0) that is ALL layers, so the grouped-f16 sk visitor takes
     them and **mmq_iq_experts gets zero dispatches on the naked default** (the audit's
     "HOT (model-gated)" is stale for this class — f16g mode-2 rearb 2026-08-02 superseded it).
   - gemma4 MoE (hybrid_forward.rs:4971-5013): `MEMRA_GEMMA_MOE_MMA` != 0 (default ON),
     gemma f16g is env-explicit-only (`moe_f16g_gemma_on` default OFF) → **HOT on
     gemma-4 26B-A4B QAT Q4_0 prefill, gate/up/down all three** (t >= 16).
   - step35 (Step-3.7-Flash): sigmoid-router arch → denied from `moe_ffn_pairs`/`moe_ffn_dev`
     by predicate (hybrid_forward.rs:2316,2343); its MoE rides `moe_ffn_grouped` whose
     per-expert GEMM is `qmatvec_view` → `qmatvec_f32` — **the expert MMA kernel never
     fires on Step**.

2. **`mmq_iq4xs_dense_kernel`** (dense-trunk IQ4_XS, m>=16 prefill only):
   - `mmq_supports` (mmq_ffi.rs:529): `MEMRA_PP_IQMMQ` != 0 (default ON) + `MEMRA_IQ_FAST`
     + in_f%256==0 → **HOT on KAT-Coder's IQ4_XS trunk (142 dense 2-D tensors)** and on the
     Step-3.7-Flash IQ4_XS trunk (the kernel-check iq4xs-mmq gate names both artifacts).

So the share measurement targets: **gemma-4 26B-A4B** (expert kernel, local 5090),
**KAT-Coder** (dense kernel, local 5090), **Step-3.7-Flash** (dense kernel share on THE SKU,
AWS pair box). q35 naked default reaches neither kernel (verified by kernel presence in the
KAT/gemma captures — same engine build).

Both kernels are prefill-only (t>=16 / m>=16; decode and spec-verify ride dp4a by the
dispatch-parity law), so the "e2e" denominators that can move are the pp/prime numbers,
not tg.

## 1. E2E-SHARE GATE — measured, verdict GO

Method: `run-share.sh` — `MEMRA_PROFILE_GEN=1` + `nsys -c cudaProfilerApi
--capture-range-end=stop` (capture = prime + timed decode only, the 2026-07-10 window law),
under `flock /tmp/gpu5090.lock`, GPU verified idle first (only the allowed co-residents: the
332MiB embedding llama-server + a 394MiB idle gateway, both <1GiB and 0% util). Share =
mmq_iq kernel ns / total GPU kernel ns from `cuda_gpu_kern_sum`. Single capture per cell
(shares, not tok/s claims — and an earlier defective attempt reproduced all four shares to
0.01%, so the numbers are stable).

**Instrument lesson (first attempt 05:54Z, all five runs invalid as receipts):** `nsys -c
cudaProfilerApi` defaults to `--capture-range-end=stop-shutdown`, which TERMINATES the app at
`cudaProfilerStop()` — every run lost its "generated N tokens" line and one read rc=143.
Kernel sums happened to be complete, but died-cause-unknown discipline says no conclusion on
them. Rerun with explicit `--capture-range-end=stop`: all rc=0, full receipts.

| capture | shape | GPU kern total | mmq_iq time | share |
|---|---|---|---|---|
| gemma26b-bal | pp2311 + tg128 | 969.4 ms | 172.7 ms (90 inst, expert kernel) | **17.81%** |
| gemma26b-pp | pp4512 + tg16 | 716.5 ms | 317.3 ms (90 inst) | **44.29%** |
| kat-bal | pp2048 + tg128 | 1178.0 ms | 193.3 ms (141 inst, dense kernel) | **16.41%** |
| kat-pp | pp4096 + tg16 | 1093.4 ms | 383.0 ms (141 inst) | **35.03%** |
| q35-bal | pp2048 + tg32 | 551.4 ms | 0 ms (0 inst) | **0.00%** |

Run receipts (rerun, all rc=0): gemma bal pp2311 6799 tok/s + tg 41.26; gemma pp pp4512
6800; kat bal pp2048 3882 + tg 32.86; kat pp pp4096 4031; q35 pp2048 5416 + tg 17.61. All
argmax gates MATCH in-run. Thermal: 56-71C across the window (logged per point).

Arithmetic: at the audit's 1.42x tile ceiling, kernel share s converts to at most
s x (1 - 1/1.42) = 0.296 s e2e. gemma-bal 5.3%, gemma-pp 13.1%, kat-bal 4.9%, kat-pp 10.4%.
Above the 3% bar on BOTH dispatching kernel classes even at the balanced shapes → **GO**.
q35's 0% is a dispatch fact (f16g mode-2), not a counter-signal; Step-3.7-Flash dispatches
the same dense-kernel class as KAT (IQ4_XS trunk, sigmoid-router MoE never reaches the
expert kernel), so KAT is the local proxy and the AWS-box Step measurement is precision,
not gate-deciding.

## 2. Evidence

raw logs: `research/iq-k32-20260807/raw/` (nsys .nsys-rep binaries stay OUT of git —
CSV summaries + console logs are committed; reps parked in /tmp/iqk32-nsys).
