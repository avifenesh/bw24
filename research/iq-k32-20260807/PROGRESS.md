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

## 2. The k32 rewrite + the audit claim verified-and-corrected

Change (`cu/mmq_iq_experts.cu`): `vec_dot_mma` now issues ONE `m16n8k32.s8` per 32-value k
step where the k16 form issued two `m16n8k16.s8` — tile shapes `tile<16,8>`/`tile<8,8>`
(the mmq_q8_0.cu canonical k32 ABI), same ldmatrix loads, B pair merged into one
`load_generic` over the same 8 ints, fold arity halved (1 C tile / 1 dA load / 1 FMA per
element vs 2/2/2). Serves BOTH kernels in the file (expert-segmented + iq4xs-dense).
Rollback: `MEMRA_IQEXP_K16=1` build seam keeps the k16 form verbatim
(`-DMEMRA_IQEXP_K16_MMA`).

SASS census (the audit's own law): k32 object = 1536x `IMMA.16832.S8.S8`, k16 object =
3072x `IMMA.16816.S8.S8` — exactly the intended 2:1 instruction halving, correct opcodes.

**The audit's "candidate bit-identical" claim is REFUTED on real weights — and the reason
is instructive.** The s32 accumulator merge IS exact (the audit's evidence — both per-16
x_df slots of a 32-block hold the same value — is correct and re-verified in source). What
the audit missed is the f32 FOLD: k16 computes `dB*(C0*d + C1*d)` (two rounded products,
then an add), k32 computes `dB*((C0+C1)*d)` (one rounded product on the exact int sum).
Same real-number value, different rounding shape → last-ulp drift that accumulates across
the k-walk. Measured (md5-pinned binary pair, same commit, one flock hold, prime-logits
byte-compare via the new `MEMRA_PP_LOGITS` GGUF dump):

| model | differing bytes | logit maxdiff | rel (vs absmax) | argmax | top-10 |
|---|---|---|---|---|---|
| gemma26b (expert kernel) | 831999/1048576 | 1.66e0 | 9.3e-2 | MATCH | same set, tail order swaps |
| KAT (dense kernel) | 733381/993280 | 6.69e-1 | 5.4e-2 | MATCH | 9/10 shared, tail swap |

So the exactness class is **branch-(b)** (MEMRA_ST_E4M3_BLK pattern): bit-identity is the
wrong bar for a changed reduction shape; the kernel's own contract (file header: argmax
MATCH + spec self-consistency + closeness, never byte-identity vs dp4a) is the bar — and it
was already branch-(b) vs dp4a before this change. Receipt: `raw/logits-ab-verdict.txt`.

## 3. Gates (k32 build)

- kernel-check model-backed on KAT IQ4_XS: **ALL GREEN** — iq4xs-mmq vs dp4a rel <= 2.7e-4
  (bar 1e-3) on synth + real trunk tensor at T=16/64/128/512; fused act+quant 0 byte
  mismatch. `raw/kernel-check-k32-kat.log`.
- run-gen argmax: **MATCH** on gemma26b (prefill=decode=236786, batched-prime=tokenwise)
  and KAT (271, both gates). `raw/rungen-k32-{gemma,kat}.log`.
- run-spec K=1..8 on KAT (owntrim drafter, real 2048-tok prompt): **8/8 PASS** — every K
  token-identical to plain generate, acceptance 64.1% -> 22.3% (the normal K decay), plus
  the "=== SELF-CONSISTENCY PASS ===" aggregate line. `raw/runspec-k32-kat.log`.

## 4. Perf A/B — N=5 interleaved, ONE flock hold, adjacent alternating pairs

`run-ab.sh perf`: pp-only median-of-3 per point, 5 pairs per model, order alternating
(k16,k32),(k32,k16),..., both md5-pinned binaries from the same commit, one lock hold for
the whole battery. Thermal 60-63C across the window (per-row temp in perf-ab.jsonl).
Prompt = depth-2048-kat.txt (gemma pp2311 / kat pp2048).

| model | kernel class | k16 median (N=5) | k32 median (N=5) | delta | separation |
|---|---|---|---|---|---|
| gemma26b Q4_0 | expert-segmented | 6893.4 tok/s | 7248.5 tok/s | **+5.15%** | DISJOINT (min k32 7244.4 > max k16 6896.9), 5/5 pairwise |
| KAT IQ4_XS | dense-trunk | 3903.6 tok/s | 4013.5 tok/s | **+2.82%** | DISJOINT (min k32 3964.7 > max k16 3914.4), 5/5 pairwise |

Cross-check vs the share gate: gemma-bal mmq_iq share 17.8% x 0.296 = 5.3% predicted
ceiling — measured +5.15%, i.e. the tile realizes ~97% of its predicted e2e ceiling at
this shape. KAT 16.4% x 0.296 = 4.9% predicted, measured +2.82% (~58% of ceiling — the
dense kernel is less MMA-bound; its instances include cp.async-stall-heavy small-m
tiles). Both above zero with disjoint distributions; the k32 form wins clean.

Rows: `perf-ab.jsonl` (20 rows, cell iqk32-perf); raw logs `raw/perf-*-p*.log`.

## 5. Verdict: k32 is the default; the k16 seam stays for the NUMERIC class, not for perf

Winners-are-defaults: naked builds get k32 (no flag). Perf shows NO ambiguity — DISJOINT
distributions, 5/5 pairwise on both kernel classes — so by the perf clause alone the k16 arm
would die. It stays as a **build-time** seam (`MEMRA_IQEXP_K16=1`, `-DMEMRA_IQEXP_K16_MMA`)
for one reason: the swap is branch-(b), NOT bit-identical (§2), and the repo's discipline for
a numeric-class change (the `MEMRA_ST_E4M3_BLK` pattern the task itself cites; also
`MEMRA_MMQ_FP8BLK_PLAIN`, which kept its form seam even for a bit-identical swap) is that the
old arithmetic remains reproducible for A/B and for any downstream logit-shift investigation.
Zero runtime cost: the seam is an #ifdef, the naked binary contains only the k32 form
(SASS census: 1536x IMMA.16832, zero IMMA.16816).

Board impact: none — the tracked boards publish decode (tg) rows for these models, and this
change is prefill-only (m>=16 / t>=16; decode and spec-verify ride dp4a by the
dispatch-parity law). No `current-board.json` change, no regeneration needed.

Step-3.7-Flash: gets k32 automatically through `mmq_iq4xs_dense_kernel` (its IQ4_XS trunk is
the same dispatch class as KAT's; its expert path rides `moe_ffn_grouped`/`qmatvec_view` and
never reaches these tiles). AWS-box share measurement = precision on THE SKU, below.

## 6. Evidence

raw logs: `research/iq-k32-20260807/raw/` (nsys .nsys-rep binaries stay OUT of git —
CSV summaries + console logs are committed; reps parked in /tmp/iqk32-nsys).
