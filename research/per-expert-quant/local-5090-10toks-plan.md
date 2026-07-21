# Hy3 local 5090: 4.6 → 10 tok/s plan

Baseline: 4.60 tok/s median (N=3 interleaved pairs, 2026-07-21 native ABI v2 receipt,
`evidence/local-5090-native-next-20260721/q2k-avxvnni-pair-win.md`). Target: sustained
10 tok/s on the same profile (Layer103.5 dual-NVMe view, 24 GB RTX 5090 laptop,
Intel 275HX, 60 GB RAM, 2x WD SN8000S).

## Measured per-token budget (from today's receipts, N=32 decode window)

217 ms/token total at 4.60 tok/s:

| stage | ms/token | evidence |
| --- | ---: | --- |
| GPU + engine glue (attn, dense, resident experts, sampler) | ~31 | window 6.955 s − exposed 5.956 s; MoE cache 100% hit, 0 H2D |
| CPU expert backend wall | ~190 | backend_wall 6.068 s / 32; 98.4% exposed at join |
| — phase_io (NVMe → RAM cache fill) | ~90 | 2.894 s / 32; 27.8 GB per 32 tok; RAM hit 54.97% at 20 GiB LRU |
| — phase_compute (275 expert-instances/token, 8 P-cores) | ~95–113 | 3.025–3.627 s / 32 |
| — prepare (per-call allocation churn) | ~6 | 0.179 s / 32 |

Structural facts (code-confirmed, `tools/bw24_cpu_experts.cpp`):

1. **io and compute are fully serial per call**: `bw24_cpu_moe_token_impl` runs
   `load_projection_weights` to completion (all misses read) before any dot product starts
   (`bw24_cpu_experts.cpp:1454→1458`). Cached experts wait on every miss in the call.
2. **16 threads (8 compute + 8 io) share P-cores 0–7**; the 16 Skymont E-cores idle.
   Skymont has AVX2 + AVX-VNNI — the paired Q2_K path would run on them.
3. Effective NVMe rate during io phase ≈ 9.6 GB/s aggregate across the mirror — near the
   device pair's practical ceiling. io time falls by reading fewer bytes, not faster.
4. GPU expert residency frozen at 5,285 blocks / 13.97 GiB; GPU serves ~79.5% of routed
   expert-instances (34,245 vs 8,809 CPU) with zero decode-window H2D.

Ruled out by receipts (do not revisit without new evidence):

- Adjacent-layer prefetch: predictor width-4 precision 57% / recall 28%
  (`window4-route-transition-analysis.json`) — wrong door.
- 32k MTP vocab trim: −20.7% plain (receipt, rejected).
- Speculative verification batching: flat (receipt, rejected, removed).
- Spec decode as currently measured: K=1 3.14 vs plain 3.72–4.60 — CPU wall per extra
  verified token exceeds acceptance gain. Re-stack MTP only after the CPU wall shrinks.

## Target arithmetic

10 tok/s = 100 ms/token. GPU ~31 ms stays; CPU section must fall 190 → ~65–70 ms with io
hidden under compute. That needs all three of: overlap (structure), wider compute (cores),
fewer bytes (cache + residency).

## Phase 0 — RESULTS (2026-07-21, all offline: simulation + CPU microbench, no GPU)

Simulator: `simulate_expert_cache_curve.py` (route trace `window4-routes.trace`, 50 decode
passes, real per-projection bytes from the plan manifest; calibrates against the live 20 GiB
anchor 55% / 0.87 GB-fills per token).

- **P0.B host-cache curve is FLAT — RAM lever DEMOTED.** LRU hit 46→55% and miss bytes
  0.858→0.715 GB/token across 20→36 GiB. The cold tail (~64 GB CPU-side bank) swamps LRU;
  doubling cache RAM buys −17% io bytes. Host cache stays 20 GiB. (Caveat: 50-pass trace
  underestimates large-cache steady state somewhat; even 2x the benefit stays weak.)
- **P0.C residency curve is the strong axis.** Sweeping HBM expert budget 13.97→18 GB:
  CPU load falls ~6.5%/GB in instances AND ~5%/GB in NVMe miss bytes
  (215.5→166.8 inst/tok, 0.858→0.711 GB/tok at +4 GB). Donors: kv-fp4 KV
  (K q8_0 + V q5_1 ≈ 145 KiB/token/80-layers → nvfp4 ≈ −38%; ~1.8 GB at 32k ctx),
  `BW24_MOE_VRAM_FRAC` 0.90→0.92+ (~+0.5–1 GB), and `enable_lm_head_fp32=True` in the
  source config — if lm_head sits in HBM at f32 (~2 GB), quantizing it is a third donor
  (verify on-GPU format at gate time).
- **P0.A compute attribution** (per-format microbench `BW24_CPU_NATIVE_BENCH`, production
  4-expert 4096x1536 shape, 8 P-cores, cache-hot, ms/call: Q2_K 1.06, IQ3_S 1.52,
  Q4_K 1.97, IQ4_XS 0.74, Q8_0 0.79, NVFP4 1.90; weighted by simulated CPU instance mix):
  **Q2_K 44% (already paired-VNNI), IQ3_S 30%, Q4_K 15%, IQ4_XS 11%.**
  Phase 4 kernel order: IQ3_S pair-decode port first, then Q4_K.
- **P0.D RAM ceiling**: 50 GB available, desktop RSS modest (swap 14 GB = cold pages).
  Moot for cache (stays 20 GiB), relevant only as safety margin.

Revised arithmetic: Phase 1 overlap → ~6.4; +E-cores → wall goes io-bound (~90 ms);
+kv-fp4/residency (−15–20% CPU load on both axes) → ~9.5; +IQ3_S/Q4_K kernels and prepare
pooling → 10+.

## Phase 1 — structural overlap (the big lever)

**Pipeline io ↔ compute inside the companion.** Per-expert readiness: compute an expert as
soon as its three projections are resident; cached experts compute immediately; misses
stream in behind. Preserve exact accumulation order (accumulate in expert index order at the
join, unchanged results). Serial 90+113 → max(io, compute)+ε.

Estimate: backend wall 190 → ~125 ms → **~6.3 tok/s**.
Gate: byte-identical output vs tokenwise control, packed-row oracle, interleaved N=32 pairs.

## Phase 2 — widen compute to E-cores

Extend workers beyond the 8 P-cores (OMP dynamic schedule already load-balances
heterogeneous cores). io threads move to E-cores; compute spans P+E. Leave 2–4 cores for
the desktop (quota rule). Estimate: compute 113 → ~65 ms; with Phase 1 the wall approaches
max(io ~90, 65) — io becomes the binding edge → Phase 3.

Estimate after phases 1+2: **~7.5–8 tok/s**.

## Phase 3 — cut bytes: host cache + HBM residency

- **3a host cache 20 → 28–32 GiB** (bounded by P0.D): hit 55% → ~70% (P0.B gives the real
  curve) → io 27.8 → ~18 GB per 32 tok → ~58 ms/token, hidden under compute by Phase 1.
- **3b kv-fp4 for KV → more resident experts**: the kf4 verdict was "capacity-only feature" —
  this is exactly the capacity case. Each freed GiB ≈ +378 blocks resident → fewer CPU-routed
  instances (est. 275 → ~240/token, −13% on both io and compute). Also retune
  `BW24_MOE_VRAM_FRAC` 0.90 → 0.92+ if headroom confirms.

Estimate after phase 3: **~9–10 tok/s**.

## Phase 4 — grind and re-stack

- Kernel: pair-decode port for the top qtype from P0.A (Q2_K pattern → Q3_K/NVFP4).
- Prepare pooling: reuse `ExpertRuntime`/activation buffers across calls (~6 ms/token).
- Re-stack MTP: once per-token CPU cost halves, K=1–2 at 55–60% acceptance flips profitable.
  Re-measure, don't assume.

## Discipline

Every phase: interleaved N=32 control/candidate pairs, cooled 55–56 °C starts, identical
token ids, post-freeze argmax MATCH, kernel-check ALL GREEN, run-spec K=1..8 PASS, raw logs
committed under `evidence/`. These are local-Hy3 numbers, never Qwen-board rows. Winners
merge, losers get a receipt and die (winners-only rule).
