# SSM + Attention PREFILL Kernel Tuning Plan (the 27% lever beyond the GEMMs)

The three hybrid-prefill kernels — `ssm_conv1d_silu` (11%), `fa_prefill` (10%),
`gdn_scan` (6%) = **27% of T=512 prefill** — have **never been tuned**. They are
correctness-first ports (`hybrid.cu:3` "ported from llama.cpp, simplified to single
sequence, all f32, no tensor cores"; `flash_attn.cu:47` "this is the CORRECTNESS-FIRST
FA assembly... throughput tuning is a follow-up"). This is the untapped prefill lever.

## 0. HONEST FRAMING vs the GEMM lever (read this first)

These kernels are **27% of prefill**. Amdahl caps the win hard:
- **2x on all three** (i.e. 27% → 13.5%) = **~13% pp512 speedup**. Best realistic case.
- **2x on conv1d + gdn alone** (17% → 8.5%) = **~9% pp512**.
- The GEMMs are the other **~73%** and carry the **43x gap** (143 → 6240 pp512,
  `GEMM-PLAN.md:9`). A 2x there ≈ +73% pp512 — **5-7x more impact** than the whole
  SSM bundle. The GEMM tile (Task #10, landed) is and remains the dominant lever.
- **So:** this plan is the *second-order* prefill cleanup, worth doing because the
  kernels are genuinely untuned (one-thread-per-channel, serial-over-T, 1.2% occupancy)
  and the changes are cheap and low-risk — NOT because they move pp512 like the GEMMs.
  Do it **after** the GEMM tile is shipped; expect **+10-13% pp512**, not a multiplier.

## 1. RANK by (impact x feasibility)

| # | kernel | %prefill | current pathology | tuned target | per-kernel rel | pp512 contrib |
|---|--------|----------|-------------------|--------------|----------------|---------------|
| **1** | `ssm_conv1d_silu` | 11% | 1-thread/channel, serial T=512, no smem, strided HBM | 2D grid + vectorized, coalesced | **3-4x** | **+7-8%** |
| **2** | `gdn_scan_s128` | 6% | warp/col, serial T=512, 512 HBM round-trips/col | chunked-parallel scan (WY-form) OR fuse-from-conv | **2-2.5x** | **+3%** |
| **3** | `fa_prefill` | 10% | 1 warp/q-tile (1.2% occ), smem-O, no pipeline | register-O + GQA + 4-warp + cp.async | 1.5-2x | +4-5% |

**Cheapest big win = #1 conv1d.** It is pure occupancy + coalescing (no math change,
no tensor cores, no WY-form algebra), so it is the lowest-risk **and** the highest %
(11%). #3 fa_prefill has the same nominal % but its wins (register-O, GQA, multi-warp,
cp.async) are a 5-step rewrite already scoped in FA34-PLAN.md and gated on Task #9 — high
effort. #2 gdn_scan is the smallest slice (6%) and the parallel-scan rewrite is the
hardest algebra (chunked WY-form). So feasibility order = conv1d ≫ gdn_scan ≈ fa_prefill;
impact order = conv1d ≈ fa_prefill > gdn_scan. **conv1d wins both axes.**

---

## 2. Per-kernel concrete change

### #1 — `ssm_conv1d_silu_f32` (cu/hybrid.cu:24-41, launch lib.rs:1051-1052)

**Pathology.** `LaunchConfig::for_num_elems(conv_dim=8192)` (`lib.rs:1052`) → only
256 blocks of 32 threads. **One thread per channel** (`c = blockIdx.x*blockDim.x+tid`,
`hybrid.cu:27`) loops `for t in 0..512` (`hybrid.cu:33`) reading `xc[t+j]` from HBM with
NO time parallelism, NO shared memory, NO warp cooperation. 8192 channels × 512 serial
steps = the whole 11%.

**Change (occupancy + coalescing + vectorize):**
1. **2D grid over (channel, time-chunk).** Grid `(conv_dim/32, (T+TT-1)/TT)`, block
   `(32, 4, 1)` with `TT=128`. Each block-row handles 32 channels × a 128-token chunk;
   the 4-deep y-dim lets 4 channel-groups share the CTA → grid rises 256 → ~2048 blocks,
   1.2% → ~20% occupancy.
2. **Smem-stage the d_conv-1 halo + chunk** (llama.cpp `ssm_conv_long_token_f32`,
   ssm-conv.cu:60-124): load `(d_conv-1 + TT)` cols per channel into smem once, then all
   TT outputs read taps from smem (no re-reading HBM per tap). Pad smem stride to avoid
   32-bank conflicts.
3. **float4 the output write** (`yc[t]`): process 4 contiguous t per thread → coalesced
   128-bit stores, kills the strided write.

Math identical (depthwise causal conv + SiLU), so argmax is bit-stable.
**Per-kernel rel: 3-4x. pp512: +7-8%.** Cheapest big win — ship first.

### #2 — `gdn_scan_s128` (cu/hybrid.cu:54-111, launch lib.rs:1068-1073)

**Pathology.** Grid `(H=32, 1, S_v/COLS=32)`, block `(32, 4)` — one column per
block-thread, then `for t in 0..512` (`hybrid.cu:71`) runs the recurrence **serially**:
each iter reads `q_t,k_t,v_t,g_t,beta_t` from HBM (`hybrid.cu:72-76`) and does two
`warp_reduce_sum` (`hybrid.cu:88,98`). 512 full HBM round-trips per column, zero
inter-timestep parallelism.

**Change — two options, ranked:**
- **(A, cheaper) Fuse conv1d→gdn input read.** The repack `qkv_to_gdn_repack`
  (`hybrid.cu:238-256`, launched `hybrid_forward.rs:207`) materializes q_g/k_g/v_g to
  HBM, then gdn_scan re-reads them 512×. Stage the per-(h,t) q/k/v rows into smem at the
  chunk boundary so the t-loop reads smem not HBM. **~1.3-1.6x, low risk, no algebra.**
- **(B, bigger) Chunked-parallel scan (WY-form).** Partition T into CS=64 chunks
  (8 chunks for T=512): per-chunk cumsum of g_log → decay mask, intra-chunk KKᵀ tile,
  `solve_tri` forward-substitution in smem, then sequential **cross-chunk** state
  recurrence (only 8 serial steps, not 512). Reference: `PHASE1-HYBRID.md:220`,
  `delta-net-base.cpp:16-287`, FLA `gated_delta_rule/chunk.py`. Tiles 64×64 →
  `mma.sync.m16n8k16`. ~15x fewer serial ops in theory (8 chunk-steps vs 512), but
  smallest slice (6%) and hardest correctness — argmax-risky. **~2-2.5x if done right.**

**Recommendation: do (A) first** (cheap, composes with conv1d fusion), reserve (B) for
if gdn_scan profiles hotter than projected. **Per-kernel rel: 1.5x (A) / 2-2.5x (B).
pp512: +3%.**

### #3 — `fa_prefill_f32` (cu/flash_attn.cu:256-423, launch lib.rs:915-918)

**Pathology.** Grid `(ceil(T/16), n_head, 1)`, block `(32,1,1)` — **one warp per
16-row q-tile**, 1.2% occupancy. O accumulator lives in smem `sO[16][256]` f32
(`flash_attn.cu:251,276`) — every KV block does an **unconditional** 16×256 smem
read-modify-write rescale (`flash_attn.cu:378-381`). K/V re-staged per Q-head despite
GQA=4 (`flash_attn.cu:267` kv_head, 4 Q-heads share one kv_head but each restages).

**Change (already fully scoped in FA34-PLAN.md P0-P5, Task #9):**
- **P0 register-O** (FA34-PLAN.md:33): move `sO` from smem to per-lane registers; kills
  the 16KB smem R/M/W per KV block. **2-3x alone** — the dominant fix.
- **P1 GQA reuse** (FA34-PLAN.md:41): `grid.y = n_head_kv(8)` not `n_head(32)`, inner
  loop over 4 Q-heads sharing staged K/V → 3.3x less K/V HBM traffic.
- **P2 4-warp CTA** (FA34-PLAN.md:47): NTHREADS 32→128, BLOCK_Q 16→64 → 1.2%→25% occ.
- **P4 cp.async double-buffer** K/V (FA34-PLAN.md:60), gated on FA_KV_FP16.

This is a multi-step rewrite (effort), not a launch tweak — that is why it ranks below
conv1d on feasibility despite equal %. **Per-kernel rel: 1.5-2x. pp512: +4-5%.**

---

## 3. GATE (mandatory, every kernel, every step)

1. **End-to-end argmax — the authoritative gate** (`GEMM-PLAN.md:111`,
   `kernel_check.rs:631`): `run_dense` (qwen3 = **268**), `run_hybrid` (qwen35 = **271**),
   35B-A3B MoE (**1178**). All three MUST hold exactly. conv1d/gdn touch the qwen35
   (271) path; fa_prefill touches 268 + 271. Any mismatch = revert.
2. **Per-kernel rel vs current** (kernel_check pattern, slack 2.5e-3): each tuned kernel
   diffed against the live f32 kernel on a fixed input. conv1d/gdn are math-identical →
   expect rel < 1e-6; fa_prefill register-O is exact, cp.async-bf16 path ≤ 2.5e-3.
3. **pp512 tok/s** (Task #3 beat-benchmark): from current; gate that the bundle moves
   pp512 by the projected **+10-13%** (NOT a multiplier — see §0).

## 4. SUMMARY

1. conv1d / fa_prefill / gdn_scan = 27% of T=512 prefill, NEVER tuned (correctness-first
   ports: hybrid.cu:3, flash_attn.cu:47). 2x on all = ~13% pp512 — second-order vs the
   GEMM 43x lever (~73% of prefill); ship after GEMM tile.
2. Rank (impact x feasibility): **#1 ssm_conv1d_silu (cheapest big win, 11%, 3-4x)** >
   #2 gdn_scan (6%, 2-2.5x) ≈ #3 fa_prefill (10%, 1.5-2x rewrite).
3. conv1d (hybrid.cu:24-41, lib.rs:1051): 1-thread/channel serial-T → 2D grid (conv_dim/32,
   T-chunks) + smem halo (ssm-conv.cu:60-124) + float4 stores. No math change. 3-4x.
4. gdn_scan (hybrid.cu:54-111, lib.rs:1068): serial-T → (A) fuse conv→gdn smem read (1.5x,
   cheap) or (B) chunked WY-form scan CS=64 (2-2.5x, PHASE1-HYBRID.md:220, argmax-risky).
5. fa_prefill (flash_attn.cu:256-423, lib.rs:915): 1 warp/tile (1.2% occ) + smem-O → P0
   register-O (2-3x) + P1 GQA + P2 4-warp + P4 cp.async (FA34-PLAN.md P0-P5, Task #9).
6. GATE: argmax 268/271/1178 exact (GEMM-PLAN.md:111) + per-kernel rel ≤ 2.5e-3 vs current
   (kernel_check.rs:631) + pp512 moves +10-13%.
7. Cheapest-big-win projection: conv1d alone = +7-8% pp512 at lowest risk (pure occupancy,
   no algebra, argmax bit-stable).
8. Honest ceiling: the whole bundle is +10-13% pp512. The GEMMs remain the real lever.
