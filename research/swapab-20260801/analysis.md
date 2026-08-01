# SWAP_AB probe — small-M expert GEMM orientation for mmq_iq_experts (research lever #8)

Lane 4, darklanes-8x, GPU 4 (H100 80GB HBM3, driver 595.71.05, 39C / 1830MHz SM at measurement).
Probe question: vLLM/DeepGEMM-style SWAP_AB (compute C^T so the small token dim sits on the
tensor-core N slot) — does the analogous operand swap pay for memra's mma.sync m16n8k16.s8
expert MMQ (`cu/mmq_iq_experts.cu`) at q35's gate/up shape (512 out x ~65-pair groups x k=2048,
the (4,252)-grid 3.11ms kernel)?

**VERDICT: REFUTED — the swap measures 1.15x SLOWER than the shipped orientation at the target
m=65, and it is dominated at every m by a 5-line early-exit in the current orientation.**
Receipts: `swapab_bench.cu` + `run1.log` + `build.log` (this directory, box copy at
`~/lane4/research/swapab-20260801/`).

## 1. Current mapping (from the full kernel read)

`vec_dot_mma` issues `mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32`:

- **A operand (M=16, ldmatrix 16-row fragments)** = weight out-rows. `x_qs` smem tile,
  stride `MMQ_MMA_TILE_X_K=84` ints/row (64 data + 16 per-16 scale slots + 4 pad),
  loaded once per k-step into registers and **reused across all 8 j0 strips**.
- **B operand (N=8, `load_generic` 8-col fragments)** = tokens. `tile_y` smem
  (`block_q8_1_mmq` layout, 36 ints/token/half: 4 scale floats + 32 data ints),
  re-loaded per (j0, k01).
- Tile = 128 out-rows x 128 tokens x 256k per kb; 8 warps = 4 row-groups x 2 j-halves;
  each warp owns 16 (row-block n x j0-strip) quanta of 16x8 output.
- Scales: `dB[l%2] * (C0*dA[..][k+0] + C1*dA[..][k+1])` — per-32 weight scale replicated
  into two per-16 `x_df` slots, per-32 activation scale in `y_df` (`d4[4]` of q8_1_mmq).

**The key structural fact: tokens ALREADY sit on the 8-granular N axis.** SWAP_AB's premise on
sm_90 is a wgmma constraint (M fixed at 64, N variable in steps of 8 — small token counts
can't shrink M, so you transpose). mma.sync m16n8k16 has no such asymmetry to escape: the
128-token padding at 65-pair groups is a **tile-loop artifact** (the j0 loop runs the full
`mmq_x=128` unconditionally; post-inc4 the dead-column *gathers* are skipped, the dead-column
*mma* is computed and discarded), not an operand-shape artifact.

## 2. What a swap port would entail

(a) **ldmatrix layouts**: activations move to the A side — the per-token gather
(`pair_tok`-indexed, today 36 contiguous ints/token, 16B cp.async chunks = inc1's +34.4%)
becomes a repack into the 84-int-stride `x_qs` layout with per-16 scale-slot replication;
the cp.async chunking would need re-derivation (writes are no longer contiguous per token).
Weights move to the B side: all three dequant-at-load tile loaders (`load_tiles_iq4xs`,
`load_tiles_q4_0`, `load_tiles_iq3s`) re-targeted to the 36-stride y layout. The W staging
ring itself is layout-agnostic (raw superblock slices), so inc2 survives.

(b) **per-32 scales**: symmetric by construction — `x_df` per-16 slots take activation scales
(replicated from per-32), `y_df` per-32 slots take weight scales; the
`dB*(C0*dA0+C1*dA1)` epilogue is scale-source-agnostic. The multiply ORDER changes
(`d_act*(C*d_w)` -> `d_w*(C*d_act)`), so this is a new FP-order class: full argmax-gate
re-run required, no byte-identity claim available.

(c) **writeback**: C transposes to [tokens, out-cols] — `y[(pair)*out_f + col]` keeps working
with i/j roles swapped (row index becomes the M/get_i axis, `it*mmq_y` moves to the j side).
Mechanical, same instruction count.

Port size: two new tile-loader families + gather repack + transposed writeback + a fresh
correctness gate — a real port, which is why this probe ran the microbench instead.

## 3. The padding arithmetic — and why it overpromises

Mission arithmetic (MAC-count view), m=65 of 128:
- current token-axis waste: (128-65)/128 = **49.2%**
- swap (16-granular M): ceil16(65)=80 -> (80-65)/80 = **18.75%** waste, mma-work ratio
  80/128 = 0.625 (ideal 1.6x on the mma share)

Two structural corrections kill this before measurement:

1. **Warp imbalance under the swap.** The machinery assigns each warp a contiguous 32-row
   group (`i0=(ty/ntx)*32`, fragments at i0, i0+16). With tokens on M at m=65, warp groups
   0-1 (rows 0-63) stay FULL, group 2 is half-live, group 3 idles — the block's vec_dot wall
   time is the busiest warp, which does the same work as at m=128. The skip only pays when
   whole 16-row fragments die *within every warp's group*, i.e. hardly at all at m=65.

2. **The current orientation can already skip at 8-granularity without any port.** Each warp's
   j0 strips stride the whole token axis (j0 in {0,16,...,112}, warp offset (ty%2)*8), so
   an early-exit `if (j0+joff > j_max) continue;` skips dead 8-token blocks EVENLY across
   warps: ceil8(65)=72 -> 72/128 = 0.5625 of the mma work, better than the swap's ideal 0.625,
   load-balanced, ~5 lines, no layout change, byte-identical live-slot numerics.

## 4. Microbench (swapab_bench.cu)

Three arms, tile machinery verbatim from the kernel, identical plain-smem data paths
(orientation is the only variable; no cp.async — biases the mma/LDS share HIGHER than the
real long_scoreboard-bound kernel, i.e. conservative in the swap's favor):

- **A cur128** — shipped orientation post-inc4 (dead-col gathers skipped, full-tile mma).
- **B swap16** — tokens on A/M via ldmatrix, out-cols dense on B/N, 16-granular row skip
  (`live[n]` fragment guards), transposed writeback.
- **C cur-exit** — shipped orientation + the 8-granular j-strip early exit.

Shape: 252 groups x (512 out x m tok x k=2048), grid (4,252), 8 warps, per-32 scales both
operands, `__launch_bounds__(256,1)`; regs 244 (A) / 254 (B) / 251 (C), 0 spills; smem
79,872B; nvcc 13.1 `/usr/local/cuda-13.1` (`~/cuda-13.3.1` toolkit is headers-incomplete:
no `include/crt`), `-arch=sm_90a -O3`. Correctness: all arms vs fp32 per-32-grouped
reference, maxabs 1.788e-7 at every m (pure fp rounding; identical across arms) — the
swapped layouts are proven correct, this is a fair fight.

Medians of N=5 same-session interleaved reps (20 launches/rep, warmed; rep spread <= 0.1%;
GPU 39C/1830MHz; full table in run1.log), us/launch:

| m   | A cur128 | B swap16 | C cur-exit | swap/cur | exit/cur | ideal-swap | ideal-exit |
|-----|----------|----------|------------|----------|----------|------------|------------|
| 16  | 875.6    | 808.7    | 650.5      | 0.924    | 0.743    | 0.125      | 0.125      |
| 32  | 973.8    | 942.0    | 794.6      | 0.967    | 0.816    | 0.250      | 0.250      |
| 65  | 1187.6   | **1367.4** | **1110.5** | **1.151** | **0.935** | 0.625    | 0.562      |
| 96  | 1405.5   | 1718.0   | 1371.2     | 1.222    | 0.976    | 0.750      | 0.750      |
| 128 | 1614.6   | 2079.0   | 1649.2     | 1.288    | 1.021    | 1.000      | 1.000      |

Findings:

1. **The swap loses outright at the target shape: +15.1% at m=65**, and carries a ~29%
   structural overhead at equal work (m=128) — with tokens on A, the streaming weight tile
   rides the per-(j0,k01) re-loaded B path (36-stride, token-boundary breaks) and loses the
   ldmatrix-once/reuse-8x economics the weights enjoy today, plus per-fragment live guards.
   Its 16-granular skip only wins below m~48 (0.92 at m=16 vs ideal 0.125 — the warp-imbalance
   + token-invariant-share prediction of §3, confirmed).
2. **cur-exit dominates the swap at every m** (swap/exit 1.19-1.26x) and is the only arm that
   beats the shipped kernel at m=65: **-6.5%**.
3. **The mma-side token-proportional share is small even here**: cur-exit removes 7/16 of the
   warp quanta at m=65 for 77us -> the full mma+B-load+FMA pipeline is ~176us ~= 15% of the
   kernel. The real kernel is long_scoreboard-bound (ncu, round 46 addendum) with cp.async
   overlap the microbench doesn't model, so its realizable share is smaller still.

## 5. Projected end-to-end effect on q35 gate/up — the arithmetic

- Real gate/up kernel: 3.11ms, long_scoreboard-dominant; W movement + dequant
  (128 rows x 2048 k per CTA) is token-INVARIANT — no token-axis fix touches it.
- mma work at peak-ish rates is ~1-2% of the 3.11ms budget (1008 CTAs x 33.5M MACs
  ~= 68 GOP vs ~2 POPS-class int8 mma.sync); microbench upper-bounds the whole
  token-proportional pipeline at ~15% of kernel in a HARSHER (compute-heavier) regime.
- Swap, measured: **+15% kernel time at m=65 — a regression**, before paying the port
  (loaders x3, gather repack, FP-order regate).
- Best case for ANY token-axis padding fix (cur-exit, measured): -6.5% kernel in the
  microbench regime; on the real latency-bound kernel expect less. q35 prime is ~15% of the
  e2e wall and the gate/up form is a fraction of that (the down form (16,256) at SM 59.9%
  is the bigger half): ceiling ~= 6.5% x 15% x <=0.5 ~= **<=0.5% e2e — below the board's
  measurement floor.**

**REFUTED. Do not port SWAP_AB to mmq_iq_experts.** The round-46-addendum verdict stands:
the fix class for the tiny-per-expert-GEMM shape is expert-BATCHED grouped GEMM (CUTLASS
grouped int8 / the MEMRA_MOE_F16G successor lane), not operand orientation.

## Residue (actionable, separate from this probe's verdict)

The 8-granular j-strip early-exit (arm C) is a ~5-line, layout-preserving, live-slot
byte-identical change that helps exactly where mma-share is high. On gate/up it is e2e-noise
(above), but the **down form (16,256) runs at SM 59.9% short-scoreboard over the same
~65-pair groups** — the exit's ~44% mma-work cut is worth a measured in-vivo A/B there.
At m=128 the naive runtime check costs +2.1%; a port should template or branch-hoist it
(`j_max >= mmq_x-1` -> the unguarded loop) so full tiles pay zero. Flagged for the lane
owner's queue; not shipped here (this probe's deliverable is the orientation verdict).
