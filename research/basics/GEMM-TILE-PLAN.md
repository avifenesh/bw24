# GEMM Tile / Accumulator Redesign — occupancy-bound → compute-bound

Target: `crates/bw24-engine/cu/qmatvec_gemm.cu` (kernel1 Q8_0/Q4_K/Q5_K lines 196-330;
kernel2 Q6_K/NVFP4 lines 456-574). Launch: `lib.rs:553-558`, `lib.rs:584-588`. Gate:
`kernel_check.rs:360-447`. HW: RTX 5090 Laptop, sm_120 / GB203 — **82 SMs, 65536 regs/SM,
100 KB smem/SM (99 KB opt-in), maxThreads/SM = 1536 (= 48 warps, NOT 64), 219 TFLOP/s int8**
(`research/sm120-empirical-capabilities.md:20-30,72`).

---

## 0. Correcting the framing before redesigning on it

The brief's "9/64 warps, ~11% peak" mixes two ceilings. GB203 caps at **1536 threads/SM = 48
resident warps** (`sm120-empirical-capabilities.md:27`), not 64. The 64-warp number is the
Blackwell *scheduler slot* count, not the achievable residency on this part. So the honest
current occupancy is **the CTA cap, not a warp-scheduler cap**: `__launch_bounds__(128,4)`
(`qmatvec_gemm.cu:332`) requests 4 CTAs × 4 warps = **16 warps/SM = 33% of the 48-warp max**.

But measured pp512 = 1264 ≈ 1264/6240 = 20% of llama, and ~11-13% of the 219 TFLOP/s int8
peak. So the kernel is **NOT hitting its requested 4 CTAs/SM** — `facc[16][4]` (64 regs) +
fragments + indices push the compiler to ~50-64 regs/thread, and at 64 regs/thread the regfile
allows only `65536/(64×128) = 8.0 → ` floor to whatever `__launch_bounds__` permits, but the
*real* resident count is throttled below 4 because the secondary `,4` min-blocks hint cannot be
honored when regs/thread × 128 × 4 > 65536 (64×128×4 = 32768 ≤ 65536 — so regfile is NOT the
wall at exactly 64 regs; the throttle is the **per-CTA work imbalance**: BN=128 means each warp
serializes **16 n-tiles** of mma+scale-fold with a single A-fragment, so only ~2-3 CTAs have
enough independent warps in flight to hide the mma+fold latency → effective ~9-12 warps doing
useful issue). **The bottleneck is latency-hiding starved by too few independent warps, which a
smaller-BN / more-CTA tile fixes — exactly the redesign target.** This matters: shrinking BN
helps because it raises the *number of concurrently-issuing CTAs*, not because the regfile is
literally full.

Skeptic's guard rail kept throughout: smaller BN trades weight-reuse (fewer tokens amortize
each decoded weight block) for occupancy. Below I find the knee, not the floor.

---

## 1. Chosen tile (with register math shown)

### Reference points
- **llama MMQ** (mmq.cuh:1003-1010, 3556): `MMQ_X` (token tile) 64-128, `MMQ_NWARPS=8`,
  per-warp accumulator split into `tile<16,8,int>` halves (4 s32-regs each), `granularity`
  picks `rows_per_warp` so facc stays ≤ ~32 regs/warp-lane → **8 warps × small-accum =
  high CTA count**. Key lesson: **8 warps/CTA, accumulator ≤ 4 s32-regs per active frag, more
  warps not bigger per-warp tiles.**
- **CUTLASS sm_120** (fp4_gemm_template_sm120.h:256-261): CTA 128×{64,128}, 1 CTA/tile,
  `StageCountAutoCarveout` (deep cp.async pipeline), warp-level m16n8k32, **占用 from pipeline
  depth not from many small CTAs** — but that's TMA-fed; bw24 has cp.async only, so bw24 cannot
  copy CUTLASS's 1-CTA model and must use llama's many-warps model.

### bw24's current tile (`qmatvec_gemm.cu:40-43,221`)
```
BM=64 (4 warps × 16 rows), BN=128, BK=32, NWARP=4, facc[BN/8][4] = facc[16][4]
```
`facc` alone = **16 × 4 = 64 s32/f32 regs/thread**. Plus afrag[4]+bfrag[2]+dacc[4] = 10,
plus rr/nn/ci/da/nt loop scalars ≈ 8. **~64-72 regs/thread → ~16-warp request, ~9-12 effective.**

### CHOSEN: BM=64, **BN=64**, BK=32, **NWARP=8**, 256 threads/CTA

This is the **two-axis** move (the knee), not a blind shrink:
1. **Halve BN 128→64** → `facc[BN/8][4] = facc[8][4] = 32 regs/thread` (halves the accumulator).
2. **Double warps 4→8** → BM stays 64 but now each warp owns **8 rows worth via 2 warps per
   16-row band**? No — keep WARP_M=16, so 8 warps cover BM=128. To keep BM=64 with 8 warps,
   instead **split the token axis across warp-pairs**: 4 warps cover rows [0..64) for tokens
   [0..32), 4 warps cover the same rows for tokens [32..64). This is llama's `ntx`/`nty`
   split (mmq.cuh:1007). Net per-warp work = **8 rows-frags? no**: each warp = 16 rows × 4
   n-tiles (32 tokens) → facc[4][4] = **16 regs/thread**.

Concretely, the chosen geometry:

| param | value | effect |
|-------|-------|--------|
| BM | 64 | 4 row-bands × 16 (unchanged output-row tile) |
| BN | 64 | tokens per CTA (was 128) |
| NWARP | 8 | 256 thr/CTA (was 128) |
| warp layout | 4 row-warps × 2 token-warps | warp `w`: rows `(w%4)*16`, tokens `(w/4)*32` |
| **facc** | **[BN_w/8][4] = [4][4] = 16 regs/thread** | **was 64 → 4× smaller** |

**Register budget math (the load-bearing numbers):**
```
facc[4][4]                = 16 regs
afrag[4] + bfrag[2]       =  6
dacc[4]                   =  4
loop/index scalars (rr,nn,ci,nt,da,g,cur,nxt) ≈ 12
                          ----
estimate                  ≈ 38 regs/thread  (vs ~64 today)
CTAs/SM by regfile        = 65536 / (38 × 256) = 65536 / 9728 = 6.7 → 6 CTAs/SM
warps/SM                  = 6 × 8 = 48 warps  = 1536 threads = maxThreads/SM (FULL)
occupancy                 = 48/48 = 100% thread occupancy (vs ~33% requested / ~20% real now)
```
Set `__launch_bounds__(256, 6)` to pin it (was `(128,4)`). If ptxas reports >40 regs and
caps at 5 CTAs/SM (40 warps = 83%), that is still a **2.5-4× occupancy lift** over today's
effective ~9-12 warps, so the tile is robust to a few regs of slop.

**smem (chosen) — verify ≤ 99 KB/CTA AND ≤ 100KB/SM ÷ CTAs:**
```
sW[3][64][32] = 3·64·32   =  6.0 KB    (NSTAGE=3, unchanged)
sA[3][64][32] = 3·64·32   =  6.0 KB    (BN halved → was 12 KB)
sWd+sWb[3][64]            =  1.5 KB
sAd+sAsum[3][64]         =  1.5 KB
                          --------
per CTA                  ≈ 15.0 KB
× 6 CTAs/SM              =  90 KB  ≤ 100 KB/SM  ✓ (kernel2 doubles weight buf: see §5)
```
smem is NOT the binding constraint at 6 CTAs/SM (90 < 100). The occupancy is now **thread-cap
bound at 48 warps — the desired compute-bound regime.**

### Why NOT BN=32 / facc[4][4] with 4 warps (the "blind shrink")
Findings' Option A2 (BN=32, 4 warps) gives facc[4][4]=16 regs too, but keeps **4 warps/CTA**,
so per-CTA weight-reuse drops to **4 tokens/decoded-block** (vs 16 today) — a **75% reuse loss
for the SAME accumulator size** the chosen tile gets at BN=64+8warps with only 50% reuse loss
on the *token* axis but recovered by 8 warps amortizing the decode 2× harder. The chosen tile
keeps **BN×NWARP/8 = 64×8/8 = 64 token-reuses-per-CTA**? No: weight block decoded once per CTA
per K-step is reused by BM rows × all 8 warps' tokens = 64 tokens. **Reuse = 64 tokens/CTA**
(was 128). That is a **2× reuse loss, not 4×** — the sweet spot. BN=32/4warp would be 4× the
CTAs but ¼ the reuse and only 4 warps to hide latency. Rejected.

---

## 2. Inner-K unroll depth

Current: **1 mma per K-step** (BK=32, one 32-block), `mma:load ≈ 1:1` (`qmatvec_gemm.cu:296-304`).
Findings' Option C1 (BK=64, 2 mmas/load) was correctly rejected *at the old 64-reg tile* because
holding afrag_lo[4]+afrag_hi[4] live cost +4 regs and pushed occupancy down.

**Now that facc dropped 64→16 regs, the register headroom EXISTS to add the inner-K unroll.**
Chosen depth: **BK_UNROLL = 2** (load `sW[..][0:64]`, `sA[..][0:64]`, issue 2 back-to-back
m16n8k32 mmas per smem fill, accumulate both into the same facc with their own per-32-block
scales). This raises `mma:load → ~1.6:1` (findings' own number) and is the single biggest
*compute-bound* lever once occupancy is fixed:
```
afrag_lo[4] + afrag_hi[4] = +4 regs    → ~42 regs/thread → still 5-6 CTAs/SM
sW[3][64][64], sA[3][64][64] = 12+12   = 24 KB smem/CTA + scales ≈ 28 KB
× 5 CTAs/SM                            = 140 KB > 100 KB  ✗  → drop NSTAGE 3→2 here:
sW[2][64][64]+sA[2][64][64] = 16 KB + scales ≈ 19 KB × 5 = 95 KB ≤ 100 ✓
```
So **BK_UNROLL=2 requires NSTAGE=2** (the deeper inner-K unroll itself provides intra-step
latency hiding, partially compensating the lost ring stage). Do this as **step 2**, gated
separately, only after the §1 tile lands and is measured — because it trades a cp.async stage
for mma density and the net is empirical. Do NOT unroll to 4: that needs BK=128 = 64 KB
weight smem alone, blows the budget, and the K=in_f/32 loop is already short enough (in_f=4096
→ 128 K-steps) that depth-2 saturates the issue pipe.

---

## 3. Warp-specialization / split-K — honest verdict at B=2-4

**Both NOT WORTH IT for this kernel.** The brief's findings already establish this; restating
with the chosen tile:

- **Warp-specialization (producer-consumer):** would dedicate ~2 of 8 warps to cp.async
  prefetch, 6 to mma. But the chosen tile already hides DRAM latency with the **NSTAGE=3
  cp.async ring** (`qmatvec_gemm.cu:266-290`) — the load is *already* async and overlapped; a
  producer/consumer split adds 2 barriers/K-step and *removes* 2 math warps from the 48-warp
  budget we just maximized. On sm_120 (no TMA, no mbarrier warp-roles in the ggml/bw24 layer)
  this is a net loss. **Verdict: no.**

- **Split-K / stream-K:** the win is filling idle SMs when the M×N tile grid is smaller than
  82 SMs. At **prefill T=512**, the grid is `(out_f/64, ceil(512/64)) = (out_f/64, 8)`. For a
  typical out_f=4096-12288 that is `64-192 × 8 = 512-1536 CTAs` — **already 6-19× oversubscribes
  82 SMs.** Split-K would only fragment K-reduction across CTAs that are already saturating the
  GPU, adding an atomic-merge pass for zero occupancy gain. **At the stated B=2-4 batch the
  prefill token count (T=512 per sequence) dominates — this is NOT thin-M; the M-tiling already
  fills the GPU.** Split-K helps *decode* (M=1-4), but decode keeps the dp4a matvec path
  (`lib.rs:425, m≥16` gate) — the GEMM never runs at small M. **Verdict: no, and the existing
  M-tiling is the correct scheduler.**

The honest lever ranking at B=2-4 prefill: **occupancy tile (§1) >> inner-K unroll (§2) >
swizzle (already a no-op per `qmatvec_gemm.cu:48-51`, revisit after tile) >> warp-spec / split-K
(zero or negative).**

---

## 4. Realistic projection: 1264 → ? pp512

Current 1264 pp512 ≈ **11-13% of 219 TFLOP/s int8 peak**, occupancy-bound at ~9-12 effective
warps. Multiplicative, with honest Amdahl (each lever shrinks as the prior removes its bucket):

| step | lever | mechanism | factor | pp512 | % peak |
|------|-------|-----------|--------|-------|--------|
| 0 | baseline | BN=128, 4 warp, facc=64reg | — | **1264** | ~12% |
| 1 | **§1 tile** BN=64, 8 warp, facc=16reg | 9-12 → 40-48 warps resident (3.5-4× occ); minus ~2× weight-reuse loss (more L2 traffic) | **×2.6-3.0** | **3300-3800** | ~33% |
| 2 | **§2 inner-K=2** (NSTAGE→2) | mma:load 1:1 → 1.6:1, compute density | **×1.3-1.45** | **4400-5400** | ~45% |
| 3 | swizzle (XOR @ stride 32) | conflict-free ldmatrix now that occupancy is the binder, not conflicts | **×1.1-1.2** | **4900-6200** | ~52% |

**Honest landing: ~4900-6200 pp512 = ~45-55% of int8 peak.** This **reaches striking distance
of llama's 6240 but likely lands just under it** (most-likely ~5200-5600). It does NOT cleanly
exceed 6240 in one redesign. What's still left after these three:

1. **Decode-off-critical-path for Q4_K/Q6_K/NVFP4** — `decode_block<QT>` ALU
   (`qmatvec_gemm.cu:116-160, 370-448`) still runs inline in `issue_load_stage`. Q8_0 is a
   memcpy (no decode) so Q8_0 hits the higher end of each band; the K-quants lose ~10-20% to
   inline unpack. Pre-decoding to an int8 staging buffer recovers it. ×1.1-1.3 **on K-quants only.**
2. **The ~896 GB/s memory roofline.** At 219 TFLOP/s int8 = 264 FLOP/byte (`sm120:72`), prefill
   T=512 with in_f=4096 weights re-read once per CTA is **compute-bound**, so the roofline is
   not the wall yet — but the ~2× weight-reuse loss from BN=128→64 raises L2/HBM weight traffic;
   if that pushes a layer memory-bound, the §1 factor lands at the low end (×2.6 not ×3.0). This
   is the skeptic's real risk and the reason BN=32 was rejected (it would push memory-bound).
3. **llama's 6240 includes its block-scale FP8/FP4 path on some layers.** Plain int8 (219) vs
   block-scale FP8 (381, `sm120:73`) is a 1.74× compute headroom bw24 leaves on the table —
   the Stage-C `mma.kind::mxf8f6f4.block_scale` path (already proven to execute on this silicon,
   `sm120:61`) is what *exceeds* 6240, and is explicitly out of scope for this tile redesign.

So: **this redesign closes the gap to ~0.8-0.95× llama and reaches ~half the int8 peak; parity/
beating 6240 needs decode-staging (K-quants) + Stage-C block-scale FP8 (a separate dtype pass).**

---

## 5. Exact `qmatvec_gemm.cu` changes (line anchors) + the gate

### A. Defines (`qmatvec_gemm.cu:40-43`)
```
#define BM 64        // unchanged
- #define BN 128
+ #define BN 64       // tokens per CTA: halve accumulator
- #define NWARP 4
+ #define NWARP 8     // 256 thr/CTA; 4 row-warps × 2 token-warps
#define WARP_M 16     // unchanged
```
Add after line 44: `#define WARP_NTOK (BN / 2)   // 32 tokens per token-warp band`.

### B. Warp→(rows,tokens) mapping in BOTH kernels (`qmatvec_gemm.cu:204`, `:464`)
```
const int warp   = threadIdx.y;          // 0..7
+ const int rwarp = warp & 3;            // 0..3 row band  → rows (rwarp*16)
+ const int twarp = warp >> 2;           // 0..1 token band → tokens (twarp*32)
```
Replace every `warp * WARP_M` (rows) with `rwarp * WARP_M` (`:296, :310, :325, :543, :555, :567`)
and every token base `nt*8` with `twarp*WARP_NTOK + nt*8`
(`:302, :311, :324, :549, :556, :568`). The B-fragment load `&sA[cur][nt*8][0]`
(`:302, :549`) becomes `&sA[cur][twarp*WARP_NTOK + nt*8][0]`.

### C. Accumulator + n-tile loop bound (`qmatvec_gemm.cu:221-225, 299, 320, 478-482, 547, 564`)
```
- float facc[BN / 8][4];        // BN/8 = 16
+ float facc[WARP_NTOK / 8][4]; // 32/8 = 4   ← the 4× register cut
```
All `for (nt = 0; nt < BN/8; nt++)` loops (`:223, :299, :320, :480, :547, :564`) →
`for (nt = 0; nt < WARP_NTOK/8; nt++)` (4 iters, was 8/16).

### D. `__launch_bounds__` on all 5 entry points (`:332, :338, :344, :576, :582`)
```
- __launch_bounds__(128, 4)
+ __launch_bounds__(256, 6)
```

### E. Launcher block/grid (`lib.rs:553-558` and `:584-588`)
```
- const BM: u32 = 64; const BN: u32 = 128;
+ const BM: u32 = 64; const BN: u32 = 64;
  grid_dim: ((out_f + BM-1)/BM, (m + BN-1)/BN, 1),
- block_dim: (32, 4, 1),
+ block_dim: (32, 8, 1),     // 8 warps = 256 threads
  shared_mem_bytes: 0,        // static smem ≈ 15 KB ≤ 48 KB cap — no opt-in needed
```
(`issue_load_stage`'s `r += NWARP*WARP_SZ` and `n += NWARP*WARP_SZ` cooperative strides at
`:229, :242, :485, :506` auto-scale with NWARP=8 — no edit needed, more threads = faster fill.)

### F. (STEP 2, separate commit) inner-K unroll — `qmatvec_gemm.cu:42, 212-213, 295-315`
```
#define BK 32  →  decode/load 2 blocks/step: sW[NSTAGE][BM][64], sA[NSTAGE][BN][64]
#define NSTAGE 3  →  2      (smem budget: 19 KB/CTA × 5 = 95 KB ≤ 100)
```
In the K-loop, `ld_A_s8(af_lo, &sW[cur][..][0], 64); ld_A_s8(af_hi, &sW[cur][..][32], 64);`
then two `mma_s8_m16n8k32` per n-tile, each scaled by its own (dw,da) 32-block pair into the
same `facc[nt][ci]`. K-loop stride becomes `g += 2`. Gate independently.

### THE GATE (non-negotiable, in order)
1. **Bit-equivalence** — `cargo run --bin kernel_check -- <gguf>`. The existing GEMM gate
   (`kernel_check.rs:360-447`, `qmatvec_gemm_raw` vs dp4a fast path, T∈{16,64,128,512},
   Q8_0/Q4_K/Q5_K/Q6_K/NVFP4) at **rel < 1e-3** (`:401,426,443`) MUST hold with no new FAIL.
   The redesign only changes **which lane owns which (row,token)** and the loop bounds — the s8
   mma inputs and the f32 scale fold are byte-identical, so rel must be **unchanged** from the
   1264-baseline run (≈1e-7). Any rel jump = the rwarp/twarp remap mis-indexed a fragment or
   the write-out (`:318-329, :563-573`) addresses the wrong (o,t). Catch it here, not at argmax.
2. **End-to-end argmax (authoritative)** — run the three checkpoints with `BW24_GEMM=1`:
   `run_dense` (qwen3) prefill argmax **268**, `run_hybrid` (qwen35) **271**, MoE 35B-A3B
   **1178** (`kernel_check`/`run_gen.rs` checkpoints, ROADMAP.md). All three MUST be unchanged.
   Decode (m=1) is unaffected (dp4a path, `lib.rs:425`).
3. **Perf** — `pp512` in `run_gen` must rise monotonically: step-1 tile should clear **>3000**
   (proves the occupancy lift bound), step-2 inner-K **>4400**, swizzle **>4900**. A step that
   does not move pp512 did not bind its target bottleneck — investigate before stacking the next.

### kernel2 (Q6_K/NVFP4) smem caveat
kernel2 (`:472-473`) keeps ONE 32-wide weight buffer (the lo/hi split is register-level,
`:544-545`), so its §1 smem is the same ~15 KB and §1 applies identically. For §2 inner-K,
kernel2 runs **2 mmas/32-block already** (`:551-552`); BK_UNROLL=2 makes it 4 mmas/step — keep
NSTAGE=2 and verify ≤100 KB before enabling (Q6_K/NVFP4 are attn_v/lm_head/MoE, the less-hot
path — acceptable to ship §1 only for kernel2 if §2 blows its budget).
