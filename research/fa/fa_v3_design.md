# fa_decode v3 design note — the d6257 lever (close35b lane, 2026-07-09)

Status: DESIGN + BASELINE ONLY. No kernel shipped this session (spec-p3 lever took priority and
shipped: BW24_SPEC_FUSED_T). This note captures the llama.cpp mechanism study + the measured
constraints so the v3 campaign starts warm.

## The measured fact base (JOB B row, 2026-07-08 + this branch)

- d6257 plain-decode residual is 100% FA: `fa_decode_vec_q_v2` grows 89.4 -> 734.7 us/tok
  (d512 -> d6257) while every other kernel is flat. llama node-traced FA pair per layer-tok:
  vec 8.2 -> 30.6 us + combine 8.5 -> 32.0 (their combine 3x OUR cost; our combine is cheaper,
  their vec 2.4x faster at depth). Net in-context deficit +223 us/tok at d6257 (~3.5% of the
  token — more than the whole 0.99x cell gap).
- Micro table on THIS branch (fa_v2_bench, 35B shape nh=16 nkv=2 hd=256, q8_0 K / q5_1 V,
  default config = FA_V2 on, N=1 micro — context races need nsys, see JOB B lesson 3):

  | t_kv | us/call | implied unique-KV GB/s |
  |---|---|---|
  | 512 | 16.4 | 29.0 |
  | 2048 | 38.6 | 49.2 |
  | 6257 | 59.6 | 97.5 |
  | 12288 | 96.5 | 118.2 |

  Unique-KV bandwidth is ~14% of the 858 GB/s wall at d6257 -> the kernel is ALU/latency-bound
  (staging dequant + syncs), NOT unique-byte-bound. That is the headroom.

## What llama's flash_attn_ext_vec actually does at this shape (source study, fattn-vec.cuh)

Dispatch: for quantized KV, vec is selected whenever Q->ne[1] <= 2 on Ada+ (fattn.cu:463-467);
the gqa>4 && t_kv>=8192 exclusion applies only to the f16/bf16 branch. ncols=1 at decode.

1. **GQA is NOT amortized** (the lane brief's "reads KV once per q-head-group" guess is WRONG
   for the vec kernel): `launch_fattn<D, cols, 1>` => ncols2=1, blocks_num.z includes
   gqa_ratio (fattn-common.cuh:1085,1175) — every q-head is a separate CTA that re-reads K/V.
   KV bytes are read gqa (=8 on the 35B) times, served from L2 (d6257 KV = 5.8 MB, L2-resident).
   ncols2 GQA-packing is the MMA kernel's trick, not vec's.
2. **int8-dp4a K.Q with register-quantized Q**: Q is pre-quantized to q8_1 in registers ONCE
   (fattn-vec.cuh:150-201), K rows are loaded as raw ints and dotted via
   vec_dot_q8_0_q8_1_impl -> dp4a, scale = K.d * Q.d (fattn-common.cuh:304-329). NO K dequant,
   no half conversion, 4x denser loads than f16 staging.
3. **Register online-softmax, zero smem KV staging, zero __syncthreads in the KV loop**:
   KQ_max/KQ_sum in registers (132-138), scores staged in a small smem KQ[] only for the V*P
   read-back. 4 warps/CTA, each warp = one KQ score cooperatively (32-lane butterfly), 128 KV
   tokens per CTA iteration; per-thread K load = 2 ints/token.
4. V (q5_1) IS dequantized (to the accumulator type) per token — the V path is dequant-to-float,
   only K rides pure int8.
5. Split-K: parallel_blocks from occupancy + a wave-efficiency loop (fattn-common.cuh:1147-80),
   tiles of 256 keys, strided (not contiguous) walk; combine = 256-thread block per q-head,
   O(parallel_blocks) — this is where their 3x-more-expensive combine comes from. KEEP OURS.

## The constraint that kills the naive vendor (already measured)

`fa_decode_vec_q_v2` REVISION 2 (flash_attn.cu ~2185): the first v2 cut vendored llama's
register streaming (no smem, each warp re-reads global) and measured **2x WORSE at depth**
(125.7 vs 65.1 us at d6257) — because our CTA packs gqa=8 warps that share ONE staged
smem tile (dequant once per CTA), while register streaming multiplies the *dequant ALU* and
loads x8. llama tolerates the x8 re-read because their per-byte cost is tiny: K needs NO
dequant at all (dp4a on raw bytes). Our staging pays: bf16 dequant of every K byte + smem
round-trip + 2 __syncthreads per 32-key tile.

## v3 candidate: the HYBRID (dp4a-K + staged-V)

Keep OUR frame (contiguous [t_lo,t_hi) split partition, [head][split] partial layout, our cheap
fa_decode_combine_f32, grid=(n_head_kv, n_splits), block=(32, gqa)):

- **K path vendored**: per warp, per key — raw q8_0 bytes from global (L2-resident, 8x re-read
  is what llama already proves affordable), lane l owns 8 consecutive quants (2 ints, block
  b=l/4, redundant d_b load per 4-lane group), acc_lane = dQ_b * dK_b * dp4a-pair, 32-lane
  butterfly => score. Q pre-quantized to q8_1 in registers once per warp (scale folded into Q
  before quantize, llama-style). Kills Phase A's K half entirely + halves smem + kills one of
  the two syncs' pressure.
- **V path kept ours**: smem-staged bf16 V tile, dequant ONCE per CTA shared by 8 warps
  (the REVISION-2 lesson — V has no int8 shortcut, its dequant is the expensive part).
- **Softmax kept v2's**: tile-batched bookkeeping (once per 32-key tile).
- NUMERIC CONFIG: int8-dp4a scores != bf16-roundtrip FMA scores => v3 is its OWN numeric
  config exactly like FA_V2 was (BW24_FA_V3, default OFF, own argmax baseline, decode+rows+dc
  twins must flip together, full battery: kernel-check rows-vs-loop bitdiff==0 within config,
  35B run-gen argmax, run-spec K=1..8 both models).
- Estimate of the prize: K staging is roughly half the Phase-A ALU + half the smem traffic;
  llama's whole-kernel depth advantage is 2.4x. If the hybrid lands even half of it,
  d6257 -223us/tok -> ~-100, which flips the 0.99x cell.

Micro harness ready: fa_v2_bench takes BW24_FA_V3 the same way (extend its config print);
compare at t_kv 512/2048/6257/12288 vs the table above + llama's node-traced 30.6+32 reference,
then in-context via depth_profile (JOB B lesson: micro tables do not settle kernel races —
confirm with the busy-marker windowed nsys before claiming the cell).

---

## BUILD RESULTS (lane/fav3, 2026-07-09)

Shipped: `fa_decode_vec_q_v3` / `_rows_v3` / `_v3_dc` behind `BW24_FA_V3=1` (default OFF, own
numeric config, all three twins share `fa_dec_v3_walk` -> bit identity within the flag by
construction). Host gate `fa_v3_active`: default q8_0/q5_1 KV formats + head_dim % 128 == 0.

### What the final kernel is (4 revisions, each measured)

- **rev-1 (the design's hybrid)**: dp4a-K on raw q8_0 bytes (Q register-quantized per warp,
  scale folded, one shared f32 scale per 32-elem block via group-amax shuffle; lane owns dpl
  CONSECUTIVE quants), staged-V smem tile kept (q5_1 recipe verbatim), v2 tile-batched softmax
  kept, first sync deferred after B2 (B1 never touches sV). 6257: 59.6 -> 55.2 us.
- **rev-2 (multi-key B1, llama's exact warp shape)**: 32/dpl keys in flight, one whole block
  per lane, log2(dpl) group reduce. EQUAL at depth, 12-19% WORSE at 512-2048 (8-deep dp4a chain
  + 17 serial loads per key lose to cross-key ILP when the grid is small). REVERTED.
- **rev-3 (A1-loads/A2-unpack pipeline around B1)**: WORSE everywhere (59.0 @6257): +24 regs
  and the unpack lands on the pre-sync critical path. REVERTED.
- **rev-4 (register-streamed V retry, wide loads + zero barriers)**: REVISION-2's lesson holds
  even fixed — 69.9 @6257 vs 55.3 staged. The gqa-packed smem V broadcast is the right
  mechanism on this shape. REVERTED. (Trick: mimicking the staged path's bf16 round-trip kept
  the output hashes IDENTICAL across rev-1/2/3/4 — structural variants validated instantly.)
- **rev-4b (SHIPPED)**: rev-1 + funnel-shift ALIGNED-u32 K loads (the qs alignment class
  ((34b+2+koff)&3 in {0,2}) is constant per lane across keys because k_tok_bytes%4==0 -> 2-3
  L1 transactions/key vs 4 u16; trailing word loaded only when the class needs it) + PAIRED B3
  smem reads (bf16x2, acc dim remap 2*lane+64*(i/2)+(i&1), store maps back) + V header d|m as
  one word. 6257: 46.9, 12288: 74.5.

### Micro table (fa_v2_bench, N=1 micro — same harness as the baseline above)

35B shape (nh16/nkv2/hd256):
| t_kv | v2 us | v3 us | delta |
|---|---|---|---|
| 512 | 16.3 | 16.4 (N=5 med, bimodal 26us clock mode hits both) | flat |
| 2048 | 38.7 | 36.7 | -5% |
| 6257 | 59.7 | 46.9 | -21% |
| 12288 | 97.0 | 74.5 | -23% |

9B shape (nh16/nkv4/hd256, gguf-verified — the "kv=8" in the brief was wrong): v3 wins every
depth: 19.0->16.9 @512, 44.7->39.7 @2048, 67.2->53.1 @6257, 127.8->98.8 @12288. No
n_head_kv gating needed.

### In-context (nsys depth-profile window, 35B d6257 eager, same-window A/B)

fa_decode kernel 1668.6 -> 1487.1 us/tok (-10.9%); fa_decode+combine 1922 -> 1749 (-9%).
The micro -21% halves in context (L1/L2 shared with the MoE kernels — the JOB B lesson again:
micro tables do not settle kernel races).

### e2e (fa-ab-bench, same-process A/B — cross-process tg128 reloads measured +-15% spread,
useless at this effect size; the new bin flips the flag between captured-graph rounds)

- 35B d6257: v2 150.5 -> v3 153.0 median (1.017x, 4 pairs both orders)
- 35B d512: 1.006x, d2048: 1.002x (flat, as micro predicts)
- 9B d6257: noise-bound, flat-to-positive — 3-pair run 0.978x, 6-pair run 1.036x (per-pair
  deltas -3.0/+10.8/-2.0/+2.3/-0.4/0.0 tok/s) — 9B FA share is small (hybrid full-attn
  minority layers); no regression

### Spec p3 daily A/B (35B, K=3 PMIN=0.4 PMIN0=1 frspec32768 trim, raw p3 prompt, NGEN=256)

Board-regime confirmed (v2 within 1.4% of the 192.8 board row). run-spec cross-process pairs:
- v2: 190.36 / 190.22 / 189.63 -> median 190.2 tok/s
- v3: 199.50 / 199.75 / 201.70 -> median 199.75 tok/s (**+5.0%**), self-consistency PASS on
  the real prompt every run (acceptance 72.5%)
- vs llama spec reference p3 201.7 (its own regime): 0.94x -> 0.99x — the last below-llama
  spec cell is at parity's edge under this flag.
- CONTAMINATION NOTE: one earlier v2 sample (50.4 tok/s) discarded — a foreign run-spec
  (another lane, PID 3644755) held the GPU; waited it out per protocol and re-ran clean.

### ncu facts (sudo, d6257)

Achieved occupancy 38% vs 83% theoretical — the vec grid (98 splits x 2 kv-heads = 196 CTAs)
starves 82 SMs; top stalls long_scoreboard 7.24/inst (global-load latency), short_scoreboard
2.93, mio_throttle 2.59, barrier 0.07 (negligible). GSUB (split the gqa warp-pack across CTAs,
pure scheduling, bit-identical) was built and measured NEGATIVE (55.3 -> 59.5 @6257 gsub=2:
warps/SM unchanged 2.4x8 = 4.8x4, V restage cost pure loss) -> REMOVED per flags doctrine.
sp32/sp128/sp256 split sweep: sp64 default optimal (combine growth eats smaller splits).

### Gate battery (ALL GREEN, 2026-07-09, within BW24_FA_V3=1)

- kernel-check 27B path ALL GREEN incl new `fa_decode_vec_q_v3(KVQ)` vs-CPU cases (rel ==
  vec's 4.75e-2/4.67e-2 at Tkv 128/257) + `fa_decode_rows_v3 vs per-row loop` bitdiff=0 x4
- run-gen argmax MATCH: 35B (1178==1178), 9B (268==268) — same tokens as the v2 baselines
- run-spec K={1,2,3,4,6,8} self-consistency PASS on 35B AND 9B
- graph-decode-gate BIT-IDENTICAL 256 steps, 30 buckets, on 35B AND 9B
