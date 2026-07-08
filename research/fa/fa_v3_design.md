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
