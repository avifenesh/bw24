# Engine techniques study — FA2 / FlashInfer / llama fattn (2026-07-22)

Owner directive: learn from the engines, not bench them; anything fitting more than 12B lands
engine-wide. Sources read at file:line depth by three parallel agents (FA2 @ Dao-AILab HEAD,
FlashInfer HEAD, llama c818263f2). Calibration numbers these must explain (box 5090, exact 12B
shapes, T=1736 nh=8 MQA): FA2 hd256 causal 0.218 ms/layer, hd256 native-window 0.111; torch-SDPA
hd512 0.826; ours: fa_w ~1.0 (5-9x off), fa512_sp ~2.5-3.3 (3-4x off).

## Field facts (three-way confirmed)

- **No bespoke bf16 sm120 prefill exists anywhere.** FlashInfer's sm120/ dir is NVFP4-only;
  blackwell/ is sm100 TMA/tcgen05 (inapplicable). Dense bf16 prefill on a 5090 = generic FA2
  class everywhere: cp.async + mma.sync m16n8k16 + ldmatrix — OUR instruction class.
- **f32 accumulation is the field standard**: FA2 f32 everywhere (scores/softmax/O); FlashInfer
  f32 QK-accum + f32 O/denominator always; llama f32 KQ/softmax — llama alone drops P and the
  P@V accumulation to f16 (its ~extra speed and its accuracy departure).
- **hd512 on 99KB parts has ONE recipe** (llama + FlashInfer independently): V time-shares K's
  smem (alias after QK consumes K), head-dim-chunked staging, O/VO split across warps,
  Q-in-smem re-read (not registers), small Q tile (32 rows, 1 Q-warp class).

## Mechanism table

| # | mechanism | source (file:line) | explains | scope | FP order |
|---|---|---|---|---|---|
| 1 | cp.async K-ahead/V-current schedule: V-copy overlaps QK-mma, next-K copy overlaps softmax+PV; single-buffered with fences | FA2 flash_fwd_kernel.h:305-339 | bulk of hd256 5x | engine-wide pp bodies | preserved |
| 2 | Bc=64 keys/tile (ours 32); Br=64 rows/head at hd256 (96KB, 4 warps — the sm120 branch FA2 itself picks) | FA2 launch_template hd256; kernel_traits | tile-efficiency share | engine-wide | **changed** (softmax update order) → battery |
| 3 | boundary/interior loop split: three-region bounds (causal top, window bottom, diagonal); only boundary tiles run mask compares, interior compiles mask-free | FA2 fwd_kernel:298-429 + FlashInfer prefill.cuh:2788-2802,1435 | window 9x + ALU waste | engine-wide (all SWA + causal) | preserved |
| 4 | smem XOR swizzle for conflict-free ldmatrix | FA2 kernel_traits:70-109; FlashInfer sparse_mla_sm120/arch/ldmatrix:76-86 | hd256 share | engine-wide helpers | preserved |
| 5 | tall per-head M-tile beats head-packing for prefill (FA2 packs heads only at decode; window keys amortized ~13x better per row than our 16-row g4) | FA2 flash_api.cpp:406; launch grid | window 9x share | engine-wide | preserved |
| 6 | hd512 fit: V-aliases-K smem + nbatch chunking + Q-in-smem + O chunked combine + occupancy-4-at-small-ncols | llama fattn-mma:1178,1917-1927,77-80; FlashInfer prefill.cuh:105-117,2811 | hd512 3-4x | hd512 stamp (31B too) | preserved (f32 port) |
| 7 | KV-split parallelism + online-softmax merge (stream-k w/ seam fixup, or chunked split+combine); fixes MQA grid starvation (nkv=1 collapses head-parallelism) | llama fattn-common:1116-1180,720-909; FlashInfer scheduler.cuh:634-668 | depth scaling, tail waves | engine-wide (prefill + later decode) | merge = own numeric config |
| 8 | KV_max mask pre-scan: one cheap kernel records first live KV block per q-tile; whole-tail skip at 256 granularity, gated to prompts >=1024 | llama fattn-common:664-716,1091 | window + causal tails | engine-wide | preserved |
| 9 | GQA ncols2 packing WITH mask (llama prefill packs up to 8 heads/CTA sharing K/V + one mask row) | llama fattn-mma:561,1221-1231 | alt to #5 when rows scarce (short prompts) | engine-wide | preserved |
| 10 | scale folded into Q at load (no softmax multiply); FA2 folds scale*log2e into one FFMA before exp2 | llama :1202; FA2 softmax.h:66-92 | minor ALU | engine-wide | changed (fold) → battery w/ #2 |
| 11 | L2 evict-first cache policy on streaming KV loads; .L2::128B prefetch cp.async variant | FlashInfer sparse_mla_sm120/arch/cp_async.cuh:50-91 | streaming efficiency | engine-wide | preserved |
| 12 | f16 P + f16 P@V accumulation (llama's remaining speed delta; the ONLY class-changing lever) | llama fattn-mma:927-931,1034 | last llama delta | opt-in door only | **class change** |
| 13 | sm_120 HAS TMA (cp.async.bulk + mbarrier) — warp-specialized producer/consumer possible | FlashInfer sparse_mla_sm120/arch/cp_async.cuh:64-84 | future | research | n/a |

## Port order (ROI on measured gaps, exactness first)

- **P1 — hd256 shared pp body rebuild** (mechs 1,3,4,5,10; then 2 as its own gated config):
  target fa_w 1.0 -> ~0.2 ms/layer class. Serves every SWA/causal model (gemma 12/26/31/E4B,
  qwen). Expected 12B pp: +6-8% (fa_w 40ms -> ~10ms).
- **P2 — hd512 stamp on the field recipe** (mech 6, f32 port): fa512 ~26 -> ~10ms class,
  +2-3% 12B pp, serves 31B globals too.
- **P3 — KV-split + merge** (mech 7): depth-prompt scaling + MQA occupancy; measure after P1/P2.
- **P4 — f16-P/V door** (mech 12): only if P1-P3 leave a llama gap; opt-in, full battery.

P1+P2 sized from references: ~45-50ms/prime recoverable at exact class -> projected 12B pp
0.966x -> ~1.05-1.08x llama BEFORE any class change. Numbers to be measured per lever, laptop
idle-gaps, interleaved A/B, battery per landing.
