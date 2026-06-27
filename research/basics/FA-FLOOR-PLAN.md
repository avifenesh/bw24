# FA FLOOR — what bw24 must match (and exceed) on sm_120

Verified this machine, this session. RTX 5090 Laptop, **compute cap 12.0 = sm_120 = `GGML_CUDA_CC_BLACKWELL` (1200)**.
Model: Qwen3.5-9B-NVFP4-MTP. From GGUF metadata: **head_dim D=256** (`key_length=value_length=256`),
`n_head=16`, `n_head_kv=4` → **gqa_ratio=4**, 33 layers, `full_attention_interval=4` (≈8 full-attn layers,
rest SWA). D=256 is the dominant fact — it changes everything below vs a D=128 assumption.

USER DIRECTIVE: FA must be AT LEAST llama's FA on this exact config (the FLOOR — copy structure 1:1 if
needed, same silicon, no magic), and MORE via the bw-specific edge llama's generic FA does not do (the
CEILING). NEVER land below llama.

---

## 1. Which FA kernel llama SELECTS on sm_120 (traced through the real dispatch)

Dispatch entry: `ggml_cuda_get_best_fattn_kernel` (`fattn.cu:340`). For cc=1200:
`turing_mma_available(1200)` is TRUE (`common.cuh:348` — NVIDIA && highest_arch ≥ TURING). So the dispatch
falls into the Turing-or-newer branch (`fattn.cu:457`):

- **PREFILL (pp512, n_q large):** `can_use_vector_kernel` is true (D=256, %64==0, ≠192) but the vec
  fast-path only fires for `Q->ne[1] <= 1/2` (n_q≤1 f16, ≤2 quant). At n_q=512 that fails → returns
  **`BEST_FATTN_KERNEL_MMA_F16`** (`fattn.cu:478`). **llama prefill = MMA-f16 (mma.sync), NOT wmma.**
  wmma is Volta-only here (`ggml_cuda_should_use_wmma_fattn` true only for Volta/RDNA3/MThreads,
  `fattn-wmma-f16.cuh:30`) — sm_120 never takes the wmma path.
- **DECODE (tg, n_q=1):** f16 KV, cc ≥ ADA_LOVELACE, `Q->ne[1]==1` → **`BEST_FATTN_KERNEL_VEC`**
  (`fattn.cu:460`). **llama decode = vec kernel.**

### Prefill MMA config actually instantiated (measured kernel name)

ncu shows the launched kernel is `flash_attn_ext_f16<256, 256, 16, 4, 0, 0>` = `<DKQ=256, DV=256,
ncols1=16, ncols2=4>`. Trace: `switch_ncols2<256,256>` with gqa_ratio=4 hits `use_gqa_opt && gqa_ratio>2`
→ **ncols2=4** (`fattn.cu:97`). Then `switch_ncols1<256,256,4>`: cc=1200 is not Turing-only and n_q is
large, so it falls through to `ncols1 = 64/ncols2 = 16` (`fattn.cu:34`). Effective tile = **ncols =
ncols1·ncols2 = 64 Q columns per CTA** (16 distinct query rows × 4 GQA query-heads sharing one KV head).

Config row (ampere table, sm_120 uses `ampere_mma_available` branch `fattn-mma-f16.cuh:232`):
`GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 128, 2, 32, 128,128,128, 2, true)` (`fattn-mma-f16.cuh:72`):
- **nthreads = 128 → 4 warps/CTA**
- **occupancy = 2 CTAs/SM** (deliberate, see §3)
- **nbatch_fa = 32** (KV rows processed per softmax-rescale step / cp.async stage)
- **nstages_target = 2** → 2-stage cp.async pipeline (active because ncols2=4 ≥ 2 and cp_async_available,
  `fattn-mma-f16.cuh:349`)
- **Q_in_reg = true → O / VKQ accumulator lives in REGISTERS, not smem**

### Decode VEC config instantiated (measured)

`flash_attn_ext_vec<256, 1, 1, 1, 0>` = `<D=256, ncols=1, …>`. **nthreads = 128 → 4 warps/CTA**
(`fattn-vec.cuh:4`, fixed 128). Grid `(1,1,16)` = grid.z = n_head = 16 blocks (one CTA per query head;
GQA handled by `head/gqa_ratio` indexing into K/V, `fattn-vec.cuh:108-111`). KV read as f16; Q is
quantized to q8_1 in smem for a dp4a-style KQ dot when KV is quantized, else half2 path
(`fattn-vec.cuh:87,178-199`). O/VKQ in registers (`fattn-vec.cuh:125`).

---

## 2. Structure to match — the FLOOR, item by item

| Property | llama MMA-f16 prefill (the floor) | bw24 `fa_prefill_q` TODAY |
|---|---|---|
| warps/CTA | **4** (128 threads) | **1** (32 threads) ← catastrophe |
| Q cols/CTA (tile) | **64** (16 rows × 4 GQA heads) | **16** rows, 1 head |
| GQA scheme | grid.y = n_head_kv; 4 query heads share 1 staged K/V (ncols2=4) | grid.y = **n_head** → KV re-read 4× |
| O accumulator | **registers** (Q_in_reg=true) | **smem** `sO[16][256]` RMW per KV tile |
| KV staging | **cp.async, 2-stage** double-buffer (nstages=2) | synchronous f32→bf16 |
| KV tile (nbatch_fa) | **32** rows/step | 64 (BK), but 1-warp serial |
| occupancy target | **2 CTAs/SM** by design | n/a (1 warp wastes SM) |
| split / combine | **stream-K split + `flash_attn_stream_k_fixup`** combine pass | none (prefill) |
| matmul | mma.sync m16n8k16 f16 | mma.sync m16n8k16 bf16 (same class) |

bw24 launcher (confirmed `lib.rs:1079-1082`): `grid=( (t+15)/16, n_head, 1 ), block=(32,1,1)` — literally
1 warp per 16-row Q tile, per query head. That is the 2.08%-occupancy / 9.26ms kernel.

The FA34-PLAN P0–P4 items map **exactly 1:1** onto llama's structure — this is not coincidence, it is the
floor:
- **P0 register-O** = llama `Q_in_reg=true`.
- **P1 GQA K/V reuse (grid.y=n_head_kv, 4 Q-heads share K/V)** = llama `ncols2=4`.
- **P2 multi-warp (4 warps, BLOCK_Q 16→64)** = llama `nthreads=128`, `ncols=64`.
- **P4 cp.async 2-stage** = llama `nstages_target=2`.
Doing P0+P1+P2+P4 *is* reconstructing the llama floor in mma.sync. P3 (conditional rescale) and P5
(swizzle) are on top. **There is no structural gap beyond these — match them and bw24 prefill FA ≈ llama.**

---

## 3. MEASURED floor numbers (ncu, this machine, this model)

### Prefill — `flash_attn_ext_f16<256,256,16,4,0,0>` (pp512, warm)
- **sm__warps_active = ~15.0%** (cap `sm__maximum_warps_per_active_cycle_pct = 16.67%`)
- **sm__throughput (SM SoL) = ~52%**
- **gpu__time_duration ≈ 167–171 µs/call**
- regs/thread = **230**; dynamic smem = **35.07 KB/block**; grid (164,1,1); block (32,4,1)=128
- occupancy limiters: registers→2 blocks AND smem→2 blocks (BOTH cap at 2 CTAs/SM)
- companion `flash_attn_stream_k_fixup_general<256,16,4>`: 87% warps, 72% SM, ~93 µs (the split combine)
- llama pp512 end-to-end with -fa: **~1837 t/s** (this run; the prior head-to-head warm number is 5450 —
  that figure is whole-prefill incl. the NVFP4 GEMMs, not FA-only; FA is ~13% of prefill per FA34-PLAN).

**The crucial insight: 15% warps-active is NOT a llama bug to beat — it is the design point.** llama
intentionally runs a **register-heavy (230 regs), 2-CTA/SM, high-ILP tensor-core** FA that is
compute/pipe-bound, not occupancy-bound; it still reaches **52% SM SoL**. bw24's 2.08% is a different
animal (1 warp/CTA = 1/8 the warps llama runs AND no register-O AND no cp.async). The floor bw24 must hit
is **~15% warps active at ~50% SM SoL**, i.e. 4 warps × 2 CTAs/SM with register-O — not "maximize
occupancy." Chasing >16.67% by cutting registers would LOSE (less ILP, spills at D=256).

### Decode — `flash_attn_ext_vec<256,1,1,1,0>` (tg, short ctx)
- warps_active = 8.3%, SM SoL = 3.3%, **mem throughput = 12.6%**, ~17.8 µs/call, 234 regs, grid 16 blocks.
- Low fill is **expected at short ctx** (16 blocks = 1 per head; nothing to do). This is the same regime
  bw24's D0 split-K already targets. llama decode does NOT split-K the vec kernel at this size; bw24's
  shipped split-K decode (`fa_decode_vec_q`, grid=(n_head_kv, n_splits)) is already structurally AHEAD of
  llama here at mid/long ctx. **bw24 decode FA is not the gap — prefill FA is.** (decode tg128 83 vs 122.5
  gap is dominated by the GEMV/MMVQ + sampler path, not FA, per the bw24 state notes.)
- llama tg64 e2e with -fa: ~23 t/s under ncu (instrumented/slow; not a perf number, just kernel id).

---

## 4. The CEILING — "more than llama" (bw-specific edge llama's generic FA does NOT do on sm_120)

llama's MMA prefill **requires f16 K/V** (`fattn.cu:558` forces `need_f16_K/V=true` for MMA/WMMA/TILE) and
its quantized-KV support is **vec-only and decode-only** (`fattn.cu:463-472`, n_q≤2). So:

1. **In-kernel quantized-KV PREFILL attention.** llama has NO quantized-KV mma/prefill kernel — quant KV
   forces the slow vec path or an f16 up-convert. bw24 already ships `fa_prefill_q` (q8_0 K / q5_1 V inline
   dequant during stage-to-smem). Once it gets the §2 floor structure (register-O + 4 warps + GQA reuse +
   cp.async), it reads **half/quarter the KV bytes llama's f16 prefill reads** while doing the same mma —
   a bandwidth edge llama structurally cannot match without rewriting its dispatch. This is the single
   biggest "more than llama" lever and bw24 already owns the dequant arithmetic.

2. **FP8 (e4m3) KV-cache attention.** This is exactly what vLLM/SGLang reach for via **FlashInfer**, which
   on THIS machine has already generated sm_120 (`120f`) prefill kernels
   `batch_prefill_with_kv_cache_dtype_q_bf16_dtype_kv_e4m3_dtype_o_bf16` and `…_e5m2_…`
   (`/data/cache/flashinfer/0.6.8.post1/120f/generated/`). So the competitive bar on sm_120 is **FP8-KV
   attention**, not f16. bw24 porting e4m3 KV with in-mma dequant (fuse dequant with the online-softmax
   rescale) matches FlashInfer's edge and beats llama's f16-only FA on KV bandwidth. (Note: those cached
   FlashInfer kernels are head_dim 64 builds; the D=256 path would be generated on demand — the point is
   the *technique* FP8-KV is the sm_120 competitive bar.)

3. **mxf8 block-scale mma for QK/PV itself (381 TFLOP path).** Candidate, not floor: if the QK/PV mma runs
   in block-scaled FP8 (the same Stage-C 381-TFLOP path tracked for GEMM), prefill FA compute throughput
   exceeds llama's bf16/f16 mma. Higher risk (numerics of softmax in FP8); gate behind the floor.

**vLLM/SGLang verdict:** both use FlashInfer on sm_120 (and SGLang can use FA3 on Hopper, but on sm_120
laptop it falls to FlashInfer too). FlashInfer's sm_120 FA is the **same algorithmic family** as llama's
MMA (CTA-tile, register accumulators, cp.async, split-K) plus **FP8/e4m3 KV** and warp-specialization. It
is not a structurally different kernel worth copying over llama's — llama's MMA-f16 is the cleaner,
readable floor reference; FlashInfer's *only* additive idea over llama for bw24 is **FP8-KV** (item 2),
which bw24 can reach via its existing quant-KV seam (item 1).

---

## 5. Concrete floor target for the FA34-PLAN prefill port

Land P0+P1+P2+P4 (register-O + GQA-reuse grid.y=n_head_kv + 4-warp 64-col tile + cp.async 2-stage), i.e.
rebuild `fa_prefill_q` as a 1:1 structural twin of `flash_attn_ext_f16<256,256,16,4>`:
- block (32,4,1) = 128 threads, **target 2 CTAs/SM**, ~230 reg budget OK (don't fight it down).
- grid.y = n_head_kv (=4), inner gq loop over 4 query heads sharing staged K/V → ncols=64 equivalent.
- O/VKQ in registers via the validated `CTile::get_i/get_j` lane map.
- nbatch_fa = 32 KV rows/step, 2-stage cp.async on the K buffer.
- **Acceptance (ncu, pp512, the x8 full-attn layers):** warps_active ≥ ~15%, SM SoL ≥ ~50%, per-call time
  within ~1.2× of llama's ~170 µs. Hitting that = floor met. Quant-KV (item 1) then takes it BELOW
  llama's time on KV-bandwidth-bound long-context prefill = ceiling.
- Do NOT chase warps_active above 16.67% by shedding registers — that trades away the ILP that makes the
  D=256 mma path fast; llama proves 2 CTAs/SM is the right point on this silicon.

## Files / evidence
- Dispatch + selection: `/data/projects/llama.cpp/ggml/src/ggml-cuda/fattn.cu` (lines cited).
- MMA config table (sm_120 = ampere branch): `/data/projects/llama.cpp/ggml/src/ggml-cuda/fattn-mma-f16.cuh:38-88,232`.
- Vec decode: `/data/projects/llama.cpp/ggml/src/ggml-cuda/fattn-vec.cuh`.
- CC helpers: `/data/projects/llama.cpp/ggml/src/ggml-cuda/common.cuh:344-358`; wmma gate `fattn-wmma-f16.cuh:26`.
- bw24 launchers: `/home/avifenesh/projects/bw24/crates/bw24-engine/src/lib.rs:1079-1123` (prefill 1-warp; decode split-K).
- bw24 kernels: `/home/avifenesh/projects/bw24/crates/bw24-engine/cu/flash_attn.cu`.
- FlashInfer sm_120 FP8-KV cache: `/data/cache/flashinfer/0.6.8.post1/120f/generated/` (e4m3 + e5m2 prefill).
- ncu numbers: measured this session (commands in §3).
