# bw24 Attention Perf Plan — FA-vec Decode + GQA Broadcast (prefill MMA: DEFER)

Target: RTX 5090 / sm_120, GGUF resident-quant (q8_0 K, q5_1 V), Rust + raw CUDA,
2–4 concurrent agents (batch ≈ 1 per agent, step-interleaved). head_dim=256 (qwen35),
GQA n_head/n_head_kv = 4:1, scale = 1/16.

## 0. The lever, restated against the actual code

Two attention kernels exist in `crates/bw24-engine/cu/flash_attn.cu`:

- **Prefill** `fa_prefill_f32` (`flash_attn.cu:250-417`) and its quantized twin
  `fa_prefill_q` (`flash_attn.cu:426-565`) — both are the **validated bf16 m16n8k16
  tensor-core FA-2** (the `ld_A`/`ld_A_trans`/`mma_bf16` primitives at
  `flash_attn.cu:90-112`). Prefill attention is only **~3% of prefill** time today
  (the 43× prefill gap is the GEMM weight-reread, owned by GEMM-PLAN.md, not attention).

- **Decode** `fa_decode_f32` (`flash_attn.cu:590-663`) + `fa_decode_combine_f32`
  (`flash_attn.cu:667-690`) — this is **scalar-ish**: `block_dim=(head_dim,1,1)=256`
  threads, **one thread per output dim**, and a **per-key scalar dequant in the hot
  loop** (`dq_q8_0_elem` at `:631`, `dq_q5_1_elem` at `:654`). Launched from
  `Engine::fa_decode` (`lib.rs:715-739`), `grid=(n_head, n_splits, 1)`,
  `n_splits=ceil(t_kv/256)` (`lib.rs:720`).

Decode is the lever. It is **bandwidth-bound** (decode arithmetic intensity ≈ 1–2
FLOP/byte; crossover is ~141 FLOP/byte for bf16 — `sm120-empirical-capabilities.md:71,92`),
so the wins are **(1) read each KV byte fewer times** and **(2) saturate the 829 GB/s
achieved read BW** (`sm120-empirical-capabilities.md:22`), NOT tensor cores.

The two structural defects in `fa_decode_f32` that leave BW on the table:

1. **GQA 4:1 is not exploited.** `kv_head = head / (n_head/n_head_kv)` is computed at
   `:604`, but each of the 4 Q heads sharing a KV head is a **separate block**
   (`grid.x = n_head`, `:601`) that **independently re-dequants the same**
   `K[t,kv_head,:]` / `V[t,kv_head,:]` (`:630-631`, `:653-654`). Each KV row is read
   from HBM (or at best L2) **4×** per token. Loading it once per KV head and
   broadcasting to its 4 Q heads is the **~1.3× KV-BW** win (XQA-style; FlashInfer
   `xqa.py:154-179`, TRT-LLM `mha.cu:76-99`).

2. **Scalar dispatch wastes the warp.** 256 threads, one active per output dim, scalar
   per-key dequant — `dq_q8_0_elem`/`dq_q5_1_elem` are a ~12-cycle dependent load+convert
   chain (`flash_attn.cu:133-155`). A warp-vectorized, register-resident accumulator with
   double-buffered KV load (the llama.cpp fattn-vec structure, `fattn-vec.cuh:104-318`)
   hides that latency.

This plan is two parts: **(A)** rewrite decode as FA-vec warp-per-token + GQA broadcast
(SHIP); **(B)** the prefill bf16 m16n8k16 mma (DEFER — honest accounting below).

---

## (A) FA-vec warp-per-token decode + GQA 4:1 broadcast — **SHIP**

### A.1 Kernel: `fa_decode_vec_q` (new, in `flash_attn.cu`)

Replace the element-per-thread `fa_decode_f32` with a **warp-per-(Q-token, KV-head-group)**
kernel modeled on llama.cpp `fattn-vec.cuh` (`fattn-vec.cuh:19-42` signature,
`:104-254` compute loop) but specialized to bw24's contiguous (non-paged) q8_0/q5_1 cache.

**CTA / warp organization (the GQA broadcast):**
- `block_dim = (32, GQA_RATIO, 1)` → one **warp per Q head** in a GQA group; the warp's
  `threadIdx.y ∈ [0, GQA_RATIO)` selects which of the 4 Q heads it serves.
- `grid = (n_head_kv, n_splits, 1)` — **one block per KV head** (was per Q head). For
  qwen35 this is 8 blocks not 32. The KV-head pointer is computed **once per block**:
  `kv_head = blockIdx.x`; the 4 Q heads are `kv_head*GQA_RATIO + threadIdx.y`.
- **Broadcast:** each KV tile `K[t,kv_head,:]`/`V[t,kv_head,:]` is **dequantized once into
  shared memory** by the block (32 lanes × GQA_RATIO warps cooperate, or one warp
  dequants and `__syncthreads()` publishes), then **all 4 warps read the same `sK`/`sV`**.
  Each KV byte leaves HBM/L2 **once per group** instead of 4×.

**Per-warp inner loop (online softmax, register-resident accumulator):**
- Stage q (one Q token, head_dim=256) into registers per warp: each lane owns
  `head_dim/32 = 8` Q elements (float, pre-scaled by `scale`).
- Loop KV tile of `T_KV_TILE` keys (start 32; tune 32→64):
  1. Dequant the tile's K and V **once per block** into `__align__(16)` smem
     `sK[T_KV_TILE][head_dim]`, `sV[T_KV_TILE][head_dim]` (reuse the **exact** byte
     layout math from `dq_q8_0_elem`/`dq_q5_1_elem`, `flash_attn.cu:133-155`, but
     **warp-vectorized**: 32 lanes unpack a 32-elem q8_0/q5_1 block in parallel — the
     amortization called out in the findings, ~12 cyc/elem → ~0.4 cyc/elem throughput).
  2. Per warp: `KQ[j] = sum_d(q[d] * sK[j][d])` via lane-parallel partial + `warp_reduce_sum`
     (`__shfl_xor_sync`, reuse `warp_amax`/`warp_max` pattern at `flash_attn.cu:158-173`).
  3. Online softmax (FA-2 recurrence, **identical to the validated prefill**,
     `flash_attn.cu:348-362`): `m_new=max(m,KQ)`, `alpha=exp2((m-m_new)*LOG2E)`,
     `p=exp2((KQ-m_new)*LOG2E)`, rescale register accumulator `acc[d]=acc[d]*alpha + p*sV[j][d]`,
     `l = l*alpha + p`. **Use `exp2f` + `LOG2E`** (`flash_attn.cu:115`); the C6 review note
     forbids the 2.079 bias — keep the self-normalizing form (`flash_attn.cu:37-45`).
- Split-K: keep the existing split decision (`n_splits=ceil(t_kv/256)`, `lib.rs:720`) and
  **reuse `fa_decode_combine_f32` unchanged** (`flash_attn.cu:667-690`) — it is already
  the LSE merge and is correct. Long-ctx (t_kv > ~4–8K) is exactly where split-K earns its
  keep at batch=1 (FlashInfer `scheduler.cuh:148-210`); the per-block KV-head fan-out also
  raises occupancy at small t_kv (8×n_splits blocks all resident).

**SM_120 fit (per findings):** pure `__shfl_xor_sync` + `expf`/`exp2f` + register accumulate;
no wgmma/tcgen05/TMA. q8_0 KQ dot can optionally use `dp4a` (native sm_120) if Q is
pre-quantized to q8_1, but the f32-Q × dequant-K path is simpler and ships first. Fits the
96 KB smem / 256-reg budget: `sK+sV` at T_KV_TILE=32, head_dim=256, bf16 = 2×32×256×2 = 32 KB
(double-buffer → 64 KB, still under 100 KB/CTA), acc = 8 floats/lane.

### A.2 Rust integration

- New kernel name in the fatbin (built at `lib.rs:47`). `Engine::fa_decode` (`lib.rs:715`)
  swaps `self.func("fa_decode_f32")` (`:726`) → `"fa_decode_vec_q"`, changes
  `grid_dim` from `(n_head, n_splits, 1)` (`:727`) to `(n_head_kv, n_splits, 1)` and
  `block_dim` from `(head_dim,1,1)` (`:728`) to `(32, gqa_ratio, 1)`, and sizes
  `shared_mem_bytes` for the double-buffered `sK`/`sV` tiles. **Args unchanged** (same
  q8_0/q5_1 cache views, k_tok_bytes/v_tok_bytes). The combine launch (`:733-737`) stays.
- Gate behind a `BW24_FA_VEC` env toggle (mirror `BW24_FAST` at `lib.rs:418`) so the old
  `fa_decode_f32` stays as the bit-reference during bring-up.
- `fa_prefill_view`/`fa_prefill` and the MTP verify path are untouched (prefill kernels
  unchanged).

### A.3 Expected win (honest)

- GQA broadcast: **~1.3× KV-BW** at batch=1 (findings; XQA sm120). Decode is BW-bound, so
  KV-BW reduction ≈ decode-attention speedup on the attention portion.
- Warp-vectorized dequant + register accumulator: removes the scalar dispatch stall; the
  combined effect should move decode attention from latency-bound toward the 829 GB/s
  ceiling. **Net decode tok/s gain is bounded by attention's share of the step** — at short
  ctx attention is a minority of decode (GEMM/MoE dominate, ROADMAP.md:13 decode 56 tok/s),
  so the headline decode tok/s lift is modest at 512 ctx but **grows with context length**
  (KV read is O(t_kv); GQA + saturated BW is the long-ctx lever). Do not claim a blanket
  decode multiplier — claim it per-ctx, measured.

---

## (B) Prefill bf16 m16n8k16 mma tuning — **DEFER**

**Honest accounting:** prefill attention is **~3% of prefill** (the 43× prefill gap is the
GEMM weight-reread — see GEMM-PLAN.md §0, owned by `qmatvec_gemm`). The prefill FA
(`fa_prefill_f32` `:250-417`) is already correct on tensor cores (validated 11/11 vs
oracle, argmax 268 — ROADMAP.md:16) and uses the proven m16n8k16 primitives.

Even a **2× speedup of a 3% slice is a ~1.5% prefill win** — below the noise of the GEMM
work and the validation argmax gate. The known inefficiencies (synchronous K/V stage then
GEMM with no cp.async double-buffer, `flash_attn.cu:296-332`; single-warp-per-tile
occupancy, `flash_attn.cu:47-50` self-documents this as a follow-up) are **real but not
worth touching now**:

- **DEFER** the cp.async ping-pong / multi-warp prefill retune until **after** the GEMM
  lever (Task #10/#16) lands and re-measures the prefill breakdown. If, post-GEMM,
  attention's share rises above ~15% of prefill (long-ctx prompts, where attention is
  O(T²)), revisit — at that point the cp.async double-buffer of `sK`/`sV` (FlashInfer
  `decode.cuh:308-320` pattern) and m16n8k16 → wider-tile retune become worthwhile.
- **DO NOT** switch the prefill mma dtype or rewrite the validated bf16 path; the
  correctness-by-construction notes (`flash_attn.cu:17-50`) are load-bearing.

The **one** prefill change worth folding in opportunistically (free, same kernel): apply
the **A.1 warp-vectorized inline dequant** to `fa_prefill_q`'s stage-to-smem loop
(`flash_attn.cu:466-473`), since it currently uses the same scalar `dq_*_elem` per element.
That is a load-path tidy, not an mma retune.

---

## Validation gate (non-negotiable, ordered)

1. **Attention rel vs current.** In `kernel_check.rs`, add `fa_decode_vec_q` alongside the
   existing decode gate (`kernel_check.rs:488-520`): same synthetic KVQ inputs
   (tkv ∈ {64,128,257}, q8_0 K / q5_1 V via `append_kv_quantized_view`), compare vs
   `cpu_sdpa` (`kernel_check.rs:454`). Gate **`rel < 6e-2`** (the documented noise floor,
   `:518`) AND **`rel` no worse than the current `fa_decode_f32`** on the same inputs
   (regression check — print both). Prefill kernels (`fa_prefill T=… rel<2e-2`,
   `kernel_check.rs:486`) must stay green (they're untouched).
2. **End-to-end argmax (the authoritative gate, `:517`).** Run the three validated
   checkpoints with `BW24_FA_VEC=1`: dense qwen3 **argmax 268**, hybrid qwen35
   **argmax 271**, MoE 35B-A3B **argmax 1178** (ROADMAP.md:16-17, run via `run_gen.rs`).
   **Any mismatch = revert.** These dominate the synthetic rel test.
3. **Decode tok/s (the perf gate).** `run_gen.rs` already times decode tok/s
   (`run_gen.rs:82,129`) and prefill pp tok/s (`:70-79`). Require decode tok/s **≥ current
   baseline (56 tok/s, ROADMAP.md:13)** with `BW24_FA_VEC=1` at short ctx (no regression),
   and a **measurable rise at long ctx** (e.g. 4K+ prompt) where GQA + saturated KV-BW pays
   off. Report per-ctx, not a single multiplier. Prefill pp512 must be **unchanged** (Part B
   defers prefill; attention is 3%).

---

## Files to touch (concrete)

- **New kernel** in `crates/bw24-engine/cu/flash_attn.cu`: `fa_decode_vec_q` (warp-per-token
  + GQA broadcast + warp-vectorized inline dequant), reusing `warp_max`/`warp_reduce`
  (`:158-173`), `LOG2E`/`exp2f` softmax (`:115`, `:348-362`), and the q8_0/q5_1 byte-layout
  from `dq_q8_0_elem`/`dq_q5_1_elem` (`:133-155`). Keep `fa_decode_combine_f32` (`:667-690`)
  as-is.
- **Edit** `crates/bw24-engine/src/lib.rs:715-739` (`Engine::fa_decode`): func name (`:726`),
  `grid_dim` `(n_head_kv, n_splits, 1)` (`:727`), `block_dim` `(32, gqa_ratio, 1)` (`:728`),
  smem sizing; gate on `BW24_FA_VEC`.
- **Edit** `crates/bw24-engine/src/bin/kernel_check.rs:488-520`: add the `fa_decode_vec_q`
  rel + regression case.
- **Unchanged:** `fa_prefill_f32` / `fa_prefill_q` (Part B DEFER), `append_quantize_kv_*`
  (`:180-232`), MTP verify path. Optional opportunistic tidy: vectorize the
  `fa_prefill_q` stage-dequant loop (`:466-473`).
