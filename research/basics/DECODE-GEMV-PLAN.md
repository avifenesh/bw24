# Decode-GEMV Plan — raise bw24 dp4a (m=1) from ~40-55% to ~86% of 847 GB/s

**Lever `decode_gemv`.** Decode (m=1) GEMV: bw24's dp4a matvec runs ~40-55% of the 847 GB/s
HBM ceiling; llama.cpp's vectorized MMVQ hits ~86%. That is the **decode 73 → 126 tok/s gap**.
Target: rewrite the `qmatvec_*_dp4a` kernels (one block per output row, 128 threads each
striding the contraction) into a **warp-per-row, vectorized-load MMVQ** matching llama's tuned
`mul_mat_vec_q` so the decode matmuls saturate DRAM.

Platform: **sm_120** (RTX 5090 Laptop, 847 GB/s), 2-4 concurrent agents, GGUF resident-quant
weights, Rust + raw CUDA. dp4a (`__dp4a`) is native sm_61+; no wgmma/tcgen05 needed.

---

## 0. The honest ceiling first (decode is bandwidth-bound — 86% is the SOL, not 100%)

Decode (m=1) is a **memory-bound GEMV**: every weight byte is read exactly once per token and
multiplied into a single activation row. Arithmetic intensity is ~one MAC per weight byte, far
below the sm_120 compute/BW crossover, so **the kernel's wall-clock floor is `weight_bytes / 847 GB/s`.**
You cannot beat that with more dp4a or tensor cores — there is nothing to amortize at m=1 (a
weight read serves one token). This is exactly the structural limit called out in
`DECODE-GAP-PLAN.md:42,97` and why the prefill GEMM root-fix (`GEMM-PLAN.md:9`) does **not** apply
to decode (`lib.rs:488` keeps m=1 on the matvec path deliberately).

- **llama's 86%** is the realistic state-of-the-art for a quant GEMV. The missing 14% is
  fixed launch/prologue cost, the last-wave tail (rows not a multiple of the grid), and L2
  imperfection — it is **not** recoverable without changing the regime (batch>1, or MTP).
- **bw24 today at ~40-55%** is leaving ~30-46 points on the table to *fixable* causes:
  per-block (not per-warp) layout that under-fills the row loop, 2-byte (`get_int_b2`) weight
  loads that defeat L1 coalescing, a 4-warp block whose cross-warp `__syncthreads` reduction
  adds latency on the critical path, and a one-row-per-block grid that wastes the tail wave.

**So the target is `~86% × 847 ≈ 728 GB/s effective`, mapping to decode ≈ 110-126 tok/s.**
Anything claiming >86% on m=1 decode is measuring wrong (likely counting L2-resident weights, or
a model small enough to be launch-bound rather than BW-bound). The plan's job is to close the
40-55% → 86% gap, **not** to invent bandwidth that physics does not allow.

---

## 1. Root cause vs llama (file:line)

| # | bw24 today | llama MMVQ | Effect on BW |
|---|---|---|---|
| 1 | **One block per output row**, 128 threads (4 warps), all 4 warps stride the *same* row's `in/32` blocks, then a shared-mem `__syncthreads` cross-warp reduce. `qmatvec_q8_0_dp4a` `cu/qmatvec.cu:386-410`; reduce `cu/qmatvec.cu:361-374`. Launch `grid=(out_f, m, 1)`, `block=(128,1,1)` `lib.rs:507`. | **One warp per row** (`ncols_dst=1`: `nwarps=4`, `rows_per_cuda_block=1` → 4 rows/block, each warp independent), `mmvq.cu:475-592`. Reduction is warp-only `__shfl_xor_sync` (`mmvq.cu:632`); shared mem only merges the (nwarps-1) *extra* token-columns, which at m=1 collapses to a no-op. | bw24 pays a `__syncthreads` + smem round trip on the m=1 critical path that llama does not. 4 warps reducing one row also means each warp does only 1/4 of the row's K-blocks before a barrier — worse latency hiding than 1 warp owning a whole row and 4 rows in flight. |
| 2 | **2-byte weight loads.** `get_int_b2(wq + k*4)` reads `uint16x2` because Q8_0 blocks are 34 B (2-byte aligned, never 16-byte). `cu/qmatvec.cu:376-381,406`. | Loads quantized blocks as `int2`/`int4` from the block struct; Q8_0 `qs` is contiguous int8[32] read as int (`vecdotq.cuh:240-255`). | bw24's 34-B stride scatters consecutive warps to byte offsets `[0,34,68,...]` → ~60% of each 128-B L1 line wasted (per the BW24-inefficiency finding). The fix is **per-warp contiguous striding** so 32 lanes read one 128-B line, plus wider loads where alignment allows. |
| 3 | **No weight prefetch.** The K-loop loads `wb` then immediately `dp4a`s it — pure load→use dependency. `cu/qmatvec.cu:398-407`. | Same structural loop, but warp-per-row + high block count keeps many independent rows in flight so the scheduler hides the load latency. | bw24's 4-warps-per-row halves the independent-row parallelism vs llama; raising rows-per-block restores it. `cp.async` double-buffer of the next K-block is a sm_120-available secondary lever. |
| 4 | **Grid `(out_f, m, 1)` one row/block.** At m=1, `out_f` blocks. For `out_f` not ≫ the SM count, the **last wave** is partly empty. `lib.rs:507`. | 4 rows/block → `out_f/4` blocks; same total work, ¼ the block count, smaller tail. | Tail-wave waste; minor for big `out_f` (`out_f=4096` → fine), real for small projections. |

(The DECODE-GAP-PLAN's Lever 2 correctly notes the "81/82 idle SMs" claim is wrong for big
`out_f`; the real wins here are #1 the per-warp layout + reduction, and #2 coalesced loads — not
SM count. `DECODE-GAP-PLAN.md:47`.)

---

## 2. Target kernel design — warp-per-row vectorized MMVQ (m=1)

New kernels `qmatvec_<dt>_mmvq` in `cu/qmatvec.cu` (keep the existing `_dp4a` ones as the
correctness oracle / fallback until the gate passes). One template family, specialized per dtype
by its existing block-decode body.

### 2.1 Warp-per-row layout (fix #1)
- `block = (32, ROWS_PER_BLOCK, 1)` — `blockDim.x = 32` (one warp), `blockDim.y = ROWS_PER_BLOCK`.
  `thread_id = 32*threadIdx.y + threadIdx.x`; **warp `threadIdx.y` owns output row
  `blockIdx.x*ROWS_PER_BLOCK + threadIdx.y`** (mirrors `mmvq.cu:475-592`).
- Each warp's 32 lanes stride that row's `in/32` quant blocks: `for (kb = lane; kb < nblk; kb += 32)`.
  Accumulate `int sumi` per lane in registers (Q8_0/Q4_K min-term as today).
- **Reduction is warp-only**: `__shfl_xor_sync`/`__shfl_down_sync` over 32 lanes (already in
  `mmvq_block_reduce_write` `cu/qmatvec.cu:361-374` — but **drop the smem `s[32]` + `__syncthreads`
  stages**, since one warp == one row now). Lane 0 writes `y[t*out_f + o]`. This removes the
  barrier from the m=1 critical path (fix #1).

### 2.2 rows-per-block & occupancy
- **`ROWS_PER_BLOCK = 4`** (block = 128 threads, 4 independent rows in flight) is the GENERIC
  `ncols_dst=1` setting llama uses for sm_120-class arches (`mmvq.cu:349-473`, GENERIC table →
  `nwarps=4` for ncols_dst=1). 4 warps/block × many blocks gives the scheduler enough independent
  rows to hide weight-load latency (fix #3) without any barrier.
- Shared mem: **0** for m=1 (warp-only reduce). Registers: ~`sumi` + scales per lane → well under
  the limit; expect ≥8 blocks/SM, ample for the 2-4 concurrent agents (each agent's decode is a
  separate stream; high occupancy keeps all agents' rows resident).
- Tune `ROWS_PER_BLOCK ∈ {2,4,8}` empirically per dtype (llama's GENERIC uses 1-2 rows depending
  on ncols_dst; for pure m=1 decode 4 is the start). This is the only real occupancy knob —
  decode has no smem-tiling lever (no wgmma on sm_120, `DECODE-GAP-PLAN.md:47`).

### 2.3 Vectorized / coalesced weight loads (fix #2)
- **Per-warp contiguous striding** already fixes most of #2: 32 lanes of one warp reading
  `kb = lane, lane+32, ...` hit consecutive blocks → one 128-B L1 transaction per warp-step for
  the activation ints (`aq` is 32-aligned int, `cu/qmatvec.cu:402`).
- **Weights:** keep `get_int_b2` for the inherently 2-byte-aligned dtypes (Q8_0 34-B block,
  `cu/qmatvec.cu:399-401`). Do **NOT** pad Q8_0 to 36 B — on a BW-bound kernel that *adds* DRAM
  bytes and is strictly slower (`DECODE-GAP-PLAN.md:47`). For Q4_K the `qs` at offset 16 in the
  144-B superblock is 4-byte aligned (`cu/qmatvec.cu:440-441`) → load as `int`/`int2` directly.
  Q6_K `ql`/`qh` likewise. The win is alignment-aware widening **where the layout already permits
  it**, not repacking.
- Add `__restrict__` to the `W`/`aq`/`ad` pointers in the new kernels (free; the existing dp4a
  kernels at `cu/qmatvec.cu:386-389,415-418,662-665` lack it) — lets the compiler assume no
  aliasing and widen loads.
- Secondary: `cp.async.cg` double-buffer the next K-block into a tiny per-warp staging (sm_120
  supported). Bank ~0-3%; only if the warp-per-row rewrite alone stalls below ~80%.

### 2.4 Scale/epilogue math — unchanged (bit-equivalence anchor)
The per-block `acc += dw * adrow[blk] * (float)sumi` (Q8_0 `cu/qmatvec.cu:407`) and the Q4_K/Q5_K
min-offset via the `0x01010101` activation-sum dp4a (`cu/qmatvec.cu:452-457`) are **kept verbatim** —
only the *layout* and *reduction* change. The int accumulate is identical, so the new kernel is
bit-for-bit equal to the dp4a path up to f32 reduction order. NVFP4 keeps its `get_int_from_table_16_d`
byte_perm codebook unpack (`cu/qmatvec.cu:690-704`) and per-tensor macro-scale applied post
(`lib.rs:512`).

---

## 3. Dtypes — which, and in what order

Decode-hot daily quants, ordered by gain × simplicity. Each gated (§5) before the next.

1. **Q8_0** — simplest (weight already int8, no decode). Proves the warp-per-row layout + warp-only
   reduction end-to-end. This is the current 73 tok/s baseline path (`qmatvec_q8_0_dp4a`
   `cu/qmatvec.cu:386`), so its delta is the cleanest measurement of the lever.
2. **Q4_K** — most common daily quant. Nibble-unpack + 6-bit sub-scales already in
   `qmatvec_q4_K_dp4a` (`cu/qmatvec.cu:415-460`); lift the body unchanged under the new layout.
   4-byte-aligned `qs` → widen its load (fix #2).
3. **Q6_K** — `qmatvec_q6_K_dp4a` body (`cu/qmatvec.cu:464-524`); symmetric, no min term.
4. **NVFP4** — `qmatvec_nvfp4_dp4a` body (`cu/qmatvec.cu:662-717`); fp4→int8 via byte_perm codebook,
   per-tensor scale post. This is the dtype of the 9B "126.6 bar" model
   (`DECODE-GAP-PLAN.md:5,22`) — the correctness fix already landed (Task #4), so the warp-per-row
   rewrite is the path to actually racing llama on the *same* model.

**Defer:** Q5_K / Q3_K / IQ4_XS / IQ3_S (`cu/qmatvec.cu:527,588,721,263`) — not daily-hot; they
stay on the existing `_dp4a` matvec (correct, ~40-55%). The template makes adding them later a new
`decode_block<QT>()` specialization only.

**Not in scope (different lever):** FP8/NVFP4 KV-cache + mma-feed dequant — that is the *attention*
BW lever (KV already q8_0-K/q5_1-V, commit 9ebf958), separate from the weight-GEMV here.

---

## 4. Rust integration (file:line)

- **New launcher** `Engine::qmatvec_mmvq` next to `matmul_pre` (`lib.rs:482`). Same pre-quantized
  `(aq: &CudaSlice<i8>, ad: &CudaSlice<f32>)` signature so it slots into the existing q8_1-shared-
  activation flow (no extra `quantize_q8_1`):
  ```rust
  let cfg = LaunchConfig {
      grid_dim:  ((out_f as u32 + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, m as u32, 1),
      block_dim: (32, ROWS_PER_BLOCK, 1),   // warp-per-row; ROWS_PER_BLOCK = 4 start
      shared_mem_bytes: 0,                   // warp-only reduce at m=1
  };
  ```
- **Dispatch (m=1 only).** In `matmul_pre` (`lib.rs:482-513`): the existing `m >= 16` GEMM branch
  (`lib.rs:488`) is untouched; the change is the **`else` decode arm at `lib.rs:498-511`** — route
  `qtype ∈ {Q8_0,Q4_K,Q6_K,NVFP4}` to `qmatvec_mmvq` instead of the per-row `_dp4a` launch
  (`lib.rs:499-507`). Other dtypes keep `_dp4a`. Gate behind a `BW24_MMVQ` env var
  (mirroring `BW24_FAST` `lib.rs:469` / `BW24_GEMM` `lib.rs:521`) so it toggles against the dp4a
  matvec during bring-up.
- **Build:** the new kernels live in the same `cu/qmatvec.cu`, already compiled to the fatbin at
  `lib.rs:47` and resolved via `self.func(...)` (`lib.rs:505`) — no build.rs change needed beyond
  the existing `cu/qmatvec.cu` entry.
- **Decode call site** (`decode.rs` projections) is unchanged: it already calls `matmul_pre` with a
  shared pre-quantized activation; only the kernel it dispatches to changes.

---

## 5. Validation gate (argmax holds + rel vs dp4a + decode tok/s rises)

Non-negotiable, ordered. The new kernel changes only layout/reduction → must be bit-equivalent to
the validated `_dp4a` path up to f32 reduction order.

1. **Bit-equivalence (kernel_check).** Add a case to `crates/bw24-engine/src/bin/kernel_check.rs`
   comparing `qmatvec_<dt>_mmvq` vs the existing `qmatvec_<dt>_dp4a` on random
   `[1, in_f]` / `[out_f, in_f]`, for `in_f ∈ {2048, 4096, 11008, 32768}` and `out_f ∈ {small, 4096}`.
   Require **rel < 1e-3** (only f32 reduction-order rounding differs; the int `sumi` is identical).
   Repeat per dtype as each lands. (Same harness/noise-floor convention as the GEMM gate,
   `GEMM-PLAN.md:110`.)
2. **End-to-end argmax — the real gate.** With `BW24_MMVQ=1`, the three validated checkpoints must
   keep their argmax **bit-exact**:
   - `run_dense` (qwen3) **prefill argmax = 268**
   - `run_hybrid` (qwen35) **= 271**
   - 35B-A3B MoE **= 1178**
   (the validated triplet, `GEMM-PLAN.md:111`, `DECODE-GAP-PLAN.md:143`). Any mismatch = revert.
   Decode is greedy → the **full 128-token generated stream must be identical** to the `_dp4a` path.
3. **Rel vs current dp4a (numerics).** Per-dtype rel-error of the decode logits vs the `_dp4a`
   kernel ≤ 3e-3 on a fixed 512-token prompt (the bar used for the landed Q4_K/Q6_K dp4a work).
   Since the int path is identical, this should be ~machine-epsilon; >3e-3 means a layout/indexing
   bug, not a numeric one.
4. **Perf gate — decode tok/s must rise.** Measure **9B decode tok/s** (Task #3 beat-benchmark) on
   the daily model. Milestones:
   - First Q8_0 landing: decode **> 90 tok/s** (proves the warp-per-row + warp-only reduce win;
     ~73 → ~90 is the layout+reduction delta alone).
   - Tuned (ROWS_PER_BLOCK swept, loads widened): decode **toward 110-126 tok/s** = ~86% of
     847 GB/s. **Confirm with `nvidia-smi dmon` that achieved DRAM BW is ~700-728 GB/s during
     decode** — if tok/s stalls but BW is already ~86%, you have hit the SOL (§0) and the remaining
     gap is the watt-wall (150-175 W, `DECODE-GAP-PLAN.md:99`), not the kernel.
   - **Regression guard:** the `_dp4a` path stays in the binary; if any dtype's `_mmvq` is slower
     than its `_dp4a` on real layer shapes, keep `_dp4a` for that dtype (per-dtype dispatch flag).

---

## 6. Realistic outcome

The warp-per-row rewrite + coalesced loads is the documented path to llama's ~86% SOL on the
*kernel itself* (`DECODE-GAP-PLAN.md:97` names exactly this rewrite as "the single largest remaining
item"). Honest projection: **73 → ~90 tok/s from the layout+reduction fix alone, toward 110-126 with
load-widening and ROWS_PER_BLOCK tuning** — i.e. closing the decode-GEMV gap to within the 86%
ceiling. It will **not** exceed 86% (decode is BW-bound; that is physics, §0). Beating 126 *as a
throughput regime* requires batch>1 or net-profitable MTP (Task #7), which are separate levers —
this lever's job is to make the m=1 GEMV stop wasting ~30-46 points of the bandwidth it already pays
for.
