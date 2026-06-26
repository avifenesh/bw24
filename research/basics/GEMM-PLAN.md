# Batched Tensor-Core Quant GEMM (prefill+decode root fix)

I have verified all the critical references. The activation already comes pre-quantized to q8_1 (`quantize_q8_1` at qmatvec.cu:341, producing `aq` int8 + `ad` f32 scales — exactly what `matmul_pre` at lib.rs:468 consumes). The dp4a kernels' inner loop (qmatvec.cu:398-407) already does the integer dot product I'll lift into the MMA path. I now have everything needed to write a concrete plan.

# Batched Tiled Quant GEMM for bw24 Prefill (T>1) on sm_120

## 0. The problem, restated against the actual code

Every `matmul` in prefill goes through `Engine::matmul` (`crates/bw24-engine/src/lib.rs:411`), which launches a per-token matvec with `grid_dim=(out_f, m, 1)` — confirmed in `matmul_pre` (`lib.rs:489`, the `qmatvec_*_dp4a` fast path) and in the Stage-A `qmatvec_f32` launch (`lib.rs:146`). The kernels themselves (`qmatvec_q8_0_dp4a` at `cu/qmatvec.cu:386-410`, all 7 siblings) index `wrow = W + o*row_bytes` **once per (o, t) block**. At T=512 each weight row is re-read and re-decoded 512×. That structural 512× weight re-read is the entire 43× prefill gap (143 vs 6240 pp512). The fix is a tile that loads a weight block ONCE and reuses it across all T tokens via tensor cores.

---

## 1. Chosen primitive: **int8 `mma.sync.m16n8k32.s32.s8.s8.s32`** (NOT bf16 m16n8k16)

**Decision: int8 MMA.** Justification, point by point:

- **sm_120 peak.** Plain int8 m16n8k32 ≈ **219 TFLOP/s** (inferred from the measured FP8 219, identical issue structure — `research/sm120-empirical-capabilities.md:72`) vs bf16 m16n8k16 = **117 TFLOP/s** measured (`sm120-empirical-capabilities.md:71`). That is 1.87× more compute headroom in the compute-bound prefill regime.
- **Reuse of existing FA helpers.** The ldmatrix discipline is already proven for bf16 in `cu/flash_attn.cu:90-112` (`ld_A`, `ld_A_trans`, `mma_bf16`) and tiled in `cu/mma_tile.cuh`. The int8 path reuses the **same** per-lane ldmatrix address `(tid%16)*stride + (tid/16)*4` and the same `__align__(16)` smem round-trip (the validated C1/C3 fixes). I change only the operand width (b8 vs b16) and the mma opcode string. This is the lowest-risk porting path: the hard part (ldmatrix addressing on sm_120) is done.
- **Decode already uses int8.** `quantize_q8_1` (`qmatvec.cu:341`) already produces `aq: int8[m,in_f]` + `ad: f32[m, in_f/32]`, and `matmul_pre` (`lib.rs:468`) already feeds it to the dp4a kernels. The prefill GEMM consumes the **identical** activation tensors — zero new activation-quant code, and prefill/decode share one quant format. The dp4a inner loop (`qmatvec.cu:404-407`, `dp4a(get_int_b2(wq), aq4[k])`) is literally the scalar version of the m16n8k32 integer dot I'm lifting to tensor cores.
- **VRAM.** Weights stay in GGUF block bytes (8.9 GB for 9B Q8_0). The bf16 path would dequant-to-bf16 (≈18 GB) and add a dequant-to-smem latency every step. The whole reason bw24 fits 24 GB where llama OOMs (ROADMAP.md:17) is keeping weights quantized — bf16 GEMM throws that away.

bf16 m16n8k16 is rejected: 1.87× slower, 2× VRAM, extra dequant latency, and no shared format with decode. Block-scaled FP8/FP4 (381/762 TFLOP, `sm120-empirical-capabilities.md:69-82`) is a **Stage-C** follow-on (CUDA-12.8 `BLACKWELL_MMA_AVAILABLE` gate) — correctness-first plain int8 ships first, per the existing `QUANT-GEMM-DECISION.md:57,90-95`.

---

## 2. Kernel design: `qmatvec_gemm_q8_0` (template over dtype)

New file `cu/qmatvec_gemm.cu` (so the 43k `qmatvec.cu` stays the decode/Stage-A file). Compute `y[T, out_f] = aq[T, in_f] (int8) · W[out_f, in_f]^T (quant)`, scaled by `ad[T, in_f/32] × dw[out_f, in_f/32]`.

### Tile shape (CTA)
- **BM = 128** output rows (`out_f` tile), **BN = 64** tokens (`T` tile), **BK = 32** contraction (one quant superblock-of-32, the natural K-atom of `quantize_q8_1`).
- 4 warps/CTA (128 threads), warp tile 64×32. Each warp owns a 64(M)×32(N) output region tiled by m16n8k32 fragments: an 8×4 grid of MMA fragments per warp (M=64/16=4 row-frags wait — use M=16 per mma → 4 M-frags, N=32/8=4 N-frags), accumulated in s32 registers.
- Grid: `(out_f/BM, ceil(T/BN), 1)` — note T (token count) is now a **block** dim, not a grid-y-per-token. This is the structural change: one CTA serves 64 tokens × 128 rows, reading each weight block once.

### Shared memory (mirror `mmq.cuh` layout, sized for sm_120's 100 KB/CTA)
```
__align__(16) int8  sW_qs[BM][BK];     // 128*32 = 4 KB  weight ints, dequant-decoded ONCE per K-step
__align__(16) int8  sA_qs[BN][BK];     // 64*32  = 2 KB  activation ints (already int8 from quantize_q8_1)
                 float sW_d[BM];         // 0.5 KB weight block scales (per 32-block)
                 float sA_d[BN];         // 0.25 KB activation block scales (ad)
```
Double-buffered over K → ≈14 KB. Comfortably under 100 KB; allows ≥4 CTAs/SM for occupancy.

### The reuse core: dequant-weight-tile-ONCE, then MMA over all T
Per K-step (one 32-block):
1. **Load + decode weight tile ONCE** — cooperative load of `BM=128` weight blocks. For Q8_0: copy the 32 int8 `qs` straight into `sW_qs` and the half scale into `sW_d` (no arithmetic — Q8_0 is already int8). For Q4_K/Q6_K/NVFP4: unpack nibbles/6-bit/fp4 to int8 here (the dtype's existing decode logic, lifted from the dp4a body, e.g. `qmatvec.cu:415-460` for Q4_K), applying only the **sub-block** structure into int8; the superblock float scale goes to `sW_d`. **This decode happens once per (row-tile, K-step), reused across all 64 tokens** — this is the 64–512× amortization.
2. **Load activation tile** — `sA_qs` ← `aq[t0..t0+64, k..k+32]`, `sA_d` ← `ad`. Already int8; just a strided copy (the token-major→K-major transpose-into-smem happens here, as analyzed: smem laid out `[k + t*BK]` so ldmatrix per-lane addressing is contiguous).
3. **`ldmatrix` + MMA** — for each warp's fragment grid, `ldmatrix.x4.b8` weight frag from `sW_qs`, `ldmatrix.x4.b8` activation frag from `sA_qs`, then `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` accumulating into **s32 register accumulators** (`acc[4][4]`). Integer accumulate — no float until the end.

### Accumulator layout & write-out
- Accumulators stay in **s32 registers per warp** for the whole K loop (the m16n8 C-fragment owns 4 s32/lane, mirroring `CTile_m16n8_f32` in `mma_tile.cuh:36` but s32).
- After all K-steps, **apply scales at the end**: `y_f32 = (float)acc_s32 × sW_d_block × sA_d_block`. Because s8×s8 over a 32-block matches one `(dw, da)` scale pair exactly (BK=32 = the quant block), each K-step's s32 partial is scaled by its own `sW_d[row]*sA_d[tok]` and summed in f32 — identical math to the dp4a `acc += dw*adrow[blk]*(float)sumi` at `qmatvec.cu:407`, just tensor-core-batched. This guarantees bit-equivalence with the validated dp4a path.
- Write `y[t*out_f + o]` (token-major, matching what `matmul_pre` returns at `lib.rs:488` and what `hybrid_forward.rs` expects).

---

## 3. Dtypes to support first (the daily models)

Ship in this order, each gated behind the validation in §5 before the next:

1. **Q8_0** — simplest (weight is already int8; smem load is a memcpy). Proves the kernel + scale math end-to-end. dtype code `QT_Q8_0=0`.
2. **Q4_K** — most common daily quant. Decode = nibble-unpack + 6-bit sub-scales into int8, lifted from `qmatvec_q4_K_dp4a` (`qmatvec.cu:415-460`). `QT_Q4_K=1`.
3. **Q6_K** — `qmatvec_q6_K_dp4a` (`qmatvec.cu:464-524`) decode body into int8. `QT_Q6_K=2`.
4. **NVFP4** — `qmatvec_nvfp4_dp4a` (`qmatvec.cu:662-717`) fp4→int8 decode + per-tensor `scale` (already applied via `scale_inplace`, `lib.rs:446`). Requires `in_f%64==0` (asserted `lib.rs:292`). `QT_NVFP4=7`.

Q5_K/Q3_K/IQ4_XS/IQ3_S stay matvec-only initially (they're not the daily hot path); they fall through to the per-token path and still work, just unaccelerated. The template is structured so adding them later is only a new `decode_block_to_int8<QT>()` specialization.

---

## 4. Rust integration

**New launcher** `Engine::qmatvec_gemm` in `lib.rs` (next to `matmul_pre` at `lib.rs:468`), same signature as `matmul_pre` (`w, aq: &CudaSlice<i8>, ad: &CudaSlice<f32>, x_fallback, m`) so it slots into the existing pre-quantized activation flow:

```rust
pub fn qmatvec_gemm(&self, w: &GpuTensor, aq: &CudaSlice<i8>, ad: &CudaSlice<f32>, m: usize)
    -> Result<CudaSlice<f32>, ...> {
    let (in_f, out_f) = (w.in_features(), w.out_features());
    let name = match qtype { QT_Q8_0 => "qmatvec_gemm_q8_0", QT_Q4_K => "qmatvec_gemm_q4_K",
                             QT_Q6_K => "qmatvec_gemm_q6_K", QT_NVFP4 => "qmatvec_gemm_nvfp4", _ => unreachable!() };
    let cfg = LaunchConfig {
        grid_dim: ((out_f as u32 + 127)/128, (m as u32 + 63)/64, 1),  // (out/BM, T/BN, 1)
        block_dim: (128, 1, 1),
        shared_mem_bytes: SMEM_BYTES,                                  // ~14 KB, double-buffered
    };
    // ... launch (W, aq, ad, &mut y, in_f, out_f, m, row_bytes) ; final per-tensor scale via scale_inplace
}
```

**Dispatch threshold.** In `matmul_pre` (`lib.rs:468`), before the dp4a launch at `lib.rs:489`, branch on m:
```rust
const GEMM_M_THRESHOLD: usize = 16;
if m >= GEMM_M_THRESHOLD && gemm_supports(qtype) {       // qtype in {Q8_0,Q4_K,Q6_K,NVFP4}
    return self.qmatvec_gemm(w, aq, ad, m);
}
// else: existing per-token dp4a (decode m=1, or unsupported dtype) — unchanged
```
- **m=1 decode keeps the dp4a matvec path** (`qmatvec_q*_dp4a`, `lib.rs:489`) — it's bandwidth-bound, MMA gives nothing there. No change to decode.
- m≥16 with a supported dtype → tiled GEMM. m in (1,16) or unsupported dtype → existing dp4a (correct, slower; tiny prefills are rare).
- The Stage-A `Engine::matmul` path (`lib.rs:411`) is untouched as the f32 correctness fallback; only the `matmul_pre` (`BW24_FAST`) prefill route gains the GEMM. Gate the whole thing behind a `BW24_GEMM` env var initially (mirroring `BW24_FAST` at `lib.rs:418`) so it can be toggled against the matvec path during bring-up.
- Activations are already quantized once per matmul in `matmul_pre` and shared across siblings — no extra `quantize_q8_1` cost vs current prefill.

**Build:** add `cu/qmatvec_gemm.cu` to the fatbin compiled at `lib.rs:47` (`QMATVEC_FATBIN_PATH`), so `self.func("qmatvec_gemm_*")` resolves via the existing loader at `lib.rs:57`.

---

## 5. Validation gate

Non-negotiable, ordered:

1. **Bit-equivalence (kernel_check).** Add a case to `crates/bw24-engine/src/bin/kernel_check.rs` (gate already at `kernel_check.rs:429`, `rel < 6e-2` noise floor) comparing `qmatvec_gemm_q8_0` output vs `qmatvec_q8_0_dp4a` on random `[T, in_f]`/`[out_f, in_f]`, T∈{16,64,128,512}. Must match to int (the s32 accumulate path is exact vs dp4a; only the final f32 scale differs in rounding) — require `rel < 1e-3`. Repeat per dtype as each lands.
2. **End-to-end argmax (the real gate).** Run `run_dense` (qwen3, prefill argmax **268**), `run_hybrid` (qwen35, **271**), and the MoE 35B-A3B (**1178**) — the three validated checkpoints (ROADMAP.md:16-17, `run_gen.rs:25-27`). `prefill argmax` MUST stay 268/271/1178 with `BW24_GEMM=1`. Any mismatch = revert; the decode argmax (m=1 path) is unaffected.
3. **Perf gate.** `pp512 tok/s` (Task #3, `beat-benchmark`) must rise from 143 toward the target. Gate intermediate milestones: first kernel landing should clear **>2000** (a >14× jump proves the amortization works); tuned should target **>6000**.

---

## 6. Realistic estimate: will this hit llama's 6240 pp512?

**Honest answer: it closes most of the gap but likely lands ~3500–5500 pp512 on the first tuned version, not 6240 — and the remaining gap is real engineering work, not a wall.**

Reasoning:
- The 43× gap is **structurally** the 64–512× weight re-read. Eliminating it via the tile is the dominant win and is most of the speedup: going from re-reading each row 512× to once-per-CTA recovers the bulk. A correct first kernel should clear >2000 easily and a tuned one 3500–5500.
- **Why not the full 6240 immediately:**
  1. **Decode/dequant-into-smem cost for Q4_K/Q6_K/NVFP4.** llama's MMQ has years of per-dtype `load_tiles_*` tuning (vectorized loads, bank-conflict-free swizzles). bw24's first decode-to-int8 will be naive and is the most likely sub-peak culprit. Q8_0 will be closest to peak (no decode); the K-quants will lag until the load path is tuned.
  2. **Occupancy / smem swizzle.** Without ldmatrix-friendly swizzling, bank conflicts on `sW_qs`/`sA_qs` cap effective TFLOP at maybe 50–70% of the 219 peak. llama's MMQ swizzle (the `MMQ_MMA_TILE_X_K_*` padding) is what gets it to ~85%.
  3. **Tile-shape tuning per layer shape.** BM/BN=128/64 is a reasonable start but the optimal differs for tall-skinny (attention proj) vs square (MLP) and for the MoE expert shapes. llama autotunes `MMQ_X/MMQ_Y` per arch (`mmq.cuh:119-170`).
  4. **No CUDA-graph / kernel-fusion** of the quantize+GEMM yet; the standalone `quantize_q8_1` launch (`lib.rs` activation staging) adds a small fixed cost per matmul.
- **The bottleneck if it stalls below 6000 will be the per-dtype decode-into-smem path and smem bank conflicts, not the MMA throughput** — at T=512 with 219 TFLOP int8 the compute is not the limiter; feeding the tensor cores conflict-free is. The path to 6240+ is then: (a) swizzle the smem tiles, (b) vectorize/tune each `decode_block_to_int8`, (c) Stage-C block-scaled FP8 (381 TFLOP, `sm120-empirical-capabilities.md:72`) which would push *past* llama's int8-MMQ ceiling on compute-heavy batches.

Net: this plan reliably gets bw24 into the same order of magnitude as llama (3500–5500 first cut), and 6240+ is reachable with the standard MMQ smem-swizzle + per-dtype load tuning as a measured follow-on — with the Stage-C FP4/FP8 paths offering headroom *above* llama since they're Blackwell-only.

---

### Files to touch (concrete)
- **New:** `crates/bw24-engine/cu/qmatvec_gemm.cu` — the 4 templated kernels + ldmatrix/mma int8 helpers (port `cu/flash_attn.cu:90-112`, swap b16→b8 and mma opcode to `m16n8k32.s32.s8.s8.s32`).
- **Edit:** `crates/bw24-engine/src/lib.rs:468-490` (`matmul_pre`) — add `m >= 16` GEMM branch + new `qmatvec_gemm` launcher; build `qmatvec_gemm.cu` into the fatbin at `lib.rs:47`.
- **Edit:** `crates/bw24-engine/src/bin/kernel_check.rs:~429` — add GEMM-vs-dp4a bit-equivalence cases.
- **Unchanged:** `cu/qmatvec.cu` (decode + Stage-A intact), `Engine::matmul` Stage-A fallback (`lib.rs:411`), `hybrid_forward.rs` call sites (`matmul` signature unchanged).