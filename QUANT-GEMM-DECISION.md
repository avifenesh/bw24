# QUANT-GEMM-DECISION.md

Resident-quantized GEMM strategy for the bw24 engine (RTX 5090 Laptop, sm_120, GB203, 24 GB, 847 GB/s, 82 SMs).

Status: decision for Task #8 (unblock OOM). Decisive, copy-oriented. Grounded in the live engine code and the local llama.cpp checkout (`/home/avifenesh/projects/llama.cpp/ggml/src/ggml-cuda/`, all line numbers below verified against that tree on 2026-06-26).

---

## 0. The bug we are killing (verified, exact location)

`crates/bw24-engine/src/model.rs` dequantizes **every** weight to f32 on load:

```rust
// model.rs:49 and :56
let f32v = dequant::dequantize(t.ggml_type, g.tensor_data(t), n);   // -> Vec<f32>
Ok(GpuTensor { data: e.htod(&f32v)?, ne: t.ne.clone() })            // GpuTensor.data: CudaSlice<f32>
```

`GpuTensor` (model.rs:12) is hardcoded `CudaSlice<f32>`. The forward pass (`forward.rs`) then calls `e.linear(&h, &layer.wq.data, ...)` which is the cuBLASLt **f32** GEMM (`runtime/src/lib.rs:59 linear_f32`). Result: a Q8_0 9B (8.9 GB on disk) becomes ~36 GB f32 -> OOM; 27B Q4_K_M (15.8 GB) -> 59 GB -> OOM. This is the whole problem.

The fix: **weights stay in their native GGUF block bytes resident in VRAM**, and dequant happens *inside* the matmul. VRAM footprint then equals on-disk quant size. This document is the design for that.

---

## 1. Chosen design: llama.cpp MMVQ (decode) + MMQ (prefill), quant-domain, per-tensor-dtype dispatch

We adopt the llama.cpp ggml-cuda quant-domain matmul as the **baseline and primary** engine GEMM. It is the only stack that natively dispatches the **full mixed per-tensor GGUF dtype set** (Q8_0 + Q4_K + Q6_K + NVFP4 + BF16 in one file) with weights staying packed. The two other candidates evaluated — vLLM CUTLASS sm120 NVFP4 and ExLlamaV3 EXL3 — are **NVFP4-only / own-format-only** and cannot read GGUF K-quants, so they are at best a per-dtype add-on (see §5), not the engine.

The engine keeps each weight tensor in its GGUF dtype bytes and, per matmul, picks the kernel by **M** (= number of activation rows / tokens, `src1->ne[1]`) and by **src0 dtype**:

| Regime | M | Kernel | Mechanism | Tensor cores? |
|---|---|---|---|---|
| **DECODE** (daily hot path) | M ≤ 8 | **MMVQ** `mul_mat_vec_q` | Activation row quantized to q8_1; each warp decodes one weight block and `__dp4a`-dots it. Memory-bound. | No (pure DP4A, cc≥610) |
| **PREFILL** | M > 8 | **MMQ** `mul_mat_q` | Activation tiled to q8_1 (or block-fp4 for native FP4); weight blocks staged to smem as int8/fp4 tiles; int8 / block-scaled-fp4 tensor-core MMA. Compute-bound. | Yes (Turing int8 mma always on sm_120; native FP4 mma optional) |

Dispatch thresholds are already in-tree and verified:
- `MMVQ_MAX_BATCH_SIZE = 8` -> `ggml_cuda_should_use_mmvq` (mmvq.cu:280).
- `ggml_cuda_should_use_mmq` (mmq.cu:267); on NVIDIA with fp16-mma it returns `ne11 < MMQ_DP4A_MAX_BATCH_SIZE (=64)` for the dp4a-vs-mma sub-choice (mmq.cu:320), but on sm_120 `turing_mma_available(1200)==true` so the MMA path is selected at **compile time** inside the kernel — sm_120 prefill always runs int8 tensor cores.
- Host selection idiom to copy: `ggml-cuda.cu:2554-2585` (`use_mul_mat_vec_q` / `use_mul_mat_q`).

### Per-dtype decision (only the dtypes our daily GGUFs actually use: Q8_0 / Q4_K / Q6_K / NVFP4)

| dtype | DECODE (M≤8) | PREFILL (M>8) | Notes |
|---|---|---|---|
| **Q8_0** | `vec_dot_q8_0_q8_1` (vecdotq.cuh:797), DP4A | MMQ int8: `mma.sync...m16n8k32.s32.s8.s8.s32` (mma.cuh:946) | Block = 32 vals + fp16 d (34 B). Already validated by our CPU oracle (`dequant.rs:64`). |
| **Q4_K** | `vec_dot_q4_K_q8_1` (vecdotq.cuh:864), DP4A | MMQ int8, K-quant load_tiles | Super-block QK_K=256, 144 B. Oracle: `dequant.rs:91`. |
| **Q6_K** | `vec_dot_q6_K_q8_1` (vecdotq.cuh:956), DP4A | MMQ int8, K-quant load_tiles | QK_K=256, 210 B. Oracle: `dequant.rs:119`. |
| **NVFP4** | `vec_dot_nvfp4_q8_1` (vecdotq.cuh:331), DP4A via `kvalues_mxfp4` table | **Two paths** — see below | block16 / ue4m3 sub-scales, 64 vals + 4 B scales (36 B). No CPU oracle yet; **add one** (action item). |
| **BF16** (norms tiny; only big ones are `token_embd` / `output`/lm_head) | not quantized | not quantized | Keep f32/f16 cuBLASLt path. Dequant-to-f16 the 2-3 big BF16 tensors only. See §5. |

NVFP4 prefill has two MMQ paths, both present in this checkout:
- **Native FP4 (fast)**: `mma_block_scaled_fp4<GGML_TYPE_NVFP4>` (mma.cuh:1126) issues
  `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3` (mma.cuh:1145).
  Gated by `BLACKWELL_MMA_AVAILABLE` (common.cuh:286-288, `__CUDA_ARCH__>=1200`). This is the exact block-scale FP4 MMA family this project measured at **762 TFLOP**. (Note: MXFP4 uses a *different* instruction — `kind::mxf4 ... scale_vec::2X ... ue8m0`, mma.cuh:1138 — do not conflate the two; NVFP4=4X/ue4m3, MXFP4=2X/ue8m0.)
- **Generic int8 fallback**: `load_tiles_nvfp4` (mmq.cuh:1069) + `MMQ_MMA_TILE_X_K_NVFP4` (mmq.cuh:221) unpack e2m1 and run the same Turing int8 mma. Runs on sm_120 with **zero** Blackwell/CUDA-12.8 dependency.

**Decision:** ship the generic int8 NVFP4 path first (correctness, no toolchain risk), add the native-FP4 path as the measured fast path once the int8 path is bit-validated.

---

## 2. The SIMPLEST CORRECT bridge to fit 24 GB THIS step, and the fast path after

The fastest *correct* thing that stops the OOM is **not** "port all of MMQ." MMVQ alone is enough to run the daily models end-to-end, because:
- Decode is M=1 -> MMVQ is *the* kernel anyway.
- Prefill at our daily prompt sizes can run MMVQ too (it accepts M up to `MMVQ_MAX_BATCH_SIZE=8`, and for larger M we can simply **loop MMVQ over groups of ≤8 columns** as a correctness stopgap — slow but it fits memory and produces correct logits). MMQ is the *throughput* upgrade for prefill, not a correctness requirement.

### Stage A — bridge NOW (fits 24 GB, correct, days not weeks)

1. **Stop f32-resident.** Change `GpuTensor.data` from `CudaSlice<f32>` to raw quant bytes + dtype:
   ```rust
   // model.rs
   pub struct GpuTensor {
       pub data: CudaSlice<u8>,   // native GGUF block bytes, htod'd verbatim (zero-copy from mmap slice)
       pub ggml_type: GgmlType,
       pub ne: Vec<u64>,
   }
   ```
   In `load`/`load_opt` (model.rs:46-61): for quant types, `e.htod_bytes(g.tensor_data(t))` — **no dequant**. For BF16/F16/F32 tensors keep a small f32/f16 buffer (only norms + embed + lm_head; trivial VRAM). This single change is what fixes the OOM.

2. **Port MMVQ + the four vec_dot decoders + the q8_1 activation quantizer**, compile to a sm_120a fatbin, expose two entry points via cudarc:
   - `quantize_row_q8_1_cuda` (quantize.cu:375) — quantize the activation matrix to q8_1.
   - `mul_mat_vec_q<type, ncols_dst>` (mmvq.cu) — the decode kernel, templated `ncols_dst` 1..8.
   Wire `e.linear` to: if dtype is quantized -> quantize activation to q8_1, call MMVQ (loop columns in chunks of ≤8 for prefill M>8 as the stopgap); else -> existing `linear_f32`.

3. **Validate** every ported decoder against the existing CPU oracle (`crates/bw24-gguf/src/dequant.rs`, which already byte-matches ggml for Q8_0/Q4_K/Q6_K). Add an NVFP4 dequant to `dequant.rs` and bit-validate `vec_dot_nvfp4_q8_1`.

After Stage A: 9B Q8_0 resident = 8.9 GB, 27B Q4_K_M = 15.8 GB — both fit 24 GB with headroom. The hybrid forward (Task #6) and the generation loop (Task #7) can run.

### Stage B — fast sm_120 path (throughput)

Port **MMQ** (`mmq.cuh` + `mmq.cu` + `mma.cuh` + the q8_1/fp4 MMQ activation quantizers) for prefill M>8: int8 tensor-core tiles for Q8_0/Q4_K/Q6_K, generic int8 NVFP4. Replace the MMVQ-column-loop stopgap with real MMQ. Selection: `M<=8 -> MMVQ`, `M>8 -> MMQ` (copy `ggml-cuda.cu:2554-2585`).

### Stage C — measured FP4 fast path (optional, NVFP4 only)

Enable `BLACKWELL_MMA_AVAILABLE` for the NVFP4 MMQ tile path (`mma_block_scaled_fp4`, mma.cuh:1126) + `quantize_mmq_fp4_cuda` (quantize.cu:422) — the 762-TFLOP route. This is the only piece that *must* be built with CUDA 12.8 (see §4). Gate it behind a cargo feature; the generic int8 NVFP4 path is the fallback.

**Do NOT** build on a "dequant whole weight to f16" stopgap: it works for 9B (~18 GB f16) but 27B (~30 GB f16) still OOMs. Skip it.

---

## 3. Exact files to copy (per kernel / dtype)

All paths relative to `/home/avifenesh/projects/llama.cpp/ggml/src/ggml-cuda/`. Compile as one (or few) self-contained sm_120a fatbin(s); these TUs have no host C++ deps beyond the headers listed.

### Headers / shared (always)
- `common.cuh` — CC macros (`GGML_CUDA_CC_DP4A=610` :51, `GGML_CUDA_CC_BLACKWELL=1200` :59); `BLACKWELL_MMA_AVAILABLE` define (:286-288); `turing_mma_available` (:348); `blackwell_mma_available` (:360); `ggml_cuda_dp4a` (:694); `NO_DEVICE_CODE`.
- Block structs + constants (`block_q8_0`, `block_q4_K`, `block_q6_K`, `block_nvfp4`, `QK_*`, `QI_*`, `QR_*`, `VDR_*`) from `ggml-common.h`; `ggml_cuda_type_traits<T>::{qk,qi}`.

### Stage A — DECODE (MMVQ)
- `mmvq.cu` + `mmvq.cuh` — host `ggml_cuda_mul_mat_vec_q`, switch `mul_mat_vec_q_switch_type`, kernel `mul_mat_vec_q`, `get_vec_dot_q_cuda` (mmvq.cu:10-33), `ggml_cuda_should_use_mmvq` (:280), `MMVQ_MAX_BATCH_SIZE`.
- `vecdotq.cuh` — `vec_dot_q8_0_q8_1` (:797), `vec_dot_q4_K_q8_1` (:864), `vec_dot_q6_K_q8_1` (:956), `vec_dot_nvfp4_q8_1` (:331), `kvalues_mxfp4` table + `get_int_from_table_16`.
- `quantize.cu` + `quantize.cuh` — `quantize_row_q8_1_cuda` (:375) (decode activation quant).

### Stage B — PREFILL (MMQ)
- `mmq.cuh` (~4176 lines) — tile structs, `mmq_type_traits` (:3266+) dispatch table, `load_tiles_*`, `vec_dot_*_q8_1_mma` / `_dp4a`, `mul_mat_q` kernel, `launch_mul_mat_q`, K-quant load_tiles, `load_tiles_nvfp4` (:1069), `vec_dot_fp4_fp4_mma` (:996), `MMQ_MMA_TILE_X_K_NVFP4` (:221).
- `mmq.cu` — `ggml_cuda_mul_mat_q` (:77), `ggml_cuda_mul_mat_q_switch_type` (:6), `ggml_cuda_should_use_mmq` (:267); per-type template instances under `template-instances/`.
- `mma.cuh` — int8 `mma.sync...m16n8k32.s32.s8.s8.s32` (:946) and m16n8k16 (:924); ldmatrix wrappers.
- `quantize.cu` — `quantize_mmq_q8_1_cuda` (:392) (prefill activation quant).

### Stage C — native FP4 (NVFP4 only, CUDA 12.8 TU)
- `mma.cuh` — `mma_block_scaled_fp4` (:1126), NVFP4 string at :1145.
- `mmq.cuh` — `block_fp4_mmq` (:52), `load_tiles_nvfp4_nvfp4` (:947), the `#if BLACKWELL_MMA_AVAILABLE` NVFP4 branches.
- `quantize.cu` — `quantize_mmq_fp4_cuda` (:422), `quantize_mmq_nvfp4` (:78).

### Engine side (Rust)
- `crates/bw24-engine/src/model.rs` — change `GpuTensor` to bytes+dtype (§2.1).
- `crates/bw24-runtime/src/lib.rs` — add `mul_mat_vec_q` / `mul_mat_q` cudarc launch wrappers next to `linear_f32`; add `htod_bytes`.
- `crates/bw24-engine/src/forward.rs` — `e.linear(...)` dispatches on `weight.ggml_type` and M.
- Reuse `crates/bw24-gguf/src/dequant.rs` as the per-kernel bit-validation oracle (add NVFP4).

Do **not** port `mmid.cu` / MoE / stream-k fixup (daily models are dense/hybrid, not MoE). Do **not** port `convert.cu` for the quant-resident path (only if you want the f16 fallback for the BF16 embed/lm_head, which `dequant.rs` already covers on CPU).

---

## 4. Build implication: dual-toolkit cargo + nvcc fatbin

The CUDA 13.x nvcc `-O3` miscompile of MMQ/MXFP4 on sm_120 is real and confirmed (llama.cpp #18331/#18398/#20195, ollama #14374): illegal memory access / "device kernel image is invalid" / ptxas "mma with block scale not supported". Both toolkits are installed here: `/usr/local/cuda-12.8` and `/usr/local/cuda-13.1`. Current `build.rs` hardcodes 13.1 `-O3` for *all* `.cu`.

**Structure: split the .cu TUs into two fatbins by toolchain risk.**

| Fatbin | TUs | nvcc | flags | why |
|---|---|---|---|---|
| `engine.fatbin` (existing) | `kernels.cu`, `hybrid.cu`, MMVQ + q8_1-quant TU, NVFP4-generic-int8 | 13.1 | `-O3` | MMVQ is light (DP4A, no block-scale mma); tolerates 13.1. cuBLASLt f32 path stays 13.1. |
| `mmq.fatbin` (new) | MMQ TU (`mmq.cu`/`mmq.cuh`), the native-FP4 TU (`mma_block_scaled_fp4`, `quantize_mmq_fp4`) | **12.8** | **`-O2`** (belt-and-suspenders) | The exact TUs that miscompile under 13.1 `-O3`. 12.8 is llama.cpp's own documented Blackwell requirement. |

Both fatbins use the same `-gencode arch=compute_120a,code=sm_120a` already proven on-device. Link both via cudarc (`cudarc::nvrtc`/module-load by path, same as the existing `BW24_ENGINE_FATBIN` env-var pattern).

Concrete `build.rs` change (the file already shells out to nvcc per-TU; just add a per-TU nvcc + flags):

```rust
// build.rs — extend the loop to carry (src, env, nvcc_path, opt)
let nvcc_13 = std::env::var("BW24_NVCC").unwrap_or("/usr/local/cuda-13.1/bin/nvcc".into());
let nvcc_12 = std::env::var("BW24_NVCC_128").unwrap_or("/usr/local/cuda-12.8/bin/nvcc".into());
for (src, env, nvcc, opt) in [
    ("cu/kernels.cu",     "BW24_ENGINE_FATBIN", &nvcc_13, "-O3"),
    ("cu/hybrid.cu",      "BW24_HYBRID_FATBIN", &nvcc_13, "-O3"),
    ("cu/mmvq.cu",        "BW24_MMVQ_FATBIN",   &nvcc_13, "-O3"),  // Stage A
    ("cu/mmq.cu",         "BW24_MMQ_FATBIN",    &nvcc_12, "-O2"),  // Stage B/C: 12.8 + O2
] {
    // ... nvcc -gencode arch=compute_120a,code=sm_120a {opt} --fatbin -o {fatbin} {src}
}
```

This is fully compatible with the existing cargo+separate-nvcc model: the Rust crate still builds under the system toolchain; only the device fatbins use two nvcc binaries. Keep `-O2` on the MMQ TU even with 12.8 as insurance.

---

## 5. NVFP4 GGUFs vs Q8_0/Q4_K mixed GGUFs — which is the primary path?

**Decision: ship Q8_0/Q4_K mixed GGUFs as the primary, correctness-first daily path NOW; treat NVFP4 GGUFs as the VRAM/throughput upgrade behind Stage C.** Reasons, decisive:

1. **What we actually have on disk today is Q8_0** (`/home/avifenesh/ai-ml/models/qwen3.5-9b-judge-q8_0.gguf`) and f16 — **no NVFP4 GGUF is present**. The Q8_0/Q4_K path is the one that runs against real files this step. NVFP4 is aspirational until we have/produce the NVFP4 GGUFs.
2. **NVFP4 only fixes one slice of a mixed file.** Daily GGUFs are mixed per-tensor (NVFP4 + Q4_K + Q6_K + Q8_0 + BF16). Even an "NVFP4" model still has Q*/BF16 tensors that need MMVQ/MMQ. So MMVQ/MMQ is required regardless; NVFP4 is additive, not a replacement engine.
3. **NVFP4's wins are real but back-loaded:** VRAM (9B 5.3 GB, 27B 5.6 GB — huge 24 GB headroom) and prefill throughput (762 TFLOP block-scale MMA). But the native FP4 path is the *only* component that needs the CUDA-12.8 build and the `BLACKWELL_MMA_AVAILABLE` gate — i.e. the highest-toolchain-risk piece. Putting it last keeps the bring-up unblocked.
4. **At M=1 decode (the daily hot path) NVFP4's advantage is bandwidth, not math.** Decode is memory-bound; the win is fewer weight bytes/token (NVFP4 9B ~5.3 GB -> ~160 tok/s ceiling at 847 GB/s vs Q8_0 8.9 GB -> ~95 tok/s). That is purely the MMVQ `vec_dot_nvfp4_q8_1` path — no native FP4 MMA needed. So even the NVFP4 *decode* benefit lands in Stage A; only NVFP4 *prefill* throughput needs Stage C.

**Net ordering:**
- **Now (Stage A):** MMVQ for Q8_0/Q4_K/Q6_K (+ generic-int8 NVFP4 decode if an NVFP4 file appears). Fixes OOM, runs daily 9B/27B Q8_0/Q4_K_M in 24 GB.
- **Next (Stage B):** MMQ prefill, int8 tensor cores, all four dtypes.
- **Then (Stage C):** native FP4 MMA for NVFP4 tensors (762 TFLOP) under CUDA 12.8 — prefer NVFP4 GGUFs at this point for the VRAM headroom + prefill speed.

vLLM CUTLASS sm120 NVFP4 (`nvfp4_scaled_mm_sm120_kernels.cu`) and ExLlamaV3 EXL3 stay **out of the engine**: vLLM only reads its own NVFP4/compressed-tensors layout (not GGUF K-quants) and EXL3 requires re-quantizing every model into its own format. Keep them only as reference / future alternate-ingest. The dense-NVFP4 CUTLASS kernel is the one component worth revisiting later *if* we want to beat MMQ's native-FP4 prefill on pure-NVFP4 tensors, but it is necessary-but-not-sufficient (solves only the NVFP4 share) and is not on the critical path.

---

## 6. Validation gate (every ported kernel)

For each dtype, generate a random weight tensor + activation, run the GPU MMVQ/MMQ kernel, and compare against `cpu_linear(dequant::dequantize(weight), activation)` (`runtime/src/lib.rs:15` + `gguf/src/dequant.rs`). Q8_0/Q4_K/Q6_K oracles already exist and byte-match ggml; **add an NVFP4 dequant to `dequant.rs`** before trusting `vec_dot_nvfp4_q8_1`. Tolerance: q8_1-domain dot products are exact-int up to the per-block fp16 scale rounding — expect agreement to ~1e-2 relative, tighter for Q8_0.

---

## TL;DR

1. Port **llama.cpp MMVQ (decode) + MMQ (prefill)**, quant-resident, per-tensor-dtype dispatch by M (≤8 MMVQ, >8 MMQ). It is the only stack covering the full mixed GGUF dtype set.
2. **Bridge now (Stage A):** flip `GpuTensor` to raw quant bytes (kills the f32-resident OOM at `model.rs:49`), port MMVQ + the 4 vec_dot decoders + q8_1 quantizer, loop MMVQ over ≤8-col chunks for prefill as a stopgap. Fits 9B Q8_0 (8.9 GB) and 27B Q4_K_M (15.8 GB) in 24 GB. **Then** MMQ (Stage B), **then** native FP4 762-TFLOP path (Stage C).
3. Copy: `common.cuh`, `mma.cuh`, `vecdotq.cuh`, `mmq.cuh`, `mmq.cu`, `mmvq.cu/.cuh`, `quantize.cu/.cuh` (+ ggml-common.h block structs). Stage A needs only mmvq+vecdotq+quantize(q8_1).
4. **Build:** keep MMVQ/cuBLASLt on CUDA **13.1 -O3**; build the **MMQ + native-FP4 TUs on CUDA 12.8 -O2** into a separate `mmq.fatbin`; both `-gencode arch=compute_120a,code=sm_120a`; link both via cudarc. Extend the existing per-TU nvcc loop in `build.rs`.
5. **Primary path = Q8_0/Q4_K mixed GGUFs** (what we have on disk, correctness-first, no toolchain risk). **NVFP4 GGUFs are the VRAM+prefill upgrade** delivered with Stage C; their decode benefit (bandwidth) already lands in Stage A via `vec_dot_nvfp4_q8_1`.
