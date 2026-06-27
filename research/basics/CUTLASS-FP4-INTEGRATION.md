# CUTLASS-FP4-INTEGRATION — adopting CUTLASS 4.2 sm120 FP4 for bw24 prefill

Status: PLAN (read-only research complete; no engine code built or edited this workflow — the FA-prefill build owns `flash_attn.cu` in parallel). Every TFLOP number is GPU-event-measured on THIS box (RTX 5090 Laptop, sm_120a / GB203, nvcc 13.1, sglang 0.5.12). Every pp512 number is Amdahl arithmetic over ONE measured kernel ratio and is a **projection, not a measurement** — each phase below is gated on an on-device re-measure before the next.

Cross-reference: this plan deliberately **conflicts with `PREFILL-GEMM-REBUILD.md` / `PREFILL-LEVERS-RANKED.md`**, which reject wholesale CUTLASS adoption and choose the in-house MMQ-pad rebuild instead. §0 and §3 reconcile the two honestly. The short version: the two plans are *not* mutually exclusive and the cheap in-house pad should be tried FIRST; CUTLASS is the justified fallback if the pad cannot reach ~40% of FP4 peak in-house.

---

## 0. The decision in one paragraph

**ADOPT CUTLASS sm120 FP4 for the prefill FP4 GEMM at m>=128 (prefill / chunked-prefill / large speculative batches); KEEP bw24's dp4a/MMVQ GEMV for decode m=1..4; KEEP bw24's hand-rolled mxf4 GEMM for the m∈[16,128) middle band — BUT only after the in-house FP4 smem-pad (the `%8==4` K-stride fix from PREFILL-GEMM-REBUILD.md) is tried first and demonstrably fails to reach ~40% of FP4 peak.** CUTLASS measured 206–232 TFLOP/s at the real model prefill shapes on this *thermally throttled* box (49–52% on the square-4096 ref; the brief's 63%/483-TFLOP is the cool-clock ceiling — treat absolute TFLOP as a FLOOR). bw24's current hand-roll measured ~120 TFLOP/16% of peak. So CUTLASS is ~1.7–2x faster on this box at the model shapes, ~4x at cool-clock square. That is a real GEMM win. **But it is Amdahl-capped:** the FP4 GEMM is only 22.5% of prefill, so FP4→CUTLASS *alone* moves pp512 from 2090 to ~2355–2515 (ceiling 2697 if FP4 went infinitely fast) — about +13–20%, ~0.4x of the llama 6139 target. CUTLASS gives **parity-via-copy** on the FP4 GEMM; the bw-EDGE that actually beats llama is everything CUTLASS does NOT do (§4): selective-expert MoE, q8_0-K/q5_1-V KV-quant, the decode MMVQ path, batched MTP partial-accept replay, and the in-flight FA-prefill rebuild.

---

## 1. THE DECISION — verified TFLOP-per-shape evidence

Model dims (qwen35-9b, `text_config`): hidden=4096, intermediate=12288, head_dim=256, n_heads=16, n_kv=4, vocab=248320. `762 TFLOP/s` is the sm_120 FP4 TC peak at the nominal 3090 MHz; the box was thermal-throttling (88C, SM clock 1875–2287 MHz = 60–74% nominal) for the whole run, so `%@clk` is the achievable-at-throttled-clock fraction. **Re-run cool/clock-locked to get the true ceiling — treat absolute TFLOP as a floor.**

### PREFILL m=512 (the binding shape) — CUTLASS `cutlass_scaled_fp4_mm`, GPU-event timed

| GEMM | m | out_f(n) | in_f(k) | us | TFLOP/s | %762 | %@clk |
|---|---|---|---|---|---|---|---|
| attn proj | 512 | 4096 | 4096 | 74.2 | 231.6 | 30.4% | 41.1% |
| ffn_gate/up | 512 | 12288 | 4096 | 250 | 206 | 27.1% | 36.6% |
| ffn_down | 512 | 4096 | 12288 | 234 | 220 | 28.9% | 40.9% |
| lm_head | 512 | 248320 | 4096 | 4544 | 229 | 30.1% | 49.6% |
| square 4096^3 ref | 4096 | 4096 | 4096 | 606 | 227 | 29.8% | 49.1% |

bw24 hand-roll (same FP4 weights, measured this session): ~120 TFLOP/s, ~16% of peak, 986us for the model's FP4 GEMM. **CUTLASS is ~1.7–2x on this throttled box; the cool-clock 483-TFLOP brief ref is ~4x.**

### DECODE small-M — CUTLASS pure-kernel latency (us), m has ~zero effect

| GEMM | out_f | in_f | m=1 | m=2 | m=4 |
|---|---|---|---|---|---|
| attn proj | 4096 | 4096 | 37.5 | 39.2 | 38.4 |
| ffn_gate | 12288 | 4096 | 96.5 | 95.4 | 95.3 |
| ffn_down | 4096 | 12288 | 102.2 | 94.9 | 102.3 |
| lm_head | 248320 | 4096 | 1668 | 1697 | 1662 |

At m=1..8 CUTLASS shows 0.1–1.3% of peak — it launches the full 128-row M-tile and discards 127/128 of the work. Latency is flat m=1→m=4: **zero batching benefit** in the B=2–4 concurrent-agent regime. A decode step has ~100+ matmuls → tens of ms/token on CUTLASS vs bandwidth-bound MMVQ reading each weight once.

### Verdict by shape

- **m=1..4 (decode): KEEP bw24 MMVQ / dp4a GEMV.** CUTLASS loses 30–100x. The existing `m >= 16` guard at `lib.rs:793` already excludes decode from `try_fp4_gemm` — correct as-is.
- **m∈[16,128) (small prefill / spec batches): KEEP bw24's `qmatvec_gemm_nvfp4_fp4`.** CUTLASS's 128-row M-tile wastes work below 128 (measured flat latency m=1→4). Crossover is exactly the M-tile = 128.
- **m>=128 (prefill m=512, chunked-prefill, large spec): ADOPT CUTLASS** — the measured 206–232 TFLOP/s (vs ~120) is the win; lm_head also wins.
- Threshold `m >= 128` is structurally justified (the M-tile), tunable to 64 via env if real shapes argue.

### Scale-factor format: GGUF and CUTLASS-NVFP4 AGREE — NO conversion

- This is the **W4A4_NVFP4_NVFP4** GemmType (`fp4_gemm_cutlass_sm120.cu:43`), SFVecSize=16 (one ue4m3 / 16 elems). `ptr_SFA/SFB` are `cutlass::float_ue4m3_t` (`fp4_gemm_template_sm120.h:117-118,239`); operand SF type `float_ue4m3_t` (`float_subbyte.h:510-511`). GGUF block_nvfp4 stores **4 ue4m3 bytes / 64-elem block (one per 16-elem subblock) = identical dtype, identical granularity, identical semantics.** No mxf4/ue8m0 conversion. (mxf4 = ue8m0 8-bit-exp scales is a DIFFERENT path; this is nvf4 = ue4m3, exactly what GGUF stores.)
- The 2x/0.5x cancellation in bw24's own kernel (`qmatvec_gemm.cu:874-876`: "GGUF e2m1 codebook = 2x HW e2m1, GGUF ue4m3 = 0.5x HW ue4m3, factors cancel") is **not a bw24 hack** — it is exactly how raw GGUF bytes map onto the STANDARD HW e2m1/ue4m3 the MMA already assumes. Numerically reproduced: GGUF codebook {0,1,2,3,4,6,8,12} = 2x standard E2M1 {0,.5,1,1.5,2,3,4,6}; GGUF ue4m3 = 0.5x bias-7 ue4m3; the product is bit-identical to the standard product. So **raw GGUF bytes carry identical real values into CUTLASS's standard-semantics MMA**. The only numeric difference from bw24 is accumulation/output dtype (CUTLASS bf16 vs bw24 f32) — gate on rel-error + argmax, not bit-equality (§2.5).

### The MMA atom differs (k32 atom vs bw24's k64) — but you do NOT touch it

- CUTLASS sm120 atom: `SM120_16x8x32_TN_VS<e2m1,e2m1,f32,ue4m3,VS=16>` → `mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32`, k=32, warp-level, fed by `SM90_TMA_LOAD` + warp-specialized pipeline (`sm120_blockscaled_mma_tma.hpp`). The faithful standalone probe (`/tmp/cutlass_probe/`) confirmed the compiled SASS contains 64x OMMA (the k64-packed `SM120_16x8x64_TN_VS` form) + UTMALDG/UTMASTG/BULK (TMA) + `SM90_TMA_LOAD` — NOT wgmma, NOT tcgen05. This TMA + warp-spec pipeline is the structural reason CUTLASS hits ~50–63% where bw24's cp.async-only hand-roll hits 16%.
- bw24 uses `m16n8k64.kind::mxf4nvf4.block_scale.scale_vec::4X.ue4m3`. Different tiling, **but you do not touch the atom — CUTLASS owns it internally.** You only supply operands in CUTLASS's expected gmem layout. `fp4_shift_A/B` (`mma_traits_sm120.hpp:220-255`) is internal; the gmem operand is standard packed-2-nibbles-per-byte e2m1 — no pre-shift in the repack.

---

## 2. INTEGRATION BUILD PLAN

### Core architecture decision: static lib, NOT fatbin

CUTLASS is a C++ template library — it cannot go through bw24's existing `nvcc --fatbin → load_module` path (`build.rs`), which produces device-code-only fatbins with no host launcher. CUTLASS's collective needs its host-side `GemmUniversalAdapter::run()` (grid calc, workspace, TMA-descriptor setup) which is *host* C++. So the 6-fatbin pattern (`build.rs`) does NOT extend. CUTLASS becomes a **7th artifact of a different kind: a static lib with one `extern "C"` host entry, called over FFI** — not `load_function`. The 6 fatbins stay byte-for-byte unchanged, so the parallel `flash_attn.cu` FA-prefill build is untouched.

### 2.1 Wrapper signature (exact) — `crates/bw24-engine/cu/cutlass_fp4_sm120.cu` (NEW)

Copy flashinfer's `fp4_gemm_cutlass_sm120.cu` + `fp4_gemm_cutlass_template_sm120.h` 1:1 (in-scope per brief), expose ONE config:

```c
// compiled to a sm_120a static lib; instantiates ONE config:
//   CutlassFp4GemmRunner<OutT, W4A4_NVFP4_NVFP4>, CtaShape128x128x128B + DP scheduler.
// CORRECTION (verified on-box): OutT MUST be cutlass::bfloat16_t (or float), NOT __nv_bfloat16 —
// the native type hits a hard static_assert "Unknown TMA Format!" in copy_sm90_desc.hpp:236.
// flashinfer maps __nv_bfloat16 -> cutlass::bfloat16_t via cutlass_dtype<T>::type. Preserve that.
// For the GATE run, instantiate OutT=float (keep prefill logits f32); switch to bf16 only after gate passes.
extern "C" int bw24_cutlass_fp4_gemm(
    const void* a_e2m1,   // activation nibbles [m, k/2] packed 2/byte
    const void* b_e2m1,   // weight     nibbles [n, k/2] CUTLASS K-major layout
    const void* sfa,      // act scales  float_ue4m3_t, swizzled SfAtom layout
    const void* sfb,      // wt  scales  float_ue4m3_t, swizzled SfAtom layout
    float       alpha,    // 1/(gsa*gsb); fold bw24 per-tensor `scale` here
    void*       d,        // output [m, n]  f32 (gate) / bf16 (prod)
    int m, int n, int k,
    void*       workspace, size_t workspace_bytes,
    void*       stream);  // CUstream (== cudaStream_t, ABI-identical)
extern "C" size_t bw24_cutlass_fp4_workspace(int m, int n, int k);
// Plus a one-time SFB swizzle helper so the swizzle logic lives with the CUTLASS that consumes it:
extern "C" int bw24_cutlass_repack_sfb(const void* sfb_linear, void* sfb_swizzled, int n, int k, void* stream);
```

Output dtype f32 for the gate keeps logits f32 (de-risks argmax); accumulator is f32 either way (`ElementAccumulator=float`), only the epilogue downcasts.

### 2.2 `build.rs` change (additive — 6-fatbin loop unchanged so FA build is safe)

```rust
// AFTER the existing fatbin loop. Separate object, NOT in the FA fatbin TU.
// COMPILE-COST: instantiate exactly ONE (tile,schedule,dtype). Do NOT pull flashinfer's runner
// heuristic that emits every config — that is the 5-20min horror case. ONE config measured cold
// = 28.7s wall / 5.2 GB RSS on this box (verified, /tmp/cutlass_probe). rerun-if-changed keeps it
// off the incremental path; never touches the 6 fatbins or the parallel FA build.
let cutlass_inc = /* VENDOR a pinned CUTLASS 4.2 header tree into the repo (or submodule),
                     do NOT point at the venv path, for reproducibility */;
Command::new(&nvcc).args([
    "-gencode","arch=compute_120a,code=sm_120a",
    "-O3","-std=c++17","--expt-relaxed-constexpr",
    "-I", cutlass_inc, "-I", &format!("{cutlass_inc}/../tools/util/include"),
    "-c","cu/cutlass_fp4_sm120.cu","-o", obj,
]).status()...;
Command::new("ar").args(["crus", lib, obj]).status()...;
println!("cargo:rustc-link-search=native={}", out.display());
// CORRECTION (verified on-box): plain static link DROPS the fatbin-registration global ctor
// (_ZL24__sti____cudaRegisterAllv in .init_array) -> device kernel never registers -> silent
// no-kernel-registered launch failure. MUST whole-archive the lib:
println!("cargo:rustc-link-arg=-Wl,--whole-archive");
println!("cargo:rustc-link-arg={}", lib.display());        // libbw24_cutlass.a
println!("cargo:rustc-link-arg=-Wl,--no-whole-archive");
println!("cargo:rustc-link-lib=dylib=cudart");             // host adapter uses runtime API
println!("cargo:rustc-link-lib=dylib=stdc++");
println!("cargo:rerun-if-changed=cu/cutlass_fp4_sm120.cu");
```

Two on-box-verified corrections vs the brief's sketch: **(A) output must be `cutlass::bfloat16_t`/`float`, not `__nv_bfloat16`** (hard `static_assert` "Unknown TMA Format!" at `copy_sm90_desc.hpp:236` otherwise); **(B) `--whole-archive` is mandatory** or the CUDART fatbin-registration ctor is dropped and the kernel silently never registers (the single most likely "builds but won't launch" trap). Both demonstrated end-to-end in `/tmp/cutlass_probe/` (probe.cu → libbw24_cutlass.a → linktest, rc=0, runs).

### 2.3 cudarc FFI interop (verified against 0.19.8) — `crates/bw24-engine/src/cutlass_ffi.rs` (NEW)

- Stream: `engine.gpu.stream.cu_stream()` → `sys::CUstream` (`core.rs:732`). `CUstream == *mut CUstream_st`, ABI-identical to `cudaStream_t` — pass straight as `void*`, the cast is a no-op.
- Device ptrs: `DevicePtr::device_ptr(&stream)` → `(CUdeviceptr, SyncOnDrop)` (`core.rs:1163`). `CUdeviceptr == u64`; cast to `*const c_void`. **Hold the `SyncOnDrop` guard across the FFI call** (the one footgun — it is the sync-on-drop token).
- Wrap the `unsafe extern "C"` call in a safe `Engine::cutlass_fp4_gemm(...)` that pulls raw ptrs, allocates/caches the workspace `CudaSlice<u8>`, and calls through.
- **Driver-vs-runtime context (one smoke test required):** cudarc uses the driver API; CUTLASS's host adapter uses the runtime API. They share state only via the CUDA **primary context** — cudarc defaults to `cuDevicePrimaryCtxRetain`, so the runtime calls bind to the same context and this works. The host-only workspace query was verified; a real GEMM launch + `cudaGetLastError` smoke test is still required (Phase 0). Pass an explicit workspace buffer (sized via `bw24_cutlass_fp4_workspace`, cached for the largest prefill GEMM) so CUTLASS never calls internal `cudaMalloc` and fights the driver context.

### 2.4 Weight repack at load (one-time) — EDIT `crates/bw24-engine/src/model.rs:47-63`

In the `Some(qt)` arm, gated `qt == QT_NVFP4 && BW24_FP4_CUTLASS`. Add a `CutlassWeight` (repacked B + swizzled SFB device buffers) alongside the existing `Quant{bytes,...}` — **keep `bytes` as-is** so the decode MMVQ/dp4a path (reads raw GGUF) is untouched. Both layouts coexist: decode reads `bytes`, prefill reads CUTLASS buffers. Two one-time transforms:

1. **Nibble de-interleave (B operand).** GGUF block_nvfp4 stores element k at `qs[(k/16)*8 + (k%16 & 7)]`, low/high nibble by `k%16<8` — a per-16-subblock interleave. CUTLASS wants plain K-contiguous packed e2m1 (2/byte). One-time gather (identical math to bw24's existing inline smem repack, but to a gmem buffer once). e2m1 bits copied unchanged (no shift — `fp4_shift_A/B` is internal).
2. **Scale scatter (SFB).** GGUF: 4 ue4m3 bytes/64-block in linear (row, k/16) order. CUTLASS: swizzled `Sm1xxBlockScaledConfig<16>` SfAtom layout, dest shape `(round_up(n,128), round_up(k/16,4)/4)` int32, dtype unchanged. **Do NOT hand-roll the swizzle — use CUTLASS's `tile_atom_to_shape_SFB(make_shape(N,K,L))`** to generate the destination layout, run inside the wrapper TU (`bw24_cutlass_repack_sfb`) so the swizzle lives with the version of CUTLASS that consumes it (the #1 silent-corruption site).

Transform #1: small host-side gather on raw bytes before H2D (simplest, one-time cost negligible) or a tiny repack kernel. Transform #2: the CUTLASS-side helper.

**alpha:** GGUF NVFP4 has no global weight scale (per-16 ue4m3 is the only one), so `global_scale_b = 1`. Fold bw24's existing per-tensor `scale` (`model.rs:55-61`) into `alpha = 1/scale` instead of the post-matmul `scale_inplace` (`lib.rs:765`) — removes one kernel from the prefill path. **Verify `scale_inplace` is a pure multiply with no clamp/round** (it is, per code) — the one place a silent gate mismatch could enter.

### 2.5 Dispatch seam — EDIT `crates/bw24-engine/src/lib.rs:882-894` (`try_fp4_gemm`)

The seam already exists, reached from `matmul_pre:793-797` (gated `m >= 16`). Change `try_fp4_gemm`:

```rust
fn try_fp4_gemm(&self, w, x, m, in_f, out_f) -> Result<Option<...>> {
    if std::env::var("BW24_FP4").is_err() { return Ok(None); }
    if let GpuTensor::Quant { qtype, scale, cutlass: Some(cw), .. } = w {
        if *qtype == QT_NVFP4 && in_f % 64 == 0 {
            if m >= 128 && std::env::var("BW24_FP4_CUTLASS").is_ok() {           // NEW
                let (aq, sfa) = self.quantize_fp4_act_cutlass(x, m, in_f)?;       // swizzled act + SFA
                let alpha = 1.0 / *scale;                                        // fold per-tensor scale
                let y = self.cutlass_fp4_gemm(&cw.b, &aq, &sfa, &cw.sfb, alpha, m, out_f, in_f)?;
                return Ok(Some(y));
            }
            let y = self.qmatvec_gemm_nvfp4_fp4(...)?;   // m in [16,128): existing hand-roll
            return Ok(Some(y));
        }
    }
    Ok(None)
}
```

Add `quantize_fp4_act_cutlass` (port sglang `scaled_fp4_quant`'s swizzle into a small kernel — this is PER-TOKEN, must be fast; if slow it eats the GEMM win — measure in end-to-end prefill, not GEMM-in-isolation). Add a cached CUTLASS workspace field to `Engine`.

### 2.6 Argmax + bit-exact gate — NEW `kernel_check` arm

CUTLASS (bf16/f32 epilogue) will NOT be bitwise identical to bw24's f32-accumulated kernel — both are NVFP4-correct. Validate via existing `kernel_check` rel-error + the argmax gate (confirm the actual constant in `kernel_check` — brief says both 268/220 and 268/271; the code is source of truth), NOT bit equality. Add an arm that runs CUTLASS on the same weight and compares against decode-of-same-weight. For the gate, instantiate `OutT=float`. **The de-interleave + SFB swizzle repack is the ONLY place a silent wrong-answer hides** (exactly the failure mode bw24's own smem-swizzle comments flag, `qmatvec_gemm.cu:905-906`) — this gate is the mitigation.

### 2.7 Compile-time risk mitigation (ranked)

1. **Layout-map corruption (highest).** Mitigation: §2.6 gate; use `tile_atom_to_shape_SFB`, never hand-roll the swizzle.
2. **CUTLASS compile time.** ONE config + separate TU + `rerun-if-changed`. Measured 28.7s/5.2GB cold for one config; the heuristic sweep is what blows up — explicitly do NOT instantiate StreamK/other configs. Vendor pinned headers.
3. **cudarc FFI.** Low — `cu_stream()`/`device_ptr()` present in 0.19.8; footgun is the `SyncOnDrop` guard lifetime (hold across the call).
4. **Link.** `--whole-archive` (§2.2) + `-lcudart -lstdc++`; explicit workspace buffer; primary-context dependency (§2.3 smoke test).
5. **Per-token SFA quant cost.** Measure in end-to-end prefill.

### Files touched (summary; all absolute under `crates/bw24-engine/`)
- NEW `cu/cutlass_fp4_sm120.cu` — wrapper TU (flashinfer 1:1) + `bw24_cutlass_repack_sfb`.
- NEW `src/cutlass_ffi.rs` — extern "C" + safe `Engine::cutlass_fp4_gemm`, `cutlass_fp4_workspace`.
- EDIT `build.rs` — static-lib compile + ar + `--whole-archive` link (additive; 6-fatbin loop unchanged).
- EDIT `src/model.rs:47-63` — `CutlassWeight` repack buffers, gated `BW24_FP4_CUTLASS`.
- EDIT `src/lib.rs:882-894` (`try_fp4_gemm`) — `m>=128` CUTLASS branch; `quantize_fp4_act_cutlass`; Engine workspace field.
- NEW `kernel_check` arm — CUTLASS-vs-decode validation.

---

## 3. HONEST pp512 PROJECTION (Amdahl math, no best-case summing)

Measured starting point (TRUSTED nsys trace, `BW24_GEMM=1 BW24_FP4=1`): **pp512 = 2090 tok/s = 0.34x of llama 6139.** (The brief's 882/43x is STALE pre-GEMM-rebuild — the int8 batched GEMM already landed and killed the 512x weight-re-read; that 43x hole is already mostly closed. Use 2090 and target 6139, not 5450.)

Prefill profile (nsys, T=512, FP4 on):
- **GEMMs = 58%**: nvfp4_fp4 22.5% + q4_K 19.4% + q5_K 16.1% + q8_0 4.4%. (The int8 cluster q4_K+q5_K+q8_0 = **39.9%** — LARGER than FP4, and CUTLASS-fp4 cannot touch it.)
- **SSM/attn = 27%**: ssm_conv1d_silu 10.9% + fa_prefill 10.4% + gdn_scan 6%.
- Other ~15%.

### FP4→CUTLASS ALONE (the headline, honestly capped)

CUTLASS speedup over bw24's current FP4 kernel at the REAL model shapes is **~1.7–2x** (206–232 vs ~120 TFLOP), NOT the ~4x square-cool-clock ref:

| Scenario | FP4 GEMM speedup | pp512 | vs llama 6139 |
|---|---|---|---|
| CUTLASS ~2x (throttled floor) | 2.0x | 2090/(1−0.225·0.50) = **~2355** | 0.38x |
| CUTLASS ~4x (cool-clock) | 4.0x | 2090/(1−0.225·0.75) = **~2515** | 0.41x |
| FP4 → infinitely fast (ABSOLUTE CEILING) | ∞ | 2090/(1−0.225) = **~2697** | 0.44x |

**FP4-via-CUTLASS alone gets prefill to ~2355–2515 (ceiling 2697): +13–20% over 2090, still ~0.4x of llama. It does NOT move pp512 "toward/past" llama.** The brief's "THE move for closing the gap" overstates a sub-slice fix.

### The full path to beat llama 6139 (everything that must ALSO move)

Even infinitely-fast GEMMs cap the stack at ~5559 (`PREFILL-GEMM-REBUILD.md:115`) because of the 27% SSM/attn cluster — **no GEMM lever, CUTLASS or in-house, reaches 6139 alone.** The honest ranked path:

1. **The in-house FP4 smem-pad FIRST (cheaper, lower-risk, possibly sufficient).** `PREFILL-GEMM-REBUILD.md` / `PREFILL-LEVERS-RANKED.md` show bw24's FP4 kernel is **bank-conflict / smem-feed-bound** (ncu: 13.3% tensor-pipe; the int8 sibling went 5.3%→38% tensor-pipe from the `%8==4` K-stride pad). That pad is ~1 line, bit-identical, trivially revertible, NO CUTLASS link/repack/cudart-vs-driver risk. **If the pad takes the in-house FP4 kernel to ~40% of peak, CUTLASS is not needed for FP4 at all** (same ~206–232 TFLOP territory, zero integration cost). This plan's central honest caveat: try the pad, re-measure, and adopt CUTLASS only if the in-house kernel demonstrably cannot reach ~40%.
2. **The int8 cluster (39.9% — the LARGER GEMM mass).** Same `%8==4` smem-pad applied to q4_K/q5_K/q8_0 (the documented next lever; `PREFILL-GEMM-REBUILD.md:121` projects int8-pad alone → ~2880–3180). CUTLASS-fp4 cannot help here; this is in-house pad work and addresses more of prefill than FP4 does. (CUTLASS does have an int8/W4A16 path, but the int8 pad is cheaper and already designed.)
3. **The SSM/attn cluster (27%).** The in-flight FA-prefill rebuild (the parallel `flash_attn.cu` build) is fa_prefill (10.4%); ssm_conv1d_silu (10.9%) + gdn_scan (6%) need their own levers (`SSM-PREFILL-PLAN.md`). Mandatory to reach 6139.

**Composite honest projection (Amdahl, not summed best-cases):** FP4-pad-or-CUTLASS (~2355–2515) → + int8 pad (39.9% slice, ~2x → push toward ~2880–3180) → + FA-prefill rebuild + SSM levers (27% slice) → only THEN does ~5500–6139 come into view. The GEMM levers (FP4 + int8) together get prefill to ~3000–3400; the SSM/attn cluster is the rest of the gap. **CUTLASS-FP4 is one of four levers, the smallest-slice one, and should be sequenced AFTER the cheap in-house FP4 pad is proven insufficient.**

---

## 4. THE bw-EDGE ON TOP OF CUTLASS

CUTLASS gives **parity-via-copy** on the FP4 GEMM — exactly what vLLM and SGLang already use, so adopting it is matching them, not beating them. The edge is everything CUTLASS does NOT do:

1. **Selective-expert MoE** (`MOE-SLRU-PLAN.md`, `ST-MOE-PLAN.md`, `moe_router.cu`). CUTLASS is a dense GEMM — it has no notion of routing/expert-skipping. bw24's SLRU expert cache + selective compute is a structural win on MoE models that a dense FP4 GEMM cannot give. CUTLASS becomes the per-expert dense kernel UNDER the selective router.
2. **KV-quant (q8_0-K / q5_1-V, 4.4x KV shrink, fused inline-dequant)** — already shipped (commit 9ebf958). Orthogonal to the GEMM; CUTLASS-FP4 does the prefill matmuls, the quantized KV cache cuts decode bandwidth and enables longer context / more concurrent agents. CUTLASS has nothing here.
3. **Decode MMVQ / dp4a GEMV** — the B=2–4 concurrent-agent regime where CUTLASS loses 30–100x (§1). bw24's bandwidth-bound warp-per-row MMVQ (`lib.rs` `qmatvec_mmvq`, reads each weight once) is the right tool; this is bw24's decode edge and CUTLASS explicitly does NOT compete here.
4. **Batched MTP partial-accept replay** (commit abef37d, T=n_acc+1, single weight read) — speculative-decode plumbing on top of the GEMV path; CUTLASS-irrelevant.
5. **The in-house FP4/int8 smem-pad kernel itself.** If the `%8==4` pad reaches ~40% of peak in-house (§3.1), bw24 keeps a CUTLASS-free FP4 GEMM with NO vendored-header / link / repack burden — a maintenance edge. CUTLASS is the fallback, and even then only for m>=128.

**Framing:** adopt CUTLASS to *erase the FP4-GEMM deficit* (parity with vLLM/SGLang on the one kernel they're faster on), then the differentiation is the selective-MoE + KV-quant + decode-GEMV + spec-decode stack that the throughput-tile collective structurally cannot provide.

---

## 5. PHASED BUILD ORDER (each gated)

**Phase 0 — in-house FP4 pad probe (do FIRST; may obviate CUTLASS for FP4).** Apply the `%8==4` K-stride smem pad to `qmatvec_gemm_nvfp4_fp4` (`PREFILL-LEVERS-RANKED.md` RANK 1 + the int8-proven fix). Run the named ncu probe; gate on measured tensor-pipe %. **GATE: if in-house FP4 reaches ~40% of peak (~206 TFLOP territory), STOP — CUTLASS-FP4 is unnecessary; skip to Phase 5 (int8 pad + SSM).** If it stalls <~25%, proceed to Phase 1. (This phase touches `qmatvec_gemm.cu` — coordinate with the parallel FA build; it is a different TU so no conflict, but confirm.)

**Phase 1 — CUTLASS build + launch smoke test.** Vendor pinned CUTLASS 4.2 headers; write `cutlass_fp4_sm120.cu` (one config, `OutT=float`); `build.rs` static-lib + `--whole-archive`; `cutlass_ffi.rs`. **GATE: a real GEMM launches via cudarc, `cudaGetLastError`==success, host+device round-trips** (the primary-context smoke test, §2.3). Already de-risked by `/tmp/cutlass_probe` (compiles, links, host-runs) — this phase confirms the device launch under cudarc's driver context.

**Phase 2 — weight repack + bit-exact gate.** `CutlassWeight` in `model.rs` (de-interleave + `bw24_cutlass_repack_sfb`); `quantize_fp4_act_cutlass`; the `kernel_check` arm. **GATE: rel-error + argmax gate PASS vs decode-of-same-weight** (the #1 silent-corruption check). Output f32 for this gate.

**Phase 3 — dispatch seam + end-to-end prefill measure.** Wire the `m>=128 && BW24_FP4_CUTLASS` branch in `try_fp4_gemm`. **GATE: end-to-end pp512 measured (NOT GEMM-in-isolation) shows net win after the per-token SFA quant cost; argmax gate still passes.** Expected ~2355–2515 (§3). If the SFA quant eats the win, fix the quant kernel before proceeding.

**Phase 4 — production dtype + tuning.** Switch `OutT=bf16` (re-run argmax gate); tune the m>=128 threshold (try 64) against real shapes; confirm lm_head routes to CUTLASS. **GATE: gate passes at bf16; pp512 ≥ Phase 3.**

**Phase 5 — the levers that actually beat llama (parallel / after).** int8 cluster `%8==4` pad (39.9% slice, ~2880–3180); SSM/attn cluster (FA-prefill rebuild already in flight + ssm_conv1d_silu + gdn_scan). **GATE: composite pp512 toward 5500–6139.** This is where the llama-beating happens; CUTLASS-FP4 (Phases 1–4) is the +13–20% down-payment, not the finish.

---

## Sources
- bw24 dispatch/seam: `crates/bw24-engine/src/lib.rs:780-895` (`matmul_pre`, `try_fp4_gemm`, `qmatvec_mmvq`); load arm `src/model.rs:40-70`; build `crates/bw24-engine/build.rs`; FP4 kernel + GGUF layout `cu/qmatvec_gemm.cu:585-635,862-906,968`.
- Prefill profile + in-house lever + CUTLASS rejection-on-cost: `research/basics/PREFILL-LEVERS-RANKED.md`, `research/basics/PREFILL-GEMM-REBUILD.md:0-36,115,121`.
- CUTLASS path: flashinfer `fp4_gemm_cutlass_template_sm120.h`, `fp4_gemm_template_sm120.h:115-118,134-136,239-240,510-511`, wrapper `fp4_gemm_cutlass_sm120.cu:43`; SF atom `cutlass/detail/sm100_blockscaled_layout.hpp:51-58,90-112`; MMA atom `cute/atom/mma_traits_sm120.hpp:119,171,220-255`, `cute/arch/mma_sm120.hpp:55-70`; collective `cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp`; dtype `float_subbyte.h:79,510-511`.
- On-box verification: standalone probe `/tmp/cutlass_probe/probe.cu` + `libbw24_cutlass.a` + `linktest.cpp` (compiles sm_120a, OMMA+TMA SASS, links with `--whole-archive`, host-runs; 28.7s/5.2GB cold one-config build); `cutlass::bfloat16_t` + `--whole-archive` corrections; cancellation reproduction `/tmp/check_cancel.py`.
- cudarc 0.19.8: `cu_stream()` core.rs:732, `DevicePtr::device_ptr` core.rs:1163, `CUstream`/`CUdeviceptr` driver/sys:419-420.
- bw-edge shipped: KV-quant commit 9ebf958, MTP replay abef37d, safetensors 41f0bc6; MoE `research/basics/MOE-SLRU-PLAN.md`, `ST-MOE-PLAN.md`.
