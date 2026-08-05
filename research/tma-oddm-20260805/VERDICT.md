# TMA odd-M prefill lane — FlashInfer #4210 pattern vs memra's prefill activation-quant path

Date: 2026-08-05. Tree: train HEAD `1e4b5209` (local /home/avifenesh/projects/bw24-unified).
Pod tree `/root/bw24-h7` is `7ac05f54`; the delta `7ac05f54..1e4b5209` was audited before any
work: it touches ONLY `hybrid_forward.rs` (chunk-invariance seams), `spec.rs`, server
`main.rs`/`worker.rs`, `concat_prime_probe.rs`, Cargo files and tools —
**zero files in the prefill quant path** (`git diff --stat 7ac05f54 1e4b5209 -- crates/memra-engine/cu/
crates/memra-engine/src/lib.rs crates/memra-engine/src/mmq_ffi.rs` is EMPTY). No rsync-forward was
needed for this verdict; the pod build at 7ac05f54 is byte-current for every file quoted below.

## THE PATTERN under evaluation

FlashInfer #4210 (research/upstream-sweeps.md 2026-08-05): their NVFP4 prefill quant path staged
activations through a PADDED COPY to make M a multiple of 128 for the TMA tensor map; the fix
deleted the staging copy — describe PHYSICAL M rows in the tensor map and let G2S TMA
out-of-bounds zero-fill cover the padded scale tiles. 212us -> 59us (3.6x) at non-128-aligned M.

## VERDICT: N/A — memra has NO M-padding/staging copy on any prefill activation-quant arm

Every prefill quant arm was mapped. Each one either (a) launches the quantizer with the PHYSICAL
token count and zero-fills tails in-register, or (b) already describes physical M in its TMA
tensor maps and rides OOB zero-fill — i.e. memra is already on the "after" side of #4210,
on every arm. There is no staging copy to delete and no staging fraction to measure.

### Arm-by-arm receipts (all paths that quantize prefill activations)

**1. MMQ family — THE default prefill GEMM class** (NVFP4 W4A8 default-on; Q8_0, Q4_0, Q4_K/Q5_K,
FP8-blk, f8f4, IQ-experts/MoE same pattern):

Quantizer grid is one CTA-column per PHYSICAL token — `n_tokens`, not a padded M
(`cu/mmq_nvfp4_w4a8.cu:920-927`):

```c
const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);   // K padded LOGICALLY, not copied
const dim3 num_blocks((unsigned) n_tokens, (unsigned) block_num_y, 1);
quantize_mmq_q8_1_d4_kernel<<<num_blocks, block_size, 0, st>>>(
    act_f32, act_scratch, ne10, /*s01*/ in_f, ne10_padded, n_tokens);
```

The K-tail "padding" is an in-register zero-fill inside the kernel — no padded buffer is ever
written or copied (`cu/mmq_nvfp4_w4a8.cu:829`, identical line in `mmq_q8_0.cu:418`,
`mmq_q4_0.cu:700`, `mmq_q45k.cu:515`, `mmq_fp8_blk.cu:452`, `mmq_nvfp4_f8f4.cu:63`,
`moe_f16_grouped.cu` fused epilogue):

```c
const float4 xi = i0 < ne00 ? x4[(i01 * s01 + i00) / 4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
```

Ragged M on the GEMM side is a scratch OVER-ALLOCATION (bytes only — never initialized, copied,
or read as data) plus a write-back guard, not a padded copy (`cu/mmq_nvfp4_w4a8.cu:860-868`):

```c
// +MMQ_X blocks: the mul_mat_q y-tile loader always reads a FULL mmq_x-column tile; for the
// final k-block with n_tokens % MMQ_X != 0 that read runs past the last real column. Padding
// the scratch keeps the overread mapped (values are garbage; write-back drops j > j_max).
return (size_t) (nblocks + MMQ_X) * sizeof(block_q8_1_mmq);
```

and the store guard (`cu/mmq_q8_0.cu:299-300`, same in every mmq_write_back):

```c
if (j > j_max) { continue; }
if (need_check && i > i_max) { continue; }
```

**2. FP8 cuBLASLt prefill** (`MEMRA_PP_FP8` / QT_F8_E4M3, `cu/fp8_prefill.cu:102-115`): the whole
quant chain (amax -> scale -> e4m3 convert) is grid-stride over EXACTLY `nelem = m*k`; cuBLASLt
layouts take physical m (`cublasLtMatrixLayoutCreate(&p.lb, CUDA_R_8F_E4M3, k, m, k)`). No pad.

**3. F16-mirror prefill** (`MEMRA_PP_F16`, `cu/f16_prefill.cu:36-50`): `memra_f16_cvt_kernel`
converts n = m*k with an explicit scalar tail loop (`// tail (n % 4) by the first threads`).
Physical m in the Lt descriptors. No pad.

**4. CUTLASS NVFP4 W4A4 — the direct #4210 analog** (compile-gated `cfg(memra_cutlass)`, opt-in
`MEMRA_FP4=1 MEMRA_FP4_CUTLASS=1`, default OFF; `cu/cutlass_fp4_sm120.cu` header: "the exact
collective shape flashinfer's DeviceGemmFp4GemmSm120 (W4A4_NVFP4_NVFP4) builds, copied 1:1").
This is the SAME kernel family FlashInfer fixed — and memra's integration was never on the bug
side. The problem shape carries PHYSICAL m (`cutlass_fp4_sm120.cu:95`):

```c
args.problem_shape = cute::make_shape(m, n, k, 1);
```

The activation quantizer runs one thread per physical (row, 16-block) with an early-out
(`memra_nvfp4_quant_kernel:196-198`: `if (blk >= total) return;` over `total = rows * ksf`), and
the SFA scatter writes ONLY the physical entries into the swizzled layout
(`memra_sf_scatter_kernel:168-170`). The swizzled SFA buffer's tile-padding region (cosize covers
ceil(m/128) atoms) is covered by CUTLASS's own TMA handling of the physical problem_shape: A-rows
>= m are OOB in the A tensor map -> G2S zero-fill, epilogue predication drops stores >= m — the
padded scale bytes are multiplied against TMA-zero-filled A values. Exactly the #4210 "after"
structure. No host- or device-side padded staging copy exists anywhere in
`cutlass_ffi.rs::cutlass_fp4_gemm` (quantize -> scatter -> GEMM, all physical-m).

**5. FA3 TMA prefill** (sm_90a lane, attention not act-quant, listed for completeness —
`cu/fa3_prefill.cu:281-309`): tensor maps encode physical T (`gdq[3] = {D, H, T}`), grid
`(T+127)/128`, tail tiles ride `cp.async.bulk.tensor` OOB zero-fill. Also already the "after"
pattern. This is the ONLY memra TMA use; the sm_120a prefill GEMM/MMQ arms are cp.async +
ldmatrix with bounds-checked tails (see 1), where OOB zero-fill is done in-register — the
"equivalent fix" the mission brief anticipated is already the shipped structure.

**6. Sole M-padded prefill in the tree — NOT a staging bug**: `src/prime_graph.rs` pads T to a
graph bucket (`prime_graph_run:128-131` memsets the pad tail of `x_in`) because CUDA-graph replay
requires baked shapes; pads are masked (gdn_pad_mask identity steps, causal attention,
device-indexed last-row gather). Physical-M description is structurally impossible for a captured
graph; #4210 does not apply. It is also NOT wired into serving — its only callers are
`bin/prime_graph_gate.rs` / `prime_graph_smoke.rs`.

## Consequence for the prime-chunk/prefill wall

The dogfood-named prefill gap (memra 1.2k vs llama 2.1k tok/s at 4k) does NOT contain a #4210-class
staging tax: odd-M prefill calls (arbitrary prompt lengths, odd last chunks of chunked prefill)
enter every quant arm at physical M with per-element-bounded tail work — the marginal odd-M cost is
one `need_check` tail tile per GEMM (guarded loads/stores on <=127 rows), not an O(M*K) staged copy.
The wall's measured causes live elsewhere (prefill-GEMM rebuild plan: GEMM coverage/Amdahl,
prime-chunk boundary arithmetic — see research/chunk-invariance-20260805/VERDICT.md).

## Status of steps 2-3

Per the mission contract ("if memra never stages, the pattern is N/A — receipt that verdict with
the quoted code and STOP"), no measurement or prototype was run: there is no staging copy to
profile and no kernel change to gate. Pod left warm, untouched by this lane except this receipt +
a STATE note; no GPU run was needed (mapping is pure code-reading, arms quoted at the pod tree's
exact bytes — delta-audited above).
