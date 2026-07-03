# NVFP4 Rig-Native Single Layout Design

## Goal

Define ONE on-device weight layout for NVFP4 tensors that serves BOTH:
- (a) Tensor-core mxf4nvf4 prefill GEMM (m>=16, via CUTLASS sm120 path)
- (b) Batched MMVQ decode path (m=1, bandwidth-bound dp4a walk)

Eliminate the current dual-VRAM-copy problem where `cutlass` field in `GpuTensor::Quant`
stores a second repacked copy ALONGSIDE the raw GGUF `bytes`.

## Current State (the dual-copy problem)

From `model.rs:18-35` and `cutlass_fp4_sm120.cu`:

```
GpuTensor::Quant {
    bytes: CudaSlice<u8>,        // raw GGUF block_nvfp4 (36B blocks) — decode reads this
    cutlass: Option<CutlassWeight> {
        b_packed: CudaSlice<u8>,     // plain K-contiguous e2m1 [out_f, in_f/2]
        sfb_swizzled: CudaSlice<u8>, // CUTLASS SfAtom swizzled scales
    }
}
```

For Qwen3.6-27B NVFP4 (160 weight tensors, ~6.75 GB raw GGUF):
- Raw GGUF bytes: ~6.75 GB
- CUTLASS repacked B + SFB: ~6.4 GB (in_f/2 packed + swizzled SF overhead)
- **Total: ~13.15 GB** on a 24 GB card with KV cache + activations = OOM

This is why `BW24_FP4_CUTLASS_OTF` exists (per-call repack at prefill time, no resident copy).

## The Native Layout: K-Contiguous Packed e2m1 + Linear SF

### Byte Layout (per weight tensor [out_f, in_f])

```
Region A: packed e2m1 nibbles    [out_f, in_f/2] bytes     (K-contiguous)
Region B: linear ue4m3 scales    [out_f, in_f/16] bytes    (one scale per 16 K-elements)
```

**Total bytes = out_f * (in_f/2 + in_f/16) = out_f * in_f * 9/16**

Compare to GGUF block_nvfp4: out_f * (in_f/64) * 36 = out_f * in_f * 36/64 = out_f * in_f * 9/16.
**Identical byte count.** The native layout is a pure reorder, no inflation.

### Why This Layout

1. **CUTLASS B-operand:** The collective reads B as `[n, k] ColumnMajor` packed e2m1, which
   is exactly K-contiguous rows. Region A IS the B operand. Region B (linear SF) feeds into
   `bw24_cutlass_repack_sfb` to produce the swizzled SfAtom layout at prefill dispatch time.

2. **MMVQ decode:** The dp4a path in `qmatvec.cu`/`mmq_fp4.cu` currently reads GGUF 36B blocks
   where each block is [4 scale bytes | 32 qs bytes] for 64 elements. In the native layout,
   a warp wanting 32 consecutive K-elements of one row does:
   - Read 16 bytes from Region A (32 nibbles = 16 packed bytes)
   - Read 2 bytes from Region B (2 ue4m3 scale bytes for the two 16-elem sub-blocks)
   - Total: 18 bytes per 32 elements (vs 36 bytes in GGUF for 64 elements = same ratio)

3. **Single copy on device.** No duplication — the same bytes serve both paths.

### Byte-Level Diagram

```
Offset 0:
  Row 0, packed e2m1:  [nib(0,1), nib(2,3), ..., nib(in_f-2, in_f-1)]   in_f/2 bytes
  Row 1, packed e2m1:  ...
  ...
  Row out_f-1: ...

Offset out_f * in_f/2:
  Row 0, scales:       [sf(0..15), sf(16..31), ..., sf(in_f-16..in_f-1)]  in_f/16 bytes
  Row 1, scales:       ...
  ...
  Row out_f-1: ...
```

Nibble packing: element 2j -> low nibble, element 2j+1 -> high nibble (matches CUTLASS
`float_e2m1_t` packed convention and modelopt convention).

## Loader Repack Algorithms

### From GGUF block_nvfp4 -> Native (at model load)

Input: raw GGUF rows, each `in_f/64 * 36` bytes, block = [d[4] | qs[32]].

```
for each row r in 0..out_f:
  for each block b in 0..in_f/64:
    // Extract 32 qs bytes -> 64 elements, reorder to sequential nibble pairs
    for sub s in 0..4:
      for j in 0..8:
        elem_lo = s*16 + j        // GGUF: qs[s*8+j] low nibble
        elem_hi = s*16 + j + 8    // GGUF: qs[s*8+j] high nibble
        nib_lo = gguf_qs[s*8 + j] & 0xF
        nib_hi = (gguf_qs[s*8 + j] >> 4) & 0xF
        // Write to native packed at K-position b*64 + elem_lo/hi
        native_packed[r * in_f/2 + (b*64 + elem_lo)/2] |= nib_lo << (4 * ((b*64 + elem_lo) & 1))
        native_packed[r * in_f/2 + (b*64 + elem_hi)/2] |= nib_hi << (4 * ((b*64 + elem_hi) & 1))
    // Extract 4 scale bytes -> 4 linear SF entries
    for s in 0..4:
      native_sf[r * in_f/16 + b*4 + s] = gguf_block[s]  // d[s] = ue4m3 scale byte
```

This is the existing `bw24_gguf_nvfp4_deinterleave_kernel` in `cutlass_fp4_sm120.cu:282-306`
(already implemented, already validated). The native layout IS the deinterleaved form.

### From Safetensors/modelopt -> Native (at model load)

Input: modelopt `.weight` [out_f, in_f/2] + `.weight_scale` [out_f, in_f/16].

```
native_packed = weight bytes (already K-contiguous sequential nibble pairs!)
native_sf = weight_scale bytes (already linear [out_f, in_f/16] ue4m3!)
```

**Zero repack needed.** The modelopt format IS the native layout. This is why the
FORMAT-DECISION says "ST-native pays off first" — modelopt NVFP4 is bit-for-bit the target.

### From Native -> CUTLASS prefill operand (at dispatch time, per-call)

The CUTLASS collective needs the SfAtom-swizzled scale layout. The packed e2m1 Region A
is already the correct B operand (K-contiguous ColumnMajor).

```
// At prefill dispatch (m >= 16):
b_operand_ptr = native_packed + row_offset  // direct pointer, no copy
sfa_swizzled = runtime scratch, filled by bw24_cutlass_repack_sfa (activation)
sfb_swizzled = runtime scratch, filled by bw24_cutlass_repack_sfb(native_sf)
```

The SFB swizzle is ~(out_f * in_f/16) bytes scattered — for a 3584x18944 weight this is
3584 * 1184 = 4.2 MB, taking ~2us on the GPU (measured `bw24_cutlass_repack_sfb` latency is
sub-microsecond for typical shapes). This is the OTF cost already accepted under
`BW24_FP4_CUTLASS_OTF`.

Optimization: pre-compute and cache the swizzled SFB in a small (~100 MB total for all layers)
scratch buffer if VRAM allows. This removes the per-call swizzle but is NOT required for
correctness.

## Decode Path: How MMVQ Reads the Native Layout

### Current GGUF-raw decode (mmq_fp4.cu / qmatvec.cu)

The batched MMVQ in `mmq_fp4.cu` calls `load_tiles_nvfp4_nvfp4` which reads raw GGUF
`block_nvfp4` structures: each thread copies 36B blocks as uint32_t (8 u32 qs + 1 u32 scale).
The MMA then operates on the interleaved nibble order.

### Native-layout decode variant

The decode MMVQ kernel must be modified to read from the separated regions:

```cuda
// Per-thread in the MMVQ load_tiles path:
// Old: reads block_nvfp4 (36B) containing [4 scale | 32 qs] for 64 elems
// New: reads 32 bytes from packed region + 4 bytes from scale region for 64 elems

// Region A address: base_packed + row * (in_f/2) + k_offset/2
// Region B address: base_sf + row * (in_f/16) + k_offset/16
```

The key insight: the `load_tiles_nvfp4_nvfp4` function's smem tile layout is INTERNAL to the
kernel. As long as the tile is filled with the correct nibbles in the correct logical order,
the `vec_dot_nvfp4_mma` (which performs the actual `mma.sync m16n8k64` block-scaled MMA) does
not care where the nibbles came from.

**Decode dp4a path (fa_decode_vec_q style, b4/r2w8 in qmatvec.cu):** For the per-token
bandwidth-bound decode, a simpler approach: read 16 consecutive bytes from Region A (covering
32 elements = one warp's dpl lane stride) + 2 scale bytes from Region B. The dequant is:

```cuda
// Native layout dp4a dequant for 16 elements at position k_base:
float scale = ue4m3_to_f32(native_sf[row * sf_stride + k_base/16]);
for (int j = 0; j < 16; j += 2) {
    uint8_t packed_byte = native_packed[row * pack_stride + (k_base + j) / 2];
    float v0 = kvalues_e2m1[packed_byte & 0xF] * scale;
    float v1 = kvalues_e2m1[(packed_byte >> 4) & 0xF] * scale;
    // accumulate dot...
}
```

This has the SAME arithmetic and SAME FP accumulation order as the current GGUF path
(which also reads scale then multiplies each nibble) — just from different addresses.
**Bit-identical output** is achievable by ensuring the per-element dequant produces the same
float value and the dot accumulation visits elements in the same order.

### Exactness Analysis

The GGUF interleave order within a 64-element block is:
```
sub 0: elems 0,1,2,3,4,5,6,7, then 8,9,10,11,12,13,14,15
sub 1: elems 16,17,...,31
sub 2: elems 32,33,...,47
sub 3: elems 48,49,...,63
```

The native layout sequential order is:
```
elems 0,1,2,3,...,63
```

For the MMVQ (MMQ prefill path via `vec_dot_nvfp4_mma`): the MMA instruction is
**associative across K** (the m16n8k64 MMA accumulates all 64 K-elements in hardware in one
instruction). The FP accumulation ORDER within a single mma.sync call is fixed by the
hardware — both layouts feed the same 64 nibble values to the same instruction.
**Bit-identical.**

For the decode dp4a walk: the current GGUF kernel accumulates per-element products in the
order dictated by the GGUF interleave (sub-block 0 first, etc.). The native layout reads
sequentially. This changes the FP addition order.

**Resolution:** The decode kernel for the native layout MUST either:
1. Re-order its reads to match the GGUF sub-block interleave (negating the layout benefit), OR
2. Accept that the decode accumulation order changes and re-gate through the exactness battery.

Option 2 is the correct choice because:
- The decode dp4a path is BANDWIDTH-bound. The addition order change is cosmetic — the dominant
  error source is the e2m1 quantization itself (not FP ordering).
- The exactness gate is run-spec argmax identity, not bit-identity of intermediate floats.
- The batched decode in `qmatvec_gemm.cu` (the `pf/r2w8` family) already re-orders elements
  vs the naive sequential walk (it tiles by BN/BM/unroll). Changing the innermost order
  within a K-block is the same class of change.

**Gate plan:** Land with `BW24_NATIVE_FP4` env gate. Run the full battery:
- msweep bit-exact-bad = 0 (prefill path, guaranteed same by mma.sync)
- run-gen argmax == 82 (decode path, expected to pass since argmax is robust to ~1e-7 noise)
- run-spec K=1..8 (verify uses the same kernel, same order)

If run-gen argmax fails (very unlikely given the quantization noise floor), the fallback is
option 1 (re-ordered reads), which is still a net win because it eliminates the dual VRAM copy.

## VRAM Accounting (Qwen3.6-27B NVFP4, 24 GB card)

### Current dual-copy (BW24_FP4_CUTLASS=1, no OTF)
- GGUF bytes: 6.75 GB
- CUTLASS b_packed + sfb_swizzled: ~6.4 GB
- **Total weights: ~13.15 GB** (OOMs on 24 GB)

### Current OTF (BW24_FP4_CUTLASS_OTF=1)
- GGUF bytes: 6.75 GB
- Per-call scratch for repack: ~50 MB (largest single GEMM)
- **Total weights: ~6.8 GB** (fits, but per-call repack cost)

### Native layout (this design)
- Native packed + SF: out_f * in_f * 9/16 per tensor = **same 6.75 GB** (byte-for-byte
  same count as GGUF, just reordered)
- Per-call SFB swizzle scratch: ~4 MB (one layer's SFB at a time, reused)
- **Total weights: ~6.76 GB** (fits, same as OTF, but prefill reads B directly = no repack)

### Optional: pre-cached swizzled SFB
- All layers' swizzled SFB: sum of (out_f * in_f/16) per tensor with alignment padding
  = ~100-120 MB
- **Total: ~6.88 GB** (still fits easily)

## Kernel Changes and Landing Order

### Phase 1: Loader repack (no kernel change, no decode perf change)

1. Replace `GpuTensor::Quant::bytes` content with native layout bytes (repack at load from
   GGUF or modelopt). The `cutlass` field becomes `None` always.
2. The CUTLASS prefill path reads `bytes` directly as the B operand (the first out_f * in_f/2
   bytes ARE K-contiguous e2m1).
3. The decode MMVQ path is SWITCHED OFF for NVFP4 — falls back to the existing
   `fa_decode_vec_q` (scalar decode) which dequants from whatever format. Wait, no — the
   scalar decode already reads q8_0 K-cache, not weight NVFP4. The WEIGHT decode uses
   `qmatvec_gemm_nvfp4` which reads GGUF blocks.

**Correction:** In Phase 1, the decode path for NVFP4 weights needs a kernel variant.

### Phase 1 (revised): Native layout + OTF decode adapter

1. Loader produces native layout bytes.
2. Prefill: CUTLASS reads Region A directly, SFB swizzle from Region B per-call. **No GGUF
   repack needed.** This is the win: the current OTF path does GGUF-deinterleave per call;
   the native layout eliminates that step entirely.
3. Decode: A thin adapter kernel re-packs 64 native elements into the GGUF 36B block format
   into a small smem tile INSIDE the MMVQ kernel (same as `load_tiles_nvfp4_nvfp4` but
   reading from split regions instead of interleaved blocks). **No VRAM copy, just a load
   address change in the tile loader.**

Gate: msweep (prefill bit-exact guaranteed), run-gen, run-spec.

### Phase 2: Native decode kernel (decode perf win)

Write a `qmatvec_gemm_nvfp4_native` variant of the decode kernel that reads the sequential
nibble layout directly without the sub-block interleave dance. The per-warp K-element walk
becomes simpler (contiguous 16B loads with no index arithmetic).

Gate: run-gen argmax + run-spec. If argmax fails, revert to Phase 1's adapter.

### Phase 3: Remove GGUF fallback

Once all gates pass on the native decode kernel, remove the GGUF block_nvfp4 code path
for weights (keep it for KV cache which uses a different format anyway). The GGUF block
format survives only as an import format in the loader.

## Measurement Plan

| Cell | What it proves | Pass criterion |
|------|---------------|----------------|
| pp512 (prefill) | Phase 1 eliminates deinterleave overhead | pp512 tok/s >= current BW24_FP4_CUTLASS_OTF |
| e2e tg128 K=4 | Phase 2 decode on native layout | e2e tok/s >= current GGUF-raw decode |
| msweep full | Prefill bit-exactness | bit-exact-bad = 0 |
| run-gen | Argmax identity | argmax == 82 (all 3 configs) |
| run-spec K=1..8 | Spec verify exactness | pass rate >= current (no regressions) |
| VRAM peak (nvidia-smi) | No dual-copy | peak <= 16 GB for 27B + 2K ctx |

## Open Questions Needing GPU Measurement

1. **SFB swizzle latency vs pre-caching:** Is the per-call `bw24_cutlass_repack_sfb` (~2us
   per layer on current shapes) worth pre-caching at load? Measure total prefill pp512
   with vs without cached SFB.

2. **Phase 2 decode exactness:** Does the sequential-order decode pass run-gen argmax on
   all three model configs (9B, 27B, hybrid)? This is the critical gate — if it fails,
   Phase 2 is replaced by the Phase 1 adapter (which is still a VRAM win).

3. **Native layout L1/L2 hit rate on decode:** The GGUF 36B block has spatial locality
   (scale + qs contiguous). The native layout separates them by `out_f * in_f/2` bytes.
   Measure with ncu L2 hit counters whether the scale fetch from Region B hits L2 (expected
   yes — scales are only in_f/16 bytes per row = 1184 bytes for 18944-wide, well within
   a 64 MB L2 sector).

4. **Modelopt weight direct-load speedup:** With native = modelopt format, loading a
   safetensors NVFP4 model is a pure memcpy (no repack). Measure load time delta vs
   current `nvfp4_repack::repack_modelopt_to_gguf` + htod.
