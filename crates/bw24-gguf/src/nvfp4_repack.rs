//! Repack a modelopt (compressed-tensors) NVFP4 weight INTO bw24's internal GGUF NVFP4 byte
//! layout that kernel2 (`qmatvec_gemm_nvfp4`) + the decode dp4a path already consume — so the HF
//! load path emits bytes IDENTICAL-IN-MEANING to the GGUF NVFP4 path, with NO kernel change.
//!
//! ── modelopt (HF compressed-tensors) NVFP4 per quantized Linear ──────────────────────────────
//!   <name>.weight         U8       [out, in/2]   packed FP4 e2m1, two 4-bit codes / byte,
//!                                                 element 2i -> low nibble, 2i+1 -> high nibble.
//!   <name>.weight_scale   F8_E4M3  [out, in/16]  per-16 UE4M3 block scale (one byte / 16 elems).
//!   <name>.weight_scale_2 F32 scalar             per-tensor macro-scale (applied POST-matmul).
//!   <name>.input_scale    F32 scalar             W4A8 activation scale (UNUSED on our path).
//!   dequant(elem) = e2m1_code(elem) * ue4m3(weight_scale[elem/16]) * weight_scale_2
//!
//! ── bw24 internal GGUF block_nvfp4 (QK=64, 36 B / 64 elems; ggml-quants.c dequantize_row_nvfp4) ─
//!   d[0..4]   : 4 UE4M3 sub-scale bytes, one per 16-elem sub-block.
//!   qs[0..32] : 32 packed bytes. Sub-block s (0..4) uses qs[s*8 .. s*8+8]; within a sub-block,
//!               low nibbles -> elems 0..7, high nibbles -> elems 8..15:
//!                 out[s*16 + j]     = KVALUES_MXFP4[ qs[s*8+j] & 0x0F ] * d_s   (j=0..8)
//!                 out[s*16 + j + 8] = KVALUES_MXFP4[ qs[s*8+j] >> 4   ] * d_s
//!   KVALUES_MXFP4 is the DOUBLED e2m1 codebook (2x the standard code); `ue4m3_to_f32` returns
//!   (the UE4M3 value) * 0.5. The 2x and the 0.5 CANCEL, so:
//!     bw24_dequant = doubled_code * (raw_ue4m3 * 0.5) = std_code * raw_ue4m3
//!   which is EXACTLY the modelopt per-element value (sans the per-tensor scale_2, applied post).
//!   => the FP4 4-bit CODE and the UE4M3 SCALE BYTE both copy through VERBATIM; only the within-row
//!      NIBBLE ORDER differs (modelopt sequential 2-per-byte vs GGUF sub-block 8-byte interleave).
//!
//! The per-tensor `weight_scale_2` is surfaced separately as the sibling `<stem>.scale` tensor that
//! `GpuTensor::load_from_source` reads as the post-matmul macro-scale (model.rs) — identical to the
//! GGUF NVFP4 path. Repack is per-row, 64-elem-block at a time; in_f must be a multiple of 64.

/// e2m1 (4-bit) -> the GGUF DOUBLED codebook value (ggml-common.h kvalues_mxfp4). Used only for the
/// reference dequant in validation; the repack itself never decodes the 4-bit code (it copies it).
pub const KVALUES_MXFP4: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// UE4M3 byte -> f32 (GGUF convention: returns value*0.5; ggml-impl.h ggml_ue4m3_to_fp32).
/// NaN codes 0 and 0x7F -> 0.0. Identical to `dequant::ue4m3_to_f32` (kept local to avoid coupling).
#[inline]
pub fn ue4m3_to_f32(x: u8) -> f32 {
    if x == 0 || x == 0x7F {
        return 0.0;
    }
    let exp = ((x >> 3) & 0xF) as i32;
    let man = (x & 0x7) as f32;
    let raw = if exp == 0 { man * 2f32.powi(-9) } else { (1.0 + man / 8.0) * 2f32.powi(exp - 7) };
    raw * 0.5
}

/// f32 -> IEEE f16 bits (round-to-nearest-even). Serde-free (no `half` dep) — used by the
/// F8->Q8_0 host re-encode (per-32 block scale is fp16 in GGUF Q8_0).
#[inline]
pub fn f32_to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xFF) as i32;
    let man = b & 0x7F_FFFF;
    if exp == 0xFF { return sign | 0x7C00 | if man != 0 { 0x200 } else { 0 }; } // inf/nan
    let e16 = exp - 127 + 15;
    if e16 >= 0x1F { return sign | 0x7C00; }            // overflow -> inf
    if e16 <= 0 {                                        // subnormal / underflow
        if e16 < -10 { return sign; }
        let man24 = man | 0x80_0000;
        let shift = (14 - e16) as u32;
        let half_man = man24 >> shift;
        let round = (man24 >> (shift - 1)) & 1;
        let sticky = (man24 & ((1 << (shift - 1)) - 1)) != 0;
        let mut m = half_man;
        if round == 1 && (sticky || (m & 1) == 1) { m += 1; }
        return sign | m as u16;
    }
    let mut m = (man >> 13) as u16;
    let round = (man >> 12) & 1;
    let sticky = (man & 0xFFF) != 0;
    let mut e = e16 as u16;
    if round == 1 && (sticky || (m & 1) == 1) {
        m += 1;
        if m == 0x400 { m = 0; e += 1; if e >= 0x1F { return sign | 0x7C00; } }
    }
    sign | (e << 10) | m
}

/// SIGNED FP8 E4M3 byte -> f32 (IEEE-754-style e4m3 with sign bit, bias 7; NaN = 0x7F/0xFF -> 0.0).
/// This is the WEIGHT dtype of the NVIDIA-official 27B linear_attn projections (per-tensor F32
/// weight_scale applied by the caller) — distinct from the UNSIGNED ue4m3 block-scale above.
#[inline]
pub fn fp8_e4m3_to_f32(x: u8) -> f32 {
    let mag = x & 0x7F;
    if mag == 0x7F {
        return 0.0; // NaN code -> 0.0 (modelopt convention)
    }
    let sign = if x & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((mag >> 3) & 0xF) as i32;
    let man = (mag & 0x7) as f32;
    let raw = if exp == 0 { (man / 8.0) * 2f32.powi(-6) } else { (1.0 + man / 8.0) * 2f32.powi(exp - 7) };
    sign * raw
}

/// f32 -> SIGNED FP8 E4M3 byte: nearest on the e4m3 grid, ties to the even code, saturate to
/// +-448 (max normal), +-0 -> 0x00. Inverse of `fp8_e4m3_to_f32` (roundtrip-exact over all
/// non-NaN codes — pinned by the test below). Used by the BW24_PP_FP8 loader to re-encode the
/// V-reordered F8 projections back to checkpoint e4m3 codes (an exact permutation round-trip:
/// inputs are `code*scale/scale`, within an f32 ULP of the grid point, and the e4m3 grid spacing
/// is >= 2^-3 relative — nearest always recovers the original code).
#[inline]
pub fn f32_to_fp8_e4m3(x: f32) -> u8 {
    // Decoded magnitudes of codes 0..=0x7E are strictly increasing (0x7F is the NaN code).
    static TABLE: std::sync::OnceLock<[f32; 127]> = std::sync::OnceLock::new();
    let t = TABLE.get_or_init(|| {
        let mut t = [0f32; 127];
        for (c, v) in t.iter_mut().enumerate() { *v = fp8_e4m3_to_f32(c as u8); }
        t
    });
    if x.is_nan() { return 0; } // NaN never encodes to the NaN code (modelopt decode maps it to 0)
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
    let ax = x.abs();
    if ax == 0.0 { return 0; }
    if ax >= t[126] { return sign | 0x7E; } // saturate to +-448 (covers +-inf too)
    // binary search: largest lo with t[lo] <= ax; then nearest of (lo, lo+1), ties-to-even code.
    let (mut lo, mut hi) = (0usize, 126usize);
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if t[mid] <= ax { lo = mid } else { hi = mid }
    }
    let (dl, dh) = (ax - t[lo], t[hi] - ax);
    let c = if dl < dh { lo } else if dh < dl { hi } else if lo % 2 == 0 { lo } else { hi };
    if c == 0 { 0 } else { sign | c as u8 }
}

/// f32 slice -> GGUF Q8_0 block bytes ({ fp16 d; int8 qs[32] } per 32 elems, 34 B/block).
/// Standard ggml quantize_row_q8_0 math (amax/127 scale, round-to-nearest-even). Used by the
/// F8-E4M3 and large-BF16 host re-encodes (NVIDIA 27B linear_attn + mtp classes). `data.len()`
/// must be a multiple of 32 (all 2D matrices here have in_f % 32 == 0).
/// f32 -> UE4M3 scale byte (unsigned e4m3, GGUF block-scale convention: stored byte decodes via
/// `ue4m3_to_f32` = raw*0.5 — caller passes the RAW target value, i.e. 2x the decoded scale).
/// Round-to-nearest over the representable grid; clamps to [smallest-subnormal, 448].
#[inline]
fn f32_to_ue4m3(x: f32) -> u8 {
    if !(x > 0.0) { return 0; }
    // brute nearest over 127 codes (1..0x7E; 0 and 0x7F are zero/NaN) — encode-time only.
    let mut best = 0u8;
    let mut bd = f32::INFINITY;
    for c in 1u8..0x7F {
        let exp = ((c >> 3) & 0xF) as i32;
        let man = (c & 0x7) as f32;
        let v = if exp == 0 { man * 2f32.powi(-9) } else { (1.0 + man / 8.0) * 2f32.powi(exp - 7) };
        let d = (v - x).abs();
        if d < bd { bd = d; best = c; }
    }
    best
}

/// f32 slice -> bw24 internal GGUF NVFP4 block bytes (header layout above: QK=64, 36 B/block,
/// 4 UE4M3 sub-scales + 32 interleaved code bytes). Per-16 sub-block: raw_scale = amax/6 (largest
/// |e2m1| std value), scale byte = nearest UE4M3 to raw_scale, codes = nearest e2m1 to x/scale
/// (decoded scale). This is a REAL re-quantization (e4m3 -> e2m1 loses mantissa bits) — callers
/// gate it behind accuracy checks; the house 27B daily runs the same 4-bit class on these layers.
/// `data.len()` must be a multiple of 64.
pub fn f32_to_nvfp4(data: &[f32]) -> Vec<u8> {
    assert!(data.len() % 64 == 0, "f32_to_nvfp4: len {} not /64", data.len());
    // std e2m1 magnitudes (KVALUES are doubled; ue4m3_to_f32 halves — they cancel, so match on std)
    const MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let n_blocks = data.len() / 64;
    let mut out = vec![0u8; n_blocks * 36];
    for (b, chunk) in data.chunks_exact(64).enumerate() {
        let blk = &mut out[b * 36..(b + 1) * 36];
        for s in 0..4 {
            let sub = &chunk[s * 16..s * 16 + 16];
            let amax = sub.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
            let sbyte = f32_to_ue4m3(amax / 6.0);
            blk[s] = sbyte;
            let dscale = ue4m3_to_f32(sbyte) * 2.0; // decoded raw scale (ue4m3_to_f32 halves)
            let inv = if dscale > 0.0 { 1.0 / dscale } else { 0.0 };
            for j in 0..8 {
                let mut byte = 0u8;
                for (nib, jj) in [(0usize, j), (1usize, j + 8)] {
                    let x = sub[jj] * inv;
                    let ax = x.abs();
                    // nearest e2m1 magnitude (ties toward the smaller code, matching RNE on grid)
                    let mut ci = 0usize;
                    let mut bd = f32::INFINITY;
                    for (i, &m) in MAG.iter().enumerate() {
                        let d = (ax - m).abs();
                        if d < bd { bd = d; ci = i; }
                    }
                    let code = if ci == 0 { 0u8 } else if x < 0.0 { (ci as u8) | 0x8 } else { ci as u8 };
                    byte |= code << (4 * nib);
                }
                blk[4 + s * 8 + j] = byte;
            }
        }
    }
    out
}

pub fn f32_to_q8_0(data: &[f32]) -> Vec<u8> {
    assert_eq!(data.len() % 32, 0, "Q8_0 encode needs len % 32 == 0, got {}", data.len());
    let mut out = Vec::with_capacity(data.len() / 32 * 34);
    for blk in data.chunks_exact(32) {
        let amax = blk.iter().fold(0f32, |a, &v| a.max(v.abs()));
        let d = amax / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        for &v in blk {
            out.push((v * id).round_ties_even() as i32 as i8 as u8);
        }
    }
    out
}

/// Repack ONE modelopt NVFP4 weight tensor `[out_f, in_f]` into bw24 GGUF block_nvfp4 bytes.
///
/// `weight` = modelopt `.weight`        U8       (out_f * in_f/2 bytes), row-major.
/// `wscale` = modelopt `.weight_scale`  F8_E4M3  (out_f * in_f/16 bytes), row-major.
/// Returns `out_f * (in_f/64) * 36` bytes, laid out as `out_f` contiguous rows of GGUF 36-B blocks
/// (exactly the `row_bytes = total/out_f` the engine computes for an NVFP4 weight).
///
/// Panics if `in_f % 64 != 0` (the NVFP4 K-block constraint) or the input byte counts disagree.
pub fn repack_modelopt_to_gguf(weight: &[u8], wscale: &[u8], out_f: usize, in_f: usize) -> Vec<u8> {
    assert_eq!(in_f % 64, 0, "NVFP4 repack requires in_f % 64 == 0, got in_f={in_f}");
    let in_bytes = in_f / 2; // modelopt packed bytes per row (2 codes/byte)
    let scl_bytes = in_f / 16; // modelopt scale bytes per row (1 UE4M3 / 16 elems)
    assert_eq!(weight.len(), out_f * in_bytes, "weight byte count != out_f*in_f/2");
    assert_eq!(wscale.len(), out_f * scl_bytes, "weight_scale byte count != out_f*in_f/16");

    let nblk = in_f / 64; // 64-elem blocks per row
    let row_bytes = nblk * 36; // GGUF NVFP4 bytes per row
    let mut out = vec![0u8; out_f * row_bytes];

    for o in 0..out_f {
        let w_row = &weight[o * in_bytes..(o + 1) * in_bytes];
        let s_row = &wscale[o * scl_bytes..(o + 1) * scl_bytes];
        let o_row = &mut out[o * row_bytes..(o + 1) * row_bytes];
        for b in 0..nblk {
            let blk = &mut o_row[b * 36..(b + 1) * 36];
            // 4 UE4M3 sub-scale bytes (one per 16 elems): copy VERBATIM. block b spans elems
            // [b*64, b*64+64); its 4 sub-blocks are scale indices [b*4, b*4+4) in the row.
            blk[0..4].copy_from_slice(&s_row[b * 4..b * 4 + 4]);
            // 32 qs bytes: re-pack the nibble order modelopt(sequential) -> GGUF(sub-block interleave).
            // For each of the 4 sub-blocks s (16 elems) the 8 GGUF bytes are qs[s*8 .. s*8+8]; GGUF
            // byte j holds elem (s*16+j) in its low nibble and elem (s*16+j+8) in its high nibble.
            // modelopt byte for elem e = (b*64 + e_local) is at w_row[(b*64+e_local)/2], low if even.
            for s in 0..4 {
                for j in 0..8 {
                    let e_lo = b * 64 + s * 16 + j; // GGUF low-nibble element
                    let e_hi = b * 64 + s * 16 + j + 8; // GGUF high-nibble element
                    blk[4 + s * 8 + j] = (nib(w_row, e_lo - b * 64, b)) | (nib(w_row, e_hi - b * 64, b) << 4);
                }
            }
        }
    }
    out
}

/// Extract the 4-bit e2m1 code for the `e_local`-th element of block `b` from the modelopt row bytes.
/// modelopt packs element `g` (global within the row) at byte g/2, low nibble if g even, high if odd.
#[inline]
fn nib(w_row: &[u8], e_local: usize, b: usize) -> u8 {
    let g = b * 64 + e_local; // global element index within the row
    let byte = w_row[g / 2];
    if g & 1 == 0 { byte & 0x0F } else { byte >> 4 }
}

/// Repack ONE modelopt/Reza NVFP4 weight tensor DIRECTLY into bw24's A6 split-plane resident
/// layout — [quant plane out_f x in_f/64 x 32B][scale plane out_f x in_f/64 x 4B], the layout
/// `repack_nvfp4_split` (bw24-engine model.rs) produces from GGUF blocks — in ONE host pass,
/// never materializing the intermediate GGUF 36B-block buffer (A1 direct import, 2026-07-04).
///
/// Two facts make this a pure fusion, not a new layout:
///   * scale plane == the modelopt `.weight_scale` bytes VERBATIM: the plane is row-major per-16
///     UE4M3 scales in element order — exactly the on-disk modelopt tensor. One memcpy.
///   * quant plane keeps the GGUF within-block sub-block nibble interleave (the A6 rp kernels'
///     walk order; bit-identity law forbids changing it), i.e. the same nibble shuffle
///     `repack_modelopt_to_gguf` does — just written at plane offsets.
///
/// Byte-for-byte equal to `repack_nvfp4_split(&repack_modelopt_to_gguf(w, s, o, i), o)`; pinned by
/// `split_plane_matches_gguf_blocks` below and the composition test in bw24-engine model.rs.
pub fn repack_modelopt_to_split(weight: &[u8], wscale: &[u8], out_f: usize, in_f: usize) -> Vec<u8> {
    assert_eq!(in_f % 64, 0, "NVFP4 repack requires in_f % 64 == 0, got in_f={in_f}");
    let in_bytes = in_f / 2; // modelopt packed bytes per row (2 codes/byte)
    let scl_bytes = in_f / 16; // modelopt scale bytes per row (1 UE4M3 / 16 elems)
    assert_eq!(weight.len(), out_f * in_bytes, "weight byte count != out_f*in_f/2");
    assert_eq!(wscale.len(), out_f * scl_bytes, "weight_scale byte count != out_f*in_f/16");

    let nblk = in_f / 64; // 64-elem blocks per row
    let qplane = out_f * nblk * 32; // scale plane starts here
    let mut out = vec![0u8; out_f * nblk * 36];
    // Scale plane: verbatim copy (see header).
    out[qplane..].copy_from_slice(wscale);
    // Quant plane: GGUF sub-block nibble interleave per 64-elem block, blocks contiguous at 32B.
    for o in 0..out_f {
        let w_row = &weight[o * in_bytes..(o + 1) * in_bytes];
        for b in 0..nblk {
            let q = &mut out[(o * nblk + b) * 32..(o * nblk + b) * 32 + 32];
            for s in 0..4 {
                for j in 0..8 {
                    q[s * 8 + j] = nib(w_row, s * 16 + j, b) | (nib(w_row, s * 16 + j + 8, b) << 4);
                }
            }
        }
    }
    out
}

/// Reference dequant of a modelopt NVFP4 row -> f32 [in_f] (modelopt convention, sans scale_2).
/// `val = std_e2m1_code * raw_ue4m3(scale[elem/16])`. Equals bw24's internal dequant of the same
/// element (the doubled-code/halved-scale conventions cancel) — the CPU validation cross-checks both.
pub fn dequant_modelopt_row(weight_row: &[u8], wscale_row: &[u8], in_f: usize) -> Vec<f32> {
    let mut out = vec![0f32; in_f];
    for e in 0..in_f {
        let byte = weight_row[e / 2];
        let code = if e & 1 == 0 { byte & 0x0F } else { byte >> 4 } as usize;
        // standard e2m1 code = doubled-table value * 0.5; raw_ue4m3 = ue4m3_to_f32 * 2 (undo the 0.5).
        let std_code = KVALUES_MXFP4[code] as f32 * 0.5;
        let raw_ue4m3 = ue4m3_to_f32(wscale_row[e / 16]) * 2.0;
        out[e] = std_code * raw_ue4m3;
    }
    out
}

/// Reference dequant of a bw24 GGUF block_nvfp4 row -> f32 [in_f] (the kernel's convention). Mirrors
/// `dequant::dequant_nvfp4` for one row; used to cross-check the repack produced equivalent bytes.
pub fn dequant_gguf_row(row: &[u8], in_f: usize) -> Vec<f32> {
    const QK: usize = 64;
    let nb = in_f / QK;
    let mut out = vec![0f32; in_f];
    for i in 0..nb {
        let base = i * 36;
        let d_bytes = &row[base..base + 4];
        let qs = &row[base + 4..base + 36];
        for s in 0..4 {
            let d = ue4m3_to_f32(d_bytes[s]);
            let yb = i * QK + s * 16;
            for j in 0..8 {
                let byte = qs[s * 8 + j];
                out[yb + j] = KVALUES_MXFP4[(byte & 0x0F) as usize] as f32 * d;
                out[yb + j + 8] = KVALUES_MXFP4[(byte >> 4) as usize] as f32 * d;
            }
        }
    }
    out
}

// ── NVFP4-preserving structural permutations (keep the weight quantized; no f32 blow-up) ──────────
//
// The qwen35 SSM V-head reorders are pure index permutations that, on a GGUF NVFP4 weight, move whole
// 64-elem blocks: a row reorder moves whole rows (= `row_bytes` byte-blocks); an in-column head reorder
// moves contiguous groups of `head_dim/64` blocks (head_dim=128 == 2 blocks here, block-aligned). So
// they can be applied DIRECTLY to the repacked bytes — no dequant-to-f32 (saves ~16 GB on the 27B).
//
// These mirror `hf_mapping::reorder_rows_v` / `reorder_cols_v` exactly (same `dst_head=j*nk+g <-
// src_head=g*num_v_per_k+j` mapping), but operate on packed rows of `row_bytes` instead of f32.

/// Reorder the V-head OUT-ROWS of a packed NVFP4 weight (rows are independent `row_bytes` blocks).
/// `out_f` rows of `row_bytes`; rows in the band `[row_lo, row_hi)` (== `nv*head_dim` rows) are
/// permuted by the V-head map, rows outside copy through. (qkv V band; z/a/b whole band.)
pub fn reorder_rows_nvfp4(packed: &[u8], out_f: usize, row_bytes: usize, num_v_heads: usize,
                          num_k_heads: usize, head_dim: usize, row_lo: usize, row_hi: usize) -> Vec<u8> {
    assert_eq!(packed.len(), out_f * row_bytes, "packed size != out_f*row_bytes");
    assert_eq!(row_hi - row_lo, num_v_heads * head_dim, "reorder band != nv*head_dim");
    let num_v_per_k = num_v_heads / num_k_heads;
    let mut out = packed.to_vec();
    for j in 0..num_v_per_k {
        for g in 0..num_k_heads {
            let dst_head = j * num_k_heads + g;
            let src_head = g * num_v_per_k + j;
            for d in 0..head_dim {
                let dst_row = row_lo + dst_head * head_dim + d;
                let src_row = row_lo + src_head * head_dim + d;
                out[dst_row * row_bytes..dst_row * row_bytes + row_bytes]
                    .copy_from_slice(&packed[src_row * row_bytes..src_row * row_bytes + row_bytes]);
            }
        }
    }
    out
}

/// Reorder the V-head IN-COLUMNS of a packed NVFP4 weight (out_proj). Columns are in-features; a head
/// is `head_dim` contiguous in-features == `head_dim/64` contiguous 64-elem blocks (must be block-
/// aligned: head_dim % 64 == 0). Permutes the per-head block-groups within EVERY row. `in_f` columns.
pub fn reorder_cols_nvfp4(packed: &[u8], out_f: usize, in_f: usize, num_v_heads: usize,
                          num_k_heads: usize, head_dim: usize) -> Vec<u8> {
    assert_eq!(in_f, num_v_heads * head_dim, "in_f != nv*head_dim");
    assert_eq!(head_dim % 64, 0, "head_dim must be NVFP4-block-aligned (got {head_dim})");
    let blocks_per_head = head_dim / 64;
    let row_bytes = (in_f / 64) * 36;
    assert_eq!(packed.len(), out_f * row_bytes, "packed size != out_f*row_bytes");
    let num_v_per_k = num_v_heads / num_k_heads;
    let head_bytes = blocks_per_head * 36; // bytes for one head's worth of in-feature blocks
    let mut out = packed.to_vec();
    for r in 0..out_f {
        let base = r * row_bytes;
        for j in 0..num_v_per_k {
            for g in 0..num_k_heads {
                let dst_head = j * num_k_heads + g;
                let src_head = g * num_v_per_k + j;
                out[base + dst_head * head_bytes..base + dst_head * head_bytes + head_bytes]
                    .copy_from_slice(
                        &packed[base + src_head * head_bytes..base + src_head * head_bytes + head_bytes],
                    );
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic modelopt row (in_f=128 = 2 blocks), repack, and assert the GGUF dequant
    /// equals the modelopt dequant element-for-element (rel < 1e-6 — they are the SAME arithmetic).
    #[test]
    fn repack_roundtrip_matches_modelopt() {
        let in_f = 128usize;
        let out_f = 3usize;
        let in_bytes = in_f / 2;
        let scl_bytes = in_f / 16;
        // deterministic pseudo-random bytes
        let mut weight = vec![0u8; out_f * in_bytes];
        let mut wscale = vec![0u8; out_f * scl_bytes];
        for (i, b) in weight.iter_mut().enumerate() {
            *b = ((i * 37 + 11) & 0xFF) as u8;
        }
        for (i, b) in wscale.iter_mut().enumerate() {
            // keep scale bytes in the finite UE4M3 range (avoid 0 / 0x7F NaN codes for a strong test)
            *b = (0x20 + ((i * 13 + 3) % 0x50)) as u8;
        }
        let packed = repack_modelopt_to_gguf(&weight, &wscale, out_f, in_f);
        let row_bytes = (in_f / 64) * 36;
        assert_eq!(packed.len(), out_f * row_bytes);
        for o in 0..out_f {
            let mref = dequant_modelopt_row(
                &weight[o * in_bytes..(o + 1) * in_bytes],
                &wscale[o * scl_bytes..(o + 1) * scl_bytes],
                in_f,
            );
            let ggu = dequant_gguf_row(&packed[o * row_bytes..(o + 1) * row_bytes], in_f);
            for e in 0..in_f {
                let a = mref[e];
                let b = ggu[e];
                let denom = a.abs().max(1e-6);
                assert!((a - b).abs() / denom < 1e-6, "row {o} elem {e}: modelopt {a} != gguf {b}");
            }
        }
    }

    #[test]
    fn f32_to_nvfp4_roundtrip_class_error() {
        // Encode a deterministic pseudo-random block set, dequant via the header math, and bound
        // the relative error by the e2m1-per-16 class limit. Also: exact codebook values with a
        // representable scale must roundtrip EXACTLY.
        let mut x = 0x12345678u32;
        let mut rnd = || { x ^= x << 13; x ^= x >> 17; x ^= x << 5; (x as f32 / u32::MAX as f32) * 4.0 - 2.0 };
        let data: Vec<f32> = (0..64 * 8).map(|_| rnd()).collect();
        let enc = f32_to_nvfp4(&data);
        assert_eq!(enc.len(), 8 * 36);
        let mut worst = 0.0f32;
        for b in 0..8 {
            let blk = &enc[b * 36..(b + 1) * 36];
            for s in 0..4 {
                let dscale = ue4m3_to_f32(blk[s]) * 2.0;
                for j in 0..8 {
                    let byte = blk[4 + s * 8 + j];
                    for (nib, jj) in [(byte & 0xF, j), (byte >> 4, j + 8)] {
                        // KVALUES doubled * 0.5 = std code value
                        let v = KVALUES_MXFP4[nib as usize] as f32 * 0.5 * dscale;
                        let orig = data[b * 64 + s * 16 + jj];
                        let sub = &data[b * 64 + s * 16..b * 64 + s * 16 + 16];
                        let amax = sub.iter().fold(0.0f32, |m, &y| m.max(y.abs()));
                        if amax > 0.0 { worst = worst.max((v - orig).abs() / amax); }
                    }
                }
            }
        }
        // e2m1 grid worst half-gap = (6-4)/2 = 1.0, relative to amax 6 => 0.167; plus UE4M3 scale
        // rounding (3 mantissa bits => up to ~6%). 0.20 = the true class bound with slack; still
        // catches sign/nibble/scale bugs instantly (those blow up to O(1)).
        assert!(worst < 0.20, "relative error {worst} exceeds e2m1 class bound");

        // exact codebook roundtrip: values = code * scale with scale=1.0 (representable)
        let vals = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, 0.0];
        let mut block = [0.0f32; 64];
        block[..16].copy_from_slice(&vals);
        let enc2 = f32_to_nvfp4(&block);
        let dscale = ue4m3_to_f32(enc2[0]) * 2.0;
        for (jj, &want) in vals.iter().enumerate() {
            let j = jj % 8;
            let byte = enc2[4 + j];
            let nib = if jj < 8 { byte & 0xF } else { byte >> 4 };
            let got = KVALUES_MXFP4[nib as usize] as f32 * 0.5 * dscale;
            assert!((got - want).abs() < 1e-6, "elem {jj}: got {got} want {want}");
        }
    }

    /// NVFP4-preserving row permutation == f32 row permutation: repack -> permute packed -> dequant
    /// must equal dequant -> f32 permute, for a ZReorderRows-style whole-band out-row reorder.
    #[test]
    fn row_perm_nvfp4_equals_f32() {
        // nv=4, nk=2, head_dim=1 -> 4 out-rows reordered; in_f=64 (1 block).
        let (nv, nk, hd, in_f, out_f) = (4usize, 2usize, 1usize, 64usize, 4usize);
        let in_bytes = in_f / 2;
        let scl_bytes = in_f / 16;
        let mut weight = vec![0u8; out_f * in_bytes];
        let mut wscale = vec![0u8; out_f * scl_bytes];
        for (i, b) in weight.iter_mut().enumerate() { *b = ((i * 41 + 7) & 0xFF) as u8; }
        for (i, b) in wscale.iter_mut().enumerate() { *b = (0x20 + ((i * 11 + 5) % 0x50)) as u8; }
        let row_bytes = (in_f / 64) * 36;
        let packed = repack_modelopt_to_gguf(&weight, &wscale, out_f, in_f);
        // NVFP4 path: permute packed rows, then dequant each row.
        let perm = reorder_rows_nvfp4(&packed, out_f, row_bytes, nv, nk, hd, 0, out_f);
        // f32 path: dequant each row, then apply the same row permutation in f32.
        let f32rows: Vec<Vec<f32>> = (0..out_f).map(|o| dequant_modelopt_row(
            &weight[o * in_bytes..(o + 1) * in_bytes],
            &wscale[o * scl_bytes..(o + 1) * scl_bytes], in_f)).collect();
        let num_v_per_k = nv / nk;
        for j in 0..num_v_per_k {
            for g in 0..nk {
                let dst = j * nk + g;       // head_dim=1 -> head index == row index
                let src = g * num_v_per_k + j;
                let nvfp4_row = dequant_gguf_row(&perm[dst * row_bytes..(dst + 1) * row_bytes], in_f);
                for e in 0..in_f {
                    let a = f32rows[src][e];
                    let b = nvfp4_row[e];
                    let denom = a.abs().max(1e-6);
                    assert!((a - b).abs() / denom < 1e-6, "dst {dst}<-src {src} e {e}: {a} != {b}");
                }
            }
        }
    }

    /// NVFP4-preserving in-column head permutation == f32 column permutation (out_proj). head_dim=128
    /// = 2 blocks; nv=4, nk=2 -> in_f=512 = 8 blocks; 1 out-row.
    #[test]
    fn col_perm_nvfp4_equals_f32() {
        let (nv, nk, hd, out_f) = (4usize, 2usize, 128usize, 1usize);
        let in_f = nv * hd; // 512
        let in_bytes = in_f / 2;
        let scl_bytes = in_f / 16;
        let mut weight = vec![0u8; out_f * in_bytes];
        let mut wscale = vec![0u8; out_f * scl_bytes];
        for (i, b) in weight.iter_mut().enumerate() { *b = ((i * 29 + 13) & 0xFF) as u8; }
        for (i, b) in wscale.iter_mut().enumerate() { *b = (0x21 + ((i * 7 + 1) % 0x50)) as u8; }
        let packed = repack_modelopt_to_gguf(&weight, &wscale, out_f, in_f);
        let perm = reorder_cols_nvfp4(&packed, out_f, in_f, nv, nk, hd);
        // f32 reference: dequant the row, permute the per-head column groups.
        let f32row = dequant_modelopt_row(&weight, &wscale, in_f);
        let nvfp4_row = dequant_gguf_row(&perm, in_f);
        let num_v_per_k = nv / nk;
        for j in 0..num_v_per_k {
            for g in 0..nk {
                let dst = j * nk + g;
                let src = g * num_v_per_k + j;
                for d in 0..hd {
                    let a = f32row[src * hd + d];
                    let b = nvfp4_row[dst * hd + d];
                    let denom = a.abs().max(1e-6);
                    assert!((a - b).abs() / denom < 1e-6, "col dst {dst}<-src {src} d {d}: {a} != {b}");
                }
            }
        }
    }

    /// A1 direct-import gate (bw24-gguf side): the fused modelopt->split repack must equal the
    /// per-block redistribution of `repack_modelopt_to_gguf`'s output — quant plane block (o,b)
    /// == GGUF block qs[4..36], scale plane 4B == GGUF block d[0..4]. (The engine-side test in
    /// model.rs additionally pins the composition through the real `repack_nvfp4_split`.)
    #[test]
    fn split_plane_matches_gguf_blocks() {
        for (out_f, in_f) in [(1usize, 64usize), (3, 128), (5, 320), (8, 1024)] {
            let in_bytes = in_f / 2;
            let scl_bytes = in_f / 16;
            let mut weight = vec![0u8; out_f * in_bytes];
            let mut wscale = vec![0u8; out_f * scl_bytes];
            for (i, b) in weight.iter_mut().enumerate() { *b = ((i * 37 + 11) & 0xFF) as u8; }
            for (i, b) in wscale.iter_mut().enumerate() { *b = (0x20 + ((i * 13 + 3) % 0x50)) as u8; }
            let gguf = repack_modelopt_to_gguf(&weight, &wscale, out_f, in_f);
            let split = repack_modelopt_to_split(&weight, &wscale, out_f, in_f);
            assert_eq!(split.len(), gguf.len());
            let nblk = in_f / 64;
            let qp = out_f * nblk * 32;
            for o in 0..out_f {
                for b in 0..nblk {
                    let blk = &gguf[(o * nblk + b) * 36..(o * nblk + b) * 36 + 36];
                    assert_eq!(&split[(o * nblk + b) * 32..(o * nblk + b) * 32 + 32], &blk[4..36],
                               "quant plane o={o} b={b} out_f={out_f} in_f={in_f}");
                    assert_eq!(&split[qp + (o * nblk + b) * 4..qp + (o * nblk + b) * 4 + 4], &blk[0..4],
                               "scale plane o={o} b={b} out_f={out_f} in_f={in_f}");
                }
            }
            // Scale plane must be the modelopt weight_scale VERBATIM (the memcpy claim).
            assert_eq!(&split[qp..], &wscale[..], "scale plane != weight_scale verbatim");
        }
    }

    /// Nibble-order conversion spot-check: element 1 (modelopt high nibble of byte 0) must land in
    /// GGUF block 0 sub-block 0 byte 1 LOW nibble (elem 1 = s=0,j=1,lo).
    #[test]
    fn nibble_order_conversion() {
        let in_f = 64usize;
        // weight byte 0 = (code_e1<<4)|code_e0 ; pick code_e0=5, code_e1=9
        let mut weight = vec![0u8; in_f / 2];
        weight[0] = (9 << 4) | 5;
        let wscale = vec![0x38u8; in_f / 16]; // arbitrary finite scale
        let packed = repack_modelopt_to_gguf(&weight, &wscale, 1, in_f);
        // GGUF block 0: blk[4 + 0*8 + 0] low nibble = elem 0 = 5 ; high nibble = elem 8.
        assert_eq!(packed[4 + 0] & 0x0F, 5, "elem0 -> block0 byte0 low");
        // elem 1 is GGUF s=0,j=1 low nibble -> blk[4 + 1] low nibble.
        assert_eq!(packed[4 + 1] & 0x0F, 9, "elem1 -> block0 byte1 low");
    }
}

#[cfg(test)]
mod fp8_tests {
    use super::fp8_e4m3_to_f32;
    #[test]
    fn e4m3_known_values() {
        // e4m3 bias 7: 0x38 = s0 e7 m0 = 1.0; 0x3C = 1.5; 0x40 = 2.0; 0xB8 = -1.0;
        // 0x08 = e1 m0 = 2^-6; 0x01 = subnormal 1/8 * 2^-6; 0x7F/0xFF = NaN -> 0
        assert_eq!(fp8_e4m3_to_f32(0x38), 1.0);
        assert_eq!(fp8_e4m3_to_f32(0x3C), 1.5);
        assert_eq!(fp8_e4m3_to_f32(0x40), 2.0);
        assert_eq!(fp8_e4m3_to_f32(0xB8), -1.0);
        assert_eq!(fp8_e4m3_to_f32(0x08), 2f32.powi(-6));
        assert_eq!(fp8_e4m3_to_f32(0x01), 0.125 * 2f32.powi(-6));
        assert_eq!(fp8_e4m3_to_f32(0x7F), 0.0);
        assert_eq!(fp8_e4m3_to_f32(0xFF), 0.0);
        assert_eq!(fp8_e4m3_to_f32(0x00), 0.0);
        // e4m3 max normal = 448 (0x7E)
        assert_eq!(fp8_e4m3_to_f32(0x7E), 448.0);
    }

    /// BW24_PP_FP8 loader gate: the f32->e4m3 encoder must invert the decoder over every non-NaN
    /// code, INCLUDING through the weight_scale fold the Transform arm performs
    /// (`encode(decode(b) * scale / scale) == b` — the exact-permutation round-trip claim).
    #[test]
    fn e4m3_encode_roundtrip() {
        use super::f32_to_fp8_e4m3;
        for b in 0u16..=0xFF {
            let b = b as u8;
            if b & 0x7F == 0x7F || b == 0x80 { continue; } // NaN codes; -0 canonicalizes to 0x00
            assert_eq!(f32_to_fp8_e4m3(fp8_e4m3_to_f32(b)), b, "roundtrip code {b:#04x}");
            for scale in [1.0f32, 0.007831, 3.25e-3, 17.0, 1.9e-5] {
                let v = fp8_e4m3_to_f32(b) * scale;
                assert_eq!(f32_to_fp8_e4m3(v / scale), b, "scale-fold roundtrip {b:#04x} s={scale}");
            }
        }
        // saturation + zero + NaN policy
        assert_eq!(f32_to_fp8_e4m3(1e30), 0x7E);
        assert_eq!(f32_to_fp8_e4m3(-1e30), 0xFE);
        assert_eq!(f32_to_fp8_e4m3(0.0), 0x00);
        assert_eq!(f32_to_fp8_e4m3(-0.0), 0x00);
        assert_eq!(f32_to_fp8_e4m3(f32::NAN), 0x00);
        // ties-to-even on the grid midpoint: between 1.0 (0x38) and 1.0625? no — e4m3 step at
        // exp 7 is 0.125: midpoint 1.0625 between 1.0 (0x38, even) and 1.125 (0x39) -> even.
        assert_eq!(f32_to_fp8_e4m3(1.0625), 0x38);
    }
}
