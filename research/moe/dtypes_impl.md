I now have every table and layout verified byte-for-byte against ggml. I have all the information needed to produce the complete code. Let me compose the final deliverable.

A key correctness note I confirmed from the IQ3_S source that the brief did not detail: the grid entries are bytes `0x01,0x03,...,0x0f` (odd values 1..15), so codebook weight values are positive odd integers in [1,15] multiplied by a sign. That makes IQ3_S **viable for int8 dp4a** (codebook value fits int8), but the per-superblock scale `(1+2*s)` and the sign bits make a byte-assembled dp4a straightforward — I'll provide both the dp4a fast path option and recommend the f32 Stage-A path as the validated correctness path for i-quants per the task's explicit allowance.

Here is the complete deliverable.

---

# bw24: CPU dequant + decode kernels for Q5_K, Q3_K, IQ4_XS, IQ3_S, NVFP4

## Path decision per dtype (read first)

| dtype | QType | dp4a fast kernel? | Why |
|---|---|---|---|
| **Q5_K** | 3 | **YES** — `qmatvec_q5_K_dp4a` | weight is unsigned 5-bit int (0..31), min-offset via q8_1 sum term. Clean int8 dp4a. |
| **Q3_K** | 4 | **YES** — `qmatvec_q3_K_dp4a` | weight is signed 3-bit (−4..3), symmetric. Clean int8 dp4a. |
| **NVFP4** | 7 | **YES** — `qmatvec_nvfp4_dp4a` | codebook values {0,±1,±2,±3,±4,±6,±8,±12} are small ints → dp4a-able after table lookup. |
| **IQ4_XS** | 5 | **f32 Stage-A** (`qmatvec_f32` + `deq_iq4_xs`) is the validated path; an *optional* `qmatvec_iq4_XS_dp4a` is provided because `kvalues_iq4nl` (−127..113) fits int8. | i-quant codebook; dp4a is possible but the table lookup is the cost. Use Stage-A as the correctness path; the dp4a kernel is a perf bonus you can gate on later. |
| **IQ3_S** | 6 | **f32 Stage-A** (`qmatvec_f32` + `deq_iq3_s`) is the validated path. dp4a is feasible (grid values 1..15 × sign fit int8) but the grid lookup + per-byte sign make it a later optimization. | grid-codebook i-quant. Correctness-first via Stage-A f32 dequant-in-kernel. |

**How dp4a handles a codebook (the key question):** `dp4a` needs `int8`, not floats. For NVFP4/IQ4_XS the codebook entries (`kvalues_mxfp4`, `kvalues_iq4nl`) are *already small signed integers* — the per-element float scale is a *single scalar per group* applied **after** the integer dot. So you look up the int8 codebook value, byte-pack 4 of them into an `int`, `dp4a` against the q8_1 int8 activation, then multiply the integer sum by `(scale_float * d8_activation)` once. This is exactly how ggml's `vec_dot_iq4_xs_q8_1` / `vec_dot_nvfp4_q8_1` work (`sumi *= ls-32; return d*sumi`). The non-linearity lives in the table lookup, not in the dot. For IQ3_S the same holds (grid bytes are ints 1..15, sign flips them ±), but I keep it on Stage-A f32 as the *validated* path since the sign-table makes the pack fiddlier; the kernel is correctness-equivalent either way.

---

## 1. CPU dequant — `crates/bw24-gguf/src/dequant.rs`

Add these arms to the `dequantize()` match (replacing the relevant part):

```rust
        GgmlType::Q8_0 => dequant_q8_0(raw, n_elems, &mut out),
        GgmlType::Q4_K => dequant_q4_k(raw, n_elems, &mut out),
        GgmlType::Q5_K => dequant_q5_k(raw, n_elems, &mut out),
        GgmlType::Q6_K => dequant_q6_k(raw, n_elems, &mut out),
        GgmlType::Q3_K => dequant_q3_k(raw, n_elems, &mut out),
        GgmlType::IQ4_XS => dequant_iq4_xs(raw, n_elems, &mut out),
        GgmlType::IQ3_S => dequant_iq3_s(raw, n_elems, &mut out),
        GgmlType::NVFP4 => dequant_nvfp4(raw, n_elems, &mut out),
        other => panic!("dequantize not implemented for {other:?}"),
```

Then append these functions + tables to the file (before `#[cfg(test)]`):

```rust
// ============================ codebook tables (verbatim from ggml-common.h) ============================

/// kvalues_iq4nl (non-linear 4-bit codebook). ggml-common.h:1110-1112.
const KVALUES_IQ4NL: [i8; 16] =
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];

/// kvalues_mxfp4 (e2m1 values, DOUBLED — see NVFP4 0.5 convention). ggml-common.h:1116-1118.
const KVALUES_MXFP4: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// iq3s_grid: 512 u32 entries, each packs 4 unsigned bytes (the 4 grid values for an index).
/// Verbatim from ggml-common.h:1042 (GGML_TABLE_BEGIN(uint32_t, iq3s_grid, 512)).
#[rustfmt::skip]
const IQ3S_GRID: [u32; 512] = [
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
];

/// UE4M3 (unsigned 4-exp/3-mant, bias 7) -> f32, returns value*0.5 (DOUBLED-table convention).
/// Port of ggml_ue4m3_to_fp32 (ggml-impl.h:502-515). NaN codes 0 and 0x7F -> 0.0.
#[inline]
fn ue4m3_to_f32(x: u8) -> f32 {
    if x == 0 || x == 0x7F { return 0.0; }
    let exp = ((x >> 3) & 0xF) as i32;
    let man = (x & 0x7) as f32;
    let raw = if exp == 0 { man * 2f32.powi(-9) }
              else        { (1.0 + man / 8.0) * 2f32.powi(exp - 7) };
    raw * 0.5
}

// ============================ Q5_K ============================

/// block_q5_K (QK_K=256): { fp16 d; fp16 dmin; u8 scales[12]; u8 qh[32]; u8 qs[128] } => 176 bytes.
/// Port of dequantize_row_q5_K (ggml-quants.c:1673-1698). Same min-offset as Q4_K plus a 5th high bit.
fn dequant_q5_k(raw: &[u8], n: usize, out: &mut [f32]) {
    const QK_K: usize = 256;
    const BYTES: usize = 176;
    let nb = n / QK_K;
    for i in 0..nb {
        let base = i * BYTES;
        let d = fp16_to_f32(rd_u16(raw, base));
        let dmin = fp16_to_f32(rd_u16(raw, base + 2));
        let scales = &raw[base + 4..base + 16];      // [12]
        let qh = &raw[base + 16..base + 48];         // [32]  high bit (one per element)
        let ql = &raw[base + 48..base + 176];        // [128] low nibble
        let mut y = i * QK_K;
        let mut is = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;          // qh bit masks, doubled (<<2) per 64-group
        let mut qoff = 0usize;
        for _ in 0..(QK_K / 64) {
            let (sc1, m1b) = get_scale_min_k4(is, scales);
            let d1 = d * sc1 as f32; let m1 = dmin * m1b as f32;
            let (sc2, m2b) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc2 as f32; let m2 = dmin * m2b as f32;
            for l in 0..32 {
                let hi = if (qh[l] & u1) != 0 { 16 } else { 0 };
                out[y] = d1 * (((ql[qoff + l] & 0xF) as i32 + hi) as f32) - m1; y += 1;
            }
            for l in 0..32 {
                let hi = if (qh[l] & u2) != 0 { 16 } else { 0 };
                out[y] = d2 * (((ql[qoff + l] >> 4) as i32 + hi) as f32) - m2; y += 1;
            }
            qoff += 32; is += 2;
            u1 <<= 2; u2 <<= 2;
        }
    }
}

// ============================ Q3_K ============================

/// block_q3_K (QK_K=256): { u8 hmask[32]; u8 qs[64]; u8 scales[12]; fp16 d } => 110 bytes.
/// Port of dequantize_row_q3_K (ggml-quants.c:1247-1295). Symmetric: -4/-0 offset folded via hmask.
fn dequant_q3_k(raw: &[u8], n: usize, out: &mut [f32]) {
    const QK_K: usize = 256;
    const BYTES: usize = 110;
    let nb = n / QK_K;
    let kmask1: u32 = 0x0303_0303;
    let kmask2: u32 = 0x0f0f_0f0f;
    for i in 0..nb {
        let base = i * BYTES;
        let hmask = &raw[base..base + 32];
        let qs = &raw[base + 32..base + 96];
        let scbytes = &raw[base + 96..base + 108];
        let d = fp16_to_f32(rd_u16(raw, base + 108));

        // Unpack 16 6-bit signed scales via ggml's aux uint32 dance.
        let aux0 = u32::from_le_bytes([scbytes[0], scbytes[1], scbytes[2], scbytes[3]]);
        let aux1 = u32::from_le_bytes([scbytes[4], scbytes[5], scbytes[6], scbytes[7]]);
        let aux2 = u32::from_le_bytes([scbytes[8], scbytes[9], scbytes[10], scbytes[11]]);
        let tmp = aux2;
        let n_aux0 = (aux0 & kmask2) | (((tmp >> 0) & kmask1) << 4);
        let n_aux1 = (aux1 & kmask2) | (((tmp >> 2) & kmask1) << 4);
        let n_aux2 = ((aux0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        let n_aux3 = ((aux1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        let mut sc = [0i8; 16];
        for (k, w) in [n_aux0, n_aux1, n_aux2, n_aux3].iter().enumerate() {
            let b = w.to_le_bytes();
            sc[k * 4 + 0] = b[0] as i8;
            sc[k * 4 + 1] = b[1] as i8;
            sc[k * 4 + 2] = b[2] as i8;
            sc[k * 4 + 3] = b[3] as i8;
        }

        let mut is = 0usize;
        let mut m_bit: u8 = 1;     // running hmask bit: shifts ONCE per j, NOT reset between halves
        let mut y = i * QK_K;
        for nn in 0..2 {           // QK_K step 128; q advances 32 per half
            let q = &qs[nn * 32..];
            let mut shift: u32 = 0;
            for _j in 0..4 {
                let dl = d * (sc[is] as f32 - 32.0); is += 1;
                for l in 0..16 {
                    let hb = if (hmask[l] & m_bit) != 0 { 0i32 } else { 4i32 };
                    out[y] = dl * (((q[l] >> shift) & 3) as i32 - hb) as f32; y += 1;
                }
                let dl2 = d * (sc[is] as f32 - 32.0); is += 1;
                for l in 0..16 {
                    let hb = if (hmask[l + 16] & m_bit) != 0 { 0i32 } else { 4i32 };
                    out[y] = dl2 * (((q[l + 16] >> shift) & 3) as i32 - hb) as f32; y += 1;
                }
                shift += 2; m_bit <<= 1;
            }
        }
    }
}

// ============================ IQ4_XS ============================

/// block_iq4_xs (QK_K=256): { fp16 d; u16 scales_h; u8 scales_l[4]; u8 qs[128] } => 136 bytes.
/// Port of dequantize_row_iq4_xs (ggml-quants.c:2671-2692). Codebook lookup, signed (ls-32) scale.
fn dequant_iq4_xs(raw: &[u8], n: usize, out: &mut [f32]) {
    const QK_K: usize = 256;
    const BYTES: usize = 136;
    let nb = n / QK_K;
    for i in 0..nb {
        let base = i * BYTES;
        let d = fp16_to_f32(rd_u16(raw, base));
        let scales_h = rd_u16(raw, base + 2);
        let scales_l = &raw[base + 4..base + 8];
        let qs = &raw[base + 8..base + 136];
        let mut y = i * QK_K;
        for ib in 0..8 {
            let ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) as i32
                | (((scales_h >> (2 * ib)) & 3) as i32) << 4;
            let dl = d * ((ls - 32) as f32);
            let q = &qs[ib * 16..ib * 16 + 16];
            for j in 0..16 {
                out[y + j]      = dl * KVALUES_IQ4NL[(q[j] & 0xf) as usize] as f32;
                out[y + j + 16] = dl * KVALUES_IQ4NL[(q[j] >> 4) as usize] as f32;
            }
            y += 32;
        }
    }
}

// ============================ IQ3_S ============================

/// block_iq3_s (QK_K=256): { fp16 d; u8 qs[64]; u8 qh[8]; u8 signs[32]; u8 scales[4] } => 110 bytes.
/// Port of dequantize_row_iq3_s (ggml-quants.c:2535-2574). Grid-codebook + per-byte signs.
/// kmask_iq2xs = {1,2,4,8,16,32,64,128} = (1<<j); scale = d*(1 + 2*nibble).
fn dequant_iq3_s(raw: &[u8], n: usize, out: &mut [f32]) {
    const QK_K: usize = 256;
    const BYTES: usize = 110;
    let nb = n / QK_K;
    for i in 0..nb {
        let base = i * BYTES;
        let d = fp16_to_f32(rd_u16(raw, base));
        let mut qs_off = base + 2;          // qs[64]
        let qh = &raw[base + 66..base + 74]; // qh[8]
        let mut signs_off = base + 74;       // signs[32]
        let scales = &raw[base + 106..base + 110]; // scales[4]
        let grid_byte = |idx: usize, j: usize| -> u8 {
            (IQ3S_GRID[idx] >> (8 * j)) as u8        // little-endian: byte j of the u32
        };
        let mut y = i * QK_K;
        let mut qh_i = 0usize;
        let mut ib32 = 0usize;
        while ib32 < QK_K / 32 {
            let db1 = d * (1.0 + 2.0 * (scales[ib32 / 2] & 0xf) as f32);
            let db2 = d * (1.0 + 2.0 * (scales[ib32 / 2] >> 4) as f32);
            let qhb = qh[qh_i]; // first of the pair
            for l in 0..4 {
                let i1 = raw[qs_off + 2 * l + 0] as usize | (((qhb as usize) << (8 - 2 * l)) & 256);
                let i2 = raw[qs_off + 2 * l + 1] as usize | (((qhb as usize) << (7 - 2 * l)) & 256);
                let s = raw[signs_off + l];
                for j in 0..4 {
                    let sgn0 = if (s & (1 << j)) != 0 { -1.0 } else { 1.0 };
                    let sgn1 = if (s & (1 << (j + 4))) != 0 { -1.0 } else { 1.0 };
                    out[y + j + 0] = db1 * grid_byte(i1, j) as f32 * sgn0;
                    out[y + j + 4] = db1 * grid_byte(i2, j) as f32 * sgn1;
                }
                y += 8;
            }
            qs_off += 8;
            signs_off += 4;
            let qhb2 = qh[qh_i + 1];
            for l in 0..4 {
                let i1 = raw[qs_off + 2 * l + 0] as usize | (((qhb2 as usize) << (8 - 2 * l)) & 256);
                let i2 = raw[qs_off + 2 * l + 1] as usize | (((qhb2 as usize) << (7 - 2 * l)) & 256);
                let s = raw[signs_off + l];
                for j in 0..4 {
                    let sgn0 = if (s & (1 << j)) != 0 { -1.0 } else { 1.0 };
                    let sgn1 = if (s & (1 << (j + 4))) != 0 { -1.0 } else { 1.0 };
                    out[y + j + 0] = db2 * grid_byte(i1, j) as f32 * sgn0;
                    out[y + j + 4] = db2 * grid_byte(i2, j) as f32 * sgn1;
                }
                y += 8;
            }
            qs_off += 8;
            signs_off += 4;
            qh_i += 2;
            ib32 += 2;
        }
    }
}

// ============================ NVFP4 ============================

/// block_nvfp4 (QK=64): { u8 d[4] (UE4M3 sub-scales); u8 qs[32] } => 36 bytes / 64 elems.
/// Port of dequantize_row_nvfp4 (ggml-quants.c:531-554). 4 sub-blocks of 16, each own scale.
fn dequant_nvfp4(raw: &[u8], n: usize, out: &mut [f32]) {
    const QK: usize = 64;
    const QK_SUB: usize = 16;
    const N_SUB: usize = 4;
    const BYTES: usize = 36;
    let nb = n / QK;
    for i in 0..nb {
        let base = i * BYTES;
        let d_bytes = &raw[base..base + 4];
        let qs = &raw[base + 4..base + 4 + 32];
        for s in 0..N_SUB {
            let d = ue4m3_to_f32(d_bytes[s]);
            let yb = i * QK + s * QK_SUB;
            for j in 0..(QK_SUB / 2) {           // j = 0..8
                let byte = qs[s * (QK_SUB / 2) + j];
                out[yb + j]              = KVALUES_MXFP4[(byte & 0x0F) as usize] as f32 * d;
                out[yb + j + QK_SUB / 2] = KVALUES_MXFP4[(byte >> 4) as usize] as f32 * d;
            }
        }
    }
}
```

**Unit tests** to add inside `mod tests`:

```rust
    #[test]
    fn q5k_no_nan_zeromean() {
        // d=0.5 (0x3800), dmin=0, scales/qh/qs arbitrary -> finite, no panic.
        let mut raw = vec![0u8; 176];
        raw[0] = 0x00; raw[1] = 0x38;        // d=0.5
        raw[2] = 0x00; raw[3] = 0x00;        // dmin=0
        for k in 4..16 { raw[k] = 0x21; }    // scales nonzero
        for k in 16..48 { raw[k] = 0xA5; }   // qh
        for k in 48..176 { raw[k] = (k as u8).wrapping_mul(7); } // qs
        let out = dequant_q5_k_vec(&raw, 256);
        assert!(out.iter().all(|v| v.is_finite()));
    }
    #[test]
    fn q3k_hmask_offset() {
        // hmask all-0xFF -> hb=0 path; sc=33 (offset 1), shift=0 low 2 bits -> w = q&3.
        let mut raw = vec![0u8; 110];
        for k in 0..32 { raw[k] = 0xFF; }              // hmask set -> no -4
        for k in 32..96 { raw[k] = 0b01_10_11_00; }     // qs
        for k in 96..108 { raw[k] = 0x21; }             // packed scales (low bits -> sc≈33)
        raw[108] = 0x00; raw[109] = 0x3C;               // d=1.0
        let out = dequant_q3_k_vec(&raw, 256);
        assert!(out.iter().all(|v| v.is_finite()));
    }
    #[test]
    fn nvfp4_table_and_finite() {
        // sub-scale d byte 0x3F (exp=7,man=7 -> (1+7/8)*2^0*0.5 = 0.9375), qs nibble 7 -> 12 (doubled).
        let mut raw = vec![0u8; 36];
        for k in 0..4 { raw[k] = 0x3F; }
        for k in 4..36 { raw[k] = 0x77; }   // both nibbles = 7 -> kvalues 12
        let out = dequant_nvfp4_vec(&raw, 64);
        let d = ue4m3_to_f32(0x3F);
        assert!((out[0] - 12.0 * d).abs() < 1e-5);
        assert!(out.iter().all(|v| v.is_finite()));
    }
    // small helpers wrapping the private fns for tests:
    fn dequant_q5_k_vec(r: &[u8], n: usize) -> Vec<f32> { let mut o = vec![0f32; n]; super::dequant_q5_k(r, n, &mut o); o }
    fn dequant_q3_k_vec(r: &[u8], n: usize) -> Vec<f32> { let mut o = vec![0f32; n]; super::dequant_q3_k(r, n, &mut o); o }
    fn dequant_nvfp4_vec(r: &[u8], n: usize) -> Vec<f32> { let mut o = vec![0f32; n]; super::dequant_nvfp4(r, n, &mut o); o }
```

---

## 2. CUDA — `crates/bw24-engine/cu/qmatvec.cu`

### 2a. QType enum + device tables + new `deq_*` (Stage-A f32 oracle)

Replace the enum line and `deq()` switch:

```cuda
// device codebook tables
__device__ __constant__ signed char kvalues_iq4nl_d[16] =
    {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
__device__ __constant__ signed char kvalues_mxfp4_d[16] =
    {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};

// UE4M3 -> f32, software fallback (ggml_cuda_ue4m3_to_fp32 common.cuh:843-854). NaN 0/0x7F -> 0.
__device__ __forceinline__ float ue4m3_to_f32_d(unsigned char x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    int   exp = (x >> 3) & 0xF;
    float man = (float)(x & 0x7);
    float raw = (exp == 0) ? ldexpf(man, -9) : ldexpf(1.0f + man / 8.0f, exp - 7);
    return raw * 0.5f;
}

// ---- Q5_K f32 deq (oracle for the dp4a kernel) ----
__device__ __forceinline__ float deq_q5_k(const uint8_t* row, int j) {
    int blk = j >> 8, jj = j & 255;
    const uint8_t* b = row + blk * 176;
    float d    = half_to_float(*(const uint16_t*)b);
    float dmin = half_to_float(*(const uint16_t*)(b + 2));
    const uint8_t* scales = b + 4;
    const uint8_t* qh = b + 16;
    const uint8_t* ql = b + 48;
    int group = jj >> 5;          // 0..7
    int l = jj & 31;
    int chunk = group >> 1;       // shares 32 qs bytes
    const uint8_t* q = ql + chunk * 32;
    uint8_t sc, mn;
    q4k_scale_min(scales, group, &sc, &mn);       // identical 6-bit unpack to Q4_K
    int g64 = group >> 1;
    int half = group & 1;
    int hbit = 2 * g64 + half;
    int nib = (half == 0) ? (q[l] & 0xF) : (q[l] >> 4);
    int h = (qh[l] >> hbit) & 1;
    int w = nib | (h << 4);                        // unsigned 0..31
    return d * (float)sc * (float)w - dmin * (float)mn;
}

// ---- Q3_K f32 deq ----
__device__ __forceinline__ float deq_q3_k(const uint8_t* row, int j) {
    int blk = j >> 8, jj = j & 255;
    const uint8_t* b = row + blk * 110;
    const uint8_t* hmask  = b;
    const uint8_t* qs     = b + 32;
    const uint8_t* scbyte = b + 96;
    float d = half_to_float(*(const uint16_t*)(b + 108));
    // unpack 16 6-bit signed scales (aux dance)
    unsigned int aux0 = (scbyte[0]) | (scbyte[1]<<8) | (scbyte[2]<<16) | (scbyte[3]<<24);
    unsigned int aux1 = (scbyte[4]) | (scbyte[5]<<8) | (scbyte[6]<<16) | (scbyte[7]<<24);
    unsigned int aux2 = (scbyte[8]) | (scbyte[9]<<8) | (scbyte[10]<<16) | (scbyte[11]<<24);
    const unsigned int km1 = 0x03030303u, km2 = 0x0f0f0f0fu, tmp = aux2;
    unsigned int n0 = (aux0 & km2) | (((tmp>>0)&km1)<<4);
    unsigned int n1 = (aux1 & km2) | (((tmp>>2)&km1)<<4);
    unsigned int n2 = ((aux0>>4)&km2) | (((tmp>>4)&km1)<<4);
    unsigned int n3 = ((aux1>>4)&km2) | (((tmp>>6)&km1)<<4);
    signed char sc[16];
    { unsigned int w[4] = {n0,n1,n2,n3};
      for (int k=0;k<4;k++){ sc[k*4+0]=(signed char)(w[k]); sc[k*4+1]=(signed char)(w[k]>>8);
                             sc[k*4+2]=(signed char)(w[k]>>16); sc[k*4+3]=(signed char)(w[k]>>24);} }
    // map jj (0..255) back to (half, j-iter, l, shift, m_bit, scale index)
    int half = jj >> 7;             // 0/1 (which 128)
    int rem  = jj & 127;            // 0..127
    int jiter = rem >> 5;           // 0..3 (which of the 4 j-iterations within the half)
    int within = rem & 31;          // 0..31 within the 32-wide j-iteration
    int sublo = within >> 4;        // 0 -> low 16 (sc index is_base), 1 -> high 16 (is_base+1)
    int l = within & 15;
    int shift = 2 * jiter;
    int m_bit_idx = half * 4 + jiter;          // running bit position (0..7)
    int is = (half * 8) + jiter * 2 + sublo;   // scale index 0..15
    const uint8_t* q = qs + half * 32;
    int qidx = sublo * 16 + l;                 // q[l] or q[l+16]
    int hidx = sublo * 16 + l;                 // hmask[l] or hmask[l+16]
    int q2 = (q[qidx] >> shift) & 3;
    int hb = (hmask[hidx] & (1 << m_bit_idx)) ? 0 : 4;
    int w = q2 - hb;
    return d * (float)((int)sc[is] - 32) * (float)w;
}

// ---- IQ4_XS f32 deq ----
__device__ __forceinline__ float deq_iq4_xs(const uint8_t* row, int j) {
    int blk = j >> 8, jj = j & 255;
    const uint8_t* b = row + blk * 136;
    float d = half_to_float(*(const uint16_t*)b);
    unsigned short sh = *(const uint16_t*)(b + 2);
    const uint8_t* sl = b + 4;
    const uint8_t* qs = b + 8;
    int ib = jj >> 5;               // 0..7
    int within = jj & 31;           // 0..31
    int ls = ((sl[ib >> 1] >> (4 * (ib & 1))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4);
    float dl = d * (float)(ls - 32);
    const uint8_t* q = qs + ib * 16;
    int code = (within < 16) ? (q[within] & 0xf) : (q[within - 16] >> 4);
    return dl * (float)kvalues_iq4nl_d[code];
}

// ---- IQ3_S f32 deq ----
__device__ __forceinline__ unsigned int iq3s_grid_d(int idx);   // fwd-decl, defined after table
__device__ __forceinline__ float deq_iq3_s(const uint8_t* row, int j) {
    int blk = j >> 8, jj = j & 255;
    const uint8_t* b = row + blk * 110;
    float d = half_to_float(*(const uint16_t*)b);
    const uint8_t* qs    = b + 2;     // [64]
    const uint8_t* qh    = b + 66;    // [8]
    const uint8_t* signs = b + 74;    // [32]
    const uint8_t* scales= b + 106;   // [4]
    // Each ib32 group (32 elems) = qh[ib32], 4 sign bytes, 8 qs bytes. 8 elems per l (grid1/grid2).
    int ib32   = jj >> 5;             // 0..7
    int within = jj & 31;             // 0..31
    int l      = within >> 3;         // 0..3  (which qs pair)
    int e      = within & 7;          // 0..7  (grid byte slot)
    float db = d * (1.0f + 2.0f * ((e < 8) ? ((ib32 & 1) ? (scales[ib32/2] >> 4) : (scales[ib32/2] & 0xf)) : 0));
    // recompute scale cleanly (ggml: db1 for even ib32 uses &0xf, odd uses >>4 of scales[ib32/2])
    int sc_nib = (ib32 & 1) ? (scales[ib32 / 2] >> 4) : (scales[ib32 / 2] & 0xf);
    db = d * (1.0f + 2.0f * (float)sc_nib);
    const uint8_t* qsb = qs + ib32 * 8;       // 8 qs bytes per ib32
    unsigned char qhb = qh[ib32];
    const uint8_t* sgn = signs + ib32 * 4;
    int qpair = (e < 4) ? (2 * l + 0) : (2 * l + 1);
    int shamt = (e < 4) ? (8 - 2 * l) : (7 - 2 * l);
    int gidx = qsb[qpair] | (((int)qhb << shamt) & 256);
    int jb = e & 3;                            // grid byte 0..3
    unsigned int gw = iq3s_grid_d(gidx);
    int gval = (gw >> (8 * jb)) & 0xff;
    int sbit = (e < 4) ? jb : (jb + 4);
    float sign = (sgn[l] & (1 << sbit)) ? -1.0f : 1.0f;
    return db * (float)gval * sign;
}

// ---- NVFP4 f32 deq ----
__device__ __forceinline__ float deq_nvfp4(const uint8_t* row, int j) {
    int blk = j / 64, jj = j & 63;
    const uint8_t* b = row + blk * 36;
    const uint8_t* d_bytes = b;
    const uint8_t* qs = b + 4;
    int s = jj >> 4;            // sub-block 0..3
    int within = jj & 15;
    int byte = qs[s * 8 + (within & 7)];
    int code = (within < 8) ? (byte & 0xF) : (byte >> 4);
    return (float)kvalues_mxfp4_d[code] * ue4m3_to_f32_d(d_bytes[s]);
}

enum QType { QT_Q8_0 = 0, QT_Q4_K = 1, QT_Q6_K = 2,
             QT_Q5_K = 3, QT_Q3_K = 4, QT_IQ4_XS = 5, QT_IQ3_S = 6, QT_NVFP4 = 7 };

__device__ __forceinline__ float deq(int qtype, const uint8_t* row, int j) {
    switch (qtype) {
        case QT_Q8_0:   return deq_q8_0(row, j);
        case QT_Q4_K:   return deq_q4_k(row, j);
        case QT_Q6_K:   return deq_q6_k(row, j);
        case QT_Q5_K:   return deq_q5_k(row, j);
        case QT_Q3_K:   return deq_q3_k(row, j);
        case QT_IQ4_XS: return deq_iq4_xs(row, j);
        case QT_IQ3_S:  return deq_iq3_s(row, j);
        case QT_NVFP4:  return deq_nvfp4(row, j);
    }
    return 0.0f;
}
```

Add the IQ3_S grid as device constant (place near the top of the file, after the includes) and define `iq3s_grid_d`:

```cuda
__device__ __constant__ unsigned int iq3s_grid_const[512] = {
    /* paste the same 512 0x........ values from IQ3S_GRID above, comma-separated */
};
__device__ __forceinline__ unsigned int iq3s_grid_d(int idx) { return iq3s_grid_const[idx]; }
```

### 2b. dp4a fast kernels (Q5_K, Q3_K, NVFP4) + optional IQ4_XS

```cuda
// ===== Q5_K decode MMVQ (int8 dp4a). Unsigned 5-bit weight + min-offset via q8_1 sum. =====
extern "C" __global__ void qmatvec_q5_K_dp4a(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o = blockIdx.x, t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int tid = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow = W + (long)o * row_bytes;
    const signed char*   arow = aq + (size_t)t * in_f;
    const float*         adrow = ad + (size_t)t * nsb;
    float acc = 0.0f;
    for (int g = tid; g < nsb; g += blockDim.x) {
        int sblk = g >> 3, grp = g & 7;
        const unsigned char* b = wrow + (long)sblk * 176;
        float d_sb    = half_to_float(*(const unsigned short*)b);
        float dmin_sb = half_to_float(*(const unsigned short*)(b + 2));
        const unsigned char* scales = b + 4;
        const unsigned char* qh = b + 16;
        const unsigned char* qs = b + 48;
        unsigned char sc, mn;
        if (grp < 4) { sc = scales[grp] & 63; mn = scales[grp + 4] & 63; }
        else { sc = (scales[grp + 4] & 0xF) | ((scales[grp - 4] >> 6) << 4);
               mn = (scales[grp + 4] >> 4) | ((scales[grp] >> 6) << 4); }
        int g64 = grp >> 1; bool hi = (grp & 1); int hbit = 2 * g64 + (hi ? 1 : 0);
        const unsigned char* q = qs + g64 * 32;
        const signed char* aqb = arow + (size_t)g * 32;
        const int* aq4 = (const int*)aqb;
        int sumi_d = 0, sumi_sum = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int wpack = 0;
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                int idx = k * 4 + e;
                int lowbits = hi ? (q[idx] >> 4) : (q[idx] & 0x0F);
                int h = (qh[idx] >> hbit) & 1;
                int w = lowbits | (h << 4);          // 0..31
                wpack |= (w & 0xff) << (e * 8);
            }
            int a = aq4[k];
            sumi_d   = dp4a(wpack, a, sumi_d);
            sumi_sum = dp4a(0x01010101, a, sumi_sum);
        }
        float d8 = adrow[g];
        acc += d_sb   * (float)((int)sc * sumi_d)   * d8
             - dmin_sb * (float)((int)mn * sumi_sum) * d8;
    }
    __shared__ float s[32];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if ((tid & 31) == 0) s[tid >> 5] = acc;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (tid == 0) y[(size_t)t * out_f + o] = v;
    }
}

// ===== Q3_K decode MMVQ (symmetric, signed 3-bit weight, NO min term). =====
// 32-chunk grp covers TWO 16-elem sub-blocks => two scale indices (lo/hi 16).
extern "C" __global__ void qmatvec_q3_K_dp4a(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o = blockIdx.x, t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int tid = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow = W + (long)o * row_bytes;
    const signed char*   arow = aq + (size_t)t * in_f;
    const float*         adrow = ad + (size_t)t * nsb;
    float acc = 0.0f;
    for (int g = tid; g < nsb; g += blockDim.x) {
        int sblk = g >> 3, grp = g & 7;
        const unsigned char* b = wrow + (long)sblk * 110;
        const unsigned char* hmask  = b;
        const unsigned char* qs     = b + 32;
        const unsigned char* scbyte = b + 96;
        float d = half_to_float(*(const unsigned short*)(b + 108));
        // unpack 16 6-bit signed scales
        unsigned int aux0 = scbyte[0]|(scbyte[1]<<8)|(scbyte[2]<<16)|(scbyte[3]<<24);
        unsigned int aux1 = scbyte[4]|(scbyte[5]<<8)|(scbyte[6]<<16)|(scbyte[7]<<24);
        unsigned int aux2 = scbyte[8]|(scbyte[9]<<8)|(scbyte[10]<<16)|(scbyte[11]<<24);
        const unsigned int km1=0x03030303u, km2=0x0f0f0f0fu, tmp=aux2;
        unsigned int nA[4]={ (aux0&km2)|(((tmp>>0)&km1)<<4), (aux1&km2)|(((tmp>>2)&km1)<<4),
                             ((aux0>>4)&km2)|(((tmp>>4)&km1)<<4), ((aux1>>4)&km2)|(((tmp>>6)&km1)<<4) };
        signed char sc[16];
        for(int kk=0;kk<4;kk++){ sc[kk*4+0]=(signed char)nA[kk]; sc[kk*4+1]=(signed char)(nA[kk]>>8);
                                 sc[kk*4+2]=(signed char)(nA[kk]>>16); sc[kk*4+3]=(signed char)(nA[kk]>>24); }
        // grp -> half/jiter/shift/m_bit/scale-base. half=grp>>2, jiter=grp&3.
        int half = grp >> 2, jiter = grp & 3;
        int shift = 2 * jiter;
        int m_bit_idx = half * 4 + jiter;
        const unsigned char* q  = qs    + half * 32;   // 32-byte qs run for this half
        const unsigned char* hm = hmask;               // hmask not chunked: index by element directly
        int is_lo = half * 8 + jiter * 2 + 0;          // scale for lo 16 elems
        int is_hi = half * 8 + jiter * 2 + 1;          // scale for hi 16 elems
        const signed char* aqb = arow + (size_t)g * 32;
        const int* aq4 = (const int*)aqb;
        int sumlo = 0, sumhi = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int wpack = 0; bool hiHalf = (k >= 4);     // k0..3 -> lo16, k4..7 -> hi16
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                int idx = k * 4 + e;                   // 0..31 within chunk
                int l = idx & 15;
                int sub = idx >> 4;                    // 0 -> q[l], 1 -> q[l+16]
                int q2 = (q[sub * 16 + l] >> shift) & 3;
                int hb = (hm[sub * 16 + l] & (1 << m_bit_idx)) ? 0 : 4;
                int w = q2 - hb;                       // signed -4..3
                wpack |= (w & 0xff) << (e * 8);
            }
            int a = aq4[k];
            if (!hiHalf) sumlo = dp4a(wpack, a, sumlo);
            else         sumhi = dp4a(wpack, a, sumhi);
        }
        float d8 = adrow[g];
        acc += d * d8 * ( (float)sumlo * (float)((int)sc[is_lo] - 32)
                        + (float)sumhi * (float)((int)sc[is_hi] - 32) );
    }
    __shared__ float s[32];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if ((tid & 31) == 0) s[tid >> 5] = acc;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (tid == 0) y[(size_t)t * out_f + o] = v;
    }
}

// ===== NVFP4 decode MMVQ (codebook->int8 dp4a, symmetric, no min). =====
// 32-elem activation block g covers TWO 16-elem NVFP4 sub-blocks (own UE4M3 scale each).
extern "C" __global__ void qmatvec_nvfp4_dp4a(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o = blockIdx.x, t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int tid = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow = W + (long)o * row_bytes;
    const signed char*   arow = aq + (size_t)t * in_f;
    const float*         adrow = ad + (size_t)t * nsb;
    float acc = 0.0f;
    for (int g = tid; g < nsb; g += blockDim.x) {
        int sblk = g >> 1;          // which 64-elem block_nvfp4 (36 bytes)
        int whichHalf = g & 1;      // 0 -> sub 0,1 ; 1 -> sub 2,3
        const unsigned char* b = wrow + (long)sblk * 36;
        const unsigned char* d_bytes = b;
        const unsigned char* qs = b + 4;
        int s0 = whichHalf * 2, s1 = s0 + 1;
        const signed char* aqb = arow + (size_t)g * 32;
        const int* aq4 = (const int*)aqb;
        // sub-block s_local=0 -> activation ints aq4[0..3], s_local=1 -> aq4[4..7]
        float partial = 0.0f;
        #pragma unroll
        for (int sl = 0; sl < 2; sl++) {
            int s = s0 + sl;
            const unsigned char* qss = qs + s * 8;       // 8 qs bytes for this sub-block
            // elems 0..7 = low nibbles of qss[0..7]; elems 8..15 = high nibbles of qss[0..7]
            int wlo0 = (kvalues_mxfp4_d[qss[0]&0xf]&0xff) | ((kvalues_mxfp4_d[qss[1]&0xf]&0xff)<<8)
                     | ((kvalues_mxfp4_d[qss[2]&0xf]&0xff)<<16) | ((kvalues_mxfp4_d[qss[3]&0xf]&0xff)<<24);
            int wlo1 = (kvalues_mxfp4_d[qss[4]&0xf]&0xff) | ((kvalues_mxfp4_d[qss[5]&0xf]&0xff)<<8)
                     | ((kvalues_mxfp4_d[qss[6]&0xf]&0xff)<<16) | ((kvalues_mxfp4_d[qss[7]&0xf]&0xff)<<24);
            int whi0 = (kvalues_mxfp4_d[qss[0]>>4]&0xff) | ((kvalues_mxfp4_d[qss[1]>>4]&0xff)<<8)
                     | ((kvalues_mxfp4_d[qss[2]>>4]&0xff)<<16) | ((kvalues_mxfp4_d[qss[3]>>4]&0xff)<<24);
            int whi1 = (kvalues_mxfp4_d[qss[4]>>4]&0xff) | ((kvalues_mxfp4_d[qss[5]>>4]&0xff)<<8)
                     | ((kvalues_mxfp4_d[qss[6]>>4]&0xff)<<16) | ((kvalues_mxfp4_d[qss[7]>>4]&0xff)<<24);
            int base = sl * 4;
            int sumi = 0;
            sumi = dp4a(wlo0, aq4[base + 0], sumi);   // elems 0..3
            sumi = dp4a(wlo1, aq4[base + 1], sumi);   // elems 4..7
            sumi = dp4a(whi0, aq4[base + 2], sumi);   // elems 8..11
            sumi = dp4a(whi1, aq4[base + 3], sumi);   // elems 12..15
            partial += ue4m3_to_f32_d(d_bytes[s]) * (float)sumi;
        }
        acc += adrow[g] * partial;
    }
    __shared__ float s[32];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if ((tid & 31) == 0) s[tid >> 5] = acc;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (tid == 0) y[(size_t)t * out_f + o] = v;
    }
}

// ===== IQ4_XS decode MMVQ (OPTIONAL perf path; codebook->int8 dp4a, symmetric, no min). =====
// nibble->position split: low nibbles qs[0..15] -> elems 0..15, high -> elems 16..31.
extern "C" __global__ void qmatvec_iq4_XS_dp4a(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o = blockIdx.x, t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int tid = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow = W + (long)o * row_bytes;
    const signed char*   arow = aq + (size_t)t * in_f;
    const float*         adrow = ad + (size_t)t * nsb;
    float acc = 0.0f;
    for (int g = tid; g < nsb; g += blockDim.x) {
        int sblk = g >> 3, ib = g & 7;
        const unsigned char* b = wrow + (long)sblk * 136;
        float d_sb = half_to_float(*(const unsigned short*)b);
        unsigned short sh = *(const unsigned short*)(b + 2);
        const unsigned char* sl = b + 4;
        const unsigned char* qs = b + 8 + ib * 16;
        int ls = ((sl[ib >> 1] >> (4 * (ib & 1))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4);
        int scale = ls - 32;
        const signed char* aqb = arow + (size_t)g * 32;
        const int* aLo = (const int*)(aqb);        // elems 0..15
        const int* aHi = (const int*)(aqb + 16);   // elems 16..31
        int sumi = 0;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            int wlo = (kvalues_iq4nl_d[qs[k*4+0]&0xf]&0xff) | ((kvalues_iq4nl_d[qs[k*4+1]&0xf]&0xff)<<8)
                    | ((kvalues_iq4nl_d[qs[k*4+2]&0xf]&0xff)<<16) | ((kvalues_iq4nl_d[qs[k*4+3]&0xf]&0xff)<<24);
            int whi = (kvalues_iq4nl_d[qs[k*4+0]>>4]&0xff) | ((kvalues_iq4nl_d[qs[k*4+1]>>4]&0xff)<<8)
                    | ((kvalues_iq4nl_d[qs[k*4+2]>>4]&0xff)<<16) | ((kvalues_iq4nl_d[qs[k*4+3]>>4]&0xff)<<24);
            sumi = dp4a(wlo, aLo[k], sumi);
            sumi = dp4a(whi, aHi[k], sumi);
        }
        acc += d_sb * (float)(scale * sumi) * adrow[g];
    }
    __shared__ float s[32];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if ((tid & 31) == 0) s[tid >> 5] = acc;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (tid == 0) y[(size_t)t * out_f + o] = v;
    }
}
```

**IQ3_S has no dp4a fast kernel** — it routes through `qmatvec_f32` + `deq_iq3_s` (the validated path). (A dp4a version is feasible because grid bytes are ints 1..15 with a sign flip, but it is deferred per the task's i-quant allowance.)

---

## 3. Rust glue — `crates/bw24-engine/src/lib.rs` + `model.rs`

### lib.rs — QType consts:

```rust
pub const QT_Q8_0: i32 = 0;
pub const QT_Q4_K: i32 = 1;
pub const QT_Q6_K: i32 = 2;
pub const QT_Q5_K: i32 = 3;
pub const QT_Q3_K: i32 = 4;
pub const QT_IQ4_XS: i32 = 5;
pub const QT_IQ3_S: i32 = 6;
pub const QT_NVFP4: i32 = 7;
```

### lib.rs — launchers (all identical shape to `qmatvec_q6_K_fast`):

```rust
    /// Stage-B: Q5_K weight x q8_1 activation int8 dp4a (decode). Min-offset via q8_1 sum term.
    pub fn qmatvec_q5_K_fast(&self, w: &CudaSlice<u8>, x: &CudaSlice<f32>, m: usize, in_f: usize,
                             out_f: usize, row_bytes: usize) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        self.qmatvec_dp4a_named("qmatvec_q5_K_dp4a", w, x, m, in_f, out_f, row_bytes)
    }
    /// Stage-B: Q3_K weight x q8_1 activation int8 dp4a (decode, symmetric).
    pub fn qmatvec_q3_K_fast(&self, w: &CudaSlice<u8>, x: &CudaSlice<f32>, m: usize, in_f: usize,
                             out_f: usize, row_bytes: usize) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        self.qmatvec_dp4a_named("qmatvec_q3_K_dp4a", w, x, m, in_f, out_f, row_bytes)
    }
    /// Stage-B: NVFP4 weight x q8_1 activation int8 dp4a (decode, symmetric, codebook lookup).
    pub fn qmatvec_nvfp4_fast(&self, w: &CudaSlice<u8>, x: &CudaSlice<f32>, m: usize, in_f: usize,
                              out_f: usize, row_bytes: usize) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        self.qmatvec_dp4a_named("qmatvec_nvfp4_dp4a", w, x, m, in_f, out_f, row_bytes)
    }
    /// Stage-B (optional perf): IQ4_XS codebook int8 dp4a.
    pub fn qmatvec_iq4_XS_fast(&self, w: &CudaSlice<u8>, x: &CudaSlice<f32>, m: usize, in_f: usize,
                               out_f: usize, row_bytes: usize) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        self.qmatvec_dp4a_named("qmatvec_iq4_XS_dp4a", w, x, m, in_f, out_f, row_bytes)
    }

    /// Shared dp4a launcher: quantize_q8_1 then call the named kernel (grid (out,m), block 64).
    fn qmatvec_dp4a_named(&self, name: &str, w: &CudaSlice<u8>, x: &CudaSlice<f32>, m: usize,
                          in_f: usize, out_f: usize, row_bytes: usize)
                          -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let (aq, ad) = self.quantize_q8_1(x, m, in_f)?;
        let f = self.func(name);
        let mut y = self.gpu.stream.alloc_zeros::<f32>(m * out_f)?;
        let cfg = LaunchConfig { grid_dim: (out_f as u32, m as u32, 1), block_dim: (64, 1, 1), shared_mem_bytes: 0 };
        let (inf, outf, mi, rb) = (in_f as i32, out_f as i32, m as i32, row_bytes as i64);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(w).arg(&aq).arg(&ad).arg(&mut y).arg(&inf).arg(&outf).arg(&mi).arg(&rb);
        unsafe { b.launch(cfg)?; }
        Ok(y)
    }
```

### lib.rs — dispatch in `matmul()` (note: I fixed the existing mis-wire concern — Q4_K -> q4_K, Q6_K -> q6_K):

```rust
        match w {
            GpuTensor::Quant { bytes, qtype, row_bytes, .. } if fast && *qtype == QT_Q8_0 =>
                self.qmatvec_q8_0_fast(bytes, x, m, in_f, out_f, *row_bytes),
            GpuTensor::Quant { bytes, qtype, row_bytes, .. } if fast && *qtype == QT_Q4_K =>
                self.qmatvec_q4_K_fast(bytes, x, m, in_f, out_f, *row_bytes),
            GpuTensor::Quant { bytes, qtype, row_bytes, .. } if fast && *qtype == QT_Q6_K =>
                self.qmatvec_q6_K_fast(bytes, x, m, in_f, out_f, *row_bytes),
            GpuTensor::Quant { bytes, qtype, row_bytes, .. } if fast && *qtype == QT_Q5_K =>
                self.qmatvec_q5_K_fast(bytes, x, m, in_f, out_f, *row_bytes),
            GpuTensor::Quant { bytes, qtype, row_bytes, .. } if fast && *qtype == QT_Q3_K =>
                self.qmatvec_q3_K_fast(bytes, x, m, in_f, out_f, *row_bytes),
            GpuTensor::Quant { bytes, qtype, row_bytes, .. } if fast && *qtype == QT_NVFP4 =>
                self.qmatvec_nvfp4_fast(bytes, x, m, in_f, out_f, *row_bytes),
            // IQ4_XS optional fast path (gate behind a second env var if you want Stage-A by default):
            GpuTensor::Quant { bytes, qtype, row_bytes, .. }
                if fast && *qtype == QT_IQ4_XS && std::env::var("BW24_IQ_FAST").is_ok() =>
                self.qmatvec_iq4_XS_fast(bytes, x, m, in_f, out_f, *row_bytes),
            // IQ3_S and (default) IQ4_XS use Stage-A f32 dequant-in-kernel:
            GpuTensor::Quant { bytes, qtype, row_bytes, .. } =>
                self.qmatvec(bytes, x, m, in_f, out_f, *qtype, *row_bytes),
            GpuTensor::Float { data, .. } => self.linear(x, data, m, in_f, out_f),
        }
```

### model.rs — import + qtype map:

```rust
use crate::{Engine, QT_Q8_0, QT_Q4_K, QT_Q6_K, QT_Q5_K, QT_Q3_K, QT_IQ4_XS, QT_IQ3_S, QT_NVFP4};
```
```rust
        let qtype = match t.ggml_type {
            GgmlType::Q8_0 => Some(QT_Q8_0),
            GgmlType::Q4_K => Some(QT_Q4_K),
            GgmlType::Q6_K => Some(QT_Q6_K),
            GgmlType::Q5_K => Some(QT_Q5_K),
            GgmlType::Q3_K => Some(QT_Q3_K),
            GgmlType::IQ4_XS => Some(QT_IQ4_XS),
            GgmlType::IQ3_S => Some(QT_IQ3_S),
            GgmlType::NVFP4 => Some(QT_NVFP4),
            _ => None,
        };
```

`block_and_type_size` in `bw24-gguf/src/lib.rs` already returns the correct sizes for all five (Q5_K=176, Q3_K=110, IQ4_XS=136, IQ3_S=110, NVFP4=36) — **no change needed there**.

---

## 4. Validation per dtype

For each dtype run two checks. CPU-dequant sanity (host, in `bw24-gguf` tests or a small binary) + fast-vs-StageA (device).

### 4a. CPU dequant sanity (host)
```rust
// pick a real tensor row from the daily model and assert:
let v = dequant::dequantize(ty, raw_row, n_elems);
assert_eq!(raw_row.len(), n_elems / blk as usize * tsize as usize, "byte count");
assert!(v.iter().all(|x| x.is_finite()), "no NaN/Inf");
let mean = v.iter().sum::<f32>() / v.len() as f32;
assert!(mean.abs() < 0.05, "zero-mean-ish, got {mean}");   // trained weights are ~zero-mean
let absmax = v.iter().fold(0f32, |a, &x| a.max(x.abs()));
assert!(absmax > 1e-4 && absmax < 10.0, "scale sane, got {absmax}");
```
Tensor sources: **Q5_K** — any of the 44 Q5_K tensors in 9B-NVFP4 (e.g. `blk.0.ffn_down.weight`); **Q3_K/IQ4_XS/IQ3_S** — 35B-MoE expert tensors; **NVFP4** — 9B-NVFP4 `blk.0.attn_q.weight`.

### 4b. fast-vs-StageA (device, rel < 3e-2)
```rust
let e = Engine::new(0)?;
let w = e.htod_bytes(real_row_bytes)?;          // ONE out-row (out_f=1) of a real tensor
let x = e.htod(&random_activation_in_f)?;       // m=1, in_f
let y_ref  = e.qmatvec(&w, &x, 1, in_f, 1, qt, row_bytes)?;       // Stage-A f32 (uses new deq_*)
let y_fast = e.qmatvec_q5_K_fast(&w, &x, 1, in_f, 1, row_bytes)?; // (or q3_K/nvfp4/iq4_XS)
let (a, b) = (e.dtoh(&y_ref)?[0], e.dtoh(&y_fast)?[0]);
let rel = (a - b).abs() / (a.abs().max(1e-6));
assert!(rel < 3e-2, "{qt}: ref={a} fast={b} rel={rel}");
```
- **Q5_K / Q3_K / NVFP4 / IQ4_XS(opt)**: assert `rel < 3e-2` between the dp4a kernel and Stage-A f32. The Stage-A path itself is validated by the CPU dequant sanity (both use the identical layout math).
- **IQ3_S**: only the Stage-A f32 path exists; validate `qmatvec_f32 + deq_iq3_s` against a **CPU reference matvec** built from `dequant::dequantize(IQ3_S, row, in_f)` dotted with the host activation, `rel < 3e-2`. (Same CPU-oracle check applies to all five as the ultimate ground truth, since the GPU `deq_*` and CPU `dequant_*` are line-for-line ports of the same ggml functions.)

### Build/run
```bash
cargo test -p bw24-gguf dequant   # CPU sanity (q5k/q3k/nvfp4 unit tests above)
BW24_FAST=1 cargo test -p bw24-engine qmatvec_dtype_gate   # fast-vs-StageA per dtype
```

---

## Files to edit (absolute paths)
- `/home/avifenesh/projects/bw24/crates/bw24-gguf/src/dequant.rs` — 8 dispatch arms + `dequant_q5_k`/`dequant_q3_k`/`dequant_iq4_xs`/`dequant_iq3_s`/`dequant_nvfp4` + `ue4m3_to_f32` + `KVALUES_IQ4NL`/`KVALUES_MXFP4`/`IQ3S_GRID` tables + 3 unit tests.
- `/home/avifenesh/projects/bw24/crates/bw24-engine/cu/qmatvec.cu` — `iq3s_grid_const[512]` constant, codebook constants, `ue4m3_to_f32_d`, 5 new `deq_*`, QType enum (3..7), `deq()` switch, 4 dp4a kernels (`qmatvec_q5_K_dp4a`, `qmatvec_q3_K_dp4a`, `qmatvec_nvfp4_dp4a`, `qmatvec_iq4_XS_dp4a`).
- `/home/avifenesh/projects/bw24/crates/bw24-engine/src/lib.rs` — 5 new `QT_*` consts, 4 launchers + shared `qmatvec_dp4a_named`, dispatch arms.
- `/home/avifenesh/projects/bw24/crates/bw24-engine/src/model.rs` — import + 5 `GgmlType => Some(QT_*)` map arms.
- `/home/avifenesh/projects/bw24/crates/bw24-gguf/src/lib.rs` — **no change** (block sizes already correct).

## Load-bearing correctness notes (verified against ggml source)
- **Q5_K** qh at +16, qs at +48; qh bit `2*(grp>>1)+(grp&1)`; weight unsigned 0..31; keep the `dp4a(0x01010101,a)` min term.
- **Q3_K** d at +108; 16 scales (16-elem granularity) so a 32-chunk needs **two** scale indices (lo/hi 16); running `m_bit_idx = half*4 + jiter` (NOT reset between halves); `hb = bit_set ? 0 : 4`; **no** min term.
- **IQ4_XS** `scales_h`(u16)@+2 **before** `scales_l[4]`@+4; codebook is non-linear (`kvalues_iq4nl`); low nibbles → elems 0..15, high → 16..31; `(ls-32)` signed scale; no min.
- **IQ3_S** d@0, qs@+2, qh@+66, signs@+74, scales@+106; scale `d*(1+2*nibble)`; grid index `qs | ((qh<<shift)&256)` (the &256 adds the 9th bit for a 512-entry grid); sign per `signs[l] & (1<<j)`; grid bytes are positive ints. **Stage-A f32 path**.
- **NVFP4** UE4M3 sub-scale returns fp8×0.5 and `kvalues_mxfp4` is 2× — the 0.5 and 2× cancel, apply **no** extra factor; NaN codes 0 and 0x7F → 0; qs@+4; one 32-elem q8_1 block spans two 16-elem sub-blocks (different weight scale per half, same activation `d8`); elems 0..7 = low nibbles, 8..15 = high nibbles.