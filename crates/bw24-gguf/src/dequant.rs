//! CPU dequantization reference — the correctness oracle for every GPU kernel.
//! Math ported 1:1 from llama.cpp `ggml/src/ggml-quants.c`. Not fast; correct.

use crate::GgmlType;

/// IEEE fp16 (binary16) -> f32. Matches ggml GGML_FP16_TO_FP32.
#[inline]
pub fn fp16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1f;
    let mant = h & 0x3ff;
    let val = match exp {
        0 => {
            if mant == 0 { 0.0 } else {
                // subnormal
                (mant as f32) * 2f32.powi(-24)
            }
        }
        0x1f => {
            if mant == 0 { f32::INFINITY } else { f32::NAN }
        }
        _ => {
            // normal: (1 + mant/1024) * 2^(exp-15)
            (1.0 + (mant as f32) / 1024.0) * 2f32.powi(exp as i32 - 15)
        }
    };
    if sign == 1 { -val } else { val }
}

/// bf16 (upper 16 bits of f32) -> f32.
#[inline]
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

#[inline]
fn rd_u16(b: &[u8], i: usize) -> u16 { u16::from_le_bytes([b[i], b[i + 1]]) }

/// Dequantize `n_elems` from raw tensor bytes into a f32 Vec.
/// `n_elems` must be a multiple of the type's block size.
pub fn dequantize(ty: GgmlType, raw: &[u8], n_elems: usize) -> Vec<f32> {
    let mut out = vec![0f32; n_elems];
    match ty {
        GgmlType::F32 => {
            for i in 0..n_elems {
                out[i] = f32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap());
            }
        }
        GgmlType::F16 => {
            for i in 0..n_elems { out[i] = fp16_to_f32(rd_u16(raw, i * 2)); }
        }
        GgmlType::BF16 => {
            for i in 0..n_elems { out[i] = bf16_to_f32(rd_u16(raw, i * 2)); }
        }
        GgmlType::Q8_0 => dequant_q8_0(raw, n_elems, &mut out),
        GgmlType::Q4_0 => dequant_q4_0(raw, n_elems, &mut out),
        GgmlType::Q2_K => dequant_q2_k(raw, n_elems, &mut out),
        GgmlType::Q4_K => dequant_q4_k(raw, n_elems, &mut out),
        GgmlType::Q5_K => dequant_q5_k(raw, n_elems, &mut out),
        GgmlType::Q6_K => dequant_q6_k(raw, n_elems, &mut out),
        GgmlType::Q3_K => dequant_q3_k(raw, n_elems, &mut out),
        GgmlType::IQ4_XS => dequant_iq4_xs(raw, n_elems, &mut out),
        GgmlType::IQ3_S => dequant_iq3_s(raw, n_elems, &mut out),
        GgmlType::NVFP4 => dequant_nvfp4(raw, n_elems, &mut out),
        other => panic!("dequantize not implemented for {other:?}"),
    }
    out
}

// ============================ Q2_K ============================

/// block_q2_K (QK_K=256): { u8 scales[16]; u8 qs[64]; fp16 d; fp16 dmin } => 84 bytes.
/// Port of ggml's `dequantize_row_q2_K`. Each 16-value group has a 4-bit scale and 4-bit
/// minimum; four groups share each 32-byte quant plane through 2-bit lanes.
fn dequant_q2_k(raw: &[u8], n: usize, out: &mut [f32]) {
    const QK_K: usize = 256;
    const BYTES: usize = 84;
    let nb = n / QK_K;
    for i in 0..nb {
        let base = i * BYTES;
        let scales = &raw[base..base + 16];
        let qs = &raw[base + 16..base + 80];
        let d = fp16_to_f32(rd_u16(raw, base + 80));
        let dmin = fp16_to_f32(rd_u16(raw, base + 82));
        for j in 0..QK_K {
            let group = j / 16;
            let half = j / 128;
            let within = j % 128;
            let shift = 2 * (within / 32);
            let q = (qs[half * 32 + within % 32] >> shift) & 3;
            let sc = scales[group];
            out[i * QK_K + j] = d * (sc & 0x0f) as f32 * q as f32
                - dmin * (sc >> 4) as f32;
        }
    }
}

/// block_q8_0: { fp16 d; int8 qs[32] } => 34 bytes / 32 elems. y = d * qs.
/// Q4_0: 18B per 32 elems — fp16 d + 16 nibble bytes; elem i (<16) = low nibble of byte i,
/// elem i (>=16) = high nibble of byte i-16; value = d * (q - 8).
fn dequant_q4_0(raw: &[u8], n_elems: usize, out: &mut [f32]) {
    let nb = n_elems / 32;
    for b in 0..nb {
        let blk = &raw[b * 18..(b + 1) * 18];
        let d = fp16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        for i in 0..16 {
            let byte = blk[2 + i];
            out[b * 32 + i]      = d * ((byte & 0x0F) as i32 - 8) as f32;
            out[b * 32 + i + 16] = d * ((byte >> 4)   as i32 - 8) as f32;
        }
    }
}

fn dequant_q8_0(raw: &[u8], n: usize, out: &mut [f32]) {
    const QK: usize = 32;
    const BYTES: usize = 34;
    let nb = n / QK;
    for i in 0..nb {
        let base = i * BYTES;
        let d = fp16_to_f32(rd_u16(raw, base));
        for j in 0..QK {
            let q = raw[base + 2 + j] as i8;
            out[i * QK + j] = d * q as f32;
        }
    }
}

/// 6-bit packed scale/min extraction (ggml get_scale_min_k4).
#[inline]
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let d = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    }
}

/// block_q4_K (QK_K=256): { fp16 d; fp16 dmin; u8 scales[12]; u8 qs[128] } => 144 bytes / 256 elems.
fn dequant_q4_k(raw: &[u8], n: usize, out: &mut [f32]) {
    const QK_K: usize = 256;
    const BYTES: usize = 144;
    let nb = n / QK_K;
    for i in 0..nb {
        let base = i * BYTES;
        let d = fp16_to_f32(rd_u16(raw, base));
        let dmin = fp16_to_f32(rd_u16(raw, base + 2));
        let scales = &raw[base + 4..base + 4 + 12];
        let q = &raw[base + 16..base + 16 + 128];
        let mut y = i * QK_K;
        let mut is = 0usize;
        let mut qoff = 0usize;
        // 4 groups of 64 elems
        for _ in 0..(QK_K / 64) {
            let (sc1, m1b) = get_scale_min_k4(is, scales);
            let d1 = d * sc1 as f32; let m1 = dmin * m1b as f32;
            let (sc2, m2b) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc2 as f32; let m2 = dmin * m2b as f32;
            for l in 0..32 { out[y] = d1 * (q[qoff + l] & 0xF) as f32 - m1; y += 1; }
            for l in 0..32 { out[y] = d2 * (q[qoff + l] >> 4) as f32 - m2; y += 1; }
            qoff += 32; is += 2;
        }
    }
}

/// block_q6_K (QK_K=256): { u8 ql[128]; u8 qh[64]; i8 scales[16]; fp16 d } => 210 bytes / 256 elems.
/// ggml layout: ql then qh then scales then d at the end.
fn dequant_q6_k(raw: &[u8], n: usize, out: &mut [f32]) {
    const QK_K: usize = 256;
    const BYTES: usize = 210;
    let nb = n / QK_K;
    for i in 0..nb {
        let base = i * BYTES;
        let ql = &raw[base..base + 128];
        let qh = &raw[base + 128..base + 128 + 64];
        let scales = &raw[base + 192..base + 192 + 16]; // i8
        let d = fp16_to_f32(rd_u16(raw, base + 208));
        // ggml dequantize_row_q6_K: two halves of 128, each processes 32-wide sub-blocks.
        let y0 = i * QK_K;
        for n2 in 0..2 {
            let ql_off = n2 * 64;
            let qh_off = n2 * 32;
            let sc_off = n2 * 8;
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[ql_off + l] & 0xF) as i32 | (((qh[qh_off + l] >> 0) & 3) as i32) << 4) - 32;
                let q2 = ((ql[ql_off + l + 32] & 0xF) as i32 | (((qh[qh_off + l] >> 2) & 3) as i32) << 4) - 32;
                let q3 = ((ql[ql_off + l] >> 4) as i32 | (((qh[qh_off + l] >> 4) & 3) as i32) << 4) - 32;
                let q4 = ((ql[ql_off + l + 32] >> 4) as i32 | (((qh[qh_off + l] >> 6) & 3) as i32) << 4) - 32;
                let y = y0 + n2 * 128 + l;
                out[y]      = d * scales[sc_off + is] as i8 as f32 * q1 as f32;
                out[y + 32] = d * scales[sc_off + is + 2] as i8 as f32 * q2 as f32;
                out[y + 64] = d * scales[sc_off + is + 4] as i8 as f32 * q3 as f32;
                out[y + 96] = d * scales[sc_off + is + 6] as i8 as f32 * q4 as f32;
            }
        }
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp16_known_values() {
        assert_eq!(fp16_to_f32(0x0000), 0.0);
        assert_eq!(fp16_to_f32(0x3c00), 1.0);   // 1.0
        assert_eq!(fp16_to_f32(0x4000), 2.0);   // 2.0
        assert_eq!(fp16_to_f32(0xc000), -2.0);  // -2.0
        assert_eq!(fp16_to_f32(0x3800), 0.5);   // 0.5
    }

    #[test]
    fn bf16_known_values() {
        assert_eq!(bf16_to_f32(0x3f80), 1.0);
        assert_eq!(bf16_to_f32(0x4000), 2.0);
        assert_eq!(bf16_to_f32(0xbf80), -1.0);
    }

    #[test]
    fn bf16_dequant_on_use_kernel_contract() {
        // BW24_FULL_PREC FloatBf16 dequant-on-use (kernels.cu `bf16_to_f32`) computes
        //   out = __uint_as_float((uint)in << 16)
        // which MUST be bit-identical to this host reference (the load-time bf16->f32 dequant the
        // plain Float path already does). If these ever diverge, the full-precision oracle path
        // stops being exact. Pin the formula equivalence here (host-only; the device run is a
        // GPU gate, deferred).
        for b in [0u16, 0x3f80, 0x4000, 0xbf80, 0x7f7f, 0x8000, 0x0001, 0xffff, 0x3c00] {
            let host = bf16_to_f32(b);
            let kernel_formula = f32::from_bits((b as u32) << 16);
            assert_eq!(host.to_bits(), kernel_formula.to_bits(), "bf16 {b:#06x} host vs kernel");
        }
    }

    #[test]
    fn q8_0_roundtrip_simple() {
        // one block: d=0.5 (fp16 0x3800), qs = 0,2,4,... => values 0,1,2,...
        let mut raw = vec![0u8; 34];
        raw[0] = 0x00; raw[1] = 0x38; // d = 0.5
        for j in 0..32 { raw[2 + j] = (j as i8 * 2) as u8; }
        let out = dequantize(GgmlType::Q8_0, &raw, 32);
        for j in 0..32 { assert!((out[j] - (j as f32)).abs() < 1e-4, "j={j} got {}", out[j]); }
    }

    #[test]
    fn q2k_planes_scales_and_minimum() {
        // scale=1, minimum multiplier=2, d=1, dmin=0.5 -> value = q - 1.
        // Each repeated 0b11_10_01_00 byte makes the four 32-value planes q=0,1,2,3.
        let mut raw = vec![0u8; 84];
        raw[..16].fill(0x21);
        raw[16..80].fill(0b11_10_01_00);
        raw[80..82].copy_from_slice(&0x3c00u16.to_le_bytes());
        raw[82..84].copy_from_slice(&0x3800u16.to_le_bytes());
        let out = dequantize(GgmlType::Q2_K, &raw, 256);
        for half in 0..2 {
            for plane in 0..4 {
                for j in 0..32 {
                    assert_eq!(out[half * 128 + plane * 32 + j], plane as f32 - 1.0);
                }
            }
        }
    }

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
        // hmask all-0xFF -> hb=0 path (no -4); shift=0 reads low 2 bits -> w = q&3.
        // (the unpacked scale comes from the full 6-bit aux dance; this test only asserts finiteness.)
        let mut raw = vec![0u8; 110];
        for k in 0..32 { raw[k] = 0xFF; }              // hmask set -> no -4
        for k in 32..96 { raw[k] = 0b01_10_11_00; }     // qs
        for k in 96..108 { raw[k] = 0x21; }             // packed scales
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

    /// Oracle test: diff bw24 IQ3_S CPU dequant against ggml dequantize_row_iq3_s output
    /// on a REAL tensor (blk.0.ffn_gate_exps.weight from the 35B-MoE IQ4_XS gguf).
    /// The C++ harness (/tmp/iq3s_oracle) dumped /tmp/iq3s_raw.bin + /tmp/iq3s_ggml.bin.
    #[test]
    fn iq3s_vs_ggml_oracle() {
        let raw = match std::fs::read("/tmp/iq3s_raw.bin") {
            Ok(r) => r,
            Err(_) => { eprintln!("SKIP: /tmp/iq3s_raw.bin missing"); return; }
        };
        let oracle_bytes = std::fs::read("/tmp/iq3s_ggml.bin").expect("ggml oracle");
        let n = oracle_bytes.len() / 4;
        let oracle: Vec<f32> = oracle_bytes.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        assert_eq!(raw.len(), (n / 256) * 110, "raw byte count");
        let got = dequantize(GgmlType::IQ3_S, &raw, n);
        let mut max_abs = 0f32;
        let mut nmis = 0usize;
        for i in 0..n {
            let d = (got[i] - oracle[i]).abs();
            if d > max_abs { max_abs = d; }
            if d > 1e-4 { nmis += 1; if nmis <= 5 { eprintln!("mismatch i={i} got={} oracle={}", got[i], oracle[i]); } }
        }
        eprintln!("IQ3_S oracle: n={n} max_abs_diff={max_abs} mismatches={nmis}");
        assert_eq!(nmis, 0, "IQ3_S CPU dequant diverges from ggml");
        assert!(max_abs < 1e-4, "max abs diff {max_abs} exceeds 1e-4");
    }
}
