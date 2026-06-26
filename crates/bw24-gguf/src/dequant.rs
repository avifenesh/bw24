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
        GgmlType::Q4_K => dequant_q4_k(raw, n_elems, &mut out),
        GgmlType::Q6_K => dequant_q6_k(raw, n_elems, &mut out),
        other => panic!("dequantize not implemented for {other:?}"),
    }
    out
}

/// block_q8_0: { fp16 d; int8 qs[32] } => 34 bytes / 32 elems. y = d * qs.
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
    fn q8_0_roundtrip_simple() {
        // one block: d=0.5 (fp16 0x3800), qs = 0,2,4,... => values 0,1,2,...
        let mut raw = vec![0u8; 34];
        raw[0] = 0x00; raw[1] = 0x38; // d = 0.5
        for j in 0..32 { raw[2 + j] = (j as i8 * 2) as u8; }
        let out = dequantize(GgmlType::Q8_0, &raw, 32);
        for j in 0..32 { assert!((out[j] - (j as f32)).abs() < 1e-4, "j={j} got {}", out[j]); }
    }
}
