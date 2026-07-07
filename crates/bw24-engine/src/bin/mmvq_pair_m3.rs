//! MMVQ-vs-batched pair check at M3 direct-import shapes (in_f 6144/4096, rp=true from
//! repack_modelopt_to_split) — the shapes kernel-check never gated (its source is GGUF 9B).
//! Synthetic NVFP4 GGUF-block bytes -> A6 repack -> m=1 MMVQ vs m=2..8 batched, bit-compare
//! column 0 (the decode-parity contract the M3 gate exercises).
use bw24_engine::Engine;
use bw24_engine::model::repack_nvfp4_split;
fn pr(i: usize) -> f32 { (((i * 2654435761) % 1000) as f32) / 500.0 - 1.0 }
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let e = Engine::new(0)?;
    // (in_f, out_f) M3 shapes: q [6144->8192], o [4096->6144], mlp gate [6144->12288], down [6144<-... 
    for (in_f, out_f) in [(6144usize, 8192usize), (4096, 6144), (6144, 12288), (12288, 6144)] {
        // synth GGUF NVFP4 rows: 36B blocks
        let nsb64 = in_f / 64;
        let row_bytes = nsb64 * 36;
        let mut raw = vec![0u8; out_f * row_bytes];
        for (i, b) in raw.iter_mut().enumerate() { *b = ((i * 2654435761usize) % 251) as u8; }
        // scale bytes: keep ue4m3 valid (avoid NaN codes 0x7f/0xff)
        for r in 0..out_f { for blk in 0..nsb64 {
            let base = r * row_bytes + blk * 36;
            for s in 0..4 { let v = raw[base + s]; raw[base + s] = if v == 0x7f || v == 0xff { 0x38 } else { v & 0x7e }; }
        }}
        let rpb = repack_nvfp4_split(&raw, out_f);
        let w = e.htod_bytes(&rpb)?;
        // m=1 MMVQ reference per column vs batched m
        for m in [2usize, 4, 8] {
            let x: Vec<f32> = (0..m * in_f).map(|i| pr(i + 7) * 0.1).collect();
            let xd = e.htod(&x)?;
            let (aq, ad) = e.quantize_q8_1(&xd, m, in_f)?;
            let yb = e.qmatvec_mmvq_batched(&w, &aq, &ad, m, in_f, out_f, 7 /*QT_NVFP4*/, row_bytes,
                                            if m <= 2 { 2 } else if m <= 4 { 4 } else { 8 }, 1.0, true)?;
            let hb = e.dtoh(&yb)?;
            let mut worst = 0usize;
            for col in 0..m {
                let xc: Vec<f32> = x[col * in_f..(col + 1) * in_f].to_vec();
                let xcd = e.htod(&xc)?;
                let y1 = e.qmatvec_mmvq_raw(&w, &xcd, 1, in_f, out_f, 7, row_bytes, true)?;
                let h1 = e.dtoh(&y1)?;
                let bad = h1.iter().zip(&hb[col * out_f..(col + 1) * out_f])
                    .filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                worst = worst.max(bad);
            }
            println!("in_f={in_f} out_f={out_f} m={m}: worst col bit-mismatch {worst}/{out_f} {}",
                     if worst == 0 { "OK" } else { "FAIL" });
        }
    }
    Ok(())
}
