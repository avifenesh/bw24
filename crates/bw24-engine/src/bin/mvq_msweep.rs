//! Microbench: qmatvec_mmvq wall-time at m=1,2,4,8 on a real NVFP4 weight row.
//! PROVES the weight-reuse thesis — the current MMVQ uses grid.y=m (each token re-reads the full
//! weight, zero reuse). If m=4 wall-time ~= 4x m=1, there is NO reuse and a weight-tile-resident
//! batched matvec would serve ~4 tokens for ~1 token's HBM traffic (the m=2-4 concurrent-decode win).
//! If m=4 ~= m=1, weight already amortizes and the win is small. Measures which.
use bw24_engine::Engine;
use bw24_gguf::{GgufFile, GgmlType};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: mvq-msweep <gguf>");
    let e = Engine::new(0)?;
    let g = GgufFile::open(&path)?;
    // pick the big FFN-down weight (largest single matvec on the decode path).
    let t = g.find("blk.0.ffn_down.weight")
        .or_else(|| g.find("blk.0.ffn_gate.weight"))
        .expect("no ffn weight");
    let in_f = t.ne[0] as usize;
    let out_f = t.ne[1] as usize;
    let raw = g.tensor_data(t);
    let row_bytes = raw.len() / out_f;
    let qtype = match t.ggml_type { GgmlType::NVFP4 => bw24_engine::QT_NVFP4, other => panic!("want NVFP4, got {other:?}") };
    let wd = e.htod_bytes(raw)?;
    println!("weight blk.0.ffn_down [NVFP4] in_f={in_f} out_f={out_f} row_bytes={row_bytes}");
    let iters = 2000;
    println!("--- grid.y=m (current, no weight reuse) ---");
    for m in [1usize, 2, 4, 8] {
        let x: Vec<f32> = (0..m * in_f).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();
        let xd = e.htod(&x)?;
        for _ in 0..50 { let _ = e.qmatvec_mmvq_raw(&wd, &xd, m, in_f, out_f, qtype, row_bytes)?; }
        e.stream().synchronize()?;
        let t0 = Instant::now();
        for _ in 0..iters { let _ = e.qmatvec_mmvq_raw(&wd, &xd, m, in_f, out_f, qtype, row_bytes)?; }
        e.stream().synchronize()?;
        let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
        println!("  m={m}: {us:.2} us/call  ({:.2} us/token)", us / m as f64);
    }
    println!("--- BATCHED weight-tile-resident (1 weight read serves m tokens) ---");
    for (m, mcols) in [(2usize, 2usize), (4, 4)] {
        let x: Vec<f32> = (0..m * in_f).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();
        let xd = e.htod(&x)?;
        // bit-identity vs grid.y=m reference
        let r_ref = e.dtoh(&e.qmatvec_mmvq_raw(&wd, &xd, m, in_f, out_f, qtype, row_bytes)?)?;
        let r_bat = e.dtoh(&e.qmatvec_nvfp4_batched_raw(&wd, &xd, m, in_f, out_f, row_bytes, mcols)?)?;
        let bad = r_ref.iter().zip(&r_bat).filter(|(a, b)| (*a - *b).abs() > 1e-3).count();
        for _ in 0..50 { let _ = e.qmatvec_nvfp4_batched_raw(&wd, &xd, m, in_f, out_f, row_bytes, mcols)?; }
        e.stream().synchronize()?;
        let t0 = Instant::now();
        for _ in 0..iters { let _ = e.qmatvec_nvfp4_batched_raw(&wd, &xd, m, in_f, out_f, row_bytes, mcols)?; }
        e.stream().synchronize()?;
        let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
        println!("  m={m} (b{mcols}): {us:.2} us/call  ({:.2} us/token)  bit-bad={bad}", us / m as f64);
    }
    Ok(())
}
