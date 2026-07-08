//! q4issue lane micro-bench: qmatvec_q4_K_mmvq (baseline) vs qmatvec_q4_K_mmvq_v (issue-reduced,
//! BW24_Q4V=1) on synthetic q4_K weights at the real decode shapes:
//!   in_f=5120 out_f=17408 (27B MLP)   and   in_f=4096 out_f=4096 (9B trunk).
//! 1) BIT-COMPARE: full out_f, baseline vs v — must be 0 mismatched bit patterns.
//! 2) TIME: N launches through the same qmatvec_mmvq path, wall-clock over [sync..sync].
//!    Q4V_COPIES device copies (default 8) rotate the weight so launches read DRAM-cold bytes
//!    (the mmvq-b4-clamp lesson: a 1-copy loop is L2-resident and overstates wins).
//! Requires BW24_MMVQ=1 in the env (kernel-family law); BW24_Q4V is toggled in-process.
use bw24_engine::Engine;
use std::time::Instant;

fn pr(i: usize) -> f32 {
    let x = (i.wrapping_mul(2654435761) ^ 0x9E3779B9) as u32;
    ((x >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    assert!(std::env::var("BW24_MMVQ").is_ok(), "run with BW24_MMVQ=1");
    let e = Engine::new(0)?;
    println!("GPU: {}", e.ctx().name()?);
    let copies: usize = std::env::var("Q4V_COPIES").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    let iters: usize = std::env::var("Q4V_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(200);
    let mut fails = 0usize;
    // real q4_K decode shapes: the 9B census (23x 4096->4096, 15x 4096->8192, 4x 4096->12288,
    // 1x 12288->4096) + the 27B Q4_K_M MLP (5120->17408).
    for (in_f, out_f) in [(4096usize, 4096usize), (4096, 8192), (4096, 12288), (12288, 4096),
                          (5120, 17408)] {
        let nsb = in_f / 256; // superblocks per row
        let row_bytes = nsb * 144;
        // synth q4_K rows: random qs/scales; d/dmin halfs pinned to a sane exponent (no inf/nan).
        let mut raw = vec![0u8; out_f * row_bytes];
        for (i, b) in raw.iter_mut().enumerate() { *b = ((i.wrapping_mul(2654435761)) % 251) as u8; }
        for r in 0..out_f {
            for s in 0..nsb {
                let base = r * row_bytes + s * 144;
                let dh: u16 = 0x2c00 | ((r * 31 + s * 7) as u16 & 0x03ff);   // ~2^-4 range half
                let mh: u16 = 0x2800 | ((r * 17 + s * 13) as u16 & 0x03ff);
                raw[base..base + 2].copy_from_slice(&dh.to_le_bytes());
                raw[base + 2..base + 4].copy_from_slice(&mh.to_le_bytes());
            }
        }
        let wds: Vec<_> = (0..copies).map(|_| e.htod_bytes(&raw)).collect::<Result<Vec<_>, _>>()?;
        let x: Vec<f32> = (0..in_f).map(|i| pr(i + 7) * 0.1).collect();
        let xd = e.htod(&x)?;
        let (aq, ad) = e.quantize_q8_1(&xd, 1, in_f)?;

        // --- 1) bit-identity: baseline vs v on the same weight buffer ---
        unsafe { std::env::remove_var("BW24_Q4V"); }
        let y0 = e.dtoh(&e.qmatvec_mmvq(&wds[0], &aq, &ad, 1, in_f, out_f, bw24_engine::QT_Q4_K, row_bytes, 1.0, false)?)?;
        unsafe { std::env::set_var("BW24_Q4V", "2"); }
        let y1 = e.dtoh(&e.qmatvec_mmvq(&wds[0], &aq, &ad, 1, in_f, out_f, bw24_engine::QT_Q4_K, row_bytes, 1.0, false)?)?;
        let bad = y0.iter().zip(&y1).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        println!("[{in_f}x{out_f}] bit-compare base vs v: {bad}/{out_f} mismatched {}",
                 if bad == 0 { "OK" } else { fails += 1; "FAIL" });

        // --- 2) timing: N launches, weight rotated over `copies` device buffers, 3 reps
        //     interleaved base/v to expose run-to-run variance (perf-claim law) ---
        let wbytes = (out_f * row_bytes) as f64;
        for rep in 0..3 {
            for (label, q4v) in [("base", false), ("v   ", true)] {
                unsafe {
                    if q4v { std::env::set_var("BW24_Q4V", "2"); }
                    else { std::env::remove_var("BW24_Q4V"); }
                }
                for i in 0..30 { // warmup
                    let _ = e.qmatvec_mmvq(&wds[i % copies], &aq, &ad, 1, in_f, out_f,
                                           bw24_engine::QT_Q4_K, row_bytes, 1.0, false)?;
                }
                e.stream().synchronize()?;
                let t0 = Instant::now();
                for i in 0..iters {
                    let _ = e.qmatvec_mmvq(&wds[i % copies], &aq, &ad, 1, in_f, out_f,
                                           bw24_engine::QT_Q4_K, row_bytes, 1.0, false)?;
                }
                e.stream().synchronize()?;
                let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
                let gbs = wbytes / (us * 1e-6) / 1e9;
                println!("[{in_f}x{out_f}] rep{rep} {label}: {us:.2} us/call  {gbs:.0} GB/s (weight-stream)");
            }
        }
    }
    unsafe { std::env::remove_var("BW24_Q4V"); }
    if fails > 0 { std::process::exit(1); }
    println!("q4v-bench DONE");
    Ok(())
}
