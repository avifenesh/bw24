//! q5issue lane micro-bench: qmatvec_q5_K_mmvq{,_mr2} reference vs the issue-reduced `_il`
//! twins (BW24_Q5K_ISSUE=1) on SYNTHETIC q5_K weights at the two real shapes:
//!   - 27B lm_head: in_f=5120 out_f=151936 (535MB row-major stream, DRAM-cold every launch)
//!   - 27B/9B trunk-class square: in_f=5120 out_f=5120 (18MB -> rotated over 8 copies so each
//!     launch reads L2-cold bytes, MSWEEP_COPIES-style)
//! For each (shape, mr) cell: bit-exactness of il vs reference (must be 0 mismatches), then
//! N-rep wall time + implied weight GB/s. Variants toggled in-process via BW24_Q5K_ISSUE /
//! BW24_MMVQ_MR (the dispatch reads env per call — house style).
use bw24_engine::Engine;
use std::time::Instant;

fn synth_q5k(out_f: usize, in_f: usize) -> Vec<u8> {
    let sb_per_row = in_f / 256;
    let row_bytes = sb_per_row * 176;
    let mut w = vec![0u8; out_f * row_bytes];
    // LCG byte fill; overwrite d/dmin with fixed small valid f16s so acc stays finite.
    let mut s: u32 = 0x9E3779B9;
    for b in w.iter_mut() {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        *b = (s >> 24) as u8;
    }
    for blk in w.chunks_exact_mut(176) {
        blk[0] = 0x00; blk[1] = 0x14; // d    = f16 0x1400 = 2^-10
        blk[2] = 0x00; blk[3] = 0x10; // dmin = f16 0x1000 = 2^-11
    }
    w
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let e = Engine::new(0)?;
    const QT: i32 = bw24_engine::QT_Q5_K;
    // dispatch requires BW24_MMVQ; single-threaded here, set_var is fine.
    unsafe { std::env::set_var("BW24_MMVQ", "1"); }
    let iters: usize = std::env::var("Q5B_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(200);
    // REAL q5_K shapes: 27B lm_head output.weight [5120,248320] (874MB -> DRAM-cold every
    // launch); 9B trunk attn_gate/attn_output-class [4096,4096] x21/tok + attn_qkv [4096,8192]
    // x7/tok (small -> rotate copies for L2-cold launches); FR-Spec trimmed draft head
    // [5120,32768] (the reason mr2 is the q5_K default — must not regress under the flag).
    for (in_f, out_f, copies, label) in [(5120usize, 248320usize, 1usize, "27B lm_head"),
                                         (4096, 4096, 16, "9B trunk 4096x4096"),
                                         (4096, 8192, 8, "9B trunk qkv 4096x8192"),
                                         (4096, 1024, 64, "9B trunk k/v 4096x1024"),
                                         (5120, 32768, 2, "frspec draft head 5120x32768")] {
        let row_bytes = (in_f / 256) * 176;
        let host = synth_q5k(out_f, in_f);
        let wds: Vec<_> = (0..copies).map(|_| e.htod_bytes(&host)).collect::<Result<_, _>>()?;
        let x: Vec<f32> = (0..in_f).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();
        let xd = e.htod(&x)?;
        let wbytes = out_f * row_bytes;
        println!("=== {label}: in_f={in_f} out_f={out_f} row_bytes={row_bytes} \
                  weight={:.1}MB copies={copies} iters={iters} ===", wbytes as f64 / 1e6);
        for mr in ["1", "2"] {
            unsafe { std::env::set_var("BW24_MMVQ_MR", mr); }
            // bit-exactness: il vs reference on the same bytes ("2" = force il at every shape).
            unsafe { std::env::set_var("BW24_Q5K_ISSUE", "0"); }
            let yref = e.dtoh(&e.qmatvec_mmvq_raw(&wds[0], &xd, 1, in_f, out_f, QT, row_bytes, false)?)?;
            unsafe { std::env::set_var("BW24_Q5K_ISSUE", "2"); }
            let yil = e.dtoh(&e.qmatvec_mmvq_raw(&wds[0], &xd, 1, in_f, out_f, QT, row_bytes, false)?)?;
            let bit_bad = yref.iter().zip(&yil).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            let finite = yref.iter().all(|v| v.is_finite());
            for (flag, tag) in [("0", "ref"), ("2", "il ")] {
                unsafe { std::env::set_var("BW24_Q5K_ISSUE", flag); }
                for _ in 0..30 { let _ = e.qmatvec_mmvq_raw(&wds[0], &xd, 1, in_f, out_f, QT, row_bytes, false)?; }
                e.stream().synchronize()?;
                let t0 = Instant::now();
                for i in 0..iters { let _ = e.qmatvec_mmvq_raw(&wds[i % copies], &xd, 1, in_f, out_f, QT, row_bytes, false)?; }
                e.stream().synchronize()?;
                let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
                let gbs = wbytes as f64 / (us * 1e-6) / 1e9;
                println!("  mr{mr} {tag}: {us:8.2} us/call  {gbs:6.1} GB/s weight-implied  \
                          bit-bad={bit_bad} finite={finite}");
            }
            assert_eq!(bit_bad, 0, "il variant NOT bit-identical at mr={mr} {label}");
        }
    }
    Ok(())
}
