//! Microbench: qmatvec_mmvq wall-time at m=1,2,4,8 on a real quantized weight row.
//! PROVES the weight-reuse thesis — the current MMVQ uses grid.y=m (each token re-reads the full
//! weight, zero reuse). If m=4 wall-time ~= 4x m=1, there is NO reuse and a weight-tile-resident
//! batched matvec would serve ~4 tokens for ~1 token's HBM traffic (the m=2-4 concurrent-decode win).
//! If m=4 ~= m=1, weight already amortizes and the win is small. Measures which.
//! Supports the 5 daily-hot dtypes (NVFP4 + Q8_0/Q4_K/Q5_K/Q6_K — the k-quant batched family);
//! MSWEEP_RP stays NVFP4-only (the A6 split-plane layout has no k-quant port).
use bw24_engine::Engine;
use bw24_gguf::{GgufFile, GgmlType};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: mvq-msweep <gguf>");
    let e = Engine::new(0)?;
    let g = GgufFile::open(&path)?;
    // pick the big FFN-down weight (largest single matvec on the decode path).
    // MSWEEP_TENSOR overrides (e.g. blk.0.ffn_gate.weight for the tall out_f/short in_f shape).
    let tname = std::env::var("MSWEEP_TENSOR").unwrap_or_else(|_| "blk.0.ffn_down.weight".into());
    let t = g.find(&tname)
        .or_else(|| g.find("blk.0.ffn_gate.weight"))
        .expect("no ffn weight");
    let in_f = t.ne[0] as usize;
    let out_f = t.ne[1] as usize;
    let raw = g.tensor_data(t);
    let row_bytes = raw.len() / out_f;
    let qtype = match t.ggml_type {
        GgmlType::NVFP4 => bw24_engine::QT_NVFP4,
        GgmlType::Q8_0 => bw24_engine::QT_Q8_0,
        GgmlType::Q4_K => bw24_engine::QT_Q4_K,
        GgmlType::Q5_K => bw24_engine::QT_Q5_K,
        GgmlType::Q6_K => bw24_engine::QT_Q6_K,
        other => panic!("unsupported msweep dtype {other:?}"),
    };
    // MSWEEP_COPIES=N (default 1): rotate the launch across N device copies of the weight so each
    // launch reads DRAM-COLD bytes, like the real trunk sweep (each verify launch reads a DIFFERENT
    // ~50MB tensor; L2 is 64MB). The default 1-copy loop is L2/DRAM-mixed and OVERSTATES wins that
    // don't transfer (mmvq-b4-clamp-NEGATIVE lesson) — use N>=8 for transfer-grade numbers.
    let copies: usize = std::env::var("MSWEEP_COPIES").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    // MSWEEP_RP=1 (A6 split-plane repack prototype): host-repack the GGUF 36B blocks into
    // [quant plane out_f x nsb64 x 32B][scale plane out_f x nsb64 x 4B] and feed the rp kernels
    // (BW24_MMVQ_BV=rp|rpr2|rpr2w8). Roundtrip gate: un-repack must reproduce the original bytes.
    let rp = std::env::var("MSWEEP_RP").is_ok();
    assert!(!rp || qtype == bw24_engine::QT_NVFP4, "MSWEEP_RP is NVFP4-only");
    let raw_owned: Vec<u8>;
    let raw = if rp {
        let nsb64 = in_f / 64;
        assert_eq!(row_bytes, nsb64 * 36, "unexpected NVFP4 row_bytes");
        let qplane = out_f * nsb64 * 32;
        let mut rpb = vec![0u8; raw.len()];
        for o in 0..out_f {
            for s in 0..nsb64 {
                let src = &raw[o * row_bytes + s * 36..o * row_bytes + s * 36 + 36];
                rpb[qplane + o * nsb64 * 4 + s * 4..qplane + o * nsb64 * 4 + s * 4 + 4]
                    .copy_from_slice(&src[0..4]);
                rpb[o * nsb64 * 32 + s * 32..o * nsb64 * 32 + s * 32 + 32]
                    .copy_from_slice(&src[4..36]);
            }
        }
        // roundtrip gate
        let mut back = vec![0u8; raw.len()];
        for o in 0..out_f {
            for s in 0..nsb64 {
                back[o * row_bytes + s * 36..o * row_bytes + s * 36 + 4]
                    .copy_from_slice(&rpb[qplane + o * nsb64 * 4 + s * 4..qplane + o * nsb64 * 4 + s * 4 + 4]);
                back[o * row_bytes + s * 36 + 4..o * row_bytes + s * 36 + 36]
                    .copy_from_slice(&rpb[o * nsb64 * 32 + s * 32..o * nsb64 * 32 + s * 32 + 32]);
            }
        }
        let rt_bad = raw.iter().zip(&back).filter(|(a, b)| a != b).count();
        println!("repack roundtrip: {} mismatched bytes of {}", rt_bad, raw.len());
        assert_eq!(rt_bad, 0, "repack roundtrip FAILED");
        raw_owned = rpb;
        &raw_owned[..]
    } else { raw };
    // the grid.y=m reference + rp bit-identity check always read the ORIGINAL layout.
    let wd_orig = if rp { Some(e.htod_bytes(g.tensor_data(t))?) } else { None };
    let wds: Vec<_> = (0..copies).map(|_| e.htod_bytes(raw)).collect::<Result<_, _>>()?;
    let bv = std::env::var("BW24_MMVQ_BV").unwrap_or_else(|_| "auto".into());
    println!("weight {tname} [{:?}] in_f={in_f} out_f={out_f} row_bytes={row_bytes} \
              copies={copies} variant={bv}", t.ggml_type);
    let iters = if copies > 1 { 800 } else { 2000 };
    if std::env::var("MSWEEP_BATCHED_ONLY").is_err() {
        println!("--- grid.y=m (current, no weight reuse) ---");
        for m in [1usize, 2, 4, 8] {
            let x: Vec<f32> = (0..m * in_f).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();
            let xd = e.htod(&x)?;
            for _ in 0..50 { let _ = e.qmatvec_mmvq_raw(&wds[0], &xd, m, in_f, out_f, qtype, row_bytes, false)?; }
            e.stream().synchronize()?;
            let t0 = Instant::now();
            for i in 0..iters { let _ = e.qmatvec_mmvq_raw(&wds[i % copies], &xd, m, in_f, out_f, qtype, row_bytes, false)?; }
            e.stream().synchronize()?;
            let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
            println!("  m={m}: {us:.2} us/call  ({:.2} us/token)", us / m as f64);
        }
    }
    println!("--- BATCHED weight-tile-resident (1 weight read serves m tokens) ---");
    // MSWEEP_M1=1 adds an m=1 row through the batched launcher (the A6 m=1-consumer probe:
    // compares the repacked kernel at m=1 vs the m=1 mmvq reference path).
    let mut cells = vec![(2usize, 2usize), (3, 4), (4, 4)];
    if std::env::var("MSWEEP_M1").is_ok() { cells.insert(0, (1, 2)); }
    for (m, mcols) in cells {
        let x: Vec<f32> = (0..m * in_f).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();
        let xd = e.htod(&x)?;
        // bit-identity vs grid.y=m reference (original layout when the sweep buffers are repacked)
        let wref = wd_orig.as_ref().unwrap_or(&wds[0]);
        let r_ref = e.dtoh(&e.qmatvec_mmvq_raw(wref, &xd, m, in_f, out_f, qtype, row_bytes, false)?)?;
        let r_bat = e.dtoh(&e.qmatvec_batched_raw(&wds[0], &xd, m, in_f, out_f, qtype, row_bytes, mcols, rp)?)?;
        let bad = r_ref.iter().zip(&r_bat).filter(|(a, b)| (*a - *b).abs() > 1e-3).count();
        let bit = r_ref.iter().zip(&r_bat).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        for _ in 0..50 { let _ = e.qmatvec_batched_raw(&wds[0], &xd, m, in_f, out_f, qtype, row_bytes, mcols, rp)?; }
        e.stream().synchronize()?;
        let t0 = Instant::now();
        for i in 0..iters { let _ = e.qmatvec_batched_raw(&wds[i % copies], &xd, m, in_f, out_f, qtype, row_bytes, mcols, rp)?; }
        e.stream().synchronize()?;
        let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
        println!("  m={m} (b{mcols}): {us:.2} us/call  ({:.2} us/token)  bit-bad={bad} bit-exact-bad={bit}", us / m as f64);
    }
    Ok(())
}
