//! clc_mmq_bench: CLC work-stealing vs static xy-tiling on the Q4_0 MMQ prefill GEMM
//! (perf-frontier lever #1, research/clc-mmq-20260802/). Kernel-only timing via the
//! GEMM-only FFI entry (activation quantized ONCE per shape outside the timed region);
//! the two arms are forced deterministically with memra_mmq_q4_0_set_clc and interleaved
//! WITHIN each session (N sessions => N interleaved medians, same-session comparison —
//! the cross-run clock-drift law). Every shape also bit-compares CLC vs static output.
//!
//! Shapes: the q9-class prefill GEMMs (4096/8192 qkv, 4096/12288 gate-up, 12288/4096 down,
//! 4096/4096 attn-gate) at T=512/1736, plus poor-wave-quantization probes (tiles % 82 far
//! from 0 — 84 tiles = 1.02 waves is the worst case the +8-15% is priced on) and the real
//! gemma-12b q4_0 tensors when the GGUF is present.
//!
//! usage: clc-mmq-bench [gguf] [reps=30] [sessions=3]   (prints a table + JSONL rows)
use memra_engine::Engine;
use memra_engine::mmq_ffi::*;
use cudarc::driver::{DevicePtr, DevicePtrMut};
use std::time::Instant;

fn pr(i: usize) -> f32 { ((i.wrapping_mul(40503) % 1000) as f32) / 1000.0 - 0.5 }
fn hb(i: usize) -> u8 { ((i.wrapping_mul(2654435761u32 as usize)) >> 13) as u8 }

/// Synthetic q4_0 rows: fp16 d = small positive, deterministic nibble bytes.
fn synth_q4_0(in_f: usize, out_f: usize) -> Vec<u8> {
    let nblk = in_f / 32 * out_f;
    let mut raw = vec![0u8; nblk * 18];
    for (bi, b) in raw.chunks_mut(18).enumerate() {
        b[0] = 0x00; b[1] = 0x2C;                       // d = f16 ~0.0156
        for k in 0..16 { b[2 + k] = hb(bi * 17 + k); }
    }
    raw
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gguf = std::env::args().nth(1);
    let reps: usize = std::env::args().nth(2).and_then(|v| v.parse().ok()).unwrap_or(30);
    let sessions: usize = std::env::args().nth(3).and_then(|v| v.parse().ok()).unwrap_or(3);
    // Third arm: the incumbent stream-k (band-class fold order) — force its kernel
    // deterministically, bypassing the per-process timing autotune (the #23 coin).
    // Must be set before the first gemm_sk call (the .cu reads it once, statically).
    unsafe { std::env::set_var("MEMRA_MMQ_SK_FORM", "sk"); }
    let e = Engine::new(0)?;
    println!("GPU: {}  reps={reps} sessions={sessions}", e.ctx().name()?);
    let clc_avail = unsafe { memra_mmq_q4_0_set_clc(-1) } == 1;
    if !clc_avail {
        println!("CLC kernel not compiled in (pre-SM100 build) — nothing to measure.");
        return Ok(());
    }
    let nsm = 82usize; // 5090 laptop; informational only (wave math printed per shape)

    // (label, in_f, out_f, T, synthetic?) — real tensors appended below when gguf present.
    let mut shapes: Vec<(String, usize, usize, usize, Option<Vec<u8>>)> = Vec::new();
    for (lbl, inf, outf) in [("q9-qkv", 4096usize, 8192usize),
                             ("q9-gateup", 4096, 12288),
                             ("q9-down", 12288, 4096),
                             ("q9-attngate", 4096, 4096)] {
        for t in [512usize, 1736] {
            shapes.push((format!("{lbl}"), inf, outf, t, None));
        }
    }
    // poor-wave-quantization probes: nty*ntx just past a wave boundary.
    // 84 tiles = 1.02 waves (eff 51%) — the priced worst case; 336 = 4.10 waves (eff 82%).
    shapes.push(("wavequant-84t".into(), 4096, 10752, 128, None));   // nty=84, ntx=1
    shapes.push(("wavequant-336t".into(), 4096, 10752, 512, None));  // nty=84, ntx=4
    shapes.push(("wavequant-168t".into(), 4096, 10752, 256, None));  // nty=84, ntx=2
    // sub-wave controls (tiles < 82 SMs -> ZERO steals possible): any CLC-vs-static delta
    // here is pure machinery overhead (fences/mbarrier/loop), not steal cost.
    shapes.push(("subwave-32t".into(), 4096, 4096, 128, None));      // nty=32, ntx=1
    shapes.push(("subwave-64t".into(), 4096, 8192, 128, None));      // nty=64, ntx=1
    if let Some(p) = &gguf {
        use memra_gguf::{GgufFile, GgmlType};
        let g = GgufFile::open(p)?;
        for tname in ["blk.0.attn_q.weight", "blk.0.ffn_gate.weight", "blk.0.ffn_down.weight"] {
            if let Some(t) = g.find(tname).filter(|t| t.ggml_type == GgmlType::Q4_0) {
                let inf = t.ne[0] as usize; let outf = t.ne[1] as usize;
                let raw = g.tensor_data(t).to_vec();
                for t in [512usize, 1736] {
                    shapes.push((format!("g12-{}", tname.trim_start_matches("blk.0.").trim_end_matches(".weight")),
                                 inf, outf, t, Some(raw.clone())));
                }
            }
        }
    }

    println!("{:<16} {:>6} {:>6} {:>5} {:>6} {:>6} {:>8} | {:>9} {:>9} {:>9} | {:>8} {:>8} | bits",
             "shape", "in_f", "out_f", "T", "tiles", "waves", "waveeff",
             "static_us", "clc_us", "sk_us", "clc", "sk");
    for (lbl, in_f, out_f, t, real) in shapes {
        let raw = real.unwrap_or_else(|| synth_q4_0(in_f, out_f));
        let wd = e.htod_bytes(&raw)?;
        let x: Vec<f32> = (0..t * in_f).map(|i| pr(i + 29) * 0.1).collect();
        let xd = e.htod(&x)?;
        let act_bytes = unsafe { memra_mmq_q4_0_act_bytes(in_f as i32, t as i32) };
        let mut scratch = e.htod_bytes(&vec![0u8; act_bytes])?;
        let fixup_bytes = unsafe { memra_mmq_q4_0_fixup_bytes() };
        let mut fixup = e.htod_bytes(&vec![0u8; fixup_bytes])?;
        let mut y = e.zeros(t * out_f)?;
        let stream = e.stream();
        let cu = stream.cu_stream() as *mut core::ffi::c_void;
        let (x_p, _gx) = xd.device_ptr(&stream);
        // quantize once, outside the timed region (the quantize-once production seam).
        {
            let (s_p, _gs) = scratch.device_ptr_mut(&stream);
            let rc = unsafe { memra_mmq_q4_0_quant_act(x_p as *const f32,
                s_p as *mut core::ffi::c_void, in_f as i32, t as i32, cu) };
            assert_eq!(rc, 0, "quant_act rc={rc}");
        }
        let (w_p, _gw) = wd.device_ptr(&stream);
        let (s_p, _gs) = scratch.device_ptr(&stream);
        // arm: 0 = static xy-tiling, 1 = CLC work-stealing, 2 = stream-k (forced via
        // MEMRA_MMQ_SK_FORM=sk at process start; band-class numerics — informational arm).
        let gemm = |arm: i32, y: &mut cudarc::driver::CudaSlice<f32>,
                    fixup: &mut cudarc::driver::CudaSlice<u8>| -> i32 {
            let stream = e.stream();
            let (y_p, _gy) = y.device_ptr_mut(&stream);
            if arm == 2 {
                unsafe { memra_mmq_q4_0_set_clc(0) };
                let (f_p, _gf) = fixup.device_ptr_mut(&stream);
                unsafe { memra_mmq_q4_0_gemm_sk(w_p as *const core::ffi::c_void,
                    s_p as *const core::ffi::c_void, y_p as *mut f32,
                    f_p as *mut core::ffi::c_void,
                    in_f as i32, out_f as i32, t as i32, cu, 0) }
            } else {
                unsafe { memra_mmq_q4_0_set_clc(arm) };
                unsafe { memra_mmq_q4_0_gemm(w_p as *const core::ffi::c_void,
                    s_p as *const core::ffi::c_void, y_p as *mut f32,
                    in_f as i32, out_f as i32, t as i32, cu, 0) }
            }
        };
        // bit-identity: static vs CLC output (belt-and-braces on top of kernel-check).
        assert_eq!(gemm(0, &mut y, &mut fixup), 0);
        stream.synchronize()?;
        let y_static = e.dtoh(&y)?;
        assert_eq!(gemm(1, &mut y, &mut fixup), 0);
        stream.synchronize()?;
        let y_clc = e.dtoh(&y)?;
        let nbad = y_static.iter().zip(y_clc.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        // warmup all arms
        for arm in [0i32, 1, 2] {
            for _ in 0..10 { assert_eq!(gemm(arm, &mut y, &mut fixup), 0); }
        }
        stream.synchronize()?;
        // N sessions, arms interleaved within each session.
        let mut med = [Vec::new(), Vec::new(), Vec::new()];
        for _s in 0..sessions {
            for arm in [0i32, 1, 2] {
                stream.synchronize()?;
                let t0 = Instant::now();
                for _ in 0..reps { assert_eq!(gemm(arm, &mut y, &mut fixup), 0); }
                stream.synchronize()?;
                let us = t0.elapsed().as_secs_f64() * 1e6 / reps as f64;
                med[arm as usize].push(us);
            }
        }
        unsafe { memra_mmq_q4_0_set_clc(-1) };
        let ms = median(&mut med[0]);
        let mc = median(&mut med[1]);
        let mk = median(&mut med[2]);
        let nty = out_f.div_ceil(128); let ntx = t.div_ceil(128);
        let tiles = nty * ntx;
        let waves = tiles as f64 / nsm as f64;
        let waveeff = tiles as f64 / (tiles.div_ceil(nsm) * nsm) as f64;
        println!("{:<16} {:>6} {:>6} {:>5} {:>6} {:>6.2} {:>7.1}% | {:>9.1} {:>9.1} {:>9.1} | {:>7.3}x {:>7.3}x | {}",
                 lbl, in_f, out_f, t, tiles, waves, waveeff * 100.0, ms, mc, mk, ms / mc, ms / mk,
                 if nbad == 0 { "IDENTICAL".to_string() } else { format!("MISMATCH {nbad}") });
        println!("JSONL {{\"lane\":\"clc-mmq\",\"shape\":\"{lbl}\",\"in_f\":{in_f},\"out_f\":{out_f},\"T\":{t},\"tiles\":{tiles},\"waves\":{waves:.3},\"wave_eff\":{waveeff:.4},\"static_us\":{ms:.2},\"clc_us\":{mc:.2},\"sk_us\":{mk:.2},\"clc_ratio\":{:.4},\"sk_ratio\":{:.4},\"bit_identical\":{},\"reps\":{reps},\"sessions\":{sessions},\"n_static\":{:?},\"n_clc\":{:?},\"n_sk\":{:?}}}",
                 ms / mc, ms / mk, nbad == 0,
                 med[0].iter().map(|v| (v * 100.0).round() / 100.0).collect::<Vec<_>>(),
                 med[1].iter().map(|v| (v * 100.0).round() / 100.0).collect::<Vec<_>>(),
                 med[2].iter().map(|v| (v * 100.0).round() / 100.0).collect::<Vec<_>>());
    }
    Ok(())
}
