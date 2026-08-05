//! gemv-e4m3-bench — THE Q2 MEASUREMENT of the FP8-ST v3 gate (lane/fp8-v3-gate, 2026-08-05).
//!
//! QUESTION. The ship path (ARM B') dequants FP8 checkpoints to a Q8_0 slab at load, so decode pays
//! Q8_0's 1.0625 B/weight (34 B per 32 weights: 32 int8 + one fp16 scale). Native e4m3 is exactly
//! 1.0 B/weight, i.e. a 5.88% smaller weight stream, and m=1 decode is weight-stream-bound. Is that
//! arithmetic realized as decode time?
//!
//! WHAT MADE THIS A BOUNDED WRITE. The native-e4m3 m=1 GEMV ALREADY EXISTS and already ships behind
//! MEMRA_ST_E4M3=1: `qmatvec_e4m3_mmvq` (cu/qmatvec.cu, body `e4m3_row_dot`) reads the raw checkpoint
//! e4m3 bytes as its weight stream — no dequant, row_bytes == in_f — against the same q8_1 activation
//! every fast decode path produces. Its correctness is already gated at m=1 by kernel-check (f64 CPU
//! e4m3 reference, plus grid.y=m and _b2/_b4/_b8 bit-parity arms). So NO new kernel was needed for
//! this question: what was missing, and all this bin adds, is the A/B PERF measurement. The kernel
//! shipped without one — its only prior evidence was end-to-end.
//!
//! THE COMPARISON. Same in_f/out_f, same m=1, same activation (both arms ride `qmatvec_mmvq_raw`,
//! which quantizes the SAME f32 x to q8_1 and launches the warp-per-row MMVQ for the given qtype):
//!   arm E4M3 : QT_F8_E4M3, row_bytes = in_f          -> out_f * in_f          bytes
//!   arm Q8_0 : QT_Q8_0,    row_bytes = in_f/32 * 34  -> out_f * in_f/32 * 34  bytes
//! `ratio = t_q8_0 / t_e4m3` (>1 means e4m3 is faster). The byte ratio is fixed at 1.0625, so a
//! bandwidth-bound pair should land near +6.25pp and an arithmetic-bound one below it — that gap is
//! the finding, because the two arms are NOT the same arithmetic: Q8_0 does 8 dp4a into s32 per
//! 32-block, e4m3 does 8 cvt + 16 fmaf in f32. This bin measures TIME; it makes no exactness claim
//! (kernel-check owns the e4m3 GEMV's numeric gate, and model-level equivalence between the two
//! CONTAINERS is v2's teacher-forced + NLL protocol, not a bit-identity question).
//!
//! DRAM-COLD DISCIPLINE. Decode re-reads the whole weight from HBM every tick, so an L2-resident
//! measurement would be a fiction. Each shape allocates `copies` independent weight buffers sized so
//! the rotation set is past L2 and rotates through them, so consecutive launches never re-read the
//! same bytes. Both arms rotate identically.
//!
//! PROTOCOL: warm up both arms, then iters x (E4M3 timed, Q8_0 timed) INTERLEAVED inside the loop so
//! both share one clock/thermal regime; median of per-iteration times. Run under
//! flock /tmp/gpu5090.lock. GPU temp is printed at entry and exit for the thermal-regime record.
//!
//! usage: gemv-e4m3-bench [iters] [27b|1p7b]          (default iters=200, 27b)

use memra_engine::Engine;
use std::time::Instant;

/// Raw e4m3 weight rows, [out_f, in_f] row-major, row stride == in_f. Magnitude 0x7F is the e4m3 NaN
/// code (hardware NaN, host convention 0.0), so it is excluded — a NaN would make the accumulator
/// path data-dependent and is refused by the real dispatch anyway.
fn synth_e4m3(out_f: usize, in_f: usize) -> Vec<u8> {
    let mut w = vec![0u8; out_f * in_f];
    let mut s: u32 = 0x1234_5678;
    for b in w.iter_mut() {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        let mag = ((s >> 16) & 0x7F) as u8;
        // never 0x7F (NaN); 0x30 is a benign mid-range substitute.
        let mag = if mag == 0x7F { 0x30 } else { mag };
        *b = mag | ((((s >> 8) & 1) as u8) << 7);
    }
    w
}

/// Raw ggml block_q8_0 weight rows: in_f/32 blocks per row, 34 B each (fp16 scale + 32 int8).
fn synth_q8_0(out_f: usize, in_f: usize) -> Vec<u8> {
    let nblk = in_f / 32;
    let mut w = vec![0u8; out_f * nblk * 34];
    let mut s: u32 = 0x9E37_79B9;
    for blk in w.chunks_exact_mut(34) {
        // d = f16 0x1400 = 2^-10 (fixed, small, valid — keeps acc finite; same trick as q5issue).
        blk[0] = 0x00;
        blk[1] = 0x14;
        for q in blk[2..].iter_mut() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *q = (s >> 24) as u8;
        }
    }
    w
}

fn gpu_temp() -> String {
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=temperature.gpu,clocks.sm", "--format=csv,noheader"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().replace('\n', " | "))
        .unwrap_or_else(|| "n/a".into())
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let iters: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(200);
    let set = std::env::args().nth(2).unwrap_or_else(|| "27b".to_string());

    let e = Engine::new(0)?;
    // The warp-per-row MMVQ dispatch reads this per call (house style); single-threaded here.
    unsafe {
        std::env::set_var("MEMRA_MMVQ", "1");
    }

    println!("GPU: {}  iters={iters}  shapes={set}  temp_in: {}", e.ctx().name()?, gpu_temp());
    println!("m=1 GEMV: native e4m3 (1.0 B/weight, qmatvec_e4m3_mmvq) vs Q8_0 MMVQ floor (1.0625 B/weight)");
    println!("DRAM-cold (rotated copies); interleaved e4m3,q8_0 per iter; median; ratio = t_q8_0/t_e4m3");
    println!(
        "{:<28} {:>4} {:>10} {:>10} {:>9} {:>9} {:>9} {:>9}",
        "shape in->out", "cp", "e4m3_us", "q8_0_us", "ratio", "delta_pp", "e4m3_GB/s", "q8_0_GB/s"
    );

    // The v2 shape sheet, verbatim.
    let shapes_27b: [(usize, usize, &str); 6] = [
        (5120, 12288, "q_proj"),
        (5120, 1024, "k/v_proj"),
        (6144, 5120, "o_proj"),
        (5120, 17408, "gate/up_proj"),
        (17408, 5120, "down_proj"),
        (5120, 5120, "square-ref"),
    ];
    let shapes_1p7b: [(usize, usize, &str); 5] = [
        (2048, 2048, "q_proj"),
        (2048, 1024, "k/v_proj"),
        (2048, 2048, "o_proj"),
        (2048, 6144, "gate/up_proj"),
        (6144, 2048, "down_proj"),
    ];
    let shapes: Vec<(usize, usize, &str)> = if set == "1p7b" {
        shapes_1p7b.to_vec()
    } else {
        shapes_27b.to_vec()
    };

    let mut sum_ln = 0.0f64;
    let mut n = 0usize;

    for (in_f, out_f, label) in shapes {
        let rb_e4m3 = in_f;
        let rb_q8_0 = (in_f / 32) * 34;
        let wb_e4m3 = out_f * rb_e4m3;
        let wb_q8_0 = out_f * rb_q8_0;

        // Enough copies that the rotation set is past L2 by a wide margin, capped so the pair stays
        // well inside VRAM alongside the sibling lane's allocation.
        let copies = (768_000_000usize / wb_q8_0).clamp(1, 64);

        let h_e4m3 = synth_e4m3(out_f, in_f);
        let h_q8_0 = synth_q8_0(out_f, in_f);
        let d_e4m3: Vec<_> = (0..copies)
            .map(|_| e.htod_bytes(&h_e4m3))
            .collect::<Result<_, _>>()?;
        let d_q8_0: Vec<_> = (0..copies)
            .map(|_| e.htod_bytes(&h_q8_0))
            .collect::<Result<_, _>>()?;
        drop(h_e4m3);
        drop(h_q8_0);

        let x: Vec<f32> = (0..in_f).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();
        let xd = e.htod(&x)?;

        // warmup both arms
        for c in 0..copies.min(4) {
            let _ = e.qmatvec_mmvq_raw(&d_e4m3[c], &xd, 1, in_f, out_f, memra_engine::QT_F8_E4M3, rb_e4m3, false)?;
            let _ = e.qmatvec_mmvq_raw(&d_q8_0[c], &xd, 1, in_f, out_f, memra_engine::QT_Q8_0, rb_q8_0, false)?;
        }
        e.stream().synchronize()?;

        let mut t_f8: Vec<f64> = Vec::with_capacity(iters);
        let mut t_q8: Vec<f64> = Vec::with_capacity(iters);
        for i in 0..iters {
            let c = i % copies;
            // INTERLEAVED: the two arms share one clock/thermal regime per iteration.
            let t0 = Instant::now();
            let _ = e.qmatvec_mmvq_raw(&d_e4m3[c], &xd, 1, in_f, out_f, memra_engine::QT_F8_E4M3, rb_e4m3, false)?;
            e.stream().synchronize()?;
            t_f8.push(t0.elapsed().as_secs_f64());

            let t1 = Instant::now();
            let _ = e.qmatvec_mmvq_raw(&d_q8_0[c], &xd, 1, in_f, out_f, memra_engine::QT_Q8_0, rb_q8_0, false)?;
            e.stream().synchronize()?;
            t_q8.push(t1.elapsed().as_secs_f64());
        }
        let a = median(&mut t_f8);
        let b = median(&mut t_q8);
        let ratio = b / a;
        println!(
            "{:<28} {:>4} {:>10.2} {:>10.2} {:>8.4}x {:>+9.2} {:>9.1} {:>9.1}",
            format!("{label} {in_f}->{out_f}"),
            copies,
            a * 1e6,
            b * 1e6,
            ratio,
            100.0 * (ratio - 1.0),
            wb_e4m3 as f64 / a / 1e9,
            wb_q8_0 as f64 / b / 1e9
        );
        sum_ln += ratio.ln();
        n += 1;
    }

    let geo = (sum_ln / n as f64).exp();
    println!(
        "GEOMEAN ratio (q8_0/e4m3) over {n} shapes: {geo:.4}x  =>  delta_pp {:+.2}",
        100.0 * (geo - 1.0)
    );
    println!("byte-stream ceiling: 34/32 = 1.0625x  =>  +6.25pp if perfectly bandwidth-bound");
    println!("temp_out: {}", gpu_temp());
    Ok(())
}
