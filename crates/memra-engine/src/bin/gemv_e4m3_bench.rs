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
//! THE BLOCK-128 ARM (lane/fp8-blk128-decode, 2026-08-05). A third arm was added, and it is the
//! VERDICT INSTRUMENT for that lane: `qmatvec_e4m3_blk_mmvq` over the same raw e4m3 plane plus a
//! resident f32 `[ceil(out_f/128), ceil(in_f/128)]` scale grid — the Qwen-official FP8 class
//! (`weight_block_size [128,128]`, the shape Qwen3.6-FP8 ships and Qwen3.8-FP8 is expected to).
//! WHY THIS BIN CARRIES THE VERDICT rather than an end-to-end run: no 27B-class block-128 artifact
//! is staged on this box (the official Qwen3.6-27B-FP8 is 29 GB and lives on the remote 2x5090),
//! and per the owner rule verdicts anchor on 27B shapes only — the 1.7B synth checkpoint is a
//! bring-up instrument, never a verdict. So the verdict is the per-shape m=1 A/B on the 27B
//! projection shapes with synthetic block-128 weights, which is exactly what these rows are.
//!
//! Arms, all three interleaved inside one iteration so they share one clock/thermal regime:
//!   arm E4M3 : QT_F8_E4M3,         row_bytes = in_f          -> 1.0 B/weight   (per-tensor scale)
//!   arm BLK  : qmatvec_e4m3_blk_mmvq, row_bytes = in_f        -> 1.0 B/weight + the grid
//!   arm Q8_0 : QT_Q8_0,            row_bytes = in_f/32 * 34   -> 1.0625 B/weight (the ARM B' floor)
//! `ratio_blk = t_q8_0 / t_blk` is the lane's number; `ratio` (per-tensor) is re-measured in the
//! same hold as a CONTROL ANCHOR — it has a published value from lane/fp8-v3-gate, so agreement
//! validates the harness and `blk_vs_e4m3` isolates what the per-k128 scale load actually costs.
//! The grid is ~21 KB even for the widest 27B projection (0.02% of the weight bytes) and is
//! allocated ONE PER WEIGHT COPY, rotated with it — which is exactly a real model's geometry (each
//! projection owns its own grid and re-reads it every decode tick).
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

/// Block-128 scale grid, [ceil(out_f/128), ceil(in_f/128)] f32 in F8BlockGrid order. Magnitudes in
/// the range a real Qwen `weight_scale_inv` shows (~2^-6..2^-2), so the accumulator stays finite.
fn synth_blk_grid(out_f: usize, in_f: usize) -> Vec<f32> {
    let (rows, cols) = (out_f.div_ceil(128), in_f.div_ceil(128));
    let mut g = vec![0f32; rows * cols];
    let mut s: u32 = 0x5BF0_3635;
    for v in g.iter_mut() {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        *v = (2f32).powi(-6 + ((s >> 20) % 5) as i32);
    }
    g
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
    println!("m=1 GEMV, THREE arms: native per-tensor e4m3 (qmatvec_e4m3_mmvq, 1.0 B/w) | native");
    println!("  BLOCK-128 e4m3 (qmatvec_e4m3_blk_mmvq, 1.0 B/w + per-k128 f32 grid) | Q8_0 MMVQ floor");
    println!("  (ARM B', 1.0625 B/w). DRAM-cold (rotated copies, grid rotates with its weight);");
    println!("  all three interleaved per iter; median. ratio_blk = t_q8_0/t_blk is the lane verdict;");
    println!("  ratio (per-tensor) is the control anchor vs lane/fp8-v3-gate's published value.");
    println!(
        "{:<26} {:>3} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "shape in->out", "cp", "e4m3_us", "blk_us", "q8_0_us", "ratio", "ratio_blk",
        "blk/e4m3", "blk_GB/s"
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
    let mut sum_ln_blk = 0.0f64;
    let mut n = 0usize;
    // JSONL receipt: raw per-shape rows next to the summary (evidence discipline — a claim whose
    // raw runs exist nowhere in the repo is not evidence).
    let mut rows: Vec<String> = Vec::new();

    for (in_f, out_f, label) in shapes {
        let rb_e4m3 = in_f;
        let rb_q8_0 = (in_f / 32) * 34;
        let wb_e4m3 = out_f * rb_e4m3;
        let wb_q8_0 = out_f * rb_q8_0;
        let scols = in_f.div_ceil(128);

        // Enough copies that the rotation set is past L2 by a wide margin, capped so the set stays
        // well inside VRAM alongside the sibling lane's allocation. Three arms now share the
        // budget, so the divisor counts all three planes per copy.
        let copies = (768_000_000usize / (wb_q8_0 + 2 * wb_e4m3)).clamp(1, 64);

        let h_e4m3 = synth_e4m3(out_f, in_f);
        let h_q8_0 = synth_q8_0(out_f, in_f);
        let h_grid = synth_blk_grid(out_f, in_f);
        let d_e4m3: Vec<_> = (0..copies)
            .map(|_| e.htod_bytes(&h_e4m3))
            .collect::<Result<_, _>>()?;
        // The BLK arm gets its OWN weight copies, not a share of the per-tensor arm's — otherwise
        // the second arm of each iteration would read bytes the first just pulled into L2 and the
        // DRAM-cold discipline would be broken for exactly the arm under test.
        let d_blk: Vec<_> = (0..copies)
            .map(|_| e.htod_bytes(&h_e4m3))
            .collect::<Result<_, _>>()?;
        let d_grid: Vec<_> = (0..copies)
            .map(|_| e.htod(&h_grid))
            .collect::<Result<_, _>>()?;
        let d_q8_0: Vec<_> = (0..copies)
            .map(|_| e.htod_bytes(&h_q8_0))
            .collect::<Result<_, _>>()?;
        drop(h_e4m3);
        drop(h_q8_0);

        let x: Vec<f32> = (0..in_f).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();
        let xd = e.htod(&x)?;

        // warmup all three arms
        for c in 0..copies.min(4) {
            let _ = e.qmatvec_mmvq_raw(&d_e4m3[c], &xd, 1, in_f, out_f, memra_engine::QT_F8_E4M3, rb_e4m3, false)?;
            let _ = e.qmatvec_e4m3_blk_mmvq_raw(&d_blk[c], &xd, &d_grid[c], 1, in_f, out_f, rb_e4m3, scols)?;
            let _ = e.qmatvec_mmvq_raw(&d_q8_0[c], &xd, 1, in_f, out_f, memra_engine::QT_Q8_0, rb_q8_0, false)?;
        }
        e.stream().synchronize()?;

        let mut t_f8: Vec<f64> = Vec::with_capacity(iters);
        let mut t_bk: Vec<f64> = Vec::with_capacity(iters);
        let mut t_q8: Vec<f64> = Vec::with_capacity(iters);
        for i in 0..iters {
            let c = i % copies;
            // INTERLEAVED: all three arms share one clock/thermal regime per iteration.
            let t0 = Instant::now();
            let _ = e.qmatvec_mmvq_raw(&d_e4m3[c], &xd, 1, in_f, out_f, memra_engine::QT_F8_E4M3, rb_e4m3, false)?;
            e.stream().synchronize()?;
            t_f8.push(t0.elapsed().as_secs_f64());

            let t1 = Instant::now();
            let _ = e.qmatvec_e4m3_blk_mmvq_raw(&d_blk[c], &xd, &d_grid[c], 1, in_f, out_f, rb_e4m3, scols)?;
            e.stream().synchronize()?;
            t_bk.push(t1.elapsed().as_secs_f64());

            let t2 = Instant::now();
            let _ = e.qmatvec_mmvq_raw(&d_q8_0[c], &xd, 1, in_f, out_f, memra_engine::QT_Q8_0, rb_q8_0, false)?;
            e.stream().synchronize()?;
            t_q8.push(t2.elapsed().as_secs_f64());
        }
        let a = median(&mut t_f8);
        let k = median(&mut t_bk);
        let b = median(&mut t_q8);
        let ratio = b / a;
        let ratio_blk = b / k;
        println!(
            "{:<26} {:>3} {:>9.2} {:>9.2} {:>9.2} {:>8.4}x {:>8.4}x {:>8.4}x {:>9.1}",
            format!("{label} {in_f}->{out_f}"),
            copies,
            a * 1e6,
            k * 1e6,
            b * 1e6,
            ratio,
            ratio_blk,
            a / k,
            wb_e4m3 as f64 / k / 1e9
        );
        rows.push(format!(
            "{{\"shape\":\"{label}\",\"in_f\":{in_f},\"out_f\":{out_f},\"copies\":{copies},\
             \"iters\":{iters},\"e4m3_us\":{:.3},\"blk_us\":{:.3},\"q8_0_us\":{:.3},\
             \"ratio_e4m3\":{ratio:.5},\"ratio_blk\":{ratio_blk:.5},\"blk_over_e4m3\":{:.5},\
             \"blk_GBs\":{:.2},\"grid_bytes\":{}}}",
            a * 1e6, k * 1e6, b * 1e6, a / k, wb_e4m3 as f64 / k / 1e9,
            out_f.div_ceil(128) * scols * 4
        ));
        sum_ln += ratio.ln();
        sum_ln_blk += ratio_blk.ln();
        n += 1;
    }

    let geo = (sum_ln / n as f64).exp();
    let geo_blk = (sum_ln_blk / n as f64).exp();
    println!(
        "GEOMEAN ratio (q8_0/e4m3, CONTROL ANCHOR) over {n} shapes: {geo:.4}x  =>  delta_pp {:+.2}",
        100.0 * (geo - 1.0)
    );
    println!(
        "GEOMEAN ratio_blk (q8_0/blk128, THE LANE VERDICT) over {n} shapes: {geo_blk:.4}x  =>  delta_pp {:+.2}",
        100.0 * (geo_blk - 1.0)
    );
    println!("byte-stream ceiling: 34/32 = 1.0625x  =>  +6.25pp if perfectly bandwidth-bound");
    println!("temp_out: {}", gpu_temp());
    if let Ok(p) = std::env::var("MEMRA_GEMV_JSONL") {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&p)?;
        for r in &rows {
            writeln!(f, "{r}")?;
        }
        println!("jsonl rows appended: {} -> {p}", rows.len());
    }
    Ok(())
}
