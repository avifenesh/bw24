//! fp8-mmq-bench — GEMM-ONLY throughput for the per-block FP8 MMQ prefill kernel vs the Q8_0 MMQ
//! floor it replaces, at the real 27B projection shapes.
//!
//! WHY THIS EXISTS separately from the model-level pp battery: on the 27B the kernel's operand (raw
//! e4m3 bytes + the f32 block grid) is made resident by the MEMRA_PP_FP8 stash, which DUPLICATES
//! every F8-origin projection on top of the resident Q8_0 and therefore has to run under a VRAM
//! budget. At the measured ~355 MiB of e4m3 per 27B layer, the 3072 MB that fits alongside a 27 GB
//! model covers a PREFIX of ~8.6 of 64 layers, so an end-to-end pp512 number is ~13% kernel and
//! ~87% floor — it cannot separate "the kernel is slower" from "the kernel barely ran". ARM A does
//! not have this problem because its scale-fold makes the operand blk=None, which admits it to the
//! MEMRA_ST_E4M3 one-copy arm (no duplicate, no budget, all 64 layers).
//!
//! So the pp comparison at equal coverage is not available on this rig, and the honest way to
//! measure the KERNEL is to measure the kernel: same shapes, same m, same device, both launchers
//! back to back, interleaved, medians. That is a GEMM-level claim and is labeled as one — it is not
//! an end-to-end speedup.
//!
//! Shapes are the 27B's own (from the safetensors headers, /root/models/qwen36-27b-fp8):
//!   q_proj 5120->12288, k/v_proj 5120->1024, o_proj 6144->5120,
//!   gate/up_proj 5120->17408, down_proj 17408->5120.
//!
//! usage: fp8-mmq-bench [m] [reps]     (default m=512, reps=9)

use memra_engine::Engine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let m: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);
    let reps: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(9);
    let e = Engine::new(0)?;
    println!("GPU: {}  m={m}  reps={reps} (interleaved fp8blk,q8_0 per rep; median of reps)", e.ctx().name()?);
    println!(
        "{:<28} {:>12} {:>12} {:>10} {:>12} {:>12}",
        "shape in->out", "fp8blk_ms", "q8_0_ms", "ratio", "fp8blk_TFLOP", "q8_0_TFLOP"
    );

    // (in_f, out_f, label)
    let shapes: [(usize, usize, &str); 6] = [
        (5120, 12288, "q_proj"),
        (5120, 1024, "k/v_proj"),
        (6144, 5120, "o_proj"),
        (5120, 17408, "gate/up_proj"),
        (17408, 5120, "down_proj"),
        (5120, 5120, "square-ref"),
    ];

    for (in_f, out_f, label) in shapes {
        // Weight operands. Both arms get the SAME logical weight: the e4m3 codes are the source of
        // truth, and the Q8_0 slab is produced from them by the merged ARM B' device dequant — i.e.
        // exactly the floor path this kernel competes with, not a synthetic Q8_0.
        let mut codes = vec![0u8; out_f * in_f];
        let mut s: u32 = 0x1234_5678;
        for c in codes.iter_mut() {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let v = ((s >> 16) & 0x7F) as u8;
            // avoid magnitude 0x7F: the hardware MMA reads it as NaN (host decodes 0.0), and the
            // dispatch path refuses any tensor containing one, so a bench must not contain one.
            let v = if v == 0x7F { 0x30 } else { v };
            *c = v | (((s >> 8) & 1) as u8) << 7;
        }
        let (rows, cols) = (out_f.div_ceil(128), in_f.div_ceil(128));
        let grid: Vec<f32> = (0..rows * cols)
            .map(|i| 0.5f32 + (i % 7) as f32 * 0.125)
            .collect();

        let w_f8 = e.htod_bytes(&codes)?;
        let g_d = e.htod(&grid)?;
        let w_q8 = e.fp8_blk_dequant_q8_0(&codes, &grid, out_f, in_f)?;

        let x: Vec<f32> = (0..m * in_f)
            .map(|i| ((i % 251) as f32 - 125.0) / 64.0)
            .collect();
        let x_d = e.htod(&x)?;

        // warmup both arms (allocation, autotune)
        let _ = e.qmatvec_mmq_fp8_blk(&w_f8, &g_d, &x_d, m, in_f, out_f)?;
        let _ = e.qmatvec_mmq_q8_0_raw(&w_q8, &x_d, m, in_f, out_f)?;
        e.stream().synchronize()?;

        let mut t_f8: Vec<f64> = Vec::with_capacity(reps);
        let mut t_q8: Vec<f64> = Vec::with_capacity(reps);
        for _ in 0..reps {
            // INTERLEAVED inside the rep loop: the two arms then share one clock/thermal regime,
            // which back-to-back blocks of N would not (clock drift is not a valid denominator).
            let t0 = std::time::Instant::now();
            let _ = e.qmatvec_mmq_fp8_blk(&w_f8, &g_d, &x_d, m, in_f, out_f)?;
            e.stream().synchronize()?;
            t_f8.push(t0.elapsed().as_secs_f64());

            let t1 = std::time::Instant::now();
            let _ = e.qmatvec_mmq_q8_0_raw(&w_q8, &x_d, m, in_f, out_f)?;
            e.stream().synchronize()?;
            t_q8.push(t1.elapsed().as_secs_f64());
        }
        t_f8.sort_by(f64::total_cmp);
        t_q8.sort_by(f64::total_cmp);
        let (a, b) = (t_f8[reps / 2], t_q8[reps / 2]);
        let flop = 2.0 * m as f64 * in_f as f64 * out_f as f64;
        println!(
            "{:<28} {:>12.4} {:>12.4} {:>9.3}x {:>12.1} {:>12.1}",
            format!("{label} {in_f}->{out_f}"),
            a * 1e3,
            b * 1e3,
            b / a,
            flop / a / 1e12,
            flop / b / 1e12
        );
    }
    Ok(())
}
