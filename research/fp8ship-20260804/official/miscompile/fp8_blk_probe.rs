//! fp8-ship item B diagnostic (2026-08-04, vast 2x5090 box): the kernel-check
//! [fp8-blk-gpu] bit-parity arm FAILS on this box (CUDA 13.0.88) while the same commit
//! passed on the laptop (nvcc 13.1). ~1 bad byte per 34-byte Q8_0 block. This probe
//! reproduces the synthetic case and localizes WHICH byte of the block differs and by
//! how much, so the finding names the mechanism (d-half encoding vs a qs rounding).
use memra_engine::Engine;
use memra_gguf::nvfp4_repack::{f32_to_q8_0, fp8_e4m3_to_f32};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let e = Engine::new(0)?;
    println!("GPU: {}", e.ctx().name()?);
    for &(out_f, in_f) in &[(8usize, 32usize), (256usize, 512usize)] {
        let (rows, cols) = (out_f.div_ceil(128), in_f.div_ceil(128));
        let codes: Vec<u8> = (0..out_f * in_f).map(|i| (i % 256) as u8).collect();
        let grid: Vec<f32> = (0..rows * cols)
            .map(|i| 2f32.powi((i % 10) as i32 - 4) * (1.0 + 0.125 * (i % 3) as f32))
            .collect();
        let mut cpu: Vec<u8> = Vec::with_capacity(out_f * (in_f / 32) * 34);
        for o in 0..out_f {
            let row: Vec<f32> = (0..in_f)
                .map(|e| fp8_e4m3_to_f32(codes[o * in_f + e]) * grid[(o >> 7) * cols + (e >> 7)])
                .collect();
            cpu.extend_from_slice(&f32_to_q8_0(&row));
        }
        let dev = e.fp8_blk_dequant_q8_0(&codes, &grid, out_f, in_f)?;
        let gpu = e.dtoh_u8(&dev)?;
        assert_eq!(gpu.len(), cpu.len());
        let mut off_hist = [0usize; 34];
        let mut n_bad = 0usize;
        let mut examples = 0;
        for (i, (a, b)) in cpu.iter().zip(&gpu).enumerate() {
            if a != b {
                n_bad += 1;
                off_hist[i % 34] += 1;
                if examples < 8 {
                    let blk = i / 34;
                    let off = i % 34;
                    if off < 2 {
                        let db = u16::from_le_bytes([cpu[blk * 34], cpu[blk * 34 + 1]]);
                        let dg = u16::from_le_bytes([gpu[blk * 34], gpu[blk * 34 + 1]]);
                        println!(
                            "  blk {blk} off {off} (d half): cpu 0x{db:04x} gpu 0x{dg:04x} (ulp diff {})",
                            dg as i32 - db as i32
                        );
                    } else {
                        println!(
                            "  blk {blk} off {off} (qs[{}]): cpu {} gpu {} (delta {})",
                            off - 2,
                            cpu[i] as i8,
                            gpu[i] as i8,
                            gpu[i] as i8 as i32 - cpu[i] as i8 as i32
                        );
                    }
                    examples += 1;
                }
            }
        }
        println!(
            "[{out_f}x{in_f}] bytes={} bad={n_bad}; bad-offset histogram (byte-in-block): {:?}",
            cpu.len(),
            off_hist
        );
    }
    Ok(())
}
