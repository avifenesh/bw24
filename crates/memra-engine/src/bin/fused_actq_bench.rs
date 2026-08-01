//! FUSED ACT-EPILOGUE microbench (lane 3, 2026-08-01): kernel-only timing of the MoE prefill
//! down-projection activation epilogue — two-pass (moe_pairs_{silu,gelu}_mul f32 write +
//! mmq_iq_quantize_act f32 re-read) vs the fused mmq_iq_fused_act_quant single launch — at the
//! REAL prefill shapes (q35 board-2048: [16384 x 512] silu; g26 pp1736: [13888 x 704] gelu).
//! One process, zero model loads (the CPU-load discipline). Prints us/iter + effective act-pass
//! GB/s per arm, plus a byte-compare of the scratches (the kernel-check contract, re-affirmed
//! at the production shapes).
use memra_engine::Engine;

fn pr(i: usize) -> f32 {
    let x = (i.wrapping_mul(2654435761) ^ 0x9E3779B9) as u32;
    ((x >> 8) as f32 / (1u32 << 24) as f32) * 8.0 - 4.0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let e = Engine::new(0)?;
    println!("GPU: {}", e.ctx().name()?);
    let reps = 200usize;
    for (name, in_f, n_pairs, act_kind) in [("q35-silu", 512usize, 16384usize, 0i32),
                                            ("g26-gelu", 704, 13888, 1)] {
        let n = n_pairs * in_f;
        let g: Vec<f32> = (0..n).map(|i| pr(i + 17)).collect();
        let u: Vec<f32> = (0..n).map(|i| pr(i + 29)).collect();
        let gd = e.htod(&g)?;
        let ud = e.htod(&u)?;
        // byte-compare once at the production shape.
        let act = if act_kind == 0 { e.moe_pairs_silu_mul(&gd, &ud, n)? }
                  else { e.moe_pairs_gelu_mul(&gd, &ud, n)? };
        let scr_ref = e.mmq_iq_quantize_act(&act, in_f, n_pairs)?;
        let scr_f = e.mmq_iq_fused_act_quant(&gd, &ud, in_f, n_pairs, act_kind)?;
        let b_ref: Vec<u8> = e.stream().clone_dtoh(&scr_ref)?;
        let b_f: Vec<u8> = e.stream().clone_dtoh(&scr_f)?;
        e.stream().synchronize()?;
        let nbad = b_ref.iter().zip(&b_f).filter(|(a, b)| a != b).count();
        // two-pass arm.
        for _ in 0..20 {
            let a = if act_kind == 0 { e.moe_pairs_silu_mul(&gd, &ud, n)? }
                    else { e.moe_pairs_gelu_mul(&gd, &ud, n)? };
            let _ = e.mmq_iq_quantize_act(&a, in_f, n_pairs)?;
        }
        e.stream().synchronize()?;
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            let a = if act_kind == 0 { e.moe_pairs_silu_mul(&gd, &ud, n)? }
                    else { e.moe_pairs_gelu_mul(&gd, &ud, n)? };
            let _ = e.mmq_iq_quantize_act(&a, in_f, n_pairs)?;
        }
        e.stream().synchronize()?;
        let us_two = t0.elapsed().as_secs_f64() * 1e6 / reps as f64;
        // fused arm.
        for _ in 0..20 { let _ = e.mmq_iq_fused_act_quant(&gd, &ud, in_f, n_pairs, act_kind)?; }
        e.stream().synchronize()?;
        let t1 = std::time::Instant::now();
        for _ in 0..reps { let _ = e.mmq_iq_fused_act_quant(&gd, &ud, in_f, n_pairs, act_kind)?; }
        e.stream().synchronize()?;
        let us_fused = t1.elapsed().as_secs_f64() * 1e6 / reps as f64;
        // traffic model: two-pass = read g+u (2n f32) + write act (n) + read act (n) + write scratch;
        // fused = read g+u (2n) + write scratch. scratch bytes = padded blocks.
        let scratch = b_ref.len() as f64;
        let two_bytes = (4 * n) as f64 * 4.0 + scratch;
        let fused_bytes = (2 * n) as f64 * 4.0 + scratch;
        println!("{name} [{n_pairs}x{in_f}] byte_mismatch={nbad}: two-pass {us_two:.1} us \
                  ({:.0} GB/s) | fused {us_fused:.1} us ({:.0} GB/s) | {:.2}x epilogue, \
                  {:.1} us saved/layer-call",
                 two_bytes / us_two / 1e3, fused_bytes / us_fused / 1e3,
                 us_two / us_fused, us_two - us_fused);
    }
    Ok(())
}
