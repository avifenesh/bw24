//! Minimal unit repro for the top_p/min_p "!" (id 0) injection.
//! Mechanism under test: the full-accept bonus draw in spec.rs passes
//! gumbel_perturb_filtered the (row_max, th) of the PREVIOUS verify column
//! (`col_stats.last()`), not of the column it actually samples. When the wrong
//! row_max is larger than the true one, every e0 = exp((x-row_max)/T) falls
//! below th, the whole perturbed vector becomes -3.4e38, and the 2-pass argmax
//! returns its smallest-index tie-break => token id 0 == "!".
use memra_engine::Engine;

fn cpu_softmax(x: &[f32], t: f32) -> Vec<f64> {
    let m = x.iter().cloned().fold(f32::MIN, f32::max) as f64;
    let e: Vec<f64> = x.iter().map(|&v| ((v as f64 - m) / t as f64).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|v| v / s).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let e = Engine::new(0)?;
    let t = 0.8f32;
    let nv = 4096usize;
    let rows0 = e.htod_i32(&[0])?;

    // Two adjacent "verify columns" with different maxima — the real situation:
    // consecutive target columns routinely differ by 1-3 logits at the peak.
    // col A (the PREVIOUS column, whose stats spec.rs wrongly reuses): peak +2.5
    // col B (the column actually sampled for the bonus): peak lower.
    let mk = |peak: f32, seedmul: usize| -> Vec<f32> {
        (0..nv).map(|i| {
            let base = ((i * seedmul) % 971) as f32 / 97.0 - 5.0;
            if i == 1234 { peak } else { base }
        }).collect::<Vec<f32>>()
    };
    let col_a = mk(9.0, 48271);   // wrong-stats donor: high peak
    let col_b = mk(6.0, 48271);   // the column being sampled: lower peak
    let (ad, bd) = (e.htod(&col_a)?, e.htod(&col_b)?);

    let stats = |v: &cudarc::driver::CudaSlice<f32>, tk: i32, tp: f32, mp: f32|
        -> Result<(f32, f32, f32), Box<dyn std::error::Error>> {
        let (mut th, mut z, mut mx) = (e.zeros(1)?, e.zeros(1)?, e.zeros(1)?);
        e.filter_stats(v, nv, &rows0, &mut th, &mut z, &mut mx, nv, 1, t, tk, tp, mp)?;
        Ok((e.dtoh(&mx)?[0], e.dtoh(&th)?[0], e.dtoh(&z)?[0]))
    };

    println!("=== mechanism: bonus draw from col_b using col_a's stats (the spec.rs bug) ===");
    println!("{:<34} {:>10} {:>12} {:>9} {:>9}", "filter", "th(col_a)", "id0 rate", "matched", "mismatched");
    for (tk, tp, mp, name) in [
        (0i32, 0.95f32, 0.0f32, "top_p=0.95 only"),
        (0, 1.0, 0.05, "min_p=0.05 only"),
        (40, 1.0, 0.0, "top_k=40 only"),
        (40, 0.95, 0.05, "llama shape k40+p0.95+m0.05"),
        (0, 1.0, 0.0, "no filter (memra default)"),
    ] {
        let sa = stats(&ad, tk, tp, mp)?;   // wrong donor stats
        let sb = stats(&bd, tk, tp, mp)?;   // correct stats
        let mut pb = e.zeros(nv)?;
        let draws = 400usize;
        let mut id0_mis = 0usize;
        let mut id0_ok = 0usize;
        for i in 0..draws {
            // MISMATCHED (today's spec.rs full-accept bonus)
            e.gumbel_perturb_filtered(&bd, &mut pb, nv, 7, i as u32, t, sa.0, sa.1)?;
            let tok = e.dtoh_u32_one(&e.argmax_token_device(&pb, nv)?)?;
            if tok <= 1 { id0_mis += 1; }
            // MATCHED (correct)
            e.gumbel_perturb_filtered(&bd, &mut pb, nv, 7, i as u32, t, sb.0, sb.1)?;
            let tok2 = e.dtoh_u32_one(&e.argmax_token_device(&pb, nv)?)?;
            if tok2 <= 1 { id0_ok += 1; }
        }
        // true probability of ids 0/1 under the correctly filtered softmax
        let sm = cpu_softmax(&col_b, t);
        let true_low = sm[0] + sm[1];
        println!("{:<34} {:>10.3e} {:>11.1}% {:>8.1}% {:>9.1}%  (true low-id prob {:.2e})",
                 name, sa.1, 100.0 * id0_mis as f64 / draws as f64,
                 100.0 * id0_ok as f64 / draws as f64,
                 100.0 * id0_mis as f64 / draws as f64, true_low);
        let _ = sb;
    }
    Ok(())
}
