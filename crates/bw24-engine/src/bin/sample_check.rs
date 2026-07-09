//! sample-check: host-reference gate for the sampled-spec primitives (spec_sample.cu, piece A).
//! Checks (all must PASS):
//!   1. gumbel temp=0  == pure copy (greedy-limit continuity)
//!   2. gumbel determinism: same (seed, pos) -> identical perturbed vector; different pos -> differs
//!   3. softmax_gather vs CPU softmax (rel < 1e-4 at temp 0.7/1.0; exact indicator at temp 0)
//!   4. residual sampler: determinism, temp->0 argmax fallback, and empirical distribution vs the
//!      CPU residual probabilities on a small vocab (10k draws, max abs freq error < 0.02)
use bw24_engine::Engine;

fn cpu_softmax(x: &[f32], t: f32) -> Vec<f64> {
    let m = x.iter().cloned().fold(f32::MIN, f32::max) as f64;
    let e: Vec<f64> = x.iter().map(|&v| ((v as f64 - m) / t as f64).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|v| v / s).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let e = Engine::new(0)?;
    let mut fails = 0;
    let n = 4096usize;
    let x: Vec<f32> = (0..n).map(|i| ((i * 2654435761usize) % 1000) as f32 / 137.0 - 3.0).collect();
    let xd = e.htod(&x)?;

    // --- 1. temp=0 copy ---
    let mut yd = e.zeros(n)?;
    e.gumbel_perturb(&xd, &mut yd, n, 42, 7, 0.0)?;
    let y = e.dtoh(&yd)?;
    let ok = y == x;
    println!("gumbel temp=0 == copy: {}", if ok { "OK" } else { fails += 1; "FAIL" });

    // --- 2. determinism ---
    let mut y1 = e.zeros(n)?; let mut y2 = e.zeros(n)?; let mut y3 = e.zeros(n)?;
    e.gumbel_perturb(&xd, &mut y1, n, 42, 7, 0.8)?;
    e.gumbel_perturb(&xd, &mut y2, n, 42, 7, 0.8)?;
    e.gumbel_perturb(&xd, &mut y3, n, 42, 8, 0.8)?;
    let (v1, v2, v3) = (e.dtoh(&y1)?, e.dtoh(&y2)?, e.dtoh(&y3)?);
    let ok = v1 == v2 && v1 != v3;
    println!("gumbel determinism (same pos ==, diff pos !=): {}", if ok { "OK" } else { fails += 1; "FAIL" });

    // --- 3. softmax_gather vs CPU ---
    for &t in &[0.7f32, 1.0] {
        let ids: Vec<u32> = vec![3, 999, 4095];
        let rows: Vec<i32> = vec![0, 0, 0];
        let idsd = e.htod_u32_v(&ids)?; let rowsd = e.htod_i32(&rows)?;
        let mut outd = e.zeros(3)?;
        e.softmax_gather(&xd, n, &idsd, &rowsd, &mut outd, n, 3, t)?;
        let out = e.dtoh(&outd)?;
        let sm = cpu_softmax(&x, t);
        let mut maxrel = 0f64;
        for (k, &id) in ids.iter().enumerate() {
            let r = ((out[k] as f64 - sm[id as usize]) / sm[id as usize]).abs();
            if r > maxrel { maxrel = r; }
        }
        let ok = maxrel < 1e-4;
        println!("softmax_gather t={t}: maxrel={maxrel:.2e} {}", if ok { "OK" } else { fails += 1; "FAIL" });
    }
    // temp=0 indicator
    {
        let am = x.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap().then(b.0.cmp(&a.0))).unwrap().0 as u32;
        let ids: Vec<u32> = vec![am, am ^ 1];
        let rows: Vec<i32> = vec![0, 0];
        let idsd = e.htod_u32_v(&ids)?; let rowsd = e.htod_i32(&rows)?;
        let mut outd = e.zeros(2)?;
        e.softmax_gather(&xd, n, &idsd, &rowsd, &mut outd, n, 2, 0.0)?;
        let out = e.dtoh(&outd)?;
        let ok = out[0] == 1.0 && out[1] == 0.0;
        println!("softmax_gather t=0 indicator: {:?} {}", out, if ok { "OK" } else { fails += 1; "FAIL" });
    }

    // --- 4. residual sampler ---
    let nv = 256usize;
    let p: Vec<f32> = (0..nv).map(|i| ((i * 7919) % 100) as f32 / 25.0).collect();
    let q: Vec<f32> = (0..nv).map(|i| ((i * 104729) % 100) as f32 / 25.0).collect();
    let pd = e.htod(&p)?; let qd = e.htod(&q)?;
    let t = 0.9f32;
    // CPU residual probabilities
    let sp = cpu_softmax(&p, t); let sq = cpu_softmax(&q, t);
    let mut r: Vec<f64> = sp.iter().zip(&sq).map(|(a, b)| (a - b).max(0.0)).collect();
    let rs: f64 = r.iter().sum();
    for v in &mut r { *v /= rs; }
    // determinism + empirical distribution (10k draws over distinct stream positions)
    let mut tokd = e.alloc_u32_zeroed(1)?;
    e.residual_sample(&pd, Some(&qd), nv, t, 42, 0, &mut tokd)?;
    let t0 = e.dtoh_u32(&tokd)?[0];
    e.residual_sample(&pd, Some(&qd), nv, t, 42, 0, &mut tokd)?;
    let t0b = e.dtoh_u32(&tokd)?[0];
    let ok = t0 == t0b;
    println!("residual determinism: {}", if ok { "OK" } else { fails += 1; "FAIL" });
    let draws = 10000usize;
    let mut freq = vec![0f64; nv];
    for i in 0..draws {
        e.residual_sample(&pd, Some(&qd), nv, t, 42, i as u32, &mut tokd)?;
        freq[e.dtoh_u32(&tokd)?[0] as usize] += 1.0 / draws as f64;
    }
    let maxerr = freq.iter().zip(&r).map(|(f, p)| (f - p).abs()).fold(0.0, f64::max);
    let ok = maxerr < 0.02;
    println!("residual empirical vs CPU (10k draws): maxerr={maxerr:.4} {}", if ok { "OK" } else { fails += 1; "FAIL" });
    // temp->0 fallback: p == q -> argmax(p)
    e.residual_sample(&pd, Some(&pd), nv, t, 42, 5, &mut tokd)?;
    let am = p.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap().then(b.0.cmp(&a.0))).unwrap().0 as u32;
    let got = e.dtoh_u32(&tokd)?[0];
    let ok = got == am;
    println!("residual p==q -> argmax fallback: got={got} want={am} {}", if ok { "OK" } else { fails += 1; "FAIL" });

    println!("{}", if fails == 0 { "=== sample-check ALL GREEN ===" } else { "=== sample-check FAILURES ===" });
    std::process::exit(if fails == 0 { 0 } else { 1 });
}
