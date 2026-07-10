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

    // --- 5. FILTERED-SPEC kernels (feat/filtered-spec) ---
    {
        let t = 0.8f32;
        let nv2 = 512usize;
        let x2: Vec<f32> = (0..nv2).map(|i| ((i * 48271) % 977) as f32 / 61.0 - 6.0).collect();
        let x2d = e.htod(&x2)?;
        let rows0 = e.htod_i32(&[0])?;
        // CPU filtered-softmax reference for (top_k, top_p, min_p)
        let cpu_filtered = |top_k: usize, top_p: f64, min_p: f64| -> Vec<f64> {
            let sm = cpu_softmax(&x2, t);
            let mut idx: Vec<usize> = (0..nv2).collect();
            idx.sort_by(|&a, &b| sm[b].partial_cmp(&sm[a]).unwrap().then(a.cmp(&b)));
            let mut keep = vec![false; nv2];
            let mut mass = 0f64;
            for (r, &i) in idx.iter().enumerate() {
                let need_k = top_k > 0 && r < top_k;
                let need_p = top_p < 1.0 && mass < top_p;
                let plain = top_k == 0 && top_p >= 1.0;
                if need_k || need_p || plain { keep[i] = true; mass += sm[i]; } else { break; }
            }
            if min_p > 0.0 {
                let mx = sm.iter().cloned().fold(0.0, f64::max);
                for i in 0..nv2 { if sm[i] < min_p * mx { keep[i] = false; } }
            }
            let z: f64 = (0..nv2).filter(|&i| keep[i]).map(|i| sm[i]).sum();
            (0..nv2).map(|i| if keep[i] { sm[i] / z } else { 0.0 }).collect()
        };
        for (tk, tp, mp, name) in [(0i32, 0.9f32, 0.0f32, "top_p=0.9"),
                                   (40, 1.0, 0.0, "top_k=40"),
                                   (0, 1.0, 0.05, "min_p=0.05"),
                                   (0, 1.0, 0.0, "no-filter")] {
            let (mut thd, mut zd, mut mxd) = (e.zeros(1)?, e.zeros(1)?, e.zeros(1)?);
            e.filter_stats(&x2d, nv2, &rows0, &mut thd, &mut zd, &mut mxd, nv2, 1, t, tk, tp, mp)?;
            let refp = cpu_filtered(tk as usize, tp as f64, mp as f64);
            // gather a spread of ids and compare
            let ids: Vec<u32> = vec![0, 7, 100, 255, 511];
            let rows: Vec<i32> = vec![0; 5];
            let idsd = e.htod_u32_v(&ids)?; let rowsd = e.htod_i32(&rows)?;
            // broadcast th/z to per-pair arrays
            let thv = e.dtoh(&thd)?[0]; let zv = e.dtoh(&zd)?[0];
            let thp = e.htod(&vec![thv; 5])?; let zp = e.htod(&vec![zv; 5])?;
            let mut outd = e.zeros(5)?;
            e.softmax_gather_filtered(&x2d, nv2, &idsd, &rowsd, &thp, &zp, &mut outd, nv2, 5, t)?;
            let out = e.dtoh(&outd)?;
            let mut maxerr = 0f64;
            for (k2, &id) in ids.iter().enumerate() {
                maxerr = maxerr.max((out[k2] as f64 - refp[id as usize]).abs());
            }
            let ok = maxerr < 2e-3;   // binary-search threshold quantization near set boundaries
            println!("filter {name}: maxabs={maxerr:.2e} {}", if ok { "OK" } else { fails += 1; "FAIL" });
        }
        // filtered residual: empirical vs CPU on top_p=0.9 filtered p/q
        let q2: Vec<f32> = (0..nv2).map(|i| ((i * 16807) % 977) as f32 / 61.0 - 6.0).collect();
        let q2d = e.htod(&q2)?;
        let fp = cpu_filtered(0, 0.9, 0.0);
        let fq = {
            let hold = x2.clone(); let _ = hold;
            // rebuild reference helper over q2
            let sm = cpu_softmax(&q2, t);
            let mut idx: Vec<usize> = (0..nv2).collect();
            idx.sort_by(|&a, &b| sm[b].partial_cmp(&sm[a]).unwrap().then(a.cmp(&b)));
            let mut keep = vec![false; nv2]; let mut mass = 0f64;
            for &i in idx.iter() { if mass < 0.9 { keep[i] = true; mass += sm[i]; } else { break; } }
            let z: f64 = (0..nv2).filter(|&i| keep[i]).map(|i| sm[i]).sum();
            let v: Vec<f64> = (0..nv2).map(|i| if keep[i] { sm[i] / z } else { 0.0 }).collect(); v
        };
        let mut r: Vec<f64> = fp.iter().zip(&fq).map(|(a, b)| (a - b).max(0.0)).collect();
        let rs: f64 = r.iter().sum();
        for v in &mut r { *v /= rs; }
        let stats = |v: &[f32], tk: i32, tp: f32| -> Result<(f32, f32, f32), Box<dyn std::error::Error>> {
            let vd = e.htod(v)?;
            let (mut thd, mut zd, mut mxd) = (e.zeros(1)?, e.zeros(1)?, e.zeros(1)?);
            e.filter_stats(&vd, v.len(), &rows0, &mut thd, &mut zd, &mut mxd, v.len(), 1, t, tk, tp, 0.0)?;
            Ok((e.dtoh(&mxd)?[0], e.dtoh(&thd)?[0], e.dtoh(&zd)?[0]))
        };
        let ps = stats(&x2, 0, 0.9)?;
        let qs = stats(&q2, 0, 0.9)?;
        let mut tokd2 = e.alloc_u32_zeroed(1)?;
        let draws = 8000usize;
        let mut freq = vec![0f64; nv2];
        for i in 0..draws {
            e.residual_sample_filtered(&x2d, Some(&q2d), nv2, t, 99, i as u32, ps, qs, &mut tokd2)?;
            freq[e.dtoh_u32(&tokd2)?[0] as usize] += 1.0 / draws as f64;
        }
        let maxerr = freq.iter().zip(&r).map(|(f, p)| (f - p).abs()).fold(0.0, f64::max);
        let ok = maxerr < 0.025;
        println!("filtered residual empirical (8k draws): maxerr={maxerr:.4} {}", if ok { "OK" } else { fails += 1; "FAIL" });
    }

    println!("{}", if fails == 0 { "=== sample-check ALL GREEN ===" } else { "=== sample-check FAILURES ===" });
    std::process::exit(if fails == 0 { 0 } else { 1 });
}
