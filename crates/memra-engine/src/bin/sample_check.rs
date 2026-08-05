//! sample-check: host-reference gate for the sampled-spec primitives (spec_sample.cu, piece A).
//! Checks (all must PASS):
//!   1. gumbel temp=0  == pure copy (greedy-limit continuity)
//!   2. gumbel determinism: same (seed, pos) -> identical perturbed vector; different pos -> differs
//!   3. softmax_gather vs CPU softmax (rel < 1e-4 at temp 0.7/1.0; exact indicator at temp 0)
//!   4. residual sampler: determinism, temp->0 argmax fallback, and empirical distribution vs the
//!      CPU residual probabilities on a small vocab (10k draws, max abs freq error < 0.02)
//!   5. filtered-spec kernels: filter_stats vs a CPU filtered-softmax reference (top_k/top_p/
//!      min_p/no-filter) + the filtered residual's empirical distribution (8k draws)
//!   6. COMPOSITION (the HANDOVER sampled-spec-arc gate (c)): the whole accept walk's OUTPUT
//!      distribution == the target p. Arms 1-5 oracle primitives in ISOLATION; only this arm
//!      catches a mis-composition (inverted accept test, residual off the wrong column, a
//!      uniform reused across slots) that leaves every individual kernel correct. 20k draws,
//!      L-inf + total-variation, with a non-degenerate-acceptance guard so it can't go vacuous.
//!   7. TRUNCATION-SET MEMBERSHIP (added 2026-08-05, lane/sampler-truncation-fix): every id a
//!      truncated draw returns must be a MEMBER of the truncation set. Arm 6 runs the DEFAULT
//!      pure-temp regime (top_k=0/top_p=1/min_p=0), where th==0 masks nothing — so it could not
//!      see the top_p/min_p bug that shipped to the public serve surface (`!` = id 0 spliced
//!      mid-word; receipts research/sampfix-20260805/). This arm sweeps the real truncation
//!      shapes INCLUDING llama's default (top_k 40 + top_p 0.95 + min_p 0.05) and asserts
//!      membership rather than distribution — a fallthrough/uninitialized id shows up as a
//!      non-member immediately, at any draw count. It also pins the specific defect: stats
//!      (row_max, th) taken from a DIFFERENT logits row than the one being sampled must never
//!      silently produce an all-masked row (the -3.4e38 wipe whose argmax tie-break returns 0).
//!
//! Why this binary matters: every token golden in the repo runs temp=0, which routes around
//! the sampler chain entirely — a broken sampled-spec kernel is INVISIBLE to argmax goldens
//! (demonstrated: research/fast-gate-20260802/break-sampling-*). Since the serve default
//! became temperature=1.0 (dogfood F4), sampled spec is the DEFAULT decode path, so this is
//! the oracle for the path the owner's daily driver actually takes.
use cudarc::driver::CudaSlice;
use memra_engine::Engine;

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

    // --- 6. COMPOSITION: the accept walk's OUTPUT distribution == the target p ---
    // This is the HANDOVER "SAMPLED-SPEC ARC" gate (c) — the one the per-kernel arms above
    // cannot reach. Arms 1-5 oracle each primitive in isolation; a spec decode can pass all
    // of them and still emit the wrong distribution if the primitives are COMPOSED wrong
    // (accept test inverted, residual fed the wrong column, a uniform reused across slots).
    //
    // The Leviathan/Chen guarantee: for ONE draft slot, the composed step
    //     x ~ q ; accept if u*q(x) < p(x) ; else x ~ norm(max(0, p - q))
    // emits x ~ p EXACTLY, for ANY draft q. So we run the real device primitives in the same
    // order and with the same host accept test spec.rs uses (`(u as f64)*(qj as f64) < pj`,
    // host_u01's Philox stream), and check the empirical output against the CPU filtered
    // softmax of p. A mis-composition shows up here as a distribution skew even though every
    // kernel is individually correct.
    //
    // Deliberately mismatched q (different logits AND a different filter) so the accept rate
    // is well below 1 — a walk that always accepts, or always rejects, would test nothing.
    {
        let t = 0.8f32;
        let nv3 = 256usize;
        let pl: Vec<f32> = (0..nv3).map(|i| ((i * 40503) % 811) as f32 / 47.0 - 5.0).collect();
        let ql: Vec<f32> = (0..nv3).map(|i| ((i * 22695) % 811) as f32 / 61.0 - 4.0).collect();
        let (pd3, qd3) = (e.htod(&pl)?, e.htod(&ql)?);
        let rows0 = e.htod_i32(&[0])?;
        let (tk, tp, mp) = (0i32, 1.0f32, 0.0f32); // pure temp: the DEFAULT serve regime
        let stats1 = |v: &CudaSlice<f32>, n: usize|
            -> Result<(f32, f32, f32), Box<dyn std::error::Error>> {
            let (mut thd, mut zd, mut mxd) = (e.zeros(1)?, e.zeros(1)?, e.zeros(1)?);
            e.filter_stats(v, n, &rows0, &mut thd, &mut zd, &mut mxd, n, 1, t, tk, tp, mp)?;
            Ok((e.dtoh(&mxd)?[0], e.dtoh(&thd)?[0], e.dtoh(&zd)?[0]))
        };
        let ps = stats1(&pd3, nv3)?;
        let qs = stats1(&qd3, nv3)?;
        // host Philox uniform — byte-for-byte the closure in spec.rs (independent stream tag).
        let host_u01 = |seed: u64, ctr: u32| -> f32 {
            let (m0, m1) = (0xD2511F53u32, 0xCD9E8D57u32);
            let (mut c0, mut c1, mut c2, mut c3) = (0xFFFF_FFFEu32, ctr, 0u32, 0u32);
            let (mut k0, mut k1) = ((seed & 0xFFFF_FFFF) as u32, (seed >> 32) as u32);
            for _ in 0..10 {
                let (h0, l0) = (((m0 as u64 * c0 as u64) >> 32) as u32, m0.wrapping_mul(c0));
                let (h1, l1) = (((m1 as u64 * c2 as u64) >> 32) as u32, m1.wrapping_mul(c2));
                let (n0, n1, n2, n3) = (h1 ^ c1 ^ k0, l1, h0 ^ c3 ^ k1, l0);
                c0 = n0; c1 = n1; c2 = n2; c3 = n3;
                k0 = k0.wrapping_add(0x9E3779B9);
                k1 = k1.wrapping_add(0xBB67AE85);
            }
            (c0 as f32 + 1.0) * (1.0 / 4294967296.0)
        };
        // CPU reference: the filtered softmax of p (pure temp => plain softmax).
        let refp = cpu_softmax(&pl, t);
        let seed = 1234u64;
        let draws = 20000usize;
        let mut freq = vec![0f64; nv3];
        let mut accepts = 0usize;
        let mut perturb = e.zeros(nv3)?;
        let mut tokd3 = e.alloc_u32_zeroed(1)?;
        let (mut idbuf, mut zbuf) = (e.zeros(1)?, e.zeros(1)?);
        for i in 0..draws {
            let sp = i as u32;
            // 1. draft proposes x ~ filtered q  (gumbel-max, the real draft primitive)
            e.gumbel_perturb_filtered(&qd3, &mut perturb, nv3, seed, sp, t, qs.0, qs.1)?;
            let xtok = e.dtoh_u32_one(&e.argmax_token_device(&perturb, nv3)?)?;
            // 2. gather p(x) and q(x) with the real filtered gathers
            let idsd = e.htod_u32_v(&[xtok])?;
            let g = |src: &CudaSlice<f32>, st: (f32, f32, f32)|
                -> Result<f32, Box<dyn std::error::Error>> {
                let thp = e.htod(&[st.1])?; let zp = e.htod(&[st.2])?;
                let mut o = e.zeros(1)?;
                e.softmax_gather_filtered(src, nv3, &idsd, &rows0, &thp, &zp, &mut o, nv3, 1, t)?;
                Ok(e.dtoh(&o)?[0])
            };
            let (pj, qj) = (g(&pd3, ps)?, g(&qd3, qs)?);
            // 3. THE ACCEPT TEST, exactly as spec.rs writes it (u*q < p, division-free)
            let u = host_u01(seed, sp);
            let tok = if (u as f64) * (qj as f64) < pj as f64 {
                accepts += 1;
                xtok
            } else {
                // 4. reject -> residual sample from norm(max(0, fp - fq))
                e.residual_sample_filtered(&pd3, Some(&qd3), nv3, t, seed, sp, ps, qs, &mut tokd3)?;
                e.dtoh_u32(&tokd3)?[0]
            };
            let _ = (&mut idbuf, &mut zbuf);
            freq[tok as usize] += 1.0 / draws as f64;
        }
        let acc_rate = accepts as f64 / draws as f64;
        // L-infinity on the empirical PMF. 20k draws over nv=256: the binomial sd at the
        // modal mass (~0.03) is ~1.2e-3, so 0.012 is ~10 sd of slack for MC noise while a
        // real composition bug (accept test inverted, residual off the wrong column) moves
        // the modal bins by 0.05-0.3 — far outside.
        let maxerr = freq.iter().zip(&refp).map(|(f, p)| (f - p).abs()).fold(0.0, f64::max);
        // total-variation distance: the aggregate view, catches diffuse skew L-inf can miss.
        let tv: f64 = freq.iter().zip(&refp).map(|(f, p)| (f - p).abs()).sum::<f64>() / 2.0;
        let ok = maxerr < 0.012 && tv < 0.05;
        println!("composed accept-walk output ~ p (20k draws, acc={acc_rate:.3}): \
                  maxabs={maxerr:.4} tv={tv:.4} {}",
                 if ok { "OK" } else { fails += 1; "FAIL" });
        // Guard the guard: a degenerate accept rate would make the arm vacuous.
        let ok2 = acc_rate > 0.05 && acc_rate < 0.95;
        println!("composed walk exercises BOTH branches (acc={acc_rate:.3} in 0.05..0.95): {}",
                 if ok2 { "OK" } else { fails += 1; "FAIL" });

        // 6b. q == p must accept ~always (the self-draft limit) and still emit p.
        let mut acc2 = 0usize;
        let n2 = 4000usize;
        for i in 0..n2 {
            let sp = 500_000u32 + i as u32;
            e.gumbel_perturb_filtered(&pd3, &mut perturb, nv3, seed, sp, t, ps.0, ps.1)?;
            let xtok = e.dtoh_u32_one(&e.argmax_token_device(&perturb, nv3)?)?;
            let idsd = e.htod_u32_v(&[xtok])?;
            let thp = e.htod(&[ps.1])?; let zp = e.htod(&[ps.2])?;
            let mut o = e.zeros(1)?;
            e.softmax_gather_filtered(&pd3, nv3, &idsd, &rows0, &thp, &zp, &mut o, nv3, 1, t)?;
            let pv = e.dtoh(&o)?[0];
            if (host_u01(seed, sp) as f64) * (pv as f64) < pv as f64 { acc2 += 1; }
        }
        let r2 = acc2 as f64 / n2 as f64;
        // u < 1 always (host_u01 returns (c0+1)/2^32 <= 1.0, and p==q cancels), so this is
        // an exactness statement about the test itself, not a statistical one.
        let ok3 = r2 > 0.999;
        println!("self-draft (q==p) accept rate == 1 (got {r2:.4}): {}",
                 if ok3 { "OK" } else { fails += 1; "FAIL" });
    }

    // --- 7. TRUNCATION-SET MEMBERSHIP (the gate that would have caught the public-surface bug) ---
    // Arm 6 above exercises the pure-temp DEFAULT regime, where th == 0 and nothing is masked.
    // The bug that shipped lived exclusively in the truncated regimes: the full-accept bonus draw
    // fed gumbel_perturb_filtered the stats of a NEIGHBOURING verify column, so `th` (a threshold
    // in e-units of its own row's max) was compared against e0 computed from the wrong row_max.
    // When the donor peak was higher by more than T*ln(1/th), every id failed `e0 >= th`, the row
    // became uniformly -3.4e38, and the argmax's smallest-index tie-break emitted id 0 ("!").
    //
    // Two assertions, both membership-based (no draw-count sensitivity, no MC slack needed):
    //   (a) MATCHED stats: every drawn id is in the CPU truncation set, for each filter shape.
    //   (b) MISMATCHED stats must NOT be silently absorbed: with a deliberately wrong donor row
    //       the kernel is allowed to draw a different token, but it must never fall through to
    //       the all-masked wipe (which is what id 0 signalled). This pins the defect class, so a
    //       future refactor that reintroduces cross-row stats reuse fails HERE, loudly.
    {
        let t = 0.8f32;
        let nv4 = 2048usize;
        let rows0 = e.htod_i32(&[0])?;
        // DISTINCT-BY-CONSTRUCTION base row: (i*48271) % nv4 is a PERMUTATION of 0..nv4 (48271
        // odd, nv4 = 2^11 => gcd 1), so no two logits tie. That matters: the device filter is a
        // THRESHOLD (binary search on the exp value), so at an exact tie on the top_k boundary it
        // keeps every tied id while a naive rank-k reference keeps only 40 — reading as spurious
        // non-members. Distinct logits make threshold and rank semantics coincide exactly.
        // Spacing is 8/2048 = 0.0039 logits => adjacent e-ratio exp(-0.0049) ~ 0.995, four orders
        // above the 2^-24 binary-search resolution, so the boundary is unambiguous.
        let base_row: Vec<f32> = (0..nv4)
            .map(|i| ((i * 48271) % nv4) as f32 / nv4 as f32 * 8.0 - 4.0)
            .collect();
        // The donor row is the SAME distribution shifted UP by SHIFT. Softmax is shift-invariant,
        // so the truncation SET and `th` (an e-unit threshold relative to the row's OWN max) are
        // bit-identical between the two rows, while `row_max` differs by exactly SHIFT. That
        // isolates the single defect variable — row_max — with nothing else moving. SHIFT=4.0 at
        // T=0.8 scales every e0 by exp(-5) = 6.7e-3, below min_p=0.05 => guaranteed full wipe.
        const SHIFT: f32 = 4.0;
        let target = base_row.clone();
        let donor: Vec<f32> = base_row.iter().map(|v| v + SHIFT).collect();
        let (dd, td) = (e.htod(&donor)?, e.htod(&target)?);
        let stats = |v: &CudaSlice<f32>, tk: i32, tp: f32, mp: f32|
            -> Result<(f32, f32, f32), Box<dyn std::error::Error>> {
            let (mut th, mut z, mut mx) = (e.zeros(1)?, e.zeros(1)?, e.zeros(1)?);
            e.filter_stats(v, nv4, &rows0, &mut th, &mut z, &mut mx, nv4, 1, t, tk, tp, mp)?;
            Ok((e.dtoh(&mx)?[0], e.dtoh(&th)?[0], e.dtoh(&z)?[0]))
        };
        // CPU truncation set, llama order (top_k, then top_p, then min_p) on the temp-scaled
        // softmax. top_k/top_p select a prefix of the desc order; the boundary is applied
        // TIE-INCLUSIVELY to match the device's threshold implementation (a no-op here, since
        // the construction has no ties — kept so the reference stays honest if the row changes).
        let cpu_keep = |row: &[f32], top_k: usize, top_p: f64, min_p: f64| -> Vec<bool> {
            let n = row.len();
            let sm = cpu_softmax(row, t);
            let mut idx: Vec<usize> = (0..n).collect();
            idx.sort_by(|&a, &b| sm[b].partial_cmp(&sm[a]).unwrap().then(a.cmp(&b)));
            let mut cut = n;
            if top_k > 0 { cut = cut.min(top_k); }
            if top_p < 1.0 {
                let mut mass = 0f64;
                let mut c = n;
                for (r, &i) in idx.iter().enumerate() {
                    mass += sm[i];
                    if mass >= top_p { c = r + 1; break; }
                }
                cut = cut.min(c);
            }
            let boundary = sm[idx[cut - 1]];
            let mut keep: Vec<bool> = (0..n).map(|i| sm[i] >= boundary).collect();
            if min_p > 0.0 {
                let mx = sm.iter().cloned().fold(0.0, f64::max);
                for i in 0..n { if sm[i] < min_p * mx { keep[i] = false; } }
            }
            keep
        };
        let mut pb = e.zeros(nv4)?;
        let draws = 512usize;
        for (tk, tp, mp, name) in [
            (0i32,  0.95f32, 0.0f32,  "top_p=0.95"),
            (0,     1.0,     0.05,    "min_p=0.05"),
            (40,    1.0,     0.0,     "top_k=40"),
            (40,    0.95,    0.05,    "llama-default k40+p0.95+m0.05"),
            (0,     1.0,     0.0,     "pure-temp (memra default)"),
        ] {
            let st = stats(&td, tk, tp, mp)?;         // CORRECT stats for the sampled row
            let sd = stats(&dd, tk, tp, mp)?;         // wrong-row donor stats (the old bug)
            let keep = cpu_keep(&target, tk as usize, tp as f64, mp as f64);
            let nkeep = keep.iter().filter(|&&k| k).count();
            // (a) THE BINDING ASSERTION: with correctly matched stats every drawn id must be a
            // MEMBER of the truncation set. Membership is draw-count-insensitive — one
            // fallthrough/uninitialized id fails it, no MC slack required.
            let mut nonmember = 0usize;
            let mut low_id = 0usize;
            for i in 0..draws {
                e.gumbel_perturb_filtered(&td, &mut pb, nv4, 4242, i as u32, t, st.0, st.1)?;
                let tok = e.dtoh_u32_one(&e.argmax_token_device(&pb, nv4)?)? as usize;
                if !keep[tok] { nonmember += 1; }
                if tok <= 1 { low_id += 1; }
            }
            let ok = nonmember == 0;
            println!("trunc-membership {name}: set={nkeep} nonmember={nonmember}/{draws} \
                      lowid={low_id} {}", if ok { "OK" } else { fails += 1; "FAIL" });
            // (b) DEFECT SIGNATURE (diagnostic): the same row sampled with the DONOR's stats.
            // th is shift-invariant but row_max is not, so a foreign row_max under-scales every
            // e0 — for min_p (th pinned at 0.05) that wipes the row to -3.4e38 and the argmax
            // tie-break emits id 0, the production "!" splice. Reported per shape so the
            // fragility ORDER (min_p >= top_p >> top_k) stays visible in the log.
            let mut wipe = 0usize;
            for i in 0..draws.min(128) {
                e.gumbel_perturb_filtered(&td, &mut pb, nv4, 4242, i as u32, t, sd.0, sd.1)?;
                let y = e.dtoh(&pb)?;
                if y.iter().all(|&v| v <= -3.4e38f32) { wipe += 1; }
            }
            println!("  defect signature (donor row_max, th={:.3e}): all-masked {}/{}",
                     sd.1, wipe, draws.min(128));
        }
    }

    // --- 8. FULL-ACCEPT BONUS COLUMN INDEXING under truncation (THE binding regression gate) ---
    // Arm 7(a) uses correctly-matched stats, so it passes with or without the fix; the defect was
    // never in a kernel, it was in WHICH ROW's stats spec.rs handed the kernel. This arm therefore
    // replicates spec.rs's full-accept bonus path on a synthetic multi-column verify buffer and
    // asserts the emitted bonus is a member of the truncation set OF THE COLUMN IT SAMPLES.
    //
    // The indexing invariant being pinned: the p-gather covers verify rows {base+j-1 : j in
    // 0..k_round, j>0 || base==1} — i.e. columns up to base+k_round-2. The full-accept bonus is
    // drawn from column base+k_round-1, which is ALWAYS one past the gathered set. Any code that
    // reuses the gathered stats for the bonus row (the shipped bug: `col_stats.last()`) samples
    // with a foreign (row_max, th) and, whenever the neighbour's peak is higher, emits id 0.
    //
    // Both base arms are covered (base=0 = no pending bonus, base=1 = pending bonus rides col 0).
    {
        let t = 0.8f32;
        let nv5 = 1024usize;
        let ncol = 4usize;
        // Columns are the SAME tie-free permutation row (see arm 7) shifted by a per-column
        // offset. Shift-invariance of softmax means all four columns share one truncation set and
        // one `th`, so the ONLY thing that differs between the correct and the buggy call is
        // row_max — the variable under test — and the CPU reference cannot drift between columns.
        // The LAST GATHERED column (2) sits 4.0 ABOVE the bonus column (3). That sign matters:
        // a donor row_max HIGHER than the sampled row's under-scales e0 by exp(-4/0.8) = 6.7e-3,
        // which is below every truncated th here (min_p 5.0e-2, top_p 5.0e-2, top_k 8.3e-1) =>
        // the whole row masks to -3.4e38 and the argmax tie-break emits id 0 — the production
        // "!" splice, reproduced exactly. (The opposite sign is also a bug, just a quieter one:
        // it INFLATES e0, over-admits, and leaks non-members instead of wiping — measured at
        // 19/256 for top_p and 169/256 for top_k on this same fixture. The membership assertion
        // catches that direction too, so both are covered.)
        // NOTE: shifts are added to EVERY id (not a single peak), which is what keeps the
        // distribution — and therefore the reference set — invariant. An earlier draft moved one
        // peak instead and silently produced no row_max delta at all (the base pattern's own max
        // dominated), which is exactly the kind of vacuous gate this arm exists to avoid.
        let col_shift = [0.0f32, 4.0, 4.0, 0.0];
        let base_row5: Vec<f32> = (0..nv5)
            .map(|i| ((i * 40503) % nv5) as f32 / nv5 as f32 * 8.0 - 4.0)
            .collect();
        let mut tl: Vec<f32> = Vec::with_capacity(ncol * nv5);
        for c in 0..ncol {
            for i in 0..nv5 { tl.push(base_row5[i] + col_shift[c]); }
        }
        let tld = e.htod(&tl)?;
        let col_of = |c: usize| -> Vec<f32> { tl[c * nv5..(c + 1) * nv5].to_vec() };
        let cpu_keep_col = |c: usize, top_k: usize, top_p: f64, min_p: f64| -> Vec<bool> {
            let row = col_of(c);
            let sm = cpu_softmax(&row, t);
            let mut idx: Vec<usize> = (0..nv5).collect();
            idx.sort_by(|&a, &b| sm[b].partial_cmp(&sm[a]).unwrap().then(a.cmp(&b)));
            let mut cut = nv5;
            if top_k > 0 { cut = cut.min(top_k); }
            if top_p < 1.0 {
                let mut mass = 0f64;
                let mut c2 = nv5;
                for (r, &i) in idx.iter().enumerate() {
                    mass += sm[i];
                    if mass >= top_p { c2 = r + 1; break; }
                }
                cut = cut.min(c2);
            }
            let boundary = sm[idx[cut - 1]];
            let mut keep: Vec<bool> = (0..nv5).map(|i| sm[i] >= boundary).collect();
            if min_p > 0.0 {
                let mx = sm.iter().cloned().fold(0.0, f64::max);
                for i in 0..nv5 { if sm[i] < min_p * mx { keep[i] = false; } }
            }
            keep
        };
        let mut pb = e.zeros(nv5)?;
        let draws = 256usize;
        for (tk, tp, mp, name) in [
            (0i32,  0.95f32, 0.0f32, "top_p=0.95"),
            (0,     1.0,     0.05,   "min_p=0.05"),
            (40,    1.0,     0.0,    "top_k=40"),
            (40,    0.95,    0.05,   "llama-default k40+p0.95+m0.05"),
        ] {
            for base in [0usize, 1usize] {
                let k_round = ncol - base;             // full accept of every drafted slot
                let bonus_col = base + k_round - 1;    // the column the bonus is drawn from
                // gathered rows, exactly spec.rs's rule
                let gathered: Vec<usize> = (0..k_round)
                    .filter(|&j| j > 0 || base == 1)
                    .map(|j| base + j - 1)
                    .collect();
                // stats per gathered row (batched, as spec.rs does)
                let rowsd = e.htod_i32(&gathered.iter().map(|&r| r as i32).collect::<Vec<_>>())?;
                let nr = gathered.len();
                let (mut thd, mut zd, mut mxd) = (e.zeros(nr)?, e.zeros(nr)?, e.zeros(nr)?);
                e.filter_stats(&tld, nv5, &rowsd, &mut thd, &mut zd, &mut mxd, nv5, nr, t, tk, tp, mp)?;
                let (thv, mxv) = (e.dtoh(&thd)?, e.dtoh(&mxd)?);
                // Sanity: the gathered set must NOT already cover the bonus column — if this
                // ever fails the invariant this arm guards has changed and the arm needs a rewrite.
                let covered = gathered.contains(&bonus_col);
                if covered {
                    fails += 1;
                    println!("bonus-col invariant BROKEN (gathered {gathered:?} covers bonus col \
                              {bonus_col}) — arm 8 needs updating: FAIL");
                    continue;
                }
                // FIXED RULE (what spec.rs does now): stats computed from the bonus column itself.
                let brow = e.htod_i32(&[bonus_col as i32])?;
                let (mut bth, mut bz, mut bmx) = (e.zeros(1)?, e.zeros(1)?, e.zeros(1)?);
                e.filter_stats(&tld, nv5, &brow, &mut bth, &mut bz, &mut bmx, nv5, 1, t, tk, tp, mp)?;
                let (fmx, fth) = (e.dtoh(&bmx)?[0], e.dtoh(&bth)?[0]);
                // OLD RULE (the shipped bug): reuse the LAST gathered row's stats.
                let (omx, oth) = (mxv[nr - 1], thv[nr - 1]);
                let keep = cpu_keep_col(bonus_col, tk as usize, tp as f64, mp as f64);
                // materialize the bonus column into an owned buffer — exactly what spec.rs's
                // `col_buf` copy_view_into does before the perturb.
                let bonus_slice = e.htod(&col_of(bonus_col))?;
                let mut bad_fixed = 0usize;
                let mut bad_old = 0usize;
                let mut low_old = 0usize;
                for i in 0..draws {
                    e.gumbel_perturb_filtered(&bonus_slice, &mut pb, nv5, 5150, i as u32, t, fmx, fth)?;
                    let tok = e.dtoh_u32_one(&e.argmax_token_device(&pb, nv5)?)? as usize;
                    if !keep[tok] { bad_fixed += 1; }
                    e.gumbel_perturb_filtered(&bonus_slice, &mut pb, nv5, 5150, i as u32, t, omx, oth)?;
                    let tok_o = e.dtoh_u32_one(&e.argmax_token_device(&pb, nv5)?)? as usize;
                    if !keep[tok_o] { bad_old += 1; }
                    if tok_o <= 1 { low_old += 1; }
                }
                let ok = bad_fixed == 0;
                println!("bonus-col {name} base={base}: gathered={gathered:?} bonus_col={bonus_col} \
                          nonmember={bad_fixed}/{draws} {}", if ok { "OK" } else { fails += 1; "FAIL" });
                // GATE TEETH, asserted (not just reported): the old cross-row rule MUST fail this
                // construction, otherwise the arm is vacuous and would silently pass a
                // reintroduced bug. Every truncated shape here has th above the exp(-5) scaling
                // error, so all three wipe to id 0 — the production signature. Pure-temp is
                // excluded by the loop (th == 0 masks nothing; that regime is genuinely immune,
                // which is why the untruncated serve default never saw this).
                let teeth_ok = bad_old > 0 && low_old > 0;
                println!("  gate teeth (old cross-row rule must FAIL here): \
                          nonmember={bad_old}/{draws} lowid={low_old}/{draws} {}",
                         if teeth_ok { "OK" } else { fails += 1; "FAIL (arm went vacuous)" });
            }
        }
    }

    println!("{}", if fails == 0 { "=== sample-check ALL GREEN ===" } else { "=== sample-check FAILURES ===" });
    std::process::exit(if fails == 0 { 0 } else { 1 });
}
