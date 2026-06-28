//! gdn_scan_s128 microbench + correctness harness, sized to the REAL hybrid-prefill shape
//! (H=32 v-heads, S_v=128, T=512). No model load -> tiny GPU footprint, ncu-able in a short
//! window between other agents' runs. Times the kernel (median of N) and validates vs a CPU
//! reference of the exact recurrence (the kernel_check oracle, scaled up to H=32/T=512).
//!
//! Usage:  ./target/release/gdn-bench [T] [H] [iters]
//! ncu  :  sudo ncu --kernel-name regex:gdn_scan --launch-skip 2 --launch-count 1 \
//!              --set full ./target/release/gdn-bench 512 32 5
use bw24_engine::Engine;
use std::time::Instant;

fn pr(i: usize) -> f32 {
    let x = (i.wrapping_mul(2654435761) ^ 0x9E3779B9) as u32;
    ((x >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
}

fn cpu_ref(q: &[f32], k: &[f32], v: &[f32], g: &[f32], beta: &[f32],
           s_v: usize, h: usize, t: usize, scale: f32) -> Vec<f32> {
    let mut out = vec![0f32; s_v * h * t];
    for hh in 0..h {
        let mut s = vec![0f32; s_v * s_v]; // s[col*s_v + i] = S[i][col]
        for tt in 0..t {
            let base = (tt * h + hh) * s_v;
            let qt = &q[base..][..s_v];
            let kt = &k[base..][..s_v];
            let vt = &v[base..][..s_v];
            let gv = (g[tt * h + hh]).exp();
            let bv = beta[tt * h + hh];
            let mut new_s = s.clone();
            for col in 0..s_v {
                let mut kv = 0.0f32;
                for i in 0..s_v { kv += s[col * s_v + i] * kt[i]; }
                let delta = (vt[col] - gv * kv) * bv;
                let mut attn = 0.0f32;
                for i in 0..s_v {
                    let ns = gv * s[col * s_v + i] + kt[i] * delta;
                    new_s[col * s_v + i] = ns;
                    attn += ns * qt[i];
                }
                out[base + col] = attn * scale;
            }
            s = new_s;
        }
    }
    out
}

fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let t: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(512);
    let h: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(32);
    let iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(20);
    let s_v = 128usize;
    let scale = 1.0 / (s_v as f32).sqrt();

    let e = Engine::new(0)?;
    println!("GPU: {}  shape H={h} S_v={s_v} T={t} iters={iters}", e.ctx().name()?);

    let q: Vec<f32> = (0..s_v * h * t).map(|i| pr(i) * 0.1).collect();
    let k: Vec<f32> = (0..s_v * h * t).map(|i| pr(i + 5) * 0.1).collect();
    let v: Vec<f32> = (0..s_v * h * t).map(|i| pr(i + 9) * 0.1).collect();
    let g: Vec<f32> = (0..h * t).map(|i| -0.05 - pr(i).abs() * 0.1).collect();
    let beta: Vec<f32> = (0..h * t).map(|i| 0.5 + pr(i + 3) * 0.2).collect();
    let st0 = vec![0f32; s_v * s_v * h];

    let qd = e.htod(&q)?; let kd = e.htod(&k)?; let vd = e.htod(&v)?;
    let gd = e.htod(&g)?; let bd = e.htod(&beta)?; let sid = e.htod(&st0)?;
    let mut sod = e.zeros(s_v * s_v * h)?;
    let mut od = e.zeros(s_v * h * t)?;

    // correctness (vs CPU recurrence)
    e.gdn_scan_s128(&qd, &kd, &vd, &gd, &bd, &sid, &mut sod, &mut od, h, t, scale)?;
    let gpu_o = e.dtoh(&od)?;
    let cpu_o = cpu_ref(&q, &k, &v, &g, &beta, s_v, h, t, scale);
    let d = maxdiff(&cpu_o, &gpu_o);
    println!("gdn_scan correctness maxdiff={d:.2e} {}", if d < 1e-4 { "OK" } else { "FAIL" });

    // warmup
    for _ in 0..3 {
        e.gdn_scan_s128(&qd, &kd, &vd, &gd, &bd, &sid, &mut sod, &mut od, h, t, scale)?;
    }
    e.ctx().synchronize()?;

    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let st = Instant::now();
        e.gdn_scan_s128(&qd, &kd, &vd, &gd, &bd, &sid, &mut sod, &mut od, h, t, scale)?;
        e.ctx().synchronize()?;
        times.push(st.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = times[times.len() / 2];
    let min = times[0];
    println!("gdn_scan time: median={med:.3} ms  min={min:.3} ms");
    Ok(())
}
