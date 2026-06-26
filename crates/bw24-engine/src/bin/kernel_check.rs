//! M1 gate: validate each Stage-1 kernel against a CPU reference. Run before wiring the forward.

use bw24_engine::Engine;

fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}
fn pr(i: usize) -> f32 {
    let x = (i.wrapping_mul(2654435761) ^ 0x9E3779B9) as u32;
    ((x >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let e = Engine::new(0)?;
    println!("GPU: {}", e.ctx().name()?);
    let mut fails = 0;

    // --- RMSNorm ---
    {
        let (ncols, nrows) = (320usize, 4usize);
        let eps = 1e-6f32;
        let x: Vec<f32> = (0..ncols * nrows).map(pr).collect();
        let w: Vec<f32> = (0..ncols).map(|i| 0.5 + pr(i + 9) * 0.1).collect();
        // cpu ref
        let mut cpu = vec![0f32; ncols * nrows];
        for r in 0..nrows {
            let xr = &x[r * ncols..r * ncols + ncols];
            let ms: f32 = xr.iter().map(|v| v * v).sum::<f32>() / ncols as f32;
            let s = 1.0 / (ms + eps).sqrt();
            for i in 0..ncols { cpu[r * ncols + i] = xr[i] * s * w[i]; }
        }
        let xd = e.htod(&x)?; let wd = e.htod(&w)?; let mut dd = e.zeros(ncols * nrows)?;
        e.rms_norm(&xd, &wd, &mut dd, ncols, nrows, eps)?;
        let gpu = e.dtoh(&dd)?;
        let d = maxdiff(&cpu, &gpu);
        println!("rms_norm     maxdiff={d:.2e} {}", if d < 1e-4 { "OK" } else { fails += 1; "FAIL" });
    }

    // --- L2 norm ---
    {
        let (ncols, nrows) = (128usize, 6usize);
        let eps = 1e-6f32;
        let x: Vec<f32> = (0..ncols * nrows).map(|i| pr(i + 3)).collect();
        let mut cpu = vec![0f32; ncols * nrows];
        for r in 0..nrows {
            let xr = &x[r * ncols..r * ncols + ncols];
            let ss: f32 = xr.iter().map(|v| v * v).sum();
            let s = 1.0 / (ss + eps).sqrt();
            for i in 0..ncols { cpu[r * ncols + i] = xr[i] * s; }
        }
        let xd = e.htod(&x)?; let mut dd = e.zeros(ncols * nrows)?;
        e.l2_norm(&xd, &mut dd, ncols, nrows, eps)?;
        let gpu = e.dtoh(&dd)?;
        let d = maxdiff(&cpu, &gpu);
        println!("l2_norm      maxdiff={d:.2e} {}", if d < 1e-4 { "OK" } else { fails += 1; "FAIL" });
    }

    // --- RoPE NEOX (full rotary, head_dim=n_dims=128, 1 head, 3 tokens) ---
    {
        let (head_dim, n_dims, n_heads, n_tokens) = (128usize, 128usize, 1usize, 3usize);
        let freq_base = 1e6f32; let freq_scale = 1.0f32;
        let theta_scale = freq_base.powf(-2.0 / n_dims as f32);
        let x: Vec<f32> = (0..head_dim * n_heads * n_tokens).map(|i| pr(i + 5)).collect();
        let pos: Vec<i32> = (0..n_tokens as i32).collect();
        // cpu ref: pairs (j, j+half)
        let half = n_dims / 2;
        let mut cpu = x.clone();
        for tok in 0..n_tokens {
            for h in 0..n_heads {
                let base = (tok * n_heads + h) * head_dim;
                for j in 0..half {
                    let theta = pos[tok] as f32 * theta_scale.powf(j as f32) * freq_scale;
                    let (c, s) = (theta.cos(), theta.sin());
                    let x0 = x[base + j]; let x1 = x[base + j + half];
                    cpu[base + j] = x0 * c - x1 * s;
                    cpu[base + j + half] = x0 * s + x1 * c;
                }
            }
        }
        let mut xd = e.htod(&x)?; let posd = e.htod_i32(&pos)?;
        e.rope_neox(&mut xd, &posd, head_dim, n_dims, n_heads, n_tokens, freq_base, freq_scale)?;
        let gpu = e.dtoh(&xd)?;
        let d = maxdiff(&cpu, &gpu);
        println!("rope_neox    maxdiff={d:.2e} {}", if d < 1e-4 { "OK" } else { fails += 1; "FAIL" });
    }

    // --- silu_mul ---
    {
        let n = 1024usize;
        let g: Vec<f32> = (0..n).map(|i| pr(i)).collect();
        let u: Vec<f32> = (0..n).map(|i| pr(i + 1)).collect();
        let cpu: Vec<f32> = (0..n).map(|i| (g[i] / (1.0 + (-g[i]).exp())) * u[i]).collect();
        let gd = e.htod(&g)?; let ud = e.htod(&u)?; let mut dd = e.zeros(n)?;
        e.silu_mul(&gd, &ud, &mut dd, n)?;
        let gpu = e.dtoh(&dd)?;
        let d = maxdiff(&cpu, &gpu);
        println!("silu_mul     maxdiff={d:.2e} {}", if d < 1e-5 { "OK" } else { fails += 1; "FAIL" });
    }

    // --- naive SDPA (1 head, no GQA, causal, head_dim=64, T=T_kv=4) ---
    {
        let (hd, nh, nhkv, t, tkv) = (64usize, 2usize, 1usize, 4usize, 4usize);
        let scale = 1.0 / (hd as f32).sqrt();
        let q: Vec<f32> = (0..hd * nh * t).map(|i| pr(i) * 0.2).collect();
        let k: Vec<f32> = (0..hd * nhkv * tkv).map(|i| pr(i + 7) * 0.2).collect();
        let v: Vec<f32> = (0..hd * nhkv * tkv).map(|i| pr(i + 11) * 0.2).collect();
        // cpu ref
        let mut cpu = vec![0f32; hd * nh * t];
        for head in 0..nh {
            let kvh = head / (nh / nhkv);
            for qt in 0..t {
                let q_pos = (tkv - t) + qt;
                let qv = &q[(qt * nh + head) * hd..][..hd];
                let mut sc = vec![0f32; tkv];
                for tk in 0..tkv {
                    let kv = &k[(tk * nhkv + kvh) * hd..][..hd];
                    let mut acc = 0.0; for d in 0..hd { acc += qv[d] * kv[d]; }
                    acc *= scale;
                    if tk > q_pos { acc = -1e30; }
                    sc[tk] = acc;
                }
                let mx = sc.iter().cloned().fold(-1e30f32, f32::max);
                let mut sum = 0.0; for s in sc.iter_mut() { *s = (*s - mx).exp(); sum += *s; }
                for s in sc.iter_mut() { *s /= sum; }
                let ov = &mut cpu[(qt * nh + head) * hd..][..hd];
                for d in 0..hd {
                    let mut acc = 0.0;
                    for tk in 0..tkv { acc += sc[tk] * v[(tk * nhkv + kvh) * hd + d]; }
                    ov[d] = acc;
                }
            }
        }
        let qd = e.htod(&q)?; let kd = e.htod(&k)?; let vd = e.htod(&v)?; let mut od = e.zeros(hd * nh * t)?;
        e.sdpa_naive(&qd, &kd, &vd, &mut od, hd, nh, nhkv, t, tkv, scale, true)?;
        let gpu = e.dtoh(&od)?;
        let d = maxdiff(&cpu, &gpu);
        println!("sdpa_naive   maxdiff={d:.2e} {}", if d < 1e-4 { "OK" } else { fails += 1; "FAIL" });
    }

    // --- ssm_conv1d + SiLU (M2) ---
    {
        let (conv_dim, t, d_conv) = (8usize, 5usize, 4usize);
        let tp = t + d_conv - 1;
        let x: Vec<f32> = (0..conv_dim * tp).map(|i| pr(i + 13)).collect();
        let w: Vec<f32> = (0..d_conv * conv_dim).map(|i| pr(i + 21) * 0.3).collect();
        // cpu ref: y[c,t] = silu( sum_j x[c, t+j]*w[c,j] )
        let mut cpu = vec![0f32; conv_dim * t];
        for c in 0..conv_dim {
            for tt in 0..t {
                let mut acc = 0.0;
                for j in 0..d_conv { acc += x[c * tp + tt + j] * w[c * d_conv + j]; }
                cpu[c * t + tt] = acc / (1.0 + (-acc).exp());
            }
        }
        let xd = e.htod(&x)?; let wd = e.htod(&w)?; let mut yd = e.zeros(conv_dim * t)?;
        e.ssm_conv1d(&xd, &wd, &mut yd, conv_dim, t, d_conv, true)?;
        let gpu = e.dtoh(&yd)?;
        let d = maxdiff(&cpu, &gpu);
        println!("ssm_conv1d   maxdiff={d:.2e} {}", if d < 1e-5 { "OK" } else { fails += 1; "FAIL" });
    }

    // --- gdn_scan (M3): one head, S_v=128, T=3. CPU ref of the exact recurrence. ---
    {
        let s_v = 128usize; let h = 1usize; let t = 3usize;
        let scale = 1.0 / (s_v as f32).sqrt();
        let q: Vec<f32> = (0..s_v * h * t).map(|i| pr(i) * 0.1).collect();
        let k: Vec<f32> = (0..s_v * h * t).map(|i| pr(i + 5) * 0.1).collect();
        let v: Vec<f32> = (0..s_v * h * t).map(|i| pr(i + 9) * 0.1).collect();
        let g: Vec<f32> = (0..h * t).map(|i| -0.05 - pr(i).abs() * 0.1).collect(); // g_log < 0 => g in (0,1)
        let beta: Vec<f32> = (0..h * t).map(|i| 0.5 + pr(i + 3) * 0.2).collect();
        let st0 = vec![0f32; s_v * s_v * h];
        // cpu ref: state S[i][col] (we store transposed M[col][i] = S[i][col]); start 0
        let mut s = vec![0f32; s_v * s_v]; // s[col*s_v + i] = S[i][col] (transposed, matches kernel)
        let mut cpu_o = vec![0f32; s_v * h * t];
        for tt in 0..t {
            let qt = &q[(tt * h) * s_v..][..s_v];
            let kt = &k[(tt * h) * s_v..][..s_v];
            let vt = &v[(tt * h) * s_v..][..s_v];
            let gv = (g[tt]).exp();
            let bv = beta[tt];
            // compute per col
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
                cpu_o[(tt * h) * s_v + col] = attn * scale;
            }
            s = new_s;
        }
        let qd = e.htod(&q)?; let kd = e.htod(&k)?; let vd = e.htod(&v)?;
        let gd = e.htod(&g)?; let bd = e.htod(&beta)?; let sid = e.htod(&st0)?;
        let mut sod = e.zeros(s_v * s_v * h)?; let mut od = e.zeros(s_v * h * t)?;
        e.gdn_scan_s128(&qd, &kd, &vd, &gd, &bd, &sid, &mut sod, &mut od, h, t, scale)?;
        let gpu_o = e.dtoh(&od)?;
        let d = maxdiff(&cpu_o, &gpu_o);
        println!("gdn_scan     maxdiff={d:.2e} {}", if d < 1e-4 { "OK" } else { fails += 1; "FAIL" });
    }

    if fails == 0 { println!("\nM1-M3 GREEN: all kernels match CPU reference."); Ok(()) }
    else { Err(format!("{fails} kernel(s) FAILED").into()) }
}
