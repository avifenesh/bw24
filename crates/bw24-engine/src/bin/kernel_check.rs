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

    // --- RANK3 LEVER (add+rmsnorm fuse): add_rms_norm must be BIT-IDENTICAL to add_f32 then
    //     rms_norm_f32 (same residual `res` AND same normed `dst`). ---
    {
        let (ncols, nrows) = (4096usize, 1usize);
        let eps = 1e-6f32;
        let a: Vec<f32> = (0..ncols * nrows).map(|i| pr(i + 61)).collect();
        let b: Vec<f32> = (0..ncols * nrows).map(|i| pr(i + 67)).collect();
        let w: Vec<f32> = (0..ncols).map(|i| 0.5 + pr(i + 71) * 0.1).collect();
        let ad = e.htod(&a)?; let bd = e.htod(&b)?; let wd = e.htod(&w)?;
        // reference: add then rms_norm.
        let mut res_ref = e.zeros(ncols * nrows)?;
        e.add(&ad, &bd, &mut res_ref, ncols * nrows)?;
        let mut z_ref = e.zeros(ncols * nrows)?;
        e.rms_norm(&res_ref, &wd, &mut z_ref, ncols, nrows, eps)?;
        // fused.
        let mut res_f = e.zeros(ncols * nrows)?;
        let mut z_f = e.zeros(ncols * nrows)?;
        e.add_rms_norm(&ad, &bd, &wd, &mut res_f, &mut z_f, ncols, nrows, eps)?;
        let rr = e.dtoh(&res_ref)?; let rf = e.dtoh(&res_f)?;
        let zr = e.dtoh(&z_ref)?; let zf = e.dtoh(&z_f)?;
        let rbad = rr.iter().zip(&rf).filter(|(x, y)| x != y).count();
        let zbad = zr.iter().zip(&zf).filter(|(x, y)| x != y).count();
        println!("add_rms_norm fused: res_mismatch={rbad} norm_mismatch={zbad} {}",
                 if rbad == 0 && zbad == 0 { "OK" } else { fails += 1; "FAIL" });
    }

    // --- DECODE GLUE-FUSION: rms_norm_q8_1 must produce BIT-IDENTICAL q8_1 to rms_norm -> quantize_q8_1
    //     (same int8 bytes, same f32 block scales). ---
    {
        let (ncols, nrows) = (4096usize, 1usize);
        let eps = 1e-6f32;
        let x: Vec<f32> = (0..ncols * nrows).map(|i| pr(i + 31)).collect();
        let w: Vec<f32> = (0..ncols).map(|i| 0.5 + pr(i + 41) * 0.1).collect();
        let xd = e.htod(&x)?; let wd = e.htod(&w)?;
        // reference: rms_norm then quantize_q8_1.
        let mut z_ref = e.zeros(ncols * nrows)?;
        e.rms_norm(&xd, &wd, &mut z_ref, ncols, nrows, eps)?;
        let (q_ref, d_ref) = e.quantize_q8_1(&z_ref, nrows, ncols)?;
        // fused.
        let (q_f, d_f) = e.rms_norm_q8_1(&xd, &wd, ncols, nrows, eps)?;
        let qr: Vec<i8> = e.stream().clone_dtoh(&q_ref)?; e.stream().synchronize()?;
        let qf: Vec<i8> = e.stream().clone_dtoh(&q_f)?; e.stream().synchronize()?;
        let dr = e.dtoh(&d_ref)?; let df = e.dtoh(&d_f)?;
        let qbad = qr.iter().zip(&qf).filter(|(x, y)| x != y).count();
        let dbad = dr.iter().zip(&df).filter(|(x, y)| x != y).count();
        println!("rms_norm_q8_1 fused: q_mismatch={qbad} d_mismatch={dbad} {}",
                 if qbad == 0 && dbad == 0 { "OK" } else { fails += 1; "FAIL" });
    }

    // --- DECODE GLUE-FUSION: add_rms_norm_q8_1 must be BIT-IDENTICAL to add_rms_norm -> quantize_q8_1
    //     (same residual `res` AND same q8_1 bytes/scales). ---
    {
        let (ncols, nrows) = (4096usize, 1usize);
        let eps = 1e-6f32;
        let a: Vec<f32> = (0..ncols * nrows).map(|i| pr(i + 61)).collect();
        let b: Vec<f32> = (0..ncols * nrows).map(|i| pr(i + 67)).collect();
        let w: Vec<f32> = (0..ncols).map(|i| 0.5 + pr(i + 71) * 0.1).collect();
        let ad = e.htod(&a)?; let bd = e.htod(&b)?; let wd = e.htod(&w)?;
        // reference: add_rms_norm (res + z) then quantize_q8_1(z).
        let mut res_ref = e.zeros(ncols * nrows)?;
        let mut z_ref = e.zeros(ncols * nrows)?;
        e.add_rms_norm(&ad, &bd, &wd, &mut res_ref, &mut z_ref, ncols, nrows, eps)?;
        let (q_ref, d_ref) = e.quantize_q8_1(&z_ref, nrows, ncols)?;
        // fused.
        let mut res_f = e.zeros(ncols * nrows)?;
        let (q_f, d_f) = e.add_rms_norm_q8_1(&ad, &bd, &wd, &mut res_f, ncols, nrows, eps)?;
        let rr = e.dtoh(&res_ref)?; let rf = e.dtoh(&res_f)?;
        let qr: Vec<i8> = e.stream().clone_dtoh(&q_ref)?; e.stream().synchronize()?;
        let qf: Vec<i8> = e.stream().clone_dtoh(&q_f)?; e.stream().synchronize()?;
        let dr = e.dtoh(&d_ref)?; let df = e.dtoh(&d_f)?;
        let rbad = rr.iter().zip(&rf).filter(|(x, y)| x != y).count();
        let qbad = qr.iter().zip(&qf).filter(|(x, y)| x != y).count();
        let dbad = dr.iter().zip(&df).filter(|(x, y)| x != y).count();
        println!("add_rms_norm_q8_1 fused: res_mismatch={rbad} q_mismatch={qbad} d_mismatch={dbad} {}",
                 if rbad == 0 && qbad == 0 && dbad == 0 { "OK" } else { fails += 1; "FAIL" });
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

    // --- RANK2 LEVER (q8_1 quant-fold): silu_mul_scaled_q8_1 must produce BIT-IDENTICAL q8_1 to the
    //     unfused silu_mul_scaled -> quantize_q8_1 (same int8 bytes, same f32 block scales). ---
    {
        let n = 2048usize;                 // multiple of 32
        let (gs, us) = (1.31f32, 0.77f32); // non-unit scales (NVFP4 macro-scale case)
        let g: Vec<f32> = (0..n).map(|i| pr(i + 3)).collect();
        let u: Vec<f32> = (0..n).map(|i| pr(i + 5)).collect();
        let gd = e.htod(&g)?;
        let ud = e.htod(&u)?;
        // unfused reference: scaled silu*mul into f32 act, then quantize_q8_1.
        let mut act = e.zeros(n)?;
        e.silu_mul_scaled(&gd, &ud, gs, us, &mut act, n)?;
        let (aq_ref, ad_ref) = e.quantize_q8_1(&act, 1, n)?;
        // fused: silu*mul + q8_1 emit in one launch.
        let (aq_f, ad_f) = e.silu_mul_scaled_q8_1(&gd, &ud, gs, us, n)?;
        let q_ref: Vec<i8> = e.stream().clone_dtoh(&aq_ref)?; e.stream().synchronize()?;
        let q_f: Vec<i8> = e.stream().clone_dtoh(&aq_f)?; e.stream().synchronize()?;
        let d_ref = e.dtoh(&ad_ref)?;
        let d_f = e.dtoh(&ad_f)?;
        let qbad = q_ref.iter().zip(&q_f).filter(|(a, b)| a != b).count();
        let dbad = d_ref.iter().zip(&d_f).filter(|(a, b)| a != b).count();
        println!("silu_mul_q8_1 fold: int8_mismatch={qbad} scale_mismatch={dbad} {}",
                 if qbad == 0 && dbad == 0 { "OK" } else { fails += 1; "FAIL" });
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

    // --- RANK3 LEVER (conv fuse, T=1 decode): ssm_conv1d_fused_decode must be BIT-IDENTICAL to the
    //     two-kernel conv_assemble_and_roll -> ssm_conv1d(T=1) path (same conv_out AND rolled state). ---
    {
        let (conv_dim, d_conv) = (96usize, 4usize);
        let pad = d_conv - 1;
        let qkv: Vec<f32> = (0..conv_dim).map(|i| pr(i + 31)).collect();
        let st0: Vec<f32> = (0..conv_dim * pad).map(|i| pr(i + 41) * 0.7).collect();
        let w: Vec<f32> = (0..d_conv * conv_dim).map(|i| pr(i + 51) * 0.3).collect();
        let qd = e.htod(&qkv)?;
        let wd = e.htod(&w)?;
        // two-kernel reference (separate state buffer).
        let mut st_ref = e.htod(&st0)?;
        let mut conv_in = e.zeros(conv_dim * (pad + 1))?;
        e.conv_assemble_and_roll(&qd, &mut st_ref, &mut conv_in, conv_dim, pad)?;
        let mut out_ref = e.zeros(conv_dim)?;
        e.ssm_conv1d(&conv_in, &wd, &mut out_ref, conv_dim, 1, d_conv, true)?;
        // fused (its own state buffer).
        let mut st_f = e.htod(&st0)?;
        let mut out_f = e.zeros(conv_dim)?;
        e.ssm_conv1d_fused_decode(&qd, &mut st_f, &wd, &mut out_f, conv_dim, d_conv)?;
        let or = e.dtoh(&out_ref)?; let of = e.dtoh(&out_f)?;
        let sr = e.dtoh(&st_ref)?; let sf = e.dtoh(&st_f)?;
        let obad = or.iter().zip(&of).filter(|(a, b)| a != b).count();
        let sbad = sr.iter().zip(&sf).filter(|(a, b)| a != b).count();
        println!("ssm_conv1d fused: out_mismatch={obad} state_mismatch={sbad} {}",
                 if obad == 0 && sbad == 0 { "OK" } else { fails += 1; "FAIL" });
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

    // --- A4 gdn chunked WY prefill: BOTH kernels vs an f64 CPU oracle of the exact recurrence.
    //     Chunked is NOT bit-identical to the sequential scan by design (different FP
    //     accumulation order) — the fair truth is f64. MEASURED noise classes (2026-07-04,
    //     adversarial synthetic: random unit-norm k rows, betas 0.3-0.9, dense random state):
    //     sequential ~4e-6 out / ~1e-5 state; chunked ~2-4e-5 out / 1.4e-5..1.1e-4 state,
    //     growing with C — the (I+A)^{-1} substitution's condition-number amplification, NOT
    //     a formulation bug (a wrong index/sign/gate produces O(1) errors). Gates:
    //     (a) chunked out rel <= 1e-4 vs truth (the SOTA-ADOPTION stop-gate), (b) state rel
    //     <= 2.5e-4 (2x headroom over the measured worst), (c) within 32x of the sequential
    //     noise (formulation-bug tripwire). run-gen argmax + e2e token agreement + run-spec
    //     remain the shipping authority.
    //     Covers: NONZERO initial state, a tail chunk (T % C != 0), T < C, and every C in
    //     {32, 64, 128}. H=4 heads, realistic magnitudes (L2-normed q/k rows, strong betas). ---
    {
        let s_v = 128usize; let h = 4usize;
        let relerr = |a: &[f64], b: &[f32]| -> f32 {
            a.iter().zip(b)
                .map(|(x, y)| ((*x - *y as f64).abs() / x.abs().max(*y as f64).max(1e-3)) as f32)
                .fold(0.0f32, f32::max)
        };
        for &(t, c) in &[(200usize, 32usize), (200, 64), (200, 128), (17, 64), (512, 64)] {
            // q/k rows ~unit-normalized like the real inputs (L2-normed), v O(1).
            let mut q = vec![0f32; s_v * h * t];
            let mut k = vec![0f32; s_v * h * t];
            for row in 0..h * t {
                let (mut nq, mut nk) = (0f32, 0f32);
                for i in 0..s_v {
                    let a = pr(row * s_v + i + 11); let b = pr(row * s_v + i + 17);
                    q[row * s_v + i] = a; k[row * s_v + i] = b;
                    nq += a * a; nk += b * b;
                }
                for i in 0..s_v {
                    q[row * s_v + i] /= nq.sqrt(); k[row * s_v + i] /= nk.sqrt();
                }
            }
            let v: Vec<f32> = (0..s_v * h * t).map(|i| pr(i + 23)).collect();
            let g: Vec<f32> = (0..h * t).map(|i| -0.02 - pr(i + 29).abs() * 0.5).collect();
            let beta: Vec<f32> = (0..h * t).map(|i| 0.3 + pr(i + 31).abs() * 0.6).collect();
            let st0: Vec<f32> = (0..s_v * s_v * h).map(|i| pr(i + 37) * 0.5).collect(); // NONZERO
            let scale = 1.0 / (s_v as f32).sqrt();
            // f64 truth (exact recurrence, per head)
            let mut o64 = vec![0f64; s_v * h * t];
            let mut s64 = vec![0f64; s_v * s_v * h];
            for hh in 0..h {
                let s = &mut s64[hh * s_v * s_v..(hh + 1) * s_v * s_v]; // s[col*s_v+i]=S[i][col]
                for (i, sv) in s.iter_mut().enumerate() { *sv = st0[hh * s_v * s_v + i] as f64; }
                for tt in 0..t {
                    let base = (tt * h + hh) * s_v;
                    let gv = (g[tt * h + hh] as f64).exp();
                    let bv = beta[tt * h + hh] as f64;
                    for col in 0..s_v {
                        let mut kv = 0f64;
                        for i in 0..s_v { kv += s[col * s_v + i] * k[base + i] as f64; }
                        let delta = (v[base + col] as f64 - gv * kv) * bv;
                        let mut attn = 0f64;
                        for i in 0..s_v {
                            let ns = gv * s[col * s_v + i] + k[base + i] as f64 * delta;
                            s[col * s_v + i] = ns;
                            attn += ns * q[base + i] as f64;
                        }
                        o64[base + col] = attn * scale as f64;
                    }
                }
            }
            let qd = e.htod(&q)?; let kd = e.htod(&k)?; let vd = e.htod(&v)?;
            let gd = e.htod(&g)?; let bd = e.htod(&beta)?; let sid = e.htod(&st0)?;
            let mut so_s = e.zeros(s_v * s_v * h)?; let mut o_s = e.zeros(s_v * h * t)?;
            e.gdn_scan_s128(&qd, &kd, &vd, &gd, &bd, &sid, &mut so_s, &mut o_s, h, t, scale)?;
            let mut so_c = e.zeros(s_v * s_v * h)?; let mut o_c = e.zeros(s_v * h * t)?;
            e.gdn_scan_chunked(&qd, &kd, &vd, &gd, &bd, &sid, &mut so_c, &mut o_c, h, t, scale, c)?;
            let (ro_s, rs_s) = (relerr(&o64, &e.dtoh(&o_s)?), relerr(&s64, &e.dtoh(&so_s)?));
            let (ro_c, rs_c) = (relerr(&o64, &e.dtoh(&o_c)?), relerr(&s64, &e.dtoh(&so_c)?));
            let ok = ro_c < 1e-4 && rs_c < 2.5e-4
                  && ro_c <= (ro_s * 32.0).max(1e-6) && rs_c <= (rs_s * 32.0).max(1e-6);
            println!("gdn_chunked  T={t:3} C={c:3} vs f64-truth: out seq={ro_s:.2e}/chunk={ro_c:.2e} \
                      state seq={rs_s:.2e}/chunk={rs_c:.2e} {}",
                     if ok { "OK" } else { fails += 1; "FAIL" });
        }
    }

    // --- qmatvec (resident-quant GEMM) vs cpu_linear(dequant(W)) on real GGUF weights ---
    if let Some(path) = std::env::args().nth(1) {
        use bw24_gguf::{GgufFile, GgmlType, dequant};
        use bw24_runtime::cpu_linear;
        let g = GgufFile::open(&path)?;
        let cases = [
            ("blk.0.ffn_gate.weight", bw24_engine::QT_Q8_0),   // exists in every layer
            ("blk.0.attn_qkv.weight", bw24_engine::QT_Q8_0),   // linear-attn layer
            ("blk.3.attn_q.weight", bw24_engine::QT_Q8_0),     // full-attn layer (il=3)
            ("blk.0.attn_v.weight", bw24_engine::QT_Q6_K),     // Q6_K in 1.7B
            ("output.weight", bw24_engine::QT_Q6_K),           // Q6_K lm_head in 1.7B
            ("token_embd.weight", bw24_engine::QT_Q8_0),
        ];
        for (tname, _) in cases {
            if let Some(t) = g.find(tname) {
                let qt = match t.ggml_type {
                    GgmlType::Q8_0 => bw24_engine::QT_Q8_0,
                    GgmlType::Q4_K => bw24_engine::QT_Q4_K,
                    GgmlType::Q6_K => bw24_engine::QT_Q6_K,
                    other => { println!("qmatvec skip {tname}: {other:?} not in stage-A"); continue; }
                };
                let in_f = t.ne[0] as usize; let out_f = t.ne[1] as usize;
                let raw = g.tensor_data(t);
                let row_bytes = raw.len() / out_f;
                let w_f32 = dequant::dequantize(t.ggml_type, raw, in_f * out_f);
                let m = 2usize;
                let x: Vec<f32> = (0..m * in_f).map(|i| pr(i + 31) * 0.1).collect();
                let cpu = cpu_linear(&x, &w_f32, m, in_f, out_f);
                let wd = e.htod_bytes(raw)?; let xd = e.htod(&x)?;
                let yd = e.qmatvec(&wd, &xd, m, in_f, out_f, qt, row_bytes)?;
                let gpu = e.dtoh(&yd)?;
                let d = maxdiff(&cpu, &gpu);
                let scale = cpu.iter().map(|v| v.abs()).fold(0.0, f32::max).max(1.0);
                let rel = d / scale;
                println!("qmatvec {tname} [{:?}] rel={rel:.2e} {}", t.ggml_type,
                         if rel < 1e-4 { "OK" } else { fails += 1; "FAIL" });
            }
        }
    } else {
        println!("(pass a GGUF path to also validate qmatvec vs CPU oracle)");
    }

    // --- Stage-B fast Q8_0 dp4a vs Stage-A f32 qmatvec (int8-activation quant => looser tol) ---
    if let Some(path) = std::env::args().nth(1) {
        use bw24_gguf::{GgufFile, GgmlType};
        let g = GgufFile::open(&path)?;
        if let Some(t) = g.find("blk.0.ffn_gate.weight").filter(|t| t.ggml_type == GgmlType::Q8_0) {
            let in_f = t.ne[0] as usize; let out_f = t.ne[1] as usize;
            let raw = g.tensor_data(t); let row_bytes = raw.len() / out_f;
            let m = 2usize;
            let x: Vec<f32> = (0..m * in_f).map(|i| pr(i + 41) * 0.1).collect();
            let wd = e.htod_bytes(raw)?; let xd = e.htod(&x)?;
            let ya = e.dtoh(&e.qmatvec(&wd, &xd, m, in_f, out_f, bw24_engine::QT_Q8_0, row_bytes)?)?;
            let yb = e.dtoh(&e.qmatvec_q8_0_fast(&wd, &xd, m, in_f, out_f, row_bytes)?)?;
            let d = maxdiff(&ya, &yb);
            let scale = ya.iter().map(|v| v.abs()).fold(0.0, f32::max).max(1e-3);
            let rel = d / scale;
            // int8 activation quant => expect ~1% rel error, not 1e-7. Gate: rel < 3e-2.
            println!("qmatvec_q8_0_fast vs Stage-A: rel={rel:.2e} {}", if rel < 3e-2 { "OK" } else { fails += 1; "FAIL" });
            println!("  (ya[0..3]={:?} yb[0..3]={:?})", &ya[..3], &yb[..3]);
        }
        // Q4_K + Q6_K fast paths vs Stage-A oracle (int8-act tolerance).
        for (tname, qt) in [("blk.0.attn_q.weight", bw24_engine::QT_Q4_K),
                            ("blk.0.attn_v.weight", bw24_engine::QT_Q6_K),
                            ("output.weight", bw24_engine::QT_Q6_K)] {
            if let Some(t) = g.find(tname) {
                let gt = match t.ggml_type { GgmlType::Q4_K => bw24_engine::QT_Q4_K, GgmlType::Q6_K => bw24_engine::QT_Q6_K, _ => continue };
                if gt != qt { continue; }
                let in_f = t.ne[0] as usize; let out_f = t.ne[1] as usize;
                let raw = g.tensor_data(t); let row_bytes = raw.len() / out_f;
                let m = 2usize;
                let x: Vec<f32> = (0..m * in_f).map(|i| pr(i + 51) * 0.1).collect();
                let wd = e.htod_bytes(raw)?; let xd = e.htod(&x)?;
                let ya = e.dtoh(&e.qmatvec(&wd, &xd, m, in_f, out_f, gt, row_bytes)?)?;
                let yb = if gt == bw24_engine::QT_Q4_K { e.dtoh(&e.qmatvec_q4_K_fast(&wd, &xd, m, in_f, out_f, row_bytes)?)? }
                         else { e.dtoh(&e.qmatvec_q6_K_fast(&wd, &xd, m, in_f, out_f, row_bytes)?)? };
                let d = maxdiff(&ya, &yb);
                let scale = ya.iter().map(|v| v.abs()).fold(0.0, f32::max).max(1e-3);
                let rel = d / scale;
                println!("{tname} [{:?}] fast vs Stage-A: rel={rel:.2e} {}", t.ggml_type, if rel < 3e-2 { "OK" } else { fails += 1; "FAIL" });
            }
        }
    }

    // --- 5 new dtypes: GPU qmatvec vs bw24 CPU-dequant oracle on REAL daily-GGUF tensors. ---
    // Oracle = cpu_linear(bw24_dequant(W), x); bw24's CPU dequant is byte-for-byte == ggml
    // dequantize_row_<type> (proven in bw24-gguf example dequant_oracle_diff), so this gates
    // the GPU paths against ggml ground truth transitively. Mirrors the Q4_K/Q6_K block above:
    //   Stage-A (dequant-in-kernel) rel < 1e-4 ; Stage-B (int8 dp4a) rel < 3e-2.
    // IQ3_S has NO dp4a fast path (intentional, see lib.rs) -> Stage-A only.
    // Skips silently if a daily GGUF is absent so the core gate still runs in CI without models.
    {
        use bw24_gguf::{GgufFile, GgmlType, dequant};
        use bw24_runtime::cpu_linear;
        const GGUF_9B: &str =
            "/home/avifenesh/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf";
        const GGUF_35B: &str =
            "/home/avifenesh/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf";
        // (gguf, tensor, expected type, QT code, fast-path selector or "" for Stage-A only)
        let cases: [(&str, &str, GgmlType, i32, &str); 5] = [
            (GGUF_9B,  "blk.0.ffn_gate.weight",      GgmlType::NVFP4,  bw24_engine::QT_NVFP4,  "nvfp4"),
            (GGUF_9B,  "blk.0.attn_gate.weight",     GgmlType::Q5_K,   bw24_engine::QT_Q5_K,   "q5k"),
            (GGUF_35B, "blk.0.ffn_gate_exps.weight", GgmlType::IQ3_S,  bw24_engine::QT_IQ3_S,  ""),
            (GGUF_35B, "blk.0.ffn_down_exps.weight", GgmlType::IQ4_XS, bw24_engine::QT_IQ4_XS, "iq4xs"),
            (GGUF_35B, "blk.40.ffn_gate_exps.weight",GgmlType::Q3_K,   bw24_engine::QT_Q3_K,   "q3k"),
        ];
        for (path, tname, gty, qt, sel) in cases {
            if !std::path::Path::new(path).exists() {
                println!("dtype5 {gty:?} {tname}: GGUF absent ({path}) — SKIP");
                continue;
            }
            let g = GgufFile::open(path)?;
            let t = match g.find(tname) {
                Some(t) if t.ggml_type == gty => t,
                Some(t) => { println!("dtype5 {tname}: type {:?} != {gty:?}", t.ggml_type); fails += 1; continue; }
                None => { println!("dtype5 {tname}: NOT FOUND in {path}"); fails += 1; continue; }
            };
            // in_f = ne[0] (K dim); out_f = ne[1] (rows). For 3D MoE tensors validate expert 0.
            let in_f = t.ne[0] as usize;
            let out_f = t.ne[1] as usize;
            let raw_all = g.tensor_data(t);
            let n_experts = if t.ne.len() >= 3 { t.ne[2] as usize } else { 1 };
            let total_rows = out_f * n_experts;
            let row_bytes = raw_all.len() / total_rows;
            let raw = &raw_all[..out_f * row_bytes]; // expert 0 slice
            let w_f32 = dequant::dequantize(gty, raw, in_f * out_f);
            let m = 2usize;
            let x: Vec<f32> = (0..m * in_f).map(|i| pr(i + 61) * 0.1).collect();
            let cpu = cpu_linear(&x, &w_f32, m, in_f, out_f);
            let scale = cpu.iter().map(|v| v.abs()).fold(0.0, f32::max).max(1.0);
            let wd = e.htod_bytes(raw)?; let xd = e.htod(&x)?;
            // Stage-A: dequant-in-kernel qmatvec (float-noise exact).
            let ya = e.dtoh(&e.qmatvec(&wd, &xd, m, in_f, out_f, qt, row_bytes)?)?;
            let rela = maxdiff(&cpu, &ya) / scale;
            println!("dtype5 [{gty:?}] {tname} (in={in_f} out={out_f}) Stage-A: rel={rela:.2e} {}",
                     if rela < 1e-4 { "OK" } else { fails += 1; "FAIL" });
            // Stage-B: int8 dp4a fast path (int8-activation tolerance), where one exists.
            if sel.is_empty() {
                println!("dtype5 [{gty:?}] {tname} Stage-B dp4a: (no fast path — Stage-A only)");
            } else {
                let yb = match sel {
                    "nvfp4" => e.dtoh(&e.qmatvec_nvfp4_fast(&wd, &xd, m, in_f, out_f, row_bytes)?)?,
                    "q5k"   => e.dtoh(&e.qmatvec_q5_K_fast(&wd, &xd, m, in_f, out_f, row_bytes)?)?,
                    "iq4xs" => e.dtoh(&e.qmatvec_iq4_XS_fast(&wd, &xd, m, in_f, out_f, row_bytes)?)?,
                    "q3k"   => e.dtoh(&e.qmatvec_q3_K_fast(&wd, &xd, m, in_f, out_f, row_bytes)?)?,
                    _ => unreachable!(),
                };
                let relb = maxdiff(&cpu, &yb) / scale;
                println!("dtype5 [{gty:?}] {tname} Stage-B dp4a: rel={relb:.2e} {}",
                         if relb < 3e-2 { "OK" } else { fails += 1; "FAIL" });
            }
        }
    }

    // --- GEMM (tensor-core int8) vs dp4a matvec: BIT-EQUIVALENCE gate (the prefill root fix). ---
    // s32 accumulate is exact vs dp4a; only the final f32 block-scale rounding differs -> rel<1e-3.
    // Runs T in {16,64,128,512} per dtype on REAL GGUF tensors. Needs a model path arg.
    if let Some(path) = std::env::args().nth(1) {
        use bw24_gguf::{GgufFile, GgmlType};
        let g = GgufFile::open(&path)?;
        // (tensor, GEMM qt, dp4a-fast selector). Each is validated if present with the right type.
        let gemm_cases: [(&str, i32, &str); 5] = [
            ("blk.0.ffn_gate.weight",  bw24_engine::QT_Q8_0,  "q8_0"),  // 35B token_embd-style Q8_0
            ("blk.0.attn_qkv.weight",  bw24_engine::QT_Q8_0,  "q8_0"),
            ("blk.3.attn_q.weight",    bw24_engine::QT_Q4_K,  "q4_K"),  // 9B/27B attn Q4_K
            ("blk.0.attn_v.weight",    bw24_engine::QT_Q6_K,  "q6_K"),
            ("output.weight",          bw24_engine::QT_Q6_K,  "q6_K"),  // Q6_K lm_head
        ];
        for (tname, want_qt, sel) in gemm_cases {
            let t = match g.find(tname) { Some(t) => t, None => continue };
            let gt = match t.ggml_type {
                GgmlType::Q8_0 => bw24_engine::QT_Q8_0, GgmlType::Q4_K => bw24_engine::QT_Q4_K,
                GgmlType::Q6_K => bw24_engine::QT_Q6_K, GgmlType::NVFP4 => bw24_engine::QT_NVFP4,
                _ => continue,
            };
            if gt != want_qt { continue; }
            let in_f = t.ne[0] as usize; let out_f = t.ne[1] as usize;
            if t.ne.len() > 2 { continue; } // skip 3D MoE expert tensors here
            let raw = g.tensor_data(t); let row_bytes = raw.len() / out_f;
            let wd = e.htod_bytes(raw)?;
            for tt in [16usize, 64, 128, 512] {
                let x: Vec<f32> = (0..tt * in_f).map(|i| pr(i + 71) * 0.1).collect();
                let xd = e.htod(&x)?;
                let ydp = match sel {
                    "q8_0" => e.qmatvec_q8_0_fast(&wd, &xd, tt, in_f, out_f, row_bytes)?,
                    "q4_K" => e.qmatvec_q4_K_fast(&wd, &xd, tt, in_f, out_f, row_bytes)?,
                    "q6_K" => e.qmatvec_q6_K_fast(&wd, &xd, tt, in_f, out_f, row_bytes)?,
                    _ => unreachable!(),
                };
                let ya = e.dtoh(&ydp)?;
                let yb = e.dtoh(&e.qmatvec_gemm_raw(&wd, &xd, tt, in_f, out_f, gt, row_bytes)?)?;
                let d = maxdiff(&ya, &yb);
                let scale = ya.iter().map(|v| v.abs()).fold(0.0, f32::max).max(1e-3);
                let rel = d / scale;
                println!("GEMM {tname} [{:?}] T={tt}: rel={rel:.2e} {}", t.ggml_type,
                         if rel < 1e-3 { "OK" } else { fails += 1; "FAIL" });
            }
        }
    }
    // NVFP4 GEMM vs dp4a on the 9B model (separate path: per-tensor macro-scale + in_f%64).
    {
        use bw24_gguf::{GgufFile, GgmlType};
        const GGUF_9B: &str =
            "/home/avifenesh/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf";
        if std::path::Path::new(GGUF_9B).exists() {
            let g = GgufFile::open(GGUF_9B)?;
            // Q5_K GEMM vs dp4a (attn_gate is Q5_K in 9B).
            if let Some(t) = g.find("blk.0.attn_gate.weight").filter(|t| t.ggml_type == GgmlType::Q5_K) {
                let in_f = t.ne[0] as usize; let out_f = t.ne[1] as usize;
                let raw = g.tensor_data(t); let row_bytes = raw.len() / out_f;
                let wd = e.htod_bytes(raw)?;
                for tt in [16usize, 64, 128, 512] {
                    let x: Vec<f32> = (0..tt * in_f).map(|i| pr(i + 91) * 0.1).collect();
                    let xd = e.htod(&x)?;
                    let ya = e.dtoh(&e.qmatvec_q5_K_fast(&wd, &xd, tt, in_f, out_f, row_bytes)?)?;
                    let yb = e.dtoh(&e.qmatvec_gemm_raw(&wd, &xd, tt, in_f, out_f, bw24_engine::QT_Q5_K, row_bytes)?)?;
                    let d = maxdiff(&ya, &yb);
                    let scale = ya.iter().map(|v| v.abs()).fold(0.0, f32::max).max(1e-3);
                    let rel = d / scale;
                    println!("GEMM blk.0.attn_gate.weight [Q5_K] T={tt}: rel={rel:.2e} {}",
                             if rel < 1e-3 { "OK" } else { fails += 1; "FAIL" });
                }
            }
            if let Some(t) = g.find("blk.0.ffn_gate.weight").filter(|t| t.ggml_type == GgmlType::NVFP4) {
                let in_f = t.ne[0] as usize; let out_f = t.ne[1] as usize;
                let raw = g.tensor_data(t); let row_bytes = raw.len() / out_f;
                let wd = e.htod_bytes(raw)?;
                for tt in [16usize, 64, 128, 512] {
                    let x: Vec<f32> = (0..tt * in_f).map(|i| pr(i + 81) * 0.1).collect();
                    let xd = e.htod(&x)?;
                    // dp4a (no macro-scale applied here; GEMM raw also skips it -> compare bare).
                    let ya = e.dtoh(&e.qmatvec_nvfp4_fast(&wd, &xd, tt, in_f, out_f, row_bytes)?)?;
                    let yb = e.dtoh(&e.qmatvec_gemm_raw(&wd, &xd, tt, in_f, out_f, bw24_engine::QT_NVFP4, row_bytes)?)?;
                    let d = maxdiff(&ya, &yb);
                    let scale = ya.iter().map(|v| v.abs()).fold(0.0, f32::max).max(1e-3);
                    let rel = d / scale;
                    println!("GEMM blk.0.ffn_gate.weight [NVFP4] T={tt}: rel={rel:.2e} {}",
                             if rel < 1e-3 { "OK" } else { fails += 1; "FAIL" });
                }
            }
            // Stage-C FP4 (mxf4nvf4 block-scale tensor-core) vs the f32 dequant oracle on NVFP4.
            // FP4 is LOSSY (e2m1 activations + e2m1 weights; scale side is lossless ue4m3) — NOT
            // bit-equivalent. Compare to cpu_linear(dequant(W)) and expect rel ~1e-2..6e-2.
            if let Some(t) = g.find("blk.0.ffn_gate.weight").filter(|t| t.ggml_type == GgmlType::NVFP4) {
                use bw24_gguf::dequant;
                use bw24_runtime::cpu_linear;
                let in_f = t.ne[0] as usize; let out_f = t.ne[1] as usize;
                let raw = g.tensor_data(t); let row_bytes = raw.len() / out_f;
                let w_f32 = dequant::dequantize(GgmlType::NVFP4, raw, in_f * out_f);
                let wd = e.htod_bytes(raw)?;
                for tt in [16usize, 64, 128, 512] {
                    let x: Vec<f32> = (0..tt * in_f).map(|i| pr(i + 83) * 0.1).collect();
                    let xd = e.htod(&x)?;
                    let cpu = cpu_linear(&x, &w_f32, tt, in_f, out_f);
                    let yb = e.dtoh(&e.qmatvec_gemm_nvfp4_fp4_raw(&wd, &xd, tt, in_f, out_f, row_bytes)?)?;
                    let d = maxdiff(&cpu, &yb);
                    let scale = cpu.iter().map(|v| v.abs()).fold(0.0, f32::max).max(1e-3);
                    let rel = d / scale;
                    // FP4 is LOSSY: e2m1 ACTIVATION quant (8 grid points/16-block) drives rel ~0.1-0.15
                    // (the weight side is bit-exact — proven by probe/fp4_4x_final.cu maxrel=0). This rel
                    // is INFORMATIONAL, NOT a hard gate: the AUTHORITATIVE FP4 gate is end-to-end argmax
                    // (BW24_FP4 run-hybrid/run-gen), which holds on the 9B and is the arbiter per the plan.
                    println!("FP4-GEMM blk.0.ffn_gate.weight [NVFP4] T={tt}: rel={rel:.2e} (informational; \
                              authoritative gate = argmax) {}", if rel < 2e-1 { "OK" } else { "HIGH" });
                }
            }
            // --- VENDORED llama NVFP4 MMQ GEMM vs the f32 dequant oracle. ---
            // W4A4-native (mxf4nvf4 block-scale mma) but with llama's 2-level FP8-e8m0/UE4M3 activation
            // quant -> should be MUCH closer to the f32 oracle than the bw24 hand-roll FP4 (rel ~0.1).
            // Authoritative gate is still end-to-end argmax; this rel is the accuracy signal that
            // llama's activation quant fixed bw24's W4A4 maxdiff 1.46.
            if let Some(t) = g.find("blk.0.ffn_gate.weight").filter(|t| t.ggml_type == GgmlType::NVFP4) {
                use bw24_gguf::dequant;
                use bw24_runtime::cpu_linear;
                let in_f = t.ne[0] as usize; let out_f = t.ne[1] as usize;
                let raw = g.tensor_data(t); let row_bytes = raw.len() / out_f;
                let _ = row_bytes;
                let w_f32 = dequant::dequantize(GgmlType::NVFP4, raw, in_f * out_f);
                let wd = e.htod_bytes(raw)?;
                for tt in [16usize, 64, 128, 512] {
                    let x: Vec<f32> = (0..tt * in_f).map(|i| pr(i + 83) * 0.1).collect();
                    let xd = e.htod(&x)?;
                    let cpu = cpu_linear(&x, &w_f32, tt, in_f, out_f);
                    let yb = e.dtoh(&e.qmatvec_mmq_nvfp4_raw(&wd, &xd, tt, in_f, out_f)?)?;
                    let d = maxdiff(&cpu, &yb);
                    let scale = cpu.iter().map(|v| v.abs()).fold(0.0, f32::max).max(1e-3);
                    let rel = d / scale;
                    println!("MMQ-GEMM blk.0.ffn_gate.weight [NVFP4] T={tt}: rel={rel:.2e} (informational; \
                              authoritative gate = argmax) {}", if rel < 2e-1 { "OK" } else { "HIGH" });
                }
            }
            // --- VENDORED llama Q4_K/Q5_K MMQ GEMM vs the f32 dequant oracle. ---
            // W-exact (int8 tile-load dequant is lossless for k-quants) + q8_1 int8 activation ->
            // rel should sit in the int8-activation band (~1e-3..1e-2). A layout/scale bug shows as
            // rel ~1.0, so a 2e-2 hard gate catches real breakage without flapping on quant noise.
            for (tname, want, qt) in [("blk.3.attn_q.weight",    GgmlType::Q4_K, bw24_engine::QT_Q4_K),
                                      ("blk.0.attn_gate.weight", GgmlType::Q5_K, bw24_engine::QT_Q5_K)] {
                let Some(t) = g.find(tname).filter(|t| t.ggml_type == want) else { continue };
                use bw24_gguf::dequant;
                use bw24_runtime::cpu_linear;
                let in_f = t.ne[0] as usize; let out_f = t.ne[1] as usize;
                let raw = g.tensor_data(t);
                let w_f32 = dequant::dequantize(want, raw, in_f * out_f);
                let wd = e.htod_bytes(raw)?;
                for tt in [16usize, 64, 128, 512] {
                    let x: Vec<f32> = (0..tt * in_f).map(|i| pr(i + 87) * 0.1).collect();
                    let xd = e.htod(&x)?;
                    let cpu = cpu_linear(&x, &w_f32, tt, in_f, out_f);
                    let yb = e.dtoh(&e.qmatvec_mmq_q45k_raw(&wd, &xd, tt, in_f, out_f, qt)?)?;
                    let d = maxdiff(&cpu, &yb);
                    let scale = cpu.iter().map(|v| v.abs()).fold(0.0, f32::max).max(1e-3);
                    let rel = d / scale;
                    println!("MMQ-GEMM {tname} [{want:?}] T={tt}: rel={rel:.2e} {}",
                             if rel < 2e-2 { "OK" } else { fails += 1; "FAIL" });
                }
            }
            // 27B ffn_down NVFP4 shape probe (in_f=17408 not a clean MMQ_ITER_K_FP4 multiple? T=512)
            // — compare MMQ vs the dp4a oracle to isolate the 27B T=513 mismatch.
            {
                const G27: &str = "/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf";
                if std::path::Path::new(G27).exists() {
                    let g27 = GgufFile::open(G27)?;
                    for tn in ["blk.0.ffn_down.weight", "blk.0.ffn_gate.weight"] {
                        if let Some(t) = g27.find(tn).filter(|t| t.ggml_type == GgmlType::NVFP4) {
                            let in_f = t.ne[0] as usize; let out_f = t.ne[1] as usize;
                            let raw = g27.tensor_data(t); let row_bytes = raw.len() / out_f;
                            let wd = e.htod_bytes(raw)?;
                            for tt in [16usize, 512] {
                                let x: Vec<f32> = (0..tt * in_f).map(|i| pr(i + 71) * 0.1).collect();
                                let xd = e.htod(&x)?;
                                let ya = e.dtoh(&e.qmatvec_nvfp4_fast(&wd, &xd, tt, in_f, out_f, row_bytes)?)?;
                                let yb = e.dtoh(&e.qmatvec_mmq_nvfp4_raw(&wd, &xd, tt, in_f, out_f)?)?;
                                let d = maxdiff(&ya, &yb);
                                let scale = ya.iter().map(|v| v.abs()).fold(0.0, f32::max).max(1e-3);
                                let rel = d / scale;
                                println!("MMQ-27B {tn} [NVFP4 in={in_f} out={out_f}] T={tt}: rel={rel:.2e} (W4A4-vs-dp4a band ~0.1) {}",
                                         if rel < 2.5e-1 { "OK" } else { "HIGH" });
                            }
                        }
                    }
                }
            }
            // --- Phase-1 CUTLASS FP4 GEMM: REPACK CORRECTNESS gate. ---
            // The de-interleave (GGUF -> plain packed e2m1) + SFB swizzle is the ONLY place a silent
            // wrong-answer hides. TWO checks isolate it:
            //  (A) WEIGHT ROUND-TRIP (activation-independent, the dispositive repack test): dequantize
            //      the CUTLASS-repacked B operand (plain packed e2m1 + LINEAR SFB) via the CUTLASS
            //      dequant oracle and compare to the GGUF f32 dequant of the SAME weight. The 2x e2m1 /
            //      0.5x ue4m3 GGUF<->standard cancellation means the real values must match to ~1e-6.
            //      A wrong nibble de-interleave or wrong scale byte breaks THIS with no activation noise.
            //  (B) GEMM-vs-f32-oracle band: CUTLASS-FP4 and hand-roll-FP4 are both LOSSY NVFP4 approxes
            //      of the same f32 matmul but use DIFFERENT activation quantizers, so they are NOT
            //      rel-1e-2 comparable to each other (~0.11 apart = activation-quant diff, NOT a bug).
            //      Correct repack => CUTLASS's rel-vs-oracle is in the SAME band as the hand-roll's.
            #[cfg(bw24_cutlass)]
            if let Some(t) = g.find("blk.0.ffn_gate.weight").filter(|t| t.ggml_type == GgmlType::NVFP4) {
                use bw24_gguf::dequant;
                use bw24_runtime::cpu_linear;
                let in_f = t.ne[0] as usize; let out_f = t.ne[1] as usize;
                let raw = g.tensor_data(t); let row_bytes = raw.len() / out_f;
                let w_f32 = dequant::dequantize(GgmlType::NVFP4, raw, in_f * out_f);
                let wd = e.htod_bytes(raw)?;
                // (A) weight round-trip. build_cutlass_weight gives swizzled SFB; for the oracle we need
                // the LINEAR SFB the dequant oracle reads, so de-interleave directly here.
                let mut b_packed = e.alloc_u8(out_f * in_f / 2)?;
                let mut sfb_lin = e.alloc_u8(out_f * (in_f / 16))?;
                e.cutlass_gguf_nvfp4_deinterleave(&wd, row_bytes, &mut b_packed, &mut sfb_lin, out_f, in_f)?;
                let mut w_rt_d = e.htod(&vec![0f32; out_f * in_f])?;
                e.cutlass_nvfp4_dequant_ref(&b_packed, &sfb_lin, &mut w_rt_d, out_f, in_f)?;
                let w_rt = e.dtoh(&w_rt_d)?;
                let wmax = w_f32.iter().map(|v| v.abs()).fold(0.0, f32::max).max(1e-6);
                let wrel = maxdiff(&w_f32, &w_rt) / wmax;
                println!("CUTLASS-FP4 weight round-trip blk.0.ffn_gate.weight [NVFP4]: rel={wrel:.2e} {}",
                         if wrel < 1e-3 { "OK" } else { fails += 1; "FAIL" });
                // (B) GEMM band. Reuse the swizzled-SFB path the real dispatch uses.
                let (b_packed_sw, sfb_sw) = e.build_cutlass_weight(&wd, out_f, in_f, row_bytes)?;
                for tt in [128usize, 512] {  // CUTLASS m>=128 regime
                    let x: Vec<f32> = (0..tt * in_f).map(|i| pr(i + 87) * 0.1).collect();
                    let xd = e.htod(&x)?;
                    let cpu = cpu_linear(&x, &w_f32, tt, in_f, out_f);
                    let scale = cpu.iter().map(|v| v.abs()).fold(0.0, f32::max).max(1e-3);
                    let yhr = e.dtoh(&e.qmatvec_gemm_nvfp4_fp4_raw(&wd, &xd, tt, in_f, out_f, row_bytes)?)?;
                    let ycl = e.dtoh(&e.cutlass_fp4_gemm(&b_packed_sw, &sfb_sw, &xd, 1.0, tt, out_f, in_f)?)?;
                    let rel_hr = maxdiff(&cpu, &yhr) / scale;
                    let rel_cl = maxdiff(&cpu, &ycl) / scale;
                    let ok = (rel_cl - rel_hr).abs() < 5e-2 && rel_cl < 2e-1;
                    println!("CUTLASS-FP4 GEMM-band blk.0.ffn_gate.weight [NVFP4] T={tt}: rel_cutlass={rel_cl:.2e} \
                              rel_handroll={rel_hr:.2e} {}", if ok { "OK" } else { fails += 1; "FAIL" });
                }
            }
        }
    }

    // --- PERF-3 MMVQ (warp-per-row decode) vs dp4a matvec: BIT-EQUIVALENCE gate. ---
    // The _mmvq kernels lift the dequant body VERBATIM from _dp4a; only layout (warp-per-row) +
    // reduction (warp-only shfl) change -> int sumi identical, only f32 reduction-order rounding
    // differs. Require rel < 1e-3. m=1 (decode regime) across in_f ∈ {model shapes} and out_f
    // small + 4096. Q8_0/Q4_K/Q6_K on the model-path arg; NVFP4 on the 9B model below.
    if let Some(path) = std::env::args().nth(1) {
        use bw24_gguf::{GgufFile, GgmlType};
        let g = GgufFile::open(&path)?;
        let mmvq_cases: [(&str, i32, &str); 5] = [
            ("blk.0.ffn_gate.weight",  bw24_engine::QT_Q8_0,  "q8_0"),
            ("blk.0.attn_qkv.weight",  bw24_engine::QT_Q8_0,  "q8_0"),
            ("blk.3.attn_q.weight",    bw24_engine::QT_Q4_K,  "q4_K"),
            ("blk.0.attn_v.weight",    bw24_engine::QT_Q6_K,  "q6_K"),
            ("output.weight",          bw24_engine::QT_Q6_K,  "q6_K"),
        ];
        for (tname, want_qt, sel) in mmvq_cases {
            let t = match g.find(tname) { Some(t) => t, None => continue };
            let gt = match t.ggml_type {
                GgmlType::Q8_0 => bw24_engine::QT_Q8_0, GgmlType::Q4_K => bw24_engine::QT_Q4_K,
                GgmlType::Q6_K => bw24_engine::QT_Q6_K, GgmlType::NVFP4 => bw24_engine::QT_NVFP4,
                _ => continue,
            };
            if gt != want_qt { continue; }
            if t.ne.len() > 2 { continue; } // skip 3D MoE expert tensors
            let in_f = t.ne[0] as usize; let out_f = t.ne[1] as usize;
            let raw = g.tensor_data(t); let row_bytes = raw.len() / out_f;
            let wd = e.htod_bytes(raw)?;
            // m=1 decode regime (the path matmul_pre routes); also m=2 to exercise blockIdx.y>0.
            for mm in [1usize, 2] {
                let x: Vec<f32> = (0..mm * in_f).map(|i| pr(i + 101) * 0.1).collect();
                let xd = e.htod(&x)?;
                let ydp = match sel {
                    "q8_0" => e.qmatvec_q8_0_fast(&wd, &xd, mm, in_f, out_f, row_bytes)?,
                    "q4_K" => e.qmatvec_q4_K_fast(&wd, &xd, mm, in_f, out_f, row_bytes)?,
                    "q6_K" => e.qmatvec_q6_K_fast(&wd, &xd, mm, in_f, out_f, row_bytes)?,
                    _ => unreachable!(),
                };
                let ya = e.dtoh(&ydp)?;
                let yb = e.dtoh(&e.qmatvec_mmvq_raw(&wd, &xd, mm, in_f, out_f, gt, row_bytes, false)?)?;
                let d = maxdiff(&ya, &yb);
                let scale = ya.iter().map(|v| v.abs()).fold(0.0, f32::max).max(1e-3);
                let rel = d / scale;
                println!("MMVQ {tname} [{:?}] m={mm}: rel={rel:.2e} {}", t.ggml_type,
                         if rel < 1e-3 { "OK" } else { fails += 1; "FAIL" });
            }
        }
    }
    // NVFP4 MMVQ vs dp4a on the 9B model (in_f%64; macro-scale skipped in both raw paths).
    {
        use bw24_gguf::{GgufFile, GgmlType};
        const GGUF_9B: &str =
            "/home/avifenesh/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf";
        if std::path::Path::new(GGUF_9B).exists() {
            let g = GgufFile::open(GGUF_9B)?;
            if let Some(t) = g.find("blk.0.ffn_gate.weight").filter(|t| t.ggml_type == GgmlType::NVFP4) {
                let in_f = t.ne[0] as usize; let out_f = t.ne[1] as usize;
                let raw = g.tensor_data(t); let row_bytes = raw.len() / out_f;
                let wd = e.htod_bytes(raw)?;
                for mm in [1usize, 2] {
                    let x: Vec<f32> = (0..mm * in_f).map(|i| pr(i + 111) * 0.1).collect();
                    let xd = e.htod(&x)?;
                    let ya = e.dtoh(&e.qmatvec_nvfp4_fast(&wd, &xd, mm, in_f, out_f, row_bytes)?)?;
                    let yb = e.dtoh(&e.qmatvec_mmvq_raw(&wd, &xd, mm, in_f, out_f, bw24_engine::QT_NVFP4, row_bytes, false)?)?;
                    let d = maxdiff(&ya, &yb);
                    let scale = ya.iter().map(|v| v.abs()).fold(0.0, f32::max).max(1e-3);
                    let rel = d / scale;
                    println!("MMVQ blk.0.ffn_gate.weight [NVFP4] m={mm}: rel={rel:.2e} {}",
                             if rel < 1e-3 { "OK" } else { fails += 1; "FAIL" });
                }
            }
        }
    }

    // --- BATCHED weight-resident matvec (_b2/_b4) vs the per-m _mmvq reference (the MTP/verify path).
    // Both quantize the same f32 activation to q8_1; the batched kernel only changes the loop nest
    // (weight loaded once, reused across m token columns) so per-(token,row) it MUST be bit-identical
    // to qmatvec_mmvq_raw (grid.y=m). m∈{2,3,4}; mcols=2 for m=2, mcols=4 for m∈{3,4}. rel<1e-3. ---
    if let Some(path) = std::env::args().nth(1) {
        use bw24_gguf::{GgufFile, GgmlType};
        let g = GgufFile::open(&path)?;
        // pick ONE 2D tensor per daily dtype (so Q8_0/Q5_K get covered regardless of model naming).
        let want: [(GgmlType, i32); 4] = [
            (GgmlType::Q8_0, bw24_engine::QT_Q8_0), (GgmlType::Q4_K, bw24_engine::QT_Q4_K),
            (GgmlType::Q5_K, bw24_engine::QT_Q5_K), (GgmlType::Q6_K, bw24_engine::QT_Q6_K),
        ];
        for (gtype, gt) in want {
            let t = match g.tensors.iter().find(|t| t.ggml_type == gtype && t.ne.len() == 2
                                                 && t.ne[0] % 256 == 0 && t.ne[1] >= 4) {
                Some(t) => t, None => continue,
            };
            let tname = t.name.clone();
            let in_f = t.ne[0] as usize; let out_f = t.ne[1] as usize;
            let raw = g.tensor_data(t); let row_bytes = raw.len() / out_f;
            let wd = e.htod_bytes(raw)?;
            for (mm, mcols) in [(2usize, 2usize), (3, 4), (4, 4)] {
                let x: Vec<f32> = (0..mm * in_f).map(|i| pr(i + 131) * 0.1).collect();
                let xd = e.htod(&x)?;
                // reference: per-m _mmvq (warp-per-row, grid.y=m). batched: _b{mcols} weight-resident.
                let yref = e.dtoh(&e.qmatvec_mmvq_raw(&wd, &xd, mm, in_f, out_f, gt, row_bytes, false)?)?;
                let ybat = e.dtoh(&e.qmatvec_batched_raw(&wd, &xd, mm, in_f, out_f, gt, row_bytes, mcols, false)?)?;
                let d = maxdiff(&yref, &ybat);
                let scale = yref.iter().map(|v| v.abs()).fold(0.0, f32::max).max(1e-3);
                let rel = d / scale;
                println!("BATCHED {tname} [{:?}] m={mm} mcols={mcols}: rel={rel:.2e} {}", t.ggml_type,
                         if rel < 1e-3 { "OK" } else { fails += 1; "FAIL" });
            }
        }
    }
    // NVFP4 batched vs per-m _mmvq on the 9B model.
    {
        use bw24_gguf::{GgufFile, GgmlType};
        const GGUF_9B: &str =
            "/home/avifenesh/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf";
        if std::path::Path::new(GGUF_9B).exists() {
            let g = GgufFile::open(GGUF_9B)?;
            if let Some(t) = g.find("blk.0.ffn_gate.weight").filter(|t| t.ggml_type == GgmlType::NVFP4) {
                let in_f = t.ne[0] as usize; let out_f = t.ne[1] as usize;
                let raw = g.tensor_data(t); let row_bytes = raw.len() / out_f;
                let wd = e.htod_bytes(raw)?;
                for (mm, mcols) in [(2usize, 2usize), (3, 4), (4, 4)] {
                    let x: Vec<f32> = (0..mm * in_f).map(|i| pr(i + 141) * 0.1).collect();
                    let xd = e.htod(&x)?;
                    let yref = e.dtoh(&e.qmatvec_mmvq_raw(&wd, &xd, mm, in_f, out_f, bw24_engine::QT_NVFP4, row_bytes, false)?)?;
                    let ybat = e.dtoh(&e.qmatvec_batched_raw(&wd, &xd, mm, in_f, out_f, bw24_engine::QT_NVFP4, row_bytes, mcols, false)?)?;
                    let d = maxdiff(&yref, &ybat);
                    let scale = yref.iter().map(|v| v.abs()).fold(0.0, f32::max).max(1e-3);
                    let rel = d / scale;
                    println!("BATCHED blk.0.ffn_gate.weight [NVFP4] m={mm} mcols={mcols}: rel={rel:.2e} {}",
                             if rel < 1e-3 { "OK" } else { fails += 1; "FAIL" });
                }
            }
        }
    }

    // --- A6 SPLIT-PLANE REPACK gates: roundtrip + byte-identity of EVERY rp consumer kernel vs
    // the original-layout reference. The repack is a pure byte permutation; each rp twin keeps the
    // exact per-(token,row) value/product order -> outputs must be BIT-identical (bit-bad == 0). ---
    {
        use bw24_gguf::{GgufFile, GgmlType};
        use bw24_engine::model::{repack_nvfp4_split, unpack_nvfp4_split};
        const GGUF_9B: &str =
            "/home/avifenesh/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf";
        let path9 = if std::path::Path::new(GGUF_9B).exists() { Some(GGUF_9B.to_string()) } else { None };
        // prefer the model under test if it has NVFP4 tensors; else the 9B.
        let srcs: Vec<String> = std::env::args().nth(1).into_iter().chain(path9).collect();
        let mut done = false;
        for path in srcs {
            if done { break; }
            let g = match GgufFile::open(&path) { Ok(g) => g, Err(_) => continue };
            // two shapes: a wide-out FFN gate (rpr2-class) and a narrow-out down/out (rpr2w8/rp-class).
            let picks: Vec<_> = g.tensors.iter()
                .filter(|t| t.ggml_type == GgmlType::NVFP4 && t.ne.len() == 2 && t.ne[0] % 64 == 0)
                .take(2).collect();
            for t in picks {
                done = true;
                let tname = t.name.clone();
                let in_f = t.ne[0] as usize; let out_f = t.ne[1] as usize;
                let raw = g.tensor_data(t); let row_bytes = raw.len() / out_f;
                let rpb = repack_nvfp4_split(raw, out_f);
                let rt_bad = unpack_nvfp4_split(&rpb, out_f).iter().zip(raw.iter())
                    .filter(|(a, b)| a != b).count();
                println!("RP roundtrip {tname}: {} mismatched bytes {}", rt_bad,
                         if rt_bad == 0 { "OK" } else { fails += 1; "FAIL" });
                let wd  = e.htod_bytes(raw)?;
                let wrp = e.htod_bytes(&rpb)?;
                let bit_bad = |a: &[f32], b: &[f32]| a.iter().zip(b)
                    .filter(|(x, y)| x.to_bits() != y.to_bits()).count();
                // m=1/2 MMVQ family (m=1 exercises mr2_rp via the default MR=2; m=2 the r1 rp twin).
                for mm in [1usize, 2] {
                    let x: Vec<f32> = (0..mm * in_f).map(|i| pr(i + 151) * 0.1).collect();
                    let xd = e.htod(&x)?;
                    let yref = e.dtoh(&e.qmatvec_mmvq_raw(&wd,  &xd, mm, in_f, out_f, bw24_engine::QT_NVFP4, row_bytes, false)?)?;
                    let yrp  = e.dtoh(&e.qmatvec_mmvq_raw(&wrp, &xd, mm, in_f, out_f, bw24_engine::QT_NVFP4, row_bytes, true)?)?;
                    let bad = bit_bad(&yref, &yrp);
                    println!("RP MMVQ {tname} m={mm}: bit-bad={bad} {}",
                             if bad == 0 { "OK" } else { fails += 1; "FAIL" });
                }
                // batched rp (auto wave-rule picks rp/rpr2/rpr2w8 per shape) vs original per-m mmvq.
                for (mm, mcols) in [(2usize, 2usize), (3, 4), (4, 4)] {
                    let x: Vec<f32> = (0..mm * in_f).map(|i| pr(i + 161) * 0.1).collect();
                    let xd = e.htod(&x)?;
                    let yref = e.dtoh(&e.qmatvec_mmvq_raw(&wd, &xd, mm, in_f, out_f, bw24_engine::QT_NVFP4, row_bytes, false)?)?;
                    let yrp = e.dtoh(&e.qmatvec_batched_raw(&wrp, &xd, mm, in_f, out_f, bw24_engine::QT_NVFP4, row_bytes, mcols, true)?)?;
                    let bad = bit_bad(&yref, &yrp);
                    println!("RP BATCHED {tname} m={mm} mcols={mcols}: bit-bad={bad} {}",
                             if bad == 0 { "OK" } else { fails += 1; "FAIL" });
                }
                // dp4a rp twin (grid (out,m), 128-thread two-level reduce) vs original dp4a.
                for mm in [1usize, 5] {
                    let x: Vec<f32> = (0..mm * in_f).map(|i| pr(i + 171) * 0.1).collect();
                    let xd = e.htod(&x)?;
                    let yref = e.dtoh(&e.qmatvec_nvfp4_fast(&wd, &xd, mm, in_f, out_f, row_bytes)?)?;
                    let yrp  = e.dtoh(&e.qmatvec_nvfp4_fast_rp(&wrp, &xd, mm, in_f, out_f, row_bytes)?)?;
                    let bad = bit_bad(&yref, &yrp);
                    println!("RP DP4A {tname} m={mm}: bit-bad={bad} {}",
                             if bad == 0 { "OK" } else { fails += 1; "FAIL" });
                }
                // prefill int8 GEMM kernel2 rp twin (the daily BW24_GEMM path) at a real T.
                {
                    let mm = 128usize;
                    let x: Vec<f32> = (0..mm * in_f).map(|i| pr(i + 181) * 0.1).collect();
                    let xd = e.htod(&x)?;
                    let yref = e.dtoh(&e.qmatvec_gemm_raw(&wd,  &xd, mm, in_f, out_f, bw24_engine::QT_NVFP4, row_bytes)?)?;
                    let yrp  = e.dtoh(&e.qmatvec_gemm_raw(&wrp, &xd, mm, in_f, out_f, bw24_engine::QT_NVFP4_RP, row_bytes)?)?;
                    let bad = bit_bad(&yref, &yrp);
                    println!("RP GEMM {tname} T={mm}: bit-bad={bad} {}",
                             if bad == 0 { "OK" } else { fails += 1; "FAIL" });
                }
                // Stage-A generic (f32 dequant-in-kernel) rp tag vs original.
                {
                    let x: Vec<f32> = (0..in_f).map(|i| pr(i + 191) * 0.1).collect();
                    let xd = e.htod(&x)?;
                    let yref = e.dtoh(&e.qmatvec(&wd,  &xd, 1, in_f, out_f, bw24_engine::QT_NVFP4, row_bytes)?)?;
                    let yrp  = e.dtoh(&e.qmatvec(&wrp, &xd, 1, in_f, out_f, bw24_engine::QT_NVFP4_RP, row_bytes)?)?;
                    let bad = bit_bad(&yref, &yrp);
                    println!("RP STAGE-A {tname}: bit-bad={bad} {}",
                             if bad == 0 { "OK" } else { fails += 1; "FAIL" });
                }
            }
        }
    }

    // --- FlashAttention prefill + decode vs CPU SDPA oracle (head_dim 256, GQA 16/4, causal) ---
    {
        let (hd, nh, nhkv) = (256usize, 16usize, 4usize);
        let scale = 1.0 / (hd as f32).sqrt();
        // CPU SDPA reference (same convention as sdpa_naive: q_pos=(T_kv-T)+qt).
        let cpu_sdpa = |q: &[f32], k: &[f32], v: &[f32], t: usize, tkv: usize| -> Vec<f32> {
            let mut o = vec![0f32; hd * nh * t];
            for head in 0..nh {
                let kvh = head / (nh / nhkv);
                for qt in 0..t {
                    let q_pos = (tkv - t) + qt;
                    let qv = &q[(qt * nh + head) * hd..][..hd];
                    let mut sc = vec![0f32; tkv];
                    for tk in 0..tkv {
                        let kv = &k[(tk * nhkv + kvh) * hd..][..hd];
                        let mut a = 0.0; for d in 0..hd { a += qv[d] * kv[d]; }
                        a *= scale; if tk > q_pos { a = -1e30; } sc[tk] = a;
                    }
                    let mx = sc.iter().cloned().fold(-1e30f32, f32::max);
                    let mut sum = 0.0; for s in sc.iter_mut() { *s = (*s - mx).exp(); sum += *s; }
                    for s in sc.iter_mut() { *s /= sum; }
                    let ov = &mut o[(qt * nh + head) * hd..][..hd];
                    for d in 0..hd { let mut a = 0.0; for tk in 0..tkv { a += sc[tk] * v[(tk*nhkv+kvh)*hd+d]; } ov[d] = a; }
                }
            }
            o
        };
        // prefill cases
        for (t, tkv) in [(16usize, 16usize), (64, 64), (100, 100), (256, 256)] {
            let q: Vec<f32> = (0..hd*nh*t).map(|i| pr(i)*0.2).collect();
            let k: Vec<f32> = (0..hd*nhkv*tkv).map(|i| pr(i+7)*0.2).collect();
            let v: Vec<f32> = (0..hd*nhkv*tkv).map(|i| pr(i+11)*0.2).collect();
            let cpu = cpu_sdpa(&q,&k,&v,t,tkv);
            let qd=e.htod(&q)?; let kd=e.htod(&k)?; let vd=e.htod(&v)?; let mut od=e.zeros(hd*nh*t)?;
            e.fa_prefill(&qd,&kd,&vd,&mut od,hd,nh,nhkv,t,tkv,scale,true)?;
            let g=e.dtoh(&od)?; let d=maxdiff(&cpu,&g);
            let sc=cpu.iter().map(|v|v.abs()).fold(0.0,f32::max).max(1e-3); let rel=d/sc;
            println!("fa_prefill T={t} Tkv={tkv}: rel={rel:.2e} {}", if rel<2e-2 {"OK"} else {fails+=1;"FAIL"});
        }
        // decode cases (T=1) — K/V come from the QUANTIZED resident cache (q8_0 K / q5_1 V).
        // Quantize the f32 K/V token-by-token via the append kernel, then fa_decode dequants
        // inline. Tolerance loosened vs the f32 path: q5_1 V (5-bit affine) is the looser link.
        let kv_dim_k = hd * nhkv;   // head_dim_k * n_head_kv (head_dim_v == head_dim_k here)
        let kv_dim_v = hd * nhkv;
        let k_tok_bytes = (kv_dim_k / 32) * 34;
        let v_tok_bytes = (kv_dim_v / 32) * 24;
        for tkv in [64usize, 128, 257] {
            let q: Vec<f32> = (0..hd*nh).map(|i| pr(i+1)*0.2).collect();
            let k: Vec<f32> = (0..hd*nhkv*tkv).map(|i| pr(i+7)*0.2).collect();
            let v: Vec<f32> = (0..hd*nhkv*tkv).map(|i| pr(i+11)*0.2).collect();
            let cpu = cpu_sdpa(&q,&k,&v,1,tkv);
            let qd=e.htod(&q)?; let kd=e.htod(&k)?; let vd=e.htod(&v)?;
            let mut kc = e.alloc_u8(tkv * k_tok_bytes)?;
            let mut vc = e.alloc_u8(tkv * v_tok_bytes)?;
            for tok in 0..tkv {
                let k_row = kd.slice(tok*kv_dim_k..(tok+1)*kv_dim_k);
                let v_row = vd.slice(tok*kv_dim_v..(tok+1)*kv_dim_v);
                e.append_kv_quantized_view(&k_row,&v_row,&mut kc,&mut vc,tok,
                                           kv_dim_k,kv_dim_v,k_tok_bytes,v_tok_bytes)?;
            }
            let kview=e.view_u8(&kc, tkv*k_tok_bytes); let vview=e.view_u8(&vc, tkv*v_tok_bytes);
            let sc=cpu.iter().map(|v|v.abs()).fold(0.0,f32::max).max(1e-3);
            // --- scalar fa_decode_f32 (the bit-reference) ---
            unsafe { std::env::remove_var("BW24_FA_VEC"); }
            let mut od=e.zeros(hd*nh)?;
            e.fa_decode(&qd,&kview,&vview,&mut od,hd,nh,nhkv,tkv,scale,k_tok_bytes,v_tok_bytes)?;
            let rel = maxdiff(&cpu,&e.dtoh(&od)?)/sc;
            // --- PERF-4 warp-per-token fa_decode_vec_q (GQA broadcast) on the SAME cache ---
            unsafe { std::env::set_var("BW24_FA_VEC", "1"); }
            let mut od_v=e.zeros(hd*nh)?;
            e.fa_decode(&qd,&kview,&vview,&mut od_v,hd,nh,nhkv,tkv,scale,k_tok_bytes,v_tok_bytes)?;
            unsafe { std::env::remove_var("BW24_FA_VEC"); }
            let rel_v = maxdiff(&cpu,&e.dtoh(&od_v)?)/sc;
            // Quantized KV (q8_0 K, q5_1 V) -> looser than f32 fa_decode (5e-3). These synthetic
            // inputs are UNIFORM-random in [-0.2,0.2] (worse than real KV: V's q5_1 affine 5-bit
            // noise ~1.35e-2/elem, amplified through the softmax-weighted average when |O| is small).
            // The block round-trip + 5th-bit gates below isolate packing CORRECTNESS; the AUTHORITATIVE
            // end-to-end gate is argmax stability on real models. Gate here: rel < 6e-2 (noise floor).
            println!("fa_decode(KVQ) Tkv={tkv}: rel={rel:.2e} {}", if rel<6e-2 {"OK"} else {fails+=1;"FAIL"});
            // PERF-4 gate: vec kernel rel < 6e-2 AND no worse than scalar within slack. The vec
            // kernel stores the dequanted KV tile in bf16 smem (8-bit mantissa) for occupancy
            // (-> the 2.2x mid-ctx decode win); the scalar path keeps f32. That adds ~1-1.5e-3
            // of bounded bf16-rounding noise vs scalar — far under the 6e-2 q5_1 noise floor, and
            // the AUTHORITATIVE end-to-end argmax gate (268/271/1178) is unaffected. Slack 2.5e-3.
            let regress = rel_v > rel + 2.5e-3;
            println!("fa_decode_vec_q(KVQ) Tkv={tkv}: rel={rel_v:.2e} (scalar {rel:.2e}) {}",
                     if rel_v<6e-2 && !regress {"OK"} else {fails+=1;"FAIL"});
        }

        // --- MULTI-ROW verify FA vs per-row loop: BYTE identity (the spec-exactness contract) ---
        // fa_decode_rows must reproduce the per-row fa_decode loop of full_attn_verify EXACTLY
        // (same per-row split partition + walk + combine order). Any nonzero bit diff here means
        // the fused kernel's per-row program diverged from fa_decode_vec_q — a run-spec argmax
        // flip waiting to happen. Cases cross a 64-key split boundary (128->129 keys => n_splits
        // 2->3 between rows) and sit at the vec-path floor (t_kv=96).
        for (base_len, t) in [(95usize, 5usize), (127, 4), (256, 3), (1000, 5)] {
            let tkv_max = base_len + t;
            let q: Vec<f32> = (0..hd*nh*t).map(|i| pr(i+3)*0.2).collect();
            let k: Vec<f32> = (0..hd*nhkv*tkv_max).map(|i| pr(i+7)*0.2).collect();
            let v: Vec<f32> = (0..hd*nhkv*tkv_max).map(|i| pr(i+11)*0.2).collect();
            let qd=e.htod(&q)?; let kd=e.htod(&k)?; let vd=e.htod(&v)?;
            let mut kc = e.alloc_u8(tkv_max * k_tok_bytes)?;
            let mut vc = e.alloc_u8(tkv_max * v_tok_bytes)?;
            for tok in 0..tkv_max {
                let k_row = kd.slice(tok*kv_dim_k..(tok+1)*kv_dim_k);
                let v_row = vd.slice(tok*kv_dim_v..(tok+1)*kv_dim_v);
                e.append_kv_quantized_view(&k_row,&v_row,&mut kc,&mut vc,tok,
                                           kv_dim_k,kv_dim_v,k_tok_bytes,v_tok_bytes)?;
            }
            // reference: the per-row loop exactly as full_attn_verify's fallback runs it
            let mut o_loop = e.zeros(hd*nh*t)?;
            for r in 0..t {
                let t_kv_r = base_len + r + 1;
                let kview=e.view_u8(&kc, t_kv_r*k_tok_bytes);
                let vview=e.view_u8(&vc, t_kv_r*v_tok_bytes);
                let mut q_row = e.zeros(hd*nh)?;
                let q_src = qd.slice(r*nh*hd..(r+1)*nh*hd);
                e.copy_view_into(&mut q_row, 0, &q_src, nh*hd)?;
                let mut o_row = e.zeros(hd*nh)?;
                e.fa_decode(&q_row,&kview,&vview,&mut o_row,hd,nh,nhkv,t_kv_r,scale,k_tok_bytes,v_tok_bytes)?;
                e.copy_into(&mut o_loop, r*nh*hd, &o_row, nh*hd)?;
            }
            // fused multi-row launch on the same cache
            let kview=e.view_u8(&kc, tkv_max*k_tok_bytes);
            let vview=e.view_u8(&vc, tkv_max*v_tok_bytes);
            let mut o_rows = e.zeros(hd*nh*t)?;
            e.fa_decode_rows(&qd,&kview,&vview,&mut o_rows,hd,nh,nhkv,base_len,t,scale,k_tok_bytes,v_tok_bytes)?;
            let a = e.dtoh(&o_loop)?; let b = e.dtoh(&o_rows)?;
            let bitdiff = a.iter().zip(&b).filter(|(x,y)| x.to_bits() != y.to_bits()).count();
            println!("fa_decode_rows vs per-row loop base={base_len} T={t}: bitdiff={bitdiff} {}",
                     if bitdiff == 0 {"OK"} else {fails+=1;"FAIL"});
        }
    }

    // --- KV-cache quantization round-trip: append-quantize then dequant (matches §A formulas) ---
    // Quantize a known f32 K/V row with the append kernel, read the bytes back, dequant on the CPU
    // via the exact ggml q8_0/q5_1 formulas, compare to the f32 input. Isolates layout/packing bugs
    // (esp. the q5_1 qh ballot) from attention. Includes a 5th-bit-boundary block (15<->16, 31).
    {
        use bw24_gguf::dequant::fp16_to_f32;
        let nblk = 4usize;                 // 4 blocks -> 128 elements
        let kv_dim_k = nblk * 32;
        let kv_dim_v = nblk * 32;
        let k_tok_bytes = (kv_dim_k / 32) * 34;
        let v_tok_bytes = (kv_dim_v / 32) * 24;
        // K input: signed random; V input: includes a block crafted to span the 5th-bit boundary.
        let kin: Vec<f32> = (0..kv_dim_k).map(|i| pr(i + 71) * 1.3).collect();
        let mut vin: Vec<f32> = (0..kv_dim_v).map(|i| pr(i + 91) * 0.7 + 0.1).collect();
        // craft block 1 of V so quantized q5 values hit 0..31 spanning bit-4 (15<->16, 31). With
        // mn=0, mx=31*d, q5(j)=round((v-mn)/d) -> set v[j]=j*step so q5 sweeps 0..31 across the warp.
        let step = 0.05f32;
        for j in 0..32 { vin[32 + j] = j as f32 * step; }
        let kd = e.htod(&kin)?; let vd = e.htod(&vin)?;
        let mut kc = e.alloc_u8(k_tok_bytes)?; let mut vc = e.alloc_u8(v_tok_bytes)?;
        e.append_kv_quantized(&kd, &vd, &mut kc, &mut vc, 0, kv_dim_k, kv_dim_v, k_tok_bytes, v_tok_bytes)?;
        let kbytes = e.dtoh_u8(&kc)?; let vbytes = e.dtoh_u8(&vc)?;
        // CPU dequant of q8_0 (K)
        let f16_to_f32 = |b: &[u8]| -> f32 { fp16_to_f32(u16::from_le_bytes([b[0], b[1]])) };
        let mut k_deq = vec![0f32; kv_dim_k];
        for blk in 0..nblk {
            let base = blk * 34;
            let d = f16_to_f32(&kbytes[base..base + 2]);
            for j in 0..32 {
                let q = kbytes[base + 2 + j] as i8;
                k_deq[blk * 32 + j] = d * q as f32;
            }
        }
        let kerr = maxdiff(&kin, &k_deq);
        // per-block d for K rel tol: q8_0 max abs err <= d/2; report relative to amax.
        let kamax = kin.iter().map(|v| v.abs()).fold(0.0, f32::max).max(1e-6);
        let krel = kerr / kamax;
        println!("kvq q8_0 K round-trip: rel={krel:.2e} {}", if krel < 5e-3 { "OK" } else { fails += 1; "FAIL" });
        // CPU dequant of q5_1 (V)
        let mut v_deq = vec![0f32; kv_dim_v];
        for blk in 0..nblk {
            let base = blk * 24;
            let d = f16_to_f32(&vbytes[base..base + 2]);
            let m = f16_to_f32(&vbytes[base + 2..base + 4]);
            let qh = u32::from_le_bytes([vbytes[base + 4], vbytes[base + 5], vbytes[base + 6], vbytes[base + 7]]);
            let qs = &vbytes[base + 8..base + 24];
            for j in 0..32 {
                let lo = if j < 16 { (qs[j] & 0x0F) as i32 } else { (qs[j - 16] >> 4) as i32 };
                let hi = (((qh >> j) & 1) << 4) as i32;
                let q5 = lo | hi;
                v_deq[blk * 32 + j] = d * q5 as f32 + m;
            }
        }
        let verr = maxdiff(&vin, &v_deq);
        let vamax = vin.iter().map(|v| v.abs()).fold(0.0, f32::max).max(1e-6);
        let vrel = verr / vamax;
        println!("kvq q5_1 V round-trip: rel={vrel:.2e} {}", if vrel < 3e-2 { "OK" } else { fails += 1; "FAIL" });
        // explicit 5th-bit-boundary check on V block 1 (q5 sweeps 0..31).
        let bnd_err = (0..32).map(|j| (vin[32 + j] - v_deq[32 + j]).abs()).fold(0.0, f32::max);
        let bnd_d = step;  // block1 d ~= (31*step - 0)/31 = step
        println!("kvq q5_1 5th-bit boundary: maxerr={bnd_err:.2e} (d~{bnd_d:.2e}) {}",
                 if bnd_err < bnd_d { "OK" } else { fails += 1; "FAIL" });
    }

    // --- BATCHED PROMPT PRIME: batched-rows KV append vs T sequential per-token appends must be
    // BYTE-IDENTICAL (same warp program per (block,token); this pins the (b,tt) grid mapping +
    // token-major row addressing against refactors). Non-trivial T and a non-zero slot base t0.
    {
        let nblk = 4usize;
        let kv_dim_k = nblk * 32;
        let kv_dim_v = nblk * 32;
        let k_tok_bytes = (kv_dim_k / 32) * 34;
        let v_tok_bytes = (kv_dim_v / 32) * 24;
        let (t0, t) = (3usize, 7usize);
        let cap = t0 + t;
        let kin: Vec<f32> = (0..t * kv_dim_k).map(|i| pr(i + 301) * 1.1).collect();
        let vin: Vec<f32> = (0..t * kv_dim_v).map(|i| pr(i + 401) * 0.6 - 0.1).collect();
        let kd = e.htod(&kin)?; let vd = e.htod(&vin)?;
        // (a) reference: T sequential per-token appends (the decode append kernel).
        let mut kc_ref = e.alloc_u8(cap * k_tok_bytes)?; let mut vc_ref = e.alloc_u8(cap * v_tok_bytes)?;
        for i in 0..t {
            let k_row = kd.slice(i * kv_dim_k..(i + 1) * kv_dim_k);
            let v_row = vd.slice(i * kv_dim_v..(i + 1) * kv_dim_v);
            e.append_kv_quantized_view(&k_row, &v_row, &mut kc_ref, &mut vc_ref, t0 + i,
                                       kv_dim_k, kv_dim_v, k_tok_bytes, v_tok_bytes)?;
        }
        // (b) batched-rows kernel, one launch.
        let mut kc_b = e.alloc_u8(cap * k_tok_bytes)?; let mut vc_b = e.alloc_u8(cap * v_tok_bytes)?;
        e.append_kv_quantized_rows(&kd, &vd, &mut kc_b, &mut vc_b, t0, t,
                                   kv_dim_k, kv_dim_v, k_tok_bytes, v_tok_bytes)?;
        let (kr, kb) = (e.dtoh_u8(&kc_ref)?, e.dtoh_u8(&kc_b)?);
        let (vr, vb) = (e.dtoh_u8(&vc_ref)?, e.dtoh_u8(&vc_b)?);
        // compare only the written slots [t0, t0+t) — the rest is uninitialized alloc garbage.
        let kmis = (t0 * k_tok_bytes..cap * k_tok_bytes).filter(|&i| kr[i] != kb[i]).count();
        let vmis = (t0 * v_tok_bytes..cap * v_tok_bytes).filter(|&i| vr[i] != vb[i]).count();
        println!("kv append rows-vs-loop bit-identity (T={t}, t0={t0}): k_mismatch={kmis} v_mismatch={vmis} {}",
                 if kmis == 0 && vmis == 0 { "OK" } else { fails += 1; "FAIL" });
    }

    // --- EDGE-1 §D.1: fused-router top-k vs the Stage-1 host softmax+sort+renorm (BIT-IDENTITY). ---
    // Synthetic logits [T,256] (no model needed). The host oracle = the exact moe_ffn host path
    // (softmax-256 -> stable DESC top-8 by (prob DESC, idx ASC) -> renorm w/ F16-min clamp). The
    // device kernel must produce IDENTICAL selected indices and weights within 0 ULP. A tie flip
    // changes routing -> would drift the argmax-1178 gate, so this MUST be exact.
    {
        let (t, n_expert, n_used) = (8usize, 256usize, 8usize);
        // include a deliberate exact tie pair so the tiebreak (smallest index wins) is exercised.
        let mut logits: Vec<f32> = (0..t * n_expert).map(|i| pr(i + 123) * 4.0).collect();
        for tok in 0..t { logits[tok * n_expert + 17] = logits[tok * n_expert + 200]; } // tie 17 vs 200
        // host oracle
        let host_route = |row: &[f32]| -> (Vec<i32>, Vec<f32>) {
            let maxl = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut probs = vec![0f32; n_expert];
            let mut den = 0f32;
            for i in 0..n_expert { let x = (row[i] - maxl).exp(); probs[i] = x; den += x; }
            for p in probs.iter_mut() { *p /= den; }
            let mut idx: Vec<usize> = (0..n_expert).collect();
            idx.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]).then(a.cmp(&b)));
            let sel = &idx[..n_used];
            let mut w: Vec<f32> = sel.iter().map(|&i| probs[i]).collect();
            let mut ws: f32 = w.iter().sum();
            ws = ws.max(6.103515625e-5_f32);
            for x in w.iter_mut() { *x /= ws; }
            (sel.iter().map(|&i| i as i32).collect(), w)
        };
        let ld = e.htod(&logits)?;
        let (sel_d, w_d) = e.moe_router_topk(&ld, t, n_expert, n_used)?;
        let sel_g = e.dtoh_i32(&sel_d)?;
        let w_g = e.dtoh(&w_d)?;
        let mut idx_ok = true;
        let mut w_max_rel = 0f32;     // max relative weight diff (host f32::exp vs device expf)
        let mut w_max_ulp = 0i64;     // max ULP gap (informational)
        for tok in 0..t {
            let (sh, wh) = host_route(&logits[tok * n_expert..(tok + 1) * n_expert]);
            for j in 0..n_used {
                if sel_g[tok * n_used + j] != sh[j] { idx_ok = false; }
                let (a, b) = (w_g[tok * n_used + j], wh[j]);
                let rel = (a - b).abs() / b.abs().max(1e-12);
                if rel > w_max_rel { w_max_rel = rel; }
                let ulp = (a.to_bits() as i64 - b.to_bits() as i64).abs();
                if ulp > w_max_ulp { w_max_ulp = ulp; }
            }
        }
        // SELECTION must be exact (a tie flip would drift the argmax-1178 gate). Weights differ only
        // by host-libm-exp vs device-expf last-ULP noise; gate on tiny relative error, report ULP.
        println!("moe_router idx-match (incl. tie 17/200): {}", if idx_ok { "OK" } else { fails += 1; "FAIL" });
        println!("moe_router weight rel={w_max_rel:.2e} (max {w_max_ulp} ULP, host-exp vs device-expf): {}",
                 if w_max_rel < 1e-5 { "OK" } else { fails += 1; "FAIL" });
    }

    // --- EDGE-1 §D.2: cache-HIT bit-identity. Stage an expert into a fresh scratch (stage-every-token)
    // and into a residency-cache slot, run the SAME qmatvec_view from each, assert BITWISE-equal y.
    // Mechanically guaranteed by §B.3 (same bytes, same kernel); this pins it vs a future refactor. ---
    {
        use bw24_gguf::{GgufFile, GgmlType};
        use bw24_engine::moe_cache::{MoeSlotCache, BlockId, PROJ_GATE};
        const GGUF_35B: &str =
            "/home/avifenesh/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf";
        if std::path::Path::new(GGUF_35B).exists() {
            let g = GgufFile::open(GGUF_35B)?;
            let t = g.find("blk.0.ffn_gate_exps.weight").expect("gate_exps");
            let in_f = t.ne[0] as usize; let out_f = t.ne[1] as usize; let n_expert = t.ne[2] as usize;
            let qt_opt = match t.ggml_type {
                GgmlType::IQ3_S => Some(bw24_engine::QT_IQ3_S), GgmlType::IQ4_XS => Some(bw24_engine::QT_IQ4_XS),
                GgmlType::Q6_K => Some(bw24_engine::QT_Q6_K), GgmlType::Q8_0 => Some(bw24_engine::QT_Q8_0),
                other => { println!("D.2 cache: gate_exps {other:?} unhandled — SKIP"); None },
            };
            if let Some(qt) = qt_opt {
                let raw = g.tensor_data(t);
                let expert_stride = raw.len() / n_expert;
                let row_bytes = raw.len() / (out_f * n_expert);
                let ex = 5usize; // arbitrary expert
                let host_bytes = &raw[ex * expert_stride..(ex + 1) * expert_stride];
                let x: Vec<f32> = (0..in_f).map(|i| pr(i + 999) * 0.1).collect();
                let xd = e.htod(&x)?;
                // (a) stage-every-token: fresh scratch
                let mut scratch = e.alloc_u8(expert_stride)?;
                e.stage_expert(host_bytes, &mut scratch, 0)?;
                let y_stage = e.dtoh(&e.qmatvec_view(&scratch, 0..expert_stride, &xd.slice(0..in_f), 1,
                    in_f, out_f, qt, row_bytes)?)?;
                // (b) residency cache: force-admit, then qmatvec_view from the resident slot.
                let mut cache = MoeSlotCache::new(&e, expert_stride)?;
                let id = BlockId::new(0, PROJ_GATE, ex as u16);
                let slot = cache.force_admit(id, host_bytes, &e)?;
                let y_hit = e.dtoh(&e.qmatvec_view(cache.slot(slot), 0..expert_stride, &xd.slice(0..in_f), 1,
                    in_f, out_f, qt, row_bytes)?)?;
                // also exercise the dispatch() HIT path (second access should be Resident).
                let _ = cache.dispatch(id, host_bytes, &e)?;
                let bitwise = y_stage.iter().zip(&y_hit).all(|(a, b)| a.to_bits() == b.to_bits());
                println!("moe cache-HIT bit-identity (stage==cache): {}",
                         if bitwise { "OK" } else { fails += 1; "FAIL" });
            }
        } else {
            println!("D.2 cache bit-identity: 35B GGUF absent — SKIP");
        }
    }

    if fails == 0 { println!("\nALL GREEN: kernels match CPU reference."); Ok(()) }
    else { Err(format!("{fails} kernel(s) FAILED").into()) }
}
