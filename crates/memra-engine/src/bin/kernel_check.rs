//! M1 gate: validate each Stage-1 kernel against a CPU reference. Run before wiring the forward.

use memra_validate::{maxdiff, pr};
use memra_engine::Engine;

/// Weight-oracle artifact resolution (lane/kc-paths, 2026-08-01). The dtype5/D.2/Q8MMQ/G12/G27
/// sections used to pin 5090-rig absolute paths (/home/avifenesh/..., /data/...), so they
/// silently SKIPped on every other box — H100 rounds 44-47 ran the battery blind on exactly
/// the models that lane fights over. Chain, first existing path wins:
///   1. $MEMRA_KC_MODELS_DIR/<file>                       (explicit; battery scripts set this)
///   2. the CLI gguf arg, when its basename == <file>     (model under test doubles as oracle)
///   3. $HOME/models/<file>, /opt/dlami/nvme/models/<file> (bench-box conventions)
///   4. the legacy rig paths                              (the 5090 rig keeps working naked)
/// A miss prints ONE loud actionable line — a skipped section must always be visible in the
/// battery log and name the env that enables it, never silent.
fn kc_model(section: &str, fname: &str, legacy: &[&str], gguf_arg: &Option<String>) -> Option<String> {
    let mut cands: Vec<String> = Vec::new();
    if let Ok(d) = std::env::var("MEMRA_KC_MODELS_DIR") {
        cands.push(format!("{}/{fname}", d.trim_end_matches('/')));
    }
    if let Some(a) = gguf_arg {
        if std::path::Path::new(a).file_name().map(|f| f == fname).unwrap_or(false) {
            cands.push(a.clone());
        }
    }
    if let Ok(h) = std::env::var("HOME") {
        cands.push(format!("{h}/models/{fname}"));
    }
    cands.push(format!("/opt/dlami/nvme/models/{fname}"));
    cands.extend(legacy.iter().map(|s| s.to_string()));
    if let Some(p) = cands.iter().find(|p| std::path::Path::new(p).exists()) {
        return Some(p.clone());
    }
    println!(
        "KC-SKIP [{section}] {fname}: absent on this box ({} candidates tried) — \
         set MEMRA_KC_MODELS_DIR=<dir containing it> to run this section",
        cands.len()
    );
    None
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let e = Engine::new(0)?;
    println!("GPU: {}", e.ctx().name()?);
    let mut fails = 0;

    // Weight-oracle sections mmap real GGUF tensors; an HF safetensors dir has none, so those
    // sections skip (the synthetic checks above them cover the kernel math either way).
    let gguf_arg: Option<String> = std::env::args().nth(1).filter(|p| {
        let is_dir = std::path::Path::new(p).is_dir();
        if is_dir {
            println!("(arg is an HF safetensors dir — GGUF weight-oracle sections will be skipped; \
                      pass a GGUF path to run them)");
        }
        !is_dir
    });

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

    // --- shexp gate fused sigmoid-dot (qwen35moe decode: g[tok] = sigmoid(dot(x[tok],w))) ---
    {
        let (n_embd, t) = (2048usize, 3usize);
        let x: Vec<f32> = (0..t * n_embd).map(|i| pr(i + 13) - 0.5).collect();
        let w: Vec<f32> = (0..n_embd).map(|i| pr(i + 41) - 0.5).collect();
        let mut cpu = vec![0f32; t];
        for r in 0..t {
            let s: f32 = (0..n_embd).map(|i| x[r * n_embd + i] * w[i]).sum();
            cpu[r] = 1.0 / (1.0 + (-s).exp());
        }
        let xd = e.htod(&x)?; let wd = e.htod(&w)?;
        let gd = e.sigmoid_dot_rows(&xd, &wd, n_embd, t)?;
        let gpu = e.dtoh(&gd)?;
        let d = maxdiff(&cpu, &gpu);
        println!("sigmoid_dot  maxdiff={d:.2e} {}", if d < 1e-5 { "OK" } else { fails += 1; "FAIL" });
    }

    // --- warp-per-row qkv norm (MEMRA_QKVNORM_W, prefill rows>=64): CPU-oracle gate on the
    // rms_norm_qkv dispatch at prefill depth (picks rms_norm_qkv_w4_f32). Own numeric config
    // (float4-lane reduce order) -> f32-band tolerance vs CPU, not bit-identity. ---
    {
        let (hd, nh, nkv, t) = (512usize, 4usize, 1usize, 32usize);
        let eps = 1e-6f32;
        let (rq, rk) = (nh * t, nkv * t);
        let q: Vec<f32> = (0..rq * hd).map(|i| pr(i + 29)).collect();
        let k: Vec<f32> = (0..rk * hd).map(|i| pr(i + 31)).collect();
        let v: Vec<f32> = (0..rk * hd).map(|i| pr(i + 37)).collect();
        let wq: Vec<f32> = (0..hd).map(|i| 0.5 + pr(i + 41) * 0.1).collect();
        let wk: Vec<f32> = (0..hd).map(|i| 0.5 + pr(i + 43) * 0.1).collect();
        let wv: Vec<f32> = vec![1.0; hd];
        let cpu_norm = |x: &[f32], w: &[f32], rows: usize| -> Vec<f32> {
            let mut o = vec![0f32; rows * hd];
            for r in 0..rows {
                let xr = &x[r * hd..(r + 1) * hd];
                let ms: f32 = xr.iter().map(|v| v * v).sum::<f32>() / hd as f32;
                let s = 1.0 / (ms + eps).sqrt();
                for i in 0..hd { o[r * hd + i] = xr[i] * s * w[i]; }
            }
            o
        };
        let (cq, ck, cv) = (cpu_norm(&q, &wq, rq), cpu_norm(&k, &wk, rk), cpu_norm(&v, &wv, rk));
        let qd = e.htod(&q)?; let kd = e.htod(&k)?; let vd = e.htod(&v)?;
        let wqd = e.htod(&wq)?; let wkd = e.htod(&wk)?; let wvd = e.htod(&wv)?;
        let mut dq = e.zeros(rq * hd)?; let mut dk = e.zeros(rk * hd)?; let mut dv = e.zeros(rk * hd)?;
        e.rms_norm_qkv(&qd, &kd, &vd, &wqd, &wkd, &wvd, &mut dq, &mut dk, &mut dv, hd, rq, rk, eps)?;
        let d = maxdiff(&cq, &e.dtoh(&dq)?)
            .max(maxdiff(&ck, &e.dtoh(&dk)?))
            .max(maxdiff(&cv, &e.dtoh(&dv)?));
        println!("rms_norm_qkv_w4 (prefill rows) maxdiff={d:.2e} {}",
                 if d < 1e-4 { "OK" } else { fails += 1; "FAIL" });
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

    // --- FUSED ACT-EPILOGUE (MoE prefill MMA arms): mmq_iq_fused_act_quant must produce a
    //     BYTE-IDENTICAL block_q8_1_mmq D4 scratch to the two-pass chain
    //     moe_pairs_{silu,gelu}_mul -> mmq_iq_quantize_act. Covers both activations and the
    //     ragged/padded in_f (gemma 704 -> GGML_PAD 512-multiple zero tail — the padded-k
    //     down-GEMM contract rides those zero bytes). ANY nonzero diff = FAIL. ---
    for (name, in_f, n_pairs, act_kind) in [("silu", 768usize, 33usize, 0i32),
                                            ("silu", 512, 7, 0),
                                            ("gelu", 704, 29, 1)] {
        let n = n_pairs * in_f;
        let g: Vec<f32> = (0..n).map(|i| pr(i + 17) * 4.0).collect();
        let u: Vec<f32> = (0..n).map(|i| pr(i + 29) * 4.0).collect();
        let gd = e.htod(&g)?;
        let ud = e.htod(&u)?;
        // two-pass reference: f32 act buffer, then the D4 quantizer re-reads it.
        let act = if act_kind == 0 { e.moe_pairs_silu_mul(&gd, &ud, n)? }
                  else { e.moe_pairs_gelu_mul(&gd, &ud, n)? };
        let scr_ref = e.mmq_iq_quantize_act(&act, in_f, n_pairs)?;
        // fused: activation in registers, only the quantized scratch is written.
        let scr_f = e.mmq_iq_fused_act_quant(&gd, &ud, in_f, n_pairs, act_kind)?;
        let b_ref: Vec<u8> = e.stream().clone_dtoh(&scr_ref)?;
        let b_f: Vec<u8> = e.stream().clone_dtoh(&scr_f)?;
        e.stream().synchronize()?;
        let nbad = b_ref.iter().zip(&b_f).filter(|(a, b)| a != b).count();
        println!("iq fused act+quant [{name} in_f={in_f} n_pairs={n_pairs}]: \
                  byte_mismatch={nbad}/{} {}",
                 b_ref.len(), if nbad == 0 && b_ref.len() == b_f.len() { "OK" }
                              else { fails += 1; "FAIL" });
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
            // pin the f32 chunked form explicitly (the default may be the mma config on the
            // Hopper lane — both configs stay pinned regardless of the shipped default).
            // SAFETY: single-threaded gate binary; the seam reads the env per call.
            unsafe { std::env::set_var("MEMRA_GDN_MMA", "0"); }
            unsafe { std::env::set_var("MEMRA_GDN_WGMMA", "0"); }
            e.gdn_scan_chunked(&qd, &kd, &vd, &gd, &bd, None, None, &sid, &mut so_c, &mut o_c, h, t, scale, c, h)?;
            unsafe { std::env::remove_var("MEMRA_GDN_MMA"); }
            let (ro_s, rs_s) = (relerr(&o64, &e.dtoh(&o_s)?), relerr(&s64, &e.dtoh(&so_s)?));
            let (ro_c, rs_c) = (relerr(&o64, &e.dtoh(&o_c)?), relerr(&s64, &e.dtoh(&so_c)?));
            let ok = ro_c < 1e-4 && rs_c < 2.5e-4
                  && ro_c <= (ro_s * 32.0).max(1e-6) && rs_c <= (rs_s * 32.0).max(1e-6);
            println!("gdn_chunked  T={t:3} C={c:3} vs f64-truth: out seq={ro_s:.2e}/chunk={ro_c:.2e} \
                      state seq={rs_s:.2e}/chunk={rs_c:.2e} {}",
                     if ok { "OK" } else { fails += 1; "FAIL" });
            // K4-MMA config pin (MEMRA_GDN_MMA opt-in, c==32 only): its OWN band — bf16
            // operand rounding measures ~4.3e-2 out / ~4.3e-1 state vs f64 truth on these
            // hostile synthetics (2026-07-26). The band guards the mma config against
            // REGRESSIONS; the f32 pin above stays the default's safety line.
            if c == 32 && cfg!(memra_hopper_mma) {
                // SAFETY: single-threaded gate binary; the seam reads the env per call.
                unsafe { std::env::set_var("MEMRA_GDN_MMA", "1"); }
                let mut so_m = e.zeros(s_v * s_v * h)?; let mut o_m = e.zeros(s_v * h * t)?;
                e.gdn_scan_chunked(&qd, &kd, &vd, &gd, &bd, None, None, &sid, &mut so_m, &mut o_m, h, t, scale, c, h)?;
                let (ro_m, rs_m) = (relerr(&o64, &e.dtoh(&o_m)?), relerr(&s64, &e.dtoh(&so_m)?));
                let okm = ro_m < 8e-2 && rs_m < 8e-1;
                println!("gdn_chunked  T={t:3} C={c:3} MMA config pin: out={ro_m:.2e} state={rs_m:.2e} {}",
                         if okm { "OK" } else { fails += 1; "FAIL" });
                // K4+K5 fused wgmma config pin (MEMRA_GDN_WGMMA, task #22): its OWN band.
                // State shares the mma bf16 class (measured 5.3e-1 on these hostile
                // synthetics, band 8e-1). OUT is a WIDER class than K5-mma: the fused
                // phase 1 stages q/M as bf16 (wgmma) where K5-mma staged fp16 (2 fewer
                // mantissa bits) — measured 2.19e-1 here, band 4e-1 (~2x headroom, the
                // mma-pin precedent). Tail chunks verified separately (harness T=200
                // in-band, O rel 5.6e-3); model-level gates: 3-seed greedy IDENTICAL,
                // chunked-continuation IDENTICAL, argmax PASS (2026-07-27).
                unsafe { std::env::set_var("MEMRA_GDN_WGMMA", "1"); }
                let mut so_w = e.zeros(s_v * s_v * h)?; let mut o_w = e.zeros(s_v * h * t)?;
                e.gdn_scan_chunked(&qd, &kd, &vd, &gd, &bd, None, None, &sid, &mut so_w, &mut o_w, h, t, scale, c, h)?;
                unsafe { std::env::remove_var("MEMRA_GDN_MMA"); }
                let (ro_w, rs_w) = (relerr(&o64, &e.dtoh(&o_w)?), relerr(&s64, &e.dtoh(&so_w)?));
                let okw = ro_w < 4e-1 && rs_w < 8e-1;
                println!("gdn_chunked  T={t:3} C={c:3} WGMMA-fused config pin: out={ro_w:.2e} state={rs_w:.2e} {}",
                         if okw { "OK" } else { fails += 1; "FAIL" });
            }
            unsafe { std::env::remove_var("MEMRA_GDN_WGMMA"); }
        }
    }

    // --- Q2_K Stage-A GPU path vs the CPU dequant oracle on deterministic synthetic blocks. ---
    // Q2_K intentionally has no dp4a fast path yet, but mixed expert artifacts rely on this
    // generic staged path. Keep this model-independent so every target-rig gate exercises it.
    {
        use memra_gguf::{GgmlType, dequant};
        use memra_runtime::cpu_linear;
        let (in_f, out_f, m, row_bytes) = (256usize, 7usize, 3usize, 84usize);
        let mut raw = vec![0u8; out_f * row_bytes];
        for row in 0..out_f {
            let base = row * row_bytes;
            for group in 0..16 {
                let scale = 1 + ((row * 3 + group * 5) % 15) as u8;
                let min = 1 + ((row * 7 + group * 2) % 15) as u8;
                raw[base + group] = scale | (min << 4);
            }
            for byte in 0..64 {
                raw[base + 16 + byte] = ((row * 41 + byte * 17 + 13) & 0xff) as u8;
            }
            raw[base + 80..base + 82].copy_from_slice(&0x2c00u16.to_le_bytes()); // f16 0.0625
            raw[base + 82..base + 84].copy_from_slice(&0x2800u16.to_le_bytes()); // f16 0.03125
        }
        let weights = dequant::dequantize(GgmlType::Q2_K, &raw, in_f * out_f);
        let x: Vec<f32> = (0..m * in_f).map(|i| pr(i + 79) * 0.1).collect();
        let cpu = cpu_linear(&x, &weights, m, in_f, out_f);
        let wd = e.htod_bytes(&raw)?;
        let xd = e.htod(&x)?;
        let gpu = e.dtoh(&e.qmatvec(
            &wd, &xd, m, in_f, out_f, memra_engine::QT_Q2_K, row_bytes,
        )?)?;
        let scale = cpu.iter().map(|value| value.abs()).fold(0.0, f32::max).max(1e-3);
        let rel = maxdiff(&cpu, &gpu) / scale;
        println!("qmatvec Q2_K synthetic Stage-A: rel={rel:.2e} {}",
                 if rel < 1e-4 { "OK" } else { fails += 1; "FAIL" });
    }

    // --- qmatvec (resident-quant GEMM) vs cpu_linear(dequant(W)) on real GGUF weights ---
    if let Some(path) = gguf_arg.clone() {
        use memra_gguf::{GgufFile, GgmlType, dequant};
        use memra_runtime::cpu_linear;
        let g = GgufFile::open(&path)?;
        let cases = [
            ("blk.0.ffn_gate.weight", memra_engine::QT_Q8_0),   // exists in every layer
            ("blk.0.attn_qkv.weight", memra_engine::QT_Q8_0),   // linear-attn layer
            ("blk.3.attn_q.weight", memra_engine::QT_Q8_0),     // full-attn layer (il=3)
            ("blk.0.attn_v.weight", memra_engine::QT_Q6_K),     // Q6_K in 1.7B
            ("output.weight", memra_engine::QT_Q6_K),           // Q6_K lm_head in 1.7B
            ("token_embd.weight", memra_engine::QT_Q8_0),
        ];
        for (tname, _) in cases {
            if let Some(t) = g.find(tname) {
                let qt = match t.ggml_type {
                    GgmlType::Q8_0 => memra_engine::QT_Q8_0,
                    GgmlType::Q4_K => memra_engine::QT_Q4_K,
                    GgmlType::Q6_K => memra_engine::QT_Q6_K,
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
    if let Some(path) = gguf_arg.clone() {
        use memra_gguf::{GgufFile, GgmlType};
        let g = GgufFile::open(&path)?;
        if let Some(t) = g.find("blk.0.ffn_gate.weight").filter(|t| t.ggml_type == GgmlType::Q8_0) {
            let in_f = t.ne[0] as usize; let out_f = t.ne[1] as usize;
            let raw = g.tensor_data(t); let row_bytes = raw.len() / out_f;
            let m = 2usize;
            let x: Vec<f32> = (0..m * in_f).map(|i| pr(i + 41) * 0.1).collect();
            let wd = e.htod_bytes(raw)?; let xd = e.htod(&x)?;
            let ya = e.dtoh(&e.qmatvec(&wd, &xd, m, in_f, out_f, memra_engine::QT_Q8_0, row_bytes)?)?;
            let yb = e.dtoh(&e.qmatvec_q8_0_fast(&wd, &xd, m, in_f, out_f, row_bytes)?)?;
            let d = maxdiff(&ya, &yb);
            let scale = ya.iter().map(|v| v.abs()).fold(0.0, f32::max).max(1e-3);
            let rel = d / scale;
            // int8 activation quant => expect ~1% rel error, not 1e-7. Gate: rel < 3e-2.
            println!("qmatvec_q8_0_fast vs Stage-A: rel={rel:.2e} {}", if rel < 3e-2 { "OK" } else { fails += 1; "FAIL" });
            println!("  (ya[0..3]={:?} yb[0..3]={:?})", &ya[..3], &yb[..3]);
        }
        // Q4_K + Q6_K fast paths vs Stage-A oracle (int8-act tolerance).
        for (tname, qt) in [("blk.0.attn_q.weight", memra_engine::QT_Q4_K),
                            ("blk.0.attn_v.weight", memra_engine::QT_Q6_K),
                            ("output.weight", memra_engine::QT_Q6_K)] {
            if let Some(t) = g.find(tname) {
                let gt = match t.ggml_type { GgmlType::Q4_K => memra_engine::QT_Q4_K, GgmlType::Q6_K => memra_engine::QT_Q6_K, _ => continue };
                if gt != qt { continue; }
                let in_f = t.ne[0] as usize; let out_f = t.ne[1] as usize;
                let raw = g.tensor_data(t); let row_bytes = raw.len() / out_f;
                let m = 2usize;
                let x: Vec<f32> = (0..m * in_f).map(|i| pr(i + 51) * 0.1).collect();
                let wd = e.htod_bytes(raw)?; let xd = e.htod(&x)?;
                let ya = e.dtoh(&e.qmatvec(&wd, &xd, m, in_f, out_f, gt, row_bytes)?)?;
                let yb = if gt == memra_engine::QT_Q4_K { e.dtoh(&e.qmatvec_q4_K_fast(&wd, &xd, m, in_f, out_f, row_bytes)?)? }
                         else { e.dtoh(&e.qmatvec_q6_K_fast(&wd, &xd, m, in_f, out_f, row_bytes)?)? };
                let d = maxdiff(&ya, &yb);
                let scale = ya.iter().map(|v| v.abs()).fold(0.0, f32::max).max(1e-3);
                let rel = d / scale;
                println!("{tname} [{:?}] fast vs Stage-A: rel={rel:.2e} {}", t.ggml_type, if rel < 3e-2 { "OK" } else { fails += 1; "FAIL" });
            }
        }
    }

    // --- 5 new dtypes: GPU qmatvec vs memra CPU-dequant oracle on REAL daily-GGUF tensors. ---
    // Oracle = cpu_linear(memra_dequant(W), x); memra's CPU dequant is byte-for-byte == ggml
    // dequantize_row_<type> (proven in memra-gguf example dequant_oracle_diff), so this gates
    // the GPU paths against ggml ground truth transitively. Mirrors the Q4_K/Q6_K block above:
    //   Stage-A (dequant-in-kernel) rel < 1e-4 ; Stage-B (int8 dp4a) rel < 3e-2.
    // IQ3_S has NO dp4a fast path (intentional, see lib.rs) -> Stage-A only.
    // Skips LOUDLY (kc_model) if a daily GGUF is absent so the core gate still runs in CI
    // without models — and a box missing the artifact shows the miss in its battery log.
    {
        use memra_gguf::{GgufFile, GgmlType, dequant};
        use memra_runtime::cpu_linear;
        let gguf_9b = kc_model("dtype5", "Qwen3.5-9B-NVFP4-MTP-GGUF.gguf",
            &["/home/avifenesh/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf"],
            &gguf_arg);
        let gguf_35b = kc_model("dtype5", "Qwen3.6-35B-A3B-UD-IQ4_XS.gguf",
            &["/home/avifenesh/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"],
            &gguf_arg);
        // (gguf, tensor, expected type, QT code, fast-path selector or "" for Stage-A only)
        let cases: [(&Option<String>, &str, GgmlType, i32, &str); 5] = [
            (&gguf_9b,  "blk.0.ffn_gate.weight",      GgmlType::NVFP4,  memra_engine::QT_NVFP4,  "nvfp4"),
            (&gguf_9b,  "blk.0.attn_gate.weight",     GgmlType::Q5_K,   memra_engine::QT_Q5_K,   "q5k"),
            (&gguf_35b, "blk.0.ffn_gate_exps.weight", GgmlType::IQ3_S,  memra_engine::QT_IQ3_S,  ""),
            (&gguf_35b, "blk.0.ffn_down_exps.weight", GgmlType::IQ4_XS, memra_engine::QT_IQ4_XS, "iq4xs"),
            (&gguf_35b, "blk.40.ffn_gate_exps.weight",GgmlType::Q3_K,   memra_engine::QT_Q3_K,   "q3k"),
        ];
        for (path, tname, gty, qt, sel) in cases {
            let Some(path) = path.as_deref() else { continue };   // kc_model already skipped loudly
            let g = GgufFile::open(path)?;
            let t = match g.find(tname).filter(|t| t.ggml_type == gty) {
                Some(t) => t,
                // The pinned tensor can be absent or re-typed in another REVISION of the same
                // artifact (the H100 box's 35B copy lacks the rig copy's blk.40 MTP layer — its
                // Q3_K source; found by this gate 2026-08-01). The case exists to gate the DTYPE
                // against ggml ground truth, the name is just a known carrier — substitute the
                // smallest same-dtype weight so the dtype stays gated on this box; only a file
                // with NO such tensor skips, and loudly. Numeric thresholds below are unchanged.
                None => match g.tensors.iter()
                        .filter(|t| t.ggml_type == gty && t.ne.len() >= 2 && t.ne[1] > 1
                                && t.name.ends_with(".weight"))
                        .min_by_key(|t| t.n_bytes) {
                    Some(t) => {
                        println!("dtype5 {gty:?}: pinned {tname} absent/re-typed in this artifact \
                                  revision — substituting {}", t.name);
                        t
                    }
                    None => {
                        println!("KC-SKIP [dtype5] {path}: no {gty:?} .weight tensor at all \
                                  (pinned {tname} absent) — this artifact revision lacks the dtype");
                        continue;
                    }
                }
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
    if let Some(path) = gguf_arg.clone() {
        use memra_gguf::{GgufFile, GgmlType};
        let g = GgufFile::open(&path)?;
        // (tensor, GEMM qt, dp4a-fast selector). Each is validated if present with the right type.
        let gemm_cases: [(&str, i32, &str); 6] = [
            ("blk.0.ffn_gate.weight",  memra_engine::QT_Q8_0,  "q8_0"),  // 35B token_embd-style Q8_0
            ("blk.0.attn_qkv.weight",  memra_engine::QT_Q8_0,  "q8_0"),
            ("blk.3.attn_q.weight",    memra_engine::QT_Q4_K,  "q4_K"),  // 9B/27B attn Q4_K
            ("blk.0.ssm_out.weight",   memra_engine::QT_Q5_K,  "q5_K"),  // q27 GDN out Q5_K
            ("blk.0.attn_v.weight",    memra_engine::QT_Q6_K,  "q6_K"),
            ("output.weight",          memra_engine::QT_Q6_K,  "q6_K"),  // Q6_K lm_head
        ];
        for (tname, want_qt, sel) in gemm_cases {
            let t = match g.find(tname) { Some(t) => t, None => continue };
            let gt = match t.ggml_type {
                GgmlType::Q8_0 => memra_engine::QT_Q8_0, GgmlType::Q4_K => memra_engine::QT_Q4_K,
                GgmlType::Q6_K => memra_engine::QT_Q6_K, GgmlType::NVFP4 => memra_engine::QT_NVFP4,
                GgmlType::Q5_K => memra_engine::QT_Q5_K,
                _ => continue,
            };
            if gt != want_qt { continue; }
            let in_f = t.ne[0] as usize; let out_f = t.ne[1] as usize;
            if t.ne.len() > 2 { continue; } // skip 3D MoE expert tensors here
            let raw = g.tensor_data(t); let row_bytes = raw.len() / out_f;
            let wd = e.htod_bytes(raw)?;
            // H100 wgmma arm (task 8): mirror built once per tensor; compared vs the mma
            // kernel inside the T loop (same numeric class -> same rel<1e-3 band).
            let wgmma_mirror = if cfg!(memra_hopper_mma) && gt == memra_engine::QT_Q8_0
                && out_f % 64 == 0 && in_f % 32 == 0 {
                Some(e.build_q8_rp4_raw(&wd, in_f, out_f)?)
            } else { None };
            // f16-mirror coverage per admitted class: Q8_0 (2026-07-26), Q4_K + Q5_K
            // (round 49 — the q27 trunk bulk + ssm_out), Q6_K (round 47; entry added
            // round 49 with Q4_K — the "gates outside the battery rot" law).
            let f16_mirror = if gt == memra_engine::QT_Q8_0 && in_f % 32 == 0 {
                Some(e.build_q8_f16_raw(&wd, in_f, out_f)?)
            } else if gt == memra_engine::QT_Q4_K && in_f % 256 == 0 {
                Some(e.build_q4k_f16_raw(&wd, in_f, out_f)?)
            } else if gt == memra_engine::QT_Q5_K && in_f % 256 == 0 {
                Some(e.build_q5k_f16_raw(&wd, in_f, out_f)?)
            } else if gt == memra_engine::QT_Q6_K && in_f % 256 == 0 {
                Some(e.build_q6k_f16_raw(&wd, in_f, out_f)?)
            } else { None };
            for tt in [16usize, 64, 128, 512] {
                let x: Vec<f32> = (0..tt * in_f).map(|i| pr(i + 71) * 0.1).collect();
                let xd = e.htod(&x)?;
                let ydp = match sel {
                    "q8_0" => e.qmatvec_q8_0_fast(&wd, &xd, tt, in_f, out_f, row_bytes)?,
                    "q4_K" => e.qmatvec_q4_K_fast(&wd, &xd, tt, in_f, out_f, row_bytes)?,
                    "q5_K" => e.qmatvec_q5_K_fast(&wd, &xd, tt, in_f, out_f, row_bytes)?,
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
                if let Some(mirror) = &wgmma_mirror {
                    let (aq, ad) = e.quantize_q8_1(&xd, tt, in_f)?;
                    let yw = e.dtoh(&e.qmatvec_gemm_q8_0_wgmma_raw(mirror, &aq, &ad, tt, in_f, out_f)?)?;
                    let dw = maxdiff(&yb, &yw);
                    let relw = dw / scale;
                    println!("GEMM {tname} wgmma T={tt}: rel={relw:.2e} {}",
                             if relw < 1e-3 { "OK" } else { fails += 1; "FAIL" });
                }
                if let Some(m16) = &f16_mirror {
                    // FP16-mirror GEMM (MEMRA_PP_F16 numeric config): fp16 products + f32
                    // accumulate vs the s32-exact + per-32 f32-fold law — a WIDER band than
                    // the int8 arms by design (rounding at d*q and the activation cast).
                    let yf = e.dtoh(&e.qmatvec_gemm_f16_raw(m16, &xd, tt, in_f, out_f)?)?;
                    let df = maxdiff(&yb, &yf);
                    let relf = df / scale;
                    println!("GEMM {tname} f16 T={tt}: rel={relf:.2e} {}",
                             if relf < 1e-2 { "OK" } else { fails += 1; "FAIL" });
                }
            }
        }
    }

    // --- f16g mode-2 sk grouped GEMM (rounds 49+51): grid-scan vs visitor forms. Synthetic ---
    // CSR with q35-like routing skew (group sizes 1..300 — the ~17x skew shape that drove the
    // round-51 visitor). Two gates per case:
    //   (a) the round-49 grid-scan kernel vs the f32 CPU oracle on the SAME f16 operands
    //       (values snapped to an f16-exact grid, so only f32-accumulate order differs);
    //   (b) every round-51 visitor form (hybrid split / all-128 / all-32) vs the grid-scan
    //       kernel BYTE-IDENTICAL — each output element's k-chain is the same ascending
    //       mma.sync m16n8k16 sequence by construction, so maxdiff MUST be exactly 0.
    // Case 2's in_f=480 (%32 but not %64) forces the sk128 in_f fallback: force-128 must
    // silently ride the 32x64 form and stay byte-identical.
    {
        fn f16_bits(x: f32) -> u16 {
            let b = x.to_bits();
            let s = ((b >> 16) & 0x8000) as u16;
            if x == 0.0 { return s; }
            let he = ((b >> 23) & 0xff) as i32 - 127 + 15; // test values are moderate normals
            let m = b & 0x7f_ffff;
            let mut h = ((he as u32) << 10) | (m >> 13);
            let rem = m & 0x1fff;
            if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) { h += 1; }
            s | h as u16
        }
        let m_sizes: [i32; 8] = [1, 3, 17, 33, 64, 129, 200, 300];
        let n_active = m_sizes.len();
        let mut ex_off_host = vec![0i32; n_active + 1];
        for (g, m) in m_sizes.iter().enumerate() { ex_off_host[g + 1] = ex_off_host[g] + m; }
        let n_pairs = *ex_off_host.last().unwrap() as usize;
        let snap = |v: f32| (v * 256.0).round() / 256.0;
        for (in_f, out_f) in [(512usize, 300usize), (480, 192)] {
            let w_f32: Vec<f32> = (0..n_active * out_f * in_f)
                .map(|i| snap(pr(i + 101) - 0.5)).collect();
            let a_f32: Vec<f32> = (0..n_pairs * in_f)
                .map(|i| snap(pr(i + 211) - 0.5)).collect();
            let scales: Vec<f32> = (0..n_pairs).map(|p| 1.0 + (p % 5) as f32 * 0.25).collect();
            let mut cpu = vec![0f32; n_pairs * out_f];
            for g in 0..n_active {
                let (lo, hi) = (ex_off_host[g] as usize, ex_off_host[g + 1] as usize);
                for p in lo..hi {
                    let arow = &a_f32[p * in_f..][..in_f];
                    for o in 0..out_f {
                        let wrow = &w_f32[(g * out_f + o) * in_f..][..in_f];
                        let s: f32 = wrow.iter().zip(arow).map(|(w, a)| w * a).sum();
                        cpu[p * out_f + o] = s * scales[p];
                    }
                }
            }
            let to_bytes = |v: &[f32]| -> Vec<u8> {
                v.iter().flat_map(|&x| f16_bits(x).to_le_bytes()).collect()
            };
            let wd = e.htod_bytes(&to_bytes(&w_f32))?;
            let ad = e.htod_bytes(&to_bytes(&a_f32))?;
            let sd = e.htod(&scales)?;
            let offd = e.htod_i32(&ex_off_host)?;
            let y_legacy = e.dtoh(&e.moe_f16g_gemm_sk_raw(&wd, &ad, &sd, &ex_off_host, &offd,
                                                          in_f, out_f, n_pairs, -1, 0)?)?;
            let scale = cpu.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-3);
            let rel = maxdiff(&cpu, &y_legacy) / scale;
            println!("f16g-sk (in={in_f} out={out_f} skew 1..300) grid-scan vs oracle: \
                      rel={rel:.2e} {}",
                     if rel < 1e-3 { "OK" } else { fails += 1; "FAIL" });
            for (name, cross) in [("visitor-hybrid(cross=64)", 64),
                                  ("visitor-128", 1), ("visitor-32", i32::MAX)] {
                let yv = e.dtoh(&e.moe_f16g_gemm_sk_raw(&wd, &ad, &sd, &ex_off_host, &offd,
                                                        in_f, out_f, n_pairs, 0, cross)?)?;
                let d = maxdiff(&y_legacy, &yv);
                println!("f16g-sk (in={in_f} out={out_f}) {name} vs grid-scan: maxdiff={d:.2e} {}",
                         if d == 0.0 { "OK (byte-identical)" } else { fails += 1; "FAIL" });
            }
        }
    }

    // --- f16g-kq-direct (lane/kquant-tile-loaders): DIRECT-FROM-QUANT Q4_K/Q6_K sk tile
    // loaders vs the dequant-workspace path. The direct kernels dequant B tiles in-register
    // from the quant superblocks (kq_q4k_val/kq_q6k_val — the workspace dequant kernels'
    // exact expressions), so every output element's mma k-chain consumes the same f16
    // operands in the same order: maxdiff MUST be exactly 0 (bitwise), per visitor form.
    // Synthetic blocks first (random nibbles/scales, safe-normal f16 d/dmin fields), then
    // real Ornith-35B expert weights below (weight-oracle section).
    {
        let m_sizes: [i32; 8] = [1, 3, 17, 33, 64, 129, 200, 300];
        let n_active = m_sizes.len();
        let mut ex_off_host = vec![0i32; n_active + 1];
        for (g, m) in m_sizes.iter().enumerate() { ex_off_host[g + 1] = ex_off_host[g] + m; }
        let n_pairs = *ex_off_host.last().unwrap() as usize;
        let (in_f, out_f) = (512usize, 300usize);   // 2 superblocks/row; ragged out tile
        let n_expert = n_active;
        for (qname, qtype, sbb) in [("q4_K", memra_engine::QT_Q4_K, 144usize),
                                    ("q6_K", memra_engine::QT_Q6_K, 210usize)] {
            let row_bytes = in_f / 256 * sbb;
            let ex_bytes = out_f * row_bytes;
            // Synthetic superblocks: random payload bytes; the f16 scale fields (q4k d/dmin
            // at +0/+2, q6k d at +208) overwritten with small positive normals (0x2C00 band)
            // so no NaN/Inf enters the mirror.
            let mut slab = vec![0u8; n_expert * ex_bytes];
            for (i, b) in slab.iter_mut().enumerate() {
                *b = (pr(i + 313) * 256.0) as u8;
            }
            for ex in 0..n_expert {
                for r in 0..out_f {
                    for s in 0..(in_f / 256) {
                        let off = ex * ex_bytes + r * row_bytes + s * sbb;
                        let seed = ex * 131 + r * 7 + s;
                        let h = |k: usize| -> [u8; 2] {
                            (0x2C00u16 + ((pr(seed + k) * 512.0) as u16)).to_le_bytes()
                        };
                        if qtype == memra_engine::QT_Q4_K {
                            slab[off..off + 2].copy_from_slice(&h(1));
                            slab[off + 2..off + 4].copy_from_slice(&h(2));
                        } else {
                            slab[off + 208..off + 210].copy_from_slice(&h(1));
                        }
                    }
                }
            }
            let slab_d = e.htod_bytes(&slab)?;
            let base = {
                use cudarc::driver::DevicePtr;
                let s = e.stream();
                let (p, _g) = slab_d.device_ptr(&s);
                p as u64
            };
            let tab: Vec<u64> = (0..n_expert).map(|ex| base + (ex * ex_bytes) as u64).collect();
            let tab_d = e.htod_u64(&tab)?;
            // active experts in REVERSED id order — exercises the ex_ids indirection.
            let ex_ids: Vec<i32> = (0..n_active as i32).rev().collect();
            let exi_d = e.htod_i32(&ex_ids)?;
            let act: Vec<u8> = (0..n_pairs * in_f).flat_map(|i| {
                let h = (0x2C00u16 + ((pr(i + 619) * 4096.0) as u16))
                    | (((i & 1) as u16) << 15);
                h.to_le_bytes()
            }).collect();
            let ad = e.htod_bytes(&act)?;
            let scales: Vec<f32> = (0..n_pairs).map(|p| 0.5 + pr(p + 733)).collect();
            let sd = e.htod(&scales)?;
            let offd = e.htod_i32(&ex_off_host)?;
            let ws = e.moe_f16g_dequant_raw(&tab_d, 0, n_expert, &exi_d,
                                            in_f, out_f, n_active, qtype, row_bytes)?;
            for (name, cross) in [("hybrid(cross=64)", 64),
                                  ("all-128", 1), ("all-32", i32::MAX)] {
                let y_ws = e.dtoh(&e.moe_f16g_gemm_sk_raw(&ws, &ad, &sd, &ex_off_host, &offd,
                                                          in_f, out_f, n_pairs, 0, cross)?)?;
                let y_dq = e.dtoh(&e.moe_kq_gemm_sk_raw(&tab_d, 0, n_expert, &exi_d, &ad, &sd,
                                                        &ex_off_host, &offd, in_f, out_f,
                                                        n_pairs, qtype, row_bytes, cross)?)?;
                let d = maxdiff(&y_ws, &y_dq);
                println!("f16g-kq-direct [{qname} synth in={in_f} out={out_f}] {name} \
                          vs workspace: maxdiff={d:.2e} {}",
                         if d == 0.0 { "OK (byte-identical)" } else { fails += 1; "FAIL" });
            }
        }
    }
    // f16g-kq-direct on REAL weights: Ornith-35B Q4_K gate_exps + Q6_K down_exps slices.
    {
        use memra_gguf::{GgufFile, GgmlType};
        let o35b = kc_model("f16g-kq-direct", "ornith-1.0-35b-Q4_K_M.gguf",
            &["/data/ai-ml/hf-models/ornith-1.0-35b-gguf/ornith-1.0-35b-Q4_K_M.gguf"],
            &gguf_arg);
        if let Some(path) = o35b.as_deref() {
            let g = GgufFile::open(path)?;
            // one Q4_K expert tensor + one Q6_K one (down flips qtype per layer; scan for it).
            let mut cases: Vec<(String, i32, usize)> = Vec::new();
            if let Some(t) = g.find("blk.0.ffn_gate_exps.weight")
                .filter(|t| t.ggml_type == GgmlType::Q4_K) {
                let _ = t; cases.push(("blk.0.ffn_gate_exps.weight".into(),
                                       memra_engine::QT_Q4_K, 144));
            }
            for l in 0..48 {
                let name = format!("blk.{l}.ffn_down_exps.weight");
                if g.find(&name).map(|t| t.ggml_type == GgmlType::Q6_K).unwrap_or(false) {
                    cases.push((name, memra_engine::QT_Q6_K, 210));
                    break;
                }
            }
            let m_sizes: [i32; 6] = [5, 33, 64, 80, 129, 17];
            let n_active = m_sizes.len();
            let mut ex_off_host = vec![0i32; n_active + 1];
            for (gg, m) in m_sizes.iter().enumerate() { ex_off_host[gg + 1] = ex_off_host[gg] + m; }
            let n_pairs = *ex_off_host.last().unwrap() as usize;
            for (tname, qtype, sbb) in cases {
                let t = g.find(&tname).unwrap();
                let (in_f, out_f, ne) = (t.ne[0] as usize, t.ne[1] as usize, t.ne[2] as usize);
                if in_f % 256 != 0 || ne < n_active {
                    println!("f16g-kq-direct [{tname}] SKIP (in_f={in_f} ne={ne})");
                    continue;
                }
                let row_bytes = in_f / 256 * sbb;
                let ex_bytes = out_f * row_bytes;
                let raw = g.tensor_data(t);
                let slab_d = e.htod_bytes(&raw[..n_active * ex_bytes])?;
                let base = {
                    use cudarc::driver::DevicePtr;
                    let s = e.stream();
                    let (p, _gg) = slab_d.device_ptr(&s);
                    p as u64
                };
                let tab: Vec<u64> = (0..n_active).map(|ex| base + (ex * ex_bytes) as u64).collect();
                let tab_d = e.htod_u64(&tab)?;
                let ex_ids: Vec<i32> = (0..n_active as i32).collect();
                let exi_d = e.htod_i32(&ex_ids)?;
                let act: Vec<u8> = (0..n_pairs * in_f).flat_map(|i| {
                    let h = (0x2C00u16 + ((pr(i + 619) * 4096.0) as u16))
                        | (((i & 1) as u16) << 15);
                    h.to_le_bytes()
                }).collect();
                let ad = e.htod_bytes(&act)?;
                let scales: Vec<f32> = (0..n_pairs).map(|p| 0.5 + pr(p + 733)).collect();
                let sd = e.htod(&scales)?;
                let offd = e.htod_i32(&ex_off_host)?;
                let ws = e.moe_f16g_dequant_raw(&tab_d, 0, n_active, &exi_d,
                                                in_f, out_f, n_active, qtype, row_bytes)?;
                for (name, cross) in [("hybrid(cross=64)", 64), ("all-128", 1)] {
                    let y_ws = e.dtoh(&e.moe_f16g_gemm_sk_raw(&ws, &ad, &sd, &ex_off_host,
                                          &offd, in_f, out_f, n_pairs, 0, cross)?)?;
                    let y_dq = e.dtoh(&e.moe_kq_gemm_sk_raw(&tab_d, 0, n_active, &exi_d, &ad,
                                          &sd, &ex_off_host, &offd, in_f, out_f,
                                          n_pairs, qtype, row_bytes, cross)?)?;
                    let d = maxdiff(&y_ws, &y_dq);
                    println!("f16g-kq-direct [{tname} in={in_f} out={out_f}] {name} \
                              vs workspace: maxdiff={d:.2e} {}",
                             if d == 0.0 { "OK (byte-identical)" } else { fails += 1; "FAIL" });
                }
            }
        }
    }

    // --- IQ4_XS dense-trunk MMQ (lane/kquant-tile-loaders): the m>=16 int8-MMA dense GEMM
    // vs the dp4a fast path (the m=1..15 decode/verify program). Same q8_1 per-32 activation
    // grid; MMA f32 fold order differs -> tolerance band (the other MMQ arms' convention),
    // not bit-identity. Synthetic blocks + a real KAT-Coder trunk tensor.
    {
        let iq4xs_gate = |e: &Engine, wd: &_, in_f: usize, out_f: usize, row_bytes: usize,
                          label: &str, fails: &mut i32|
            -> Result<(), Box<dyn std::error::Error>> {
            for tt in [16usize, 64, 128, 512] {
                let x: Vec<f32> = (0..tt * in_f).map(|i| pr(i + 47) * 0.1).collect();
                let xd = e.htod(&x)?;
                let ya = e.dtoh(&e.qmatvec_iq4_XS_fast(wd, &xd, tt, in_f, out_f, row_bytes)?)?;
                let yb = e.dtoh(&e.qmatvec_mmq_iq4xs_raw(wd, &xd, tt, in_f, out_f, row_bytes)?)?;
                let d = maxdiff(&ya, &yb);
                let scale = ya.iter().map(|v| v.abs()).fold(0.0, f32::max).max(1e-3);
                let rel = d / scale;
                println!("iq4xs-mmq [{label}] T={tt}: rel={rel:.2e} {}",
                         if rel < 1e-3 { "OK" } else { *fails += 1; "FAIL" });
            }
            Ok(())
        };
        // synthetic: random payload, safe-normal f16 d field per 136B superblock.
        {
            let (in_f, out_f) = (512usize, 300usize);
            let row_bytes = in_f / 256 * 136;
            let mut w = vec![0u8; out_f * row_bytes];
            for (i, b) in w.iter_mut().enumerate() { *b = (pr(i + 409) * 256.0) as u8; }
            for r in 0..out_f {
                for s in 0..(in_f / 256) {
                    let off = r * row_bytes + s * 136;
                    let h = 0x2C00u16 + ((pr(r * 7 + s + 3) * 512.0) as u16);
                    w[off..off + 2].copy_from_slice(&h.to_le_bytes());
                }
            }
            let wd = e.htod_bytes(&w)?;
            iq4xs_gate(&e, &wd, in_f, out_f, row_bytes, "synth", &mut fails)?;
        }
        // real KAT trunk tensor (first 2-D IQ4_XS with in_f%256==0, out_f>=128).
        {
            use memra_gguf::{GgufFile, GgmlType};
            let kat = kc_model("iq4xs-mmq", "Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf",
                &["/data/ai-ml/hf-models/kat-coder-v25-dev-gguf/Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf"],
                &gguf_arg);
            if let Some(path) = kat.as_deref() {
                let g = GgufFile::open(path)?;
                if let Some(t) = g.tensors.iter().find(|t| {
                    t.ggml_type == GgmlType::IQ4_XS && t.ne.len() == 2
                        && t.ne[0] as usize % 256 == 0 && t.ne[1] >= 128
                }) {
                    let (in_f, out_f) = (t.ne[0] as usize, t.ne[1] as usize);
                    let raw = g.tensor_data(t);
                    let row_bytes = raw.len() / out_f;
                    let wd = e.htod_bytes(raw)?;
                    iq4xs_gate(&e, &wd, in_f, out_f, row_bytes, &t.name, &mut fails)?;
                }
            }
        }
    }
    // NVFP4 GEMM vs dp4a on the 9B model (separate path: per-tensor macro-scale + in_f%64).
    {
        use memra_gguf::{GgufFile, GgmlType};
        // Resolve the first existing NVFP4 model (9B preferred, 27B-MTP fallback). The gates
        // below filter by tensor name+type, so a model that lacks a given tensor just skips it.
        let gguf_9b_owned = kc_model("nvfp4-gemm", "Qwen3.5-9B-NVFP4-MTP-GGUF.gguf",
            &["/home/avifenesh/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf",
              "/home/ubuntu/memra-bench/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf"],
            &gguf_arg)
        .or_else(|| kc_model("nvfp4-gemm", "Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf",
            &["/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf"],
            &gguf_arg));
        if let Some(gguf_9b) = gguf_9b_owned.as_deref() {
            let g = GgufFile::open(gguf_9b)?;
            // Q5_K GEMM vs dp4a (attn_gate is Q5_K in 9B).
            if let Some(t) = g.find("blk.0.attn_gate.weight").filter(|t| t.ggml_type == GgmlType::Q5_K) {
                let in_f = t.ne[0] as usize; let out_f = t.ne[1] as usize;
                let raw = g.tensor_data(t); let row_bytes = raw.len() / out_f;
                let wd = e.htod_bytes(raw)?;
                for tt in [16usize, 64, 128, 512] {
                    let x: Vec<f32> = (0..tt * in_f).map(|i| pr(i + 91) * 0.1).collect();
                    let xd = e.htod(&x)?;
                    let ya = e.dtoh(&e.qmatvec_q5_K_fast(&wd, &xd, tt, in_f, out_f, row_bytes)?)?;
                    let yb = e.dtoh(&e.qmatvec_gemm_raw(&wd, &xd, tt, in_f, out_f, memra_engine::QT_Q5_K, row_bytes)?)?;
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
                    let yb = e.dtoh(&e.qmatvec_gemm_raw(&wd, &xd, tt, in_f, out_f, memra_engine::QT_NVFP4, row_bytes)?)?;
                    let d = maxdiff(&ya, &yb);
                    let scale = ya.iter().map(|v| v.abs()).fold(0.0, f32::max).max(1e-3);
                    let rel = d / scale;
                    println!("GEMM blk.0.ffn_gate.weight [NVFP4] T={tt}: rel={rel:.2e} {}",
                             if rel < 1e-3 { "OK" } else { fails += 1; "FAIL" });
                }
            }
            // sm_89 (pure portable) skips everything here. sm_90a (portable + hopper_mma)
            // SKIPS only the NVFP4-family checks (fail-closed stubs there) but MUST run the
            // Q4_K/Q8_0/Q4_0 MMQ checks — those kernels are live on Hopper through the
            // hopper_mma re-admission, and the old whole-section skip left the battery
            // blind to the #23 stream-K corruption (2026-07-31).
            let nvfp4_checks = !cfg!(memra_portable_cuda);
            if cfg!(memra_portable_cuda) && !cfg!(memra_hopper_mma) {
                println!("portable CUDA: native FP4 and static-MMQ model-backed checks — SKIP");
            } else {
            if nvfp4_checks {
            // Stage-C FP4 (mxf4nvf4 block-scale tensor-core) vs the f32 dequant oracle on NVFP4.
            // FP4 is LOSSY (e2m1 activations + e2m1 weights; scale side is lossless ue4m3) — NOT
            // bit-equivalent. Compare to cpu_linear(dequant(W)) and expect rel ~1e-2..6e-2.
            if let Some(t) = g.find("blk.0.ffn_gate.weight").filter(|t| t.ggml_type == GgmlType::NVFP4) {
                use memra_gguf::dequant;
                use memra_runtime::cpu_linear;
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
                    // (MEMRA_FP4 run-hybrid/run-gen), which holds on the 9B and is the arbiter per the plan.
                    println!("FP4-GEMM blk.0.ffn_gate.weight [NVFP4] T={tt}: rel={rel:.2e} (informational; \
                              authoritative gate = argmax) {}", if rel < 2e-1 { "OK" } else { "HIGH" });
                }
            }
            // --- VENDORED llama NVFP4 MMQ GEMM vs the f32 dequant oracle. ---
            // W4A4-native (mxf4nvf4 block-scale mma) but with llama's 2-level FP8-e8m0/UE4M3 activation
            // quant -> should be MUCH closer to the f32 oracle than the memra hand-roll FP4 (rel ~0.1).
            // Authoritative gate is still end-to-end argmax; this rel is the accuracy signal that
            // llama's activation quant fixed memra's W4A4 maxdiff 1.46.
            if let Some(t) = g.find("blk.0.ffn_gate.weight").filter(|t| t.ggml_type == GgmlType::NVFP4) {
                use memra_gguf::dequant;
                use memra_runtime::cpu_linear;
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
            // --- STAGE 2: VENDORED llama NVFP4 W4A8 MMQ GEMM vs the f32 dequant oracle. ---
            // The accuracy-safe rung: weight FP4 is LUT-dequantized to int8 (bit-exact) and the
            // activation stays q8_1 int8 -> rel MUST sit in the int8-activation band (~1e-3..1e-2),
            // NOT the 0.1 W4A4 band. This is a HARD gate (2e-2) — the whole point of the rung is that
            // it holds the int8 accuracy class the default GEMM passes all e2e gates with.
            if let Some(t) = g.find("blk.0.ffn_gate.weight").filter(|t| t.ggml_type == GgmlType::NVFP4) {
                use memra_gguf::dequant;
                use memra_engine::model::repack_nvfp4_split;
                use memra_runtime::cpu_linear;
                let in_f = t.ne[0] as usize; let out_f = t.ne[1] as usize;
                let raw = g.tensor_data(t);
                let w_f32 = dequant::dequantize(GgmlType::NVFP4, raw, in_f * out_f);
                let wd = e.htod_bytes(raw)?;
                // A6 split-plane copy of the SAME weight — the rp tile loader must be BIT-identical.
                let wd_rp = e.htod_bytes(&repack_nvfp4_split(raw, out_f))?;
                for tt in [16usize, 64, 128, 512] {
                    let x: Vec<f32> = (0..tt * in_f).map(|i| pr(i + 83) * 0.1).collect();
                    let xd = e.htod(&x)?;
                    let cpu = cpu_linear(&x, &w_f32, tt, in_f, out_f);
                    let yb = e.dtoh(&e.qmatvec_mmq_nvfp4_w4a8_raw(&wd, &xd, tt, in_f, out_f)?)?;
                    let d = maxdiff(&cpu, &yb);
                    let scale = cpu.iter().map(|v| v.abs()).fold(0.0, f32::max).max(1e-3);
                    let rel = d / scale;
                    println!("MMQ-W4A8 blk.0.ffn_gate.weight [NVFP4] T={tt}: rel={rel:.2e} (int8 band ~1e-3) {}",
                             if rel < 2e-2 { "OK" } else { fails += 1; "FAIL" });
                    // rp-loader BIT-IDENTITY gate: split-plane loader vs GGUF loader on the same
                    // weight+activation must agree on every f32 bit (pure address remap, same FP
                    // ops in the same order). ANY nonzero diff = layout bug = HARD FAIL.
                    let yr = e.dtoh(&e.qmatvec_mmq_nvfp4_w4a8_raw_rp(&wd_rp, &xd, tt, in_f, out_f)?)?;
                    let nbad = yb.iter().zip(yr.iter())
                        .filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                    println!("MMQ-W4A8-RP blk.0.ffn_gate.weight [NVFP4] T={tt}: bit-mismatch {nbad}/{} {}",
                             yb.len(), if nbad == 0 { "OK" } else { fails += 1; "FAIL" });
                }
            }
            } // nvfp4_checks
            // --- VENDORED llama Q4_K/Q5_K MMQ GEMM vs the f32 dequant oracle. ---
            // W-exact (int8 tile-load dequant is lossless for k-quants) + q8_1 int8 activation ->
            // rel should sit in the int8-activation band (~1e-3..1e-2). A layout/scale bug shows as
            // rel ~1.0, so a 2e-2 hard gate catches real breakage without flapping on quant noise.
            for (tname, want, qt) in [("blk.3.attn_q.weight",    GgmlType::Q4_K, memra_engine::QT_Q4_K),
                                      ("blk.0.attn_gate.weight", GgmlType::Q5_K, memra_engine::QT_Q5_K)] {
                let Some(t) = g.find(tname).filter(|t| t.ggml_type == want) else { continue };
                use memra_gguf::dequant;
                use memra_runtime::cpu_linear;
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
            #[cfg(memra_cutlass)]
            if let Some(t) = g.find("blk.0.ffn_gate.weight").filter(|t| t.ggml_type == GgmlType::NVFP4) {
                use memra_gguf::dequant;
                use memra_runtime::cpu_linear;
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
            // The three sections below were HOISTED out of the `if let Some(gguf_9b)`
            // NVFP4 block above (lane/kc-paths, 2026-08-01): they gate Q8_0-MMQ/q4_0-MMQ/
            // 27B-shape oracles that do NOT need the 9B NVFP4 artifact, but the nesting
            // silently disabled them on every box without it (the same blindness class
            // as the hardcoded paths).
            // --- VENDORED llama Q8_0 MMQ GEMM (MEMRA_PP_Q8MMQ) vs the f32 dequant oracle. ---
            // Q8_0 weight IS int8 (lossless tile-load) + q8_1 D4 activation -> same int8-activation
            // band as q45k (~1e-3..1e-2). 2e-2 hard gate. Uses the 35B model's Q8_0 projections.
            {
                let g35_path = kc_model("q8mmq-gemm", "Qwen3.6-35B-A3B-UD-IQ4_XS.gguf",
                    &["/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf",
                      "/home/avifenesh/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"],
                    &gguf_arg);
                if let Some(g35_path) = g35_path {
                    let g35 = GgufFile::open(&g35_path)?;
                    use memra_gguf::dequant;
                    use memra_runtime::cpu_linear;
                    for tname in ["blk.0.attn_qkv.weight", "blk.0.ffn_gate_shexp.weight"] {
                        let Some(t) = g35.find(tname).filter(|t| t.ggml_type == GgmlType::Q8_0) else { continue };
                        let in_f = t.ne[0] as usize; let out_f = t.ne[1] as usize;
                        let raw = g35.tensor_data(t);
                        let w_f32 = dequant::dequantize(GgmlType::Q8_0, raw, in_f * out_f);
                        let wd = e.htod_bytes(raw)?;
                        for tt in [16usize, 64, 128, 512] {
                            let x: Vec<f32> = (0..tt * in_f).map(|i| pr(i + 53) * 0.1).collect();
                            let xd = e.htod(&x)?;
                            let cpu = cpu_linear(&x, &w_f32, tt, in_f, out_f);
                            let yb = e.dtoh(&e.qmatvec_mmq_q8_0_raw(&wd, &xd, tt, in_f, out_f)?)?;
                            let d = maxdiff(&cpu, &yb);
                            let scale = cpu.iter().map(|v| v.abs()).fold(0.0, f32::max).max(1e-3);
                            let rel = d / scale;
                            println!("MMQ-Q8_0 {tname} [Q8_0 in={in_f} out={out_f}] T={tt}: rel={rel:.2e} {}",
                                     if rel < 2e-2 { "OK" } else { fails += 1; "FAIL" });
                        }
                    }
                }
            }
            // --- VENDORED llama Q4_0 MMQ GEMM (MEMRA_PP_Q4MMQ) vs the f32 dequant oracle. ---
            // Nibble->int8 tile-load dequant is lossless ((q-8) exact in int8) + q8_1 D4 activation
            // -> same int8-activation band as Q8_0 (~1e-3..1e-2). 2e-2 hard gate. Uses the 12B
            // gemma QAT q4_0 projections. Also gates the rp split-plane loader BIT-identical to
            // the raw-18B-block loader (pure address remap, same FP ops in the same order).
            {
                let g12_path = kc_model("q4_0-mmq", "gemma-4-12b-it-qat-q4_0.gguf",
                    &["/data/ai-ml/models/gemma-4-12b-it-qat/gemma-4-12b-it-qat-q4_0.gguf"],
                    &gguf_arg);
                // Host mirror of q4_0_split_rp_build: qs plane (16B/block, block-major) then fp16
                // d plane (2B/block) at out_f*nblk*16.
                fn repack_q4_0_split(raw: &[u8], nblocks: usize) -> Vec<u8> {
                    let mut out = vec![0u8; nblocks * 18];
                    let dplane = nblocks * 16;
                    for i in 0..nblocks {
                        let b = &raw[i * 18..i * 18 + 18];
                        out[i * 16..i * 16 + 16].copy_from_slice(&b[2..18]);
                        out[dplane + i * 2] = b[0];
                        out[dplane + i * 2 + 1] = b[1];
                    }
                    out
                }
                if let Some(g12_path) = g12_path {
                    let g12 = GgufFile::open(&g12_path)?;
                    use memra_gguf::dequant;
                    use memra_runtime::cpu_linear;
                    for tname in ["blk.0.attn_q.weight", "blk.0.ffn_gate.weight"] {
                        let Some(t) = g12.find(tname).filter(|t| t.ggml_type == GgmlType::Q4_0) else { continue };
                        let in_f = t.ne[0] as usize; let out_f = t.ne[1] as usize;
                        let raw = g12.tensor_data(t);
                        let w_f32 = dequant::dequantize(GgmlType::Q4_0, raw, in_f * out_f);
                        let wd = e.htod_bytes(raw)?;
                        let wd_rp = e.htod_bytes(&repack_q4_0_split(raw, out_f * in_f / 32))?;
                        for tt in [16usize, 64, 128, 512] {
                            let x: Vec<f32> = (0..tt * in_f).map(|i| pr(i + 59) * 0.1).collect();
                            let xd = e.htod(&x)?;
                            let cpu = cpu_linear(&x, &w_f32, tt, in_f, out_f);
                            let yb = e.dtoh(&e.qmatvec_mmq_q4_0_raw(&wd, &xd, tt, in_f, out_f, false)?)?;
                            let d = maxdiff(&cpu, &yb);
                            let scale = cpu.iter().map(|v| v.abs()).fold(0.0, f32::max).max(1e-3);
                            let rel = d / scale;
                            println!("MMQ-Q4_0 {tname} [Q4_0 in={in_f} out={out_f}] T={tt}: rel={rel:.2e} {}",
                                     if rel < 2e-2 { "OK" } else { fails += 1; "FAIL" });
                            let yr = e.dtoh(&e.qmatvec_mmq_q4_0_raw(&wd_rp, &xd, tt, in_f, out_f, true)?)?;
                            let nbad = yb.iter().zip(yr.iter())
                                .filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                            println!("MMQ-Q4_0-RP {tname} T={tt}: bit-mismatch {nbad}/{} {}",
                                     yb.len(), if nbad == 0 { "OK" } else { fails += 1; "FAIL" });
                        }
                    }
                }
            }
            // 27B ffn_down NVFP4 shape probe (in_f=17408 not a clean MMQ_ITER_K_FP4 multiple? T=512)
            // — compare MMQ vs the dp4a oracle to isolate the 27B T=513 mismatch.
            {
                let g27_path = kc_model("nvfp4-27b-shape", "Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf",
                    &["/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf"],
                    &gguf_arg);
                if let Some(g27_path) = g27_path {
                    let g27 = GgufFile::open(&g27_path)?;
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
            // #23 regression (2026-07-31): the 26B a4b shared-MLP shape (in=2816, out=2112 —
            // out % 128 != 0 -> need_check=true clamped last row-tile) through the FORCED
            // stream-K arm. On the H100 board this shape+SK produced garbage prefill logits
            // above the mb=256 autotune bucket while the xy-tiling form was exact; the
            // timing-picked autotune hid the arm from the 5090 battery. Force both forms
            // deterministically and pin each against the CPU reference.
            {
                let g26_path = kc_model("q4_0-sk-arm", "gemma-4-26B_q4_0-it.gguf",
                    &["/data/ai-ml/hf-models/gemma4-26b-a4b-qat-gguf/gemma-4-26B_q4_0-it.gguf"],
                    &gguf_arg);
                if let Some(g26_path) = g26_path {
                    let g26 = GgufFile::open(&g26_path)?;
                    use memra_gguf::dequant;
                    use memra_runtime::cpu_linear;
                    // synthetic ragged-k, nc=false twin (in=2112 -> 66 blocks, out=2560 = 20*128):
                    // separates the ragged-k mechanism from the clamped-last-row (need_check) one.
                    {
                        let (in_f, out_f) = (2112usize, 2560usize);
                        let nblk = in_f / 32 * out_f;
                        let mut raw = vec![0u8; nblk * 18];
                        for (bi, b) in raw.chunks_mut(18).enumerate() {
                            b[0] = 0x00; b[1] = 0x3C;   // d = f16 1.0
                            for k in 0..16 { b[2 + k] = ((bi * 31 + k * 7) % 251) as u8; }
                        }
                        let w_f32 = dequant::dequantize(GgmlType::Q4_0, &raw, in_f * out_f);
                        let wd = e.htod_bytes(&raw)?;
                        for tt in [103usize, 479] {
                            let x: Vec<f32> = (0..tt * in_f).map(|i| pr(i + 83) * 0.1).collect();
                            let xd = e.htod(&x)?;
                            let cpu = cpu_linear(&x, &w_f32, tt, in_f, out_f);
                            let scale = cpu.iter().map(|v| v.abs()).fold(0.0, f32::max).max(1e-3);
                            for (force, label) in [(0i8, "TILE"), (1, "SK")] {
                                memra_engine::MMQ_SK_FORCE.store(force, std::sync::atomic::Ordering::Relaxed);
                                let yb = e.dtoh(&e.qmatvec_mmq_q4_0_raw(&wd, &xd, tt, in_f, out_f, false)?)?;
                                let rel = maxdiff(&cpu, &yb) / scale;
                                println!("MMQ-Q4_0-RAGK {label} [in={in_f} out={out_f} nc=false] T={tt}: rel={rel:.2e} {}",
                                         if rel < 2e-2 { "OK" } else { fails += 1; "FAIL" });
                            }
                            memra_engine::MMQ_SK_FORCE.store(-1, std::sync::atomic::Ordering::Relaxed);
                        }
                    }
                    for tname in ["blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight",
                                  "blk.0.attn_output.weight", "blk.0.ffn_gate.weight",
                                  "blk.0.ffn_down.weight"] {
                        let Some(t) = g26.find(tname).filter(|t| t.ggml_type == GgmlType::Q4_0) else { continue };
                        let in_f = t.ne[0] as usize; let out_f = t.ne[1] as usize;
                        let raw = g26.tensor_data(t);
                        let w_f32 = dequant::dequantize(GgmlType::Q4_0, raw, in_f * out_f);
                        let wd = e.htod_bytes(raw)?;
                        for tt in [103usize, 229, 479, 1024, 2048, 2151] {
                            let x: Vec<f32> = (0..tt * in_f).map(|i| pr(i + 83) * 0.1).collect();
                            let xd = e.htod(&x)?;
                            let cpu = cpu_linear(&x, &w_f32, tt, in_f, out_f);
                            let scale = cpu.iter().map(|v| v.abs()).fold(0.0, f32::max).max(1e-3);
                            for (force, label) in [(0i8, "TILE"), (1, "SK")] {
                                memra_engine::MMQ_SK_FORCE.store(force, std::sync::atomic::Ordering::Relaxed);
                                let yb = e.dtoh(&e.qmatvec_mmq_q4_0_raw(&wd, &xd, tt, in_f, out_f, false)?)?;
                                let rel = maxdiff(&cpu, &yb) / scale;
                                println!("MMQ-Q4_0-NC26 {tname} {label} [in={in_f} out={out_f}] T={tt}: rel={rel:.2e} {}",
                                         if rel < 2e-2 { "OK" } else { fails += 1; "FAIL" });
                            }
                            memra_engine::MMQ_SK_FORCE.store(-1, std::sync::atomic::Ordering::Relaxed);
                        }
                    }
                }
            }
    }

    // --- PERF-3 MMVQ (warp-per-row decode) vs dp4a matvec: BIT-EQUIVALENCE gate. ---
    // The _mmvq kernels lift the dequant body VERBATIM from _dp4a; only layout (warp-per-row) +
    // reduction (warp-only shfl) change -> int sumi identical, only f32 reduction-order rounding
    // differs. Require rel < 1e-3. m=1 (decode regime) across in_f ∈ {model shapes} and out_f
    // small + 4096. Q8_0/Q4_K/Q6_K on the model-path arg; NVFP4 on the 9B model below.
    if let Some(path) = gguf_arg.clone() {
        use memra_gguf::{GgufFile, GgmlType};
        let g = GgufFile::open(&path)?;
        let mmvq_cases: [(&str, i32, &str); 5] = [
            ("blk.0.ffn_gate.weight",  memra_engine::QT_Q8_0,  "q8_0"),
            ("blk.0.attn_qkv.weight",  memra_engine::QT_Q8_0,  "q8_0"),
            ("blk.3.attn_q.weight",    memra_engine::QT_Q4_K,  "q4_K"),
            ("blk.0.attn_v.weight",    memra_engine::QT_Q6_K,  "q6_K"),
            ("output.weight",          memra_engine::QT_Q6_K,  "q6_K"),
        ];
        for (tname, want_qt, sel) in mmvq_cases {
            let t = match g.find(tname) { Some(t) => t, None => continue };
            let gt = match t.ggml_type {
                GgmlType::Q8_0 => memra_engine::QT_Q8_0, GgmlType::Q4_K => memra_engine::QT_Q4_K,
                GgmlType::Q6_K => memra_engine::QT_Q6_K, GgmlType::NVFP4 => memra_engine::QT_NVFP4,
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
    // --- Q8 TRUNK-FUSION (fused2/fused3) vs per-tensor MMVQ: BIT-IDENTITY gate. The fused kernels
    // run q8_0_mmvq_row1 (the qmatvec_q8_0_mmvq body verbatim, t=0) per (tensor,row) with only the
    // grid split changed -> outputs must be BIT-IDENTICAL (rel == 0.0) to separate m=1 launches.
    // Uses the model's real Q8_0 tensors when >=2 same-in_f ones exist (35B: attn_qkv+attn_gate
    // uneven pair + wq/wk/wv triple; other GGUFs: any same-in_f q8_0 pair). ---
    if let Some(path) = gguf_arg.clone() {
        use memra_gguf::{GgufFile, GgmlType};
        let g = GgufFile::open(&path)?;
        // candidate name sets, first (pair) and (triple) that fully resolve as Q8_0 win.
        let pair_sets: [(&str, &str); 3] = [
            ("blk.0.attn_qkv.weight",  "blk.0.attn_gate.weight"),   // 35B uneven 8192/4096
            ("blk.0.ffn_gate_shexp.weight", "blk.0.ffn_up_shexp.weight"), // 35B even 512/512
            ("blk.0.ssm_beta.weight",  "blk.0.ssm_alpha.weight"),   // 9B tiny 32/32
        ];
        let grab = |name: &str| -> Option<(usize, usize, usize, Vec<u8>)> {
            let t = g.find(name)?;
            if t.ggml_type != GgmlType::Q8_0 || t.ne.len() > 2 { return None; }
            let in_f = t.ne[0] as usize; let out_f = t.ne[1] as usize;
            let raw = g.tensor_data(t);
            Some((in_f, out_f, raw.len() / out_f, raw.to_vec()))
        };
        for (n0, n1) in pair_sets {
            let (Some(t0), Some(t1)) = (grab(n0), grab(n1)) else { continue };
            if t0.0 != t1.0 { continue; }
            let (in_f, rb) = (t0.0, t0.2);
            let w0 = e.htod_bytes(&t0.3)?;
            let w1 = e.htod_bytes(&t1.3)?;
            let x: Vec<f32> = (0..in_f).map(|i| pr(i + 131) * 0.1).collect();
            let xd = e.htod(&x)?;
            let r0 = e.dtoh(&e.qmatvec_mmvq_raw(&w0, &xd, 1, in_f, t0.1, memra_engine::QT_Q8_0, rb, false)?)?;
            let r1 = e.dtoh(&e.qmatvec_mmvq_raw(&w1, &xd, 1, in_f, t1.1, memra_engine::QT_Q8_0, rb, false)?)?;
            let (f0, f1) = e.qmatvec_q8_fused2_raw(&w0, &w1, &xd, in_f, t0.1, t1.1, rb)?;
            let (f0, f1) = (e.dtoh(&f0)?, e.dtoh(&f1)?);
            let bits_ok = r0.iter().zip(f0.iter()).all(|(a, b)| a.to_bits() == b.to_bits())
                && r1.iter().zip(f1.iter()).all(|(a, b)| a.to_bits() == b.to_bits());
            let d = maxdiff(&r0, &f0).max(maxdiff(&r1, &f1));
            println!("Q8-FUSED2 {n0}+{n1} [Q8_0] out=({},{}): rel={d:.2e} bits={} {}",
                     t0.1, t1.1, bits_ok,
                     if bits_ok { "OK" } else { fails += 1; "FAIL" });
            // BATCHED twin (verify t=2-4 tier, MEMRA_SPEC_FUSED_T): fused2_b vs the per-tensor
            // _b2/_b4 launches matmul_decode_exact dispatches — body verbatim, must be
            // BIT-IDENTICAL per (tensor,token,row).
            for mm in [2usize, 3, 4] {
                let xm: Vec<f32> = (0..mm * in_f).map(|i| pr(i + 151 + mm) * 0.1).collect();
                let xmd = e.htod(&xm)?;
                let (aq, ad) = e.quantize_q8_1(&xmd, mm, in_f)?;
                let mc = memra_engine::Engine::batched_mcols(mm);
                let r0 = e.dtoh(&e.qmatvec_mmvq_batched(&w0, &aq, &ad, mm, in_f, t0.1,
                                                        memra_engine::QT_Q8_0, rb, mc, 1.0, false)?)?;
                let r1 = e.dtoh(&e.qmatvec_mmvq_batched(&w1, &aq, &ad, mm, in_f, t1.1,
                                                        memra_engine::QT_Q8_0, rb, mc, 1.0, false)?)?;
                let (f0, f1) = e.qmatvec_q8_fused2_t_raw(&w0, &w1, &xmd, mm, in_f, t0.1, t1.1, rb)?;
                let (f0, f1) = (e.dtoh(&f0)?, e.dtoh(&f1)?);
                let bits_ok = r0.iter().zip(f0.iter()).all(|(a, b)| a.to_bits() == b.to_bits())
                    && r1.iter().zip(f1.iter()).all(|(a, b)| a.to_bits() == b.to_bits());
                let d = maxdiff(&r0, &f0).max(maxdiff(&r1, &f1));
                println!("Q8-FUSED2-B {n0}+{n1} [Q8_0] m={mm} out=({},{}): rel={d:.2e} bits={} {}",
                         t0.1, t1.1, bits_ok,
                         if bits_ok { "OK" } else { fails += 1; "FAIL" });
            }
        }
        // triple: 35B full-attn wq/wk/wv (blk.3 is the first full-attn layer).
        let tri: [&str; 3] = ["blk.3.attn_q.weight", "blk.3.attn_k.weight", "blk.3.attn_v.weight"];
        if let (Some(t0), Some(t1), Some(t2)) = (grab(tri[0]), grab(tri[1]), grab(tri[2])) {
            if t0.0 == t1.0 && t1.0 == t2.0 {
                let (in_f, rb) = (t0.0, t0.2);
                let w0 = e.htod_bytes(&t0.3)?;
                let w1 = e.htod_bytes(&t1.3)?;
                let w2 = e.htod_bytes(&t2.3)?;
                let x: Vec<f32> = (0..in_f).map(|i| pr(i + 137) * 0.1).collect();
                let xd = e.htod(&x)?;
                let r0 = e.dtoh(&e.qmatvec_mmvq_raw(&w0, &xd, 1, in_f, t0.1, memra_engine::QT_Q8_0, rb, false)?)?;
                let r1 = e.dtoh(&e.qmatvec_mmvq_raw(&w1, &xd, 1, in_f, t1.1, memra_engine::QT_Q8_0, rb, false)?)?;
                let r2 = e.dtoh(&e.qmatvec_mmvq_raw(&w2, &xd, 1, in_f, t2.1, memra_engine::QT_Q8_0, rb, false)?)?;
                let (f0, f1, f2) = e.qmatvec_q8_fused3_raw(&w0, &w1, &w2, &xd, in_f, t0.1, t1.1, t2.1, rb)?;
                let (f0, f1, f2) = (e.dtoh(&f0)?, e.dtoh(&f1)?, e.dtoh(&f2)?);
                let bits_ok = r0.iter().zip(f0.iter()).all(|(a, b)| a.to_bits() == b.to_bits())
                    && r1.iter().zip(f1.iter()).all(|(a, b)| a.to_bits() == b.to_bits())
                    && r2.iter().zip(f2.iter()).all(|(a, b)| a.to_bits() == b.to_bits());
                let d = maxdiff(&r0, &f0).max(maxdiff(&r1, &f1)).max(maxdiff(&r2, &f2));
                println!("Q8-FUSED3 wq+wk+wv [Q8_0] out=({},{},{}): rel={d:.2e} bits={} {}",
                         t0.1, t1.1, t2.1, bits_ok,
                         if bits_ok { "OK" } else { fails += 1; "FAIL" });
                // BATCHED twin (verify t=2-4 tier): fused3_b vs three per-tensor batched launches.
                for mm in [2usize, 3, 4] {
                    let xm: Vec<f32> = (0..mm * in_f).map(|i| pr(i + 157 + mm) * 0.1).collect();
                    let xmd = e.htod(&xm)?;
                    let (aq, ad) = e.quantize_q8_1(&xmd, mm, in_f)?;
                    let mc = memra_engine::Engine::batched_mcols(mm);
                    let r0 = e.dtoh(&e.qmatvec_mmvq_batched(&w0, &aq, &ad, mm, in_f, t0.1,
                                                            memra_engine::QT_Q8_0, rb, mc, 1.0, false)?)?;
                    let r1 = e.dtoh(&e.qmatvec_mmvq_batched(&w1, &aq, &ad, mm, in_f, t1.1,
                                                            memra_engine::QT_Q8_0, rb, mc, 1.0, false)?)?;
                    let r2 = e.dtoh(&e.qmatvec_mmvq_batched(&w2, &aq, &ad, mm, in_f, t2.1,
                                                            memra_engine::QT_Q8_0, rb, mc, 1.0, false)?)?;
                    let (f0, f1, f2) = e.qmatvec_q8_fused3_t_raw(&w0, &w1, &w2, &xmd, mm, in_f,
                                                                 t0.1, t1.1, t2.1, rb)?;
                    let (f0, f1, f2) = (e.dtoh(&f0)?, e.dtoh(&f1)?, e.dtoh(&f2)?);
                    let bits_ok = r0.iter().zip(f0.iter()).all(|(a, b)| a.to_bits() == b.to_bits())
                        && r1.iter().zip(f1.iter()).all(|(a, b)| a.to_bits() == b.to_bits())
                        && r2.iter().zip(f2.iter()).all(|(a, b)| a.to_bits() == b.to_bits());
                    let d = maxdiff(&r0, &f0).max(maxdiff(&r1, &f1)).max(maxdiff(&r2, &f2));
                    println!("Q8-FUSED3-B wq+wk+wv [Q8_0] m={mm} out=({},{},{}): rel={d:.2e} bits={} {}",
                             t0.1, t1.1, t2.1, bits_ok,
                             if bits_ok { "OK" } else { fails += 1; "FAIL" });
                }
            }
        }
    }
    // NVFP4 MMVQ vs dp4a on the 9B model (in_f%64; macro-scale skipped in both raw paths).
    {
        use memra_gguf::{GgufFile, GgmlType};
        let gguf_9b = kc_model("nvfp4-mmvq", "Qwen3.5-9B-NVFP4-MTP-GGUF.gguf",
            &["/home/avifenesh/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf"],
            &gguf_arg);
        if let Some(gguf_9b) = gguf_9b {
            let g = GgufFile::open(&gguf_9b)?;
            if let Some(t) = g.find("blk.0.ffn_gate.weight").filter(|t| t.ggml_type == GgmlType::NVFP4) {
                let in_f = t.ne[0] as usize; let out_f = t.ne[1] as usize;
                let raw = g.tensor_data(t); let row_bytes = raw.len() / out_f;
                let wd = e.htod_bytes(raw)?;
                for mm in [1usize, 2] {
                    let x: Vec<f32> = (0..mm * in_f).map(|i| pr(i + 111) * 0.1).collect();
                    let xd = e.htod(&x)?;
                    let ya = e.dtoh(&e.qmatvec_nvfp4_fast(&wd, &xd, mm, in_f, out_f, row_bytes)?)?;
                    let yb = e.dtoh(&e.qmatvec_mmvq_raw(&wd, &xd, mm, in_f, out_f, memra_engine::QT_NVFP4, row_bytes, false)?)?;
                    let d = maxdiff(&ya, &yb);
                    let scale = ya.iter().map(|v| v.abs()).fold(0.0, f32::max).max(1e-3);
                    let rel = d / scale;
                    println!("MMVQ blk.0.ffn_gate.weight [NVFP4] m={mm}: rel={rel:.2e} {}",
                             if rel < 1e-3 { "OK" } else { fails += 1; "FAIL" });
                }
            }
        }
    }

    // --- F8-E4M3 matvec (MEMRA_ST_E4M3 decode class, lane e4m3dec): synthetic weights, THREE gates.
    // (1) CPU REFERENCE: qmatvec_e4m3_mmvq vs an f64 CPU dot over the SAME q8_1 activation bytes
    //     (aq/ad read back from the GPU quantizer — the kernel's actual input) and a CPU e4m3
    //     decode. rel < 1e-3 (f32 fmaf chain vs f64; same gate class as the MMVQ checks).
    // (2) DECODE-PARITY: the grid.y=m launch must be BIT-IDENTICAL per (token,row) to the m=1
    //     launch on that token's row (the spec verify==decode law; per-warp body is independent
    //     of blockIdx.y by construction — this gate pins it).
    // (3) BATCHED TWINS: _b2/_b4/_b8 must be BIT-IDENTICAL to the grid.y=m mmvq (weight bytes
    //     read once for all columns; identical fmaf chain per (token,row)). ---
    {
        // CPU e4m3 decode: sign / 4-bit exp (bias 7) / 3-bit mantissa, subnormals (mirrors the
        // KV-format gate's closure; NaN never generated below).
        let e4m3 = |b: u8| -> f32 {
            let s = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
            let ex = ((b >> 3) & 0x0F) as i32;
            let mn = (b & 0x07) as f32;
            if ex == 0 { s * mn * (2f32).powi(-9) }
            else if ex == 15 && mn == 7.0 { f32::NAN }
            else { s * (1.0 + mn / 8.0) * (2f32).powi(ex - 7) }
        };
        let qt = memra_engine::QT_F8_E4M3;
        for (in_f, out_f) in [(5120usize, 512usize), (2048, 320)] {
            // pseudo-random e4m3 bytes; remap the two NaN codes (0x7F/0xFF -> exp field 0xE).
            let wb: Vec<u8> = (0..in_f * out_f).map(|i| {
                let mut b = ((i.wrapping_mul(2654435761) ^ 0x9E3779B9) >> 9) as u8;
                if b & 0x7F == 0x7F { b &= 0xF7; }
                b
            }).collect();
            let wd = e.htod_bytes(&wb)?;
            let row_bytes = in_f;   // raw e4m3: 1 B/element
            for mm in [1usize, 2, 5, 9] {
                let x: Vec<f32> = (0..mm * in_f).map(|i| pr(i + 151) * 0.1).collect();
                let xd = e.htod(&x)?;
                let (aqd, add) = e.quantize_q8_1(&xd, mm, in_f)?;
                let y = e.dtoh(&e.qmatvec_mmvq(&wd, &aqd, &add, mm, in_f, out_f, qt, row_bytes,
                                               1.0, false)?)?;
                // (1) CPU reference from the kernel's exact q8_1 inputs, f64 accumulate.
                let aq: Vec<i8> = e.stream().clone_dtoh(&aqd)?; e.stream().synchronize()?;
                let ad = e.dtoh(&add)?;
                let nblk = in_f / 32;
                let mut cpu = vec![0f32; mm * out_f];
                for t in 0..mm {
                    for o in 0..out_f {
                        let mut acc = 0f64;
                        for blk in 0..nblk {
                            let mut bs = 0f64;
                            for j in 0..32 {
                                let w = e4m3(wb[o * in_f + blk * 32 + j]) as f64;
                                bs += w * aq[t * in_f + blk * 32 + j] as f64;
                            }
                            acc += ad[t * nblk + blk] as f64 * bs;
                        }
                        cpu[t * out_f + o] = acc as f32;
                    }
                }
                let scale = cpu.iter().map(|v| v.abs()).fold(0.0, f32::max).max(1e-3);
                let rel = maxdiff(&cpu, &y) / scale;
                let mut ok = rel < 1e-3;
                // (2) decode-parity: token t's rows at grid.y=m == the m=1 launch on token t alone.
                let mut bits_ok = true;
                if mm > 1 {
                    for t in 0..mm {
                        let xt = &x[t * in_f..(t + 1) * in_f];
                        let xtd = e.htod(xt)?;
                        let y1 = e.dtoh(&e.qmatvec_mmvq_raw(&wd, &xtd, 1, in_f, out_f, qt,
                                                            row_bytes, false)?)?;
                        bits_ok &= y1.iter().zip(&y[t * out_f..(t + 1) * out_f])
                            .all(|(a, b)| a.to_bits() == b.to_bits());
                    }
                    ok &= bits_ok;
                }
                println!("E4M3-MMVQ synth [{in_f}x{out_f}] m={mm}: rel={rel:.2e} m1-bits={bits_ok} {}",
                         if ok { "OK" } else { fails += 1; "FAIL" });
            }
            // (3) batched twins vs grid.y=m mmvq: bit-exact.
            for mm in 2..=8usize {
                let mcols = memra_engine::Engine::batched_mcols(mm);
                let x: Vec<f32> = (0..mm * in_f).map(|i| pr(i + 163) * 0.1).collect();
                let xd = e.htod(&x)?;
                let yref = e.dtoh(&e.qmatvec_mmvq_raw(&wd, &xd, mm, in_f, out_f, qt, row_bytes, false)?)?;
                let yb = e.dtoh(&e.qmatvec_batched_raw(&wd, &xd, mm, in_f, out_f, qt, row_bytes,
                                                       mcols, false)?)?;
                let bits_bad = yref.iter().zip(&yb).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                let d = maxdiff(&yref, &yb);
                println!("E4M3-BATCHED synth [{in_f}x{out_f}] m={mm} b{mcols}: rel={d:.2e} bit-bad={bits_bad} {}",
                         if bits_bad == 0 { "OK" } else { fails += 1; "FAIL" });
            }
        }
    }

    // --- BATCHED weight-resident matvec (_b2/_b4/_b8) vs the per-m _mmvq reference (the MTP/verify
    // path). Both quantize the same f32 activation to q8_1; the batched kernel only changes the loop
    // nest (weight loaded once, reused across m token columns) so per-(token,row) it MUST be
    // bit-identical to qmatvec_mmvq_raw (grid.y=m). m∈{2..8}; mcols=2/4/8 tiers (b8 = the K=4..7
    // spec verify T=5..8 fix; masked columns c>=m must not perturb c<m). rel<1e-3 + bit-exact. ---
    if let Some(path) = gguf_arg.clone() {
        use memra_gguf::{GgufFile, GgmlType};
        let g = GgufFile::open(&path)?;
        // pick ONE 2D tensor per daily dtype (so Q8_0/Q5_K get covered regardless of model naming).
        let want: [(GgmlType, i32); 4] = [
            (GgmlType::Q8_0, memra_engine::QT_Q8_0), (GgmlType::Q4_K, memra_engine::QT_Q4_K),
            (GgmlType::Q5_K, memra_engine::QT_Q5_K), (GgmlType::Q6_K, memra_engine::QT_Q6_K),
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
            for (mm, mcols) in [(2usize, 2usize), (3, 4), (4, 4), (5, 8), (6, 8), (8, 8)] {
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
    // --- K-QUANT SPLIT-PLANE (rp) gates: q4_K/q6_K mirror vs GGUF layout, every decode
    // consumer (m=1 _mmvq_rp + batched _b{2,4,8}_rp; q6_K adds b16). The mirror is a pure
    // byte permutation and each rp twin keeps the exact per-(token,row) value/product
    // order -> outputs must be BIT-identical (bit-bad == 0). H100 K-quant coalescing fix,
    // 2026-08-01. ---
    if let Some(path) = gguf_arg.clone() {
        use memra_gguf::{GgufFile, GgmlType};
        let g = GgufFile::open(&path)?;
        let want: [(GgmlType, i32); 2] = [
            (GgmlType::Q4_K, memra_engine::QT_Q4_K), (GgmlType::Q6_K, memra_engine::QT_Q6_K),
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
            let mir = e.build_kq_rp4_raw(&wd, in_f, out_f, gt)?;
            // m=1 rp twin vs GGUF-layout mmvq: bit-identical.
            {
                let x: Vec<f32> = (0..in_f).map(|i| pr(i + 151) * 0.1).collect();
                let xd = e.htod(&x)?;
                let yref = e.dtoh(&e.qmatvec_mmvq_raw(&wd, &xd, 1, in_f, out_f, gt, row_bytes, false)?)?;
                let yrp = e.dtoh(&e.qmatvec_mmvq_raw(&mir, &xd, 1, in_f, out_f, gt, row_bytes, true)?)?;
                let bad = yref.iter().zip(&yrp).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                println!("KQRP {tname} [{:?}] m=1 mmvq_rp: bit-bad={bad} {}", t.ggml_type,
                         if bad == 0 { "OK" } else { fails += 1; "FAIL" });
            }
            // batched rp twins vs GGUF-layout batched: bit-identical (b16 tier is q6_K-only).
            let tiers: &[(usize, usize)] = if gt == memra_engine::QT_Q6_K {
                &[(2, 2), (3, 4), (4, 4), (5, 8), (8, 8), (12, 16)]
            } else {
                &[(2, 2), (3, 4), (4, 4), (5, 8), (8, 8)]
            };
            for &(mm, mcols) in tiers {
                let x: Vec<f32> = (0..mm * in_f).map(|i| pr(i + 161) * 0.1).collect();
                let xd = e.htod(&x)?;
                let yref = e.dtoh(&e.qmatvec_batched_raw(&wd, &xd, mm, in_f, out_f, gt, row_bytes, mcols, false)?)?;
                let yrp = e.dtoh(&e.qmatvec_batched_raw(&mir, &xd, mm, in_f, out_f, gt, row_bytes, mcols, true)?)?;
                let bad = yref.iter().zip(&yrp).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                println!("KQRP {tname} [{:?}] m={mm} mcols={mcols} batched_rp: bit-bad={bad} {}", t.ggml_type,
                         if bad == 0 { "OK" } else { fails += 1; "FAIL" });
            }
        }
    }
    // NVFP4 batched vs per-m _mmvq on the 9B model.
    {
        use memra_gguf::{GgufFile, GgmlType};
        let gguf_9b = kc_model("nvfp4-batched", "Qwen3.5-9B-NVFP4-MTP-GGUF.gguf",
            &["/home/avifenesh/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf"],
            &gguf_arg);
        if let Some(gguf_9b) = gguf_9b {
            let g = GgufFile::open(&gguf_9b)?;
            if let Some(t) = g.find("blk.0.ffn_gate.weight").filter(|t| t.ggml_type == GgmlType::NVFP4) {
                let in_f = t.ne[0] as usize; let out_f = t.ne[1] as usize;
                let raw = g.tensor_data(t); let row_bytes = raw.len() / out_f;
                let wd = e.htod_bytes(raw)?;
                for (mm, mcols) in [(2usize, 2usize), (3, 4), (4, 4), (5, 8), (6, 8), (8, 8)] {
                    let x: Vec<f32> = (0..mm * in_f).map(|i| pr(i + 141) * 0.1).collect();
                    let xd = e.htod(&x)?;
                    let yref = e.dtoh(&e.qmatvec_mmvq_raw(&wd, &xd, mm, in_f, out_f, memra_engine::QT_NVFP4, row_bytes, false)?)?;
                    let ybat = e.dtoh(&e.qmatvec_batched_raw(&wd, &xd, mm, in_f, out_f, memra_engine::QT_NVFP4, row_bytes, mcols, false)?)?;
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
        use memra_gguf::{GgufFile, GgmlType};
        use memra_engine::model::{repack_nvfp4_split, unpack_nvfp4_split};
        let path9 = kc_model("a6-split-plane(9b-fallback)", "Qwen3.5-9B-NVFP4-MTP-GGUF.gguf",
            &["/home/avifenesh/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf"],
            &gguf_arg);
        // prefer the model under test if it has NVFP4 tensors; else the 9B.
        let srcs: Vec<String> = gguf_arg.clone().into_iter().chain(path9).collect();
        let mut done = false;
        for path in srcs {
            if done { break; }
            let g = match GgufFile::open(&path) { Ok(g) => g, Err(_) => continue };
            // three shapes: a wide-out FFN gate (rpr2-class), a narrow-out down/out (rpr2w8/rp-
            // class), and a DEEP-k tensor (in_f >= 6144: the rpks/rpksc k-split auto window —
            // added 2026-07-06 so the non-bit-identical family is always gate-covered).
            let mut picks: Vec<_> = g.tensors.iter()
                .filter(|t| t.ggml_type == GgmlType::NVFP4 && t.ne.len() == 2 && t.ne[0] % 64 == 0)
                .take(2).collect();
            if let Some(deep) = g.tensors.iter().find(|t| t.ggml_type == GgmlType::NVFP4
                    && t.ne.len() == 2 && t.ne[0] % 512 == 0 && t.ne[0] >= 6144) {
                if !picks.iter().any(|p| p.name == deep.name) { picks.push(deep); }
            }
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
                    let yref = e.dtoh(&e.qmatvec_mmvq_raw(&wd,  &xd, mm, in_f, out_f, memra_engine::QT_NVFP4, row_bytes, false)?)?;
                    let yrp  = e.dtoh(&e.qmatvec_mmvq_raw(&wrp, &xd, mm, in_f, out_f, memra_engine::QT_NVFP4, row_bytes, true)?)?;
                    let bad = bit_bad(&yref, &yrp);
                    println!("RP MMVQ {tname} m={mm}: bit-bad={bad} {}",
                             if bad == 0 { "OK" } else { fails += 1; "FAIL" });
                }
                // batched rp (auto rule picks rp/rpr2/rpr2w8/rpsc/rpks/rpksc per shape) vs
                // original per-m mmvq. CONTRACT SPLIT (2026-07-06): the k-split family (rpks*)
                // reduces k in two chunks -> deterministic but NOT bit-identical to the reference
                // (FP add order). Its gate = rel<1e-6-of-max + run-to-run BIT determinism; every
                // other variant keeps the strict bit-bad==0 contract.
                for (mm, mcols) in [(2usize, 2usize), (3, 4), (4, 4), (5, 8), (6, 8), (8, 8)] {
                    let x: Vec<f32> = (0..mm * in_f).map(|i| pr(i + 161) * 0.1).collect();
                    let xd = e.htod(&x)?;
                    let yref = e.dtoh(&e.qmatvec_mmvq_raw(&wd, &xd, mm, in_f, out_f, memra_engine::QT_NVFP4, row_bytes, false)?)?;
                    let yrp = e.dtoh(&e.qmatvec_batched_raw(&wrp, &xd, mm, in_f, out_f, memra_engine::QT_NVFP4, row_bytes, mcols, true)?)?;
                    let v = e.batched_variant(mm, in_f, out_f, memra_engine::QT_NVFP4, row_bytes, mcols, true);
                    if v.starts_with("rpks") {
                        let d = maxdiff(&yref, &yrp);
                        let scale = yref.iter().map(|v| v.abs()).fold(0.0, f32::max).max(1e-3);
                        let rel = d / scale;
                        let y2 = e.dtoh(&e.qmatvec_batched_raw(&wrp, &xd, mm, in_f, out_f, memra_engine::QT_NVFP4, row_bytes, mcols, true)?)?;
                        let det = bit_bad(&yrp, &y2);
                        println!("RP BATCHED {tname} m={mm} mcols={mcols} [{v}]: rel={rel:.2e} det-bad={det} {}",
                                 if rel < 1e-6 && det == 0 { "OK" } else { fails += 1; "FAIL" });
                    } else {
                        let bad = bit_bad(&yref, &yrp);
                        println!("RP BATCHED {tname} m={mm} mcols={mcols} [{v}]: bit-bad={bad} {}",
                                 if bad == 0 { "OK" } else { fails += 1; "FAIL" });
                    }
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
                // prefill int8 GEMM kernel2 rp twin (the daily MEMRA_GEMM path) at a real T.
                {
                    let mm = 128usize;
                    let x: Vec<f32> = (0..mm * in_f).map(|i| pr(i + 181) * 0.1).collect();
                    let xd = e.htod(&x)?;
                    let yref = e.dtoh(&e.qmatvec_gemm_raw(&wd,  &xd, mm, in_f, out_f, memra_engine::QT_NVFP4, row_bytes)?)?;
                    let yrp  = e.dtoh(&e.qmatvec_gemm_raw(&wrp, &xd, mm, in_f, out_f, memra_engine::QT_NVFP4_RP, row_bytes)?)?;
                    let bad = bit_bad(&yref, &yrp);
                    println!("RP GEMM {tname} T={mm}: bit-bad={bad} {}",
                             if bad == 0 { "OK" } else { fails += 1; "FAIL" });
                }
                // Stage-A generic (f32 dequant-in-kernel) rp tag vs original.
                {
                    let x: Vec<f32> = (0..in_f).map(|i| pr(i + 191) * 0.1).collect();
                    let xd = e.htod(&x)?;
                    let yref = e.dtoh(&e.qmatvec(&wd,  &xd, 1, in_f, out_f, memra_engine::QT_NVFP4, row_bytes)?)?;
                    let yrp  = e.dtoh(&e.qmatvec(&wrp, &xd, 1, in_f, out_f, memra_engine::QT_NVFP4_RP, row_bytes)?)?;
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
        // --- windowed FA prefill (gemma4 SWA, hd256): CPU-oracle rel + f32-vs-bf16-stage BIT
        // identity (same pre-converter argument as hd512/mmq — any nonzero diff = staging bug).
        {
            let (hdw, nhw, nkvw, wnd) = (256usize, 4usize, 1usize, 32usize);
            let scalew = 1.0f32 / (hdw as f32).sqrt();
            let cpu_sdpa_w = |q: &[f32], k: &[f32], v: &[f32], t: usize, tkv: usize| -> Vec<f32> {
                let mut o = vec![0.0f32; t * nhw * hdw];
                for head in 0..nhw { for qt in 0..t {
                    let q_pos = (tkv - t) + qt;
                    let qv = &q[(qt * nhw + head) * hdw..][..hdw];
                    let mut sc = vec![0.0f32; tkv];
                    for (tk, s) in sc.iter_mut().enumerate() {
                        let kv = &k[tk * hdw..][..hdw];
                        let mut a = 0.0; for d in 0..hdw { a += qv[d] * kv[d]; }
                        a *= scalew;
                        if tk > q_pos || (q_pos >= wnd && tk < q_pos - (wnd - 1)) { a = -1e30; }
                        *s = a;
                    }
                    let mx = sc.iter().cloned().fold(-1e30f32, f32::max);
                    let mut sum = 0.0; for s in sc.iter_mut() { *s = (*s - mx).exp(); sum += *s; }
                    for s in sc.iter_mut() { *s /= sum; }
                    let ov = &mut o[(qt * nhw + head) * hdw..][..hdw];
                    for d in 0..hdw {
                        let mut a = 0.0; for tk in 0..tkv { a += sc[tk] * v[tk * hdw + d]; }
                        ov[d] = a;
                    }
                } }
                o
            };
            for (t, tkv) in [(64usize, 64usize), (100, 100)] {
                let q: Vec<f32> = (0..hdw*nhw*t).map(|i| pr(i+47)*0.2).collect();
                let k: Vec<f32> = (0..hdw*nkvw*tkv).map(|i| pr(i+53)*0.2).collect();
                let v: Vec<f32> = (0..hdw*nkvw*tkv).map(|i| pr(i+61)*0.2).collect();
                let cpu = cpu_sdpa_w(&q, &k, &v, t, tkv);
                let qd=e.htod(&q)?; let kd=e.htod(&k)?; let vd=e.htod(&v)?;
                let mut o_f32=e.zeros(hdw*nhw*t)?; let mut o_bf=e.zeros(hdw*nhw*t)?;
                e.fa_prefill_w_arm(&qd,&kd,&vd,&mut o_f32,hdw,nhw,nkvw,t,tkv,scalew,true,wnd,true,false)?;
                e.fa_prefill_w_arm(&qd,&kd,&vd,&mut o_bf,hdw,nhw,nkvw,t,tkv,scalew,true,wnd,false,false)?;
                let gf=e.dtoh(&o_f32)?; let gb=e.dtoh(&o_bf)?;
                let d=maxdiff(&cpu,&gf);
                let sc=cpu.iter().map(|x|x.abs()).fold(0.0,f32::max).max(1e-3); let rel=d/sc;
                println!("fa_prefill_w T={t} Tkv={tkv} w={wnd}: rel={rel:.2e} {}",
                         if rel<2e-2 {"OK"} else {fails+=1;"FAIL"});
                // The SWA hp door (MEMRA_FAW_HP + MEMRA_FA_F16PV) swaps the bf16 arm for the
                // f16-P/V h2 kernel — a different numeric class; the oracle band above is
                // its gate and bit-identity does not apply.
                let hp_door = memra_engine::fa_f16pv_on() && memra_engine::faw_hp_on();
                if hp_door {
                    println!("fa_prefill_w bf16-stage T={t}: SKIPPED (hp door numeric class)");
                } else {
                    let nbad = gf.iter().zip(gb.iter()).filter(|(a,b)| a.to_bits()!=b.to_bits()).count();
                    println!("fa_prefill_w bf16-stage T={t}: bit-mismatch {nbad}/{} {}",
                             gf.len(), if nbad==0 {"OK"} else {fails+=1;"FAIL"});
                }
            }
        }
        // --- hd512 FA prefill (gemma4 globals, MQA nkv=1): CPU-oracle rel gate on BOTH stage
        // arms + f32-vs-bf16-stage BIT identity (the pre-converter applies the exact
        // __float2bfloat16 the in-kernel stage applied -> ANY nonzero diff = staging bug).
        {
            let (hd5, nh5, nhkv5) = (512usize, 8usize, 1usize);
            let scale5 = 1.0f32 / (hd5 as f32).sqrt();
            let cpu_sdpa5 = |q: &[f32], k: &[f32], v: &[f32], t: usize, tkv: usize| -> Vec<f32> {
                let mut o = vec![0.0f32; t * nh5 * hd5];
                for head in 0..nh5 { for qt in 0..t {
                    let q_pos = (tkv - t) + qt;
                    let qv = &q[(qt * nh5 + head) * hd5..][..hd5];
                    let mut sc = vec![0.0f32; tkv];
                    for (tk, s) in sc.iter_mut().enumerate() {
                        let kv = &k[tk * hd5..][..hd5];
                        let mut a = 0.0; for d in 0..hd5 { a += qv[d] * kv[d]; }
                        a *= scale5; if tk > q_pos { a = -1e30; } *s = a;
                    }
                    let mx = sc.iter().cloned().fold(-1e30f32, f32::max);
                    let mut sum = 0.0; for s in sc.iter_mut() { *s = (*s - mx).exp(); sum += *s; }
                    for s in sc.iter_mut() { *s /= sum; }
                    let ov = &mut o[(qt * nh5 + head) * hd5..][..hd5];
                    for d in 0..hd5 {
                        let mut a = 0.0; for tk in 0..tkv { a += sc[tk] * v[tk * hd5 + d]; }
                        ov[d] = a;
                    }
                } }
                o
            };
            for (t, tkv) in [(64usize, 64usize), (100, 100)] {
                let q: Vec<f32> = (0..hd5*nh5*t).map(|i| pr(i+13)*0.2).collect();
                let k: Vec<f32> = (0..hd5*nhkv5*tkv).map(|i| pr(i+17)*0.2).collect();
                let v: Vec<f32> = (0..hd5*nhkv5*tkv).map(|i| pr(i+23)*0.2).collect();
                let cpu = cpu_sdpa5(&q, &k, &v, t, tkv);
                let qd=e.htod(&q)?; let kd=e.htod(&k)?; let vd=e.htod(&v)?;
                let mut o_f32=e.zeros(hd5*nh5*t)?; let mut o_bf=e.zeros(hd5*nh5*t)?;
                let mut o_sp=e.zeros(hd5*nh5*t)?; let mut o_sp16=e.zeros(hd5*nh5*t)?;
                e.fa_prefill_hd512_arm(&qd,&kd,&vd,&mut o_f32,hd5,nh5,nhkv5,t,tkv,scale5,true,true,false,false)?;
                e.fa_prefill_hd512_arm(&qd,&kd,&vd,&mut o_bf,hd5,nh5,nhkv5,t,tkv,scale5,true,false,false,false)?;
                e.fa_prefill_hd512_arm(&qd,&kd,&vd,&mut o_sp,hd5,nh5,nhkv5,t,tkv,scale5,true,false,true,false)?;
                e.fa_prefill_hd512_arm(&qd,&kd,&vd,&mut o_sp16,hd5,nh5,nhkv5,t,tkv,scale5,true,false,true,true)?;
                let gf=e.dtoh(&o_f32)?; let gb=e.dtoh(&o_bf)?; let gs=e.dtoh(&o_sp)?;
                let gs16=e.dtoh(&o_sp16)?;
                let d=maxdiff(&cpu,&gf);
                let sc=cpu.iter().map(|x|x.abs()).fold(0.0,f32::max).max(1e-3); let rel=d/sc;
                println!("fa_prefill_hd512 T={t} Tkv={tkv}: rel={rel:.2e} {}",
                         if rel<2e-2 {"OK"} else {fails+=1;"FAIL"});
                let nbad = gf.iter().zip(gb.iter()).filter(|(a,b)| a.to_bits()!=b.to_bits()).count();
                println!("fa_prefill_hd512 bf16-stage T={t}: bit-mismatch {nbad}/{} {}",
                         gf.len(), if nbad==0 {"OK"} else {fails+=1;"FAIL"});
                // Single-pass arm: own numeric config (split-K partial order) — oracle band, not bit.
                let dsp=maxdiff(&cpu,&gs); let relsp=dsp/sc;
                println!("fa_prefill_hd512_sp T={t} Tkv={tkv}: rel={relsp:.2e} {}",
                         if relsp<2e-2 {"OK"} else {fails+=1;"FAIL"});
                // f16-P/V door (MEMRA_FA_F16PV): f16 P + f16 P@V accum — own numeric class.
                // Same 2e-2 oracle band: f16's 11-bit mantissa on softmax-weighted O(1) sums
                // sits at ~1e-3; a band miss means a real staging/fragment bug, not rounding.
                let d16=maxdiff(&cpu,&gs16); let rel16=d16/sc;
                println!("fa_prefill_hd512_sp16 T={t} Tkv={tkv}: rel={rel16:.2e} {}",
                         if rel16<2e-2 {"OK"} else {fails+=1;"FAIL"});
            }
        }
        // --- hd512 GQA gate (31B globals: nkv>1, even group — the h2 head-pair arm's real
        // shape; the nkv=1 case above never exercises kv_head sharing). CPU oracle indexes
        // K/V by kv_head = head / (nh/nkv). 2026-07-23: the 31B D512 argmax MISMATCH traced
        // to the hp arm — this gate pins the class band at the GQA shape.
        {
            let (hd6, nh6, nhkv6) = (512usize, 8usize, 4usize);
            let scale6 = 1.0f32 / (hd6 as f32).sqrt();
            let grp = nh6 / nhkv6;
            let cpu_sdpa6 = |q: &[f32], k: &[f32], v: &[f32], t: usize, tkv: usize| -> Vec<f32> {
                let mut o = vec![0.0f32; t * nh6 * hd6];
                for head in 0..nh6 { for qt in 0..t {
                    let kvh = head / grp;
                    let q_pos = (tkv - t) + qt;
                    let qv = &q[(qt * nh6 + head) * hd6..][..hd6];
                    let mut sc = vec![0.0f32; tkv];
                    for (tk, sv) in sc.iter_mut().enumerate() {
                        let kv = &k[(tk * nhkv6 + kvh) * hd6..][..hd6];
                        let mut a = 0.0; for d in 0..hd6 { a += qv[d] * kv[d]; }
                        a *= scale6; if tk > q_pos { a = -1e30; } *sv = a;
                    }
                    let mx = sc.iter().cloned().fold(-1e30f32, f32::max);
                    let mut sum = 0.0; for sv in sc.iter_mut() { *sv = (*sv - mx).exp(); sum += *sv; }
                    for sv in sc.iter_mut() { *sv /= sum; }
                    let ov = &mut o[(qt * nh6 + head) * hd6..][..hd6];
                    for d in 0..hd6 {
                        let mut a = 0.0;
                        for tk in 0..tkv { a += sc[tk] * v[(tk * nhkv6 + kvh) * hd6 + d]; }
                        ov[d] = a;
                    }
                } }
                o
            };
            for (t, tkv) in [(64usize, 64usize), (100, 100)] {
                let q: Vec<f32> = (0..hd6*nh6*t).map(|i| pr(i+29)*0.2).collect();
                let k: Vec<f32> = (0..hd6*nhkv6*tkv).map(|i| pr(i+31)*0.2).collect();
                let v: Vec<f32> = (0..hd6*nhkv6*tkv).map(|i| pr(i+37)*0.2).collect();
                let cpu = cpu_sdpa6(&q, &k, &v, t, tkv);
                let qd=e.htod(&q)?; let kd=e.htod(&k)?; let vd=e.htod(&v)?;
                let mut o_hp=e.zeros(hd6*nh6*t)?;
                e.fa_prefill_hd512_arm(&qd,&kd,&vd,&mut o_hp,hd6,nh6,nhkv6,t,tkv,scale6,true,false,true,true)?;
                let gh=e.dtoh(&o_hp)?;
                let d=maxdiff(&cpu,&gh);
                let sc=cpu.iter().map(|x|x.abs()).fold(0.0,f32::max).max(1e-3); let rel=d/sc;
                println!("fa_prefill_hd512 GQA nkv=4 (hp arm) T={t} Tkv={tkv}: rel={rel:.2e} {}",
                         if rel<2e-2 {"OK"} else {fails+=1;"FAIL"});
            }
        }
        // decode cases (T=1) — K/V come from the QUANTIZED resident cache (q8_0 K / q5_1 V).
        // Quantize the f32 K/V token-by-token via the append kernel, then fa_decode dequants
        // inline. Tolerance loosened vs the f32 path: q5_1 V (5-bit affine) is the looser link.
        let kv_dim_k = hd * nhkv;   // head_dim_k * n_head_kv (head_dim_v == head_dim_k here)
        let kv_dim_v = hd * nhkv;
        let (kbb, vbb) = memra_engine::kv_blk_bytes();  // env-selected KV formats (default 34/24)
        let k_tok_bytes = (kv_dim_k / 32) * kbb;
        let v_tok_bytes = (kv_dim_v / 32) * vbb;
        // format noise floor on the uniform-random synth: default q8_0/q5_1 = 6e-2 (validated).
        // V-format element noise MEASURED by the round-trip gate below (rel to amax): q5_1
        // 1.35e-2, fp8 3.23e-2 (2.4x), q4_0 6.06e-2 (4.5x, == its amax/16 theory bound — the
        // symmetric-4-bit cost). The SDPA rel scales with V element noise because |O| is a
        // small softmax average of the noise-carrying V (the amplification already documented
        // for q5_1 above) -> scale the gate by the measured ratio. Packing correctness is
        // pinned exactly by the round-trip gate; quality arbitration for non-default formats
        // = run-spec acceptance within the config (the kvbytes-lane protocol).
        let kvq_tol: f32 = 6e-2 * match memra_engine::kv_cache_formats().1 {
            "q4_0" => 5.0, "fp8" => 2.5, _ => 1.0,
        };
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
                                           kv_dim_k,kv_dim_v,k_tok_bytes,v_tok_bytes, false)?;
            }
            let kview=e.view_u8(&kc, tkv*k_tok_bytes); let vview=e.view_u8(&vc, tkv*v_tok_bytes);
            let sc=cpu.iter().map(|v|v.abs()).fold(0.0,f32::max).max(1e-3);
            // --- scalar fa_decode_f32 (the bit-reference; MEMRA_NO_FA_VEC=1 forces it — the
            //     old MEMRA_FA_VEC opt-in is retired, vec is the default above FA_VEC_MIN_TKV) ---
            unsafe { std::env::set_var("MEMRA_NO_FA_VEC", "1"); }
            let mut od=e.zeros(hd*nh)?;
            e.fa_decode(&qd,&kview,&vview,&mut od,hd,nh,nhkv,tkv,scale,k_tok_bytes,v_tok_bytes)?;
            let rel = maxdiff(&cpu,&e.dtoh(&od)?)/sc;
            // --- PERF-4 warp-per-token fa_decode_vec_q (GQA broadcast) on the SAME cache.
            //     (tkv=64 sits below FA_VEC_MIN_TKV so both arms run scalar there — the vec
            //     cells are the tkv>=128 rows.) ---
            unsafe { std::env::remove_var("MEMRA_NO_FA_VEC"); }
            let mut od_v=e.zeros(hd*nh)?;
            e.fa_decode(&qd,&kview,&vview,&mut od_v,hd,nh,nhkv,tkv,scale,k_tok_bytes,v_tok_bytes)?;
            let rel_v = maxdiff(&cpu,&e.dtoh(&od_v)?)/sc;
            // Quantized KV (q8_0 K, q5_1 V) -> looser than f32 fa_decode (5e-3). These synthetic
            // inputs are UNIFORM-random in [-0.2,0.2] (worse than real KV: V's q5_1 affine 5-bit
            // noise ~1.35e-2/elem, amplified through the softmax-weighted average when |O| is small).
            // The block round-trip + 5th-bit gates below isolate packing CORRECTNESS; the AUTHORITATIVE
            // end-to-end gate is argmax stability on real models. Gate here: rel < 6e-2 (noise floor).
            println!("fa_decode(KVQ) Tkv={tkv}: rel={rel:.2e} {}", if rel<kvq_tol {"OK"} else {fails+=1;"FAIL"});
            // PERF-4 gate: vec kernel rel < 6e-2 AND no worse than scalar within slack. The vec
            // kernel stores the dequanted KV tile in bf16 smem (8-bit mantissa) for occupancy
            // (-> the 2.2x mid-ctx decode win); the scalar path keeps f32. That adds ~1-1.5e-3
            // of bounded bf16-rounding noise vs scalar — far under the 6e-2 q5_1 noise floor, and
            // the AUTHORITATIVE end-to-end argmax gate (268/271/1178) is unaffected. Slack 2.5e-3.
            let regress = rel_v > rel + 2.5e-3;
            println!("fa_decode_vec_q(KVQ) Tkv={tkv}: rel={rel_v:.2e} (scalar {rel:.2e}) {}",
                     if rel_v<kvq_tol && !regress {"OK"} else {fails+=1;"FAIL"});
            // --- FA v3 (MEMRA_FA_V3=1, dp4a-K hybrid — its OWN numeric config) vs the SAME
            //     CPU oracle. Q rides int8 (one shared scale per 32-elem block, amax/127)
            //     -> adds bounded Q-quantization noise on the scores beyond the bf16 rounding
            //     of the vec path; measured ~2-4e-3 extra on this synthetic. Slack 1e-2 over
            //     scalar, still far under the q5_1 6e-2 noise floor. Only meaningful when the
            //     v3 gate can actually engage (default KV formats + hd%128==0 + vec range).
            if memra_engine::kv_cache_formats() == ("q8_0", "q5_1") && hd % 128 == 0 {
                unsafe { std::env::set_var("MEMRA_FA_V3", "1"); }
                let mut od_3=e.zeros(hd*nh)?;
                e.fa_decode(&qd,&kview,&vview,&mut od_3,hd,nh,nhkv,tkv,scale,k_tok_bytes,v_tok_bytes)?;
                unsafe { std::env::remove_var("MEMRA_FA_V3"); }
                let rel_3 = maxdiff(&cpu,&e.dtoh(&od_3)?)/sc;
                let regress3 = rel_3 > rel + 1e-2;
                println!("fa_decode_vec_q_v3(KVQ) Tkv={tkv}: rel={rel_3:.2e} (scalar {rel:.2e}) {}",
                         if rel_3<kvq_tol && !regress3 {"OK"} else {fails+=1;"FAIL"});
            }
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
                                           kv_dim_k,kv_dim_v,k_tok_bytes,v_tok_bytes, false)?;
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
            e.fa_decode_rows(&qd,&kview,&vview,&mut o_rows,hd,nh,nhkv,base_len,t,scale,k_tok_bytes,v_tok_bytes,None,false, false, None)?;
            let a = e.dtoh(&o_loop)?; let b = e.dtoh(&o_rows)?;
            let bitdiff = a.iter().zip(&b).filter(|(x,y)| x.to_bits() != y.to_bits()).count();
            println!("fa_decode_rows vs per-row loop base={base_len} T={t}: bitdiff={bitdiff} {}",
                     if bitdiff == 0 {"OK"} else {fails+=1;"FAIL"});
            // --- Same rows-vs-loop BYTE identity WITHIN the v3 config (MEMRA_FA_V3=1): the
            //     rows_v3 twin calls the SAME fa_dec_v3_walk as eager v3 -> bitdiff must be 0
            //     (the spec-exactness contract, per numeric config). ---
            if memra_engine::kv_cache_formats() == ("q8_0", "q5_1") && hd % 128 == 0 {
                unsafe { std::env::set_var("MEMRA_FA_V3", "1"); }
                let mut o_loop3 = e.zeros(hd*nh*t)?;
                for r in 0..t {
                    let t_kv_r = base_len + r + 1;
                    let kview=e.view_u8(&kc, t_kv_r*k_tok_bytes);
                    let vview=e.view_u8(&vc, t_kv_r*v_tok_bytes);
                    let mut q_row = e.zeros(hd*nh)?;
                    let q_src = qd.slice(r*nh*hd..(r+1)*nh*hd);
                    e.copy_view_into(&mut q_row, 0, &q_src, nh*hd)?;
                    let mut o_row = e.zeros(hd*nh)?;
                    e.fa_decode(&q_row,&kview,&vview,&mut o_row,hd,nh,nhkv,t_kv_r,scale,k_tok_bytes,v_tok_bytes)?;
                    e.copy_into(&mut o_loop3, r*nh*hd, &o_row, nh*hd)?;
                }
                let kview=e.view_u8(&kc, tkv_max*k_tok_bytes);
                let vview=e.view_u8(&vc, tkv_max*v_tok_bytes);
                let mut o_rows3 = e.zeros(hd*nh*t)?;
                e.fa_decode_rows(&qd,&kview,&vview,&mut o_rows3,hd,nh,nhkv,base_len,t,scale,k_tok_bytes,v_tok_bytes,None,false, false, None)?;
                unsafe { std::env::remove_var("MEMRA_FA_V3"); }
                let a3 = e.dtoh(&o_loop3)?; let b3 = e.dtoh(&o_rows3)?;
                let bd3 = a3.iter().zip(&b3).filter(|(x,y)| x.to_bits() != y.to_bits()).count();
                println!("fa_decode_rows_v3 vs per-row loop (FA_V3) base={base_len} T={t}: bitdiff={bd3} {}",
                         if bd3 == 0 {"OK"} else {fails+=1;"FAIL"});
            }
        }

        // --- BATCHED-TICK increment 2: z-batched SEQS decode (append + FA) vs the per-seq
        // loop: BYTE identity. The seqs append twin must write bit-identical cache bytes to
        // B per-seq append calls, and fa_decode_vec_q_seqs_v4 + combine_seqs must reproduce
        // the per-seq eager v4 (fa_decode) program exactly — same in-kernel split partition
        // (ONE-PARTITION LAW at the shared rung), same walk, same combine order. Depths mix
        // uneven sequences crossing split boundaries within one fa_split_keys rung.
        {
            use cudarc::driver::DevicePtr;
            for depths in [vec![96usize, 128, 257, 511], vec![200; 8]] {
                let b_n = depths.len();
                let sp0 = memra_engine::fa_split_keys_pub(depths[0], nhkv);
                let eligible = depths.iter().all(|&t| memra_engine::fa_seqs_eligible(t, hd))
                    && depths.iter().all(|&t| memra_engine::fa_split_keys_pub(t, nhkv) == sp0);
                if !eligible { continue; }   // non-v4 geometry/config: the seqs arm never fires
                let t_kv_max = *depths.iter().max().unwrap();
                // per-seq caches primed to depth-1 tokens from a shared random pool
                let kpool: Vec<f32> = (0..kv_dim_k*t_kv_max).map(|i| pr(i+13)*0.2).collect();
                let vpool: Vec<f32> = (0..kv_dim_v*t_kv_max).map(|i| pr(i+17)*0.2).collect();
                let kpd = e.htod(&kpool)?; let vpd = e.htod(&vpool)?;
                let mut kcs: Vec<_> = Vec::new(); let mut vcs: Vec<_> = Vec::new();
                let mut kcs2: Vec<_> = Vec::new(); let mut vcs2: Vec<_> = Vec::new();
                for &tkv in &depths {
                    let mut kc = e.alloc_u8(tkv * k_tok_bytes)?;
                    let mut vc = e.alloc_u8(tkv * v_tok_bytes)?;
                    for tok in 0..tkv-1 {
                        let k_row = kpd.slice(tok*kv_dim_k..(tok+1)*kv_dim_k);
                        let v_row = vpd.slice(tok*kv_dim_v..(tok+1)*kv_dim_v);
                        e.append_kv_quantized_view(&k_row,&v_row,&mut kc,&mut vc,tok,
                                                   kv_dim_k,kv_dim_v,k_tok_bytes,v_tok_bytes,false)?;
                    }
                    // twin cache with the SAME primed prefix (bytes copied via re-append)
                    let mut kc2 = e.alloc_u8(tkv * k_tok_bytes)?;
                    let mut vc2 = e.alloc_u8(tkv * v_tok_bytes)?;
                    for tok in 0..tkv-1 {
                        let k_row = kpd.slice(tok*kv_dim_k..(tok+1)*kv_dim_k);
                        let v_row = vpd.slice(tok*kv_dim_v..(tok+1)*kv_dim_v);
                        e.append_kv_quantized_view(&k_row,&v_row,&mut kc2,&mut vc2,tok,
                                                   kv_dim_k,kv_dim_v,k_tok_bytes,v_tok_bytes,false)?;
                    }
                    kcs.push(kc); vcs.push(vc); kcs2.push(kc2); vcs2.push(vc2);
                }
                // this tick's stacked new rows + positions (slot = depth-1)
                let knew: Vec<f32> = (0..kv_dim_k*b_n).map(|i| pr(i+23)*0.2).collect();
                let vnew: Vec<f32> = (0..kv_dim_v*b_n).map(|i| pr(i+27)*0.2).collect();
                let knd = e.htod(&knew)?; let vnd = e.htod(&vnew)?;
                let pos: Vec<i32> = depths.iter().map(|&t| (t - 1) as i32).collect();
                let pos_d = e.htod_i32(&pos)?;
                // arm 1 (loop): per-seq append into kcs/vcs
                for z in 0..b_n {
                    let k_row = knd.slice(z*kv_dim_k..(z+1)*kv_dim_k);
                    let v_row = vnd.slice(z*kv_dim_v..(z+1)*kv_dim_v);
                    e.append_kv_quantized_view(&k_row,&v_row,&mut kcs[z],&mut vcs[z],depths[z]-1,
                                               kv_dim_k,kv_dim_v,k_tok_bytes,v_tok_bytes,false)?;
                }
                // arm 2 (seqs): one z-batched launch into kcs2/vcs2 via the pointer table
                let es = e.gpu.stream();
                let mut ptrs2: Vec<u64> = Vec::new();
                for z in 0..b_n {
                    let (pk, _g) = kcs2[z].device_ptr(&es);
                    let (pv, _g2) = vcs2[z].device_ptr(&es);
                    ptrs2.push(pk as u64); ptrs2.push(pv as u64);
                }
                let table2 = e.htod_u64(&ptrs2)?;
                let tv2 = table2.slice(0..2*b_n);
                e.append_kv_quantized_seqs(&knd,&vnd,&tv2,&pos_d,b_n,
                                           kv_dim_k,kv_dim_v,k_tok_bytes,v_tok_bytes)?;
                let mut ap_diff = 0usize;
                for z in 0..b_n {
                    let a = e.dtoh_u8(&kcs[z])?; let b = e.dtoh_u8(&kcs2[z])?;
                    ap_diff += a.iter().zip(&b).filter(|(x,y)| x != y).count();
                    let a = e.dtoh_u8(&vcs[z])?; let b = e.dtoh_u8(&vcs2[z])?;
                    ap_diff += a.iter().zip(&b).filter(|(x,y)| x != y).count();
                }
                println!("append_kv_seqs vs per-seq loop B={b_n}: bytediff={ap_diff} {}",
                         if ap_diff == 0 {"OK"} else {fails+=1;"FAIL"});
                // FA: per-seq eager loop (q-row copies, the fallback arm's exact program)
                // vs one seqs launch reading the SAME caches (arm-1 set — isolates FA).
                let q: Vec<f32> = (0..hd*nh*b_n).map(|i| pr(i+31)*0.2).collect();
                let qd = e.htod(&q)?;
                let mut o_loop = e.zeros(hd*nh*b_n)?;
                for z in 0..b_n {
                    let kview = e.view_u8(&kcs[z], depths[z]*k_tok_bytes);
                    let vview = e.view_u8(&vcs[z], depths[z]*v_tok_bytes);
                    let mut q_row = e.zeros(hd*nh)?;
                    let q_src = qd.slice(z*nh*hd..(z+1)*nh*hd);
                    e.copy_view_into(&mut q_row, 0, &q_src, nh*hd)?;
                    let mut o_row = e.zeros(hd*nh)?;
                    e.fa_decode(&q_row,&kview,&vview,&mut o_row,hd,nh,nhkv,depths[z],scale,
                                k_tok_bytes,v_tok_bytes)?;
                    e.copy_into(&mut o_loop, z*nh*hd, &o_row, nh*hd)?;
                }
                let mut ptrs1: Vec<u64> = Vec::new();
                for z in 0..b_n {
                    let (pk, _g) = kcs[z].device_ptr(&es);
                    let (pv, _g2) = vcs[z].device_ptr(&es);
                    ptrs1.push(pk as u64); ptrs1.push(pv as u64);
                }
                let table1 = e.htod_u64(&ptrs1)?;
                let tv1 = table1.slice(0..2*b_n);
                let mut o_seqs = e.zeros(hd*nh*b_n)?;
                e.fa_decode_batch_seqs_v4(&qd,&tv1,&pos_d,&mut o_seqs,hd,nh,nhkv,b_n,
                                          t_kv_max,scale,sp0,k_tok_bytes,v_tok_bytes)?;
                let a = e.dtoh(&o_loop)?; let b = e.dtoh(&o_seqs)?;
                let bitdiff = a.iter().zip(&b).filter(|(x,y)| x.to_bits() != y.to_bits()).count();
                println!("fa_decode_seqs_v4 vs per-seq loop B={b_n} depths={depths:?}: bitdiff={bitdiff} {}",
                         if bitdiff == 0 {"OK"} else {fails+=1;"FAIL"});
            }
        }

        // --- ARC B: fa_prefill_view_ws (dequant-once bf16 workspace) vs fa_prefill_view: BYTE
        // identity. The workspace stores __float2bfloat16(dq_*_elem(...)) — the identical value
        // fa_prefill_q stages to smem — and fa_prefill_qw's MMA/softmax/PV code is byte-identical,
        // so O must match BIT-FOR-BIT (this is the chunk-prime token-identity contract). Cases
        // cover a continuation chunk (T < T_kv, the chunk-prime shape) and a BK-unaligned tail.
        for (t, tkv) in [(64usize, 192usize), (100, 100), (37, 297)] {
            let q: Vec<f32> = (0..hd*nh*t).map(|i| pr(i+5)*0.2).collect();
            let k: Vec<f32> = (0..hd*nhkv*tkv).map(|i| pr(i+7)*0.2).collect();
            let v: Vec<f32> = (0..hd*nhkv*tkv).map(|i| pr(i+11)*0.2).collect();
            let qd=e.htod(&q)?; let kd=e.htod(&k)?; let vd=e.htod(&v)?;
            let mut kc = e.alloc_u8(tkv * k_tok_bytes)?;
            let mut vc = e.alloc_u8(tkv * v_tok_bytes)?;
            for tok in 0..tkv {
                let k_row = kd.slice(tok*kv_dim_k..(tok+1)*kv_dim_k);
                let v_row = vd.slice(tok*kv_dim_v..(tok+1)*kv_dim_v);
                e.append_kv_quantized_view(&k_row,&v_row,&mut kc,&mut vc,tok,
                                           kv_dim_k,kv_dim_v,k_tok_bytes,v_tok_bytes, false)?;
            }
            let kview=e.view_u8(&kc, tkv*k_tok_bytes); let vview=e.view_u8(&vc, tkv*v_tok_bytes);
            let mut o_inl = e.zeros(hd*nh*t)?;
            e.fa_prefill_view(&qd,&kview,&vview,&mut o_inl,hd,nh,nhkv,t,tkv,scale,true,
                              k_tok_bytes,v_tok_bytes, false)?;
            let mut o_ws = e.zeros(hd*nh*t)?;
            e.fa_prefill_view_ws(&qd,&kview,&vview,&mut o_ws,hd,nh,nhkv,t,tkv,scale,true,
                                 k_tok_bytes,v_tok_bytes, false)?;
            let a = e.dtoh(&o_inl)?; let b = e.dtoh(&o_ws)?;
            let bitdiff = a.iter().zip(&b).filter(|(x,y)| x.to_bits() != y.to_bits()).count();
            println!("fa_prefill_view_ws vs inline-dequant T={t} Tkv={tkv}: bitdiff={bitdiff} {}",
                     if bitdiff == 0 {"OK"} else {fails+=1;"FAIL"});
        }
    }

    // --- KV-cache quantization round-trip: append-quantize then dequant (matches §A formulas) ---
    // Quantize a known f32 K/V row with the append kernel, read the bytes back, dequant on the CPU
    // via the exact ggml q8_0/q5_1 formulas, compare to the f32 input. Isolates layout/packing bugs
    // (esp. the q5_1 qh ballot) from attention. Includes a 5th-bit-boundary block (15<->16, 31).
    {
        use memra_gguf::dequant::fp16_to_f32;
        let (kfmt, vfmt) = memra_engine::kv_cache_formats();
        let (kbb, vbb) = memra_engine::kv_blk_bytes();
        let nblk = 4usize;                 // 4 blocks -> 128 elements
        let kv_dim_k = nblk * 32;
        let kv_dim_v = nblk * 32;
        let k_tok_bytes = (kv_dim_k / 32) * kbb;
        let v_tok_bytes = (kv_dim_v / 32) * vbb;
        // K input: signed random; V input: includes a block crafted to span the 5th-bit boundary.
        let kin: Vec<f32> = (0..kv_dim_k).map(|i| pr(i + 71) * 1.3).collect();
        let mut vin: Vec<f32> = (0..kv_dim_v).map(|i| pr(i + 91) * 0.7 + 0.1).collect();
        // craft block 1 of V so quantized q5 values hit 0..31 spanning bit-4 (15<->16, 31). With
        // mn=0, mx=31*d, q5(j)=round((v-mn)/d) -> set v[j]=j*step so q5 sweeps 0..31 across the warp.
        let step = 0.05f32;
        for j in 0..32 { vin[32 + j] = j as f32 * step; }
        let kd = e.htod(&kin)?; let vd = e.htod(&vin)?;
        let mut kc = e.alloc_u8(k_tok_bytes)?; let mut vc = e.alloc_u8(v_tok_bytes)?;
        e.append_kv_quantized(&kd, &vd, &mut kc, &mut vc, 0, kv_dim_k, kv_dim_v, k_tok_bytes, v_tok_bytes, false)?;
        let kbytes = e.dtoh_u8(&kc)?; let vbytes = e.dtoh_u8(&vc)?;
        let f16_to_f32 = |b: &[u8]| -> f32 { fp16_to_f32(u16::from_le_bytes([b[0], b[1]])) };
        // CPU e4m3 decode (raw-fp8 arms): sign / 4-bit exp (bias 7) / 3-bit mantissa, subnormals.
        let e4m3 = |b: u8| -> f32 {
            let s = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
            let ex = ((b >> 3) & 0x0F) as i32;
            let mn = (b & 0x07) as f32;
            if ex == 0 { s * mn * (2f32).powi(-9) }                 // subnormal: 2^-6 * m/8
            else if ex == 15 && mn == 7.0 { f32::NAN }              // e4m3 NaN encoding
            else { s * (1.0 + mn / 8.0) * (2f32).powi(ex - 7) }
        };
        // ---- K round-trip (format-exact CPU dequant) ----
        let mut k_deq = vec![0f32; kv_dim_k];
        for blk in 0..nblk {
            let base = blk * kbb;
            match kfmt {
                "fp8" => for j in 0..32 { k_deq[blk * 32 + j] = e4m3(kbytes[base + j]); },
                _ => {
                    let d = f16_to_f32(&kbytes[base..base + 2]);
                    for j in 0..32 { k_deq[blk * 32 + j] = d * (kbytes[base + 2 + j] as i8) as f32; }
                }
            }
        }
        let kerr = maxdiff(&kin, &k_deq);
        // q8_0 abs err <= d/2 (rel 5e-3 vs amax, validated); raw e4m3 rel err <= 2^-4 -> gate 7e-2.
        let kamax = kin.iter().map(|v| v.abs()).fold(0.0, f32::max).max(1e-6);
        let krel = kerr / kamax;
        let ktol = if kfmt == "fp8" { 7e-2 } else { 5e-3 };
        println!("kvq {kfmt} K round-trip: rel={krel:.2e} {}", if krel < ktol { "OK" } else { fails += 1; "FAIL" });
        // ---- V round-trip (format-exact CPU dequant) ----
        let mut v_deq = vec![0f32; kv_dim_v];
        for blk in 0..nblk {
            let base = blk * vbb;
            match vfmt {
                "fp8" => for j in 0..32 { v_deq[blk * 32 + j] = e4m3(vbytes[base + j]); },
                "q4_0" => {
                    let d = f16_to_f32(&vbytes[base..base + 2]);
                    let qs = &vbytes[base + 2..base + 18];
                    for j in 0..32 {
                        let q = if j < 16 { (qs[j] & 0x0F) as i32 } else { (qs[j - 16] >> 4) as i32 };
                        v_deq[blk * 32 + j] = d * (q - 8) as f32;
                    }
                }
                _ => {
                    let d = f16_to_f32(&vbytes[base..base + 2]);
                    let m = f16_to_f32(&vbytes[base + 2..base + 4]);
                    let qh = u32::from_le_bytes([vbytes[base + 4], vbytes[base + 5], vbytes[base + 6], vbytes[base + 7]]);
                    let qs = &vbytes[base + 8..base + 24];
                    for j in 0..32 {
                        let lo = if j < 16 { (qs[j] & 0x0F) as i32 } else { (qs[j - 16] >> 4) as i32 };
                        let hi = (((qh >> j) & 1) << 4) as i32;
                        v_deq[blk * 32 + j] = d * (lo | hi) as f32 + m;
                    }
                }
            }
        }
        let verr = maxdiff(&vin, &v_deq);
        let vamax = vin.iter().map(|v| v.abs()).fold(0.0, f32::max).max(1e-6);
        let vrel = verr / vamax;
        // q5_1 3e-2 (validated); q4_0 half-step = amax/16 -> 7e-2; raw e4m3 -> 7e-2.
        let vtol = if vfmt == "q5_1" { 3e-2 } else { 7e-2 };
        println!("kvq {vfmt} V round-trip: rel={vrel:.2e} {}", if vrel < vtol { "OK" } else { fails += 1; "FAIL" });
        // explicit 5th-bit-boundary check on V block 1 (q5 sweeps 0..31) — q5_1 layout only.
        if vfmt == "q5_1" {
            let bnd_err = (0..32).map(|j| (vin[32 + j] - v_deq[32 + j]).abs()).fold(0.0, f32::max);
            let bnd_d = step;  // block1 d ~= (31*step - 0)/31 = step
            println!("kvq q5_1 5th-bit boundary: maxerr={bnd_err:.2e} (d~{bnd_d:.2e}) {}",
                     if bnd_err < bnd_d { "OK" } else { fails += 1; "FAIL" });
        }
    }

    // --- BATCHED PROMPT PRIME: batched-rows KV append vs T sequential per-token appends must be
    // BYTE-IDENTICAL (same warp program per (block,token); this pins the (b,tt) grid mapping +
    // token-major row addressing against refactors). Non-trivial T and a non-zero slot base t0.
    {
        let nblk = 4usize;
        let kv_dim_k = nblk * 32;
        let kv_dim_v = nblk * 32;
        let (kbb, vbb) = memra_engine::kv_blk_bytes();
        let k_tok_bytes = (kv_dim_k / 32) * kbb;
        let v_tok_bytes = (kv_dim_v / 32) * vbb;
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
                                       kv_dim_k, kv_dim_v, k_tok_bytes, v_tok_bytes, false)?;
        }
        // (b) batched-rows kernel, one launch.
        let mut kc_b = e.alloc_u8(cap * k_tok_bytes)?; let mut vc_b = e.alloc_u8(cap * v_tok_bytes)?;
        e.append_kv_quantized_rows(&kd, &vd, &mut kc_b, &mut vc_b, t0, t,
                                   kv_dim_k, kv_dim_v, k_tok_bytes, v_tok_bytes, false)?;
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
        use memra_gguf::{GgufFile, GgmlType};
        use memra_engine::moe_cache::{MoeSlotCache, BlockId, PROJ_GATE};
        let gguf_35b = kc_model("d2-cache-bit-identity", "Qwen3.6-35B-A3B-UD-IQ4_XS.gguf",
            &["/home/avifenesh/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"],
            &gguf_arg);
        if let Some(gguf_35b) = gguf_35b {
            let g = GgufFile::open(&gguf_35b)?;
            let t = g.find("blk.0.ffn_gate_exps.weight").expect("gate_exps");
            let in_f = t.ne[0] as usize; let out_f = t.ne[1] as usize; let n_expert = t.ne[2] as usize;
            let qt_opt = match t.ggml_type {
                GgmlType::IQ3_S => Some(memra_engine::QT_IQ3_S), GgmlType::IQ4_XS => Some(memra_engine::QT_IQ4_XS),
                GgmlType::Q6_K => Some(memra_engine::QT_Q6_K), GgmlType::Q8_0 => Some(memra_engine::QT_Q8_0),
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
        }
    }

    // --- FAST-ROUTER batch-twin bit-identity (lane/fast-router, 2026-08-02): the prefill-exact
    // contract routes prefill through decode's m-invariant router_gemv; router_gemv_f32_w8_batch
    // register-tiles the SAME per-row FP chains for GEMM-shaped m. Gate: bitwise equality vs
    // the per-(expert,token) w8 form at every m in a 1..2048 sweep on the REAL q35 router
    // weights, plus m-invariance of the batch form itself (rows of y(m) == the m=2048 run's
    // prefix). Any bit diff = a broken reduction order — fix the kernel, not the gate.
    // Crossover between forms is therefore pure perf, never a numeric config. (The shexp
    // sigmoid-dot twin passed this same gate but measured slower at every t — killed;
    // research/fast-router-20260802/.) ---
    {
        use memra_gguf::{GgufFile, GgmlType};
        let gguf_q35 = kc_model("fast-router-batch", "Qwen3.6-35B-A3B-UD-IQ4_XS.gguf",
            &["/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf",
              "/home/avifenesh/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"],
            &gguf_arg);
        if let Some(p) = gguf_q35 {
            let g = GgufFile::open(&p)?;
            let tw = g.find("blk.0.ffn_gate_inp.weight").expect("gate_inp");
            assert!(matches!(tw.ggml_type, GgmlType::F32), "gate_inp must be F32");
            let n_embd = tw.ne[0] as usize;
            let n_experts = tw.ne[1] as usize;
            let le = |b: &[u8]| f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            let wf: Vec<f32> = g.tensor_data(tw).chunks_exact(4).map(le).collect();
            let t_max = 2048usize;
            let x: Vec<f32> = (0..t_max * n_embd).map(|i| (pr(i + 7) - 0.5) * 4.0).collect();
            let wd = e.htod(&wf)?; let xd = e.htod(&x)?;
            // m=2048 plain-w8 run: its row prefixes are the m-invariance oracle.
            let yref = e.dtoh(&e.router_gemv_form(&wd, &xd, n_embd, n_experts, t_max, true, false)?)?;
            let ms: [usize; 32] = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65,
                                   75, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024,
                                   1025, 2047, 2048];
            let (mut r_bits, mut r_minv) = (0usize, 0usize);
            for &m in &ms {
                let y_p = e.dtoh(&e.router_gemv_form(&wd, &xd, n_embd, n_experts, m, true, false)?)?;
                let y_b = e.dtoh(&e.router_gemv_form(&wd, &xd, n_embd, n_experts, m, true, true)?)?;
                r_bits += y_p.iter().zip(&y_b).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                r_minv += y_b.iter().zip(&yref[..m * n_experts])
                    .filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            }
            println!("router batch-twin bit-identity (real q35 router, {} m-points 1..{t_max}): mism={r_bits} {}",
                     ms.len(), if r_bits == 0 { "OK" } else { fails += 1; "FAIL" });
            println!("router batch-twin m-invariance (rows vs plain m={t_max} prefix): mism={r_minv} {}",
                     if r_minv == 0 { "OK" } else { fails += 1; "FAIL" });
        }
    }

    // --- EDGE-1 §C.2/C.3: copy-stream prefetch publication + store-before-reuse ordering. Fill an
    // 8-slot cache without synchronizing, asynchronously replace one victim, then dispatch/read it.
    // The read must see the new bytes, while the explicitly protected current block stays resident.
    {
        use memra_engine::moe_cache::{BlockId, DispatchSlot, MoeSlotCache, PROJ_GATE};
        let old_slots = std::env::var_os("MEMRA_MOE_SLOTS");
        // SAFETY: kernel-check is a single-threaded process and no other code reads this variable
        // while the scoped synthetic cache is being constructed.
        unsafe { std::env::set_var("MEMRA_MOE_SLOTS", "8"); }
        let block_len = 4096usize;
        let mut cache = MoeSlotCache::new(&e, block_len)?;
        let sources: Vec<Vec<u8>> = (0..8).map(|i| vec![0xA0 + i as u8; block_len]).collect();
        for (i, src) in sources.iter().enumerate() {
            cache.force_admit(BlockId::new(7, PROJ_GATE, i as u16), src, &e)?;
        }
        let keep = [BlockId::new(7, PROJ_GATE, 0)];
        let next_id = BlockId::new(7, PROJ_GATE, 8);
        let next = vec![0xF8; block_len];
        let queued = cache.prefetch(next_id, &next, &keep, &e)?;
        let hidden_while_pending = cache.resident(next_id).is_none();
        let DispatchSlot::Resident(next_slot) = cache.dispatch(next_id, &next, &e)?;
        // slots carry a +8 tail pad (wide-load expert dots, b6f0ffe) — compare payload only.
        let next_got = e.dtoh_u8(cache.slot(next_slot))?[..block_len].to_vec();
        let visible_after_wait = cache.resident(next_id) == Some(next_slot);
        let _ = cache.dispatch(next_id, &next, &e)?;
        let keep_slot = cache.resident(keep[0]);
        let keep_got = match keep_slot {
            Some(slot) => e.dtoh_u8(cache.slot(slot))?[..block_len].to_vec(),
            None => Vec::new(),
        };
        let counters_ok = cache.hits == 1 && cache.misses == 1
            && cache.staged_bytes == 9 * block_len as u64;
        let ok = queued && hidden_while_pending && visible_after_wait
            && next_got == next && keep_got == sources[0] && counters_ok;
        if !ok {
            eprintln!("[prefetch-check] queued={queued} hidden={hidden_while_pending} \
                       visible={visible_after_wait} bytes_ok={} keep_ok={} counters: hits={} \
                       misses={} staged={} (want 1/1/{})",
                      next_got == next, keep_got == sources[0], cache.hits, cache.misses,
                      cache.staged_bytes, 9 * block_len);
        }
        println!("moe async-prefetch ordering + protected victim: {}",
                 if ok { "OK" } else { fails += 1; "FAIL" });
        unsafe {
            match old_slots {
                Some(v) => std::env::set_var("MEMRA_MOE_SLOTS", v),
                None => std::env::remove_var("MEMRA_MOE_SLOTS"),
            }
        }
    }

    if fails == 0 { println!("\nALL GREEN: kernels match CPU reference."); Ok(()) }
    else { Err(format!("{fails} kernel(s) FAILED").into()) }
}
