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
        // decode cases (T=1)
        for tkv in [64usize, 128, 257] {
            let q: Vec<f32> = (0..hd*nh).map(|i| pr(i+1)*0.2).collect();
            let k: Vec<f32> = (0..hd*nhkv*tkv).map(|i| pr(i+7)*0.2).collect();
            let v: Vec<f32> = (0..hd*nhkv*tkv).map(|i| pr(i+11)*0.2).collect();
            let cpu = cpu_sdpa(&q,&k,&v,1,tkv);
            let qd=e.htod(&q)?; let kd=e.htod(&k)?; let vd=e.htod(&v)?; let mut od=e.zeros(hd*nh)?;
            let kview=e.view(&kd, hd*nhkv*tkv); let vview=e.view(&vd, hd*nhkv*tkv);
            e.fa_decode(&qd,&kview,&vview,&mut od,hd,nh,nhkv,tkv,scale)?;
            let g=e.dtoh(&od)?; let d=maxdiff(&cpu,&g);
            let sc=cpu.iter().map(|v|v.abs()).fold(0.0,f32::max).max(1e-3); let rel=d/sc;
            println!("fa_decode  Tkv={tkv}: rel={rel:.2e} {}", if rel<5e-3 {"OK"} else {fails+=1;"FAIL"});
        }
    }

    if fails == 0 { println!("\nALL GREEN: kernels match CPU reference."); Ok(()) }
    else { Err(format!("{fails} kernel(s) FAILED").into()) }
}
