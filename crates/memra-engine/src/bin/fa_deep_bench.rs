//! FA-DEEP lane bench (2026-08-02, research/fa-decode-deep-20260802/): same-process A/B of
//! the v4 vs v4_deep decode twins on production-appended synthetic KV at the depth-decay
//! class geometry (hd256 / n_head 16 / n_head_kv 2 — q35/KAT/o35b per gguf headers).
//!
//! Two jobs, one process:
//!   1. BIT gate: fa_decode (eager) and fa_decode_dc (graph twin, incl. a bucketed
//!      bucket_max > t_kv replay case) must be BIT-IDENTICAL between MEMRA_FA_DEEP=0 and
//!      the deep twins forced on (MEMRA_FA_DEEP_MIN=0) at every depth, incl. split-ladder
//!      rung crossings (3071/3072/3073) and tail tiles (t_kv % 32 != 0). Exit 1 on any diff.
//!   2. TIMING: interleaved per-call wall micro-timing of the production dc form (memsets +
//!      vec kernel + combine, same overheads both arms), medians over interleaved rounds.
//!
//! usage: fa-deep-bench [iters] (default 200)   — run under `flock /tmp/gpu5090.lock`.
use memra_engine::Engine;
use memra_validate::pr;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let iters: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(200);
    let e = Engine::new(0)?;
    let (hd, nh, nhkv) = (256usize, 16usize, 2usize);
    let scale = 1.0f32 / (hd as f32).sqrt();
    let kv_dim = hd * nhkv;
    let (kbb, vbb) = memra_engine::kv_blk_bytes();
    let k_tok_bytes = (kv_dim / 32) * kbb;
    let v_tok_bytes = (kv_dim / 32) * vbb;

    // depths: the sweep points + rung crossings + tail tiles (511/512/513 cross the
    // re-swept sp8->sp64 rung at 512, lane/ladder-3072; 3071/3072/3073 crossed the old
    // 3072 boundary and stay as coverage)
    let bit_depths: Vec<usize> = vec![511, 512, 513, 2048, 3071, 3072, 3073, 4096, 4097, 6143, 6144, 6200];
    // fine grid for the MEMRA_FA_DEEP_MIN floor sweep (`sweep` mode; default = board depths).
    // lane/ladder-3072: +1024/1536 in the default grid (sp-ladder rung sweep needs the region
    // below d2048; MEMRA_FA_SPLIT forces the arm per process — OnceLock, one split per run).
    let time_depths: Vec<usize> = if std::env::args().nth(1).as_deref() == Some("sweep") {
        vec![96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 5120, 6144]
    } else {
        vec![512, 1024, 1536, 2048, 3072, 4096, 6144]
    };
    let t_max = 6272usize;

    // Build the synthetic cache once via the PRODUCTION append kernel (kernel_check recipe).
    let kf: Vec<f32> = (0..kv_dim * t_max).map(|i| pr(i + 7) * 0.2).collect();
    let vf: Vec<f32> = (0..kv_dim * t_max).map(|i| pr(i + 11) * 0.2).collect();
    let kd = e.htod(&kf)?;
    let vd = e.htod(&vf)?;
    let mut kc = e.alloc_u8(t_max * k_tok_bytes)?;
    let mut vc = e.alloc_u8(t_max * v_tok_bytes)?;
    for tok in 0..t_max {
        let k_row = kd.slice(tok * kv_dim..(tok + 1) * kv_dim);
        let v_row = vd.slice(tok * kv_dim..(tok + 1) * kv_dim);
        e.append_kv_quantized_view(&k_row, &v_row, &mut kc, &mut vc, tok,
                                   kv_dim, kv_dim, k_tok_bytes, v_tok_bytes, false)?;
    }
    let q: Vec<f32> = (0..hd * nh).map(|i| pr(i + 1) * 0.2).collect();
    let qd = e.htod(&q)?;
    let mut fails = 0usize;

    // ncu mode: `fa-deep-bench ncu [depth]` — launch each dc arm x10 at one depth, nothing
    // else, so `ncu -k regex:fa_decode_vec_q_v4` profiles exactly these.
    if std::env::args().nth(1).as_deref() == Some("ncu") {
        let d: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(6144);
        unsafe { std::env::set_var("MEMRA_FA_DEEP_MIN", "0"); }
        let tdev = e.htod_i32(&[d as i32])?;
        let kview = e.view_u8(&kc, d * k_tok_bytes);
        let vview = e.view_u8(&vc, d * v_tok_bytes);
        let mut o = e.zeros(hd * nh)?;
        for arm in ["0", "1"] {
            unsafe { std::env::set_var("MEMRA_FA_DEEP", arm); }
            for _ in 0..10 {
                e.fa_decode_dc(&qd, &kview, &vview, &mut o, hd, nh, nhkv, &tdev, d,
                               scale, k_tok_bytes, v_tok_bytes, false)?;
            }
            e.stream().synchronize()?;
        }
        println!("ncu mode done (depth {d})");
        return Ok(());
    }

    // ---- 1. BIT gate ----
    unsafe { std::env::set_var("MEMRA_FA_DEEP_MIN", "0"); }   // force deep at every depth
    for &d in &bit_depths {
        let kview = e.view_u8(&kc, d * k_tok_bytes);
        let vview = e.view_u8(&vc, d * v_tok_bytes);
        // eager pair
        unsafe { std::env::set_var("MEMRA_FA_DEEP", "0"); }
        let mut o_v4 = e.zeros(hd * nh)?;
        e.fa_decode(&qd, &kview, &vview, &mut o_v4, hd, nh, nhkv, d, scale,
                    k_tok_bytes, v_tok_bytes)?;
        unsafe { std::env::set_var("MEMRA_FA_DEEP", "1"); }
        let mut o_dp = e.zeros(hd * nh)?;
        e.fa_decode(&qd, &kview, &vview, &mut o_dp, hd, nh, nhkv, d, scale,
                    k_tok_bytes, v_tok_bytes)?;
        let (a, b) = (e.dtoh(&o_v4)?, e.dtoh(&o_dp)?);
        let bd = a.iter().zip(&b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        println!("deep-vs-v4 eager t_kv={d}: bitdiff={bd} {}",
                 if bd == 0 { "OK" } else { fails += 1; "FAIL" });
        // dc pair (exact bucket + a bucketed replay: bucket_max = next 512-multiple + 64)
        let tdev = e.htod_i32(&[d as i32])?;
        for bucket in [d, (d + 511) / 512 * 512 + 64] {
            unsafe { std::env::set_var("MEMRA_FA_DEEP", "0"); }
            let mut o_v4dc = e.zeros(hd * nh)?;
            e.fa_decode_dc(&qd, &kview, &vview, &mut o_v4dc, hd, nh, nhkv, &tdev, bucket,
                           scale, k_tok_bytes, v_tok_bytes, false)?;
            unsafe { std::env::set_var("MEMRA_FA_DEEP", "1"); }
            let mut o_dpdc = e.zeros(hd * nh)?;
            e.fa_decode_dc(&qd, &kview, &vview, &mut o_dpdc, hd, nh, nhkv, &tdev, bucket,
                           scale, k_tok_bytes, v_tok_bytes, false)?;
            let (adc, bdc) = (e.dtoh(&o_v4dc)?, e.dtoh(&o_dpdc)?);
            let bd2 = adc.iter().zip(&bdc).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            // dc-vs-eager per arm: MUST MATCH EACH OTHER element-wise. When bucket straddles
            // a split-ladder rung (e.g. t_kv 3072 / bucket 3136), dc legitimately differs
            // from eager (the documented v4 bucketing property: same math, different split
            // grouping) — the deep arm must reproduce v4's straddle EXACTLY, not hide it.
            let bd3_v4 = a.iter().zip(&adc).map(|(x, y)| x.to_bits() != y.to_bits()).collect::<Vec<_>>();
            let bd3_dp = b.iter().zip(&bdc).map(|(x, y)| x.to_bits() != y.to_bits()).collect::<Vec<_>>();
            let straddle_ok = bd3_v4 == bd3_dp;
            let n3 = bd3_dp.iter().filter(|&&x| x).count();
            println!("deep-vs-v4 dc t_kv={d} bucket={bucket}: bitdiff={bd2} (dc-vs-eager {n3}, straddle-matched {straddle_ok}) {}",
                     if bd2 == 0 && straddle_ok { "OK" } else { fails += 1; "FAIL" });
        }
    }

    // ---- 2. TIMING (production dc form; interleaved rounds, medians) ----
    let time_arm = |arm: &str, d: usize, tdev: &cudarc::driver::CudaSlice<i32>,
                    o: &mut cudarc::driver::CudaSlice<f32>|
                    -> Result<f64, Box<dyn std::error::Error>> {
        unsafe { std::env::set_var("MEMRA_FA_DEEP", arm); }
        let kview = e.view_u8(&kc, d * k_tok_bytes);
        let vview = e.view_u8(&vc, d * v_tok_bytes);
        for _ in 0..20 {
            e.fa_decode_dc(&qd, &kview, &vview, o, hd, nh, nhkv, tdev, d,
                           scale, k_tok_bytes, v_tok_bytes, false)?;
        }
        e.stream().synchronize()?;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            e.fa_decode_dc(&qd, &kview, &vview, o, hd, nh, nhkv, tdev, d,
                           scale, k_tok_bytes, v_tok_bytes, false)?;
        }
        e.stream().synchronize()?;
        Ok(t0.elapsed().as_secs_f64() * 1e6 / iters as f64)
    };
    println!("\ntiming (us/call, dc form incl. memsets+combine; iters={iters}, 3 interleaved rounds, median):");
    for &d in &time_depths {
        let tdev = e.htod_i32(&[d as i32])?;
        let mut o = e.zeros(hd * nh)?;
        let (mut t4, mut tdp) = (Vec::new(), Vec::new());
        for r in 0..3 {
            // both orders per round (fa_ab_bench pattern)
            if r % 2 == 0 {
                t4.push(time_arm("0", d, &tdev, &mut o)?);
                tdp.push(time_arm("1", d, &tdev, &mut o)?);
            } else {
                tdp.push(time_arm("1", d, &tdev, &mut o)?);
                t4.push(time_arm("0", d, &tdev, &mut o)?);
            }
        }
        let (r4, rdp) = (t4.clone(), tdp.clone());
        let med = |v: &mut Vec<f64>| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] };
        let (m4, mdp) = (med(&mut t4), med(&mut tdp));
        println!("t_kv={d}: v4 {m4:.2} us | deep {mdp:.2} us | ratio {:.3}x  (v4 reps {r4:.2?} deep reps {rdp:.2?})",
                 m4 / mdp);
    }
    if fails > 0 { println!("\nFAILS={fails}"); std::process::exit(1); }
    println!("\nALL BIT GATES GREEN");
    Ok(())
}
