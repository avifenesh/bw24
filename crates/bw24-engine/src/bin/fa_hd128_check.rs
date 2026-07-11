//! hd128 FA-prefill twin gate (2026-07-07): the flash_attn.cu prefill kernels are
//! template-stamped at HEAD_DIM 256 (original names) and 128 (`_hd128`, the MiniMax-M3
//! class). This standalone microbench validates EVERY hd128 twin against the sdpa_naive
//! oracle at T up to 512, rel < 1e-2 (bf16-MMA class tolerance; the f32 twins measure
//! ~1e-3, the q8_0/q5_1 quant twins ride the same 6e-2 bound as kernel_check's hd256
//! rows — quantization noise, not the FA math). No model needed.
//!
//!   fa_prefill_f32_pp_hd128   (default prefill)      vs sdpa_naive @ hd128
//!   fa_prefill_f32_hd128      (floor twin)            vs sdpa_naive @ hd128
//!   fa_prefill_q_hd128        (quant view)            vs sdpa_naive @ hd128 (quant tol)
//!   fa_prefill_qw/_db_hd128   (dequant-once ws twins) vs fa_prefill_q_hd128 BIT-IDENTITY
//!
//! Shapes: the generic (nh=16, nhkv=4) harness geometry at T=16..512 including
//! BK/BLOCK_Q-unaligned tails, plus the real M3 attention geometry (nh=64, nhkv=8).
//! hd256 rows re-run alongside to prove the stamped-256 dispatch is unchanged.

use bw24_engine::Engine;

fn pr(i: usize) -> f32 { (((i.wrapping_mul(2654435761)) >> 8) & 0xffff) as f32 / 32768.0 - 1.0 }

fn rel_diff(a: &[f32], b: &[f32]) -> f32 {
    let d = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    let s = a.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-3);
    d / s
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let e = Engine::new(0)?;
    let mut fails = 0usize;

    // (head_dim, n_head, n_head_kv, label)
    let geoms = [
        (128usize, 16usize, 4usize, "generic hd128"),
        (128, 64, 8, "M3 hd128 (64h/8kv)"),
        (256, 16, 4, "hd256 regression"),
    ];
    // T cases: aligned, tail-unaligned (BLOCK_Q=64, BK=32), and the 512 gate bound.
    let t_cases = [16usize, 64, 100, 256, 512];

    for &(hd, nh, nhkv, label) in &geoms {
        let scale = 1.0 / (hd as f32).sqrt();
        for &t in &t_cases {
            let tkv = t;
            let q: Vec<f32> = (0..hd * nh * t).map(|i| pr(i) * 0.2).collect();
            let k: Vec<f32> = (0..hd * nhkv * tkv).map(|i| pr(i + 7) * 0.2).collect();
            let v: Vec<f32> = (0..hd * nhkv * tkv).map(|i| pr(i + 11) * 0.2).collect();
            let qd = e.htod(&q)?;
            let kd = e.htod(&k)?;
            let vd = e.htod(&v)?;

            // oracle: sdpa_naive (the current M3 bring-up path)
            let mut o_ref = e.zeros(hd * nh * t)?;
            e.sdpa_naive(&qd, &kd, &vd, &mut o_ref, hd, nh, nhkv, t, tkv, scale, true)?;
            let oref = e.dtoh(&o_ref)?;

            // 1) default prefill kernel (fa_prefill_f32_pp / _hd128)
            let mut o_pp = e.zeros(hd * nh * t)?;
            e.fa_prefill(&qd, &kd, &vd, &mut o_pp, hd, nh, nhkv, t, tkv, scale, true)?;
            let rel = rel_diff(&oref, &e.dtoh(&o_pp)?);
            println!("[{label}] fa_prefill(pp)  T={t}: rel={rel:.2e} {}",
                     if rel < 1e-2 { "OK" } else { fails += 1; "FAIL" });

            // 2) floor twin (fa_prefill_f32 / _hd128) via BW24_FA_FLOOR
            unsafe { std::env::set_var("BW24_FA_FLOOR", "1"); }
            let mut o_fl = e.zeros(hd * nh * t)?;
            e.fa_prefill(&qd, &kd, &vd, &mut o_fl, hd, nh, nhkv, t, tkv, scale, true)?;
            unsafe { std::env::remove_var("BW24_FA_FLOOR"); }
            let rel = rel_diff(&oref, &e.dtoh(&o_fl)?);
            println!("[{label}] fa_prefill(flr) T={t}: rel={rel:.2e} {}",
                     if rel < 1e-2 { "OK" } else { fails += 1; "FAIL" });

            // 3) quant view twin (fa_prefill_q / _hd128): quantize K/V into a resident
            //    q8_0/q5_1 cache first. Tolerance = kernel_check's quant bound (6e-2).
            let kv_dim = hd * nhkv;
            let (kbb, vbb) = bw24_engine::kv_blk_bytes();  // env-selected KV formats
            let k_tok_bytes = (kv_dim / 32) * kbb;
            let v_tok_bytes = (kv_dim / 32) * vbb;
            let mut kc = e.alloc_u8(tkv * k_tok_bytes)?;
            let mut vc = e.alloc_u8(tkv * v_tok_bytes)?;
            for tok in 0..tkv {
                let k_row = kd.slice(tok * kv_dim..(tok + 1) * kv_dim);
                let v_row = vd.slice(tok * kv_dim..(tok + 1) * kv_dim);
                e.append_kv_quantized_view(&k_row, &v_row, &mut kc, &mut vc, tok,
                                           kv_dim, kv_dim, k_tok_bytes, v_tok_bytes, false)?;
            }
            let kview = e.view_u8(&kc, tkv * k_tok_bytes);
            let vview = e.view_u8(&vc, tkv * v_tok_bytes);
            let mut o_q = e.zeros(hd * nh * t)?;
            e.fa_prefill_view(&qd, &kview, &vview, &mut o_q, hd, nh, nhkv, t, tkv, scale,
                              true, k_tok_bytes, v_tok_bytes)?;
            let oq = e.dtoh(&o_q)?;
            let rel = rel_diff(&oref, &oq);
            println!("[{label}] fa_prefill_q    T={t}: rel={rel:.2e} {}",
                     if rel < 6e-2 { "OK" } else { fails += 1; "FAIL" });

            // 4) dequant-once workspace twins (qw db-on and db-off): BIT-IDENTITY vs the
            //    inline-dequant kernel (the same contract kernel_check pins at hd256).
            for db in ["1", "0"] {
                unsafe { std::env::set_var("BW24_PRIME_DEQW_DB", db); }
                let mut o_ws = e.zeros(hd * nh * t)?;
                e.fa_prefill_view_ws(&qd, &kview, &vview, &mut o_ws, hd, nh, nhkv, t, tkv,
                                     scale, true, k_tok_bytes, v_tok_bytes)?;
                let ows = e.dtoh(&o_ws)?;
                let bitdiff = oq.iter().zip(&ows)
                    .filter(|(x, y)| x.to_bits() != y.to_bits()).count();
                println!("[{label}] fa_prefill_qw(db={db}) T={t}: bitdiff={bitdiff} {}",
                         if bitdiff == 0 { "OK" } else { fails += 1; "FAIL" });
            }
            unsafe { std::env::remove_var("BW24_PRIME_DEQW_DB"); }
        }
    }

    if fails == 0 {
        println!("FA-HD128 GREEN");
    } else {
        println!("FA-HD128 FAIL ({fails})");
        std::process::exit(1);
    }
    Ok(())
}
