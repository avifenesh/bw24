//! FAST-ROUTER crossover bench (lane/fast-router, 2026-08-02): plain per-(expert,token) w8
//! router form vs the register-tiled 8x8 batch twin across t, with the m-DEPENDENT cuBLASLt
//! GEMM (`linear`) as the headroom reference (never a dispatch arm — it is the kernel the
//! exactness contract bans from prefill). The twin is BIT-IDENTICAL per row (kernel-check
//! gate), so this sweep decides ROUTER_BATCH_MIN_T on pure perf. Real q35 router weights
//! when the GGUF is present (same path chain as kernel-check), synthetic 256x2048 otherwise.
//! JSONL rows to stdout; medians of 5 interleaved passes.
//!
//! Killed arms (receipts research/fast-router-20260802/): the 8x16 tile lost to 8x8 at
//! every t (crossover-router-tiles.jsonl); the same-shape sigmoid_dot_rows twin (out_f=1)
//! measured 0.62-0.89x at every prefill t (crossover-router.jsonl) — both were
//! bit-identity-green before dying, per flags doctrine.
//!
//! usage: router-batch-bench [model.gguf]

use memra_engine::Engine;
use memra_validate::pr;

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let e = Engine::new(0)?;
    eprintln!("GPU: {}", e.ctx().name()?);

    // real router weights when available (kernel-check path chain), else synthetic.
    let arg: Option<String> = std::env::args().nth(1);
    let mut cands: Vec<String> = Vec::new();
    if let Some(a) = &arg { cands.push(a.clone()); }
    if let Ok(d) = std::env::var("MEMRA_KC_MODELS_DIR") {
        cands.push(format!("{}/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf", d.trim_end_matches('/')));
    }
    cands.push("/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf".into());
    let (n_embd, n_experts, wf, src);
    if let Some(p) = cands.iter().find(|p| std::path::Path::new(p).exists()) {
        let g = memra_gguf::GgufFile::open(p)?;
        let tw = g.find("blk.0.ffn_gate_inp.weight").expect("gate_inp");
        n_embd = tw.ne[0] as usize;
        n_experts = tw.ne[1] as usize;
        let le = |b: &[u8]| f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        wf = g.tensor_data(tw).chunks_exact(4).map(le).collect::<Vec<f32>>();
        src = p.clone();
    } else {
        n_embd = 2048; n_experts = 256;
        wf = (0..n_experts * n_embd).map(|i| (pr(i + 3) - 0.5) * 0.1).collect();
        src = "synthetic".into();
    }
    eprintln!("weights: {src} (n_embd={n_embd}, n_experts={n_experts})");

    let t_max = 2048usize;
    let x: Vec<f32> = (0..t_max * n_embd).map(|i| (pr(i + 7) - 0.5) * 4.0).collect();
    let wd = e.htod(&wf)?; let xd = e.htod(&x)?;

    let ts_sweep = [1usize, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512,
                    768, 1024, 1536, 2048];
    println!("{{\"bench\":\"router-batch-crossover\",\"weights\":\"{src}\",\"n_embd\":{n_embd},\"n_experts\":{n_experts},\"passes\":5}}");
    for &t in &ts_sweep {
        let n_iter = (20000 / t).clamp(30, 4000);
        // interleaved passes: plain,batch,cublas per pass x5; median per form.
        let mut us: [Vec<f64>; 3] = [vec![], vec![], vec![]];
        for _ in 0..5 {
            for (slot, batch) in [(0usize, false), (1usize, true)] {
                for _ in 0..3 { let _ = e.router_gemv_form(&wd, &xd, n_embd, n_experts, t, true, batch)?; }
                e.stream().synchronize()?;
                let t0 = std::time::Instant::now();
                for _ in 0..n_iter { let _ = e.router_gemv_form(&wd, &xd, n_embd, n_experts, t, true, batch)?; }
                e.stream().synchronize()?;
                us[slot].push(t0.elapsed().as_secs_f64() * 1e6 / n_iter as f64);
            }
            {
                for _ in 0..3 { let _ = e.linear(&xd, &wd, t, n_embd, n_experts)?; }
                e.stream().synchronize()?;
                let t0 = std::time::Instant::now();
                for _ in 0..n_iter { let _ = e.linear(&xd, &wd, t, n_embd, n_experts)?; }
                e.stream().synchronize()?;
                us[2].push(t0.elapsed().as_secs_f64() * 1e6 / n_iter as f64);
            }
        }
        let (rp, rb, rc) = (median(&mut us[0]), median(&mut us[1]), median(&mut us[2]));
        println!("{{\"op\":\"router\",\"t\":{t},\"n_iter\":{n_iter},\"plain_us\":{rp:.3},\"batch_us\":{rb:.3},\"cublas_us\":{rc:.3},\"speedup\":{:.3}}}",
                 rp / rb);
    }
    Ok(())
}
