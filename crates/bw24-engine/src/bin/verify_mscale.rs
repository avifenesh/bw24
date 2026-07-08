//! Verify m-scaling probe (35B spec-gap triage, lane/close35 2026-07-08): times the spec
//! VERIFY forward (`decode_step_t_h_emb_dev` — the exact hot-loop kernel chain: resident-embed
//! gather, device logits, no host logits dtoh) at m = 1,2,3,4,6 from a FIXED primed depth,
//! rolling the cache back between calls so every timed call sees the identical state.
//! Prints us/call median + p10/p90 per m, the m=1-normalized cost curve, and the eager
//! decode_step reference. Read the curve against llama-bench `-d <depth> -p 1,2,3,4,6 -n 0`
//! (their verify batch = llama_decode of m tokens at depth — same dispatch as their MTP verify).
//!
//! MEASUREMENT-ONLY: no kernel/dispatch change; pure Instant+sync timing around existing calls.
//!
//! usage: verify-mscale <model.gguf> [depth=512] [reps=40] [m-list="1,2,3,4,6"]
//! env law: BW24_FAST=1 BW24_GEMM=1 BW24_MMVQ=1 BW24_FA_VEC=1 (+BW24_MOE_CACHE=1 on the 35B)

use bw24_engine::Engine;
use bw24_engine::hybrid::HybridModel;
use bw24_engine::forward::argmax;
use bw24_gguf::GgufFile;

fn pct(sorted: &[f64], p: f64) -> f64 {
    let i = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[i]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: verify-mscale <model> [depth] [reps] [ms]");
    let depth: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(512);
    let reps: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(40);
    let ms: Vec<usize> = std::env::args().nth(4)
        .unwrap_or_else(|| "1,2,3,4,6".to_string())
        .split(',').filter_map(|s| s.parse().ok()).collect();
    let max_m = ms.iter().copied().max().unwrap_or(6);

    let e = Engine::new(0)?;
    let g = GgufFile::open(&path)?;
    let model = HybridModel::load(&e, &g)?;
    let n_embd = model.cfg.n_embd as usize;

    // Same synthetic prompt family as decode-bench (comparable depth state).
    let prompt: Vec<u32> = (0..depth).map(|i| (100 + (i * 7) % 900) as u32).collect();
    let mut cache = bw24_engine::cache::Cache::new(&e, &model.cfg, depth + max_m + 32)?;
    let t_prime = std::time::Instant::now();
    let mut last_logits: Vec<f32> = if depth >= bw24_engine::hybrid_forward::PRIME_MIN_T {
        let (l, _h, _hiddens) = model.prime_cache(&e, &prompt, &mut cache)?;
        l
    } else {
        let mut l = Vec::new();
        for &t in &prompt { l = model.decode_step(&e, t, &mut cache)?; }
        l
    };
    e.stream().synchronize()?;
    println!("primed depth={} in {:.2}s", cache.pos, t_prime.elapsed().as_secs_f64());

    // Realistic verify tokens: the model's OWN greedy continuation (real expert routing),
    // generated eagerly then rolled back.
    let snap = cache.snapshot(&e)?;
    let pos0 = cache.pos;
    let mut gold: Vec<u32> = Vec::with_capacity(max_m);
    let mut ll = last_logits.clone();
    for _ in 0..max_m {
        let nx = argmax(&ll) as u32;
        gold.push(nx);
        ll = model.decode_step(&e, nx, &mut cache)?;
    }
    cache.rollback(&e, &snap, 0)?;
    let _ = &mut last_logits;
    println!("verify tokens (greedy continuation): {gold:?}");

    // Resident embed table — the spec hot loop's gather source.
    let embd_gpu = model.embd_gpu.get_or_init(|| {
        e.upload_u8(&model.embd.raw).expect("embed table upload")
    });
    let (embd_qt, embd_rb) = model.embd.qt_and_row_bytes(n_embd);

    // BW24_MSCALE_NOEAGER=1: skip the eager reference (keeps an nsys trace verify-only).
    // BW24_MSCALE_PROFILE=1: bracket the TIMED verify reps in cuProfilerStart/Stop so
    // `nsys --capture-range=cudaProfilerApi` records ONLY the verify kernel chain (run with a
    // single-m list; warmups + rollbacks sit outside the bracket per rep is not possible —
    // rollback D2D copies are inside the window and must be subtracted by name).
    let profile = std::env::var("BW24_MSCALE_PROFILE").is_ok();
    // Eager decode_step reference (the plain-decode per-token cost at this depth).
    if std::env::var("BW24_MSCALE_NOEAGER").is_err() {
        for _ in 0..3 { let _ = model.decode_step(&e, gold[0], &mut cache)?; cache.rollback(&e, &snap, 0)?; }
        e.stream().synchronize()?;
        let mut ts: Vec<f64> = Vec::with_capacity(reps);
        for _ in 0..reps {
            e.stream().synchronize()?;
            let t0 = std::time::Instant::now();
            let _ = model.decode_step(&e, gold[0], &mut cache)?;
            e.stream().synchronize()?;
            ts.push(t0.elapsed().as_secs_f64() * 1e6);
            cache.rollback(&e, &snap, 0)?;
        }
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("eager decode_step  @d{depth}: median {:8.1} us  p10 {:8.1}  p90 {:8.1}",
                 pct(&ts, 0.5), pct(&ts, 0.1), pct(&ts, 0.9));
    }

    let mut med1 = 0.0f64;
    for &m in &ms {
        let toks = &gold[0..m];
        // warmup
        for _ in 0..3 {
            let _ = model.decode_step_t_h_emb_dev(&e, toks, pos0, &mut cache,
                                                  Some((embd_gpu, embd_qt, embd_rb)))?;
            cache.rollback(&e, &snap, 0)?;
        }
        e.stream().synchronize()?;
        if profile { unsafe { cudarc::driver::sys::cuProfilerStart().result()?; } }
        let mut ts: Vec<f64> = Vec::with_capacity(reps);
        for _ in 0..reps {
            e.stream().synchronize()?;
            let t0 = std::time::Instant::now();
            let _ = model.decode_step_t_h_emb_dev(&e, toks, pos0, &mut cache,
                                                  Some((embd_gpu, embd_qt, embd_rb)))?;
            e.stream().synchronize()?;
            ts.push(t0.elapsed().as_secs_f64() * 1e6);
            cache.rollback(&e, &snap, 0)?;
        }
        if profile { unsafe { cudarc::driver::sys::cuProfilerStop().result()?; } }
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = pct(&ts, 0.5);
        if m == 1 || med1 == 0.0 { if m == ms[0] { med1 = med; } }
        println!("verify m={m} @d{depth}: median {:8.1} us  p10 {:8.1}  p90 {:8.1}  | x{:.3} vs m={}  | {:7.1} us/tok",
                 med, pct(&ts, 0.1), pct(&ts, 0.9), med / med1, ms[0], med / m as f64);
    }
    Ok(())
}
