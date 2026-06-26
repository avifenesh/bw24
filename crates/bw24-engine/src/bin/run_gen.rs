//! M7: end-to-end greedy generation with KV cache. Serves a model: prompt tokens -> generated tokens.

use bw24_engine::Engine;
use bw24_engine::hybrid::HybridModel;
use bw24_engine::forward::argmax;
use bw24_gguf::GgufFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: run-gen <model.gguf> [tok ids...]");
    let e = Engine::new(0)?;
    let g = GgufFile::open(&path)?;
    let model = HybridModel::load(&e, &g)?;
    println!("loaded {} ({} layers)", g.arch().unwrap_or("?"), model.cfg.n_layer);

    let prompt: Vec<u32> = std::env::args().skip(2).filter_map(|s| s.parse().ok()).collect();
    let prompt = if prompt.is_empty() { vec![55u32] } else { prompt };
    println!("prompt tokens: {prompt:?}");

    // --- correctness gate: decode-step prefix logits MUST match the prefill forward ---
    let prefill = model.forward_last(&e, &prompt)?;
    // decode the prompt step by step, capture last logits
    let mut cache = bw24_engine::cache::Cache::new(&e, &model.cfg, prompt.len() + 64)?;
    let mut dec_logits = Vec::new();
    for &t in &prompt { dec_logits = model.decode_step(&e, t, &mut cache)?; }
    let am_p = argmax(&prefill); let am_d = argmax(&dec_logits);
    let maxdiff = prefill.iter().zip(&dec_logits).map(|(a,b)| (a-b).abs()).fold(0.0, f32::max);
    println!("prefill argmax={am_p}  decode argmax={am_d}  logit maxdiff={maxdiff:.3e}  {}",
             if am_p == am_d { "MATCH" } else { "MISMATCH" });
    assert_eq!(am_p, am_d, "decode-step diverges from prefill — cache threading bug");

    // --- generate + time decode tok/s (honest Stage-A baseline) ---
    let n_new = std::env::var("BW24_NGEN").ok().and_then(|s| s.parse().ok()).unwrap_or(16usize);
    let mut cache = bw24_engine::cache::Cache::new(&e, &model.cfg, prompt.len() + n_new + 8)?;
    let mut ll = Vec::new();
    for &t in &prompt { ll = model.decode_step(&e, t, &mut cache)?; }
    e.stream().synchronize()?;
    let t0 = std::time::Instant::now();
    let mut out = Vec::with_capacity(n_new);
    for _ in 0..n_new {
        let next = argmax(&ll) as u32;
        out.push(next);
        ll = model.decode_step(&e, next, &mut cache)?;
    }
    e.stream().synchronize()?;
    let dt = t0.elapsed().as_secs_f64();
    println!("generated {} tokens in {:.3}s = {:.2} tok/s (Stage-A f32-dequant decode)", n_new, dt, n_new as f64 / dt);
    println!("tokens: {out:?}");
    Ok(())
}
