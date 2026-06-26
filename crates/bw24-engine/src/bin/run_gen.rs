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
    let mut cache = bw24_engine::cache::Cache::new(&model.cfg);
    let mut dec_logits = Vec::new();
    for &t in &prompt { dec_logits = model.decode_step(&e, t, &mut cache)?; }
    let am_p = argmax(&prefill); let am_d = argmax(&dec_logits);
    let maxdiff = prefill.iter().zip(&dec_logits).map(|(a,b)| (a-b).abs()).fold(0.0, f32::max);
    println!("prefill argmax={am_p}  decode argmax={am_d}  logit maxdiff={maxdiff:.3e}  {}",
             if am_p == am_d { "MATCH" } else { "MISMATCH" });
    assert_eq!(am_p, am_d, "decode-step diverges from prefill — cache threading bug");

    // --- generate ---
    let out = model.generate(&e, &prompt, 16)?;
    println!("generated: {out:?}");
    Ok(())
}
