//! M6: hybrid (qwen35) forward. Loads the daily-driver hybrid model, runs tokens, prints
//! argmax + top-5 of last-token logits. Validate vs llama.cpp ground truth (tools/llama_logits).

use bw24_engine::Engine;
use bw24_engine::hybrid::HybridModel;
use bw24_engine::forward::argmax;
use bw24_gguf::GgufFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: run-hybrid <hybrid_model.gguf> [tok ids...]");
    let e = Engine::new(0)?;
    let g = GgufFile::open(&path)?;
    println!("GPU: {}  arch: {:?}", e.ctx().name()?, g.arch());

    let model = HybridModel::load(&e, &g)?;
    let full = model.cfg.n_full_attn_layers();
    println!("loaded hybrid: n_layer={} ({} full-attn, {} linear) n_embd={} n_head={}/{} head_dim={} n_vocab={}",
        model.cfg.n_layer, full, model.cfg.n_layer - full, model.cfg.n_embd,
        model.cfg.n_head, model.cfg.n_head_kv, model.cfg.head_dim_k, model.cfg.n_vocab);

    let toks: Vec<u32> = std::env::args().skip(2).filter_map(|s| s.parse().ok()).collect();
    let toks = if toks.is_empty() { vec![1u32, 2, 3, 4] } else { toks };
    println!("tokens: {toks:?}");

    let logits = model.forward_last(&e, &toks)?;
    let am = argmax(&logits);
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    let nan = logits.iter().filter(|v| !v.is_finite()).count();
    println!("argmax token = {am}  logit = {:.4}  non-finite={nan}/{}", logits[am], logits.len());
    println!("top-5: {:?}", idx[..5].iter().map(|&i| (i, logits[i])).collect::<Vec<_>>());
    let bad = logits.iter().filter(|v| !v.is_finite()).count();
    println!("non-finite: {bad}");
    assert_eq!(bad, 0, "non-finite logits");
    Ok(())
}
