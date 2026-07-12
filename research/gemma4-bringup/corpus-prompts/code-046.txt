//! M0: pure-dense forward (spine proof). Loads a dense GGUF, runs a token sequence,
//! prints the argmax + top-k of the last-token logits. Validates the whole pipeline end-to-end.

use bw24_engine::Engine;
use bw24_engine::model::Model;
use bw24_engine::forward::argmax;
use bw24_gguf::GgufFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: run-dense <dense_model.gguf> [tok ids...]");
    let e = Engine::new(0)?;
    let g = GgufFile::open(&path)?;
    println!("GPU: {}  model arch: {:?}", e.ctx().name()?, g.arch());

    let model = Model::load_dense(&e, &g)?;
    println!("loaded: n_layer={} n_embd={} n_head={}/{} head_dim={} n_ff={} n_vocab={}",
        model.cfg.n_layer, model.cfg.n_embd, model.cfg.n_head, model.cfg.n_head_kv,
        model.cfg.head_dim_k, model.cfg.n_ff, model.cfg.n_vocab);

    // token ids from args (after model path), default to a tiny BOS-ish sequence.
    let toks: Vec<u32> = std::env::args().skip(2).filter_map(|s| s.parse().ok()).collect();
    let toks = if toks.is_empty() { vec![1u32, 2, 3, 4] } else { toks };
    println!("tokens: {toks:?}");

    let logits = model.forward_last(&e, &toks)?;
    let am = argmax(&logits);
    // top-5
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    println!("argmax token = {am}  logit = {:.4}", logits[am]);
    println!("top-5: {:?}", idx[..5].iter().map(|&i| (i, logits[i])).collect::<Vec<_>>());

    // sanity: no NaN/Inf
    let bad = logits.iter().filter(|v| !v.is_finite()).count();
    println!("non-finite logits: {bad}  (total {})", logits.len());
    assert_eq!(bad, 0, "non-finite logits — forward is broken");
    println!("\nrun-dense: forward produced finite logits. (validate argmax vs llama.cpp next)");
    Ok(())
}
