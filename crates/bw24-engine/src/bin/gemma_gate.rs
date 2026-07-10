//! gemma4 bring-up gate: prefill-only forward on raw token ids, prints greedy continuation
//! (each step re-runs the full prefill — O(n^2), gate-only) + top-8 (id, logit) of the first
//! step for logit-level comparison against llama.cpp on the IDENTICAL GGUF.
use bw24_engine::Engine;
use bw24_engine::hybrid::HybridModel;
use bw24_gguf::GgufFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: gemma-gate <model.gguf> <tok ids...>");
    let mut toks: Vec<u32> = std::env::args().skip(2).filter_map(|s| s.parse().ok()).collect();
    let n_new: usize = std::env::var("BW24_NGEN").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    let e = Engine::new(0)?;
    let g = GgufFile::open(&path)?;
    let model = HybridModel::load(&e, &g)?;
    println!("loaded {} ({} layers), prompt {} toks", g.arch().unwrap_or("?"), model.cfg.n_layer, toks.len());

    for step in 0..n_new {
        let logits = model.forward_last(&e, &toks)?;
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        if step == 0 {
            let top: Vec<String> = idx[..8].iter().map(|&i| format!("{i}:{:.4}", logits[i])).collect();
            println!("step0 top8: {}", top.join(" "));
            println!("step0 logits[0..3]={:?} [9079]={:.4} [506]={:.4}",
                     &logits[..3], logits[9079], logits[506]);
        }
        toks.push(idx[0] as u32);
        println!("step {step}: tok {}", idx[0]);
    }
    println!("continuation: {:?}", &toks[toks.len() - n_new..]);
    Ok(())
}
