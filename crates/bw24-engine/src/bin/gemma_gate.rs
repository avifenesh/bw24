//! gemma4 bring-up gate: prefill-only forward on raw token ids, prints greedy continuation
//! (each step re-runs the full prefill — O(n^2), gate-only) + top-8 (id, logit) of the first
//! step for logit-level comparison against llama.cpp on the IDENTICAL GGUF.
use bw24_engine::Engine;
use bw24_engine::hybrid::HybridModel;
use bw24_gguf::GgufFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: gemma-gate <model.gguf> <tok ids...>");
    let mut toks: Vec<u32> = std::env::args().skip(2).filter_map(|s| s.parse().ok()).collect();
    // BW24_VERIFY_GATE=K: batched-verify self-consistency — decode the prompt tokenwise
    // (reference), then on a fresh cache decode the prefix and run ONE decode_step_t over the
    // last K tokens; per-position argmax must match the tokenwise chain (the spec K-gate).
    if let Ok(kk) = std::env::var("BW24_VERIFY_GATE") {
        let k: usize = kk.parse().unwrap_or(4);
        let e = bw24_engine::Engine::new(0)?;
        let g = bw24_gguf::GgufFile::open(&path)?;
        let model = bw24_engine::hybrid::HybridModel::load(&e, &g)?;
        let n_vocab = model.output.out_features();
        let toks: Vec<u32> = std::env::args().skip(2).filter_map(|s| s.parse().ok()).collect();
        assert!(toks.len() > k + 1, "prompt must exceed K+1");
        let split = toks.len() - k;
        // reference: tokenwise decode over the whole prompt
        let mut c1 = bw24_engine::cache::Cache::new(&e, &model.cfg, toks.len() + 8)?;
        let mut ref_am: Vec<usize> = Vec::new();
        for (i, &tk) in toks.iter().enumerate() {
            let l = model.decode_step(&e, tk, &mut c1)?;
            if i >= split {
                ref_am.push(bw24_engine::forward::argmax(&l));
            }
        }
        // candidate: prefix tokenwise, tail as ONE batched verify
        let mut c2 = bw24_engine::cache::Cache::new(&e, &model.cfg, toks.len() + 8)?;
        for &tk in &toks[..split] { let _ = model.decode_step(&e, tk, &mut c2)?; }
        let lv = model.decode_step_t(&e, &toks[split..], split, &mut c2)?;
        let mut all_ok = true;
        for i in 0..k {
            let am = bw24_engine::forward::argmax(&lv[i * n_vocab..(i + 1) * n_vocab]);
            let ok = am == ref_am[i];
            all_ok &= ok;
            println!("verify pos {i}: batched={am} tokenwise={} {}", ref_am[i],
                     if ok { "MATCH" } else { "MISMATCH" });
        }
        println!("VERIFY-GATE K={k}: {}", if all_ok { "PASS" } else { "FAIL" });
        return Ok(());
    }
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
