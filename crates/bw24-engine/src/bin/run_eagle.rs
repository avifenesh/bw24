//! EAGLE3.1 greedy-chain speculative decode driver + self-consistency gate (EAGLE-PLAN §6).
//!
//! Greedy spec decode is mathematically EXACT: generate_spec_eagle(prompt,n,k) MUST produce a
//! token-IDENTICAL sequence to generate(prompt,n) for every K. This binary asserts that equality
//! for K in {1,2,3,4} and prints the acceptance rate (n_accepted/drafted) + tok/s vs plain generate.
//! A wrong draft still yields correct output via the bonus token but with ~0 acceptance — so BOTH
//! gates must hold: identical tokens AND acceptance > 0.
//!
//! Run:
//!   export PATH=/usr/local/cuda-13.1/bin:$PATH
//!   BW24_FAST=1 BW24_EAGLE=/home/avifenesh/ai-ml/hf-models/eagle3-qwen35-9b \
//!     cargo run --release -p bw24-engine --bin run-eagle -- <target.gguf> [tok ids...]

use bw24_engine::Engine;
use bw24_engine::hybrid::HybridModel;
use bw24_engine::eagle::Eagle3Draft;
use bw24_gguf::GgufFile;

fn first_divergence(a: &[u32], b: &[u32]) -> Option<usize> {
    let n = a.len().min(b.len());
    for i in 0..n { if a[i] != b[i] { return Some(i); } }
    if a.len() != b.len() { return Some(n); }
    None
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: run-eagle <target.gguf> [tok ids...]");
    let eagle_dir = std::env::var("BW24_EAGLE")
        .expect("set BW24_EAGLE=<eagle3 draft dir or .safetensors path>");
    let e = Engine::new(0)?;
    let g = GgufFile::open(&path)?;
    let model = HybridModel::load(&e, &g)?;
    println!("loaded target {} ({} layers, nextn={}, n_embd={})", g.arch().unwrap_or("?"),
             model.cfg.n_layer, model.cfg.nextn_predict_layers, model.cfg.n_embd);

    // --- LOAD THE EAGLE3 DRAFT (asset gate). If absent/unloadable, report and STOP. ---
    let draft = match Eagle3Draft::load(&e, std::path::Path::new(&eagle_dir)) {
        Ok(d) => d,
        Err(err) => {
            eprintln!("EAGLE3 ASSET-BLOCKED: failed to load draft from {eagle_dir}: {err}");
            std::process::exit(3);
        }
    };
    println!("loaded EAGLE3 draft: n_embd={} n_head={}:{} head_dim={} n_ff={} draft_vocab={} \
              rope(dim={},theta={:.0}) aux_layers={:?}",
             draft.n_embd, draft.n_head, draft.n_head_kv, draft.head_dim, draft.n_ff,
             draft.draft_vocab, draft.rope_dim_count, draft.rope_theta, draft.aux_layers);

    let prompt: Vec<u32> = std::env::args().skip(2).filter_map(|s| s.parse().ok()).collect();
    let prompt = if prompt.is_empty() { vec![785u32, 3491, 374, 264] } else { prompt };
    println!("prompt tokens: {prompt:?}");
    let n_new = std::env::var("BW24_NGEN").ok().and_then(|s| s.parse().ok()).unwrap_or(64usize);

    // --- ORACLE: plain greedy generate (the exact reference) ---
    e.stream().synchronize()?;
    let t0 = std::time::Instant::now();
    let gold = model.generate(&e, &prompt, n_new)?;
    e.stream().synchronize()?;
    let gen_dt = t0.elapsed().as_secs_f64();
    let gen_tps = n_new as f64 / gen_dt;
    println!("\n[generate]    {n_new} tok in {gen_dt:.3}s = {gen_tps:.2} tok/s");
    println!("  tokens: {gold:?}");

    let mut all_pass = true;
    let mut any_accept = false;
    for &k in &[1usize, 2, 3, 4] {
        e.stream().synchronize()?;
        let t1 = std::time::Instant::now();
        let (spec, drafted, accepted) = model.generate_spec_eagle(&e, &draft, &prompt, n_new, k)?;
        e.stream().synchronize()?;
        let spec_dt = t1.elapsed().as_secs_f64();
        let spec_tps = n_new as f64 / spec_dt;
        let acc_rate = if drafted > 0 { accepted as f64 / drafted as f64 } else { 0.0 };

        let pass = first_divergence(&gold, &spec).is_none();
        all_pass &= pass;
        any_accept |= accepted > 0;
        println!("\n[generate_spec_eagle K={k}] {n_new} tok in {spec_dt:.3}s = {spec_tps:.2} tok/s \
                  ({:.2}x vs generate)", spec_tps / gen_tps);
        println!("  acceptance: {accepted}/{drafted} = {:.1}%   self-consistency: {}",
                 acc_rate * 100.0, if pass { "PASS (identical to generate)" } else { "FAIL" });
        if !pass {
            let idx = first_divergence(&gold, &spec).unwrap();
            println!("  FIRST DIVERGENCE at index {idx}:");
            println!("    generate          [..]: {:?}", &gold[idx.saturating_sub(2)..(idx + 3).min(gold.len())]);
            println!("    generate_spec_eagle[..]: {:?}", &spec[idx.saturating_sub(2)..(idx + 3).min(spec.len())]);
            println!("    eagle tokens: {spec:?}");
        }
        if pass && accepted == 0 {
            println!("  WARNING: acceptance == 0 with identical output — the draft forward is likely \
                      wrong (bonus-token masking). No speedup possible.");
        }
    }

    println!("\n=== EAGLE3 SELF-CONSISTENCY {} | acceptance>0 {} ===",
             if all_pass { "PASS" } else { "FAIL" },
             if any_accept { "PASS" } else { "FAIL" });
    assert!(all_pass, "generate_spec_eagle diverged from generate — greedy spec decode must be exact");
    assert!(any_accept, "acceptance was 0 across all K — the draft forward is wrong (bonus masking)");
    Ok(())
}
