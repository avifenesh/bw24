//! MTP greedy speculative decode driver + self-consistency gate (research/mtp/MTP-PLAN.md §E).
//!
//! Greedy spec decode is mathematically EXACT: generate_spec(prompt, n, k) MUST produce a
//! token-IDENTICAL sequence to generate(prompt, n). This binary asserts that equality for
//! K in {1,2,4} and prints the acceptance rate (n_accepted / drafted) + tok/s for each K vs
//! plain generate. A wrong MTP head still yields correct output via the bonus token but with
//! ~0 acceptance — so BOTH must hold: identical tokens AND acceptance > 0.
//!
//! Run: BW24_FAST=1 cargo run --release -p bw24-engine --bin run-spec -- <model.gguf> [tok ids...]

use bw24_engine::Engine;
use bw24_engine::hybrid::HybridModel;
use bw24_gguf::GgufFile;

fn first_divergence(a: &[u32], b: &[u32]) -> Option<usize> {
    let n = a.len().min(b.len());
    for i in 0..n { if a[i] != b[i] { return Some(i); } }
    if a.len() != b.len() { return Some(n); }
    None
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: run-spec <model.gguf> [tok ids...]");
    let e = Engine::new(0)?;
    let g = GgufFile::open(&path)?;
    let model = HybridModel::load(&e, &g)?;
    println!("loaded {} ({} layers, nextn={})", g.arch().unwrap_or("?"),
             model.cfg.n_layer, model.cfg.nextn_predict_layers);
    if model.mtp.is_none() {
        eprintln!("ERROR: model has no MTP/NextN head (nextn_predict_layers={}, no blk.N.nextn.eh_proj). \
                   generate_spec is unavailable for this file.", model.cfg.nextn_predict_layers);
        std::process::exit(2);
    }

    let prompt: Vec<u32> = std::env::args().skip(2).filter_map(|s| s.parse().ok()).collect();
    let prompt = if prompt.is_empty() { vec![55u32] } else { prompt };
    println!("prompt tokens: {prompt:?}");

    let n_new = std::env::var("BW24_NGEN").ok().and_then(|s| s.parse().ok()).unwrap_or(32usize);

    // --- ORACLE: plain greedy generate (the exact reference) ---
    e.stream().synchronize()?;
    let t0 = std::time::Instant::now();
    let gold = model.generate(&e, &prompt, n_new)?;
    e.stream().synchronize()?;
    let gen_dt = t0.elapsed().as_secs_f64();
    let gen_tps = n_new as f64 / gen_dt;
    println!("\n[generate]   {n_new} tok in {gen_dt:.3}s = {gen_tps:.2} tok/s");
    println!("  tokens: {gold:?}");

    let mut all_pass = true;
    for &k in &[1usize, 2, 3, 4, 6, 8] {
        e.stream().synchronize()?;
        let t1 = std::time::Instant::now();
        let (spec, drafted, accepted) = model.generate_spec(&e, &prompt, n_new, k)?;
        e.stream().synchronize()?;
        let spec_dt = t1.elapsed().as_secs_f64();
        let spec_tps = n_new as f64 / spec_dt;
        let acc_rate = if drafted > 0 { accepted as f64 / drafted as f64 } else { 0.0 };

        let pass = first_divergence(&gold, &spec).is_none();
        all_pass &= pass;
        println!("\n[generate_spec K={k}] {n_new} tok in {spec_dt:.3}s = {spec_tps:.2} tok/s \
                  ({:.2}x vs generate)", spec_tps / gen_tps);
        println!("  acceptance: {accepted}/{drafted} = {:.1}%   self-consistency: {}",
                 acc_rate * 100.0, if pass { "PASS (identical to generate)" } else { "FAIL" });
        if !pass {
            let idx = first_divergence(&gold, &spec).unwrap();
            println!("  FIRST DIVERGENCE at index {idx}:");
            println!("    generate     [..]: {:?}", &gold[idx.saturating_sub(2)..(idx + 3).min(gold.len())]);
            println!("    generate_spec[..]: {:?}", &spec[idx.saturating_sub(2)..(idx + 3).min(spec.len())]);
            println!("    spec tokens: {spec:?}");
        }
        // acceptance>0 is the SECOND gate (a wrong head passes self-consistency via the bonus token
        // but accepts nothing). Report it; the assert below covers the exactness gate.
        if pass && accepted == 0 {
            println!("  WARNING: acceptance == 0 with identical output — MTP head is likely \
                      forwarded wrong (bonus-token masking). Speedup will be absent.");
        }
    }

    println!("\n=== SELF-CONSISTENCY {} ===", if all_pass { "PASS" } else { "FAIL" });
    assert!(all_pass, "generate_spec diverged from generate — greedy spec decode must be exact");
    Ok(())
}
