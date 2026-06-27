//! CUDA-GRAPH-PLAN Phase 1 gate: device argmax_logits_f32_to_u32 must be BIT-IDENTICAL to the
//! host argmax (smallest-index tie-break) over real decode logits. Runs N greedy steps, compares
//! the device token id to the host argmax token id every step. Any mismatch = FAIL.
use bw24_engine::Engine;
use bw24_engine::hybrid::HybridModel;
use bw24_engine::forward::argmax;
use bw24_gguf::GgufFile;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: argmax-gate <model> [P] [N]");
    let p: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(64);
    let n: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(256);
    let e = Engine::new(0)?;
    let g = GgufFile::open(&path)?;
    let m = HybridModel::load(&e, &g)?;
    let prompt: Vec<u32> = (0..p).map(|i| (100 + (i * 7) % 900) as u32).collect();
    let mut cache = bw24_engine::cache::Cache::new(&e, &m.cfg, p + n + 8)?;
    let mut ll = Vec::new();
    for &t in &prompt { ll = m.decode_step(&e, t, &mut cache)?; }
    let n_vocab = ll.len();
    let mut mismatches = 0;
    for step in 0..n {
        let host_tok = argmax(&ll) as u32;
        // upload the same logits, run the device kernel, read back the [1] u32
        let logits_d = e.htod(&ll)?;
        let tok_d = e.argmax_token_device(&logits_d, n_vocab)?;
        let dev_tok = e.dtoh_u32_one(&tok_d)?;
        if dev_tok != host_tok {
            mismatches += 1;
            if mismatches <= 5 {
                println!("MISMATCH step {step}: host={host_tok} dev={dev_tok} \
                          host_logit={:.6} dev_logit={:.6}", ll[host_tok as usize], ll[dev_tok as usize]);
            }
        }
        ll = m.decode_step(&e, host_tok, &mut cache)?;
    }
    if mismatches == 0 { println!("argmax gate PASS: {n} steps device==host (n_vocab={n_vocab})"); }
    else { println!("argmax gate FAIL: {mismatches}/{n} mismatches"); std::process::exit(1); }
    Ok(())
}
