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
    let n_embd = m.cfg.n_embd as usize;
    // upload the embed table once for the device-gather check
    let embd_gpu = e.upload_u8(&m.embd.raw)?;
    let (qt, row_bytes) = m.embd.qt_and_row_bytes(n_embd);
    let mut argmax_mm = 0;
    let mut embed_mm = 0;
    for step in 0..n {
        let host_tok = argmax(&ll) as u32;
        // (1) device argmax
        let logits_d = e.htod(&ll)?;
        let tok_d = e.argmax_token_device(&logits_d, n_vocab)?;
        let dev_tok = e.dtoh_u32_one(&tok_d)?;
        if dev_tok != host_tok { argmax_mm += 1; if argmax_mm <= 3 {
            println!("ARGMAX MISMATCH step {step}: host={host_tok} dev={dev_tok}"); } }
        // (2) device embed-gather of the (device) token vs host gather — bit-exact
        let host_emb = m.embd.gather(n_embd, &[host_tok]);
        let dev_emb_d = e.embed_gather_device(&embd_gpu, &tok_d, n_embd, qt, row_bytes)?;
        let dev_emb = e.dtoh(&dev_emb_d)?;
        let maxd = host_emb.iter().zip(&dev_emb).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
        if maxd != 0.0 { embed_mm += 1; if embed_mm <= 3 {
            println!("EMBED MISMATCH step {step}: maxdiff={maxd:.3e}"); } }
        ll = m.decode_step(&e, host_tok, &mut cache)?;
    }
    let ok = argmax_mm == 0 && embed_mm == 0;
    if ok { println!("Phase-1 gate PASS: {n} steps device argmax==host AND device embed==host (n_vocab={n_vocab}, n_embd={n_embd}, embed_qt={qt})"); }
    else { println!("Phase-1 gate FAIL: argmax_mm={argmax_mm} embed_mm={embed_mm}/{n}"); std::process::exit(1); }
    Ok(())
}
