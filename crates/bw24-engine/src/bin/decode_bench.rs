//! Clean decode tok/s: prime a P-token prompt OUTSIDE the timer (like llama tg = prefill separate),
//! then time ONLY the N greedy gen steps. Matches llama-bench tg128 protocol.
use bw24_engine::Engine;
use bw24_engine::hybrid::HybridModel;
use bw24_engine::forward::argmax;
use bw24_gguf::GgufFile;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: decode-bench <model> <P> <N>");
    let p: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(512);
    let n: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(128);
    let e = Engine::new(0)?;
    let g = GgufFile::open(&path)?;
    let m = HybridModel::load(&e, &g)?;
    let prompt: Vec<u32> = (0..p).map(|i| (100 + (i*7)%900) as u32).collect();
    let mut cache = bw24_engine::cache::Cache::new(&e, &m.cfg, p + n + 8)?;
    let mut ll = Vec::new();
    for &t in &prompt { ll = m.decode_step(&e, t, &mut cache)?; }  // PRIME (untimed)
    e.stream().synchronize()?;
    let t0 = std::time::Instant::now();
    for _ in 0..n { let nx = argmax(&ll) as u32; ll = m.decode_step(&e, nx, &mut cache)?; }
    e.stream().synchronize()?;
    let dt = t0.elapsed().as_secs_f64();
    println!("decode tg{n} @ctx{p}: {:.1} tok/s ({:.2} ms/tok)", n as f64/dt, dt*1000.0/n as f64);
    Ok(())
}
