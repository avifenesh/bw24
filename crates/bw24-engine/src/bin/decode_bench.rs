//! Clean decode tok/s: prime a P-token prompt OUTSIDE the timer (like llama tg = prefill separate),
//! then time ONLY the N greedy gen steps. Matches llama-bench tg128 protocol.
//!
//! usage: decode-bench <model> [P] [N] [mode]
//!   mode (4th arg): "eager" (default) times decode_step; "graph" times generate_graph (CUDA-graph
//!   capture/replay — RANK3 LEVER 1); "both" times eager THEN graph back-to-back and prints the
//!   speedup. The graph path primes via the device-counter decode and replays a captured graph per
//!   t_kv bucket; prime is measured separately (generate_graph(prompt, 0)) and subtracted so the
//!   reported graph tok/s is gen-only, same protocol as graph_decode_gate's bench.
use bw24_engine::Engine;
use bw24_engine::hybrid::HybridModel;
use bw24_engine::decode::GraphDecodeState;
use bw24_engine::forward::argmax;
use bw24_gguf::GgufFile;

fn bench_eager(e: &Engine, m: &HybridModel, prompt: &[u32], n: usize)
               -> Result<f64, Box<dyn std::error::Error>> {
    let mut cache = bw24_engine::cache::Cache::new(e, &m.cfg, prompt.len() + n + 8)?;
    let mut ll = Vec::new();
    for &t in prompt { ll = m.decode_step(e, t, &mut cache)?; }  // PRIME (untimed)
    e.stream().synchronize()?;
    let t0 = std::time::Instant::now();
    for _ in 0..n { let nx = argmax(&ll) as u32; ll = m.decode_step(e, nx, &mut cache)?; }
    e.stream().synchronize()?;
    Ok(t0.elapsed().as_secs_f64())
}

/// Graph gen-only time = total(generate_graph(prompt, n)) - prime(generate_graph(prompt, 0)).
/// The gen window includes the real re-capture cost (one per t_kv bucket, ~every 64 tokens), NOT
/// subtracted — honest steady-state-with-amortized-recapture number. Returns (gen_secs, captures).
fn bench_graph(e: &Engine, m: &HybridModel, prompt: &[u32], n: usize)
               -> Result<(f64, usize), Box<dyn std::error::Error>> {
    // measure pure prime
    let mut gsp = GraphDecodeState::new(e)?;
    e.stream().synchronize()?;
    let tp = std::time::Instant::now();
    let _ = m.generate_graph(e, &mut gsp, prompt, 0)?;
    e.stream().synchronize()?;
    let dt_prime = tp.elapsed().as_secs_f64();
    // measure prime + gen
    let mut gsb = GraphDecodeState::new(e)?;
    e.stream().synchronize()?;
    let t1 = std::time::Instant::now();
    let _ = m.generate_graph(e, &mut gsb, prompt, n)?;
    e.stream().synchronize()?;
    let dt_total = t1.elapsed().as_secs_f64();
    Ok(((dt_total - dt_prime).max(1e-9), gsb.captures))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: decode-bench <model> [P] [N] [eager|graph|both]");
    let p: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(512);
    let n: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(128);
    let mode = std::env::args().nth(4).unwrap_or_else(|| "eager".to_string());
    let e = Engine::new(0)?;
    let g = GgufFile::open(&path)?;
    let m = HybridModel::load(&e, &g)?;
    let prompt: Vec<u32> = (0..p).map(|i| (100 + (i*7)%900) as u32).collect();

    match mode.as_str() {
        "eager" => {
            let dt = bench_eager(&e, &m, &prompt, n)?;
            println!("decode tg{n} @ctx{p}: EAGER {:.1} tok/s ({:.2} ms/tok)", n as f64/dt, dt*1000.0/n as f64);
        }
        "graph" => {
            let (dt, caps) = bench_graph(&e, &m, &prompt, n)?;
            println!("decode tg{n} @ctx{p}: GRAPH {:.1} tok/s ({:.2} ms/tok) [recaptures={caps}]",
                     n as f64/dt, dt*1000.0/n as f64);
        }
        "both" => {
            let dt_e = bench_eager(&e, &m, &prompt, n)?;
            let (dt_g, caps) = bench_graph(&e, &m, &prompt, n)?;
            println!("decode tg{n} @ctx{p}: EAGER {:.1} tok/s ({:.2} ms/tok) | GRAPH {:.1} tok/s ({:.2} ms/tok) [recaptures={caps}] | speedup {:.3}x",
                     n as f64/dt_e, dt_e*1000.0/n as f64,
                     n as f64/dt_g, dt_g*1000.0/n as f64,
                     dt_e/dt_g);
        }
        other => return Err(format!("unknown mode {other:?} (use eager|graph|both)").into()),
    }
    Ok(())
}
