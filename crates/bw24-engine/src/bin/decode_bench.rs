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
    // Directory path = safetensors HF checkpoint or manifest-backed bw24 repack, matching run-gen.
    let m = if std::path::Path::new(&path).is_dir() {
        let dir = std::path::Path::new(&path);
        if dir.join("manifest.json").exists() {
            let repack = bw24_gguf::source::Hy3RepackSource::open(dir)?;
            HybridModel::load_from_source(&e, &repack)?
        } else {
            let st = bw24_gguf::source::SafetensorsSource::open(dir)?;
            HybridModel::load_from_source(&e, &st)?
        }
    } else {
        let g = GgufFile::open(&path)?;
        HybridModel::load(&e, &g)?
    };
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
        "g4graph" => {
            // EAGER reference first (tokens + time), then the gemma4 plain-graph loop on a
            // fresh cache; token-sequence identity is the gate, speedup is the number.
            let mut cache_e = bw24_engine::cache::Cache::new(&e, &m.cfg, p + n + 8)?;
            let mut ll = Vec::new();
            for &t in &prompt { ll = m.decode_step(&e, t, &mut cache_e)?; }
            e.stream().synchronize()?;
            let t0 = std::time::Instant::now();
            let mut toks_e = Vec::with_capacity(n);
            for _ in 0..n {
                let nx = argmax(&ll) as u32;
                toks_e.push(nx);
                ll = m.decode_step(&e, nx, &mut cache_e)?;
            }
            e.stream().synchronize()?;
            let dt_e = t0.elapsed().as_secs_f64();

            let mut cache_g = bw24_engine::cache::Cache::new(&e, &m.cfg, p + n + 8)?;
            let mut ll2 = Vec::new();
            for &t in &prompt { ll2 = m.decode_step(&e, t, &mut cache_g)?; }
            let first = argmax(&ll2) as u32;
            e.stream().synchronize()?;
            let t1 = std::time::Instant::now();
            let toks_g = m.gemma4_generate_plain_graph(&e, &mut cache_g, first, n, &[])?;
            e.stream().synchronize()?;
            let dt_g = t1.elapsed().as_secs_f64();
            // eager emits [argmax0, next...]; graph feeds argmax0 and emits its successors —
            // align: graph token i corresponds to eager token i+1... both sequences start
            // from the same primed state: eager toks_e[0] = argmax(prime) = `first`;
            // graph out[0] = argmax after decoding `first` = toks_e[1]. Compare shifted.
            let n_cmp = toks_g.len().min(toks_e.len().saturating_sub(1));
            let mism = (0..n_cmp).filter(|&i| toks_g[i] != toks_e[i + 1]).count();
            println!("decode tg{n} @ctx{p}: EAGER {:.1} tok/s | G4GRAPH {:.1} tok/s ({:.2} ms/tok) | speedup {:.3}x | token-mismatches {mism}/{n_cmp} {}",
                     n as f64/dt_e, toks_g.len() as f64/dt_g, dt_g*1000.0/toks_g.len().max(1) as f64,
                     dt_e / dt_g * (toks_g.len() as f64 / n as f64),
                     if mism == 0 { "MATCH" } else { "MISMATCH" });
        }
        other => return Err(format!("unknown mode {other:?} (use eager|graph|both|g4graph)").into()),
    }
    Ok(())
}
