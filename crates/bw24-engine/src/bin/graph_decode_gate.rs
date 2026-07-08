//! CUDA-GRAPH-PLAN Phase 3 gate: the CAPTURE/REPLAY decode path (`generate_graph`) must produce a
//! BIT-IDENTICAL greedy token stream to the eager `decode_step` over N steps, across >= 2 t_kv
//! buckets (prime P, then generate N crossing the n_splits=ceil(t_kv/64) split boundaries). Any token
//! mismatch = FAIL.
//!
//! `generate_graph` primes the prompt eagerly (device-counter `decode_step_dc`), then replays a
//! captured CUDA graph per gen step. The graph is captured once per t_kv bucket (key = eager
//! (fa_vec, n_splits)); the captured fa_decode_dc reads the ACTUAL t_kv from a device counter and
//! sizes n_splits from the bucket so replay reproduces eager's split geometry bit-for-bit. The KV
//! write slot + rope pos + token id are all device-resident; only a [1] u32 token is read back per
//! step. NO host sync inside the captured region.
//!
//! usage: graph-decode-gate <model> [P] [N] [bench]   (fast-path core is default-on)
use bw24_engine::Engine;
use bw24_engine::hybrid::HybridModel;
use bw24_engine::decode::GraphDecodeState;
use bw24_engine::forward::argmax;
use bw24_gguf::GgufFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: graph-decode-gate <model> [P] [N] [bench]");
    let p: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(64);
    let n: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(256);
    let e = Engine::new(0)?;
    let g = GgufFile::open(&path)?;
    let m = HybridModel::load(&e, &g)?;

    let prompt: Vec<u32> = (0..p).map(|i| (100 + (i * 7) % 900) as u32).collect();

    // ---- EAGER reference: prime + generate N greedy tokens via decode_step ----
    let mut cache_e = bw24_engine::cache::Cache::new(&e, &m.cfg, p + n + 8)?;
    let mut ll = Vec::new();
    for &t in &prompt { ll = m.decode_step(&e, t, &mut cache_e)?; }
    let mut eager_in = argmax(&ll) as u32;
    let mut eager_tokens = Vec::with_capacity(n);
    let mut buckets_seen: Vec<(bool, usize)> = Vec::new();
    let head_dim = m.cfg.head_dim_k as usize;
    for _ in 0..n {
        let t_kv = cache_e.kv.iter().filter_map(|k| k.as_ref()).map(|k| k.len + 1).next().unwrap_or(0);
        let key = e.fa_bucket_key(t_kv, head_dim, m.cfg.n_head_kv as usize);
        if !buckets_seen.contains(&key) { buckets_seen.push(key); }
        ll = m.decode_step(&e, eager_in, &mut cache_e)?;
        let nx = argmax(&ll) as u32;
        eager_tokens.push(nx);
        eager_in = nx;
    }

    // ---- GRAPH path: prime + generate N via capture/replay ----
    let mut gs = GraphDecodeState::new(&e)?;
    let graph_tokens = m.generate_graph(&e, &mut gs, &prompt, n)?;

    // ---- compare token-by-token ----
    let mut mismatches = 0usize;
    let mut first_mm: Option<(usize, u32, u32)> = None;
    for (step, (&a, &b)) in eager_tokens.iter().zip(graph_tokens.iter()).enumerate() {
        if a != b {
            mismatches += 1;
            if first_mm.is_none() { first_mm = Some((step, a, b)); }
            if mismatches <= 5 { println!("MISMATCH step {step}: eager={a} graph={b}"); }
        }
    }
    if graph_tokens.len() != eager_tokens.len() {
        println!("LENGTH MISMATCH: eager={} graph={}", eager_tokens.len(), graph_tokens.len());
        mismatches += 1;
    }

    println!("buckets crossed (fa_vec, n_splits): {:?}", buckets_seen);
    println!("graph (re)captures: {}", gs.captures);

    // ---- perf: eager decode_step vs graph replay (prime OUTSIDE the timer) ----
    if std::env::args().nth(4).as_deref() == Some("bench") {
        let bn = 256usize;
        // EAGER timed
        let mut cache_eb = bw24_engine::cache::Cache::new(&e, &m.cfg, p + bn + 8)?;
        let mut llb = Vec::new();
        for &t in &prompt { llb = m.decode_step(&e, t, &mut cache_eb)?; }
        let mut ein = argmax(&llb) as u32;
        e.stream().synchronize()?;
        let t0 = std::time::Instant::now();
        for _ in 0..bn { llb = m.decode_step(&e, ein, &mut cache_eb)?; ein = argmax(&llb) as u32; }
        e.stream().synchronize()?;
        let dt_e = t0.elapsed().as_secs_f64();

        // GRAPH timed (HONEST prime/gen split): measure prime EXACTLY via generate_graph(prompt, 0)
        // (it primes then runs 0 gen steps, so dt_prime is the pure eager dc-prime), then measure
        // generate_graph(prompt, bn) end-to-end; gen tok/s = bn / (total - prime). The gen window
        // includes the real re-capture cost (one per t_kv bucket, ~every 64 tokens) — not subtracted,
        // so this is the steady-state-with-amortized-recapture number, reported honestly.
        let mut gsp = GraphDecodeState::new(&e)?;
        e.stream().synchronize()?;
        let tp = std::time::Instant::now();
        let _ = m.generate_graph(&e, &mut gsp, &prompt, 0)?;   // prime only
        e.stream().synchronize()?;
        let dt_prime = tp.elapsed().as_secs_f64();

        let mut gsb = GraphDecodeState::new(&e)?;
        e.stream().synchronize()?;
        let t1 = std::time::Instant::now();
        let gt = m.generate_graph(&e, &mut gsb, &prompt, bn)?;
        e.stream().synchronize()?;
        let dt_g_total = t1.elapsed().as_secs_f64();
        let dt_g = (dt_g_total - dt_prime).max(1e-9);
        let _ = gt.len();
        println!("decode tok/s  eager={:.1}  graph={:.1}  (tg{bn} @ctx{p}, ms/tok eager={:.2} graph={:.2}; graph_total={:.1}ms prime={:.1}ms recaptures={})",
                 bn as f64 / dt_e, bn as f64 / dt_g, dt_e * 1000.0 / bn as f64, dt_g * 1000.0 / bn as f64,
                 dt_g_total * 1000.0, dt_prime * 1000.0, gsb.captures);
    }

    if mismatches == 0 {
        println!("Phase-3 gate PASS: {n} steps generate_graph == decode_step (BIT-IDENTICAL), \
                  buckets={} captures={}", buckets_seen.len(), gs.captures);
    } else {
        let (s, a, b) = first_mm.unwrap_or((0, 0, 0));
        println!("Phase-3 gate FAIL: {mismatches}/{n} mismatches (first @ step {s}: eager={a} graph={b})");
        std::process::exit(1);
    }
    Ok(())
}
