//! Depth-residual profile harness (35B d6257 triage, lane/close35 2026-07-08): primes a
//! D-token cache via prime_cache (batched prefill, fast), sleeps 300ms as a TIMELINE GAP
//! MARKER, then runs N eager decode_step calls (the exact plain-decode board path, env law
//! applies). Run under `nsys profile -t cuda`; post-process the sqlite export by taking all
//! kernels AFTER the widest inter-kernel gap — that window is exactly the N decode steps.
//! Diff d512 vs d6257 kernel-by-kernel: which kernels grow with depth beyond FA, and how much
//! inter-kernel idle (launch/scheduling) the window carries.
//!
//! MEASUREMENT-ONLY: no kernel/dispatch change.
//!
//! usage: depth-profile <model.gguf> [depth=512] [n=32]

use bw24_engine::Engine;
use bw24_engine::hybrid::HybridModel;
use bw24_engine::forward::argmax;
use bw24_gguf::GgufFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: depth-profile <model> [depth] [n]");
    let depth: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(512);
    let n: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(32);

    let e = Engine::new(0)?;
    let g = GgufFile::open(&path)?;
    let model = HybridModel::load(&e, &g)?;

    // Same synthetic prompt family as decode-bench.
    let prompt: Vec<u32> = (0..depth).map(|i| (100 + (i * 7) % 900) as u32).collect();
    let mut cache = bw24_engine::cache::Cache::new(&e, &model.cfg, depth + n + 72)?;
    let t_prime = std::time::Instant::now();
    let mut ll: Vec<f32> = if depth >= bw24_engine::hybrid_forward::PRIME_MIN_T {
        let (l, _h, _hiddens) = model.prime_cache(&e, &prompt, &mut cache)?;
        l
    } else {
        let mut l = Vec::new();
        for &t in &prompt { l = model.decode_step(&e, t, &mut cache)?; }
        l
    };
    // Warmup decode steps BEFORE the marker: JIT/alloc pools AND the laptop-GPU clock ramp —
    // 4 steps measured 50-78 tok/s windows vs the 153-173 steady state; 64 steps ramps fully.
    for _ in 0..64 { let nx = argmax(&ll) as u32; ll = model.decode_step(&e, nx, &mut cache)?; }
    e.stream().synchronize()?;
    println!("primed depth={} (+64 warmup) in {:.2}s", cache.pos, t_prime.elapsed().as_secs_f64());

    // BUSY MARKER (not idle: a sleep gap downclocks the laptop GPU and poisons the window —
    // measured 78 vs 162 tok/s): 64 memsets of a DISTINCTIVE 7777792-byte scratch buffer keep
    // the clocks up; post-process the trace by taking kernels AFTER the last such memset.
    {
        let mut scratch = e.uninit(1_944_448)?;   // 7_777_792 bytes — the marker signature
        for _ in 0..64 {
            let mut v = scratch.slice_mut(0..1_944_448);
            e.memset_zeros_view(&mut v)?;
        }
        e.stream().synchronize()?;
    }

    let t0 = std::time::Instant::now();
    for _ in 0..n { let nx = argmax(&ll) as u32; ll = model.decode_step(&e, nx, &mut cache)?; }
    e.stream().synchronize()?;
    let dt = t0.elapsed().as_secs_f64();
    println!("decode {n} toks @d{}..{}: {:.2} tok/s ({:.1} us/tok)",
             depth + 4, depth + 4 + n, n as f64 / dt, dt * 1e6 / n as f64);
    Ok(())
}
