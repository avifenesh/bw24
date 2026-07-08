//! Decode-window profiler harness (35B depth-residual triage, lane/close35 2026-07-08): primes a
//! P-token prompt UNTIMED, then brackets exactly N eager decode_step calls between
//! cuProfilerStart/cuProfilerStop so `nsys --capture-range=cudaProfilerApi` records a CLEAN
//! decode-only kernel trace (no prime kernels polluting the sums). Kernel-by-kernel diff of the
//! d512 vs d6257 traces = which kernels GROW with depth beyond FA; the trace's inter-kernel gaps
//! = the scheduling/launch-latency component.
//!
//! MEASUREMENT-ONLY: no kernel/dispatch change; profiler markers around the existing eager path
//! (decode-bench eager protocol: greedy argmax feed, host logits — the plain-board code path).
//!
//! usage: decode-window-profile <model.gguf> [depth=512] [n=32]
//!   nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
//!     -o /tmp/... ./target/release/decode-window-profile <model> 6257 32
//! env: fast-path core is default-on; nothing to set (MoE cache included).

use bw24_engine::Engine;
use bw24_engine::hybrid::HybridModel;
use bw24_engine::forward::argmax;
use bw24_gguf::GgufFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: decode-window-profile <model> [depth] [n]");
    let depth: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(512);
    let n: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(32);

    let e = Engine::new(0)?;
    let g = GgufFile::open(&path)?;
    let m = HybridModel::load(&e, &g)?;

    // decode-bench synthetic prompt + eager protocol (the plain-board code path).
    let prompt: Vec<u32> = (0..depth).map(|i| (100 + (i * 7) % 900) as u32).collect();
    let mut cache = bw24_engine::cache::Cache::new(&e, &m.cfg, depth + n + 8)?;
    let mut ll = Vec::new();
    for &t in &prompt { ll = m.decode_step(&e, t, &mut cache)?; }   // PRIME (uncaptured)
    // warmup decode steps OUTSIDE the capture window (L2/allocator steady state)
    for _ in 0..4 { let nx = argmax(&ll) as u32; ll = m.decode_step(&e, nx, &mut cache)?; }
    e.stream().synchronize()?;

    unsafe { cudarc::driver::sys::cuProfilerStart().result()?; }
    let t0 = std::time::Instant::now();
    for _ in 0..n { let nx = argmax(&ll) as u32; ll = m.decode_step(&e, nx, &mut cache)?; }
    e.stream().synchronize()?;
    let dt = t0.elapsed().as_secs_f64();
    unsafe { cudarc::driver::sys::cuProfilerStop().result()?; }

    println!("decode window @d{depth}: {n} tokens in {dt:.3}s = {:.1} tok/s ({:.1} us/tok)",
             n as f64 / dt, dt * 1e6 / n as f64);
    Ok(())
}
