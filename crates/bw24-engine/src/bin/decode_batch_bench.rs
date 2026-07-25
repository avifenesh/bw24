//! decode-batch-bench: aggregate decode throughput vs batch size (ARCHITECTURE-H100.md B2').
//!
//! The number that funds the multi-tenant thesis: decode is weight-stream-bound, so
//! aggregate tok/s should scale near-linearly with B until attention/launch overheads
//! bite. Reports per-B aggregate and per-seq rates over N timed reps of `steps` batched
//! decode steps (greedy), medians over reps. Run AFTER decode-batch-gate is green.
//!
//! Usage: decode-batch-bench <model.gguf> [--steps 128] [--reps 5] [--batches 1,2,4,8]

use bw24_engine::cache::Cache;
use bw24_engine::forward::argmax;
use bw24_engine::hybrid::HybridModel;
use bw24_engine::Engine;
use bw24_gguf::GgufFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: decode-batch-bench <model.gguf> [--steps N] [--reps R] [--batches 1,2,4,8]");
    let rest: Vec<String> = args.collect();
    let steps: usize = rest.iter().position(|a| a == "--steps")
        .and_then(|i| rest.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(128);
    let reps: usize = rest.iter().position(|a| a == "--reps")
        .and_then(|i| rest.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(5);
    let batches: Vec<usize> = rest.iter().position(|a| a == "--batches")
        .and_then(|i| rest.get(i + 1))
        .map(|v| v.split(',').filter_map(|s| s.parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 2, 4, 8]);

    let e = Engine::new(0)?;
    let g = GgufFile::open(&path)?;
    let model = HybridModel::load_without_mtp(&e, &g)?;
    println!("loaded {} ({} layers); steps={steps} reps={reps} batches={batches:?}",
             g.arch().unwrap_or("?"), model.layers.len());

    let ctx = 64 + (steps + 8) * (reps + 1);
    let mut results: Vec<(usize, f64, f64)> = Vec::new();

    for &b_n in &batches {
        // Fresh caches per batch size; prompts >= 16 tokens (PRIME_MIN_T), distinct.
        let mut caches: Vec<Cache> = Vec::new();
        let mut toks: Vec<u32> = Vec::new();
        for i in 0..b_n {
            let prompt: Vec<u32> = (0..24 + i as u32 * 7)
                .map(|j| 55 + i as u32 * 97 + j * 31).collect();
            let mut c = Cache::new(&e, &model.cfg, ctx)?;
            let _ = model.prime_cache(&e, &prompt, &mut c)?;
            toks.push(*prompt.last().unwrap());
            caches.push(c);
        }
        // Warmup rep + timed reps.
        let mut rates: Vec<f64> = Vec::new();
        for rep in 0..=reps {
            let t0 = std::time::Instant::now();
            for _ in 0..steps {
                let mut cache_refs: Vec<&mut Cache> = caches.iter_mut().collect();
                let logits = model.decode_step_batch(&e, &toks, &mut cache_refs)?;
                for (bi, l) in logits.iter().enumerate() {
                    toks[bi] = argmax(l) as u32;
                }
            }
            let dt = t0.elapsed().as_secs_f64();
            if rep > 0 {
                rates.push((b_n * steps) as f64 / dt);
            }
        }
        rates.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let agg = rates[rates.len() / 2];
        let per = agg / b_n as f64;
        println!("B={b_n}: aggregate {agg:.1} tok/s, per-seq {per:.1} tok/s (median of {reps})");
        results.push((b_n, agg, per));
    }

    // Scaling summary vs B=1.
    if let Some(&(_, base, _)) = results.first() {
        for &(b_n, agg, _) in &results {
            println!("scale B={b_n}: {:.2}x aggregate vs B=1", agg / base);
        }
    }
    Ok(())
}
