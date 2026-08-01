//! decode-batch-bench: aggregate decode throughput vs batch size (ARCHITECTURE-H100.md B2').
//!
//! The number that funds the multi-tenant thesis: decode is weight-stream-bound, so
//! aggregate tok/s should scale near-linearly with B until attention/launch overheads
//! bite. Reports per-B aggregate and per-seq rates over N timed reps of `steps` batched
//! decode steps (greedy), medians over reps. Run AFTER decode-batch-gate is green.
//!
//! Usage: decode-batch-bench <model.gguf> [--steps 128] [--reps 5] [--batches 1,2,4,8]

use memra_engine::cache::Cache;
use memra_engine::forward::argmax;
use memra_engine::hybrid::HybridModel;
use memra_engine::Engine;
use memra_gguf::GgufFile;

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

    // inc3 (3a) CHUNK-SIZE SWEEP: `--seqs N --chunk C` advances N sequences per tick via
    // ceil(N/C) chunked decode_step_batch calls (the worker's group_chunks shape) and prints
    // one aggregate tok/s line — the per-tick cost of chunking policy C for an N-seq batch.
    // One chunk config per invocation (env-dependent dispatch reads once); interleave
    // invocations at the script level for the x5 medians.
    let seqs: Option<usize> = rest.iter().position(|a| a == "--seqs")
        .and_then(|i| rest.get(i + 1)).and_then(|v| v.parse().ok());
    let chunk: usize = rest.iter().position(|a| a == "--chunk")
        .and_then(|i| rest.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(8);

    let e = Engine::new(0)?;
    let g = GgufFile::open(&path)?;
    let model = HybridModel::load_without_mtp(&e, &g)?;
    println!("loaded {} ({} layers); steps={steps} reps={reps} batches={batches:?}",
             g.arch().unwrap_or("?"), model.layers.len());

    if let Some(n_seqs) = seqs {
        let prompt_t: usize = rest.iter().position(|a| a == "--ctx")
            .and_then(|i| rest.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(512);
        let ctx = prompt_t + n_seqs * 7 + 64 + (steps + 8) * (reps + 1);
        let mut caches: Vec<Cache> = Vec::new();
        let mut toks: Vec<u32> = Vec::new();
        for i in 0..n_seqs {
            let prompt: Vec<u32> = (0..(prompt_t as u32) + i as u32 * 7)
                .map(|j| 55 + i as u32 * 97 + j * 31).collect();
            let mut c = Cache::new(&e, &model.cfg, ctx)?;
            let _ = model.prime_cache(&e, &prompt, &mut c)?;
            toks.push(*prompt.last().unwrap());
            caches.push(c);
        }
        let mut rates: Vec<f64> = Vec::new();
        for rep in 0..=reps {
            let t0 = std::time::Instant::now();
            for _ in 0..steps {
                let mut next: Vec<u32> = Vec::with_capacity(n_seqs);
                for (cs, ts) in caches.chunks_mut(chunk).zip(toks.chunks(chunk)) {
                    let mut refs: Vec<&mut Cache> = cs.iter_mut().collect();
                    let logits = model.decode_step_batch(&e, ts, &mut refs)?;
                    for l in &logits {
                        next.push(argmax(l) as u32);
                    }
                }
                toks = next;
            }
            let dt = t0.elapsed().as_secs_f64();
            if rep > 0 {
                rates.push((n_seqs * steps) as f64 / dt);
            }
        }
        rates.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let agg = rates[rates.len() / 2];
        println!("CHUNKSWEEP seqs={n_seqs} chunk={chunk}: aggregate {agg:.1} tok/s \
                  ({:.2} ms/tick, median of {reps})",
                 n_seqs as f64 / agg * 1e3);
        if memra_engine::decode_batch::batch_phase_on() {
            println!("{}", memra_engine::decode_batch::batch_phase_report());
        }
        return Ok(());
    }

    let ctx_extra: usize = rest.iter().position(|a| a == "--ctx")
        .and_then(|i| rest.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(512);
    let ctx = ctx_extra + 8 * 7 + 64 + (steps + 8) * (reps + 1);
    let mut results: Vec<(usize, f64, f64)> = Vec::new();

    for &b_n in &batches {
        // Fresh caches per batch size; prompts >= 16 tokens (PRIME_MIN_T), distinct.
        let mut caches: Vec<Cache> = Vec::new();
        let mut toks: Vec<u32> = Vec::new();
        // Prompts sized to the SERVING regime (default 512 tokens; --ctx overrides): short
        // prompts under fa_vec_min_tkv silently fall to the f32 attention path and skew the
        // step profile (2026-07-26 nsys finding: fa_decode_f32 at 21% on 24-tok prompts).
        let prompt_t: usize = rest.iter().position(|a| a == "--ctx")
            .and_then(|i| rest.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(512);
        for i in 0..b_n {
            let prompt: Vec<u32> = (0..(prompt_t as u32) + i as u32 * 7)
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

    // MEMRA_BATCH_PHASE=1: engine tick decomposition (sync-bounded — shares rank, not walltime).
    if memra_engine::decode_batch::batch_phase_on() {
        println!("{}", memra_engine::decode_batch::batch_phase_report());
    }

    // Host sample/emit cost at the serving vocab (the worker tick's per-seq host stage): time
    // greedy argmax vs the load harness's temp-0.7 sample on REAL last-step logits, per row.
    {
        use memra_engine::sampler::{Sampler, SamplerConfig};
        let b_n = *batches.last().unwrap();
        let mut caches: Vec<Cache> = Vec::new();
        let mut toks: Vec<u32> = Vec::new();
        for i in 0..b_n {
            let prompt: Vec<u32> = (0..512u32).map(|j| 55 + i as u32 * 97 + j * 31).collect();
            let mut c = Cache::new(&e, &model.cfg, 640)?;
            let _ = model.prime_cache(&e, &prompt, &mut c)?;
            toks.push(*prompt.last().unwrap());
            caches.push(c);
        }
        let mut cache_refs: Vec<&mut Cache> = caches.iter_mut().collect();
        let rows = model.decode_step_batch(&e, &toks, &mut cache_refs)?;
        let reps = 50usize;
        let t0 = std::time::Instant::now();
        let mut sink = 0u32;
        for _ in 0..reps {
            for r in &rows {
                sink = sink.wrapping_add(argmax(r) as u32);
            }
        }
        let greedy_us = t0.elapsed().as_secs_f64() * 1e6 / (reps * rows.len()) as f64;
        let mut smp = Sampler::new(SamplerConfig { temperature: 0.7, seed: 7, ..Default::default() });
        let t1 = std::time::Instant::now();
        for _ in 0..reps {
            for r in &rows {
                sink = sink.wrapping_add(smp.sample(r));
            }
        }
        let temp_us = t1.elapsed().as_secs_f64() * 1e6 / (reps * rows.len()) as f64;
        println!("[host-sample] n_vocab={} greedy argmax {greedy_us:.0} us/row | temp0.7 sample \
                  {temp_us:.0} us/row | x B={b_n} rows/tick = {:.2} ms (greedy) / {:.2} ms (temp) [sink {sink}]",
                 rows[0].len(), greedy_us * b_n as f64 / 1e3, temp_us * b_n as f64 / 1e3);
    }
    Ok(())
}
