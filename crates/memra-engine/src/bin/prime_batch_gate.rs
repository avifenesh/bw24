//! prime-batch-gate (task #13): cross-request batched prime vs individual primes.
//!
//! The concat GEMM (m = sum_T) is a NEW NUMERIC CONFIG vs per-seq primes (different K
//! tiling) — the gate is therefore argmax/stream equality per sequence, not bit-identity:
//!   1. prefill argmax per seq: batched == individual (hard FAIL on mismatch)
//!   2. 16 greedy decode steps from each primed cache: streams must MATCH per seq
//!      (decode itself is untouched; drift here would mean the batched prime left a
//!      different cache/recurrent state beyond numeric-config tolerance).
//!
//! usage: prime-batch-gate <model.gguf> [--batch 3]

use memra_engine::cache::Cache;
use memra_engine::forward::argmax;
use memra_engine::hybrid::HybridModel;
use memra_engine::Engine;
use memra_gguf::GgufFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: prime-batch-gate <model.gguf> [--batch N]");
    let rest: Vec<String> = args.collect();
    let b: usize = rest.iter().position(|a| a == "--batch")
        .and_then(|i| rest.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(3);

    let e = Engine::new(0)?;
    let g = GgufFile::open(&path)?;
    let model = HybridModel::load_without_mtp(&e, &g)?;
    println!("loaded {} ({} layers); batch={b}", g.arch().unwrap_or("?"), model.layers.len());

    // deliberately UNEVEN prompt lengths (24, 41, 58, ...) so offsets/tails are exercised
    let prompts: Vec<Vec<u32>> = (0..b)
        .map(|i| (0..24 + i as u32 * 17).map(|j| 55 + i as u32 * 97 + j * 31).collect())
        .collect();
    let ctx = 512;
    let steps = 16usize;
    let mut fails = 0usize;

    // individual reference primes + decode streams
    let mut ref_streams: Vec<Vec<u32>> = Vec::with_capacity(b);
    let mut ref_argmax: Vec<u32> = Vec::with_capacity(b);
    for p in &prompts {
        let mut c = Cache::new(&e, &model.cfg, ctx)?;
        let (logits, _, _) = model.prime_cache(&e, p, &mut c, 0)?;
        let mut t = argmax(&logits) as u32;
        ref_argmax.push(t);
        let mut stream = Vec::with_capacity(steps);
        for _ in 0..steps {
            let (l, _) = model.decode_step_h(&e, t, &mut c)?;
            t = argmax(&l) as u32;
            stream.push(t);
        }
        ref_streams.push(stream);
    }

    // batched prime + decode streams
    let mut caches: Vec<Cache> = (0..b).map(|_| Cache::new(&e, &model.cfg, ctx)).collect::<Result<_, _>>()?;
    {
        let prompt_refs: Vec<&[u32]> = prompts.iter().map(|p| p.as_slice()).collect();
        let mut cache_refs: Vec<&mut Cache> = caches.iter_mut().collect();
        let outs = model.prime_cache_batch(&e, &prompt_refs, &mut cache_refs)?;
        for (s, (logits, _, _)) in outs.iter().enumerate() {
            let a = argmax(logits) as u32;
            let ok = a == ref_argmax[s];
            println!("seq {s} (T={}): prefill argmax batched={a} individual={} {}",
                     prompts[s].len(), ref_argmax[s], if ok { "MATCH" } else { fails += 1; "MISMATCH" });
        }
    }
    for (s, c) in caches.iter_mut().enumerate() {
        let mut t = ref_argmax[s]; // decode from the agreed first token
        let mut stream = Vec::with_capacity(steps);
        for _ in 0..steps {
            let (l, _) = model.decode_step_h(&e, t, c)?;
            t = argmax(&l) as u32;
            stream.push(t);
        }
        let ok = stream == ref_streams[s];
        println!("seq {s}: decode-{steps} stream {}", if ok { "MATCH" } else { fails += 1; "DIVERGED" });
    }

    // --carried: CONTINUATION batch gate (increment (b), 2026-07-30). Per seq: fresh
    // single prime of a prefix, then the SUFFIX primed (1) single continuation
    // (prime_cache, pos>0 — the session-gate-validated arm) vs (2) batched continuation
    // (prime_cache_batch with pos>0 caches). Same standard as the fresh gate above:
    // suffix argmax + 16-step decode stream must MATCH per sequence.
    if rest.iter().any(|a| a == "--carried") {
        let prefixes: Vec<Vec<u32>> = (0..b)
            .map(|i| (0..40 + i as u32 * 13).map(|j| 61 + i as u32 * 89 + j * 29).collect())
            .collect();
        let suffixes: Vec<Vec<u32>> = (0..b)
            .map(|i| (0..18 + i as u32 * 7).map(|j| 77 + i as u32 * 53 + j * 37).collect())
            .collect();
        // reference: single continuation per seq
        let mut ref_streams: Vec<Vec<u32>> = Vec::with_capacity(b);
        let mut ref_argmax: Vec<u32> = Vec::with_capacity(b);
        for s in 0..b {
            let mut c = Cache::new(&e, &model.cfg, ctx)?;
            let _ = model.prime_cache(&e, &prefixes[s], &mut c, 0)?;
            let (logits, _, _) = model.prime_cache(&e, &suffixes[s], &mut c, 0)?;
            let mut t = argmax(&logits) as u32;
            ref_argmax.push(t);
            let mut stream = Vec::with_capacity(steps);
            for _ in 0..steps {
                let (l, _) = model.decode_step_h(&e, t, &mut c)?;
                t = argmax(&l) as u32;
                stream.push(t);
            }
            ref_streams.push(stream);
        }
        // batched continuation: fresh prefix primes (single), then ONE batched suffix prime
        let mut caches: Vec<Cache> = (0..b).map(|_| Cache::new(&e, &model.cfg, ctx)).collect::<Result<_, _>>()?;
        for s in 0..b {
            let _ = model.prime_cache(&e, &prefixes[s], &mut caches[s], 0)?;
        }
        {
            let suffix_refs: Vec<&[u32]> = suffixes.iter().map(|p| p.as_slice()).collect();
            let mut cache_refs: Vec<&mut Cache> = caches.iter_mut().collect();
            let outs = model.prime_cache_batch(&e, &suffix_refs, &mut cache_refs)?;
            for (s, (logits, _, _)) in outs.iter().enumerate() {
                let a = argmax(logits) as u32;
                let ok = a == ref_argmax[s];
                println!("carried seq {s} (P={},S={}): suffix argmax batched={a} single={} {}",
                         prefixes[s].len(), suffixes[s].len(), ref_argmax[s],
                         if ok { "MATCH" } else { fails += 1; "MISMATCH" });
            }
        }
        for (s, c) in caches.iter_mut().enumerate() {
            let mut t = ref_argmax[s];
            let mut stream = Vec::with_capacity(steps);
            for _ in 0..steps {
                let (l, _) = model.decode_step_h(&e, t, c)?;
                t = argmax(&l) as u32;
                stream.push(t);
            }
            let ok = stream == ref_streams[s];
            println!("carried seq {s}: decode-{steps} stream {}", if ok { "MATCH" } else { fails += 1; "DIVERGED" });
        }
    }

    // --bench T: N=5 medians, B x T-token prompts, sequential vs batched wall time
    if let Some(bt) = rest.iter().position(|a| a == "--bench")
        .and_then(|i| rest.get(i + 1)).and_then(|v| v.parse::<usize>().ok()) {
        let bp: Vec<Vec<u32>> = (0..b).map(|i| (0..bt as u32).map(|j| 55 + i as u32 * 97 + j * 31).collect()).collect();
        let mut seq_times = Vec::new();
        let mut bat_times = Vec::new();
        for _ in 0..5 {
            let t0 = std::time::Instant::now();
            for p in &bp {
                let mut c = Cache::new(&e, &model.cfg, bt + 64)?;
                let _ = model.prime_cache(&e, p, &mut c, 0)?;
            }
            e.stream().synchronize()?;
            seq_times.push(t0.elapsed().as_secs_f64());
            let mut cs: Vec<Cache> = (0..b).map(|_| Cache::new(&e, &model.cfg, bt + 64)).collect::<Result<_, _>>()?;
            let pr: Vec<&[u32]> = bp.iter().map(|p| p.as_slice()).collect();
            let mut cr: Vec<&mut Cache> = cs.iter_mut().collect();
            let t0 = std::time::Instant::now();
            let _ = model.prime_cache_batch(&e, &pr, &mut cr)?;
            e.stream().synchronize()?;
            bat_times.push(t0.elapsed().as_secs_f64());
        }
        seq_times.sort_by(|a, c| a.partial_cmp(c).unwrap());
        bat_times.sort_by(|a, c| a.partial_cmp(c).unwrap());
        let (sm, bm) = (seq_times[2], bat_times[2]);
        let n = (b * bt) as f64;
        println!("bench B={b} T={bt}: sequential {:.1} tok/s vs BATCHED {:.1} tok/s ({:+.1}%)",
                 n / sm, n / bm, 100.0 * (sm / bm - 1.0));
    }

    if fails == 0 {
        println!("ALL GREEN: prime-batch gate (batch={b}, uneven lengths)");
        Ok(())
    } else {
        Err(format!("prime-batch-gate: {fails} FAIL(s)").into())
    }
}
