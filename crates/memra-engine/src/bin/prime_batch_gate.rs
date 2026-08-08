//! prime-batch-gate (task #13): cross-request batched prime vs individual primes.
//!
//! The concat GEMM (m = sum_T) is a NEW NUMERIC CONFIG vs per-seq primes (different K
//! tiling) — the gate is therefore argmax/stream equality per sequence, not bit-identity:
//!   1. prefill argmax per seq: batched == individual (hard FAIL on mismatch)
//!   2. 16 greedy decode steps from each primed cache: streams must MATCH per seq
//!      (decode itself is untouched; drift here would mean the batched prime left a
//!      different cache/recurrent state beyond numeric-config tolerance).
//!
//! usage: prime-batch-gate <model.gguf> [--batch 3] [--plen 24] [--steps 16]
//!                         [--exact] [--require-pp-split]

use memra_engine::cache::Cache;
use memra_engine::forward::argmax;
use memra_engine::hybrid::HybridModel;
use memra_engine::Engine;
use memra_gguf::GgufFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: prime-batch-gate <model.gguf> [--batch N]");
    let rest: Vec<String> = args.collect();
    let opt_usize = |name: &str, default: usize| {
        rest.iter().position(|a| a == name)
            .and_then(|i| rest.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(default)
    };
    let b = opt_usize("--batch", 3);
    let plen = opt_usize("--plen", 24);
    let steps = opt_usize("--steps", 16);
    let exact = rest.iter().any(|a| a == "--exact");
    let require_pp_split = rest.iter().any(|a| a == "--require-pp-split");

    let e = Engine::new(0)?;
    let g = GgufFile::open(&path)?;
    let model = HybridModel::load_without_mtp(&e, &g)?;
    println!("loaded {} ({} layers); batch={b}", g.arch().unwrap_or("?"), model.layers.len());

    // Deliberately UNEVEN prompt lengths so offsets/tails are exercised. Step35's standing
    // gate uses plen=520: every sequence crosses its 512-token SWA window.
    let prompts: Vec<Vec<u32>> = (0..b)
        .map(|i| (0..plen as u32 + i as u32 * 17)
             .map(|j| 55 + i as u32 * 97 + j * 31).collect())
        .collect();
    let ctx = prompts.iter().map(Vec::len).max().unwrap_or(0) + steps + 32;
    let mut fails = 0usize;
    let bit_diff = |a: &[f32], b: &[f32]| -> usize {
        a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
            + a.len().abs_diff(b.len())
    };

    // individual reference primes + decode streams
    let mut ref_streams: Vec<Vec<u32>> = Vec::with_capacity(b);
    let mut ref_argmax: Vec<u32> = Vec::with_capacity(b);
    let mut ref_logits: Vec<Vec<f32>> = Vec::with_capacity(b);
    let mut ref_hseeds: Vec<Vec<f32>> = Vec::with_capacity(b);
    let mut ref_hiddens: Vec<Vec<f32>> = Vec::with_capacity(b);
    let mut ref_decode_logits: Vec<Vec<Vec<f32>>> = Vec::with_capacity(b);
    let mut ref_decode_inputs: Vec<Vec<u32>> = Vec::with_capacity(b);
    for p in &prompts {
        let mut c = memra_engine::pp::new_cache(&e, &model.cfg, ctx)?;
        let (logits, h_seed, hidden) = model.prime_cache(&e, p, &mut c, 0)?;
        let mut t = argmax(&logits) as u32;
        ref_argmax.push(t);
        let mut stream = Vec::with_capacity(steps);
        let mut dec = Vec::with_capacity(steps);
        let mut inputs = Vec::with_capacity(steps);
        for _ in 0..steps {
            inputs.push(t);
            let (l, _) = model.decode_step_h(&e, t, &mut c)?;
            t = argmax(&l) as u32;
            stream.push(t);
            dec.push(l);
        }
        ref_logits.push(logits);
        ref_hseeds.push(e.dtoh(&h_seed)?);
        ref_hiddens.push(e.dtoh(&hidden)?);
        ref_decode_logits.push(dec);
        ref_decode_inputs.push(inputs);
        ref_streams.push(stream);
    }

    // batched prime + decode streams
    let mut caches: Vec<Cache> = (0..b)
        .map(|_| memra_engine::pp::new_cache(&e, &model.cfg, ctx))
        .collect::<Result<_, _>>()?;
    let batch0 = memra_engine::pp::step35_prime_batches();
    let split0 = memra_engine::pp::step35_prime_batch_splits();
    {
        let prompt_refs: Vec<&[u32]> = prompts.iter().map(|p| p.as_slice()).collect();
        let mut cache_refs: Vec<&mut Cache> = caches.iter_mut().collect();
        let outs = model.prime_cache_batch(&e, &prompt_refs, &mut cache_refs)?;
        for (s, (logits, h_seed, hidden)) in outs.iter().enumerate() {
            let a = argmax(logits) as u32;
            let ok = a == ref_argmax[s];
            println!("seq {s} (T={}): prefill argmax batched={a} individual={} {}",
                     prompts[s].len(), ref_argmax[s], if ok { "MATCH" } else { fails += 1; "MISMATCH" });
            if exact {
                let hs = e.dtoh(h_seed)?;
                let hid = e.dtoh(hidden)?;
                let dl = bit_diff(&ref_logits[s], logits);
                let dhs = bit_diff(&ref_hseeds[s], &hs);
                let dh = bit_diff(&ref_hiddens[s], &hid);
                println!("seq {s}: exact logits diff {dl}/{} h_seed diff {dhs}/{} hidden diff {dh}/{}",
                         logits.len(), hs.len(), hid.len());
                if dl + dhs + dh != 0 {
                    fails += 1;
                }
            }
        }
    }
    for (s, c) in caches.iter_mut().enumerate() {
        let mut stream = Vec::with_capacity(steps);
        let mut dec_diff = 0usize;
        for step in 0..steps {
            // Teacher-force the reference inputs in exact mode so a first mismatch cannot
            // desynchronize later KV comparisons.
            let input = if exact {
                ref_decode_inputs[s][step]
            } else if step == 0 {
                ref_argmax[s]
            } else {
                *stream.last().unwrap()
            };
            let (l, _) = model.decode_step_h(&e, input, c)?;
            let t = argmax(&l) as u32;
            stream.push(t);
            if exact {
                dec_diff += bit_diff(&ref_decode_logits[s][step], &l);
            }
        }
        let ok = stream == ref_streams[s];
        println!("seq {s}: decode-{steps} stream {}", if ok { "MATCH" } else { fails += 1; "DIVERGED" });
        if exact {
            println!("seq {s}: teacher-forced decode logit diff {dec_diff}");
            if dec_diff != 0 {
                fails += 1;
            }
        }
    }
    if exact && model.cfg.step35.is_some() {
        let nb = memra_engine::pp::step35_prime_batches() - batch0;
        let ns = memra_engine::pp::step35_prime_batch_splits() - split0;
        let live = nb >= 1 && (!require_pp_split || ns >= 1);
        println!("step35 batched-prime liveness: batches={nb} pp_splits={ns} require_pp_split={require_pp_split} {}",
                 if live { "LIVE" } else { "NOT-LIVE" });
        if !live {
            fails += 1;
        }
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
            let mut c = memra_engine::pp::new_cache(&e, &model.cfg, ctx)?;
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
        let mut caches: Vec<Cache> = (0..b)
            .map(|_| memra_engine::pp::new_cache(&e, &model.cfg, ctx))
            .collect::<Result<_, _>>()?;
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

    // --bench T: N=5 paired medians, B x T-token prompts, sequential vs batched wall time.
    // Alternate arm order so a fixed warmup/thermal trend cannot favor the same arm in every
    // pair, and print every raw pair before the summary median.
    if let Some(bt) = rest.iter().position(|a| a == "--bench")
        .and_then(|i| rest.get(i + 1)).and_then(|v| v.parse::<usize>().ok()) {
        let bp: Vec<Vec<u32>> = (0..b).map(|i| (0..bt as u32).map(|j| 55 + i as u32 * 97 + j * 31).collect()).collect();
        let mut seq_times = Vec::new();
        let mut bat_times = Vec::new();
        let run_seq = || -> Result<f64, Box<dyn std::error::Error>> {
            let t0 = std::time::Instant::now();
            for p in &bp {
                let mut c = memra_engine::pp::new_cache(&e, &model.cfg, bt + 64)?;
                let _ = model.prime_cache(&e, p, &mut c, 0)?;
            }
            e.stream().synchronize()?;
            Ok(t0.elapsed().as_secs_f64())
        };
        let run_batch = || -> Result<f64, Box<dyn std::error::Error>> {
            let mut cs: Vec<Cache> = (0..b)
                .map(|_| memra_engine::pp::new_cache(&e, &model.cfg, bt + 64))
                .collect::<Result<_, _>>()?;
            let pr: Vec<&[u32]> = bp.iter().map(|p| p.as_slice()).collect();
            let mut cr: Vec<&mut Cache> = cs.iter_mut().collect();
            let t0 = std::time::Instant::now();
            let _ = model.prime_cache_batch(&e, &pr, &mut cr)?;
            e.stream().synchronize()?;
            Ok(t0.elapsed().as_secs_f64())
        };
        for rep in 0..5 {
            let (order, st, bt) = if rep % 2 == 0 {
                ("serial,batch", run_seq()?, run_batch()?)
            } else {
                let bt = run_batch()?;
                ("batch,serial", run_seq()?, bt)
            };
            println!(
                "bench-run B={b} T={} rep={} order={order} serial_ms={:.3} batch_ms={:.3}",
                bp[0].len(),
                rep + 1,
                st * 1e3,
                bt * 1e3,
            );
            seq_times.push(st);
            bat_times.push(bt);
        }
        seq_times.sort_by(|a, c| a.partial_cmp(c).unwrap());
        bat_times.sort_by(|a, c| a.partial_cmp(c).unwrap());
        let (sm, bm) = (seq_times[2], bat_times[2]);
        let n = (b * bt) as f64;
        println!(
            "bench B={b} T={bt} N=5 alternating: serial_wall_ms={:.3} \
             batch_wall_ms={:.3} sequential={:.1} tok/s batched={:.1} tok/s ({:+.1}%)",
            sm * 1e3,
            bm * 1e3,
            n / sm,
            n / bm,
            100.0 * (sm / bm - 1.0),
        );
    }

    if fails == 0 {
        println!("ALL GREEN: prime-batch gate (batch={b}, uneven lengths)");
        Ok(())
    } else {
        Err(format!("prime-batch-gate: {fails} FAIL(s)").into())
    }
}
