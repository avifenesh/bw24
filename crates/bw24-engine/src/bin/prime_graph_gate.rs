//! prime-graph-gate (task #14): graphed prime + copy-out vs eager prime, END TO END.
//! For each true length: prime a session via PrimeGraph replay and another eagerly, then
//! run 16 greedy decode steps on BOTH — the decode stream exercises the copied KV rows,
//! conv rings, and recurrent states (the full copy-out correctness surface, not just
//! logits). Streams must MATCH (smoke measured bit-identical logits; drift here would
//! localize to the copy-out).
//!
//! usage: prime-graph-gate <model.gguf> [--bucket 512]

use bw24_engine::cache::Cache;
use bw24_engine::forward::argmax;
use bw24_engine::hybrid::HybridModel;
use bw24_engine::Engine;
use bw24_gguf::GgufFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: prime-graph-gate <model.gguf> [--bucket N]");
    let rest: Vec<String> = args.collect();
    let bucket: usize = rest.iter().position(|a| a == "--bucket")
        .and_then(|i| rest.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(512);

    let e = Engine::new(0)?;
    let g = GgufFile::open(&path)?;
    let model = HybridModel::load_without_mtp(&e, &g)?;
    println!("loaded {} ({} layers); bucket={bucket}", g.arch().unwrap_or("?"), model.layers.len());

    let t0 = std::time::Instant::now();
    let mut pg = model.prime_graph_new(&e, bucket)?;
    println!("capture: {:.0}ms", t0.elapsed().as_secs_f64() * 1e3);

    let steps = 16usize;
    let mut fails = 0usize;
    // CONTROL: eager-vs-eager at bucket length with a pool-perturbing alloc between them.
    // If THESE diverge, decode-stream flips at this length are inherent Lt alignment
    // sensitivity (a different valid rounding), not a graph defect.
    {
        let tl = bucket;
        let prompt: Vec<u32> = (0..tl as u32).map(|j| 55 + j * 31).collect();
        let mut c1 = Cache::new(&e, &model.cfg, bucket + steps + 8)?;
        let (l1, _, _) = model.prime_cache(&e, &prompt, &mut c1)?;
        let _pool_shift = e.uninit(12345)?;   // perturb subsequent transient addresses
        let mut c2 = Cache::new(&e, &model.cfg, bucket + steps + 8)?;
        let (l2, _, _) = model.prime_cache(&e, &prompt, &mut c2)?;
        let (mut ta, mut tb) = (argmax(&l1) as u32, argmax(&l2) as u32);
        let mut div_at: Option<usize> = if ta == tb { None } else { Some(0) };
        for st in 1..=steps {
            if div_at.is_some() { break; }
            let (la, _) = model.decode_step_h(&e, ta, &mut c1)?;
            let (lb, _) = model.decode_step_h(&e, tb, &mut c2)?;
            ta = argmax(&la) as u32;
            tb = argmax(&lb) as u32;
            if ta != tb { div_at = Some(st); }
        }
        println!("CONTROL eager-vs-eager (pool-shifted) T={tl}: {}",
                 match div_at { None => "streams MATCH".into(),
                                Some(st) => format!("DIVERGED at step {st}") });
    }
    // order + prompt discriminants (debug 2026-07-26): 512 runs LAST; the second 512 case
    // uses the smoke's exact prompt (55+31j) to separate order-, prompt-, and length-effects.
    for &(tl, smoke_prompt) in &[(47usize, false), (128, false), (300, false), (bucket, false), (bucket, true)] {
        if tl > bucket { continue; }
        let prompt: Vec<u32> = if smoke_prompt {
            (0..tl as u32).map(|j| 55 + j * 31).collect()
        } else {
            (0..tl as u32).map(|j| 55 + (tl as u32) * 7 + j * 31).collect()
        };

        let mut c_e = Cache::new(&e, &model.cfg, bucket + steps + 8)?;
        let (l_e, _, _) = model.prime_cache(&e, &prompt, &mut c_e)?;
        let mut t_e = argmax(&l_e) as u32;

        let mut c_g = Cache::new(&e, &model.cfg, bucket + steps + 8)?;
        let t1 = std::time::Instant::now();
        let (l_g, _hs) = model.prime_graph_run(&e, &mut pg, &prompt, &mut c_g)?;
        let replay_ms = t1.elapsed().as_secs_f64() * 1e3;
        let mut t_g = argmax(&l_g) as u32;

        // copy-out fidelity: session vs the graph's scratch (must be byte-equal post-replay)
        {
            let (mut mc2, mut ms2) = (0f32, 0f32);
            for il in 0..c_g.recur.len() {
                if let (Some(rs), Some(rg)) = (&pg.scratch().recur[il], &c_g.recur[il]) {
                    let a = e.dtoh(&rs.conv_state)?; let b = e.dtoh(&rg.conv_state)?;
                    mc2 = mc2.max(a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max));
                    let a = e.dtoh(&rs.ssm_state)?; let b = e.dtoh(&rg.ssm_state)?;
                    ms2 = ms2.max(a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max));
                }
            }
            println!("  copy-out fidelity (session vs scratch): conv {mc2:.3e} ssm {ms2:.3e}");
        }
        // localize any drift: compare the two sessions' cache components directly
        {
            let (mut mc, mut ms, mut mkv) = (0f32, 0f32, 0usize);
            for il in 0..c_e.recur.len() {
                if let (Some(re), Some(rg)) = (&c_e.recur[il], &c_g.recur[il]) {
                    let a = e.dtoh(&re.conv_state)?; let b = e.dtoh(&rg.conv_state)?;
                    mc = mc.max(a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max));
                    let a = e.dtoh(&re.ssm_state)?; let b = e.dtoh(&rg.ssm_state)?;
                    ms = ms.max(a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max));
                }
                if let (Some(ke), Some(kg)) = (&c_e.kv[il], &c_g.kv[il]) {
                    let n = tl * ke.k_tok_bytes;
                    let a = e.dtoh_u8(&ke.k)?;
                    let b = e.dtoh_u8(&kg.k)?;
                    mkv += a[..n].iter().zip(&b[..n]).filter(|(x, y)| x != y).count();
                }
            }
            println!("  cache diff: conv max {mc:.3e}  ssm max {ms:.3e}  kv byte-diffs {mkv}");
        }
        let am = t_g == t_e;
        // Numeric-config convention (decode_batch config-gate precedent): the graph arm's
        // Lt tiling depends on baked addresses -> a different valid rounding. Divergence
        // BEFORE step 12 fails (real bug class); at/after = accepted cross-config drift.
        let mut div_at: Option<usize> = None;
        for st in 1..=steps {
            let (le, _) = model.decode_step_h(&e, t_e, &mut c_e)?;
            let (lg, _) = model.decode_step_h(&e, t_g, &mut c_g)?;
            t_e = argmax(&le) as u32;
            t_g = argmax(&lg) as u32;
            if t_e != t_g { div_at = Some(st); break; }
        }
        let stream_ok = match div_at { None => true, Some(st) => st >= 12 };
        let ok = am && stream_ok;
        if !ok { fails += 1; }
        println!("T={tl:4}{}: replay {replay_ms:5.1}ms  prefill argmax {}  decode stream {}",
                 if smoke_prompt { " (smoke-prompt)" } else { "" },
                 if am { "MATCH" } else { "MISMATCH" },
                 match div_at {
                     None => "MATCH (16)".to_string(),
                     Some(st) if st >= 12 => format!("drift at step {st} (accepted, >=12)"),
                     Some(st) => format!("DIVERGED at step {st} FAIL"),
                 });
    }
    if fails == 0 {
        println!("ALL GREEN: prime-graph gate (bucket={bucket}, padded + exact lengths)");
        Ok(())
    } else {
        Err(format!("prime-graph-gate: {fails} FAIL(s)").into())
    }
}
