//! gemma4 bring-up gate: prefill-only forward on raw token ids, prints greedy continuation
//! (each step re-runs the full prefill — O(n^2), gate-only) + top-8 (id, logit) of the first
//! step for logit-level comparison against llama.cpp on the IDENTICAL GGUF.
use bw24_engine::Engine;
use bw24_engine::hybrid::HybridModel;
use bw24_gguf::GgufFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: gemma-gate <model.gguf> <tok ids...>");
    let mut toks: Vec<u32> = std::env::args().skip(2).filter_map(|s| s.parse().ok()).collect();
    // BW24_GRAPH_GATE=N: graph-replay gate — stream identity vs eager + throughputs.
    if let Ok(nn) = std::env::var("BW24_GRAPH_GATE") {
        let n: usize = nn.parse().unwrap_or(96);
        let e = bw24_engine::Engine::new(0)?;
        let g = bw24_gguf::GgufFile::open(&path)?;
        let model = bw24_engine::hybrid::HybridModel::load(&e, &g)?;
        let toks: Vec<u32> = std::env::args().skip(2).filter_map(|s| s.parse().ok()).collect();
        // eager reference
        let mut c1 = bw24_engine::cache::Cache::new(&e, &model.cfg, toks.len() + n + 8)?;
        let mut ll = Vec::new();
        for &t in &toks { ll = model.decode_step(&e, t, &mut c1)?; }
        e.stream().synchronize()?;
        let t0 = std::time::Instant::now();
        let mut eager: Vec<u32> = Vec::new();
        let mut next = bw24_engine::forward::argmax(&ll) as u32;
        for _ in 0..n {
            eager.push(next);
            ll = model.decode_step(&e, next, &mut c1)?;
            next = bw24_engine::forward::argmax(&ll) as u32;
        }
        e.stream().synchronize()?;
        let dt_e = t0.elapsed().as_secs_f64();
        // graph loop
        let mut c2 = bw24_engine::cache::Cache::new(&e, &model.cfg, toks.len() + n + 8)?;
        let mut ll2 = Vec::new();
        for &t in &toks { ll2 = model.decode_step(&e, t, &mut c2)?; }
        let first = bw24_engine::forward::argmax(&ll2) as u32;
        e.stream().synchronize()?;
        let t1 = std::time::Instant::now();
        let (graph, _reason) = model.gemma4_generate_graph(&e, c2.pos, first, &mut c2, n, &[],
                                                           |_| true)?;
        e.stream().synchronize()?;
        let dt_g = t1.elapsed().as_secs_f64();
        let same = eager.iter().zip(&graph).take_while(|(a, b)| a == b).count();
        println!("GRAPH-GATE: eager {:.2} tok/s | graph {:.2} tok/s | stream {}/{} {}",
                 n as f64 / dt_e, graph.len() as f64 / dt_g, same, n,
                 if same == n.min(graph.len()) { "IDENTICAL" } else { "MISMATCH" });
        if same < n.min(graph.len()) {
            println!("eager: {:?}", &eager[..8.min(eager.len())]);
            println!("graph: {:?}", &graph[..8.min(graph.len())]);
        }
        return Ok(());
    }
    // BW24_DC_GATE=N: device-counter decode gate — the dc chain's N-token greedy stream must
    // be IDENTICAL to the eager decode_step chain; prints both throughputs.
    if let Ok(nn) = std::env::var("BW24_DC_GATE") {
        let n: usize = nn.parse().unwrap_or(64);
        let e = bw24_engine::Engine::new(0)?;
        let g = bw24_gguf::GgufFile::open(&path)?;
        let model = bw24_engine::hybrid::HybridModel::load(&e, &g)?;
        let toks: Vec<u32> = std::env::args().skip(2).filter_map(|s| s.parse().ok()).collect();
        let n_vocab = model.output.out_features();
        // eager reference
        let mut c1 = bw24_engine::cache::Cache::new(&e, &model.cfg, toks.len() + n + 8)?;
        let mut ll = Vec::new();
        for &t in &toks { ll = model.decode_step(&e, t, &mut c1)?; }
        e.stream().synchronize()?;
        let t0 = std::time::Instant::now();
        let mut eager: Vec<u32> = Vec::new();
        let mut next = bw24_engine::forward::argmax(&ll) as u32;
        for _ in 0..n {
            eager.push(next);
            ll = model.decode_step(&e, next, &mut c1)?;
            next = bw24_engine::forward::argmax(&ll) as u32;
        }
        e.stream().synchronize()?;
        let dt_e = t0.elapsed().as_secs_f64();
        // dc chain
        let mut c2 = bw24_engine::cache::Cache::new(&e, &model.cfg, toks.len() + n + 8)?;
        let mut ll2 = Vec::new();
        for &t in &toks { ll2 = model.decode_step(&e, t, &mut c2)?; }
        let embd_gpu = e.upload_u8(&model.embd.raw)?;
        let (qt, rb) = model.embd.qt_and_row_bytes(model.cfg.n_embd as usize);
        let first = bw24_engine::forward::argmax(&ll2) as u32;
        // sync the device counters to the host-primed mirrors (eager never touches len_d).
        for kvl in c2.kv.iter_mut().flatten() {
            e.set_i32_one(&mut kvl.len_d, kvl.len as i32)?;
        }
        let mut token_d = e.stream().clone_htod(&[first])?;
        let mut pos_d = e.htod_i32(&[c2.pos as i32])?;
        e.stream().synchronize()?;
        let t1 = std::time::Instant::now();
        let mut dc: Vec<u32> = Vec::new();
        for _ in 0..n {
            dc.push(e.dtoh_u32(&token_d)?[0]);
            token_d = model.gemma4_decode_step_dc(&e, &token_d, &mut pos_d, &embd_gpu, qt, rb,
                                                  &mut c2, n_vocab, None)?;
        }
        e.stream().synchronize()?;
        let dt_d = t1.elapsed().as_secs_f64();
        let same = eager.iter().zip(&dc).take_while(|(a, b)| a == b).count();
        println!("DC-GATE: eager {:.2} tok/s | dc {:.2} tok/s | stream {}/{} {}",
                 n as f64 / dt_e, n as f64 / dt_d, same, n,
                 if same == n { "IDENTICAL" } else { "MISMATCH" });
        if same < n {
            println!("eager: {:?}", &eager[..same.min(eager.len()).saturating_add(4).min(n)]);
            println!("dc   : {:?}", &dc[..same.saturating_add(4).min(n)]);
        }
        return Ok(());
    }
    // BW24_SPEC=K + BW24_DRAFT=<drafter.gguf>: MTP spec loop — prime, draft K, verify, accept.
    // Compares the spec token stream against plain greedy (self-consistency) + times both.
    if let (Ok(kk), Ok(dpath)) = (std::env::var("BW24_SPEC"), std::env::var("BW24_DRAFT")) {
        let k: usize = kk.parse().unwrap_or(4);
        let n_new: usize = std::env::var("BW24_NGEN").ok().and_then(|s| s.parse().ok()).unwrap_or(64);
        let e = bw24_engine::Engine::new(0)?;
        let g = bw24_gguf::GgufFile::open(&path)?;
        let model = bw24_engine::hybrid::HybridModel::load(&e, &g)?;
        let dg = bw24_gguf::GgufFile::open(&dpath)?;
        let draft = bw24_engine::gemma_spec::GemmaDraft::load(&e, &dg)?;
        let toks: Vec<u32> = std::env::args().skip(2).filter_map(|s| s.parse().ok()).collect();
        println!("spec K={k} n_new={n_new} prompt={} toks", toks.len());
        // plain greedy reference + timing
        e.stream().synchronize()?;
        let t0 = std::time::Instant::now();
        let plain = model.generate(&e, &toks, n_new)?;
        e.stream().synchronize()?;
        let dt_plain = t0.elapsed().as_secs_f64()
            - bw24_engine::PRIME_NANOS.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9;
        // spec run + timing
        let t1 = std::time::Instant::now();
        let spec = model.generate_spec_gemma(&e, &draft, &toks, n_new, k, &[])?;
        e.stream().synchronize()?;
        let dt_spec = t1.elapsed().as_secs_f64()
            - bw24_engine::PRIME_NANOS.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9;
        let same = plain.iter().zip(&spec).take_while(|(a, b)| a == b).count();
        println!("plain: {:.2} tok/s | spec: {:.2} tok/s ({:.2}x) | stream agreement {}/{}",
                 plain.len() as f64 / dt_plain, spec.len() as f64 / dt_spec,
                 dt_plain / dt_spec * (spec.len() as f64 / plain.len() as f64),
                 same, plain.len().min(spec.len()));
        if same < plain.len().min(spec.len()) {
            println!("plain: {:?}", &plain[..plain.len().min(24)]);
            println!("spec : {:?}", &spec[..spec.len().min(24)]);
        }
        return Ok(());
    }
    // BW24_VERIFY_GATE=K: batched-verify self-consistency — decode the prompt tokenwise
    // (reference), then on a fresh cache decode the prefix and run ONE decode_step_t over the
    // last K tokens; per-position argmax must match the tokenwise chain (the spec K-gate).
    if let Ok(kk) = std::env::var("BW24_VERIFY_GATE") {
        let k: usize = kk.parse().unwrap_or(4);
        let e = bw24_engine::Engine::new(0)?;
        let g = bw24_gguf::GgufFile::open(&path)?;
        let model = bw24_engine::hybrid::HybridModel::load(&e, &g)?;
        let n_vocab = model.output.out_features();
        let toks: Vec<u32> = std::env::args().skip(2).filter_map(|s| s.parse().ok()).collect();
        assert!(toks.len() > k + 1, "prompt must exceed K+1");
        let split = toks.len() - k;
        // reference: tokenwise decode over the whole prompt
        let mut c1 = bw24_engine::cache::Cache::new(&e, &model.cfg, toks.len() + 8)?;
        let mut ref_am: Vec<usize> = Vec::new();
        let mut ref_logits: Vec<Vec<f32>> = Vec::new();
        for (i, &tk) in toks.iter().enumerate() {
            let l = model.decode_step(&e, tk, &mut c1)?;
            if i >= split {
                ref_am.push(bw24_engine::forward::argmax(&l));
                ref_logits.push(l.clone());
            }
        }
        // candidate: prefix tokenwise, tail as ONE batched verify
        let mut c2 = bw24_engine::cache::Cache::new(&e, &model.cfg, toks.len() + 8)?;
        for &tk in &toks[..split] { let _ = model.decode_step(&e, tk, &mut c2)?; }
        let lv = model.decode_step_t(&e, &toks[split..], split, &mut c2)?;
        let mut all_ok = true;
        for i in 0..k {
            let am = bw24_engine::forward::argmax(&lv[i * n_vocab..(i + 1) * n_vocab]);
            let ok = am == ref_am[i];
            all_ok &= ok;
            println!("verify pos {i}: batched={am} tokenwise={} {}", ref_am[i],
                     if ok { "MATCH" } else { "MISMATCH" });
            if let Some(rl) = ref_logits.get(i) {
                let md = rl.iter().zip(&lv[i * n_vocab..(i + 1) * n_vocab])
                    .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                println!("  pos {i} logit maxdiff={md:.3e}");
            }
        }
        println!("VERIFY-GATE K={k}: {}", if all_ok { "PASS" } else { "FAIL" });
        return Ok(());
    }
    let n_new: usize = std::env::var("BW24_NGEN").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    let e = Engine::new(0)?;
    let g = GgufFile::open(&path)?;
    let model = HybridModel::load(&e, &g)?;
    println!("loaded {} ({} layers), prompt {} toks", g.arch().unwrap_or("?"), model.cfg.n_layer, toks.len());

    for step in 0..n_new {
        let logits = model.forward_last(&e, &toks)?;
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        if step == 0 {
            let top: Vec<String> = idx[..8].iter().map(|&i| format!("{i}:{:.4}", logits[i])).collect();
            println!("step0 top8: {}", top.join(" "));
            println!("step0 logits[0..3]={:?} [9079]={:.4} [506]={:.4}",
                     &logits[..3], logits[9079], logits[506]);
        }
        toks.push(idx[0] as u32);
        println!("step {step}: tok {}", idx[0]);
    }
    println!("continuation: {:?}", &toks[toks.len() - n_new..]);
    Ok(())
}
