//! Spec-verify economics probe (lane/verify-economics, 2026-08-02): per-step wall cost of the
//! T=1 decode program vs the batched verify forward at T=1..TMAX, at a FIXED cache position
//! (snapshot/rollback between every measurement — every arm sees the identical KV length and
//! recurrent state, so the numbers are directly comparable; no position drift).
//!
//! Arms, interleaved round-robin (thermal drift spreads evenly across arms):
//!   decode_h   — eager `decode_step_h` (the T=1 decode program incl. its [n_vocab] logits dtoh)
//!   verify_tN  — `decode_step_t_h_emb_dev` at T=N (the spec verify kernel chain, device
//!                logits — the accept walk's argmax/readback is NOT included; it is measured
//!                separately by the MEMRA_SPEC_PHASE seam in the real loop)
//!
//! The verify tokens are a REAL greedy continuation (collected eagerly, then rolled back), so
//! attention/router content is realistic. Timing is sync-to-sync (stream.synchronize before
//! Instant::now and again before elapsed). Rollback cost sits OUTSIDE the timed region.
//!
//! Usage: spec-econ <model.gguf>
//!   MEMRA_PROMPT_FILE / MEMRA_PROMPT — prompt text (run-spec conventions, MEMRA_CHAT honored)
//!   MEMRA_ECON_N    — measured iterations per arm (default 50; 3 warmups extra)
//!   MEMRA_ECON_TMAX — max verify T (default 6 = the K=5 verify tier w/ pending col)
//!   MEMRA_ECON_ONLY — run ONE arm ("decode_h" or "verify_tN") — the nsys attribution mode:
//!                     a profile of the process then contains only that arm's kernel stream
//!                     (plus prime/setup, excluded by timestamp or cudaProfilerApi capture).
//! Prints one human line per arm + one `[econ-json]` line for receipts.

use memra_engine::Engine;
use memra_engine::hybrid::HybridModel;
use memra_engine::forward::argmax;
use memra_gguf::GgufFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: spec-econ <model.gguf>");
    let n_iter: usize = std::env::var("MEMRA_ECON_N").ok().and_then(|s| s.parse().ok()).unwrap_or(50);
    let t_max: usize = std::env::var("MEMRA_ECON_TMAX").ok().and_then(|s| s.parse().ok()).unwrap_or(6);
    let warmup = 3usize;

    let e = Engine::new(0)?;
    let g = GgufFile::open(&path)?;
    let model = HybridModel::load(&e, &g)?;

    let prompt: Vec<u32> = if let Ok(text) = std::env::var("MEMRA_PROMPT_FILE")
        .map(|f| std::fs::read_to_string(&f).expect("MEMRA_PROMPT_FILE unreadable"))
        .or_else(|_| std::env::var("MEMRA_PROMPT"))
    {
        let tok = memra_tokenizer::Tokenizer::from_gguf(&g)?;
        let to_encode = if std::env::var("MEMRA_CHAT").is_ok() {
            tok.apply_chat_template(&[("user", &text)], true)
        } else {
            text
        };
        tok.encode(&to_encode, true)
    } else {
        (101..=228).collect()
    };
    println!("[econ] model={path} prompt_tokens={} N={n_iter} t_max={t_max}", prompt.len());

    let max_ctx = prompt.len() + t_max + 32;
    let mut cache = memra_engine::cache::Cache::new(&e, &model.cfg, max_ctx)?;

    // Prime + collect a real greedy continuation of t_max tokens (rolled back afterwards).
    let (logits, _h, _hs) = model.prime_cache(&e, &prompt, &mut cache)?;
    e.stream().synchronize()?;
    let snap = cache.snapshot(&e)?;
    let pos0 = cache.pos;
    let mut cont: Vec<u32> = vec![argmax(&logits) as u32];
    for i in 0..t_max {
        let (l, _) = model.decode_step_h(&e, cont[i], &mut cache)?;
        cont.push(argmax(&l) as u32);
    }
    cache.rollback(&e, &snap, 0)?;
    e.stream().synchronize()?;
    println!("[econ] primed pos={pos0}, continuation={cont:?}");

    // arm 0 = decode_h; arms 1..=t_max = verify_tN.
    let n_arms = 1 + t_max;
    let only: Option<usize> = std::env::var("MEMRA_ECON_ONLY").ok().map(|v| {
        if v == "decode_h" { 0 } else {
            v.strip_prefix("verify_t").and_then(|n| n.parse().ok()).expect("MEMRA_ECON_ONLY: decode_h|verify_tN")
        }
    });
    let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(n_iter); n_arms];
    for it in 0..(warmup + n_iter) {
        for arm in 0..n_arms {
            if let Some(o) = only {
                if arm != o {
                    continue;
                }
            }
            e.stream().synchronize()?;
            let t0 = std::time::Instant::now();
            if arm == 0 {
                let (_l, _h) = model.decode_step_h(&e, cont[0], &mut cache)?;
            } else {
                let toks = &cont[0..arm];
                let (_ld, _hs) = model.decode_step_t_h_emb_dev(&e, toks, pos0, &mut cache, None)?;
            }
            e.stream().synchronize()?;
            let dt = t0.elapsed().as_secs_f64() * 1e3;
            cache.rollback(&e, &snap, 0)?;
            e.stream().synchronize()?;
            if it >= warmup {
                samples[arm].push(dt);
            }
        }
    }

    let stats = |v: &mut Vec<f64>| -> (f64, f64, f64, f64) {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = v[v.len() / 2];
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        (med, mean, v[0], v[(v.len() as f64 * 0.9) as usize])
    };
    let mut med0 = 0.0;
    let mut first = true;
    let mut json = format!(
        "{{\"probe\":\"spec-econ\",\"model\":\"{path}\",\"prompt_tokens\":{},\"pos\":{pos0},\"n\":{n_iter},\"arms\":{{",
        prompt.len()
    );
    for arm in 0..n_arms {
        if samples[arm].is_empty() {
            continue;
        }
        let name = if arm == 0 { "decode_h".to_string() } else { format!("verify_t{arm}") };
        let (med, mean, min, p90) = stats(&mut samples[arm]);
        if arm == 0 {
            med0 = med;
        }
        let rel = if med0 > 0.0 { format!("  x{:.3} vs decode_h", med / med0) } else { String::new() };
        println!(
            "[econ] {name:10} N={n_iter}  med={med:7.3}ms  mean={mean:7.3}ms  min={min:7.3}ms  p90={p90:7.3}ms{rel}"
        );
        json.push_str(&format!(
            "{}\"{name}\":{{\"med_ms\":{med:.4},\"mean_ms\":{mean:.4},\"min_ms\":{min:.4},\"p90_ms\":{p90:.4}}}",
            if first { "" } else { "," }
        ));
        first = false;
    }
    json.push_str("}}");
    println!("[econ-json] {json}");
    Ok(())
}
