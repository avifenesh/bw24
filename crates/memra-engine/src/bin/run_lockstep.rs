//! Lane-3 M1 harness: lockstep multi-stream decode over one Hy3 checkpoint.
//!
//! `run-lockstep <hf_or_repack_dir>` with MEMRA_LOCKSTEP_M streams (default 2), MEMRA_NGEN
//! tokens per stream (default 32), MEMRA_PROMPT/MEMRA_CHAT as in run-gen. Every stream serves
//! the same prompt, so the correctness gate is internal: all streams must emit identical
//! token sequences (each stream's math is decode_step_h's, so this also matches the
//! single-stream run). Prints per-stream tokens, aggregate and per-stream throughput, and
//! the decode-window CPU expert counters that show the cross-stream io amortization.

use memra_engine::hybrid::HybridModel;
use memra_engine::Engine;
use memra_tokenizer::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: run-lockstep <hy3 runtime dir>");
    let e = Engine::new(0)?;
    let dir = std::path::Path::new(&path);
    if !dir.is_dir() {
        return Err("run-lockstep expects the Hy3 runtime directory".into());
    }
    let is_repack = dir.join("manifest.json").exists();
    let (src, tok_dir): (Box<dyn memra_gguf::source::TensorSource>, std::path::PathBuf) =
        if is_repack {
            let rs = memra_gguf::source::Hy3RepackSource::open(dir)?;
            let td = rs
                .source_dir()
                .filter(|d| d.join("tokenizer.json").exists())
                .unwrap_or(dir)
                .to_path_buf();
            (Box::new(rs), td)
        } else {
            (
                Box::new(memra_gguf::source::SafetensorsSource::open(dir)?),
                dir.to_path_buf(),
            )
        };
    let model = HybridModel::load_from_source_without_mtp(&e, src.as_ref())?;
    println!("loaded {:?} ({} trunk layers)", model.cfg.arch, model.layers.len());

    let m: usize = std::env::var("MEMRA_LOCKSTEP_M")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&m| (1..=16).contains(&m))
        .unwrap_or(2);
    let n_new: usize = std::env::var("MEMRA_NGEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);

    // MEMRA_PROMPTS_FILE: one prompt per line, cycled across streams — the honest
    // mixed-workload regime (same-prompt streams route identically and overstate expert
    // overlap). Default: MEMRA_PROMPT for every stream (the identity-gate regime).
    let texts: Vec<String> = match std::env::var("MEMRA_PROMPTS_FILE") {
        Ok(path) => std::fs::read_to_string(&path)?
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(str::to_string)
            .collect(),
        Err(_) => vec![std::env::var("MEMRA_PROMPT")
            .unwrap_or_else(|_| "Explain speculative decoding briefly.".to_string())],
    };
    if texts.is_empty() {
        return Err("MEMRA_PROMPTS_FILE contains no prompts".into());
    }
    let tok = Tokenizer::from_hf_dir(&tok_dir)
        .map_err(|err| format!("HF tokenizer init failed: {err}"))?;
    let chat = std::env::var("MEMRA_CHAT").is_ok();
    let prompts: Vec<Vec<u32>> = (0..m)
        .map(|s| {
            let text = &texts[s % texts.len()];
            let to_encode = if chat {
                tok.apply_chat_template(&[("user", text.as_str())], true)
            } else {
                text.clone()
            };
            tok.encode(&to_encode, true)
        })
        .collect();
    let identical_prompts = prompts.windows(2).all(|w| w[0] == w[1]);
    println!(
        "prompts: {} distinct across m={m} streams (lens {:?}), n_new={n_new}",
        texts.len().min(m),
        prompts.iter().map(Vec::len).collect::<Vec<_>>()
    );

    // Residency: restore the freeze profile (mandatory here — a profiling warmup would need
    // its own generate pass; lockstep assumes the profile exists from a run-gen session).
    let freeze_profile = std::env::var("MEMRA_CPU_EXPERT_FREEZE_PROFILE")
        .ok()
        .filter(|v| !v.is_empty())
        .map(std::path::PathBuf::from)
        .ok_or("run-lockstep requires MEMRA_CPU_EXPERT_FREEZE_PROFILE (saved by run-gen)")?;
    if !model.restore_cpu_expert_residency_profile(&e, &freeze_profile)? {
        return Err("freeze profile did not restore (geometry mismatch or missing)".into());
    }
    e.stream().synchronize()?;

    // Prime each stream's cache over its own prompt (tokenwise, the frozen-serving path).
    let mut caches = Vec::with_capacity(m);
    let mut last_logits: Vec<Vec<f32>> = Vec::with_capacity(m);
    for prompt in &prompts {
        let max_ctx = prompt.len() + n_new + 8;
        let mut cache = memra_engine::cache::Cache::new(&e, &model.cfg, max_ctx)?;
        let mut dec = Vec::new();
        for &token in prompt {
            dec = model.decode_step(&e, token, &mut cache)?;
        }
        caches.push(cache);
        last_logits.push(dec);
    }
    e.stream().synchronize()?;

    let argmax = |v: &[f32]| -> u32 {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .unwrap_or(0)
    };
    let mut next: Vec<u32> = last_logits.iter().map(|l| argmax(l)).collect();
    let mut outputs: Vec<Vec<u32>> = (0..m).map(|_| Vec::with_capacity(n_new)).collect();

    let cpu_before = e.cpu_expert_stats();
    let t0 = std::time::Instant::now();
    for _ in 0..n_new {
        for (s, &t) in next.iter().enumerate() {
            outputs[s].push(t);
        }
        let logits = model.decode_step_lockstep(&e, &next, &mut caches)?;
        next = logits.iter().map(|l| argmax(l)).collect();
    }
    e.stream().synchronize()?;
    let dt = t0.elapsed().as_secs_f64();
    let total = m * n_new;
    println!(
        "lockstep m={m}: {total} tokens in {dt:.3}s = {:.2} tok/s aggregate, {:.2} tok/s per stream",
        total as f64 / dt,
        n_new as f64 / dt
    );
    if let (Some(before), Some(after)) = (cpu_before, e.cpu_expert_stats()) {
        println!(
            "CPU experts DECODE-WINDOW: calls={} experts={} backend_wall={:.3}s \
             RAM_hits={} RAM_misses={} RAM_fills={:.2} GB io={:.3}s compute={:.3}s",
            after.0.saturating_sub(before.0),
            after.1.saturating_sub(before.1),
            (after.2.saturating_sub(before.2)) as f64 / 1e9,
            after.3.saturating_sub(before.3),
            after.4.saturating_sub(before.4),
            (after.5.saturating_sub(before.5)) as f64 / 1e9,
            (after.8.saturating_sub(before.8)) as f64 / 1e9,
            (after.10.saturating_sub(before.10)) as f64 / 1e9,
        );
    }
    for (s, out) in outputs.iter().enumerate() {
        println!("stream {s}: {out:?}");
    }
    if identical_prompts {
        let identical = outputs.windows(2).all(|w| w[0] == w[1]);
        println!(
            "stream-identity gate: {}",
            if identical { "PASS (all streams identical)" } else { "FAIL" }
        );
        if !identical {
            std::process::exit(1);
        }
    } else {
        println!("stream-identity gate: SKIP (distinct prompts)");
    }
    Ok(())
}
