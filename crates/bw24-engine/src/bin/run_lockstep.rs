//! Lane-3 M1 harness: lockstep multi-stream decode over one Hy3 checkpoint.
//!
//! `run-lockstep <hf_or_repack_dir>` with BW24_LOCKSTEP_M streams (default 2), BW24_NGEN
//! tokens per stream (default 32), BW24_PROMPT/BW24_CHAT as in run-gen. Every stream serves
//! the same prompt, so the correctness gate is internal: all streams must emit identical
//! token sequences (each stream's math is decode_step_h's, so this also matches the
//! single-stream run). Prints per-stream tokens, aggregate and per-stream throughput, and
//! the decode-window CPU expert counters that show the cross-stream io amortization.

use bw24_engine::hybrid::HybridModel;
use bw24_engine::Engine;
use bw24_tokenizer::Tokenizer;

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
    let (src, tok_dir): (Box<dyn bw24_gguf::source::TensorSource>, std::path::PathBuf) =
        if is_repack {
            let rs = bw24_gguf::source::Hy3RepackSource::open(dir)?;
            let td = rs
                .source_dir()
                .filter(|d| d.join("tokenizer.json").exists())
                .unwrap_or(dir)
                .to_path_buf();
            (Box::new(rs), td)
        } else {
            (
                Box::new(bw24_gguf::source::SafetensorsSource::open(dir)?),
                dir.to_path_buf(),
            )
        };
    let model = HybridModel::load_from_source_without_mtp(&e, src.as_ref())?;
    println!("loaded {:?} ({} trunk layers)", model.cfg.arch, model.layers.len());

    let m: usize = std::env::var("BW24_LOCKSTEP_M")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&m| (1..=16).contains(&m))
        .unwrap_or(2);
    let n_new: usize = std::env::var("BW24_NGEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);

    let text = std::env::var("BW24_PROMPT")
        .unwrap_or_else(|_| "Explain speculative decoding briefly.".to_string());
    let tok = Tokenizer::from_hf_dir(&tok_dir)
        .map_err(|err| format!("HF tokenizer init failed: {err}"))?;
    let to_encode = if std::env::var("BW24_CHAT").is_ok() {
        tok.apply_chat_template(&[("user", text.as_str())], true)
    } else {
        text.clone()
    };
    let prompt = tok.encode(&to_encode, true);
    println!("prompt tokens: {} (m={m} streams, n_new={n_new})", prompt.len());

    // Residency: restore the freeze profile (mandatory here — a profiling warmup would need
    // its own generate pass; lockstep assumes the profile exists from a run-gen session).
    let freeze_profile = std::env::var("BW24_CPU_EXPERT_FREEZE_PROFILE")
        .ok()
        .filter(|v| !v.is_empty())
        .map(std::path::PathBuf::from)
        .ok_or("run-lockstep requires BW24_CPU_EXPERT_FREEZE_PROFILE (saved by run-gen)")?;
    if !model.restore_cpu_expert_residency_profile(&e, &freeze_profile)? {
        return Err("freeze profile did not restore (geometry mismatch or missing)".into());
    }
    e.stream().synchronize()?;

    // Prime each stream's cache over the shared prompt (tokenwise, the frozen-serving path).
    let max_ctx = prompt.len() + n_new + 8;
    let mut caches = Vec::with_capacity(m);
    let mut last_logits: Vec<Vec<f32>> = Vec::with_capacity(m);
    for _ in 0..m {
        let mut cache = bw24_engine::cache::Cache::new(&e, &model.cfg, max_ctx)?;
        let mut dec = Vec::new();
        for &token in &prompt {
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
    for (s, out) in outputs.iter().enumerate() {
        println!("stream {s}: {out:?}");
    }
    let identical = outputs.windows(2).all(|w| w[0] == w[1]);
    println!(
        "stream-identity gate: {}",
        if identical { "PASS (all streams identical)" } else { "FAIL" }
    );
    if !identical {
        std::process::exit(1);
    }
    Ok(())
}
