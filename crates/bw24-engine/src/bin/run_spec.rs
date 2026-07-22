//! MTP greedy speculative decode driver + self-consistency gate (research/mtp/MTP-PLAN.md §E).
//!
//! Greedy spec decode is mathematically EXACT: generate_spec(prompt, n, k) MUST produce a
//! token-IDENTICAL sequence to generate(prompt, n). This binary asserts that equality for
//! K=1..8 and prints the acceptance rate (n_accepted / drafted) + tok/s for each K vs
//! plain generate. A wrong MTP head still yields correct output via the bonus token but with
//! ~0 acceptance — so BOTH must hold: identical tokens AND acceptance > 0.
//!
//! Run: BW24_FAST=1 cargo run --release -p bw24-engine --bin run-spec -- <model.gguf> [tok ids...]

use bw24_engine::hybrid::HybridModel;
use bw24_engine::Engine;
use bw24_gguf::GgufFile;

// BW24_PROFILE_SPEC=1: bracket ONLY the generate_spec calls with cudaProfiler{Start,Stop} —
// with `nsys profile -c cudaProfilerApi` the capture then contains the spec phase alone
// (phase isolation by subtraction is unworkable on MoE: primes are not fungible, the first
// one cold-stages the expert cache — measured 2026-07-10).
unsafe extern "C" {
    fn cudaProfilerStart() -> i32;
    fn cudaProfilerStop() -> i32;
}

fn first_divergence(a: &[u32], b: &[u32]) -> Option<usize> {
    let n = a.len().min(b.len());
    for i in 0..n {
        if a[i] != b[i] {
            return Some(i);
        }
    }
    if a.len() != b.len() {
        return Some(n);
    }
    None
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1)
        .expect("usage: run-spec <model.gguf|hf_dir|hf:owner/repo[:file]> [tok ids...]");
    let path = bw24_gguf::hf::resolve_arg(&path)?;
    let e = Engine::new(0)?;
    // DIRECTORY path = safetensors HF checkpoint or manifest-backed bw24 repack/overlay; file = GGUF.
    let is_dir = std::path::Path::new(&path).is_dir();
    let g: Option<GgufFile> = if is_dir {
        None
    } else {
        Some(GgufFile::open(&path)?)
    };
    let mut source: Option<Box<dyn bw24_gguf::source::TensorSource>> = None;
    let mut tok_dir = std::path::PathBuf::from(&path);
    if is_dir {
        let dir = std::path::Path::new(&path);
        if dir.join("manifest.json").exists() {
            let repack = bw24_gguf::source::Hy3RepackSource::open(dir)?;
            tok_dir = repack
                .source_dir()
                .filter(|source| source.join("tokenizer.json").exists())
                .unwrap_or(dir)
                .to_path_buf();
            source = Some(Box::new(repack));
        } else {
            source = Some(Box::new(bw24_gguf::source::SafetensorsSource::open(dir)?));
        }
    }
    let model = match (&g, &source) {
        (Some(g), _) => HybridModel::load(&e, g)?,
        (None, Some(source)) => HybridModel::load_from_source(&e, source.as_ref())?,
        _ => unreachable!(),
    };
    println!(
        "loaded {} ({} layers, nextn={})",
        g.as_ref().and_then(|g| g.arch()).unwrap_or(
            if std::path::Path::new(&path).join("manifest.json").exists() {
                "bw24-repack"
            } else {
                "safetensors"
            }
        ),
        model.cfg.n_layer,
        model.cfg.nextn_predict_layers
    );
    if model.mtp.is_none() {
        eprintln!("ERROR: model has no MTP/NextN head (nextn_predict_layers={}, no blk.N.nextn.eh_proj). \
                   generate_spec is unavailable for this file.", model.cfg.nextn_predict_layers);
        std::process::exit(2);
    }

    // BW24_PROMPT_DIR=<dir>: BENCH-SWEEP mode — iterate every *.txt prompt in the dir with ONE
    // resident model (per-invocation model load dominates multi-prompt sweeps: a Spec-Bench /
    // MT-Bench pass is 80+ prompts). Per prompt: one plain generate (baseline) + one
    // generate_spec at BW24_SPEC_K (default 4); prints per-prompt and aggregate acceptance +
    // gen-only tok/s. Honors BW24_CHAT / BW24_NGEN / BW24_SPEC_TEMP / BW24_SEED like the
    // single-prompt path. No self-consistency gate here (bench mode, not the exactness battery).
    if let Ok(dir) = std::env::var("BW24_PROMPT_DIR") {
        let tok = match &g {
            Some(g) => bw24_tokenizer::Tokenizer::from_gguf(g)?,
            None => bw24_tokenizer::Tokenizer::from_hf_dir(&tok_dir)
                .map_err(|err| format!("HF tokenizer init failed: {err}"))?,
        };
        let n_new: usize = std::env::var("BW24_NGEN").ok().and_then(|s| s.parse().ok()).unwrap_or(256);
        let k: usize = std::env::var("BW24_SPEC_K").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
        let chat = std::env::var("BW24_CHAT").is_ok();
        let mut files: Vec<_> = std::fs::read_dir(&dir)
            .map_err(|e| format!("BW24_PROMPT_DIR {dir}: {e}"))?
            .filter_map(|d| d.ok().map(|d| d.path()))
            .filter(|p| p.extension().map(|x| x == "txt").unwrap_or(false)).collect();
        files.sort();
        if files.is_empty() {
            eprintln!("BW24_PROMPT_DIR {dir}: no *.txt prompts");
            std::process::exit(2);
        }
        let _ = model.generate(&e, &[55u32], 1)?;   // cold-start warmup
        let (mut tot_acc, mut tot_draft, mut sum_gen_t, mut sum_spec_t, mut tot_tok) =
            (0usize, 0usize, 0f64, 0f64, 0usize);
        for fp in &files {
            let text = std::fs::read_to_string(fp)?;
            let to_encode = if chat { tok.apply_chat_template(&[("user", &text)], true) }
                            else { text };
            let ids = tok.encode(&to_encode, true);
            e.stream().synchronize()?;
            let t0 = std::time::Instant::now();
            let _gold = model.generate(&e, &ids, n_new)?;
            e.stream().synchronize()?;
            let p1 = bw24_engine::PRIME_NANOS.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9;
            let gen_dt = (t0.elapsed().as_secs_f64() - p1).max(1e-9);
            let t1 = std::time::Instant::now();
            let (_spec, drafted, accepted) = model.generate_spec(&e, &ids, n_new, k)?;
            e.stream().synchronize()?;
            let p2 = bw24_engine::PRIME_NANOS.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9;
            let spec_dt = (t1.elapsed().as_secs_f64() - p2).max(1e-9);
            let stem = fp.file_stem().unwrap().to_string_lossy();
            println!("[{stem}] prompt {} tok | acc {accepted}/{drafted} = {:.1}% | gen {:.1} \
                      spec {:.1} tok/s ({:.2}x)",
                     ids.len(),
                     if drafted > 0 { accepted as f64 / drafted as f64 * 100.0 } else { 0.0 },
                     (n_new - 1) as f64 / gen_dt, (n_new - 1) as f64 / spec_dt, gen_dt / spec_dt);
            tot_acc += accepted; tot_draft += drafted;
            sum_gen_t += gen_dt; sum_spec_t += spec_dt; tot_tok += n_new - 1;
        }
        println!("\n[SWEEP] {} prompts K={k} | acceptance {tot_acc}/{tot_draft} = {:.1}% | \
                  gen {:.1} tok/s spec {:.1} tok/s ({:.2}x)",
                 files.len(),
                 if tot_draft > 0 { tot_acc as f64 / tot_draft as f64 * 100.0 } else { 0.0 },
                 tot_tok as f64 / sum_gen_t, tot_tok as f64 / sum_spec_t,
                 sum_gen_t / sum_spec_t);
        return Ok(());
    }

    // TEXT prompt path (BW24_PROMPT env): tokenize with the model's own tokenizer — REAL prompts
    // give the true acceptance rate (synthetic numeric sequences under-state it badly: the 27B
    // measured 45% on seq-101..228 vs ~75% on real code in the llama serve config).
    let prompt: Vec<u32> = if let Ok(text) = std::env::var("BW24_PROMPT_FILE")
        .map(|f| std::fs::read_to_string(&f).expect("BW24_PROMPT_FILE unreadable"))
        .or_else(|_| std::env::var("BW24_PROMPT"))
    {
        let tok = match &g {
            Some(g) => bw24_tokenizer::Tokenizer::from_gguf(g)?,
            None => bw24_tokenizer::Tokenizer::from_hf_dir(&tok_dir)
                .map_err(|err| format!("HF tokenizer init failed: {err}"))?,
        };
        // BW24_CHAT=1 wraps the prompt in the model's chat template (single user turn +
        // assistant generation prompt) — the run-gen contract. Raw continuation stays default.
        let to_encode = if std::env::var("BW24_CHAT").is_ok() {
            tok.apply_chat_template(&[("user", &text)], true)
        } else {
            text.clone()
        };
        let ids = tok.encode(&to_encode, true);
        println!(
            "text prompt ({} chars{}) -> {} tokens",
            text.len(),
            if to_encode.len() != text.len() {
                ", chat-templated"
            } else {
                ""
            },
            ids.len()
        );
        ids
    } else {
        let p: Vec<u32> = std::env::args()
            .skip(2)
            .filter_map(|s| s.parse().ok())
            .collect();
        if p.is_empty() {
            vec![55u32]
        } else {
            p
        }
    };
    println!("prompt tokens: {prompt:?}");

    let n_new = std::env::var("BW24_NGEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32usize);

    // --- ORACLE: plain greedy generate (the exact reference) ---
    // PRIME-SUBTRACT TIMING (2026-07-04): generate()/generate_spec() prime the prompt inside the
    // timed region, deflating tok/s ~2x on long prompts (the run_gen.rs known-bug class). Measure
    // a prime-only pass (max_new=1, minimal gen) and subtract its cost from every timed run so the
    // printed number is GEN-ONLY throughput, comparable to llama-bench tg / serve-script numbers.
    // BW24_GEN_ONLY=1: run ONLY the plain-generate oracle (no warmup, no spec Ks) and exit —
    // the prime-path A/B gate mode (compare `tokens:` between BW24_PRIME_TOKENWISE=1 and
    // batched-prime runs without paying 3 primes per invocation).
    let gen_only = std::env::var("BW24_GEN_ONLY").is_ok();
    let freeze_warmup_tokens = std::env::var("BW24_CPU_EXPERT_FREEZE_WARMUP_TOKENS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let freeze_profile = std::env::var("BW24_CPU_EXPERT_FREEZE_PROFILE")
        .ok()
        .filter(|value| !value.is_empty())
        .map(std::path::PathBuf::from);
    if freeze_warmup_tokens > 0 {
        let restored = match &freeze_profile {
            Some(path) => model.restore_cpu_expert_residency_profile(&e, path)?,
            None => false,
        };
        if !restored {
            if let Some(k) = std::env::var("BW24_CPU_EXPERT_FREEZE_WARMUP_SPEC_K")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
            {
                println!(
                    "[moe-cache] warming {freeze_warmup_tokens} discarded speculative tokens at K={k} before fixed residency"
                );
                let _ = model.generate_spec(&e, &prompt, freeze_warmup_tokens + 1, k)?;
            } else {
                println!(
                    "[moe-cache] warming {freeze_warmup_tokens} discarded decode tokens before fixed residency"
                );
                let _ = model.generate(&e, &prompt, freeze_warmup_tokens + 1)?;
            }
            e.stream().synchronize()?;
            model.freeze_cpu_expert_residency(&e)?;
            if let Some(path) = &freeze_profile {
                model.save_cpu_expert_residency_profile(&e, path)?;
            }
        }
    } else if !gen_only {
        let _ = model.generate(&e, &prompt, 1)?; // cold-start warmup (weights/L2/allocator)
    }
    if freeze_warmup_tokens > 0
        && std::env::var("BW24_MOE_PREFETCH").is_ok_and(|value| value != "0")
    {
        model.start_moe_prefetch_predictor(&e, &model.cfg)?;
    }
    e.stream().synchronize()?;
    let t0 = std::time::Instant::now();
    let gold = model.generate(&e, &prompt, n_new)?;
    e.stream().synchronize()?;
    // Gen-only via the in-API prime timer (crate::PRIME_NANOS): the old subtract-a-separate-
    // prime-run approach amplified prime jitter into the gen number (±80% at 6k-token prompts).
    let prime_dt = bw24_engine::PRIME_NANOS.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9;
    let gen_dt = (t0.elapsed().as_secs_f64() - prime_dt).max(1e-9);
    let gen_tps = (n_new - 1) as f64 / gen_dt;
    println!("\n[generate]   {} tok in {gen_dt:.3}s = {gen_tps:.2} tok/s (gen-only; this run's prime {prime_dt:.3}s)", n_new - 1);
    println!("  tokens: {gold:?}");
    // BW24_PRINT_TEXT=1: decode the greedy output between stable markers (agent-loop harnesses
    // append it to a growing transcript; ids alone are not transcript-usable).
    if std::env::var("BW24_PRINT_TEXT").as_deref() == Ok("1") {
        let tok = match &g {
            Some(g) => bw24_tokenizer::Tokenizer::from_gguf(g)?,
            None => bw24_tokenizer::Tokenizer::from_hf_dir(std::path::Path::new(&path))
                .map_err(|err| format!("HF tokenizer init failed: {err}"))?,
        };
        println!("--- generated text ---\n{}\n--- end ---", tok.decode(&gold));
    }
    if gen_only {
        return Ok(());
    }

    let mut all_pass = true;
    // BW24_SPEC_K=<k>: run ONLY one K (e2e bench mode — the full sweep is the gate battery).
    let ks: Vec<usize> = match std::env::var("BW24_SPEC_K")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        Some(k) => vec![k],
        None => (1..=8).collect(),
    };
    for &k in &ks {
        e.stream().synchronize()?;
        let prof = std::env::var("BW24_PROFILE_SPEC").as_deref() == Ok("1");
        let t1 = std::time::Instant::now();
        if prof {
            unsafe {
                cudaProfilerStart();
            }
        }
        let (spec, drafted, accepted) = model.generate_spec(&e, &prompt, n_new, k)?;
        e.stream().synchronize()?;
        if prof {
            unsafe {
                cudaProfilerStop();
            }
        }
        let spec_prime =
            bw24_engine::PRIME_NANOS.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9;
        let spec_dt = (t1.elapsed().as_secs_f64() - spec_prime).max(1e-9);
        let spec_tps = (n_new - 1) as f64 / spec_dt;
        let acc_rate = if drafted > 0 {
            accepted as f64 / drafted as f64
        } else {
            0.0
        };

        // SAMPLED MODE (BW24_SPEC_TEMP>0): the greedy-identity gate is undefined (spec and
        // plain sampling consume randomness differently — Leviathan/Chen guarantee equality of
        // DISTRIBUTION, not streams). Its gate = seeded REPRODUCIBILITY: the same
        // (seed, prompt, K) must reproduce the identical token stream on a second run.
        let sampled = std::env::var("BW24_SPEC_TEMP")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .map(|t| t > 0.0)
            .unwrap_or(false);
        let pass = if sampled {
            let (spec2, _, _) = model.generate_spec(&e, &prompt, n_new, k)?;
            e.stream().synchronize()?;
            first_divergence(&spec, &spec2).is_none()
        } else {
            first_divergence(&gold, &spec).is_none()
        };
        all_pass &= pass;
        println!(
            "\n[generate_spec K={k}] {n_new} tok in {spec_dt:.3}s = {spec_tps:.2} tok/s \
                  ({:.2}x vs generate; this run's prime {spec_prime:.3}s)",
            spec_tps / gen_tps
        );
        println!(
            "  acceptance: {accepted}/{drafted} = {:.1}%   self-consistency: {}",
            acc_rate * 100.0,
            if pass {
                if sampled {
                    "PASS (seeded rerun identical)"
                } else {
                    "PASS (identical to generate)"
                }
            } else {
                "FAIL"
            }
        );
        // Sampled stream printed on PASS too: the graph-vs-eager identity gate (same seed,
        // BW24_SPEC_NOGRAPH=1 vs default) diffs these token ids across separate invocations.
        if sampled {
            println!("  sampled tokens: {spec:?}");
        }
        if !pass {
            let idx = first_divergence(&gold, &spec).unwrap();
            println!("  FIRST DIVERGENCE at index {idx}:");
            println!(
                "    generate     [..]: {:?}",
                &gold[idx.saturating_sub(2)..(idx + 3).min(gold.len())]
            );
            println!(
                "    generate_spec[..]: {:?}",
                &spec[idx.saturating_sub(2)..(idx + 3).min(spec.len())]
            );
            println!("    spec tokens: {spec:?}");
        }
        if sampled && std::env::var("BW24_PRINT_TEXT").as_deref() == Ok("1") {
            let tok = match &g {
                Some(g) => bw24_tokenizer::Tokenizer::from_gguf(g)?,
                None => bw24_tokenizer::Tokenizer::from_hf_dir(std::path::Path::new(&path))
                    .map_err(|err| format!("HF tokenizer init failed: {err}"))?,
            };
            println!(
                "--- sampled text (K={k}) ---\n{}\n--- end ---",
                tok.decode(&spec)
            );
        }
        // acceptance>0 is the SECOND gate (a wrong head passes self-consistency via the bonus token
        // but accepts nothing). Report it; the assert below covers the exactness gate.
        if pass && accepted == 0 {
            println!(
                "  WARNING: acceptance == 0 with identical output — MTP head is likely \
                      forwarded wrong (bonus-token masking). Speedup will be absent."
            );
        }
    }

    println!(
        "\n=== SELF-CONSISTENCY {} ===",
        if all_pass { "PASS" } else { "FAIL" }
    );
    assert!(
        all_pass,
        "generate_spec diverged from generate — greedy spec decode must be exact"
    );
    Ok(())
}
