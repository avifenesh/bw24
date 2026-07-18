//! FR-Spec trimmed-head builder — the trim law as a user-facing CLI.
//!
//! Rank files are vocab+distribution artifacts of the EXACT serving model: the tool
//! generates WITH the model over a prompt set (external text is prompts only, never the
//! counted distribution), ranks the emitted token ids, writes the d2t artifact every trim
//! consumer takes, and (--validate) measures speculative acceptance with and without the
//! trim so the artifact ships with evidence instead of hope.
//!
//! usage: frspec-owngen <model.gguf|hf_dir|hf:spec> <out.gguf> [topN] [flags] [prompts...]
//!   topN     draft vocab size (default 32768)
//!   prompts  .txt/.md file = one prompt; directory = recursed; .jsonl = one per line
//!            (raw string or {"prompt": ...}); hfds:owner/name[:include-glob] = download
//!            a Hugging Face DATASET via the hf CLI (text/jsonl files; parquet-only
//!            datasets need a local export first). No prompts => the built-in mixed pack.
//!   flags:   --preset code|chat|agentic|mixed   built-in prompt subset (default mixed)
//!            --ngen N     tokens generated per prompt (default 256)
//!            --temp T     sampling temperature (default 0 = greedy, the serving class)
//!            --raw        skip the chat template
//!            --validate   reload with the trim applied and A/B spec acceptance (K=3)
//!
//! Output: <out.gguf> (d2t i32 [topN]) + <out.gguf>.txt (one id/line, rank order).
//! Serve with:  BW24_FRSPEC_TRIM=<out.gguf> ...        (qwen MTP heads)
//!              BW24_GEMMA_DRAFT_RANKS=<out.gguf>.txt  (gemma drafters)

use bw24_engine::Engine;
use bw24_engine::decode::GenParams;
use bw24_engine::hybrid::HybridModel;
use bw24_engine::sampler::{Sampler, SamplerConfig};
use bw24_gguf::GgufFile;
use bw24_tokenizer::Tokenizer;

// ---- built-in prompt pack (mixed serving distribution; each &str = one prompt) ----
const PROMPTS_CODE: &[&str] = &[
    "Write a Python function that parses an ISO-8601 timestamp string and returns the number of seconds since the Unix epoch, handling timezone offsets correctly. Include error handling and a few unit tests.",
    "Refactor this into idiomatic Rust with proper error handling:\n\nfn read_config(path: &str) -> Config {\n    let s = std::fs::read_to_string(path).unwrap();\n    let c: Config = serde_json::from_str(&s).unwrap();\n    return c;\n}\n\nExplain each change briefly.",
    "Implement a thread-safe LRU cache in C++ with get/put in O(1). Use std::list and std::unordered_map, guard with a mutex, and explain the iterator-invalidation pitfall.",
];
const PROMPTS_CHAT: &[&str] = &[
    "Explain the difference between TCP and UDP to someone who knows basic networking, with one concrete example each of when you'd pick one over the other.",
    "I have 500g of chicken thighs, rice, onions, and soy sauce. Suggest a simple dinner I can cook in 30 minutes, with steps.",
];
const PROMPTS_AGENTIC: &[&str] = &[
    "You are a coding agent in a git repository. The test suite fails with: `AssertionError: expected 200, got 404` in tests/api/test_users.py::test_get_user. List the steps you would take to diagnose and fix this, naming the exact commands you would run.",
    "Plan a database migration that renames a column in a table with 50M rows on PostgreSQL without downtime. Produce a numbered runbook with rollback points.",
];
const PROMPTS_REASONING: &[&str] = &[
    "A train leaves city A at 9:00 traveling 80 km/h. Another leaves city B (240 km away) at 9:30 traveling toward A at 100 km/h. At what time do they meet? Show the algebra step by step.",
    "Three friends split a restaurant bill of 87.50 with a 15% tip on the pre-tip amount. One had a dish 12 more expensive than each of the others' equal dishes. How much does each pay if they split fairly? Work it out.",
];
const PROMPTS_WRITING: &[&str] = &[
    "Write a 150-word product-release note for a CLI tool that speeds up model inference by 40%, aimed at developers, with one code example.",
];
// Held-out eval prompts for --validate (acceptance measurement, never counted in ranks).
const PROMPTS_EVAL: &[&str] = &[
    "Write a bash script that finds the 10 largest files under a directory and prints them with human-readable sizes.",
    "Explain what a KV cache is in transformer inference and why it makes generation faster.",
    "Given a sorted array with duplicates, write a binary search returning the FIRST index of the target, in Python, with tests.",
];

fn collect_prompts(path: &std::path::Path, out: &mut Vec<String>) {
    if path.is_dir() {
        if let Ok(rd) = std::fs::read_dir(path) {
            let mut entries: Vec<_> = rd.flatten().map(|e| e.path()).collect();
            entries.sort();
            for p in entries {
                collect_prompts(&p, out);
            }
        }
        return;
    }
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let Ok(text) = std::fs::read_to_string(path) else { return };
    match ext {
        "jsonl" => {
            for line in text.lines().filter(|l| !l.trim().is_empty()) {
                let p = line.trim();
                let prompt = if let Some(i) = p.find("\"prompt\"") {
                    p[i + 8..].trim_start_matches([':', ' ']).trim_matches('"').to_string()
                } else {
                    p.trim_matches('"').to_string()
                };
                if !prompt.is_empty() {
                    out.push(prompt);
                }
            }
        }
        "txt" | "md" => out.push(text),
        _ => {}
    }
}

fn load_model(e: &Engine, path: &str) -> Result<(HybridModel, Tokenizer), Box<dyn std::error::Error>> {
    let mp = std::path::Path::new(path);
    if mp.is_dir() {
        let src = bw24_gguf::source::SafetensorsSource::open(mp)?;
        Ok((HybridModel::load_from_source(e, &src)?,
            Tokenizer::from_hf_dir(mp).map_err(|err| format!("HF tokenizer: {err}"))?))
    } else {
        let g = GgufFile::open(path)?;
        let tok = Tokenizer::from_gguf(&g).map_err(|err| format!("tokenizer: {err}"))?;
        Ok((HybridModel::load(e, &g)?, tok))
    }
}

/// K=3 greedy spec over the held-out eval prompts: returns (acceptance, tok_s) means.
fn spec_eval(e: &Engine, model: &HybridModel, tok: &Tokenizer)
             -> Result<(f64, f64), Box<dyn std::error::Error>> {
    let (mut acc_n, mut acc_d, mut toks, mut secs) = (0usize, 0usize, 0usize, 0f64);
    for p in PROMPTS_EVAL {
        let ids = tok.encode(&tok.apply_chat_template(&[("user", p)], true), true);
        let t0 = std::time::Instant::now();
        let (out, drafted, accepted) = model.generate_spec(e, &ids, 64, 3)?;
        secs += t0.elapsed().as_secs_f64();
        toks += out.len();
        acc_n += accepted;
        acc_d += drafted;
    }
    Ok((acc_n as f64 / acc_d.max(1) as f64, toks as f64 / secs.max(1e-9)))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: frspec-owngen <model.gguf|hf_dir|hf:spec> <out.gguf> [topN] \
                   [--preset code|chat|agentic|mixed] [--ngen N] [--temp T] [--raw] [--validate] [prompts...]");
        std::process::exit(1);
    }
    let model_arg = bw24_gguf::hf::resolve_arg(&args[1])?;
    let out_path = args[2].clone();

    let (mut top_n, mut ngen, mut temp) = (32768usize, 256usize, 0.0f32);
    let (mut raw, mut validate) = (false, false);
    let mut preset = "mixed".to_string();
    let mut sources: Vec<String> = Vec::new();
    let mut i = 3;
    if i < args.len() && let Ok(n) = args[3].parse::<usize>() {
        top_n = n;
        i += 1;
    }
    while i < args.len() {
        match args[i].as_str() {
            "--ngen" => { ngen = args[i + 1].parse()?; i += 2; }
            "--temp" => { temp = args[i + 1].parse()?; i += 2; }
            "--preset" => { preset = args[i + 1].clone(); i += 2; }
            "--raw" => { raw = true; i += 1; }
            "--validate" => { validate = true; i += 1; }
            s => { sources.push(s.to_string()); i += 1; }
        }
    }

    // Gather prompts: explicit sources win; otherwise the built-in pack (preset subset).
    let mut prompts: Vec<String> = Vec::new();
    for src in &sources {
        if let Some(spec) = src.strip_prefix("hfds:") {
            let (repo, glob) = match spec.split_once(':') {
                Some((r, g)) => (r, Some(g)),
                None => (spec, None),
            };
            let root = std::env::var("BW24_MODELS_DIR").unwrap_or_else(|_| {
                format!("{}/.cache/bw24/models", std::env::var("HOME").unwrap_or_default())
            });
            let dest = format!("{root}/hf-datasets/{}", repo.replace('/', "--"));
            if !std::path::Path::new(&dest).is_dir() {
                eprintln!("[frspec-owngen] downloading dataset {repo} -> {dest}");
                let mut cmd = std::process::Command::new("hf");
                cmd.args(["download", repo, "--repo-type", "dataset", "--local-dir", &dest]);
                if let Some(g) = glob {
                    cmd.arg("--include").arg(format!("*{g}*"));
                }
                let st = cmd.status().map_err(|e| format!("hf CLI not runnable: {e}"))?;
                if !st.success() {
                    return Err(format!("hf dataset download {repo} failed ({st})").into());
                }
            }
            collect_prompts(std::path::Path::new(&dest), &mut prompts);
        } else {
            collect_prompts(std::path::Path::new(src), &mut prompts);
        }
    }
    if prompts.is_empty() {
        let packs: &[&[&str]] = match preset.as_str() {
            "code" => &[PROMPTS_CODE],
            "chat" => &[PROMPTS_CHAT],
            "agentic" => &[PROMPTS_AGENTIC],
            _ => &[PROMPTS_CODE, PROMPTS_CHAT, PROMPTS_AGENTIC, PROMPTS_REASONING, PROMPTS_WRITING],
        };
        for pack in packs {
            prompts.extend(pack.iter().map(|s| s.to_string()));
        }
        eprintln!("[frspec-owngen] no prompt sources given — built-in '{preset}' pack ({} prompts)",
                  prompts.len());
    }

    // ---- phase 1: own-generation corpus + baseline spec eval on the UNTRIMMED head ----
    let e = Engine::new(0)?;
    let (model, tok) = load_model(&e, &model_arg)?;
    let vocab = tok.vocab_size();
    eprintln!("[frspec-owngen] {} prompts, ngen {}, temp {}, vocab {}",
              prompts.len(), ngen, temp, vocab);

    let mut counts = vec![0u64; vocab];
    let mut total = 0u64;
    let params = GenParams { max_new: ngen, max_ctx: None, eos: vec![tok.eos_id()] };
    for (pi, prompt) in prompts.iter().enumerate() {
        let text = if raw { prompt.clone() } else { tok.apply_chat_template(&[("user", prompt)], true) };
        let ids = tok.encode(&text, true);
        let mut sampler = Sampler::new(SamplerConfig {
            temperature: temp, seed: 42 + pi as u64, ..Default::default()
        });
        let out = model.generate_with(&e, &ids, &params, &mut sampler, |_| true)?;
        for &id in &out.tokens {
            if (id as usize) < vocab {
                counts[id as usize] += 1;
                total += 1;
            }
        }
        eprintln!("[frspec-owngen] prompt {}/{}: +{} tokens (total {})",
                  pi + 1, prompts.len(), out.tokens.len(), total);
    }

    let baseline = if validate && model.mtp.is_some() {
        let b = spec_eval(&e, &model, &tok)?;
        eprintln!("[frspec-owngen] baseline (untrimmed): acceptance {:.1}% | {:.1} tok/s",
                  b.0 * 100.0, b.1);
        Some(b)
    } else {
        if validate {
            eprintln!("[frspec-owngen] --validate skipped: model carries no MTP head \
                       (gemma drafters validate through their own gates)");
        }
        None
    };
    drop(model);

    // ---- rank + write the artifact ----
    let d2t = bw24_gguf::d2t::rank_top_n(&counts, top_n);
    let covered: u64 = d2t.iter().map(|&i| counts[i as usize]).sum();
    let distinct = counts.iter().filter(|&&c| c > 0).count();
    eprintln!("[frspec-owngen] top {} covers {:.2}% of {} own-generated tokens ({} distinct)",
              d2t.len(), covered as f64 / total.max(1) as f64 * 100.0, total, distinct);
    if (total as usize) < 4 * top_n {
        eprintln!("[frspec-owngen] WARNING: {total} tokens is a SMALL corpus for topN {top_n} — \
                   ranks past the distribution head are noise (measured -6.5 acceptance pts at \
                   2.5k tokens). Aim for >= {} generated tokens: more prompts, --ngen 512+, or \
                   a dataset source.", 4 * top_n);
    }
    bw24_gguf::d2t::write_d2t(&out_path, &d2t)?;
    eprintln!("[frspec-owngen] wrote {out_path} + {out_path}.txt");

    // ---- phase 2 (--validate): reload with the trim applied, A/B acceptance ----
    if let Some((base_acc, base_tps)) = baseline {
        // Single-threaded CLI; the var must be set before the reload reads it.
        unsafe { std::env::set_var("BW24_FRSPEC_TRIM", &out_path) };
        let (model2, _) = load_model(&e, &model_arg)?;
        let (trim_acc, trim_tps) = spec_eval(&e, &model2, &tok)?;
        eprintln!("[frspec-owngen] trimmed:              acceptance {:.1}% | {:.1} tok/s",
                  trim_acc * 100.0, trim_tps);
        // Verdict metric is E2E TOK/S (owner law 2026-07-17); acceptance is the diagnostic
        // for WHY, never the decision basis.
        let d_tps = trim_tps / base_tps - 1.0;
        let d_acc = trim_acc - base_acc;
        let verdict = if d_tps >= 0.0 { "GOOD — trim wins e2e" }
                      else if d_tps >= -0.02 { "WASH — no e2e gain; keep only if the VRAM saving matters" }
                      else { "BAD — trim loses e2e on this model; widen topN or regenerate ranks" };
        eprintln!("[frspec-owngen] verdict: {verdict} (e2e {:+.1}% | diagnostic: Δacceptance {:+.1} pts)",
                  d_tps * 100.0, d_acc * 100.0);
    }

    println!("serve with:");
    println!("  BW24_FRSPEC_TRIM={out_path} ...            # qwen MTP heads");
    println!("  BW24_GEMMA_DRAFT_RANKS={out_path}.txt ...  # gemma drafters");
    Ok(())
}
