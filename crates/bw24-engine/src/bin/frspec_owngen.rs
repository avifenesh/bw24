//! FR-Spec OWN-GENERATION ranking builder: generate with the model itself over a prompt
//! set and rank the EMITTED token ids by frequency. This is the trim law in tool form —
//! rank files are vocab+distribution artifacts of the exact serving model; generic-corpus
//! tokenization ranks (frspec-rank) are the fallback, own-generation ranks are the real
//! thing (corpus rule: external text = prompts only, never the counted distribution).
//!
//! usage: frspec-owngen <model.gguf|hf_dir|hf:spec> <out.gguf> <topN> [flags] <prompts>...
//!   prompts: .txt file = one prompt; directory = recursed for .txt; .jsonl = one prompt
//!            per line (raw string or {"prompt": ...}); hfds:owner/name[:include-glob] =
//!            download a Hugging Face DATASET via the hf CLI, then recurse it.
//!   flags:   --ngen N   tokens generated per prompt (default 256)
//!            --temp T   sampling temperature (default 0 = greedy; serving-greedy class)
//!            --raw      skip the chat template (default wraps each prompt as a user turn)
//!
//! Output: <out.gguf> (1-tensor d2t i32 [topN]) + <out.gguf>.txt (one id/line, rank order)
//! — the artifacts BW24_FRSPEC_TRIM / BW24_MTP_DRAFT / BW24_GEMMA_DRAFT_RANKS consume.

use bw24_engine::Engine;
use bw24_engine::decode::GenParams;
use bw24_engine::hybrid::HybridModel;
use bw24_engine::sampler::{Sampler, SamplerConfig};
use bw24_gguf::GgufFile;
use bw24_tokenizer::Tokenizer;

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
                // {"prompt": "..."} or a bare JSON string or raw text line.
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!("usage: frspec-owngen <model.gguf|hf_dir|hf:spec> <out.gguf> <topN> [--ngen N] [--temp T] [--raw] <prompts>...");
        std::process::exit(1);
    }
    let model_arg = bw24_gguf::hf::resolve_arg(&args[1])?;
    let out_path = &args[2];
    let top_n: usize = args[3].parse()?;

    let mut ngen = 256usize;
    let mut temp = 0.0f32;
    let mut raw = false;
    let mut sources: Vec<String> = Vec::new();
    let mut i = 4;
    while i < args.len() {
        match args[i].as_str() {
            "--ngen" => { ngen = args[i + 1].parse()?; i += 2; }
            "--temp" => { temp = args[i + 1].parse()?; i += 2; }
            "--raw" => { raw = true; i += 1; }
            s => { sources.push(s.to_string()); i += 1; }
        }
    }

    // Gather prompts (hfds: specs download first).
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
        return Err("no prompts collected".into());
    }

    // Load model + tokenizer (GGUF file or safetensors dir — same pair run-gen serves).
    let e = Engine::new(0)?;
    let mp = std::path::Path::new(&model_arg);
    let (model, tok): (HybridModel, Tokenizer) = if mp.is_dir() {
        let src = bw24_gguf::source::SafetensorsSource::open(mp)?;
        (HybridModel::load_from_source(&e, &src)?,
         Tokenizer::from_hf_dir(mp).map_err(|err| format!("HF tokenizer: {err}"))?)
    } else {
        let g = GgufFile::open(&model_arg)?;
        let tok = Tokenizer::from_gguf(&g).map_err(|err| format!("tokenizer: {err}"))?;
        (HybridModel::load(&e, &g)?, tok)
    };
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

    let d2t = bw24_gguf::d2t::rank_top_n(&counts, top_n);
    let covered: u64 = d2t.iter().map(|&i| counts[i as usize]).sum();
    eprintln!("[frspec-owngen] top {} covers {:.2}% of {} own-generated tokens",
              d2t.len(), covered as f64 / total.max(1) as f64 * 100.0, total);
    bw24_gguf::d2t::write_d2t(out_path, &d2t)?;
    eprintln!("[frspec-owngen] wrote {out_path} + {out_path}.txt");
    Ok(())
}
