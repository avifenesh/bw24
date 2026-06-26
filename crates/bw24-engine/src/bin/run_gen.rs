//! M7: end-to-end greedy generation with KV cache. Serves a model: prompt tokens -> generated tokens.
//!
//! Two prompt paths (back-compat):
//!   1. raw token ids:  `run-gen <model.gguf> 9419 11 1814 0`   (validation-gate path)
//!   2. TEXT prompt:    `run-gen <model.gguf> --prompt "Hello, world!"`  (or env BW24_PROMPT)
//!      The text is tokenized with bw24-tokenizer, generated, then the output ids are
//!      DETOKENIZED back to text and printed. Set BW24_CHAT=1 to wrap the prompt in the
//!      model's chat template (single user turn + assistant generation prompt).

use bw24_engine::Engine;
use bw24_engine::forward::argmax;
use bw24_engine::hybrid::HybridModel;
use bw24_gguf::GgufFile;
use bw24_tokenizer::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: run-gen <model.gguf> [tok ids...] | --prompt \"text\"");
    let e = Engine::new(0)?;
    let g = GgufFile::open(&path)?;
    let model = HybridModel::load(&e, &g)?;
    println!("loaded {} ({} layers)", g.arch().unwrap_or("?"), model.cfg.n_layer);

    // --- resolve the prompt: TEXT path (--prompt / BW24_PROMPT) vs raw-u32 path ---
    let args: Vec<String> = std::env::args().skip(2).collect();
    let arg_prompt: Option<String> = args
        .iter()
        .position(|a| a == "--prompt")
        .and_then(|i| args.get(i + 1).cloned());
    let prompt_text: Option<String> = arg_prompt.or_else(|| std::env::var("BW24_PROMPT").ok());

    // Lazily build the tokenizer only when we need text I/O (it parses the 248K vocab).
    let mut tokenizer: Option<Tokenizer> = None;

    let prompt: Vec<u32> = if let Some(text) = &prompt_text {
        let tok = Tokenizer::from_gguf(&g)
            .map_err(|err| format!("tokenizer init failed: {err}"))?;
        // Optional chat-template wrapping (single user turn).
        let to_encode = if std::env::var("BW24_CHAT").is_ok() {
            let rendered = tok.apply_chat_template(&[("user", text)], true);
            println!("chat-templated prompt:\n{rendered}");
            rendered
        } else {
            text.clone()
        };
        let ids = tok.encode(&to_encode, true);
        println!("prompt text: {text:?}");
        tokenizer = Some(tok);
        ids
    } else {
        // raw u32 ids off the CLI (skip the "--prompt"/value tokens if present)
        args.iter().filter_map(|s| s.parse::<u32>().ok()).collect()
    };
    let prompt = if prompt.is_empty() { vec![55u32] } else { prompt };
    println!("prompt tokens: {prompt:?}");

    // --- correctness gate: decode-step prefix logits MUST match the prefill forward ---
    let prefill = model.forward_last(&e, &prompt)?;
    // decode the prompt step by step, capture last logits
    let mut cache = bw24_engine::cache::Cache::new(&e, &model.cfg, prompt.len() + 64)?;
    let mut dec_logits = Vec::new();
    for &t in &prompt { dec_logits = model.decode_step(&e, t, &mut cache)?; }
    let am_p = argmax(&prefill); let am_d = argmax(&dec_logits);
    let maxdiff = prefill.iter().zip(&dec_logits).map(|(a,b)| (a-b).abs()).fold(0.0, f32::max);
    println!("prefill argmax={am_p}  decode argmax={am_d}  logit maxdiff={maxdiff:.3e}  {}",
             if am_p == am_d { "MATCH" } else { "MISMATCH" });
    assert_eq!(am_p, am_d, "decode-step diverges from prefill — cache threading bug");

    // --- time PREFILL tok/s (batched forward over the whole prompt) for the pp comparison vs
    //     llama-bench pp512. 1 warmup discarded, then time one forward of the full prompt. ---
    if prompt.len() >= 8 {
        let _ = model.forward_last(&e, &prompt)?;   // warmup
        e.stream().synchronize()?;
        let tp = std::time::Instant::now();
        let _ = model.forward_last(&e, &prompt)?;
        e.stream().synchronize()?;
        let dtp = tp.elapsed().as_secs_f64();
        println!("prefill {} tok in {:.4}s = {:.1} tok/s (pp{})", prompt.len(), dtp, prompt.len() as f64 / dtp, prompt.len());
    }

    // --- generate + time decode tok/s (honest Stage-A baseline) ---
    let n_new = std::env::var("BW24_NGEN").ok().and_then(|s| s.parse().ok()).unwrap_or(16usize);
    let eos = tokenizer.as_ref().map(|t| t.eos_id());
    let mut cache = bw24_engine::cache::Cache::new(&e, &model.cfg, prompt.len() + n_new + 8)?;
    let mut ll = Vec::new();
    for &t in &prompt { ll = model.decode_step(&e, t, &mut cache)?; }
    e.stream().synchronize()?;
    let t0 = std::time::Instant::now();
    let mut out = Vec::with_capacity(n_new);
    let mut emitted = 0usize;
    for _ in 0..n_new {
        let next = argmax(&ll) as u32;
        out.push(next);
        emitted += 1;
        // EOS stop (only when we know the eos id, i.e. the text path).
        if Some(next) == eos {
            break;
        }
        ll = model.decode_step(&e, next, &mut cache)?;
    }
    e.stream().synchronize()?;
    let dt = t0.elapsed().as_secs_f64();
    let path = if std::env::var("BW24_FAST").is_ok() { "Stage-B int8 dp4a" } else { "Stage-A f32-dequant" };
    println!("generated {} tokens in {:.3}s = {:.2} tok/s ({path} decode)", emitted, dt, emitted as f64 / dt);
    println!("tokens: {out:?}");

    // --- detokenize the output ids back to TEXT (text path only) ---
    if let Some(tok) = &tokenizer {
        // drop a trailing EOS for the printed text (keep it in the raw `tokens:` line above).
        let text_ids: Vec<u32> = out
            .iter()
            .copied()
            .filter(|&id| Some(id) != eos)
            .collect();
        let text = tok.decode(&text_ids);
        println!("OUTPUT TEXT: {text:?}");
        println!("--- generated text ---\n{text}");
    }
    Ok(())
}
