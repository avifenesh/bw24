//! ST serve-spec divergence probe (#68, fp8-ship lane 2026-08-04): mimic the WORKER's
//! generate_spec_session burst pattern on a dir-loaded checkpoint OUTSIDE the server —
//! turn 1 primes the chat-templated prompt, later calls are empty-suffix continuation
//! bursts of MEMRA_SPEC_BURST tokens — and compare against plain greedy generate().
//! Isolates engine-side burst-boundary state from worker-environment suspects.
//!
//! usage: spec-st-probe <ckpt dir|gguf> (env: MEMRA_SPEC_BURST=32 MEMRA_SPEC_K=3
//!        MEMRA_NGEN=400 MEMRA_PROMPT=..., MEMRA_SPEC_NOGRAPH honored by the engine)
use memra_engine::Engine;
use memra_engine::hybrid::HybridModel;
use memra_engine::memra_gguf::GgufFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: spec-st-probe <ckpt>");
    let e = Engine::new(0)?;
    let is_dir = std::path::Path::new(&path).is_dir();
    let g: Option<GgufFile> = if is_dir { None } else { Some(GgufFile::open(&path)?) };
    let src: Option<memra_engine::memra_gguf::source::SafetensorsSource> = if is_dir {
        Some(memra_engine::memra_gguf::source::SafetensorsSource::open(
            std::path::Path::new(&path),
        )?)
    } else {
        None
    };
    let model = match (&g, &src) {
        (Some(g), _) => HybridModel::load(&e, g)?,
        (None, Some(s)) => HybridModel::load_from_source(&e, s)?,
        _ => unreachable!(),
    };
    let tok = match &g {
        Some(g) => memra_tokenizer::Tokenizer::from_gguf(g)?,
        None => memra_tokenizer::Tokenizer::from_hf_dir(std::path::Path::new(&path))?,
    };
    let prompt_text = std::env::var("MEMRA_PROMPT")
        .unwrap_or_else(|_| "What is the capital of France? Answer in one short sentence.".into());
    let rendered = tok.apply_chat_template(&[("user", &prompt_text)], true);
    let ids = tok.encode(&rendered, true);
    let n_new: usize = std::env::var("MEMRA_NGEN").ok().and_then(|v| v.parse().ok()).unwrap_or(400);
    let burst: usize = std::env::var("MEMRA_SPEC_BURST").ok().and_then(|v| v.parse().ok()).unwrap_or(32);
    let k: usize = std::env::var("MEMRA_SPEC_K").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    println!("prompt {} tok, n_new {n_new}, burst {burst}, k {k}", ids.len());

    // oracle: plain greedy
    let reference = model.generate(&e, &ids, n_new)?;

    // worker pattern: session bursts (turn 1 = prompt suffix, then empty-suffix continuations)
    // MEMRA_PROBE_CTX: session cache capacity (the worker allocates at the MEMRA_CTX floor
    // 8192, not prompt+gen — scratch.cap feeds fa_decode_dc's n_splits geometry).
    let ctx_cap: usize = std::env::var("MEMRA_PROBE_CTX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(ids.len() + n_new + 64);
    let mut sess = model.new_session(&e, ctx_cap)?;
    let mut out: Vec<u32> = Vec::new();
    let mut first = true;
    let mut nb = 0usize;
    while out.len() < n_new {
        let room = (n_new - out.len()).min(burst);
        let suffix: Vec<u32> = if first { ids.clone() } else { Vec::new() };
        first = false;
        nb += 1;
        let (b, d, a) = model.generate_spec_session(&e, &mut sess, &suffix, room, k)?;
        // per-burst divergence localization against the reference prefix
        let start = out.len();
        let div = b
            .iter()
            .enumerate()
            .position(|(j, t)| reference.get(start + j) != Some(t));
        println!(
            "  burst {nb}: {} tok, acc {a}/{d}{}",
            b.len(),
            match div {
                Some(j) => format!("  <-- FIRST DIVERGENCE at burst-local tok {j} (global {})", start + j),
                None => String::new(),
            }
        );
        if b.is_empty() { break; }
        out.extend_from_slice(&b);
    }

    match reference.iter().zip(out.iter()).position(|(a, b)| a != b) {
        None if reference.len().min(out.len()) >= n_new.min(out.len()) && !out.is_empty() => {
            println!("MATCH ({} tok)", out.len());
        }
        None => println!("SHORT: ref {} vs session {}", reference.len(), out.len()),
        Some(i) => {
            println!(
                "DIVERGE at tok {i}: ref={:?} sess={:?}",
                &reference[i..(i + 5).min(reference.len())],
                &out[i..(i + 5).min(out.len())]
            );
            std::process::exit(1);
        }
    }
    Ok(())
}
