//! plainbatch margin probe (lane/plainbatch-probe, 2026-08-04): teacher-force a SERVED
//! token stream through the CLI tokenwise-oracle config (batched prime_cache + decode_step)
//! and record every position where the oracle's greedy argmax disagrees with the served
//! token, plus the logit gap at each flip (l[greedy] - l[served]) and the running top1-top2
//! margin distribution. Separates INDEPENDENT flips from post-flip sequence separation —
//! the naive stream diff saturates after the first flip because the two sequences condition
//! on different prefixes.
//!
//! FP near-tie class criterion (mission receipt): flip gaps sub-0.2 vs a median step margin
//! in the ~3.x range, flips sparse and depth-stable. Growing flip density with depth or
//! wide-margin flips = NOT the accepted class -> instrument further.
//!
//! usage: plainbatch-margin-probe <ckpt dir|gguf> <served-tokens-file>
//!        (env: MEMRA_PROMPT, chat template always applied — matches the probe battery arms)
//! tokens file: any text; every ASCII digit run is a token id (run-gen forced-file format).
use memra_engine::forward::argmax;
use memra_engine::hybrid::HybridModel;
use memra_engine::memra_gguf::GgufFile;
use memra_engine::Engine;

fn top2(l: &[f32]) -> (usize, f32, usize, f32) {
    let (mut i1, mut v1, mut i2, mut v2) = (0usize, f32::NEG_INFINITY, 0usize, f32::NEG_INFINITY);
    for (i, &v) in l.iter().enumerate() {
        if v > v1 {
            i2 = i1; v2 = v1; i1 = i; v1 = v;
        } else if v > v2 {
            i2 = i; v2 = v;
        }
    }
    (i1, v1, i2, v2)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: plainbatch-margin-probe <ckpt> <served-tokens-file>");
    let forced_file = args.next().expect("need <served-tokens-file>");
    let raw = std::fs::read_to_string(&forced_file)?;
    let served: Vec<u32> = raw
        .split(|c: char| !c.is_ascii_digit())
        .filter(|f| !f.is_empty())
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    assert!(!served.is_empty(), "no token ids in {forced_file}");

    let e = Engine::new(0)?;
    let is_dir = std::path::Path::new(&path).is_dir();
    let g: Option<GgufFile> = if is_dir { None } else { Some(GgufFile::open(&path)?) };
    let src = if is_dir {
        Some(memra_engine::memra_gguf::source::SafetensorsSource::open(std::path::Path::new(&path))?)
    } else {
        None
    };
    let model = match (&g, &src) {
        (Some(g), _) => HybridModel::load_without_mtp(&e, g)?,
        (None, Some(s)) => HybridModel::load_from_source_without_mtp(&e, s)?,
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
    println!("prompt {} tok, served stream {} tok", ids.len(), served.len());

    // CLI-oracle config: batched prime (prime_cache), then tokenwise decode_step.
    let mut cache = memra_engine::cache::Cache::new(&e, &model.cfg, ids.len() + served.len() + 8)?;
    let (mut logits, _h, _x) = model.prime_cache(&e, &ids, &mut cache, 0)?;

    let mut margins: Vec<f32> = Vec::with_capacity(served.len());
    let mut n_flips = 0usize;
    for (step, &srv_tok) in served.iter().enumerate() {
        let greedy = argmax(&logits) as u32;
        let (i1, v1, i2, v2) = top2(&logits);
        margins.push(v1 - v2);
        if greedy != srv_tok {
            n_flips += 1;
            let gap = logits[greedy as usize] - logits[srv_tok as usize];
            println!(
                "FLIP step {step}: oracle greedy {greedy} vs served {srv_tok}  \
                 gap={gap:.4}  top2=({i1}:{v1:.4}, {i2}:{v2:.4})  step_margin={:.4}",
                v1 - v2
            );
        }
        logits = model.decode_step(&e, srv_tok, &mut cache)?;
    }
    let mut sorted = margins.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let med = sorted[sorted.len() / 2];
    let p10 = sorted[sorted.len() / 10];
    println!(
        "SUMMARY: {} steps, {} independent flips ({:.3}/100tok)  \
         step-margin median={med:.3} p10={p10:.3} min={:.4}",
        served.len(),
        n_flips,
        n_flips as f64 * 100.0 / served.len() as f64,
        sorted[0]
    );
    Ok(())
}
