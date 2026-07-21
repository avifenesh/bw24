//! FR-Spec ranking builder: tokenize a corpus with a model's OWN tokenizer, rank token ids by
//! frequency, and emit a minimal GGUF carrying the top-N `d2t` (i32) list that BW24_FRSPEC_TRIM /
//! BW24_MTP_DRAFT consume. Needed because trim files are VOCAB artifacts — the published Qwen3.6
//! rankings from another tokenizer cannot transfer to Hy3's vocabulary.
//!
//! usage: frspec-rank <model.gguf|hf_dir> <out.gguf> <topN> <corpus file/dir>...
//!
//! Accepts EITHER a .gguf file (tokenizer from GGUF metadata) OR an HF safetensors directory
//! containing tokenizer.json (the ST-native path — no GGUF dependency for ST checkpoints).
use bw24_gguf::GgufFile;
use bw24_tokenizer::Tokenizer;

fn collect_files(path: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(meta) = std::fs::symlink_metadata(path) else {
        return;
    };
    // Corpus roots often contain convenience symlinks back into a projects tree. Following those
    // makes the walk cyclic and silently weights the same source multiple times.
    if meta.file_type().is_symlink() {
        return;
    }
    if path.is_dir() {
        if matches!(
            path.file_name().and_then(|name| name.to_str()),
            Some(".git" | ".venv" | "node_modules" | "target")
        ) {
            return;
        }
        if let Ok(rd) = std::fs::read_dir(path) {
            for e in rd.flatten() {
                collect_files(&e.path(), out);
            }
        }
    } else if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        // text-ish sources only
        if matches!(
            ext,
            "txt"
                | "md"
                | "rs"
                | "py"
                | "cu"
                | "c"
                | "h"
                | "cpp"
                | "json"
                | "toml"
                | "sh"
                | "js"
                | "ts"
        ) {
            out.push(path.to_path_buf());
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!("usage: frspec-rank <model.gguf|hf_dir> <out.gguf> <topN> <corpus>...");
        std::process::exit(1);
    }
    // DIRECTORY path = HF safetensors checkpoint (tokenizer.json); file = GGUF.
    let model_path = std::path::Path::new(&args[1]);
    let is_dir = model_path.is_dir();
    let g: Option<GgufFile> = if is_dir {
        None
    } else {
        Some(GgufFile::open(&args[1])?)
    };
    let tok = match &g {
        Some(g) => Tokenizer::from_gguf(g).map_err(|e| format!("tokenizer: {e}"))?,
        None => Tokenizer::from_hf_dir(model_path)
            .map_err(|e| format!("HF tokenizer init failed: {e}"))?,
    };
    let top_n: usize = args[3].parse()?;
    let vocab = tok.vocab_size();

    let mut files = Vec::new();
    for a in &args[4..] {
        collect_files(std::path::Path::new(a), &mut files);
    }
    eprintln!(
        "[frspec-rank] {} corpus files, vocab {}",
        files.len(),
        vocab
    );

    let mut counts = vec![0u64; vocab];
    let mut total = 0u64;
    for f in &files {
        let Ok(text) = std::fs::read_to_string(f) else {
            continue;
        };
        for id in tok.encode(&text, false) {
            if (id as usize) < vocab {
                counts[id as usize] += 1;
                total += 1;
            }
        }
    }
    eprintln!("[frspec-rank] {total} tokens counted");

    // rank by frequency desc, id asc tiebreak (deterministic). Zero-count ids are EXCLUDED from
    // preference but the list must still fill top_n — pad with ascending unseen ids (they cost
    // nothing: the draft never proposes what the head's trimmed rows can't produce... they simply
    // occupy cover slots). Practical corpora cover ~60-120k distinct ids.
    let d2t = bw24_gguf::d2t::rank_top_n(&counts, top_n);
    let covered: u64 = d2t.iter().map(|&i| counts[i as usize]).sum();
    eprintln!("[frspec-rank] top {} covers {:.2}% of corpus tokens",
              d2t.len(), covered as f64 / total.max(1) as f64 * 100.0);
    bw24_gguf::d2t::write_d2t(&args[2], &d2t)?;
    eprintln!("[frspec-rank] wrote {} ({} ids) + {}.txt", args[2], d2t.len(), args[2]);
    Ok(())
}
