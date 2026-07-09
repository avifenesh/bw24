//! FR-Spec ranking builder: tokenize a corpus with a model's OWN tokenizer, rank token ids by
//! frequency, and emit a minimal GGUF carrying the top-N `d2t` (i32) list that BW24_FRSPEC_TRIM /
//! BW24_MTP_DRAFT consume. Needed because trim files are VOCAB artifacts — the published Qwen3.6
//! rankings (151936-vocab) cannot transfer to the Qwen3.5-9B (248320-vocab).
//!
//! usage: frspec-rank <model.gguf|hf_dir> <out.gguf> <topN> <corpus file/dir>...
//!
//! Accepts either a GGUF file (tokenizer from GGUF metadata) or an HF directory containing
//! tokenizer.json (safetensors checkpoints). ST artifacts derive from ST inputs.
use bw24_gguf::GgufFile;
use bw24_tokenizer::Tokenizer;
use std::io::Write;

fn collect_files(path: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    if path.is_dir() {
        if let Ok(rd) = std::fs::read_dir(path) {
            for e in rd.flatten() {
                collect_files(&e.path(), out);
            }
        }
    } else if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        // text-ish sources only
        if matches!(ext, "txt" | "md" | "rs" | "py" | "cu" | "c" | "h" | "cpp" | "json" | "toml" | "sh" | "js" | "ts") {
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
    let model_path = std::path::Path::new(&args[1]);
    let tok = if model_path.is_dir() {
        // HF directory with tokenizer.json (safetensors checkpoints)
        Tokenizer::from_hf_dir(model_path)
            .map_err(|e| format!("HF tokenizer from {}: {e}", args[1]))?
    } else {
        // GGUF file
        let g = GgufFile::open(&args[1])?;
        Tokenizer::from_gguf(&g).map_err(|e| format!("tokenizer: {e}"))?
    };
    let top_n: usize = args[3].parse()?;
    let vocab = tok.vocab_size();

    let mut files = Vec::new();
    for a in &args[4..] {
        collect_files(std::path::Path::new(a), &mut files);
    }
    eprintln!("[frspec-rank] {} corpus files, vocab {}", files.len(), vocab);

    let mut counts = vec![0u64; vocab];
    let mut total = 0u64;
    for f in &files {
        let Ok(text) = std::fs::read_to_string(f) else { continue };
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
    let mut idx: Vec<u32> = (0..vocab as u32).collect();
    idx.sort_by(|&a, &b| counts[b as usize].cmp(&counts[a as usize]).then(a.cmp(&b)));
    let d2t: Vec<i32> = idx[..top_n.min(vocab)].iter().map(|&i| i as i32).collect();
    let covered: u64 = d2t.iter().map(|&i| counts[i as usize]).sum();
    eprintln!("[frspec-rank] top {} covers {:.2}% of corpus tokens",
              d2t.len(), covered as f64 / total.max(1) as f64 * 100.0);

    // ---- minimal GGUF v3 write: 0 KV, 1 tensor "d2t" i32 [top_n] ----
    let mut out = std::fs::File::create(&args[2])?;
    out.write_all(b"GGUF")?;
    out.write_all(&3u32.to_le_bytes())?;            // version
    out.write_all(&1u64.to_le_bytes())?;            // n_tensors
    out.write_all(&1u64.to_le_bytes())?;            // n_kv (alignment key below)
    // one KV: general.alignment = 32 (u32 type id 4)
    let k = b"general.alignment";
    out.write_all(&(k.len() as u64).to_le_bytes())?;
    out.write_all(k)?;
    out.write_all(&4u32.to_le_bytes())?;            // GGUF_TYPE_UINT32
    out.write_all(&32u32.to_le_bytes())?;
    // tensor info: name "d2t", 1 dim [top_n], ggml type I32 (=26), offset 0
    let name = b"d2t";
    out.write_all(&(name.len() as u64).to_le_bytes())?;
    out.write_all(name)?;
    out.write_all(&1u32.to_le_bytes())?;            // n_dims
    out.write_all(&(d2t.len() as u64).to_le_bytes())?;
    out.write_all(&26u32.to_le_bytes())?;           // GGML_TYPE_I32
    out.write_all(&0u64.to_le_bytes())?;            // offset
    // pad header to alignment 32
    let pos = out.metadata()?.len();
    let pad = (32 - (pos % 32)) % 32;
    out.write_all(&vec![0u8; pad as usize])?;
    for v in &d2t {
        out.write_all(&v.to_le_bytes())?;
    }
    eprintln!("[frspec-rank] wrote {} ({} ids)", args[2], d2t.len());
    Ok(())
}
