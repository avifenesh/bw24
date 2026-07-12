//! tok-freq: corpus token-frequency ranking for FR-Spec-class draft-head trims.
//! Tokenizes every .md/.txt under <dir> (recursive) with the model's own tokenizer, counts
//! ids, writes the top-N ids (one per line, rank order) to <out>.
//! usage: tok-freq <model.gguf> <corpus_dir> <N> <out.txt>

use bw24_gguf::GgufFile;
use bw24_tokenizer::Tokenizer;

fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, out);
            } else if matches!(p.extension().and_then(|x| x.to_str()), Some("md" | "txt")) {
                out.push(p);
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut a = std::env::args().skip(1);
    let (model, dir, n, out) = (
        a.next().expect("usage: tok-freq <model.gguf> <corpus_dir> <N> <out.txt>"),
        a.next().expect("corpus_dir"),
        a.next().expect("N").parse::<usize>()?,
        a.next().expect("out.txt"),
    );
    let g = GgufFile::open(&model)?;
    let tok = Tokenizer::from_gguf(&g)?;
    let mut files = Vec::new();
    walk(std::path::Path::new(&dir), &mut files);
    let mut counts: std::collections::HashMap<u32, u64> = std::collections::HashMap::new();
    let mut total = 0u64;
    for f in &files {
        if let Ok(text) = std::fs::read_to_string(f) {
            for id in tok.encode(&text, false) {
                *counts.entry(id).or_insert(0) += 1;
                total += 1;
            }
        }
    }
    let mut ranked: Vec<(u32, u64)> = counts.into_iter().collect();
    ranked.sort_by(|x, y| y.1.cmp(&x.1).then(x.0.cmp(&y.0)));
    let covered: u64 = ranked.iter().take(n).map(|x| x.1).sum();
    eprintln!("{} files, {} tokens, {} unique; top-{} covers {:.2}%",
              files.len(), total, ranked.len(), n, covered as f64 / total as f64 * 100.0);
    let ids: Vec<String> = ranked.iter().take(n).map(|x| x.0.to_string()).collect();
    std::fs::write(&out, ids.join("\n"))?;
    Ok(())
}
