//! Teacher-forced replay acceptance driver (hqmtp MTP-heal protocol; see
//! HybridModel::replay_acceptance for the metric definition). Walks a FIXED corpus text and
//! scores the MTP head's draft chain against the trunk's own teacher-forced greedy picks at
//! sampled positions — no generation, so degenerate self-generated loops cannot inflate
//! acceptance, and two arms (bf16 full-prec ceiling vs NVFP4) score on IDENTICAL contexts.
//!
//! Run: replay-acceptance <model.gguf|hf_dir>   (corpus text via BW24_PROMPT_FILE)
//! Env:
//!   BW24_REPLAY_K=4        draft chain length per eval position
//!   BW24_REPLAY_STRIDE=16  eval every Nth corpus position
//!   BW24_REPLAY_CHUNK=512  forced-pass chunk (logits buffer = chunk x n_vocab f32)
//!   BW24_REPLAY_T=0        cap corpus tokens (0 = all)
//!   BW24_REPLAY_DUMP=f.jsonl  per-position rows: {"pos","drafts","targets","hits"}
//!   BW24_REPLAY_GATE=1     re-run a 2048-token prefix at chunk=64 and require identical
//!                          greedy track + drafts (chunk-boundary correctness gate)
//!   BW24_FULL_PREC=1       arm A (bf16 ST ceiling) — same knob as run-spec

use bw24_engine::Engine;
use bw24_engine::hybrid::HybridModel;
use bw24_gguf::GgufFile;
use std::io::Write;

const BUCKETS: &[(usize, usize)] = &[
    (0, 512), (512, 2048), (2048, 8192), (8192, 16384),
    (16384, 32768), (32768, 65536), (65536, usize::MAX),
];

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1)
        .expect("usage: replay-acceptance <model.gguf|hf_dir>  (corpus via BW24_PROMPT_FILE)");
    let e = Engine::new(0)?;
    let is_dir = std::path::Path::new(&path).is_dir();
    let g: Option<GgufFile> = if is_dir { None } else { Some(GgufFile::open(&path)?) };
    let model = match &g {
        Some(g) => HybridModel::load(&e, g)?,
        None => {
            let st = bw24_gguf::source::SafetensorsSource::open(std::path::Path::new(&path))?;
            HybridModel::load_from_source(&e, &st)?
        }
    };
    if model.mtp.is_none() {
        eprintln!("ERROR: model has no MTP/NextN head — replay acceptance is undefined.");
        std::process::exit(2);
    }

    let text = std::env::var("BW24_PROMPT_FILE")
        .map(|f| std::fs::read_to_string(&f).expect("BW24_PROMPT_FILE unreadable"))
        .expect("replay-acceptance needs BW24_PROMPT_FILE (the corpus text)");
    let tok = match &g {
        Some(g) => bw24_tokenizer::Tokenizer::from_gguf(g)?,
        None => bw24_tokenizer::Tokenizer::from_hf_dir(std::path::Path::new(&path))
            .map_err(|err| format!("HF tokenizer init failed: {err}"))?,
    };
    let mut ids = tok.encode(&text, true);
    let cap = env_usize("BW24_REPLAY_T", 0);
    if cap > 0 && ids.len() > cap { ids.truncate(cap); }
    let k = env_usize("BW24_REPLAY_K", 4);
    let stride = env_usize("BW24_REPLAY_STRIDE", 16);
    let chunk = env_usize("BW24_REPLAY_CHUNK", 512);
    println!("corpus: {} chars -> {} tokens | K={k} stride={stride} chunk={chunk} full_prec={}",
             text.len(), ids.len(), std::env::var("BW24_FULL_PREC").is_ok());

    let t0 = std::time::Instant::now();
    let (rows, bg) = model.replay_acceptance(&e, &ids, k, stride, chunk)?;
    let dt = t0.elapsed().as_secs_f64();
    println!("replay: {} eval positions in {dt:.1}s ({:.1} corpus tok/s incl. chains)",
             rows.len(), ids.len() as f64 / dt);

    // Chunk-boundary gate: an independent pass over a prefix with a DIFFERENT chunk size must
    // reproduce the greedy track and every draft exactly (catches position/rope/fill offset
    // bugs at chunk seams; FP-order exactness of the verify path is gated elsewhere).
    if std::env::var("BW24_REPLAY_GATE").is_ok() {
        let n_gate = 2048.min(ids.len());
        let (rows_g, bg_g) = model.replay_acceptance(&e, &ids[..n_gate], k, stride, 64)?;
        let mut bad = 0usize;
        for i in 1..n_gate.saturating_sub(1) {
            if bg[i] != bg_g[i] { bad += 1; if bad <= 3 {
                eprintln!("[gate] bg mismatch at pos {i}: {} vs {}", bg[i], bg_g[i]); } }
        }
        let main_prefix: Vec<_> = rows.iter().filter(|r| r.0 + k <= n_gate).collect();
        for (a, b) in main_prefix.iter().zip(rows_g.iter()) {
            if a.0 != b.0 || a.1 != b.1 { bad += 1; if bad <= 6 {
                eprintln!("[gate] draft mismatch at pos {}: {:?} vs {:?}", a.0, a.1, b.1); } }
        }
        if bad > 0 {
            eprintln!("[gate] FAIL: {bad} mismatches (chunk={chunk} vs 64)");
            std::process::exit(3);
        }
        println!("[gate] PASS: chunk={chunk} vs chunk=64 identical over {n_gate}-token prefix");
    }

    // aggregate per context-depth bucket
    let slot_hdr: String = (0..k).map(|j| format!("  slot{j}")).collect();
    println!("{:>14} {:>7}{slot_hdr}   chain", "ctx-bucket", "n");
    for &(lo, hi) in BUCKETS {
        let sel: Vec<_> = rows.iter().filter(|r| r.0 >= lo && r.0 < hi).collect();
        if sel.is_empty() { continue; }
        let n = sel.len();
        let mut slot_hit = vec![0usize; k];
        let mut chain_len = 0usize;
        for (_, d, t) in &sel {
            for j in 0..k { if d[j] == t[j] { slot_hit[j] += 1; } }
            let mut c = 0usize;
            while c < k && d[c] == t[c] { c += 1; }
            chain_len += c;
        }
        let slots: String = (0..k)
            .map(|j| format!(" {:6.3}", slot_hit[j] as f64 / n as f64)).collect();
        let hi_s = if hi == usize::MAX { "inf".into() } else { hi.to_string() };
        println!("{:>14} {:>7}{slots}   {:5.3}", format!("{lo}-{hi_s}"), n,
                 chain_len as f64 / (n * k) as f64);
    }

    if let Ok(dump) = std::env::var("BW24_REPLAY_DUMP") {
        let mut f = std::fs::File::create(&dump)?;
        for (p, d, t) in &rows {
            let hits: Vec<bool> = d.iter().zip(t.iter()).map(|(a, b)| a == b).collect();
            writeln!(f, "{{\"pos\":{p},\"drafts\":{d:?},\"targets\":{t:?},\"hits\":{hits:?}}}")?;
        }
        println!("wrote {} rows -> {dump}", rows.len());
    }
    Ok(())
}
