//! Integer-exact parity test: bw24 tokenizer vs llama.cpp `llama-tokenize`.
//!
//! For each corpus string we assert `Tokenizer::encode(s, false)` equals the token
//! ids produced by `llama-tokenize --no-bos --ids -m MODEL -p s`, token-for-token,
//! with NO tolerance. Round-trip (`decode(encode(s)) == s`) is also asserted.
//!
//! The model + llama-tokenize binary paths can be overridden via env:
//!   BW24_TEST_MODEL   (default: the Qwen3.5-9B NVFP4 MTP gguf)
//!   BW24_LLAMA_TOKENIZE (default: llama.cpp build/bin/llama-tokenize)
//!
//! If either is missing the test is skipped (prints a notice) rather than failing,
//! so the suite still runs in environments without the model/reference binary.

use bw24_gguf::GgufFile;
use bw24_tokenizer::Tokenizer;
use std::path::Path;
use std::process::Command;

const DEFAULT_MODEL: &str =
    "/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf";
const DEFAULT_LLAMA_TOKENIZE: &str =
    "/home/avifenesh/projects/llama.cpp/build/bin/llama-tokenize";

fn model_path() -> String {
    std::env::var("BW24_TEST_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string())
}
fn llama_tokenize_path() -> String {
    std::env::var("BW24_LLAMA_TOKENIZE").unwrap_or_else(|_| DEFAULT_LLAMA_TOKENIZE.to_string())
}

/// Varied corpus: plain ascii, unicode, code w/ symbols, leading/trailing spaces,
/// chatml special tokens, numbers, mixed, newlines, emoji.
fn corpus() -> Vec<&'static str> {
    vec![
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "héllo café 日本語 — naïve façade",
        "fn main() { let x: i32 = 42; println!(\"{}\", x*2); }",
        "   leading and trailing spaces   ",
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
        "1234567890 3.14159 0xDEADBEEF 1e-9",
        "Mixed: ALLCAPS lowercase CamelCase snake_case kebab-case",
        "line one\nline two\n\nline four\ttabbed",
        "emoji test 🚀🔥✅ and math ∑∫√π≠≤",
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
        "a",
    ]
}

/// Shell out to llama-tokenize and parse the `[id, id, ...]` output of `--ids`.
fn llama_tokenize(model: &str, bin: &str, text: &str) -> Option<Vec<u32>> {
    // --no-escape: pass the prompt bytes verbatim (our corpus contains real
    // newlines/tabs, not backslash escapes, so llama must not re-interpret them).
    let out = Command::new(bin)
        .args(["--no-bos", "--no-escape", "--ids", "-m", model, "-p", text])
        .output()
        .ok()?;
    if !out.status.success() {
        eprintln!(
            "llama-tokenize failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        return None;
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    // output is like: "[9419, 11, 1814, 0]" possibly with surrounding whitespace/newlines.
    let start = stdout.find('[')?;
    let end = stdout[start..].find(']')? + start;
    let inner = &stdout[start + 1..end];
    let ids: Vec<u32> = inner
        .split(',')
        .filter_map(|s| s.trim().parse::<u32>().ok())
        .collect();
    Some(ids)
}

#[test]
fn parity_with_llama_tokenize() {
    let model = model_path();
    let bin = llama_tokenize_path();
    if !Path::new(&model).exists() {
        eprintln!("SKIP: model not found at {model} (set BW24_TEST_MODEL)");
        return;
    }
    if !Path::new(&bin).exists() {
        eprintln!("SKIP: llama-tokenize not found at {bin} (set BW24_LLAMA_TOKENIZE)");
        return;
    }

    let g = GgufFile::open(&model).expect("open gguf");
    let tok = Tokenizer::from_gguf(&g).expect("build tokenizer");

    let mut total = 0;
    let mut matched = 0;
    let mut failures: Vec<String> = Vec::new();

    for s in corpus() {
        let Some(reference) = llama_tokenize(&model, &bin, s) else {
            failures.push(format!("llama-tokenize produced no output for {s:?}"));
            total += 1;
            continue;
        };
        let ours = tok.encode(s, false);
        total += 1;
        if ours == reference {
            matched += 1;
        } else {
            failures.push(format!(
                "MISMATCH for {s:?}\n   ours:  {ours:?}\n   llama: {reference:?}"
            ));
        }
    }

    eprintln!("parity: {matched}/{total} corpus strings matched llama-tokenize EXACTLY");
    if !failures.is_empty() {
        for f in &failures {
            eprintln!("{f}");
        }
        panic!("{} corpus string(s) did not match llama-tokenize", failures.len());
    }
}

#[test]
fn round_trip() {
    let model = model_path();
    if !Path::new(&model).exists() {
        eprintln!("SKIP: model not found at {model}");
        return;
    }
    let g = GgufFile::open(&model).expect("open gguf");
    let tok = Tokenizer::from_gguf(&g).expect("build tokenizer");

    // Round-trip should reproduce the input exactly for these (no lossy bytes).
    for s in corpus() {
        let ids = tok.encode(s, false);
        let back = tok.decode_special(&ids, true);
        assert_eq!(back, s, "round-trip differs for {s:?}: got {back:?}");
    }
}

/// Hardcoded golden pairs captured from llama-tokenize (so a smoke check still
/// runs even when the reference binary/model are unavailable).
#[test]
fn golden_pairs() {
    let model = model_path();
    if !Path::new(&model).exists() {
        eprintln!("SKIP: model not found at {model}");
        return;
    }
    let g = GgufFile::open(&model).expect("open gguf");
    let tok = Tokenizer::from_gguf(&g).expect("build tokenizer");

    let golden: &[(&str, &[u32])] = &[
        ("Hello, world!", &[9419, 11, 1814, 0]),
        ("<|im_start|>user", &[248045, 846]),
        ("héllo café 日本語", &[71, 17951, 379, 50203, 220, 247359]),
        ("   spaced", &[256, 61674]),
    ];
    for (s, ids) in golden {
        assert_eq!(&tok.encode(s, false), ids, "golden mismatch for {s:?}");
    }
}
