//! gguf-inspect: dump GGUF header, key metadata, and tensor summary.
//! Validates the parser against real models + extracts model hyperparams.

use bw24_gguf::{GgufFile, MetaValue};
use std::collections::BTreeMap;

fn main() {
    let path = std::env::args().nth(1).expect("usage: gguf-inspect <model.gguf>");
    let g = GgufFile::open(&path).expect("open gguf");

    println!("== {path} ==");
    println!("version={} alignment={} data_start={}", g.version, g.alignment, g.data_start);
    println!("metadata kv={} tensors={}", g.metadata.len(), g.tensors.len());
    println!("arch={:?}", g.arch());

    // Key hyperparams (Qwen3-style names; meta_arch tries {arch}.{suffix}).
    let hp = [
        "context_length", "embedding_length", "block_count", "feed_forward_length",
        "attention.head_count", "attention.head_count_kv", "attention.key_length",
        "attention.value_length", "attention.layer_norm_rms_epsilon", "rope.freq_base",
        "vocab_size", "expert_count", "expert_used_count", "expert_feed_forward_length",
    ];
    println!("\n-- hyperparams --");
    for k in hp {
        if let Some(v) = g.meta_arch(k) {
            println!("  {k} = {}", fmt_short(v));
        }
    }

    // Tokenizer summary (don't dump the whole vocab).
    println!("\n-- tokenizer --");
    for k in ["tokenizer.ggml.model", "tokenizer.ggml.pre", "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id"] {
        if let Some(v) = g.metadata.get(k) { println!("  {k} = {}", fmt_short(v)); }
    }
    if let Some(MetaValue::Array(a)) = g.metadata.get("tokenizer.ggml.tokens") {
        println!("  tokenizer.ggml.tokens = [{} entries]", a.len());
    }

    // Tensor type histogram + a few sample tensors.
    let mut by_type: BTreeMap<String, (u64, u64)> = BTreeMap::new(); // type -> (count, bytes)
    let mut total_bytes = 0u64;
    for t in &g.tensors {
        let e = by_type.entry(format!("{:?}", t.ggml_type)).or_default();
        e.0 += 1; e.1 += t.n_bytes;
        total_bytes += t.n_bytes;
    }
    println!("\n-- tensor types --");
    for (ty, (cnt, bytes)) in &by_type {
        println!("  {ty:8} count={cnt:4}  {:.2} GB", *bytes as f64 / 1e9);
    }
    println!("  TOTAL tensor bytes = {:.2} GB", total_bytes as f64 / 1e9);

    println!("\n-- sample tensors --");
    for t in g.tensors.iter().take(6) {
        println!("  {:40} {:?} ne={:?} off={} {} B", t.name, t.ggml_type, t.ne, t.offset, t.n_bytes);
    }
    // Show the lm_head / embedding / first block tensors specifically.
    println!("\n-- key tensors --");
    for name in ["token_embd.weight", "output_norm.weight", "output.weight",
                 "blk.0.attn_norm.weight", "blk.0.attn_q.weight", "blk.0.attn_k.weight",
                 "blk.0.attn_v.weight", "blk.0.attn_output.weight", "blk.0.ffn_gate.weight",
                 "blk.0.ffn_up.weight", "blk.0.ffn_down.weight", "blk.0.attn_q_norm.weight",
                 "blk.0.attn_k_norm.weight", "blk.0.ffn_norm.weight"] {
        if let Some(t) = g.find(name) {
            println!("  {:32} {:?} ne={:?}", name, t.ggml_type, t.ne);
        }
    }

    // Verify data section bounds: last tensor end must fit in the file.
    if let Some(last) = g.tensors.iter().max_by_key(|t| t.offset + t.n_bytes) {
        let end = g.data_start + last.offset + last.n_bytes;
        println!("\nlast tensor data ends at byte {end}");
    }
}

fn fmt_short(v: &MetaValue) -> String {
    match v {
        MetaValue::Array(a) => format!("[{} elems]", a.len()),
        MetaValue::String(s) if s.len() > 60 => format!("\"{}...\"", &s[..60]),
        other => format!("{other:?}"),
    }
}
