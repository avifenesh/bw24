//! Dump the serving-path Hy3 embedding and post-layer-0 residual as JSONL.
//!
//! This intentionally uses `decode_step_aux`: the dump observes the same T=1 path as serving,
//! including its KV representation and quantized matmuls.  The Python sidecar in
//! `research/per-expert-quant/hy3_layer0_reference.py` computes the corresponding official
//! Transformers layer-0 reference directly from the pinned BF16 source checkpoint.

use std::io::{BufWriter, Write};
use std::path::Path;

use bw24_engine::Engine;
use bw24_engine::cache::Cache;
use bw24_engine::hybrid::HybridModel;
use bw24_gguf::source::{Hy3RepackSource, TensorSource};

fn write_json_string(out: &mut impl Write, value: &str) -> std::io::Result<()> {
    write!(out, "\"")?;
    for ch in value.chars() {
        match ch {
            '"' => write!(out, "\\\"")?,
            '\\' => write!(out, "\\\\")?,
            '\n' => write!(out, "\\n")?,
            '\r' => write!(out, "\\r")?,
            '\t' => write!(out, "\\t")?,
            c if c <= '\u{1f}' => write!(out, "\\u{:04x}", c as u32)?,
            c => write!(out, "{c}")?,
        }
    }
    write!(out, "\"")
}

fn write_f32_array(out: &mut impl Write, values: &[f32]) -> std::io::Result<()> {
    write!(out, "[")?;
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("non-finite value at vector index {index}"),
            ));
        }
        if index != 0 {
            write!(out, ",")?;
        }
        write!(out, "{value}")?;
    }
    write!(out, "]")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let artifact = args
        .next()
        .ok_or("usage: hy3-layer0-oracle <overlay-dir> <token-id> [token-id ...]")?;
    let tokens: Vec<u32> = args
        .map(|arg| {
            arg.parse::<u32>()
                .map_err(|err| format!("invalid token id {arg:?}: {err}"))
        })
        .collect::<Result<_, _>>()?;
    if tokens.is_empty() {
        return Err("at least one token id is required".into());
    }

    let source = Hy3RepackSource::open(Path::new(&artifact))?;
    let cfg = source.config();
    if !cfg.arch.is_hy3() {
        return Err(format!("artifact architecture is {:?}, expected Hy3", cfg.arch).into());
    }
    if let Some(&bad) = tokens.iter().find(|&&token| token >= cfg.n_vocab) {
        return Err(format!("token id {bad} is outside vocabulary size {}", cfg.n_vocab).into());
    }

    let engine = Engine::new(0)?;
    let model = HybridModel::load_from_source(&engine, &source)?;
    let embeddings = engine.dtoh(&model.embed(&engine, &tokens)?)?;
    let n_embd = model.cfg.n_embd as usize;
    let mut cache = Cache::new(&engine, &model.cfg, tokens.len() + 1)?;
    let mut out = BufWriter::new(std::io::stdout().lock());

    write!(
        out,
        "{{\"schema\":\"bw24.hy3.layer0.v1\",\"producer\":\"bw24\",\"artifact\":"
    )?;
    write_json_string(&mut out, &artifact)?;
    write!(
        out,
        ",\"tokens\":{:?},\"n_embd\":{},\"precision\":\"runtime\"}}\n",
        tokens, n_embd
    )?;

    for (position, &token_id) in tokens.iter().enumerate() {
        let (_, aux) = model.decode_step_aux(&engine, token_id, &mut cache, &[0])?;
        let layer0 = engine.dtoh(
            aux.first()
                .ok_or("decode_step_aux returned no layer-0 residual")?,
        )?;
        let embedding = &embeddings[position * n_embd..(position + 1) * n_embd];

        write!(
            out,
            "{{\"schema\":\"bw24.hy3.layer0.v1\",\"producer\":\"bw24\",\"position\":{position},\"token_id\":{token_id},\"embedding\":"
        )?;
        write_f32_array(&mut out, embedding)?;
        write!(out, ",\"layer0_residual\":")?;
        write_f32_array(&mut out, &layer0)?;
        write!(out, "}}\n")?;
    }
    out.flush()?;
    Ok(())
}
