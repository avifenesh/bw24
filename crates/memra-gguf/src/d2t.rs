//! Minimal GGUF v3 writer for FR-Spec `d2t` rank artifacts (one i32 tensor + txt sidecar).
//! Shared by frspec-rank (corpus tokenize) and frspec-owngen (model-own generations) —
//! trim files are VOCAB artifacts and must always be derived per model family.

use std::io::Write;

/// Write `<path>` as a 1-tensor GGUF (`d2t` i32 [n]) plus a `<path>.txt` sidecar
/// (one id per line, rank order — the gemma drafter trim consumes this form).
pub fn write_d2t(path: &str, d2t: &[i32]) -> std::io::Result<()> {
    let mut out = std::fs::File::create(path)?;
    out.write_all(b"GGUF")?;
    out.write_all(&3u32.to_le_bytes())?;            // version
    out.write_all(&1u64.to_le_bytes())?;            // n_tensors
    out.write_all(&1u64.to_le_bytes())?;            // n_kv
    let k = b"general.alignment";
    out.write_all(&(k.len() as u64).to_le_bytes())?;
    out.write_all(k)?;
    out.write_all(&4u32.to_le_bytes())?;            // GGUF_TYPE_UINT32
    out.write_all(&32u32.to_le_bytes())?;
    let name = b"d2t";
    out.write_all(&(name.len() as u64).to_le_bytes())?;
    out.write_all(name)?;
    out.write_all(&1u32.to_le_bytes())?;            // n_dims
    out.write_all(&(d2t.len() as u64).to_le_bytes())?;
    out.write_all(&26u32.to_le_bytes())?;           // GGML_TYPE_I32
    out.write_all(&0u64.to_le_bytes())?;            // offset
    let pos = out.metadata()?.len();
    let pad = (32 - (pos % 32)) % 32;
    out.write_all(&vec![0u8; pad as usize])?;
    for v in d2t {
        out.write_all(&v.to_le_bytes())?;
    }
    let mut tf = std::fs::File::create(format!("{path}.txt"))?;
    for v in d2t {
        writeln!(tf, "{v}")?;
    }
    Ok(())
}

/// Rank ids by count desc (id-asc tiebreak, deterministic) and take the top `n`,
/// padding with ascending unseen ids (cover slots — the draft never proposes them).
pub fn rank_top_n(counts: &[u64], n: usize) -> Vec<i32> {
    let mut idx: Vec<u32> = (0..counts.len() as u32).collect();
    idx.sort_by(|&a, &b| counts[b as usize].cmp(&counts[a as usize]).then(a.cmp(&b)));
    idx[..n.min(counts.len())].iter().map(|&i| i as i32).collect()
}
