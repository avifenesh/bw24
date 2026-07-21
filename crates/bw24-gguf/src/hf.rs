//! `hf:` model-spec resolution — one-command model fetch.
//!
//! Spec forms accepted anywhere a model path is (run-gen, run-spec, bw24-server):
//!   hf:owner/repo             single-GGUF repo, or a safetensors checkpoint dir
//!   hf:owner/repo:Q4_K_M      substring-match one .gguf inside a multi-quant repo
//!
//! Downloads via the `hf` CLI (huggingface_hub) into `$BW24_MODELS_DIR/hf/owner--repo`
//! (fallback `~/.cache/bw24/models/hf/owner--repo`) and returns the local path. Already-
//! downloaded specs resolve offline without touching the network.

use std::path::PathBuf;

/// Expand an `hf:` model spec to a local path; anything else passes through unchanged.
pub fn resolve_arg(arg: &str) -> Result<String, String> {
    let Some(spec) = arg.strip_prefix("hf:") else { return Ok(arg.to_string()) };
    let (repo, pattern) = match spec.split_once(':') {
        Some((r, p)) => (r, Some(p)),
        None => (spec, None),
    };
    if repo.split('/').count() != 2 {
        return Err(format!("bad hf spec {arg:?} — want hf:owner/repo[:file-substring]"));
    }

    let root = std::env::var("BW24_MODELS_DIR").map(PathBuf::from).unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        PathBuf::from(home).join(".cache/bw24/models")
    });
    let dest = root.join("hf").join(repo.replace('/', "--"));

    if let Some(path) = pick(&dest, pattern)? {
        return Ok(path);
    }

    eprintln!("[hf] downloading {repo} -> {}", dest.display());
    let mut cmd = std::process::Command::new("hf");
    cmd.args(["download", repo, "--local-dir"]).arg(&dest);
    if let Some(p) = pattern {
        // Pull only matching GGUFs from multi-quant repos; bare repos fetch everything.
        cmd.arg("--include").arg(format!("*{p}*"));
    }
    let status = cmd.status().map_err(|e| {
        format!("`hf` CLI not runnable ({e}) — install with: pip install -U 'huggingface_hub[cli]'")
    })?;
    if !status.success() {
        return Err(format!("hf download {repo} failed ({status})"));
    }

    pick(&dest, pattern)?
        .ok_or_else(|| format!("{repo} downloaded but no usable model found in {}", dest.display()))
}

/// Choose the model inside `dest`: a pattern-matched (or lone) .gguf file, else the
/// directory itself when it holds a safetensors checkpoint. Ok(None) = nothing there yet.
fn pick(dest: &std::path::Path, pattern: Option<&str>) -> Result<Option<String>, String> {
    if !dest.is_dir() {
        return Ok(None);
    }
    let mut ggufs: Vec<PathBuf> = std::fs::read_dir(dest)
        .map_err(|e| format!("read {}: {e}", dest.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "gguf"))
        .filter(|p| {
            // Never auto-pick multimodal projector sidecars as the model.
            !p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("mmproj"))
        })
        .collect();
    if let Some(p) = pattern {
        ggufs.retain(|f| f.file_name().is_some_and(|n| n.to_string_lossy().contains(p)));
    }
    match ggufs.len() {
        1 => return Ok(Some(ggufs.remove(0).display().to_string())),
        0 => {}
        _ => {
            ggufs.sort();
            let names: Vec<_> = ggufs.iter().filter_map(|f| f.file_name()).collect();
            return Err(format!(
                "ambiguous — {} GGUFs match in {}: {names:?}\nnarrow with hf:owner/repo:<substring>",
                ggufs.len(), dest.display()
            ));
        }
    }
    if pattern.is_none() && dest.join("config.json").exists() {
        return Ok(Some(dest.display().to_string()));
    }
    Ok(None)
}
