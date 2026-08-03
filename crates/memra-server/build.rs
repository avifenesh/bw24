//! Bake the build's git SHA into the binary for the OpenAI `system_fingerprint` field
//! (gap-scan F1): the fingerprint identifies the backend configuration a response was
//! produced by, so determinism claims (`seed`) are checkable across deploys.

fn main() {
    let sha = std::process::Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=MEMRA_BUILD_SHA={sha}");
}
