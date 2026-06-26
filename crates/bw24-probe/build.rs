// Phase-0 spine: compile a hand-written .cu kernel to a sm_120a fatbin via nvcc at build time.
// Proves the cargo -> build.rs -> nvcc -> fatbin -> cudarc-load path that the whole engine rests on.
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let cu = "src/kernels.cu";
    println!("cargo:rerun-if-changed={cu}");

    let fatbin = out.join("kernels.fatbin");
    // CRITICAL: -gencode arch=compute_120a,code=sm_120a — the only form that assembles
    // FP4/FP8 block-scale mma on sm_120 (the bare -arch=sm_120a shortcut misroutes to compute_120).
    let nvcc = std::env::var("BW24_NVCC").unwrap_or_else(|_| "/usr/local/cuda-13.1/bin/nvcc".into());
    let status = Command::new(&nvcc)
        .args([
            "-gencode", "arch=compute_120a,code=sm_120a",
            "-O3", "--fatbin",
            "-o", fatbin.to_str().unwrap(),
            cu,
        ])
        .status()
        .expect("failed to spawn nvcc");
    assert!(status.success(), "nvcc fatbin build failed");
    println!("cargo:rustc-env=BW24_FATBIN={}", fatbin.display());
}
