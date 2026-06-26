// Compile engine .cu kernels to a sm_120a fatbin (same pattern as bw24-probe).
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let cu = "cu/kernels.cu";
    println!("cargo:rerun-if-changed={cu}");

    let fatbin = out.join("kernels.fatbin");
    let nvcc = std::env::var("BW24_NVCC").unwrap_or_else(|_| "/usr/local/cuda-13.1/bin/nvcc".into());
    let status = Command::new(&nvcc)
        .args([
            "-gencode", "arch=compute_120a,code=sm_120a",
            "-O3", "--fatbin",
            "-o", fatbin.to_str().unwrap(),
            cu,
        ])
        .status()
        .expect("spawn nvcc");
    assert!(status.success(), "nvcc fatbin build failed");
    println!("cargo:rustc-env=BW24_ENGINE_FATBIN={}", fatbin.display());
}
