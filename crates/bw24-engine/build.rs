// Compile engine .cu kernels to a sm_120a fatbin (same pattern as bw24-probe).
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let nvcc = std::env::var("BW24_NVCC").unwrap_or_else(|_| "/usr/local/cuda-13.1/bin/nvcc".into());

    for (src, env) in [("cu/kernels.cu", "BW24_ENGINE_FATBIN"), ("cu/hybrid.cu", "BW24_HYBRID_FATBIN"),
                       ("cu/qmatvec.cu", "BW24_QMATVEC_FATBIN"), ("cu/flash_attn.cu", "BW24_FLASH_FATBIN"),
                       ("cu/qmatvec_gemm.cu", "BW24_GEMM_FATBIN"), ("cu/moe_router.cu", "BW24_ROUTER_FATBIN")] {
        println!("cargo:rerun-if-changed={src}");
        let stem = src.split('/').last().unwrap().trim_end_matches(".cu");
        let fatbin = out.join(format!("{stem}.fatbin"));
        let status = Command::new(&nvcc)
            .args([
                "-gencode", "arch=compute_120a,code=sm_120a",
                "-O3", "--fatbin",
                "-o", fatbin.to_str().unwrap(),
                src,
            ])
            .status()
            .expect("spawn nvcc");
        assert!(status.success(), "nvcc fatbin build failed for {src}");
        println!("cargo:rustc-env={env}={}", fatbin.display());
    }
}
