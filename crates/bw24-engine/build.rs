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

    // ---- Vendored llama MMQ GEMMs: a STATIC LIB with C-ABI host launchers (extern "C"). ----
    // Same kind as the CUTLASS artifact (a host-side launcher cannot go through the device-only fatbin
    // path), but ALWAYS built (no external header deps — fully ggml-decoupled). The launchers do
    // cudaFuncSetAttribute (>48KB dynamic smem) + the mul_mat_q kernel launch internally.
    // Called from Rust via FFI (mmq_ffi.rs), dispatched behind BW24_MMQ=1.
    // Two translation units: mmq_fp4.cu (Blackwell mxf4nvf4 W4A4) and mmq_q45k.cu
    // (Q4_K/Q5_K int8-MMA W4A8, sm_75+ portable). Both archived into one libbw24_mmq.a.
    {
        let mut objs: Vec<PathBuf> = Vec::new();
        // TUNE SEAM: BW24_MMQ_X_Q45K=64 rebuilds the k-quant MMQ with a 64-token tile
        // (47KB smem -> 2 CTA/SM vs 57KB/1; the q45k occupancy ceiling found by ncu).
        println!("cargo:rerun-if-env-changed=BW24_MMQ_X_Q45K");
        let q45k_x = std::env::var("BW24_MMQ_X_Q45K").ok();
        for mmq_src in ["cu/mmq_fp4.cu", "cu/mmq_q45k.cu"] {
            println!("cargo:rerun-if-changed={mmq_src}");
            let stem = mmq_src.split('/').last().unwrap().trim_end_matches(".cu");
            let obj = out.join(format!("{stem}.o"));
            let mut args: Vec<String> = vec![
                "-gencode".into(), "arch=compute_120a,code=sm_120a".into(),
                "-O3".into(), "-std=c++17".into(), "--expt-relaxed-constexpr".into(),
            ];
            if mmq_src.ends_with("mmq_q45k.cu") {
                if let Some(x) = &q45k_x { args.push(format!("-DMMQ_X={x}")); }
            }
            args.extend(["-c".into(), mmq_src.into(), "-o".into(), obj.to_str().unwrap().into()]);
            let status = Command::new(&nvcc)
                .args(&args)
                .status()
                .expect("spawn nvcc (mmq)");
            assert!(status.success(), "nvcc static-lib build failed for {mmq_src}");
            objs.push(obj);
        }
        let lib = out.join("libbw24_mmq.a");
        let _ = std::fs::remove_file(&lib);
        let mut ar_args = vec!["crus".to_string(), lib.to_str().unwrap().to_string()];
        ar_args.extend(objs.iter().map(|o| o.to_str().unwrap().to_string()));
        let status = Command::new("ar")
            .args(&ar_args)
            .status()
            .expect("spawn ar (mmq)");
        assert!(status.success(), "ar failed for {}", lib.display());
        // --whole-archive: keep the CUDART fatbin-registration global ctor so the device kernel
        // registers (same MANDATORY reasoning as the CUTLASS link below).
        println!("cargo:rustc-link-search=native={}", out.display());
        println!("cargo:rustc-link-arg=-Wl,--whole-archive");
        println!("cargo:rustc-link-arg={}", lib.display());
        println!("cargo:rustc-link-arg=-Wl,--no-whole-archive");
        let cuda_lib = std::path::Path::new(&nvcc).parent().and_then(|p| p.parent())
            .map(|p| p.join("lib64")).unwrap_or_else(|| std::path::PathBuf::from("/usr/local/cuda-13.1/lib64"));
        println!("cargo:rustc-link-search=native={}", cuda_lib.display());
        println!("cargo:rustc-link-arg=-Wl,--push-state,--as-needed");
        println!("cargo:rustc-link-arg=-lcudart");
        println!("cargo:rustc-link-arg=-lstdc++");
        println!("cargo:rustc-link-arg=-Wl,--pop-state");
    }

    // ---- CUTLASS sm_120a NVFP4 GEMM: a STATIC LIB (7th artifact, different kind), NOT a fatbin ----
    // CUTLASS needs its host-side GemmUniversalAdapter::run() (host C++), so it cannot go through the
    // fatbin/load_module path above. It is compiled to an object, archived, and whole-archived at link.
    // Additive: the 6-fatbin loop above is byte-for-byte unchanged (the parallel flash_attn.cu FA build
    // is untouched). Guarded by BW24_CUTLASS so the default build is unaffected until Phase 0 lands.
    if std::env::var("BW24_CUTLASS").is_ok() {
        let cutlass_src = "cu/cutlass_fp4_sm120.cu";
        println!("cargo:rerun-if-changed={cutlass_src}");
        println!("cargo:rerun-if-env-changed=BW24_CUTLASS");
        // CUTLASS 4.x header tree (on-box, probe-verified). TODO Phase 1: vendor a pinned tree into the
        // repo for reproducibility rather than pointing at the venv install.
        let cutlass_root = std::env::var("BW24_CUTLASS_ROOT").unwrap_or_else(|_|
            "/home/avifenesh/.venvs/torch/lib/python3.12/site-packages/flashinfer/data/cutlass".into());
        let cutlass_inc = format!("{cutlass_root}/include");
        let cutlass_util = format!("{cutlass_root}/tools/util/include");
        let obj = out.join("cutlass_fp4_sm120.o");
        let lib = out.join("libbw24_cutlass.a");
        let status = Command::new(&nvcc)
            .args([
                "-gencode", "arch=compute_120a,code=sm_120a",
                "-O3", "-std=c++17", "--expt-relaxed-constexpr",
                "-DENABLE_BF16", "-DENABLE_FP4", "-DCUTLASS_ENABLE_GDC_FOR_SM100=1",
                "-I", &cutlass_inc, "-I", &cutlass_util,
                "-c", cutlass_src, "-o", obj.to_str().unwrap(),
            ])
            .status()
            .expect("spawn nvcc (cutlass)");
        assert!(status.success(), "nvcc static-lib build failed for {cutlass_src}");
        let _ = std::fs::remove_file(&lib);
        let status = Command::new("ar")
            .args(["crus", lib.to_str().unwrap(), obj.to_str().unwrap()])
            .status()
            .expect("spawn ar");
        assert!(status.success(), "ar failed for {}", lib.display());
        // --whole-archive is MANDATORY: a plain static link drops the CUDART fatbin-registration global
        // ctor (_ZL24__sti____cudaRegisterAllv in .init_array) -> the device kernel silently never
        // registers -> no-kernel launch failure. Verified on-box (plan §2.2).
        println!("cargo:rustc-link-search=native={}", out.display());
        println!("cargo:rustc-link-arg=-Wl,--whole-archive");
        println!("cargo:rustc-link-arg={}", lib.display());
        println!("cargo:rustc-link-arg=-Wl,--no-whole-archive");
        // libstdc++ AFTER the archive (link-arg, not link-lib) so the function-local-static guard
        // symbols (__cxa_guard_acquire/release, from CUTLASS's tile_atom_to_shape statics) resolve.
        // A plain `link-lib=stdc++` can be ordered before the archive under -nodefaultlibs/lld and
        // leave them undefined for bins other than cutlass-smoke (whole-archive applies to ALL bins).
        // cudart (CUTLASS host adapter uses the runtime API) and stdc++ BOTH as trailing link-args so
        // they sit AFTER the whole-archive; the cudart fatbin-registration ctors + the C++ static
        // guards in cutlass_fp4_sm120.o resolve against them. The CUDA lib dir is needed for -lcudart.
        let cuda_lib = std::path::Path::new(&nvcc).parent().and_then(|p| p.parent())
            .map(|p| p.join("lib64")).unwrap_or_else(|| std::path::PathBuf::from("/usr/local/cuda-13.1/lib64"));
        println!("cargo:rustc-link-search=native={}", cuda_lib.display());
        println!("cargo:rustc-link-arg=-Wl,--push-state,--as-needed");
        println!("cargo:rustc-link-arg=-lcudart");
        println!("cargo:rustc-link-arg=-lstdc++");
        println!("cargo:rustc-link-arg=-Wl,--pop-state");
        // Let the smoke-test bin gate compile out cleanly when CUTLASS is not built.
        println!("cargo:rustc-cfg=bw24_cutlass");
    }
}
