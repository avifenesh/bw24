// Compile engine .cu kernels to the selected CUDA fatbin (same pattern as memra-probe).
use std::path::PathBuf;
use std::process::Command;

/// Arch auto-detection (MEMRA_CUDA_ARCH unset): probe the first GPU's compute capability
/// via nvidia-smi and pick the matching build arch. GPU-less machines (CI compile gate)
/// and unrecognized caps fall back to 120a — the naked build stays the sm_120a build.
/// An explicit MEMRA_CUDA_ARCH always wins (this fn is not called then).
fn detect_arch() -> String {
    let cap = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output().ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.lines().next().map(|l| l.trim().to_string()));
    let arch = match cap.as_deref() {
        Some("12.0") | Some("12.1") => "120a",
        Some("10.0") => "100a",
        Some("9.0") => "90a",
        Some("8.9") => "89",
        _ => "120a",
    };
    match &cap {
        Some(c) => println!("cargo:warning=MEMRA_CUDA_ARCH auto-detected {arch} (compute_cap {c}); set MEMRA_CUDA_ARCH to override"),
        None => println!("cargo:warning=no GPU visible; defaulting MEMRA_CUDA_ARCH=120a (compile-only)"),
    }
    arch.to_string()
}

fn main() {
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    // docs.rs builders have no nvcc and no CUDA libs: emit empty placeholder fatbins so
    // the include_bytes!/env! consts compile, skip every nvcc/ar/link step. The resulting
    // rlib is documentation-only — Engine::new would fail to load a zero-byte module, but
    // docs.rs never runs code. Normal builds (CI included) never set DOCS_RS.
    if std::env::var_os("DOCS_RS").is_some() {
        for stem in ["kernels", "hybrid", "qmatvec", "flash_attn", "qmatvec_gemm",
                     "moe_router", "spec_sample", "flash_attn_vq4", "flash_attn_vf8",
                     "flash_attn_kf8", "flash_attn_kf8vq4", "flash_attn_kf8vf8"] {
            std::fs::write(out.join(format!("{stem}.fatbin")), []).unwrap();
        }
        for (env, stem) in [("MEMRA_ENGINE_FATBIN", "kernels"), ("MEMRA_HYBRID_FATBIN", "hybrid"),
                            ("MEMRA_QMATVEC_FATBIN", "qmatvec"), ("MEMRA_FLASH_FATBIN", "flash_attn"),
                            ("MEMRA_GEMM_FATBIN", "qmatvec_gemm"), ("MEMRA_ROUTER_FATBIN", "moe_router"),
                            ("MEMRA_SAMPLE_FATBIN", "spec_sample"),
                            ("MEMRA_FLASH_FATBIN_VQ4", "flash_attn_vq4"),
                            ("MEMRA_FLASH_FATBIN_VF8", "flash_attn_vf8"),
                            ("MEMRA_FLASH_FATBIN_KF8", "flash_attn_kf8"),
                            ("MEMRA_FLASH_FATBIN_KF8VQ4", "flash_attn_kf8vq4"),
                            ("MEMRA_FLASH_FATBIN_KF8VF8", "flash_attn_kf8vf8")] {
            println!("cargo:rustc-env={env}={}", out.join(format!("{stem}.fatbin")).display());
        }
        println!("cargo:rustc-check-cfg=cfg(memra_portable_cuda)");
        println!("cargo:rustc-check-cfg=cfg(memra_hopper_mma)");
        println!("cargo:rustc-check-cfg=cfg(memra_cutlass)");
        println!("cargo:rustc-env=MEMRA_BUILT_CUDA_ARCH=120a");
        return;
    }

    let nvcc = std::env::var("MEMRA_NVCC").unwrap_or_else(|_| "/usr/local/cuda-13.1/bin/nvcc".into());
    println!("cargo:rerun-if-env-changed=MEMRA_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=MEMRA_CUTLASS");
    println!("cargo:rustc-check-cfg=cfg(memra_portable_cuda)");
    println!("cargo:rustc-check-cfg=cfg(memra_hopper_mma)");
    println!("cargo:rustc-check-cfg=cfg(memra_cutlass)");
    let cuda_arch = std::env::var("MEMRA_CUDA_ARCH").unwrap_or_else(|_| detect_arch());
    assert!(matches!(cuda_arch.as_str(), "120a" | "100a" | "90a" | "89"),
            "MEMRA_CUDA_ARCH must be 120a (default), 100a (B200), 90a (Hopper), or 89 (portable eval)");
    // Hopper boot arch rides the portable-CUDA correctness path: sm_90a SASS, no
    // sm_120a/sm_100a MMA kinds. Tuned wgmma paths are a separate later lane.
    // Phase A (ARCHITECTURE-H100.md): 90a additionally re-enables the portable-PTX
    // tensor-core paths the boot lane gated off — int8 mma.m16n8k32/k16.s8, bf16
    // m16n8k16, ldmatrix, cp.async are sm_80-class and run natively on Hopper.
    // The sm_120a/sm_100a-only MMA kinds (mxf4nvf4, kind::f8f6f4) stay dead: their
    // launchers remain fail-closed stubs on this arch.
    let portable = matches!(cuda_arch.as_str(), "89" | "90a");
    let hopper_mma = cuda_arch == "90a";
    assert!(!(cuda_arch != "120a" && std::env::var_os("MEMRA_CUTLASS").is_some()),
            "MEMRA_CUTLASS is sm_120a-only and cannot be enabled for this CUDA architecture");
    let gencode = format!("arch=compute_{cuda_arch},code=sm_{cuda_arch}");
    if portable {
        println!("cargo:rustc-cfg=memra_portable_cuda");
    }
    if hopper_mma {
        println!("cargo:rustc-cfg=memra_hopper_mma");
    }
    // Runtime arch guard reads this (Engine::new): fatbins are single-arch SASS, so the
    // engine verifies the device's compute capability matches the built arch at init.
    println!("cargo:rustc-env=MEMRA_BUILT_CUDA_ARCH={cuda_arch}");

    for (src, env) in [("cu/kernels.cu", "MEMRA_ENGINE_FATBIN"), ("cu/hybrid.cu", "MEMRA_HYBRID_FATBIN"),
                       ("cu/qmatvec.cu", "MEMRA_QMATVEC_FATBIN"), ("cu/flash_attn.cu", "MEMRA_FLASH_FATBIN"),
                       ("cu/qmatvec_gemm.cu", "MEMRA_GEMM_FATBIN"), ("cu/moe_router.cu", "MEMRA_ROUTER_FATBIN"),
                       ("cu/spec_sample.cu", "MEMRA_SAMPLE_FATBIN")] {
        println!("cargo:rerun-if-changed={src}");
        println!("cargo:rerun-if-changed=cu/wgmma_common.cuh");
        let stem = src.split('/').last().unwrap().trim_end_matches(".cu");
        let fatbin = out.join(format!("{stem}.fatbin"));
        let mut args = vec!["-gencode", &gencode, "-O3", "--fatbin"];
        if portable {
            args.push("-DMEMRA_PORTABLE_CUDA=1");
        }
        if hopper_mma {
            args.push("-DMEMRA_HOPPER_MMA=1");
        }
        // The hand-written mxf4nvf4 block-scale MMA in qmatvec_gemm.cu is an sm_120a
        // instruction encoding. B200 (sm_100a) uses the accuracy-safe NVFP4 W4A8 int8
        // path below instead, so omit only that unused opt-in kernel from its fatbin.
        if cuda_arch == "100a" && src == "cu/qmatvec_gemm.cu" {
            args.push("-DMEMRA_DISABLE_NATIVE_FP4=1");
        }
        // TUNE SEAM: MEMRA_FA_PP_MINBLOCKS sweeps fa_prefill_f32_pp's __launch_bounds__
        // min-blocks (occupancy vs register-spill tradeoff; H100 ncu 2026-07-26).
        println!("cargo:rerun-if-env-changed=MEMRA_FA_PP_MINBLOCKS");
        let fa_mb = std::env::var("MEMRA_FA_PP_MINBLOCKS").ok();
        let fa_mb_arg;
        if let (Some(mb), "cu/flash_attn.cu") = (&fa_mb, src) {
            fa_mb_arg = format!("-DFA_PP_MINBLOCKS={mb}");
            args.push(&fa_mb_arg);
        }
        args.extend(["-o", fatbin.to_str().unwrap(), src]);
        let status = Command::new(&nvcc)
            .args(args)
            .status()
            .expect("spawn nvcc");
        assert!(status.success(), "nvcc fatbin build failed for {src}");
        println!("cargo:rustc-env={env}={}", fatbin.display());
    }

    // ---- KV-format fatbin variants of flash_attn.cu (kvbytes lane, 2026-07-08) ----
    // Same kernels/entry names, compile-time K/V cache format via -D. Engine::new picks the
    // fatbin at runtime from env MEMRA_KV_K / MEMRA_KV_V (lib.rs flash_fatbin_bytes); the default
    // (no env) loads the plain flash_attn.fatbin built above — bit-identical daily config.
    for (suffix, kfmt, vfmt) in [("VQ4", 0, 1), ("VF8", 0, 2), ("KF8", 1, 0),
                                 ("KF8VQ4", 1, 1), ("KF8VF8", 1, 2)] {
        let fatbin = out.join(format!("flash_attn_{}.fatbin", suffix.to_lowercase()));
        let mut args = vec![
            "-gencode".to_string(), gencode.clone(), "-O3".to_string(), "--fatbin".to_string(),
        ];
        if portable {
            args.push("-DMEMRA_PORTABLE_CUDA=1".to_string());
        }
        if hopper_mma {
            args.push("-DMEMRA_HOPPER_MMA=1".to_string());
        }
        args.extend([
            format!("-DMEMRA_KV_KFMT={kfmt}"), format!("-DMEMRA_KV_VFMT={vfmt}"),
            "-o".to_string(), fatbin.to_string_lossy().into_owned(), "cu/flash_attn.cu".to_string(),
        ]);
        let status = Command::new(&nvcc)
            .args(args)
            .status()
            .expect("spawn nvcc (flash_attn kv-format variant)");
        assert!(status.success(), "nvcc fatbin build failed for flash_attn kv variant {suffix}");
        println!("cargo:rustc-env=MEMRA_FLASH_FATBIN_{suffix}={}", fatbin.display());
    }

    // ---- Vendored llama MMQ GEMMs: a STATIC LIB with C-ABI host launchers (extern "C"). ----
    // Same kind as the CUTLASS artifact (a host-side launcher cannot go through the device-only fatbin
    // path), but ALWAYS built (no external header deps — fully ggml-decoupled). The launchers do
    // cudaFuncSetAttribute (>48KB dynamic smem) + the mul_mat_q kernel launch internally.
    // Called from Rust via FFI (mmq_ffi.rs), dispatched behind MEMRA_MMQ=1.
    // Two translation units: mmq_fp4.cu (Blackwell mxf4nvf4 W4A4) and mmq_q45k.cu
    // (Q4_K/Q5_K int8-MMA W4A8, sm_75+ portable). Both archived into one libmemra_mmq.a.
    {
        let mut objs: Vec<PathBuf> = Vec::new();
        // TUNE SEAM: MEMRA_MMQ_X_Q45K=64 rebuilds the k-quant MMQ with a 64-token tile
        // (47KB smem -> 2 CTA/SM vs 57KB/1; the q45k occupancy ceiling found by ncu).
        println!("cargo:rerun-if-env-changed=MEMRA_MMQ_X_Q45K");
        let q45k_x = std::env::var("MEMRA_MMQ_X_Q45K").ok();
        // TUNE SEAM: MEMRA_MMQ_X_Q4=64|96 shrinks the q4_0 MMQ token-tile (same axis as the
        // q45k/w4a8 seams; build-time — the tile is a template constant).
        println!("cargo:rerun-if-env-changed=MEMRA_MMQ_X_Q4");
        let q4_x = std::env::var("MEMRA_MMQ_X_Q4").ok();
        // TUNE SEAM: MEMRA_MMQ_X_W4A8=<n> rebuilds the NVFP4 W4A8 MMQ with an n-token tile.
        // ncu 2026-07-06 (27B pp6257): default 128x128 tile = 61KB smem = 1 CTA/SM ->
        // warps_active 16.7%, tensor pipe 53% — the same occupancy ceiling q45k hit.
        println!("cargo:rerun-if-env-changed=MEMRA_MMQ_X_W4A8");
        let w4a8_x = std::env::var("MEMRA_MMQ_X_W4A8").ok();
        // TUNE SEAM: MEMRA_MMQ_X_IQEXP=<n> rebuilds the expert-segmented MMQ with an n-token
        // tile (the round-45 kernel-rate dig: 128 costs 64 accumulator regs/thread at
        // occupancy 12.5%; smaller tiles trade MMA j-reuse for CTAs/SM).
        println!("cargo:rerun-if-env-changed=MEMRA_MMQ_X_IQEXP");
        let iqexp_x = std::env::var("MEMRA_MMQ_X_IQEXP").ok();
        // TUNE SEAM: MEMRA_MMQ_Y_W4A8=64 halves the row tile AND warp count together (mmq_y =
        // nwarps*16) — 42KB->21KB tile_x, 2 CTA/SM. Unlike MMQ_X, this axis doesn't duplicate
        // weight reads, so it attacks the 16.7%-warps occupancy ceiling for free.
        println!("cargo:rerun-if-env-changed=MEMRA_MMQ_Y_W4A8");
        let w4a8_y = std::env::var("MEMRA_MMQ_Y_W4A8").ok();
        // TUNE SEAM: MEMRA_MMQ_X_FP8=<n> rebuilds the per-block FP8 prefill tile with an n-token
        // tile (it sets the WIDE candidate; the launcher picks between it and FP8_MMQ_X_SMALL per
        // call by wave fill). v1 needed this seam because its 128-token default sat below the Q8_0
        // floor at every 27B shape; v2's restructure made X=256 affordable and it is now the
        // default. Neither arm is MMA-bound at any width measured (110-130 TF against f8f6f4's
        // 381-TF class), so geometry, not the MMA, is what this seam moves.
        //
        // MEMRA_MMQ_Y_FP8 / MEMRA_MMQ_OCC_FP8 / MEMRA_MMQ_PIPE_FP8 are v2 slice-2 and slice-3 seams
        // that CONCLUDED NEGATIVE (research/fp8st-20260804/mmq-v2/RESULTS.jsonl slices 2-3 and
        // experiment B): halving Y splits the same 8 warps across two CTAs rather than raising
        // warps/SM, Y=128 with OCC=2 spills the 64-float accumulator, and cp.async on the weight
        // tile cannot pay while the equally-large activation tile is still a synchronous copy
        // between the issue and the wait. They are kept only as the reproduction path for those
        // rows — per the flags doctrine they are deletion candidates, which is an owner call.
        println!("cargo:rerun-if-env-changed=MEMRA_MMQ_X_FP8");
        println!("cargo:rerun-if-env-changed=MEMRA_MMQ_Y_FP8");
        println!("cargo:rerun-if-env-changed=MEMRA_MMQ_OCC_FP8");
        let fp8_x = std::env::var("MEMRA_MMQ_X_FP8").ok();
        let fp8_y = std::env::var("MEMRA_MMQ_Y_FP8").ok();
        println!("cargo:rerun-if-env-changed=MEMRA_MMQ_PIPE_FP8");
        let fp8_occ = std::env::var("MEMRA_MMQ_OCC_FP8").ok();
        let fp8_pipe = std::env::var("MEMRA_MMQ_PIPE_FP8").ok();
        // fp8_prefill.cu rides the same static-lib kind: a cuBLASLt host launcher + quantize
        // kernels for the MEMRA_PP_FP8 prefill path (runtime-gated; always built — no external
        // header deps beyond the CUDA toolkit, which ships cublasLt).
        for mmq_src in ["cu/mmq_fp4.cu", "cu/mmq_q45k.cu", "cu/mmq_nvfp4_w4a8.cu", "cu/mmq_iq_experts.cu",
                        "cu/mmq_q8_0.cu", "cu/mmq_q4_0.cu", "cu/fp8_prefill.cu", "cu/f16_prefill.cu",
                        "cu/mmq_nvfp4_f8f4.cu", "cu/fa3_prefill.cu", "cu/moe_f16_grouped.cu",
                        "cu/fp8_blk_dequant.cu", "cu/mmq_fp8_blk.cu"] {
            println!("cargo:rerun-if-changed={mmq_src}");
            let compile_src = if cuda_arch != "120a" && mmq_src == "cu/mmq_fp4.cu" {
                // The explicit MEMRA_MMQ=1 W4A4 launcher is sm_120a-only (mxf4nvf4
                // block-scale MMA). Keep its C ABI present on B200 and portable Ada,
                // but make accidental use fail closed; normal evaluation uses the
                // separately compiled W4A8 launcher.
                "cu/mmq_fp4_stub.cu"
            } else if portable && mmq_src == "cu/mmq_nvfp4_w4a8.cu" {
                // The W4A8/F8F4 launchers use .kind::f8f6f4 tile MMA (sm_100a+);
                // sm_89 gets fail-closed ABI stubs.
                "cu/mmq_nvfp4_w4a8_stub.cu"
            } else if portable && mmq_src == "cu/mmq_fp8_blk.cu" {
                // Per-block FP8 MMQ: same .kind::f8f6f4 gate as the W4A8/F8F4 launchers.
                "cu/mmq_fp8_blk_stub.cu"
            } else {
                mmq_src
            };
            println!("cargo:rerun-if-changed={compile_src}");
            let stem = mmq_src.split('/').last().unwrap().trim_end_matches(".cu");
            let obj = out.join(format!("{stem}.o"));
            let mut args: Vec<String> = vec![
                "-gencode".into(), gencode.clone(),
                "-O3".into(), "-std=c++17".into(), "--expt-relaxed-constexpr".into(),
            ];
            if mmq_src.ends_with("mmq_q45k.cu") {
                if let Some(x) = &q45k_x { args.push(format!("-DMMQ_X={x}")); }
            }
            if mmq_src.ends_with("mmq_q4_0.cu") {
                if let Some(x) = &q4_x { args.push(format!("-DMMQ_X={x}")); }
            }
            if mmq_src.ends_with("mmq_nvfp4_w4a8.cu") {
                if let Some(x) = &w4a8_x { args.push(format!("-DMMQ_X={x}")); }
                if let Some(y) = &w4a8_y { args.push(format!("-DMMQ_Y={y}")); }
            }
            if mmq_src.ends_with("mmq_iq_experts.cu") {
                if let Some(x) = &iqexp_x { args.push(format!("-DMMQ_X={x}")); }
            }
            // Per-block FP8 MMQ tile geometry. v2 opens Y (and its paired occupancy target)
            // alongside X: the scale-row hoist only needs FP8_MMQ_Y to DIVIDE the 128 block edge
            // (a static_assert enforces that), not to equal it, so halving Y and nwarps together
            // is legal and is the occupancy lever the v1 profile asked for. FP8_MMQ_OCC is the
            // __launch_bounds__ minBlocksPerMultiprocessor.
            if mmq_src.ends_with("mmq_fp8_blk.cu") {
                if let Some(x) = &fp8_x { args.push(format!("-DFP8_MMQ_X={x}")); }
                if let Some(y) = &fp8_y { args.push(format!("-DFP8_MMQ_Y={y}")); }
                if let Some(o) = &fp8_occ { args.push(format!("-DFP8_MMQ_OCC={o}")); }
                if let Some(p) = &fp8_pipe { args.push(format!("-DFP8_MMQ_PIPE={p}")); }
            }
            if mmq_src.ends_with("fa3_prefill.cu") && cuda_arch != "90a" {
                args.push("-DMEMRA_FA3_STUB".into());
            }
            args.extend(["-c".into(), compile_src.into(), "-o".into(), obj.to_str().unwrap().into()]);
            let status = Command::new(&nvcc)
                .args(&args)
                .status()
                .expect("spawn nvcc (mmq)");
            assert!(status.success(), "nvcc static-lib build failed for {mmq_src}");
            objs.push(obj);
        }
        let lib = out.join("libmemra_mmq.a");
        let _ = std::fs::remove_file(&lib);
        let mut ar_args = vec!["crus".to_string(), lib.to_str().unwrap().to_string()];
        ar_args.extend(objs.iter().map(|o| o.to_str().unwrap().to_string()));
        let status = Command::new("ar")
            .args(&ar_args)
            .status()
            .expect("spawn ar (mmq)");
        assert!(status.success(), "ar failed for {}", lib.display());
        // rustc-link-lib (NOT rustc-link-arg): link-arg applies only to THIS package's own
        // binaries, so downstream crates (memra-server) failed to link the MMQ symbols. link-lib
        // metadata propagates through the dependency graph; +whole-archive keeps the CUDART
        // fatbin-registration global ctor alive (same MANDATORY reasoning as the CUTLASS link).
        println!("cargo:rustc-link-search=native={}", out.display());
        println!("cargo:rustc-link-lib=static:+whole-archive=memra_mmq");
        let cuda_lib = std::path::Path::new(&nvcc).parent().and_then(|p| p.parent())
            .map(|p| p.join("lib64")).unwrap_or_else(|| std::path::PathBuf::from("/usr/local/cuda-13.1/lib64"));
        println!("cargo:rustc-link-search=native={}", cuda_lib.display());
        // dylib link-lib (not link-arg) so cudart/stdc++ propagate to downstream binaries too.
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=stdc++");
        // fp8_prefill.cu calls the cuBLASLt host API directly (same lib64 search path as cudart).
        println!("cargo:rustc-link-lib=dylib=cublasLt");
        // plain cublas: the MoE grouped f16 GEMM (cublasGemmGroupedBatchedEx, CUDA >= 12.5)
        println!("cargo:rustc-link-lib=dylib=cublas");
        // fa3_prefill.cu calls the driver API (cuTensorMapEncodeTiled). GPU-less build
        // machines (CI compile gate, release matrix) have no driver libcuda.so — the
        // toolkit's stubs dir satisfies the link; at runtime ld.so resolves the real
        // libcuda.so.1 from the driver (the stubs dir is not on any runtime path).
        println!("cargo:rustc-link-search=native={}/stubs", cuda_lib.display());
        println!("cargo:rustc-link-lib=dylib=cuda");
    }

    // ---- CUTLASS sm_120a NVFP4 GEMM: a STATIC LIB (7th artifact, different kind), NOT a fatbin ----
    // CUTLASS needs its host-side GemmUniversalAdapter::run() (host C++), so it cannot go through the
    // fatbin/load_module path above. It is compiled to an object, archived, and whole-archived at link.
    // Additive: the 6-fatbin loop above is byte-for-byte unchanged (the parallel flash_attn.cu FA build
    // is untouched). Guarded by MEMRA_CUTLASS so the default build is unaffected until Phase 0 lands.
    if std::env::var("MEMRA_CUTLASS").is_ok() {
        let cutlass_src = "cu/cutlass_fp4_sm120.cu";
        println!("cargo:rerun-if-changed={cutlass_src}");
        // CUTLASS 4.x header tree (on-box, probe-verified). TODO Phase 1: vendor a pinned tree into the
        // repo for reproducibility rather than pointing at the venv install.
        let cutlass_root = std::env::var("MEMRA_CUTLASS_ROOT").unwrap_or_else(|_|
            "/home/avifenesh/.venvs/torch/lib/python3.12/site-packages/flashinfer/data/cutlass".into());
        let cutlass_inc = format!("{cutlass_root}/include");
        let cutlass_util = format!("{cutlass_root}/tools/util/include");
        let obj = out.join("cutlass_fp4_sm120.o");
        let lib = out.join("libmemra_cutlass.a");
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
        // dylib link-lib (not link-arg) so cudart/stdc++ propagate to downstream binaries too.
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=stdc++");
        // Let the smoke-test bin gate compile out cleanly when CUTLASS is not built.
        println!("cargo:rustc-cfg=memra_cutlass");
    }
}
