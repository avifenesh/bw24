//! Phase-0 CUTLASS NVFP4 GEMM smoke test (the make-or-break gate).
//!
//! Proves: (1) the CUTLASS static lib links + the extern "C" FFI resolves; (2) the cudarc driver-API
//! context and the CUTLASS runtime-API host adapter share the SAME primary context (a real GEMM
//! launches and the explicit workspace is honored — no internal cudaMalloc fight); (3) cudaGetLastError()
//! == 0 after the launch; (4) the result is numerically correct vs a CPU reference computed from the
//! exact round-tripped NVFP4 operands.
//!
//! Build/run: BW24_CUTLASS=1 cargo run --release -p bw24-engine --bin cutlass_smoke
//! Compiled out (prints a skip notice) when BW24_CUTLASS is not set at build time.

#[cfg(not(bw24_cutlass))]
fn main() {
    eprintln!("cutlass_smoke: built without BW24_CUTLASS — rebuild with BW24_CUTLASS=1 to run the gate.");
    std::process::exit(2);
}

#[cfg(bw24_cutlass)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use bw24_engine::Engine;
    let eng = Engine::new(0)?;

    // The prefill shape family the gate cares about (plan §1): square ref + the qwen35-9b prefill GEMMs.
    // D[m,n] = A[m,k] @ B[n,k]^T. m=512 is the binding prefill shape.
    let shapes: &[(usize, usize, usize, &str)] = &[
        (256, 256, 256, "square-256 (sanity)"),
        (512, 4096, 4096, "attn_proj m=512"),
        (512, 12288, 4096, "ffn_gate/up m=512"),
        (512, 4096, 12288, "ffn_down m=512"),
    ];
    let mut all_ok = true;
    for &(m, n, k, name) in shapes {
        let (ok, le, rl2) = run_shape(&eng, m, n, k)?;
        println!("[{name}] m={m} n={n} k={k}: cudaGetLastError={le} rel_L2={rl2:.3e} -> {}",
                 if ok { "PASS" } else { "FAIL" });
        all_ok &= ok;
    }
    if all_ok {
        println!("SMOKE PASS: all shapes cudaGetLastError==0 and rel_L2 < 1e-3");
        Ok(())
    } else {
        Err("SMOKE FAIL: one or more shapes failed".into())
    }
}

#[cfg(bw24_cutlass)]
fn run_shape(eng: &bw24_engine::Engine, m: usize, n: usize, k: usize)
             -> Result<(bool, i32, f64), Box<dyn std::error::Error>> {
    use cudarc::driver::DevicePtr;

    // ---- host operands: small deterministic values in [-1,1] (well inside E2M1 max=6) ----
    let mut a_h = vec![0f32; m * k];
    let mut b_h = vec![0f32; n * k];
    for i in 0..m * k { a_h[i] = (((i * 1103515245 + 12345) % 2001) as f32 / 1000.0) - 1.0; }
    for i in 0..n * k { b_h[i] = (((i * 1664525 + 1013904223) % 2001) as f32 / 1000.0) - 1.0; }
    let a_d = eng.htod(&a_h)?;
    let b_d = eng.htod(&b_h)?;

    // ---- NVFP4 quantize both operands (oracle, CUTLASS dtype ctors) ----
    let mut a_packed = eng.alloc_u8(m * k / 2)?;
    let mut a_sf_lin = eng.alloc_u8(m * k / 16)?;
    let mut b_packed = eng.alloc_u8(n * k / 2)?;
    let mut b_sf_lin = eng.alloc_u8(n * k / 16)?;
    eng.cutlass_nvfp4_quant_ref(&a_d, &mut a_packed, &mut a_sf_lin, m, k)?;
    eng.cutlass_nvfp4_quant_ref(&b_d, &mut b_packed, &mut b_sf_lin, n, k)?;

    // ---- scatter linear scales -> CUTLASS swizzled SF layout ----
    let sfa_bytes = eng.cutlass_sfa_size(m, k);
    let sfb_bytes = eng.cutlass_sfb_size(n, k);
    let mut a_sf_sw = eng.alloc_u8(sfa_bytes)?;
    let mut b_sf_sw = eng.alloc_u8(sfb_bytes)?;
    eng.cutlass_repack_sfa(&a_sf_lin, &mut a_sf_sw, m, k)?;
    eng.cutlass_repack_sfb(&b_sf_lin, &mut b_sf_sw, n, k)?;

    // ---- alpha = 1.0 (epilogue scalar lives in device memory) ----
    let alpha_d = eng.htod(&[1.0f32])?;

    // ---- workspace sized via the host-only query (explicit buffer; CUTLASS never internal-mallocs) ----
    let ws_bytes = eng.cutlass_fp4_workspace_size(m, n, k);
    let mut workspace = eng.alloc_u8(ws_bytes.max(1))?;

    // ---- run the GEMM ----
    let mut d_d = eng.htod(&vec![0f32; m * n])?;
    eng.cutlass_fp4_gemm_raw(&a_packed, &b_packed, &a_sf_sw, &b_sf_sw, &alpha_d,
                             &mut d_d, m, n, k, &mut workspace)?;

    // ---- the driver/runtime context interop check: cudaGetLastError() == 0 ----
    eng.stream().synchronize()?;
    let last_err = unsafe {
        let stream = eng.stream();
        let (_p, _g) = workspace.device_ptr(stream);     // keep ctx live
        cuda_get_last_error()
    };

    // ---- CPU reference from the EXACT round-tripped NVFP4 operands ----
    let mut a_rt_d = eng.htod(&vec![0f32; m * k])?;
    let mut b_rt_d = eng.htod(&vec![0f32; n * k])?;
    eng.cutlass_nvfp4_dequant_ref(&a_packed, &a_sf_lin, &mut a_rt_d, m, k)?;
    eng.cutlass_nvfp4_dequant_ref(&b_packed, &b_sf_lin, &mut b_rt_d, n, k)?;
    let a_rt = eng.dtoh(&a_rt_d)?;
    let b_rt = eng.dtoh(&b_rt_d)?;
    let d_ref = bw24_runtime::cpu_linear(&a_rt, &b_rt, m, k, n);  // y = A @ B^T

    let d_gpu = eng.dtoh(&d_d)?;

    // ---- compare ----
    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut ref_norm = 0f64;
    let mut diff_norm = 0f64;
    for i in 0..m * n {
        let dr = d_ref[i];
        let dg = d_gpu[i];
        let ad = (dr - dg).abs();
        if ad > max_abs { max_abs = ad; }
        let rel = ad / (dr.abs().max(1e-3));
        if rel > max_rel { max_rel = rel; }
        ref_norm += (dr as f64) * (dr as f64);
        diff_norm += (ad as f64) * (ad as f64);
    }
    let rel_l2 = (diff_norm.sqrt()) / (ref_norm.sqrt().max(1e-9));
    let _ = max_abs; let _ = max_rel;
    // Gate: f32 epilogue, identical round-tripped operands -> should be ~accumulation-order noise.
    let ok = last_err == 0 && rel_l2 < 1e-3;
    Ok((ok, last_err, rel_l2))
}

#[cfg(bw24_cutlass)]
unsafe extern "C" { fn cudaGetLastError() -> i32; }
#[cfg(bw24_cutlass)]
unsafe fn cuda_get_last_error() -> i32 { cudaGetLastError() }
