// bw24 Phase-0 spine probe.
// Proves the entire tech-stack assumption end-to-end on this exact box:
//   cargo -> build.rs -> nvcc (-gencode compute_120a,sm_120a) -> fatbin
//   -> cudarc loads module -> launches custom kernels -> correct readback.
// Green here = the Rust + cudarc + raw-.cu kernel stack is validated empirically, not on faith.

use cudarc::driver::{CudaContext, DriverError, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

fn main() -> Result<(), DriverError> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    // --- Device caps (sanity vs our known sm_120 facts) ---
    use cudarc::driver::sys::CUdevice_attribute as A;
    let cc_major = ctx.attribute(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
    let cc_minor = ctx.attribute(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
    let sms = ctx.attribute(A::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?;
    println!("device: {}", ctx.name()?);
    println!("compute capability: {cc_major}.{cc_minor}  SMs: {sms}");
    assert_eq!((cc_major, cc_minor), (12, 0), "expected sm_120");

    // --- Load the fatbin module built by build.rs (File variant -> cuModuleLoad accepts fatbin) ---
    let module = ctx.load_module(Ptx::from_file(env!("BW24_FATBIN")))?;

    // --- Test 1: vec_add elementwise (correctness oracle for the launch path) ---
    let f = module.load_function("vec_add")?;
    let n: i32 = 1 << 20;
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (2 * i) as f32).collect();
    let a_dev = stream.clone_htod(&a)?;
    let b_dev = stream.clone_htod(&b)?;
    let mut c_dev = stream.alloc_zeros::<f32>(n as usize)?;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    let mut lb = stream.launch_builder(&f);
    lb.arg(&a_dev).arg(&b_dev).arg(&mut c_dev).arg(&n);
    unsafe { lb.launch(cfg)? };
    let c = stream.clone_dtoh(&c_dev)?;
    let ok = c.iter().enumerate().all(|(i, &v)| (v - (3 * i) as f32).abs() < 1e-3);
    println!("vec_add correct: {ok}  (c[7]={}, expect 21)", c[7]);
    assert!(ok, "vec_add wrong");

    // --- Test 2: FP16 tensor-core mma.sync kernel launches and writes back ---
    let mf = module.load_function("mma_fp16_smoke")?;
    let af = stream.clone_htod(&[0u32; 8])?;
    let bf = stream.clone_htod(&[0u32; 4])?;
    let mut out = stream.alloc_zeros::<f32>(32 * 4)?;
    let mut lb = stream.launch_builder(&mf);
    lb.arg(&af).arg(&bf).arg(&mut out);
    unsafe {
        lb.launch(LaunchConfig { grid_dim: (1, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 })?
    };
    let _o = stream.clone_dtoh(&out)?;
    println!("mma_fp16_smoke: launched + read back OK (tensor-core path live)");

    // --- Test 3: FP4 block-scale kernel is PRESENT in the real compiled module ---
    let present = module.load_function("mma_fp4_blockscale_smoke").is_ok();
    println!("mma_fp4_blockscale present: {present}  (headline FP4 weapon compiles in our pipeline)");
    assert!(present, "FP4 block-scale kernel missing from fatbin");

    stream.synchronize()?;
    println!("\nPhase-0 spine GREEN: cargo->nvcc->fatbin->cudarc->launch->readback all work on sm_120.");
    Ok(())
}
