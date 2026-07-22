use std::ffi::{CStr, CString, c_char, c_void};

use bw24_gguf::{GgmlType, dequant};

type DotFn = unsafe extern "C" fn(
    i32,
    *const u8,
    usize,
    *const f32,
    i32,
    *mut f32,
    *mut c_char,
    usize,
) -> i32;
type AbiVersionFn = unsafe extern "C" fn() -> u32;
type StatsFn = unsafe extern "C" fn(*mut u64, *mut u64, *mut u64, *mut u64);

#[repr(C)]
#[derive(Clone, Copy)]
struct Projection {
    weights: *const u8,
    qtype: i32,
    in_features: i32,
    out_features: i32,
    row_bytes: usize,
    byte_len: usize,
    file_fd: i32,
    file_offset: u64,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Expert {
    gate: Projection,
    up: Projection,
    down: Projection,
    route_weight: f32,
}

type MoeFn =
    unsafe extern "C" fn(*const Expert, i32, *const f32, *mut f32, i32, *mut c_char, usize) -> i32;

unsafe fn required_symbol<T: Copy>(
    handle: *mut c_void,
    name: &CStr,
) -> Result<T, Box<dyn std::error::Error>> {
    let symbol = unsafe { libc::dlsym(handle, name.as_ptr()) };
    if symbol.is_null() {
        return Err(format!("native companion is missing {}", name.to_string_lossy()).into());
    }
    Ok(unsafe { std::mem::transmute_copy::<*mut c_void, T>(&symbol) })
}

fn qtype(ty: GgmlType) -> i32 {
    match ty {
        GgmlType::Q8_0 => 0,
        GgmlType::Q4_K => 1,
        GgmlType::Q6_K => 2,
        GgmlType::Q5_K => 3,
        GgmlType::Q3_K => 4,
        GgmlType::IQ4_XS => 5,
        GgmlType::IQ3_S => 6,
        GgmlType::NVFP4 => 7,
        GgmlType::F32 => 8,
        GgmlType::BF16 => 11,
        GgmlType::Q4_0 => 12,
        GgmlType::Q2_K => 13,
        other => panic!("no native qtype for {other:?}"),
    }
}

fn write_half(raw: &mut [u8], offset: usize, value: u16) {
    raw[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn fixture(ty: GgmlType, n: usize) -> Vec<u8> {
    let (block, type_size) = ty.block_and_type_size();
    let mut raw = vec![0u8; n / block as usize * type_size as usize];
    let mut state = 0x1234_5678u32;
    for byte in &mut raw {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *byte = (state >> 24) as u8;
    }
    match ty {
        GgmlType::F32 => {
            for index in 0..n {
                let value = ((index as f32) * 0.17).sin() * 0.25;
                raw[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
            }
        }
        GgmlType::BF16 => {
            for index in 0..n {
                let value = ((index as f32) * 0.17).sin() * 0.25;
                raw[index * 2..index * 2 + 2]
                    .copy_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
            }
        }
        GgmlType::Q8_0 | GgmlType::Q4_0 => {
            let bytes = type_size as usize;
            for block in raw.chunks_exact_mut(bytes) {
                write_half(block, 0, 0x3800);
            }
        }
        GgmlType::Q2_K => {
            for block in raw.chunks_exact_mut(84) {
                write_half(block, 80, 0x3400);
                write_half(block, 82, 0x3000);
            }
        }
        GgmlType::Q4_K => {
            for block in raw.chunks_exact_mut(144) {
                write_half(block, 0, 0x3000);
                write_half(block, 2, 0x2c00);
            }
        }
        GgmlType::Q5_K => {
            for block in raw.chunks_exact_mut(176) {
                write_half(block, 0, 0x3000);
                write_half(block, 2, 0x2c00);
            }
        }
        GgmlType::Q6_K => {
            for block in raw.chunks_exact_mut(210) {
                write_half(block, 208, 0x3000);
            }
        }
        GgmlType::Q3_K | GgmlType::IQ3_S => {
            for block in raw.chunks_exact_mut(110) {
                let offset = if ty == GgmlType::Q3_K { 108 } else { 0 };
                write_half(block, offset, 0x3000);
            }
        }
        GgmlType::IQ4_XS => {
            for block in raw.chunks_exact_mut(136) {
                write_half(block, 0, 0x3000);
            }
        }
        GgmlType::NVFP4 => {
            for block in raw.chunks_exact_mut(36) {
                block[..4].fill(0x3f);
            }
        }
        other => panic!("no fixture for {other:?}"),
    }
    raw
}

fn quantized_activation(input: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0f32; input.len()];
    for (source, destination) in input.chunks_exact(16).zip(output.chunks_exact_mut(16)) {
        let absolute_max = source
            .iter()
            .fold(0.0f32, |value, item| value.max(item.abs()));
        let scale = if absolute_max == 0.0 {
            0.0
        } else {
            absolute_max / 127.0
        };
        let inverse = if scale == 0.0 { 0.0 } else { 1.0 / scale };
        for (input, output) in source.iter().zip(destination) {
            let quantized = (input * inverse).round_ties_even().clamp(-127.0, 127.0);
            *output = quantized * scale;
        }
    }
    output
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::var("BW24_CPU_EXPERT_LIB")?;
    let c_path = CString::new(path.clone())?;
    let handle = unsafe { libc::dlopen(c_path.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
    if handle.is_null() {
        let error = unsafe { CStr::from_ptr(libc::dlerror()) }.to_string_lossy();
        return Err(format!("cannot load {path}: {error}").into());
    }
    let abi: AbiVersionFn = unsafe { required_symbol(handle, c"bw24_cpu_experts_abi_version")? };
    let dot: DotFn = unsafe { required_symbol(handle, c"bw24_cpu_dot_v2")? };
    let moe: MoeFn = unsafe { required_symbol(handle, c"bw24_cpu_moe_token_v2")? };
    let _cache_stats: StatsFn =
        unsafe { required_symbol(handle, c"bw24_cpu_expert_cache_stats_v2")? };
    let _profile_stats: StatsFn =
        unsafe { required_symbol(handle, c"bw24_cpu_expert_profile_stats_v2")? };
    let version = unsafe { abi() };
    if version != 2 {
        return Err(format!("native companion reported ABI {version}, expected 2").into());
    }

    let types = [
        GgmlType::F32,
        GgmlType::BF16,
        GgmlType::Q8_0,
        GgmlType::Q4_0,
        GgmlType::Q2_K,
        GgmlType::Q3_K,
        GgmlType::Q4_K,
        GgmlType::Q5_K,
        GgmlType::Q6_K,
        GgmlType::IQ3_S,
        GgmlType::IQ4_XS,
        GgmlType::NVFP4,
    ];
    for n in [256, 1536, 4096] {
        let input: Vec<f32> = (0..n)
            .map(|index| 0.1 + 1.7 * ((index as f32) * 0.31).cos())
            .collect();
        let quantized = quantized_activation(&input);
        for ty in types {
            let raw = fixture(ty, n);
            let weights = dequant::dequantize(ty, &raw, n);
            let reference_input = if matches!(ty, GgmlType::F32 | GgmlType::BF16) {
                &input
            } else {
                &quantized
            };
            let expected = weights
                .iter()
                .zip(reference_input)
                .fold(0.0f32, |sum, (weight, input)| sum + weight * input);
            let mut actual = 0.0f32;
            let mut error = vec![0i8; 512];
            let status = unsafe {
                dot(
                    qtype(ty),
                    raw.as_ptr(),
                    raw.len(),
                    input.as_ptr(),
                    n as i32,
                    &mut actual,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                let message = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
                return Err(format!("{ty:?}/{n}: native dot failed: {message}").into());
            }
            let absolute = (actual - expected).abs();
            let relative = absolute / expected.abs().max(1.0);
            println!(
                "{ty:?}/{n}: expected={expected:.8} actual={actual:.8} abs={absolute:.3e} rel={relative:.3e}"
            );
            if absolute > 2.0e-3 && relative > 2.0e-5 {
                return Err(
                    format!("{ty:?}/{n}: native dot diverged from bw24 dequant oracle").into(),
                );
            }
        }
    }

    let gate_row = fixture(GgmlType::Q2_K, 256);
    let mut up_row = gate_row.clone();
    let mut down_row = gate_row.clone();
    up_row[16] ^= 0x55;
    down_row[17] ^= 0xaa;
    let gate_weights = gate_row.repeat(256);
    let up_weights = up_row.repeat(256);
    let down_weights = down_row.repeat(256);
    let gate_projection = Projection {
        weights: gate_weights.as_ptr(),
        qtype: qtype(GgmlType::Q2_K),
        in_features: 256,
        out_features: 256,
        row_bytes: gate_row.len(),
        byte_len: gate_weights.len(),
        file_fd: -1,
        file_offset: 0,
        scale: 0.5,
    };
    let up_projection = Projection {
        weights: up_weights.as_ptr(),
        row_bytes: up_row.len(),
        byte_len: up_weights.len(),
        scale: 0.25,
        ..gate_projection
    };
    let down_projection = Projection {
        weights: down_weights.as_ptr(),
        row_bytes: down_row.len(),
        byte_len: down_weights.len(),
        scale: 0.75,
        ..gate_projection
    };
    let expert = Expert {
        gate: gate_projection,
        up: up_projection,
        down: down_projection,
        route_weight: 0.125,
    };
    let smoke_input: Vec<f32> = (0..256)
        .map(|index| 0.01 * ((index as f32) * 0.07).sin())
        .collect();
    let mut smoke_output = vec![f32::NAN; 256];
    let mut error = vec![0i8; 512];
    let status = unsafe {
        moe(
            &expert,
            1,
            smoke_input.as_ptr(),
            smoke_output.as_mut_ptr(),
            1,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    if status != 0 || smoke_output.iter().any(|value| !value.is_finite()) {
        let message = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
        return Err(format!("native production-MoE smoke failed: {message}").into());
    }
    let row_dot = |row: &[u8], activation: &[f32]| {
        dequant::dequantize(GgmlType::Q2_K, row, activation.len())
            .iter()
            .zip(quantized_activation(activation))
            .fold(0.0f32, |sum, (weight, input)| sum + weight * input)
    };
    let gate = row_dot(&gate_row, &smoke_input) * gate_projection.scale;
    let up = row_dot(&up_row, &smoke_input) * up_projection.scale;
    let activation = vec![(gate / (1.0 + (-gate).exp())) * up; 256];
    let expected = row_dot(&down_row, &activation) * down_projection.scale * expert.route_weight;
    for (index, actual) in smoke_output.iter().copied().enumerate() {
        let absolute = (actual - expected).abs();
        let relative = absolute / expected.abs().max(1.0);
        if absolute > 2.0e-3 && relative > 2.0e-5 {
            return Err(format!(
                "native production-MoE output {index} diverged: expected={expected} actual={actual}"
            )
            .into());
        }
    }
    let invalid_projection = Projection {
        weights: gate_weights.as_ptr(),
        qtype: qtype(GgmlType::F32),
        in_features: 255,
        out_features: 255,
        row_bytes: 255 * std::mem::size_of::<f32>(),
        byte_len: 255 * 255 * std::mem::size_of::<f32>(),
        file_fd: -1,
        file_offset: 0,
        scale: 1.0,
    };
    let invalid_expert = Expert {
        gate: invalid_projection,
        up: invalid_projection,
        down: invalid_projection,
        route_weight: 1.0,
    };
    let status = unsafe {
        moe(
            &invalid_expert,
            1,
            smoke_input.as_ptr(),
            smoke_output.as_mut_ptr(),
            1,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    if status == 0 {
        return Err("native production MoE accepted dimensions not divisible by 16".into());
    }

    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut input = smoke_input.clone();
        input[0] = invalid;
        let status = unsafe {
            dot(
                qtype(GgmlType::Q2_K),
                gate_row.as_ptr(),
                gate_row.len(),
                input.as_ptr(),
                input.len() as i32,
                smoke_output.as_mut_ptr(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status == 0 {
            return Err("native dot accepted a non-finite activation".into());
        }
    }
    let subnormal_input = vec![f32::from_bits(1); 256];
    let status = unsafe {
        dot(
            qtype(GgmlType::Q2_K),
            gate_row.as_ptr(),
            gate_row.len(),
            subnormal_input.as_ptr(),
            subnormal_input.len() as i32,
            smoke_output.as_mut_ptr(),
            error.as_mut_ptr(),
            error.len(),
        )
    };
    if status != 0 || !smoke_output[0].is_finite() {
        return Err("native dot mishandled finite subnormal activations".into());
    }

    // File-backed identity: the same multi-expert MoE token served through file descriptors
    // (asynchronous pipelined reads on first touch, RAM cache hits on second) must match the
    // in-memory-weights result bit for bit.
    {
        let n_experts = 4usize;
        let mut blobs: Vec<Vec<u8>> = Vec::with_capacity(n_experts * 3);
        for expert_index in 0..n_experts {
            for projection_index in 0..3usize {
                let mut row = gate_row.clone();
                row[7] ^= (0x11 * (expert_index as u8 + 1)) ^ (0x40 >> projection_index);
                blobs.push(row.repeat(256));
            }
        }
        let mut file_bytes: Vec<u8> = Vec::new();
        let mut offsets: Vec<u64> = Vec::with_capacity(blobs.len());
        for blob in &blobs {
            offsets.push(file_bytes.len() as u64);
            file_bytes.extend_from_slice(blob);
        }
        // BW24_CPU_FILE_IDENTITY_KEEP names a persistent fixture path (written only if
        // absent, never removed) so shm-cache warm restarts can be exercised across
        // processes; default remains an unlinked per-process temp file.
        let keep_path = std::env::var_os("BW24_CPU_FILE_IDENTITY_KEEP");
        let path = match &keep_path {
            Some(p) => std::path::PathBuf::from(p),
            None => std::env::temp_dir().join(format!(
                "bw24-cpu-file-identity-{}.bin",
                std::process::id()
            )),
        };
        if keep_path.is_none() || !path.exists() {
            std::fs::write(&path, &file_bytes)?;
        }
        let file = std::fs::File::open(&path)?;
        if keep_path.is_none() {
            std::fs::remove_file(&path)?;
        }
        let fd = {
            use std::os::unix::io::AsRawFd;
            file.as_raw_fd()
        };
        let projection = |weights: *const u8, fd: i32, offset: u64, scale: f32| Projection {
            weights,
            qtype: qtype(GgmlType::Q2_K),
            in_features: 256,
            out_features: 256,
            row_bytes: gate_row.len(),
            byte_len: gate_row.len() * 256,
            file_fd: fd,
            file_offset: offset,
            scale,
        };
        let scales = [0.5f32, 0.25, 0.75];
        let route_weights = [0.125f32, 0.375, -0.25, 0.5];
        let build_experts = |file_backed: bool| -> Vec<Expert> {
            (0..n_experts)
                .map(|expert_index| {
                    let projection_at = |projection_index: usize| {
                        let blob_index = expert_index * 3 + projection_index;
                        if file_backed {
                            projection(
                                std::ptr::null(),
                                fd,
                                offsets[blob_index],
                                scales[projection_index],
                            )
                        } else {
                            projection(
                                blobs[blob_index].as_ptr(),
                                -1,
                                0,
                                scales[projection_index],
                            )
                        }
                    };
                    Expert {
                        gate: projection_at(0),
                        up: projection_at(1),
                        down: projection_at(2),
                        route_weight: route_weights[expert_index],
                    }
                })
                .collect()
        };
        let run = |experts: &[Expert]| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
            let mut output = vec![f32::NAN; 256];
            let mut error = vec![0i8; 512];
            let status = unsafe {
                moe(
                    experts.as_ptr(),
                    experts.len() as i32,
                    smoke_input.as_ptr(),
                    output.as_mut_ptr(),
                    8,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                let message = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
                return Err(format!("file-backed MoE call failed: {message}").into());
            }
            Ok(output)
        };
        let memory_output = run(&build_experts(false))?;
        let file_experts = build_experts(true);
        let cold_output = run(&file_experts)?;
        let warm_output = run(&file_experts)?;
        for (label, output) in [("cold", &cold_output), ("warm", &warm_output)] {
            for (index, (actual, expected)) in
                output.iter().zip(memory_output.iter()).enumerate()
            {
                if actual.to_bits() != expected.to_bits() {
                    return Err(format!(
                        "file-backed {label} MoE output {index} not bit-identical: \
                         expected={expected} actual={actual}"
                    )
                    .into());
                }
            }
        }
        println!("file-backed pipelined-read MoE identity (cold + warm): PASS");
    }

    println!("ABI v2 symbols, production MoE, and non-finite/subnormal checks: PASS");

    if std::env::var_os("BW24_CPU_NATIVE_BENCH").is_some() {
        let bench_type = match std::env::var("BW24_CPU_NATIVE_BENCH_QTYPE")
            .unwrap_or_else(|_| "Q2_K".to_string())
            .as_str()
        {
            "Q8_0" => GgmlType::Q8_0,
            "Q4_0" => GgmlType::Q4_0,
            "Q2_K" => GgmlType::Q2_K,
            "Q3_K" => GgmlType::Q3_K,
            "Q4_K" => GgmlType::Q4_K,
            "Q5_K" => GgmlType::Q5_K,
            "Q6_K" => GgmlType::Q6_K,
            "IQ3_S" => GgmlType::IQ3_S,
            "IQ4_XS" => GgmlType::IQ4_XS,
            "NVFP4" => GgmlType::NVFP4,
            other => return Err(format!("unsupported benchmark qtype {other}").into()),
        };
        let gate_row = fixture(bench_type, 4096);
        let down_row = fixture(bench_type, 1536);
        let gate_weights = gate_row.repeat(1536);
        let down_weights = down_row.repeat(4096);
        let gate = Projection {
            weights: gate_weights.as_ptr(),
            qtype: qtype(bench_type),
            in_features: 4096,
            out_features: 1536,
            row_bytes: gate_row.len(),
            byte_len: gate_weights.len(),
            file_fd: -1,
            file_offset: 0,
            scale: 1.0,
        };
        let down = Projection {
            weights: down_weights.as_ptr(),
            qtype: qtype(bench_type),
            in_features: 1536,
            out_features: 4096,
            row_bytes: down_row.len(),
            byte_len: down_weights.len(),
            file_fd: -1,
            file_offset: 0,
            scale: 1.0,
        };
        let experts = vec![
            Expert {
                gate,
                up: gate,
                down,
                route_weight: 0.25,
            };
            4
        ];
        let input: Vec<f32> = (0..4096)
            .map(|index| 0.1 + 1.7 * ((index as f32) * 0.31).cos())
            .collect();
        let mut output = vec![0.0f32; 4096];
        let mut error = vec![0i8; 512];
        let threads = std::env::var("BW24_CPU_NATIVE_BENCH_THREADS")
            .ok()
            .and_then(|value| value.parse::<i32>().ok())
            .unwrap_or(8);
        let iterations = 100;
        for _ in 0..10 {
            let status = unsafe {
                moe(
                    experts.as_ptr(),
                    experts.len() as i32,
                    input.as_ptr(),
                    output.as_mut_ptr(),
                    threads,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                let message = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
                return Err(format!("native MoE benchmark failed: {message}").into());
            }
        }
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            unsafe {
                moe(
                    experts.as_ptr(),
                    experts.len() as i32,
                    input.as_ptr(),
                    output.as_mut_ptr(),
                    threads,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
        }
        let elapsed = start.elapsed().as_secs_f64();
        println!(
            "native {bench_type:?} 4-expert 4096x1536x4096 t={threads}: {:.3} ms/token checksum={:.8}",
            elapsed * 1_000.0 / iterations as f64,
            output.iter().sum::<f32>()
        );
    }
    unsafe { libc::dlclose(handle) };
    println!("bw24 native CPU quant check: ALL GREEN");
    Ok(())
}
