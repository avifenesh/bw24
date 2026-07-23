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
        let run_with_input = |experts: &[Expert],
                              input: &[f32]|
         -> Result<Vec<f32>, Box<dyn std::error::Error>> {
            let mut output = vec![f32::NAN; 256];
            let mut error = vec![0i8; 512];
            let status = unsafe {
                moe(
                    experts.as_ptr(),
                    experts.len() as i32,
                    input.as_ptr(),
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
        let run = |experts: &[Expert]| run_with_input(experts, &smoke_input);
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

        // Detached speculative prefetch (optional symbol; older companions skip): prefetch a
        // never-touched file-backed expert, poll cache stats until its bytes land, then the
        // MoE call over it must still be bit-identical to the in-memory result.
        type PrefetchFn = unsafe extern "C" fn(*const Projection, i32, *mut i8, usize) -> i32;
        let prefetch: Option<PrefetchFn> = unsafe {
            let symbol = libc::dlsym(handle, c"bw24_cpu_expert_prefetch_v2".as_ptr());
            if symbol.is_null() { None } else { Some(std::mem::transmute(symbol)) }
        };
        if let Some(prefetch) = prefetch {
            let cache_stats: StatsFn =
                unsafe { required_symbol(handle, c"bw24_cpu_expert_cache_stats_v2")? };
            let mut extra_blobs: Vec<Vec<u8>> = Vec::new();
            for projection_index in 0..3usize {
                let mut row = gate_row.clone();
                row[9] ^= 0x77 ^ (projection_index as u8);
                extra_blobs.push(row.repeat(256));
            }
            let prefetch_path = std::env::temp_dir().join(format!(
                "bw24-cpu-prefetch-{}.bin",
                std::process::id()
            ));
            let mut extra_offsets = Vec::new();
            let mut prefetch_bytes: Vec<u8> = Vec::new();
            for blob in &extra_blobs {
                extra_offsets.push(prefetch_bytes.len() as u64);
                prefetch_bytes.extend_from_slice(blob);
            }
            std::fs::write(&prefetch_path, &prefetch_bytes)?;
            let file = std::fs::File::open(&prefetch_path)?;
            std::fs::remove_file(&prefetch_path)?;
            let fd = {
                use std::os::unix::io::AsRawFd;
                file.as_raw_fd()
            };
            let projections: Vec<Projection> = (0..3)
                .map(|i| projection(std::ptr::null(), fd, extra_offsets[i], scales[i]))
                .collect();
            let mut error = vec![0i8; 512];
            let submitted = unsafe {
                prefetch(projections.as_ptr(), 3, error.as_mut_ptr(), error.len())
            };
            if submitted != 3 {
                let message = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
                return Err(format!(
                    "prefetch submitted {submitted}/3: {message}"
                )
                .into());
            }
            let prefetch_stats: StatsFn = {
                let symbol = unsafe {
                    libc::dlsym(handle, c"bw24_cpu_expert_prefetch_stats_v2".as_ptr())
                };
                if symbol.is_null() {
                    // Annex-era companions always export the stats symbol alongside prefetch.
                    let _ = cache_stats;
                    return Err("prefetch symbol present but prefetch stats symbol absent".into());
                }
                unsafe { std::mem::transmute(symbol) }
            };
            let mut landed = false;
            for _ in 0..200 {
                std::thread::sleep(std::time::Duration::from_millis(10));
                let (mut submitted, mut inflight) = (0u64, 0u64);
                unsafe {
                    prefetch_stats(&mut submitted, std::ptr::null_mut(),
                                   std::ptr::null_mut(), &mut inflight);
                }
                if submitted >= 3 && inflight == 0 {
                    landed = true;
                    break;
                }
            }
            if !landed {
                return Err("prefetched projections never completed".into());
            }
            let prefetched_expert = Expert {
                gate: projections[0],
                up: projections[1],
                down: projections[2],
                route_weight: 0.25,
            };
            let memory_expert = Expert {
                gate: projection(extra_blobs[0].as_ptr(), -1, 0, scales[0]),
                up: projection(extra_blobs[1].as_ptr(), -1, 0, scales[1]),
                down: projection(extra_blobs[2].as_ptr(), -1, 0, scales[2]),
                route_weight: 0.25,
            };
            let from_prefetch = run(std::slice::from_ref(&prefetched_expert))?;
            let from_memory = run(std::slice::from_ref(&memory_expert))?;
            for (index, (actual, expected)) in
                from_prefetch.iter().zip(from_memory.iter()).enumerate()
            {
                if actual.to_bits() != expected.to_bits() {
                    return Err(format!(
                        "prefetched MoE output {index} not bit-identical"
                    )
                    .into());
                }
            }
            let mut promoted = 0u64;
            unsafe {
                prefetch_stats(std::ptr::null_mut(), &mut promoted,
                               std::ptr::null_mut(), std::ptr::null_mut());
            }
            if promoted < 3 {
                return Err(format!(
                    "annex promoted {promoted}/3 speculated projections on demand"
                )
                .into());
            }
            println!("detached prefetch (annex promote + bit-identity): PASS");
        } else {
            println!("detached prefetch: SKIP (symbol absent)");
        }

        // Multi-row expert path (optional symbol): rows_v2 over m_r activation rows must be
        // bit-identical to m_r single-expert moe calls with the same route weights.
        type RowsFn = unsafe extern "C" fn(
            *const Expert, *const f32, i32, *const f32, *mut f32, i32, *mut i8, usize) -> i32;
        let rows_fn: Option<RowsFn> = unsafe {
            let symbol = libc::dlsym(handle, c"bw24_cpu_expert_rows_v2".as_ptr());
            if symbol.is_null() { None } else { Some(std::mem::transmute(symbol)) }
        };
        if let Some(rows_fn) = rows_fn {
          for rows_type in [GgmlType::Q2_K, GgmlType::IQ3_S, GgmlType::Q4_K] {
            let type_row = fixture(rows_type, 256);
            let mut rows_blobs: Vec<Vec<u8>> = Vec::new();
            for projection_index in 0..3usize {
                let mut row = type_row.clone();
                row[11] ^= 0x2b ^ (projection_index as u8);
                rows_blobs.push(row.repeat(256));
            }
            let rows_scales = [0.5f32, 0.25, 0.75];
            let rows_projection = |weights: *const u8, scale: f32| Projection {
                weights,
                qtype: qtype(rows_type),
                in_features: 256,
                out_features: 256,
                row_bytes: type_row.len(),
                byte_len: type_row.len() * 256,
                file_fd: -1,
                file_offset: 0,
                scale,
            };
            let expert = Expert {
                gate: rows_projection(rows_blobs[0].as_ptr(), rows_scales[0]),
                up: rows_projection(rows_blobs[1].as_ptr(), rows_scales[1]),
                down: rows_projection(rows_blobs[2].as_ptr(), rows_scales[2]),
                route_weight: 0.0, // per-row weights come from the rows argument
            };
            let m_r = 3usize;
            let mut inputs = Vec::with_capacity(m_r * 256);
            for r in 0..m_r {
                for i in 0..256 {
                    inputs.push(0.01 * ((i as f32) * 0.07 + r as f32 * 0.3).sin());
                }
            }
            let weights_r = [0.5f32, -0.25, 0.125];
            let mut rows_out = vec![f32::NAN; m_r * 256];
            let mut error = vec![0i8; 512];
            let status = unsafe {
                rows_fn(&expert, inputs.as_ptr(), m_r as i32, weights_r.as_ptr(),
                        rows_out.as_mut_ptr(), 8, error.as_mut_ptr(), error.len())
            };
            if status != 0 {
                let message = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
                return Err(format!("expert rows call failed: {message}").into());
            }
            for r in 0..m_r {
                let single = Expert { route_weight: weights_r[r], ..expert };
                let reference = run_with_input(
                    std::slice::from_ref(&single),
                    &inputs[r * 256..(r + 1) * 256],
                )?;
                for (index, (actual, expected)) in rows_out[r * 256..(r + 1) * 256]
                    .iter()
                    .zip(reference.iter())
                    .enumerate()
                {
                    if actual.to_bits() != expected.to_bits() {
                        return Err(format!(
                            "expert rows row {r} output {index} not bit-identical: \
                             expected={expected} actual={actual}"
                        )
                        .into());
                    }
                }
            }
            println!(
                "multi-row expert path (decode-amortized, bit-identity, {rows_type:?}): PASS"
            );
          }
        } else {
            println!("multi-row expert path: SKIP (symbol absent)");
        }
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
