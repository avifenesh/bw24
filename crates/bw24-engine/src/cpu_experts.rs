//! Opt-in native CPU backend for Hy3 routed experts.
//!
//! `BW24_CPU_EXPERT_LIB=/path/libbw24-cpu-experts.so` dynamically loads the stable v1 C ABI
//! implemented by `tools/bw24_cpu_experts.cpp`. Keeping the loader dynamic is deliberate: naked
//! bw24 builds and CI retain no llama.cpp header, library, OpenMP, or CPU-ISA dependency. The
//! experimental backend consumes the original host-resident GGUF bytes and returns only one f32
//! hidden-state contribution to CUDA.

use std::ffi::{c_char, c_void, CStr, CString};
use std::os::unix::ffi::OsStrExt as _;
use std::os::unix::io::AsRawFd as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender};
use std::sync::OnceLock;

use crate::hybrid::MoeWeights;
use crate::model::{ExpertKeepalive, ExpertSource};
use crate::{
    QT_BF16, QT_F32, QT_IQ3_S, QT_IQ4_XS, QT_NVFP4, QT_Q2_K, QT_Q3_K, QT_Q4_0, QT_Q4_K, QT_Q5_K,
    QT_Q6_K, QT_Q8_0,
};

#[repr(C)]
#[derive(Clone, Copy)]
struct CpuProjectionV1 {
    weights: *const u8,
    ggml_type: i32,
    in_features: i32,
    out_features: i32,
    row_bytes: usize,
    byte_len: usize,
    file_fd: i32,
    file_offset: u64,
    scale: f32,
}

#[repr(C)]
struct CpuExpertV1 {
    gate: CpuProjectionV1,
    up: CpuProjectionV1,
    down: CpuProjectionV1,
    route_weight: f32,
}

#[derive(Clone)]
struct OwnedProjection {
    weights: usize,
    ggml_type: i32,
    in_features: i32,
    out_features: i32,
    row_bytes: usize,
    byte_len: usize,
    file: Option<std::sync::Arc<std::fs::File>>,
    file_offset: u64,
    scale: f32,
}

#[derive(Clone)]
struct OwnedExpert {
    gate: OwnedProjection,
    up: OwnedProjection,
    down: OwnedProjection,
    route_weight: f32,
}

/// Owned request sent to the persistent CPU worker thread. Weight addresses point into immutable
/// `HostExps` stores. The caller must join the worker before the borrowed `MoeWeights` can be
/// dropped. `CpuExpertTicket::drop` enforces that lifetime even when the GPU path returns early.
pub(crate) struct CpuExpertJob {
    experts: Vec<OwnedExpert>,
    input: Vec<f32>,
    output_features: usize,
    threads: i32,
}

type AbiVersionFn = unsafe extern "C" fn() -> u32;
type MoeTokenFn = unsafe extern "C" fn(
    *const CpuExpertV1,
    i32,
    *const f32,
    *mut f32,
    i32,
    *mut c_char,
    usize,
) -> i32;
type CacheStatsFn = unsafe extern "C" fn(*mut u64, *mut u64, *mut u64, *mut u64);
type ProfileStatsFn = unsafe extern "C" fn(*mut u64, *mut u64, *mut u64, *mut u64);

struct CpuBackend {
    // Kept open for process lifetime so the function pointer and llama.cpp traits remain valid.
    _handle: usize,
    moe_token: MoeTokenFn,
    cache_stats: CacheStatsFn,
    profile_stats: ProfileStatsFn,
}

// The dlopen handle names process-global immutable code after initialization.
unsafe impl Send for CpuBackend {}
unsafe impl Sync for CpuBackend {}

static BACKEND: OnceLock<Result<CpuBackend, String>> = OnceLock::new();
static CALLS: AtomicU64 = AtomicU64::new(0);
static EXPERTS: AtomicU64 = AtomicU64::new(0);
static WALL_NS: AtomicU64 = AtomicU64::new(0);
static GPU_RESIDENT_0: AtomicU64 = AtomicU64::new(0);
static GPU_RESIDENT_1: AtomicU64 = AtomicU64::new(0);
static GPU_RESIDENT_2: AtomicU64 = AtomicU64::new(0);
struct CpuRequest {
    job: CpuExpertJob,
    reply: SyncSender<Result<Vec<f32>, String>>,
}

struct CpuExecutor {
    sender: SyncSender<CpuRequest>,
}

pub(crate) struct CpuExpertTicket {
    receiver: Option<Receiver<Result<Vec<f32>, String>>>,
}

static EXECUTOR: OnceLock<Result<CpuExecutor, String>> = OnceLock::new();

pub(crate) fn configured() -> bool {
    std::env::var_os("BW24_CPU_EXPERT_LIB").is_some()
}

fn threads_from_env() -> Result<i32, String> {
    let value = match std::env::var("BW24_CPU_EXPERT_THREADS") {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => return Err(format!("cannot read BW24_CPU_EXPERT_THREADS: {error}")),
    };
    parse_thread_count(value.as_deref())
}

fn parse_thread_count(value: Option<&str>) -> Result<i32, String> {
    let raw = value.unwrap_or("8");
    let threads = raw
        .parse::<i32>()
        .map_err(|_| format!("BW24_CPU_EXPERT_THREADS={raw:?} is not an integer"))?;
    if !(1..=256).contains(&threads) {
        return Err(format!(
            "BW24_CPU_EXPERT_THREADS={threads} is outside 1..=256"
        ));
    }
    Ok(threads)
}

fn dl_error(context: &str) -> String {
    // SAFETY: dlerror returns either null or a process-owned NUL-terminated diagnostic.
    let detail = unsafe {
        let pointer = libc::dlerror();
        if pointer.is_null() {
            "unknown dynamic-loader error".to_string()
        } else {
            CStr::from_ptr(pointer).to_string_lossy().into_owned()
        }
    };
    format!("{context}: {detail}")
}

fn load_symbol(handle: *mut c_void, name: &'static [u8]) -> Result<*mut c_void, String> {
    debug_assert_eq!(name.last(), Some(&0));
    // SAFETY: the name is statically NUL-terminated and the handle stays open for process life.
    unsafe {
        libc::dlerror();
    }
    let symbol = unsafe { libc::dlsym(handle, name.as_ptr().cast()) };
    if symbol.is_null() {
        Err(dl_error(&format!(
            "missing symbol {}",
            String::from_utf8_lossy(&name[..name.len() - 1])
        )))
    } else {
        Ok(symbol)
    }
}

fn load_backend_from_path(path: &std::ffi::OsStr) -> Result<CpuBackend, String> {
    let c_path = CString::new(path.as_bytes())
        .map_err(|_| "BW24_CPU_EXPERT_LIB contains a NUL byte".to_string())?;
    // SAFETY: c_path is NUL-terminated. The successful handle is intentionally never closed.
    let handle = unsafe { libc::dlopen(c_path.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
    if handle.is_null() {
        return Err(dl_error(&format!("cannot load {}", path.to_string_lossy())));
    }
    let result = (|| {
        let version_symbol = load_symbol(handle, b"bw24_cpu_experts_abi_version\0")?;
        let token_symbol = load_symbol(handle, b"bw24_cpu_moe_token_v1\0")?;
        let stats_symbol = load_symbol(handle, b"bw24_cpu_expert_cache_stats_v1\0")?;
        let profile_symbol = load_symbol(handle, b"bw24_cpu_expert_profile_stats_v1\0")?;
        // SAFETY: the companion library exports these exact v1 C signatures.
        let version: AbiVersionFn = unsafe { std::mem::transmute(version_symbol) };
        let abi = unsafe { version() };
        require_abi_v1(abi)?;
        let moe_token: MoeTokenFn = unsafe { std::mem::transmute(token_symbol) };
        let cache_stats: CacheStatsFn = unsafe { std::mem::transmute(stats_symbol) };
        let profile_stats: ProfileStatsFn = unsafe { std::mem::transmute(profile_symbol) };
        eprintln!(
            "[bw24] experimental CPU expert backend: {} (threads={})",
            path.to_string_lossy(),
            threads_from_env()?,
        );
        Ok(CpuBackend {
            _handle: handle as usize,
            moe_token,
            cache_stats,
            profile_stats,
        })
    })();
    if result.is_err() {
        // SAFETY: no function pointer escapes the failed initialization path.
        unsafe {
            libc::dlclose(handle);
        }
    }
    result
}

fn require_abi_v1(abi: u32) -> Result<(), String> {
    if abi == 1 {
        Ok(())
    } else {
        Err(format!(
            "CPU expert ABI {abi} is incompatible; bw24 requires v1"
        ))
    }
}

fn load_backend() -> Result<CpuBackend, String> {
    let path = std::env::var_os("BW24_CPU_EXPERT_LIB")
        .ok_or_else(|| "BW24_CPU_EXPERT_LIB is not set".to_string())?;
    load_backend_from_path(&path)
}

fn backend() -> Result<&'static CpuBackend, String> {
    match BACKEND.get_or_init(load_backend) {
        Ok(backend) => Ok(backend),
        Err(error) => Err(error.clone()),
    }
}

fn ggml_type(qtype: i32) -> Result<i32, String> {
    match qtype {
        QT_F32 => Ok(0),
        QT_Q4_0 => Ok(2),
        QT_Q8_0 => Ok(8),
        QT_Q2_K => Ok(10),
        QT_Q3_K => Ok(11),
        QT_Q4_K => Ok(12),
        QT_Q5_K => Ok(13),
        QT_Q6_K => Ok(14),
        QT_IQ3_S => Ok(21),
        QT_IQ4_XS => Ok(23),
        QT_BF16 => Ok(30),
        QT_NVFP4 => Ok(40),
        other => Err(format!(
            "CPU expert backend does not map bw24 qtype {other}"
        )),
    }
}

fn projection(exps: &crate::model::HostExps, expert: usize) -> Result<OwnedProjection, String> {
    let layout = exps.expert_layout(expert);
    let bytes = exps.expert_bytes(expert);
    if bytes.len() != layout.len || layout.len != layout.row_bytes * exps.out_f {
        return Err(format!(
            "expert {expert} extent mismatch: bytes={} layout={} rows={}x{}",
            bytes.len(),
            layout.len,
            exps.out_f,
            layout.row_bytes
        ));
    }
    let (file, file_offset) = match exps.expert_source(expert) {
        ExpertSource::Disk {
            file, offset, len, ..
        } => {
            if len != layout.len {
                return Err(format!(
                    "expert {expert} disk extent {len} differs from layout {}",
                    layout.len
                ));
            }
            (Some(file.clone()), offset)
        }
        ExpertSource::Memory {
            keepalive: Some(ExpertKeepalive::Pinned(_)),
            ..
        }
        | ExpertSource::Memory {
            keepalive: Some(ExpertKeepalive::Buffer(_)),
            ..
        } => {
            return Err(format!(
                "expert {expert} is in CUDA write-combined host memory; CPU reads are disabled"
            ));
        }
        ExpertSource::Memory { .. } => (None, 0),
    };
    Ok(OwnedProjection {
        weights: bytes.as_ptr() as usize,
        ggml_type: ggml_type(layout.qtype)?,
        in_features: i32::try_from(exps.in_f)
            .map_err(|_| format!("expert input width {} exceeds i32", exps.in_f))?,
        out_features: i32::try_from(exps.out_f)
            .map_err(|_| format!("expert output width {} exceeds i32", exps.out_f))?,
        row_bytes: layout.row_bytes,
        byte_len: layout.len,
        file,
        file_offset,
        scale: exps.macro_scale(expert),
    })
}

pub(crate) fn prepare_job(
    weights: &MoeWeights,
    _layer: u16,
    selected: &[(usize, f32)],
    input: &[f32],
) -> Result<CpuExpertJob, String> {
    if selected.is_empty() {
        return Err("cannot prepare an empty CPU expert job".to_string());
    }
    if input.len() != weights.gate_exps.in_f {
        return Err(format!(
            "CPU expert input has {} values, expected {}",
            input.len(),
            weights.gate_exps.in_f
        ));
    }
    let mut experts = Vec::with_capacity(selected.len());
    for &(expert, route_weight) in selected {
        if expert >= weights.gate_exps.n_expert {
            return Err(format!("CPU expert id {expert} is out of range"));
        }
        if weights
            .active_experts
            .as_ref()
            .is_some_and(|active| !active[expert])
        {
            return Err(format!("router selected pruned CPU expert id {expert}"));
        }
        experts.push(OwnedExpert {
            gate: projection(&weights.gate_exps, expert)?,
            up: projection(&weights.up_exps, expert)?,
            down: projection(&weights.down_exps, expert)?,
            route_weight,
        });
    }
    Ok(CpuExpertJob {
        experts,
        input: input.to_vec(),
        output_features: weights.down_exps.out_f,
        threads: threads_from_env()?,
    })
}

fn ffi_projection(value: &OwnedProjection) -> CpuProjectionV1 {
    CpuProjectionV1 {
        weights: value.weights as *const u8,
        ggml_type: value.ggml_type,
        in_features: value.in_features,
        out_features: value.out_features,
        row_bytes: value.row_bytes,
        byte_len: value.byte_len,
        file_fd: value.file.as_ref().map_or(-1, |file| file.as_raw_fd()),
        file_offset: value.file_offset,
        scale: value.scale,
    }
}

fn ffi_expert(expert: &OwnedExpert) -> CpuExpertV1 {
    CpuExpertV1 {
        gate: ffi_projection(&expert.gate),
        up: ffi_projection(&expert.up),
        down: ffi_projection(&expert.down),
        route_weight: expert.route_weight,
    }
}

fn execute(job: CpuExpertJob) -> Result<Vec<f32>, String> {
    let backend = backend()?;
    let experts: Vec<_> = job.experts.iter().map(ffi_expert).collect();
    let count =
        i32::try_from(experts.len()).map_err(|_| "CPU expert count exceeds i32".to_string())?;
    let mut output = vec![0.0f32; job.output_features];
    let mut error = vec![0i8; 1024];
    let start = std::time::Instant::now();
    // SAFETY: every descriptor points into immutable model-owned bytes retained until the caller
    // joins this job; all input/output/error spans have the dimensions encoded in the descriptors.
    // SAFETY: every descriptor and span satisfies the stable v1 ABI.
    let status = unsafe {
        (backend.moe_token)(
            experts.as_ptr(),
            count,
            job.input.as_ptr(),
            output.as_mut_ptr(),
            job.threads,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    WALL_NS.fetch_add(
        start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
        Ordering::Relaxed,
    );
    CALLS.fetch_add(1, Ordering::Relaxed);
    EXPERTS.fetch_add(experts.len() as u64, Ordering::Relaxed);
    if status != 0 {
        // SAFETY: the companion ABI always NUL-terminates this fixed-size error buffer.
        let message = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
        return Err(format!("CPU expert backend failed: {message}"));
    }
    Ok(output)
}

fn start_executor() -> Result<CpuExecutor, String> {
    let (sender, receiver) = std::sync::mpsc::sync_channel::<CpuRequest>(1);
    std::thread::Builder::new()
        .name("bw24-cpu-executor".to_string())
        .spawn(move || {
            while let Ok(request) = receiver.recv() {
                let result = execute(request.job);
                let _ = request.reply.send(result);
            }
        })
        .map_err(|error| format!("cannot start persistent CPU expert executor: {error}"))?;
    Ok(CpuExecutor { sender })
}

fn executor() -> Result<&'static CpuExecutor, String> {
    match EXECUTOR.get_or_init(start_executor) {
        Ok(executor) => Ok(executor),
        Err(error) => Err(error.clone()),
    }
}

pub(crate) fn submit(job: CpuExpertJob) -> Result<CpuExpertTicket, String> {
    let (reply, receiver) = std::sync::mpsc::sync_channel(1);
    executor()?
        .sender
        .send(CpuRequest { job, reply })
        .map_err(|_| "persistent CPU expert executor stopped".to_string())?;
    Ok(CpuExpertTicket {
        receiver: Some(receiver),
    })
}

pub(crate) fn record_incomplete_gpu_residency(resident_projections: usize) {
    match resident_projections {
        0 => GPU_RESIDENT_0.fetch_add(1, Ordering::Relaxed),
        1 => GPU_RESIDENT_1.fetch_add(1, Ordering::Relaxed),
        2 => GPU_RESIDENT_2.fetch_add(1, Ordering::Relaxed),
        _ => return,
    };
}

pub(crate) fn incomplete_gpu_residency_stats() -> (u64, u64, u64) {
    (
        GPU_RESIDENT_0.load(Ordering::Relaxed),
        GPU_RESIDENT_1.load(Ordering::Relaxed),
        GPU_RESIDENT_2.load(Ordering::Relaxed),
    )
}

impl CpuExpertTicket {
    pub(crate) fn wait(mut self) -> Result<Vec<f32>, String> {
        self.receiver
            .take()
            .expect("CPU expert ticket receiver is present until wait")
            .recv()
            .map_err(|_| "persistent CPU expert executor dropped a result".to_string())?
    }
}

impl Drop for CpuExpertTicket {
    fn drop(&mut self) {
        // Jobs borrow immutable model bytes through raw pointers. Every early-return path must
        // therefore wait until the persistent worker has stopped reading those bytes.
        if let Some(receiver) = self.receiver.take() {
            let _ = receiver.recv();
        }
    }
}

pub(crate) fn stats() -> (u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64) {
    let mut cache_hits = 0;
    let mut cache_misses = 0;
    let mut read_bytes = 0;
    let mut resident_bytes = 0;
    let mut prepare_ns = 0;
    let mut io_ns = 0;
    let mut insert_ns = 0;
    let mut compute_ns = 0;
    if let Ok(backend) = backend() {
        // SAFETY: all four outputs point to initialized u64 storage owned by this call.
        unsafe {
            (backend.cache_stats)(
                &mut cache_hits,
                &mut cache_misses,
                &mut read_bytes,
                &mut resident_bytes,
            );
            (backend.profile_stats)(&mut prepare_ns, &mut io_ns, &mut insert_ns, &mut compute_ns);
        }
    }
    (
        CALLS.load(Ordering::Relaxed),
        EXPERTS.load(Ordering::Relaxed),
        WALL_NS.load(Ordering::Relaxed),
        cache_hits,
        cache_misses,
        read_bytes,
        resident_bytes,
        prepare_ns,
        io_ns,
        insert_ns,
        compute_ns,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_supported_ggml_types() {
        assert_eq!(ggml_type(QT_Q2_K).unwrap(), 10);
        assert_eq!(ggml_type(QT_Q3_K).unwrap(), 11);
        assert_eq!(ggml_type(QT_Q4_K).unwrap(), 12);
        assert_eq!(ggml_type(QT_IQ3_S).unwrap(), 21);
        assert_eq!(ggml_type(QT_IQ4_XS).unwrap(), 23);
        assert_eq!(ggml_type(QT_Q8_0).unwrap(), 8);
    }

    #[test]
    fn rejects_missing_symbols_and_wrong_abi() {
        let error = match load_backend_from_path(std::ffi::OsStr::new("libc.so.6")) {
            Ok(_) => panic!("libc unexpectedly provided the CPU expert ABI"),
            Err(error) => error,
        };
        assert!(error.contains("missing symbol bw24_cpu_experts_abi_version"));
        assert!(require_abi_v1(0).is_err());
        assert!(require_abi_v1(2).is_err());
        assert!(require_abi_v1(1).is_ok());
    }

    #[test]
    fn validates_thread_count() {
        assert_eq!(parse_thread_count(None).unwrap(), 8);
        assert_eq!(parse_thread_count(Some("1")).unwrap(), 1);
        assert_eq!(parse_thread_count(Some("256")).unwrap(), 256);
        for invalid in ["", "0", "257", "eight"] {
            assert!(parse_thread_count(Some(invalid)).is_err());
        }
    }

    #[test]
    fn dropped_ticket_joins_outstanding_worker() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        let (reply, receiver) = std::sync::mpsc::sync_channel(1);
        let (release, start) = std::sync::mpsc::sync_channel(0);
        let completed = Arc::new(AtomicBool::new(false));
        let worker_completed = Arc::clone(&completed);
        let worker = std::thread::spawn(move || {
            start.recv().unwrap();
            worker_completed.store(true, Ordering::Release);
            reply.send(Ok(Vec::new())).unwrap();
        });
        let ticket = CpuExpertTicket {
            receiver: Some(receiver),
        };
        release.send(()).unwrap();
        drop(ticket);
        assert!(completed.load(Ordering::Acquire));
        worker.join().unwrap();
    }
}
