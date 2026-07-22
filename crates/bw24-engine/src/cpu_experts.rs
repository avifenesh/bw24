//! Opt-in native CPU backend for Hy3 routed experts.
//!
//! `BW24_CPU_EXPERT_LIB=/path/libbw24-cpu-experts.so` dynamically loads bw24's stable v2 C ABI,
//! implemented by `tools/bw24_cpu_experts.cpp`. The companion owns its packed-format decoders,
//! activation quantizer, AVX2/AVX-VNNI dots, cache, and positioned-I/O path; no external inference
//! runtime is linked or loaded. It consumes original host-resident GGUF bytes and returns one f32
//! hidden state contribution to CUDA.

use std::ffi::{CStr, CString, c_char, c_void};
use std::os::unix::ffi::OsStrExt as _;
use std::os::unix::io::AsRawFd as _;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender};

use crate::hybrid::MoeWeights;
use crate::model::{ExpertKeepalive, ExpertSource};
use crate::{
    QT_BF16, QT_F32, QT_IQ3_S, QT_IQ4_XS, QT_NVFP4, QT_Q2_K, QT_Q3_K, QT_Q4_0, QT_Q4_K, QT_Q5_K,
    QT_Q6_K, QT_Q8_0,
};

#[repr(C)]
#[derive(Clone, Copy)]
struct CpuProjectionV2 {
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
struct CpuExpertV2 {
    gate: CpuProjectionV2,
    up: CpuProjectionV2,
    down: CpuProjectionV2,
    route_weight: f32,
}

#[derive(Clone)]
pub(crate) struct OwnedProjection {
    weights: usize,
    qtype: i32,
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
    *const CpuExpertV2,
    i32,
    *const f32,
    *mut f32,
    i32,
    *mut c_char,
    usize,
) -> i32;
type CacheStatsFn = unsafe extern "C" fn(*mut u64, *mut u64, *mut u64, *mut u64);
type ProfileStatsFn = unsafe extern "C" fn(*mut u64, *mut u64, *mut u64, *mut u64);
type PrefetchFn = unsafe extern "C" fn(*const CpuProjectionV2, i32, *mut i8, usize) -> i32;

struct CpuBackend {
    // Kept open for process lifetime so the native bw24 function pointers remain valid.
    _handle: usize,
    moe_token: MoeTokenFn,
    cache_stats: CacheStatsFn,
    profile_stats: ProfileStatsFn,
    // Optional (added after ABI v2 shipped): absent in older companions, prefetch disabled.
    prefetch: Option<PrefetchFn>,
}

// The dlopen handle names process-global immutable code after initialization.
unsafe impl Send for CpuBackend {}
unsafe impl Sync for CpuBackend {}

static BACKEND: OnceLock<Result<CpuBackend, String>> = OnceLock::new();
static CALLS: AtomicU64 = AtomicU64::new(0);
static EXPERTS: AtomicU64 = AtomicU64::new(0);
static WALL_NS: AtomicU64 = AtomicU64::new(0);
static EXPOSED_WAIT_NS: AtomicU64 = AtomicU64::new(0);
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
        let token_symbol = load_symbol(handle, b"bw24_cpu_moe_token_v2\0")?;
        let stats_symbol = load_symbol(handle, b"bw24_cpu_expert_cache_stats_v2\0")?;
        let profile_symbol = load_symbol(handle, b"bw24_cpu_expert_profile_stats_v2\0")?;
        // SAFETY: the companion library exports these exact v2 C signatures.
        let version: AbiVersionFn = unsafe { std::mem::transmute(version_symbol) };
        let abi = unsafe { version() };
        require_abi_v2(abi)?;
        let moe_token: MoeTokenFn = unsafe { std::mem::transmute(token_symbol) };
        let cache_stats: CacheStatsFn = unsafe { std::mem::transmute(stats_symbol) };
        let profile_stats: ProfileStatsFn = unsafe { std::mem::transmute(profile_symbol) };
        // Optional symbol: older companions predate speculative prefetch.
        let prefetch: Option<PrefetchFn> =
            load_symbol(handle, b"bw24_cpu_expert_prefetch_v2\0")
                .ok()
                // SAFETY: when present the companion exports this exact v2 C signature.
                .map(|symbol| unsafe { std::mem::transmute::<*mut c_void, PrefetchFn>(symbol) });
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
            prefetch,
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

fn require_abi_v2(abi: u32) -> Result<(), String> {
    if abi == 2 {
        Ok(())
    } else {
        Err(format!(
            "CPU expert ABI {abi} is incompatible; bw24 requires native v2"
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

fn native_qtype(qtype: i32) -> Result<i32, String> {
    match qtype {
        QT_F32 | QT_Q4_0 | QT_Q8_0 | QT_Q2_K | QT_Q3_K | QT_Q4_K | QT_Q5_K | QT_Q6_K | QT_IQ3_S
        | QT_IQ4_XS | QT_BF16 | QT_NVFP4 => Ok(qtype),
        other => Err(format!(
            "native CPU expert backend does not support bw24 qtype {other}"
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
        qtype: native_qtype(layout.qtype)?,
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

fn ffi_projection(value: &OwnedProjection) -> CpuProjectionV2 {
    CpuProjectionV2 {
        weights: value.weights as *const u8,
        qtype: value.qtype,
        in_features: value.in_features,
        out_features: value.out_features,
        row_bytes: value.row_bytes,
        byte_len: value.byte_len,
        file_fd: value.file.as_ref().map_or(-1, |file| file.as_raw_fd()),
        file_offset: value.file_offset,
        scale: value.scale,
    }
}

fn ffi_expert(expert: &OwnedExpert) -> CpuExpertV2 {
    CpuExpertV2 {
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
    // SAFETY: every descriptor and span satisfies the stable native v2 ABI.
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
        // Defend the Rust side even if a broken companion violates the ABI contract.
        if let Some(last) = error.last_mut() {
            *last = 0;
        }
        // SAFETY: the fixed-size buffer contains at least the NUL written above.
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
        let start = std::time::Instant::now();
        let result = self
            .receiver
            .take()
            .expect("CPU expert ticket receiver is present until wait")
            .recv()
            .map_err(|_| "persistent CPU expert executor dropped a result".to_string())?;
        EXPOSED_WAIT_NS.fetch_add(
            start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
        result
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

// ---- prediction-guided speculative prefetch (increment 2) --------------------------------
// Grounded in research/moe/expert-prefetch-prediction-pilot.md: applying layer j's router to
// layer k's MoE input predicts the deep half's routed experts at 84-100% argmax precision for
// j-k <= 4. A dedicated worker thread (never the decode thread) scores lookahead layers with
// host copies of the router weights, filters HBM-resident and pruned experts, and hands the
// predicted-and-missing projections to the companion's detached cold-insert prefetch.

struct PredictLayer {
    /// Router weights transposed to [n_expert][n_embd] for sequential dot products.
    router_t: Vec<f32>,
    bias: Option<Vec<f32>>,
    active: Option<Vec<bool>>,
    n_embd: usize,
    n_expert: usize,
    n_used: usize,
    sig: (f32, bool),
    /// Prebuilt per-expert projection descriptors (gate/up/down); None for pruned ids. The
    /// OwnedProjection keeps the backing file handles alive for the process lifetime.
    experts: Vec<Option<[OwnedProjection; 3]>>,
}

struct Predictor {
    sender: SyncSender<(u16, Vec<f32>)>,
    submitted: AtomicU64,
    dropped: AtomicU64,
}

static PREDICTOR: OnceLock<Option<Predictor>> = OnceLock::new();

fn prefetch_depth_from_env() -> usize {
    std::env::var("BW24_MOE_PREFETCH")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&depth| (1..=8).contains(&depth))
        .unwrap_or(0)
}

/// Build the predictor from per-layer inputs and start its worker. Called once from the model
/// after residency freeze (the resident snapshot must be final). `layers` carries, per MoE
/// layer index, everything the worker needs — fully owned, no model references escape.
pub(crate) fn start_prefetch_predictor(
    layers: Vec<(u16, PredictLayerInit)>,
    resident: std::collections::HashSet<(u16, u8, u16)>,
) -> Result<(), String> {
    let depth = prefetch_depth_from_env();
    if depth == 0 {
        return Err("BW24_MOE_PREFETCH is not enabled".to_string());
    }
    let backend = backend()?;
    if backend.prefetch.is_none() {
        return Err("companion library lacks bw24_cpu_expert_prefetch_v2".to_string());
    }
    let top = std::env::var("BW24_MOE_PREFETCH_TOP")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&t| (1..=8).contains(&t))
        .unwrap_or(1);
    let min_layer = std::env::var("BW24_MOE_PREFETCH_MIN_LAYER")
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(40);
    let mut table: std::collections::HashMap<u16, PredictLayer> = Default::default();
    for (layer_index, init) in layers {
        let mut experts = Vec::with_capacity(init.weights_n_expert);
        for expert in 0..init.weights_n_expert {
            experts.push(init.build_expert(expert));
        }
        let n_embd = init.n_embd;
        let n_expert = init.weights_n_expert;
        // Transpose [n_embd, n_expert] -> [n_expert][n_embd].
        let mut router_t = vec![0.0f32; n_embd * n_expert];
        for row in 0..n_embd {
            for expert in 0..n_expert {
                router_t[expert * n_embd + row] = init.router[row * n_expert + expert];
            }
        }
        table.insert(layer_index, PredictLayer {
            router_t,
            bias: init.bias,
            active: init.active,
            n_embd,
            n_expert,
            n_used: init.n_used,
            sig: init.sig,
            experts,
        });
    }
    let (sender, receiver) = std::sync::mpsc::sync_channel::<(u16, Vec<f32>)>(8);
    std::thread::Builder::new()
        .name("bw24-moe-prefetch".to_string())
        .spawn(move || prefetch_worker(receiver, table, resident, depth, top, min_layer))
        .map_err(|error| format!("cannot spawn prefetch worker: {error}"))?;
    let created = PREDICTOR
        .set(Some(Predictor {
            sender,
            submitted: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
        }))
        .is_ok();
    if !created {
        return Err("prefetch predictor already started".to_string());
    }
    eprintln!("[bw24] moe prefetch predictor: depth={depth} top={top}");
    Ok(())
}

/// Per-layer construction inputs, fully owned.
pub(crate) struct PredictLayerInit {
    pub router: Vec<f32>,
    pub bias: Option<Vec<f32>>,
    pub active: Option<Vec<bool>>,
    pub n_embd: usize,
    pub n_used: usize,
    pub sig: (f32, bool),
    pub weights_n_expert: usize,
    pub gate: Vec<Option<crate::cpu_experts::OwnedProjection>>,
    pub up: Vec<Option<crate::cpu_experts::OwnedProjection>>,
    pub down: Vec<Option<crate::cpu_experts::OwnedProjection>>,
}

impl PredictLayerInit {
    fn build_expert(&self, expert: usize) -> Option<[OwnedProjection; 3]> {
        Some([
            self.gate[expert].clone()?,
            self.up[expert].clone()?,
            self.down[expert].clone()?,
        ])
    }
}

/// Build one OwnedProjection for the predictor table (pruned/CUDA-pinned experts yield None).
pub(crate) fn predictor_projection(
    exps: &crate::model::HostExps,
    expert: usize,
) -> Option<OwnedProjection> {
    projection(exps, expert).ok()
}

/// Fire-and-forget: hand layer il's MoE input to the predictor. Never blocks the decode
/// thread — a full channel drops the sample (speculation must not backpressure decode).
pub(crate) fn predictor_submit(layer: u16, input: &[f32]) {
    let Some(Some(predictor)) = PREDICTOR.get().map(Option::as_ref) else { return };
    match predictor.sender.try_send((layer, input.to_vec())) {
        Ok(()) => {
            predictor.submitted.fetch_add(1, Ordering::Relaxed);
        }
        Err(_) => {
            predictor.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }
}

pub(crate) fn predictor_enabled() -> bool {
    matches!(PREDICTOR.get(), Some(Some(_)))
}

pub(crate) fn predictor_stats() -> (u64, u64) {
    match PREDICTOR.get().map(Option::as_ref) {
        Some(Some(p)) => (
            p.submitted.load(Ordering::Relaxed),
            p.dropped.load(Ordering::Relaxed),
        ),
        _ => (0, 0),
    }
}

fn prefetch_worker(
    receiver: Receiver<(u16, Vec<f32>)>,
    table: std::collections::HashMap<u16, PredictLayer>,
    resident: std::collections::HashSet<(u16, u8, u16)>,
    depth: usize,
    top: usize,
    min_layer: u16,
) {
    let Ok(backend) = backend() else { return };
    let Some(prefetch) = backend.prefetch else { return };
    let mut error = vec![0i8; 512];
    while let Ok((layer, input)) = receiver.recv() {
        for d in 1..=depth {
            let target = layer + d as u16;
            if target < min_layer {
                continue;  // pilot precision only justifies the deep half (k >= ~40)
            }
            let Some(predict) = table.get(&target) else { continue };
            if input.len() != predict.n_embd {
                continue;
            }
            // logits = router @ x with sigmoid selection parity to the runtime oracle.
            let mut logits = vec![0.0f32; predict.n_expert];
            for (expert, logit) in logits.iter_mut().enumerate() {
                let row = &predict.router_t[expert * predict.n_embd
                    ..(expert + 1) * predict.n_embd];
                *logit = row
                    .iter()
                    .zip(&input)
                    .map(|(weight, value)| weight * value)
                    .sum();
            }
            let Ok((sel, _weights)) = crate::hybrid::HybridModel::moe_route_sigmoid_host_public(
                &logits,
                1,
                predict.n_expert,
                predict.n_used,
                predict.bias.as_deref(),
                predict.sig.0,
                predict.sig.1,
                predict.active.as_deref(),
            ) else {
                continue;
            };
            let mut descs: Vec<CpuProjectionV2> = Vec::new();
            for &expert in sel.iter().take(top) {
                let expert_index = expert as usize;
                let Some(Some(projections)) = predict.experts.get(expert_index) else {
                    continue;
                };
                for (proj_index, owned) in projections.iter().enumerate() {
                    if owned.file.is_none() {
                        continue;  // memory-backed, nothing to read
                    }
                    if resident.contains(&(target, proj_index as u8, expert as u16)) {
                        continue;  // HBM-resident, never CPU-routed
                    }
                    descs.push(ffi_projection(owned));
                }
            }
            if descs.is_empty() {
                continue;
            }
            let count = descs.len() as i32;
            // SAFETY: descs outlive the call; the companion copies what it keeps.
            unsafe {
                prefetch(descs.as_ptr(), count, error.as_mut_ptr(), error.len());
            }
        }
    }
}

pub(crate) fn exposed_wait_ns() -> u64 {
    EXPOSED_WAIT_NS.load(Ordering::Relaxed)
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
    fn accepts_supported_native_qtypes_without_translation() {
        for qtype in [QT_Q2_K, QT_Q3_K, QT_Q4_K, QT_IQ3_S, QT_IQ4_XS, QT_Q8_0] {
            assert_eq!(native_qtype(qtype).unwrap(), qtype);
        }
    }

    #[test]
    fn rejects_missing_symbols_and_wrong_abi() {
        let error = match load_backend_from_path(std::ffi::OsStr::new("libc.so.6")) {
            Ok(_) => panic!("libc unexpectedly provided the CPU expert ABI"),
            Err(error) => error,
        };
        assert!(error.contains("missing symbol bw24_cpu_experts_abi_version"));
        assert!(require_abi_v2(0).is_err());
        assert!(require_abi_v2(1).is_err());
        assert!(require_abi_v2(2).is_ok());
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
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

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
