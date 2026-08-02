//! M1 pipeline-parallel 2-stage seam.
//!
//! Door: `MEMRA_PP_STAGES=2` (default OFF — unset/0/1 = no behavior change anywhere).
//! Split: `MEMRA_PP_SPLIT=<i>` puts layers [0, i) in stage 0 and [i, n) in stage 1;
//! default i = n_layers/2.
//!
//! Increment 1 (merged): seam + gate, single device — the eager decode walk runs as two
//! stage subgraphs with an explicit activation handoff (TX copy into a dedicated boundary
//! buffer, RX copy out — never an alias). Cross-layer launch fusions (the generic loop's
//! pending (x1, ffn_out) add-carry; gemma4's pre-quantized next-norm carry) are LOCAL to
//! a stage: the boundary materializes the residual with the same kernels the last layer
//! of an unsplit walk uses (the kernel-check-pinned `add_rms_norm_q8_1 == add then
//! rms_norm_q8_1` / `add_scale_rms_norm_q8_1` identities), so split logits are
//! BIT-IDENTICAL to unsplit — `pp2-gate` proves it per step.
//!
//! Increment 2 (this file): REAL TRANSPORT.
//!   - Per-stage CUDA streams: stage 0 and stage 1 each launch on their own created
//!     stream (`Pp2Rt::enter` pushes the thread-ambient stream override — see
//!     memra_runtime::Gpu::stream). The boundary TX records a CudaEvent on stage-0's
//!     stream; stage-1's RX waits on it before copying out. Single-device this is a
//!     correctness no-op with real synchronization semantics — pp2-gate stays
//!     BIT-IDENTICAL with streams on.
//!   - Device placement: `MEMRA_PP_DEVICES=<d0>,<d1>` maps stage s to device ds (default:
//!     both on the primary engine's device = increment-1 behavior). A remote stage gets
//!     its OWN Engine (modules/functions are per-context) in that device's primary
//!     context; model weights stay on the primary device and are read through enabled
//!     peer access (bring-up placement: correctness first — weight sharding is a later
//!     increment). Stage-owned KV allocation: `new_cache` allocates each layer range's
//!     cache on the owning stage's device (`Cache::new_pp2`).
//!   - Transport-selected boundary copy: same-device -> dtod on stage-0's stream
//!     (today's bytes, unchanged); cross-device -> cudaMemcpyPeerAsync issued on
//!     stage-0's (owning/publishing) stream — the M0 choreography
//!     (research/m0-nccl-20260801: peer copy beats NCCL 2.8x at PP activation sizes).
//!     Guarded by cuDeviceCanAccessPeer BOTH directions: peer-inaccessible pairs FAIL
//!     LOUDLY (stage kernels also dereference cross-device weight/pos pointers, which
//!     requires peer access — a silently host-staged copy would fake a gate PASS).
//!   - Overlap scaffolding (M2 seed, `MEMRA_PP_OVERLAP=1`, default OFF): the boundary
//!     becomes DOUBLE-BUFFERED (two slots, alternating per step) and TX's
//!     write-after-read guard waits only on the PREVIOUS use of the same slot — so a
//!     future pipelined loop can issue token t+1's stage 0 while token t's stage 1 runs.
//!     Each token's math is fully ordered by the slot events either way; the eager
//!     decode_step API still syncs on its own logits D2H, so on one device overlap=1
//!     changes scheduling structure, not math — pp2-gate must stay bit-identical.
//!   - Rollback seam: `MEMRA_PP_STREAMS=0` = the increment-1 same-stream two-dtod seam.
//!
//! Ownership across the seam (unchanged from increment 1):
//!   - hidden state [n_embd] f32 is the ONLY tensor that crosses the boundary;
//!   - KV/linear-attn cache entries are per-layer: stage s exclusively owns cache state
//!     for its layer range (and, under MEMRA_PP_DEVICES, allocates it on its device);
//!   - position/rope state is the scalar `cache.pos` snapshot taken once per step
//!     (each stage reads the same value; a per-stage counter replicates it later);
//!   - the embed table lives with stage 0, output_norm + lm head with the last stage.
//!
//! THE MULTI-STREAM LAW (why this is safe with cudarc event tracking disabled): all
//! cross-stage bytes flow through the persistent boundary slots, ordered by ev_tx/ev_rx;
//! per-stage scratch is allocated AND freed on that stage's stream (stream-ordered); the
//! async mem pool runs with opportunistic reuse OFF + internal dependencies ON
//! (memra-runtime), so a block freed on stream A and reused on stream B carries a
//! driver-inserted dependency. pos_d is uploaded on stage-0's stream BEFORE the TX event
//! that stage 1 waits on, weights are load-time state no stage stream can precede, and
//! the step's terminal logits D2H drains stage 1 (whose TX-wait transitively drains
//! stage 0) before any step-scoped buffer drops.
//!
//! Increment-2 scope: plain eager decode only (generic + gemma4 arms). NOT wired:
//! batch/dc/graph/spec loops and the gemma4-E4B eager arm (`warn_unwired_once` fires).

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{CudaContext, CudaEvent, CudaSlice, CudaStream};

use crate::Engine;

/// Returns `Some(split)` iff the pp2 door is open: `MEMRA_PP_STAGES=2` with a valid
/// split in [1, n_layers-1]. Reads the environment on every call (gates toggle the
/// door in-process); the cost is two getenv per decode step, eager-loop noise.
pub fn pp2_split(n_layers: usize) -> Option<usize> {
    match std::env::var("MEMRA_PP_STAGES") {
        Ok(v) if v == "2" => {}
        Ok(v) if v.is_empty() || v == "0" || v == "1" => return None,
        Ok(v) => {
            warn_bad_once(&format!(
                "MEMRA_PP_STAGES={v} unsupported (increment 2 is 2-stage only); door stays OFF"
            ));
            return None;
        }
        Err(_) => return None,
    }
    let split = match std::env::var("MEMRA_PP_SPLIT") {
        Ok(v) => match v.parse::<usize>() {
            Ok(s) => s,
            Err(_) => {
                warn_bad_once(&format!("MEMRA_PP_SPLIT={v} unparseable; door stays OFF"));
                return None;
            }
        },
        Err(_) => n_layers / 2,
    };
    if split == 0 || split >= n_layers {
        warn_bad_once(&format!(
            "MEMRA_PP_SPLIT={split} outside [1, {}]; door stays OFF",
            n_layers - 1
        ));
        return None;
    }
    Some(split)
}

/// MEMRA_PP_STREAMS=0: rollback to the increment-1 same-stream seam (two dtod copies on
/// the ambient compute stream, no per-stage streams/events). Anything else = increment 2.
pub fn pp2_streams_off() -> bool {
    matches!(std::env::var("MEMRA_PP_STREAMS").as_deref(), Ok("0"))
}

/// MEMRA_PP_OVERLAP=1: double-buffered boundary slots (the M2 seed). Default OFF —
/// scheduling structure only, never math. Read per step so the gate can A/B in-process.
pub fn pp2_overlap() -> bool {
    matches!(std::env::var("MEMRA_PP_OVERLAP").as_deref(), Ok("1"))
}

/// Raw `MEMRA_PP_DEVICES` (parsed/validated at Pp2Rt build — a bad string must fail the
/// decode step loudly, never silently fall back to same-device and fake a gate PASS).
fn pp2_devices_env() -> Option<String> {
    std::env::var("MEMRA_PP_DEVICES").ok().filter(|v| !v.is_empty())
}

static WARNED_BAD: AtomicBool = AtomicBool::new(false);
fn warn_bad_once(msg: &str) {
    if !WARNED_BAD.swap(true, Ordering::Relaxed) {
        eprintln!("[pp2] {msg}");
    }
}

static WARNED_UNWIRED: AtomicBool = AtomicBool::new(false);
/// One-time notice when the door is set but the executing path has no pp2 arm
/// (increments 1-2 wire only the plain eager decode loops).
pub fn warn_unwired_once(path: &str) {
    if std::env::var("MEMRA_PP_STAGES").map(|v| v == "2").unwrap_or(false)
        && !WARNED_UNWIRED.swap(true, Ordering::Relaxed)
    {
        eprintln!("[pp2] MEMRA_PP_STAGES=2 set but `{path}` has no pp2 arm (increment 2); running unsplit");
    }
}

// ======================================================================================
//  Pp2Rt: the increment-2 transport runtime (per-stage streams, events, boundary slots)
// ======================================================================================

/// One pipeline stage's execution home: device, context, launch stream, and (for a stage
/// remote to the primary engine's device) a dedicated Engine in that device's context.
pub struct StageRt {
    pub dev: usize,
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    /// `Some` only when `dev` differs from the primary engine's device: CUmodules are
    /// per-context, so a remote stage needs its own Engine. Weights stay on the primary
    /// device (read via peer access); KV moves with `Cache::new_pp2`.
    engine: Option<Engine>,
}

/// One boundary slot: a persistent RX-side buffer + its TX/RX completion events.
/// PERSISTENT because the buffer is written by stage-0's stream and read by stage-1's:
/// a per-step alloc/free would enqueue the free on ONE stream while the other might
/// still be reading (the cross-stream free hazard) — a never-freed slot cannot race.
struct BoundarySlot {
    buf: Mutex<Option<CudaSlice<f32>>>,
    /// Recorded on stage-0's stream after the TX copy; RX waits on it. Created in
    /// stage-0's context (cuEventRecord requires event ctx == stream ctx).
    ev_tx: CudaEvent,
    /// Recorded on stage-1's stream after the RX copy; the NEXT TX into this slot waits
    /// on it (write-after-read guard). Created in stage-1's context. Waiting on a
    /// never-recorded event is a defined no-op, so step 0 needs no special case.
    ev_rx: CudaEvent,
}

pub struct Pp2Rt {
    stages: [StageRt; 2],
    /// true iff the two stages live on different devices (peer transport selected).
    cross: bool,
    slots: [BoundarySlot; 2],
    step: AtomicUsize,
}

static RT2: OnceLock<Result<Pp2Rt, String>> = OnceLock::new();

impl Pp2Rt {
    /// The process-wide transport runtime, built on first use against the primary engine.
    /// The device map freezes at first build (one MEMRA_PP_DEVICES config per process —
    /// the gate runs one placement per invocation). Build errors are sticky and loud.
    pub fn get(e: &Engine) -> Result<&'static Pp2Rt, Box<dyn std::error::Error>> {
        RT2.get_or_init(|| Self::build(e).map_err(|err| err.to_string()))
            .as_ref()
            .map_err(|s| -> Box<dyn std::error::Error> { s.clone().into() })
    }

    fn build(e: &Engine) -> Result<Pp2Rt, Box<dyn std::error::Error>> {
        let primary_dev = e.ctx().ordinal();
        let devices: [usize; 2] = match pp2_devices_env() {
            None => [primary_dev, primary_dev],
            Some(s) => {
                let parts: Vec<_> = s.split(',').map(|p| p.trim().parse::<usize>()).collect();
                match (parts.len(), parts.first(), parts.get(1)) {
                    (2, Some(Ok(a)), Some(Ok(b))) => [*a, *b],
                    _ => {
                        return Err(format!(
                            "MEMRA_PP_DEVICES={s} unparseable (want <d0>,<d1> e.g. 0,1)"
                        )
                        .into())
                    }
                }
            }
        };
        let cross = devices[0] != devices[1];
        if cross {
            // cuDeviceCanAccessPeer guard, BOTH directions: the boundary peer copy needs
            // one, stage kernels dereferencing the other device's weights need the other.
            let n = cudarc::driver::result::device::get_count()? as usize;
            for &d in &devices {
                if d >= n {
                    return Err(format!(
                        "MEMRA_PP_DEVICES={},{} but only {n} CUDA device(s) present",
                        devices[0], devices[1]
                    )
                    .into());
                }
            }
            for (a, b) in [(devices[0], devices[1]), (devices[1], devices[0])] {
                let da = cudarc::driver::result::device::get(a as i32)?;
                let db = cudarc::driver::result::device::get(b as i32)?;
                let mut can: i32 = 0;
                unsafe {
                    cudarc::driver::sys::cuDeviceCanAccessPeer(&mut can, da, db).result()?;
                }
                if can == 0 {
                    return Err(format!(
                        "device {a} cannot peer-access device {b} (cuDeviceCanAccessPeer=0); \
                         pp2 cross-device needs P2P — refusing a silently-staged path"
                    )
                    .into());
                }
            }
        }

        let mk_stage = |s: usize| -> Result<StageRt, Box<dyn std::error::Error>> {
            let dev = devices[s];
            if dev == primary_dev {
                let ctx = e.ctx().clone();
                let stream = ctx.new_stream()?;
                Ok(StageRt { dev, ctx, stream, engine: None })
            } else {
                let eng = Engine::new(dev)?;
                let ctx = eng.ctx().clone();
                let stream = ctx.new_stream()?;
                Ok(StageRt { dev, ctx, stream, engine: Some(eng) })
            }
        };
        let stages = [mk_stage(0)?, mk_stage(1)?];

        if cross {
            // Enable peer access BOTH ways (idempotent; ALREADY_ENABLED is success).
            for (me, peer) in [(0usize, 1usize), (1, 0)] {
                stages[me].ctx.bind_to_thread()?;
                let rc = unsafe {
                    cudarc::driver::sys::cuCtxEnablePeerAccess(stages[peer].ctx.cu_ctx(), 0)
                };
                use cudarc::driver::sys::cudaError_enum as E;
                if rc != E::CUDA_SUCCESS && rc != E::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED {
                    return Err(format!(
                        "cuCtxEnablePeerAccess(dev{} -> dev{}) failed: {rc:?}",
                        stages[me].dev, stages[peer].dev
                    )
                    .into());
                }
            }
            // MEM-POOL access grant (8x box 2026-08-02, cross-device fix #2):
            // cuCtxEnablePeerAccess does NOT map STREAM-ORDERED POOL allocations, and every
            // engine buffer/weight goes through the device default pool (cuMemAllocAsync via
            // cudarc; memra-runtime configures that pool). A stage-1 kernel dereferencing
            // dev0 weights — or the stage-0 peer TX writing dev1's RX slot — needs
            // cuMemPoolSetAccess on the OWNING device's default pool for the ACCESSING
            // device; without it the first remote dereference is CUDA_ERROR_ILLEGAL_ADDRESS
            // (reported at the next API call in the poisoned context). Grant both ways.
            for (owner, accessor) in [(stages[0].dev, stages[1].dev), (stages[1].dev, stages[0].dev)] {
                let dev = cudarc::driver::result::device::get(owner as i32)?;
                let mut pool: cudarc::driver::sys::CUmemoryPool = std::ptr::null_mut();
                unsafe {
                    cudarc::driver::sys::cuDeviceGetDefaultMemPool(&mut pool, dev).result()?;
                }
                let desc = cudarc::driver::sys::CUmemAccessDesc {
                    location: cudarc::driver::sys::CUmemLocation {
                        type_: cudarc::driver::sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
                        id: accessor as i32,
                    },
                    flags: cudarc::driver::sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
                };
                let rc = unsafe { cudarc::driver::sys::cuMemPoolSetAccess(pool, &desc, 1) };
                if rc != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                    return Err(format!(
                        "cuMemPoolSetAccess(dev{owner} pool -> dev{accessor}) failed: {rc:?}"
                    )
                    .into());
                }
            }
            // restore the primary context for the caller's subsequent work
            e.ctx().bind_to_thread()?;
            eprintln!(
                "[pp2] cross-device transport: stage0=dev{} stage1=dev{} (cudaMemcpyPeerAsync; \
                 peer access enabled both ways + default-pool access both ways; weights remain \
                 on dev{primary_dev})",
                stages[0].dev, stages[1].dev
            );
        }

        let mk_slot = |st: &[StageRt; 2]| -> Result<BoundarySlot, Box<dyn std::error::Error>> {
            Ok(BoundarySlot {
                buf: Mutex::new(None),
                ev_tx: st[0].ctx.new_event(None)?,
                ev_rx: st[1].ctx.new_event(None)?,
            })
        };
        let slots = [mk_slot(&stages)?, mk_slot(&stages)?];
        Ok(Pp2Rt { stages, cross, slots, step: AtomicUsize::new(0) })
    }

    /// True iff the boundary crosses devices (transport = cudaMemcpyPeerAsync).
    pub fn cross_device(&self) -> bool { self.cross }

    /// The engine a stage's subgraph must run through: the primary engine when the stage
    /// lives on the primary device, else the stage's own (remote-context) engine.
    pub fn engine<'a>(&'a self, s: usize, primary: &'a Engine) -> &'a Engine {
        self.stages[s].engine.as_ref().unwrap_or(primary)
    }

    /// Enter stage `s`: until the guard drops, every engine op on this thread launches on
    /// the stage's stream (memra_runtime ambient-stream override).
    pub fn enter(&self, s: usize) -> memra_runtime::StreamOverride {
        memra_runtime::push_stream_override(self.stages[s].stream.clone())
    }

    /// Boundary TX (call within the stage-0 scope; `x` = the materialized [n] residual):
    /// wait for the slot's previous RX (write-after-read guard), copy `x` into the slot's
    /// persistent buffer via the selected transport on stage-0's stream (the owning-
    /// stream/publication law), record ev_tx. Returns the slot index for the paired rx().
    pub fn tx(&self, x: &CudaSlice<f32>, n: usize) -> Result<usize, Box<dyn std::error::Error>> {
        assert_eq!(x.len(), n, "pp2 tx: residual length mismatch");
        let slot_idx = if pp2_overlap() {
            self.step.fetch_add(1, Ordering::Relaxed) % 2
        } else {
            0
        };
        let sl = &self.slots[slot_idx];
        let s0 = &self.stages[0].stream;
        s0.wait(&sl.ev_rx)?;
        let mut guard = sl.buf.lock().unwrap();
        if guard.as_ref().map(|b| b.len() != n).unwrap_or(true) {
            // allocated on the RX stage's stream: the buffer lives on the RX device.
            *guard = Some(self.stages[1].stream.alloc_zeros::<f32>(n)?);
        }
        let buf = guard.as_mut().unwrap();
        if !self.cross {
            s0.memcpy_dtod(x, buf)?;
        } else {
            // cudaMemcpyPeerAsync (M0: 2.8x NCCL at PP activation sizes), issued on the
            // publishing stage-0 stream with explicit src/dst contexts.
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            let (sp, _g0) = x.device_ptr(s0);
            let (dp, _g1) = buf.device_ptr_mut(s0);
            self.stages[0].ctx.bind_to_thread()?;
            unsafe {
                cudarc::driver::result::memcpy_peer_async(
                    self.stages[1].ctx.cu_ctx(),
                    dp,
                    self.stages[0].ctx.cu_ctx(),
                    sp,
                    n * std::mem::size_of::<f32>(),
                    s0.cu_stream(),
                )?;
            }
        }
        sl.ev_tx.record(s0)?;
        Ok(slot_idx)
    }

    /// Boundary RX (call within the stage-1 scope): wait on the slot's ev_tx, copy the
    /// boundary buffer into a fresh stage-1 working buffer (dtod on stage-1's stream —
    /// local on the RX device in both transports), record ev_rx. The returned buffer is
    /// stage-1-owned: allocated, consumed, and eventually freed on stage-1's stream.
    pub fn rx(&self, slot_idx: usize, n: usize)
              -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let sl = &self.slots[slot_idx];
        let s1 = &self.stages[1].stream;
        s1.wait(&sl.ev_tx)?;
        let guard = sl.buf.lock().unwrap();
        let buf = guard.as_ref().expect("pp2 rx before tx");
        // uninit working buffer (fully overwritten by the copy), allocated explicitly on
        // the stage stream so rx() is correct even outside an enter(1) scope.
        let mut work = unsafe { s1.alloc::<f32>(n)? };
        s1.memcpy_dtod(buf, &mut work)?;
        sl.ev_rx.record(s1)?;
        Ok(work)
    }
}

/// Stage-owned cache allocation door: when the pp2 door is open AND `MEMRA_PP_DEVICES`
/// is set (increment-2 placement plumbing), each layer's cache is allocated by its
/// OWNING stage's engine — on one device (`0,0`) this is byte-for-byte today's
/// allocation (gate it); cross-device it puts each stage's KV on that stage's HBM.
/// Door shut or devices unset: plain `Cache::new` (zero behavior change).
pub fn new_cache(e: &Engine, cfg: &memra_gguf::config::ModelConfig, max_ctx: usize)
                 -> Result<crate::cache::Cache, Box<dyn std::error::Error>> {
    if let Some(split) = pp2_split(cfg.n_layer as usize) {
        if pp2_devices_env().is_some() && !pp2_streams_off() {
            let rt = Pp2Rt::get(e)?;
            let d0 = rt.engine(0, e);
            let d1 = rt.engine(1, e);
            return crate::cache::Cache::new_pp2(d0, d1, split, cfg, max_ctx);
        }
    }
    crate::cache::Cache::new(e, cfg, max_ctx)
}
