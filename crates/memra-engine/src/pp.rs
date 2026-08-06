//! M2 pipeline-parallel N-stage runtime (generalizes the M1 2-stage seam).
//!
//! Door: `MEMRA_PP_STAGES=N` (default OFF — unset/0/1 = no behavior change anywhere).
//! Stage map: N stages over the trunk layers with N-1 cuts. `MEMRA_PP_SPLITS=c1,..,cN-1`
//! sets the cuts explicitly (strictly increasing, in (0, n_layers)); `MEMRA_PP_SPLIT=<i>`
//! is the N=2 back-compat spelling; default = even split (cut s = s*n_layers/N).
//! Placement: `MEMRA_PP_DEVICES=d0,..,dN-1` maps stage s to device ds (default: all on
//! the primary engine's device).
//!
//! M1 history (increments 1-2, merged + hardened on the 8x box 2026-08-02): seam + gate
//! single-device; then real transport — per-stage streams/events, device placement,
//! peer-copy boundary (M0: cudaMemcpyPeerAsync beats NCCL 2.8x at PP activation sizes),
//! per-context PDL module caches, default-mempool peer grants. All five r3 gates PASS
//! bit-identical (receipts ~/receipts/m1-pp2/ on darklanes-bench).
//!
//! M2 increment 1 (this file): N-STAGE GENERALIZATION — `Pp2Rt` becomes `PpNRt`:
//!   - `stages`: Vec of per-stage execution homes (device, context, stream, remote Engine);
//!   - `boundaries`: N-1 boundary runtimes, each with TWO persistent double-buffered slots
//!     (ev_tx/ev_rx per slot) and its own overlap step counter; transport is selected PER
//!     BOUNDARY (dtod same-device / cudaMemcpyPeerAsync cross-device);
//!   - peer + default-mempool access is granted between EVERY distinct pair of devices in
//!     use (stage devices + the primary): stage kernels may dereference the primary's
//!     weights (bring-up placement) and stage-0's pos_d, and each boundary peer-copies.
//!
//! M2 increment 2 (weight sharding): the loader uploads each stage's layer range THROUGH
//! that stage's engine (`layer_engine`), so weights land on the device that runs them —
//! the bring-up peer-read placement dies. `output_norm` + lm head load through the LAST
//! stage's engine; the embed table stays host-side with stage 0. Split-plane/f16 decode
//! mirrors are built per layer through the owning stage's engine too (the rp4 mirrors ARE
//! the decode weights on the q8 path — leaving them on dev0 would fake the kill).
//! Rollback seam: `MEMRA_PP_SHARD=0` = M1 bring-up placement (all weights on primary,
//! remote stages peer-read).
//!
//! M2 increment 3 (deferred readback — the pipelining seed): `PendingLogits` — the eager
//! decode arm can END a step without the logits D2H (`decode_step_h_ppn_deferred`): the
//! logits stay device-resident with a completion event; `wait()` drains them through a
//! DEDICATED readback stream (waits the event, copies, syncs) so tokens t+1.. keep
//! enqueuing on the stage streams while token t drains. Per-token math is fully
//! event-ordered (same slots, same ev_tx/ev_rx chain) — scheduling changes, math does
//! not; the pipelined replay arm of `ppn-gate` proves bit-identity per step.
//!
//! Ownership across a boundary (unchanged from M1):
//!   - hidden state [n_embd] f32 is the ONLY tensor that crosses;
//!   - KV/linear-attn cache entries are per-layer: stage s exclusively owns cache state
//!     for its layer range (and, under MEMRA_PP_DEVICES, allocates it on its device);
//!   - position/rope state is the scalar `cache.pos` snapshot taken once per step, uploaded
//!     on stage-0's stream BEFORE the first TX event — every later stage's wait chain
//!     transitively orders it (stage s waits boundary s-1's ev_tx, which was recorded after
//!     stage s-1's work, which waited boundary s-2's ev_tx, ... back to stage 0);
//!   - the embed table lives with stage 0, output_norm + lm head with the last stage.
//!
//! THE MULTI-STREAM LAW (why this is safe with cudarc event tracking disabled): all
//! cross-stage bytes flow through the persistent boundary slots, ordered by ev_tx/ev_rx;
//! per-stage scratch is allocated AND freed on that stage's stream (stream-ordered); the
//! async mem pool runs with opportunistic reuse OFF + internal dependencies ON
//! (memra-runtime), so a block freed on stream A and reused on stream B carries a
//! driver-inserted dependency. Weights are load-time state no stage stream can precede,
//! and the step's terminal logits readback (sync D2H, or PendingLogits' event-ordered
//! readback stream) drains the last stage, whose TX-wait chain transitively drains all.
//!
//! Scope: plain eager decode only (generic arm N-stage; gemma4 arm 2-stage). NOT wired:
//! batch/dc/graph/spec loops and the gemma4-E4B eager arm.
//!
//! CORRECTION (pp2-hardening 2026-08-06): this header used to add "(`warn_unwired_once`
//! fires)" to that list, which was wrong. `warn_unwired_once` has exactly two call sites
//! and BOTH are gemma4-specific (decode.rs, hybrid_forward.rs) — the batch/dc/graph/spec
//! loops never warned. Worse, the batched loop did not merely run unsplit: it walked the
//! whole trunk on the primary stream and, under a sharded cross-device placement,
//! peer-read every remote stage's weights each step — 28x slower at B=1 with all three
//! `decode-batch-gate` gates PASSING (peer reads are byte-exact, so only perf broke).
//! `decode_step_batch` now FAILS CLOSED in that regime via `pp_sharded_cross_device()`
//! (`MEMRA_PP_ALLOW_UNSPLIT_BATCH=1` = measurement override). "Unwired" for dc/graph/spec
//! still means "runs unsplit, silently" — audit each before trusting it on a pair.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{CudaContext, CudaEvent, CudaSlice, CudaStream};

use crate::Engine;

/// Returns the stage fence iff the ppN door is open: `MEMRA_PP_STAGES=N` (N >= 2) with a
/// valid cut list. The fence has N+1 entries: `[0, c1, .., cN-1, n_layers]`; stage s runs
/// layers `[fence[s], fence[s+1])`. Reads the environment on every call (gates toggle the
/// door in-process); the cost is a few getenv per decode step, eager-loop noise.
pub fn pp_cuts(n_layers: usize) -> Option<Vec<usize>> {
    let n_st: usize = match std::env::var("MEMRA_PP_STAGES") {
        Ok(v) if v.is_empty() || v == "0" || v == "1" => return None,
        Ok(v) => match v.parse::<usize>() {
            Ok(n) => n,
            Err(_) => {
                warn_bad_once(&format!("MEMRA_PP_STAGES={v} unparseable; door stays OFF"));
                return None;
            }
        },
        Err(_) => return None,
    };
    if n_st < 2 || n_st > n_layers {
        warn_bad_once(&format!(
            "MEMRA_PP_STAGES={n_st} outside [2, n_layers={n_layers}]; door stays OFF"
        ));
        return None;
    }
    let mut fence = Vec::with_capacity(n_st + 1);
    fence.push(0usize);
    if let Ok(s) = std::env::var("MEMRA_PP_SPLITS") {
        let parts: Result<Vec<usize>, _> =
            s.split(',').map(|p| p.trim().parse::<usize>()).collect();
        match parts {
            Ok(cuts) if cuts.len() == n_st - 1 => fence.extend(cuts),
            _ => {
                warn_bad_once(&format!(
                    "MEMRA_PP_SPLITS={s} invalid (want {} comma-separated cuts); door stays OFF",
                    n_st - 1
                ));
                return None;
            }
        }
    } else if let Ok(v) = std::env::var("MEMRA_PP_SPLIT") {
        // N=2 back-compat spelling. With N>2 a single split is ambiguous — fail the door
        // loudly rather than guess (a silent even-split would fake a gate config).
        if n_st != 2 {
            warn_bad_once(&format!(
                "MEMRA_PP_SPLIT={v} set with MEMRA_PP_STAGES={n_st}; use MEMRA_PP_SPLITS \
                 for N>2 — door stays OFF"
            ));
            return None;
        }
        match v.parse::<usize>() {
            Ok(c) => fence.push(c),
            Err(_) => {
                warn_bad_once(&format!("MEMRA_PP_SPLIT={v} unparseable; door stays OFF"));
                return None;
            }
        }
    } else {
        for s in 1..n_st {
            fence.push(s * n_layers / n_st);
        }
    }
    fence.push(n_layers);
    for w in fence.windows(2) {
        if w[0] >= w[1] {
            warn_bad_once(&format!(
                "pp stage fence {fence:?} not strictly increasing over [0, {n_layers}]; \
                 door stays OFF"
            ));
            return None;
        }
    }
    Some(fence)
}

/// N=2 back-compat view of the door (the gemma4 arm and `pp2-gate` are 2-stage): `Some(cut)`
/// iff the door is open with EXACTLY two stages.
pub fn pp2_split(n_layers: usize) -> Option<usize> {
    pp_cuts(n_layers).filter(|f| f.len() == 3).map(|f| f[1])
}

/// The stage that owns layer `il` under `fence` (see `pp_cuts`).
pub fn stage_of(fence: &[usize], il: usize) -> usize {
    debug_assert!(fence.len() >= 2);
    match fence[1..fence.len() - 1].binary_search(&il) {
        // fence[1..][k] == il means il is the FIRST layer of stage k+1
        Ok(k) => k + 1,
        Err(k) => k,
    }
}

/// MEMRA_PP_STREAMS=0: rollback to the increment-1 same-stream seam (boundary = two plain
/// dtod copies on the ambient compute stream, no per-stage streams/events/devices).
pub fn pp2_streams_off() -> bool {
    matches!(std::env::var("MEMRA_PP_STREAMS").as_deref(), Ok("0"))
}

/// True iff the ppN door would put TWO OR MORE stage streams on ONE device (devices
/// unset = all stages on the primary; or an explicit placement with a repeated device).
/// The deferred-readback (pipelined) arm is REFUSED in this regime: the 2026-08-02 x20
/// soak record — singledev pipelined 13/20 PASS default, 7 failures each diverging at a
/// different step (timing-race signature); MEMRA_PDL=0 went 20/20 on one soak but a
/// second same-config soak on the auto-gated build failed 2/20 (n2) and battery-4 failed
/// n4 — so PDL narrows the window without closing it, and the true root cause (same
/// Engine kernels concurrent on two streams of one device) is NOT fixed by any flag yet.
/// Cross-device pipelined (one stage stream per device) is 23/23 clean post-fix. Refuse
/// loudly rather than return silently-wrong logits. Env-only read (callable pre-runtime).
pub fn pp_multi_stream_same_device() -> bool {
    let stages_open = std::env::var("MEMRA_PP_STAGES")
        .map(|v| v.parse::<usize>().map(|n| n >= 2).unwrap_or(false))
        .unwrap_or(false);
    let devices = std::env::var("MEMRA_PP_DEVICES").ok().filter(|v| !v.is_empty());
    if (!stages_open && devices.is_none()) || pp2_streams_off() {
        return false;
    }
    match devices {
        None => true, // door open, no placement: every stage stream lands on the primary
        Some(s) => {
            let mut v: Vec<&str> = s.split(',').map(|p| p.trim()).collect();
            let n = v.len();
            v.sort_unstable();
            v.dedup();
            v.len() < n // repeated device = shared-device streams
        }
    }
}

/// True iff the ppN door is open AND the placement spans 2+ DISTINCT devices AND the
/// per-stage sharded loader is on — i.e. some layers' weights live on a device other than
/// the primary. Any path that walks the WHOLE trunk on one stream in this regime reads
/// those weights over PCIe every step. Env-only read (callable pre-runtime).
///
/// Measured cost of doing that (pp2-hardening 2026-08-06, 2x RTX PRO 6000, PCIe Gen5 x16
/// P2P, decode-batch-bench q9, N=5 interleaved, `research/pp2-hardening-20260806`):
/// **B=1 7.4 vs 208.9 tok/s (28x), B=4 29.8 vs 491.3 (16.5x), B=8 47.4 vs 657.0 (13.9x)**.
/// The same sweep with `MEMRA_PP_SHARD=0` (weights all home) returns 178.5/491.1/656.6 —
/// identical to the single-device door-open arm — so the entire cliff is the peer read,
/// not the door and not the placement plumbing. Exactness is NOT the issue: peer reads
/// return identical bytes and every `decode-batch-gate` gate PASSED on this config, which
/// is precisely why it needs a refusal rather than a gate.
pub fn pp_sharded_cross_device() -> bool {
    let stages_open = std::env::var("MEMRA_PP_STAGES")
        .map(|v| v.parse::<usize>().map(|n| n >= 2).unwrap_or(false))
        .unwrap_or(false);
    // MEMRA_PP_STREAMS=0 (2026-08-06, pp2-batch): the same-stream rollback seam ALSO turns
    // the sharded loader off — `layer_engine` returns the primary engine whenever
    // `pp2_streams_off()`, and `new_cache` skips `Cache::new_ppn` on the same condition. So
    // in that regime every weight and every cache is home on the primary and an unsplit walk
    // peer-reads NOTHING. Without this term the guard refused that config too: a spurious
    // refusal of a placement that is sound and full-speed. Found wiring the batched pp arm.
    if !stages_open || pp_shard_off() || pp2_streams_off() {
        return false;
    }
    match pp2_devices_env() {
        None => false, // no placement: every stage is the primary device, nothing remote
        Some(s) => {
            let mut v: Vec<&str> = s.split(',').map(|p| p.trim()).collect();
            v.sort_unstable();
            v.dedup();
            v.len() >= 2
        }
    }
}

/// The shared fail-closed guard for EVERY decode path that has no pp stage split.
/// Returns `Err` iff `pp_sharded_cross_device()` — i.e. the caller would walk the whole
/// trunk on one stream while some layers' weights live on another device, peer-reading
/// them every step. `path` names the refusing function so the operator knows which loop
/// they hit; `alt` names the working alternative for that loop.
///
/// One helper rather than four copies because the audit found FOUR paths with the same
/// hole (`decode_step_batch`, `decode_step_dc`, the graph capture that wraps dc, and
/// `decode_step_t*` verify), and a per-path copy is how one gets missed on the next
/// addition. Override: `MEMRA_PP_ALLOW_UNSPLIT_BATCH=1` (one door for all of them —
/// they are the same measurement question).
pub fn refuse_unsplit_if_remote(path: &str, alt: &str) -> Result<(), Box<dyn std::error::Error>> {
    if pp_sharded_cross_device()
        && std::env::var("MEMRA_PP_ALLOW_UNSPLIT_BATCH").as_deref() != Ok("1")
    {
        return Err(format!(
            "{path}: refused with the ppN door open across 2+ devices — this path has no pp \
             stage split, so it would walk ALL layers on one stream and peer-read every \
             remote stage's weights each step (measured 28x slower at B=1, 13.9x at B=8 on \
             a PRO 6000 pair over PCIe Gen5 x16 P2P; research/pp2-hardening-20260806). \
             Exactness is unaffected — peer reads return identical bytes and the exactness \
             gates PASS on this config — which is exactly why it must refuse instead of \
             being caught by a gate. Fixes, in order: {alt}; or MEMRA_PP_SHARD=0 (all \
             weights home on the primary — full speed, forfeits the capacity PP-2 exists \
             for); or close the pp door. MEMRA_PP_ALLOW_UNSPLIT_BATCH=1 overrides for \
             measurement."
        )
        .into());
    }
    Ok(())
}

/// MEMRA_BATCH_PP=0: rollback/A-B seam for the BATCHED stage split (pp2-batch 2026-08-06).
/// Default ON — with the ppN door open the batched decode step takes its own stage split
/// (`decode_step_batch_ppn`) exactly as the eager step does. Setting 0 sends the batched
/// path back through the unsplit body, which under a sharded cross-device placement is
/// then caught by `refuse_unsplit_if_remote` (the 28x peer-read regime) rather than run
/// silently. Exists so the bit-identity gate can A/B split vs unsplit IN ONE PROCESS
/// against the same loaded weights — read per step, never memoized, for that reason.
pub fn batch_pp_on() -> bool {
    std::env::var("MEMRA_BATCH_PP").as_deref() != Ok("0")
}

/// MEMRA_PP_OVERLAP=1: alternate the double-buffered boundary slots per step (the
/// pipelining seed). Default OFF — scheduling structure only, never math. Read per step
/// so gates can A/B in-process.
pub fn pp2_overlap() -> bool {
    matches!(std::env::var("MEMRA_PP_OVERLAP").as_deref(), Ok("1"))
}

/// M2 increment 2 rollback seam: MEMRA_PP_SHARD=0 = the M1 bring-up placement (all
/// weights upload through the primary engine; remote stages peer-read). Default ON —
/// under MEMRA_PP_DEVICES each stage's layer range uploads through its own engine.
pub fn pp_shard_off() -> bool {
    matches!(std::env::var("MEMRA_PP_SHARD").as_deref(), Ok("0"))
}

/// Raw `MEMRA_PP_DEVICES` (parsed/validated at PpNRt build — a bad string must fail the
/// decode step loudly, never silently fall back to same-device and fake a gate PASS).
fn pp2_devices_env() -> Option<String> {
    std::env::var("MEMRA_PP_DEVICES").ok().filter(|v| !v.is_empty())
}

static WARNED_BAD: AtomicBool = AtomicBool::new(false);
fn warn_bad_once(msg: &str) {
    if !WARNED_BAD.swap(true, Ordering::Relaxed) {
        eprintln!("[pp] {msg}");
    }
}

static WARNED_UNWIRED: AtomicBool = AtomicBool::new(false);
/// One-time notice when the door is set but the executing path has no pp arm
/// (M2 wires the generic eager decode at any N and the gemma4 eager arm at N=2).
pub fn warn_unwired_once(path: &str) {
    let open = std::env::var("MEMRA_PP_STAGES")
        .map(|v| !v.is_empty() && v != "0" && v != "1")
        .unwrap_or(false);
    if open && !WARNED_UNWIRED.swap(true, Ordering::Relaxed) {
        eprintln!(
            "[pp] MEMRA_PP_STAGES set but `{path}` has no pp arm at this N; running unsplit"
        );
    }
}

// ======================================================================================
//  PpNRt: the M2 transport runtime (per-stage streams, per-boundary events + slots)
// ======================================================================================

/// One pipeline stage's execution home: device, context, launch stream, and (for a stage
/// remote to the primary engine's device) a dedicated Engine in that device's primary
/// context (CUmodules are per-context).
pub struct StageRt {
    pub dev: usize,
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    /// `Some` only when `dev` differs from the primary engine's device.
    engine: Option<Engine>,
}

/// One boundary slot: a persistent RX-side buffer + its TX/RX completion events.
/// PERSISTENT because the buffer is written by the TX stage's stream and read by the RX
/// stage's: a per-step alloc/free would enqueue the free on ONE stream while the other
/// might still be reading (the cross-stream free hazard) — a never-freed slot cannot race.
struct BoundarySlot {
    buf: Mutex<Option<CudaSlice<f32>>>,
    /// Recorded on the TX stage's stream after the TX copy; RX waits on it. Created in
    /// the TX stage's context (cuEventRecord requires event ctx == stream ctx).
    ev_tx: CudaEvent,
    /// Recorded on the RX stage's stream after the RX copy; the NEXT TX into this slot
    /// waits on it (write-after-read guard). Created in the RX stage's context. Waiting
    /// on a never-recorded event is a defined no-op, so step 0 needs no special case.
    ev_rx: CudaEvent,
}

/// Boundary b sits between stage b (TX) and stage b+1 (RX). Two slots, alternating per
/// step under MEMRA_PP_OVERLAP=1 (each boundary counts its own steps — a decode step
/// crosses every boundary exactly once, so the counters stay in lockstep).
struct BoundaryRt {
    slots: [BoundarySlot; 2],
    step: AtomicUsize,
    /// true iff stage b and stage b+1 live on different devices (peer transport).
    cross: bool,
}

pub struct PpNRt {
    stages: Vec<StageRt>,
    boundaries: Vec<BoundaryRt>,
    /// true iff ANY boundary crosses devices.
    cross_any: bool,
    /// Dedicated readback stream in the LAST stage's context (deferred logits D2H —
    /// waiting there instead of on the compute stream keeps later tokens enqueuable).
    readback: Arc<CudaStream>,
}

/// M1 name kept alive for external callers (`pp-transport-smoke`, receipts, docs).
pub type Pp2Rt = PpNRt;

static RTN: OnceLock<Result<PpNRt, String>> = OnceLock::new();

impl PpNRt {
    /// The process-wide transport runtime, built on first use against the primary engine.
    /// The stage count + device map freeze at first build (one config per process — gates
    /// run one placement per invocation). Build errors are sticky and loud.
    pub fn get(e: &Engine) -> Result<&'static PpNRt, Box<dyn std::error::Error>> {
        RTN.get_or_init(|| Self::build(e).map_err(|err| err.to_string()))
            .as_ref()
            .map_err(|s| -> Box<dyn std::error::Error> { s.clone().into() })
    }

    fn build(e: &Engine) -> Result<PpNRt, Box<dyn std::error::Error>> {
        let primary_dev = e.ctx().ordinal();
        // Stage count: MEMRA_PP_DEVICES length wins when set (it IS the placement);
        // else MEMRA_PP_STAGES; else 2 (the M1 default — pp-transport-smoke runs doorless).
        let devices: Vec<usize> = match pp2_devices_env() {
            Some(s) => {
                let parts: Result<Vec<usize>, _> =
                    s.split(',').map(|p| p.trim().parse::<usize>()).collect();
                match parts {
                    Ok(v) if v.len() >= 2 => v,
                    _ => {
                        return Err(format!(
                            "MEMRA_PP_DEVICES={s} unparseable (want <d0>,..,<dN-1> e.g. 0,1,2,3)"
                        )
                        .into())
                    }
                }
            }
            None => {
                let n_st = std::env::var("MEMRA_PP_STAGES")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .filter(|&n| n >= 2)
                    .unwrap_or(2);
                vec![primary_dev; n_st]
            }
        };
        if let Ok(v) = std::env::var("MEMRA_PP_STAGES") {
            if let Ok(n) = v.parse::<usize>() {
                if n >= 2 && n != devices.len() {
                    return Err(format!(
                        "MEMRA_PP_DEVICES lists {} devices but MEMRA_PP_STAGES={n} — \
                         refusing an ambiguous placement",
                        devices.len()
                    )
                    .into());
                }
            }
        }
        let n_st = devices.len();
        let cross_any = devices.iter().any(|&d| d != devices[0]);

        // Every distinct device pair in use must peer-access BOTH ways: boundaries copy
        // between consecutive stages, stage kernels may dereference primary-device weights
        // (bring-up placement / MEMRA_PP_SHARD=0) and stage-0's pos_d upload.
        let mut used: Vec<usize> = devices.clone();
        used.push(primary_dev);
        used.sort_unstable();
        used.dedup();
        if used.len() > 1 {
            let n = cudarc::driver::result::device::get_count()? as usize;
            for &d in &used {
                if d >= n {
                    return Err(format!(
                        "MEMRA_PP_DEVICES={devices:?} but only {n} CUDA device(s) present"
                    )
                    .into());
                }
            }
            for &a in &used {
                for &b in &used {
                    if a == b {
                        continue;
                    }
                    let da = cudarc::driver::result::device::get(a as i32)?;
                    let db = cudarc::driver::result::device::get(b as i32)?;
                    let mut can: i32 = 0;
                    unsafe {
                        cudarc::driver::sys::cuDeviceCanAccessPeer(&mut can, da, db).result()?;
                    }
                    if can == 0 {
                        return Err(format!(
                            "device {a} cannot peer-access device {b} (cuDeviceCanAccessPeer=0); \
                             ppN cross-device needs P2P — refusing a silently-staged path"
                        )
                        .into());
                    }
                }
            }
        }

        // PER-STAGE ENGINE ISOLATION (2026-08-02 singledev pipelined find): Engine owns
        // lazily-grown SHARED scratch pools (fa_part_pool, fa_vf16_scratch, argmax
        // partials, ...) that are stable-pointer by design — safe on one stream, a data
        // race the moment two stage streams run concurrently through the SAME Engine
        // (deferred readback, >=2 tokens in flight: token t+1's stage-0 fa memsets the
        // partials while token t's stage-s fa still reads them — the nondeterministic
        // all-logits divergence; cross-device arms were immune because remote stages
        // already got their own Engine). Every stage s>0 gets its OWN Engine even on the
        // primary device: same CUcontext (primary retain), so the per-context CUmodule
        // cache makes it cheap; scratch pools are per-Engine, so stages never share.
        // Stage 0 keeps the primary engine (single-threaded host issue: the only
        // concurrent user of `e` during a pp walk is stage 0 itself).
        let mk_stage = |dev: usize, s: usize| -> Result<StageRt, Box<dyn std::error::Error>> {
            if dev == primary_dev && s == 0 {
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
        let mut stages = Vec::with_capacity(n_st);
        for (s, &d) in devices.iter().enumerate() {
            stages.push(mk_stage(d, s)?);
        }

        if used.len() > 1 {
            // A context per distinct device (first stage that lives there; the primary's
            // context for the primary device).
            let ctx_of = |d: usize| -> &Arc<CudaContext> {
                if d == primary_dev {
                    e.ctx()
                } else {
                    &stages.iter().find(|s| s.dev == d).unwrap().ctx
                }
            };
            // Enable peer access BOTH ways for every distinct pair (idempotent;
            // ALREADY_ENABLED is success).
            for &a in &used {
                for &b in &used {
                    if a == b {
                        continue;
                    }
                    ctx_of(a).bind_to_thread()?;
                    let rc = unsafe {
                        cudarc::driver::sys::cuCtxEnablePeerAccess(ctx_of(b).cu_ctx(), 0)
                    };
                    use cudarc::driver::sys::cudaError_enum as E;
                    if rc != E::CUDA_SUCCESS && rc != E::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED {
                        return Err(format!(
                            "cuCtxEnablePeerAccess(dev{a} -> dev{b}) failed: {rc:?}"
                        )
                        .into());
                    }
                }
            }
            // MEM-POOL access grant (8x box 2026-08-02, M1 cross-device fix #2):
            // cuCtxEnablePeerAccess does NOT map STREAM-ORDERED POOL allocations, and every
            // engine buffer/weight goes through the device default pool (cuMemAllocAsync via
            // cudarc; memra-runtime configures that pool). A stage kernel dereferencing
            // another device's weights — or a boundary peer TX writing the RX slot — needs
            // cuMemPoolSetAccess on the OWNING device's default pool for the ACCESSING
            // device; without it the first remote dereference is CUDA_ERROR_ILLEGAL_ADDRESS
            // (reported at the next API call in the poisoned context). Grant all pairs.
            for &owner in &used {
                for &accessor in &used {
                    if owner == accessor {
                        continue;
                    }
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
                "[pp] cross-device transport: {} (cudaMemcpyPeerAsync per cross boundary; \
                 peer + default-pool access granted all pairs over {used:?}; weight home: {})",
                devices
                    .iter()
                    .enumerate()
                    .map(|(s, d)| format!("stage{s}=dev{d}"))
                    .collect::<Vec<_>>()
                    .join(" "),
                if pp_shard_off() {
                    format!("dev{primary_dev} (MEMRA_PP_SHARD=0 bring-up placement)")
                } else {
                    "per-stage (sharded loader)".to_string()
                }
            );
        }

        let mk_slot = |tx: &StageRt, rx: &StageRt| -> Result<BoundarySlot, Box<dyn std::error::Error>> {
            Ok(BoundarySlot {
                buf: Mutex::new(None),
                ev_tx: tx.ctx.new_event(None)?,
                ev_rx: rx.ctx.new_event(None)?,
            })
        };
        let mut boundaries = Vec::with_capacity(n_st - 1);
        for b in 0..n_st - 1 {
            let (tx, rx) = (&stages[b], &stages[b + 1]);
            boundaries.push(BoundaryRt {
                slots: [mk_slot(tx, rx)?, mk_slot(tx, rx)?],
                step: AtomicUsize::new(0),
                cross: tx.dev != rx.dev,
            });
        }
        let readback = stages[n_st - 1].ctx.new_stream()?;
        Ok(PpNRt { stages, boundaries, cross_any, readback })
    }

    pub fn n_stages(&self) -> usize {
        self.stages.len()
    }

    /// True iff any boundary crosses devices (transport = cudaMemcpyPeerAsync there).
    pub fn cross_device(&self) -> bool {
        self.cross_any
    }

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

    /// Boundary TX at boundary `b` (call within the stage-`b` scope; `x` = the
    /// materialized [n] residual): wait for the slot's previous RX (write-after-read
    /// guard), copy `x` into the slot's persistent buffer via the boundary's transport on
    /// stage-b's stream (the owning-stream/publication law), record ev_tx. Returns the
    /// slot index for the paired rx().
    ///
    /// `n` is the PAYLOAD ELEMENT COUNT, not a fixed model constant: the eager arm passes
    /// `n_embd` (one row), the batched arm passes `b_n * n_embd` (B stacked rows, the
    /// [B, n_embd] boundary). The slot buffer is GROW-ONLY and the transport moves exactly
    /// the first `n` elements — batched serving changes B every tick (chunk fill), and a
    /// realloc-on-every-size-change would host-sync the RX stream per width change (see the
    /// SLOT FIRST-USE ORDERING note below for why each allocation needs that sync). Growing
    /// to the high-water mark makes the syncs O(distinct widths) instead of O(width changes).
    pub fn tx(&self, b: usize, x: &CudaSlice<f32>, n: usize)
              -> Result<usize, Box<dyn std::error::Error>> {
        assert_eq!(x.len(), n, "pp tx: residual length mismatch");
        let bd = &self.boundaries[b];
        let slot_idx = if pp2_overlap() {
            bd.step.fetch_add(1, Ordering::Relaxed) % 2
        } else {
            0
        };
        let sl = &bd.slots[slot_idx];
        let s_tx = &self.stages[b].stream;
        s_tx.wait(&sl.ev_rx)?;
        let mut guard = sl.buf.lock().unwrap();
        if guard.as_ref().map(|bf| bf.len() < n).unwrap_or(true) {
            // allocated on the RX stage's stream: the buffer lives on the RX device.
            let s_rx = &self.stages[b + 1].stream;
            *guard = Some(s_rx.alloc_zeros::<f32>(n)?);
            // SLOT FIRST-USE ORDERING (2026-08-02 pipelined-gate find): the lazy alloc's
            // pool-alloc + memset enqueue on the RX stream; the TX copy below issues on
            // the TX stream, and on a slot's FIRST use ev_rx has never been recorded —
            // nothing orders them. With >=2 tokens in flight the RX stream is still busy
            // with the previous token, the memset lands AFTER the TX copy, and the
            // boundary residual is zeroed (window=1 passed, window>=2 failed at the
            // slot-1 first-use step; -overlap arms passed because the synchronous serial
            // arm pre-warmed both slots). Host-sync the RX stream once per slot
            // allocation — at most 2*(N-1) one-time syncs per process, all during prime.
            s_rx.synchronize()?;
        }
        let buf = guard.as_mut().unwrap();
        if !bd.cross {
            s_tx.memcpy_dtod(x, buf)?;
        } else {
            // cudaMemcpyPeerAsync (M0: 2.8x NCCL at PP activation sizes), issued on the
            // publishing TX stream with explicit src/dst contexts.
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            let (sp, _g0) = x.device_ptr(s_tx);
            let (dp, _g1) = buf.device_ptr_mut(s_tx);
            self.stages[b].ctx.bind_to_thread()?;
            unsafe {
                cudarc::driver::result::memcpy_peer_async(
                    self.stages[b + 1].ctx.cu_ctx(),
                    dp,
                    self.stages[b].ctx.cu_ctx(),
                    sp,
                    n * std::mem::size_of::<f32>(),
                    s_tx.cu_stream(),
                )?;
            }
        }
        sl.ev_tx.record(s_tx)?;
        Ok(slot_idx)
    }

    /// Boundary RX at boundary `b` (call within the stage-`b+1` scope): wait on the slot's
    /// ev_tx, copy the boundary buffer into a fresh working buffer (dtod on the RX stream —
    /// local on the RX device in both transports), record ev_rx. The returned buffer is
    /// RX-stage-owned: allocated, consumed, and eventually freed on that stage's stream.
    pub fn rx(&self, b: usize, slot_idx: usize, n: usize)
              -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let sl = &self.boundaries[b].slots[slot_idx];
        let s_rx = &self.stages[b + 1].stream;
        s_rx.wait(&sl.ev_tx)?;
        let guard = sl.buf.lock().unwrap();
        let buf = guard.as_ref().expect("pp rx before tx");
        assert!(buf.len() >= n, "pp rx: slot holds {} < requested {n}", buf.len());
        // uninit working buffer (fully overwritten by the copy), allocated explicitly on
        // the stage stream so rx() is correct even outside an enter() scope.
        let mut work = unsafe { s_rx.alloc::<f32>(n)? };
        // Slice the slot to the payload: the buffer is grow-only (see tx), so at a narrower
        // width it is LONGER than `work` and cudarc's memcpy_dtod (dst.len() >= src.len())
        // would assert. The paired tx wrote exactly these first n elements.
        s_rx.memcpy_dtod(&buf.slice(0..n), &mut work)?;
        sl.ev_rx.record(s_rx)?;
        Ok(work)
    }

    /// Deferred readback: record a fresh completion event on the LAST stage's stream
    /// (call after the step's logits matmul has been enqueued there).
    pub fn record_done(&self) -> Result<CudaEvent, Box<dyn std::error::Error>> {
        let last = &self.stages[self.stages.len() - 1];
        let ev = last.ctx.new_event(None)?;
        ev.record(&last.stream)?;
        Ok(ev)
    }

    /// The dedicated readback stream (last stage's context).
    pub fn readback_stream(&self) -> &Arc<CudaStream> {
        &self.readback
    }
}

/// M2 increment 3: a step's logits, still device-resident on the LAST stage. `wait()`
/// orders the readback stream behind the step's completion event, copies, and syncs —
/// tokens enqueued after this step keep running on the stage streams while the caller
/// drains token t. Dropping without waiting is safe (buffers free stream-ordered).
pub struct PendingLogits {
    logits: CudaSlice<f32>,
    ev: CudaEvent,
    rb: Arc<CudaStream>,
}

impl PendingLogits {
    pub fn new(logits: CudaSlice<f32>, ev: CudaEvent, rb: Arc<CudaStream>) -> Self {
        PendingLogits { logits, ev, rb }
    }

    /// Blocks until this step's logits are computed, returns them host-side. Only this
    /// step's work is waited on (event-ordered) — NOT later tokens already enqueued on
    /// the stage streams.
    pub fn wait(self) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        self.rb.wait(&self.ev)?;
        let host = self.rb.clone_dtoh(&self.logits)?;
        self.rb.synchronize()?;
        // logits drop AFTER the sync: the D2H has fully completed, so the stream-ordered
        // free on the compute stream cannot race the copy.
        Ok(host)
    }
}

/// Stage-owned cache allocation door: when the ppN door is open AND `MEMRA_PP_DEVICES`
/// is set (placement plumbing), each layer's cache is allocated by its OWNING stage's
/// engine — on one device this is byte-for-byte today's allocation (gated); cross-device
/// it puts each stage's KV on that stage's HBM. Door shut or devices unset: plain
/// `Cache::new` (zero behavior change). Trailing MTP/NextN layers (beyond the trunk)
/// map to the LAST stage.
pub fn new_cache(e: &Engine, cfg: &memra_gguf::config::ModelConfig, max_ctx: usize)
                 -> Result<crate::cache::Cache, Box<dyn std::error::Error>> {
    let n_trunk = (cfg.n_layer - cfg.nextn_predict_layers) as usize;
    if let Some(fence) = pp_cuts(n_trunk) {
        if pp2_devices_env().is_some() && !pp2_streams_off() {
            let rt = PpNRt::get(e)?;
            let n_st = fence.len() - 1;
            assert_eq!(
                rt.n_stages(), n_st,
                "PpNRt stage count {} != fence stages {n_st}", rt.n_stages()
            );
            let devs: Vec<&dyn memra_kv::KvDev> =
                (0..n_st).map(|s| rt.engine(s, e) as &dyn memra_kv::KvDev).collect();
            let cache = crate::cache::Cache::new_ppn(&devs, &fence, cfg, max_ctx)?;
            sync_stages_after_load(e, n_trunk)?;
            return Ok(cache);
        }
        if !pp2_streams_off() {
            // CACHE BIRTH BARRIER (2026-08-02 pipelined-arm residual race): with the door
            // open but no device placement, Cache::new's alloc_zeros memsets enqueue on
            // the PRIMARY worker stream while the first KV appends / recurrent-state
            // reads run on the per-stage streams — no event orders them, and under
            // deferred readback the stage streams are hot immediately (a memset tail
            // can zero an already-appended KV row; intermittent, ~1-in-3 gate FAIL).
            // One context-sync per cache creation kills the class.
            let cache = crate::cache::Cache::new(e, cfg, max_ctx)?;
            sync_stages_after_load(e, n_trunk)?;
            return Ok(cache);
        }
    }
    crate::cache::Cache::new(e, cfg, max_ctx)
}

/// M2 increment 2 LOAD BARRIER: weight uploads and decode-mirror builds enqueue on the
/// loading engines' WORKER streams; the first consumer launches on a DIFFERENT stream
/// with no load->decode event — the door-off reference walk on the primary worker
/// stream (sharded load: remote builds still in flight), or a fresh per-stage stream.
/// The 2026-08-02 gate finds (n2-dev01 step-0 168k-logit graze; split5 ref=0.0 head —
/// a half-built rp4 mirror — poisoning step-0 KV and every later step): one
/// context-wide synchronize per stage at load end kills the class. No-op when the door
/// is shut at load (single-stream load+decode is ordered by the stream itself).
pub fn sync_stages_after_load(e: &Engine, n_trunk: usize)
                              -> Result<(), Box<dyn std::error::Error>> {
    if pp2_streams_off() || pp_cuts(n_trunk).is_none() {
        return Ok(());
    }
    let rt = PpNRt::get(e)?;
    for s in 0..rt.n_stages() {
        rt.stages[s].ctx.bind_to_thread()?;
        unsafe {
            cudarc::driver::sys::cuCtxSynchronize().result()?;
        }
    }
    e.ctx().bind_to_thread()?;
    unsafe {
        cudarc::driver::sys::cuCtxSynchronize().result()?;
    }
    Ok(())
}

/// M2 increment 2 (weight sharding): the engine that should UPLOAD layer `il`'s weights
/// (and build its decode mirrors) — the owning stage's engine when the door is open with
/// device placement and sharding not rolled back; else the primary. `il >= n_trunk`
/// (MTP/NextN blocks) maps to the last stage. The head (output_norm + lm head) belongs
/// to the last trunk layer's stage — call with `il = n_trunk - 1`.
pub fn layer_engine<'a>(e: &'a Engine, n_trunk: usize, il: usize)
                        -> Result<&'a Engine, Box<dyn std::error::Error>> {
    if pp_shard_off() || pp2_devices_env().is_none() || pp2_streams_off() {
        return Ok(e);
    }
    let Some(fence) = pp_cuts(n_trunk) else { return Ok(e) };
    let rt = PpNRt::get(e)?;
    let s = stage_of(&fence, il.min(n_trunk - 1));
    Ok(rt.engine(s, e))
}
