//! Dead-darklane background jobs — valley detection + a yield-first job runner
//! (lane/darklane-training, 2026-08-07).
//!
//! THE THESIS (owner, standing): idle serve capacity carries owner research/training jobs,
//! yielding instantly to paying traffic. This module is the ENGINE half only: when the box
//! is idle, what runs, and how it gets out of the way. Scheduling policy and economics
//! (which jobs, what they're worth, when a valley is "worth" filling) belong to the product
//! repo — the seam is `MEMRA_BG_JOB` + the checkpoint protocol below.
//!
//! VALLEY DETECTION reuses worker truth that already exists instead of inventing a new
//! sensor: the scheduler flips `health` to `PHASE_IDLE` exactly when `active.is_empty() &&
//! queue.is_empty()` (worker.rs loop top) and `set_phase` stamps the beat on entry — so
//! `phase == IDLE` + `beat_age_ms` IS the idle duration, to the millisecond, with zero new
//! hot-path cost. `PENDING_ADMITS` closes the HTTP→worker handoff gap (a request the handler
//! has submitted but the worker hasn't popped yet is traffic, not idleness).
//!
//! THE LANE CLASS: below EVERY serving lane. Harvest is still a *request* class the engine
//! admits and schedules; a background job is not a request at all — it runs only while the
//! engine has NOTHING (no interactive, no judge, no harvest, no queue) and yields on the
//! first sign of any of them. Asymmetric hysteresis on purpose: yield fires on the busy
//! EDGE (any activity, no debounce — paying traffic never waits for a threshold), resume
//! waits for a full `MEMRA_VALLEY_S` of quiet (a between-requests gap in an active
//! conversation is not a valley).
//!
//! YIELD MECHANISM v1 — simplest honest first: the job is a CHILD PROCESS in its own
//! process group; yield is SIGSTOP to the group, resume is SIGCONT. Bounded by the poll
//! interval (`MEMRA_BG_POLL_MS`, default 25 ms) + signal delivery — measured receipts in
//! `research/darktrain-20260807/`. Two consequences the operator must know:
//!   * a SIGSTOPPED process KEEPS its memory — VRAM included. The VRAM budget below is
//!     therefore carved out for the LIFE of the job, not per-valley.
//!   * SIGSTOP is only cheap for jobs whose working set can sit cold. GPU-resident training
//!     that can't sit on stopped VRAM uses checkpoint mode instead.
//!
//! CHECKPOINT PROTOCOL v1 (`MEMRA_BG_YIELD_MODE=checkpoint`) — the seam for training-class
//! jobs. The "checkpoint callback" is process-level: SIGUSR1 to the job's group means
//! "checkpoint NOW and exit"; the job writes its state to disk and exits **75**
//! (EX_TEMPFAIL, the sysexits convention for "transient — retry me"). The runner relaunches
//! the SAME command in the next valley and the job resumes from its own checkpoint file.
//! Exit 0 = complete (never relaunched); any other exit = failed (never relaunched, loud).
//! A job that outlives `MEMRA_BG_CKPT_GRACE_MS` after SIGUSR1 is SIGKILLed — the yield
//! bound holds even against a wedged job, and the on-disk checkpoint is whatever it last
//! wrote (at-least-once semantics: a training step may repeat, never be lost). Toy proof:
//! `tools/bg-ckpt-counter.py`. An in-process trainer API can replace this seam later
//! without changing the valley/scheduler half.
//!
//! GPU MEMORY DISCIPLINE: the job gets a VRAM budget (`MEMRA_BG_VRAM_MB`, default 0 =
//! CPU-only) from the serve headroom and is REFUSED at launch if it doesn't fit:
//! launch requires `min free across visible GPUs >= budget + serve headroom`, where the
//! headroom term composes with the existing `MEMRA_MOE_RESIDENT_HEADROOM_GB` pattern
//! (default 2.0 GB — the same class of reserve the resident planner keeps beside weights).
//! Fail-closed: a budget > 0 with no readable `nvidia-smi` is a refusal, not a shrug. The
//! runner enforces fit at launch; staying inside the budget at runtime is the job's
//! contract (documented in FLAGS.md) — v1 has no cgroup-style VRAM enforcement.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, Ordering};

use crate::health::{PHASE_IDLE, SharedHealth};
use crate::worker::PENDING_ADMITS;

// ---------------------------------------------------------------------------
// Valley detection
// ---------------------------------------------------------------------------

/// `MEMRA_VALLEY_S` (default 2.0): how long the worker must be COMPLETELY idle (no active
/// sessions, no queued admissions, no pending HTTP handoffs) before the box is "in a
/// valley". Read once — the threshold must not move under a running process.
pub fn valley_threshold_s() -> f64 {
    static T: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *T.get_or_init(|| {
        std::env::var("MEMRA_VALLEY_S").ok().and_then(|v| v.parse().ok()).unwrap_or(2.0)
    })
}

/// The valley signal — a read-only view over worker truth (health phase + beat age +
/// the pending-admit gauge). Cheap enough to evaluate per /metrics call AND per runner
/// poll: two atomic loads and a subtraction.
#[derive(Clone)]
pub struct ValleySignal {
    health: SharedHealth,
}

impl ValleySignal {
    pub fn new(health: SharedHealth) -> Self {
        Self { health }
    }

    /// Seconds the worker has been completely idle; 0.0 the instant there is ANY work
    /// (active/queued sessions => phase != IDLE; submitted-not-yet-popped requests =>
    /// PENDING_ADMITS > 0; loading/dead phases are not idleness either).
    pub fn idle_seconds(&self) -> f64 {
        let s = self.health.snapshot();
        if s.phase == PHASE_IDLE && PENDING_ADMITS.load(Ordering::Acquire) == 0 {
            s.beat_age_ms as f64 / 1000.0
        } else {
            0.0
        }
    }

    /// The resume-side signal: a full threshold of quiet.
    pub fn in_valley(&self) -> bool {
        self.idle_seconds() >= valley_threshold_s()
    }

    /// The yield-side signal: ANY activity, no debounce. Deliberately not `!in_valley()` —
    /// the asymmetry (instant yield, debounced resume) is the whole point.
    pub fn busy_now(&self) -> bool {
        self.idle_seconds() == 0.0
    }
}

// ---------------------------------------------------------------------------
// Background job runner
// ---------------------------------------------------------------------------

/// Job lifecycle states, published at /metrics ("bg" block). u8-backed so the /metrics
/// read is a bare atomic load.
pub const BG_WAITING: u8 = 0; // configured, waiting for the first/next valley to launch
pub const BG_RUNNING: u8 = 1; // child running in a valley
pub const BG_YIELDED: u8 = 2; // SIGSTOPped for serve traffic (stop mode)
pub const BG_PREEMPTED: u8 = 3; // checkpointed out (ckpt mode), relaunches next valley
pub const BG_DONE: u8 = 4; // exit 0 — complete, never relaunched
pub const BG_FAILED: u8 = 5; // non-0/75 exit — never relaunched, loud
pub const BG_REFUSED: u8 = 6; // VRAM budget did not fit at launch (retries next valley)

pub fn bg_state_str(s: u8) -> &'static str {
    match s {
        BG_WAITING => "waiting_valley",
        BG_RUNNING => "running",
        BG_YIELDED => "yielded",
        BG_PREEMPTED => "preempted",
        BG_DONE => "done",
        BG_FAILED => "failed",
        BG_REFUSED => "refused_vram",
        _ => "unknown",
    }
}

/// Shared observable state — the /metrics "bg" block reads ONLY this (atomics, never the
/// child handle). Counters are cumulative for the process life.
#[derive(Default)]
pub struct BgJobState {
    pub state: AtomicU8,
    pub launches: AtomicU64,
    pub yields: AtomicU64,
    pub resumes: AtomicU64,
    /// checkpoint-mode preemptions (SIGUSR1 sent); `ckpt_kills` counts the subset that
    /// blew the grace window and were SIGKILLed (dirty preempt — at-least-once resume).
    pub preempts: AtomicU64,
    pub ckpt_kills: AtomicU64,
    /// wall micros from busy-edge observation to the yield signal having been SENT (stop
    /// mode: SIGSTOP returned; ckpt mode: SIGUSR1 returned). The detection half of the
    /// yield bound; the full bound adds one poll interval. Last observed value.
    pub last_yield_signal_us: AtomicU64,
    pub job_pid: AtomicU32, // 0 = no live child
    pub vram_budget_mb: AtomicU64,
}

impl BgJobState {
    pub fn to_json(&self, yield_mode: &str) -> serde_json::Value {
        let pid = self.job_pid.load(Ordering::Acquire);
        serde_json::json!({
            "state": bg_state_str(self.state.load(Ordering::Acquire)),
            "yield_mode": yield_mode,
            "launches": self.launches.load(Ordering::Relaxed),
            "yields": self.yields.load(Ordering::Relaxed),
            "resumes": self.resumes.load(Ordering::Relaxed),
            "preempts": self.preempts.load(Ordering::Relaxed),
            "ckpt_kills": self.ckpt_kills.load(Ordering::Relaxed),
            "last_yield_signal_us": self.last_yield_signal_us.load(Ordering::Relaxed),
            "job_pid": if pid == 0 { serde_json::Value::Null } else { pid.into() },
            "vram_budget_mb": self.vram_budget_mb.load(Ordering::Relaxed),
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum YieldMode {
    /// SIGSTOP/SIGCONT the job's process group. Memory (VRAM included) stays resident
    /// while stopped — budget accordingly.
    Stop,
    /// SIGUSR1 = "checkpoint and exit 75"; relaunch next valley. For jobs whose stopped
    /// working set must not squat on VRAM.
    Checkpoint,
}

impl YieldMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            YieldMode::Stop => "stop",
            YieldMode::Checkpoint => "checkpoint",
        }
    }
}

pub struct BgConfig {
    /// The job command, run via `sh -c` in its own process group.
    pub cmd: String,
    pub poll_ms: u64,
    pub yield_mode: YieldMode,
    pub ckpt_grace_ms: u64,
    /// VRAM the job may use (0 = CPU-only, no GPU probe at launch).
    pub vram_budget_mb: u64,
    /// Serve headroom (MB) that must REMAIN free above the job's budget at launch —
    /// the MEMRA_MOE_RESIDENT_HEADROOM_GB composition.
    pub headroom_mb: u64,
}

impl BgConfig {
    /// None when MEMRA_BG_JOB is unset — the runner (and its /metrics block) simply does
    /// not exist; zero cost on every deployment that doesn't ask for it.
    pub fn from_env() -> Option<Self> {
        let cmd = std::env::var("MEMRA_BG_JOB").ok().filter(|c| !c.trim().is_empty())?;
        let u = |k: &str, d: u64| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        let mode = match std::env::var("MEMRA_BG_YIELD_MODE").as_deref() {
            Ok("checkpoint") => YieldMode::Checkpoint,
            Ok("stop") | Err(_) => YieldMode::Stop,
            Ok(other) => {
                eprintln!("[darklane] WARN: bad MEMRA_BG_YIELD_MODE {other:?} \
                           (stop|checkpoint); using stop");
                YieldMode::Stop
            }
        };
        let headroom_gb: f64 = std::env::var("MEMRA_MOE_RESIDENT_HEADROOM_GB")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(2.0);
        Some(Self {
            cmd,
            poll_ms: u("MEMRA_BG_POLL_MS", 25).max(1),
            yield_mode: mode,
            ckpt_grace_ms: u("MEMRA_BG_CKPT_GRACE_MS", 5000),
            vram_budget_mb: u("MEMRA_BG_VRAM_MB", 0),
            headroom_mb: (headroom_gb * 1024.0) as u64,
        })
    }
}

/// Min free VRAM (MB) across visible GPUs, via nvidia-smi. Min, not sum: v1 does not know
/// which card the job lands on, and on a PP-2 pair BOTH cards carry serve shards — the
/// budget must fit the tightest one. None (no tool / parse failure) = fail closed.
fn nvidia_min_free_mb() -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        .output().ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(|l| l.trim().parse::<u64>().ok())
        .collect::<Option<Vec<u64>>>()?
        .into_iter()
        .min()
}

/// Handle main() keeps: flip `stop`, then `join()` — the loop polls the flag every
/// `poll_ms` and exits after cleaning up the child (see shutdown() below).
pub struct BgHandle {
    pub state: Arc<BgJobState>,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl BgHandle {
    /// Called after axum::serve returns (drain complete). MUST run before process exit in
    /// the graceful path: a SIGSTOPped orphan would stay frozen forever. The ungraceful
    /// path (SIGKILL on the server) is covered by PR_SET_PDEATHSIG=SIGKILL on the child.
    pub fn shutdown(mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// Signal a whole process group. pgid == child pid (process_group(0) at spawn).
fn kill_group(pgid: u32, sig: i32) {
    unsafe {
        libc::kill(-(pgid as i32), sig);
    }
}

/// Spawn the supervisor thread. `in_valley` / `busy_now` are injected (prod: ValleySignal;
/// tests: atomics) so the whole state machine is pinned GPU-free. `vram_probe` likewise
/// (prod: nvidia-smi; tests: a constant).
pub fn spawn_runner(
    cfg: BgConfig,
    in_valley: Arc<dyn Fn() -> bool + Send + Sync>,
    busy_now: Arc<dyn Fn() -> bool + Send + Sync>,
    vram_probe: Arc<dyn Fn() -> Option<u64> + Send + Sync>,
) -> BgHandle {
    let state = Arc::new(BgJobState::default());
    state.vram_budget_mb.store(cfg.vram_budget_mb, Ordering::Relaxed);
    let stop = Arc::new(AtomicBool::new(false));
    let (st, sp) = (state.clone(), stop.clone());
    eprintln!("[darklane] bg runner armed: cmd={:?} mode={} poll={}ms vram_budget={}MB \
               (+{}MB serve headroom must stay free)",
              cfg.cmd, cfg.yield_mode.as_str(), cfg.poll_ms, cfg.vram_budget_mb,
              cfg.headroom_mb);
    let thread = std::thread::Builder::new()
        .name("memra-bg-runner".into())
        .spawn(move || runner_loop(cfg, in_valley, busy_now, vram_probe, st, sp))
        .expect("spawn bg runner thread");
    BgHandle { state, stop, thread: Some(thread) }
}

/// Prod wiring: valley signal from worker health.
pub fn spawn_from_env(health: SharedHealth) -> Option<BgHandle> {
    let cfg = BgConfig::from_env()?;
    let v = ValleySignal::new(health);
    let v2 = v.clone();
    Some(spawn_runner(
        cfg,
        Arc::new(move || v.in_valley()),
        Arc::new(move || v2.busy_now()),
        Arc::new(nvidia_min_free_mb),
    ))
}

fn runner_loop(
    cfg: BgConfig,
    in_valley: Arc<dyn Fn() -> bool + Send + Sync>,
    busy_now: Arc<dyn Fn() -> bool + Send + Sync>,
    vram_probe: Arc<dyn Fn() -> Option<u64> + Send + Sync>,
    st: Arc<BgJobState>,
    stop: Arc<AtomicBool>,
) {
    let mut child: Option<std::process::Child> = None;
    let poll = std::time::Duration::from_millis(cfg.poll_ms);
    // one loud line per refusal EPISODE, not per poll (a tight card would log 40/s).
    let mut refusal_logged = false;

    loop {
        if stop.load(Ordering::Acquire) {
            break;
        }
        let s = st.state.load(Ordering::Acquire);
        match s {
            BG_WAITING | BG_PREEMPTED | BG_REFUSED => {
                if in_valley() {
                    // GPU memory discipline: fits-or-refused, re-checked every launch
                    // (headroom moves as sessions park/retire — a refusal is not forever).
                    if cfg.vram_budget_mb > 0 {
                        let free = vram_probe();
                        let need = cfg.vram_budget_mb + cfg.headroom_mb;
                        let fits = free.is_some_and(|f| f >= need);
                        if !fits {
                            if !refusal_logged {
                                eprintln!("[darklane] REFUSED: vram budget {}MB + serve \
                                           headroom {}MB > min free {:?}MB (fail-closed \
                                           when unreadable); will retry next valley",
                                          cfg.vram_budget_mb, cfg.headroom_mb, free);
                                refusal_logged = true;
                            }
                            st.state.store(BG_REFUSED, Ordering::Release);
                            std::thread::sleep(poll);
                            continue;
                        }
                    }
                    refusal_logged = false;
                    match launch(&cfg.cmd, cfg.vram_budget_mb) {
                        Ok(c) => {
                            st.job_pid.store(c.id(), Ordering::Release);
                            st.launches.fetch_add(1, Ordering::Relaxed);
                            st.state.store(BG_RUNNING, Ordering::Release);
                            eprintln!("[darklane] job launched (pid {}, {})",
                                      c.id(),
                                      if s == BG_PREEMPTED { "resume from checkpoint" }
                                      else { "fresh" });
                            child = Some(c);
                        }
                        Err(err) => {
                            eprintln!("[darklane] job spawn FAILED: {err}");
                            st.state.store(BG_FAILED, Ordering::Release);
                        }
                    }
                }
            }
            BG_RUNNING => {
                let c = child.as_mut().expect("running state implies child");
                // exit first: a finished job must not be signaled.
                match c.try_wait() {
                    Ok(Some(status)) => {
                        handle_exit(status, &st, /*preempting=*/ false);
                        st.job_pid.store(0, Ordering::Release);
                        child = None;
                    }
                    Ok(None) => {
                        if busy_now() {
                            let t0 = std::time::Instant::now();
                            let pgid = c.id();
                            match cfg.yield_mode {
                                YieldMode::Stop => {
                                    kill_group(pgid, libc::SIGSTOP);
                                    st.last_yield_signal_us.store(
                                        t0.elapsed().as_micros() as u64, Ordering::Relaxed);
                                    st.yields.fetch_add(1, Ordering::Relaxed);
                                    st.state.store(BG_YIELDED, Ordering::Release);
                                }
                                YieldMode::Checkpoint => {
                                    kill_group(pgid, libc::SIGUSR1);
                                    st.last_yield_signal_us.store(
                                        t0.elapsed().as_micros() as u64, Ordering::Relaxed);
                                    st.preempts.fetch_add(1, Ordering::Relaxed);
                                    preempt_wait(c, &st, cfg.ckpt_grace_ms);
                                    st.job_pid.store(0, Ordering::Release);
                                    child = None;
                                }
                            }
                        }
                    }
                    Err(err) => {
                        eprintln!("[darklane] try_wait failed: {err}; treating job as failed");
                        st.state.store(BG_FAILED, Ordering::Release);
                        st.job_pid.store(0, Ordering::Release);
                        child = None;
                    }
                }
            }
            BG_YIELDED => {
                // a STOPPED process cannot exit — no try_wait needed until resumed.
                if in_valley() {
                    let c = child.as_ref().expect("yielded state implies child");
                    kill_group(c.id(), libc::SIGCONT);
                    st.resumes.fetch_add(1, Ordering::Relaxed);
                    st.state.store(BG_RUNNING, Ordering::Release);
                }
            }
            _ => break, // DONE / FAILED — terminal; the thread's work is over.
        }
        std::thread::sleep(poll);
    }

    // Shutdown cleanup: never leave a stopped orphan. CONT first (a stopped process
    // cannot act on TERM), then TERM (checkpoint-class jobs get their handler), brief
    // grace, then KILL the group.
    if let Some(mut c) = child.take() {
        let pgid = c.id();
        kill_group(pgid, libc::SIGCONT);
        kill_group(pgid, libc::SIGTERM);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        loop {
            match c.try_wait() {
                Ok(Some(status)) => {
                    eprintln!("[darklane] job terminated at shutdown ({status})");
                    break;
                }
                Ok(None) if std::time::Instant::now() < deadline => {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                _ => {
                    kill_group(pgid, libc::SIGKILL);
                    let _ = c.wait();
                    eprintln!("[darklane] job SIGKILLed at shutdown (grace expired)");
                    break;
                }
            }
        }
        st.job_pid.store(0, Ordering::Release);
    }
}

/// Spawn the job: own process group (signals hit the whole tree, never the server) +
/// PDEATHSIG=SIGKILL (a SIGKILLed server cannot run cleanup; the kernel reaps the job —
/// KILL not TERM because a job its runner had STOPPED would never act on TERM).
/// The budget is exported so the job can size itself (`MEMRA_BG_VRAM_MB`).
fn launch(cmd: &str, vram_budget_mb: u64) -> std::io::Result<std::process::Child> {
    use std::os::unix::process::CommandExt;
    let mut c = std::process::Command::new("sh");
    c.arg("-c").arg(cmd)
        .env("MEMRA_BG_VRAM_MB", vram_budget_mb.to_string())
        .process_group(0);
    unsafe {
        c.pre_exec(|| {
            libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL);
            Ok(())
        });
    }
    c.spawn()
}

/// Classify a job exit. Exit 75 (EX_TEMPFAIL) = "checkpointed, resume me" — the job may
/// use it voluntarily too (self-preempting at a step boundary), not only under SIGUSR1.
fn handle_exit(status: std::process::ExitStatus, st: &BgJobState, preempting: bool) {
    match status.code() {
        Some(0) => {
            eprintln!("[darklane] job COMPLETE (exit 0)");
            st.state.store(BG_DONE, Ordering::Release);
        }
        Some(75) => {
            eprintln!("[darklane] job checkpointed (exit 75) — relaunches next valley");
            st.state.store(BG_PREEMPTED, Ordering::Release);
        }
        code => {
            if preempting {
                // died under SIGUSR1 without the protocol exit — treat the on-disk
                // checkpoint as authoritative and relaunch (at-least-once).
                eprintln!("[darklane] job exited {code:?} during preemption — \
                           relaunching from last checkpoint next valley");
                st.state.store(BG_PREEMPTED, Ordering::Release);
            } else {
                eprintln!("[darklane] job FAILED ({status}) — not relaunching");
                st.state.store(BG_FAILED, Ordering::Release);
            }
        }
    }
}

/// Checkpoint-mode preemption wait: the job has `grace_ms` after SIGUSR1 to write its
/// checkpoint and exit; past that it is SIGKILLed (the yield bound holds even against a
/// wedged job — VRAM is freed by process death either way).
fn preempt_wait(c: &mut std::process::Child, st: &BgJobState, grace_ms: u64) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(grace_ms);
    loop {
        match c.try_wait() {
            Ok(Some(status)) => {
                handle_exit(status, st, /*preempting=*/ true);
                return;
            }
            Ok(None) if std::time::Instant::now() < deadline => {
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            _ => {
                kill_group(c.id(), libc::SIGKILL);
                let _ = c.wait();
                st.ckpt_kills.fetch_add(1, Ordering::Relaxed);
                eprintln!("[darklane] job blew the {grace_ms}ms checkpoint grace — \
                           SIGKILLed (dirty preempt; resumes from last on-disk state)");
                st.state.store(BG_PREEMPTED, Ordering::Release);
                return;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests — the whole state machine, GPU-free (injected valley/busy/vram signals)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::health::WorkerHealth;

    /// /proc/<pid>/stat field 3 — 'T' is stopped, 'R'/'S' running/sleeping, gone = None.
    fn proc_state(pid: u32) -> Option<char> {
        let s = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
        // field 2 (comm) may contain spaces/parens — state is the char after the LAST ')'.
        s.rfind(')').and_then(|i| s[i + 1..].trim_start().chars().next())
    }

    fn wait_for<F: Fn() -> bool>(what: &str, ms: u64, f: F) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(ms);
        while std::time::Instant::now() < deadline {
            if f() {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        panic!("timed out ({ms}ms) waiting for: {what}");
    }

    struct Sig {
        valley: Arc<AtomicBool>,
        busy: Arc<AtomicBool>,
    }
    fn sigs() -> (Sig, Arc<dyn Fn() -> bool + Send + Sync>, Arc<dyn Fn() -> bool + Send + Sync>)
    {
        let valley = Arc::new(AtomicBool::new(false));
        let busy = Arc::new(AtomicBool::new(false));
        let (v, b) = (valley.clone(), busy.clone());
        (
            Sig { valley, busy },
            Arc::new(move || v.load(Ordering::Acquire)),
            Arc::new(move || b.load(Ordering::Acquire)),
        )
    }

    fn cfg(cmd: &str, mode: YieldMode) -> BgConfig {
        BgConfig {
            cmd: cmd.into(),
            poll_ms: 5,
            yield_mode: mode,
            ckpt_grace_ms: 2000,
            vram_budget_mb: 0,
            headroom_mb: 2048,
        }
    }

    #[test]
    fn valley_signal_reads_worker_truth() {
        let h = WorkerHealth::new();
        let v = ValleySignal::new(h.clone());
        // LOADING is not idleness.
        assert_eq!(v.idle_seconds(), 0.0);
        assert!(v.busy_now());
        // IDLE: age accrues from the phase stamp. Retry-tolerant: PENDING_ADMITS is
        // process-global and handler tests running in parallel bump it transiently.
        h.set_phase(PHASE_IDLE);
        std::thread::sleep(std::time::Duration::from_millis(30));
        // (>= 0.02 implies !busy_now at that instant — no separate racy assert.)
        wait_for("idle age accrues", 2000, || v.idle_seconds() >= 0.02);
        // a pending admit is traffic even while the phase is still IDLE (the handoff gap).
        PENDING_ADMITS.fetch_add(1, Ordering::Release);
        assert_eq!(v.idle_seconds(), 0.0);
        assert!(v.busy_now());
        PENDING_ADMITS.fetch_sub(1, Ordering::Release);
        // BUSY: zero again.
        h.beat_busy();
        assert_eq!(v.idle_seconds(), 0.0);
    }

    #[test]
    fn stop_mode_full_cycle_launch_yield_resume_shutdown() {
        let (sig, v, b) = sigs();
        let h = spawn_runner(cfg("while :; do sleep 0.01; done", YieldMode::Stop),
                             v, b, Arc::new(|| None));
        let st = h.state.clone();
        // no valley -> no launch.
        std::thread::sleep(std::time::Duration::from_millis(30));
        assert_eq!(st.state.load(Ordering::Acquire), BG_WAITING);
        // valley -> launch.
        sig.valley.store(true, Ordering::Release);
        wait_for("launch", 1000, || st.state.load(Ordering::Acquire) == BG_RUNNING);
        let pid = st.job_pid.load(Ordering::Acquire);
        assert!(pid > 0);
        wait_for("job running", 1000, || matches!(proc_state(pid), Some('R' | 'S')));
        // busy edge -> SIGSTOP within the bound (poll 5ms; assert well under 500ms).
        sig.valley.store(false, Ordering::Release);
        sig.busy.store(true, Ordering::Release);
        let t0 = std::time::Instant::now();
        wait_for("yield to T", 500, || proc_state(pid) == Some('T'));
        assert!(t0.elapsed().as_millis() < 500);
        assert_eq!(st.state.load(Ordering::Acquire), BG_YIELDED);
        assert_eq!(st.yields.load(Ordering::Relaxed), 1);
        // back to valley -> SIGCONT.
        sig.busy.store(false, Ordering::Release);
        sig.valley.store(true, Ordering::Release);
        wait_for("resume", 1000, || matches!(proc_state(pid), Some('R' | 'S')));
        assert_eq!(st.resumes.load(Ordering::Relaxed), 1);
        // shutdown never leaves an orphan (stopped or otherwise).
        h.shutdown();
        wait_for("job reaped", 3000, || proc_state(pid).is_none_or(|s| s == 'Z'));
    }

    #[test]
    fn checkpoint_mode_preempts_and_resumes_counter() {
        // The toy proof in miniature: a counter that checkpoints to disk on SIGUSR1 and
        // exits 75; a relaunch resumes FROM the file. This is the deliverable-4 seam.
        let dir = std::env::temp_dir().join(format!("darklane-ckpt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("counter");
        let job = format!(
            "n=$(cat {f} 2>/dev/null || echo 0); trap 'echo $n > {f}; exit 75' USR1; \
             while :; do n=$((n+1)); echo $n > {f}.live; sleep 0.005; done",
            f = f.display());
        let (sig, v, b) = sigs();
        let h = spawn_runner(cfg(&job, YieldMode::Checkpoint), v, b, Arc::new(|| None));
        let st = h.state.clone();
        sig.valley.store(true, Ordering::Release);
        wait_for("launch", 1000, || st.state.load(Ordering::Acquire) == BG_RUNNING);
        // let it make progress, then preempt.
        wait_for("progress", 2000, || {
            std::fs::read_to_string(f.with_extension("live"))
                .is_ok_and(|s| s.trim().parse::<u64>().unwrap_or(0) >= 3)
        });
        sig.valley.store(false, Ordering::Release);
        sig.busy.store(true, Ordering::Release);
        wait_for("preempted", 3000, || st.state.load(Ordering::Acquire) == BG_PREEMPTED);
        let ck1: u64 = std::fs::read_to_string(&f).unwrap().trim().parse().unwrap();
        assert!(ck1 >= 3, "checkpoint wrote the counter (got {ck1})");
        assert_eq!(st.preempts.load(Ordering::Relaxed), 1);
        assert_eq!(st.ckpt_kills.load(Ordering::Relaxed), 0, "clean ckpt, no kill");
        // next valley -> relaunch resumes FROM the checkpoint (not from zero).
        sig.busy.store(false, Ordering::Release);
        sig.valley.store(true, Ordering::Release);
        wait_for("relaunch", 1000, || st.launches.load(Ordering::Relaxed) == 2);
        wait_for("resumed past ck1", 2000, || {
            std::fs::read_to_string(f.with_extension("live"))
                .is_ok_and(|s| s.trim().parse::<u64>().unwrap_or(0) > ck1)
        });
        h.shutdown();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn job_completion_and_failure_are_terminal() {
        // exit 0 = done, never relaunched.
        let (sig, v, b) = sigs();
        let h = spawn_runner(cfg("true", YieldMode::Stop), v, b, Arc::new(|| None));
        sig.valley.store(true, Ordering::Release);
        let st = h.state.clone();
        wait_for("done", 2000, || st.state.load(Ordering::Acquire) == BG_DONE);
        std::thread::sleep(std::time::Duration::from_millis(30));
        assert_eq!(st.launches.load(Ordering::Relaxed), 1, "done is terminal");
        h.shutdown();
        // nonzero exit = failed, never relaunched.
        let (sig, v, b) = sigs();
        let h = spawn_runner(cfg("exit 3", YieldMode::Stop), v, b, Arc::new(|| None));
        sig.valley.store(true, Ordering::Release);
        let st = h.state.clone();
        wait_for("failed", 2000, || st.state.load(Ordering::Acquire) == BG_FAILED);
        std::thread::sleep(std::time::Duration::from_millis(30));
        assert_eq!(st.launches.load(Ordering::Relaxed), 1, "failed is terminal");
        h.shutdown();
    }

    #[test]
    fn vram_budget_refused_when_it_does_not_fit_and_fail_closed() {
        // budget 1000MB + headroom 2048MB > free 2500MB -> refused.
        let (sig, v, b) = sigs();
        let mut c = cfg("true", YieldMode::Stop);
        c.vram_budget_mb = 1000;
        let h = spawn_runner(c, v, b, Arc::new(|| Some(2500)));
        sig.valley.store(true, Ordering::Release);
        let st = h.state.clone();
        wait_for("refused", 1000, || st.state.load(Ordering::Acquire) == BG_REFUSED);
        assert_eq!(st.launches.load(Ordering::Relaxed), 0);
        h.shutdown();
        // unreadable probe -> fail closed (still refused).
        let (sig, v, b) = sigs();
        let mut c = cfg("true", YieldMode::Stop);
        c.vram_budget_mb = 1;
        let h = spawn_runner(c, v, b, Arc::new(|| None));
        sig.valley.store(true, Ordering::Release);
        let st = h.state.clone();
        wait_for("refused (fail closed)", 1000,
                 || st.state.load(Ordering::Acquire) == BG_REFUSED);
        h.shutdown();
        // fits -> launches (and completes).
        let (sig, v, b) = sigs();
        let mut c = cfg("true", YieldMode::Stop);
        c.vram_budget_mb = 1000;
        let h = spawn_runner(c, v, b, Arc::new(|| Some(4000)));
        sig.valley.store(true, Ordering::Release);
        let st = h.state.clone();
        wait_for("launched when fits", 2000, || st.state.load(Ordering::Acquire) == BG_DONE);
        h.shutdown();
    }
}
