//! Lane model — the yield-gate invariants, native in the scheduler.
//! (ARCHITECTURE-H100.md §3 B3; thresholds mirror the out-of-process sidecar gate so the
//! external sidecar remains a compatible outer layer.)
//!
//! Three lanes:
//!   interactive — the protected class; always admitted; its decode step latency IS the
//!                 SLO sensor (true engine timings, not the sidecar's network-gap proxy).
//!   judge       — prefill-shaped read work; admitted while interactive p99 < 100% of SLO.
//!   harvest     — decode-shaped batch generation; sheds first, at 90% of SLO.
//!
//! Invariants (the B2/D1 lessons):
//!   - shed happens at ADMISSION, never by queueing inside the engine ("the engine's own
//!     queue is where the tail dies") — a shed request gets an immediate retryable error;
//!   - per-lane PREFILL BUDGETS per tick replace vLLM's single global chunked-prefill knob
//!     (the knob that taxed baseline p99 11.6 -> 40.6 ms at zero parasite load);
//!   - interactive is never preempted; harvest yields first.

use std::collections::VecDeque;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lane {
    Interactive,
    Judge,
    Harvest,
}

impl Lane {
    /// Parse an `x-lane` header value. Unknown => None (handler answers 400, like the sidecar).
    pub fn parse(v: &str) -> Option<Lane> {
        match v {
            "interactive" => Some(Lane::Interactive),
            "judge" => Some(Lane::Judge),
            "harvest" => Some(Lane::Harvest),
            _ => None,
        }
    }
    pub fn as_str(&self) -> &'static str {
        match self {
            Lane::Interactive => "interactive",
            Lane::Judge => "judge",
            Lane::Harvest => "harvest",
        }
    }
    pub const ALL: [Lane; 3] = [Lane::Interactive, Lane::Judge, Lane::Harvest];
    pub fn idx(&self) -> usize {
        match self {
            Lane::Interactive => 0,
            Lane::Judge => 1,
            Lane::Harvest => 2,
        }
    }
}

/// Windowed percentile over interactive decode-step latencies (ms). Engine ground truth:
/// the worker records the wall time of each batched decode tick that contained at least
/// one interactive session — that IS the interactive TPOT for that tick.
pub struct StepStats {
    window: VecDeque<(Instant, f32)>,
    window_s: f32,
}

impl StepStats {
    pub fn new(window_s: f32) -> Self {
        Self { window: VecDeque::with_capacity(4096), window_s }
    }
    pub fn record(&mut self, ms: f32) {
        self.window.push_back((Instant::now(), ms));
        if self.window.len() > 16384 {
            self.window.pop_front();
        }
    }
    fn evict(&mut self) {
        let cutoff = self.window_s;
        while let Some(&(t, _)) = self.window.front() {
            if t.elapsed().as_secs_f32() > cutoff {
                self.window.pop_front();
            } else {
                break;
            }
        }
    }
    /// q in [0,100]. None until the window has signal (cold start => lanes admit).
    pub fn p(&mut self, q: f32) -> Option<f32> {
        self.evict();
        if self.window.is_empty() {
            return None;
        }
        let mut v: Vec<f32> = self.window.iter().map(|&(_, g)| g).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let i = ((q / 100.0) * (v.len() - 1) as f32).round() as usize;
        Some(v[i.min(v.len() - 1)])
    }
}

/// Admission policy: thresholds as fractions of the SLO, per lane (yieldgate.py SHED_AT).
pub struct LanePolicy {
    pub slo_p99_ms: f32,
    /// admission threshold per lane index (interactive unused — always admitted)
    pub shed_at: [f32; 3],
    /// per-tick prefill token budgets [interactive, judge, harvest]; interactive budget is
    /// the tick chunk size (uncapped policy-wise), judge/harvest are the dark-lane budgets.
    pub prefill_budget: [usize; 3],
    /// max resident sessions per lane
    pub max_sessions: [usize; 3],
}

impl LanePolicy {
    pub fn from_env() -> Self {
        let f = |k: &str, d: f32| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        let u = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        Self {
            slo_p99_ms: f("MEMRA_SLO_P99_MS", 50.0),
            shed_at: [f32::INFINITY, f("MEMRA_SHED_JUDGE", 1.00), f("MEMRA_SHED_HARVEST", 0.90)],
            // Dark-lane budgets are PER-TICK STALL BOUNDS, not throughput knobs: a 2048-tok
            // judge chunk = ~230 ms of decode starvation per tick (measured 2026-07-26,
            // native-judge battery: p99 17.6 -> 282 ms at 1 req/s). 256 tok ≈ 30 ms bound.
            prefill_budget: [
                u("MEMRA_PREFILL_TICK", 1024),
                u("MEMRA_PREFILL_JUDGE", 256),
                u("MEMRA_PREFILL_HARVEST", 256),
            ],
            // Interactive cap sizes CONCURRENCY, not batching (decode runs multiple <=8-row
            // chunks per tick). 2026-07-26 battery: cap 4 vs 32 closed-loop clients made
            // QUEUE WAIT the interactive bottleneck (client streams starved in FIFO while
            // the engine idled at p99 17.8ms) — the cap must admit the whole protected set;
            // aggregate decode throughput is the real limit, enforced by the SLO estimator.
            max_sessions: [
                u("MEMRA_LANE_MAX_INTERACTIVE", 32),
                u("MEMRA_LANE_MAX_JUDGE", 4),
                u("MEMRA_LANE_MAX_HARVEST", 8),
            ],
        }
    }

    /// The yield gate: interactive always; judge/harvest against the measured interactive
    /// p99 vs their SLO fraction. No signal yet (cold estimator) => admit, like the sidecar.
    ///
    /// `starved` closes the estimator's blind spot (2026-07-26 native-judge battery, rates
    /// 4-8: interactive decoded ZERO tokens, so no p99 samples arrived, the window aged out,
    /// and the cold-start rule kept admitting judges into total starvation). The worker sets
    /// it when interactive work EXISTS but no interactive decode tick has run within the SLO
    /// age — starvation IS an SLO breach even though the estimator can't see it.
    pub fn admit(&self, lane: Lane, stats: &mut StepStats, starved: bool) -> bool {
        if lane == Lane::Interactive {
            return true;
        }
        if starved {
            return false;
        }
        match stats.p(99.0) {
            Some(p99) => p99 < self.slo_p99_ms * self.shed_at[lane.idx()],
            None => true,
        }
    }
}

/// Per-lane counters + latency snapshot, exported at /yield/metrics (sidecar-compatible shape).
#[derive(Default, Clone, serde::Serialize)]
pub struct LaneMetrics {
    pub admitted: [u64; 3],
    pub shed: [u64; 3],
    pub completed: [u64; 3],
    pub tokens_out: [u64; 3],
    pub step_p50_ms: f32,
    pub step_p99_ms: f32,
    pub batch_size_last: usize,
}

pub type SharedMetrics = std::sync::Arc<std::sync::Mutex<LaneMetrics>>;
