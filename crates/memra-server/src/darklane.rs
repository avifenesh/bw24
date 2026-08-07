//! Dead-darklane valley detection (lane/darklane-training, 2026-08-07).
//!
//! THE THESIS (owner, standing): idle serve capacity carries owner research/training jobs,
//! yielding instantly to paying traffic. This module is the ENGINE half only: when the box
//! is idle, and (next) what runs in the valley and how it gets out of the way. Scheduling
//! policy and economics belong to the product repo.
//!
//! VALLEY DETECTION reuses worker truth that already exists instead of inventing a new
//! sensor: the scheduler flips `health` to `PHASE_IDLE` exactly when `active.is_empty() &&
//! queue.is_empty()` (worker.rs loop top) and `set_phase` stamps the beat on entry — so
//! `phase == IDLE` + `beat_age_ms` IS the idle duration, to the millisecond, with zero new
//! hot-path cost. `PENDING_ADMITS` closes the HTTP→worker handoff gap (a request the handler
//! has submitted but the worker hasn't popped yet is traffic, not idleness).
//!
//! Exposed two ways: `/metrics.serve_idle_seconds` (always published — the idle sensor is
//! useful whether or not anything consumes it) and the `ValleySignal` hook (the in-process
//! consumer seam for the background-job runner).

use std::sync::atomic::Ordering;

use crate::health::{PHASE_IDLE, SharedHealth};
use crate::worker::PENDING_ADMITS;

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
    /// the asymmetry (instant yield, debounced resume) is the point: paying traffic never
    /// waits for a threshold, background work does.
    pub fn busy_now(&self) -> bool {
        self.idle_seconds() == 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::health::WorkerHealth;

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

    #[test]
    fn valley_signal_reads_worker_truth() {
        let h = WorkerHealth::new();
        let v = ValleySignal::new(h.clone());
        // LOADING is not idleness.
        assert_eq!(v.idle_seconds(), 0.0);
        assert!(v.busy_now());
        // IDLE: age accrues from the phase stamp. Retry-tolerant: PENDING_ADMITS is
        // process-global and handler tests running in parallel bump it transiently.
        // (>= 0.02 implies !busy_now at that instant — no separate racy assert.)
        h.set_phase(PHASE_IDLE);
        std::thread::sleep(std::time::Duration::from_millis(30));
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
}
