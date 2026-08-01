//! M1 pipeline-parallel 2-stage seam (increment 1: seam + gate, single device).
//!
//! Door: `MEMRA_PP_STAGES=2` (default OFF — unset/0/1 = no behavior change anywhere).
//! Split: `MEMRA_PP_SPLIT=<i>` puts layers [0, i) in stage 0 and [i, n) in stage 1;
//! default i = n_layers/2.
//!
//! What the seam is: the plain eager decode step (`decode_step` -> `decode_step_h`,
//! and the gemma4 arm `gemma4_decode_step_h`) runs its per-layer walk as TWO stage
//! subgraphs with an explicit activation handoff between them — a real device-buffer
//! copy of the residual hidden state (TX into a dedicated boundary buffer, RX out of
//! it), never an alias, so the seam survives the increment-2 move to a real P2P
//! transport between devices. Cross-layer launch fusions (the generic loop's pending
//! (x1, ffn_out) add-carry; gemma4's pre-quantized next-norm carry) are LOCAL to a
//! stage: the boundary materializes the residual with the same kernels the last layer
//! of an unsplit walk uses, which is the kernel-check-pinned identity
//! (`add_rms_norm_q8_1 == add then rms_norm_q8_1`; `add_scale_rms_norm_q8_1 ==
//! add_scale then rms_norm_q8_1`), so split logits are BIT-IDENTICAL to unsplit —
//! `pp2-gate` proves it per step.
//!
//! Ownership across the seam (the M1 contract increment 2 inherits):
//!   - hidden state [n_embd] f32 is the ONLY tensor that crosses the boundary;
//!   - KV/linear-attn cache entries are per-layer, so stage s exclusively owns
//!     `cache` state for its own layer range — nothing else is shared;
//!   - position/rope state is the scalar `cache.pos` snapshot taken once per step
//!     (each stage reads the same value; a per-stage counter replicates it later);
//!   - the embed table lives with stage 0, output_norm + lm head with the last stage.
//!
//! Increment-1 scope: plain eager decode only. NOT wired: batch/dc/graph/spec loops
//! and the gemma4-E4B eager arm (`warn_unwired_once` fires if the door is set there).

use std::sync::atomic::{AtomicBool, Ordering};

/// Returns `Some(split)` iff the pp2 door is open: `MEMRA_PP_STAGES=2` with a valid
/// split in [1, n_layers-1]. Reads the environment on every call (gates toggle the
/// door in-process); the cost is two getenv per decode step, eager-loop noise.
pub fn pp2_split(n_layers: usize) -> Option<usize> {
    match std::env::var("MEMRA_PP_STAGES") {
        Ok(v) if v == "2" => {}
        Ok(v) if v.is_empty() || v == "0" || v == "1" => return None,
        Ok(v) => {
            warn_bad_once(&format!(
                "MEMRA_PP_STAGES={v} unsupported (increment 1 is 2-stage only); door stays OFF"
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

static WARNED_BAD: AtomicBool = AtomicBool::new(false);
fn warn_bad_once(msg: &str) {
    if !WARNED_BAD.swap(true, Ordering::Relaxed) {
        eprintln!("[pp2] {msg}");
    }
}

static WARNED_UNWIRED: AtomicBool = AtomicBool::new(false);
/// One-time notice when the door is set but the executing path has no pp2 arm
/// (increment 1 wires only the plain eager decode loops).
pub fn warn_unwired_once(path: &str) {
    if std::env::var("MEMRA_PP_STAGES").map(|v| v == "2").unwrap_or(false)
        && !WARNED_UNWIRED.swap(true, Ordering::Relaxed)
    {
        eprintln!("[pp2] MEMRA_PP_STAGES=2 set but `{path}` has no pp2 arm (increment 1); running unsplit");
    }
}
