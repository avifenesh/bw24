//! M2 ppN gate: the N-stage pipeline-split eager decode (`MEMRA_PP_STAGES=N`, crate::pp)
//! must produce BIT-IDENTICAL logits to the unsplit `decode_step` at EVERY step — the
//! boundary handoffs are exact copies, so ANY differing bit (in any of the n_vocab f32
//! logits, prime or generate) = seam bug = FAIL. Two split arms are gated per invocation:
//!
//!   1. SERIAL replay: the door-on `decode_step` walk (N stage subgraphs, sync logits
//!      D2H per step) — the M2 increment-1/2 arm.
//!   2. PIPELINED replay (generic arm only): `decode_step_h_ppn_deferred` with a window
//!      of 3 tokens in flight and MEMRA_PP_OVERLAP forced on — the increment-3 arm. The
//!      deferred API is a SCHEDULING change (event-ordered math per token), so its
//!      logits must match the same reference bit-for-bit.
//!
//! Method: run P prompt + N generated steps with the door OFF recording full logits per
//! step, then replay the IDENTICAL token sequence into fresh caches with the door ON and
//! compare every f32 bit. The replayed inputs come from the reference greedy stream so a
//! mismatch cannot desync the comparison.
//!
//! WEIGHT SHARDING (increment 2) is exercised by SETTING THE DOOR BEFORE LOAD: under
//! MEMRA_PP_DEVICES (and MEMRA_PP_SHARD != 0) each stage's layer range uploads through
//! its own device. The door-OFF reference then runs the unsplit walk against the SAME
//! sharded placement — peer reads return identical bytes, so the reference stays exact
//! regardless of where the loader put the weights.
//!
//! Knobs PASS THROUGH from the caller's environment and are printed in the verdict:
//!   MEMRA_PP_DEVICES=d0,..,dN-1  stage->device placement (8x box: `0,1,..,N-1`)
//!   MEMRA_PP_SPLITS=c1,..,cN-1   explicit cuts (default even split)
//!   MEMRA_PP_SHARD=0             rollback to bring-up placement (weights all-primary)
//!   MEMRA_PP_STREAMS=0           increment-1 same-stream seam (serial arm only)
//!   MEMRA_PP_OVERLAP=1           double-buffered slots in the serial arm too
//!
//! usage: ppn-gate <model.gguf> [stages=2] [P=16] [N=32]
use memra_engine::forward::argmax;
use memra_engine::hybrid::HybridModel;
use memra_engine::Engine;
use memra_gguf::GgufFile;
use std::collections::VecDeque;

struct ArmCheck {
    name: &'static str,
    bad_steps: usize,
    first: Option<(usize, usize, f32, f32)>, // (step, idx, ref, got)
}

impl ArmCheck {
    fn new(name: &'static str) -> Self {
        ArmCheck { name, bad_steps: 0, first: None }
    }
    fn check(&mut self, step: usize, p: usize, got: &[f32], r: &[f32]) {
        let diffs = got.iter().zip(r.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        if diffs > 0 {
            self.bad_steps += 1;
            let (idx, (a, b)) = got
                .iter()
                .zip(r.iter())
                .enumerate()
                .find(|(_, (a, b))| a.to_bits() != b.to_bits())
                .map(|(i, (a, b))| (i, (*b, *a)))
                .unwrap();
            if self.first.is_none() {
                self.first = Some((step, idx, a, b));
            }
            if self.bad_steps <= 5 {
                println!(
                    "[{}] MISMATCH step {step} ({}): {diffs}/{} logits differ, first @[{idx}] \
                     ref={a:?} pp={b:?}",
                    self.name,
                    if step < p { "prime" } else { "gen" },
                    r.len()
                );
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: ppn-gate <model.gguf> [stages=2] [P=16] [N=32]");
    let stages: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(2);
    let p: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(16);
    let n: usize = std::env::args().nth(4).and_then(|s| s.parse().ok()).unwrap_or(32);

    // The gate owns the door and OPENS IT BEFORE LOAD (weight sharding is a load-time
    // decision). All other knobs deliberately pass through from the caller.
    unsafe {
        std::env::set_var("MEMRA_PP_STAGES", stages.to_string());
    }
    let devices_env = std::env::var("MEMRA_PP_DEVICES").unwrap_or_default();
    let primary_dev: usize = devices_env
        .split(',')
        .next()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0);
    let knobs = format!(
        "stages={stages} streams={} overlap={} devices={} splits={} shard={}",
        if memra_engine::pp::pp2_streams_off() { "OFF(inc1 seam)" } else { "per-stage" },
        if memra_engine::pp::pp2_overlap() { "1(double-buffered)" } else { "0" },
        if devices_env.is_empty() { "default(primary)" } else { &devices_env },
        std::env::var("MEMRA_PP_SPLITS").unwrap_or_else(|_| "default(even)".into()),
        if memra_engine::pp::pp_shard_off() { "OFF(bring-up placement)" } else { "per-stage" },
    );
    println!("ppn-gate M2 config: {knobs}");

    let e = Engine::new(primary_dev)?;
    let g = GgufFile::open(&path)?;
    let m = HybridModel::load(&e, &g)?; // sharded load when devices set + shard on
    let n_layers = m.layers.len();
    let fence = memra_engine::pp::pp_cuts(n_layers)
        .unwrap_or_else(|| panic!("ppn door failed to open (n_layers={n_layers}, stages={stages})"));
    assert_eq!(fence.len() - 1, stages, "fence {fence:?} != stages {stages}");
    println!("stage fence: {fence:?} over {n_layers} layers");
    let prompt: Vec<u32> = (0..p).map(|i| (100 + (i * 7) % 900) as u32).collect();

    // ---- reference: door OFF, record full logits at every step + the greedy stream ----
    unsafe {
        std::env::remove_var("MEMRA_PP_STAGES");
    }
    let mut cache_ref = memra_engine::cache::Cache::new(&e, &m.cfg, p + n + 8)?;
    let mut inputs: Vec<u32> = Vec::with_capacity(p + n);
    let mut ref_logits: Vec<Vec<f32>> = Vec::with_capacity(p + n);
    let mut next = 0u32;
    for step in 0..p + n {
        let tok = if step < p { prompt[step] } else { next };
        inputs.push(tok);
        let ll = m.decode_step(&e, tok, &mut cache_ref)?;
        next = argmax(&ll) as u32;
        ref_logits.push(ll);
    }
    let n_vocab = ref_logits[0].len();

    // ---- arm 1 (SERIAL): door ON, replay identical inputs, compare every bit ----
    unsafe {
        std::env::set_var("MEMRA_PP_STAGES", stages.to_string());
    }
    let mut serial = ArmCheck::new("serial");
    {
        let mut cache_pp = memra_engine::pp::new_cache(&e, &m.cfg, p + n + 8)?;
        for (step, &tok) in inputs.iter().enumerate() {
            let ll = m.decode_step(&e, tok, &mut cache_pp)?;
            serial.check(step, p, &ll, &ref_logits[step]);
        }
    }

    // ---- arm 2 (PIPELINED, generic arm only): deferred readback, window 3, overlap on ----
    // MEMRA_PP_STREAMS=0 is the increment-1 same-stream rollback seam and is documented
    // serial-only: the deferred API needs per-stage streams and (correctly) errors. Skip
    // the arm instead of aborting — otherwise the serial verdict never prints (the
    // 2026-08-02 streams0 no-verdict runs).
    let mut pipelined = if memra_engine::pp::pp2_streams_off() {
        println!("ppn gate NOTE: pipelined arm skipped (MEMRA_PP_STREAMS=0 is serial-only)");
        None
    } else if m.cfg.gemma4.is_none() {
        let overlap_prev = std::env::var("MEMRA_PP_OVERLAP").ok();
        unsafe {
            std::env::set_var("MEMRA_PP_OVERLAP", "1");
        }
        // Debug knob (bisect seam, not a config): MEMRA_PPN_GATE_WINDOW caps the
        // tokens-in-flight window of the pipelined arm (default 3; 1 = drain every step).
        let window: usize = std::env::var("MEMRA_PPN_GATE_WINDOW")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&w| w >= 1)
            .unwrap_or(3);
        let mut arm = ArmCheck::new("pipelined");
        {
            let mut cache_pl = memra_engine::pp::new_cache(&e, &m.cfg, p + n + 8)?;
            let mut pend: VecDeque<(usize, memra_engine::pp::PendingLogits)> = VecDeque::new();
            for (step, &tok) in inputs.iter().enumerate() {
                pend.push_back((step, m.decode_step_h_ppn_deferred(&e, tok, &mut cache_pl)?));
                if pend.len() >= window {
                    let (s0, pl) = pend.pop_front().unwrap();
                    arm.check(s0, p, &pl.wait()?, &ref_logits[s0]);
                }
            }
            while let Some((s0, pl)) = pend.pop_front() {
                arm.check(s0, p, &pl.wait()?, &ref_logits[s0]);
            }
        }
        unsafe {
            match overlap_prev {
                Some(v) => std::env::set_var("MEMRA_PP_OVERLAP", v),
                None => std::env::remove_var("MEMRA_PP_OVERLAP"),
            }
        }
        Some(arm)
    } else {
        println!("ppn gate NOTE: gemma4 arm is serial-only (pipelined arm skipped)");
        None
    };

    // ---- verdicts ----
    let total = p + n;
    let mut fail = false;
    for arm in [Some(&mut serial), pipelined.as_mut()].into_iter().flatten() {
        if arm.bad_steps == 0 {
            println!(
                "ppn gate PASS [{}]: {total} steps ({p} prime + {n} gen) BIT-IDENTICAL logits \
                 (n_vocab={n_vocab}, fence={fence:?}; {knobs})",
                arm.name
            );
        } else {
            let (s, i, a, b) = arm.first.unwrap();
            println!(
                "ppn gate FAIL [{}]: {}/{total} steps mismatched (first @ step {s} idx {i}: \
                 ref={a:?} pp={b:?}; fence={fence:?}; {knobs})",
                arm.name, arm.bad_steps
            );
            fail = true;
        }
    }
    if fail {
        std::process::exit(1);
    }
    Ok(())
}
