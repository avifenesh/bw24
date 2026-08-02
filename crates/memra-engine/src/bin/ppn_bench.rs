//! M2 ppN throughput bench: per-token decode cost of the N-stage pipeline vs unsplit,
//! and of the deferred-readback pipelined loop vs the serial one. Emits one JSONL row
//! per (arm, rep) — the raw log IS the receipt (evidence discipline: tee it).
//!
//! Arms (interleaved in-process, rep-major — never back-to-back same-arm batches):
//!   - door CLOSED (no MEMRA_PP_STAGES in the caller env): `serial-off` only — the
//!     single-GPU baseline invocation. Run this as its OWN invocation with no pp env so
//!     the load is unsharded (an in-process door toggle after a sharded load would time
//!     peer-reads and pollute the baseline).
//!   - door OPEN (MEMRA_PP_STAGES=N [+ MEMRA_PP_DEVICES/SPLITS/SHARD pass-through]):
//!     `serial-pp` (door-on decode_step, sync D2H per token) and `pipelined-pp`
//!     (decode_step_h_ppn_deferred, window 3, MEMRA_PP_OVERLAP forced on).
//!
//! Method: door-off greedy P+G run records the token stream (and warms every kernel);
//! each timed arm replays the IDENTICAL stream into a fresh cache — prime steps run
//! and are then EXCLUDED from timing; the clock covers the G generate-step replays
//! (deferred arm: including the full drain).
//!
//! usage: ppn-bench <model.gguf> [P=32] [G=128] [reps=5]
use memra_engine::forward::argmax;
use memra_engine::hybrid::HybridModel;
use memra_engine::Engine;
use memra_gguf::GgufFile;
use std::collections::VecDeque;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: ppn-bench <model.gguf> [P=32] [G=128] [reps=5]");
    let p: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(32);
    let g: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(128);
    let reps: usize = std::env::args().nth(4).and_then(|s| s.parse().ok()).unwrap_or(5);

    let stages_env = std::env::var("MEMRA_PP_STAGES").ok().filter(|v| {
        !v.is_empty() && v != "0" && v != "1"
    });
    let door_open = stages_env.is_some();
    let devices_env = std::env::var("MEMRA_PP_DEVICES").unwrap_or_default();
    let primary_dev: usize = devices_env
        .split(',')
        .next()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0);
    let knobs = format!(
        "stages={} devices={} splits={} shard={} streams={}",
        stages_env.clone().unwrap_or_else(|| "OFF".into()),
        if devices_env.is_empty() { "default(primary)" } else { &devices_env },
        std::env::var("MEMRA_PP_SPLITS").unwrap_or_else(|_| "default(even)".into()),
        if memra_engine::pp::pp_shard_off() { "OFF" } else { "per-stage" },
        if memra_engine::pp::pp2_streams_off() { "OFF(inc1)" } else { "per-stage" },
    );
    println!("ppn-bench M2 config: {knobs} P={p} G={g} reps={reps} model={path}");

    let e = Engine::new(primary_dev)?;
    let gf = GgufFile::open(&path)?;
    let m = HybridModel::load(&e, &gf)?; // sharded when the caller opened the door
    let n_layers = m.layers.len();
    if door_open {
        let fence = memra_engine::pp::pp_cuts(n_layers)
            .expect("ppn-bench: door env set but pp_cuts is None");
        println!("stage fence: {fence:?} over {n_layers} layers");
    }
    let prompt: Vec<u32> = (0..p).map(|i| (100 + (i * 7) % 900) as u32).collect();

    // ---- reference stream (door OFF; also the warmup) ----
    let saved_stages = stages_env.clone();
    unsafe {
        std::env::remove_var("MEMRA_PP_STAGES");
    }
    let mut inputs: Vec<u32> = Vec::with_capacity(p + g);
    {
        let mut cache = memra_engine::cache::Cache::new(&e, &m.cfg, p + g + 8)?;
        let mut next = 0u32;
        for step in 0..p + g {
            let tok = if step < p { prompt[step] } else { next };
            inputs.push(tok);
            let ll = m.decode_step(&e, tok, &mut cache)?;
            next = argmax(&ll) as u32;
        }
    }

    #[derive(Clone, Copy, PartialEq)]
    enum Arm {
        SerialOff,
        SerialPp,
        PipelinedPp,
    }
    let arms: Vec<(Arm, &str)> = if door_open {
        let mut v = vec![(Arm::SerialPp, "serial-pp")];
        if m.cfg.gemma4.is_none() && !memra_engine::pp::pp2_streams_off() {
            v.push((Arm::PipelinedPp, "pipelined-pp"));
        }
        v
    } else {
        vec![(Arm::SerialOff, "serial-off")]
    };

    let mut times: Vec<(String, Vec<f64>)> =
        arms.iter().map(|(_, name)| (name.to_string(), Vec::new())).collect();

    for rep in 0..reps {
        for (ai, &(arm, name)) in arms.iter().enumerate() {
            // per-arm door state
            unsafe {
                match arm {
                    Arm::SerialOff => std::env::remove_var("MEMRA_PP_STAGES"),
                    _ => std::env::set_var("MEMRA_PP_STAGES", saved_stages.clone().unwrap()),
                }
                if arm == Arm::PipelinedPp {
                    std::env::set_var("MEMRA_PP_OVERLAP", "1");
                } else {
                    std::env::remove_var("MEMRA_PP_OVERLAP");
                }
            }
            let mut cache = memra_engine::pp::new_cache(&e, &m.cfg, p + g + 8)?;
            // prime (untimed, serial in every arm)
            for &tok in inputs.iter().take(p) {
                m.decode_step(&e, tok, &mut cache)?;
            }
            let ms = match arm {
                Arm::SerialOff | Arm::SerialPp => {
                    let t0 = Instant::now();
                    for &tok in inputs.iter().skip(p) {
                        m.decode_step(&e, tok, &mut cache)?;
                    }
                    t0.elapsed().as_secs_f64() * 1e3
                }
                Arm::PipelinedPp => {
                    let mut pend: VecDeque<memra_engine::pp::PendingLogits> = VecDeque::new();
                    let t0 = Instant::now();
                    for &tok in inputs.iter().skip(p) {
                        pend.push_back(m.decode_step_h_ppn_deferred(&e, tok, &mut cache)?);
                        if pend.len() >= 3 {
                            pend.pop_front().unwrap().wait()?;
                        }
                    }
                    while let Some(pl) = pend.pop_front() {
                        pl.wait()?;
                    }
                    t0.elapsed().as_secs_f64() * 1e3
                }
            };
            let us_tok = ms * 1e3 / g as f64;
            println!(
                "{{\"arm\":\"{name}\",\"rep\":{rep},\"g\":{g},\"ms\":{ms:.3},\
                 \"us_per_tok\":{us_tok:.2},\"tok_s\":{:.2}}}",
                g as f64 / (ms / 1e3)
            );
            times[ai].1.push(us_tok);
        }
    }

    println!("---- medians (us/token over {g} gen steps, N={reps} reps, interleaved) ----");
    for (name, mut v) in times {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = v[v.len() / 2];
        println!(
            "{name}: median {med:.2} us/tok ({:.2} tok/s)  all={:?}",
            1e6 / med,
            v.iter().map(|x| (x * 100.0).round() / 100.0).collect::<Vec<_>>()
        );
    }
    Ok(())
}
