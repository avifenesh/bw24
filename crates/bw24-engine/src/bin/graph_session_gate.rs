//! graph-session-gate: GraphSession::step() must reproduce generate_graph token-for-token
//! (same prompt, same max_new) — the step-wise lift may not change one bit of the stream.
//! Also A/Bs step-loop tok/s vs eager decode_step for the serving-policy record.
//!
//! usage: graph-session-gate <model.gguf> [--steps 96]

use bw24_engine::cache::Cache;
use bw24_engine::decode::GraphDecodeState;
use bw24_engine::forward::argmax;
use bw24_engine::hybrid::HybridModel;
use bw24_engine::Engine;
use bw24_gguf::GgufFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: graph-session-gate <model.gguf> [--steps N]");
    let rest: Vec<String> = args.collect();
    let steps: usize = rest.iter().position(|a| a == "--steps")
        .and_then(|i| rest.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(96);

    let e = Engine::new(0)?;
    let g = GgufFile::open(&path)?;
    let model = HybridModel::load_without_mtp(&e, &g)?;
    let prompt: Vec<u32> = (0..48u32).map(|j| 55 + j * 31).collect();
    println!("loaded {} ({} layers); steps={steps}", g.arch().unwrap_or("?"), model.layers.len());

    // reference: generate_graph whole-loop
    let mut gs = GraphDecodeState::new(&e)?;
    let ref_out = model.generate_graph(&e, &mut gs, &prompt, steps)?;

    // session: prime+capture once, then step()
    let (mut sess, first) = model.graph_session_new(&e, &prompt, steps)?;
    let mut out = Vec::with_capacity(steps);
    out.push(first);
    let t0 = std::time::Instant::now();
    for _ in 1..steps {
        out.push(sess.step(&e)?);
    }
    let dt = t0.elapsed().as_secs_f64();
    let sess_tps = (steps - 1) as f64 / dt;

    // BW24_GS_PROF=1: decompose the step — fa_apply vs launch+sync vs token D2H.
    // Fresh session (the main gate consumed its budget).
    if std::env::var("BW24_GS_PROF").as_deref() == Ok("1") {
        let (mut sess, _first) = model.graph_session_new(&e, &prompt, 80)?;
        let (mut t_apply, mut t_launch, mut t_d2h) = (0.0f64, 0.0f64, 0.0f64);
        let n = 64.min(sess.bucket_max.saturating_sub(sess.cache.pos + 2));
        for _ in 0..n {
            let t0 = std::time::Instant::now();
            sess.prof_apply(&e)?;
            t_apply += t0.elapsed().as_secs_f64();
            let t0 = std::time::Instant::now();
            sess.prof_launch()?;
            t_launch += t0.elapsed().as_secs_f64();
            let t0 = std::time::Instant::now();
            let _ = sess.prof_read(&e)?;
            t_d2h += t0.elapsed().as_secs_f64();
        }
        println!("prof over {n}: fa_apply {:.0}us  launch(async) {:.0}us  d2h+sync {:.0}us  per step",
                 t_apply / n as f64 * 1e6, t_launch / n as f64 * 1e6, t_d2h / n as f64 * 1e6);
    }

    let ok = ref_out == out;
    println!("gate (session vs generate_graph, {steps} tokens): {}",
             if ok { "PASS" } else { "FAIL" });
    if !ok {
        let d = ref_out.iter().zip(out.iter()).position(|(a, b)| a != b);
        println!("  diverged at {:?}; ref[..8]={:?} sess[..8]={:?}",
                 d, &ref_out[..8.min(ref_out.len())], &out[..8.min(out.len())]);
    }

    // eager A/B on the same prompt (policy record)
    let mut c = Cache::new(&e, &model.cfg, prompt.len() + steps + 8)?;
    let _ = model.prime_cache(&e, &prompt, &mut c)?;
    let mut t = *prompt.last().unwrap();
    let t0 = std::time::Instant::now();
    for _ in 0..steps {
        let (l, _) = model.decode_step_h(&e, t, &mut c)?;
        t = argmax(&l) as u32;
    }
    let eager_tps = steps as f64 / t0.elapsed().as_secs_f64();
    println!("perf: session {sess_tps:.1} tok/s vs eager {eager_tps:.1} tok/s ({:+.1}%)",
             100.0 * (sess_tps - eager_tps) / eager_tps);

    if ok { println!("ALL GREEN: graph-session gate"); Ok(()) }
    else { Err("graph-session-gate FAILED".into()) }
}
