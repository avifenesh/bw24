//! graph-allocfree-probe (Q1 lane, 2026-08-05): the generic GraphSession capture's
//! mem-node tax, measured rather than modelled.
//!
//! The sweep-audit brief (research/sweep-audits-20260805/AUDIT.md item 1a) predicted the
//! generic `decode_step_dc_cap_masked` capture carries per-layer alloc nodes + ~144 fa
//! memset nodes and pays `AUTO_FREE_ON_LAUNCH`'s ~0.25us/node launch-time mem-pool scan
//! ("205us on the 826-node step"). This probe measures the four quantities that decide
//! whether the slotted-buffer refactor is worth its correctness risk:
//!
//!   1. node census by type (the real ALLOC/FREE/MEMSET/KERNEL counts)
//!   2. per-step ASYNC launch cost — `graph.launch()` with NO sync, which is exactly the
//!      host-side work the auto-free scan inflates (the audit's target quantity)
//!   3. steady-state decode tok/s through `GraphSession::step`
//!   4. capture wall time (session-start latency)
//!
//! MEMRA_GRAPH_IFLAG={upload,priority} re-instantiates the SAME captured topology under a
//! non-auto-free flag: an A/B across that flag isolates the launch-scan cost from every
//! other property of the graph. Interleave arms with --reps for clock-drift-invariance
//! (the H100 lane's law 1).
//!
//! usage: graph-allocfree-probe <model.gguf> [--steps 96] [--reps 5] [--prompt-len 48]

use memra_engine::hybrid::HybridModel;
use memra_engine::Engine;
use memra_gguf::GgufFile;

fn arg_val(rest: &[String], key: &str) -> Option<String> {
    rest.iter().position(|a| a == key).and_then(|i| rest.get(i + 1)).cloned()
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n == 0 { return f64::NAN; }
    if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: graph-allocfree-probe <model.gguf> [--steps N] [--reps N]");
    let rest: Vec<String> = args.collect();
    let steps: usize = arg_val(&rest, "--steps").and_then(|v| v.parse().ok()).unwrap_or(96);
    let reps: usize = arg_val(&rest, "--reps").and_then(|v| v.parse().ok()).unwrap_or(5);
    let plen: usize = arg_val(&rest, "--prompt-len").and_then(|v| v.parse().ok()).unwrap_or(48);

    let e = Engine::new(0)?;
    let g = GgufFile::open(&path)?;
    let model = HybridModel::load_without_mtp(&e, &g)?;
    let prompt: Vec<u32> = (0..plen as u32).map(|j| 55 + j * 31).collect();
    let iflag = std::env::var("MEMRA_GRAPH_IFLAG").unwrap_or_else(|_| "auto_free".into());
    println!("model {path} arch {} ({} layers) steps={steps} reps={reps} prompt={plen} iflag={iflag}",
             g.arch().unwrap_or("?"), model.layers.len());

    // ---- capture time + node census (one session; census is topology, not timing) ----
    let mut cap_ms: Vec<f64> = Vec::new();
    for _ in 0..reps {
        let t0 = std::time::Instant::now();
        let (_sess, _first) = model.graph_session_new(&e, &prompt, steps)?;
        cap_ms.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    // NOTE: graph_session_new includes the eager prompt prime, so this is
    // session-start latency, not capture alone. MEMRA_GRAPH_CENSUS=1 prints the census.
    println!("capture+prime wall: median {:.1} ms over {reps} (min {:.1} max {:.1})",
             median(&mut cap_ms.clone()),
             cap_ms.iter().cloned().fold(f64::MAX, f64::min),
             cap_ms.iter().cloned().fold(0.0, f64::max));

    // ---- RECAPTURE cost, prime excluded: the node-count-scaling quantity ----
    // graph_session_recapture re-runs capture+instantiate on an ALREADY-PRIMED session, so
    // its wall time is the capture path alone (2 warmups + capture + instantiate + upload).
    // This is what a mem-node reduction would actually shrink, and it is paid per
    // kernel-class crossing during a live generation, not just at session start.
    {
        let (mut sess, _f) = model.graph_session_new(&e, &prompt, steps)?;
        let mut recap_ms: Vec<f64> = Vec::new();
        for _ in 0..reps {
            let t0 = std::time::Instant::now();
            model.graph_session_recapture_pub(&e, &mut sess)?;
            recap_ms.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        println!("recapture (capture+instantiate, no prime): median {:.1} ms  (raw {:?})",
                 median(&mut recap_ms.clone()),
                 recap_ms.iter().map(|v| format!("{v:.1}")).collect::<Vec<_>>());
    }

    // ---- per-step async launch cost: the auto-free scan's target quantity ----
    // prof_launch() issues cuGraphLaunch WITHOUT syncing, so the wall time is host-side
    // launch work only (the mem-pool scan lives here). prof_read carries the sync.
    let mut launch_us: Vec<f64> = Vec::new();
    let mut tps: Vec<f64> = Vec::new();
    for _ in 0..reps {
        let (mut sess, _first) = model.graph_session_new(&e, &prompt, steps)?;
        let n = 64.min(sess.bucket_max.saturating_sub(sess.cache.pos + 2));
        let mut acc = 0.0f64;
        for _ in 0..n {
            sess.prof_apply(&e)?;
            let t0 = std::time::Instant::now();
            sess.prof_launch()?;
            acc += t0.elapsed().as_secs_f64();
            let _ = sess.prof_read(&e)?;
        }
        launch_us.push(acc / n as f64 * 1e6);

        // steady-state tok/s on a fresh session (the decode arm)
        let (mut s2, _f2) = model.graph_session_new(&e, &prompt, steps)?;
        let n2 = (steps - 1).min(s2.bucket_max.saturating_sub(s2.cache.pos + 2));
        let t0 = std::time::Instant::now();
        for _ in 0..n2 { let _ = s2.step(&e, &model)?; }
        tps.push(n2 as f64 / t0.elapsed().as_secs_f64());
    }
    println!("launch(async) per step: median {:.1} us  (raw {:?})",
             median(&mut launch_us.clone()),
             launch_us.iter().map(|v| format!("{v:.1}")).collect::<Vec<_>>());
    println!("decode tok/s (session step): median {:.2}  (raw {:?})",
             median(&mut tps.clone()),
             tps.iter().map(|v| format!("{v:.2}")).collect::<Vec<_>>());
    println!("SUMMARY iflag={iflag} launch_us={:.1} tps={:.2} cap_ms={:.1}",
             median(&mut launch_us), median(&mut tps), median(&mut cap_ms));
    Ok(())
}
