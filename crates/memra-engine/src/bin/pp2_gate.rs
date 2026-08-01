//! M1-PP2 gate (increment 2): the 2-stage pipeline-split eager decode
//! (`MEMRA_PP_STAGES=2`, crate::pp) must produce BIT-IDENTICAL logits to the unsplit
//! `decode_step` at EVERY step — the boundary handoff is an exact copy, so ANY differing
//! bit (in any of the n_vocab f32 logits, prime or generate) = seam bug = FAIL.
//!
//! Method: run P prompt + N generated steps with the door OFF recording full logits per
//! step, then replay the IDENTICAL token sequence into a fresh cache with the door ON and
//! compare every f32 bit. The replayed inputs come from the reference greedy stream so a
//! mismatch cannot desync the comparison (every later step still compares like-for-like).
//!
//! Increment-2 knobs PASS THROUGH from the caller's environment and are printed in the
//! verdict so receipts are self-describing:
//!   MEMRA_PP_STREAMS=0   increment-1 same-stream seam (rollback);
//!                        default = per-stage streams + boundary events (increment 2)
//!   MEMRA_PP_OVERLAP=1   double-buffered boundary slots (M2 seed; default off)
//!   MEMRA_PP_DEVICES=a,b stage->device placement; sets stage-owned cache allocation.
//!                        On the 8x box the cross-device gate is exactly:
//!                        `MEMRA_PP_DEVICES=0,1 pp2-gate <model.gguf>`
//!
//! usage: pp2-gate <model.gguf> [P=16] [N=32] [split=n_layers/2]
use memra_engine::Engine;
use memra_engine::forward::argmax;
use memra_engine::hybrid::HybridModel;
use memra_gguf::GgufFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: pp2-gate <model.gguf> [P] [N] [split]");
    let p: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(16);
    let n: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(32);
    let split_arg: Option<usize> = std::env::args().nth(4).and_then(|s| s.parse().ok());

    // the gate owns the door: start clean regardless of the caller's environment.
    // (The increment-2 knobs — STREAMS/OVERLAP/DEVICES — deliberately pass through.)
    unsafe {
        std::env::remove_var("MEMRA_PP_STAGES");
        std::env::remove_var("MEMRA_PP_SPLIT");
    }
    let knobs = format!(
        "streams={} overlap={} devices={}",
        if memra_engine::pp::pp2_streams_off() { "OFF(inc1 seam)" } else { "per-stage" },
        if memra_engine::pp::pp2_overlap() { "1(double-buffered)" } else { "0" },
        std::env::var("MEMRA_PP_DEVICES").unwrap_or_else(|_| "default(primary)".into()),
    );
    println!("pp2-gate increment-2 config: {knobs}");

    let e = Engine::new(0)?;
    let g = GgufFile::open(&path)?;
    let m = HybridModel::load(&e, &g)?;
    let n_layers = m.layers.len();
    let split = split_arg.unwrap_or(n_layers / 2);
    let prompt: Vec<u32> = (0..p).map(|i| (100 + (i * 7) % 900) as u32).collect();

    // ---- reference: door OFF, record full logits at every step + the greedy stream ----
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

    // ---- door ON: replay the identical inputs into a fresh cache, compare every bit ----
    unsafe {
        std::env::set_var("MEMRA_PP_STAGES", "2");
        if let Some(s) = split_arg {
            std::env::set_var("MEMRA_PP_SPLIT", s.to_string());
        }
    }
    assert_eq!(
        memra_engine::pp::pp2_split(n_layers),
        Some(split),
        "pp2 door failed to open (n_layers={n_layers}, split={split})"
    );
    // stage-owned cache allocation when MEMRA_PP_DEVICES is set (increment-2 plumbing;
    // identical bytes on one device — that identity is part of what this gate proves).
    let mut cache_pp = memra_engine::pp::new_cache(&e, &m.cfg, p + n + 8)?;
    let mut bad_steps = 0usize;
    let mut first: Option<(usize, usize, f32, f32)> = None; // (step, idx, ref, pp)
    for (step, &tok) in inputs.iter().enumerate() {
        let ll = m.decode_step(&e, tok, &mut cache_pp)?;
        let r = &ref_logits[step];
        let diffs = ll
            .iter()
            .zip(r.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        if diffs > 0 {
            bad_steps += 1;
            let (idx, (a, b)) = ll
                .iter()
                .zip(r.iter())
                .enumerate()
                .find(|(_, (a, b))| a.to_bits() != b.to_bits())
                .map(|(i, (a, b))| (i, (*b, *a)))
                .unwrap();
            if first.is_none() {
                first = Some((step, idx, a, b));
            }
            if bad_steps <= 5 {
                println!(
                    "MISMATCH step {step} ({}): {diffs}/{n_vocab} logits differ, first @[{idx}] ref={a:?} pp2={b:?}",
                    if step < p { "prime" } else { "gen" }
                );
            }
        }
    }

    let total = p + n;
    if bad_steps == 0 {
        println!(
            "pp2 gate PASS: {total} steps ({p} prime + {n} gen) BIT-IDENTICAL logits \
             (n_vocab={n_vocab}, n_layers={n_layers}, stage0=[0,{split}), stage1=[{split},{n_layers}); {knobs})"
        );
        Ok(())
    } else {
        let (s, i, a, b) = first.unwrap();
        println!(
            "pp2 gate FAIL: {bad_steps}/{total} steps mismatched (first @ step {s} idx {i}: \
             ref={a:?} pp2={b:?}; split={split}/{n_layers}; {knobs})"
        );
        std::process::exit(1);
    }
}
