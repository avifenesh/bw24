//! Slice-3 model-level gate for the per-block FP8 MMQ prefill kernel (lane/fp8-mmq).
//!
//! Why a new bin: `run-gen` takes the HybridModel path only (`load_from_source_without_mtp`) and
//! panics "not a hybrid arch" on a dense checkpoint — see
//! research/fp8st-20260803/armb/rungen-gpu.log, the same wall ARM B' hit. The block-128 FP8
//! checkpoint on this rig is dense (Qwen3-1.7B), so the greedy stream needs the dense seam.
//!
//! What it does: greedy N-token continuation by re-prefilling the growing sequence each step
//! (`Model::forward_last`). That is deliberately the *prefill* path every step, so with
//! `MEMRA_FP8_MMQ=1` every projection GEMM in every step routes through the new tile (m = T >= 16
//! clears GEMM_M_THRESHOLD). A decode-cache stream would instead run m=1 MMVQ and never touch it.
//!
//! Output is one line per step, `step tok logit top2 margin`, plus a final FNV-1a digest over the
//! emitted ids — so two runs (reference vs MEMRA_FP8_MMQ=1) diff mechanically.
//!
//! `MEMRA_FP8_MMQ_TF=<file>` switches to TEACHER-FORCED mode: instead of feeding back its own
//! argmax, every step appends the id read from `<file>` (one `stream ids: [...]` line, i.e. a
//! previous run's log). Both arms then see BIT-IDENTICAL inputs at every position, which separates
//! "the arithmetic drifted" from "a near-tie flip at step k rerouted every later step". Without
//! it, a single flip makes the two streams incomparable after that point.
//!
//! `MEMRA_FP8_MMQ_LOGITS=<file>` dumps the step-0 logit vector as raw little-endian f32 — the
//! input for the max_abs / rms_rel drift numbers.
//!
//! `MEMRA_FP8_MMQ_NLL=<text_file>` runs QUALITY SANITY instead of a stream: one prefill over a
//! real text window and the mean token NLL (natural log) over positions 1..T. This is the arm
//! comparison that does not favour any arm by construction — an NLL measured on the floor's own
//! greedy output would reward whichever arm reproduces the floor, which is the thing under test.
//! Requires `tokenizer.json` in the checkpoint dir.
//!
//! Usage: fp8-mmq-stream <hf_dir> <n_steps> [tok ids...]

use memra_engine::Engine;
use memra_engine::forward::{argmax, top2};
use memra_engine::model::Model;
use memra_gguf::source::{SafetensorsSource, TensorSource};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: fp8-mmq-stream <hf_dir> <n_steps> [tok ids...]");
    let n_steps: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let mut toks: Vec<u32> = std::env::args().skip(3).filter_map(|s| s.parse().ok()).collect();
    if toks.is_empty() {
        // 24 ids: past GEMM_M_THRESHOLD=16 from the very first step, so the kernel is live for
        // every step of the stream rather than only the later ones.
        toks = vec![
            151643, 9707, 11, 1879, 30, 33464, 264, 3766, 315, 279, 1372, 220, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 26, 27,
        ];
    }

    let e = Engine::new(0)?;
    let src = SafetensorsSource::open(std::path::Path::new(&path))?;
    let cfg = src.config();
    println!("GPU: {}  arch: {:?}", e.ctx().name()?, cfg.arch);
    println!(
        "MEMRA_FP8_MMQ={}  MEMRA_PP_FP8={}  MEMRA_FP8_BLK_GPU={}  MEMRA_PP_FP8_BUDGET_MB={}",
        std::env::var("MEMRA_FP8_MMQ").unwrap_or_else(|_| "<unset>".into()),
        std::env::var("MEMRA_PP_FP8").unwrap_or_else(|_| "<unset>".into()),
        std::env::var("MEMRA_FP8_BLK_GPU").unwrap_or_else(|_| "<unset>".into()),
        std::env::var("MEMRA_PP_FP8_BUDGET_MB").unwrap_or_else(|_| "<unset>".into()),
    );
    let model = Model::load_dense_from_source(&e, &src)?;
    println!(
        "loaded dense: n_layer={} n_embd={} n_ff={} n_vocab={}  prompt={} toks",
        model.cfg.n_layer, model.cfg.n_embd, model.cfg.n_ff, model.cfg.n_vocab, toks.len()
    );

    // --- PERF: pp-only prefill throughput (slice 4) ---------------------------------------------
    // MEMRA_PP_ONLY=1 + MEMRA_PP_REPS=<n>: warmup forward, then n timed prefills over a prompt of
    // MEMRA_PP_TOKENS tokens (from MEMRA_PROMPT_FILE if set, else the synthetic ramp), printing the
    // per-rep and MEDIAN tok/s in run-gen's line format so the same parsers work. Exits before any
    // generation, so the number is PURE prefill — the stage this kernel lives in.
    if std::env::var("MEMRA_PP_ONLY").is_ok() {
        let reps: usize = std::env::var("MEMRA_PP_REPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3);
        let want: usize = std::env::var("MEMRA_PP_TOKENS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(512);
        let ids: Vec<u32> = match std::env::var("MEMRA_PROMPT_FILE") {
            Ok(f) => {
                let text = std::fs::read_to_string(&f)?;
                let tok = memra_tokenizer::Tokenizer::from_hf_dir(std::path::Path::new(&path))
                    .map_err(|err| format!("HF tokenizer init failed: {err}"))?;
                let mut v = tok.encode(&text, true);
                if v.len() < want {
                    return Err(format!(
                        "MEMRA_PROMPT_FILE {f} tokenizes to {} tokens, need {want}",
                        v.len()
                    )
                    .into());
                }
                v.truncate(want);
                v
            }
            // Synthetic ramp: deterministic, in-vocab, and long enough for any pp length.
            Err(_) => (0..want).map(|i| (1000 + (i * 7919) % 90000) as u32).collect(),
        };
        let t = ids.len();
        println!("pp-only: {t} tokens, {reps} timed reps (+1 warmup)");
        let _ = model.forward_last(&e, &ids)?; // warmup: allocations, NaN scan, autotune
        let mut secs: Vec<f64> = Vec::with_capacity(reps);
        for r in 0..reps {
            let t0 = std::time::Instant::now();
            let _ = model.forward_last(&e, &ids)?;
            let dt = t0.elapsed().as_secs_f64();
            secs.push(dt);
            println!("pp-only rep {r}: {t} tok in {dt:.4}s = {:.1} tok/s", t as f64 / dt);
        }
        secs.sort_by(f64::total_cmp);
        let med = secs[secs.len() / 2];
        println!(
            "pp-only MEDIAN: {t} tok in {med:.4}s = {:.1} tok/s",
            t as f64 / med
        );
        return Ok(());
    }

    // --- QUALITY SANITY: mean token NLL over a real text window (one prefill) ------------------
    if let Ok(txt_file) = std::env::var("MEMRA_FP8_MMQ_NLL") {
        let text = std::fs::read_to_string(&txt_file)?;
        let tok = memra_tokenizer::Tokenizer::from_hf_dir(std::path::Path::new(&path))
            .map_err(|err| format!("HF tokenizer init failed: {err}"))?;
        let mut ids = tok.encode(&text, true);
        ids.truncate(n_steps.max(2)); // n_steps doubles as the window length here
        let t = ids.len();
        let all = model.forward(&e, &ids)?;
        let n_vocab = model.cfg.n_vocab as usize;
        // mean NLL over positions 1..T: -log softmax(logits[p-1])[ids[p]]
        let mut sum = 0.0f64;
        for p in 1..t {
            let row = &all[(p - 1) * n_vocab..p * n_vocab];
            let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
            let lse = mx + row.iter().map(|&v| ((v as f64) - mx).exp()).sum::<f64>().ln();
            sum += lse - row[ids[p] as usize] as f64;
        }
        let nll = sum / (t - 1) as f64;
        println!(
            "NLL window: file={txt_file} tokens={t} mean_nll={nll:.6} ppl={:.6}",
            nll.exp()
        );
        return Ok(());
    }

    // Teacher-forcing tape: the `stream ids: [...]` line of a prior run.
    let forced: Option<Vec<u32>> = match std::env::var("MEMRA_FP8_MMQ_TF") {
        Ok(f) => {
            let txt = std::fs::read_to_string(&f)?;
            let line = txt
                .lines()
                .find(|l| l.starts_with("stream ids: ["))
                .ok_or_else(|| format!("no `stream ids: [` line in {f}"))?;
            let inner = line
                .trim_start_matches("stream ids: [")
                .trim_end_matches(']');
            let ids: Vec<u32> = inner
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            println!("teacher-forced from {f}: {} ids", ids.len());
            Some(ids)
        }
        Err(_) => None,
    };
    let dump_logits = std::env::var("MEMRA_FP8_MMQ_LOGITS").ok();

    let mut digest: u64 = 0xcbf2_9ce4_8422_2325;
    let mut emitted: Vec<u32> = Vec::with_capacity(n_steps);
    let mut flips = 0usize;
    for step in 0..n_steps {
        let logits = model.forward_last(&e, &toks)?;
        let bad = logits.iter().filter(|v| !v.is_finite()).count();
        if bad != 0 {
            println!("STEP {step}: non-finite logits {bad}/{} — forward broken", logits.len());
            std::process::exit(1);
        }
        if step == 0 {
            if let Some(f) = &dump_logits {
                let mut raw = Vec::with_capacity(logits.len() * 4);
                for v in &logits {
                    raw.extend_from_slice(&v.to_le_bytes());
                }
                std::fs::write(f, &raw)?;
                println!("step-0 logits -> {f} ({} f32)", logits.len());
            }
        }
        let (i1, v1, i2, v2) = top2(&logits);
        debug_assert_eq!(i1, argmax(&logits));
        // In teacher-forced mode the argmax is still recorded (that is the comparison signal); the
        // NEXT input comes from the tape, so both arms stay on the same trajectory regardless.
        let next = match &forced {
            Some(ids) => *ids.get(step).ok_or_else(|| {
                format!("teacher tape has {} ids, need {n_steps}", ids.len())
            })?,
            None => i1 as u32,
        };
        if forced.is_some() && next != i1 as u32 {
            flips += 1;
        }
        println!(
            "step {step:3} tok {i1:6} logit {v1:.6} top2 {i2:6} {v2:.6} margin {:.6}",
            v1 - v2
        );
        for b in (i1 as u32).to_le_bytes() {
            digest ^= b as u64;
            digest = digest.wrapping_mul(0x0100_0000_01b3);
        }
        emitted.push(i1 as u32);
        toks.push(next);
    }
    if forced.is_some() {
        println!("teacher-forced argmax disagreements: {flips}/{n_steps}");
    }
    println!("stream ids: {emitted:?}");
    println!("stream digest: {digest:#018x}  steps={n_steps}");
    Ok(())
}
