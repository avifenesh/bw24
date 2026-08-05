//! k27div-probe (lane/k27-divergence, 2026-08-05): teacher-forced divergence localizer for
//! the k27 fast-gate rig-divergence red (v0.70.0 release battery, pod vs 5090).
//!
//! Feeds a REFERENCE token tape (the 5090 golden) through the tokenwise decode path under
//! this rig's arithmetic, so both rigs see BIT-IDENTICAL inputs at every position. At each
//! step it prints this rig's greedy argmax, the top1-top2 margin, and the logit values of
//! the two DIVERGENCE-CANDIDATE tokens (golden pos-6 = 7246, pod pos-6 = 5638 by default),
//! so the first-div position's gap is measured, not inferred. The near-tie question —
//! "is the rig flip a margin~0 coin or a real numeric gap?" — is answered by
//! gap = l[watch_a] - l[watch_b] at the first disagreeing step.
//!
//! usage: k27div-probe <model.gguf> <prompt-file> <tape: comma-separated ids>
//!        [--watch a,b] (default 7246,5638)
//! env: the usual engine knobs (MEMRA_FA_SPLIT etc.) steer the arm under test.

use memra_engine::cache::Cache;
use memra_engine::forward::argmax;
use memra_engine::hybrid::HybridModel;
use memra_engine::Engine;
use memra_gguf::GgufFile;
use memra_tokenizer::Tokenizer;

fn top2(l: &[f32]) -> (usize, f32, usize, f32) {
    let (mut i1, mut v1, mut i2, mut v2) = (0usize, f32::NEG_INFINITY, 0usize, f32::NEG_INFINITY);
    for (i, &v) in l.iter().enumerate() {
        if v > v1 {
            i2 = i1; v2 = v1; i1 = i; v1 = v;
        } else if v > v2 {
            i2 = i; v2 = v;
        }
    }
    (i1, v1, i2, v2)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let model_path = args.first().expect("usage: k27div-probe <model.gguf> <prompt-file> <tape-ids>");
    let prompt_file = args.get(1).expect("need <prompt-file>");
    let tape: Vec<u32> = args.get(2).expect("need <tape ids>")
        .split(|c: char| !c.is_ascii_digit())
        .filter(|f| !f.is_empty())
        .map(|f| f.parse().unwrap())
        .collect();
    let watch: Vec<usize> = args.iter().position(|a| a == "--watch")
        .and_then(|i| args.get(i + 1))
        .map(|v| v.split(',').filter_map(|s| s.parse().ok()).collect())
        .unwrap_or_else(|| vec![7246, 5638]);
    assert!(!tape.is_empty(), "empty tape");

    let e = Engine::new(0)?;
    let g = GgufFile::open(model_path)?;
    let model = HybridModel::load_without_mtp(&e, &g)?;
    let tok = Tokenizer::from_gguf(&g).map_err(|err| format!("tokenizer: {err}"))?;
    let text = std::fs::read_to_string(prompt_file)?;
    let prompt = tok.encode(&text, true);
    println!("k27div: T={} tape={} watch={:?} sm_count={}", prompt.len(), tape.len(), watch,
             e.sm_count());
    println!("env: MEMRA_FA_SPLIT={} MEMRA_FAST={}",
             std::env::var("MEMRA_FA_SPLIT").unwrap_or_else(|_| "<unset>".into()),
             std::env::var("MEMRA_FAST").unwrap_or_else(|_| "<unset>".into()));

    let mut cache = Cache::new(&e, &model.cfg, prompt.len() + tape.len() + 8)?;
    let (mut logits, _, _) = model.prime_cache(&e, &prompt, &mut cache)?;
    let mut first_div: Option<usize> = None;
    for (step, &ref_tok) in tape.iter().enumerate() {
        let am = argmax(&logits) as u32;
        let (i1, v1, i2, v2) = top2(&logits);
        let ws: Vec<String> = watch.iter()
            .map(|&w| format!("l[{w}]={:.4}", logits[w]))
            .collect();
        let dis = if am != ref_tok { "  <-- DISAGREE" } else { "" };
        if am != ref_tok && first_div.is_none() { first_div = Some(step); }
        println!("step {step:2} t_kv={} argmax={am} ref={ref_tok} top2=({i1}:{v1:.4},{i2}:{v2:.4}) \
                  margin={:.4} {} gap_w0_w1={:.4}{dis}",
                 cache.pos, v1 - v2, ws.join(" "),
                 logits[watch[0]] - logits[watch[1]]);
        let (l, _) = model.decode_step_h(&e, ref_tok, &mut cache)?;
        logits = l;
    }
    match first_div {
        Some(s) => println!("FIRST-DIV vs tape at step {s}"),
        None => println!("NO DISAGREEMENT with tape over {} steps", tape.len()),
    }
    Ok(())
}
