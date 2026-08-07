//! prime-gate (gap #46): BATCHED-PRIME last-position logits vs the TOKENWISE-PRIME reference.
//!
//! The run-gen argmax gate compares forward_last vs the tokenwise decode loop — but real
//! generation (generate/generate_with, and therefore serving) seeds its first token from
//! `prime_cache`, a THIRD numeric config the gate never covered. The residency-cap lane
//! caught the gap live: naked Qwen3.6-35B flipped a near-tie first token (365 -> 198 "\n",
//! then EOS at 2 tokens) on the pp512 probe prompt while the run-gen gate was green.
//!
//! Per prompt this gate reports, for the same token ids:
//!   tw : tokenwise prime  (decode_step loop, m=1 — the oracle-stream config)
//!   bp : batched prime    (prime_cache, prefill GEMMs — the config that seeds generation)
//!   fl : forward_last     (the existing gate's reference, for triangulation)
//! argmax each + top1-top2 margins + full-vocab logit maxdiff(tw,bp), a bp determinism
//! rerun (bit compare), and optionally `--steps N` greedy streams from both primed caches
//! (first divergence step + EOS step — the "hit EOS at 2 tokens" symptom class).
//!
//! Verdict per prompt (shared with run-gen's line; forward::prime_gate_verdict):
//!   MATCH                 argmax agrees
//!   FLIP-NEARTIE          argmax differs on a near-tie within the calibrated drift bounds
//!                         — the accepted cross-config FP-composition class, REPORTED
//!   STRUCTURED            wide-margin flip or maxdiff beyond bounds — hard FAIL
//! Non-determinism of the batched prime is always a hard FAIL.
//!
//! usage: prime-gate <model.gguf> (--prompt "text" | --prompts-file f) [--chat]
//!                   [--steps N] [--jsonl out] [--strict]
//!   prompts-file: one prompt per line; a line starting with '@' names a FILE whose whole
//!   content (multi-line) is the prompt (e.g. @research/e2e/prompts/pp512.txt).
//!   --strict: near-tie flips also exit non-zero (for lanes that require stream identity).

use memra_engine::cache::Cache;
use memra_engine::forward::{argmax, prime_gate_verdict, top2, PrimeGateClass};
use memra_engine::hybrid::HybridModel;
use memra_engine::hybrid_forward::PRIME_MIN_T;
use memra_engine::Engine;
use memra_gguf::GgufFile;
use memra_tokenizer::Tokenizer;
use std::io::Write as _;

fn arg(rest: &[String], key: &str) -> Option<String> {
    rest.iter().position(|a| a == key).and_then(|i| rest.get(i + 1)).cloned()
}

fn load_prompts(rest: &[String]) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut prompts = Vec::new();
    if let Some(p) = arg(rest, "--prompt") {
        match p.strip_prefix('@') {
            Some(path) => prompts.push(std::fs::read_to_string(path)?),
            None => prompts.push(p),
        }
    }
    if let Some(f) = arg(rest, "--prompts-file") {
        for line in std::fs::read_to_string(&f)?.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Some(path) = line.strip_prefix('@') {
                prompts.push(std::fs::read_to_string(path)?);
            } else {
                prompts.push(line.to_string());
            }
        }
    }
    if prompts.is_empty() {
        return Err("prime-gate: need --prompt or --prompts-file".into());
    }
    Ok(prompts)
}

fn greedy_stream(
    e: &Engine,
    model: &HybridModel,
    cache: &mut Cache,
    seed: u32,
    steps: usize,
    eos: Option<u32>,
) -> Result<(Vec<u32>, Option<usize>), Box<dyn std::error::Error>> {
    // stream INCLUDES the seed (the first generated token) — EOS step is 1-based over it.
    let mut stream = vec![seed];
    let mut eos_step = if Some(seed) == eos { Some(1) } else { None };
    let mut t = seed;
    for s in 2..=steps {
        if eos_step.is_some() {
            break;
        }
        let l = model.decode_step(e, t, cache)?;
        t = argmax(&l) as u32;
        stream.push(t);
        if Some(t) == eos {
            eos_step = Some(s);
        }
    }
    Ok((stream, eos_step))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: prime-gate <model.gguf> [opts]");
    let rest: Vec<String> = args.collect();
    let chat = rest.iter().any(|a| a == "--chat");
    let strict = rest.iter().any(|a| a == "--strict");
    let steps: usize = arg(&rest, "--steps").and_then(|v| v.parse().ok()).unwrap_or(16);
    let jsonl = arg(&rest, "--jsonl");
    let prompts = load_prompts(&rest)?;

    let e = Engine::new(0)?;
    let g = GgufFile::open(&path)?;
    let model = HybridModel::load_without_mtp(&e, &g)?;
    let tok = Tokenizer::from_gguf(&g).map_err(|err| format!("tokenizer: {err}"))?;
    let eos = Some(tok.eos_id());
    println!(
        "prime-gate: {} ({} layers) prompts={} chat={chat} steps={steps}",
        g.arch().unwrap_or("?"),
        model.layers.len(),
        prompts.len()
    );

    let mut out = jsonl.as_ref().map(std::fs::File::create).transpose()?;
    let (mut n_match, mut n_neartie, mut n_structured, mut n_det_fail, mut n_skip) =
        (0usize, 0usize, 0usize, 0usize, 0usize);
    let (mut worst_maxdiff, mut min_flip_margin) = (0.0f32, f32::INFINITY);

    for (i, text) in prompts.iter().enumerate() {
        let toks: Vec<u32> = if chat {
            let rendered = tok.apply_chat_template(&[("user", text.as_str())], true);
            tok.encode(&rendered, true)
        } else {
            tok.encode(text, true)
        };
        let t = toks.len();
        if t < PRIME_MIN_T {
            println!("prompt {i:2}: T={t} < PRIME_MIN_T={PRIME_MIN_T} — SKIP (batched prime never engages)");
            n_skip += 1;
            continue;
        }
        let ctx = t + steps + 8;

        // tokenwise reference (the oracle-stream config; cache kept for the stream leg)
        let mut c_tw = Cache::new(&e, &model.cfg, ctx)?;
        let mut l_tw = Vec::new();
        for &tk in &toks {
            l_tw = model.decode_step(&e, tk, &mut c_tw)?;
        }

        // batched prime (the config that seeds generation) + determinism rerun
        let mut c_bp = Cache::new(&e, &model.cfg, ctx)?;
        let (l_bp, _, _) = model.prime_cache(&e, &toks, &mut c_bp, 0)?;
        let det_ok = {
            let mut c2 = Cache::new(&e, &model.cfg, ctx)?;
            let (l2, _, _) = model.prime_cache(&e, &toks, &mut c2, 0)?;
            l_bp.iter().zip(&l2).all(|(a, b)| a.to_bits() == b.to_bits())
        };
        if !det_ok {
            n_det_fail += 1;
        }

        // forward_last triangulation (the existing run-gen gate's reference config)
        let l_fl = model.forward_last(&e, &toks)?;
        let (fl1, ..) = top2(&l_fl);

        let v = prime_gate_verdict(&l_tw, &l_bp);
        let label = match v.class {
            PrimeGateClass::Match => "MATCH",
            PrimeGateClass::NearTieFlip => "FLIP-NEARTIE",
            PrimeGateClass::Structured => "STRUCTURED",
        };
        match v.class {
            PrimeGateClass::Match => n_match += 1,
            PrimeGateClass::NearTieFlip => n_neartie += 1,
            PrimeGateClass::Structured => n_structured += 1,
        }
        worst_maxdiff = worst_maxdiff.max(v.maxdiff);
        if v.tw_argmax != v.bp_argmax {
            min_flip_margin = min_flip_margin.min(v.tw_margin);
        }
        println!(
            "prompt {i:2} (T={t}): tw={} (margin {:.4}) bp={} (margin {:.4}) fl={fl1} \
             maxdiff={:.4e} det={} {label}",
            v.tw_argmax,
            v.tw_margin,
            v.bp_argmax,
            v.bp_margin,
            v.maxdiff,
            if det_ok { "BIT-IDENTICAL" } else { "*** NON-DETERMINISTIC ***" },
        );

        // greedy streams from both primed caches (serving symptom: early EOS / stream flip)
        let (mut first_div, mut tw_eos, mut bp_eos) = (None::<usize>, None, None);
        let (mut s_tw, mut s_bp) = (Vec::new(), Vec::new());
        if steps > 0 {
            let (a, ea) = greedy_stream(&e, &model, &mut c_tw, v.tw_argmax as u32, steps, eos)?;
            let (b, eb) = greedy_stream(&e, &model, &mut c_bp, v.bp_argmax as u32, steps, eos)?;
            first_div = a
                .iter()
                .zip(&b)
                .position(|(x, y)| x != y)
                .or_else(|| (a.len() != b.len()).then_some(a.len().min(b.len())));
            (tw_eos, bp_eos) = (ea, eb);
            println!(
                "          stream({steps}): {} tw_eos={tw_eos:?} bp_eos={bp_eos:?}",
                match first_div {
                    None => "MATCH".to_string(),
                    Some(d) => format!("DIVERGED at step {d} (0-based)"),
                }
            );
            (s_tw, s_bp) = (a, b);
        }

        if let Some(f) = out.as_mut() {
            writeln!(
                f,
                "{{\"i\":{i},\"t\":{t},\"chat\":{chat},\"tw_argmax\":{},\"bp_argmax\":{},\"fl_argmax\":{fl1},\
                 \"tw_margin\":{:.6},\"bp_margin\":{:.6},\"maxdiff\":{:.6e},\"det\":{det_ok},\
                 \"class\":\"{label}\",\"first_div\":{},\"tw_eos\":{},\"bp_eos\":{},\
                 \"tw_stream\":{:?},\"bp_stream\":{:?}}}",
                v.tw_argmax,
                v.bp_argmax,
                v.tw_margin,
                v.bp_margin,
                v.maxdiff,
                first_div.map_or("null".into(), |d| d.to_string()),
                tw_eos.map_or("null".into(), |d| d.to_string()),
                bp_eos.map_or("null".into(), |d| d.to_string()),
                s_tw,
                s_bp,
            )?;
        }
    }

    println!(
        "prime-gate SUMMARY: {} prompts ({n_skip} skipped short) | MATCH={n_match} \
         FLIP-NEARTIE={n_neartie} STRUCTURED={n_structured} det_fails={n_det_fail} | \
         worst maxdiff={worst_maxdiff:.4e} min flip margin={}",
        prompts.len(),
        if min_flip_margin.is_finite() {
            format!("{min_flip_margin:.4}")
        } else {
            "-".into()
        },
    );
    if n_structured > 0 || n_det_fail > 0 || (strict && n_neartie > 0) {
        Err(format!(
            "prime-gate FAIL: structured={n_structured} det_fails={n_det_fail} neartie={n_neartie} (strict={strict})"
        )
        .into())
    } else {
        println!(
            "prime-gate: {}",
            if n_neartie > 0 {
                "GREEN with reported near-tie flips (cross-config drift class)"
            } else {
                "ALL GREEN"
            }
        );
        Ok(())
    }
}
