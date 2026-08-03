//! W4A4 greedy-decode comparison gate (lane/w4a4-rescue).
//!
//! The W4A4 FP4 MMQ prefill arm is exactness-blocked: its e2m1 ACTIVATION quantization forks
//! greedy decode from the reference (W4A8 int8-activation) path on long real prompts. Every prior
//! reject was recorded as a one-line "argmax A vs B MISMATCH" from two SEPARATE run-gen
//! invocations — which cannot separate an activation-quant fork from load-order/clock noise, and
//! never captured WHERE in the generated stream the two arms part company.
//!
//! This gate runs BOTH arms in ONE process against ONE set of loaded weights. The prefill matmul
//! dispatch (`Engine::mmq_supports`) reads `MEMRA_MMQ` fresh on every call, so the arms are
//! switched at runtime between passes: the ONLY difference between them is which activation
//! contract the NVFP4 prefill GEMM uses. Requires `MEMRA_RP=0` at process start (the split-plane
//! repack is a load-time decision, and an rp weight always takes W4A8).
//!
//! Each arm makes three passes: `forward_last` (batched prefill logits, run-gen's gate reference),
//! a tokenwise `decode_step` sweep (m=1 throughout, so it never reaches the prefill GEMM — the
//! arm-invariant control), and `prime_cache` + greedy decode, which is what `generate_with`
//! actually serves. The third pass is the one that matters: priming through the batched prefill
//! writes EVERY prompt position's K/V under the arm's activation contract, so the continuation
//! carries the arm's numeric class the whole way. Seeding a tokenwise-built cache instead puts the
//! arm in the seed token only and hides the fork almost entirely.
//!
//! Emitted per cell: the prefill/decode/prime argmaxes per arm, the first divergent greedy position
//! with both arms' token ids AND both arms' top-logit values at that position, and the logit
//! maxdiff over the compared prefix. One JSON object per line on stdout (prefix `JSONL `) so the
//! caller tees a raw log and parses the log, never the pipe.
//!
//! usage: w4a4-gate <model.gguf> <prompt-file> [ngen]

use memra_engine::forward::argmax;
use memra_engine::hybrid::HybridModel;
use memra_engine::Engine;
use memra_gguf::GgufFile;
use memra_tokenizer::Tokenizer;

/// Which activation contract the NVFP4 prefill GEMM runs under.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arm {
    /// int8 q8_1 activation (the shipped default, the exactness reference).
    W4a8Reference,
    /// e2m1 FP4 activation (the blocked speed arm under test).
    W4a4Test,
}

impl Arm {
    fn label(self) -> &'static str {
        match self {
            Arm::W4a8Reference => "w4a8-reference",
            Arm::W4a4Test => "w4a4-test",
        }
    }

    /// Flip the prefill dispatch. Safe here: single-threaded gate, and the dispatch re-reads the
    /// var per matmul rather than caching it in a OnceLock.
    fn engage(self) {
        match self {
            Arm::W4a8Reference => unsafe { std::env::remove_var("MEMRA_MMQ") },
            Arm::W4a4Test => unsafe { std::env::set_var("MEMRA_MMQ", "1") },
        }
    }
}

/// One arm's pass over a prompt.
struct ArmRun {
    /// Batched `forward_last` verdict (run-gen's prefill gate reference).
    prefill_argmax: usize,
    /// Tokenwise decode verdict at the same position. m=1 throughout, so this never reaches the
    /// prefill GEMM and is identical in both arms — the in-arm control.
    decode_argmax: usize,
    /// max |prefill - decode| over the vocab row — the in-arm prefill/decode agreement.
    self_maxdiff: f32,
    /// `forward_last` logit row, kept for the cross-arm compare.
    prefill_logits: Vec<f32>,
    /// `prime_cache` (the SERVING prefill) logit row at the last prompt position.
    prime_logits: Vec<f32>,
    prime_argmax: usize,
    /// Greedy continuation off the serving-primed cache — the stream a user actually sees.
    tokens: Vec<u32>,
    /// Greedy continuation off the TOKENWISE-built cache, same arm, same weights, same process.
    /// The gate's NOISE FLOOR: this stream and `tokens` differ only in which prefill entry point
    /// wrote the prompt's K/V (batched `prime_cache` vs m=1 `decode_step`), never in the activation
    /// contract. If a single arm forks its own continuation here, then "byte-identical greedy output
    /// vs the reference" is not a property any prefill arm has, and the W4A4 verdict has to be read
    /// against this floor rather than against zero.
    tokens_tokenwise: Vec<u32>,
    /// `decide_rows[p]` is the logit row whose argmax produced `tokens[p]` (row 0 IS `prime_logits`).
    /// Kept so the FIRST DIVERGENT position can be classified instead of only position 0: up to that
    /// position both arms fed the same prefix, so the two rows are directly comparable, and quoting
    /// both arms' values for both candidate ids separates a near-tie (two tokens within noise, the
    /// arm merely rounded the other way) from a structural fork (the arm prefers a different token
    /// by a wide margin). ~29 MB per arm at ngen=48 on a 150k vocab — worth it for the signature.
    decide_rows: Vec<Vec<f32>>,
}

fn run_arm(
    e: &Engine,
    model: &HybridModel,
    prompt: &[u32],
    ngen: usize,
    arm: Arm,
) -> Result<ArmRun, Box<dyn std::error::Error>> {
    arm.engage();
    let max_ctx = prompt.len() + ngen + 8;

    // (1) Batched prefill last-position logits — the path the W4A4 GEMM serves.
    let prefill_logits = model.forward_last(e, prompt)?;
    let prefill_argmax = argmax(&prefill_logits);

    // (2) Tokenwise decode over the same prompt: the arm-invariant control.
    // Sized for the prompt AND the noise-floor continuation (pass 2b) off the same cache.
    let mut dcache = memra_engine::cache::Cache::new(e, &model.cfg, max_ctx)?;
    let mut logits = Vec::new();
    for &t in prompt {
        logits = model.decode_step(e, t, &mut dcache)?;
    }
    let decode_argmax = argmax(&logits);
    let self_maxdiff = prefill_logits
        .iter()
        .zip(&logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // (2b) NOISE FLOOR: greedy-decode the same ngen off the TOKENWISE-primed cache. Same arm, so any
    //      fork against pass (3) is entry-point noise (batched prefill vs m=1 decode over the same
    //      prompt), not an activation-contract difference. Without this control a cross-arm fork
    //      can't be attributed to W4A4 at all.
    let mut tokens_tokenwise = Vec::with_capacity(ngen);
    {
        let mut nx = decode_argmax as u32;
        for _ in 0..ngen {
            tokens_tokenwise.push(nx);
            let l = model.decode_step(e, nx, &mut dcache)?;
            nx = argmax(&l) as u32;
        }
    }
    drop(dcache);

    // (3) The SERVING path: prime the whole KV cache through the batched prefill (this is what
    //     generate_with does), then greedy-decode off it. Every prompt position's K/V was
    //     produced under this arm's activation contract, so the continuation carries the arm's
    //     numeric class throughout — not just in the seed token. Priming tokenwise instead
    //     (the earlier shape of this gate) hides the fork almost entirely.
    let mut cache = memra_engine::cache::Cache::new(e, &model.cfg, max_ctx)?;
    let (prime_logits, _h, _x) = model.prime_cache(e, prompt, &mut cache)?;
    let prime_argmax = argmax(&prime_logits);

    let mut tokens = Vec::with_capacity(ngen);
    let mut decide_rows = Vec::with_capacity(ngen);
    let mut next = prime_argmax as u32;
    let mut deciding = prime_logits.clone();
    for _ in 0..ngen {
        tokens.push(next);
        decide_rows.push(std::mem::take(&mut deciding));
        logits = model.decode_step(e, next, &mut cache)?;
        next = argmax(&logits) as u32;
        deciding = logits.clone();
    }

    Ok(ArmRun {
        prefill_argmax,
        decode_argmax,
        self_maxdiff,
        prefill_logits,
        prime_logits,
        prime_argmax,
        tokens,
        tokens_tokenwise,
        decide_rows,
    })
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = std::env::args()
        .nth(1)
        .expect("usage: w4a4-gate <model.gguf> <prompt-file> [ngen]");
    let prompt_path = std::env::args()
        .nth(2)
        .expect("usage: w4a4-gate <model.gguf> <prompt-file> [ngen]");
    let ngen: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(48);

    // The rp split-plane repack must be OFF: an rp weight always takes W4A8, so the W4A4 arm
    // would silently never engage and the gate would report a false PASS.
    if std::env::var("MEMRA_RP").as_deref() != Ok("0") {
        return Err("w4a4-gate requires MEMRA_RP=0 (an rp weight always takes the W4A8 tile, \
                    so the W4A4 arm would never engage)"
            .into());
    }

    let e = Engine::new(0)?;
    let g = GgufFile::open(&model_path)?;
    let model = HybridModel::load(&e, &g)?;
    let tok = Tokenizer::from_gguf(&g).ok();

    let text = std::fs::read_to_string(&prompt_path)?;
    let prompt: Vec<u32> = match &tok {
        Some(t) => t.encode(&text, true),
        None => return Err("model carries no usable tokenizer — cannot build the text prompt".into()),
    };
    if prompt.len() < memra_engine::hybrid_forward::PRIME_MIN_T {
        return Err(format!(
            "prompt is {} tokens; the batched serving prime needs >= {} — a shorter prompt never \
             reaches the prefill GEMM and the two arms are trivially identical",
            prompt.len(),
            memra_engine::hybrid_forward::PRIME_MIN_T
        )
        .into());
    }
    let cell = std::path::Path::new(&prompt_path)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "prompt".into());
    let model_name = std::path::Path::new(&model_path)
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| model_path.clone());

    println!(
        "w4a4-gate: model={model_name} cell={cell} prompt_tokens={} ngen={ngen}",
        prompt.len()
    );

    // Reference first, then the arm under test. Same process, same weights, same clocks.
    let reference = run_arm(&e, &model, &prompt, ngen, Arm::W4a8Reference)?;
    let test = run_arm(&e, &model, &prompt, ngen, Arm::W4a4Test)?;

    for (arm, r) in [
        (Arm::W4a8Reference, &reference),
        (Arm::W4a4Test, &test),
    ] {
        println!(
            "  [{}] prefill argmax={} decode argmax={} self maxdiff={:.3e} {} | prime argmax={}",
            arm.label(),
            r.prefill_argmax,
            r.decode_argmax,
            r.self_maxdiff,
            if r.prefill_argmax == r.decode_argmax {
                "MATCH"
            } else {
                "MISMATCH"
            },
            r.prime_argmax,
        );
    }

    // Cross-arm logit distance at the seeded position, for both prefill entry points.
    let cross_maxdiff = reference
        .prefill_logits
        .iter()
        .zip(&test.prefill_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let cross_prime_maxdiff = reference
        .prime_logits
        .iter()
        .zip(&test.prime_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // First divergent greedy position + both arms' view of BOTH candidate ids there. A near-tie
    // (both arms' margins tiny) is a different failure class from a wide-margin structural fork.
    let first_div = reference
        .tokens
        .iter()
        .zip(&test.tokens)
        .position(|(a, b)| a != b);

    let (div_json, verdict) = match first_div {
        None => (
            format!(
                "\"first_divergent_pos\":null,\"cross_prefill_maxdiff\":{cross_maxdiff:.6e},\
                 \"cross_prime_maxdiff\":{cross_prime_maxdiff:.6e}"
            ),
            "IDENTICAL",
        ),
        Some(pos) => {
            let ref_id = reference.tokens[pos] as usize;
            let test_id = test.tokens[pos] as usize;
            // Quote both arms' logits for BOTH candidate ids at the position where they actually
            // part company. Every token before `pos` matched, so both arms decoded the same prefix
            // and their deciding rows are directly comparable — the margins classify the fork:
            // both small = near-tie (the arm rounded the other way on a coin flip), either large =
            // structural (the arm genuinely prefers a different continuation). Also carry the
            // cross-arm distance of that row, which is the quantization error that did the damage.
            let ref_row = &reference.decide_rows[pos];
            let test_row = &test.decide_rows[pos];
            let row_maxdiff = ref_row
                .iter()
                .zip(test_row)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let quoted = format!(
                ",\"ref_logit_ref_id\":{:.6},\"ref_logit_test_id\":{:.6},\
                 \"test_logit_ref_id\":{:.6},\"test_logit_test_id\":{:.6},\
                 \"ref_margin\":{:.6},\"test_margin\":{:.6},\"div_row_maxdiff\":{:.6e}",
                ref_row[ref_id],
                ref_row[test_id],
                test_row[ref_id],
                test_row[test_id],
                ref_row[ref_id] - ref_row[test_id],
                test_row[test_id] - test_row[ref_id],
                row_maxdiff,
            );
            (
                format!(
                    "\"first_divergent_pos\":{pos},\"ref_token\":{ref_id},\"test_token\":{test_id},\
                     \"cross_prefill_maxdiff\":{cross_maxdiff:.6e},\
                     \"cross_prime_maxdiff\":{cross_prime_maxdiff:.6e}{quoted}"
                ),
                "DIVERGENT",
            )
        }
    };

    // The noise floor, per arm: where does an arm's OWN batched-primed stream leave its OWN
    // tokenwise-primed stream? Read the cross-arm verdict against this, not against zero.
    let floor = |r: &ArmRun| -> String {
        match r
            .tokens
            .iter()
            .zip(&r.tokens_tokenwise)
            .position(|(a, b)| a != b)
        {
            None => "null".to_string(),
            Some(p) => p.to_string(),
        }
    };

    let ref_text = tok.as_ref().map(|t| t.decode(&reference.tokens)).unwrap_or_default();
    let test_text = tok.as_ref().map(|t| t.decode(&test.tokens)).unwrap_or_default();

    println!(
        "JSONL {{\"cell\":\"{cell}\",\"model\":\"{model_name}\",\"prompt_tokens\":{},\"ngen\":{ngen},\
         \"ref_prefill_argmax\":{},\"ref_decode_argmax\":{},\"ref_self_maxdiff\":{:.6e},\
         \"ref_prime_argmax\":{},\
         \"test_prefill_argmax\":{},\"test_decode_argmax\":{},\"test_self_maxdiff\":{:.6e},\
         \"test_prime_argmax\":{},\
         \"ref_entrypoint_floor_pos\":{},\"test_entrypoint_floor_pos\":{},\
         {div_json},\"verdict\":\"{verdict}\",\
         \"ref_tokens\":{:?},\"test_tokens\":{:?},\
         \"ref_tokens_tokenwise\":{:?},\"test_tokens_tokenwise\":{:?},\
         \"ref_text\":\"{}\",\"test_text\":\"{}\"}}",
        prompt.len(),
        reference.prefill_argmax,
        reference.decode_argmax,
        reference.self_maxdiff,
        reference.prime_argmax,
        test.prefill_argmax,
        test.decode_argmax,
        test.self_maxdiff,
        test.prime_argmax,
        floor(&reference),
        floor(&test),
        reference.tokens,
        test.tokens,
        reference.tokens_tokenwise,
        test.tokens_tokenwise,
        json_escape(&ref_text),
        json_escape(&test_text),
    );

    println!("VERDICT {cell}: {verdict} (cross-arm prefill maxdiff {cross_maxdiff:.3e})");
    if verdict == "DIVERGENT" {
        std::process::exit(1);
    }
    Ok(())
}
