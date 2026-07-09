//! M7: end-to-end greedy generation with KV cache. Serves a model: prompt tokens -> generated tokens.
//!
//! Two prompt paths (back-compat):
//!   1. raw token ids:  `run-gen <model.gguf> 9419 11 1814 0`   (validation-gate path)
//!   2. TEXT prompt:    `run-gen <model.gguf> --prompt "Hello, world!"`  (or env BW24_PROMPT)
//!      The text is tokenized with bw24-tokenizer, generated, then the output ids are
//!      DETOKENIZED back to text and printed. Set BW24_CHAT=1 to wrap the prompt in the
//!      model's chat template (single user turn + assistant generation prompt).

use bw24_engine::Engine;
use bw24_engine::forward::argmax;
use bw24_engine::hybrid::HybridModel;
use bw24_gguf::GgufFile;
use bw24_tokenizer::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: run-gen <model.gguf|hf_dir> [tok ids...] | --prompt \"text\"");
    let e = Engine::new(0)?;
    // DIRECTORY path = safetensors HF checkpoint (MiniMax-M3 first-load path) OR a bw24 repack
    // dir (Hy3 Q4_K transcode: manifest.json + tensors/ + experts/). GGUF stays the dense norm.
    if std::path::Path::new(&path).is_dir() {
        let dir = std::path::Path::new(&path);
        // Repack dirs carry only weights; tokenizer files live in the manifest's source_dir.
        let is_repack = dir.join("manifest.json").exists();
        let (src, tok_dir): (Box<dyn bw24_gguf::source::TensorSource>, std::path::PathBuf) =
            if is_repack {
                let rs = bw24_gguf::source::Hy3RepackSource::open(dir)?;
                let td = rs.source_dir()
                    .filter(|d| d.join("tokenizer.json").exists())
                    .unwrap_or(dir).to_path_buf();
                (Box::new(rs), td)
            } else {
                (Box::new(bw24_gguf::source::SafetensorsSource::open(dir)?), dir.to_path_buf())
            };
        let model = HybridModel::load_from_source(&e, src.as_ref())?;
        println!("loaded {:?} from {} ({} layers)", model.cfg.arch,
                 if is_repack { "bw24 repack dir" } else { "safetensors" }, model.cfg.n_layer);

        // --- prompt: TEXT path (--prompt / BW24_PROMPT_FILE / BW24_PROMPT, tokenizer from the
        //     HF dir's tokenizer.json) or raw u32 ids (back-compat, the validation-gate path) ---
        let args: Vec<String> = std::env::args().skip(2).collect();
        let prompt_text: Option<String> = args
            .iter()
            .position(|a| a == "--prompt")
            .and_then(|i| args.get(i + 1).cloned())
            .or_else(|| std::env::var("BW24_PROMPT_FILE").ok()
                .map(|f| std::fs::read_to_string(&f).expect("BW24_PROMPT_FILE unreadable")))
            .or_else(|| std::env::var("BW24_PROMPT").ok());
        let mut tokenizer: Option<Tokenizer> = None;
        let prompt: Vec<u32> = if let Some(text) = &prompt_text {
            let tok = Tokenizer::from_hf_dir(&tok_dir)
                .map_err(|err| format!("HF tokenizer init failed: {err}"))?;
            let to_encode = if std::env::var("BW24_CHAT").is_ok() {
                let rendered = tok.apply_chat_template(&[("user", text)], true);
                println!("chat-templated prompt:\n{rendered}");
                rendered
            } else {
                text.clone()
            };
            let ids = tok.encode(&to_encode, true);
            println!("prompt text: {text:?}");
            tokenizer = Some(tok);
            ids
        } else {
            args.iter().filter_map(|s| s.parse::<u32>().ok()).collect()
        };
        let prompt = if prompt.is_empty() { vec![55u32] } else { prompt };
        println!("prompt tokens: {prompt:?}");

        // BW24_PP_ONLY (ST arm): prefill-anatomy profiling mode (nsys) — warmup + BW24_PP_REPS
        // timed SERVING prefills (prime_cache, the same pass PRIME_NANOS measures in run-spec)
        // and exit. Mirrors the GGUF arm's PP_ONLY; skips the decode gate so the profile is pure
        // prefill. Fresh cache per rep (fresh-prompt prime, cache.pos==0 each time).
        if std::env::var("BW24_PP_ONLY").is_ok() {
            let reps: usize = std::env::var("BW24_PP_REPS").ok()
                .and_then(|s| s.parse().ok()).unwrap_or(3);
            let warmups: usize = std::env::var("BW24_PP_WARMUP").ok()
                .and_then(|s| s.parse().ok()).unwrap_or(1);
            for _ in 0..warmups {
                let mut c = bw24_engine::cache::Cache::new(&e, &model.cfg, prompt.len() + 64)?;
                let _ = model.prime_cache(&e, &prompt, &mut c)?;
            }
            e.stream().synchronize()?;
            let mut times = Vec::with_capacity(reps);
            for r in 0..reps {
                let mut c = bw24_engine::cache::Cache::new(&e, &model.cfg, prompt.len() + 64)?;
                let tp = std::time::Instant::now();
                let _ = model.prime_cache(&e, &prompt, &mut c)?;
                e.stream().synchronize()?;
                let dt = tp.elapsed().as_secs_f64();
                times.push(dt);
                println!("pp-only rep {r}: {:.4}s = {:.1} tok/s", dt, prompt.len() as f64 / dt);
            }
            let mut ts = times.clone();
            ts.sort_by(|a, b| a.total_cmp(b));
            let med = ts[ts.len() / 2];
            println!("pp-only MEDIAN: {} tok in {:.4}s = {:.1} tok/s (pp{}, {} reps)",
                     prompt.len(), med, prompt.len() as f64 / med, prompt.len(), reps);
            return Ok(());
        }
        // GATE REFERENCE = the batched VERIFY path (decode_step_t: quantized-KV attend, the same
        // dispatch class as the real serving prime). forward_last's fresh-f32-KV attention is NOT
        // the M3 serving path, and its KV-precision delta amplifies through the sigmoid router's
        // discontinuous top-k (expert flips) into false MISMATCHes (t2probe 2026-07-06: decode ==
        // verify EXACT all 60 layers; forward-vs-decode drifts 5e-2 -> >1 by L2 via routing flips).
        // n_new read up-front so the gate's decode cache is already sized for the generation
        // that follows (no tokenwise re-prime — an 80-layer spilled MoE pays minutes per pass).
        let n_new: usize = std::env::var("BW24_NGEN").ok()
            .and_then(|s| s.parse().ok()).unwrap_or(16);
        let max_ctx = prompt.len() + n_new.max(64) + 8;
        let mut vcache = bw24_engine::cache::Cache::new(&e, &model.cfg, max_ctx)?;
        let prefill = model.decode_step_t(&e, &prompt, 0, &mut vcache)?;
        let n_vocab = model.output.out_features();
        let prefill_last = &prefill[(prompt.len() - 1) * n_vocab..prompt.len() * n_vocab];
        let mut cache = bw24_engine::cache::Cache::new(&e, &model.cfg, max_ctx)?;
        let mut dec = Vec::new();
        for &t in &prompt { dec = model.decode_step(&e, t, &mut cache)?; }
        let (ap, ad) = (argmax(prefill_last), argmax(&dec));
        let md = prefill_last.iter().zip(&dec).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!("verify-prefill argmax={ap}  decode argmax={ad}  logit maxdiff={md:.3e}  {}",
                 if ap == ad { "MATCH" } else { "MISMATCH" });

        // --- TEXT path: greedy-generate BW24_NGEN tokens on the (already primed) decode cache
        //     and DETOKENIZE (mirrors the GGUF text path; raw-id runs keep the gate-only exit).
        if let Some(tok) = &tokenizer {
            let eos = tok.eos_id();
            let (mut gcache, mut logits) = (cache, dec);
            let mut out: Vec<u32> = Vec::new();
            e.stream().synchronize()?;
            let t0 = std::time::Instant::now();
            for _ in 0..n_new {
                let next = argmax(&logits) as u32;
                out.push(next);
                if next == eos { break; }
                logits = model.decode_step(&e, next, &mut gcache)?;
            }
            e.stream().synchronize()?;
            let dt = t0.elapsed().as_secs_f64();
            println!("generated {} tokens in {dt:.3}s = {:.2} tok/s (ST greedy decode)",
                     out.len(), out.len() as f64 / dt);
            println!("tokens: {out:?}");
            // MoE residency-cache report (hit-rate + PCIe) — the spill-tier health signal.
            if let Some((hits, misses, staged, n_slots)) = e.moe_cache_stats() {
                let total = hits + misses;
                println!("MoE cache: {n_slots} slots | cumulative hits={hits} misses={misses} \
                          (hit-rate={:.1}%) | staged {:.2} GB H2D",
                         if total > 0 { hits as f64 / total as f64 * 100.0 } else { 0.0 },
                         staged as f64 / 1e9);
            }
            let text_ids: Vec<u32> = out.iter().copied().filter(|&id| id != eos).collect();
            let text = tok.decode(&text_ids);
            println!("OUTPUT TEXT: {text:?}");
            println!("--- generated text ---\n{text}");
        }
        return Ok(());
    }
    let g = GgufFile::open(&path)?;
    let model = HybridModel::load(&e, &g)?;
    println!("loaded {} ({} layers)", g.arch().unwrap_or("?"), model.cfg.n_layer);

    // --- resolve the prompt: TEXT path (--prompt / BW24_PROMPT) vs raw-u32 path ---
    let args: Vec<String> = std::env::args().skip(2).collect();
    let arg_prompt: Option<String> = args
        .iter()
        .position(|a| a == "--prompt")
        .and_then(|i| args.get(i + 1).cloned());
    let prompt_text: Option<String> = arg_prompt
        .or_else(|| std::env::var("BW24_PROMPT_FILE").ok()
            .map(|f| std::fs::read_to_string(&f).expect("BW24_PROMPT_FILE unreadable")))
        .or_else(|| std::env::var("BW24_PROMPT").ok());

    // Lazily build the tokenizer only when we need text I/O (it parses the 248K vocab).
    let mut tokenizer: Option<Tokenizer> = None;

    let prompt: Vec<u32> = if let Some(text) = &prompt_text {
        let tok = Tokenizer::from_gguf(&g)
            .map_err(|err| format!("tokenizer init failed: {err}"))?;
        // Optional chat-template wrapping (single user turn).
        let to_encode = if std::env::var("BW24_CHAT").is_ok() {
            let rendered = tok.apply_chat_template(&[("user", text)], true);
            println!("chat-templated prompt:\n{rendered}");
            rendered
        } else {
            text.clone()
        };
        let ids = tok.encode(&to_encode, true);
        println!("prompt text: {text:?}");
        tokenizer = Some(tok);
        ids
    } else {
        // raw u32 ids off the CLI (skip the "--prompt"/value tokens if present)
        args.iter().filter_map(|s| s.parse::<u32>().ok()).collect()
    };
    let prompt = if prompt.is_empty() { vec![55u32] } else { prompt };
    println!("prompt tokens: {prompt:?}");

    // BW24_PP_ONLY: prefill-anatomy profiling mode (nsys) — run warmup + BW24_PP_REPS timed
    // prefill forwards and exit. Skips the decode gate + generation so the profile is PURE prefill.
    if std::env::var("BW24_PP_ONLY").is_ok() {
        let reps: usize = std::env::var("BW24_PP_REPS").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
        // Warmup count knob: the MoE SLRU ghost filter admits on the SECOND miss, so a capped
        // (spill-regime) cache needs >=2 warmup forwards to reach steady residency before timing.
        let warmups: usize = std::env::var("BW24_PP_WARMUP").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
        for _ in 0..warmups { let _ = model.forward_last(&e, &prompt)?; }
        e.stream().synchronize()?;
        if let Some((hits, misses, staged, n_slots)) = e.moe_cache_stats() {
            println!("pp-only MoE cache after {warmups} warmup(s): {n_slots} slots hits={hits} misses={misses} staged_bytes={staged}");
        }
        // Per-rep timing (median-friendly: one process load, N samples) + per-rep H2D bytes.
        let mut times = Vec::with_capacity(reps);
        for r in 0..reps {
            e.moe_cache_reset_counters();
            let tp = std::time::Instant::now();
            let _ = model.forward_last(&e, &prompt)?;
            e.stream().synchronize()?;
            let dt = tp.elapsed().as_secs_f64();
            times.push(dt);
            match e.moe_cache_stats() {
                Some((h, m, s, _)) => println!(
                    "pp-only rep {r}: {:.4}s = {:.1} tok/s | hits={h} misses={m} staged_bytes={s} ({:.2} GB H2D)",
                    dt, prompt.len() as f64 / dt, s as f64 / 1e9),
                None => println!("pp-only rep {r}: {:.4}s = {:.1} tok/s", dt, prompt.len() as f64 / dt),
            }
        }
        let mut ts = times.clone();
        ts.sort_by(|a, b| a.total_cmp(b));
        let med = ts[ts.len() / 2];
        println!("pp-only MEDIAN: {} tok in {:.4}s = {:.1} tok/s (pp{}, {} reps)",
                 prompt.len(), med, prompt.len() as f64 / med, prompt.len(), reps);
        return Ok(());
    }

    // --- correctness gate: decode-step prefix logits MUST match the prefill forward ---
    let prefill = model.forward_last(&e, &prompt)?;
    // decode the prompt step by step, capture last logits
    let mut cache = bw24_engine::cache::Cache::new(&e, &model.cfg, prompt.len() + 64)?;
    let mut dec_logits = Vec::new();
    for &t in &prompt { dec_logits = model.decode_step(&e, t, &mut cache)?; }
    let am_p = argmax(&prefill); let am_d = argmax(&dec_logits);
    let maxdiff = prefill.iter().zip(&dec_logits).map(|(a,b)| (a-b).abs()).fold(0.0, f32::max);
    println!("prefill argmax={am_p}  decode argmax={am_d}  logit maxdiff={maxdiff:.3e}  {}",
             if am_p == am_d { "MATCH" } else { "MISMATCH" });
    assert_eq!(am_p, am_d, "decode-step diverges from prefill — cache threading bug");

    // --- time PREFILL tok/s (batched forward over the whole prompt) for the pp comparison vs
    //     llama-bench pp512. 1 warmup discarded, then time one forward of the full prompt. ---
    if prompt.len() >= 8 {
        let _ = model.forward_last(&e, &prompt)?;   // warmup
        e.stream().synchronize()?;
        let tp = std::time::Instant::now();
        let _ = model.forward_last(&e, &prompt)?;
        e.stream().synchronize()?;
        let dtp = tp.elapsed().as_secs_f64();
        println!("prefill {} tok in {:.4}s = {:.1} tok/s (pp{})", prompt.len(), dtp, prompt.len() as f64 / dtp, prompt.len());
    }

    // --- generate + time decode tok/s (honest Stage-A baseline) ---
    let n_new = std::env::var("BW24_NGEN").ok().and_then(|s| s.parse().ok()).unwrap_or(16usize);
    let eos = tokenizer.as_ref().map(|t| t.eos_id());
    // Sampler config from env (defaults = greedy, the bit-exact reference). BW24_TEMP>0 enables
    // the full chain: BW24_TOP_K / BW24_TOP_P / BW24_MIN_P / BW24_PENALTY_REPEAT / BW24_SEED.
    let env_f = |k: &str, d: f32| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
    let env_u = |k: &str, d: usize| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
    let scfg = bw24_engine::sampler::SamplerConfig {
        temperature: env_f("BW24_TEMP", 0.0),
        top_k: env_u("BW24_TOP_K", 0),
        top_p: env_f("BW24_TOP_P", 1.0),
        min_p: env_f("BW24_MIN_P", 0.0),
        penalty_last_n: env_u("BW24_PENALTY_LAST_N", 0),
        penalty_repeat: env_f("BW24_PENALTY_REPEAT", 1.0),
        penalty_freq: env_f("BW24_PENALTY_FREQ", 0.0),
        penalty_present: env_f("BW24_PENALTY_PRESENT", 0.0),
        seed: std::env::var("BW24_SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(0),
    };
    let mut sampler = bw24_engine::sampler::Sampler::new(scfg);
    // Stop conditions: EOS (text path) + optional stop-strings (BW24_STOP="a,b").
    let eos_ids: Vec<u32> = eos.into_iter().collect();
    let stop_strs: Vec<String> = std::env::var("BW24_STOP").ok()
        .map(|s| s.split(',').map(|x| x.to_string()).filter(|x| !x.is_empty()).collect())
        .unwrap_or_default();
    let params = bw24_engine::decode::GenParams {
        max_new: n_new, max_ctx: Some(prompt.len() + n_new + 8), eos: eos_ids,
    };
    // The reusable serving API (BASE-3). Stop-string match runs on the detokenized tail in the
    // per-token callback. Streaming hook: callback returns false to halt.
    let mut emitted_ids: Vec<u32> = Vec::new();
    let tok_ref = tokenizer.as_ref();
    e.stream().synchronize()?;
    let t0 = std::time::Instant::now();
    let gen_out = model.generate_with(&e, &prompt, &params, &mut sampler, |id| {
        emitted_ids.push(id);
        // stop-string check on the detokenized tail (text path only).
        if let (Some(tok), false) = (tok_ref, stop_strs.is_empty()) {
            let tail = tok.decode(&emitted_ids);
            if stop_strs.iter().any(|s| tail.contains(s.as_str())) { return false; }
        }
        true
    })?;
    e.stream().synchronize()?;
    let dt_total = t0.elapsed().as_secs_f64();
    // GEN-ONLY timing (2026-07-06 fix): generate_with primes INSIDE the timed span — at long
    // prompts the old number was prime-inclusive (35B @256-tok prime read 33.7 when decode was
    // ~51). PRIME_NANOS is the engine's published prime wall (same contract as run-spec).
    let prime_s = bw24_engine::PRIME_NANOS.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9;
    let dt = (dt_total - prime_s).max(1e-9);
    let out = gen_out.tokens;
    let emitted = out.len();
    let path = if std::env::var("BW24_FAST").as_deref() != Ok("0") { "Stage-B int8 dp4a" } else { "Stage-A f32-dequant" };
    println!("generated {} tokens in {:.3}s = {:.2} tok/s ({path} decode, gen-only; prime {:.3}s) [stop: {:?}]",
             emitted, dt, emitted as f64 / dt, prime_s, gen_out.stop_reason);
    println!("tokens: {out:?}");

    // --- EDGE-1 §D.4: MoE residency-cache PCIe report. The Stage-1 (no-cache) baseline re-stages
    //     every routed block every layer every token = `stage1_h2d_per_token()` (~907 MB/decode-token
    //     for the 35B-A3B over 40 layers). The cache drives that toward the one-time hot-set fill;
    //     after warmup the per-decode-token H2D should be a fraction of that. ---
    if let Some((hits, misses, _staged, n_slots)) = e.moe_cache_stats() {
        let total = hits + misses;
        let base_mb = model.stage1_h2d_per_token() as f64 / (1024.0 * 1024.0);
        println!("MoE cache: {n_slots} slots | cumulative hits={hits} misses={misses} (hit-rate={:.1}%) | \
                  Stage-1 baseline = {:.0} MB/decode-token (every block, every layer, every token)",
                 if total > 0 { hits as f64 / total as f64 * 100.0 } else { 0.0 }, base_mb);

        // Steady-state window: keep the WARM residency cache, re-build only the (dropped) KV cache by
        // re-priming, then reset the byte/hit counters and run BW24_NMEASURE more greedy decode tokens.
        // This isolates the post-warmup per-token H2D — the hot set is resident so PCIe -> a fraction.
        let n_measure: usize = std::env::var("BW24_NMEASURE").ok().and_then(|s| s.parse().ok()).unwrap_or(32);
        if n_measure > 0 {
            let mut warm_cache = bw24_engine::cache::Cache::new(&e, &model.cfg, prompt.len() + n_new + n_measure + 8)?;
            let mut ll = Vec::new();
            for &t in &prompt { ll = model.decode_step(&e, t, &mut warm_cache)?; }
            for &t in &out { ll = model.decode_step(&e, t, &mut warm_cache)?; }
            e.moe_cache_reset_counters();   // measure ONLY the steady-state window below
            for _ in 0..n_measure {
                let next = argmax(&ll) as u32;
                ll = model.decode_step(&e, next, &mut warm_cache)?;
            }
            if let Some((h2, m2, s2, _)) = e.moe_cache_stats() {
                let mb_tok = (s2 as f64 / (1024.0 * 1024.0)) / n_measure as f64;
                let tot2 = h2 + m2;
                println!("MoE cache STEADY-STATE ({n_measure} tokens after warmup): \
                          hit-rate={:.1}% | {:.1} MB/decode-token (vs {:.0} MB/token Stage-1 => {:.1}x less PCIe)",
                         if tot2 > 0 { h2 as f64 / tot2 as f64 * 100.0 } else { 0.0 },
                         mb_tok, base_mb, if mb_tok > 0.0 { base_mb / mb_tok } else { f64::INFINITY });
            }
        }
    }

    // --- detokenize the output ids back to TEXT (text path only) ---
    if let Some(tok) = &tokenizer {
        // drop a trailing EOS for the printed text (keep it in the raw `tokens:` line above).
        let text_ids: Vec<u32> = out
            .iter()
            .copied()
            .filter(|&id| Some(id) != eos)
            .collect();
        let text = tok.decode(&text_ids);
        println!("OUTPUT TEXT: {text:?}");
        println!("--- generated text ---\n{text}");
    }
    Ok(())
}
