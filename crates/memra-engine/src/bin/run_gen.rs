//! M7: end-to-end greedy generation with KV cache. Serves a model: prompt tokens -> generated tokens.
//!
//! Two prompt paths (back-compat):
//!   1. raw token ids:  `run-gen <model.gguf> 9419 11 1814 0`   (validation-gate path)
//!   2. TEXT prompt:    `run-gen <model.gguf> --prompt "Hello, world!"`  (or env MEMRA_PROMPT)
//!      The text is tokenized with memra-tokenizer, generated, then the output ids are
//!      DETOKENIZED back to text and printed. Set MEMRA_CHAT=1 to wrap the prompt in the
//!      model's chat template (single user turn + assistant generation prompt).

use memra_engine::forward::argmax;
use memra_engine::hybrid::HybridModel;
use memra_engine::Engine;
use memra_gguf::GgufFile;
use memra_tokenizer::Tokenizer;

type CpuExpertStatsSnapshot = (u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64);

#[derive(Debug, PartialEq, Eq)]
struct CpuExpertStatsDelta {
    calls: u64,
    experts: u64,
    wall_ns: u64,
    exposed_wait_ns: u64,
    ram_hits: u64,
    ram_misses: u64,
    ram_reads: u64,
    resident_bytes: u64,
    prepare_ns: u64,
    io_ns: u64,
    insert_ns: u64,
    compute_ns: u64,
}

fn cpu_expert_stats_delta(
    before: CpuExpertStatsSnapshot,
    after: CpuExpertStatsSnapshot,
    wait_before: u64,
    wait_after: u64,
) -> CpuExpertStatsDelta {
    CpuExpertStatsDelta {
        calls: after.0.saturating_sub(before.0),
        experts: after.1.saturating_sub(before.1),
        wall_ns: after.2.saturating_sub(before.2),
        exposed_wait_ns: wait_after.saturating_sub(wait_before),
        ram_hits: after.3.saturating_sub(before.3),
        ram_misses: after.4.saturating_sub(before.4),
        ram_reads: after.5.saturating_sub(before.5),
        resident_bytes: after.6,
        prepare_ns: after.7.saturating_sub(before.7),
        io_ns: after.8.saturating_sub(before.8),
        insert_ns: after.9.saturating_sub(before.9),
        compute_ns: after.10.saturating_sub(before.10),
    }
}

fn process_read_bytes() -> Option<u64> {
    std::fs::read_to_string("/proc/self/io")
        .ok()?
        .lines()
        .find_map(|line| line.strip_prefix("read_bytes:")?.trim().parse().ok())
}

fn forced_decode_tokens() -> Result<Option<Vec<u32>>, Box<dyn std::error::Error>> {
    let Some(path) = std::env::var_os("MEMRA_FORCE_TOKENS_FILE") else {
        return Ok(None);
    };
    let path = std::path::PathBuf::from(path);
    let raw = std::fs::read_to_string(&path)?;
    let tokens: Vec<u32> = raw
        .split(|character: char| !character.is_ascii_digit())
        .filter(|field| !field.is_empty())
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    if tokens.is_empty() {
        return Err(format!("{} contains no token ids", path.display()).into());
    }
    println!(
        "teacher-forced decode: {} tokens from {}",
        tokens.len(),
        path.display()
    );
    Ok(Some(tokens))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: run-gen <model.gguf|hf_dir|hf:owner/repo[:file]> [tok ids...] | --prompt \"text\"");
    let path = memra_gguf::hf::resolve_arg(&path)?;
    let e = Engine::new(0)?;
    // DIRECTORY path = safetensors HF checkpoint (MiniMax-M3 first-load path) OR a memra repack
    // dir (Hy3 Q4_K transcode: manifest.json + tensors/ + experts/). GGUF stays the dense norm.
    if std::path::Path::new(&path).is_dir() {
        let dir = std::path::Path::new(&path);
        // Repack dirs carry only weights; tokenizer files live in the manifest's source_dir.
        let is_repack = dir.join("manifest.json").exists();
        let (src, tok_dir): (Box<dyn memra_gguf::source::TensorSource>, std::path::PathBuf) =
            if is_repack {
                let rs = memra_gguf::source::Hy3RepackSource::open(dir)?;
                let td = rs
                    .source_dir()
                    .filter(|d| d.join("tokenizer.json").exists())
                    .unwrap_or(dir)
                    .to_path_buf();
                (Box::new(rs), td)
            } else {
                (
                    Box::new(memra_gguf::source::SafetensorsSource::open(dir)?),
                    dir.to_path_buf(),
                )
            };
        let model = HybridModel::load_from_source_without_mtp(&e, src.as_ref())?;
        println!(
            "loaded {:?} from {} ({} trunk layers; optional MTP skipped)",
            model.cfg.arch,
            if is_repack {
                "memra repack dir"
            } else {
                "safetensors"
            },
            model.layers.len()
        );

        // --- prompt: TEXT path (--prompt / MEMRA_PROMPT_FILE / MEMRA_PROMPT, tokenizer from the
        //     HF dir's tokenizer.json) or raw u32 ids (back-compat, the validation-gate path) ---
        let args: Vec<String> = std::env::args().skip(2).collect();
        let prompt_text: Option<String> = args
            .iter()
            .position(|a| a == "--prompt")
            .and_then(|i| args.get(i + 1).cloned())
            .or_else(|| {
                std::env::var("MEMRA_PROMPT_FILE")
                    .ok()
                    .map(|f| std::fs::read_to_string(&f).expect("MEMRA_PROMPT_FILE unreadable"))
            })
            .or_else(|| std::env::var("MEMRA_PROMPT").ok());
        let mut tokenizer: Option<Tokenizer> = None;
        let prompt: Vec<u32> = if let Some(text) = &prompt_text {
            let tok = Tokenizer::from_hf_dir(&tok_dir)
                .map_err(|err| format!("HF tokenizer init failed: {err}"))?;
            let to_encode = if std::env::var("MEMRA_CHAT").is_ok() {
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
        let prompt = if prompt.is_empty() {
            vec![55u32]
        } else {
            prompt
        };
        println!("prompt tokens: {prompt:?}");

        // MEMRA_PP_ONLY (ST arm): prefill-anatomy profiling mode (nsys) — warmup + MEMRA_PP_REPS
        // timed SERVING prefills (prime_cache, the same pass PRIME_NANOS measures in run-spec)
        // and exit. Mirrors the GGUF arm's PP_ONLY; skips the decode gate so the profile is pure
        // prefill. Fresh cache per rep (fresh-prompt prime, cache.pos==0 each time).
        if std::env::var("MEMRA_PP_ONLY").is_ok() {
            let reps: usize = std::env::var("MEMRA_PP_REPS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(3);
            let warmups: usize = std::env::var("MEMRA_PP_WARMUP")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1);
            for _ in 0..warmups {
                let mut c = memra_engine::cache::Cache::new(&e, &model.cfg, prompt.len() + 64)?;
                let _ = model.prime_cache(&e, &prompt, &mut c)?;
            }
            e.stream().synchronize()?;
            let mut times = Vec::with_capacity(reps);
            for r in 0..reps {
                let mut c = memra_engine::cache::Cache::new(&e, &model.cfg, prompt.len() + 64)?;
                let tp = std::time::Instant::now();
                let _ = model.prime_cache(&e, &prompt, &mut c)?;
                e.stream().synchronize()?;
                let dt = tp.elapsed().as_secs_f64();
                times.push(dt);
                println!(
                    "pp-only rep {r}: {:.4}s = {:.1} tok/s",
                    dt,
                    prompt.len() as f64 / dt
                );
            }
            let mut ts = times.clone();
            ts.sort_by(|a, b| a.total_cmp(b));
            let med = ts[ts.len() / 2];
            println!(
                "pp-only MEDIAN: {} tok in {:.4}s = {:.1} tok/s (pp{}, {} reps)",
                prompt.len(),
                med,
                prompt.len() as f64 / med,
                prompt.len(),
                reps
            );
            // PREFILL-PATH EXACTNESS (lane/fp8-blk128-decode, 2026-08-05). WHY IT LIVES *HERE* and
            // not next to the verify-prefill gate 100 lines below: those are two different prefill
            // dispatch classes, and only this one can reach a prefill GEMM kernel at all.
            //   * `prime_cache` (this arm, and what `generate`/`generate_spec`/serving actually
            //     prime with) runs its projections through `matmul` / `matmul_group` -> `matmul`,
            //     which carries the m>=16 GEMM/MMQ hooks (try_fp8_gemm, try_fp8_blk_mmq,
            //     try_f16_gemm) — measured 1984 hook entries, 832 dispatches on the 27B.
            //   * `decode_step_t` (the verify-prefill gate) runs them through
            //     `matmul_decode_exact`, which by DESIGN has no GEMM/MMQ arm whatsoever: its whole
            //     contract is that every token row take the exact m=1 MMVQ program (the
            //     decode-parity law). So `fp8-mmq dispatches after prefill: 0` on that gate is
            //     CORRECT BEHAVIOR, not a wiring bug — and any exactness number taken from a full
            //     `run-gen` run is measuring the fallback arm, whatever flag was set.
            // Hence: a prefill-GEMM arm's exactness has to be measured on the prime path.
            //
            // MEMRA_PP_LOGITS=<file>: prime_cache's last-row logits as raw LE f32 — the cross-arm
            // drift vector (max_abs / rms_rel / top-k order), the same instrument
            // MEMRA_PREFILL_LOGITS is for the verify path.
            //
            // MEMRA_PP_NLL=1: TEACHER-FORCED prefill quality with NO reference-tape asymmetry.
            // The prompt IS the tape: position i's logits score the prompt's own token i+1, so
            // both arms are scored on the identical externally-given sequence and neither can win
            // by reproducing itself (the decode battery needed a reverse-tape control precisely
            // because its tape was one arm's own output; this quantity needs none by construction).
            // Reports argmax disagreement vs the prompt continuation + mean NLL over 0..T-2.
            // The [T, n_embd] pre-output_norm stack `prime_cache` returns is the trunk's whole
            // output, so it carries every block-128 projection's contribution; the norm+head
            // applied to it here is the SAME dispatch in both arms, so the comparison is fair even
            // though it is not prime's own m=1 head.
            {
                let mut c = memra_engine::cache::Cache::new(&e, &model.cfg, prompt.len() + 64)?;
                let (last, _h_seed, hiddens) = model.prime_cache(&e, &prompt, &mut c)?;
                if let Ok(f) = std::env::var("MEMRA_PP_LOGITS") {
                    let mut raw = Vec::with_capacity(last.len() * 4);
                    for v in &last {
                        raw.extend_from_slice(&v.to_le_bytes());
                    }
                    std::fs::write(&f, &raw)?;
                    println!("pp-only prime logits -> {f} ({} f32)", last.len());
                }
                if std::env::var("MEMRA_PP_NLL").is_ok_and(|v| v != "0") {
                    let n_embd = model.cfg.n_embd as usize;
                    let eps = model.cfg.rms_eps;
                    let t = prompt.len();
                    // 64-row chunks bound the logit allocation (64 * n_vocab * 4B) and keep every
                    // chunk past GEMM_M_THRESHOLD=16, so the head's dispatch class is one class
                    // for the whole sweep instead of changing on the tail.
                    const CH: usize = 64;
                    let (mut nll, mut disagree, mut positions) = (0.0f64, 0usize, 0usize);
                    let mut first_disagree: Option<usize> = None;
                    let mut start = 0usize;
                    while start + 1 < t {
                        let rows = CH.min(t - 1 - start);
                        let mut xs = e.uninit(rows * n_embd)?;
                        let src = hiddens.slice(start * n_embd..(start + rows) * n_embd);
                        e.copy_view_into(&mut xs, 0, &src, rows * n_embd)?;
                        let mut hn = e.uninit(rows * n_embd)?;
                        e.rms_norm(
                            &xs,
                            model.output_norm.float_data(),
                            &mut hn,
                            n_embd,
                            rows,
                            eps,
                        )?;
                        let lg = e.dtoh(&e.matmul(&model.output, &hn, rows)?)?;
                        let n_vocab = lg.len() / rows;
                        for r in 0..rows {
                            let row = &lg[r * n_vocab..(r + 1) * n_vocab];
                            let want = prompt[start + r + 1] as usize;
                            if argmax(row) != want {
                                disagree += 1;
                                if first_disagree.is_none() {
                                    first_disagree = Some(start + r);
                                }
                            }
                            let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                            let lse =
                                mx + row.iter().map(|l| (l - mx).exp()).sum::<f32>().ln();
                            nll += (lse - row[want]) as f64;
                            positions += 1;
                        }
                        start += rows;
                    }
                    println!(
                        "prefill-path EXACTNESS (prime_cache): disagreements={disagree}/{positions} \
                         ({:.2}%) first_at={}  mean_nll={:.6}  total_nll={:.4}",
                        100.0 * disagree as f64 / positions.max(1) as f64,
                        first_disagree.map_or("-".to_string(), |s| s.to_string()),
                        nll / positions.max(1) as f64,
                        nll,
                    );
                }
            }
            // Coverage receipt (lane/fp8-mmq): how many prefill GEMMs went through the per-block
            // FP8 MMQ tile. A refused precondition (no block operand resident, stash budget spent
            // before the tensor, a NaN code) reads exactly like "no perf change", so a pp number
            // for that arm is only evidence alongside a nonzero count.
            let (ent, gate, h, no_op, shp, scl, nan) = memra_engine::fp8_ffi::fp8_mmq_ledger();
            println!(
                "fp8-mmq dispatches: {h}  (hook entries={ent} gate_off={gate} \
                 no_operand={no_op} bad_shape={shp} bad_scale={scl} nan={nan})"
            );
            return Ok(());
        }
        // GATE REFERENCE = the batched VERIFY path (decode_step_t: quantized-KV attend, the same
        // dispatch class as the real serving prime). forward_last's fresh-f32-KV attention is NOT
        // the M3 serving path, and its KV-precision delta amplifies through the sigmoid router's
        // discontinuous top-k (expert flips) into false MISMATCHes (t2probe 2026-07-06: decode ==
        // verify EXACT all 60 layers; forward-vs-decode drifts 5e-2 -> >1 by L2 via routing flips).
        // n_new read up-front so the gate's decode cache is already sized for the generation
        // that follows (no tokenwise re-prime — an 80-layer spilled MoE pays minutes per pass).
        let n_new: usize = std::env::var("MEMRA_NGEN")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(16);
        let max_ctx = prompt.len() + n_new.max(64) + 8;

        // A heterogeneous CPU/GPU expert split needs one immutable backend assignment for exact
        // repeatability. Optionally learn a decode-hot assignment from discarded tokens, freeze it,
        // then run both gate paths under that fixed assignment before any measured output. Skip the
        // non-authoritative pre-freeze gate so warmup is the only input to residency selection.
        let freeze_warmup_tokens = std::env::var("MEMRA_CPU_EXPERT_FREEZE_WARMUP_TOKENS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0);
        // MEMRA_CPU_EXPERT_FREEZE_PROFILE: restage a saved residency set and skip the warmup
        // entirely (the warmup streams ~200 GB through the spill path; a restage reads only
        // the chosen blocks). A run that still warms up writes the profile for the next one.
        let freeze_profile = std::env::var("MEMRA_CPU_EXPERT_FREEZE_PROFILE")
            .ok()
            .filter(|value| !value.is_empty())
            .map(std::path::PathBuf::from);
        let gate_label = if freeze_warmup_tokens > 0 {
            let restored = match &freeze_profile {
                Some(path) => model.restore_cpu_expert_residency_profile(&e, path)?,
                None => false,
            };
            if restored {
                e.stream().synchronize()?;
                "post-freeze verify-prefill"
            } else {
                println!(
                    "[moe-cache] warming {freeze_warmup_tokens} discarded decode tokens before fixed residency"
                );
                let _ = model.generate(&e, &prompt, freeze_warmup_tokens + 1)?;
                e.stream().synchronize()?;
                model.freeze_cpu_expert_residency(&e)?;
                if let Some(path) = &freeze_profile {
                    model.save_cpu_expert_residency_profile(&e, path)?;
                }
                "post-freeze verify-prefill"
            }
        } else {
            "verify-prefill"
        };
        if freeze_warmup_tokens > 0
            && std::env::var("MEMRA_MOE_PREFETCH").is_ok_and(|value| value != "0")
        {
            model.start_moe_prefetch_predictor(&e, &model.cfg)?;
        }

        // Scope the batched reference cache so only one max-context GPU KV allocation is live at
        // a time. The serving cache below is the one retained for measured generation.
        let n_vocab = model.output.out_features();
        let prefill_last = {
            let mut vcache = memra_engine::cache::Cache::new(&e, &model.cfg, max_ctx)?;
            let prefill = model.decode_step_t(&e, &prompt, 0, &mut vcache)?;
            prefill[(prompt.len() - 1) * n_vocab..prompt.len() * n_vocab].to_vec()
        };
        // MEMRA_PREFILL_LOGITS=<file>: dump this batched-prefill logit row as raw LE f32. The
        // gate line below compares prefill vs THIS RUN's own decode, so it cannot see a
        // cross-ARM difference; this dump is the cross-arm instrument for a kernel that changes
        // only the VERIFY-class prefill, and the 128-token stream that follows is pure m=1 decode.
        //
        // NOT AN INSTRUMENT FOR A PREFILL *GEMM* ARM (correction, lane/fp8-blk128-decode
        // 2026-08-05 — this comment previously claimed it was "the only cross-arm exactness
        // instrument" for lane/fp8-mmq, and that is wrong): `decode_step_t` dispatches through
        // `matmul_decode_exact`, which has NO GEMM/MMQ arm by design (decode-parity law: every
        // token row takes the exact m=1 MMVQ program). try_fp8_gemm / try_fp8_blk_mmq /
        // try_f16_gemm live only on `matmul` / `matmul_pre`, so no prefill-GEMM kernel can run
        // here no matter what flag is set, and this vector is IDENTICAL across such arms —
        // silently, which reads exactly like "bit-identical". The ledger line below is what makes
        // that visible: expect `hook entries=0` here on the 27B ST class. The instrument for a
        // prefill-GEMM arm is MEMRA_PP_ONLY + MEMRA_PP_LOGITS / MEMRA_PP_NLL above, which measures
        // `prime_cache` — the class that actually carries the hooks, and the one serving primes on.
        if let Ok(f) = std::env::var("MEMRA_PREFILL_LOGITS") {
            let mut raw = Vec::with_capacity(prefill_last.len() * 4);
            for v in &prefill_last {
                raw.extend_from_slice(&v.to_le_bytes());
            }
            std::fs::write(&f, &raw)?;
            println!("prefill logits -> {f} ({} f32)", prefill_last.len());
        }
        {
            let (ent, gate, h, no_op, shp, scl, nan) = memra_engine::fp8_ffi::fp8_mmq_ledger();
            println!(
                "fp8-mmq dispatches after prefill: {h}  (hook entries={ent} gate_off={gate} \
                 no_operand={no_op} bad_shape={shp} bad_scale={scl} nan={nan})"
            );
        }
        // MEMRA_RESIDENCY_CENSUS=1 (lane/fp8-decode-v1): which container each 2D matmul weight
        // ACTUALLY went resident in, and its bytes. The FP8-ST decode arm is a residency change,
        // so this is its primary evidence — tok/s alone cannot separate "arm ran, flat" from
        // "arm never engaged on this checkpoint".
        if std::env::var("MEMRA_RESIDENCY_CENSUS").is_ok_and(|v| v != "0") {
            println!("{}", memra_engine::model::residency_census_report());
        }
        let mut cache = memra_engine::cache::Cache::new(&e, &model.cfg, max_ctx)?;
        let mut dec = Vec::new();
        for &token in &prompt {
            dec = model.decode_step(&e, token, &mut cache)?;
        }
        let (ap, ad) = (argmax(&prefill_last), argmax(&dec));
        let md = prefill_last
            .iter()
            .zip(&dec)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let serving_gate_match = ap == ad;
        println!(
            "{gate_label} argmax={ap}  decode argmax={ad}  logit maxdiff={md:.3e}  {}",
            if serving_gate_match { "MATCH" } else { "MISMATCH" }
        );
        if !serving_gate_match {
            return Err("prefill/decode argmax gate failed for serving expert assignment".into());
        }

        // --- TEXT path: greedy-generate MEMRA_NGEN tokens on the (already primed) decode cache
        //     and DETOKENIZE (mirrors the GGUF text path; raw-id runs keep the gate-only exit).
        if let Some(tok) = &tokenizer {
            let eos = tok.eos_id();
            let (mut gcache, mut logits) = (cache, dec);
            let mut out: Vec<u32> = Vec::new();
            let forced_tokens = forced_decode_tokens()?;
            if forced_tokens.as_ref().is_some_and(|tokens| tokens.len() < n_new) {
                return Err(format!(
                    "MEMRA_FORCE_TOKENS_FILE needs at least {n_new} ids for the decode window"
                )
                .into());
            }
            e.stream().synchronize()?;
            // The verify/prompt passes above warm residency. Reset only counters—not cache state—so
            // this timed decode window reports its own hit rate, H2D bytes, and worker-I/O deltas.
            e.moe_cache_reset_counters();
            let pread_before = e.moe_pread_stats();
            let cpu_before = e.cpu_expert_stats();
            let cpu_wait_before = e.cpu_expert_exposed_wait_ns();
            let cpu_residency_before = e.cpu_expert_gpu_residency_stats();
            let disk_before = process_read_bytes();
            let (mut tf_disagree, mut tf_positions) = (0usize, 0usize);
            let mut tf_first_disagree: Option<usize> = None;
            let mut tf_nll = 0.0f64;
            let t0 = std::time::Instant::now();
            for step in 0..n_new {
                // Keep the ordinary host argmax cost inside teacher-forced A/B windows; only the
                // token fed into the next decode step changes.
                let greedy = argmax(&logits) as u32;
                let next = forced_tokens
                    .as_ref()
                    .map(|tokens| tokens[step])
                    .unwrap_or(greedy);
                // TEACHER-FORCED DISAGREEMENTS + NLL (lane/fp8-decode-v1, 2026-08-05): under
                // forcing, both arms see BIT-IDENTICAL inputs at every position, so
                // `greedy != next` is this arm's own argmax disagreeing with the reference tape
                // at a position the reference actually visited. That is the branch-(b) quantity
                // for two containers whose arithmetic differs (e4m3 in-kernel dequant vs the
                // Q8_0 re-encode): bit-identity is the WRONG question, disagreement count and
                // the forced-tape NLL are the right ones. NLL = -log softmax(logits)[next],
                // summed over the window: the tape's own likelihood under this arm, so a lower
                // total is a strictly better model of the SAME token sequence.
                if forced_tokens.is_some() {
                    if greedy != next {
                        tf_disagree += 1;
                        if tf_first_disagree.is_none() { tf_first_disagree = Some(step); }
                    }
                    let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let lse = mx + logits.iter().map(|l| (l - mx).exp()).sum::<f32>().ln();
                    tf_nll += (lse - logits[next as usize]) as f64;
                    tf_positions += 1;
                }
                out.push(next);
                if next == eos {
                    break;
                }
                logits = model.decode_step(&e, next, &mut gcache)?;
            }
            e.stream().synchronize()?;
            let dt = t0.elapsed().as_secs_f64();
            println!(
                "generated {} tokens in {dt:.3}s = {:.2} tok/s (ST {} decode)",
                out.len(),
                out.len() as f64 / dt,
                if forced_tokens.is_some() { "teacher-forced" } else { "greedy" },
            );
            if tf_positions > 0 {
                println!(
                    "teacher-forced EXACTNESS: disagreements={tf_disagree}/{tf_positions} \
                     ({:.2}%) first_at={}  mean_nll={:.6}  total_nll={:.4}",
                    100.0 * tf_disagree as f64 / tf_positions as f64,
                    tf_first_disagree.map_or("-".to_string(), |s| s.to_string()),
                    tf_nll / tf_positions as f64,
                    tf_nll,
                );
            }
            println!("tokens: {out:?}");
            // MoE residency-cache report (hit-rate + PCIe) — this decode window only.
            if let Some((hits, misses, staged, n_slots)) = e.moe_cache_stats() {
                let total = hits + misses;
                let mb_tok = staged as f64 / (1024.0 * 1024.0) / out.len().max(1) as f64;
                println!(
                    "MoE cache DECODE-WINDOW: {n_slots} slots | hits={hits} misses={misses} \
                          (hit-rate={:.1}%) | staged {:.2} GB H2D ({mb_tok:.1} MB/token)",
                    if total > 0 {
                        hits as f64 / total as f64 * 100.0
                    } else {
                        0.0
                    },
                    staged as f64 / 1e9
                );
            }
            if let (Some(before), Some(after)) = (pread_before, e.moe_pread_stats()) {
                println!(
                    "spill worker DECODE-WINDOW: reads={} bytes={} waits={} ring_full={} fallbacks={}",
                    after.0.saturating_sub(before.0),
                    after.1.saturating_sub(before.1),
                    after.5.saturating_sub(before.5),
                    after.6.saturating_sub(before.6),
                    after.4.saturating_sub(before.4),
                );
            }
            if let (Some(before), Some(after), Some(wait_before), Some(wait_after)) = (
                cpu_before,
                e.cpu_expert_stats(),
                cpu_wait_before,
                e.cpu_expert_exposed_wait_ns(),
            ) {
                let stats = cpu_expert_stats_delta(before, after, wait_before, wait_after);
                println!(
                    "CPU experts DECODE-WINDOW: calls={} experts={} \
                     backend_wall={:.3}s exposed_wait={:.3}s RAM_hits={} RAM_misses={} \
                     RAM_fills={:.2} GB RAM_resident={:.2} GB \
                     phase_prepare={:.3}s phase_io={:.3}s phase_insert={:.3}s phase_compute={:.3}s",
                    stats.calls,
                    stats.experts,
                    stats.wall_ns as f64 / 1e9,
                    stats.exposed_wait_ns as f64 / 1e9,
                    stats.ram_hits,
                    stats.ram_misses,
                    stats.ram_reads as f64 / 1e9,
                    stats.resident_bytes as f64 / 1e9,
                    stats.prepare_ns as f64 / 1e9,
                    stats.io_ns as f64 / 1e9,
                    stats.insert_ns as f64 / 1e9,
                    stats.compute_ns as f64 / 1e9,
                );
                let (predictor_submitted, predictor_dropped) =
                    e.cpu_expert_predictor_stats();
                if predictor_submitted > 0 || predictor_dropped > 0 {
                    println!(
                        "MoE prefetch predictor: submitted={predictor_submitted} dropped={predictor_dropped}"
                    );
                }
            }
            if let (Some(before), Some(after)) =
                (cpu_residency_before, e.cpu_expert_gpu_residency_stats())
            {
                println!(
                    "CPU expert HBM fragments DECODE-WINDOW: resident_0={} resident_1={} resident_2={}",
                    after.0.saturating_sub(before.0),
                    after.1.saturating_sub(before.1),
                    after.2.saturating_sub(before.2),
                );
            }
            if let (Some(before), Some(after)) = (disk_before, process_read_bytes()) {
                println!(
                    "storage DECODE-WINDOW: {:.2} GB physical reads",
                    after.saturating_sub(before) as f64 / 1e9
                );
            }

            // Optional repeatable steady-state benchmark. Rebuild only KV state by replaying the
            // same prompt and generated prefix; the process-wide GPU and CPU expert caches stay
            // warm. Every repetition therefore measures the same continuation, not progressively
            // different text. Defaults off so ordinary generation does no extra work.
            let n_measure: usize = std::env::var("MEMRA_NMEASURE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            if n_measure > 0 {
                if forced_tokens
                    .as_ref()
                    .is_some_and(|tokens| tokens.len() < out.len() + n_measure)
                {
                    return Err(format!(
                        "MEMRA_FORCE_TOKENS_FILE needs at least {} ids for decode + measurement",
                        out.len() + n_measure
                    )
                    .into());
                }
                if out.contains(&eos) {
                    println!("steady-state benchmark skipped: initial generation reached EOS");
                } else {
                    let measure_reps: usize = std::env::var("MEMRA_NMEASURE_REPS")
                        .ok()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1)
                        .max(1);
                    let mut reference: Option<Vec<u32>> = None;
                    let mut rates = Vec::with_capacity(measure_reps);

                    for rep in 0..measure_reps {
                        let mut warm_cache = memra_engine::cache::Cache::new(
                            &e,
                            &model.cfg,
                            prompt.len() + out.len() + n_measure + 8,
                        )?;
                        let mut warm_logits = Vec::new();
                        for &token in &prompt {
                            warm_logits = model.decode_step(&e, token, &mut warm_cache)?;
                        }
                        for &token in &out {
                            warm_logits = model.decode_step(&e, token, &mut warm_cache)?;
                        }
                        e.stream().synchronize()?;

                        e.moe_cache_reset_counters();
                        let warm_pread_before = e.moe_pread_stats();
                        let warm_cpu_before = e.cpu_expert_stats();
                        let warm_cpu_wait_before = e.cpu_expert_exposed_wait_ns();
                        let warm_residency_before = e.cpu_expert_gpu_residency_stats();
                        let warm_disk_before = process_read_bytes();
                        let mut measured = Vec::with_capacity(n_measure);
                        let warm_t0 = std::time::Instant::now();
                        for step in 0..n_measure {
                            let greedy = argmax(&warm_logits) as u32;
                            let next = forced_tokens
                                .as_ref()
                                .map(|tokens| tokens[out.len() + step])
                                .unwrap_or(greedy);
                            measured.push(next);
                            if next == eos {
                                break;
                            }
                            warm_logits = model.decode_step(&e, next, &mut warm_cache)?;
                        }
                        e.stream().synchronize()?;
                        let warm_dt = warm_t0.elapsed().as_secs_f64();

                        if let Some(expected) = &reference {
                            if measured != *expected {
                                return Err(format!(
                                    "steady-state repetition {rep} changed token sequence: \
                                     expected {expected:?}, got {measured:?}"
                                )
                                .into());
                            }
                        } else {
                            reference = Some(measured.clone());
                        }

                        let rate = measured.len() as f64 / warm_dt;
                        rates.push(rate);
                        println!(
                            "steady-state rep {rep}: generated {} tokens in {warm_dt:.3}s = \
                             {rate:.2} tok/s (same-prefix warm-cache greedy decode)",
                            measured.len()
                        );
                        println!("steady-state tokens: {measured:?}");

                        if let Some((hits, misses, staged, n_slots)) = e.moe_cache_stats() {
                            let total = hits + misses;
                            let mb_tok =
                                staged as f64 / (1024.0 * 1024.0) / measured.len().max(1) as f64;
                            println!(
                                "MoE cache STEADY-STATE rep {rep}: {n_slots} slots | \
                                 hits={hits} misses={misses} (hit-rate={:.1}%) | \
                                 staged {:.2} GB H2D ({mb_tok:.1} MB/token)",
                                if total > 0 {
                                    hits as f64 / total as f64 * 100.0
                                } else {
                                    0.0
                                },
                                staged as f64 / 1e9
                            );
                        }
                        if let (Some(before), Some(after)) =
                            (warm_pread_before, e.moe_pread_stats())
                        {
                            println!(
                                "spill worker STEADY-STATE rep {rep}: reads={} bytes={} waits={} \
                                 ring_full={} fallbacks={}",
                                after.0.saturating_sub(before.0),
                                after.1.saturating_sub(before.1),
                                after.5.saturating_sub(before.5),
                                after.6.saturating_sub(before.6),
                                after.4.saturating_sub(before.4),
                            );
                        }
                        if let (Some(before), Some(after), Some(wait_before), Some(wait_after)) = (
                            warm_cpu_before,
                            e.cpu_expert_stats(),
                            warm_cpu_wait_before,
                            e.cpu_expert_exposed_wait_ns(),
                        ) {
                            let stats =
                                cpu_expert_stats_delta(before, after, wait_before, wait_after);
                            println!(
                                "CPU experts STEADY-STATE rep {rep}: calls={} experts={} \
                                 backend_wall={:.3}s exposed_wait={:.3}s RAM_hits={} RAM_misses={} \
                                 RAM_fills={:.2} GB RAM_resident={:.2} GB phase_prepare={:.3}s \
                                 phase_io={:.3}s phase_insert={:.3}s phase_compute={:.3}s",
                                stats.calls,
                                stats.experts,
                                stats.wall_ns as f64 / 1e9,
                                stats.exposed_wait_ns as f64 / 1e9,
                                stats.ram_hits,
                                stats.ram_misses,
                                stats.ram_reads as f64 / 1e9,
                                stats.resident_bytes as f64 / 1e9,
                                stats.prepare_ns as f64 / 1e9,
                                stats.io_ns as f64 / 1e9,
                                stats.insert_ns as f64 / 1e9,
                                stats.compute_ns as f64 / 1e9,
                            );
                        }
                        if let (Some(before), Some(after)) =
                            (warm_residency_before, e.cpu_expert_gpu_residency_stats())
                        {
                            println!(
                                "CPU expert HBM fragments STEADY-STATE rep {rep}: \
                                 resident_0={} resident_1={} resident_2={}",
                                after.0.saturating_sub(before.0),
                                after.1.saturating_sub(before.1),
                                after.2.saturating_sub(before.2),
                            );
                        }
                        if let (Some(before), Some(after)) =
                            (warm_disk_before, process_read_bytes())
                        {
                            println!(
                                "storage STEADY-STATE rep {rep}: {:.2} GB physical reads",
                                after.saturating_sub(before) as f64 / 1e9
                            );
                        }
                    }

                    rates.sort_by(|a, b| a.total_cmp(b));
                    println!(
                        "steady-state MEDIAN: {n_measure}-token same-prefix window, \
                         N={measure_reps}, {:.2} tok/s (warm HBM/RAM expert caches)",
                        rates[rates.len() / 2]
                    );
                }
            }

            let text_ids: Vec<u32> = out.iter().copied().filter(|&id| id != eos).collect();
            let text = tok.decode(&text_ids);
            println!("OUTPUT TEXT: {text:?}");
            println!("--- generated text ---\n{text}");
        }
        // Coverage receipt — see the PP_ONLY arm above. A greedy stream that matches the floor
        // because the kernel never dispatched is not an exactness result.
        println!("fp8-mmq dispatches: {}", memra_engine::fp8_ffi::fp8_mmq_hits());
        return Ok(());
    }
    let g = GgufFile::open(&path)?;
    let model = HybridModel::load_without_mtp(&e, &g)?;
    println!(
        "loaded {} ({} trunk layers; optional MTP skipped)",
        g.arch().unwrap_or("?"),
        model.layers.len()
    );

    // --- resolve the prompt: TEXT path (--prompt / MEMRA_PROMPT) vs raw-u32 path ---
    let args: Vec<String> = std::env::args().skip(2).collect();
    let arg_prompt: Option<String> = args
        .iter()
        .position(|a| a == "--prompt")
        .and_then(|i| args.get(i + 1).cloned());
    let prompt_text: Option<String> = arg_prompt
        .or_else(|| {
            std::env::var("MEMRA_PROMPT_FILE")
                .ok()
                .map(|f| std::fs::read_to_string(&f).expect("MEMRA_PROMPT_FILE unreadable"))
        })
        .or_else(|| std::env::var("MEMRA_PROMPT").ok());

    // Lazily build the tokenizer only when we need text I/O (it parses the 248K vocab).
    let mut tokenizer: Option<Tokenizer> = None;

    let prompt: Vec<u32> = if let Some(text) = &prompt_text {
        let tok =
            Tokenizer::from_gguf(&g).map_err(|err| format!("tokenizer init failed: {err}"))?;
        // Optional chat-template wrapping (single user turn).
        let to_encode = if std::env::var("MEMRA_CHAT").is_ok() {
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
    let prompt = if prompt.is_empty() {
        vec![55u32]
    } else {
        prompt
    };
    println!("prompt tokens: {prompt:?}");

    // MEMRA_PP_ONLY: prefill-anatomy profiling mode (nsys) — run warmup + MEMRA_PP_REPS timed
    // prefill forwards and exit. Skips the decode gate + generation so the profile is PURE prefill.
    if std::env::var("MEMRA_PP_ONLY").is_ok() {
        let reps: usize = std::env::var("MEMRA_PP_REPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3);
        // Warmup count knob: the MoE SLRU ghost filter admits on the SECOND miss, so a capped
        // (spill-regime) cache needs >=2 warmup forwards to reach steady residency before timing.
        let warmups: usize = std::env::var("MEMRA_PP_WARMUP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        for _ in 0..warmups {
            let _ = model.forward_last(&e, &prompt)?;
        }
        // MEMRA_PP_LOGITS=<path>: dump the last-row prefill logits (raw LE f32) — the GGUF twin
        // of the ST branch's dump above. Diagnostic: cross-arm byte-compare of two builds'
        // prefill output (e.g. the iq-k32 MMA-form A/B, research/iq-k32-20260807/).
        if let Ok(f) = std::env::var("MEMRA_PP_LOGITS") {
            let last = model.forward_last(&e, &prompt)?;
            let mut raw = Vec::with_capacity(last.len() * 4);
            for v in &last {
                raw.extend_from_slice(&v.to_le_bytes());
            }
            std::fs::write(&f, &raw)?;
            println!("pp-only prefill logits -> {f} ({} f32)", last.len());
        }
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
        println!(
            "pp-only MEDIAN: {} tok in {:.4}s = {:.1} tok/s (pp{}, {} reps)",
            prompt.len(),
            med,
            prompt.len() as f64 / med,
            prompt.len(),
            reps
        );
        return Ok(());
    }

    // --- correctness gate: decode-step prefix logits MUST match the prefill forward ---
    let prefill = model.forward_last(&e, &prompt)?;
    // decode the prompt step by step, capture last logits
    let mut cache = memra_engine::cache::Cache::new(&e, &model.cfg, prompt.len() + 64)?;
    let mut dec_logits = Vec::new();
    for &t in &prompt {
        dec_logits = model.decode_step(&e, t, &mut cache)?;
    }
    let am_p = argmax(&prefill);
    let am_d = argmax(&dec_logits);
    let maxdiff = prefill
        .iter()
        .zip(&dec_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    println!(
        "prefill argmax={am_p}  decode argmax={am_d}  logit maxdiff={maxdiff:.3e}  {}",
        if am_p == am_d { "MATCH" } else { "MISMATCH" }
    );
    if am_p != am_d {
        // near-tie vs real-gap diagnosis before the panic: both sides' view of both ids, PLUS
        // the number that actually decides which of the two it is (lane/q8-argmax,
        // research/q8-argmax-20260806/VERDICT.md). A flip is only meaningful if the config
        // spread at the contending ids is large enough to reach across the top-2 margin:
        // margin >= spread means a real numeric defect moved a logit further than the gap it
        // crossed; margin << spread is the documented near-tie coin. `logit maxdiff` above is
        // NOT that number — it is the max over a ~250k-wide vocab, dominated by tail noise,
        // and it is routinely LARGER on runs this same gate calls MATCH (measured: MATCH at
        // 1.165 beside MISMATCH at 0.466). Do not read it as severity.
        let margin_p = (prefill[am_p] - prefill[am_d]).abs();
        let margin_d = (dec_logits[am_d] - dec_logits[am_p]).abs();
        let spread = (prefill[am_p] - dec_logits[am_p])
            .abs()
            .max((prefill[am_d] - dec_logits[am_d]).abs());
        eprintln!("[gate] prefill: l[{am_p}]={:.4} l[{am_d}]={:.4} | decode: l[{am_p}]={:.4} l[{am_d}]={:.4}",
                  prefill[am_p], prefill[am_d], dec_logits[am_p], dec_logits[am_d]);
        eprintln!("[gate] top-2 margin: prefill {margin_p:.4} decode {margin_d:.4} | config spread at these ids {spread:.4} -> {}",
                  if spread > margin_p.min(margin_d) {
                      "NEAR-TIE class (the spread covers the margin; run tools/argmax-margin-gate.sh \
                       to see this position's margin against the prompt's own distribution)"
                  } else {
                      "WIDE-MARGIN flip — the spread does NOT cover the margin; this is a real defect"
                  });
    }
    assert_eq!(
        am_p, am_d,
        "decode-step diverges from prefill at the last position (see the [gate] lines above for \
         the near-tie-vs-defect diagnosis; a wide-margin flip means a cache/threading/kernel bug, \
         a margin inside the config spread is the documented cross-config drift class)"
    );

    // --- gap #46: the BATCHED-PRIME config (prime_cache — what actually seeds generation in
    //     generate/generate_with and serving) was never argmax-gated. forward_last and the
    //     tokenwise loop can BOTH be green while the batched prime flips a near-tie first
    //     token (Qwen3.6-35B pp512 probe: 365 -> 198 "\n" then EOS at 2 tokens;
    //     research/residency-cap-20260802 §4, differential in
    //     research/prime-gate-coverage-20260802). Compare its last-position logits against
    //     the tokenwise reference above: a near-tie flip is the documented cross-config
    //     drift class (REPORTED, non-fatal — MEMRA_PRIME_TOKENWISE=1 restores the tokenwise
    //     stream); a wide-margin flip or drift beyond the calibrated bounds is structural
    //     and fails hard. MEMRA_PRIME_GATE=0 skips (diagnostics seam).
    if prompt.len() >= memra_engine::hybrid_forward::PRIME_MIN_T
        && std::env::var("MEMRA_PRIME_TOKENWISE").is_err()
        && std::env::var("MEMRA_PRIME_GATE").as_deref() != Ok("0")
        && !e.frozen_cpu_experts_prefer_tokenwise_prime()
    {
        use memra_engine::forward::PrimeGateClass;
        let mut pc = memra_engine::cache::Cache::new(&e, &model.cfg, prompt.len() + 8)?;
        let (l_bp, _, _) = model.prime_cache(&e, &prompt, &mut pc)?;
        let v = memra_engine::forward::prime_gate_verdict(&dec_logits, &l_bp);
        println!(
            "batched-prime argmax={}  tokenwise argmax={}  logit maxdiff={:.3e}  {}",
            v.bp_argmax,
            v.tw_argmax,
            v.maxdiff,
            match v.class {
                PrimeGateClass::Match => "MATCH".into(),
                PrimeGateClass::NearTieFlip => format!(
                    "FLIP-NEARTIE (tokenwise margin {:.4} — cross-config drift class; the \
                     first generated token may differ from the tokenwise stream)",
                    v.tw_margin
                ),
                PrimeGateClass::Structured =>
                    format!("MISMATCH-STRUCTURED (tokenwise margin {:.4})", v.tw_margin),
            }
        );
        if v.class == PrimeGateClass::Structured {
            return Err(
                "batched-prime last-position logits diverge structurally from the tokenwise reference"
                    .into(),
            );
        }
    }

    // --- time PREFILL tok/s (batched forward over the whole prompt) for the pp comparison vs
    //     llama-bench pp512. 1 warmup discarded, then time one forward of the full prompt. ---
    if prompt.len() >= 8 {
        let _ = model.forward_last(&e, &prompt)?; // warmup
        e.stream().synchronize()?;
        let tp = std::time::Instant::now();
        let _ = model.forward_last(&e, &prompt)?;
        e.stream().synchronize()?;
        let dtp = tp.elapsed().as_secs_f64();
        println!(
            "prefill {} tok in {:.4}s = {:.1} tok/s (pp{})",
            prompt.len(),
            dtp,
            prompt.len() as f64 / dtp,
            prompt.len()
        );
    }

    // --- generate + time decode tok/s (honest Stage-A baseline) ---
    let n_new = std::env::var("MEMRA_NGEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16usize);
    let eos = tokenizer.as_ref().map(|t| t.eos_id());
    let eog: Vec<u32> = tokenizer.as_ref().map(|t| t.eog_ids()).unwrap_or_default();
    // Sampler config from env (defaults = greedy, the bit-exact reference). MEMRA_TEMP>0 enables
    // the full chain: MEMRA_TOP_K / MEMRA_TOP_P / MEMRA_MIN_P / MEMRA_PENALTY_REPEAT / MEMRA_SEED.
    let env_f = |k: &str, d: f32| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(d)
    };
    let env_u = |k: &str, d: usize| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(d)
    };
    let scfg = memra_engine::sampler::SamplerConfig {
        temperature: env_f("MEMRA_TEMP", 0.0),
        top_k: env_u("MEMRA_TOP_K", 0),
        top_p: env_f("MEMRA_TOP_P", 1.0),
        min_p: env_f("MEMRA_MIN_P", 0.0),
        penalty_last_n: env_u("MEMRA_PENALTY_LAST_N", 0),
        penalty_repeat: env_f("MEMRA_PENALTY_REPEAT", 1.0),
        penalty_freq: env_f("MEMRA_PENALTY_FREQ", 0.0),
        penalty_present: env_f("MEMRA_PENALTY_PRESENT", 0.0),
        seed: std::env::var("MEMRA_SEED")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0),
    };
    let mut sampler = memra_engine::sampler::Sampler::new(scfg);
    // Stop conditions: EOS (text path) + optional stop-strings (MEMRA_STOP="a,b").
    let mut eos_ids: Vec<u32> = eos.into_iter().collect();
    for id in eog {
        if !eos_ids.contains(&id) {
            eos_ids.push(id);
        }
    }
    let stop_strs: Vec<String> = std::env::var("MEMRA_STOP")
        .ok()
        .map(|s| {
            s.split(',')
                .map(|x| x.to_string())
                .filter(|x| !x.is_empty())
                .collect()
        })
        .unwrap_or_default();
    let params = memra_engine::decode::GenParams {
        max_new: n_new,
        max_ctx: Some(prompt.len() + n_new + 8),
        eos: eos_ids,
    };
    // The reusable serving API (BASE-3). Stop-string match runs on the detokenized tail in the
    // per-token callback. Streaming hook: callback returns false to halt.
    let mut emitted_ids: Vec<u32> = Vec::new();
    let tok_ref = tokenizer.as_ref();
    e.stream().synchronize()?;
    // MEMRA_PROFILE_GEN=1: cudaProfiler{Start,Stop} brackets ONLY the timed generate_with (pair
    // with `nsys -c cudaProfilerApi`) — window-cutting a whole-run capture misattributes the
    // tokenwise argmax-gate loop + prime into the decode share map (measured 2026-07-10: the
    // gate's small-t_kv fa_decode_f32 calls read as a phantom 5% decode share).
    let prof_gen = std::env::var("MEMRA_PROFILE_GEN").as_deref() == Ok("1");
    unsafe extern "C" {
        fn cudaProfilerStart() -> i32;
        fn cudaProfilerStop() -> i32;
    }
    if prof_gen {
        unsafe {
            cudaProfilerStart();
        }
    }
    let t0 = std::time::Instant::now();
    let gen_out = model.generate_with(&e, &prompt, &params, &mut sampler, |id| {
        emitted_ids.push(id);
        // stop-string check on the detokenized tail (text path only).
        if let (Some(tok), false) = (tok_ref, stop_strs.is_empty()) {
            let tail = tok.decode(&emitted_ids);
            if stop_strs.iter().any(|s| tail.contains(s.as_str())) {
                return false;
            }
        }
        true
    })?;
    e.stream().synchronize()?;
    if prof_gen {
        unsafe {
            cudaProfilerStop();
        }
    }
    let dt_total = t0.elapsed().as_secs_f64();
    // GEN-ONLY timing (2026-07-06 fix): generate_with primes INSIDE the timed span — at long
    // prompts the old number was prime-inclusive (35B @256-tok prime read 33.7 when decode was
    // ~51). PRIME_NANOS is the engine's published prime wall (same contract as run-spec).
    let prime_s = memra_engine::PRIME_NANOS.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9;
    let dt = (dt_total - prime_s).max(1e-9);
    let out = gen_out.tokens;
    let emitted = out.len();
    let path = if std::env::var("MEMRA_FAST").as_deref() != Ok("0") {
        "Stage-B int8 dp4a"
    } else {
        "Stage-A f32-dequant"
    };
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
        // re-priming, then reset the byte/hit counters and run MEMRA_NMEASURE more greedy decode tokens.
        // This isolates the post-warmup per-token H2D — the hot set is resident so PCIe -> a fraction.
        let n_measure: usize = std::env::var("MEMRA_NMEASURE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(32);
        if n_measure > 0 {
            let mut warm_cache = memra_engine::cache::Cache::new(
                &e,
                &model.cfg,
                prompt.len() + n_new + n_measure + 8,
            )?;
            let mut ll = Vec::new();
            for &t in &prompt {
                ll = model.decode_step(&e, t, &mut warm_cache)?;
            }
            for &t in &out {
                ll = model.decode_step(&e, t, &mut warm_cache)?;
            }
            e.moe_cache_reset_counters(); // measure ONLY the steady-state window below
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
        let text_ids: Vec<u32> = out.iter().copied().filter(|&id| Some(id) != eos).collect();
        let text = tok.decode(&text_ids);
        println!("OUTPUT TEXT: {text:?}");
        println!("--- generated text ---\n{text}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_cpu_expert_stats_without_positional_drift() {
        let before = (1, 2, 3, 4, 5, 6, 700, 8, 9, 10, 11);
        let after = (11, 22, 33, 44, 55, 66, 7_000, 88, 99, 110, 121);
        assert_eq!(
            cpu_expert_stats_delta(before, after, 12, 132),
            CpuExpertStatsDelta {
                calls: 10,
                experts: 20,
                wall_ns: 30,
                exposed_wait_ns: 120,
                ram_hits: 40,
                ram_misses: 50,
                ram_reads: 60,
                resident_bytes: 7_000,
                prepare_ns: 80,
                io_ns: 90,
                insert_ns: 100,
                compute_ns: 110,
            }
        );
    }
}
