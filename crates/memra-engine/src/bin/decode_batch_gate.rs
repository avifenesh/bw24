//! decode-batch-gate: the batched decode step's exactness battery (ARCHITECTURE-H100.md §3 B2').
//!
//! decode_step_batch is a NEW NUMERIC CONFIG vs decode_step_h's fused m=1 tier: the fused
//! path folds q8_1 scales as a separable post-op (matmul_pre_noscale + silu_mul_scaled),
//! which is m==1-only by construction; the batched path folds scales inside the matvec
//! (matmul_pre) + plain silu_mul. Same math, different FP composition — the GDN-chunked
//! prefill precedent. PROOF the plumbing is exact: under the equalized composition
//! (MEMRA_MMVQ=0 MEMRA_NO_FUSE_NORMQ=1, both paths on dp4a + unfused norms) the battery is
//! BIT-IDENTICAL at B=1 and B=N-vs-isolated (verified 2026-07-26 on H100).
//!
//! Modes (--mode, default "config"):
//!   strict — bit-identity gates; run under the EQUALIZED env or expect gate1 bit-diffs:
//!     gate1: B=1 logits bit-identical to decode_step_h, every step.
//!     gate2: B=N per-seq streams == isolated decode_step_h streams (argmax).
//!   config — the default-env battery (fused tier active in the reference):
//!     gate1: B=1 argmax stream vs decode_step_h — divergence before step 16 FAILs,
//!            later divergence is the accepted cross-config drift class (WARN).
//!     gate2: B=N per-seq LOGITS bit-identical to isolated decode_step_batch B=1 runs —
//!            the serving isolation contract (batchmates must not change your stream),
//!            enforced at full bit strength WITHIN the config.
//!
//! Usage: decode-batch-gate <model.gguf> [--steps 32] [--batch 4] [--mode config|strict]

use memra_engine::cache::Cache;
use memra_engine::forward::argmax;
use memra_engine::hybrid::HybridModel;
use memra_engine::Engine;
use memra_gguf::GgufFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: decode-batch-gate <model.gguf> [--steps N] [--batch B]");
    let rest: Vec<String> = args.collect();
    let steps: usize = rest.iter().position(|a| a == "--steps")
        .and_then(|i| rest.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(32);
    let b_n: usize = rest.iter().position(|a| a == "--batch")
        .and_then(|i| rest.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(4);
    let strict: bool = rest.iter().position(|a| a == "--mode")
        .and_then(|i| rest.get(i + 1)).map(|v| v == "strict").unwrap_or(false);

    // PIN THE PRIME CONFIG (2026-07-26): this gate compares DECODE configs
    // (decode_step_batch vs decode_step_h) from a shared primed state — the GDN prime
    // config is a nuisance variable here. The K4/K5-MMA prime (Hopper default) shifts
    // near-tie logits enough to flip the config-mode step-16 threshold on the fixed
    // prompt (observed: step-1 argmax flip; STRICT bit-gate and gate2 both still PASS,
    // proving decode itself is untouched). The mma prime's own correctness is covered
    // by its kernel-check pin + the state-carry battery + run-gen argmax gates.
    // SAFETY: single-threaded gate binary; the GDN seam reads the env per call.
    unsafe { std::env::set_var("MEMRA_GDN_MMA", "0"); }
    // Same rationale for the l2 prefill v2 config (round 27): its primed state shifts
    // the same near-tie logits at step 1. Gate tests DECODE; prime stays pinned f32-class.
    unsafe { std::env::set_var("MEMRA_L2_V2", "0"); }
    unsafe { std::env::set_var("MEMRA_FA3", "0"); }
    unsafe { std::env::set_var("MEMRA_GDN_WGMMA", "0"); }
    let e = Engine::new(0)?;
    let g = GgufFile::open(&path)?;
    let model = HybridModel::load_without_mtp(&e, &g)?;
    println!("loaded {} ({} layers); steps={steps} batch={b_n}",
             g.arch().unwrap_or("?"), model.layers.len());

    // Distinct prompts per lane so caches/states genuinely diverge; length >= 16
    // (PRIME_MIN_T floor) and deliberately uneven so positions differ across the batch.
    // MEMRA_GATE_SEED offsets the token pattern — the cross-config drift class is a
    // near-tie roulette on any single synthetic prompt, so calibration sweeps need
    // several draws (the 2026-07-31 shexp-dot re-sweep; default 0 = the historic prompt).
    let seed: u32 = std::env::var("MEMRA_GATE_SEED").ok()
        .and_then(|v| v.parse().ok()).unwrap_or(0);
    let prompts: Vec<Vec<u32>> = (0..b_n.max(2))
        .map(|i| (0..20 + i as u32 * 5).map(|j| 55 + seed * 13 + i as u32 * 97 + j * 31).collect())
        .collect();
    let ctx = 512 + steps + 64;

    // ---- Gate 1: B=1 vs decode_step_h ----
    // strict: bit-identity on the seed prompt (run under the EQUALIZED env).
    // config: MULTI-SEED calibration (re-swept 2026-07-31, the shexp-dot dig): the two
    // decode configs carry an ACCEPTED FP-composition gap, and on any single synthetic
    // prompt the first argmax divergence is a near-tie roulette — a 6-seed sweep of the
    // PRE-change tree failed the old single-prompt step-16 rule on 3/6 draws (steps
    // 7/8/15), so that rule detected the dice, not the plumbing. Plumbing bugs (wrong
    // token fed, KV misindexed) diverge at step 0-2 on EVERY draw; observed numeric-tie
    // flips start at step 6+. FAIL iff any seed diverges before step 3.
    let mut g1_fail = 0usize;
    let g1_seeds: u32 = if strict { 1 } else { 6 };
    for gs in 0..g1_seeds {
        let p0: Vec<u32> = (0..20).map(|j| 55 + (seed + gs) * 13 + j * 31).collect();
        let mut c_ref = Cache::new(&e, &model.cfg, ctx)?;
        let mut c_bat = Cache::new(&e, &model.cfg, ctx)?;
        let _ = model.prime_cache(&e, &p0, &mut c_ref)?;
        let _ = model.prime_cache(&e, &p0, &mut c_bat)?;
        let mut t_ref = *p0.last().unwrap();
        let mut t_bat = t_ref;
        let mut diverged: Option<usize> = None;
        for s in 0..steps {
            let (l_ref, _) = model.decode_step_h(&e, t_ref, &mut c_ref)?;
            let l_bat = {
                let mut caches = [&mut c_bat];
                model.decode_step_batch(&e, &[t_bat], &mut caches)?.remove(0)
            };
            if strict {
                let bits_equal = l_ref.len() == l_bat.len()
                    && l_ref.iter().zip(l_bat.iter()).all(|(a, b)| a.to_bits() == b.to_bits());
                if !bits_equal {
                    let md = l_ref.iter().zip(l_bat.iter())
                        .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                    println!("gate1 step {s}: BIT-DIFF (maxdiff {md:.3e}) FAIL");
                    g1_fail += 1;
                    if g1_fail > 3 { break; }
                }
            }
            t_ref = argmax(&l_ref) as u32;
            t_bat = argmax(&l_bat) as u32;
            if t_ref != t_bat { diverged = Some(s); break; }
        }
        match diverged {
            Some(s) if strict || s < 3 => {
                println!("gate1 seed {gs} step {s}: token diverged FAIL");
                g1_fail += 1;
            }
            Some(s) => println!("gate1 seed {gs} step {s}: token diverged — accepted \
                                 cross-config drift (WARN)"),
            None => println!("gate1 seed {gs}: agreement all {steps} steps"),
        }
    }
    println!("gate1 (B=1 {} vs decode_step_h, {steps} steps, {g1_seeds} seed(s)): {}",
             if strict { "bit-identity" } else { "argmax agreement" },
             if g1_fail == 0 { "PASS" } else { "FAIL" });

    // ---- Gate 2: B=N vs isolated (the serving isolation contract) ----
    // Reference = isolated runs of the SAME config: strict mode uses decode_step_h,
    // config mode uses decode_step_batch at B=1 — within-config, bit strength applies.
    let mut ref_streams: Vec<Vec<u32>> = Vec::new();
    let mut ref_logits: Vec<Vec<Vec<f32>>> = Vec::new();
    for p in prompts.iter().take(b_n) {
        let mut c = Cache::new(&e, &model.cfg, ctx)?;
        let _ = model.prime_cache(&e, p, &mut c)?;
        let mut t = *p.last().unwrap();
        let mut out = Vec::with_capacity(steps);
        let mut ls = Vec::with_capacity(steps);
        for _ in 0..steps {
            let l = if strict {
                model.decode_step_h(&e, t, &mut c)?.0
            } else {
                let mut caches = [&mut c];
                model.decode_step_batch(&e, &[t], &mut caches)?.remove(0)
            };
            t = argmax(&l) as u32;
            out.push(t);
            ls.push(l);
        }
        ref_streams.push(out);
        ref_logits.push(ls);
    }
    // Batched run over fresh caches primed identically.
    let mut caches: Vec<Cache> = Vec::new();
    for p in prompts.iter().take(b_n) {
        let mut c = Cache::new(&e, &model.cfg, ctx)?;
        let _ = model.prime_cache(&e, p, &mut c)?;
        caches.push(c);
    }
    let mut toks: Vec<u32> = prompts.iter().take(b_n).map(|p| *p.last().unwrap()).collect();
    let mut g2_fail = 0usize;
    for s in 0..steps {
        let mut cache_refs: Vec<&mut Cache> = caches.iter_mut().collect();
        let logits = model.decode_step_batch(&e, &toks, &mut cache_refs)?;
        for (bi, l) in logits.iter().enumerate() {
            toks[bi] = argmax(l) as u32;
            if toks[bi] != ref_streams[bi][s] {
                println!("gate2 seq {bi}: token DIVERGED from isolated at step {s} FAIL");
                g2_fail += 1;
            } else if !strict {
                // within-config: batchmates must not perturb even one bit of your logits
                let r = &ref_logits[bi][s];
                if !(r.len() == l.len()
                     && r.iter().zip(l.iter()).all(|(a, b)| a.to_bits() == b.to_bits())) {
                    let md = r.iter().zip(l.iter())
                        .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                    println!("gate2 seq {bi} step {s}: LOGIT BIT-DIFF vs isolated \
                              (maxdiff {md:.3e}) FAIL");
                    g2_fail += 1;
                }
            }
        }
        if g2_fail > 6 { break; }
    }
    println!("gate2 (B={b_n} vs isolated {}, {steps} steps): {}",
             if strict { "decode_step_h" } else { "batched-B=1, bit-checked" },
             if g2_fail == 0 { "PASS" } else { "FAIL" });

    // ---- Gate 3: DEVICE-SIDE SAMPLING isolation + greedy identity (2026-08-01 lever) ----
    // (a) greedy device rows: decode_step_batch_sampled's device argmax token must equal the
    //     host argmax of the SAME returned logits row, every row, every step (the argmax-gate
    //     contract surfaced at the batched-tick API).
    // (b) sampled isolation: per-seq (temp=0.7, seed=seq, ctr=step) device draws at B=N must
    //     equal the SAME metas' draws at B=1 over identically-primed caches — the serving
    //     isolation contract for the device sampler (batchmates must not change your stream).
    let mut g3_fail = 0usize;
    {
        // (a) greedy identity inside the batch.
        let mut caches: Vec<Cache> = Vec::new();
        for p in prompts.iter().take(b_n) {
            let mut c = Cache::new(&e, &model.cfg, ctx)?;
            let _ = model.prime_cache(&e, p, &mut c)?;
            caches.push(c);
        }
        let mut toks: Vec<u32> = prompts.iter().take(b_n).map(|p| *p.last().unwrap()).collect();
        let samp_g: Vec<Option<(f32, u64, u32)>> = vec![Some((0.0, 0, 0)); b_n];
        for _s in 0..steps.min(16) {
            let mut cache_refs: Vec<&mut Cache> = caches.iter_mut().collect();
            let (rows, next) = model.decode_step_batch_sampled(&e, &toks, &mut cache_refs, &samp_g)?;
            for (bi, l) in rows.iter().enumerate() {
                let host_am = argmax(l) as u32;
                let dev = next[bi].expect("greedy device row missing token");
                if dev != host_am {
                    println!("gate3a seq {bi}: device argmax {dev} != host argmax {host_am} FAIL");
                    g3_fail += 1;
                }
                toks[bi] = host_am;
            }
            if g3_fail > 4 { break; }
        }
        // (b) sampled isolation: B=N vs B=1, same per-seq (seed, ctr) schedule.
        let n_s = steps.min(16);
        let mut iso: Vec<Vec<u32>> = Vec::with_capacity(b_n);
        for bi in 0..b_n {
            let mut c = Cache::new(&e, &model.cfg, ctx)?;
            let _ = model.prime_cache(&e, &prompts[bi], &mut c)?;
            let mut t = *prompts[bi].last().unwrap();
            let mut out = Vec::with_capacity(n_s);
            for s in 0..n_s {
                let mut refs = [&mut c];
                let samp = [Some((0.7f32, bi as u64 + 1, s as u32))];
                let (_, nx) = model.decode_step_batch_sampled(&e, &[t], &mut refs, &samp)?;
                t = nx[0].expect("sampled row missing token");
                out.push(t);
            }
            iso.push(out);
        }
        let mut bat: Vec<Vec<u32>> = vec![Vec::with_capacity(n_s); b_n];
        {
            let mut caches: Vec<Cache> = Vec::new();
            for p in prompts.iter().take(b_n) {
                let mut c = Cache::new(&e, &model.cfg, ctx)?;
                let _ = model.prime_cache(&e, p, &mut c)?;
                caches.push(c);
            }
            let mut toks: Vec<u32> =
                prompts.iter().take(b_n).map(|p| *p.last().unwrap()).collect();
            for s in 0..n_s {
                let samp: Vec<Option<(f32, u64, u32)>> =
                    (0..b_n).map(|bi| Some((0.7f32, bi as u64 + 1, s as u32))).collect();
                let mut cache_refs: Vec<&mut Cache> = caches.iter_mut().collect();
                let (_, nx) = model.decode_step_batch_sampled(&e, &toks, &mut cache_refs, &samp)?;
                for bi in 0..b_n {
                    toks[bi] = nx[bi].expect("sampled row missing token");
                    bat[bi].push(toks[bi]);
                }
            }
        }
        for bi in 0..b_n {
            if iso[bi] != bat[bi] {
                let d = iso[bi].iter().zip(&bat[bi]).position(|(a, b)| a != b);
                println!("gate3b seq {bi}: sampled stream DIVERGED batched-vs-isolated at \
                          step {d:?} FAIL");
                g3_fail += 1;
            }
        }
        // (c) LEAN-LOGITS identity (inc2 component 3): the lean tick must (i) produce the
        //     SAME device tokens as the full tick, (ii) park every sampled row's logits
        //     on-device BIT-IDENTICALLY to the full tick's returned host row, (iii) leave
        //     unsampled rows' returned host rows bit-identical. Mixed metas (alternating
        //     greedy-device / host rows) exercise the partial-D2H path.
        {
            let n_s = steps.min(8);
            let mut caches_f: Vec<Cache> = Vec::new();
            let mut caches_l: Vec<Cache> = Vec::new();
            for p in prompts.iter().take(b_n) {
                let mut c = Cache::new(&e, &model.cfg, ctx)?;
                let _ = model.prime_cache(&e, p, &mut c)?;
                caches_f.push(c);
                let mut c = Cache::new(&e, &model.cfg, ctx)?;
                let _ = model.prime_cache(&e, p, &mut c)?;
                caches_l.push(c);
            }
            let mut toks: Vec<u32> = prompts.iter().take(b_n).map(|p| *p.last().unwrap()).collect();
            for _s in 0..n_s {
                let samp: Vec<Option<(f32, u64, u32)>> = (0..b_n)
                    .map(|bi| if bi % 2 == 0 { Some((0.0, 0, 0)) } else { None })
                    .collect();
                let (rows_f, next_f) = {
                    let mut refs: Vec<&mut Cache> = caches_f.iter_mut().collect();
                    model.decode_step_batch_sampled_lean(&e, &toks, &mut refs, &samp, false)?
                };
                let (rows_l, next_l) = {
                    let mut refs: Vec<&mut Cache> = caches_l.iter_mut().collect();
                    model.decode_step_batch_sampled_lean(&e, &toks, &mut refs, &samp, true)?
                };
                for bi in 0..b_n {
                    if samp[bi].is_some() {
                        if next_f[bi] != next_l[bi] {
                            println!("gate3c seq {bi}: lean token {:?} != full token {:?} FAIL",
                                     next_l[bi], next_f[bi]);
                            g3_fail += 1;
                        }
                        if !rows_l[bi].is_empty() {
                            println!("gate3c seq {bi}: lean sampled row NOT empty FAIL");
                            g3_fail += 1;
                        }
                        let parked = e.dtoh(caches_l[bi].last_logits_dev.as_ref()
                                            .expect("lean row missing device park"))?;
                        let r = &rows_f[bi];
                        if !(parked.len() == r.len()
                             && parked.iter().zip(r.iter()).all(|(a, b)| a.to_bits() == b.to_bits())) {
                            println!("gate3c seq {bi}: parked device logits != full host row FAIL");
                            g3_fail += 1;
                        }
                        toks[bi] = next_f[bi].unwrap();
                    } else {
                        let (r, l) = (&rows_f[bi], &rows_l[bi]);
                        if !(r.len() == l.len()
                             && r.iter().zip(l.iter()).all(|(a, b)| a.to_bits() == b.to_bits())) {
                            println!("gate3c seq {bi}: unsampled row lean != full FAIL");
                            g3_fail += 1;
                        }
                        toks[bi] = argmax(r) as u32;
                    }
                }
                if g3_fail > 8 { break; }
            }
        }
    }
    println!("gate3 (device sampling: greedy==host-argmax + sampled B={b_n} vs isolated \
              + lean-logits identity): {}",
             if g3_fail == 0 { "PASS" } else { "FAIL" });

    if g1_fail + g2_fail + g3_fail == 0 {
        println!("ALL GREEN: decode_step_batch exactness battery");
        Ok(())
    } else {
        Err("decode-batch-gate FAILED".into())
    }
}
