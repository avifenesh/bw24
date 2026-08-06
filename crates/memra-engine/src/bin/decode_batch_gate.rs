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
//! STRICT-MODE EQUALIZATION NOW COVERS NVFP4 (lane/nvfp4-strict, 2026-08-05). History: the
//! equalizing env was Q8/dp4a-shaped — MEMRA_MMVQ=0 steered the Q8_0-class arms (their
//! fused launches all sit behind `q8_fused_params`, which refuses under MMVQ=0 by the
//! FP-order law) but the NVFP4 gate+up/beta+alpha pair door (`matmul_pre_dual_noscale`)
//! had no such check: the oracle kept dispatching `qmatvec_nvfp4_mmvq_dual_mr2` (MMVQ
//! 32-thread warp-reduce family) while the batched side fell to dp4a (128-thread
//! two-level reduce) — a mixed-family comparison, so `--mode strict` FAILED on ANY NVFP4
//! model at pristine trees (train-HEAD receipts: gate1 maxdiff 1.639e-1 @ step 2 on q9,
//! research/servepath-p2-20260805/logs/dbg-strict-b4-TRAINHEAD.log; gate2 step-6 token
//! divergence on q27 at 93420980, research/nvfp4-strict-20260805/repro.log). The fix
//! applies the SAME law to the NVFP4 arm: `matmul_pre_dual_noscale` returns None when
//! `mmvq_supports(QT_NVFP4)` is false, so under the equalized env BOTH sides ride dp4a
//! and strict bit-identity holds (default env is dispatch-unchanged). A strict FAIL on an
//! NVFP4 model is a REAL failure again.
//!
//! Modes (--mode, default "config"):
//!   strict — bit-identity gates; run under the EQUALIZED env or expect gate1 bit-diffs:
//!     gate1: B=1 logits bit-identical to decode_step_h, every step.
//!     gate2: B=N per-seq streams == isolated decode_step_h streams (argmax).
//!   config — the default-env battery (fused tier active in the reference):
//!     gate1: B=1 argmax stream vs decode_step_h, 6 prompt draws — FAILs iff >= 4 of the
//!            6 draws diverge before step 3 (the plumbing class shows on EVERY draw);
//!            fewer/later divergence is the accepted cross-config drift class (WARN).
//!     gate2: B=N per-seq LOGITS bit-identical to isolated decode_step_batch B=1 runs —
//!            the serving isolation contract (batchmates must not change your stream),
//!            enforced at full bit strength WITHIN the config.
//!   pp     — THE BATCHED STAGE-SPLIT EXACTNESS GATE (pp2-batch 2026-08-06). See
//!            `pp_battery` below. Opens `MEMRA_PP_STAGES` BEFORE load (weight sharding is
//!            a load-time decision), then proves `decode_step_batch_ppn` is BIT-IDENTICAL
//!            to the unsplit batched body over the same weights, per row, per step.
//!            gate1/2/3 are skipped in this mode — they are single-device jurisdiction and
//!            run in their own invocations.
//!
//! Usage: decode-batch-gate <model.gguf> [--steps 32] [--batch 4] [--mode config|strict|pp]
//!        pp mode also honours: --batch 1,4,8 (list), --stages N (default 2), --reps R
//!        (default 2 — the split arm repeats, because the shared-scratch race class this
//!        gate must catch was a 35% FLAKE, so one green replay is not evidence), and passes
//!        MEMRA_PP_DEVICES / MEMRA_PP_SPLITS / MEMRA_PP_SHARD through from the caller.

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
    // --batch takes a comma list in pp mode (one battery per width in ONE process, so all
    // widths are measured against the SAME loaded weights); the other modes use the first.
    let batches: Vec<usize> = rest.iter().position(|a| a == "--batch")
        .and_then(|i| rest.get(i + 1))
        .map(|v| v.split(',').filter_map(|p| p.trim().parse().ok()).collect::<Vec<usize>>())
        .filter(|v: &Vec<usize>| !v.is_empty())
        .unwrap_or_else(|| vec![4]);
    let b_n: usize = batches[0];
    let mode: String = rest.iter().position(|a| a == "--mode")
        .and_then(|i| rest.get(i + 1)).cloned().unwrap_or_else(|| "config".into());
    let strict: bool = mode == "strict";
    let pp_mode: bool = mode == "pp";
    let stages: usize = rest.iter().position(|a| a == "--stages")
        .and_then(|i| rest.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(2);
    let reps: usize = rest.iter().position(|a| a == "--reps")
        .and_then(|i| rest.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(2);

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
    // Round 49: the grouped f16 expert-prefill door (Hopper default mode 1; sm_120a naked
    // default mode 2 since lane/f16g-default-rearb 2026-08-02 — "0" fully closes the door
    // under every mode semantics, so this pin is default-flip-invariant) is another PRIME
    // nuisance — same signature as the K4/K5-MMA precedent (gate1 seed flip at step 0,
    // gate2+gate3 bit-strength PASS, and pinning it off restores 6/6 seeds). The door's
    // own correctness is covered by kernel-check + run-gen argmax + run-spec gates.
    unsafe { std::env::set_var("MEMRA_MOE_F16G", "0"); }
    // PP MODE OPENS THE DOOR BEFORE LOAD (ppn-gate's method): weight sharding is a
    // LOAD-TIME decision — `hybrid.rs` asks `pp::layer_engine` per layer, so a door opened
    // after load would test a split walk over unsharded weights, i.e. not the serving
    // placement at all. The primary device follows MEMRA_PP_DEVICES[0] for the same reason
    // (stage 0's engine IS the primary engine).
    let primary_dev: usize = if pp_mode {
        unsafe { std::env::set_var("MEMRA_PP_STAGES", stages.to_string()); }
        std::env::var("MEMRA_PP_DEVICES").ok()
            .and_then(|v| v.split(',').next().and_then(|s| s.trim().parse().ok()))
            .unwrap_or(0)
    } else {
        0
    };
    let e = Engine::new(primary_dev)?;
    let g = GgufFile::open(&path)?;
    let model = HybridModel::load_without_mtp(&e, &g)?;
    println!("loaded {} ({} layers); steps={steps} batch={b_n}",
             g.arch().unwrap_or("?"), model.layers.len());

    if pp_mode {
        let seed: u32 = std::env::var("MEMRA_GATE_SEED").ok()
            .and_then(|v| v.parse().ok()).unwrap_or(0);
        let fails = pp_battery(&e, &model, stages, steps, &batches, reps, seed)?;
        if fails == 0 {
            println!("ALL GREEN: batched PP-{stages} stage-split exactness battery");
            return Ok(());
        }
        return Err("decode-batch-gate --mode pp FAILED".into());
    }

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
    // config: MULTI-SEED calibration. The two decode configs carry an ACCEPTED
    // FP-composition gap, and on any single synthetic prompt the first argmax divergence
    // is a near-tie roulette — the H100 re-sweep (2026-07-31, the shexp-dot dig) failed
    // the old single-prompt step-16 rule on 3/6 draws (steps 7/8/15), and its
    // replacement ("FAIL iff ANY seed diverges before step 3") assumed tie flips start
    // at step 6+ — an H100-only observation. The 5090 re-sweep (2026-08-02,
    // research/gate1-recal-20260802/: 18 draws x {q9j Q8_0, q35 IQ4_XS}) saw legal dice
    // at steps 0/1/3/4 (q35 seeds 16/17 flip at step 0; q9j seed 0 at step 1), each
    // PROVEN dice by bit-identity under the equalized strict env on the very same draws.
    // The per-draw step threshold carries no rig-invariant signal; the FRACTION does:
    // plumbing (wrong token fed, KV misindexed) diverges at step 0-2 on EVERY draw,
    // observed dice reach at most 2 early draws per 6-window. FAIL iff >= G1_EARLY_K of
    // the 6 draws diverge before step G1_EARLY_STEP — margin 2 above the observed dice
    // maximum, margin 2 below the plumbing floor (6/6). Teeth verified by the
    // MEMRA_GATE_CANARY=1 wrong-token canary (must FAIL 6/6). Strict gate1 + gate2 +
    // gate3 keep full bit strength — they remain the hard exactness floor.
    const G1_EARLY_STEP: usize = 3; // plumbing window: wrong token/KV shows at step 0-2
    const G1_EARLY_K: usize = 4; // FAIL iff this many draws diverge inside the window
    let canary = std::env::var("MEMRA_GATE_CANARY").map(|v| v == "1").unwrap_or(false);
    let mut g1_fail = 0usize;
    let mut g1_early = 0usize;
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
            // TEST-ONLY plumbing canary (MEMRA_GATE_CANARY=1): feed the batched lane one
            // wrong token — the class the fraction rule must keep catching (FAIL 6/6).
            if canary && s == 1 {
                t_bat = if t_bat == 0 { 1 } else { t_bat - 1 };
            }
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
            Some(s) if strict => {
                println!("gate1 seed {gs} step {s}: token diverged FAIL");
                g1_fail += 1;
            }
            Some(s) if s < G1_EARLY_STEP => {
                g1_early += 1;
                println!("gate1 seed {gs} step {s}: token diverged EARLY \
                          (step < {G1_EARLY_STEP}; plumbing iff every draw)");
            }
            Some(s) => println!("gate1 seed {gs} step {s}: token diverged — accepted \
                                 cross-config drift (WARN)"),
            None => println!("gate1 seed {gs}: agreement all {steps} steps"),
        }
    }
    if !strict {
        println!("gate1 early draws (step < {G1_EARLY_STEP}): {g1_early}/{g1_seeds} \
                  (FAIL threshold >= {G1_EARLY_K}; plumbing class = every draw)");
        if g1_early >= G1_EARLY_K {
            g1_fail += 1;
        }
    }
    println!("gate1 (B=1 {} vs decode_step_h, {steps} steps, {g1_seeds} seed(s)): {}",
             if strict { "bit-identity" } else { "argmax agreement" },
             if g1_fail == 0 { "PASS" } else { "FAIL" });

    // ---- Gate 2: B=N vs isolated (the serving isolation contract) ----
    // Reference = isolated runs of the SAME config: strict mode uses decode_step_h,
    // config mode uses decode_step_batch at B=1 — within-config, bit strength applies.
    //
    // H3 (serve-path phase 2, 2026-08-05) PINS the B=1 REFERENCE ARM of gate2 AND gate3b
    // to the batched body. Reason: the B=1 fast path routes solo sequences onto the m=1
    // FUSED trunk, so an unpinned `decode_step_batch(&[t])` reference would no longer run
    // the code these gates exist to test — their bit/stream checks would silently degrade
    // from "batchmates must not perturb your logits" (the real teeth) into a cross-config
    // FP-composition comparison, which gate1's config mode already tolerates by design.
    // Pinning keeps their jurisdiction exactly where it was: the BATCHED m>=2 body.
    // The fast path's own exactness is gate1's job (STRICT gate1 = bit-identity to
    // decode_step_h, which PASSes ONLY with the fast path ON — verified on-box: OFF fails
    // at maxdiff 1.591e-1). Set through the explicit seam rather than the env because
    // gate1 above already ran with the fast path live and the read is memoized.
    let b1_fast_live = HybridModel::b1_fast_on();
    HybridModel::set_b1_fast(false);
    println!("gate2/gate3 B=1 reference arm: batched body (B=1 fast path pinned OFF; \
              live default = {})", if b1_fast_live { "ON" } else { "OFF" });
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
    // 24GB-card capacity (inc3, 2026-08-01): free gate2's cache herd + host logits before
    // gate3 allocates its own (B=16 with the q8rp mirror OOM'd gate3 on the 5090 while
    // every verdict was green — harness footprint, not model state).
    drop(caches);
    drop(ref_logits);
    drop(ref_streams);

    // ---- Gate 3: DEVICE-SIDE SAMPLING isolation + greedy identity (2026-08-01 lever) ----
    // (a) greedy device rows: decode_step_batch_sampled's device argmax token must equal the
    //     host argmax of the SAME returned logits row, every row, every step (the argmax-gate
    //     contract surfaced at the batched-tick API).
    // (b) sampled isolation: per-seq (temp=0.7, seed=seq, ctr=step) device draws at B=N must
    //     equal the SAME metas' draws at B=1 over identically-primed caches — the serving
    //     isolation contract for the device sampler (batchmates must not change your stream).
    let mut g3_fail = 0usize;
    {
        // (a) greedy identity inside the batch. (Own block: the cache herd frees before (b) —
        // the 24GB-card capacity rule above.)
        {
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

/// Per-arm bit ledger (the ppn-gate `ArmCheck` pattern, widened to per-row): a batched tick
/// returns B logit rows, so a mismatch is located by (step, row, index) — the row index is
/// what tells a stage-split bug (every row wrong at one step) apart from a per-row cache
/// bug (one row wrong from its own step onward).
struct BitCheck {
    name: String,
    bad: usize,
    first: Option<(usize, usize, usize, f32, f32)>, // (step, row, idx, ref, got)
    compared: usize,
}

impl BitCheck {
    fn new(name: String) -> Self {
        BitCheck { name, bad: 0, first: None, compared: 0 }
    }
    fn check(&mut self, step: usize, row: usize, got: &[f32], r: &[f32]) {
        assert_eq!(got.len(), r.len(), "row length mismatch (ref {} vs got {})", r.len(), got.len());
        self.compared += got.len();
        let diffs =
            got.iter().zip(r.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        if diffs > 0 {
            self.bad += 1;
            let (idx, (a, b)) = got.iter().zip(r.iter()).enumerate()
                .find(|(_, (a, b))| a.to_bits() != b.to_bits())
                .map(|(i, (a, b))| (i, (*b, *a)))
                .unwrap();
            if self.first.is_none() {
                self.first = Some((step, row, idx, a, b));
            }
            if self.bad <= 5 {
                println!("[{}] MISMATCH step {step} row {row}: {diffs}/{} logits differ, \
                          first @[{idx}] ref={a:?} pp={b:?}", self.name, r.len());
            }
        }
    }
    /// Returns 1 on failure (the caller's fail counter increments), 0 on pass.
    fn verdict(&self) -> usize {
        if self.bad == 0 {
            println!("pp gate PASS [{}]: {} f32 logits BIT-IDENTICAL (0 differing bits)",
                     self.name, self.compared);
            0
        } else {
            let (s, row, i, a, b) = self.first.unwrap();
            println!("pp gate FAIL [{}]: {} rows mismatched of {} f32 compared (first @ step \
                      {s} row {row} idx {i}: ref={a:?} pp={b:?})",
                     self.name, self.bad, self.compared);
            1
        }
    }
}

/// THE BATCHED STAGE-SPLIT EXACTNESS GATE (`--mode pp`, pp2-batch 2026-08-06).
///
/// `decode_step_batch_ppn` runs each stage's layer range on its own engine/stream with a
/// `[B, n_embd]` boundary copy between them. That copy is exact (dtod / cudaMemcpyPeerAsync,
/// no conversion) and every stage runs the SAME kernels on the SAME bytes in the same order,
/// so PP-N adds ZERO deviation: the split MUST be BIT-IDENTICAL to the unsplit batched body,
/// per row, per step. The batched analogue of the eager arm's bar (48 steps x 248,320 f32
/// logits, zero differing bits — research/pp2-hardening-20260806).
///
/// METHOD (ppn-gate's, widened to B rows): the door opens BEFORE LOAD so the weights are
/// genuinely sharded, then the door is CLOSED for the reference walk. That reference is the
/// unsplit batched body over the SAME sharded placement — it peer-reads the remote stages'
/// weights, which is slow (13.9-28x) but BYTE-EXACT, which is precisely why the placement
/// needed a refusal rather than a gate. The recorded inputs come from the reference's own
/// greedy stream, so a mismatch can never desync the comparison.
///
/// THREE ARMS, and the middle one is the localizer:
///   1. `split`      — door ON, ppN caches, the stage split. Repeated `reps` times: the
///                     shared-Engine scratch race this design avoids was a 35% FLAKE
///                     (2026-08-02), so ONE green replay is not evidence of absence.
///   2. `unsplit@ppncache` — door ON, ppN caches (identical placement to arm 1), but
///                     MEMRA_BATCH_PP=0 forces the UNSPLIT walk, with
///                     MEMRA_PP_ALLOW_UNSPLIT_BATCH=1 to pass the fail-closed guard. This
///                     holds cache placement constant and varies ONLY the walk, so an arm-1
///                     failure with arm 2 green localizes to the stage split, and both
///                     failing localizes to stage-owned cache allocation.
///   3. `epilogue`   — the last-stage epilogue: device-sampled greedy rows must equal the
///                     host argmax of their own returned row, and a lean tick must park
///                     logits bit-identically to the full tick's host row. New jurisdiction
///                     for this lane, because under the split that epilogue (mask ->
///                     sampler -> `cache.last_logits_dev`) runs on the LAST stage's engine
///                     and device, not the primary's.
///
/// The B=1 FAST PATH IS PINNED OFF for the whole battery: with the door shut its condition
/// is satisfied, so the reference at B=1 would be the m=1 FUSED trunk instead of the batched
/// body — an accepted cross-config FP-composition gap (gate1's jurisdiction) that would show
/// up here as a fake stage-split failure. Pinned through the explicit seam.
#[allow(clippy::too_many_arguments)]
fn pp_battery(
    e: &Engine,
    model: &HybridModel,
    stages: usize,
    steps: usize,
    batches: &[usize],
    reps: usize,
    seed: u32,
) -> Result<usize, Box<dyn std::error::Error>> {
    let n_layers = model.layers.len();
    let fence = memra_engine::pp::pp_cuts(n_layers).unwrap_or_else(|| {
        panic!("pp mode: door failed to open (n_layers={n_layers}, stages={stages})")
    });
    assert_eq!(fence.len() - 1, stages, "fence {fence:?} != stages {stages}");
    let devices = std::env::var("MEMRA_PP_DEVICES").unwrap_or_default();
    let knobs = format!(
        "stages={stages} fence={fence:?} devices={} splits={} shard={} streams={}",
        if devices.is_empty() { "default(primary)".into() } else { devices.clone() },
        std::env::var("MEMRA_PP_SPLITS").unwrap_or_else(|_| "default(even)".into()),
        if memra_engine::pp::pp_shard_off() { "OFF(all-primary)" } else { "per-stage" },
        if memra_engine::pp::pp2_streams_off() { "OFF(same-stream)" } else { "per-stage" },
    );
    println!("pp mode: batched stage-split exactness battery over {n_layers} layers; {knobs}");
    println!("pp mode: batches={batches:?} steps={steps} reps={reps} (split arm)");
    // See the fn doc: the reference must be the BATCHED body at every B, including B=1.
    let b1_live = HybridModel::b1_fast_on();
    HybridModel::set_b1_fast(false);
    println!("pp mode: B=1 fast path pinned OFF (live default = {})",
             if b1_live { "ON" } else { "OFF" });

    let ctx = 512 + steps + 64;
    let mut fails = 0usize;

    for &b in batches {
        // Uneven prompt lengths => uneven cache.pos across rows, which is the real serving
        // shape (per-row t_kv, so the split's per-stage pointer tables and the t_kv_max
        // padding path are both exercised rather than a degenerate all-equal-pos batch).
        let prompts: Vec<Vec<u32>> = (0..b)
            .map(|i| (0..20 + i as u32 * 5)
                 .map(|j| 55 + seed * 13 + i as u32 * 97 + j * 31).collect())
            .collect();

        // ---- REFERENCE: door OFF, unsplit batched body, primary-allocated caches ----
        // Sharded weights stay where the loader put them; peer reads are byte-exact.
        unsafe { std::env::remove_var("MEMRA_PP_STAGES"); }
        let mut inputs: Vec<Vec<u32>> = Vec::with_capacity(steps);
        let mut ref_logits: Vec<Vec<Vec<f32>>> = Vec::with_capacity(steps);
        {
            let mut caches: Vec<Cache> = Vec::with_capacity(b);
            for p in prompts.iter() {
                let mut c = Cache::new(e, &model.cfg, ctx)?;
                let _ = model.prime_cache(e, p, &mut c)?;
                caches.push(c);
            }
            let mut toks: Vec<u32> = prompts.iter().map(|p| *p.last().unwrap()).collect();
            for _ in 0..steps {
                inputs.push(toks.clone());
                let mut refs: Vec<&mut Cache> = caches.iter_mut().collect();
                let rows = model.decode_step_batch(e, &toks, &mut refs)?;
                for (bi, l) in rows.iter().enumerate() {
                    toks[bi] = argmax(l) as u32;
                }
                ref_logits.push(rows);
            }
        }
        let n_vocab = ref_logits[0][0].len();
        println!("-- B={b}: reference recorded ({steps} steps x {b} rows x {n_vocab} f32, \
                  door OFF over the sharded placement)");

        // ---- ARM 1: THE SPLIT (door ON, ppN caches), repeated for the flake class ----
        unsafe { std::env::set_var("MEMRA_PP_STAGES", stages.to_string()); }
        for rep in 0..reps.max(1) {
            let mut chk = BitCheck::new(format!("split B={b} rep{rep}"));
            let mut caches: Vec<Cache> = Vec::with_capacity(b);
            for p in prompts.iter() {
                let mut c = memra_engine::pp::new_cache(e, &model.cfg, ctx)?;
                let _ = model.prime_cache(e, p, &mut c)?;
                caches.push(c);
            }
            for (s, toks) in inputs.iter().enumerate() {
                let mut refs: Vec<&mut Cache> = caches.iter_mut().collect();
                let rows = model.decode_step_batch(e, toks, &mut refs)?;
                for (bi, l) in rows.iter().enumerate() {
                    chk.check(s, bi, l, &ref_logits[s][bi]);
                }
            }
            fails += chk.verdict();
        }

        // ---- ARM 2: UNSPLIT WALK over the SAME ppN cache placement (the localizer) ----
        {
            let mut chk = BitCheck::new(format!("unsplit@ppncache B={b}"));
            let mut caches: Vec<Cache> = Vec::with_capacity(b);
            for p in prompts.iter() {
                let mut c = memra_engine::pp::new_cache(e, &model.cfg, ctx)?;
                let _ = model.prime_cache(e, p, &mut c)?;
                caches.push(c);
            }
            // MEMRA_BATCH_PP=0 selects the unsplit body; the ALLOW override is required
            // because that body is exactly what `refuse_unsplit_if_remote` fails closed on
            // under a sharded cross-device placement. Both are restored right after.
            unsafe {
                std::env::set_var("MEMRA_BATCH_PP", "0");
                std::env::set_var("MEMRA_PP_ALLOW_UNSPLIT_BATCH", "1");
            }
            let r = (|| -> Result<(), Box<dyn std::error::Error>> {
                for (s, toks) in inputs.iter().enumerate() {
                    let mut refs: Vec<&mut Cache> = caches.iter_mut().collect();
                    let rows = model.decode_step_batch(e, toks, &mut refs)?;
                    for (bi, l) in rows.iter().enumerate() {
                        chk.check(s, bi, l, &ref_logits[s][bi]);
                    }
                }
                Ok(())
            })();
            unsafe {
                std::env::remove_var("MEMRA_BATCH_PP");
                std::env::remove_var("MEMRA_PP_ALLOW_UNSPLIT_BATCH");
            }
            r?;
            fails += chk.verdict();
        }
    }

    // ---- ARM 4: B=1 PER-STAGE FAST PATH vs the EAGER stage-split (decode_step_h_ppn) ----
    // Its own reference, because its bar is a DIFFERENT one. Arms 1-2 pin b1_fast OFF so B=1
    // compares batched-vs-batched; that is what makes them a clean stage-split test, but it
    // means they never execute the path a solo serving session actually takes once the door is
    // open. The B=1 stage-fast path routes each stage's range through `decode_layers_eager` —
    // the SAME per-stage call `decode_step_h_ppn` makes on the same fence with the same
    // engines/streams/slots — so the bar here is BIT-IDENTITY TO THE EAGER SPLIT ARM, not to
    // the batched body (against which it carries the accepted m=1 fusion FP gap by design; see
    // decode_batch.rs `b1_stage_fast`). Both arms run over pp::new_cache placements, and both
    // have the door open, so the only difference is which public entry point is called.
    // WHY THIS ARM EARNS ITS KEEP: it is the only gate that would catch the stage-fast branch
    // wiring the wrong fence range, reusing stage 0's engine for a later range, or advancing
    // pos twice — mistakes that leave arms 1-3 fully green because they never run it.
    {
        let mut chk = BitCheck::new("b1-stagefast vs eager-ppn B=1".to_string());
        let prompt: Vec<u32> = (0..24u32).map(|j| 55 + seed * 13 + j * 31).collect();
        let n_s = steps.min(16);
        // b1_fast ON is the whole point of the arm (arms 1-2 left it OFF).
        HybridModel::set_b1_fast(true);
        let mut c_eager = memra_engine::pp::new_cache(e, &model.cfg, ctx)?;
        let _ = model.prime_cache(e, &prompt, &mut c_eager)?;
        let mut c_batch = memra_engine::pp::new_cache(e, &model.cfg, ctx)?;
        let _ = model.prime_cache(e, &prompt, &mut c_batch)?;
        let mut tok = *prompt.last().unwrap();
        for s in 0..n_s {
            let (ref_row, _) = model.decode_step_h(e, tok, &mut c_eager)?;
            let got = {
                let mut refs: Vec<&mut Cache> = vec![&mut c_batch];
                model.decode_step_batch(e, &[tok], &mut refs)?
            };
            chk.check(s, 0, &got[0], &ref_row);
            // Advance on the REFERENCE's argmax so both arms stay on one token stream; a
            // divergence shows up as differing bits, not as two arms exploring different text.
            tok = argmax(&ref_row) as u32;
            assert_eq!(c_eager.pos, c_batch.pos,
                       "b1-stagefast pos {} != eager pos {} at step {s} — one arm advanced \
                        the cache differently", c_batch.pos, c_eager.pos);
        }
        fails += chk.verdict();
        // Re-pin OFF: arm 3 (below) compares batched-vs-batched sampled/lean rows and must not
        // have one of its two caches on the m=1 fusion side of the accepted FP gap.
        HybridModel::set_b1_fast(false);
    }

    // ---- ARM 3: the LAST-STAGE epilogue (device sampling + lean park) ----
    // Runs at the widest requested B, on the split path, with MIXED metas so the partial-D2H
    // path is exercised: even rows device-sampled greedy, odd rows host rows.
    {
        let b = *batches.iter().max().unwrap();
        let prompts: Vec<Vec<u32>> = (0..b)
            .map(|i| (0..20 + i as u32 * 5)
                 .map(|j| 55 + seed * 13 + i as u32 * 97 + j * 31).collect())
            .collect();
        let n_s = steps.min(8);
        let mut ep_fail = 0usize;
        let mut caches_f: Vec<Cache> = Vec::with_capacity(b);
        let mut caches_l: Vec<Cache> = Vec::with_capacity(b);
        for p in prompts.iter() {
            let mut c = memra_engine::pp::new_cache(e, &model.cfg, ctx)?;
            let _ = model.prime_cache(e, p, &mut c)?;
            caches_f.push(c);
            let mut c = memra_engine::pp::new_cache(e, &model.cfg, ctx)?;
            let _ = model.prime_cache(e, p, &mut c)?;
            caches_l.push(c);
        }
        let mut toks: Vec<u32> = prompts.iter().map(|p| *p.last().unwrap()).collect();
        for _ in 0..n_s {
            let samp: Vec<Option<(f32, u64, u32)>> = (0..b)
                .map(|bi| if bi % 2 == 0 { Some((0.0, 0, 0)) } else { None })
                .collect();
            let (rows_f, next_f) = {
                let mut refs: Vec<&mut Cache> = caches_f.iter_mut().collect();
                model.decode_step_batch_sampled_lean(e, &toks, &mut refs, &samp, false)?
            };
            let (rows_l, next_l) = {
                let mut refs: Vec<&mut Cache> = caches_l.iter_mut().collect();
                model.decode_step_batch_sampled_lean(e, &toks, &mut refs, &samp, true)?
            };
            for bi in 0..b {
                if samp[bi].is_some() {
                    let host_am = argmax(&rows_f[bi]) as u32;
                    let dev = next_f[bi].expect("split greedy row missing device token");
                    if dev != host_am {
                        println!("pp gate epilogue row {bi}: device argmax {dev} != host \
                                  argmax {host_am} FAIL");
                        ep_fail += 1;
                    }
                    if next_l[bi] != next_f[bi] {
                        println!("pp gate epilogue row {bi}: lean token {:?} != full token \
                                  {:?} FAIL", next_l[bi], next_f[bi]);
                        ep_fail += 1;
                    }
                    if !rows_l[bi].is_empty() {
                        println!("pp gate epilogue row {bi}: lean sampled row NOT empty FAIL");
                        ep_fail += 1;
                    }
                    // Parked on the LAST STAGE's device under the split — the D2H reads it
                    // through UVA from the primary context, which is the same thing the
                    // server's retire path does.
                    let parked = e.dtoh(caches_l[bi].last_logits_dev.as_ref()
                                        .expect("lean row missing device park"))?;
                    let r = &rows_f[bi];
                    if !(parked.len() == r.len()
                         && parked.iter().zip(r.iter()).all(|(a, b)| a.to_bits() == b.to_bits())) {
                        println!("pp gate epilogue row {bi}: parked device logits != full \
                                  host row FAIL");
                        ep_fail += 1;
                    }
                    toks[bi] = host_am;
                } else {
                    let (r, l) = (&rows_f[bi], &rows_l[bi]);
                    if !(r.len() == l.len()
                         && r.iter().zip(l.iter()).all(|(a, b)| a.to_bits() == b.to_bits())) {
                        println!("pp gate epilogue row {bi}: unsampled row lean != full FAIL");
                        ep_fail += 1;
                    }
                    toks[bi] = argmax(r) as u32;
                }
            }
            if ep_fail > 8 { break; }
        }
        println!("pp gate {} [epilogue B={b}]: last-stage device sampling + lean park",
                 if ep_fail == 0 { "PASS" } else { "FAIL" });
        fails += usize::from(ep_fail > 0);
    }

    // Restore the live default (arms 1-3 pinned it OFF, arm 4 flipped it): this process may
    // run further gates, and a leaked pin would silently re-tier them.
    HybridModel::set_b1_fast(b1_live);
    println!("pp mode verdict: {fails} failing arm(s); {knobs}");
    Ok(fails)
}
