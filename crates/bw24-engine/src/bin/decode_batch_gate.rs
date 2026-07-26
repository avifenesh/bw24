//! decode-batch-gate: the batched decode step's exactness battery (ARCHITECTURE-H100.md §3 B2').
//!
//! decode_step_batch is a NEW NUMERIC CONFIG vs decode_step_h's fused m=1 tier: the fused
//! path folds q8_1 scales as a separable post-op (matmul_pre_noscale + silu_mul_scaled),
//! which is m==1-only by construction; the batched path folds scales inside the matvec
//! (matmul_pre) + plain silu_mul. Same math, different FP composition — the GDN-chunked
//! prefill precedent. PROOF the plumbing is exact: under the equalized composition
//! (BW24_MMVQ=0 BW24_NO_FUSE_NORMQ=1, both paths on dp4a + unfused norms) the battery is
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

use bw24_engine::cache::Cache;
use bw24_engine::forward::argmax;
use bw24_engine::hybrid::HybridModel;
use bw24_engine::Engine;
use bw24_gguf::GgufFile;

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
    unsafe { std::env::set_var("BW24_GDN_MMA", "0"); }
    // Same rationale for the l2 prefill v2 config (round 27): its primed state shifts
    // the same near-tie logits at step 1. Gate tests DECODE; prime stays pinned f32-class.
    unsafe { std::env::set_var("BW24_L2_V2", "0"); }
    unsafe { std::env::set_var("BW24_FA3", "0"); }
    let e = Engine::new(0)?;
    let g = GgufFile::open(&path)?;
    let model = HybridModel::load_without_mtp(&e, &g)?;
    println!("loaded {} ({} layers); steps={steps} batch={b_n}",
             g.arch().unwrap_or("?"), model.layers.len());

    // Distinct prompts per lane so caches/states genuinely diverge; length >= 16
    // (PRIME_MIN_T floor) and deliberately uneven so positions differ across the batch.
    let prompts: Vec<Vec<u32>> = (0..b_n.max(2))
        .map(|i| (0..20 + i as u32 * 5).map(|j| 55 + i as u32 * 97 + j * 31).collect())
        .collect();
    let ctx = 512 + steps + 64;

    // ---- Gate 1: B=1 bit-identity ----
    let mut c_ref = Cache::new(&e, &model.cfg, ctx)?;
    let mut c_bat = Cache::new(&e, &model.cfg, ctx)?;
    let _ = model.prime_cache(&e, &prompts[0], &mut c_ref)?;
    let _ = model.prime_cache(&e, &prompts[0], &mut c_bat)?;
    let mut t_ref = *prompts[0].last().unwrap();
    let mut t_bat = t_ref;
    let mut g1_fail = 0usize;
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
        if t_ref != t_bat {
            if strict || s < 16 {
                println!("gate1 step {s}: token diverged FAIL");
                g1_fail += 1;
            } else {
                println!("gate1 step {s}: token diverged — accepted cross-config drift (WARN)");
            }
            break;
        }
    }
    println!("gate1 (B=1 {} vs decode_step_h, {steps} steps): {}",
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

    if g1_fail + g2_fail == 0 {
        println!("ALL GREEN: decode_step_batch exactness battery");
        Ok(())
    } else {
        Err("decode-batch-gate FAILED".into())
    }
}
