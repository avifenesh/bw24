//! iso-gap-probe (lane/iso-gap, task #91): STAGGERED-DEPTH isolation differential at FIXED B.
//!
//! THE QUESTION (research/iso-gap-20260807/PROGRESS.md §1.5): does a co-resident session at a
//! DIFFERENT depth change session X's logits bits, at fixed batch width, on the batched body?
//! The serve receipt (spec-gate REF vs REF_LOAD, bytes 1347/2379) cannot attribute between:
//!   H-A  a rung-straddle (or other depth-coupled selection) breaking bit identity at fixed B;
//!   H-B  the documented alone-vs-batched config gap (B=1 fast path / graph replay vs batched
//!        body) plus B fluctuation — i.e. co-RESIDENCE flips the program, depth is innocent.
//!
//! METHOD. Both arms run decode_step_batch with the B=1 fast path pinned OFF (the gate2
//! precedent — set through the seam so the reference actually runs the batched body):
//!   REF   X alone (B=1, batched body): prime X to depth dx, decode N greedy steps, store
//!         every logits row (bits).
//!   TEST  X with Y (B=2): fresh caches primed IDENTICALLY (X to dx, Y to dy), decode N steps
//!         batched; X's row must be BIT-IDENTICAL to REF at every step.
//! Every step prints its depth pair + each row's OWN fa_split_keys rung + whether the seqs
//! (z-batched) FA arm fires (the decode_batch predicate, recomputed here from the public
//! twins), so a divergence is attributable to the exact step the arms' kernel programs split.
//!
//! ARMS (--dx/--dy; the 5090/q9 ladder has its live rung boundary at t_kv=512):
//!   control-same-rung  dx=300  dy=310   both sp8 throughout       -> expected PASS
//!   straddle           dx=480  dy=800   X sp8, Y sp64 for ~32 steps (batch straddles the
//!                                       512 rung; seqs arm off -> per-seq eager loop), then
//!                                       X crosses and the seqs arm re-fires -> the class
//!   straddle-reverse   dx=800  dy=480   X on sp64 both ways; Y crosses mid-run
//!   deep-control       dx=800  dy=810   both sp64 throughout      -> expected PASS
//!
//! Greedy only; token feedback per arm is its own argmax, so a first bit-diff step is also
//! checked for argmax flip (bit-diff without flip = FP-order leak below the tie margin;
//! bit-diff with flip = the serve-visible class).

use memra_engine::cache::Cache;
use memra_engine::forward::argmax;
use memra_engine::hybrid::HybridModel;
use memra_engine::Engine;
use memra_gguf::GgufFile;

fn synth(len: usize, salt: u32) -> Vec<u32> {
    // gate-binary pattern: small deterministic ids, model-agnostic (vocab floor ~10k).
    (0..len as u32).map(|j| 55 + salt * 97 + (j % 401) * 23).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect(
        "usage: iso-gap-probe <model.gguf> [--dx 480] [--dy 800] [--steps 96] [--ctx 4096]");
    let rest: Vec<String> = args.collect();
    let get = |k: &str, d: usize| -> usize {
        rest.iter().position(|a| a == k)
            .and_then(|i| rest.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(d)
    };
    let dx = get("--dx", 480);
    let dy = get("--dy", 800);
    let steps = get("--steps", 96);
    let ctx = get("--ctx", 4096);

    let e = Engine::new(0)?;
    let g = GgufFile::open(&path)?;
    let arch = g.arch().unwrap_or("?").to_string();
    let model = HybridModel::load_without_mtp(&e, &g)?;
    let cfg = &model.cfg;
    let nhkv = cfg.n_head_kv as usize;
    let hd = cfg.head_dim_k as usize;
    println!("loaded {arch} ({} layers, n_head_kv={nhkv}, head_dim={hd}); \
              dx={dx} dy={dy} steps={steps}", model.layers.len());

    // BOTH arms on the batched body (the gate2/gate3 precedent): the B=1 fast path routes
    // solo rows onto the m=1 fused trunk — a DIFFERENT documented config. This probe tests
    // depth-co-residence WITHIN one config, so the reference must run the same body.
    let b1_live = HybridModel::b1_fast_on();
    HybridModel::set_b1_fast(false);
    println!("B=1 reference arm: batched body (fast path pinned OFF; live default = {})",
             if b1_live { "ON" } else { "OFF" });

    let px = synth(dx, 1);
    let py = synth(dy, 5);
    let tx0 = *px.last().unwrap();
    let ty0 = *py.last().unwrap();

    // ---- REF: X alone, B=1 ----
    let mut c_ref = Cache::new(&e, cfg, ctx)?;
    let _ = model.prime_cache(&e, &px, &mut c_ref)?;
    let mut t = tx0;
    let mut ref_logits: Vec<Vec<f32>> = Vec::with_capacity(steps);
    let mut ref_toks: Vec<u32> = Vec::with_capacity(steps);
    for _ in 0..steps {
        let l = {
            let mut refs = [&mut c_ref];
            model.decode_step_batch(&e, &[t], &mut refs)?.remove(0)
        };
        t = argmax(&l) as u32;
        ref_toks.push(t);
        ref_logits.push(l);
    }
    drop(c_ref);

    // ---- TEST: X with Y, B=2, identically primed ----
    let mut c_x = Cache::new(&e, cfg, ctx)?;
    let _ = model.prime_cache(&e, &px, &mut c_x)?;
    let mut c_y = Cache::new(&e, cfg, ctx)?;
    let _ = model.prime_cache(&e, &py, &mut c_y)?;
    let mut toks = [tx0, ty0];
    let mut first_div: Option<usize> = None;
    let mut div_steps = 0usize;
    let mut flip_steps = 0usize;
    for s in 0..steps {
        // the decode_batch seqs-arm predicate, from the public twins (pre-step t_kv = pos+1
        // AFTER this step's append == pos+1 at entry +1; decode_batch computes it from
        // caches[i].pos + 1 at entry, which is the POST-append bound of this step)
        let tkx = c_x.pos + 1;
        let tky = c_y.pos + 1;
        let spx = memra_engine::fa_split_keys_pub(tkx, nhkv);
        let spy = memra_engine::fa_split_keys_pub(tky, nhkv);
        let seqs = memra_engine::fa_seqs_eligible(tkx, hd)
            && memra_engine::fa_seqs_eligible(tky, hd) && spx == spy;
        let logits = {
            let mut refs = [&mut c_x, &mut c_y];
            model.decode_step_batch(&e, &toks, &mut refs)?
        };
        let lx = &logits[0];
        let r = &ref_logits[s];
        let bits_eq = r.len() == lx.len()
            && r.iter().zip(lx.iter()).all(|(a, b)| a.to_bits() == b.to_bits());
        let am = argmax(lx) as u32;
        if !bits_eq {
            div_steps += 1;
            let md = r.iter().zip(lx.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            let nd = r.iter().zip(lx.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            let flip = am != ref_toks[s];
            if flip { flip_steps += 1; }
            if first_div.is_none() {
                first_div = Some(s);
            }
            if div_steps <= 8 || flip {
                println!("step {s}: X BIT-DIFF vs solo (t_kv X={tkx} sp{spx} / Y={tky} sp{spy}, \
                          seqs_arm={seqs}) ndiff={nd} maxdiff={md:.3e}{}",
                         if flip { format!(" ARGMAX FLIP {} -> {am}", ref_toks[s]) } else { String::new() });
            }
        }
        // token feedback: X follows ITS OWN argmax in this arm (the serve regime); a
        // post-flip trajectory diverges by construction, so report tracks bits AND flips.
        toks = [am, argmax(&logits[1]) as u32];
    }
    match first_div {
        None => println!("VERDICT dx={dx} dy={dy}: PASS — X bit-identical solo-vs-coresident, \
                          all {steps} steps"),
        Some(s) => println!("VERDICT dx={dx} dy={dy}: FAIL — first bit-diff at step {s} \
                             (t_kv X={}), {div_steps}/{steps} steps differ, {flip_steps} argmax \
                             flips", dx + s + 1),
    }
    HybridModel::set_b1_fast(b1_live);
    if first_div.is_some() { std::process::exit(1); }
    Ok(())
}
