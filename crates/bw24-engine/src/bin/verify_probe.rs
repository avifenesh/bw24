//! Verify-vs-decode divergence probe (spec exactness triage). Feeds a prompt + a committed token
//! prefix eagerly (decode_step_h — the gold path), snapshots the cache, then computes the logits
//! for the SAME next token through four shapes: eager T=1 decode, verify T=1, verify T=2, verify
//! T=3 (rolling back between). Prints per-shape argmax + the top-2 logit values + max |diff| vs
//! eager, so an exactness failure can be attributed to a T-shape and sized (ULP tie vs state bug).
//!
//! Usage: BW24_PROMPT='...' verify-probe <model.gguf> <prefix tok ids...> -- <probe tok> [f1 f2]
//!   The prefix tokens are the GOLD generated tokens up to the divergence; probe tok = the token
//!   whose next-token logits diverge; f1/f2 = the two competing next-token ids to print.

use bw24_engine::Engine;
use bw24_engine::hybrid::HybridModel;
use bw24_engine::forward::argmax;
use bw24_gguf::GgufFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args.first().expect("usage: verify-probe <model> <prefix..> -- <probe> [f1 f2]");
    let sep = args.iter().position(|a| a == "--").expect("need -- <probe tok>");
    let prefix: Vec<u32> = args[1..sep].iter().filter_map(|s| s.parse().ok()).collect();
    let tail: Vec<u32> = args[sep + 1..].iter().filter_map(|s| s.parse().ok()).collect();
    let probe = tail[0];
    let watch: Vec<u32> = tail[1..].to_vec();

    let e = Engine::new(0)?;
    let g = GgufFile::open(path)?;
    let model = HybridModel::load(&e, &g)?;
    let prompt: Vec<u32> = if let Ok(text) = std::env::var("BW24_PROMPT") {
        let tok = bw24_tokenizer::Tokenizer::from_gguf(&g)?;
        tok.encode(&text, true)
    } else {
        (101..=228).collect()
    };
    println!("prompt {} toks, prefix {} toks, probe {probe}, watch {watch:?}", prompt.len(), prefix.len());

    let n_vocab = model.output.out_features();
    let max_ctx = prompt.len() + prefix.len() + 16;
    let mut cache = bw24_engine::cache::Cache::new(&e, &model.cfg, max_ctx)?;
    for &t in prompt.iter().chain(prefix.iter()) {
        let _ = model.decode_step_h(&e, t, &mut cache)?;
    }
    let pos = cache.pos;
    let snap = cache.snapshot(&e)?;
    println!("state primed to pos={pos}");

    let report = |name: &str, l: &[f32], base: Option<&[f32]>| {
        let am = argmax(l);
        let mut top: Vec<(usize, f32)> = l.iter().cloned().enumerate()
            .map(|(i, v)| (i, v)).collect();
        top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let md = base.map(|b| l.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max));
        print!("{name}: argmax={am}  top2=({}:{:.6}, {}:{:.6})", top[0].0, top[0].1, top[1].0, top[1].1);
        for &w in &watch { print!("  l[{w}]={:.6}", l[w as usize]); }
        if let Some(md) = md { print!("  maxdiff_vs_eager={md:.3e}"); }
        println!();
    };

    // A: eager decode (the gold path)
    let (le, _h) = model.decode_step_h(&e, probe, &mut cache)?;
    report("eager T=1      ", &le, None);
    cache.rollback(&e, &snap, 0)?;

    // B: verify T=1
    let (l1, _) = model.decode_step_t_h(&e, &[probe], pos, &mut cache)?;
    report("verify T=1 col0", &l1[0..n_vocab], Some(&le));
    cache.rollback(&e, &snap, 0)?;

    // C: verify T=2 (probe + eager's argmax as filler)
    let filler = argmax(&le) as u32;
    let (l2, _) = model.decode_step_t_h(&e, &[probe, filler], pos, &mut cache)?;
    report("verify T=2 col0", &l2[0..n_vocab], Some(&le));
    cache.rollback(&e, &snap, 0)?;

    // D: verify T=3
    let (l3, _) = model.decode_step_t_h(&e, &[probe, filler, filler], pos, &mut cache)?;
    report("verify T=3 col0", &l3[0..n_vocab], Some(&le));
    cache.rollback(&e, &snap, 0)?;

    // E: PER-LAYER bisect — eager aux residuals vs verify(T=1) aux residuals for the same token.
    // First layer with nonzero maxdiff = where the chains stop being bit-identical.
    let n_layer = model.cfg.n_layer as usize;
    let all: Vec<usize> = (0..n_layer).collect();
    let (_l, aux_e) = model.decode_step_aux(&e, probe, &mut cache, &all)?;
    cache.rollback(&e, &snap, 0)?;
    let (_l2, aux_v, _) = model.decode_step_t_aux2(&e, &[probe], pos, &mut cache, &all, None)?;
    cache.rollback(&e, &snap, 0)?;

    // E2: PER-LAYER bisect at T=2 col0 (2026-07-06, the 35B blind spot): verify [probe, filler]
    // col-0 per-layer rows vs the same eager aux — the first layer whose col-0 row diverges
    // names the kernel family whose T=2 dispatch breaks the decode-exact contract.
    {
        let (_lt2, _last2, aux_p) =
            model.decode_step_t_aux2(&e, &[probe, filler], pos, &mut cache, &all, Some(0))?;
        cache.rollback(&e, &snap, 0)?;
        match aux_p {
            Some(aux_p) => {
                println!("per-layer col0 residual maxdiff at T=2 (vs eager) — first nonzero = the bug:");
                let mut shown = 0;
                for il in 0..n_layer.min(aux_p.len()) {
                    let a = e.dtoh(&aux_e[il])?;
                    let b = e.dtoh(&aux_p[il])?;
                    let md = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
                    let kind = match &model.layers[il].mixer {
                        bw24_engine::hybrid::Mixer::Full(_) => "full",
                        bw24_engine::hybrid::Mixer::Linear(_) => "lin ",
                    };
                    if md != 0.0 || il < 3 || il == n_layer - 1 {
                        println!("  T2 layer {il:2} [{kind}]: maxdiff={md:.3e}");
                        if md != 0.0 { shown += 1; if shown >= 6 { break; } }
                    }
                }
            }
            None => println!("T=2 aux: pred_col rows unavailable"),
        }
    }
    // F: NORM+QUANT pair check on each layer's true input: fused rms_norm_q8_1 (eager) vs
    // rms_norm_decode + quantize_q8_1 (verify). Compares int8 bytes + block scales bitwise.
    let n_embd = model.cfg.n_embd as usize;
    let eps = model.cfg.rms_eps;
    println!("norm+quant pair check (fused vs unfused) on per-layer inputs:");
    for il in 0..aux_e.len().min(8) {
        let x_in = &aux_e[il]; // input to layer il+1
        let w = model.layers[il + 1].attn_norm.float_data();
        let (hq_f, hd_f) = e.rms_norm_q8_1(x_in, w, n_embd, 1, eps)?;
        let mut h = e.zeros(n_embd)?;
        e.rms_norm_decode(x_in, w, &mut h, n_embd, 1, eps)?;
        let (hq_u, hd_u) = e.quantize_q8_1(&h, 1, n_embd)?;
        let qf: Vec<i8> = e.stream().clone_dtoh(&hq_f)?;
        let qu: Vec<i8> = e.stream().clone_dtoh(&hq_u)?;
        e.stream().synchronize()?;
        let df = e.dtoh(&hd_f)?;
        let du = e.dtoh(&hd_u)?;
        let q_mm = qf.iter().zip(&qu).filter(|(a, b)| a != b).count();
        let d_mm = df.iter().zip(&du).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!("  layer {:2} input: int8 mismatches={q_mm}/{}  scale maxdiff={d_mm:.3e}", il + 1, qf.len());
    }

    // G: FFN chain pair check on real data: EAGER fused (add_rms_norm_q8_1 -> dual_noscale ->
    // silu_mul_scaled_q8_1 -> matmul_pre(down)) vs VERIFY decode-exact (add -> rms_norm_decode ->
    // matmul_decode_exact x3 + silu_mul). x = aux_e[1], mixed = aux_e[3] (real-valued stand-ins).
    {
        use bw24_engine::hybrid::Ffn;
        let x_in = &aux_e[1];
        let mixed = &aux_e[3];
        for il in [2usize, 4, 5] {
            if let Ffn::Dense { ffn_gate, ffn_up, ffn_down } = &model.layers[il].ffn {
                let n_ff = ffn_gate.out_features();
                let pnorm = model.layers[il].post_attn_norm.float_data();
                // eager fused chain
                let mut x1f = e.zeros(n_embd)?;
                let (zqf, zdf) = e.add_rms_norm_q8_1(x_in, mixed, pnorm, &mut x1f, n_embd, 1, eps)?;
                let (gate, gs, up, us) = match e.matmul_pre_dual_noscale(ffn_gate, ffn_up, &zqf, &zdf, 1)? {
                    Some(((g, gsv), (u, usv))) => (g, gsv, u, usv),
                    None => { println!("  layer {il}: dual_noscale None (not NVFP4 pair) — skipping"); continue; }
                };
                let (aqf, adf) = e.silu_mul_scaled_q8_1(&gate, &up, gs, us, n_ff)?;
                let ff_f = e.matmul_pre(ffn_down, &aqf, &adf, &gate, 1)?;
                // verify decode-exact chain
                let mut x1u = e.zeros(n_embd)?;
                e.add(x_in, mixed, &mut x1u, n_embd)?;
                let mut z = e.zeros(n_embd)?;
                e.rms_norm_decode(&x1u, pnorm, &mut z, n_embd, 1, eps)?;
                let gu = e.matmul_decode_exact(ffn_gate, &z, 1)?;
                let uu = e.matmul_decode_exact(ffn_up, &z, 1)?;
                let mut act = e.zeros(n_ff)?;
                e.silu_mul(&gu, &uu, &mut act, n_ff)?;
                let ff_u = e.matmul_decode_exact(ffn_down, &act, 1)?;
                // compare stage by stage
                let hf = e.dtoh(&ff_f)?; let hu = e.dtoh(&ff_u)?;
                let md_out = hf.iter().zip(&hu).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                let gf = e.dtoh(&gate)?; let guh = e.dtoh(&gu)?;
                // eager gate is UNSCALED (gs separate); scale before compare
                let md_gate = gf.iter().map(|v| v * gs).zip(&guh).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                let qf2: Vec<i8> = e.stream().clone_dtoh(&zqf)?;
                let (aq_u, _ad_u) = e.quantize_q8_1(&z, 1, n_embd)?;
                let qu2: Vec<i8> = e.stream().clone_dtoh(&aq_u)?;
                e.stream().synchronize()?;
                let z_mm = qf2.iter().zip(&qu2).filter(|(a, b)| a != b).count();
                let aqf_h: Vec<i8> = e.stream().clone_dtoh(&aqf)?;
                let (aq_v, _) = e.quantize_q8_1(&act, 1, n_ff)?;
                let aqv_h: Vec<i8> = e.stream().clone_dtoh(&aq_v)?;
                e.stream().synchronize()?;
                let a_mm = aqf_h.iter().zip(&aqv_h).filter(|(a, b)| a != b).count();
                println!("  ffn pair layer {il}: zq int8 mm={z_mm}  gate(scaled) maxdiff={md_gate:.3e}  actq int8 mm={a_mm}  ffn_out maxdiff={md_out:.3e}");
            }
        }
    }

    // H: DETERMINISM/STATE-RESTORE check — the SAME path twice with rollback between. Any nonzero
    // diff here means rollback does not fully restore state (or the path is nondeterministic),
    // and the eager-vs-verify comparison above is contaminated.
    {
        let (_l, aux_e2) = model.decode_step_aux(&e, probe, &mut cache, &all)?;
        cache.rollback(&e, &snap, 0)?;
        let mut worst = (0usize, 0.0f32);
        for il in 0..aux_e.len() {
            let a = e.dtoh(&aux_e[il])?;
            let b = e.dtoh(&aux_e2[il])?;
            let md = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
            if md > worst.1 { worst = (il, md); }
        }
        println!("eager-vs-eager (rollback between): worst layer {} maxdiff={:.3e}", worst.0, worst.1);
        let (_l3, aux_v2, _) = model.decode_step_t_aux2(&e, &[probe], pos, &mut cache, &all, None)?;
        cache.rollback(&e, &snap, 0)?;
        let mut worst2 = (0usize, 0.0f32);
        for il in 0..aux_v.len() {
            let a = e.dtoh(&aux_v[il])?;
            let b = e.dtoh(&aux_v2[il])?;
            let md = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
            if md > worst2.1 { worst2 = (il, md); }
        }
        println!("verify-vs-verify (rollback between): worst layer {} maxdiff={:.3e}", worst2.0, worst2.1);
    }

    // I0: per-layer per-tensor q8_1-fast map — non-fast (Float) projections flip eager decode to
    // the UNFUSED branch (rms_norm 256-thread) and matmul()/cublas paths; verify must mirror.
    {
        use bw24_engine::hybrid::{Mixer, Ffn};
        println!("per-layer fastness (mixer projections + ffn gate/up):");
        for (il, layer) in model.layers.iter().enumerate() {
            let mix = match &layer.mixer {
                Mixer::Full(fa) => vec![("wq", e.uses_q8_1_fast(&fa.wq)), ("wk", e.uses_q8_1_fast(&fa.wk)),
                                        ("wv", e.uses_q8_1_fast(&fa.wv)), ("wo", e.uses_q8_1_fast(&fa.wo))],
                Mixer::Linear(la) => vec![("wqkv", e.uses_q8_1_fast(&la.wqkv)), ("gate", e.uses_q8_1_fast(&la.wqkv_gate)),
                                          ("beta", e.uses_q8_1_fast(&la.ssm_beta)), ("alpha", e.uses_q8_1_fast(&la.ssm_alpha)),
                                          ("out", e.uses_q8_1_fast(&la.ssm_out))],
            };
            let ffn = match &layer.ffn {
                Ffn::Dense { ffn_gate, ffn_up, ffn_down } =>
                    vec![("fg", e.uses_q8_1_fast(ffn_gate)), ("fu", e.uses_q8_1_fast(ffn_up)), ("fd", e.uses_q8_1_fast(ffn_down))],
                Ffn::Moe(_) => vec![("moe", false)],
            };
            let slow: Vec<&str> = mix.iter().chain(ffn.iter()).filter(|(_, f)| !f).map(|(n, _)| *n).collect();
            if !slow.is_empty() { println!("  layer {il:2}: NON-FAST {slow:?}"); }
        }
    }

    println!("per-layer residual maxdiff (eager decode_step_aux vs verify decode_step_t_aux2 T=1):");
    let n_layer = aux_e.len();
    for il in 0..n_layer {
        let a = e.dtoh(&aux_e[il])?;
        let b = e.dtoh(&aux_v[il])?;
        let md = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        let kind = match &model.layers[il].mixer {
            bw24_engine::hybrid::Mixer::Full(_) => "full",
            bw24_engine::hybrid::Mixer::Linear(_) => "lin ",
        };
        if md != 0.0 || il < 3 || il == n_layer - 1 {
            println!("  layer {il:2} [{kind}]: maxdiff={md:.3e}");
        }
    }
    Ok(())
}
