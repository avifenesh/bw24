//! M3 T=2 divergence bisect: prefill T=2 residuals vs 2-chained-decode residuals per layer.
use bw24_engine::Engine;
use bw24_engine::hybrid::HybridModel;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: t2probe <hf_dir>");
    let e = Engine::new(0)?;
    let st = bw24_gguf::source::SafetensorsSource::open(std::path::Path::new(&path))?;
    let model = HybridModel::load_from_source(&e, &st)?;
    let toks = [9419u32, 11u32];
    let n_layer = model.cfg.n_layer as usize;
    let all: Vec<usize> = (0..n_layer).collect();
    // decode chain: token 0 then token 1, capture aux at token 1
    let mut cache = bw24_engine::cache::Cache::new(&e, &model.cfg, 16)?;
    let _ = model.decode_step(&e, toks[0], &mut cache)?;
    let (_l, aux_d) = model.decode_step_aux(&e, toks[1], &mut cache, &all)?;
    // verify T=2 col1 (same as prefill batched math): decode_step_t_aux2 on fresh cache
    let mut cache2 = bw24_engine::cache::Cache::new(&e, &model.cfg, 16)?;
    let (_l2, aux_v, _) = model.decode_step_t_aux2(&e, &toks, 0, &mut cache2, &all, None)?;
    println!("per-layer residual maxdiff (2-chained decode vs T=2 batched verify, last col):");
    for il in 0..n_layer {
        let a = e.dtoh(&aux_d[il])?;
        let b = e.dtoh(&aux_v[il])?;
        let md = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        if md > 1e-3 || il < 2 || il == n_layer - 1 {
            println!("  L{il:2}: maxdiff={md:.3e}");
            if md > 1.0 { break; }
        }
    }
    // FORWARD (prefill batched) per-layer x vs decode aux — replicate forward()'s loop inline.
    {
        use bw24_engine::hybrid::{Mixer, Ffn};
        let cfg = &model.cfg;
        let n_embd = cfg.n_embd as usize;
        let t = toks.len();
        let eps = cfg.rms_eps;
        let pos: Vec<i32> = (0..t as i32).collect();
        let pos_d = e.htod_i32(&pos)?;
        let mut x = model.embed(&e, &toks)?;
        println!("per-layer maxdiff (prefill forward last-row vs decode aux):");
        for (il, layer) in model.layers.iter().enumerate() {
            let mut h = e.zeros(t * n_embd)?;
            e.rms_norm(&x, layer.attn_norm.float_data(), &mut h, n_embd, t, eps)?;
            let mixed = match &layer.mixer {
                Mixer::Full(fa) => model.full_attn(&e, fa, &h, &pos_d, t)?,
                Mixer::Linear(la) => model.linear_attn(&e, la, &h, t)?,
            };
            let mut x1 = e.zeros(t * n_embd)?;
            e.add(&x, &mixed, &mut x1, t * n_embd)?;
            let mut z = e.zeros(t * n_embd)?;
            e.rms_norm(&x1, layer.post_attn_norm.float_data(), &mut z, n_embd, t, eps)?;
            let ffn_out = match &layer.ffn {
                Ffn::Dense { ffn_gate, ffn_up, ffn_down } => {
                    let n_ff = ffn_gate.out_features();
                    let gate = e.matmul(ffn_gate, &z, t)?;
                    let up = e.matmul(ffn_up, &z, t)?;
                    let mut act = e.zeros(t * n_ff)?;
                    HybridModel::ffn_act(&e, cfg, &gate, &up, &mut act, t * n_ff)?;
                    e.matmul(ffn_down, &act, t)?
                }
                Ffn::Moe(m) => model.moe_ffn_il(&e, m, &z, t, il as u16)?,
            };
            let mut x2 = e.zeros(t * n_embd)?;
            e.add(&x1, &ffn_out, &mut x2, t * n_embd)?;
            // diff LAST row vs decode aux
            let full = e.dtoh(&x2)?;
            let last = &full[(t - 1) * n_embd..t * n_embd];
            let a = e.dtoh(&aux_d[il])?;
            let md = last.iter().zip(&a).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
            if md > 1e-2 || il < 3 || il == n_layer - 1 {
                println!("  L{il:2}: maxdiff={md:.3e}");
                if md > 1.0 { break; }
            }
            x = x2;
        }
    }
    Ok(())
}
