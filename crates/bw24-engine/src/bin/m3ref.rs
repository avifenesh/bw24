//! M3 external-reference checks (bring-up): dump OUR embed row + layer-0 input norm for a token,
//! for comparison against the HF safetensors bytes read directly in Python.
use bw24_engine::Engine;
use bw24_engine::hybrid::HybridModel;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: m3ref <hf_dir> <tok>");
    let tok: u32 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(20768);
    let e = Engine::new(0)?;
    let st = bw24_gguf::source::SafetensorsSource::open(std::path::Path::new(&path))?;
    let model = HybridModel::load_from_source(&e, &st)?;
    let x = model.embed(&e, &[tok])?;
    let h = e.dtoh(&x)?;
    println!("our embed[{tok}][:8]: {:?}", &h[..8]);
    let norm: f32 = h.iter().map(|v| v * v).sum::<f32>().sqrt();
    println!("norm: {norm}");
    // norm-weight fold check: our loaded blk.5.attn_norm should be HF value + 1 (gemma fold)
    let nw = model.layers[5].attn_norm.float_data();
    let nh = e.dtoh(nw)?;
    println!("our blk.5.attn_norm[:3]: {:?} (HF raw was ~[-0.365, -0.393, -0.404]; +1 fold -> ~[0.635, 0.607, 0.596])", &nh[..3]);
    let qn = match &model.layers[5].mixer {
        bw24_engine::hybrid::Mixer::Full(fa) => e.dtoh(fa.q_norm.float_data())?,
        _ => vec![],
    };
    if !qn.is_empty() {
        println!("our blk.5.q_norm[:3]: {:?} (HF raw ~[0.332, 0.305, 0.301]; +1 -> ~[1.332, 1.305, 1.301])", &qn[..3]);
    }
    // hidden trace through layer 0-2 for token 20768 at T=1 (decode path): print per-layer
    // residual norms — a wrong weight class (e.g. swapped gate/up, bad repack) shows as an
    // exploding/collapsing norm long before the head.
    let mut cache = bw24_engine::cache::Cache::new(&e, &model.cfg, 8)?;
    let n_layer = model.cfg.n_layer as usize;
    let all: Vec<usize> = (0..n_layer).collect();
    let (logits, aux) = model.decode_step_aux(&e, tok, &mut cache, &all)?;
    for il in [0usize, 1, 2, 3, 10, 30, 59] {
        let a = e.dtoh(&aux[il])?;
        let nrm: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
        let amax = a.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        println!("L{il:2} residual norm {nrm:10.2} amax {amax:9.3}");
    }
    let mut top: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
    top.sort_by(|a, b| b.1.total_cmp(&a.1));
    println!("top5 logits: {:?}", &top[..5]);
    Ok(())
}
