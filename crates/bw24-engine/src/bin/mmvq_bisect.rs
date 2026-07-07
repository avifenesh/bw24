//! M3 MMVQ divergence bisect: per-layer residuals, decode with BW24_MMVQ off vs on,
//! SAME process / SAME loaded model (env is read at dispatch time). The layer where
//! maxdiff first jumps names the arm; combined with the bit-exact kernel-pair probe
//! (mmvq_pair_m3) this isolates the norm/quantize pairing site.
use bw24_engine::Engine;
use bw24_engine::hybrid::HybridModel;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: mmvq_bisect <hf_dir>");
    assert!(std::env::var("BW24_MMVQ").is_err(), "run WITHOUT BW24_MMVQ set");
    let e = Engine::new(0)?;
    let st = bw24_gguf::source::SafetensorsSource::open(std::path::Path::new(&path))?;
    let model = HybridModel::load_from_source(&e, &st)?;
    let n_layer = model.cfg.n_layer as usize;
    let all: Vec<usize> = (0..n_layer).collect();
    // prompt prefix so the divergence token has realistic state: first 6 tokens of the
    // merge-sorted-lists chat prompt (divergence observed at step 6).
    let prefix = [60u32, 124, 324, 22242, 109675, 3995];

    let run = |mmvq: bool| -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        unsafe { if mmvq { std::env::set_var("BW24_MMVQ", "1"); } else { std::env::remove_var("BW24_MMVQ"); } }
        let mut cache = bw24_engine::cache::Cache::new(&e, &model.cfg, 32)?;
        for &t in &prefix[..prefix.len() - 1] {
            let _ = model.decode_step(&e, t, &mut cache)?;
        }
        let (logits, aux) = model.decode_step_aux(&e, prefix[prefix.len() - 1], &mut cache, &all)?;
        let am = logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        println!("[mmvq={}] argmax={} top-logit={:.4}", mmvq as u8, am, logits[am]);
        let mut out = Vec::with_capacity(n_layer + 1);
        for a in &aux { out.push(e.dtoh(a)?); }
        out.push(logits);
        Ok(out)
    };

    let base = run(false)?;
    let fast = run(true)?;
    println!("per-layer residual maxdiff (no-MMVQ vs MMVQ, decode step {}):", prefix.len());
    let mut first: Option<usize> = None;
    for il in 0..n_layer {
        let md = base[il].iter().zip(&fast[il]).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        if md > 0.0 && first.is_none() { first = Some(il); }
        if md > 1e-3 || il < 2 || il == n_layer - 1 || Some(il) == first {
            println!("  L{il:2}: maxdiff={md:.3e}{}", if Some(il) == first { "   <-- FIRST NONZERO" } else { "" });
        }
    }
    let lm = base[n_layer].iter().zip(&fast[n_layer]).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    println!("logits maxdiff={lm:.3e}");
    Ok(())
}
