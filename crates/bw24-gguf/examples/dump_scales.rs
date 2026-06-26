//! Throwaway: dump every "*.scale" / "*.input_scale" scalar tensor value, grouped by the
//! weight tensor's quant type. Finds which dtypes carry a per-tensor macro-scale that bw24
//! currently applies ONLY for NVFP4 (model.rs GpuTensor::load). If Q4_K/Q5_K scales != 1.0,
//! the hybrid forward drops them -> wrong logits.
use bw24_gguf::GgufFile;

fn main() {
    let path = std::env::args().nth(1).expect("usage: dump_scales <model.gguf>");
    let g = GgufFile::open(&path).unwrap();

    // Collect scale tensors and their owning weight's dtype.
    let mut by_owner_type: std::collections::BTreeMap<String, (usize, f32, f32, f32)> = Default::default();
    for t in &g.tensors {
        let (stem, kind) = if let Some(s) = t.name.strip_suffix(".input_scale") { (s, "input_scale") }
            else if let Some(s) = t.name.strip_suffix(".scale") { (s, "scale") }
            else { continue };
        // value (these are F32 ne=[1])
        let v = f32::from_le_bytes(g.tensor_data(t)[..4].try_into().unwrap());
        // owner weight dtype
        let owner = g.find(&format!("{stem}.weight"));
        let dty = owner.map(|o| format!("{:?}", o.ggml_type)).unwrap_or("?".into());
        let key = format!("{dty}/{kind}");
        let e = by_owner_type.entry(key).or_insert((0, f32::INFINITY, f32::NEG_INFINITY, 0.0));
        e.0 += 1; e.1 = e.1.min(v); e.2 = e.2.max(v); e.3 += v;
    }
    println!("scale tensors grouped by  ownerDtype/kind :  count  min  max  mean");
    for (k, (n, mn, mx, sum)) in &by_owner_type {
        println!("  {k:22}  n={n:<4} min={mn:.6} max={mx:.6} mean={:.6}", sum / *n as f32);
    }
    // Also: show a few concrete examples for blk.0
    println!("\nblk.0 examples:");
    for t in &g.tensors {
        if t.name.starts_with("blk.0.") && (t.name.ends_with(".scale") || t.name.ends_with(".input_scale")) {
            let v = f32::from_le_bytes(g.tensor_data(t)[..4].try_into().unwrap());
            println!("  {:40} = {v:.6}", t.name);
        }
    }
}
