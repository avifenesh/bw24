//! CPU validation gate for the modelopt-NVFP4 -> bw24 GGUF NVFP4 repack (NO GPU).
//!
//! Opens an HF (compressed-tensors NVFP4) checkpoint, picks a few quantized Linear tensors, and for
//! each cross-checks the two CPU dequants element-for-element:
//!   modelopt ref:  e2m1_code * raw_ue4m3(weight_scale[e/16])         (the HF-native arithmetic)
//!   bw24 internal: dequant the repacked GGUF block_nvfp4 row          (what the kernel decodes)
//! The two must agree to rel < 1e-3 (they are, in fact, bit-identical arithmetic — the doubled-code
//! / halved-scale conventions cancel). Also reports load timing for the full repack of a big tensor.
//!
//! Usage: nvfp4-validate <hf_dir>   (defaults to the 27B pure-NVFP4 model on this box)

use std::time::Instant;
use bw24_gguf::nvfp4_repack::{dequant_gguf_row, dequant_modelopt_row, repack_modelopt_to_gguf};
use bw24_gguf::safetensors::StModel;

fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| {
        "/data/ai-ml/hf-models/qwen36-27b-text-nvfp4-mtp-hf".to_string()
    });
    let m = StModel::open(std::path::Path::new(&dir)).expect("open safetensors");
    println!("opened {dir}  ({} tensors)", m.n_tensors());

    // A spread of quantized Linears: a self_attn proj (full-attn layer), an MLP proj, and an SSM
    // in_proj (linear-attn layer) — covers the three NVFP4 weight families in the checkpoint.
    let names = [
        "model.language_model.layers.3.self_attn.q_proj",
        "model.language_model.layers.3.mlp.down_proj",
        "model.language_model.layers.3.mlp.gate_proj",
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
        "model.language_model.layers.0.linear_attn.out_proj",
    ];

    let mut worst_rel = 0f32;
    let mut checked = 0usize;
    for stem in names {
        let wname = format!("{stem}.weight");
        let sname = format!("{stem}.weight_scale");
        let (winfo, wbytes) = match m.raw(&wname) {
            Some(r) => r,
            None => {
                println!("SKIP {stem}: no .weight");
                continue;
            }
        };
        let (sinfo, sbytes) = m.raw(&sname).expect("weight_scale sibling");
        assert_eq!(winfo.dtype, "U8", "{wname} not U8");
        assert_eq!(sinfo.dtype, "F8_E4M3", "{sname} not F8_E4M3");
        let out_f = winfo.shape[0] as usize;
        let in_f = (winfo.shape[1] as usize) * 2;

        // Repack the WHOLE tensor (timed) — proves the per-tensor load cost.
        let t0 = Instant::now();
        let packed = repack_modelopt_to_gguf(wbytes, sbytes, out_f, in_f);
        let dt = t0.elapsed();
        let row_bytes = (in_f / 64) * 36;
        assert_eq!(packed.len(), out_f * row_bytes);

        // Cross-check a sample of rows (every 257th, capped) element-for-element.
        let in_bytes = in_f / 2;
        let scl_bytes = in_f / 16;
        let mut rel = 0f32;
        let mut rows_checked = 0usize;
        let step = (out_f / 64).max(1);
        for o in (0..out_f).step_by(step) {
            let mref = dequant_modelopt_row(
                &wbytes[o * in_bytes..(o + 1) * in_bytes],
                &sbytes[o * scl_bytes..(o + 1) * scl_bytes],
                in_f,
            );
            let ggu = dequant_gguf_row(&packed[o * row_bytes..(o + 1) * row_bytes], in_f);
            for e in 0..in_f {
                let denom = mref[e].abs().max(1e-6);
                rel = rel.max((mref[e] - ggu[e]).abs() / denom);
            }
            rows_checked += 1;
        }
        worst_rel = worst_rel.max(rel);
        checked += 1;
        println!(
            "OK {stem}  [out={out_f}, in={in_f}]  rows_checked={rows_checked}  max_rel={rel:.2e}  \
             repack {:.1} MB in {:?}",
            packed.len() as f64 / 1e6,
            dt
        );
    }
    println!("\n=== {checked} tensors checked, worst rel = {worst_rel:.2e} (gate: rel < 1e-3) ===");
    assert!(worst_rel < 1e-3, "repack rel {worst_rel:.2e} exceeds 1e-3 gate");
    println!("PASS");
}
