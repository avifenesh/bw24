//! CPU validation gate for the HF-NVFP4 -> bw24 GGUF NVFP4 repack (NO GPU). Handles BOTH on-disk
//! encodings: modelopt compressed-tensors (`.weight`/`.weight_scale`/`.weight_scale_2`) AND the Reza
//! "custom_nvfp4_e2m1_e4m3_scales" format (`.weight.nvfp4_packed`/`.weight.nvfp4_scale_e4m3`). Both
//! encode the SAME e2m1 weights + per-16 FP8(e4m3) scales, so the repack/dequant byte math is shared.
//!
//! For each picked quantized Linear it cross-checks the two CPU dequants element-for-element:
//!   HF ref:        e2m1_code * raw_ue4m3(per16_scale[e/16])          (the HF-native arithmetic)
//!   bw24 internal: dequant the repacked GGUF block_nvfp4 row          (what the kernel decodes)
//! The two must agree to rel < 1e-3 (they are, in fact, bit-identical arithmetic — the doubled-code
//! / halved-scale conventions cancel). Also reports load timing for the full repack of a big tensor.
//!
//! Usage: nvfp4-validate <hf_dir> [tensor_stem...]   (defaults to the 27B modelopt model on this box;
//! the stems default per detected format). The packed tensor's safetensors file is auto-located.

use std::time::Instant;
use bw24_gguf::nvfp4_repack::{dequant_gguf_row, dequant_modelopt_row, repack_modelopt_to_gguf};
use bw24_gguf::safetensors::StModel;

/// Resolve the (packed_bytes, scale_bytes, out_f, in_f) for a Linear `stem` under EITHER format.
/// modelopt: `<stem>.weight`(U8) + `<stem>.weight_scale`(F8_E4M3). Reza: `<stem>.weight.nvfp4_packed`
/// + `<stem>.weight.nvfp4_scale_e4m3`. Returns None if neither set of siblings is present.
fn resolve<'a>(m: &'a StModel, stem: &str) -> Option<(&'a [u8], &'a [u8], usize, usize, &'static str)> {
    // modelopt
    if let Some((winfo, wbytes)) = m.raw(&format!("{stem}.weight")) {
        if winfo.dtype == "U8" {
            if let Some((_si, sbytes)) = m.raw(&format!("{stem}.weight_scale")) {
                let out_f = winfo.shape[0] as usize;
                let in_f = (winfo.shape[1] as usize) * 2;
                return Some((wbytes, sbytes, out_f, in_f, "modelopt"));
            }
        }
    }
    // Reza custom
    if let Some((winfo, wbytes)) = m.raw(&format!("{stem}.weight.nvfp4_packed")) {
        let (_si, sbytes) = m.raw(&format!("{stem}.weight.nvfp4_scale_e4m3"))?;
        let out_f = winfo.shape[0] as usize;
        let in_f = (winfo.shape[1] as usize) * 2;
        return Some((wbytes, sbytes, out_f, in_f, "reza"));
    }
    None
}

fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| {
        "/data/ai-ml/hf-models/qwen36-27b-text-nvfp4-mtp-hf".to_string()
    });
    let m = StModel::open(std::path::Path::new(&dir)).expect("open safetensors");
    println!("opened {dir}  ({} tensors)", m.n_tensors());

    // Stems to check. CLI overrides; else a default spread that covers the qwen modelopt families and
    // (when absent) silently SKIPs — so the same bin runs on a Reza checkpoint with `<dir> <stem...>`.
    let cli: Vec<String> = std::env::args().skip(2).collect();
    let default_stems = [
        "model.language_model.layers.3.self_attn.q_proj",
        "model.language_model.layers.3.mlp.down_proj",
        "model.language_model.layers.3.mlp.gate_proj",
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
        "model.language_model.layers.0.linear_attn.out_proj",
        // Reza gemma defaults (present only on that checkpoint; SKIP otherwise).
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.self_attn.o_proj",
    ];
    let names: Vec<String> = if cli.is_empty() {
        default_stems.iter().map(|s| s.to_string()).collect()
    } else {
        cli
    };

    let mut worst_rel = 0f32;
    let mut checked = 0usize;
    for stem in &names {
        let (wbytes, sbytes, out_f, in_f, fmt) = match resolve(&m, stem) {
            Some(r) => r,
            None => {
                println!("SKIP {stem}: no NVFP4 siblings (modelopt or reza)");
                continue;
            }
        };

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
            "OK [{fmt}] {stem}  [out={out_f}, in={in_f}]  rows_checked={rows_checked}  max_rel={rel:.2e}  \
             repack {:.1} MB in {:?}",
            packed.len() as f64 / 1e6,
            dt
        );
    }
    println!("\n=== {checked} tensors checked, worst rel = {worst_rel:.2e} (gate: rel < 1e-3) ===");
    assert!(worst_rel < 1e-3, "repack rel {worst_rel:.2e} exceeds 1e-3 gate");
    println!("PASS");
}
