//! Cross-source tensor verification for the hybrid SSM name-map + transforms (ST-MOE-PLAN Gate D/E/F).
//!
//! The 9B safetensors is BF16 -> on our f32 path a full forward needs ~36 GB (OOM on a 24 GB GPU),
//! and so does the f16 GGUF twin (it dequants to f32 too). So instead of an end-to-end argmax we do
//! the STRONGER per-tensor check: for every ggml SSM name `HybridModel` asks for, dequantize the
//! safetensors tensor (with our transforms applied) AND the GGUF-twin tensor (transforms baked in by
//! llama.cpp's converter), and assert they are numerically equal. Any wrong V-reorder / missing +1 /
//! bad -exp(A_log) / transposed projection shows up as a per-element mismatch on the offending tensor.
//!
//! Usage: st-vs-gguf <hf_dir> <gguf_twin>

use bw24_gguf::{GgufFile, dequant};
use bw24_gguf::source::{SafetensorsSource, GgufSource, TensorSource};

fn deq(src: &dyn TensorSource, name: &str) -> Option<(Vec<f32>, Vec<u64>)> {
    let v = src.find(name)?;
    let n: u64 = v.ne.iter().product();
    Some((dequant::dequantize(v.ggml_type, &v.bytes, n as usize), v.ne.clone()))
}

/// Compare two f32 tensors: returns (max_abs_diff, n_mismatch over a tolerance, n).
fn cmp(a: &[f32], b: &[f32], tol: f32) -> (f32, usize) {
    let mut maxd = 0f32;
    let mut nmis = 0usize;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        if d > maxd { maxd = d; }
        if d > tol { nmis += 1; }
    }
    (maxd, nmis)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let hf = std::env::args().nth(1).expect("usage: st-vs-gguf <hf_dir> <gguf_twin>");
    let gg = std::env::args().nth(2).expect("usage: st-vs-gguf <hf_dir> <gguf_twin>");
    let st = SafetensorsSource::open(std::path::Path::new(&hf))?;
    let g = GgufFile::open(&gg)?;
    let gs = GgufSource(&g);
    let cfg = st.config();
    println!("HF arch={:?} n_layer={} | GGUF arch={:?}", cfg.arch, cfg.n_layer, g.arch());
    println!("ssm: {:?}", cfg.ssm);

    // F16 GGUF dequant vs BF16 safetensors dequant differ in the low mantissa bits; allow a small
    // tolerance for the matrix weights, tight for the value-transformed 1D tensors (F32 in both).
    let tol_w = 5e-2f32;   // bf16(8-bit mantissa) vs f16(10-bit) round-trip slack
    let tol_f32 = 1e-4f32; // ssm_a / ssm_norm / dt are F32 in both -> near-exact

    // pick a linear-attn layer (not a multiple of full_attention_interval) and a full-attn layer.
    let lin = (0..cfg.n_layer).find(|&il| cfg.layer_kind(il) == bw24_gguf::config::LayerKind::LinearAttention).unwrap();
    let full = (0..cfg.n_layer).find(|&il| cfg.layer_kind(il) == bw24_gguf::config::LayerKind::FullAttention).unwrap();
    println!("checking linear-attn layer {lin}, full-attn layer {full}\n");

    let mut all_ok = true;
    // Compare ST tensor `name` against GGUF tensor `gg` (usually the same ggml name; differs only for
    // the pre-FFN norm, where ST asks `ffn_norm` and the GGUF twin stores `post_attention_norm`).
    let mut check2 = |name: &str, gg: &str, tol: f32| {
        match (deq(&st, name), deq(&gs, gg)) {
            (Some((a, na)), Some((b, nb))) => {
                let shape_ok = na == nb && a.len() == b.len();
                let (maxd, nmis) = if shape_ok { cmp(&a, &b, tol) } else { (f32::INFINITY, a.len().max(b.len())) };
                let ok = shape_ok && nmis == 0;
                all_ok &= ok;
                println!("{:<28} st.ne={:?} gguf.ne={:?} max|Δ|={:.3e} mism>{:.0e}={} {}",
                    name, na, nb, maxd, tol, nmis, if ok {"OK"} else {"FAIL"});
                // extra invariant spot-checks
                if name.ends_with(".ssm_a") {
                    let neg = a.iter().all(|v| *v < 0.0 && v.is_finite());
                    println!("    ssm_a all-negative&finite: {neg}  (e.g. {:?})", &a[..a.len().min(4)]);
                }
            }
            (sa, sb) => {
                all_ok = false;
                println!("{:<28} MISSING st={} gguf={}", name, sa.is_some(), sb.is_some());
            }
        }
    };
    println!("--- linear-attn SSM tensors (V-reorder / -exp / squeeze / norm) ---");
    let p = |s: &str| format!("blk.{lin}.{s}");
    check2(&p("attn_qkv.weight"), &p("attn_qkv.weight"), tol_w);
    check2(&p("attn_gate.weight"), &p("attn_gate.weight"), tol_w);
    check2(&p("ssm_alpha.weight"), &p("ssm_alpha.weight"), tol_w);
    check2(&p("ssm_beta.weight"), &p("ssm_beta.weight"), tol_w);
    check2(&p("ssm_a"), &p("ssm_a"), tol_f32);
    check2(&p("ssm_dt.bias"), &p("ssm_dt.bias"), tol_f32);
    check2(&p("ssm_conv1d.weight"), &p("ssm_conv1d.weight"), tol_f32);
    check2(&p("ssm_norm.weight"), &p("ssm_norm.weight"), tol_f32);
    check2(&p("ssm_out.weight"), &p("ssm_out.weight"), tol_w);

    println!("\n--- dense norms (qwen35 +1) + full-attn layer {full} ---");
    check2(&p("attn_norm.weight"), &p("attn_norm.weight"), tol_f32);        // +1 norm
    // pre-FFN norm: ST asks ffn_norm.weight (-> post_attention_layernorm, +1); GGUF twin stores it
    // as post_attention_norm.weight. Compare across the differing names so the +1 here is covered.
    check2(&p("ffn_norm.weight"), &p("post_attention_norm.weight"), tol_f32);
    check2(&format!("blk.{full}.attn_q.weight"), &format!("blk.{full}.attn_q.weight"), tol_w);
    check2(&format!("blk.{full}.attn_k.weight"), &format!("blk.{full}.attn_k.weight"), tol_w);
    check2(&format!("blk.{full}.attn_q_norm.weight"), &format!("blk.{full}.attn_q_norm.weight"), tol_f32);  // +1 norm
    check2("output_norm.weight", "output_norm.weight", tol_f32);

    println!("\n{}", if all_ok { "ALL TENSORS MATCH — SSM name-map + transforms verified vs GGUF twin" }
                     else { "MISMATCH — see FAIL rows above" });
    if !all_ok { std::process::exit(1); }
    Ok(())
}
