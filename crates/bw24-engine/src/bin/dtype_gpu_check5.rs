//! GPU dequant/qmatvec validation for ALL 5 new dtypes on REAL daily-GGUF tensors.
//!
//! Oracle = cpu_linear(bw24_dequant(W), x). bw24's CPU dequant is proven byte-for-byte
//! identical to ggml dequantize_row_<type> (see bw24-gguf example dequant_oracle_diff,
//! which diffs the SAME quant bytes against ggml's to_float and reports max_abs=0.0).
//! So this validates the GPU paths against the ggml ground truth transitively.
//!
//! For each dtype we check:
//!   Stage-A: qmatvec_f32 (dequant-in-kernel)             -> expect rel < 1e-4
//!   Stage-B: int8 dp4a fast path (where one exists)      -> expect rel < 3e-2
//! IQ3_S has NO dp4a fast path (intentional, see lib.rs) so only Stage-A is checked.
//!
//! Tensors (all confirmed present via tools/ggml_list_tensors):
//!   NVFP4  <- 9B-NVFP4   blk.0.ffn_gate.weight        (2D)
//!   Q5_K   <- 9B-NVFP4   blk.0.attn_gate.weight       (2D)
//!   IQ3_S  <- 35B-IQ4_XS blk.0.ffn_gate_exps.weight   (3D MoE, slice expert 0)
//!   IQ4_XS <- 35B-IQ4_XS blk.0.ffn_down_exps.weight   (3D MoE, slice expert 0)
//!   Q3_K   <- 35B-IQ4_XS blk.40.ffn_gate_exps.weight  (3D MoE, slice expert 0)

use bw24_engine::Engine;
use bw24_gguf::{dequant, GgmlType, GgufFile};
use bw24_runtime::cpu_linear;

fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}
fn pr(i: usize) -> f32 {
    let x = (i.wrapping_mul(2654435761) ^ 0x9E3779B9) as u32;
    ((x >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
}

const GGUF_9B: &str =
    "/home/avifenesh/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf";
const GGUF_35B: &str =
    "/home/avifenesh/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let e = Engine::new(0)?;
    println!("GPU: {}\n", e.ctx().name()?);
    let mut fails = 0;

    // (gguf, tensor, dtype, QT code, fast-path selector or "" for none)
    let cases: [(&str, &str, GgmlType, i32, &str); 5] = [
        (GGUF_9B, "blk.0.ffn_gate.weight", GgmlType::NVFP4, bw24_engine::QT_NVFP4, "nvfp4"),
        (GGUF_9B, "blk.0.attn_gate.weight", GgmlType::Q5_K, bw24_engine::QT_Q5_K, "q5k"),
        (GGUF_35B, "blk.0.ffn_gate_exps.weight", GgmlType::IQ3_S, bw24_engine::QT_IQ3_S, ""),
        (GGUF_35B, "blk.0.ffn_down_exps.weight", GgmlType::IQ4_XS, bw24_engine::QT_IQ4_XS, "iq4xs"),
        (GGUF_35B, "blk.40.ffn_gate_exps.weight", GgmlType::Q3_K, bw24_engine::QT_Q3_K, "q3k"),
    ];

    for (path, tname, gty, qt, sel) in cases {
        let g = GgufFile::open(path)?;
        let t = match g.find(tname) {
            Some(t) => t,
            None => { println!("{tname}: NOT FOUND in {path}"); fails += 1; continue; }
        };
        if t.ggml_type != gty {
            println!("{tname}: type {:?} != expected {gty:?}", t.ggml_type);
            fails += 1;
            continue;
        }
        // in_f = ne[0] (contracted/K dim); out_f = ne[1] (rows). For 3D MoE tensors validate expert 0.
        let in_f = t.ne[0] as usize;
        let out_f = t.ne[1] as usize;
        let raw_all = g.tensor_data(t);
        let one_mat_elems = in_f * out_f;
        let n_experts = if t.ne.len() >= 3 { t.ne[2] as usize } else { 1 };
        let total_rows = out_f * n_experts;
        let row_bytes = raw_all.len() / total_rows;
        let mat_bytes = out_f * row_bytes; // expert 0 slice
        let raw = &raw_all[..mat_bytes];

        let w_f32 = dequant::dequantize(gty, raw, one_mat_elems);
        let m = 2usize;
        let x: Vec<f32> = (0..m * in_f).map(|i| pr(i + 31) * 0.1).collect();
        let cpu = cpu_linear(&x, &w_f32, m, in_f, out_f);
        let scale = cpu.iter().map(|v| v.abs()).fold(0.0, f32::max).max(1.0);

        let wd = e.htod_bytes(raw)?;
        let xd = e.htod(&x)?;

        // Stage-A: dequant-in-kernel qmatvec.
        let ya = e.dtoh(&e.qmatvec(&wd, &xd, m, in_f, out_f, qt, row_bytes)?)?;
        let rela = maxdiff(&cpu, &ya) / scale;
        let okA = rela < 1e-4;
        println!("[{gty:?}] {tname} (in={in_f} out={out_f}) Stage-A qmatvec: rel={rela:.3e} {}",
                 if okA { "OK" } else { fails += 1; "FAIL" });
        if !okA { for i in 0..3 { println!("    A[{i}] cpu={} gpu={}", cpu[i], ya[i]); } }

        // Stage-B: int8 dp4a fast path (only where one exists).
        if sel.is_empty() {
            println!("[{gty:?}] {tname} Stage-B dp4a   : (no fast path — Stage-A only) SKIP");
        } else {
            let yb = match sel {
                "nvfp4" => e.dtoh(&e.qmatvec_nvfp4_fast(&wd, &xd, m, in_f, out_f, row_bytes)?)?,
                "q5k" => e.dtoh(&e.qmatvec_q5_K_fast(&wd, &xd, m, in_f, out_f, row_bytes)?)?,
                "iq4xs" => e.dtoh(&e.qmatvec_iq4_XS_fast(&wd, &xd, m, in_f, out_f, row_bytes)?)?,
                "q3k" => e.dtoh(&e.qmatvec_q3_K_fast(&wd, &xd, m, in_f, out_f, row_bytes)?)?,
                _ => unreachable!(),
            };
            let relb = maxdiff(&cpu, &yb) / scale;
            let okB = relb < 3e-2;
            println!("[{gty:?}] {tname} Stage-B dp4a   : rel={relb:.3e} {}",
                     if okB { "OK" } else { fails += 1; "FAIL" });
            if !okB { for i in 0..3 { println!("    B[{i}] cpu={} gpu={}", cpu[i], yb[i]); } }
        }
        println!();
    }

    if fails == 0 {
        println!("ALL GREEN: NVFP4/Q5_K/IQ3_S/IQ4_XS/Q3_K GPU qmatvec match the ggml-equivalent CPU oracle.");
        Ok(())
    } else {
        Err(format!("{fails} GPU dtype check(s) FAILED").into())
    }
}
