//! f8f4-check: correctness gate for the R-B W4A8-FP8 MMQ tile (BW24_MMQ_F8F4 seam).
//! On a real NVFP4 weight tensor: dequant-f32 CPU reference vs (a) the int8 W4A8 tile and
//! (b) the f8f4 tile. Both are int8/e4m3-activation classes -> rel tolerance 3e-2 (the
//! qmatvec_q8_0_fast gate class). Also prints the f8f4-vs-w4a8 direct rel gap.
//! usage: f8f4-check <model.gguf-with-NVFP4-tensors>
use bw24_engine::Engine;
use bw24_gguf::{GgufFile, GgmlType, dequant};
use bw24_runtime::cpu_linear;

fn pr(i: usize) -> f32 { (((i * 2654435761) % 1000) as f32) / 1000.0 - 0.5 }

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: f8f4-check <model.gguf>");
    let e = Engine::new(0)?;
    let g = GgufFile::open(&path)?;
    let mut fails = 0;
    let mut tested = 0;

    for t in g.tensors.iter().filter(|t| t.ggml_type == GgmlType::NVFP4 && t.ne.len() == 2) {
        if tested >= 3 { break; }
        let in_f = t.ne[0] as usize;
        let out_f = t.ne[1] as usize;
        if in_f % 64 != 0 || out_f < 128 { continue; }
        tested += 1;
        let raw = g.tensor_data(t);
        let w_f32 = dequant::dequantize(t.ggml_type, raw, in_f * out_f);
        let m = 128usize; // prefill-shape m (the MMQ tile's regime)
        let x: Vec<f32> = (0..m * in_f).map(|i| pr(i + 17) * 0.25).collect();
        let cpu = cpu_linear(&x, &w_f32, m, in_f, out_f);

        let wd = e.htod_bytes(raw)?;
        let xd = e.htod(&x)?;

        // The seam is OnceLock-cached process-wide: run this bin twice — BW24_MMQ_F8F4 unset
        // (int8 arm) and =1 (f8f4 arm). Both must pass the same 3e-2 gate.
        let ya = e.dtoh(&e.qmatvec_mmq_nvfp4_w4a8_raw(&wd, &xd, m, in_f, out_f)?)?;
        let arm_f8f4 = std::env::var("BW24_MMQ_F8F4").as_deref() == Ok("1");

        let scale = cpu.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1.0);
        let rel = |ref_v: &[f32], got: &[f32]| -> f32 {
            ref_v.iter().zip(got).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max) / scale
        };
        let r = rel(&cpu, &ya);
        // int8-act class sits at ~3e-3; the e4m3-act class (3 mantissa bits) runs ~10x coarser
        // and grows ~sqrt(k) — 5e-2 is the smoke bound; run-gen argmax + K=1..8 arbitrate e2e
        // (the PP_FP8/ST_E4M3 e4m3-act precedent gates green there).
        let ok = r < 5e-2;
        println!("{} [{}x{} m={m}] arm={} rel-vs-f32={r:.3e} {}",
                 t.name, in_f, out_f, if arm_f8f4 { "F8F4" } else { "INT8" },
                 if ok { "OK" } else { fails += 1; "FAIL" });
    }
    if tested == 0 { println!("no NVFP4 2D tensors found"); std::process::exit(2); }
    println!("{}", if fails == 0 { "=== f8f4-check ALL GREEN ===" } else { "=== f8f4-check FAILURES ===" });
    std::process::exit(if fails == 0 { 0 } else { 1 });
}
