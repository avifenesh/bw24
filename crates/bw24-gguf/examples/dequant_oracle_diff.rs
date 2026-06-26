//! Diff bw24's CPU `dequant::dequantize` against ggml's `dequantize_row_<type>`
//! ground truth, byte-for-byte, on REAL tensors from the daily GGUFs.
//!
//! Pairs with tools/ggml_dequant_ref.cpp, which writes /tmp/dq/<name>.{raw,ref}:
//!   <name>.raw = the exact quant bytes ggml dequantized
//!   <name>.ref = ggml's f32 output (little-endian)
//! Here we feed the SAME .raw bytes into bw24's dequantize and compare to .ref.
//!
//! Run the C++ oracle first (see tools/run_dequant_validation.sh), then:
//!   cargo run -p bw24-gguf --example dequant_oracle_diff

use bw24_gguf::dequant::dequantize;
use bw24_gguf::GgmlType;

fn read_bytes(p: &str) -> Vec<u8> {
    std::fs::read(p).unwrap_or_else(|e| panic!("read {p}: {e}"))
}
fn read_f32(p: &str) -> Vec<f32> {
    read_bytes(p)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

struct Report {
    name: &'static str,
    n: usize,
    max_abs: f32,
    max_rel: f32,
    n_bad: usize,
    pass: bool,
}

fn check(name: &'static str, ty: GgmlType) -> Report {
    let raw = read_bytes(&format!("/tmp/dq/{name}.raw"));
    let refv = read_f32(&format!("/tmp/dq/{name}.ref"));
    let n = refv.len();
    let got = dequantize(ty, &raw, n);
    assert_eq!(got.len(), n, "{name}: length mismatch");

    // scale for relative tolerance: max |ref| across the tensor (per-tensor, robust to
    // the many tiny/zero weights that would blow up a pointwise rel metric).
    let scale = refv.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-12);

    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut worst_i = 0usize;
    let mut n_bad = 0usize;
    let mut first_bad: Vec<usize> = Vec::new();
    for i in 0..n {
        let d = (got[i] - refv[i]).abs();
        if d > max_abs {
            max_abs = d;
            worst_i = i;
        }
        let rel = d / scale;
        if rel > max_rel {
            max_rel = rel;
        }
        if rel > 1e-3 {
            n_bad += 1;
            if first_bad.len() < 8 {
                first_bad.push(i);
            }
        }
    }
    let pass = n_bad == 0;
    println!(
        "[{name}] n={n} max_abs={max_abs:.3e} max_rel(tensor-scaled)={max_rel:.3e} \
         n_bad(rel>1e-3)={n_bad} -> {}",
        if pass { "MATCH" } else { "MISMATCH" }
    );
    println!(
        "    worst[{worst_i}] (blk {} elem {}) got={} ref={}",
        worst_i / dequantize_blk(ty),
        worst_i % dequantize_blk(ty),
        got[worst_i],
        refv[worst_i]
    );
    if !pass {
        for &i in &first_bad {
            println!(
                "    MISMATCH[{i}] (blk {} elem {}) got={} ref={} absdiff={}",
                i / dequantize_blk(ty),
                i % dequantize_blk(ty),
                got[i],
                refv[i],
                (got[i] - refv[i]).abs()
            );
        }
    }
    Report { name, n, max_abs, max_rel, n_bad, pass }
}

fn dequantize_blk(ty: GgmlType) -> usize {
    ty.block_and_type_size().0 as usize
}

fn main() {
    println!("== bw24 CPU dequant vs ggml dequantize_row_<type> (real daily-GGUF tensors) ==\n");
    let reports = [
        check("nvfp4", GgmlType::NVFP4),
        check("q5k", GgmlType::Q5_K),
        check("iq3s", GgmlType::IQ3_S),
        check("iq4xs", GgmlType::IQ4_XS),
        check("q3k", GgmlType::Q3_K),
    ];

    println!("\n== SUMMARY ==");
    let mut all_ok = true;
    for r in &reports {
        println!(
            "  {:<7} n={:<6} max_abs={:.3e} max_rel={:.3e} n_bad={:<3} {}",
            r.name,
            r.n,
            r.max_abs,
            r.max_rel,
            r.n_bad,
            if r.pass { "MATCH" } else { "MISMATCH" }
        );
        all_ok &= r.pass;
    }
    if all_ok {
        println!("\nALL 5 dtypes MATCH ggml byte-for-byte (rel < 1e-3).");
    } else {
        std::process::exit(1);
    }
}
