// Diff bw24 CPU dequant vs ggml dequantize_row_<type> ground truth.
// Reads /tmp/orc.<TYPE>.{raw,ref} produced by /tmp/oracle (ggml libggml link).
use bw24_gguf::dequant::dequantize;
use bw24_gguf::GgmlType;

fn read_bytes(p: &str) -> Vec<u8> {
    std::fs::read(p).unwrap_or_else(|e| panic!("read {p}: {e}"))
}
fn read_f32(p: &str) -> Vec<f32> {
    let b = read_bytes(p);
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn check(name: &str, ty: GgmlType) {
    let raw = read_bytes(&format!("/tmp/orc.{name}.raw"));
    let refv = read_f32(&format!("/tmp/orc.{name}.ref"));
    let n = refv.len();
    let got = dequantize(ty, &raw, n);
    assert_eq!(got.len(), n, "{name}: length mismatch");
    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut worst_i = 0usize;
    let mut n_bad = 0usize;
    for i in 0..n {
        let a = got[i];
        let b = refv[i];
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
            worst_i = i;
        }
        let denom = b.abs().max(1e-6);
        let rel = d / denom;
        if rel > max_rel {
            max_rel = rel;
        }
        // tolerance: both are f32 from identical quant bytes -> should be ~exact
        if d > 1e-3 && rel > 1e-3 {
            n_bad += 1;
        }
    }
    println!(
        "{name}: n={n} max_abs={max_abs:.3e} max_rel={max_rel:.3e} n_bad(>1e-3)={n_bad} worst[{worst_i}] got={} ref={}",
        got[worst_i], refv[worst_i]
    );
    if n_bad > 0 {
        // print a few mismatches
        let mut shown = 0;
        for i in 0..n {
            let d = (got[i] - refv[i]).abs();
            let rel = d / refv[i].abs().max(1e-6);
            if d > 1e-3 && rel > 1e-3 {
                println!("    MISMATCH[{i}] got={} ref={} (blk {} elem {})", got[i], refv[i], i / 256, i % 256);
                shown += 1;
                if shown >= 12 { break; }
            }
        }
        println!("    >>> {name} FAILED: {n_bad} elements differ beyond tolerance");
    } else {
        println!("    >>> {name} PASS (byte-for-byte within fp tolerance)");
    }
}

fn main() {
    check("Q5_K", GgmlType::Q5_K);
    check("Q3_K", GgmlType::Q3_K);
    check("IQ4_XS", GgmlType::IQ4_XS);
}
