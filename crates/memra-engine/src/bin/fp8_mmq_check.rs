//! fp8-mmq-check — the REAL exactness gate for the per-block FP8 MMQ prefill kernel
//! (cu/mmq_fp8_blk.cu, lane/fp8-mmq-v2).
//!
//! WHY A SYNTHETIC GATE IS THE GATE (not a model stream compare): per-block-FP8 arithmetic is a
//! DIFFERENT arithmetic path than the ARM B' Q8_0-requant floor, so their model logits may
//! legitimately differ in the last bits. The claim this kernel makes is "no precision is lost
//! relative to the checkpoint bytes", and the only way to prove that is against a host reference of
//! the SAME arithmetic. That is this binary.
//!
//! ARM 1 — EXACT (bit-identity, integer-valued):
//!   Weights and activations are drawn from e4m3 codes whose decoded values are small integers, and
//!   all block scales are powers of two. Every product and partial sum is then an exactly
//!   representable f32 integer (|partial| < 2^24 by construction), so f32 addition is exact and
//!   ORDER-INDEPENDENT. That removes the two things the kernel does not define (the 32-product
//!   reduction inside a single MMA, and v2's chaining of the four MMAs of a scale block into one
//!   accumulator — both hardware-internal) and turns the comparison into a true BIT-IDENTITY test:
//!   kernel bits must equal host bits, 0 ULP, no tolerance.
//!   The activation quantizer is exercised honestly: its per-128 amax/448 scale is a power of two by
//!   construction (amax is a power of two), so cvt.rn.satfinite.e4m3x2 re-codes x/d exactly.
//!
//! ARM 2 — RANDOM (bounded): real e4m3 codes and real f32 scales, so f32 add order does matter.
//!   Compared against the same host reference with a rel tolerance; this bounds residual rounding
//!   and, critically, catches any INDEXING error (wrong scale block, wrong k tail, wrong row) which
//!   would produce O(1) relative error, not 1e-7.
//!
//! RAGGED SHAPES are the point: n (out_f) not a multiple of 128 exercises need_check + the scale
//! grid's partial last row; in_f not a multiple of 128 exercises the k tail (zero-fill) and the
//! clamped scale column.
//!
//! CODE COVERAGE is gated too, not asserted: the run counts the distinct e4m3 byte values actually
//! present in the weight and quantized-activation operands and FAILS below 254 — all 256 codes
//! minus the two NaN magnitudes (0x7F/0xFF) the dispatch precondition refuses.
//!
//! V2 NOTE: the host reference below implements v2's grouping (scale block as the outer k loop, one
//! chained accumulator per block, one fold, per-128 activation scale). It is deliberately NOT v1's
//! arithmetic — the reference DEFINES the arithmetic and this gate proves the kernel implements it.
//!
//! usage: fp8-mmq-check            (default shape battery)
//!        fp8-mmq-check <in_f> <out_f> <m>   (single shape)

use memra_engine::Engine;

/// e4m3 code -> f32, EXACTLY the host convention (nvfp4_repack::fp8_e4m3_to_f32 /
/// memra_e4m3_to_f32 in cu/fp8_blk_dequant.cu): magnitude 0x7F decodes to 0.0 (modelopt).
fn e4m3_to_f32(x: u8) -> f32 {
    let mag = (x & 0x7F) as u32;
    if mag == 0x7F {
        return 0.0;
    }
    let exp = ((mag >> 3) & 0xF) as i32;
    let man = (mag & 0x7) as f32;
    let raw = if exp == 0 {
        (man * 0.125) * 0.015625 // (man/8) * 2^-6
    } else {
        (1.0 + man * 0.125) * (2f32).powi(exp - 7)
    };
    if x & 0x80 != 0 {
        -raw
    } else {
        raw
    }
}

/// f32 -> e4m3 code, cvt.rn.satfinite semantics (round-to-nearest-even, saturate to +-448).
/// Used only to model the ACTIVATION quantizer in the host reference.
fn f32_to_e4m3(v: f32) -> u8 {
    if v.is_nan() {
        return 0x7F;
    }
    let sign = if v < 0.0 || (v == 0.0 && v.is_sign_negative()) {
        0x80u8
    } else {
        0x00u8
    };
    let a = v.abs();
    if a >= 448.0 {
        return sign | 0x7E; // saturate to max finite (448)
    }
    // Exhaustive nearest-even search over the 127 finite magnitudes — small, obviously correct,
    // and this is a test binary (the kernel side uses the hardware cvt).
    let mut best = 0u8;
    let mut best_err = f32::INFINITY;
    for code in 0u8..=0x7E {
        let c = e4m3_to_f32(code);
        let err = (c - a).abs();
        if err < best_err {
            best_err = err;
            best = code;
        } else if err == best_err {
            // tie -> even mantissa
            if code & 1 == 0 {
                best = code;
            }
        }
    }
    sign | best
}

/// Host reference for the kernel's arithmetic, in the kernel's ORDER (see the ARITHMETIC CONTRACT
/// block in cu/mmq_fp8_blk.cu).
///
/// V2 GROUPING — this is the reference the v2 kernel must implement, and it is deliberately NOT
/// v1's: the 128-wide scale block is the outer k loop, the four 32-k MMAs of a block CHAIN into one
/// unscaled f32 accumulator, and (s_blk * dB) folds ONCE per block. dB is now the per-128
/// activation scale (v2's quantize_mmq_e4m3_d128_kernel), which is what makes it hoistable out of
/// the 128-k run at all. The gate's job is to prove the kernel implements THIS arithmetic exactly;
/// the reference defines it.
#[allow(clippy::too_many_arguments)]
fn host_ref(
    w: &[u8],           // [out_f x in_f] e4m3
    scales: &[f32],     // [srows x scols]
    act_q: &[u8],       // [m x in_f_pad] e4m3 quantized activation
    act_d: &[f32],      // [m x (in_f_pad/128)] per-128 activation scale
    in_f: usize,
    out_f: usize,
    m: usize,
    in_f_pad: usize,
) -> Vec<f32> {
    let scols = in_f.div_ceil(128);
    let mut y = vec![0f32; m * out_f];
    let k_iter_end = in_f.div_ceil(128) * 128;
    for i in 0..out_f {
        let srow = i / 128;
        for j in 0..m {
            let mut sum = 0f32;
            let mut kb = 0usize;
            while kb < k_iter_end {
                let s_blk = scales[srow * scols + (kb / 128).min(scols - 1)];
                let db = if kb < in_f_pad {
                    act_d[j * (in_f_pad / 128) + kb / 128]
                } else {
                    0.0
                };
                // ONE chained accumulator for the whole 128-k block, no intermediate scaling.
                let mut c = 0f32;
                for k01q in 0..4usize {
                    let g0 = kb + 32 * k01q;
                    for t in 0..32usize {
                        let g = g0 + t;
                        let wv = if g < in_f {
                            e4m3_to_f32(w[i * in_f + g])
                        } else {
                            0.0
                        };
                        let av = if g < in_f_pad {
                            e4m3_to_f32(act_q[j * in_f_pad + g])
                        } else {
                            0.0
                        };
                        c += wv * av;
                    }
                }
                sum += (s_blk * db) * c;
                kb += 128;
            }
            y[j * out_f + i] = sum;
        }
    }
    y
}

/// Model the v2 activation quantizer (cu/mmq_fp8_blk.cu quantize_mmq_e4m3_d128_kernel): per **128**
/// values, d = amax/448, q = cvt_e4m3(x * 448/amax). Zero-pads to MATRIX_ROW_PADDING (512), which
/// is a multiple of 128 so blocks never straddle the pad boundary.
fn quantize_act_ref(x: &[f32], in_f: usize, m: usize) -> (usize, Vec<u8>, Vec<f32>) {
    let in_f_pad = in_f.div_ceil(512) * 512;
    let mut q = vec![0u8; m * in_f_pad];
    let mut d = vec![0f32; m * (in_f_pad / 128)];
    for j in 0..m {
        for b in 0..(in_f_pad / 128) {
            let mut amax = 0f32;
            for t in 0..128 {
                let g = b * 128 + t;
                let v = if g < in_f { x[j * in_f + g] } else { 0.0 };
                amax = amax.max(v.abs());
            }
            let (dv, dinv) = if amax == 0.0 {
                (0.0, 0.0)
            } else {
                (amax / 448.0, 448.0 / amax)
            };
            d[j * (in_f_pad / 128) + b] = dv;
            for t in 0..128 {
                let g = b * 128 + t;
                let v = if g < in_f { x[j * in_f + g] } else { 0.0 };
                q[j * in_f_pad + g] = f32_to_e4m3(v * dinv);
            }
        }
    }
    (in_f_pad, q, d)
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn u8(&mut self) -> u8 {
        (self.next() >> 33) as u8
    }
    fn f01(&mut self) -> f32 {
        ((self.next() >> 40) as f32) / (1u32 << 24) as f32
    }
}

/// e4m3 codes whose decoded value is a small INTEGER in [-8, 8] (exact arm). Excludes 0x7F.
/// Integers only (no 0.5/1.5): with integer weights and integer activations every MMA partial sum
/// is an exact integer, so the hardware-internal 32-product reduction order cannot matter.
const INT_CODES: [u8; 13] = [
    0x00, // 0
    0x38, 0xB8, // +-1
    0x40, 0xC0, // +-2
    0x44, 0xC4, // +-3
    0x48, 0xC8, // +-4
    0x4C, 0xCC, // +-6
    0x50, 0xD0, // +-8
];

struct ArmResult {
    max_abs: f32,
    /// max|got-want| / rms(want) — the GEMM-standard relative measure. Per-ELEMENT relative error
    /// is meaningless here: with random e4m3 weights many outputs are near-total cancellations, so
    /// dividing by |want| manufactures a huge ratio out of a 1e-3 absolute difference while an
    /// actual indexing bug (wrong scale block / wrong k offset) shows up at O(1) on THIS measure.
    rms_rel: f32,
    ulp_mismatch: usize,
    n: usize,
}

fn compare(got: &[f32], want: &[f32]) -> ArmResult {
    let mut r = ArmResult { max_abs: 0.0, rms_rel: 0.0, ulp_mismatch: 0, n: got.len() };
    let mut sq = 0f64;
    for (g, w) in got.iter().zip(want.iter()) {
        if g.to_bits() != w.to_bits() {
            r.ulp_mismatch += 1;
        }
        let a = (g - w).abs();
        if a > r.max_abs {
            r.max_abs = a;
        }
        sq += (*w as f64) * (*w as f64);
    }
    let rms = (sq / got.len().max(1) as f64).sqrt() as f32;
    r.rms_rel = if rms > 0.0 { r.max_abs / rms } else { r.max_abs };
    r
}

#[allow(clippy::type_complexity)]
fn run_shape(
    e: &Engine,
    in_f: usize,
    out_f: usize,
    m: usize,
    exact_arm: bool,
    seed: u64,
) -> Result<(ArmResult, u32, usize), Box<dyn std::error::Error>> {
    let srows = (out_f + 127) / 128;
    let scols = (in_f + 127) / 128;
    let mut rng = Rng(seed);

    // ---- weights: raw e4m3 plane ----
    let mut w = vec![0u8; out_f * in_f];
    if exact_arm {
        for b in w.iter_mut() {
            *b = INT_CODES[(rng.u8() as usize) % INT_CODES.len()];
        }
    } else {
        for b in w.iter_mut() {
            let c = rng.u8();
            // avoid the NaN magnitude: the hardware MMA and the host convention disagree there,
            // which is exactly why the dispatch refuses tensors containing it.
            *b = if c & 0x7F == 0x7F { c & 0xBF } else { c };
        }
    }

    // ---- scale grid ----
    let mut scales = vec![0f32; srows * scols];
    if exact_arm {
        for s in scales.iter_mut() {
            // powers of two in [2^-4, 2^3] — exact multiplication
            *s = (2f32).powi(((rng.u8() % 8) as i32) - 4);
        }
    } else {
        for s in scales.iter_mut() {
            *s = 0.002 + 0.5 * rng.f01();
        }
    }

    // ---- activations ----
    let mut x = vec![0f32; m * in_f];
    if exact_arm {
        // EXACT-arm activation construction. The v2 quantizer computes d = amax/448 and
        // q = cvt_e4m3(x * 448/amax) per **128** values, so pinning amax to EXACTLY 448 in every
        // 128-block makes d == 1.0 and q == x bit-for-bit — the activation side is not
        // re-quantized at all. With x restricted to small integers, every MMA product and partial
        // sum is an exact f32 integer (|sum| well below 2^24, see the budget), so f32 addition is
        // exact AND associative. That is what licenses a 0-ULP bit-identity comparison against a
        // host reference whose in-MMA summation order (and now also whose MMA-chaining order)
        // differs from the hardware's.
        //   budget: per 128-k block |C| <= 8*448 + 127*8*4 = 7648; * s_blk (<= 8) = 61184;
        //           * (in_f/128 <= 40 blocks here) = 2.4e6 << 2^24 = 1.6e7.
        // A 128-block that is only PARTIALLY inside in_f is left all-zero on purpose: amax == 0
        // then gives d == 0 and the block contributes exact zeros on both sides, which keeps the
        // k-tail shapes inside the bit-identity arm instead of excusing them from it.
        for j in 0..m {
            let nb = in_f / 128;
            for b in 0..nb {
                for t in 0..128 {
                    let v = (rng.u8() % 5) as f32; // 0..4
                    let sgn = if rng.u8() & 1 == 0 { 1.0 } else { -1.0 };
                    x[j * in_f + b * 128 + t] = sgn * v;
                }
                // Pin amax = 448 (e4m3 max finite) at a rotating slot so the planted value does
                // not sit at the same k offset in every block.
                x[j * in_f + b * 128 + (b * 37 + j) % 128] = 448.0;
            }
        }
    } else {
        for v in x.iter_mut() {
            *v = 2.0 * rng.f01() - 1.0;
        }
    }

    // ---- host reference ----
    let (in_f_pad, aq, ad) = quantize_act_ref(&x, in_f, m);
    let want = host_ref(&w, &scales, &aq, &ad, in_f, out_f, m, in_f_pad);

    // ---- code coverage, MEASURED not assumed ----
    // The claim "every e4m3 code the kernel can legally see is exercised" is only evidence if it is
    // counted. 255 = all 256 minus the single NaN magnitude the dispatch refuses (0x7F/0xFF are
    // excluded by construction above, because hardware reads them as NaN and the host convention
    // reads 0.0 — that disagreement is the reason for the refusal, not something to test through).
    let mut seen = [false; 256];
    for b in w.iter().chain(aq.iter()) {
        seen[*b as usize] = true;
    }
    let codes = seen.iter().filter(|s| **s).count();

    // ---- kernel ----
    let wd = e.htod_bytes(&w)?;
    let sd = e.htod(&scales)?;
    let xd = e.htod(&x)?;
    let nan = e.fp8_blk_nan_count(&wd)?;
    let got = e.dtoh(&e.qmatvec_mmq_fp8_blk(&wd, &sd, &xd, m, in_f, out_f)?)?;

    Ok((compare(&got, &want), nan, codes))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let e = Engine::new(0)?;
    let args: Vec<String> = std::env::args().skip(1).collect();

    // (in_f, out_f, m) — ragged on purpose.
    let shapes: Vec<(usize, usize, usize)> = if args.len() == 3 {
        vec![(args[0].parse()?, args[1].parse()?, args[2].parse()?)]
    } else {
        vec![
            (128, 128, 128),   // one scale block, one tile — the smallest complete case
            (256, 256, 64),    // 2x2 grid, m < MMQ_X (y-tile overread path)
            (512, 384, 128),   // 3 scale rows exactly
            (5120, 1536, 512), // a real projection shape (27B q_proj class)
            (128, 200, 128),   // out_f % 128 != 0 -> need_check + partial last scale ROW
            (384, 320, 96),    // both axes ragged
            (144, 128, 128),   // in_f % 128 != 0 -> partial last scale COLUMN + k tail
            (272, 136, 40),    // in_f % 256 != 0 (k tail mid-iteration) + ragged out_f + tiny m
            (5120, 1536, 1),   // m = 1 (single-token prefill edge)
        ]
    };

    let mut fails = 0usize;
    let mut codes_max = 0usize;
    println!(
        "{:<22} {:>6} {:>12} {:>12} {:>15} {:>5} {:>6}",
        "shape(in,out,m)", "arm", "max_abs", "rms_rel", "bit_mismatch", "nan", "codes"
    );
    for (in_f, out_f, m) in shapes {
        // ARM 1: exact / bit-identity. NO tolerance — 0 differing bits or FAIL.
        let (r, nan, c1) = run_shape(&e, in_f, out_f, m, true, 0xC0FFEE_u64 ^ (in_f * 7919) as u64)?;
        let ok1 = r.ulp_mismatch == 0 && nan == 0;
        println!(
            "{:<22} {:>6} {:>12.3e} {:>12.3e} {:>9}/{:<5} {:>5} {:>6}  {}",
            format!("{in_f},{out_f},{m}"),
            "EXACT",
            r.max_abs,
            r.rms_rel,
            r.ulp_mismatch,
            r.n,
            nan,
            c1,
            if ok1 { "PASS" } else { "FAIL" }
        );
        if !ok1 {
            fails += 1;
        }

        // ARM 2: random e4m3 + real f32 scales. f32 add order differs from the host reference, so
        // bit-identity is not expected; 1e-5 of the output RMS is ~2 orders above the observed
        // reorder noise and ~5 below any indexing bug (which lands at O(1) on this measure).
        let (r2, nan2, c2) =
            run_shape(&e, in_f, out_f, m, false, 0xBADC0DE_u64 ^ (out_f * 104729) as u64)?;
        let ok2 = r2.rms_rel < 1e-5 && nan2 == 0;
        println!(
            "{:<22} {:>6} {:>12.3e} {:>12.3e} {:>9}/{:<5} {:>5} {:>6}  {}",
            format!("{in_f},{out_f},{m}"),
            "RAND",
            r2.max_abs,
            r2.rms_rel,
            r2.ulp_mismatch,
            r2.n,
            nan2,
            c2,
            if ok2 { "PASS" } else { "FAIL" }
        );
        if !ok2 {
            fails += 1;
        }
        codes_max = codes_max.max(c1).max(c2);
    }

    println!();
    // Code coverage is a GATE, not a footnote: all 254 legal e4m3 codes must actually appear in an
    // operand, or the "all codes" claim is unearned. 254 == 256 minus BOTH NaN magnitudes (0x7F and
    // 0xFF) — the two the dispatch precondition refuses, because hardware reads them as NaN while
    // the host / ARM B' convention reads 0.0.
    let codes_ok = codes_max >= 254;
    println!("e4m3 code coverage: {codes_max}/254 legal codes exercised (both NaN magnitudes 0x7F/0xFF excluded by the dispatch precondition)  {}",
             if codes_ok { "PASS" } else { "FAIL" });
    if !codes_ok {
        fails += 1;
    }

    println!();
    if fails == 0 {
        println!("=== fp8-mmq-check ALL GREEN (EXACT arm bit-identical, RAND arm < 1e-5 of RMS, 254/254 codes) ===");
        Ok(())
    } else {
        println!("=== fp8-mmq-check {fails} FAILURES ===");
        std::process::exit(1);
    }
}
