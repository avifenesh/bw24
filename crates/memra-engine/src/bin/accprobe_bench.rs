//! accprobe-bench — THE Q1 MEASUREMENT of the FP8-ST v3 gate (lane/fp8-v3-gate, 2026-08-05).
//!
//! PRICES ONE CLAIM. research/fp8st-20260804/mmq-v2/LANE-VERDICT.jsonl §3 ends its ceiling analysis
//! with: "What is left is the f32 accumulator itself against the floor's s32, and that is structural
//! to per-block FP8 as formulated." §6 then makes it a gate: a v3 (quantize the e4m3 mantissa into an
//! int8-compatible product per 128-block so the chain accumulates in s32) "should not start without a
//! receipted estimate that s32-vs-f32 accumulate is worth the >= 10pp it would have to buy."
//!
//! THE INSTRUMENT (cu/mmq_q8_0_f32acc.cu): the Q8_0 MMQ FLOOR kernel with the accumulator as its ONE
//! free variable. Arm S32 is the floor's `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`; arm F32
//! is the SAME m16n8k32 shape with the SAME A (4x b32) / B (2x b32) / D (4-reg) fragment ABI running
//! `mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32` — the exact op v2 accumulates
//! in. Same tiles, same smem, same loaders, same launch config, same fold count, same device
//! buffers. So `ratio = t_f32 / t_s32` is the accumulator's cost on this silicon, and
//! `delta_pp = 100 * (ratio - 1)` is an UPPER BOUND on what a v3's s32 conversion could recover
//! (a v3 would additionally pay per-128-block mantissa extraction).
//!
//! WHY MEASURE IT ON THE FLOOR AND NOT ON v2's KERNEL: v2 already chains four k32 MMAs into one
//! accumulator and folds ONCE per 128-k block, where the floor folds every k32 — so v2's epilogue
//! f32 work is already strictly cheaper than the floor's and cannot be the residual gap. Swapping
//! the accumulator inside the floor holds tiles, traffic, occupancy and fold count fixed and moves
//! only the named variable. Building an s32 v2 instead would change the arithmetic contract, need a
//! new host reference, and be the v3 this gate is deciding whether to fund.
//!
//! GEMM-ONLY AND QUANTIZER-FREE: both arms consume the SAME pre-quantized block_q8_1_mmq activation
//! buffer, synthesized here once per shape. Two reasons it is synthesized rather than produced by a
//! device quantizer: (1) the accumulator lives in the GEMM, so a quantizer inside the timed region
//! would dilute exactly the quantity being measured; (2) an int8 quantizer emits +-127 at every block
//! amax, and 0x7F is the e4m3 NaN code — the F32 arm would then run on NaNs and could take a
//! different path than the S32 arm, making it not the same experiment. The synthesized bytes are
//! therefore restricted to magnitudes <= 126, i.e. finite and non-NaN in BOTH readings, and both arms
//! read the identical bytes.
//!
//! NEITHER ARM'S OUTPUT IS A NUMERIC CLAIM. The two arms compute different arithmetic on the same
//! bytes by construction (int8 products vs e4m3 products); this bin measures TIME only. Exactness for
//! the real kernels is owned by fp8-mmq-check / kernel-check.
//!
//! PROTOCOL (evidence discipline): warm up both arms, then reps x (F32 timed, S32 timed) INTERLEAVED
//! inside the rep loop so both share one clock/thermal regime; median of reps; run under
//! flock /tmp/gpu5090.lock. Temps are recorded by the caller's telemetry, not here.
//!
//! usage: accprobe-bench [m] [reps] [27b|1p7b]        (default m=512, reps=9, 27b)

use memra_engine::Engine;

/// One block_q8_1_mmq record: 4x f32 scale (D4, no sum term) + 128 int8 quants = 144 B.
const BLOCK_BYTES: usize = 4 * 4 + 128;

/// Synthesize a block_q8_1_mmq activation stream directly, k-major then column, exactly as
/// cu/mmq_q8_0.cu's quantizer lays it out: block index `ib = (i0 / 128) * n_tokens + token`.
/// Quant magnitudes stay <= 126 so the bytes are finite and non-NaN read as int8 OR as e4m3.
fn synth_act_q8_1(in_f: usize, m: usize, total_bytes: usize, mid: bool) -> Vec<u8> {
    let mut buf = vec![0u8; total_bytes];
    // in_f padded to 512 (MATRIX_ROW_PADDING), 128 values per block.
    let ne10_padded = in_f.next_multiple_of(512);
    let kblocks = ne10_padded / 128;
    let mut s: u32 = 0x9E37_79B9;
    for kb in 0..kblocks {
        for t in 0..m {
            let ib = kb * m + t;
            let off = ib * BLOCK_BYTES;
            if off + BLOCK_BYTES > buf.len() {
                continue;
            }
            // Four f32 D4 scales — one per 32 values, as the real quantizer writes.
            for sl in 0..4 {
                let d: f32 = 0.0125 + (((kb + t + sl) % 5) as f32) * 0.0037;
                buf[off + sl * 4..off + sl * 4 + 4].copy_from_slice(&d.to_le_bytes());
            }
            // 128 int8 quants. Magnitude clamped to 126: 0x7F/0xFF are the e4m3 NaN codes, and the
            // F32 arm must not be handed a NaN the S32 arm does not see. `mid` narrows the bytes to
            // e4m3-normal mid exponents — see the ACCPROBE_DIST control on synth_w_q8_0.
            for q in 0..128 {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                buf[off + 16 + q] = if mid {
                    (0x30 + ((s >> 17) % 0x20) as u8) | ((((s >> 9) & 1) as u8) << 7)
                } else {
                    let v = ((s >> 17) % 253) as i32 - 126; // -126 ..= 126
                    (v as i8) as u8
                };
            }
        }
    }
    buf
}

/// Synthesize raw ggml block_q8_0 weight rows: in_f/32 blocks per row, 34 B each (fp16 scale + 32
/// int8). Same magnitude clamp and the same non-NaN reasoning as the activation.
///
/// DATA-DISTRIBUTION CONTROL (`ACCPROBE_DIST`): the F32 arm reads these bytes as e4m3, and e4m3
/// codes 0x01-0x07 are DENORMALS. If QMMA had any denormal or data-dependent slow path, the headline
/// ratio would be an artifact of the byte distribution rather than a property of the accumulator.
/// `wide` (default) spans the full magnitude range 1..=126, so ~5.5% of bytes are e4m3 denormals and
/// the exponent range is ~15 binades; `mid` restricts magnitudes to 0x30..=0x4F, which as e4m3 is a
/// narrow band of NORMAL values with no denormals and no near-max exponents. If the two agree, the
/// ratio is a property of the instruction, not the data — which is what the gate needs.
fn synth_w_q8_0(in_f: usize, out_f: usize, mid: bool) -> Vec<u8> {
    let nblk = in_f / 32;
    let mut buf = vec![0u8; out_f * nblk * 34];
    let mut s: u32 = 0x1234_5678;
    for r in 0..out_f {
        for b in 0..nblk {
            let off = (r * nblk + b) * 34;
            // fp16 block scale ~0.01-0.03, built from the f32 bit pattern (round-to-zero is fine —
            // this is timing data, and both arms read the same bytes either way).
            let d: f32 = 0.01 + (((r + b) % 7) as f32) * 0.003;
            let h = f32_to_f16_bits(d);
            buf[off..off + 2].copy_from_slice(&h.to_le_bytes());
            for q in 0..32 {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                buf[off + 2 + q] = if mid {
                    // magnitude 0x30..=0x4F (e4m3 normal, mid exponents), random sign
                    (0x30 + ((s >> 17) % 0x20) as u8) | ((((s >> 9) & 1) as u8) << 7)
                } else {
                    let v = ((s >> 17) % 253) as i32 - 126; // -126 ..= 126
                    (v as i8) as u8
                };
            }
        }
    }
    buf
}

/// Minimal f32 -> IEEE binary16 (round-to-nearest-even), normals only — the synthesized range
/// (0.01-0.03) is comfortably normal in fp16, so no subnormal/overflow handling is needed.
fn f32_to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xFF) as i32 - 127 + 15;
    assert!((1..=30).contains(&exp), "f32_to_f16_bits: {x} out of normal fp16 range");
    let mant = b & 0x007F_FFFF;
    let mut h = sign | ((exp as u16) << 10) | ((mant >> 13) as u16);
    // round-to-nearest-even on the dropped 13 bits
    let dropped = mant & 0x1FFF;
    if dropped > 0x1000 || (dropped == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    h
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let m: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(512);
    let reps: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(9);
    let set = std::env::args().nth(3).unwrap_or_else(|| "27b".to_string());

    // Data-distribution control: "wide" (default) spans e4m3's full range incl. ~5.5% denormals,
    // "mid" is e4m3-normal mid exponents only. Agreement proves the ratio is the instruction, not
    // the byte distribution.
    let dist = std::env::var("ACCPROBE_DIST").unwrap_or_else(|_| "wide".into());
    let mid = dist == "mid";

    let e = Engine::new(0)?;
    println!(
        "GPU: {}  m={m}  reps={reps}  shapes={set}  dist={dist}",
        e.ctx().name()?
    );
    println!(
        "ACCUMULATOR INSTRUMENT: one kernel (cu/mmq_q8_0_f32acc.cu), one variable (s32 vs f32 MMA)."
    );
    println!("interleaved f32,s32 per rep; median of reps; ratio = t_f32/t_s32; delta_pp = 100*(ratio-1)");
    println!(
        "{:<28} {:>11} {:>11} {:>9} {:>10} {:>11} {:>11}",
        "shape in->out", "f32acc_ms", "s32acc_ms", "ratio", "delta_pp", "f32_TFLOP", "s32_TFLOP"
    );

    // The v2 shape sheet, verbatim (research/fp8st-20260804/mmq-v2 §3 final_sheet).
    let shapes_27b: [(usize, usize, &str); 6] = [
        (5120, 12288, "q_proj"),
        (5120, 1024, "k/v_proj"),
        (6144, 5120, "o_proj"),
        (5120, 17408, "gate/up_proj"),
        (17408, 5120, "down_proj"),
        (5120, 5120, "square-ref"),
    ];
    let shapes_1p7b: [(usize, usize, &str); 5] = [
        (2048, 2048, "q_proj"),
        (2048, 1024, "k/v_proj"),
        (2048, 2048, "o_proj"),
        (2048, 6144, "gate/up_proj"),
        (6144, 2048, "down_proj"),
    ];
    let shapes: Vec<(usize, usize, &str)> = if set == "1p7b" {
        shapes_1p7b.to_vec()
    } else {
        shapes_27b.to_vec()
    };

    let mut sum_ln_ratio = 0.0f64;
    let mut n_cells = 0usize;

    for (in_f, out_f, label) in shapes {
        let w = synth_w_q8_0(in_f, out_f, mid);
        let w_d = e.htod_bytes(&w)?;

        let act_bytes = e.accprobe_act_bytes(in_f, m);
        let act = synth_act_q8_1(in_f, m, act_bytes, mid);
        let act_d = e.htod_bytes(&act)?;

        // warmup both arms (allocation, code load)
        let _ = e.accprobe_gemm(&w_d, &act_d, m, in_f, out_f, true)?;
        let _ = e.accprobe_gemm(&w_d, &act_d, m, in_f, out_f, false)?;
        e.stream().synchronize()?;

        let mut t_f32: Vec<f64> = Vec::with_capacity(reps);
        let mut t_s32: Vec<f64> = Vec::with_capacity(reps);
        for _ in 0..reps {
            // INTERLEAVED inside the rep loop: both arms then share one clock/thermal regime.
            let t0 = std::time::Instant::now();
            let _ = e.accprobe_gemm(&w_d, &act_d, m, in_f, out_f, true)?;
            e.stream().synchronize()?;
            t_f32.push(t0.elapsed().as_secs_f64());

            let t1 = std::time::Instant::now();
            let _ = e.accprobe_gemm(&w_d, &act_d, m, in_f, out_f, false)?;
            e.stream().synchronize()?;
            t_s32.push(t1.elapsed().as_secs_f64());
        }
        t_f32.sort_by(f64::total_cmp);
        t_s32.sort_by(f64::total_cmp);
        let (a, b) = (t_f32[reps / 2], t_s32[reps / 2]);
        let ratio = a / b;
        let flop = 2.0 * m as f64 * in_f as f64 * out_f as f64;
        println!(
            "{:<28} {:>11.4} {:>11.4} {:>8.3}x {:>+10.1} {:>11.1} {:>11.1}",
            format!("{label} {in_f}->{out_f}"),
            a * 1e3,
            b * 1e3,
            ratio,
            100.0 * (ratio - 1.0),
            flop / a / 1e12,
            flop / b / 1e12
        );
        sum_ln_ratio += ratio.ln();
        n_cells += 1;
    }

    let geo = (sum_ln_ratio / n_cells as f64).exp();
    println!(
        "GEOMEAN ratio (f32/s32) over {n_cells} shapes: {geo:.4}x  =>  delta_pp {:+.1}",
        100.0 * (geo - 1.0)
    );
    println!(
        "READ: delta_pp is what s32 accumulation is worth at fixed geometry — an UPPER BOUND on a v3."
    );
    Ok(())
}
