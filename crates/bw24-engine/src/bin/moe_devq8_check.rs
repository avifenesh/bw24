//! down8-lane gate: BYTE-compare the wide-load (_v) MoE dev_q8 twins against their scalar
//! references on random weights/activations. The _v kernels change ONLY the IQ4_XS group-dot
//! load path (value-identical wide loads, see expert_dot_iq4xs_g_v in qmatvec.cu); every f32
//! output bit must match. Also exercises the non-8-aligned slab fallback guard (slab+4 table)
//! and the non-IQ4_XS pass-through (IQ3_S gate_up). Prints ALL GREEN on success.
use bw24_engine::{Engine, QT_IQ3_S, QT_IQ4_XS};
use cudarc::driver::DevicePtr;

// deterministic byte/float noise (kernel_check's pr() recipe)
fn hb(i: usize) -> u8 {
    ((i.wrapping_mul(2654435761) ^ 0x9E3779B9) >> 13) as u8
}
fn pr(i: usize) -> f32 {
    let x = (i.wrapping_mul(2654435761) ^ 0x9E3779B9) as u32;
    ((x >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
}

/// Random quant slab: `n` bytes of noise with each block's leading f16 `d` patched to a small
/// finite half (exp 8..15 -> |d| in [2^-7, 2)) so no inf/nan superblock scales.
fn rand_slab(n: usize, block_bytes: usize, seed: usize) -> Vec<u8> {
    let mut v: Vec<u8> = (0..n).map(|i| hb(i.wrapping_add(seed))).collect();
    let mut off = 0;
    while off + 2 <= n {
        let h = (hb(off ^ seed) as u16) | ((hb(off ^ seed ^ 77) as u16) << 8);
        let half = (h & 0x83FF) | ((8 + ((h >> 10) & 7)) << 10);
        v[off] = (half & 0xff) as u8;
        v[off + 1] = (half >> 8) as u8;
        off += block_bytes;
    }
    v
}

fn cmp_bits(a: &[f32], b: &[f32]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let e = Engine::new(0)?;
    println!("GPU: {}", e.ctx().name()?);
    let mut fails = 0;

    let n_expert = 16usize;
    let n_used = 8usize;
    let sel_h: Vec<i32> = vec![3, 7, 0, 12, 5, 9, 14, 1];
    let w_h: Vec<f32> = (0..n_used).map(|i| pr(i + 5) * 0.5).collect();
    let sel_d = e.htod_i32(&sel_h)?;
    let w_d = e.htod(&w_h)?;

    // ---- 35B down shape: IQ4_XS in_f=512 (nsb=16) out_f=2048, rb=272 ----
    let (in_f, out_f, rb) = (512usize, 2048usize, 272usize);
    let stride = rb * out_f;
    // gate/up strides unused by the down kernel but the table is [3, n_expert]; point all
    // three proj rows at real allocations so any addressing slip faults loudly.
    let down_slab = rand_slab(n_expert * stride, 136, 101);
    let down_d = e.htod_bytes(&down_slab)?;
    let (pd, _ev) = down_d.device_ptr(e.stream());
    let mut table_h = vec![0u64; 3 * n_expert];
    for ex in 0..n_expert {
        table_h[ex] = pd + (ex * stride) as u64;                // proj rows 0/1 also
        table_h[n_expert + ex] = pd + (ex * stride) as u64;     // -> down slab (harmless)
        table_h[2 * n_expert + ex] = pd + (ex * stride) as u64;
    }
    let table_d = e.htod_u64(&table_h)?;
    let aq2_h: Vec<i8> = (0..n_used * in_f).map(|i| hb(i + 900) as i8).collect();
    let ad2_h: Vec<f32> = (0..n_used * (in_f / 32)).map(|i| (pr(i + 33) + 1.5) * 0.01).collect();
    let aq2_d = e.htod_i8(&aq2_h)?;
    let ad2_d = e.htod(&ad2_h)?;

    let run_down = |variant: &str, table: &cudarc::driver::CudaSlice<u64>|
                    -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let mut dst = e.zeros(out_f)?;
        {
            let mut dv = dst.slice_mut(0..out_f);
            e.moe_down8_fma_dev_q8_variant(variant, table, &sel_d.slice(0..n_used),
                                           &w_d.slice(0..n_used), &aq2_d, &ad2_d, &mut dv,
                                           in_f, out_f, n_used, n_expert, QT_IQ4_XS, rb)?;
        }
        e.dtoh(&dst)
    };

    let ref_w8h2 = run_down("w8h2", &table_d)?;
    for v in ["base", "w8h2v", "w8h2r2", "w8h2r2v"] {
        let got = run_down(v, &table_d)?;
        let bad = cmp_bits(&ref_w8h2, &got);
        println!("down {v:>8} vs w8h2 (IQ4_XS 512x2048 x8): bit_mismatch={bad} {}",
                 if bad == 0 { "OK" } else { fails += 1; "FAIL" });
    }

    // ---- non-8-aligned slab: same bytes at base+4; _v must take the scalar fallback ----
    let mut pad_slab = vec![hb(1); 4];
    pad_slab.extend_from_slice(&down_slab);
    let pad_d = e.htod_bytes(&pad_slab)?;
    let (pp, _ev2) = pad_d.device_ptr(e.stream());
    let mut table_pad_h = vec![0u64; 3 * n_expert];
    for ex in 0..n_expert {
        let p = pp + 4 + (ex * stride) as u64;
        table_pad_h[ex] = p; table_pad_h[n_expert + ex] = p; table_pad_h[2 * n_expert + ex] = p;
    }
    let table_pad_d = e.htod_u64(&table_pad_h)?;
    let ref_pad = run_down("w8h2", &table_pad_d)?;
    let got_pad = run_down("w8h2v", &table_pad_d)?;
    let bad = cmp_bits(&ref_pad, &got_pad);
    println!("down    w8h2v vs w8h2 (slab+4 misaligned -> guard fallback): bit_mismatch={bad} {}",
             if bad == 0 { "OK" } else { fails += 1; "FAIL" });
    // and the misaligned run must still equal the aligned run's values (same weight bytes)
    let bad = cmp_bits(&ref_w8h2, &ref_pad);
    println!("down     w8h2 aligned vs misaligned same-bytes: bit_mismatch={bad} {}",
             if bad == 0 { "OK" } else { fails += 1; "FAIL" });

    // ---- 35B gate_up shape: IQ4_XS in_f=2048 (nsb=64) n_ff=512, rb=1088 ----
    let (gin_f, n_ff, grb) = (2048usize, 512usize, 1088usize);
    let gstride = grb * n_ff;
    let gate_slab = rand_slab(n_expert * gstride, 136, 202);
    let up_slab = rand_slab(n_expert * gstride, 136, 303);
    let gate_d = e.htod_bytes(&gate_slab)?;
    let up_d = e.htod_bytes(&up_slab)?;
    let (pg, _e0) = gate_d.device_ptr(e.stream());
    let (pu, _e1) = up_d.device_ptr(e.stream());
    let mut gtable_h = vec![0u64; 3 * n_expert];
    for ex in 0..n_expert {
        gtable_h[ex] = pg + (ex * gstride) as u64;
        gtable_h[n_expert + ex] = pu + (ex * gstride) as u64;
        gtable_h[2 * n_expert + ex] = pd + (ex * stride) as u64;
    }
    let gtable_d = e.htod_u64(&gtable_h)?;
    let aq_h: Vec<i8> = (0..gin_f).map(|i| hb(i + 400) as i8).collect();
    let ad_h: Vec<f32> = (0..gin_f / 32).map(|i| (pr(i + 55) + 1.5) * 0.01).collect();
    let aq_d = e.htod_i8(&aq_h)?;
    let ad_d = e.htod(&ad_h)?;

    let gu_ref = e.moe_gate_up_silu8_dev_q8_variant("base", &gtable_d, &sel_d.slice(0..n_used),
                    &aq_d, &ad_d, gin_f, n_ff, n_used, n_expert, QT_IQ4_XS, QT_IQ4_XS, grb, grb)?;
    let gu_v = e.moe_gate_up_silu8_dev_q8_variant("v", &gtable_d, &sel_d.slice(0..n_used),
                    &aq_d, &ad_d, gin_f, n_ff, n_used, n_expert, QT_IQ4_XS, QT_IQ4_XS, grb, grb)?;
    let bad = cmp_bits(&e.dtoh(&gu_ref)?, &e.dtoh(&gu_v)?);
    println!("gate_up v vs base (IQ4_XS 2048x512 x8): bit_mismatch={bad} {}",
             if bad == 0 { "OK" } else { fails += 1; "FAIL" });

    // ---- non-IQ4_XS pass-through: IQ3_S gate_up (expert_dot_g_v must fall to expert_dot_g) ----
    let irb = (gin_f / 256) * 110;   // IQ3_S block = 110B / 256 elems
    let istride = irb * n_ff;
    let i_gate = rand_slab(n_expert * istride, 110, 404);
    let i_up = rand_slab(n_expert * istride, 110, 505);
    let ig_d = e.htod_bytes(&i_gate)?;
    let iu_d = e.htod_bytes(&i_up)?;
    let (pig, _e2) = ig_d.device_ptr(e.stream());
    let (piu, _e3) = iu_d.device_ptr(e.stream());
    let mut itable_h = vec![0u64; 3 * n_expert];
    for ex in 0..n_expert {
        itable_h[ex] = pig + (ex * istride) as u64;
        itable_h[n_expert + ex] = piu + (ex * istride) as u64;
        itable_h[2 * n_expert + ex] = pig + (ex * istride) as u64;
    }
    let itable_d = e.htod_u64(&itable_h)?;
    let i_ref = e.moe_gate_up_silu8_dev_q8_variant("base", &itable_d, &sel_d.slice(0..n_used),
                    &aq_d, &ad_d, gin_f, n_ff, n_used, n_expert, QT_IQ3_S, QT_IQ3_S, irb, irb)?;
    let i_v = e.moe_gate_up_silu8_dev_q8_variant("v", &itable_d, &sel_d.slice(0..n_used),
                    &aq_d, &ad_d, gin_f, n_ff, n_used, n_expert, QT_IQ3_S, QT_IQ3_S, irb, irb)?;
    let bad = cmp_bits(&e.dtoh(&i_ref)?, &e.dtoh(&i_v)?);
    println!("gate_up v vs base (IQ3_S pass-through): bit_mismatch={bad} {}",
             if bad == 0 { "OK" } else { fails += 1; "FAIL" });

    if fails == 0 { println!("ALL GREEN"); } else { println!("{fails} FAILURES"); std::process::exit(1); }
    Ok(())
}
