//! WALL-GAP ARC bench (2026-07-10): kernel-only timing of the gate_up dev_q8 GU variants on a
//! REALISTIC synthetic 35B expert slab (IQ3_S, in_f=2048, n_ff=512, 8-of-16 experts) — one
//! process, zero model loads (the CPU-load discipline: e2e A/B cost 4 x 35B loads per pair).
//! Prints us/launch + achieved GB/s per variant, plus a byte-compare against the `v` baseline.
//! Variant comes from BW24_MOE_DEVQ8_GU exactly like production dispatch; run per variant:
//!   for v in "" vsm vsm2; do BW24_MOE_DEVQ8_GU=$v ./gu-wall-bench; done
use bw24_engine::Engine;

fn hb(i: usize) -> u8 { ((i.wrapping_mul(2654435761)) >> 13) as u8 }
fn pr(i: usize) -> f32 { ((i.wrapping_mul(40503) % 1000) as f32) / 1000.0 - 0.5 }

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cudarc::driver::DevicePtr;
    let e = Engine::new(0)?;
    let n_expert = 16usize;
    let n_used = 8usize;
    let (in_f, n_ff) = (2048usize, 512usize);
    let rb = in_f / 256 * 110;                 // IQ3_S row bytes = 880
    let qt = bw24_engine::QT_IQ3_S;
    let stride = rb * n_ff;
    let slab: Vec<u8> = (0..n_expert * stride).map(|i| hb(i + 7)).collect();
    let slab_d = e.htod_bytes(&slab)?;
    let (p0, _g) = slab_d.device_ptr(e.stream());
    let mut table_h = vec![0u64; 3 * n_expert];
    for ex in 0..n_expert {
        table_h[ex] = p0 + (ex * stride) as u64;
        table_h[n_expert + ex] = p0 + (ex * stride) as u64;   // up -> same slab (timing-equal)
        table_h[2 * n_expert + ex] = p0 + (ex * stride) as u64;
    }
    let table_d = e.htod_u64(&table_h)?;
    let sel_d = e.htod_i32(&[3, 7, 0, 12, 5, 9, 14, 1])?;
    let aq: Vec<i8> = (0..in_f).map(|i| hb(i + 11) as i8).collect();
    let ad: Vec<f32> = (0..in_f / 32).map(|i| (pr(i) + 1.5) * 0.01).collect();
    let aq_d = e.htod_i8(&aq)?;
    let ad_d = e.htod(&ad)?;

    let variant = std::env::var("BW24_MOE_DEVQ8_GU").unwrap_or_default();
    let bytes = (n_used * 2 * n_ff * rb / (in_f / 32) * (in_f / 32)) as f64; // 8ex*2rows*512*880
    let bytes = (n_used * 2 * n_ff) as f64 * rb as f64;
    // warmup + reps
    let reps = 300usize;
    for _ in 0..20 {
        let _ = e.moe_gate_up_silu8_dev_q8(&table_d, &sel_d.slice(0..n_used), &aq_d, &ad_d,
                                           in_f, n_ff, n_used, n_expert, qt, qt, rb, rb)?;
    }
    e.stream().synchronize()?;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        let _ = e.moe_gate_up_silu8_dev_q8(&table_d, &sel_d.slice(0..n_used), &aq_d, &ad_d,
                                           in_f, n_ff, n_used, n_expert, qt, qt, rb, rb)?;
    }
    e.stream().synchronize()?;
    let us = t0.elapsed().as_secs_f64() * 1e6 / reps as f64;
    println!("variant[{}]: {us:.2} us/launch  {:.0} GB/s ({:.0}% of 858)",
             if variant.is_empty() { "v(auto)" } else { &variant },
             bytes / us / 1e3, bytes / us / 1e3 / 858.0 * 100.0);
    // ---- down8 floor (w8h2v, the 35B shape: IQ4_XS in_f=512 out_f=2048 rb=272) ----
    let (din, dout, drb) = (512usize, 2048usize, 272usize);
    let dstride = drb * dout;
    let dslab: Vec<u8> = (0..n_expert * dstride).map(|i| hb(i + 31)).collect();
    let dslab_d = e.htod_bytes(&dslab)?;
    let (pdn, _g2) = dslab_d.device_ptr(e.stream());
    let mut dtab = vec![0u64; 3 * n_expert];
    for ex in 0..n_expert {
        dtab[ex] = pdn + (ex * dstride) as u64;
        dtab[n_expert + ex] = pdn + (ex * dstride) as u64;
        dtab[2 * n_expert + ex] = pdn + (ex * dstride) as u64;
    }
    let dtab_d = e.htod_u64(&dtab)?;
    let w_d = e.htod(&(0..n_used).map(|i| pr(i) * 0.4).collect::<Vec<f32>>())?;
    let aq2: Vec<i8> = (0..n_used * din).map(|i| hb(i + 77) as i8).collect();
    let ad2: Vec<f32> = (0..n_used * (din / 32)).map(|i| (pr(i) + 1.5) * 0.01).collect();
    let aq2_d = e.htod_i8(&aq2)?;
    let ad2_d = e.htod(&ad2)?;
    let mut ddst = e.zeros(dout)?;
    let run = |ddst: &mut cudarc::driver::CudaSlice<f32>| -> Result<(), Box<dyn std::error::Error>> {
        let mut dv = ddst.slice_mut(0..dout);
        e.moe_down8_fma_dev_q8_variant("w8h2v", &dtab_d, &sel_d.slice(0..n_used),
                                       &w_d.slice(0..n_used), &aq2_d, &ad2_d, &mut dv,
                                       din, dout, n_used, n_expert, bw24_engine::QT_IQ4_XS, drb)?;
        Ok(())
    };
    for _ in 0..20 { run(&mut ddst)?; }
    e.stream().synchronize()?;
    let t1 = std::time::Instant::now();
    for _ in 0..reps { run(&mut ddst)?; }
    e.stream().synchronize()?;
    let dus = t1.elapsed().as_secs_f64() * 1e6 / reps as f64;
    let dbytes = (n_used * dout) as f64 * drb as f64;
    println!("down8[w8h2v]: {dus:.2} us/launch  {:.0} GB/s ({:.0}% of 858)",
             dbytes / dus / 1e3, dbytes / dus / 1e3 / 858.0 * 100.0);
    Ok(())
}
