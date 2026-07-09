// synthetic known-answer test through the actual launcher (both arms).
use bw24_engine::Engine;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let e = Engine::new(0)?;
    let (in_f, out_f, m) = (64usize, 128usize, 128usize);
    // one NVFP4 block per row: d = 4x ue4m3(1.0)=0x38? ue4m3: e4m3 unsigned — 1.0 = 0x38.
    // qs: 32 bytes of nibble-code pairs. code 2 = 1.0 -> byte 0x22.
    let mut raw = Vec::with_capacity(out_f * 36);
    for _ in 0..out_f {
        raw.extend_from_slice(&[0x38u8; 4]);
        raw.extend_from_slice(&[0x22u8; 32]);
    }
    let x: Vec<f32> = vec![1.0f32; m * in_f];
    let wd = e.htod_bytes(&raw)?;
    let xd = e.htod(&x)?;
    let y = e.dtoh(&e.qmatvec_mmq_nvfp4_w4a8_raw(&wd, &xd, m, in_f, out_f)?)?;
    let arm = std::env::var("BW24_MMQ_F8F4").unwrap_or_default();
    println!("arm={} y[0]={} y[1]={} y[last]={} (expect 64)", arm, y[0], y[1], y[m*out_f-1]);
    // k-resolution probes: indicator activations per 16-value group of the 64-k row.
    for k in 0..8usize {
        let xg: Vec<f32> = (0..m*in_f).map(|i| if i % in_f == k { 1.0 } else { 0.0 }).collect();
        let xdg = e.htod(&xg)?;
        let yg = e.dtoh(&e.qmatvec_mmq_nvfp4_w4a8_raw(&wd, &xdg, m, in_f, out_f)?)?;
        println!("k={k}: y[0]={} (expect 1)", yg[0]);
    }
    Ok(())
}
