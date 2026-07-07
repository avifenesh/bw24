//! rms_norm_q8_1 (fused) vs rms_norm_decode + quantize_q8_1 (unfused) width check — the
//! FP-order pair contract at arbitrary ncols (M3 = 6144, never gated before).
use bw24_engine::Engine;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let e = Engine::new(0)?;
    for ncols in [2048usize, 5120, 6144, 12288] {
        let x: Vec<f32> = (0..ncols).map(|i| ((i * 2654435761) % 2000) as f32 / 700.0 - 1.4).collect();
        let w: Vec<f32> = (0..ncols).map(|i| ((i * 40503) % 1000) as f32 / 900.0 + 0.5).collect();
        let xd = e.htod(&x)?;
        let wd = e.htod(&w)?;
        let (fq, fd) = e.rms_norm_q8_1(&xd, &wd, ncols, 1, 1e-6)?;
        let mut h = e.zeros(ncols)?;
        e.rms_norm_decode(&xd, &wd, &mut h, ncols, 1, 1e-6)?;
        let (uq, ud) = e.quantize_q8_1(&h, 1, ncols)?;
        let fqh: Vec<i8> = e.stream().clone_dtoh(&fq)?;
        let uqh: Vec<i8> = e.stream().clone_dtoh(&uq)?;
        e.stream().synchronize()?;
        let fdh = e.dtoh(&fd)?;
        let udh = e.dtoh(&ud)?;
        let qmm = fqh.iter().zip(&uqh).filter(|(a, b)| a != b).count();
        let dmd = fdh.iter().zip(&udh).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!("ncols={ncols}: int8 mismatches {qmm}/{ncols}, scale maxdiff {dmd:.3e}");
    }
    Ok(())
}
