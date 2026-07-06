// cuBLASLt n-dependence probe: same weights+input col, m=1 vs m=2 -> col0 bitwise equal?
use bw24_engine::Engine;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let e = Engine::new(0)?;
    let (in_f, out_f) = (2048usize, 32usize);
    let w: Vec<f32> = (0..in_f*out_f).map(|i| ((i*2654435761usize)%1000) as f32/1000.0-0.5).collect();
    let x1: Vec<f32> = (0..in_f).map(|i| ((i*40503usize)%1000) as f32/500.0-1.0).collect();
    let mut x2 = x1.clone(); x2.extend(x1.iter().map(|v| v*0.7+0.1));
    let wd = e.htod(&w)?; let x1d = e.htod(&x1)?; let x2d = e.htod(&x2)?;
    let y1 = e.linear(&x1d, &wd, 1, in_f, out_f)?;
    let y2 = e.linear(&x2d, &wd, 2, in_f, out_f)?;
    let h1 = e.dtoh(&y1)?; let h2 = e.dtoh(&y2)?;
    let mut maxd = 0.0f32; let mut nbit = 0;
    for i in 0..out_f {
        let d = (h1[i]-h2[i]).abs(); if d > maxd { maxd = d; }
        if h1[i].to_bits() != h2[i].to_bits() { nbit += 1; }
    }
    println!("cuBLASLt col0 m=1 vs m=2: bit-diff {}/{} maxdiff {:.3e}", nbit, out_f, maxd);
    Ok(())
}
