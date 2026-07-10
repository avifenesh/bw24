//! rp_q4_probe: Q4_0 GGUF-block b4 vs the split-plane (rp) twin on the gemma verify-trunk
//! wq shape. Bitwise-gated inside Engine::rp_probe_q4. usage: rp_q4_probe [m]
use bw24_engine::Engine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let m: usize = std::env::args().nth(1).and_then(|v| v.parse().ok()).unwrap_or(3);
    let e = Engine::new(0)?;
    for rep in 0..3 {
        let (blk, rp) = e.rp_probe_q4(m)?;
        let bytes = (2048 * 88 * 18) as f64;
        println!("rep{rep} m={m}: blk {blk:7.2}us ({:5.1} GB/s) | rp {rp:7.2}us ({:5.1} GB/s) | {:4.2}x",
                 bytes / blk / 1e3, bytes / rp / 1e3, blk / rp);
    }
    Ok(())
}
