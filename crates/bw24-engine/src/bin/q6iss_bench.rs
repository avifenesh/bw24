//! lane/q6issue micro-bench: qmatvec_q6_K_mmvq baseline vs _iss/_issv issue-reduction variants
//! on SYNTHETIC q6_K weights at the two real lm_head shapes (9B: in_f=4096 out_f=151936,
//! 35B: in_f=2048 out_f=151936). Bit-identity gate first (exact f32 bit compare vs baseline),
//! then N-rep wall timing (weight is 510MB/255MB >> 64MB L2 -> every call streams DRAM-cold,
//! matching the real lm_head regime). Prints us/call + implied weight GB/s.
use bw24_engine::Engine;
use std::time::Instant;

// Deterministic LCG so runs are reproducible.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

/// Synthesize plausible q6_K blocks: random ql/qh, random int8 scales in [-32,31],
/// d = small positive half (0x2c00 ~ 0.0625 scaled by a random nudge) — no NaN/Inf paths.
fn synth_q6k(rows: usize, in_f: usize, rng: &mut Lcg) -> Vec<u8> {
    let blocks_per_row = in_f / 256;
    let row_bytes = blocks_per_row * 210;
    let mut w = vec![0u8; rows * row_bytes];
    for chunk in w.chunks_exact_mut(210) {
        for b in chunk[..192].iter_mut() {
            *b = rng.next() as u8;
        }
        for b in chunk[192..208].iter_mut() {
            *b = ((rng.next() % 64) as i32 - 32) as i8 as u8; // scales
        }
        // d: half in ~[0.03, 0.09]: exponent fixed, random mantissa bits
        let mant = (rng.next() & 0x3ff) as u16;
        let h: u16 = 0x2800 | mant;
        chunk[208..210].copy_from_slice(&h.to_le_bytes());
    }
    w
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let e = Engine::new(0)?;
    let reps: usize = std::env::var("Q6ISS_REPS").ok().and_then(|s| s.parse().ok()).unwrap_or(200);
    // Q6ISS_COPIES=N rotates the launch across N device copies of the weight so every call reads
    // DRAM-COLD bytes, like the real decode (trunk streams ~5GB through L2 between lm_head calls;
    // a 1-copy loop keeps ~64MB/486MB L2-hot and OVERSTATES — the mmvq-b4-clamp lesson).
    let copies: usize = std::env::var("Q6ISS_COPIES").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let out_f = 151936usize;
    for in_f in [4096usize, 2048] {
        let mut rng = Lcg(0x9b + in_f as u64);
        let row_bytes = in_f / 256 * 210;
        let w = synth_q6k(out_f, in_f, &mut rng);
        let wds: Vec<_> = (0..copies).map(|_| e.htod_bytes(&w)).collect::<Result<_, _>>()?;
        let wd = &wds[0];
        let x: Vec<f32> = (0..in_f).map(|i| (((i * 37 + 11) % 251) as f32 - 125.0) * 0.011).collect();
        let xd = e.htod(&x)?;
        let (aq, ad) = e.quantize_q8_1(&xd, 1, in_f)?;
        println!("=== shape in_f={in_f} out_f={out_f} row_bytes={row_bytes} weight={}MB reps={reps} ===",
                 w.len() / (1 << 20));

        // ---- bit-identity gate: baseline vs each variant, exact f32 bits ----
        let run = |flag: Option<&str>| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
            match flag {
                Some(v) => unsafe { std::env::set_var("BW24_Q6_ISSUE", v) },
                None => unsafe { std::env::remove_var("BW24_Q6_ISSUE") },
            }
            let y = e.qmatvec_mmvq(wd, &aq, &ad, 1, in_f, out_f, bw24_engine::QT_Q6_K,
                                   row_bytes, 1.0, false)?;
            e.stream().synchronize()?;
            Ok(e.dtoh(&y)?)
        };
        let y_base = run(None)?;
        for (name, flag) in [("iss", "1"), ("issv", "2"), ("issp", "3"), ("iss2g", "4"), ("iss2gp", "5")] {
            let y_v = run(Some(flag))?;
            let bad = y_base.iter().zip(&y_v)
                .filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            println!("bit-identity {name}: {} mismatched of {} {}",
                     bad, y_base.len(), if bad == 0 { "PASS" } else { "FAIL" });
            assert_eq!(bad, 0, "{name} not bit-identical");
        }

        // ---- timing ----
        let gb = (out_f * row_bytes) as f64 / 1e9;
        let mut order: Vec<(&str, Option<&str>)> =
            vec![("base", None), ("iss", Some("1")), ("issv", Some("2")),
                 ("issp", Some("3")), ("iss2g", Some("4")), ("iss2gp", Some("5"))];
        // Q6ISS_ORDER=rev reverses so thermal ramp biases AGAINST the later-run variants —
        // run fwd + rev and compare to control for heat-order bias in sustained runs.
        if std::env::var("Q6ISS_ORDER").as_deref() == Ok("rev") { order.reverse(); }
        for (name, flag) in order {
            match flag {
                Some(v) => unsafe { std::env::set_var("BW24_Q6_ISSUE", v) },
                None => unsafe { std::env::remove_var("BW24_Q6_ISSUE") },
            }
            for _ in 0..30 {
                let _ = e.qmatvec_mmvq(wd, &aq, &ad, 1, in_f, out_f, bw24_engine::QT_Q6_K,
                                       row_bytes, 1.0, false)?;
            }
            e.stream().synchronize()?;
            let t0 = Instant::now();
            for i in 0..reps {
                let _ = e.qmatvec_mmvq(&wds[i % copies], &aq, &ad, 1, in_f, out_f,
                                       bw24_engine::QT_Q6_K, row_bytes, 1.0, false)?;
            }
            e.stream().synchronize()?;
            let us = t0.elapsed().as_secs_f64() * 1e6 / reps as f64;
            println!("{name:>5}: {us:8.1} us/call   {:6.1} GB/s weight-stream ({:4.1}% of 858)",
                     gb / (us / 1e6), gb / (us / 1e6) / 858.0 * 100.0);
        }
        unsafe { std::env::remove_var("BW24_Q6_ISSUE") };
    }
    Ok(())
}
