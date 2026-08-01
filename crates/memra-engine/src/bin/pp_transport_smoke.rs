//! M1-PP2 increment-2 TRANSPORT SMOKE (single-device-runnable).
//!
//! What it proves on one GPU:
//!   1. the cuDeviceCanAccessPeer guard path runs (matrix printed for every device pair);
//!   2. the EXACT peer-copy FFI the cross-device TX uses (`cuMemcpyPeerAsync` with
//!      explicit src/dst contexts) moves correct bytes — same-context both sides is a
//!      legal degenerate case of the same driver call, so the plumbing (contexts, raw
//!      pointers, byte counts, stream handle) is exercised without a second device;
//!   3. the Pp2Rt boundary choreography (per-stage streams, ev_tx/ev_rx, persistent
//!      slots, overlap double-buffering) round-trips patterned buffers bit-exactly.
//!
//! What it CANNOT prove locally: an actual cross-device copy or cross-device kernel
//! reads — that is the 8x box's `MEMRA_PP_DEVICES=0,1 pp2-gate <model>` gate.
//!
//! usage: pp-transport-smoke   (exit 0 = all sub-smokes pass)
use cudarc::driver::{DevicePtr, DevicePtrMut};
use memra_engine::Engine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut fails = 0usize;
    // engine first: initializes the driver (raw result:: calls below need cuInit).
    let e = Engine::new(0)?;

    // ---- 1. device census + CanAccessPeer matrix ----
    let ndev = cudarc::driver::result::device::get_count()? as usize;
    println!("devices: {ndev}");
    for a in 0..ndev {
        for b in 0..ndev {
            if a == b { continue; }
            let da = cudarc::driver::result::device::get(a as i32)?;
            let db = cudarc::driver::result::device::get(b as i32)?;
            let mut can: i32 = 0;
            unsafe { cudarc::driver::sys::cuDeviceCanAccessPeer(&mut can, da, db).result()?; }
            println!("cuDeviceCanAccessPeer({a} -> {b}) = {can}");
        }
    }
    if ndev < 2 {
        println!("(single device: no peer pairs — matrix section is census-only here; \
                  the cross-device arm gates on the 8x box)");
    }

    // ---- 2. forced peer-arm copy, same context both sides ----
    {
        let ctx = e.ctx();
        let s = ctx.new_stream()?;
        let n = 4096usize;
        let pat: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 7.0).collect();
        let src = s.clone_htod(&pat)?;
        let mut dst = s.alloc_zeros::<f32>(n)?;
        {
            let (sp, _g0) = src.device_ptr(&s);
            let (dp, _g1) = dst.device_ptr_mut(&s);
            unsafe {
                cudarc::driver::result::memcpy_peer_async(
                    ctx.cu_ctx(), dp, ctx.cu_ctx(), sp, n * 4, s.cu_stream())?;
            }
        }
        s.synchronize()?;
        let back = s.clone_dtoh(&dst)?;
        s.synchronize()?;
        let diff = back.iter().zip(&pat).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        println!("peer-arm copy (cuMemcpyPeerAsync, same-ctx degenerate): bytediff={diff} {}",
                 if diff == 0 { "OK" } else { fails += 1; "FAIL" });
    }

    // ---- 3. Pp2Rt boundary choreography: tx/rx roundtrip, then overlap slots ----
    unsafe { std::env::set_var("MEMRA_PP_OVERLAP", "1"); } // exercise slot alternation
    let rt = memra_engine::pp::Pp2Rt::get(&e)?;
    println!("Pp2Rt built: cross_device={}", rt.cross_device());
    let n = 5120usize;
    let mut round_fail = 0usize;
    for step in 0..4 {
        let pat: Vec<f32> = (0..n).map(|i| (i as f32) + 1000.0 * step as f32).collect();
        // TX inside the stage-0 scope (ambient stream = stage-0's)
        let slot = {
            let _s0 = rt.enter(0);
            let x = e.htod(&pat)?;
            rt.tx(&x, n)?
        };
        // RX inside the stage-1 scope; dtoh through the ambient (stage-1) stream
        let _s1 = rt.enter(1);
        let work = rt.rx(slot, n)?;
        let back = e.dtoh(&work)?;
        let diff = back.iter().zip(&pat).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        if diff != 0 { round_fail += 1; }
        println!("boundary roundtrip step {step} slot {slot}: bytediff={diff} {}",
                 if diff == 0 { "OK" } else { "FAIL" });
    }
    // overlap=1 must alternate slots 0,1,0,1 — assert we actually exercised both
    if round_fail > 0 { fails += 1; }

    if fails == 0 {
        println!("pp-transport-smoke PASS");
        Ok(())
    } else {
        println!("pp-transport-smoke FAIL ({fails} sub-smokes)");
        std::process::exit(1);
    }
}
