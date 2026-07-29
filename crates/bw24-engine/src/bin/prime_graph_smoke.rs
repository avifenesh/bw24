//! prime-graph-smoke (task #14 increment 1): does the FULL prime trunk (~900 launches,
//! cuBLASLt fp16 GEMMs, cp.async kernels, GDN mma pair) CAPTURE under RELAXED stream
//! capture and REPLAY with logits matching the eager prime? This is the design's riskiest
//! unknown — everything after it is plumbing (device counters, pointer table, buckets).
//!
//! Scope caveats (documented in the ledger): append host-len bookkeeping drifts across the
//! warmup runs (junk rows past the true KV window — logits are computed from the in-prime
//! f32 K/V, so the comparison is untouched); cache side-effect correctness is increment 3's
//! gate, not this smoke.
//!
//! usage: prime-graph-smoke <model.gguf> [--t 512]

use bw24_engine::cache::Cache;
use bw24_engine::forward::argmax;
use bw24_engine::hybrid::HybridModel;
use bw24_engine::Engine;
use bw24_gguf::GgufFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: prime-graph-smoke <model.gguf> [--t N]");
    let rest: Vec<String> = args.collect();
    let t: usize = rest.iter().position(|a| a == "--t")
        .and_then(|i| rest.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(512);
    // --true-len L: PAD-PROOF mode — bucket = t, real prompt = L tokens, rows [L, t) are
    // pads. The graph's logits must match the eager prime of the L-token prompt.
    let true_len: usize = rest.iter().position(|a| a == "--true-len")
        .and_then(|i| rest.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(t);
    assert!(true_len <= t);

    let e = Engine::new(0)?;
    let g = GgufFile::open(&path)?;
    let model = HybridModel::load_without_mtp(&e, &g)?;
    let prompt: Vec<u32> = (0..true_len as u32).map(|j| 55 + j * 31).collect();
    println!("loaded {} ({} layers); bucket T={t} true_len={true_len}",
             g.arch().unwrap_or("?"), model.layers.len());

    // eager reference
    let mut c_ref = Cache::new(&e, &model.cfg, t + 64)?;
    let (l_ref, _, _) = model.prime_cache(&e, &prompt, &mut c_ref)?;
    let a_ref = argmax(&l_ref) as u32;

    // graph arm: stable input buffers + capture
    let n_embd = model.cfg.n_embd as usize;
    let x_embed = model.embed(&e, &prompt)?;           // eager embed (stays outside the graph)
    let mut x_in = e.zeros(t * n_embd)?;               // stable graph input; pad rows ZERO
    e.copy_into(&mut x_in, 0, &x_embed, true_len * n_embd)?;
    let pos: Vec<i32> = (0..t as i32).collect();
    let pos_d = e.htod_i32(&pos)?;
    let len_d = e.htod_i32(&[true_len as i32])?;
    let mut c_g = Cache::new(&e, &model.cfg, t + 64)?;

    let n_vocab = l_ref.len();
    let logits_out = std::cell::RefCell::new(e.uninit(n_vocab)?);
    let h_seed_out = std::cell::RefCell::new(e.uninit(n_embd)?);
    // PRE-FLIGHT: run the closure body once OUTSIDE capture to separate closure bugs
    // from capture-legality bugs.
    {
        for kvl in c_g.kv.iter_mut().flatten() {
            kvl.len = 0;
            // len_d := 0 via memset (set_i32_one is a SYNCHRONOUS host memcpy — capture-illegal;
            // fresh-prime needs zero anyway, so the memset node is exact)
            e.stream().memset_zeros(&mut kvl.len_d)?;
        }
        for rl in c_g.recur.iter_mut().flatten() {
            e.stream().memset_zeros(&mut rl.conv_state)?;
            e.stream().memset_zeros(&mut rl.ssm_state)?;
            e.stream().memset_zeros(&mut rl.ssm_state_alt)?;
        }
        model.prime_chunk_captured(&e, &x_in, &pos_d, t, &mut c_g, &len_d,
                                   &mut logits_out.borrow_mut(), &mut h_seed_out.borrow_mut())?;
        e.stream().synchronize()?;
        let lv = e.dtoh(&logits_out.borrow())?;
        println!("pre-flight (no capture): argmax={} (eager {a_ref}) {}",
                 argmax(&lv), if argmax(&lv) as u32 == a_ref { "MATCH" } else { "MISMATCH" });
    }

    // MANUAL staged capture: pinpoint which stage throws (warmup / begin / body / end).
    {
        use cudarc::driver::sys::{CUstreamCaptureMode, CUgraphInstantiate_flags};
        let mut body = |e: &Engine| -> Result<(), Box<dyn std::error::Error>> {
            for kvl in c_g.kv.iter_mut().flatten() {
                kvl.len = 0;
                e.stream().memset_zeros(&mut kvl.len_d)?;
            }
            for rl in c_g.recur.iter_mut().flatten() {
                e.stream().memset_zeros(&mut rl.conv_state)?;
                e.stream().memset_zeros(&mut rl.ssm_state)?;
                e.stream().memset_zeros(&mut rl.ssm_state_alt)?;
            }
            model.prime_chunk_captured(e, &x_in, &pos_d, t, &mut c_g, &len_d,
                                       &mut logits_out.borrow_mut(), &mut h_seed_out.borrow_mut())?;
            Ok(())
        };
        body(&e).map_err(|er| format!("STAGE warmup1: {er}"))?;
        body(&e).map_err(|er| format!("STAGE warmup2: {er}"))?;
        e.stream().synchronize().map_err(|er| format!("STAGE sync: {er}"))?;
        let t0 = std::time::Instant::now();
        e.stream().begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .map_err(|er| format!("STAGE begin: {er}"))?;
        let r = body(&e);
        let g = e.stream().end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH);
        if let Err(er) = r { println!("STAGE body-in-capture: {er}"); }
        let graph = match g {
            Ok(Some(gr)) => { println!("STAGE end: capture INSTANTIATED OK ({:.0}ms)", t0.elapsed().as_secs_f64()*1e3); gr }
            Ok(None) => { println!("STAGE end: no graph"); return Ok(()); }
            Err(er) => { println!("STAGE end: {er}"); return Ok(()); }
        };
        let mut all_ok = true;
        for r in 0..3 {
            let t0 = std::time::Instant::now();
            graph.launch().map_err(|er| format!("replay launch: {er}"))?;
            e.stream().synchronize()?;
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            let l = e.dtoh(&logits_out.borrow())?;
            let a = argmax(&l) as u32;
            let md = l.iter().zip(&l_ref).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
            let ok = a == a_ref;
            all_ok &= ok;
            println!("replay {r}: {ms:.2}ms  argmax={a} (eager {a_ref})  maxdiff {md:.3e}  {}",
                     if ok { "MATCH" } else { "MISMATCH" });
        }
            // state diff vs the eager cache (same-m case must be bit-zero; padded cases show
        // the GEMM-shape numeric band)
        let (mut mc, mut ms) = (0f32, 0f32);
        for il in 0..c_ref.recur.len() {
            if let (Some(re), Some(rg)) = (&c_ref.recur[il], &c_g.recur[il]) {
                let a = e.dtoh(&re.conv_state)?; let b = e.dtoh(&rg.conv_state)?;
                mc = mc.max(a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max));
                let a = e.dtoh(&re.ssm_state)?; let b = e.dtoh(&rg.ssm_state)?;
                ms = ms.max(a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max));
            }
        }
        println!("scratch-vs-eager state: conv max {mc:.3e}  ssm max {ms:.3e}");
    println!("{}", if all_ok { "ALL GREEN: prime-graph smoke (manual capture)" } else { "SMOKE FAILED" });
        return Ok(());
    }
}
