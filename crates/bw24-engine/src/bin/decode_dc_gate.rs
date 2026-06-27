//! CUDA-GRAPH-PLAN Phase 2 gate: the DEVICE-COUNTER decode path (`decode_step_dc`) must produce a
//! BIT-IDENTICAL greedy token stream to the eager `decode_step` over N steps, across >= 2 t_kv
//! buckets (prime P, then generate N crossing the powers-of-two split boundaries). Any token
//! mismatch = FAIL.
//!
//! decode_step_dc removes the two per-step VARYING host kernel-args by reading them from device
//! counters: the KV-append write slot (kvl.len_d) and the fa_decode t_kv bound (kvl.len_d post-inc),
//! plus it keeps the token id + rope pos device-resident. With bucket_max==t_kv the _dc fa_decode
//! reproduces the eager n_splits/per/combine EXACTLY -> bit-identical.
//!
//! usage: decode-dc-gate <model> [P] [N]    (env: BW24_FAST=1 BW24_MMVQ=1 BW24_FA_VEC=1)
use bw24_engine::Engine;
use bw24_engine::hybrid::HybridModel;
use bw24_engine::forward::argmax;
use bw24_gguf::GgufFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: decode-dc-gate <model> [P] [N]");
    let p: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(64);
    let n: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(256);
    let e = Engine::new(0)?;
    let g = GgufFile::open(&path)?;
    let m = HybridModel::load(&e, &g)?;
    let n_embd = m.cfg.n_embd as usize;
    let (qt, row_bytes) = m.embd.qt_and_row_bytes(n_embd);
    let embd_gpu = e.upload_u8(&m.embd.raw)?;

    let prompt: Vec<u32> = (0..p).map(|i| (100 + (i * 7) % 900) as u32).collect();

    // ---- EAGER reference: prime + generate N greedy tokens via decode_step ----
    let mut cache_e = bw24_engine::cache::Cache::new(&e, &m.cfg, p + n + 8)?;
    let mut ll = Vec::new();
    for &t in &prompt { ll = m.decode_step(&e, t, &mut cache_e)?; }
    let mut eager_in = argmax(&ll) as u32;           // first generated input token
    let mut eager_tokens = Vec::with_capacity(n);
    // ---- DEVICE-COUNTER path: prime an IDENTICAL second cache via decode_step_dc ----
    let mut cache_d = bw24_engine::cache::Cache::new(&e, &m.cfg, p + n + 8)?;
    let mut pos_d = e.htod_i32(&[0])?;               // resident rope pos counter (== cache.pos)
    let n_vocab = ll.len();
    // prime decode_step_dc with the prompt (feed each prompt token as a device-resident u32 id).
    for &t in &prompt {
        let cur_tok_d = upload_u32(&e, t)?;
        let _next = m.decode_step_dc(&e, &cur_tok_d, &mut pos_d, &embd_gpu, qt, row_bytes, &mut cache_d, n_vocab)?;
    }
    // after priming, the dc path's "next" input is the same first generated token as eager.
    let mut dc_in_d = upload_u32(&e, eager_in)?;

    let mut mismatches = 0usize;
    let mut first_mm: Option<(usize, u32, u32)> = None;
    let mut buckets_seen: Vec<usize> = Vec::new();
    for step in 0..n {
        // bucket boundary tracking (vec path splits at ceil(t_kv/64)): record the t_kv bucket.
        let t_kv = cache_e.kv.iter().filter_map(|k| k.as_ref()).map(|k| k.len + 1).next().unwrap_or(0);
        let bucket = ((t_kv + 63) / 64).max(1);
        if !buckets_seen.contains(&bucket) { buckets_seen.push(bucket); }

        // eager step
        ll = m.decode_step(&e, eager_in, &mut cache_e)?;
        let eager_next = argmax(&ll) as u32;
        eager_tokens.push(eager_next);

        // device-counter step (feed the SAME input token, kept device-resident)
        let dc_next_d = m.decode_step_dc(&e, &dc_in_d, &mut pos_d, &embd_gpu, qt, row_bytes, &mut cache_d, n_vocab)?;
        let dc_next = e.dtoh_u32_one(&dc_next_d)?;

        if dc_next != eager_next {
            mismatches += 1;
            if first_mm.is_none() { first_mm = Some((step, eager_next, dc_next)); }
            if mismatches <= 5 {
                println!("MISMATCH step {step} (t_kv={t_kv}, bucket={bucket}): eager={eager_next} dc={dc_next}");
            }
        }
        // canonical next input for BOTH paths = the eager token (so a single mismatch doesn't
        // desync the comparison; we still record every per-step disagreement).
        eager_in = eager_next;
        dc_in_d = upload_u32(&e, eager_next)?;
    }

    println!("buckets crossed (ceil(t_kv/64)): {:?}", buckets_seen);

    // ---- perf: time eager decode_step vs decode_step_dc (prime OUTSIDE the timer) ----
    if std::env::args().nth(4).as_deref() == Some("bench") {
        let bn = 256usize;
        // EAGER timed run
        let mut cache_eb = bw24_engine::cache::Cache::new(&e, &m.cfg, p + bn + 8)?;
        let mut llb = Vec::new();
        for &t in &prompt { llb = m.decode_step(&e, t, &mut cache_eb)?; }
        let mut ein = argmax(&llb) as u32;
        e.stream().synchronize()?;
        let t0 = std::time::Instant::now();
        for _ in 0..bn { llb = m.decode_step(&e, ein, &mut cache_eb)?; ein = argmax(&llb) as u32; }
        e.stream().synchronize()?;
        let dt_e = t0.elapsed().as_secs_f64();

        // DEVICE-COUNTER timed run
        let mut cache_db = bw24_engine::cache::Cache::new(&e, &m.cfg, p + bn + 8)?;
        let mut posb = e.htod_i32(&[0])?;
        for &t in &prompt {
            let td = upload_u32(&e, t)?;
            let _ = m.decode_step_dc(&e, &td, &mut posb, &embd_gpu, qt, row_bytes, &mut cache_db, n_vocab)?;
        }
        let mut din = upload_u32(&e, eager_tokens[0])?;
        e.stream().synchronize()?;
        let t1 = std::time::Instant::now();
        for _ in 0..bn {
            let nd = m.decode_step_dc(&e, &din, &mut posb, &embd_gpu, qt, row_bytes, &mut cache_db, n_vocab)?;
            din = nd;   // next token stays device-resident (no host argmax)
        }
        e.stream().synchronize()?;
        let dt_d = t1.elapsed().as_secs_f64();
        println!("decode tok/s  eager={:.1}  dc={:.1}  (tg{bn} @ctx{p}, ms/tok eager={:.2} dc={:.2})",
                 bn as f64 / dt_e, bn as f64 / dt_d, dt_e * 1000.0 / bn as f64, dt_d * 1000.0 / bn as f64);
    }
    if mismatches == 0 {
        println!("Phase-2 gate PASS: {n} steps decode_step_dc == decode_step (BIT-IDENTICAL), \
                  buckets={} (n_vocab={n_vocab})", buckets_seen.len());
    } else {
        let (s, a, b) = first_mm.unwrap();
        println!("Phase-2 gate FAIL: {mismatches}/{n} mismatches (first @ step {s}: eager={a} dc={b})");
        std::process::exit(1);
    }
    Ok(())
}

/// Build a device u32[1] holding `tok` (the resident decode input token id).
fn upload_u32(e: &Engine, tok: u32) -> Result<cudarc::driver::CudaSlice<u32>, Box<dyn std::error::Error>> {
    // htod of a &[u32] -> CudaSlice<u32>
    Ok(e.stream().clone_htod(&[tok])?)
}
