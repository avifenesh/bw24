//! KVBYTES lane micro-bench (2026-07-08): fa_decode at the REAL 35B full-attn shape
//! (n_head=16, n_head_kv=2, head_dim=256) on a synthetic cache built through the REAL
//! append kernel — so it runs on ANY env-selected KV format (BW24_KV_K / BW24_KV_V),
//! unlike fa_ppool_bench which hand-crafts q8_0/q5_1 bytes.
//!
//! Per t_kv (default 2048 6257 12288; args override): times N reps of the full
//! fa_decode host call, prints us/call, the format's unique-KV bytes/token + implied
//! GB/s, and an FNV-1a output hash (identical across reps; DIFFERS across formats —
//! each format is its own numeric config).
//!
//! Run (env law): BW24_FAST=1 BW24_FA_VEC=1 [BW24_KV_K=fp8] [BW24_KV_V=q4_0] \
//!   cargo run --release -p bw24-engine --bin kv-fmt-bench [t_kv...]
use bw24_engine::Engine;

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes { h ^= b as u64; h = h.wrapping_mul(0x100000001b3); }
    h
}

// deterministic LCG so every process builds the IDENTICAL synthetic K/V rows
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 { self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (self.0 >> 33) as u32 }
    fn unit(&mut self) -> f32 { (self.next() as f32 / u32::MAX as f32) - 0.5 }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (kfmt, vfmt) = bw24_engine::kv_cache_formats();
    let (kbb, vbb) = bw24_engine::kv_blk_bytes();
    let e = Engine::new(0)?;
    // 35B full-attn geometry (Qwen3.6-35B-A3B gguf metadata)
    let (n_head, n_head_kv, head_dim) = (16usize, 2usize, 256usize);
    let kv_dim = n_head_kv * head_dim;              // 512
    let nblk = kv_dim / 32;                         // 16 blocks/token
    let k_tok_bytes = nblk * kbb;
    let v_tok_bytes = nblk * vbb;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let tkvs: Vec<usize> = {
        let a: Vec<usize> = std::env::args().skip(1).filter_map(|s| s.parse().ok()).collect();
        if a.is_empty() { vec![2048, 6257, 12288] } else { a }
    };
    let t_kv_max = *tkvs.iter().max().unwrap();

    // synthetic f32 K/V rows (scaled ~[-0.65, 0.65], K a bit wider like post-QK-norm keys),
    // quantized into the resident cache via the REAL batched append kernel (format-correct
    // bytes for whatever fatbin the engine loaded).
    let mut rng = Lcg(0x5eed_2026_0708);
    let krows: Vec<f32> = (0..t_kv_max * kv_dim).map(|_| rng.unit() * 2.6).collect();
    let vrows: Vec<f32> = (0..t_kv_max * kv_dim).map(|_| rng.unit() * 1.3).collect();
    let krd = e.htod(&krows)?;
    let vrd = e.htod(&vrows)?;
    let mut kc = e.alloc_u8(t_kv_max * k_tok_bytes)?;
    let mut vc = e.alloc_u8(t_kv_max * v_tok_bytes)?;
    e.append_kv_quantized_rows(&krd, &vrd, &mut kc, &mut vc, 0, t_kv_max,
                               kv_dim, kv_dim, k_tok_bytes, v_tok_bytes)?;
    let qh: Vec<f32> = (0..n_head * head_dim).map(|_| rng.unit()).collect();
    let qd = e.htod(&qh)?;

    println!("# kv-fmt-bench K={kfmt}({kbb}B/32) V={vfmt}({vbb}B/32)  tok_bytes k={k_tok_bytes} v={v_tok_bytes}  shape nh={n_head} nkv={n_head_kv} hd={head_dim}");

    const REPS: usize = 200;
    for &t_kv in &tkvs {
        let kv = e.view_u8(&kc, t_kv * k_tok_bytes);
        let vv = e.view_u8(&vc, t_kv * v_tok_bytes);
        let mut od = e.uninit(n_head * head_dim)?;
        // correctness snapshot (before timing): output bytes hash, rep-stable
        e.fa_decode(&qd, &kv, &vv, &mut od, head_dim, n_head, n_head_kv, t_kv, scale, k_tok_bytes, v_tok_bytes)?;
        let out = e.dtoh(&od)?;
        let hash = fnv1a(bytemuck_bytes(&out));
        let finite = out.iter().all(|x| x.is_finite());
        for _ in 0..10 {
            e.fa_decode(&qd, &kv, &vv, &mut od, head_dim, n_head, n_head_kv, t_kv, scale, k_tok_bytes, v_tok_bytes)?;
        }
        e.stream().synchronize()?;
        let t0 = std::time::Instant::now();
        for _ in 0..REPS {
            e.fa_decode(&qd, &kv, &vv, &mut od, head_dim, n_head, n_head_kv, t_kv, scale, k_tok_bytes, v_tok_bytes)?;
        }
        e.stream().synchronize()?;
        let us = t0.elapsed().as_secs_f64() * 1e6 / REPS as f64;
        let uniq_bytes = t_kv * (k_tok_bytes + v_tok_bytes);
        let gbs = uniq_bytes as f64 / (us * 1e-6) / 1e9;
        println!("t_kv={t_kv:5}  {us:8.2} us/call  uniqKV {:.3} MB ({gbs:6.1} GB/s)  out_hash={hash:016x} finite={finite}",
                 uniq_bytes as f64 / 1e6);
    }
    Ok(())
}

fn bytemuck_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
