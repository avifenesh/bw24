//! FAVENDOR lane micro-bench (2026-07-08): fa_decode at the REAL 35B full-attn shape
//! (n_head=16, n_head_kv=2, head_dim=256, q8_0 K / q5_1 V) on synthetic KV, per-depth.
//! Deep synthetic cache (t_kv_max=12288) for the depth-law matrix.
//!
//! Sweep configs via env, ONE config per process (fa_decode reads the flags per call but
//! SMEM_TKV/SPLIT are process OnceLocks):
//!   BW24_FA_V2={0,1}  BW24_FA_SMEM_TKV=...  BW24_FA_SPLIT=...
//! Prints per-t_kv: us/call, implied unique-KV GB/s, and an FNV-1a output hash.
//! NOTE: the hash is expected to DIFFER between BW24_FA_V2=0 and =1 (v2 is its own numeric
//! config — tile-level softmax regrouping); it must be IDENTICAL across runs of the SAME config.
use bw24_engine::Engine;

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes { h ^= b as u64; h = h.wrapping_mul(0x100000001b3); }
    h
}

// deterministic LCG so every process builds the IDENTICAL synthetic cache
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 { self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (self.0 >> 33) as u32 }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let e = Engine::new(0)?;
    // Default: 35B full-attn geometry (Qwen3.6-35B-A3B gguf metadata).
    // BW24_FA_SHAPE=9b selects the 9B geometry (nh16/nkv4/hd256, gqa=4 — gguf metadata);
    // BW24_FA_SHAPE=nh,nkv,hd sets it explicitly (FA v3 lane: v3 must not regress the 9B shape).
    let (n_head, n_head_kv, head_dim) = match std::env::var("BW24_FA_SHAPE").as_deref() {
        Ok("9b") => (16usize, 4usize, 256usize),
        Ok(s) if s.contains(',') => {
            let p: Vec<usize> = s.split(',').filter_map(|x| x.parse().ok()).collect();
            assert_eq!(p.len(), 3, "BW24_FA_SHAPE=nh,nkv,hd");
            (p[0], p[1], p[2])
        }
        _ => (16usize, 2usize, 256usize),
    };
    let kv_dim = n_head_kv * head_dim;              // 512
    let nblk = kv_dim / 32;                         // 16 blocks/token
    let k_tok_bytes = nblk * 34;                    // q8_0
    let v_tok_bytes = nblk * 24;                    // q5_1
    let t_kv_max = 12288usize;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    // f32 -> f16 (normal range only — enough for the fixed scale constants below)
    fn f16le(f: f32) -> [u8; 2] {
        let b = f.to_bits();
        let sign = ((b >> 16) & 0x8000) as u16;
        let exp = ((b >> 23) & 0xFF) as i32 - 127 + 15;
        let man = ((b >> 13) & 0x3FF) as u16;
        (sign | ((exp as u16) << 10) | man).to_le_bytes()
    }
    let half = f16le;
    let mut rng = Lcg(0x5eed_2026_0708);
    let mut kbytes = vec![0u8; t_kv_max * k_tok_bytes];
    for t in 0..t_kv_max {
        for b in 0..nblk {
            let off = t * k_tok_bytes + b * 34;
            kbytes[off..off + 2].copy_from_slice(&half(0.02));
            for i in 0..32 { kbytes[off + 2 + i] = (rng.next() & 0xFF) as u8; }
        }
    }
    let mut vbytes = vec![0u8; t_kv_max * v_tok_bytes];
    for t in 0..t_kv_max {
        for b in 0..nblk {
            let off = t * v_tok_bytes + b * 24;
            vbytes[off..off + 2].copy_from_slice(&half(0.01));
            vbytes[off + 2..off + 4].copy_from_slice(&half(-0.1));
            let qh = rng.next();
            vbytes[off + 4..off + 8].copy_from_slice(&qh.to_le_bytes());
            for i in 0..16 { vbytes[off + 8 + i] = (rng.next() & 0xFF) as u8; }
        }
    }
    let kd = e.htod_bytes(&kbytes)?;
    let vd = e.htod_bytes(&vbytes)?;
    let qh: Vec<f32> = (0..n_head * head_dim).map(|_| (rng.next() as f32 / u32::MAX as f32) - 0.5).collect();
    let qd = e.htod(&qh)?;

    let v2 = std::env::var("BW24_FA_V2").unwrap_or_default();
    let v3 = std::env::var("BW24_FA_V3").unwrap_or_default();
    let smem = std::env::var("BW24_FA_SMEM_TKV").unwrap_or_else(|_| "1024".into());
    let split = std::env::var("BW24_FA_SPLIT").unwrap_or_else(|_| "default".into());
    println!("# config v2={v2} v3={v3} smem_tkv={smem} split={split}  shape nh={n_head} nkv={n_head_kv} hd={head_dim}");

    const REPS: usize = 200;
    let tkvs: Vec<usize> = {
        let a: Vec<usize> = std::env::args().skip(1).filter_map(|s| s.parse().ok()).collect();
        if a.is_empty() { vec![512, 2048, 6257, 12288] } else { a }
    };
    for &t_kv in &tkvs {
        let kv = e.view_u8(&kd, t_kv * k_tok_bytes);
        let vv = e.view_u8(&vd, t_kv * v_tok_bytes);
        let mut od = e.uninit(n_head * head_dim)?;
        // correctness snapshot (before timing): output bytes hash
        e.fa_decode(&qd, &kv, &vv, &mut od, head_dim, n_head, n_head_kv, t_kv, scale, k_tok_bytes, v_tok_bytes)?;
        let out = e.dtoh(&od)?;
        let hash = fnv1a(bytemuck_bytes(&out));
        // warmup
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
        println!("t_kv={t_kv:5}  {us:8.2} us/call  uniqKV {gbs:6.1} GB/s  out_hash={hash:016x}");
    }
    Ok(())
}

fn bytemuck_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
