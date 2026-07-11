// Focused FA-prefill sanitizer harness: exercises fa_prefill_f32 AND fa_prefill_q
// (the quant twin) with the SAME shapes/tolerances as kernel_check's FA section,
// WITHOUT the unrelated SSM/other kernels that abort the full kernel_check under
// compute-sanitizer. Run under: compute-sanitizer --tool memcheck|racecheck.
use bw24_engine::Engine;

fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}
fn pr(i: usize) -> f32 { (((i.wrapping_mul(2654435761)) >> 8) & 0xffff) as f32 / 32768.0 - 1.0 }

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let e = Engine::new(0)?;
    let (hd, nh, nhkv) = (256usize, 16usize, 4usize);
    let scale = 1.0 / (hd as f32).sqrt();
    let cpu_sdpa = |q: &[f32], k: &[f32], v: &[f32], t: usize, tkv: usize| -> Vec<f32> {
        let mut o = vec![0f32; hd * nh * t];
        for head in 0..nh {
            let kvh = head / (nh / nhkv);
            for qt in 0..t {
                let q_pos = (tkv - t) + qt;
                let qv = &q[(qt * nh + head) * hd..][..hd];
                let mut sc = vec![0f32; tkv];
                for tk in 0..tkv {
                    let kv = &k[(tk * nhkv + kvh) * hd..][..hd];
                    let mut a = 0.0; for d in 0..hd { a += qv[d] * kv[d]; }
                    a *= scale; if tk > q_pos { a = -1e30; } sc[tk] = a;
                }
                let mx = sc.iter().cloned().fold(-1e30f32, f32::max);
                let mut sum = 0.0; for s in sc.iter_mut() { *s = (*s - mx).exp(); sum += *s; }
                for s in sc.iter_mut() { *s /= sum; }
                let ov = &mut o[(qt * nh + head) * hd..][..hd];
                for d in 0..hd { let mut a = 0.0; for tk in 0..tkv { a += sc[tk] * v[(tk*nhkv+kvh)*hd+d]; } ov[d] = a; }
            }
        }
        o
    };
    let mut fails = 0;
    // f32 prefill (fa_prefill_f32)
    for (t, tkv) in [(16usize, 16usize), (64, 64), (100, 100), (256, 256)] {
        let q: Vec<f32> = (0..hd*nh*t).map(|i| pr(i)*0.2).collect();
        let k: Vec<f32> = (0..hd*nhkv*tkv).map(|i| pr(i+7)*0.2).collect();
        let v: Vec<f32> = (0..hd*nhkv*tkv).map(|i| pr(i+11)*0.2).collect();
        let cpu = cpu_sdpa(&q,&k,&v,t,tkv);
        let qd=e.htod(&q)?; let kd=e.htod(&k)?; let vd=e.htod(&v)?; let mut od=e.zeros(hd*nh*t)?;
        e.fa_prefill(&qd,&kd,&vd,&mut od,hd,nh,nhkv,t,tkv,scale,true)?;
        let g=e.dtoh(&od)?; let d=maxdiff(&cpu,&g);
        let sc=cpu.iter().map(|v|v.abs()).fold(0.0,f32::max).max(1e-3); let rel=d/sc;
        println!("fa_prefill_f32 T={t} Tkv={tkv}: rel={rel:.2e} {}", if rel<2e-2 {"OK"} else {fails+=1;"FAIL"});
    }
    // quant prefill twin (fa_prefill_q via fa_prefill_view) — quantize K/V then run.
    let kv_dim_k = hd * nhkv; let kv_dim_v = hd * nhkv;
    let (kbb, vbb) = bw24_engine::kv_blk_bytes();  // env-selected KV formats
    let k_tok_bytes = (kv_dim_k / 32) * kbb; let v_tok_bytes = (kv_dim_v / 32) * vbb;
    for (t, tkv) in [(16usize, 16usize), (64, 64), (100, 100), (256, 256)] {
        let q: Vec<f32> = (0..hd*nh*t).map(|i| pr(i)*0.2).collect();
        let k: Vec<f32> = (0..hd*nhkv*tkv).map(|i| pr(i+7)*0.2).collect();
        let v: Vec<f32> = (0..hd*nhkv*tkv).map(|i| pr(i+11)*0.2).collect();
        let cpu = cpu_sdpa(&q,&k,&v,t,tkv);
        let qd=e.htod(&q)?; let kd=e.htod(&k)?; let vd=e.htod(&v)?;
        let mut kc = e.alloc_u8(tkv * k_tok_bytes)?;
        let mut vc = e.alloc_u8(tkv * v_tok_bytes)?;
        for tok in 0..tkv {
            let k_row = kd.slice(tok*kv_dim_k..(tok+1)*kv_dim_k);
            let v_row = vd.slice(tok*kv_dim_v..(tok+1)*kv_dim_v);
            e.append_kv_quantized_view(&k_row,&v_row,&mut kc,&mut vc,tok,kv_dim_k,kv_dim_v,k_tok_bytes,v_tok_bytes, false)?;
        }
        let kview=e.view_u8(&kc, tkv*k_tok_bytes); let vview=e.view_u8(&vc, tkv*v_tok_bytes);
        let mut od=e.zeros(hd*nh*t)?;
        e.fa_prefill_view(&qd,&kview,&vview,&mut od,hd,nh,nhkv,t,tkv,scale,true,k_tok_bytes,v_tok_bytes)?;
        let g=e.dtoh(&od)?; let d=maxdiff(&cpu,&g);
        let sc=cpu.iter().map(|v|v.abs()).fold(0.0,f32::max).max(1e-3); let rel=d/sc;
        // quant twin: looser (q8_0 K / q5_1 V) — gate at q5_1 noise floor
        println!("fa_prefill_q   T={t} Tkv={tkv}: rel={rel:.2e} {}", if rel<6e-2 {"OK"} else {fails+=1;"FAIL"});
    }
    if fails == 0 { println!("FA-SANITIZE GREEN"); } else { println!("FA-SANITIZE FAIL ({fails})"); std::process::exit(1); }
    Ok(())
}
