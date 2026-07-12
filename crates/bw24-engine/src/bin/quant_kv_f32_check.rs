//! Focused correctness gate for the quantized-KV-to-f32 SDPA fallback.
//!
//! The default prefill dispatch is intentionally untouched. This gate invokes the explicit f32
//! reference path, then checks both optimized quantized-view APIs against it.

use bw24_engine::Engine;

fn pr(i: usize) -> f32 {
    (((i.wrapping_mul(2_654_435_761)) >> 8) & 0xffff) as f32 / 32_768.0 - 1.0
}

fn rel_diff(reference: &[f32], actual: &[f32]) -> f32 {
    let max_diff = reference
        .iter()
        .zip(actual)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let scale = reference
        .iter()
        .map(|v| v.abs())
        .fold(0.0f32, f32::max)
        .max(1e-3);
    max_diff / scale
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = Engine::new(0)?;
    let mut failures = 0usize;

    // Includes the Hy3/M3 head geometry and the existing hd256 path. T < T_kv covers a
    // continuation prefill while the odd lengths exercise KV-tile and query-tile tails.
    for (head_dim, n_head, n_head_kv, t, t_kv, label) in [
        (128usize, 64usize, 8usize, 19usize, 37usize, "hd128-gqa"),
        (256, 16, 4, 37, 97, "hd256-gqa"),
    ] {
        let kv_dim = head_dim * n_head_kv;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q: Vec<f32> = (0..head_dim * n_head * t)
            .map(|i| pr(i + 5) * 0.2)
            .collect();
        let k: Vec<f32> = (0..kv_dim * t_kv).map(|i| pr(i + 7) * 0.2).collect();
        let v: Vec<f32> = (0..kv_dim * t_kv).map(|i| pr(i + 11) * 0.2).collect();
        let qd = engine.htod(&q)?;
        let kd = engine.htod(&k)?;
        let vd = engine.htod(&v)?;

        let (k_block_bytes, v_block_bytes) = bw24_engine::kv_blk_bytes();
        let k_tok_bytes = (kv_dim / 32) * k_block_bytes;
        let v_tok_bytes = (kv_dim / 32) * v_block_bytes;
        let mut k_cache = engine.alloc_u8(t_kv * k_tok_bytes)?;
        let mut v_cache = engine.alloc_u8(t_kv * v_tok_bytes)?;
        for tok in 0..t_kv {
            let k_row = kd.slice(tok * kv_dim..(tok + 1) * kv_dim);
            let v_row = vd.slice(tok * kv_dim..(tok + 1) * kv_dim);
            engine.append_kv_quantized_view(
                &k_row,
                &v_row,
                &mut k_cache,
                &mut v_cache,
                tok,
                kv_dim,
                kv_dim,
                k_tok_bytes,
                v_tok_bytes,
                false,
            )?;
        }
        let k_view = engine.view_u8(&k_cache, t_kv * k_tok_bytes);
        let v_view = engine.view_u8(&v_cache, t_kv * v_tok_bytes);

        let mut reference = engine.zeros(head_dim * n_head * t)?;
        engine.sdpa_naive_quantized_view(
            &qd,
            &k_view,
            &v_view,
            &mut reference,
            head_dim,
            n_head,
            n_head_kv,
            t,
            t_kv,
            scale,
            true,
            k_tok_bytes,
            v_tok_bytes,
        )?;
        let reference = engine.dtoh(&reference)?;

        let mut inline = engine.zeros(head_dim * n_head * t)?;
        engine.fa_prefill_view(
            &qd,
            &k_view,
            &v_view,
            &mut inline,
            head_dim,
            n_head,
            n_head_kv,
            t,
            t_kv,
            scale,
            true,
            k_tok_bytes,
            v_tok_bytes,
            false,
        )?;
        let inline = engine.dtoh(&inline)?;
        let inline_rel = rel_diff(&reference, &inline);

        let mut workspace = engine.zeros(head_dim * n_head * t)?;
        engine.fa_prefill_view_ws(
            &qd,
            &k_view,
            &v_view,
            &mut workspace,
            head_dim,
            n_head,
            n_head_kv,
            t,
            t_kv,
            scale,
            true,
            k_tok_bytes,
            v_tok_bytes,
            false,
        )?;
        let workspace = engine.dtoh(&workspace)?;
        let workspace_rel = rel_diff(&reference, &workspace);
        let optimized_bitdiff = inline
            .iter()
            .zip(&workspace)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();

        let ok = inline_rel < 1e-2 && workspace_rel < 1e-2 && optimized_bitdiff == 0;
        println!(
            "{label} T={t} Tkv={t_kv}: inline-rel={inline_rel:.2e} workspace-rel={workspace_rel:.2e} optimized-bitdiff={optimized_bitdiff} {}",
            if ok { "OK" } else { "FAIL" }
        );
        if !ok {
            failures += 1;
        }
    }

    if failures == 0 {
        println!("QUANT-KV-F32 GREEN");
        Ok(())
    } else {
        Err(format!("QUANT-KV-F32 FAIL ({failures})").into())
    }
}
