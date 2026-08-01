//! Increment 2, deliverable 4: CPU-reference MLA attention-block forward at FIXTURE scale.
//!
//! Wires `memra_engine::mla` (the increment-1 absorbed-form oracle) to tensors actually READ
//! from a glm-dsa micro GGUF (generated at test time — nothing committed): the full projection
//! chain h -> attn_norm -> wq_a -> q_a_norm -> wq_b -> [nope|rope] split, h -> wkv_a ->
//! [c_kv|k_pe] split -> kv_a_norm, interleaved rope (GLM-5.2 `rope_interleave: true`), the
//! MLA core (naive AND absorbed), per-head wv_b decompression, wo. Logits-free: the gate is
//! the attention block output, absorbed vs naive, on the SAME loaded fixture tensors.
//!
//! This makes increment 3 (real weights, 78 layers, on the 8xH100 box) a scaling exercise:
//! the tensor layouts, split conventions, norm/rope order, and softmax scale are all pinned
//! and gated HERE, weights-free.

use memra_engine::mla::{mla_attend_absorbed, mla_attend_naive, rope_interleaved, MlaDims, MlaInputs};
use memra_gguf::config::ModelConfig;
use memra_gguf::micro_gguf::{write_glm_dsa_micro, Rng};
use memra_gguf::GgufFile;

const EPS: f32 = 1e-5;
const ROPE_BASE: f32 = 8_000_000.0;

fn f32s(g: &GgufFile, name: &str) -> Vec<f32> {
    let t = g.find(name).unwrap_or_else(|| panic!("missing {name}"));
    assert_eq!(t.ggml_type, memra_gguf::GgmlType::F32, "{name} must be F32 in the fixture");
    g.tensor_data(t)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// y[o] = sum_i w[o*in + i] * x[i]  — GGUF weight ne = {in, out}, row-major out rows.
fn matvec(w: &[f32], x: &[f32], in_f: usize, out_f: usize) -> Vec<f32> {
    assert_eq!(w.len(), in_f * out_f);
    assert_eq!(x.len(), in_f);
    (0..out_f)
        .map(|o| w[o * in_f..(o + 1) * in_f].iter().zip(x).map(|(a, b)| a * b).sum())
        .collect()
}

fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    x.iter().zip(w).map(|(v, g)| v * inv * g).collect()
}

fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

struct LayerW {
    attn_norm: Vec<f32>,
    wq_a: Vec<f32>,
    q_a_norm: Vec<f32>,
    wq_b: Vec<f32>,
    wkv_a: Vec<f32>,
    kv_a_norm: Vec<f32>,
    w_uk: Vec<f32>, // [n_head][d_nope][kv_rank] — mla.rs layout, from attn_k_b (un-transposed)
    w_uv: Vec<f32>, // [n_head][d_v][kv_rank]    — mla.rs layout, from attn_v_b (direct)
    wo: Vec<f32>,
}

/// Read layer `il`'s attention tensors and rearrange the 3D split tensors into the mla.rs
/// weight layouts, CROSS-CHECKED against the unsplit attn_kv_b (the conversion-split gate:
/// if the loader's layout assumption were wrong, this assert fires — not a silent maxdiff).
fn load_layer(g: &GgufFile, il: u32, d: &MlaDims) -> LayerW {
    let p = |s: &str| format!("blk.{il}.{s}");
    let (nh, dn, dv, r) = (d.n_head, d.d_nope, d.d_v, d.kv_rank);

    // attn_k_b ne {nope, kv_rank, head}: element (h, rank, nope) — TRANSPOSE to [h][nope][rank].
    let k_b = f32s(g, &p("attn_k_b.weight"));
    let mut w_uk = vec![0.0f32; nh * dn * r];
    for h in 0..nh {
        for rr in 0..r {
            for pn in 0..dn {
                w_uk[h * dn * r + pn * r + rr] = k_b[h * r * dn + rr * dn + pn];
            }
        }
    }
    // attn_v_b ne {kv_rank, v, head}: element (h, v, rank) — ALREADY [h][v][rank].
    let w_uv = f32s(g, &p("attn_v_b.weight"));
    assert_eq!(w_uv.len(), nh * dv * r);

    // Cross-check against the unsplit kv_b {kv_rank, head*(nope+v)}: per head, first `nope`
    // rows are w_uk[h] (rank fastest), next `v` rows are w_uv[h]. Bit-equality required —
    // the fixture derives the splits from kv_b, so any mismatch is a LAYOUT bug here/loader.
    let kv_b = f32s(g, &p("attn_kv_b.weight"));
    for h in 0..nh {
        for pn in 0..dn {
            for rr in 0..r {
                assert_eq!(
                    w_uk[h * dn * r + pn * r + rr],
                    kv_b[(h * (dn + dv) + pn) * r + rr],
                    "w_uk vs unsplit kv_b: layer {il} head {h} nope {pn} rank {rr}"
                );
            }
        }
        for j in 0..dv {
            for rr in 0..r {
                assert_eq!(
                    w_uv[h * dv * r + j * r + rr],
                    kv_b[(h * (dn + dv) + dn + j) * r + rr],
                    "w_uv vs unsplit kv_b: layer {il} head {h} v {j} rank {rr}"
                );
            }
        }
    }

    LayerW {
        attn_norm: f32s(g, &p("attn_norm.weight")),
        wq_a: f32s(g, &p("attn_q_a.weight")),
        q_a_norm: f32s(g, &p("attn_q_a_norm.weight")),
        wq_b: f32s(g, &p("attn_q_b.weight")),
        wkv_a: f32s(g, &p("attn_kv_a_mqa.weight")),
        kv_a_norm: f32s(g, &p("attn_kv_a_norm.weight")),
        w_uk,
        w_uv,
        wo: f32s(g, &p("attn_output.weight")),
    }
}

/// Projection chain for `t` tokens at positions pos0..pos0+t: returns per-token
/// (q_nope [t][nh][dn], roped q_pe [t][nh][dr], latent rows c_kv [t][r], roped k_pe [t][dr]).
#[allow(clippy::type_complexity)]
fn project(
    lw: &LayerW,
    d: &MlaDims,
    h_in: &[f32],
    n_embd: usize,
    q_lora: usize,
    t: usize,
    pos0: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let (nh, dn, dr, r) = (d.n_head, d.d_nope, d.d_rope, d.kv_rank);
    let mut q_nope = Vec::with_capacity(t * nh * dn);
    let mut q_pe = Vec::with_capacity(t * nh * dr);
    let mut c_kv = Vec::with_capacity(t * r);
    let mut k_pe = Vec::with_capacity(t * dr);
    for i in 0..t {
        let pos = (pos0 + i) as f32;
        let x = rms_norm(&h_in[i * n_embd..(i + 1) * n_embd], &lw.attn_norm, EPS);
        // q path: wq_a -> q_a_norm -> wq_b -> per-head [nope | rope], rope INTERLEAVED
        let q_a = matvec(&lw.wq_a, &x, n_embd, q_lora);
        let q_lat = rms_norm(&q_a, &lw.q_a_norm, EPS);
        let q = matvec(&lw.wq_b, &q_lat, q_lora, nh * (dn + dr));
        for hd in 0..nh {
            q_nope.extend_from_slice(&q[hd * (dn + dr)..hd * (dn + dr) + dn]);
            let mut pe = q[hd * (dn + dr) + dn..(hd + 1) * (dn + dr)].to_vec();
            rope_interleaved(&mut pe, dr, pos, ROPE_BASE);
            q_pe.extend_from_slice(&pe);
        }
        // kv path: wkv_a -> [c_kv (rms-normed) | k_pe (roped, NOT normed)]
        let kv = matvec(&lw.wkv_a, &x, n_embd, r + dr);
        c_kv.extend_from_slice(&rms_norm(&kv[..r], &lw.kv_a_norm, EPS));
        let mut pe = kv[r..].to_vec();
        rope_interleaved(&mut pe, dr, pos, ROPE_BASE);
        k_pe.extend_from_slice(&pe);
    }
    (q_nope, q_pe, c_kv, k_pe)
}

/// wo(concat_heads(attn_out)) for each of t_q tokens.
fn out_proj(lw: &LayerW, attn: &[f32], nh: usize, dv: usize, n_embd: usize, t_q: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(t_q * n_embd);
    for i in 0..t_q {
        out.extend(matvec(&lw.wo, &attn[i * nh * dv..(i + 1) * nh * dv], nh * dv, n_embd));
    }
    out
}

/// Run the whole attention block (projections shared, core = naive vs absorbed) and gate.
fn gate_block(g: &GgufFile, cfg: &ModelConfig, il: u32, t_q: usize, t_kv: usize, seed: u64) {
    let m = cfg.mla.as_ref().expect("fixture parses MlaConfig");
    let d = MlaDims {
        n_head: cfg.n_head as usize,
        d_nope: m.qk_nope_head_dim as usize,
        d_rope: m.qk_rope_head_dim as usize,
        d_v: m.v_head_dim as usize,
        kv_rank: m.kv_lora_rank as usize,
    };
    let n_embd = cfg.n_embd as usize;
    let lw = load_layer(g, il, &d);

    // random hidden states for the whole context window (cache tokens + queries)
    let mut rng = Rng(seed | 1);
    let h_in = rng.fill(t_kv * n_embd, 1.0);
    let (q_nope_all, q_pe_all, c_kv, k_pe) =
        project(&lw, &d, &h_in, n_embd, m.q_lora_rank as usize, t_kv, 0);
    // queries = the LAST t_q positions (mla.rs suffix convention)
    let qo = (t_kv - t_q) * d.n_head;
    let x = MlaInputs {
        q_nope: &q_nope_all[qo * d.d_nope..],
        q_pe: &q_pe_all[qo * d.d_rope..],
        c_kv: &c_kv,
        k_pe: &k_pe,
        w_uk: &lw.w_uk,
        w_uv: &lw.w_uv,
        t_q,
        t_kv,
    };
    // sanity: the metadata-derived softmax scale is the ORIGINAL qk dim (1/sqrt(nope+rope))
    assert!((d.scale() - m.scale()).abs() < 1e-9);

    let naive = mla_attend_naive(&d, &x);
    let absorbed = mla_attend_absorbed(&d, &x);
    let out_n = out_proj(&lw, &naive, d.n_head, d.d_v, n_embd, t_q);
    let out_a = out_proj(&lw, &absorbed, d.n_head, d.d_v, n_embd, t_q);

    let md = maxdiff(&out_n, &out_a);
    let scale = out_n.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-3);
    assert!(
        md <= 1e-5 * scale.max(1.0),
        "layer {il}: absorbed vs naive block-out maxdiff {md:.3e} (scale {scale:.3e}) \
         t_q {t_q} t_kv {t_kv}"
    );
    assert!(out_n.iter().all(|v| v.is_finite()));
    assert!(scale > 1e-3, "block output degenerate (all ~zero) — a projection is dead");
}

fn fixture() -> (std::path::PathBuf, GgufFile, ModelConfig) {
    let p = std::env::temp_dir().join(format!(
        "memra-mla-fwd-{}-{:?}.gguf",
        std::process::id(),
        std::thread::current().id()
    ));
    write_glm_dsa_micro(&p, 0x1_9C0_0802).unwrap();
    let g = GgufFile::open(&p).unwrap();
    let cfg = ModelConfig::from_gguf(&g);
    (p, g, cfg)
}

/// Causal prefill (t_q = t_kv): every trunk layer + the MTP block (dense MLA, same math).
#[test]
fn cpu_block_forward_absorbed_equals_naive_prefill() {
    let (p, g, cfg) = fixture();
    for il in 0..cfg.n_layer {
        gate_block(&g, &cfg, il, 6, 6, 42 + il as u64);
    }
    std::fs::remove_file(&p).ok();
}

/// Decode shape (t_q = 1 against a populated cache) — the production MLA regime.
#[test]
fn cpu_block_forward_absorbed_equals_naive_decode() {
    let (p, g, cfg) = fixture();
    for il in 0..cfg.n_layer {
        gate_block(&g, &cfg, il, 1, 9, 1234 + il as u64);
    }
    std::fs::remove_file(&p).ok();
}

/// Mixed: t_q new tokens over past context (chunked-prefill shape).
#[test]
fn cpu_block_forward_absorbed_equals_naive_chunked() {
    let (p, g, cfg) = fixture();
    gate_block(&g, &cfg, 0, 3, 11, 0xB1E55ED);
    std::fs::remove_file(&p).ok();
}
