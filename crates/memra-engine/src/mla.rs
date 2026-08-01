//! MLA (multi-head latent attention, DeepSeek lineage / GLM-5 "MLA-256") — CPU f32 reference.
//!
//! Increment 1 of the GLM-5.2 bring-up lane (`research/mla-bringup-20260801/DESIGN.md`).
//! This module pins the decode-path math BEFORE any kernel work: both the naive form
//! (decompress the latent cache to per-head K/V, then attend — vLLM "forward_mha") and the
//! absorbed form (fold W_UK into the query, attend in latent space as MQA, decompress the
//! output through W_UV — vLLM "forward_mqa", llama.cpp glm-dsa.cpp). The unit tests prove the
//! two forms agree to f32 tolerance on random inputs across shapes (t=1 decode and small
//! causal prefill), including full GLM-5.2 dims (64 heads, nope 192, rope 64, v 256, rank 512).
//!
//! Also pinned here: the interleaved ("NORM", `rope_interleave: true`) vs NEOX rope pairing and
//! the load-time permutation that maps one onto the other (DESIGN.md §1.4) — memra only ships a
//! NEOX kernel, GLM-5.2 needs NORM, and the permutation trick lets the existing kernel serve.
//!
//! Everything is plain CPU f32, no CUDA, no engine deps: this is the permanent oracle for the
//! MLA kernel family's maxdiff gates.

/// MLA head geometry. GLM-5.2: n_head=64, d_nope=192, d_rope=64, d_v=256, kv_rank=512.
#[derive(Clone, Copy, Debug)]
pub struct MlaDims {
    pub n_head: usize,
    /// qk nope head dim (P)
    pub d_nope: usize,
    /// qk rope head dim (R); latent cache row = kv_rank + d_rope
    pub d_rope: usize,
    /// v head dim (V)
    pub d_v: usize,
    /// kv lora rank (Lkv)
    pub kv_rank: usize,
}

impl MlaDims {
    pub const GLM52: MlaDims =
        MlaDims { n_head: 64, d_nope: 192, d_rope: 64, d_v: 256, kv_rank: 512 };

    /// Softmax scale: 1/sqrt(d_nope + d_rope) — the ORIGINAL qk head dim (256 for GLM-5.2),
    /// NOT the absorbed width (576). llama.cpp glm-dsa.cpp `kq_scale` with mscale=1 (no yarn).
    pub fn scale(&self) -> f32 {
        1.0 / ((self.d_nope + self.d_rope) as f32).sqrt()
    }
}

/// Inputs shared by both forms. Rope is already applied to `q_pe`/`k_pe` (it happens upstream
/// of the attention core and is identical in both forms). `c_kv` is already RMS-normed.
///
/// Layouts (row-major):
///   q_nope: [t_q][n_head][d_nope]
///   q_pe:   [t_q][n_head][d_rope]
///   c_kv:   [t_kv][kv_rank]           — the latent KV cache (one row per token, all heads)
///   k_pe:   [t_kv][d_rope]            — decoupled rope key (one per token, all heads)
///   w_uk:   [n_head][d_nope][kv_rank] — k_nope_h = w_uk[h] · c_kv
///   w_uv:   [n_head][d_v][kv_rank]    — v_h      = w_uv[h] · c_kv
///
/// The queries occupy the LAST `t_q` positions of the cache (decode/prefill convention:
/// their own rows are already appended). Causal: query i attends to cache rows
/// 0 ..= (t_kv - t_q + i).
pub struct MlaInputs<'a> {
    pub q_nope: &'a [f32],
    pub q_pe: &'a [f32],
    pub c_kv: &'a [f32],
    pub k_pe: &'a [f32],
    pub w_uk: &'a [f32],
    pub w_uv: &'a [f32],
    pub t_q: usize,
    pub t_kv: usize,
}

fn check_shapes(d: &MlaDims, x: &MlaInputs) {
    assert_eq!(x.q_nope.len(), x.t_q * d.n_head * d.d_nope, "q_nope shape");
    assert_eq!(x.q_pe.len(), x.t_q * d.n_head * d.d_rope, "q_pe shape");
    assert_eq!(x.c_kv.len(), x.t_kv * d.kv_rank, "c_kv shape");
    assert_eq!(x.k_pe.len(), x.t_kv * d.d_rope, "k_pe shape");
    assert_eq!(x.w_uk.len(), d.n_head * d.d_nope * d.kv_rank, "w_uk shape");
    assert_eq!(x.w_uv.len(), d.n_head * d.d_v * d.kv_rank, "w_uv shape");
    assert!(x.t_q <= x.t_kv, "queries must be a suffix of the cache");
}

/// In-place softmax with max-subtraction over `s[..n]`.
fn softmax(s: &mut [f32]) {
    let m = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in s.iter_mut() {
        *v = (*v - m).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in s.iter_mut() {
        *v *= inv;
    }
}

/// Naive form: decompress k_nope/v per head from the latent cache, attend at qk dim
/// d_nope+d_rope, output [t_q][n_head][d_v]. Quadratic decompression cost — prefill-only
/// shape in production; here it is the independent oracle.
pub fn mla_attend_naive(d: &MlaDims, x: &MlaInputs) -> Vec<f32> {
    check_shapes(d, x);
    let (nh, dn, dr, dv, r) = (d.n_head, d.d_nope, d.d_rope, d.d_v, d.kv_rank);
    let scale = d.scale();
    let mut out = vec![0.0f32; x.t_q * nh * dv];

    // Decompress the whole cache per head: k_nope[t][dn], v[t][dv].
    let mut k_nope = vec![0.0f32; x.t_kv * dn];
    let mut v = vec![0.0f32; x.t_kv * dv];
    let mut scores = vec![0.0f32; x.t_kv];
    for h in 0..nh {
        let wuk = &x.w_uk[h * dn * r..(h + 1) * dn * r];
        let wuv = &x.w_uv[h * dv * r..(h + 1) * dv * r];
        for t in 0..x.t_kv {
            let c = &x.c_kv[t * r..(t + 1) * r];
            for p in 0..dn {
                let row = &wuk[p * r..(p + 1) * r];
                let mut acc = 0.0f32;
                for l in 0..r {
                    acc += row[l] * c[l];
                }
                k_nope[t * dn + p] = acc;
            }
            for j in 0..dv {
                let row = &wuv[j * r..(j + 1) * r];
                let mut acc = 0.0f32;
                for l in 0..r {
                    acc += row[l] * c[l];
                }
                v[t * dv + j] = acc;
            }
        }
        for i in 0..x.t_q {
            let visible = x.t_kv - x.t_q + i + 1; // causal horizon for query i
            let qn = &x.q_nope[(i * nh + h) * dn..(i * nh + h + 1) * dn];
            let qp = &x.q_pe[(i * nh + h) * dr..(i * nh + h + 1) * dr];
            for t in 0..visible {
                let mut s = 0.0f32;
                let kn = &k_nope[t * dn..(t + 1) * dn];
                for p in 0..dn {
                    s += qn[p] * kn[p];
                }
                let kp = &x.k_pe[t * dr..(t + 1) * dr];
                for p in 0..dr {
                    s += qp[p] * kp[p];
                }
                scores[t] = s * scale;
            }
            softmax(&mut scores[..visible]);
            let o = &mut out[(i * nh + h) * dv..(i * nh + h + 1) * dv];
            for t in 0..visible {
                let p = scores[t];
                let vt = &v[t * dv..(t + 1) * dv];
                for j in 0..dv {
                    o[j] += p * vt[j];
                }
            }
        }
    }
    out
}

/// Absorbed form (decode form): q̃_h = w_uk[h]ᵀ·q_nope_h (rank-space, kv_rank wide), scores are
/// MQA dots against the raw latent rows [c_kv | k_pe] (kv_rank + d_rope wide), the attention
/// output is accumulated in latent space (kv_rank wide) and decompressed once through w_uv.
/// Identical result to `mla_attend_naive` by associativity + linearity (DESIGN.md §1.3).
pub fn mla_attend_absorbed(d: &MlaDims, x: &MlaInputs) -> Vec<f32> {
    check_shapes(d, x);
    let (nh, dn, dr, dv, r) = (d.n_head, d.d_nope, d.d_rope, d.d_v, d.kv_rank);
    let scale = d.scale();
    let mut out = vec![0.0f32; x.t_q * nh * dv];

    let mut q_lat = vec![0.0f32; r]; // absorbed query, rank space
    let mut o_lat = vec![0.0f32; r]; // attention output, latent space
    let mut scores = vec![0.0f32; x.t_kv];
    for h in 0..nh {
        let wuk = &x.w_uk[h * dn * r..(h + 1) * dn * r];
        let wuv = &x.w_uv[h * dv * r..(h + 1) * dv * r];
        for i in 0..x.t_q {
            let visible = x.t_kv - x.t_q + i + 1;
            let qn = &x.q_nope[(i * nh + h) * dn..(i * nh + h + 1) * dn];
            let qp = &x.q_pe[(i * nh + h) * dr..(i * nh + h + 1) * dr];
            // absorb: q_lat[l] = sum_p q_nope[p] * w_uk[h][p][l]
            q_lat.iter_mut().for_each(|v| *v = 0.0);
            for p in 0..dn {
                let row = &wuk[p * r..(p + 1) * r];
                let qv = qn[p];
                for l in 0..r {
                    q_lat[l] += qv * row[l];
                }
            }
            // MQA scores against the 576-wide latent rows
            for t in 0..visible {
                let c = &x.c_kv[t * r..(t + 1) * r];
                let mut s = 0.0f32;
                for l in 0..r {
                    s += q_lat[l] * c[l];
                }
                let kp = &x.k_pe[t * dr..(t + 1) * dr];
                for p in 0..dr {
                    s += qp[p] * kp[p];
                }
                scores[t] = s * scale;
            }
            softmax(&mut scores[..visible]);
            // latent-space AV
            o_lat.iter_mut().for_each(|v| *v = 0.0);
            for t in 0..visible {
                let p = scores[t];
                let c = &x.c_kv[t * r..(t + 1) * r];
                for l in 0..r {
                    o_lat[l] += p * c[l];
                }
            }
            // decompress once: out[j] = sum_l w_uv[h][j][l] * o_lat[l]
            let o = &mut out[(i * nh + h) * dv..(i * nh + h + 1) * dv];
            for j in 0..dv {
                let row = &wuv[j * r..(j + 1) * r];
                let mut acc = 0.0f32;
                for l in 0..r {
                    acc += row[l] * o_lat[l];
                }
                o[j] = acc;
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// RoPE: GLM-5.2 is `rope_interleave: true` == llama.cpp LLAMA_ROPE_TYPE_NORM.
// memra ships only NEOX pairing; the permutation below maps NORM onto NEOX at
// weight-load time (DESIGN.md §1.4). Both variants + the permutation live here
// so the equivalence is a pinned, tested fact.
// ---------------------------------------------------------------------------

/// Interleaved ("NORM") rope over the first `n_dims` of `x`: pair (x[2j], x[2j+1]) rotated by
/// theta_j = pos * base^(-2j/n_dims). Matches ggml GGML_ROPE_TYPE_NORM / HF interleaved.
pub fn rope_interleaved(x: &mut [f32], n_dims: usize, pos: f32, base: f32) {
    let half = n_dims / 2;
    let theta_scale = base.powf(-2.0 / n_dims as f32);
    let mut theta = pos;
    for j in 0..half {
        let (sin, cos) = theta.sin_cos();
        let a = x[2 * j];
        let b = x[2 * j + 1];
        x[2 * j] = a * cos - b * sin;
        x[2 * j + 1] = a * sin + b * cos;
        theta *= theta_scale;
    }
}

/// NEOX rope over the first `n_dims` of `x`: pair (x[j], x[j+half]) rotated by the same
/// theta_j sequence. Matches memra's `rope_neox_f32` (kernels.cu) angle recurrence.
pub fn rope_neox(x: &mut [f32], n_dims: usize, pos: f32, base: f32) {
    let half = n_dims / 2;
    let theta_scale = base.powf(-2.0 / n_dims as f32);
    let mut theta = pos;
    for j in 0..half {
        let (sin, cos) = theta.sin_cos();
        let a = x[j];
        let b = x[j + half];
        x[j] = a * cos - b * sin;
        x[j + half] = a * sin + b * cos;
        theta *= theta_scale;
    }
}

/// The load-time permutation: source (interleaved-layout) index -> NEOX-layout index.
/// pi(2j) = j, pi(2j+1) = j + n_dims/2. Applied to the rope rows of wq_b / wkv_a_mqa at load,
/// it makes the existing NEOX kernel compute exactly the interleaved rotation (dot-product
/// consumers only — which is all of them).
pub fn norm_to_neox_perm(n_dims: usize) -> Vec<usize> {
    let half = n_dims / 2;
    let mut p = vec![0usize; n_dims];
    for j in 0..half {
        p[2 * j] = j;
        p[2 * j + 1] = j + half;
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    /// xorshift64* — deterministic, no external crates.
    struct Rng(u64);
    impl Rng {
        fn next_f32(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            let v = (self.0.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as u32;
            (v as f32 / (1u32 << 24) as f32) * 2.0 - 1.0 // uniform [-1, 1)
        }
        fn fill(&mut self, n: usize, scale: f32) -> Vec<f32> {
            (0..n).map(|_| self.next_f32() * scale).collect()
        }
    }

    fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
    }
    fn maxabs(a: &[f32]) -> f32 {
        a.iter().map(|x| x.abs()).fold(0.0f32, f32::max)
    }

    /// Build random inputs at unit-ish scale: weights ~ 1/sqrt(rank) so decompressed values and
    /// scores stay O(1) and the f32 tolerance is meaningful.
    fn random_case(d: &MlaDims, t_q: usize, t_kv: usize, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut rng = Rng(seed | 1);
        let ws = 1.0 / (d.kv_rank as f32).sqrt();
        (
            rng.fill(t_q * d.n_head * d.d_nope, 1.0),
            rng.fill(t_q * d.n_head * d.d_rope, 1.0),
            rng.fill(t_kv * d.kv_rank, 1.0),
            rng.fill(t_kv * d.d_rope, 1.0),
            rng.fill(d.n_head * d.d_nope * d.kv_rank, ws),
            rng.fill(d.n_head * d.d_v * d.kv_rank, ws),
        )
    }

    fn run_case(d: &MlaDims, t_q: usize, t_kv: usize, seed: u64, tol: f32) {
        let (q_nope, q_pe, c_kv, k_pe, w_uk, w_uv) = random_case(d, t_q, t_kv, seed);
        let x = MlaInputs {
            q_nope: &q_nope, q_pe: &q_pe, c_kv: &c_kv, k_pe: &k_pe,
            w_uk: &w_uk, w_uv: &w_uv, t_q, t_kv,
        };
        let naive = mla_attend_naive(d, &x);
        let absorbed = mla_attend_absorbed(d, &x);
        let md = maxdiff(&naive, &absorbed);
        let scale = maxabs(&naive).max(1.0);
        assert!(
            md <= tol * scale,
            "naive vs absorbed disagree: maxdiff {md:.3e} (scale {scale:.3e}, rel {:.3e}) \
             dims {d:?} t_q {t_q} t_kv {t_kv} seed {seed}",
            md / scale
        );
        // sanity: outputs are finite and not trivially zero
        assert!(naive.iter().all(|v| v.is_finite()));
        assert!(maxabs(&naive) > 1e-6);
    }

    #[test]
    fn naive_equals_absorbed_decode_t1() {
        // t=1 decode against a populated cache, several synthetic shapes + seeds.
        let shapes = [
            MlaDims { n_head: 4, d_nope: 24, d_rope: 8, d_v: 32, kv_rank: 64 },
            MlaDims { n_head: 2, d_nope: 16, d_rope: 16, d_v: 16, kv_rank: 32 },
            // GLM-5.2 ratio at 1/8 scale: nope 24, rope 8, v 32, rank 64 handled above;
            // an asymmetric case where d_v > d_nope (the GLM-5.2 signature, v 256 > nope 192):
            MlaDims { n_head: 3, d_nope: 12, d_rope: 4, d_v: 20, kv_rank: 48 },
        ];
        for (i, d) in shapes.iter().enumerate() {
            for seed in [7, 1234, 0xB1E55ED] {
                run_case(d, 1, 17, seed + i as u64, 1e-5);
            }
        }
    }

    #[test]
    fn naive_equals_absorbed_prefill_causal() {
        // small prefill: t_q new tokens over t_kv-t_q past tokens, causal horizon per query.
        let d = MlaDims { n_head: 4, d_nope: 24, d_rope: 8, d_v: 32, kv_rank: 64 };
        run_case(&d, 5, 9, 42, 1e-5);
        run_case(&d, 8, 8, 43, 1e-5); // pure prefill, no past
        let d2 = MlaDims { n_head: 2, d_nope: 16, d_rope: 16, d_v: 16, kv_rank: 32 };
        run_case(&d2, 3, 11, 44, 1e-5);
    }

    #[test]
    fn naive_equals_absorbed_glm52_full_dims() {
        // Full GLM-5.2 geometry (64 heads, 192/64/256, rank 512) — decode t=1, T=8.
        // Wider accumulations (576-dot, rank-512 decompress) ⇒ slightly looser f32 tolerance.
        run_case(&MlaDims::GLM52, 1, 8, 20260801, 1e-4);
    }

    #[test]
    fn rope_norm_equals_permuted_neox() {
        // DESIGN.md §1.4: permuting the rope dims at load time (pi(2j)=j, pi(2j+1)=j+half)
        // makes the NEOX kernel compute the interleaved ("NORM") rotation. Verify:
        //   permute(rope_interleaved(x)) == rope_neox(permute(x))
        // for the GLM-5.2 rope width (64) at several positions, and that dot products between
        // two identically-permuted roped vectors match the un-permuted interleaved dots.
        let n_dims = 64;
        let base = 8_000_000.0f32; // GLM-5.2 rope_theta
        let perm = norm_to_neox_perm(n_dims);
        let mut rng = Rng(99);
        for pos in [0.0f32, 1.0, 17.0, 4096.0, 1_000_000.0] {
            let x0: Vec<f32> = (0..n_dims).map(|_| rng.next_f32()).collect();
            let y0: Vec<f32> = (0..n_dims).map(|_| rng.next_f32()).collect();

            // path A: interleaved rope, then permute
            let mut xa = x0.clone();
            rope_interleaved(&mut xa, n_dims, pos, base);
            let mut xa_p = vec![0.0f32; n_dims];
            for (src, &dst) in perm.iter().enumerate() {
                xa_p[dst] = xa[src];
            }
            // path B: permute, then neox rope
            let mut xb = vec![0.0f32; n_dims];
            for (src, &dst) in perm.iter().enumerate() {
                xb[dst] = x0[src];
            }
            rope_neox(&mut xb, n_dims, pos, base);

            assert!(
                maxdiff(&xa_p, &xb) <= 1e-6,
                "perm/rope orders disagree at pos {pos}"
            );

            // dot-product invariance (what attention actually consumes)
            let mut ya = y0.clone();
            rope_interleaved(&mut ya, n_dims, pos, base);
            let dot_norm: f32 = xa.iter().zip(&ya).map(|(a, b)| a * b).sum();

            let mut yb = vec![0.0f32; n_dims];
            for (src, &dst) in perm.iter().enumerate() {
                yb[dst] = y0[src];
            }
            rope_neox(&mut yb, n_dims, pos, base);
            let dot_neox: f32 = xb.iter().zip(&yb).map(|(a, b)| a * b).sum();

            assert!(
                (dot_norm - dot_neox).abs() <= 1e-4 * dot_norm.abs().max(1.0),
                "roped dot products diverge at pos {pos}: {dot_norm} vs {dot_neox}"
            );
        }
    }
}
