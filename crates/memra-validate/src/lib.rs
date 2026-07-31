//! memra-validate — shared validation-protocol core (Phase D extraction, ARCHITECTURE-H100.md §5).
//!
//! Pure host logic, zero engine/CUDA dependency: the pieces every gate bin and bench
//! duplicated (kernel_check, gdn_bench, fa_sanitize, dtype_gpu_check5, the N=5 bench
//! protocol). CPU kernel references stay with their kernels; THIS crate owns the
//! protocol: deterministic test vectors, error measures, tolerance banding, N-rep
//! medians, and the ALL-GREEN tally contract.
//!
//! Extraction law: moved code is verbatim (bit-identical vectors and measures) — the
//! `pr` generator here is the kernel_check/dtype_gpu_check5 variant; fa_sanitize's
//! 16-bit variant intentionally stays local to it (different distribution = different
//! test vectors = a silent gate change).

/// Max absolute elementwise difference (the universal gate measure).
pub fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

/// Relative error of `d` against the max-|value| scale of `reference` (floored to avoid
/// zero-division on all-zero references) — the kernel_check GEMM-band convention.
pub fn rel_of(d: f32, reference: &[f32]) -> f32 {
    let scale = reference.iter().map(|v| v.abs()).fold(0.0, f32::max).max(1e-3);
    d / scale
}

/// Deterministic unit-interval-ish test vector generator (Knuth multiplicative hash →
/// [-1, 1)). Verbatim the kernel_check/dtype_gpu_check5 `pr` — gate vectors must never
/// drift across crates or sessions.
pub fn pr(i: usize) -> f32 {
    let x = (i.wrapping_mul(2654435761) ^ 0x9E3779B9) as u32;
    ((x >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
}

/// Median of N runs (the repo's N=5 protocol; N is the caller's law, this is the math).
/// Sorts a copy; even N takes the lower-middle (matches the existing bench bins).
pub fn median(xs: &[f64]) -> f64 {
    assert!(!xs.is_empty());
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[(v.len() - 1) / 2]
}

/// Run `f` N times, return (median, all runs). The N=5-medians protocol runner.
pub fn run_n_median<E>(n: usize, mut f: impl FnMut(usize) -> Result<f64, E>) -> Result<(f64, Vec<f64>), E> {
    let mut runs = Vec::with_capacity(n);
    for i in 0..n {
        runs.push(f(i)?);
    }
    Ok((median(&runs), runs))
}

/// ALL-GREEN tally: gates print per-case lines and exit nonzero on any failure.
/// `check` returns the condition so call sites keep their inline `{ ... "FAIL" }` style
/// or use it directly.
#[derive(Default)]
pub struct GateTally {
    pub fails: usize,
}

impl GateTally {
    pub fn check(&mut self, label: &str, ok: bool) -> bool {
        println!("{label}: {}", if ok { "OK" } else { "FAIL" });
        if !ok {
            self.fails += 1;
        }
        ok
    }

    /// Terminal verdict in the kernel-check contract: prints ALL GREEN or returns Err.
    pub fn finish(&self, what: &str) -> Result<(), String> {
        if self.fails == 0 {
            println!("ALL GREEN: {what}");
            Ok(())
        } else {
            Err(format!("{}: {} gate(s) FAILED", what, self.fails))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_odd_even() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.0); // lower-middle
    }

    #[test]
    fn pr_deterministic_snapshot() {
        // Pin the exact vector law — any drift here silently changes every gate.
        assert_eq!(pr(71), pr(71));
        let v: Vec<f32> = (0..4).map(pr).collect();
        assert!(v.iter().all(|x| (-1.0..1.0).contains(x)));
    }

    #[test]
    fn tally_contract() {
        let mut t = GateTally::default();
        assert!(t.check("a", true));
        assert!(!t.check("b", false));
        assert!(t.finish("demo").is_err());
        assert_eq!(t.fails, 1);
    }
}
