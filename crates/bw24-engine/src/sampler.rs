//! Host-side sampler chain (BASE-2, BW24-BUILD-MAP §BASE-2). Ports llama.cpp CPU sampler
//! semantics (llama-sampler.cpp): repetition/freq/presence penalties -> temperature -> top-k ->
//! top-p -> min-p -> categorical draw. Greedy (temp<=0) = argmax, the bit-exact reference.
//!
//! Runs on the host over the full [n_vocab] f32 logit vector already brought back by the per-step
//! D2H sync (decode.rs) — at B=2-4 this is single-µs, no GPU kernel needed (the GPU-fused sampler
//! is a deferred PERF item, only needed once CUDA-graph removes the D2H barrier).

/// Sampler configuration. Defaults = greedy (temp 0). Order of application matches llama.cpp.
#[derive(Clone, Debug)]
pub struct SamplerConfig {
    pub temperature: f32,   // <= 0.0 => greedy argmax (penalties/top-k/p ignored)
    pub top_k: usize,       // 0 => disabled (keep all)
    pub top_p: f32,         // 1.0 => disabled
    pub min_p: f32,         // 0.0 => disabled
    pub penalty_last_n: usize, // window of recent tokens for penalties (0 => disabled)
    pub penalty_repeat: f32,   // 1.0 => disabled (llama default 1.0)
    pub penalty_freq: f32,     // 0.0 => disabled
    pub penalty_present: f32,  // 0.0 => disabled
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        SamplerConfig {
            temperature: 0.0, top_k: 0, top_p: 1.0, min_p: 0.0,
            penalty_last_n: 0, penalty_repeat: 1.0, penalty_freq: 0.0, penalty_present: 0.0,
            seed: 0,
        }
    }
}

/// Stateful sampler: owns the RNG + the recent-token history (for penalties).
pub struct Sampler {
    cfg: SamplerConfig,
    rng: SplitMix64,
    history: Vec<u32>,   // recently emitted tokens (for penalty window)
}

impl Sampler {
    pub fn new(cfg: SamplerConfig) -> Self {
        let rng = SplitMix64::new(cfg.seed);
        Sampler { cfg, rng, history: Vec::new() }
    }

    pub fn is_greedy(&self) -> bool { self.cfg.temperature <= 0.0 }

    /// Record an emitted token so subsequent penalties see it.
    pub fn accept(&mut self, token: u32) { self.history.push(token); }

    /// Sample the next token id from raw logits [n_vocab]. Does NOT mutate logits in place beyond
    /// a local copy. Returns the chosen token id. (Caller should `accept()` it afterwards.)
    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        // Greedy fast path: argmax over RAW logits (penalties don't change the argmax direction
        // enough to matter for the reference path; llama greedy is also pre-penalty argmax only
        // when no penalties set — but to stay correct under penalties we still apply them first).
        if self.is_greedy()
            && self.cfg.penalty_repeat == 1.0
            && self.cfg.penalty_freq == 0.0
            && self.cfg.penalty_present == 0.0
        {
            return argmax_u32(logits);
        }

        // Work on (id, logit) candidates.
        let mut cand: Vec<(u32, f32)> =
            logits.iter().enumerate().map(|(i, &l)| (i as u32, l)).collect();

        // 1. Penalties (operate on logits, over the last-n history window).
        self.apply_penalties(&mut cand);

        // Greedy-with-penalties: argmax after penalties, no sampling.
        if self.is_greedy() {
            let mut best = cand[0];
            for &c in &cand[1..] { if c.1 > best.1 { best = c; } }
            return best.0;
        }

        // 2. Temperature scale.
        if self.cfg.temperature > 0.0 && self.cfg.temperature != 1.0 {
            let inv = 1.0 / self.cfg.temperature;
            for c in cand.iter_mut() { c.1 *= inv; }
        }

        // 3. top-k: keep the k highest-logit candidates (partial sort by logit desc).
        if self.cfg.top_k > 0 && self.cfg.top_k < cand.len() {
            cand.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            cand.truncate(self.cfg.top_k);
        }

        // softmax over the surviving candidates (numerically stable).
        softmax_inplace(&mut cand);

        // 4. top-p (nucleus): smallest set whose cumulative prob >= top_p. Needs desc-by-prob order.
        if self.cfg.top_p < 1.0 {
            cand.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            let mut cum = 0.0f32;
            let mut keep = 0usize;
            for (i, c) in cand.iter().enumerate() {
                cum += c.1;
                keep = i + 1;
                if cum >= self.cfg.top_p { break; }
            }
            cand.truncate(keep.max(1));
        }

        // 5. min-p: keep candidates with prob >= min_p * max_prob.
        if self.cfg.min_p > 0.0 {
            let maxp = cand.iter().map(|c| c.1).fold(0.0f32, f32::max);
            let thresh = self.cfg.min_p * maxp;
            cand.retain(|c| c.1 >= thresh);
            if cand.is_empty() { return argmax_u32(logits); } // safety
        }

        // renormalize the surviving probs and draw.
        let sum: f32 = cand.iter().map(|c| c.1).sum();
        let r = self.rng.next_f32() * sum;
        let mut acc = 0.0f32;
        for c in &cand { acc += c.1; if acc >= r { return c.0; } }
        cand.last().unwrap().0
    }

    /// llama.cpp penalty: for each token in the last-n history, repeat-divide/multiply its logit
    /// and apply frequency*count + presence. (llama-sampler.cpp penalties.)
    fn apply_penalties(&self, cand: &mut [(u32, f32)]) {
        let n = self.cfg.penalty_last_n;
        if n == 0 { return; }
        if self.cfg.penalty_repeat == 1.0 && self.cfg.penalty_freq == 0.0 && self.cfg.penalty_present == 0.0 {
            return;
        }
        let start = self.history.len().saturating_sub(n);
        let window = &self.history[start..];
        if window.is_empty() { return; }
        // count occurrences in the window
        use std::collections::HashMap;
        let mut counts: HashMap<u32, i32> = HashMap::new();
        for &t in window { *counts.entry(t).or_insert(0) += 1; }
        for c in cand.iter_mut() {
            if let Some(&cnt) = counts.get(&c.0) {
                // repeat: llama divides if logit>0 else multiplies (penalize toward 0)
                if self.cfg.penalty_repeat != 1.0 {
                    if c.1 > 0.0 { c.1 /= self.cfg.penalty_repeat; }
                    else { c.1 *= self.cfg.penalty_repeat; }
                }
                c.1 -= cnt as f32 * self.cfg.penalty_freq;
                c.1 -= self.cfg.penalty_present; // presence: applied once if count>0
            }
        }
    }
}

fn argmax_u32(logits: &[f32]) -> u32 {
    let mut best = 0u32; let mut bv = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() { if v > bv { bv = v; best = i as u32; } }
    best
}

/// Stable softmax over candidate logits, writing probs back into the logit slot.
fn softmax_inplace(cand: &mut [(u32, f32)]) {
    let maxl = cand.iter().map(|c| c.1).fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for c in cand.iter_mut() { let e = (c.1 - maxl).exp(); c.1 = e; sum += e; }
    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    for c in cand.iter_mut() { c.1 *= inv; }
}

/// SplitMix64 — deterministic seedable RNG (so a fixed seed reproduces the token stream for the
/// validation gate). Not crypto; fine for sampling.
struct SplitMix64 { state: u64 }
impl SplitMix64 {
    fn new(seed: u64) -> Self { SplitMix64 { state: seed.wrapping_add(0x9E3779B97F4A7C15) } }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    /// uniform f32 in [0,1).
    fn next_f32(&mut self) -> f32 {
        // top 24 bits -> [0,1)
        ((self.next_u64() >> 40) as f32) / (1u32 << 24) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_is_argmax() {
        let mut s = Sampler::new(SamplerConfig::default()); // temp 0
        let logits = vec![0.1, 5.0, 2.0, -1.0];
        assert_eq!(s.sample(&logits), 1);
    }

    #[test]
    fn temp_sampling_deterministic_with_seed() {
        let cfg = SamplerConfig { temperature: 1.0, seed: 42, ..Default::default() };
        let logits = vec![1.0, 2.0, 3.0, 0.5];
        let a = Sampler::new(cfg.clone()).sample(&logits);
        let b = Sampler::new(cfg).sample(&logits);
        assert_eq!(a, b, "same seed must reproduce the draw");
        assert!(a < 4);
    }

    #[test]
    fn top_k_one_is_argmax() {
        let cfg = SamplerConfig { temperature: 1.0, top_k: 1, seed: 7, ..Default::default() };
        let logits = vec![0.1, 5.0, 2.0, -1.0];
        assert_eq!(Sampler::new(cfg).sample(&logits), 1, "top_k=1 collapses to argmax");
    }

    #[test]
    fn min_p_keeps_only_high_prob() {
        // logit 10 dominates; min_p 0.5 should drop the rest -> always pick id 2.
        let cfg = SamplerConfig { temperature: 1.0, min_p: 0.5, seed: 3, ..Default::default() };
        let logits = vec![0.0, 0.0, 10.0, 0.0];
        for _ in 0..16 {
            assert_eq!(Sampler::new(cfg.clone()).sample(&logits), 2);
        }
    }

    #[test]
    fn repeat_penalty_suppresses_recent() {
        // greedy + heavy repeat penalty: id 1 is argmax but recently emitted -> should drop it.
        let mut cfg = SamplerConfig::default();
        cfg.penalty_last_n = 8; cfg.penalty_repeat = 100.0;
        let mut s = Sampler::new(cfg);
        s.accept(1); // 1 was just emitted
        let logits = vec![4.0, 5.0, 4.5, 1.0]; // raw argmax = 1
        let got = s.sample(&logits);
        assert_ne!(got, 1, "recent token must be penalized out of greedy argmax");
        assert_eq!(got, 2, "next-highest after penalizing 1");
    }
}
