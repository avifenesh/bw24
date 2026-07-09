# Sampled-spec implementation map (2026-07-09, from spec.rs read @1500-1545)

Owner decision: rejection-sampling spec = the prod solution (HANDOVER "SAMPLED-SPEC ARC").
Survey: research/greedy-degeneration-protocol-survey.md (Leviathan/Chen rule, no penalty precedent).

## Code anchors (spec.rs, generate_spec_inner2)
- Draft chain (§1, ~line 1440-1507): per-slot device argmax -> `draft[j]`; pmin path already
  dtoh's the head prob of the chosen token (`p_d`) — the q(x) plumbing shape exists.
- Verify (§3, ~1526-1545): `argmax_token_device_col` per column -> preds[] -> greedy prefix walk
  `t_pred(j)==draft[j]`; `bonus = t_pred(n_acc)`.

## Transform (BW24_SPEC_TEMP > 0 arm; temp==0/unset keeps today's greedy path untouched)
A. **Philox device RNG + Gumbel-max sampler kernel**: sample = argmax(logits/T + Gumbel(Philox(seed, stream_pos)))
   — REUSES the 2-pass device-argmax kernels (add-noise pass first). Returns token + softmax-prob
   gather (q_j). Counter-based: (seed, global position) -> deterministic, graph-replay-safe.
B. **Draft chain arm**: replace argmax draft with A at head logits; RETAIN K head-logit buffers
   (K x vocab f32, ~4MB @9B — fixed slots, graph-compatible) for the reject-slot residual.
C. **Verify accept**: device-gather p_j = softmax_target(col j-1)[draft[j]] for all j (one small
   kernel, K floats out) + K Philox uniforms; host walk: accept while u_j < p_j/q_j. At first
   reject slot j*: residual-sampling kernel over vocab: sample from norm(max(0, p - q)) using
   retained head logits (q) + verify col (p). Full accept: bonus = Gumbel-max sample from last col.
D. **Plumbing**: BW24_SPEC_TEMP + BW24_SEED in run-spec; plain-sampled decode reference in
   run-spec (same Gumbel stream on single-step decode) for the reproducibility A/B.
E. **Gates**: (1) temp=1e-9 K=1..8 == greedy spec token-identical (continuity — Gumbel noise
   scaled by T vanishes); (2) same-seed rerun = identical stream; (3) aggregate: mean acceptance
   within noise of plain-vs-spec sampled agreement on fixed prompts; text audit (no loops at
   temp 0.7 on p3 = the arc's acceptance criterion).
- Softmax normalization detail: p and q must be TRUE softmax (with temp applied) — the pmin
  path's prob is already softmaxed head prob; verify cols need softmax (currently argmax-only,
  no normalization computed) -> the p-gather kernel does row max + sumexp + gather in one pass.
- FR-Spec trim interaction: trim reshapes the HEAD's vocab (draft q lives on trimmed rows);
  residual norm(p-q) must treat non-trim rows as q=0 — CORRECT by construction (head can't
  propose them). Gather maps via d2t.

## Order
1. A kernel + unit check vs host reference (probe-style bin or kernel-check section).
2. B+C eager path, BW24_SPEC_NOGRAPH=1 first. Gates (1)(2).
3. Graph-draft integration (fixed RNG-counter buffers).
4. D reference + (3) aggregate gate + llama matched-temp pairing protocol.
5. Re-baseline battery (owner's full-board order) with p3 under temp 0.7.
