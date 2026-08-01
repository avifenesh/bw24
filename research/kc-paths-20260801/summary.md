# kernel-check battery blindness kill — model-path resolution + loud skips (lane/kc-paths)

Lane 4, darklanes-8x GPU 4 (H100). Baseline e040e149. The dtype5/D.2 weight-oracle sections
pinned 5090-rig absolute paths, so H100 rounds 44-47 ran the battery blind on exactly the
models this lane fights over (IQ4_XS/IQ3_S = q35). This lane converts every hardcoded model
path in `crates/memra-engine/src/bin/kernel_check.rs` to a resolution chain and makes every
miss loud and actionable.

## The mechanism

`kc_model(section, fname, legacy, gguf_arg)` — first existing path wins:
1. `$MEMRA_KC_MODELS_DIR/<file>` (explicit env)
2. the CLI gguf arg when its basename matches (model under test doubles as oracle)
3. `$HOME/models/<file>`, `/opt/dlami/nvme/models/<file>` (bench-box conventions)
4. the legacy rig paths (`/home/avifenesh/...`, `/data/...`) — the 5090 rig keeps working naked
On a miss: ONE `KC-SKIP [section] <file>: absent on this box (N candidates tried) — set
MEMRA_KC_MODELS_DIR=...` line. A skip is never silent.

Converted call sites (10): dtype5 (9B + 35B), nvfp4-gemm (9B + 27B fallback), q8mmq-gemm (35B),
q4_0-mmq (g12), nvfp4-27b-shape (27B), q4_0-sk-arm (g26, previously ad-hoc HOME fallback),
nvfp4-mmvq (9B), nvfp4-batched (9B), a6-split-plane 9b-fallback, d2-cache-bit-identity (35B).

## Two deeper blindness layers found by the new coverage

1. **Same filename, different artifact revision.** The rig's `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`
   (Jun 21, 18G) carries a blk.40 MTP/NextN layer whose experts are the file's only Q3_K
   tensors; the box copy (Jul 28, 17.7G) tops out at blk.39 and has NO Q3_K tensor anywhere.
   The dtype5 Q3_K pin (`blk.40.ffn_gate_exps.weight`) therefore hard-FAILED the battery on
   first unlock (kernel-check-after.log). Fix: when the pinned tensor is absent/re-typed,
   substitute the smallest same-dtype `.weight` tensor (the case gates the DTYPE, the name is
   just a carrier); a file with no such tensor skips loudly. Numeric thresholds unchanged.
2. **Double-gating.** The vendored Q8_0-MMQ (35B), q4_0-MMQ + rp split-plane bit-identity
   (g12), and 27B-shape sections were NESTED inside `if let Some(gguf_9b)` — the 9B NVFP4
   artifact's if-let — so on any box without that one file they produced no output at all
   (not even a skip line), despite needing only their own models. Hoisted to section level.
   (First hoist attempt re-paired the braces back inside the if-let because the trio's span
   carried the enclosing block's close — caught because the box run stayed silent, fixed, and
   verified by brace-depth trace + on-box entries; the arc is in kernel-check-after2/3.log.)

## Before -> after on the box (GPU 4)

Before (rounds 44-47 state, receipt: research/jstrip-exit-20260801/kernel-check-exit.log):
- dtype5 x5: "GGUF absent (<5090 path>) — SKIP" (rig-path-pinned, no remedy named)
- D.2: "35B GGUF absent — SKIP"
- Q8MMQ-35B / q4_0-MMQ-g12+RP / 27B-shape: INVISIBLE (dead code, zero lines)
- nvfp4-gemm/mmvq/batched/a6: silent None resolution, zero lines
- 163 OK entries.

After (kernel-check-after4.log): **ALL GREEN, 191 OK entries (+28)**, every skip loud:
- NOW RUN: dtype5 IQ3_S Stage-A rel=1.19e-7; dtype5 IQ4_XS Stage-A rel=2.24e-8 + Stage-B dp4a
  rel=1.09e-4; D.2 moe cache-HIT bit-identity OK (the q35 expert bytes through stage-vs-cache);
  MMQ-Q8_0 x8 OK (35B attn_qkv 2048x8192, ffn_gate_shexp); MMQ-Q4_0 x8 + MMQ-Q4_0-RP
  bit-mismatch 0 x8 (g12 attn_q 3840x4096, ffn_gate 3840x15360).
- LOUD SKIPS (artifact genuinely absent on this box — the deliverable for these): 9B
  NVFP4-MTP (dtype5, nvfp4-gemm, nvfp4-mmvq, nvfp4-batched, a6-split-plane) and 27B
  NVFP4-Q4_K_M-mtp (nvfp4-gemm fallback, nvfp4-27b-shape) — each names MEMRA_KC_MODELS_DIR.
- REVISION SKIP: dtype5 Q3_K — box 35B revision lacks the dtype entirely (loud, names the
  pinned tensor and the file).

5090-rig behavior: legacy paths are still in every chain (last), the hoisted trio now runs
regardless of the 9B artifact (same-or-more coverage), and the dtype5 substitution only
activates when a pin is absent/re-typed. No thresholds moved.

Files: kernel-check-after.log (first unlock, exposes the Q3_K revision FAIL),
after2 (substitution fix, ALL GREEN, trio still dead), after3 (hoist attempt 1, identical to
after2 = the dead-chunk detection), after4 (final, ALL GREEN + new entries), build.log.
Box copies: ~/lane4/research/kc-paths-20260801/.
