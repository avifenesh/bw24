# kquant-tile-loaders: direct-from-quant tile loaders — the shared dequant-pass kill (2026-08-02)

Lane `lane/kquant-tile-loaders` (from `restructure/public-split`, 1576d8b3; kernel commit
2ac63454). Rig: RTX 5090 Laptop 24463 MiB, platform_profile `performance`, `gpu-full-power on`.
Every GPU run under `flock /tmp/gpu5090.lock` (co-resident `llama-server --embedding` 332 MiB
allowlisted, inside every peak figure). llama.cpp arm: local fork build `bb090d1f1` (same binary
as the q4k-expert-prefill/kat-anomaly lanes). Models:
`/data/ai-ml/hf-models/ornith-1.0-35b-gguf/ornith-1.0-35b-Q4_K_M.gguf` (RESIDENT every run),
`/data/ai-ml/hf-models/kat-coder-v25-dev-gguf/Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf`,
ctrl `/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`. All medians N=3
process-interleaved unless stated; pp2048 process values are medians of 5 in-process reps
(+1 warmup), the sk-bm128 protocol.

## 1. Stage 1 — Q4_K/Q6_K expert tile loaders in the sk visitor GEMM (Ornith-35B)

The q4k-expert-prefill §5 finding: with AUTO-KQUANT, the q4_K/q6_K → f16 dequant passes are
41.8% of t=512 kernel time — a fixed per-(layer,projection) cost (~44 GB f16 write+read per
pass at the 858 GB/s wall) that amortizes at 2048+ but dominates pp512. The kill is NO dequant
pass: `moe_kq_sk{32,128}v_kernel` (cu/moe_f16_grouped.cu) are the round-51 visitor forms with
the B-side cp.async tile loads replaced by dequant-in-register directly from the Q4_K/Q6_K
superblocks in the expert slab. Raw quant bytes for kb+1 prefetch into registers behind kb's
mma (the global latency hides behind tensor-core work); per-16-value-window scale loads and the
left-assoc first products (`dd*sc8`, `dmin*m8`, `dd*sc`) hoist with the value math's exact DAG.
The A-side (activation) pipeline is untouched; B is single-buffered (the trailing
`__syncthreads` of each kb fences the overwrite).

**Numeric class: NONE — bit-identical to the workspace path by construction.** The B smem tile
holds the same f16 values (kq_q4k_val/kq_q6k_val, the workspace dequant kernels' exact
expressions) in the same positions, so every output element's mma k-chain is unchanged.

- First cut (per-value synchronous dequant in the k-loop) was byte-identical but **0.53x** —
  the dequant ALU + uncovered global reads sat on the critical path (v1 rows in
  `stage1-sweep.jsonl`, git=1576d8b3, superseded same-session). The register-pipelined form
  (git=2ac63454) is the shipped kernel. Both forms' receipts kept per evidence discipline.

### Gates (all green)

- kernel-check **ALL GREEN** 0 FAIL (`kernel-check-r1.log`, 283-section battery): new
  `f16g-kq-direct` section gates direct-vs-workspace **maxdiff == 0 (byte-identical)** on
  synthetic q4_K/q6_K (skew CSR 1..300, reversed ex_ids) and REAL Ornith weights
  (blk.0.ffn_gate_exps Q4_K in=2048 out=512; blk.0.ffn_down_exps Q6_K in=512 out=2048), every
  visitor form (hybrid/all-128/all-32).
- **Cross-binary bit-identity anchor:** o35b naked gen512 token sha `c0c12c3b350dc7f5` — the
  q4k-expert-prefill lane's anchor — reproduced in EVERY run of BOTH arms (12/12), proving (a)
  the kq_*_val refactor did not move the workspace path's bits, (b) the direct path is
  end-to-end bit-identical. argmax MATCH 2/2 every run.
- o35b run-spec K=1..8 with the adopted own-trim drafter: see gates section below.

### Perf — interleaved x3, same session (`stage1-sweep.jsonl` git=2ac63454, ranges disjoint)

| arm | pp512 (gen512 prefill, med N=3) | pp2048 (med N=3) | decode tg128 | peak VRAM MiB (gen512) |
|---|---|---|---|---|
| ws (`MEMRA_F16G_DIRECT=0`, pre-lane default) | 1667.2 [1664.1–1674.2] | 3464.0 [3463.5–3464.0] | 208.98 | 22386 |
| direct (naked) | **3155.6** [3154.8–3157.0] | **4779.9** [4777.5–4795.2] | 209.79 | 21426 |

**pp512 +89.3%, pp2048 +38.0%, decode flat, peak VRAM −960 MiB (the f16 workspace no longer
exists).** Against the q4k-expert-prefill same-session llama numbers (pp512 3972.3 / pp2048
3803.7): pp512 0.415x → ~0.79x, pp2048 0.907x → ~1.26x (WIN). Same-session llama re-ratio in §3.

## 2. Stage 2 — IQ4_XS dense-trunk MMQ (KAT-Coder)

kat-anomaly §6 priced it: KAT's remaining bar-binding gap was prefill 0.169x — every trunk
IQ4_XS matmul (attn_qkv x30, attn_gate x30, ssm_out x16, attn_q x5, shexp x60, ~0.52 GB) rode
the per-column dp4a grid with zero weight reuse across tokens. `mmq_iq4xs_dense_kernel`
(cu/mmq_iq_experts.cu) is the dense analog of the expert MMQ: the same `load_tiles_iq4xs`
decode-at-tile-load + `vec_dot_mma` int8 MMA + W kb-slice cp.async staging ring, on
conventional xy-tiling with the token-major D4 q8_1 activation. Admission: `mmq_supports`
IQ4_XS arm at m>=16, out_f>=128, in_f%256==0 — decode and spec-verify (m=1..15) keep dp4a
(the kat-anomaly dispatch-parity law). `MEMRA_PP_IQMMQ=0` reverts the arm; `MEMRA_IQ_FAST=0`
kills the whole path. MMA-reduction numeric class — argmax/spec gated, not byte-identity.

### Gates (all green)

- kernel-check `iq4xs-mmq`: rel <= 2.7e-4 vs the dp4a program at T=16/64/128/512, synthetic
  blocks + a real KAT IQ4_XS tensor (battery ALL GREEN, 0 FAIL).
- KAT gen512 argmax MATCH 2/2 every run; run-spec K=1..8 self-consistency **PASS 8/8** with
  the owntrim drafter (`kat-spec-k1-8.log`).
- dp4a rollback arm (`MEMRA_PP_IQMMQ=0`) token sha == the kat-anomaly naked anchor
  `9102ffd0b8241a65` 3/3 — the seam restores the old stream exactly. mmq arm sha rep-stable
  `e5d59ecedc57aa7d` 3/3 (the expected MMA-class shift, arbitrated by argmax+spec).
- Ctrl exposure: q35 carries ZERO non-expert IQ4_XS 2-D tensors (kat-anomaly tensor mix) —
  dispatch-unchanged by construction, and the §4 q35 guard sha held on the same binary.

### Perf — interleaved x3, same session (`stage2-sweep.jsonl`, ranges disjoint)

| arm | pp512 (gen512 prefill, med N=3) | pp2048 (med N=3) | decode tg128 | peak MiB |
|---|---|---|---|---|
| dp4a (`MEMRA_PP_IQMMQ=0`) | 695.9 [695.4–700.2] | 764.0 [764.0–767.2] | 194.94 | 19150 |
| mmq (naked) | **2057.5** [2056.5–2059.2] | **3028.6** [3025.3–3032.2] | 194.11 | 19150 |

**pp512 2.96x, pp2048 3.96x, decode flat.** The ~2315 ctrl-class ceiling: 2057.5 = 89% of it —
the residual is the ctrl's Q8_0-trunk MMQ vs this IQ4_XS tile decode plus KAT's ssm layers.

## 3. Bar re-checks — same-session llama, interleaved x3

### Ornith-35B (`barcheck-o35b.jsonl`, `obar-*`; plain rows reparsed from raw logs —
### the jsonl `plain_decode_toks` regex missed run-spec's column padding, logs are canonical)

Board plain-vs-plain: memra pp512 3148.6 / pp2048 4771.7 / tg128 209.8 vs llama-bench
(`-ngl 999 -fa 1 -ctk q8_0 -ctv q5_1`) pp512 3977.4 / pp2048 3823.3 / tg128 192.6:

- **prefill ratio: pp512 0.792x (was 0.415x), pp2048 1.248x (was 0.907x — flips to a WIN).**
- **plain e2e (512+128): 0.773 s vs 0.793 s = 1.027x — the 0.860x plain CROSSES 1.0.**
- decode 1.089x (unchanged — this lane didn't touch decode).

Best-vs-best per class (board convention: memra = adopted drafter spec K=2, self-consistency
PASS every run; llama = plain; e2e = prime wall + 256/decode-rate, llama rates from the same
interleaved `-p 27,1845,6257 -n 256` call):

| class | memra e2e | llama e2e | ratio | was (q4k-expert-prefill) |
|---|---|---|---|---|
| p1-code-short (27) | **0.984 s** (0.048 + 256@273.6) | 1.356 s | **1.379x** | 1.314x |
| p2-code-medium (1845) | **1.440 s** (0.399 + 256@245.9) | 1.834 s | **1.274x** | 1.136x |
| p3-agentic-long (6257) | **2.412 s** (1.295 + 256@229.1) | 3.046 s | **1.262x** | 1.115x |

Acceptance 68.1/62.7/59.9%, rep-identical — the drafter is untouched. **Ornith-35B holds
DEPLOY with margin on every class, and its plain e2e now beats llama outright.**

### KAT-Coder (`barcheck-kat.jsonl`, `kbar-*`) — VERDICT: HOLD (pre-deployment), gap halved

Board plain-vs-plain: memra pp512 2060.2 / pp2048 3032.6 / tg128 194.5 vs llama pp512 4254.5 /
pp2048 4127.9 / tg128 194.7 (same-session interleaved x3):

| leg | memra | llama | ratio | was (kat-anomaly) |
|---|---|---|---|---|
| decode plain | 194.5 | 194.7 | **0.998x** (parity) | 1.016x |
| prefill pp512 | 2060.2 | 4254.5 | **0.484x** | 0.169x |
| prefill pp2048 | 3032.6 | 4127.9 | **0.735x** | — |
| plain e2e 512+128 | 0.907 s | 0.778 s | **0.858x** | 0.57x |

Best-vs-best per class (memra = min(plain, spec K=2), spec self-consistency PASS every run;
llama plain, same interleaved `-p 27,1845,6257 -n 256` call):

| class | memra best | llama e2e | ratio | 1.1x bar |
|---|---|---|---|---|
| p1-code-short (27) | **1.162 s** (spec K=2, acc 82.5%) | 1.342 s | **1.156x** | **PASS** |
| p2-code-medium (1845) | 1.872 s (spec K=2, acc 68.5%) | 1.782 s | 0.952x | FAIL |
| p3-agentic-long (6257) | 3.368 s (plain) | 2.908 s | 0.863x | FAIL |

**KAT stays onboarded, pre-deployment.** The IQ4_XS-trunk MMQ delivered its priced share
(trunk prefill 3-4x, e2e 0.57x -> 0.858x, code-short class flips to a 1.156x PASS with the
drafter — note the faster prefill also lifted spec: p2 spec is now net-positive 1.06x vs its
0.96x at kat-anomaly). The remaining bar-binding gap is the MoE **expert** prefill class
shared with q35 (its IQ3_S/IQ4_XS int8-MMA expert tiles vs llama's — the unowned q35-vs-llama
prefill gap, kat-anomaly §6's "rest of the ctrl's gap"), not the trunk this stage owned.

## 4. Guards

- **q35 ctrl guard** (naked, x3, post-both-stages binary): token sha `86dc5f7105a3716b` ==
  the q4k-expert-prefill anchor 3/3 — generated stream bit-identical across the lane.
  pp2048 4090.1/4098.1/4100.5 (prev-lane post cell 4070.6 med), gen512 prefill 2493.9–2501.2
  (prev 2450), decode 244.7–252.9: flat-or-better on every column (cross-session comparison,
  sha-anchored; the run-to-run rates are same-session self-consistent).
- **o35b** run-spec K=1..8 self-consistency PASS 8/8 (`o35b-spec-k1-8.log`); gen512 argmax
  MATCH + sha anchor in every run of every arm (see §1).
- **kat** spec + anchors in §2.

## 5. What remains (priced, not built here)

- **KAT p2/p3 e2e (0.952x/0.863x):** bar-binding gap is now the MoE expert prefill (q35-class
  IQ3_S/IQ4_XS expert MMQ vs llama) plus llama's stronger long-ctx attention prefill — its own
  lane; the trunk is no longer the wall. KAT pp512 2058 sits at 89% of the ~2315 ctrl-class
  ceiling this stage was priced against.
- **Ornith pp512 residual (0.792x):** the nsys stragglers from q4k-expert-prefill §5
  (`qmatvec_gemm_q6_K` 5.9% + `mul_mat_q_q45k` 5.5% trunk shares) are now the visible next
  slice, plus the sk GEMM itself (the sk128 tail-form rung from sk-bm128). pp2048 already
  beats llama 1.25x.
- **Merge-time board note:** this lane moves published numbers (Ornith pp512/pp2048/e2e rows,
  KAT rows if listed) — `current-board.json` + README/perf-card regeneration are owed in the
  merge/tag commit per the perf-board rule (not done on this lane branch; NO pushes from here).
- Hopper: the direct loaders are sm_80-portable by construction (same cp.async/ldmatrix/
  mma.sync class as the visitor forms) but are admitted only via the sm_120a AUTO-KQUANT
  mode-3 path today; the Hopper mode-1 (cublas) default is untouched. Re-sweep there before
  any flip (stale-verdict law).

## Files

`run-stage1-sweep.sh`, `run-stage2-sweep.sh`, `run-gates.sh`, `run-barcheck-o35b.sh`,
`run-barcheck-kat.sh`; `stage1-sweep.jsonl` (v1 rows git=1576d8b3 = the unpipelined first cut,
v2 rows git=2ac63454 = shipped), `stage2-sweep.jsonl`, `gates.jsonl`, `barcheck-*.jsonl`;
per-run logs `s1-*`, `s2-*`, `q35-guard-*`, `obar-*`, `kbar-*`, `probe-direct-v2-*`;
`kernel-check-r1.log`; `token-hashes.log`; consoles `sweep-console.log`/`gates-console.log`.
