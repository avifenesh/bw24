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

(fill: sweep + gates + bar)

## 3. Bar re-checks

(fill: o35b barcheck re-ratio; KAT deploy verdict)

## 4. Guards

(fill: q35 ctrl guard; spec batteries)

## Files

`run-stage1-sweep.sh`, `run-stage2-sweep.sh`, `run-gates.sh`, `run-barcheck-o35b.sh`,
`run-barcheck-kat.sh`; `stage1-sweep.jsonl` (v1 rows git=1576d8b3 = the unpipelined first cut,
v2 rows git=2ac63454 = shipped), `stage2-sweep.jsonl`, `gates.jsonl`, `barcheck-*.jsonl`;
per-run logs `s1-*`, `s2-*`, `q35-guard-*`, `obar-*`, `kbar-*`, `probe-direct-v2-*`;
`kernel-check-r1.log`; `token-hashes.log`; consoles `sweep-console.log`/`gates-console.log`.
