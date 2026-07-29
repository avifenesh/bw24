# Next targets — hot paths + adoptable external developments (2026-07-30)

Written at the unified-engine merge (`lane/unified-engine`: sm_120a + sm_90a, one tree,
arch auto-detected). Ranks what to attack next, from (a) the repo's own measured evidence
and (b) an online sweep of developments through July 2026. Every external claim below is
vendor/paper-reported, not independently reproduced; adoption still goes through the
standard battery.

## The measured hot paths (state at merge)

### H100 lane (from ARCHITECTURE-H100.md, rounds 26–36)
| slice | state | headroom |
|---|---|---|
| decode single-seq | 220.5 tok/s = 122.8% vLLM (graph door) | ~1.4 ms/step in-graph non-matvec GPU work; norm-chain fusion, combine batching mapped |
| prefill single-seq | 26.3k = 73–79% of vLLM 35–36k | **OWNER-GATED**: residual is the int8-GEMM dtype edge; exact routes triple-refuted (C7517) — w8a8 accuracy relaxation is the only crossing |
| batched prime | 20.9k (B=6 serving shape, +55% campaign) | cross-request prefill CONCATENATION scheduler (task #13 extension) — vLLM's remaining serving edge |
| serving B-curve | B=8 537.8 aggregate (3.30x) — unified-tree battery 2026-07-30 | mmvq_b8 kernel ~30-35% of peak BW (ncu mapped, not attacked) |
| GDN chunk stack | K2 wgmma'd, K4+K5 fused, K3 double-refuted-optimal | closed on measurement at this design generation |
| FA prefill | FA3 v11 wgmma/TMA, 205 µs harness (4.8x) | V^T map, 3-deep ring — diminishing; not the bottleneck anymore |

### 5090 lane (from the tuning campaign / perf board)
| slice | state | headroom |
|---|---|---|
| plain decode | 1.06–1.08x llama.cpp, DRAM-wall cells at 1.00–1.06x | at the m=1 bandwidth floor on most models |
| MTP spec | 1.06–2.30x, adaptive-K floor + in-round confidence cut shipped | tree-draft lane designed (root-anchored paths, v1 path-duplicated verify) — next spec mechanism |
| prefill | 0.59–0.78x llama (their W4A4 numeric class, refused) | standing refusal; exactness outranks the column |
| Hy3 spill | 5.13 tok/s served median; 90.7% Q2_K post-demotion | I/O-wall: artifact demotion axis exhausted; target 10 tok/s needs a mechanism, not a tier tweak |

## Ranked next targets

1. **CUDA 13.3U1 toolchain probe (days; both arches).** 13.3 fixed a ptxas
   wgmma wait-group copy-propagation race, and CUTLASS's changelog confirms a 13.1
   Hopper-attention codegen regression cluster — the repo's C7514/15/17/19 findings sit
   in that neighborhood. Re-run the wgmma serialization repros on 13.3U1; if the
   C7515-class serialization lifts, the refuted FA3 producer/consumer and s8-rescale
   shapes deserve one re-measure each. Also ships **CompileIQ/ACF**
   (`ptxas --apply-controls`): evolutionary search over scheduling/regalloc, Meta
   reports up to 15% on "done" kernels. Objective function = existing bench harnesses;
   winners re-pass the full battery; ACFs commit per-kernel. Zero algorithm risk.
2. **ReplaySSM-style ring spec-verify for GDN state (5090 spec lane).** SGLang v0.5.16
   dropped per-draft SSM snapshots for checkpoint-ring + replay: 6.4x smaller spec
   scratch on Qwen3.5-35B-A3B TP1. bw24's MTP verify on GDN hybrids pays the same
   snapshot cost; replay recomputes bit-identical state under our determinism laws.
   VRAM reclaimed → KV/context headroom on the 24 GB card.
3. **FA4's numerics-free scheduling tricks (both arches).** LPT worktile scheduling +
   causal-mask grid swizzle for varlen batched prefill are explicitly
   hardware-independent ("we also use them in FA3") and bit-safe. The exp2-FMA
   emulation + conditional-rescale pieces are a numeric-config change — battery-gated,
   whole-engine or not at all (decode==verify law).
4. **DSpark verify-window policy (spec lane).** DeepSeek's confidence-scheduled
   verification (SGLang/TRT-LLM/vLLM all shipped it) is the industrial version of the
   just-shipped in-round confidence cut. The adoptable delta: the verify-cost-aware
   throughput-profile term in window sizing. Host-side Rust, no kernels, exactness-neutral.
5. **Cross-request prefill concatenation scheduler (H100 serving).** The lane's own
   closing verdict: scheduler work, not kernels — vl cores it needs are shipped. This is
   the biggest serving-prefill lever that stays inside exact math.
6. **DFlash block-diffusion drafter (watch, then adopt).** The new SOTA drafter class
   (>2.5x over EAGLE-3 reported; single-GPU 5.1–5.8x over autoregressive). Verification
   plumbing = existing MTP path; drafter is batched GEMMs + attention (no sequential
   loop). Blocked on a Qwen3.5-class checkpoint (z-lab publishes Qwen-family; no 3.5-9B
   yet). Track z-lab/dflash; training our own is an artifact-axis decision.
7. **Tree drafting validation point (5090 tree-draft lane).** SGLang Spec V2 ships tree
   drafts default-on including hybrid-linear (GDN) models — closest existing system to
   the designed root-anchored-paths lane; their page>1 verify is the reference to read
   before building v1.
8. **cuBLASLt warning (defensive):** 13.3U1 marks non-default epilogues with int8
   cublasLtMatmul "planned for removal next major release" — never build the int8
   prefill path on Lt epilogues; any future w8a8 arm goes through a hand kernel or
   CUTLASS (which is address-deterministic anyway, unlike Lt/nvjet — the prime-graph
   blocker).
9. **SM120 references banked:** CUTLASS 4.5/4.6 added 128x32/64 and tileN=8/16
   block-scaled tiles for sm_120 (decode-shaped GEMM signal) and ptr-array grouped FP8
   GEMM (Hy3 expert-bank relevance); the Mattana FP4-attention writeup documents sm_120
   `mxf8f6f4` fragment layouts + 99 KiB smem + `scale_vec::1X`-only — load-bearing facts
   for any future sm_120 block-scaled kernel. Q8_0 cannot map losslessly onto hardware
   block-scale (ue8m0 exponent scales vs fp16 d) — epilogue-side scaling stays the exact
   route.
10. **Competitive re-pins:** vLLM v0.25 made Model Runner V2 default — the 122.8% decode
    standing was measured against 0.26; re-pin the version on the next H100 head-to-head.
    llama.cpp shipped MTP with adaptive draft length (July 2026) — re-sweep its spec-best
    flags before the next 5090 board pairing. HF TGI entered maintenance; no new
    single-GPU competitor emerged.

## Explicitly not taken
- FP4/INT8 attention (SageAttention3-class): blocked by exact-math laws on decode and
  accuracy laws on scored prefill.
- FP8/FP4 KV on decode paths: fp8-kv door just reverted by measurement (−1% at 12k);
  the ecosystem's FP4-KV-with-full-precision-linear-state split is the template if a
  prefill/judge-lane door ever opens.
- Dual-batch overlap: all current activity is multi-rank EP — nothing single-GPU adopts.
