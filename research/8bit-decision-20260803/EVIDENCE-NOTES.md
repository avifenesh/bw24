# 8-bit decision — supporting evidence notes (raw extracts + verification trail)

Companion to DECISION.md. Raw quotes preserved here so the decision doc's citations can be
audited without re-opening the sources. Gathered 2026-08-03, repo tree 69cdd1eb
(restructure/public-split), no GPU runs.

## 1. The g7e fp8_lt probe row — full recovered fragment

`research/tune-data/g7e-rtx6000.jsonl` line 39 is a `recovered-fragment` wrapper whose `raw`
field is the original row (ts 2026-07-08T03:30:00+03:00, rig g7e-rtx6000-sm120-188sm, commit
lane/prefill-fp8). Key fields verbatim:

- change: "FP8-ACT PREFILL CARD — MICRO-PROBE PHASE (probe-first law): cuBLASLt FP8-E4M3 GEMM
  vs the current 27B prefill GEMM classes at the real NVIDIA-27B shapes (n_embd 5120, ffn
  17408, T=512/2048/4096/6257). Baseline anatomy first: BW24_PP_ONLY nsys of pp6257 …
  = 2024.8 tok/s median (N=3, spread <0.2%). Kernel shares: qmatvec_gemm_q8_0 46.5% of GPU
  time (the F8-E4M3-origin attn/linear-attn projections re-encoded Q8_0 at load) +
  mul_mat_q_nvfp4_w4a8 30.4% (MLP gate/up/down NVFP4) = 77% GEMM. Per-shape TFLOPS from nsys
  grid buckets at m=4096-chunk: q8_0 GEMM 47-72 TF (kv_proj 47, o_proj 62, lin_ba 68,
  lin_qkv 72, q_gate 72); W4A8 MMQ 241 TF (both MLP shapes)"
- result.cublaslt_fp8_e4m3_tflops (m_axis [512, 2048, 4096, 6257]):
  - o_proj_5120x6144: [624, 703, 676, 668]
  - lin_qkv_10240x5120: [626, 659, 726, 726]
  - q_gate_12288x5120: [668, 707, 723, 779]
  - kv_proj_1024x5120: [346, 630, 665, 612]
  - lin_ba_6144x5120: [613, 684, 682, 670]
  - ffn_gate_up_17408x5120: [695, 758, 769, 794]
  - ffn_down_5120x17408: [734, 790, 772, 730]
- result.speedup_vs_current: attn_linear_q8_0_layers "8.7-14.2x", mlp_mmq_layers "2.9-3.3x"
- act_quantize_cost: "f32->fp8 per-token-scale kernel 0.007-0.118 ms at k=5120 m=512-6257
  (1.4-3.1 TB/s eff) — quant+GEMM chained still ~700 TF effective at q_gate shape"
- scale_mode_probes.per_token_OUTER_VEC_32F: "NOT_SUPPORTED sm120
  (cublasLtMatmulAlgoGetHeuristic status=7 nh=0 all m) — per-token act scale must be folded
  outside the GEMM (row-rescale epilogue or scale folded into fp8 codes + f32-D row" [fragment
  truncates here]

Fragment caveat: the row is marked recovered; the fragment ends mid-sentence. The probe
sources still exist: `probe/fp8_lt_prefill.cu`, `probe/fp8_lt_scale_probe.cu`,
`probe/fp8_vec16_probe.cu` (verified present).

Also on that file, line 35: the vLLM reference row for nvidia/Qwen3.6-27B-NVFP4 (vLLM 0.24.0,
prefill steady ~6700 tok/s, decode 65.7-66.6; Marlin-not-native-FP4 on sm_120 noted).

## 2. The F8→Q8_0 re-encode path — file:line inventory

| What | Where |
|---|---|
| Raw ST dtype mapper panics on F8_E4M3/E5M2/E8M0 | crates/memra-gguf/src/safetensors.rs:40-42 |
| Plain-arm F8_E4M3 2D + `.weight_scale` → f32 dequant → `f32_to_q8_0` | crates/memra-gguf/src/source.rs:1012-1046 (re-encode at :1041-1046) |
| In-code rationale ("~1.06B/elem instead of a 22GB f32 blow-up (OOM, measured)… per-32 q8 re-quant is a FINER grid") | source.rs:981-986 |
| `MEMRA_NV_W4=1` F8→NVFP4 alternative (0.56 B/w) | source.rs:1030-1040 |
| Transform-arm (V-reorder) twin re-encode | source.rs:1097-1112 |
| BF16≥1M-element → Q8_0 "loader law" (Float-poison trap) | source.rs:988-1011 |
| `f8_row_scales` — accepts n==1 or n==out_f ONLY (the block-128 blocker) | source.rs:838-851 |
| FP8-native access for the prefill stash (`fp8_native`) | source.rs:876-882 (doc), impl below |
| `MEMRA_PP_FP8` gate + probe-verdict header | crates/memra-engine/src/fp8_ffi.rs:1-20, 43-52 |
| `MEMRA_ST_E4M3` one-copy mode doc | fp8_ffi.rs:55-62 |
| cuBLASLt heuristic cached per plan (determinism note) | crates/memra-engine/cu/fp8_prefill.cu:14, 143-148 |
| ST serve entry: dir path → SafetensorsSource in the server worker | crates/memra-server/src/worker.rs:610-625 |
| ST gate harness | crates/memra-engine/src/bin/run_safetensors.rs:5,14 |
| ST consumers incl. run_gen/run_spec/decode_bench/frspec_owngen | grep list, 13 files (run_lockstep, st_vs_gguf, replay_acceptance, …) |

## 3. rig5090.jsonl rows used (line → gist)

- 184: NV-27B load-tail OOM root cause — Transform-arm F8 fell through to F32 (462MB/layer x48);
  fix = post-reorder Q8_0 re-encode.
- 186: NV-27B full gate battery with model-trained MTP head live from safetensors; BF16
  embed_tokens → Q8_0 host re-encode; run-spec ST K-sweep ALL PASS.
- 204: PLAIN-FIRST board 2026-07-07 — 27B bw24 44.3-46.1 vs llama 44.3-45.6 "PARITY (both
  bandwidth-bound at same weight bytes)" — the decode-format-insensitivity receipt.
- 233: FP8 budget sweep, pp1845 887.9→970.0@1536→1035.3@2560→1110.2@3584 (+25%), OOM@4608.
- 234: bracket completed — 1136.3@4096 (+28.0%); gates at 3584: argmax MATCH maxdiff 0.000e0,
  chat decode 39.75 = no decode regression.
- 256: BW24_ST_E4M3 local verdict — pp1845 1291.2→1364.1 (+5.6%) at FULL coverage, −7.3GB
  resident; argmax MATCH; **spec K=3 p2 69.0→63.6/64.2 (−7%, consistent x2) — OPPOSITE of the
  box (+7.1%)**. "J/token law, third confirmation… Kernel verdicts DO NOT TRANSFER across
  power walls." Verdict: opt-in pp-heavy serve mode on the laptop rig.
- 266 (tag 9bst-modelopt): 9B ST modelopt full battery — tg128 127.88 (GGUF-parity), argmax
  MATCH maxdiff 0.000e0, coherent 3/3, run-spec K1-8 8/8 PASS.
- 268: NV-27B ST standing config — spec best K=3 HPOST=1 pmin0.4 NV_W4=1 FRSPEC_TRIM = 95.4
  tok/s (2.01x plain 47.5); frspec_rank patched to accept HF dirs (GGUF-free ST toolchain).
- 292: k32-imma CLOSED DOMINATED (int8-lineage acceptance tax; prefill-KV law).
- 294: pp-gap-SOLVED-llama-runs-w4a4 — llama's NVFP4 prefill lead is W4A4 e2m1 activations
  (exactness-rejected on our stack).

## 4. Board / current-board.json extracts (updated 2026-08-02)

- plain_decode q27 row: memra 47.6 vs llama 43.7 (NVFP4 GGUF, tg128 d512).
- speculative q27 row: memra 116.4/101.2/86.0 vs llama 91.7/93.3/81.5 (K=3 + own-gen trim).
- h100_board q27 row: memra e2e 96 vs vLLM-0.26-FP8 73 → 1.31x (vllm_artifact "FP8").
- supported_models: 9B "NVFP4 (5090), Q8_0 (H100)"; 27B "NVFP4, Q4_K_M MTP-baked".
- **Gap confirmed: no q27 Q8_0 cell exists on any rig** (also qwen38-prep AUDIT.md §5 notes
  3.6-27B never got a perf-cells entry).

H100 q27 bring-up context (ARCHITECTURE-H100.md:2208-2222): artifacts = unsloth Q4_K_M
MTP-baked GGUF (17GB) + Qwen/Qwen3.6-27B-FP8 (31GB, the vLLM arm); first cell memra decode
87.5 / prefill 1965 → e2e 74.3 vs vLLM FP8 74.3 / **15054** → 72.9.

## 5. verify-tier (where q8_0 GLUE costs live)

research/verify-tier-20260802/RESULTS.md §2 (q27): T=2 premium +2.446 ms/pass = 50% glue /
21% fa_attn / 20% net matvec; glue share falls to 6% by T=8. Top glue kernels at T=2
(glue-attribution.md q27 detail): quantize_q8_1 0.861 ms, rms_norm_f32 0.682, add_f32 0.234,
l2_norm_f32 0.180. Interpretation for this decision: activation-side (q8_1 quantize + norms),
identical under any weight format — no format credit.

## 6. kv-compress survey (KV side is already settled)

research/kv-compress-20260802/REPORT.md §1.1-1.3:
- memra q8_0/q5_1 KV = 58 B per 32-elem K+V block = 45.3% of BF16 — smaller than fp8-flat (50%).
- FLAGS.md:84: `MEMRA_KV_K` fp8 FLIP-BLOCKED (e2e flat + 9B-ST spec acceptance 74%→20.5% FAIL).
- vLLM 2026-04-22 law #1: accumulation precision (not storage) was the long-ctx killer;
  Hopper FA3-FP8 128k NIAH 91%→13%; Blackwell unaffected; our FA accumulates f32 order-pinned.
- Law #2 (skip sliding-window layers) = our MEMRA_GEMMA_WKV acceptance-law refinement.

## 7. DeltaServe / LoRA triggers (research/deltaserve-assessment-20260803/ASSESSMENT.md)

- :113-116 "memra has no LoRA support at all (grep of src/ — zero hits), no multi-LoRA
  batching, no autograd, and serves GGUF-quantized weights through hand-written kernels."
- :142-146 the shared-frozen-base assumption breaks on GGUF-served weights: backward needs
  QLoRA-style dequant-backward or a bf16 copy (VRAM-prohibitive next to a served 27-35B).
- :149-158 exactness collision — mixed-batch co-serving changes GEMM m-dims, the
  `MEMRA_ROUTER_PREFILL_EXACT` defect class; compatible subset = temporal-idle FT + MPS
  backward.
- :190-203 GO-later triggers (all currently unmet): measured idle >25-30% over ≥2 weeks of
  real traffic; fine-tune track unblocked and LoRA-shaped; a backward story for quantized
  weights; timing-interference gate on-box.

## 8. Qwen 3.8 / day-one artifact expectations

- WATCH.md: official announcement 2026-08-03 (X + TechNode), release expected week of
  2026-08-10; architecture UNKNOWN (no config.json yet); unsloth announced 17GB-local intent.
- WATCH.md:26-28: "Whether an FP8/NVFP4 official quant ships alongside BF16: unknown (3.6
  precedent: NVIDIA published an official NVFP4 repo ~5 weeks after release; unsloth NVFP4
  came earlier)." — note the 3.6 precedent ALSO includes Qwen's own -FP8 sibling (below),
  which the WATCH note did not enumerate.
- HF (fetched 2026-08-03): **Qwen/Qwen3.6-27B-FP8 exists**, model card: "The quantization
  method is fine-grained fp8 quantization with block size of 128, and its performance metrics
  are nearly identical to those of the original model." Search results date the repo to the
  3.6 release month (April 2026). Qwen ships -FP8 siblings across the line (Qwen3-0.6B/8B-FP8
  since 2025-05, Qwen3.6-35B-A3B-FP8, Qwen3-Next-80B-FP8 — all "fine-grained fp8… block
  size 128").
- Bring-up wall-clock, same-arch fast path: ~8-11 h one working day
  (docs/qwen38-bringup-runbook.md, estimate table).

## 9. W8A8-INT8 rejection receipts

- docs/FLAGS.md:417-421 (sm_90a refuted list): "W8A8/fp8/CUTLASS/Lt-autotune prefill GEMMs at
  m=512 AND m=2048; … Q8_0-EXACT int8 GEMM (triple-refuted: per-block rescale 5.4x naive, 17x
  pipelined…). The remaining single-seq prefill residual (~27% vs vLLM's INT8 GEMM class) is
  an owner-gated accuracy decision (w8a8-class numerics change model outputs)."
- rig5090.jsonl:292 k32-imma CLOSED DOMINATED.
- Ecosystem: vLLM docs position INT8-W8A8 (LLM-Compressor) as the Ampere/pre-FP8 path; FP8
  W8A8 is the Hopper+ recipe with >99% recovery (docs.vllm.ai fp8 page; Red Hat 2024-07-15).

## 10. Target-box context (2x5090)

- research/hw-buy-20260802/REPORT.md:443 — "2x 5090 used, $8.4k, 600 tok/s sat, $0.47/Mtok"
  rank-1 buy row; :41-44 laptop dense decode ~85% of peak BW; desktop = 1.79 TB/s/card, ~2x
  the laptop's bandwidth.
- PP-2: `MEMRA_PP_STAGES`/`MEMRA_PP_SPLITS` M1 bit-identical (2026-08-01), M2 N=2/4/8
  bit-identical serial arm; deferred readback experimental (FLAGS.md:321,323).
- 27B at 8-bit ≈ 27 GB weights + KV: single 32 GB card is tight (KV + drafter + activations);
  PP-2 sharded is the comfortable shape. Decode ceiling projection (labeled in DECISION.md
  §2a): ~55 tok/s plain single-stream.

## 11. Web source list (all fetched/searched 2026-08-03)

1. https://huggingface.co/Qwen/Qwen3.6-27B-FP8 — full card fetched (block-128 FP8 quote,
   MTP serving configs, vllm/sglang commands).
2. https://docs.vllm.ai/en/v0.21.0/features/quantization/fp8/ — FP8 W8A8: per-tensor W scale,
   dynamic per-token activations.
3. https://developers.redhat.com/articles/2024/07/15/vllm-brings-fp8-inference-open-source-community
   — ">99% accuracy preservation" lm-eval receipts.
4. https://github.com/vllm-project/vllm/issues/33301 — RFC: FP8 LoRA (2026-01).
5. https://docs.vllm.ai/en/stable/features/lora/ — multi-LoRA serving baseline.
6. https://developer.nvidia.com/blog/model-quantization-post-training-quantization-using-nvidia-model-optimizer
   — FP8_CFG = W8A8 per-tensor static AbsMax; granularity menu.
7. https://nvidia.github.io/TensorRT-LLM/1.2.0rc5/features/quantization.html — sm100 FP8
   block-wise = MXFP8 recipe (E4M3 + UE8M0 scales) vs sm90 (E4M3 + FP32 scales); sm120
   unstated — probe P1 required.
8. https://www.spheron.network/blog/tensorrt-model-optimizer-modelopt-quantization-guide/ —
   ModelOpt targeting guidance (FP8→Hopper, NVFP4→Blackwell).
9. Qwen -FP8 sibling repos (search hits): Qwen3-0.6B-FP8, Qwen3-8B-FP8, Qwen3.6-35B-A3B-FP8,
   Qwen3-VL-8B-Instruct-FP8, Qwen3-Next-80B-A3B-Instruct-FP8 — all fine-grained block-128.

Not verified first-hand: SmoothQuant paper (characterized via vLLM docs positioning); cuBLASLt
block-scaled FP8 support matrix on sm_120 (CUDA 12.8/12.9 documents sm90/sm100; probe P1);
LLMStation/FlexLLM (characterized via the DeltaServe assessment, which itself flags this).
