export const meta = {
  name: 'bw24-pp-and-35b-closure',
  description: 'Prefill MMQ-class vendor (the 0.55-0.74x structural gap) + 35B closure lane (spec verify m-scaling + d6257 residual)',
  phases: [{ title: 'Lanes', detail: 'two worktree lanes; probes allowed, full-model timing main-thread' }],
}

phase('Lanes')

const COMMON = [
"CONTEXT: bw24 = from-scratch Rust+CUDA LLM inference engine at /home/avifenesh/projects/bw24 (RTX 5090 Laptop sm_120a, 82 SMs, 24GB, 858 GB/s, CUDA 12.8 for kernels). llama.cpp floor, source at /home/avifenesh/projects/llama.cpp (build with CUDA 12.8, bins in build/bin). TODAY'S STATE: plain decode beats llama on all 3 models at d512 (1.02-1.10x, FA_V2 tile-batched softmax default since today); PREFILL LOSES EVERYWHERE: 9B 4631 vs 6287 (0.74x), 27B 1297 vs 2348 (0.55x), 35B 2387 vs 3981 (0.60x) — pp1845, PP_ONLY protocol. 35B spec loses 0.92-0.97x.",
"HARD RULES:",
"1. WORKTREE: git -C /home/avifenesh/projects/bw24 worktree add /home/avifenesh/projects/bw24-LANENAME -b lane/LANENAME main; build cd <worktree> && cargo build --release --bins. Never touch main checkout.",
"2. GPU: gates + micro-probes allowed (single-kernel probe bins or short nsys, <60s/run). Ignore the persistent bge embedder in nvidia-smi (-ngl 0); other processes = wait 60s retry 10x. Full-model tok/s timing = main thread only.",
"3. ENV LAW: BW24_FAST=1 BW24_GEMM=1 BW24_MMVQ=1 BW24_FA_VEC=1 (+BW24_MOE_CACHE=1 for the 35B). Models: 9B /data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf, 27B /data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf, 35B /data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf.",
"4. EXACTNESS: prefill kernels must keep the per-(token,row) dot FP order OR be gated as an explicitly different numeric config with the full battery green (kernel-check ALL GREEN on the 27B, run-gen argmax MATCH on every affected model, run-spec self-consistency PASS). FP-order lessons say: within-config bit-identity is the contract.",
"5. DELIVERABLE: env-flagged (default OFF), micro numbers, gates green, committed on the lane branch (NOT pushed/merged), JSONL row (date 2026-07-09 if past midnight else 2026-07-08, honest incl failures), exact main-thread bench line.",
"6. Blocked >30min: commit partial, record blocker, report.",
].join('\n')

const MMQ = [
"YOUR LANE (key: ppmmq, worktree bw24-ppmmq, branch lane/ppmmq): close the prefill gap by vendoring llama.cpp's MMQ mechanism (per-block int8 tensor-core GEMM for quantized weights) for our prefill GEMM class.",
"FACT BASE: llama's pp wins 1.4-1.8x with the SAME weight bytes. Their mechanism = mul_mat_q (ggml/src/ggml-cuda/mmq.cu*): activations quantized to q8_1 per-block, weights' int8 codes fed to mma.sync m16n8k16 (or dp4a fallback) with per-block scale application in the epilogue — quality-identical math to our dp4a GEMM class, ~4-8x the throughput. OUR current prefill GEMMs: nsys says prefill time is dominated by qmatvec_nvfp4_mmvq_* (that is the DECODE matvec running at m=1 per prompt token in some paths — check!), mul_mat_q_nvfp4_w4a8 (existing int8 MMA GEMM), and the qmatvec_gemm_q8_0 class (47-72 TF measured). Known census artifact: the argmax gate re-runs the prompt token-serial — profile with BW24_PP_ONLY=1 to see the REAL prefill kernel mix.",
"STEP 1 (diagnose, ~1h): nsys profile BW24_PP_ONLY=1 run-gen on the 27B and 9B; build the honest prefill kernel-time table per model. Answer: what fraction rides (a) real GEMM kernels and at what TF, (b) token-serial matvec fallbacks, (c) glue. THEN nsys llama-bench -p 1845 same models: their mul_mat_q per-quant kernel times for the same shapes. The gap decomposition decides step 2.",
"STEP 2 (attack the biggest slice): likely candidates in order — (i) if token-serial matvec appears in OUR pp path: route those tensors through the existing GEMM class (dispatch bug/threshold, cheap fix); (ii) vendor MMQ tile loaders for the quant types where our GEMM class is slow (q8_0 first — the 27B attn projections; then IQ4_XS for the 35B experts at prefill; NVFP4 already has w4a8); (iii) if our w4a8 GEMM itself is far below llama's mmq TF at the same shapes, diff the kernels (tile sizes, cp.async pipeline depth, smem layout) and vendor their tile geometry. MMQ is big — a SINGLE quant type done well beats three done half.",
"Micro-bench: probe bin timing your GEMM vs the old path at real prefill shapes (m=512/1845/2048), plus llama's kernel time from nsys as the bar. Env flag BW24_PP_MMQ=1.",
"Main-thread bench line: BW24_PP_ONLY=1 BW24_PP_REPS=3 pp1845 A/B per model.",
].join('\n')

const CLOSE35 = [
"YOUR LANE (key: close35, worktree bw24-close35, branch lane/close35): the 35B's two remaining losing cells — spec verify cost (spec board 0.92-0.97x at 84% acceptance vs llama 76%: OUR per-round cost is the gap) and the d6257 plain residual (158.5 vs 159.9 after FA_V2; favendor proved the FA core is now FASTER than llama's — the residual lives in KV-append slope / GDN chain / scheduling).",
"JOB A (measure first): the verify m-scaling curve. Time decode_step_t on the 35B at m=1,2,3,4,6 (the spec verify batch sizes; there are batched b2/b4/b8 kernel tiers) via a probe or instrumented run-spec: us per verify call vs m. Compare llama: nsys llama-server self-MTP (--spec-type draft-mtp --spec-draft-n-max 2) short run, extract their verify-batch kernel times. If our m=3 costs >1.2x our m=1 while theirs ~1.05x, THAT is the 35B spec gap — then attack the worst batched tier (the census says MoE expert batching at tiny m is the suspect: moe_ffn_dev t>1 path).",
"JOB B (depth residual): profile d6257 decode (nsys, 32 tokens) and diff vs d512 kernel-by-kernel: which kernels GROW beyond FA (kv-append memset? gdn state? combine? scheduling gaps between kernels — measure the inter-kernel idle too, launch-count arc history says ~600 launches/tok has device-side latency). llama tg32@d6257 nsys as reference for what SHOULD grow. Fix what's fixable behind flags; report what is structural.",
"Deliverables: the two measurement tables (m-scaling curve both engines, depth-growth diff both engines) are REQUIRED even if no kernel change ships — they decide the next campaign. Any kernel/dispatch change: env-flagged + gated per COMMON.",
].join('\n')

const results = await parallel([
  () => agent(COMMON + '\n\n' + MMQ, { label: 'ppmmq', phase: 'Lanes' }).then(r => ({ lane: 'ppmmq', report: r })),
  () => agent(COMMON + '\n\n' + CLOSE35, { label: 'close35', phase: 'Lanes' }).then(r => ({ lane: 'close35', report: r })),
])

return results.filter(Boolean)