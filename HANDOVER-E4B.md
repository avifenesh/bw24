# E4B — GRAPH-EXEC-UPDATE ARC: DONE, VERDICT IN (2026-07-13)

BUILT (branch lane/graph-exec-update): graph_update.rs — shared, model-agnostic kernel-node
enumeration + per-token launch-geometry retune (cuGraphExecKernelNodeSetParams_v2 on grid.y
+ n_splits arg + the partO-paired combine's n_splits). E4B door rides it in generate +
generate_with via gemma4_e4b_graph_exec_loop (ONE capture at bucket=win, snapshot/rollback
around the capture warmups — the parked door DROPPED the 2 warmup tokens, stream was 3/64
not the recorded 64/64).

LANDMINES (all fixed, all gated): hd512 globals must capture UNDER the fa512 floor (else
dpl16_dc vs eager scalar = numeric-class drift); the combine merge count must track the
main exactly (in-graph memset means skipped split slots hold m=0.0, not NEG_INF empties);
main->combine pairing must be positional (partO pointers are reused pool transients).
CAPTURE-KEEP TAX (shared fix, lib.rs): retention clones inside the capture region became
1,440 dead D2D copy nodes/token (0.74ms) — keep scope is now warmups-only.

VERDICT (valid window, known cell 197.3): stream 400/400 IDENTICAL; argmax + chat green;
NGEN=128 -1.3%, NGEN=400 +0.9% -> steady-state replay beats eager, capture (~30ms)
crosses over ~200 tokens. DEFAULT: door ON at max_new >= 256 under window (BW24_E4B_GRAPH
forces 1/0). The est +10-15% did NOT materialize: eager enqueue-ahead already hides the
gaps; llama's 216.9-vs-188 edge is glue EXECUTION (dependency-chain latency), not gaps.
NEXT LEVERS: glue critical-path reduction, then the 26B/31B applications of graph_update
if their regimes ever need per-token geometry. E4B SPEC: WIRED 2026-07-13 (lane/e4b-spec —
shared GemmaDraft loader + round loop; gemma4_draft_kv_target rule; e4b batched verify;
short-prompt prime = tokenwise, the PRIME-FA class-skew fix). Gates: agreement 64/64 id +
128/128 chat all K, VERIFY-GATE bit-exact both KV classes. First light: chat K=2 236.2 vs
plain 187.7 = 1.26x self, 1.089x vs llama-plain-216.9 (llama cannot run the assistant on
this rig — fattn abort; surgered artifact gemma-4-E4B-it-assistant-llama.Q8_0.gguf). Own-gen
FR ranks banked (e4b-owngen-ranks-32768.txt, accept holds at trim). PENDING valid window:
trim A/B + K/serving freeze + board rows.

PROBES CLOSED NEGATIVE (do not retry): launch_bounds forcing (spills), fixed-bucket graph
replay (oversized grids -6..-11%), one-block PLE mega-fusion (single-SM weight pull, 126
vs 189 — kernel deleted, jsonl 2026-07-13), whole-token graph replay for launch-gap
recovery on 26B/31B/E4B-short (eager enqueue-ahead hides gaps; jsonl 2026-07-12/13).

# E4B — TRACING VERDICT (2026-07-13, supersedes the concurrency hypothesis below)

MEASURED: DRAM plateaus equal (52 vs 55%) but warps-in-flight 34% (ours) vs 72% (llama).
llama's kernels are NOT faster per-op; their mechanism = CUDA graphs with (likely)
cudaGraphExecUpdate per token: zero launch gaps AND exact grid shapes. Our fixed-bucket
graph replay measured -6..-11% (oversized fa grids per token); our eager enqueue-ahead
beats it. launch_bounds forcing = register spills = negative both models (reverted).
ARCS RANKED: (1) cudaGraphExecUpdate wrapper (cudarc lacks it - FFI to the driver API,
update kernel params in the instantiated exec each token, keep exact grids): the actual
llama mechanism, est +10-15%. (2) PLE mega-fusion one-block kernel (inp_gate 3.99us at
93GB/s + proj + gelu + quantizes -> one launch): +4% est. (3) wo-combine q8 fold: +1%.

# E4B — THE CONCURRENCY HYPOTHESIS (2026-07-12 night, THE lead for the 0.87x -> 1x gap)

nsys on llama's E4B decode: their kernels are SLOWER than ours per-op (big-FFN 43.5us vs
our 36.2; small mmvq 9.9 vs 5.1; more quantizes/norms; fa 7-10.9 vs 3.9) yet they serve
216.9 vs our 188.8. Their one-token kernel SUM appears to EXCEED their wall -> CUDA-graph
NODE CONCURRENCY: ggml's graph exposes independent ops and graph exec overlaps them across
SMs. Our engine is strictly serial (one stream; even our graph captures a serial stream).
NEXT: (1) confirm with nsys --gpu-metrics on llama (SM overlap) and a clean full-run
window; (2) if confirmed, restructure the E4B token enqueue into 2-3 streams with events
(e.g. attention chain || PLE tail of the PREVIOUS layer? — find true independence in the
dataflow; per-layer: qkv-cat matvec is independent of the PREVIOUS layer's PLE tail once
xn is out — the emit produces xn early) then capture THAT as a parallel graph. This is a
bigger arc than fold-grinding and likely the whole remaining gap.

# E4B GLUE-FUSION LANE — STATE + WAVE 3 SPEC (2026-07-12, branch lane/e4b-glue-fusion, merged through wave 2)

STANDING: 186.6 vs llama 216.9 same-window = 0.86x. llama serves AT the DRAM wall
(4.61ms/token = 3.9GB trunk at 847GB/s, ~zero glue). Bar = wall-serving.
WAVES DONE (+7.6%): fused3 qkv; PLE post_norm -> closing emit (rms_pre_add_scale_rms_norm_q8_1);
gelu_tanh_mul_q8_1 both sites (down/proj on matmul_pre); post-attn rms -> tail entry
(tail_core_pn); t=1 PLE row as contiguous view; tail entry emits zsh q8 (q8z, E4B arm only);
PLE residual add emits q8 (add_q8_1). ALL epilogues = quantize_q8_1's program verbatim;
battery green each wave (E4B argmax/chat/t=2, 26B/31B argmax+spec, kernel-check).
WAVE 4 DONE (2026-07-12 evening, merged): single-phase emit reductions (+1.2% — four
simultaneous sums, sum(v^2) recovered algebraically, one barrier round; FP-order change
gated MATCH) + qkv OUT-concat matvec (fused3 retired from the E4B path, flat, kept).
STANDING 188.8 = 0.87x vs llama 216.9. Waves 3a/3b (norm+rope folds) landed flat —
launch merges do NOT pay; the eager stream already overlaps. LESSON REFINED: wins come
from (a) removing HBM round-trips (waves 1-2, +5.2%) and (b) shortening the per-layer
DEPENDENCY CHAIN latency (wave 4a, +1.2%). Remaining ~28 tok/s = op-count restructure:
the MEGA-FUSION sketch (whole PLE tail as one kernel with inline Q4_0 matvecs — the
qmatvec_gemm decode_stage_inline pattern; 256-dim intermediate fits smem; kills ~8us of
chain latency/layer ~= +6%) is the next big swing; attention-block fold (qkv_rope +
append + fa at t=1, t_kv<=win) the one after.
PROFILE UPDATE (post-wave-2 nsys + RMS sweep): the matvecs are AT the wall (fused2 96% eff,
down 14.7MB at 17.3us = floor; head 0.64ms = floor). Glue = ~1.1ms across ~350 launches whose
EXECUTION floor is ~1-1.3us each — fusion pays exactly one launch-floor per kill, ~+1% each.
llama 4.61ms decomposes as ~4.2 weights + ~0.4 glue => their op count is ~2x lower; parity
needs ~6 more fold-kills OR op-count restructuring. RMS_BLOCK already 1024 for gemma (256
measures 168 vs 186.6 — do NOT lower). Two-reduction fused kernels run 3.8-3.9us (barrier
latency): separate smem arrays could shave one barrier (~0.3us x 90).
Glue inventory/token (med x count): rms_pre_emit .164, q8z .160, gelu_q8 .176 (84 calls,
2 sites), rms_f32 .158 (69 calls - AUDIT WHERE: shared-q 18 + inp_pl 1 + ???), fa_v4 .158,
combine .084, add_q8 .066, quantize .063 (50 calls - audit), rope .060, rms_qkv .045.
WAVE 3 (mechanical, ~1% each): (1) rope fold into rms_norm_qkv (pos+ff args in, one kernel);
(2) wo-input quantize via a combine-q8 variant (fa_decode_combine_f32 emits (q8,d), wo rides
matmul_pre — E4B-only call site to avoid qwen churn); (3) rope+append fold (rope epilogue
writes the quantized cache rows). WAVE 4 (the real remainder, ~0.3ms): small-matvec efficiency
— mr2_rp at ~79% of byte floor on the 2560-in mats (wq 4.4us vs 3.5 floor); occupancy/ILP
sweep on the mr2 tier (BW24_MMVQ_ROWS class knobs), consider fused2 for (inp_gate,proj-next?)
no — different inputs; consider CONCAT-WEIGHTS build for wq|wk|wv into ONE tensor at load
(single matvec, no fused3 grid split) — the fused3 grid is 3 sub-grids, a concat is one.
GOAL MATH: enumerable glue left ~0.35ms + matvec ineff ~0.3ms; killing both -> ~4.9ms = 204;
crossing 217 NEEDS the matvec arc to land near-fully. E4B has NO drafter (spec N/A) - plain is
the only cell.
# E4B STATUS 2026-07-12 (correctness UNBLOCKED + fusion port)

- ROOT-CAUSE FIX (commit 71fb028): global layers stored HALF their K/V — the cache kv-dim
  fallback ignored the SCALAR head_count_kv=2 (globals are 2x512=1024, not 512). Every
  cross-mode symptom (maxdiff 20-30, chat-prompt argmax flips, "noise amplification from
  layer ~6") was this. Gates now ALL GREEN: id maxdiff 0.67 MATCH, t=2 0.82 MATCH, chat
  water-cycle 2.09 MATCH + coherent. 26B/31B clean.
- FUSION PORT (commit c403a21): 26B/31B trunk structure (rms_norm_q8_1 carried pair ->
  matmul_pre wq/wk/wv, fused rms_norm_qkv, add_scale_rms_norm_q8_1 closing emit, head on
  matmul_pre). 164.2 -> 170.3, then 173.5 with the kv fix.
- STANDING: short plain 173.5 vs llama 208 = 0.83x. Remaining gap ~35 tok/s: profile said
  glue was 26% pre-port; remaining levers = per-row attention loop (fa rows arms for E4B),
  dc/graph serving arc, prime fa (wired, NOFA seam), head Q6_K at byte floor (leave).
- NOTE: correct-model output DIFFERS from the pre-fix stream (the old model was broken);
  any earlier E4B "baseline" numbers/streams predate correctness.

# E4B — FIRST LIGHT PASSED (2026-07-11) + PERF LANES WIRED (compile-clean, GPU-ungated)

## Perf lanes (this branch — GPU gates NOT yet run, main lane owns them)
1. DC ARM: `gemma4_e4b_decode_step_dc` — device token/pos/argmax, 4B/token host. The layer
   stack is `gemma4_e4b_trunk_core`, the SAME functions the eager chain runs (embed gather +
   per-layer-table gather moved to device-token cores: `gemma4_e4b_inp_pl_dev`) — stream
   identity with eager is by construction, not twin-kernel parity. Host KV mirrors advance
   dc-eager style; window views host math. Routed in decode.rs `generate` + `generate_with`
   greedy loops (the E4B exclusion gates dropped). Graph capture NOT wired —
   `gemma4_generate_graph` errors loudly on E4B.
2. Q4RP MIRRORS: hybrid.rs hook widened to e4b layers — attn wq/wk/wv/wo + dense ffn
   gate/up/down + inp_gate/proj (~2.2GB mirror for the 5.2GB model; build_q4_rp4 no-ops on
   non-Q4_0 like the F16 model_proj). KV-shared layers' aliased wk/wv mirror twice
   (~1.5MB/layer waste — dedupe with the weight-alias TODO).
3. PRIME FA: fresh-prompt own-KV hd256 layers with t <= window(512) ride ONE f32 fa_prefill
   (26B prime pattern) instead of t per-row quantized launches; globals + KV-shared layers
   keep the per-row loop. BW24_NOFA=1 reverts. NOTE: changes prime numerics class (f32 fa vs
   quantized per-row) — same split the 26B ships; argmax/chat gates arbitrate.

## GPU gate sequence for the main lane (in order)
E4B=/data/ai-ml/hf-models/gemma4-e4b-qat-gguf/gemma-4-E4B_q4_0-it.gguf
1. `BW24_NGEN=8 run-gen $E4B 2 818 5279 529 7001 563` — argmax MATCH (expect " Paris"; note
   prime numerics changed with lane 3, prefill logits shift within the f32-vs-quant class).
2. Chat coherence: water-cycle prompt n=48 — structured coherent answer.
3. DC-GATE equivalent: generate() now rides the dc loop — compare a 64-token greedy stream
   against BW24_PRIME_TOKENWISE-style eager decode_step chain (or temporarily force the
   eager loop) for stream identity; the dc arm uses the same trunk fns so any mismatch is a
   routing bug, not numerics.
4. Perf: run-gen tok/s (was 154.8 eager pre-mirrors) — expect the q4rp mirror + dc-argmax
   gain; then prime speed on a ~500-token prompt (fa arm).
5. If green: BW24_Q4RP=0 A/B for the mirror contribution, then jsonl + board rows.



Loads + coherent chat + id-prompt argmax MATCH (" Paris"). 154.8 tok/s eager (unoptimized).
Cross-mode (prefill-vs-decode) logit maxdiff runs 20-30 — 5-10x the 26B's — noise-amplification
signature (smooth growth from layer ~6), argmax gates MATCH; treat as the E4B baseline, re-check
after perf lanes land.

# E4B bring-up — fork-lane status (2026-07-11, forward wired)

FILES: main QAT 5.2GB + assistant drafters (F16/Q8_0) at
/data/ai-ml/hf-models/gemma4-e4b-qat-gguf/ (repo ids in research/gemma4-bringup/e4b-arch-map.md).
Arch map: research/gemma4-bringup/e4b-arch-map.md (E4B = gemma4 arch; per-layer embeddings
n_epl 256 + 18 KV-shared tail layers; NO altup/laurel; window 512; 42 dense layers ffn 10240).

## WIRED (compiles clean: cargo build --release --bin run-gen --bin gemma-gate)
- Loader: Gemma4E4bLayer/Gemma4E4bModel by tensor presence; KV-shared layers borrow the share
  target's k/v tensor handles; per-layer token table host-side + OnceLock device upload.
- Cache: E4B kv dims from the scalar-metadata fallback (512 both kinds); KV-shared layers get
  NO KvLayer (None — accidental use is a loud unwrap).
- FORWARD (dedicated first-light path, hybrid_forward.rs "gemma-4 E4B" block — the tuned
  26B/31B paths untouched):
  - prologue gemma4_e4b_inp_pl (gather*sqrt(256) + rms(model_proj . x / sqrt(2560)) then
    (a+b)/sqrt(2) — llama project_per_layer_inputs exact order);
  - gemma4_e4b_attn: own-KV layers project+norm+rope+append (REAL wv — no wv:=wk on E4B);
    KV-shared layers Q-only over cache.kv[target] (target 22 swa / 23 global); per-row causal
    fa_decode over the quantized cache (correctness path, window 512);
  - dense ffn via the 31B gemma4_layer_tail_core arm; per-layer-embed tail
    (gelu(inp_gate.resid) * inp_pl[il] -> proj -> rms(post_norm) -> +resid) then
    (resid+tail)*layer_output_scale — llama order;
  - gemma4_e4b_trunk (t-wide) drives: forward() prefill logits, prime_cache
    (gemma4_e4b_prime), eager decode_step_h (gemma4_e4b_decode_step_h). generate()/
    generate_with() route E4B to the eager loop (dc arm gated off).
- New primitives: copy_rows_strided_f32 kernel + Engine::copy_rows_strided (per-layer input
  slice gather).

## NOT WIRED (next)
- dc/graph serving arms (generate's device-counter loop is 26B/31B-only for now).
- verify/spec (gemma4_decode_step_t*, VERIFY-GATE, adaptive rounds) + the MTP drafter
  (NEW centroid head — n_centroids 2048/top_k 32; read llama gemma4-assistant.cpp first).
- prefill fa (prime is the per-row loop — slow, correct); q4rp mirrors for the dense trunk;
  chunked prime; geometry re-derivation asserts vs llama (global nkv=1x512 vs 2x256 — VERIFY
  at first light).

## FIRST-LIGHT SEQUENCE (main lane, GPU)
E4B=/data/ai-ml/hf-models/gemma4-e4b-qat-gguf/gemma-4-E4B_q4_0-it.gguf
1. BW24_NGEN=8 run-gen $E4B 2 818 5279 529 7001 563      # prefill-vs-decode MATCH gate
2. llama-server -m $E4B -ngl 99 --port 8099 -fa on; same prompt greedy n=32 — token-stream
   compare vs run-gen (the llama-match oracle; use BW24_GEMMA_PROBE=1-style bisect if off).
3. chat prompt (physics, chat-22 ids re-tokenized for E4B vocab=same gemma4 tokenizer) sanity.
4. depth prompt run-gen MATCH (the parity-law blind-spot oracle).
5. Then perf lanes: dc arm port, prefill fa, q4rp mirrors, spec+drafter.
