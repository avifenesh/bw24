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
