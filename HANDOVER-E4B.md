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
