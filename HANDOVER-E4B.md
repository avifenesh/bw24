# E4B bring-up — fork-lane groundwork (2026-07-11)

DONE (this branch):
- Files on disk: main QAT 5.2GB + assistant drafters (F16/Q8_0) at
  /data/ai-ml/hf-models/gemma4-e4b-qat-gguf/ (repo ids in e4b-arch-map.md).
- Arch fully mapped: research/gemma4-bringup/e4b-arch-map.md — E4B is gemma4.cpp itself
  (NO altup/laurel); new pieces = KV-sharing (18 tail layers) + per-layer embeddings (256).
- Loader skeleton COMPILES (cargo check clean): Gemma4Config gains n_embd_per_layer +
  shared_kv_layers; Gemma4E4bLayer (inp_gate/proj/post_norm + kv_share target) loaded per
  layer by tensor presence; Gemma4E4bModel (tok-embd bytes host-side + model_proj F16 +
  proj_norm) on GemmaAux; KV-shared layers load the SHARE TARGET's k/v tensors (forward must
  skip k/v compute there). Loud eprintln marks the forward as unwired.
- Latent build fix the MAIN LANE also needs: kernel_check.rs's two fa_decode_rows calls
  missed the device-len signature change (parent built only gemma-gate/run-gen) — fixed here.

FORWARD WIRING TODO (ranked, per arch map):
1. gemma4_geom: per-layer kv head derivation for E4B (head_count_kv=2 metadata is SCALAR —
   Gemma4Config.head_count_kv Vec is EMPTY for E4B; derive from k tensor shapes: swa 2x256,
   global k-out 512 = 1x512 or 2x256 — VERIFY against llama at first light).
2. Cache: kv_share plumbing — shared layers allocate NO KvLayer, attention reads
   cache.kv[target]; append skips; rewind/len bookkeeping via the target (verify+spec safe).
3. Prologue: per-layer-embed inputs (gather Q6_K rows + F16 matmul + rms + scales — see map
   formulas); decide resident-vs-host for the 2.3GB tok table (26B embed-table pattern).
4. Layer tail: gelu(inp_gate.cur) * inp_pl[il] -> proj -> rms(post_norm) -> residual,
   BEFORE layer_output_scale. Small mmvq work; t>1 for verify.
5. window=512 geometry check (fa dims/dc buckets fine in principle; swa 5:1 pattern).
6. First-light gates: run-gen argmax vs llama (llama-server -m e4b), tokenizer fuzz reuse,
   then the standard battery (short+depth run-gen, VERIFY-GATE, DC/GRAPH, spec).
7. MTP drafter phase: gemma4_assistant with NEW centroid head (n_centroids 2048, top_k 32,
   use_ordered_embeddings, k_eq_v=false) — read llama gemma4-assistant.cpp before wiring.
