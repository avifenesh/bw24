# DFlash block-drafter bring-up (the 31B spec-depth lane, started 2026-07-13)

Target cell: 31B spec d1736 = 0.893x (87.7 vs llama server-MTP 98.2). Every kernel/policy/
parity lever on the MTP round is measured to a verdict (jsonl 2026-07-13 rows); the remaining
mechanism is MORE ACCEPTED TOKENS PER ROUND — DFlash/DSpark is the published drafter class
that delivers it (llama PR #25549 table: beats gemma4 MTP assistant on EVERY row, 3.90-4.46x
vs 3.44-3.99x, accept 68-88%; Hikari07jp 31B card: +5% over z-lab on Blackwell at K=15).
Same class exists for 26B AND qwen (shared clause of the goal).

## Assets

- **Checkpoint (downloading)**: /data/ai-ml/hf-models/dspark-gemma4-31b-draft
  (Hikari07jp/DSpark-Gemma-4-31B-draft, Apache-2.0, retrained from z-lab
  gemma-4-31B-it-DFlash). Two variants: repo root = DSpark semi-AR markov head (needs the
  vLLM patch semantics — SKIP first); `backbone-only/` = stock DFlash (the reference-code
  semantics below — START HERE).
- **Reference impl**: /data/projects/dflash (z-lab/dflash, shallow clone). THE file:
  `dflash/model.py` (366 lines, transformers backend) — the round loop `dflash_generate`
  (l.63) + `DFlashDraftModel` (l.302).
- z-lab also publishes drafts for gemma-4-26B-A4B-it, Qwen3.5-9B, Qwen3.6-27B/35B-A3B — the
  whole board. 12B official DSpark: deepseek-ai/dspark_gemma4_12b_block7.

## Draft model architecture (from config.json + model.py, verified)

5-layer qwen3-CLASS mini-transformer (model_type "qwen3" but it drafts for gemma — the
draft's own arch is qwen3-shaped):
- hidden 5376 (== 31B target hidden), 64 q heads x hd128, 8 kv heads, mlp 10752 (silu),
  rms eps from config, rope standard qwen3, q_norm/k_norm per head_dim.
- layer_types: 4x sliding_attention + 1x full_attention (sliding_window from config).
- Extra weights vs a plain qwen3 layer stack: `fc` = Linear(6*5376 -> 5376, no bias),
  `hidden_norm` = RMSNorm(5376), final `norm` = RMSNorm(5376).
- NO embed, NO lm_head of its own: reuses TARGET's embed_tokens and lm_head.
- block_size 16, mask_token_id 4, target_layer_ids [1,12,23,35,46,57] (6 taps).
- bf16 safetensors, ~1.5B params ≈ 3GB (quantize later; drafter-quant is a measured arm,
  NOT a default — exactness lives in the verify).

## Round semantics (dflash_generate, exact)

1. PREFILL: target forward over the prompt with hidden-state taps. target_hidden =
   concat over 6 taps of hidden_states[layer_id + 1] (offset 1: index 0 = embeddings, so
   [l+1] = OUTPUT of layer l) -> [t, 6*5376].
2. Round (block = 16): block_ids = [last_accepted, MASK x 15]. noise_embedding =
   target.embed_tokens(block_ids).
3. Draft forward: ctx = hidden_norm(fc(target_hidden)) for the NEW accepted tokens only
   (draft KV caches ctx K/V across rounds — crop to `start` after each round). Per layer:
   q = q_proj(noise) [q_len=16 only]; k/v = [proj(ctx_new) ; proj(noise)] appended to the
   draft KV; q_norm/k_norm; rope at ABSOLUTE positions (ctx tokens at their positions,
   block at start..start+15); attention NON-CAUSAL (block queries see all ctx + whole
   block), sliding window per layer type. Standard residual+mlp.
4. draft_logits = target.lm_head(norm(h)[:, -15:]) -> block_ids[1..] = argmax (greedy).
5. VERIFY: ONE target forward over the 16-token block (positions start..start+15), taps on.
   posterior = argmax(target logits). accept = cumprod(block[1:] == posterior[:-1]);
   commit accepted+1 (bonus = posterior[accept]); crop target KV to start'; draft ctx
   features for the next round = taps of the ACCEPTED+1 rows only.
6. Repeat. Greedy chain == plain decode BY CONSTRUCTION (same verify contract as MTP).

## bw24 implementation map

1. **Loader** (new `dflash.rs` or gemma_spec extension): safetensors via SafetensorsSource;
   names = `layers.{i}.self_attn.{q,k,v,o}_proj.weight`, `.q_norm/.k_norm`, `layers.{i}.mlp.
   {gate,up,down}_proj`, `layers.{i}.{input,post_attention}_layernorm`, `fc.weight`,
   `hidden_norm.weight`, `norm.weight`. bf16 -> FloatBf16/quant-on-load reuse.
2. **Target taps**: gemma4 trunk (31B dense path, hybrid_forward) — optional tap sink:
   after layer l in {1,12,23,35,46,57}, copy the residual-stream row(s) into a device tap
   buffer [t, 6, 5376]. Prefill primes taps for the whole prompt (transient ~224MB f32 at
   1736 — or capture bf16/f32 rows layer-by-layer into a preallocated buffer). Verify
   emits taps for its 16 rows (reuse the same sink).
3. **Draft kernels**: ALL existing primitives — matmul/matmul_pre (fc, qkv, o, mlp),
   rms_norm (+ per-head q/k norm = rms_norm_qkv exists), rope_neox (qwen3 rope), and
   attention: q_len=16 non-causal over [ctx_kv + 16] with sliding window on 4 layers —
   fa_prefill has a causal flag; need mask = full-visible (non-causal) + window on ctx.
   NOTE the window: sliding layers attend the last `sliding_window` ctx tokens — same
   window-view trick as gemma swa (view offset), then non-causal within.
4. **Draft KV**: 5 layers x nkv8 x hd128, grows with ctx (crop = len rollback, host math —
   same KvLayer machinery, f32 or q8_0 later).
5. **Round integration**: new arm in gemma spec serving (BW24_DRAFT_DFLASH=path seam):
   replaces the MTP K-chain; verify path UNCHANGED (decode_step_t_am_dev at t=16). The
   straddle-split fix (2026-07-13) already makes t=16 verify exact across ladder rungs.
6. **Gates**: run-spec/gemma-gate self-consistency 64/64 + 256/256 (the E4B lesson: gate at
   256 minimum), VERIFY-GATE maxdiff 0.000e0 (parity law), then the depth cell A/B.

## Cost model (why this should clear 0.893x)

MTP round at depth: ~6 chained drafts + t=7 verify, accept ~2.5-3/round, verify reads whole
trunk (~20ms byte floor at d1736 on 5090).
DFlash round: ONE 16-row draft forward (~3GB bf16 drafter ≈ 3.5ms; quantized q8 ≈ 1.8ms)
+ ONE t=16 verify (same trunk read; t=16 b-tiers amortize BETTER than t=7) + published
accept length ~4-6 at T=1. Net: fewer, fatter rounds; tokens/round nearly doubles while
round cost grows ~15-25% -> the +23% depth gap is inside the published envelope (llama PR
table shows 3.90x vs 3.44x on exactly this comparison).

## Order of work

1. Verify download complete; inspect backbone-only/model.safetensors names + shapes.
2. Loader + CPU-oracle forward of the draft (tiny: 5 layers) vs the transformers reference
   on one block (torch on-box in ~/.venvs/torch) — argmax parity of draft tokens.
3. Target taps (31B trunk) + prefill tap priming.
4. Round arm + gates.
5. Depth cell A/B (frozen protocol NGEN=128) + short cell; 26B draft next (z-lab
   gemma-4-26B-A4B-it-DFlash); qwen drafts ride the same code (shared clause).
