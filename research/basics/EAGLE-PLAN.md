# EAGLE3.1 Speculative Decode — bw24 Plan

Capability: `eagle` (roadmap task #7, `[pending]`). EAGLE3.1 spec decode is mandatory. bw24 already
ships MTP greedy spec (`spec.rs`) plus the full verify/accept/rollback infra. EAGLE reuses the
target's hidden states to drive a separate draft model.

GATE (must both hold): the EAGLE3 spec token stream is **bit-identical** to plain greedy
`generate` (`decode.rs:99`), AND measured acceptance > 0 (target: > 0.3, healthy 0.6–0.8).

Bottom line up front: the heavy infra (verify, accept-walk, snapshot/rollback, quantized-KV
attention) is **already built and reused as-is**. EAGLE3 adds a draft model load + a draft
forward + aux-hidden capture. The on-disk draft checkpoint is real (`eagle3-qwen35-9b`, 800 MB),
so EAGLE3 for the 9B target is **NOT asset-gated**. For the 27B target it **IS** asset-gated (no
draft on disk) — there MTP stays the shipped spec. See §7.

---

## 0. Ground truth: the on-disk draft, and how it differs from the FINDINGS

The actual draft checkpoint at `/home/avifenesh/ai-ml/hf-models/eagle3-qwen35-9b/config.json`
forces several corrections to the research FINDINGS:

| Field (config.json) | Value | Consequence for bw24 |
|---|---|---|
| `architectures` | `LlamaForCausalLMEagle3` | Llama-style decoder layer (RoPE NeoX, SwiGLU), not Qwen3.5 GDN. Reuses `FullAttnLayer`/`Ffn::Dense` shapes. |
| `num_hidden_layers` | `1` | Draft = ONE transformer layer (decoder). Matches FINDINGS. |
| `vocab_size` | `248320` | target vocab |
| `draft_vocab_size` | `32000` | **draft has its OWN small vocab → `d2t` map is MANDATORY here, not optional** (FINDINGS understated this). |
| `eagle_config.eagle_aux_hidden_state_layer_ids` | `[1, 15, 28]` | the 3 trunk layers whose hidden states feed the encoder `fc`. |
| `eagle_config.use_aux_hidden_state` | `true` | confirms the 3-layer-concat encoder input. |
| `hidden_size` | `4096` | matches the 9B target trunk `n_embd` (`hybrid.rs:133`, `cfg.n_embd`). |
| `head_dim` / `num_attention_heads` / `num_key_value_heads` | `256` / `16` / `4` | draft attn is GQA 16:4, head_dim 256 — same shape family as trunk full-attn (`FullAttnLayer`). |
| `partial_rotary_factor` | `0.25` | draft applies RoPE to only 64 of 256 head dims → its own `rope_dim_count`, distinct from trunk. `rope_neox` already supports partial rotary (`spec.rs:154`, `cfg.rope_dim_count`). |
| `rope_parameters.rope_theta` | `1e7` | draft has its OWN rope base; do NOT reuse `cfg.rope_freq_base`. |
| `tie_word_embeddings` | `false` | draft ships its own lm_head; cannot fall back to target `output`. |
| format | `model.safetensors` (bf16, 800 MB) | loads via the safetensors seam (commit `41f0bc6`, `bw24-gguf/src/safetensors.rs`) — no GGUF conversion needed. |

Net: this is a **genuine separate EAGLE3 draft model**, NOT a NextN head. The bw24 `MtpHead`
struct (`hybrid.rs:120-130`) does NOT fit it — EAGLE3 needs a new `Eagle3Draft` struct (§3).

---

## 1. What EAGLE3.1 actually does (greedy, chain — the v1 we ship)

Per round, with K draft tokens:

1. **Encode (once/round):** take the 3 trunk aux-hidden states for the just-committed token
   (layers 1, 15, 28), concat → `[3*4096]`, project through `fc` → draft hidden `g [4096]`.
2. **Draft K tokens (autoregressive, T=1 each):** the single draft decoder layer takes
   `concat(embed(prev_draft_tok), g_prev) [2*4096]`, runs attn (its own scratch KV) + FFN →
   draft logits over the 32000 draft vocab → argmax → map through `d2t` → target token id.
   The draft's hidden output becomes `g` for the next step (EAGLE recurrence). K argmaxes → K
   candidate target tokens.
3. **Verify (one batched target forward, T=K):** `decode_step_t` over the K candidates
   (`spec.rs:179`) — UNCHANGED.
4. **Accept walk + bonus + commit + rollback:** UNCHANGED (`spec.rs:386-432`).

Difference vs MTP draft generation (the ONLY part that changes):
- **MTP** drafts with `mtp_head_forward(e_tok, h_seed)` and advances `h_seed` (a trunk-coupled
  pre-output_norm hidden) per step (`spec.rs:46`, `spec.rs:371-375`).
- **EAGLE3** drafts with a separate model fed `g` (encoder output of trunk aux hiddens), advancing
  the draft layer's own hidden per step. Target coupling is only through the once-per-round `fc`
  fusion of committed-context aux hiddens, not a per-step `h_seed`.

---

## 2. What REUSES MTP infra (no change) vs what is NEW

### Reused verbatim (the expensive, validated parts)

| Component | File:line | Why it transfers |
|---|---|---|
| Batched verify forward (all-column logits, T=K) | `spec.rs:179` `decode_step_t` / `spec.rs:188` `decode_step_t_h` | Verify is over the TARGET trunk; identical regardless of draft source. EAGLE candidates feed in exactly like MTP draft tokens. |
| Causal T>1 attention over quantized KV | `spec.rs:265` `full_attn_verify` → `lib.rs` `fa_prefill_view` | Same q8_0-K / q5_1-V resident cache, same causal mask alignment (`spec.rs:309-312`). |
| Greedy accept-prefix walk + bonus token | `spec.rs:386-396` | Greedy spec is exact for ANY draft; `t_pred(j)` math is unchanged. |
| Commit + advance | `spec.rs:401-407` | unchanged. |
| Snapshot before round | `cache.rs:98` `Cache::snapshot` (called `spec.rs:363`) | KV-len record + D2D copy of conv/ssm. Generic over draft source. |
| Rollback (KV truncate + recur restore) | `cache.rs:126` `Cache::rollback` (called `spec.rs:425`) | unchanged. |
| Batched partial-accept replay (single weight read) | `spec.rs:428` `decode_step_t_h` replay | unchanged. |
| Full-accept fast path (feed bonus, T=1) | `spec.rs:410-415` | unchanged. |
| Draft scratch-KV pattern | `spec.rs:17-38` `MtpScratch` | EAGLE draft layer needs its own scratch KV; clone this struct as `Eagle3Scratch` (1 full-attn layer, reset per round at `spec.rs:366`). |
| Safetensors loader seam | commit `41f0bc6`, `bw24-gguf/src/safetensors.rs`, `bw24-gguf/src/hf_mapping.rs` | draft ships as `model.safetensors`; load through the existing source-agnostic path. |
| Resident quantized GEMM dispatch | `decode_step_h` FFN/attn paths (`decode.rs:64-80`, `spec.rs:90-106`) | draft `fc`/attn/FFN are dense matmuls → same `matmul`/`matmul_pre` dispatch. No new kernel. |

### NEW (must build)

| Component | Where | Notes |
|---|---|---|
| **N1. Aux-hidden capture** — return trunk hidden of layers `[1,15,28]` (pre-`output_norm`, i.e. the residual-stream `x` after each of those blocks) from the trunk forward | new `decode_step_aux` variant of `decode.rs:39` `decode_step_h` and `spec.rs:188` `decode_step_t_h` | bw24 today captures only the FINAL pre-output_norm hidden (`decode.rs:87`, `spec.rs:248-253`). EAGLE needs 3 INTERMEDIATE layer outputs. Add a capture-layer set; at `il ∈ {1,15,28}` clone the current residual `x` (the `x2` produced at `decode.rs:83` / `spec.rs:244`) into 3 device buffers. Cheap: 3 × `clone_dtod` of `[n_embd]` (decode) or last column `[n_embd]` (verify). |
| **N2. `Eagle3Draft` struct + loader** | new in `hybrid.rs` (parallel to `MtpHead` at `hybrid.rs:120`) | Holds `fc [3*n_embd, n_embd]`, one `FullAttnLayer` (draft attn, partial-rotary, own rope_theta), `Ffn::Dense` (draft FFN), draft `output_norm`/`lm_head`, `d2t: CudaSlice<i32>` (32000→target id), draft RoPE params, aux layer ids. Loaded from the draft safetensors via `hf_mapping.rs` keys. |
| **N3. Encoder forward** `eagle3_encode(aux: [3*n_embd]) -> g [n_embd]` | new fn in `spec.rs` | one matmul `fc @ concat(aux0,aux1,aux2)`. Once per round. |
| **N4. Draft decoder forward** `eagle3_draft_token(prev_tok, g, scratch, pos) -> (draft_logits[32000], g_next)` | new fn in `spec.rs` (parallel to `mtp_head_forward` at `spec.rs:46`) | embed prev_tok (draft embd), concat with norm(g) → `[2*n_embd]`, draft attn (own scratch KV, partial RoPE, own theta), FFN, draft norm + lm_head. Mirrors the op structure of `mtp_head_forward` but with EAGLE's 2*n_embd concat-of-(embd, g) instead of (e_norm, h_norm). |
| **N5. d2t mapping** — map draft argmax (0..32000) to target id before push/verify | inside the draft loop in `generate_spec_eagle3` | `d2t` is a `[32000]` i32 table from the checkpoint. Apply on host after argmax (cheap). |
| **N6. Orchestrator** `generate_spec_eagle3` | new fn in `spec.rs` (parallel to `generate_spec` at `spec.rs:329`) | Same skeleton as `generate_spec`; swap the draft loop (`spec.rs:365-376`) for encode + N4 + d2t; everything else (verify/accept/rollback) reused. Prime loop must use the aux-capturing decode (N1) so `g` is available from token 1. |
| **N7. Draft embeddings** | `Eagle3Draft` (separate `EmbedHost`) | draft has its own (non-tied) token_embd over the 32000 draft vocab; load it, don't reuse target `embd`. |

Estimated new code: ~350–450 Rust (N1–N7), **0 new CUDA** — encoder/decoder are dense matmuls +
existing norm/rope/attn/ffn kernels (`rms_norm`, `rope_neox`, `fa_decode`, `silu_mul`, `matmul`).

---

## 3. Draft-head forward, reusing target hidden states (concrete)

Op sequence for `eagle3_draft_token` (one draft step, T=1), mirroring `mtp_head_forward`
(`spec.rs:46-122`) but EAGLE-shaped:

```
inputs: prev_tok (target id of previous accepted/draft token, mapped back via t2d if needed),
        g (encoder output [n_embd], from §N3, computed once per round),
        scratch (Eagle3Scratch: 1 full-attn layer KV), pos.

op1  e   = draft_embd.gather(prev_tok)            # draft vocab embedding [n_embd]
op2  gN  = RMSNorm(g, draft.g_norm)               # [n_embd]   (g already from fc)
op3  cat = [e ; gN]                                # [2*n_embd]
op4  inpSA = draft.fc_in? @ cat                    # if checkpoint fuses to n_embd; else attn takes 2*n_embd
op5  aN  = RMSNorm(inpSA, draft.attn_norm)
op6  attn = full_attn(aN, scratch, pos)            # partial RoPE (64/256 dims), rope_theta=1e7, GQA16:4
op7  x1  = inpSA + attn
op8  z   = RMSNorm(x1, draft.post_attn_norm)
op9  ffn = SwiGLU(z, draft.ffn_{gate,up,down})     # Ffn::Dense path, intermediate 12288
op10 h   = x1 + ffn                                # draft hidden; becomes g for next step (EAGLE recurrence)
op11 fN  = RMSNorm(h, draft.output_norm)
op12 dl  = draft.lm_head @ fN                       # draft_logits [32000]
return (dl_host, h)
```

The exact concat width and whether `fc` maps `3*n_embd→n_embd` (encoder) vs a separate input proj
inside the decoder must be confirmed against the safetensors tensor names at implementation time
(read `model.safetensors` header). The structure above follows llama.cpp `eagle3.cpp` encoder
(`fc`, lines 94-142) + decoder (concat embd+g, single layer, lines 145-323).

Reuse of target hidden states: `g` is `fc @ concat(aux_h[1], aux_h[15], aux_h[28])` of the
last committed token (N1 capture). It is computed ONCE per round and is **immutable across the K
draft steps** — so encoder needs no replay on partial accept (cheaper than MTP's per-step
`h_seed`).

---

## 4. Tree-vs-chain verify

- **v1 = chain (greedy, single path).** Draft K tokens linearly, verify K columns, accept the
  longest matching prefix. This is exactly what `generate_spec` does today (`spec.rs:380-396`).
  It is the GATE path: chain greedy spec is mathematically exact, so the stream is bit-identical
  to `generate`. Ship this first. K=3 (`draft-n-max 3` per the daily serve script / ROADMAP).
- **Tree (deferred, optional v2).** Expand multiple draft branches per position, verify all with a
  tree attention mask. bw24's verify already returns all-T columns (`decode_step_t`,
  `spec.rs:179`) and `fa_prefill_view` takes a causal flag (`spec.rs:316`) — a tree mask would be
  a NEW mask variant (custom additive mask instead of the boolean `causal`). This needs a new
  attention entry (tree mask gen + masked `fa_prefill`), so it is real new kernel work, NOT free.
  Defer until chain acceptance is measured; only build tree if accept-rate analysis shows the
  branch-fan-out wins on this 9B target. **Greedy tree must still be bit-identical** (accept the
  best matching root-to-leaf path), so the GATE is unchanged.

Recommendation: chain for v1 (proven by llama.cpp `eagle3.cpp`, which is T=1 greedy). Revisit tree
only after the chain GATE passes and a benchmark justifies the kernel cost.

---

## 5. Weight / asset requirements

| Asset | 9B target | 27B target |
|---|---|---|
| Draft checkpoint | **PRESENT** `eagle3-qwen35-9b/model.safetensors` (800 MB bf16) | **ABSENT** — no `eagle3-qwen3{5,6}-27b` on disk (confirmed by `find`; only `qwen36-27b-nvfp4-mtp` = MTP, not EAGLE) |
| `d2t` map | in the checkpoint (32000→248320), MANDATORY | n/a |
| aux layer ids | `[1,15,28]` from `eagle_config` | n/a |
| target trunk | `qwen35-9b-*` (already loaded) | `qwen36-27b-*` (already loaded) |

VRAM (RTX 5090 laptop, 24 GB, `ARCHITECTURE.md:3`): draft adds ≈ 0.8 GB resident (quantizable to
~0.3–0.4 GB at W4/W8) + ~0.2 GB aux-hidden snapshot buffers. Target 9B ~5–9 GB + KV. Total well
under 24 GB. **No VRAM blocker.**

sm_120 (`ARCHITECTURE.md:15-18`): draft is dense matmul + norm/rope/attn/ffn — runs on existing
`mma.sync m16n8k16/k32` GEMM + `fa_decode`/`fa_prefill_view` kernels. **No wgmma/tcgen05 needed,
no new kernel for chain v1.**

---

## 6. Validation strategy (the GATE)

1. **Bit-identical gate (mandatory).** For each prompt and K ∈ {1,2,3,4}:
   `generate_spec_eagle3(prompt, max_new, k)` token stream MUST equal `generate(prompt, max_new)`
   (`decode.rs:99`) token-for-token. Any divergence = bug in encode / draft / d2t / verify /
   rollback. This is identical in spirit to the MTP gate and is guaranteed by greedy-exactness.
2. **Acceptance > 0 (mandatory), target > 0.3.** `generate_spec_eagle3` returns
   `(tokens, total_drafted, total_accepted)` like `generate_spec` (`spec.rs:329`). Log
   `total_accepted/total_drafted`. Near-zero acceptance WITH correct output = the draft forward is
   wrong but masked by the bonus token (target self-correction) — gate on BOTH exactness AND
   acceptance to catch this. Healthy EAGLE3 on a matched 9B target: 0.6–0.8.
3. **Cross-check vs llama.cpp EAGLE3** (`eagle3.cpp`) if a reference run is available: first-token
   argmax + greedy continuation parity, reusing the existing oracle infra.

Add a `kernel_check`-style bin entry (`crates/bw24-engine/src/bin/kernel_check.rs`) for the
EAGLE3 gate so it runs in the validation harness.

---

## 7. Honest blocker assessment

- **9B target: NOT asset-gated.** The draft (`eagle3-qwen35-9b`, 800 MB) is on disk and loadable
  via the safetensors seam. EAGLE3 for the 9B target is implementable now; the only real work is
  N1–N7 (Rust, ~350–450 lines, no new CUDA for chain v1). The two sharp edges to confirm at
  implementation time (both surmountable, neither a blocker): (a) exact safetensors tensor names /
  whether `fc` is encoder-only vs an in-decoder input proj; (b) the draft's partial-rotary +
  distinct `rope_theta=1e7` must be plumbed separately from the trunk's RoPE config (do NOT reuse
  `cfg.rope_freq_base`).
- **27B target: asset-gated.** No EAGLE3 draft exists on disk. Until one is obtained/trained,
  **MTP (NextN) remains the shipped spec for 27B** — it ships built-in via the dropped trunk block
  (`qwen36-27b-nvfp4-mtp`, `hybrid.rs:171-190`) and is already validated (greedy, batched
  partial-accept replay, `spec.rs:329`).
- **Coexistence (recommended).** Keep BOTH paths and select at runtime: if an EAGLE3 draft asset
  is present for the loaded target → EAGLE3; else if `nextn_predict_layers>0` and the NextN head
  loaded (`hybrid.rs:174`) → MTP; else plain greedy. This matches the daily serve script's
  `--spec-type` toggle (ROADMAP). EAGLE3's reuse of the snapshot/rollback/verify plumbing means
  the two share everything except draft generation.

---

## 8. Implementation order

1. N1 aux-hidden capture (`decode_step_aux` + verify variant) — verify against a trunk run that
   the captured `[1,15,28]` residuals match a known reference.
2. N2 `Eagle3Draft` struct + safetensors loader (confirm tensor names, d2t, rope params).
3. N3 encoder + N4 draft decoder + N5 d2t + N7 draft embd.
4. N6 `generate_spec_eagle3` (clone `generate_spec`, swap draft loop only).
5. GATE: bit-identical vs `generate` for K∈{1,2,3,4} + acceptance log.
6. (Optional v2) tree mask + masked `fa_prefill` if benchmark justifies.

Key landmarks: draft loop to swap `spec.rs:365-376`; verify reuse `spec.rs:380`; accept/commit
`spec.rs:386-407`; rollback reuse `spec.rs:410-432`; snapshot `cache.rs:98`; aux capture insertion
`decode.rs:50-83` and `spec.rs:200-244`; struct template `hybrid.rs:120-130`; draft-forward
template `spec.rs:46-122`; scratch-KV template `spec.rs:17-38`.
