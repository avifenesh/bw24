# MLA bring-up design — GLM-5.2 on memra (increment 1, 2026-08-01)

Ground truth and citations: `RECEIPTS.md` in this directory (all facts fetched live 2026-08-01;
config pinned as `glm-5.2-config.pinned.json`, llama.cpp reference pinned as
`llamacpp-glm-dsa.cpp.pinned`). CPU reference implementation + proof tests:
`crates/memra-engine/src/mla.rs` (this lane's increment-1 code deliverable).

Scope of this document: the decode-path math, the exact deltas to memra, and the increment
ladder to a running GLM-5.2. No CUDA kernels in this increment.

---

## 1. The math

### 1.1 Symbols (GLM-5.2 values)

| Sym | Meaning | GLM-5.2 |
|---|---|---|
| H | hidden size | 6144 |
| N | attention heads | 64 |
| P | qk nope head dim | 192 |
| R | qk rope head dim | 64 |
| Dqk | P + R | 256 |
| V | v head dim | 256 |
| Lq | q lora rank | 2048 |
| Lkv | kv lora rank | 512 |
| C | latent cache row = Lkv + R | **576** |
| T | context length (tokens in cache) | ≤ 1,048,576 |

Weights per layer (HF name → role):

```
W_DQ  = q_a_proj            [H  → Lq]        + RMSNorm(q_a_layernorm)
W_UQ  = q_b_proj            [Lq → N·Dqk]       (per head: [nope P | rope R])
W_DKV = kv_a_proj_with_mqa  [H  → Lkv + R]     (rows: [c_kv Lkv | k_pe R]; c_kv RMSNorm'd)
W_UK  = kv_b_proj nope part [Lkv → N·P]        (per head Lkv×P)
W_UV  = kv_b_proj v part    [Lkv → N·V]        (per head Lkv×V)
W_O   = o_proj              [N·V → H]
```

Per token, the *only* thing that enters the KV cache is one 576-vector:
`c̃(t) = [ rmsnorm(c_kv(t)) (512) | rope(k_pe(t)) (64) ]` — shared by all 64 heads (MQA).

### 1.2 Naive form (decompress-then-attend, "MHA form")

For every cached token t and head h:

```
k_h(t) = [ W_UKᵀ_h · c_kv(t) (P) | k_pe(t) (R) ]     — Dqk = 256 per head
v_h(t) =   W_UVᵀ_h · c_kv(t)                          — V = 256 per head
o_h    =   softmax_t( q_h·k_h(t) / √Dqk ) · v_h(t)
out    =   W_O · concat_h(o_h)
```

Correct, and compute-friendly when amortized over many query tokens (prefill). At decode it is
catastrophic: either re-decompress the whole context every step
(64·2·(192+256)·512·T ≈ **29.4 MFLOP·T per layer per token**) or cache decompressed K/V
(64·(256+256) = 32,768 elems/token/layer = 2.56 MB/token f16 over 78 layers — dead on arrival
at 1M context).

### 1.3 Absorbed form (decode form; llama.cpp glm-dsa.cpp, vLLM "forward_mqa")

Fold W_UK into the query instead of into the keys — legal because
`q_h · (W_UKᵀ c_kv) = (W_UK q_h) · c_kv` (associativity), and fold W_UV *after* the
softmax·V product — legal because softmax·V is linear in V:

```
q̃_h    = [ W_UK_h · q_nope_h (Lkv=512) | q_pe_h (R=64) ]      — 576 per head
k̃(t)   = c̃(t)                                                 — 576, ONE per token (MQA)
score  = q̃_h · k̃(t) / √Dqk        ← scale is √256 = 16, NOT √576 (llama.cpp + paper)
õ_h    = softmax_t(score) · c_kv(t)                            — 512 (latent-space output)
o_h    = W_UVᵀ_h · õ_h                                          — 256
out    = W_O · concat_h(o_h)
```

Identical output to §1.2 (proved to f32 tolerance in `mla.rs::tests`, decode t=1 and causal
prefill, synthetic dims + GLM-ratio dims). Decode attends **in latent space**: the KV cache is
576 elems/token/layer, all 64 heads stream the same rows.

Softmax scale note: no yarn (`rope_type: "default"`) so llama.cpp's mscale factor is exactly 1;
scale = 1/√256 = 0.0625. If a future GLM ships yarn, the mscale² correction re-enters (see
glm-dsa.cpp `kq_scale`).

### 1.4 RoPE — the trap

GLM-5.2 is `rope_interleave: true` → llama.cpp `LLAMA_ROPE_TYPE_NORM` (adjacent pairs
(x[2i], x[2i+1]) share angle θ_i). memra only implements NEOX pairing (x[j], x[j+d/2]).
Applied to the *last* 64 dims of each 256-d q head and to the 64-d shared k_pe; θ base 8e6,
no scaling. Two implementation options:

1. New `rope_norm` kernel (adjacent-pair twin of `rope_neox_f32`).
2. **Load-time permutation** (zero new kernels): permute the R rope rows of `W_UQ` (per head) and
   of `W_DKV` with π(2j)=j, π(2j+1)=j+R/2, then run the existing `rope_neox` unchanged. Valid
   because q_pe and k_pe meet only in dot products, which are invariant under a common
   permutation, and π maps interleaved pairing onto half-offset pairing exactly.
   `mla.rs::tests::rope_norm_equals_permuted_neox` proves the equivalence numerically.

Option 2 is the increment-4 default (reuses fused rope+append kernels); option 1 is the
fallback if any consumer of raw k_pe order appears (none known — the indexer has the same
dot-product-only property).

### 1.5 DSA on top (GLM-5.2-specific; NOT in increment 1-5 scope)

Per "full" indexer layer (21 of 78: layers 0,1,2 then every 4th from 6 to 74):

```
qᵢ (32×128) = indexer.wq_b · q_c        (from the SAME q latent, post q_a_layernorm)
kᵢ (128)    = k_norm(indexer.wk · h)    (one per token → own tiny cache, 128/token/full-layer)
both: split [rope 64 | nope 64], rope(NORM), concat(pe,nope), Hadamard rotate
w  (32)     = indexer.weights_proj · h,  scaled 1/√(128·32)
score(t)    = Σ_h w_h · ReLU(qᵢ_h · kᵢ(t)),  masked, top-k 2048 indices
```

The main MLA attention then runs over only the 2048 selected tokens. "Shared" layers (57 of 78)
reuse the previous full layer's indices verbatim (IndexShare / IndexCache, arXiv 2603.12201).
MTP/NextN layer runs dense MLA, no indexer. Key exactness property for gating: **for T ≤ 2048
the top-k selects every token → DSA output must be bit-identical to the dense arm.**

---

## 2. FLOPs / bytes at GLM-5.2 dims vs memra's current GQA path

### 2.1 Per-layer per-decode-token cost (absorbed form)

Fixed (context-independent) matvec FLOPs:

| Stage | FLOPs |
|---|---|
| W_DQ (6144→2048) | 25.2 M |
| W_UQ (2048→16384) | 67.1 M |
| W_DKV (6144→576) | 7.1 M |
| absorb q_nope (64 × 192×512) | 12.6 M |
| W_UV out (64 × 512×256) | 16.8 M |
| W_O (16384→6144) | 201.3 M |
| **Σ fixed** | **330 M** (165.0 M weight elems/layer) |
| + indexer (21 layers only) | ≈ 19.8 M fixed + 8.2 K·T scan |

Context-dependent attention FLOPs: scores 64·2·576·T + AV 64·2·512·T = **139.3 K·T** per layer
(dense) — capped at T_eff = 2048 by DSA → ≤ 285 M. Dense at 128K would be 18.3 G/layer; DSA's
1.5-2x total-model claim (paper §2.1.1) is consistent.

### 2.2 Cache footprint (the reason MLA exists)

| | per token, 78 layers | 1M-token session |
|---|---|---|
| MLA latent f16 (576/tok/layer) | 87.8 KB | 89.9 GB |
| MLA latent q8_0-class (612 B/tok/layer) | 46.6 KB | 47.7 GB |
| + indexer keys, 21 full layers, f16 | 5.3 KB | 5.4 GB |
| hypothetical GQA-8 d128 same depth, f16 | 312 KB | 319.5 GB |

3.6× smaller than a GQA-8 equivalent before quantization; with memra's quantized-KV discipline a
full 1M context is a ~50 GB object — serveable on the 8×H100 node with DP-attention.

### 2.3 Where the kernel shape breaks vs memra's fa_decode

memra decode attention today (`fa_decode_vec_q`, flash_attn.cu:5033): grid `(n_head_kv, splits)`,
block `(32, GQA_RATIO)`, per-lane K/V dequant, head_dim ≤ 256 (dpl16 twin = 512), dk == dv,
NEOX rope, K plane + V plane per layer.

MLA decode is a different animal:

- n_head_kv = 1, GQA_RATIO = 64 → the vec-lane shape degenerates; one KV stream feeds 64 heads.
- dk = 576 (> 512 dpl16 ceiling), dv = 512, **dk ≠ dv**, and V is a *prefix view* of K.
- Arithmetic intensity flips: MLA decode attention ≈ 121 FLOP/byte-of-KV (139.3K·T FLOPs over
  1152·T f16 bytes) vs ≈ 12 for GQA-8 d128 — scores are a (64×576)·(576×T) GEMM per token,
  i.e. a **tensor-core shape**, not a vec-dot shape. The paper's MLA-256 head-count reduction
  exists precisely because MLA decode is compute-heavy (§2.1 "576-dimensional dot product").
- The fixed 330 MF of per-layer projections (incl. two *per-head batched* GEMMs: absorb and W_UV)
  are new structures — memra's current arm has no per-head weight GEMMs inside attention.

So the fused kernel (increment 5) is a small GEMM-attention (FlashMLA-style), not a fa_decode
variant; the scalar/naive arm (increment 4) extends `fa_decode_f32`-class generic kernels with
split dk/dv first.

Quantization note: 576 = 18×32 and the V view boundary at 512 = 16×32 falls **on a quant-block
boundary** — one quantized latent plane serves both K̃ (18 blocks) and V (first 16 blocks) with
zero duplication. `head_dim % 32 == 0` (memra-kv assert) holds.

---

## 3. What changes in memra

### 3.1 GGUF tensor mapping (their name → memra slot)

Upstream GGUFs exist (unsloth/GLM-5.2-GGUF; community REAP variants — see RECEIPTS §5), arch
string **`glm-dsa`**. memra's gguf crate parses names generically (`GgufFile::find`), so only the
loader mapping + config keys are new. Proposed `MlaAttnLayer` (new struct next to
`FullAttnLayer`, hybrid.rs):

| GGUF tensor (llama.cpp glm-dsa) | HF source | logical shape | memra slot |
|---|---|---|---|
| `blk.N.attn_q_a.weight` | `self_attn.q_a_proj` | H×Lq | `wq_a` |
| `blk.N.attn_q_a_norm.weight` | `self_attn.q_a_layernorm` | Lq | `q_a_norm` |
| `blk.N.attn_q_b.weight` | `self_attn.q_b_proj` | Lq×(N·Dqk) | `wq_b` |
| `blk.N.attn_kv_a_mqa.weight` | `self_attn.kv_a_proj_with_mqa` | H×(Lkv+R) | `wkv_a` |
| `blk.N.attn_kv_a_norm.weight` | `self_attn.kv_a_layernorm` | Lkv | `kv_a_norm` |
| `blk.N.attn_k_b.weight` | kv_b_proj nope slice, **transposed at conversion** | N×(Lkv×P) | `wk_b` (absorb GEMM) |
| `blk.N.attn_v_b.weight` | kv_b_proj v slice | N×(V×Lkv) | `wv_b` (output decompress) |
| `blk.N.attn_kv_b.weight` | `self_attn.kv_b_proj` (unsplit; optional) | Lkv×(N·(P+V)) | unused v1 (MHA-prefill later) |
| `blk.N.attn_output.weight` | `self_attn.o_proj` | (N·V)×H | `wo` |
| `blk.N.attn_norm.weight` / `ffn_norm` | input/post layernorms | H | existing slots |
| `blk.N.indexer.attn_q_b.weight` | `self_attn.indexer.wq_b` | Lq×(32·128) | `idx_wq_b` (inc-6) |
| `blk.N.indexer.attn_k.weight` | `self_attn.indexer.wk` | H×128 | `idx_wk` (inc-6) |
| `blk.N.indexer.k_norm.{weight,bias}` | `self_attn.indexer.k_norm` | 128 | `idx_k_norm` (inc-6) |
| `blk.N.indexer.proj.weight` | `self_attn.indexer.weights_proj` | H×32 | `idx_wproj` (inc-6) |
| `blk.78.nextn.*` | MTP block | — | MTP arm (inc-7; existing MTP precedent) |
| FFN/MoE `blk.N.ffn_*` | deepseek-style MoE | — | existing MoE slots (sigmoid router + `noaux_tc` bias `ffn_exp_probs_b` — Hy3-class machinery) |

GGUF metadata → `MlaConfig` (new `Option<MlaConfig>` on `ModelConfig`, gemma4-pattern):

| key | value | field |
|---|---|---|
| `glm-dsa.attention.q_lora_rank` | 2048 | `q_lora_rank` |
| `glm-dsa.attention.kv_lora_rank` | 512 | `kv_lora_rank` |
| `glm-dsa.attention.key_length` | **576** (cache K row) | cross-check only |
| `glm-dsa.attention.value_length` | **512** (cache V view) | cross-check only |
| `glm-dsa.attention.key_length_mla` | 256 | `qk_head_dim` |
| `glm-dsa.attention.value_length_mla` | 256 | `v_head_dim` |
| `glm-dsa.rope.dimension_count` | 64 | `qk_rope_head_dim` (nope = key_length_mla − this) |
| `glm-dsa.attention.indexer.{head_count,key_length,top_k,types}` | 32 / 128 / 2048 / [bool;78] | `DsaConfig` (inc-6) |
| `glm-dsa.nextn_predict_layers` | 1 | existing MTP plumbing |

Also new: `Arch::GlmDsa` variant + parse arm (`crates/memra-gguf/src/config.rs:24-41`),
`is_hybrid()` = true so `HybridModel` loads it; chat template (`[gMASK]<sop>` family, thinking
default-on with `reasoning_effort` max/high — pinned in RECEIPTS S3 HF API dump).

### 3.2 Cache (`crates/memra-kv`)

Reuse `KvLayer` with MLA geometry per layer (the gemma4 per-layer override at
memra-kv lib.rs:206-259 is the precedent):

- `kv_dim_k` = 576, `n_head_kv` = 1 semantics; **no V plane** — V is the first 512 elems
  (16 quant blocks) of each K row. Concretely: `v` slice empty, `v_tok_bytes = 0`, and the MLA
  attention arm reads the K plane for both scores and AV. Existing appenders write K-only.
- Append: one fused kernel eventually (rmsnorm(c_kv) ‖ rope(k_pe) → quantize row), but
  increment 4 uses the existing `append_kv_quantized` split into a K-row append.
- Indexer cache (inc-6): second tiny plane, 128 elems/token, allocated only on the 21 full
  layers (`Vec<Option<...>>` pattern already there).
- Snapshot/rollback for spec decode: unchanged (row-copy semantics carry).

### 3.3 Attention arm (`crates/memra-engine`)

New `Mixer::Mla(MlaAttnLayer)` arm slotted exactly like gemma4's parallel family:
prefill in `hybrid_forward.rs` (branch at :261/:279), eager decode in `decode.rs`
(`decode_step_h` loop :613-663), dc/graph twin (:1563 pattern), batched tick later. v1 decode
sequence per layer:

```
h → rmsnorm → wq_a → rms(q_a_norm) → wq_b                       (existing matmul arms)
  → per-head absorb: q̃_nope = wk_b ⊛ q_nope                     (new batched-GEMM call, 64×[192×512])
  → rope_neox(q_pe, k_pe) via permuted weights (§1.4)            (existing kernel)
  → append [rms(c_kv) | k_pe] row                                (existing appender, K-only)
  → MLA attention over latent plane (dk 576 / dv 512, MQA)       (new kernel, scalar first)
  → per-head decompress: o = wv_b ⊛ õ                            (new batched-GEMM call)
  → wo                                                            (existing)
```

MoE/FFN, router (sigmoid + expert bias), norms: existing Hy3-class machinery, no changes.

### 3.4 CPU reference

`crates/memra-engine/src/mla.rs` (this increment): f32 CPU implementation of §1.2 and §1.3 with
interleaved-rope helper and the NEOX-permutation equivalence, tested naive≡absorbed on random
inputs (t=1 decode + causal prefill, multiple shapes incl. GLM-5.2 ratios and full GLM-5.2 dims).
This module is the oracle for every later increment's maxdiff gates and stays permanently as the
`kernel-check`-style reference for the MLA family.

---

## 4. Increment ladder to a running model

Effort class honest total: 3-6 weeks (new kernel family), per the corrected model-selection
synthesis. Increments 1-3 are CPU/code-lane; 4+ needs GPUs.

| # | Deliverable | Exactness gate |
|---|---|---|
| 1 | **(this)** ground truth + CPU f32 reference (naive + absorbed + rope) | `cargo test -p memra-engine --lib mla` green: naive≡absorbed ≤ f32 tol across shapes; rope NORM≡permuted-NEOX |
| 2 | GGUF plumbing: `Arch::GlmDsa` (+`deepseek2` for the dev vehicle), `MlaConfig` parse, `MlaAttnLayer` loader, synthetic micro-GGUF fixture; pick + fetch the small MLA dev model (DeepSeek-V2-Lite 15.7B `deepseek2` GGUF — abundant upstream; caveat: yarn rope, memra has none → run at ctx ≤ original 4k window where yarn ≈ mscale-only, or gate GLM-4.7-Flash as alternative after verifying its size/arch — RECEIPTS §7.5) | config-parse unit tests vs pinned metadata; tensor-presence audit vs `gguf-dump` of the real file; NO weights on the 5090 rig beyond the dev model |
| 3 | CPU-reference end-to-end forward (mla.rs math wired to dequantized weights) on the dev model, one layer then full stack | layer-0 activation maxdiff vs llama.cpp `--dump-tensors`-class reference < 1e-3 (f32 vs quant source); then greedy `run-gen` argmax MATCH vs llama.cpp same-GGUF same-prompt |
| 4 | CUDA naive arm: existing matmul/rope/append kernels + new scalar MLA decode kernel (generic dk/dv split of `fa_decode_f32` class) + latent-plane cache | per-layer maxdiff vs mla.rs ≤ existing kernel-check thresholds; `run-gen` argmax MATCH on dev model; kernel-check pins added for every new kernel |
| 5 | Fused decode kernel (GEMM-shaped: 64 heads × 576 × T tiles, split-K, in-flight latent dequant; sm_120a first, sm_90a port under `memra_hopper_mma` for the H100 box) + absorbed prefill; then GLM-5.2 conversion + load on the 8×H100 box (weights to `/data`, ~410-420 GB Q4 class) | bit-parity vs increment-4 arm; before/after per `research/benchmarks.md`, interleaved ×5; GLM-5.2 `run-gen` argmax + `run-spec` K=1..8 on-box |
| 6 | DSA: indexer arm + indexer cache + top-k + gathered sparse attention + IndexShare layer wiring | **T ≤ 2048 ⇒ bit-identical to dense arm** (top-k total); long-ctx quality spot-gates vs llama.cpp; perf at 32K/128K |
| 7 | Graph capture + batched tick + MTP head (dense-MLA NextN layer, 5-draft serving config) + hand-off to M2/M3 (PP8/EP8) serving lanes | graph-decode/graph-session batteries; spec acceptance vs llama.cpp MTP; full validate battery ALL GREEN before merge/tag |

Dependencies: 1→2→3→4→5 strictly serial; 6 needs 4 (dense arm as its oracle); 7 needs 5.
GLM-5.2 real-weight work (5+) runs on the 8×H100 capacity-block box, not this rig.

## 5. Answers to the standing questions

- **Do GLM-5.2 GGUFs exist upstream?** Yes — unsloth/GLM-5.2-GGUF plus community REAP-pruned
  variants; llama.cpp arch `glm-dsa`; early loaders broke on GLM-5.2's partial indexer tensors,
  fixed via `indexer.types` metadata (RECEIPTS §5). memra's gguf crate parses the files as-is
  (generic name lookup); the work is the §3.1 mapping + config, not the parser.
- **Biggest divergence memra ↔ MLA:** memra's entire attention plumbing assumes per-layer K+V
  planes consumed by uniform-GQA vec-lane kernels (dk=dv ≤ 256/512, NEOX rope, no per-head weight
  GEMMs inside attention). MLA needs a single 576-wide MQA latent plane (V = prefix view of K),
  dk≠dv, two per-head batched GEMMs inside the arm, interleaved partial rope, and a
  tensor-core-shaped decode kernel — plus, for GLM-5.2 specifically, a second sparse-selection
  stage (DSA) with its own cache and cross-layer index reuse.
