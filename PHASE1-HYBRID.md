# PHASE 1 — Correct logit-matched Qwen3.5-9B hybrid forward in Rust + cudarc on sm_120

Goal of this phase: a **bit-faithful, logit-matched** forward pass for the daily-target hybrid
(Qwen3.5-9B judge first, then 27B dense) running on the RTX 5090 Laptop (sm_120a fatbin via
cudarc 0.19.8). **Correctness first, speed second.** Every kernel has a CPU/llama.cpp oracle and a
numeric gate before any sm_120-fast variant is allowed to replace it.

Ground truth lives at `/home/avifenesh/projects/llama.cpp` (the qwen35 arch is fully implemented
there) and at the FLA / vLLM references. **Copy, do not invent.** This doc gives exact files, line
ranges, shapes, and the order to build them.

The repo already has the spine pieces in place — build on them, do not duplicate:
- `crates/bw24-gguf/src/lib.rs` — GGUF v3 mmap reader (`GgufFile::open`, `find`, `tensor_data`, `meta_arch`), `GgmlType` block sizes incl. MXFP4(39)/NVFP4(40).
- `crates/bw24-gguf/src/config.rs` — `ModelConfig::from_gguf`, `Arch`, `LayerKind`, `SsmConfig`, `MoeConfig`, `layer_kind(il)`, `n_full_attn_layers()`. **Already classifies layers correctly** (`(il+1)%full_attention_interval==0` ⇒ FullAttention).
- `crates/bw24-gguf/src/dequant.rs` — CPU dequant oracle: `fp16_to_f32`, `bf16_to_f32`, `dequantize(ty, raw, n)` for F32/F16/BF16/Q8_0/Q4_K/Q6_K (extend as needed). This is the correctness oracle for GPU dequant.
- `crates/bw24-gguf/src/bin/inspect.rs` — `gguf-inspect` (built at `target/release/gguf-inspect`), and a `rope_dump` binary already exist.
- `crates/bw24-probe` — the proven sm_120a nvcc→fatbin + cudarc launch spine (FP16/FP8/block-FP4 mma.sync verified). Reuse its `build.rs` nvcc pattern for all `.cu` kernels.

Verified daily-target geometry (Qwen3.5-9B, Q8_0 GGUF `/home/avifenesh/ai-ml/models/qwen3.5-9b-judge-q8_0.gguf`):
n_embd=4096, n_layer=32 (8 full-attn at il=3,7,11,15,19,23,27,31; 24 linear), n_head=16, n_head_kv=4,
head_dim=256, n_ff=12288, n_vocab=248320, rms_eps=1e-6, rope_freq_base=1e7, n_rot=64,
rope_sections=[11,11,10,0], rope_type=IMROPE(40), full_attention_interval=4, +1 MTP block (il=32).
SSM: d_conv=4, d_inner=4096, d_state=128, dt_rank(=num_v_heads)=32, n_group(=num_k_heads)=16,
head_k_dim=head_v_dim=128, key_dim=2048, value_dim=4096, conv_dim=8192.

---

## 1. Rust module layout

New crate `crates/bw24-engine` (add to workspace `members`). Depends on `bw24-gguf`, `cudarc = "0.19.8"`.
Kernels are `.cu` compiled by `build.rs` (clone the `bw24-probe` nvcc→`-gencode arch=compute_120a,code=sm_120a`
fatbin pattern) and loaded as cudarc modules.

```
crates/bw24-engine/
  build.rs                      # nvcc each .cu -> sm_120a fatbin, embed via include_bytes!/OUT_DIR (copy bw24-probe/build.rs)
  cu/                           # raw CUDA C++ kernels (ported, see §4/§5)
    rmsnorm.cu                  # rms_norm_f32, l2_norm_f32 (norm.cu port)
    rope_neox.cu                # partial NEOX rope (IMRoPE-collapsed; rope.cu port)
    ssm_conv.cu                 # depthwise causal conv1d + SiLU (ssm-conv.cu port)
    gated_delta_net.cu          # GDN recurrent scan <S_v=128,KDA=false> (gated_delta_net.cu port)
    softmax.cu                  # masked row softmax f32 (correctness-oracle SDPA)
    elementwise.cu             # silu, sigmoid, mul, add, gate-mul, copy/view-extract
    dequant.cu                  # Q8_0/Q4_K/Q6_K/F16 -> f32 (or -> f16) on device (matches dequant.rs)
  src/
    lib.rs                      # re-exports, error type, sm120 cap assertions
    device.rs                  # CudaDevice/stream/module handles, kernel registry, launch helpers
    tensors.rs                 # DeviceTensor { ptr, shape, dtype }, upload/download, ggml-name -> tensor map
    weights.rs                 # Weights: load every blk.* tensor from GgufFile, dequant policy per tensor
    model.rs                   # Model { cfg: ModelConfig, embd, output_norm, output, layers: Vec<Layer>, mtp: Option<MtpBlock> }
    layer.rs                   # enum LayerKind { Full(FullAttnW), Linear(LinearW) } + shared { attn_norm, attn_post_norm, ffn(/moe) }
    cache.rs                   # Cache { kv: Vec<Option<KvLayer>>, recur: Vec<Option<RecurLayer>> } (see §3)
    gemm.rs                    # matmul facade: Stage-1 = cuBLASLt (dequant->f16/f32 GEMM); Stage-2 = MMQ/Marlin/NVFP4
    ops.rs                     # thin Rust launchers for each .cu kernel (rmsnorm, rope, conv, gdn, softmax, silu...)
    forward.rs                 # forward(), forward_full_attn(), forward_linear_attn(), forward_ffn(), lm_head()
    forward_mtp.rs             # forward_mtp(token, h_seed, &mut Cache) (NextN draft head)
    sdpa.rs                    # Stage-1 naive SDPA (KQ/softmax/KQV); Stage-2 swaps in fa-2 mma.sync
    bin/
      run_dense.rs             # MILESTONE 0: pure-dense Qwen3-1.7B forward (spine proof)
      run_hybrid.rs            # MILESTONE >=4: qwen35-9B forward, prints top-k logits
      logit_check.rs           # MILESTONE harness: compare argmax/top-32 vs llama.cpp dump (§6)
```

Design rules:
- **Host-side orchestration.** `forward.rs` loops `0..cfg.n_layer` on the host and launches one kernel per op (no fused megakernel in Phase 1). This is slower but trivially debuggable and matches llama.cpp's graph node-for-node.
- **All recurrent/elementwise math in f32** in Phase 1 (GDN, conv, norms, rope, softmax). Only the GEMM operands get quantized. This keeps tolerances tight (§6).
- **One `DeviceTensor` type** carrying `shape: Vec<usize>` (ne[0] fastest, ggml convention) + dtype. Strides mirror ggml so kernel offset math copies 1:1.
- Reuse `ModelConfig::layer_kind(il)` — do **not** re-derive the interval anywhere else.

---

## 2. Per-layer-type forward graphs (exact, with shapes)

Top-level forward (ground truth: `llama.cpp/src/models/qwen35.cpp:136-228`):

```
x = tok_embd[token]                                   # [n_embd=4096, T]
for il in 0..n_layer:                                 # 0..32 ; MTP block il=32 loaded but NOT run
    inpSA = x
    h    = RMSNorm(x, attn_norm[il], eps=1e-6)         # qwen35.cpp:163
    if layer_kind(il)==Linear:  y = LINEAR_ATTN(h, il) # qwen35.cpp:171
    else:                       y = FULL_ATTN(h, il)   # qwen35.cpp:174
    x        = y + inpSA                               # residual #1 (attn only)  qwen35.cpp:183
    ffn_res  = x
    z        = RMSNorm(x, attn_post_norm[il], eps=1e-6)# PRE-FFN norm (named "post_attention_norm") qwen35.cpp:190
    f        = SwiGLU_FFN(z, il)                        # qwen35.cpp:194 / build_layer_ffn:472-485
    x        = f + ffn_res                              # residual #2 (ffn only)   qwen35.cpp:198
h = RMSNorm(x, output_norm, eps=1e-6)                  # == t_h_nextn (MTP seed)   qwen35.cpp:209-212
h = h[:, last_token]                                   # get_rows(out_ids)         qwen35.cpp:214-216
logits = output @ h                                    # [n_vocab=248320, 1]; output tied to tok_embd if absent  qwen35.cpp:222
```

### 2a. FULL_ATTN(h, il)  — ground truth `qwen35.cpp:257-336`
Geometry: n_head=16, n_head_kv=4 (GQA 4:1), head_dim=256, n_rot=64, kq_scale=1/sqrt(256)=0.0625.

```
Qcur_full = wq @ h                       # [8192, T]  (8192 = head_dim*2 * n_head = 256*2*16)  :269
# FUSED per-head [q256|gate256], stride 512 per head:
Q   = view(Qcur_full, [256,16,T], nb1=512*elsz, nb2=512*16*elsz, off=0)        # :272 query
gate= view(Qcur_full, [256,16,T], nb1=512*elsz, nb2=512*16*elsz, off=256*elsz) # :292 output-gate
Q   = RMSNorm_perhead(Q, attn_q_norm[256], eps=1e-6)                            # :278  (weight already +1-baked in GGUF)
K   = reshape(wk @ h, [256,4,T]); K = RMSNorm_perhead(K, attn_k_norm[256])      # :288-290
V   = reshape(wv @ h, [256,4,T])                                                # NOT normed, NOT roped  :299
Q   = RoPE_partial(Q, pos, n_rot=64, base=1e7)   # only dims 0..63 rotated; 64..255 copied  :302
K   = RoPE_partial(K, pos, n_rot=64, base=1e7)                                  # :308
attn= SDPA(Q,K,V, scale=0.0625, causal, GQA kv_head=q_head/4) -> [256,16,T]     # :321 build_attn
attn= attn * sigmoid(gate)                                                       # :326-329 (the output gate)
out = wo @ reshape(attn,[4096,T])        # wo [4096,4096] -> [4096,T]            # :332
```
RoPE is IMROPE (mode 40) but **text-only collapses to NEOX partial** (all 4 pos channels share one position;
`llama-graph.cpp:144-160`). theta_scale = 1e7^(-2/64). For j in 0..31: out[j]=x[j]cos - x[j+32]sin;
out[j+32]=x[j]sin + x[j+32]cos. dims 64..255 untouched.

### 2b. LINEAR_ATTN(h, il)  — ground truth `qwen35.cpp:338-470`
Geometry: d_inner=4096, S=d_state=128, num_k_heads=16, num_v_heads=32, head_k=head_v=128,
key_dim=2048, value_dim=4096, conv_dim=8192, scale=1/sqrt(128)=0.0883883.

```
qkv_mixed = wqkv @ h           # [8192, T]   (q|k|v packed: q[0:2048] k[2048:4096] v[4096:8192])  build_qkvz :236
z         = wqkv_gate @ h      # [4096, T]   output gate (NOT conv'd)                              :240
beta  = sigmoid(ssm_beta  @ h);                reshape [1,32,T]                                    :361-365
alpha = ssm_alpha @ h
g_log = ssm_a * softplus(alpha + ssm_dt.bias); reshape [1,32,T]   # ssm_a holds -exp(A_log) (pre-negated) :372-379
conv_in = concat(conv_state[3 cols], qkv_mixed) along time   # build_conv_state (delta-net-base.cpp:449-525)
conv    = SiLU( depthwise_causal_conv1d(conv_in, ssm_conv1d[4,8192]) )   # :394-398
q_c = view conv[0:2048]   -> [128,16,T]                                  # :407
k_c = view conv[2048:4096]-> [128,16,T]                                  # :413
v_c = view conv[4096:8192]-> [128,32,T]                                  # :419
q_c = L2norm(q_c, eps=1e-6); k_c = L2norm(k_c, eps=1e-6)                 # :431-432  (V not normed)
# fused GDN op maps 16 k-heads -> 32 v-heads internally (no explicit repeat); if manual, repeat-interleave x2
o   = GDN_scan(q_c, k_c, v_c, g_log, beta, state, scale=0.0883883)       # :450  recurrence below
out = RMSNorm(o, ssm_norm[128]) * SiLU(z)                                # build_norm_gated :246-255 / :456
cur = ssm_out @ reshape(out,[4096,T])                                    # ssm_out[4096,4096] :463
```
GDN recurrence (per v-head h, per token t; state S∈R^{128×128}; gated_delta_net.cu:84-111):
```
g_t = exp(g_log_t)                       # kernel does the exp; pass g_log RAW
kv[col]    = sum_i S[i][col]*k~[i]
delta[col] = (v[col] - g_t*kv[col]) * beta_t
S[i][col]  = g_t*S[i][col] + k~[i]*delta[col]
o[col]     = (sum_i S[i][col]*q~[i]) * scale
```
State stored **transposed**: M[col][i]=S[i][col] (row `col` contiguous; gated_delta_net.cu:54).

### 2c. SwiGLU FFN  — `qwen35.cpp:472-485`  (dense 9B/27B; `ffn_gate_inp==nullptr` asserted)
```
f = ffn_down @ ( SiLU(ffn_gate @ z) * (ffn_up @ z) )   # ffn_gate/up [4096,12288], ffn_down [12288,4096]
```
(qwen35moe replaces this with softmax-gated experts + sigmoid-gated shared expert; out of Phase-1 scope — dense only.)

### 2d. MTP / NextN  — `qwen35.cpp:488-644` (Phase-1 optional; same kernels)
```
tok = tok_embd[token]                         # nextn.embed_tokens absent -> fall back to tok_embd  :521
e   = RMSNorm(tok, nextn.enorm); hn = RMSNorm(h_seed, nextn.hnorm)   # h_seed = t_h_nextn of prev pos
cur = nextn.eh_proj @ concat([e; hn])         # eh_proj [2*n_embd, n_embd]  :542-552
# then a normal FULL_ATTN block (same wq/wk/wv/wo/q_norm/k_norm/IMRoPE/gate) + residual + post_norm + FFN + residual
cur = RMSNorm(cur, nextn.shared_head_norm or output_norm)
logits_mtp = (nextn.shared_head_head or output) @ cur[:, last]   # both absent in 9B/27B -> tie to output  :636
```

---

## 3. Dual cache design (the 24GB / long-context payoff)

Two independent caches, sized completely differently — this asymmetry is the whole point of the hybrid.

`Cache { kv: Vec<Option<KvLayer>>, recur: Vec<Option<RecurLayer>> }`, length `n_layer`, only the
relevant slot populated per layer (use `layer_kind(il)`).

**(A) Growing KV cache — ONLY the 8 full-attn layers (+1 MTP).** Per token per layer:
K = n_embd_k_gqa = head_dim*n_head_kv = 256*4 = 1024 floats; V same = 1024.
```
KvLayer { k: [1024, n_ctx], v: [1024, n_ctx], len: usize }   # store fp16 in Phase 1.5+, f32 in Phase 1
```
KV bytes/token (9B, 8 layers, fp16) = 8 * 2 * 1024 * 2 = **32 KiB/token**.
→ 262144-ctx ≈ **8 GiB**. This is the *only* part that grows with context.

**(B) Fixed recurrent state — the 24 linear layers, context-INDEPENDENT.** Per layer per sequence
(mirrors `llama-hparams.cpp` n_embd_r:183-205, n_embd_s:207-222):
```
RecurLayer {
  conv_state: [3, 8192],     # n_embd_r = (d_conv-1)*conv_dim = 3*8192 = 24576 floats; oldest->newest cols
  ssm_state:  [128,128,32],  # n_embd_s = d_state*d_inner = 128*4096 = 524288 floats; transposed M[col][i]
}
```
Total recurrent state (9B) = 24 * (24576 + 524288) * 4B ≈ **52.7 MiB / sequence, CONSTANT for any context**.

**Payoff.** For the dense 27B/9B "judge" daily use at long context, ~75% of layers carry zero per-token
memory growth; only the 8/16 full-attn layers grow. A 24GB card holds the 9B weights (Q8_0 ≈ 9.5GB) +
the full recurrent state (≈53MB) + a very long KV cache, where a plain transformer of the same depth
would pay 4× the KV. This is why the hybrid is the daily target.

Cache invariants:
- Decode (T=1): conv_state is a 3-col shift register (shift in newest col); ssm_state overwritten in place each step. Both are pure register/elementwise work — cheap.
- Prefill (T>1): conv writes back last 3 input cols; GDN scan writes final S. **Prefill and decode MUST produce identical state** — Phase 1 guarantees this by using the *same* sequential scan for both (see §4).
- Single-sequence engine: use the `n_rs_seq==0` path (delta-net-base.cpp:479-496); skip the rollback-splits TODO (L498-500).

---

## 4. Staged kernel plan

### STAGE 1 — CORRECTNESS FIRST (no tensor cores beyond GEMM; all aux math f32)

| Op | Stage-1 impl | Oracle |
|----|--------------|--------|
| GEMM (wq/wk/wv/wo/wqkv/wqkv_gate/ssm_*/ffn/lm_head) | dequant weight (`dequant.cu`, validated vs `dequant.rs`) → f16/f32, then **cuBLASLt** GEMM via cudarc | llama.cpp CPU matmul |
| RMSNorm / L2Norm | `rmsnorm.cu` (1 block/row, block-reduce Σx², scale=rsqrt(mean+eps); L2 = /sqrt(Σx²+eps), no weight) | norm.cu:76-155 / ggml CPU |
| RoPE | `rope_neox.cu` partial NEOX over first 64 dims, base 1e7, dims 64..255 copied | rope.cu:185-268 (imrope branch :234-243) |
| Conv1d+SiLU | `ssm_conv.cu`: decode = 1-thread/channel shift-register; prefill = port `ssm_conv_f32` (n_t<=32) + `ssm_conv_long_token_f32` (n_t>32) | ssm-conv.cu:5-124 / ops.cpp:9409-9460 |
| GDN scan | `gated_delta_net.cu`: port `gated_delta_net_cuda<128,false,false>` verbatim. **Prefill = same kernel looping t=0..T-1** (state in registers, O(T) but logit-exact) | gated_delta_net.cu:84-111 / delta-net-base.cpp:289-371 (AR ref) |
| SDPA | `softmax.cu` naive: KQ = Kᵀ@Q (GQA broadcast kv=q/4) → scale → causal mask → row softmax (f32) → V@softmax | llama-graph.cpp:2131-2188 (non-flash) |
| gate / silu / sigmoid / add / mul | `elementwise.cu` | trivial |

Stage-1 GDN launch (decode & prefill): grid=(H=32, n_seqs, ceil(128/4)=32), block=(min(warp,128)=32, 4).
scale=1/sqrtf(128). g passed RAW (kernel does expf), beta pre-sigmoid'd, q/k pre-L2-normed. K=1 ⇒
`keep_rs_t=false`. Feed state from `recur[il].ssm_state` (transposed layout) and write it back.

### STAGE 2 — sm_120-FAST (only after Stage-1 logits match)
- **GEMM:** replace cuBLASLt with the proven sm_120a mma.sync paths (block-FP8 381 TFLOP / MXFP4 762 TFLOP / FP16 117) from the `bw24-probe` spine — MMQ-style (quant×quant) for Q8_0/Q4_K, Marlin-style or NVFP4 for the NVFP4 GGUFs. Keep cuBLASLt as the fallback/oracle.
- **GDN prefill:** chunked WY form (delta-net-base.cpp:16-287, CS=64 for GDA / FLA `chunk.py`): g cumsum → decay_mask exp → kkt → solve_tri (per-chunk forward-substitution in smem) → v_new → sequential cross-chunk state recurrence + intra/inter outputs. Tiles 64×64 and 128×64 → sm_120 `mma.sync.m16n8k16` FP16/TF32 (verified). Keep the recurrent kernel for T==1.
- **Attention prefill:** port `fattn-mma-f16.cuh` DKQ=DV=256 (uses only `mma.sync.m16n8k16`, Ampere-class PTX — sm_120-safe; **no wgmma/tcgen05/CUTLASS**). Cast K/V to f16, pad KV%FATTN_KQ_STRIDE. Reference fast hand-written sm_120 FA: gau-nernst fa-5090 `07_attention@e83c256` (head_dim 128 → split into 2×128 K-tiles for 256).
- **Attention decode:** port `fattn-vec.cuh` D=256 instance (plain CUDA warp-per-KV-tile, 128 threads, no TC).
- Conv/norm/rope/gated-norm stay as the small f32 kernels — they're bandwidth-bound; no MMA needed.
- Do **not** pursue wgmma or tcgen05 (do not run on sm_120). FP4 block-scale needs `-gencode arch=compute_120a,code=sm_120a` (already the project default).

---

## 5. Exact files to copy, per kernel

All paths under `/home/avifenesh/projects/llama.cpp` unless noted. Copy the math node-for-node.

**GDN scan** (`cu/gated_delta_net.cu`):
- `ggml/src/ggml-cuda/gated_delta_net.cu:4-168` kernel — port the **!KDA branch lines 84-111** for decode; lines 161-167 write state back. Drop the `keep_rs_t` branch (145-158) for Phase 1. Drop `ggml_cuda_pdl_sync()` (optional overlap).
- Launch config & strides: `:170-224` (grid/block, S_v=128 case :213-219), op dispatch & scale=1/sqrt(S_v): `:226-312` (kda detect `src_g->ne[0]==S_v` :247 — assert it's 1 here).
- Reference math/state-cache: `src/models/delta-net-base.cpp:289-371` (AR decode, CPU-checkable), `:449-525` (conv-state), `:527-606` (state write-back). Output buffer contract: `ggml/src/ggml.c:6216-6269`.
- Cross-check: FLA `fla/ops/gated_delta_rule/fused_recurrent.py` (STATE_V_FIRST=False inner loop); chunked = `fla/ops/gated_delta_rule/chunk.py` (+ `common/chunk_delta_h.py`, `chunk_o.py`, `wy_fast.py`).

**Conv1d + SiLU** (`cu/ssm_conv.cu`):
- `ggml/src/ggml-cuda/ssm-conv.cu:5-58` `ssm_conv_f32` (n_t<=32, register ring buffer) — set `d_conv=4, apply_silu=true, bias=nullptr`. `:60-124` `ssm_conv_long_token_f32` (n_t>32, smem). Launcher `:126-158` (case 4). Op/strides `:160-206`.
- CPU oracle: `ggml/src/ggml-cpu/ops.cpp:9409-9460`. Shape contract: `ggml/src/ggml.c` `ggml_ssm_conv`. State-cache assembly/writeback: `delta-net-base.cpp:449-525` (single-seq path 479-496).
- Decode shift-register reference: FLA `fla/modules/conv/short_conv.py` `ShortConvolution.step()` (cache.roll(-1,-1); cache[:,:,-1]=x; y=Σ(cache*w)).

**Full-attn (SDPA decode/prefill)** (`cu/` — Stage 2):
- Decode: `ggml/src/ggml-cuda/fattn-vec.cuh` D=256 instance. Prefill: `ggml/src/ggml-cuda/fattn-mma-f16.cuh` DKQ=DV=256. MMA PTX: `ggml/src/ggml-cuda/mma.cuh:1001-1022` (m16n8k16 f16) / `:1162-1174` (f32 acc). Dispatch: `fattn.cu:122-243` (case 256), `:245-329` (FATTN_VEC_CASE 256), `:340-538` (best-kernel select).
- Stage-1 oracle (non-flash): `src/llama-graph.cpp:2066-2194` (KQ/softmax/KQV at :2131-2188).

**RMSNorm / L2Norm** (`cu/rmsnorm.cu`): `ggml/src/ggml-cuda/norm.cu:76-155` `rms_norm_f32<block_size,do_multiply>` (drop the fused add/fastmodulo). RMS def: `src/llama-graph.cpp:1154-1187` (plain rms*weight, NO +1, NO bias). L2norm = same minus the `*weight` and minus the mean (divide by sqrt(Σx²+eps)).

**RoPE** (`cu/rope_neox.cu`): `ggml/src/ggml-cuda/rope.cu:185-268` (imrope branch :234-243, partial-rotary copy-through :222-227), launcher `:420-459` (theta_scale=base^(-2/n_dims) :448), `rope_yarn:15-37` (ext_factor=0 ⇒ skip). For Phase-1 text-only, hard-simplify to NEOX partial with a single shared pos array.

**FFN**: dense SwiGLU = three GEMMs + silu*mul (`qwen35.cpp:472-485`, `build_ffn` LLM_FFN_SILU/LLM_FFN_PAR).

**Cross-validation (Qwen3-Next == Qwen3.5 arch)**: vLLM `vllm/model_executor/models/qwen3_next.py` (`Qwen3NextAttention`, `Qwen3NextGatedDeltaNet`). NOTE vLLM adds +1 to norm weights (GemmaRMSNorm, raw HF); **GGUF has +1 pre-baked — do NOT add it**.

---

## 6. Logit-match harness vs llama.cpp

Pin everything identical: same Q8_0 GGUF, same prompt, batch=1, greedy/temp=0, RoPE base 1e7, eps 1e-6,
KV+state f32 in our impl first.

1. **Reference dump (llama.cpp):** build llama.cpp with CUDA; run
   `llama-cli -m /home/avifenesh/ai-ml/models/qwen3.5-9b-judge-q8_0.gguf -p "<prompt>" -n 1 --no-warmup --temp 0 -ngl 99`
   and dump `result_output` (first-token logits) — add a debug print of the `result_output` tensor for a
   single ubatch, or use a tiny eval harness. The GGML CUDA `GGML_OP_GATED_DELTA_NET` path is live
   (`ggml-cuda.cu:5433-5439`), so you can also diff the GDN op output directly on the 5090.
2. **Our dump:** `bin/logit_check.rs` runs the same prompt through `forward()`, downloads the
   `[n_vocab]` logits, writes them to JSON.
3. **Gate (must pass before Stage 2):**
   - **argmax token MUST match exactly.**
   - top-32 logits: abs tol **1e-2**, rel tol **1e-3** (fp16 GEMM operands). GDN/conv/norm done in f32 ⇒ those intermediates should be tighter (~1e-4).
4. **Bisect order on mismatch** (cb-tag intermediates and diff layer-by-layer against llama.cpp callback dumps `attn_norm`, `Qcur_normed`, `gate`, `q_conv_predelta`, `final_output`, `l_out`, `result_norm`):
   - argmax wrong ⇒ check **rope partial-rotary** (only 64 dims), **q|gate split** (stride 512 per head), **ssm_a sign** (already -exp), **state transpose** (M[col][i]).
   - small drift ⇒ check eps placement, freq_base=1e7, kq_scale=1/sqrt(256) vs gdn scale=1/sqrt(128), beta=plain sigmoid (NOT 2*sigmoid).

Validate kernels bottom-up *before* the full forward:
- `dequant.cu` vs `dequant.rs` (exact bit match expected).
- `rmsnorm.cu` / `rope_neox.cu` / `ssm_conv.cu` / `gated_delta_net.cu` each against a tiny hand-fed input vs the ggml CPU op (`ops.cpp`) or a Python/FLA reference. Conv: feed a [3+1]-col window, diff vs `ggml_compute_forward_ssm_conv_f32`.

---

## 7. Ordered task list (smallest runnable milestone first)

Existing workspace tasks: #3 GGUF loader (in_progress), #4 dequant oracle (done), #5 vanilla-dense
spine (pending), #6 hybrid forward (pending). This list refines #5/#6.

**M0 — Spine proof: pure-dense Qwen3-1.7B forward.** (`bin/run_dense.rs`)
Smallest end-to-end thing that exercises the whole pipeline with NO hybrid pieces. Tensors:
tok_embd, per-layer attn_norm/q/k/v/o/q_norm/k_norm/ffn_norm/ffn_gate/up/down, output_norm, output.
Kernels needed: dequant, cuBLASLt GEMM, RMSNorm, RoPE (full NEOX, head_dim 128), naive SDPA (GQA),
silu/mul, lm_head. Gate: argmax logit matches llama.cpp on Qwen3-1.7B. **This proves cudarc + sm_120a
fatbin + GEMM + the orchestration loop end-to-end.** Everything after is additive.

**M1 — RMSNorm + RoPE + dequant kernels validated** standalone vs oracle (subset of M0, but lock the numeric gates here).

**M2 — `ssm_conv.cu` decode + prefill** validated vs `ggml_compute_forward_ssm_conv_f32` on a single channel window. (Daily hot path is decode — do the 1-thread/channel shift register first.)

**M3 — `gated_delta_net.cu` scan** validated vs FLA / `build_delta_net_autoregressive` on one v-head, one token, then T tokens (prefill = same kernel). Confirm state round-trips across decode steps (transposed layout).

**M4 — Single LINEAR_ATTN layer** wired (wqkv/wqkv_gate/ssm_beta/ssm_alpha GEMMs → beta/g_log → conv → split → L2norm → GDN → gated-norm → ssm_out) vs llama.cpp `linear_attn_out` cb dump for il=0.

**M5 — Single FULL_ATTN layer** wired (q|gate split, q/k norm, partial RoPE, naive SDPA, sigmoid-gate, wo) vs `attn_output` cb dump for il=3.

**M6 — Full Qwen3.5-9B forward, prefill+decode, dual cache.** (`bin/run_hybrid.rs` + `bin/logit_check.rs`)
Loop all 32 layers, output_norm, lm_head. **Gate: argmax matches + top-32 within tol (§6).** This is the
Phase-1 finish line for the 9B judge.

**M7 — Qwen3.5-27B dense** (same code; n_layer=64, 16 full-attn, n_embd=5120, ssm inner=6144, dt_rank=48). Only config/shapes change.

**M8 (optional) — MTP/NextN draft head** (`forward_mtp.rs`): feed `t_h_nextn` + new token through enorm/hnorm/eh_proj + a full-attn block; reuse M5 kernels. Validate +1-token draft logits.

**M9 — Stage 2 fast path** (only after M6 green): swap cuBLASLt→sm_120a mma.sync GEMM, add chunked GDN prefill, add fattn-mma-f16 D=256 prefill + fattn-vec D=256 decode. Re-run §6 gate after each swap.

Key gotchas to keep pinned (full list in the component research): RoPE is partial (64 of 256) IMROPE→NEOX
text-only; q|gate fused per-head stride 512; output gate is `attn*sigmoid(gate)` AFTER SDPA; ssm_a is
pre-negated; g passed RAW (kernel exps); beta pre-sigmoid'd; GDN state transposed M[col][i]; two separate
residuals; post_attention_norm is the PRE-FFN norm; conv has NO bias, weight is kernel-major [4,8192];
MTP block loaded at il=32 but NOT run in the main loop; output/embed_tokens/shared_head_head tie to
tok_embd/output when absent.
