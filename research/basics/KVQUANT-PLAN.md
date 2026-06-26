# KV-Cache Quantization (q8_0 K / q5_1 V) — Build Plan

I have everything needed. Note a critical detail the findings missed: `head_dim_v` may differ from `head_dim_k` (config.rs:69-70, 100), and the prefill path in `hybrid_forward.rs:68-117` builds K/V locally per-tile and does NOT use the KV cache. Now writing the plan.

---

# KV-Cache Quantization (q8_0 K / q5_1 V) — Concrete Implementation Plan for bw24

## 0. Critical corrections to the FINDINGS before anyone codes

The research findings contain factual errors that would corrupt the cache layout. Fix these first:

| # | Findings claim | Ground truth | Source |
|---|---|---|---|
| **C1** | q5_1 block = **20 bytes** (one finding) / **12 bytes** (used everywhere in the impl map) | **q5_1 block = 24 bytes**: `dm`(union d+m, 4B) + `qh[4]`(4B) + `qs[16]`(16B). `static_assert(sizeof(block_q5_1)==2*sizeof(ggml_half)+sizeof(uint32_t)+QK5_1/2)` → 4+4+16=24. The 12-byte figure is fiction; **every `v_block_bytes = (kv_dim/32)*12` in the impl map is wrong → use `*24`**. | `ggml-common.h:227-239` (verified) |
| **C2** | "fa_decode at flash_attn.cu:356-393 is a single-pass loop with `m_i/l_i`" | The real kernel is `fa_decode_f32` (flash_attn.cu:325-399), a **flash-decoding split-K** kernel writing `partO/partM/partL`, **followed by a separate `fa_decode_combine_f32`** merge (lib.rs:532-536). Both must be touched; the dequant goes only in the inner key loop (flash_attn.cu:360-394). | `flash_attn.cu:325-399`, `lib.rs:516-537` (verified) |
| **C3** | "prefill builds a quantized cache during prefill; fa_prefill reads the cache" | `hybrid_forward.rs:68-117` computes q/k/v **for the whole prompt locally** and calls `fa_prefill(&q,&k,&v,...)` (line 105) with **fresh f32 K/V — it never touches `cache.kv`**. So prefill quantization is *optional* and only matters if you want prefill to *populate* the cache for the subsequent decode. **Current code re-derives nothing from cache at prefill.** See §E3. | `hybrid_forward.rs:103-105`, `forward.rs:75` (verified) |
| **C4** | head_dim is always 256, divisible by 32 | `head_dim_k` and `head_dim_v` are **independent** config fields (`config.rs:69-70`, `head_dim_v` defaults to `head_dim_k` at line 100). K cache must use `head_dim_k`, V cache must use `head_dim_v`. For Qwen3.5 both are 256, but **assert each %32==0 separately**. | `config.rs:68-70,96-100` (verified) |
| **C5** | `fa_decode_f32` block = `head_dim` threads, one thread per output dim | Confirmed (lib.rs:527 `block_dim:(head_dim,1,1)`). This shapes the dequant: **thread `tid` owns dim `tid`**, so each thread reads exactly one quantized element from K and one from V. This is the *easy* case — no warp-cooperative block decode needed. | `flash_attn.cu:339,370,391` (verified) |

The plan below uses the **corrected 24-byte q5_1 layout** throughout.

---

## A. Exact block layouts + dequant formulas

### A.1 q8_0 (K) — symmetric, 34 bytes / 32 elements
```
block_q8_0 {            // ggml-common.h:241-246
  ggml_half d;          // bytes [0..2)  f16 scale
  int8_t    qs[32];     // bytes [2..34) signed quants
}                       // total 34 B   (1.0625 B/elem)
```
Dequant (per element j): `x[j] = f16_to_f32(d) * (float)qs[j]`
(`ggml-quants.c` dequantize_row_q8_0; CUDA `dequantize.cuh:89-99`).

### A.2 q5_1 (V) — asymmetric/affine, **24 bytes** / 32 elements
```
block_q5_1 {            // ggml-common.h:227-239
  union { struct{ ggml_half d; ggml_half m; }; ggml_half2 dm; }; // bytes [0..4)
  uint8_t qh[4];        // bytes [4..8)   32 high-bits (bit i = 5th bit of elem i)
  uint8_t qs[16];       // bytes [8..24)  two 4-bit nibbles per byte
}                       // total 24 B   (0.75 B/elem)
```
Dequant (per element j, j∈[0,32)):
```
d  = f16_to_f32(dm.x);  m = f16_to_f32(dm.y)
lo = (j < 16) ? (qs[j]      & 0x0F)        // low nibble of byte j
              : (qs[j-16] >> 4)            // high nibble of byte j-16
hi = ((qh_u32 >> j) & 1) << 4              // 5th bit -> position 4
q5 = lo | hi                               // value in [0,31]
x[j] = d * (float)q5 + m
```
`qh` is read as one `uint32_t` (little-endian) so `bit j` is the high bit of element `j` (`ggml-quants.c` quantize_row_q5_1_ref / dequantize_row_q5_1; CUDA `dequantize.cuh:71-87`). The flash-attn variant in `fattn-common.cuh` packs the same bit but is irrelevant to bw24's per-element decode.

### A.3 Quantize (encode) formulas — needed by the append kernel
**q8_0 (over a 32-elem block):** `amax = max|x|; d = amax/127; id = d?1/d:0; qs[j] = round(x[j]*id)` clamped to [-127,127].
**q5_1 (affine, over 32-elem block):** `mn=min(x); mx=max(x); d=(mx-mn)/31; id=d?1/d:0; m=mn; q5=clamp(round((x[j]-mn)*id),0,31); qs nibble=q5&0xF; qh bit j=(q5>>4)&1`.

---

## B. New `CudaSlice<u8>` KV cache layout

`KvLayer` (cache.rs:12-17) changes from f32 to byte storage with per-element-type head dims and precomputed block strides:

```rust
pub struct KvLayer {
    pub k: CudaSlice<u8>,        // q8_0 packed
    pub v: CudaSlice<u8>,        // q5_1 packed
    pub kv_dim_k: usize,         // head_dim_k * n_head_kv
    pub kv_dim_v: usize,         // head_dim_v * n_head_kv
    pub k_tok_bytes: usize,      // (kv_dim_k/32)*34   bytes per token, K
    pub v_tok_bytes: usize,      // (kv_dim_v/32)*24   bytes per token, V  (NOT 12!)
    pub len: usize,
}
```

**Per-token offsets** (replacing the f32 `off = len*kv_dim` at decode.rs:124):
- K byte offset for token t: `t * k_tok_bytes`
- V byte offset for token t: `t * v_tok_bytes`

**Within a token, per (kv_head h, head_dim d)** — keep the existing `[token, kv_head, dim]` element order so the kernel's `kv_head*head_dim + d` indexing maps cleanly onto blocks. Global element index within token = `h*head_dim + d`. Its block = `idx/32`, intra-block lane = `idx%32`. K block base byte = `t*k_tok_bytes + (idx/32)*34`; V block base byte = `t*v_tok_bytes + (idx/32)*24`. **Requirement:** `head_dim_k % 32 == 0` AND `head_dim_v % 32 == 0` so a 32-block never straddles two heads (assert in `Cache::new`).

**Allocation (cache.rs:50-54):**
```rust
k: e.alloc_u8(max_ctx * k_tok_bytes)?,   // alloc_u8 exists, lib.rs:92
v: e.alloc_u8(max_ctx * v_tok_bytes)?,
```
Memory vs f32 for Qwen3.5 (head_dim=256, n_head_kv=4 → kv_dim=1024, 32 blocks/token): K f32 4096 B/tok → q8_0 1088 B/tok (3.77×); V f32 4096 B/tok → q5_1 768 B/tok (5.33×). Combined ~4.5× KV shrink.

---

## C. Append-quantize kernel(s) (new CUDA, in flash_attn.cu or a new kv_quant.cu)

One kernel handling both K and V for the single new token:

```c
// grid = (max(kv_dim_k, kv_dim_v)/32, 1, 1)  -- one CTA per 32-elem block
// block = (32,1,1)                            -- one thread per element
extern "C" __global__ void append_quantize_kv_q8_0_q5_1(
    const float* __restrict__ k_row,  // [kv_dim_k]   post-RoPE K for this token
    const float* __restrict__ v_row,  // [kv_dim_v]
    uint8_t* __restrict__ K,          // cache base
    uint8_t* __restrict__ V,
    int t, int kv_dim_k, int kv_dim_v,
    long k_tok_bytes, long v_tok_bytes)
```
Each CTA `b` does a 32-lane warp reduction over its block:
- **K block b** (if `b*32 < kv_dim_k`): warp-reduce `amax` over `fabsf(k_row[b*32+lane])`; lane0 computes `d`, stores `__float2half(d)` to `K + t*k_tok_bytes + b*34`; each lane writes `qs[lane] = clamp(round(k_row[..]/d),-127,127)` to byte `+2+lane`.
- **V block b** (if `b*32 < kv_dim_v`): warp-reduce `min` and `max`; lane0 computes `d=(mx-mn)/31, m=mn`, stores `dm` (4B). Each lane computes `q5=clamp(round((v-mn)/d),0,31)`. Pack nibble via `atomicOr`-free scheme: lane writes its low-nibble using one `__byte_perm`/shared staging, OR use the canonical two-pass (lanes 0..15 own low nibble of `qs[lane]`, lanes 16..31 own high nibble of `qs[lane-16]`) and `qh` bits via `__ballot_sync(0xFFFFFFFF, (q5>>4)&1)` → that single 32-bit ballot **is exactly `qh`** (bit j set iff elem j has 5th bit). Lane0 writes the ballot result as the 4-byte `qh`. This is the clean, branch-free q5_1 pack.

Use `__ballot_sync` for `qh` — it produces the precise little-endian bit layout q5_1 expects, matching `dequantize.cuh:71-87`.

---

## D. Fused inline-dequant changes

### D.1 `fa_decode_f32` (flash_attn.cu:325-399) — the marked HOOK at 362-370

Signature change (line 327-328): `const float* K, V` → `const uint8_t* __restrict__ K, V`. Add params `long k_tok_bytes, long v_tok_bytes` (token strides differ from f32). Thread `tid` owns dim `tid` of `kv_head` (line 338-339).

**K dequant — replaces line 369-370** (`kt[tid]`):
```c
const int kidx = kv_head*head_dim + tid;        // element index within token t
const uint8_t* kblk = K + (size_t)t*k_tok_bytes + (kidx>>5)*34;
const half  kd_h = *(const half*)kblk;
const int8_t kq  = ((const int8_t*)(kblk+2))[kidx & 31];
float ktv = __half2float(kd_h) * (float)kq;
float prod = (tid < head_dim) ? sq[tid] * ktv : 0.0f;
```
The warp+block reduction (372-383) and online softmax (387-393) are **unchanged** (FINDINGS C2-correct: m_i/l_i stay f32).

**V dequant — replaces line 390-391** (`vt[tid]`):
```c
const int vidx = kv_head*head_dim + tid;        // NOTE: use head_dim_v if it differs
const uint8_t* vblk = V + (size_t)t*v_tok_bytes + (vidx>>5)*24;
const half vd = *(const half*)vblk;             // dm.x
const half vm = *(const half*)(vblk+2);         // dm.y
uint32_t qh = *(const uint32_t*)(vblk+4);
const int j  = vidx & 31;
const uint8_t* qs = vblk + 8;
int lo = (j < 16) ? (qs[j] & 0x0F) : (qs[j-16] >> 4);
int q5 = lo | (((qh >> j) & 1u) << 4);
float vtv = __half2float(vd) * (float)q5 + __half2float(vm);
if (tid < head_dim) acc = acc * alpha + p * vtv;
```
`fa_decode_combine_f32` (flash_attn.cu:403+) is **dtype-agnostic** (it only merges partials) → no change.

### D.2 `fa_prefill_f32` (flash_attn.cu:133-300) — staging loop at 180-186

The prefill kernel stages K/V into bf16 smem (`sK`,`sV`) before tensor-core MMA. **Dequant happens during the staging copy** (lines 182-185), so MMA, softmax, and PV are untouched — the cleanest integration point (matches FINDINGS open-question: bw24 *stages*, never inline-MMA-dequants).

Replace lines 182-185:
```c
int kidx = kv_head*head_dim + d;
// q8_0 K
const uint8_t* kblk = K + (size_t)(k0+kk)*k_tok_bytes + (kidx>>5)*34;
float kv = (kk < nk) ? __half2float(*(const half*)kblk) *
                       (float)((const int8_t*)(kblk+2))[kidx&31] : 0.0f;
// q5_1 V  (same decode as D.1)
float vv = (kk < nk) ? dequant_q5_1_elem(V,(k0+kk),v_tok_bytes,kv_head,d,head_dim) : 0.0f;
sK[i] = __float2bfloat16(kv);
sV[i] = __float2bfloat16(vv);
```
Factor the two decodes into `__device__ inline` helpers (`dq_q8_0_elem`, `dq_q5_1_elem`) shared by D.1/D.2/append. Signature change: `const float* K,V` → `const uint8_t* K,V` + `long k_tok_bytes, v_tok_bytes` (line 134-135).

---

## E. bw24 call-site changes (file:line)

### E1 — `cache.rs` (KvLayer + Cache::new)
- **cache.rs:12-17**: replace struct as in §B.
- **cache.rs:40**: split into `kv_dim_k = head_dim_k*n_head_kv`, `kv_dim_v = head_dim_v*n_head_kv`.
- **cache.rs:47-54**: `assert!(head_dim_k%32==0 && head_dim_v%32==0)`; compute `k_tok_bytes=(kv_dim_k/32)*34`, `v_tok_bytes=(kv_dim_v/32)*24`; allocate via `e.alloc_u8(max_ctx*k_tok_bytes)?` / `*v_tok_bytes`; populate the new fields.

### E2 — `decode.rs` `full_attn_decode` (decode.rs:88-147)
- Note RoPE uses `head_dim` (decode.rs:94 = `head_dim_k`); K/V are still produced as **f32** by matmul (lines 102-117) — no change to projection. If `head_dim_v != head_dim_k`, V projection width must use `head_dim_v` (currently the code assumes one `head_dim`; flag this — see Contradictions).
- **decode.rs:124-127**: delete `off`/`copy_into`×2; replace with `e.append_kv_quantized(&k, &v, &mut kvl.k, &mut kvl.v, kvl.len, kvl.kv_dim_k, kvl.kv_dim_v, kvl.k_tok_bytes, kvl.v_tok_bytes)?;` (kvl.len is the pre-increment token index). Keep `kvl.len += 1` (line 127).
- **decode.rs:131-132**: views become byte views: `let k_view = e.view_u8(&kvl.k, t_kv * kvl.k_tok_bytes); let v_view = e.view_u8(&kvl.v, t_kv * kvl.v_tok_bytes);`
- **decode.rs:135** (`BW24_NOFA` naive path): naive SDPA stays f32-only → either (a) error out when quantized KV active, or (b) gate quantization behind an env flag so NOFA still works. Recommend (b): `BW24_KVQUANT` opt-in; when unset, keep current f32 cache entirely.
- **decode.rs:137**: `e.fa_decode(&q, &k_view, &v_view, &mut attn, head_dim, n_head, n_head_kv, t_kv, scale)?` — same arity; `fa_decode` now takes `CudaView<u8>` and passes the two tok_bytes (see E4).

### E3 — `hybrid_forward.rs` prefill (`full_attn`, lines 68-117)
- Per **C3**, prefill at line 105 uses fresh f32 q/k/v and **does not read/write the cache**. Two options:
  1. **Minimal (recommended for Stage-1 validation):** leave `fa_prefill` f32. Decode (E2) builds the quantized cache token-by-token starting from `len=0`. Prefill correctness is then validated independently. The token[55] test (§F) drives decode, which is where quantization lives.
  2. **Full prefill-populates-cache:** after computing f32 k/v for all T prompt tokens (lines ~99-104), call a batched `append_quantize_kv` for T tokens, then call the **quantized** `fa_prefill` reading the cache. This needs `fa_prefill` to accept `CudaView<u8>` and is only required if you want prefill itself to use the compressed cache. Defer unless prefill VRAM is the bottleneck.
- **forward.rs:75** is the non-FA dense prefill (`sdpa_naive`) — leave f32; same reasoning.

### E4 — `lib.rs` launchers
- **lib.rs:74-76**: add `pub fn view_u8<'a>(&self, b:&'a CudaSlice<u8>, len:usize) -> CudaView<'a,u8> { b.slice(0..len) }`.
- **New launcher** (pattern from `quantize_q8_1`, lib.rs:158-170): `append_kv_quantized(&self, k:&CudaSlice<f32>, v:&CudaSlice<f32>, kc:&mut CudaSlice<u8>, vc:&mut CudaSlice<u8>, t:usize, kv_dim_k, kv_dim_v, k_tok_bytes, v_tok_bytes)` → `func("append_quantize_kv_q8_0_q5_1")`, grid `(max(kv_dim_k,kv_dim_v)/32,1,1)`, block `(32,1,1)`. Args: k_row, v_row, kc, vc, `t as i32`, dims, tok_bytes as `i64`. **Must pass the cache `t` offset inside the kernel** (kernel computes base byte from `t*tok_bytes`).
- **fa_decode** (lib.rs:516-537): change `k:&CudaView<f32>, v:&CudaView<f32>` → `&CudaView<u8>`; add `k_tok_bytes:usize, v_tok_bytes:usize` params; append `.arg(&(k_tok_bytes as i64)).arg(&(v_tok_bytes as i64))` to the launch builder (line 529-530). Combine launch (532-536) unchanged.
- **fa_prefill** (lib.rs:493-513): only if E3 option-2 chosen — change K/V to `&CudaView<u8>`, add the two tok_bytes args (line 510).
- The `quantize_q8_1` kernel (qmatvec.cu, lib.rs:160) is for **activations (q8_1, 4B scale+sum per block)**, NOT reusable for q8_0/q5_1 cache — write fresh device helpers.

---

## F. Validation

**Invariant:** with `BW24_KVQUANT` on vs off, **argmax of the logits at token[55] must be identical** on the 9B and 27B models; the full softmax distribution should match within int8/5-bit quantization tolerance.

1. **Block round-trip unit test** (CPU oracle): quantize a random f32 row with the new kernels, DtoH, dequantize with the §A formulas, compare against `dequantize_row_q8_0`/`q5_1` from the existing `tools/ggml_dequant_ref` (already in the tree per git status). Assert max abs err ≤ `d` (q8_0) and ≤ `d/2` (q5_1) per element. This isolates layout/packing bugs from attention bugs.
2. **`qh` ballot correctness**: explicitly test a block where elements straddle the 5th-bit boundary (values 15↔16, 31) so a wrong bit position is caught — this is the #1 risk given the 24-vs-12-byte confusion.
3. **End-to-end argmax at token[55]**:
   - Baseline: run current f32-KV decode, capture argmax(logits) at generated token index 55 for 9B and 27B (the repo already validates argmax==llama.cpp for MoE per commit `02af8fc` — reuse that harness).
   - Quantized: same prompt, `BW24_KVQUANT=1`. **Require argmax_q == argmax_f32** at token[55] for both models. Token[55] gives ≥55 cached keys → exercises multi-block, multi-split (`n_splits` flips >256 keys per lib.rs:520) and the combine kernel.
   - Tolerance: report `max|logit_q - logit_f32|` and top-5 KL; argmax equality is the hard gate, logit drift is informational. q8_0 K typically perturbs scores <0.1%; q5_1 V is the looser link — if argmax flips, first suspect the q5_1 `qh` packing (test #2), then the affine `m` offset.
4. **Regression**: confirm `BW24_KVQUANT` unset path is byte-identical to today (f32 cache untouched), and `BW24_NOFA` still works (f32 fallback).

---

## Contradictions / open issues to flag to the implementer

1. **q5_1 = 24 bytes, not 12/20.** The entire FINDINGS impl map sizes V at `(kv_dim/32)*12`. Using 12 would truncate every V block by half and silently corrupt attention. Use **24** (verified `ggml-common.h:239`). This is the single highest-risk error in the research.
2. **`fa_decode` is split-K + a combine kernel** (lib.rs:516-537, two launches), not the single-pass loop FINDINGS describes. The dequant lives only in the inner loop of `fa_decode_f32`; the combine kernel needs no change. Don't look for FINDINGS' `m_i/l_i` single-pass structure — it's the split version.
3. **Prefill does not use the KV cache today** (`hybrid_forward.rs:105` passes fresh f32 q/k/v). The task asks to modify `fa_prefill`'s inner loop — but that's only meaningful under E3 option-2 (prefill-populates-cache). For pure correctness/memory at decode, **fa_prefill changes are not required**; flag whether the goal is decode-cache compression (E3.1, minimal) or full prefill-on-quantized-cache (E3.2).
4. **`head_dim_k` vs `head_dim_v`** are independent (`config.rs:69-70`). FINDINGS assumes one `head_dim`. K cache, V cache, and the decode/prefill V-dequant must use the correct one. The current `full_attn_decode` (decode.rs:94) uses a single `head_dim=head_dim_k` for both q and v — verify the model has `head_dim_v==head_dim_k` (Qwen3.5: yes, 256==256) or generalize before enabling on other models.
5. **`__ballot_sync` for `qh`** assumes the append kernel runs exactly one warp (32 threads) per 32-element block — true with `block_dim=(32,1,1)`. If you ever widen the block, the ballot mask must change.

**Files to touch:** `crates/bw24-engine/src/cache.rs:12-17,40,47-54`; `crates/bw24-engine/src/decode.rs:124-137`; `crates/bw24-engine/src/lib.rs:74-76,493-513,516-537` (+ new `append_kv_quantized`); `crates/bw24-engine/cu/flash_attn.cu:133-300,325-399` (+ new `append_quantize_kv_q8_0_q5_1` kernel and two `__device__` decode helpers). Optional: `crates/bw24-engine/src/hybrid_forward.rs:105`. Validation oracle: reuse `tools/ggml_dequant_ref` (present in working tree).
