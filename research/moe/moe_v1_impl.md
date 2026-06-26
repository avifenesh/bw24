I have everything I need. The actual code confirms the research: `GpuTensor` enum, `qmatvec(w_slice, x, m, in_f, out_f, qtype, row_bytes)` takes a `&CudaSlice<u8>` so a `.slice()` view works, the hybrid loader has the `assert!(cfg.moe.is_none())` to remove, and the FFN block in `hybrid_forward.rs` lines 35-46 is the exact branch point. Note IQ4_XS/IQ3_S are NOT in the qmatvec.cu QType enum nor dequant — so the Q6_K_XL model is the Stage-1 path.

Here is the complete, code-accurate implementation plan.

---

# bw24 MoE FFN (qwen35moe / Qwen3.6-35B-A3B) — complete implementation

## 0. Critical facts verified against actual bw24 source

- `Engine::qmatvec(w: &CudaSlice<u8>, x, m, in_f, out_f, qtype, row_bytes)` (lib.rs:70) takes a `&CudaSlice<u8>`. A `bytes.slice(a..b)` returns a `CudaView<u8>`, **not** `&CudaSlice<u8>` — so I add one thin `qmatvec_at` that takes a byte offset and slices internally, OR store experts as one big `CudaSlice<u8>` and pass a sub-slice. cudarc's `clone_htod`/`alloc` produce `CudaSlice`; `.slice()` yields a borrowed view. The kernel arg is `*const u8`, and `launch_builder.arg()` accepts both `&CudaSlice` and `&CudaView`. I add an offset-aware wrapper to avoid lifetime friction (below).
- `GpuTensor::load` (model.rs:36-38) computes `out_f = ne[1]`, `row_bytes = raw.len()/ne[1]` — **wrong for 3D stacked exps** (divides total by 512, ignoring the 256 expert dim → row_bytes is 256× too big). MoE exps must NOT use `load` as-is; add a 3D-aware loader.
- Quant support: qmatvec.cu QType = `{QT_Q8_0=0, QT_Q4_K=1, QT_Q6_K=2}` and `dequant.rs` = `{Q8_0,Q4_K,Q6_K}` only. **IQ4_XS/IQ3_S are NOT supported.** So Stage-1 validation uses the **Q6_K_XL** model (Q6_K gate/up, Q8_0 down/shexp), zero new kernels. The IQ4_XS model needs new dequant+qmatvec paths first (out of scope for the FFN wiring; noted in §5).
- F32 router (`ffn_gate_inp`, `ffn_gate_inp_shexp`) → `GpuTensor::Float` → goes through `e.linear`/`e.matmul` automatically.

---

## 1. MoE weights on `HybridLayer` + load/slice (model.rs + hybrid.rs)

### 1a. `config.rs` — add the shared-FFN length

```rust
// crates/bw24-gguf/src/config.rs
#[derive(Debug, Clone)]
pub struct MoeConfig {
    pub expert_count: u32,
    pub expert_used_count: u32,
    pub expert_ff_length: u32,
    pub expert_shared_ff_length: u32,   // NEW: qwen35moe.expert_shared_feed_forward_length (=512)
}
```
```rust
// in from_gguf(), replace the moe block:
let moe = if arch.is_moe() {
    Some(MoeConfig {
        expert_count:           u("expert_count").unwrap_or(0),
        expert_used_count:      u("expert_used_count").unwrap_or(0),
        expert_ff_length:       u("expert_feed_forward_length").unwrap_or(0),
        expert_shared_ff_length:u("expert_shared_feed_forward_length").unwrap_or(0),
    })
} else { None };
```

### 1b. `model.rs` — 3D stacked-expert loader + per-expert slice accessor on `GpuTensor`

`GpuTensor::Quant` already carries `ne`, so for a 3D exps tensor `ne = [in, out, n_expert]`. Add a loader that computes `row_bytes` correctly (divides by `out * n_expert`, not just `out`) and an accessor returning the byte range of one expert.

```rust
// crates/bw24-engine/src/model.rs  (add methods to impl GpuTensor)

impl GpuTensor {
    /// Load a STACKED 3D expert tensor (ggml ne = [in, out, n_expert]) as packed quant bytes.
    /// Unlike `load`, computes row_bytes = total / (out * n_expert) so each expert is a
    /// correctly-strided [out, in] quant matrix. Panics if the tensor is not a supported quant.
    pub fn load_exps(e: &Engine, g: &GgufFile, name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let t = g.find(name).unwrap_or_else(|| panic!("missing exps tensor {name}"));
        let raw = g.tensor_data(t);
        let qtype = match t.ggml_type {
            GgmlType::Q8_0 => QT_Q8_0,
            GgmlType::Q4_K => QT_Q4_K,
            GgmlType::Q6_K => QT_Q6_K,
            other => panic!("exps tensor {name} has unsupported quant {other:?} \
                             (only Q8_0/Q4_K/Q6_K; IQ4_XS/IQ3_S need new kernels)"),
        };
        assert!(t.ne.len() == 3, "load_exps expects 3D [in,out,n_expert], got {:?}", t.ne);
        let out_f   = t.ne[1] as usize;
        let n_exp   = t.ne[2] as usize;
        let row_bytes = raw.len() / (out_f * n_exp);          // bytes of ONE [in]-wide quant row
        Ok(GpuTensor::Quant { bytes: e.htod_bytes(raw)?, qtype, row_bytes, ne: t.ne.clone() })
    }

    /// Byte range of expert `eidx`'s [out, in] sub-matrix inside a stacked Quant exps tensor,
    /// plus its (in_f, out_f, qtype, row_bytes) for a direct qmatvec call.
    pub fn expert_slice(&self, eidx: usize) -> (std::ops::Range<usize>, usize, usize, i32, usize) {
        match self {
            GpuTensor::Quant { qtype, row_bytes, ne, .. } => {
                let in_f  = ne[0] as usize;
                let out_f = ne[1] as usize;
                let per_expert = out_f * row_bytes;            // bytes for one expert's full matrix
                let start = eidx * per_expert;
                (start..start + per_expert, in_f, out_f, *qtype, *row_bytes)
            }
            GpuTensor::Float { .. } => panic!("expert_slice on a Float tensor"),
        }
    }
}
```

### 1c. `hybrid.rs` — `MoeWeights`, `Ffn` enum, loader

```rust
// crates/bw24-engine/src/hybrid.rs

/// All MoE FFN weights for one layer. Routed exps are stacked 3D quant tensors;
/// router + shared-gate are F32. Shared expert is a normal SwiGLU triple.
pub struct MoeWeights {
    pub gate_inp:        GpuTensor,   // F32 [n_embd, n_expert]      router
    pub gate_exps:       GpuTensor,   // Q6_K [n_embd, n_ff_exp, n_expert]
    pub up_exps:         GpuTensor,   // Q6_K [n_embd, n_ff_exp, n_expert]
    pub down_exps:       GpuTensor,   // Q8_0 [n_ff_exp, n_embd, n_expert]
    pub gate_inp_shexp:  GpuTensor,   // F32 [n_embd]                shared sigmoid gate
    pub gate_shexp:      GpuTensor,   // Q8_0 [n_embd, n_ff_shexp]
    pub up_shexp:        GpuTensor,   // Q8_0 [n_embd, n_ff_shexp]
    pub down_shexp:      GpuTensor,   // Q8_0 [n_ff_shexp, n_embd]
}

pub enum Ffn {
    Dense { gate: GpuTensor, up: GpuTensor, down: GpuTensor },
    Moe(MoeWeights),
}

pub struct HybridLayer {
    pub attn_norm: GpuTensor,
    pub post_attn_norm: GpuTensor,
    pub mixer: Mixer,
    pub ffn: Ffn,                      // was: ffn_gate / ffn_up / ffn_down
}
```

Loader change (replace the `assert!(cfg.moe.is_none())` at hybrid.rs:56 and the dense FFN fields at 93-95):

```rust
// in HybridModel::load(), remove line 56's assert. Keep cfg as before.
// ... inside the `for il` loop, replace the dense ffn_* loads with:

let ffn = if cfg.moe.is_some() {
    Ffn::Moe(MoeWeights {
        gate_inp:       load_t(e, g, &p("ffn_gate_inp.weight"))?,          // F32 -> Float
        gate_exps:      GpuTensor::load_exps(e, g, &p("ffn_gate_exps.weight"))?,
        up_exps:        GpuTensor::load_exps(e, g, &p("ffn_up_exps.weight"))?,
        down_exps:      GpuTensor::load_exps(e, g, &p("ffn_down_exps.weight"))?,
        gate_inp_shexp: load_t(e, g, &p("ffn_gate_inp_shexp.weight"))?,    // F32 [n_embd]
        gate_shexp:     load_t(e, g, &p("ffn_gate_shexp.weight"))?,
        up_shexp:       load_t(e, g, &p("ffn_up_shexp.weight"))?,
        down_shexp:     load_t(e, g, &p("ffn_down_shexp.weight"))?,
    })
} else {
    Ffn::Dense {
        gate: load_t(e, g, &p("ffn_gate.weight"))?,
        up:   load_t(e, g, &p("ffn_up.weight"))?,
        down: load_t(e, g, &p("ffn_down.weight"))?,
    }
};

layers.push(HybridLayer {
    attn_norm: load_t(e, g, &p("attn_norm.weight"))?,
    post_attn_norm: load_opt(e, g, &p("post_attention_norm.weight"))?
        .or(load_opt(e, g, &p("ffn_norm.weight"))?)
        .expect("need post_attention_norm or ffn_norm"),
    mixer,
    ffn,
});
```

> **VRAM note:** all 41 layers' exps are Q6_K/Q8_0 → the Q6_K_XL model is 33 GB > 24 GB. Resident-on-GPU won't fit on the 5090. For the Stage-1 correctness gate, run with weights spilling to host or use a partial-layer check; the *math* below is what the gate validates, independent of residency. (The 24 GB-fitting IQ4_XS path requires the IQ kernels — see §5.)

---

## 2. The `moe_ffn` forward (new method in `hybrid_forward.rs`)

`T=1` decode (and small prefill) does router top-k on the host — 256 floats, trivial. Router matmul goes through `e.matmul` (Float → cuBLASLt). Each expert is one `qmatvec` on its byte-sliced row range. The `expert_slice` returns a `Range<usize>`; I slice the resident `CudaSlice<u8>` to a `CudaView<u8>` and pass it to a small `qmatvec_view` (added in §4). Weighted accumulation reuses existing `mul`/`add` plus one tiny scalar-axpy kernel (§4).

```rust
// crates/bw24-engine/src/hybrid_forward.rs
use crate::hybrid::{HybridModel, Mixer, FullAttnLayer, LinearAttnLayer, Ffn, MoeWeights};

impl HybridModel {
    /// MoE FFN block. `z` = post_attention_norm(x1), token-major [T, n_embd].
    /// Returns ffn_out [T, n_embd] = routed-moe + sigmoid-gated shared expert.
    /// Router math is byte-for-byte the qwen35moe path: softmax-over-256 -> top-8 ->
    /// renorm (clamp 6.103515625e-5) -> per-expert SwiGLU -> weighted sum; shared added on top.
    fn moe_ffn(&self, e: &Engine, m: &MoeWeights, z: &CudaSlice<f32>, t: usize)
               -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let moe = cfg.moe.as_ref().unwrap();
        let n_embd   = cfg.n_embd as usize;
        let n_expert = moe.expert_count as usize;        // 256
        let n_used   = moe.expert_used_count as usize;   // 8
        let n_ff_exp = moe.expert_ff_length as usize;    // 512

        // ---- 1. router logits = gate_inp @ z  -> [T, n_expert] (F32 path) ----
        let logits = e.matmul(&m.gate_inp, z, t)?;        // [T, 256]
        let logits_h = e.dtoh(&logits)?;

        // accumulator for routed output, [T, n_embd]
        let mut moe_out = e.zeros(t * n_embd)?;

        // process token by token (T is tiny in decode; prefill loops T times)
        for tok in 0..t {
            let lg = &logits_h[tok * n_expert..(tok + 1) * n_expert];

            // ---- 2. softmax over ALL 256 (numerically stable) BEFORE top-k ----
            let maxl = lg.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut probs = vec![0f32; n_expert];
            let mut denom = 0f32;
            for i in 0..n_expert { let ex = (lg[i] - maxl).exp(); probs[i] = ex; denom += ex; }
            for p in probs.iter_mut() { *p /= denom; }

            // ---- 3. top-8 by prob (selection uses the same unbiased probs; no exp_probs_b) ----
            let mut idx: Vec<usize> = (0..n_expert).collect();
            idx.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
            let sel = &idx[..n_used];

            // ---- 4. gather selected probs as weights, ---- 5./6. renorm to sum 1 (norm_topk_prob)
            let mut w: Vec<f32> = sel.iter().map(|&i| probs[i]).collect();
            let mut wsum: f32 = w.iter().sum();
            wsum = wsum.max(6.103515625e-5);              // clamp to F16 min, matches llama.cpp
            for x in w.iter_mut() { *x /= wsum; }
            // ---- 7. w_scale skipped: expert_weights_scale == 0.0 for this model ----

            // per-token hidden slice [n_embd] (z is token-major)
            let zt = z.slice(tok * n_embd..(tok + 1) * n_embd);

            // ---- per selected expert: y_e = down_e( silu(gate_e@h) * up_e@h ); moe += w_e * y_e ----
            for (j, &ex) in sel.iter().enumerate() {
                // gate_e, up_e : [n_ff_exp]
                let (gr, gin, gout, gqt, grb) = m.gate_exps.expert_slice(ex);
                let (ur, uin, uout, uqt, urb) = m.up_exps.expert_slice(ex);
                let gate = e.qmatvec_view(&m.gate_exps, gr, &zt, 1, gin, gout, gqt, grb)?;
                let up   = e.qmatvec_view(&m.up_exps,   ur, &zt, 1, uin, uout, uqt, urb)?;
                let mut act = e.zeros(n_ff_exp)?;
                e.silu_mul(&gate, &up, &mut act, n_ff_exp)?;        // silu(gate)*up
                // down_e : [n_embd]
                let (dr, din, dout, dqt, drb) = m.down_exps.expert_slice(ex);
                let y = e.qmatvec_view(&m.down_exps, dr, &act, 1, din, dout, dqt, drb)?;
                // moe_out[tok] += w[j] * y
                let dst = moe_out.slice(tok * n_embd..(tok + 1) * n_embd);
                e.axpy_into(&y, w[j], &dst, n_embd)?;               // dst += alpha*y, in place
            }
        }

        // ---- SHARED EXPERT (always-on, sigmoid scalar gate), on the SAME z ----
        // sg = down_shexp( silu(gate_shexp@z) * (up_shexp@z) )  -> [T, n_embd]
        let n_ff_sh = m.gate_shexp.out_features();                 // 512
        let sg_gate = e.matmul(&m.gate_shexp, z, t)?;              // [T,512] (Q8_0 qmatvec)
        let sg_up   = e.matmul(&m.up_shexp,   z, t)?;
        let mut sg_act = e.zeros(t * n_ff_sh)?;
        e.silu_mul(&sg_gate, &sg_up, &mut sg_act, t * n_ff_sh)?;
        let shexp = e.matmul(&m.down_shexp, &sg_act, t)?;          // [T, n_embd]

        // scalar gate g = sigmoid(gate_inp_shexp . z) per token -> [T,1]
        let gate_scalar = e.matmul(&m.gate_inp_shexp, z, t)?;     // F32, out_f==1 -> [T,1]
        let mut g_sig = e.zeros(t)?;
        e.sigmoid(&gate_scalar, &mut g_sig, t)?;

        // ffn_out = moe_out + shexp * broadcast(g_sig)  (per-token scalar broadcast over n_embd)
        e.add_scaled_rows(&shexp, &g_sig, &mut moe_out, n_embd, t)?; // moe_out += shexp[r]*g_sig[r]
        Ok(moe_out)
    }
}
```

**Why host top-k is correct & sufficient:** softmax/top-8/renorm operate on 256 floats per token; in decode `T=1`, this is one pass. It reproduces llama.cpp `build_moe_ffn` exactly: softmax-over-all-256 first, argsort top-8 on those probs, gather the unbiased probs, renorm with the F16-min clamp, no `w_scale`. The expert SwiGLU and the sigmoid-gated shared expert match qwen35moe.cpp:496-547.

---

## 3. Plug into `hybrid_forward.rs::forward` (the FFN branch)

Replace lines 35-46 (the dense SwiGLU block) with a match on `layer.ffn`:

```rust
// pre-FFN norm (post_attention_norm)
let mut z = e.zeros(t * n_embd)?;
e.rms_norm(&x1, layer.post_attn_norm.float_data(), &mut z, n_embd, t, eps)?;

let ffn_out = match &layer.ffn {
    Ffn::Dense { gate, up, down } => {
        let n_ff = gate.out_features();
        let g  = e.matmul(gate, &z, t)?;
        let u  = e.matmul(up,   &z, t)?;
        let mut act = e.zeros(t * n_ff)?;
        e.silu_mul(&g, &u, &mut act, t * n_ff)?;
        e.matmul(down, &act, t)?
    }
    Ffn::Moe(m) => self.moe_ffn(e, m, &z, t)?,
};

// residual 2
let mut x2 = e.zeros(t * n_embd)?;
e.add(&x1, &ffn_out, &mut x2, t * n_embd)?;   // ffn_residual = x1 (pre-norm), Qwen3.5/3.6 style
x = x2;
```

Residual wiring is unchanged: `ffn_out` is added back to `x1` (the post-attn-residual, pre-post-norm tensor), exactly as the existing dense path and qwen35moe.cpp.

Also: `HybridModel::load` already drops the moe assert (§1c). The dispatcher that picks dense vs hybrid path must route `Arch::Qwen35Moe` to `HybridModel` (it already would, since `is_hybrid()` is true for `Qwen35Moe`). `Model::load_dense` keeps its `assert!(cfg.moe.is_none())` (dense path stays MoE-free).

---

## 4. New engine helpers (lib.rs) + one tiny kernel

Three additions. Two are pure Rust wrappers (no new kernel); one needs a tiny CUDA kernel for the per-token scalar-broadcast add. `axpy_into` can reuse an existing kernel if you prefer, but a 3-line kernel is cleanest.

```rust
// crates/bw24-engine/src/lib.rs  (impl Engine)

/// qmatvec on a byte sub-range of a stacked quant tensor (one expert's matrix).
/// `range` is the byte offset into the tensor's resident CudaSlice<u8>.
pub fn qmatvec_view(&self, w: &crate::model::GpuTensor, range: std::ops::Range<usize>,
                    x: &cudarc::driver::CudaView<f32>, m: usize, in_f: usize, out_f: usize,
                    qtype: i32, row_bytes: usize)
                    -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
    use crate::model::GpuTensor;
    let bytes = match w { GpuTensor::Quant { bytes, .. } => bytes, _ => panic!("qmatvec_view on Float") };
    let wv = bytes.slice(range);                       // CudaView<u8> of expert e
    let f = self.func("qmatvec_f32");
    let mut y = self.gpu.stream.alloc_zeros::<f32>(m * out_f)?;
    let cfg = LaunchConfig { grid_dim: (out_f as u32, m as u32, 1), block_dim: (256,1,1), shared_mem_bytes: 0 };
    let (inf, outf, mi, qt, rb) = (in_f as i32, out_f as i32, m as i32, qtype, row_bytes as i64);
    let mut b = self.gpu.stream.launch_builder(&f);
    b.arg(&wv).arg(x).arg(&mut y).arg(&inf).arg(&outf).arg(&mi).arg(&qt).arg(&rb);
    unsafe { b.launch(cfg)?; }
    Ok(y)
}
```

> Note: `x` here is a `CudaView<f32>` (the per-token slice `zt`). The `qmatvec_f32` kernel reads `x + t*in_f`; with `m=1` and a length-`in_f` view, `t=0` indexes correctly. Activation (`act`) for `down_e` is a full `CudaSlice<f32>`; wrap it with `.slice(0..n_ff_exp)` to pass as a view, or add a `CudaSlice` overload. Both `&CudaSlice` and `&CudaView` satisfy `launch_builder.arg`.

```rust
/// dst[0..n] += alpha * src[0..n]  (in-place AXPY into a device view). For weighted expert sum.
pub fn axpy_into(&self, src: &CudaSlice<f32>, alpha: f32,
                 dst: &cudarc::driver::CudaView<f32>, n: usize)
                 -> Result<(), Box<dyn std::error::Error>> {
    let f = self.func("axpy_f32");
    let cfg = LaunchConfig::for_num_elems(n as u32);
    let (a, ni) = (alpha, n as i32);
    let mut b = self.gpu.stream.launch_builder(&f);
    b.arg(src).arg(dst).arg(&a).arg(&ni);
    unsafe { b.launch(cfg)?; }
    Ok(())
}

/// dst[r*ncols + c] += src[r*ncols + c] * scale[r]  — per-row scalar broadcast add.
/// Used for shared-expert: moe_out += shexp * sigmoid(gate)[token].
pub fn add_scaled_rows(&self, src: &CudaSlice<f32>, scale: &CudaSlice<f32>,
                       dst: &mut CudaSlice<f32>, ncols: usize, nrows: usize)
                       -> Result<(), Box<dyn std::error::Error>> {
    let f = self.func("add_scaled_rows_f32");
    let cfg = LaunchConfig::for_num_elems((ncols * nrows) as u32);
    let (nc, nr) = (ncols as i32, nrows as i32);
    let mut b = self.gpu.stream.launch_builder(&f);
    b.arg(src).arg(scale).arg(dst).arg(&nc).arg(&nr);
    unsafe { b.launch(cfg)?; }
    Ok(())
}
```

CUDA kernels (append to the hybrid fatbin `.cu`, or wherever `add_f32`/`mul_f32` live so `func()` finds them):

```cuda
// elementwise AXPY into existing dst (dst is a sub-view; kernel sees a base ptr of length n).
extern "C" __global__ void axpy_f32(const float* __restrict__ src, float* __restrict__ dst,
                                    float alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += alpha * src[i];
}

// dst[r*ncols+c] += src[r*ncols+c] * scale[r]  (per-row scalar broadcast).
extern "C" __global__ void add_scaled_rows_f32(const float* __restrict__ src,
                                               const float* __restrict__ scale,
                                               float* __restrict__ dst, int ncols, int nrows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = ncols * nrows;
    if (i < n) { int r = i / ncols; dst[i] += src[i] * scale[r]; }
}
```

No top-k kernel needed (host). No softmax kernel needed (host, 256 floats). The only genuinely new device code is `axpy_f32` + `add_scaled_rows_f32` (both trivial); `qmatvec_view` reuses the existing `qmatvec_f32`.

---

## 5. Validation plan (argmax-match llama.cpp, same gate as the hybrid path)

**Ground truth.** Use the existing dumper `tools/llama_logits.cpp`:
```
llama_logits /home/avifenesh/ai-ml/hf-models/qwen36-35b-mtp/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf <tok0> <tok1> ...
```
It prints argmax + top-5 of the last-token logits with `n_gpu_layers=999`. Use the **same fixed token IDs** that the prior hybrid validation used (no tokenizer in the loop) so the comparison is apples-to-apples.

**Model choice (decisive):** Use **Q6_K_XL** (`.../qwen36-35b-mtp/...UD-Q6_K_XL.gguf`). Its exps are Q6_K (gate/up) + Q8_0 (down), and all shexp are Q8_0 — every dtype is already in qmatvec.cu (`QT_Q6_K`, `QT_Q8_0`) and `dequant.rs`. **Zero new quant kernels.** The IQ4_XS file's exps are IQ3_S/IQ4_XS — **unsupported by qmatvec.cu and dequant.rs**; using it requires adding IQ3_S+IQ4_XS dequant (dequant.rs) and qmatvec (qmatvec.cu QType + deq + dp4a) paths first. Do NOT attempt the IQ4_XS argmax gate until those kernels exist; it will panic in `load_exps`.

**Gate procedure (mirrors the hybrid gate that passed):**
1. Run `HybridModel::forward_last` on the fixed tokens; take argmax of the returned last-token logits.
2. Run `llama_logits` on the identical tokens.
3. **Pass = top-1 token matches.** Also compare the **top-5 ordering** — a subtly-wrong renorm or a softmax-after-topk bug shifts near-ties without always flipping top-1, so top-5 catches it.
4. If mismatch, bisect by layer: dump `z` (post-attn-norm input to FFN) and `ffn_out` at layer 0 vs llama.cpp `ffn_moe_*` cb tensors. The usual culprits, in order:
   - softmax done **after** top-8 instead of before (the #1 trap),
   - missing/!=F16-min-clamped renorm,
   - applied a `w_scale` (must be skipped, `expert_weights_scale==0`),
   - forgot the shared expert or its `sigmoid` scalar gate,
   - wrong expert slice (`down_exps` is `[n_ff_exp, n_embd, n_expert]` → `in=512, out=2048`, transposed vs gate/up).

**Isolation micro-checks before the full gate (cheap, catch slicing bugs early):**
- Dequantize expert `e=0`'s `gate_exps` slice on host (via `dequant.rs`) and compare a `qmatvec_view` result against a host f32 matmul for a random `z` — verifies `load_exps` row_bytes + `expert_slice` offsets.
- Verify the router: host-softmax of `gate_inp @ z` top-8 indices == llama.cpp `ffn_moe_topk` for layer 0.

**Stage-A only for the gate.** Do NOT set `BW24_FAST`. The shared-expert `e.matmul` and any future fast path go through dp4a; the correctness gate must use the validated `qmatvec_f32` dequant path (`matmul` falls back to it when `BW24_FAST` is unset, and `qmatvec_view` always uses `qmatvec_f32`). Add the Stage-B fast MoE only after the argmax gate is green.

---

## Files touched (all absolute)

- `/home/avifenesh/projects/bw24/crates/bw24-gguf/src/config.rs` — add `MoeConfig::expert_shared_ff_length`.
- `/home/avifenesh/projects/bw24/crates/bw24-engine/src/model.rs` — `GpuTensor::load_exps`, `GpuTensor::expert_slice`.
- `/home/avifenesh/projects/bw24/crates/bw24-engine/src/hybrid.rs` — `MoeWeights`, `Ffn` enum, `HybridLayer.ffn`, loader branch (remove the `moe.is_none()` assert at line 56).
- `/home/avifenesh/projects/bw24/crates/bw24-engine/src/hybrid_forward.rs` — `moe_ffn` method + FFN match in `forward` (replaces dense block lines 35-46).
- `/home/avifenesh/projects/bw24/crates/bw24-engine/src/lib.rs` — `qmatvec_view`, `axpy_into`, `add_scaled_rows`.
- `/home/avifenesh/projects/bw24/crates/bw24-engine/cu/` — add `axpy_f32` + `add_scaled_rows_f32` to the hybrid `.cu` (the build wires it into `BW24_HYBRID_FATBIN`).

**Key correctness invariants (the load-bearing math):** softmax over all 256 → argsort top-8 on those probs → gather unbiased probs → renorm by `max(sum, 6.103515625e-5)` → no `w_scale` → `moe = Σ wⱼ·downⱼ(silu(gateⱼ@h)·upⱼ@h)` → `out = moe + sigmoid(gate_inp_shexp·h)·down_sh(silu(gate_sh@h)·up_sh@h)`; residual adds `out` to the pre-post-norm tensor `x1`.