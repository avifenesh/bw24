Confirmed: `&CudaView<u8>` implements `PushKernelArg`, so I can slice the stacked expert tensor's `CudaSlice<u8>` into a per-expert `CudaView` and pass it to a qmatvec variant that accepts a view. This is the clean Stage-1 approach.

I now have full grounding. Here is the complete implementation.

---

# bw24 MoE FFN (qwen35moe / Qwen3.6-35B-A3B) — Complete Implementation

Grounded against the real codebase: `hybrid.rs`, `hybrid_forward.rs`, `lib.rs` (`Engine`), `model.rs` (`GpuTensor`), `config.rs` (`MoeConfig`), `qmatvec.cu`, `bw24-gguf/lib.rs`. Router math taken verbatim from the verified llama.cpp trace in the research.

## Critical findings that shape the design

1. **`Engine::qmatvec*` take `&CudaSlice<u8>`, not a view.** To slice expert `e` out of a stacked tensor I add **view-accepting wrappers** (`qmatvec_view` / `qmatvec_q8_0_fast_view` / `qmatvec_q6_K_fast_view`). cudarc 0.19.8 implements `PushKernelArg<&CudaView<u8>>` (confirmed in `launch.rs:135`), and `CudaSlice::slice(range) -> CudaView` exists (`lib.rs:1742`). The kernels are unchanged — they already index `W + o*row_bytes` from arg `W`, so a sub-view base is sufficient.

2. **IQ4_XS (type 23) is not handled.** `GpuTensor::load` maps only Q8_0/Q4_K/Q6_K to `Quant`; everything else goes to `Float` via `dequant::dequantize`, which **panics on IQ4_XS** (not implemented). So the **IQ4_XS GGUF cannot be loaded as-is** for the expert tensors. Two options, both covered below: **(A)** validate first on the **Q6_K_XL** GGUF (`/data/ai-ml/hf-models/qwen36-35b-mtp/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf`) where gate/up=Q6_K and down=Q8_0 are already supported — **this is the recommended Stage-1 gate**; **(B)** add an IQ4_XS dequant + qtype to unblock the IQ4_XS file (sketch in §5). The plan keeps the forward dtype-agnostic via the matmul dispatch so both work once the dtype is supported.

3. **Router (`ffn_gate_inp`) and shared gate (`ffn_gate_inp_shexp`) are F32** → load as `GpuTensor::Float`, run through `Engine::linear` (cuBLASLt). The shared gate is `ne=[2048]` (1D) → `out_features()` reads `ne[1]` which doesn't exist; handle it as a raw F32 dot (see `MoeLayer::shared_gate`).

4. **Top-k is tiny (256 logits, k=8)** → do softmax + argsort + renorm **on host** in Stage-1. Matches the "host-side top-k is fine" guidance and avoids a new kernel.

---

## 1. Config: add the shared-expert FF length

```rust
// crates/bw24-gguf/src/config.rs  — extend MoeConfig
#[derive(Debug, Clone)]
pub struct MoeConfig {
    pub expert_count: u32,
    pub expert_used_count: u32,
    pub expert_ff_length: u32,
    pub expert_shared_ff_length: u32,   // NEW: qwen35moe.expert_shared_feed_forward_length
}
```
```rust
// in ModelConfig::from_gguf, the `moe = if arch.is_moe()` block:
let moe = if arch.is_moe() {
    Some(MoeConfig {
        expert_count: u("expert_count").unwrap_or(0),
        expert_used_count: u("expert_used_count").unwrap_or(0),
        expert_ff_length: u("expert_feed_forward_length").unwrap_or(0),
        expert_shared_ff_length: u("expert_shared_feed_forward_length").unwrap_or(0), // NEW
    })
} else { None };
```

---

## 2. Weights: `MoeLayer` + load/slice helpers

The stacked expert tensors load through the **existing `GpuTensor::load`** unchanged — a 3D `[in, out, n_expert]` Q6_K/Q8_0 tensor becomes `GpuTensor::Quant { bytes, qtype, row_bytes, ne }`. Note `GpuTensor::load` computes `row_bytes = raw.len() / ne[1]` — for a 3D tensor `raw.len() = n_expert * per_expert_bytes`, and `ne[1] = out_f`, so `row_bytes = n_expert * per_expert_bytes / out_f = (in_f/block)*type_size`, which is the **correct per-row byte stride** (n_expert cancels). So `row_bytes` is right as-is; I only need the **per-expert byte stride** to pick the slice base.

```rust
// crates/bw24-engine/src/hybrid.rs  — new MoeLayer + add to HybridLayer

/// One layer's MoE FFN weights. Routed experts are stacked 3D GGUF tensors kept packed;
/// expert e's 2D matrix is a byte-view at e*per_expert_bytes. Shared expert is plain 2D.
pub struct MoeLayer {
    // router: F32 [n_embd, n_expert] -> logits[n_expert]
    pub gate_inp: GpuTensor,
    // routed experts, stacked: ne=[n_embd, n_ff_exp, n_expert] (gate/up); [n_ff_exp, n_embd, n_expert] (down)
    pub gate_exps: GpuTensor,
    pub up_exps: GpuTensor,
    pub down_exps: GpuTensor,
    // shared expert (always-on, sigmoid-gated)
    pub shexp_gate_inp: GpuTensor, // F32 [n_embd] -> 1 scalar/token
    pub gate_shexp: GpuTensor,     // [n_embd, n_ff_shexp]
    pub up_shexp: GpuTensor,       // [n_embd, n_ff_shexp]
    pub down_shexp: GpuTensor,     // [n_ff_shexp, n_embd]
    // cached dims
    pub n_expert: usize,
    pub n_used: usize,
    pub n_ff_exp: usize,
}

impl MoeLayer {
    /// per-expert byte stride for a stacked tensor = ne[0]*ne[1] elems / block * type_size.
    /// Equivalent to row_bytes(ne[1]) * ne[1] — the contiguous 2D matrix for one expert.
    fn per_expert_bytes(t: &GpuTensor) -> usize {
        match t {
            GpuTensor::Quant { bytes, ne, .. } => {
                let n_expert = ne[2] as usize;
                bytes.len() / n_expert        // total / n_expert = one expert's 2D block
            }
            GpuTensor::Float { ne, .. } => (ne[0] * ne[1]) as usize, // f32 elems (not bytes); see view fn
        }
    }
}

// In HybridLayer: make the FFN either dense OR moe.
pub enum Ffn {
    Dense { gate: GpuTensor, up: GpuTensor, down: GpuTensor },
    Moe(MoeLayer),
}

pub struct HybridLayer {
    pub attn_norm: GpuTensor,
    pub post_attn_norm: GpuTensor,
    pub mixer: Mixer,
    pub ffn: Ffn,            // was: ffn_gate/ffn_up/ffn_down
}
```

```rust
// crates/bw24-engine/src/hybrid.rs  — in HybridModel::load:
//   * REMOVE: assert!(cfg.moe.is_none(), "...");
//   * build the FFN per-layer:

let ffn = if let Some(moe) = cfg.moe.as_ref() {
    Ffn::Moe(MoeLayer {
        gate_inp:       load_t(e, g, &p("ffn_gate_inp.weight"))?,
        gate_exps:      load_t(e, g, &p("ffn_gate_exps.weight"))?,
        up_exps:        load_t(e, g, &p("ffn_up_exps.weight"))?,
        down_exps:      load_t(e, g, &p("ffn_down_exps.weight"))?,
        shexp_gate_inp: load_t(e, g, &p("ffn_gate_inp_shexp.weight"))?,
        gate_shexp:     load_t(e, g, &p("ffn_gate_shexp.weight"))?,
        up_shexp:       load_t(e, g, &p("ffn_up_shexp.weight"))?,
        down_shexp:     load_t(e, g, &p("ffn_down_shexp.weight"))?,
        n_expert:       moe.expert_count as usize,
        n_used:         moe.expert_used_count as usize,
        n_ff_exp:       moe.expert_ff_length as usize,
    })
} else {
    Ffn::Dense {
        gate: load_t(e, g, &p("ffn_gate.weight"))?,
        up:   load_t(e, g, &p("ffn_up.weight"))?,
        down: load_t(e, g, &p("ffn_down.weight"))?,
    }
};
layers.push(HybridLayer { attn_norm, post_attn_norm, mixer, ffn });
```

> Note: this GGUF has **separate** `ffn_gate_exps` + `ffn_up_exps` (no merged `ffn_gate_up_exps`), confirmed for both UD-IQ4_XS and UD-Q6_K_XL. If a merged tensor ever appears, detect it with `load_opt` and split rows `[0:n_ff]`/`[n_ff:2*n_ff]`; not needed for these two files.

---

## 4. New engine methods: per-expert view qmatvec (the only "new kernel" needed = none; just view wrappers)

No new `.cu` kernel is required. The existing `qmatvec_f32` / `qmatvec_q*_dp4a` kernels index `W` from its base, so passing a **sub-view** as `W` selects expert `e`. I add view-accepting wrappers and a helper to build the per-expert view. Weighted-accumulate of the 8 experts is done with the existing `add` + a scalar-scale (a 3-line `scale_f32` kernel, or reuse `mul` against a filled buffer). I include a tiny `axpy`-style host-light path that uses one new trivial kernel `scale_add_f32` for the weighted sum (cleanest); it can also be done with `mul`+`add`.

```rust
// crates/bw24-engine/src/lib.rs  — add to impl Engine

/// qmatvec where W is a CudaView (a per-expert byte slice of a stacked tensor).
pub fn qmatvec_view(&self, w: &cudarc::driver::CudaView<u8>, x: &CudaSlice<f32>, m: usize,
                    in_f: usize, out_f: usize, qtype: i32, row_bytes: usize)
                    -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
    let f = self.func("qmatvec_f32");
    let mut y = self.gpu.stream.alloc_zeros::<f32>(m * out_f)?;
    let cfg = LaunchConfig { grid_dim: (out_f as u32, m as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
    let (inf, outf, mi, qt, rb) = (in_f as i32, out_f as i32, m as i32, qtype, row_bytes as i64);
    let mut b = self.gpu.stream.launch_builder(&f);
    b.arg(w).arg(x).arg(&mut y).arg(&inf).arg(&outf).arg(&mi).arg(&qt).arg(&rb);
    unsafe { b.launch(cfg)?; }
    Ok(y)
}

/// Stage-B fast int8 dp4a variants taking a weight VIEW (for Q8_0 / Q6_K stacked experts).
pub fn qmatvec_q8_0_fast_view(&self, w: &cudarc::driver::CudaView<u8>, x: &CudaSlice<f32>, m: usize,
                              in_f: usize, out_f: usize, row_bytes: usize)
                              -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
    let (aq, ad) = self.quantize_q8_1(x, m, in_f)?;
    let f = self.func("qmatvec_q8_0_dp4a");
    let mut y = self.gpu.stream.alloc_zeros::<f32>(m * out_f)?;
    let cfg = LaunchConfig { grid_dim: (out_f as u32, m as u32, 1), block_dim: (64, 1, 1), shared_mem_bytes: 0 };
    let (inf, outf, mi, rb) = (in_f as i32, out_f as i32, m as i32, row_bytes as i64);
    let mut b = self.gpu.stream.launch_builder(&f);
    b.arg(w).arg(&aq).arg(&ad).arg(&mut y).arg(&inf).arg(&outf).arg(&mi).arg(&rb);
    unsafe { b.launch(cfg)?; }
    Ok(y)
}
pub fn qmatvec_q6_K_fast_view(&self, w: &cudarc::driver::CudaView<u8>, x: &CudaSlice<f32>, m: usize,
                              in_f: usize, out_f: usize, row_bytes: usize)
                              -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
    let (aq, ad) = self.quantize_q8_1(x, m, in_f)?;
    let f = self.func("qmatvec_q6_K_dp4a");
    let mut y = self.gpu.stream.alloc_zeros::<f32>(m * out_f)?;
    let cfg = LaunchConfig { grid_dim: (out_f as u32, m as u32, 1), block_dim: (64, 1, 1), shared_mem_bytes: 0 };
    let (inf, outf, mi, rb) = (in_f as i32, out_f as i32, m as i32, row_bytes as i64);
    let mut b = self.gpu.stream.launch_builder(&f);
    b.arg(w).arg(&aq).arg(&ad).arg(&mut y).arg(&inf).arg(&outf).arg(&mi).arg(&rb);
    unsafe { b.launch(cfg)?; }
    Ok(y)
}

/// y += s * x  (weighted-accumulate for summing 8 expert outputs). s is a host scalar.
pub fn scale_add(&self, x: &CudaSlice<f32>, s: f32, dst: &mut CudaSlice<f32>, n: usize)
                 -> Result<(), Box<dyn std::error::Error>> {
    let f = self.func("scale_add_f32");
    let cfg = LaunchConfig::for_num_elems(n as u32);
    let ni = n as i32;
    let mut b = self.gpu.stream.launch_builder(&f);
    b.arg(x).arg(&s).arg(dst).arg(&ni);
    unsafe { b.launch(cfg)?; }
    Ok(())
}
```

The single tiny new kernel (append to `crates/bw24-engine/cu/qmatvec.cu` or the hybrid `.cu`):

```cuda
// dst[i] += s * x[i]   (weighted accumulate of expert outputs)
extern "C" __global__ void scale_add_f32(const float* __restrict__ x, float s,
                                         float* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += s * x[i];
}
```

And a dispatch helper that picks the right view-qmatvec by the stacked tensor's qtype, slicing expert `e`:

```rust
// crates/bw24-engine/src/hybrid_forward.rs  (or an impl in lib.rs)
use crate::model::GpuTensor;
use crate::{QT_Q8_0, QT_Q6_K, QT_Q4_K};

impl HybridModel {
    /// Run expert e's projection: y[m,out] = h @ W_e^T, where W_e is the e-th 2D slice of `stack`.
    /// `m` is #tokens routed to this expert (Stage-1: m=1, single token at a time).
    fn expert_matvec(e_: &Engine, stack: &GpuTensor, expert: usize, h: &CudaSlice<f32>,
                     m: usize, in_f: usize, out_f: usize)
                     -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        match stack {
            GpuTensor::Quant { bytes, qtype, row_bytes, ne } => {
                let n_expert = ne[2] as usize;
                let per = bytes.len() / n_expert;           // per-expert byte stride
                let off = expert * per;
                let w = bytes.slice(off..off + per);        // CudaView<u8> into expert e
                let fast = std::env::var("BW24_FAST").is_ok();
                match *qtype {
                    QT_Q8_0 if fast => e_.qmatvec_q8_0_fast_view(&w, h, m, in_f, out_f, *row_bytes),
                    QT_Q6_K if fast => e_.qmatvec_q6_K_fast_view(&w, h, m, in_f, out_f, *row_bytes),
                    qt              => e_.qmatvec_view(&w, h, m, in_f, out_f, qt, *row_bytes),
                }
            }
            GpuTensor::Float { .. } => unreachable!("expert stack must be quantized"),
        }
    }
}
```

> The shared expert (`gate_shexp`/`up_shexp`/`down_shexp`, plain 2D `Quant`) just uses the existing `Engine::matmul` directly.

---

## 2+3. The MoE forward + where it plugs into `hybrid_forward.rs`

Replace the dense-FFN block (lines 35–46) with a dispatch on `layer.ffn`:

```rust
// crates/bw24-engine/src/hybrid_forward.rs  — inside the per-layer loop, after residual-1 (x1):

// pre-FFN norm (post_attention_norm)
let mut z = e.zeros(t * n_embd)?;
e.rms_norm(&x1, layer.post_attn_norm.float_data(), &mut z, n_embd, t, eps)?;

let ffn_out = match &layer.ffn {
    Ffn::Dense { gate, up, down } => {
        let n_ff = gate.out_features();
        let g = e.matmul(gate, &z, t)?;
        let u = e.matmul(up, &z, t)?;
        let mut act = e.zeros(t * n_ff)?;
        e.silu_mul(&g, &u, &mut act, t * n_ff)?;
        e.matmul(down, &act, t)?
    }
    Ffn::Moe(moe) => self.moe_ffn(e, moe, &z, t, n_embd)?,
};

let mut x2 = e.zeros(t * n_embd)?;
e.add(&x1, &ffn_out, &mut x2, t * n_embd)?;   // residual to PRE-post-norm tensor (x1) — correct
x = x2;
```

The MoE forward itself (router math verbatim from the verified trace):

```rust
// crates/bw24-engine/src/hybrid_forward.rs  — new method on HybridModel

use crate::hybrid::{Ffn, MoeLayer};

impl HybridModel {
    /// MoE FFN. z = post_attention_norm(h) [T, n_embd] token-major. Returns ffn_out [T, n_embd].
    /// Router: logits = gate_inp@z; softmax over ALL n_expert; top-k; gather; renorm. Then
    /// 8 experts (silu(gate)*up -> down, weighted) + always-on sigmoid-gated shared expert.
    fn moe_ffn(&self, e: &Engine, moe: &MoeLayer, z: &CudaSlice<f32>, t: usize, n_embd: usize)
               -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let n_expert = moe.n_expert;     // 256
        let n_used   = moe.n_used;       // 8
        let n_ff     = moe.n_ff_exp;     // 512
        let min_w = 6.103515625e-5f32;   // F16 min (graph.cpp:1633 clamp)

        // ---- router logits = gate_inp @ z : [T, n_expert] (gate_inp is F32 -> cuBLASLt linear) ----
        let logits = e.matmul(&moe.gate_inp, z, t)?;    // out_features = n_expert
        let logits_h = e.dtoh(&logits)?;

        // accumulate per-token MoE output on device
        let mut moe_out = e.zeros(t * n_embd)?;

        for tok in 0..t {
            let lg = &logits_h[tok * n_expert..(tok + 1) * n_expert];

            // 1) softmax over ALL n_expert (BEFORE top-k)
            let maxl = lg.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut probs = vec![0f32; n_expert];
            let mut sum = 0f32;
            for i in 0..n_expert { let ex = (lg[i] - maxl).exp(); probs[i] = ex; sum += ex; }
            for p in probs.iter_mut() { *p /= sum; }

            // 2) top-k (8 HIGHEST probs), descending — argsort then take first k.
            //    Tie order: llama.cpp argsort is stable on index for equal keys; match by using
            //    a stable sort with index as the tiebreaker (lower index first), descending on prob.
            let mut idx: Vec<usize> = (0..n_expert).collect();
            idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap()
                                 .then(a.cmp(&b)));      // stable: smaller index wins ties
            let sel = &idx[..n_used];

            // 3) gather the selected softmax probs as weights, then renorm-to-sum-1 (norm_topk_prob)
            let mut w: Vec<f32> = sel.iter().map(|&i| probs[i]).collect();
            let mut wsum: f32 = w.iter().sum();
            wsum = wsum.max(min_w);                       // clamp BEFORE divide
            for x in w.iter_mut() { *x /= wsum; }         // NO routed_scaling_factor (w_scale==0)

            // single-token hidden slice [1, n_embd] for the expert matvecs
            let h_tok = e.htod(&e.dtoh(z)?[tok * n_embd..(tok + 1) * n_embd])?;  // [1, n_embd]
            // (Stage-1 simplicity; decode path with t=1 skips this round-trip.)

            // 4) 8 experts: y_e = down_e(silu(gate_e@h)*up_e@h); moe_out += w_e * y_e
            for (j, &expert) in sel.iter().enumerate() {
                let g_e = Self::expert_matvec(e, &moe.gate_exps, expert, &h_tok, 1, n_embd, n_ff)?;
                let u_e = Self::expert_matvec(e, &moe.up_exps,   expert, &h_tok, 1, n_embd, n_ff)?;
                let mut act = e.zeros(n_ff)?;
                e.silu_mul(&g_e, &u_e, &mut act, n_ff)?;          // silu(gate)*up
                let y_e = Self::expert_matvec(e, &moe.down_exps, expert, &act, 1, n_ff, n_embd)?;
                // weighted accumulate into this token's slice of moe_out
                let mut view = moe_out.slice_mut(tok * n_embd..(tok + 1) * n_embd);
                // scale_add over the view: dst += w[j]*y_e  (use a contiguous temp then dtod, or
                //  pass the view to a view-aware scale_add). Simplest: accumulate into a per-token
                //  buffer then copy. Shown here with a temp:
                drop(view);
                e.scale_add(&y_e, w[j], &mut /*per-token accum*/ moe_out_tok_buf(tok), n_embd)?;
            }

            // 5) shared expert (always-on, sigmoid-gated SCALAR):
            //    sh = down_shexp(silu(gate_shexp@h) * up_shexp@h); sg = sigmoid(gate_inp_shexp·h); sh*=sg
            let sg = e.matmul(&moe.gate_shexp, &h_tok, 1)?;       // [n_ff_shexp]
            let su = e.matmul(&moe.up_shexp,   &h_tok, 1)?;
            let n_ff_sh = moe.gate_shexp.out_features();
            let mut sact = e.zeros(n_ff_sh)?;
            e.silu_mul(&sg, &su, &mut sact, n_ff_sh)?;
            let sh = e.matmul(&moe.down_shexp, &sact, 1)?;        // [n_embd]
            // scalar gate = sigmoid(dot(shexp_gate_inp[n_embd], h)). shexp_gate_inp is F32 [n_embd].
            let gate_scalar = e.dtoh(&sh_gate_dot(e, &moe.shexp_gate_inp, &h_tok, n_embd)?)?[0];
            let sg_val = 1.0f32 / (1.0f32 + (-gate_scalar).exp());
            e.scale_add(&sh, sg_val, &mut moe_out_tok_buf(tok), n_embd)?;  // moe_out += sigmoid*shared
        }
        Ok(moe_out)   // residual add happens in the caller (add to x1)
    }
}
```

**Cleaner accumulation** (avoids the `moe_out_tok_buf` pseudocode): since Stage-1 decode runs **one token at a time** (`t=1` in the decode loop), accumulate into a single `acc = zeros(n_embd)` and write it back. For the prefill (`t>1`) path, allocate `acc` per token. Concretely:

```rust
let mut acc = e.zeros(n_embd)?;                     // per-token accumulator
for (j, &expert) in sel.iter().enumerate() {
    let g_e = Self::expert_matvec(e, &moe.gate_exps, expert, &h_tok, 1, n_embd, n_ff)?;
    let u_e = Self::expert_matvec(e, &moe.up_exps,   expert, &h_tok, 1, n_embd, n_ff)?;
    let mut act = e.zeros(n_ff)?;
    e.silu_mul(&g_e, &u_e, &mut act, n_ff)?;
    let y_e = Self::expert_matvec(e, &moe.down_exps, expert, &act, 1, n_ff, n_embd)?;
    e.scale_add(&y_e, w[j], &mut acc, n_embd)?;     // acc += w_j * y_e
}
// shared expert -> acc += sigmoid(gate)*shared (as above)
e.scale_add(&sh, sg_val, &mut acc, n_embd)?;
// write acc into moe_out[tok]
e.copy_into(&mut moe_out, tok * n_embd, &acc, n_embd)?;   // existing dtod helper
```

The scalar shared-gate dot (`shexp_gate_inp` is 1D F32 `[n_embd]`):

```rust
/// sigmoid-gate scalar input: returns a [1] device buffer = shexp_gate_inp · h.
fn sh_gate_dot(e: &Engine, gate_inp: &GpuTensor, h: &CudaSlice<f32>, n_embd: usize)
               -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
    // gate_inp is Float [n_embd] (ne=[2048]); treat as a 1xN matrix -> out_f=1.
    let w = gate_inp.float_data();          // CudaSlice<f32> length n_embd
    e.linear(h, w, 1, n_embd, 1)            // y[1,1] = h[1,n_embd] @ w[1,n_embd]^T
}
```

> `Engine::linear` maps `out_f=1, in_f=n_embd`, weights row-major `[1, n_embd]` — exactly the dot product. The result `[1]` is the scalar logit; sigmoid is applied host-side (one value).

---

## Exact router math (the correctness-critical part), restated as implemented

1. `logits = ffn_gate_inp @ z` over all 256 experts (F32 cuBLASLt).
2. `probs = softmax(logits)` over **all 256** — **before** top-k.
3. `idx = argsort_desc(probs)`, stable tiebreak on lower index; `sel = idx[..8]` (8 highest).
4. `w[j] = probs[sel[j]]` (gather the softmax-over-256 probs of the 8 selected).
5. `wsum = max(sum(w), 6.103515625e-5)`; `w /= wsum` (norm_topk_prob). **No** `routed_scaling_factor`, **no** `exp_probs_b`, **no** expert groups, **no** softmax-after-topk.
6. Per expert: `y_e = down_e(silu(gate_e@h) * up_e@h)`; `moe_out += w[j] * y_e` (weight applied **after** down).
7. Shared: `sh = down_shexp(silu(gate_shexp@h)*up_shexp@h)`; `sg = sigmoid(shexp_gate_inp·h)` scalar; `moe_out += sg * sh`.
8. Caller: `x_next = x1 + moe_out` (residual to the **pre-post-norm** tensor). Skip blk.40 (MTP) — the loop already runs only `0..n_layer` (=40 trunk layers), MTP is `nextn_predict_layers` and not in `self.layers`.

---

## 5. dtype paths & the IQ4_XS gap

| tensor | Q6_K_XL file | IQ4_XS file | qmatvec path |
|---|---|---|---|
| `ffn_gate_inp` (router) | F32 | F32 | `Float` → `linear` (cuBLASLt) ✅ |
| `ffn_gate_exps` / `ffn_up_exps` | Q6_K | IQ4_XS | Q6_K→`qmatvec_view`/`q6_K_fast_view` ✅; **IQ4_XS unsupported** ✗ |
| `ffn_down_exps` | Q8_0 | Q8_0 | `qmatvec_view`(Q8_0) / `q8_0_fast_view` ✅ |
| `ffn_*_shexp` | Q8_0 | Q8_0 | `matmul` (existing Quant path) ✅ |
| `ffn_gate_inp_shexp` | F32 | F32 | `linear` (out_f=1) ✅ |

**For the IQ4_XS file**, add IQ4_XS support before it can load:
- Add `QT_IQ4_XS: i32 = 3` in `lib.rs`; map `GgmlType::IQ4_XS => Some(QT_IQ4_XS)` in `GpuTensor::load`.
- Add `deq_iq4_xs(row, j)` to `qmatvec.cu` (256-elem superblock, 136 bytes: `fp16 d`, `u16 scales_h`, `u8 scales_l[4]`, `u8 qs[128]`; values via the IQ4_NL 16-entry lookup `kvalues_iq4nl`, 6-bit per-32 sub-scales). Wire it into the `deq()` switch.
- Add a matching `dequant_iq4_xs` in `bw24-gguf/dequant.rs` for the CPU oracle.
This is a separate ~60-line addition; **not required to land/validate the MoE forward** — do the gate on Q6_K_XL first.

---

## VALIDATION PLAN (argmax-match llama.cpp, same gate that passed the hybrid forward)

**Gate model:** `/data/ai-ml/hf-models/qwen36-35b-mtp/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf` (all expert dtypes already supported — no IQ4_XS work needed to validate router math). 33 GB fits in 32 GB VRAM only tightly; if OOM, validate on IQ4_XS **after** adding the IQ4_XS dequant (§5), since the IQ4_XS file is 18 GB.

1. **bw24 side** — extend `run_hybrid.rs` (it already loads `HybridModel`, runs `forward_last`, prints argmax + top-5). After removing the `moe.is_none()` assert and wiring `Ffn::Moe`, it works unchanged:
   ```
   cargo run -q --bin run-hybrid --release -- \
     /data/ai-ml/hf-models/qwen36-35b-mtp/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf 1 2 3 4
   ```
   Use the **Stage-A correctness path** first (no `BW24_FAST`) so experts go through `qmatvec_view` (exact f32 dequant), matching the validated dense/hybrid oracle. Then re-run with `BW24_FAST=1` and require the argmax to be identical (the dp4a int8 path must not change the top token).

2. **llama.cpp ground truth** — same fixed tokens, last-token logits. The repo has `llama-eval-callback` built; use it (or build `llama-logits` if present) to dump the final-layer logits for token sequence `[1,2,3,4]`:
   ```
   /home/avifenesh/projects/llama.cpp/build/bin/llama-eval-callback \
     -m .../Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf -p "" --tokens 1 2 3 4   # or the project's llama_logits wrapper
   ```
   (Match whatever wrapper the hybrid gate used in task #9; the research references `tools/llama_logits`.)

3. **Pass criteria** (same bar as the hybrid argmax gate, task #9):
   - **argmax token identical** to llama.cpp for the fixed input (primary gate).
   - **top-5 token set identical** (ordering may differ only on near-ties).
   - **non-finite count == 0** (already asserted in `run_hybrid.rs`).
   - Stage-A and Stage-B (`BW24_FAST`) produce the **same argmax**.

4. **Isolation sub-gates** (bisect like task #9 if argmax mismatches), each comparable against a CPU reference built from `dequant::dequantize` + the router pseudocode:
   - **Router only**: dump `probs[256]` and the 8 selected indices for layer 0, token 0; compare to a hand-computed softmax-then-top-k from the same logits. Catches the softmax-before-topk vs after-topk error (the #1 pitfall).
   - **Single expert matvec**: compare `expert_matvec(gate_exps, e=sel[0])` against `dequantize(Q6_K, expert-e bytes)` @ h on CPU (verifies the per-expert byte-slice offset `e*per_expert_bytes` and `row_bytes`).
   - **Renorm**: assert `sum(w)==1.0` (±1e-6) after renorm.
   - **Shared expert**: assert it is added every token (disable it and confirm argmax changes), and that the gate is `sigmoid(scalar)`, not softmax.

5. **Decode-loop check**: run `run_gen` for ~16 tokens; output must be coherent text (the existing greedy loop in `decode.rs` calls the same forward, so MoE rides along once `Ffn::Moe` is wired).

---

### Files to edit
- `/home/avifenesh/projects/bw24/crates/bw24-gguf/src/config.rs` — add `expert_shared_ff_length` to `MoeConfig` + load key.
- `/home/avifenesh/projects/bw24/crates/bw24-engine/src/hybrid.rs` — add `MoeLayer`, `Ffn` enum; replace `HybridLayer.ffn_*` with `ffn: Ffn`; remove `assert!(cfg.moe.is_none())`; load MoE tensors.
- `/home/avifenesh/projects/bw24/crates/bw24-engine/src/hybrid_forward.rs` — dispatch FFN on `Ffn`; add `moe_ffn`, `expert_matvec`, `sh_gate_dot`.
- `/home/avifenesh/projects/bw24/crates/bw24-engine/src/lib.rs` — add `qmatvec_view`, `qmatvec_q8_0_fast_view`, `qmatvec_q6_K_fast_view`, `scale_add`.
- `/home/avifenesh/projects/bw24/crates/bw24-engine/cu/qmatvec.cu` — add `scale_add_f32` kernel (and IQ4_XS deq only if validating the IQ4_XS file).
- (only for IQ4_XS file) `/home/avifenesh/projects/bw24/crates/bw24-gguf/src/dequant.rs` + `model.rs` qtype mapping.

### Key correctness anchors enforced in the code above
Softmax over all 256 **before** top-k; stable descending argsort; renorm with `6.103515625e-5` clamp; **no** scaling factor / bias / groups; weight applied **after** the down projection; shared expert always-on with a **scalar sigmoid** gate; residual added to the **pre-post-norm** tensor (`x1`); MTP block (blk.40) excluded (loop runs `0..n_layer`).

One caveat to flag: the `moe_ffn` sketch does a `dtoh(z)` per token to build `h_tok`; in the actual decode hot path `t==1`, so `h_tok` is just `z` (no round-trip) — drop the per-token host copy for `t==1` and slice `z` directly. The host round-trip is only the Stage-1 prefill simplification, consistent with the existing hybrid forward's host repacks.