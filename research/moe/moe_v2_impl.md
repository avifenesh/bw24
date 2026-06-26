All signatures verified. I have everything needed. Here is the complete, code-accurate implementation.

---

## 1. `crates/bw24-gguf/src/config.rs` — add `expert_shared_ff_length`

Edit the `MoeConfig` struct (lines 53-58) and its populate site (lines 116-122):

```rust
// REPLACE the MoeConfig struct (currently config.rs:53-58) with:
#[derive(Debug, Clone)]
pub struct MoeConfig {
    pub expert_count: u32,
    pub expert_used_count: u32,
    pub expert_ff_length: u32,
    pub expert_shared_ff_length: u32,   // NEW: qwen35moe.expert_shared_feed_forward_length = 512
}
```

```rust
// REPLACE the `let moe = ...` block (currently config.rs:116-122) with:
        let moe = if arch.is_moe() {
            Some(MoeConfig {
                expert_count: u("expert_count").unwrap_or(0),
                expert_used_count: u("expert_used_count").unwrap_or(0),
                expert_ff_length: u("expert_feed_forward_length").unwrap_or(0),
                // meta_arch tries "qwen35moe.expert_shared_feed_forward_length" first, then bare key
                expert_shared_ff_length: u("expert_shared_feed_forward_length").unwrap_or(0),
            })
        } else { None };
```

Everything else in `config.rs` (the `u` closure at line 90, `meta_arch` resolution) is unchanged — `u("expert_shared_feed_forward_length")` resolves the arch-prefixed key correctly via `meta_arch`.

---

## 2. `crates/bw24-engine/src/model.rs` — `HostExps` + 3D `load_exps`

Append this after the `EmbedHost` impl (the `GpuTensor`, `Layer`, `EmbedHost` definitions stay as-is). I keep the experts in a **host `Vec<u8>`** (Stage-1: simplest correct; pinned alloc is a Stage-2 swap that does not change `expert_bytes` semantics — see the validation note). The 3D fix is `row_bytes = raw.len() / (out_f * n_expert)`.

```rust
// ==== APPEND to crates/bw24-engine/src/model.rs ====

use bw24_gguf::GgmlType;  // already imported at top of model.rs alongside dequant; keep single import

/// One layer's stacked 256-expert tensor, raw GGUF quant bytes held HOST-RESIDENT.
///
/// EDGE-1: these bytes are NEVER uploaded at load (uploading 29.75GB would OOM a 24GB GPU —
/// this is BUG-4). Per token, only the 8 routed experts are staged H2D into a small GPU scratch.
///
/// ne = [in_f, out_f, n_expert]; the expert axis (ne[2]) is the slowest/highest-stride axis, so
/// expert `e` occupies the CONTIGUOUS byte block `bytes[e*expert_stride .. (e+1)*expert_stride]`.
///
/// THE 3D FIX: GpuTensor::load computes `row_bytes = raw.len()/ne[1]`, which for a stacked 3D
/// tensor ignores the 256-expert axis and is 256x too large (gate_exps -> 430080 instead of 1680).
/// load() here uses `row_bytes = raw.len() / (out_f * n_expert)` (= 1680 gate/up, 544 down).
pub struct HostExps {
    pub bytes: Vec<u8>,        // raw GGUF block bytes (host); per-token DMA src for the 8 routed exps
    pub qtype: i32,            // QT_Q6_K (gate/up) | QT_Q8_0 (down)
    pub in_f: usize,           // ne[0]   (gate/up = 2048, down = 512)
    pub out_f: usize,          // ne[1]   (gate/up = 512,  down = 2048)
    pub n_expert: usize,       // ne[2] = 256
    pub row_bytes: usize,      // raw.len()/(out_f*n_expert)  -> 1680 (gate/up) / 544 (down)
    pub expert_stride: usize,  // raw.len()/n_expert          -> 860160 (gate/up) / 1114112 (down)
}

impl HostExps {
    /// Load a stacked 3D expert tensor, keeping its quant bytes on the HOST.
    pub fn load(g: &GgufFile, name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let t = g.find(name).unwrap_or_else(|| panic!("missing exps tensor {name}"));
        assert_eq!(t.ne.len(), 3, "{name} is not a 3D stacked-expert tensor (ne={:?})", t.ne);
        let raw = g.tensor_data(t);
        let qtype = match t.ggml_type {
            GgmlType::Q6_K => QT_Q6_K,
            GgmlType::Q8_0 => QT_Q8_0,
            other => panic!("exps {name} unsupported quant {other:?} (use the Q6_K_XL file)"),
        };
        let in_f = t.ne[0] as usize;
        let out_f = t.ne[1] as usize;
        let n_expert = t.ne[2] as usize;
        // VERIFIED: gate/up Q6_K total/256 = 860160; row = total/(512*256) = 1680.
        //           down  Q8_0 total/256 = 1114112; row = total/(2048*256) = 544.
        let expert_stride = raw.len() / n_expert;
        let row_bytes = raw.len() / (out_f * n_expert);
        // sanity: expert_stride must equal out_f * row_bytes exactly (catches a dim mixup)
        assert_eq!(expert_stride, out_f * row_bytes,
            "{name} stride mismatch: stride={expert_stride} out_f={out_f} row_bytes={row_bytes}");
        Ok(HostExps { bytes: raw.to_vec(), qtype, in_f, out_f, n_expert, row_bytes, expert_stride })
    }

    /// Host byte slice for expert `e` (the H2D DMA source). Contiguous block, offset honored.
    #[inline]
    pub fn expert_bytes(&self, e: usize) -> &[u8] {
        debug_assert!(e < self.n_expert, "expert index {e} >= n_expert {}", self.n_expert);
        &self.bytes[e * self.expert_stride..(e + 1) * self.expert_stride]
    }
}
```

> Note on `use`: `model.rs:7` already does `use bw24_gguf::{GgufFile, GgmlType, dequant};`, so `GgmlType` and `GgufFile` are in scope — drop the extra `use bw24_gguf::GgmlType;` line above if it duplicates; I include it only for clarity. `QT_Q6_K`/`QT_Q8_0` are already imported at `model.rs:9`.

---

## 3. `crates/bw24-engine/src/hybrid.rs` — `Ffn` enum + `MoeWeights` + loader

Replace the bare `ffn_gate/ffn_up/ffn_down` fields on `HybridLayer` with `ffn: Ffn`, add `MoeWeights`/`Ffn`, and branch the loader.

```rust
// ==== crates/bw24-engine/src/hybrid.rs ====
// Add these struct + enum definitions (place after `pub enum Mixer { ... }`, before HybridLayer):

/// MoE weights for one layer. Router + shared expert stay GPU-RESIDENT (tiny); the 256 routed
/// experts stay HOST-RESIDENT (HostExps) and are staged per-token (EDGE-1).
pub struct MoeWeights {
    pub gate_inp: GpuTensor,        // F32 [2048,256] router          (GPU resident, Float)
    pub gate_inp_shexp: GpuTensor,  // F32 [2048] 1-D shared gate dot (GPU resident, Float, out_f=1)
    pub gate_exps: HostExps,        // Q6_K [2048,512,256]            (HOST)
    pub up_exps: HostExps,          // Q6_K [2048,512,256]            (HOST)
    pub down_exps: HostExps,        // Q8_0 [512,2048,256] TRANSPOSED (HOST; in=512,out=2048)
    pub gate_shexp: GpuTensor,      // Q8_0 [2048,512]                (GPU resident)
    pub up_shexp: GpuTensor,        // Q8_0 [2048,512]                (GPU resident)
    pub down_shexp: GpuTensor,      // Q8_0 [512,2048]                (GPU resident)
}

/// Per-layer FFN: dense SwiGLU (qwen35) or 256-expert MoE (qwen35moe).
pub enum Ffn {
    Dense { ffn_gate: GpuTensor, ffn_up: GpuTensor, ffn_down: GpuTensor },
    Moe(MoeWeights),
}
```

```rust
// REPLACE the HybridLayer struct (hybrid.rs:37-42) with:
pub struct HybridLayer {
    pub attn_norm: GpuTensor,
    pub post_attn_norm: GpuTensor,  // "post_attention_norm" = PRE-FFN norm
    pub mixer: Mixer,
    pub ffn: Ffn,
}
```

```rust
// In HybridModel::load:
//   - REMOVE the assert at hybrid.rs:56  -> `assert!(cfg.moe.is_none(), ...);`
//   - REPLACE the layers.push(HybridLayer { ... ffn_gate/up/down ... }) block (hybrid.rs:87-96)
//     with the version below that builds `ffn` from an arch branch.

            let ffn = if cfg.moe.is_some() {
                Ffn::Moe(MoeWeights {
                    gate_inp:       load_t(e, g, &p("ffn_gate_inp.weight"))?,        // F32 -> Float
                    gate_inp_shexp: load_t(e, g, &p("ffn_gate_inp_shexp.weight"))?,  // F32 1-D
                    gate_exps: HostExps::load(g, &p("ffn_gate_exps.weight"))?,       // HOST Q6_K
                    up_exps:   HostExps::load(g, &p("ffn_up_exps.weight"))?,         // HOST Q6_K
                    down_exps: HostExps::load(g, &p("ffn_down_exps.weight"))?,       // HOST Q8_0
                    gate_shexp: load_t(e, g, &p("ffn_gate_shexp.weight"))?,
                    up_shexp:   load_t(e, g, &p("ffn_up_shexp.weight"))?,
                    down_shexp: load_t(e, g, &p("ffn_down_shexp.weight"))?,
                })
            } else {
                Ffn::Dense {
                    ffn_gate: load_t(e, g, &p("ffn_gate.weight"))?,
                    ffn_up:   load_t(e, g, &p("ffn_up.weight"))?,
                    ffn_down: load_t(e, g, &p("ffn_down.weight"))?,
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

Add the import of `HostExps` at the top of `hybrid.rs` (it currently imports `use crate::model::{GpuTensor, EmbedHost};`):

```rust
use crate::model::{GpuTensor, EmbedHost, HostExps};
```

> The loader only iterates `0..cfg.n_layer` (= 40 trunk layers; MTP block 40 is excluded because `n_layer` excludes `nextn`). The MoE FFN is built for **every** trunk layer regardless of `Mixer::Full`/`Linear` — verified against qwen35moe.cpp (all trunk layers carry `_exps`).

---

## 4. Engine staging helper — `crates/bw24-engine/src/lib.rs`

Add these three methods inside `impl Engine`. `stage_expert` copies a host byte-slice into a GPU scratch slot (async H2D on the default stream — ordered before the qmatvec that reads it). `qmatvec_view` runs the validated `qmatvec_f32` kernel over a sub-range of a `CudaSlice<u8>` (a staged scratch slot). `axpy_into` and `add_scaled_rows` accumulate into the MoE output.

```rust
// ==== APPEND inside `impl Engine` in crates/bw24-engine/src/lib.rs ====

    /// Allocate a reusable u8 GPU scratch buffer (for staged expert weights).
    pub fn alloc_u8(&self, n: usize) -> Result<CudaSlice<u8>, Box<dyn std::error::Error>> {
        Ok(self.gpu.stream.alloc_zeros::<u8>(n)?)
    }

    /// EDGE-1 staging: copy `host_bytes` (a sub-slice of a HostExps buffer) into `scratch`
    /// at byte offset `off` (async H2D on the default stream). Length is host_bytes.len().
    /// The qmatvec_view that reads `scratch[off..]` is enqueued on the SAME stream after this,
    /// so ordering is guaranteed without an explicit sync (Stage-1; Stage-2 prefetch on a 2nd
    /// stream would require an event).
    pub fn stage_expert(&self, host_bytes: &[u8], scratch: &mut CudaSlice<u8>, off: usize)
                        -> Result<(), Box<dyn std::error::Error>> {
        let mut dst = scratch.slice_mut(off..off + host_bytes.len());  // CudaViewMut<u8>
        self.gpu.stream.memcpy_htod(host_bytes, &mut dst)?;            // accepts &[u8] HostSlice src
        Ok(())
    }

    /// qmatvec over a byte sub-range of a (resident/scratch) CudaSlice<u8> holding ONE expert
    /// matrix. x is a CudaView<f32> (a sliced row of z, or a sliced activation). Reuses the
    /// validated qmatvec_f32 dequant path (NOT a fast path — the correctness gate). The
    /// CudaView base+offset pointer is honored by the launch arg.
    pub fn qmatvec_view(&self, w: &CudaSlice<u8>, range: std::ops::Range<usize>,
                        x: &cudarc::driver::CudaView<f32>, m: usize, in_f: usize, out_f: usize,
                        qtype: i32, row_bytes: usize)
                        -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let f = self.func("qmatvec_f32");
        let wv = w.slice(range);  // CudaView<u8>, offset honored
        let mut y = self.gpu.stream.alloc_zeros::<f32>(m * out_f)?;
        let cfg = LaunchConfig { grid_dim: (out_f as u32, m as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let (inf, outf, mi, qt, rb) = (in_f as i32, out_f as i32, m as i32, qtype, row_bytes as i64);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(&wv).arg(x).arg(&mut y).arg(&inf).arg(&outf).arg(&mi).arg(&qt).arg(&rb);
        unsafe { b.launch(cfg)?; }
        Ok(y)
    }

    /// dst[i] += alpha * src[i], i in 0..n. dst is a CudaViewMut (a row of moe_out).
    pub fn axpy_into(&self, src: &CudaSlice<f32>, alpha: f32,
                     dst: &mut cudarc::driver::CudaViewMut<f32>, n: usize)
                     -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("axpy_f32");
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let (a, ni) = (alpha, n as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(src).arg(dst).arg(&a).arg(&ni);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// dst[r*ncols + c] += src[r*ncols + c] * scale[r]. Per-row scalar accumulate (shared expert).
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

### New CUDA kernels — append to `crates/bw24-engine/cu/hybrid.cu` (built into `BW24_HYBRID_FATBIN`)

```cuda
// ==== APPEND to cu/hybrid.cu ====

// dst[i] += alpha * src[i]
extern "C" __global__ void axpy_f32(const float* src, float* dst, float alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += alpha * src[i];
}

// dst[r*ncols + c] += src[r*ncols + c] * scale[r]   (r = i / ncols)
extern "C" __global__ void add_scaled_rows_f32(const float* src, const float* scale,
                                               float* dst, int ncols, int nrows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = ncols * nrows;
    if (i < total) {
        int r = i / ncols;
        dst[i] += src[i] * scale[r];
    }
}
```

---

## 5. `moe_ffn` forward — `hybrid_forward.rs` (prefill) and `decode.rs` (T=1)

Both call a shared `moe_ffn` defined on `HybridModel`. First the dispatch swap.

### `hybrid_forward.rs` — swap the dense FFN block (lines 38-45)

```rust
// REPLACE hybrid_forward.rs lines 36-46 (the pre-FFN norm + dense SwiGLU + residual) with:
            // pre-FFN norm (post_attention_norm), FFN (Dense or MoE), residual 2
            let mut z = e.zeros(t * n_embd)?;
            e.rms_norm(&x1, layer.post_attn_norm.float_data(), &mut z, n_embd, t, eps)?;
            let ffn_out = match &layer.ffn {
                crate::hybrid::Ffn::Dense { ffn_gate, ffn_up, ffn_down } => {
                    let n_ff = ffn_gate.out_features();
                    let gate = e.matmul(ffn_gate, &z, t)?;
                    let up = e.matmul(ffn_up, &z, t)?;
                    let mut act = e.zeros(t * n_ff)?;
                    e.silu_mul(&gate, &up, &mut act, t * n_ff)?;
                    e.matmul(ffn_down, &act, t)?
                }
                crate::hybrid::Ffn::Moe(m) => self.moe_ffn(e, m, &z, t)?,
            };
            let mut x2 = e.zeros(t * n_embd)?;
            e.add(&x1, &ffn_out, &mut x2, t * n_embd)?;
            x = x2;
```

### `decode.rs` — same swap (lines 34-44), with `t = 1`

```rust
// REPLACE decode.rs lines 34-44 with:
            let mut z = e.zeros(n_embd)?;
            e.rms_norm(&x1, layer.post_attn_norm.float_data(), &mut z, n_embd, 1, eps)?;
            let ffn_out = match &layer.ffn {
                crate::hybrid::Ffn::Dense { ffn_gate, ffn_up, ffn_down } => {
                    let n_ff = ffn_gate.out_features();
                    let gate = e.matmul(ffn_gate, &z, 1)?;
                    let up = e.matmul(ffn_up, &z, 1)?;
                    let mut act = e.zeros(n_ff)?;
                    e.silu_mul(&gate, &up, &mut act, n_ff)?;
                    e.matmul(ffn_down, &act, 1)?
                }
                crate::hybrid::Ffn::Moe(m) => self.moe_ffn(e, m, &z, 1)?,
            };
            let mut x2 = e.zeros(n_embd)?;
            e.add(&x1, &ffn_out, &mut x2, n_embd)?;
            x = x2;
```

### `moe_ffn` — the EDGE-1 forward (add to a new `impl HybridModel` block)

This handles `t >= 1` so prefill (loops T) and decode (T=1) share it. Each token routes to its own 8 experts (per-token staging — prefill cannot share a staged set across tokens). Add this in `hybrid_forward.rs` (so `MoeWeights`/`HostExps` are reachable) — or a new `moe.rs` module; I put it in `hybrid_forward.rs`.

```rust
// ==== ADD to crates/bw24-engine/src/hybrid_forward.rs ====
use crate::hybrid::{MoeWeights};
use crate::{QT_Q6_K, QT_Q8_0};

impl HybridModel {
    /// MoE FFN (EDGE-1 Stage-1, host-resident experts, per-token H2D of ONLY the routed 8).
    /// z: [T, n_embd] (already post-attention-normed). Returns moe_out [T, n_embd].
    /// Node-for-node vs llama.cpp build_moe_ffn + qwen35moe::build_layer_ffn.
    fn moe_ffn(&self, e: &Engine, m: &MoeWeights, z: &CudaSlice<f32>, t: usize)
               -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let moe = cfg.moe.as_ref().unwrap();
        let n_embd = cfg.n_embd as usize;          // 2048 (gate/up in_f, down out_f)
        let n_expert = moe.expert_count as usize;  // 256
        let n_used = moe.expert_used_count as usize; // 8
        let n_ff_exp = moe.expert_ff_length as usize; // 512 (gate/up out_f, down in_f)

        // verify the HostExps dims match cfg (catches a wrong-file / transpose mixup)
        debug_assert_eq!(m.gate_exps.in_f, n_embd);
        debug_assert_eq!(m.gate_exps.out_f, n_ff_exp);
        debug_assert_eq!(m.down_exps.in_f, n_ff_exp);  // down is TRANSPOSED: in=512
        debug_assert_eq!(m.down_exps.out_f, n_embd);   //                     out=2048
        debug_assert_eq!(m.gate_exps.n_expert, n_expert);

        // 1. ROUTER: logits = ffn_gate_inp @ z  -> [T, 256]. gate_inp is F32 -> e.linear.
        let logits = e.matmul(&m.gate_inp, z, t)?;
        let lg = e.dtoh(&logits)?;   // [T*256] host

        let mut moe_out = e.zeros(t * n_embd)?;

        // GPU scratch: one slot per proj, big enough for ONE expert (Stage-1, no cache).
        let g_len = m.gate_exps.expert_stride;  // 860160
        let u_len = m.up_exps.expert_stride;    // 860160
        let d_len = m.down_exps.expert_stride;  // 1114112
        let mut scratch_g = e.alloc_u8(g_len)?;
        let mut scratch_u = e.alloc_u8(u_len)?;
        let mut scratch_d = e.alloc_u8(d_len)?;

        // 2. PER TOKEN: softmax-over-256, stable top-8, renorm, routed-expert loop.
        for tok in 0..t {
            let row = &lg[tok * n_expert..(tok + 1) * n_expert];

            // softmax over ALL 256 (stable: subtract max)
            let maxl = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut probs = vec![0f32; n_expert];
            let mut den = 0f32;
            for i in 0..n_expert { let x = (row[i] - maxl).exp(); probs[i] = x; den += x; }
            for p in probs.iter_mut() { *p /= den; }

            // BUG-1 FIX: stable DESC sort matching CUDA cub radix (argsort DESC, stable on ties).
            // total_cmp is NaN-safe; .then(a.cmp(&b)) gives ascending-index tiebreak.
            let mut idx: Vec<usize> = (0..n_expert).collect();
            idx.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]).then(a.cmp(&b)));
            let sel = &idx[..n_used];

            // gather UNBIASED probs as weights, then renorm: sum -> clamp -> divide. NO w_scale.
            let mut w: Vec<f32> = sel.iter().map(|&i| probs[i]).collect();
            let mut ws: f32 = w.iter().sum();
            ws = ws.max(6.103515625e-5_f32);  // F16 smallest normal, clamp BEFORE divide
            for x in w.iter_mut() { *x /= ws; }

            let zt = z.slice(tok * n_embd..(tok + 1) * n_embd);  // CudaView<f32>

            for (j, &ex) in sel.iter().enumerate() {
                // stage gate/up/down for expert `ex` (async H2D, ordered before the qmatvec below)
                e.stage_expert(m.gate_exps.expert_bytes(ex), &mut scratch_g, 0)?;
                let gate = e.qmatvec_view(&scratch_g, 0..g_len, &zt, 1,
                    m.gate_exps.in_f, m.gate_exps.out_f, QT_Q6_K, m.gate_exps.row_bytes)?;

                e.stage_expert(m.up_exps.expert_bytes(ex), &mut scratch_u, 0)?;
                let up = e.qmatvec_view(&scratch_u, 0..u_len, &zt, 1,
                    m.up_exps.in_f, m.up_exps.out_f, QT_Q6_K, m.up_exps.row_bytes)?;

                // act = silu(gate) * up   (length n_ff_exp = 512)
                let mut act = e.zeros(n_ff_exp)?;
                e.silu_mul(&gate, &up, &mut act, n_ff_exp)?;

                // down: TRANSPOSED, in=512 out=2048. x arg must be a CudaView (BUG-8).
                e.stage_expert(m.down_exps.expert_bytes(ex), &mut scratch_d, 0)?;
                let actv = act.slice(0..n_ff_exp);
                let y = e.qmatvec_view(&scratch_d, 0..d_len, &actv, 1,
                    m.down_exps.in_f, m.down_exps.out_f, QT_Q8_0, m.down_exps.row_bytes)?;

                // moe_out[tok] += w[j] * y  (BUG-9: slice_mut -> CudaViewMut)
                let mut dst = moe_out.slice_mut(tok * n_embd..(tok + 1) * n_embd);
                e.axpy_into(&y, w[j], &mut dst, n_embd)?;
            }
        }

        // 3. SHARED EXPERT (ALWAYS-ON, no routing) on the SAME z.
        let n_ff_sh = m.gate_shexp.out_features();  // 512
        let sg_gate = e.matmul(&m.gate_shexp, z, t)?;  // [T, 512]
        let sg_up = e.matmul(&m.up_shexp, z, t)?;      // [T, 512]
        let mut sa = e.zeros(t * n_ff_sh)?;
        e.silu_mul(&sg_gate, &sg_up, &mut sa, t * n_ff_sh)?;
        let sh = e.matmul(&m.down_shexp, &sa, t)?;     // [T, n_embd]

        // BUG-2 FIX: ffn_gate_inp_shexp is 1-D ne=[2048] -> out_f=1. Use e.linear(.., out_f=1),
        // NOT matmul/out_features (which would index ne[1] out of bounds).
        let gs = e.linear(z, m.gate_inp_shexp.float_data(), t, n_embd, 1)?;  // [T, 1]
        let mut g = e.zeros(t)?;
        e.sigmoid(&gs, &mut g, t)?;

        // moe_out[r, :] += sh[r, :] * g[r]   (per-token scalar gate)
        e.add_scaled_rows(&sh, &g, &mut moe_out, n_embd, t)?;

        Ok(moe_out)
    }
}
```

> **Stage-1 correctness gate**: do NOT set `BW24_FAST`. `qmatvec_view` already calls `qmatvec_f32`; the shared-expert `e.matmul` falls back to `qmatvec` (f32 dequant) when `BW24_FAST` is unset. The down-expert transpose is handled purely by passing `m.down_exps.in_f`/`out_f`/`row_bytes` (512/2048/544) — no special-casing in the kernel.

---

## 6. VALIDATION PLAN — exact commands

**Model:** `/home/avifenesh/ai-ml/hf-models/qwen36-35b-mtp/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf` (Q6_K_XL only — the IQ4_XS file has IQ3_S/IQ4_XS experts that `qmatvec.cu`/`dequant.rs` do not support and will panic in `HostExps::load`). Fixed token ids: `785 3974 13876 39935`.

**Build (Stage-1, no fast path):**
```bash
cd /home/avifenesh/projects/bw24 && unset BW24_FAST && cargo build --release -p bw24-engine 2>&1 | tail -20
```

**Stage A — argmax + top-5 match vs llama.cpp (the headline gate):**
```bash
# Ground truth (raw token ids, no BOS auto-prepend):
/home/avifenesh/projects/bw24/tools/llama_logits \
  /home/avifenesh/ai-ml/hf-models/qwen36-35b-mtp/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf \
  785 3974 13876 39935

# bw24 (run_hybrid already prints argmax + top-5; HybridModel::load now takes the MoE path):
/home/avifenesh/projects/bw24/target/release/run-hybrid \
  /home/avifenesh/ai-ml/hf-models/qwen36-35b-mtp/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf \
  785 3974 13876 39935
```
PASS = argmax token id EXACT (`==`) match; top-5 ids equal as a SET; last-token max logit within ~1-2% relative (f32 dequant path). Top-5 *order* may differ only on near-ties.

> `run_hybrid.rs:36` sorts with `partial_cmp(...).unwrap()` — NaN-unsafe. The MoE host top-k uses the BUG-1 `total_cmp().then(idx)` ordering (already in `moe_ffn`); the run_hybrid printout sort is display-only and does not affect routing, but switch it to `total_cmp` for consistency if you want deterministic top-5 display.

**Stage B — layer-0 top-8 bisection (BUG-1 is the #1 risk):**
```bash
# llama.cpp dumps EVERY cb()-named tensor (default cb_data -> empty filter). Grep layer-0 MoE:
/home/avifenesh/projects/llama.cpp/build/bin/llama-eval-callback \
  -m /home/avifenesh/ai-ml/hf-models/qwen36-35b-mtp/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf \
  -p "The quick brown fox" -n 0 2>&1 \
  | grep -A30 'ffn_moe_topk-0\|ffn_moe_weights_norm-0\|ffn_out-0'
```
In bw24, add a debug print in `moe_ffn` gated on the layer index for `il == 0` (thread `il` in via the `match &layer.ffn` call site, or a `BW24_MOE_DEBUG` env check): print the host top-8 `sel` indices, the normalized `w[..8]`, and `e.dtoh(&moe_out)[0..8]` for the last token.

PASS criteria, cheapest-first (mandatory order):
1. **Load micro-check** — host-dequant `gate_exps.expert_bytes(0)` (via `dequant::dequantize(GgmlType::Q6_K, ...)` per row using `row_bytes=1680`) and compare to `qmatvec_view(&scratch_g, 0..g_len, zt, ...)` for a random `z`. Proves `row_bytes` + `expert_stride` + staging. Repeat for `down_exps` (in=512,out=2048) to catch the transpose trap.
2. **Router top-8** — bw24 layer-0 `sel` SET == llama.cpp `ffn_moe_topk-0` indices (last token); normalized weights match `ffn_moe_weights_norm-0` to ~1e-4. Proves router + BUG-1 tiebreak.
3. **Full forward** — Stage A argmax + top-5 set-equality.

> Stage-B caveat: `llama-eval-callback` uses `common_tokenize` with `add_bos=true` on the prompt string, while `llama_logits` feeds raw ids with no BOS. For the Stage-B index comparison, account for the prepended BOS (positions shift by 1) or feed eval-callback a matching pre-tokenized path; only the *last-token* top-8 set needs to line up.

---

## Files touched (all absolute)

- `/home/avifenesh/projects/bw24/crates/bw24-gguf/src/config.rs` — `MoeConfig.expert_shared_ff_length` + populate site.
- `/home/avifenesh/projects/bw24/crates/bw24-engine/src/model.rs` — `HostExps` struct + `load` (3D `row_bytes = raw.len()/(out_f*n_expert)` fix) + `expert_bytes`.
- `/home/avifenesh/projects/bw24/crates/bw24-engine/src/hybrid.rs` — `MoeWeights`, `Ffn` enum, `HybridLayer.ffn`, loader branch, removed `moe.is_none()` assert, added `HostExps` import.
- `/home/avifenesh/projects/bw24/crates/bw24-engine/src/lib.rs` — `alloc_u8`, `stage_expert`, `qmatvec_view`, `axpy_into`, `add_scaled_rows`.
- `/home/avifenesh/projects/bw24/crates/bw24-engine/cu/hybrid.cu` — `axpy_f32`, `add_scaled_rows_f32` kernels.
- `/home/avifenesh/projects/bw24/crates/bw24-engine/src/hybrid_forward.rs` — `moe_ffn` method + dense/MoE dispatch (prefill).
- `/home/avifenesh/projects/bw24/crates/bw24-engine/src/decode.rs` — dense/MoE dispatch (T=1).

**Key correctness anchors:** 3D `row_bytes`=1680/544, `expert_stride`=860160/1114112; experts stay in host `Vec<u8>` (never `htod` — avoids BUG-4 OOM); BUG-1 `total_cmp().then(a.cmp(&b))` stable DESC sort; renorm clamp `6.103515625e-5` before divide, no `w_scale`; BUG-2 shared gate via `e.linear(z, ..., out_f=1)` then sigmoid (1-D, not matmul); down-expert transpose carried purely through `in_f`/`out_f`/`row_bytes` args; Stage-1 stays on the validated `qmatvec_f32` path (no `BW24_FAST`).