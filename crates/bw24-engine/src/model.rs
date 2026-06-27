//! Dense transformer model: loads GGUF weights to GPU (Stage-1: dequant→f32), runs the
//! shared full-attention + SwiGLU forward graph. Arch-agnostic via ModelConfig; this path is
//! exactly the dense-transformer graph (qwen3) and the full-attention layers of hybrids.

use std::collections::HashMap;
use cudarc::driver::CudaSlice;
use bw24_gguf::{GgufFile, GgmlType, dequant};
use bw24_gguf::config::ModelConfig;
use bw24_gguf::source::{TensorSource, GgufSource};
use crate::{Engine, QT_Q8_0, QT_Q4_K, QT_Q6_K, QT_Q5_K, QT_Q3_K, QT_IQ4_XS, QT_IQ3_S, QT_NVFP4, QT_F32};

/// A weight tensor resident on GPU. Quantized weights stay in GGUF block bytes (`Quant`);
/// small non-quant tensors (norms, sometimes embed/lm_head) are kept dequantized as f32 (`Float`).
/// This keeps VRAM ~= on-disk quant size (fixes the f32-on-load OOM).
pub enum GpuTensor {
    Quant { bytes: CudaSlice<u8>, qtype: i32, row_bytes: usize, ne: Vec<u64>, scale: f32 },
    Float { data: CudaSlice<f32>, ne: Vec<u64> },
}

impl GpuTensor {
    pub fn ne(&self) -> &[u64] { match self { GpuTensor::Quant { ne, .. } => ne, GpuTensor::Float { ne, .. } => ne } }
    pub fn in_features(&self) -> usize { self.ne()[0] as usize }
    pub fn out_features(&self) -> usize { self.ne()[1] as usize }

    /// Load a tensor, keeping quant types packed and float types as f32. (GGUF entry point —
    /// thin wrapper over the source-agnostic `load_from_source`; behavior is unchanged.)
    pub fn load(e: &Engine, g: &GgufFile, name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        Self::load_from_source(e, &GgufSource(g), name)
    }

    /// Source-agnostic load: works from any `TensorSource` (GGUF or safetensors). The engine's
    /// forward graph only ever asks for ggml-style names; the source maps them to its own layout.
    pub fn load_from_source(e: &Engine, src: &dyn TensorSource, name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let v = src.find(name).unwrap_or_else(|| panic!("missing tensor {name}"));
        let qtype = match v.ggml_type {
            GgmlType::Q8_0 => Some(QT_Q8_0),
            GgmlType::Q4_K => Some(QT_Q4_K),
            GgmlType::Q6_K => Some(QT_Q6_K),
            GgmlType::Q5_K => Some(QT_Q5_K),
            GgmlType::Q3_K => Some(QT_Q3_K),
            GgmlType::IQ4_XS => Some(QT_IQ4_XS),
            GgmlType::IQ3_S => Some(QT_IQ3_S),
            GgmlType::NVFP4 => Some(QT_NVFP4),
            // F32/F16/BF16 (the dtypes safetensors carries) -> Float path below.
            _ => None,
        };
        match qtype {
            Some(qt) => {
                let out_f = v.ne[1] as usize;
                let row_bytes = v.bytes.len() / out_f;
                // NVFP4 two-level scale: per-16 ue4m3 micro-scale is in the dequant; the per-tensor
                // F32 macro-scale lives in a sibling "<stem>.scale" tensor, applied POST-matmul
                // (llama build_lora_mm: ggml_mul(res, w_s)). ".input_scale" is the W4A4 activation
                // scale — UNUSED on our W4A16/f32 path. Only NVFP4 carries it; others -> 1.0 (no-op).
                let scale = if qt == QT_NVFP4 {
                    let stem = name.strip_suffix(".weight").unwrap_or(name);
                    match src.find(&format!("{stem}.scale")) {
                        Some(sv) => f32::from_le_bytes(sv.bytes[..4].try_into().unwrap()),
                        None => 1.0,
                    }
                } else { 1.0 };
                Ok(GpuTensor::Quant { bytes: e.htod_bytes(&v.bytes)?, qtype: qt, row_bytes, ne: v.ne.clone(), scale })
            }
            None => {
                // F32/F16/BF16 (or as-yet-unhandled quant): dequant to f32. Small tensors only.
                let n: u64 = v.ne.iter().product();
                let f32v = dequant::dequantize(v.ggml_type, &v.bytes, n as usize);
                Ok(GpuTensor::Float { data: e.htod(&f32v)?, ne: v.ne.clone() })
            }
        }
    }

    pub fn load_opt(e: &Engine, g: &GgufFile, name: &str) -> Result<Option<Self>, Box<dyn std::error::Error>> {
        Self::load_opt_from_source(e, &GgufSource(g), name)
    }

    pub fn load_opt_from_source(e: &Engine, src: &dyn TensorSource, name: &str) -> Result<Option<Self>, Box<dyn std::error::Error>> {
        if src.has(name) { Ok(Some(Self::load_from_source(e, src, name)?)) } else { Ok(None) }
    }

    /// Accessor for tensors that MUST be f32 (norm weights). Panics if quantized.
    pub fn float_data(&self) -> &CudaSlice<f32> {
        match self { GpuTensor::Float { data, .. } => data, GpuTensor::Quant { .. } => panic!("expected float tensor (norm), got quantized") }
    }
}

pub struct Layer {
    pub attn_norm: GpuTensor,
    pub wq: GpuTensor, pub wk: GpuTensor, pub wv: GpuTensor, pub wo: GpuTensor,
    pub q_norm: Option<GpuTensor>, pub k_norm: Option<GpuTensor>,
    pub ffn_norm: GpuTensor,
    /// FFN: dense SwiGLU or routed MoE (OLMoE — dense attention + MoE FFN). Reuses the hybrid
    /// `Ffn` enum + `load_ffn` so the routed-expert forward is shared with `HybridModel::moe_ffn`.
    pub ffn: crate::hybrid::Ffn,
}

/// Host-resident embedding table for row gather (dequant only the needed token rows).
pub struct EmbedHost {
    pub raw: Vec<u8>,
    pub ggml_type: GgmlType,
    pub n_embd: usize,
}
impl EmbedHost {
    pub fn from_gguf(g: &GgufFile, name: &str) -> Self {
        Self::from_source(&GgufSource(g), name)
    }
    pub fn from_source(src: &dyn TensorSource, name: &str) -> Self {
        let v = src.find(name).unwrap_or_else(|| panic!("missing embed {name}"));
        EmbedHost { raw: v.bytes.to_vec(), ggml_type: v.ggml_type, n_embd: v.ne[0] as usize }
    }
    /// Gather rows for tokens -> [T, n_embd] f32. Dequant per-row from raw bytes.
    pub fn gather(&self, n_embd: usize, tokens: &[u32]) -> Vec<f32> {
        let (blk, tsize) = self.ggml_type.block_and_type_size();
        let row_bytes = (n_embd as u64 / blk * tsize) as usize;
        let mut x = vec![0f32; tokens.len() * n_embd];
        for (ti, &tok) in tokens.iter().enumerate() {
            let off = tok as usize * row_bytes;
            let row = dequant::dequantize(self.ggml_type, &self.raw[off..off + row_bytes], n_embd);
            x[ti * n_embd..ti * n_embd + n_embd].copy_from_slice(&row);
        }
        x
    }
}

pub struct Model {
    pub cfg: ModelConfig,
    pub embd: EmbedHost,
    pub output_norm: GpuTensor,
    pub output: GpuTensor,
    pub layers: Vec<Layer>,
}

impl Model {
    /// Load a dense (vanilla-transformer) model from GGUF. Thin wrapper over
    /// `load_dense_from_source`. Panics if the arch has SSM/MoE layers.
    pub fn load_dense(e: &Engine, g: &GgufFile) -> Result<Self, Box<dyn std::error::Error>> {
        Self::load_dense_from_source(e, &GgufSource(g))
    }

    /// Load a dense-attention model from any `TensorSource` — GGUF or a safetensors HF checkpoint.
    /// The whole loop speaks ggml names; the source maps them. The FFN is dense SwiGLU OR routed MoE
    /// (OLMoE: dense full-attention + MoE FFN). Panics on hybrid (SSM) arches — use the hybrid path.
    pub fn load_dense_from_source(e: &Engine, src: &dyn TensorSource) -> Result<Self, Box<dyn std::error::Error>> {
        let cfg = src.config();
        assert!(cfg.full_attention_interval == 0, "model has linear-attn layers; use hybrid path");

        let embd = EmbedHost::from_source(src, "token_embd.weight");
        let output_norm = GpuTensor::load_from_source(e, src, "output_norm.weight")?;
        // tied embeddings: fall back to tok_embd if output.weight absent (OLMoE has untied output).
        let output = if src.has("output.weight") {
            GpuTensor::load_from_source(e, src, "output.weight")?
        } else {
            GpuTensor::load_from_source(e, src, "token_embd.weight")?
        };

        let mut layers = Vec::with_capacity(cfg.n_layer as usize);
        for il in 0..cfg.n_layer {
            let p = |s: &str| format!("blk.{il}.{s}");
            layers.push(Layer {
                attn_norm: GpuTensor::load_from_source(e, src, &p("attn_norm.weight"))?,
                wq: GpuTensor::load_from_source(e, src, &p("attn_q.weight"))?,
                wk: GpuTensor::load_from_source(e, src, &p("attn_k.weight"))?,
                wv: GpuTensor::load_from_source(e, src, &p("attn_v.weight"))?,
                wo: GpuTensor::load_from_source(e, src, &p("attn_output.weight"))?,
                q_norm: GpuTensor::load_opt_from_source(e, src, &p("attn_q_norm.weight"))?,
                k_norm: GpuTensor::load_opt_from_source(e, src, &p("attn_k_norm.weight"))?,
                ffn_norm: GpuTensor::load_from_source(e, src, &p("ffn_norm.weight"))?,
                ffn: crate::hybrid::load_ffn(e, src, &cfg, il, None)?,
            });
        }
        Ok(Model { cfg, embd, output_norm, output, layers })
    }

    /// Largest expert block (bytes) across all MoE layers — the fixed cache-slot size (mirrors
    /// `HybridModel::max_moe_block`). 0 for a dense (non-MoE) model.
    pub(crate) fn max_moe_block(&self) -> usize {
        use crate::hybrid::Ffn;
        let mut mx = 0usize;
        for l in &self.layers {
            if let Ffn::Moe(m) = &l.ffn {
                mx = mx.max(m.gate_exps.expert_stride)
                       .max(m.up_exps.expert_stride)
                       .max(m.down_exps.expert_stride);
            }
        }
        mx
    }

    /// Gather embedding rows into f32 [T, n_embd] (token-major) by dequantizing only the needed
    /// rows from the host-side embedding bytes (token_embd is [n_embd, n_vocab], row per token).
    pub fn embed_tokens(&self, e: &Engine, tokens: &[u32]) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let n_embd = self.cfg.n_embd as usize;
        let x = self.embd.gather(n_embd, tokens);
        Ok(e.htod(&x)?)
    }
}

pub type TensorMap = HashMap<String, GpuTensor>;

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
/// Host byte storage for the expert blocks. Default = a pageable `Vec<u8>` (current behavior). Under
/// BW24_MOE_PINNED (auto-on when BW24_MOE_CACHE is set), the bytes live in CUDA pinned host memory so
/// the miss-path `memcpy_htod` is a true DMA, not a pageable bounce copy (MOE-SLRU-PLAN §C.1).
///
/// CAVEAT (§C.1): `alloc_pinned` uses CU_MEMHOSTALLOC_WRITECOMBINED — great for H2D-only (the expert
/// bytes are never read by the CPU on the hot path), but write-combined memory is SLOW for CPU reads.
/// A future CPU-VNNI cold-expert fallback must NOT read from this buffer.
pub enum HostBuf {
    Paged(Vec<u8>),
    /// Pinned host memory. We keep the `PinnedHostSlice` alive (it owns the allocation; Drop frees it)
    /// AND cache its raw base pointer + len so the hot-path `as_bytes()` needs no per-call event sync.
    Pinned { slice: cudarc::driver::PinnedHostSlice<u8>, base: *const u8, len: usize },
    /// SPILLING-PLAN §1, Tier 2 (disk): the bytes live in an mmap'd region of the GGUF file, NOT in
    /// RAM. `map` is `MAP_SHARED`, no `MAP_POPULATE` — zero upfront copy. The first `memcpy_htod` of
    /// this slice page-faults → NVMe read → DMA (the demand-fault disk path). `off`/`len` select this
    /// expert's contiguous block within the shared file mmap. Bit-identical to `Paged`/`Pinned` —
    /// those copied FROM exactly these on-disk bytes, so the GEMM result is unchanged.
    Mmap { map: std::sync::Arc<memmap2::Mmap>, off: usize, len: usize },
}
// SAFETY: `base` is a stable pinned-host pointer owned by `slice`; the buffer is written once at load
// then only READ for H2D. HostExps is shared `&` across the (single per-Engine) forward, so Send/Sync
// mirror the underlying PinnedHostSlice (which is already Send+Sync). The `Mmap` arm holds an
// `Arc<Mmap>` (Mmap is Send+Sync) plus plain usize fields, so it does not weaken these bounds.
unsafe impl Send for HostBuf {}
unsafe impl Sync for HostBuf {}
impl HostBuf {
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            HostBuf::Paged(v) => v.as_slice(),
            // SAFETY: base+len are the pinned allocation's stable extent; written once at load, then
            // read-only. We avoid `as_slice()` here because it would synchronize the buffer's event
            // on every hot-path call.
            HostBuf::Pinned { base, len, .. } => unsafe { std::slice::from_raw_parts(*base, *len) },
            // Slicing the mmap is the same `&[u8]` the kernel DMAs; the read page-faults the NVMe.
            HostBuf::Mmap { map, off, len } => &map[*off..*off + *len],
        }
    }
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            HostBuf::Paged(v) => v.len(),
            HostBuf::Pinned { len, .. } => *len,
            HostBuf::Mmap { len, .. } => *len,
        }
    }
}

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
    pub bytes: HostBuf,        // raw GGUF block bytes (host); per-token DMA src for the 8 routed exps
    /// SPILLING-PLAN §1.1: per-expert backing tier. `None` => the layer fits in one `bytes` store and
    /// every expert slices it (the unchanged in-RAM path). `Some` => per-expert split: the hottest
    /// experts are `Pinned` (Tier 1, fast async DMA), the rest `Mmap` into the GGUF (Tier 2, disk
    /// demand-fault). `expert_bytes(e)` resolves `tiers[e]` if present, else slices `bytes`.
    pub tiers: Option<Vec<HostBuf>>,
    pub qtype: i32,            // QT_Q6_K (gate/up) | QT_Q8_0 (down)
    pub in_f: usize,           // ne[0]   (gate/up = 2048, down = 512)
    pub out_f: usize,          // ne[1]   (gate/up = 512,  down = 2048)
    pub n_expert: usize,       // ne[2] = 256
    pub row_bytes: usize,      // raw.len()/(out_f*n_expert)  -> 1680 (gate/up) / 544 (down)
    pub expert_stride: usize,  // raw.len()/n_expert          -> 860160 (gate/up) / 1114112 (down)
}

impl HostExps {
    /// Load a stacked 3D expert tensor, keeping its quant bytes on the HOST. `e` supplies the CUDA
    /// context for the optional pinned allocation (§C.1). Default storage is pageable `Vec<u8>`
    /// (identical to the prior behavior); pinned is chosen when BW24_MOE_PINNED or BW24_MOE_CACHE is set.
    pub fn load(e: &Engine, g: &GgufFile, name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        Self::load_stacked_from_source(e, &GgufSource(g), name)
    }

    /// Load a STACKED 3D expert tensor (`ne=[in_f,out_f,n_expert]`) from any source. GGUF stores the
    /// experts this way; the source returns the same mmap bytes (`GgufSource::find` == `tensor_data`),
    /// so the GGUF path is byte-identical to the prior direct-`GgufFile` loader. (Safetensors stores N
    /// 2D tensors instead — those go through `load_from_source`, which gathers them.)
    pub fn load_stacked_from_source(e: &Engine, src: &dyn TensorSource, name: &str)
                                    -> Result<Self, Box<dyn std::error::Error>> {
        let t = src.find(name).unwrap_or_else(|| panic!("missing exps tensor {name}"));
        assert_eq!(t.ne.len(), 3, "{name} is not a 3D stacked-expert tensor (ne={:?})", t.ne);
        let raw: &[u8] = &t.bytes;
        // All quant types the staged-expert qmatvec can decode (dp4a-fast or Stage-A f32).
        let qtype = match t.ggml_type {
            GgmlType::Q8_0 => QT_Q8_0,
            GgmlType::Q4_K => QT_Q4_K,
            GgmlType::Q6_K => QT_Q6_K,
            GgmlType::Q5_K => QT_Q5_K,
            GgmlType::Q3_K => QT_Q3_K,
            GgmlType::IQ4_XS => QT_IQ4_XS,
            GgmlType::IQ3_S => QT_IQ3_S,
            GgmlType::NVFP4 => QT_NVFP4,
            other => panic!("exps {name} unsupported quant {other:?}"),
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

        let pinned = std::env::var("BW24_MOE_PINNED").is_ok() || std::env::var("BW24_MOE_CACHE").is_ok();
        let bytes = if pinned {
            // alloc pinned host memory, copy the GGUF block bytes in once, cache the base pointer.
            let mut p = unsafe { e.ctx().alloc_pinned::<u8>(raw.len())? };
            { let dst = p.as_mut_slice()?; dst.copy_from_slice(raw); }
            let base = p.as_ptr()? as *const u8;   // syncs once here at load; stable afterward
            let len = raw.len();
            HostBuf::Pinned { slice: p, base, len }
        } else {
            HostBuf::Paged(raw.to_vec())
        };
        Ok(HostExps { bytes, tiers: None, qtype, in_f, out_f, n_expert, row_bytes, expert_stride })
    }

    /// SPILLING-PLAN §1.1, §2 step 4: load a stacked 3D expert tensor with a PER-EXPERT tier split.
    /// Under `BW24_SPILL_DISK`, the hottest experts (greedy in expert order, until the shared pinned
    /// budget in `ctx` is exhausted) get `HostBuf::Pinned` (Tier 1, fast async DMA); every remaining
    /// expert is `HostBuf::Mmap` into the GGUF (Tier 2, demand-faulted from disk on first H2D). The
    /// resulting bytes are bit-identical to the in-RAM path either way — `qmatvec_view` is untouched.
    ///
    /// `ctx.file_map` is ONE shared `MAP_SHARED` mmap of the whole GGUF (`Arc`-cloned per spilled
    /// expert), so the 120 expert tensors of a 40-layer MoE never open the file more than once.
    pub fn load_tiered(e: &Engine, g: &GgufFile, name: &str, ctx: &mut crate::spill::SpillCtx)
                       -> Result<Self, Box<dyn std::error::Error>> {
        let t = g.find(name).unwrap_or_else(|| panic!("missing exps tensor {name}"));
        assert_eq!(t.ne.len(), 3, "{name} is not a 3D stacked-expert tensor (ne={:?})", t.ne);
        let raw = g.tensor_data(t);
        let qtype = match t.ggml_type {
            GgmlType::Q8_0 => QT_Q8_0,
            GgmlType::Q4_K => QT_Q4_K,
            GgmlType::Q6_K => QT_Q6_K,
            GgmlType::Q5_K => QT_Q5_K,
            GgmlType::Q3_K => QT_Q3_K,
            GgmlType::IQ4_XS => QT_IQ4_XS,
            GgmlType::IQ3_S => QT_IQ3_S,
            GgmlType::NVFP4 => QT_NVFP4,
            other => panic!("exps {name} unsupported quant {other:?}"),
        };
        let in_f = t.ne[0] as usize;
        let out_f = t.ne[1] as usize;
        let n_expert = t.ne[2] as usize;
        let expert_stride = raw.len() / n_expert;
        let row_bytes = raw.len() / (out_f * n_expert);
        assert_eq!(expert_stride, out_f * row_bytes,
            "{name} stride mismatch: stride={expert_stride} out_f={out_f} row_bytes={row_bytes}");

        // Absolute file offset of this tensor's data (start of expert 0); each expert is the next
        // `expert_stride` bytes. The `Mmap` arm slices `ctx.file_map` at these offsets.
        let (file_start, _file_end) = g.tensor_file_range(t);

        // Per-expert tier decision under the shared running budget. `bytes` keeps a 0-byte sentinel
        // (`Paged(empty)`) since every read now goes through `tiers`.
        let mut tiers = Vec::with_capacity(n_expert);
        for ex in 0..n_expert {
            let blk = &raw[ex * expert_stride..(ex + 1) * expert_stride];
            let file_off = file_start + ex * expert_stride;
            tiers.push(crate::spill::place_expert(ctx, e, blk, file_off)?);
        }
        Ok(HostExps {
            bytes: HostBuf::Paged(Vec::new()),  // unused when `tiers` is Some
            tiers: Some(tiers),
            qtype, in_f, out_f, n_expert, row_bytes, expert_stride,
        })
    }

    /// MoE expert GATHER from a `TensorSource` (the safetensors path; ST-MOE-PLAN §1.3). GGUF stacks
    /// all experts into ONE 3D tensor; HF stores them as N separate 2D tensors
    /// `model.layers.{il}.mlp.experts.{e}.{gate,up,down}_proj.weight`. `find` returns `None` for the
    /// ggml `*_exps` name on purpose, so the experts are gathered out-of-band here.
    ///
    /// PATH A (load-time only, no quantize): each HF 2D expert tensor is dequantized to f32 and the
    /// per-expert blocks are concatenated expert-axis-slowest into ONE contiguous buffer — exactly the
    /// layout `expert_bytes(e)` slices and the staged `qmatvec_view` (qtype=QT_F32) reads. The same
    /// `expert_stride == out_f*row_bytes` invariant as the GGUF path is asserted at the end.
    ///
    /// `ggml_exps_name` is `blk.{il}.ffn_{gate,up,down}_exps.weight`; it is split to recover `il` and
    /// the proj. `n_expert` comes from `cfg.moe`. The HF per-expert literal `mlp.experts.{e}.{p}_proj`
    /// is the qwen3moe / olmoe layout (a future arch with `block_sparse_moe.experts.*` would need a
    /// branch in `hf_expert_name`).
    pub fn load_from_source(e: &Engine, src: &dyn TensorSource, ggml_exps_name: &str, n_expert: usize)
                            -> Result<Self, Box<dyn std::error::Error>> {
        // Recover il + proj from `blk.{il}.ffn_{gate,up,down}_exps.weight`.
        let rest = ggml_exps_name.strip_prefix("blk.")
            .unwrap_or_else(|| panic!("not a blk.* name: {ggml_exps_name}"));
        let (il_s, suffix) = rest.split_once('.').unwrap();
        let il: u32 = il_s.parse().unwrap();
        let proj = match suffix {
            "ffn_gate_exps.weight" => "gate",
            "ffn_up_exps.weight"   => "up",
            "ffn_down_exps.weight" => "down",
            other => panic!("not a *_exps suffix: {other}"),
        };

        // expert 0 fixes (in_f, out_f); every later expert must match (catches a layer/arch mixup).
        let mut buf: Vec<u8> = Vec::new();
        let mut in_f = 0usize;
        let mut out_f = 0usize;
        for ex in 0..n_expert {
            // Per-expert ggml name; the source maps it to the HF expert tensor (ST-MOE-PLAN §1.3).
            let name = format!("blk.{il}.ffn_{proj}_exps.{ex}.weight");
            let v = src.find(&name).unwrap_or_else(|| panic!("missing expert tensor {name}"));
            assert_eq!(v.ne.len(), 2, "expert {name} is not 2D (ne={:?})", v.ne);
            let (cur_in, cur_out) = (v.ne[0] as usize, v.ne[1] as usize);
            if ex == 0 { in_f = cur_in; out_f = cur_out; }
            else { assert_eq!((cur_in, cur_out), (in_f, out_f),
                "expert {ex} dims {:?} != expert 0 [{in_f},{out_f}]", (cur_in, cur_out)); }
            // PATH A: dequant the 2D expert (F32/F16/BF16) to f32, append its bytes verbatim. The
            // dequantized [out_f, in_f] row-major f32 block is exactly one expert_stride slow→fast.
            let n = cur_in * cur_out;
            let f32v = dequant::dequantize(v.ggml_type, &v.bytes, n);
            buf.reserve(n * 4);
            for f in &f32v { buf.extend_from_slice(&f.to_le_bytes()); }
        }
        let row_bytes = in_f * 4;                 // one out-row = in_f contiguous f32s
        let expert_stride = out_f * row_bytes;
        assert_eq!(buf.len(), n_expert * expert_stride,
            "{ggml_exps_name} gather size {} != n_expert*stride {}", buf.len(), n_expert * expert_stride);
        // Hold to the identical invariant as the GGUF path (ST-MOE-PLAN §1.3 step 4).
        assert_eq!(expert_stride, out_f * row_bytes,
            "{ggml_exps_name} stride mismatch: stride={expert_stride} out_f={out_f} row_bytes={row_bytes}");

        // Same pinned-vs-paged choice as the GGUF loader (the bytes are H2D-only on the hot path).
        let pinned = std::env::var("BW24_MOE_PINNED").is_ok() || std::env::var("BW24_MOE_CACHE").is_ok();
        let bytes = if pinned {
            let mut p = unsafe { e.ctx().alloc_pinned::<u8>(buf.len())? };
            { let dst = p.as_mut_slice()?; dst.copy_from_slice(&buf); }
            let base = p.as_ptr()? as *const u8;
            let len = buf.len();
            HostBuf::Pinned { slice: p, base, len }
        } else {
            HostBuf::Paged(buf)
        };
        Ok(HostExps { bytes, tiers: None, qtype: QT_F32, in_f, out_f, n_expert, row_bytes, expert_stride })
    }

    /// Host byte slice for expert `e` (the H2D DMA source). Contiguous block, offset honored.
    /// Resolves the per-expert tier when spilling is active (`tiers` Some), else slices the single
    /// backing store (unchanged in-RAM path). Each `tiers[e]` is exactly one expert's stride.
    #[inline]
    pub fn expert_bytes(&self, e: usize) -> &[u8] {
        debug_assert!(e < self.n_expert, "expert index {e} >= n_expert {}", self.n_expert);
        match &self.tiers {
            Some(tiers) => tiers[e].as_bytes(),
            None => &self.bytes.as_bytes()[e * self.expert_stride..(e + 1) * self.expert_stride],
        }
    }
}
