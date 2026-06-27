//! Dense transformer model: loads GGUF weights to GPU (Stage-1: dequant→f32), runs the
//! shared full-attention + SwiGLU forward graph. Arch-agnostic via ModelConfig; this path is
//! exactly the dense-transformer graph (qwen3) and the full-attention layers of hybrids.

use std::collections::HashMap;
use cudarc::driver::CudaSlice;
use bw24_gguf::{GgufFile, GgmlType, dequant};
use bw24_gguf::config::ModelConfig;
use bw24_gguf::source::{TensorSource, GgufSource};
use crate::{Engine, QT_Q8_0, QT_Q4_K, QT_Q6_K, QT_Q5_K, QT_Q3_K, QT_IQ4_XS, QT_IQ3_S, QT_NVFP4};

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
                Ok(GpuTensor::Quant { bytes: e.htod_bytes(v.bytes)?, qtype: qt, row_bytes, ne: v.ne.clone(), scale })
            }
            None => {
                // F32/F16/BF16 (or as-yet-unhandled quant): dequant to f32. Small tensors only.
                let n: u64 = v.ne.iter().product();
                let f32v = dequant::dequantize(v.ggml_type, v.bytes, n as usize);
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
    pub ffn_gate: GpuTensor, pub ffn_up: GpuTensor, pub ffn_down: GpuTensor,
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

    /// Load a dense model from any `TensorSource` — GGUF or a safetensors HF checkpoint.
    /// The whole loop speaks ggml names; the source maps them. Panics on SSM/MoE arches.
    pub fn load_dense_from_source(e: &Engine, src: &dyn TensorSource) -> Result<Self, Box<dyn std::error::Error>> {
        let cfg = src.config();
        assert!(cfg.full_attention_interval == 0, "model has linear-attn layers; use hybrid path");
        assert!(cfg.moe.is_none(), "model is MoE; use MoE path");

        let embd = EmbedHost::from_source(src, "token_embd.weight");
        let output_norm = GpuTensor::load_from_source(e, src, "output_norm.weight")?;
        // tied embeddings: fall back to tok_embd if output.weight absent
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
                ffn_gate: GpuTensor::load_from_source(e, src, &p("ffn_gate.weight"))?,
                ffn_up: GpuTensor::load_from_source(e, src, &p("ffn_up.weight"))?,
                ffn_down: GpuTensor::load_from_source(e, src, &p("ffn_down.weight"))?,
            });
        }
        Ok(Model { cfg, embd, output_norm, output, layers })
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
}
// SAFETY: `base` is a stable pinned-host pointer owned by `slice`; the buffer is written once at load
// then only READ for H2D. HostExps is shared `&` across the (single per-Engine) forward, so Send/Sync
// mirror the underlying PinnedHostSlice (which is already Send+Sync).
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
        }
    }
    #[inline]
    pub fn len(&self) -> usize {
        match self { HostBuf::Paged(v) => v.len(), HostBuf::Pinned { len, .. } => *len }
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
        let t = g.find(name).unwrap_or_else(|| panic!("missing exps tensor {name}"));
        assert_eq!(t.ne.len(), 3, "{name} is not a 3D stacked-expert tensor (ne={:?})", t.ne);
        let raw = g.tensor_data(t);
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
        Ok(HostExps { bytes, qtype, in_f, out_f, n_expert, row_bytes, expert_stride })
    }

    /// Host byte slice for expert `e` (the H2D DMA source). Contiguous block, offset honored.
    #[inline]
    pub fn expert_bytes(&self, e: usize) -> &[u8] {
        debug_assert!(e < self.n_expert, "expert index {e} >= n_expert {}", self.n_expert);
        &self.bytes.as_bytes()[e * self.expert_stride..(e + 1) * self.expert_stride]
    }
}
