//! Dense transformer model: loads GGUF weights to GPU (Stage-1: dequant→f32), runs the
//! shared full-attention + SwiGLU forward graph. Arch-agnostic via ModelConfig; this path is
//! exactly the dense-transformer graph (qwen3) and the full-attention layers of hybrids.

use std::collections::HashMap;
use cudarc::driver::CudaSlice;
use bw24_gguf::{GgufFile, GgmlType, dequant};
use bw24_gguf::config::ModelConfig;
use crate::{Engine, QT_Q8_0, QT_Q4_K, QT_Q6_K, QT_Q5_K, QT_Q3_K, QT_IQ4_XS, QT_IQ3_S, QT_NVFP4};

/// A weight tensor resident on GPU. Quantized weights stay in GGUF block bytes (`Quant`);
/// small non-quant tensors (norms, sometimes embed/lm_head) are kept dequantized as f32 (`Float`).
/// This keeps VRAM ~= on-disk quant size (fixes the f32-on-load OOM).
pub enum GpuTensor {
    Quant { bytes: CudaSlice<u8>, qtype: i32, row_bytes: usize, ne: Vec<u64> },
    Float { data: CudaSlice<f32>, ne: Vec<u64> },
}

impl GpuTensor {
    pub fn ne(&self) -> &[u64] { match self { GpuTensor::Quant { ne, .. } => ne, GpuTensor::Float { ne, .. } => ne } }
    pub fn in_features(&self) -> usize { self.ne()[0] as usize }
    pub fn out_features(&self) -> usize { self.ne()[1] as usize }

    /// Load a tensor, keeping quant types packed and float types as f32.
    pub fn load(e: &Engine, g: &GgufFile, name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let t = g.find(name).unwrap_or_else(|| panic!("missing tensor {name}"));
        let raw = g.tensor_data(t);
        let qtype = match t.ggml_type {
            GgmlType::Q8_0 => Some(QT_Q8_0),
            GgmlType::Q4_K => Some(QT_Q4_K),
            GgmlType::Q6_K => Some(QT_Q6_K),
            GgmlType::Q5_K => Some(QT_Q5_K),
            GgmlType::Q3_K => Some(QT_Q3_K),
            GgmlType::IQ4_XS => Some(QT_IQ4_XS),
            GgmlType::IQ3_S => Some(QT_IQ3_S),
            GgmlType::NVFP4 => Some(QT_NVFP4),
            _ => None,
        };
        match qtype {
            Some(qt) => {
                let out_f = t.ne[1] as usize;
                let row_bytes = raw.len() / out_f;
                Ok(GpuTensor::Quant { bytes: e.htod_bytes(raw)?, qtype: qt, row_bytes, ne: t.ne.clone() })
            }
            None => {
                // F32/F16/BF16 (or as-yet-unhandled quant): dequant to f32. Small tensors only.
                let n = t.n_elements() as usize;
                let f32v = dequant::dequantize(t.ggml_type, raw, n);
                Ok(GpuTensor::Float { data: e.htod(&f32v)?, ne: t.ne.clone() })
            }
        }
    }

    pub fn load_opt(e: &Engine, g: &GgufFile, name: &str) -> Result<Option<Self>, Box<dyn std::error::Error>> {
        match g.find(name) { Some(_) => Ok(Some(Self::load(e, g, name)?)), None => Ok(None) }
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
        let t = g.find(name).unwrap_or_else(|| panic!("missing embed {name}"));
        EmbedHost { raw: g.tensor_data(t).to_vec(), ggml_type: t.ggml_type, n_embd: t.ne[0] as usize }
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
    /// Load a dense (vanilla-transformer) model. Panics if the arch has SSM/MoE layers
    /// (those need the hybrid/MoE path, not yet wired).
    pub fn load_dense(e: &Engine, g: &GgufFile) -> Result<Self, Box<dyn std::error::Error>> {
        let cfg = ModelConfig::from_gguf(g);
        assert!(cfg.full_attention_interval == 0, "model has linear-attn layers; use hybrid path");
        assert!(cfg.moe.is_none(), "model is MoE; use MoE path");

        let embd = EmbedHost::from_gguf(g, "token_embd.weight");
        let output_norm = GpuTensor::load(e, g, "output_norm.weight")?;
        // tied embeddings: fall back to tok_embd if output.weight absent
        let output = match g.find("output.weight") {
            Some(_) => GpuTensor::load(e, g, "output.weight")?,
            None => GpuTensor::load(e, g, "token_embd.weight")?,
        };

        let mut layers = Vec::with_capacity(cfg.n_layer as usize);
        for il in 0..cfg.n_layer {
            let p = |s: &str| format!("blk.{il}.{s}");
            layers.push(Layer {
                attn_norm: GpuTensor::load(e, g, &p("attn_norm.weight"))?,
                wq: GpuTensor::load(e, g, &p("attn_q.weight"))?,
                wk: GpuTensor::load(e, g, &p("attn_k.weight"))?,
                wv: GpuTensor::load(e, g, &p("attn_v.weight"))?,
                wo: GpuTensor::load(e, g, &p("attn_output.weight"))?,
                q_norm: GpuTensor::load_opt(e, g, &p("attn_q_norm.weight"))?,
                k_norm: GpuTensor::load_opt(e, g, &p("attn_k_norm.weight"))?,
                ffn_norm: GpuTensor::load(e, g, &p("ffn_norm.weight"))?,
                ffn_gate: GpuTensor::load(e, g, &p("ffn_gate.weight"))?,
                ffn_up: GpuTensor::load(e, g, &p("ffn_up.weight"))?,
                ffn_down: GpuTensor::load(e, g, &p("ffn_down.weight"))?,
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
        Ok(HostExps { bytes: raw.to_vec(), qtype, in_f, out_f, n_expert, row_bytes, expert_stride })
    }

    /// Host byte slice for expert `e` (the H2D DMA source). Contiguous block, offset honored.
    #[inline]
    pub fn expert_bytes(&self, e: usize) -> &[u8] {
        debug_assert!(e < self.n_expert, "expert index {e} >= n_expert {}", self.n_expert);
        &self.bytes[e * self.expert_stride..(e + 1) * self.expert_stride]
    }
}
