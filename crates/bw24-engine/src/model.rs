//! Dense transformer model: loads GGUF weights to GPU (Stage-1: dequant→f32), runs the
//! shared full-attention + SwiGLU forward graph. Arch-agnostic via ModelConfig; this path is
//! exactly the dense-transformer graph (qwen3) and the full-attention layers of hybrids.

use std::collections::HashMap;
use cudarc::driver::CudaSlice;
use bw24_gguf::{GgufFile, GgmlType, dequant};
use bw24_gguf::config::ModelConfig;
use crate::{Engine, QT_Q8_0, QT_Q4_K, QT_Q6_K};

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
