//! Dense transformer model: loads GGUF weights to GPU (Stage-1: dequant→f32), runs the
//! shared full-attention + SwiGLU forward graph. Arch-agnostic via ModelConfig; this path is
//! exactly the dense-transformer graph (qwen3) and the full-attention layers of hybrids.

use std::collections::HashMap;
use cudarc::driver::CudaSlice;
use bw24_gguf::{GgufFile, dequant};
use bw24_gguf::config::ModelConfig;
use crate::Engine;

/// A weight tensor resident on GPU as f32 (Stage-1). shape = ggml ne (ne[0] fastest).
pub struct GpuTensor {
    pub data: CudaSlice<f32>,
    pub ne: Vec<u64>,
}

impl GpuTensor {
    pub fn in_features(&self) -> usize { self.ne[0] as usize }
    pub fn out_features(&self) -> usize { self.ne[1] as usize }
}

pub struct Layer {
    pub attn_norm: GpuTensor,
    pub wq: GpuTensor, pub wk: GpuTensor, pub wv: GpuTensor, pub wo: GpuTensor,
    pub q_norm: Option<GpuTensor>, pub k_norm: Option<GpuTensor>,
    pub ffn_norm: GpuTensor,
    pub ffn_gate: GpuTensor, pub ffn_up: GpuTensor, pub ffn_down: GpuTensor,
}

pub struct Model {
    pub cfg: ModelConfig,
    pub tok_embd: GpuTensor,
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

        let load = |name: &str| -> Result<GpuTensor, Box<dyn std::error::Error>> {
            let t = g.find(name).unwrap_or_else(|| panic!("missing tensor {name}"));
            let n = t.n_elements() as usize;
            let f32v = dequant::dequantize(t.ggml_type, g.tensor_data(t), n);
            Ok(GpuTensor { data: e.htod(&f32v)?, ne: t.ne.clone() })
        };
        let load_opt = |name: &str| -> Result<Option<GpuTensor>, Box<dyn std::error::Error>> {
            match g.find(name) {
                Some(t) => {
                    let n = t.n_elements() as usize;
                    let f32v = dequant::dequantize(t.ggml_type, g.tensor_data(t), n);
                    Ok(Some(GpuTensor { data: e.htod(&f32v)?, ne: t.ne.clone() }))
                }
                None => Ok(None),
            }
        };

        let tok_embd = load("token_embd.weight")?;
        let output_norm = load("output_norm.weight")?;
        // tied embeddings: fall back to tok_embd if output.weight absent
        let output = match g.find("output.weight") {
            Some(_) => load("output.weight")?,
            None => load("token_embd.weight")?,
        };

        let mut layers = Vec::with_capacity(cfg.n_layer as usize);
        for il in 0..cfg.n_layer {
            layers.push(Layer {
                attn_norm: load(&format!("blk.{il}.attn_norm.weight"))?,
                wq: load(&format!("blk.{il}.attn_q.weight"))?,
                wk: load(&format!("blk.{il}.attn_k.weight"))?,
                wv: load(&format!("blk.{il}.attn_v.weight"))?,
                wo: load(&format!("blk.{il}.attn_output.weight"))?,
                q_norm: load_opt(&format!("blk.{il}.attn_q_norm.weight"))?,
                k_norm: load_opt(&format!("blk.{il}.attn_k_norm.weight"))?,
                ffn_norm: load(&format!("blk.{il}.ffn_norm.weight"))?,
                ffn_gate: load(&format!("blk.{il}.ffn_gate.weight"))?,
                ffn_up: load(&format!("blk.{il}.ffn_up.weight"))?,
                ffn_down: load(&format!("blk.{il}.ffn_down.weight"))?,
            });
        }
        Ok(Model { cfg, tok_embd, output_norm, output, layers })
    }

    /// Gather embedding rows for a token sequence into a host f32 [n_embd, T] (row-major per token).
    /// token_embd is [n_embd, n_vocab] (ne[0]=n_embd fastest), so row for token t starts at t*n_embd.
    pub fn embed_tokens(&self, e: &Engine, tokens: &[u32]) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let n_embd = self.cfg.n_embd as usize;
        let embd_host = e.dtoh(&self.tok_embd.data)?; // [n_vocab * n_embd]
        let mut x = vec![0f32; tokens.len() * n_embd];
        for (ti, &tok) in tokens.iter().enumerate() {
            let src = tok as usize * n_embd;
            x[ti * n_embd..ti * n_embd + n_embd].copy_from_slice(&embd_host[src..src + n_embd]);
        }
        Ok(e.htod(&x)?)
    }
}

pub type TensorMap = HashMap<String, GpuTensor>;
