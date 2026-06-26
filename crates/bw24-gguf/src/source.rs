//! Source-agnostic weight access: a `TensorSource` trait the engine loads from, implemented by
//! BOTH the GGUF reader and the safetensors reader. The engine only ever asks for ggml-style
//! names + gets back `{ggml_type, ne, &[u8]}`; the source hides where the bytes come from.
//!
//! This trait lives in bw24-gguf (not bw24-engine) because it returns bw24-gguf types
//! (`GgmlType`, `ModelConfig`) and both readers live here. bw24-engine already depends on
//! bw24-gguf, so `GpuTensor::load_from_source(&dyn TensorSource, ...)` introduces no new dep.

use crate::config::{Arch, ModelConfig};
use crate::safetensors::StModel;
use crate::{GgufFile, GgmlType};

/// A view of one tensor's data, source-agnostic.
pub struct TensorView<'a> {
    pub bytes: &'a [u8],
    pub ggml_type: GgmlType,
    pub ne: Vec<u64>, // inner-fastest (ne[0] = in_features for a [in,out] weight)
}

/// A weight source the engine can load from. GGUF and safetensors both implement it.
pub trait TensorSource {
    /// The model configuration (from GGUF metadata or config.json).
    fn config(&self) -> ModelConfig;
    /// Find a tensor by its **ggml-style** name. Returns None if absent/unmapped.
    fn find(&self, ggml_name: &str) -> Option<TensorView<'_>>;
    /// Whether a ggml-named tensor is present.
    fn has(&self, ggml_name: &str) -> bool {
        self.find(ggml_name).is_some()
    }
}

/// GGUF-backed source (the existing path). Zero behavior change vs. direct GgufFile use.
pub struct GgufSource<'g>(pub &'g GgufFile);

impl<'g> TensorSource for GgufSource<'g> {
    fn config(&self) -> ModelConfig {
        ModelConfig::from_gguf(self.0)
    }
    fn find(&self, name: &str) -> Option<TensorView<'_>> {
        let t = self.0.find(name)?;
        Some(TensorView {
            bytes: self.0.tensor_data(t),
            ggml_type: t.ggml_type,
            ne: t.ne.clone(),
        })
    }
}

/// safetensors-backed source: an HF checkpoint (config.json + one/more .safetensors shards).
/// `find` translates the requested ggml name into the HF name, looks it up, and reverses the
/// shape into ggml `ne` order.
pub struct SafetensorsSource {
    model: StModel,
    cfg: ModelConfig,
}

impl SafetensorsSource {
    /// Open an HF model directory: expects a `config.json` plus `model.safetensors`
    /// (single) or `model.safetensors.index.json` (+ shards). `dir` may also be a direct
    /// path to a single `.safetensors` file (config.json must then sit beside it).
    pub fn open(path: &std::path::Path) -> std::io::Result<Self> {
        let dir = if path.is_file() {
            path.parent().unwrap_or(std::path::Path::new("."))
        } else {
            path
        };
        let cfg = ModelConfig::from_config_json(&dir.join("config.json"))?;
        let model = StModel::open(path)?;
        Ok(Self { model, cfg })
    }

    /// Open with an explicitly-provided config (e.g. tests, or config.json elsewhere).
    pub fn open_with_config(path: &std::path::Path, cfg: ModelConfig) -> std::io::Result<Self> {
        let model = StModel::open(path)?;
        Ok(Self { model, cfg })
    }

    pub fn arch(&self) -> &Arch {
        &self.cfg.arch
    }

    /// Direct HF-name access (used by the MoE expert gather path; not arch-mapped).
    pub fn raw_hf(&self, hf_name: &str) -> Option<TensorView<'_>> {
        let (info, bytes) = self.model.raw(hf_name)?;
        Some(TensorView { bytes, ggml_type: info.ggml_type(), ne: info.ne() })
    }
}

impl TensorSource for SafetensorsSource {
    fn config(&self) -> ModelConfig {
        self.cfg.clone()
    }
    fn find(&self, ggml_name: &str) -> Option<TensorView<'_>> {
        let hf = crate::hf_mapping::ggml_to_hf(ggml_name, &self.cfg.arch)?;
        self.raw_hf(&hf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safetensors_source_find_maps_names() {
        // Build a tiny HF-named safetensors file + config.json, open via SafetensorsSource,
        // and assert ggml-name lookups resolve through the mapper with reversed shape.
        let dir = std::env::temp_dir().join(format!("bw24_src_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // one F32 attn_q weight, HF shape [out=4, in=2] -> ne should be [2,4]
        let json = r#"{"model.layers.0.self_attn.q_proj.weight":{"dtype":"F32","shape":[4,2],"data_offsets":[0,32]}}"#;
        let mut buf = Vec::new();
        buf.extend_from_slice(&(json.len() as u64).to_le_bytes());
        buf.extend_from_slice(json.as_bytes());
        for v in 0..8u32 {
            buf.extend_from_slice(&(v as f32).to_le_bytes());
        }
        std::fs::write(dir.join("model.safetensors"), &buf).unwrap();

        let cfg_json = r#"{"model_type":"qwen3","num_hidden_layers":1,"hidden_size":4,"num_attention_heads":2,"intermediate_size":8,"vocab_size":10,"max_position_embeddings":128}"#;
        std::fs::write(dir.join("config.json"), cfg_json).unwrap();

        let src = SafetensorsSource::open(&dir).unwrap();
        assert_eq!(src.config().arch, Arch::Qwen3);
        assert_eq!(src.config().n_layer, 1);

        let v = src.find("blk.0.attn_q.weight").expect("ggml name maps to HF and resolves");
        assert_eq!(v.ggml_type, GgmlType::F32);
        // shape-reversal assertion: HF [out=4,in=2] -> ne [in=2,out=4]
        assert_eq!(v.ne, vec![2, 4]);
        assert_eq!(v.bytes.len(), 32);
        assert!(src.has("blk.0.attn_q.weight"));
        // unmapped ggml name (no SSM tensors in this dense model)
        assert!(src.find("blk.0.ssm_a").is_none());

        std::fs::remove_dir_all(&dir).ok();
    }
}
