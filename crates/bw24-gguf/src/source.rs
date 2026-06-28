//! Source-agnostic weight access: a `TensorSource` trait the engine loads from, implemented by
//! BOTH the GGUF reader and the safetensors reader. The engine only ever asks for ggml-style
//! names + gets back `{ggml_type, ne, &[u8]}`; the source hides where the bytes come from.
//!
//! This trait lives in bw24-gguf (not bw24-engine) because it returns bw24-gguf types
//! (`GgmlType`, `ModelConfig`) and both readers live here. bw24-engine already depends on
//! bw24-gguf, so `GpuTensor::load_from_source(&dyn TensorSource, ...)` introduces no new dep.

use std::borrow::Cow;
use crate::config::{Arch, ModelConfig};
use crate::safetensors::StModel;
use crate::{GgufFile, GgmlType};

/// A view of one tensor's data, source-agnostic.
///
/// `bytes` is a `Cow`: the GGUF path and the zero-copy dense safetensors path BORROW the mmap
/// (no allocation); the hybrid-SSM transforms (`-exp(A_log)`, norm `+1`, conv1d squeeze, V-reorder)
/// produce an OWNED buffer (ST-MOE-PLAN §2.1) since they cannot be expressed as a borrow of the
/// on-disk bytes. All consumers read it as `&[u8]` via `&v.bytes`, so the fast path is untouched.
pub struct TensorView<'a> {
    pub bytes: Cow<'a, [u8]>,
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
    /// The backing GGUF file, if this source IS a GGUF (None for safetensors). Used by the
    /// disk-spill tier, which needs the on-disk file mmap (`g.path()` / per-expert byte ranges).
    fn gguf(&self) -> Option<&GgufFile> { None }
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
            bytes: Cow::Borrowed(self.0.tensor_data(t)),
            ggml_type: t.ggml_type,
            ne: t.ne.clone(),
        })
    }
    fn gguf(&self) -> Option<&GgufFile> { Some(self.0) }
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

    /// Direct HF-name access (zero-copy). Applies the prefix-fallback so a wrapper prefix like
    /// `model.language_model.` (qwen35 VLM) resolves against the plain `model.` namespace and vice
    /// versa (ST-MOE-PLAN §2.0). Returns a BORROWED view (no transform).
    pub fn raw_hf(&self, hf_name: &str) -> Option<TensorView<'_>> {
        let (info, bytes) = self.lookup(hf_name)?;
        Some(TensorView { bytes: Cow::Borrowed(bytes), ggml_type: info.ggml_type(), ne: info.ne() })
    }

    /// Resolve an HF tensor name, trying it verbatim then with the qwen35 multimodal wrapper prefix
    /// inserted/removed (`model.` <-> `model.language_model.`). The dense map and the SSM map share
    /// one `model.layers.{il}.` namespace this way (ST-MOE-PLAN §2.0).
    fn lookup(&self, hf_name: &str) -> Option<(&crate::safetensors::StInfo, &[u8])> {
        if let Some(r) = self.model.raw(hf_name) { return Some(r); }
        // model.layers.* -> model.language_model.layers.*  (and the symmetric strip)
        if let Some(rest) = hf_name.strip_prefix("model.") {
            if !rest.starts_with("language_model.") && !rest.starts_with("visual.") {
                let alt = format!("model.language_model.{rest}");
                if let Some(r) = self.model.raw(&alt) { return Some(r); }
            }
        }
        if let Some(rest) = hf_name.strip_prefix("model.language_model.") {
            let alt = format!("model.{rest}");
            if let Some(r) = self.model.raw(&alt) { return Some(r); }
        }
        None
    }

    /// Dequantize an HF tensor to f32 (used by the value-transform producers). Handles BOTH plain
    /// F32/F16/BF16 tensors AND modelopt (compressed-tensors) NVFP4 weights: a `<name>.weight` stored
    /// `U8` with a sibling `<name>.weight_scale` is dequantized through the NVFP4 path (per-16 UE4M3
    /// block scale × the per-tensor `weight_scale_2`), so the hybrid SSM V-reorder transforms (which
    /// operate on f32) work on an NVFP4 checkpoint exactly as on a BF16 one.
    fn deq_f32(&self, hf_name: &str) -> Option<(Vec<f32>, Vec<u64>)> {
        let (info, bytes) = self.lookup(hf_name)?;
        // modelopt NVFP4 weight? (U8 .weight with a sibling .weight_scale)
        if info.dtype == "U8" && hf_name.ends_with(".weight") {
            if let Some((out_f, in_f, wscale, scale2)) = self.modelopt_nvfp4(hf_name) {
                use crate::nvfp4_repack::dequant_modelopt_row;
                let in_bytes = in_f / 2;
                let scl_bytes = in_f / 16;
                let mut data = vec![0f32; out_f * in_f];
                for o in 0..out_f {
                    let row = dequant_modelopt_row(
                        &bytes[o * in_bytes..(o + 1) * in_bytes],
                        &wscale[o * scl_bytes..(o + 1) * scl_bytes],
                        in_f,
                    );
                    for (e, v) in row.iter().enumerate() {
                        data[o * in_f + e] = v * scale2; // fold the per-tensor macro-scale into f32
                    }
                }
                return Some((data, vec![in_f as u64, out_f as u64]));
            }
        }
        let ne = info.ne();
        let n: u64 = ne.iter().product();
        Some((crate::dequant::dequantize(info.ggml_type(), bytes, n as usize), ne))
    }

    /// If `hf_weight` (a `<name>.weight`) is a modelopt NVFP4 quantized Linear, return
    /// `(out_f, in_f, weight_scale_bytes, weight_scale_2)`. `None` otherwise (plain dense weight, or
    /// missing siblings). `out_f`/`in_f` are the logical [out, in] dims (weight is [out, in/2] U8).
    fn modelopt_nvfp4(&self, hf_weight: &str) -> Option<(usize, usize, &[u8], f32)> {
        let (winfo, _) = self.lookup(hf_weight)?;
        if winfo.dtype != "U8" || winfo.shape.len() != 2 {
            return None;
        }
        let stem = hf_weight.strip_suffix(".weight")?;
        let (sinfo, sbytes) = self.lookup(&format!("{stem}.weight_scale"))?;
        if sinfo.dtype != "F8_E4M3" {
            return None;
        }
        let out_f = winfo.shape[0] as usize; // HF row-major [out, in/2]
        let in_f = (winfo.shape[1] as usize) * 2; // logical in-features (U8 packs 2 codes/byte)
        // weight_scale_2 is a per-tensor F32 scalar; default to 1.0 if absent (then macro-scale==1).
        let scale2 = match self.lookup(&format!("{stem}.weight_scale_2")) {
            Some((_, b)) if b.len() >= 4 => f32::from_le_bytes(b[..4].try_into().unwrap()),
            _ => 1.0,
        };
        Some((out_f, in_f, sbytes, scale2))
    }
}

impl TensorSource for SafetensorsSource {
    fn config(&self) -> ModelConfig {
        self.cfg.clone()
    }
    fn find(&self, ggml_name: &str) -> Option<TensorView<'_>> {
        use crate::hf_mapping::{HfTarget, resolve_ggml};
        // NVFP4 per-tensor macro-scale sibling: the engine asks for `<stem>.scale` (model.rs) and
        // expects an F32 scalar. Map `<stem>.scale` -> the modelopt `<hf>.weight_scale_2`. Returns
        // None for non-quantized weights (then the engine defaults the macro-scale to 1.0).
        if let Some(stem) = ggml_name.strip_suffix(".scale") {
            let hf_weight = match resolve_ggml(&format!("{stem}.weight"), &self.cfg)? {
                HfTarget::Plain(hf) | HfTarget::Transform { hf, .. } => hf,
            };
            let s2 = format!("{}.weight_scale_2", hf_weight.strip_suffix(".weight")?);
            let (info, bytes) = self.lookup(&s2)?;
            // weight_scale_2 is a 0-dim F32 scalar (4 bytes); surface it as ne=[1] F32.
            return Some(TensorView {
                bytes: Cow::Borrowed(bytes),
                ggml_type: info.ggml_type(),
                ne: vec![1],
            });
        }
        match resolve_ggml(ggml_name, &self.cfg)? {
            // Zero-copy: a plain rename (dense path + most SSM matrices), borrow the mmap directly.
            // NVFP4 modelopt weights take the repack arm (owned GGUF block bytes); else borrow.
            HfTarget::Plain(hf) => {
                if let Some((out_f, in_f, wscale, _scale2)) = self.modelopt_nvfp4(&hf) {
                    // Repack modelopt NVFP4 -> bw24 internal GGUF block_nvfp4 bytes (NO kernel change).
                    let (winfo, wbytes) = self.lookup(&hf)?;
                    let _ = winfo;
                    let packed = crate::nvfp4_repack::repack_modelopt_to_gguf(wbytes, wscale, out_f, in_f);
                    return Some(TensorView {
                        bytes: Cow::Owned(packed),
                        ggml_type: GgmlType::NVFP4,
                        ne: vec![in_f as u64, out_f as u64], // ggml ne: [in, out]
                    });
                }
                self.raw_hf(&hf)
            }
            // Owned f32 buffer: a value transform that cannot borrow the on-disk bytes (§2.1).
            // `deq_f32` is NVFP4-aware, so an NVFP4 SSM weight dequants through the modelopt path.
            HfTarget::Transform { hf, kind } => {
                let (mut data, ne_in) = self.deq_f32(&hf)?;
                let cfg = &self.cfg;
                let (ne, bytes) = kind.apply(&mut data, ne_in, cfg);
                Some(TensorView { bytes: Cow::Owned(bytes), ggml_type: GgmlType::F32, ne })
            }
        }
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
