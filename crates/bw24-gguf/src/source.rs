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

/// Raw NVFP4-native (modelopt/Reza) weight view: the packed e2m1 codes + per-16 UE4M3 scales
/// exactly as they sit in the file, for the engine's DIRECT split-plane repack (A1 direct import).
/// Only returned for a PLAIN (untransformed) quantized Linear; anything needing a V-reorder
/// transform keeps the GGUF-block path. The per-tensor macro-scale still rides the `<stem>.scale`
/// sibling via `find` (identical to the GGUF NVFP4 path).
pub struct Nvfp4Native<'a> {
    pub wbytes: &'a [u8], // packed e2m1, [out_f, in_f/2] row-major, 2 codes/byte
    pub wscale: &'a [u8], // UE4M3 per-16 scales, [out_f, in_f/16] row-major
    pub out_f: usize,
    pub in_f: usize,
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
    /// NVFP4-native access (A1 direct import): the raw modelopt/Reza packed codes + scales for a
    /// plain (untransformed) NVFP4 weight, so the engine can repack straight into its split-plane
    /// resident layout without materializing GGUF 36B blocks. None for GGUF sources (already the
    /// import layout), for transformed weights, and for non-NVFP4 tensors.
    fn find_nvfp4_native(&self, _ggml_name: &str) -> Option<Nvfp4Native<'_>> { None }
    /// The checkpoint directory, if this source is a safetensors HF dir (None for GGUF). Used by
    /// the ST expert disk-tier to place its repack cache next to the shards.
    fn st_dir(&self) -> Option<&std::path::Path> { None }
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
    dir: std::path::PathBuf,
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
        Ok(Self { model, cfg, dir: dir.to_path_buf() })
    }

    /// Open with an explicitly-provided config (e.g. tests, or config.json elsewhere).
    pub fn open_with_config(path: &std::path::Path, cfg: ModelConfig) -> std::io::Result<Self> {
        let model = StModel::open(path)?;
        let dir = if path.is_file() {
            path.parent().unwrap_or(std::path::Path::new(".")).to_path_buf()
        } else { path.to_path_buf() };
        Ok(Self { model, cfg, dir })
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
        // MiniMax-M3-VL nests the OTHER way round: `language_model.model.layers.*` /
        // `language_model.lm_head.weight` (whole text model under a `language_model.` root).
        if hf_name.starts_with("model.") || hf_name == "lm_head.weight" {
            let alt = format!("language_model.{hf_name}");
            if let Some(r) = self.model.raw(&alt) { return Some(r); }
        }
        if let Some(rest) = hf_name.strip_prefix("language_model.") {
            if let Some(r) = self.model.raw(rest) { return Some(r); }
        }
        None
    }

    /// Dequantize an HF tensor to f32 (used by the value-transform producers). Handles BOTH plain
    /// F32/F16/BF16 tensors AND modelopt (compressed-tensors) NVFP4 weights: a `<name>.weight` stored
    /// `U8` with a sibling `<name>.weight_scale` is dequantized through the NVFP4 path (per-16 UE4M3
    /// block scale × the per-tensor `weight_scale_2`), so the hybrid SSM V-reorder transforms (which
    /// operate on f32) work on an NVFP4 checkpoint exactly as on a BF16 one.
    fn deq_f32(&self, hf_name: &str) -> Option<(Vec<f32>, Vec<u64>)> {
        // NVFP4 weight (modelopt OR Reza)? Dequant through the NVFP4 path so the hybrid SSM V-reorder
        // transforms (which operate on f32) work on an NVFP4 checkpoint exactly as on a BF16 one.
        if hf_name.ends_with(".weight") {
            if let Some((out_f, in_f, wbytes, wscale, macro_s)) = self.nvfp4_quant(hf_name) {
                use crate::nvfp4_repack::dequant_modelopt_row;
                let in_bytes = in_f / 2;
                let scl_bytes = in_f / 16;
                let mut data = vec![0f32; out_f * in_f];
                for o in 0..out_f {
                    let row = dequant_modelopt_row(
                        &wbytes[o * in_bytes..(o + 1) * in_bytes],
                        &wscale[o * scl_bytes..(o + 1) * scl_bytes],
                        in_f,
                    );
                    for (e, v) in row.iter().enumerate() {
                        data[o * in_f + e] = v * macro_s; // fold the per-tensor macro-scale into f32
                    }
                }
                return Some((data, vec![in_f as u64, out_f as u64]));
            }
        }
        // FP8 E4M3 weight + scalar F32 weight_scale (NVIDIA 27B linear_attn class): dequant to
        // f32 here so the V-reorder transforms consume it like a BF16 tensor.
        if hf_name.ends_with(".weight") {
            if let Some((info, bytes)) = self.lookup(hf_name) {
                if info.dtype == "F8_E4M3" && info.shape.len() == 2 {
                    let stem = hf_name.strip_suffix(".weight").unwrap_or(hf_name);
                    if let Some((sinfo, sbytes)) = self.lookup(&format!("{stem}.weight_scale")) {
                        if sinfo.dtype == "F32" && sbytes.len() >= 4 {
                            let scale = f32::from_le_bytes(sbytes[..4].try_into().unwrap());
                            let ne = info.ne();
                            let data: Vec<f32> = bytes.iter()
                                .map(|&b| crate::nvfp4_repack::fp8_e4m3_to_f32(b) * scale)
                                .collect();
                            return Some((data, ne));
                        }
                    }
                }
            }
        }
        let (info, bytes) = self.lookup(hf_name)?;
        let ne = info.ne();
        let n: u64 = ne.iter().product();
        Some((crate::dequant::dequantize(info.ggml_type(), bytes, n as usize), ne))
    }

    /// Detect an HF NVFP4 quantized Linear under EITHER on-disk encoding and return everything the
    /// repack needs: `(out_f, in_f, packed_bytes, per16_fp8_scale_bytes, macro_scale)`. Both encodings
    /// store the SAME e2m1 weights + per-16 FP8(e4m3) scales — only names + macro-scale differ:
    ///   * modelopt: `<name>.weight`(U8 packed) + `<name>.weight_scale`(F8_E4M3) +
    ///     `<name>.weight_scale_2`(F32 per-tensor macro, default 1.0).
    ///   * Reza "custom_nvfp4_e2m1_e4m3_scales": `<name>.weight.nvfp4_packed`(U8) +
    ///     `<name>.weight.nvfp4_scale_e4m3`(U8/FP8 bytes), NO macro-scale (=> 1.0).
    /// `out_f`/`in_f` are the logical [out, in] dims (packed weight is [out, in/2] U8). `None` for a
    /// plain (non-quantized) weight or missing siblings.
    fn nvfp4_quant(&self, hf_weight: &str) -> Option<(usize, usize, &[u8], &[u8], f32)> {
        // modelopt: the `.weight` itself is the U8 packed tensor with a `.weight_scale` sibling.
        if let Some((winfo, wbytes)) = self.lookup(hf_weight) {
            if winfo.dtype == "U8" && winfo.shape.len() == 2 {
                let stem = hf_weight.strip_suffix(".weight")?;
                if let Some((sinfo, sbytes)) = self.lookup(&format!("{stem}.weight_scale")) {
                    if sinfo.dtype == "F8_E4M3" {
                        let out_f = winfo.shape[0] as usize; // HF row-major [out, in/2]
                        let in_f = (winfo.shape[1] as usize) * 2; // U8 packs 2 codes/byte
                        let macro_s = match self.lookup(&format!("{stem}.weight_scale_2")) {
                            Some((_, b)) if b.len() >= 4 => f32::from_le_bytes(b[..4].try_into().unwrap()),
                            _ => 1.0,
                        };
                        return Some((out_f, in_f, wbytes, sbytes, macro_s));
                    }
                }
            }
        }
        // Reza custom: `<name>.weight.nvfp4_packed` (U8) + `<name>.weight.nvfp4_scale_e4m3`. No macro.
        let (winfo, wbytes) = self.lookup(&format!("{hf_weight}.nvfp4_packed"))?;
        if winfo.dtype != "U8" || winfo.shape.len() != 2 {
            return None;
        }
        let (_sinfo, sbytes) = self.lookup(&format!("{hf_weight}.nvfp4_scale_e4m3"))?;
        let out_f = winfo.shape[0] as usize;
        let in_f = (winfo.shape[1] as usize) * 2;
        Some((out_f, in_f, wbytes, sbytes, 1.0))
    }
}

impl TensorSource for SafetensorsSource {
    fn config(&self) -> ModelConfig {
        self.cfg.clone()
    }
    fn st_dir(&self) -> Option<&std::path::Path> { Some(&self.dir) }
    /// Presence check without the repack: `find` on a plain NVFP4 weight materializes the whole
    /// repacked buffer just to answer `has` (then `load_opt_from_source` repacks AGAIN to load).
    /// The native lookup is header-only, so answer from it first.
    fn has(&self, ggml_name: &str) -> bool {
        self.find_nvfp4_native(ggml_name).is_some() || self.find(ggml_name).is_some()
    }
    /// A1 direct import: plain (untransformed) modelopt/Reza NVFP4 weights expose their raw file
    /// bytes so the engine repacks modelopt -> split-plane in ONE pass. Transform targets (the
    /// hybrid V-reorders) return None and keep the GGUF-block hop (`kind.apply_nvfp4`).
    fn find_nvfp4_native(&self, ggml_name: &str) -> Option<Nvfp4Native<'_>> {
        use crate::hf_mapping::{HfTarget, resolve_ggml};
        let hf = match resolve_ggml(ggml_name, &self.cfg)? {
            HfTarget::Plain(hf) => hf,
            HfTarget::Transform { .. } => return None,
        };
        let (out_f, in_f, wbytes, wscale, _macro) = self.nvfp4_quant(&hf)?;
        Some(Nvfp4Native { wbytes, wscale, out_f, in_f })
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
            // Only the modelopt encoding carries a per-tensor `weight_scale_2`. Reza has no macro-scale,
            // so its `.scale` lookup returns None and the engine defaults the macro-scale to 1.0.
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
                // NVFP4 (modelopt OR Reza) -> repack to bw24 internal GGUF block_nvfp4 bytes (NO kernel
                // change). `nvfp4_quant` returns the packed bytes directly (in Reza the packed tensor
                // is `<hf>.nvfp4_packed`, not `<hf>` itself), so no second lookup.
                if let Some((out_f, in_f, wbytes, wscale, _macro)) = self.nvfp4_quant(&hf) {
                    let packed = crate::nvfp4_repack::repack_modelopt_to_gguf(wbytes, wscale, out_f, in_f);
                    return Some(TensorView {
                        bytes: Cow::Owned(packed),
                        ggml_type: GgmlType::NVFP4,
                        ne: vec![in_f as u64, out_f as u64], // ggml ne: [in, out]
                    });
                }
                // FP8 E4M3 weight (NVIDIA official 27B linear_attn projections): modelopt
                // per-TENSOR quant — F8 codes + scalar F32 `weight_scale`. Re-encode host-side to
                // GGUF Q8_0 blocks (per-32 fp16 scale + int8): rides the proven q8-fast/MMVQ/fused3
                // path at ~1.06B/elem instead of a 22GB f32 blow-up (OOM, measured). Accuracy: the
                // source is 4-mantissa-bit FP8 with ONE per-tensor scale; per-32 q8 re-quant is a
                // FINER grid — class-equal or better. FP8-native matvec = later perf rung.
                if let Some((info, bytes)) = self.lookup(&hf) {
                    // Large BF16 2D matrices (NVIDIA 27B mtp.* block, ~2GB as f32): re-encode to
                    // Q8_0 like the F8 arm below — the MTP head is a draft (acceptance-gated, never
                    // exactness-bearing), and q8 is the same class the GGUF MTP drafts already use.
                    if info.dtype == "BF16" && info.shape.len() == 2
                        && info.shape.iter().product::<u64>() >= 1_000_000
                        && hf.starts_with("mtp.") {
                        let ne = info.ne();
                        let n = bytes.len() / 2;
                        if n % 32 == 0 {
                            let mut out = Vec::with_capacity(n / 32 * 34);
                            let mut vals = [0f32; 32];
                            for blk in bytes.chunks_exact(64) {
                                let mut amax = 0f32;
                                for i in 0..32 {
                                    let hb = u16::from_le_bytes([blk[2 * i], blk[2 * i + 1]]);
                                    let v = f32::from_bits((hb as u32) << 16);
                                    vals[i] = v;
                                    amax = amax.max(v.abs());
                                }
                                let d = amax / 127.0;
                                let id = if d > 0.0 { 1.0 / d } else { 0.0 };
                                out.extend_from_slice(&crate::nvfp4_repack::f32_to_f16_bits(d).to_le_bytes());
                                for &v in &vals {
                                    out.push((v * id).round_ties_even() as i32 as i8 as u8);
                                }
                            }
                            return Some(TensorView { bytes: Cow::Owned(out), ggml_type: GgmlType::Q8_0, ne });
                        }
                    }
                    if info.dtype == "F8_E4M3" && info.shape.len() == 2 {
                        let stem = hf.strip_suffix(".weight").unwrap_or(&hf);
                        if let Some((sinfo, sbytes)) = self.lookup(&format!("{stem}.weight_scale")) {
                            if sinfo.dtype == "F32" && sbytes.len() >= 4 {
                                let scale = f32::from_le_bytes(sbytes[..4].try_into().unwrap());
                                let ne = info.ne();
                                let n = bytes.len();
                                assert!(n % 32 == 0, "F8 tensor {hf} len {n} not 32-aligned");
                                let mut out = Vec::with_capacity(n / 32 * 34);
                                let mut vals = [0f32; 32];
                                for blk in bytes.chunks_exact(32) {
                                    let mut amax = 0f32;
                                    for (i, &b) in blk.iter().enumerate() {
                                        let v = crate::nvfp4_repack::fp8_e4m3_to_f32(b) * scale;
                                        vals[i] = v;
                                        amax = amax.max(v.abs());
                                    }
                                    let d = amax / 127.0;
                                    let id = if d > 0.0 { 1.0 / d } else { 0.0 };
                                    out.extend_from_slice(&crate::nvfp4_repack::f32_to_f16_bits(d).to_le_bytes());
                                    for &v in &vals {
                                        out.push((v * id).round_ties_even() as i32 as i8 as u8);
                                    }
                                }
                                return Some(TensorView {
                                    bytes: Cow::Owned(out),
                                    ggml_type: GgmlType::Q8_0,
                                    ne,
                                });
                            }
                        }
                    }
                }
                self.raw_hf(&hf)
            }
            // A value transform. Two paths:
            //  (a) NVFP4-preserving: a modelopt NVFP4 weight + a pure structural V-head permutation
            //      (qkv/z/a/b row reorder, out_proj col reorder) -> repack then permute the PACKED
            //      bytes, keeping the weight NVFP4 (no ~8x f32 blow-up; the macro-scale rides the
            //      `<stem>.scale` sibling, applied post-matmul exactly like the Plain NVFP4 arm).
            //  (b) f32 fallback: value transforms (`-exp`, `+1`, conv1d squeeze, identity) operate on
            //      the tiny BF16 SSM tensors; `deq_f32` is NVFP4-aware for any NVFP4 weight here too.
            HfTarget::Transform { hf, kind } => {
                if let Some((out_f, in_f, wbytes, wscale, _macro)) = self.nvfp4_quant(&hf) {
                    let packed = crate::nvfp4_repack::repack_modelopt_to_gguf(wbytes, wscale, out_f, in_f);
                    if let Some((ne, bytes)) = kind.apply_nvfp4(&packed, out_f, in_f, &self.cfg) {
                        return Some(TensorView { bytes: Cow::Owned(bytes), ggml_type: GgmlType::NVFP4, ne });
                    }
                    // fall through to f32 (value transform on an NVFP4 weight — rare/none in qwen35).
                }
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
