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
                    // ... and the token-embedding table: the spec/graph decode paths gather rows
                    // ON DEVICE (embed_gather kernel), which has no BF16 arm — and 2.5GB BF16 vs
                    // 1.35GB Q8_0 matters on the 24GB card. Q8_0 is FINER than the GGUF twin's own
                    // Q5_K embed class. Host-side row gather dequants Q8_0 identically.
                    // ... and lm_head (loadersweep 2026-07-08, LOADER LAW): MiniMax-M3 ships a BF16
                    // lm_head [200064,6144] while every other Linear is NVFP4 — falling through to
                    // the F32 surface made the head a 4.9GB GpuTensor::Float riding cuBLAS f32 GEMV
                    // EVERY decoded token (the 4th occurrence of the Float-poison trap: any F32/BF16
                    // 2D matmul tensor fails uses_q8_1_fast and rides dot_kernel+reduce_1Block
                    // pairs). Q8_0 is FINER than the Q5_K/Q6_K lm_heads every GGUF twin ships.
                    if info.dtype == "BF16" && info.shape.len() == 2
                        && info.shape.iter().product::<u64>() >= 1_000_000
                        && (hf.starts_with("mtp.") || hf == "model.embed_tokens.weight"
                            || hf == "lm_head.weight") {
                        let ne = info.ne();
                        if (bytes.len() / 2) % 32 == 0 {
                            let data: Vec<f32> = bytes.chunks_exact(2)
                                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                                .collect();
                            let out = crate::nvfp4_repack::f32_to_q8_0(&data);
                            return Some(TensorView { bytes: Cow::Owned(out), ggml_type: GgmlType::Q8_0, ne });
                        }
                    }
                    if info.dtype == "F8_E4M3" && info.shape.len() == 2 {
                        let stem = hf.strip_suffix(".weight").unwrap_or(&hf);
                        if let Some((sinfo, sbytes)) = self.lookup(&format!("{stem}.weight_scale")) {
                            if sinfo.dtype == "F32" && sbytes.len() >= 4 {
                                let scale = f32::from_le_bytes(sbytes[..4].try_into().unwrap());
                                let ne = info.ne();
                                assert!(bytes.len() % 32 == 0,
                                        "F8 tensor {hf} len {} not 32-aligned", bytes.len());
                                let data: Vec<f32> = bytes.iter()
                                    .map(|&b| crate::nvfp4_repack::fp8_e4m3_to_f32(b) * scale)
                                    .collect();
                                // BW24_NV_W4=1: F8 attention weights -> NVFP4 (0.56 B/w vs Q8_0's
                                // 1.06) — decode is bandwidth-bound and these layers are 35-40% of
                                // per-token kernel time (nsys 2026-07-07). Real e4m3->e2m1 re-quant;
                                // same 4-bit class the daily GGUF runs on these layers. Opt-in until
                                // the acceptance/text battery proves the class locally.
                                if std::env::var("BW24_NV_W4").map(|v| v == "1").unwrap_or(false)
                                    && ne[0] % 64 == 0 {
                                    let out = crate::nvfp4_repack::f32_to_nvfp4(&data);
                                    return Some(TensorView {
                                        bytes: Cow::Owned(out), ggml_type: GgmlType::NVFP4, ne });
                                }
                                let out = crate::nvfp4_repack::f32_to_q8_0(&data);
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
                // F8-E4M3-sourced LARGE 2D projections (NVIDIA 27B linear_attn in_proj_qkv/z +
                // out_proj, V-reordered above): surfacing F32 is 461MB/linear-layer -> 22GB across
                // the 48 linear layers (the load-tail OOM). Re-encode the post-reorder f32 to GGUF
                // Q8_0 (same class as the Plain-arm F8 re-encode; per-32 fp16 scale is FINER than
                // the source's single per-tensor scale). Small norm-class tensors (ssm_a/dt/
                // conv1d/norms) stay F32 — the engine consumes them via float_data().
                // BF16-sourced in_proj_a/b [48,5120]: below the 1M-element BF16 gate, but leaving
                // them F32 breaks `mixer_in_q8_1_fast` (requires beta+alpha quant) for EVERY
                // linear layer -> unfused norm path + cuBLAS f32 GEMV pairs (nsys: 96 dot+reduce
                // launches/pass + rms_norm_f32 at 12.5us vs 1.8us fused). Q8_0 puts them on the
                // fused2/dual q8-fast chain like the GGUF twin's quantized a/b.
                let is_bf16 = self.lookup(&hf).is_some_and(|(i, _)| i.dtype == "BF16");
                if is_bf16 && ne.len() == 2 && ne[0] % 32 == 0
                    && (hf.ends_with("in_proj_a.weight") || hf.ends_with("in_proj_b.weight")) {
                    let f32s: Vec<f32> = bytes.chunks_exact(4)
                        .map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
                    let out = crate::nvfp4_repack::f32_to_q8_0(&f32s);
                    return Some(TensorView { bytes: Cow::Owned(out), ggml_type: GgmlType::Q8_0, ne });
                }
                let is_f8 = self.lookup(&hf).is_some_and(|(i, _)| i.dtype == "F8_E4M3");
                if is_f8 && ne.len() == 2 && ne[0] % 32 == 0
                    && ne.iter().product::<u64>() >= 1_000_000 {
                    let f32s: Vec<f32> = bytes.chunks_exact(4)
                        .map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
                    // BW24_NV_W4: same F8->NVFP4 re-quant as the Plain arm (see there for the
                    // bandwidth math and class argument) — post-reorder, so the V-permutation is
                    // already baked into the f32 surface.
                    if std::env::var("BW24_NV_W4").map(|v| v == "1").unwrap_or(false)
                        && ne[0] % 64 == 0 {
                        let out = crate::nvfp4_repack::f32_to_nvfp4(&f32s);
                        return Some(TensorView { bytes: Cow::Owned(out), ggml_type: GgmlType::NVFP4, ne });
                    }
                    let out = crate::nvfp4_repack::f32_to_q8_0(&f32s);
                    return Some(TensorView { bytes: Cow::Owned(out), ggml_type: GgmlType::Q8_0, ne });
                }
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

#[cfg(test)]
mod nv27b_probe {
    use super::*;

    /// NVIDIA-official 27B (mixed FP8/NVFP4 ckpt) dtype-routing regression: every BIG tensor class
    /// must surface memory-bounded (Q8_0 / NVFP4), NEVER F32 (461MB/linear-layer x 48 = the 22GB
    /// load-tail OOM, 2026-07-07). Skips when the checkpoint is absent.
    #[test]
    fn nvidia_27b_dtype_routing() {
        let dir = std::path::Path::new("/data/ai-ml/hf-models/nvidia-qwen36-27b-nvfp4");
        if !dir.join("model.safetensors.index.json").exists() { eprintln!("SKIP: ckpt absent"); return; }
        let src = SafetensorsSource::open(dir).unwrap();
        let ty = |n: &str| src.find(n).unwrap_or_else(|| panic!("{n} unresolved")).ggml_type;
        // linear_attn F8 projections (Transform arm, V-reordered) -> Q8_0, never F32.
        assert_eq!(ty("blk.0.attn_qkv.weight"), GgmlType::Q8_0);
        assert_eq!(ty("blk.0.attn_gate.weight"), GgmlType::Q8_0);
        assert_eq!(ty("blk.0.ssm_out.weight"), GgmlType::Q8_0);
        // full-attn F8 projections (Plain arm) -> Q8_0.
        assert_eq!(ty("blk.3.attn_q.weight"), GgmlType::Q8_0);
        assert_eq!(ty("blk.3.attn_output.weight"), GgmlType::Q8_0);
        // NVFP4 mlp + lm_head keep the native direct-import path.
        assert!(src.find_nvfp4_native("blk.0.ffn_gate.weight").is_some());
        assert!(src.find_nvfp4_native("output.weight").is_some());
        // ssm_alpha/beta (<- BF16 in_proj_a/b) are MATMUL-class: the Transform-arm gate re-encodes
        // them Q8_0 so mixer_in_q8_1_fast holds on every linear layer (the loader law; leaving
        // them Float unfused EVERY linear-attn mixer onto cuBLAS f32 GEMV pairs).
        assert_eq!(ty("blk.0.ssm_alpha.weight"), GgmlType::Q8_0);
        assert_eq!(ty("blk.0.ssm_beta.weight"), GgmlType::Q8_0);
        // norm-class SSM tensors stay F32 (engine consumes them via float_data()).
        assert_eq!(ty("blk.0.ssm_conv1d.weight"), GgmlType::F32);
        assert_eq!(ty("blk.0.ssm_a"), GgmlType::F32);
    }

    /// MTP head mapping: blk.64.* (GGUF NextN numbering) resolves into the HF `mtp.*` namespace
    /// with the exact ne the engine loads from the GGUF twin (blk.64 census 2026-07-07).
    /// nextn_predict_layers must come from `mtp_num_hidden_layers` (the 27B HF key).
    #[test]
    fn nvidia_27b_mtp_mapping() {
        let dir = std::path::Path::new("/data/ai-ml/hf-models/nvidia-qwen36-27b-nvfp4");
        if !dir.join("model.safetensors.index.json").exists() { eprintln!("SKIP: ckpt absent"); return; }
        let src = SafetensorsSource::open(dir).unwrap();
        let cfg = src.config();
        assert_eq!(cfg.nextn_predict_layers, 1, "mtp_num_hidden_layers -> nextn");
        assert_eq!(cfg.n_layer, 65, "n_layer includes the MTP block (GGUF block_count convention)");
        let v = |n: &str| src.find(n).unwrap_or_else(|| panic!("{n} unresolved"));
        // glue (GGUF-twin ne reference: enorm/hnorm/shared_head_norm [5120], eh_proj [10240,5120])
        assert_eq!(v("blk.64.nextn.enorm.weight").ne, vec![5120]);
        assert_eq!(v("blk.64.nextn.hnorm.weight").ne, vec![5120]);
        assert_eq!(v("blk.64.nextn.eh_proj.weight").ne, vec![10240, 5120]);
        assert_eq!(v("blk.64.nextn.shared_head_norm.weight").ne, vec![5120]);
        assert!(src.find("blk.64.nextn.shared_head.weight").is_none(), "head reuses lm_head");
        // block tensors (full-attn block: q [5120,12288], k/v [5120,1024], o [6144,5120])
        assert_eq!(v("blk.64.attn_q.weight").ne, vec![5120, 12288]);
        assert_eq!(v("blk.64.attn_k.weight").ne, vec![5120, 1024]);
        assert_eq!(v("blk.64.attn_v.weight").ne, vec![5120, 1024]);
        assert_eq!(v("blk.64.attn_output.weight").ne, vec![6144, 5120]);
        assert_eq!(v("blk.64.attn_q_norm.weight").ne, vec![256]);
        assert_eq!(v("blk.64.ffn_gate.weight").ne, vec![5120, 17408]);
        assert_eq!(v("blk.64.ffn_down.weight").ne, vec![17408, 5120]);
        assert_eq!(v("blk.64.attn_norm.weight").ne, vec![5120]);
        assert_eq!(v("blk.64.post_attention_norm.weight").ne, vec![5120]);
        // big BF16 mtp matrices -> Q8_0 (draft class), norms -> F32 with the +1 fold.
        assert_eq!(v("blk.64.attn_q.weight").ggml_type, GgmlType::Q8_0);
        assert_eq!(v("blk.64.nextn.eh_proj.weight").ggml_type, GgmlType::Q8_0);
        let en = v("blk.64.nextn.enorm.weight");
        assert_eq!(en.ggml_type, GgmlType::F32);
        // +1 fold check vs GGUF twin blk.64.nextn.enorm first value 0.4375 (raw bf16 -0.5625).
        let first = f32::from_le_bytes(en.bytes[..4].try_into().unwrap());
        assert!((first - 0.4375).abs() < 1e-3, "enorm +1 fold: got {first}");
    }
}

#[cfg(test)]
mod m3_probe {
    use super::*;

    /// MiniMax-M3 dtype-routing regression (loadersweep 2026-07-08): the REAP50 ckpt ships lm_head
    /// as the ONLY BF16 Linear (everything else NVFP4). Before the lm_head gate it surfaced F32 ->
    /// a 4.9GB GpuTensor::Float whose per-token decode matmul rode cuBLAS f32 GEMV (the loader-law
    /// Float-poison trap, occurrence #4). Must surface Q8_0. The router (ffn_gate_inp) is the
    /// AUDITED Float exception: selection-sensitive top-k, llama.cpp keeps every router F32, and it
    /// sits on no all-or-nothing predicate — it must STAY F32. Skips when the ckpt is absent.
    #[test]
    fn minimax_m3_lm_head_q8() {
        let dir = std::path::Path::new("/data/ai-ml/hf-models/minimax-m3-nvfp4-reap50");
        if !dir.join("model.safetensors.index.json").exists() { eprintln!("SKIP: ckpt absent"); return; }
        let src = SafetensorsSource::open(dir).unwrap();
        // router: deliberately Float (audited exception — see model.rs float_2d_audited).
        let router = src.find("blk.3.ffn_gate_inp.weight").expect("M3 router unresolved");
        assert_eq!(router.ggml_type, GgmlType::F32);
        assert_eq!(router.ne, vec![6144, 64]);
        // lm_head: BF16 on disk -> must re-encode Q8_0 (matmul-class, hot every decoded token).
        let head = src.find("output.weight").expect("M3 lm_head unresolved");
        assert_eq!(head.ggml_type, GgmlType::Q8_0, "BF16 lm_head must surface Q8_0, not Float");
        assert_eq!(head.ne, vec![6144, 200064]);
    }
}

#[cfg(test)]
mod nv27b_twin_parity {
    use super::*;

    /// Full-tensor numeric parity vs the GGUF twin (converted from the same NVIDIA ckpt):
    /// every F32-surfaced tensor (norms incl +1 folds, ssm_a -exp, dt_bias, conv1d) must match
    /// the twin's dequant bit-for-bit (both go bf16 -> f32 exactly). Skips when either is absent.
    #[test]
    fn nvidia_27b_vs_gguf_twin_f32_parity() {
        let st_dir = std::path::Path::new("/data/ai-ml/hf-models/nvidia-qwen36-27b-nvfp4");
        let twin = std::path::Path::new(
            "/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf");
        if !st_dir.join("model.safetensors.index.json").exists() || !twin.exists() {
            eprintln!("SKIP: ckpt/twin absent"); return;
        }
        let src = SafetensorsSource::open(st_dir).unwrap();
        let g = crate::GgufFile::open(twin.to_str().unwrap()).unwrap();
        // NOTE: trunk pre-FFN norm resolves via the loader's ffn_norm fallback (the twin names it
        // post_attention_norm) — compare ST ffn_norm vs twin post_attention_norm below.
        let names = [
            "blk.0.attn_norm.weight",
            "blk.0.ssm_norm.weight", "blk.0.ssm_a", "blk.0.ssm_dt.bias",
            "blk.0.ssm_conv1d.weight",
            "blk.3.attn_q_norm.weight", "blk.3.attn_k_norm.weight",
            "blk.64.attn_norm.weight", "blk.64.nextn.enorm.weight",
            "blk.64.nextn.hnorm.weight", "blk.64.nextn.shared_head_norm.weight",
            "output_norm.weight",
        ];
        // (ST ggml name, twin ggml name) pairs where the two sources use different aliases.
        let pairs = [("blk.0.ffn_norm.weight", "blk.0.post_attention_norm.weight")];
        for (st_name, twin_name) in names.iter().map(|&n| (n, n)).chain(pairs) {
            let name = st_name;
            let sv = src.find(st_name).unwrap_or_else(|| panic!("{st_name}: ST unresolved"));
            let gt = g.find(twin_name).unwrap_or_else(|| panic!("{twin_name}: not in twin"));
            assert_eq!(sv.ne, gt.ne, "{name}: ne mismatch");
            let n: u64 = sv.ne.iter().product();
            let a = crate::dequant::dequantize(sv.ggml_type, &sv.bytes, n as usize);
            let b = crate::dequant::dequantize(gt.ggml_type, g.tensor_data(gt), n as usize);
            let md = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
            // ssm_a = -exp(A_log): Rust libm expf vs the converter's numpy exp differ by ULPs.
            // Everything else (renames, +1 folds, conv1d squeeze/reorder) must be bit-exact.
            let tol = if name.ends_with("ssm_a") { 1e-7 } else { 0.0 };
            assert!(md <= tol, "{name}: maxdiff {md} > tol {tol}");
            eprintln!("{name:40} n={n:6} maxdiff={md:.1e}");
        }
    }
}
