//! Source-agnostic weight access: a `TensorSource` trait the engine loads from, implemented by
//! BOTH the GGUF reader and the safetensors reader. The engine only ever asks for ggml-style
//! names + gets back `{ggml_type, ne, &[u8]}`; the source hides where the bytes come from.
//!
//! This trait lives in bw24-gguf (not bw24-engine) because it returns bw24-gguf types
//! (`GgmlType`, `ModelConfig`) and both readers live here. bw24-engine already depends on
//! bw24-gguf, so `GpuTensor::load_from_source(&dyn TensorSource, ...)` introduces no new dep.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use memmap2::Mmap;
use crate::config::{Arch, JsonObj, ModelConfig};
use crate::safetensors::StModel;
use crate::{GgufFile, GgmlType};

/// Whole-map access advice for expert slab files. `Random` preserves the original spill behavior;
/// `Normal` lets Linux use its ordinary mmap readahead for each multi-megabyte expert access.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertMmapAdvice {
    Random,
    Normal,
}

/// Pure parser kept separate from the cached environment lookup so invalid/default behavior is
/// unit-testable without mutating process-global environment state.
pub fn parse_expert_mmap_advice(value: Option<&str>) -> Result<ExpertMmapAdvice, &'static str> {
    match value.unwrap_or("random") {
        "random" => Ok(ExpertMmapAdvice::Random),
        "normal" => Ok(ExpertMmapAdvice::Normal),
        _ => Err("expected random or normal"),
    }
}

pub fn expert_mmap_advice() -> ExpertMmapAdvice {
    static MODE: std::sync::OnceLock<ExpertMmapAdvice> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        let raw = std::env::var("BW24_MOE_MMAP_ADVICE").ok();
        match parse_expert_mmap_advice(raw.as_deref()) {
            Ok(mode) => mode,
            Err(reason) => {
                eprintln!(
                    "[spill] invalid BW24_MOE_MMAP_ADVICE={:?} ({reason}); using random",
                    raw.as_deref().unwrap_or("")
                );
                ExpertMmapAdvice::Random
            }
        }
    })
}

/// Apply the selected policy to one expert mmap. `MADV_NORMAL` explicitly clears a prior
/// `MADV_RANDOM` VMA policy, so this is safe for maps reused across loader paths.
pub fn apply_expert_mmap_advice(map: &Mmap) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        let advice = match expert_mmap_advice() {
            ExpertMmapAdvice::Random => memmap2::Advice::Random,
            ExpertMmapAdvice::Normal => memmap2::Advice::Normal,
        };
        map.advise(advice)
    }
    #[cfg(not(unix))]
    {
        let _ = map;
        Ok(())
    }
}

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

/// Raw FP8-E4M3-native weight view (BW24_PP_FP8 prefill operand): the checkpoint's e4m3 codes
/// `[out_f, in_f]` row-major + the per-tensor f32 `weight_scale`. For a PLAIN (untransformed)
/// F8 Linear this borrows the mmap verbatim (EXACT bytes); for the hybrid V-reordered F8
/// projections it is an OWNED buffer produced by the f32 transform round-trip — exact too, since
/// the V-reorder is a pure permutation and every f32 value is `e4m3_code * scale`, which the
/// nearest-e4m3 re-encode of `value/scale` recovers bit-for-bit (grid spacing >> f32 rounding).
pub struct Fp8Native<'a> {
    pub bytes: Cow<'a, [u8]>, // e4m3 codes, [out_f, in_f] row-major
    pub scale: f32,           // per-tensor weight_scale (dequant multiplier)
    pub out_f: usize,
    pub in_f: usize,
}

/// One immutable expert extent backed by an opened file and its whole-file mmap. Retaining both
/// handles lets the engine choose explicit positioned I/O later while keeping the mmap bytes as the
/// permanent correctness fallback. `offset` is absolute within both the file and the whole mapping.
#[derive(Clone)]
pub struct DiskExtent {
    pub map: Arc<Mmap>,
    pub file: Arc<File>,
    pub offset: u64,
    pub len: usize,
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
    /// FP8-E4M3-native access (BW24_PP_FP8 prefill operand): the raw e4m3 codes + per-tensor f32
    /// `weight_scale` for an F8-sourced 2D projection, so the engine can stash the exact weight
    /// bytes for the cuBLASLt FP8 prefill GEMM alongside its Q8_0 re-encode. None for GGUF
    /// sources and for anything that is not an F8-E4M3 2D Linear.
    fn find_fp8_native(&self, _ggml_name: &str) -> Option<Fp8Native<'_>> { None }
    /// The checkpoint directory, if this source is a safetensors HF dir (None for GGUF). Used by
    /// the ST expert disk-tier to place its repack cache next to the shards.
    fn st_dir(&self) -> Option<&std::path::Path> { None }
    /// Whether per-expert tensor encodings exposed by this source are an intentional artifact
    /// contract and must be retained even when every expert happens to use the same encoding.
    /// Sparse overlays use this so an all-Q4_K control does not get normalized back to F32.
    fn preserve_expert_encodings(&self) -> bool { false }
    /// Optional per-layer routed-expert mask. A false entry is physically absent from a pruned
    /// overlay and must be excluded before top-k routing. Keeping the original router width makes
    /// usage-driven pruning possible without rewriting router tensors or expert ids.
    fn active_experts(&self, _layer: u32) -> Option<&[bool]> { None }
    /// Zero-copy mmap window for an expert tensor (the disk-spill tier's byte source):
    /// `(shared file mmap, tensor byte offset, tensor bytes)`. This covers both stacked 3D expert
    /// slabs and v2 per-expert overlay entries. The engine then backs `HostExps` with
    /// `HostBuf::Mmap` directly (page cache = RAM tier, faults = NVMe tier) instead of copying a
    /// potentially >RAM expert set. None for sources that require gathering or repacking.
    fn find_expert_disk(&self, _ggml_name: &str) -> Option<DiskExtent> { None }
    /// Compatibility view for callers that only need mmap access. New disk-aware consumers should
    /// use `find_expert_disk` so the opened file remains available after the source is dropped.
    fn find_expert_mmap(&self, ggml_name: &str) -> Option<(Arc<Mmap>, usize, usize)> {
        let extent = self.find_expert_disk(ggml_name)?;
        let offset = usize::try_from(extent.offset).ok()?;
        Some((extent.map, offset, extent.len))
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
            bytes: Cow::Borrowed(self.0.tensor_data(t)),
            ggml_type: t.ggml_type,
            ne: t.ne.clone(),
        })
    }
    fn gguf(&self) -> Option<&GgufFile> { Some(self.0) }
}

#[derive(Debug, Clone)]
struct RepackTensor {
    file: PathBuf,
    offset: usize,
    ggml_type: GgmlType,
    ne: Vec<u64>,
    bytes: usize,
    expert_stride: Option<usize>,
}

struct RepackFile {
    // Only expert files retain an fd. Dense-only manifest files keep their mmap but do not consume
    // the process fd budget because positioned I/O is never requested for them.
    file: Option<Arc<File>>,
    map: Arc<Mmap>,
}

enum RepackFallback {
    Safetensors(SafetensorsSource),
    Repack(Box<Hy3RepackSource>),
}

impl RepackFallback {
    fn config(&self) -> ModelConfig {
        match self {
            Self::Safetensors(source) => source.config(),
            Self::Repack(source) => source.config(),
        }
    }

    fn find(&self, name: &str) -> Option<TensorView<'_>> {
        match self {
            Self::Safetensors(source) => source.find(name),
            Self::Repack(source) => source.find(name),
        }
    }

    fn find_nvfp4_native(&self, name: &str) -> Option<Nvfp4Native<'_>> {
        match self {
            Self::Safetensors(source) => source.find_nvfp4_native(name),
            Self::Repack(source) => source.find_nvfp4_native(name),
        }
    }

    fn find_fp8_native(&self, name: &str) -> Option<Fp8Native<'_>> {
        match self {
            Self::Safetensors(source) => source.find_fp8_native(name),
            Self::Repack(source) => source.find_fp8_native(name),
        }
    }

    fn st_dir(&self) -> Option<&Path> {
        match self {
            Self::Safetensors(source) => source.st_dir(),
            Self::Repack(source) => source.st_dir(),
        }
    }
}

/// Manifest-backed source for bw24 repack directories and sparse per-expert overlays.
///
/// The transcoder writes one file per tensor (including stacked expert slabs) plus a manifest with
/// ggml-style names. This source presents those bytes directly to the existing loaders without a
/// single-file GGUF wrapper. Expert overlays may fall back to either an HF checkpoint or another
/// manifest-backed repack; the latter lets a multi-tier expert artifact reuse the established Hy3
/// dense/router repack without copying it. Files are mmap'd lazily by the OS; opening the 80G
/// repack maps address space but does not fault tensor pages into RAM. The public type retains its
/// historical Hy3 name for compatibility with existing callers.
pub struct Hy3RepackSource {
    cfg: ModelConfig,
    dir: PathBuf,
    source_dir: Option<PathBuf>,
    tensors: BTreeMap<String, RepackTensor>,
    // Retain both handles so stacked expert slabs can be handed to the engine's disk-aware mmap tier
    // while `find` keeps borrowing the same mapping.
    files: BTreeMap<PathBuf, RepackFile>,
    // Expert overlays store only overridden tensors. Everything else resolves from either the
    // original HF checkpoint (v1) or a complete manifest repack (v2).
    fallback: Option<RepackFallback>,
    active_experts: BTreeMap<u32, Vec<bool>>,
}

impl Hy3RepackSource {
    pub fn open(path: &Path) -> std::io::Result<Self> {
        let manifest = if path.is_dir() { path.join("manifest.json") } else { path.to_path_buf() };
        let dir = manifest.parent().unwrap_or_else(|| Path::new(".")).to_path_buf();
        let txt = std::fs::read_to_string(&manifest)?;
        let top = JsonObj::parse(&txt);
        let tensors_obj = top.object("tensors").ok_or_else(|| invalid_data("manifest missing tensors object"))?;
        let mut tensors = BTreeMap::new();
        for (name, raw) in tensors_obj.fields() {
            let obj = JsonObj::parse(raw);
            let file = obj.string("file")
                .ok_or_else(|| invalid_data(format!("manifest tensor {name} missing file")))?;
            let qtype = obj.string("qtype")
                .ok_or_else(|| invalid_data(format!("manifest tensor {name} missing qtype")))?;
            let ne = obj.u64_array("ne")
                .ok_or_else(|| invalid_data(format!("manifest tensor {name} missing ne")))?;
            let bytes = obj.u64("bytes")
                .ok_or_else(|| invalid_data(format!("manifest tensor {name} missing bytes")))? as usize;
            tensors.insert(name.to_string(), RepackTensor {
                file: PathBuf::from(file),
                offset: obj.u64("offset").unwrap_or(0) as usize,
                ggml_type: manifest_qtype(&qtype)
                    .ok_or_else(|| invalid_data(format!("manifest tensor {name} unsupported qtype {qtype}")))?,
                ne,
                bytes,
                expert_stride: obj.u64("expert_stride").map(|x| x as usize),
            });
        }

        let source_dir = top.string("source_dir").map(PathBuf::from).map(|path| {
            if path.is_absolute() { path } else { dir.join(path) }
        });
        let format = top.string("format");
        let is_overlay = matches!(
            format.as_deref(),
            Some("bw24-expert-overlay-v1" | "bw24-expert-overlay-v2")
        );
        let fallback = if is_overlay {
            let source = source_dir.as_deref().ok_or_else(|| {
                invalid_data("expert overlay manifest missing source_dir")
            })?;
            if source.join("manifest.json").exists() {
                Some(RepackFallback::Repack(Box::new(Hy3RepackSource::open(source)?)))
            } else {
                Some(RepackFallback::Safetensors(SafetensorsSource::open(source)?))
            }
        } else {
            None
        };
        let mut cfg = if let Some(source) = &fallback {
            source.config()
        } else {
            let cfg_path = source_dir.clone()
                .map(|p| p.join("config.json"))
                .filter(|p| p.exists())
                .unwrap_or_else(|| dir.join("config.json"));
            ModelConfig::from_config_json(&cfg_path)?
        };
        apply_stripped_mtp_override(&mut cfg, &tensors);
        let mut active_experts = BTreeMap::new();
        if let Some(pruned) = top.object("pruned_experts") {
            let moe = cfg.moe.as_ref().ok_or_else(|| {
                invalid_data("pruned_experts is present but the model config has no MoE")
            })?;
            let n_expert = moe.expert_count as usize;
            let n_used = moe.expert_used_count as usize;
            for (layer, raw) in pruned.fields() {
                let layer: u32 = layer.parse().map_err(|_| {
                    invalid_data(format!("invalid pruned_experts layer key {layer:?}"))
                })?;
                let wrapper = JsonObj::parse(&format!("{{\"v\":{raw}}}"));
                let ids = wrapper.u64_array("v").ok_or_else(|| {
                    invalid_data(format!("pruned_experts.{layer} must be an integer array"))
                })?;
                let mut mask = vec![true; n_expert];
                for id in ids {
                    let id = id as usize;
                    if id >= n_expert {
                        return Err(invalid_data(format!(
                            "pruned_experts.{layer} contains {id}, expert_count={n_expert}"
                        )));
                    }
                    mask[id] = false;
                }
                if mask.iter().filter(|&&active| active).count() < n_used {
                    return Err(invalid_data(format!(
                        "pruned_experts.{layer} leaves fewer than top-k {n_used} experts"
                    )));
                }
                active_experts.insert(layer, mask);
            }
        }

        let expert_files: BTreeSet<PathBuf> = tensors.iter()
            .filter(|(name, _)| name.contains("_exps."))
            .map(|(_, tensor)| tensor.file.clone())
            .collect();
        let mut files = BTreeMap::new();
        let mut seen = BTreeSet::new();
        for t in tensors.values() {
            if !seen.insert(t.file.clone()) { continue; }
            let p = dir.join(&t.file);
            let file = Arc::new(File::open(&p)?);
            let map = unsafe { Mmap::map(file.as_ref())? };
            let retain_file = t.expert_stride.is_some() || expert_files.contains(&t.file);
            // Expert slabs use the configured whole-map policy. Default random is the historical
            // behavior; normal restores Linux readahead for multi-megabyte expert reads.
            if retain_file {
                let _ = apply_expert_mmap_advice(&map);
            }
            files.insert(t.file.clone(), RepackFile {
                file: retain_file.then_some(file),
                map: Arc::new(map),
            });
        }
        for (name, t) in &tensors {
            let len = files.get(&t.file)
                .ok_or_else(|| invalid_data(format!("manifest tensor {name} file not mapped")))?
                .map.len();
            if len < t.offset + t.bytes {
                return Err(invalid_data(format!(
                    "manifest tensor {name} declares offset {} + {} bytes but {:?} has {len}",
                    t.offset, t.bytes, t.file
                )));
            }
        }

        Ok(Self { cfg, dir, source_dir, tensors, files, fallback, active_experts })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The original HF checkpoint dir recorded by the transcoder (tokenizer files live there —
    /// the repack dir carries only weights + manifest).
    pub fn source_dir(&self) -> Option<&Path> {
        match self.fallback.as_ref() {
            Some(RepackFallback::Safetensors(source)) => source.st_dir(),
            Some(RepackFallback::Repack(source)) => source.source_dir(),
            None => self.source_dir.as_deref(),
        }
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn expert_stride(&self, ggml_name: &str) -> Option<usize> {
        self.tensors.get(ggml_name).and_then(|t| t.expert_stride)
    }
}

impl TensorSource for Hy3RepackSource {
    fn config(&self) -> ModelConfig {
        self.cfg.clone()
    }

    fn find(&self, ggml_name: &str) -> Option<TensorView<'_>> {
        if let Some(t) = self.tensors.get(ggml_name) {
            let file = self.files.get(&t.file)?;
            let raw = &file.map[t.offset..t.offset + t.bytes];
            if t.ggml_type == GgmlType::BF16 && t.ne.len() == 1 {
                let n: u64 = t.ne.iter().product();
                let vals = crate::dequant::dequantize(t.ggml_type, raw, n as usize);
                let mut bytes = Vec::with_capacity(vals.len() * 4);
                for f in vals {
                    bytes.extend_from_slice(&f.to_le_bytes());
                }
                return Some(TensorView {
                    bytes: Cow::Owned(bytes),
                    ggml_type: GgmlType::F32,
                    ne: t.ne.clone(),
                });
            }
            return Some(TensorView {
                bytes: Cow::Borrowed(raw),
                ggml_type: t.ggml_type,
                ne: t.ne.clone(),
            });
        }
        // A v2 overlay stores mixed experts separately while its fallback may contain the old
        // uniform stacked slab. Do not let that slab bypass the per-expert override loader.
        if ggml_name.contains("_exps.weight") {
            let prefix = format!("{}.", ggml_name.strip_suffix(".weight")?);
            if self.tensors.range(prefix.clone()..).next()
                .is_some_and(|(name, _)| name.starts_with(&prefix))
            {
                return None;
            }
        }
        self.fallback.as_ref()?.find(ggml_name)
    }

    fn find_nvfp4_native(&self, ggml_name: &str) -> Option<Nvfp4Native<'_>> {
        if self.tensors.contains_key(ggml_name) { return None; }
        self.fallback.as_ref()?.find_nvfp4_native(ggml_name)
    }

    fn find_fp8_native(&self, ggml_name: &str) -> Option<Fp8Native<'_>> {
        if self.tensors.contains_key(ggml_name) { return None; }
        self.fallback.as_ref()?.find_fp8_native(ggml_name)
    }

    fn st_dir(&self) -> Option<&Path> {
        self.fallback.as_ref().and_then(RepackFallback::st_dir)
    }

    fn preserve_expert_encodings(&self) -> bool {
        self.fallback.is_some()
    }

    fn active_experts(&self, layer: u32) -> Option<&[bool]> {
        self.active_experts.get(&layer).map(Vec::as_slice)
    }

    /// Hand stacked expert slabs and v2 per-expert overlay entries to the engine as shared mmap
    /// windows. Both layouts are already kernel-ready; copying them into Vec would make the 161 GB
    /// full-bank control impossible on a 124 GB host.
    fn find_expert_disk(&self, ggml_name: &str) -> Option<DiskExtent> {
        let t = self.tensors.get(ggml_name)?;
        let file = self.files.get(&t.file)?;
        Some(DiskExtent {
            map: file.map.clone(),
            file: file.file.as_ref()?.clone(),
            offset: t.offset as u64,
            len: t.bytes,
        })
    }
}

fn manifest_qtype(s: &str) -> Option<GgmlType> {
    Some(match s {
        "F32" => GgmlType::F32,
        "F16" => GgmlType::F16,
        "BF16" => GgmlType::BF16,
        "Q8_0" => GgmlType::Q8_0,
        "Q2_K" => GgmlType::Q2_K,
        "Q4_K" => GgmlType::Q4_K,
        "Q5_K" => GgmlType::Q5_K,
        "Q6_K" => GgmlType::Q6_K,
        "Q3_K" => GgmlType::Q3_K,
        "IQ4_XS" => GgmlType::IQ4_XS,
        "IQ3_S" => GgmlType::IQ3_S,
        "NVFP4" => GgmlType::NVFP4,
        _ => return None,
    })
}

fn apply_stripped_mtp_override(cfg: &mut ModelConfig, tensors: &BTreeMap<String, RepackTensor>) {
    if cfg.nextn_predict_layers == 0 {
        return;
    }
    let max_blk = tensors.keys().filter_map(|name| {
        let rest = name.strip_prefix("blk.")?;
        let (il, _) = rest.split_once('.')?;
        il.parse::<u32>().ok()
    }).max();
    let Some(max_blk) = max_blk else { return; };
    let manifest_layers = max_blk + 1;
    let trunk_layers = cfg.n_layer.saturating_sub(cfg.nextn_predict_layers);
    let has_nextn_names = tensors.keys().any(|name| name.contains(".nextn."));
    if manifest_layers <= trunk_layers && !has_nextn_names {
        cfg.n_layer = manifest_layers;
        cfg.n_layer_total = manifest_layers;
        cfg.nextn_predict_layers = 0;
    }
}

fn invalid_data(msg: impl Into<String>) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, msg.into())
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

    /// Detect an HF NVFP4 quantized Linear under ANY on-disk encoding and return everything the
    /// repack needs: `(out_f, in_f, packed_bytes, per16_fp8_scale_bytes, macro_scale)`. All encodings
    /// store the SAME e2m1 weights + per-16 FP8(e4m3) scales — only names + macro-scale differ:
    ///   * modelopt: `<name>.weight`(U8 packed) + `<name>.weight_scale`(F8_E4M3) +
    ///     `<name>.weight_scale_2`(F32 per-tensor macro, default 1.0).
    ///   * compressed-tensors (llm-compressor): `<name>.weight_packed`(U8) +
    ///     `<name>.weight_scale`(F8_E4M3 per-16) + `<name>.weight_global_scale`(F32 per-tensor macro).
    ///     The plain `<name>.weight` coexists as a BF16 tensor (unused by us when packed is present).
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
        // compressed-tensors (llm-compressor): `<name>.weight_packed` (U8) + `<name>.weight_scale`
        // (F8_E4M3 per-16) + `<name>.weight_global_scale` (F32 per-tensor). The plain
        // `<name>.weight` (BF16) coexists but is UNUSED when the packed sibling is present —
        // the packed representation IS the quantized model output. CRITICAL SEMANTICS DIFFERENCE:
        // compressed-tensors' `weight_global_scale` is a DIVISOR (dequant = code * micro / global),
        // whereas modelopt's `weight_scale_2` is a MULTIPLIER (dequant = code * micro * scale_2).
        // The packed bytes + micro scales are byte-identical between the two formats (verified on
        // the AxionML vs apolo13x pair), only the macro-scale semantics differ. We invert here so
        // the engine's post-matmul multiply stays unchanged.
        //   compressed-tensors: elem = e2m1_code * ue4m3_scale_per16 / weight_global_scale
        //   modelopt:           elem = e2m1_code * ue4m3_scale_per16 * weight_scale_2
        //   => macro_s = 1.0 / weight_global_scale
        let stem = hf_weight.strip_suffix(".weight")?;
        if let Some((winfo, wbytes)) = self.lookup(&format!("{stem}.weight_packed")) {
            if winfo.dtype == "U8" && winfo.shape.len() == 2 {
                if let Some((sinfo, sbytes)) = self.lookup(&format!("{stem}.weight_scale")) {
                    if sinfo.dtype == "F8_E4M3" {
                        let out_f = winfo.shape[0] as usize;
                        let in_f = (winfo.shape[1] as usize) * 2;
                        let macro_s = match self.lookup(&format!("{stem}.weight_global_scale")) {
                            Some((_, b)) if b.len() >= 4 => {
                                let gs = f32::from_le_bytes(b[..4].try_into().unwrap());
                                if gs > 0.0 && gs.is_finite() { 1.0 / gs } else { 1.0 }
                            }
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
    /// FP8-E4M3-native access (BW24_PP_FP8 prefill operand). Two arms, mirroring the Q8_0
    /// re-encode arms in `find` (the engine only stashes this when `find` surfaced Q8_0):
    ///  * Plain: borrow the checkpoint's e4m3 bytes verbatim (zero copy, EXACT).
    ///  * Transform (hybrid V-reorders): run the SAME `deq_f32` + `kind.apply` the Q8_0 arm runs,
    ///    then re-encode `value/scale` to nearest e4m3 — exact for a pure permutation (every value
    ///    is `code*scale`; the e4m3 grid spacing dwarfs the f32 divide rounding).
    /// Dim gates: 2D, in_f/out_f % 16 == 0 (cuBLASLt FP8 TN alignment), and the Transform arm
    /// keeps the >=1M-element gate of its Q8_0 twin (small tensors stay F32 there).
    fn find_fp8_native(&self, ggml_name: &str) -> Option<Fp8Native<'_>> {
        use crate::hf_mapping::{HfTarget, resolve_ggml};
        let (hf, kind) = match resolve_ggml(ggml_name, &self.cfg)? {
            HfTarget::Plain(hf) => (hf, None),
            HfTarget::Transform { hf, kind } => (hf, Some(kind)),
        };
        let (info, bytes) = self.lookup(&hf)?;
        if info.dtype != "F8_E4M3" || info.shape.len() != 2 { return None; }
        let stem = hf.strip_suffix(".weight").unwrap_or(&hf);
        let (sinfo, sbytes) = self.lookup(&format!("{stem}.weight_scale"))?;
        if sinfo.dtype != "F32" || sbytes.len() < 4 { return None; }
        let scale = f32::from_le_bytes(sbytes[..4].try_into().unwrap());
        if !(scale > 0.0) || !scale.is_finite() { return None; }
        match kind {
            None => {
                let ne = info.ne();
                let (in_f, out_f) = (ne[0] as usize, ne[1] as usize);
                if in_f % 16 != 0 || out_f % 16 != 0 { return None; }
                Some(Fp8Native { bytes: Cow::Borrowed(bytes), scale, out_f, in_f })
            }
            Some(kind) => {
                let (mut data, ne_in) = self.deq_f32(&hf)?;
                let (ne, fbytes) = kind.apply(&mut data, ne_in, &self.cfg);
                if ne.len() != 2 || ne.iter().product::<u64>() < 1_000_000 { return None; }
                let (in_f, out_f) = (ne[0] as usize, ne[1] as usize);
                if in_f % 16 != 0 || out_f % 16 != 0 { return None; }
                let enc: Vec<u8> = fbytes.chunks_exact(4)
                    .map(|c| crate::nvfp4_repack::f32_to_fp8_e4m3(
                        f32::from_le_bytes(c.try_into().unwrap()) / scale))
                    .collect();
                Some(Fp8Native { bytes: Cow::Owned(enc), scale, out_f, in_f })
            }
        }
    }
    fn find(&self, ggml_name: &str) -> Option<TensorView<'_>> {
        use crate::hf_mapping::{HfTarget, resolve_ggml};
        // NVFP4 per-tensor macro-scale sibling: the engine asks for `<stem>.scale` (model.rs) and
        // expects an F32 scalar. Map `<stem>.scale` -> modelopt `<hf>.weight_scale_2` OR
        // compressed-tensors `<hf>.weight_global_scale`. Returns None for non-quantized weights
        // (then the engine defaults the macro-scale to 1.0). Reza has no macro-scale at all.
        if let Some(stem) = ggml_name.strip_suffix(".scale") {
            let hf_weight = match resolve_ggml(&format!("{stem}.weight"), &self.cfg)? {
                HfTarget::Plain(hf) | HfTarget::Transform { hf, .. } => hf,
            };
            let hf_stem = hf_weight.strip_suffix(".weight")?;
            // Try modelopt `weight_scale_2` first (direct multiplier, borrow zero-copy).
            if let Some((info, bytes)) = self.lookup(&format!("{hf_stem}.weight_scale_2")) {
                return Some(TensorView {
                    bytes: Cow::Borrowed(bytes),
                    ggml_type: info.ggml_type(),
                    ne: vec![1],
                });
            }
            // compressed-tensors `weight_global_scale`: DIVISOR semantics, must INVERT to match
            // the engine's multiplier convention (engine does: result *= macro_scale).
            if let Some((_, bytes)) = self.lookup(&format!("{hf_stem}.weight_global_scale")) {
                if bytes.len() >= 4 {
                    let gs = f32::from_le_bytes(bytes[..4].try_into().unwrap());
                    let inv = if gs > 0.0 && gs.is_finite() { 1.0 / gs } else { 1.0 };
                    return Some(TensorView {
                        bytes: Cow::Owned(inv.to_le_bytes().to_vec()),
                        ggml_type: GgmlType::F32,
                        ne: vec![1],
                    });
                }
            }
            return None;
        }
        match resolve_ggml(ggml_name, &self.cfg)? {
            // Zero-copy: a plain rename (dense path + most SSM matrices), borrow the mmap directly.
            // NVFP4 modelopt weights take the repack arm (owned GGUF block bytes); else borrow.
            HfTarget::Plain(mut hf) => {
                // Hy3 changed the correction-bias key between the preview and current releases.
                // Prefer the current mapper spelling, but keep old repacks/checkpoints loadable.
                if self.cfg.arch.is_hy3() && ggml_name.ends_with(".exp_probs_b.bias")
                    && self.lookup(&hf).is_none() {
                    let legacy = hf.replace(".mlp.expert_bias", ".mlp.router.expert_bias");
                    if self.lookup(&legacy).is_some() {
                        hf = legacy;
                    }
                }
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
                    // Large BF16 2D matrices -> Q8_0 re-encode (LOADER LAW, 2026-07-08):
                    // ANY BF16 2D weight >= 1M elements that reaches the engine as Float/FloatBf16
                    // fails `uses_q8_1_fast` and rides the slow dot_kernel+reduce_1Block cuBLAS f32
                    // GEMV pairs (the "Float-poison" trap, occurrences 1-5). Q8_0 per-32 fp16 scale
                    // + int8 is a FINER grid than BF16 (7-bit mantissa int8*scale vs bf16's 7-bit
                    // significand) — same class, strictly no worse accuracy — and puts every tensor
                    // on the proven q8-fast/MMVQ/fused3 path. Covers: mtp.* (draft, same class as
                    // the GGUF Q8_0 twin), lm_head, embed_tokens, AND any unquantized attention
                    // projections in checkpoints where they remain BF16 (compressed-tensors/apolo:
                    // linear_attn.in_proj_qkv/z/out_proj + self_attn.q/k/v/o_proj, per recipe.yaml
                    // ignore list). BW24_FULL_PREC=1 bypasses this to surface raw BF16 (the engine
                    // keeps them FloatBf16-resident for the MTP-heal protocol).
                    let full_prec = std::env::var("BW24_FULL_PREC").as_deref() == Ok("1");
                    if !full_prec && info.dtype == "BF16" && info.shape.len() == 2
                        && info.shape.iter().product::<u64>() >= 1_000_000 {
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
                // BF16-sourced LARGE 2D projections (compressed-tensors: in_proj_qkv/z/out_proj +
                // self_attn q/k/v/o_proj ALL BF16): same Float-poison loader law as the Plain arm.
                // Q8_0 per-32 is FINER than BF16's 7-bit significand. The in_proj_a/b gate below
                // catches the small-but-matmul-class a/b (below the 1M-element gate but still must
                // ride q8-fast for mixer_in_q8_1_fast). BW24_FULL_PREC=1 bypasses both.
                let full_prec = std::env::var("BW24_FULL_PREC").as_deref() == Ok("1");
                let is_bf16 = self.lookup(&hf).is_some_and(|(i, _)| i.dtype == "BF16");
                if !full_prec && is_bf16 && ne.len() == 2 && ne[0] % 32 == 0
                    && (ne.iter().product::<u64>() >= 1_000_000
                        || hf.ends_with("in_proj_a.weight") || hf.ends_with("in_proj_b.weight")) {
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
    fn expert_mmap_advice_parser_preserves_random_default() {
        assert_eq!(parse_expert_mmap_advice(None), Ok(ExpertMmapAdvice::Random));
        assert_eq!(parse_expert_mmap_advice(Some("random")), Ok(ExpertMmapAdvice::Random));
        assert_eq!(parse_expert_mmap_advice(Some("normal")), Ok(ExpertMmapAdvice::Normal));
        assert!(parse_expert_mmap_advice(Some("sequential")).is_err());
        assert!(parse_expert_mmap_advice(Some("")).is_err());
    }

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

    #[test]
    fn hy3_expert_bias_resolves_current_and_preview_keys() {
        for (tag, hf_name) in [
            ("current", "model.layers.1.mlp.expert_bias"),
            ("preview", "model.layers.1.mlp.router.expert_bias"),
        ] {
            let dir = std::env::temp_dir().join(format!(
                "bw24_hy3_bias_{tag}_{}", std::process::id()
            ));
            std::fs::create_dir_all(&dir).unwrap();

            let header = format!(
                r#"{{"{hf_name}":{{"dtype":"F32","shape":[3],"data_offsets":[0,12]}}}}"#
            );
            let mut buf = Vec::new();
            buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
            buf.extend_from_slice(header.as_bytes());
            for value in [0.25f32, -0.5, 0.75] {
                buf.extend_from_slice(&value.to_le_bytes());
            }
            std::fs::write(dir.join("model.safetensors"), buf).unwrap();
            std::fs::write(
                dir.join("config.json"),
                r#"{"model_type":"hy_v3","num_hidden_layers":2,"hidden_size":4,"num_attention_heads":1,"num_key_value_heads":1,"head_dim":4,"intermediate_size":8,"vocab_size":10,"max_position_embeddings":128,"num_experts":3,"num_experts_per_tok":1,"moe_intermediate_size":4,"first_k_dense_replace":1,"moe_router_use_sigmoid":true,"moe_router_enable_expert_bias":true}"#,
            ).unwrap();

            let src = SafetensorsSource::open(&dir).unwrap();
            let bias = src.find("blk.1.exp_probs_b.bias")
                .unwrap_or_else(|| panic!("Hy3 {tag} expert-bias key did not resolve"));
            assert_eq!(bias.ggml_type, GgmlType::F32);
            assert_eq!(bias.ne, vec![3]);
            assert_eq!(bias.bytes.len(), 12);

            std::fs::remove_dir_all(&dir).ok();
        }
    }

    #[test]
    fn expert_overlay_overrides_selected_tensor_and_falls_back_to_hf() {
        let root = std::env::temp_dir().join(format!("bw24_overlay_test_{}", std::process::id()));
        let base = root.join("base");
        let overlay = root.join("overlay");
        std::fs::create_dir_all(&base).unwrap();
        std::fs::create_dir_all(overlay.join("experts")).unwrap();

        let json = r#"{"model.layers.0.self_attn.q_proj.weight":{"dtype":"F32","shape":[4,2],"data_offsets":[0,32]}}"#;
        let mut buf = Vec::new();
        buf.extend_from_slice(&(json.len() as u64).to_le_bytes());
        buf.extend_from_slice(json.as_bytes());
        for v in 0..8u32 { buf.extend_from_slice(&(v as f32).to_le_bytes()); }
        std::fs::write(base.join("model.safetensors"), &buf).unwrap();
        let cfg_json = r#"{"model_type":"qwen3","num_hidden_layers":1,"hidden_size":4,"num_attention_heads":2,"intermediate_size":8,"vocab_size":10,"max_position_embeddings":128}"#;
        std::fs::write(base.join("config.json"), cfg_json).unwrap();

        let q4k = vec![0xa5u8; 144];
        std::fs::write(overlay.join("experts/e0.q4k"), &q4k).unwrap();
        let manifest = format!(r#"{{
            "format":"bw24-expert-overlay-v1",
            "source_dir":"{}",
            "tensors":{{
                "blk.0.ffn_gate_exps.0.weight":{{
                    "file":"experts/e0.q4k","qtype":"Q4_K","ne":[256,1],"bytes":144
                }}
            }}
        }}"#, base.display());
        std::fs::write(overlay.join("manifest.json"), manifest).unwrap();

        let src = Hy3RepackSource::open(&overlay).unwrap();
        assert!(src.preserve_expert_encodings());
        let selected = src.find("blk.0.ffn_gate_exps.0.weight").unwrap();
        assert_eq!(selected.ggml_type, GgmlType::Q4_K);
        assert_eq!(&*selected.bytes, &q4k);
        let (map, off, len) = src.find_expert_mmap("blk.0.ffn_gate_exps.0.weight").unwrap();
        assert_eq!(&map[off..off + len], &q4k);
        let fallback = src.find("blk.0.attn_q.weight").unwrap();
        assert_eq!(fallback.ggml_type, GgmlType::F32);
        assert_eq!(fallback.ne, vec![2, 4]);
        assert_eq!(src.st_dir(), Some(base.as_path()));

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn v2_overlay_supports_repack_fallback_offsets_and_prune_mask() {
        let root = std::env::temp_dir().join(format!("bw24_overlay_v2_test_{}", std::process::id()));
        let base = root.join("base");
        let overlay = root.join("overlay");
        std::fs::create_dir_all(&base).unwrap();
        std::fs::create_dir_all(overlay.join("experts")).unwrap();
        let cfg = r#"{"model_type":"qwen3_moe","num_hidden_layers":1,"hidden_size":4,"num_attention_heads":2,"num_key_value_heads":1,"intermediate_size":8,"vocab_size":10,"max_position_embeddings":128,"num_experts":3,"num_experts_per_tok":2,"moe_intermediate_size":4}"#;
        std::fs::write(base.join("config.json"), cfg).unwrap();
        std::fs::write(base.join("dense.bin"), [9u8, 8, 7, 1, 2, 3, 4, 6]).unwrap();
        std::fs::write(base.join("manifest.json"), r#"{
            "format":"bw24-test-repack-v1","source_dir":".",
            "tensors":{"blk.0.attn_norm.weight":{
                "file":"dense.bin","offset":3,"qtype":"F32","ne":[1],"bytes":4
            }}
        }"#).unwrap();

        let mut expert_blob = vec![0x55u8, 0x66];
        expert_blob.extend(vec![0x22u8; 84]);
        expert_blob.extend(vec![0x33u8; 34]);
        std::fs::write(overlay.join("experts/mixed.bin"), expert_blob).unwrap();
        let router: Vec<u8> = (0..12u32)
            .flat_map(|value| (value as f32 + 0.25).to_le_bytes())
            .collect();
        std::fs::write(overlay.join("router.bin"), &router).unwrap();
        let manifest = format!(r#"{{
            "format":"bw24-expert-overlay-v2","source_dir":"{}",
            "pruned_experts":{{"0":[1]}},
            "tensors":{{"blk.0.ffn_gate_exps.0.weight":{{
                "file":"experts/mixed.bin","offset":2,"qtype":"Q2_K","ne":[256,1],"bytes":84
            }},"blk.0.ffn_gate_exps.2.weight":{{
                "file":"experts/mixed.bin","offset":86,"qtype":"Q8_0","ne":[32,1],"bytes":34
            }},"blk.0.ffn_gate_inp.weight":{{
                "file":"router.bin","offset":0,"qtype":"F32","ne":[4,3],"bytes":48
            }}}}
        }}"#, base.display());
        std::fs::write(overlay.join("manifest.json"), manifest).unwrap();

        let src = Hy3RepackSource::open(&overlay).unwrap();
        assert_eq!(src.config().moe.as_ref().unwrap().expert_count, 3);
        assert_eq!(src.active_experts(0), Some(&[true, false, true][..]));
        let expert = src.find("blk.0.ffn_gate_exps.0.weight").unwrap();
        assert_eq!(expert.ggml_type, GgmlType::Q2_K);
        assert_eq!(&*expert.bytes, &[0x22u8; 84]);
        let q8 = src.find("blk.0.ffn_gate_exps.2.weight").unwrap();
        assert_eq!(q8.ggml_type, GgmlType::Q8_0);
        assert_eq!(&*q8.bytes, &[0x33u8; 34]);
        let (map, off, len) = src.find_expert_mmap("blk.0.ffn_gate_exps.0.weight").unwrap();
        assert_eq!((off, len), (2, 84));
        assert_eq!(&map[off..off + len], &[0x22u8; 84]);
        let disk = src.find_expert_disk("blk.0.ffn_gate_exps.0.weight").unwrap();
        assert_eq!((disk.offset, disk.len), (2, 84));
        assert!(std::sync::Arc::ptr_eq(&disk.map, &map));
        let dense = src.find("blk.0.attn_norm.weight").unwrap();
        assert_eq!(&*dense.bytes, &[1, 2, 3, 4]);
        let healed_router = src.find("blk.0.ffn_gate_inp.weight").unwrap();
        assert_eq!(healed_router.ggml_type, GgmlType::F32);
        assert_eq!(healed_router.ne, vec![4, 3]);
        assert_eq!(&*healed_router.bytes, &router);

        // The opened inode is part of the extent, not borrowed from the loader source.
        drop(src);
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            let expert_path = overlay.join("experts/mixed.bin");
            std::fs::remove_file(&expert_path).unwrap();
            let mut replacement = vec![0x99u8, 0x98];
            replacement.extend(vec![0x77u8; 118]);
            std::fs::write(&expert_path, &replacement).unwrap();

            let mut direct = vec![0u8; disk.len];
            assert_eq!(disk.file.read_at(&mut direct, disk.offset).unwrap(), disk.len);
            assert_eq!(direct, vec![0x22u8; 84]);
            assert_eq!(&std::fs::read(&expert_path).unwrap()[2..86], &[0x77u8; 84]);
        }

        std::fs::remove_dir_all(&root).ok();
    }
}

#[cfg(test)]
mod hy3_repack_probe {
    use super::*;
    use std::io::{Read, Seek, SeekFrom};

    fn repack_dir() -> Option<&'static Path> {
        let dir = Path::new("/data/ai-ml/hf-models/hy3-reap50-q4k-bw24");
        if dir.join("manifest.json").exists() { Some(dir) } else { None }
    }

    fn tsv_row(source_name: &str) -> Option<(String, String)> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../research/hy3-reap50-tensor-inventory.tsv");
        let txt = std::fs::read_to_string(path).ok()?;
        for line in txt.lines().skip(1) {
            let mut cols = line.split('\t');
            let name = cols.next()?;
            let dtype = cols.next()?;
            let shape = cols.next()?;
            if name == source_name {
                return Some((dtype.to_string(), shape.to_string()));
            }
        }
        None
    }

    #[test]
    fn hy3_manifest_offset_roundtrip() {
        let Some(dir) = repack_dir() else { eprintln!("SKIP: Hy3 repack absent"); return; };
        let src = Hy3RepackSource::open(dir).unwrap();
        assert_eq!(src.dir(), dir);
        assert_eq!(src.tensor_count(), 1278);
        let cfg = src.config();
        assert_eq!(cfg.arch, Arch::Hy3);
        assert_eq!(cfg.n_layer, 80, "REAP manifest stripped the appended MTP block");
        assert_eq!(cfg.nextn_predict_layers, 0);
        assert_eq!(cfg.n_layer_total, 80);
        assert_eq!(cfg.moe.as_ref().unwrap().expert_count, 96);
        assert_eq!(cfg.moe.as_ref().unwrap().expert_used_count, 8);
        assert_eq!(cfg.hy3.as_ref().unwrap().first_k_dense_replace, 1);

        let name = "blk.1.ffn_gate_exps.weight";
        let v = src.find(name).unwrap();
        assert_eq!(v.ggml_type, GgmlType::Q4_K);
        assert_eq!(v.ne, vec![4096, 1536, 96]);
        let stride = src.expert_stride(name).unwrap();
        assert_eq!(stride, 3_538_944);
        assert_eq!(v.bytes.len(), stride * 96);

        let expert = 37usize;
        let offset = expert * stride;
        let mut f = std::fs::File::open(dir.join("experts/blk1-gate-96x1536x4096.q4k")).unwrap();
        f.seek(SeekFrom::Start(offset as u64)).unwrap();
        let mut buf = [0u8; 64];
        f.read_exact(&mut buf).unwrap();
        assert_eq!(&v.bytes[offset..offset + buf.len()], &buf);
    }

    #[test]
    fn hy3_inventory_dtype_shape_assertions() {
        let Some(dir) = repack_dir() else { eprintln!("SKIP: Hy3 repack absent"); return; };
        let src = Hy3RepackSource::open(dir).unwrap();
        let cases = [
            ("model.layers.0.self_attn.q_proj.weight", "U32", "8192x512",
             "blk.0.attn_q.weight", GgmlType::Q4_K, vec![4096, 8192]),
            ("model.layers.0.mlp.down_proj.weight", "U32", "4096x1664",
             "blk.0.ffn_down.weight", GgmlType::Q4_K, vec![13312, 4096]),
            ("model.layers.1.mlp.router.gate.weight", "U32", "96x1024",
             "blk.1.ffn_gate_inp.weight", GgmlType::F32, vec![4096, 96]),
            ("model.layers.1.mlp.router.expert_bias", "F32", "96",
             "blk.1.exp_probs_b.bias", GgmlType::F32, vec![96]),
            ("model.layers.1.mlp.shared_mlp.gate_proj.weight", "U32", "1536x512",
             "blk.1.ffn_gate_shexp.weight", GgmlType::Q4_K, vec![4096, 1536]),
            ("model.layers.1.mlp.switch_mlp.gate_proj.weight", "U32", "96x1536x512",
             "blk.1.ffn_gate_exps.weight", GgmlType::Q4_K, vec![4096, 1536, 96]),
            ("model.norm.weight", "BF16", "4096",
             "output_norm.weight", GgmlType::F32, vec![4096]),
        ];
        for (source_name, dtype, shape, ggml_name, qtype, ne) in cases {
            let row = tsv_row(source_name).unwrap_or_else(|| panic!("missing TSV row {source_name}"));
            assert_eq!(row, (dtype.to_string(), shape.to_string()), "{source_name}");
            let v = src.find(ggml_name).unwrap_or_else(|| panic!("missing manifest tensor {ggml_name}"));
            assert_eq!(v.ggml_type, qtype, "{ggml_name}");
            assert_eq!(v.ne, ne, "{ggml_name}");
        }
    }

    #[test]
    fn hy3_load_plan_dry_run_no_cuda() {
        let Some(dir) = repack_dir() else { eprintln!("SKIP: Hy3 repack absent"); return; };
        let src = Hy3RepackSource::open(dir).unwrap();
        let cfg = src.config();
        let hy3 = cfg.hy3.as_ref().unwrap();

        for name in ["token_embd.weight", "output_norm.weight", "output.weight"] {
            assert!(src.find(name).is_some(), "missing {name}");
        }
        for il in 0..cfg.n_layer {
            let p = |s: &str| format!("blk.{il}.{s}");
            for name in [
                p("attn_norm.weight"),
                p("attn_q.weight"),
                p("attn_k.weight"),
                p("attn_v.weight"),
                p("attn_output.weight"),
                p("attn_q_norm.weight"),
                p("attn_k_norm.weight"),
                p("ffn_norm.weight"),
            ] {
                assert!(src.find(&name).is_some(), "missing load-plan tensor {name}");
            }
            if il < hy3.first_k_dense_replace {
                for name in [p("ffn_gate.weight"), p("ffn_up.weight"), p("ffn_down.weight")] {
                    let v = src.find(&name).unwrap_or_else(|| panic!("missing dense tensor {name}"));
                    assert_eq!(v.ggml_type, GgmlType::Q4_K, "{name}");
                    assert_eq!(v.ne.len(), 2, "{name}");
                }
            } else {
                for (name, qtype, rank) in [
                    (p("ffn_gate_inp.weight"), GgmlType::F32, 2usize),
                    (p("exp_probs_b.bias"), GgmlType::F32, 1),
                    (p("ffn_gate_shexp.weight"), GgmlType::Q4_K, 2),
                    (p("ffn_up_shexp.weight"), GgmlType::Q4_K, 2),
                    (p("ffn_down_shexp.weight"), GgmlType::Q4_K, 2),
                    (p("ffn_gate_exps.weight"), GgmlType::Q4_K, 3),
                    (p("ffn_up_exps.weight"), GgmlType::Q4_K, 3),
                    (p("ffn_down_exps.weight"), GgmlType::Q4_K, 3),
                ] {
                    let v = src.find(&name).unwrap_or_else(|| panic!("missing MoE tensor {name}"));
                    assert_eq!(v.ggml_type, qtype, "{name}");
                    assert_eq!(v.ne.len(), rank, "{name}");
                }
            }
        }
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

#[cfg(test)]
mod compressed_tensors_roundtrip {
    use super::*;

    /// CPU roundtrip: build a synthetic compressed-tensors safetensors (weight_packed U8 +
    /// weight_scale F8_E4M3 + weight_global_scale F32 DIVISOR), open via SafetensorsSource, and
    /// verify: (1) nvfp4_quant returns macro_s = 1/global_scale, (2) the .scale sibling lookup
    /// returns the INVERTED value (multiplier, not divisor), (3) the repacked GGUF blocks
    /// dequantize to the correct magnitude range.
    #[test]
    fn ct_global_scale_inversion() {
        let dir = std::env::temp_dir().join(format!("bw24_ct_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // Shape: one layer, out_f=2, in_f=64 (minimum for NVFP4 64-elem blocks)
        let out_f: usize = 2;
        let in_f: usize = 64;
        let packed_bytes = out_f * in_f / 2; // 64 bytes (U8 [2,32])
        let scale_bytes = out_f * in_f / 16; // 8 bytes (F8_E4M3 [2,4])
        let global_scale: f32 = 9408.0; // typical DIVISOR value from real checkpoint

        // Fill packed with non-zero e2m1 codes (code=3 in each nibble -> magnitude 1.5)
        let weight_packed = vec![0x33u8; packed_bytes]; // code 3 in both nibbles
        // Fill scales with a known UE4M3 value (0x38 = exp=7 man=0 -> (1.0+0)*2^0 = 1.0 raw, *0.5 = 0.5)
        let weight_scale = vec![0x38u8; scale_bytes];
        // global_scale as F32 scalar
        let gs_bytes = global_scale.to_le_bytes();

        // Also need a BF16 placeholder for the ".weight" tensor (compressed-tensors has both).
        // We use shape [2,64] BF16 (256 bytes) but it should be IGNORED by the NVFP4 path.
        let bf16_weight = vec![0u8; out_f * in_f * 2]; // zeros

        // Build safetensors file:
        // tensors: model.layers.0.self_attn.q_proj.weight (BF16 [2,64])
        //          model.layers.0.self_attn.q_proj.weight_packed (U8 [2,32])
        //          model.layers.0.self_attn.q_proj.weight_scale (F8_E4M3 [2,4])
        //          model.layers.0.self_attn.q_proj.weight_global_scale (F32 [1])
        let data_len = bf16_weight.len() + weight_packed.len() + weight_scale.len() + gs_bytes.len();
        let mut off = 0usize;
        let bf16_start = off; off += bf16_weight.len();
        let packed_start = off; off += weight_packed.len();
        let scale_start = off; off += weight_scale.len();
        let gs_start = off; off += gs_bytes.len();
        assert_eq!(off, data_len);

        let json = format!(
            concat!(
                "{{",
                "\"model.layers.0.self_attn.q_proj.weight\":{{\"dtype\":\"BF16\",\"shape\":[2,64],\"data_offsets\":[{},{}]}},",
                "\"model.layers.0.self_attn.q_proj.weight_packed\":{{\"dtype\":\"U8\",\"shape\":[2,32],\"data_offsets\":[{},{}]}},",
                "\"model.layers.0.self_attn.q_proj.weight_scale\":{{\"dtype\":\"F8_E4M3\",\"shape\":[2,4],\"data_offsets\":[{},{}]}},",
                "\"model.layers.0.self_attn.q_proj.weight_global_scale\":{{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[{},{}]}}",
                "}}"
            ),
            bf16_start, bf16_start + bf16_weight.len(),
            packed_start, packed_start + weight_packed.len(),
            scale_start, scale_start + weight_scale.len(),
            gs_start, gs_start + gs_bytes.len(),
        );

        let mut buf = Vec::new();
        buf.extend_from_slice(&(json.len() as u64).to_le_bytes());
        buf.extend_from_slice(json.as_bytes());
        buf.extend_from_slice(&bf16_weight);
        buf.extend_from_slice(&weight_packed);
        buf.extend_from_slice(&weight_scale);
        buf.extend_from_slice(&gs_bytes);
        std::fs::write(dir.join("model.safetensors"), &buf).unwrap();

        let cfg_json = r#"{"model_type":"qwen3","num_hidden_layers":1,"hidden_size":64,"num_attention_heads":2,"intermediate_size":128,"vocab_size":10,"max_position_embeddings":128,"num_key_value_heads":2,"head_dim":32}"#;
        std::fs::write(dir.join("config.json"), cfg_json).unwrap();

        let src = SafetensorsSource::open(&dir).unwrap();

        // Test 1: nvfp4_quant returns correct inverted macro_s
        let hf_name = "model.layers.0.self_attn.q_proj.weight";
        let (o, i, _packed, _scales, macro_s) = src.nvfp4_quant(hf_name)
            .expect("nvfp4_quant must detect compressed-tensors format");
        assert_eq!(o, out_f);
        assert_eq!(i, in_f);
        let expected_macro = 1.0 / global_scale;
        assert!(
            (macro_s - expected_macro).abs() < 1e-10,
            "macro_s should be 1/global_scale = {expected_macro}, got {macro_s}"
        );

        // Test 2: the .scale sibling lookup returns the INVERTED value
        let scale_view = src.find("blk.0.attn_q.scale")
            .expect(".scale sibling must resolve for NVFP4 tensors");
        assert_eq!(scale_view.ggml_type, GgmlType::F32);
        assert_eq!(scale_view.ne, vec![1]);
        let returned_scale = f32::from_le_bytes(scale_view.bytes[..4].try_into().unwrap());
        assert!(
            (returned_scale - expected_macro).abs() < 1e-10,
            ".scale lookup should return 1/global_scale = {expected_macro}, got {returned_scale}"
        );

        // Test 3: the returned bytes are Owned (not borrowed raw divisor)
        assert!(
            matches!(scale_view.bytes, std::borrow::Cow::Owned(_)),
            ".scale for compressed-tensors must be Cow::Owned (inverted value)"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Verify modelopt arm still borrows the raw weight_scale_2 (no inversion).
    #[test]
    fn modelopt_scale2_direct_borrow() {
        let dir = std::env::temp_dir().join(format!("bw24_mo_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let out_f: usize = 2;
        let in_f: usize = 64;
        let packed_bytes = out_f * in_f / 2;
        let scale_bytes = out_f * in_f / 16;
        let scale_2: f32 = 0.000106; // typical MULTIPLIER

        let weight = vec![0x33u8; packed_bytes];
        let wscale = vec![0x38u8; scale_bytes];
        let s2_bytes = scale_2.to_le_bytes();

        let mut off = 0usize;
        let w_start = off; off += weight.len();
        let s_start = off; off += wscale.len();
        let s2_start = off; off += s2_bytes.len();

        let json = format!(
            concat!(
                "{{",
                "\"model.layers.0.self_attn.q_proj.weight\":{{\"dtype\":\"U8\",\"shape\":[2,32],\"data_offsets\":[{},{}]}},",
                "\"model.layers.0.self_attn.q_proj.weight_scale\":{{\"dtype\":\"F8_E4M3\",\"shape\":[2,4],\"data_offsets\":[{},{}]}},",
                "\"model.layers.0.self_attn.q_proj.weight_scale_2\":{{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[{},{}]}}",
                "}}"
            ),
            w_start, w_start + weight.len(),
            s_start, s_start + wscale.len(),
            s2_start, s2_start + s2_bytes.len(),
        );

        let mut buf = Vec::new();
        buf.extend_from_slice(&(json.len() as u64).to_le_bytes());
        buf.extend_from_slice(json.as_bytes());
        buf.extend_from_slice(&weight);
        buf.extend_from_slice(&wscale);
        buf.extend_from_slice(&s2_bytes);
        std::fs::write(dir.join("model.safetensors"), &buf).unwrap();

        let cfg_json = r#"{"model_type":"qwen3","num_hidden_layers":1,"hidden_size":64,"num_attention_heads":2,"intermediate_size":128,"vocab_size":10,"max_position_embeddings":128,"num_key_value_heads":2,"head_dim":32}"#;
        std::fs::write(dir.join("config.json"), cfg_json).unwrap();

        let src = SafetensorsSource::open(&dir).unwrap();

        // nvfp4_quant: macro_s should be the raw scale_2 value (direct multiplier)
        let (_, _, _, _, macro_s) = src.nvfp4_quant("model.layers.0.self_attn.q_proj.weight")
            .expect("nvfp4_quant must detect modelopt format");
        assert!(
            (macro_s - scale_2).abs() < 1e-10,
            "modelopt macro_s should be scale_2 directly = {scale_2}, got {macro_s}"
        );

        // .scale sibling: borrowed directly, value == scale_2
        let sv = src.find("blk.0.attn_q.scale")
            .expect(".scale sibling must resolve for modelopt NVFP4");
        let v = f32::from_le_bytes(sv.bytes[..4].try_into().unwrap());
        assert!(
            (v - scale_2).abs() < 1e-10,
            "modelopt .scale should be raw scale_2 = {scale_2}, got {v}"
        );
        assert!(
            matches!(sv.bytes, std::borrow::Cow::Borrowed(_)),
            "modelopt .scale should be Cow::Borrowed (zero-copy)"
        );

        std::fs::remove_dir_all(&dir).ok();
    }
}
