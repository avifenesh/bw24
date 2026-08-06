//! Minimal GGUF v3 reader, mmap-based, layout copied 1:1 from llama.cpp `ggml/src/gguf.cpp`.
//!
//! On-disk layout (little-endian):
//!   magic "GGUF" (4 bytes) | version u32 (==3) | n_tensors i64 | n_kv i64
//!   n_kv × { key: gguf_string | value_type: u32 | value }
//!   n_tensors × { name: gguf_string | n_dims: u32 | ne[n_dims]: i64 | ggml_type: u32 | offset: u64 }
//!   padding to `general.alignment` (default 32)
//!   tensor data blob (each tensor at data_start + offset)
//!
//! gguf_string = len: u64 | bytes[len]  (no NUL terminator)
//!
//! SPLIT (multi-shard) models: `llama-gguf-split` writes one complete GGUF per shard, each with
//! its own header, its own tensor-info table, and its own data blob, tagged by three KV keys —
//! `split.no` (u16, 0-based), `split.count` (u16), `split.tensors.count` (i32, the TOTAL across
//! all shards). Tensor `offset`s are relative to the OWNING shard's `data_start`. Shard 0 carries
//! the full architecture/tokenizer metadata; later shards carry only the three split keys.
//! `GgufFile::open` on any shard of such a set discovers its siblings by the standard
//! `-%05d-of-%05d.gguf` filename form and presents one merged tensor table, so every caller sees
//! a split model exactly as it sees a single-file one. Step-3.7-Flash IQ4_XS (97.78 GiB, 754
//! tensors over 3 shards) is the model that forced this: blocks 0..21 live in shard 1 and the
//! loader died at `blk.22` with `need post_attention_norm or ffn_norm`.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use memmap2::Mmap;

pub mod d2t;
pub mod dequant;
pub mod config;
pub mod micro_gguf;
pub mod safetensors;
pub mod hf;
pub mod hf_mapping;
pub mod nvfp4_repack;
pub mod source;

pub const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian
pub const GGUF_DEFAULT_ALIGNMENT: u64 = 32;

/// ggml_type ids — values are the on-disk integers (ggml/include/ggml.h).
/// Variant names mirror ggml's C enum exactly (Q4_0, Q8_K, …) by design.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7,
    Q8_0 = 8, Q8_1 = 9, Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13,
    Q6_K = 14, Q8_K = 15, IQ2_XXS = 16, IQ2_XS = 17, IQ3_XXS = 18,
    IQ1_S = 19, IQ4_NL = 20, IQ3_S = 21, IQ2_S = 22, IQ4_XS = 23,
    I8 = 24, I16 = 25, I32 = 26, I64 = 27, F64 = 28, IQ1_M = 29,
    BF16 = 30, TQ1_0 = 34, TQ2_0 = 35, MXFP4 = 39, NVFP4 = 40, Q1_0 = 41,
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Option<Self> {
        use GgmlType::*;
        Some(match v {
            0 => F32, 1 => F16, 2 => Q4_0, 3 => Q4_1, 6 => Q5_0, 7 => Q5_1,
            8 => Q8_0, 9 => Q8_1, 10 => Q2_K, 11 => Q3_K, 12 => Q4_K, 13 => Q5_K,
            14 => Q6_K, 15 => Q8_K, 16 => IQ2_XXS, 17 => IQ2_XS, 18 => IQ3_XXS,
            19 => IQ1_S, 20 => IQ4_NL, 21 => IQ3_S, 22 => IQ2_S, 23 => IQ4_XS,
            24 => I8, 25 => I16, 26 => I32, 27 => I64, 28 => F64, 29 => IQ1_M,
            30 => BF16, 34 => TQ1_0, 35 => TQ2_0, 39 => MXFP4, 40 => NVFP4, 41 => Q1_0,
            _ => return None,
        })
    }

    /// (block_size in elements, type_size in bytes) — from ggml.c type_traits.
    /// bytes_for_n_elems = n_elems / block_size * type_size.
    pub fn block_and_type_size(self) -> (u64, u64) {
        use GgmlType::*;
        match self {
            F32 => (1, 4), F16 => (1, 2), BF16 => (1, 2), F64 => (1, 8),
            I8 => (1, 1), I16 => (1, 2), I32 => (1, 4), I64 => (1, 8),
            Q4_0 => (32, 18),  // 2 (d) + 16 (16 bytes for 32×4bit)
            Q4_1 => (32, 20),  // 2 d + 2 m + 16
            Q5_0 => (32, 22),  // 2 d + 4 qh + 16
            Q5_1 => (32, 24),  // 2 d + 2 m + 4 qh + 16
            Q8_0 => (32, 34),  // 2 d + 32 int8
            Q8_1 => (32, 36),  // 4 (d,s as fp16×2) + 32
            // k-quants, super-block QK_K=256
            Q2_K => (256, 84),
            Q3_K => (256, 110),
            Q4_K => (256, 144),
            Q5_K => (256, 176),
            Q6_K => (256, 210),
            Q8_K => (256, 292),
            IQ4_NL => (32, 18),
            IQ4_XS => (256, 136),
            // i-quants (all QK_K=256 super-blocks) — sizes from ggml-common.h static_asserts
            IQ2_XXS => (256, 66),
            IQ2_XS => (256, 74),
            IQ2_S => (256, 82),
            IQ3_XXS => (256, 98),
            IQ3_S => (256, 110),
            IQ1_S => (256, 50),
            IQ1_M => (256, 56),
            MXFP4 => (32, 17),  // 1 (E8M0 scale) + 16 (32×4bit e2m1)
            NVFP4 => (64, 36),  // 4 (UE4M3 sub-scales, 1 per 16 elems) + 32 (64×4bit e2m1)
            // remaining (Q2_K..Q5_K covered above; k-quant variants): panic-on-use
            other => panic!("block_and_type_size not implemented for {other:?}"),
        }
    }
}

/// A metadata value. Arrays keep their element type + raw decoded values.
#[derive(Debug, Clone)]
pub enum MetaValue {
    U8(u8), I8(i8), U16(u16), I16(i16), U32(u32), I32(i32),
    U64(u64), I64(i64), F32(f32), F64(f64), Bool(bool), String(String),
    Array(Vec<MetaValue>),
}

impl MetaValue {
    pub fn as_u64(&self) -> Option<u64> {
        Some(match self {
            MetaValue::U8(v) => *v as u64, MetaValue::U16(v) => *v as u64,
            MetaValue::U32(v) => *v as u64, MetaValue::U64(v) => *v,
            MetaValue::I8(v) => *v as u64, MetaValue::I16(v) => *v as u64,
            MetaValue::I32(v) => *v as u64, MetaValue::I64(v) => *v as u64,
            MetaValue::Bool(v) => *v as u64,
            _ => return None,
        })
    }
    pub fn as_f32(&self) -> Option<f32> {
        Some(match self {
            MetaValue::F32(v) => *v, MetaValue::F64(v) => *v as f32,
            _ => return None,
        })
    }
    pub fn as_str(&self) -> Option<&str> {
        if let MetaValue::String(s) = self { Some(s) } else { None }
    }
    pub fn as_str_array(&self) -> Option<Vec<&str>> {
        if let MetaValue::Array(a) = self {
            a.iter().map(|v| v.as_str()).collect()
        } else { None }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub ne: Vec<u64>,        // dimensions, ne[0] fastest
    pub ggml_type: GgmlType,
    pub offset: u64,         // relative to the OWNING shard's data_start
    pub n_bytes: u64,        // computed size
    /// Index into `GgufFile::shards` of the file this tensor's bytes live in. Always 0 for a
    /// single-file GGUF, so `offset` keeps its historical meaning there (relative to `data_start`).
    pub shard: usize,
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 { self.ne.iter().product() }
}

/// One physical GGUF file. A single-file model has exactly one; a split model has `split.count`.
struct Shard {
    mmap: Mmap,
    /// The same opened inode backing `mmap`, retained for disk-tier positioned reads.
    file: Arc<File>,
    /// On-disk path, retained for diagnostics and adjacent artifact lookup.
    path: PathBuf,
    /// Where this shard's tensor-data blob begins — each shard has its OWN header, so each has
    /// its own `data_start`. A tensor's absolute offset is `shards[t.shard].data_start + t.offset`.
    data_start: u64,
}

pub struct GgufFile {
    /// Shard 0 first, then ascending `split.no`. Length 1 for a single-file model.
    shards: Vec<Shard>,
    pub version: u32,
    /// Merged metadata. Shard 0 carries the architecture/tokenizer KVs and wins every collision;
    /// later shards contribute only keys shard 0 lacks (in practice nothing but `split.no`).
    pub metadata: BTreeMap<String, MetaValue>,
    /// Every tensor across every shard, in shard-then-file order. `TensorInfo::shard` says where.
    pub tensors: Vec<TensorInfo>,
    /// Shard 0's data start. Kept public for back-compat; use `tensor_file_range` for a tensor's
    /// real absolute offset, which on a split model is relative to ITS OWN shard.
    pub data_start: u64,
    pub alignment: u64,
}

struct Cursor<'a> { buf: &'a [u8], pos: usize }

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self { Self { buf, pos: 0 } }
    fn read<const N: usize>(&mut self) -> [u8; N] {
        let s: [u8; N] = self.buf[self.pos..self.pos + N].try_into().unwrap();
        self.pos += N;
        s
    }
    fn u32(&mut self) -> u32 { u32::from_le_bytes(self.read::<4>()) }
    fn i64(&mut self) -> i64 { i64::from_le_bytes(self.read::<8>()) }
    fn u64(&mut self) -> u64 { u64::from_le_bytes(self.read::<8>()) }
    fn string(&mut self) -> String {
        let len = self.u64() as usize;
        let s = String::from_utf8_lossy(&self.buf[self.pos..self.pos + len]).into_owned();
        self.pos += len;
        s
    }
    fn value(&mut self, type_id: u32) -> MetaValue {
        match type_id {
            0 => MetaValue::U8(self.read::<1>()[0]),
            1 => MetaValue::I8(self.read::<1>()[0] as i8),
            2 => MetaValue::U16(u16::from_le_bytes(self.read::<2>())),
            3 => MetaValue::I16(i16::from_le_bytes(self.read::<2>())),
            4 => MetaValue::U32(self.u32()),
            5 => MetaValue::I32(i32::from_le_bytes(self.read::<4>())),
            6 => MetaValue::F32(f32::from_le_bytes(self.read::<4>())),
            7 => MetaValue::Bool(self.read::<1>()[0] != 0),
            8 => MetaValue::String(self.string()),
            9 => {
                let elem_type = self.u32();
                let n = self.u64() as usize;
                let mut v = Vec::with_capacity(n);
                for _ in 0..n { v.push(self.value(elem_type)); }
                MetaValue::Array(v)
            }
            10 => MetaValue::U64(self.u64()),
            11 => MetaValue::I64(self.i64()),
            12 => MetaValue::F64(f64::from_le_bytes(self.read::<8>())),
            other => panic!("unknown gguf_type {other}"),
        }
    }
}

/// Parse ONE physical GGUF file: `(shard, version, metadata, tensor infos with shard=usize::MAX)`.
/// `TensorInfo::shard` is patched by the caller once the shard's index is known.
fn parse_one(path: PathBuf) -> std::io::Result<(Shard, u32, BTreeMap<String, MetaValue>, Vec<TensorInfo>)> {
    let file = Arc::new(File::open(&path)?);
    let mmap = unsafe { Mmap::map(file.as_ref())? };
    let mut c = Cursor::new(&mmap);

    let magic = c.u32();
    assert_eq!(magic, GGUF_MAGIC, "bad GGUF magic: {magic:#x} in {}", path.display());
    let version = c.u32();
    assert_eq!(version, 3, "only GGUF v3 supported, got {version}");
    let n_tensors = c.i64();
    let n_kv = c.i64();

    // --- metadata KV ---
    let mut metadata = BTreeMap::new();
    for _ in 0..n_kv {
        let key = c.string();
        let vtype = c.u32();
        let val = c.value(vtype);
        metadata.insert(key, val);
    }

    let alignment = metadata.get("general.alignment")
        .and_then(|v| v.as_u64())
        .unwrap_or(GGUF_DEFAULT_ALIGNMENT);

    // --- tensor infos ---
    let mut tensors = Vec::with_capacity(n_tensors as usize);
    for _ in 0..n_tensors {
        let name = c.string();
        let n_dims = c.u32() as usize;
        let mut ne = Vec::with_capacity(n_dims);
        for _ in 0..n_dims { ne.push(c.i64() as u64); }
        let ggml_type = GgmlType::from_u32(c.u32())
            .unwrap_or_else(|| panic!("unknown ggml_type in tensor {name}"));
        let offset = c.u64();
        let n_elems: u64 = ne.iter().product();
        let (blck, tsize) = ggml_type.block_and_type_size();
        assert!(n_elems % blck == 0, "tensor {name} elems {n_elems} not divisible by block {blck}");
        let n_bytes = n_elems / blck * tsize;
        tensors.push(TensorInfo { name, ne, ggml_type, offset, n_bytes, shard: usize::MAX });
    }

    // data section starts at the next `alignment` boundary after the header.
    let header_end = c.pos as u64;
    let data_start = header_end.div_ceil(alignment) * alignment;
    Ok((Shard { mmap, file, path, data_start }, version, metadata, tensors))
}

/// Sibling paths of a split shard, in ascending `split.no`, given ANY member's path.
///
/// `llama-gguf-split` names shards `<prefix>-%05d-of-%05d.gguf` (gguf-split.cpp's
/// `SPLIT_PATH_FORMAT`). Rather than parse the number out of the name we rebuild every expected
/// name from `count` — so a shard whose filename disagrees with its own `split.no` cannot silently
/// map to the wrong bytes. Returns None if the name does not carry the standard suffix.
fn split_sibling_paths(path: &Path, count: usize) -> Option<Vec<PathBuf>> {
    let name = path.file_name()?.to_str()?;
    let stem = name.strip_suffix(".gguf")?;
    // trailing "-%05d-of-%05d"
    let (head, tail) = stem.rsplit_once("-of-")?;
    if tail.len() != 5 || !tail.bytes().all(|b| b.is_ascii_digit()) { return None; }
    let (prefix, num) = head.rsplit_once('-')?;
    if num.len() != 5 || !num.bytes().all(|b| b.is_ascii_digit()) { return None; }
    let dir = path.parent()?;
    Some((1..=count)
        .map(|i| dir.join(format!("{prefix}-{i:05}-of-{count:05}.gguf")))
        .collect())
}

impl GgufFile {
    /// Open a GGUF model. If `path` names a shard of a split (multi-file) model — detected by the
    /// `split.count` KV — every sibling shard is opened too and the result presents ONE merged
    /// tensor table. Callers cannot tell a split model from a single-file one.
    pub fn open<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let (shard0, version, mut metadata, mut tensors) = parse_one(path.clone())?;
        let alignment = metadata.get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(GGUF_DEFAULT_ALIGNMENT);
        let data_start = shard0.data_start;

        let split_count = metadata.get("split.count").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        // count 0 or 1 = not split. Any member of the set is a valid entry point: the sibling
        // names come from the filename form, and shard 0's KVs (architecture, tokenizer) win the
        // merge below, so opening shard 3 yields the same model as opening shard 1.
        if split_count <= 1 {
            for t in &mut tensors { t.shard = 0; }
            return Ok(Self { shards: vec![shard0], version, metadata, tensors, data_start, alignment });
        }

        let paths = split_sibling_paths(&path, split_count).ok_or_else(|| std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("{} declares split.count={split_count} but its name is not the \
                     -%05d-of-%05d.gguf split form; cannot find sibling shards", path.display()),
        ))?;

        let total_expected = metadata.get("split.tensors.count")
            .and_then(|v| v.as_u64()).unwrap_or(0);

        let mut shards: Vec<Shard> = Vec::with_capacity(split_count);
        let mut all: Vec<TensorInfo> = Vec::new();
        let mut merged: BTreeMap<String, MetaValue> = BTreeMap::new();
        // Every shard is (re)parsed here, including the one we were handed — one extra header
        // parse + lazy mmap is nothing against a 105 GB model, and it keeps the merge loop
        // uniform. `tensors`/`shard0` from the probe above are dropped.
        drop(shard0);
        drop(tensors);
        for (i, p) in paths.iter().enumerate() {
            let (sh, ver, meta, mut ts) = parse_one(p.clone())?;
            assert_eq!(ver, version, "shard {} GGUF version {ver} != {version}", p.display());
            let no = meta.get("split.no").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            assert_eq!(no, i, "shard {} declares split.no={no}, expected {i} from its filename",
                       p.display());
            let cnt = meta.get("split.count").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            assert_eq!(cnt, split_count, "shard {} declares split.count={cnt} != {split_count}",
                       p.display());
            for t in &mut ts { t.shard = i; }
            all.append(&mut ts);
            // Shard 0 wins: it holds the architecture + tokenizer KVs. Later shards may only ADD.
            for (k, v) in meta {
                merged.entry(k).or_insert(v);
            }
            shards.push(sh);
        }
        if total_expected > 0 {
            assert_eq!(all.len() as u64, total_expected,
                       "split model has {} tensors across {split_count} shards but \
                        split.tensors.count={total_expected}", all.len());
        }
        metadata = merged;
        let alignment = metadata.get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(GGUF_DEFAULT_ALIGNMENT);
        let data_start = shards[0].data_start;
        Ok(Self { shards, version, metadata, tensors: all, data_start, alignment })
    }

    /// Raw bytes for a tensor (mmap'd, zero-copy slice) from its OWNING shard.
    pub fn tensor_data(&self, t: &TensorInfo) -> &[u8] {
        let sh = &self.shards[t.shard];
        let start = (sh.data_start + t.offset) as usize;
        &sh.mmap[start..start + t.n_bytes as usize]
    }

    /// On-disk path of shard 0 (the whole model for a single-file GGUF).
    pub fn path(&self) -> &Path { &self.shards[0].path }

    /// Number of physical files backing this model (1 unless split).
    pub fn n_shards(&self) -> usize { self.shards.len() }

    /// On-disk path of a given shard.
    pub fn shard_path(&self, i: usize) -> &Path { &self.shards[i].path }

    /// Opened inode backing shard 0's parsed mmap. Disk-tier consumers clone this handle instead
    /// of reopening `path`, so a path replacement cannot change the bytes behind a loaded model.
    /// SPLIT MODELS: use `shard_file(t.shard)` — shard 0's inode does not hold every tensor.
    pub fn opened_file(&self) -> &Arc<File> { &self.shards[0].file }

    /// Opened inode backing a given shard's parsed mmap.
    pub fn shard_file(&self, i: usize) -> &Arc<File> { &self.shards[i].file }

    /// Absolute byte range `[start, end)` of a tensor's data within **its own shard's** file.
    /// `start = shards[t.shard].data_start + t.offset`; the disk-tier `HostBuf::Mmap` slices the
    /// mmap of that same shard (pair this with `shard_mmap_of`/`shard_file`, never with shard 0's).
    pub fn tensor_file_range(&self, t: &TensorInfo) -> (usize, usize) {
        let start = (self.shards[t.shard].data_start + t.offset) as usize;
        (start, start + t.n_bytes as usize)
    }

    pub fn find(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    pub fn arch(&self) -> Option<&str> {
        self.metadata.get("general.architecture").and_then(|v| v.as_str())
    }

    /// Get a metadata value, trying `{arch}.{suffix}` then the literal key.
    pub fn meta_arch(&self, suffix: &str) -> Option<&MetaValue> {
        if let Some(arch) = self.arch() {
            if let Some(v) = self.metadata.get(&format!("{arch}.{suffix}")) {
                return Some(v);
            }
        }
        self.metadata.get(suffix)
    }
}

#[cfg(test)]
mod split_tests {
    use super::*;

    /// Serialize a minimal but REAL GGUF v3 file: header, KVs, tensor infos, aligned data blob.
    /// Every tensor is F32 with `ne = [n]` and its bytes are `fill` repeated, so a wrong-shard read
    /// is detectable by value rather than only by length.
    fn write_gguf(path: &Path, kv: &[(&str, MetaValue)], tensors: &[(&str, u64, u8)]) {
        let mut h: Vec<u8> = Vec::new();
        h.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        h.extend_from_slice(&3u32.to_le_bytes());
        h.extend_from_slice(&(tensors.len() as i64).to_le_bytes());
        h.extend_from_slice(&(kv.len() as i64).to_le_bytes());
        let put_str = |h: &mut Vec<u8>, s: &str| {
            h.extend_from_slice(&(s.len() as u64).to_le_bytes());
            h.extend_from_slice(s.as_bytes());
        };
        for (k, v) in kv {
            put_str(&mut h, k);
            match v {
                MetaValue::U16(x) => {
                    h.extend_from_slice(&2u32.to_le_bytes());
                    h.extend_from_slice(&x.to_le_bytes());
                }
                MetaValue::I32(x) => {
                    h.extend_from_slice(&5u32.to_le_bytes());
                    h.extend_from_slice(&x.to_le_bytes());
                }
                MetaValue::String(s) => {
                    h.extend_from_slice(&8u32.to_le_bytes());
                    put_str(&mut h, s);
                }
                other => panic!("test writer does not handle {other:?}"),
            }
        }
        // tensor infos: offsets are relative to THIS file's data_start, packed in order
        let mut off = 0u64;
        for (name, n, _) in tensors {
            put_str(&mut h, name);
            h.extend_from_slice(&1u32.to_le_bytes()); // n_dims
            h.extend_from_slice(&(*n as i64).to_le_bytes());
            h.extend_from_slice(&(GgmlType::F32 as u32).to_le_bytes());
            h.extend_from_slice(&off.to_le_bytes());
            off += n * 4;
        }
        let data_start = (h.len() as u64).div_ceil(GGUF_DEFAULT_ALIGNMENT) * GGUF_DEFAULT_ALIGNMENT;
        h.resize(data_start as usize, 0);
        for (_, n, fill) in tensors {
            h.extend(std::iter::repeat_n(*fill, (*n * 4) as usize));
        }
        std::fs::write(path, &h).unwrap();
    }

    /// A 2-shard split pair in a fresh temp dir. Shard 0 carries the arch KV; shard 1 carries only
    /// the split keys — exactly how llama-gguf-split writes them (verified against the real
    /// Step-3.7-Flash IQ4_XS headers, research/step37-bringup-20260802/raw/).
    fn write_split_pair(tag: &str) -> (PathBuf, PathBuf, PathBuf) {
        let dir = std::env::temp_dir().join(format!("memra-split-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p0 = dir.join("m-00001-of-00002.gguf");
        let p1 = dir.join("m-00002-of-00002.gguf");
        write_gguf(&p0, &[
            ("general.architecture", MetaValue::String("step35".into())),
            ("split.no", MetaValue::U16(0)),
            ("split.count", MetaValue::U16(2)),
            ("split.tensors.count", MetaValue::I32(3)),
        ], &[("blk.0.w", 8, 0xA1), ("blk.1.w", 4, 0xA2)]);
        write_gguf(&p1, &[
            ("split.no", MetaValue::U16(1)),
            ("split.count", MetaValue::U16(2)),
            ("split.tensors.count", MetaValue::I32(3)),
        ], &[("blk.2.w", 16, 0xB1)]);
        (dir, p0, p1)
    }

    #[test]
    fn split_model_presents_one_merged_tensor_table() {
        let (dir, p0, _p1) = write_split_pair("merge");
        let g = GgufFile::open(&p0).unwrap();
        assert_eq!(g.n_shards(), 2);
        assert_eq!(g.tensors.len(), 3, "all three tensors must be visible from shard 0");
        // The tensor that lives in shard 1 is the one the step37 boot died on.
        let t = g.find("blk.2.w").expect("blk.2.w is in shard 1 and must be found");
        assert_eq!(t.shard, 1);
        assert_eq!(g.tensor_data(t), vec![0xB1u8; 64].as_slice());
        // ...and shard 0's tensors still read correctly.
        assert_eq!(g.tensor_data(g.find("blk.0.w").unwrap()), vec![0xA1u8; 32].as_slice());
        assert_eq!(g.tensor_data(g.find("blk.1.w").unwrap()), vec![0xA2u8; 16].as_slice());
        // Metadata comes from shard 0 (shard 1 has no architecture KV at all).
        assert_eq!(g.arch(), Some("step35"));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn any_shard_is_a_valid_entry_point() {
        let (dir, p0, p1) = write_split_pair("entry");
        let from0 = GgufFile::open(&p0).unwrap();
        let from1 = GgufFile::open(&p1).unwrap();
        // Opening the LAST shard must yield the same model, including shard 0's metadata.
        assert_eq!(from1.arch(), Some("step35"));
        assert_eq!(from1.tensors.len(), from0.tensors.len());
        for t in &from0.tensors {
            let u = from1.find(&t.name).unwrap();
            assert_eq!((u.shard, u.offset, u.n_bytes), (t.shard, t.offset, t.n_bytes));
            assert_eq!(from1.tensor_data(u), from0.tensor_data(t));
        }
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn tensor_file_range_is_relative_to_the_owning_shard() {
        let (dir, p0, p1) = write_split_pair("range");
        let g = GgufFile::open(&p0).unwrap();
        // The shard-1 tensor's range must address shard 1's file, so it must reproduce those bytes
        // when applied to shard 1 — and must NOT be a global offset past the end of shard 0.
        let t = g.find("blk.2.w").unwrap();
        let (s, e) = g.tensor_file_range(t);
        let raw1 = std::fs::read(&p1).unwrap();
        assert_eq!(&raw1[s..e], vec![0xB1u8; 64].as_slice());
        assert_eq!(g.shard_path(1), p1.as_path());
        assert!(!std::sync::Arc::ptr_eq(g.shard_file(0), g.shard_file(1)));
        let raw0 = std::fs::read(&p0).unwrap();
        let (s0, e0) = g.tensor_file_range(g.find("blk.0.w").unwrap());
        assert_eq!(&raw0[s0..e0], vec![0xA1u8; 32].as_slice());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn single_file_gguf_is_unchanged_one_shard() {
        let dir = std::env::temp_dir().join(format!("memra-split-single-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("solo.gguf");
        write_gguf(&p, &[("general.architecture", MetaValue::String("qwen35".into()))],
                   &[("tok_embd.weight", 8, 0x5A)]);
        let g = GgufFile::open(&p).unwrap();
        assert_eq!(g.n_shards(), 1);
        let t = g.find("tok_embd.weight").unwrap();
        assert_eq!(t.shard, 0);
        assert_eq!(g.tensor_data(t), vec![0x5Au8; 32].as_slice());
        // data_start keeps its historical meaning for a single-file model.
        let (s, _e) = g.tensor_file_range(t);
        assert_eq!(s as u64, g.data_start + t.offset);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn split_shard_with_a_nonstandard_filename_is_a_clear_error() {
        let dir = std::env::temp_dir().join(format!("memra-split-badname-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("renamed-by-hand.gguf");
        write_gguf(&p, &[
            ("split.no", MetaValue::U16(0)),
            ("split.count", MetaValue::U16(3)),
        ], &[("blk.0.w", 4, 0x11)]);
        let msg = match GgufFile::open(&p) {
            Ok(_) => panic!("a split shard with a non-standard filename must not open silently"),
            Err(e) => e.to_string(),
        };
        assert!(msg.contains("split.count=3") && msg.contains("sibling shards"),
                "error must name the split count and the missing siblings, got: {msg}");
        std::fs::remove_dir_all(dir).ok();
    }
}
