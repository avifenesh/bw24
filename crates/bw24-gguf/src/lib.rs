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

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use memmap2::Mmap;

pub mod dequant;
pub mod config;
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
    pub offset: u64,         // relative to data_start
    pub n_bytes: u64,        // computed size
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 { self.ne.iter().product() }
}

pub struct GgufFile {
    mmap: Mmap,
    /// The same opened inode backing `mmap`, retained for disk-tier positioned reads.
    file: Arc<File>,
    /// Original on-disk path, retained for diagnostics and adjacent artifact lookup.
    path: PathBuf,
    pub version: u32,
    pub metadata: BTreeMap<String, MetaValue>,
    pub tensors: Vec<TensorInfo>,
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

impl GgufFile {
    pub fn open<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = Arc::new(File::open(&path)?);
        let mmap = unsafe { Mmap::map(file.as_ref())? };
        let mut c = Cursor::new(&mmap);

        let magic = c.u32();
        assert_eq!(magic, GGUF_MAGIC, "bad GGUF magic: {magic:#x}");
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
            tensors.push(TensorInfo { name, ne, ggml_type, offset, n_bytes });
        }

        // data section starts at the next `alignment` boundary after the header.
        let header_end = c.pos as u64;
        let data_start = header_end.div_ceil(alignment) * alignment;

        Ok(Self { mmap, file, path, version, metadata, tensors, data_start, alignment })
    }

    /// Raw bytes for a tensor (mmap'd, zero-copy slice).
    pub fn tensor_data(&self, t: &TensorInfo) -> &[u8] {
        let start = (self.data_start + t.offset) as usize;
        &self.mmap[start..start + t.n_bytes as usize]
    }

    /// Original on-disk path of this GGUF file.
    pub fn path(&self) -> &Path { &self.path }

    /// Opened inode backing the parsed mmap. Disk-tier consumers clone this handle instead of
    /// reopening `path`, so a path replacement cannot change the bytes behind a loaded model.
    pub fn opened_file(&self) -> &Arc<File> { &self.file }

    /// Absolute byte range `[start, end)` of a tensor's data WITHIN the GGUF file on disk.
    /// `start = data_start + t.offset`; the disk-tier `HostBuf::Mmap` slices its own file mmap here.
    pub fn tensor_file_range(&self, t: &TensorInfo) -> (usize, usize) {
        let start = (self.data_start + t.offset) as usize;
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
