//! Minimal safetensors reader, mmap-based. Parallel to the GGUF reader.
//!
//! On-disk layout (little-endian):
//!   header_len: u64 (8 bytes) | header_len bytes of UTF-8 JSON | raw tensor byte buffer
//!
//! Header JSON: { tensor_name: { "dtype": str, "shape": [usize], "data_offsets": [begin, end] }, ... }
//! plus an optional free-form "__metadata__" object. `data_offsets` are byte ranges INTO the
//! post-header buffer (which begins at byte `8 + header_len`), row-major.
//!
//! Multi-shard: a `model.safetensors.index.json` maps each tensor name to its shard file:
//!   { "metadata": {...}, "weight_map": { tensor_name -> "model-0000X-of-0000N.safetensors" } }
//!
//! NOTE: this module hand-parses the small header JSON (no serde dependency in bw24-gguf).
//! The grammar we accept is exactly what `safetensors` emits: objects, arrays, strings, integers.

use std::collections::HashMap;
use std::path::Path;
use memmap2::Mmap;

use crate::GgmlType;

const N_LEN: usize = 8; // size_of::<u64>()
const MAX_HEADER: usize = 100_000_000; // DoS guard (matches safetensors crate)

/// safetensors dtype string -> bw24 GgmlType. FP8 is deferred (explicit panic, never silent).
/// Shape note handled by the caller: safetensors shape is row-major outer..inner, bw24 `ne` is
/// inner-fastest, so the caller reverses the shape.
pub fn st_dtype_to_ggml(s: &str) -> GgmlType {
    use GgmlType::*;
    match s {
        "F32" => F32,
        "F16" => F16,
        "BF16" => BF16,
        "F64" => F64,
        "I8" => I8,
        "I16" => I16,
        "I32" => I32,
        "I64" => I64,
        // FP8 deferred in v1 — explicit failure, NOT silent garbage.
        "F8_E4M3" | "F8_E5M2" | "F8_E8M0" => {
            panic!("FP8 ({s}) safetensors not yet supported; use the GGUF twin or an F16/BF16 checkpoint")
        }
        // U8 / BOOL have no GgmlType equivalent and never appear as model weights here.
        other => panic!("unsupported safetensors dtype {other}"),
    }
}

/// One tensor's header entry.
#[derive(Debug, Clone)]
pub struct StInfo {
    pub dtype: String,
    pub shape: Vec<u64>,         // row-major, outer..inner (as stored)
    pub data_offsets: [usize; 2], // [begin, end) into the post-header buffer
}

impl StInfo {
    /// bw24 `ne` (inner-fastest) is the reverse of the safetensors shape.
    pub fn ne(&self) -> Vec<u64> {
        self.shape.iter().rev().cloned().collect()
    }
    pub fn ggml_type(&self) -> GgmlType {
        st_dtype_to_ggml(&self.dtype)
    }
}

/// A single mmap'd safetensors shard.
pub struct StShard {
    mmap: Mmap,
    data_base: usize, // 8 + header_len
    infos: HashMap<String, StInfo>,
}

impl StShard {
    pub fn open<P: AsRef<Path>>(p: P) -> std::io::Result<Self> {
        let f = std::fs::File::open(p)?;
        let mmap = unsafe { Mmap::map(&f)? };
        Self::from_mmap(mmap)
    }

    fn from_mmap(mmap: Mmap) -> std::io::Result<Self> {
        assert!(mmap.len() >= N_LEN, "safetensors file too small for header length");
        let hlen = u64::from_le_bytes(mmap[..N_LEN].try_into().unwrap()) as usize;
        assert!(
            hlen <= MAX_HEADER && N_LEN + hlen <= mmap.len(),
            "bad/oversized safetensors header (len={hlen}, file={})",
            mmap.len()
        );
        let json = std::str::from_utf8(&mmap[N_LEN..N_LEN + hlen])
            .expect("safetensors header is not valid UTF-8");
        let infos = parse_header(json);
        Ok(Self { mmap, data_base: N_LEN + hlen, infos })
    }

    /// Zero-copy bytes for a tensor (mirrors GgufFile::tensor_data).
    pub fn raw(&self, name: &str) -> Option<(&StInfo, &[u8])> {
        let i = self.infos.get(name)?;
        let s = self.data_base + i.data_offsets[0];
        let e = self.data_base + i.data_offsets[1];
        Some((i, &self.mmap[s..e]))
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.infos.keys()
    }

    pub fn len(&self) -> usize {
        self.infos.len()
    }
    pub fn is_empty(&self) -> bool {
        self.infos.is_empty()
    }
}

/// A whole safetensors model: one or more shards, routed by tensor name.
pub struct StModel {
    shards: Vec<StShard>,
    map: HashMap<String, usize>, // tensor_name -> shard index
}

impl StModel {
    /// Open a model from a directory. Prefers `model.safetensors.index.json` (multi-shard);
    /// falls back to a single `model.safetensors`. Also accepts an explicit file path.
    pub fn open(path: &Path) -> std::io::Result<Self> {
        if path.is_file() {
            // Direct path to a single safetensors file.
            let sh = StShard::open(path)?;
            let map = sh.names().map(|n| (n.clone(), 0)).collect();
            return Ok(Self { shards: vec![sh], map });
        }
        let dir = path;
        let idx = dir.join("model.safetensors.index.json");
        if idx.exists() {
            let txt = std::fs::read_to_string(&idx)?;
            let weight_map = parse_index_weight_map(&txt);
            // distinct shard file names, in stable sorted order
            let mut files: Vec<String> = weight_map.values().cloned().collect();
            files.sort();
            files.dedup();
            let pos: HashMap<&String, usize> =
                files.iter().enumerate().map(|(n, f)| (f, n)).collect();
            let shards = files
                .iter()
                .map(|f| StShard::open(dir.join(f)))
                .collect::<Result<Vec<_>, _>>()?;
            let map = weight_map
                .iter()
                .map(|(t, f)| (t.clone(), pos[f]))
                .collect();
            Ok(Self { shards, map })
        } else {
            let single = dir.join("model.safetensors");
            let sh = StShard::open(single)?;
            let map = sh.names().map(|n| (n.clone(), 0)).collect();
            Ok(Self { shards: vec![sh], map })
        }
    }

    /// Zero-copy bytes + header info for a tensor, routed to the owning shard.
    pub fn raw(&self, name: &str) -> Option<(&StInfo, &[u8])> {
        let &si = self.map.get(name)?;
        self.shards[si].raw(name)
    }

    /// All tensor names across all shards.
    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.map.keys()
    }

    pub fn n_tensors(&self) -> usize {
        self.map.len()
    }
}

// ============================ minimal JSON parsing (header + index) ============================
//
// We only parse the exact shapes the safetensors writer emits. This avoids adding a serde
// dependency to bw24-gguf (which currently has only `memmap2`). Tolerant of whitespace.

struct Json<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Json<'a> {
    fn new(s: &'a str) -> Self {
        Self { b: s.as_bytes(), i: 0 }
    }
    fn skip_ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }
    fn peek(&mut self) -> u8 {
        self.skip_ws();
        self.b[self.i]
    }
    fn eat(&mut self, c: u8) {
        self.skip_ws();
        assert_eq!(self.b[self.i], c, "json: expected '{}' at {}", c as char, self.i);
        self.i += 1;
    }
    /// Parse a JSON string (no escape handling beyond \" and \\, which suffices for tensor names).
    fn string(&mut self) -> String {
        self.eat(b'"');
        let mut out = String::new();
        while self.i < self.b.len() {
            let c = self.b[self.i];
            self.i += 1;
            match c {
                b'"' => break,
                b'\\' => {
                    let e = self.b[self.i];
                    self.i += 1;
                    out.push(match e {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'n' => '\n',
                        b't' => '\t',
                        b'r' => '\r',
                        other => other as char,
                    });
                }
                _ => out.push(c as char),
            }
        }
        out
    }
    /// Parse a non-negative integer (offsets/shape are always >= 0 in safetensors).
    fn u64(&mut self) -> u64 {
        self.skip_ws();
        let start = self.i;
        while self.i < self.b.len() && self.b[self.i].is_ascii_digit() {
            self.i += 1;
        }
        assert!(self.i > start, "json: expected integer at {start}");
        std::str::from_utf8(&self.b[start..self.i])
            .unwrap()
            .parse()
            .unwrap()
    }
    /// Skip an arbitrary JSON value (used for "__metadata__" and index "metadata").
    fn skip_value(&mut self) {
        match self.peek() {
            b'{' => {
                self.eat(b'{');
                if self.peek() != b'}' {
                    loop {
                        let _ = self.string();
                        self.eat(b':');
                        self.skip_value();
                        if self.peek() == b',' {
                            self.eat(b',');
                        } else {
                            break;
                        }
                    }
                }
                self.eat(b'}');
            }
            b'[' => {
                self.eat(b'[');
                if self.peek() != b']' {
                    loop {
                        self.skip_value();
                        if self.peek() == b',' {
                            self.eat(b',');
                        } else {
                            break;
                        }
                    }
                }
                self.eat(b']');
            }
            b'"' => {
                let _ = self.string();
            }
            _ => {
                // number, true, false, or null
                self.skip_ws();
                while self.i < self.b.len()
                    && !matches!(self.b[self.i], b',' | b'}' | b']' | b' ' | b'\t' | b'\n' | b'\r')
                {
                    self.i += 1;
                }
            }
        }
    }
}

/// Parse the safetensors header JSON into the tensor info map (`__metadata__` dropped).
fn parse_header(json: &str) -> HashMap<String, StInfo> {
    let mut p = Json::new(json);
    let mut out = HashMap::new();
    p.eat(b'{');
    if p.peek() == b'}' {
        p.eat(b'}');
        return out;
    }
    loop {
        let key = p.string();
        p.eat(b':');
        if key == "__metadata__" {
            p.skip_value();
        } else {
            // { "dtype": "...", "shape": [...], "data_offsets": [a,b] } — fields in any order.
            p.eat(b'{');
            let mut dtype = String::new();
            let mut shape: Vec<u64> = Vec::new();
            let mut offsets = [0usize; 2];
            loop {
                let field = p.string();
                p.eat(b':');
                match field.as_str() {
                    "dtype" => dtype = p.string(),
                    "shape" => {
                        p.eat(b'[');
                        if p.peek() != b']' {
                            loop {
                                shape.push(p.u64());
                                if p.peek() == b',' {
                                    p.eat(b',');
                                } else {
                                    break;
                                }
                            }
                        }
                        p.eat(b']');
                    }
                    "data_offsets" => {
                        p.eat(b'[');
                        offsets[0] = p.u64() as usize;
                        p.eat(b',');
                        offsets[1] = p.u64() as usize;
                        p.eat(b']');
                    }
                    _ => p.skip_value(),
                }
                if p.peek() == b',' {
                    p.eat(b',');
                } else {
                    break;
                }
            }
            p.eat(b'}');
            out.insert(key, StInfo { dtype, shape, data_offsets: offsets });
        }
        if p.peek() == b',' {
            p.eat(b',');
        } else {
            break;
        }
    }
    p.eat(b'}');
    out
}

/// Parse a `model.safetensors.index.json`, returning the weight_map (tensor -> shard file).
fn parse_index_weight_map(json: &str) -> HashMap<String, String> {
    let mut p = Json::new(json);
    let mut out = HashMap::new();
    p.eat(b'{');
    loop {
        let key = p.string();
        p.eat(b':');
        if key == "weight_map" {
            p.eat(b'{');
            if p.peek() != b'}' {
                loop {
                    let t = p.string();
                    p.eat(b':');
                    let f = p.string();
                    out.insert(t, f);
                    if p.peek() == b',' {
                        p.eat(b',');
                    } else {
                        break;
                    }
                }
            }
            p.eat(b'}');
        } else {
            p.skip_value();
        }
        if p.peek() == b',' {
            p.eat(b',');
        } else {
            break;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny valid safetensors byte buffer by hand and parse it back.
    fn build_synthetic() -> Vec<u8> {
        // two tensors:
        //   "a.weight" F32 shape [2,3] -> 6 f32 = 24 bytes at [0,24)
        //   "b.norm"   BF16 shape [3]  -> 3 bf16 = 6 bytes  at [24,30)
        let json = r#"{"__metadata__":{"format":"pt"},"a.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"b.norm":{"dtype":"BF16","shape":[3],"data_offsets":[24,30]}}"#;
        let jb = json.as_bytes();
        let mut buf = Vec::new();
        buf.extend_from_slice(&(jb.len() as u64).to_le_bytes());
        buf.extend_from_slice(jb);
        // tensor data: a.weight = [0,1,2,3,4,5] f32
        for v in 0..6u32 {
            buf.extend_from_slice(&(v as f32).to_le_bytes());
        }
        // b.norm = three bf16 (1.0, 2.0, -1.0) = 0x3F80, 0x4000, 0xBF80
        for v in [0x3F80u16, 0x4000, 0xBF80] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf
    }

    fn write_temp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("bw24_st_test_{}_{}.safetensors", std::process::id(), name));
        std::fs::write(&p, bytes).unwrap();
        p
    }

    #[test]
    fn synthetic_header_roundtrip() {
        let bytes = build_synthetic();
        let p = write_temp("rt", &bytes);
        let sh = StShard::open(&p).unwrap();
        assert_eq!(sh.len(), 2, "two tensors (metadata dropped)");

        let (ia, ra) = sh.raw("a.weight").expect("a.weight");
        assert_eq!(ia.dtype, "F32");
        assert_eq!(ia.shape, vec![2, 3]);
        // shape-reversal: ne = [3,2] -> in_features=ne[0]=3, out_features=ne[1]=2
        assert_eq!(ia.ne(), vec![3, 2]);
        assert_eq!(ia.ggml_type(), GgmlType::F32);
        assert_eq!(ra.len(), 24);
        let f0 = f32::from_le_bytes(ra[0..4].try_into().unwrap());
        let f5 = f32::from_le_bytes(ra[20..24].try_into().unwrap());
        assert_eq!(f0, 0.0);
        assert_eq!(f5, 5.0);

        let (ib, rb) = sh.raw("b.norm").expect("b.norm");
        assert_eq!(ib.dtype, "BF16");
        assert_eq!(ib.shape, vec![3]);
        assert_eq!(ib.ne(), vec![3]);
        assert_eq!(rb.len(), 6);
        // bytes honored at the right offset
        assert_eq!(u16::from_le_bytes(rb[0..2].try_into().unwrap()), 0x3F80);

        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn synthetic_dequant_through_path() {
        // round-trip the bytes back through the shared dequant path: F32 verbatim, BF16 known.
        let bytes = build_synthetic();
        let p = write_temp("dq", &bytes);
        let sh = StShard::open(&p).unwrap();
        let (ia, ra) = sh.raw("a.weight").unwrap();
        let av = crate::dequant::dequantize(ia.ggml_type(), ra, 6);
        assert_eq!(av, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let (ib, rb) = sh.raw("b.norm").unwrap();
        let bv = crate::dequant::dequantize(ib.ggml_type(), rb, 3);
        assert_eq!(bv, vec![1.0, 2.0, -1.0]);
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn single_file_model_open() {
        let bytes = build_synthetic();
        let p = write_temp("single", &bytes);
        let m = StModel::open(&p).unwrap(); // explicit file path
        assert_eq!(m.n_tensors(), 2);
        let (_, ra) = m.raw("a.weight").expect("routed");
        assert_eq!(ra.len(), 24);
        assert!(m.raw("missing").is_none());
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn multi_shard_index_routing() {
        // Two shards, an index.json mapping each tensor to its file. Assert routing.
        let dir = std::env::temp_dir().join(format!("bw24_st_idx_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // shard 0: one F32 tensor
        let j0 = r#"{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let mut s0 = Vec::new();
        s0.extend_from_slice(&(j0.len() as u64).to_le_bytes());
        s0.extend_from_slice(j0.as_bytes());
        s0.extend_from_slice(&7.0f32.to_le_bytes());
        s0.extend_from_slice(&8.0f32.to_le_bytes());
        std::fs::write(dir.join("model-00001-of-00002.safetensors"), &s0).unwrap();

        // shard 1: one F32 tensor
        let j1 = r#"{"y":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut s1 = Vec::new();
        s1.extend_from_slice(&(j1.len() as u64).to_le_bytes());
        s1.extend_from_slice(j1.as_bytes());
        s1.extend_from_slice(&9.0f32.to_le_bytes());
        std::fs::write(dir.join("model-00002-of-00002.safetensors"), &s1).unwrap();

        let idx = r#"{"metadata":{"total_size":12},"weight_map":{"x":"model-00001-of-00002.safetensors","y":"model-00002-of-00002.safetensors"}}"#;
        std::fs::write(dir.join("model.safetensors.index.json"), idx).unwrap();

        let m = StModel::open(&dir).unwrap();
        assert_eq!(m.n_tensors(), 2);
        let (_, rx) = m.raw("x").expect("x routes to shard 0");
        assert_eq!(f32::from_le_bytes(rx[0..4].try_into().unwrap()), 7.0);
        let (_, ry) = m.raw("y").expect("y routes to shard 1");
        assert_eq!(f32::from_le_bytes(ry[0..4].try_into().unwrap()), 9.0);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    #[should_panic(expected = "FP8")]
    fn fp8_panics_explicitly() {
        st_dtype_to_ggml("F8_E4M3");
    }

    /// Real on-disk safetensors: parse a multi-shard HF checkpoint and assert known tensor
    /// shapes/dtypes/offsets round-trip. Skipped (not failed) when the model is absent.
    /// Point BW24_ST_TEST_DIR at an HF model dir (with model.safetensors.index.json) to run.
    /// Defaults to the Qwen3-1.7B snapshot present on this dev box.
    #[test]
    fn real_qwen3_17b_header() {
        let dir = std::env::var("BW24_ST_TEST_DIR").unwrap_or_else(|_| {
            "/data/ai-ml/hf-models/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e".to_string()
        });
        let dirp = std::path::Path::new(&dir);
        if !dirp.join("model.safetensors.index.json").exists() {
            eprintln!("SKIP real_qwen3_17b_header: no model at {dir}");
            return;
        }
        let m = StModel::open(dirp).expect("open multi-shard model");
        // Qwen3-1.7B has 311 weights spread over 2 shards (310 in shard 1 + lm_head in shard 2).
        assert_eq!(m.n_tensors(), 311, "tensor count");

        // embed_tokens: BF16 shape [vocab=151936, hidden=2048] -> ne [2048, 151936]
        let (e, eb) = m.raw("model.embed_tokens.weight").expect("embed");
        assert_eq!(e.dtype, "BF16");
        assert_eq!(e.shape, vec![151936, 2048]);
        assert_eq!(e.ne(), vec![2048, 151936]);
        assert_eq!(eb.len(), 151936 * 2048 * 2, "BF16 = 2 bytes/elem");

        // layer-0 q_proj: BF16 [2048,2048] -> ne [2048,2048], offsets honored.
        let (q, qb) = m.raw("model.layers.0.self_attn.q_proj.weight").expect("q_proj");
        assert_eq!(q.dtype, "BF16");
        assert_eq!(q.shape, vec![2048, 2048]);
        assert_eq!(qb.len(), 2048 * 2048 * 2);

        // norm weight present and small (1-D, [2048]).
        let (nrm, nb) = m.raw("model.norm.weight").expect("final norm");
        assert_eq!(nrm.shape, vec![2048]);
        assert_eq!(nb.len(), 2048 * 2);

        // lm_head lives in shard 2 — proves cross-shard routing on a real file.
        let (lm, _) = m.raw("lm_head.weight").expect("lm_head (shard 2)");
        assert_eq!(lm.dtype, "BF16");
        assert_eq!(lm.shape, vec![151936, 2048]);

        // BF16 bytes dequant to finite f32 (spot-check the first row of the final norm).
        let nv = crate::dequant::dequantize(nrm.ggml_type(), nb, 2048);
        assert!(nv.iter().all(|v| v.is_finite()), "norm dequants finite");
        eprintln!("real_qwen3_17b_header OK: {} tensors, norm[0]={}", m.n_tensors(), nv[0]);
    }

    /// Real config.json -> ModelConfig parse against the same on-disk model. Skipped if absent.
    #[test]
    fn real_qwen3_17b_config() {
        let dir = std::env::var("BW24_ST_TEST_DIR").unwrap_or_else(|_| {
            "/data/ai-ml/hf-models/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e".to_string()
        });
        let cfgp = std::path::Path::new(&dir).join("config.json");
        if !cfgp.exists() {
            eprintln!("SKIP real_qwen3_17b_config: no config at {cfgp:?}");
            return;
        }
        let mc = crate::config::ModelConfig::from_config_json(&cfgp).expect("parse config.json");
        assert_eq!(mc.arch, crate::config::Arch::Qwen3);
        assert_eq!(mc.n_layer, 28);
        assert_eq!(mc.n_embd, 2048);
        assert_eq!(mc.n_head, 16);
        assert_eq!(mc.n_head_kv, 8);
        assert_eq!(mc.head_dim_k, 128);
        assert_eq!(mc.n_ff, 6144);
        assert_eq!(mc.n_vocab, 151936);
        assert!(mc.moe.is_none() && mc.ssm.is_none());
    }
}
