//! Minimal GGUF v3 WRITER + the glm-dsa micro fixture (MLA bring-up increment 2).
//!
//! Nothing big is ever committed: tests call `write_glm_dsa_micro` / `write_glm52_meta_only`
//! at test time into a temp path (~100 KB) and parse/load against that. The writer emits the
//! exact on-disk layout `GgufFile::open` reads (lib.rs header comment); the fixture emits the
//! exact tensor names/shapes the llama.cpp glm-dsa converter writes, pinned in
//! `research/mla-bringup-20260801/DESIGN.md` §3.1 + `RECEIPTS.md` §5 — including the
//! `attn_kv_b` -> (`attn_k_b` TRANSPOSED nope slice, `attn_v_b` v slice) split convention
//! and the `blk.N.nextn.*` MTP tensor names.
//!
//! Scope guard: this is a test/fixture generator, not a production export path. GGUF stays
//! memra's delivery format; this writer only knows the value types the fixtures need.

use crate::{GgmlType, GGUF_DEFAULT_ALIGNMENT, GGUF_MAGIC};
use std::io::Write;
use std::path::Path;

/// Metadata value for the writer (subset of gguf types the fixtures need).
pub enum MetaW {
    U32(u32),
    F32(f32),
    Bool(bool),
    Str(&'static str),
    ArrBool(Vec<bool>),
    /// Per-layer u32 array — step35 writes `attention.head_count`/`head_count_kv` this way.
    ArrU32(Vec<u32>),
    /// Per-layer f32 array — step35's `swiglu_clamp_exp`/`_shexp`.
    ArrF32(Vec<f32>),
}

pub struct GgufWriter {
    meta: Vec<(String, MetaW)>,
    tensors: Vec<(String, Vec<u64>, GgmlType, Vec<u8>)>,
}

impl Default for GgufWriter {
    fn default() -> Self { Self::new() }
}

impl GgufWriter {
    pub fn new() -> Self {
        GgufWriter { meta: Vec::new(), tensors: Vec::new() }
    }

    pub fn kv(&mut self, key: &str, v: MetaW) {
        self.meta.push((key.to_string(), v));
    }

    pub fn tensor_f32(&mut self, name: &str, ne: &[u64], data: &[f32]) {
        assert_eq!(ne.iter().product::<u64>() as usize, data.len(), "{name} ne/data mismatch");
        let bytes = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.tensors.push((name.to_string(), ne.to_vec(), GgmlType::F32, bytes));
    }

    /// Pre-encoded quant tensor (e.g. Q8_0 expert slabs via `nvfp4_repack::f32_to_q8_0`).
    pub fn tensor_raw(&mut self, name: &str, ne: &[u64], ty: GgmlType, bytes: Vec<u8>) {
        let n: u64 = ne.iter().product();
        let (blck, tsize) = ty.block_and_type_size();
        assert_eq!(n % blck, 0, "{name}: {n} elems not divisible by block {blck}");
        assert_eq!((n / blck * tsize) as usize, bytes.len(), "{name} byte-size mismatch");
        self.tensors.push((name.to_string(), ne.to_vec(), ty, bytes));
    }

    pub fn write(self, path: &Path) -> std::io::Result<()> {
        let align = GGUF_DEFAULT_ALIGNMENT;
        let mut buf: Vec<u8> = Vec::new();
        buf.extend(GGUF_MAGIC.to_le_bytes());
        buf.extend(3u32.to_le_bytes()); // version
        buf.extend((self.tensors.len() as i64).to_le_bytes());
        buf.extend((self.meta.len() as i64).to_le_bytes());

        fn wstr(buf: &mut Vec<u8>, s: &str) {
            buf.extend((s.len() as u64).to_le_bytes());
            buf.extend(s.as_bytes());
        }
        // value type ids: U32=4, F32=6, Bool=7, String=8, Array=9 (see lib.rs Cursor::value)
        for (k, v) in &self.meta {
            wstr(&mut buf, k);
            match v {
                MetaW::U32(x) => { buf.extend(4u32.to_le_bytes()); buf.extend(x.to_le_bytes()); }
                MetaW::F32(x) => { buf.extend(6u32.to_le_bytes()); buf.extend(x.to_le_bytes()); }
                MetaW::Bool(x) => { buf.extend(7u32.to_le_bytes()); buf.push(*x as u8); }
                MetaW::Str(s) => { buf.extend(8u32.to_le_bytes()); wstr(&mut buf, s); }
                MetaW::ArrBool(a) => {
                    buf.extend(9u32.to_le_bytes());
                    buf.extend(7u32.to_le_bytes()); // elem type bool
                    buf.extend((a.len() as u64).to_le_bytes());
                    for &b in a { buf.push(b as u8); }
                }
                MetaW::ArrU32(a) => {
                    buf.extend(9u32.to_le_bytes());
                    buf.extend(4u32.to_le_bytes()); // elem type u32
                    buf.extend((a.len() as u64).to_le_bytes());
                    for &x in a { buf.extend(x.to_le_bytes()); }
                }
                MetaW::ArrF32(a) => {
                    buf.extend(9u32.to_le_bytes());
                    buf.extend(6u32.to_le_bytes()); // elem type f32
                    buf.extend((a.len() as u64).to_le_bytes());
                    for &x in a { buf.extend(x.to_le_bytes()); }
                }
            }
        }

        // tensor infos: offsets are relative to data_start, each aligned to `align`.
        let mut offset = 0u64;
        let mut offsets = Vec::with_capacity(self.tensors.len());
        for (_, _, _, bytes) in &self.tensors {
            offsets.push(offset);
            offset = (offset + bytes.len() as u64).div_ceil(align) * align;
        }
        for ((name, ne, ty, _), off) in self.tensors.iter().zip(&offsets) {
            wstr(&mut buf, name);
            buf.extend((ne.len() as u32).to_le_bytes());
            for &d in ne { buf.extend((d as i64).to_le_bytes()); }
            buf.extend((*ty as u32).to_le_bytes());
            buf.extend(off.to_le_bytes());
        }

        // pad to data_start, then the tensor blob (each tensor padded to align).
        let data_start = (buf.len() as u64).div_ceil(align) * align;
        buf.resize(data_start as usize, 0);
        for ((_, _, _, bytes), off) in self.tensors.iter().zip(&offsets) {
            buf.resize((data_start + off) as usize, 0);
            buf.extend(bytes);
        }

        let mut f = std::fs::File::create(path)?;
        f.write_all(&buf)?;
        f.flush()
    }
}

// ---------------------------------------------------------------------------------------------
// glm-dsa micro fixture — 2 trunk layers + 1 NextN/MTP layer at hidden 64 / rank-16 class.
// ---------------------------------------------------------------------------------------------

/// Micro fixture geometry (GLM-5.2 ratios shrunk to hidden 64 / kv rank 16).
#[derive(Clone, Copy, Debug)]
pub struct MicroGlmDims {
    pub n_embd: u64,      // 64   (GLM-5.2: 6144)
    pub n_head: u64,      // 4    (64)
    pub q_lora: u64,      // 32   (2048)
    pub kv_lora: u64,     // 16   (512)
    pub d_nope: u64,      // 24   (192)
    pub d_rope: u64,      // 8    (64)
    pub d_v: u64,         // 32   (256)
    pub n_vocab: u64,     // 128  (154880)
    pub n_ff: u64,        // 96   (12288)
    pub n_expert: u64,    // 8    (256)
    pub n_used: u64,      // 2    (8)
    pub moe_ff: u64,      // 32   (2048)
    pub n_trunk: u64,     // 2    (78)   — block_count = n_trunk + nextn
    pub nextn: u64,       // 1    (1)
    pub idx_heads: u64,   // 2    (32)
    pub idx_dim: u64,     // 16   (128)
    pub idx_topk: u64,    // 4    (2048)
}

pub const MICRO: MicroGlmDims = MicroGlmDims {
    n_embd: 64, n_head: 4, q_lora: 32, kv_lora: 16, d_nope: 24, d_rope: 8, d_v: 32,
    n_vocab: 128, n_ff: 96, n_expert: 8, n_used: 2, moe_ff: 32, n_trunk: 2, nextn: 1,
    idx_heads: 2, idx_dim: 16, idx_topk: 4,
};

impl MicroGlmDims {
    pub fn qk_head_dim(&self) -> u64 { self.d_nope + self.d_rope }         // key_length_mla
    pub fn latent_dim(&self) -> u64 { self.kv_lora + self.d_rope }         // attention.key_length
    pub fn block_count(&self) -> u64 { self.n_trunk + self.nextn }
}

/// xorshift64* — deterministic weights, no external crates (same generator as mla.rs tests).
pub struct Rng(pub u64);
impl Rng {
    pub fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        let v = (self.0.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as u32;
        (v as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
    /// n values uniform in [-scale, scale).
    pub fn fill(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next_f32() * scale).collect()
    }
}

/// The 21-full/57-shared GLM-5.2 indexer layout: full at layers 0,1,2 then every 4th from 6 to
/// 74 (pinned config `indexer_types`, RECEIPTS.md §2 — identical set to llama.cpp's default
/// table, `config::glm52_default_indexer_types`).
pub fn glm52_indexer_types() -> Vec<bool> {
    crate::config::glm52_default_indexer_types(78)
}

/// GLM-5.2 metadata-only GGUF (zero tensors): every key the llama.cpp converter writes, at the
/// pinned real values (RECEIPTS.md §2/§5). The parse-arm unit tests read `ModelConfig` off this.
/// `attention.head_count_kv` is intentionally OMITTED — the increment-1 receipts do not pin it;
/// the on-box gguf-dump audit (increment 3) confirms it against the real artifact.
pub fn write_glm52_meta_only(path: &Path) -> std::io::Result<()> {
    let mut w = GgufWriter::new();
    let a = |s: &str| format!("glm-dsa.{s}");
    w.kv("general.architecture", MetaW::Str("glm-dsa"));
    w.kv("general.name", MetaW::Str("GLM-5.2-meta-pin"));
    // block_count INCLUDES the NextN/MTP layer (79 = 78 trunk + 1), matching memra's
    // convention (hybrid.rs: n_trunk = n_layer - nextn) and llama.cpp glm-dsa's
    // `n_layer() excludes the nextn layer` comment for the case-78 type match.
    w.kv(&a("block_count"), MetaW::U32(79));
    w.kv(&a("context_length"), MetaW::U32(1_048_576));
    w.kv(&a("embedding_length"), MetaW::U32(6144));
    w.kv(&a("feed_forward_length"), MetaW::U32(12288));
    w.kv(&a("attention.head_count"), MetaW::U32(64));
    w.kv(&a("attention.layer_norm_rms_epsilon"), MetaW::F32(1e-5));
    w.kv(&a("rope.freq_base"), MetaW::F32(8_000_000.0));
    w.kv(&a("rope.dimension_count"), MetaW::U32(64));
    w.kv(&a("attention.key_length"), MetaW::U32(576));
    w.kv(&a("attention.value_length"), MetaW::U32(512));
    w.kv(&a("attention.key_length_mla"), MetaW::U32(256));
    w.kv(&a("attention.value_length_mla"), MetaW::U32(256));
    w.kv(&a("attention.q_lora_rank"), MetaW::U32(2048));
    w.kv(&a("attention.kv_lora_rank"), MetaW::U32(512));
    w.kv(&a("expert_count"), MetaW::U32(256));
    w.kv(&a("expert_used_count"), MetaW::U32(8));
    w.kv(&a("expert_feed_forward_length"), MetaW::U32(2048));
    w.kv(&a("expert_shared_count"), MetaW::U32(1));
    w.kv(&a("expert_weights_scale"), MetaW::F32(2.5));
    w.kv(&a("expert_weights_norm"), MetaW::Bool(true));
    w.kv(&a("expert_gating_func"), MetaW::U32(2)); // LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID
    w.kv(&a("leading_dense_block_count"), MetaW::U32(3));
    w.kv(&a("nextn_predict_layers"), MetaW::U32(1));
    w.kv(&a("attention.indexer.head_count"), MetaW::U32(32));
    w.kv(&a("attention.indexer.key_length"), MetaW::U32(128));
    w.kv(&a("attention.indexer.top_k"), MetaW::U32(2048));
    w.kv(&a("attention.indexer.types"), MetaW::ArrBool(glm52_indexer_types()));
    w.kv(&a("vocab_size"), MetaW::U32(154_880));
    w.write(path)
}

// ---------------------------------------------------------------------------------------------
// step35 (StepFun Step-3.7-Flash) metadata pin
// ---------------------------------------------------------------------------------------------

/// The 3:1 SWA pattern of Step-3.7-Flash, as the official GGUF serializes it:
/// `[false, true, true, true]` repeating — full attention exactly where `il % 4 == 0`
/// (12 full + 33 SWA over 45 trunk blocks).
pub fn step35_swa_pattern(n: usize) -> Vec<bool> {
    (0..n).map(|il| il % 4 != 0).collect()
}

/// Per-layer query-head counts: 64 on the full-attn layers, 96 on the SWA layers.
pub fn step35_head_counts(n: usize) -> Vec<u32> {
    step35_swa_pattern(n).iter().map(|&swa| if swa { 96 } else { 64 }).collect()
}

/// Step-3.7-Flash metadata-only GGUF (zero tensors): every `step35.*` key at the values parsed
/// from the REAL official IQ4_XS artifact header — receipt
/// `research/step37-bringup-20260802/raw/gguf-header-stepfun-iq4xs-shard1-20260802.txt`
/// (49 KVs, 754 tensors, 45 blocks). The config parse-arm tests read `ModelConfig` off this, so
/// a drift in the artifact's key names/types fails a unit test rather than a 105 GB load.
///
/// `nextn_predict_layers` is intentionally OMITTED: the TRUNK GGUF does not carry it (the MTP
/// blocks ship in the standalone `Step3.7-flash-mtp-*.gguf`, which writes 48 blocks + the key).
pub fn write_step35_meta_only(path: &Path) -> std::io::Result<()> {
    let mut w = GgufWriter::new();
    let a = |s: &str| format!("step35.{s}");
    const N: usize = 45;
    w.kv("general.architecture", MetaW::Str("step35"));
    w.kv("general.type", MetaW::Str("model"));
    w.kv("general.name", MetaW::Str("Step-3.7"));
    w.kv("general.size_label", MetaW::Str("288x7.4B"));
    w.kv(&a("block_count"), MetaW::U32(N as u32));
    w.kv(&a("context_length"), MetaW::U32(262_144));
    w.kv(&a("embedding_length"), MetaW::U32(4096));
    w.kv(&a("feed_forward_length"), MetaW::U32(11264));
    // ARRAY, not scalar — 64 on full-attn layers, 96 on SWA. This is the key that panics a
    // scalar-only reader (`as_u64` returns None on an Array).
    w.kv(&a("attention.head_count"), MetaW::ArrU32(step35_head_counts(N)));
    w.kv(&a("attention.head_count_kv"), MetaW::ArrU32(vec![8; N]));
    w.kv(&a("attention.key_length"), MetaW::U32(128));
    w.kv(&a("attention.value_length"), MetaW::U32(128));
    w.kv(&a("attention.layer_norm_rms_epsilon"), MetaW::F32(1e-5));
    w.kv(&a("attention.sliding_window"), MetaW::U32(512));
    // BOOL array in the real file (llama.cpp reads it into `is_swa_impl`).
    w.kv(&a("attention.sliding_window_pattern"), MetaW::ArrBool(step35_swa_pattern(N)));
    w.kv(&a("rope.freq_base"), MetaW::F32(5_000_000.0));
    w.kv(&a("rope.freq_base_swa"), MetaW::F32(10_000.0));
    w.kv(&a("expert_count"), MetaW::U32(288));
    w.kv(&a("expert_used_count"), MetaW::U32(8));
    w.kv(&a("expert_feed_forward_length"), MetaW::U32(1280));
    w.kv(&a("expert_shared_feed_forward_length"), MetaW::U32(1280));
    w.kv(&a("expert_weights_scale"), MetaW::F32(3.0));
    w.kv(&a("expert_weights_norm"), MetaW::Bool(true));
    w.kv(&a("expert_gating_func"), MetaW::U32(2)); // sigmoid
    w.kv(&a("leading_dense_block_count"), MetaW::U32(3));
    w.kv(&a("moe_every_n_layers"), MetaW::U32(1));
    // Clamp arrays: zero everywhere except the last two layers (43 -> 7.0, 44 -> 16.0).
    let mut clamp = vec![0.0f32; N];
    clamp[43] = 7.0;
    clamp[44] = 16.0;
    w.kv(&a("swiglu_clamp_exp"), MetaW::ArrF32(clamp.clone()));
    w.kv(&a("swiglu_clamp_shexp"), MetaW::ArrF32(clamp));
    w.kv("tokenizer.ggml.model", MetaW::Str("gpt2"));
    w.kv("tokenizer.ggml.pre", MetaW::Str("deepseek-v3"));
    w.kv("tokenizer.ggml.bos_token_id", MetaW::U32(0));
    w.kv("tokenizer.ggml.eos_token_id", MetaW::U32(128_007));
    w.kv("tokenizer.ggml.padding_token_id", MetaW::U32(1));
    w.kv("tokenizer.ggml.add_bos_token", MetaW::Bool(true));
    // NOTE: the real artifact carries NO `step35.vocab_size` key — n_vocab comes off
    // `token_embd.weight`'s last dim. A meta-only fixture therefore parses n_vocab == 0; the
    // test asserts that (it is the behavior the 105 GB load depends on, not a fixture artifact).
    w.write(path)
}

/// Write the glm-dsa micro fixture: `MICRO` dims, deterministic random weights, every tensor
/// name/shape of DESIGN.md §3.1. Layer 0 = dense FFN + FULL indexer; layer 1 = MoE FFN + shared
/// indexer (NO indexer tensors — the GLM-5.2 partial-indexer property early loaders broke on);
/// blk.2 = the NextN/MTP layer (dense-MLA attention + MoE FFN + nextn.* glue).
///
/// Split convention (RECEIPTS §5): `attn_kv_b` [kv_lora, n_head*(nope+v)] is generated as the
/// source of truth; `attn_k_b` = its per-head nope slice TRANSPOSED to (nope, kv_lora, head) and
/// `attn_v_b` = its per-head v slice (kv_lora, v, head) — byte-derived, so a consumer that
/// decodes either representation must agree exactly.
///
/// MoE expert slabs are Q8_0-encoded (HostExps rejects F32); everything else stays F32 so CPU
/// reference tests read exact values.
pub fn write_glm_dsa_micro(path: &Path, seed: u64) -> std::io::Result<MicroGlmDims> {
    let d = MICRO;
    let mut rng = Rng(seed | 1);
    let mut w = GgufWriter::new();
    let a = |s: &str| format!("glm-dsa.{s}");

    w.kv("general.architecture", MetaW::Str("glm-dsa"));
    w.kv("general.name", MetaW::Str("glm-dsa-micro"));
    w.kv(&a("block_count"), MetaW::U32(d.block_count() as u32));
    w.kv(&a("context_length"), MetaW::U32(512));
    w.kv(&a("embedding_length"), MetaW::U32(d.n_embd as u32));
    w.kv(&a("feed_forward_length"), MetaW::U32(d.n_ff as u32));
    w.kv(&a("attention.head_count"), MetaW::U32(d.n_head as u32));
    w.kv(&a("attention.head_count_kv"), MetaW::U32(1)); // MLA decode is MQA: one latent stream
    w.kv(&a("attention.layer_norm_rms_epsilon"), MetaW::F32(1e-5));
    w.kv(&a("rope.freq_base"), MetaW::F32(8_000_000.0));
    w.kv(&a("rope.dimension_count"), MetaW::U32(d.d_rope as u32));
    w.kv(&a("attention.key_length"), MetaW::U32(d.latent_dim() as u32));
    w.kv(&a("attention.value_length"), MetaW::U32(d.kv_lora as u32));
    w.kv(&a("attention.key_length_mla"), MetaW::U32(d.qk_head_dim() as u32));
    w.kv(&a("attention.value_length_mla"), MetaW::U32(d.d_v as u32));
    w.kv(&a("attention.q_lora_rank"), MetaW::U32(d.q_lora as u32));
    w.kv(&a("attention.kv_lora_rank"), MetaW::U32(d.kv_lora as u32));
    w.kv(&a("expert_count"), MetaW::U32(d.n_expert as u32));
    w.kv(&a("expert_used_count"), MetaW::U32(d.n_used as u32));
    w.kv(&a("expert_feed_forward_length"), MetaW::U32(d.moe_ff as u32));
    w.kv(&a("expert_shared_count"), MetaW::U32(1));
    w.kv(&a("expert_weights_scale"), MetaW::F32(2.5));
    w.kv(&a("expert_weights_norm"), MetaW::Bool(true));
    w.kv(&a("expert_gating_func"), MetaW::U32(2));
    w.kv(&a("leading_dense_block_count"), MetaW::U32(1));
    w.kv(&a("nextn_predict_layers"), MetaW::U32(d.nextn as u32));
    w.kv(&a("attention.indexer.head_count"), MetaW::U32(d.idx_heads as u32));
    w.kv(&a("attention.indexer.key_length"), MetaW::U32(d.idx_dim as u32));
    w.kv(&a("attention.indexer.top_k"), MetaW::U32(d.idx_topk as u32));
    w.kv(&a("attention.indexer.types"), MetaW::ArrBool(vec![true, false]));
    w.kv(&a("vocab_size"), MetaW::U32(d.n_vocab as u32));

    // weights ~ U[-1/sqrt(in), 1/sqrt(in)) keep activations O(1) for the CPU reference gates.
    let ws = |rng: &mut Rng, in_f: u64, out: u64| {
        let s = 1.0 / (in_f as f32).sqrt();
        rng.fill((in_f * out) as usize, s)
    };
    // norm weights near 1.0 (jittered so a dropped norm is caught by the gates).
    let nw = |rng: &mut Rng, n: u64| -> Vec<f32> {
        (0..n).map(|_| 1.0 + 0.1 * rng.next_f32()).collect()
    };

    w.tensor_f32("token_embd.weight", &[d.n_embd, d.n_vocab], &ws(&mut rng, d.n_embd, d.n_vocab));
    w.tensor_f32("output_norm.weight", &[d.n_embd], &nw(&mut rng, d.n_embd));
    w.tensor_f32("output.weight", &[d.n_embd, d.n_vocab], &ws(&mut rng, d.n_embd, d.n_vocab));

    let (nh, nope, rope, v, rq, rkv, h) =
        (d.n_head, d.d_nope, d.d_rope, d.d_v, d.q_lora, d.kv_lora, d.n_embd);

    for il in 0..d.block_count() {
        let p = |s: &str| format!("blk.{il}.{s}");

        // ---- MLA attention (identical set on trunk + MTP blocks) ----
        w.tensor_f32(&p("attn_norm.weight"), &[h], &nw(&mut rng, h));
        w.tensor_f32(&p("attn_q_a.weight"), &[h, rq], &ws(&mut rng, h, rq));
        w.tensor_f32(&p("attn_q_a_norm.weight"), &[rq], &nw(&mut rng, rq));
        w.tensor_f32(&p("attn_q_b.weight"), &[rq, nh * (nope + rope)],
                     &ws(&mut rng, rq, nh * (nope + rope)));
        w.tensor_f32(&p("attn_kv_a_mqa.weight"), &[h, rkv + rope], &ws(&mut rng, h, rkv + rope));
        w.tensor_f32(&p("attn_kv_a_norm.weight"), &[rkv], &nw(&mut rng, rkv));

        // kv_b is the SOURCE; k_b/v_b are byte-derived per the conversion split.
        // kv_b layout: ne {kv_lora, n_head*(nope+v)} — row n = head*(nope+v)+j, rank fastest.
        let kv_b = ws(&mut rng, rkv, nh * (nope + v));
        let (nope_u, v_u, rkv_u, nh_u) = (nope as usize, v as usize, rkv as usize, nh as usize);
        // k_b: ne {nope, kv_lora, head} — element (hd, r, p) = kv_b[hd*(nope+v)+p][r] (TRANSPOSED)
        let mut k_b = vec![0f32; nh_u * rkv_u * nope_u];
        // v_b: ne {kv_lora, v, head} — element (hd, j, r) = kv_b[hd*(nope+v)+nope+j][r]
        let mut v_b = vec![0f32; nh_u * v_u * rkv_u];
        for hd in 0..nh_u {
            for r in 0..rkv_u {
                for pn in 0..nope_u {
                    k_b[hd * rkv_u * nope_u + r * nope_u + pn] =
                        kv_b[(hd * (nope_u + v_u) + pn) * rkv_u + r];
                }
            }
            for j in 0..v_u {
                for r in 0..rkv_u {
                    v_b[hd * v_u * rkv_u + j * rkv_u + r] =
                        kv_b[(hd * (nope_u + v_u) + nope_u + j) * rkv_u + r];
                }
            }
        }
        w.tensor_f32(&p("attn_kv_b.weight"), &[rkv, nh * (nope + v)], &kv_b);
        w.tensor_f32(&p("attn_k_b.weight"), &[nope, rkv, nh], &k_b);
        w.tensor_f32(&p("attn_v_b.weight"), &[rkv, v, nh], &v_b);
        w.tensor_f32(&p("attn_output.weight"), &[nh * v, h], &ws(&mut rng, nh * v, h));
        w.tensor_f32(&p("ffn_norm.weight"), &[h], &nw(&mut rng, h));

        // ---- DSA indexer: FULL layers only (layer 0 here); shared layers ship NO indexer
        // tensors (the GLM-5.2 property that broke early llama.cpp loaders). MTP has none.
        if il == 0 {
            w.tensor_f32(&p("indexer.attn_q_b.weight"), &[rq, d.idx_heads * d.idx_dim],
                         &ws(&mut rng, rq, d.idx_heads * d.idx_dim));
            w.tensor_f32(&p("indexer.attn_k.weight"), &[h, d.idx_dim], &ws(&mut rng, h, d.idx_dim));
            w.tensor_f32(&p("indexer.k_norm.weight"), &[d.idx_dim], &nw(&mut rng, d.idx_dim));
            w.tensor_f32(&p("indexer.k_norm.bias"), &[d.idx_dim],
                         &rng.fill(d.idx_dim as usize, 0.01));
            w.tensor_f32(&p("indexer.proj.weight"), &[h, d.idx_heads],
                         &ws(&mut rng, h, d.idx_heads));
        }

        // ---- FFN: leading_dense_block_count=1 -> layer 0 dense, layers 1+ (incl MTP) MoE ----
        if il == 0 {
            w.tensor_f32(&p("ffn_gate.weight"), &[h, d.n_ff], &ws(&mut rng, h, d.n_ff));
            w.tensor_f32(&p("ffn_up.weight"), &[h, d.n_ff], &ws(&mut rng, h, d.n_ff));
            w.tensor_f32(&p("ffn_down.weight"), &[d.n_ff, h], &ws(&mut rng, d.n_ff, h));
        } else {
            w.tensor_f32(&p("ffn_gate_inp.weight"), &[h, d.n_expert], &ws(&mut rng, h, d.n_expert));
            w.tensor_f32(&p("exp_probs_b.bias"), &[d.n_expert],
                         &rng.fill(d.n_expert as usize, 0.1));
            let q8 = |rng: &mut Rng, in_f: u64, out: u64| {
                crate::nvfp4_repack::f32_to_q8_0(&ws(rng, in_f, out * d.n_expert))
            };
            w.tensor_raw(&p("ffn_gate_exps.weight"), &[h, d.moe_ff, d.n_expert], GgmlType::Q8_0,
                         q8(&mut rng, h, d.moe_ff));
            w.tensor_raw(&p("ffn_up_exps.weight"), &[h, d.moe_ff, d.n_expert], GgmlType::Q8_0,
                         q8(&mut rng, h, d.moe_ff));
            w.tensor_raw(&p("ffn_down_exps.weight"), &[d.moe_ff, d.n_embd, d.n_expert],
                         GgmlType::Q8_0, q8(&mut rng, d.moe_ff, d.n_embd));
            w.tensor_f32(&p("ffn_gate_shexp.weight"), &[h, d.moe_ff], &ws(&mut rng, h, d.moe_ff));
            w.tensor_f32(&p("ffn_up_shexp.weight"), &[h, d.moe_ff], &ws(&mut rng, h, d.moe_ff));
            w.tensor_f32(&p("ffn_down_shexp.weight"), &[d.moe_ff, h], &ws(&mut rng, d.moe_ff, h));
        }

        // ---- NextN/MTP glue (the block the trunk loop drops) ----
        // Set matches the REAL unsloth GLM-5.2 artifact (header audit 2026-08-01): eh_proj +
        // enorm + hnorm + shared_head_norm. NO nextn.embed_tokens / nextn.shared_head_head
        // (both TENSOR_NOT_REQUIRED in llama.cpp; the artifact reuses token_embd / output).
        if il == d.n_trunk {
            w.tensor_f32(&p("nextn.eh_proj.weight"), &[2 * h, h], &ws(&mut rng, 2 * h, h));
            w.tensor_f32(&p("nextn.enorm.weight"), &[h], &nw(&mut rng, h));
            w.tensor_f32(&p("nextn.hnorm.weight"), &[h], &nw(&mut rng, h));
            w.tensor_f32(&p("nextn.shared_head_norm.weight"), &[h], &nw(&mut rng, h));
        }
    }

    w.write(path)?;
    Ok(d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Arch, ModelConfig};
    use crate::GgufFile;

    fn tmp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("memra-{name}-{}.gguf", std::process::id()))
    }

    /// Parse-arm gate: every MlaConfig field against the pinned GLM-5.2 values (RECEIPTS §2/§5).
    #[test]
    fn parse_glm52_pinned_metadata() {
        let p = tmp("glm52-meta");
        write_glm52_meta_only(&p).unwrap();
        let g = GgufFile::open(&p).unwrap();
        let c = ModelConfig::from_gguf(&g);
        std::fs::remove_file(&p).ok();

        assert_eq!(c.arch, Arch::GlmDsa);
        assert!(c.arch.is_hybrid() && c.arch.is_moe());
        assert_eq!(c.n_layer, 79, "block_count includes the NextN layer");
        assert_eq!(c.nextn_predict_layers, 1);
        assert_eq!(c.n_layer - c.nextn_predict_layers, 78, "78 trunk layers");
        assert_eq!(c.n_embd, 6144);
        assert_eq!(c.n_head, 64);
        assert_eq!(c.head_dim_k, 576, "attention.key_length = latent cache row");
        assert_eq!(c.head_dim_v, 512, "attention.value_length = latent V view");
        assert_eq!(c.n_vocab, 154_880);
        assert_eq!(c.context_length, 1_048_576);
        assert!((c.rope_freq_base - 8e6).abs() < 1.0);
        assert_eq!(c.rope_dim_count, 64);
        assert!((c.rms_eps - 1e-5).abs() < 1e-12);
        assert!(!c.attn_out_gate(), "MLA wq_b carries no fused [q|gate] output gate");

        let mla = c.mla.as_ref().expect("glm-dsa parses MlaConfig");
        assert_eq!(mla.q_lora_rank, 2048);
        assert_eq!(mla.kv_lora_rank, 512);
        assert_eq!(mla.qk_head_dim, 256);
        assert_eq!(mla.qk_nope_head_dim, 192);
        assert_eq!(mla.qk_rope_head_dim, 64);
        assert_eq!(mla.v_head_dim, 256);
        assert_eq!(mla.latent_dim(), 576);
        assert_eq!(mla.v_view_dim(), 512);
        assert!((mla.scale() - 0.0625).abs() < 1e-9, "1/sqrt(256), NOT 1/sqrt(576)");
        assert!(mla.sigmoid_routing);
        assert!((mla.routed_scaling_factor - 2.5).abs() < 1e-6);
        assert!(mla.route_norm);
        assert_eq!(mla.n_shared_experts, 1);
        assert_eq!(mla.first_k_dense_replace, 3);
        assert_eq!(c.sigmoid_router(), Some((2.5, true)));

        let moe = c.moe.as_ref().expect("glm-dsa is MoE");
        assert_eq!(moe.expert_count, 256);
        assert_eq!(moe.expert_used_count, 8);
        assert_eq!(moe.expert_ff_length, 2048);

        let dsa = mla.dsa.as_ref().expect("GLM-5.2 GGUF carries the indexer keys");
        assert_eq!(dsa.index_n_heads, 32);
        assert_eq!(dsa.index_head_dim, 128);
        assert_eq!(dsa.index_top_k, 2048);
        assert_eq!(dsa.indexer_full.len(), 78, "indexer.types spans the TRUNK layers");
        assert_eq!(dsa.indexer_full.iter().filter(|&&b| b).count(), 21, "21 full-indexer layers");
        // full at 0,1,2 then every 4th from 6 to 74 (pinned indexer_types pattern)
        for (i, &full) in dsa.indexer_full.iter().enumerate() {
            let want = i < 3 || (i >= 6 && (i - 6) % 4 == 0);
            assert_eq!(full, want, "indexer type at layer {i}");
        }
    }

    /// Parse-arm gate for `step35` (StepFun Step-3.7-Flash), every value pinned to the REAL
    /// official IQ4_XS header receipt
    /// `research/step37-bringup-20260802/raw/gguf-header-stepfun-iq4xs-shard1-20260802.txt`.
    /// This arch is memra's first with a PER-LAYER query-head count, so the scalar-only readers
    /// are the thing under test as much as the values are.
    #[test]
    fn parse_step35_pinned_metadata() {
        let p = tmp("step35-meta");
        write_step35_meta_only(&p).unwrap();
        let g = GgufFile::open(&p).unwrap();
        let c = ModelConfig::from_gguf(&g);
        std::fs::remove_file(&p).ok();

        assert_eq!(c.arch, Arch::Step35);
        assert!(c.arch.is_step35());
        assert!(c.arch.is_moe(), "288 experts top-8");
        assert!(c.arch.is_hybrid(), "SWA/full mix routes through the hybrid layer loop");
        assert_eq!(c.n_layer, 45, "trunk GGUF: 3 dense + 42 MoE, MTP ships separately");
        assert_eq!(c.nextn_predict_layers, 0, "the TRUNK file carries no nextn_predict_layers");
        assert_eq!(c.n_layer_total, 45);
        assert_eq!(c.n_embd, 4096);
        assert_eq!(c.head_dim_k, 128);
        assert_eq!(c.head_dim_v, 128);
        assert_eq!(c.n_ff, 11264, "dense-layer FFN width (blocks 0-2)");
        assert_eq!(c.context_length, 262_144);
        assert!((c.rms_eps - 1e-5).abs() < 1e-9);
        assert_eq!(c.n_vocab, 0, "no vocab_size key; real loads read token_embd's last dim");

        // --- the per-layer-scalar contract: the global n_head is the MAX, not the first value ---
        assert_eq!(c.n_head, 96, "global scalar sizes shared buffers: max(64,96), not 64");
        assert_eq!(c.n_head_kv, 8, "uniform KV heads");

        let s = c.step35.as_ref().expect("step35 parses Step35Config");
        assert_eq!(s.head_count.len(), 45);
        assert_eq!(s.swa_pattern.len(), 45);
        assert_eq!(s.sliding_window, 512);
        assert!((s.rope_base_global - 5e6).abs() < 1.0);
        assert!((s.rope_base_swa - 1e4).abs() < 1e-3);
        // Upstream ordering: the generic loader seeds n_rot_swa from n_rot_full (= key_length,
        // 128) BEFORE step35.cpp halves n_rot_full. SWA keeps 128; full attention gets 64.
        assert_eq!(s.rope_dims_swa, 128);
        assert_eq!(s.rope_dims_full, 64);
        assert_eq!(s.n_full_attn(45), 12, "3:1 over 45 blocks = 12 full + 33 SWA");

        for il in 0..45u32 {
            let full = il % 4 == 0;
            assert_eq!(s.is_swa(il), !full, "swa flag at layer {il}");
            assert_eq!(c.is_swa_at(il), !full, "ModelConfig::is_swa_at at layer {il}");
            assert_eq!(s.n_head(il), if full { 64 } else { 96 }, "n_head at layer {il}");
            assert_eq!(c.n_head_at(il), s.n_head(il), "n_head_at at layer {il}");
            assert_eq!(c.n_head_kv_at(il), 8, "n_head_kv_at at layer {il}");
            // half rotary on the FULL layers only
            assert_eq!(s.n_rot(il), if full { 64 } else { 128 }, "n_rot at layer {il}");
            assert!((s.rope_base(il) - if full { 5e6 } else { 1e4 }).abs() < 1.0,
                "rope base at layer {il}");
        }

        // --- SwiGLU clamp: unset everywhere but 43 (7.0) and 44 (16.0) ---
        for il in 0..43u32 {
            assert_eq!(s.clamp_exp(il), None, "clamp_exp at layer {il}");
            assert_eq!(s.clamp_shexp(il), None, "clamp_shexp at layer {il}");
        }
        assert_eq!(s.clamp_exp(43), Some(7.0));
        assert_eq!(s.clamp_exp(44), Some(16.0));
        assert_eq!(s.clamp_shexp(43), Some(7.0));
        assert_eq!(s.clamp_shexp(44), Some(16.0));

        // --- MoE: the DeepSeek-V3-class sigmoid router, verbatim ---
        let moe = c.moe.as_ref().expect("step35 is MoE");
        assert_eq!(moe.expert_count, 288);
        assert_eq!(moe.expert_used_count, 8);
        assert_eq!(moe.expert_ff_length, 1280);
        assert_eq!(moe.expert_shared_ff_length, 1280, "1 shared expert of the same width");
        assert!(s.sigmoid_routing, "expert_gating_func == 2");
        assert!((s.routed_scaling_factor - 3.0).abs() < 1e-6);
        assert!(s.route_norm);
        assert_eq!(s.first_k_dense_replace, 3);
        assert_eq!(c.sigmoid_router(), Some((3.0, true)));

        // --- the gate predicates: separate head-wise tensor, NOT the qwen35 fused-in-wq form ---
        assert!(!c.attn_out_gate(),
            "step35 wq carries NO fused [q|gate]; a true here mis-splits wq 2x out of bounds");
        assert!(c.attn_gate_separate(), "blk.N.attn_gate.weight [n_embd, n_head_l] is a tensor");
    }

    /// The REAL artifact shape: the 2026-06 unsloth GLM-5.2 GGUF ships WITHOUT
    /// `attention.indexer.types` (header audit, ARTIFACT.md). llama.cpp BC then supplies the
    /// hardcoded 21-full/57-shared table for 1M-ctx models — the parse arm must do the same.
    #[test]
    fn parse_glm52_without_indexer_types_key() {
        // rebuild the pinned metadata minus the types key (writer is append-only: re-emit).
        let p = tmp("glm52-meta-notypes");
        {
            let mut w = GgufWriter::new();
            let a = |s: &str| format!("glm-dsa.{s}");
            w.kv("general.architecture", MetaW::Str("glm-dsa"));
            w.kv(&a("block_count"), MetaW::U32(79));
            w.kv(&a("context_length"), MetaW::U32(1_048_576));
            w.kv(&a("embedding_length"), MetaW::U32(6144));
            w.kv(&a("attention.head_count"), MetaW::U32(64));
            w.kv(&a("attention.key_length_mla"), MetaW::U32(256));
            w.kv(&a("attention.value_length_mla"), MetaW::U32(256));
            w.kv(&a("attention.q_lora_rank"), MetaW::U32(2048));
            w.kv(&a("attention.kv_lora_rank"), MetaW::U32(512));
            w.kv(&a("rope.dimension_count"), MetaW::U32(64));
            w.kv(&a("nextn_predict_layers"), MetaW::U32(1));
            w.kv(&a("attention.indexer.head_count"), MetaW::U32(32));
            w.kv(&a("attention.indexer.key_length"), MetaW::U32(128));
            w.kv(&a("attention.indexer.top_k"), MetaW::U32(2048));
            w.kv(&a("vocab_size"), MetaW::U32(154_880));
            w.write(&p).unwrap();
        }
        let g = GgufFile::open(&p).unwrap();
        let c = ModelConfig::from_gguf(&g);
        std::fs::remove_file(&p).ok();
        let dsa = c.mla.as_ref().unwrap().dsa.as_ref().unwrap();
        assert_eq!(dsa.indexer_full.len(), 78);
        assert_eq!(dsa.indexer_full.iter().filter(|&&b| b).count(), 21,
                   "absent types key on a 1M-ctx model => the llama.cpp default table");
        assert_eq!(dsa.indexer_full, glm52_indexer_types());
    }

    /// Micro fixture: parses to the same config shape, and every §3.1 tensor name/shape is
    /// present (the tensor-presence audit, fixture-scale twin of the on-box gguf-dump audit).
    #[test]
    fn micro_fixture_parse_and_tensor_audit() {
        let p = tmp("glm-dsa-micro");
        let d = write_glm_dsa_micro(&p, 0x9_0261_0802).unwrap();
        let g = GgufFile::open(&p).unwrap();
        let c = ModelConfig::from_gguf(&g);

        assert_eq!(c.arch, Arch::GlmDsa);
        assert_eq!(c.n_layer, 3);
        assert_eq!(c.nextn_predict_layers, 1);
        let mla = c.mla.as_ref().expect("micro fixture parses MlaConfig");
        assert_eq!(
            (mla.q_lora_rank as u64, mla.kv_lora_rank as u64, mla.qk_nope_head_dim as u64,
             mla.qk_rope_head_dim as u64, mla.v_head_dim as u64),
            (d.q_lora, d.kv_lora, d.d_nope, d.d_rope, d.d_v)
        );
        assert_eq!(mla.latent_dim() as u64, d.latent_dim());
        assert_eq!(mla.first_k_dense_replace, 1);
        let dsa = mla.dsa.as_ref().unwrap();
        assert_eq!(dsa.indexer_full, vec![true, false]);

        let shape = |name: &str| -> Vec<u64> {
            g.find(name).unwrap_or_else(|| panic!("missing tensor {name}")).ne.clone()
        };
        let (nh, nope, rope, v, rq, rkv, h) =
            (d.n_head, d.d_nope, d.d_rope, d.d_v, d.q_lora, d.kv_lora, d.n_embd);
        for il in 0..3u64 {
            let p = |s: &str| format!("blk.{il}.{s}");
            assert_eq!(shape(&p("attn_q_a.weight")), vec![h, rq]);
            assert_eq!(shape(&p("attn_q_a_norm.weight")), vec![rq]);
            assert_eq!(shape(&p("attn_q_b.weight")), vec![rq, nh * (nope + rope)]);
            assert_eq!(shape(&p("attn_kv_a_mqa.weight")), vec![h, rkv + rope]);
            assert_eq!(shape(&p("attn_kv_a_norm.weight")), vec![rkv]);
            assert_eq!(shape(&p("attn_kv_b.weight")), vec![rkv, nh * (nope + v)]);
            assert_eq!(shape(&p("attn_k_b.weight")), vec![nope, rkv, nh], "k_b TRANSPOSED 3D");
            assert_eq!(shape(&p("attn_v_b.weight")), vec![rkv, v, nh], "v_b 3D");
            assert_eq!(shape(&p("attn_output.weight")), vec![nh * v, h]);
        }
        // partial indexer: FULL layer 0 has the tensors, SHARED layer 1 and MTP must NOT.
        assert!(g.find("blk.0.indexer.attn_q_b.weight").is_some());
        assert_eq!(shape("blk.0.indexer.attn_q_b.weight"), vec![rq, d.idx_heads * d.idx_dim]);
        assert_eq!(shape("blk.0.indexer.attn_k.weight"), vec![h, d.idx_dim]);
        assert_eq!(shape("blk.0.indexer.proj.weight"), vec![h, d.idx_heads]);
        assert!(g.find("blk.1.indexer.attn_q_b.weight").is_none(), "shared layer: no indexer");
        assert!(g.find("blk.2.indexer.attn_q_b.weight").is_none(), "MTP layer: no indexer");
        // FFN split: layer 0 dense, layer 1 MoE
        assert!(g.find("blk.0.ffn_gate.weight").is_some());
        assert!(g.find("blk.0.ffn_gate_exps.weight").is_none());
        assert_eq!(shape("blk.1.ffn_gate_exps.weight"), vec![h, d.moe_ff, d.n_expert]);
        assert_eq!(shape("blk.1.exp_probs_b.bias"), vec![d.n_expert]);
        // MTP glue (the real artifact's exact set)
        assert_eq!(shape("blk.2.nextn.eh_proj.weight"), vec![2 * h, h]);
        assert!(g.find("blk.2.nextn.enorm.weight").is_some());
        assert!(g.find("blk.2.nextn.hnorm.weight").is_some());
        assert!(g.find("blk.2.nextn.shared_head_norm.weight").is_some());
        assert!(g.find("blk.2.nextn.shared_head_head.weight").is_none(),
                "artifact ships no shared_head_head — head falls back to output.weight");

        std::fs::remove_file(&p).ok();
    }

    /// The kv_b -> (k_b transposed, v_b) split convention, verified VALUE-level through the
    /// reader: decompressing per-head W_UK/W_UV from either representation must agree exactly.
    #[test]
    fn micro_fixture_kv_b_split_convention() {
        let p = tmp("glm-dsa-split");
        let d = write_glm_dsa_micro(&p, 0xC0FFEE).unwrap();
        let g = GgufFile::open(&p).unwrap();
        let f32s = |name: &str| -> Vec<f32> {
            let t = g.find(name).unwrap();
            g.tensor_data(t).chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
        };
        let (nh, nope, v, rkv) =
            (d.n_head as usize, d.d_nope as usize, d.d_v as usize, d.kv_lora as usize);
        for il in 0..3 {
            let kv_b = f32s(&format!("blk.{il}.attn_kv_b.weight"));
            let k_b = f32s(&format!("blk.{il}.attn_k_b.weight"));
            let v_b = f32s(&format!("blk.{il}.attn_v_b.weight"));
            for hd in 0..nh {
                for pn in 0..nope {
                    for r in 0..rkv {
                        assert_eq!(
                            k_b[hd * rkv * nope + r * nope + pn],
                            kv_b[(hd * (nope + v) + pn) * rkv + r],
                            "k_b transpose mismatch l{il} h{hd} p{pn} r{r}"
                        );
                    }
                }
                for j in 0..v {
                    for r in 0..rkv {
                        assert_eq!(
                            v_b[hd * v * rkv + j * rkv + r],
                            kv_b[(hd * (nope + v) + nope + j) * rkv + r],
                            "v_b slice mismatch l{il} h{hd} j{j} r{r}"
                        );
                    }
                }
            }
        }
        std::fs::remove_file(&p).ok();
    }

    /// Non-glm arches are untouched: mla stays None (zero-behavior-change guard).
    #[test]
    fn non_glm_arch_has_no_mla() {
        let json = r#"{"model_type":"qwen3","num_hidden_layers":2,"hidden_size":256,
            "num_attention_heads":8,"intermediate_size":512,"vocab_size":1000,
            "max_position_embeddings":2048}"#;
        let c = ModelConfig::from_hf(&crate::config::HfConfig::parse(json));
        assert!(c.mla.is_none());
        assert!(c.attn_out_gate());
    }
}
