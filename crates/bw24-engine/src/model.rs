//! Dense transformer model: loads GGUF weights to GPU (Stage-1: dequant→f32), runs the
//! shared full-attention + SwiGLU forward graph. Arch-agnostic via ModelConfig; this path is
//! exactly the dense-transformer graph (qwen3) and the full-attention layers of hybrids.

use std::collections::HashMap;
use cudarc::driver::CudaSlice;
use bw24_gguf::{GgufFile, GgmlType, dequant};
use bw24_gguf::config::ModelConfig;
use bw24_gguf::source::{TensorSource, GgufSource};
use crate::{Engine, QT_Q8_0, QT_Q2_K, QT_Q4_K, QT_Q6_K, QT_Q5_K, QT_Q3_K, QT_IQ4_XS, QT_IQ3_S, QT_NVFP4, QT_F32, QT_BF16};

/// A weight tensor resident on GPU. Quantized weights stay in GGUF block bytes (`Quant`);
/// small non-quant tensors (norms, sometimes embed/lm_head) are kept dequantized as f32 (`Float`).
/// This keeps VRAM ~= on-disk quant size (fixes the f32-on-load OOM).
pub enum GpuTensor {
    Quant {
        bytes: CudaSlice<u8>, qtype: i32, row_bytes: usize, ne: Vec<u64>, scale: f32,
        /// SPLIT-PLANE walk-order repack (A6, 2026-07-04): NVFP4 matmul weights are repacked at
        /// load into [quant plane out_f x in_f/64 x 32B][scale plane out_f x in_f/64 x 4B] — same
        /// bytes, same total size, but a lane's per-group weight read becomes ONE 16B-aligned
        /// LDG.128 + a dense 4B scale word instead of 5 scattered 4B LDGs at 36B stride (the "18B
        /// straggle"). Every consumer kernel has an `_rp` twin (bit-identical: pure byte
        /// permutation, same dot order). `rp=false` = original GGUF block layout (all other
        /// dtypes, MoE-staged expert bytes, BW24_RP=0 escape).
        rp: bool,
        /// CUTLASS NVFP4 prefill operand (repacked B + swizzled SFB), built ALONGSIDE `bytes` at load
        /// when BW24_FP4_CUTLASS is set. `bytes` stays raw GGUF so decode (MMVQ/dp4a) is untouched;
        /// prefill (m>=128) reads this. Only ever Some for NVFP4 weights under cfg(bw24_cutlass).
        #[cfg(bw24_cutlass)]
        cutlass: Option<CutlassWeight>,
        /// FP8-ACT PREFILL operand (BW24_PP_FP8=1, probe verdict 2026-07-08): the checkpoint's RAW
        /// e4m3 bytes + per-tensor f32 weight_scale, stashed ALONGSIDE the Q8_0 re-encode for the
        /// F8-E4M3-origin 2D projections (~1 B/w extra on those layers). `bytes` stays Q8_0 so
        /// decode (dp4a/MMVQ) is untouched; only the m>=16 prefill dispatch (cuBLASLt FP8 TN,
        /// fp8_ffi.rs) reads this. None unless the env is set at load (zero VRAM cost by default).
        fp8: Option<Fp8Weight>,
    },
    Float { data: CudaSlice<f32>, ne: Vec<u64> },
    /// BF16-RESIDENT full-precision matmul weight (BW24_FULL_PREC only). Holds the checkpoint's raw
    /// bf16 bytes (`u8`, little-endian u16 pairs) — 2 B/w vs the 4 B/w a `Float` f32 materialization
    /// would cost, so the 9B trunk stays ~18GB in VRAM instead of ~36GB. Consumed via dequant-on-use:
    /// each matmul expands this to a transient f32 scratch and rides the SAME cuBLASLt f32 GEMV the
    /// `Float` arm uses (bit-identical to a load-time bf16->f32 dequant, just deferred). Never a norm
    /// (norms stay `Float` f32); never on a fast/GEMM/MMQ path (uses_q8_1_fast/gemm_supports = false).
    FloatBf16 { data: CudaSlice<u8>, ne: Vec<u64> },
}

/// FP8-native prefill operand: raw checkpoint e4m3 codes `[out_f, in_f]` row-major (EXACT — the
/// weight side of the FP8 GEMM does no re-quantization) + the per-tensor `weight_scale` dequant
/// scalar, folded into the GEMM's scale pointer together with the per-batch activation scale.
pub struct Fp8Weight {
    pub bytes: CudaSlice<u8>,
    pub scale: f32,
}

/// Host-side split-plane repack of NVFP4 GGUF block bytes (A6). Input: out_f rows of in_f/64
/// 36-byte blocks ([4B UE4M3 scales][32B packed e2m1]). Output (same length): quant plane
/// (out_f x nsb64 x 32B) followed by scale plane (out_f x nsb64 x 4B). Pure byte permutation.
pub fn repack_nvfp4_split(bytes: &[u8], out_f: usize) -> Vec<u8> {
    let row_bytes = bytes.len() / out_f;
    let nsb64 = row_bytes / 36;
    debug_assert_eq!(row_bytes % 36, 0, "NVFP4 row_bytes must be a multiple of 36");
    let qplane = out_f * nsb64 * 32;
    let mut rp = vec![0u8; bytes.len()];
    for o in 0..out_f {
        for s in 0..nsb64 {
            let src = &bytes[o * row_bytes + s * 36..o * row_bytes + s * 36 + 36];
            rp[qplane + (o * nsb64 + s) * 4..qplane + (o * nsb64 + s) * 4 + 4]
                .copy_from_slice(&src[0..4]);
            rp[(o * nsb64 + s) * 32..(o * nsb64 + s) * 32 + 32].copy_from_slice(&src[4..36]);
        }
    }
    rp
}

/// Inverse of `repack_nvfp4_split` (the roundtrip gate).
pub fn unpack_nvfp4_split(rp: &[u8], out_f: usize) -> Vec<u8> {
    let row_bytes = rp.len() / out_f;
    let nsb64 = row_bytes / 36;
    let qplane = out_f * nsb64 * 32;
    let mut back = vec![0u8; rp.len()];
    for o in 0..out_f {
        for s in 0..nsb64 {
            back[o * row_bytes + s * 36..o * row_bytes + s * 36 + 4]
                .copy_from_slice(&rp[qplane + (o * nsb64 + s) * 4..qplane + (o * nsb64 + s) * 4 + 4]);
            back[o * row_bytes + s * 36 + 4..o * row_bytes + s * 36 + 36]
                .copy_from_slice(&rp[(o * nsb64 + s) * 32..(o * nsb64 + s) * 32 + 32]);
        }
    }
    back
}

/// A6 repack seam: default ON, `BW24_RP=0` restores the GGUF block layout everywhere (rollback/A-B).
pub fn rp_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("BW24_RP").map(|v| v != "0").unwrap_or(true))
}

/// FULL-PRECISION LOADER MODE (BW24_FULL_PREC=1, default OFF — MTP-heal research platform).
/// Bypasses the standing loader law (large BF16/F8 -> Q8_0/NVFP4 re-encode, the "Float-poison"
/// tripwire). Under this flag every weight loads as Float and compute rides the Stage-A f32 oracle
/// path end to end — SLOW IS FINE, this mode exists for exactness (the MTP acceptance CEILING at
/// full precision), not speed. Large 2D matmul weights stay bf16-resident (`GpuTensor::FloatBf16`)
/// with dequant-on-use so the 9B (~18GB bf16) + f32 activations fit 24GB instead of blowing to
/// ~38GB as an all-f32 materialization. The Float-poison tripwire warnings are CORRECT behavior
/// here and are suppressed. See docs/FLAGS.md and HANDOVER "BW24 DUAL-SHAPE".
pub fn full_prec_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("BW24_FULL_PREC").map(|v| v == "1").unwrap_or(false))
}

/// LOADER-LAW allowlist (loadersweep audit 2026-07-08): 2D Float tensors that are DELIBERATELY
/// Float despite being matmul-class. Every entry needs an audit rationale — this list silences
/// the tripwire below, so an unjustified entry re-opens the trap.
///   * ffn_gate_inp (MoE router, 35B GGUF F32 [2048,256] / M3 ST F32 [6144,64]): the router's
///     top-k SELECTION is discontinuous — quantizing shifts logits and flips expert choice (a
///     class change, not an FP-order change). llama.cpp keeps every router F32 (its converter
///     forces F32) so Float is bench-parity, it sits on NO all-or-nothing predicate, and the
///     decode-exact contract is already built around its cuBLASLt path
///     (hybrid_forward.rs moe_ffn_sequential_zq8 router comment).
fn float_2d_audited(name: &str) -> bool {
    name.ends_with("ffn_gate_inp.weight")
}

/// Once-per-name-pattern loader-law warning (`blk.{il}.` collapses to `blk.*.` so a 48-layer
/// offender prints ONE line, not 48). See the call site in `load_from_source` for the law.
fn warn_float_2d_once(name: &str, ne: &[u64], src_type: GgmlType) {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<std::collections::HashSet<String>>> = OnceLock::new();
    let pat = match name.strip_prefix("blk.").and_then(|r| r.split_once('.')) {
        Some((_, suffix)) => format!("blk.*.{suffix}"),
        None => name.to_string(),
    };
    let mut seen = SEEN.get_or_init(|| Mutex::new(std::collections::HashSet::new())).lock().unwrap();
    if seen.insert(pat.clone()) {
        eprintln!("[loader-law] WARNING: {pat} loads as 2D Float ne={ne:?} (src {src_type:?}) — \
                   a Float matmul weight rides cuBLAS f32 GEMV and poisons all-or-nothing q8-fast \
                   predicates (uses_q8_1_fast/mixer_in_q8_1_fast). If matmul-class: Q8_0-encode at \
                   load (model.rs ssm arm / source.rs BF16+F8 gates). If deliberately Float: add \
                   it to float_2d_audited with the audit rationale.");
    }
}

/// CUTLASS-layout NVFP4 weight (B operand) for the prefill FP4 GEMM. Built once at load from the raw
/// GGUF bytes (de-interleave + SFB swizzle). Coexists with the raw `bytes` (decode reads bytes).
#[cfg(bw24_cutlass)]
pub struct CutlassWeight {
    /// Plain K-contiguous packed e2m1, [out_f, in_f/2] bytes.
    pub b_packed: CudaSlice<u8>,
    /// Swizzled SFB (CUTLASS SfAtom layout), sized via cutlass_sfb_size(out_f, in_f).
    pub sfb_swizzled: CudaSlice<u8>,
}

impl GpuTensor {
    pub fn ne(&self) -> &[u64] { match self { GpuTensor::Quant { ne, .. } => ne, GpuTensor::Float { ne, .. } => ne, GpuTensor::FloatBf16 { ne, .. } => ne } }
    pub fn in_features(&self) -> usize { self.ne()[0] as usize }
    pub fn out_features(&self) -> usize { self.ne()[1] as usize }
    /// Per-tensor post-matmul macro-scale (NVFP4 carries scale != 1.0; all others -> 1.0, a no-op).
    /// Used by the fused SwiGLU epilogue to fold the gate/up scale into one kernel.
    pub fn scale(&self) -> f32 { match self { GpuTensor::Quant { scale, .. } => *scale, GpuTensor::Float { .. } | GpuTensor::FloatBf16 { .. } => 1.0 } }

    /// Load a tensor, keeping quant types packed and float types as f32. (GGUF entry point —
    /// thin wrapper over the source-agnostic `load_from_source`; behavior is unchanged.)
    pub fn load(e: &Engine, g: &GgufFile, name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        Self::load_from_source(e, &GgufSource(g), name)
    }

    /// Source-agnostic load: works from any `TensorSource` (GGUF or safetensors). The engine's
    /// forward graph only ever asks for ggml-style names; the source maps them to its own layout.
    pub fn load_from_source(e: &Engine, src: &dyn TensorSource, name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        // A1 DIRECT NVFP4 IMPORT (2026-07-04): a PLAIN modelopt/Reza NVFP4 weight from a
        // safetensors source repacks straight into the A6 split-plane resident layout in ONE host
        // pass (nvfp4_repack::repack_modelopt_to_split — the scale plane is the file's
        // weight_scale bytes verbatim), never materializing the GGUF 36B-block intermediate.
        // The GGUF hop remains only for BW24_ST_DIRECT=0 (rollback/A-B seam — byte-identical
        // resident weights either way), BW24_RP=0, the hybrid V-reorder transforms, and the
        // opt-in CUTLASS resident operand (which is built from raw GGUF-layout bytes).
        let cutlass_wants_raw = cfg!(bw24_cutlass) && std::env::var("BW24_FP4_CUTLASS").is_ok();
        let st_direct = std::env::var("BW24_ST_DIRECT").map(|v| v != "0").unwrap_or(true);
        if rp_enabled() && st_direct && !cutlass_wants_raw {
            if let Some(nv) = src.find_nvfp4_native(name) {
                if nv.in_f % 64 == 0 && nv.out_f > 0 {
                    // Same post-matmul macro-scale sibling lookup as the GGUF-layout arm below.
                    let stem = name.strip_suffix(".weight").unwrap_or(name);
                    let scale = match src.find(&format!("{stem}.scale")) {
                        Some(sv) => f32::from_le_bytes(sv.bytes[..4].try_into().unwrap()),
                        None => 1.0,
                    };
                    let bytes = e.htod_bytes(&bw24_gguf::nvfp4_repack::repack_modelopt_to_split(
                        nv.wbytes, nv.wscale, nv.out_f, nv.in_f))?;
                    return Ok(GpuTensor::Quant {
                        bytes, qtype: QT_NVFP4, row_bytes: nv.in_f / 64 * 36,
                        ne: vec![nv.in_f as u64, nv.out_f as u64], scale, rp: true,
                        #[cfg(bw24_cutlass)]
                        cutlass: None,
                        fp8: None,
                    });
                }
            }
        }
        // E4M3-DIRECT (BW24_ST_E4M3=1, lane e4m3dec 2026-07-08): F8-E4M3-origin 2D projections keep
        // the checkpoint's RAW e4m3 device bytes + per-tensor weight_scale as the ONE resident copy
        // (QT_F8_E4M3) instead of the Q8_0 re-encode — decode dequants e4m3 in-kernel
        // (qmatvec_e4m3_mmvq, the checkpoint's own precision, no lossy re-quant hop), prefill
        // (m>=16) rides the cuBLASLt FP8 GEMM on the SAME bytes (try_fp8_gemm). Frees the Q8_0
        // duplicate the BW24_PP_FP8 stash needed (~3.4GB on the NV-27B) — full FP8 prefill coverage
        // with no VRAM budget. Placed BEFORE `find` so the host-side F8->Q8_0 re-encode is skipped
        // entirely (faster load). in_f%32 is the q8_1 activation block gate (every F8 projection in
        // the NV-27B satisfies it; a violator falls through to the Q8_0 arm unchanged).
        if crate::fp8_ffi::st_e4m3_enabled() {
            if let Some(f8) = src.find_fp8_native(name) {
                if f8.in_f % 32 == 0 && f8.out_f > 0 {
                    return Ok(GpuTensor::Quant {
                        bytes: e.htod_bytes(&f8.bytes)?, qtype: crate::QT_F8_E4M3,
                        row_bytes: f8.in_f, ne: vec![f8.in_f as u64, f8.out_f as u64],
                        scale: f8.scale, rp: false,
                        #[cfg(bw24_cutlass)]
                        cutlass: None,
                        fp8: None,
                    });
                }
            }
        }
        let mut v = src.find(name).unwrap_or_else(|| panic!("missing tensor {name}"));
        // BW24_KQ_NVFP4=1 (opt-in, 2026-07-08): re-encode Q4_K/Q5_K 2D matmul weights to NVFP4 at
        // load. The k-quant mmvq family runs at 61-70% of the bandwidth wall on this rig (measured
        // BOTH engines — the kernels share ancestry) while the in-house NVFP4 path runs at 96%.
        // The daily GGUF's quant mix was chosen for llama's kernels, not ours: Q4_K -> NVFP4 is
        // 4-bit -> 4-bit at +26pp kernel efficiency; Q5_K -> NVFP4 also drops bytes (0.69 -> 0.56
        // B/w) at a small real re-quant cost (5 -> 4 bit; gates + acceptance arbitrate). Q6_K/Q8_0
        // excluded (6/8-bit -> 4-bit is a real quality cliff — the lm_head stays untouched).
        // BW24_KQ_NVFP4 (opt-in SPEED-OVER-QUALITY mode, measured 2026-07-08 on the 9B):
        // =2 (Q4_K+Q5_K -> NVFP4): +3.9% plain decode (129.5 -> 134.5, the Q5 bytes win),
        //    acceptance tax ~3pts on hard content (p2 74.0 -> 70.7, p3 66.9 -> 64.9).
        // =1 (Q4_K only): NO perf gain AND still ~3pts tax — Q4_K is ASYMMETRIC (6-bit
        //    scale+min per 32); NVFP4 is symmetric e2m1: dropping the zero-point is real
        //    error even 4-bit -> 4-bit. The "same bpw = same class" assumption is FALSE
        //    across asymmetric/symmetric formats. Kept only for the record.
        let kq = std::env::var("BW24_KQ_NVFP4").ok().and_then(|x| x.parse::<u8>().ok()).unwrap_or(0);
        if (kq >= 1 && v.ggml_type == GgmlType::Q4_K
                || kq >= 2 && v.ggml_type == GgmlType::Q5_K)
            && v.ne.len() == 2 && v.ne[0] % 64 == 0 && !name.starts_with("output")
        {
            let n: u64 = v.ne.iter().product();
            let f32v = dequant::dequantize(v.ggml_type, &v.bytes, n as usize);
            let packed = bw24_gguf::nvfp4_repack::f32_to_nvfp4(&f32v);
            v = bw24_gguf::source::TensorView {
                bytes: std::borrow::Cow::Owned(packed), ggml_type: GgmlType::NVFP4, ne: v.ne.clone(),
            };
        }
        let qtype = match v.ggml_type {
            GgmlType::Q8_0 => Some(QT_Q8_0),
            GgmlType::Q4_K => Some(QT_Q4_K),
            GgmlType::Q6_K => Some(QT_Q6_K),
            GgmlType::Q5_K => Some(QT_Q5_K),
            GgmlType::Q3_K => Some(QT_Q3_K),
            GgmlType::IQ4_XS => Some(QT_IQ4_XS),
            GgmlType::IQ3_S => Some(QT_IQ3_S),
            GgmlType::NVFP4 => Some(QT_NVFP4),
            // F32/F16/BF16 (the dtypes safetensors carries) -> Float path below.
            _ => None,
        };
        match qtype {
            Some(qt) => {
                let out_f = v.ne[1] as usize;
                let row_bytes = v.bytes.len() / out_f;
                // NVFP4 two-level scale: per-16 ue4m3 micro-scale is in the dequant; the per-tensor
                // F32 macro-scale lives in a sibling "<stem>.scale" tensor, applied POST-matmul
                // (llama build_lora_mm: ggml_mul(res, w_s)). ".input_scale" is the W4A4 activation
                // scale — UNUSED on our W4A16/f32 path. Only NVFP4 carries it; others -> 1.0 (no-op).
                let scale = if qt == QT_NVFP4 {
                    let stem = name.strip_suffix(".weight").unwrap_or(name);
                    match src.find(&format!("{stem}.scale")) {
                        Some(sv) => f32::from_le_bytes(sv.bytes[..4].try_into().unwrap()),
                        None => 1.0,
                    }
                } else { 1.0 };
                // A6 SPLIT-PLANE repack: NVFP4 2-D matmul weights upload in walk-order layout
                // (host-side permutation before htod — zero VRAM spike, layer-streamed by
                // construction). Every consumer kernel dispatches its `_rp` twin off the flag.
                let rp = qt == QT_NVFP4 && v.ne.len() == 2 && (v.ne[0] as usize) % 64 == 0
                    && v.bytes.len() % out_f == 0 && (v.bytes.len() / out_f) % 36 == 0
                    && rp_enabled();
                let bytes = if rp {
                    e.htod_bytes(&repack_nvfp4_split(&v.bytes, out_f))?
                } else {
                    e.htod_bytes(&v.bytes)?
                };
                // CUTLASS NVFP4 prefill operand, built from the RAW GGUF bytes (a temp raw upload
                // when the resident `bytes` are repacked). Gated: only NVFP4 weights, only when
                // BW24_FP4_CUTLASS is set, only under cfg(bw24_cutlass). in_f%64==0 is the NVFP4
                // K-block constraint (same as the dispatch).
                #[cfg(bw24_cutlass)]
                let cutlass = {
                    let in_f = v.ne[0] as usize;
                    // Skip the resident repack when OTF is requested (per-call repack instead) — the
                    // resident path ~doubles NVFP4 weight VRAM and OOMs larger models (e.g. 27B/24GB).
                    if qt == QT_NVFP4 && in_f % 64 == 0 && v.ne.len() == 2
                        && std::env::var("BW24_FP4_CUTLASS").is_ok()
                        && std::env::var("BW24_FP4_CUTLASS_OTF").is_err()
                    {
                        let raw_dev;
                        let src_dev = if rp { raw_dev = e.htod_bytes(&v.bytes)?; &raw_dev }
                                      else { &bytes };
                        let (b_packed, sfb_swizzled) =
                            e.build_cutlass_weight(src_dev, out_f, in_f, row_bytes)?;
                        Some(CutlassWeight { b_packed, sfb_swizzled })
                    } else { None }
                };
                // FP8-ACT PREFILL operand (BW24_PP_FP8=1): for F8-E4M3-sourced projections (they
                // surface as Q8_0 from the source's re-encode) ALSO stash the raw e4m3 device
                // bytes + weight_scale. The source guarantees byte order matches `v` (the
                // Transform arm's V-reorder is baked into both); the ne check guards a mixup.
                // VRAM BUDGET (24GB rigs, 2026-07-08): the stash duplicates every F8-origin
                // projection (~+3.4GB on the 27B) — fine on the 96GB box, OOM here. The stash
                // spends from BW24_PP_FP8_BUDGET_MB (default 1536); once spent, remaining
                // tensors ride the old path. Load order is layer order, so the budget covers a
                // PREFIX of layers — coverage (and the prefill win) scales with the budget.
                let fp8 = if qt == QT_Q8_0 && crate::fp8_ffi::pp_fp8_enabled() {
                    match src.find_fp8_native(name) {
                        Some(f8) if v.ne.len() == 2
                            && f8.in_f as u64 == v.ne[0] && f8.out_f as u64 == v.ne[1] => {
                            use std::sync::atomic::{AtomicUsize, Ordering};
                            static FP8_SPENT: AtomicUsize = AtomicUsize::new(0);
                            static FP8_BUDGET: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
                            let budget = *FP8_BUDGET.get_or_init(|| {
                                std::env::var("BW24_PP_FP8_BUDGET_MB").ok()
                                    .and_then(|v| v.parse::<usize>().ok()).unwrap_or(1536) << 20
                            });
                            let sz = f8.bytes.len();
                            if FP8_SPENT.fetch_add(sz, Ordering::Relaxed) + sz <= budget {
                                Some(Fp8Weight { bytes: e.htod_bytes(&f8.bytes)?, scale: f8.scale })
                            } else {
                                FP8_SPENT.fetch_sub(sz, Ordering::Relaxed);
                                None
                            }
                        }
                        _ => None,
                    }
                } else { None };
                Ok(GpuTensor::Quant {
                    bytes, qtype: qt, row_bytes, ne: v.ne.clone(), scale, rp,
                    #[cfg(bw24_cutlass)]
                    cutlass,
                    fp8,
                })
            }
            None => {
                let n: u64 = v.ne.iter().product();
                // FULL-PRECISION MODE (BW24_FULL_PREC): NO re-encodes. Large 2D bf16 matmul weights
                // stay bf16-resident (FloatBf16, dequant-on-use) so the trunk fits VRAM; everything
                // else (small 2D, 1D norms, F16/F32) rides the exact f32 Float path below. The ssm
                // Q8_0 re-encode and the Float-poison tripwire are BYPASSED here (both are the loader
                // law this mode exists to suspend — the warnings would be correct but noise).
                if full_prec_enabled() {
                    // Only bf16 sources take the resident-bf16 arm; F16/F32 fall through to f32 Float
                    // (exact, and tiny/absent in the bf16 ST checkpoints this mode targets). The 1M
                    // threshold keeps small tensors (norms, gate_inp) on the proven f32 path — only
                    // the big trunk matrices need the 2 B/w VRAM saving.
                    if v.ggml_type == GgmlType::BF16 && v.ne.len() == 2 && n >= 1_000_000 {
                        let data = e.htod_bytes(&v.bytes)?; // raw bf16 bytes, u16 LE pairs
                        return Ok(GpuTensor::FloatBf16 { data, ne: v.ne.clone() });
                    }
                    let f32v = dequant::dequantize(v.ggml_type, &v.bytes, n as usize);
                    return Ok(GpuTensor::Float { data: e.htod(&f32v)?, ne: v.ne.clone() });
                }
                let f32v = dequant::dequantize(v.ggml_type, &v.bytes, n as usize);
                // ssm_beta/ssm_alpha stored F32 (the 35B GGUF): Q8_0-encode at load. F32 here
                // fails `mixer_in_q8_1_fast` for the whole linear-attn mixer -> every linear
                // layer falls off the fused norm+quantize chain onto cuBLAS f32 GEMV pairs
                // (the NV-27B in_proj_a/b lesson, same all-or-nothing capability check; nsys
                // 35B: 100 dot+reduce launches/token). Q8_0 of an F32 source is the same
                // class-lossless step every 9B GGUF already ships for these tensors.
                if v.ne.len() == 2 && v.ne[0] % 32 == 0
                    && (name.ends_with("ssm_beta.weight") || name.ends_with("ssm_alpha.weight")) {
                    let q8 = bw24_gguf::nvfp4_repack::f32_to_q8_0(&f32v);
                    return GpuTensor::from_quant_bytes(e, &q8, GgmlType::Q8_0, v.ne[0], v.ne[1], 1.0);
                }
                // LOADER-LAW TRIPWIRE (loadersweep 2026-07-08): a 2D Float tensor with both dims
                // >= 16 is almost certainly MATMUL-class, and a Float matmul weight (a) rides
                // cuBLAS f32 GEMV pairs (dot_kernel + reduce_1Block in nsys) and (b) fails
                // uses_q8_1_fast, poisoning every ALL-OR-NOTHING fast-path predicate it sits on
                // (mixer_in_q8_1_fast etc.) — the trap that cost measurable perf 4 times (NV-27B
                // in_proj_a/b BF16, 35B ssm_beta/alpha F32, M3 shexp cousin, M3 BF16 lm_head).
                // Fix recipe: name-gated f32_to_q8_0 encode at load (see the ssm arm above /
                // source.rs BF16+F8 gates). Norm-class tensors are 1D or have a dim < 16
                // (conv1d ne[0]=4) and never reach this warning.
                if v.ne.len() == 2 && v.ne[0] >= 16 && v.ne[1] >= 16 && !float_2d_audited(name) {
                    warn_float_2d_once(name, &v.ne, v.ggml_type);
                }
                // F32/F16/BF16 (or as-yet-unhandled quant): dequant to f32. Small tensors only.
                Ok(GpuTensor::Float { data: e.htod(&f32v)?, ne: v.ne.clone() })
            }
        }
    }

    /// Build a Quant tensor directly from raw ggml block bytes (FR-Spec self-trim: byte-level row
    /// gather from an already-loaded weight — rows in every ggml quant are independent, so a
    /// contiguous per-row byte copy is a lossless "trim"). `ne0` = in_features, `ne1` = rows.
    pub fn from_quant_bytes(e: &Engine, bytes: &[u8], ty: GgmlType, ne0: u64, ne1: u64, scale: f32)
                            -> Result<Self, Box<dyn std::error::Error>> {
        let qt = match ty {
            GgmlType::Q8_0 => QT_Q8_0, GgmlType::Q4_K => QT_Q4_K, GgmlType::Q6_K => QT_Q6_K,
            GgmlType::Q5_K => QT_Q5_K, GgmlType::Q3_K => QT_Q3_K, GgmlType::IQ4_XS => QT_IQ4_XS,
            GgmlType::IQ3_S => QT_IQ3_S, GgmlType::NVFP4 => QT_NVFP4,
            other => panic!("from_quant_bytes: unsupported dtype {other:?}"),
        };
        let row_bytes = bytes.len() / ne1 as usize;
        // Same A6 repack as load_from_source: callers pass GGUF-layout host bytes (the FR-Spec
        // self-trim row-gathers from the source file bytes, which are always original layout).
        let rp = qt == QT_NVFP4 && ne0 % 64 == 0 && row_bytes % 36 == 0 && rp_enabled();
        let dev = if rp { e.htod_bytes(&repack_nvfp4_split(bytes, ne1 as usize))? }
                  else { e.htod_bytes(bytes)? };
        Ok(GpuTensor::Quant {
            bytes: dev, qtype: qt, row_bytes, ne: vec![ne0, ne1], scale, rp,
            #[cfg(bw24_cutlass)]
            cutlass: None,
            fp8: None,
        })
    }

    pub fn load_opt(e: &Engine, g: &GgufFile, name: &str) -> Result<Option<Self>, Box<dyn std::error::Error>> {
        Self::load_opt_from_source(e, &GgufSource(g), name)
    }

    pub fn load_opt_from_source(e: &Engine, src: &dyn TensorSource, name: &str) -> Result<Option<Self>, Box<dyn std::error::Error>> {
        if src.has(name) { Ok(Some(Self::load_from_source(e, src, name)?)) } else { Ok(None) }
    }

    /// Accessor for tensors that MUST be f32 (norm weights). Panics if quantized.
    pub fn float_data(&self) -> &CudaSlice<f32> {
        match self { GpuTensor::Float { data, .. } => data,
            GpuTensor::Quant { .. } => panic!("expected float tensor (norm), got quantized"),
            GpuTensor::FloatBf16 { .. } => panic!("expected f32 float tensor (norm), got bf16-resident matmul weight") }
    }
}

pub struct Layer {
    pub attn_norm: GpuTensor,
    pub wq: GpuTensor, pub wk: GpuTensor, pub wv: GpuTensor, pub wo: GpuTensor,
    pub q_norm: Option<GpuTensor>, pub k_norm: Option<GpuTensor>,
    pub ffn_norm: GpuTensor,
    /// FFN: dense SwiGLU or routed MoE (OLMoE — dense attention + MoE FFN). Reuses the hybrid
    /// `Ffn` enum + `load_ffn` so the routed-expert forward is shared with `HybridModel::moe_ffn`.
    pub ffn: crate::hybrid::Ffn,
}

/// Host-resident embedding table for row gather (dequant only the needed token rows).
pub struct EmbedHost {
    pub raw: Vec<u8>,
    pub ggml_type: GgmlType,
    pub n_embd: usize,
}
impl EmbedHost {
    pub fn from_gguf(g: &GgufFile, name: &str) -> Self {
        Self::from_source(&GgufSource(g), name)
    }
    pub fn from_source(src: &dyn TensorSource, name: &str) -> Self {
        let v = src.find(name).unwrap_or_else(|| panic!("missing embed {name}"));
        EmbedHost { raw: v.bytes.to_vec(), ggml_type: v.ggml_type, n_embd: v.ne[0] as usize }
    }
    /// QT int + row_bytes for this embed table's dtype (for the device embed-gather kernel).
    /// CUDA-GRAPH-PLAN Phase 1. Mirrors the GpuTensor qtype mapping.
    pub fn qt_and_row_bytes(&self, n_embd: usize) -> (i32, usize) {
        let (blk, tsize) = self.ggml_type.block_and_type_size();
        let row_bytes = (n_embd as u64 / blk * tsize) as usize;
        let qt = match self.ggml_type {
            GgmlType::Q8_0 => QT_Q8_0, GgmlType::Q4_K => QT_Q4_K, GgmlType::Q6_K => QT_Q6_K,
            GgmlType::Q5_K => QT_Q5_K, GgmlType::Q3_K => QT_Q3_K, GgmlType::IQ4_XS => QT_IQ4_XS,
            GgmlType::IQ3_S => QT_IQ3_S, GgmlType::NVFP4 => QT_NVFP4, GgmlType::F32 => QT_F32,
            // BF16 embed table (FULL_PREC research mode: qwen35-9b-hf) — device gather does the
            // exact bits<<16 expansion; 2 B/elem resident instead of an f32-doubled table.
            GgmlType::BF16 => QT_BF16,
            other => panic!("embed_gather: unsupported dtype {other:?}"),
        };
        (qt, row_bytes)
    }

    /// Gather rows for tokens -> [T, n_embd] f32. Dequant per-row from raw bytes.
    pub fn gather(&self, n_embd: usize, tokens: &[u32]) -> Vec<f32> {
        let (blk, tsize) = self.ggml_type.block_and_type_size();
        let row_bytes = (n_embd as u64 / blk * tsize) as usize;
        let mut x = vec![0f32; tokens.len() * n_embd];
        for (ti, &tok) in tokens.iter().enumerate() {
            let off = tok as usize * row_bytes;
            let row = dequant::dequantize(self.ggml_type, &self.raw[off..off + row_bytes], n_embd);
            x[ti * n_embd..ti * n_embd + n_embd].copy_from_slice(&row);
        }
        x
    }
}

pub struct Model {
    pub cfg: ModelConfig,
    pub embd: EmbedHost,
    pub output_norm: GpuTensor,
    pub output: GpuTensor,
    pub layers: Vec<Layer>,
}

impl Model {
    /// Load a dense (vanilla-transformer) model from GGUF. Thin wrapper over
    /// `load_dense_from_source`. Panics if the arch has SSM/MoE layers.
    pub fn load_dense(e: &Engine, g: &GgufFile) -> Result<Self, Box<dyn std::error::Error>> {
        Self::load_dense_from_source(e, &GgufSource(g))
    }

    /// Load a dense-attention model from any `TensorSource` — GGUF or a safetensors HF checkpoint.
    /// The whole loop speaks ggml names; the source maps them. The FFN is dense SwiGLU OR routed MoE
    /// (OLMoE: dense full-attention + MoE FFN). Panics on hybrid (SSM) arches — use the hybrid path.
    pub fn load_dense_from_source(e: &Engine, src: &dyn TensorSource) -> Result<Self, Box<dyn std::error::Error>> {
        let cfg = src.config();
        assert!(cfg.full_attention_interval == 0, "model has linear-attn layers; use hybrid path");

        let embd = EmbedHost::from_source(src, "token_embd.weight");
        let output_norm = GpuTensor::load_from_source(e, src, "output_norm.weight")?;
        // tied embeddings: fall back to tok_embd if output.weight absent (OLMoE has untied output).
        let output = if src.has("output.weight") {
            GpuTensor::load_from_source(e, src, "output.weight")?
        } else {
            GpuTensor::load_from_source(e, src, "token_embd.weight")?
        };

        let mut layers = Vec::with_capacity(cfg.n_layer as usize);
        for il in 0..cfg.n_layer {
            let p = |s: &str| format!("blk.{il}.{s}");
            let hy3_dense_ffn = cfg.hy3.as_ref()
                .is_some_and(|h| il < h.first_k_dense_replace);
            let ffn = if hy3_dense_ffn {
                crate::hybrid::Ffn::Dense {
                    ffn_gate: GpuTensor::load_from_source(e, src, &p("ffn_gate.weight"))?,
                    ffn_up: GpuTensor::load_from_source(e, src, &p("ffn_up.weight"))?,
                    ffn_down: GpuTensor::load_from_source(e, src, &p("ffn_down.weight"))?,
                }
            } else {
                crate::hybrid::load_ffn(e, src, &cfg, il, None)?
            };
            layers.push(Layer {
                attn_norm: GpuTensor::load_from_source(e, src, &p("attn_norm.weight"))?,
                wq: GpuTensor::load_from_source(e, src, &p("attn_q.weight"))?,
                wk: GpuTensor::load_from_source(e, src, &p("attn_k.weight"))?,
                wv: GpuTensor::load_from_source(e, src, &p("attn_v.weight"))?,
                wo: GpuTensor::load_from_source(e, src, &p("attn_output.weight"))?,
                q_norm: GpuTensor::load_opt_from_source(e, src, &p("attn_q_norm.weight"))?,
                k_norm: GpuTensor::load_opt_from_source(e, src, &p("attn_k_norm.weight"))?,
                ffn_norm: GpuTensor::load_from_source(e, src, &p("ffn_norm.weight"))?,
                ffn,
            });
        }
        Ok(Model { cfg, embd, output_norm, output, layers })
    }

    /// Largest expert block (bytes) across all MoE layers — the fixed cache-slot size (mirrors
    /// `HybridModel::max_moe_block`). 0 for a dense (non-MoE) model.
    pub(crate) fn max_moe_block(&self) -> usize {
        use crate::hybrid::Ffn;
        let mut mx = 0usize;
        for l in &self.layers {
            if let Ffn::Moe(m) = &l.ffn {
                mx = mx.max(m.gate_exps.max_expert_bytes())
                       .max(m.up_exps.max_expert_bytes())
                       .max(m.down_exps.max_expert_bytes());
            }
        }
        mx
    }

    /// Gather embedding rows into f32 [T, n_embd] (token-major) by dequantizing only the needed
    /// rows from the host-side embedding bytes (token_embd is [n_embd, n_vocab], row per token).
    pub fn embed_tokens(&self, e: &Engine, tokens: &[u32]) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let n_embd = self.cfg.n_embd as usize;
        let x = self.embd.gather(n_embd, tokens);
        Ok(e.htod(&x)?)
    }
}

pub type TensorMap = HashMap<String, GpuTensor>;

/// One layer's stacked 256-expert tensor, raw GGUF quant bytes held HOST-RESIDENT.
///
/// EDGE-1: these bytes are NEVER uploaded at load (uploading 29.75GB would OOM a 24GB GPU —
/// this is BUG-4). Per token, only the 8 routed experts are staged H2D into a small GPU scratch.
///
/// ne = [in_f, out_f, n_expert]; the expert axis (ne[2]) is the slowest/highest-stride axis, so
/// expert `e` occupies the CONTIGUOUS byte block `bytes[e*expert_stride .. (e+1)*expert_stride]`.
///
/// THE 3D FIX: GpuTensor::load computes `row_bytes = raw.len()/ne[1]`, which for a stacked 3D
/// tensor ignores the 256-expert axis and is 256x too large (gate_exps -> 430080 instead of 1680).
/// load() here uses `row_bytes = raw.len() / (out_f * n_expert)` (= 1680 gate/up, 544 down).
/// Host byte storage for the expert blocks. Default = a pageable `Vec<u8>` (current behavior). Under
/// BW24_MOE_PINNED (auto-on when BW24_MOE_CACHE is set), the bytes live in CUDA pinned host memory so
/// the miss-path `memcpy_htod` is a true DMA, not a pageable bounce copy (MOE-SLRU-PLAN §C.1).
///
/// CAVEAT (§C.1): `alloc_pinned` uses CU_MEMHOSTALLOC_WRITECOMBINED — great for H2D-only (the expert
/// bytes are never read by the CPU on the hot path), but write-combined memory is SLOW for CPU reads.
/// A future CPU-VNNI cold-expert fallback must NOT read from this buffer.
pub enum HostBuf {
    Paged(Vec<u8>),
    /// Pinned host memory. We keep the `PinnedHostSlice` alive (it owns the allocation; Drop frees it)
    /// AND cache its raw base pointer + len so the hot-path `as_bytes()` needs no per-call event sync.
    Pinned { slice: cudarc::driver::PinnedHostSlice<u8>, base: *const u8, len: usize },
    /// Alias into a shared pinned slab (ST pinned tier): `owner` keeps the slab alive; `base`/`len`
    /// select this expert's window. Same DMA class as `Pinned`.
    PinnedAlias { owner: std::sync::Arc<HostBuf>, base: *const u8, len: usize },
    /// SPILLING-PLAN §1, Tier 2 (disk): the bytes live in an mmap'd region of the GGUF file, NOT in
    /// RAM. `map` is `MAP_SHARED`, no `MAP_POPULATE` — zero upfront copy. The first `memcpy_htod` of
    /// this slice page-faults → NVMe read → DMA (the demand-fault disk path). `off`/`len` select this
    /// expert's contiguous block within the shared file mmap. Bit-identical to `Paged`/`Pinned` —
    /// those copied FROM exactly these on-disk bytes, so the GEMM result is unchanged.
    Mmap { map: std::sync::Arc<memmap2::Mmap>, off: usize, len: usize },
}
// SAFETY: `base` is a stable pinned-host pointer owned by `slice`; the buffer is written once at load
// then only READ for H2D. HostExps is shared `&` across the (single per-Engine) forward, so Send/Sync
// mirror the underlying PinnedHostSlice (which is already Send+Sync). The `Mmap` arm holds an
// `Arc<Mmap>` (Mmap is Send+Sync) plus plain usize fields, so it does not weaken these bounds.
unsafe impl Send for HostBuf {}
unsafe impl Sync for HostBuf {}
impl HostBuf {
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            HostBuf::Paged(v) => v.as_slice(),
            // SAFETY: base+len are the pinned allocation's stable extent; written once at load, then
            // read-only. We avoid `as_slice()` here because it would synchronize the buffer's event
            // on every hot-path call.
            HostBuf::Pinned { base, len, .. } => unsafe { std::slice::from_raw_parts(*base, *len) },
            HostBuf::PinnedAlias { base, len, .. } => unsafe { std::slice::from_raw_parts(*base, *len) },
            // Slicing the mmap is the same `&[u8]` the kernel DMAs; the read page-faults the NVMe.
            HostBuf::Mmap { map, off, len } => &map[*off..*off + *len],
        }
    }
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            HostBuf::Paged(v) => v.len(),
            HostBuf::Pinned { len, .. } => *len,
            HostBuf::PinnedAlias { len, .. } => *len,
            HostBuf::Mmap { len, .. } => *len,
        }
    }
}

/// One layer's stacked 256-expert tensor, raw GGUF quant bytes held HOST-RESIDENT.
///
/// EDGE-1: these bytes are NEVER uploaded at load (uploading 29.75GB would OOM a 24GB GPU —
/// this is BUG-4). Per token, only the 8 routed experts are staged H2D into a small GPU scratch.
///
/// ne = [in_f, out_f, n_expert]; the expert axis (ne[2]) is the slowest/highest-stride axis, so
/// expert `e` occupies the CONTIGUOUS byte block `bytes[e*expert_stride .. (e+1)*expert_stride]`.
///
/// THE 3D FIX: GpuTensor::load computes `row_bytes = raw.len()/ne[1]`, which for a stacked 3D
/// tensor ignores the 256-expert axis and is 256x too large (gate_exps -> 430080 instead of 1680).
/// load() here uses `row_bytes = raw.len() / (out_f * n_expert)` (= 1680 gate/up, 544 down).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertLayout {
    pub offset: usize,
    pub len: usize,
    pub qtype: i32,
    pub row_bytes: usize,
}

fn staged_expert_qtype(ty: GgmlType) -> Option<i32> {
    Some(match ty {
        GgmlType::Q8_0 => QT_Q8_0,
        GgmlType::Q2_K => QT_Q2_K,
        GgmlType::Q4_K => QT_Q4_K,
        GgmlType::Q6_K => QT_Q6_K,
        GgmlType::Q5_K => QT_Q5_K,
        GgmlType::Q3_K => QT_Q3_K,
        GgmlType::IQ4_XS => QT_IQ4_XS,
        GgmlType::IQ3_S => QT_IQ3_S,
        GgmlType::NVFP4 => QT_NVFP4,
        GgmlType::F32 => QT_F32,
        GgmlType::BF16 => QT_BF16,
        _ => return None,
    })
}

fn staged_expert_row_bytes(ty: GgmlType, in_f: usize) -> Option<usize> {
    staged_expert_qtype(ty)?;
    let (block, type_size) = ty.block_and_type_size();
    assert_eq!(in_f as u64 % block, 0,
        "expert row width {in_f} is not divisible by {ty:?} block {block}");
    Some((in_f as u64 / block * type_size) as usize)
}

pub struct HostExps {
    pub bytes: HostBuf,        // raw GGUF block bytes (host); per-token DMA src for the 8 routed exps
    /// SPILLING-PLAN §1.1: per-expert backing tier. `None` => the layer fits in one `bytes` store and
    /// every expert slices it (the unchanged in-RAM path). `Some` => per-expert split: the hottest
    /// experts are `Pinned` (Tier 1, fast async DMA), the rest `Mmap` into the GGUF (Tier 2, disk
    /// demand-fault). `expert_bytes(e)` resolves `tiers[e]` if present, else slices `bytes`.
    pub tiers: Option<Vec<HostBuf>>,
    pub qtype: i32,            // QT_Q6_K (gate/up) | QT_Q8_0 (down)
    pub in_f: usize,           // ne[0]   (gate/up = 2048, down = 512)
    pub out_f: usize,          // ne[1]   (gate/up = 512,  down = 2048)
    pub n_expert: usize,       // ne[2] = 256
    pub row_bytes: usize,      // raw.len()/(out_f*n_expert)  -> 1680 (gate/up) / 544 (down)
    pub expert_stride: usize,  // raw.len()/n_expert          -> 860160 (gate/up) / 1114112 (down)
    /// Per-expert encoding metadata when experts in this projection do not share one dtype/layout.
    /// `None` preserves the existing uniform slab contract and every resident/fused fast path.
    /// `Some` routes through the per-expert staged/cache path, using each entry's qtype/row size.
    pub layouts: Option<Vec<ExpertLayout>>,
    /// Per-expert post-matmul macro-scale (ModelOpt NVFP4 `weight_scale_2`, one scalar per expert
    /// tensor). `None` => all 1.0 (GGUF experts; block scales carry everything). The MoE forward
    /// folds gate/up macros into the activation epilogue (gs/us) and the down macro into the
    /// per-expert accumulate weight.
    pub macros: Option<Vec<f32>>,
}

impl HostExps {
    /// Load a stacked 3D expert tensor, keeping its quant bytes on the HOST. `e` supplies the CUDA
    /// context for the optional pinned allocation (§C.1). Default storage is pageable `Vec<u8>`
    /// (identical to the prior behavior); pinned is chosen when BW24_MOE_PINNED or BW24_MOE_CACHE is set.
    pub fn load(e: &Engine, g: &GgufFile, name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        Self::load_stacked_from_source(e, &GgufSource(g), name)
    }

    /// Load a STACKED 3D expert tensor (`ne=[in_f,out_f,n_expert]`) from any source. GGUF stores the
    /// experts this way; the source returns the same mmap bytes (`GgufSource::find` == `tensor_data`),
    /// so the GGUF path is byte-identical to the prior direct-`GgufFile` loader. (Safetensors stores N
    /// 2D tensors instead — those go through `load_from_source`, which gathers them.)
    pub fn load_stacked_from_source(e: &Engine, src: &dyn TensorSource, name: &str)
                                    -> Result<Self, Box<dyn std::error::Error>> {
        let t = src.find(name).unwrap_or_else(|| panic!("missing exps tensor {name}"));
        assert_eq!(t.ne.len(), 3, "{name} is not a 3D stacked-expert tensor (ne={:?})", t.ne);
        // MMAP-BACKED SPILL TIER (Hy3 repack dir, 2026-07-09): when the source's on-disk layout IS
        // already the engine's expert layout (one expert-axis-slowest slab file per (layer, proj),
        // the transcoder's contract), back the HostExps with `HostBuf::Mmap` directly — ZERO host
        // copy. The default copy path below would pin/allocate the WHOLE stacked slab (80.5 GB for
        // Hy3-REAP50 on a 60 GB host = the M3 first-load OOM class); the mmap tier instead lets the
        // page cache carry the hot expert mass (RAM tier) and demand-faults the overflow from NVMe,
        // exactly like the proven M3 `.bw24-repack` path (model.rs NVFP4 disk arm). Bit-identity:
        // `expert_bytes(e)` slices the same on-disk bytes the copy would have staged. The SLRU VRAM
        // cache stacks on top unchanged. MADV_RANDOM is applied by the source at open.
        if let Some((map, off, len)) = src.find_expert_mmap(name) {
            let qtype = match t.ggml_type {
                GgmlType::Q8_0 => QT_Q8_0,
                GgmlType::Q4_K => QT_Q4_K,
                GgmlType::Q6_K => QT_Q6_K,
                GgmlType::Q5_K => QT_Q5_K,
                GgmlType::Q3_K => QT_Q3_K,
                GgmlType::IQ4_XS => QT_IQ4_XS,
                GgmlType::IQ3_S => QT_IQ3_S,
                GgmlType::NVFP4 => QT_NVFP4,
                other => panic!("exps {name} unsupported quant {other:?}"),
            };
            let in_f = t.ne[0] as usize;
            let out_f = t.ne[1] as usize;
            let n_expert = t.ne[2] as usize;
            let expert_stride = len / n_expert;
            let row_bytes = len / (out_f * n_expert);
            assert_eq!(expert_stride, out_f * row_bytes,
                "{name} stride mismatch: stride={expert_stride} out_f={out_f} row_bytes={row_bytes}");
            assert_eq!(len, n_expert * expert_stride, "{name} mmap len != n_expert*stride");
            return Ok(HostExps {
                bytes: HostBuf::Mmap { map, off, len },
                tiers: None, qtype, in_f, out_f, n_expert, row_bytes, expert_stride,
                layouts: None, macros: None,
            });
        }
        let raw: &[u8] = &t.bytes;
        // All quant types the staged-expert qmatvec can decode (dp4a-fast or Stage-A f32).
        let qtype = match t.ggml_type {
            GgmlType::Q8_0 => QT_Q8_0,
            GgmlType::Q4_K => QT_Q4_K,
            GgmlType::Q6_K => QT_Q6_K,
            GgmlType::Q5_K => QT_Q5_K,
            GgmlType::Q3_K => QT_Q3_K,
            GgmlType::IQ4_XS => QT_IQ4_XS,
            GgmlType::IQ3_S => QT_IQ3_S,
            GgmlType::NVFP4 => QT_NVFP4,
            other => panic!("exps {name} unsupported quant {other:?}"),
        };
        let in_f = t.ne[0] as usize;
        let out_f = t.ne[1] as usize;
        let n_expert = t.ne[2] as usize;
        // VERIFIED: gate/up Q6_K total/256 = 860160; row = total/(512*256) = 1680.
        //           down  Q8_0 total/256 = 1114112; row = total/(2048*256) = 544.
        let expert_stride = raw.len() / n_expert;
        let row_bytes = raw.len() / (out_f * n_expert);
        // sanity: expert_stride must equal out_f * row_bytes exactly (catches a dim mixup)
        assert_eq!(expert_stride, out_f * row_bytes,
            "{name} stride mismatch: stride={expert_stride} out_f={out_f} row_bytes={row_bytes}");

        let pinned = std::env::var("BW24_MOE_PINNED").is_ok()
            || std::env::var("BW24_MOE_CACHE").as_deref() != Ok("0");
        let bytes = if pinned {
            // alloc pinned host memory, copy the GGUF block bytes in once, cache the base pointer.
            let mut p = unsafe { e.ctx().alloc_pinned::<u8>(raw.len())? };
            { let dst = p.as_mut_slice()?; dst.copy_from_slice(raw); }
            let base = p.as_ptr()? as *const u8;   // syncs once here at load; stable afterward
            let len = raw.len();
            HostBuf::Pinned { slice: p, base, len }
        } else {
            HostBuf::Paged(raw.to_vec())
        };
        Ok(HostExps { bytes, tiers: None, qtype, in_f, out_f, n_expert, row_bytes,
                      expert_stride, layouts: None, macros: None })
    }

    /// SPILLING-PLAN §1.1, §2 step 4: load a stacked 3D expert tensor with a PER-EXPERT tier split.
    /// Under `BW24_SPILL_DISK`, the hottest experts (greedy in expert order, until the shared pinned
    /// budget in `ctx` is exhausted) get `HostBuf::Pinned` (Tier 1, fast async DMA); every remaining
    /// expert is `HostBuf::Mmap` into the GGUF (Tier 2, demand-faulted from disk on first H2D). The
    /// resulting bytes are bit-identical to the in-RAM path either way — `qmatvec_view` is untouched.
    ///
    /// `ctx.file_map` is ONE shared `MAP_SHARED` mmap of the whole GGUF (`Arc`-cloned per spilled
    /// expert), so the 120 expert tensors of a 40-layer MoE never open the file more than once.
    pub fn load_tiered(e: &Engine, g: &GgufFile, name: &str, ctx: &mut crate::spill::SpillCtx)
                       -> Result<Self, Box<dyn std::error::Error>> {
        let t = g.find(name).unwrap_or_else(|| panic!("missing exps tensor {name}"));
        assert_eq!(t.ne.len(), 3, "{name} is not a 3D stacked-expert tensor (ne={:?})", t.ne);
        let raw = g.tensor_data(t);
        let qtype = match t.ggml_type {
            GgmlType::Q8_0 => QT_Q8_0,
            GgmlType::Q4_K => QT_Q4_K,
            GgmlType::Q6_K => QT_Q6_K,
            GgmlType::Q5_K => QT_Q5_K,
            GgmlType::Q3_K => QT_Q3_K,
            GgmlType::IQ4_XS => QT_IQ4_XS,
            GgmlType::IQ3_S => QT_IQ3_S,
            GgmlType::NVFP4 => QT_NVFP4,
            other => panic!("exps {name} unsupported quant {other:?}"),
        };
        let in_f = t.ne[0] as usize;
        let out_f = t.ne[1] as usize;
        let n_expert = t.ne[2] as usize;
        let expert_stride = raw.len() / n_expert;
        let row_bytes = raw.len() / (out_f * n_expert);
        assert_eq!(expert_stride, out_f * row_bytes,
            "{name} stride mismatch: stride={expert_stride} out_f={out_f} row_bytes={row_bytes}");

        // Absolute file offset of this tensor's data (start of expert 0); each expert is the next
        // `expert_stride` bytes. The `Mmap` arm slices `ctx.file_map` at these offsets.
        let (file_start, _file_end) = g.tensor_file_range(t);

        // Per-expert tier decision under the shared running budget. `bytes` keeps a 0-byte sentinel
        // (`Paged(empty)`) since every read now goes through `tiers`.
        let mut tiers = Vec::with_capacity(n_expert);
        for ex in 0..n_expert {
            let blk = &raw[ex * expert_stride..(ex + 1) * expert_stride];
            let file_off = file_start + ex * expert_stride;
            tiers.push(crate::spill::place_expert(ctx, e, blk, file_off)?);
        }
        Ok(HostExps {
            bytes: HostBuf::Paged(Vec::new()),  // unused when `tiers` is Some
            tiers: Some(tiers),
            qtype, in_f, out_f, n_expert, row_bytes, expert_stride, layouts: None, macros: None,
        })
    }

    /// MoE expert GATHER from a `TensorSource` (the safetensors path; ST-MOE-PLAN §1.3). GGUF stacks
    /// all experts into ONE 3D tensor; HF stores them as N separate 2D tensors
    /// `model.layers.{il}.mlp.experts.{e}.{gate,up,down}_proj.weight`. `find` returns `None` for the
    /// ggml `*_exps` name on purpose, so the experts are gathered out-of-band here.
    ///
    /// PATH A (load-time only, no quantize): each HF 2D expert tensor is dequantized to f32 and the
    /// per-expert blocks are concatenated expert-axis-slowest into ONE contiguous buffer — exactly the
    /// layout `expert_bytes(e)` slices and the staged `qmatvec_view` (qtype=QT_F32) reads. The same
    /// `expert_stride == out_f*row_bytes` invariant as the GGUF path is asserted at the end.
    ///
    /// `ggml_exps_name` is `blk.{il}.ffn_{gate,up,down}_exps.weight`; it is split to recover `il` and
    /// the proj. `n_expert` comes from `cfg.moe`. The HF per-expert literal `mlp.experts.{e}.{p}_proj`
    /// is the qwen3moe / olmoe layout (a future arch with `block_sparse_moe.experts.*` would need a
    /// branch in `hf_expert_name`).
    pub fn load_from_source(e: &Engine, src: &dyn TensorSource, ggml_exps_name: &str, n_expert: usize)
                            -> Result<Self, Box<dyn std::error::Error>> {
        // Recover il + proj from `blk.{il}.ffn_{gate,up,down}_exps.weight`.
        let rest = ggml_exps_name.strip_prefix("blk.")
            .unwrap_or_else(|| panic!("not a blk.* name: {ggml_exps_name}"));
        let (il_s, suffix) = rest.split_once('.').unwrap();
        let il: u32 = il_s.parse().unwrap();
        let proj = match suffix {
            "ffn_gate_exps.weight" => "gate",
            "ffn_up_exps.weight"   => "up",
            "ffn_down_exps.weight" => "down",
            other => panic!("not a *_exps suffix: {other}"),
        };

        // A mixed-precision safetensors/repack source exposes experts as separate 2D tensors.
        // Detect a dtype/layout change before the uniform gather paths normalize the whole layer
        // to one encoding. Uniform checkpoints take the unchanged optimized path below.
        let mut signatures = Vec::with_capacity(n_expert);
        let active = src.active_experts(il);
        for ex in 0..n_expert {
            if active.is_some_and(|mask| !mask[ex]) {
                signatures.push((i32::MIN, 0));
                continue;
            }
            let name = format!("blk.{il}.ffn_{proj}_exps.{ex}.weight");
            if let Some(nv) = src.find_nvfp4_native(&name) {
                signatures.push((QT_NVFP4, nv.in_f / 64 * 36));
            } else {
                let v = src.find(&name).unwrap_or_else(|| panic!("missing expert tensor {name}"));
                let in_f = v.ne[0] as usize;
                signatures.push(match staged_expert_row_bytes(v.ggml_type, in_f) {
                    Some(row_bytes) => (staged_expert_qtype(v.ggml_type).unwrap(), row_bytes),
                    None => (QT_F32, in_f * 4),
                });
            }
        }
        if src.preserve_expert_encodings()
            || signatures.windows(2).any(|pair| pair[0] != pair[1]) {
            return Self::load_mixed_from_source(src, il, proj, n_expert);
        }

        // PATH B (NVFP4-NATIVE GATHER, 2026-07-05): when the source exposes the experts as packed
        // ModelOpt/Reza NVFP4 (find_nvfp4_native), keep them QUANTIZED — repack each expert's
        // modelopt bytes to the GGUF 36B-block layout the staged qmatvec decodes, and concatenate.
        // No f32 blow-up: a 129GB checkpoint gathers to ~the same bytes instead of ~8x (which is
        // what makes MiniMax-M3 REAP50 loadable on a 60GB-RAM host at all, with spill on top).
        // Per-expert `weight_scale_2` macros go to `macros` (folded post-matmul by the MoE forward).
        {
            let name0 = format!("blk.{il}.ffn_{proj}_exps.0.weight");
            if let Some(nv0) = src.find_nvfp4_native(&name0) {
                let (in_f, out_f) = (nv0.in_f, nv0.out_f);
                let row_bytes = in_f / 64 * 36;
                let expert_stride = out_f * row_bytes;
                // ST DISK TIER (2026-07-06, the MiniMax OOM fix): when the total expert bytes
                // exceed host RAM (M3 REAP50 = 122GB repacked on a 60GB host, first-load host-OOM
                // at layer ~24), repack each layer ONCE into an on-disk cache file next to the
                // checkpoint and mmap it (HostBuf::Mmap, MAP_SHARED no-populate — the same tier-2
                // mechanism the GGUF spill path uses). Reloads hit the cache (size-checked), pay
                // zero repack. BW24_ST_REPACK_DISK=0 forces the old in-RAM gather.
                let disk = std::env::var("BW24_ST_REPACK_DISK").map(|v| v != "0").unwrap_or(true)
                    && src.st_dir().is_some();
                let cache_path = src.st_dir().map(|d| {
                    let cd = d.join(".bw24-repack");
                    let _ = std::fs::create_dir_all(&cd);
                    cd.join(format!("blk{il}-{proj}-{n_expert}x{out_f}x{in_f}.nvfp4"))
                });
                let total = n_expert * expert_stride;
                let mut macros = vec![1.0f32; n_expert];
                let read_macros = |macros: &mut Vec<f32>| {
                    for ex in 0..n_expert {
                        let stem = format!("blk.{il}.ffn_{proj}_exps.{ex}");
                        if let Some(sv) = src.find(&format!("{stem}.scale")) {
                            macros[ex] = f32::from_le_bytes(sv.bytes[..4].try_into().unwrap());
                        }
                    }
                };
                let bytes = if disk {
                    let cp = cache_path.as_ref().unwrap();
                    let fresh = std::fs::metadata(cp).map(|m| m.len() as usize == total).unwrap_or(false);
                    if !fresh {
                        // stream one expert at a time to disk — peak RAM = one expert (~8MB)
                        use std::io::Write;
                        let mut f = std::io::BufWriter::new(std::fs::File::create(cp)?);
                        for ex in 0..n_expert {
                            let name = format!("blk.{il}.ffn_{proj}_exps.{ex}.weight");
                            let nv = src.find_nvfp4_native(&name)
                                .unwrap_or_else(|| panic!("expert {name} lost NVFP4-native mid-gather"));
                            assert_eq!((nv.in_f, nv.out_f), (in_f, out_f),
                                "expert {ex} dims ({},{}) != expert 0 ({in_f},{out_f})", nv.in_f, nv.out_f);
                            f.write_all(&bw24_gguf::nvfp4_repack::repack_modelopt_to_gguf(
                                nv.wbytes, nv.wscale, out_f, in_f))?;
                        }
                        f.flush()?;
                    }
                    read_macros(&mut macros);
                    let file = std::fs::File::open(cp)?;
                    let map = unsafe { memmap2::Mmap::map(&file)? };
                    assert_eq!(map.len(), total, "repack cache {cp:?} size mismatch");
                    // MADV_RANDOM: expert access is routing-driven random; kill readahead waste.
                    #[cfg(target_os = "linux")]
                    unsafe {
                        unsafe extern "C" { fn madvise(a: *mut core::ffi::c_void, l: usize, ad: i32) -> i32; }
                        let _ = madvise(map.as_ptr() as *mut core::ffi::c_void, map.len(), 1);
                    }
                    let map = std::sync::Arc::new(map);
                    // ST PINNED TIER (2026-07-07, the M3 1.5-tok/s lever): mmap-only backing makes
                    // every SLRU miss a page-cache (or NVMe) synchronous read into the H2D copy.
                    // Pin as many experts as the live budget allows (same MemBudget probe + 0.6
                    // MemAvailable cap as the GGUF spill tier) — pinned pages upload via true
                    // async DMA at full PCIe. Budget is GLOBAL across layers (first-come: earlier
                    // layers pin first; routing is roughly uniform so early-layer bias is benign).
                    // BW24_ST_PINNED=0 disables (pure-mmap, the 2026-07-06 behavior).
                    // DEFAULT OFF (2026-07-07 measured): with a 122GB expert set on 60GB RAM,
                    // pinning 26GB EVICTED the page cache backing the mmap tier — every unpinned
                    // expert faulted cold from NVMe and gen fell 1.5 -> 0.05 tok/s (30x WORSE).
                    // Pinning only pays when (total - pinned) fits page cache; here it never can.
                    // BW24_ST_PINNED=1 opt-in for fits-in-RAM checkpoints (e.g. REAP-heavier cuts).
                    let tiers = if std::env::var("BW24_ST_PINNED").map(|v| v == "1").unwrap_or(false) {
                        static PIN_BUDGET: std::sync::OnceLock<std::sync::Mutex<usize>> = std::sync::OnceLock::new();
                        let budget = PIN_BUDGET.get_or_init(|| {
                            let b = crate::spill::MemBudget::probe(e)
                                .map(|b| b.free_pinnable_ram).unwrap_or(0);
                            eprintln!("[st-spill] pinned budget {:.1} GB", b as f64 / 1e9);
                            std::sync::Mutex::new(b)
                        });
                        let mut rem = budget.lock().unwrap();
                        // ONE pinned slab per file prefix (n_pin experts contiguous): 1 alloc +
                        // 1 bulk copy instead of n_pin small allocs (per-expert cudaHostAllocs
                        // stalled the 122GB M3 load >10min).
                        let n_pin = (*rem / expert_stride).min(n_expert);
                        if n_pin == 0 { None } else {
                            let slab_len = n_pin * expert_stride;
                            let mut pn = unsafe { e.ctx().alloc_pinned::<u8>(slab_len)? };
                            { let dst = pn.as_mut_slice()?; dst.copy_from_slice(&map[..slab_len]); }
                            let base = pn.as_ptr()? as *const u8;
                            *rem -= slab_len;
                            let slab = std::sync::Arc::new(HostBuf::Pinned { slice: pn, base, len: slab_len });
                            let mut tiers: Vec<HostBuf> = Vec::with_capacity(n_expert);
                            for ex in 0..n_expert {
                                let off = ex * expert_stride;
                                if ex < n_pin {
                                    tiers.push(HostBuf::PinnedAlias { owner: slab.clone(),
                                        base: unsafe { base.add(off) }, len: expert_stride });
                                } else {
                                    tiers.push(HostBuf::Mmap { map: map.clone(), off, len: expert_stride });
                                }
                            }
                            Some(tiers)
                        }
                    } else { None };
                    if let Some(tiers) = tiers {
                        let all_one = macros.iter().all(|&m| m == 1.0);
                        return Ok(HostExps {
                            bytes: HostBuf::Mmap { map, off: 0, len: total },
                            tiers: Some(tiers), qtype: QT_NVFP4, in_f, out_f, n_expert,
                            row_bytes, expert_stride, layouts: None,
                            macros: if all_one { None } else { Some(macros) },
                        });
                    }
                    HostBuf::Mmap { map, off: 0, len: total }
                } else {
                    let mut buf: Vec<u8> = Vec::with_capacity(total);
                    for ex in 0..n_expert {
                        let name = format!("blk.{il}.ffn_{proj}_exps.{ex}.weight");
                        let nv = src.find_nvfp4_native(&name)
                            .unwrap_or_else(|| panic!("expert {name} lost NVFP4-native mid-gather"));
                        assert_eq!((nv.in_f, nv.out_f), (in_f, out_f),
                            "expert {ex} dims ({},{}) != expert 0 ({in_f},{out_f})", nv.in_f, nv.out_f);
                        buf.extend_from_slice(&bw24_gguf::nvfp4_repack::repack_modelopt_to_gguf(
                            nv.wbytes, nv.wscale, out_f, in_f));
                    }
                    assert_eq!(buf.len(), total);
                    read_macros(&mut macros);
                    let pinned = std::env::var("BW24_MOE_PINNED").is_ok()
                        || std::env::var("BW24_MOE_CACHE").as_deref() != Ok("0");
                    if pinned {
                        let mut p = unsafe { e.ctx().alloc_pinned::<u8>(buf.len())? };
                        { let dst = p.as_mut_slice()?; dst.copy_from_slice(&buf); }
                        let base = p.as_ptr()? as *const u8;
                        let len = buf.len();
                        HostBuf::Pinned { slice: p, base, len }
                    } else { HostBuf::Paged(buf) }
                };
                let all_one = macros.iter().all(|&m| m == 1.0);
                return Ok(HostExps {
                    bytes, tiers: None, qtype: QT_NVFP4, in_f, out_f, n_expert,
                    row_bytes, expert_stride, layouts: None,
                    macros: if all_one { None } else { Some(macros) },
                });
            }
        }

        // expert 0 fixes (in_f, out_f); every later expert must match (catches a layer/arch mixup).
        let mut buf: Vec<u8> = Vec::new();
        let mut in_f = 0usize;
        let mut out_f = 0usize;
        for ex in 0..n_expert {
            // Per-expert ggml name; the source maps it to the HF expert tensor (ST-MOE-PLAN §1.3).
            let name = format!("blk.{il}.ffn_{proj}_exps.{ex}.weight");
            let v = src.find(&name).unwrap_or_else(|| panic!("missing expert tensor {name}"));
            assert_eq!(v.ne.len(), 2, "expert {name} is not 2D (ne={:?})", v.ne);
            let (cur_in, cur_out) = (v.ne[0] as usize, v.ne[1] as usize);
            if ex == 0 { in_f = cur_in; out_f = cur_out; }
            else { assert_eq!((cur_in, cur_out), (in_f, out_f),
                "expert {ex} dims {:?} != expert 0 [{in_f},{out_f}]", (cur_in, cur_out)); }
            // PATH A: dequant the 2D expert (F32/F16/BF16) to f32, append its bytes verbatim. The
            // dequantized [out_f, in_f] row-major f32 block is exactly one expert_stride slow→fast.
            let n = cur_in * cur_out;
            let f32v = dequant::dequantize(v.ggml_type, &v.bytes, n);
            buf.reserve(n * 4);
            for f in &f32v { buf.extend_from_slice(&f.to_le_bytes()); }
        }
        let row_bytes = in_f * 4;                 // one out-row = in_f contiguous f32s
        let expert_stride = out_f * row_bytes;
        assert_eq!(buf.len(), n_expert * expert_stride,
            "{ggml_exps_name} gather size {} != n_expert*stride {}", buf.len(), n_expert * expert_stride);
        // Hold to the identical invariant as the GGUF path (ST-MOE-PLAN §1.3 step 4).
        assert_eq!(expert_stride, out_f * row_bytes,
            "{ggml_exps_name} stride mismatch: stride={expert_stride} out_f={out_f} row_bytes={row_bytes}");

        // Same pinned-vs-paged choice as the GGUF loader (the bytes are H2D-only on the hot path).
        let pinned = std::env::var("BW24_MOE_PINNED").is_ok()
            || std::env::var("BW24_MOE_CACHE").as_deref() != Ok("0");
        let bytes = if pinned {
            let mut p = unsafe { e.ctx().alloc_pinned::<u8>(buf.len())? };
            { let dst = p.as_mut_slice()?; dst.copy_from_slice(&buf); }
            let base = p.as_ptr()? as *const u8;
            let len = buf.len();
            HostBuf::Pinned { slice: p, base, len }
        } else {
            HostBuf::Paged(buf)
        };
        Ok(HostExps { bytes, tiers: None, qtype: QT_F32, in_f, out_f, n_expert, row_bytes,
                      expert_stride, layouts: None, macros: None })
    }

    fn load_mixed_from_source(src: &dyn TensorSource, il: u32, proj: &str, n_expert: usize)
                              -> Result<Self, Box<dyn std::error::Error>> {
        let mut tiers = Vec::with_capacity(n_expert);
        let mut layouts = Vec::with_capacity(n_expert);
        let mut macros = vec![1.0f32; n_expert];
        let mut in_f = 0usize;
        let mut out_f = 0usize;
        let active = src.active_experts(il);
        let mut first_active = None;

        for ex in 0..n_expert {
            if active.is_some_and(|mask| !mask[ex]) {
                layouts.push(ExpertLayout { offset: 0, len: 0, qtype: QT_F32, row_bytes: 0 });
                tiers.push(HostBuf::Paged(Vec::new()));
                continue;
            }
            let name = format!("blk.{il}.ffn_{proj}_exps.{ex}.weight");
            let (bytes, qtype, row_bytes, cur_in, cur_out) =
                if let Some(nv) = src.find_nvfp4_native(&name) {
                    let stem = format!("blk.{il}.ffn_{proj}_exps.{ex}");
                    if let Some(scale) = src.find(&format!("{stem}.scale")) {
                        macros[ex] = f32::from_le_bytes(scale.bytes[..4].try_into().unwrap());
                    }
                    let bytes = bw24_gguf::nvfp4_repack::repack_modelopt_to_gguf(
                        nv.wbytes, nv.wscale, nv.out_f, nv.in_f);
                    let row_bytes = nv.in_f / 64 * 36;
                    (bytes, QT_NVFP4, row_bytes, nv.in_f, nv.out_f)
                } else {
                    let v = src.find(&name).unwrap_or_else(|| panic!("missing expert tensor {name}"));
                    assert_eq!(v.ne.len(), 2, "expert {name} is not 2D (ne={:?})", v.ne);
                    let (cur_in, cur_out) = (v.ne[0] as usize, v.ne[1] as usize);
                    if let Some(row_bytes) = staged_expert_row_bytes(v.ggml_type, cur_in) {
                        (v.bytes.into_owned(), staged_expert_qtype(v.ggml_type).unwrap(),
                         row_bytes, cur_in, cur_out)
                    } else {
                        let f32v = dequant::dequantize(v.ggml_type, &v.bytes, cur_in * cur_out);
                        let mut bytes = Vec::with_capacity(f32v.len() * 4);
                        for f in f32v { bytes.extend_from_slice(&f.to_le_bytes()); }
                        (bytes, QT_F32, cur_in * 4, cur_in, cur_out)
                    }
                };

            if first_active.is_none() {
                in_f = cur_in;
                out_f = cur_out;
                first_active = Some(ex);
            } else {
                assert_eq!((cur_in, cur_out), (in_f, out_f),
                    "expert {ex} dims ({cur_in},{cur_out}) != first active expert ({in_f},{out_f})");
            }
            assert_eq!(bytes.len(), cur_out * row_bytes,
                "expert {name} bytes {} != out_f*row_bytes {}", bytes.len(), cur_out * row_bytes);
            layouts.push(ExpertLayout { offset: 0, len: bytes.len(), qtype, row_bytes });
            tiers.push(HostBuf::Paged(bytes));
        }

        let first = layouts[*first_active.as_ref().expect("expert mask pruned every expert")];
        let expert_stride = layouts.iter().map(|layout| layout.len).max().unwrap_or(0);
        let all_one = macros.iter().all(|&scale| scale == 1.0);
        Ok(HostExps {
            bytes: HostBuf::Paged(Vec::new()),
            tiers: Some(tiers),
            qtype: first.qtype,
            in_f,
            out_f,
            n_expert,
            row_bytes: first.row_bytes,
            expert_stride,
            layouts: Some(layouts),
            macros: if all_one { None } else { Some(macros) },
        })
    }

    /// Host byte slice for expert `e` (the H2D DMA source). Contiguous block, offset honored.
    /// Resolves the per-expert tier when spilling is active (`tiers` Some), else slices the single
    /// Per-expert post-matmul macro-scale (1.0 when absent).
    #[inline]
    pub fn macro_scale(&self, e: usize) -> f32 {
        self.macros.as_ref().map(|m| m[e]).unwrap_or(1.0)
    }

    #[inline]
    pub fn is_uniform_layout(&self) -> bool {
        self.layouts.is_none()
    }

    #[inline]
    pub fn expert_layout(&self, e: usize) -> ExpertLayout {
        debug_assert!(e < self.n_expert, "expert index {e} >= n_expert {}", self.n_expert);
        self.layouts.as_ref().map(|layouts| layouts[e]).unwrap_or(ExpertLayout {
            offset: e * self.expert_stride,
            len: self.expert_stride,
            qtype: self.qtype,
            row_bytes: self.row_bytes,
        })
    }

    #[inline]
    pub fn max_expert_bytes(&self) -> usize {
        self.layouts.as_ref()
            .and_then(|layouts| layouts.iter().map(|layout| layout.len).max())
            .unwrap_or(self.expert_stride)
    }

    /// backing store (unchanged in-RAM path). Each `tiers[e]` is exactly one expert's stride.
    #[inline]
    pub fn expert_bytes(&self, e: usize) -> &[u8] {
        let layout = self.expert_layout(e);
        match &self.tiers {
            Some(tiers) => {
                debug_assert_eq!(tiers[e].len(), layout.len);
                tiers[e].as_bytes()
            }
            None => &self.bytes.as_bytes()[layout.offset..layout.offset + layout.len],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{HostExps, QT_BF16, QT_Q2_K, QT_Q4_K, QT_NVFP4, repack_nvfp4_split, unpack_nvfp4_split};
    use std::borrow::Cow;
    use bw24_gguf::{GgmlType, config::ModelConfig};
    use bw24_gguf::nvfp4_repack::{repack_modelopt_to_gguf, repack_modelopt_to_split};
    use bw24_gguf::source::{TensorSource, TensorView};

    struct MixedExpertSource {
        bf16: Vec<u8>,
        q4k: Vec<u8>,
    }

    impl TensorSource for MixedExpertSource {
        fn config(&self) -> ModelConfig { panic!("unused by HostExps mixed-loader test") }

        fn find(&self, name: &str) -> Option<TensorView<'_>> {
            let (bytes, ggml_type) = if name == "blk.0.ffn_gate_exps.0.weight" {
                (&self.bf16, GgmlType::BF16)
            } else if name == "blk.0.ffn_gate_exps.1.weight" {
                (&self.q4k, GgmlType::Q4_K)
            } else {
                return None;
            };
            Some(TensorView {
                bytes: Cow::Borrowed(bytes),
                ggml_type,
                ne: vec![256, 2],
            })
        }
    }

    struct PrunedExpertSource {
        q2k: Vec<u8>,
        nvfp4: Vec<u8>,
        active: Vec<bool>,
    }

    impl TensorSource for PrunedExpertSource {
        fn config(&self) -> ModelConfig { panic!("unused by HostExps pruned-loader test") }
        fn active_experts(&self, layer: u32) -> Option<&[bool]> {
            (layer == 0).then_some(self.active.as_slice())
        }
        fn find(&self, name: &str) -> Option<TensorView<'_>> {
            let (bytes, ggml_type) = match name {
                "blk.0.ffn_gate_exps.0.weight" => (&self.q2k, GgmlType::Q2_K),
                "blk.0.ffn_gate_exps.2.weight" => (&self.nvfp4, GgmlType::NVFP4),
                _ => return None,
            };
            Some(TensorView { bytes: Cow::Borrowed(bytes), ggml_type, ne: vec![256, 2] })
        }
    }

    /// A1 direct-import gate (engine side): the fused modelopt->split repack must be byte-for-byte
    /// the composition of the two passes it replaces (modelopt->GGUF blocks, then the A6
    /// split-plane repack). Also pins the split roundtrip on the same buffers.
    #[test]
    fn direct_split_equals_chained() {
        for (out_f, in_f) in [(1usize, 64usize), (3, 128), (5, 320), (8, 1024)] {
            let mut w = vec![0u8; out_f * in_f / 2];
            let mut s = vec![0u8; out_f * in_f / 16];
            for (i, b) in w.iter_mut().enumerate() { *b = ((i * 41 + 7) & 0xFF) as u8; }
            for (i, b) in s.iter_mut().enumerate() { *b = (0x20 + ((i * 11 + 5) % 0x50)) as u8; }
            let gguf = repack_modelopt_to_gguf(&w, &s, out_f, in_f);
            let chained = repack_nvfp4_split(&gguf, out_f);
            let direct = repack_modelopt_to_split(&w, &s, out_f, in_f);
            assert_eq!(direct, chained, "fused != chained at out_f={out_f} in_f={in_f}");
            assert_eq!(unpack_nvfp4_split(&direct, out_f), gguf,
                       "split roundtrip broken at out_f={out_f} in_f={in_f}");
        }
    }

    #[test]
    fn mixed_expert_loader_keeps_each_encoding_and_extent() {
        let source = MixedExpertSource {
            bf16: vec![0x5a; 256 * 2 * 2],
            q4k: vec![0xa5; 2 * 144],
        };
        let exps = HostExps::load_mixed_from_source(&source, 0, "gate", 2).unwrap();
        assert!(!exps.is_uniform_layout());
        assert_eq!(exps.max_expert_bytes(), 1024);
        assert_eq!(exps.expert_layout(0).qtype, QT_BF16);
        assert_eq!(exps.expert_layout(0).row_bytes, 512);
        assert_eq!(exps.expert_layout(0).len, 1024);
        assert_eq!(exps.expert_layout(1).qtype, QT_Q4_K);
        assert_eq!(exps.expert_layout(1).row_bytes, 144);
        assert_eq!(exps.expert_layout(1).len, 288);
        assert_eq!(exps.expert_bytes(0), source.bf16);
        assert_eq!(exps.expert_bytes(1), source.q4k);
    }

    #[test]
    fn mixed_expert_loader_omits_masked_expert_bytes() {
        let source = PrunedExpertSource {
            q2k: vec![0x22; 2 * 84],
            nvfp4: vec![0x44; 2 * 4 * 36],
            active: vec![true, false, true],
        };
        let exps = HostExps::load_mixed_from_source(&source, 0, "gate", 3).unwrap();
        assert_eq!(exps.expert_layout(0).qtype, QT_Q2_K);
        assert_eq!(exps.expert_layout(0).row_bytes, 84);
        assert_eq!(exps.expert_layout(1).len, 0);
        assert_eq!(exps.expert_bytes(1), &[]);
        assert_eq!(exps.expert_layout(2).qtype, QT_NVFP4);
        assert_eq!(exps.expert_layout(2).row_bytes, 4 * 36);
    }
}
