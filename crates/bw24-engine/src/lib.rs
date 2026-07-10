//! bw24 engine: Stage-1 correctness-first forward-pass kernels + ops, on sm_120 via cudarc.

use std::sync::{Arc, Mutex};
use cudarc::driver::{CudaContext, CudaStream, CudaModule, CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

pub use bw24_gguf;
pub use bw24_runtime;

pub mod model;
pub mod forward;
pub mod hybrid;
pub mod hybrid_forward;
pub mod cache;
pub mod decode;
pub mod spec;
pub mod eagle;
pub mod sampler;

/// In-house MoE router GEMV on the spec-verify small-t path (DEFAULT ON since 2026-07-10:
/// battery green on 35B p2/p3 K=1..8, acceptance bit-identical, +2-4% spec e2e — replaces
/// ~240 per-column cuBLAS gemv launches/round). BW24_ROUTER_KERNEL=0 is the rollback seam.
pub fn router_kernel_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("BW24_ROUTER_KERNEL").as_deref() != Ok("0");
        if !on { eprintln!("[bw24] router kernel OFF (rollback: per-column cuBLAS gemv)"); }
        on
    })
}
pub mod moe_cache;
pub mod spill;
#[cfg(bw24_cutlass)]
pub mod cutlass_ffi;
pub mod mmq_ffi;
pub mod fp8_ffi;

const FATBIN_PATH: &str = env!("BW24_ENGINE_FATBIN");
const HYBRID_FATBIN_PATH: &str = env!("BW24_HYBRID_FATBIN");
const QMATVEC_FATBIN_PATH: &str = env!("BW24_QMATVEC_FATBIN");
const FLASH_FATBIN_PATH: &str = env!("BW24_FLASH_FATBIN");
const GEMM_FATBIN_PATH: &str = env!("BW24_GEMM_FATBIN");
const ROUTER_FATBIN_PATH: &str = env!("BW24_ROUTER_FATBIN");
/// spec_sample.cu: sampled-spec primitives (Philox Gumbel-max / softmax gather / residual sampler).
const SAMPLE_FATBIN_PATH: &str = env!("BW24_SAMPLE_FATBIN");

/// TUNE SEAM (tools/sweep): a RUNTIME `BW24_GEMM_FATBIN=<path>` overrides the baked-in
/// qmatvec_gemm.cu fatbin path (build.rs bakes the same name at COMPILE time via
/// cargo:rustc-env — that constant is the default). Lets the sweep harness swap in a
/// `-D`-tuned fatbin per process with NO rust rebuild. Unset at runtime => the
/// compile-time default (zero behavior change).
fn gemm_fatbin_path() -> String {
    std::env::var("BW24_GEMM_FATBIN").unwrap_or_else(|_| GEMM_FATBIN_PATH.to_string())
}

// ---- KV-cache format selection (kvbytes lane, 2026-07-08; default OFF = daily config) ----
// `BW24_KV_K` = q8_0 (default, 34 B/32elem) | fp8 (raw e4m3, 32 B — the -6% K-bytes arm)
// `BW24_KV_V` = q5_1 (default, 24 B/32elem) | q4_0 (18 B, -25% V bytes) | fp8 (32 B, +33%)
// A non-default format is a NEW NUMERIC CONFIG: its own run-gen argmax baseline is legal,
// but the gate battery (kernel-check, run-spec self-consistency) must pass WITHIN it and
// the choice is explicit env, never silent. flash_attn.cu is compiled once per format pair
// (build.rs); the kernels keep their names — Engine::new just loads the matching fatbin.
const FLASH_FATBIN_VQ4: &str = env!("BW24_FLASH_FATBIN_VQ4");
const FLASH_FATBIN_VF8: &str = env!("BW24_FLASH_FATBIN_VF8");
const FLASH_FATBIN_KF8: &str = env!("BW24_FLASH_FATBIN_KF8");
const FLASH_FATBIN_KF8VQ4: &str = env!("BW24_FLASH_FATBIN_KF8VQ4");
const FLASH_FATBIN_KF8VF8: &str = env!("BW24_FLASH_FATBIN_KF8VF8");

/// The (K, V) cache formats picked by env (cached; both the fatbin pick and every
/// tok-bytes computation MUST come through here so they can never diverge).
pub fn kv_cache_formats() -> (&'static str, &'static str) {
    static F: std::sync::OnceLock<(&'static str, &'static str)> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        let k = match std::env::var("BW24_KV_K").as_deref() {
            Ok("fp8") => "fp8",
            Ok("q8_0") | Ok("") | Err(_) => "q8_0",
            Ok(o) => panic!("BW24_KV_K={o} unsupported (q8_0 | fp8)"),
        };
        let v = match std::env::var("BW24_KV_V").as_deref() {
            Ok("q4_0") => "q4_0",
            Ok("fp8") => "fp8",
            Ok("q5_1") | Ok("") | Err(_) => "q5_1",
            Ok(o) => panic!("BW24_KV_V={o} unsupported (q5_1 | q4_0 | fp8)"),
        };
        if (k, v) != ("q8_0", "q5_1") {
            eprintln!("[bw24] KV cache format: K={k} V={v} (non-default — new numeric config)");
        }
        (k, v)
    })
}

/// Per-32-element block bytes for the selected (K, V) formats. Callers compute
/// `tok_bytes = (kv_dim/32) * blk_bytes` (cache.rs, spec.rs, eagle.rs, gates, benches).
pub fn kv_blk_bytes() -> (usize, usize) {
    let (k, v) = kv_cache_formats();
    let kb = match k { "fp8" => 32, _ => 34 };
    let vb = match v { "q4_0" => 18, "fp8" => 32, _ => 24 };
    (kb, vb)
}

/// The flash_attn fatbin matching the selected KV formats.
fn flash_fatbin_path() -> &'static str {
    match kv_cache_formats() {
        ("q8_0", "q5_1") => FLASH_FATBIN_PATH,
        ("q8_0", "q4_0") => FLASH_FATBIN_VQ4,
        ("q8_0", "fp8")  => FLASH_FATBIN_VF8,
        ("fp8",  "q5_1") => FLASH_FATBIN_KF8,
        ("fp8",  "q4_0") => FLASH_FATBIN_KF8VQ4,
        ("fp8",  "fp8")  => FLASH_FATBIN_KF8VF8,
        other => unreachable!("kv_cache_formats returned {other:?}"),
    }
}

/// TUNE SEAM (tools/sweep): kernel1 (Q8_0/Q4_K/Q5_K) launch-tile override,
/// `BW24_GEMM_K1_LAUNCH="BM,BN,NWARP"`. MUST match the `-D K1_BM/K1_BN/NWARP` the swept
/// fatbin was compiled with (the .cu tile and the host launch grid/block have to agree —
/// the hardcoded (128,128,8) in qmatvec_gemm/qmatvec_gemm_raw is the shipped default).
/// Kernel2 (Q6_K/NVFP4) launch is untouched. Unset or malformed => None => shipped
/// defaults (zero behavior change).
fn k1_launch_override() -> Option<(u32, u32, u32)> {
    static K1: std::sync::OnceLock<Option<(u32, u32, u32)>> = std::sync::OnceLock::new();
    *K1.get_or_init(|| {
        let v = std::env::var("BW24_GEMM_K1_LAUNCH").ok()?;
        let p: Vec<u32> = v.split(',').filter_map(|s| s.trim().parse().ok()).collect();
        match p.as_slice() { [bm, bn, w] => Some((*bm, *bn, *w)), _ => None }
    })
}

/// TUNE SEAM: keys per FA-decode split (`BW24_FA_SPLIT` forces a fixed size; default 64). Smaller
/// splits raise grid.y so grid = n_head_kv * n_splits fills the 82 SMs at short/mid ctx (vec path
/// launches only n_head_kv=8 CTAs per split). Swept clock-locked 2026-07-03 (graph tg128): 32 beat
/// 64 at ctx 128/512 (+0.5/+1.2%) and lost at 2048 (-3%) — BUT the adaptive 32/64 default BROKE the
/// MTP spec-decode exact-match gate (run-spec K=1/2 self-consistency FAIL with 32; PASS with 64):
/// the split count changes the combine's FP summation order, and the spec verify's batched forward
/// only argmax-matches single-step decode under the 64-split order on real prompts. Spec exactness
/// (the bigger lever) outranks a <=1.2% decode win -> default stays FIXED 64; sweeps use the env.
/// Takes t_kv so eager, _dc capture, and fa_geom_eager stay signature-compatible for future
/// adaptive retries (any retry MUST pass run-spec self-consistency first).
/// Minimum t_kv for the warp-per-token vec FA path (below it the scalar path's 4x-more-blocks
/// hides latency better — measured crossover, see `fa_decode`). Shared by fa_decode / fa_decode_dc /
/// fa_geom_eager / fa_decode_rows-eligibility (spec verify) so the kernel pick NEVER diverges
/// between eager decode and the verify (the spec-exactness law).
pub const FA_VEC_MIN_TKV: usize = 96;

fn fa_split_keys(t_kv: usize, n_head_kv: usize) -> usize {
    static S: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    if let Some(forced) = *S.get_or_init(|| {
        std::env::var("BW24_FA_SPLIT").ok().and_then(|v| v.parse().ok())
            .filter(|&s: &usize| s >= 8 && s % 8 == 0)
    }) { return forced; }
    // CTX-ADAPTIVE default (2026-07-05 40k sweep: sp32 24.5 vs sp128 26.0 tok/s = +5.8% — at
    // deep ctx the n_splits count explodes (40k/32 = 1265 splits x 8 kv-heads) and the combine
    // + partial-buffer cost dominates; at short ctx small splits fill the SMs). Exactness: split
    // size only changes the PARTITION of keys; the rows/combine order per split is fixed and the
    // gate battery (kernel-check + run-spec K=1..8) arbitrates every default change.
    //
    // SM-AWARE SHORT-CTX RUNG (2026-07-06 g7e): the 32-key rung was tuned on the 82-SM 5090.
    // On 188 SMs the vec grid (n_head_kv x n_splits CTAs) starves at short ctx — the 35B has
    // n_head_kv=2, so ctx128/split32 = 8 CTAs on 188 SMs. Measured on g7e (N=1 sweep + N=3
    // interleaved confirm): 35B ctx128 sp16 179 vs sp32 161 (+11%), ctx512 178 vs 158, ctx2048
    // flat, ctx>=4096 sp64 edges sp16 by ~3%; 27B ctx128 70.9 vs 66.3 (+7%); 9B 177 vs 163
    // (+9%). Rigs <=100 SMs keep the validated 5090 ladder EXACTLY (default unchanged there —
    // rig-divergence law: this branch is measured on 188 SMs only).
    let big_rig = fa_sm_count() >= 128;
    if big_rig {
        let _ = n_head_kv;
        if t_kv <= 2048 { 16 } else if t_kv <= 16384 { 64 } else { 128 }
    } else if n_head_kv <= 4 {
        // KV-HEAD-AWARE RUNG (2026-07-08, 5090): the 8192->32 rung was validated on kv=8 models
        // (27B/9B: 8 heads x n_splits fills 82 SMs). The 35B has n_head_kv=2 — at ctx512/sp32
        // the vec grid is 2 x 20 = 40 CTAs on 82 SMs (half idle). Measured (35B, run-gen 128tok
        // N=1 sweep + N=3 confirm): sp8 162.1 / sp16 161.3 / sp32 159.4 at short ctx.
        // DEPTH TAPER (same day, the deep-ctx lesson re-learned on this rung): sp8 at d6257 =
        // 782 splits -> combine + partial-buffer cost dominates (141.2 tok/s); the d6257 sweep
        // says sp64 = 153.0 (sp16/32 147, sp96 147.6, sp128 141). Few-kv-head models need the
        // taper EARLIER than kv=8 (per-split grid 4x thinner, same per-split combine cost).
        // Crossover hunt: sp8 vs sp64 = 156.7/155.9 at d3072, 151.7/155.6 at d4096 -> boundary 3072.
        if t_kv <= 3072 { 8 } else if t_kv <= 16384 { 64 } else { 128 }
    } else {
        if t_kv <= 8192 { 32 } else if t_kv <= 16384 { 64 } else { 128 }
    }
}

/// SM count of device 0, cached (used by fa_split_keys' rig-size rung; primary-context query,
/// same attribute Engine::batched_variant reads).
fn fa_sm_count() -> i32 {
    static N: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        cudarc::driver::result::init().ok();
        cudarc::driver::result::device::get(0)
            .and_then(|d| unsafe { cudarc::driver::result::device::get_attribute(
                d, cudarc::driver::sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) })
            .unwrap_or(82)
    })
}

/// FA-prefill kernel-name suffix for a head_dim (the template-stamped twins in flash_attn.cu):
/// 256 = the original names (qwen35 class, dispatch unchanged), 128 = `_hd128` (MiniMax-M3).
/// Any other dim errors — callers gate to sdpa_naive before dispatching FA.
fn fa_hd_suffix(head_dim: usize) -> Result<&'static str, Box<dyn std::error::Error>> {
    match head_dim {
        256 => Ok(""),
        128 => Ok("_hd128"),
        d => Err(format!("fa_prefill: no kernel stamped for head_dim={d} (only 256/128); \
                          callers must gate to sdpa_naive").into()),
    }
}

/// Quant type codes matching qmatvec.cu QType enum.
pub const QT_Q8_0: i32 = 0;
pub const QT_Q4_K: i32 = 1;
pub const QT_Q6_K: i32 = 2;
pub const QT_Q5_K: i32 = 3;
pub const QT_Q3_K: i32 = 4;
pub const QT_IQ4_XS: i32 = 5;
pub const QT_IQ3_S: i32 = 6;
pub const QT_NVFP4: i32 = 7;
/// Checkpoint-native FP8-E4M3 (BW24_ST_E4M3, lane e4m3dec): raw safetensors e4m3 weight bytes
/// [out_f, in_f] row-major (row_bytes == in_f), per-tensor f32 weight_scale in GpuTensor `scale`
/// (fused at the mmvq write / post-matmul scale_inplace). Decode = qmatvec_e4m3_mmvq (+ _b2/_b4/_b8
/// batched twins); prefill (m>=16) = the cuBLASLt FP8 GEMM on the SAME resident bytes (fp8_ffi.rs)
/// — ONE weight copy total, no Q8_0 re-encode duplicate.
pub const QT_F8_E4M3: i32 = 10;
/// Device-side tag for the A6 SPLIT-PLANE repacked NVFP4 layout (Stage-A generic kernel only;
/// GpuTensor keeps qtype=QT_NVFP4 + an `rp` flag — this tag never lives in a GpuTensor).
pub const QT_NVFP4_RP: i32 = 9;
/// Unquantized f32 weight (safetensors MoE Path A: experts dequantized to f32 host-resident).
pub const QT_F32: i32 = 8;
pub const QT_BF16: i32 = 11;   // raw bf16 rows (FULL_PREC embed gather; exact bits<<16 in-kernel)

/// Engine device context: CUDA context, stream, loaded kernel modules, cuBLASLt (via runtime::Gpu).
pub struct Engine {
    pub gpu: bw24_runtime::Gpu,
    module: Arc<CudaModule>,
    hybrid: Arc<CudaModule>,
    qmatvec: Arc<CudaModule>,
    flash: Arc<CudaModule>,
    gemm: Arc<CudaModule>,
    router: Arc<CudaModule>,
    /// Sampled-spec kernels (research/sampled-spec-impl-map.md piece A).
    sample: Arc<CudaModule>,
    /// EDGE-1 §B: one shared SLRU expert-residency cache, lazily built on first MoE dispatch under
    /// BW24_MOE_CACHE. `Mutex` makes it multi-agent safe (§E.2); the lock covers only lookup/admit/
    /// memcpy-issue (µs), NOT the GEMM, so streams still overlap. `None` => cache disabled.
    moe_cache: Mutex<Option<crate::moe_cache::MoeSlotCache>>,
    /// EDGE-1 §C.2: dedicated H2D copy stream for async prefetch (event-synced to the compute stream).
    pub copy_stream: Arc<CudaStream>,
    /// Resident CUTLASS NVFP4 prefill scratch (workspace + a_packed + sfa_linear + sfa_sw + y + alpha),
    /// allocated ONCE and grown to the largest prefill GEMM shape, then reused per-call. Removes the
    /// 6 fresh allocations + alpha htod that `cutlass_fp4_gemm` did every prefill matmul (~200/prefill).
    /// Safe as a single shared buffer because all GPU compute serializes on the one `gpu.stream` worker
    /// thread (the server runs one GPU worker; no concurrent CUTLASS GEMMs share this scratch). `None`
    /// until the first CUTLASS FP4 GEMM. Mutex guards lazy build/grow only (matches `moe_cache`).
    #[cfg(bw24_cutlass)]
    cutlass_scratch: Mutex<Option<crate::cutlass_ffi::CutlassScratch>>,
    /// FP8-ACT PREFILL scratch (BW24_PP_FP8): quantized-activation buffer + scale block + cuBLASLt
    /// workspace, allocated once and grown to the largest prefill m*k (see fp8_ffi.rs). `None`
    /// until the first FP8 prefill GEMM; Mutex guards lazy build/grow only (matches cutlass_scratch).
    fp8_scratch: Mutex<Option<crate::fp8_ffi::Fp8Scratch>>,
    /// RANK1 LEVER (parallel argmax): resident pass-1 partials scratch (part_v[NB] f32, part_i[NB] i32),
    /// allocated ONCE on first parallel-argmax call and reused. Stable pointers so the 2-pass argmax
    /// is CUDA-graph-capturable (the buffer is referenced by both captured passes; lazy-allocated
    /// before capture under the generate_graph tracking-off window so it carries no events).
    argmax_partials: Mutex<Option<(CudaSlice<f32>, CudaSlice<i32>)>>,
    /// ARC B (chunk-prime dequant-once): resident bf16 K/V workspace for `fa_prefill_view_ws`
    /// ((K bytes, V bytes) u8 buffers holding [t_kv, kv_dim] bf16). Grown lazily to the largest
    /// (t_kv, kv_dim) seen, REUSED across layers/chunks/calls (contents rewritten per launch —
    /// safe because all compute serializes on the one gpu.stream). ~82MB at 40k ctx on the 27B.
    prime_deqw_ws: Mutex<Option<(CudaSlice<u8>, CudaSlice<u8>)>>,
    /// LAUNCH-STRUCTURE STAGE 1: persistent PINNED (cacheable, flags=0) host staging buffer for the
    /// fused-router sel/w readback — one async DtoH pair + ONE sync instead of two synced dtohs.
    /// Grown lazily; reused every MoE layer (single-threaded decode serializes on the sync).
    router_stage: Mutex<Option<PinnedStage>>,
}

/// FAVENDOR lane env gate (2026-07-08): BW24_FA_V2=1 dispatches the llama-fattn-vec-mechanism
/// decode kernels (fa_decode_vec_q_v2 / fa_decode_vec_q_rows_v2 / fa_decode_vec_q_v2_dc):
/// tile-batched online softmax (one alpha rescale per 32-key tile instead of per key) + wide-load
/// block dequant in the staging phase. NOTE rev2: llama's register streaming (no smem) was ALSO
/// tried and measured 2x WORSE at depth in our gqa-warps frame — the smem KV-tile broadcast stays
/// (see the kernel comment). NEW NUMERIC CONFIG (tile-level softmax regrouping changes FP order vs
/// the per-key twins) — own argmax baseline; eager decode, the spec-verify rows path AND the
/// graph _dc path switch TOGETHER (the spec-exactness law). Default OFF. Read per call (not
/// OnceLock) so the gate battery can A/B within one process, matching the BW24_NO_FA_VEC pattern.
fn fa_v2_on() -> bool {
    // DEFAULT ON since 2026-07-08 (BW24_FA_V2=0 reverts): tile-batched online softmax, e2e
    // measured across every model x depth — 35B 168.7->173.4 (d512) / 153.1->158.5 (d6257),
    // 9B 131.2->132.7 / 108.4->124.5 (+15% — the engine-wide depth-slope fix), 27B 47.2->47.7 /
    // 42.2->44.9. One-time numeric-config change; kernel-check + argmax + spec self-consistency
    // + graph bit-identity green on all three models.
    std::env::var("BW24_FA_V2").map(|v| v != "0").unwrap_or(true)
}

/// FA v3 gate (default ON since 2026-07-09; BW24_FA_V3=0 reverts to v2 — research/fa/fa_v3_design.md):
/// HYBRID decode twins (fa_decode_vec_q_v3 / _rows_v3 / _v3_dc): llama's int8-dp4a K.Q with
/// register-quantized Q (no K dequant, no K smem) + OUR CTA-shared staged bf16 V tile + OUR
/// split partition/combine. NEW NUMERIC CONFIG (int8 Q quantization changes the K.Q accumulation
/// vs the bf16-roundtrip FMA chain) — own argmax baseline; eager decode, the spec-verify rows
/// path AND the graph _dc path switch TOGETHER (the spec-exactness law). Read per call so the
/// gate battery can A/B within one process (the BW24_FA_V2 pattern).
fn fa_v3_on() -> bool {
    // DEFAULT ON since 2026-07-09 (BW24_FA_V3=0 reverts to v2): dp4a-K hybrid FA decode —
    // fa kernel -21-23% at depth (micro), 35B spec p3 +5% (190->200, the last spec cell),
    // d6257 +1.7%. Own numeric config; full battery green on 35B+9B incl graph bit-identity.
    std::env::var("BW24_FA_V3").map(|v| v != "0").unwrap_or(true)
}

/// The v3 dp4a K path reads RAW q8_0 bytes (34B blocks) and stages q5_1 V verbatim — it is only
/// correct on the DEFAULT KV formats — and needs dpl % 4 == 0 consecutive quants per lane
/// (head_dim % 128 == 0; both daily models are hd256). All three dispatch sites share this
/// predicate so the twins can never diverge.
fn fa_v3_active(head_dim: usize) -> bool {
    fa_v3_on() && head_dim % 128 == 0 && kv_cache_formats() == ("q8_0", "q5_1")
}

/// A raw pinned (page-locked, CACHEABLE — flags=0, not write-combined) host allocation for
/// DtoH staging. cudarc's `alloc_pinned` uses CU_MEMHOSTALLOC_WRITECOMBINED, which is right for
/// HtoD streams but pathologically slow for host READS — the router readback is host-read-heavy,
/// so we allocate through `result::malloc_host` with flags=0 directly.
struct PinnedStage {
    ptr: *mut u8,
    cap: usize,
}
unsafe impl Send for PinnedStage {}
impl PinnedStage {
    fn new(cap: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let ptr = unsafe { cudarc::driver::result::malloc_host(cap, 0)? } as *mut u8;
        Ok(PinnedStage { ptr, cap })
    }
}
impl Drop for PinnedStage {
    fn drop(&mut self) {
        let _ = unsafe { cudarc::driver::result::free_host(self.ptr as _) };
    }
}

/// Number of pass-1 blocks for the parallel argmax (fan-out across SMs to saturate HBM). 256 blocks
/// x 256 threads = 65536 threads covering the 248K-vocab scan in ~4 strided loads/thread.
pub const ARGMAX_NB: usize = 256;

/// STAGE-2 GROUPED DECODE: 8 expert weight-block device pointers passed BY VALUE as one kernel
/// param (matches the CUDA `wptr8_t` struct: 8x 64-bit pointers, `#[repr(C)]` => identical
/// layout). The pointers are SLRU cache-slot base addresses — fixed for the engine's lifetime
/// (slots are never re-allocated), so passing raw values is stable across the launch.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct WPtr8(pub [u64; 8]);
unsafe impl cudarc::driver::DeviceRepr for WPtr8 {}

/// STAGE-2 GROUPED DECODE: the 8 routed-expert weights by value (CUDA `f32x8_t`).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct F32x8(pub [f32; 8]);
unsafe impl cudarc::driver::DeviceRepr for F32x8 {}

/// Harness timing contract: wall nanos of the LAST generate/generate_spec prompt prime on this
/// process. Bench binaries read it right after the call to print gen-only throughput without the
/// prime-subtraction hack (which amplifies prime jitter into the gen number at long prompts).
pub static PRIME_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl Engine {
    pub fn new(ordinal: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let gpu = bw24_runtime::Gpu::new(ordinal)?;
        let module = gpu.ctx.load_module(Ptx::from_file(FATBIN_PATH))?;
        let hybrid = gpu.ctx.load_module(Ptx::from_file(HYBRID_FATBIN_PATH))?;
        let qmatvec = gpu.ctx.load_module(Ptx::from_file(QMATVEC_FATBIN_PATH))?;
        let flash = gpu.ctx.load_module(Ptx::from_file(flash_fatbin_path()))?;
        let gemm = gpu.ctx.load_module(Ptx::from_file(gemm_fatbin_path()))?;
        let router = gpu.ctx.load_module(Ptx::from_file(ROUTER_FATBIN_PATH))?;
        let sample = gpu.ctx.load_module(Ptx::from_file(SAMPLE_FATBIN_PATH))?;
        let copy_stream = gpu.ctx.new_stream()?;
        // DECODE EVENT-TRACKING ELISION — DEFAULT ON (2026-07-05; BW24_EVT=1 = escape hatch).
        // cudarc is in multi-stream mode (main stream +
        // copy_stream are both created streams), so with tracking on EVERY launch arg records a
        // read/write CudaEvent and inserts cuStreamWaitEvent on prior events. On the 35B MoE decode
        // that is ~19k cuStreamWaitEvent + ~9k cuEventRecord + ~6k event create/destroy per token
        // (~7 ms/tok host time, measured nsys 2026-07-04 g7e), and +4.6% measured on 27B decode —
        // protecting NOTHING: every hot-path kernel/memcpy runs on the ONE gpu.stream.
        // CROSS-STREAM HAZARD AUDIT (2026-07-05, default-flip gate): copy_stream is touched by
        // exactly two sites — (a) stage_expert_async (lib.rs), which has ZERO callers (grep-verified;
        // the MoE async-prefetch stage is unbuilt), and (b) the store-before-evict barrier in
        // moe_cache::admit, gated on `prefetch_active` whose setter is never called. The graph-capture
        // sites (capture_graph, spec.rs, decode.rs) use `was_tracking` guards that read the live state,
        // so they degrade to no-ops. If the async-prefetch stage ever wires stage_expert_async, its
        // event handoff (record on copy_stream -> compute_wait on gpu.stream) is EXPLICIT and does not
        // rely on cudarc's implicit tracking — but re-audit this flip then.
        // SAFETY: single-stream ordering is total; the runtime mem-pool is configured with
        // internal-dependency reuse (bw24-runtime), so alloc reuse is stream-ordered too.
        if std::env::var("BW24_EVT").map(|v| v == "1").unwrap_or(false) {
            // escape hatch: keep cudarc's implicit cross-stream event tracking.
        } else {
            unsafe { gpu.ctx.disable_event_tracking(); }
        }
        Ok(Self { gpu, module, hybrid, qmatvec, flash, gemm, router, sample,
                  moe_cache: Mutex::new(None), copy_stream,
                  argmax_partials: Mutex::new(None),
                  prime_deqw_ws: Mutex::new(None),
                  router_stage: Mutex::new(None),
                  fp8_scratch: Mutex::new(None),
                  #[cfg(bw24_cutlass)]
                  cutlass_scratch: Mutex::new(None) })
    }

    pub fn ctx(&self) -> &Arc<CudaContext> { &self.gpu.ctx }
    pub fn stream(&self) -> &Arc<CudaStream> { &self.gpu.stream }
    fn func(&self, name: &str) -> CudaFunction {
        self.module.load_function(name)
            .or_else(|_| self.hybrid.load_function(name))
            .or_else(|_| self.qmatvec.load_function(name))
            .or_else(|_| self.flash.load_function(name))
            .or_else(|_| self.gemm.load_function(name))
            .or_else(|_| self.router.load_function(name))
            .or_else(|_| self.sample.load_function(name))
            .unwrap_or_else(|_| panic!("kernel {name} not in any fatbin"))
    }

    /// Scatter trimmed draft logits into full-vocab space: dst = -inf everywhere, then
    /// dst[d2t[i]] = src[i]. Two launches (fill, scatter) — no grid-wide sync needed.
    pub fn scatter_trim_logits(&self, src: &CudaSlice<f32>, d2t: &CudaSlice<u32>,
                               dst: &mut CudaSlice<f32>, d_vocab: usize, n_vocab: usize)
                               -> Result<(), Box<dyn std::error::Error>> {
        let f1 = self.func("scatter_trim_logits_f32");
        let f2 = self.func("scatter_trim_logits_pass2_f32");
        let (dv, nv) = (d_vocab as i32, n_vocab as i32);
        let cfg1 = LaunchConfig { grid_dim: (256, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let mut b1 = self.gpu.stream.launch_builder(&f1);
        b1.arg(src).arg(d2t).arg(&mut *dst).arg(&dv).arg(&nv);
        unsafe { b1.launch(cfg1)?; }
        let cfg2 = LaunchConfig { grid_dim: (d_vocab.div_ceil(256) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let mut b2 = self.gpu.stream.launch_builder(&f2);
        b2.arg(src).arg(d2t).arg(&mut *dst).arg(&dv);
        unsafe { b2.launch(cfg2)?; }
        Ok(())
    }

    // ---- FILTERED-SPEC (feat/filtered-spec): top-k/p/min-p transforms applied symmetrically
    // to p and q — rejection sampling stays distribution-exact for the filtered target. ----

    /// Per-row filtered-softmax stats: out[r] = (threshold_e, renorm_mass_e, row_max) for the
    /// filter (top_k, top_p, min_p) at `temp`. Rows index into x with row_stride f32s.
    #[allow(clippy::too_many_arguments)]
    pub fn filter_stats(&self, x: &CudaSlice<f32>, row_stride: usize, rows: &CudaSlice<i32>,
                        out_th: &mut CudaSlice<f32>, out_z: &mut CudaSlice<f32>,
                        out_max: &mut CudaSlice<f32>, n: usize, nrow: usize,
                        temp: f32, top_k: i32, top_p: f32, min_p: f32)
                        -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("filter_stats_f32");
        let (ni, nr, rs) = (n as i32, nrow as i32, row_stride as i64);
        let cfg = LaunchConfig { grid_dim: (nrow as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(x).arg(&rs).arg(rows).arg(&mut *out_th).arg(&mut *out_z).arg(&mut *out_max)
         .arg(&ni).arg(&nr).arg(&temp).arg(&top_k).arg(&top_p).arg(&min_p);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// out[pair] = filtered-softmax prob of ids[pair] in row rows[pair] (th/z per PAIR).
    #[allow(clippy::too_many_arguments)]
    pub fn softmax_gather_filtered(&self, x: &CudaSlice<f32>, row_stride: usize,
                                   ids: &CudaSlice<u32>, rows: &CudaSlice<i32>,
                                   th: &CudaSlice<f32>, z: &CudaSlice<f32>,
                                   out: &mut CudaSlice<f32>, n: usize, npair: usize, temp: f32)
                                   -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("softmax_gather_filtered_f32");
        let (ni, np, rs) = (n as i32, npair as i32, row_stride as i64);
        let cfg = LaunchConfig { grid_dim: (npair as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(x).arg(&rs).arg(ids).arg(rows).arg(th).arg(z).arg(&mut *out).arg(&ni).arg(&np).arg(&temp);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// Filtered residual sample: token ~ norm(max(0, fp - fq)) with fp/fq the filtered softmaxes.
    #[allow(clippy::too_many_arguments)]
    pub fn residual_sample_filtered(&self, p: &CudaSlice<f32>, q: Option<&CudaSlice<f32>>, n: usize,
                                    temp: f32, seed: u64, stream_pos: u32,
                                    p_stats: (f32, f32, f32), q_stats: (f32, f32, f32),
                                    out_tok: &mut CudaSlice<u32>)
                                    -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("residual_sample_filtered_f32");
        let (ni, slo, shi) = (n as i32, (seed & 0xFFFF_FFFF) as u32, (seed >> 32) as u32);
        let has_q: i32 = q.is_some() as i32;
        let qbuf = q.unwrap_or(p);
        let (pm, pth, pz) = p_stats; let (qm, qth, qz) = q_stats;
        let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(p).arg(qbuf).arg(&has_q).arg(&ni).arg(&temp).arg(&slo).arg(&shi).arg(&stream_pos)
         .arg(&pm).arg(&pth).arg(&pz).arg(&qm).arg(&qth).arg(&qz).arg(&mut *out_tok);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// Gumbel-max draw from the FILTERED distribution (masked perturb; argmax after).
    #[allow(clippy::too_many_arguments)]
    pub fn gumbel_perturb_filtered(&self, x: &CudaSlice<f32>, y: &mut CudaSlice<f32>, n: usize,
                                   seed: u64, stream_pos: u32, temp: f32, row_max: f32, th: f32)
                                   -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("gumbel_perturb_filtered_f32");
        let (ni, slo, shi) = (n as i32, (seed & 0xFFFF_FFFF) as u32, (seed >> 32) as u32);
        let cfg = LaunchConfig { grid_dim: (n.div_ceil(256) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(x).arg(&mut *y).arg(&ni).arg(&slo).arg(&shi).arg(&stream_pos).arg(&temp).arg(&row_max).arg(&th);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// Keskar penalties applied IN PLACE to a logits buffer: history token ids get
    /// rep-divided/multiplied + freq*count + presence subtracted. Symmetric p/q usage keeps
    /// filtered rejection sampling exact for the penalized target.
    #[allow(clippy::too_many_arguments)]
    pub fn penalize_logits(&self, x: &mut CudaSlice<f32>, hist: &CudaSlice<u32>, n_hist: usize,
                           rep: f32, freq: f32, present: f32, n: usize)
                           -> Result<(), Box<dyn std::error::Error>> {
        if n_hist == 0 { return Ok(()); }
        let f = self.func("penalize_logits_f32");
        let (nh, ni) = (n_hist as i32, n as i32);
        let cfg = LaunchConfig { grid_dim: (n_hist.div_ceil(128) as u32, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(&mut *x).arg(hist).arg(&nh).arg(&rep).arg(&freq).arg(&present).arg(&ni);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// Rows variant: penalize `nrow` contiguous rows of length n in one launch.
    #[allow(clippy::too_many_arguments)]
    pub fn penalize_logits_rows(&self, x: &mut CudaSlice<f32>, hist: &CudaSlice<u32>, n_hist: usize,
                                rep: f32, freq: f32, present: f32, n: usize, nrow: usize)
                                -> Result<(), Box<dyn std::error::Error>> {
        if n_hist == 0 || nrow == 0 { return Ok(()); }
        let f = self.func("penalize_logits_rows_f32");
        let (nh, ni, nr) = (n_hist as i32, n as i32, nrow as i32);
        let cfg = LaunchConfig { grid_dim: (n_hist.div_ceil(128) as u32, nrow as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(&mut *x).arg(hist).arg(&nh).arg(&rep).arg(&freq).arg(&present).arg(&ni).arg(&nr);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// MoE router GEMV (BW24_ROUTER_KERNEL): deterministic warp-per-(expert,token) f32 dot.
    /// Different FP order than the cuBLAS path it replaces — battery-gated numeric config.
    pub fn router_gemv(&self, w: &CudaSlice<f32>, x: &CudaSlice<f32>, n_embd: usize,
                       n_experts: usize, t: usize)
                       -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let mut y = self.alloc_uninit::<f32>(t * n_experts)?;
        let f = self.func("router_gemv_f32");
        let (ne, nx, ti) = (n_embd as i32, n_experts as i32, t as i32);
        let cfg = LaunchConfig { grid_dim: (n_experts as u32, t as u32, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(w).arg(x).arg(&mut y).arg(&ne).arg(&nx).arg(&ti);
        unsafe { b.launch(cfg)?; }
        Ok(y)
    }

    /// ROUND-STREAM stage (b): device next-round seed gather (see spec_seed_gather header).
    /// Caller D2Ds h_seed into fill_prev after (both slots carry the same value in every arm).
    pub fn spec_seed_gather(&self, vx: &CudaSlice<f32>, fill_prev: &CudaSlice<f32>,
                            acc: &CudaSlice<u32>, h_seed: &mut CudaSlice<f32>,
                            base: usize, n_embd: usize)
                            -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("spec_seed_gather");
        let (b, ne) = (base as i32, n_embd as i32);
        let cfg = LaunchConfig { grid_dim: (n_embd.div_ceil(256) as u32, 1, 1),
                                 block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let mut bl = self.gpu.stream.launch_builder(&f);
        bl.arg(vx).arg(fill_prev).arg(acc).arg(h_seed).arg(&b).arg(&ne);
        unsafe { bl.launch(cfg)?; }
        Ok(())
    }


    /// ROUND-STREAM stage (a): device greedy accept walk (see spec_accept_greedy header).
    pub fn spec_accept_greedy(&self, preds: &CudaSlice<u32>, draft: &CudaSlice<u32>,
                              last_pred: u32, base: usize, k_round: usize,
                              out: &mut CudaSlice<u32>)
                              -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("spec_accept_greedy");
        let (b, k) = (base as i32, k_round as i32);
        let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
        let mut bl = self.gpu.stream.launch_builder(&f);
        bl.arg(preds).arg(draft).arg(&last_pred).arg(&b).arg(&k).arg(out);
        unsafe { bl.launch(cfg)?; }
        Ok(())
    }

    // ================= SAMPLED-SPEC PRIMITIVES (spec_sample.cu, piece A) =================
    // Counter-based randomness: every call takes (seed, stream_pos) — the caller owns the
    // event counter (one per sampled token). temp <= 0 arms are exact greedy limits.

    /// y = x/temp + Gumbel(Philox(seed, stream_pos)) over n logits (then run device argmax on y
    /// = one categorical sample at temperature `temp`). temp<=0: y = x (pure copy).
    pub fn gumbel_perturb(&self, x: &CudaSlice<f32>, y: &mut CudaSlice<f32>, n: usize,
                          seed: u64, stream_pos: u32, temp: f32)
                          -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("gumbel_perturb_f32");
        let (ni, slo, shi) = (n as i32, (seed & 0xFFFF_FFFF) as u32, (seed >> 32) as u32);
        let cfg = LaunchConfig { grid_dim: (n.div_ceil(256) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(x).arg(&mut *y).arg(&ni).arg(&slo).arg(&shi).arg(&stream_pos).arg(&temp);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// In-graph sampling-event counter bump (spec_sample.cu kernel 5): ctr[0] += 1. The sampled
    /// graph-draft chain replays with FIXED kernel args, so the Philox event counter must be
    /// DEVICE data — the host seeds it once per round; every replay bumps it before the perturb
    /// reads it (counter is data, not state — graph-replay-safe).
    pub fn sctr_inc(&self, ctr: &mut CudaSlice<u32>) -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("bw24_sctr_inc");
        let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 };
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(&mut *ctr);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// Graph-capturable `gumbel_perturb`: the sampling-event counter comes from DEVICE memory
    /// (`ctr[0]`) instead of a host scalar. Identical math to `gumbel_perturb` at
    /// stream_pos == ctr[0] (same Philox call, same lane mapping) — the eager and graph sampled
    /// chains produce bit-identical perturbations for the same (seed, counter, temp).
    pub fn gumbel_perturb_ctr(&self, x: &CudaSlice<f32>, y: &mut CudaSlice<f32>, n: usize,
                              seed: u64, ctr: &CudaSlice<u32>, temp: f32)
                              -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("gumbel_perturb_ctr_f32");
        let (ni, slo, shi) = (n as i32, (seed & 0xFFFF_FFFF) as u32, (seed >> 32) as u32);
        let cfg = LaunchConfig { grid_dim: (n.div_ceil(256) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(x).arg(&mut *y).arg(&ni).arg(&slo).arg(&shi).arg(ctr).arg(&temp);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// out[pair] = softmax_temp(x[rows[pair]])[ids[pair]] for npair (row, id) pairs; rows index
    /// into x with `row_stride` f32s per row. temp<=0: out = 1.0 iff id is the row argmax
    /// (smallest-index tie-break — matches the argmax-gate contract).
    pub fn softmax_gather(&self, x: &CudaSlice<f32>, row_stride: usize,
                          ids: &CudaSlice<u32>, rows: &CudaSlice<i32>,
                          out: &mut CudaSlice<f32>, n: usize, npair: usize, temp: f32)
                          -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("softmax_gather_f32");
        let (ni, rs) = (n as i32, row_stride as i64);
        let np = npair as i32;
        let cfg = LaunchConfig { grid_dim: (npair as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(x).arg(&rs).arg(ids).arg(rows).arg(&mut *out).arg(&ni).arg(&np).arg(&temp);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// Sample token from norm(max(0, softmax_temp(p) - softmax_temp(q))) (q = None -> plain
    /// categorical from softmax_temp(p)). Row stats (max, sumexp at temp) must be precomputed
    /// (softmax_gather's pass-1 values; see spec.rs caller). Deterministic fixed-order CDF walk.
    pub fn residual_sample(&self, p: &CudaSlice<f32>, q: Option<&CudaSlice<f32>>, n: usize,
                           temp: f32, seed: u64, stream_pos: u32,
                           out_tok: &mut CudaSlice<u32>)
                           -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("residual_sample_f32");
        let (ni, slo, shi) = (n as i32, (seed & 0xFFFF_FFFF) as u32, (seed >> 32) as u32);
        let nth = 1024u32;
        let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (nth, 1, 1), shared_mem_bytes: 0 };
        let has_q: i32 = q.is_some() as i32;
        let qbuf = q.unwrap_or(p);   // dummy when absent; kernel gates on has_q
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(p).arg(qbuf).arg(&has_q).arg(&ni).arg(&temp).arg(&slo).arg(&shi).arg(&stream_pos)
         .arg(&mut *out_tok);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// Access the shared MoE residency cache (EDGE-1 §B), building it on first use under
    /// BW24_MOE_CACHE. The closure runs while the lock is held — keep it to lookup/admit/issue, not
    /// the GEMM. `max_block_bytes` sizes the slots (largest of gate/up/down). Returns the closure's
    /// result. If BW24_MOE_CACHE is unset this is never called (the caller checks the env first).
    pub fn with_moe_cache<R>(&self, max_block_bytes: usize,
                             f: impl FnOnce(&mut crate::moe_cache::MoeSlotCache, &Engine) -> Result<R, Box<dyn std::error::Error>>)
                             -> Result<R, Box<dyn std::error::Error>> {
        let mut guard = self.moe_cache.lock().unwrap();
        if guard.is_none() {
            *guard = Some(crate::moe_cache::MoeSlotCache::new(self, max_block_bytes)?);
        }
        let cache = guard.as_mut().unwrap();
        f(cache, self)
    }

    /// True if the MoE residency cache is enabled (BW24_MOE_CACHE set).
    pub fn moe_cache_enabled() -> bool { std::env::var("BW24_MOE_CACHE").as_deref() != Ok("0") }

    /// Snapshot the MoE cache counters (hits, misses, staged_bytes, n_slots) for the §D.4 PCIe gate.
    /// Returns None if the cache was never built (disabled or no MoE forward ran).
    pub fn moe_cache_stats(&self) -> Option<(u64, u64, u64, usize)> {
        let guard = self.moe_cache.lock().unwrap();
        guard.as_ref().map(|c| (c.hits, c.misses, c.staged_bytes, c.n_slots()))
    }

    /// Reset the MoE cache perf counters (to separate warmup from steady-state windows).
    pub fn moe_cache_reset_counters(&self) {
        if let Some(c) = self.moe_cache.lock().unwrap().as_mut() { c.reset_counters(); }
    }

    pub fn htod_bytes(&self, v: &[u8]) -> Result<CudaSlice<u8>, Box<dyn std::error::Error>> {
        Ok(self.gpu.stream.clone_htod(v)?)
    }

    /// Device-to-device copy of `src` into `dst[off..off+len]` (f32). For in-place KV append.
    pub fn copy_into(&self, dst: &mut CudaSlice<f32>, off: usize, src: &CudaSlice<f32>, len: usize)
                     -> Result<(), Box<dyn std::error::Error>> {
        let mut view = dst.slice_mut(off..off + len);
        self.gpu.stream.memcpy_dtod(&src.slice(0..len), &mut view)?;
        Ok(())
    }

    /// View a sub-range of a device buffer (for attending over [0..len) of a KV cache).
    pub fn view<'a>(&self, b: &'a CudaSlice<f32>, len: usize) -> cudarc::driver::CudaView<'a, f32> {
        b.slice(0..len)
    }

    /// View the first `len` BYTES of a u8 device buffer (quantized KV cache: [0..t_kv*tok_bytes)).
    pub fn view_u8<'a>(&self, b: &'a CudaSlice<u8>, len: usize) -> cudarc::driver::CudaView<'a, u8> {
        b.slice(0..len)
    }

    /// Append-quantize ONE token's post-RoPE K (q8_0) and V (q5_1) into the resident byte caches at
    /// token index `t` (KVQUANT-PLAN §C). One CTA (one warp) per 32-element block; the kernel writes
    /// the f16 scale(s) + packed quants for K and V. k_row/v_row are f32 [kv_dim_k]/[kv_dim_v].
    pub fn append_kv_quantized(&self, k_row: &CudaSlice<f32>, v_row: &CudaSlice<f32>,
                               kc: &mut CudaSlice<u8>, vc: &mut CudaSlice<u8>, t: usize,
                               kv_dim_k: usize, kv_dim_v: usize,
                               k_tok_bytes: usize, v_tok_bytes: usize)
                               -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("append_quantize_kv_q8_0_q5_1");
        let nblk = (kv_dim_k.max(kv_dim_v) / 32) as u32;
        let cfg = LaunchConfig { grid_dim: (nblk, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
        let (ti, kdk, kdv) = (t as i32, kv_dim_k as i32, kv_dim_v as i32);
        let (ktb, vtb) = (k_tok_bytes as i64, v_tok_bytes as i64);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(k_row).arg(v_row).arg(kc).arg(vc).arg(&ti).arg(&kdk).arg(&kdv).arg(&ktb).arg(&vtb);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// Device-counter variant of `append_kv_quantized` (CUDA-GRAPH-PLAN Phase 2): the write slot
    /// `t` is read from `t_dev[0]` (a resident device i32[1]) instead of a host int arg, so the
    /// launch args are FIXED across decode steps (graph-capturable). Identical quant math.
    pub fn append_kv_quantized_dc(&self, k_row: &CudaSlice<f32>, v_row: &CudaSlice<f32>,
                                  kc: &mut CudaSlice<u8>, vc: &mut CudaSlice<u8>, t_dev: &CudaSlice<i32>,
                                  kv_dim_k: usize, kv_dim_v: usize,
                                  k_tok_bytes: usize, v_tok_bytes: usize)
                                  -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("append_quantize_kv_q8_0_q5_1_dc");
        let nblk = (kv_dim_k.max(kv_dim_v) / 32) as u32;
        let cfg = LaunchConfig { grid_dim: (nblk, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
        let (kdk, kdv) = (kv_dim_k as i32, kv_dim_v as i32);
        let (ktb, vtb) = (k_tok_bytes as i64, v_tok_bytes as i64);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(k_row).arg(v_row).arg(kc).arg(vc).arg(t_dev).arg(&kdk).arg(&kdv).arg(&ktb).arg(&vtb);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// Append-quantize T token rows in one shot (BATCHED PROMPT PRIME). k_rows/v_rows are
    /// token-major [T, kv_dim] post-RoPE f32; rows land at cache slots t0..t0+T. Default = the
    /// batched `_rows` kernel: one (nblk, T) launch whose per-(block,token) warp program is the
    /// per-token append kernel verbatim -> every written row is BIT-IDENTICAL to T sequential
    /// `append_kv_quantized_view` calls (kernel_check pins the bytes). BW24_PRIME_APPEND_LOOP=1
    /// forces the T-launch per-row loop (the A/B seam that measured the launch overhead).
    #[allow(clippy::too_many_arguments)]
    pub fn append_kv_quantized_rows(&self, k_rows: &CudaSlice<f32>, v_rows: &CudaSlice<f32>,
                                    kc: &mut CudaSlice<u8>, vc: &mut CudaSlice<u8>,
                                    t0: usize, t: usize, kv_dim_k: usize, kv_dim_v: usize,
                                    k_tok_bytes: usize, v_tok_bytes: usize)
                                    -> Result<(), Box<dyn std::error::Error>> {
        if std::env::var("BW24_PRIME_APPEND_LOOP").is_ok() {
            for i in 0..t {
                let k_row = k_rows.slice(i * kv_dim_k..(i + 1) * kv_dim_k);
                let v_row = v_rows.slice(i * kv_dim_v..(i + 1) * kv_dim_v);
                self.append_kv_quantized_view(&k_row, &v_row, kc, vc, t0 + i,
                                              kv_dim_k, kv_dim_v, k_tok_bytes, v_tok_bytes)?;
            }
            return Ok(());
        }
        let f = self.func("append_quantize_kv_q8_0_q5_1_rows");
        let nblk = (kv_dim_k.max(kv_dim_v) / 32) as u32;
        let cfg = LaunchConfig { grid_dim: (nblk, t as u32, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
        let (t0i, kdk, kdv) = (t0 as i32, kv_dim_k as i32, kv_dim_v as i32);
        let (ktb, vtb) = (k_tok_bytes as i64, v_tok_bytes as i64);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(k_rows).arg(v_rows).arg(kc).arg(vc).arg(&t0i).arg(&kdk).arg(&kdv).arg(&ktb).arg(&vtb);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// Increment a device i32[1] counter in place (p[0] += 1) via the resident `inc_i32` kernel.
    /// Used to advance the device-resident seqlen/pos counters inside the decode-dc path (and,
    /// later, inside a captured graph) without a host round-trip.
    pub fn inc_seqlen(&self, p: &mut CudaSlice<i32>) -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("inc_i32");
        let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 };
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(p);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// Like `append_kv_quantized` but k_row/v_row are CudaViews (one token's row sliced out of a
    /// token-major [T, kv_dim] activation buffer — the MTP verify path appends T tokens).
    pub fn append_kv_quantized_view(&self, k_row: &cudarc::driver::CudaView<f32>,
                                    v_row: &cudarc::driver::CudaView<f32>,
                                    kc: &mut CudaSlice<u8>, vc: &mut CudaSlice<u8>, t: usize,
                                    kv_dim_k: usize, kv_dim_v: usize,
                                    k_tok_bytes: usize, v_tok_bytes: usize)
                                    -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("append_quantize_kv_q8_0_q5_1");
        let nblk = (kv_dim_k.max(kv_dim_v) / 32) as u32;
        let cfg = LaunchConfig { grid_dim: (nblk, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
        let (ti, kdk, kdv) = (t as i32, kv_dim_k as i32, kv_dim_v as i32);
        let (ktb, vtb) = (k_tok_bytes as i64, v_tok_bytes as i64);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(k_row).arg(v_row).arg(kc).arg(vc).arg(&ti).arg(&kdk).arg(&kdv).arg(&ktb).arg(&vtb);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// Device-to-device copy of a CudaView `src` into `dst[off..off+len]` (f32). Like `copy_into`
    /// but the source is a sub-view (e.g. one column of a token-major activation buffer).
    pub fn copy_view_into(&self, dst: &mut CudaSlice<f32>, off: usize,
                          src: &cudarc::driver::CudaView<f32>, len: usize)
                          -> Result<(), Box<dyn std::error::Error>> {
        let mut view = dst.slice_mut(off..off + len);
        self.gpu.stream.memcpy_dtod(&src.slice(0..len), &mut view)?;
        Ok(())
    }

    /// Real device-to-device COPY of `src` into a freshly allocated buffer (NOT an Arc clone).
    /// Used for cache snapshots (MTP-PLAN §D.4): `CudaSlice::clone()` only bumps a refcount and
    /// would alias the live buffer; this allocs new device memory and memcpy_dtod's the contents.
    pub fn clone_dtod(&self, src: &CudaSlice<f32>) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let mut dst = self.gpu.stream.alloc_zeros::<f32>(src.len())?;
        self.gpu.stream.memcpy_dtod(src, &mut dst)?;
        Ok(dst)
    }

    /// Resident-quantized linear (Stage-A: f32 dequant-in-kernel). y[m,out]=x[m,in]@W[out,in]^T.
    pub fn qmatvec(&self, w: &CudaSlice<u8>, x: &CudaSlice<f32>, m: usize, in_f: usize, out_f: usize,
                   qtype: i32, row_bytes: usize) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let f = self.func("qmatvec_f32");
        let mut y = self.alloc_uninit::<f32>(m * out_f)?;  // full-overwrite output: skip memset
        let cfg = LaunchConfig { grid_dim: (out_f as u32, m as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let (inf, outf, mi, qt, rb) = (in_f as i32, out_f as i32, m as i32, qtype, row_bytes as i64);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(w).arg(x).arg(&mut y).arg(&inf).arg(&outf).arg(&mi).arg(&qt).arg(&rb);
        unsafe { b.launch(cfg)?; }
        Ok(y)
    }

    /// Allocate a reusable u8 GPU scratch buffer (for staged expert weights).
    pub fn alloc_u8(&self, n: usize) -> Result<CudaSlice<u8>, Box<dyn std::error::Error>> {
        Ok(self.gpu.stream.alloc_zeros::<u8>(n)?)
    }

    /// Uninitialized u8 scratch — skips alloc_zeros' memset. ONLY for staging buffers whose read
    /// range is fully overwritten by a stage_expert H2D before any kernel reads it (LAUNCH-STRUCTURE
    /// STAGE 2: the per-layer MoE scratch trio was 3 dead ~1MB memsets per layer per decode token).
    pub fn alloc_u8_uninit(&self, n: usize) -> Result<CudaSlice<u8>, Box<dyn std::error::Error>> {
        Ok(unsafe { self.gpu.stream.alloc::<u8>(n)? })
    }

    /// Zero a SUB-RANGE of an f32 buffer (CudaViewMut) — the row-sized memset the moe_out
    /// memset-elision uses for tokens that fall off the gdec fast path (LAUNCH-STRUCTURE STAGE 2).
    pub fn memset_zeros_view(&self, dst: &mut cudarc::driver::CudaViewMut<f32>)
                             -> Result<(), Box<dyn std::error::Error>> {
        self.gpu.stream.memset_zeros(dst)?;
        Ok(())
    }

    /// EDGE-1 staging: copy `host_bytes` (a sub-slice of a HostExps buffer) into `scratch`
    /// at byte offset `off` (async H2D on the default stream). Length is host_bytes.len().
    /// The qmatvec_view that reads `scratch[off..]` is enqueued on the SAME stream after this,
    /// so ordering is guaranteed without an explicit sync (Stage-1; Stage-2 prefetch on a 2nd
    /// stream would require an event).
    pub fn stage_expert(&self, host_bytes: &[u8], scratch: &mut CudaSlice<u8>, off: usize)
                        -> Result<(), Box<dyn std::error::Error>> {
        let mut dst = scratch.slice_mut(off..off + host_bytes.len());  // CudaViewMut<u8>
        self.gpu.stream.memcpy_htod(host_bytes, &mut dst)?;            // accepts &[u8] HostSlice src
        Ok(())
    }

    /// EDGE-1 §A: fused MoE router. `logits` is the router output [t, n_expert] (device, f32, the
    /// `gate_inp @ z` result). Returns (sel_idx [t, n_used] i32, sel_w [t, n_used] f32): the top-k
    /// expert ids (DESC by prob, ascending-index tiebreak) and renormalized weights. Replaces the
    /// host dtoh + softmax-256 + stable DESC top-8 sort + renorm (hybrid_forward.rs ~281-298).
    /// One CTA per token row, 256 threads (one per expert).
    pub fn moe_router_topk(&self, logits: &CudaSlice<f32>, t: usize, n_expert: usize, n_used: usize)
                           -> Result<(CudaSlice<i32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let f = self.func("moe_router_topk_f32");
        let mut sel_idx = self.alloc_uninit::<i32>(t * n_used)?;  // kernel fully overwrites
        let mut sel_w = self.alloc_uninit::<f32>(t * n_used)?;    // kernel fully overwrites
        let cfg = LaunchConfig { grid_dim: (t as u32, 1, 1), block_dim: (n_expert as u32, 1, 1),
                                 shared_mem_bytes: 0 };
        let (ne, nu) = (n_expert as i32, n_used as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(logits).arg(&mut sel_idx).arg(&mut sel_w).arg(&ne).arg(&nu);
        unsafe { b.launch(cfg)?; }
        Ok((sel_idx, sel_w))
    }

    /// LAUNCH-STRUCTURE STAGE 1 (2026-07-05): fused router + SINGLE-SYNC host readback. The old
    /// BW24_FUSED_ROUTER path lost 2% at t=1 because it paid TWO full stream syncs (dtoh_i32 then
    /// dtoh, each = clone_dtoh + synchronize) + two alloc_zeros memsets per MoE layer, where the
    /// host route pays ONE sync on the 1KB logits dtoh. This variant: uninit outputs (kernel fully
    /// overwrites), both DtoH copies issued ASYNC into a persistent PINNED host staging buffer
    /// (flags=0 — cacheable, NOT cudarc's WRITECOMBINED default, so the host-side reads of sel/w
    /// stay cached), then ONE synchronize. Numerics identical to `moe_router_topk` (same kernel).
    pub fn moe_router_topk_host(&self, logits: &CudaSlice<f32>, t: usize, n_expert: usize, n_used: usize)
                                -> Result<(Vec<u32>, Vec<f32>), Box<dyn std::error::Error>> {
        let f = self.func("moe_router_topk_f32");
        let n = t * n_used;
        let mut sel_idx = self.alloc_uninit::<i32>(n)?;
        let mut sel_w = self.alloc_uninit::<f32>(n)?;
        let cfg = LaunchConfig { grid_dim: (t as u32, 1, 1), block_dim: (n_expert as u32, 1, 1),
                                 shared_mem_bytes: 0 };
        let (ne, nu) = (n_expert as i32, n_used as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(logits).arg(&mut sel_idx).arg(&mut sel_w).arg(&ne).arg(&nu);
        unsafe { b.launch(cfg)?; }
        // single-sync readback: sel (i32) at offset 0, w (f32) at offset n*4 of the pinned stage.
        let bytes = n * 8;
        let mut guard = self.router_stage.lock().unwrap();
        if guard.as_ref().map(|p| p.cap < bytes).unwrap_or(true) {
            *guard = Some(PinnedStage::new(bytes.max(4096))?);
        }
        let stage = guard.as_mut().unwrap();
        let (si, sw) = unsafe {
            (std::slice::from_raw_parts_mut(stage.ptr as *mut i32, n),
             std::slice::from_raw_parts_mut(stage.ptr.add(n * 4) as *mut f32, n))
        };
        self.gpu.stream.memcpy_dtoh(&sel_idx, si)?;   // async (pinned dst)
        self.gpu.stream.memcpy_dtoh(&sel_w, sw)?;     // async (pinned dst)
        self.gpu.stream.synchronize()?;               // ONE sync for both
        Ok((si.iter().map(|&i| i as u32).collect(), sw.to_vec()))
    }

    /// EDGE-1 §C.2: async H2D of `host_bytes` into `scratch[off..]` on the COPY stream, returning a
    /// recorded event the compute stream can `wait` on before the dependent GEMM. Used for in-token
    /// expert prefetch (pipeline by one). `host_bytes` should be pinned for a true DMA (§C.1).
    pub fn stage_expert_async(&self, host_bytes: &[u8], scratch: &mut CudaSlice<u8>, off: usize)
                              -> Result<cudarc::driver::CudaEvent, Box<dyn std::error::Error>> {
        let mut dst = scratch.slice_mut(off..off + host_bytes.len());
        self.copy_stream.memcpy_htod(host_bytes, &mut dst)?;
        Ok(self.copy_stream.record_event(None)?)
    }

    /// Make the compute stream wait for an async copy event (the consumer side of `stage_expert_async`).
    pub fn compute_wait(&self, ev: &cudarc::driver::CudaEvent) -> Result<(), Box<dyn std::error::Error>> {
        self.gpu.stream.wait(ev)?;
        Ok(())
    }

    /// qmatvec over a byte sub-range of a (resident/scratch) CudaSlice<u8> holding ONE expert
    /// matrix. x is a CudaView<f32> (a sliced row of z, or a sliced activation). Reuses the
    /// validated qmatvec_f32 dequant path (NOT a fast path — the correctness gate). The
    /// CudaView base+offset pointer is honored by the launch arg.
    pub fn qmatvec_view(&self, w: &CudaSlice<u8>, range: std::ops::Range<usize>,
                        x: &cudarc::driver::CudaView<f32>, m: usize, in_f: usize, out_f: usize,
                        qtype: i32, row_bytes: usize)
                        -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let f = self.func("qmatvec_f32");
        let wv = w.slice(range);  // CudaView<u8>, offset honored
        let mut y = self.alloc_uninit::<f32>(m * out_f)?;  // full-overwrite output: skip memset
        let cfg = LaunchConfig { grid_dim: (out_f as u32, m as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let (inf, outf, mi, qt, rb) = (in_f as i32, out_f as i32, m as i32, qtype, row_bytes as i64);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(&wv).arg(x).arg(&mut y).arg(&inf).arg(&outf).arg(&mi).arg(&qt).arg(&rb);
        unsafe { b.launch(cfg)?; }
        Ok(y)
    }

    /// STAGE-2 GROUPED DECODE (2026-07-04): one MoE layer's gate+up+SiLU for all `n_used` routed
    /// experts of ONE token in ONE launch (replaces 8x qmatvec(gate) + 8x qmatvec(up) + 8x
    /// silu_mul = 24 launches). `gp`/`up` are the 8 expert weight-block device pointers (SLRU
    /// cache slots — fixed-address, stable for the launch). Returns act [n_used, n_ff].
    /// BIT-IDENTICAL to the sequential chain: each dot reproduces qmatvec_f32's exact 256-thread
    /// reduction; the SiLU epilogue is silu_mul_f32's exact expression (see kernel header).
    #[allow(clippy::too_many_arguments)]
    /// dp4a q8 twins (MoE expert dp4a arc, 2026-07-06): same contract as the _f32 versions but
    /// consume a PRE-QUANTIZED q8_1 activation. FP-order differs from _f32 (int dot + warp tree)
    /// — the argmax/stream-identity battery arbitrates; BW24_MOE_Q8=0 restores f32.
    pub fn moe_gate_up_silu8_q8(&self, gp: WPtr8, up: WPtr8,
                                aq: &CudaSlice<i8>, ad: &CudaSlice<f32>,
                                in_f: usize, n_ff: usize, n_used: usize, qt_g: i32, qt_u: i32,
                                rb_g: usize, rb_u: usize)
                                -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let f = self.func("moe_gate_up_silu8_q8");
        let mut act = self.alloc_uninit::<f32>(n_used * n_ff)?;
        let cfg = LaunchConfig { grid_dim: (n_ff as u32, n_used as u32, 1),
                                 block_dim: (32, 1, 1), shared_mem_bytes: 0 };
        let (inf, nff, rbg, rbu) = (in_f as i32, n_ff as i32, rb_g as i64, rb_u as i64);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(&gp).arg(&up).arg(aq).arg(ad).arg(&mut act)
         .arg(&inf).arg(&nff).arg(&qt_g).arg(&qt_u).arg(&rbg).arg(&rbu);
        unsafe { b.launch(cfg)?; }
        Ok(act)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn moe_down8_fma_q8(&self, dp: WPtr8, w: F32x8,
                            aq2: &CudaSlice<i8>, ad2: &CudaSlice<f32>,
                            dst: &mut cudarc::driver::CudaViewMut<f32>,
                            in_f: usize, out_f: usize, n_used: usize, qt: i32, rb: usize)
                            -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("moe_down8_fma_q8");
        let cfg = LaunchConfig { grid_dim: (out_f as u32, 1, 1),
                                 block_dim: (32, 1, 1), shared_mem_bytes: 0 };
        let (inf, outf, nu, rbi) = (in_f as i32, out_f as i32, n_used as i32, rb as i64);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(&dp).arg(&w).arg(aq2).arg(ad2).arg(dst)
         .arg(&inf).arg(&outf).arg(&nu).arg(&qt).arg(&rbi);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// q8 sequential expert matvec (staged path twin of qmatvec_view for IQ3_S/IQ4_XS).
    pub fn qmatvec_expert_q8(&self, w: &CudaSlice<u8>, range: std::ops::Range<usize>,
                             aq: &CudaSlice<i8>, ad: &CudaSlice<f32>, m: usize,
                             in_f: usize, out_f: usize, qtype: i32, row_bytes: usize)
                             -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let f = self.func("qmatvec_expert_q8");
        let wv = w.slice(range);
        let mut y = self.alloc_uninit::<f32>(m * out_f)?;
        const ROWS: u32 = 4;   // BW24_MMVQ_ROWS
        let cfg = LaunchConfig { grid_dim: ((out_f as u32 + ROWS - 1) / ROWS, m as u32, 1),
                                 block_dim: (32, ROWS, 1), shared_mem_bytes: 0 };
        let (inf, outf, mi, rbi) = (in_f as i32, out_f as i32, m as i32, row_bytes as i64);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(&wv).arg(aq).arg(ad).arg(&mut y).arg(&inf).arg(&outf).arg(&mi).arg(&qtype).arg(&rbi);
        unsafe { b.launch(cfg)?; }
        Ok(y)
    }

    pub fn moe_gate_up_silu8(&self, gp: WPtr8, up: WPtr8, x: &cudarc::driver::CudaView<f32>,
                             in_f: usize, n_ff: usize, n_used: usize, qt_g: i32, qt_u: i32,
                             rb_g: usize, rb_u: usize)
                             -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let f = self.func("moe_gate_up_silu8_f32");
        let mut act = self.alloc_uninit::<f32>(n_used * n_ff)?;  // fully overwritten
        let cfg = LaunchConfig { grid_dim: (n_ff as u32, n_used as u32, 1),
                                 block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let (inf, nff, rbg, rbu) = (in_f as i32, n_ff as i32, rb_g as i64, rb_u as i64);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(&gp).arg(&up).arg(x).arg(&mut act)
         .arg(&inf).arg(&nff).arg(&qt_g).arg(&qt_u).arg(&rbg).arg(&rbu);
        unsafe { b.launch(cfg)?; }
        Ok(act)
    }

    /// STAGE-2 GROUPED DECODE: one MoE layer's down-proj + weighted accumulation for all `n_used`
    /// routed experts in ONE launch (replaces 8x qmatvec(down) + 8x axpy = 16 launches), writing
    /// the token's moe_out row DIRECTLY (`dst` is the zeroed row; the in-kernel slot-ordered
    /// __fmaf_rn chain starting at 0.0f reproduces the sequential axpy_f32 accumulation into the
    /// zeroed row bit-for-bit — the A2 byte-identity scheme at m=1).
    #[allow(clippy::too_many_arguments)]
    pub fn moe_down8_fma_into(&self, dp: WPtr8, w: F32x8, act: &CudaSlice<f32>,
                              dst: &mut cudarc::driver::CudaViewMut<f32>,
                              in_f: usize, out_f: usize, n_used: usize, qt: i32, rb: usize)
                              -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("moe_down8_fma_f32");
        let cfg = LaunchConfig { grid_dim: (out_f as u32, 1, 1),
                                 block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let (inf, outf, nu, rbv) = (in_f as i32, out_f as i32, n_used as i32, rb as i64);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(&dp).arg(&w).arg(act).arg(dst).arg(&inf).arg(&outf).arg(&nu).arg(&qt).arg(&rbv);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// LAUNCH-STRUCTURE STAGE 3: device-dispatch twin of `moe_gate_up_silu8` for FULLY-RESIDENT
    /// layers. The expert ids come from the router kernel's DEVICE `sel` output (no DtoH) and the
    /// weight pointers from the per-layer device table `[3, n_expert]` of slot base addresses.
    /// BIT-IDENTICAL math (same grid/block/reduction; only the pointer/id source differs).
    #[allow(clippy::too_many_arguments)]
    /// dp4a q8 twin of the _dev pair (resident-experts arc).
    ///
    /// GEOMETRY VARIANTS (multirow/occupancy arc 2026-07-05): all outputs are BIT-IDENTICAL to
    /// the base one-warp-per-(row,slot) kernel (same expert_dot_g g-order + warp tree per row;
    /// down's FMA chain stays slot-ordered serial). Seams:
    ///   BW24_MOE_DEVQ8_GU   = 0(base) | 1 | 2 | 4 -> _r{1,2,4} multirow twin (RPW rows/warp)
    ///                       | s2 (gate/up warp split) | s2z (s2 + WPB rows packed per block)
    ///                       | gs4 (gate/up x low/high-group 4-warp split, nsb==64 only)
    ///                       | u64 (nsb==64 unrolled ILP twin, geometry unchanged)
    ///   BW24_MOE_DEVQ8_WPB  = warps per block for _r twins / z-rows for s2z (default 4)
    ///   BW24_MOE_DEVQ8_DOWN = auto(default: w8h2 when in_f==512 & n_used<=8 — measured +3.8%
    ///                       decode on 35B/G7e) | 0 (base one-warp serial-slot) | 1 | 2 | 4 ->
    ///                       _w8r{1,2,4} slot-parallel twin | h2 (half-warp dual-row, nsb==16
    ///                       only) | w8h2 (h2 x slot-parallel)
    #[allow(clippy::too_many_arguments)]
    /// MoE PREFILL pair-batch matvec: one launch covers all (token,expert) pairs for one proj.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_pairs_matvec_q8(&self, table: &CudaSlice<u64>, proj: i32,
                               pair_tok: &CudaSlice<i32>, pair_ex: &CudaSlice<i32>,
                               aq: &CudaSlice<i8>, ad: &CudaSlice<f32>,
                               in_f: usize, out_f: usize, n_expert: usize, n_pairs: usize,
                               qtype: i32, row_bytes: usize)
                               -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let f = self.func("moe_pairs_matvec_q8");
        let mut y = self.alloc_uninit::<f32>(n_pairs * out_f)?;
        const ROWS: u32 = 4;
        let cfg = LaunchConfig { grid_dim: ((out_f as u32 + ROWS - 1) / ROWS, n_pairs as u32, 1),
                                 block_dim: (32, ROWS, 1), shared_mem_bytes: 0 };
        let (inf, outf, ne, np, rbi) = (in_f as i32, out_f as i32, n_expert as i32,
                                        n_pairs as i32, row_bytes as i64);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(table).arg(&proj).arg(pair_tok).arg(pair_ex).arg(aq).arg(ad).arg(&mut y)
         .arg(&inf).arg(&outf).arg(&ne).arg(&np).arg(&qtype).arg(&rbi);
        unsafe { b.launch(cfg)?; }
        Ok(y)
    }

    /// Expert-major pair matvec (weight-reuse across each expert's token group).
    #[allow(clippy::too_many_arguments)]
    pub fn moe_pairs_matvec_q8_em(&self, table: &CudaSlice<u64>, proj: i32,
                                  ex_ids: &CudaSlice<i32>, ex_off: &CudaSlice<i32>,
                                  ex_pairs: &CudaSlice<i32>, pair_tok: &CudaSlice<i32>,
                                  aq: &CudaSlice<i8>, ad: &CudaSlice<f32>,
                                  in_f: usize, out_f: usize, n_expert: usize, n_active: usize,
                                  n_pairs: usize, qtype: i32, row_bytes: usize)
                                  -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let f = self.func("moe_pairs_matvec_q8_em");
        let mut y = self.alloc_uninit::<f32>(n_pairs * out_f)?;
        const ROWS: u32 = 4;
        let cfg = LaunchConfig { grid_dim: ((out_f as u32 + ROWS - 1) / ROWS, n_active as u32, 1),
                                 block_dim: (32, ROWS, 1), shared_mem_bytes: 0 };
        let (inf, outf, ne, na, rbi) = (in_f as i32, out_f as i32, n_expert as i32,
                                        n_active as i32, row_bytes as i64);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(table).arg(&proj).arg(ex_ids).arg(ex_off).arg(ex_pairs).arg(pair_tok)
         .arg(aq).arg(ad).arg(&mut y)
         .arg(&inf).arg(&outf).arg(&ne).arg(&na).arg(&qtype).arg(&rbi);
        unsafe { b.launch(cfg)?; }
        Ok(y)
    }

    // Decode-once expert-major MMQ (rung 3). Same CSR inputs/geometry as _em; kernel dequants each
    // weight group once per (row,group) then dp4a's across the expert's token group.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_pairs_matvec_q8_dec(&self, table: &CudaSlice<u64>, proj: i32,
                                   ex_ids: &CudaSlice<i32>, ex_off: &CudaSlice<i32>,
                                   ex_pairs: &CudaSlice<i32>, pair_tok: &CudaSlice<i32>,
                                   aq: &CudaSlice<i8>, ad: &CudaSlice<f32>,
                                   in_f: usize, out_f: usize, n_expert: usize, n_active: usize,
                                   n_pairs: usize, qtype: i32, row_bytes: usize)
                                   -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let f = self.func("moe_pairs_matvec_q8_dec");
        let mut y = self.alloc_uninit::<f32>(n_pairs * out_f)?;
        const ROWS: u32 = 4;
        let cfg = LaunchConfig { grid_dim: ((out_f as u32 + ROWS - 1) / ROWS, n_active as u32, 1),
                                 block_dim: (32, ROWS, 1), shared_mem_bytes: 0 };
        let (inf, outf, ne, na, rbi) = (in_f as i32, out_f as i32, n_expert as i32,
                                        n_active as i32, row_bytes as i64);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(table).arg(&proj).arg(ex_ids).arg(ex_off).arg(ex_pairs).arg(pair_tok)
         .arg(aq).arg(ad).arg(&mut y)
         .arg(&inf).arg(&outf).arg(&ne).arg(&na).arg(&qtype).arg(&rbi);
        unsafe { b.launch(cfg)?; }
        Ok(y)
    }

    pub fn moe_pairs_silu_mul(&self, gate: &CudaSlice<f32>, up: &CudaSlice<f32>, n: usize)
                              -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let f = self.func("moe_pairs_silu_mul");
        let mut act = self.alloc_uninit::<f32>(n)?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let nl = n as i64;
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(gate).arg(up).arg(&mut act).arg(&nl);
        unsafe { b.launch(cfg)?; }
        Ok(act)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn moe_pairs_scatter(&self, y_down: &CudaSlice<f32>, pair_w: &CudaSlice<f32>,
                             tok_pair_off: &CudaSlice<i32>, tok_pair_ids: &CudaSlice<i32>,
                             moe_out: &mut CudaSlice<f32>, t: usize, n_embd: usize)
                             -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("moe_pairs_scatter");
        let cfg = LaunchConfig { grid_dim: (((n_embd + 255) / 256) as u32, t as u32, 1),
                                 block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let ne = n_embd as i32;
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(y_down).arg(pair_w).arg(tok_pair_off).arg(tok_pair_ids).arg(moe_out).arg(&ne);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    pub fn moe_gate_up_silu8_dev_q8(&self, table: &CudaSlice<u64>, sel: &cudarc::driver::CudaView<i32>,
                                    aq: &CudaSlice<i8>, ad: &CudaSlice<f32>,
                                    in_f: usize, n_ff: usize, n_used: usize, n_expert: usize,
                                    qt_g: i32, qt_u: i32, rb_g: usize, rb_u: usize)
                                    -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        static GU: std::sync::OnceLock<(String, u32)> = std::sync::OnceLock::new();
        let (mode, wpb) = GU.get_or_init(|| {
            let mode = std::env::var("BW24_MOE_DEVQ8_GU").unwrap_or_default();
            let wpb = std::env::var("BW24_MOE_DEVQ8_WPB").ok()
                .and_then(|v| v.parse().ok()).unwrap_or(4u32).clamp(1, 16);
            (mode, wpb)
        });
        let (mode, wpb) = (mode.as_str(), *wpb);
        let mut act = self.alloc_uninit::<f32>(n_used * n_ff)?;
        let (inf, nff, ne, rbg, rbu) = (in_f as i32, n_ff as i32, n_expert as i32,
                                        rb_g as i64, rb_u as i64);
        let (f, cfg) = match mode {
            "1" | "2" | "4" => {
                let rpw: u32 = mode.parse().unwrap();
                let f = self.func(match rpw { 1 => "moe_gate_up_silu8_dev_q8_r1",
                                              2 => "moe_gate_up_silu8_dev_q8_r2",
                                              _ => "moe_gate_up_silu8_dev_q8_r4" });
                let rows_per_block = (rpw * wpb) as usize;
                let gx = n_ff.div_ceil(rows_per_block) as u32;
                (f, LaunchConfig { grid_dim: (gx, n_used as u32, 1),
                                   block_dim: (32, wpb, 1), shared_mem_bytes: 0 })
            }
            "j8" if n_used <= 32 => (self.func("moe_gate_up_silu8_dev_q8_j8"),
                     LaunchConfig { grid_dim: (n_ff as u32, 1, 1),
                                    block_dim: (32, n_used as u32, 1), shared_mem_bytes: 0 }),
            // SMEM-GRID twins (IQ3_S 2KB grid copied to shared, static smem — bit-identical dots)
            "sg" => (self.func("moe_gate_up_silu8_dev_q8_sg"),
                     LaunchConfig { grid_dim: (n_ff as u32, n_used as u32, 1),
                                    block_dim: (32, 1, 1), shared_mem_bytes: 0 }),
            "j8sg" if n_used <= 32 => (self.func("moe_gate_up_silu8_dev_q8_j8sg"),
                     LaunchConfig { grid_dim: (n_ff as u32, 1, 1),
                                    block_dim: (32, n_used as u32, 1), shared_mem_bytes: 0 }),
            "u64" if in_f == 2048 => (self.func("moe_gate_up_silu8_dev_q8_u64"),
                     LaunchConfig { grid_dim: (n_ff as u32, n_used as u32, 1),
                                    block_dim: (32, 1, 1), shared_mem_bytes: 0 }),
            "gs4" if in_f == 2048 => (self.func("moe_gate_up_silu8_dev_q8_gs4"),
                     LaunchConfig { grid_dim: (n_ff as u32, n_used as u32, 1),
                                    block_dim: (32, 4, 1), shared_mem_bytes: 0 }),
            // _v twin (down8 lane 2026-07-08): wide-load IQ4_XS dot, base geometry, bit-identical.
            "v" | "" => (self.func("moe_gate_up_silu8_dev_q8_v"),
                    LaunchConfig { grid_dim: (n_ff as u32, n_used as u32, 1),
                                   block_dim: (32, 1, 1), shared_mem_bytes: 0 }),
            "s2" => (self.func("moe_gate_up_silu8_dev_q8_s2"),
                     LaunchConfig { grid_dim: (n_ff as u32, n_used as u32, 1),
                                    block_dim: (32, 2, 1), shared_mem_bytes: 0 }),
            "s2z" => {
                let rz = wpb.min(16);        // s2z smem tile is [16][2]
                (self.func("moe_gate_up_silu8_dev_q8_s2z"),
                 LaunchConfig { grid_dim: (n_ff.div_ceil(rz as usize) as u32, n_used as u32, 1),
                                block_dim: (32, 2, rz), shared_mem_bytes: 0 })
            }
            _ => (self.func("moe_gate_up_silu8_dev_q8"),
                  LaunchConfig { grid_dim: (n_ff as u32, n_used as u32, 1),
                                 block_dim: (32, 1, 1), shared_mem_bytes: 0 }),
        };
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(table).arg(sel).arg(aq).arg(ad).arg(&mut act)
         .arg(&inf).arg(&nff).arg(&ne).arg(&qt_g).arg(&qt_u).arg(&rbg).arg(&rbu);
        unsafe { b.launch(cfg)?; }
        Ok(act)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn moe_down8_fma_dev_q8(&self, table: &CudaSlice<u64>, sel: &cudarc::driver::CudaView<i32>,
                                w: &cudarc::driver::CudaView<f32>,
                                aq2: &CudaSlice<i8>, ad2: &CudaSlice<f32>,
                                dst: &mut cudarc::driver::CudaViewMut<f32>,
                                in_f: usize, out_f: usize, n_used: usize, n_expert: usize,
                                qt: i32, rb: usize)
                                -> Result<(), Box<dyn std::error::Error>> {
        static DOWN: std::sync::OnceLock<String> = std::sync::OnceLock::new();
        let mode = DOWN.get_or_init(|| std::env::var("BW24_MOE_DEVQ8_DOWN").unwrap_or_default());
        let (inf, outf, nu, ne, rbi) = (in_f as i32, out_f as i32, n_used as i32,
                                        n_expert as i32, rb as i64);
        // the w8 twins' smem tile is [RPW][8] — n_used must fit the 8-slot tile;
        // the h2 twins are nsb==16 (in_f==512) shape-gated.
        let (f, cfg) = match mode.as_str() {
            m @ ("1" | "2" | "4") if n_used <= 8 => {
                let rpw: usize = m.parse().unwrap();
                let f = self.func(match rpw { 1 => "moe_down8_fma_dev_q8_w8r1",
                                              2 => "moe_down8_fma_dev_q8_w8r2",
                                              _ => "moe_down8_fma_dev_q8_w8r4" });
                (f, LaunchConfig { grid_dim: (out_f.div_ceil(rpw) as u32, 1, 1),
                                   block_dim: (32, n_used as u32, 1), shared_mem_bytes: 0 })
            }
            "h2" if in_f == 512 => (self.func("moe_down8_fma_dev_q8_h2"),
                LaunchConfig { grid_dim: (out_f.div_ceil(2) as u32, 1, 1),
                               block_dim: (32, 1, 1), shared_mem_bytes: 0 }),
            // "" = AUTO: the measured winner for the 35B expert shape (arc 2026-07-05, +3.8%);
            // any shape the h2 kernels can't take (nsb!=16 / n_used>8) falls to base via `_`.
            // _v twins (down8 lane 2026-07-08): wide-load IQ4_XS dot, bit-identical outputs.
            "w8h2v" | "" if in_f == 512 && n_used <= 8 =>
                (self.func("moe_down8_fma_dev_q8_w8h2v"),
                 LaunchConfig { grid_dim: (out_f.div_ceil(2) as u32, 1, 1),
                                block_dim: (32, n_used as u32, 1), shared_mem_bytes: 0 }),
            "w8h2r2v" if in_f == 512 && n_used <= 8 =>
                (self.func("moe_down8_fma_dev_q8_w8h2r2v"),
                 LaunchConfig { grid_dim: (out_f.div_ceil(4) as u32, 1, 1),
                                block_dim: (32, n_used as u32, 1), shared_mem_bytes: 0 }),
            "w8h2r2" if in_f == 512 && n_used <= 8 =>
                (self.func("moe_down8_fma_dev_q8_w8h2r2"),
                 LaunchConfig { grid_dim: (out_f.div_ceil(4) as u32, 1, 1),
                                block_dim: (32, n_used as u32, 1), shared_mem_bytes: 0 }),
            "w8h2" if in_f == 512 && n_used <= 8 =>
                (self.func("moe_down8_fma_dev_q8_w8h2"),
                 LaunchConfig { grid_dim: (out_f.div_ceil(2) as u32, 1, 1),
                                block_dim: (32, n_used as u32, 1), shared_mem_bytes: 0 }),
            _ => (self.func("moe_down8_fma_dev_q8"),
                  LaunchConfig { grid_dim: (out_f as u32, 1, 1),
                                 block_dim: (32, 1, 1), shared_mem_bytes: 0 }),
        };
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(table).arg(sel).arg(w).arg(aq2).arg(ad2).arg(dst)
         .arg(&inf).arg(&outf).arg(&nu).arg(&ne).arg(&qt).arg(&rbi);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// SMALL-M VERIFY rows twin (BW24_SPEC_M2, lane/spec-m2): ONE launch covers all `t` tokens
    /// of the spec verify's MoE dev gate/up (grid.z = token) — the _v geometry per token, with
    /// tok-offset sel/aq/ad/act pointers matching the serial loop's slices. BIT-IDENTICAL per
    /// token (see the kernel header). aq/ad are the BATCHED z-quantize ([t, in_f] rows —
    /// quantize_q8_1's per-32-block program is row-independent, so batched rows == the serial
    /// loop's per-token quantize_q8_1_view bytes). Returns act [t, n_used, n_ff].
    #[allow(clippy::too_many_arguments)]
    pub fn moe_gate_up_silu8_dev_q8_rows(&self, table: &CudaSlice<u64>, sel: &CudaSlice<i32>,
                                         aq: &CudaSlice<i8>, ad: &CudaSlice<f32>, t: usize,
                                         in_f: usize, n_ff: usize, n_used: usize, n_expert: usize,
                                         qt_g: i32, qt_u: i32, rb_g: usize, rb_u: usize)
                                         -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let f = self.func("moe_gate_up_silu8_dev_q8_v_rows");
        let mut act = self.alloc_uninit::<f32>(t * n_used * n_ff)?;
        let cfg = LaunchConfig { grid_dim: (n_ff as u32, n_used as u32, t as u32),
                                 block_dim: (32, 1, 1), shared_mem_bytes: 0 };
        let (inf, nff, ne, nu, rbg, rbu) = (in_f as i32, n_ff as i32, n_expert as i32,
                                            n_used as i32, rb_g as i64, rb_u as i64);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(table).arg(sel).arg(aq).arg(ad).arg(&mut act)
         .arg(&inf).arg(&nff).arg(&ne).arg(&qt_g).arg(&qt_u).arg(&rbg).arg(&rbu).arg(&nu);
        unsafe { b.launch(cfg)?; }
        Ok(act)
    }

    /// SMALL-M VERIFY rows twin of the down proj: w8h2v geometry per token on a grid.z token
    /// axis. Caller gates the w8h2v shape contract (in_f == 512, n_used <= 8) — same gate as
    /// the AUTO dispatch in `moe_down8_fma_dev_q8`. aq2/ad2 = batched act quantize
    /// ([t*n_used, in_f] rows). dst rows are FULLY overwritten per token.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_down8_fma_dev_q8_rows(&self, table: &CudaSlice<u64>, sel: &CudaSlice<i32>,
                                     w: &CudaSlice<f32>, aq2: &CudaSlice<i8>, ad2: &CudaSlice<f32>,
                                     dst: &mut CudaSlice<f32>, t: usize,
                                     in_f: usize, out_f: usize, n_used: usize, n_expert: usize,
                                     qt: i32, rb: usize)
                                     -> Result<(), Box<dyn std::error::Error>> {
        assert!(in_f == 512 && n_used <= 8, "down rows twin is w8h2v shape-gated");
        let f = self.func("moe_down8_fma_dev_q8_w8h2v_rows");
        let cfg = LaunchConfig { grid_dim: (out_f.div_ceil(2) as u32, 1, t as u32),
                                 block_dim: (32, n_used as u32, 1), shared_mem_bytes: 0 };
        let (inf, outf, nu, ne, rbi) = (in_f as i32, out_f as i32, n_used as i32,
                                        n_expert as i32, rb as i64);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(table).arg(sel).arg(w).arg(aq2).arg(ad2).arg(dst)
         .arg(&inf).arg(&outf).arg(&nu).arg(&ne).arg(&qt).arg(&rbi);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// CSR gate/up v3 (owner-scan dedup, no build kernel): qtypes {IQ4_XS, IQ3_S} (caller
    /// gates), grid.y = pair index; the first pair of each expert serves all its pairs.
    /// Bit-identical to moe_gate_up_silu8_dev_q8_v_rows (explicit-intrinsic accumulate).
    #[allow(clippy::too_many_arguments)]
    pub fn moe_gate_up_silu8_dev_q8_csr(&self, table: &CudaSlice<u64>, sel: &CudaSlice<i32>,
                                        aq: &CudaSlice<i8>, ad: &CudaSlice<f32>,
                                        n_pairs: usize, in_f: usize, n_ff: usize, n_used: usize,
                                        n_expert: usize, qt_g: i32, qt_u: i32, rb_g: usize, rb_u: usize)
                                        -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let f = self.func("moe_gate_up_silu8_dev_q8_csr_iq4");
        let mut act = self.alloc_uninit::<f32>(n_pairs * n_ff)?;
        let cfg = LaunchConfig { grid_dim: (n_ff as u32, n_pairs as u32, 1),
                                 block_dim: (32, 1, 1), shared_mem_bytes: 0 };
        let (inf, nff, ne, nu, npi, rbg, rbu) = (in_f as i32, n_ff as i32, n_expert as i32,
                                                 n_used as i32, n_pairs as i32, rb_g as i64, rb_u as i64);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(table).arg(sel).arg(aq).arg(ad).arg(&mut act)
         .arg(&inf).arg(&nff).arg(&ne).arg(&qt_g).arg(&qt_u).arg(&rbg).arg(&rbu).arg(&nu).arg(&npi);
        unsafe { b.launch(cfg)?; }
        Ok(act)
    }


    /// TEST SEAM (down8 lane 2026-07-08): launch a down dev_q8 variant BY NAME with its
    /// canonical geometry, bypassing the env-cached dispatch so moe-devq8-check can byte-
    /// compare variants in one process. Variants: "base", "w8h2", "w8h2r2", "w8h2v", "w8h2r2v".
    #[allow(clippy::too_many_arguments)]
    pub fn moe_down8_fma_dev_q8_variant(&self, variant: &str, table: &CudaSlice<u64>,
                                        sel: &cudarc::driver::CudaView<i32>,
                                        w: &cudarc::driver::CudaView<f32>,
                                        aq2: &CudaSlice<i8>, ad2: &CudaSlice<f32>,
                                        dst: &mut cudarc::driver::CudaViewMut<f32>,
                                        in_f: usize, out_f: usize, n_used: usize, n_expert: usize,
                                        qt: i32, rb: usize)
                                        -> Result<(), Box<dyn std::error::Error>> {
        let (inf, outf, nu, ne, rbi) = (in_f as i32, out_f as i32, n_used as i32,
                                        n_expert as i32, rb as i64);
        let (f, cfg) = match variant {
            "w8h2" | "w8h2v" => {
                (self.func(if variant == "w8h2" { "moe_down8_fma_dev_q8_w8h2" }
                           else { "moe_down8_fma_dev_q8_w8h2v" }),
                 LaunchConfig { grid_dim: (out_f.div_ceil(2) as u32, 1, 1),
                                block_dim: (32, n_used as u32, 1), shared_mem_bytes: 0 })
            }
            "w8h2r2" | "w8h2r2v" => {
                (self.func(if variant == "w8h2r2" { "moe_down8_fma_dev_q8_w8h2r2" }
                           else { "moe_down8_fma_dev_q8_w8h2r2v" }),
                 LaunchConfig { grid_dim: (out_f.div_ceil(4) as u32, 1, 1),
                                block_dim: (32, n_used as u32, 1), shared_mem_bytes: 0 })
            }
            _ => (self.func("moe_down8_fma_dev_q8"),
                  LaunchConfig { grid_dim: (out_f as u32, 1, 1),
                                 block_dim: (32, 1, 1), shared_mem_bytes: 0 }),
        };
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(table).arg(sel).arg(w).arg(aq2).arg(ad2).arg(dst)
         .arg(&inf).arg(&outf).arg(&nu).arg(&ne).arg(&qt).arg(&rbi);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// TEST SEAM (down8 lane): gate_up twin of the above. Variants: "base", "v".
    #[allow(clippy::too_many_arguments)]
    pub fn moe_gate_up_silu8_dev_q8_variant(&self, variant: &str, table: &CudaSlice<u64>,
                                            sel: &cudarc::driver::CudaView<i32>,
                                            aq: &CudaSlice<i8>, ad: &CudaSlice<f32>,
                                            in_f: usize, n_ff: usize, n_used: usize,
                                            n_expert: usize, qt_g: i32, qt_u: i32,
                                            rb_g: usize, rb_u: usize)
                                            -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let mut act = self.alloc_uninit::<f32>(n_used * n_ff)?;
        let (inf, nff, ne, rbg, rbu) = (in_f as i32, n_ff as i32, n_expert as i32,
                                        rb_g as i64, rb_u as i64);
        let f = self.func(if variant == "v" { "moe_gate_up_silu8_dev_q8_v" }
                          else { "moe_gate_up_silu8_dev_q8" });
        let cfg = LaunchConfig { grid_dim: (n_ff as u32, n_used as u32, 1),
                                 block_dim: (32, 1, 1), shared_mem_bytes: 0 };
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(table).arg(sel).arg(aq).arg(ad).arg(&mut act)
         .arg(&inf).arg(&nff).arg(&ne).arg(&qt_g).arg(&qt_u).arg(&rbg).arg(&rbu);
        unsafe { b.launch(cfg)?; }
        Ok(act)
    }

    pub fn moe_gate_up_silu8_dev(&self, table: &CudaSlice<u64>, sel: &cudarc::driver::CudaView<i32>,
                                 x: &cudarc::driver::CudaView<f32>,
                                 in_f: usize, n_ff: usize, n_used: usize, n_expert: usize,
                                 qt_g: i32, qt_u: i32, rb_g: usize, rb_u: usize)
                                 -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let f = self.func("moe_gate_up_silu8_dev");
        let mut act = self.alloc_uninit::<f32>(n_used * n_ff)?;  // fully overwritten
        let cfg = LaunchConfig { grid_dim: (n_ff as u32, n_used as u32, 1),
                                 block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let (inf, nff, ne, rbg, rbu) = (in_f as i32, n_ff as i32, n_expert as i32,
                                        rb_g as i64, rb_u as i64);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(table).arg(sel).arg(x).arg(&mut act)
         .arg(&inf).arg(&nff).arg(&ne).arg(&qt_g).arg(&qt_u).arg(&rbg).arg(&rbu);
        unsafe { b.launch(cfg)?; }
        Ok(act)
    }

    /// LAUNCH-STRUCTURE STAGE 3: device-dispatch twin of `moe_down8_fma_into` — expert ids AND
    /// renormalized weights read from the router kernel's device output. BIT-IDENTICAL chain.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_down8_fma_dev(&self, table: &CudaSlice<u64>, sel: &cudarc::driver::CudaView<i32>,
                             w: &cudarc::driver::CudaView<f32>, act: &CudaSlice<f32>,
                             dst: &mut cudarc::driver::CudaViewMut<f32>,
                             in_f: usize, out_f: usize, n_used: usize, n_expert: usize,
                             qt: i32, rb: usize)
                             -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("moe_down8_fma_dev");
        let cfg = LaunchConfig { grid_dim: (out_f as u32, 1, 1),
                                 block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let (inf, outf, nu, ne, rbv) = (in_f as i32, out_f as i32, n_used as i32,
                                        n_expert as i32, rb as i64);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(table).arg(sel).arg(w).arg(act).arg(dst)
         .arg(&inf).arg(&outf).arg(&nu).arg(&ne).arg(&qt).arg(&rbv);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// dst[i] += alpha * src[i], i in 0..n. dst is a CudaViewMut (a row of moe_out).
    pub fn axpy_into(&self, src: &CudaSlice<f32>, alpha: f32,
                     dst: &mut cudarc::driver::CudaViewMut<f32>, n: usize)
                     -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("axpy_f32");
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let (a, ni) = (alpha, n as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(src).arg(dst).arg(&a).arg(&ni);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// dst[r*ncols + c] += src[r*ncols + c] * scale[r]. Per-row scalar accumulate (shared expert).
    pub fn add_scaled_rows(&self, src: &CudaSlice<f32>, scale: &CudaSlice<f32>,
                           dst: &mut CudaSlice<f32>, ncols: usize, nrows: usize)
                           -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("add_scaled_rows_f32");
        let cfg = LaunchConfig::for_num_elems((ncols * nrows) as u32);
        let (nc, nr) = (ncols as i32, nrows as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(src).arg(scale).arg(dst).arg(&nc).arg(&nr);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    // ======== A2 GROUPED MoE PREFILL KERNELS ========

    /// Gather m_e rows from src[T, ncols] into dst[m_e, ncols] using index array idx[m_e].
    pub fn gather_rows(&self, src: &CudaSlice<f32>, idx: &CudaSlice<i32>,
                       dst: &mut CudaSlice<f32>, ncols: usize, m_e: usize)
                       -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("gather_rows_f32");
        let cfg = LaunchConfig::for_num_elems((m_e * ncols) as u32);
        let (nc, me) = (ncols as i32, m_e as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(src).arg(idx).arg(dst).arg(&nc).arg(&me);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// Scatter expert outputs into per-token slots: dst[tok_idx[r], slot_idx[r], :] = src[r, :] * weight[r].
    /// dst is [T, n_used, ncols], zero-initialized. Each (expert, token) pair maps to a unique slot.
    /// Scatter expert outputs into per-token slots (raw copy, no weight multiply).
    /// Weight stored into wbuf[tok*n_used + slot] for FMA in reduce step.
    pub fn scatter_slot(&self, src: &CudaSlice<f32>, tok_idx: &CudaSlice<i32>,
                        slot_idx: &CudaSlice<i32>, weight: &CudaSlice<f32>,
                        dst: &mut CudaSlice<f32>, wbuf: &mut CudaSlice<f32>,
                        ncols: usize, n_used: usize, m_e: usize)
                        -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("scatter_add_slot_f32");
        let cfg = LaunchConfig::for_num_elems((m_e * ncols) as u32);
        let (nc, nu, me) = (ncols as i32, n_used as i32, m_e as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(src).arg(tok_idx).arg(slot_idx).arg(weight).arg(dst).arg(wbuf).arg(&nc).arg(&nu).arg(&me);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// Reduce n_used slots per token: dst[t, col] = sum_s slots[t, s, col].
    /// Reduce n_used slots per token: dst[t, col] = sum_s FMA(wbuf[t,s], slots[t,s,col], acc).
    /// Uses FMA for bit-identity with the sequential axpy path.
    pub fn reduce_slots(&self, slots: &CudaSlice<f32>, wbuf: &CudaSlice<f32>,
                        dst: &mut CudaSlice<f32>, ncols: usize, n_used: usize, t: usize)
                        -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("reduce_slots_f32");
        let cfg = LaunchConfig::for_num_elems((t * ncols) as u32);
        let (nc, nu, ti) = (ncols as i32, n_used as i32, t as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(slots).arg(wbuf).arg(dst).arg(&nc).arg(&nu).arg(&ti);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// Stage-B: quantize activation [m,in] f32 -> q8_1 (int8 qs + per-block f32 scale).
    /// Quantize an activation [m, in_f] to q8_1 (int8 qs + per-32 f32 scale). Public so the
    /// forward can quantize a SHARED activation ONCE and feed it to several matmuls (gate+up
    /// share `z`; q/k/v and wqkv/gate/beta/alpha share `h`) — quantize_q8_1 was 13.5% of decode
    /// GPU time, ~half of it redundant re-quantization of the same row.
    /// quantize_q8_1 over a CudaView (a sliced z-row) — same kernel, offset-honoring arg.
    pub fn quantize_q8_1_view(&self, x: &cudarc::driver::CudaView<f32>, m: usize, in_f: usize)
                     -> Result<(CudaSlice<i8>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let f = self.func("quantize_q8_1");
        let nblk = in_f / 32;
        let mut q = self.alloc_uninit::<i8>(m * in_f)?;
        let mut d = self.alloc_uninit::<f32>(m * nblk)?;
        let cfg = LaunchConfig::for_num_elems((m * in_f) as u32);
        let (inf, mi) = (in_f as i32, m as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(x).arg(&mut q).arg(&mut d).arg(&inf).arg(&mi);
        unsafe { b.launch(cfg)?; }
        Ok((q, d))
    }

    pub fn quantize_q8_1(&self, x: &CudaSlice<f32>, m: usize, in_f: usize)
                     -> Result<(CudaSlice<i8>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let f = self.func("quantize_q8_1");
        let nblk = in_f / 32;
        let mut q = self.alloc_uninit::<i8>(m * in_f)?;  // full-overwrite output: skip memset
        let mut d = self.alloc_uninit::<f32>(m * nblk)?;  // full-overwrite output: skip memset
        // WARP-PER-BLOCK kernel: one warp per 32-block -> m*in_f threads total.
        let cfg = LaunchConfig::for_num_elems((m * in_f) as u32);
        let (inf, mi) = (in_f as i32, m as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(x).arg(&mut q).arg(&mut d).arg(&inf).arg(&mi);
        unsafe { b.launch(cfg)?; }
        Ok((q, d))
    }

    /// Stage-C FP4: quantize activation [m,in] f32 -> e2m1 nibbles (aq4: u32 [m, in/8]) + per-16
    /// UE4M3 scale (ad4: u8 [m, in/16]), the layout the mxf4nvf4 block-scale GEMM B-operand wants.
    /// in_f must be a multiple of 64 (one NVFP4 K-block). One thread per (token, 16-block).
    pub fn quantize_fp4_act(&self, x: &CudaSlice<f32>, m: usize, in_f: usize)
                     -> Result<(CudaSlice<u32>, CudaSlice<u8>), Box<dyn std::error::Error>> {
        let f = self.func("quantize_fp4_act");
        let nb16 = in_f / 16;
        let mut aq4 = self.alloc_uninit::<u32>(m * (in_f / 8))?;  // full-overwrite output: skip memset
        let mut ad4 = self.alloc_uninit::<u8>(m * nb16)?;  // full-overwrite output: skip memset
        let cfg = LaunchConfig::for_num_elems((m * nb16) as u32);
        let (inf, mi) = (in_f as i32, m as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(x).arg(&mut aq4).arg(&mut ad4).arg(&inf).arg(&mi);
        unsafe { b.launch(cfg)?; }
        Ok((aq4, ad4))
    }

    /// Stage-C FP4 GEMM (NVFP4 weights): native mxf4nvf4 block-scale tensor-core matmul. Feeds raw
    /// e2m1 weight nibbles + raw UE4M3 micro-scales directly to mma.sync.m16n8k64 (762 TFLOP/s peak,
    /// 3.5x int8). Activation `x` is quantized to FP4 e2m1 here. NVFP4 per-tensor macro-scale applied
    /// post (scale==1.0 -> no-op). `bytes` = raw NVFP4 weight rows. Used by the BW24_FP4 prefill path.
    pub fn qmatvec_gemm_nvfp4_fp4(&self, bytes: &CudaSlice<u8>, x: &CudaSlice<f32>, m: usize,
                                  in_f: usize, out_f: usize, row_bytes: usize, scale: f32)
                                  -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        assert!(in_f % 64 == 0, "FP4 GEMM requires in_f % 64 == 0, got {in_f}");
        let (aq4, ad4) = self.quantize_fp4_act(x, m, in_f)?;
        let mut y = self.fp4_gemm_launch(bytes, &aq4, &ad4, m, in_f, out_f, row_bytes)?;
        if scale != 1.0 { self.scale_inplace(&mut y, scale, m * out_f)?; }
        Ok(y)
    }

    /// Shared mxf4 GEMM launch (pre-quantized FP4 activation aq4/ad4). Same CTA tile as the int8 GEMM
    /// (BM=64 rows x BN=128 tokens, 4 warps). No macro-scale applied here.
    fn fp4_gemm_launch(&self, bytes: &CudaSlice<u8>, aq4: &CudaSlice<u32>, ad4: &CudaSlice<u8>,
                       m: usize, in_f: usize, out_f: usize, row_bytes: usize)
                       -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let f = self.func("qmatvec_gemm_nvfp4_fp4");
        let mut y = self.alloc_uninit::<f32>(m * out_f)?;  // full-overwrite output: skip memset
        const BM: u32 = 64; const BN: u32 = 256;
        let cfg = LaunchConfig {
            grid_dim: ((out_f as u32 + BM - 1) / BM, (m as u32 + BN - 1) / BN, 1),
            block_dim: (32, 4, 1), shared_mem_bytes: 0,
        };
        let (inf, outf, mi, rb) = (in_f as i32, out_f as i32, m as i32, row_bytes as i64);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(bytes).arg(aq4).arg(ad4).arg(&mut y).arg(&inf).arg(&outf).arg(&mi).arg(&rb);
        unsafe { b.launch(cfg)?; }
        Ok(y)
    }

    /// Test entry (kernel_check): run the FP4 GEMM from raw bytes; NO macro-scale (caller compares bare).
    pub fn qmatvec_gemm_nvfp4_fp4_raw(&self, bytes: &CudaSlice<u8>, x: &CudaSlice<f32>, m: usize,
                                      in_f: usize, out_f: usize, row_bytes: usize)
                                      -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        assert!(in_f % 64 == 0, "FP4 GEMM requires in_f % 64 == 0, got {in_f}");
        let (aq4, ad4) = self.quantize_fp4_act(x, m, in_f)?;
        self.fp4_gemm_launch(bytes, &aq4, &ad4, m, in_f, out_f, row_bytes)
    }

    /// Stage-B: Q8_0 weight x q8_1 activation int8 dp4a matmul. y[m,out]=x@W^T.
    pub fn qmatvec_q8_0_fast(&self, w: &CudaSlice<u8>, x: &CudaSlice<f32>, m: usize, in_f: usize,
                             out_f: usize, row_bytes: usize) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let (aq, ad) = self.quantize_q8_1(x, m, in_f)?;
        let f = self.func("qmatvec_q8_0_dp4a");
        let mut y = self.alloc_uninit::<f32>(m * out_f)?;  // full-overwrite output: skip memset
        let cfg = LaunchConfig { grid_dim: (out_f as u32, m as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
        let (inf, outf, mi, rb) = (in_f as i32, out_f as i32, m as i32, row_bytes as i64);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(w).arg(&aq).arg(&ad).arg(&mut y).arg(&inf).arg(&outf).arg(&mi).arg(&rb);
        unsafe { b.launch(cfg)?; }
        Ok(y)
    }

    /// Stage-B: Q4_K weight x q8_1 activation int8 dp4a (decode). Min-offset via q8_1 sum term.
    pub fn qmatvec_q4_K_fast(&self, w: &CudaSlice<u8>, x: &CudaSlice<f32>, m: usize, in_f: usize,
                             out_f: usize, row_bytes: usize) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let (aq, ad) = self.quantize_q8_1(x, m, in_f)?;
        let f = self.func("qmatvec_q4_K_dp4a");
        let mut y = self.alloc_uninit::<f32>(m * out_f)?;  // full-overwrite output: skip memset
        let cfg = LaunchConfig { grid_dim: (out_f as u32, m as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
        let (inf, outf, mi, rb) = (in_f as i32, out_f as i32, m as i32, row_bytes as i64);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(w).arg(&aq).arg(&ad).arg(&mut y).arg(&inf).arg(&outf).arg(&mi).arg(&rb);
        unsafe { b.launch(cfg)?; }
        Ok(y)
    }

    /// Stage-B: Q6_K weight x q8_1 activation int8 dp4a (decode, symmetric).
    pub fn qmatvec_q6_K_fast(&self, w: &CudaSlice<u8>, x: &CudaSlice<f32>, m: usize, in_f: usize,
                             out_f: usize, row_bytes: usize) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let (aq, ad) = self.quantize_q8_1(x, m, in_f)?;
        let f = self.func("qmatvec_q6_K_dp4a");
        let mut y = self.alloc_uninit::<f32>(m * out_f)?;  // full-overwrite output: skip memset
        let cfg = LaunchConfig { grid_dim: (out_f as u32, m as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
        let (inf, outf, mi, rb) = (in_f as i32, out_f as i32, m as i32, row_bytes as i64);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(w).arg(&aq).arg(&ad).arg(&mut y).arg(&inf).arg(&outf).arg(&mi).arg(&rb);
        unsafe { b.launch(cfg)?; }
        Ok(y)
    }

    /// Stage-B: Q5_K weight x q8_1 activation int8 dp4a (decode). Min-offset via q8_1 sum term.
    pub fn qmatvec_q5_K_fast(&self, w: &CudaSlice<u8>, x: &CudaSlice<f32>, m: usize, in_f: usize,
                             out_f: usize, row_bytes: usize) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        self.qmatvec_dp4a_named("qmatvec_q5_K_dp4a", w, x, m, in_f, out_f, row_bytes)
    }
    /// Stage-B: Q3_K weight x q8_1 activation int8 dp4a (decode, symmetric).
    pub fn qmatvec_q3_K_fast(&self, w: &CudaSlice<u8>, x: &CudaSlice<f32>, m: usize, in_f: usize,
                             out_f: usize, row_bytes: usize) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        self.qmatvec_dp4a_named("qmatvec_q3_K_dp4a", w, x, m, in_f, out_f, row_bytes)
    }
    /// A6 split-plane twin of `qmatvec_nvfp4_fast` (weights repacked; used by the rp gates).
    pub fn qmatvec_nvfp4_fast_rp(&self, w: &CudaSlice<u8>, x: &CudaSlice<f32>, m: usize, in_f: usize,
                                 out_f: usize, row_bytes: usize) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        assert!(in_f % 64 == 0, "NVFP4 dp4a requires in_f % 64 == 0, got {in_f}");
        self.qmatvec_dp4a_named("qmatvec_nvfp4_dp4a_rp", w, x, m, in_f, out_f, row_bytes)
    }
    /// Stage-B: NVFP4 weight x q8_1 activation int8 dp4a (decode, symmetric, codebook lookup).
    pub fn qmatvec_nvfp4_fast(&self, w: &CudaSlice<u8>, x: &CudaSlice<f32>, m: usize, in_f: usize,
                              out_f: usize, row_bytes: usize) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        // B1: the NVFP4 dp4a kernel maps two 32-elem q8_1 blocks onto one 64-elem block_nvfp4
        // (sblk = g >> 1). in_f must be a multiple of 64 or the last block reads a partial superblock.
        assert!(in_f % 64 == 0, "NVFP4 dp4a requires in_f % 64 == 0, got {in_f}");
        self.qmatvec_dp4a_named("qmatvec_nvfp4_dp4a", w, x, m, in_f, out_f, row_bytes)
    }
    /// Stage-B (optional perf): IQ4_XS codebook int8 dp4a.
    pub fn qmatvec_iq4_XS_fast(&self, w: &CudaSlice<u8>, x: &CudaSlice<f32>, m: usize, in_f: usize,
                               out_f: usize, row_bytes: usize) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        self.qmatvec_dp4a_named("qmatvec_iq4_XS_dp4a", w, x, m, in_f, out_f, row_bytes)
    }

    /// Shared dp4a launcher: quantize_q8_1 then call the named kernel (grid (out,m), block 64).
    fn qmatvec_dp4a_named(&self, name: &str, w: &CudaSlice<u8>, x: &CudaSlice<f32>, m: usize,
                          in_f: usize, out_f: usize, row_bytes: usize)
                          -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let (aq, ad) = self.quantize_q8_1(x, m, in_f)?;
        let f = self.func(name);
        let mut y = self.alloc_uninit::<f32>(m * out_f)?;  // full-overwrite output: skip memset
        let cfg = LaunchConfig { grid_dim: (out_f as u32, m as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
        let (inf, outf, mi, rb) = (in_f as i32, out_f as i32, m as i32, row_bytes as i64);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(w).arg(&aq).arg(&ad).arg(&mut y).arg(&inf).arg(&outf).arg(&mi).arg(&rb);
        unsafe { b.launch(cfg)?; }
        Ok(y)
    }

    pub fn htod(&self, v: &[f32]) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        Ok(self.gpu.stream.clone_htod(v)?)
    }
    pub fn htod_i32(&self, v: &[i32]) -> Result<CudaSlice<i32>, Box<dyn std::error::Error>> {
        Ok(self.gpu.stream.clone_htod(v)?)
    }
    /// i8 upload (moe-devq8-check: synthetic q8_1 activation bytes).
    pub fn htod_i8(&self, v: &[i8]) -> Result<CudaSlice<i8>, Box<dyn std::error::Error>> {
        Ok(self.gpu.stream.clone_htod(v)?)
    }
    pub fn htod_u64(&self, v: &[u64]) -> Result<CudaSlice<u64>, Box<dyn std::error::Error>> {
        Ok(self.gpu.stream.clone_htod(v)?)
    }
    pub fn dtoh(&self, d: &CudaSlice<f32>) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let v = self.gpu.stream.clone_dtoh(d)?;
        self.gpu.stream.synchronize()?;
        Ok(v)
    }
    /// Device-to-host copy of an i32 buffer (fused-router sel_idx readback).
    pub fn dtoh_i32(&self, d: &CudaSlice<i32>) -> Result<Vec<i32>, Box<dyn std::error::Error>> {
        let v = self.gpu.stream.clone_dtoh(d)?;
        self.gpu.stream.synchronize()?;
        Ok(v)
    }
    /// Device-to-host copy of a u8 buffer (used to read back the quantized KV cache for validation).
    pub fn dtoh_u8(&self, d: &CudaSlice<u8>) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let v = self.gpu.stream.clone_dtoh(d)?;
        self.gpu.stream.synchronize()?;
        Ok(v)
    }
    pub fn zeros(&self, n: usize) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        Ok(self.gpu.stream.alloc_zeros::<f32>(n)?)
    }

    /// GPU-resident greedy argmax (CUDA-GRAPH-PLAN Phase 1): logits[n_vocab] -> token id in a
    /// resident device u32 [1]. PARALLEL 2-pass (RANK1 LEVER): the old single-CTA scan (one 256-thread
    /// block on one SM over 248K logits) was memory-starved at ~426us/token. Now pass 1 fans NB=256
    /// blocks across the SMs to saturate HBM, pass 2 reduces the NB partials. Bit-identical to host
    /// `argmax` (smallest index on tie). The whole point is NOT to dtoh logits — only a [1] u32 is read
    /// back (or kept resident for graph replay). Returns the device token buffer.
    /// Softmax probability of the (already-argmaxed) token `tok` under `logits` — the spec-decode
    /// p-min confidence signal. 2-pass like the parallel argmax; returns a device [1] f32.
    pub fn prob_of_token_device(&self, logits: &CudaSlice<f32>, tok: &CudaSlice<u32>, n_vocab: usize)
                                -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let nb = ARGMAX_NB;
        let mut part = self.alloc_uninit::<f32>(nb)?;
        let mut p = self.alloc_uninit::<f32>(1)?;
        let f1 = self.func("prob_of_token_partial_f32");
        let cfg1 = LaunchConfig { grid_dim: (nb as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let nv = n_vocab as i32;
        let mut b1 = self.gpu.stream.launch_builder(&f1);
        b1.arg(logits).arg(tok).arg(&mut part).arg(&nv);
        unsafe { b1.launch(cfg1)?; }
        let f2 = self.func("prob_of_token_final_f32");
        let cfg2 = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let nbi = nb as i32;
        let mut b2 = self.gpu.stream.launch_builder(&f2);
        b2.arg(&part).arg(&mut p).arg(&nbi);
        unsafe { b2.launch(cfg2)?; }
        Ok(p)
    }

    /// Like `prob_of_token_device` but writes into a PERSISTENT `p_out` buffer (stable pointer).
    /// Required for CUDA-graph capture of the draft chain: the captured prob kernels must write
    /// where the host reads the p-min confidence between replays. Same kernels, same math.
    pub fn prob_of_token_device_into(&self, logits: &CudaSlice<f32>, tok: &CudaSlice<u32>,
                                     p_out: &mut CudaSlice<f32>, n_vocab: usize)
                                     -> Result<(), Box<dyn std::error::Error>> {
        let nb = ARGMAX_NB;
        let mut part = self.alloc_uninit::<f32>(nb)?;
        let f1 = self.func("prob_of_token_partial_f32");
        let cfg1 = LaunchConfig { grid_dim: (nb as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let nv = n_vocab as i32;
        let mut b1 = self.gpu.stream.launch_builder(&f1);
        b1.arg(logits).arg(tok).arg(&mut part).arg(&nv);
        unsafe { b1.launch(cfg1)?; }
        let f2 = self.func("prob_of_token_final_f32");
        let cfg2 = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let nbi = nb as i32;
        let mut b2 = self.gpu.stream.launch_builder(&f2);
        b2.arg(&part).arg(p_out).arg(&nbi);
        unsafe { b2.launch(cfg2)?; }
        Ok(())
    }

    pub fn argmax_token_device(&self, logits: &CudaSlice<f32>, n_vocab: usize)
                               -> Result<CudaSlice<u32>, Box<dyn std::error::Error>> {
        let mut tok = unsafe { self.gpu.stream.alloc::<u32>(1)? };
        self.argmax_token_device_into(logits, &mut tok, n_vocab)?;
        Ok(tok)
    }
    /// Like `argmax_token_device` but writes into a PERSISTENT `tok` buffer (stable pointer) instead
    /// of allocating a fresh one. Required for CUDA-graph capture: the captured argmax must write the
    /// next token into the SAME device buffer the next replay's embed_gather reads, so the buffer
    /// pointer is baked once and the token id never round-trips to host inside steady state. The
    /// pass-1 partials scratch (`argmax_partials`) is also a resident stable-pointer buffer so both
    /// captured passes bake fixed addresses.
    pub fn argmax_token_device_into(&self, logits: &CudaSlice<f32>, tok: &mut CudaSlice<u32>,
                                    n_vocab: usize) -> Result<(), Box<dyn std::error::Error>> {
        let nb = ARGMAX_NB;
        let f1 = self.func("argmax_partial_f32");
        let f2 = self.func("argmax_final_f32");
        let mut guard = self.argmax_partials.lock().unwrap();
        if guard.is_none() {
            // allocate ONCE; under generate_graph this runs in the tracking-off prime window so the
            // buffers carry no cudarc events (illegal inside capture).
            let pv = self.gpu.stream.alloc_zeros::<f32>(nb)?;
            let pi = self.gpu.stream.alloc_zeros::<i32>(nb)?;
            *guard = Some((pv, pi));
        }
        let (part_v, part_i) = guard.as_mut().unwrap();
        let nv = n_vocab as i32;
        let nbi = nb as i32;
        // pass 1: NB blocks x 256 threads grid-stride scan -> per-block (val, idx) partials.
        let cfg1 = LaunchConfig { grid_dim: (nb as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let mut b1 = self.gpu.stream.launch_builder(&f1);
        b1.arg(logits).arg(&mut *part_v).arg(&mut *part_i).arg(&nv);
        unsafe { b1.launch(cfg1)?; }
        // pass 2: one block reduces NB partials -> token_out[0].
        let cfg2 = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let mut b2 = self.gpu.stream.launch_builder(&f2);
        b2.arg(&*part_v).arg(&*part_i).arg(tok).arg(&nbi);
        unsafe { b2.launch(cfg2)?; }
        Ok(())
    }
    /// Column-`col` device argmax over a stacked verify-logits buffer [t, n_vocab] (spec accept
    /// walk): toks[out_idx] = argmax(logits[col*n_vocab .. (col+1)*n_vocab]). SAME 2-pass kernels
    /// and tie-break contract as `argmax_token_device_into` (bit-identical to host argmax,
    /// argmax_gate-validated) — only the input pointer (a column view) and the output slot differ.
    /// Lets the accept walk read ONE [t] u32 instead of dtoh'ing the full [t, n_vocab] logits.
    pub fn argmax_token_device_col(&self, logits: &CudaSlice<f32>, col: usize, n_vocab: usize,
                                   toks: &mut CudaSlice<u32>, out_idx: usize)
                                   -> Result<(), Box<dyn std::error::Error>> {
        let nb = ARGMAX_NB;
        let f1 = self.func("argmax_partial_f32");
        let f2 = self.func("argmax_final_f32");
        let mut guard = self.argmax_partials.lock().unwrap();
        if guard.is_none() {
            let pv = self.gpu.stream.alloc_zeros::<f32>(nb)?;
            let pi = self.gpu.stream.alloc_zeros::<i32>(nb)?;
            *guard = Some((pv, pi));
        }
        let (part_v, part_i) = guard.as_mut().unwrap();
        let col_view = logits.slice(col * n_vocab..(col + 1) * n_vocab);
        let nv = n_vocab as i32;
        let nbi = nb as i32;
        let cfg1 = LaunchConfig { grid_dim: (nb as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let mut b1 = self.gpu.stream.launch_builder(&f1);
        b1.arg(&col_view).arg(&mut *part_v).arg(&mut *part_i).arg(&nv);
        unsafe { b1.launch(cfg1)?; }
        let mut tok_view = toks.slice_mut(out_idx..out_idx + 1);
        let cfg2 = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let mut b2 = self.gpu.stream.launch_builder(&f2);
        b2.arg(&*part_v).arg(&*part_i).arg(&mut tok_view).arg(&nbi);
        unsafe { b2.launch(cfg2)?; }
        Ok(())
    }
    /// Read back a device u32 buffer (the spec accept walk's [t] per-column argmax tokens).
    pub fn htod_u32_v(&self, v: &[u32]) -> Result<CudaSlice<u32>, Box<dyn std::error::Error>> {
        Ok(self.gpu.stream.memcpy_stod(v)?)
    }
    pub fn dtoh_u32(&self, d: &CudaSlice<u32>) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let v = self.gpu.stream.clone_dtoh(d)?;
        self.gpu.stream.synchronize()?;
        Ok(v)
    }
    /// Allocate a zeroed device u32 buffer (persistent spec-loop prediction slots).
    pub fn alloc_u32_zeroed(&self, n: usize) -> Result<CudaSlice<u32>, Box<dyn std::error::Error>> {
        Ok(self.gpu.stream.alloc_zeros::<u32>(n)?)
    }
    /// embed_gather into a PERSISTENT `x_out` buffer (stable pointer) for CUDA-graph capture (the
    /// embed output starts the per-step kernel chain and must be at a fixed address across replays).
    pub fn embed_gather_device_into(&self, embd: &CudaSlice<u8>, token_d: &CudaSlice<u32>,
                                    x_out: &mut CudaSlice<f32>, n_embd: usize, qtype: i32,
                                    row_bytes: usize) -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("embed_gather_u32");
        let cfg = LaunchConfig { grid_dim: (((n_embd as u32 + 255) / 256).max(1), 1, 1),
                                 block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let (ne, qt, rb) = (n_embd as i32, qtype, row_bytes as i64);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(embd).arg(token_d).arg(x_out).arg(&ne).arg(&qt).arg(&rb);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }
    /// Read a [1] i32 device counter (pos / seqlen) back to host. Tiny D2H + sync.
    pub fn dtoh_i32_one(&self, d: &CudaSlice<i32>) -> Result<i32, Box<dyn std::error::Error>> {
        let v = self.gpu.stream.clone_dtoh(d)?;
        self.gpu.stream.synchronize()?;
        Ok(v[0])
    }
    /// Set a [1] i32 device counter IN PLACE (keeps the buffer pointer stable — required for the
    /// graph-resident pos/seqlen counters whose addresses are baked into captured graphs). Restores
    /// the counter value after the throwaway capture warmups corrupt it.
    pub fn set_i32_one(&self, d: &mut CudaSlice<i32>, v: i32) -> Result<(), Box<dyn std::error::Error>> {
        self.gpu.stream.memcpy_htod(&[v], d)?;
        Ok(())
    }
    /// Set a [1] u32 device buffer IN PLACE (stable pointer) — for the resident `token_d` counter
    /// during priming / capture-state restore.
    pub fn set_u32_one(&self, d: &mut CudaSlice<u32>, v: u32) -> Result<(), Box<dyn std::error::Error>> {
        self.gpu.stream.memcpy_htod(&[v], d)?;
        Ok(())
    }
    /// Read back a [1] u32 device buffer (the argmax token). One tiny D2H + sync.
    pub fn dtoh_u32_one(&self, d: &CudaSlice<u32>) -> Result<u32, Box<dyn std::error::Error>> {
        let v = self.gpu.stream.clone_dtoh(d)?;
        self.gpu.stream.synchronize()?;
        Ok(v[0])
    }
    /// Upload raw bytes to a resident device u8 buffer (e.g. the embed table for device gather).
    pub fn upload_u8(&self, bytes: &[u8]) -> Result<CudaSlice<u8>, Box<dyn std::error::Error>> {
        Ok(self.gpu.stream.clone_htod(bytes)?)
    }
    /// Embed-from-device (CUDA-GRAPH-PLAN Phase 1): gather+dequant the row for the token id in
    /// `token_d[0]` from the resident embed table `embd` -> x_out[n_embd]. Bit-identical to host
    /// EmbedHost::gather (same per-dtype `deq`). No host round-trip of the token id.
    pub fn embed_gather_device(&self, embd: &CudaSlice<u8>, token_d: &CudaSlice<u32>,
                               n_embd: usize, qtype: i32, row_bytes: usize)
                               -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let f = self.func("embed_gather_u32");
        let mut x = self.alloc_uninit::<f32>(n_embd)?;
        let cfg = LaunchConfig { grid_dim: (((n_embd as u32 + 255) / 256).max(1), 1, 1),
                                 block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let (ne, qt, rb) = (n_embd as i32, qtype, row_bytes as i64);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(embd).arg(token_d).arg(&mut x).arg(&ne).arg(&qt).arg(&rb);
        unsafe { b.launch(cfg)?; }
        Ok(x)
    }

    /// T-token device embed gather (spec verify/replay): tokens uploaded as a tiny [T] u32 htod,
    /// rows dequanted on-device -> x[T, n_embd]. Replaces host per-row dequant + T*n_embd*4B htod
    /// (nsys: 84% of spec API time was HtoD). Bit-identical rows (same per-dtype deq).
    pub fn embed_gather_device_t(&self, embd: &CudaSlice<u8>, tokens: &[u32],
                                 n_embd: usize, qtype: i32, row_bytes: usize)
                                 -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let t = tokens.len();
        let tok_d = self.gpu.stream.clone_htod(tokens)?;
        let f = self.func("embed_gather_u32_t");
        let mut x = self.alloc_uninit::<f32>(t * n_embd)?;
        let cfg = LaunchConfig { grid_dim: (((n_embd as u32 + 255) / 256).max(1), t as u32, 1),
                                 block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let (ne, qt, rb, ti) = (n_embd as i32, qtype, row_bytes as i64, t as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(embd).arg(&tok_d).arg(&mut x).arg(&ne).arg(&qt).arg(&rb).arg(&ti);
        unsafe { b.launch(cfg)?; }
        Ok(x)
    }

    /// Uninitialized device buffer — SKIPS the memset that `alloc_zeros` always issues. Decode
    /// profile (nsys): ~1050 memsets/token = 6.5% of decode GPU time + ~half the launch count, the
    /// dominant contributor to the 19% inter-kernel idle gap and a blocker for clean CUDA-graph
    /// capture. Use ONLY for buffers a kernel FULLY overwrites (every element written, no `+=`).
    /// SAFETY: caller guarantees the producing kernel writes every element before any read.
    #[inline]
    fn alloc_uninit<T: cudarc::driver::DeviceRepr>(&self, n: usize)
            -> Result<CudaSlice<T>, Box<dyn std::error::Error>> {
        Ok(unsafe { self.gpu.stream.alloc::<T>(n)? })
    }

    /// Public f32 uninitialized scratch (see `alloc_uninit`). For decode/forward scratch a kernel
    /// fully overwrites. SAFETY: producing kernel must write every element before any read.
    pub fn uninit(&self, n: usize) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        self.alloc_uninit::<f32>(n)
    }

    /// RMSNorm: x[ncols,nrows] row-major, weight[ncols] -> dst. One block/row, 256 threads.
    pub fn rms_norm(&self, x: &CudaSlice<f32>, w: &CudaSlice<f32>, dst: &mut CudaSlice<f32>,
                    ncols: usize, nrows: usize, eps: f32) -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("rms_norm_f32");
        let cfg = LaunchConfig { grid_dim: (nrows as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let (nc, e) = (ncols as i32, eps);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(x).arg(w).arg(dst).arg(&nc).arg(&e);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// RMS-norm with blockDim=1024 — BIT-IDENTICAL to the fused `rms_norm_q8_1` and
    /// `add_rms_norm_q8_1` kernels' sum-of-squares reduction. The spec verify path MUST use this
    /// to match decode's FP accumulation order: the standard `rms_norm` at blockDim=256 has a
    /// different per-thread stride (ncols/256 partials vs ncols/1024 partials) and therefore a
    /// different shfl-tree reduction that can shift `scale = rsqrt(sum/n + eps)` by ULPs, causing
    /// divergence through the GDN scan and argmax flips on the 9B text prompt. The underlying
    /// `rms_norm_f32` kernel supports any blockDim (generic reduce with shared[32]).
    pub fn rms_norm_decode(&self, x: &CudaSlice<f32>, w: &CudaSlice<f32>, dst: &mut CudaSlice<f32>,
                           ncols: usize, nrows: usize, eps: f32) -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("rms_norm_f32");
        let cfg = LaunchConfig { grid_dim: (nrows as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
        let (nc, e) = (ncols as i32, eps);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(x).arg(w).arg(dst).arg(&nc).arg(&e);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// DECODE GLUE-FUSION LEVER: `z = rms_norm(x)*w` emitted DIRECTLY as q8_1 (no f32 `z` materialized,
    /// no standalone quantize_q8_1 launch). Returns (out_q [nrows*ncols i8], out_d [nrows*nblk f32])
    /// ready to feed matmul_pre. BIT-IDENTICAL to rms_norm + quantize_q8_1. ncols % 32 == 0.
    pub fn rms_norm_q8_1(&self, x: &CudaSlice<f32>, w: &CudaSlice<f32>, ncols: usize, nrows: usize,
                         eps: f32) -> Result<(CudaSlice<i8>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let nblk = ncols / 32;
        let mut q = self.alloc_uninit::<i8>(nrows * ncols)?;
        let mut d = self.alloc_uninit::<f32>(nrows * nblk)?;
        let f = self.func("rms_norm_q8_1");
        // 1024 threads: decode is nrows=1 -> ONE CTA; 32 warps hide the pass1->pass2 latency
        // (s[32] reduce already sized for 32 warps). Same shape math at any blockDim.
        let cfg = LaunchConfig { grid_dim: (nrows as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
        let (nc, e) = (ncols as i32, eps);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(x).arg(w).arg(&mut q).arg(&mut d).arg(&nc).arg(&e);
        unsafe { b.launch(cfg)?; }
        Ok((q, d))
    }

    /// DECODE GLUE-FUSION LEVER: `res = a+b; z = rms_norm(res)*w` with z emitted as q8_1. `res` is
    /// still written (the post-ffn residual add reads it). Fuses add_rms_norm + quantize_q8_1.
    /// Returns (out_q, out_d) for matmul_pre. BIT-IDENTICAL. ncols % 32 == 0.
    pub fn add_rms_norm_q8_1(&self, a: &CudaSlice<f32>, b_in: &CudaSlice<f32>, w: &CudaSlice<f32>,
                             res: &mut CudaSlice<f32>, ncols: usize, nrows: usize, eps: f32)
                             -> Result<(CudaSlice<i8>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let nblk = ncols / 32;
        let mut q = self.alloc_uninit::<i8>(nrows * ncols)?;
        let mut d = self.alloc_uninit::<f32>(nrows * nblk)?;
        let f = self.func("add_rms_norm_q8_1");
        // 1024 threads: same single-CTA-at-decode reasoning as rms_norm_q8_1.
        let cfg = LaunchConfig { grid_dim: (nrows as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
        let (nc, e) = (ncols as i32, eps);
        let mut bld = self.gpu.stream.launch_builder(&f);
        bld.arg(a).arg(b_in).arg(w).arg(res).arg(&mut q).arg(&mut d).arg(&nc).arg(&e);
        unsafe { bld.launch(cfg)?; }
        Ok((q, d))
    }

    /// RANK3 LEVER (add+rmsnorm fuse): `res = a + b; dst = rms_norm(res) * w` in ONE launch. Fuses
    /// e.add(a,b,res) + e.rms_norm(res,w,dst), removing one launch + one HBM read of the residual per
    /// residual+norm pair. BIT-IDENTICAL to the two-kernel sequence (same IEEE add, same reduction).
    pub fn add_rms_norm(&self, a: &CudaSlice<f32>, b: &CudaSlice<f32>, w: &CudaSlice<f32>,
                        res: &mut CudaSlice<f32>, dst: &mut CudaSlice<f32>, ncols: usize, nrows: usize,
                        eps: f32) -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("add_rms_norm_f32");
        let cfg = LaunchConfig { grid_dim: (nrows as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let (nc, e) = (ncols as i32, eps);
        let mut b2 = self.gpu.stream.launch_builder(&f);
        b2.arg(a).arg(b).arg(w).arg(&mut *res).arg(&mut *dst).arg(&nc).arg(&e);
        unsafe { b2.launch(cfg)?; }
        Ok(())
    }

    /// L2 norm per row (head_dim), no weight.
    pub fn l2_norm(&self, x: &CudaSlice<f32>, dst: &mut CudaSlice<f32>, ncols: usize, nrows: usize,
                   eps: f32) -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("l2_norm_f32");
        let cfg = LaunchConfig { grid_dim: (nrows as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let (nc, e) = (ncols as i32, eps);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(x).arg(dst).arg(&nc).arg(&e);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// L2-norm with blockDim=32 (warp-tree reduction) — BIT-IDENTICAL to gdn_prep_decode_f32's
    /// per-warp L2 norm. The verify path MUST use this to match decode's FP accumulation order:
    /// l2_norm at blockDim=256 produces a different shfl-tree reduction of the 128-element
    /// squared-sum (pairwise tree vs serial-4-then-warp-tree), causing ULP differences that
    /// propagate through gdn_scan and flip argmax on marginal logits.
    pub fn l2_norm_decode(&self, x: &CudaSlice<f32>, dst: &mut CudaSlice<f32>, ncols: usize,
                          nrows: usize, eps: f32) -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("l2_norm_f32");
        let cfg = LaunchConfig { grid_dim: (nrows as u32, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
        let (nc, e) = (ncols as i32, eps);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(x).arg(dst).arg(&nc).arg(&e);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// RoPE NEOX in-place. x:[head_dim, n_heads, n_tokens], pos:[n_tokens].
    pub fn rope_neox(&self, x: &mut CudaSlice<f32>, pos: &CudaSlice<i32>, head_dim: usize,
                     n_dims: usize, n_heads: usize, n_tokens: usize, freq_base: f32, freq_scale: f32)
                     -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("rope_neox_f32");
        let theta_scale = (freq_base).powf(-2.0 / n_dims as f32);
        let grid = (n_heads * n_tokens) as u32;
        let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: ((head_dim / 2) as u32, 1, 1), shared_mem_bytes: 0 };
        let (hd, nd, nh) = (head_dim as i32, n_dims as i32, n_heads as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(x).arg(pos).arg(&hd).arg(&nd).arg(&nh).arg(&theta_scale).arg(&freq_scale);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    pub fn silu_mul(&self, gate: &CudaSlice<f32>, up: &CudaSlice<f32>, dst: &mut CudaSlice<f32>, n: usize)
                    -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("silu_mul_f32");
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let ni = n as i32;
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(gate).arg(up).arg(dst).arg(&ni);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// FFN SwiGLU epilogue fusion (RANK3 LEVER 2): `dst = silu(gate*gs) * (up*us)` in ONE launch,
    /// folding the per-tensor NVFP4 macro-scale (`gs`,`us`) that would otherwise be two separate
    /// `scale_inplace` launches on the gate/up matmul outputs. BIT-IDENTICAL to
    /// scale_inplace(gate,gs); scale_inplace(up,us); silu_mul(gate,up,dst) — identical float ops in
    /// identical order. For non-NVFP4 weights gs==us==1.0 -> identical to `silu_mul`. Net: -2
    /// launches per dense FFN layer (the gate+up post-matmul scales).
    pub fn silu_mul_scaled(&self, gate: &CudaSlice<f32>, up: &CudaSlice<f32>, gs: f32, us: f32,
                           dst: &mut CudaSlice<f32>, n: usize) -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("silu_mul_scaled_f32");
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let ni = n as i32;
        let (gsf, usf) = (gs, us);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(gate).arg(up).arg(&gsf).arg(&usf).arg(dst).arg(&ni);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// swigluoai (MiniMax-M3 / GPT-OSS): clamped SwiGLU epilogue, math 1:1 vs llama.cpp
    /// ggml_cuda_op_swiglu_oai_single. `dst = swish_alpha(clamp(gate*gs)) * (1 + clamp(up*us))`.
    /// gs/us fold the NVFP4 macro-scales exactly like `silu_mul_scaled`.
    #[allow(clippy::too_many_arguments)]
    pub fn swigluoai_mul_scaled(&self, gate: &CudaSlice<f32>, up: &CudaSlice<f32>, gs: f32, us: f32,
                                alpha: f32, limit: f32, dst: &mut CudaSlice<f32>, n: usize)
                                -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("swigluoai_mul_scaled_f32");
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let ni = n as i32;
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(gate).arg(up).arg(&gs).arg(&us).arg(&alpha).arg(&limit).arg(dst).arg(&ni);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// RANK2 LEVER (q8_1 quant-fold): SwiGLU epilogue that EMITS the q8_1 quantization of `act`
    /// directly (aq int8 [n] + ad f32 [n/32]), so ffn_down's standalone `quantize_q8_1` launch is
    /// removed — the down-proj activation has one consumer, so the quant folds into the producer for
    /// free (no extra HBM read; no f32 `act` write). gs/us fold the gate/up NVFP4 macro-scales like
    /// `silu_mul_scaled`. BIT-IDENTICAL q8_1 to silu_mul_scaled(...) then quantize_q8_1(...). Only
    /// valid when ffn_down uses the q8_1 dp4a/mmvq path; the caller checks `uses_q8_1_fast(ffn_down)`.
    /// n must be a multiple of 32 (n_ff always is).
    pub fn silu_mul_scaled_q8_1(&self, gate: &CudaSlice<f32>, up: &CudaSlice<f32>, gs: f32, us: f32,
                                n: usize)
                                -> Result<(CudaSlice<i8>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let f = self.func("silu_mul_scaled_q8_1");
        let nblk = n / 32;
        let mut aq = self.alloc_uninit::<i8>(n)?;       // full-overwrite output
        let mut ad = self.alloc_uninit::<f32>(nblk)?;   // full-overwrite output
        // WARP-PER-BLOCK kernel: one warp (32 lanes) per 32-block -> n threads total.
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let (gsf, usf, ni) = (gs, us, n as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(gate).arg(up).arg(&gsf).arg(&usf).arg(&mut aq).arg(&mut ad).arg(&ni);
        unsafe { b.launch(cfg)?; }
        Ok((aq, ad))
    }

    pub fn add(&self, a: &CudaSlice<f32>, b_in: &CudaSlice<f32>, dst: &mut CudaSlice<f32>, n: usize)
               -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("add_f32");
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let ni = n as i32;
        let mut bld = self.gpu.stream.launch_builder(&f);
        bld.arg(a).arg(b_in).arg(dst).arg(&ni);
        unsafe { bld.launch(cfg)?; }
        Ok(())
    }

    pub fn mul(&self, a: &CudaSlice<f32>, b_in: &CudaSlice<f32>, dst: &mut CudaSlice<f32>, n: usize)
               -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("mul_f32");
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let ni = n as i32;
        let mut bld = self.gpu.stream.launch_builder(&f);
        bld.arg(a).arg(b_in).arg(dst).arg(&ni);
        unsafe { bld.launch(cfg)?; }
        Ok(())
    }

    /// Unified weight-tensor matmul: dispatches quant tensors to qmatvec (weights packed) and
    /// float tensors to cuBLASLt. y[m,out] = x[m,in] @ W[out,in]^T.
    pub fn matmul(&self, w: &crate::model::GpuTensor, x: &CudaSlice<f32>, m: usize)
                  -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        use crate::model::GpuTensor;
        let in_f = w.in_features();
        let out_f = w.out_features();
        // PREFILL (T>1) ROOT FIX: batched tensor-core int8 GEMM. Decodes each weight tile to int8
        // in smem ONCE and reuses across all tokens via mma — vs the dp4a matvec's per-token weight
        // re-read. Only the 4 daily-hot dtypes; m=1 decode keeps dp4a (it's bandwidth-bound, mma
        // gives nothing). Quantize the activation once here then call the GEMM.
        // m cutoff FIXED at 16: the m=4 MMA-verify A/B (2026-07-06, was BW24_GEMM_M) measured
        // NEGATIVE — the MMA tile grid starves at m=4 (BN=256 -> grid.y=1) and its FP order
        // shifted verify argmax at tight margins. Do not lower without re-running that battery.
        #[allow(non_snake_case)]
        let GEMM_M_THRESHOLD = 16usize;

        // PREFILL GEMM (m>=16). ACCURACY-FIRST dispatch (2026-06-28, prefill-gemm-beat-research wf
        // wllbyo6vc step 1): the int8 W4A8 GEMM (qmatvec_gemm, q8_1 activation, s32 accumulate) is
        // ACCURATE (prefill logit maxdiff 0.159, < dp4a 0.55) and the default. The FP4 W4A4 mxf4 path
        // (try_fp4_gemm) quantizes the ACTIVATION to e2m1 4-bit (8 magnitude levels) -> maxdiff 1.0
        // when combined — a real accuracy loss, NOT a math bug. So FP4-W4A4 is taken ONLY under the
        // explicit BW24_FP4 opt-in AND it must come SECOND (int8 W4A8 is the correct default for NVFP4).
        // The workflow plan rebuilds the FP4 path (kill per-K repack, widen K, deepen pipeline, TMA) to
        // be both fast AND accurate; until then NVFP4 prefill defaults to the accurate int8 GEMM.
        // TINY-OUT_F GUARD (2026-06-28, ncu trace): the tiling GEMM's grid is (ceil(out_f/BM=64),
        // ceil(m/BN=256)). For tiny out_f (ssm_beta/ssm_alpha out_f=num_v_heads~32), grid.x=1 -> only
        // ceil(m/256) CTAs (e.g. 2 for m=512) on 82 SMs = 0.39% SM throughput, 852us EACH (measured
        // worst offender). The dp4a path grids (out_f, m) = far more CTAs, filling the GPU. So route
        // out_f < 2*BM to dp4a (skip the tiling GEMM which structurally can't fill the SMs here).
        const GEMM_MIN_OUT_F: usize = 128;   // 2*BM; below this the GEMM grid.x starves the 82 SMs
        // VENDORED llama MMQ prefill GEMMs. NVFP4 W4A8 is DEFAULT-ON (2026-07-05 flip: same int8
        // accuracy class as the int8 GEMM below at ~1.9x pp512, rp-loader coexists with the A6
        // repack; BW24_MMQ_W4A8=0 = escape hatch). W4A4 mxf4nvf4 + Q4_K/Q5_K stay behind BW24_MMQ=1.
        // The env policy lives in mmq_supports/qmatvec_mmq. Feeds raw f32 activation `x` (the
        // launcher quantizes internally). out_f>=MMQ_Y/2 keeps the tile grid from starving the SMs.
        // FP8-ACT PREFILL (BW24_PP_FP8=1, probe verdict 2026-07-08): F8-E4M3-origin projections
        // carry their raw e4m3 device bytes (the `fp8` operand stashed at load next to the Q8_0
        // re-encode) — cuBLASLt FP8 TN at 620-795 TF vs 47-72 TF for this class's int8 GEMM.
        // Weight side EXACT (checkpoint bytes); activation rides ONE per-batch e4m3 scale
        // (amax/448) folded with weight_scale in-GEMM. Prefill only; decode keeps Q8_0 untouched.
        if m >= GEMM_M_THRESHOLD {
            if let Some(y) = self.try_fp8_gemm(w, x, m)? { return Ok(y); }
        }
        if m >= GEMM_M_THRESHOLD && out_f >= GEMM_MIN_OUT_F && self.mmq_supports(w) {
            return self.qmatvec_mmq(w, x, m);
        }
        if m >= GEMM_M_THRESHOLD && out_f >= GEMM_MIN_OUT_F && self.gemm_supports(w) {
            let (aq, ad) = self.quantize_q8_1(x, m, in_f)?;
            return self.qmatvec_gemm(w, &aq, &ad, m);
        }
        // FP4 W4A4 only as an explicit speed/accuracy tradeoff opt-in, and only if the int8 GEMM
        // above didn't already handle this weight (e.g. NVFP4 with in_f%64!=0, or BW24_NO_GEMM set).
        if m >= GEMM_M_THRESHOLD {
            if let Some(y) = self.try_fp4_gemm(w, x, m, in_f, out_f)? { return Ok(y); }
        }
        // Stage-B fast int8 dp4a is the DEFAULT since 2026-07-08 (it has been the daily path
        // for weeks; the old opt-in flag was a silent-slow-path landmine). BW24_FAST=0 reverts
        // to Stage-A f32-dequant (the correctness oracle path).
        let fast = std::env::var("BW24_FAST").as_deref() != Ok("0");
        // PERF-3 decode-GEMV: m=1 warp-per-row MMVQ (BW24_MMVQ). The big decode matvecs reach
        // `matmul` directly (ffn_down, lm_head output, wo), so route them here too — not only the
        // matmul_pre siblings. qmatvec_mmvq_raw quantizes the activation internally (q8_1) like the
        // _fast paths; the NVFP4 macro-scale is applied by the `scale != 1.0` block below.
        if m == 1 && fast {
            if let GpuTensor::Quant { bytes, qtype, row_bytes, rp, scale, .. } = w {
                if self.mmvq_supports(*qtype) {
                    // NVFP4 macro-scale rides the kernel's fused epilogue arg (one launch total);
                    // non-NVFP4 has scale==1.0 so qmatvec_mmvq skips scale_inplace either way.
                    let (aq, ad) = self.quantize_q8_1(x, m, in_f)?;
                    return self.qmatvec_mmvq(bytes, &aq, &ad, m, in_f, out_f, *qtype, *row_bytes, *scale, *rp);
                }
            }
        }
        // BATCHED weight-resident matvec for the m=2-4 band (the MTP/verify forward's ffn_down, wo, and
        // lm_head `output` reach `matmul` directly at m=T=2..4). Walks the weight ONCE, dp4a vs all m
        // activation columns -> 1 weight read for m tokens (vs grid.y=m re-reading m times below). Quant
        // the activation once here (q8_1) like the _fast paths; macro-scale applied via the scale!=1.0
        // block below. BW24_NO_BATCHED -> per-m path.
        //
        // DECODE-PARITY GATE (2026-07-07, the 9B synth K=3/4/6 spec FAIL root cause): the batched
        // kernels are bit-identical per (token,row) to MMVQ's 32-thread warp reduce, NOT to the
        // dp4a kernels' 128-thread two-level reduce. Without BW24_MMVQ the m=1 decode chain rides
        // dp4a, so a verify riding batched here has a DIFFERENT FP order than the decode it must
        // match bit-for-bit — greedy spec flips at tight-margin tokens (the old HANDOVER "ENV LAW:
        // FAST+MMVQ both required" footgun, closed here). Parity law: the m>1 kernel CLASS must be
        // a pure function of (dtype, env) equal to the m=1 class — batched iff MMVQ. Without MMVQ
        // the verify falls to the per-m grid.y=m dp4a path below (each column = the exact m=1
        // dp4a program). BW24_MMVQ=1 (the daily config) is dispatch-unchanged.
        if (2..=8).contains(&m) && fast && std::env::var("BW24_NO_BATCHED").is_err()
            && (m <= 4 || Self::b8_enabled()) {
            if let GpuTensor::Quant { bytes, qtype, row_bytes, rp, .. } = w {
                if self.batched_supports(*qtype) && self.mmvq_supports(*qtype) {
                    let mcols = Self::batched_mcols(m);
                    let (aq, ad) = self.quantize_q8_1(x, m, in_f)?;
                    let mut y = self.qmatvec_mmvq_batched(bytes, &aq, &ad, m, in_f, out_f, *qtype, *row_bytes, mcols, 1.0, *rp)?;
                    if let GpuTensor::Quant { scale, .. } = w {
                        if *scale != 1.0 { self.scale_inplace(&mut y, *scale, m * out_f)?; }
                    }
                    return Ok(y);
                }
            }
        }
        // F8-E4M3 (BW24_ST_E4M3) catch-all for the m<16 band the arms above didn't take (m=9..15,
        // the K=8 verify tier; or m=2..8 under BW24_NO_BATCHED/BW24_B8=0): grid.y=m e4m3 mmvq —
        // the SAME per-(token,row) program as the m=1 decode launch (bit-identical by construction),
        // weight re-read m times (rare tier; exactness over bandwidth here). There is no _dp4a twin
        // for this dtype, so the generic match below must never see it under `fast`.
        if fast {
            if let GpuTensor::Quant { bytes, qtype, row_bytes, scale, .. } = w {
                if *qtype == QT_F8_E4M3 {
                    let (aq, ad) = self.quantize_q8_1(x, m, in_f)?;
                    return self.qmatvec_mmvq(bytes, &aq, &ad, m, in_f, out_f, *qtype, *row_bytes,
                                             *scale, false);
                }
            }
        }
        let mut y = match w {
            GpuTensor::Quant { bytes, qtype, row_bytes, .. } if fast && *qtype == QT_Q8_0 =>
                self.qmatvec_q8_0_fast(bytes, x, m, in_f, out_f, *row_bytes)?,
            GpuTensor::Quant { bytes, qtype, row_bytes, .. } if fast && *qtype == QT_Q4_K =>
                self.qmatvec_q4_K_fast(bytes, x, m, in_f, out_f, *row_bytes)?,
            GpuTensor::Quant { bytes, qtype, row_bytes, .. } if fast && *qtype == QT_Q6_K =>
                self.qmatvec_q6_K_fast(bytes, x, m, in_f, out_f, *row_bytes)?,
            GpuTensor::Quant { bytes, qtype, row_bytes, .. } if fast && *qtype == QT_Q5_K =>
                self.qmatvec_q5_K_fast(bytes, x, m, in_f, out_f, *row_bytes)?,
            GpuTensor::Quant { bytes, qtype, row_bytes, .. } if fast && *qtype == QT_Q3_K =>
                self.qmatvec_q3_K_fast(bytes, x, m, in_f, out_f, *row_bytes)?,
            GpuTensor::Quant { bytes, qtype, row_bytes, rp, .. } if fast && *qtype == QT_NVFP4 =>
                self.qmatvec_dp4a_named(
                    if *rp { "qmatvec_nvfp4_dp4a_rp" } else { "qmatvec_nvfp4_dp4a" },
                    bytes, x, m, in_f, out_f, *row_bytes)?,
            // IQ4_XS optional fast path (gate behind a second env var; Stage-A is the default).
            GpuTensor::Quant { bytes, qtype, row_bytes, .. }
                if fast && *qtype == QT_IQ4_XS && std::env::var("BW24_IQ_FAST").is_ok() =>
                self.qmatvec_iq4_XS_fast(bytes, x, m, in_f, out_f, *row_bytes)?,
            // B3: IQ3_S and (default) IQ4_XS use the Stage-A f32 dequant-in-kernel path. There is
            // NO qmatvec_iq3_s_dp4a / (default) iq4_XS fast kernel — do NOT add a `*qtype == QT_IQ3_S`
            // (or unconditional QT_IQ4_XS) fast guard here without first writing the matching kernel,
            // or func() will panic "kernel ... not in any fatbin".
            GpuTensor::Quant { bytes, qtype, row_bytes, rp, .. } =>
                // Stage-A generic: repacked NVFP4 uses the device-side split-plane tag (the
                // deq(row,j) form cannot address the planes; same value/product order).
                self.qmatvec(bytes, x, m, in_f, out_f,
                             if *rp && *qtype == QT_NVFP4 { QT_NVFP4_RP } else { *qtype },
                             *row_bytes)?,
            GpuTensor::Float { data, .. } => self.linear(x, data, m, in_f, out_f)?,
            // BW24_FULL_PREC bf16-resident weight: dequant-on-use to f32 scratch, then the same
            // cuBLASLt f32 GEMV as the Float arm.
            GpuTensor::FloatBf16 { data, .. } =>
                self.linear_bf16_chunked(x, data, m, in_f, out_f, false)?,
        };
        // NVFP4 per-tensor macro-scale (post-matmul). scale==1.0 for all other quants/float -> no-op.
        if let GpuTensor::Quant { scale, .. } = w {
            if *scale != 1.0 { self.scale_inplace(&mut y, *scale, m * out_f)?; }
        }
        Ok(y)
    }

    /// True if `w` would take the int8-dp4a fast path under BW24_FAST (so its activation can be
    /// pre-quantized once and shared across sibling matmuls via `matmul_pre`).
    pub fn uses_q8_1_fast(&self, w: &crate::model::GpuTensor) -> bool {
        use crate::model::GpuTensor;
        if std::env::var("BW24_FAST").as_deref() == Ok("0") { return false; }
        match w {
            GpuTensor::Quant { qtype, .. } => matches!(*qtype,
                QT_Q8_0 | QT_Q4_K | QT_Q6_K | QT_Q5_K | QT_Q3_K | QT_NVFP4 | QT_F8_E4M3)
                || (*qtype == QT_IQ4_XS && std::env::var("BW24_IQ_FAST").is_ok()),
            GpuTensor::Float { .. } | GpuTensor::FloatBf16 { .. } => false,
        }
    }

    /// matmul with a PRE-QUANTIZED q8_1 activation (aq,ad from `quantize_q8_1`). Skips the
    /// per-matmul re-quantize so sibling matmuls that share an input (gate+up share `z`;
    /// q/k/v + wqkv/gate/beta/alpha share `h`) quantize ONCE. Caller MUST have checked
    /// `uses_q8_1_fast(w)`; falls back to plain `matmul` otherwise (Stage-A / Float / non-fast).
    pub fn matmul_pre(&self, w: &crate::model::GpuTensor, aq: &CudaSlice<i8>, ad: &CudaSlice<f32>,
                      x_fallback: &CudaSlice<f32>, m: usize)
                      -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        use crate::model::GpuTensor;
        // FP8-ACT PREFILL (BW24_PP_FP8=1): same arm as `matmul` — the fp8 operand needs the RAW
        // f32 activation (per-batch e4m3 quant differs from q8_1), so x_fallback not aq/ad.
        if m >= 16 {
            if let Some(y) = self.try_fp8_gemm(w, x_fallback, m)? { return Ok(y); }
        }
        // VENDORED llama MMQ prefill GEMMs (NVFP4 W4A8 default-on; W4A4/k-quant behind BW24_MMQ=1
        // — policy in mmq_supports) — use the RAW f32 activation (their own internal quant:
        // q8_1 D4 for NVFP4 W4A8, FP8/UE4M3 for W4A4, q8_1 DS4 for Q4_K/Q5_K), so x_fallback not aq/ad.
        if m >= 16 && w.out_features() >= 128 && self.mmq_supports(w) {
            return self.qmatvec_mmq(w, x_fallback, m);
        }
        // Stage-C FP4 prefill (BW24_FP4): native mxf4 GEMM needs the f32 activation (FP4-quant differs
        // from q8_1), so re-quantize from x_fallback rather than reuse aq/ad. NVFP4 only, m>=16.
        if m >= 16 {
            if let Some(y) = self.try_fp4_gemm(w, x_fallback, m, w.in_features(), w.out_features())? {
                return Ok(y);
            }
        }
        // Prefill GEMM root fix: if T>1 and the dtype has a GEMM kernel, batch via tensor cores
        // (reuses the already-quantized aq/ad — no extra quantize). m=1 falls through to dp4a.
        if m >= 16 && self.gemm_supports(w) {
            return self.qmatvec_gemm(w, aq, ad, m);
        }
        if !self.uses_q8_1_fast(w) { return self.matmul(w, x_fallback, m); }
        let in_f = w.in_features();
        let out_f = w.out_features();
        let (bytes, qtype, row_bytes, scale, rp) = match w {
            GpuTensor::Quant { bytes, qtype, row_bytes, scale, rp, .. } => (bytes, *qtype, *row_bytes, *scale, *rp),
            _ => unreachable!("uses_q8_1_fast guaranteed Quant"),
        };
        // PERF-3 decode-GEMV: warp-per-row MMVQ for the m=1 decode arm, gated behind BW24_MMVQ.
        // Only the 4 daily-hot dtypes have an _mmvq kernel (Q8_0/Q4_K/Q6_K/NVFP4); Q5_K/Q3_K/IQ4_XS
        // keep _dp4a (the oracle/fallback). Bit-equivalent to _dp4a up to f32 reduction order.
        if m == 1 && self.mmvq_supports(qtype) {
            return self.qmatvec_mmvq(bytes, aq, ad, m, in_f, out_f, qtype, row_bytes, scale, rp);
        }
        // BATCHED weight-resident matvec for the m=2-4 band (the MTP/verify forward: full_attn_verify
        // and decode_step_t run their projections at m=T=k=2..4). The plain _dp4a path below launches
        // grid.y=m INDEPENDENT blocks per output row -> the weight row is re-read m times from HBM/L2.
        // The _b2/_b4 kernels walk the weight ONCE and dp4a vs all m activation columns, so m tokens
        // cost ~1 weight read instead of m (decode is weight-BW-bound). BIT-IDENTICAL per (token,row)
        // to the _mmvq path (32-thread warp reduce — NOT the dp4a 128-thread reduce below).
        // m=2 -> mcols=2; m∈{3,4} -> mcols=4; m∈{5..8} -> mcols=8 (kernel guards c>=m).
        // BW24_NO_BATCHED forces the per-m grid.y=m path (the A/B reference); BW24_B8=0 keeps
        // m=5..8 on the old per-m path (b8-tier-only seam).
        // DECODE-PARITY GATE (2026-07-07): batched iff mmvq_supports — see matmul's parity note.
        // Without BW24_MMVQ, m=1 decode rides dp4a (the arm below at m=1); the verify must ride
        // the SAME class per column (grid.y=m dp4a = the exact m=1 dp4a program per column).
        if (2..=8).contains(&m) && self.batched_supports(qtype) && self.mmvq_supports(qtype)
            && std::env::var("BW24_NO_BATCHED").is_err()
            && (m <= 4 || Self::b8_enabled()) {
            let mcols = Self::batched_mcols(m);
            return self.qmatvec_mmvq_batched(bytes, aq, ad, m, in_f, out_f, qtype, row_bytes, mcols, scale, rp);
        }
        // F8-E4M3 catch-all (m=9..15 / batched-disabled seams): grid.y=m e4m3 mmvq — this dtype
        // has NO _dp4a twin, and per (token,row) the mmvq body is the exact m=1 decode program.
        if qtype == QT_F8_E4M3 {
            return self.qmatvec_mmvq(bytes, aq, ad, m, in_f, out_f, qtype, row_bytes, scale, rp);
        }
        let name = match qtype {
            QT_Q8_0 => "qmatvec_q8_0_dp4a", QT_Q4_K => "qmatvec_q4_K_dp4a",
            QT_Q6_K => "qmatvec_q6_K_dp4a", QT_Q5_K => "qmatvec_q5_K_dp4a",
            QT_Q3_K => "qmatvec_q3_K_dp4a",
            QT_NVFP4 => if rp { "qmatvec_nvfp4_dp4a_rp" } else { "qmatvec_nvfp4_dp4a" },
            QT_IQ4_XS => "qmatvec_iq4_XS_dp4a",
            _ => unreachable!(),
        };
        let f = self.func(name);
        let mut y = self.alloc_uninit::<f32>(m * out_f)?;  // full-overwrite GEMM output: skip memset
        let cfg = LaunchConfig { grid_dim: (out_f as u32, m as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
        let (inf, outf, mi, rb) = (in_f as i32, out_f as i32, m as i32, row_bytes as i64);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(bytes).arg(aq).arg(ad).arg(&mut y).arg(&inf).arg(&outf).arg(&mi).arg(&rb);
        unsafe { b.launch(cfg)?; }
        if scale != 1.0 { self.scale_inplace(&mut y, scale, m * out_f)?; }
        Ok(y)
    }

    /// DECODE-EXACT matmul at any m: guarantees the SAME warp-per-row (MMVQ, 32-thread) FP
    /// accumulation order as the T=1 decode path for EVERY token row. The spec-decode verify MUST
    /// use this for linear-attn projections to be bit-identical to greedy decode. The dp4a kernel
    /// (128 threads, two-level reduction) used by `matmul`/`matmul_pre` at m>=5 has a different
    /// shfl-tree shape that produces ULP differences propagating through gdn_scan into argmax flips.
    /// The MMVQ kernel with grid.y=m already processes each row independently (same 32-thread warp
    /// reduce as m=1); this method just forces that path unconditionally.
    pub fn matmul_decode_exact(&self, w: &crate::model::GpuTensor, x: &CudaSlice<f32>, m: usize)
                               -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        use crate::model::GpuTensor;
        // FLOAT tensors (35B ssm_beta/ssm_alpha on every linear layer, F32 ne=[2048,32]): the
        // generic path is cuBLASLt, whose reduction splits are n-DEPENDENT — m=1 vs m=2 col-0
        // outputs differ in every bit (probe 2026-07-06: 32/32 bit-diff, maxdiff 3.5e-3), which
        // shifted 35B verify logits 0.26-0.56 vs eager and flipped greedy at tight margins (the
        // p3 spec FAIL). Decode-exact contract: per-COLUMN m=1 cuBLASLt calls — each column's
        // reduction is the exact kernel the T=1 decode path runs, so verify==decode bit-for-bit.
        // m<=10 here (K+2 verify tier), so the extra launches are a handful of 4us gemvs.
        if let GpuTensor::Float { data, .. } = w {
            return self.linear_decode_exact(x, data, m, w.in_features(), w.out_features());
        }
        // BW24_FULL_PREC bf16-resident weight: dequant-on-use, then the per-column decode-exact
        // float linear (same n-independent reduction contract as the Float arm above).
        if let GpuTensor::FloatBf16 { data, .. } = w {
            let (in_f, out_f) = (w.in_features(), w.out_features());
            return self.linear_bf16_chunked(x, data, m, in_f, out_f, true);
        }
        if !self.uses_q8_1_fast(w) { return self.matmul(w, x, m); }
        let in_f = w.in_features();
        let out_f = w.out_features();
        let (bytes, qtype, row_bytes, scale, rp) = match w {
            GpuTensor::Quant { bytes, qtype, row_bytes, scale, rp, .. } => (bytes, *qtype, *row_bytes, *scale, *rp),
            _ => return self.matmul(w, x, m),
        };
        let (aq, ad) = self.quantize_q8_1(x, m, in_f)?;
        // Batched weight-resident matvec for m=2-8: BIT-IDENTICAL per (token,row) to MMVQ (exact
        // integer dp4a, same warp reduce — kernel-check gate rel=0.00e0), one weight read for m
        // tokens. The dispatch the divergence fix must avoid is dp4a's 128-thread two-level
        // reduce, NOT this. m=5..8 is the K=4..7 spec-verify tier (b8): pre-b8 T=5 fell to the
        // grid.y=m per-row MMVQ below = 5 full weight reads/launch — the measured 27B K=4 cliff.
        // DECODE-PARITY GATE (2026-07-07): batched (MMVQ-class order) only when the m=1 decode
        // chain rides MMVQ too — without BW24_MMVQ decode is dp4a, so the exact-contract here
        // must be per-column dp4a (matmul_pre fallthrough), not the MMVQ order.
        if (2..=8).contains(&m) && self.batched_supports(qtype) && self.mmvq_supports(qtype)
            && std::env::var("BW24_NO_BATCHED").is_err()
            && (m <= 4 || Self::b8_enabled()) {
            let mcols = Self::batched_mcols(m);
            return self.qmatvec_mmvq_batched(bytes, &aq, &ad, m, in_f, out_f, qtype, row_bytes, mcols, scale, rp);
        }
        if self.mmvq_supports(qtype) {
            // MMVQ at grid.y=m: each row is processed by its own warp independently — same 32-thread
            // accumulation + warp_reduce_sum as m=1 decode. Bit-identical per row.
            return self.qmatvec_mmvq(bytes, &aq, &ad, m, in_f, out_f, qtype, row_bytes, scale, rp);
        }
        // Fallback for non-MMVQ quant types (Q5_K, Q3_K): use dp4a (the only available kernel).
        // These types are not used in the 27B's linear-attn NVFP4+Q4_K layers.
        self.matmul_pre(w, &aq, &ad, x, m)
    }

    /// Like `matmul_pre` but RETURNS THE RAW (un-macro-scaled) matmul output together with the
    /// per-tensor NVFP4 scale, instead of applying `scale_inplace` internally. Used by the fused
    /// SwiGLU epilogue (RANK3 LEVER 2) so the gate/up scales fold into one `silu_mul_scaled` launch.
    /// `Some((y_raw, scale))` only on the m==1 decode fast path (mmvq / dp4a) where the scale is a
    /// separate post-launch op we can defer; returns `None` for every other path (prefill GEMM, FP4
    /// GEMM, Stage-A, Float) so the caller falls back to the scaled `matmul_pre` + `silu_mul`.
    /// DUAL gate+up NVFP4 matvec (mm-fusion): ONE launch computes both projections (same
    /// activation, same shape) — grid.y selects the tensor. Bit-identical per element to two
    /// mr2 launches at m=1. Returns (gate_raw, up_raw) un-scaled (caller folds the two macro
    /// scales into the SwiGLU epilogue, same as the matmul_pre_noscale contract). None unless
    /// both tensors are NVFP4 q8_1-fast with identical (in_f, out_f, row_bytes) and m==1.
    pub fn matmul_pre_dual_noscale(&self, w0: &crate::model::GpuTensor, w1: &crate::model::GpuTensor,
                                   aq: &CudaSlice<i8>, ad: &CudaSlice<f32>, m: usize)
        -> Result<Option<((CudaSlice<f32>, f32), (CudaSlice<f32>, f32))>, Box<dyn std::error::Error>> {
        use crate::model::GpuTensor;
        if m != 1 || !self.uses_q8_1_fast(w0) || !self.uses_q8_1_fast(w1) { return Ok(None); }
        let (in_f, out_f) = (w0.in_features(), w0.out_features());
        if w1.in_features() != in_f || w1.out_features() != out_f { return Ok(None); }
        let (b0, q0, rb0, s0, rp0) = match w0 {
            GpuTensor::Quant { bytes, qtype, row_bytes, scale, rp, .. } => (bytes, *qtype, *row_bytes, *scale, *rp),
            _ => return Ok(None),
        };
        let (b1, q1, rb1, s1, rp1) = match w1 {
            GpuTensor::Quant { bytes, qtype, row_bytes, scale, rp, .. } => (bytes, *qtype, *row_bytes, *scale, *rp),
            _ => return Ok(None),
        };
        if q0 != QT_NVFP4 || q1 != QT_NVFP4 || rb0 != rb1 || rp0 != rp1 { return Ok(None); }
        const ROWS_PER_BLOCK: u32 = 4;   // matches BW24_MMVQ_ROWS in qmatvec.cu
        const RPW: u32 = 2;
        let rows_per_block = ROWS_PER_BLOCK * RPW;
        let f = self.func(if rp0 { "qmatvec_nvfp4_mmvq_dual_mr2_rp" } else { "qmatvec_nvfp4_mmvq_dual_mr2" });
        let mut y0 = self.alloc_uninit::<f32>(out_f)?;
        let mut y1 = self.alloc_uninit::<f32>(out_f)?;
        let cfg = LaunchConfig {
            grid_dim: ((out_f as u32 + rows_per_block - 1) / rows_per_block, 2, 1),
            block_dim: (32, ROWS_PER_BLOCK, 1), shared_mem_bytes: 0,
        };
        let (inf, outf, mi, rb) = (in_f as i32, out_f as i32, 1i32, rb0 as i64);
        // noscale contract: the caller folds s0/s1 into the SwiGLU epilogue — the kernel's fused
        // yscale args stay 1.0 here (they exist for the single-tensor callers).
        let one = 1.0f32;
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(b0).arg(b1).arg(aq).arg(ad).arg(&mut y0).arg(&mut y1)
         .arg(&inf).arg(&outf).arg(&mi).arg(&rb).arg(&one).arg(&one);
        unsafe { b.launch(cfg)?; }
        Ok(Some(((y0, s0), (y1, s1))))
    }

    /// FUSED Q8_0 m=1 matvec PAIR with UNEQUAL out_f (trunk launch-fusion, 2026-07-05). Folds two
    /// same-input q8_0 projections (35B trunk: wqkv+wqkv_gate 8192/4096, gate_shexp+up_shexp
    /// 512/512) into ONE launch via a block-offset split (blocks [0,nb0) -> w0, rest -> w1) — the
    /// dual-mr2 recipe with the same-out_f restriction lifted. Per (tensor,row) the kernel body is
    /// qmatvec_q8_0_mmvq VERBATIM -> BIT-IDENTICAL to two separate m=1 launches. Returns None when
    /// ineligible (not both Q8_0 / in_f mismatch / BW24_MMVQ off / BW24_Q8_DUAL=0) — caller falls
    /// back to the per-tensor path.
    pub fn matmul_q8_fused2(&self, w0: &crate::model::GpuTensor, w1: &crate::model::GpuTensor,
                            aq: &CudaSlice<i8>, ad: &CudaSlice<f32>)
        -> Result<Option<(CudaSlice<f32>, CudaSlice<f32>)>, Box<dyn std::error::Error>> {
        let Some([p0, p1]) = self.q8_fused_params(&[w0, w1]) else { return Ok(None) };
        Ok(Some(self.q8_fused2_core(p0.0, p1.0, aq, ad, w0.in_features(), p0.1, p1.1, p0.2)?))
    }

    #[allow(clippy::too_many_arguments)]
    fn q8_fused2_core(&self, b0: &CudaSlice<u8>, b1: &CudaSlice<u8>,
                      aq: &CudaSlice<i8>, ad: &CudaSlice<f32>,
                      in_f: usize, out0: usize, out1: usize, row_bytes: usize)
        -> Result<(CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        const ROWS_PER_BLOCK: u32 = 4;   // matches BW24_MMVQ_ROWS in qmatvec.cu
        let nb0 = (out0 as u32).div_ceil(ROWS_PER_BLOCK);
        let nb1 = (out1 as u32).div_ceil(ROWS_PER_BLOCK);
        let f = self.func("qmatvec_q8_0_mmvq_fused2");
        let mut y0 = self.alloc_uninit::<f32>(out0)?;
        let mut y1 = self.alloc_uninit::<f32>(out1)?;
        let cfg = LaunchConfig { grid_dim: (nb0 + nb1, 1, 1), block_dim: (32, ROWS_PER_BLOCK, 1),
                                 shared_mem_bytes: 0 };
        let (inf, o0, o1, rbl) = (in_f as i32, out0 as i32, out1 as i32, row_bytes as i64);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(b0).arg(b1).arg(aq).arg(ad).arg(&mut y0).arg(&mut y1)
         .arg(&inf).arg(&o0).arg(&o1).arg(&rbl);
        unsafe { b.launch(cfg)?; }
        Ok((y0, y1))
    }

    /// f32-activation entry for the fused2 pair: quantizes x to q8_1 ONCE then runs the fused
    /// launch — replaces two `matmul(w, x, 1)` calls that would each re-quantize the same x
    /// (35B shared-expert gate+up per MoE layer per token). Same bits: quantize_q8_1 is
    /// deterministic, the fused body is the MMVQ kernel verbatim. None when ineligible (the
    /// callers' m==1-under-BW24_FAST dispatch would take MMVQ; anything else falls back).
    pub fn matmul_q8_fused2_x(&self, w0: &crate::model::GpuTensor, w1: &crate::model::GpuTensor,
                              x: &CudaSlice<f32>)
        -> Result<Option<(CudaSlice<f32>, CudaSlice<f32>)>, Box<dyn std::error::Error>> {
        if !self.uses_q8_1_fast(w0) || !self.uses_q8_1_fast(w1) { return Ok(None); }
        let Some([p0, p1]) = self.q8_fused_params(&[w0, w1]) else { return Ok(None) };
        let (aq, ad) = self.quantize_q8_1(x, 1, w0.in_features())?;
        Ok(Some(self.q8_fused2_core(p0.0, p1.0, &aq, &ad, w0.in_features(), p0.1, p1.1, p0.2)?))
    }

    /// Test entry for the kernel_check gate: launch the fused2 kernel from raw weight bytes,
    /// quantizing the f32 activation internally (mirrors qmatvec_mmvq_raw; no env gating).
    #[allow(clippy::too_many_arguments)]
    pub fn qmatvec_q8_fused2_raw(&self, b0: &CudaSlice<u8>, b1: &CudaSlice<u8>, x: &CudaSlice<f32>,
                                 in_f: usize, out0: usize, out1: usize, row_bytes: usize)
        -> Result<(CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let (aq, ad) = self.quantize_q8_1(x, 1, in_f)?;
        self.q8_fused2_core(b0, b1, &aq, &ad, in_f, out0, out1, row_bytes)
    }

    /// FUSED Q8_0 m=1 matvec TRIPLE (wq+wk+wv on the 35B full-attn layers: out_f 8192/512/512).
    /// Same block-offset recipe as `matmul_q8_fused2` with three ranges. BIT-IDENTICAL per
    /// (tensor,row) to three separate m=1 MMVQ launches.
    pub fn matmul_q8_fused3(&self, w0: &crate::model::GpuTensor, w1: &crate::model::GpuTensor,
                            w2: &crate::model::GpuTensor,
                            aq: &CudaSlice<i8>, ad: &CudaSlice<f32>)
        -> Result<Option<(CudaSlice<f32>, CudaSlice<f32>, CudaSlice<f32>)>, Box<dyn std::error::Error>> {
        let Some([p0, p1, p2]) = self.q8_fused_params(&[w0, w1, w2]) else { return Ok(None) };
        Ok(Some(self.q8_fused3_core(p0.0, p1.0, p2.0, aq, ad, w0.in_features(),
                                    p0.1, p1.1, p2.1, p0.2)?))
    }

    #[allow(clippy::too_many_arguments)]
    fn q8_fused3_core(&self, b0: &CudaSlice<u8>, b1: &CudaSlice<u8>, b2: &CudaSlice<u8>,
                      aq: &CudaSlice<i8>, ad: &CudaSlice<f32>,
                      in_f: usize, out0: usize, out1: usize, out2: usize, row_bytes: usize)
        -> Result<(CudaSlice<f32>, CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        const ROWS_PER_BLOCK: u32 = 4;
        let nb0 = (out0 as u32).div_ceil(ROWS_PER_BLOCK);
        let nb1 = (out1 as u32).div_ceil(ROWS_PER_BLOCK);
        let nb2 = (out2 as u32).div_ceil(ROWS_PER_BLOCK);
        let f = self.func("qmatvec_q8_0_mmvq_fused3");
        let mut y0 = self.alloc_uninit::<f32>(out0)?;
        let mut y1 = self.alloc_uninit::<f32>(out1)?;
        let mut y2 = self.alloc_uninit::<f32>(out2)?;
        let cfg = LaunchConfig { grid_dim: (nb0 + nb1 + nb2, 1, 1), block_dim: (32, ROWS_PER_BLOCK, 1),
                                 shared_mem_bytes: 0 };
        let (inf, o0, o1, o2, rbl) = (in_f as i32, out0 as i32, out1 as i32, out2 as i32, row_bytes as i64);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(b0).arg(b1).arg(b2).arg(aq).arg(ad).arg(&mut y0).arg(&mut y1).arg(&mut y2)
         .arg(&inf).arg(&o0).arg(&o1).arg(&o2).arg(&rbl);
        unsafe { b.launch(cfg)?; }
        Ok((y0, y1, y2))
    }

    /// Test entry for the kernel_check gate: fused3 from raw weight bytes (internal q8_1 quant).
    #[allow(clippy::too_many_arguments)]
    pub fn qmatvec_q8_fused3_raw(&self, b0: &CudaSlice<u8>, b1: &CudaSlice<u8>, b2: &CudaSlice<u8>,
                                 x: &CudaSlice<f32>, in_f: usize, out0: usize, out1: usize,
                                 out2: usize, row_bytes: usize)
        -> Result<(CudaSlice<f32>, CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let (aq, ad) = self.quantize_q8_1(x, 1, in_f)?;
        self.q8_fused3_core(b0, b1, b2, &aq, &ad, in_f, out0, out1, out2, row_bytes)
    }

    /// BATCHED twin of `matmul_q8_fused2` for the verify t=2-4 tier (BW24_SPEC_FUSED_T call
    /// sites, lane/close35b): ONE launch computes both same-input Q8_0 projections for m tokens.
    /// Per (tensor,token,row) the kernel body is q8_0_mmvq_batched VERBATIM with the identical
    /// row mapping (Q8_0's batched_variant is always "base") -> BIT-IDENTICAL to the two
    /// per-tensor _b2/_b4 launches `matmul_decode_exact` dispatches at m=2-4, with the caller's
    /// single shared q8_1 activation replacing two per-call re-quantizes (quantize_q8_1 is
    /// deterministic -> same bytes). None when ineligible (m outside 2..=4 / not both Q8_0 /
    /// in_f mismatch / BW24_MMVQ=0 / BW24_Q8_DUAL=0 / BW24_NO_BATCHED set — the last keeps
    /// dispatch parity: without batched kernels decode-exact runs grid.y=m MMVQ, and the fused
    /// twin must not introduce a batched program the reference path would not run).
    pub fn matmul_q8_fused2_t(&self, w0: &crate::model::GpuTensor, w1: &crate::model::GpuTensor,
                              aq: &CudaSlice<i8>, ad: &CudaSlice<f32>, m: usize)
        -> Result<Option<(CudaSlice<f32>, CudaSlice<f32>)>, Box<dyn std::error::Error>> {
        if !(2..=4).contains(&m) || std::env::var("BW24_NO_BATCHED").is_ok() { return Ok(None); }
        let Some([p0, p1]) = self.q8_fused_params(&[w0, w1]) else { return Ok(None) };
        Ok(Some(self.q8_fused2_t_core(p0.0, p1.0, aq, ad, m, w0.in_features(), p0.1, p1.1, p0.2)?))
    }

    #[allow(clippy::too_many_arguments)]
    fn q8_fused2_t_core(&self, b0: &CudaSlice<u8>, b1: &CudaSlice<u8>,
                        aq: &CudaSlice<i8>, ad: &CudaSlice<f32>, m: usize,
                        in_f: usize, out0: usize, out1: usize, row_bytes: usize)
        -> Result<(CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        const ROWS_PER_BLOCK: u32 = 4;   // matches BW24_MMVQ_ROWS in qmatvec.cu
        let nb0 = (out0 as u32).div_ceil(ROWS_PER_BLOCK);
        let nb1 = (out1 as u32).div_ceil(ROWS_PER_BLOCK);
        let f = self.func(if Self::batched_mcols(m) == 2 { "qmatvec_q8_0_mmvq_fused2_b2" }
                          else { "qmatvec_q8_0_mmvq_fused2_b4" });
        let mut y0 = self.alloc_uninit::<f32>(m * out0)?;
        let mut y1 = self.alloc_uninit::<f32>(m * out1)?;
        let cfg = LaunchConfig { grid_dim: (nb0 + nb1, 1, 1), block_dim: (32, ROWS_PER_BLOCK, 1),
                                 shared_mem_bytes: 0 };
        let (inf, o0, o1, mi, rbl) = (in_f as i32, out0 as i32, out1 as i32, m as i32, row_bytes as i64);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(b0).arg(b1).arg(aq).arg(ad).arg(&mut y0).arg(&mut y1)
         .arg(&inf).arg(&o0).arg(&o1).arg(&mi).arg(&rbl);
        unsafe { b.launch(cfg)?; }
        Ok((y0, y1))
    }

    /// Test entry for the kernel_check gate: fused2 batched from raw weight bytes (internal
    /// q8_1 quant of the [m, in_f] activation), no env gating.
    #[allow(clippy::too_many_arguments)]
    pub fn qmatvec_q8_fused2_t_raw(&self, b0: &CudaSlice<u8>, b1: &CudaSlice<u8>,
                                   x: &CudaSlice<f32>, m: usize,
                                   in_f: usize, out0: usize, out1: usize, row_bytes: usize)
        -> Result<(CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let (aq, ad) = self.quantize_q8_1(x, m, in_f)?;
        self.q8_fused2_t_core(b0, b1, &aq, &ad, m, in_f, out0, out1, row_bytes)
    }

    /// BATCHED twin of `matmul_q8_fused3` (wq+wk+wv at verify t=2-4). Same contract as
    /// `matmul_q8_fused2_t` with three ranges.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_q8_fused3_t(&self, w0: &crate::model::GpuTensor, w1: &crate::model::GpuTensor,
                              w2: &crate::model::GpuTensor,
                              aq: &CudaSlice<i8>, ad: &CudaSlice<f32>, m: usize)
        -> Result<Option<(CudaSlice<f32>, CudaSlice<f32>, CudaSlice<f32>)>, Box<dyn std::error::Error>> {
        if !(2..=4).contains(&m) || std::env::var("BW24_NO_BATCHED").is_ok() { return Ok(None); }
        let Some([p0, p1, p2]) = self.q8_fused_params(&[w0, w1, w2]) else { return Ok(None) };
        Ok(Some(self.q8_fused3_t_core(p0.0, p1.0, p2.0, aq, ad, m, w0.in_features(),
                                      p0.1, p1.1, p2.1, p0.2)?))
    }

    #[allow(clippy::too_many_arguments)]
    fn q8_fused3_t_core(&self, b0: &CudaSlice<u8>, b1: &CudaSlice<u8>, b2: &CudaSlice<u8>,
                        aq: &CudaSlice<i8>, ad: &CudaSlice<f32>, m: usize,
                        in_f: usize, out0: usize, out1: usize, out2: usize, row_bytes: usize)
        -> Result<(CudaSlice<f32>, CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        const ROWS_PER_BLOCK: u32 = 4;
        let nb0 = (out0 as u32).div_ceil(ROWS_PER_BLOCK);
        let nb1 = (out1 as u32).div_ceil(ROWS_PER_BLOCK);
        let nb2 = (out2 as u32).div_ceil(ROWS_PER_BLOCK);
        let f = self.func(if Self::batched_mcols(m) == 2 { "qmatvec_q8_0_mmvq_fused3_b2" }
                          else { "qmatvec_q8_0_mmvq_fused3_b4" });
        let mut y0 = self.alloc_uninit::<f32>(m * out0)?;
        let mut y1 = self.alloc_uninit::<f32>(m * out1)?;
        let mut y2 = self.alloc_uninit::<f32>(m * out2)?;
        let cfg = LaunchConfig { grid_dim: (nb0 + nb1 + nb2, 1, 1), block_dim: (32, ROWS_PER_BLOCK, 1),
                                 shared_mem_bytes: 0 };
        let (inf, o0, o1, o2, mi, rbl) = (in_f as i32, out0 as i32, out1 as i32, out2 as i32,
                                          m as i32, row_bytes as i64);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(b0).arg(b1).arg(b2).arg(aq).arg(ad).arg(&mut y0).arg(&mut y1).arg(&mut y2)
         .arg(&inf).arg(&o0).arg(&o1).arg(&o2).arg(&mi).arg(&rbl);
        unsafe { b.launch(cfg)?; }
        Ok((y0, y1, y2))
    }

    /// Test entry for the kernel_check gate: fused3 batched from raw weight bytes.
    #[allow(clippy::too_many_arguments)]
    pub fn qmatvec_q8_fused3_t_raw(&self, b0: &CudaSlice<u8>, b1: &CudaSlice<u8>, b2: &CudaSlice<u8>,
                                   x: &CudaSlice<f32>, m: usize, in_f: usize, out0: usize,
                                   out1: usize, out2: usize, row_bytes: usize)
        -> Result<(CudaSlice<f32>, CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let (aq, ad) = self.quantize_q8_1(x, m, in_f)?;
        self.q8_fused3_t_core(b0, b1, b2, &aq, &ad, m, in_f, out0, out1, out2, row_bytes)
    }

    /// Eligibility + param extraction for the fused q8_0 launches: every tensor must be Quant Q8_0
    /// with macro-scale 1.0 (always true for GGUF q8_0; only NVFP4 carries scale) and share w[0]'s
    /// in_f (q8_0 row_bytes is a pure function of in_f, so equal in_f => equal row_bytes). BW24_MMVQ
    /// must be on: the fused body is the MMVQ kernel; without it decode m=1 runs dp4a and fusing
    /// would mix dispatch families (FP-order law). BW24_Q8_DUAL=0 = rollback seam.
    #[allow(clippy::type_complexity)]
    fn q8_fused_params<'w, const N: usize>(&self, ws: &[&'w crate::model::GpuTensor; N])
        -> Option<[(&'w CudaSlice<u8>, usize, usize); N]> {
        use crate::model::GpuTensor;
        if std::env::var("BW24_MMVQ").as_deref() == Ok("0") { return None; }
        if std::env::var("BW24_Q8_DUAL").is_ok_and(|v| v == "0") { return None; }
        let in_f = ws[0].in_features();
        let mut out: [Option<(&CudaSlice<u8>, usize, usize)>; N] = [None; N];
        for (i, w) in ws.iter().enumerate() {
            match w {
                GpuTensor::Quant { bytes, qtype, row_bytes, scale, .. }
                    if *qtype == QT_Q8_0 && *scale == 1.0 && w.in_features() == in_f =>
                        out[i] = Some((bytes, w.out_features(), *row_bytes)),
                _ => return None,
            }
        }
        Some(out.map(|o| o.unwrap()))
    }

    pub fn matmul_pre_noscale(&self, w: &crate::model::GpuTensor, aq: &CudaSlice<i8>, ad: &CudaSlice<f32>,
                              m: usize) -> Result<Option<(CudaSlice<f32>, f32)>, Box<dyn std::error::Error>> {
        use crate::model::GpuTensor;
        // Only the m==1 fast path applies the scale as a separable post-op; bail everywhere else.
        if m != 1 || !self.uses_q8_1_fast(w) { return Ok(None); }
        let in_f = w.in_features();
        let out_f = w.out_features();
        let (bytes, qtype, row_bytes, scale, rp) = match w {
            GpuTensor::Quant { bytes, qtype, row_bytes, scale, rp, .. } => (bytes, *qtype, *row_bytes, *scale, *rp),
            _ => return Ok(None),
        };
        // MMVQ warp-per-row (scale==1.0 passed -> kernel skips its internal scale; we return scale).
        if self.mmvq_supports(qtype) {
            let y = self.qmatvec_mmvq(bytes, aq, ad, m, in_f, out_f, qtype, row_bytes, /*scale*/ 1.0, rp)?;
            return Ok(Some((y, scale)));
        }
        // dp4a fallback: same launch as matmul_pre but WITHOUT the post scale_inplace.
        let name = match qtype {
            QT_Q8_0 => "qmatvec_q8_0_dp4a", QT_Q4_K => "qmatvec_q4_K_dp4a",
            QT_Q6_K => "qmatvec_q6_K_dp4a", QT_Q5_K => "qmatvec_q5_K_dp4a",
            QT_Q3_K => "qmatvec_q3_K_dp4a",
            QT_NVFP4 => if rp { "qmatvec_nvfp4_dp4a_rp" } else { "qmatvec_nvfp4_dp4a" },
            QT_IQ4_XS => "qmatvec_iq4_XS_dp4a",
            _ => return Ok(None),
        };
        let f = self.func(name);
        let mut y = self.alloc_uninit::<f32>(m * out_f)?;
        let cfg = LaunchConfig { grid_dim: (out_f as u32, m as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
        let (inf, outf, mi, rb) = (in_f as i32, out_f as i32, m as i32, row_bytes as i64);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(bytes).arg(aq).arg(ad).arg(&mut y).arg(&inf).arg(&outf).arg(&mi).arg(&rb);
        unsafe { b.launch(cfg)?; }
        Ok(Some((y, scale)))
    }

    /// True if `qtype` has a warp-per-row MMVQ decode kernel AND BW24_MMVQ is set. Only the 4
    /// daily-hot dtypes (Q8_0, Q4_K, Q6_K, NVFP4) — others keep the _dp4a matvec (oracle/fallback).
    pub fn mmvq_supports(&self, qtype: i32) -> bool {
        // DEFAULT ON since 2026-07-08 (BW24_MMVQ=0 reverts to the _dp4a matvec class).
        // QT_F8_E4M3 is exempt from the BW24_MMVQ=0 escape: the e4m3 mmvq family is that dtype's
        // ONLY int8-act kernel class (there is no _dp4a twin), so its m=1/verify/batched dispatch
        // is a pure function of the dtype — the decode-parity law holds under every env.
        if qtype == QT_F8_E4M3 { return true; }
        if std::env::var("BW24_MMVQ").as_deref() == Ok("0") { return false; }
        matches!(qtype, QT_Q8_0 | QT_Q4_K | QT_Q5_K | QT_Q6_K | QT_NVFP4)
    }

    /// PERF-3 warp-per-row MMVQ launcher (decode m=1 hot path). block=(32,ROWS_PER_BLOCK,1):
    /// one warp owns one output row, warp-only __shfl reduction (no smem barrier). Bit-equivalent
    /// to qmatvec_*_dp4a up to f32 reduction order. Pre-quantized q8_1 activation (aq,ad). NVFP4
    /// per-tensor macro-scale applied post (scale==1.0 for other dtypes -> no-op).
    pub fn qmatvec_mmvq(&self, bytes: &CudaSlice<u8>, aq: &CudaSlice<i8>, ad: &CudaSlice<f32>,
                        m: usize, in_f: usize, out_f: usize, qtype: i32, row_bytes: usize, scale: f32,
                        rp: bool)
                        -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        const ROWS_PER_BLOCK: u32 = 4;   // matches BW24_MMVQ_ROWS in qmatvec.cu
        // Multi-row-per-warp (mr2) policy, fixed since the 2026-07 sweeps (the BW24_MMVQ_MR
        // override + mr4 kernel were retired 2026-07-08 — mr4 regressed on register pressure and
        // crashed under rp; q4_K/q6_K mr2 measured flat, "no gain = no change"):
        //   NVFP4 m=1 -> mr2 (clean +1-2% on 9B: RPW acc chains hide the weight-load latency
        //     that pins the single-row kernel at 30-46% DRAM). Bit-identical per row.
        //   Q5_K m=1 -> mr2 (2026-07-05: the FR-Spec trimmed draft head is Q5_K 32768 rows = 8%
        //     of the 27B p3 spec wall; latency-bound like the other k-quants pre-fix).
        //   Q4_K/Q6_K m=1 -> single-row (mr2 measured +0.7% / flat — weight-bandwidth-bound).
        let mut mr: u32 = if m == 1 && (qtype == QT_NVFP4 || qtype == QT_Q5_K) { 2 } else { 1 };
        // q5issue lane (2026-07-08): BW24_Q5K_ISSUE swaps the q5_K m=1 mmvq kernels for the
        // issue-reduced `_il` bodies (uint4 header/qh/qs loads + branchless scale decode —
        // cuts ~34 LDG.U16 + ~5 LDG.U8 + a warp-divergent scale branch per 32-elem group-row
        // to 5 LDG.128). Bit-identical per (token,row) to the reference kernels.
        // `1` = shape-aware policy (N=3 clock-locked micro-bench, mem P0, synthetic real shapes):
        //   out_f <= 65536 (trunk/frspec regime): il at the default mr — mr2_il -9.5%/-10.5%
        //     on 4096x4096/4096x8192, -3.1% on the 32768 frspec head vs the mr2-ref default;
        //   out_f > 65536 (the 248320-row 27B lm_head, already ~97% of the mem wall): mr2_il
        //     REGRESSES +22% there but mr1_il wins -2.1% vs the mr2-ref default -> force mr=1.
        // `2` = force il at the current mr for EVERY shape (A/B probe seam). Default OFF.
        let q5_mode = std::env::var("BW24_Q5K_ISSUE").ok();
        let q5_force = q5_mode.as_deref() == Some("2");
        // DEFAULT ON since 2026-07-08 (BW24_Q5K_ISSUE=0 reverts): +1.8% 9B plain e2e N=3
        // (128.2 -> 130.4), 27B flat (its big head is already at the mem wall), all gates green.
        let q5_il = qtype == QT_Q5_K && m == 1
            && (q5_force || q5_mode.as_deref().map(|v| v != "0").unwrap_or(true));
        if q5_il && !q5_force && out_f > 65536 { mr = 1; }
        let name = match (qtype, mr, rp) {
            (QT_NVFP4, 2, false) => "qmatvec_nvfp4_mmvq_mr2",
            (QT_NVFP4, 2, true)  => "qmatvec_nvfp4_mmvq_mr2_rp",
            (QT_NVFP4, _, true)  => "qmatvec_nvfp4_mmvq_rp",
            (QT_Q5_K, 2, _) => if q5_il { "qmatvec_q5_K_mmvq_mr2_il" } else { "qmatvec_q5_K_mmvq_mr2" },
            (QT_Q8_0, _, _) => "qmatvec_q8_0_mmvq",
            (QT_Q4_K, _, _) => "qmatvec_q4_K_mmvq",
            (QT_Q5_K, _, _) => if q5_il { "qmatvec_q5_K_mmvq_il" } else { "qmatvec_q5_K_mmvq" },
            (QT_Q6_K, _, _) => "qmatvec_q6_K_mmvq",
            (QT_NVFP4, _, false) => "qmatvec_nvfp4_mmvq",
            (QT_F8_E4M3, _, _) => "qmatvec_e4m3_mmvq",
            _ => panic!("qmatvec_mmvq: qtype {qtype} has no MMVQ kernel"),
        };
        let f = self.func(name);
        let mut y = self.alloc_uninit::<f32>(m * out_f)?;  // full-overwrite GEMM output: skip memset
        // each block still has ROWS_PER_BLOCK warps; with mr rows/warp it covers ROWS_PER_BLOCK*mr rows.
        let rows_per_block = ROWS_PER_BLOCK * mr;
        let cfg = LaunchConfig {
            grid_dim: ((out_f as u32 + rows_per_block - 1) / rows_per_block, m as u32, 1),
            block_dim: (32, ROWS_PER_BLOCK, 1),   // warp-per-row (x mr rows each)
            shared_mem_bytes: 0,                  // warp-only reduce at m=1
        };
        let (inf, outf, mi, rb) = (in_f as i32, out_f as i32, m as i32, row_bytes as i64);
        let mut b = self.gpu.stream.launch_builder(&f);
        // NVFP4 + e4m3 mmvq kernels take the macro-scale as a fused epilogue arg (applied at the
        // write — bit-identical to the old separate scale_inplace pass, minus one launch per matvec:
        // 53 scale launches/token on the 9B; for e4m3 the scale is the checkpoint's per-tensor f32
        // weight_scale). Other mmvq kernels keep the 8-arg signature.
        if qtype == QT_NVFP4 || qtype == QT_F8_E4M3 {
            b.arg(bytes).arg(aq).arg(ad).arg(&mut y).arg(&inf).arg(&outf).arg(&mi).arg(&rb).arg(&scale);
            unsafe { b.launch(cfg)?; }
        } else {
            b.arg(bytes).arg(aq).arg(ad).arg(&mut y).arg(&inf).arg(&outf).arg(&mi).arg(&rb);
            unsafe { b.launch(cfg)?; }
            if scale != 1.0 { self.scale_inplace(&mut y, scale, m * out_f)?; }
        }
        Ok(y)
    }

    /// Test entry for the kernel_check bit-equivalence gate: run the warp-per-row MMVQ directly
    /// from raw weight bytes (quantize the f32 activation `x` to q8_1 internally). NVFP4 per-tensor
    /// macro-scale is NOT applied (caller compares bare, like qmatvec_*_fast). Mirrors qmatvec_gemm_raw.
    pub fn qmatvec_mmvq_raw(&self, bytes: &CudaSlice<u8>, x: &CudaSlice<f32>, m: usize, in_f: usize,
                            out_f: usize, qtype: i32, row_bytes: usize, rp: bool)
                            -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let (aq, ad) = self.quantize_q8_1(x, m, in_f)?;
        self.qmatvec_mmvq(bytes, &aq, &ad, m, in_f, out_f, qtype, row_bytes, 1.0, rp)
    }

    /// True if `qtype` has a batched weight-resident (`_b2`/`_b4`) matvec kernel. These mirror the
    /// `_mmvq` kernels but iterate the m token columns INSIDE one warp/row, so the weight bytes leave
    /// HBM/L2 once for m tokens (vs grid.y=m re-reading m times). The 5 daily-hot dtypes have them.
    pub fn batched_supports(&self, qtype: i32) -> bool {
        matches!(qtype, QT_Q8_0 | QT_Q4_K | QT_Q5_K | QT_Q6_K | QT_NVFP4 | QT_F8_E4M3)
    }

    /// b8 tier seam: BW24_B8=0 keeps m=5..8 on the per-m grid.y=m path (m=2..4 batched dispatch
    /// unaffected). Default ON — the K=4..7 spec-verify weight-read-once fix.
    pub fn b8_enabled() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("BW24_B8").map(|v| v != "0").unwrap_or(true))
    }

    /// Compile-time column batch for a runtime m: 2 -> b2, 3..4 -> b4, 5..8 -> b8.
    pub fn batched_mcols(m: usize) -> usize {
        if m == 2 { 2 } else if m <= 4 { 4 } else { 8 }
    }

    /// Kernel name for the batched matvec of `(qtype, mcols)`. mcols ∈ {2,4,8}. The b8 tier is the
    /// K=4..7 spec-verify fix (T=5..8): pre-b8 those T fell to grid.y=m per-row MMVQ = m full
    /// weight reads/launch — the measured 27B K=4 cliff (101 -> 73 tok/s at p3 despite acceptance
    /// holding 54%). One b8 launch reads the weight ONCE for up to 8 columns (c >= m masked).
    fn batched_kernel_name(qtype: i32, mcols: usize) -> Option<&'static str> {
        Some(match (qtype, mcols) {
            (QT_Q8_0, 2) => "qmatvec_q8_0_mmvq_b2", (QT_Q8_0, 4) => "qmatvec_q8_0_mmvq_b4",
            (QT_Q8_0, 8) => "qmatvec_q8_0_mmvq_b8",
            (QT_Q4_K, 2) => "qmatvec_q4_K_mmvq_b2", (QT_Q4_K, 4) => "qmatvec_q4_K_mmvq_b4",
            (QT_Q4_K, 8) => "qmatvec_q4_K_mmvq_b8",
            (QT_Q5_K, 2) => "qmatvec_q5_K_mmvq_b2", (QT_Q5_K, 4) => "qmatvec_q5_K_mmvq_b4",
            (QT_Q5_K, 8) => "qmatvec_q5_K_mmvq_b8",
            (QT_Q6_K, 2) => "qmatvec_q6_K_mmvq_b2", (QT_Q6_K, 4) => "qmatvec_q6_K_mmvq_b4",
            (QT_Q6_K, 8) => "qmatvec_q6_K_mmvq_b8",
            (QT_NVFP4, 2) => "qmatvec_nvfp4_mmvq_b2", (QT_NVFP4, 4) => "qmatvec_nvfp4_mmvq_b4",
            (QT_NVFP4, 8) => "qmatvec_nvfp4_mmvq_b8",
            (QT_F8_E4M3, 2) => "qmatvec_e4m3_mmvq_b2", (QT_F8_E4M3, 4) => "qmatvec_e4m3_mmvq_b4",
            (QT_F8_E4M3, 8) => "qmatvec_e4m3_mmvq_b8",
            _ => return None,
        })
    }

    /// BATCHED weight-tile-resident matvec from a PRE-QUANTIZED q8_1 activation (the m=2-8 verify/MTP
    /// win). One warp walks the weight row ONCE, dp4a vs all m activation columns -> weight HBM/L2
    /// traffic 1x for m tokens (vs grid.y=m re-reading it m times). `mcols` ∈ {2,4,8} is the
    /// compile-time batch; m must be <= mcols (the c >= m columns are masked in-kernel). y is
    /// [m, out_f] token-major. NVFP4 per-tensor macro-scale applied post
    /// (scale==1.0 for other dtypes -> no-op). BIT-IDENTICAL per (token,row) to qmatvec_*_mmvq.
    ///
    /// NVFP4 VARIANT DISPATCH: the batched NVFP4 kernel measured memory-LATENCY bound on the real
    /// 27B verify (ncu --set full, 12 steady launches: long_scoreboard 18-30 stalls/issue vs <=1.7
    /// for every other reason, DRAM only 41-51% active, lg_throttle 0.7, L1 hit 94% — ONE 6-LDG
    /// weight wavefront in flight per warp is the binding constraint, NOT bandwidth and NOT the
    /// column-unroll break). Two exactness-free fixes, chosen PER SHAPE from the DRAM-cold 8-copy
    /// msweep on all six 27B shapes (2026-07-03):
    ///   `pf` = next-g weight-prefetch double-buffer (48 regs, occupancy intact) — wins everywhere
    ///          it applies for b4 (-3..-14%), never loses;
    ///   `r2` = two rows/warp (67 regs -> 7 resident blocks/SM) — the bigger win (-8.5..-30%) but
    ///          wave-quantization-sensitive: with the grid halved to ceil(out_f/8) blocks, a
    ///          fractional straggler wave (waves in ~1.05-1.5) costs a full extra latency round on
    ///          a latency-bound kernel (27B ffn_down 640 blocks / 574 resident = 1.11 waves: +17%),
    ///          while <=1 wave (9B ffn_down 0.89: -30%) or >=2 waves (tail amortized; qkv 2.2:
    ///          -8.5%, ffn_gate 3.8: -12.5%) win. For b2, r2 wins on DEEP k-loops (in_f>=6144:
    ///          -8..-19%) where the 2-col body starves weight MLP hardest; pf measured negative.
    /// b4: r2 when waves(out_f) <= 1 (and grid fills >=half the SMs) or >= 2, else pf.
    /// b2: in_f>=6144 -> r2, else base.
    /// BW24_MMVQ_BV=base|pf|r2|pfr2 forces one variant everywhere (A/B + rollback seam).
    /// All variants BIT-IDENTICAL per (token,row): same dp4a order, scales, adg factor, reduce —
    /// only load issue time and the row->warp mapping change (kernel-check gates all of them).
    /// `rp` = the weight buffer is the A6 SPLIT-PLANE repacked layout (NVFP4 only): the same
    /// wave-aware auto rule applies, mapped onto the `_rp` twins (rp/rpr2/rpr2w8 mirror
    /// pf/r2/r2w8 — regs 44/67/64 land in the same residency classes).
    /// The variant the batched dispatch will pick for this (shape, m, mcols, layout) — exposed so
    /// gates can distinguish bit-identical variants (bit-bad==0 required) from the k-split family
    /// (deterministic but k-reduce-order-shifted: rel<1e-3 + run-to-run bit-identity required).
    pub fn batched_variant(&self, m: usize, in_f: usize, out_f: usize, qtype: i32,
                           row_bytes: usize, mcols: usize, rp: bool) -> &'static str {
        static BV: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();
        let bv = *BV.get_or_init(|| match std::env::var("BW24_MMVQ_BV").as_deref() {
            Ok("base") => "base", Ok("pf") => "pf", Ok("r2") => "r2", Ok("r2w8") => "r2w8",
            Ok("pfr2") => "pfr2", Ok("ca") => "ca", Ok("car2") => "car2",
            // rp* = SPLIT-PLANE REPACKED layout kernels (A6 prototype): W must already be the
            // repacked buffer (msweep MSWEEP_RP harness) — never valid on GGUF-layout weights.
            Ok("rp") => "rp", Ok("rpr2") => "rpr2", Ok("rpr2w8") => "rpr2w8",
            // rpca* = cp.async software-pipelined split-plane (2026-07-05): hides the _rp
            // long_scoreboard load stall. rp-layout only; b4/b2 (no b8 twin).
            Ok("rpca") => "rpca", Ok("rpcar2") => "rpcar2",
            // 2026-07-06 m-small latency arc: rpsc = rpr2 + per-warp smem scale prestage (kills
            // the scale-plane global dependency, zero reg growth); rpms/rpmsc = m-split x2
            // across warp pairs (2x blocks of rpr2, column halves per warp, BIT-identical to
            // _rp); rpks/rpksc = k-split x2 (fastest microbench cells but k-reduce-order-shifted:
            // run-spec self-consistency FAILED on the 27B daily driver — verify logits must be
            // bit-identical to the decode path — measurement corpus ONLY, never auto).
            Ok("rpsc") => "rpsc", Ok("rpms") => "rpms", Ok("rpmsc") => "rpmsc",
            Ok("rpks") => "rpks", Ok("rpksc") => "rpksc",
            _ => "auto",
        });
        // cp.async ring variants need 16B-aligned rows (in_f%256==0 -> (in_f/64)*36 % 16 == 0)
        // and whole 32-group warp iterations (nsb%32==0 <=> in_f%1024==0). All 27B/9B trunk
        // shapes qualify; anything else falls back to the register variants.
        let ca_ok = qtype == QT_NVFP4 && (row_bytes % 16 == 0) && (in_f % 1024 == 0);
        // rpsc: smem scale plane fits (nsb64 <= 272) + int4-aligned staging (nsb64 % 4 == 0).
        // rpks/rpksc: half-plane staging alignment needs nsb64 % 8 == 0 (in_f % 512 == 0).
        // BW24_KS=0 removes the 2026-07-06 rpsc/rpks/rpksc entries from AUTO (rollback seam;
        // forced BW24_MMVQ_BV values still work).
        static KS_ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let ks_on = *KS_ON.get_or_init(|| std::env::var("BW24_KS").as_deref() != Ok("0"));
        let sc_ok = ks_on && qtype == QT_NVFP4 && (in_f % 256 == 0) && (in_f / 64 <= 272);
        let ks_ok = ks_on && qtype == QT_NVFP4 && (in_f % 512 == 0) && (in_f / 64 <= 272);
        static SMS: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
        let sms = *SMS.get_or_init(|| {
            use cudarc::driver::sys::CUdevice_attribute_enum as A;
            self.gpu.ctx.attribute(A::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT).unwrap_or(82)
        });
        // k-quant r2 port (2026-07-04): q4_K/q5_K/q6_K have _r2/_r2w8 twins. ncu on the DRAM-cold
        // 9B msweep showed q4_K/q5_K b4 memory-latency bound like NVFP4 pre-fix (long_scoreboard
        // 19.6/16.4 per issue, DRAM 47.7/38.2%, L2 weight hit ~13%); q6_K lm_head is the exception
        // at DRAM 90-91% = wall-bound (yet r2 still wins -8%: deeper MLP raises achieved DRAM).
        // No _pf port (a k-quant group stages 10+ words vs NVFP4's 5 — register cost outweighs;
        // r2 covers the same MLP) and no rp (GGUF layout only). Q8_0 stays base: its only real
        // batched shapes are the tiny out_f=32 ssm_alpha/beta (8-block grids never fill one SM).
        // AUTO RULE = the measured winners table (differs from NVFP4's!):
        //   r2w8 NEVER in auto — the reg squeeze (72 -> 64 regs = stack spill) loses to unbounded
        //     r2 on every measured k-quant cell, incl. the wave-crossing lm_heads (q6_K 1316 vs
        //     r2 1258us) — kernels kept behind the force seam for the corpus;
        //   q4_K: r2 whenever the halved grid fills the SMs (blocks >= 4*SMs), INCLUDING the
        //     1.05-2.0 straggler window where NVFP4's r2 lost (qkv 1.78 waves: r2 -15% here; the
        //     k-quant base kernel leaves more latency on the table than a straggler wave costs);
        //   q5_K/q6_K: r2 only at waves >= 2 (the 248320-row lm_heads, 48+ waves: q6_K -8%, q5_K
        //     -2%); mid shapes measured base-or-flat (q5_K qkv 49.1 base vs 49.7 r2, attn_gate
        //     flat, attn_k base) — the 5/6-bit two-stream unpack makes r2's staging pricier.
        //   b2 same table with 8-row blocks: q4_K r2 when filled (-3..-22% all measured shapes),
        //     q5_K/q6_K r2 at waves >= 2 (27B lm_head -2.9%; 9B q6_K flat, harmless).
        let kq_r2 = matches!(qtype, QT_Q4_K | QT_Q5_K | QT_Q6_K);
        // BW24_KQ_BV=base|r2|r2w8 forces the k-quant variant WITHOUT touching the NVFP4 dispatch
        // (BW24_MMVQ_BV is global — an interleaved k-quant-only e2e A/B needs this narrower seam).
        static KQBV: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();
        let kq_bv = *KQBV.get_or_init(|| match std::env::var("BW24_KQ_BV").as_deref() {
            Ok("base") => "base", Ok("r2") => "r2", Ok("r2w8") => "r2w8",
            _ => "auto",
        });
        let variant: &'static str = if qtype != QT_NVFP4 && !kq_r2 {
            "base"
        } else if kq_r2 {
            // k-quant r2w8 only exists at b4 (b2_r2 already 8-resident; b8 has no w8 twin) ->
            // mcols != 4 forced r2w8 falls to unbounded r2.
            if kq_bv != "auto" {
                if kq_bv == "r2w8" && mcols != 4 { "r2" } else { kq_bv }
            } else if bv != "auto" {
                match bv {
                    "r2" | "pfr2" | "rpr2" | "car2" => "r2",
                    "r2w8" | "rpr2w8" => if mcols != 4 { "r2" } else { "r2w8" },
                    _ => "base",   // base/pf/ca/rp forced -> base (no such k-quant kernels)
                }
            } else {
                let blocks = (out_f + 7) / 8;
                let waves = blocks as f64 / (7 * sms as usize) as f64;
                let filled = blocks >= 4 * sms as usize;
                let use_r2 = if qtype == QT_Q4_K { filled } else { waves >= 2.0 };
                if use_r2 { "r2" } else { "base" }
            }
        } else if bv != "auto" {
            // r2w8 only exists for b4/b8 (the b2_r2 kernel is already 8-blocks-resident at 60 regs).
            // ca/car2 need the alignment gate AND have no b8 twins; pfr2 has no b8 twin either —
            // unsupported (shape, mcols) combos fall back to pf/r2.
            // On rp buffers, forced legacy names map to their rp twins (layout law).
            let v = if bv == "r2w8" && mcols == 2 { "r2" }
                else if bv == "ca" && (!ca_ok || mcols == 8) { "pf" }
                else if bv == "car2" && (!ca_ok || mcols == 8) { "r2" }
                else if bv == "pfr2" && mcols == 8 { "r2" }
                else if (bv == "rpr2w8" || bv == "rpr2") && mcols == 2 { "rpr2" }
                // rpca* has no b8 twin (falls to rpr2w8/rpr2); needs the ca alignment gate.
                else if (bv == "rpca" || bv == "rpcar2") && (!ca_ok || mcols == 8) {
                    if mcols == 8 { "rpr2w8" } else { "rpr2" }
                }
                else if bv == "rpcar2" && mcols == 2 { "rpca" }
                // rpsc/rpmsc/rpks* gate on smem-fit + alignment; fall to rpr2 outside it
                // (rpms has no smem and no alignment need — always valid on rp buffers).
                else if (bv == "rpsc" || bv == "rpmsc") && !sc_ok { "rpr2" }
                else if (bv == "rpks" || bv == "rpksc") && !ks_ok { "rpr2" }
                else { bv };
            if rp {
                match v {
                    "base" | "pf" | "ca" | "rp" => "rp",
                    "r2" | "pfr2" | "car2" | "rpr2" => "rpr2",
                    "r2w8" | "rpr2w8" => if mcols == 2 { "rpr2" } else { "rpr2w8" },
                    other => other,   // rpca/rpcar2/rpsc/rpks/rpksc pass through (already rp-layout)
                }
            } else { v }
        } else if mcols == 8 {
            // b8 AUTO (2026-07-06 m-small latency arc, g7e DRAM-cold rp msweep m=5/6/8 all five
            // 27B shapes): rpsc — the rpr2w8 schedule with the warp's scale rows prestaged to
            // smem, leaving ONE global dependency (the quant stream) in the k-loop at zero reg
            // growth. BIT-identical to rpr2w8 and wins or ties EVERY b8 cell: ffn_gate m5
            // 50.7->46.9 m8 64.1->57.1 (-11%), qkv m8 34.6->33.0, ssm_out m8 29.7->28.8,
            // attn_gate m8 26.9->26.1, ffn_down m5 58.2->56.9. The faster split-grid twins are
            // OUT: rpksc (k-split, ffn_down m5 -21%) broke run-spec self-consistency (k-reduce
            // order shifts verify argmax at tie margins — verify must stay bit-identical to the
            // m=1 decode chain); rpmsc (m-split, bit-identical) measured NEGATIVE everywhere
            // (twin warp's duplicated weight stream: ffn_down m5 85.7 vs 56.9).
            if rp { if sc_ok { "rpsc" } else { "rpr2w8" } } else { "r2w8" }
        } else if mcols >= 4 {
            // r2 runs 7 resident blocks/SM (67 regs); its __launch_bounds__(128,8) twin `r2w8`
            // (64 regs) runs 8. grid = ceil(out_f/8) for both. rp twins land in the same
            // residency classes (rp 44 regs ~ pf-class occupancy, rpr2 67, rpr2w8 64).
            let blocks = (out_f + 7) / 8;
            let r7 = 7 * sms as usize;
            let r8 = 8 * sms as usize;
            let waves = blocks as f64 / r7 as f64;
            let filled = blocks >= 4 * sms as usize;
            // 2026-07-06 m-small latency arc: b4 keeps the wave rule (rpms/rpmsc measured
            // flat-to-negative at m=3/4 on every shape — the m-split twin duplicates the weight
            // stream; rpsc b4 also negative on r2-class picks, ffn_down m4 51.1 vs 46.5).
            if filled && blocks.div_ceil(r8) < blocks.div_ceil(r7) {
                // the extra residency drops the INTEGER wave count -> the straggler wave a
                // latency-bound kernel pays in full disappears (ffn_down 1.11 -> 0.98 waves:
                // 112.5 -> 81.6us, beats pf 90.1; qkv 2.23 -> 1.95: 58.1 -> 51.1).
                if rp { "rpr2w8" } else { "r2w8" }
            } else if waves >= 2.0 || (waves <= 1.0 && filled) {
                // tail amortized (>=2 waves) or single wave: unbounded r2 (no reg-squeeze tax —
                // gate/up 81.1 vs 83.9 bounded, attn_q 61.0 vs 63.4).
                if rp { "rpr2" } else { "r2" }
            } else {
                // fractional straggler-wave window with no crossing, or grid too small to fill
                // the SMs (tiny out_f<=1024 shapes want max row-parallelism): prefetch variant
                // (rp = the r1 split-plane twin — measured the attn_gate winner, 35.4 vs pf 36.4).
                if rp { "rp" } else { "pf" }
            }
        } else if in_f >= 6144 {
            // b2 deep-k (2026-07-06): every new twin measured flat-to-negative here (rpms 44.1
            // vs rpr2 40.8 ffn_down; rpsc 43.6; the winning rpks is banned on k-order) — rpr2
            // stays.
            if rp { "rpr2" } else { "r2" }
        }
        else if rp {
            // b2 shallow-k: qkv (out_f=10240, 0.97 waves at 7-resident) is the one measured cell
            // where the r2-schedule scale-prestage twin beats the r1 rp pick (24.7 vs 28.9us
            // -15%); the wider (ffn_gate 1.65 waves) and smaller (attn_gate 0.58) shapes LOSE
            // (41.8 vs 38.2 / 16.6 vs 14.6) — gate on the single-wave window.
            let waves = ((out_f + 7) / 8) as f64 / (7 * sms as usize) as f64;
            if sc_ok && waves >= 0.9 && waves <= 1.1 { "rpsc" } else { "rp" }
        } else { "base" };
        variant
    }

    pub fn qmatvec_mmvq_batched(&self, bytes: &CudaSlice<u8>, aq: &CudaSlice<i8>, ad: &CudaSlice<f32>,
                                m: usize, in_f: usize, out_f: usize, qtype: i32, row_bytes: usize,
                                mcols: usize, scale: f32, rp: bool)
                                -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        const ROWS_PER_BLOCK: u32 = 4;
        let variant = self.batched_variant(m, in_f, out_f, qtype, row_bytes, mcols, rp);
        let base_name = Self::batched_kernel_name(qtype, mcols)
            .ok_or_else(|| format!("qmatvec_mmvq_batched: no kernel for qtype {qtype} mcols {mcols}"))?;
        let (name, rows_per_block): (std::borrow::Cow<'static, str>, u32) = match variant {
            "base" => (base_name.into(), ROWS_PER_BLOCK),
            "pf" => (format!("{base_name}_pf").into(), ROWS_PER_BLOCK),
            "ca" => (format!("{base_name}_ca").into(), ROWS_PER_BLOCK),
            "rp" => (format!("{base_name}_rp").into(), ROWS_PER_BLOCK),
            "rpca" => (format!("{base_name}_rpca").into(), ROWS_PER_BLOCK), // 1 row/warp cp.async
            // split families: 2 warp-pairs x 2 rows = 4 rows/block (the k-range or column set
            // splits across the pair's two warps; grid.x doubles vs rpr2 at the same regs).
            "rpks" => (format!("{base_name}_rpks").into(), ROWS_PER_BLOCK),
            "rpksc" => (format!("{base_name}_rpksc").into(), ROWS_PER_BLOCK),
            "rpms" => (format!("{base_name}_rpms").into(), ROWS_PER_BLOCK),
            "rpmsc" => (format!("{base_name}_rpmsc").into(), ROWS_PER_BLOCK),
            v => (format!("{base_name}_{v}").into(), ROWS_PER_BLOCK * 2), // r2-class: 2 rows/warp
        };
        debug_assert!(!rp || name.contains("_rp"), "rp weight dispatched to a GGUF-layout kernel");
        let f = self.func(&name);
        let mut y = self.alloc_uninit::<f32>(m * out_f)?;
        let cfg = LaunchConfig {
            grid_dim: ((out_f as u32 + rows_per_block - 1) / rows_per_block, 1, 1),
            block_dim: (32, ROWS_PER_BLOCK, 1), shared_mem_bytes: 0 };
        let (inf, outf, mi, rb) = (in_f as i32, out_f as i32, m as i32, row_bytes as i64);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(bytes).arg(aq).arg(ad).arg(&mut y).arg(&inf).arg(&outf).arg(&mi).arg(&rb);
        unsafe { b.launch(cfg)?; }
        if scale != 1.0 { self.scale_inplace(&mut y, scale, m * out_f)?; }
        Ok(y)
    }

    /// BATCHED weight-tile-resident matvec from raw weight bytes (quantizes the f32 activation `x` to
    /// q8_1 internally; macro-scale NOT applied — caller compares bare, like qmatvec_*_fast). For the
    /// kernel_check bit-equivalence gate. `mcols` ∈ {2,4,8}. Works for Q8_0/Q4_K/Q5_K/Q6_K/NVFP4.
    pub fn qmatvec_batched_raw(&self, bytes: &CudaSlice<u8>, x: &CudaSlice<f32>, m: usize,
                               in_f: usize, out_f: usize, qtype: i32, row_bytes: usize, mcols: usize,
                               rp: bool)
                               -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let (aq, ad) = self.quantize_q8_1(x, m, in_f)?;
        self.qmatvec_mmvq_batched(bytes, &aq, &ad, m, in_f, out_f, qtype, row_bytes, mcols, 1.0, rp)
    }

    /// Back-compat NVFP4-only batched raw launcher (used by older gates). Delegates to the generic one.
    pub fn qmatvec_nvfp4_batched_raw(&self, bytes: &CudaSlice<u8>, x: &CudaSlice<f32>, m: usize,
                                     in_f: usize, out_f: usize, row_bytes: usize, mcols: usize,
                                     rp: bool)
                                     -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        self.qmatvec_batched_raw(bytes, x, m, in_f, out_f, QT_NVFP4, row_bytes, mcols, rp)
    }

    /// Stage-C FP4 gate (BW24_FP4): if `w` is an NVFP4 weight with in_f%64==0, run the native mxf4
    /// block-scale GEMM and apply the per-tensor macro-scale, returning Some(y). Else None (caller
    /// falls through to the int8 GEMM / dp4a). Strict opt-in over the proven int8 path; m>=16 only.
    fn try_fp4_gemm(&self, w: &crate::model::GpuTensor, x: &CudaSlice<f32>, m: usize,
                    in_f: usize, out_f: usize)
                    -> Result<Option<CudaSlice<f32>>, Box<dyn std::error::Error>> {
        use crate::model::GpuTensor;
        if std::env::var("BW24_FP4").is_err() { return Ok(None); }
        // CUTLASS prefill branch (m>=128 + BW24_FP4_CUTLASS + a repacked CutlassWeight present): route
        // to the CUTLASS sm120 NVFP4 GEMM, folding the per-tensor macro-scale into the epilogue alpha
        // (1/scale) — no post-matmul scale_inplace. Decode (m<128) and the m∈[16,128) middle band keep
        // the hand-roll below: CUTLASS's 128-row M-tile wastes work under 128.
        // The hand-roll applies the per-tensor macro-scale as a POST-matmul MULTIPLY (scale_inplace(y,
        // scale)); CUTLASS's epilogue does D = alpha * (A@B^T), so alpha == scale reproduces it exactly
        // (NOT 1/scale — the plan sketch had this inverted; the kernel_check arm gates it). scale==1.0
        // for the common no-macro-scale case.
        #[cfg(bw24_cutlass)]
        if m >= 128 && std::env::var("BW24_FP4_CUTLASS").is_ok() {
            if let GpuTensor::Quant { bytes, qtype, scale, row_bytes, cutlass, .. } = w {
                if *qtype == QT_NVFP4 && in_f % 64 == 0 {
                    if let Some(cw) = cutlass {
                        // Resident fast path: load-time-repacked B + swizzled SFB (no per-call repack).
                        let y = self.cutlass_fp4_gemm(&cw.b_packed, &cw.sfb_swizzled, x, *scale,
                                                      m, out_f, in_f)?;
                        return Ok(Some(y));
                    } else if std::env::var("BW24_FP4_CUTLASS_OTF").is_ok() {
                        // On-the-fly repack (BW24_FP4_CUTLASS_OTF): de-interleave + swizzle the B operand
                        // from raw bytes per prefill call. No resident doubling of the NVFP4 weight VRAM
                        // (the load-time repack ~doubles it) — needed for models that don't fit the
                        // resident path (e.g. the 27B on 24GB). Slower (per-call repack) but argmax-exact.
                        let (b_packed, sfb_sw) = self.build_cutlass_weight(bytes, out_f, in_f, *row_bytes)?;
                        let y = self.cutlass_fp4_gemm(&b_packed, &sfb_sw, x, *scale, m, out_f, in_f)?;
                        return Ok(Some(y));
                    }
                }
            }
        }
        if let GpuTensor::Quant { bytes, qtype, row_bytes, scale, rp, .. } = w {
            // A6: the hand-rolled W4A4 mxf4 GEMM reads 36B GGUF blocks — no rp port (BW24_FP4 is
            // an opt-in accuracy tradeoff); repacked tensors fall through to the int8 GEMM.
            if *qtype == QT_NVFP4 && in_f % 64 == 0 && !*rp {
                let y = self.qmatvec_gemm_nvfp4_fp4(bytes, x, m, in_f, out_f, *row_bytes, *scale)?;
                return Ok(Some(y));
            }
        }
        Ok(None)
    }

    /// True if `w`'s qtype has a batched tensor-core GEMM kernel (the prefill T>1 root fix).
    /// Only the 4 daily-hot dtypes: Q8_0, Q4_K, Q6_K, NVFP4. NVFP4 needs in_f % 64 == 0.
    /// DEFAULT-ON (2026-06-28): measured pp512 9B-NVFP4 = 1413 tok/s WITH this GEMM vs 298 with the
    /// dp4a fallback (4.7x) AND MORE accurate (prefill logit maxdiff 0.159 vs dp4a 0.55, both argmax
    /// MATCH). The int8 tensor-core GEMM is unconditional (its historical BW24_GEMM opt-in gate
    /// shipped with Phase 0 — mma + smem swizzle + cp.async — and was removed). Prefill-only
    /// (m>=GEMM_M_THRESHOLD); m=1 decode keeps dp4a/MMVQ (this returns true but matmul only calls it
    /// at m>=threshold). BW24_NO_GEMM forces the dp4a fallback (the bit-reference).
    pub fn gemm_supports(&self, w: &crate::model::GpuTensor) -> bool {
        use crate::model::GpuTensor;
        if std::env::var("BW24_NO_GEMM").is_ok() { return false; }
        match w {
            GpuTensor::Quant { qtype, .. } =>
                matches!(*qtype, QT_Q8_0 | QT_Q4_K | QT_Q6_K | QT_Q5_K)
                || (*qtype == QT_NVFP4 && w.in_features() % 64 == 0),
            GpuTensor::Float { .. } | GpuTensor::FloatBf16 { .. } => false,
        }
    }

    /// Batched tensor-core int8 GEMM with a PRE-QUANTIZED q8_1 activation (aq,ad). The prefill
    /// (T>1) root fix: decode each weight 32-block to int8 in shared memory ONCE per (row-tile,
    /// K-step) and reuse it across all BN tokens via mma.sync.m16n8k32.s8 — amortizing the weight
    /// read/decode N-fold (vs the dp4a matvec's per-token re-read). s32 accumulate is exact vs
    /// dp4a; only the final f32 block-scale rounding differs. Caller MUST have checked
    /// `gemm_supports(w)`. y[m,out] token-major. NVFP4 per-tensor macro-scale applied post.
    pub fn qmatvec_gemm(&self, w: &crate::model::GpuTensor, aq: &CudaSlice<i8>, ad: &CudaSlice<f32>,
                        m: usize) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        use crate::model::GpuTensor;
        let in_f = w.in_features();
        let out_f = w.out_features();
        let (bytes, qtype, row_bytes, scale, rp) = match w {
            GpuTensor::Quant { bytes, qtype, row_bytes, scale, rp, .. } => (bytes, *qtype, *row_bytes, *scale, *rp),
            _ => unreachable!("gemm_supports guaranteed Quant"),
        };
        let name = match qtype {
            QT_Q8_0 => "qmatvec_gemm_q8_0", QT_Q4_K => "qmatvec_gemm_q4_K",
            QT_Q5_K => "qmatvec_gemm_q5_K",
            QT_Q6_K => "qmatvec_gemm_q6_K",
            QT_NVFP4 => if rp { "qmatvec_gemm_nvfp4_rp" } else { "qmatvec_gemm_nvfp4" },
            _ => unreachable!(),
        };
        let f = self.func(name);
        let mut y = self.alloc_uninit::<f32>(m * out_f)?;  // full-overwrite GEMM output: skip memset
        // CTA tile MUST match the .cu per-kernel tile. MMQ-PORT: kernel1 (Q8_0/Q4_K/Q5_K) runs llama's
        // 128x128 SQUARE tile (K1_BM=128 x K1_BN=128, 8 warps); kernel2 (Q6_K/NVFP4) keeps 64x256, 4 warps
        // (the macro BM/BN in the .cu). Grid dims are selected by qtype so each launches its own tile.
        let is_k1 = matches!(qtype, QT_Q8_0 | QT_Q4_K | QT_Q5_K);
        // TUNE SEAM: BW24_GEMM_K1_LAUNCH overrides kernel1's launch tile to match a -D-swept fatbin.
        let k1_tile = if is_k1 { k1_launch_override().unwrap_or((128, 128, 8)) } else { (128, 128, 8) };
        let (bm, bn): (u32, u32) = if is_k1 { (k1_tile.0, k1_tile.1) } else { (64, 256) };
        let warps: u32 = if is_k1 { k1_tile.2 } else {
            match qtype { QT_NVFP4 => 8, _ => 4 }
        };
        let cfg = LaunchConfig {
            grid_dim: ((out_f as u32 + bm - 1) / bm, (m as u32 + bn - 1) / bn, 1),
            block_dim: (32, warps, 1),
            shared_mem_bytes: 0,
        };
        let (inf, outf, mi, rb) = (in_f as i32, out_f as i32, m as i32, row_bytes as i64);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(bytes).arg(aq).arg(ad).arg(&mut y).arg(&inf).arg(&outf).arg(&mi).arg(&rb);
        unsafe { b.launch(cfg)?; }
        if scale != 1.0 { self.scale_inplace(&mut y, scale, m * out_f)?; }
        Ok(y)
    }

    /// Test entry: run the GEMM directly from raw weight bytes + qtype (no GpuTensor). Quantizes
    /// the f32 activation `x` to q8_1 internally then launches the tensor-core GEMM. NVFP4 per-tensor
    /// macro-scale is NOT applied here (caller passes it separately, like the dp4a path). Used by
    /// kernel_check for the bit-equivalence gate vs qmatvec_*_dp4a.
    pub fn qmatvec_gemm_raw(&self, bytes: &CudaSlice<u8>, x: &CudaSlice<f32>, m: usize, in_f: usize,
                            out_f: usize, qtype: i32, row_bytes: usize)
                            -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let (aq, ad) = self.quantize_q8_1(x, m, in_f)?;
        let name = match qtype {
            QT_Q8_0 => "qmatvec_gemm_q8_0", QT_Q4_K => "qmatvec_gemm_q4_K",
            QT_Q5_K => "qmatvec_gemm_q5_K",
            QT_Q6_K => "qmatvec_gemm_q6_K", QT_NVFP4 => "qmatvec_gemm_nvfp4",
            QT_NVFP4_RP => "qmatvec_gemm_nvfp4_rp",
            _ => panic!("qmatvec_gemm_raw: qtype {qtype} has no GEMM kernel"),
        };
        let f = self.func(name);
        let mut y = self.alloc_uninit::<f32>(m * out_f)?;  // full-overwrite GEMM output: skip memset
        // MMQ-PORT: kernel1 (Q8_0/Q4_K/Q5_K) = llama 128x128 tile, 8 warps; kernel2 (Q6_K/NVFP4) = 64x256,
        // 4/8 warps. Grid tile per qtype (must match the .cu K1_BM/K1_BN vs BM/BN). KEEP IN SYNC w/ qmatvec_gemm.
        let is_k1 = matches!(qtype, QT_Q8_0 | QT_Q4_K | QT_Q5_K);
        // TUNE SEAM: BW24_GEMM_K1_LAUNCH overrides kernel1's launch tile to match a -D-swept fatbin.
        let k1_tile = if is_k1 { k1_launch_override().unwrap_or((128, 128, 8)) } else { (128, 128, 8) };
        let (bm, bn): (u32, u32) = if is_k1 { (k1_tile.0, k1_tile.1) } else { (64, 256) };
        let warps: u32 = if is_k1 { k1_tile.2 } else {
            match qtype { QT_NVFP4 | QT_NVFP4_RP => 8, _ => 4 }
        };
        let cfg = LaunchConfig {
            grid_dim: ((out_f as u32 + bm - 1) / bm, (m as u32 + bn - 1) / bn, 1),
            block_dim: (32, warps, 1), shared_mem_bytes: 0,
        };
        let (inf, outf, mi, rb) = (in_f as i32, out_f as i32, m as i32, row_bytes as i64);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(bytes).arg(&aq).arg(&ad).arg(&mut y).arg(&inf).arg(&outf).arg(&mi).arg(&rb);
        unsafe { b.launch(cfg)?; }
        Ok(y)
    }

    /// y[i] *= s. NVFP4 per-tensor macro-scale broadcast over the whole output.
    pub fn scale_inplace(&self, y: &mut CudaSlice<f32>, s: f32, n: usize)
                         -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("scale_f32");
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let (sf, ni) = (s, n as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(y).arg(&sf).arg(&ni);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// BW24_FULL_PREC dequant-on-use: expand a bf16-resident weight (`GpuTensor::FloatBf16`, raw
    /// bf16 bytes) to a transient f32 scratch of `n` elements, which then feeds the existing f32
    /// cuBLASLt GEMV. The scratch is freed when the caller drops it, so peak VRAM = resident bf16
    /// weights + ONE (largest) weight's f32 expansion + activations. SLOW IS FINE (research mode).
    pub fn bf16_to_f32(&self, data: &cudarc::driver::CudaView<'_, u8>, n: usize)
                       -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let mut out = self.alloc_uninit::<f32>(n)?;
        let f = self.func("bf16_to_f32");
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let ni = n as i32;
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(data).arg(&mut out).arg(&ni);
        unsafe { b.launch(cfg)?; }
        Ok(out)
    }

    /// Chunked bf16 linear (BW24_FULL_PREC): y[m,out] = x @ W_bf16^T with the f32 dequant scratch
    /// bounded to CHUNK_ROWS rows (256MB at in_f=4096) instead of the whole weight — the 4GB
    /// lm_head expansion OOM'd the 24GB budget. Row-chunking partitions OUTPUT rows; each row's
    /// dot is computed by the identical kernel on identical bytes, so per-(token,row) results are
    /// bit-identical to the unchunked form. `exact` selects linear_decode_exact (per-column m=1
    /// calls, the spec-verify contract) vs plain linear.
    fn linear_bf16_chunked(&self, x: &CudaSlice<f32>, data: &CudaSlice<u8>, m: usize,
                           in_f: usize, out_f: usize, exact: bool)
                           -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        const CHUNK_BYTES: usize = 256 << 20;
        let chunk_rows = (CHUNK_BYTES / (in_f * 4)).max(1).min(out_f);
        if chunk_rows >= out_f {
            let wf32 = self.bf16_to_f32(&data.slice(0..in_f * out_f * 2), in_f * out_f)?;
            return if exact { self.linear_decode_exact(x, &wf32, m, in_f, out_f) }
                   else { self.linear(x, &wf32, m, in_f, out_f) };
        }
        let mut y = self.alloc_uninit::<f32>(m * out_f)?;
        let mut r0 = 0usize;
        while r0 < out_f {
            let rows = chunk_rows.min(out_f - r0);
            let wslice = data.slice(r0 * in_f * 2..(r0 + rows) * in_f * 2);
            let wf32 = self.bf16_to_f32(&wslice, in_f * rows)?;
            let yc = if exact { self.linear_decode_exact(x, &wf32, m, in_f, rows)? }
                     else { self.linear(x, &wf32, m, in_f, rows)? };
            // scatter [m, rows] into y[m, out_f] at column offset r0 (m is tiny in decode/verify)
            for mi in 0..m {
                let src = yc.slice(mi * rows..(mi + 1) * rows);
                let mut dst = y.slice_mut(mi * out_f + r0..mi * out_f + r0 + rows);
                self.gpu.stream.memcpy_dtod(&src, &mut dst)?;
            }
            r0 += rows;
        }
        Ok(y)
    }

    /// On-device linear: y[m,out] = x[m,in] @ W[out,in]^T, weights row-major [out,in] (ggml).
    /// cuBLASLt col-major mapping (see bw24_runtime::Gpu::linear_f32 for the derivation).
    /// DECODE-EXACT float linear: per-column m=1 cuBLASLt calls. cuBLASLt's reduction split is
    /// n-dependent (lt_ndep probe: m=1 vs m=2 col0 differs every bit), so spec-verify batches
    /// must not batch float matmuls the T=1 decode chain runs at m=1. Used by the small-t MoE
    /// router/shexp sites and matmul_decode_exact's Float arm.
    pub fn linear_decode_exact(&self, x: &CudaSlice<f32>, w: &CudaSlice<f32>, m_tokens: usize,
                               in_f: usize, out_f: usize)
                               -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        if m_tokens == 1 { return self.linear(x, w, 1, in_f, out_f); }
        let xv = self.view(x, m_tokens * in_f);
        let mut y = self.alloc_uninit::<f32>(m_tokens * out_f)?;
        for t in 0..m_tokens {
            let row = xv.slice(t * in_f..(t + 1) * in_f);
            let mut xr = self.alloc_uninit::<f32>(in_f)?;
            self.copy_view_into(&mut xr, 0, &row, in_f)?;
            let yr = self.linear(&xr, w, 1, in_f, out_f)?;
            self.copy_into(&mut y, t * out_f, &yr, out_f)?;
        }
        Ok(y)
    }

    pub fn linear(&self, x: &CudaSlice<f32>, w: &CudaSlice<f32>, m_tokens: usize, in_f: usize, out_f: usize)
                  -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        use cudarc::cublaslt::{Matmul, MatmulConfig};
        let mut c = self.alloc_uninit::<f32>(m_tokens * out_f)?;  // cuBLASLt beta=0: C fully written
        let cfg = MatmulConfig {
            transa: true, transb: false, transc: false,
            m: out_f as u64, n: m_tokens as u64, k: in_f as u64,
            alpha: 1.0, lda: in_f as i64, ldb: in_f as i64, beta: 0.0, ldc: out_f as i64,
            stride_a: None, stride_b: None, stride_c: None, stride_bias: None, batch_size: None,
        };
        unsafe { self.gpu.blas.matmul(cfg, w, x, &mut c, None, None)?; }
        Ok(c)
    }

    /// Naive SDPA. Q:[head_dim,n_head,T], K/V:[head_dim,n_head_kv,T_kv] -> O:[head_dim,n_head,T].
    pub fn sdpa_naive(&self, q: &CudaSlice<f32>, k: &CudaSlice<f32>, v: &CudaSlice<f32>,
                      o: &mut CudaSlice<f32>, head_dim: usize, n_head: usize, n_head_kv: usize,
                      t: usize, t_kv: usize, scale: f32, causal: bool)
                      -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("sdpa_naive_f32");
        let cfg = LaunchConfig {
            grid_dim: (n_head as u32, t as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: (t_kv * 4) as u32,
        };
        let (hd, nh, nhkv, ti, tkvi, cz) = (head_dim as i32, n_head as i32, n_head_kv as i32, t as i32, t_kv as i32, causal as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(q).arg(k).arg(v).arg(o).arg(&hd).arg(&nh).arg(&nhkv).arg(&ti).arg(&tkvi).arg(&scale).arg(&cz);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// SDPA where K/V are CudaViews into a resident KV cache (decode hot path, no host round-trip).
    pub fn sdpa_naive_view(&self, q: &CudaSlice<f32>, k: &cudarc::driver::CudaView<f32>,
                           v: &cudarc::driver::CudaView<f32>, o: &mut CudaSlice<f32>,
                           head_dim: usize, n_head: usize, n_head_kv: usize, t: usize, t_kv: usize,
                           scale: f32, causal: bool) -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("sdpa_naive_f32");
        let cfg = LaunchConfig {
            grid_dim: (n_head as u32, t as u32, 1), block_dim: (128, 1, 1),
            shared_mem_bytes: (t_kv * 4) as u32,
        };
        let (hd, nh, nhkv, ti, tkvi, cz) = (head_dim as i32, n_head as i32, n_head_kv as i32, t as i32, t_kv as i32, causal as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(q).arg(k).arg(v).arg(o).arg(&hd).arg(&nh).arg(&nhkv).arg(&ti).arg(&tkvi).arg(&scale).arg(&cz);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// Hand-written FlashAttention prefill (sm_120, FA-2 online softmax on validated mma.sync,
    /// head_dim 256 or 128 (template-stamped twins), GQA, causal). Replaces sdpa_naive for T>1.
    /// Q/K/V/O [head_dim, n_head(_kv), T].
    pub fn fa_prefill(&self, q: &CudaSlice<f32>, k: &CudaSlice<f32>, v: &CudaSlice<f32>,
                      o: &mut CudaSlice<f32>, head_dim: usize, n_head: usize, n_head_kv: usize,
                      t: usize, t_kv: usize, scale: f32, causal: bool)
                      -> Result<(), Box<dyn std::error::Error>> {
        // FLOOR PORT (P2+P0a+P0b+P1): 4 warps/CTA, BLOCK_Q=64 query rows, BK=32 KV tile,
        // Q-in-reg + register-O, grid.y=n_head_kv (4 Q-heads share staged K/V).
        // Edge 5a (DEFAULT): fa_prefill_f32_pp — register-resident softmax (no sSw smem
        // round-trip), the FA3 softmax-GEMM overlap variant. ncu (pp512): short_scoreboard
        // 4.32->3.47, wait 1.99->1.45, per-call ~577us->~440us (1.31x) at flat 12.1% warps /
        // 255 regs / 2 CTAs (occupancy preserved). Bit-safe: 9B+27B argmax MATCH, rel 2.55e-3
        // vs floor 3.03e-3. BW24_FA_FLOOR reverts to the serialized-softmax floor kernel.
        const BLOCK_Q: usize = 64; const BK: usize = 32;
        // hd128 twins (2026-07-07): the prefill kernels are template-stamped at 256 (original
        // names, dispatch unchanged) and 128 (`_hd128`, the MiniMax-M3 class). Callers gate
        // other head_dims to sdpa_naive before reaching here.
        let hd_sfx = fa_hd_suffix(head_dim)?;
        let floor = std::env::var("BW24_FA_FLOOR").is_ok();
        let f = self.func(&format!("fa_prefill_f32{}{hd_sfx}", if floor { "" } else { "_pp" }));
        // persistent smem: bf16*(sK + sV + sP) + f32*(sS + sM + sL)
        //   = bf16*(2*BK*hd + BLOCK_Q*BK) + f32*(BLOCK_Q*BK + 2*BLOCK_Q)
        let shmem = (2 * (2 * BK * head_dim + BLOCK_Q * BK)
                   + 4 * (BLOCK_Q * BK + 2 * BLOCK_Q)) as u32;
        use cudarc::driver::sys::CUfunction_attribute_enum as A;
        f.set_attribute(A::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmem as i32)?;
        let cfg = LaunchConfig {
            grid_dim: ((t as u32 + BLOCK_Q as u32 - 1) / BLOCK_Q as u32, n_head as u32, 1),
            block_dim: (32, 4, 1), shared_mem_bytes: shmem,
        };
        let (hd, nh, nhkv, ti, tkvi, cz) = (head_dim as i32, n_head as i32, n_head_kv as i32, t as i32, t_kv as i32, causal as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(q).arg(k).arg(v).arg(o).arg(&hd).arg(&nh).arg(&nhkv).arg(&ti).arg(&tkvi).arg(&scale).arg(&cz);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// FA prefill where K/V are QUANTIZED CudaViews into the resident byte KV cache (the T=K verify
    /// path, MTP-PLAN §D.3). Uses `fa_prefill_q` (inline-dequant during stage-to-smem). The view's
    /// base+offset pointer is honored; the kernel reads [0..t_kv*tok_bytes). Q is the T fresh query
    /// rows; t = T, t_kv = cache len. k_tok_bytes/v_tok_bytes are the per-token byte strides.
    pub fn fa_prefill_view(&self, q: &CudaSlice<f32>, k: &cudarc::driver::CudaView<u8>,
                           v: &cudarc::driver::CudaView<u8>, o: &mut CudaSlice<f32>,
                           head_dim: usize, n_head: usize, n_head_kv: usize,
                           t: usize, t_kv: usize, scale: f32, causal: bool,
                           k_tok_bytes: usize, v_tok_bytes: usize)
                           -> Result<(), Box<dyn std::error::Error>> {
        const BLOCK_Q: usize = 64; const BK: usize = 32;
        let f = self.func(&format!("fa_prefill_q{}", fa_hd_suffix(head_dim)?));
        let shmem = (2 * (2 * BK * head_dim + BLOCK_Q * BK)
                   + 4 * (BLOCK_Q * BK + 2 * BLOCK_Q)) as u32;
        use cudarc::driver::sys::CUfunction_attribute_enum as A;
        f.set_attribute(A::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmem as i32)?;
        let cfg = LaunchConfig {
            grid_dim: ((t as u32 + BLOCK_Q as u32 - 1) / BLOCK_Q as u32, n_head as u32, 1),
            block_dim: (32, 4, 1), shared_mem_bytes: shmem,
        };
        let (hd, nh, nhkv, ti, tkvi, cz) = (head_dim as i32, n_head as i32, n_head_kv as i32, t as i32, t_kv as i32, causal as i32);
        let (ktb, vtb) = (k_tok_bytes as i64, v_tok_bytes as i64);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(q).arg(k).arg(v).arg(o).arg(&hd).arg(&nh).arg(&nhkv).arg(&ti).arg(&tkvi).arg(&scale).arg(&cz)
         .arg(&ktb).arg(&vtb);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// ARC B (2026-07-05): dequant-once chunk-prime FA. Same contract as `fa_prefill_view`, but
    /// instead of every (q-block, head) CTA re-dequanting the whole quantized KV stream inline
    /// (T/64 x n_head redundant at chunk prime — 30.5% of the 32k prime wall), dequant the full
    /// [t_kv, kv_dim] K and V ONCE into a resident bf16 workspace (fa_dequant_kv_ws_bf16), then
    /// run `fa_prefill_qw` (the bf16-workspace twin) over it. EXACT: the workspace holds the same
    /// __float2bfloat16(dq_*_elem(...)) values fa_prefill_q stages to smem, and the twin's MMA/
    /// softmax/PV code is byte-identical -> bit-identical O (kernel_check pins bitdiff=0).
    /// The workspace allocation is REUSED across layers/chunks (grown to the largest shape);
    /// contents are rewritten per call. BW24_PRIME_DEQW=0 falls back to fa_prefill_view (callers gate).
    #[allow(clippy::too_many_arguments)]
    pub fn fa_prefill_view_ws(&self, q: &CudaSlice<f32>, k: &cudarc::driver::CudaView<u8>,
                              v: &cudarc::driver::CudaView<u8>, o: &mut CudaSlice<f32>,
                              head_dim: usize, n_head: usize, n_head_kv: usize,
                              t: usize, t_kv: usize, scale: f32, causal: bool,
                              k_tok_bytes: usize, v_tok_bytes: usize)
                              -> Result<(), Box<dyn std::error::Error>> {
        const BLOCK_Q: usize = 64; const BK: usize = 32;
        let kv_dim_k = n_head_kv * head_dim;
        let kv_dim_v = n_head_kv * head_dim;
        let k_ws_bytes = t_kv * kv_dim_k * 2;   // bf16
        let v_ws_bytes = t_kv * kv_dim_v * 2;
        // Lock held across BOTH launches: enqueue-only (µs), all compute serializes on gpu.stream.
        let mut guard = self.prime_deqw_ws.lock().unwrap();
        let need_grow = match guard.as_ref() {
            Some((kw, vw)) => kw.len() < k_ws_bytes || vw.len() < v_ws_bytes,
            None => true,
        };
        if need_grow {
            let grow = |cur: usize, need: usize| if cur >= need { cur } else { need };
            let (ck, cv) = guard.as_ref().map(|(a, b)| (a.len(), b.len())).unwrap_or((0, 0));
            *guard = Some((self.alloc_u8(grow(ck, k_ws_bytes))?, self.alloc_u8(grow(cv, v_ws_bytes))?));
        }
        let (kw, vw) = guard.as_mut().unwrap();
        // pass 1: dequant K+V once into the bf16 workspace (grid-stride, 1 thread/elem)
        {
            let f = self.func("fa_dequant_kv_ws_bf16");
            let total = (t_kv * (kv_dim_k + kv_dim_v)) as u64;
            let nblk = ((total + 255) / 256).min(65535 * 16) as u32;
            let cfg = LaunchConfig { grid_dim: (nblk.max(1), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            let (kdk, kdv, tkvi) = (kv_dim_k as i32, kv_dim_v as i32, t_kv as i32);
            let (ktb, vtb) = (k_tok_bytes as i64, v_tok_bytes as i64);
            let mut b = self.gpu.stream.launch_builder(&f);
            b.arg(k).arg(v).arg(&mut *kw).arg(&mut *vw).arg(&kdk).arg(&kdv).arg(&tkvi).arg(&ktb).arg(&vtb);
            unsafe { b.launch(cfg)?; }
        }
        // pass 2: the bf16-workspace prefill twin (same tile sizes/loop structure as fa_prefill_q).
        // DEFAULT: cp.async double-buffered staging twin (fa_prefill_qw_db, +32KB smem for the
        // second K/V tile pair, 1 CTA/SM): overlaps tile n+1's L2->smem copy with tile n's MMA.
        // Bit-identical output (staging is a pure byte copy; kernel_check pins bitdiff=0 under
        // both twins). A/B (27B g7e, N=3): 32k prime 17.10->16.51s, 16k 9.09->8.65s — the copy
        // latency hides behind the MMA pipe and beats the 2-CTA/SM occupancy of the sync twin.
        // BW24_PRIME_DEQW_DB=0 falls back to the single-buffer twin.
        let db = std::env::var("BW24_PRIME_DEQW_DB").map(|v| v != "0").unwrap_or(true);
        {
            let hd_sfx = fa_hd_suffix(head_dim)?;
            let f = self.func(&format!("fa_prefill_qw{}{hd_sfx}", if db { "_db" } else { "" }));
            let shmem = if db {
                // 4x KV tile buffers (bf16) + sP (bf16) + sL (f32)
                (2 * (4 * BK * head_dim + BLOCK_Q * BK) + 4 * BLOCK_Q) as u32
            } else {
                (2 * (2 * BK * head_dim + BLOCK_Q * BK)
                       + 4 * (BLOCK_Q * BK + 2 * BLOCK_Q)) as u32
            };
            use cudarc::driver::sys::CUfunction_attribute_enum as A;
            f.set_attribute(A::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmem as i32)?;
            let cfg = LaunchConfig {
                grid_dim: ((t as u32 + BLOCK_Q as u32 - 1) / BLOCK_Q as u32, n_head as u32, 1),
                block_dim: (32, 4, 1), shared_mem_bytes: shmem,
            };
            let (hd, nh, nhkv, ti, tkvi, cz) = (head_dim as i32, n_head as i32, n_head_kv as i32, t as i32, t_kv as i32, causal as i32);
            let (kdk, kdv) = (kv_dim_k as i32, kv_dim_v as i32);
            let mut b = self.gpu.stream.launch_builder(&f);
            b.arg(q).arg(&*kw).arg(&*vw).arg(o).arg(&hd).arg(&nh).arg(&nhkv).arg(&ti).arg(&tkvi).arg(&scale).arg(&cz)
             .arg(&kdk).arg(&kdv);
            unsafe { b.launch(cfg)?; }
        }
        Ok(())
    }

    /// FA decode (T=1 split-K) over the resident QUANTIZED KV cache (q8_0 K / q5_1 V) as u8 views.
    /// Replaces sdpa_naive_view for decode; inline-dequants per element. k_tok_bytes/v_tok_bytes are
    /// the per-token byte strides (differ: q8_0=34*nblk, q5_1=24*nblk per token).
    pub fn fa_decode(&self, q: &CudaSlice<f32>, k: &cudarc::driver::CudaView<u8>,
                     v: &cudarc::driver::CudaView<u8>, o: &mut CudaSlice<f32>,
                     head_dim: usize, n_head: usize, n_head_kv: usize, t_kv: usize, scale: f32,
                     k_tok_bytes: usize, v_tok_bytes: usize)
                     -> Result<(), Box<dyn std::error::Error>> {
        // PERF-4: the warp-per-token vec path replaces the scalar element-per-thread fa_decode_f32 —
        // warp-per-token fa_decode_vec_q (grid=(n_head_kv,n_splits), block=(32,gqa_ratio)).
        // The block dequants each KV tile ONCE into smem (bf16) and broadcasts to all gqa Q-head
        // warps -> each KV byte leaves HBM/L2 ~1x/group (vs 4x). ARGS identical; func/grid/block/
        // smem/n_splits differ. fa_decode_f32 stays the bit-reference fallback. Combine is shared.
        //
        // SPLIT-K: the scalar path has grid.x=n_head (32) blocks; the vec path only has
        // grid.x=n_head_kv (8). To avoid starving the GPU at mid ctx, the vec path splits MORE
        // aggressively (64 keys/split vs 256) so grid.y rises and 8*n_splits fills the SMs.
        // At VERY short ctx (t_kv<96) even 1 split can't fill the GPU from 8 KV heads, so the
        // broadcast can't beat the scalar path's 4x-more-blocks latency hiding — fall back to
        // scalar there (measured crossover: vec 0.68x at t_kv=64, 1.23x at t_kv=96, 2.2x at 256).
        // DEFAULT-ON (2026-06-28): clean clock-locked sweep proved vec beats scalar at every
        // t_kv>=96 and the gain WIDENS with ctx (graph decode: +9.5% @128, +11.6% @512, +11.8%
        // @2048) — the KV-byte-broadcast (4x fewer HBM reads/group) compounds as attention grows.
        // BW24_NO_FA_VEC forces the scalar bit-reference. Below FA_VEC_MIN_TKV the scalar path's
        // 4x-more-blocks (grid.x=n_head=32 vs n_head_kv=8) hides latency better, so keep scalar there.
        let fa_vec = std::env::var("BW24_NO_FA_VEC").is_err() && t_kv >= FA_VEC_MIN_TKV;
        let sp = fa_split_keys(t_kv, n_head_kv);
        let n_splits = if fa_vec { ((t_kv + sp - 1) / sp).max(1) } else { ((t_kv + 255) / 256).max(1) };
        let o_len = n_head * n_splits * head_dim;
        let ml_len = n_head * n_splits;
        let (mut part_o, mut part_m, mut part_l) =
            (self.zeros(o_len)?, self.zeros(ml_len)?, self.zeros(ml_len)?);
        let (part_o, part_m, part_l) = (&mut part_o, &mut part_m, &mut part_l);
        let (hd, nh, nhkv, tkvi, nsp) = (head_dim as i32, n_head as i32, n_head_kv as i32, t_kv as i32, n_splits as i32);
        let (ktb, vtb) = (k_tok_bytes as i64, v_tok_bytes as i64);
        // The vec kernel holds head_dim/32 register accumulators (FA_DEC_MAX_DPL=8 -> head_dim<=256).
        // All shipped models use head_dim=256; fall back to scalar for anything wider rather than
        // silently truncating the accumulator.
        let fa_vec = fa_vec && head_dim <= 256 && head_dim % 32 == 0;
        let (f, cfg) = if fa_vec {
            let gqa = (n_head / n_head_kv).max(1) as u32;
            // DEEP-CTX smem twin (2026-07-05): the register-dequant path's GQA reuse rides L2,
            // which holds to ~8k ctx but dies at 40k (layer KV ~37MB) — the 4 GQA warps then
            // re-read every KV byte from DRAM (4x traffic). Above BW24_FA_SMEM_TKV (default
            // 1024 — the 2026-07-05 crossover re-sweep on real prompts: p3 spec 73.8->79.2 at
            // 2048, flat down to 512, p2 +5%, p1/9B unchanged; the ARC-A probe's synthetic
            // 2.1x smem-at-all-depths pointed here; 0=never) dispatch the smem-broadcast twin:
            // dequant each tile ONCE per block.
            // Bit-identical per (token,split): same bf16 round-trip, same accumulation order,
            // same partial layout -> same combine. Short/mid ctx keeps the register path (it won
            // there by 12x — latency, not bandwidth, rules small KV).
            static SMEM_TKV: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
            let smem_tkv = *SMEM_TKV.get_or_init(|| {
                std::env::var("BW24_FA_SMEM_TKV").ok().and_then(|v| v.parse().ok()).unwrap_or(1024)
            });
            if fa_v3_active(head_dim) {
                // FA v3 lane: dp4a-K hybrid (register-quantized Q, raw q8_0 K, staged-V kept).
                // smem = sV only (half of v2's).
                let fv = self.func("fa_decode_vec_q_v3");
                let shmem = (32 * head_dim * 2) as u32;      // sV bf16 [FA_DEC_TILE=32][hd]
                (fv,
                 LaunchConfig { grid_dim: (n_head_kv as u32, n_splits as u32, 1),
                     block_dim: (32, gqa, 1), shared_mem_bytes: shmem })
            } else if fa_v2_on() {
                // FAVENDOR lane: llama fattn-vec tile-batched softmax + wide-load staging on
                // OUR smem KV broadcast. Replaces BOTH per-key twins when on; same grid/block/
                // partials; same 32KB sK+sV tile as the smem twin.
                let fv = self.func("fa_decode_vec_q_v2");
                let shmem = (2 * 32 * head_dim * 2) as u32;   // sK+sV bf16 [FA_DEC_TILE=32][hd]
                (fv,
                 LaunchConfig { grid_dim: (n_head_kv as u32, n_splits as u32, 1),
                     block_dim: (32, gqa, 1), shared_mem_bytes: shmem })
            } else if smem_tkv > 0 && t_kv >= smem_tkv {
                let fv = self.func("fa_decode_vec_q_smem");
                let shmem = (2 * 32 * head_dim * 2) as u32;   // sK+sV bf16 [FA_DEC_TILE=32][hd]
                use cudarc::driver::sys::CUfunction_attribute_enum as A;
                fv.set_attribute(A::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmem as i32)?;
                (fv,
                 LaunchConfig { grid_dim: (n_head_kv as u32, n_splits as u32, 1),
                     block_dim: (32, gqa, 1), shared_mem_bytes: shmem })
            } else {
                // REGISTER-DEQUANT kernel (2026-07-03): per-warp direct q8_0/q5_1 register
                // dequant, zero dynamic shared memory.
                let fv = self.func("fa_decode_vec_q");
                (fv,
                 LaunchConfig { grid_dim: (n_head_kv as u32, n_splits as u32, 1),
                     block_dim: (32, gqa, 1), shared_mem_bytes: 0 })
            }
        } else {
            (self.func("fa_decode_f32"),
             LaunchConfig { grid_dim: (n_head as u32, n_splits as u32, 1),
                 block_dim: (head_dim as u32, 1, 1), shared_mem_bytes: (4 * (head_dim + 32)) as u32 })
        };
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(q).arg(k).arg(v).arg(&mut *part_o).arg(&mut *part_m).arg(&mut *part_l)
         .arg(&hd).arg(&nh).arg(&nhkv).arg(&tkvi).arg(&scale).arg(&nsp).arg(&ktb).arg(&vtb);
        unsafe { b.launch(cfg)?; }
        let (fc, cfg2) = (self.func("fa_decode_combine_f32"),
            LaunchConfig { grid_dim: (n_head as u32, 1, 1), block_dim: (head_dim as u32, 1, 1), shared_mem_bytes: 0 });
        let mut b2 = self.gpu.stream.launch_builder(&fc);
        b2.arg(&*part_o).arg(&*part_m).arg(&*part_l).arg(o).arg(&hd).arg(&nh).arg(&nsp);
        unsafe { b2.launch(cfg2)?; }
        Ok(())
    }

    /// True iff the MULTI-ROW verify FA (`fa_decode_rows`) is usable for a verify batch whose
    /// FIRST row attends `base_len + 1` keys: every row must take the SAME kernel eager decode
    /// would (the vec path) — mirrors fa_decode's gate exactly (BW24_NO_FA_VEC + FA_VEC_MIN_TKV +
    /// head_dim), evaluated at the MINIMUM row bound so no row could have picked scalar.
    /// BW24_FA_ROWS_OFF=1 is the A/B + fallback seam (per-row loop).
    pub fn fa_rows_eligible(&self, base_len: usize, head_dim: usize) -> bool {
        std::env::var("BW24_NO_FA_VEC").is_err()
            && std::env::var("BW24_FA_ROWS_OFF").is_err()
            && base_len + 1 >= FA_VEC_MIN_TKV
            && head_dim <= 256 && head_dim % 32 == 0
    }

    /// MULTI-ROW verify FA: run fa_decode_vec_q's EXACT per-row program for T causal query rows
    /// (row r attends keys [0..base_len+r+1)) in ONE kernel launch with grid.z = row, plus ONE
    /// row-batched combine. Replaces the T separate (fa_decode + combine) launches of the spec
    /// verify — same per-row split partition (n_splits_r = ceil(t_kv_r/split_keys), the
    /// fa_split_keys formula), same key-walk order, same reduce shapes => bit-identical outputs
    /// per row (kernel-check pins rows-vs-loop byte identity; run-spec is the end gate).
    /// Caller must have checked `fa_rows_eligible(base_len, head_dim)`.
    /// q is the verify's token-major [T, n_head, head_dim] stack; o is written [T, n_head, head_dim].
    #[allow(clippy::too_many_arguments)]
    pub fn fa_decode_rows(&self, q: &CudaSlice<f32>, k: &cudarc::driver::CudaView<u8>,
                          v: &cudarc::driver::CudaView<u8>, o: &mut CudaSlice<f32>,
                          head_dim: usize, n_head: usize, n_head_kv: usize,
                          base_len: usize, t: usize, scale: f32,
                          k_tok_bytes: usize, v_tok_bytes: usize)
                          -> Result<(), Box<dyn std::error::Error>> {
        debug_assert!(base_len + 1 >= FA_VEC_MIN_TKV && head_dim <= 256 && head_dim % 32 == 0);
        let t_kv_max = base_len + t;                       // LAST row's key bound
        let sp = fa_split_keys(t_kv_max, n_head_kv);       // env/default — same value every row
        let n_splits_max = (t_kv_max + sp - 1) / sp;
        let (hd, nh, nhkv) = (head_dim as i32, n_head as i32, n_head_kv as i32);
        let (base_i, nspm, spk) = (base_len as i32, n_splits_max as i32, sp as i32);
        let (ktb, vtb) = (k_tok_bytes as i64, v_tok_bytes as i64);
        let gqa = (n_head / n_head_kv).max(1) as u32;
        let o_len = t * n_head * n_splits_max * head_dim;
        let ml_len = t * n_head * n_splits_max;
        let (mut part_o, mut part_m, mut part_l) =
            (self.zeros(o_len)?, self.zeros(ml_len)?, self.zeros(ml_len)?);
        let (part_o, part_m, part_l) = (&mut part_o, &mut part_m, &mut part_l);
        // Deep-ctx smem twin for the VERIFY rows (2026-07-05): same threshold + rationale as
        // fa_decode's dispatch — at 40k the register path's GQA L2-reuse premise is dead and the
        // verify multiplies the 4x DRAM re-read by T rows. Bit-identical per (row,token,split).
        static SMEM_TKV_R: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        let smem_tkv = *SMEM_TKV_R.get_or_init(|| {
            std::env::var("BW24_FA_SMEM_TKV").ok().and_then(|v| v.parse().ok()).unwrap_or(1024)
        });
        let v3 = fa_v3_active(head_dim);
        let smem_rows = !v3 && !fa_v2_on() && smem_tkv > 0 && t_kv_max >= smem_tkv;
        let fname = if v3 { "fa_decode_vec_q_rows_v3" }
                    else if fa_v2_on() { "fa_decode_vec_q_rows_v2" }
                    else if smem_rows { "fa_decode_vec_q_rows_smem" }
                    else { "fa_decode_vec_q_rows" };
        let f = self.func(fname);
        let shmem = if v3 || smem_rows || fa_v2_on() {
            // v3 stages sV only (16KB @hd256); v2/smem twins stage sK+sV (32KB).
            let sh = (if v3 { 32 * head_dim * 2 } else { 2 * 32 * head_dim * 2 }) as u32;
            use cudarc::driver::sys::CUfunction_attribute_enum as A;
            f.set_attribute(A::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, sh as i32)?;
            sh
        } else { 0 };
        let cfg = LaunchConfig { grid_dim: (n_head_kv as u32, n_splits_max as u32, t as u32),
            block_dim: (32, gqa, 1), shared_mem_bytes: shmem };
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(q).arg(k).arg(v).arg(&mut *part_o).arg(&mut *part_m).arg(&mut *part_l)
         .arg(&hd).arg(&nh).arg(&nhkv).arg(&base_i).arg(&scale).arg(&nspm).arg(&spk)
         .arg(&ktb).arg(&vtb);
        unsafe { b.launch(cfg)?; }
        let (fc, cfg2) = (self.func("fa_decode_combine_rows"),
            LaunchConfig { grid_dim: (n_head as u32, t as u32, 1),
                block_dim: (head_dim as u32, 1, 1), shared_mem_bytes: 0 });
        let mut b2 = self.gpu.stream.launch_builder(&fc);
        b2.arg(&*part_o).arg(&*part_m).arg(&*part_l).arg(o).arg(&hd).arg(&nh)
          .arg(&base_i).arg(&nspm).arg(&spk);
        unsafe { b2.launch(cfg2)?; }
        Ok(())
    }

    /// Device-counter variant of `fa_decode` (CUDA-GRAPH-PLAN Phase 2). The sequence length is read
    /// from `t_kv_dev[0]` (resident device i32[1]) for the attention loop bound + per-split key range;
    /// the GRID `n_splits` is sized for `bucket_max` (the bucket's max t_kv — baked at capture time).
    /// Empty splits (key range beyond the actual t_kv) write an empty partial (m=NEG_INF) so the
    /// shared combine skips them -> bit-correct for ANY actual t_kv <= bucket_max.
    ///
    /// BIT-IDENTITY (the gate): pass `bucket_max == actual_t_kv` and this reproduces `fa_decode`
    /// EXACTLY (same n_splits, same per, same split boundaries, same combine) while reading t_kv from
    /// device. Bucketing (bucket_max > t_kv) is for the future captured path and changes split
    /// grouping (different but mathematically-equal log-sum-exp merge).
    pub fn fa_decode_dc(&self, q: &CudaSlice<f32>, k: &cudarc::driver::CudaView<u8>,
                        v: &cudarc::driver::CudaView<u8>, o: &mut CudaSlice<f32>,
                        head_dim: usize, n_head: usize, n_head_kv: usize,
                        t_kv_dev: &CudaSlice<i32>, bucket_max: usize, scale: f32,
                        k_tok_bytes: usize, v_tok_bytes: usize)
                        -> Result<(), Box<dyn std::error::Error>> {
        // The fa_vec gate + n_splits are sized from bucket_max (host, fixed at capture). The kernel
        // reads the ACTUAL t_kv from t_kv_dev for the per-split bound. DEFAULT-ON to MATCH the eager
        // `fa_decode` gate above — graph capture must mirror eager's kernel choice or the graph-vs-eager
        // bit-identity gate breaks. BW24_NO_FA_VEC forces scalar on BOTH paths in lockstep.
        let fa_vec = std::env::var("BW24_NO_FA_VEC").is_err() && bucket_max >= FA_VEC_MIN_TKV;
        let sp = fa_split_keys(bucket_max, n_head_kv);
        let n_splits = if fa_vec { ((bucket_max + sp - 1) / sp).max(1) } else { ((bucket_max + 255) / 256).max(1) };
        let mut part_o = self.zeros(n_head * n_splits * head_dim)?;
        let mut part_m = self.zeros(n_head * n_splits)?;
        let mut part_l = self.zeros(n_head * n_splits)?;
        let (hd, nh, nhkv, nsp) = (head_dim as i32, n_head as i32, n_head_kv as i32, n_splits as i32);
        let (ktb, vtb) = (k_tok_bytes as i64, v_tok_bytes as i64);
        let fa_vec = fa_vec && head_dim <= 256 && head_dim % 32 == 0;
        let (f, cfg) = if fa_vec && fa_v3_active(head_dim) {
            // FA v3 lane _dc twin: the captured graph must run the SAME walk body as eager
            // under BW24_FA_V3=1 (eager, rows-verify and graph switch together).
            let gqa = (n_head / n_head_kv).max(1) as u32;
            let fv = self.func("fa_decode_vec_q_v3_dc");
            let shmem = (32 * head_dim * 2) as u32;       // sV bf16 [FA_DEC_TILE=32][hd]
            (fv,
             LaunchConfig { grid_dim: (n_head_kv as u32, n_splits as u32, 1),
                 block_dim: (32, gqa, 1), shared_mem_bytes: shmem })
        } else if fa_vec && fa_v2_on() {
            // FAVENDOR lane: v2 _dc twin — the captured graph must run the SAME walk body as
            // eager under BW24_FA_V2=1 or graph_decode_gate's bit-identity breaks (the flag is
            // a numeric config; eager, rows-verify and graph all switch together).
            let gqa = (n_head / n_head_kv).max(1) as u32;
            let fv = self.func("fa_decode_vec_q_v2_dc");
            let shmem = (2 * 32 * head_dim * 2) as u32;   // sK+sV bf16 [FA_DEC_TILE=32][hd]
            (fv,
             LaunchConfig { grid_dim: (n_head_kv as u32, n_splits as u32, 1),
                 block_dim: (32, gqa, 1), shared_mem_bytes: shmem })
        } else if fa_vec {
            let gqa = (n_head / n_head_kv).max(1) as u32;
            // REGISTER-DEQUANT twin: zero dynamic smem (see fa_decode above).
            let fv = self.func("fa_decode_vec_q_dc");
            (fv,
             LaunchConfig { grid_dim: (n_head_kv as u32, n_splits as u32, 1),
                 block_dim: (32, gqa, 1), shared_mem_bytes: 0 })
        } else {
            (self.func("fa_decode_f32_dc"),
             LaunchConfig { grid_dim: (n_head as u32, n_splits as u32, 1),
                 block_dim: (head_dim as u32, 1, 1), shared_mem_bytes: (4 * (head_dim + 32)) as u32 })
        };
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(q).arg(k).arg(v).arg(&mut part_o).arg(&mut part_m).arg(&mut part_l)
         .arg(&hd).arg(&nh).arg(&nhkv).arg(t_kv_dev).arg(&scale).arg(&nsp).arg(&ktb).arg(&vtb);
        unsafe { b.launch(cfg)?; }
        let fc = self.func("fa_decode_combine_f32");
        let cfg2 = LaunchConfig { grid_dim: (n_head as u32, 1, 1), block_dim: (head_dim as u32, 1, 1), shared_mem_bytes: 0 };
        let mut b2 = self.gpu.stream.launch_builder(&fc);
        b2.arg(&part_o).arg(&part_m).arg(&part_l).arg(o).arg(&hd).arg(&nh).arg(&nsp);
        unsafe { b2.launch(cfg2)?; }
        Ok(())
    }

    /// EAGER fa_decode geometry for a given actual `t_kv` (CUDA-GRAPH-PLAN §3.3 bucketing). Returns
    /// `(fa_vec, n_splits)` EXACTLY as `fa_decode` computes them so the graph-capture path can key its
    /// bucket on the same `(kernel, n_splits)` pair and pass a `bucket_max` that reproduces eager's
    /// n_splits bit-for-bit. (Per = ceil(t_kv/n_splits) is then recomputed from the DEVICE t_kv inside
    /// the kernel and matches eager when n_splits matches — the bit-identity contract.)
    pub fn fa_geom_eager(&self, t_kv: usize, head_dim: usize, n_head_kv: usize) -> (bool, usize) {
        // MUST mirror `fa_decode` / `fa_decode_dc` (default-ON 2026-06-28). This is the bucket-key
        // source: if it disagrees with the actual kernel pick, the graph captures the wrong path and
        // replay diverges from eager. All three sites read BW24_NO_FA_VEC in lockstep.
        let mut fa_vec = std::env::var("BW24_NO_FA_VEC").is_err() && t_kv >= FA_VEC_MIN_TKV;
        fa_vec = fa_vec && head_dim <= 256 && head_dim % 32 == 0;
        let sp = fa_split_keys(t_kv, n_head_kv);
        let n_splits = if fa_vec { ((t_kv + sp - 1) / sp).max(1) } else { ((t_kv + 255) / 256).max(1) };
        (fa_vec, n_splits)
    }

    /// `bucket_max` (host t_kv to feed `fa_decode_dc` / `full_attn_decode_dc`) that makes the _dc
    /// kernel pick the SAME (fa_vec, n_splits) as eager would for actual `t_kv`. Because the dc
    /// launcher derives both from `bucket_max` via the same formulas, we just hand it `t_kv` itself:
    /// the n_splits is then identical, and the per-split boundaries (computed from the DEVICE t_kv in
    /// the kernel) match eager exactly. The bucket KEY (for the graph HashMap) is `(fa_vec, n_splits)`.
    pub fn fa_bucket_key(&self, t_kv: usize, head_dim: usize, n_head_kv: usize) -> (bool, usize) {
        self.fa_geom_eager(t_kv, head_dim, n_head_kv)
    }

    /// CUDA-graph capture wrapper (CUDA-GRAPH-PLAN §3.2, llama.cpp warmup pattern). Runs `step`
    /// inline TWICE (warmup — lets the caching allocator settle to stable pointers and any one-time
    /// kernel attribute/JIT happen outside capture), then captures a THIRD invocation on the Engine's
    /// decode stream (RELAXED mode) and instantiates it into a replayable `CudaGraph`. The closure
    /// must enqueue ONLY device work on `e.stream()` (no dtoh / no synchronize / no host branch on
    /// device data) — every per-step varying scalar must come from a device counter. Returns the
    /// instantiated graph; `CudaGraph::launch()` replays the whole step in one dispatch.
    pub fn capture_graph<F>(&self, mut step: F) -> Result<cudarc::driver::CudaGraph, Box<dyn std::error::Error>>
        where F: FnMut(&Engine) -> Result<(), Box<dyn std::error::Error>>
    {
        use cudarc::driver::sys::{CUstreamCaptureMode, CUgraphInstantiate_flags};
        // EVENT TRACKING OFF for capture. The Engine creates a 2nd stream (copy_stream) so cudarc is in
        // multi-stream mode and, by default, records a CudaEvent per CudaSlice alloc/use to serialize
        // cross-stream access. Those per-buffer event waits issue stream ops that are NOT permitted
        // inside a capture region (CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED). The captured decode step is
        // strictly SINGLE-STREAM (every kernel on gpu.stream), so this synchronization is unnecessary
        // here — disable it for the whole warmup+capture, re-enable after. SAFETY: the decode-dc path
        // touches only gpu.stream; no buffer crosses to copy_stream during capture.
        let was_tracking = self.gpu.ctx.is_event_tracking();
        if was_tracking { unsafe { self.gpu.ctx.disable_event_tracking(); } }
        let mut run = || -> Result<cudarc::driver::CudaGraph, Box<dyn std::error::Error>> {
            // warmup: two inline runs (no capture) so allocator pointers + kernel attrs are stable.
            step(self)?;
            step(self)?;
            self.gpu.stream.synchronize()?;
            // capture the third run.
            self.gpu.stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)?;
            // If the body errors mid-capture, end the capture before propagating so the stream isn't
            // left in a capturing state.
            let r = step(self);
            let g = self.gpu.stream.end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH);
            r?;
            let graph = g?.ok_or("capture produced no graph (stream was not capturing)")?;
            graph.upload()?;
            Ok(graph)
        };
        let result = run();
        if was_tracking { unsafe { self.gpu.ctx.enable_event_tracking(); } }
        result
    }

    /// gdn_scan variant where state_in/out are CudaViews (resident SSM state, in-place per step).
    pub fn gdn_scan_s128_view(&self, q: &CudaSlice<f32>, k: &CudaSlice<f32>, v: &CudaSlice<f32>,
                              g: &CudaSlice<f32>, beta: &CudaSlice<f32>,
                              state_in: &cudarc::driver::CudaView<f32>,
                              state_out: &mut cudarc::driver::CudaViewMut<f32>,
                              o: &mut CudaSlice<f32>, n_head: usize, t: usize, scale: f32)
                              -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("gdn_scan_s128");
        const S_V: u32 = 128; const WARP: u32 = 32; const COLS: u32 = 4;
        let cfg = LaunchConfig { grid_dim: (n_head as u32, 1, S_V / COLS), block_dim: (WARP, COLS, 1), shared_mem_bytes: 0 };
        let (h, ti) = (n_head as i32, t as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(q).arg(k).arg(v).arg(g).arg(beta).arg(state_in).arg(state_out).arg(o).arg(&h).arg(&ti).arg(&scale);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// conv1d where the input is a CudaView (resident conv state assembled in place).
    pub fn ssm_conv1d_view(&self, x: &cudarc::driver::CudaView<f32>, w: &CudaSlice<f32>, y: &mut CudaSlice<f32>,
                           conv_dim: usize, t: usize, d_conv: usize, silu: bool)
                           -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("ssm_conv1d_silu_f32");
        // grid.x = channel, grid.y = T-tiles (block 256 strides over T) — parallel over both axes.
        let cfg = LaunchConfig { grid_dim: (conv_dim as u32, ((t as u32 + 255) / 256).max(1), 1),
                                 block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let (cd, ti, dc, s) = (conv_dim as i32, t as i32, d_conv as i32, silu as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(x).arg(w).arg(y).arg(&cd).arg(&ti).arg(&dc).arg(&s);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// Depthwise causal conv1d + optional SiLU.
    /// x:[conv_dim, T+d_conv-1] channel-major (first d_conv-1 cols = carried state),
    /// w:[d_conv, conv_dim] kernel-major, y:[conv_dim, T] channel-major.
    /// FUSED prefill conv (token-major input, zero left-state): replaces
    /// transpose + zeros + conv_left_pad + ssm_conv1d with ONE launch reading the matmul output
    /// directly. Output channel-major [conv_dim, T], SiLU applied. BIT-IDENTICAL accumulation.
    pub fn ssm_conv1d_tm(&self, qkv_tm: &CudaSlice<f32>, w: &CudaSlice<f32>, y: &mut CudaSlice<f32>,
                         conv_dim: usize, t: usize, d_conv: usize)
                         -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("ssm_conv1d_tm_f32");
        let cfg = LaunchConfig {
            grid_dim: (((conv_dim + 255) / 256) as u32, t as u32, 1),
            block_dim: (256, 1, 1), shared_mem_bytes: 0,
        };
        let (cd, ti, dc) = (conv_dim as i32, t as i32, d_conv as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(qkv_tm).arg(w).arg(y).arg(&cd).arg(&ti).arg(&dc);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// BATCHED verify conv (T>1, carried state): window reads the resident conv ring for
    /// negative rows; separate ring-update launch afterwards. BIT-IDENTICAL per value to the
    /// T=1 chain. T >= pad rides the pure input-column ring update (unchanged legacy path);
    /// T < pad (the BW24_SPEC_M2 t=2 verify arm) needs old-ring sources for the roll — the
    /// update kernel would race reading the ring it rewrites, so that arm clones the ring
    /// (dtod) and rolls via ssm_conv_ring_rebuild (PURE COPIES: the ring stores raw input
    /// columns; the final ring == what T sequential decode ring rolls leave).
    pub fn ssm_conv1d_tm_state(&self, qkv_tm: &CudaSlice<f32>, conv_state: &mut CudaSlice<f32>,
                               w: &CudaSlice<f32>, y: &mut CudaSlice<f32>,
                               conv_dim: usize, t: usize, d_conv: usize)
                               -> Result<(), Box<dyn std::error::Error>> {
        assert!(t >= 1, "ssm_conv1d_tm_state requires T >= 1");
        // clone BEFORE the window kernel is issued is not required (stream-ordered: the dtod and
        // the window kernel both read the pre-roll ring; the roll launches after both) — but
        // cloning first keeps the ordering trivially correct under any future stream split.
        let ring_old = if t < d_conv - 1 { Some(self.clone_dtod(conv_state)?) } else { None };
        {
            let f = self.func("ssm_conv1d_tm_state_f32");
            let cfg = LaunchConfig {
                grid_dim: (((conv_dim + 255) / 256) as u32, t as u32, 1),
                block_dim: (256, 1, 1), shared_mem_bytes: 0,
            };
            let (cd, ti, dc) = (conv_dim as i32, t as i32, d_conv as i32);
            let mut b = self.gpu.stream.launch_builder(&f);
            b.arg(qkv_tm).arg(&*conv_state).arg(w).arg(y).arg(&cd).arg(&ti).arg(&dc);
            unsafe { b.launch(cfg)?; }
        }
        match ring_old {
            None => {
                let f = self.func("ssm_conv_ring_update_f32");
                let n = conv_dim * (d_conv - 1);
                let cfg = LaunchConfig::for_num_elems(n as u32);
                let (cd, ti, dc) = (conv_dim as i32, t as i32, d_conv as i32);
                let mut b = self.gpu.stream.launch_builder(&f);
                b.arg(qkv_tm).arg(conv_state).arg(&cd).arg(&ti).arg(&dc);
                unsafe { b.launch(cfg)?; }
            }
            Some(old) => self.ssm_conv_ring_rebuild(qkv_tm, &old, conv_state, conv_dim, t, d_conv)?,
        }
        Ok(())
    }

    /// PREFIX conv-ring rebuild (spec REPLAY-FREE partial accept): overwrite the resident ring
    /// with the state a T=1 chain holds after only the FIRST `tc` columns of `qkv_tm` — the last
    /// `pad` entries of [ring_old | cols 0..tc-1]. PURE COPIES (the ring stores raw inputs; no
    /// arithmetic, cannot perturb FP order). `ring_old` = the pre-round snapshot ring.
    pub fn ssm_conv_ring_rebuild(&self, qkv_tm: &CudaSlice<f32>, ring_old: &CudaSlice<f32>,
                                 conv_state: &mut CudaSlice<f32>,
                                 conv_dim: usize, tc: usize, d_conv: usize)
                                 -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("ssm_conv_ring_rebuild_f32");
        let n = conv_dim * (d_conv - 1);
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let (cd, ti, dc) = (conv_dim as i32, tc as i32, d_conv as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(qkv_tm).arg(ring_old).arg(conv_state).arg(&cd).arg(&ti).arg(&dc);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// FUSED decode GDN prep (T=1): repack + q/k L2-norm + beta sigmoid + g_log in one launch.
    /// Replaces 5 tiny serialized kernels on the decode critical path. L2 reduce runs as a 32-lane
    /// warp tree (vs l2_norm_f32's 256-thread two-level tree) — same math, different FP sum order;
    /// the argmax + run-spec gates are the authority.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_prep_decode(&self, conv_out: &CudaSlice<f32>, beta_raw: &CudaSlice<f32>,
                           alpha: &CudaSlice<f32>, dt_bias: &CudaSlice<f32>, a: &CudaSlice<f32>,
                           q_l2: &mut CudaSlice<f32>, k_l2: &mut CudaSlice<f32>, v_g: &mut CudaSlice<f32>,
                           beta: &mut CudaSlice<f32>, g_log: &mut CudaSlice<f32>,
                           d_state: usize, num_v: usize, num_k: usize, key_dim: usize, eps: f32)
                           -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("gdn_prep_decode_f32");
        let cfg = LaunchConfig { grid_dim: (num_v as u32, 1, 1), block_dim: (32, 4, 1), shared_mem_bytes: 0 };
        let (ds, nv, nk, kd) = (d_state as i32, num_v as i32, num_k as i32, key_dim as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(conv_out).arg(beta_raw).arg(alpha).arg(dt_bias).arg(a)
         .arg(q_l2).arg(k_l2).arg(v_g).arg(beta).arg(g_log)
         .arg(&ds).arg(&nv).arg(&nk).arg(&kd).arg(&eps);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// FUSED prefill conv + GDN repack: token-major qkv -> q_g/k_g/v_g in ONE launch (no conv_out
    /// materialization, no qkv_to_gdn_repack pass). BIT-IDENTICAL values; scatter matches
    /// qkv_to_gdn_repack's modulo head-repeat mapping exactly.
    #[allow(clippy::too_many_arguments)]
    pub fn ssm_conv1d_gdn(&self, qkv_tm: &CudaSlice<f32>, w: &CudaSlice<f32>,
                          q_g: &mut CudaSlice<f32>, k_g: &mut CudaSlice<f32>, v_g: &mut CudaSlice<f32>,
                          conv_dim: usize, t: usize, d_conv: usize,
                          d_state: usize, num_v: usize, num_k: usize, key_dim: usize)
                          -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("ssm_conv1d_gdn_f32");
        let cfg = LaunchConfig {
            grid_dim: (((conv_dim + 255) / 256) as u32, t as u32, 1),
            block_dim: (256, 1, 1), shared_mem_bytes: 0,
        };
        let (cd, ti, dc) = (conv_dim as i32, t as i32, d_conv as i32);
        let (ds, nv, nk, kd) = (d_state as i32, num_v as i32, num_k as i32, key_dim as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(qkv_tm).arg(w).arg(q_g).arg(k_g).arg(v_g)
         .arg(&cd).arg(&ti).arg(&dc).arg(&ds).arg(&nv).arg(&nk).arg(&kd);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    pub fn ssm_conv1d(&self, x: &CudaSlice<f32>, w: &CudaSlice<f32>, y: &mut CudaSlice<f32>,
                      conv_dim: usize, t: usize, d_conv: usize, silu: bool)
                      -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("ssm_conv1d_silu_f32");
        let cfg = LaunchConfig { grid_dim: (conv_dim as u32, ((t as u32 + 255) / 256).max(1), 1),
                                 block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let (cd, ti, dc, s) = (conv_dim as i32, t as i32, d_conv as i32, silu as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(x).arg(w).arg(y).arg(&cd).arg(&ti).arg(&dc).arg(&s);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// Gated DeltaNet scan, S_v=128. q,k,v:[128,H,T]; g,beta:[H,T]; state:[128,128,H] transposed;
    /// o:[128,H,T]. Single sequence.
    pub fn gdn_scan_s128(&self, q: &CudaSlice<f32>, k: &CudaSlice<f32>, v: &CudaSlice<f32>,
                         g: &CudaSlice<f32>, beta: &CudaSlice<f32>, state_in: &CudaSlice<f32>,
                         state_out: &mut CudaSlice<f32>, o: &mut CudaSlice<f32>,
                         n_head: usize, t: usize, scale: f32)
                         -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("gdn_scan_s128");
        const S_V: u32 = 128; const WARP: u32 = 32; const COLS_PER_BLOCK: u32 = 4;
        let cfg = LaunchConfig {
            grid_dim: (n_head as u32, 1, S_V / COLS_PER_BLOCK),
            block_dim: (WARP, COLS_PER_BLOCK, 1),
            shared_mem_bytes: 0,
        };
        let (h, ti) = (n_head as i32, t as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(q).arg(k).arg(v).arg(g).arg(beta).arg(state_in).arg(state_out).arg(o).arg(&h).arg(&ti).arg(&scale);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// A4 seam: chunked WY GDN prefill. DEFAULT ON (`BW24_GDN_CHUNKED=0` = rollback to the
    /// sequential scan). Flipped 2026-07-04 with the full battery green: kernel-check ALL
    /// GREEN x {9B, 27B} incl the f64-truth chunk gates; run-gen argmax 82==82 both models
    /// on AND off (24/24 sweep runs); run-spec K={1,2,3,4,6,8} PASS x {9B synth, 9B text,
    /// 27B p2, 27B p3}; e2e first-16-token agreement 6/6 (full-256 drifts at index 47-125
    /// on 5/6 prompts — accepted cache-state-FP class, batched-prime precedent).
    /// PREFILL-ONLY: decode + spec verify never route here (decode==verify dispatch
    /// identity law); prime_cache/forward/forward_last are the only callers.
    pub fn gdn_chunked_enabled() -> bool {
        static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *E.get_or_init(|| std::env::var("BW24_GDN_CHUNKED").map(|v| v != "0").unwrap_or(true))
    }

    /// A4 chunk size (BW24_GDN_CHUNK, default 32 — the sweep winner: the O(T*C) chunk
    /// matrices grow with C while the sequential state pass is C-flat, so smaller chunks
    /// win; C=32/64 also get the register-history solve template). Clamped to multiples
    /// of 32 in [32, 128] (kernel row mappings require it).
    pub fn gdn_chunk_size() -> usize {
        static C: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *C.get_or_init(|| {
            let c: usize = std::env::var("BW24_GDN_CHUNK").ok()
                .and_then(|v| v.parse().ok()).unwrap_or(32);
            c.clamp(32, 128) / 32 * 32
        })
    }

    /// A4: chunked WY / blockwise-inverse GDN prefill (see cu/hybrid.cu K1-K5 header for the
    /// math). Same contract as `gdn_scan_s128` (layouts, state ping-pong) but chunk-parallel:
    /// NOT bit-identical to the sequential scan (chunked FP accumulation order); run-gen
    /// argmax + run-spec batteries are the accuracy authority. PREFILL callers only.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_scan_chunked(&self, q: &CudaSlice<f32>, k: &CudaSlice<f32>, v: &CudaSlice<f32>,
                            g: &CudaSlice<f32>, beta: &CudaSlice<f32>, state_in: &CudaSlice<f32>,
                            state_out: &mut CudaSlice<f32>, o: &mut CudaSlice<f32>,
                            n_head: usize, t: usize, scale: f32, c: usize)
                            -> Result<(), Box<dyn std::error::Error>> {
        const D: usize = 128;
        const NSPLIT: u32 = 4;
        assert!(c >= 1 && c <= 128, "gdn_scan_chunked: C must be in 1..=128");
        let h = n_head;
        let nc = (t + c - 1) / c;
        let (hi, ti, ci) = (h as i32, t as i32, c as i32);
        // scratch (stream-ordered allocs; pool-reused across layers)
        let mut gcum = self.uninit(t * h)?;
        let mut a = self.uninit(nc * h * c * c)?;
        let mut p = self.uninit(nc * h * c * c)?;
        let mut u = self.uninit(nc * h * c * D)?;
        let mut w = self.uninit(nc * h * c * D)?;
        let mut y = self.uninit(nc * h * c * D)?;
        let mut ssnap = self.uninit(nc * h * D * D)?;   // chunk-start state snapshots (K5 phase 1)
        {   // K1
            let f = self.func("gdn_chunk_cumgate_f32");
            let cfg = LaunchConfig { grid_dim: (nc as u32, h as u32, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
            let mut b = self.gpu.stream.launch_builder(&f);
            b.arg(g).arg(&mut gcum).arg(&hi).arg(&ti).arg(&ci);
            unsafe { b.launch(cfg)?; }
        }
        if c <= 64 {   // K2 register-tiled (2x2 outputs/thread, whole-chunk smem k tile)
            let f = self.func("gdn_chunk_attn_f32");
            let jt = ((c + 31) / 32) as u32;
            let cfg = LaunchConfig { grid_dim: (nc as u32, h as u32, jt), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            let mut b = self.gpu.stream.launch_builder(&f);
            b.arg(q).arg(k).arg(&gcum).arg(beta).arg(&mut a).arg(&mut p).arg(&hi).arg(&ti).arg(&ci);
            unsafe { b.launch(cfg)?; }
        } else {       // K2 generic (C = 128)
            let f = self.func("gdn_chunk_attn_g_f32");
            let cfg = LaunchConfig { grid_dim: (nc as u32, h as u32, 1), block_dim: (32, 8, 1), shared_mem_bytes: 0 };
            let mut b = self.gpu.stream.launch_builder(&f);
            b.arg(q).arg(k).arg(&gcum).arg(beta).arg(&mut a).arg(&mut p).arg(&hi).arg(&ti).arg(&ci);
            unsafe { b.launch(cfg)?; }
        }
        {   // K3 (register-history templates for C=32/64; local-memory generic otherwise)
            let cfg = LaunchConfig { grid_dim: (nc as u32, h as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            match c {
                32 | 64 => {
                    let f = self.func(if c == 32 { "gdn_chunk_solve32_f32" } else { "gdn_chunk_solve64_f32" });
                    let mut b = self.gpu.stream.launch_builder(&f);
                    b.arg(v).arg(k).arg(&a).arg(&gcum).arg(&mut u).arg(&mut w).arg(&hi).arg(&ti);
                    unsafe { b.launch(cfg)?; }
                }
                _ => {
                    let f = self.func("gdn_chunk_solve_f32");
                    let mut b = self.gpu.stream.launch_builder(&f);
                    b.arg(v).arg(k).arg(&a).arg(&gcum).arg(&mut u).arg(&mut w).arg(&hi).arg(&ti).arg(&ci);
                    unsafe { b.launch(cfg)?; }
                }
            }
        }
        {   // K4 (sequential over chunks inside; blocks col-partition the state)
            let f = self.func("gdn_chunk_state_f32");
            let cfg = LaunchConfig { grid_dim: (h as u32, NSPLIT, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            let mut b = self.gpu.stream.launch_builder(&f);
            b.arg(k).arg(&gcum).arg(beta).arg(&u).arg(&w).arg(&mut y).arg(&mut ssnap)
             .arg(state_in).arg(&mut *state_out).arg(&hi).arg(&ti).arg(&ci);
            unsafe { b.launch(cfg)?; }
        }
        {   // K5 (j-blocked: grid.z = 32-row output blocks per chunk; writes o fully)
            let f = self.func("gdn_chunk_output_f32");
            let jt = ((c + 31) / 32) as u32;
            let cfg = LaunchConfig { grid_dim: (nc as u32, h as u32, jt), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            let mut b = self.gpu.stream.launch_builder(&f);
            b.arg(q).arg(&gcum).arg(&p).arg(&y).arg(&ssnap).arg(o).arg(&hi).arg(&ti).arg(&ci).arg(&scale);
            unsafe { b.launch(cfg)?; }
        }
        Ok(())
    }

    /// PREFILL GDN scan dispatch (the A4 seam): chunked WY form when enabled and T is in the
    /// batched-prefill regime, else the sequential scan. Callers: hybrid_forward::linear_attn
    /// (forward/forward_last) + linear_attn_prime (prime_cache). Decode (T=1) and the spec
    /// verify call `gdn_scan_s128` DIRECTLY — the decode==verify dispatch identity is untouched.
    ///
    /// BW24_GDN_DIFF=1: numerical-oracle mode — runs BOTH forms on the same inputs, prints the
    /// per-call (== per-layer, in call order) output/state error distribution, and keeps the
    /// SEQUENTIAL results so the run stays on the shipped path (stage-1 prototype evidence).
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_scan_prefill(&self, q: &CudaSlice<f32>, k: &CudaSlice<f32>, v: &CudaSlice<f32>,
                            g: &CudaSlice<f32>, beta: &CudaSlice<f32>, state_in: &CudaSlice<f32>,
                            state_out: &mut CudaSlice<f32>, o: &mut CudaSlice<f32>,
                            n_head: usize, t: usize, scale: f32)
                            -> Result<(), Box<dyn std::error::Error>> {
        if std::env::var("BW24_GDN_DIFF").is_ok() && t >= 16 {
            return self.gdn_scan_diff(q, k, v, g, beta, state_in, state_out, o, n_head, t, scale);
        }
        if Self::gdn_chunked_enabled() && t >= 16 {
            self.gdn_scan_chunked(q, k, v, g, beta, state_in, state_out, o, n_head, t, scale,
                                  Self::gdn_chunk_size())
        } else {
            self.gdn_scan_s128(q, k, v, g, beta, state_in, state_out, o, n_head, t, scale)
        }
    }

    /// Stage-1 oracle: run sequential AND chunked, report per-call error stats, keep sequential.
    #[allow(clippy::too_many_arguments)]
    fn gdn_scan_diff(&self, q: &CudaSlice<f32>, k: &CudaSlice<f32>, v: &CudaSlice<f32>,
                     g: &CudaSlice<f32>, beta: &CudaSlice<f32>, state_in: &CudaSlice<f32>,
                     state_out: &mut CudaSlice<f32>, o: &mut CudaSlice<f32>,
                     n_head: usize, t: usize, scale: f32)
                     -> Result<(), Box<dyn std::error::Error>> {
        static CALL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let call = CALL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut o_c = self.uninit(o.len())?;
        let mut st_c = self.uninit(state_out.len())?;
        self.gdn_scan_chunked(q, k, v, g, beta, state_in, &mut st_c, &mut o_c,
                              n_head, t, scale, Self::gdn_chunk_size())?;
        self.gdn_scan_s128(q, k, v, g, beta, state_in, state_out, o, n_head, t, scale)?;
        let (oh_s, oh_c) = (self.dtoh(o)?, self.dtoh(&o_c)?);
        let (sh_s, sh_c) = (self.dtoh(state_out)?, self.dtoh(&st_c)?);
        let stats = |a: &[f32], b: &[f32]| -> (f32, f32, f64) {
            let mut max_abs = 0f32; let mut max_rel = 0f32; let mut sum_rel = 0f64;
            for (x, y) in a.iter().zip(b) {
                let ad = (x - y).abs();
                let rel = ad / x.abs().max(y.abs()).max(1e-3);
                if ad > max_abs { max_abs = ad; }
                if rel > max_rel { max_rel = rel; }
                sum_rel += rel as f64;
            }
            (max_abs, max_rel, sum_rel / a.len() as f64)
        };
        let (o_ma, o_mr, o_mean) = stats(&oh_s, &oh_c);
        let (s_ma, s_mr, s_mean) = stats(&sh_s, &sh_c);
        println!("[gdn-diff call {call:3} T={t} C={}] out: max_abs={o_ma:.3e} max_rel={o_mr:.3e} mean_rel={o_mean:.3e} | \
                  state: max_abs={s_ma:.3e} max_rel={s_mr:.3e} mean_rel={s_mean:.3e}",
                 Self::gdn_chunk_size());
        Ok(())
    }

    /// softplus-based g_log: g_log[h,t] = a[h] * softplus(alpha[h,t] + dt_bias[h]). a pre-negated.
    pub fn gdn_glog(&self, alpha: &CudaSlice<f32>, dt_bias: &CudaSlice<f32>, a: &CudaSlice<f32>,
                    g_log: &mut CudaSlice<f32>, n_head: usize, t: usize)
                    -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("gdn_glog_f32");
        let cfg = LaunchConfig::for_num_elems((n_head * t) as u32);
        let (h, ti) = (n_head as i32, t as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(alpha).arg(dt_bias).arg(a).arg(g_log).arg(&h).arg(&ti);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    pub fn sigmoid(&self, x: &CudaSlice<f32>, y: &mut CudaSlice<f32>, n: usize)
                   -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("sigmoid_f32");
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let ni = n as i32;
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(x).arg(y).arg(&ni);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// gated RMSNorm: dst = RMSNorm(o, w[ncols]) * silu(z), per row of ncols. nrows blocks.
    pub fn gated_rmsnorm(&self, o: &CudaSlice<f32>, w: &CudaSlice<f32>, z: &CudaSlice<f32>,
                         dst: &mut CudaSlice<f32>, ncols: usize, nrows: usize, eps: f32)
                         -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("gated_rmsnorm_f32");
        let cfg = LaunchConfig { grid_dim: (nrows as u32, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
        let (nc, e) = (ncols as i32, eps);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(o).arg(w).arg(z).arg(dst).arg(&nc).arg(&e);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// add+RMSNorm emitting the f32 normed row AND its q8_1 quantization in one launch (the MoE
    /// layer input: z feeds the router matmul as f32, the expert dp4a as q8_1). BIT-IDENTICAL to
    /// add_rms_norm + quantize_q8_1. Returns (q, d) alongside the caller-provided res/z buffers.
    #[allow(clippy::too_many_arguments)]
    pub fn add_rms_norm_zq8(&self, a: &CudaSlice<f32>, b_in: &CudaSlice<f32>, w: &CudaSlice<f32>,
                            res: &mut CudaSlice<f32>, z: &mut CudaSlice<f32>,
                            ncols: usize, nrows: usize, eps: f32)
                            -> Result<(CudaSlice<i8>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        assert!(ncols % 32 == 0);
        let mut q = self.alloc_uninit::<i8>(nrows * ncols)?;
        let mut d = self.alloc_uninit::<f32>(nrows * (ncols / 32))?;
        let f = self.func("add_rms_norm_zq8");
        let cfg = LaunchConfig { grid_dim: (nrows as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
        let (nc, ep) = (ncols as i32, eps);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(a).arg(b_in).arg(w).arg(res).arg(z).arg(&mut q).arg(&mut d).arg(&nc).arg(&ep);
        unsafe { b.launch(cfg)?; }
        Ok((q, d))
    }

    /// gated RMSNorm emitting q8_1 directly (fused quantize epilogue) — the ssm_out matvec input.
    /// BIT-IDENTICAL bytes to gated_rmsnorm + quantize_q8_1 (ncols % 32 == 0; blocks never straddle
    /// rows). Saves one launch per linear-attn layer (36/token on the 9B).
    pub fn gated_rmsnorm_q8_1(&self, o: &CudaSlice<f32>, w: &CudaSlice<f32>, z: &CudaSlice<f32>,
                              ncols: usize, nrows: usize, eps: f32)
                              -> Result<(CudaSlice<i8>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        assert!(ncols % 32 == 0);
        let f = self.func("gated_rmsnorm_q8_1");
        let mut out_q = self.alloc_uninit::<i8>(nrows * ncols)?;
        let mut out_d = self.alloc_uninit::<f32>(nrows * (ncols / 32))?;
        let cfg = LaunchConfig { grid_dim: (nrows as u32, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
        let (nc, ep) = (ncols as i32, eps);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(o).arg(w).arg(z).arg(&mut out_q).arg(&mut out_d).arg(&nc).arg(&ep);
        unsafe { b.launch(cfg)?; }
        Ok((out_q, out_d))
    }

    /// transpose [rows,cols] row-major -> [cols,rows] row-major.
    pub fn transpose(&self, inp: &CudaSlice<f32>, rows: usize, cols: usize)
                     -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let f = self.func("transpose_f32");
        let mut out = self.zeros(rows * cols)?;
        let cfg = LaunchConfig::for_num_elems((rows * cols) as u32);
        let (r, c) = (rows as i32, cols as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(inp).arg(&mut out).arg(&r).arg(&c);
        unsafe { b.launch(cfg)?; }
        Ok(out)
    }

    /// repeat-interleave heads: in[head_dim,n_in,T] -> out[head_dim,n_out,T].
    pub fn repeat_heads(&self, inp: &CudaSlice<f32>, out: &mut CudaSlice<f32>,
                        head_dim: usize, n_in: usize, n_out: usize, t: usize)
                        -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("repeat_heads_f32");
        let cfg = LaunchConfig::for_num_elems((head_dim * n_out * t) as u32);
        let (hd, ni, no, ti) = (head_dim as i32, n_in as i32, n_out as i32, t as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(inp).arg(out).arg(&hd).arg(&ni).arg(&no).arg(&ti);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// q|gate split (on-device). qf:[T, n_head*2*head_dim] -> q_out,gate_out:[head_dim,n_head,T].
    /// Replaces the dtoh->host-double-loop->htod in full_attn / full_attn_decode.
    pub fn q_gate_split(&self, qf: &CudaSlice<f32>, q_out: &mut CudaSlice<f32>,
                        gate_out: &mut CudaSlice<f32>, head_dim: usize, n_head: usize, t: usize)
                        -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("q_gate_split_f32");
        let cfg = LaunchConfig::for_num_elems((head_dim * n_head * t) as u32);
        let (hd, nh, ti) = (head_dim as i32, n_head as i32, t as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(qf).arg(q_out).arg(gate_out).arg(&hd).arg(&nh).arg(&ti);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// qkv->GDN repack (on-device). conv_out:[conv_dim,T] channel-major ->
    /// q_g/k_g/v_g:[d_state,num_v,T] with q/k head-repeat kh = vh % num_k (validated modulo mapping).
    /// Replaces the dtoh->host-q/k/v-repack->3x-htod in linear_attn / linear_attn_decode.
    pub fn qkv_to_gdn_repack(&self, conv_out: &CudaSlice<f32>, q_g: &mut CudaSlice<f32>,
                             k_g: &mut CudaSlice<f32>, v_g: &mut CudaSlice<f32>,
                             d_state: usize, num_v: usize, num_k: usize, key_dim: usize, t: usize)
                             -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("qkv_to_gdn_repack_f32");
        let cfg = LaunchConfig::for_num_elems((d_state * num_v * t) as u32);
        let (ds, nv, nk, kd, ti) = (d_state as i32, num_v as i32, num_k as i32, key_dim as i32, t as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(conv_out).arg(q_g).arg(k_g).arg(v_g).arg(&ds).arg(&nv).arg(&nk).arg(&kd).arg(&ti);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// conv left zero-pad (prefill from zero state). src:[conv_dim,T] -> dst:[conv_dim,T+pad],
    /// cols 0..pad = 0, cols pad..pad+T = src. `dst` MUST be pre-zeroed. No dtoh/host-loop/htod.
    pub fn conv_left_pad(&self, src: &CudaSlice<f32>, dst: &mut CudaSlice<f32>,
                         conv_dim: usize, t: usize, pad: usize)
                         -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("conv_left_pad_f32");
        let cfg = LaunchConfig::for_num_elems((conv_dim * t) as u32);
        let (cd, ti, p) = (conv_dim as i32, t as i32, pad as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(src).arg(dst).arg(&cd).arg(&ti).arg(&p);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// conv-state assemble + ring roll (decode T=1). conv_state:[conv_dim,pad] (resident),
    /// qkv_col:[conv_dim] -> conv_in:[conv_dim,pad+1]; AND rolls conv_state (keep last pad cols).
    /// Replaces the dtoh->host-conv-ring-assemble->ring-update->htod in linear_attn_decode.
    pub fn conv_assemble_and_roll(&self, qkv_col: &CudaSlice<f32>, conv_state: &mut CudaSlice<f32>,
                                  conv_in: &mut CudaSlice<f32>, conv_dim: usize, pad: usize)
                                  -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("conv_assemble_and_roll_f32");
        let cfg = LaunchConfig::for_num_elems(conv_dim as u32);
        let (cd, p) = (conv_dim as i32, pad as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(qkv_col).arg(conv_state).arg(conv_in).arg(&cd).arg(&p);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// RANK3 LEVER (conv fuse, T=1 DECODE): fused conv_assemble_and_roll + ssm_conv1d_silu in ONE
    /// launch. Assembles the conv window [conv_state | qkv_col] in registers, computes the depthwise
    /// causal conv + SiLU into `conv_out`, and rolls the ring — never materializing conv_in to HBM.
    /// Replaces e.conv_assemble_and_roll(...) + e.ssm_conv1d(...). BIT-IDENTICAL to that two-kernel
    /// sequence (same 8-wide accumulation order, same SiLU). `conv_out` is [conv_dim] (T=1).
    pub fn ssm_conv1d_fused_decode(&self, qkv_col: &CudaSlice<f32>, conv_state: &mut CudaSlice<f32>,
                                   w: &CudaSlice<f32>, conv_out: &mut CudaSlice<f32>,
                                   conv_dim: usize, d_conv: usize)
                                   -> Result<(), Box<dyn std::error::Error>> {
        let f = self.func("ssm_conv1d_fused_decode_f32");
        let cfg = LaunchConfig::for_num_elems(conv_dim as u32);
        let (cd, dc) = (conv_dim as i32, d_conv as i32);
        let mut b = self.gpu.stream.launch_builder(&f);
        b.arg(qkv_col).arg(conv_state).arg(w).arg(conv_out).arg(&cd).arg(&dc);
        unsafe { b.launch(cfg)?; }
        Ok(())
    }

    /// Copy a contiguous range [start, start+len) out of src into a fresh slice (device→device via host).
    /// Used for qkv split views. Small/rare; not perf-critical in Stage 1.
    pub fn slice_range(&self, src: &CudaSlice<f32>, start: usize, len: usize)
                       -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let host = self.gpu.stream.clone_dtoh(src)?;
        self.gpu.stream.synchronize()?;
        Ok(self.htod(&host[start..start + len])?)
    }
}
