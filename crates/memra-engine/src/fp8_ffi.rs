//! FP8-ACT PREFILL (MEMRA_PP_FP8=1): cuBLASLt FP8-E4M3 TN GEMM for the F8-E4M3-origin projections.
//!
//! Probe verdict 2026-07-08 (probe/fp8_lt_prefill.cu, JSONL row in research/tune-data): cuBLASLt
//! FP8 GEMM runs 620-795 TF at the 27B prefill shapes vs 47-72 TF for the qmatvec_gemm_q8_0 class
//! those weights ride today (46.5% of pp GPU time) — projected ~1.85x pp from the F8-native
//! layers alone. The weight side is EXACT: the checkpoint's raw e4m3 bytes + per-tensor f32
//! weight_scale are stashed at load next to the Q8_0 re-encode (`GpuTensor::Quant { fp8 }`,
//! following the `cutlass` optional-operand precedent). The only new rounding vs today is the
//! ACTIVATION: f32 -> e4m3 with ONE per-batch scalar scale (amax/448) instead of q8_1's per-32
//! int8 — finer mantissa lost, coarser scale granularity; the run-gen argmax gate arbitrates.
//!
//! Dispatch: `matmul`/`matmul_pre` m>=16 arms ONLY (prefill). Decode (m<16) keeps the Q8_0
//! dp4a/MMVQ chain bit-for-bit — the spec-exactness law is untouched, and the m=K+1 verify tier
//! (m<=9) never reaches this path.
//!
//! All device work (amax reduce, scale finalize, e4m3 quantize, cublasLtMatmul) runs on the one
//! `gpu.stream` inside a single C-ABI call (cu/fp8_prefill.cu) — no host sync anywhere: the act
//! scale is folded with weight_scale into a device scalar fed to the GEMM's B_SCALE_POINTER
//! (per-token OUTER_VEC B-scales are NOT supported on sm_120 — probed; scalar scales verified
//! exact there).

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

unsafe extern "C" {
    /// One FP8 prefill GEMM: quantize act f32->e4m3 (per-batch scalar) + cublasLtMatmul TN.
    /// Returns 0 on success (see cu/fp8_prefill.cu for the error-code bands).
    fn memra_fp8_pp_gemm(
        w_e4m3: *const core::ffi::c_void,
        x_f32: *const f32,
        xq_e4m3: *mut core::ffi::c_void,
        scales: *mut f32,
        y_f32: *mut f32,
        m: i32,
        n: i32,
        k: i32,
        w_scale: f32,
        ws: *mut core::ffi::c_void,
        ws_bytes: usize,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

/// `MEMRA_PP_FP8=1` gate (default OFF), read once. Gates BOTH the loader stash (model.rs) and the
/// prefill dispatch — unset means zero VRAM / zero dispatch change.
pub fn pp_fp8_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("MEMRA_PP_FP8")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// F8-E4M3-origin safetensors projections load as RAW e4m3 (QT_F8_E4M3) instead of the Q8_0
/// re-encode. NEW NUMERIC CONFIG: decode reads the checkpoint's own e4m3 precision (the Q8_0
/// re-encode was a lossy extra hop) via qmatvec_e4m3_mmvq; prefill (m>=16) rides the cuBLASLt FP8
/// GEMM on the SAME resident bytes — one weight copy total (frees the ~GBs the MEMRA_PP_FP8 stash
/// duplicated, no budget cap needed). Superset relationship: with this on, MEMRA_PP_FP8 and its
/// budget are irrelevant for F8-origin tensors (they never surface as Q8_0, so the stash arm never
/// fires).
///
/// DEFAULT ON since lane/fp8-decode-v1 (2026-08-05); `MEMRA_ST_E4M3=0` is the rollback seam back to
/// the Q8_0 slab. Flipped on the 27B FP8-ST receipts in `research/fp8dec-20260805/`: decode +2.58pp
/// with non-overlapping distributions (N=5 interleaved, one binary), 430 MiB freed at a measured
/// byte ratio of exactly 1.06250 (= theory, so single residency and no duplicate copy), teacher-
/// forced exactness 2/128 near-tie flips with LOWER NLL on the reference's own tape than the slab
/// arm scores on it, kernel-check ALL GREEN, run-spec K=1..8 8/8 PASS, serve-st-gate 0 failed.
///
/// SCOPE — the flip only reaches the per-tensor scalar-scale class. `find_fp8_native` returns
/// `blk: Some(grid)` for the block-128 class and `None` for per-row, and the resident arm in
/// model.rs additionally requires `blk.is_none()`, so both of those classes still take the Q8_0
/// re-encode. Nothing here changes GGUF: `TensorSource::find_fp8_native` is None for GGUF sources.
pub fn st_e4m3_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_ST_E4M3").as_deref() != Ok("0"))
}

/// Native residency for the BLOCK-128 e4m3 scale class (`QT_F8_E4M3_BLK`, lane/fp8-blk128-decode
/// 2026-08-05) — the Qwen-official FP8 class that `st_e4m3_enabled`'s arm deliberately excludes.
///
/// SHARES the `MEMRA_ST_E4M3=0` rollback seam rather than adding a second knob, per flags doctrine:
/// both arms are the same mechanism (checkpoint-native e4m3 residency + in-kernel dequant) applied
/// to the two scale classes, and one seam that turns ALL native e4m3 residency back into the Q8_0
/// slab is the behaviour a rollback wants. `MEMRA_ST_E4M3_BLK=0` additionally disables JUST this
/// class — the narrow seam that isolates the block arm while leaving the (already shipped,
/// already receipted) per-tensor arm on its default, which is what an A/B of this lane needs.
pub fn st_e4m3_blk_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        st_e4m3_enabled() && std::env::var("MEMRA_ST_E4M3_BLK").as_deref() != Ok("0")
    })
}

/// A block-128 tensor that PASSED every shape precondition for native residency but carried e4m3
/// NaN codes, so it fell through to the Q8_0 floor. Counted because "0 tensors resident as
/// F8_E4M3_BLK" is otherwise ambiguous between "not a block-128 checkpoint", "env off", and "the
/// bytes were ineligible" — three facts demanding three different responses.
static BLK_NATIVE_NAN_REFUSED: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

pub fn note_blk_native_nan_refused() {
    BLK_NATIVE_NAN_REFUSED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Tensors declined by the native block-128 arm's NaN precondition this process.
pub fn blk_native_nan_refused() -> usize {
    BLK_NATIVE_NAN_REFUSED.load(std::sync::atomic::Ordering::Relaxed)
}

/// Resident scratch for the FP8 prefill GEMM (mirrors `CutlassScratch`): the quantized activation
/// (grown to the largest m*k seen), the 4-float scale block ([0]=amax, [1]=quant mul, [2]=folded
/// B_SCALE — the GEMM desc holds a POINTER to slot 2, so the buffer must be resident/stable), and
/// the cuBLASLt workspace (64MB, the probe's size). Single GPU worker => no concurrent use; the
/// Mutex guards lazy build/grow only (matches moe_cache / cutlass_scratch).
pub struct Fp8Scratch {
    pub xq: CudaSlice<u8>,
    pub scales: CudaSlice<f32>,
    pub ws: CudaSlice<u8>,
    cap_xq: usize,
}

/// cuBLASLt workspace size — same 64MB the probe ran its heuristics with.
const FP8_WS_BYTES: usize = 64 << 20;

impl crate::Engine {
    /// FP8 prefill GEMM for a weight carrying the fp8 operand: y[m,out] = x[m,in] @ (e4m3 W)^T
    /// with the per-batch act scale and per-tensor weight_scale folded in-GEMM. Returns None when
    /// the env is off or the weight has no fp8 operand (caller falls through to the Q8_0 path).
    pub fn try_fp8_gemm(
        &self,
        w: &crate::model::GpuTensor,
        x: &CudaSlice<f32>,
        m: usize,
    ) -> Result<Option<CudaSlice<f32>>, Box<dyn std::error::Error>> {
        use crate::model::GpuTensor;
        // H100 (sm_90) is the first-class FP8 arch: cuBLASLt e4m3 GEMM is native there,
        // so the Hopper-MMA lane re-admits this path (Phase A5, ARCHITECTURE-H100.md).
        if crate::portable_mma_gated() {
            return Ok(None);
        }
        // Two e4m3 operand sources, one GEMM:
        //  * QT_F8_E4M3 (MEMRA_ST_E4M3): the RESIDENT decode bytes ARE the raw checkpoint e4m3 —
        //    prefill rides them directly (one copy, no budget). Unconditional: this dtype has no
        //    other prefill GEMM class, so the FP8 path is inherent to the config, not a flag.
        //  * fp8 stash (MEMRA_PP_FP8=1): the Q8_0-decode config's optional duplicate operand.
        //    Block-128 stash operands (blk: Some, Qwen official FP8) are SKIPPED: this GEMM
        //    feeds ONE folded scalar via B_SCALE_POINTER; a block grid through it would apply
        //    scale 1.0 to every tile. The block-scaled GEMM is P1 (probe/fp8_lt_blk_probe.cu
        //    arbitrates cuBLASLt BLK128x128 vs a scale-fold pre-pass on sm_120).
        let (w_bytes, w_scale, ne) = match w {
            GpuTensor::Quant {
                qtype,
                bytes,
                scale,
                ne,
                ..
            } if *qtype == crate::QT_F8_E4M3 => (bytes, *scale, ne),
            GpuTensor::Quant {
                fp8: Some(f8), ne, ..
            } if pp_fp8_enabled() && f8.blk.is_none() => (&f8.bytes, f8.scale, ne),
            _ => return Ok(None),
        };
        let (in_f, out_f) = (ne[0] as usize, ne[1] as usize);

        // lazy build / grow the resident scratch to this m*k
        let need_xq = m * in_f;
        let mut guard = self.fp8_scratch.lock().unwrap();
        if guard.is_none() {
            *guard = Some(Fp8Scratch {
                xq: self.alloc_u8_uninit(need_xq)?,
                scales: self.alloc_uninit::<f32>(4)?,
                ws: self.alloc_u8_uninit(FP8_WS_BYTES)?,
                cap_xq: need_xq,
            });
        }
        let s = guard.as_mut().unwrap();
        if need_xq > s.cap_xq {
            s.xq = self.alloc_u8_uninit(need_xq)?;
            s.cap_xq = need_xq;
        }

        let mut y = self.uninit(m * out_f)?; // full-overwrite GEMM output: skip memset
        let rc = {
            let stream = self.gpu.stream();
            // Hold every SyncOnDrop guard across the FFI call (same pattern as cutlass_ffi);
            // the block scope drops them before `y` is returned.
            let (w_p, _gw) = w_bytes.device_ptr(&stream);
            let (x_p, _gx) = x.device_ptr(&stream);
            let (q_p, _gq) = s.xq.device_ptr_mut(&stream);
            let (sc_p, _gs) = s.scales.device_ptr_mut(&stream);
            let (y_p, _gy) = y.device_ptr_mut(&stream);
            let (ws_p, _gws) = s.ws.device_ptr_mut(&stream);
            unsafe {
                memra_fp8_pp_gemm(
                    w_p as *const core::ffi::c_void,
                    x_p as *const f32,
                    q_p as *mut core::ffi::c_void,
                    sc_p as *mut f32,
                    y_p as *mut f32,
                    m as i32,
                    out_f as i32,
                    in_f as i32,
                    w_scale,
                    ws_p as *mut core::ffi::c_void,
                    FP8_WS_BYTES,
                    stream.cu_stream() as *mut core::ffi::c_void,
                )
            }
        };
        if rc != 0 {
            return Err(format!(
                "memra_fp8_pp_gemm rc={rc} (m={m} n={out_f} k={in_f}; 1xxxx=cudaError quant chain, \
                 2xxxx=no cublasLt algo, 3xxxx=matmul status)"
            )
            .into());
        }
        Ok(Some(y))
    }
}

// ============================================================================================
// P1 option (b) — PER-BLOCK FP8 MMQ prefill (cu/mmq_fp8_blk.cu, lane/fp8-mmq)
// ============================================================================================

/// `MEMRA_FP8_MMQ=1` gate for the per-block MMQ tile's **STASH** operand source (default OFF;
/// lane/fp8-mmq 2026-08-04): a SECOND e4m3 copy uploaded next to an already-resident Q8_0 slab,
/// spending from `MEMRA_PP_FP8_BUDGET_MB`.
///
/// This is the third and only exact-AND-fast option from P1-VERDICT.md. cuBLASLt cannot take the
/// grid at all on sm_120; ARM A's per-tensor fold is fast but diverges at greedy pos 20; ARM B' is
/// exact but lands on the Q8_0 MMQ. This arm consumes the checkpoint's e4m3 bytes and the
/// per-[128x128] f32 grid directly, with no re-quantization on either operand's weight side.
///
/// STAYS DEFAULT OFF for the stash source, and the reason is the v2 verdict, not inertia: against a
/// floor whose Q8_0 slab is ALREADY RESIDENT the tile is 0.85-1.09x GEMM-only, so paying a full
/// duplicate weight copy to reach it is not a win (`lane/fp8-mmq-v2` LANE-VERDICT.jsonl). This
/// function is ALSO what admits the stash at load (`model.rs`), so it must stay an explicit opt-in:
/// the native-resident flip below must not silently start duplicating Q8_0 tensors.
pub fn fp8_mmq_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("MEMRA_FP8_MMQ")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// Same tile, **NATIVE-RESIDENT** operand source: a `QT_F8_E4M3_BLK` tensor's own `blk` grid, the
/// checkpoint's single copy with no slab and no stash. DEFAULT ON since lane/fp8-blk128-decode
/// (2026-08-05); `MEMRA_FP8_MMQ=0` is the narrow seam back to dequant-per-call.
///
/// WHY THE SAME TILE DEFAULTS DIFFERENTLY BY SOURCE — the denominator differs, so the sign does.
/// With a stash the floor already has its Q8_0 slab resident and the comparison is tile vs tile
/// (v2: 0.85-1.09x, i.e. not worth a duplicate copy). On the native-resident class the floor must
/// also CREATE that slab on every prefill call — 27.9 ms/pass of dequant after the vector rewrite,
/// 14.19 GB of extra weight traffic — so the tile only has to not be 27.9 ms worse than a kernel it
/// trails by at most ~15% on a subset of shapes. Measured 3-arm interleaved on the 27B block-128
/// checkpoint (research/fp8blk-20260805/, N=3, one lock hold, one md5-pinned binary):
/// slab 1540.5 / dequant-per-call 1449.1 / **this tile 1553.3** tok/s, min(C) 1552.4 > max(A) 1541.1
/// (non-overlapping) = +0.83% pp512 AHEAD of the Q8_0 floor instead of -5.8% behind it.
///
/// EXACTNESS, the condition the flip was deferred on (6b741068: "the default flip waits on this
/// arm's own exactness cells ... rather than inheriting the dequant arm's"). Branch-(b): per-block
/// f8f6f4 MMA is not the Q8_0 re-encode's arithmetic, so bit-identity is the wrong bar. Measured on
/// `prime_cache` — the class that actually dispatches this kernel — with a dispatch ledger on every
/// arm (624 = 208 projections x 3 passes, full coverage) and an A==B bit-identical control proving
/// the instrument can see zero where zero is: argmax UNCHANGED and the top-10 order identical to the
/// floor's, rms_rel 2.5e-2 on an rms-2.7155 logit vector, and teacher-forced NLL on the prompt's own
/// continuation LOWER than the floor's (2.764722 vs 2.787267) on a tape neither arm produced.
/// `MEMRA_ST_E4M3_BLK=0` / `MEMRA_ST_E4M3=0` also disable this route, by removing the native operand
/// it consumes — this seam exists for the narrower question (keep native decode residency, revert
/// only the prefill route).
pub fn fp8_blk_mmq_native_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_FP8_MMQ").as_deref() != Ok("0"))
}

/// Per-tensor e4m3-NaN verdicts, keyed by the weight's device pointer. The hardware MMA reads
/// magnitude 0x7F as NaN while the host / ARM B' reference decodes it to 0.0, so a tensor
/// containing any must NOT ride this kernel. The scan is a full pass over the weight, so it runs
/// ONCE per tensor (first prefill dispatch) and the verdict is cached — never per-GEMM.
static FP8_MMQ_NAN_OK: std::sync::Mutex<Option<std::collections::HashMap<u64, bool>>> =
    std::sync::Mutex::new(None);

impl crate::Engine {
    /// PER-BLOCK FP8 MMQ prefill GEMM for a weight carrying a block-128 fp8 operand:
    /// y[m,out] = x[m,in] @ (e4m3 W)^T with each [128x128] weight block scaled by its own f32.
    /// Returns None when the env is off, the weight has no block-128 fp8 operand, the shape is
    /// unsupported, or the NaN precondition fails (caller falls through to the Q8_0 floor).
    pub fn try_fp8_blk_mmq(
        &self,
        w: &crate::model::GpuTensor,
        x: &CudaSlice<f32>,
        m: usize,
    ) -> Result<Option<CudaSlice<f32>>, Box<dyn std::error::Error>> {
        use crate::model::GpuTensor;
        // Entry counter BEFORE the env gate: a ledger of all zeros is otherwise ambiguous between
        // "the flag was not seen" and "no prefill GEMM ever reached this hook".
        FP8_MMQ_ENTRIES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if crate::portable_mma_gated() {
            FP8_MMQ_GATE_OFF.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok(None);
        }
        // TWO OPERAND SOURCES, in this order, EACH WITH ITS OWN DEFAULT (2026-08-05):
        //
        //  (1) the `fp8` STASH — a SECOND e4m3 copy uploaded alongside a resident Q8_0 slab under
        //      `MEMRA_PP_FP8` / `MEMRA_PP_FP8_BUDGET_MB`. This is what the v1/v2 MMQ lanes measured,
        //      and it stays behind `MEMRA_FP8_MMQ=1` (`fp8_mmq_enabled`): against a floor whose slab
        //      is already resident the tile is 0.85-1.09x, which does not pay for a duplicate copy.
        //
        //  (2) the `blk` RESIDENCY field on a `QT_F8_E4M3_BLK` tensor (lane/fp8-blk128-decode) —
        //      the checkpoint-native single copy, no slab and no stash. Same bytes, same grid, same
        //      layout contract, so the kernel cannot tell them apart; only the owner differs. This
        //      source is DEFAULT ON (`fp8_blk_mmq_native_enabled`), because its floor must build the
        //      Q8_0 slab every call (27.9 ms/pass) and the tile measured +0.83% pp512 ahead of it
        //      with non-overlapping distributions.
        //
        // The gate is therefore checked PER SOURCE, after the operand is known — not once up front.
        // Checking it before the match would make the native flip also flip the stash, and
        // `fp8_mmq_enabled` is what admits that stash at LOAD time (model.rs), so a shared gate
        // would silently start duplicating every Q8_0 tensor with an fp8 sibling.
        //
        // Why (1) first: when a stash exists the tensor is ALSO a Q8_0 slab, and the stash is the
        // operand that arm's budget accounting owns. A `QT_F8_E4M3_BLK` tensor never has a stash
        // (its residency arm sets `fp8: None`), so the two cases are disjoint in practice and the
        // order only fixes a hypothetical.
        let (f8_bytes, f8_scale, blk, ne) = match w {
            GpuTensor::Quant {
                fp8: Some(f8), ne, ..
            } if f8.blk.is_some() => {
                if !fp8_mmq_enabled() {
                    FP8_MMQ_GATE_OFF.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return Ok(None);
                }
                (&f8.bytes, f8.scale, f8.blk.as_ref().unwrap(), ne)
            }
            GpuTensor::Quant {
                bytes, qtype, scale, blk: Some(g), ne, ..
            } if *qtype == crate::QT_F8_E4M3_BLK => {
                if !fp8_blk_mmq_native_enabled() {
                    FP8_MMQ_GATE_OFF.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return Ok(None);
                }
                (bytes, *scale, g, ne)
            }
            // A zero dispatch count is ambiguous on its own: no block operand resident looks
            // exactly like a shape refusal. Count the no-operand case separately so the receipt
            // says WHICH, and never per-GEMM-log (this fires on every projection of every layer).
            _ => {
                FP8_MMQ_NO_OPERAND.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Ok(None);
            }
        };
        if ne.len() != 2 {
            FP8_MMQ_BAD_SHAPE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok(None);
        }
        let (in_f, out_f) = (ne[0] as usize, ne[1] as usize);
        // in_f % 16: the kernel's 16B tile-copy line. blk grid dims must match the shape — a
        // mismatch means the operand and the grid came from different tensors; refuse rather than
        // index a wrong block.
        if in_f % 16 != 0
            || blk.rows != out_f.div_ceil(128)
            || blk.cols != in_f.div_ceil(128)
            || f8_bytes.len() < out_f * in_f
            || x.len() < m * in_f
        {
            FP8_MMQ_BAD_SHAPE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok(None);
        }
        // The per-tensor scale must be the block class's identity (source.rs sets 1.0 alongside a
        // grid); anything else would mean a second, unapplied scale factor.
        if f8_scale != 1.0 {
            FP8_MMQ_BAD_SCALE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok(None);
        }

        // One-time NaN precondition per tensor (cached by device pointer).
        {
            let key = {
                let stream = self.gpu.stream();
                let (p, _g) = f8_bytes.device_ptr(&stream);
                p as u64
            };
            let mut guard = FP8_MMQ_NAN_OK.lock().unwrap();
            let map = guard.get_or_insert_with(std::collections::HashMap::new);
            let ok = match map.get(&key) {
                Some(v) => *v,
                None => {
                    let v = self.fp8_blk_nan_count(f8_bytes)? == 0;
                    map.insert(key, v);
                    v
                }
            };
            if !ok {
                FP8_MMQ_NAN_REFUSED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Ok(None);
            }
        }

        let y = self.qmatvec_mmq_fp8_blk(f8_bytes, &blk.scales, x, m, in_f, out_f)?;
        FP8_MMQ_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(Some(y))
    }
}

/// Dispatch counter for this kernel. A model-level exactness or perf result is only evidence if
/// the kernel actually RAN — a silently-refused precondition (no block operand made resident, the
/// stash budget spent before the tensor, a NaN code present) looks exactly like "bit-identical to
/// the floor" and "no perf change". `MEMRA_FP8_MMQ_STATS=1` prints the count at process exit so
/// every such run carries its own proof of coverage.
static FP8_MMQ_HITS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Refusal counters, one per precondition. Without these a `dispatches: 0` receipt says only
/// "the kernel did not run", which is the same string for "no block operand was ever made
/// resident" (budget spent / loader arm not taken / not a block-128 checkpoint) and for "the
/// operand was there but the shape or the NaN scan rejected it". Those demand opposite fixes, so
/// the receipt has to distinguish them.
static FP8_MMQ_NO_OPERAND: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static FP8_MMQ_BAD_SHAPE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static FP8_MMQ_BAD_SCALE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static FP8_MMQ_NAN_REFUSED: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
static FP8_MMQ_ENTRIES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static FP8_MMQ_GATE_OFF: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Number of prefill GEMMs that went through the per-block FP8 MMQ tile so far this process.
pub fn fp8_mmq_hits() -> usize {
    FP8_MMQ_HITS.load(std::sync::atomic::Ordering::Relaxed)
}

/// `entries, gate_off, hits, no_operand, bad_shape, bad_scale, nan_refused` — the full ledger.
/// `entries` is incremented before any guard, so `entries == 0` means no prefill GEMM reached the
/// hook at all (a dispatch-wiring fact), while `gate_off == entries` means the flag was not seen.
pub fn fp8_mmq_ledger() -> (usize, usize, usize, usize, usize, usize, usize) {
    use std::sync::atomic::Ordering::Relaxed;
    (
        FP8_MMQ_ENTRIES.load(Relaxed),
        FP8_MMQ_GATE_OFF.load(Relaxed),
        FP8_MMQ_HITS.load(Relaxed),
        FP8_MMQ_NO_OPERAND.load(Relaxed),
        FP8_MMQ_BAD_SHAPE.load(Relaxed),
        FP8_MMQ_BAD_SCALE.load(Relaxed),
        FP8_MMQ_NAN_REFUSED.load(Relaxed),
    )
}

// ============================================================================================
// ARM B' — device-side block-128 FP8 -> Q8_0 dequant pass (cu/fp8_blk_dequant.cu)
// ============================================================================================

unsafe extern "C" {
    /// Q8_0 slab bytes for an `[out_dim x in_dim]` weight (0 = bad dims).
    fn memra_fp8_blk_q8_0_bytes(out_dim: i32, in_dim: i32) -> usize;
    /// One device pass: e4m3 codes + block-128 f32 scale grid -> Q8_0 blocks.
    /// rc: 0 ok, 1 bad dims, else a cudaError_t.
    fn memra_fp8_blk_dequant_q8_0(
        f8_weights: *const core::ffi::c_void,
        blk_scales: *const f32,
        out_q8: *mut core::ffi::c_void,
        out_dim: i32,
        in_dim: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

/// `MEMRA_FP8_BLK_GPU=1` gate (default OFF; ARM B', lane fp8-gemm-arm 2026-08-03): block-128
/// FP8 safetensors weights dequant to Q8_0 ON THE GPU at load instead of host-dequant +
/// host-re-encode. Bit-parity with the CPU path is a kernel-check gate (`fp8-blk-gpu` arm) and
/// a real-checkpoint argmax gate — the flag exists because the CPU path stays default until
/// both are green on the 5090.
pub fn fp8_blk_gpu_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("MEMRA_FP8_BLK_GPU")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

impl crate::Engine {
    /// Q8_0 slab byte count for an `[out_f, in_f]` block-128 FP8 weight.
    pub fn fp8_blk_q8_0_bytes(out_f: usize, in_f: usize) -> usize {
        unsafe { memra_fp8_blk_q8_0_bytes(out_f as i32, in_f as i32) }
    }

    /// ARM B' load-time pass: upload the raw e4m3 codes + the block-128 scale grid, dequant on
    /// the GPU, and return the Q8_0 slab (byte-identical to the host re-encode). `f8` is the
    /// checkpoint's row-major `[out_f x in_f]` codes; `grid` is the row-major
    /// `[ceil(out_f/128) x ceil(in_f/128)]` f32 scale grid (F8BlockGrid order, verbatim).
    pub fn fp8_blk_dequant_q8_0(
        &self,
        f8: &[u8],
        grid: &[f32],
        out_f: usize,
        in_f: usize,
    ) -> Result<CudaSlice<u8>, Box<dyn std::error::Error>> {
        let (rows, cols) = (out_f.div_ceil(128), in_f.div_ceil(128));
        if f8.len() != out_f * in_f {
            return Err(format!(
                "fp8_blk_dequant_q8_0: f8 len {} != out_f*in_f {}",
                f8.len(),
                out_f * in_f
            )
            .into());
        }
        if grid.len() != rows * cols {
            return Err(format!(
                "fp8_blk_dequant_q8_0: grid len {} != rows*cols {rows}*{cols}",
                grid.len()
            )
            .into());
        }
        let need = Self::fp8_blk_q8_0_bytes(out_f, in_f);
        if need == 0 {
            return Err(format!(
                "fp8_blk_dequant_q8_0: bad dims out_f={out_f} in_f={in_f} (in_f must be %32)"
            )
            .into());
        }
        let src = self.htod_bytes(f8)?;
        let scales = self.htod(grid)?;
        let dst = self.fp8_blk_dequant_q8_0_dev(&src, &scales, out_f, in_f)?;
        self.gpu.stream().synchronize()?;
        Ok(dst)
    }

    /// DEVICE-RESIDENT twin of `fp8_blk_dequant_q8_0` (lane/fp8-blk128-decode): identical kernel,
    /// identical output bytes, but the e4m3 codes and the scale grid are ALREADY on the device and
    /// there is no trailing `synchronize`.
    ///
    /// Both differences matter to its caller (`try_e4m3_blk_prefill`, per prefill call rather than
    /// once per load): the host arm's two htods would re-upload a weight that is already resident,
    /// and its `synchronize` would stall the CUDA owner thread on every prefill projection. Stream
    /// ordering is sufficient without it — the dequant and the Q8_0 GEMM that consumes `dst` are
    /// issued to the SAME stream, so the GEMM cannot observe a partially written slab.
    pub fn fp8_blk_dequant_q8_0_dev(
        &self,
        f8: &CudaSlice<u8>,
        grid: &CudaSlice<f32>,
        out_f: usize,
        in_f: usize,
    ) -> Result<CudaSlice<u8>, Box<dyn std::error::Error>> {
        let (rows, cols) = (out_f.div_ceil(128), in_f.div_ceil(128));
        if f8.len() < out_f * in_f {
            return Err(format!(
                "fp8_blk_dequant_q8_0_dev: f8 len {} < out_f*in_f {}",
                f8.len(),
                out_f * in_f
            )
            .into());
        }
        if grid.len() < rows * cols {
            return Err(format!(
                "fp8_blk_dequant_q8_0_dev: grid len {} < rows*cols {rows}*{cols}",
                grid.len()
            )
            .into());
        }
        let need = Self::fp8_blk_q8_0_bytes(out_f, in_f);
        if need == 0 {
            return Err(format!(
                "fp8_blk_dequant_q8_0_dev: bad dims out_f={out_f} in_f={in_f} (in_f must be %32)"
            )
            .into());
        }
        let mut dst = self.alloc_u8_uninit(need)?;
        let rc = {
            let stream = self.gpu.stream();
            let (s_p, _gs) = f8.device_ptr(&stream);
            let (g_p, _gg) = grid.device_ptr(&stream);
            let (d_p, _gd) = dst.device_ptr_mut(&stream);
            unsafe {
                memra_fp8_blk_dequant_q8_0(
                    s_p as *const core::ffi::c_void,
                    g_p as *const f32,
                    d_p as *mut core::ffi::c_void,
                    out_f as i32,
                    in_f as i32,
                    stream.cu_stream() as *mut core::ffi::c_void,
                )
            }
        };
        if rc != 0 {
            return Err(format!(
                "memra_fp8_blk_dequant_q8_0 rc={rc} (out_f={out_f} in_f={in_f}; 1=bad dims, \
                 else cudaError_t)"
            )
            .into());
        }
        Ok(dst)
    }
}
