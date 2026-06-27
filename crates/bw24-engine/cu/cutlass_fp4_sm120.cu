// bw24 CUTLASS 4.x sm_120a NVFP4 (W4A4) GEMM wrapper — Phase 0.
//
// ONE instantiated config: CtaShape 128x128x128, 1x1x1 cluster, KernelTmaWarpSpecializedCooperative,
// StaticPersistentScheduler, OutT = float (gate dtype; keeps prefill logits f32). This is the exact
// collective shape flashinfer's DeviceGemmFp4GemmSm120 (W4A4_NVFP4_NVFP4) builds, copied 1:1 per the
// integration plan §2.1, distilled down to a single non-templated TU so nvcc instantiates ONE kernel
// (measured ~29s / 5.2GB on-box for one config; the heuristic sweep is what blows up — we never do it).
//
// Compiled to a sm_120a STATIC LIB (build.rs), NOT a fatbin: CUTLASS needs its host-side
// GemmUniversalAdapter::run() (grid calc, TMA descriptors, workspace), which is host C++ called over FFI.
//
// On-box verified corrections baked in (see /tmp/cutlass_probe, plan §2.2):
//   - OutT MUST be cutlass::bfloat16_t / float, NOT __nv_bfloat16 (else hard static_assert
//     "Unknown TMA Format!" at copy_sm90_desc.hpp).
//   - The static archive MUST be whole-archived at link (build.rs) or the CUDART fatbin-registration
//     global ctor is dropped and the device kernel silently never registers.
//
// Operand layout contract (what the caller in lib.rs must supply):
//   A (activation): [m, k] RowMajor, packed e2m1 2-nibbles/byte  -> a_e2m1 is [m, k/2] bytes
//   B (weight):     [n, k] ColumnMajor (== K-major rows of n)    -> b_e2m1 is [n, k/2] bytes
//   SFA/SFB:        float_ue4m3_t, one scale per SFVecSize=16 elems, in CUTLASS's swizzled SfAtom
//                   layout. Size each via bw24_cutlass_sf_size(); scatter linear->swizzled via
//                   bw24_cutlass_repack_sf{a,b} so the swizzle lives in the TU that consumes it.
//   alpha:          scalar fused into the epilogue (LinearCombination alpha_ptr). Fold bw24's
//                   per-tensor scale here as 1/scale (host passes a device float*).
//   D (output):     [m, n] RowMajor, float.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"

using namespace cute;

// ----- the ONE config (mirrors flashinfer W4A4_NVFP4_NVFP4, CtaShape128x128x128, SM120) -----
using OutElementType  = float;                       // gate dtype (Phase 0); flip to cutlass::bfloat16_t for prod
using CTAShape        = cute::Shape<cute::Int<128>, cute::Int<128>, cute::Int<128>>;
using Arch            = cutlass::arch::Sm120;
using ClusterShape    = cute::Shape<_1, _1, _1>;
using ElementA        = cutlass::nv_float4_t<cutlass::float_e2m1_t>;   // NVFP4 activation
using LayoutA         = cutlass::layout::RowMajor;
static constexpr int AlignmentA = 32;
using ElementB        = cutlass::nv_float4_t<cutlass::float_e2m1_t>;   // NVFP4 weight
using LayoutB         = cutlass::layout::ColumnMajor;
static constexpr int AlignmentB = 32;
using ElementC        = void;
using LayoutC         = cutlass::layout::RowMajor;
static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<OutElementType>::value;

using SFType             = cutlass::float_ue4m3_t;
using ElementCompute     = float;
using ElementAccumulator = float;
using OperatorClass      = cutlass::arch::OpClassBlockScaledTensorOp;
using FusionOperation    =
    cutlass::epilogue::fusion::LinearCombination<OutElementType, float, void, float>;
using ThreadBlockShape   = CTAShape;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    Arch, cutlass::arch::OpClassTensorOp, ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC, OutElementType, LayoutC, AlignmentC,
    cutlass::epilogue::TmaWarpSpecialized, FusionOperation>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    Arch, OperatorClass, ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
    ElementAccumulator, ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;

using TileSchedulerTag = cutlass::gemm::StaticPersistentScheduler;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, TileSchedulerTag>;
using Gemm = typename cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using ElementD = typename Gemm::ElementD;

// Build CUTLASS Arguments for one (m,n,k). global_sf is a device float* holding `alpha` (the epilogue
// LinearCombination scalar); we route bw24's per-tensor scale through it instead of a post-matmul kernel.
static typename Gemm::Arguments make_args(
    const void* a_e2m1, const void* b_e2m1, const void* sfa, const void* sfb,
    const float* alpha_dev, void* d, int m, int n, int k) {
  typename Gemm::Arguments args;
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.problem_shape = cute::make_shape(m, n, k, 1);

  args.mainloop.ptr_A  = static_cast<cutlass::float_e2m1_t const*>(a_e2m1);
  args.mainloop.ptr_B  = static_cast<cutlass::float_e2m1_t const*>(b_e2m1);
  args.mainloop.ptr_SFA = static_cast<cutlass::float_ue4m3_t const*>(sfa);
  args.mainloop.ptr_SFB = static_cast<cutlass::float_ue4m3_t const*>(sfb);
  args.mainloop.dA = cutlass::make_cute_packed_stride(
      typename Gemm::GemmKernel::StrideA{}, cute::make_shape(m, k, 1));
  args.mainloop.dB = cutlass::make_cute_packed_stride(
      typename Gemm::GemmKernel::StrideB{}, cute::make_shape(n, k, 1));
  args.mainloop.layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(args.problem_shape);
  args.mainloop.layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(args.problem_shape);

  args.epilogue.ptr_C = nullptr;                                 // ElementC = void
  args.epilogue.ptr_D = static_cast<ElementD*>(d);
  args.epilogue.dC = cutlass::make_cute_packed_stride(
      typename Gemm::GemmKernel::StrideC{}, cute::make_shape(m, n, 1));
  args.epilogue.dD = cutlass::make_cute_packed_stride(
      typename Gemm::GemmKernel::StrideD{}, cute::make_shape(m, n, 1));
  args.epilogue.thread.alpha_ptr = alpha_dev;                    // device scalar (== 1/scale)
  return args;
}

extern "C" {

// Workspace size for (m,n,k); host-only, no launch. Caller pre-allocs the max over prefill shapes.
size_t bw24_cutlass_fp4_workspace(int m, int n, int k) {
  auto args = make_args(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, m, n, k);
  Gemm gemm;
  return Gemm::get_workspace_size(args);
}

// Number of float_ue4m3_t scale-factor bytes for the swizzled SFA layout of an (m,k) operand.
// (cosize of the CUTLASS SfAtom layout — what tile_atom_to_shape_SFA covers.)
size_t bw24_cutlass_sfa_size(int m, int k) {
  auto ps = cute::make_shape(m, /*n*/1, k, 1);
  auto layout = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(ps);
  return static_cast<size_t>(cute::cosize(layout));
}
size_t bw24_cutlass_sfb_size(int n, int k) {
  auto ps = cute::make_shape(/*m*/1, n, k, 1);
  auto layout = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(ps);
  return static_cast<size_t>(cute::cosize(layout));
}

// Run one NVFP4 GEMM. alpha_dev: device float* == 1/scale (epilogue scalar). Returns 0 on success,
// else (cutlass::Status as int). Output D is f32 [m,n] RowMajor.
int bw24_cutlass_fp4_gemm(
    const void* a_e2m1, const void* b_e2m1, const void* sfa, const void* sfb,
    const float* alpha_dev, void* d, int m, int n, int k,
    void* workspace, size_t workspace_bytes, void* stream) {
  auto args = make_args(a_e2m1, b_e2m1, sfa, sfb, alpha_dev, d, m, n, k);
  Gemm gemm;
  size_t need = gemm.get_workspace_size(args);
  if (need > workspace_bytes) return 1001;                       // caller under-sized the workspace
  auto st = gemm.can_implement(args);
  if (st != cutlass::Status::kSuccess) return 2000 + static_cast<int>(st);
  st = gemm.initialize(args, workspace, reinterpret_cast<cudaStream_t>(stream));
  if (st != cutlass::Status::kSuccess) return 3000 + static_cast<int>(st);
  st = gemm.run(args, workspace, reinterpret_cast<cudaStream_t>(stream));
  return static_cast<int>(st);
}

} // extern "C"

// ----- SF scatter helpers: linear (row, k/16) ue4m3 bytes -> CUTLASS swizzled SfAtom layout -----
// The swizzle lives in THIS TU (plan §2.4: "do NOT hand-roll the swizzle"): we index the exact
// cute layout that tile_atom_to_shape_SF{A,B} returns. One device thread per (mn, ksf) scale.
template <bool IsA>
__global__ void bw24_sf_scatter_kernel(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
                                       int mn, int k) {
  // src is linear: src[row * (k/16) + ksf]; one ue4m3 byte per 16 K-elements.
  int ksf_count = k / 16;
  long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
  long total = (long)mn * ksf_count;
  if (idx >= total) return;
  int row = (int)(idx / ksf_count);
  int ksf = (int)(idx % ksf_count);
  // The SF layout spans the FULL (MN, K, L) coordinate space: the inner SFVecSize=16 K-sub-dim has
  // stride 0, so the 16 consecutive K elements [ksf*16 .. ksf*16+15] all map to ONE SF byte. Index
  // with the full-K coordinate ksf*16 (NOT ksf) and let cute compute the swizzled byte offset.
  int k_full = ksf * 16;
  if (IsA) {
    auto layout = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(mn, 1, k, 1));
    long off = layout(cute::make_coord(row, k_full, 0));
    dst[off] = src[idx];
  } else {
    auto layout = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(1, mn, k, 1));
    long off = layout(cute::make_coord(row, k_full, 0));
    dst[off] = src[idx];
  }
}

// ----- NVFP4 reference quantizer (Phase-0 smoke test ONLY) -----
// Quantize a [rows, k] f32 matrix to packed e2m1 (2/byte) + linear ue4m3 per-16 scales, using
// CUTLASS's OWN float_e2m1_t/float_ue4m3_t converting constructors so the bytes are exactly what the
// GEMM decodes. NVFP4 semantics: scale = amax(block_of_16)/6 (E2M1 max=6); q = round(x/scale) -> e2m1.
// This is a TEST oracle (the production path quantizes from GGUF bytes, no re-quant). One thread/elem.
__global__ void bw24_nvfp4_quant_kernel(const float* __restrict__ src, uint8_t* __restrict__ packed,
                                        uint8_t* __restrict__ scales, int rows, int k) {
  int ksf = k / 16;
  long blk = blockIdx.x * (long)blockDim.x + threadIdx.x;   // one thread per 16-elem block
  long total = (long)rows * ksf;
  if (blk >= total) return;
  int row = (int)(blk / ksf);
  int sb  = (int)(blk % ksf);
  const float* p = src + (long)row * k + (long)sb * 16;
  float amax = 0.f;
  for (int i = 0; i < 16; ++i) { float a = fabsf(p[i]); if (a > amax) amax = a; }
  float scale = amax > 0.f ? amax / 6.0f : 1.0f;            // E2M1 max magnitude = 6
  cutlass::float_ue4m3_t sf(scale);
  float scale_q = float(sf);                                 // the actually-stored (rounded) scale
  float inv = scale_q > 0.f ? 1.0f / scale_q : 0.f;
  scales[(long)row * ksf + sb] = sf.storage;                 // linear (row, ksf) ue4m3 byte
  // pack 16 e2m1 nibbles -> 8 bytes, low nibble = even index
  for (int i = 0; i < 16; i += 2) {
    cutlass::float_e2m1_t lo(p[i]   * inv);
    cutlass::float_e2m1_t hi(p[i+1] * inv);
    uint8_t b = (uint8_t)((lo.storage & 0xf) | ((hi.storage & 0xf) << 4));
    packed[((long)row * k + (long)sb * 16 + i) / 2] = b;
  }
}

// Dequantize packed e2m1 + linear ue4m3 scales back to f32 [rows,k] (TEST oracle). Uses CUTLASS's own
// float_e2m1_t/float_ue4m3_t -> float conversions so the CPU reference sees exactly the GEMM's inputs.
__global__ void bw24_nvfp4_dequant_kernel(const uint8_t* __restrict__ packed,
                                          const uint8_t* __restrict__ scales,
                                          float* __restrict__ dst, int rows, int k) {
  long e = blockIdx.x * (long)blockDim.x + threadIdx.x;     // one thread per element
  long total = (long)rows * k;
  if (e >= total) return;
  int row = (int)(e / k);
  int col = (int)(e % k);
  int ksf = k / 16;
  uint8_t sf_byte = scales[(long)row * ksf + col / 16];
  cutlass::float_ue4m3_t sf = cutlass::float_ue4m3_t::bitcast(sf_byte);
  uint8_t byte = packed[e / 2];
  uint8_t nib = (col & 1) ? (byte >> 4) & 0xf : byte & 0xf;
  cutlass::float_e2m1_t q = cutlass::float_e2m1_t::bitcast(nib);
  dst[e] = float(q) * float(sf);
}

extern "C" {
// Dequantize packed e2m1 + linear ue4m3 scales -> f32 [rows,k].
int bw24_nvfp4_dequant_ref(const void* packed_e2m1, const void* scales_linear, void* dst_f32,
                           int rows, int k, void* stream) {
  long total = (long)rows * k;
  int block = 256;
  int grid = (int)((total + block - 1) / block);
  bw24_nvfp4_dequant_kernel<<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint8_t*>(packed_e2m1), static_cast<const uint8_t*>(scales_linear),
      static_cast<float*>(dst_f32), rows, k);
  return static_cast<int>(cudaGetLastError());
}
} // extern "C"

extern "C" {
// Quantize [rows,k] f32 -> packed e2m1 (rows*k/2 bytes) + linear ue4m3 scales (rows*k/16 bytes).
int bw24_nvfp4_quant_ref(const void* src_f32, void* packed_e2m1, void* scales_linear,
                         int rows, int k, void* stream) {
  int ksf = k / 16;
  long total = (long)rows * ksf;
  int block = 256;
  int grid = (int)((total + block - 1) / block);
  bw24_nvfp4_quant_kernel<<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(src_f32), static_cast<uint8_t*>(packed_e2m1),
      static_cast<uint8_t*>(scales_linear), rows, k);
  return static_cast<int>(cudaGetLastError());
}
} // extern "C"

// ----- GGUF NVFP4 weight de-interleave (Phase 1, repack #1) -----
// GGUF block_nvfp4: per ROW, in_f/64 blocks of 36 bytes = [4 ue4m3 scale bytes | 32 qs e2m1 bytes].
// Element k of a row: block g=k/64, r=k%64, sub-block s=r/16, within w=r%16; the e2m1 nibble lives in
// qs[s*8 + (w&7)], LOW nibble if w<8 else HIGH. (Same per-16-subblock interleave the in-kernel repack
// at qmatvec_gemm.cu undoes — bits copied UNCHANGED, no fp4_shift: that is internal to CUTLASS.)
//
// CUTLASS wants the B operand as plain K-contiguous packed e2m1, 2 nibbles/byte: dst[(row,k)] occupies
// byte (row*k + k_idx)/2, low nibble if k_idx even. We pack two consecutive K elements (2j, 2j+1) per
// output byte. The 4 ue4m3 scale bytes/block are emitted as a LINEAR [n, k/16] SFB (one per 16 elems),
// exactly the format bw24_cutlass_repack_sfb then scatters into the swizzled layout. One thread per
// output byte (covers K elements 2j and 2j+1 of one row).
__device__ __forceinline__ uint8_t gguf_nvfp4_nibble(const uint8_t* qs_block, int r) {
  int s = r >> 4, w = r & 15;
  uint8_t byte = qs_block[s * 8 + (w & 7)];
  return (w < 8) ? (byte & 0xF) : ((byte >> 4) & 0xF);
}
__global__ void bw24_gguf_nvfp4_deinterleave_kernel(const uint8_t* __restrict__ src, long row_bytes,
                                                    uint8_t* __restrict__ b_packed,
                                                    uint8_t* __restrict__ sfb_linear,
                                                    int n, int k) {
  long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;   // one thread per packed output byte
  long total = (long)n * (k / 2);
  if (idx >= total) return;
  int kp = k / 2;
  int row = (int)(idx / kp);
  int j   = (int)(idx % kp);                                 // packs K elements 2j, 2j+1
  const uint8_t* rowp = src + (long)row * row_bytes;
  int k0 = 2 * j, k1 = 2 * j + 1;
  const uint8_t* blk0 = rowp + (long)(k0 / 64) * 36;
  const uint8_t* blk1 = rowp + (long)(k1 / 64) * 36;
  uint8_t lo = gguf_nvfp4_nibble(blk0 + 4, k0 % 64);
  uint8_t hi = gguf_nvfp4_nibble(blk1 + 4, k1 % 64);
  b_packed[idx] = (uint8_t)((lo & 0xF) | ((hi & 0xF) << 4));
  // Emit the linear SFB byte for this 16-elem sub-block once (when this thread owns its first element).
  if ((k0 & 15) == 0) {
    int ksf = k0 >> 4;                                       // sub-block index along K
    const uint8_t* blk = rowp + (long)(k0 / 64) * 36;
    int s = (k0 % 64) >> 4;                                  // which of the 4 scale bytes in the block
    sfb_linear[(long)row * (k / 16) + ksf] = blk[s];
  }
}

extern "C" {
// De-interleave a GGUF NVFP4 weight tensor (raw [n] rows of row_bytes) into CUTLASS B operand:
//   b_packed   : [n, k/2] plain K-contiguous packed e2m1 (2 nibbles/byte)
//   sfb_linear : [n, k/16] ue4m3 scales in linear (row, k/16) order (feed to bw24_cutlass_repack_sfb)
// Returns cudaGetLastError as int. One-time, at load.
int bw24_gguf_nvfp4_deinterleave(const void* src, long row_bytes, void* b_packed, void* sfb_linear,
                                 int n, int k, void* stream) {
  long total = (long)n * (k / 2);
  int block = 256;
  int grid = (int)((total + block - 1) / block);
  bw24_gguf_nvfp4_deinterleave_kernel<<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint8_t*>(src), row_bytes, static_cast<uint8_t*>(b_packed),
      static_cast<uint8_t*>(sfb_linear), n, k);
  return static_cast<int>(cudaGetLastError());
}
} // extern "C"

extern "C" {
// Scatter weight scales: src linear [n, k/16] ue4m3 -> dst swizzled (size = bw24_cutlass_sfb_size).
int bw24_cutlass_repack_sfb(const void* sfb_linear, void* sfb_swizzled, int n, int k, void* stream) {
  int ksf = k / 16;
  long total = (long)n * ksf;
  int block = 256;
  int grid = (int)((total + block - 1) / block);
  bw24_sf_scatter_kernel<false><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint8_t*>(sfb_linear), static_cast<uint8_t*>(sfb_swizzled), n, k);
  return static_cast<int>(cudaGetLastError());
}
// Scatter activation scales: src linear [m, k/16] ue4m3 -> dst swizzled (size = bw24_cutlass_sfa_size).
int bw24_cutlass_repack_sfa(const void* sfa_linear, void* sfa_swizzled, int m, int k, void* stream) {
  int ksf = k / 16;
  long total = (long)m * ksf;
  int block = 256;
  int grid = (int)((total + block - 1) / block);
  bw24_sf_scatter_kernel<true><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint8_t*>(sfa_linear), static_cast<uint8_t*>(sfa_swizzled), m, k);
  return static_cast<int>(cudaGetLastError());
}
} // extern "C"
