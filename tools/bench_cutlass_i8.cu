// CUTLASS sm90 INT8 GEMM decision probe (W8A8 reopen: vLLM-class int8 rate?) (task #14 reclaim path, ARCHITECTURE-H100.md
// closing verdict): can a FIXED-schedule CUTLASS instantiation replace cuBLASLt as the
// prefill GEMM engine? Two questions, both measured here:
//   1. RATE: >= ~550TF at the m=512 model shapes (Lt measured 611-687) keeps the fp16
//      lane's speed; below that the graph-determinism win costs eager throughput.
//   2. DETERMINISM: bit-identical outputs when operand ADDRESSES shift (the property
//      whose absence in Lt/nvjet blocked prime graphs — alignment-specialized variants).
//
// Build (box): nvcc -std=c++17 -O3 -arch=sm_90a --expt-relaxed-constexpr \
//   -I $CUTLASS/include -I $CUTLASS/tools/util/include -o /tmp/cutf16 tools/bench_cutlass_f16.cu
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1);} } while (0)
#define CT(x) do { cutlass::Status s_ = (x); if (s_ != cutlass::Status::kSuccess) { printf("CUTLASS %d @%d\n", (int)s_, __LINE__); exit(1);} } while (0)

using namespace cute;

// y[m, n] = x[m, k] @ W[n, k]^T : C[M,N] = A[M,K](row) * B[K,N](col, ldb=K == W row-major)
using ElementA = int8_t;
using ElementB = int8_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

#ifndef TILE_M
#define TILE_M 128
#endif
#ifndef TILE_N
#define TILE_N 128
#endif
#ifndef TILE_K
#define TILE_K 64
#endif
#ifndef CLUS_M
#define CLUS_M 1
#endif
#ifndef CLUS_N
#define CLUS_N 2
#endif
using TileShape = Shape<Int<TILE_M>, Int<TILE_N>, Int<TILE_K>>;
using ClusterShape = Shape<Int<CLUS_M>, Int<CLUS_N>, _1>;
#ifdef SCHED_PINGPONG
using KSched = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using ESched = cutlass::epilogue::TmaWarpSpecialized;
#elif defined(SCHED_COOP)
using KSched = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using ESched = cutlass::epilogue::TmaWarpSpecializedCooperative;
#else
using KSched = cutlass::gemm::collective::KernelScheduleAuto;
using ESched = cutlass::epilogue::collective::EpilogueScheduleAuto;
#endif

#ifdef FUSED_EVT
// w8a8 fused epilogue (2026-07-31, round-41 arc): y = acc * act_scale[m] * w_scale[n]
// as an EVT tree — the per-token (row) and per-out-row (col) scales fold into the
// epilogue, deleting the separate dequant_rc launch the Lt route pays.
// No named fusion op covers per-M AND per-N multiplicative scale vectors together
// (ScaledLinComb* = scalar scale_a/scale_b; PerRow/PerColLinComb* = one axis), so the
// tree is hand-built. Axis convention (D[M,N] row-major): Sm90ColBroadcast is the
// column vector Stride<_1,_0,_0> — varies with M = per-token act scale;
// Sm90RowBroadcast is the row vector Stride<_0,_1,_0> — varies with N = per-out-feature
// weight scale.
#include "cutlass/epilogue/fusion/operations.hpp"
using FusionOp = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies, float, float,
                                           cutlass::FloatRoundStyle::round_to_nearest>,
    cutlass::epilogue::fusion::Sm90ColBroadcast<0, TileShape, float>,   // act_scale[m]
    cutlass::epilogue::fusion::Sm90EVT<
        cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies, float, float,
                                               cutlass::FloatRoundStyle::round_to_nearest>,
        cutlass::epilogue::fusion::Sm90RowBroadcast<0, TileShape, float>, // w_scale[n]
        cutlass::epilogue::fusion::Sm90AccFetch>>;
// Epilogue is built FIRST so the mainloop stage count can carve out its smem. With
// plain StageCountAuto the mainloop claimed the full capacity on top of the EVT
// epilogue's real TensorStorage -> SharedStorageSize 296960B > the H100's 227KB
// dynamic-smem cap -> cudaFuncSetAttribute "invalid argument" -> initialize()
// status 7 (kErrorInternal). ElementC=void: the tree never fetches C, so drop the
// C smem tiles entirely (beta path unused anyway).
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    int32_t, float,
    void, LayoutC, 4,
    float, LayoutC, 4,
    ESched, FusionOp>::CollectiveOp;
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, 16,
    ElementB, LayoutB, 16,
    int32_t,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KSched>::CollectiveOp;
#else
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, 16,
    ElementB, LayoutB, 16,
    int32_t,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAuto,
    KSched>::CollectiveOp;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    int32_t, float,
    float, LayoutC, 4,
    float, LayoutC, 4,
    ESched>::CollectiveOp;
#endif

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

static void run_gemm(void* w, void* x, float* y, int m, int n, int k, void* ws,
                     float* act_s = nullptr, float* w_s = nullptr) {
    auto sA = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
    auto sB = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
    auto sC = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k, 1},
        {(ElementA*)x, sA, (ElementB*)w, sB},
        {{}, y, sC, y, sC},
    };
#ifdef FUSED_EVT
    // EVT arg tree mirrors the FusionOp nesting: mul(col_bcast(act_s), mul(row_bcast(w_s), acc))
    args.epilogue.thread = {
        {act_s},                    // Sm90ColBroadcast: per-m activation scale
        {                           // inner EVT
            {w_s},                  // Sm90RowBroadcast: per-n weight scale
            {},                     // Sm90AccFetch
            {}                      // inner compute
        },
        {}                          // outer compute
    };
#else
    args.epilogue.thread = {1.0f, 0.0f};
    (void)act_s; (void)w_s;
#endif
    Gemm gemm;
    CT(gemm.can_implement(args));
    CT(gemm.initialize(args, (uint8_t*)ws));
    CT(gemm.run());
}

struct ShapeRef { int in_f, out_f; float lt_us; const char* tag; };

int main() {
    const int m = getenv("BENCH_M") ? atoi(getenv("BENCH_M")) : 512;
    ShapeRef shapes[] = {
        {4096, 12288, 57.8f, "wqkv"},   // Lt column = cublasGemmEx int8 (the refuted probe)
        {4096, 8192, 40.4f, "mid"},
        {4096, 4096, 26.3f, "square"},
        {11008, 4096, 65.5f, "ffn_down"},
        {4096, 11008, 63.2f, "ffn_gate/up"},
        {4096, 1024, 10.2f, "small"},
    };
    void* ws;
    CK(cudaMalloc(&ws, 64 << 20));
    srand(7);
    for (auto& sh : shapes) {
        int k = sh.in_f, n = sh.out_f;
        // determinism harness: same logical operands at TWO address placements
        size_t wb = (size_t)n * k, xb = (size_t)m * k, yb = (size_t)m * n * 4;
        char *w0, *x0, *w1, *x1;
        float *y0, *y1;
        CK(cudaMalloc(&w0, wb + 4096)); CK(cudaMalloc(&x0, xb + 4096));
        CK(cudaMalloc(&w1, wb + 4096)); CK(cudaMalloc(&x1, xb + 4096));
        CK(cudaMalloc(&y0, yb)); CK(cudaMalloc(&y1, yb));
        {
            int8_t* hw = (int8_t*)malloc(wb);
            int8_t* hx = (int8_t*)malloc(xb);
            for (size_t i = 0; i < (size_t)n * k; i++) hw[i] = (int8_t)(rand() % 255 - 127);
            for (size_t i = 0; i < (size_t)m * k; i++) hx[i] = (int8_t)(rand() % 255 - 127);
            CK(cudaMemcpy(w0, hw, wb, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(x0, hx, xb, cudaMemcpyHostToDevice));
            // placement 2: shifted by 256B (a different alignment class for nvjet-style dispatch)
            CK(cudaMemcpy(w1 + 256, hw, wb, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(x1 + 256, hx, xb, cudaMemcpyHostToDevice));
            free(hw); free(hx);
        }
        // scale vectors for the fused-EVT variant (1.0f = neutral; the fused epilogue's
        // COST is what we measure — value-correctness is pinned by the diff harness)
        float *sc_a, *sc_w;
        CK(cudaMalloc(&sc_a, m * 4)); CK(cudaMalloc(&sc_w, n * 4));
        {
            float* h1f = (float*)malloc((m > n ? m : n) * 4);
            for (int i = 0; i < (m > n ? m : n); i++) h1f[i] = 1.0f;
            CK(cudaMemcpy(sc_a, h1f, m * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(sc_w, h1f, n * 4, cudaMemcpyHostToDevice));
            free(h1f);
        }
        run_gemm(w0, x0, y0, m, n, k, ws, sc_a, sc_w);
        run_gemm(w1 + 256, x1 + 256, y1, m, n, k, ws, sc_a, sc_w);
        CK(cudaDeviceSynchronize());
        float *h0 = (float*)malloc(yb), *h1 = (float*)malloc(yb);
        CK(cudaMemcpy(h0, y0, yb, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h1, y1, yb, cudaMemcpyDeviceToHost));
        size_t nd = 0;
        for (size_t i = 0; i < (size_t)m * n; i++) if (h0[i] != h1[i]) nd++;
        // rate
        cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        for (int i = 0; i < 10; i++) run_gemm(w0, x0, y0, m, n, k, ws, sc_a, sc_w);
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(a));
        for (int i = 0; i < 100; i++) run_gemm(w0, x0, y0, m, n, k, ws, sc_a, sc_w);
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms, a, b));
        double us = ms * 10.0;
        double tf = 2.0 * n * (double)k * m / (us * 1e6);
        printf("%-12s | cutlass %6.1fus (%4.0f TF)  Lt %5.1fus  ratio %.2fx  addr-shift diffs %zu %s\n",
               sh.tag, us, tf, sh.lt_us, sh.lt_us / us, nd, nd == 0 ? "DETERMINISTIC" : "VARIANT");
        cudaFree(w0); cudaFree(x0); cudaFree(w1); cudaFree(x1); cudaFree(y0); cudaFree(y1);
        free(h0); free(h1);
    }
    return 0;
}
