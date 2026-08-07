// mmq_iq_experts.cu — IQ3_S / IQ4_XS expert-segmented int8-MMA MMQ for 35B MoE prefill (sm_120a).
//
// The int8 m16n8k16.s8 tensor-core analog of moe_pairs_matvec_q8_dec (qmatvec.cu). Same CSR
// expert->token grouping (ex_ids/ex_off/ex_pairs/pair_tok) and same device weight-slab pointer
// table (table[proj*n_expert + ex]) as the dp4a _dec kernel, but the per-expert matvec runs as a
// 128x128-tile int8 MMA GEMM over the expert's token group instead of warp-per-row dp4a. Weight
// FP4/IQ nibbles are decoded to int8 AT TILE-LOAD (natural k-order) + a per-32 float scale; the
// activation is q8_1 (block_q8_1_mmq D4, the SAME quant class as the dp4a path). The MMA dot +
// write-back are byte-for-byte the vendored nvfp4_w4a8 machinery (mma.cuh tile<>/ldmatrix/mma).
//
// DECOUPLED like mmq_nvfp4_w4a8.cu (no ggml headers). C-ABI: memra_mmq_iq_experts.
//
// FP-ORDER: MMA is a different reduction than dp4a -> logits SHIFT (like the W4A8 path). Gated on
// run-gen argmax MATCH + run-spec self-consistency + numerical closeness (maxdiff < ~1e-1),
// NOT byte-identity. Standalone proof (iq4xs_mma_test.cu): IQ4_XS maxdiff ~1.5e-3 vs dp4a.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdlib>

#define WARP_SIZE 32
#define GGML_PAD(x,n) (((x)+(n)-1)/(n)*(n))
#define QK_K 256
#define QK8_1 32
#define QI8_1 8
#define MATRIX_ROW_PADDING 512
#define MMQ_TILE_NE_K 32
#define MMQ_ITER_K 256
#define MMQ_MMA_TILE_X_K (2*MMQ_TILE_NE_K + MMQ_TILE_NE_K/2 + 4)   // 84
#define MMQ_TILE_Y_K (MMQ_TILE_NE_K + MMQ_TILE_NE_K/QI8_1)          // 36
#define MMQ_WARP_SIZE 32
#define MMQ_NWARPS 8
#define MMQ_Y 128
#ifndef MMQ_X
#define MMQ_X 128
#endif
#define CUDA_QUANTIZE_BLOCK_SIZE_MMQ 128

// qtype tags (match qmatvec.cu QType).
#define QT_IQ4_XS 5
#define QT_IQ3_S  6
#define QT_Q4_0   12

__device__ __forceinline__ float half_to_float(uint16_t h){ return __half2float(*reinterpret_cast<const __half*>(&h)); }
__constant__ signed char kvalues_iq4nl_d[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};

// IQ3_S grid: 512 u32 (from qmatvec.cu, verbatim ggml-common.h:1042).
__device__ __constant__ unsigned int iq3s_grid_const[512] = {
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
};
__device__ __forceinline__ unsigned int iq3s_grid_d(int idx){ return iq3s_grid_const[idx]; }

struct block_q8_1_mmq { union { float d4[4]; }; int8_t qs[4*QK8_1]; };
static_assert(sizeof(block_q8_1_mmq) == 4*MMQ_TILE_Y_K, "block_q8_1_mmq size");

// ======================= mma.cuh: tile<>, ldmatrix, m16n8k16 s8 mma =======================
namespace ggml_cuda_mma {
    template<int I_,int J_,typename T> struct tile {
        static constexpr int I=I_,J=J_,ne=I*J/32; T x[ne]={0};
        static __device__ __forceinline__ int get_i(int l){
            if constexpr(I==8&&J==4) return threadIdx.x/4;
            else if constexpr(I==8&&J==8) return threadIdx.x/4;
            else if constexpr(I==16&&J==8) return ((l/2)*8)+(threadIdx.x/4);
            else return -1; }
        static __device__ __forceinline__ int get_j(int l){
            if constexpr(I==8&&J==4) return threadIdx.x%4;
            else if constexpr(I==8&&J==8) return (l*4)+(threadIdx.x%4);
            else if constexpr(I==16&&J==8) return ((threadIdx.x%4)*2)+(l%2);
            else return -1; }
    };
    template<int I,int J,typename T> static __device__ __forceinline__ void load_generic(tile<I,J,T>&t,const T* xs0,int stride){
        #pragma unroll
        for(int l=0;l<t.ne;l++) t.x[l]=xs0[t.get_i(l)*stride+t.get_j(l)];
    }
    template<typename T> static __device__ __forceinline__ void load_ldmatrix(tile<16,8,T>&t,const T* xs0,int stride){
        int* xi=(int*)t.x;
        const int* xs=(const int*)xs0 + (threadIdx.x%t.I)*stride + (threadIdx.x/t.I)*(t.J/2);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];"
            :"=r"(xi[0]),"=r"(xi[1]),"=r"(xi[2]),"=r"(xi[3]):"l"(xs));
    }
    // rate-audited 2026-08-06, see research/sm120-empirical-capabilities.md; k16 -> k32 taken
    // 2026-08-07 (lane/iq-experts-k32, e2e-share gate GO: this kernel family holds 16-44% of GPU
    // kernel time on gemma26b/KAT, research/iq-k32-20260807/).
    //   The int8 pipe is K-FREE on sm_120: m16n8k32.s8 costs the SAME 16.06 cyc/warp-MMA as
    //   m16n8k16.s8 for 2x the MACs (309.7 vs 155.2 TOP/s), and ptxas rejects s8 k64 -- k32 is the
    //   deepest form. LEGALITY here (unlike the NVFP4 tile): all three loaders index the 16 x_df
    //   slots through the 32-block id `s>>1` (IQ4_XS via `g`, Q4_0 via `g*18`, IQ3_S via `ib32`),
    //   so the two per-16 scale slots of a 32-block hold the SAME value and one k32 MMA may sum
    //   both 16-halves inside the exact s32 accumulator before the (shared) scale applies.
    //   EXACTNESS CLASS: the s32 merge is exact, but the f32 FOLD changes shape -- k16 computed
    //   dB*(C0*d + C1*d) (two rounded products + add), k32 computes dB*(C32*d) (one rounded
    //   product on the exact int sum C32 = C0+C1; |C| <= 16*127^2 << 2^24 so every int->f32
    //   conversion is exact). The two expressions can differ in the last ulp, so bit-identity vs
    //   the k16 form is NOT guaranteed by construction -- it is MEASURED (A/B byte-compare,
    //   research/iq-k32-20260807/). The kernel's own gate class vs dp4a is unchanged either way
    //   (closeness + argmax + spec, see file header).
    //   MEMRA_IQEXP_K16=1 (build.rs -> MEMRA_IQEXP_K16_MMA) rebuilds the ORIGINAL k16 form.
#ifdef MEMRA_IQEXP_K16_MMA
    static __device__ __forceinline__ void mma(tile<16,8,int>&D,const tile<16,4,int>&A,const tile<8,4,int>&B){
        asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
            :"+r"(D.x[0]),"+r"(D.x[1]),"+r"(D.x[2]),"+r"(D.x[3]):"r"(A.x[0]),"r"(A.x[1]),"r"(B.x[0]));
    }
#else
    static __device__ __forceinline__ void mma(tile<16,8,int>&D,const tile<16,8,int>&A,const tile<8,8,int>&B){
        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            :"+r"(D.x[0]),"+r"(D.x[1]),"+r"(D.x[2]),"+r"(D.x[3])
            :"r"(A.x[0]),"r"(A.x[1]),"r"(A.x[2]),"r"(A.x[3]),"r"(B.x[0]),"r"(B.x[1]));
    }
#endif
}
using namespace ggml_cuda_mma;
static constexpr __device__ int mmq_get_granularity_device(int mmq_x){ return mmq_x>=48?16:8; }

// 16B async global->smem copy (the qmatvec_gemm.cu helper; cp.async.cg native on sm_80+).
// The Y gather was 4B ld.global -> reg -> st.shared per int — every load serialized on its
// register return. cp.async carries no register dependency, so the whole 144B/token gather
// is in flight at once (ncu round 45: long_scoreboard = 66% of stall samples).
__device__ __forceinline__ void cp_async16(void* smem, const void* g){
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0],[%1],16;" :: "r"(s), "l"(g));
}
__device__ __forceinline__ void cp_async4(void* smem, const void* g){
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0],[%1],4;" :: "r"(s), "l"(g));
}
__device__ __forceinline__ void cp_async_commit_wait_all(){
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
}
// staged W row stride: superblock slice padded to 16B (144 covers q4_0's 144 and iq4_xs's 136).
#define W_STAGE_STRIDE 144

// ======================= tile loaders (decode-at-load -> int8 x_qs + per-32 float scale) =======================
// x_qs int-column c (0..63) holds 4 int8 for values [4c..4c+4). Per-32 scale (group c>>3) replicated
// into the 2 per-16 x_df slots of that group. Both IQ4_XS and IQ3_S emit weights in NATURAL k-order
// (matching the q8_1 activation int order + ldmatrix contiguity). Proven vs dp4a in iq4xs_mma_test.cu.
template<int mmq_y, bool nc>
static __device__ __forceinline__ void load_tiles_iq4xs(const uint8_t* __restrict__ W, int* __restrict__ x_tile,
        int kb0, int i_max, long row_bytes){
    int* x_qs = x_tile;
    float* x_df = (float*)(x_qs + MMQ_TILE_NE_K*2);
    for(int i0=0;i0<mmq_y;i0+=MMQ_NWARPS){
        int i = i0 + threadIdx.y; if constexpr(nc) i=min(i,i_max);
        const uint8_t* b = W + (long)i*row_bytes + (long)kb0*136;   // 136B IQ4_XS superblock
        float d_sb = half_to_float(*(const uint16_t*)b);
        uint16_t sh = *(const uint16_t*)(b+2);
        const uint8_t* sl = b+4; const uint8_t* qs = b+8;
        #pragma unroll
        for(int c2=0;c2<2;c2++){
            int c = threadIdx.x + c2*32; int g = c>>3; int lc = c - g*8;
            const uint8_t* gqs = qs + g*16;
            int w0,w1,w2,w3;
            #pragma unroll
            for(int r=0;r<4;r++){ int v=lc*4+r; int code=(v<16)?(gqs[v]&0xf):(gqs[v-16]>>4);
                int wv=kvalues_iq4nl_d[code]; if(r==0)w0=wv;else if(r==1)w1=wv;else if(r==2)w2=wv;else w3=wv; }
            x_qs[i*MMQ_MMA_TILE_X_K + c] = (w0&0xff)|((w1&0xff)<<8)|((w2&0xff)<<16)|((w3&0xff)<<24);
        }
        if(threadIdx.x<16){ int s=threadIdx.x; int g=s>>1;
            int ls = ((sl[g>>1]>>(4*(g&1)))&0xf) | (((sh>>(2*g))&3)<<4);
            x_df[i*MMQ_MMA_TILE_X_K + s] = d_sb * (float)(ls-32);
        }
    }
}

// Q4_0 (gemma4 QAT): 8 consecutive 18B/32-val blocks per 256-val superblock (144B). int weight
// = (nib - 8) folded via __vsub4 (same integer identity as expert_decode_q4_0_g); per-16 scale
// slots both map to the owning 32-block's fp16 d.
template<int mmq_y, bool nc>
static __device__ __forceinline__ void load_tiles_q4_0(const uint8_t* __restrict__ W, int* __restrict__ x_tile,
        int kb0, int i_max, long row_bytes){
    int* x_qs = x_tile;
    float* x_df = (float*)(x_qs + MMQ_TILE_NE_K*2);
    for(int i0=0;i0<mmq_y;i0+=MMQ_NWARPS){
        int i = i0 + threadIdx.y; if constexpr(nc) i=min(i,i_max);
        const uint8_t* sb = W + (long)i*row_bytes + (long)kb0*144;   // 8 x 18B Q4_0 blocks
        #pragma unroll
        for(int c2=0;c2<2;c2++){
            int c = threadIdx.x + c2*32;          // int-col 0..63 = vals 4c..4c+3
            int g = c >> 3;                        // 32-val block within the superblock
            int v0 = (c - g*8) * 4;                // first val within the block (0,4,..,28)
            const uint8_t* qs = sb + g*18 + 2;
            int w;
            if (v0 < 16) {
                // lo nibbles of qs[v0..v0+3]
                uint32_t raw; memcpy(&raw, qs + v0, 4);
                w = __vsub4((int)(raw & 0x0F0F0F0Fu), 0x08080808);
            } else {
                uint32_t raw; memcpy(&raw, qs + (v0 - 16), 4);
                w = __vsub4((int)((raw >> 4) & 0x0F0F0F0Fu), 0x08080808);
            }
            x_qs[i*MMQ_MMA_TILE_X_K + c] = w;
        }
        if(threadIdx.x<16){ int s=threadIdx.x; int g=s>>1;
            x_df[i*MMQ_MMA_TILE_X_K + s] = half_to_float(*(const uint16_t*)(sb + g*18));
        }
    }
}

// IQ3_S: 110B block / 256 vals: d(2) qs[64](2..66) qh[8](66..74) signs[32](74..106) scales[4](106..110).
// Decode uses the iq3s_grid + sign trick (expert_decode_iq3s_g). db = d*(1+2*sc_nib), applied as the
// per-32 float scale (iscale==1 there). Natural k-order: group g (0..7) = 32 values = 8 int-cols.
template<int mmq_y, bool nc>
static __device__ __forceinline__ void load_tiles_iq3s(const uint8_t* __restrict__ W, int* __restrict__ x_tile,
        int kb0, int i_max, long row_bytes){
    int* x_qs = x_tile;
    float* x_df = (float*)(x_qs + MMQ_TILE_NE_K*2);
    for(int i0=0;i0<mmq_y;i0+=MMQ_NWARPS){
        int i = i0 + threadIdx.y; if constexpr(nc) i=min(i,i_max);
        const uint8_t* b = W + (long)i*row_bytes + (long)kb0*110;
        float d = half_to_float(*(const uint16_t*)b);
        const uint8_t* qs_all = b + 2;
        const uint8_t* qh_all = b + 66;
        const uint8_t* signs_all = b + 74;
        const uint8_t* scales = b + 106;
        // 8 groups (ib32) x 8 int-cols each = 64 int-cols. lane loads 2 int-cols.
        #pragma unroll
        for(int c2=0;c2<2;c2++){
            int c = threadIdx.x + c2*32; int ib32 = c>>3; int lc = c - ib32*8;  // 0..7 int-col in group
            // expert_decode_iq3s_g packs wq[l0..l0+1] for l0 in {0,2,4,6}: wq[l0]=grid_l, wq[l0+1]=grid_h.
            // int-col lc maps to wq[lc]. l0 = lc & ~1; is this col grid_l (even) or grid_h (odd)?
            const uint8_t* qs = qs_all + ib32*8;
            unsigned char qh = qh_all[ib32];
            const uint8_t* signs = signs_all + ib32*4;
            int l0 = lc & ~1;         // 0,0,2,2,4,4,6,6
            int gl = iq3s_grid_d(qs[l0+0] | (((int)qh << (8-l0)) & 0x100));
            int gh = iq3s_grid_d(qs[l0+1] | (((int)qh << (7-l0)) & 0x100));
            unsigned char sb = signs[l0/2];
            int signs0 = __vcmpne4(((sb&0x03)<<7)|((sb&0x0C)<<21), 0);
            int signs1 = __vcmpne4(((sb&0x30)<<3)|((sb&0xC0)<<17), 0);
            int grid_l = __vsub4(gl ^ signs0, signs0);
            int grid_h = __vsub4(gh ^ signs1, signs1);
            x_qs[i*MMQ_MMA_TILE_X_K + c] = (lc&1) ? grid_h : grid_l;
        }
        if(threadIdx.x<16){ int s=threadIdx.x; int ib32=s>>1;
            int sc_nib = (ib32&1) ? (scales[ib32/2]>>4) : (scales[ib32/2]&0xf);
            x_df[i*MMQ_MMA_TILE_X_K + s] = d * (1.0f + 2.0f*(float)sc_nib);
        }
    }
}

// ======================= vec_dot (nvfp4_w4a8 machinery, verbatim) =======================
// jexit (swapab-probe residue, 2026-08-01): ragged token tiles skip j0 strips whose 8-token
// block lies entirely past j_max. Each warp's strips stride the whole token axis (j0 +
// (ty%ntx)*8), so the skip is load-balanced at 8-token granularity — ceil8(65)/128 = 0.5625
// of the mma work at q35 gate/up's ~65-pair groups. The skipped strips' sum slots are exactly
// the writeback's discarded j>j_max columns, so live output slots are bit-identical. Dispatch
// at the call site keeps full tiles (cnt == mmq_x) on the jexit=false instantiation — the
// guard folds away and full-tile codegen is unchanged (the naked runtime check cost +2.1% at
// m=128 in the swapab microbench; research/swapab-20260801/).
#ifdef MEMRA_IQEXP_K16_MMA
// ORIGINAL k16 form (rollback build: MEMRA_IQEXP_K16=1). Two m16n8k16 MMAs per 32-value k step,
// fold applies the two (equal) per-16 dA slots separately: dB*(C0*dA0 + C1*dA1).
template<int mmq_x, int mmq_y, bool jexit>
static __device__ __forceinline__ void vec_dot_mma(const int* x, const int* y, float* sum, int k00, int j_max){
    typedef tile<16,4,int> tA; typedef tile<16,8,int> tA8; typedef tile<8,4,int> tB; typedef tile<16,8,int> tC;
    constexpr int g=mmq_get_granularity_device(mmq_x); constexpr int rpw=2*g; constexpr int ntx=rpw/tC::I;
    const int joff = (threadIdx.y%ntx)*tC::J;
    y += joff*MMQ_TILE_Y_K;
    const int* x_qs=x; const float* x_df=(const float*)x_qs + MMQ_TILE_NE_K*2;
    const int* y_qs=(const int*)y+4; const float* y_df=(const float*)y;
    const int i0=(threadIdx.y/ntx)*(ntx*tA::I);
    tA A[ntx][8]; float dA[ntx][tC::ne/2][8];
    #pragma unroll
    for(int n=0;n<ntx;n++){
        #pragma unroll
        for(int k01=0;k01<MMQ_TILE_NE_K;k01+=8)
            load_ldmatrix(((tA8*)A[n])[k01/8], x_qs+(i0+n*tA::I)*MMQ_MMA_TILE_X_K+(k00+k01), MMQ_MMA_TILE_X_K);
        #pragma unroll
        for(int l=0;l<tC::ne/2;l++){
            int i=i0+n*tC::I+tC::get_i(2*l);
            #pragma unroll
            for(int k01=0;k01<MMQ_TILE_NE_K;k01+=4) dA[n][l][k01/4]=x_df[i*MMQ_MMA_TILE_X_K+(k00+k01)/4];
        }
    }
    #pragma unroll
    for(int j0=0;j0<mmq_x;j0+=ntx*tC::J){
        if(jexit && j0+joff > j_max) continue;   // strip fully past the tile's live tokens
        #pragma unroll
        for(int k01=0;k01<MMQ_TILE_NE_K;k01+=8){
            tB B[2]; float dB[tC::ne/2];
            load_generic(B[0], y_qs+j0*MMQ_TILE_Y_K+(k01+0), MMQ_TILE_Y_K);
            load_generic(B[1], y_qs+j0*MMQ_TILE_Y_K+(k01+tB::J), MMQ_TILE_Y_K);
            #pragma unroll
            for(int l=0;l<tC::ne/2;l++){ int j=j0+tC::get_j(l); dB[l]=y_df[j*MMQ_TILE_Y_K+k01/QI8_1]; }
            #pragma unroll
            for(int n=0;n<ntx;n++){
                tC C[2];
                mma(C[0],A[n][k01/4+0],B[0]);
                mma(C[1],A[n][k01/4+1],B[1]);
                #pragma unroll
                for(int l=0;l<tC::ne;l++)
                    sum[(j0/tC::J+n)*tC::ne+l] += dB[l%2]*(C[0].x[l]*dA[n][l/2][k01/4+0]+C[1].x[l]*dA[n][l/2][k01/4+1]);
            }
        }
    }
}
#else
// k32 form (default since lane/iq-experts-k32, 2026-08-07): ONE m16n8k32 MMA per 32-value k step
// where the k16 form issued two m16n8k16 — the int8 pipe is K-FREE on sm_120a (both forms 16.06
// cyc/warp-MMA), so this halves the MMA issue slots AND halves the fold arity (1 C tile, 1 dA
// load, 1 FMA per element vs 2/2/2). LEGAL because both per-16 x_df slots of a 32-block hold the
// same value (all three loaders index scales by the 32-block id s>>1) and s32 accumulation is
// exact; the fold's f32 rounding shape changes (see the mma() comment), so exactness vs k16 is
// measured, not assumed. Same ldmatrix loads (tA8 was already loaded per-8-ints); the B pair
// becomes one tile<8,8> load_generic over the same 8 ints.
template<int mmq_x, int mmq_y, bool jexit>
static __device__ __forceinline__ void vec_dot_mma(const int* x, const int* y, float* sum, int k00, int j_max){
    typedef tile<16,8,int> tA; typedef tile<8,8,int> tB; typedef tile<16,8,int> tC;
    constexpr int g=mmq_get_granularity_device(mmq_x); constexpr int rpw=2*g; constexpr int ntx=rpw/tC::I;
    const int joff = (threadIdx.y%ntx)*tC::J;
    y += joff*MMQ_TILE_Y_K;
    const int* x_qs=x; const float* x_df=(const float*)x_qs + MMQ_TILE_NE_K*2;
    const int* y_qs=(const int*)y+4; const float* y_df=(const float*)y;
    const int i0=(threadIdx.y/ntx)*(ntx*tA::I);
    tA A[ntx][MMQ_TILE_NE_K/8]; float dA[ntx][tC::ne/2][MMQ_TILE_NE_K/8];
    #pragma unroll
    for(int n=0;n<ntx;n++){
        #pragma unroll
        for(int k01=0;k01<MMQ_TILE_NE_K;k01+=8)
            load_ldmatrix(A[n][k01/8], x_qs+(i0+n*tA::I)*MMQ_MMA_TILE_X_K+(k00+k01), MMQ_MMA_TILE_X_K);
        #pragma unroll
        for(int l=0;l<tC::ne/2;l++){
            int i=i0+n*tC::I+tC::get_i(2*l);
            // per-32 weight scale: x_df slots 2m and 2m+1 are equal (loader invariant) — read
            // the even slot once per 8-int (32-value) k step.
            #pragma unroll
            for(int k01=0;k01<MMQ_TILE_NE_K;k01+=8) dA[n][l][k01/8]=x_df[i*MMQ_MMA_TILE_X_K+(k00+k01)/4];
        }
    }
    #pragma unroll
    for(int j0=0;j0<mmq_x;j0+=ntx*tC::J){
        if(jexit && j0+joff > j_max) continue;   // strip fully past the tile's live tokens
        #pragma unroll
        for(int k01=0;k01<MMQ_TILE_NE_K;k01+=8){
            tB B; float dB[tC::ne/2];
            load_generic(B, y_qs+j0*MMQ_TILE_Y_K+k01, MMQ_TILE_Y_K);
            #pragma unroll
            for(int l=0;l<tC::ne/2;l++){ int j=j0+tC::get_j(l); dB[l]=y_df[j*MMQ_TILE_Y_K+k01/QI8_1]; }
            #pragma unroll
            for(int n=0;n<ntx;n++){
                tC C;
                mma(C,A[n][k01/8],B);
                #pragma unroll
                for(int l=0;l<tC::ne;l++)
                    sum[(j0/tC::J+n)*tC::ne+l] += dB[l%2]*C.x[l]*dA[n][l/2][k01/8];
            }
        }
    }
}
#endif

// ======================= expert-segmented MMQ kernel =======================
// grid.x = out-row tile (N/128); grid.y = active-expert segment. One block walks the expert's token
// group (ex_off[seg]..ex_off[seg+1]) in 128-token tiles. Activation is GATHERED per token via
// pair_tok[ex_pairs[base+j]] from the pre-quantized token-major q8_1_mmq buffer. Output row = the
// pair id (pair-major y, [n_pairs, out_f]) — matches moe_pairs_matvec_q8_dec's y layout.
// minblocks=2 (round 45, the kernel-rate dig): minblocks=1 let ptxas take 255 regs/thread —
// Block Limit Registers 1, occupancy 12.5%, SM 13-20% / DRAM 3% (pure latency underfill,
// ncu 2026-07-31). Two CTAs/SM caps regs at 128; grid supplies 546-2002 CTAs on 132 SMs.
template<int mmq_x, bool nc>
__global__ void __launch_bounds__(MMQ_WARP_SIZE*MMQ_NWARPS,1)
mmq_iq_experts_kernel(
        const unsigned long long* __restrict__ table, int proj, int n_expert,
        const int* __restrict__ ex_ids, const int* __restrict__ ex_off, const int* __restrict__ ex_pairs,
        const int* __restrict__ pair_tok,
        const int* __restrict__ Yq,               // pre-quantized q8_1_mmq, block-k-major token-minor
        float* __restrict__ y,                     // [n_pairs, out_f]
        int in_f, int out_f, int n_active, int qtype, long row_bytes, int n_tokens){
    constexpr int mmq_y = MMQ_Y;
    int seg = blockIdx.y; if(seg>=n_active) return;
    int it = blockIdx.x;                            // out-row tile
    int ex = ex_ids[seg];
    int lo = ex_off[seg], hi = ex_off[seg+1];
    const uint8_t* W = (const uint8_t*)table[(size_t)proj*n_expert + ex] + (long)it*mmq_y*row_bytes;
    int i_max = out_f - it*mmq_y - 1;
    int nsblk = in_f/256;
    constexpr int sz = sizeof(block_q8_1_mmq)/sizeof(int);   // 36

    extern __shared__ int smem[];
    int* ids = smem;                                // mmq_x pair ids for this token-tile
    int* tile_y = smem + mmq_x;                     // 2 half-buffers (ping-pong)
    int* tile_x = tile_y + 2*GGML_PAD(mmq_x*MMQ_TILE_Y_K, MMQ_NWARPS*MMQ_WARP_SIZE);
    uint8_t* w_stage = (uint8_t*)(tile_x + mmq_y*MMQ_MMA_TILE_X_K);   // 2 x mmq_y x 144B ring

    // W kb-slice STAGING (round 45 increment 2): the raw superblock slice cp.asyncs into a
    // smem ring one kb AHEAD; dequant then reads RESIDENT smem (the qmatvec_gemm FIX-A
    // pattern — the long-scoreboard global weight read leaves the dequant chain). 16B
    // chunks when every (row, kb) offset stays 16B-aligned (q4_0's 144B slices on 16B rows),
    // 4B otherwise (iq4_xs 136B slices; q4_0 down's 396B rows). IQ3_S (110B, non-4B-multiple)
    // keeps the direct global loader. nc row-clamp happens at stage time.
    const int sb_bytes = (qtype==QT_Q4_0) ? 144 : (qtype==QT_IQ4_XS) ? 136 : 0;
    const bool w16 = qtype==QT_Q4_0 && (row_bytes % 16 == 0);
    auto stage_w = [&](int kb, int buf){
        uint8_t* dst0 = w_stage + (size_t)buf*mmq_y*W_STAGE_STRIDE;
        if(w16){
            const int npr = sb_bytes/16;
            const int tot = mmq_y*npr;
            for(int c0=0;c0<tot;c0+=MMQ_NWARPS*MMQ_WARP_SIZE){
                int c=c0+threadIdx.y*MMQ_WARP_SIZE+threadIdx.x; if(c>=tot) break;
                int r=c/npr, k=c%npr;
                int rr = nc ? min(r,i_max) : r;
                cp_async16(dst0+(size_t)r*W_STAGE_STRIDE+k*16,
                           W+(long)rr*row_bytes+(long)kb*sb_bytes+k*16);
            }
        } else {
            const int npr = sb_bytes/4;
            const int tot = mmq_y*npr;
            for(int c0=0;c0<tot;c0+=MMQ_NWARPS*MMQ_WARP_SIZE){
                int c=c0+threadIdx.y*MMQ_WARP_SIZE+threadIdx.x; if(c>=tot) break;
                int r=c/npr, k=c%npr;
                int rr = nc ? min(r,i_max) : r;
                cp_async4(dst0+(size_t)r*W_STAGE_STRIDE+k*4,
                          W+(long)rr*row_bytes+(long)kb*sb_bytes+k*4);
            }
        }
        asm volatile("cp.async.commit_group;");
    };
    const bool staged = sb_bytes != 0;

    for(int base=lo; base<hi; base+=mmq_x){
        int cnt = min(mmq_x, hi-base);
        int j_max = cnt - 1;
        // publish this tile's output pair ids (and gather-source token per column).
        for(int j0=0;j0<mmq_x;j0+=MMQ_NWARPS*MMQ_WARP_SIZE){
            int j=j0+threadIdx.y*MMQ_WARP_SIZE+threadIdx.x;
            if(j<mmq_x) ids[j] = (j<cnt) ? ex_pairs[base+j] : ex_pairs[base]; // clamp OOB to a valid row
        }
        __syncthreads();
        float sum[mmq_x*mmq_y/(MMQ_NWARPS*MMQ_WARP_SIZE)] = {0.0f};
        // full tiles take the jexit=false instantiation (guard folds away, codegen unchanged);
        // ragged tiles skip dead 8-token j0 strips (live sum slots bit-identical — the skipped
        // slots are the writeback's discarded j>j_max columns).
        auto vdot = [&](const int* ty, int k00){
            if(cnt == mmq_x) vec_dot_mma<mmq_x,mmq_y,false>(tile_x, ty, sum, k00, j_max);
            else             vec_dot_mma<mmq_x,mmq_y,true >(tile_x, ty, sum, k00, j_max);
        };
        if(staged) stage_w(0, 0);
        for(int kb=0;kb<nsblk;kb++){
            if(staged){
                asm volatile("cp.async.wait_group 0;");
                __syncthreads();                       // W[kb&1] resident for every warp
                const uint8_t* Ws = w_stage + (size_t)(kb&1)*mmq_y*W_STAGE_STRIDE;
                // rows were clamped at stage time -> loaders run the nc=false form.
                if(qtype==QT_Q4_0) load_tiles_q4_0 <mmq_y,false>(Ws, tile_x, 0, i_max, W_STAGE_STRIDE);
                else               load_tiles_iq4xs<mmq_y,false>(Ws, tile_x, 0, i_max, W_STAGE_STRIDE);
                if(kb+1<nsblk) stage_w(kb+1, (kb+1)&1);   // in flight behind the halves' mma
            }
            else                      load_tiles_iq3s <mmq_y,nc>(W, tile_x, kb, i_max, row_bytes);
            // Y GATHER PING-PONG (round 45 increment 3): both halves' gathers issue as
            // separate cp.async groups at kb top; groups complete IN ORDER, so wait_group 1
            // leaves h1's gather in flight behind h0's mma. 16B chunks (9/token: sz=36 ints
            // = 144B, 16B-aligned both sides) — no register dependency per load.
            constexpr int CH = 4;                                      // ints per 16B chunk
            const int ybuf_stride = GGML_PAD(mmq_x*MMQ_TILE_Y_K, MMQ_NWARPS*MMQ_WARP_SIZE);
            #pragma unroll
            for(int half=0;half<2;half++){
                int blockk = kb*2 + half;              // 128-value chunk index (block_q8_1_mmq)
                int* ty = tile_y + half*ybuf_stride;
                for(int c0=0;c0<mmq_x*(sz/CH);c0+=MMQ_NWARPS*MMQ_WARP_SIZE){
                    int c=c0+threadIdx.y*MMQ_WARP_SIZE+threadIdx.x;
                    if(c>=mmq_x*(sz/CH)) break;
                    int token_c = c / (sz/CH), ii = (c % (sz/CH))*CH;
                    // clamped tail columns (token_c > j_max) are DISCARDED at write-back —
                    // skip their gather entirely (q35 gate/up groups average ~65 pairs:
                    // ~half of every 128-tile was duplicate gather traffic, round 46 inc4).
                    // Their mma consumes stale smem; the products never leave the registers.
                    if(token_c > j_max) continue;
                    int src_tok = pair_tok[ids[token_c]];
                    cp_async16(&ty[token_c*sz+ii],
                               &Yq[((size_t)blockk*n_tokens + src_tok)*sz + ii]);
                }
                asm volatile("cp.async.commit_group;");
            }
            // pending groups (issue order): [W(kb+1) if staged], Y(h0), Y(h1).
            asm volatile("cp.async.wait_group 1;");    // W(kb+1)+Y(h0) done; Y(h1) in flight
            __syncthreads();
            vdot(tile_y, 0*MMQ_TILE_NE_K);
            asm volatile("cp.async.wait_group 0;");    // Y(h1) done (overlapped the h0 mma)
            __syncthreads();
            vdot(tile_y + ybuf_stride, 1*MMQ_TILE_NE_K);
            __syncthreads();
        }
        // write-back (nvfp4_w4a8 machinery): row = ids[j] (pair id), col = it*mmq_y + i.
        {
            typedef tile<16,8,int> tC;
            constexpr int g=mmq_get_granularity_device(mmq_x); constexpr int rpw=2*g; constexpr int ntx=rpw/tC::I;
            int i0=(threadIdx.y/ntx)*(ntx*tC::I);
            #pragma unroll
            for(int j0=0;j0<mmq_x;j0+=ntx*tC::J){
                #pragma unroll
                for(int n=0;n<ntx;n++){
                    #pragma unroll
                    for(int l=0;l<tC::ne;l++){
                        int j=j0+(threadIdx.y%ntx)*tC::J+tC::get_j(l); if(j>j_max) continue;
                        int i=i0+n*tC::I+tC::get_i(l); if(nc&&i>i_max) continue;
                        y[(size_t)ids[j]*out_f + (it*mmq_y+i)] = sum[(j0/tC::J+n)*tC::ne+l];
                    }
                }
            }
        }
        __syncthreads();
    }
}

// ======================= activation quantizer (D4, token-major) =======================
static __global__ void quantize_mmq_q8_1_d4_kernel(const float* __restrict__ x, void* __restrict__ vy,
        int64_t ne00, int64_t s01, int64_t ne0, int ne1){
    int64_t i0 = ((int64_t)blockDim.x*blockIdx.y + threadIdx.x)*4;
    if(i0>=ne0) return;
    int64_t i1=blockIdx.x;
    const float4* x4=(const float4*)x; block_q8_1_mmq* yb=(block_q8_1_mmq*)vy;
    int64_t ib=(i0/(4*QK8_1))*ne1 + blockIdx.x;
    int64_t iqs=i0%(4*QK8_1);
    float4 xi = i0<ne00 ? x4[(i1*s01+i0)/4] : make_float4(0,0,0,0);
    float amax=fabsf(xi.x); amax=fmaxf(amax,fabsf(xi.y)); amax=fmaxf(amax,fabsf(xi.z)); amax=fmaxf(amax,fabsf(xi.w));
    #pragma unroll
    for(int off=32/8;off>0;off>>=1) amax=fmaxf(amax,__shfl_xor_sync(0xffffffff,amax,off,WARP_SIZE));
    float di=127.0f/amax; char4 q; q.x=roundf(xi.x*di);q.y=roundf(xi.y*di);q.z=roundf(xi.z*di);q.w=roundf(xi.w*di);
    ((char4*)yb[ib].qs)[iqs/4]=q;
    if(iqs%32!=0) return;
    yb[ib].d4[iqs/32] = amax==0.0f?0.0f:1.0f/di;
}

// ======================= fused activation + quantize epilogue =======================
// silu/gelu(gate)*up + D4 quantize in ONE launch (fused act-epilogue lever): the two-pass
// chain (moe_pairs_silu_mul / moe_pairs_gelu_mul writes act f32 to HBM; the quantizer above
// re-reads it) costs two full passes over the [n_pairs x n_ff] f32 activation. Here the
// activation is computed in registers from gate/up and ONLY the quantized scratch is written
// (read gate+up, write scratch — the f32 act buffer never exists).
// BIT-IDENTITY CONTRACT: the activation expressions are moe_pairs_silu_mul /
// moe_pairs_gelu_mul (qmatvec.cu) VERBATIM and the quantize fold is
// quantize_mmq_q8_1_d4_kernel's exact per-thread-amax + 8-lane shfl_xor pattern — the
// scratch bytes must equal the two-pass path's (kernel-check `iq fused act+quant` gates
// byte-for-byte, both activations, aligned + ragged/padded in_f).
template <int ACT>   // 0 = silu*mul (qwen35moe), 1 = gelu_tanh*mul (gemma4)
static __global__ void fused_act_quant_mmq_q8_1_d4_kernel(
        const float* __restrict__ gate, const float* __restrict__ up, void* __restrict__ vy,
        int64_t ne00, int64_t s01, int64_t ne0, int ne1){
    int64_t i0 = ((int64_t)blockDim.x*blockIdx.y + threadIdx.x)*4;
    if(i0>=ne0) return;
    int64_t i1=blockIdx.x;
    block_q8_1_mmq* yb=(block_q8_1_mmq*)vy;
    int64_t ib=(i0/(4*QK8_1))*ne1 + blockIdx.x;
    int64_t iqs=i0%(4*QK8_1);
    float4 xi = make_float4(0,0,0,0);   // i0>=ne00: zero padding, same as the two-pass quantizer
    if(i0<ne00){
        const float4 g4 = *(const float4*)(gate + i1*s01 + i0);
        const float4 u4 = *(const float4*)(up   + i1*s01 + i0);
        const float gg[4] = {g4.x,g4.y,g4.z,g4.w};
        const float uu[4] = {u4.x,u4.y,u4.z,u4.w};
        float aa[4];
        #pragma unroll
        for(int j=0;j<4;j++){
            const float g = gg[j];
            if(ACT==0){
                aa[j] = (g / (1.0f + expf(-g))) * uu[j];           // moe_pairs_silu_mul verbatim
            } else {                                                // moe_pairs_gelu_mul verbatim
                float th = tanhf(0.79788456080286535587989211986876f * g * (1.0f + 0.044715f * g * g));
                aa[j] = 0.5f * g * (1.0f + th) * uu[j];
            }
        }
        xi = make_float4(aa[0],aa[1],aa[2],aa[3]);
    }
    float amax=fabsf(xi.x); amax=fmaxf(amax,fabsf(xi.y)); amax=fmaxf(amax,fabsf(xi.z)); amax=fmaxf(amax,fabsf(xi.w));
    #pragma unroll
    for(int off=32/8;off>0;off>>=1) amax=fmaxf(amax,__shfl_xor_sync(0xffffffff,amax,off,WARP_SIZE));
    float di=127.0f/amax; char4 q; q.x=roundf(xi.x*di);q.y=roundf(xi.y*di);q.z=roundf(xi.z*di);q.w=roundf(xi.w*di);
    ((char4*)yb[ib].qs)[iqs/4]=q;
    if(iqs%32!=0) return;
    yb[ib].d4[iqs/32] = amax==0.0f?0.0f:1.0f/di;
}

// ======================= C-ABI launchers =======================
extern "C" {

size_t memra_mmq_iq_experts_act_bytes(int in_f, int n_tokens){
    int64_t nep = GGML_PAD((int64_t)in_f, MATRIX_ROW_PADDING);
    int64_t nblk = (int64_t)n_tokens * (nep/(4*QK8_1));
    return (size_t)nblk * sizeof(block_q8_1_mmq);
}

// Quantize token-major f32 activation [n_tokens, in_f] -> block_q8_1_mmq scratch (D4).
int memra_mmq_iq_quantize_act(const float* act_f32, void* act_scratch, int in_f, int n_tokens, void* stream){
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    int64_t ne10 = in_f, nep = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    int64_t bny = (nep + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1)/(4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    dim3 blk(CUDA_QUANTIZE_BLOCK_SIZE_MMQ,1,1), grid((unsigned)n_tokens,(unsigned)bny,1);
    quantize_mmq_q8_1_d4_kernel<<<grid,blk,0,st>>>(act_f32, act_scratch, ne10, in_f, nep, n_tokens);
    cudaError_t e=cudaGetLastError(); return e?1000+(int)e:0;
}

// Fused act-epilogue: silu/gelu(gate)*up + D4 quantize, one launch, no f32 act buffer.
// gate/up are pair-major [n_tokens, in_f] f32 (the mmq_iq_experts gate/up outputs);
// scratch layout/bytes identical to memra_mmq_iq_quantize_act (memra_mmq_iq_experts_act_bytes).
// act_kind: 0 = silu*mul (qwen35moe), 1 = gelu_tanh*mul (gemma4). Byte-identical to the
// two-pass path by contract (see kernel comment). Returns 0 or 1000+cudaError.
int memra_mmq_iq_fused_act_quant(const float* gate, const float* up, void* act_scratch,
        int in_f, int n_tokens, int act_kind, void* stream){
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    int64_t ne10 = in_f, nep = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    int64_t bny = (nep + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1)/(4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    dim3 blk(CUDA_QUANTIZE_BLOCK_SIZE_MMQ,1,1), grid((unsigned)n_tokens,(unsigned)bny,1);
    if(act_kind==0)
        fused_act_quant_mmq_q8_1_d4_kernel<0><<<grid,blk,0,st>>>(gate, up, act_scratch, ne10, in_f, nep, n_tokens);
    else
        fused_act_quant_mmq_q8_1_d4_kernel<1><<<grid,blk,0,st>>>(gate, up, act_scratch, ne10, in_f, nep, n_tokens);
    cudaError_t e=cudaGetLastError(); return e?1000+(int)e:0;
}

// Expert-segmented IQ MMA MMQ. y[n_pairs, out_f]. `act_scratch` pre-quantized via
// memra_mmq_iq_quantize_act (token-major over n_tokens). qtype: 5=IQ4_XS, 6=IQ3_S.
} // extern "C" (paused: the tile-width launch helper below is a template — no C linkage)

// Dynamic-smem bytes for a given token-tile width (the mmq_x-dependent pieces: pair ids +
// the round-45/46 Y ping-pong; tile_x + the W staging ring are mmq_y-fixed).
static size_t mmq_iq_experts_smem(int mmq_x){
    return (size_t)mmq_x*sizeof(int)
        + 2*GGML_PAD((size_t)mmq_x*MMQ_TILE_Y_K, MMQ_NWARPS*MMQ_WARP_SIZE)*sizeof(int)  // Y ping-pong
        + (size_t)MMQ_Y*MMQ_MMA_TILE_X_K*sizeof(int)
        + 2*(size_t)MMQ_Y*W_STAGE_STRIDE;              // W kb-slice staging ring
}

template<int MX>
static int mmq_iq_experts_launch(const unsigned long long* table, int proj, int n_expert,
        const int* ex_ids, const int* ex_off, const int* ex_pairs, const int* pair_tok,
        const int* Yq, float* y, int in_f, int out_f, int n_active, int n_tokens,
        int qtype, long row_bytes, cudaStream_t st){
    const int nty = (out_f + MMQ_Y - 1)/MMQ_Y;
    dim3 grid((unsigned)nty, (unsigned)n_active, 1);
    dim3 block(MMQ_WARP_SIZE, MMQ_NWARPS, 1);
    size_t smem = mmq_iq_experts_smem(MX);
    const bool nc = (out_f % MMQ_Y) != 0;
    cudaError_t ae;
    if(nc){
        ae = cudaFuncSetAttribute(mmq_iq_experts_kernel<MX,true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if(ae != cudaSuccess) return 2000+(int)ae;
        mmq_iq_experts_kernel<MX,true><<<grid,block,smem,st>>>(table,proj,n_expert,ex_ids,ex_off,ex_pairs,pair_tok,Yq,y,in_f,out_f,n_active,qtype,row_bytes,n_tokens);
    } else {
        ae = cudaFuncSetAttribute(mmq_iq_experts_kernel<MX,false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if(ae != cudaSuccess) return 2000+(int)ae;
        mmq_iq_experts_kernel<MX,false><<<grid,block,smem,st>>>(table,proj,n_expert,ex_ids,ex_off,ex_pairs,pair_tok,Yq,y,in_f,out_f,n_active,qtype,row_bytes,n_tokens);
    }
    cudaError_t e=cudaGetLastError(); return e?1000+(int)e:0;
}

// ======================= IQ4_XS DENSE-TRUNK MMQ (lane/kquant-tile-loaders) =======================
// The dense analog of the expert kernel above, for NON-expert IQ4_XS 2-D matmuls (the KAT-Coder
// trunk class: attn_qkv/attn_gate/ssm_out/attn_q/shexp — kat-anomaly §6). Same load_tiles_iq4xs
// decode-at-tile-load + vec_dot_mma int8 MMA + W kb-slice cp.async staging ring; conventional
// xy-tiling (grid = (out tiles, token tiles)), token-major D4 q8_1 activation, y[token][out_f].
// FP-ORDER: MMA reduction != the dp4a per-column program -> logits SHIFT; gated on kernel-check
// closeness vs dp4a + run-gen argmax + run-spec (m>=16 only — decode/verify keep dp4a, dispatch
// parity preserved by the m-threshold at the call site).
template<int mmq_x, bool nc>
__global__ void __launch_bounds__(MMQ_WARP_SIZE*MMQ_NWARPS,1)
mmq_iq4xs_dense_kernel(
        const uint8_t* __restrict__ W, const int* __restrict__ Yq, float* __restrict__ y,
        int in_f, int out_f, int n_tokens, long row_bytes){
    constexpr int mmq_y = MMQ_Y;
    const int it = blockIdx.x;                 // out-row tile
    const int jt = blockIdx.y;                 // token tile
    const uint8_t* Wt = W + (long)it*mmq_y*row_bytes;
    const int i_max = out_f - it*mmq_y - 1;
    const int base = jt*mmq_x;
    const int cnt = min(mmq_x, n_tokens - base);
    const int j_max = cnt - 1;
    const int nsblk = in_f/256;
    constexpr int sz = sizeof(block_q8_1_mmq)/sizeof(int);   // 36

    extern __shared__ int smem[];
    int* tile_y = smem;                        // 2 half-buffers (ping-pong)
    int* tile_x = tile_y + 2*GGML_PAD(mmq_x*MMQ_TILE_Y_K, MMQ_NWARPS*MMQ_WARP_SIZE);
    uint8_t* w_stage = (uint8_t*)(tile_x + mmq_y*MMQ_MMA_TILE_X_K);   // 2 x mmq_y x 144B ring

    // W kb-slice staging (the expert kernel's 4B path — 136B IQ4_XS slices).
    auto stage_w = [&](int kb, int buf){
        uint8_t* dst0 = w_stage + (size_t)buf*mmq_y*W_STAGE_STRIDE;
        const int npr = 136/4;
        const int tot = mmq_y*npr;
        for(int c0=0;c0<tot;c0+=MMQ_NWARPS*MMQ_WARP_SIZE){
            int c=c0+threadIdx.y*MMQ_WARP_SIZE+threadIdx.x; if(c>=tot) break;
            int r=c/npr, k=c%npr;
            int rr = nc ? min(r,i_max) : r;
            cp_async4(dst0+(size_t)r*W_STAGE_STRIDE+k*4,
                      Wt+(long)rr*row_bytes+(long)kb*136+k*4);
        }
        asm volatile("cp.async.commit_group;");
    };

    float sum[mmq_x*mmq_y/(MMQ_NWARPS*MMQ_WARP_SIZE)] = {0.0f};
    auto vdot = [&](const int* ty, int k00){
        if(cnt == mmq_x) vec_dot_mma<mmq_x,mmq_y,false>(tile_x, ty, sum, k00, j_max);
        else             vec_dot_mma<mmq_x,mmq_y,true >(tile_x, ty, sum, k00, j_max);
    };
    stage_w(0, 0);
    for(int kb=0;kb<nsblk;kb++){
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        const uint8_t* Ws = w_stage + (size_t)(kb&1)*mmq_y*W_STAGE_STRIDE;
        load_tiles_iq4xs<mmq_y,false>(Ws, tile_x, 0, i_max, W_STAGE_STRIDE);
        if(kb+1<nsblk) stage_w(kb+1, (kb+1)&1);
        constexpr int CH = 4;                                      // ints per 16B chunk
        const int ybuf_stride = GGML_PAD(mmq_x*MMQ_TILE_Y_K, MMQ_NWARPS*MMQ_WARP_SIZE);
        #pragma unroll
        for(int half=0;half<2;half++){
            int blockk = kb*2 + half;
            int* ty = tile_y + half*ybuf_stride;
            for(int c0=0;c0<mmq_x*(sz/CH);c0+=MMQ_NWARPS*MMQ_WARP_SIZE){
                int c=c0+threadIdx.y*MMQ_WARP_SIZE+threadIdx.x;
                if(c>=mmq_x*(sz/CH)) break;
                int token_c = c / (sz/CH), ii = (c % (sz/CH))*CH;
                if(token_c > j_max) continue;      // tail columns discarded at write-back
                cp_async16(&ty[token_c*sz+ii],
                           &Yq[((size_t)blockk*n_tokens + base + token_c)*sz + ii]);
            }
            asm volatile("cp.async.commit_group;");
        }
        asm volatile("cp.async.wait_group 1;");    // W(kb+1)+Y(h0) done; Y(h1) in flight
        __syncthreads();
        vdot(tile_y, 0*MMQ_TILE_NE_K);
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        vdot(tile_y + ybuf_stride, 1*MMQ_TILE_NE_K);
        __syncthreads();
    }
    // write-back: row = token (base+j), col = it*mmq_y + i.
    {
        typedef tile<16,8,int> tC;
        constexpr int g=mmq_get_granularity_device(mmq_x); constexpr int rpw=2*g; constexpr int ntw=rpw/tC::I;
        int i0=(threadIdx.y/ntw)*(ntw*tC::I);
        #pragma unroll
        for(int j0=0;j0<mmq_x;j0+=ntw*tC::J){
            #pragma unroll
            for(int n=0;n<ntw;n++){
                #pragma unroll
                for(int l=0;l<tC::ne;l++){
                    int j=j0+(threadIdx.y%ntw)*tC::J+tC::get_j(l); if(j>j_max) continue;
                    int i=i0+n*tC::I+tC::get_i(l); if(nc&&i>i_max) continue;
                    y[(size_t)(base+j)*out_f + (it*mmq_y+i)] = sum[(j0/tC::J+n)*tC::ne+l];
                }
            }
        }
    }
}

// Dense dynamic-smem bytes (no pair-id array — tokens are the identity).
static size_t mmq_iq4xs_dense_smem(int mmq_x){
    return 2*GGML_PAD((size_t)mmq_x*MMQ_TILE_Y_K, MMQ_NWARPS*MMQ_WARP_SIZE)*sizeof(int)
        + (size_t)MMQ_Y*MMQ_MMA_TILE_X_K*sizeof(int)
        + 2*(size_t)MMQ_Y*W_STAGE_STRIDE;
}

template<int MX>
static int mmq_iq4xs_dense_launch(const uint8_t* W, const int* Yq, float* y,
        int in_f, int out_f, int n_tokens, long row_bytes, cudaStream_t st){
    const int nty = (out_f + MMQ_Y - 1)/MMQ_Y;
    const int ntj = (n_tokens + MX - 1)/MX;
    dim3 grid((unsigned)nty, (unsigned)ntj, 1);
    dim3 block(MMQ_WARP_SIZE, MMQ_NWARPS, 1);
    size_t smem = mmq_iq4xs_dense_smem(MX);
    const bool nc = (out_f % MMQ_Y) != 0;
    cudaError_t ae;
    if(nc){
        ae = cudaFuncSetAttribute(mmq_iq4xs_dense_kernel<MX,true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if(ae != cudaSuccess) return 2000+(int)ae;
        mmq_iq4xs_dense_kernel<MX,true><<<grid,block,smem,st>>>(W,Yq,y,in_f,out_f,n_tokens,row_bytes);
    } else {
        ae = cudaFuncSetAttribute(mmq_iq4xs_dense_kernel<MX,false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if(ae != cudaSuccess) return 2000+(int)ae;
        mmq_iq4xs_dense_kernel<MX,false><<<grid,block,smem,st>>>(W,Yq,y,in_f,out_f,n_tokens,row_bytes);
    }
    cudaError_t e=cudaGetLastError(); return e?1000+(int)e:0;
}

extern "C" {

int memra_mmq_iq_experts(const unsigned long long* table, int proj, int n_expert,
        const int* ex_ids, const int* ex_off, const int* ex_pairs, const int* pair_tok,
        const void* act_scratch, float* y,
        int in_f, int out_f, int n_active, int n_tokens, int qtype, long row_bytes, void* stream){
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    const int* Yq = (const int*)act_scratch;
    // DEVICE SMEM GUARD (2026-08-01, found on the 5090): the round-46 async-movement growth
    // (Y ping-pong + W staging ring) put the 128-token tile at 114.5KB dynamic smem — fine on
    // H100's 228KB opt-in, OVER sm_120a's ~99KB. cudaFuncSetAttribute was failing UNCHECKED and
    // the launch died cudaErrorInvalidValue: q35/g26 prime was broken on the deployment rig.
    // Fall back to the 64-token tile (98.3KB, fits) when the built tile exceeds the device.
    static int smem_optin = -1;
    if(smem_optin < 0){
        int dev = 0; cudaGetDevice(&dev);
        if(cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess)
            smem_optin = 48*1024;
    }
    if(mmq_iq_experts_smem(MMQ_X) > (size_t)smem_optin){
        if(mmq_iq_experts_smem(64) > (size_t)smem_optin) return 3;   // no tile fits this device
        return mmq_iq_experts_launch<64>(table,proj,n_expert,ex_ids,ex_off,ex_pairs,pair_tok,
                                         Yq,y,in_f,out_f,n_active,n_tokens,qtype,row_bytes,st);
    }
    return mmq_iq_experts_launch<MMQ_X>(table,proj,n_expert,ex_ids,ex_off,ex_pairs,pair_tok,
                                        Yq,y,in_f,out_f,n_active,n_tokens,qtype,row_bytes,st);
}

// Dense-trunk IQ4_XS MMQ: y[n_tokens, out_f] = act[n_tokens, in_f] @ W[out_f, in_f]^T.
// Quantizes the f32 activation to D4 q8_1_mmq internally (act_scratch sized by
// memra_mmq_iq_experts_act_bytes — same layout as the expert path). Requires
// in_f % 256 == 0. Same device-smem guard as the expert launcher (128-tile > sm_120a's
// ~99KB opt-in -> 64-token tile). Returns 0, 3 (no tile fits), or 1000+cudaError.
int memra_mmq_iq4xs_dense(const void* W_blocks, const float* act_f32, float* y,
        int in_f, int out_f, int n_tokens, long row_bytes, void* act_scratch, void* stream){
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    {
        int64_t ne10 = in_f, nep = GGML_PAD(ne10, MATRIX_ROW_PADDING);
        int64_t bny = (nep + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1)/(4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
        dim3 blk(CUDA_QUANTIZE_BLOCK_SIZE_MMQ,1,1), grid((unsigned)n_tokens,(unsigned)bny,1);
        quantize_mmq_q8_1_d4_kernel<<<grid,blk,0,st>>>(act_f32, act_scratch, ne10, in_f, nep, n_tokens);
        cudaError_t e=cudaGetLastError(); if(e) return 1000+(int)e;
    }
    static int smem_optin = -1;
    if(smem_optin < 0){
        int dev = 0; cudaGetDevice(&dev);
        if(cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess)
            smem_optin = 48*1024;
    }
    const uint8_t* W = (const uint8_t*)W_blocks;
    const int* Yq = (const int*)act_scratch;
    if(mmq_iq4xs_dense_smem(MMQ_X) > (size_t)smem_optin){
        if(mmq_iq4xs_dense_smem(64) > (size_t)smem_optin) return 3;
        return mmq_iq4xs_dense_launch<64>(W, Yq, y, in_f, out_f, n_tokens, row_bytes, st);
    }
    return mmq_iq4xs_dense_launch<MMQ_X>(W, Yq, y, in_f, out_f, n_tokens, row_bytes, st);
}

} // extern "C"
