// swapab_bench.cu — SWAP_AB orientation microbench for mmq_iq_experts' mma.sync m16n8k16.s8 form.
//
// Question (research lever #8): vLLM/DeepGEMM SWAP_AB puts the small token dim on the tensor-core
// N slot on sm_90 (wgmma: M fixed 64, N 8-granular). memra's mmq_iq_experts already has tokens on
// the mma.sync N=8 axis — the 128-token padding is a TILE-LOOP artifact, not an operand-shape one.
// This bench compares three arms at the q35 gate/up shape (out_f=512, k=2048, 252 expert groups,
// m tokens/group in {16,32,65,96,128}), all consuming per-32 float scales on both operands:
//
//   arm A "cur128"   — current orientation (out-rows on A/M via ldmatrix, tokens on B/N),
//                      full 128x128 tile mma regardless of m (mimics shipped kernel post-inc4:
//                      dead-column gathers skipped, dead-column mma computed + discarded).
//   arm C "cur-exit" — current orientation + 8-granular j-strip early exit in vec_dot
//                      (each warp skips (n-rowblock x j0-strip) quanta whose 8-token block is
//                      entirely dead). The 5-line-change alternative to a port.
//   arm B "swap16"   — SWAP_AB: tokens on A/M (ldmatrix 16-row fragments, 16-granular row skip),
//                      out-cols on B/N (dense 128). The port under probe.
//
// Tile machinery (tile<>, ldmatrix, mma, 84/36 strides, 8 warps, 128x128, per-32 scales applied
// as dB*(C0*dA0+C1*dA1)) is copied verbatim from cu/mmq_iq_experts.cu. Data path is identical
// across arms (plain smem loads, no cp.async): the variable under test is ORIENTATION only.
// What this bench does NOT model: the real kernel's cp.async W-staging ring + gather (token-
// invariant W movement dominates the real (4,252) kernel — see analysis.md).
//
// Build: nvcc -arch=sm_90a -O3 -std=c++17 --expt-relaxed-constexpr -Xptxas=-v
// Run:   CUDA_VISIBLE_DEVICES=4 ./swapab_bench

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>

#define CK(x) do{ cudaError_t e_=(x); if(e_){ printf("CUDA ERR %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1);} }while(0)

#define MMQ_TILE_NE_K 32
#define MMQ_MMA_TILE_X_K 84
#define MMQ_TILE_Y_K 36
#define NW 8
#define TILE 128

// ======================= mma.cuh machinery — verbatim mmq_iq_experts.cu =======================
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
    static __device__ __forceinline__ void mma(tile<16,8,int>&D,const tile<16,4,int>&A,const tile<8,4,int>&B){
        asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
            :"+r"(D.x[0]),"+r"(D.x[1]),"+r"(D.x[2]),"+r"(D.x[3]):"r"(A.x[0]),"r"(A.x[1]),"r"(B.x[0]));
    }
}
using namespace ggml_cuda_mma;
static constexpr __device__ int mmq_get_granularity_device(int mmq_x){ return mmq_x>=48?16:8; }

// ======================= vec_dot variants =======================
// verbatim vec_dot_mma (mmq_x=128, mmq_y=128), JEXIT adds the 8-granular token-strip skip,
// ROWSKIP is the swapped orientation's 16-granular token-row skip (tokens on A/M).
template<int mmq_x, int mmq_y, bool JEXIT>
static __device__ __forceinline__ void vec_dot_cur(const int* x, const int* y, float* sum, int k00, int j_max){
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
        if(JEXIT && (j0+joff) > j_max) continue;   // warp's 8-token block entirely dead
        #pragma unroll
        for(int k01=0;k01<MMQ_TILE_NE_K;k01+=8){
            tB B[2]; float dB[tC::ne/2];
            load_generic(B[0], y_qs+j0*MMQ_TILE_Y_K+(k01+0), MMQ_TILE_Y_K);
            load_generic(B[1], y_qs+j0*MMQ_TILE_Y_K+(k01+tB::J), MMQ_TILE_Y_K);
            #pragma unroll
            for(int l=0;l<tC::ne/2;l++){ int j=j0+tC::get_j(l); dB[l]=y_df[j*MMQ_TILE_Y_K+k01/8]; }
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

// SWAP_AB orientation: x side = TOKENS (A/M, ldmatrix), y side = OUT-COLS (B/N, dense).
// 16-granular token skip: fragment n is dead when its 16-row block starts past t_max.
template<int mmq_x, int mmq_y>
static __device__ __forceinline__ void vec_dot_swap(const int* x, const int* y, float* sum, int k00, int t_max){
    typedef tile<16,4,int> tA; typedef tile<16,8,int> tA8; typedef tile<8,4,int> tB; typedef tile<16,8,int> tC;
    constexpr int g=mmq_get_granularity_device(mmq_x); constexpr int rpw=2*g; constexpr int ntx=rpw/tC::I;
    y += (threadIdx.y%ntx)*(tC::J*MMQ_TILE_Y_K);
    const int* x_qs=x; const float* x_df=(const float*)x_qs + MMQ_TILE_NE_K*2;
    const int* y_qs=(const int*)y+4; const float* y_df=(const float*)y;
    const int i0=(threadIdx.y/ntx)*(ntx*tA::I);
    bool live[ntx];
    #pragma unroll
    for(int n=0;n<ntx;n++) live[n] = (i0+n*tA::I) <= t_max;
    if(!live[0]) return;                          // whole warp row-group dead (fragments ordered)
    tA A[ntx][8]; float dA[ntx][tC::ne/2][8];
    #pragma unroll
    for(int n=0;n<ntx;n++){
        if(!live[n]) continue;
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
        #pragma unroll
        for(int k01=0;k01<MMQ_TILE_NE_K;k01+=8){
            tB B[2]; float dB[tC::ne/2];
            load_generic(B[0], y_qs+j0*MMQ_TILE_Y_K+(k01+0), MMQ_TILE_Y_K);
            load_generic(B[1], y_qs+j0*MMQ_TILE_Y_K+(k01+tB::J), MMQ_TILE_Y_K);
            #pragma unroll
            for(int l=0;l<tC::ne/2;l++){ int j=j0+tC::get_j(l); dB[l]=y_df[j*MMQ_TILE_Y_K+k01/8]; }
            #pragma unroll
            for(int n=0;n<ntx;n++){
                if(!live[n]) continue;
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

// ======================= bench kernels =======================
// Shapes: OUTF x m x KDIM per group, GROUPS groups. grid=(OUTF/128, GROUPS), block=(32,8).
// Global layouts: W int8 [G][OUTF][KDIM] + Ws float [G][OUTF][KDIM/32];
//                 acts int8 [128][KDIM] + As float [128][KDIM/32] (shared across groups);
//                 out float [G][128][OUTF] (row=token, col=out) — identical for all arms.
static constexpr int GROUPS = 252;
static constexpr int OUTF   = 512;
static constexpr int KDIM   = 2048;
static constexpr int NSB    = KDIM/256;     // 8 superblocks, 2 vec_dot halves each
static constexpr int KS32   = KDIM/32;      // 64 per-32 scale groups per row

// current orientation (arm A: JEXIT=false, arm C: JEXIT=true)
template<bool JEXIT>
__global__ void __launch_bounds__(32*NW,1) k_current(const int8_t* __restrict__ W, const float* __restrict__ Ws,
        const int8_t* __restrict__ Aq, const float* __restrict__ As, float* __restrict__ out, int m){
    const int g = blockIdx.y, bx = blockIdx.x;
    const int* Wg  = (const int*)(W + ((size_t)g*OUTF + (size_t)bx*TILE)*KDIM);   // int cols
    const float* Wsg = Ws + ((size_t)g*OUTF + (size_t)bx*TILE)*KS32;
    const int* Aqi = (const int*)Aq;
    extern __shared__ int smem[];
    int* x_tile = smem;                                  // 128 x 84 (weights)
    int* ty0 = x_tile + TILE*MMQ_MMA_TILE_X_K;           // 128 x 36 (acts, half 0)
    int* ty1 = ty0 + TILE*MMQ_TILE_Y_K;                  // half 1
    const int tid = threadIdx.y*32 + threadIdx.x;
    const int j_max = m-1;
    float sum[TILE*TILE/(32*NW)] = {0.0f};
    for(int kb=0;kb<NSB;kb++){
        // X = weights: 128 rows x (64 data ints + 16 scale floats)
        for(int c0=tid;c0<TILE*80;c0+=32*NW){
            int i=c0/80, q=c0%80;
            if(q<64) x_tile[i*MMQ_MMA_TILE_X_K+q] = Wg[(size_t)i*(KDIM/4) + kb*64 + q];
            else { int s=q-64; ((float*)(x_tile+i*MMQ_MMA_TILE_X_K+64))[s] = Wsg[(size_t)i*KS32 + kb*8 + (s>>1)]; }
        }
        // Y = acts, 2 halves x 128 cols x 36 ints; dead cols (j>=m) skipped (mimics shipped inc4)
        #pragma unroll
        for(int h=0;h<2;h++){
            int* ty = h? ty1: ty0;
            for(int c0=tid;c0<TILE*MMQ_TILE_Y_K;c0+=32*NW){
                int j=c0/MMQ_TILE_Y_K, ii=c0%MMQ_TILE_Y_K;
                if(j>=m) continue;
                if(ii<4) ((float*)(ty+j*MMQ_TILE_Y_K))[ii] = As[(size_t)j*KS32 + kb*8 + h*4 + ii];
                else     ty[j*MMQ_TILE_Y_K+ii] = Aqi[(size_t)j*(KDIM/4) + (kb*2+h)*32 + (ii-4)];
            }
        }
        __syncthreads();
        vec_dot_cur<TILE,TILE,JEXIT>(x_tile, ty0, sum, 0,               j_max);
        vec_dot_cur<TILE,TILE,JEXIT>(x_tile, ty1, sum, MMQ_TILE_NE_K,   j_max);
        __syncthreads();
    }
    {   // writeback: row = token j, col = out i (verbatim pattern)
        typedef tile<16,8,int> tC;
        constexpr int gsz=mmq_get_granularity_device(TILE); constexpr int rpw=2*gsz; constexpr int ntx=rpw/tC::I;
        int i0=(threadIdx.y/ntx)*(ntx*tC::I);
        #pragma unroll
        for(int j0=0;j0<TILE;j0+=ntx*tC::J){
            #pragma unroll
            for(int n=0;n<ntx;n++){
                #pragma unroll
                for(int l=0;l<tC::ne;l++){
                    int j=j0+(threadIdx.y%ntx)*tC::J+tC::get_j(l); if(j>j_max) continue;
                    int i=i0+n*tC::I+tC::get_i(l);
                    out[((size_t)g*TILE + j)*OUTF + bx*TILE + i] = sum[(j0/tC::J+n)*tC::ne+l];
                }
            }
        }
    }
}

// SWAP_AB orientation (arm B): tokens on A/M (16-granular skip), out-cols on B/N (dense)
__global__ void __launch_bounds__(32*NW,1) k_swap(const int8_t* __restrict__ W, const float* __restrict__ Ws,
        const int8_t* __restrict__ Aq, const float* __restrict__ As, float* __restrict__ out, int m){
    const int g = blockIdx.y, bx = blockIdx.x;                        // bx = out-col tile
    const int* Wg  = (const int*)(W + ((size_t)g*OUTF + (size_t)bx*TILE)*KDIM);
    const float* Wsg = Ws + ((size_t)g*OUTF + (size_t)bx*TILE)*KS32;
    const int* Aqi = (const int*)Aq;
    extern __shared__ int smem[];
    int* x_tile = smem;                                  // 128 x 84 (tokens)
    int* ty0 = x_tile + TILE*MMQ_MMA_TILE_X_K;           // 128 x 36 (weights, half 0)
    int* ty1 = ty0 + TILE*MMQ_TILE_Y_K;
    const int tid = threadIdx.y*32 + threadIdx.x;
    const int t_max = m-1;
    const int mp = (m+15)&~15;                           // 16-granular padded token rows
    float sum[TILE*TILE/(32*NW)] = {0.0f};
    for(int kb=0;kb<NSB;kb++){
        // X = tokens: only mp rows staged (pad rows zeroed)
        for(int c0=tid;c0<mp*80;c0+=32*NW){
            int t=c0/80, q=c0%80;
            if(q<64) x_tile[t*MMQ_MMA_TILE_X_K+q] = (t<m) ? Aqi[(size_t)t*(KDIM/4) + kb*64 + q] : 0;
            else { int s=q-64; ((float*)(x_tile+t*MMQ_MMA_TILE_X_K+64))[s] = (t<m) ? As[(size_t)t*KS32 + kb*8 + (s>>1)] : 0.0f; }
        }
        // Y = weights: 2 halves x 128 out-cols x 36 ints (per-32 weight scales in y_df slots)
        #pragma unroll
        for(int h=0;h<2;h++){
            int* ty = h? ty1: ty0;
            for(int c0=tid;c0<TILE*MMQ_TILE_Y_K;c0+=32*NW){
                int j=c0/MMQ_TILE_Y_K, ii=c0%MMQ_TILE_Y_K;
                if(ii<4) ((float*)(ty+j*MMQ_TILE_Y_K))[ii] = Wsg[(size_t)j*KS32 + kb*8 + h*4 + ii];
                else     ty[j*MMQ_TILE_Y_K+ii] = Wg[(size_t)j*(KDIM/4) + (kb*2+h)*32 + (ii-4)];
            }
        }
        __syncthreads();
        vec_dot_swap<TILE,TILE>(x_tile, ty0, sum, 0,             t_max);
        vec_dot_swap<TILE,TILE>(x_tile, ty1, sum, MMQ_TILE_NE_K, t_max);
        __syncthreads();
    }
    {   // writeback transposed: row = token i, col = out j
        typedef tile<16,8,int> tC;
        constexpr int gsz=mmq_get_granularity_device(TILE); constexpr int rpw=2*gsz; constexpr int ntx=rpw/tC::I;
        int i0=(threadIdx.y/ntx)*(ntx*tC::I);
        #pragma unroll
        for(int j0=0;j0<TILE;j0+=ntx*tC::J){
            #pragma unroll
            for(int n=0;n<ntx;n++){
                #pragma unroll
                for(int l=0;l<tC::ne;l++){
                    int i=i0+n*tC::I+tC::get_i(l); if(i>t_max) continue;
                    int j=j0+(threadIdx.y%ntx)*tC::J+tC::get_j(l);
                    out[((size_t)g*TILE + i)*OUTF + bx*TILE + j] = sum[(j0/tC::J+n)*tC::ne+l];
                }
            }
        }
    }
}

// fp32 reference: per-32 grouped exactly like the mma path (int dot per 32, then ws*as scale),
// same k ascending accumulation order.
__global__ void k_ref(const int8_t* __restrict__ W, const float* __restrict__ Ws,
        const int8_t* __restrict__ Aq, const float* __restrict__ As, float* __restrict__ out, int m){
    const int g = blockIdx.y;
    const int o = blockIdx.x*blockDim.x + threadIdx.x; if(o>=OUTF) return;
    const int8_t* Wr = W + ((size_t)g*OUTF + o)*KDIM;
    const float* Wsr = Ws + ((size_t)g*OUTF + o)*KS32;
    for(int t=0;t<m;t++){
        float acc=0.0f;
        for(int q=0;q<KS32;q++){
            int s=0;
            #pragma unroll
            for(int r=0;r<32;r++) s += (int)Wr[q*32+r] * (int)Aq[(size_t)t*KDIM + q*32+r];
            acc += Wsr[q]*As[(size_t)t*KS32+q]*(float)s;
        }
        out[((size_t)g*TILE + t)*OUTF + o] = acc;
    }
}

// ======================= host =======================
static float median5(float* v,int n){ std::sort(v,v+n); return n%2? v[n/2] : 0.5f*(v[n/2-1]+v[n/2]); }

int main(){
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop,0));
    printf("# device: %s, SMs=%d\n", prop.name, prop.multiProcessorCount);

    const size_t wN = (size_t)GROUPS*OUTF*KDIM, wsN = (size_t)GROUPS*OUTF*KS32;
    const size_t aN = (size_t)TILE*KDIM, asN = (size_t)TILE*KS32;
    const size_t oN = (size_t)GROUPS*TILE*OUTF;
    int8_t *dW, *dA; float *dWs, *dAs, *dOut;
    CK(cudaMalloc(&dW, wN)); CK(cudaMalloc(&dWs, wsN*4));
    CK(cudaMalloc(&dA, aN)); CK(cudaMalloc(&dAs, asN*4));
    CK(cudaMalloc(&dOut, oN*4));

    // host init: int8 in [-8,7], scales in (0.001, 0.021) — magnitudes tame, per-32 exact int dots
    srand(24);
    { std::vector<int8_t> h(wN); for(size_t i=0;i<wN;i++) h[i]=(int8_t)((rand()%16)-8); CK(cudaMemcpy(dW,h.data(),wN,cudaMemcpyHostToDevice)); }
    { std::vector<float> h(wsN); for(size_t i=0;i<wsN;i++) h[i]=0.001f+0.02f*(rand()/(float)RAND_MAX); CK(cudaMemcpy(dWs,h.data(),wsN*4,cudaMemcpyHostToDevice)); }
    { std::vector<int8_t> h(aN); for(size_t i=0;i<aN;i++) h[i]=(int8_t)((rand()%16)-8); CK(cudaMemcpy(dA,h.data(),aN,cudaMemcpyHostToDevice)); }
    { std::vector<float> h(asN); for(size_t i=0;i<asN;i++) h[i]=0.001f+0.02f*(rand()/(float)RAND_MAX); CK(cudaMemcpy(dAs,h.data(),asN*4,cudaMemcpyHostToDevice)); }

    const size_t smem = (size_t)TILE*MMQ_MMA_TILE_X_K*4 + 2*(size_t)TILE*MMQ_TILE_Y_K*4;   // 79,872B
    CK(cudaFuncSetAttribute(k_current<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    CK(cudaFuncSetAttribute(k_current<true>,  cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    CK(cudaFuncSetAttribute(k_swap,           cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    printf("# smem/CTA = %zu B, grid=(%d,%d), block=(32,%d)\n", smem, OUTF/TILE, GROUPS, NW);

    dim3 grid(OUTF/TILE, GROUPS), block(32, NW);
    std::vector<float> href(oN), harm(oN);
    const int ms[5] = {16,32,65,96,128};
    const char* names[3] = {"A cur128 ", "B swap16 ", "C cur-exit"};
    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

    for(int mi=0;mi<5;mi++){
        int m=ms[mi];
        // ---- correctness ----
        CK(cudaMemset(dOut,0,oN*4));
        k_ref<<<dim3(2,GROUPS),256>>>(dW,dWs,dA,dAs,dOut,m); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(href.data(),dOut,oN*4,cudaMemcpyDeviceToHost));
        for(int arm=0;arm<3;arm++){
            CK(cudaMemset(dOut,0,oN*4));
            if(arm==0) k_current<false><<<grid,block,smem>>>(dW,dWs,dA,dAs,dOut,m);
            if(arm==1) k_swap          <<<grid,block,smem>>>(dW,dWs,dA,dAs,dOut,m);
            if(arm==2) k_current<true> <<<grid,block,smem>>>(dW,dWs,dA,dAs,dOut,m);
            CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(harm.data(),dOut,oN*4,cudaMemcpyDeviceToHost));
            double maxrel=0, maxabs=0; size_t bad=0;
            for(int g=0;g<GROUPS;g++) for(int t=0;t<m;t++) for(int o=0;o<OUTF;o++){
                size_t idx=((size_t)g*TILE+t)*OUTF+o;
                double r=href[idx], v=harm[idx];
                double ad=fabs(v-r), rel=ad/(fabs(r)+1e-6);
                if(rel>maxrel) maxrel=rel;
                if(ad>maxabs) maxabs=ad;
                if(ad>1e-3 && rel>1e-3) bad++;   // rounding-only diffs sit near zero refs; a
                                                 // layout bug is O(1) abs error on most elements
            }
            printf("m=%3d  %s correctness: maxabs=%.3e maxrel=%.3e  bad(abs&rel>1e-3)=%zu  %s\n",
                   m, names[arm], maxabs, maxrel, bad, bad? "FAIL":"PASS");
            if(bad){ printf("ABORT: correctness failure\n"); return 1; }
        }
        // ---- timing: 5 reps, arms interleaved within each rep; 20 launches/measure, 3 warmup ----
        float med[3]; float t[3][5];
        for(int rep=0;rep<5;rep++){
            for(int arm=0;arm<3;arm++){
                for(int w=0;w<3;w++){
                    if(arm==0) k_current<false><<<grid,block,smem>>>(dW,dWs,dA,dAs,dOut,m);
                    if(arm==1) k_swap          <<<grid,block,smem>>>(dW,dWs,dA,dAs,dOut,m);
                    if(arm==2) k_current<true> <<<grid,block,smem>>>(dW,dWs,dA,dAs,dOut,m);
                }
                CK(cudaDeviceSynchronize());
                CK(cudaEventRecord(e0));
                for(int it=0;it<20;it++){
                    if(arm==0) k_current<false><<<grid,block,smem>>>(dW,dWs,dA,dAs,dOut,m);
                    if(arm==1) k_swap          <<<grid,block,smem>>>(dW,dWs,dA,dAs,dOut,m);
                    if(arm==2) k_current<true> <<<grid,block,smem>>>(dW,dWs,dA,dAs,dOut,m);
                }
                CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
                float ms_el; CK(cudaEventElapsedTime(&ms_el,e0,e1));
                t[arm][rep] = ms_el*1000.0f/20.0f;   // us per launch
            }
        }
        for(int arm=0;arm<3;arm++){
            float v[5]; memcpy(v,t[arm],sizeof(v)); med[arm]=median5(v,5);
            printf("m=%3d  %s us/launch: reps=[%.1f %.1f %.1f %.1f %.1f]  median=%.1f\n",
                   m, names[arm], t[arm][0],t[arm][1],t[arm][2],t[arm][3],t[arm][4], med[arm]);
        }
        printf("m=%3d  ratios: swap/cur128=%.3f  exit/cur128=%.3f  swap/exit=%.3f  (ideal mma-work: %.3f / %.3f)\n\n",
               m, med[1]/med[0], med[2]/med[0], med[1]/med[2],
               ((m+15)&~15)/128.0f, ((m+7)&~7)/128.0f);
    }
    return 0;
}
