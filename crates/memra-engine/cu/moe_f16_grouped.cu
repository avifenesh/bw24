// moe_f16_grouped.cu — per-layer expert dequant to f16 + ONE grouped f16 GEMM per projection
// over the CSR expert->token groups (round 46 arc 2, the "vLLM shape" lever from round 44).
//
// WHY: the expert-segmented MMQ kernel re-dequants the same expert weights per
// (out-tile x token-tile x k-block) — at ~65-145 tokens/expert that is up to 12x redundant
// dequant, 13x scalar instructions per mma, and ~50% tile-padding waste (ncu round 46).
// Here the active experts dequant ONCE per (layer, projection) into an f16 workspace and
// cublasGemmGroupedBatchedEx runs every expert's [m_e x out_f x in_f] GEMM in one call at
// tensor-core f16 rate.
//
// NUMERICS: the f16-mirror class (campaign A / battery-adopted): dequanted weight values
// round to f16, f32 accumulate. NOT byte-identical to the MMQ path — argmax + spec gates
// arbitrate, per-model, like every numeric-class promotion. Experimental door
// MEMRA_MOE_F16G=1 until gated.
//
// Column-major mapping (derivation in the round-46 ledger): per group g,
//   C(m=out_f, n=m_e, k=in_f), opA=T opB=N,
//   A = Wf16[g]   (row-major [out_f][in_f], lda=in_f),
//   B = actf16 + pair_off[g]*in_f (pair-major row-major, ldb=in_f),
//   C = y + pair_off[g]*out_f     (f32 pair-major row-major, ldc=out_f).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cstdio>

#define QT_IQ4_XS 5
#define QT_Q4_0   12

__device__ __forceinline__ float g_half_to_float(uint16_t h){ return __half2float(*reinterpret_cast<const __half*>(&h)); }
__constant__ signed char g_kvalues_iq4nl[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};

// ---- dequant: one expert row slice -> f16. grid.x = out-row, grid.y = active-expert seg.
// 256 threads walk the row's k. Layout out: dst[seg][row][k] row-major (lda = in_f).
static __global__ void dequant_q4_0_f16_kernel(
        const unsigned long long* __restrict__ table, int proj, int n_expert,
        const int* __restrict__ ex_ids, __half* __restrict__ dst,
        int in_f, int out_f, long row_bytes){
    int row = blockIdx.x; int seg = blockIdx.y;
    const uint8_t* W = (const uint8_t*)table[(size_t)proj*n_expert + ex_ids[seg]];
    const uint8_t* r = W + (size_t)row*row_bytes;
    __half* d = dst + ((size_t)seg*out_f + row)*in_f;
    // q4_0: 18B block = f16 d + 16B nibbles for 32 values (lo nibbles vals 0..15, hi 16..31).
    for(int v=threadIdx.x; v<in_f; v+=blockDim.x){
        int blk=v>>5, l=v&31;
        const uint8_t* b = r + (size_t)blk*18;
        float dscale = g_half_to_float(*(const uint16_t*)b);
        uint8_t q = b[2 + (l&15)];
        int nib = (l<16) ? (q & 0xF) : (q >> 4);
        d[v] = __float2half(dscale * (float)(nib - 8));
    }
}

static __global__ void dequant_iq4xs_f16_kernel(
        const unsigned long long* __restrict__ table, int proj, int n_expert,
        const int* __restrict__ ex_ids, __half* __restrict__ dst,
        int in_f, int out_f, long row_bytes){
    int row = blockIdx.x; int seg = blockIdx.y;
    const uint8_t* W = (const uint8_t*)table[(size_t)proj*n_expert + ex_ids[seg]];
    const uint8_t* r = W + (size_t)row*row_bytes;
    __half* d = dst + ((size_t)seg*out_f + row)*in_f;
    // iq4_xs: 136B superblock = f16 d, u16 scales_h, 4B scales_l, 128B nibbles (256 vals).
    for(int v=threadIdx.x; v<in_f; v+=blockDim.x){
        int sb=v>>8, l=v&255;
        const uint8_t* b = r + (size_t)sb*136;
        float d_sb = g_half_to_float(*(const uint16_t*)b);
        uint16_t sh = *(const uint16_t*)(b+2);
        const uint8_t* sl = b+4; const uint8_t* qs = b+8;
        int g = l>>5, lg = l&31;
        int ls = ((sl[g>>1]>>(4*(g&1)))&0xf) | (((sh>>(2*g))&3)<<4);
        const uint8_t* gqs = qs + g*16;
        int code = (lg<16) ? (gqs[lg]&0xf) : (gqs[lg-16]>>4);
        d[v] = __float2half(d_sb * (float)(ls-32) * (float)g_kvalues_iq4nl[code]);
    }
}

// ---- f16 -> f32 elementwise (the grouped GEMM emits f16 C: the grouped API's type
// matrix has no 16F-in/32F-out combo — rc 20015 on H100 cublas 13).
static __global__ void h2f_kernel(const __half* __restrict__ s, float* __restrict__ d, size_t n){
    size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    if(i<n) d[i] = __half2float(s[i]);
}
// f16 -> f32 with the per-row activation scale folded back (row = pair, C row-major).
static __global__ void h2f_rows_scale_kernel(const __half* __restrict__ src, float* __restrict__ d,
                                             const float* __restrict__ s, int ncols, int nrows){
    int r = blockIdx.x; if(r>=nrows) return;
    float sc = s[r];
    const __half* sr = src + (size_t)r*ncols;
    float* dr = d + (size_t)r*ncols;
    for(int c=threadIdx.x;c<ncols;c+=blockDim.x) dr[c] = __half2float(sr[c]) * sc;
}

// ---- activation gather+convert: f32 x[token][in_f] -> f16 B[pair][in_f] via pair_tok,
// with PER-ROW amax normalization (gemma's late-layer activation spikes overflow raw f16
// — the round-46 NaN find; the MMQ path survives on per-32 q8 scales). The row scale
// folds back into the GEMM output (y row *= s[pair]). pair_tok == nullptr => identity.
static __global__ void gather_act_f16_kernel(
        const float* __restrict__ x, const int* __restrict__ pair_tok,
        __half* __restrict__ dst, float* __restrict__ s, int in_f, int n_pairs){
    int p = blockIdx.x; if(p>=n_pairs) return;
    int src = pair_tok ? pair_tok[p] : p;
    const float* xs = x + (size_t)src*in_f;
    __half* d = dst + (size_t)p*in_f;
    __shared__ float red[256];
    float amax = 0.0f;
    for(int v=threadIdx.x; v<in_f; v+=blockDim.x) amax = fmaxf(amax, fabsf(xs[v]));
    red[threadIdx.x] = amax; __syncthreads();
    for(int off=128; off>0; off>>=1){
        if(threadIdx.x<off) red[threadIdx.x]=fmaxf(red[threadIdx.x],red[threadIdx.x+off]);
        __syncthreads();
    }
    amax = red[0];
    float inv = (amax>0.0f) ? 1.0f/amax : 0.0f;
    if(threadIdx.x==0) s[p] = amax;
    for(int v=threadIdx.x; v<in_f; v+=blockDim.x) d[v] = __float2half(xs[v]*inv);
}

extern "C" {

size_t memra_moe_f16g_w_bytes(int n_active, int out_f, int in_f){
    return (size_t)n_active*out_f*in_f*sizeof(__half);
}
size_t memra_moe_f16g_act_bytes(int n_pairs, int in_f){
    return (size_t)n_pairs*in_f*sizeof(__half);
}

int memra_moe_f16g_dequant(const unsigned long long* table, int proj, int n_expert,
        const int* ex_ids, void* w_f16, int in_f, int out_f, int n_active,
        int qtype, long row_bytes, void* stream){
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    dim3 grid((unsigned)out_f, (unsigned)n_active, 1), blk(256,1,1);
    if(qtype==QT_Q4_0)
        dequant_q4_0_f16_kernel<<<grid,blk,0,st>>>(table,proj,n_expert,ex_ids,(__half*)w_f16,in_f,out_f,row_bytes);
    else if(qtype==QT_IQ4_XS)
        dequant_iq4xs_f16_kernel<<<grid,blk,0,st>>>(table,proj,n_expert,ex_ids,(__half*)w_f16,in_f,out_f,row_bytes);
    else return 2;   // unsupported qtype => caller falls back to the MMQ arm
    cudaError_t e=cudaGetLastError(); return e?1000+(int)e:0;
}

int memra_moe_f16g_gather_act(const float* x, const int* pair_tok_or_null, void* act_f16,
        float* row_scale, int in_f, int n_pairs, void* stream){
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    dim3 grid((unsigned)n_pairs,1,1), blk(256,1,1);
    gather_act_f16_kernel<<<grid,blk,0,st>>>(x, pair_tok_or_null, (__half*)act_f16, row_scale,
                                             in_f, n_pairs);
    cudaError_t e=cudaGetLastError(); return e?1000+(int)e:0;
}

int memra_moe_f16g_h2f_scaled(const void* src_f16, float* dst, const float* row_scale,
        int ncols, int nrows, void* stream){
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    h2f_rows_scale_kernel<<<(unsigned)nrows,256,0,st>>>((const __half*)src_f16, dst, row_scale,
                                                        ncols, nrows);
    cudaError_t e=cudaGetLastError(); return e?1000+(int)e:0;
}

int memra_moe_f16g_h2f(const void* src_f16, float* dst, size_t n, void* stream){
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    size_t blocks = (n + 255)/256;
    h2f_kernel<<<(unsigned)blocks,256,0,st>>>((const __half*)src_f16, dst, n);
    cudaError_t e=cudaGetLastError(); return e?1000+(int)e:0;
}

// One grouped GEMM over the CSR groups. ex_off_host = HOST copy of the CSR offsets
// (n_active+1 ints). y = f16 [n_pairs, out_f] (pair-major; caller converts via h2f —
// the grouped API's type matrix has no 16F-in/32F-out combo).
int memra_moe_f16g_gemm(const void* w_f16, const void* act_f16, void* y,
        const int* ex_off_host, int n_active, int in_f, int out_f, void* stream){
    static cublasHandle_t handle = nullptr;
    if(!handle){
        if(cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) return 3;
    }
    cublasSetStream(handle, reinterpret_cast<cudaStream_t>(stream));

    // per-group host arrays (n_active <= a few hundred; stack-scale, heap for safety)
    const int G = n_active;
    static thread_local int cap = 0;
    static thread_local cublasOperation_t *ta=nullptr,*tb=nullptr;
    static thread_local int *ma=nullptr,*na=nullptr,*ka=nullptr,*lda=nullptr,*ldb=nullptr,*ldc=nullptr,*gsz=nullptr;
    static thread_local float *al=nullptr,*be=nullptr;
    static thread_local const void **Aa=nullptr,**Ba=nullptr; static thread_local void **Ca=nullptr;
    if(G>cap){
        delete[] ta; delete[] tb; delete[] ma; delete[] na; delete[] ka; delete[] lda; delete[] ldb;
        delete[] ldc; delete[] gsz; delete[] al; delete[] be; delete[] Aa; delete[] Ba; delete[] Ca;
        cap=G;
        ta=new cublasOperation_t[G]; tb=new cublasOperation_t[G];
        ma=new int[G]; na=new int[G]; ka=new int[G]; lda=new int[G]; ldb=new int[G]; ldc=new int[G];
        gsz=new int[G]; al=new float[G]; be=new float[G];
        Aa=new const void*[G]; Ba=new const void*[G]; Ca=new void*[G];
    }
    int g_used = 0;
    for(int g=0; g<G; g++){
        int lo=ex_off_host[g], hi=ex_off_host[g+1], m_e=hi-lo;
        if(m_e<=0) continue;
        ta[g_used]=CUBLAS_OP_T; tb[g_used]=CUBLAS_OP_N;
        ma[g_used]=out_f; na[g_used]=m_e; ka[g_used]=in_f;
        lda[g_used]=in_f; ldb[g_used]=in_f; ldc[g_used]=out_f;
        al[g_used]=1.0f; be[g_used]=0.0f; gsz[g_used]=1;
        Aa[g_used]=(const uint8_t*)w_f16 + (size_t)g*out_f*in_f*sizeof(__half);
        Ba[g_used]=(const uint8_t*)act_f16 + (size_t)lo*in_f*sizeof(__half);
        Ca[g_used]=(uint8_t*)y + (size_t)lo*out_f*sizeof(__half);
        g_used++;
    }
    if(g_used==0) return 0;
    cublasStatus_t s = cublasGemmGroupedBatchedEx(handle, ta, tb, ma, na, ka,
        al, Aa, CUDA_R_16F, lda, Ba, CUDA_R_16F, ldb,
        be, Ca, CUDA_R_16F, ldc, g_used, gsz, CUBLAS_COMPUTE_32F);
    if(s != CUBLAS_STATUS_SUCCESS) return 20000 + (int)s;
    return 0;
}

} // extern "C"
