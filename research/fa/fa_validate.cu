// fa_validate.cu — end-to-end validation of flash_attn.cu vs the sdpa_naive_f32 oracle.
//
// Includes the REAL kernels from crates/bw24-engine/cu/flash_attn.cu and the oracle
// sdpa_naive_f32 (copied verbatim from crates/bw24-engine/cu/kernels.cu:99-146), runs
// both on identical random Q/K/V at head_dim=256, GQA 4, causal, for several (T,T_kv)
// shapes including T=1 (which exercises fa_decode_f32 + combine), and reports max|abs|/
// max|rel| vs the oracle. bf16-mma tolerance ~1e-2.
//
// Build: nvcc -gencode arch=compute_120a,code=sm_120a -O3 -o fa_validate \
//        research/fa/fa_validate.cu
// (flash_attn.cu is #included; the oracle is defined here to avoid a 2nd TU.)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// Pull in the real FA kernels (fa_prefill_f32, fa_decode_f32, fa_decode_combine_f32).
#include "../../crates/bw24-engine/cu/flash_attn.cu"

// ---- oracle: sdpa_naive_f32 (verbatim from kernels.cu:99-146) ----
extern "C" __global__ void sdpa_oracle_f32(const float* __restrict__ Q, const float* __restrict__ K,
                                           const float* __restrict__ V, float* __restrict__ O,
                                           int head_dim, int n_head, int n_head_kv, int T, int T_kv,
                                           float scale, int causal) {
    int head = blockIdx.x;
    int qt = blockIdx.y;
    if (head >= n_head || qt >= T) return;
    int kv_head = head / (n_head / n_head_kv);
    int tid = threadIdx.x;
    extern __shared__ float scores[];
    const float* q = Q + ((size_t)qt * n_head + head) * head_dim;
    int q_pos = (T_kv - T) + qt;
    for (int t = tid; t < T_kv; t += blockDim.x) {
        const float* k = K + ((size_t)t * n_head_kv + kv_head) * head_dim;
        float acc = 0.0f;
        for (int d = 0; d < head_dim; d++) acc += q[d] * k[d];
        acc *= scale;
        if (causal && t > q_pos) acc = -1e30f;
        scores[t] = acc;
    }
    __syncthreads();
    __shared__ float red[1];
    if (tid == 0) {
        float mx = -1e30f;
        for (int t = 0; t < T_kv; t++) mx = fmaxf(mx, scores[t]);
        float sum = 0.0f;
        for (int t = 0; t < T_kv; t++) { float e = expf(scores[t] - mx); scores[t] = e; sum += e; }
        float inv = 1.0f / sum;
        for (int t = 0; t < T_kv; t++) scores[t] *= inv;
        red[0] = 0.0f;
    }
    __syncthreads();
    float* o = O + ((size_t)qt * n_head + head) * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < T_kv; t++) {
            const float* v = V + ((size_t)t * n_head_kv + kv_head) * head_dim;
            acc += scores[t] * v[d];
        }
        o[d] = acc;
    }
}

static float frand() { return (float)((rand() % 2001) - 1000) / 1000.0f; }

// run one (T,T_kv) case; returns 0 on PASS.
static int run_case(int T, int T_kv) {
    const int HD = 256, NH = 16, NHKV = 4;
    const float scale = 1.0f / 16.0f;
    const int causal = 1;
    srand(1000 + T * 131 + T_kv);

    size_t qn = (size_t)T * NH * HD, kn = (size_t)T_kv * NHKV * HD, on = qn;
    float *Qh = (float*)malloc(qn*4), *Kh = (float*)malloc(kn*4), *Vh = (float*)malloc(kn*4);
    for (size_t i=0;i<qn;i++) Qh[i]=frand();
    for (size_t i=0;i<kn;i++) Kh[i]=frand();
    for (size_t i=0;i<kn;i++) Vh[i]=frand();

    float *dQ,*dK,*dV,*dOo,*dOf;
    cudaMalloc(&dQ,qn*4); cudaMalloc(&dK,kn*4); cudaMalloc(&dV,kn*4);
    cudaMalloc(&dOo,on*4); cudaMalloc(&dOf,on*4);
    cudaMemcpy(dQ,Qh,qn*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dK,Kh,kn*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dV,Vh,kn*4,cudaMemcpyHostToDevice);

    // ---- oracle ----
    {
        dim3 grid(NH, T, 1);
        sdpa_oracle_f32<<<grid, 128, (size_t)T_kv*4>>>(dQ,dK,dV,dOo,HD,NH,NHKV,T,T_kv,scale,causal);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) { printf("  oracle launch err: %s\n", cudaGetErrorString(e)); return 1; }
    }

    // ---- flash ----
    if (T == 1) {
        // decode path: split-K + combine
        int n_splits = (T_kv >= 256) ? 8 : (T_kv >= 64 ? 4 : 1);
        float *pO,*pM,*pL;
        cudaMalloc(&pO,(size_t)NH*n_splits*HD*4);
        cudaMalloc(&pM,(size_t)NH*n_splits*4);
        cudaMalloc(&pL,(size_t)NH*n_splits*4);
        dim3 grid(NH, n_splits, 1);
        size_t dsmem = (size_t)(HD + 256) * 4;   // sq[head_dim] + red[block]
        fa_decode_f32<<<grid, 256, dsmem>>>(dQ,dK,dV,pO,pM,pL,HD,NH,NHKV,T_kv,scale,n_splits);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) { printf("  decode launch err: %s\n", cudaGetErrorString(e)); return 1; }
        fa_decode_combine_f32<<<dim3(NH,1,1), HD>>>(pO,pM,pL,dOf,HD,NH,n_splits);
        e = cudaDeviceSynchronize();
        if (e != cudaSuccess) { printf("  combine launch err: %s\n", cudaGetErrorString(e)); return 1; }
        cudaFree(pO); cudaFree(pM); cudaFree(pL);
    } else {
        // prefill path
        int q_tiles = (T + 15) / 16;
        dim3 grid(q_tiles, NH, 1);
        // smem bytes: sQ 16*HD*2 + sK BK*HD*2 + sV BK*HD*2 + sP 16*BK*2 + sO 16*HD*4 + sS 16*BK*4 + sM 16*4 + sL 16*4
        const int BK_ = 64;
        size_t smem = (size_t)(16*HD)*2 + (size_t)(BK_*HD)*2*2 + (size_t)(16*BK_)*2
                    + (size_t)(16*HD)*4 + (size_t)(16*BK_)*4 + (size_t)16*4*2;
        cudaFuncSetAttribute(fa_prefill_f32, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        fa_prefill_f32<<<grid, 32, smem>>>(dQ,dK,dV,dOf,HD,NH,NHKV,T,T_kv,scale,causal);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) { printf("  prefill launch err (smem=%zu): %s\n", smem, cudaGetErrorString(e)); return 1; }
    }

    float *Oo=(float*)malloc(on*4), *Of=(float*)malloc(on*4);
    cudaMemcpy(Oo,dOo,on*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(Of,dOf,on*4,cudaMemcpyDeviceToHost);
    double maxabs=0,maxrel=0; int bad=0; int badr=0,badc=0;
    for (size_t i=0;i<on;i++) {
        double a=fabs((double)Oo[i]-(double)Of[i]);
        double rel=a/(fabs((double)Oo[i])+1e-4);
        if(a>maxabs)maxabs=a; if(rel>maxrel)maxrel=rel;
        if(a>2e-2 && rel>5e-2){ if(bad<4){badr=(int)(i/HD); badc=(int)(i%HD); printf("    bad O[%d] oracle=%.4f flash=%.4f abs=%.4f rel=%.4f\n",(int)i,Oo[i],Of[i],a,rel);} bad++; }
    }
    int pass = (bad==0);
    printf("  T=%-4d T_kv=%-5d  maxabs=%.3e maxrel=%.3e bad=%-5d -> %s\n",
           T, T_kv, maxabs, maxrel, bad, pass?"PASS":"FAIL");
    (void)badr;(void)badc;
    free(Qh);free(Kh);free(Vh);free(Oo);free(Of);
    cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dOo);cudaFree(dOf);
    return pass?0:1;
}

int main() {
    int fails = 0;
    printf("FlashAttention vs sdpa_naive_f32 oracle (head_dim=256, GQA 4, causal, bf16-mma tol)\n");
    printf("== prefill cases ==\n");
    fails += run_case(16, 16);     // exact one tile, square
    fails += run_case(16, 64);     // one q-tile, one full KV tile
    fails += run_case(16, 100);    // ragged KV (padding path)
    fails += run_case(32, 32);     // two q-tiles
    fails += run_case(48, 200);    // multi q-tile, multi KV tile, ragged
    fails += run_case(7,  7);      // ragged q (nq<16), ragged KV
    fails += run_case(64, 512);    // longer context
    printf("== decode (T=1) cases ==\n");
    fails += run_case(1, 32);      // single short, 1 split
    fails += run_case(1, 64);      // 4 splits
    fails += run_case(1, 200);     // 4 splits, ragged
    fails += run_case(1, 512);     // 8 splits
    printf("\n=== %s ===\n", fails==0?"ALL PASS":"FAILURES");
    return fails;
}
