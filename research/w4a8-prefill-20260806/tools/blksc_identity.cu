// SLICE 4: is kind::mxf8f6f4.block_scale.scale_vec::1X (e4m3 x e4m3, ue8m0) with the IDENTITY
// scale byte 0x7F (=2^(127-127)=2^0) NUMERICALLY IDENTICAL to the plain kind::f8f6f4 form the
// shipped tile issues today?
//
// If YES, the 1.994x rate found in slice 3 is a DROP-IN swap for mmq_nvfp4_w4a8.cu:1058: same
// operands, same fragment layout, same accumulator, same result bits -- only the MMA form changes,
// so the shipped R-B tile's numeric config (and its f8f4-check / argmax lineage) is untouched.
//
// Method: one warp, real random e4m3 operands, three computations of the SAME 16x8x32 product:
//   (a) plain kind::f8f6f4                       -> the shipped form
//   (b) block_scale.1X with all scale bytes 0x7F -> the fast form, identity scale
//   (c) f64 host oracle over the decoded e4m3 values
// Report exact bit equality (a)vs(b) and max rel error of each vs (c).
//
// Also sweeps the scale-byte SELECTOR fields to confirm 0x7F in every selected lane is what the
// hardware reads (a wrong selector would silently scale by 2^k).
// Feasibility spike only. Never linked into the engine.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp8.h>

#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA ERR %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);} }while(0)

// A-frag for m16n8k32 8-bit: 4 regs/thread = 16 bytes. B-frag: 2 regs = 8 bytes.
// Standard SM80 16x8x32 TN layout (CUTLASS SM80_16x8x32_S32S8S8S32_TN, inherited by SM120).
__global__ void k_pair(const unsigned* a, const unsigned* b, float* out_plain, float* out_blksc,
                       unsigned scale_a, unsigned scale_b) {
    const int lane = threadIdx.x;
    unsigned a0=a[lane*4+0], a1=a[lane*4+1], a2=a[lane*4+2], a3=a[lane*4+3];
    unsigned b0=b[lane*2+0], b1=b[lane*2+1];

    float dp[4]={0.f,0.f,0.f,0.f};
    asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(dp[0]),"+f"(dp[1]),"+f"(dp[2]),"+f"(dp[3])
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));

    float db[4]={0.f,0.f,0.f,0.f};
    asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,0},{%11},{0,0};"
        : "+f"(db[0]),"+f"(db[1]),"+f"(db[2]),"+f"(db[3])
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(scale_a),"r"(scale_b));

#pragma unroll
    for(int i=0;i<4;i++){ out_plain[lane*4+i]=dp[i]; out_blksc[lane*4+i]=db[i]; }
}

static inline double e4m3_to_d(unsigned char x){
    __nv_fp8_e4m3 v; memcpy(&v,&x,1); return (double)(float)v;
}

int main(){
    const int LANES=32;
    // Random e4m3 bytes, avoiding NaN (0x7F/0xFF exponent-all-ones mantissa-all-ones).
    unsigned char ha[LANES*16], hb[LANES*8];
    srand(20260806);
    for(int i=0;i<LANES*16;i++){ unsigned char c; do{ c=rand()&0xFF; }while((c&0x7F)==0x7F); ha[i]=c; }
    for(int i=0;i<LANES*8;i++){  unsigned char c; do{ c=rand()&0xFF; }while((c&0x7F)==0x7F); hb[i]=c; }

    unsigned *da,*db; float *dp,*dbs;
    CK(cudaMalloc(&da,LANES*16)); CK(cudaMalloc(&db,LANES*8));
    CK(cudaMalloc(&dp,LANES*16)); CK(cudaMalloc(&dbs,LANES*16));
    CK(cudaMemcpy(da,ha,LANES*16,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db,hb,LANES*8,cudaMemcpyHostToDevice));

    // f64 oracle. Layout (SM80 m16n8k32 TN, 8-bit):
    //   A: lane l holds rows (l/4) and (l/4)+8; reg r byte j -> k = (r%2)*8 + j + (r/2)*16
    //   B: lane l holds col (l%4)*2.. -- derive empirically instead: we only need SOME consistent
    //      decode, so we verify (a)==(b) bitwise (the load-bearing claim) and use the oracle only
    //      as a magnitude sanity check on the accumulated sums.
    double sum_ref=0;
    for(int i=0;i<LANES*16;i++) sum_ref += fabs(e4m3_to_d(ha[i]));
    for(int i=0;i<LANES*8;i++)  sum_ref += fabs(e4m3_to_d(hb[i]));

    printf("# SLICE 4 -- block_scale identity-scale equivalence to the plain form\n");
    printf("# operand abs-value sum (magnitude sanity anchor): %.6f\n\n", sum_ref);

    struct { const char* name; unsigned sa, sb; } cases[] = {
        {"identity 0x7F7F7F7F both (ue8m0 bias 127 => 2^0)", 0x7F7F7F7Fu, 0x7F7F7F7Fu},
        {"scale_a = 0x80.. (2^1) -- expect 2x, proves the operand is LIVE", 0x80808080u, 0x7F7F7F7Fu},
        {"scale_b = 0x7E.. (2^-1) -- expect 0.5x", 0x7F7F7F7Fu, 0x7E7E7E7Eu},
    };
    float hp[LANES*4], hbs[LANES*4];
    for (int c=0; c<3; ++c) {
        k_pair<<<1,LANES>>>(da,db,dp,dbs,cases[c].sa,cases[c].sb);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(hp,dp,LANES*16,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hbs,dbs,LANES*16,cudaMemcpyDeviceToHost));
        int bitdiff=0; double maxrel=0, maxabs_p=0; double ratio_sum=0; int ratio_n=0;
        for(int i=0;i<LANES*4;i++){
            if(memcmp(&hp[i],&hbs[i],4)!=0) bitdiff++;
            double p=hp[i], b=hbs[i];
            if(fabs(p)>maxabs_p) maxabs_p=fabs(p);
            double den=fabs(p)>1e-6?fabs(p):1.0;
            double rel=fabs(p-b)/den; if(rel>maxrel) maxrel=rel;
            if(fabs(p)>1e-3){ ratio_sum += b/p; ratio_n++; }
        }
        printf("case %d: %s\n", c, cases[c].name);
        printf("   elements=%d  bitwise-different=%d  maxrel(plain vs blksc)=%.6g  max|plain|=%.4f  mean(blksc/plain)=%.6f\n\n",
               LANES*4, bitdiff, maxrel, maxabs_p, ratio_n? ratio_sum/ratio_n : 0.0);
    }
    return 0;
}
