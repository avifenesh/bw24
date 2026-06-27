// FINAL 4X NVF4 micro-GEMM with the empirically-verified layout. Full 16-row M, per-16-K ue4m3.
#include <cstdio>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#define MMA4(D,A,B,sa,sb) \
  asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,1},{%11},{0,1};" \
    : "+f"(D[0]),"+f"(D[1]),"+f"(D[2]),"+f"(D[3]) \
    : "r"(A[0]),"r"(A[1]),"r"(A[2]),"r"(A[3]),"r"(B[0]),"r"(B[1]),"r"(sa),"r"(sb))
__host__ __device__ float e2m1val(int c){const float t[16]={0,0.5f,1,1.5f,2,3,4,6,-0.f,-0.5f,-1,-1.5f,-2,-3,-4,-6};return t[c&15];}
__host__ __device__ float hw_ue4m3(unsigned char x){int e=(x>>3)&0xF,m=x&7; if(e==0)return ldexpf((float)m/8.0f,-6); return ldexpf(1.0f+(float)m/8.0f,e-7);}
__global__ void k(const unsigned char* Ac,const unsigned char* Bc,const unsigned char* saB,const unsigned char* sbB,float* out){
    int lane=threadIdx.x; unsigned A[4]={0,0,0,0},B[2]={0,0}; float D[4]={0,0,0,0};
    int r0=lane/4,g0=(lane%4)*8;
    for(int n=0;n<8;n++){A[0]|=((unsigned)Ac[r0*64+g0+n])<<(4*n);A[1]|=((unsigned)Ac[(r0+8)*64+g0+n])<<(4*n);
        A[2]|=((unsigned)Ac[r0*64+g0+32+n])<<(4*n);A[3]|=((unsigned)Ac[(r0+8)*64+g0+32+n])<<(4*n);}
    int col=lane/4; for(int n=0;n<8;n++){B[0]|=((unsigned)Bc[((lane%4)*8+n)*8+col])<<(4*n);B[1]|=((unsigned)Bc[((lane%4)*8+32+n)*8+col])<<(4*n);}
    // SFA: lane%4==2 -> row r0 ; lane%4==3 -> row r0+8. 4 bytes = 4 k16 blocks.
    int q=lane&3; unsigned sa=0;
    int sarow=-1; if(q==2)sarow=r0; else if(q==3)sarow=r0+8;
    if(sarow>=0) sa=(unsigned)saB[sarow*4]|((unsigned)saB[sarow*4+1]<<8)|((unsigned)saB[sarow*4+2]<<16)|((unsigned)saB[sarow*4+3]<<24);
    // SFB: lane%4==1 -> col r0.
    unsigned sb=0; if(q==1) sb=(unsigned)sbB[col*4]|((unsigned)sbB[col*4+1]<<8)|((unsigned)sbB[col*4+2]<<16)|((unsigned)sbB[col*4+3]<<24);
    MMA4(D,A,B,sa,sb);
    for(int ci=0;ci<4;ci++){int r=lane/4+(ci/2)*8,c=(lane%4)*2+(ci&1); out[r*8+c]=D[ci];}
}
int main(){srand(31415); unsigned char Ac[1024],Bc[512],saB[64],sbB[32];
    for(int i=0;i<1024;i++)Ac[i]=rand()&15; for(int i=0;i<512;i++)Bc[i]=rand()&15;
    for(int i=0;i<64;i++) saB[i]=((6+(rand()%3))<<3)|(rand()&7);
    for(int i=0;i<32;i++) sbB[i]=((6+(rand()%3))<<3)|(rand()&7);
    float oracle[128];
    for(int r=0;r<16;r++)for(int c=0;c<8;c++){double acc=0; for(int kk=0;kk<64;kk++){int k16=kk/16;
        acc+=(double)(e2m1val(Ac[r*64+kk])*hw_ue4m3(saB[r*4+k16]))*(e2m1val(Bc[kk*8+c])*hw_ue4m3(sbB[c*4+k16]));}oracle[r*8+c]=(float)acc;}
    unsigned char*dA,*dB,*dsa,*dsb;float*dO;cudaMalloc(&dA,1024);cudaMalloc(&dB,512);cudaMalloc(&dsa,64);cudaMalloc(&dsb,32);cudaMalloc(&dO,512);
    cudaMemcpy(dA,Ac,1024,cudaMemcpyHostToDevice);cudaMemcpy(dB,Bc,512,cudaMemcpyHostToDevice);cudaMemcpy(dsa,saB,64,cudaMemcpyHostToDevice);cudaMemcpy(dsb,sbB,32,cudaMemcpyHostToDevice);
    cudaMemset(dO,0,512);k<<<1,32>>>(dA,dB,dsa,dsb,dO);cudaError_t e=cudaDeviceSynchronize();
    if(e){printf("ERR %s\n",cudaGetErrorString(e));return 1;}
    float h[128];cudaMemcpy(h,dO,512,cudaMemcpyDeviceToHost);
    float mr=0;int br=-1; for(int i=0;i<128;i++){float d=fabsf(h[i]-oracle[i])/fmaxf(fabsf(oracle[i]),1e-3f);if(d>mr){mr=d;br=i/8;}}
    printf("FINAL 4X NVF4 micro-GEMM: maxrel=%.4g %s (worst row %d)\n",mr,mr<1e-3?"PASS":"FAIL",br);
    return 0;}
