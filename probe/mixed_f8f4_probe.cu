// Probe 0 for the prefill mxf8f6f4 arc (research/prefill-mxf8f6f4-design.md):
// (1) does the MIXED-operand form assemble+execute: A=e4m3 activations, B=e2m1 weights,
//     kind::mxf8f6f4 block_scale ue8m0?
// (2) issue-rate of the mixed form vs the e4m3xe4m3 form (is mixed a slower pipe?)
// (3) CORRECTNESS: one-warp m16n8k32 tile vs a host f64 reference, real fragment layouts,
//     scales = 2^0 — verifies the e2m1-in-8-bit-container encoding and the scale semantics.
// Build: nvcc -arch=compute_120a -code=sm_120a -O3 probe/mixed_f8f4_probe.cu -o probe/mixed_f8f4_probe
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

#define MMA_MIXED(c, a0,a1,a2,a3, b0,b1, sa,sb) \
  asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e2m1.f32.ue8m0 " \
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,1},{%11},{0,1};" \
    :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]) \
    :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(sa),"r"(sb))

// ---------- (2) rate kernel: mixed e4m3 x e2m1 ----------
template<int IT> __global__ void k_rate_mixed(float* s){
  unsigned a[4]={1,2,3,4},b[2]={5,6},sa=1,sb=1; float c0[4]={0},c1[4]={0};
  #pragma unroll 1
  for(int i=0;i<IT;i++){
    MMA_MIXED(c0, a[0],a[1],a[2],a[3], b[0],b[1], sa,sb);
    MMA_MIXED(c1, a[0],a[1],a[2],a[3], b[0],b[1], sa,sb);
  }
  if(c0[0]==-1.f)s[0]=c0[0]+c1[0];
}

// ---------- (3) correctness kernel: one warp, real fragments ----------
// m16n8k32, 8-bit containers. Thread t (0..31), canonical layouts:
//   A (16x32 row-major): a0 -> row t/4,   k = 4*(t%4)+{0..3}
//                        a1 -> row t/4+8, same k
//                        a2 -> row t/4,   k+16 ; a3 -> row t/4+8, k+16
//   B (32x8, col-major fragment): b0 -> k = 4*(t%4)+{0..3}, col t/4 ; b1 -> k+16
//   D (16x8): c0,c1 -> row t/4, col 2*(t%4)+{0,1} ; c2,c3 -> row t/4+8
__global__ void k_correct(const unsigned char* A, const unsigned char* B, float* D,
                          unsigned sa_byte, unsigned sb_byte){
  int t = threadIdx.x;
  unsigned a[4], b[2];
  unsigned char ab[4][4], bb[2][4];
  for(int h=0; h<2; ++h)         // k-halves
    for(int r=0; r<2; ++r)       // row groups
      for(int i=0; i<4; ++i){
        int row = t/4 + 8*r, k = 4*(t%4) + i + 16*h;
        ab[2*h+r][i] = A[row*32 + k];
      }
  for(int h=0; h<2; ++h)
    for(int i=0; i<4; ++i){
      int k = 4*(t%4) + i + 16*h, col = t/4;
      bb[h][i] = B[col*32 + k];  // B stored col-major: [col][k]
    }
  memcpy(a, ab, 16); memcpy(b, bb, 8);
  unsigned sa = sa_byte * 0x01010101u, sb = sb_byte * 0x01010101u;
  float c[4] = {0,0,0,0};
  MMA_MIXED(c, a[0],a[1],a[2],a[3], b[0],b[1], sa,sb);
  for(int r=0; r<2; ++r)
    for(int i=0; i<2; ++i)
      D[(t/4 + 8*r)*8 + 2*(t%4) + i] = c[2*r+i];
}

// ---------- host reference ----------
static double e4m3_val(unsigned char v){
  int s=v>>7, e=(v>>3)&0xF, m=v&7;
  if(e==0xF && m==7) return 0.0/0.0;                     // nan
  double x = e ? ldexp(1.0 + m/8.0, e-7) : ldexp(m/8.0, -6);
  return s ? -x : x;
}
static double e2m1_val(unsigned char v){                  // low 4 bits used
  static const double tab[8] = {0,0.5,1,1.5,2,3,4,6};
  double x = tab[v&7]; return (v&8) ? -x : x;
}
int main(){
  // ---- rate ----
  const int IT=4096; int bl=82*4, th=256;
  float* s; cudaMalloc(&s,4);
  k_rate_mixed<IT><<<bl,th>>>(s);
  cudaError_t err = cudaDeviceSynchronize();
  if(err != cudaSuccess){ printf("MIXED FORM FAILED TO EXECUTE: %s\n", cudaGetErrorString(err)); return 1; }
  cudaEvent_t ea,eb; cudaEventCreate(&ea); cudaEventCreate(&eb);
  int reps=10; cudaEventRecord(ea);
  for(int r=0;r<reps;r++) k_rate_mixed<IT><<<bl,th>>>(s);
  cudaEventRecord(eb); cudaDeviceSynchronize();
  float ms; cudaEventElapsedTime(&ms,ea,eb);
  double w = (double)(bl*th/32);
  double tf = w*(2.0*16*8*32*2)*IT*reps/(ms/1e3)/1e12;
  printf("mixed e4m3 x e2m1 mxf8f6f4 block-scale: %.0f TFLOP/s (e4m3xe4m3 ref: 381)\n", tf);

  // ---- correctness ----
  unsigned char hA[16*32], hB[8*32];
  // varied-but-exact values: e4m3 patterns from a small exact set, e2m1 cycles all 15 codes
  const unsigned char e4set[6] = {0x38, 0xB8, 0x40, 0x30, 0x44, 0x00}; // 1,-1,2,0.5,3,0
  for(int i=0;i<16*32;i++) hA[i] = e4set[(i*7+i/32)%6];
  for(int i=0;i<8*32;i++)  hB[i] = (unsigned char)((i*5+i/32)%15);     // e2m1 codes 0..14 low-4bit
  unsigned char *dA,*dB; float *dD;
  cudaMalloc(&dA,sizeof hA); cudaMalloc(&dB,sizeof hB); cudaMalloc(&dD,16*8*4);
  cudaMemcpy(dA,hA,sizeof hA,cudaMemcpyHostToDevice);
  cudaMemcpy(dB,hB,sizeof hB,cudaMemcpyHostToDevice);
  k_correct<<<1,32>>>(dA,dB,dD,127,127);                 // ue8m0 127 = 2^0
  err = cudaDeviceSynchronize();
  if(err != cudaSuccess){ printf("CORRECTNESS KERNEL FAILED: %s\n", cudaGetErrorString(err)); return 1; }
  float hD[16*8]; cudaMemcpy(hD,dD,sizeof hD,cudaMemcpyDeviceToHost);
  double maxd=0; int bad=0;
  for(int i=0;i<16;i++) for(int j=0;j<8;j++){
    double ref=0;
    for(int k=0;k<32;k++) ref += e4m3_val(hA[i*32+k]) * e2m1_val(hB[j*32+k]);
    double d = fabs(ref - (double)hD[i*8+j]);
    if(d > maxd) maxd = d;
    if(d > 1e-6) bad++;
  }
  printf("correctness vs f64 ref: maxdiff=%.3e  bad=%d/128  %s\n", maxd, bad, bad? "FAIL":"OK");
  return bad ? 1 : 0;
}
