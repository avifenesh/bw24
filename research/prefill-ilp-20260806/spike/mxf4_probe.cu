// PROBE 2 (a): does sm_120a accept mma.sync.aligned.m16n8k64.kind::mxf4nvf4.block_scale?
// Feasibility spike ONLY — never linked. Compile-only receipt.
#include <cstdint>
__global__ void k(const uint32_t* __restrict__ a, const uint32_t* __restrict__ b,
                  const uint32_t* __restrict__ sfa, const uint32_t* __restrict__ sfb,
                  float* __restrict__ out) {
    uint32_t A0=a[0],A1=a[1],A2=a[2],A3=a[3];
    uint32_t B0=b[0],B1=b[1];
    uint32_t SFA=sfa[0], SFB=sfb[0];
    float d0=0.f,d1=0.f,d2=0.f,d3=0.f;
    asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {0,0}, {%11}, {0,0};"
      : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
      : "r"(A0),"r"(A1),"r"(A2),"r"(A3),"r"(B0),"r"(B1),"r"(SFA),"r"(SFB));
    out[threadIdx.x] = d0+d1+d2+d3;
}
