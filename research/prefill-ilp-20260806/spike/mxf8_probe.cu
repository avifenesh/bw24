#include <cstdint>
// mxf8f6f4 m16n8k32: e2m1/e3m2/e2m3 operands are PADDED into 8-bit containers.
// A = 16x32 x 8bit / 32 lanes = 16B = 4 regs. B = 8x32 x 8bit / 32 lanes = 8B = 2 regs.
__global__ void k(const uint32_t* a, const uint32_t* b, const uint32_t* sfa, const uint32_t* sfb, float* out) {
    uint32_t A0=a[0],A1=a[1],A2=a[2],A3=a[3];
    uint32_t B0=b[0],B1=b[1];
    uint32_t SFA=sfa[0], SFB=sfb[0];
    float d0=0.f,d1=0.f,d2=0.f,d3=0.f;
    asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e2m1.e4m3.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {0,0}, {%11}, {0,0};"
      : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
      : "r"(A0),"r"(A1),"r"(A2),"r"(A3),"r"(B0),"r"(B1),"r"(SFA),"r"(SFB));
    out[threadIdx.x]=d0+d1+d2+d3;
}
