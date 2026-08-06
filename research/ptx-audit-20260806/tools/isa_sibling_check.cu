#include <cstdio>
__global__ void k(unsigned* s, float* o){
  unsigned a[8]; float d[4];
  for(int i=0;i<8;i++) a[i]=s[i]; for(int i=0;i<4;i++) d[i]=0.f;
#if F==1  // bf16 f32-acc at k32? (does a deeper-K equal-math sibling exist?)
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
    :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(a[4]),"r"(a[5]));
#elif F==2  // f16 f32-acc at k32?
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
    :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(a[4]),"r"(a[5]));
#elif F==3  // bf16 f32-acc via kind::f8f6f4-style block_scale? (bf16 has no block_scale kind)
  asm volatile("mma.sync.aligned.m16n8k16.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.bf16.bf16.f32.ue8m0 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,0},{%11},{0,0};"
    :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(a[4]),"r"(a[5]),"r"(a[6]),"r"(a[7]));
#elif F==4  // int8 s32-acc at k64? (deeper than k32)
  int di[4]={0,0,0,0};
  asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
    :"+r"(di[0]),"+r"(di[1]),"+r"(di[2]),"+r"(di[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(a[4]),"r"(a[5]));
  d[0]=(float)di[0];
#elif F==5  // f8f6f4 block_scale at k64? (deeper than k32)
  asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,0},{%11},{0,0};"
    :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(a[4]),"r"(a[5]),"r"(a[6]),"r"(a[7]));
#elif F==6  // f16 f16-OUT accumulate at k32?
  unsigned dh[2]={0,0};
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.f16.f16.f16 {%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1};"
    :"+r"(dh[0]),"+r"(dh[1]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(a[4]),"r"(a[5]));
  d[0]=(float)dh[0];
#elif F==7  // mxf4nvf4 at k128? (deeper than k64)
  asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k128.row.col.f32.e2m1.e2m1.f32.ue4m3 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3}, %10,{0,0}, %11,{0,0};"
    :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(a[4]),"r"(a[5]),"r"(a[6]),"r"(a[7]));
#endif
  o[0]=d[0]+d[1]+d[2]+d[3];
}
int main(){return 0;}
