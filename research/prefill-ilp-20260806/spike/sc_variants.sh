try() {
  name="$1"; inst="$2"; na="$3"; nb="$4"
  areg=""; for i in $(seq 0 $((na-1))); do areg="$areg,\"r\"(A$i)"; done
  breg=""; for i in $(seq 0 $((nb-1))); do breg="$breg,\"r\"(B$i)"; done
  aph=""; k=4; for i in $(seq 0 $((na-1))); do aph="$aph,%$k"; k=$((k+1)); done
  bph=""; for i in $(seq 0 $((nb-1))); do bph="$bph,%$k"; k=$((k+1)); done
  sa="%$k"; k=$((k+1)); sb="%$k"
  { echo "#include <cstdint>"
    echo "__global__ void kk(const uint32_t*s,float*o){"
    for i in $(seq 0 $((na-1))); do echo "uint32_t A$i=s[$i];"; done
    for i in $(seq 0 $((nb-1))); do echo "uint32_t B$i=s[$((i+8))];"; done
    echo "uint32_t SA=s[20],SB=s[21]; float d0=0,d1=0,d2=0,d3=0;"
    echo "asm volatile(\"$inst {%0,%1,%2,%3},{${aph#,}},{${bph#,}},{%0,%1,%2,%3},{$sa},{0,0},{$sb},{0,0};\""
    echo ": \"+f\"(d0),\"+f\"(d1),\"+f\"(d2),\"+f\"(d3) : ${areg#,}${breg}, \"r\"(SA),\"r\"(SB));"
    echo "o[threadIdx.x]=d0+d1+d2+d3;}"
  } > /tmp/v.cu
  if nvcc -gencode arch=compute_120a,code=sm_120a -O3 -std=c++17 -cubin -o /tmp/v.cubin /tmp/v.cu 2>/tmp/v.err; then
    s=$(cuobjdump -sass /tmp/v.cubin | grep -oE "[OQ]MMA\.[A-Z0-9.]+" | head -1)
    printf "ACCEPTED  %-70s SASS=%s\n" "$name" "$s"
  else
    printf "REJECTED  %-70s %s\n" "$name" "$(grep -m1 error /tmp/v.err | sed 's/.*error *: *//')"
  fi
}
# The 4x door (NVFP4-native): A=e2m1 B=e2m1, UE4M3 per-16 -- what NVFP4 actually stores
try "m16n8k64 mxf4nvf4 4X ue4m3 (e2m1 x e2m1)  [W4A4]" "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3" 4 2
try "m16n8k64 mxf4nvf4 2X ue4m3 (e2m1 x e2m1)  [W4A4]" "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue4m3" 4 2
try "m16n8k64 mxf4nvf4 4X ue8m0 (e2m1 x e2m1)"        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue8m0" 4 2
# The 2x door candidate: keep 8-BIT activations (e4m3) => would preserve W4A8 numerics
try "m16n8k32 mxf8f6f4 1X ue8m0 (e2m1 x e4m3)  [W4A8?]" "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e2m1.e4m3.f32.ue8m0" 4 2
try "m16n8k32 mxf8f6f4 1X ue4m3 (e2m1 x e4m3)  [W4A8?]" "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e2m1.e4m3.f32.ue4m3" 4 2
try "m16n8k32 mxf8f6f4 2X ue4m3 (e2m1 x e4m3)"         "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::2X.f32.e2m1.e4m3.f32.ue4m3" 4 2
try "m16n8k32 mxf8f6f4 4X ue4m3 (e2m1 x e4m3)"         "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::4X.f32.e2m1.e4m3.f32.ue4m3" 4 2
# is there an INTEGER (s8 activation) block-scale form? that would be the closest to today's kernel
try "m16n8k64 mxf4nvf4 4X ue4m3 (e2m1 x e4m3)"         "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e4m3.f32.ue4m3" 4 2
