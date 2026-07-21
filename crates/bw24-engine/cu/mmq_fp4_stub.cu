// B200 (sm_100a) fail-closed ABI stubs for the sm_120a-only W4A4 MMQ path.
// The normal NVFP4 evaluation path is W4A8 and is implemented by
// mmq_nvfp4_w4a8.cu. These symbols exist only so the shared Rust FFI surface
// links without making an unsupported block-scaled MMA instruction available.
#include <cstddef>

extern "C" size_t bw24_mmq_nvfp4_act_bytes(int, int) {
    return 0;
}

extern "C" int bw24_mmq_nvfp4(
        const void *, const float *, float *, int, int, int, void *, void *, float) {
    return 2901;
}
