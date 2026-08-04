// Portable Ada (sm_89) fail-closed ABI stubs for the per-block FP8 MMQ launcher.
// mmq_fp8_blk.cu's tile MMA is .kind::f8f6f4 (sm_100a+); these symbols exist only so the shared
// Rust FFI surface links on sm_89 without making an unsupported MMA instruction available.
#include <cstddef>

extern "C" size_t memra_mmq_nvfp4_f8f4_act_bytes(int, int);

extern "C" size_t memra_mmq_fp8_blk_act_bytes(int in_f, int n_tokens) {
    return memra_mmq_nvfp4_f8f4_act_bytes(in_f, n_tokens);
}

extern "C" int memra_mmq_fp8_blk_scale_rows(int out_f) { return (out_f + 127) / 128; }
extern "C" int memra_mmq_fp8_blk_scale_cols(int in_f)  { return (in_f  + 127) / 128; }

extern "C" int memra_mmq_fp8_blk(
        const void *, const float *, const float *, float *, int, int, int, void *, void *, float) {
    return 2904;
}

extern "C" int memra_fp8_blk_count_nan(const void *, size_t, unsigned int *, void *) {
    return 2905;
}
