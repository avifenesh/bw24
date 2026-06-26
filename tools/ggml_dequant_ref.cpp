// ggml ground-truth dequant oracle.
//
// Given a GGUF path + a tensor NAME (+ how many superblocks to take), this:
//   1. opens the gguf with the ggml C API (gguf_init_from_file)
//   2. locates the tensor by name (gguf_find_tensor), reads its on-disk type,
//      block size, element-block byte size, and data-section offset
//      (gguf_get_tensor_offset + gguf_get_data_offset)
//   3. reads the first <nblocks> blocks of raw quant bytes straight off disk
//   4. dequantizes them with ggml's OWN reference dispatch:
//          ggml_get_type_traits(type)->to_float(block_ptr, out, n_elems)
//      which is exactly dequantize_row_<type> for k-/i-quants and nvfp4.
//   5. writes <prefix>.raw (quant bytes), <prefix>.ref (f32 little-endian),
//      and prints first-8 + min/max/mean/checksum to stdout.
//
// Build (see tools/build_dequant_ref.sh):
//   g++ -O2 -std=c++17 ggml_dequant_ref.cpp -o ggml_dequant_ref \
//     -I<llama>/include -I<llama>/ggml/include \
//     -L<llama>/build/bin -lllama -lggml -lggml-base
//
// Run with LD_LIBRARY_PATH=<llama>/build/bin so the .so's resolve.

#include "ggml.h"
#include "ggml-cpu.h"   // pulls in CPU traits registration
#include "gguf.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>

static void dump(const std::string& path, const void* p, size_t n) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    if (fwrite(p, 1, n, f) != n) { fprintf(stderr, "short write %s\n", path.c_str()); exit(1); }
    fclose(f);
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <gguf> <tensor_name> <nblocks> <out_prefix>\n", argv[0]);
        return 2;
    }
    const char* gguf_path = argv[1];
    const char* tname     = argv[2];
    int64_t     want_blocks = atoll(argv[3]);
    std::string prefix    = argv[4];

    struct gguf_init_params params = { /*no_alloc=*/true, /*ctx=*/nullptr };
    struct gguf_context* ctx = gguf_init_from_file(gguf_path, params);
    if (!ctx) { fprintf(stderr, "gguf load fail: %s\n", gguf_path); return 1; }

    int64_t tid = gguf_find_tensor(ctx, tname);
    if (tid < 0) { fprintf(stderr, "tensor not found: %s\n", tname); return 1; }

    enum ggml_type ty = gguf_get_tensor_type(ctx, tid);
    size_t data_off   = gguf_get_data_offset(ctx);     // absolute file offset of data section
    size_t toff       = gguf_get_tensor_offset(ctx, tid); // relative to data section
    size_t tsize      = gguf_get_tensor_size(ctx, tid);   // total bytes for the tensor

    int64_t blck = ggml_blck_size(ty);   // elems per block
    size_t  tysz = ggml_type_size(ty);   // bytes per block
    int64_t total_blocks = (int64_t)(tsize / tysz);

    int64_t use_blocks = want_blocks <= 0 ? total_blocks
                                          : (want_blocks < total_blocks ? want_blocks : total_blocks);
    int64_t n_elems  = use_blocks * blck;
    size_t  raw_bytes = (size_t)use_blocks * tysz;

    std::vector<uint8_t> raw(raw_bytes);
    FILE* gf = fopen(gguf_path, "rb");
    if (!gf) { fprintf(stderr, "reopen fail\n"); return 1; }
    if (fseek(gf, (long)(data_off + toff), SEEK_SET) != 0) { fprintf(stderr, "seek fail\n"); return 1; }
    if (fread(raw.data(), 1, raw_bytes, gf) != raw_bytes) { fprintf(stderr, "read fail\n"); return 1; }
    fclose(gf);

    // ggml's own reference dequant: to_float == dequantize_row_<type>.
    const struct ggml_type_traits* tr = ggml_get_type_traits(ty);
    if (!tr || !tr->to_float) { fprintf(stderr, "no to_float for type %s\n", ggml_type_name(ty)); return 1; }
    std::vector<float> ref((size_t)n_elems);
    tr->to_float(raw.data(), ref.data(), n_elems);

    // stats
    double mn = 1e300, mx = -1e300, sum = 0.0, sumsq = 0.0;
    for (int64_t i = 0; i < n_elems; ++i) {
        double v = ref[i];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        sum += v;
        sumsq += v * (double)v;
    }
    double mean = sum / (double)n_elems;
    // checksum: sum of (i+1)*v, sensitive to ordering as well as values
    double checksum = 0.0;
    for (int64_t i = 0; i < n_elems; ++i) checksum += (double)(i + 1) * (double)ref[i];

    dump(prefix + ".raw", raw.data(), raw_bytes);
    dump(prefix + ".ref", ref.data(), ref.size() * sizeof(float));

    printf("type=%s tensor=%s blck=%ld type_size=%zu use_blocks=%ld n_elems=%ld raw_bytes=%zu\n",
           ggml_type_name(ty), tname, (long)blck, tysz, (long)use_blocks, (long)n_elems, raw_bytes);
    printf("first8=");
    for (int i = 0; i < 8 && i < n_elems; ++i) printf("%.8g ", ref[i]);
    printf("\nmin=%.8g max=%.8g mean=%.8g sumsq=%.8g checksum=%.10g\n", mn, mx, mean, sumsq, checksum);

    gguf_free(ctx);
    return 0;
}
